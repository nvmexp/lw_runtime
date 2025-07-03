#include "THCGeneral.h"
#include "TH.h"
#include "THCAllocator.h"
#include "THCCachingHostAllocator.h"
#include "THCStream.h"
#include "THCThreadLocal.h"
#include "THCTensorRandom.h"
#include <stdlib.h>
#include <stdint.h>

/* Size of scratch space available in global memory per each SM + stream */
#define MIN_GLOBAL_SCRATCH_SPACE_PER_SM_STREAM 4 * sizeof(float)

/* Minimum amount of scratch space per device. Total scratch memory per
 * device is either this amount, or the # of SMs * the space per SM defined
 * above, whichever is greater.*/
#define MIN_GLOBAL_SCRATCH_SPACE_PER_DEVICE 32768 * sizeof(float)

/* Maximum number of P2P connections (if there are more than 9 then P2P is
 * enabled in groups of 8). */
#define THC_LWDA_MAX_PEER_SIZE 8

THCLwdaResourcesPerDevice* THCState_getDeviceResourcePtr(
  THCState *state, int device);

THCState* THCState_alloc(void)
{
  THCState* state = (THCState*) malloc(sizeof(THCState));
  memset(state, 0, sizeof(THCState));
  return state;
}

void THCState_free(THCState* state)
{
  free(state);
}

static lwdaError_t lwdaMallocWrapper(void* ctx, void** devPtr, size_t size, lwdaStream_t stream)
{
  return lwdaMalloc(devPtr, size);
}

static lwdaError_t lwdaFreeWrapper(void* ctx, void* devPtr)
{
  return lwdaFree(devPtr);
}

static THCDeviceAllocator defaultDeviceAllocator = {
  &lwdaMallocWrapper,
  NULL,
  &lwdaFreeWrapper,
  NULL,
  NULL,
  NULL
};

void THLwdaInit(THCState* state)
{
  if (!state->lwdaDeviceAllocator) {
    state->lwdaDeviceAllocator = &defaultDeviceAllocator;
  }
  if (!state->lwdaHostAllocator) {
    state->lwdaHostAllocator = &THLwdaHostAllocator;
  }
  if (!state->lwdaUVAAllocator) {
    state->lwdaUVAAllocator = &THLWVAAllocator;
  }

  int numDevices = 0;
  THLwdaCheck(lwdaGetDeviceCount(&numDevices));
  state->numDevices = numDevices;

  int device = 0;
  THLwdaCheck(lwdaGetDevice(&device));

  /* Start in the default stream on the current device */
  state->lwrrentStreams = (THCThreadLocal*) malloc(numDevices * sizeof(THCThreadLocal));
  for (int i = 0; i < numDevices; ++i) {
    state->lwrrentStreams[i] = THCThreadLocal_alloc();
  }
  state->lwrrentPerDeviceBlasHandle = THCThreadLocal_alloc();
  state->lwrrentPerDeviceSparseHandle = THCThreadLocal_alloc();

  state->resourcesPerDevice = (THCLwdaResourcesPerDevice*)
    malloc(numDevices * sizeof(THCLwdaResourcesPerDevice));
  memset(state->resourcesPerDevice, 0, numDevices * sizeof(THCLwdaResourcesPerDevice));

  state->deviceProperties =
    (struct lwdaDeviceProp*)malloc(numDevices * sizeof(struct lwdaDeviceProp));

  state->rngState = (THCRNGState*)malloc(sizeof(THCRNGState));
  THCRandom_init(state, numDevices, device);

  // By default, all direct p2p kernel access (besides copy) is disallowed,
  // since direct access without knowing whether or not a certain operation
  // should be cross-GPU leads to synchronization errors. The user can choose
  // to disable this functionality, however.
  state->p2pKernelAccessEnabled = 0;

  // p2pAccessEnabled records if p2p copies are allowed between pairs of
  // devices. Values include "1" (copy allowed), "0" (copy not allowed), and
  // "-1" (unknown).
  // Lwrrently the max number of gpus in P2P group is 8, so if there are more
  // we enable P2P in groups of 8
  state->p2pAccessEnabled = (int**) malloc(sizeof(int*) * numDevices);
  for (int i = 0; i < numDevices; ++i) {
    state->p2pAccessEnabled[i] = (int*) malloc(sizeof(int) * numDevices);
    for (int j = 0; j < numDevices; ++j)
      if (i == j)
        state->p2pAccessEnabled[i][j] = 1;
      else if (j / THC_LWDA_MAX_PEER_SIZE != i / THC_LWDA_MAX_PEER_SIZE)
        state->p2pAccessEnabled[i][j] = 0;
      else
        state->p2pAccessEnabled[i][j] = -1;
  }

  for (int i = 0; i < numDevices; ++i) {
    THCLwdaResourcesPerDevice* res = THCState_getDeviceResourcePtr(state, i);
    THLwdaCheck(lwdaSetDevice(i));
    THLwdaCheck(lwdaGetDeviceProperties(&state->deviceProperties[i], i));

    // Allocate space for the default stream
    res->streams = (THCStream**) malloc(sizeof(THCStream*));
    res->streams[0] = THCStream_defaultStream(i);

    /* The scratch space that we want to have available per each device is
       based on the number of SMs available per device. We guarantee a
       minimum of 128kb of space per device, but to future-proof against
       future architectures that may have huge #s of SMs, we guarantee that
       we have at least 16 bytes for each SM. */
    int numSM = state->deviceProperties[i].multiProcessorCount;
    size_t sizePerStream =
      MIN_GLOBAL_SCRATCH_SPACE_PER_DEVICE >= numSM * MIN_GLOBAL_SCRATCH_SPACE_PER_SM_STREAM ?
      MIN_GLOBAL_SCRATCH_SPACE_PER_DEVICE :
      numSM * MIN_GLOBAL_SCRATCH_SPACE_PER_SM_STREAM;
    res->scratchSpacePerStream = sizePerStream;
  }

  /* Restore to previous device */
  THLwdaCheck(lwdaSetDevice(device));

  // Unlike LWCA streams, there is no NULL lwBLAS handle. The default THC
  // lwBLAS handle is the first user BLAS handle. Note that the actual BLAS
  // handles are created lazily.
  state->numUserBlasHandles = 1;
  state->numUserSparseHandles = 1;

  state->heapSoftmax = 3e8; // 300MB, adjusted upward dynamically
  state->heapDelta = 0;
}

void THLwdaShutdown(THCState* state)
{
  THCRandom_shutdown(state);

  free(state->rngState);
  free(state->deviceProperties);

  int deviceCount = 0;
  int prevDev = -1;
  THLwdaCheck(lwdaGetDevice(&prevDev));
  THLwdaCheck(lwdaGetDeviceCount(&deviceCount));

  /* cleanup p2p access state */
  for (int dev = 0; dev < deviceCount; ++dev) {
    free(state->p2pAccessEnabled[dev]);
  }
  free(state->p2pAccessEnabled);

  /* cleanup per-device state */
  for (int dev = 0; dev < deviceCount; ++dev) {
    THLwdaCheck(lwdaSetDevice(dev));
    THCLwdaResourcesPerDevice* res = &(state->resourcesPerDevice[dev]);
    /* Free all streams */
    for (int i = 0; i <= state->numUserStreams; ++i) {
      THCStream_free(res->streams[i]);
    }
    /* Free user defined BLAS handles */
    for (int i = 0; i < res->numBlasHandles; ++i) {
      THLwblasCheck(lwblasDestroy(res->blasHandles[i]));
    }
    /* Free user defined sparse handles */
    for (int i = 0; i < res->numSparseHandles; ++i) {
      THLwsparseCheck(lwsparseDestroy(res->sparseHandles[i]));
    }
    /* Free per-stream scratch space; starts at 0 because there is space for
       the default stream as well*/
    if (res->devScratchSpacePerStream) {
      for (int stream = 0; stream <= state->numUserStreams; ++stream) {
        THLwdaCheck(THLwdaFree(state, res->devScratchSpacePerStream[stream]));
      }
    }

    free(res->streams);
    free(res->blasHandles);
    free(res->sparseHandles);
    free(res->devScratchSpacePerStream);
    THCStream_free((THCStream*)THCThreadLocal_get(state->lwrrentStreams[dev]));
    THCThreadLocal_free(state->lwrrentStreams[dev]);
  }
  free(state->resourcesPerDevice);
  if (state->lwdaDeviceAllocator->emptyCache) {
    state->lwdaDeviceAllocator->emptyCache(state->lwdaDeviceAllocator->state);
  }
  if (state->lwdaHostAllocator == &THCCachingHostAllocator) {
    THCCachingHostAllocator_emptyCache();
  }
  free(state->lwrrentStreams);
  THCThreadLocal_free(state->lwrrentPerDeviceBlasHandle);

  THLwdaCheck(lwdaSetDevice(prevDev));
}

int THCState_getPeerToPeerAccess(THCState* state, int dev, int devToAccess)
{
  if (dev < 0 || dev >= state->numDevices) {
    THError("%d is not a device", dev);
  }
  if (devToAccess < 0 || devToAccess >= state->numDevices) {
    THError("%d is not a device", devToAccess);
  }
  if (state->p2pAccessEnabled[dev][devToAccess] == -1) {
    int prevDev = 0;
    THLwdaCheck(lwdaGetDevice(&prevDev));
    THLwdaCheck(lwdaSetDevice(dev));

    int access = 0;
    THLwdaCheck(lwdaDeviceCanAccessPeer(&access, dev, devToAccess));
    if (access) {
      lwdaError_t err = lwdaDeviceEnablePeerAccess(devToAccess, 0);
      if (err == lwdaErrorPeerAccessAlreadyEnabled) {
        // ignore and clear the error if access was already enabled
        lwdaGetLastError();
      } else {
        THLwdaCheck(err);
      }
      state->p2pAccessEnabled[dev][devToAccess] = 1;
    } else {
      state->p2pAccessEnabled[dev][devToAccess] = 0;
    }

    THLwdaCheck(lwdaSetDevice(prevDev));
  }
  return state->p2pAccessEnabled[dev][devToAccess];
}

void THCState_setPeerToPeerAccess(THCState* state, int dev, int devToAccess,
                                  int enable)
{
  /* This will perform device bounds checking for us */
  int prevEnabled = THCState_getPeerToPeerAccess(state, dev, devToAccess);

  if (enable != prevEnabled) {
    /* If we're attempting to enable p2p access but p2p access isn't */
    /* supported, throw an error */
    if (enable) {
      int access = 0;
      THLwdaCheck(lwdaDeviceCanAccessPeer(&access, dev, devToAccess));

      if (!access) {
        THError("p2p access not supported for %d accessing %d",
                dev, devToAccess);
      }
    }

    state->p2pAccessEnabled[dev][devToAccess] = enable;

    int prevDev = 0;
    THLwdaCheck(lwdaGetDevice(&prevDev));
    THLwdaCheck(lwdaSetDevice(dev));

    /* This should be in sync with the current access state */
    if (enable) {
      THLwdaCheck(lwdaDeviceEnablePeerAccess(devToAccess, 0));
    } else {
      THLwdaCheck(lwdaDeviceDisablePeerAccess(devToAccess));
    }

    THLwdaCheck(lwdaSetDevice(prevDev));
  }
}

int THCState_getKernelPeerToPeerAccessEnabled(THCState* state) {
  return state->p2pKernelAccessEnabled;
}

void THCState_setKernelPeerToPeerAccessEnabled(THCState* state, int val) {
  state->p2pKernelAccessEnabled = val;
}

struct lwdaDeviceProp* THCState_getLwrrentDeviceProperties(THCState* state)
{
  int lwrDev = -1;
  THLwdaCheck(lwdaGetDevice(&lwrDev));

  return &(state->deviceProperties[lwrDev]);
}

struct THCRNGState* THCState_getRngState(THCState *state)
{
  return state->rngState;
}

THAllocator* THCState_getLwdaHostAllocator(THCState* state)
{
  return state->lwdaHostAllocator;
}

THAllocator* THCState_getLwdaUVAAllocator(THCState* state)
{
  return state->lwdaUVAAllocator;
}

THC_API THCDeviceAllocator* THCState_getDeviceAllocator(THCState* state)
{
  return state->lwdaDeviceAllocator;
}

void THCState_setDeviceAllocator(THCState* state, THCDeviceAllocator* allocator)
{
  state->lwdaDeviceAllocator = allocator;
}

int THCState_isCachingAllocatorEnabled(THCState* state) {
  return state->lwdaHostAllocator == &THCCachingHostAllocator;
}

int THCState_getNumDevices(THCState *state)
{
  return state->numDevices;
}

static void THCState_initializeScratchSpace(THCState* state, int dev)
{
  THCLwdaResourcesPerDevice* res = THCState_getDeviceResourcePtr(state, dev);
  if (res->devScratchSpacePerStream) {
    return;
  }
  size_t size = (state->numUserStreams + 1) * sizeof(void*);
  void** scratch = (void**)malloc(size);
  for (int i = 0; i <= state->numUserStreams; ++i) {
    THLwdaCheck(THLwdaMalloc(state, &scratch[i], res->scratchSpacePerStream));
  }
  res->devScratchSpacePerStream = scratch;
}

void THCState_reserveStreams(THCState* state, int numStreams, int nonBlocking)
{
  if (numStreams <= state->numUserStreams)
  {
    return;
  }

  int prevDev = -1;
  THLwdaCheck(lwdaGetDevice(&prevDev));

  /* Otherwise, we have to allocate a new set of streams and stream data */
  for (int dev = 0; dev < state->numDevices; ++dev) {
    THLwdaCheck(lwdaSetDevice(dev));
    THCLwdaResourcesPerDevice* res = THCState_getDeviceResourcePtr(state, dev);

    /* +1 for the default stream as well */
    THCStream** newStreams = realloc(res->streams, (numStreams + 1) * sizeof(THCStream*));
    THAssert(newStreams);

    THCState_initializeScratchSpace(state, dev);
    void** newScratchSpace = realloc(res->devScratchSpacePerStream, (numStreams + 1) * sizeof(void*));
    THAssert(newScratchSpace);

    /* Allocate new stream resources */
    size_t scratchSpaceSize = THCState_getDeviceScratchSpaceSize(state, dev);
    unsigned int flags =
      nonBlocking ? lwdaStreamNonBlocking : lwdaStreamDefault;

    for (int stream = state->numUserStreams + 1; stream <= numStreams; ++stream) {
      newStreams[stream] = THCStream_new(flags);
      newScratchSpace[stream] = NULL;
      THLwdaCheck(THLwdaMalloc(state, &newScratchSpace[stream], scratchSpaceSize));
    }

    res->streams = newStreams;
    res->devScratchSpacePerStream = newScratchSpace;
  }

  state->numUserStreams = numStreams;

  THLwdaCheck(lwdaSetDevice(prevDev));
}

void THCState_reserveDeviceBlasHandles(THCState* state, int device, int numBlasHandles)
{
  int prevDev = -1;
  THCLwdaResourcesPerDevice* res = THCState_getDeviceResourcePtr(state, device);
  if (numBlasHandles <= res->numBlasHandles) {
    return;
  }

  THLwdaCheck(lwdaGetDevice(&prevDev));
  THLwdaCheck(lwdaSetDevice(device));

  size_t size = numBlasHandles * sizeof(lwblasHandle_t);
  lwblasHandle_t* handles = (lwblasHandle_t*) realloc(res->blasHandles, size);
  for (int i = res->numBlasHandles; i < numBlasHandles; ++i) {
    handles[i] = NULL;
    THLwblasCheck(lwblasCreate(&handles[i]));
  }
  res->blasHandles = handles;
  res->numBlasHandles = numBlasHandles;

  THLwdaCheck(lwdaSetDevice(prevDev));
}

void THCState_reserveDeviceSparseHandles(THCState* state, int device, int numSparseHandles)
{
  int prevDev = -1;
  THCLwdaResourcesPerDevice* res = THCState_getDeviceResourcePtr(state, device);
  if (numSparseHandles <= res->numSparseHandles) {
    return;
  }

  THLwdaCheck(lwdaGetDevice(&prevDev));
  THLwdaCheck(lwdaSetDevice(device));

  size_t size = numSparseHandles * sizeof(lwsparseHandle_t);
  lwsparseHandle_t* handles = (lwsparseHandle_t*) realloc(res->sparseHandles, size);
  for (int i = res->numSparseHandles; i < numSparseHandles; ++i) {
    handles[i] = NULL;
    THLwsparseCheck(lwsparseCreate(&handles[i]));
  }
  res->sparseHandles = handles;
  res->numSparseHandles = numSparseHandles;

  THLwdaCheck(lwdaSetDevice(prevDev));
}

void THCState_reserveBlasHandles(THCState* state, int numBlasHandles)
{
  // lwBLAS handles are created lazily from THCState_getDeviceBlasHandle
  // to avoid initializing unused devices
  if (numBlasHandles > state->numUserBlasHandles)
  {
    state->numUserBlasHandles = numBlasHandles;
  }
}

void THCState_reserveSparseHandles(THCState* state, int numSparseHandles)
{
  // lwBLAS handles are created lazily from THCState_getDeviceSparseHandle
  // to avoid initializing unused devices
  if (numSparseHandles > state->numUserSparseHandles)
  {
    state->numUserSparseHandles = numSparseHandles;
  }
}

int THCState_getNumStreams(THCState* state)
{
  return state->numUserStreams;
}

int THCState_getNumBlasHandles(THCState* state)
{
  return state->numUserBlasHandles;
}

int THCState_getNumSparseHandles(THCState* state)
{
  return state->numUserSparseHandles;
}

THCLwdaResourcesPerDevice* THCState_getDeviceResourcePtr(
  THCState *state, int device)
{
  /* `device` is a LWCA index */
  if (device >= state->numDevices || device < 0)
  {
    THError("%d is not a device", device + 1 /* back to Torch index */);
  }

  return &(state->resourcesPerDevice[device]);
}

lwdaStream_t THCState_getDeviceStream(THCState *state, int device, int streamIndex)
{
  if (streamIndex > state->numUserStreams || streamIndex < 0)
  {
    THError("%d is not a stream", streamIndex);
  }
  THCLwdaResourcesPerDevice* res = THCState_getDeviceResourcePtr(state, device);
  THCStream* stream = res->streams[streamIndex];
  return stream->stream;
}

lwblasHandle_t THCState_getDeviceBlasHandle(THCState *state, int device, int handle)
{
  if (handle <= 0 || handle > state->numUserBlasHandles) {
    THError("%d is not a valid handle, valid range is: (1, %d)",
            handle, state->numUserBlasHandles);
  }
  THCLwdaResourcesPerDevice* res = THCState_getDeviceResourcePtr(state, device);
  THCState_reserveDeviceBlasHandles(state, device, handle);
  return res->blasHandles[handle - 1];
}

lwsparseHandle_t THCState_getDeviceSparseHandle(THCState *state, int device, int handle)
{
  if (handle <= 0 || handle > state->numUserSparseHandles) {
    THError("%d is not a valid handle, valid range is: (1, %d)",
            handle, state->numUserSparseHandles);
  }
  THCLwdaResourcesPerDevice* res = THCState_getDeviceResourcePtr(state, device);
  THCState_reserveDeviceSparseHandles(state, device, handle);
  return res->sparseHandles[handle - 1];
}

static THCStream* THCState_getStreamOnDevice(THCState* state, int device)
{
  THCThreadLocal local = state->lwrrentStreams[device];
  THCStream* stream = (THCStream*)THCThreadLocal_get(local);
  if (!stream) {
    stream = THCStream_defaultStream(device);
    THCStream_retain(stream);
    THCThreadLocal_set(local, stream);
  }
  return stream;
}

static void THCState_setStreamOnDevice(THCState *state, int device, THCStream *stream)
{
  THAssert(stream);
  if (stream->device != device) {
    THError("invalid stream; expected stream for device %d, but was on %d",
        device, stream->device);
  }
  THCStream_retain(stream);
  THCThreadLocal local = state->lwrrentStreams[device];
  THCStream_free((THCStream*)THCThreadLocal_get(local));
  THCThreadLocal_set(local, stream);
}

lwdaStream_t THCState_getLwrrentStreamOnDevice(THCState *state, int device)
{
  THCStream* stream = THCState_getStreamOnDevice(state, device);
  THAssert(stream);
  return stream->stream;
}

lwdaStream_t THCState_getLwrrentStream(THCState *state)
{
  /* This is called at the point of kernel exelwtion.
     For some debugging code or improperly instrumented kernels,
     `state` is null */
  if (state) {
    int device;
    THLwdaCheck(lwdaGetDevice(&device));
    return THCState_getLwrrentStreamOnDevice(state, device);
  } else {
    /* assume default stream */
    return NULL;
  }
}

lwblasHandle_t THCState_getLwrrentBlasHandle(THCState *state)
{
  /* This is called at the point of kernel exelwtion.
     For some debugging code or improperly instrumented kernels,
     `state` is null */
  if (state) {
    int device;
    THLwdaCheck(lwdaGetDevice(&device));

    int handle = THCState_getLwrrentBlasHandleIndex(state);
    return THCState_getDeviceBlasHandle(state, device, handle);
  }
  THError("THCState and blasHandles must be set as there is no default blasHandle");
  return NULL;
}

lwsparseHandle_t THCState_getLwrrentSparseHandle(THCState *state)
{
  /* This is called at the point of kernel exelwtion.
     For some debugging code or improperly instrumented kernels,
     `state` is null */
  if (state) {
    int device;
    THLwdaCheck(lwdaGetDevice(&device));

    int handle = THCState_getLwrrentSparseHandleIndex(state);
    return THCState_getDeviceSparseHandle(state, device, handle);
  }
  THError("THCState and sparseHandles must be set as there is no default sparseHandle");
  return NULL;
}

int THCState_getLwrrentStreamIndex(THCState *state)
{
  THCStream* stream = THCState_getStream(state);

  int device;
  THLwdaCheck(lwdaGetDevice(&device));
  THCLwdaResourcesPerDevice* res = THCState_getDeviceResourcePtr(state, device);
  for (int i = 0; i <= state->numUserStreams; ++i) {
    if (res->streams[i] == stream) {
      return i;
    }
  }

  return -1;
}

int THCState_getLwrrentBlasHandleIndex(THCState *state)
{
  void* value = THCThreadLocal_get(state->lwrrentPerDeviceBlasHandle);
  if (value == NULL) {
    return 1;
  }
  return (int) (intptr_t) value;
}

int THCState_getLwrrentSparseHandleIndex(THCState *state)
{
  void* value = THCThreadLocal_get(state->lwrrentPerDeviceSparseHandle);
  if (value == NULL) {
    return 1;
  }
  return (int) (intptr_t) value;
}

THCStream* THCState_getStream(THCState *state)
{
  int device;
  THLwdaCheck(lwdaGetDevice(&device));
  return THCState_getStreamOnDevice(state, device);
}

void THCState_setStream(THCState *state, THCStream *stream)
{
  int device;
  THLwdaCheck(lwdaGetDevice(&device));
  THCState_setStreamOnDevice(state, device, stream);
}

void THCState_setLwrrentStreamIndex(THCState *state, int streamIndex)
{
  if (streamIndex < 0 || streamIndex > state->numUserStreams) {
    THError("%d is not a valid stream, valid range is: (0, %d)", streamIndex,
        state->numUserStreams);
  }

  int device;
  for (device = 0; device < state->numDevices; ++device) {
    THCLwdaResourcesPerDevice* res = THCState_getDeviceResourcePtr(state, device);
    THCState_setStreamOnDevice(state, device, res->streams[streamIndex]);
  }
}

void THCState_setLwrrentBlasHandleIndex(THCState *state, int handle)
{
  if (handle > state->numUserBlasHandles || handle <= 0)
  {
    THError("%d is not a valid handle, valid range is: (1, %d)",
            handle, state->numUserBlasHandles);
  }
  THCThreadLocal_set(state->lwrrentPerDeviceBlasHandle, (void*)(intptr_t)handle);
}

void THCState_setLwrrentSparseHandleIndex(THCState *state, int handle)
{
  if (handle > state->numUserSparseHandles || handle <= 0)
  {
    THError("%d is not a valid handle, valid range is: (1, %d)",
            handle, state->numUserSparseHandles);
  }
  THCThreadLocal_set(state->lwrrentPerDeviceSparseHandle, (void*)(intptr_t)handle);
}

void* THCState_getLwrrentDeviceScratchSpace(THCState* state)
{
  int device = -1;
  THLwdaCheck(lwdaGetDevice(&device));
  int stream = THCState_getLwrrentStreamIndex(state);
  if (stream < 0) {
    // new stream API
    return NULL;
  }
  return THCState_getDeviceScratchSpace(state, device, stream);
}

void* THCState_getDeviceScratchSpace(THCState* state, int dev, int stream)
{
  THCLwdaResourcesPerDevice* res = THCState_getDeviceResourcePtr(state, dev);
  if (stream > state->numUserStreams || stream < 0) {
    THError("%d is not a stream", stream);
  }
  THCState_initializeScratchSpace(state, dev);
  return res->devScratchSpacePerStream[stream];
}

size_t THCState_getLwrrentDeviceScratchSpaceSize(THCState* state)
{
  int device = -1;
  THLwdaCheck(lwdaGetDevice(&device));
  return THCState_getDeviceScratchSpaceSize(state, device);
}

size_t THCState_getDeviceScratchSpaceSize(THCState* state, int device)
{
  THCLwdaResourcesPerDevice* res =
    THCState_getDeviceResourcePtr(state, device);

  return res->scratchSpacePerStream;
}

void __THLwdaCheck(lwdaError_t err, const char *file, const int line)
{
  if(err != lwdaSuccess)
  {
    static int alreadyFailed = 0;
    if(!alreadyFailed) {
      fprintf(stderr, "THLwdaCheck FAIL file=%s line=%i error=%i : %s\n", file, line, err, lwdaGetErrorString(err));
      alreadyFailed = 1;
    }
    _THError(file, line, "lwca runtime error (%d) : %s", err,
             lwdaGetErrorString(err));
  }
}

void __THLwdaCheckWarn(lwdaError_t err, const char *file, const int line)
{
  if(err != lwdaSuccess)
  {
    fprintf(stderr, "THLwdaCheckWarn FAIL file=%s line=%i error=%i : %s\n", file, line, err, lwdaGetErrorString(err));
  }
}

void __THLwblasCheck(lwblasStatus_t status, const char *file, const int line)
{
  if(status != LWBLAS_STATUS_SUCCESS)
  {
    const char* errmsg = NULL;

    switch(status)
    {
      case LWBLAS_STATUS_NOT_INITIALIZED:
        errmsg = "library not initialized";
        break;

      case LWBLAS_STATUS_ALLOC_FAILED:
        errmsg = "resource allocation failed";
        break;

      case LWBLAS_STATUS_ILWALID_VALUE:
        errmsg = "an invalid numeric value was used as an argument";
        break;

      case LWBLAS_STATUS_ARCH_MISMATCH:
        errmsg = "an absent device architectural feature is required";
        break;

      case LWBLAS_STATUS_MAPPING_ERROR:
        errmsg = "an access to GPU memory space failed";
        break;

      case LWBLAS_STATUS_EXELWTION_FAILED:
        errmsg = "the GPU program failed to execute";
        break;

      case LWBLAS_STATUS_INTERNAL_ERROR:
        errmsg = "an internal operation failed";
        break;

      default:
        errmsg = "unknown error";
        break;
    }

    _THError(file, line, "lwblas runtime error : %s", errmsg);
  }
}

void __THLwsparseCheck(lwsparseStatus_t status, const char *file, const int line)
{
  if(status != LWSPARSE_STATUS_SUCCESS)
  {
    const char* errmsg = NULL;

    switch(status)
    {
      case LWSPARSE_STATUS_NOT_INITIALIZED:
        errmsg = "library not initialized";
        break;

      case LWSPARSE_STATUS_ALLOC_FAILED:
        errmsg = "resource allocation failed";
        break;

      case LWSPARSE_STATUS_ILWALID_VALUE:
        errmsg = "an invalid numeric value was used as an argument";
        break;

      case LWSPARSE_STATUS_ARCH_MISMATCH:
        errmsg = "an absent device architectural feature is required";
        break;

      case LWSPARSE_STATUS_MAPPING_ERROR:
        errmsg = "an access to GPU memory space failed";
        break;

      case LWSPARSE_STATUS_EXELWTION_FAILED:
        errmsg = "the GPU program failed to execute";
        break;

      case LWSPARSE_STATUS_INTERNAL_ERROR:
        errmsg = "an internal operation failed";
        break;

      case LWSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
        errmsg = "the matrix type is not supported by this function";
        break;

      default:
        errmsg = "unknown error";
        break;
    }

    _THError(file, line, "lwsparse runtime error : %s", errmsg);
  }
}

static ptrdiff_t heapSize = 0; // not thread-local
static const ptrdiff_t heapMaxDelta = (ptrdiff_t)1e6;
static const ptrdiff_t heapMinDelta = (ptrdiff_t)-1e6;
static const double heapSoftmaxGrowthThresh = 0.8; // grow softmax if >80% max after GC
static const double heapSoftmaxGrowthFactor = 1.4; // grow softmax by 40%

void THCSetGCHandler(THCState *state, void (*lwtorchGCFunction_)(void *data), void *data )
{
  state->lwtorchGCFunction = lwtorchGCFunction_;
  state->lwtorchGCData = data;
}

lwdaError_t THLwdaMalloc(THCState *state, void** ptr, size_t size)
{
  THLwdaCheck(lwdaGetLastError());
  lwdaStream_t stream = THCState_getLwrrentStream(state);
  THCDeviceAllocator* allocator = state->lwdaDeviceAllocator;
  lwdaError_t err = allocator->malloc(allocator->state, ptr, size, stream);
  if (state->lwtorchGCFunction != NULL && err != lwdaSuccess) {
    lwdaGetLastError(); // reset OOM error
    (state->lwtorchGCFunction)(state->lwtorchGCData);
    err = allocator->malloc(allocator->state, ptr, size, stream);
  }
  return err;
}

lwdaError_t THLwdaFree(THCState *state, void *ptr)
{
  THCDeviceAllocator* allocator = state->lwdaDeviceAllocator;
  return allocator->free(allocator->state, ptr);
}

void* THLwdaHostAlloc(THCState *state, size_t size)
{
  THLwdaCheck(lwdaGetLastError());
  THAllocator* allocator = state->lwdaHostAllocator;
  return allocator->malloc(NULL, size);
}

void THLwdaHostFree(THCState *state, void *ptr)
{
  THAllocator* allocator = state->lwdaHostAllocator;
  return allocator->free(NULL, ptr);
}

void THLwdaHostRecord(THCState *state, void *ptr)
{
  if (state->lwdaHostAllocator == &THCCachingHostAllocator) {
    THCStream* stream = THCState_getStream(state);
    THCCachingHostAllocator_recordEvent(ptr, stream);
  }
}

lwdaError_t THLwdaMemGetInfo(THCState *state,  size_t* freeBytes, size_t* totalBytes)
{
  size_t largestBlock = 0;
  return THLwdaMemGetInfoCached(state, freeBytes, totalBytes, &largestBlock);
}

lwdaError_t THLwdaMemGetInfoCached(THCState *state,  size_t* freeBytes, size_t* totalBytes, size_t* largestBlock)
{
  size_t cachedBytes = 0;
  THCDeviceAllocator* allocator = state->lwdaDeviceAllocator;

  *largestBlock = 0;
  /* get info from LWCA first */
  lwdaError_t ret = lwdaMemGetInfo(freeBytes, totalBytes);
  if (ret!= lwdaSuccess)
    return ret;

  int device;
  ret = lwdaGetDevice(&device);
  if (ret!= lwdaSuccess)
    return ret;

  /* not always true - our optimistic guess here */
  *largestBlock = *freeBytes;

  if (allocator->cacheInfo != NULL)
    allocator->cacheInfo(allocator->state, device, &cachedBytes, largestBlock);
  
  /* Adjust resulting free bytes number. largesBlock unused for now */
  *freeBytes += cachedBytes;
  return lwdaSuccess;
}

static ptrdiff_t applyHeapDelta(THCState *state) {
  ptrdiff_t newHeapSize = THAtomicAddPtrdiff(&heapSize, state->heapDelta) + state->heapDelta;
  state->heapDelta = 0;
  return newHeapSize;
}

// Here we maintain a dynamic softmax threshold for THC-allocated storages.
// When THC heap size goes above this softmax, the GC hook is triggered.
// If heap size is above 80% of the softmax after GC, then the softmax is
// increased.
static void maybeTriggerGC(THCState *state, ptrdiff_t lwrHeapSize) {
  if (state->lwtorchGCFunction != NULL && lwrHeapSize > state->heapSoftmax) {
    (state->lwtorchGCFunction)(state->lwtorchGCData);

    // ensure heapSize is accurate before updating heapSoftmax
    ptrdiff_t newHeapSize = applyHeapDelta(state);

    if (newHeapSize > state->heapSoftmax * heapSoftmaxGrowthThresh) {
      state->heapSoftmax = (ptrdiff_t)state->heapSoftmax * heapSoftmaxGrowthFactor;
    }
  }
}

void THCHeapUpdate(THCState *state, ptrdiff_t size) {
  state->heapDelta += size;
  // batch updates to global heapSize to minimize thread contention
  if (state->heapDelta < heapMaxDelta && state->heapDelta > heapMinDelta) {
    return;
  }

  ptrdiff_t newHeapSize = applyHeapDelta(state);
  if (size > 0) {
    maybeTriggerGC(state, newHeapSize);
  }
}

#undef MIN_GLOBAL_SCRATCH_SPACE_PER_SM_STREAM
#undef MIN_GLOBAL_SCRATCH_SPACE_PER_DEVICE

#include "THCStorage.c"
#include "THCAllocator.c"

/* from THCHalf.h */

half THC_float2half(float f)
{
  half h;
  TH_float2halfbits(&f, &h.x);
  return h;
}

float  THC_half2float(half h)
{
  float f;
  TH_halfbits2float(&h.x, &f);
  return f;
}
