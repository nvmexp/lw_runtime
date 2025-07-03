#include "utils.h"
#include "luaT.h"
#include "THCGeneral.h"
#include "THCCachingAllocator.h"
#include "THCCachingHostAllocator.h"
#include "THCSleep.h"
#include "THCTensorRandom.h"
#include "THCHalf.h" // for LWDA_HALF_TENSOR

extern void lwtorch_LwdaByteStorage_init(lua_State* L);
extern void lwtorch_LwdaCharStorage_init(lua_State* L);
extern void lwtorch_LwdaShortStorage_init(lua_State* L);
extern void lwtorch_LwdaIntStorage_init(lua_State* L);
extern void lwtorch_LwdaLongStorage_init(lua_State* L);
extern void lwtorch_LwdaStorage_init(lua_State* L);
extern void lwtorch_LwdaDoubleStorage_init(lua_State* L);
#ifdef LWDA_HALF_TENSOR
extern void lwtorch_LwdaHalfStorage_init(lua_State* L);
#else
extern void lwtorch_HalfStorageCopy_init(lua_State *L);
#endif

extern void lwtorch_LwdaByteTensor_init(lua_State* L);
extern void lwtorch_LwdaCharTensor_init(lua_State* L);
extern void lwtorch_LwdaShortTensor_init(lua_State* L);
extern void lwtorch_LwdaIntTensor_init(lua_State* L);
extern void lwtorch_LwdaLongTensor_init(lua_State* L);
extern void lwtorch_LwdaTensor_init(lua_State* L);
extern void lwtorch_LwdaDoubleTensor_init(lua_State* L);
#ifdef LWDA_HALF_TENSOR
extern void lwtorch_LwdaHalfTensor_init(lua_State* L);
#else
extern void lwtorch_HalfTensorCopy_init(lua_State *L);
#endif

extern void lwtorch_LwdaByteTensorOperator_init(lua_State* L);
extern void lwtorch_LwdaCharTensorOperator_init(lua_State* L);
extern void lwtorch_LwdaShortTensorOperator_init(lua_State* L);
extern void lwtorch_LwdaIntTensorOperator_init(lua_State* L);
extern void lwtorch_LwdaLongTensorOperator_init(lua_State* L);
extern void lwtorch_LwdaTensorOperator_init(lua_State* L);
extern void lwtorch_LwdaDoubleTensorOperator_init(lua_State* L);
#ifdef LWDA_HALF_TENSOR
extern void lwtorch_LwdaHalfTensorOperator_init(lua_State* L);
#endif

extern void lwtorch_LwdaByteTensorMath_init(lua_State* L);
extern void lwtorch_LwdaCharTensorMath_init(lua_State* L);
extern void lwtorch_LwdaShortTensorMath_init(lua_State* L);
extern void lwtorch_LwdaIntTensorMath_init(lua_State* L);
extern void lwtorch_LwdaLongTensorMath_init(lua_State* L);
extern void lwtorch_LwdaTensorMath_init(lua_State* L);
extern void lwtorch_LwdaDoubleTensorMath_init(lua_State* L);
#ifdef LWDA_HALF_TENSOR
extern void lwtorch_LwdaHalfTensorMath_init(lua_State* L);
#endif


/*
   Iteration utilities for lists of streams and lists of gpus with streams
*/

int checkAndCountListOfStreams(lua_State *L, THCState *state, int arg,
                               int device)
{
  if (!lua_istable(L, arg)) {
    THError("expecting array of device streams");
  }

  /* Push table to top */
  lua_pushvalue(L, arg);

  /* Check that all values in the table are numeric and in bounds */
  int streams = 0;
  lua_pushnil(L);
  while (lua_next(L, -2)) {
    if (!lua_isnumber(L, -2)) {
      THError("expected array of streams, not table");
    }
    if (!lua_isnumber(L, -1)) {
      THError("array of stream ids must contain numeric ids");
    }
    int streamId = (int) lua_tonumber(L, -1);

    /* This will error out if the stream is not in bounds */
    THCState_getDeviceStream(state, device, streamId);

    ++streams;
    lua_pop(L, 1);
  }

  /* Pop table from top */
  lua_pop(L, 1);
  return streams;
}

void checkAndCountListOfGPUStreamPairs(lua_State *L, THCState *state, int arg,
                                       int* gpus,
                                       int* streams)
{
  if (!lua_istable(L, arg)) {
    THError("expecting table of gpu={streams...}");
  }

  /* Push table to top */
  lua_pushvalue(L, arg);

  /* Check that all values in the table are tables of numeric and in bounds */
  *gpus = 0;
  *streams = 0;

  lua_pushnil(L);
  while (lua_next(L, -2)) {
    /* -2 is key (device), -1 is value, in the form device={streams...} */
    if (!lua_isnumber(L, -2) || !lua_istable(L, -1)) {
      THError("expecting table of gpu={streams...}");
    }

    int device = (int) lua_tonumber(L, -2) - 1;
    /* Verify device is in range */
    if (device < 0 || device >= THCState_getNumDevices(state)) {
      THError("%d is not a device", device + 1);
    }

    /* Verify that the list is a list of streams */
    *streams += checkAndCountListOfStreams(L, state, -1, device);
    ++(*gpus);
    lua_pop(L, 1);
  }

  /* Pop table from top */
  lua_pop(L, 1);
}

int createSingleDeviceEvents(lua_State *L, THCState *state, int arg,
                             int device, lwdaEvent_t* event)
{

  /* Push table to top */
  lua_pushvalue(L, arg);

  /* Record events */
  lua_pushnil(L);
  int i = 0;
  while (lua_next(L, -2)) {
    int streamId = (int) lua_tonumber(L, -1);
    lwdaStream_t streamWaitingOn =
      THCState_getDeviceStream(state, device, streamId);
    THLwdaCheck(lwdaEventCreateWithFlags(&event[i], lwdaEventDisableTiming));
    THLwdaCheck(lwdaEventRecord(event[i], streamWaitingOn));
    lua_pop(L, 1);
    i++;
  }
  /* Pop table from top */
  lua_pop(L, 1);
  return i;
}

void createMultiDeviceEvents(lua_State *L, THCState *state, int arg,
                             lwdaEvent_t* events)
{
  /* Push {gpu={streams...}} table */
  lua_pushvalue(L, arg);

  /* Create and record events per each GPU */
  int gpu = 0;
  lua_pushnil(L);
  while (lua_next(L, -2)) {
    int device = (int) lua_tonumber(L, -2) - 1;
    THLwdaCheck(lwdaSetDevice(device));
    events += createSingleDeviceEvents(L, state, -1, device, events);
    ++gpu;

    lua_pop(L, 1);
  }

  /* Pop {gpu={streams...}} table */
  lua_pop(L, 1);
}

void waitSingleDeviceEvents(lua_State *L, THCState *state, int arg,
                           int device, lwdaEvent_t * event, int numEvents)
{
  /* Push table to top */
  lua_pushvalue(L, arg);

  /* Then, wait on the events. Each stream is actually waiting on itself here
     too, but that's harmless and isn't worth weeding out. */
  lua_pushnil(L);
  while (lua_next(L, -2)) {
    int streamId = (int) lua_tonumber(L, -1);
    lwdaStream_t stream =
      THCState_getDeviceStream(state, device, streamId);
    for (int i = 0; i < numEvents; i++) {
      THLwdaCheck(lwdaStreamWaitEvent(stream, event[i], 0));
    }
    lua_pop(L, 1);
  }

  /* Pop table from top */
  lua_pop(L, 1);
}


void waitMultiDeviceEvents(lua_State *L, THCState *state, int arg,
                           lwdaEvent_t* events, int streams)
{
  /* Push {gpu={streams...}} table */
  lua_pushvalue(L, arg);

  /* Then, wait on the events. Each stream is actually waiting on itself here
     too, but that's harmless and isn't worth weeding out. */
  lua_pushnil(L);
  while (lua_next(L, -2)) {
    int device = (int) lua_tonumber(L, -2) - 1;
    THLwdaCheck(lwdaSetDevice(device));

    /* Push stream table */
    lua_pushvalue(L, -1);
    lua_pushnil(L);
    while (lua_next(L, -2)) {
      int streamId = (int) lua_tonumber(L, -1);

      lwdaStream_t stream =
        THCState_getDeviceStream(state, device, streamId);

      /* Each stream waits on all events */
      for (int i = 0; i < streams; ++i) {
        THLwdaCheck(lwdaStreamWaitEvent(stream, events[i], 0));
      }

      lua_pop(L, 1);
    }

    /* Pop stream table and GPU entry */
    lua_pop(L, 2);
  }

  /* Pop {gpu={streams...}} table */
  lua_pop(L, 1);
}

/* Synchronizes the host with respect to the current device */
static int lwtorch_synchronize(lua_State *L)
{
  THLwdaCheck(lwdaDeviceSynchronize());
  return 0;
}

/* Synchronizes the host with respect to all devices */
static int lwtorch_synchronizeAll(lua_State *L)
{
  int prevDev = -1;
  THLwdaCheck(lwdaGetDevice(&prevDev));

  int devices = -1;
  THLwdaCheck(lwdaGetDeviceCount(&devices));

  for (int i = 0; i < devices; ++i) {
    THLwdaCheck(lwdaSetDevice(i));
    THLwdaCheck(lwdaDeviceSynchronize());
  }

  THLwdaCheck(lwdaSetDevice(prevDev));

  return 0;
}

/*
   Usage:
   lwtorch.reserveStreams(n)
   Allocates n user streams for every device present. If fewer than
   n streams are lwrrently allocated, an additional number will be added.
   If more than n streams are lwrrently allocated, does nothing.
   The default LWCA stream is assumed to be stream 0 and is always present;
   the allocated streams are user streams on top of the LWCA streams
   (thus, reserveStreams(1) will create 1 user stream with two being available,
   the default stream 0 and the user stream 1, on each device).
*/
static int lwtorch_reserveStreams(lua_State *L)
{
  THCState *state = lwtorch_getstate(L);
  int numStreams = (int) luaL_checknumber(L, 1);
  int nonBlocking = lua_toboolean(L, 2);
  THCState_reserveStreams(state, numStreams, nonBlocking);

  return 0;
}

/*
   Usage:
   lwtorch.reserveBlasHandles(n)
   Allocates n blasHandles for every device present. If fewer than
   n blasHandles are lwrrently allocated, an additional number will be added.
   If more than n blasHandles are lwrrently allocated, does nothing.
   Unlike for streams, there is no default blasHandle.
*/
static int lwtorch_reserveBlasHandles(lua_State *L)
{
  THCState *state = lwtorch_getstate(L);
  int numHandles = (int) luaL_checknumber(L, 1);
  THCState_reserveBlasHandles(state, numHandles);

  return 0;
}

/*
   Usage:
   n = lwtorch.getNumStreams()
   Returns the number of user streams allocated for every device present.
   By default, is 0.
*/
static int lwtorch_getNumStreams(lua_State *L)
{
  THCState *state = lwtorch_getstate(L);
  lua_pushnumber(L, THCState_getNumStreams(state));

  return 1;
}

/*
   Usage:
   n = lwtorch.getNumBlasHandles()
   Returns the number of user blasHandles allocated for every device present.
   By default, is 1.
*/
static int lwtorch_getNumBlasHandles(lua_State *L)
{
  THCState *state = lwtorch_getstate(L);
  lua_pushnumber(L, THCState_getNumBlasHandles(state));

  return 1;
}

/*
   Usage:
   lwtorch.setStream(n)
   For all devices, sets the current user stream in use to the index
   specified. e.g.,
   ---
   lwtorch.setDevice(1)
   lwtorch.setStream(3)
   -- device 1 stream 3 in use here
   lwtorch.setDevice(2)
   -- device 2 stream 3 in use here
   ---
   0 is the default stream on the device.
*/
static int lwtorch_setStream(lua_State *L)
{
  THCState *state = lwtorch_getstate(L);
  int stream = (int) luaL_checknumber(L, 1);
  THCState_setLwrrentStreamIndex(state, stream);

  return 0;
}

/*
   Usage:
   lwtorch.setBlasHandle(n)
   For all devices, sets the current blasHandle in use to the index
   specified. e.g.,
   ---
   lwtorch.setDevice(1)
   lwtorch.setBlasHandle(3)
   -- device 1 blasHandle 3 in use here
   lwtorch.setDevice(2)
   -- device 2 blasHandle 3 in use here
   ---
*/
static int lwtorch_setBlasHandle(lua_State *L)
{
  THCState *state = lwtorch_getstate(L);
  int handle = (int) luaL_checknumber(L, 1);
  THCState_setLwrrentBlasHandleIndex(state, handle);

  return 0;
}

/*
   Usage:
   n = lwtorch.getStream()
   Returns the current user stream for all devices in use (as previously
   set via lwtorch.setStream(n). 0 is the default stream on the device
   and is its initial value.
*/
static int lwtorch_getStream(lua_State *L)
{
  THCState *state = lwtorch_getstate(L);
  lua_pushnumber(L, THCState_getLwrrentStreamIndex(state));

  return 1;
}

/*
   Usage:
   n = lwtorch.getBlasHandle()
   Returns the current blasHandle for all devices in use (as previously
   set via lwtorch.setBlasHandle(n).
*/
static int lwtorch_getBlasHandle(lua_State *L)
{
  THCState *state = lwtorch_getstate(L);
  lua_pushnumber(L, THCState_getLwrrentBlasHandleIndex(state));

  return 1;
}

/*
   Usage:
   lwtorch.setDefaultStream()
   Equivalent to lwtorch.setStream(0).
*/
static int lwtorch_setDefaultStream(lua_State *L)
{
  THCState *state = lwtorch_getstate(L);
  THCState_setStream(state, NULL);
  return 0;
}

/*
   Usage:
   lwtorch.streamWaitFor(waiterStream, {waitForStream1, ..., waitForStreamN})
   for streams on the current device. Creates a one-way barrier where
   waiterStream waits for waitForStream1-N to reach the current point.
*/
static int lwtorch_streamWaitFor(lua_State *L)
{
  THCState *state = lwtorch_getstate(L);

  int lwrDev = -1;
  THLwdaCheck(lwdaGetDevice(&lwrDev));

  /* Check that the waiting stream is in bounds; this will error out if not */
  int waitingId = (int) luaL_checknumber(L, 1);
  lwdaStream_t streamWaiting =
    THCState_getDeviceStream(state, lwrDev, waitingId);

  /* Validate the streams that we are waiting on */
  int streams = checkAndCountListOfStreams(L, state, 2, lwrDev);

  if (streams < 1) {
    /* nothing to synchronize */
    return 0;
  }
  /* One-way dependency; streamWaiting will wait for the list of streams to
     wait on to complete exelwtion of pending scheduled kernels/events */
  lwdaEvent_t * events = (lwdaEvent_t*)malloc(sizeof(lwdaEvent_t) * streams);
  createSingleDeviceEvents(L, state, 2, lwrDev, events);
  /* Then, wait on them */
  for (int i = 0; i < streams; i++) {
    THLwdaCheck(lwdaStreamWaitEvent(streamWaiting, events[i], 0));
    THLwdaCheck(lwdaEventDestroy(events[i]));
  }
  free(events);
  return 0;
}

/*
   Usage:
   lwtorch.streamWaitForMultiDevice(gpuWaiter, streamWaiter,
                                    {[gpu1]={stream1_1, ..., stream1_N},
                                    [gpuK]={streamK_1, ..., streamK_M}})
   with a specified GPU per each list of streams.
   Stream (gpuWaiter, streamWaiter) will wait on all of the other streams
   (gpu1, stream1_1), ..., (gpu1, stream1_N), ...,
   (gpuK, streamK_1), ..., (gpuK, streamK_M) to complete fully, as a one-way
   barrier only (only streamWaiter is blocked).
   The streams to wait on are bucketed per device. Equivalent to
   streamWaitFor() if only one GPU's streams are listed.
*/
static int lwtorch_streamWaitForMultiDevice(lua_State *L)
{
  THCState *state = lwtorch_getstate(L);

  int prevDev = -1;
  THLwdaCheck(lwdaGetDevice(&prevDev));

  /* Validate waiting (gpu, stream); this will error out if not */
  int gpuWaiter = (int) luaL_checknumber(L, 1) - 1;
  int streamWaiter = (int) luaL_checknumber(L, 2);
  lwdaStream_t streamWaiting =
    THCState_getDeviceStream(state, gpuWaiter, streamWaiter);

  /* Validate and count set of {gpu={streams...}} we are waiting on */
  int gpus = 0;
  int streams = 0;
  checkAndCountListOfGPUStreamPairs(L, state, 3, &gpus, &streams);

  if (streams < 1) {
    /* nothing to synchronize together */
    return 0;
  }

  /*
     Events can only be recorded on the same device on which they are created.
     -For each GPU, create and record event per each stream given
     for that GPU.
     -For (gpuWaiter, streamWaiter), wait on all of the above events.
  */
  lwdaEvent_t* events = (lwdaEvent_t*) malloc(sizeof(lwdaEvent_t) * streams);

  /* First, create an event per GPU and record events for the specified stream
     on that GPU */
  createMultiDeviceEvents(L, state, 3, events);

  /* Then, wait on the events */
  THLwdaCheck(lwdaSetDevice(gpuWaiter));
  for (int i = 0; i < streams; ++i) {
    THLwdaCheck(lwdaStreamWaitEvent(streamWaiting, events[i], 0));
  }

  /* Clean up events */
  for (int i = 0; i < streams; ++i) {
    THLwdaCheck(lwdaEventDestroy(events[i]));
  }
  free(events);
  THLwdaCheck(lwdaSetDevice(prevDev));

  return 0;
}

/*
   Usage:
   lwtorch.streamBarrier({stream1, stream2, ..., streamN})
   applies to streams for the current device. Creates a N-way barrier
   to synchronize all of the streams given
*/
static int lwtorch_streamBarrier(lua_State *L)
{
  THCState *state = lwtorch_getstate(L);

  int lwrDev = -1;
  THLwdaCheck(lwdaGetDevice(&lwrDev));

  int streams = checkAndCountListOfStreams(L, state, 1, lwrDev);

  if (streams < 2) {
    /* nothing to synchronize together */
    return 0;
  }
  /* Multi-way dependency (barrier); all streams must complete exelwtion
     of pending scheduled kernels/events */
  lwdaEvent_t * events = (lwdaEvent_t*)malloc(sizeof(lwdaEvent_t) * streams);
  /* First, create an event and record them for all streams */
  int eventsCreated =  createSingleDeviceEvents(L, state, 1, lwrDev, events);

  /* Then, wait on the event. Each stream is actually waiting on itself here
     too, but that's harmless and isn't worth weeding out. */
  waitSingleDeviceEvents(L, state, 1, lwrDev, events, eventsCreated);
  for (int i = 0; i < eventsCreated; i++)
    THLwdaCheck(lwdaEventDestroy(events[i]));

  free(events);
  return 0;
}

/* usage:
   lwtorch.streamBarrierMultiDevice({[gpu1]={stream1_1, ..., stream1_N},
                                     [gpuK]={streamK_1, ..., streamK_M}})
   with a specified GPU per each list of streams.
   Each stream (gpu1, stream1_1), ..., (gpu1, stream1_N), ...,
               (gpuK, streamK_1), ..., (gpuK, streamK_M) will wait
   for all others to complete fully.
   Streams are bucketed per device. Equivalent to streamBarrier() if only
   one GPU is specified.
 */
static int lwtorch_streamBarrierMultiDevice(lua_State *L)
{
  THCState *state = lwtorch_getstate(L);

  int prevDev = -1;
  THLwdaCheck(lwdaGetDevice(&prevDev));

  /* Validate and count set of {gpu={streams...}} that are mutually waiting */
  int gpus = 0;
  int streams = 0;
  checkAndCountListOfGPUStreamPairs(L, state, 1, &gpus, &streams);

  if (streams < 2) {
    /* nothing to synchronize together */
    return 0;
  }

  /*
     Events can only be recorded on the same device on which they are created.
     -For each GPU, create an event, and record that event on each stream given
     for that GPU.
     -For each GPU, for each stream, wait on the event created by each other
     GPU.
  */
  lwdaEvent_t* events = (lwdaEvent_t*) malloc(sizeof(lwdaEvent_t) * streams);

  /* First, create an event per GPU and record events for the specified stream
     on that GPU */
  createMultiDeviceEvents(L, state, 1, events);

  /* Then, wait on the events. Each stream is actually waiting on itself here
     too, but that's harmless and isn't worth weeding out. */
  waitMultiDeviceEvents(L, state, 1, events, streams);

  /* Clean up events */
  for (int i = 0; i < streams; ++i) {
    THLwdaCheck(lwdaEventDestroy(events[i]));
  }
  free(events);
  THLwdaCheck(lwdaSetDevice(prevDev));

  return 0;
}

/*
   Usage:
   lwtorch.streamSynchronize(n)
   For the current device, synchronizes with the given stream only
   (lwdaStreamSynchronize).
   0 is the default stream on the device.
*/
static int lwtorch_streamSynchronize(lua_State *L)
{
  THCState *state = lwtorch_getstate(L);
  int streamId = (int) luaL_checknumber(L, 1);

  int lwrDev = -1;
  THLwdaCheck(lwdaGetDevice(&lwrDev));

  /* This also validates the stream */
  lwdaStream_t stream = THCState_getDeviceStream(state, lwrDev, streamId);
  THLwdaCheck(lwdaStreamSynchronize(stream));

  return 0;
}

static int lwtorch_getDevice(lua_State *L)
{
  int device;
  THLwdaCheck(lwdaGetDevice(&device));
  device++;
  lua_pushnumber(L, device);
  return 1;
}

static int lwtorch_deviceReset(lua_State *L)
{
  printf("WARNING: lwtorch.deviceReset has been depreceated."
	 " Just remove the call from your code.\n");
  return 0;
}

static int lwtorch_getDeviceCount(lua_State *L)
{
  int ndevice;
  THLwdaCheck(lwdaGetDeviceCount(&ndevice));
  lua_pushnumber(L, ndevice);
  return 1;
}

static int lwtorch_getPeerToPeerAccess(lua_State *L)
{
  THCState *state = lwtorch_getstate(L);
  int dev = (int) luaL_checknumber(L, 1) - 1;
  int devToAccess = (int) luaL_checknumber(L, 2) - 1;

  /* device bounds checking is performed within */
  int enabled = THCState_getPeerToPeerAccess(state, dev, devToAccess);
  lua_pushboolean(L, enabled);

  return 1;
}

static int lwtorch_setPeerToPeerAccess(lua_State *L)
{
  THCState *state = lwtorch_getstate(L);
  int dev = (int) luaL_checknumber(L, 1) - 1;
  int devToAccess = (int) luaL_checknumber(L, 2) - 1;
  int enable = lua_toboolean(L, 3);

  /* device bounds checking is performed within */
  THCState_setPeerToPeerAccess(state, dev, devToAccess, enable);

  return 0;
}

static int lwtorch_getKernelPeerToPeerAccess(lua_State *L)
{
  THCState *state = lwtorch_getstate(L);
  lua_pushboolean(L, THCState_getKernelPeerToPeerAccessEnabled(state));

  return 1;
}

static int lwtorch_setKernelPeerToPeerAccess(lua_State *L)
{
  THCState *state = lwtorch_getstate(L);

  int val = lua_toboolean(L, -1);
  THCState_setKernelPeerToPeerAccessEnabled(state, val);

  return 0;
}

static int lwtorch_isCachingAllocatorEnabled(lua_State *L)
{
  THCState *state = lwtorch_getstate(L);
  lua_pushboolean(L, THCState_isCachingAllocatorEnabled(state));

  return 1;
}

static int lwtorch_getMemoryUsage(lua_State *L) {
  size_t freeBytes = 0;
  size_t totalBytes = 0;
  int lwrDevice;
  THLwdaCheck(lwdaGetDevice(&lwrDevice));
  THCState *state = lwtorch_getstate(L);

  int device = luaL_optint(L, 1, -10);
  if (device == -10) { /* no argument passed, current device mem usage */
    THLwdaCheck(THLwdaMemGetInfo(state, &freeBytes, &totalBytes));
  } else { /* argument was given, particular device's memory usage */
    THLwdaCheck(lwdaSetDevice(device-1)); /* zero indexed */
    THLwdaCheck(THLwdaMemGetInfo(state, &freeBytes, &totalBytes));
    THLwdaCheck(lwdaSetDevice(lwrDevice));
  }
  lua_pushnumber(L, freeBytes);
  lua_pushnumber(L, totalBytes);
  return 2;
}

static int lwtorch_setDevice(lua_State *L)
{
  THCState *state = lwtorch_getstate(L);
  int device = (int)luaL_checknumber(L, 1)-1;
  THLwdaCheck(lwdaSetDevice(device));
  return 0;
}

#define SET_DEVN_PROP(NAME) \
  lua_pushnumber(L, prop.NAME); \
  lua_setfield(L, -2, #NAME);

static int lwtorch_getDeviceProperties(lua_State *L)
{
  int device = (int)luaL_checknumber(L, 1)-1;

  // switch context to given device so the call to lwdaMemGetInfo is for the correct device
  int oldDevice;
  THLwdaCheck(lwdaGetDevice(&oldDevice));
  THLwdaCheck(lwdaSetDevice(device));

  struct lwdaDeviceProp prop;
  THLwdaCheck(lwdaGetDeviceProperties(&prop, device));
  lua_newtable(L);
  SET_DEVN_PROP(canMapHostMemory);
  SET_DEVN_PROP(clockRate);
  SET_DEVN_PROP(computeMode);
  SET_DEVN_PROP(deviceOverlap);
  SET_DEVN_PROP(integrated);
  SET_DEVN_PROP(kernelExecTimeoutEnabled);
  SET_DEVN_PROP(major);
  SET_DEVN_PROP(maxThreadsPerBlock);
  SET_DEVN_PROP(memPitch);
  SET_DEVN_PROP(minor);
  SET_DEVN_PROP(multiProcessorCount);
  SET_DEVN_PROP(regsPerBlock);
  SET_DEVN_PROP(sharedMemPerBlock);
  SET_DEVN_PROP(textureAlignment);
  SET_DEVN_PROP(totalConstMem);
  SET_DEVN_PROP(totalGlobalMem);
  SET_DEVN_PROP(warpSize);
  SET_DEVN_PROP(pciBusID);
  SET_DEVN_PROP(pciDeviceID);
  SET_DEVN_PROP(pciDomainID);
  SET_DEVN_PROP(maxTexture1D);
  SET_DEVN_PROP(maxTexture1DLinear);

  size_t freeMem;
  THLwdaCheck(lwdaMemGetInfo (&freeMem, NULL));
  lua_pushnumber(L, freeMem);
  lua_setfield(L, -2, "freeGlobalMem");

  lua_pushstring(L, prop.name);
  lua_setfield(L, -2, "name");

  // restore context
  THLwdaCheck(lwdaSetDevice(oldDevice));

  return 1;
}

static int lwtorch_getRuntimeVersion(lua_State *L)
{
  int version;
  THLwdaCheck(lwdaRuntimeGetVersion(&version));
  lua_pushnumber(L, version);
  return 1;
}

static int lwtorch_getDriverVersion(lua_State *L)
{
  int version;
  THLwdaCheck(lwdaDriverGetVersion(&version));
  lua_pushnumber(L, version);
  return 1;
}

static int lwtorch_seed(lua_State *L)
{
  unsigned long long seed = THCRandom_seed(lwtorch_getstate(L));
  lua_pushnumber(L, seed);
  return 1;
}

static int lwtorch_seedAll(lua_State *L)
{
  unsigned long long seed = THCRandom_seedAll(lwtorch_getstate(L));
  lua_pushnumber(L, seed);
  return 1;
}

static int lwtorch_initialSeed(lua_State *L)
{
  unsigned long long seed = THCRandom_initialSeed(lwtorch_getstate(L));
  lua_pushnumber(L, seed);
  return 1;
}

static int lwtorch_manualSeed(lua_State *L)
{
  unsigned long long seed = luaL_checknumber(L, 1);
  THCRandom_manualSeed(lwtorch_getstate(L), seed);
  return 0;
}

static int lwtorch_manualSeedAll(lua_State* L)
{
  unsigned long long seed = luaL_checknumber(L, 1);
  THCRandom_manualSeedAll(lwtorch_getstate(L), seed);
  return 0;
}

static int lwtorch_getRNGState(lua_State *L)
{
  THByteTensor* t = THByteTensor_new();
  THCRandom_getRNGState(lwtorch_getstate(L), t);
  luaT_pushudata(L, t, "torch.ByteTensor");
  return 1;
}

static int lwtorch_setRNGState(lua_State *L)
{
  THByteTensor* t = luaT_checkudata(L, 1, "torch.ByteTensor");
  THCRandom_setRNGState(lwtorch_getstate(L), t);
  return 0;
}

static int lwtorch_getState(lua_State *L)
{
  lua_getglobal(L, "lwtorch");
  lua_getfield(L, -1, "_state");
  lua_remove(L, -2);
  return 1;
}

static int lwtorch_Event_new(lua_State *L)
{
  lwdaEvent_t *event = luaT_alloc(L, sizeof(lwdaEvent_t));
  THLwdaCheck(lwdaEventCreate(event));

  THCState *state = lwtorch_getstate(L);
  THLwdaCheck(lwdaEventRecord(*event, THCState_getLwrrentStream(state)));
  luaT_pushudata(L, event, "lwtorch.Event");

  return 1;
}

static int lwtorch_Event_free(lua_State *L)
{
  lwdaEvent_t *event = luaT_checkudata(L, 1, "lwtorch.Event");
  THLwdaCheck(lwdaEventDestroy(*event));
  luaT_free(L, event);

  return 0;
}

static int lwtorch_Event_waitOn(lua_State *L)
{
  lwdaEvent_t *event = luaT_checkudata(L, 1, "lwtorch.Event");
  THCState *state = lwtorch_getstate(L);
  THLwdaCheck(lwdaStreamWaitEvent(THCState_getLwrrentStream(state), *event, 0));

  return 0;
}

static const struct luaL_Reg lwtorch_Event__[] = {
  {"waitOn", lwtorch_Event_waitOn},
  {NULL, NULL}
};

static void lwtorch_Event_init(lua_State *L)
{
  luaT_newmetatable(L, "lwtorch.Event", NULL, lwtorch_Event_new, lwtorch_Event_free, NULL);
  luaT_setfuncs(L, lwtorch_Event__, 0);
  lua_pop(L, 1);
}

static void luaLwtorchGCFunction(void *data)
{
  lua_State *L = data;
  lua_gc(L, LUA_GCCOLLECT, 0);
}

static int lwtorch_setHeapTracking(lua_State *L)
{
  THCState *state = lwtorch_getstate(L);
  int enabled = luaT_checkboolean(L,1);
  if(enabled) {
    THCSetGCHandler(state, luaLwtorchGCFunction, L);
  } else {
    THCSetGCHandler(state, NULL, NULL);
  }
  return 0;
}

static int lwtorch_isManagedPtr(lua_State *L)
{
  THCState *state = lwtorch_getstate(L);
  if(lua_type(L, 1) != LUA_TNUMBER) {
    THError("Must receive a ptr cast as a number");
  }
  void* ptr = (void* )luaL_optinteger(L, 1, 0);
  struct lwdaPointerAttributes attributes;
  lwdaError_t res = lwdaPointerGetAttributes(&attributes, ptr);
  if (res == lwdaErrorIlwalidValue) {
    lua_pushboolean(L, 0);
  } else {
    THLwdaCheck(res);
    lua_pushboolean(L, attributes.isManaged);
  }
  return 1;
}

static int lwtorch_shutdown(lua_State *L)
{
  THCState **state = (THCState **) lua_topointer(L, 1);
  THLwdaShutdown(*state);
  THCState_free(*state);
  return 0;
}

static int lwtorch_hasHalfInstructions(lua_State *L) {
  THCState *state = lwtorch_getstate(L);
#ifdef LWDA_HALF_TENSOR
  lua_pushboolean(L, THC_nativeHalfInstructions(state));
#else
  lua_pushboolean(L, 0);
#endif
  return 1;
}

static int lwtorch_hasFastHalfInstructions(lua_State *L) {
  THCState *state = lwtorch_getstate(L);
#ifdef LWDA_HALF_TENSOR
  lua_pushboolean(L, THC_fastHalfInstructions(state));
#else
  lua_pushboolean(L, 0);
#endif
  return 1;
}

static int lwtorch_sleep(lua_State *L) {
  THCState *state = lwtorch_getstate(L);
  if (!luaT_checklong(L, 1)) {
      THError("expected number 'cycles'");
  }
  THC_sleep(state, luaT_tolong(L, 1));
  return 0;
}

static const struct luaL_Reg lwtorch_stuff__ [] = {
  {"synchronize", lwtorch_synchronize},
  {"synchronizeAll", lwtorch_synchronizeAll},
  {"reserveBlasHandles", lwtorch_reserveBlasHandles},
  {"getNumBlasHandles", lwtorch_getNumBlasHandles},
  {"setBlasHandle", lwtorch_setBlasHandle},
  {"getBlasHandle", lwtorch_getBlasHandle},
  {"reserveStreams", lwtorch_reserveStreams},
  {"getNumStreams", lwtorch_getNumStreams},
  {"setStream", lwtorch_setStream},
  {"getStream", lwtorch_getStream},
  {"setDefaultStream", lwtorch_setDefaultStream},
  {"streamWaitFor", lwtorch_streamWaitFor},
  {"streamWaitForMultiDevice", lwtorch_streamWaitForMultiDevice},
  {"streamBarrier", lwtorch_streamBarrier},
  {"streamBarrierMultiDevice", lwtorch_streamBarrierMultiDevice},
  {"streamSynchronize", lwtorch_streamSynchronize},
  {"getDevice", lwtorch_getDevice},
  {"deviceReset", lwtorch_deviceReset},
  {"getDeviceCount", lwtorch_getDeviceCount},
  {"getPeerToPeerAccess", lwtorch_getPeerToPeerAccess},
  {"setPeerToPeerAccess", lwtorch_setPeerToPeerAccess},
  {"setKernelPeerToPeerAccess", lwtorch_setKernelPeerToPeerAccess},
  {"getKernelPeerToPeerAccess", lwtorch_getKernelPeerToPeerAccess},
  {"isCachingAllocatorEnabled", lwtorch_isCachingAllocatorEnabled},
  {"getDeviceProperties", lwtorch_getDeviceProperties},
  {"getRuntimeVersion", lwtorch_getRuntimeVersion},
  {"getDriverVersion", lwtorch_getDriverVersion},
  {"getMemoryUsage", lwtorch_getMemoryUsage},
  {"hasHalfInstructions", lwtorch_hasHalfInstructions},
  {"hasFastHalfInstructions", lwtorch_hasFastHalfInstructions},
  {"setDevice", lwtorch_setDevice},
  {"seed", lwtorch_seed},
  {"seedAll", lwtorch_seedAll},
  {"initialSeed", lwtorch_initialSeed},
  {"manualSeed", lwtorch_manualSeed},
  {"manualSeedAll", lwtorch_manualSeedAll},
  {"_sleep", lwtorch_sleep},
  {"getRNGState", lwtorch_getRNGState},
  {"setRNGState", lwtorch_setRNGState},
  {"getState", lwtorch_getState},
  {"setHeapTracking", lwtorch_setHeapTracking},
  {"isManagedPtr", lwtorch_isManagedPtr},
  {NULL, NULL}
};

LUA_EXTERNC DLL_EXPORT int luaopen_liblwtorch(lua_State *L);

int luaopen_liblwtorch(lua_State *L)
{
  lua_newtable(L);
  lua_pushvalue(L, -1);
  lua_setglobal(L, "lwtorch");
  luaL_setfuncs(L, lwtorch_stuff__, 0);

  THCState* state = THCState_alloc();

  /* Enable the caching allocator unless THC_CACHING_ALLOCATOR=0 */
  char* thc_caching_allocator = getelw("THC_CACHING_ALLOCATOR");
  if (!thc_caching_allocator || strcmp(thc_caching_allocator, "0") != 0) {
    THCState_setDeviceAllocator(state, THCCachingAllocator_get());
    state->lwdaHostAllocator = &THCCachingHostAllocator;
  }

  THLwdaInit(state);

  /* Register torch.LwdaHostAllocator. */
  luaT_pushudata(L, THCState_getLwdaHostAllocator(state), "torch.Allocator");
  lua_setfield(L, -2, "LwdaHostAllocator");

  /* Register torch.LwdaUVAHostAllocator. */
  luaT_pushudata(L, THCState_getLwdaUVAAllocator(state), "torch.Allocator");
  lua_setfield(L, -2, "LwdaUVAAllocator");

#ifdef USE_MAGMA
  THCMagma_init(state);
  lua_pushboolean(L, 1);
  lua_setfield(L, -2, "magma");
#endif

  lwtorch_LwdaByteStorage_init(L);
  lwtorch_LwdaCharStorage_init(L);
  lwtorch_LwdaShortStorage_init(L);
  lwtorch_LwdaIntStorage_init(L);
  lwtorch_LwdaLongStorage_init(L);
  lwtorch_LwdaStorage_init(L);
  lwtorch_LwdaDoubleStorage_init(L);
#ifdef LWDA_HALF_TENSOR
  lwtorch_LwdaHalfStorage_init(L);
#else
  lwtorch_HalfStorageCopy_init(L);
#endif

  lwtorch_LwdaByteTensor_init(L);
  lwtorch_LwdaCharTensor_init(L);
  lwtorch_LwdaShortTensor_init(L);
  lwtorch_LwdaIntTensor_init(L);
  lwtorch_LwdaLongTensor_init(L);
  lwtorch_LwdaTensor_init(L);
  lwtorch_LwdaDoubleTensor_init(L);
#ifdef LWDA_HALF_TENSOR
  lwtorch_LwdaHalfTensor_init(L);
#else
  lwtorch_HalfTensorCopy_init(L);
#endif

  lwtorch_LwdaByteTensorOperator_init(L);
  lwtorch_LwdaCharTensorOperator_init(L);
  lwtorch_LwdaShortTensorOperator_init(L);
  lwtorch_LwdaIntTensorOperator_init(L);
  lwtorch_LwdaLongTensorOperator_init(L);
  lwtorch_LwdaTensorOperator_init(L);
  lwtorch_LwdaDoubleTensorOperator_init(L);
#ifdef LWDA_HALF_TENSOR
  lwtorch_LwdaHalfTensorOperator_init(L);
#endif

  lwtorch_LwdaByteTensorMath_init(L);
  lwtorch_LwdaCharTensorMath_init(L);
  lwtorch_LwdaShortTensorMath_init(L);
  lwtorch_LwdaIntTensorMath_init(L);
  lwtorch_LwdaLongTensorMath_init(L);
  lwtorch_LwdaTensorMath_init(L);
  lwtorch_LwdaDoubleTensorMath_init(L);
#ifdef LWDA_HALF_TENSOR
  lwtorch_LwdaHalfTensorMath_init(L);
#endif

  lwtorch_Event_init(L);

  /* Store state in lwtorch table. */
  lua_pushlightuserdata(L, state);
  lua_setfield(L, -2, "_state");

#ifdef LWDA_HALF_TENSOR
  lua_pushboolean(L, 1);
#else
  lua_pushboolean(L, 0);
#endif
  lua_setfield(L, -2, "hasHalf");

  /* store gpu driver version in field */
  int driverVersion;
  THLwdaCheck(lwdaDriverGetVersion(&driverVersion));
  lua_pushinteger(L, driverVersion);
  lua_setfield(L, -2, "driverVersion");

  /* when lwtorch goes out of scope, we need to make sure THCState is properly
     shut down (so that memory doesn not leak. Since _state is a lightuserdata
     we cannot associate an __gc method with it. Hence, create a userdata, and
     associate a metatable with it, which has an __gc method which properly
     calls THLwdaShutdown.
  */
  /* create a new userdata type which is a pointer to a pointer */
  THCState **thc_pointer = (THCState**)lua_newuserdata(L, sizeof(void*));
  /* set the state pointer */
  *thc_pointer = state;
  /* create a table that will be used as the metatable */
  lua_newtable(L);
  /* push the gc function onto the stack */
  lua_pushcfunction(L, &lwtorch_shutdown);
  /* set the __gc field in the table to the function (function is popped) */
  lua_setfield(L, -2, "__gc");
  /* now the table is on the top of the stack, and the userdata below it,
     setmetatable on the userdata with the table. table is popped */
  lua_setmetatable(L, -2);
  /* now the userdata is on top, with the lwtorch table below it,
     set the field lwtorch.__stategc to this userdata.
     userdata is popped, leaving lwtorch table on top of the stack */
  lua_setfield(L, -2, "_stategc");

  return 1;
}
