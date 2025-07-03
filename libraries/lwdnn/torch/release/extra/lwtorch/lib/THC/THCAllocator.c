#include "THCAllocator.h"

static void *THLwdaHostAllocator_malloc(void* ctx, ptrdiff_t size) {
  void* ptr;

  if (size < 0) THError("Invalid memory size: %ld", size);

  if (size == 0) return NULL;

  THLwdaCheck(lwdaMallocHost(&ptr, size));

  return ptr;
}

static void THLwdaHostAllocator_free(void* ctx, void* ptr) {
  if (!ptr) return;

  THLwdaCheck(lwdaFreeHost(ptr));
}

THAllocator THLwdaHostAllocator = {
  &THLwdaHostAllocator_malloc,
  NULL,
  &THLwdaHostAllocator_free
};

static lwdaError_t THCIpcAllocator_malloc(void* ctx, void** devPtr, size_t size, lwdaStream_t stream)
{
  THError("THCIpcAllocator.malloc() not supported");
  return lwdaSuccess;
}

static lwdaError_t THCIpcAllocator_free(void* ctx, void* devPtr)
{
  lwdaError_t err;
  int prev_device;
  int device = (int)(long)ctx;

  err = lwdaGetDevice(&prev_device);
  if (err != lwdaSuccess) { return err; }

  err = lwdaSetDevice(device);
  if (err != lwdaSuccess) { return err; }

  err = lwdaIpcCloseMemHandle(devPtr);

  lwdaSetDevice(prev_device);
  return err;
}

THCDeviceAllocator THCIpcAllocator = {
  &THCIpcAllocator_malloc,
  NULL,
  &THCIpcAllocator_free,
  NULL,
  NULL
};

static void *THLWVAAllocator_alloc(void* ctx, ptrdiff_t size) {
  if (size < 0) THError("Invalid memory size: %ld", size);

  if (size == 0) return NULL;

  // See J.1.1 of the LWDA_C_Programming_Guide.pdf for UVA and coherence rules
  // on various compute capabilities.
  void* ptr;
  THLwdaCheck(lwdaMallocManaged(&ptr, size, lwdaMemAttachGlobal));
  return ptr;
}

static void THLWVAAllocator_free(void* ctx, void* ptr) {
  if (!ptr) return;
  THLwdaCheck(lwdaFree(ptr));
}

THAllocator THLWVAAllocator = {
  &THLWVAAllocator_alloc,
  NULL,
  &THLWVAAllocator_free
};
