/*************************************************************************
 * Copyright (c) 2019, LWPU CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_ALLOC_H_
#define NCCL_ALLOC_H_

#include "lwcl.h"
#include "checks.h"
#include <sys/mman.h>

static inline ncclResult_t ncclLwdaHostAlloc(void** ptr, void** devPtr, size_t size) {
  LWDACHECK(lwdaHostAlloc(ptr, size, lwdaHostAllocMapped));
  memset(*ptr, 0, size);
  *devPtr = *ptr;
  return ncclSuccess;
}

static inline ncclResult_t ncclLwdaHostFree(void* ptr) {
  LWDACHECK(lwdaFreeHost(ptr));
  return ncclSuccess;
}

template <typename T>
static ncclResult_t ncclCalloc(T** ptr, size_t nelem) {
  void* p = malloc(nelem*sizeof(T));
  if (p == NULL) {
    WARN("Failed to malloc %ld bytes", nelem*sizeof(T));
    return ncclSystemError;
  }
  memset(p, 0, nelem*sizeof(T));
  *ptr = (T*)p;
  return ncclSuccess;
}

template <typename T>
static ncclResult_t ncclLwdaCalloc(T** ptr, size_t nelem) {
  LWDACHECK(lwdaMalloc(ptr, nelem*sizeof(T)));
  LWDACHECK(lwdaMemset(*ptr, 0, nelem*sizeof(T)));
  return ncclSuccess;
}

template <typename T>
static ncclResult_t ncclLwdaMemcpy(T* dst, T* src, size_t nelem) {
  LWDACHECK(lwdaMemcpy(dst, src, nelem*sizeof(T), lwdaMemcpyDefault));
  return ncclSuccess;
}

#endif
