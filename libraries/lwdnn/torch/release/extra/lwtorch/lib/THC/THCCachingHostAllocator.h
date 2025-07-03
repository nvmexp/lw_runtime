#ifndef THC_CACHING_HOST_ALLOCATOR_INC
#define THC_CACHING_HOST_ALLOCATOR_INC

#include "THCGeneral.h"
#include "THCStream.h"

//
// A caching allocator for LWCA host allocations (pinned memory).
//
// This provides a drop-in replacement for THLwdaHostAllocator, which re-uses
// freed pinned (page-locked) memory allocations. This avoids device
// synchronizations due to lwdaFreeHost calls.
//
// To ensure correct behavior, THCCachingHostAllocator_recordEvent must be
// called anytime a pointer from this allocator is used in a lwdaMemcpyAsync
// call between host and device. The THC library implements this for storages
// and tensors in THCTensor_(copyAsyncCPU) and THCTensor_(copyAsyncLwda).
//
// Note that this allocator does not split larger allocations into smaller
// blocks, unlike the caching device allocator.
//
THC_API THAllocator THCCachingHostAllocator;

// Records an event in the specified stream. The allocation 'ptr' will not be
// re-used until the event has oclwrred.
THC_API lwdaError_t THCCachingHostAllocator_recordEvent(void *ptr, THCStream *stream);

// Releases cached pinned memory allocations via lwdaHostFree
THC_API void THCCachingHostAllocator_emptyCache(void);

#endif
