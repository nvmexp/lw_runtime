/*
* Copyright (c) 2015, Lwpu Corporation.  All rights reserved.
*
* THE INFORMATION CONTAINED HEREIN IS PROPRIETARY AND CONFIDENTIAL TO
* LWPU, CORPORATION.  USE, REPRODUCTION OR DISCLOSURE TO ANY THIRD PARTY
* IS SUBJECT TO WRITTEN PRE-APPROVAL BY LWPU, CORPORATION.
*
*
*/
#ifndef __lwnUtil_AlignedStorage_h__
#define __lwnUtil_AlignedStorage_h__

#include "lwn/lwn.h"

namespace lwnUtil {

//
// ALIGNED STORAGE UTILITY FUNCTIONS
//
// The functions are provided to facilitate in allocating aligned storage
// using allocators that don't already return pointers with sufficient
// alignment.
//
// These utilities pad out the size requests so there is sufficient memory in
// the allocation after we align the allocation pointer up to the next
// multiple of <alignment> while ensuring that we also have space to store the
// original allocation pointer immediately before the new aligned pointer.


// Template function aligning a pointer to type <T> up to the next multiple of
// alignment.
template <typename T> 
static inline T *AlignPointer(T *unalignedPtr, size_t alignment)
{
    const size_t alignMask = alignment - 1;
    return (T *)((uintptr_t(unalignedPtr) + alignMask) & (~alignMask));
}

// Utility function padding out size given by <size> to the next multiple of
// <alignment>.
static inline size_t AlignSize(size_t size, size_t alignment)
{
    const size_t alignMask = alignment - 1;
    return (uintptr_t(size) + alignMask) & (~alignMask);
}

// Allocate at least <size> bytes of memory, returning a pointer aligned to a
// multiple of <alignment>.
void *AlignedStorageAlloc(size_t size, size_t alignment);

// Free aligned storage, using the original pointer stashed immediately before
// <data>.
void AlignedStorageFree(void *data);


//
// POOL STORAGE UTILITY FUNCTIONS
//
// Specialized versions of the aligned storage functions used to set up
// allocations aligned on page boundaries as required by memory pools.
//
template <typename T>
static inline T *PoolStorageAlign(T *unalignedValue)
{
    return AlignPointer(unalignedValue, LWN_MEMORY_POOL_STORAGE_ALIGNMENT);
}

static inline size_t PoolStorageSize(size_t size)
{
    return AlignSize(size, LWN_MEMORY_POOL_STORAGE_ALIGNMENT);
}

static inline void *PoolStorageAlloc(size_t size)
{
    return AlignedStorageAlloc(size, LWN_MEMORY_POOL_STORAGE_ALIGNMENT);
}

static inline void PoolStorageFree(void *data)
{
    return AlignedStorageFree(data);
}

} // namespace lwnUtil

#endif // #ifndef __lwnUtil_AlignedStorage_h__
