/*
* Copyright (c) 2015, Lwpu Corporation.  All rights reserved.
*
* THE INFORMATION CONTAINED HEREIN IS PROPRIETARY AND CONFIDENTIAL TO
* LWPU, CORPORATION.  USE, REPRODUCTION OR DISCLOSURE TO ANY THIRD PARTY
* IS SUBJECT TO WRITTEN PRE-APPROVAL BY LWPU, CORPORATION.
*
*
*/
#ifndef __lwnUtil_AlignedStorageImpl_h__
#define __lwnUtil_AlignedStorageImpl_h__

#include "lwn/lwn.h"
#include "lwnUtil_AlignedStorage.h"

namespace lwnUtil {

void* AlignedStorageAlloc(size_t size, size_t alignment)
{
    if (!size) {
        return NULL;
    }

    // Allocate sufficient memory to hold the padded-out size, plus an
    // original unaligned pointer (for freeing), enough memory to account for
    // an arbitrarily misaligned original allocation.
    size_t unalignedSize = AlignSize(size, alignment) + sizeof(char *) + alignment - 1;
    char *unalignedData = new char[unalignedSize];

    // Compute an aligned pointer by first reserving space for the original
    // pointer, and then realigning.
    char *alignedData = AlignPointer(unalignedData + sizeof(char *), alignment);

    // Stash the original allocation immediately before the aligned pointer
    // value.
    (reinterpret_cast<char **>(alignedData))[-1] = unalignedData;

    // Return the aligned pointer.
    return alignedData;
}

void AlignedStorageFree(void* data)
{
    // Make sure doing an aligned free of NULL is a NOP.
    if (!data) {
        return;
    }

    // If we have a non-NULL aligned pointer, pull the original unaligned
    // pointer from the memory immediately below it and then free that memory.
    char *unalignedData = (reinterpret_cast<char **>(data))[-1];
    delete[] unalignedData;
}


} // namespace lwnUtil

#endif // #ifndef __lwnUtil_AlignedStorageImpl_h__
