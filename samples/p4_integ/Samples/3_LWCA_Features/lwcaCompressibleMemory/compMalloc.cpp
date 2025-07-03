/* Copyright (c) 2022, LWPU CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of LWPU CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROLWREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <stdio.h>
#include <string.h>
#include <helper_lwda.h>
#include <lwca.h>
#include <lwda_runtime_api.h>

lwdaError_t setProp(LWmemAllocationProp *prop, bool UseCompressibleMemory)
{
    LWdevice lwrrentDevice;
    if (lwCtxGetDevice(&lwrrentDevice) != LWDA_SUCCESS)
        return lwdaErrorMemoryAllocation;

    memset(prop, 0, sizeof(LWmemAllocationProp));
    prop->type = LW_MEM_ALLOCATION_TYPE_PINNED;
    prop->location.type = LW_MEM_LOCATION_TYPE_DEVICE;
    prop->location.id = lwrrentDevice;

    if (UseCompressibleMemory)
        prop->allocFlags.compressionType = LW_MEM_ALLOCATION_COMP_GENERIC;

    return lwdaSuccess;
}

lwdaError_t allocateCompressible(void **adr, size_t size, bool UseCompressibleMemory)
{
    LWmemAllocationProp prop = {};
    lwdaError_t err = setProp(&prop, UseCompressibleMemory);
    if (err != lwdaSuccess)
        return err;

    size_t granularity = 0;
    if (lwMemGetAllocationGranularity(&granularity, &prop,
                                      LW_MEM_ALLOC_GRANULARITY_MINIMUM) != LWDA_SUCCESS)
        return lwdaErrorMemoryAllocation;
    size = ((size - 1) / granularity + 1) * granularity;
    LWdeviceptr dptr;
    if (lwMemAddressReserve(&dptr, size, 0, 0, 0) != LWDA_SUCCESS)
        return lwdaErrorMemoryAllocation;

    LWmemGenericAllocationHandle allocationHandle;
    if (lwMemCreate(&allocationHandle, size, &prop, 0) != LWDA_SUCCESS)
        return lwdaErrorMemoryAllocation;

    // Check if lwMemCreate was able to allocate compressible memory.
    if (UseCompressibleMemory) {
        LWmemAllocationProp allocationProp = {};
        lwMemGetAllocationPropertiesFromHandle(&allocationProp, allocationHandle);
        if (allocationProp.allocFlags.compressionType != LW_MEM_ALLOCATION_COMP_GENERIC) {
            printf("Could not allocate compressible memory... so waiving exelwtion\n");
            exit(EXIT_WAIVED);
        }
    }

    if (lwMemMap(dptr, size, 0, allocationHandle, 0) != LWDA_SUCCESS)
        return lwdaErrorMemoryAllocation;

    if (lwMemRelease(allocationHandle) != LWDA_SUCCESS)
        return lwdaErrorMemoryAllocation;

    LWmemAccessDesc accessDescriptor;
    accessDescriptor.location.id = prop.location.id;
    accessDescriptor.location.type = prop.location.type;
    accessDescriptor.flags = LW_MEM_ACCESS_FLAGS_PROT_READWRITE;

    if (lwMemSetAccess(dptr, size, &accessDescriptor, 1) != LWDA_SUCCESS)
        return lwdaErrorMemoryAllocation;

    *adr = (void *)dptr;
    return lwdaSuccess;
}

lwdaError_t freeCompressible(void *ptr, size_t size, bool UseCompressibleMemory)
{
    LWmemAllocationProp prop = {};
    lwdaError_t err = setProp(&prop, UseCompressibleMemory);
    if (err != lwdaSuccess)
        return err;

    size_t granularity = 0;
    if (lwMemGetAllocationGranularity(&granularity, &prop,
                                      LW_MEM_ALLOC_GRANULARITY_MINIMUM) != LWDA_SUCCESS)
        return lwdaErrorMemoryAllocation;
    size = ((size - 1) / granularity + 1) * granularity;

    if (ptr == NULL)
        return lwdaSuccess;
    if (lwMemUnmap((LWdeviceptr)ptr, size) != LWDA_SUCCESS ||
        lwMemAddressFree((LWdeviceptr)ptr, size) != LWDA_SUCCESS)
        return lwdaErrorIlwalidValue;
    return lwdaSuccess;
}
