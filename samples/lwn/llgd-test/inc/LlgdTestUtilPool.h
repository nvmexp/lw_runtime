/*
 * Copyright (c) 2017, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 */

#pragma once

#include <lwn/lwn.h>
#include <lwn/lwn_Cpp.h>
#include <lwn/lwn_CppMethods.h>

#include <LlgdTestUtil.h>
#include <LlgdTestUtilLWN.h>

namespace llgd_lwn
{
enum class PoolType {
    Normal = 0,
    Physical,
    Virtual,
    Compressed,
    CompressedPhysical, // Comptags alloc is lazy
    CompressedPhysicalHasCompbits,
    CompressedVirtual,
    Last,
};

class PoolUtil
{
public:
    PoolUtil()
    {
        _storage = LlgdAlignedAllocPodType<uint8_t>(BIG_SIZE, ALIGNT);
    }
    void InitPool(MemoryPoolHolder& pool, PoolType poolType);

    static size_t MinSize(PoolType poolType);
    static bool IsVirtual(PoolType poolType);
    static bool IsPhysical(PoolType poolType);
    static const char* DescString(PoolType poolType);

private:
    lwn::MemoryPoolFlags MakeFlags(PoolType poolType);

    static const size_t SMALL_SIZE = 4096;
    static const size_t BIG_SIZE = 65536;
    static const size_t ALIGNT = LWN_MEMORY_POOL_STORAGE_ALIGNMENT;
    LlgdUniqueUint8PtrWithLwstomDeleter _storage;
};
} // llgd_lwn
