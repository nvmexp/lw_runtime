/*
 * Copyright (c) 2017, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property and
 * proprietary rights in and to this software and related documentation.  Any
 * use, reproduction, disclosure or distribution of this software and related
 * documentation without an express license agreement from LWPU Corporation
 * is strictly prohibited.
 */

#include <LlgdTestUtilPool.h>

namespace llgd_lwn
{
lwn::MemoryPoolFlags PoolUtil::MakeFlags(PoolType poolType)
{
    using Flags = MemoryPoolFlags;
    switch (poolType) {
    default:
        __builtin_trap();
    case PoolType::Normal:
        return Flags::CPU_UNCACHED | Flags::GPU_CACHED;
    case PoolType::Physical:
        return Flags::CPU_UNCACHED | Flags::GPU_NO_ACCESS | Flags::PHYSICAL;
    case PoolType::Virtual:
        return Flags::CPU_NO_ACCESS | Flags::GPU_CACHED | Flags::VIRTUAL;
    case PoolType::Compressed:
        return Flags::CPU_UNCACHED | Flags::GPU_CACHED | Flags::COMPRESSIBLE;
    case PoolType::CompressedPhysical:
    case PoolType::CompressedPhysicalHasCompbits:
        return Flags::CPU_UNCACHED | Flags::GPU_NO_ACCESS | Flags::PHYSICAL | Flags::COMPRESSIBLE;
    case PoolType::CompressedVirtual:
        return Flags::CPU_NO_ACCESS | Flags::GPU_CACHED | Flags::VIRTUAL | Flags::COMPRESSIBLE;
    }
}

size_t PoolUtil::MinSize(PoolType poolType)
{
    switch (poolType) {
    default:
        __builtin_trap();
    case PoolType::Normal:
    case PoolType::Compressed:
        // Test this to make sure our ALIGN UP things work.
        return SMALL_SIZE;
    case PoolType::Physical:
    case PoolType::Virtual:
    case PoolType::CompressedPhysical:
    case PoolType::CompressedPhysicalHasCompbits:
    case PoolType::CompressedVirtual:
        return BIG_SIZE;
    }
}

bool PoolUtil::IsVirtual(PoolType poolType)
{
    return poolType == PoolType::Virtual || poolType == PoolType::CompressedVirtual;
}

bool PoolUtil::IsPhysical(PoolType poolType)
{
    return poolType == PoolType::Physical ||
        poolType == PoolType::CompressedPhysical ||
        poolType == PoolType::CompressedPhysicalHasCompbits;
}

void PoolUtil::InitPool(MemoryPoolHolder& pool, PoolType poolType)
{
    void* storage = IsVirtual(poolType) ? nullptr : &(*_storage);

    MemoryPoolBuilder mpb;
    mpb.SetDevice(g_device).SetDefaults()
        .SetFlags(MakeFlags(poolType))
        .SetStorage(storage, MinSize(poolType));
    if (!pool.Initialize(&mpb)) { __builtin_trap(); }
}

const char* PoolUtil::DescString(PoolType poolType)
{
    switch (poolType) {
    case PoolType::Normal:
        return "Normal";
    case PoolType::Compressed:
        return "Compressed";
    case PoolType::Physical:
        return "Physical";
    case PoolType::Virtual:
        return "Virtual";
    case PoolType::CompressedPhysical:
        return "CompressedPhysical";
    case PoolType::CompressedPhysicalHasCompbits:
        return "CompressedPhysicalHasCompbits";
    case PoolType::CompressedVirtual:
        return "CompressedVirtual";
    default:
        return "Invalid";
    }
}
} // llgd_lwn