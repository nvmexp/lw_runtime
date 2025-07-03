/*
 * Copyright (c) 2015 - 2016 LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#ifndef __lwnUtil_TexIDPoolImpl_h__
#define __lwnUtil_TexIDPoolImpl_h__

#include "lwnUtil_AlignedStorage.h"
#include "lwnUtil_PoolAllocator.h"
#include "lwnUtil_TexIDPool.h"

namespace lwnUtil {

IndexAllocator::IndexAllocator()
{
    mMinIndex = 0;
    mSize = 0;
    mIndexPool = NULL;
    mLastWord = 0;
}

IndexAllocator::~IndexAllocator()
{
    delete[] mIndexPool;
}

void IndexAllocator::Initialize(int minIndex, int size)
{
    mMinIndex = minIndex;
    mSize = size;
    mIndexPool = new uint32_t[NumPoolWords()];
    memset(mIndexPool, 0, NumPoolWords() * sizeof(uint32_t));
    mLastWord = 0;
}

int IndexAllocator::Alloc()
{
    // As a heuristic, try to allocate conselwtive IDs in different words. This spreads the allocations
    // throughout the pool and hopefully makes finding a free slot faster.
    assert(mIndexPool);
    int wordIndex;
    // Bit pattern we'd expect in the last word of the map
    uint32_t fullLastWord = (mSize % 32) ? (1UL << (mSize % 32)) - 1 : 0xffffffff;
    for (wordIndex = (mLastWord + 1) % NumPoolWords(); wordIndex != mLastWord;
         wordIndex = (wordIndex + 1) % NumPoolWords()) {
        if ((mIndexPool[wordIndex] != 0xffffffff && wordIndex < NumPoolWords() - 1) ||
            (mIndexPool[wordIndex] != fullLastWord && wordIndex == NumPoolWords() - 1)) {
            break;
        }
    }
    if (wordIndex == mLastWord) {
        // You're gonna need a bigger boat.
        assert(!"IndexAllocator capacity exhausted.");
        return 0;
    }
    uint32_t word = mIndexPool[wordIndex];
    int bitIndex = 0;
    if ((word & 0xffff) == 0xffff) {
        bitIndex += 16;
        word >>= 16;
    }
    if ((word & 0xff) == 0xff) {
        bitIndex += 8;
        word >>= 8;
    }
    if ((word & 0xf) == 0xf) {
        bitIndex += 4;
        word >>= 4;
    }
    if ((word & 0x3) == 0x3) {
        bitIndex += 2;
        word >>= 2;
    }
    if ((word & 0x1) == 0x1) {
        bitIndex += 1;
    }
    mIndexPool[wordIndex] |= (0x1 << bitIndex);
    mLastWord = wordIndex;
    return wordIndex * 32 + bitIndex + mMinIndex;
}

void IndexAllocator::Free(int index)
{
    index -= mMinIndex;
    assert(index >= 0);
    assert(index < mSize);
    int wordIndex = index / 32;
    int bitIndex = index % 32;
    mIndexPool[wordIndex] &= ~(1 << bitIndex);
}

void TexIDPool::init(LWNdevice* device)
{
    mDevice = device;
    LWNint textureSize, samplerSize;
    lwnDeviceGetInteger(device, LWN_DEVICE_INFO_TEXTURE_DESCRIPTOR_SIZE, &textureSize);
    lwnDeviceGetInteger(device, LWN_DEVICE_INFO_SAMPLER_DESCRIPTOR_SIZE, &samplerSize);
    lwnDeviceGetInteger(device, LWN_DEVICE_INFO_RESERVED_TEXTURE_DESCRIPTORS, &mNumReservedTextures);
    lwnDeviceGetInteger(device, LWN_DEVICE_INFO_RESERVED_SAMPLER_DESCRIPTORS, &mNumReservedSamplers);
    LWNuint targetSize = (mNumReservedSamplers + NUM_PUBLIC_SAMPLERS) * samplerSize +
                         (mNumReservedTextures + NUM_PUBLIC_TEXTURES) * textureSize;

    size_t poolSize = PoolStorageSize(targetSize);
    mPoolMemory = PoolStorageAlloc(poolSize);

    LWNmemoryPoolBuilder builder;
    lwnMemoryPoolBuilderSetDefaults(&builder);
    lwnMemoryPoolBuilderSetDevice(&builder, device);
    lwnMemoryPoolBuilderSetStorage(&builder, mPoolMemory, poolSize);
#if defined(_WIN32)
    lwnMemoryPoolBuilderSetFlags(&builder, (LWN_MEMORY_POOL_FLAGS_CPU_NO_ACCESS_BIT |
                                            LWN_MEMORY_POOL_FLAGS_GPU_CACHED_BIT));
#else
    lwnMemoryPoolBuilderSetFlags(&builder, (LWN_MEMORY_POOL_FLAGS_CPU_UNCACHED_BIT |
                                            LWN_MEMORY_POOL_FLAGS_GPU_CACHED_BIT));
#endif
    lwnMemoryPoolInitialize(&mDescriptorPool, &builder);

    lwnTexturePoolInitialize(&mAPITexturePool, &mDescriptorPool,
                             0, mNumReservedSamplers + NUM_PUBLIC_SAMPLERS);
    lwnSamplerPoolInitialize(&mAPISamplerPool, &mDescriptorPool, 
                             (mNumReservedSamplers + NUM_PUBLIC_SAMPLERS) * samplerSize,
                             mNumReservedTextures + NUM_PUBLIC_TEXTURES);

    mTextureAllocator.Initialize(mNumReservedTextures, NUM_PUBLIC_TEXTURES);
    mSamplerAllocator.Initialize(mNumReservedSamplers, NUM_PUBLIC_SAMPLERS);
}

TexIDPool::~TexIDPool()
{
    lwnSamplerPoolFinalize(&mAPISamplerPool);
    lwnTexturePoolFinalize(&mAPITexturePool);
    lwnMemoryPoolFinalize(&mDescriptorPool);
    PoolStorageFree(mPoolMemory);
}

LWNuint TexIDPool::AllocTextureID()
{
    return mTextureAllocator.Alloc();
}

LWNuint TexIDPool::AllocSamplerID()
{
    return mSamplerAllocator.Alloc();
}

void TexIDPool::FreeTextureID(LWNuint id)
{
    if (id == 0) {
        return;
    }
    mTextureAllocator.Free(id);
}

void TexIDPool::FreeSamplerID(LWNuint id)
{
    if (id == 0) {
        return;
    }
    mSamplerAllocator.Free(id);
}

LWNuint TexIDPool::Register(const LWNtexture* texture, const LWNtextureView *view /* = NULL */)
{
    assert(texture);
    LWNuint id = AllocTextureID();
    lwnTexturePoolRegisterTexture(&mAPITexturePool, id, texture, view);
    RegisteredTexture registration = { texture, RegisteredAsTexture };
    RegistrationList &regList = mTextureIDMap[registration];
    regList.push_back(id);
    return id;
}

LWNuint TexIDPool::RegisterImage(const LWNtexture* texture, const LWNtextureView *view /* = NULL */)
{
    assert(texture);
    LWNuint id = AllocTextureID();
    lwnTexturePoolRegisterImage(&mAPITexturePool, id, texture, view);
    RegisteredTexture registration = { texture, RegisteredAsImage };
    RegistrationList &regList = mTextureIDMap[registration];
    regList.push_back(id);
    return id;
}

LWNuint TexIDPool::Register(const LWNsampler* sampler)
{
    assert(sampler);
    LWNuint id = AllocSamplerID();
    lwnSamplerPoolRegisterSampler(&mAPISamplerPool, id, sampler);
    RegisteredSampler registration = { sampler };
    RegistrationList &regList = mSamplerIDMap[registration];
    regList.push_back(id);
    return id;
}

LWNuint TexIDPool::Register(const LWNsamplerBuilder* builder)
{
    assert(builder);
    LWNuint id = AllocSamplerID();
    lwnSamplerPoolRegisterSamplerBuilder(&mAPISamplerPool, id, builder);
    // Don't register sampler builder state in the registration list
    // (mSamplerIDMap); entries registered using this API need to be freed
    // explicitly.
    return id;
}

void TexIDPool::Deregister(const LWNtexture* texture)
{
    if (!texture) {
        return;
    }
    RegisteredTexture registration = { texture, RegisteredAsTexture };
    TextureIDMap::iterator it = mTextureIDMap.find(registration);
    assert(it != mTextureIDMap.end());
    RegistrationList &regList = it->second;
    for (RegistrationList::iterator rit = regList.begin(); rit != regList.end(); rit++) {
        FreeTextureID(*rit);
    }
    mTextureIDMap.erase(it);
}

void TexIDPool::Bind(LWNcommandBuffer *cmdBuf)
{
    lwnCommandBufferSetTexturePool(cmdBuf, &mAPITexturePool);
    lwnCommandBufferSetSamplerPool(cmdBuf, &mAPISamplerPool);
}

void TexIDPool::DeregisterImage(const LWNtexture* texture)
{
    if (!texture) {
        return;
    }
    RegisteredTexture registration = { texture, RegisteredAsImage };
    TextureIDMap::iterator it = mTextureIDMap.find(registration);
    if (it == mTextureIDMap.end()) {
        // We don't register all textures for image use.
        return;
    }
    RegistrationList &regList = it->second;
    for (RegistrationList::iterator rit = regList.begin(); rit != regList.end(); rit++) {
        FreeTextureID(*rit);
    }
    mTextureIDMap.erase(it);
}

void TexIDPool::Deregister(const LWNsampler* sampler)
{
    if (!sampler) {
        return;
    }
    RegisteredSampler registration = { sampler };
    SamplerIDMap::iterator it = mSamplerIDMap.find(registration);
    assert(it != mSamplerIDMap.end());
    RegistrationList &regList = it->second;
    for (RegistrationList::iterator rit = regList.begin(); rit != regList.end(); rit++) {
        FreeSamplerID(*rit);
    }
    mSamplerIDMap.erase(it);
}

} // namespace lwnUtil

#endif // #ifndef __lwnUtil_TexIDPoolImpl_h__
