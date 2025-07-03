/*
 * Copyright (c) 2015 - 2016 LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#ifndef __lwnUtil_TexIDPool_h__
#define __lwnUtil_TexIDPool_h__

#include "lwnUtil_Interface.h"

#include "lwn/lwn.h"
#include <list>
#include <map>

namespace lwnUtil {

// Class for allocating unique integers from a limited range of values
class IndexAllocator
{
public:
    IndexAllocator();
    ~IndexAllocator();
    void Initialize(int minIndex, int size);
    int Alloc();
    void Free(int index);

private:
    int NumPoolWords() const { return (mSize + 31) / 32; }
    int mMinIndex;
    int mSize;
    uint32_t* mIndexPool;
    int mLastWord;
};

// Class for managing texture and sampler pools and for allocating out IDs
// for the same.
class TexIDPool {
    void init(LWNdevice* device);
public:
    explicit TexIDPool(LWNdevice* device) {
        init(device);
    }
#if LWNUTIL_INTERFACE_TYPE == LWNUTIL_INTERFACE_TYPE_CPP
    explicit TexIDPool(lwn::Device* device) {
        init(reinterpret_cast<LWNdevice *>(device));
    }
#endif
    ~TexIDPool();

    // Simple allocation and deallocation of IDs
    LWNuint AllocTextureID();
    LWNuint AllocSamplerID();
    void FreeTextureID(LWNuint id);
    void FreeSamplerID(LWNuint id);

    // Allocation/deallocation of IDs, plus registration. LWN doesn't actually
    // deregister IDs, but we need a hook for maintaining the mapping between
    // objects and IDs, so let's just pretend.
    LWNuint Register(const LWNtexture* texture, const LWNtextureView *view = NULL);
    LWNuint RegisterImage(const LWNtexture* texture, const LWNtextureView *view = NULL);
    LWNuint Register(const LWNsampler* sampler);
    LWNuint Register(const LWNsamplerBuilder* builder);
    void Deregister(const LWNtexture* texture);
    void DeregisterImage(const LWNtexture* texture);
    void Deregister(const LWNsampler* sampler);

    // Bind the texture and sampler pool objects in this class to the queue
    // using <cmdBuf>.
    void Bind(LWNcommandBuffer *cmdBuf);

    // Texture and sampler pool queries either return a pointer to the pool
    // objects, using either the native C interface or the native C++
    // interface via reinterpret_cast.
#if LWNUTIL_INTERFACE_TYPE == LWNUTIL_INTERFACE_TYPE_C
    const LWNtexturePool *GetTexturePool() const { return &mAPITexturePool; }
    const LWNsamplerPool *GetSamplerPool() const { return &mAPISamplerPool; }
#else
    const lwn::TexturePool *GetTexturePool() const
    {
        return reinterpret_cast<const lwn::TexturePool *>(&mAPITexturePool);
    }
    const lwn::SamplerPool *GetSamplerPool() const
    {
        return reinterpret_cast<const lwn::SamplerPool *>(&mAPISamplerPool);
    }
#endif

#if LWNUTIL_INTERFACE_TYPE == LWNUTIL_INTERFACE_TYPE_CPP
    //
    // Methods to provide a native C++ interface to the core TexIDPool class,
    // using reinterpret_cast to colwert between C and C++ object types.
    //
    LWNuint Register(const lwn::Texture *texture, const lwn::TextureView *view = NULL)
    {
        const LWNtexture *ctexture = reinterpret_cast<const LWNtexture *>(texture);
        const LWNtextureView *cview = reinterpret_cast<const LWNtextureView *>(view);
        return Register(ctexture, cview);
    }
    LWNuint RegisterImage(const lwn::Texture* texture, const lwn::TextureView *view = NULL)
    {
        const LWNtexture *ctexture = reinterpret_cast<const LWNtexture *>(texture);
        const LWNtextureView *cview = reinterpret_cast<const LWNtextureView *>(view);
        return RegisterImage(ctexture, cview);
    }
    LWNuint Register(const lwn::Sampler* sampler)
    {
        const LWNsampler *csampler = reinterpret_cast<const LWNsampler *>(sampler);
        return Register(csampler);
    }
    LWNuint Register(const lwn::SamplerBuilder* builder)
    {
        const LWNsamplerBuilder *cbuilder = reinterpret_cast<const LWNsamplerBuilder *>(builder);
        return Register(cbuilder);
    }
    void Deregister(const lwn::Texture* texture)
    {
        const LWNtexture *ctexture = reinterpret_cast<const LWNtexture *>(texture);
        return Deregister(ctexture);
    }
    void DeregisterImage(const lwn::Texture* texture)
    {
        const LWNtexture *ctexture = reinterpret_cast<const LWNtexture *>(texture);
        return DeregisterImage(ctexture);
    }
    void Deregister(const lwn::Sampler* sampler)
    {
        const LWNsampler *csampler = reinterpret_cast<const LWNsampler *>(sampler);
        return Deregister(csampler);
    }
    void Bind(lwn::CommandBuffer *cmdBuf)
    {
        LWNcommandBuffer *ccb = reinterpret_cast<LWNcommandBuffer *>(cmdBuf);
        Bind(ccb);
    }
#endif
    enum {
        NUM_PUBLIC_TEXTURES = 2000,
        NUM_PUBLIC_SAMPLERS = 2000
    };

private:
    // Textures can be registered as textures or as images in the pool, so we
    // need to distinguish between the two use cases.
    enum TextureRegistrationType {
        RegisteredAsTexture,
        RegisteredAsImage,
    };

    // For our map of textures, we index by both the texture and registration
    // type.  The "<" operator is needed for std::map.
    struct RegisteredTexture {
        const LWNtexture *texture;
        TextureRegistrationType registrationType;
        bool operator <(const RegisteredTexture &other) const
        {
            if (texture < other.texture) return true;
            if (texture > other.texture) return false;
            return registrationType < other.registrationType;
        }
    };

    // For our map of samplers, we index just by the sampler object, but use a
    // structure to match texture usage.  The "<" operator is needed for
    // std::map.
    struct RegisteredSampler {
        const LWNsampler *sampler;
        bool operator < (const RegisteredSampler &other) const
        {
            return sampler < other.sampler;
        }
    };

    typedef std::list<LWNuint> RegistrationList;
    typedef std::map<RegisteredTexture, RegistrationList> TextureIDMap;
    typedef std::map<RegisteredSampler, RegistrationList> SamplerIDMap;

    LWNdevice* mDevice;
    LWNint mNumReservedTextures;
    LWNint mNumReservedSamplers;
    void* mPoolMemory;
    LWNmemoryPool mDescriptorPool;
    LWNtexturePool mAPITexturePool;
    LWNsamplerPool mAPISamplerPool;
    IndexAllocator mTextureAllocator;
    IndexAllocator mSamplerAllocator;
    TextureIDMap mTextureIDMap;
    SamplerIDMap mSamplerIDMap;
};

} // namespace lwnUtil

#endif // #ifndef __lwnUtil_TexIDPool_h__
