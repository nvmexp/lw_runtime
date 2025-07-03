/*
 * Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
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
#include <lwn/lwn_DeviceConstantsNX.h>

#include <AftermathTestUtils.h>
#include <AftermathTestUtilsDisplay.h>

#include <functional>
#include <memory>

using namespace lwn;

namespace AftermathTest {
namespace LWN {

void SetupLWNGraphics();
void SetupLWNDevice(DeviceFlagBits flags);
void ShutdownLWNDevice();

// Allocate aligned buffer of POD type (no constructor, destructor will be called)
//
// Note: Use this function to allocate aligned storage as possible.
// Don't use std::aligned_storage in wrong way for memory pool.
// We need to follow strict aliasing rules in C++ to use aligned_storage:
//  https://gist.github.com/shafik/848ae25ee209f698763cffee272a58f8
//  https://whereswalden.com/tag/stdaligned_storage/
// Correct way to use aligned_storage is like that:
//   auto spStorage = make_unique<typename aligned_storage<SIZE,ALIGN>::type>();
//   auto storage = new(spStorage.get()) uint8_t[SIZE];  // replacement new
// Otherwise, the behavior will be undefined. So use this function instead if you wouldn't like to write code above.
template <typename Pod_T>
std::unique_ptr<Pod_T, std::function<void(Pod_T*)>> AlignedAllocPodType(size_t sizeInBytes, size_t alignment)
{
    // Allocate memory storage which is aligned to a given argument
    auto alignPtr = [](Pod_T *ptr, uint64_t alignment) {
        return reinterpret_cast<Pod_T*>(Utils::AlignUp(reinterpret_cast<uint64_t>(ptr), alignment));
    };

    // Raw storage
    auto storage = reinterpret_cast<Pod_T*>(std::malloc(sizeInBytes + alignment));

    // Aligned storage
    auto alignedStorage = alignPtr(storage, alignment);

    // Wrap by smart pointer. Storage will be deleted automatically
    using UniqueWithDeleter = std::unique_ptr<Pod_T, std::function<void(Pod_T*)>>;
    return UniqueWithDeleter(alignedStorage, [storage](Pod_T*) { std::free(storage); });
}

// Maybe used from various places
using UniqueUint8PtrWithLwstomDeleter = std::unique_ptr<uint8_t, std::function<void(uint8_t*)>>;

template<typename T, typename V>
class GenericHolder {
public:
    virtual ~GenericHolder() { Finalize(); }
    operator T*()  { return &object; } // Colwerts to API call as arg.
    operator V*()  { return reinterpret_cast<V*>(&object); } // Colwerts to c-style API call as arg.
    T& operator *()  { return object; } // Smart ptr behavior.
    T* operator ->()  { return &object; } // Smart ptr behavior.
    template <class... Args>
    bool Initialize(Args... args) { _initialized = true; return object.Initialize(args...); }
    void Finalize() { if (_initialized) { object.Finalize(); _initialized = false; } }
    // const operators
    operator const T*() const { return &object; } // Colwerts to API call as arg.
    operator const V*() const { return reinterpret_cast<const V*>(&object); } // Colwerts to c-style API call as arg.
    const T& operator *() const { return object; } // Smart ptr behavior.
    const T* operator ->() const { return &object; } // Smart ptr behavior.
protected:
    bool _initialized = false;
    T object;
};

using BufferHolder = GenericHolder<Buffer, LWNbuffer>;
using MemoryPoolHolder = GenericHolder<MemoryPool, LWNmemoryPool>;
using CommandBufferHolder = GenericHolder<CommandBuffer, LWNcommandBuffer>;
using TextureHolder = GenericHolder<Texture, LWNtexture>;
using TexturePoolHolder = GenericHolder<TexturePool, LWNtexturePool>;
using SamplerHolder = GenericHolder<Sampler, LWNsampler>;
using SamplerPoolHolder = GenericHolder<SamplerPool, LWNsamplerPool>;
using ProgramHolder = GenericHolder<Program, LWNprogram>;
using SyncHolder = GenericHolder<Sync, LWNsync>;

class QueueHolder : public GenericHolder<Queue, LWNqueue> {
public:
    virtual ~QueueHolder();
    void Initialize(Device* d);
};

class DeviceHolder : public GenericHolder<Device, LWNdevice> {
public:
    virtual ~DeviceHolder();
    void Initialize(DeviceFlagBits flags);
};

class WindowHolder : public GenericHolder<Window, LWNwindow> {
public:
    virtual ~WindowHolder();
    void Initialize(Device* d);
    void Finalize();
    int GetWidth() { return WINDOW_W; }
    int GetHeight() { return WINDOW_H; }
    Texture* GetColorRt(int index) { return index == 0 ? th_f : th_b; }
private:
    void InitializeTexture(ptrdiff_t poolOffset, TextureHolder& tex, Device* d);

    static const int        WINDOW_W = 256;
    static const int        WINDOW_H = 256;
    static const size_t     POOL_SIZE_IN_BYTES = 2 * WINDOW_W * WINDOW_H * 4; // RGBA8 front + back buffer
    uint8_t                 pool[POOL_SIZE_IN_BYTES] __attribute__((aligned(4096)));

    MemoryPoolHolder mph;
    TextureHolder th_f, th_b;
};

class ShaderBufferHolder : public BufferHolder {
public:
    virtual ~ShaderBufferHolder();
    void Initialize(Device* d, const void* shaderMainData, size_t mainDataSize);
    void Finalize();

private:
    UniqueUint8PtrWithLwstomDeleter shaderDataPool;
    MemoryPoolHolder mphShaderData;
};
} // namespace LWN
} // namespace AftermathTest

template <typename T, typename V>
std::ostream& operator<<(std::ostream& out, const AftermathTest::LWN::GenericHolder<T, V>& v) {
    out << static_cast<const V*>(v);
    return out;
}

extern AftermathTest::LWN::DeviceHolder g_device;
extern AftermathTest::LWN::DisplayUtil g_displayUtil;
