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
#include <LlgdTestUtilDisplay.h>
#include <LlgdGpuState.h>

#include <functional>
#include <memory>

using namespace lwn;

namespace llgd_lwn
{
void SetupLWNMemory();
void SetupLWNMappings();

template<typename T, typename V>
class GenericHolder {
public:
    ~GenericHolder() { if (_initialized) { _t.Finalize(); _initialized = false; } }
    operator T*()  { return &_t; } // Colwerts to API call as arg.
    operator V*()  { return reinterpret_cast<V*>(&_t); } // Colwerts to c-style API call as arg.
    T& operator *()  { return _t; } // Smart ptr behavior.
    T* operator ->()  { return &_t; } // Smart ptr behavior.
    template <class... Args>
    bool Initialize(Args... args) { _initialized = true; return _t.Initialize(args...); }
    void Finalize() { if (_initialized) { _t.Finalize(); _initialized = false; } }
    // const operators
    operator const T*() const { return &_t; } // Colwerts to API call as arg.
    operator const V*() const { return reinterpret_cast<const V*>(&_t); } // Colwerts to c-style API call as arg.
    const T& operator *() const { return _t; } // Smart ptr behavior.
    const T* operator ->() const { return &_t; } // Smart ptr behavior.
protected:
    bool _initialized = false;
    T _t;
};

using MemoryPoolHolder = llgd_lwn::GenericHolder<MemoryPool, LWNmemoryPool>;
using CommandBufferHolder = llgd_lwn::GenericHolder<CommandBuffer, LWNcommandBuffer>;
using TextureHolder = llgd_lwn::GenericHolder<Texture, LWNtexture>;
using TexturePoolHolder = llgd_lwn::GenericHolder<TexturePool, LWNtexturePool>;
using SamplerPoolHolder = llgd_lwn::GenericHolder<SamplerPool, LWNsamplerPool>;
using ProgramHolder = llgd_lwn::GenericHolder<Program, LWNprogram>;
using SyncHolder = llgd_lwn::GenericHolder<Sync, LWNsync>;

class QueueHolder : public GenericHolder<Queue, LWNqueue> {
public:
    void Initialize(Device *d);
};
GpuState ExtractGpuState(LWNqueue* queue);

class DeviceHolder : public GenericHolder<Device, LWNdevice> {
public:
    void Initialize();
};

class WindowHolder : public GenericHolder<Window, LWNwindow> {
public:
    void Initialize(Device *d);
    ~WindowHolder();
private:
    void InitializeTexture(MemoryPool* mp, Texture* tex, Device* d);

    static const int        WINDOW_W = 64;
    static const int        WINDOW_H = 64;
    static const size_t     POOL_SIZE = 65536;
    LlgdUniqueUint8PtrWithLwstomDeleter spPool;

    MemoryPoolHolder mph_f, mph_b;
    TextureHolder th_f, th_b;
};

class ShaderBufferHolder : public GenericHolder<Buffer, LWNbuffer> {
public:
    void Initialize(Device *d, const void *shaderMainData, size_t mainDataSize);
private:
    LlgdUniqueUint8PtrWithLwstomDeleter m_shaderDataPool;

    MemoryPoolHolder mphShaderData;
};
} // llgd_lwn

template <typename T, typename V>
std::ostream& operator<<(std::ostream& out, const llgd_lwn::GenericHolder<T, V>& v) {
    out << static_cast<const V*>(v);
    return out;
}

extern llgd_lwn::DeviceHolder g_device;
extern llgd_lwn::DisplayUtil g_displayUtil;
