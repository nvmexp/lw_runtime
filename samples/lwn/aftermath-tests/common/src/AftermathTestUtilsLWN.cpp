/*
 * Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 */

#include <AftermathTestUtilsLWN.h>
#include <AftermathTestLogging.h>

#include <vector>

#include "lwn/lwn.h"
#include <lwn/lwn_CppFuncPtrImpl.h>

#include <nn/mem/mem_StandardAllocator.h>
#include <nn/nn_Log.h>
#include <nn/vi.h>
#include <lw/lw_MemoryManagement.h>
#include <lw/lw_ServiceName.h>
#include <lwassert.h>

extern "C" PFNLWNGENERICFUNCPTRPROC LWNAPIENTRY lwnBootstrapLoader(const char *name);

AftermathTest::LWN::DeviceHolder g_device;
AftermathTest::LWN::DisplayUtil g_displayUtil;

namespace AftermathTest {
namespace LWN {

static const uint32_t CLIENT_HEAP_SIZE    = 32 << 20;
static const uint32_t FIRMWARE_HEAP_SIZE  =  8 << 20;

static char store[CLIENT_HEAP_SIZE];
static char donation[FIRMWARE_HEAP_SIZE]
            __attribute__((aligned(4096)));

static nn::mem::StandardAllocator grAlloc(store, CLIENT_HEAP_SIZE);

// strip out usr ptr
static void* halloc(size_t s, size_t algn, void* x) { return grAlloc.Allocate(s, algn);    }
static void  hfree (void* addr, void* x)            { grAlloc.Free(addr);                  }
static void* hreloc(void* addr, size_t ns, void* x) { return grAlloc.Reallocate(addr, ns); }

void SetupLWNGraphics()
{
    lw::SetGraphicsAllocator(halloc, hfree, hreloc, NULL);
    lw::SetGraphicsDevtoolsAllocator(halloc, hfree, hreloc, NULL);
    lw::InitializeGraphics(donation, FIRMWARE_HEAP_SIZE);
}

static void SetupLWNMappings()
{
    GenericFuncPtrFunc getProcAddress =
        (GenericFuncPtrFunc) ((*lwnBootstrapLoader)("lwnDeviceGetProcAddress"));
    LW_ASSERT(getProcAddress);

    DeviceGetProcAddressFunc cppGetProcAddress =
        reinterpret_cast<DeviceGetProcAddressFunc>(getProcAddress);
    lwnLoadCPPProcs(NULL, cppGetProcAddress);
}

void SetupLWNDevice(DeviceFlagBits flags)
{
    SetupLWNMappings();
    g_device.Initialize(flags);
}

void ShutdownLWNDevice()
{
    g_device.Finalize();
}

static void LWNAPIENTRY nn_log_debug(
    DebugCallbackSource::Enum source, DebugCallbackType::Enum type, int id,
    DebugCallbackSeverity::Enum severity, const char *message, void* userParam)
{
    NN_LOG("debug layer: %s\n", message);
}

DeviceHolder::~DeviceHolder()
{
    Finalize();
}

void DeviceHolder::Initialize(DeviceFlagBits flags)
{
    int version;

    DeviceBuilder db;
    db.SetDefaults();
    db.SetFlags(flags);

    LW_ASSERT(object.Initialize(&db));
    _initialized = true;

    if ((flags & DeviceFlagBits::DEBUG_ENABLE_LEVEL_0) ||
        (flags & DeviceFlagBits::DEBUG_ENABLE_LEVEL_1) ||
        (flags & DeviceFlagBits::DEBUG_ENABLE_LEVEL_2) ||
        (flags & DeviceFlagBits::DEBUG_ENABLE_LEVEL_3) ||
        (flags & DeviceFlagBits::DEBUG_ENABLE_LEVEL_4)) {
        object.InstallDebugCallback(nn_log_debug, NULL, LWN_TRUE);
    }

    auto getProcAddress = lwnBootstrapLoader("lwnDeviceGetProcAddress");
    LW_ASSERT(getProcAddress);

    auto cppGetProcAddress = reinterpret_cast<DeviceGetProcAddressFunc>(getProcAddress);
    lwnLoadCPPProcs(&object, cppGetProcAddress);

    object.GetInteger(DeviceInfo::API_MAJOR_VERSION, &version);
    LW_ASSERT(version == LWN_API_MAJOR_VERSION);

    object.GetInteger(DeviceInfo::API_MINOR_VERSION, &version);
    LW_ASSERT(version == LWN_API_MINOR_VERSION);
}

QueueHolder::~QueueHolder()
{
    Finalize();
}

void QueueHolder::Initialize(Device *d)
{
    QueueBuilder qb;

    qb.SetDevice(d);
    qb.SetDefaults();
    LW_ASSERT(object.Initialize(&qb));
    _initialized = true;
}

WindowHolder::~WindowHolder()
{
    Finalize();
}

void WindowHolder::InitializeTexture(ptrdiff_t poolOffset, TextureHolder& tex, Device* d)
{
    TextureBuilder tex_builder;
    tex_builder.SetDevice(d).SetDefaults()
        .SetFlags(TextureFlags::DISPLAY | TextureFlags::COMPRESSIBLE)
        .SetTarget(TextureTarget::TARGET_2D)
        .SetSize2D(WINDOW_W, WINDOW_H)
        .SetFormat(Format::RGBA8);
    tex_builder.SetStorage(mph, poolOffset);
    LW_ASSERT(tex.Initialize(&tex_builder));
}

void WindowHolder::Initialize(Device *d)
{
    // Initialize nn::vi and get the native window
    g_displayUtil.Initialize();

    MemoryPoolBuilder pool_builder;
    pool_builder.SetDevice(d).SetDefaults()
        .SetFlags(MemoryPoolFlags::CPU_NO_ACCESS | MemoryPoolFlags::GPU_CACHED | MemoryPoolFlags::COMPRESSIBLE);
    pool_builder.SetStorage(pool, POOL_SIZE_IN_BYTES);
    LW_ASSERT(mph.Initialize(&pool_builder));

    InitializeTexture(0, th_f, d);
    InitializeTexture(POOL_SIZE_IN_BYTES / 2, th_b, d);

    Texture* textures[2];
    textures[0] = th_f;
    textures[1] = th_b;

    WindowBuilder win_builder;
    win_builder.SetDevice(d);
    win_builder.SetDefaults();
    win_builder.SetNativeWindow(g_displayUtil.GetNativeWindow());
    win_builder.SetTextures(2, textures);
    LW_ASSERT(object.Initialize(&win_builder));
    _initialized = true;
}

void WindowHolder::Finalize()
{
    if (_initialized) {
        // These objects must be finalized before shutting down nn::vi
        object.Finalize();
        th_f.Finalize();
        th_b.Finalize();
        mph.Finalize();

        g_displayUtil.Finalize();

        _initialized = false;
    }
}

ShaderBufferHolder::~ShaderBufferHolder()
{
    Finalize();
}

void ShaderBufferHolder::Initialize(Device *d, const void *shaderMainData, size_t mainDataSize)
{
    // 1. Allocate aligned storage
    auto alignedPoolSize = Utils::AlignUp(mainDataSize, LWN_MEMORY_POOL_STORAGE_GRANULARITY);
    shaderDataPool = AlignedAllocPodType<uint8_t>(alignedPoolSize, LWN_MEMORY_POOL_STORAGE_GRANULARITY);

    // 2. Copy shader data
    memcpy(reinterpret_cast<void *>(shaderDataPool.get()), shaderMainData, mainDataSize);

    // 3. Create a memory pool for shader main data
    MemoryPoolBuilder mpb;
    mpb.SetDevice(d)
       .SetDefaults()
       .SetFlags(MemoryPoolFlags::CPU_NO_ACCESS | MemoryPoolFlags::GPU_CACHED | MemoryPoolFlags::SHADER_CODE)
       .SetStorage(shaderDataPool.get(), alignedPoolSize);
    mphShaderData.Initialize(&mpb);

    // 4. Create a Buffer and combine to the memory pools
    BufferBuilder bufferBuilder;
    bufferBuilder
        .SetDevice(d)
        .SetDefaults()
        .SetStorage(mphShaderData, 0, mainDataSize);
    object.Initialize(&bufferBuilder);

    _initialized = true;
}

void ShaderBufferHolder::Finalize()
{
    if (_initialized) {
        object.Finalize();
        mphShaderData.Finalize();
        shaderDataPool = nullptr;
        _initialized = false;
    }
}

} // namespace LWN
} // namespace AftermathTest
