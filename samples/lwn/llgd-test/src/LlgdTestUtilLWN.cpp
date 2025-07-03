/*
 * Copyright (c) 2017-2020, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 */

#include <LlgdTestUtilLWN.h>
#include <LlgdTestUtil.h>
#include <lwndevtools_bootstrap.h>

#include <vector>

#include <lwn/lwn.h>
#include <lwn/lwn_CppFuncPtrImpl.h>

#if defined(LW_HOS)
#include <nn/mem/mem_StandardAllocator.h>
#include <nn/nn_Log.h>
#include <nn/vi.h>
#include <lw/lw_MemoryManagement.h>
#include <lw/lw_ServiceName.h>
#endif

extern "C" PFNLWNGENERICFUNCPTRPROC LWNAPIENTRY lwnBootstrapLoader(const char *name);

llgd_lwn::DeviceHolder g_device;
llgd_lwn::DisplayUtil g_displayUtil;
#define DEBUG_LAYER 0

namespace llgd_lwn
{
#if defined(LW_HOS)
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

void SetupLWNMemory()
{
    lw::SetGraphicsAllocator(halloc, hfree, hreloc, NULL);
    lw::SetGraphicsDevtoolsAllocator(halloc, hfree, hreloc, NULL);
    lw::InitializeGraphics(donation, FIRMWARE_HEAP_SIZE);
}
#endif

void SetupLWNMappings()
{
    GenericFuncPtrFunc getProcAddress =
        (GenericFuncPtrFunc) ((*lwnBootstrapLoader)("lwnDeviceGetProcAddress"));
    CHECK(getProcAddress);

    DeviceGetProcAddressFunc cppGetProcAddress =
        reinterpret_cast<DeviceGetProcAddressFunc>(getProcAddress);
    lwnLoadCPPProcs(NULL, cppGetProcAddress);
}

GpuState ExtractGpuState(LWNqueue* queue)
{
    static const LWNdevtoolsBootstrapFunctions* devtools = lwnDevtoolsBootstrap();

    // Setup GPU state from the HW mme register shadow in the ctx state blob
    size_t ctxSize = 0;
    devtools->GetGrCtxSizeForQueue(queue, &ctxSize);

    std::vector<uint8_t> ctxVector;
    ctxVector.resize(ctxSize);
    uint8_t* ctxBuffer = ctxVector.data();
    devtools->GetGrCtxForQueue(queue, ctxBuffer, ctxSize);

    const uint8_t* mme = &ctxBuffer[8704];

    MmeShadowRegistersMemBanks<uint32_t> mmeShadowRegisterMemBanks;
    memcpy(mmeShadowRegisterMemBanks.bank0, mme, 1024);
    mme += 1032;
    memcpy(mmeShadowRegisterMemBanks.bank1, mme, 1024);
    mme += 1032;
    memcpy(mmeShadowRegisterMemBanks.bank2, mme, 1024);
    mme += 1032;
    memcpy(mmeShadowRegisterMemBanks.bank3, mme, 1024);
    mme += 1032;
    memcpy(mmeShadowRegisterMemBanks.bank4, mme, 1024);
    mme += 1032;
    memcpy(mmeShadowRegisterMemBanks.bank5, mme, 280);

    // LlgdGpuStateExtractor.cpp : ColwertInitialStateToMmeRegisters
#define LLGD_FAKE_METHOD_INDEX_TO_BANK_OFFSET(_index) (_index - LLGD_FAKE_METHOD_INDEX_BASE + 3)
    const auto ReadUint32 = [](const uint8_t* bytes, size_t offset) -> uint32_t {
        uint32_t method = 0;
        memcpy(&method, bytes + offset, sizeof(method));
        return method;
    };
    // Extract PRI_SETUP_DEBUG_CONSERVATIVE_RASTER_BLOAT
    const uint32_t gpcsSetupDebug = ReadUint32(ctxBuffer, 0x2204);
    mmeShadowRegisterMemBanks.bank6[LLGD_FAKE_METHOD_INDEX_TO_BANK_OFFSET(LLGD_FAKE_METHOD_INDEX_PRI_CONSERVATIVE_RASTER_BLOAT)] = (gpcsSetupDebug >> 23) & 0x3;
    // Extract alpha to coverage dithering enable state
    const uint32_t alphaToCoverageDitheringEnable = ReadUint32(ctxBuffer, 0x7684);
    mmeShadowRegisterMemBanks.bank6[LLGD_FAKE_METHOD_INDEX_TO_BANK_OFFSET(LLGD_FAKE_METHOD_INDEX_ALPHA_TO_COVERAGE_DITHERING_ENABLE)] = alphaToCoverageDitheringEnable;

    // TODO: (https://jirasw.lwpu.com/browse/LLGD-2386) Extract states for testing zf32DepthClearValueIdx, ZF32AsZ16Disabled
    // LLGD_FAKE_METHOD_INDEX_ZLWLL_ZF32_COMPRESSION_ENABLE (stored in LWNqueue), LLGD_FAKE_METHOD_INDEX_ZF32_DEPTH_CLEAR_VALUE_IDX (stored in LWNqueue)

    // Extract Render enable states
    const uint32_t renderEnableA = ReadUint32(ctxBuffer, 0x354);
    const uint32_t renderEnableB = ReadUint32(ctxBuffer, 0x358);
    const uint32_t renderEnable = ReadUint32(ctxBuffer, 0x374);
    // Store special state in fake bank 6
    mmeShadowRegisterMemBanks.bank6[0] = renderEnableA;
    mmeShadowRegisterMemBanks.bank6[1] = renderEnableB;
    // See LlgdLwnState.h for encoding explanation.  Bit 0 is the 3D enable bit.
    mmeShadowRegisterMemBanks.bank6[2] = (renderEnable & 0x1) + 5;

#undef LLGD_FAKE_METHOD_INDEX_TO_BANK_OFFSET

    GpuState state;
    state.mmeShadowRegisters.Initialize(mmeShadowRegisterMemBanks);

    return state;
}

#if DEBUG_LAYER
static void LWNAPIENTRY llgd_test_log_debug(
    DebugCallbackSource::Enum source, DebugCallbackType::Enum type, int id,
    DebugCallbackSeverity::Enum severity, const char *message, void* userParam)
{
    LLGD_TEST_LOG("debug layer: %s\n", message);
}
#endif

void DeviceHolder::Initialize()
{
    int version;

    DeviceBuilder db;
    db.SetDefaults();
    auto flags = DeviceFlagBits::ENABLE_SEPARATE_SAMPLER_TEXTURE_SUPPORT;
#if DEBUG_LAYER
    flags |= DeviceFlagBits::DEBUG_ENABLE_LEVEL_4;
#endif
    db.SetFlags(flags);

    CHECK(_t.Initialize(&db));
    _initialized = true;

#if DEBUG_LAYER
    _t.InstallDebugCallback(llgd_test_log_debug, NULL, LWN_TRUE);

    auto getProcAddress = lwnBootstrapLoader("lwnDeviceGetProcAddress");
    CHECK(getProcAddress);

    auto cppGetProcAddress = reinterpret_cast<DeviceGetProcAddressFunc>(getProcAddress);
    lwnLoadCPPProcs(&_t, cppGetProcAddress);
#endif

#define __CHK_VERSION(x) \
    _t.GetInteger(DeviceInfo::API_##x##_VERSION, &version);\
    CHECK(version == LWN_API_##x##_VERSION);

    __CHK_VERSION(MAJOR);
    __CHK_VERSION(MINOR);

#undef __CHK_VERSION
}

void QueueHolder::Initialize(Device *d)
{
    QueueBuilder qb;

    qb.SetDevice(d);
    qb.SetDefaults();
    CHECK(_t.Initialize(&qb));
    _initialized = true;

    // Some tests expect state to reach the HW
    // at the end of Initialize, and query it
    // without Finish()ing. Modify semantics of QH.Initialize
    // to help with this assumption.
    _t.Finish();
}

WindowHolder::~WindowHolder()
{
    if (_initialized)
    {
        // These objects must be finalized before shutting down nn::vi
        _t.Finalize();
        th_f->Finalize(); th_b->Finalize();
        mph_f->Finalize(); mph_b->Finalize();

        g_displayUtil.Finalize();

        _initialized = false;
    }
}

void WindowHolder::InitializeTexture(MemoryPool* mp, Texture* tex, Device* d)
{
    spPool = LlgdAlignedAllocPodType<uint8_t>(POOL_SIZE, 4096);

    MemoryPoolBuilder pool_builder;
    pool_builder.SetDevice(d).SetDefaults()
        .SetFlags(MemoryPoolFlags::CPU_NO_ACCESS | MemoryPoolFlags::GPU_CACHED | MemoryPoolFlags::COMPRESSIBLE);
    pool_builder.SetStorage(spPool.get(), POOL_SIZE);
    CHECK(mp->Initialize(&pool_builder));

    TextureBuilder tex_builder;
    tex_builder.SetDevice(d).SetDefaults()
        .SetFlags(TextureFlags::DISPLAY | TextureFlags::COMPRESSIBLE)
        .SetTarget(TextureTarget::TARGET_2D)
        .SetSize2D(WINDOW_W, WINDOW_H)
        .SetFormat(Format::RGBA8);
    tex_builder.SetStorage(mp, 0);
    CHECK(tex->Initialize(&tex_builder));
}

void WindowHolder::Initialize(Device *d)
{
    // Initialize nn::vi and get the native window
    g_displayUtil.Initialize();

    InitializeTexture(mph_f, th_f, d);
    InitializeTexture(mph_b, th_b, d);

    Texture* textures[2];
    textures[0] = th_f;
    textures[1] = th_b;

    WindowBuilder win_builder;
    win_builder.SetDevice(d);
    win_builder.SetDefaults();
    win_builder.SetNativeWindow(g_displayUtil.GetNativeWindow());
    win_builder.SetTextures(2, textures);
    CHECK(_t.Initialize(&win_builder));
    _initialized = true;
}

void ShaderBufferHolder::Initialize(Device *d, const void *shaderMainData, size_t mainDataSize)
{
    // 1. Allocate aligned storage
    auto alignedPoolSize    = LlgdAlignUp(mainDataSize, LWN_MEMORY_POOL_STORAGE_GRANULARITY);
    m_shaderDataPool        = LlgdAlignedAllocPodType<uint8_t>(alignedPoolSize, LWN_MEMORY_POOL_STORAGE_GRANULARITY);

    // 2. Copy shader data
    memcpy(reinterpret_cast<void *>(m_shaderDataPool.get()), shaderMainData, mainDataSize);

    // 3. Create a memory pool for shader main data
    MemoryPoolBuilder mpb;
    mpb.SetDevice(d).SetDefaults()
        .SetFlags(MemoryPoolFlags::CPU_NO_ACCESS | MemoryPoolFlags::GPU_CACHED | MemoryPoolFlags::SHADER_CODE)
        .SetStorage(m_shaderDataPool.get(), alignedPoolSize);
    mphShaderData.Initialize(&mpb);

    // 4. Create a Buffer and combine to the memory pools
    BufferBuilder bufferBuilder;
    bufferBuilder.SetDevice(d).SetDefaults()
        .SetStorage(mphShaderData, 0, mainDataSize);
    _t.Initialize(&bufferBuilder);

    _initialized = true;
}
} // llgd_lwn
