/*
 * Copyright (c) 2015 LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#include "lwntest_cpp.h"
#include "lwn/lwn_DeviceConstantsNX.h"
#include "lwn_utils.h"

using namespace lwn;

class LWNQueryTest
{
    bool testDeviceInfoQueryConst(Device *device) const;

public:
    LWNTEST_CppMethods();
};

lwString LWNQueryTest::getDescription() const
{
    lwStringBuf sb;
    sb <<
        "Basic query test for LWN.  Renders in green if queries produce "
        "expected results, or red otherwise.";
    return sb.str();
}

int LWNQueryTest::isSupported() const
{
    return lwogCheckLWNAPIVersion(52, 7);
}

bool LWNQueryTest::testDeviceInfoQueryConst(Device *device) const
{
#if !defined(LW_WINDOWS)

    int val;
    #define CHECK_DEVINFO(E) \
        device->GetInteger(DeviceInfo::Enum::E, &val); \
        if (val != LWN_DEVICE_INFO_CONSTANT_NX_ ## E) return false;

    CHECK_DEVINFO(UNIFORM_BUFFER_BINDINGS_PER_STAGE);
    CHECK_DEVINFO(MAX_UNIFORM_BUFFER_SIZE);
    CHECK_DEVINFO(UNIFORM_BUFFER_ALIGNMENT);
    CHECK_DEVINFO(COLOR_BUFFER_BINDINGS);
    CHECK_DEVINFO(VERTEX_BUFFER_BINDINGS);
    CHECK_DEVINFO(TRANSFORM_FEEDBACK_BUFFER_BINDINGS);
    CHECK_DEVINFO(SHADER_STORAGE_BUFFER_BINDINGS_PER_STAGE);
    CHECK_DEVINFO(TEXTURE_BINDINGS_PER_STAGE);
    CHECK_DEVINFO(COUNTER_ALIGNMENT);
    CHECK_DEVINFO(TRANSFORM_FEEDBACK_BUFFER_ALIGNMENT);
    CHECK_DEVINFO(TRANSFORM_FEEDBACK_CONTROL_ALIGNMENT);
    CHECK_DEVINFO(INDIRECT_DRAW_ALIGNMENT);
    CHECK_DEVINFO(VERTEX_ATTRIBUTES);
    CHECK_DEVINFO(TEXTURE_DESCRIPTOR_SIZE);
    CHECK_DEVINFO(SAMPLER_DESCRIPTOR_SIZE);
    CHECK_DEVINFO(RESERVED_TEXTURE_DESCRIPTORS);
    CHECK_DEVINFO(RESERVED_SAMPLER_DESCRIPTORS);
    CHECK_DEVINFO(COMMAND_BUFFER_COMMAND_ALIGNMENT);
    CHECK_DEVINFO(COMMAND_BUFFER_CONTROL_ALIGNMENT);
    CHECK_DEVINFO(COMMAND_BUFFER_MIN_COMMAND_SIZE);
    CHECK_DEVINFO(COMMAND_BUFFER_MIN_CONTROL_SIZE);
    CHECK_DEVINFO(SHADER_SCRATCH_MEMORY_SCALE_FACTOR_MINIMUM);
    CHECK_DEVINFO(SHADER_SCRATCH_MEMORY_SCALE_FACTOR_RECOMMENDED);
    CHECK_DEVINFO(SHADER_SCRATCH_MEMORY_ALIGNMENT);
    CHECK_DEVINFO(SHADER_SCRATCH_MEMORY_GRANULARITY);
    CHECK_DEVINFO(MAX_TEXTURE_ANISOTROPY);
    CHECK_DEVINFO(MAX_COMPUTE_WORK_GROUP_SIZE_X);
    CHECK_DEVINFO(MAX_COMPUTE_WORK_GROUP_SIZE_Y);
    CHECK_DEVINFO(MAX_COMPUTE_WORK_GROUP_SIZE_Z);
    CHECK_DEVINFO(MAX_COMPUTE_WORK_GROUP_SIZE_THREADS);
    CHECK_DEVINFO(MAX_COMPUTE_DISPATCH_WORK_GROUPS_X);
    CHECK_DEVINFO(MAX_COMPUTE_DISPATCH_WORK_GROUPS_Y);
    CHECK_DEVINFO(MAX_COMPUTE_DISPATCH_WORK_GROUPS_Z);
    CHECK_DEVINFO(IMAGE_BINDINGS_PER_STAGE);
    CHECK_DEVINFO(MAX_TEXTURE_POOL_SIZE);
    CHECK_DEVINFO(MAX_SAMPLER_POOL_SIZE);
    CHECK_DEVINFO(MAX_VIEWPORTS);
    CHECK_DEVINFO(MEMPOOL_TEXTURE_OBJECT_PAGE_ALIGNMENT);
    CHECK_DEVINFO(SUPPORTS_MIN_MAX_FILTERING);
    CHECK_DEVINFO(SUPPORTS_STENCIL8_FORMAT);
    CHECK_DEVINFO(SUPPORTS_ASTC_FORMATS);
    CHECK_DEVINFO(L2_SIZE);
    CHECK_DEVINFO(MAX_TEXTURE_LEVELS);
    CHECK_DEVINFO(MAX_TEXTURE_LAYERS);
    CHECK_DEVINFO(GLSLC_MAX_SUPPORTED_GPU_CODE_MAJOR_VERSION);
    CHECK_DEVINFO(GLSLC_MIN_SUPPORTED_GPU_CODE_MAJOR_VERSION);
    CHECK_DEVINFO(GLSLC_MAX_SUPPORTED_GPU_CODE_MINOR_VERSION);
    CHECK_DEVINFO(GLSLC_MIN_SUPPORTED_GPU_CODE_MINOR_VERSION);
    CHECK_DEVINFO(SUPPORTS_CONSERVATIVE_RASTER);
    CHECK_DEVINFO(SUBPIXEL_BITS);
    CHECK_DEVINFO(MAX_SUBPIXEL_BIAS_BITS);
    CHECK_DEVINFO(INDIRECT_DISPATCH_ALIGNMENT);
    CHECK_DEVINFO(ZLWLL_SAVE_RESTORE_ALIGNMENT);
    CHECK_DEVINFO(SHADER_SCRATCH_MEMORY_COMPUTE_SCALE_FACTOR_MINIMUM);
    CHECK_DEVINFO(LINEAR_TEXTURE_STRIDE_ALIGNMENT);
    CHECK_DEVINFO(LINEAR_RENDER_TARGET_STRIDE_ALIGNMENT);
    CHECK_DEVINFO(MEMORY_POOL_PAGE_SIZE);
    CHECK_DEVINFO(SUPPORTS_ZERO_FROM_UNMAPPED_VIRTUAL_POOL_PAGES);
    CHECK_DEVINFO(UNIFORM_BUFFER_UPDATE_ALIGNMENT);
    CHECK_DEVINFO(MAX_TEXTURE_SIZE);
    CHECK_DEVINFO(MAX_BUFFER_TEXTURE_SIZE);
    CHECK_DEVINFO(MAX_3D_TEXTURE_SIZE);
    CHECK_DEVINFO(MAX_LWBE_MAP_TEXTURE_SIZE);
    CHECK_DEVINFO(MAX_RECTANGLE_TEXTURE_SIZE);
    CHECK_DEVINFO(SUPPORTS_PASSTHROUGH_GEOMETRY_SHADERS);
    CHECK_DEVINFO(SUPPORTS_VIEWPORT_SWIZZLE);
    CHECK_DEVINFO(SUPPORTS_SPARSE_TILED_PACKAGED_TEXTURES);
    CHECK_DEVINFO(SUPPORTS_ADVANCED_BLEND_MODES);
    CHECK_DEVINFO(MAX_PRESENT_INTERVAL);
    CHECK_DEVINFO(SUPPORTS_DRAW_TEXTURE);
    CHECK_DEVINFO(SUPPORTS_TARGET_INDEPENDENT_RASTERIZATION);
    CHECK_DEVINFO(SUPPORTS_FRAGMENT_COVERAGE_TO_COLOR);
    CHECK_DEVINFO(SUPPORTS_POST_DEPTH_COVERAGE);
    CHECK_DEVINFO(SUPPORTS_IMAGES_USING_TEXTURE_HANDLES);
    CHECK_DEVINFO(SUPPORTS_SAMPLE_LOCATIONS);
    CHECK_DEVINFO(MAX_SAMPLE_LOCATION_TABLE_ENTRIES);
    CHECK_DEVINFO(SHADER_CODE_MEMORY_POOL_PADDING_SIZE);
    CHECK_DEVINFO(MAX_PATCH_SIZE);
    CHECK_DEVINFO(QUEUE_COMMAND_MEMORY_GRANULARITY);
    CHECK_DEVINFO(QUEUE_COMMAND_MEMORY_MIN_SIZE);
    CHECK_DEVINFO(QUEUE_COMMAND_MEMORY_DEFAULT_SIZE);
    CHECK_DEVINFO(QUEUE_COMPUTE_MEMORY_GRANULARITY);
    CHECK_DEVINFO(QUEUE_COMPUTE_MEMORY_MIN_SIZE);
    CHECK_DEVINFO(QUEUE_COMPUTE_MEMORY_DEFAULT_SIZE);
    CHECK_DEVINFO(QUEUE_COMMAND_MEMORY_MIN_FLUSH_THRESHOLD);
    CHECK_DEVINFO(SUPPORTS_FRAGMENT_SHADER_INTERLOCK);
    CHECK_DEVINFO(MAX_TEXTURES_PER_WINDOW);
    CHECK_DEVINFO(MIN_TEXTURES_PER_WINDOW);
    CHECK_DEVINFO(QUEUE_CONTROL_MEMORY_MIN_SIZE);
    CHECK_DEVINFO(QUEUE_CONTROL_MEMORY_DEFAULT_SIZE);
    CHECK_DEVINFO(QUEUE_CONTROL_MEMORY_GRANULARITY);
    CHECK_DEVINFO(SHADER_CODE_ALIGNMENT);

#endif
    return true;
}

void LWNQueryTest::doGraphics() const
{
    DeviceState *deviceState = DeviceState::GetActive();
    Device *device = deviceState->getDevice();
    QueueCommandBuffer &queueCB = deviceState->getQueueCB();
    Queue *queue = deviceState->getQueue();
    bool result = true;

    const int nMemPools = 4;
    const size_t memPoolSize = 64 * 1024;
    char *poolMem = (char *) PoolStorageAlloc(nMemPools * memPoolSize);
    MemoryPoolBuilder mpb;
    MemoryPool pools[nMemPools];
    mpb.SetDefaults().SetDevice(device);
#if defined(LW_WINDOWS)
    // Use CPU_NO_ACCESS on Windows because block linear textures can't use
    // CPU_UNCACHED there.
    mpb.SetFlags(MemoryPoolFlags::CPU_NO_ACCESS | MemoryPoolFlags::GPU_CACHED);
#else
    // Use CPU_UNCACHED on CheetAh because texture/sampler pools must be
    // GPU-accessible.
    mpb.SetFlags(MemoryPoolFlags::CPU_UNCACHED | MemoryPoolFlags::GPU_CACHED);
#endif
    for (int i = 0; i < nMemPools; i++) {
        mpb.SetStorage(poolMem + i * memPoolSize, memPoolSize);
        pools[i].Initialize(&mpb);
    }

    TextureBuilder tb;
    tb.SetDefaults().SetDevice(device);
    tb.SetTarget(TextureTarget::TARGET_2D);
    tb.SetLevels(1);
    tb.SetSize2D(4, 4);
    tb.SetFormat(Format::RGBA8);

    BufferBuilder bb;
    bb.SetDefaults().SetDevice(device);

    for (int i = 0; i < 4; i++) {

        TexturePool texPool;
        texPool.Initialize(pools + i, i * 256, 1024);
        if (texPool.GetMemoryPool() != pools + i) {
            result = false;
        }
        if (texPool.GetMemoryOffset() != (i * 256)) {
            result = false;
        }
        if (texPool.GetSize() != 1024) {
            result = false;
        }
        texPool.Finalize();

        SamplerPool smpPool;
        smpPool.Initialize(pools + i, (3 - i) * 256, 1024);
        if (smpPool.GetMemoryPool() != pools + i) {
            result = false;
        }
        if (smpPool.GetMemoryOffset() != ((3 - i) * 256)) {
            result = false;
        }
        if (smpPool.GetSize() != 1024) {
            result = false;
        }
        smpPool.Finalize();

        Texture tex;
        tb.SetStorage(pools + i, i * 8192);
        if (tb.GetMemoryPool() != pools + i) {
            result = false;
        }
        if (tb.GetMemoryOffset() != (i * 8192)) {
            result = false;
        }
        tex.Initialize(&tb);
        if (tex.GetMemoryPool() != pools + i) {
            result = false;
        }
        if (tex.GetMemoryOffset() != (i * 8192)) {
            result = false;
        }
        tex.Finalize();

        Buffer buf;
        bb.SetStorage(pools + i, (3 - i) * 8192, 8192);
        if (bb.GetMemoryPool() != pools + i) {
            result = false;
        }
        if (bb.GetMemoryOffset() != ((3 - i) * 8192)) {
            result = false;
        }
        if (bb.GetSize() != 8192) {
            result = false;
        }
        buf.Initialize(&bb);
        if (buf.GetMemoryPool() != pools + i) {
            result = false;
        }
        if (buf.GetMemoryOffset() != ((3 - i) * 8192)) {
            result = false;
        }
        if (buf.GetSize() != 8192) {
            result = false;
        }
        buf.Finalize();
    }

    // DeviceInfo query tests.
    if (!testDeviceInfoQueryConst(device)) {
        result = false;
    }

    if (result) {
        queueCB.ClearColor(0, 0.0, 1.0, 0.0, 1.0);
    } else {
        queueCB.ClearColor(0, 1.0, 0.0, 0.0, 1.0);
    }

    queueCB.submit();
    queue->Finish();

    for (int i = 0; i < 4; i++) {
        pools[i].Finalize();
    }

    PoolStorageFree(poolMem);
}

OGTEST_CppTest(LWNQueryTest, lwn_query, );
