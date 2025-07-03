/*
 * Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 */

#include <deque>

#include <AftermathTest.h>
#include <AftermathTestLogging.h>
#include <AftermathTestUtilsLWN.h>

#include <AftermathApi.h>

namespace AftermathTest {

class ApiTest
{
public:
    bool Initialize();

    bool Test(const Options& options);

private:

    bool TestPreInitializeGraphics();
    bool TestPostInitializeGraphics();
    bool TestPostInitializeDevice();
    bool TestUserResourceTracking();

    bool CreateResources();

    LWN::UniqueUint8PtrWithLwstomDeleter m_poolMemory;
    LWN::MemoryPoolHolder m_memoryPool;
    std::deque<LWN::TextureHolder> m_textures;
    std::deque<LWN::BufferHolder> m_buffers;
    std::deque<LWN::SamplerPoolHolder> m_samplerPools;
    std::deque<LWN::TexturePoolHolder> m_texturePools;
};

bool ApiTest::TestPreInitializeGraphics()
{
    bool enabled = false;
    AftermathApiFeatureLevel featureLevel = AftermathApiFeatureLevel_Default;
    int flags = 0xdeadbeef;

    // aftermathIsEnabled should fail, if called before lw::InitializeGraphics()
    TEST_EQ(aftermathIsEnabled(&enabled), AftermathApiError_GraphicsNotInitialized);

    // aftermathGetFeatureLevel should fail, if called before lw::InitializeGraphics()
    TEST_EQ(aftermathGetFeatureLevel(&featureLevel), AftermathApiError_GraphicsNotInitialized);

    // aftermathGetConfigurationFlags should fail, if called before lw::InitializeGraphics()
    TEST_EQ(aftermathGetConfigurationFlags(&flags), AftermathApiError_GraphicsNotInitialized);

    return true;
}

bool ApiTest::TestPostInitializeGraphics()
{
    bool enabled = false;
    AftermathApiFeatureLevel featureLevel = AftermathApiFeatureLevel_Default;
    int flags = 0xdeadbeef;

    // aftermathIsEnabled should work, if called after lw::InitializeGraphics()
    TEST_EQ(aftermathIsEnabled(&enabled), AftermathApiError_None);

    // Expect that GPU crash dumps are enabled in DevMenu!
    TEST_EQ(enabled, true);

    // aftermathGetFeatureLevel should work, if called after lw::InitializeGraphics()
    TEST_EQ(aftermathGetFeatureLevel(&featureLevel), AftermathApiError_None);

    // Expect basic feature level until we have DevMenu support for controlling it
    TEST_EQ(featureLevel, AftermathApiFeatureLevel_Basic);

    // Set feature level - this can be only done before the first device is created.
    TEST_EQ(aftermathSetFeatureLevel(AftermathApiFeatureLevel_Full), AftermathApiError_None);

    // Make sure the feature level sticks
    TEST_EQ(aftermathGetFeatureLevel(&featureLevel), AftermathApiError_None);
    TEST_EQ(featureLevel, AftermathApiFeatureLevel_Full);

    // aftermathGetConfigurationFlags should work, if called before lw::InitializeGraphics()
    TEST_EQ(aftermathGetConfigurationFlags(&flags), AftermathApiError_None);
    TEST_EQ(flags, AftermathApiConfigurationFlags_Default);

    // Set flags - this can be only done before the first device is created.
    TEST_EQ(aftermathSetConfigurationFlags(AftermathApiConfigurationFlags_AutomaticResourceTracking), AftermathApiError_None);

    // Make sure the flags stick
    TEST_EQ(aftermathGetConfigurationFlags(&flags), AftermathApiError_None);
    TEST_EQ(flags, AftermathApiConfigurationFlags_AutomaticResourceTracking);

    // Reset flags to default for the test
    TEST_EQ(aftermathSetConfigurationFlags(AftermathApiConfigurationFlags_Default), AftermathApiError_None);
    TEST_EQ(aftermathGetConfigurationFlags(&flags), AftermathApiError_None);
    TEST_EQ(flags, AftermathApiConfigurationFlags_Default);

    return true;
}

bool ApiTest::TestPostInitializeDevice()
{
    bool enabled = false;
    AftermathApiFeatureLevel featureLevel = AftermathApiFeatureLevel_Default;
    int flags = 0xdeadbeef;

    // Setting feature level after the device is initialized is not allowed.
    TEST_EQ(aftermathSetFeatureLevel(AftermathApiFeatureLevel_Enhanced), AftermathApiError_DeviceAlreadyInitialized);

    // Make sure the feature level sticks
    TEST_EQ(aftermathGetFeatureLevel(&featureLevel), AftermathApiError_None);
    TEST_EQ(featureLevel, AftermathApiFeatureLevel_Full);

    // Setting configuration flags after the device is initialized is not allowed.
    TEST_EQ(aftermathSetConfigurationFlags(AftermathApiConfigurationFlags_Default), AftermathApiError_DeviceAlreadyInitialized);

    // Flags shoudl be set to default for the test
    TEST_EQ(aftermathGetConfigurationFlags(&flags), AftermathApiError_None);
    TEST_EQ(flags, AftermathApiConfigurationFlags_Default);

    // aftermathIsEnabled should work, if called after lw::InitializeGraphics()
    TEST_EQ(aftermathIsEnabled(&enabled), AftermathApiError_None);
    TEST_EQ(enabled, true);

    return true;
}

bool ApiTest::CreateResources()
{
    const int numTextures = 200;
    const int numBuffers = 200;
    const int numSamplerPools = 200;
    const int numTexturePools = 200;

    TextureBuilder textureBuilder;
    textureBuilder
        .SetDevice(g_device)
        .SetDefaults()
        .SetTarget(TextureTarget::TARGET_2D)
        .SetFormat(Format::RGBA8)
        .SetSize2D(16, 32);

    size_t BufferSize = 256;
    BufferBuilder bufferBuilder;
    bufferBuilder
        .SetDevice(g_device)
        .SetDefaults();

    // Determine required pool size
    size_t poolMemorySize = 0;
    for (int i = 0; i < numTextures; ++i) {
        poolMemorySize = Utils::AlignUp(poolMemorySize, textureBuilder.GetStorageAlignment());
        poolMemorySize += textureBuilder.GetStorageSize();
    }
    for (int i = 0; i < numBuffers; ++i) {
        poolMemorySize = Utils::AlignUp(poolMemorySize, LWN_DEVICE_INFO_CONSTANT_NX_UNIFORM_BUFFER_ALIGNMENT);
        poolMemorySize += BufferSize;
    }
    const size_t numSamplerDescriptors = numTextures + LWN_DEVICE_INFO_CONSTANT_NX_RESERVED_SAMPLER_DESCRIPTORS;
    for (int i = 0; i < numSamplerPools; ++i) {
        poolMemorySize = Utils::AlignUp(poolMemorySize, LWN_DEVICE_INFO_CONSTANT_NX_SAMPLER_DESCRIPTOR_SIZE);
        poolMemorySize += numSamplerDescriptors * LWN_DEVICE_INFO_CONSTANT_NX_SAMPLER_DESCRIPTOR_SIZE;
    }
    const size_t numTextureDescriptors = numTextures + LWN_DEVICE_INFO_CONSTANT_NX_RESERVED_TEXTURE_DESCRIPTORS;
    for (int i = 0; i < numTexturePools; ++i) {
        poolMemorySize = Utils::AlignUp(poolMemorySize, LWN_DEVICE_INFO_CONSTANT_NX_TEXTURE_DESCRIPTOR_SIZE);
        poolMemorySize += numTextureDescriptors * LWN_DEVICE_INFO_CONSTANT_NX_TEXTURE_DESCRIPTOR_SIZE;
    }

    poolMemorySize = Utils::AlignUp(poolMemorySize, LWN_MEMORY_POOL_STORAGE_GRANULARITY);

    // Allocate memory
    m_poolMemory = LWN::AlignedAllocPodType<uint8_t>(poolMemorySize, LWN_MEMORY_POOL_STORAGE_ALIGNMENT);
    TEST_NE(m_poolMemory, nullptr);

    // Create pool
    MemoryPoolBuilder poolBuilder;
    poolBuilder.SetDevice(g_device)
        .SetDefaults()
        .SetFlags(MemoryPoolFlags::CPU_UNCACHED | MemoryPoolFlags::GPU_CACHED)
        .SetStorage(m_poolMemory.get(), poolMemorySize);
    TEST(m_memoryPool.Initialize(&poolBuilder));

    size_t memoryPoolOffset = 0;

    // Create textures
    for (int i = 0; i < numTextures; ++i) {
        memoryPoolOffset = Utils::AlignUp(memoryPoolOffset, textureBuilder.GetStorageAlignment());
        textureBuilder.SetStorage(m_memoryPool, memoryPoolOffset);
        memoryPoolOffset += textureBuilder.GetStorageSize();

        m_textures.emplace_back();
        LWN::TextureHolder& texture = m_textures.back();
        TEST(texture.Initialize(&textureBuilder));
    }

    // Create buffers
    for (int i = 0; i < numBuffers; ++i) {
        memoryPoolOffset = Utils::AlignUp(memoryPoolOffset, LWN_DEVICE_INFO_CONSTANT_NX_UNIFORM_BUFFER_ALIGNMENT);
        bufferBuilder.SetStorage(m_memoryPool, memoryPoolOffset, BufferSize);
        memoryPoolOffset += BufferSize;

        m_buffers.emplace_back();
        LWN::BufferHolder& buffer = m_buffers.back();
        TEST(buffer.Initialize(&bufferBuilder));
    }

    // Create sampler pools
    for (int i = 0; i < numSamplerPools; ++i) {
        m_samplerPools.emplace_back();
        LWN::SamplerPoolHolder& samplerPool = m_samplerPools.back();
        TEST(samplerPool.Initialize((MemoryPool*)m_memoryPool, memoryPoolOffset, numSamplerDescriptors));
        memoryPoolOffset += numSamplerDescriptors * LWN_DEVICE_INFO_CONSTANT_NX_SAMPLER_DESCRIPTOR_SIZE;
    }

    // Create texture pools
    for (int i = 0; i < numTexturePools; ++i) {
        m_texturePools.emplace_back();
        LWN::TexturePoolHolder& texturePool = m_texturePools.back();
        TEST(texturePool.Initialize((MemoryPool*)m_memoryPool, memoryPoolOffset, numTextureDescriptors));
        memoryPoolOffset += numTextureDescriptors * LWN_DEVICE_INFO_CONSTANT_NX_TEXTURE_DESCRIPTOR_SIZE;
    }

    return true;
}

static bool VerifyResourceTrackingCounts(const AftermathTestTrackedResourcesCounts& expectedResourceCounts, size_t numUsedResourceSlots)
{
    TEST_EQ(numUsedResourceSlots, expectedResourceCounts.numTextures + expectedResourceCounts.numBuffers + expectedResourceCounts.numSamplerPools + expectedResourceCounts.numTexturePools);

    AftermathTestTrackedResourcesCounts resourceCounts = {};
    TEST_EQ(aftermathTestGetNumTrackedResources(&resourceCounts), AftermathApiError_None);
    TEST_EQ(resourceCounts.numTextures, expectedResourceCounts.numTextures);
    TEST_EQ(resourceCounts.numBuffers, expectedResourceCounts.numBuffers);
    TEST_EQ(resourceCounts.numSamplerPools, expectedResourceCounts.numSamplerPools);
    TEST_EQ(resourceCounts.numTexturePools, expectedResourceCounts.numTexturePools);

    return true;
}

bool ApiTest::TestUserResourceTracking()
{
    // This is the minimum number of resource slots that should be supported.
    const size_t MinAvailableResourceSlots = 32;


    TEST(CreateResources());

    size_t numUsedResourceSlots = 0;
    AftermathTestTrackedResourcesCounts expectedResourceCounts = {};

    // Verify tracking counts
    TEST(VerifyResourceTrackingCounts(expectedResourceCounts, numUsedResourceSlots));

    // Track as many textures as possible
    AftermathApiError result = AftermathApiError_None;
    for (size_t i = 0; i < m_textures.size(); ++i) {
        LWN::TextureHolder& texture = m_textures[i];
        result = aftermathTrackTextureResource(texture);
        if (result == AftermathApiError_OutOfResourceSlots) {
            break;
        }
        TEST_EQ(result, AftermathApiError_None);
        ++numUsedResourceSlots;
        ++expectedResourceCounts.numTextures;
    }
    TEST_GE(numUsedResourceSlots, MinAvailableResourceSlots);


    // Verify tracking counts
    TEST(VerifyResourceTrackingCounts(expectedResourceCounts, numUsedResourceSlots));

    // Untrack every second texture
    size_t numResourcesToUntrack = numUsedResourceSlots / 2;
    for (size_t i = 0; i < numResourcesToUntrack; ++i) {
        LWN::TextureHolder& texture = m_textures[2 * i];
        result = aftermathUntrackTextureResource(texture);
        TEST_EQ(result, AftermathApiError_None);
        --numUsedResourceSlots;
        --expectedResourceCounts.numTextures;
    }

    // Verify tracking counts
    TEST(VerifyResourceTrackingCounts(expectedResourceCounts, numUsedResourceSlots));

    // Track as many buffers as possible
    for (size_t i = 0; i < m_buffers.size(); ++i) {
        LWN::BufferHolder& buffer = m_buffers[i];
        result = aftermathTrackBufferResource(buffer);
        if (result == AftermathApiError_OutOfResourceSlots) {
            break;
        }
        TEST_EQ(result, AftermathApiError_None);
        ++numUsedResourceSlots;
        ++expectedResourceCounts.numBuffers;
    }

    // Verify tracking counts
    TEST(VerifyResourceTrackingCounts(expectedResourceCounts, numUsedResourceSlots));

    // Untrack all resources
    aftermathUntrackAllResources();
    numUsedResourceSlots = 0;
    expectedResourceCounts.numTextures = 0;
    expectedResourceCounts.numBuffers = 0;
    expectedResourceCounts.numSamplerPools = 0;

    // Verify tracking counts
    TEST(VerifyResourceTrackingCounts(expectedResourceCounts, numUsedResourceSlots));

    // Track every tenth some texture again
    size_t numResourcesToTrack = m_textures.size() / 10;
    for (size_t i = 0; i < numResourcesToTrack; ++i) {
        LWN::TextureHolder& texture = m_textures[10 * i];
        result = aftermathTrackTextureResource(texture);
        if (result == AftermathApiError_OutOfResourceSlots) {
            break;
        }
        TEST_EQ(result, AftermathApiError_None);
        ++numUsedResourceSlots;
        ++expectedResourceCounts.numTextures;
    }
    TEST_EQ(numUsedResourceSlots, numResourcesToTrack);

    // Verify tracking counts
    TEST(VerifyResourceTrackingCounts(expectedResourceCounts, numUsedResourceSlots));

    // Track as many sampler pools as possible
    for (size_t i = 0; i < m_samplerPools.size(); ++i) {
        LWN::SamplerPoolHolder& samplerPool = m_samplerPools[i];
        result = aftermathTrackSamplerPoolResource(samplerPool);
        if (result == AftermathApiError_OutOfResourceSlots) {
            break;
        }
        TEST_EQ(result, AftermathApiError_None);
        ++numUsedResourceSlots;
        ++expectedResourceCounts.numSamplerPools;
    }

    // Verify tracking counts
    TEST(VerifyResourceTrackingCounts(expectedResourceCounts, numUsedResourceSlots));

    // Untrack every third sampler pool
    numResourcesToUntrack = expectedResourceCounts.numSamplerPools / 3;
    for (size_t i = 0; i < numResourcesToUntrack; ++i) {
        LWN::SamplerPoolHolder& samplerPool = m_samplerPools[3 * i];
        result = aftermathUntrackSamplerPoolResource(samplerPool);
        TEST_EQ(result, AftermathApiError_None);
        --numUsedResourceSlots;
        --expectedResourceCounts.numSamplerPools;
    }

    // Verify tracking counts
    TEST(VerifyResourceTrackingCounts(expectedResourceCounts, numUsedResourceSlots));

    // Track as many texture pools as possible
    for (size_t i = 0; i < m_texturePools.size(); ++i) {
        LWN::TexturePoolHolder& texturePool = m_texturePools[i];
        result = aftermathTrackTexturePoolResource(texturePool);
        if (result == AftermathApiError_OutOfResourceSlots) {
            break;
        }
        TEST_EQ(result, AftermathApiError_None);
        ++numUsedResourceSlots;
        ++expectedResourceCounts.numTexturePools;
    }

    // Verify tracking counts
    TEST(VerifyResourceTrackingCounts(expectedResourceCounts, numUsedResourceSlots));

    // Finalize the first two texture pools
    m_texturePools[0].Finalize();
    m_texturePools[1].Finalize();

    // Try to untrack the finalized texture pools (should fail)
    result = aftermathUntrackTexturePoolResource(m_texturePools[0]);
    TEST_EQ(result, AftermathApiError_ResourceNotTracked);
    result = aftermathUntrackTexturePoolResource(m_texturePools[1]);
    TEST_EQ(result, AftermathApiError_ResourceNotTracked);

    // Verify tracking counts
    TEST(VerifyResourceTrackingCounts(expectedResourceCounts, numUsedResourceSlots));

    // Untrack all resources
    aftermathUntrackAllResources();
    numUsedResourceSlots = 0;
    expectedResourceCounts.numTextures = 0;
    expectedResourceCounts.numBuffers = 0;
    expectedResourceCounts.numSamplerPools = 0;
    expectedResourceCounts.numTexturePools = 0;

    // Verify tracking counts
    TEST(VerifyResourceTrackingCounts(expectedResourceCounts, numUsedResourceSlots));

    return true;
}

bool ApiTest::Test(const Options& options)
{
    TEST(TestPreInitializeGraphics());

    // Initialize graphics
    LWN::SetupLWNGraphics();

    TEST(TestPostInitializeGraphics());

    // Initialize the device
    const DeviceFlagBits flags = options.disableDebugLayer ? 0 : DeviceFlagBits::DEBUG_ENABLE_LEVEL_4;
    LWN::SetupLWNDevice(flags);

    TEST(TestPostInitializeDevice());

    TEST(TestUserResourceTracking());

    return true;
}

// Integration test - requires Aftermath to be enabled by DevMenu setting!
AFTERMATH_DEFINE_TEST(API, INTEGRATION,
    LwError Execute(const Options& options)
    {
        ApiTest test;
        bool success = test.Test(options);
        return success ? LwSuccess : LwError_IlwalidState;
    }
);

} // namespace AftermathTest
