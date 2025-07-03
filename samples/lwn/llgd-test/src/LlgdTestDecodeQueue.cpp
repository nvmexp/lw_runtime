/*
 * Copyright (c) 2017, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 */

#include <LlgdTest.h>
#include <LlgdTestUtil.h>
#include <LlgdTestUtilLWN.h>

#include <liblwn-llgd.h>

class DecodeQueueValidator {
public:
    void Initialize();
    bool Test();

private:
    void InitializeQueues();
    void InitializeCommandBuffer();
    void InitializeTexturePools();
    void InitializeSamplerPools();

    bool TestGetQueueInfo();
    bool TestGetQueueTexturePool();
    bool TestGetQueueSamplerPool();

    static const size_t SIZE = 4096 * 64;
    static const size_t ALIGNMENT = 4096;
    LlgdUniqueUint8PtrWithLwstomDeleter cmd_space;
    LlgdUniqueUint8PtrWithLwstomDeleter ctrl_space;

    static const size_t POOL_SIZE = 4096 * 4;
    LlgdUniqueUint8PtrWithLwstomDeleter texture_space;
    LlgdUniqueUint8PtrWithLwstomDeleter sampler_space;

    // Queues
    llgd_lwn::QueueHolder qh1;
    llgd_lwn::QueueHolder qh2;
    llgd_lwn::QueueHolder qh3;

    // Command Buffer
    llgd_lwn::MemoryPoolHolder cmd_mph;
    llgd_lwn::CommandBufferHolder cbh;

    // Texture Pools
    llgd_lwn::MemoryPoolHolder tex_mph;
    llgd_lwn::TexturePoolHolder tph1, tph2;

    // Sampler Pools
    llgd_lwn::MemoryPoolHolder smp_mph;
    llgd_lwn::SamplerPoolHolder sph1, sph2;
};

void DecodeQueueValidator::InitializeQueues()
{
    QueueBuilder queue_builder;
    queue_builder.SetDevice(g_device).SetDefaults();

    queue_builder.SetCommandMemorySize(65536)
        .SetControlMemorySize(4096)
        .SetCommandFlushThreshold(4096);

    CHECK(qh1.GenericHolder::Initialize(&queue_builder)); // call the super class version

    queue_builder.SetCommandMemorySize(73728)
        .SetControlMemorySize(65536)
        .SetCommandFlushThreshold(8192)
        .SetFlags(QueueFlags::NO_FRAGMENT_INTERLOCK | QueueFlags::NO_ZLWLL);

    CHECK(qh2.GenericHolder::Initialize(&queue_builder));

    queue_builder.SetCommandMemorySize(147456)
        .SetControlMemorySize(131072)
        .SetCommandFlushThreshold(16384)
        .SetFlags(QueueFlags::NO_FRAGMENT_INTERLOCK);

    CHECK(qh3.GenericHolder::Initialize(&queue_builder));
}

void DecodeQueueValidator::InitializeCommandBuffer()
{
    MemoryPoolBuilder pool_builder;

    cmd_space = LlgdAlignedAllocPodType<uint8_t>(SIZE, ALIGNMENT);
    ctrl_space = LlgdAlignedAllocPodType<uint8_t>(SIZE, ALIGNMENT);

    pool_builder.SetDevice(g_device).SetDefaults()
        .SetFlags(MemoryPoolFlags::CPU_CACHED | MemoryPoolFlags::GPU_CACHED);
    pool_builder.SetStorage(cmd_space.get(), SIZE);
    CHECK(cmd_mph.Initialize(&pool_builder));

    CHECK(cbh.Initialize((Device*)g_device));

    cbh->AddCommandMemory(cmd_mph, 0, SIZE);
    cbh->AddControlMemory(ctrl_space.get(), SIZE);
}

void DecodeQueueValidator::InitializeTexturePools()
{
    int tex_head_size;
    g_device->GetInteger(DeviceInfo::TEXTURE_DESCRIPTOR_SIZE, &tex_head_size);

    texture_space = LlgdAlignedAllocPodType<uint8_t>(POOL_SIZE * 2, ALIGNMENT);

    MemoryPoolBuilder pool_builder;
    pool_builder.SetDevice(g_device).SetDefaults()
        .SetFlags(MemoryPoolFlags::CPU_CACHED | MemoryPoolFlags::GPU_CACHED | MemoryPoolFlags::COMPRESSIBLE);
    pool_builder.SetStorage(texture_space.get(), POOL_SIZE * 2);
    CHECK(tex_mph.Initialize(&pool_builder));

    CHECK(tph1.Initialize((const MemoryPool *)tex_mph, 0, POOL_SIZE / tex_head_size));
    CHECK(tph2.Initialize((const MemoryPool *)tex_mph, POOL_SIZE, POOL_SIZE / tex_head_size));
}

void DecodeQueueValidator::InitializeSamplerPools()
{
    int smp_head_size;
    g_device->GetInteger(DeviceInfo::SAMPLER_DESCRIPTOR_SIZE, &smp_head_size);

    sampler_space = LlgdAlignedAllocPodType<uint8_t>(POOL_SIZE * 2, ALIGNMENT);

    MemoryPoolBuilder pool_builder;
    pool_builder.SetDevice(g_device).SetDefaults()
        .SetFlags(MemoryPoolFlags::CPU_CACHED | MemoryPoolFlags::GPU_CACHED | MemoryPoolFlags::COMPRESSIBLE);
    pool_builder.SetStorage(sampler_space.get(), POOL_SIZE * 2);
    CHECK(smp_mph.Initialize(&pool_builder));

    CHECK(sph1.Initialize((const MemoryPool *)smp_mph, 0, POOL_SIZE / smp_head_size));
    CHECK(sph2.Initialize((const MemoryPool *)smp_mph, POOL_SIZE, POOL_SIZE / smp_head_size));
}

void DecodeQueueValidator::Initialize()
{
    InitializeQueues();
    InitializeCommandBuffer();
    InitializeTexturePools();
    InitializeSamplerPools();
}

static unsigned int CalcExpectedControlMemorySize(unsigned int requestedControlMemorySize)
{
#if defined(LW_HOS)
    return requestedControlMemorySize;
#else // Driver callwates control memory size differently on L4T
    auto nextPowerOf2 = [](unsigned int n)-> unsigned int {
        unsigned int p = 1;
        if (n && !(n & (n - 1))) {
            return n;
        }
        while (p < n) {
            p <<= 1;
        }
        return p;
    };

    const auto hwGpEntrySize = 8U;
    auto numGpFifoEntries = nextPowerOf2((requestedControlMemorySize / hwGpEntrySize) * 3U + 1U);
    return numGpFifoEntries * hwGpEntrySize;
#endif
}

bool DecodeQueueValidator::TestGetQueueInfo()
{
#define VALIDATE_QUEUE_INFO(index, commandMemorySize, controlMemorySize, flushThresh, flags) \
    TEST_EQ(llgdLwnGetQueueCommandMemorySize(qh##index), commandMemorySize); \
    TEST_EQ(llgdLwnGetQueueControlMemorySize(qh##index), controlMemorySize); \
    TEST_EQ(llgdLwnGetQueueFlushThreshold(qh##index), flushThresh); \
    TEST_EQ(llgdLwnGetQueueStateIndex(qh##index), (index-1)); /* 0-based state index */ \
    TEST_EQ(llgdLwnGetQueueFlags(qh##index), (flags)); \
    TEST_EQ(llgdLwnIsDriverOwnedMemoryPool((LWNmemoryPool*)llgdLwnGetQueueMemoryPool(qh##index)), true); \
    TEST_EQ(llgdLwnIsDriverOwnedMemoryPool((LWNmemoryPool*)llgdLwnGetQueueCommandMemoryPool(qh##index)), true);

    VALIDATE_QUEUE_INFO(1, 65536, CalcExpectedControlMemorySize(4096), 4096, LWN_QUEUE_FLAGS_NONE);
    VALIDATE_QUEUE_INFO(2, 73728, CalcExpectedControlMemorySize(65536), 8192, LWN_QUEUE_FLAGS_NO_FRAGMENT_INTERLOCK_BIT | LWN_QUEUE_FLAGS_NO_ZLWLL_BIT);
    VALIDATE_QUEUE_INFO(3, 147456, CalcExpectedControlMemorySize(131072), 16384, LWN_QUEUE_FLAGS_NO_FRAGMENT_INTERLOCK_BIT);

    return true;
}

// Test llgdLwnGetQueueTexturePool
bool DecodeQueueValidator::TestGetQueueTexturePool()
{
    TEST_EQ(llgdLwnGetQueueTexturePool(qh1), nullptr);
    TEST_EQ(llgdLwnGetQueueTexturePool(qh2), nullptr);
    TEST_EQ(llgdLwnGetQueueTexturePool(qh3), nullptr);

    cbh->BeginRecording();
    {
        cbh->SetTexturePool(tph1);
    }
    CommandHandle handle = cbh->EndRecording();

    qh1->SubmitCommands(1, &handle);

    TEST_EQ(llgdLwnGetQueueTexturePool(qh1), tph1);
    TEST_EQ(llgdLwnGetQueueTexturePool(qh2), nullptr);
    TEST_EQ(llgdLwnGetQueueTexturePool(qh3), nullptr);

    qh2->SubmitCommands(1, &handle);

    TEST_EQ(llgdLwnGetQueueTexturePool(qh1), tph1);
    TEST_EQ(llgdLwnGetQueueTexturePool(qh2), tph1);
    TEST_EQ(llgdLwnGetQueueTexturePool(qh3), nullptr);

    cbh->BeginRecording();
    {
        cbh->SetTexturePool(tph2);
    }
    handle = cbh->EndRecording();

    qh1->SubmitCommands(1, &handle);
    qh3->SubmitCommands(1, &handle);

    TEST_EQ(llgdLwnGetQueueTexturePool(qh1), tph2);
    TEST_EQ(llgdLwnGetQueueTexturePool(qh2), tph1);
    TEST_EQ(llgdLwnGetQueueTexturePool(qh3), tph2);

    return true;
}

// Test llgdLwnGetQueueSamplerPool
bool DecodeQueueValidator::TestGetQueueSamplerPool()
{
    TEST_EQ(llgdLwnGetQueueSamplerPool(qh1), nullptr);
    TEST_EQ(llgdLwnGetQueueSamplerPool(qh2), nullptr);
    TEST_EQ(llgdLwnGetQueueSamplerPool(qh3), nullptr);

    cbh->BeginRecording();
    {
        cbh->SetSamplerPool(sph1);
    }
    CommandHandle handle = cbh->EndRecording();

    qh1->SubmitCommands(1, &handle);

    TEST_EQ(llgdLwnGetQueueSamplerPool(qh1), sph1);
    TEST_EQ(llgdLwnGetQueueSamplerPool(qh2), nullptr);
    TEST_EQ(llgdLwnGetQueueSamplerPool(qh3), nullptr);

    qh2->SubmitCommands(1, &handle);

    TEST_EQ(llgdLwnGetQueueSamplerPool(qh1), sph1);
    TEST_EQ(llgdLwnGetQueueSamplerPool(qh2), sph1);
    TEST_EQ(llgdLwnGetQueueSamplerPool(qh3), nullptr);

    cbh->BeginRecording();
    {
        cbh->SetSamplerPool(sph2);
    }
    handle = cbh->EndRecording();

    qh1->SubmitCommands(1, &handle);
    qh3->SubmitCommands(1, &handle);

    TEST_EQ(llgdLwnGetQueueSamplerPool(qh1), sph2);
    TEST_EQ(llgdLwnGetQueueSamplerPool(qh2), sph1);
    TEST_EQ(llgdLwnGetQueueSamplerPool(qh3), sph2);

    return true;
}

bool DecodeQueueValidator::Test()
{
    if (!TestGetQueueInfo()) { return false; }
    if (!TestGetQueueTexturePool()) { return false; }
    if (!TestGetQueueSamplerPool()) { return false; }
    return true;
}

LLGD_DEFINE_TEST(DecodeQueue, UNIT,
LwError Execute()
{
    DecodeQueueValidator v;
    v.Initialize();

    if (!v.Test())  { return LwError_IlwalidState; }
    else            { return LwSuccess;            }
}
); // LLGD_DEFINE_TEST
