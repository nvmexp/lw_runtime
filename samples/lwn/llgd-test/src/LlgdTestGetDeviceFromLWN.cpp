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

class GetDeviceFromLWLWalidator {
public:
    void Initialize();
    bool Test();

private:
    void InitializeLWN();
    void InitializeQueue();
    void InitializeProgram();
    void InitializePool();
    void InitializeWindow();
    void InitializeSync();

    static const size_t     SIZE = 4096;
    std::unique_ptr<uint8_t, std::function<void(uint8_t*)>> spPool;

    llgd_lwn::DeviceHolder dh; // Use a local device instead of the global one
    llgd_lwn::QueueHolder qh;
    llgd_lwn::ProgramHolder ph;
    llgd_lwn::MemoryPoolHolder mph;
    llgd_lwn::WindowHolder wh;
    llgd_lwn::SyncHolder sh;
};

void GetDeviceFromLWLWalidator::InitializeLWN()
{
    dh.Initialize();
}

void GetDeviceFromLWLWalidator::InitializeQueue()
{
    qh.Initialize(dh);
}

void GetDeviceFromLWLWalidator::InitializeProgram()
{
    CHECK(ph.Initialize((Device*)dh));
}

void GetDeviceFromLWLWalidator::InitializePool()
{
    spPool = LlgdAlignedAllocPodType<uint8_t>(SIZE, 4096);

    MemoryPoolBuilder pool_builder;

    pool_builder.SetDevice(dh).SetDefaults()
                .SetFlags(MemoryPoolFlags::CPU_CACHED |
                          MemoryPoolFlags::GPU_CACHED);

    pool_builder.SetStorage(spPool.get(), SIZE);
    CHECK(mph.Initialize(&pool_builder));
}

void GetDeviceFromLWLWalidator::InitializeWindow()
{
    wh.Initialize(dh);
}

void GetDeviceFromLWLWalidator::InitializeSync()
{
    CHECK(sh.Initialize((Device*)dh));
}

void GetDeviceFromLWLWalidator::Initialize()
{
    InitializeLWN();
    InitializeQueue();
    InitializeProgram();
    InitializePool();
    InitializeWindow();
    InitializeSync();
}

bool GetDeviceFromLWLWalidator::Test()
{
    LWNdevice* dev(dh);

    TEST_EQ(llgdLwnGetDeviceFromQueue(qh), dev);
    TEST_EQ(llgdLwnGetDeviceFromProgram(ph), dev);
    TEST_EQ(llgdLwnGetDeviceFromMemoryPool(mph), dev);
    TEST_EQ(llgdLwnGetDeviceFromWindow(wh), dev);
    TEST_EQ(llgdLwnGetDeviceFromSync(sh), dev);

    return true;
}

LLGD_DEFINE_TEST(GetDeviceFromLWN, UNIT,
LwError Execute()
{
    GetDeviceFromLWLWalidator v;
    v.Initialize();

    if (!v.Test()) { return LwError_IlwalidState; }
    else { return LwSuccess; }
}
); // LLGD_DEFINE_TEST
