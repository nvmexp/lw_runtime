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
#include <LlgdTestUtilPool.h>

#include <liblwn-llgd.h>

class DecodeMemoryPoolValidator {
public:
    void Initialize();
    bool Test();

private:
    llgd_lwn::PoolUtil poolUtil;
    llgd_lwn::QueueHolder qh;
};

void DecodeMemoryPoolValidator::Initialize()
{
    qh.Initialize(g_device);
}

bool DecodeMemoryPoolValidator::Test()
{
    // Test user pool
    for (int type = 0; type < static_cast<int>(llgd_lwn::PoolType::Last); ++type)
    {
        auto poolType = static_cast<llgd_lwn::PoolType>(type);
#if defined(LW_LINUX) //WAR
        // TODO: (http://lwbugs/3102903) Remove this check when REMAP is ported to linux
        if (llgd_lwn::PoolUtil::IsVirtual(poolType)) {
            continue;
        }
#endif
        llgd_lwn::MemoryPoolHolder mph;
        poolUtil.InitPool(mph, poolType);

        TEST_EQ_FMT(llgdLwnIsDriverOwnedMemoryPool(mph), false, "poolType = %s", llgd_lwn::PoolUtil::DescString(poolType));
        TEST_EQ_FMT(llgdLwnGetMemoryPoolPitchAddress(mph), mph->GetBufferAddress(), "poolType = %s", llgd_lwn::PoolUtil::DescString(poolType));
    }

    // Test driver pool
    TEST_EQ(llgdLwnIsDriverOwnedMemoryPool((LWNmemoryPool*)llgdLwnGetDeviceSyncPool(g_device)), true);
    TEST_EQ(llgdLwnIsDriverOwnedMemoryPool((LWNmemoryPool*)llgdLwnGetQueueMemoryPool(qh)), true);
    TEST_EQ(llgdLwnIsDriverOwnedMemoryPool((LWNmemoryPool*)llgdLwnGetQueueCommandMemoryPool(qh)), true);

    return true;
}

LLGD_DEFINE_TEST(DecodeMemoryPool, UNIT,
LwError Execute()
{
    DecodeMemoryPoolValidator v;
    v.Initialize();

    if (!v.Test())  { return LwError_IlwalidState; }
    else            { return LwSuccess;            }
}
); // LLGD_DEFINE_TEST
