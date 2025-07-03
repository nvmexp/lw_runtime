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

class TrackMemoryPoolFlagsValidator {
public:
    void Initialize();
    bool Test();

private:
    static const size_t     SIZE = 4096;
    LlgdUniqueUint8PtrWithLwstomDeleter  pool;

    llgd_lwn::MemoryPoolHolder mph;
};

void TrackMemoryPoolFlagsValidator::Initialize()
{
    MemoryPoolBuilder pool_builder;

    pool = LlgdAlignedAllocPodType<uint8_t>(SIZE, 4096);

    pool_builder.SetDevice(g_device).SetDefaults()
                .SetFlags(MemoryPoolFlags::CPU_CACHED |
                          MemoryPoolFlags::GPU_CACHED);

    pool_builder.SetStorage(pool.get(), SIZE);
    CHECK(mph.Initialize(&pool_builder));
}

bool TrackMemoryPoolFlagsValidator::Test()
{
    llgdLwnTrackMemoryPool(mph);
    bool isTracked = llgdLwnIsTrackedMemoryPool(mph);
    if (!isTracked) {
        return false;
    }

    llgdLwnUntrackMemoryPool(mph);
    isTracked = llgdLwnIsTrackedMemoryPool(mph);
    if (isTracked) {
        return false;
    }

    return true;
}

LLGD_DEFINE_TEST(TrackMemoryPoolFlags, UNIT,
LwError Execute()
{
    TrackMemoryPoolFlagsValidator v;
    v.Initialize();

    if (!v.Test())  { return LwError_IlwalidState; }
    else            { return LwSuccess;            }
}
); // LLGD_DEFINE_TEST
