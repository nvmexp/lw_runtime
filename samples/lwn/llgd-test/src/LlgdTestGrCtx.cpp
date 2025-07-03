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

#include <lwndevtools_bootstrap.h>

inline bool TestGrCtx()
{
    auto devtools = lwnDevtoolsBootstrap();

    bool success = false;
    static const size_t BUFFER_SIZE = 80000;
    uint8_t buffer[BUFFER_SIZE];
    size_t size = 0;

    llgd_lwn::QueueHolder qh;
    qh.Initialize(g_device);

    success = devtools->GetGrCtxSizeForQueue(qh, &size);
    TEST_FMT(success, "get ctx size failed!");
    TEST_FMT(size > 0, "0 ctx size!!");
    TEST_FMT(size < BUFFER_SIZE, "ctx too big!");

    memset(buffer, 0, BUFFER_SIZE);

    success = devtools->GetGrCtxForQueue(qh, buffer, size);
    TEST_FMT(success, "get ctx failed!");

    for (int i = 0; i < 60; i++) {
        // Look for one non NULL byte in the first 60.
        if (buffer[i]) { return true; }
    }

    return false;
}

LLGD_DEFINE_TEST(GrCtx, UNIT, LwError Execute() { return TestGrCtx() ? LwSuccess : LwError_IlwalidState; });
