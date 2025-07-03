//! \file
//! \brief Utility functions for lwscistrem perf tests.
//!
//! \copyright
//! Copyright (c) 2019-2021 LWPU Corporation. All rights reserved.
//!
//! LWPU Corporation and its licensors retain all intellectual property and
//! proprietary rights in and to this software, related documentation and any
//! modifications thereto. Any use, reproduction, disclosure or distribution
//! of this software and related documentation without an express license
//! agreement from LWPU Corporation is strictly prohibited.

#ifndef UTIL_H
#define UTIL_H

#include <cstdlib>
#include <cstdio>
#include <cassert>
#include <cstring>
#include "lwscistream.h"

#ifdef __cplusplus
extern "C" {
#endif

constexpr uint32_t NUM_PACKETS{ 1U };
constexpr uint32_t NUM_ELEMENTS{ 1U };
constexpr uint32_t MAX_SYNCS{ 8U };
constexpr uint32_t MAX_CONS{ 8U };

constexpr int64_t  QUERY_TIMEOUT{ -1 };
constexpr int64_t  LW_WAIT_INFINITE{ 0xFFFFFFFF };

#define CHECK_LWSCIERR(e) {                                 \
    if (e != LwSciError_Success) {                          \
        printf ("%s, %s:%d, LwSci error %0x\n",             \
            __FILE__, __func__, __LINE__, e);               \
        exit(-1);                                           \
    }                                                       \
}

enum TestType {
    IntraProcess,
    CrossProcProd,
    CrossProcCons
};

struct TestArg {
    TestType            testType{ IntraProcess };
    bool                isC2c{ false };
    uint32_t            numConsumers{ 1U };
    uint32_t            consIndex{ 0U };
    uint32_t            numFrames{ 100U };
    uint32_t            bufSize{ 1024U * 1024U };
    uint32_t            numSyncs{ 1U };
    uint32_t            sleepUs{ 0U };
    bool                latency{ false };
    bool                verbose{ false };
    double              avgTarget{ 0.0f };
    double              maxTarget{ 0.0f };
};

#ifdef __cplusplus
}
#endif
#endif
