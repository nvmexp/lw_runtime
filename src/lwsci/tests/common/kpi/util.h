//! \file
//! \brief LwSciSync perf test util.
//!
//! \copyright
//! Copyright (c) 2021 LWPU Corporation. All rights reserved.
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

#define CHECK_LWSCIERR(e) {                                 \
    if (e != LwSciError_Success) {                          \
        printf ("%s, %s:%d, LwSci error %0x\n",             \
            __FILE__, __func__, __LINE__, e);               \
        return;                                             \
    }                                                       \
}

#endif // UTIL_H
