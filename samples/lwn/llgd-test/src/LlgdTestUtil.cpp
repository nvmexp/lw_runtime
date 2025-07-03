/*
 * Copyright (c) 2017, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property and
 * proprietary rights in and to this software and related documentation.  Any
 * use, reproduction, disclosure or distribution of this software and related
 * documentation without an express license agreement from LWPU Corporation
 * is strictly prohibited.
 */

#include <LlgdTestUtil.h>

#include <stdarg.h>

static LlgdLogMethod s_LogMethod = LlgdLogMethod::METHOD_LIBRARY;

void LlgdSetLogMethod(LlgdLogMethod method)
{
    s_LogMethod = method;
}

LlgdLogMethod LlgdGetLogMethod()
{
    return s_LogMethod;
}

LwError LlgdMsg(LlgdLogLevel level, const char *format, ...)
{
    LwError e = LwSuccess;

    va_list ap;

    va_start(ap, format);
    switch (s_LogMethod)
    {
    case LlgdLogMethod::METHOD_LIBRARY:
        // XXX register new log stream
        //NN_DETAIL_LWSCHED_VINFO(format, ap);
        LLGD_TEST_VLOG(format, ap);
        break;
    case LlgdLogMethod::METHOD_TEST:
        LLGD_TEST_VLOG(format, ap);

        break;
    default:
        ORIGINATE_ERROR(LwError_BadParameter);
    }
    va_end(ap);

    return e;
}

uint64_t LlgdAlignUp(uint64_t val, uint64_t alignment)
{
    return (alignment - (val % alignment)) % alignment + val;
}
