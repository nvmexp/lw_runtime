/*
 * Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property and
 * proprietary rights in and to this software and related documentation.  Any
 * use, reproduction, disclosure or distribution of this software and related
 * documentation without an express license agreement from LWPU Corporation
 * is strictly prohibited.
 */

#include <stdarg.h>
#include <nn/nn_Log.h>

#include <AftermathTestLogging.h>

namespace AftermathTest {
namespace Logging {

static LogMethod s_LogMethod = LogMethod::METHOD_LIBRARY;

void SetLogMethod(LogMethod logMethod)
{
    s_LogMethod = logMethod;
}

LogMethod GetLogMethod()
{
    return s_LogMethod;
}

LwError Msg(LogLevel level, const char* format, ...)
{
    LwError e = LwSuccess;

    va_list ap;

    va_start(ap, format);
    switch (s_LogMethod) {
    case LogMethod::METHOD_LIBRARY:
        // XXX register new log stream
        //NN_DETAIL_LWSCHED_VINFO(format, ap);
        NN_VLOG(format, ap);
        break;
    case LogMethod::METHOD_TEST:
        NN_VLOG(format, ap);
        break;
    default:
        ORIGINATE_ERROR(LwError_BadParameter);
    }
    va_end(ap);

    return e;
}

} // namespace Logging
} // namespace AftermathTest
