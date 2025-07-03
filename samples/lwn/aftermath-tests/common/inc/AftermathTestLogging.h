/*
 * Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 */

#pragma once

#include <lwerror.h>

#include <stdint.h>

#include <sstream>

namespace AftermathTest {
namespace Logging {

enum class LogLevel
{
    LEVEL_DEBUG,
    LEVEL_INFO,
    LEVEL_WARNING,
    LEVEL_ERROR
};

enum class LogMethod
{
    METHOD_LIBRARY,
    METHOD_TEST,
};

LwError Msg(LogLevel level, const char *format, ...);

void SetLogMask(uint32_t mask);
uint32_t GetLogMask();

void SetLogMethod(LogMethod logMethod);
LogMethod GetLogMethod();

} // namespace Logging
} // namespace AftermathTest

#define AftermathDebug(fmt, ...) \
    AftermathTest::Logging::Msg(AftermathTest::Logging::LogLevel::LEVEL_DEBUG,    "%s: " fmt "\n", __FUNCTION__, ##__VA_ARGS__)
#define AftermathInfo(fmt, ...) \
    AftermathTest::Logging::Msg(AftermathTest::Logging::LogLevel::LEVEL_INFO,     "%s: " fmt "\n", __FUNCTION__, ##__VA_ARGS__)
#define AftermathWarn(fmt, ...) \
    AftermathTest::Logging::Msg(AftermathTest::Logging::LogLevel::LEVEL_WARNING,  "%s: " fmt "\n", __FUNCTION__, ##__VA_ARGS__)
#define AftermathErr(fmt, ...) \
    AftermathTest::Logging::Msg(AftermathTest::Logging::LogLevel::LEVEL_ERROR,    "%s: " fmt "\n", __FUNCTION__, ##__VA_ARGS__)

#define AftermathErrTestFailedMsg(left, leftValName, right, rightValName, cmpStr, fmt, ...) {   \
    char strTmp[2048];                                                                          \
    std::stringstream valuesStr;                                                                \
    valuesStr << leftValName "=" << (left) << " " rightValName "=" << (right);                  \
    std::snprintf(strTmp, 2048,                                                                 \
        "(" #left " " cmpStr " " #right ") failed, %s, " fmt ", %s:%d",                         \
        valuesStr.str().c_str(), ##__VA_ARGS__, __FILE__, __LINE__);                            \
    AftermathErr("%s", strTmp);                                                                 \
}

#define TEST_FMT(x,fmt,...)                                                                     \
    if( !(x) ) {                                                                                \
        AftermathErr(#x " failed, %s:%d " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__);         \
        return false;                                                                           \
    }

#define TEST(x) TEST_FMT(x, "")

#define TEST_EQ_FMT(actual,expected,fmt,...) {                                                  \
    if(!((actual) == (expected))) {                                                             \
        AftermathErrTestFailedMsg(actual,"actual",expected,"expected","==",fmt,##__VA_ARGS__);  \
        return false;                                                                           \
    }                                                                                           \
}

#define TEST_EQ(actual, expected) TEST_EQ_FMT(actual, expected, "")

#define TEST_NE_FMT(left,right,fmt,...) {                                                       \
    if(!((left) != (right))) {                                                                  \
        AftermathErrTestFailedMsg(left,"left",right,"right","!=",fmt,##__VA_ARGS__);            \
        return false;                                                                           \
    }                                                                                           \
}

#define TEST_NE(left, right) TEST_NE_FMT(left, right, "")

#define TEST_LT_FMT(left,right,fmt,...) {                                                       \
    if(!((left) < (right))) {                                                                   \
        AftermathErrTestFailedMsg(left,"left",right,"right","<",fmt,##__VA_ARGS__);             \
        return false;                                                                           \
    }                                                                                           \
}

#define TEST_LT(left, right) TEST_LT_FMT(left, right, "")

#define TEST_LE_FMT(left,right,fmt,...) {                                                       \
    if(!((left) <= (right))) {                                                                  \
        AftermathErrTestFailedMsg(left,"left",right,"right","<=",fmt,##__VA_ARGS__);            \
        return false;                                                                           \
    }                                                                                           \
}

#define TEST_LE(left, right) TEST_LE_FMT(left, right, "")

#define TEST_GT_FMT(left,right,fmt,...) {                                                       \
    if(!((left) > (right))) {                                                                   \
        AftermathErrTestFailedMsg(left,"left",right,"right",">",fmt,##__VA_ARGS__);             \
        return false;                                                                           \
    }                                                                                           \
}

#define TEST_GT(left, expected) TEST_GT_FMT(left, expected, "")

#define TEST_GE_FMT(left,right,fmt,...) {                                                       \
    if(!((left) >= (right))) {                                                                  \
        AftermathErrTestFailedMsg(left,"left",right,"right",">=",fmt,##__VA_ARGS__);            \
        return false;                                                                           \
    }                                                                                           \
}

#define TEST_GE(left, right) TEST_GE_FMT(left, right, "")

#define TEST_ALMOST_EQ(actual, expected) TEST(abs((actual)-(expected))/(expected) < 10e-4)

#define ORIGINATE_ERROR(e) do {                                                                 \
    AftermathErr("%s:%d: Raising error %d", __FILE__, __LINE__, e);                             \
    return (e);                                                                                 \
} while (0)

#define ORIGINATE_ERROR_CLEANUP(e) do {                                                         \
    AftermathErr("%s:%d: Raising error %d", __FILE__, __LINE__, e);                             \
    LW_CHECK_ERROR_CLEANUP(e);                                                                  \
} while (0)

#define PROPAGATE_ERROR(e) do {                                                                 \
    LwError res = (e);                                                                          \
    if (res != LwSuccess) {                                                                     \
        AftermathErr("%s:%d: Propagating error %d", __FILE__, __LINE__, res);                   \
    }                                                                                           \
    LW_CHECK_ERROR(res);                                                                        \
} while (0)

#define PROPAGATE_ERROR_CLEANUP(e) do {                                                         \
    LwError res = (e);                                                                          \
    if (res != LwSuccess) {                                                                     \
        AftermathErr("%s:%d: Propagating error %d", __FILE__, __LINE__, res);                   \
    }                                                                                           \
    LW_CHECK_ERROR_CLEANUP(res);                                                                \
} while (0)
