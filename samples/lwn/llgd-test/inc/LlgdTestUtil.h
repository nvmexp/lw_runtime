/*
 * Copyright (c) 2017, LWPU CORPORATION.  All rights reserved.
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

#include <functional>
#include <memory>
#include <sstream>
#include <iostream>

#if defined(LW_HOS)
#include <nn/nn_Log.h>
#define LLGD_TEST_LOG       NN_LOG
#define LLGD_TEST_VLOG      NN_VLOG
#else
#define LLGD_TEST_LOG       printf
#define LLGD_TEST_VLOG      vprintf
#endif

#define CHECK(x_) if (!(x_)) { LLGD_TEST_LOG("%s: %d '%s' assertion failed\n", __FILE__, __LINE__, #x_); __builtin_trap(); }

enum class LlgdLogLevel
{
    LEVEL_DEBUG,
    LEVEL_INFO,
    LEVEL_WARNING,
    LEVEL_ERROR
};

enum class LlgdLogMethod
{
    METHOD_LIBRARY,
    METHOD_TEST,
};

LwError LlgdMsg(LlgdLogLevel level, const char *format, ...);

void LlgdSetLogMask(uint32_t mask);
uint32_t LlgdGetLogMask();

void LlgdSetLogMethod(LlgdLogMethod log_method);
LlgdLogMethod LlgdGetLogMethod();

uint64_t LlgdAlignUp(uint64_t val, uint64_t alignemtn);

// Allocate aligned buffer of POD type (no constructor, destructor will be called)
//
// Note: Use this function to allocate aligned storage as possible.
// Don't use std::aligned_storage in wrong way for memory pool.
// We need to follow strict aliasing rules in C++ to use aligned_storage:
//  https://gist.github.com/shafik/848ae25ee209f698763cffee272a58f8
//  https://whereswalden.com/tag/stdaligned_storage/
// Correct way to use aligned_storage is like that:
//   auto spStorage = make_unique<typename aligned_storage<SIZE,ALIGN>::type>();
//   auto storage = new(spStorage.get()) uint8_t[SIZE];  // replacement new
// Otherwise, the behavior will be undefined. So use this function instead if you wouldn't like to write code above.
template <typename Pod_T>
std::unique_ptr<Pod_T, std::function<void(Pod_T*)>> LlgdAlignedAllocPodType(size_t sizeInBytes, size_t alignment)
{
    // Allocate memory storage which is aligned to a given argument
    auto alignPtr = [](Pod_T *ptr, uint64_t alignment) {
        return reinterpret_cast<Pod_T*>(LlgdAlignUp(reinterpret_cast<uint64_t>(ptr), alignment));
    };

    // Raw storage
    auto storage = reinterpret_cast<Pod_T*>(std::malloc(sizeInBytes + alignment));

    // Aligned storage
    auto alignedStorage = alignPtr(storage, alignment);

    // Wrap by smart pointer. Storage will be deleted automatically
    using UniqueWithDeleter = std::unique_ptr<Pod_T, std::function<void(Pod_T*)>>;
    return UniqueWithDeleter(alignedStorage, [storage](Pod_T*) { std::free(storage); });
}

// Maybe used from various places
using LlgdUniqueUint8PtrWithLwstomDeleter = std::unique_ptr<uint8_t, std::function<void(uint8_t*)>>;

#define LlgdDebug(fmt, ...) \
    LlgdMsg(LlgdLogLevel::LEVEL_DEBUG,    "%s: " fmt "\n", __FUNCTION__, ##__VA_ARGS__)
#define LlgdInfo(fmt, ...) \
    LlgdMsg(LlgdLogLevel::LEVEL_INFO,     "%s: " fmt "\n", __FUNCTION__, ##__VA_ARGS__)
#define LlgdWarn(fmt, ...) \
    LlgdMsg(LlgdLogLevel::LEVEL_WARNING,  "%s: " fmt "\n", __FUNCTION__, ##__VA_ARGS__)
#define LlgdErr(fmt, ...) \
    LlgdMsg(LlgdLogLevel::LEVEL_ERROR,    "%s: " fmt "\n", __FUNCTION__, ##__VA_ARGS__)

#if defined(LW_LINUX)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-declarations"
inline std::ostream & operator<<(std::ostream &s, std::nullptr_t) {
    return s << "nullptr";
}
#pragma GCC diagnostic pop
#endif

#define LlgdErrTestFailedMsg(left, leftValName, right, rightValName, cmpStr, fmt, ...) { \
    char strTmp[2048];                                                                   \
    std::stringstream valuesStr;                                                         \
    valuesStr << leftValName "=" << (left) << " " rightValName "=" << (right);           \
    std::snprintf(strTmp, 2048,                                                          \
        "(" #left " " cmpStr " " #right ") failed, %s, " fmt ", %s:%d",                  \
        valuesStr.str().c_str(), ##__VA_ARGS__, __FILE__, __LINE__);                     \
    LlgdErr("%s", strTmp);                                                               \
}

#define TEST_FMT(x,fmt,...)                                                        \
    if( !(x) ) {                                                                   \
        LlgdErr(#x " failed, %s:%d " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__); \
        return false;                                                              \
    }

#define TEST(x) TEST_FMT(x, "")

#define TEST_EQ_FMT(actual,expected,fmt,...) {                                            \
    if(!((actual) == (expected))) {                                                       \
        LlgdErrTestFailedMsg(actual,"actual",expected,"expected","==",fmt,##__VA_ARGS__); \
        return false;                                                                     \
    }                                                                                     \
}
#define TEST_EQ(actual, expected) TEST_EQ_FMT(actual, expected, "")

#define TEST_NEQ_FMT(left,right,fmt,...) {                                      \
    if(!((left) != (right))) {                                                  \
        LlgdErrTestFailedMsg(left,"left",right,"right","!=",fmt,##__VA_ARGS__); \
        return false;                                                           \
    }                                                                           \
}
#define TEST_NEQ(left, right) TEST_NEQ_FMT(left, right, "")

#define TEST_LT_FMT(left,right,fmt,...) {                                      \
    if(!((left) < (right))) {                                                  \
        LlgdErrTestFailedMsg(left,"left",right,"right","<",fmt,##__VA_ARGS__); \
        return false;                                                          \
    }                                                                          \
}
#define TEST_LT(left, right) TEST_LT_FMT(left, right, "")

#define TEST_LE_FMT(left,right,fmt,...) {                                       \
    if(!((left) <= (right))) {                                                  \
        LlgdErrTestFailedMsg(left,"left",right,"right","<=",fmt,##__VA_ARGS__); \
        return false;                                                           \
    }                                                                           \
}
#define TEST_LE(left, right) TEST_LE_FMT(left, right, "")

#define TEST_GT_FMT(left,right,fmt,...) {                                      \
    if(!((left) > (right))) {                                                  \
        LlgdErrTestFailedMsg(left,"left",right,"right",">",fmt,##__VA_ARGS__); \
        return false;                                                          \
    }                                                                          \
}
#define TEST_GT(left, expected) TEST_GT_FMT(left, expected, "")

#define TEST_GE_FMT(left,right,fmt,...) {                                       \
    if(!((left) >= (right))) {                                                  \
        LlgdErrTestFailedMsg(left,"left",right,"right",">=",fmt,##__VA_ARGS__); \
        return false;                                                           \
    }                                                                           \
}
#define TEST_GE(left, right) TEST_GE_FMT(left, right, "")

#define TEST_ALMOST_EQ(actual, expected) TEST(abs((actual)-(expected))/(expected) < 10e-4)

#define ORIGINATE_ERROR(e) do { \
    LlgdErr("%s:%d: Raising error %d", __FILE__, __LINE__, e); \
    return (e); \
} while (0)

#define ORIGINATE_ERROR_CLEANUP(e) do { \
    LlgdErr("%s:%d: Raising error %d", __FILE__, __LINE__, e); \
    LW_CHECK_ERROR_CLEANUP(e); \
} while (0)

#define PROPAGATE_ERROR(e) do { \
    LwError res = (e); \
    if (res != LwSuccess) { \
        LlgdErr("%s:%d: Propagating error %d", __FILE__, __LINE__, res); \
    } \
    LW_CHECK_ERROR(res); \
} while (0)

#define PROPAGATE_ERROR_CLEANUP(e) do { \
    LwError res = (e); \
    if (res != LwSuccess) { \
        LlgdErr("%s:%d: Propagating error %d", __FILE__, __LINE__, res); \
    } \
    LW_CHECK_ERROR_CLEANUP(res); \
} while (0)
