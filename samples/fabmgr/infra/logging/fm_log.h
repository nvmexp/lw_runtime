/*
 *  Copyright 2018-2021 LWPU Corporation.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 *
 */

#pragma once
#include "lwos.h"

#ifdef  __cplusplus
#include <string>
extern "C" {
#endif

// this attribute causes the compiler to check that the supplied arguments are in the
// correct format for the specified function. The Log macros take all arguments as
// strings (%d, %c etc) and this attribute causes printf to do type checking.
#ifdef __GNUC__
#define ATTRIBUTE_PRINTF(m, n) __attribute__((format(printf, m, n)))
#else // __GNUC__
#define ATTRIBUTE_PRINTF(m, n)
#endif // __GUNC__

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <string.h>
#include <errno.h>
#ifdef __linux__
#include <syslog.h>
#endif

#include "proc.h"
#include "spinlock.h"

typedef enum {
    FM_LOG_LEVEL_DISABLED = 0,
    FM_LOG_LEVEL_CRITICAL,
    FM_LOG_LEVEL_ERROR,
    FM_LOG_LEVEL_WARNING,
    FM_LOG_LEVEL_INFO,
    FM_LOG_LEVEL_DEBUG
} FMLogLevel_t;

typedef enum {
    FM_LOG_MODE_FILE = 0,
    FM_LOG_MODE_SYSLOG,
} FMLogMod_t;

extern FMLogLevel_t fmLogLevel;

int
fmLogPrintf(const char *fmt, ...) ATTRIBUTE_PRINTF(1, 2);

std::string
fmLogGetDateTimeStamp(void);

#if defined(LW_MODS) && !defined(LW_MODS_GDM_BUILD)
int fmLogModsPrintf(int lvl, const char *fmt, ...) ATTRIBUTE_PRINTF(2, 3);
#endif

#if defined(_WINDOWS)
void logWindowsEvent(const char *fmt, ...);
#endif

// in Debug build, we will print file and function and line number information. But
// for Release build, we will skip those information
#if defined(_WINDOWS) && !defined(__x86_64__)
#ifdef _DEBUG
#define _LOG(LVL, LVLSTR, dev_fmt, ...)                                  \
    ((fmLogLevel >= LVL)                                                          \
     ? (fmLogPrintf ("%s:[tid %lu] [%.06fs - %s:%s:%d] " dev_fmt "\n",          \
             LVLSTR, getLwrrentThreadId(), lwosGetTimer(&fmLogTimer) * 0.001f,     \
             __FILE__, __FUNCTION__, __LINE__, ##__VA_ARGS__))                     \
     : 0)
#else
#define _LOG(LVL, LVLSTR, dev_fmt, ...)                                   \
    ((fmLogLevel >= LVL)                                                           \
     ? (fmLogPrintf ("%s: [tid %lu]  [%.06fs] " dev_fmt "\n",                     \
             LVLSTR, getLwrrentThreadId(), lwosGetTimer(&fmLogTimer) * 0.001f,     \
              ##__VA_ARGS__))                                 \
     : 0)
#endif
#else
#if defined(LW_MODS) && !defined(LW_MODS_GDM_BUILD)
#define _LOG(LVL, LVLSTR, dev_fmt, ...)                                  \
    (fmLogModsPrintf (LVL, "FM: %s: " dev_fmt "\n",          \
             LVLSTR, ##__VA_ARGS__))
#elif defined(_DEBUG)
#define _LOG(LVL, LVLSTR, dev_fmt, ...)                                                \
    ((fmLogLevel >= LVL)                                                               \
     ? (fmLogPrintf ("[%s] [%s] [tid %llu] [%s:%s:%d] " dev_fmt "\n",                  \
                     fmLogGetDateTimeStamp().c_str(), LVLSTR, getLwrrentThreadId(),    \
                    __FILE__, __FUNCTION__, __LINE__, ##__VA_ARGS__))                  \
     : 0)
#else
#define _LOG(LVL, LVLSTR, dev_fmt, ...)                                                \
    ((fmLogLevel >= LVL)                                                               \
     ? (fmLogPrintf ("[%s] [%s] [tid %llu] " dev_fmt "\n",                             \
                     fmLogGetDateTimeStamp().c_str(), LVLSTR, getLwrrentThreadId(),    \
                     ##__VA_ARGS__))                                                   \
     : 0)
#endif
#endif

#define FM_LOG_CRITICAL(dev_fmt, ...)                                    \
    _LOG(FM_LOG_LEVEL_CRITICAL, "CRITICAL", dev_fmt, ##__VA_ARGS__)

#define FM_LOG_ERROR(dev_fmt,  ...)                                      \
    _LOG(FM_LOG_LEVEL_ERROR, "ERROR", dev_fmt, ##__VA_ARGS__)

#define FM_LOG_WARNING(dev_fmt, ...)                                     \
    _LOG(FM_LOG_LEVEL_WARNING, "WARNING", dev_fmt, ##__VA_ARGS__)

#define FM_LOG_INFO(dev_fmt, ...)                                        \
    _LOG(FM_LOG_LEVEL_INFO, "INFO", dev_fmt, ##__VA_ARGS__)

// make debug logs as empty in release builds
#if defined(_DEBUG) || (defined(LW_MODS) && !defined(LW_MODS_GDM_BUILD))
#define FM_LOG_DEBUG(dev_fmt, ...)                                       \
    _LOG(FM_LOG_LEVEL_DEBUG, "DEBUG", dev_fmt, ##__VA_ARGS__)
#else
#define FM_LOG_DEBUG(dev_fmt, ...)
#endif


// these macros are used to log directly to syslog. Used for critical errors/notice
// which FM needs to log directly instead of the usual log file
#ifdef __linux__
#define FM_SYSLOG_ERR(fmt, ...) \
        syslog(LOG_ERR, fmt, ##__VA_ARGS__);

#define FM_SYSLOG_NOTICE(fmt, ...)  \
        syslog(LOG_NOTICE, fmt, ##__VA_ARGS__);

#define FM_SYSLOG_WARNING(fmt, ...)  \
        syslog(LOG_WARNING, fmt, ##__VA_ARGS__);

#else
#define FM_SYSLOG_ERR(fmt, ...) \
        logWindowsEvent(fmt, ##__VA_ARGS__)

#define FM_SYSLOG_NOTICE(fmt, ...) \
        logWindowsEvent(fmt, ##__VA_ARGS__)

#define FM_SYSLOG_WARNING(fmt, ...) \
        logWindowsEvent(fmt, ##__VA_ARGS__)

#endif

// public methods
void
fabricManagerInitLog(unsigned int logLevel, char* logFileName,
                     bool appendToLog, unsigned int logFileSize, bool useSysLog);
void
fabricManagerShutdownLog(void);

#ifdef  __cplusplus
}
#endif