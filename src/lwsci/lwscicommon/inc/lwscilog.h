/*
 * Copyright (c) 2019-2021, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 */

/**
 * \file
 * \brief <b>LwSci Logger Interface</b>
 *
 * @b Description: This file contains LwSci Logging interfaces
 */

#ifndef INCLUDED_LWSCILOG_H
#define INCLUDED_LWSCILOG_H

#include <stdio.h>
#include <stdarg.h>
#include <inttypes.h>

#if defined(LW_TEGRA_MIRROR_INCLUDES)
#include "mobile_common.h"
#elif defined(LW_TEGRA_DIRECT_INCLUDES)
#include "lwos.h"
#endif

/**
 * @defgroup lwscicommon_blanket_statements LwSciCommon blanket statements.
 * Generic statements applicable for LwSciCommon interfaces.
 * @{
 */

/**
 * \page page_blanket_statements LwSciCommon blanket statements
 *
 * \section element_dependency Dependency on other elements
 * LwSciCommon calls below liblwos_s3_safety interfaces:
 * - LwOsDebugPrintStr() to print string to system logger.
 * - LwOsDebugPrintStrInt() to print signed integer to system logger.
 * - LwOsDebugPrintStrUInt() to print unsigned integer to system logger.
 * - LwOsDebugPrintStrSLong() to print long signed integer to system logger.
 * - LwOsDebugPrintStrULong() to print unsigned long integer to system logger.
 * - LwOsDebugPrintStrHexULong() to print unsigned long integer in hex format to
 *  system logger.
 *
 */

/**
 * @}
 */



#ifdef __cplusplus
extern "C" {
#endif

static inline void LwSciTracePrintf(const char *format, ...)
{
#if (LW_IS_SAFETY != 0)
    (void)format;
#else
    va_list ap;

    va_start(ap, format);
    vprintf(format, ap);
    va_end(ap);
#endif
}

/* Enable LWSCI_DEBUG to switch on DEBUG macros. */
#define LWSCI_DEBUG 0

#if LWSCI_DEBUG

#define LWSCI_FNENTRY(fmt, ...) \
    LwSciTracePrintf("[ENTER: %s]: " fmt "\n", __FUNCTION__, ##__VA_ARGS__); \

#define LWSCI_FNEXIT(fmt, ...) \
    LwSciTracePrintf("[EXIT: %s]: " fmt "\n", __FUNCTION__, ##__VA_ARGS__); \

#else
#define LWSCI_FNENTRY(fmt, ...)
#define LWSCI_FNEXIT(fmt, ...)
#endif

#define LWSCI_ERR_LOG_LEVEL     0
#define LWSCI_WARN_LOG_LEVEL    (LWSCI_ERR_LOG_LEVEL + 1)
#define LWSCI_INFO_LOG_LEVEL    (LWSCI_WARN_LOG_LEVEL + 1)

/* Log level controlling amount of logs displayed */
#define LWSCI_LWRRENT_LOG_LEVEL LWSCI_ERR_LOG_LEVEL

#if LWSCI_LWRRENT_LOG_LEVEL >= LWSCI_ERR_LOG_LEVEL
#if defined(__x86_64__) || !defined(LW_TEGRA_DIRECT_INCLUDES)
#define LWSCI_ERR(fmt, ...) LwSciTracePrintf("[ERROR: %s]: " fmt, \
                                            __FUNCTION__, ##__VA_ARGS__)
#define LWSCI_ERR_STR(str) LwSciTracePrintf("[ERROR: %s]: %s\n", \
                                         __FUNCTION__, str)
#define LWSCI_ERR_INT(str, val) LwSciTracePrintf("[ERROR: %s]: " str "%d\n", \
                                         __FUNCTION__, val)
#define LWSCI_ERR_UINT(str, val) LwSciTracePrintf("[ERROR: %s]: " str "%u\n", \
                                         __FUNCTION__, val)
#define LWSCI_ERR_SLONG(str, val) LwSciTracePrintf("[ERROR: %s]: " str "%ld\n", \
                                         __FUNCTION__, val)
#define LWSCI_ERR_ULONG(str, val) LwSciTracePrintf("[ERROR: %s]: " str "%lu\n", \
                                         __FUNCTION__, val)
#define LWSCI_ERR_HEXUINT(str, val) LwSciTracePrintf("[ERROR: %s]: " str "%x\n", \
                                         __FUNCTION__, val)
#else
// SLOG2_ERROR is only defined for qnx. Need to define it for linux
#ifndef SLOG2_ERROR
#define SLOG2_ERROR 2
#endif

/**
 * \brief Thin wrapper around LwOsDebugPrint safety logging api which prints strings.
 *
 * \param[in] str debug string that needs to be printed.
 *
 * \conlwrrency:
 *  - Thread-safe: Yes
 *  - Synchronization
 *      - No access to any shared data.
 *
 */
#define LWSCI_ERR_STR(str) \
    (void)LwOsDebugPrintStr(LWOS_SLOG_CODE_LWSCI, SLOG2_ERROR, str)

/**
 * \brief Thin wrapper around LwOsDebugPrint safety logging api which prints
 *  signed integer.
 *
 * \param[in] str debug string that needs to be printed.
 * \param[in] val signed integer value that needs to be printed.
 *
 * \conlwrrency:
 *  - Thread-safe: Yes
 *  - Synchronization
 *      - No access to any shared data.
 *
 */
#define LWSCI_ERR_INT(str, val) \
    (void)LwOsDebugPrintStrInt(LWOS_SLOG_CODE_LWSCI, SLOG2_ERROR, str, (int32_t)val)

/**
 * \brief Thin wrapper around LwOsDebugPrint safety logging api which prints
 *  unsigned integer.
 *
 * \param[in] str debug string that needs to be printed.
 * \param[in] val unsigned integer that needs to be printed.
 *
 * \conlwrrency:
 *  - Thread-safe: Yes
 *  - Synchronization
 *      - No access to any shared data.
 *
 */
#define LWSCI_ERR_UINT(str, val) \
    (void)LwOsDebugPrintStrUInt(LWOS_SLOG_CODE_LWSCI, SLOG2_ERROR, str, (uint32_t)val)

/**
 * \brief Thin wrapper around LwOsDebugPrint safety logging api which prints
 *  signed long integer.
 *
 * \param[in] str debug string that needs to be printed.
 * \param[in] val signed long integer that needs to be printed.
 *
 * \conlwrrency:
 *  - Thread-safe: Yes
 *  - Synchronization
 *      - No access to any shared data.
 *
 */
#define LWSCI_ERR_SLONG(str, val) \
    (void)LwOsDebugPrintStrSLong(LWOS_SLOG_CODE_LWSCI, SLOG2_ERROR, str, (int64_t)val)

/**
 * \brief Thin wrapper around LwOsDebugPrint safety logging api which prints
 *  unsigned long integer.
 *
 * \param[in] str debug string that needs to be printed.
 * \param[in] val unsigned long integer that needs to be printed.
 *
 * \conlwrrency:
 *  - Thread-safe: Yes
 *  - Synchronization
 *      - No access to any shared data.
 *
 */
#define LWSCI_ERR_ULONG(str, val) \
    (void)LwOsDebugPrintStrULong(LWOS_SLOG_CODE_LWSCI, SLOG2_ERROR, str, (uint64_t)val)

/**
 * \brief Thin wrapper around LwOsDebugPrint safety logging api which prints
 *  unsigned integer in hexadecimal format.
 *
 * \param[in] str debug string that needs to be printed.
 * \param[in] val unsigned integer that needs to be printed.
 *
 * \conlwrrency:
 *  - Thread-safe: Yes
 *  - Synchronization
 *      - No access to any shared data.
 *
 */
#define LWSCI_ERR_HEXUINT(str, val) \
    (void)LwOsDebugPrintStrHexULong(LWOS_SLOG_CODE_LWSCI, SLOG2_ERROR, str, (uint64_t)val)
#endif

#else
#define LWSCI_ERR(fmt, ...)
#endif

#if LWSCI_LWRRENT_LOG_LEVEL >= LWSCI_WARN_LOG_LEVEL
#define LWSCI_WARN(fmt, ...) LwSciTracePrintf("[WARN: %s]: " fmt "\n", \
                                            __FUNCTION__, ##__VA_ARGS__)
#else
#define LWSCI_WARN(fmt, ...)
#endif

#if LWSCI_LWRRENT_LOG_LEVEL >= LWSCI_INFO_LOG_LEVEL
#define LWSCI_INFO(fmt, ...) LwSciTracePrintf("[INFO: %s]: " fmt "\n", \
                                            __FUNCTION__, ##__VA_ARGS__)
#else
#define LWSCI_INFO(fmt, ...)
#endif

#ifdef __cplusplus
}
#endif

#endif /* INCLUDED_LWSCILOG_H */
