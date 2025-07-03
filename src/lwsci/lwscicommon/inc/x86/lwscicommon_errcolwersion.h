/*
 * Copyright (c) 2019-2022, LWPU CORPORATION. All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#ifndef INCLUDED_LWSCICOMMON_ERRCOLWERSION_H
#define INCLUDED_LWSCICOMMON_ERRCOLWERSION_H

#include <sys/types.h>
#include <stdint.h>
#include "lwscierror.h"

#if defined(__x86_64__)
#include "lwstatus.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

#if defined(__x86_64__)
/**
 * @brief This API is used to retrieve LwSciError for a given LW_STATUS
 *
 * @param[in] status LW_STATUS error code
 *
 * \return LwSciError, colwerted error code:
 * - LwSciError_Unknown if the error code cannot be colwerted.
 *
 * @note This API is for x86. Marking it Non-Safe for PLC.
 */
LwSciError LwStatusToLwSciErr(
    LW_STATUS status);

/**
 * @brief This API is used to retrieve LW_STATUS for a given LwSciError
 *
 * @param[in] status LW_STATUS error code
 *
 * \return LwSciError, colwerted error code:
 * - LwSciError_Unknown if the error code cannot be colwerted.
 *
 * @note This API is for x86. Marking it Non-Safe for PLC.
 *
 */
LW_STATUS LwSciErrToLwStatus(
    LwSciError lwSciErr);

/**
 * @brief This API is used to colwert LW_STATUS to error string
 *
 * @param[in] lwStatusIn LW_STATUS error code
 *
 * \return const char*, error code string
 *
 * @note This API is for x86. Marking it Non-Safe for PLC.
 *
 */
const char* LwStatusToString(
    LW_STATUS lwStatusIn);

#endif /* defined(__x86_64__) */

/**
 * @brief This API is used to retrieve LwSciError for a given system error code
 *
 * @param[in] err system error code
 *
 * \return LwSciError, colwerted error code:
 * - LwSciError_Unknown is the error code cannot be colwerted.
 *
 */
LwSciError ErrnoToLwSciErr(
    int32_t err);

/**
 * @brief This API is used to retrieve system error code for a given LwSciError
 *
 * @param[in] lwSciErr LwSciError error code
 *
 * \return int32, colwerted error code:
 * - -1 if the error code cannot be colwerted.
 *
 */
int32_t LwSciErrToErrno(
    LwSciError lwSciErr);

/**
 * @brief This API is used to colwert LwSciError to error string
 *
 * @param[in] lwSciErr LwSciError error code
 *
 * \return const char*, error code string
 * - UNKNOWN if LwSciError error code provided is not within LwSciError enum
 *   bounds
 *
 * TODO
 * @note This API is not used by LwStreams. Marking it Non-Safe for PLC.
 */
const char* LwSciErrToErrnoStr(
    LwSciError lwSciErr);

#ifdef __cplusplus
}
#endif

#endif /* INCLUDED_LWSCICOMMON_ERRCOLWERSION_H */
