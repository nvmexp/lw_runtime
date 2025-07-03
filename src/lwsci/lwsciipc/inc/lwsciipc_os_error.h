/*
 * Copyright (c) 2020, LWPU CORPORATION. All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#ifndef INCLUDED_LWSCIIPC_OS_ERROR_H
#define INCLUDED_LWSCIIPC_OS_ERROR_H

#include <sys/types.h>
#include <stdint.h>
#include <lwscierror.h>
#if defined(LW_IS_TRACER_ENABLED)
#include <tracer.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief This API is used to retrieve LwSciError for a given system error code
 *
 * @param[in] err system error code
 *
 * \return LwSciError, colwerted error code:
 * - LwSciError_Unknown is the error code cannot be colwerted.
 */
LwSciError ErrnoToLwSciErr(int32_t err);

/**
 * @brief This API is used to retrieve system error code for a given LwSciError
 *
 * @param[in] lwSciErr LwSciError error code
 *
 * \return int32, colwerted error code:
 * - -1 if the error code cannot be colwerted.
 */
int32_t LwSciErrToErrno(LwSciError lwSciErr);

#ifdef __cplusplus
}
#endif

#endif /* INCLUDED_LWSCIIPC_OS_ERROR_H */
