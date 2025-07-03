/*
 * Copyright (c) 2019-2020, LWPU CORPORATION. All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

/**
 * \file
 * \brief <b>LwSciCommon posix Interface</b>
 *
 * @b Description: This file contains LwSciCommon posix definitions
 */

#ifndef INCLUDED_LWSCICOMMON_POSIX_H
#define INCLUDED_LWSCICOMMON_POSIX_H

#include <pthread.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup lwscicommon_platformutils_api LwSciCommon APIs for platform utils.
 * List of APIs exposed at Inter-Element level.
 * @{
 */

/**
 * \brief Structure represents synchronization primitive for synchronizing
 *  access to critical sections within one process.
 *
 */
typedef pthread_mutex_t LwSciCommonMutex;

/**
 * @}
 */

#ifdef __cplusplus
}
#endif

#endif
