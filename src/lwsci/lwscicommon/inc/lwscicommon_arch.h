/*
 * Copyright (c) 2019-2022, LWPU CORPORATION. All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

/**
 * \file
 * \brief <b>LwSciCommon arch specific Interface</b>
 *
 * @b Description: This file contains LwSciCommon arch APIs
 */

#ifndef INCLUDED_LWSCICOMMON_ARCH_H
#define INCLUDED_LWSCICOMMON_ARCH_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \brief Returns current CPU time in microseconds
 *
 * \return uint64_t time in micro-seconds
 *
 */
uint64_t LwSciCommonGetTimeUS(void);

/**
 * \brief Full system Data Memory Barrier.
 *
 */
void LwSciCommonDMB(void);

#ifdef __cplusplus
}
#endif

#endif
