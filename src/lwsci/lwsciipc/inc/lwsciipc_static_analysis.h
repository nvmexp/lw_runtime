/*
 * Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited
 */

#ifndef INCLUDED_LWSCIIPC_STATIC_ANALYSIS_H
#define INCLUDED_LWSCIIPC_STATIC_ANALYSIS_H

#include <stdint.h>
#include <stdbool.h>
#if defined(__QNX__)
#include <unistd.h>
#if defined(LW_IS_TRACER_ENABLED)
#include <tracer.h>
#endif
#endif /* __QNX__ */

/** CERT INT30-C:
 *  Precondition test to ensure casting signed integer to unsigned integer
 *  do not wrap.
 */
static inline bool LwSciIpcCastS32toU16(int32_t op1, uint16_t *result)
{
    bool e = true;

    if ((op1 < 0) || (op1 > (int32_t)INT16_MAX)) {
         e = false;
    } else {
         *result = (uint16_t)op1;
    }
    return e;
}

#endif /* INCLUDED_LWSCIIPC_STATIC_ANALYSIS_H */

