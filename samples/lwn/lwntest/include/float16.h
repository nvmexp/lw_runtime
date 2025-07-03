#ifndef __FLOAT16_H__
#define __FLOAT16_H__

/*
 * Copyright (c) 2001-2002, Lwpu Corporation.  All rights reserved.
 *
 * THE INFORMATION CONTAINED HEREIN IS PROPRIETARY AND CONFIDENTIAL TO
 * LWPU, CORPORATION.  USE, REPRODUCTION OR DISCLOSURE TO ANY THIRD PARTY
 * IS SUBJECT TO WRITTEN PRE-APPROVAL BY LWPU, CORPORATION.
 */
/*! \brief
 * Contains utility functions to colwert back and forth between LW 16-bit
 * floating point (GLhalf) and IEEE 32-bit floating point representations.
 *
 * float16.h
 */

#define PKFLT_MAN_BITS      10u       // 10 mantissa bits for s10e5
#define PKFLT_EXP_BITS      5u        // 5 exponent bits for s10e5
#define PKFLT_SGN_BIT       1         // 1 sign bit for s10e5
#define PKFLT_SATURATE_TO_MAXFLOAT 0  // Values greater than MAXFLOAT map to INF

#define PKFLT_TYPE          MAKE_PKFLT_TYPE(10,5)
#define PKFLT_TO_UI32       lwS10E5toUI32
#define PKFLT_TO_F32        lwS10E5toF32
#define UI32_TO_PKFLT       lwUI32toS10E5
#define F32_TO_PKFLT        lwF32toS10E5

#include "float_packed.h"

#endif // #ifndef __FLOAT16_H__
