#ifndef __FLOATU11_H__
#define __FLOATU11_H__

/*
 * Copyright (c) 2001-2002, Lwpu Corporation.  All rights reserved.
 *
 * THE INFORMATION CONTAINED HEREIN IS PROPRIETARY AND CONFIDENTIAL TO
 * LWPU, CORPORATION.  USE, REPRODUCTION OR DISCLOSURE TO ANY THIRD PARTY
 * IS SUBJECT TO WRITTEN PRE-APPROVAL BY LWPU, CORPORATION.
 */
/*! \brief
 * Contains utility functions to colwert back and forth between s5e5
 * floating point and IEEE 32-bit floating point representations.
 *
 * floatu11.h
 */

#define PKFLT_MAN_BITS      6u        // 5 mantissa bits for u6e5
#define PKFLT_EXP_BITS      5u        // 5 exponent bits for u6e5
#define PKFLT_SGN_BIT       0         // 0 sign bit for u6e5
#define PKFLT_SATURATE_TO_MAXFLOAT 1  // Values greater than MAXFLOAT map to INF

#define PKFLT_TYPE          MAKE_PKFLT_TYPE(6,5)
#define PKFLT_TO_UI32       MAKE_PKFLT_TO_UI32(6,5)
#define PKFLT_TO_F32        MAKE_PKFLT_TO_F32(6,5)
#define UI32_TO_PKFLT       MAKE_UI32_TO_PKFLT(6,5)
#define F32_TO_PKFLT        MAKE_F32_TO_PKFLT(6,5)

#include "float_packed.h"

#endif // #ifndef __FLOATU11_H__
