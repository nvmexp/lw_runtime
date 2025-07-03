/*
 * Copyright (c) 2014 - 2015 LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#ifndef __LWOGTYPES_H
#define __LWOGTYPES_H

#include "lwctassert.h"
#include "ossymbols.h"

#if (defined(_MSC_VER) && (_MSC_VER < 1600)) || defined(LW_MACINTOSH_OS9) || defined(vxworks) || defined(__DJGPP__)
#if !defined(LW_WINDOWS_64)
typedef unsigned int uintptr_t;
typedef int intptr_t;
#endif
#else
#define __STDC_LIMIT_MACROS
#include <stdint.h>
#define LW_STDINT_INCLUDED
#endif

#ifndef LW_STDINT_INCLUDED

#ifdef _MSC_VER
typedef __int8                  int8_t;
typedef unsigned __int8         uint8_t;
typedef __int16                 int16_t;
typedef unsigned __int16        uint16_t;
typedef __int32                 int32_t;
typedef unsigned __int32        uint32_t;
typedef __int64                 int64_t;
typedef unsigned __int64        uint64_t;
#else
typedef signed char int8_t;
typedef unsigned char uint8_t;
typedef short int16_t;
typedef unsigned short uint16_t;
#ifdef __alpha
typedef int int32_t;
typedef unsigned int uint32_t;
#else
typedef long int32_t;
typedef unsigned long uint32_t;
#endif
typedef long long int64_t;
typedef unsigned long long uint64_t;
#endif

#endif // LW_STDINT_INCLUDED

// On some platforms stdint.h does not include these macros when compiling for C++
// even when the __STDC_LIMIT_MACROS macro is set.
#if !defined(INT8_MAX)
#define INT8_MAX (127)
#define INT8_MIN (-128)
#define UINT8_MAX (255)
#define INT16_MAX (32767)
#define INT16_MIN (-32767-1)
#define UINT16_MAX (65535U)
#define INT32_MAX (2147483647L)
#define INT32_MIN (-2147483647L-1)
#define UINT32_MAX (4294967295UL)
#define INT64_MAX (9223372036854775807LL)
#define INT64_MIN (-9223372036854775807LL-1)
#define UINT64_MAX (18446744073709551615ULL)

#if defined(LW_WINDOWS_64)
 #define INTPTR_MIN      INT64_MIN
 #define INTPTR_MAX      INT64_MAX
 #define UINTPTR_MAX     UINT64_MAX
#else
 #define INTPTR_MIN      INT32_MIN
 #define INTPTR_MAX      INT32_MAX
 #define UINTPTR_MAX     UINT32_MAX
#endif
#endif

ct_assert(sizeof(int8_t)==1);
ct_assert(sizeof(uint8_t)==1);
ct_assert(sizeof(int16_t)==2);
ct_assert(sizeof(uint16_t)==2);
ct_assert(sizeof(int32_t)==4);
ct_assert(sizeof(uint32_t)==4);
ct_assert(sizeof(int64_t)==8);
ct_assert(sizeof(uint64_t)==8);


#endif // __LWOGTYPES_H
