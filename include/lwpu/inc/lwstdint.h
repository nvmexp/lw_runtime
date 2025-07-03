/*
 * Copyright (c) 2016, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 */

 /***************************************************************************\
|*                                                                           *|
|*                         LW Architecture Interface                         *|
|*                                                                           *|
|*  <lwstdint.h> defines integer types equivalent to those defined in the    *|
|*  C99 <stdint.h> header file.                                              *|
 \***************************************************************************/


#ifndef LWSTDINT_INCLUDED
#define LWSTDINT_INCLUDED

#ifdef __cplusplus
extern "C" {
#endif

#include "lwtypes.h"

 /***************************************************************************\
|*                                 Typedefs                                  *|
 \***************************************************************************/
typedef LwS8    int8_t;     /* -128 to 127                             */
typedef LwS16   int16_t;    /* -32768 to 32767                         */
typedef LwS32   int32_t;    /* -2147483648 to 2147483647               */
typedef LwS64   int64_t;    /* 2^-63 to 2^63-1                         */

typedef LwU8    uint8_t;    /* 0 to 255                                */
typedef LwU16   uint16_t;   /* 0 to 65535                              */
typedef LwU32   uint32_t;   /* 0 to 4294967295                         */
typedef LwU64   uint64_t;   /* 0 to 18446744073709551615               */

#ifdef __cplusplus
};
#endif

#endif /* LWSTDINT_INCLUDED */

