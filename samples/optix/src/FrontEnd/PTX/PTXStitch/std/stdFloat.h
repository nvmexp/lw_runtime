/*
 * LWIDIA_COPYRIGHT_BEGIN
 *
 * Copyright (c) 2012-2021, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 *
 * LWIDIA_COPYRIGHT_END
 */
/*
 *  Module name              : stdFloat.h
 *
 *  Description              :
 *
 *      IEEE 754 floating point representation functions.
 */

#ifndef stdFloat_INCLUDED
#define stdFloat_INCLUDED

/*--------------------------------- Includes ---------------------------------*/

#include "stdTypes.h"

#ifdef __cplusplus
extern "C" {
#endif

/*----------------------------------- Types ----------------------------------*/

typedef enum {
    stdIntImmType,
    stdBitSetImmType,
    stdF16ImmType,
    stdF32ImmType,
    stdF64ImmType,
    stdE6M9ImmType,
    stdE8M7ImmType,
    stdE8M10ImmType,
} stdImmType;

#define stdIsFloatImmType(kind) \
            ( (kind) >= stdF16ImmType )

#define  stdDoubleINF    U64_CONST( 0x7ff0000000000000 )
#define  stdDoubleQNAN   U64_CONST( 0x7ff8000000000000 )
#define  stdDoubleSNAN   U64_CONST( 0x7ff4000000000000 )

#define  stdFloatINF     U64_CONST( 0x7f800000 )
#define  stdFloatQNAN    U64_CONST( 0x7fc00000 )
#define  stdFloatSNAN    U64_CONST( 0x7fa00000 )

#define  stdTF32INF      U64_CONST( 0x3fc00 )
#define  stdTF32QNAN     U64_CONST( 0x3fe00 )
#define  stdTF32SNAN     U64_CONST( 0x3fd00 )

#define  stdF16INF       U64_CONST( 0x7c00 )
#define  stdF16QNAN      U64_CONST( 0x7e00 )
#define  stdF16SNAN      U64_CONST( 0x7d00 )

#define  stdBF16INF       U64_CONST( 0x7f80 )
#define  stdBF16QNAN      U64_CONST( 0x7fc0 )
#define  stdBF16SNAN      U64_CONST( 0x7fa0 )
                                         
#define  stdE6M9INF       U64_CONST( 0x7e00 )
#define  stdE6M9QNAN      U64_CONST( 0x7f00 )
#define  stdE6M9SNAN      U64_CONST( 0x7e80 )

// Following macro definitions are taken from //sw/gpgpu/lwca/apps/common/njcommon.h
#define REF_ROUND_ZERO  0
#define REF_ROUND_DOWN  1
#define REF_ROUND_UP    2
#define REF_ROUND_NEAR  3

/*----------------------- Floating Point Representation ----------------------*/

/*
 * Function         : Obtain textual representation of special float value
 * Parameters       : value           (I) IEEE 754 64 bit float representation
 * Function Result  : Textual representation if value is special, Nil otherwise
 */
String STD_CDECL stdGetSpecialFloatValueRepresentation( uInt64 value );


/*
 * Function         : Colwert integer representation of IEEE 754 64 bit float
 *                    into integer representation of 16, 32 or 64 bit float
 *                    with specified representation length. 
 *                    In case this representation length does not match the
 *                    'natural' representation length of the specified immType,
 *                    then lower order mantissa bits will be added or removed,
 *                    respectively.
 *                  
 * Parameters       : value           (I) float representation
 *                    immType         (I) Requested representation type of result
 *                    length          (I) Number of bits to encode final result in
 *                    result          (O) Result location
 * Function Result  : True iff. colwersion succeeded
 */
Bool STD_CDECL stdDecanonicalizeFloatValue_S( uInt64 value, stdImmType immType, uInt length, uInt64 *result );


/*
 * Function         : Colwert integer representation of IEEE 754 16, 32 or 64 bit float 
 *                    with specified representation length into integer representation of 64 bit float. 
 *                    In case this representation length does not match the
 *                    'natural' representation length of the specified immType,
 *                    then lower order mantissa bits will be ignored or assumed zero,
 *                    respectively.
 * Parameters       : value           (I) float representation
 *                    immType         (I) Representation type of 'value'
 * Function Result  : 64 bit float representation of 'value'
 */
uInt64 STD_CDECL stdCanonicalizeFloatValue_S( uInt64 value, stdImmType immType, uInt length );


/*
 * Function         : Colwert integer representation of IEEE 754 64 bit float
 *                    into integer representation of 16, 32 or 64 bit float 
 * Parameters       : value           (I) float representation
 *                    immType         (I) Requested representation type of result
 *                    result          (O) Result location
 * Function Result  : True iff. colwersion succeeded
 */
Bool STD_CDECL stdDecanonicalizeFloatValue( uInt64 value, stdImmType immType, uInt64 *result );


/*
 * Function         : Colwert integer representation of IEEE 754 16, 32 or 64 bit float 
 *                    into integer representation of 64 bit float. 
 * Parameters       : value           (I) float representation
 *                    immType         (I) Representation type of 'value'
 * Function Result  : 64 bit float representation of 'value'
 */
uInt64 STD_CDECL stdCanonicalizeFloatValue( uInt64 value, stdImmType immType );




#ifdef __cplusplus
}
#endif

#endif  /* stdFloat_INCLUDED */
