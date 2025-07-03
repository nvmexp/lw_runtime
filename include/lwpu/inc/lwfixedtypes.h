/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2022 LWPU CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#ifndef LWFIXEDTYPES_INCLUDED
#define LWFIXEDTYPES_INCLUDED

#include "lwtypes.h"

/*!
 * Fixed-point master data types.
 *
 * These are master-types represent the total number of bits contained within
 * the FXP type.  All FXP types below should be based on one of these master
 * types.
 */
typedef LwS16                                                         LwSFXP16;
typedef LwS32                                                         LwSFXP32;
typedef LwS64                                                         LwSFXP64;
typedef LwU16                                                         LwUFXP16;
typedef LwU32                                                         LwUFXP32;
typedef LwU64                                                         LwUFXP64;


/*!
 * Fixed-point data types.
 *
 * These are all integer types with precision indicated in the naming of the
 * form: Lw<sign>FXP<num_bits_above_radix>_<num bits below radix>.  The actual
 * size of the data type is callwlated as num_bits_above_radix +
 * num_bit_below_radix.
 *
 * All of these FXP types should be based on one of the master types above.
 */
typedef LwSFXP16                                                    LwSFXP11_5;
typedef LwSFXP16                                                    LwSFXP4_12;
typedef LwSFXP16                                                     LwSFXP8_8;
typedef LwSFXP32                                                    LwSFXP8_24;
typedef LwSFXP32                                                   LwSFXP10_22;
typedef LwSFXP32                                                   LwSFXP16_16;
typedef LwSFXP32                                                   LwSFXP18_14;
typedef LwSFXP32                                                   LwSFXP20_12;
typedef LwSFXP32                                                    LwSFXP24_8;
typedef LwSFXP32                                                    LwSFXP27_5;
typedef LwSFXP32                                                    LwSFXP28_4;
typedef LwSFXP32                                                    LwSFXP29_3;
typedef LwSFXP32                                                    LwSFXP31_1;
typedef LwSFXP64                                                   LwSFXP52_12;

typedef LwUFXP16                                                    LwUFXP0_16;
typedef LwUFXP16                                                    LwUFXP4_12;
typedef LwUFXP16                                                     LwUFXP8_8;
typedef LwUFXP32                                                    LwUFXP3_29;
typedef LwUFXP32                                                    LwUFXP4_28;
typedef LwUFXP32                                                    LwUFXP7_25;
typedef LwUFXP32                                                    LwUFXP8_24;
typedef LwUFXP32                                                    LwUFXP9_23;
typedef LwUFXP32                                                   LwUFXP10_22;
typedef LwUFXP32                                                   LwUFXP15_17;
typedef LwUFXP32                                                   LwUFXP16_16;
typedef LwUFXP32                                                   LwUFXP18_14;
typedef LwUFXP32                                                   LwUFXP20_12;
typedef LwUFXP32                                                    LwUFXP24_8;
typedef LwUFXP32                                                    LwUFXP25_7;
typedef LwUFXP32                                                    LwUFXP26_6;
typedef LwUFXP32                                                    LwUFXP28_4;

typedef LwUFXP64                                                   LwUFXP40_24;
typedef LwUFXP64                                                   LwUFXP48_16;
typedef LwUFXP64                                                   LwUFXP52_12;

/*!
 * Utility macros used in colwerting between signed integers and fixed-point
 * notation.
 *
 * - COMMON - These are used by both signed and unsigned.
 */
#define LW_TYPES_FXP_INTEGER(x, y)                              ((x)+(y)-1):(y)
#define LW_TYPES_FXP_FRACTIONAL(x, y)                                 ((y)-1):0
#define LW_TYPES_FXP_FRACTIONAL_MSB(x, y)                       ((y)-1):((y)-1)
#define LW_TYPES_FXP_FRACTIONAL_MSB_ONE                              0x00000001
#define LW_TYPES_FXP_FRACTIONAL_MSB_ZERO                             0x00000000
#define LW_TYPES_FXP_ZERO                                                   (0)

/*!
 * - UNSIGNED - These are only used for unsigned.
 */
#define LW_TYPES_UFXP_INTEGER_MAX(x, y)                      (~(LWBIT((y))-1U))
#define LW_TYPES_UFXP_INTEGER_MIN(x, y)                                    (0U)

/*!
 * - SIGNED - These are only used for signed.
 */
#define LW_TYPES_SFXP_INTEGER_SIGN(x, y)                ((x)+(y)-1):((x)+(y)-1)
#define LW_TYPES_SFXP_INTEGER_SIGN_NEGATIVE                          0x00000001
#define LW_TYPES_SFXP_INTEGER_SIGN_POSITIVE                          0x00000000
#define LW_TYPES_SFXP_S32_SIGN_EXTENSION(x, y)                           31:(x)
#define LW_TYPES_SFXP_S32_SIGN_EXTENSION_POSITIVE(x, y)              0x00000000
#define LW_TYPES_SFXP_S32_SIGN_EXTENSION_NEGATIVE(x, y)      (LWBIT(32-(x))-1U)
#define LW_TYPES_SFXP_INTEGER_MAX(x, y)                         (LWBIT((x))-1U)
#define LW_TYPES_SFXP_INTEGER_MIN(x, y)                      (~(LWBIT((x))-1U))
#define LW_TYPES_SFXP_S64_SIGN_EXTENSION(x, y)                           63:(x)
#define LW_TYPES_SFXP_S64_SIGN_EXTENSION_POSITIVE(x, y)      0x0000000000000000
#define LW_TYPES_SFXP_S64_SIGN_EXTENSION_NEGATIVE(x, y)    (LWBIT64(64-(x))-1U)
#define LW_TYPES_SFXP_S64_INTEGER_MAX(x, y)                 (LWBIT64((x)-1)-1U)
#define LW_TYPES_SFXP_S64_INTEGER_MIN(x, y)              (~(LWBIT64((x)-1)-1U))

/*!
 * Colwersion macros used for colwerting between integer and fixed point
 * representations.  Both signed and unsigned variants.
 *
 * Warning:
 * Note that most of the macros below can overflow if applied on values that can
 * not fit the destination type.  It's caller responsibility to ensure that such
 * situations will not occur.
 *
 * Some colwersions perform some commonly performed tasks other than just
 * bit-shifting:
 *
 * - _SCALED:
 *   For integer -> fixed-point we add handling divisors to represent
 *   non-integer values.
 *
 * - _ROUNDED:
 *   For fixed-point -> integer we add rounding to integer values.
 */

// 32-bit Unsigned FXP:
#define LW_TYPES_U32_TO_UFXP_X_Y(x, y, integer)                               \
    ((LwUFXP##x##_##y) (((LwU32) (integer)) <<                                \
                        DRF_SHIFT(LW_TYPES_FXP_INTEGER((x), (y)))))

#define LW_TYPES_U32_TO_UFXP_X_Y_SCALED(x, y, integer, scale)                 \
    ((LwUFXP##x##_##y) ((((((LwU32) (integer)) <<                             \
                        DRF_SHIFT(LW_TYPES_FXP_INTEGER((x), (y))))) /         \
                            (scale)) +                                        \
                        ((((((LwU32) (integer)) <<                            \
                            DRF_SHIFT(LW_TYPES_FXP_INTEGER((x), (y)))) %      \
                                (scale)) > ((scale) >> 1)) ? 1U : 0U)))

#define LW_TYPES_UFXP_X_Y_TO_U32(x, y, fxp)                                   \
    ((LwU32) (DRF_VAL(_TYPES, _FXP, _INTEGER((x), (y)),                       \
                    ((LwUFXP##x##_##y) (fxp)))))

#define LW_TYPES_UFXP_X_Y_TO_U32_ROUNDED(x, y, fxp)                           \
    (LW_TYPES_UFXP_X_Y_TO_U32(x, y, (fxp)) +                                  \
        (FLD_TEST_DRF_NUM(_TYPES, _FXP, _FRACTIONAL_MSB((x), (y)),            \
            LW_TYPES_FXP_FRACTIONAL_MSB_ONE, ((LwUFXP##x##_##y) (fxp))) ?     \
            1U : 0U))

// 64-bit Unsigned FXP
#define LW_TYPES_U64_TO_UFXP_X_Y(x, y, integer)                               \
    ((LwUFXP##x##_##y) (((LwU64) (integer)) <<                                \
                        DRF_SHIFT64(LW_TYPES_FXP_INTEGER((x), (y)))))

#define LW_TYPES_U64_TO_UFXP_X_Y_SCALED(x, y, integer, scale)                 \
    ((LwUFXP##x##_##y) (((((LwU64) (integer)) <<                              \
                             DRF_SHIFT64(LW_TYPES_FXP_INTEGER((x), (y)))) +   \
                         ((scale) >> 1)) /                                    \
                        (scale)))

#define LW_TYPES_UFXP_X_Y_TO_U64(x, y, fxp)                                   \
    ((LwU64) (DRF_VAL64(_TYPES, _FXP, _INTEGER((x), (y)),                     \
                    ((LwUFXP##x##_##y) (fxp)))))

#define LW_TYPES_UFXP_X_Y_TO_U64_ROUNDED(x, y, fxp)                           \
    (LW_TYPES_UFXP_X_Y_TO_U64(x, y, (fxp)) +                                  \
        (FLD_TEST_DRF_NUM64(_TYPES, _FXP, _FRACTIONAL_MSB((x), (y)),          \
            LW_TYPES_FXP_FRACTIONAL_MSB_ONE, ((LwUFXP##x##_##y) (fxp))) ?     \
            1U : 0U))

//
// 32-bit Signed FXP:
// Some compilers do not support left shift negative values
// so typecast integer to LwU32 instead of LwS32
//
// Note that there is an issue with the rounding in
// LW_TYPES_S32_TO_SFXP_X_Y_SCALED. In particular, when the signs of the
// numerator and denominator don't match, the rounding is done towards positive
// infinity, rather than away from 0. This will need to be fixed in a follow-up
// change.
//
#define LW_TYPES_S32_TO_SFXP_X_Y(x, y, integer)                               \
    ((LwSFXP##x##_##y) (((LwU32) (integer)) <<                                \
                        DRF_SHIFT(LW_TYPES_FXP_INTEGER((x), (y)))))

#define LW_TYPES_S32_TO_SFXP_X_Y_SCALED(x, y, integer, scale)                 \
    ((LwSFXP##x##_##y) (((((LwS32) (integer)) <<                              \
                             DRF_SHIFT(LW_TYPES_FXP_INTEGER((x), (y)))) +     \
                         ((scale) >> 1)) /                                    \
                        (scale)))

#define LW_TYPES_SFXP_X_Y_TO_S32(x, y, fxp)                                   \
    ((LwS32) ((DRF_VAL(_TYPES, _FXP, _INTEGER((x), (y)),                      \
                    ((LwSFXP##x##_##y) (fxp)))) |                             \
              ((DRF_VAL(_TYPES, _SFXP, _INTEGER_SIGN((x), (y)), (fxp)) ==     \
                    LW_TYPES_SFXP_INTEGER_SIGN_NEGATIVE) ?                    \
                DRF_NUM(_TYPES, _SFXP, _S32_SIGN_EXTENSION((x), (y)),         \
                    LW_TYPES_SFXP_S32_SIGN_EXTENSION_NEGATIVE((x), (y))) :    \
                DRF_NUM(_TYPES, _SFXP, _S32_SIGN_EXTENSION((x), (y)),         \
                    LW_TYPES_SFXP_S32_SIGN_EXTENSION_POSITIVE((x), (y))))))

/*!
 * Note: The rounding action for signed numbers should ideally round away from
 *       0 in both the positive and the negative regions.
 *       For positive numbers, we add 1 if the fractional MSb is 1.
 *       For negative numbers, we add -1 (equivalent to subtracting 1) if the
 *       fractional MSb is 1.
 */
#define LW_TYPES_SFXP_X_Y_TO_S32_ROUNDED(x, y, fxp)                           \
    (LW_TYPES_SFXP_X_Y_TO_S32(x, y, (fxp)) +                                  \
        (FLD_TEST_DRF_NUM(_TYPES, _FXP, _FRACTIONAL_MSB((x), (y)),            \
            LW_TYPES_FXP_FRACTIONAL_MSB_ONE, ((LwSFXP##x##_##y) (fxp))) ?     \
                ((DRF_VAL(_TYPES, _SFXP, _INTEGER_SIGN((x), (y)), (fxp)) ==   \
                    LW_TYPES_SFXP_INTEGER_SIGN_POSITIVE) ? 1 : -1) : 0))

#define LW_TYPES_SFXP_X_Y_TO_FLOAT32(x, y, fxp)                               \
    ((LwF32) LW_TYPES_SFXP_X_Y_TO_S32(x, y, (fxp)) +                          \
        ((LwF32) DRF_NUM(_TYPES, _FXP, _FRACTIONAL((x), (y)),                 \
            ((LwSFXP##x##_##y) (fxp))) / (LwF32) (1 << (y))))

//
// 64-bit Signed FXP:
// Some compilers do not support left shift negative values
// so typecast integer to LwU64 instead of LwS64
//
// Note that there is an issue with the rounding in
// LW_TYPES_S64_TO_SFXP_X_Y_SCALED. In particular, when the signs of the
// numerator and denominator don't match, the rounding is done towards positive
// infinity, rather than away from 0. This will need to be fixed in a follow-up
// change.
//
#define LW_TYPES_S64_TO_SFXP_X_Y(x, y, integer)                               \
    ((LwSFXP##x##_##y) (((LwU64) (integer)) <<                                \
                        DRF_SHIFT64(LW_TYPES_FXP_INTEGER((x), (y)))))

#define LW_TYPES_S64_TO_SFXP_X_Y_SCALED(x, y, integer, scale)                 \
    ((LwSFXP##x##_##y) (((((LwS64) (integer)) <<                              \
                             DRF_SHIFT64(LW_TYPES_FXP_INTEGER((x), (y)))) +     \
                         ((scale) >> 1)) /                                    \
                        (scale)))

#define LW_TYPES_SFXP_X_Y_TO_S64(x, y, fxp)                                   \
    ((LwS64) ((DRF_VAL64(_TYPES, _FXP, _INTEGER((x), (y)),                    \
                    ((LwSFXP##x##_##y) (fxp)))) |                             \
              ((DRF_VAL64(_TYPES, _SFXP, _INTEGER_SIGN((x), (y)), (fxp)) ==   \
                    LW_TYPES_SFXP_INTEGER_SIGN_NEGATIVE) ?                    \
                DRF_NUM64(_TYPES, _SFXP, _S64_SIGN_EXTENSION((x), (y)),       \
                    LW_TYPES_SFXP_S64_SIGN_EXTENSION_NEGATIVE((x), (y))) :    \
                DRF_NUM64(_TYPES, _SFXP, _S64_SIGN_EXTENSION((x), (y)),       \
                    LW_TYPES_SFXP_S64_SIGN_EXTENSION_POSITIVE((x), (y))))))

/*!
 * Note: The rounding action for signed numbers should ideally round away from
 *       0 in both the positive and the negative regions.
 *       For positive numbers, we add 1 if the fractional MSb is 1.
 *       For negative numbers, we add -1 (equivalent to subtracting 1) if the
 *       fractional MSb is 1.
 */
#define LW_TYPES_SFXP_X_Y_TO_S64_ROUNDED(x, y, fxp)                           \
    (LW_TYPES_SFXP_X_Y_TO_S64(x, y, (fxp)) +                                  \
        (FLD_TEST_DRF_NUM64(_TYPES, _FXP, _FRACTIONAL_MSB((x), (y)),          \
            LW_TYPES_FXP_FRACTIONAL_MSB_ONE, ((LwSFXP##x##_##y) (fxp))) ?     \
                ((DRF_VAL64(_TYPES, _SFXP, _INTEGER_SIGN((x), (y)), (fxp)) == \
                    LW_TYPES_SFXP_INTEGER_SIGN_POSITIVE) ? 1 : -1) : 0))

/*!
 * Macros representing the single-precision IEEE 754 floating point format for
 * "binary32", also known as "single" and "float".
 *
 * Single precision floating point format wiki [1]
 *
 * _SIGN
 *     Single bit representing the sign of the number.
 * _EXPONENT
 *     Unsigned 8-bit number representing the exponent value by which to scale
 *     the mantissa.
 *     _BIAS - The value by which to offset the exponent to account for sign.
 * _MANTISSA
 *     Explicit 23-bit significand of the value.  When exponent != 0, this is an
 *     implicitly 24-bit number with a leading 1 prepended.  This 24-bit number
 *     can be conceptualized as FXP 9.23.
 *
 * With these definitions, the value of a floating point number can be
 * callwlated as:
 *     (-1)^(_SIGN) *
 *         2^(_EXPONENT - _EXPONENT_BIAS) *
 *         (1 + _MANTISSA / (1 << 23))
 */
// [1] : http://en.wikipedia.org/wiki/Single_precision_floating-point_format
#define LW_TYPES_SINGLE_SIGN                                               31:31
#define LW_TYPES_SINGLE_SIGN_POSITIVE                                 0x00000000
#define LW_TYPES_SINGLE_SIGN_NEGATIVE                                 0x00000001
#define LW_TYPES_SINGLE_EXPONENT                                           30:23
#define LW_TYPES_SINGLE_EXPONENT_ZERO                                 0x00000000
#define LW_TYPES_SINGLE_EXPONENT_BIAS                                 0x0000007F
#define LW_TYPES_SINGLE_MANTISSA                                            22:0


/*!
 * Helper macro to return a IEEE 754 single-precision value's mantissa as an
 * unsigned FXP 9.23 value.
 *
 * @param[in] single   IEEE 754 single-precision value to manipulate.
 *
 * @return IEEE 754 single-precision values mantissa represented as an unsigned
 *     FXP 9.23 value.
 */
#define LW_TYPES_SINGLE_MANTISSA_TO_UFXP9_23(single)                           \
    ((LwUFXP9_23)(FLD_TEST_DRF(_TYPES, _SINGLE, _EXPONENT, _ZERO, single) ?    \
                    LW_TYPES_U32_TO_UFXP_X_Y(9, 23, 0) :                       \
                    (LW_TYPES_U32_TO_UFXP_X_Y(9, 23, 1) +                      \
                        DRF_VAL(_TYPES, _SINGLE, _MANTISSA, single))))

/*!
 * Helper macro to return an IEEE 754 single-precision value's exponent,
 * including the bias.
 *
 * @param[in] single   IEEE 754 single-precision value to manipulate.
 *
 * @return Signed exponent value for IEEE 754 single-precision.
 */
#define LW_TYPES_SINGLE_EXPONENT_BIASED(single)                                \
    ((LwS32)(DRF_VAL(_TYPES, _SINGLE, _EXPONENT, single) -                     \
        LW_TYPES_SINGLE_EXPONENT_BIAS))

/*!
 * LwTemp - temperature data type introduced to avoid bugs in colwersion between
 * various existing notations.
 */
typedef LwSFXP24_8              LwTemp;

/*!
 * Macros for LwType <-> Celsius temperature colwersion.
 */
#define LW_TYPES_CELSIUS_TO_LW_TEMP(cel)                                      \
                                LW_TYPES_S32_TO_SFXP_X_Y(24,8,(cel))
#define LW_TYPES_LW_TEMP_TO_CELSIUS_TRUNCED(lwt)                              \
                                LW_TYPES_SFXP_X_Y_TO_S32(24,8,(lwt))
#define LW_TYPES_LW_TEMP_TO_CELSIUS_ROUNDED(lwt)                              \
                                LW_TYPES_SFXP_X_Y_TO_S32_ROUNDED(24,8,(lwt))
#define LW_TYPES_LW_TEMP_TO_CELSIUS_FLOAT(lwt)                                \
                                LW_TYPES_SFXP_X_Y_TO_FLOAT32(24,8,(lwt))

/*!
 * Macro for LwType -> number of bits colwersion
 */
#define LW_NBITS_IN_TYPE(type) (8 * sizeof(type))

/*!
 * Macro to colwert SFXP 11.5 to LwTemp.
 */
#define LW_TYPES_LWSFXP11_5_TO_LW_TEMP(x) ((LwTemp)(x) << 3)

/*!
 * Macro to colwert UFXP11.5 Watts to LwU32 milli-Watts.
 */
#define LW_TYPES_LWUFXP11_5_WATTS_TO_LWU32_MILLI_WATTS(x) ((((LwU32)(x)) * ((LwU32)1000)) >> 5)

#endif /* LWFIXEDTYPES_INCLUDED */
