
/*
 * Copyright (c) 2001-2005, Lwpu Corporation.  All rights reserved.
 *
 * THE INFORMATION CONTAINED HEREIN IS PROPRIETARY AND CONFIDENTIAL TO
 * LWPU, CORPORATION.  USE, REPRODUCTION OR DISCLOSURE TO ANY THIRD PARTY
 * IS SUBJECT TO WRITTEN PRE-APPROVAL BY LWPU, CORPORATION.
 */
/*! \brief
 * Contains utility functions to colwert back and forth between packed
 * floating point and IEEE 32-bit floating point representations.
 *
 * float_packed.h
 */

// USAGE EXAMPLE:
//
// #define PKFLT_MAN_BITS      10u       // 10 mantissa bits for s10e5
// #define PKFLT_EXP_BITS      5u        // 5 exponent bits for s10e5
// #define PKFLT_SGN_BIT       1u        // 1 sign bit for s10e5
// #define PKFLT_SATURATE_TO_MAXFLOAT 0  // Values greater than MAXFLOAT map to INF
//
// #define PKFLT_TYPE          MAKE_PKFLT_TYPE(10,5)
// #define PKFLT_TO_UI32       MAKE_PKFLT_TO_UI32(10,5)
// #define PKFLT_TO_F32        MAKE_PKFLT_TO_F32(10,5)
// #define UI32_TO_PKFLT       MAKE_UI32_TO_PKFLT(10,5)
// #define F32_TO_PKFLT        MAKE_F32_TO_PKFLT(10,5)
//
// #include "float_packed.h"

#if PKFLT_SGN_BIT==0
# define MAKE_PKFLT_NAME(man,exp)     lwU##man##E##exp
# define MAKE_PKFLT_TYPE(man,exp)     LWu##man##e##exp
# define MAKE_PKFLT_TO_UI32(man,exp)  lw##U##man##E##exp##toUI32
# define MAKE_PKFLT_TO_F32(man,exp)   lw##U##man##E##exp##toF32
# define MAKE_UI32_TO_PKFLT(man,exp)  lwUI32to##U##man##E##exp
# define MAKE_F32_TO_PKFLT(man,exp)   lwF32to##U##man##E##exp
#else
# define MAKE_PKFLT_NAME(man,exp)     lwS##man##E##exp
# define MAKE_PKFLT_TYPE(man,exp)     LWs##man##e##exp
# define MAKE_PKFLT_TO_UI32(man,exp)  lw##S##man##E##exp##toUI32
# define MAKE_PKFLT_TO_F32(man,exp)   lw##S##man##E##exp##toF32
# define MAKE_UI32_TO_PKFLT(man,exp)  lwUI32to##S##man##E##exp
# define MAKE_F32_TO_PKFLT(man,exp)   lwF32to##S##man##E##exp
#endif

#define FP32_MAN_BITS       23u                                      // 23 mantissa bits for s23e8
#define FP32_EXP_BITS       8u                                       // 8 exponent bits for s23e8
#define FP32_BITS           (FP32_MAN_BITS+FP32_EXP_BITS+1)          // 32 for s23e8 (includes sign bit)
#define FP32_EXP_BIAS       ((1u<<(FP32_EXP_BITS-1))-1)              // 127 for s23e8
#define FP32_MANEXP_EXP_LSB (1u<<FP32_MAN_BITS)                      // 0x00800000 for s23e8
#define FP32_MANEXP_INF     (((1u<<FP32_EXP_BITS)-1)<<FP32_MAN_BITS) // 0x7F800000 for s23e8
#define FP32_MANEXP_NAN     ((1u<<(FP32_EXP_BITS+FP32_MAN_BITS))-1)  // 0x7FFFFFFF for s23e8
#define FP32_EXP_MAX        ((1u<<FP32_EXP_BITS)-1)                  // 255
#define FP32_EXP_MASK       ((1u<<FP32_EXP_BITS)-1)                  // 255
#define FP32_SGN_MASK       (1u<<(FP32_MAN_BITS+FP32_EXP_BITS))      // MSB set for s23e8

#define PKFLT_EXP_BIAS         ((1u<<(PKFLT_EXP_BITS-1))-1)                      // 15 for s10e5
#define PKFLT_EXP_MAX          ((1u<<PKFLT_EXP_BITS)-1)                          // 31 for s10e5
#define PKFLT_BITS             (PKFLT_MAN_BITS+PKFLT_EXP_BITS+PKFLT_SGN_BIT)     // 16 for s10e5 (includes sign bit)
#define PKFLT_MANEXP_MASK      ((1u<<(PKFLT_MAN_BITS+PKFLT_EXP_BITS))-1)         // 0x7FFF for s10e5
#define PKFLT_MANEXP_EXP_LSB   (1u<<PKFLT_MAN_BITS)                              // 0x400 for s10e5
#define PKFLT_MANEXP_MAN_MASK  ((1u<<PKFLT_MAN_BITS)-1)                          // 0x3FF for s10e5
#define PKFLT_MANEXP_EXP_MASK  (((1u<<PKFLT_EXP_BITS)-1)<<PKFLT_MAN_BITS)        // 0x7C00 for s10e5
#define PKFLT_EXP_BIAS_DIFF    (FP32_EXP_BIAS-PKFLT_EXP_BIAS)                    // 112 for s10e5
#define PKFLT_SGN_MASK         (PKFLT_SGN_BIT<<(PKFLT_MAN_BITS+PKFLT_EXP_BITS))  // 0x8000 for s10e5
#define PKFLT_MANEXP_INF       (((1u<<PKFLT_EXP_BITS)-1)<<PKFLT_MAN_BITS)        // 0x7C00 for s10e5
#define PKFLT_MANEXP_NAN       ((1u<<(PKFLT_EXP_BITS+PKFLT_MAN_BITS))-1)         // 0x7FFF for s10e5
#define PKFLT_MANEXP_MAXFLOAT  ((PKFLT_MANEXP_EXP_MASK-PKFLT_MANEXP_EXP_LSB)|PKFLT_MANEXP_MAN_MASK) // 0x7Bff for s10e5

// Constants for colwerting from/to packed float and fp32
#define COLW_MANEXP_SHIFT   (FP32_MAN_BITS-PKFLT_MAN_BITS)                    // 13 for s10e5
#define COLW_MANEXP_SUBLSB  (1u<<(COLW_MANEXP_SHIFT-1))                       // 0x1000 for s10e5
#define COLW_SHIFT          (FP32_BITS-PKFLT_BITS)                            // 16 for s10e5
#define COLW_EXP_BIAS_SUM   (FP32_EXP_BIAS+PKFLT_EXP_BIAS)                    // 142 for s10e5
#define COLW_DENORM_START   ((FP32_EXP_BIAS-PKFLT_EXP_BIAS+1)<<FP32_MAN_BITS) // 0x38800000 for s10e5

// PKFLT_TO_UI32:  Colwert a packed float to an unsigned int
// holding the IEEE encoding of the equivalent floating-point value.
uint32_t PKFLT_TO_UI32(PKFLT_TYPE f)
{
    uint32_t manexp, data;

    // Extract the mantissa and exponent.
    manexp = f & PKFLT_MANEXP_MASK;

    if (manexp < PKFLT_MANEXP_EXP_LSB) {
        // Exponent == 0, implies 0.0 or Denorm.
        if (manexp == 0) {
            data = 0;
        } else {
            // Denorm -- shift the mantissa left until we find a leading one.
            // Each shift drops one off the final exponent.
            data = COLW_DENORM_START;
            do {
                data -= FP32_MANEXP_EXP_LSB; // multiply by 1/2
                manexp *= 2;
            } while (!(manexp & PKFLT_MANEXP_EXP_LSB));

            // Now shift the mantissa into the final location.
            data |= (manexp & PKFLT_MANEXP_MAN_MASK) << COLW_MANEXP_SHIFT;
        }
    } else if (manexp >= PKFLT_MANEXP_EXP_MASK) {
        // Exponent = EXP_MAX, implies INF or NaN.
        if (manexp == PKFLT_MANEXP_EXP_MASK) {
            data = FP32_MANEXP_INF;  // INF
        } else {
            data = FP32_MANEXP_NAN;  // NaN
        }
    } else {
        // Normal float -- (1) shift over mantissa/exponent, (2) add bias to
        // exponent, and (3)
        data = (manexp << COLW_MANEXP_SHIFT);
        data += (PKFLT_EXP_BIAS_DIFF << FP32_MAN_BITS);
    }

    // Or in the sign bit and return the result.
    return data | (f & PKFLT_SGN_MASK) << COLW_SHIFT;
}


// PKFLT_TO_F32:  Colwert a pack float to a single-precision IEEE float.
float PKFLT_TO_F32(PKFLT_TYPE f)
{
    uint32_t ui = PKFLT_TO_UI32(f);
    return uint32_as_float(ui);
}


// UI32_TO_PKFLT:  Colwert an unsigned int holding the IEEE encoding of a
// floating-point value to a packed float.
PKFLT_TYPE UI32_TO_PKFLT(uint32_t ui)
{
    uint32_t data, exp, man;

    // Extract the exponent and the MAN_BITS MSBs of the mantissa from the fp32
    // number.
    exp = (ui >> FP32_MAN_BITS) & FP32_EXP_MASK;
    man = (ui >> COLW_MANEXP_SHIFT) & PKFLT_MANEXP_MAN_MASK;

    // Round on type colwersion.  Check mantissa bit FP32_MANEXP_SHIFT in the 32-bit number.
    // If set, round the mantissa up.  If the mantissa overflows, bump the
    // exponent by 1 and clear the mantissa.
    if (ui & COLW_MANEXP_SUBLSB) {
        man++;
        if (man & PKFLT_MANEXP_EXP_LSB) {
            man = 0;
            exp++;
        }
    }

    if (exp <= PKFLT_EXP_BIAS_DIFF) {
        // |x| < 2^-14, implies 0.0 or Denorm

        // If |x| < 2^-25, we will flush to zero.  Otherwise, we will or in
        // the leading one and shift to the appropriate location.
        if (exp < (PKFLT_EXP_BIAS_DIFF - PKFLT_MAN_BITS)) {
            data = 0;           // 0.0
        } else {
            data = (man | PKFLT_MANEXP_EXP_LSB) >> (PKFLT_EXP_BIAS_DIFF+1 - exp);
        }
    } else if (exp > COLW_EXP_BIAS_SUM) {
        // |x| > 2^15, implies overflow, an existing INF, or NaN.  Fp32 NaN is any
        // non-zero mantissa with an exponent of +128 (255).  Note that our
        // rounding algorithm could have kicked the exponent up to 256.
        if (exp == FP32_EXP_MAX) {
            if (man) {
                data = PKFLT_MANEXP_NAN;
                // Return allows -NaN to return as NaN even if there is no sign bit.
                return data | ((ui >> COLW_SHIFT) & PKFLT_SGN_MASK);
            } else {
                data = PKFLT_MANEXP_INF;
            }
        } else {
#if PKFLT_SATURATE_TO_MAXFLOAT
            data = PKFLT_MANEXP_MAXFLOAT;
#else
            // Values over MAXFLOAT map to INF
            data = PKFLT_MANEXP_INF;
#endif
        }
    } else {
        exp -= PKFLT_EXP_BIAS_DIFF;
        data = (exp << PKFLT_MAN_BITS) | man;
    }

    if (PKFLT_SGN_BIT) {
        data |= ((ui >> COLW_SHIFT) & PKFLT_SGN_MASK);
    } else {
        if (ui & FP32_SGN_MASK) {
            // Clamp negative values (except -NaN, see above) to zero.
            data = 0;           // 0.0
        }
    }

    return data;
}


// F32_TO_PKFLT:  Colwert an IEEE single-precision floating-point value to packed float.
PKFLT_TYPE F32_TO_PKFLT(float f)
{
    return UI32_TO_PKFLT(float_as_uint32(f));
}

#ifndef __GL_NO_UNDEF_PKFLT_DEFINES

// Undefine everything
#undef MAKE_PKFLT_NAME
#undef MAKE_PKFLT_TYPE
#undef MAKE_PKFLT_TO_UI32
#undef MAKE_PKFLT_TO_F32
#undef MAKE_UI32_TO_PKFLT
#undef MAKE_F32_TO_PKFLT

#undef FP32_MAN_BITS
#undef FP32_EXP_BITS
#undef FP32_BITS
#undef FP32_EXP_BIAS
#undef FP32_MANEXP_EXP_LSB
#undef FP32_MANEXP_INF
#undef FP32_MANEXP_NAN
#undef FP32_EXP_MAX
#undef FP32_EXP_MASK
#undef FP32_SGN_MASK

#undef PKFLT_EXP_BIAS
#undef PKFLT_EXP_MAX
#undef PKFLT_BITS
#undef PKFLT_MANEXP_MASK
#undef PKFLT_MANEXP_EXP_LSB
#undef PKFLT_MANEXP_MAN_MASK
#undef PKFLT_MANEXP_EXP_MASK
#undef PKFLT_EXP_BIAS_DIFF
#undef PKFLT_SGN_MASK
#undef PKFLT_MANEXP_INF
#undef PKFLT_MANEXP_NAN

#undef COLW_MANEXP_SHIFT
#undef COLW_MANEXP_SUBLSB
#undef COLW_SHIFT
#undef COLW_EXP_BIAS_SUM
#undef COLW_DENORM_START

#undef PKFLT_MAN_BITS
#undef PKFLT_EXP_BITS
#undef PKFLT_SGN_BIT
#undef PKFLT_SATURATE_TO_MAXFLOAT

#undef PKFLT_TYPE
#undef PKFLT_TO_UI32
#undef PKFLT_TO_F32
#undef UI32_TO_PKFLT
#undef F32_TO_PKFLT

#endif // __GL_NO_UNDEF_PKFLT_DEFINES
