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
 *  Module name              : stdFloat.c
 *
 *  Description              :
 *
 *      IEEE 754 floating point representation functions.
 */

/*--------------------------------- Includes ---------------------------------*/

#include "stdFloat.h"
#include "stdLocal.h"


/*----------------------- Floating Point Representation ----------------------*/

// internalFloat2half is taken from //sw/gpgpu/lwca/tools/lwdart/lwda_fp16.hpp
typedef unsigned short  __half;

/*
 * functions ftu(), dtull(), ulltd() and float2double() are from Lwca-math team.
 */
static unsigned STD_CDECL ftu(float x)
{
    return *((unsigned *)&x);
}

static unsigned long long STD_CDECL dtull(double x)
{
    return *((unsigned long long *)&x);
}

static double STD_CDECL ulltd(unsigned long long x)
{
    return *((double *)&x);
}

static double STD_CDECL float2double(float x, int ftz)
{
    unsigned ux, aux;
    unsigned long long ullr;
    double r;
    ux = ftu(x);
    aux = ux & 0x7FFFFFFF;

    if (aux >= 0x7f800000)
    {   // Inf / NaN
        if (aux > 0x7f800000)
        {
            // NaN

            // grab sign
            unsigned uSign = ux & 0x80000000;
            // quietize NaNs
            unsigned uMant = (ux & 0x003fffff) | 0x00400000;
            ullr  = (unsigned long long)uSign << 32;
            ullr |= 0x7ff0000000000000ull;
            ullr |= (unsigned long long)uMant << (32 - 3);
        }
        else
        {
            // Inf
            ullr = ((unsigned long long)(ux | 0x00700000)) << 32;
        }
    }
    else if (aux <= 0x007fffff)
    {
        // Zero and Denormals

        unsigned uSign = ux & 0x80000000;
        if (!ftz)
        {
            ullr  = (unsigned long long)aux << (32 - 3);
            ullr |= (unsigned long long)(-126 + 1023) << 52;
            r  = ulltd(ullr);
            // normalize using double precision arithmetic
            r -= ulltd((unsigned long long)(-126 + 1023) << 52);
        }
        else
        {   //FTZ
            r = 0.0;
        }
        // restore sign
        ullr = dtull(r) | ((unsigned long long)uSign << 32);
    }
    else
    {
        unsigned uSign = ux & 0x80000000;
        unsigned uExp  = ux & 0x7f800000;
        unsigned uMant = ux & 0x007fffff;
        uExp = (uExp >> 23) - 127 + 1023;
        ullr  = (unsigned long long)uSign << 32;
        ullr |= (unsigned long long)uExp  << 52;
        ullr |= (unsigned long long)uMant << (32 - 3);
    }

    r = ulltd(ullr);
    return r;
}

// internalHalf2float is taken from //sw/gpgpu/lwca/tools/lwdart/lwda_fp16.hpp
// Corresponding function in lwda_fp16.hpp is __internal_half2float()
static float STD_CDECL internalHalf2float(const unsigned short h)
{
    unsigned int sign = (((unsigned int)(h) >> 15U) & 1U);
    unsigned int exponent = (((unsigned int)(h) >> 10U) & 0x1fU);
    unsigned int mantissa = (((unsigned int)(h) & 0x3ffU) << 13U);
    float f;
    if (exponent == 0x1fU) { /* NaN or Inf */
        // if NaN   then    QNaN     else Inf
        mantissa = ((mantissa != 0U) ? (mantissa | 0x00400000) : 0U);
        exponent = 0xffU;
    } else if (exponent == 0U) { /* Denorm or Zero */
        if (mantissa != 0U) {
            unsigned int msb;
            exponent = 0x71U;
            do {
                msb = (mantissa & 0x400000U);
                mantissa <<= 1U; /* normalize */
                --exponent;
            } while (msb == 0U);
            mantissa &= 0x7fffffU; /* 1.mantissa is implicit */
        }
    } else {
        exponent += 0x70U;
    }
    unsigned int u = ((sign << 31U) | (exponent << 23U) | mantissa);
    (void)memcpy(&f, &u, sizeof(u));
    return f;
}

// half2float is taken from //sw/gpgpu/lwca/tools/lwdart/lwda_fp16.hpp
// Corresponding function in lwda_fp16.hpp is __half2float()
float STD_CDECL half2float(const __half a)
{
    float val;
    val = internalHalf2float((__half)a);
    return val;
}

// This routine is pruned version of stdCanonicalizeFloatValue_S
// from module_compiler/drivers/compiler/utilities/std/stdFloat.c#4
static uInt64 e6m9_to_double( uInt64 value )
{
    uInt64 sign, mantissa, expR, exp;
    uInt   mantissaBits, expBits, bias;
    
    expBits      = 6;
    mantissaBits = 9;
    
    bias      = (1ULL << (expBits-1)) - 1;
    sign      = (value >> (mantissaBits + expBits)) << 63;
    expR      = (value >> mantissaBits) & stdBITMASK64(0, expBits);
    exp       = expR - bias + 1023;
    mantissa  = (value & stdBITMASK64(0, mantissaBits)) << (52 - mantissaBits);
    
    if (expR == ((1ULL << expBits) - 1)) {
        /* Special values */
        exp = 0x7ff;
    } else if (expR == 0) {
        if (mantissa == 0) {
            /* +/- 0.0 */
            exp = 0;
        } else {
            /* Subnormal, which gets normalized */
            exp++;
            while ((mantissa & (1ULL<<52)) == 0) {
                mantissa <<= 1;
                exp--;
            }
            mantissa &= ~(1ULL<<52);
        }
    }
    value = sign | (exp << 52) | mantissa;
    
    return value;
}

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
uInt64 STD_CDECL stdCanonicalizeFloatValue_S( uInt64 value, stdImmType immType, uInt length )
{
    uInt   fmtLength;

    switch (immType) {
    case stdIntImmType : {
                             stdDoubleColw colw;

                             colw.d = (Int64)value;
                             return colw.i;
                         }

    case stdF16ImmType : {
                            __half halfVal;
                            stdDoubleColw dcolw;
                            float floatVal;
                            fmtLength    = 16;
                            halfVal = (length < fmtLength) ? (value << (fmtLength - length)) : value;
                            floatVal = half2float(halfVal);
                            dcolw.d = float2double(floatVal, 0);
                            value = dcolw.i;
                            break;
                         }

    case stdE6M9ImmType: {  
                            fmtLength    = 16;
                            stdASSERT((length == fmtLength), ("Incorrect E6M9 immediate size"));
                            value = e6m9_to_double(value);
                            break;
                         }
    // stdE8M7ImmType is taken from //sw/gpgpu/lwca/tools/lwdart/lwda_bf16.hpp
    // Corresponding function is __internal_bfloat162float
    case stdE8M7ImmType: {
                            fmtLength = 16;
                            stdASSERT((length == fmtLength), ("Incorrect E8M7 immediate size"));
                            stdDoubleColw dcolw;
                            stdFloatColw  fcolw;
                            value = value << fmtLength;
                            fcolw.i = value;
                            dcolw.d = float2double(fcolw.f, 0);
                            value = dcolw.i;
                            break;
                         }
    case stdE8M10ImmType:{
                            stdASSERT((value == (uInt64)(value & 0xffffe000)), ("Incorrect tf32 number"));
                            stdDoubleColw dcolw;
                            stdFloatColw  fcolw;
                            fcolw.i = value;
                            dcolw.d = float2double(fcolw.f, 0);
                            value = dcolw.i;
                            break;
                         }
    case stdF32ImmType:{
                            stdDoubleColw dcolw;
                            stdFloatColw  fcolw;
                            fmtLength    = 32;
                            value = (length < fmtLength) ? (value << (fmtLength - length)) : value;
                            fcolw.i = value;
                            dcolw.d = float2double(fcolw.f, 0);
                            value = dcolw.i;
                            break;
                       }
    case stdF64ImmType : fmtLength    = 64;
                         value = (length < fmtLength) ? (value << (fmtLength - length)) : value;
                         break;

    default            : stdASSERT( False, ("Case label out of range") );
                         return 0;
    }

    return value;
}


    static uInt64 reducePrecision( uInt64 value, uInt inputBits, uInt outputBits, Bool round )
    {
        Bool roundUp = round && ((value >> (inputBits - outputBits - 1)) & 0x1);

        value >>= (inputBits - outputBits);
        if (roundUp)
            value += 1;
        return value;
    }

// Corresponding function in lwda_fp16.hpp is __internal_float2half()
static unsigned short STD_CDECL internalFloat2half(const float f, unsigned int *sign, unsigned int *remainder)
{
    unsigned int x;
    unsigned int u;
    unsigned int result = 0U;
    (void)memcpy(&x, &f, sizeof(f));
    u = (x & 0x7fffffffU);
    *sign = ((x >> 16U) & 0x8000U);
    // NaN/+Inf/-Inf
    if (u >= 0x7f800000U) {
        *remainder = 0U;
        // if NaN then QNaN and preserve 10 bits of payload; else Inf
        result = *sign | 0x7c00U | ((u != 0x7f800000U) ? (0x0200U | ((u & 0x007FE000U) >> 13)) : 0U);
    } else if (u > 0x477fefffU) { // Overflows
        *remainder = 0x80000000U;
        result = (*sign | 0x7bffU);
    } else if (u >= 0x38800000U) { // Normal numbers
        *remainder = u << 19U;
        u -= 0x38000000U;
        result = (*sign | (u >> 13U));
    } else if (u < 0x33000001U) { // +0/-0
        *remainder = u;
        result = *sign;
    } else { // Denormal numbers
        const unsigned int exponent = u >> 23U;
        const unsigned int shift = 0x7eU - exponent;
        unsigned int mantissa = (u & 0x7fffffU);
        mantissa |= 0x800000U;
        *remainder = mantissa << (32U - shift);
        result = (*sign | (mantissa >> shift));
    }
    return (unsigned short)result;
}

// float2half is taken from //sw/gpgpu/lwca/tools/lwdart/lwda_fp16.hpp
// Corresponding function in lwda_fp16.hpp is __float2half()
static __half STD_CDECL float2half(const float a)
{
    __half val;
    __half r;
    unsigned int sign;
    unsigned int remainder;
    r = internalFloat2half(a, &sign, &remainder);
    if ((remainder > 0x80000000U) || ((remainder == 0x80000000U) && ((r & 0x1U) != 0U))) {
        r++;
    }
    val = r;
    return val;
}

// double2half is taken from //sw/gpgpu/lwca/tools/lwdart/lwda_fp16.hpp
// Corresponding function in lwda_fp16.hpp is __double2half()
static __half STD_CDECL double2half(const double x)
{
    // Perform rounding to 11 bits of precision, colwert value
    // to float and call existing float to half colwersion.
    // By pre-rounding to 11 bits we avoid additional rounding
    // in float to half colwersion.
    unsigned long long int absx;
    unsigned long long int ux;
    (void)memcpy(&ux, &x, sizeof(x));
    absx = (ux & 0x7fffffffffffffffull);
    if ((absx >= 0x40f0000000000000ull) || (absx <= 0x3e60000000000000ull))
    {
        // |x| >= 2^16 or NaN or |x| <= 2^(-25)
        // double-rounding is not a problem
        return float2half((float)x);
    }

    // here 2^(-25) < |x| < 2^16
    // prepare shifter value such that x + shifter
    // done in double precision performs round-to-nearest-even
    // and (x + shifter) - shifter results in x rounded to
    // 11 bits of precision. Shifter needs to have exponent of
    // x plus 53 - 11 = 42 and a leading bit in mantissa to guard
    // against negative values.
    // So need to have |x| capped to avoid overflow in exponent.
    // For inputs that are smaller than half precision minnorm
    // we prepare fixed shifter exponent.
    unsigned long long shifterBits = ux & 0x7ff0000000000000ull;
    if (absx >= 0x3f10000000000000ull)
    {   // |x| >= 2^(-14)
        // add 42 to exponent bits
        shifterBits += 42ull << 52;
    }
    else
    {   // 2^(-25) < |x| < 2^(-14), potentially results in denormal
        // set exponent bits to 42 - 14 + bias
        shifterBits = ((42ull - 14 + 1023) << 52);
    }
    // set leading mantissa bit to protect against negative inputs
    shifterBits |= 1ull << 51;
    double shifter;
    (void)memcpy(&shifter, &shifterBits, sizeof(shifterBits));
    double xShiftRound = x + shifter;

    // Prevent the compiler from optimizing away x + shifter - shifter
    // by doing intermediate memcopy and harmless bitwize operation
    unsigned long long int xShiftRoundBits;
    (void)memcpy(&xShiftRoundBits, &xShiftRound, sizeof(xShiftRound));

    // the value is positive, so this operation doesn't change anything
    xShiftRoundBits &= 0x7fffffffffffffffull;

    (void)memcpy(&xShiftRound, &xShiftRoundBits, sizeof(xShiftRound));

    double xRounded = xShiftRound - shifter;
    float  xRndFlt = (float)xRounded;
    __half res =  float2half(xRndFlt);
    return res;
}

// double2float is taken from //sw/gpgpu/lwca/apps/common/njcommon.c as it is
// Corresponding function in njcommon.c is refDbl2Flt()
static float STD_CDECL double2float (double a, int roundMode, int ftz)
{
    volatile union {
        double d;
        unsigned long long int i;
    } xx;
    volatile union {
        float f;
        unsigned int i;
    } res;

    unsigned long long sticky;
    unsigned int lsb;
    unsigned int rnd;
    int shift;
    xx.d = a;
    res.i = (((unsigned int) (xx.i >> 32)) & 0x80000000);
    if (a == 0.0) {
        /* Zero */
        return res.f;
    }
    if ((xx.i & 0x7ff0000000000000ULL) == 0x7ff0000000000000ULL) {
        if ((xx.i & 0x7fffffffffffffffULL) > 0x7ff0000000000000ULL) {
            /* Nan */
            res.i = ((unsigned int)((xx.i >> 32) & 0x80000000) |
                     (255U << 23) | 0x00400000 |
                     (unsigned int)((xx.i >> (53 - 24)) & 0x007fffff));
        } else {
            /* Inf */
            res.i |= 0x7f800000;
        }
        return res.f;
    }
    shift = ((int) ((xx.i >> 52) & 0x7ff)) - 1023;
    /* Overflow */
    xx.i = (xx.i & 0x000fffffffffffffULL);
    if (shift >= 128) {
        if ((roundMode == REF_ROUND_ZERO) ||
            ((roundMode == REF_ROUND_DOWN) && !res.i) ||
            ((roundMode == REF_ROUND_UP) &&  res.i)) {
            res.i |= 0x7f7fffff;
        } else {
            res.i |= 0x7f800000;
        }
        return res.f;
    }
    if (shift <= -127) {
        if (!ftz) {
            /* Underflow */
            xx.i |= 0x0010000000000000ULL;
            if (shift < -180) {
                lsb = 0;
                rnd = 0;
                sticky = xx.i;
                xx.i = 0;
            } else {
                sticky = xx.i << (64 - (-126 - shift));
                xx.i >>= (-126 - shift);
                lsb = (unsigned int)((xx.i >> 29) & 1);
                rnd = (unsigned int)((xx.i >> 28) & 1);
                sticky |= xx.i << (64 - 28);
            }
            if (((roundMode == REF_ROUND_UP) && !res.i) ||
                ((roundMode == REF_ROUND_DOWN) &&  res.i)) {
                sticky |= rnd;
                if (sticky) {
                    res.i += 1;
                }
            } else if (roundMode == REF_ROUND_NEAR) {
                if (rnd && (lsb || sticky)) {
                    res.i += 1;
                }
            }
            res.i += ((unsigned int) (xx.i >> 29)) & 0x007fffff;
            return res.f;
        } else {
            /* FTZ mode */
            sticky = xx.i << (64 - 29);
            if ((((roundMode == REF_ROUND_UP) && !res.i) ||
                 ((roundMode == REF_ROUND_DOWN) &&  res.i)) &&
                sticky) {
                res.i += 1;
            } else if (roundMode == REF_ROUND_NEAR) {
                lsb = (unsigned int)((xx.i >> 29) & 1);
                rnd = (unsigned int)((xx.i >> 28) & 1);
                sticky = xx.i << (64 - 28);
                if (rnd && (lsb || sticky)) {
                    res.i += 1;
                }
            }
            res.i += ((unsigned int) (xx.i >> 29)) & 0x007fffff;
            if (res.i & 0x00800000) {
                shift++;
            }
            if (shift <= -127) {
                res.i &= 0x80000000;
            }
            return res.f;
        }
    }
    /* Normals */
    sticky = xx.i << (64 - 29);
    if ((((roundMode == REF_ROUND_UP) && !res.i) ||
         ((roundMode == REF_ROUND_DOWN) &&  res.i)) &&
        sticky) {
        res.i += 1;
    } else if (roundMode == REF_ROUND_NEAR) {
        lsb = (unsigned int)((xx.i >> 29) & 1);
        rnd = (unsigned int)((xx.i >> 28) & 1);
        sticky = xx.i << (64 - 28);
        if (rnd && (lsb || sticky)) {
            res.i += 1;
        }
    }
    res.i += ((unsigned int) (xx.i >> 29)) & 0x007fffff;
    res.i += (unsigned int) (127 + shift) << 23;
    return res.f;
}

// This routine is pruned version of stdDecanonicalizeFloatValue_S
// from module_compiler/drivers/compiler/utilities/std/stdFloat.c#4
static uInt64 double2e6m9( uInt64 value )
{
    uInt   mantissaBits, expBits, bias, maxExp;

    uInt64 sign      = value >> 63;
    uInt64 mantissa  = value & stdBITMASK64(0,52);
    uInt64 expR      = (value >> 52) & 0x7ff;
     Int64 exp       = expR - 1023;

    Bool   round     = True;

    expBits      = 6;
    mantissaBits = 9;
    
    bias   = (1ULL << (expBits-1)) - 1;
    maxExp = (1ULL <<  expBits   ) - 1;
    
    if (expR == 0x7ff) {
        /* Special values */
        exp   = maxExp;
        round = False;
    } else if (expR == 0) {
        /* Underflow or +/- 0.0 */
        exp      = 0;
        mantissa = 0;
    } else {
        exp += bias;
        if (exp >= maxExp) {
            /* Overflow */
            exp      = maxExp;
            mantissa = 0;
            round    = False;
        } else if (exp <= 0) {
            if (exp >= -(Int)mantissaBits) {
                /* Subnormal */
                mantissa |= 1ULL << 52;
                mantissa >>= (-exp+1);
                exp = 0;
            } else {
                /* Underflow */
                exp      = 0;
                mantissa = 0;
            }
        }
    }
    /* include exponent in precision reduction in case mantissa overflows */
    value = (sign << (expBits+mantissaBits)) | reducePrecision((exp << 52) | mantissa, 52 + expBits, mantissaBits + expBits, round);

    return value;
}

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
Bool STD_CDECL stdDecanonicalizeFloatValue_S( uInt64 value, stdImmType immType, uInt length, uInt64 *result )
{
    uInt   fmtLength;

    Bool   round     = False;
    stdFloatColw fp32res;
    stdDoubleColw fp64Colw;
    
    fp64Colw.i = value;

    switch (immType) {
    case stdF16ImmType : fmtLength    = 16;
                         value        = double2half (fp64Colw.d);
                         break;

    case stdE6M9ImmType: fmtLength    = 16;
                         stdASSERT((length == fmtLength), ("Incorrect E6M9 immediate size"));
                         value = double2e6m9(value);
                         break;
                        
    case stdE8M7ImmType:
    case stdE8M10ImmType:
    case stdF32ImmType : fmtLength    = 32;
                         fp32res.f    = double2float (fp64Colw.d, REF_ROUND_NEAR, 0);  
                         value        = fp32res.i;
                         break;

    case stdF64ImmType : fmtLength    = 64;
                         break;

    default            : stdASSERT( False, ("Case label out of range") );
                         return False;
    }

    //TODO: round to nearest even in case of truncation.
    if (length < fmtLength) {
        value = reducePrecision(value, fmtLength, length, round);
    }
    
   *result = value;
    return True;
}


/*
 * Function         : Obtain textual representation of special float value
 * Parameters       : value           (I) IEEE 754 64 bit float representation
 * Function Result  : Textual representation if value is special, Nil otherwise
 */
String STD_CDECL stdGetSpecialFloatValueRepresentation( uInt64 value )
{
    uInt64 sign     = value >> 63;
    uInt64 mantissa = value & stdBITMASK64(0,52);
    uInt64 expR     = (value >> 52) & 0x7ff;

    if (expR == 0x7ff) { 
        if (sign) {
            if (!mantissa       ) { return "-INF";  } else
            if ( mantissa >> 51 ) { return "-QNAN"; } else
                                  { return "-SNAN"; }
        } else {
            if (!mantissa       ) { return "+INF";  } else
            if ( mantissa >> 51 ) { return "+QNAN"; } else
                                  { return "+SNAN"; }
        }
    }

    if (!mantissa && !expR && sign) {
        /* 
         * Treating -0.0 as a special case to avoid disturbing the general double number formatting.
         * With general double formatting, -0.0 will be printed as -0 due to the C shortest representation 
         * formatting specification (%g). This is need to stop lwasm from colwerting -0 to 0.0 when used 
         * in the context of float numbers. (Bug 200524080)
         * The issue is as follows: 
         * When lwdisasm output is passed to lwasm, the lwasm frontend will treat -0 as integer value 
         * and will drop the sign before colwerting it to float. The fix is to dump -0.0 as "-0.0" so that
         * lwasm treat it as a double number rather than integer.
         */
        return "-0.0";
    }
    
    return Nil;
}


/*
 * Function         : Colwert integer representation of IEEE 754 64 bit float
 *                    into integer representation of 16, 32 or 64 bit float 
 * Parameters       : value           (I) float representation
 *                    immType         (I) Requested representation type of result
 *                    result          (O) Result location
 * Function Result  : True iff. colwersion succeeded
 */
Bool STD_CDECL stdDecanonicalizeFloatValue( uInt64 value, stdImmType immType, uInt64 *result )
{
    switch (immType) {
    case stdF16ImmType   : return stdDecanonicalizeFloatValue_S(value,immType,16,result);
    case stdF32ImmType   : return stdDecanonicalizeFloatValue_S(value,immType,32,result);
    case stdF64ImmType   : return stdDecanonicalizeFloatValue_S(value,immType,64,result);
    case stdE6M9ImmType  : return stdDecanonicalizeFloatValue_S(value,immType,16,result);
    case stdE8M7ImmType  : return stdDecanonicalizeFloatValue_S(value,immType,16,result);
    case stdE8M10ImmType : return stdDecanonicalizeFloatValue_S(value,immType,19,result);
    case stdIntImmType   : return stdDecanonicalizeFloatValue_S(value,immType,64,result);
    
    default              : stdASSERT( False, ("Case label out of range") );
                        return 0;
    }
}


/*
 * Function         : Colwert integer representation of IEEE 754 16, 32 or 64 bit float 
 *                    into integer representation of 64 bit float. 
 * Parameters       : value           (I) float representation
 *                    immType         (I) Representation type of 'value'
 * Function Result  : 64 bit float representation of 'value'
 */
uInt64 STD_CDECL stdCanonicalizeFloatValue( uInt64 value, stdImmType immType )
{
    switch (immType) {
    case stdF16ImmType   : return stdCanonicalizeFloatValue_S(value,immType,16);
    case stdF32ImmType   : return stdCanonicalizeFloatValue_S(value,immType,32);
    case stdF64ImmType   : return stdCanonicalizeFloatValue_S(value,immType,64);
    case stdE6M9ImmType  : return stdCanonicalizeFloatValue_S(value,immType,16);
    case stdE8M7ImmType  : return stdCanonicalizeFloatValue_S(value,immType,16);
    case stdE8M10ImmType : return stdCanonicalizeFloatValue_S(value,immType,19);
    case stdIntImmType   : return stdCanonicalizeFloatValue_S(value,immType,64);
                        
    default              : stdASSERT( False, ("Case label out of range") );
                        return 0;
    }
}

