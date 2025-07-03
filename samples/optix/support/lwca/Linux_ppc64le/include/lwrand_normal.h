
 /* Copyright 2010-2014 LWPU Corporation.  All rights reserved.
  *
  * NOTICE TO LICENSEE:
  *
  * The source code and/or documentation ("Licensed Deliverables") are
  * subject to LWPU intellectual property rights under U.S. and
  * international Copyright laws.
  *
  * The Licensed Deliverables contained herein are PROPRIETARY and
  * CONFIDENTIAL to LWPU and are being provided under the terms and
  * conditions of a form of LWPU software license agreement by and
  * between LWPU and Licensee ("License Agreement") or electronically
  * accepted by Licensee.  Notwithstanding any terms or conditions to
  * the contrary in the License Agreement, reproduction or disclosure
  * of the Licensed Deliverables to any third party without the express
  * written consent of LWPU is prohibited.
  *
  * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
  * LICENSE AGREEMENT, LWPU MAKES NO REPRESENTATION ABOUT THE
  * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  THEY ARE
  * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
  * LWPU DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
  * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
  * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
  * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
  * LICENSE AGREEMENT, IN NO EVENT SHALL LWPU BE LIABLE FOR ANY
  * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
  * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
  * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
  * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
  * OF THESE LICENSED DELIVERABLES.
  *
  * U.S. Government End Users.  These Licensed Deliverables are a
  * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
  * 1995), consisting of "commercial computer software" and "commercial
  * computer software documentation" as such terms are used in 48
  * C.F.R. 12.212 (SEPT 1995) and are provided to the U.S. Government
  * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
  * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
  * U.S. Government End Users acquire the Licensed Deliverables with
  * only those rights set forth herein.
  *
  * Any use of the Licensed Deliverables in individual and commercial
  * software must include, in the user documentation and internal
  * comments to the code, the above Disclaimer and U.S. Government End
  * Users Notice.
  */


#if !defined(LWRAND_NORMAL_H_)
#define LWRAND_NORMAL_H_

/**
 * \defgroup DEVICE Device API
 *
 * @{
 */

#ifndef __LWDACC_RTC__
#include <math.h>
#endif // __LWDACC_RTC__

#include "lwrand_mrg32k3a.h"
#include "lwrand_mtgp32_kernel.h"
#include "lwrand_philox4x32_x.h"
#include "lwrand_normal_static.h"

QUALIFIERS float2 _lwrand_box_muller(unsigned int x, unsigned int y)
{
    float2 result;
    float u = x * LWRAND_2POW32_ILW + (LWRAND_2POW32_ILW/2);
    float v = y * LWRAND_2POW32_ILW_2PI + (LWRAND_2POW32_ILW_2PI/2);
#if __LWDA_ARCH__ > 0
    float s = sqrtf(-2.0f * logf(u));
    __sincosf(v, &result.x, &result.y);
#else
    float s = sqrtf(-2.0f * logf(u));
    result.x = sinf(v);
    result.y = cosf(v);
#endif
    result.x *= s;
    result.y *= s;
    return result;
}

QUALIFIERS float2 lwrand_box_muller_mrg(lwrandStateMRG32k3a_t * state)
{
    float x, y;
    x = lwrand_uniform(state);
    y = lwrand_uniform(state) * LWRAND_2PI;
    float2 result;
#if __LWDA_ARCH__ > 0
    float s = sqrtf(-2.0f * logf(x));
    __sincosf(y, &result.x, &result.y);
#else
    float s = sqrtf(-2.0f * logf(x));
    result.x = sinf(y);
    result.y = cosf(y);
#endif
    result.x *= s;
    result.y *= s;
    return result;
}

QUALIFIERS double2
_lwrand_box_muller_double(unsigned int x0, unsigned int x1,
                          unsigned int y0, unsigned int y1)
{
    double2 result;
    unsigned long long zx = (unsigned long long)x0 ^
        ((unsigned long long)x1 << (53 - 32));
    double u = zx * LWRAND_2POW53_ILW_DOUBLE + (LWRAND_2POW53_ILW_DOUBLE/2.0);
    unsigned long long zy = (unsigned long long)y0 ^
        ((unsigned long long)y1 << (53 - 32));
    double v = zy * (LWRAND_2POW53_ILW_DOUBLE*2.0) + LWRAND_2POW53_ILW_DOUBLE;
    double s = sqrt(-2.0 * log(u));

#if __LWDA_ARCH__ > 0
    sincospi(v, &result.x, &result.y);
#else
    result.x = sin(v*LWRAND_PI_DOUBLE);
    result.y = cos(v*LWRAND_PI_DOUBLE);
#endif
    result.x *= s;
    result.y *= s;

    return result;
}

QUALIFIERS double2
lwrand_box_muller_mrg_double(lwrandStateMRG32k3a_t * state)
{
    double x, y;
    double2 result;
    x = lwrand_uniform_double(state);
    y = lwrand_uniform_double(state) * 2.0;

    double s = sqrt(-2.0 * log(x));
#if __LWDA_ARCH__ > 0
    sincospi(y, &result.x, &result.y);
#else
    result.x = sin(y*LWRAND_PI_DOUBLE);
    result.y = cos(y*LWRAND_PI_DOUBLE);
#endif
    result.x *= s;
    result.y *= s;
    return result;
}

template <typename R>
QUALIFIERS float2 lwrand_box_muller(R *state)
{
    float2 result;
    unsigned int x = lwrand(state);
    unsigned int y = lwrand(state);
    result = _lwrand_box_muller(x, y);
    return result;
}

template <typename R>
QUALIFIERS float4 lwrand_box_muller4(R *state)
{
    float4 result;
    float2 _result;
    uint4 x = lwrand4(state);
    //unsigned int y = lwrand(state);
    _result = _lwrand_box_muller(x.x, x.y);
    result.x = _result.x;
    result.y = _result.y;
    _result = _lwrand_box_muller(x.z, x.w);
    result.z = _result.x;
    result.w = _result.y;
    return result;
}

template <typename R>
QUALIFIERS double2 lwrand_box_muller_double(R *state)
{
    double2 result;
    unsigned int x0 = lwrand(state);
    unsigned int x1 = lwrand(state);
    unsigned int y0 = lwrand(state);
    unsigned int y1 = lwrand(state);
    result = _lwrand_box_muller_double(x0, x1, y0, y1);
    return result;
}

template <typename R>
QUALIFIERS double2 lwrand_box_muller2_double(R *state)
{
    double2 result;
    uint4 _x;
    _x = lwrand4(state);
    result = _lwrand_box_muller_double(_x.x, _x.y, _x.z, _x.w);
    return result;
}


template <typename R>
QUALIFIERS double4 lwrand_box_muller4_double(R *state)
{
    double4 result;
    double2 _res1;
    double2 _res2;
    uint4 _x;
    uint4 _y;
    _x = lwrand4(state);
    _y = lwrand4(state);
    _res1 = _lwrand_box_muller_double(_x.x, _x.y, _x.z, _x.w);
    _res2 = _lwrand_box_muller_double(_y.x, _y.y, _y.z, _y.w);
    result.x = _res1.x;
    result.y = _res1.y;
    result.z = _res2.x;
    result.w = _res2.y;
    return result;
}

//QUALIFIERS float _lwrand_normal_icdf(unsigned int x)
//{
//#if __LWDA_ARCH__ > 0 || defined(HOST_HAVE_ERFCILWF)
//    float s = LWRAND_SQRT2;
//    // Mirror to avoid loss of precision
//    if(x > 0x80000000UL) {
//        x = 0xffffffffUL - x;
//        s = -s;
//    }
//    float p = x * LWRAND_2POW32_ILW + (LWRAND_2POW32_ILW/2.0f);
//    // p is in (0, 0.5], 2p is in (0, 1]
//    return s * erfcilwf(2.0f * p);
//#else
//    x++;    //suppress warnings
//    return 0.0f;
//#endif
//}
//
//QUALIFIERS float _lwrand_normal_icdf(unsigned long long x)
//{
//#if __LWDA_ARCH__ > 0 || defined(HOST_HAVE_ERFCILWF)
//    unsigned int t = (unsigned int)(x >> 32);
//    float s = LWRAND_SQRT2;
//    // Mirror to avoid loss of precision
//    if(t > 0x80000000UL) {
//        t = 0xffffffffUL - t;
//        s = -s;
//    }
//    float p = t * LWRAND_2POW32_ILW + (LWRAND_2POW32_ILW/2.0f);
//    // p is in (0, 0.5], 2p is in (0, 1]
//    return s * erfcilwf(2.0f * p);
//#else
//    x++;
//    return 0.0f;
//#endif
//}
//
//QUALIFIERS double _lwrand_normal_icdf_double(unsigned int x)
//{
//#if __LWDA_ARCH__ > 0 || defined(HOST_HAVE_ERFCILWF)
//    double s = LWRAND_SQRT2_DOUBLE;
//    // Mirror to avoid loss of precision
//    if(x > 0x80000000UL) {
//        x = 0xffffffffUL - x;
//        s = -s;
//    }
//    double p = x * LWRAND_2POW32_ILW_DOUBLE + (LWRAND_2POW32_ILW_DOUBLE/2.0);
//    // p is in (0, 0.5], 2p is in (0, 1]
//    return s * erfcilw(2.0 * p);
//#else
//    x++;
//    return 0.0;
//#endif
//}
//
//QUALIFIERS double _lwrand_normal_icdf_double(unsigned long long x)
//{
//#if __LWDA_ARCH__ > 0 || defined(HOST_HAVE_ERFCILWF)
//    double s = LWRAND_SQRT2_DOUBLE;
//    x >>= 11;
//    // Mirror to avoid loss of precision
//    if(x > 0x10000000000000UL) {
//        x = 0x1fffffffffffffUL - x;
//        s = -s;
//    }
//    double p = x * LWRAND_2POW53_ILW_DOUBLE + (LWRAND_2POW53_ILW_DOUBLE/2.0);
//    // p is in (0, 0.5], 2p is in (0, 1]
//    return s * erfcilw(2.0 * p);
//#else
//    x++;
//    return 0.0;
//#endif
//}
//

/**
 * \brief Return a normally distributed float from an XORWOW generator.
 *
 * Return a single normally distributed float with mean \p 0.0f and
 * standard deviation \p 1.0f from the XORWOW generator in \p state,
 * increment position of generator by one.
 *
 * The implementation uses a Box-Muller transform to generate two
 * normally distributed results, then returns them one at a time.
 * See ::lwrand_normal2() for a more efficient version that returns
 * both results at once.
 *
 * \param state - Pointer to state to update
 *
 * \return Normally distributed float with mean \p 0.0f and standard deviation \p 1.0f
 */
QUALIFIERS float lwrand_normal(lwrandStateXORWOW_t *state)
{
    if(state->boxmuller_flag != EXTRA_FLAG_NORMAL) {
        unsigned int x, y;
        x = lwrand(state);
        y = lwrand(state);
        float2 v = _lwrand_box_muller(x, y);
        state->boxmuller_extra = v.y;
        state->boxmuller_flag = EXTRA_FLAG_NORMAL;
        return v.x;
    }
    state->boxmuller_flag = 0;
    return state->boxmuller_extra;
}

/**
 * \brief Return a normally distributed float from an Philox4_32_10 generator.
 *
 * Return a single normally distributed float with mean \p 0.0f and
 * standard deviation \p 1.0f from the Philox4_32_10 generator in \p state,
 * increment position of generator by one.
 *
 * The implementation uses a Box-Muller transform to generate two
 * normally distributed results, then returns them one at a time.
 * See ::lwrand_normal2() for a more efficient version that returns
 * both results at once.
 *
 * \param state - Pointer to state to update
 *
 * \return Normally distributed float with mean \p 0.0f and standard deviation \p 1.0f
 */

QUALIFIERS float lwrand_normal(lwrandStatePhilox4_32_10_t *state)
{
    if(state->boxmuller_flag != EXTRA_FLAG_NORMAL) {
        unsigned int x, y;
        x = lwrand(state);
        y = lwrand(state);
        float2 v = _lwrand_box_muller(x, y);
        state->boxmuller_extra = v.y;
        state->boxmuller_flag = EXTRA_FLAG_NORMAL;
        return v.x;
    }
    state->boxmuller_flag = 0;
    return state->boxmuller_extra;
}



/**
 * \brief Return a normally distributed float from an MRG32k3a generator.
 *
 * Return a single normally distributed float with mean \p 0.0f and
 * standard deviation \p 1.0f from the MRG32k3a generator in \p state,
 * increment position of generator by one.
 *
 * The implementation uses a Box-Muller transform to generate two
 * normally distributed results, then returns them one at a time.
 * See ::lwrand_normal2() for a more efficient version that returns
 * both results at once.
 *
 * \param state - Pointer to state to update
 *
 * \return Normally distributed float with mean \p 0.0f and standard deviation \p 1.0f
 */
QUALIFIERS float lwrand_normal(lwrandStateMRG32k3a_t *state)
{
    if(state->boxmuller_flag != EXTRA_FLAG_NORMAL) {
        float2 v = lwrand_box_muller_mrg(state);
        state->boxmuller_extra = v.y;
        state->boxmuller_flag = EXTRA_FLAG_NORMAL;
        return v.x;
    }
    state->boxmuller_flag = 0;
    return state->boxmuller_extra;
}

/**
 * \brief Return two normally distributed floats from an XORWOW generator.
 *
 * Return two normally distributed floats with mean \p 0.0f and
 * standard deviation \p 1.0f from the XORWOW generator in \p state,
 * increment position of generator by two.
 *
 * The implementation uses a Box-Muller transform to generate two
 * normally distributed results.
 *
 * \param state - Pointer to state to update
 *
 * \return Normally distributed float2 where each element is from a
 * distribution with mean \p 0.0f and standard deviation \p 1.0f
 */
QUALIFIERS float2 lwrand_normal2(lwrandStateXORWOW_t *state)
{
    return lwrand_box_muller(state);
}
/**
 * \brief Return two normally distributed floats from an Philox4_32_10 generator.
 *
 * Return two normally distributed floats with mean \p 0.0f and
 * standard deviation \p 1.0f from the Philox4_32_10 generator in \p state,
 * increment position of generator by two.
 *
 * The implementation uses a Box-Muller transform to generate two
 * normally distributed results.
 *
 * \param state - Pointer to state to update
 *
 * \return Normally distributed float2 where each element is from a
 * distribution with mean \p 0.0f and standard deviation \p 1.0f
 */
QUALIFIERS float2 lwrand_normal2(lwrandStatePhilox4_32_10_t *state)
{
    return lwrand_box_muller(state);
}

/**
 * \brief Return four normally distributed floats from an Philox4_32_10 generator.
 *
 * Return four normally distributed floats with mean \p 0.0f and
 * standard deviation \p 1.0f from the Philox4_32_10 generator in \p state,
 * increment position of generator by four.
 *
 * The implementation uses a Box-Muller transform to generate two
 * normally distributed results.
 *
 * \param state - Pointer to state to update
 *
 * \return Normally distributed float2 where each element is from a
 * distribution with mean \p 0.0f and standard deviation \p 1.0f
 */
QUALIFIERS float4 lwrand_normal4(lwrandStatePhilox4_32_10_t *state)
{
    return lwrand_box_muller4(state);
}



/**
 * \brief Return two normally distributed floats from an MRG32k3a generator.
 *
 * Return two normally distributed floats with mean \p 0.0f and
 * standard deviation \p 1.0f from the MRG32k3a generator in \p state,
 * increment position of generator by two.
 *
 * The implementation uses a Box-Muller transform to generate two
 * normally distributed results.
 *
 * \param state - Pointer to state to update
 *
 * \return Normally distributed float2 where each element is from a
 * distribution with mean \p 0.0f and standard deviation \p 1.0f
 */
QUALIFIERS float2 lwrand_normal2(lwrandStateMRG32k3a_t *state)
{
    return lwrand_box_muller_mrg(state);
}

/**
 * \brief Return a normally distributed float from a MTGP32 generator.
 *
 * Return a single normally distributed float with mean \p 0.0f and
 * standard deviation \p 1.0f from the MTGP32 generator in \p state,
 * increment position of generator.
 *
 * The implementation uses the ilwerse cumulative distribution function
 * to generate normally distributed results.
 *
 * \param state - Pointer to state to update
 *
 * \return Normally distributed float with mean \p 0.0f and standard deviation \p 1.0f
 */
QUALIFIERS float lwrand_normal(lwrandStateMtgp32_t *state)
{
    return _lwrand_normal_icdf(lwrand(state));
}
/**
 * \brief Return a normally distributed float from a Sobol32 generator.
 *
 * Return a single normally distributed float with mean \p 0.0f and
 * standard deviation \p 1.0f from the Sobol32 generator in \p state,
 * increment position of generator by one.
 *
 * The implementation uses the ilwerse cumulative distribution function
 * to generate normally distributed results.
 *
 * \param state - Pointer to state to update
 *
 * \return Normally distributed float with mean \p 0.0f and standard deviation \p 1.0f
 */
QUALIFIERS float lwrand_normal(lwrandStateSobol32_t *state)
{
    return _lwrand_normal_icdf(lwrand(state));
}

/**
 * \brief Return a normally distributed float from a scrambled Sobol32 generator.
 *
 * Return a single normally distributed float with mean \p 0.0f and
 * standard deviation \p 1.0f from the scrambled Sobol32 generator in \p state,
 * increment position of generator by one.
 *
 * The implementation uses the ilwerse cumulative distribution function
 * to generate normally distributed results.
 *
 * \param state - Pointer to state to update
 *
 * \return Normally distributed float with mean \p 0.0f and standard deviation \p 1.0f
 */
QUALIFIERS float lwrand_normal(lwrandStateScrambledSobol32_t *state)
{
    return _lwrand_normal_icdf(lwrand(state));
}

/**
 * \brief Return a normally distributed float from a Sobol64 generator.
 *
 * Return a single normally distributed float with mean \p 0.0f and
 * standard deviation \p 1.0f from the Sobol64 generator in \p state,
 * increment position of generator by one.
 *
 * The implementation uses the ilwerse cumulative distribution function
 * to generate normally distributed results.
 *
 * \param state - Pointer to state to update
 *
 * \return Normally distributed float with mean \p 0.0f and standard deviation \p 1.0f
 */
QUALIFIERS float lwrand_normal(lwrandStateSobol64_t *state)
{
    return _lwrand_normal_icdf(lwrand(state));
}

/**
 * \brief Return a normally distributed float from a scrambled Sobol64 generator.
 *
 * Return a single normally distributed float with mean \p 0.0f and
 * standard deviation \p 1.0f from the scrambled Sobol64 generator in \p state,
 * increment position of generator by one.
 *
 * The implementation uses the ilwerse cumulative distribution function
 * to generate normally distributed results.
 *
 * \param state - Pointer to state to update
 *
 * \return Normally distributed float with mean \p 0.0f and standard deviation \p 1.0f
 */
QUALIFIERS float lwrand_normal(lwrandStateScrambledSobol64_t *state)
{
    return _lwrand_normal_icdf(lwrand(state));
}

/**
 * \brief Return a normally distributed double from an XORWOW generator.
 *
 * Return a single normally distributed double with mean \p 0.0 and
 * standard deviation \p 1.0 from the XORWOW generator in \p state,
 * increment position of generator.
 *
 * The implementation uses a Box-Muller transform to generate two
 * normally distributed results, then returns them one at a time.
 * See ::lwrand_normal2_double() for a more efficient version that returns
 * both results at once.
 *
 * \param state - Pointer to state to update
 *
 * \return Normally distributed double with mean \p 0.0 and standard deviation \p 1.0
 */
QUALIFIERS double lwrand_normal_double(lwrandStateXORWOW_t *state)
{
    if(state->boxmuller_flag_double != EXTRA_FLAG_NORMAL) {
        unsigned int x0, x1, y0, y1;
        x0 = lwrand(state);
        x1 = lwrand(state);
        y0 = lwrand(state);
        y1 = lwrand(state);
        double2 v = _lwrand_box_muller_double(x0, x1, y0, y1);
        state->boxmuller_extra_double = v.y;
        state->boxmuller_flag_double = EXTRA_FLAG_NORMAL;
        return v.x;
    }
    state->boxmuller_flag_double = 0;
    return state->boxmuller_extra_double;
}

/**
 * \brief Return a normally distributed double from an Philox4_32_10 generator.
 *
 * Return a single normally distributed double with mean \p 0.0 and
 * standard deviation \p 1.0 from the Philox4_32_10 generator in \p state,
 * increment position of generator.
 *
 * The implementation uses a Box-Muller transform to generate two
 * normally distributed results, then returns them one at a time.
 * See ::lwrand_normal2_double() for a more efficient version that returns
 * both results at once.
 *
 * \param state - Pointer to state to update
 *
 * \return Normally distributed double with mean \p 0.0 and standard deviation \p 1.0
 */

QUALIFIERS double lwrand_normal_double(lwrandStatePhilox4_32_10_t *state)
{
    if(state->boxmuller_flag_double != EXTRA_FLAG_NORMAL) {
        uint4 _x;
        _x = lwrand4(state);
        double2 v = _lwrand_box_muller_double(_x.x, _x.y, _x.z, _x.w);
        state->boxmuller_extra_double = v.y;
        state->boxmuller_flag_double = EXTRA_FLAG_NORMAL;
        return v.x;
    }
    state->boxmuller_flag_double = 0;
    return state->boxmuller_extra_double;
}


/**
 * \brief Return a normally distributed double from an MRG32k3a generator.
 *
 * Return a single normally distributed double with mean \p 0.0 and
 * standard deviation \p 1.0 from the XORWOW generator in \p state,
 * increment position of generator.
 *
 * The implementation uses a Box-Muller transform to generate two
 * normally distributed results, then returns them one at a time.
 * See ::lwrand_normal2_double() for a more efficient version that returns
 * both results at once.
 *
 * \param state - Pointer to state to update
 *
 * \return Normally distributed double with mean \p 0.0 and standard deviation \p 1.0
 */
QUALIFIERS double lwrand_normal_double(lwrandStateMRG32k3a_t *state)
{
    if(state->boxmuller_flag_double != EXTRA_FLAG_NORMAL) {
        double2 v = lwrand_box_muller_mrg_double(state);
        state->boxmuller_extra_double = v.y;
        state->boxmuller_flag_double = EXTRA_FLAG_NORMAL;
        return v.x;
    }
    state->boxmuller_flag_double = 0;
    return state->boxmuller_extra_double;
}

/**
 * \brief Return two normally distributed doubles from an XORWOW generator.
 *
 * Return two normally distributed doubles with mean \p 0.0 and
 * standard deviation \p 1.0 from the XORWOW generator in \p state,
 * increment position of generator by 2.
 *
 * The implementation uses a Box-Muller transform to generate two
 * normally distributed results.
 *
 * \param state - Pointer to state to update
 *
 * \return Normally distributed double2 where each element is from a
 * distribution with mean \p 0.0 and standard deviation \p 1.0
 */
QUALIFIERS double2 lwrand_normal2_double(lwrandStateXORWOW_t *state)
{
    return lwrand_box_muller_double(state);
}

/**
 * \brief Return two normally distributed doubles from an Philox4_32_10 generator.
 *
 * Return two normally distributed doubles with mean \p 0.0 and
 * standard deviation \p 1.0 from the Philox4_32_10 generator in \p state,
 * increment position of generator by 2.
 *
 * The implementation uses a Box-Muller transform to generate two
 * normally distributed results.
 *
 * \param state - Pointer to state to update
 *
 * \return Normally distributed double2 where each element is from a
 * distribution with mean \p 0.0 and standard deviation \p 1.0
 */
QUALIFIERS double2 lwrand_normal2_double(lwrandStatePhilox4_32_10_t *state)
{
    uint4 _x;
    double2 result;

    _x = lwrand4(state);
    double2 v1 = _lwrand_box_muller_double(_x.x, _x.y, _x.z, _x.w);
    result.x = v1.x;
    result.y = v1.y;

    return result;
}

 // not a part of API
QUALIFIERS double4 lwrand_normal4_double(lwrandStatePhilox4_32_10_t *state)
{
    uint4 _x;
    uint4 _y;
    double4 result;

    _x = lwrand4(state);
    _y = lwrand4(state);
    double2 v1 = _lwrand_box_muller_double(_x.x, _x.y, _x.z, _x.w);
    double2 v2 = _lwrand_box_muller_double(_y.x, _y.y, _y.z, _y.w);
    result.x = v1.x;
    result.y = v1.y;
    result.z = v2.x;
    result.w = v2.y;

    return result;
}


/**
 * \brief Return two normally distributed doubles from an MRG32k3a generator.
 *
 * Return two normally distributed doubles with mean \p 0.0 and
 * standard deviation \p 1.0 from the MRG32k3a generator in \p state,
 * increment position of generator.
 *
 * The implementation uses a Box-Muller transform to generate two
 * normally distributed results.
 *
 * \param state - Pointer to state to update
 *
 * \return Normally distributed double2 where each element is from a
 * distribution with mean \p 0.0 and standard deviation \p 1.0
 */
QUALIFIERS double2 lwrand_normal2_double(lwrandStateMRG32k3a_t *state)
{
    return lwrand_box_muller_mrg_double(state);
}

/**
 * \brief Return a normally distributed double from an MTGP32 generator.
 *
 * Return a single normally distributed double with mean \p 0.0 and
 * standard deviation \p 1.0 from the MTGP32 generator in \p state,
 * increment position of generator.
 *
 * The implementation uses the ilwerse cumulative distribution function
 * to generate normally distributed results.
 *
 * \param state - Pointer to state to update
 *
 * \return Normally distributed double with mean \p 0.0 and standard deviation \p 1.0
 */
QUALIFIERS double lwrand_normal_double(lwrandStateMtgp32_t *state)
{
    return _lwrand_normal_icdf_double(lwrand(state));
}

/**
 * \brief Return a normally distributed double from an Sobol32 generator.
 *
 * Return a single normally distributed double with mean \p 0.0 and
 * standard deviation \p 1.0 from the Sobol32 generator in \p state,
 * increment position of generator by one.
 *
 * The implementation uses the ilwerse cumulative distribution function
 * to generate normally distributed results.
 *
 * \param state - Pointer to state to update
 *
 * \return Normally distributed double with mean \p 0.0 and standard deviation \p 1.0
 */
QUALIFIERS double lwrand_normal_double(lwrandStateSobol32_t *state)
{
    return _lwrand_normal_icdf_double(lwrand(state));
}

/**
 * \brief Return a normally distributed double from a scrambled Sobol32 generator.
 *
 * Return a single normally distributed double with mean \p 0.0 and
 * standard deviation \p 1.0 from the scrambled Sobol32 generator in \p state,
 * increment position of generator by one.
 *
 * The implementation uses the ilwerse cumulative distribution function
 * to generate normally distributed results.
 *
 * \param state - Pointer to state to update
 *
 * \return Normally distributed double with mean \p 0.0 and standard deviation \p 1.0
 */
QUALIFIERS double lwrand_normal_double(lwrandStateScrambledSobol32_t *state)
{
    return _lwrand_normal_icdf_double(lwrand(state));
}

/**
 * \brief Return a normally distributed double from a Sobol64 generator.
 *
 * Return a single normally distributed double with mean \p 0.0 and
 * standard deviation \p 1.0 from the Sobol64 generator in \p state,
 * increment position of generator by one.
 *
 * The implementation uses the ilwerse cumulative distribution function
 * to generate normally distributed results.
 *
 * \param state - Pointer to state to update
 *
 * \return Normally distributed double with mean \p 0.0 and standard deviation \p 1.0
 */
QUALIFIERS double lwrand_normal_double(lwrandStateSobol64_t *state)
{
    return _lwrand_normal_icdf_double(lwrand(state));
}

/**
 * \brief Return a normally distributed double from a scrambled Sobol64 generator.
 *
 * Return a single normally distributed double with mean \p 0.0 and
 * standard deviation \p 1.0 from the scrambled Sobol64 generator in \p state,
 * increment position of generator by one.
 *
 * The implementation uses the ilwerse cumulative distribution function
 * to generate normally distributed results.
 *
 * \param state - Pointer to state to update
 *
 * \return Normally distributed double with mean \p 0.0 and standard deviation \p 1.0
 */
QUALIFIERS double lwrand_normal_double(lwrandStateScrambledSobol64_t *state)
{
    return _lwrand_normal_icdf_double(lwrand(state));
}
#endif // !defined(LWRAND_NORMAL_H_)
