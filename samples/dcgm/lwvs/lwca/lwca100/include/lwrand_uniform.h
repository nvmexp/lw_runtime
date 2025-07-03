
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


#if !defined(LWRAND_UNIFORM_H_)
#define LWRAND_UNIFORM_H_

/**
 * \defgroup DEVICE Device API
 *
 * @{
 */

#include "lwrand_mrg32k3a.h"
#include "lwrand_mtgp32_kernel.h"
#include <math.h>

#include "lwrand_philox4x32_x.h" 


QUALIFIERS float _lwrand_uniform(unsigned int x)
{
    return x * LWRAND_2POW32_ILW + (LWRAND_2POW32_ILW/2.0f);
}

QUALIFIERS float4 _lwrand_uniform4(uint4 x)
{
    float4 y;
    y.x = x.x * LWRAND_2POW32_ILW + (LWRAND_2POW32_ILW/2.0f);
    y.y = x.y * LWRAND_2POW32_ILW + (LWRAND_2POW32_ILW/2.0f);
    y.z = x.z * LWRAND_2POW32_ILW + (LWRAND_2POW32_ILW/2.0f);
    y.w = x.w * LWRAND_2POW32_ILW + (LWRAND_2POW32_ILW/2.0f);
    return y; 
}

QUALIFIERS float _lwrand_uniform(unsigned long long x)
{
    unsigned int t;
    t = (unsigned int)(x >> 32); 
    return t * LWRAND_2POW32_ILW + (LWRAND_2POW32_ILW/2.0f);
}

QUALIFIERS double _lwrand_uniform_double(unsigned int x)
{
    return x * LWRAND_2POW32_ILW_DOUBLE + (LWRAND_2POW32_ILW_DOUBLE/2.0);
}

QUALIFIERS double _lwrand_uniform_double(unsigned long long x)
{
    return (x >> 11) * LWRAND_2POW53_ILW_DOUBLE + (LWRAND_2POW53_ILW_DOUBLE/2.0);
}

QUALIFIERS double4 _lwrand_uniform4_double(uint4 x)
{
    double4 result;
    result.x = (x.x>>11) * LWRAND_2POW32_ILW_DOUBLE + (LWRAND_2POW32_ILW_DOUBLE/2.0);
    result.y = (x.y>>11) * LWRAND_2POW32_ILW_DOUBLE + (LWRAND_2POW32_ILW_DOUBLE/2.0);
    result.z = (x.z>>11) * LWRAND_2POW32_ILW_DOUBLE + (LWRAND_2POW32_ILW_DOUBLE/2.0);
    result.w = (x.w>>11) * LWRAND_2POW32_ILW_DOUBLE + (LWRAND_2POW32_ILW_DOUBLE/2.0);
    return result;
}

QUALIFIERS double _lwrand_uniform_double_hq(unsigned int x, unsigned int y)
{
    unsigned long long z = (unsigned long long)x ^ 
        ((unsigned long long)y << (53 - 32));
    return z * LWRAND_2POW53_ILW_DOUBLE + (LWRAND_2POW53_ILW_DOUBLE/2.0);
}

QUALIFIERS float lwrand_uniform(lwrandStateTest_t *state)
{
    return _lwrand_uniform(lwrand(state));
}

QUALIFIERS double lwrand_uniform_double(lwrandStateTest_t *state)
{
    return _lwrand_uniform_double(lwrand(state));
}

/**
 * \brief Return a uniformly distributed float from an XORWOW generator.
 *
 * Return a uniformly distributed float between \p 0.0f and \p 1.0f 
 * from the XORWOW generator in \p state, increment position of generator.
 * Output range excludes \p 0.0f but includes \p 1.0f.  Denormalized floating
 * point outputs are never returned.
 *
 * The implementation may use any number of calls to \p lwrand() to
 * get enough random bits to create the return value.  The current
 * implementation uses one call.
 *
 * \param state - Pointer to state to update
 *
 * \return uniformly distributed float between \p 0.0f and \p 1.0f
 */
QUALIFIERS float lwrand_uniform(lwrandStateXORWOW_t *state)
{
    return _lwrand_uniform(lwrand(state));
}

/**
 * \brief Return a uniformly distributed double from an XORWOW generator.
 *
 * Return a uniformly distributed double between \p 0.0 and \p 1.0 
 * from the XORWOW generator in \p state, increment position of generator.
 * Output range excludes \p 0.0 but includes \p 1.0.  Denormalized floating
 * point outputs are never returned.
 *
 * The implementation may use any number of calls to \p lwrand() to
 * get enough random bits to create the return value.  The current
 * implementation uses exactly two calls.
 *
 * \param state - Pointer to state to update
 *
 * \return uniformly distributed double between \p 0.0 and \p 1.0
 */
QUALIFIERS double lwrand_uniform_double(lwrandStateXORWOW_t *state)
{
    unsigned int x, y;
    x = lwrand(state);
    y = lwrand(state);
    return _lwrand_uniform_double_hq(x, y);
}
/**
 * \brief Return a uniformly distributed float from an MRG32k3a generator.
 *
 * Return a uniformly distributed float between \p 0.0f and \p 1.0f 
 * from the MRG32k3a generator in \p state, increment position of generator.
 * Output range excludes \p 0.0f but includes \p 1.0f.  Denormalized floating
 * point outputs are never returned.
 *
 * The implementation returns up to 23 bits of mantissa, with the minimum 
 * return value \f$ 2^{-32} \f$ 
 *
 * \param state - Pointer to state to update
 *
 * \return uniformly distributed float between \p 0.0f and \p 1.0f
 */
QUALIFIERS float lwrand_uniform(lwrandStateMRG32k3a_t *state)
{
    return ((float)(lwrand_MRG32k3a(state)*MRG32K3A_NORM));
}

/**
 * \brief Return a uniformly distributed double from an MRG32k3a generator.
 *
 * Return a uniformly distributed double between \p 0.0 and \p 1.0 
 * from the MRG32k3a generator in \p state, increment position of generator.
 * Output range excludes \p 0.0 but includes \p 1.0.  Denormalized floating
 * point outputs are never returned. 
 *
 * Note the implementation returns at most 32 random bits of mantissa as 
 * outlined in the seminal paper by L'Elwyer.
 *
 * \param state - Pointer to state to update
 *
 * \return uniformly distributed double between \p 0.0 and \p 1.0
 */
QUALIFIERS double lwrand_uniform_double(lwrandStateMRG32k3a_t *state)
{
    return lwrand_MRG32k3a(state)*MRG32K3A_NORM;
}



/**
 * \brief Return a uniformly distributed tuple of 2 doubles from an Philox4_32_10 generator.
 *
 * Return a uniformly distributed 2 doubles (double4) between \p 0.0 and \p 1.0 
 * from the Philox4_32_10 generator in \p state, increment position of generator by 4.
 * Output range excludes \p 0.0 but includes \p 1.0.  Denormalized floating
 * point outputs are never returned. 
 *
 * \param state - Pointer to state to update
 *
 * \return 2 uniformly distributed doubles between \p 0.0 and \p 1.0
 */

QUALIFIERS double2 lwrand_uniform2_double(lwrandStatePhilox4_32_10_t *state)
{
    uint4 _x;
    double2 result;
    _x = lwrand4(state);
    result.x = _lwrand_uniform_double_hq(_x.x,_x.y);
    result.y = _lwrand_uniform_double_hq(_x.z,_x.w);
    return result;
}


// not a part of API
QUALIFIERS double4 lwrand_uniform4_double(lwrandStatePhilox4_32_10_t *state)
{
    uint4 _x, _y;
    double4 result;
    _x = lwrand4(state);
    _y = lwrand4(state);
    result.x = _lwrand_uniform_double_hq(_x.x,_x.y);
    result.y = _lwrand_uniform_double_hq(_x.z,_x.w);
    result.z = _lwrand_uniform_double_hq(_y.x,_y.y);
    result.w = _lwrand_uniform_double_hq(_y.z,_y.w);
    return result;
}

/**
 * \brief Return a uniformly distributed float from a Philox4_32_10 generator.
 *
 * Return a uniformly distributed float between \p 0.0f and \p 1.0f 
 * from the Philox4_32_10 generator in \p state, increment position of generator.
 * Output range excludes \p 0.0f but includes \p 1.0f.  Denormalized floating
 * point outputs are never returned.
 * 
 * \param state - Pointer to state to update
 *
 * \return uniformly distributed float between \p 0.0 and \p 1.0

 */
QUALIFIERS float lwrand_uniform(lwrandStatePhilox4_32_10_t *state)
{
   return _lwrand_uniform(lwrand(state));
}

/**
 * \brief Return a uniformly distributed tuple of 4 floats from a Philox4_32_10 generator.
 *
 * Return a uniformly distributed 4 floats between \p 0.0f and \p 1.0f 
 * from the Philox4_32_10 generator in \p state, increment position of generator by 4.
 * Output range excludes \p 0.0f but includes \p 1.0f.  Denormalized floating
 * point outputs are never returned.
 * 
 * \param state - Pointer to state to update
 *
 * \return uniformly distributed float between \p 0.0 and \p 1.0

 */

QUALIFIERS float4 lwrand_uniform4(lwrandStatePhilox4_32_10_t *state)
{
   return _lwrand_uniform4(lwrand4(state));
}

/**
 * \brief Return a uniformly distributed float from a MTGP32 generator.
 *
 * Return a uniformly distributed float between \p 0.0f and \p 1.0f 
 * from the MTGP32 generator in \p state, increment position of generator.
 * Output range excludes \p 0.0f but includes \p 1.0f.  Denormalized floating
 * point outputs are never returned.
 *
 * \param state - Pointer to state to update
 *
 * \return uniformly distributed float between \p 0.0f and \p 1.0f
 */
QUALIFIERS float lwrand_uniform(lwrandStateMtgp32_t *state)
{
    return _lwrand_uniform(lwrand(state));
}
/**
 * \brief Return a uniformly distributed double from a MTGP32 generator.
 *
 * Return a uniformly distributed double between \p 0.0f and \p 1.0f 
 * from the MTGP32 generator in \p state, increment position of generator.
 * Output range excludes \p 0.0f but includes \p 1.0f.  Denormalized floating
 * point outputs are never returned.
 *
 * \param state - Pointer to state to update
 *
 * \return uniformly distributed double between \p 0.0f and \p 1.0f
 */
QUALIFIERS double lwrand_uniform_double(lwrandStateMtgp32_t *state)
{
    return _lwrand_uniform_double(lwrand(state));
}

/**
 * \brief Return a uniformly distributed double from a Philox4_32_10 generator.
 *
 * Return a uniformly distributed double between \p 0.0f and \p 1.0f 
 * from the Philox4_32_10 generator in \p state, increment position of generator.
 * Output range excludes \p 0.0f but includes \p 1.0f.  Denormalized floating
 * point outputs are never returned.
 *
 * \param state - Pointer to state to update
 *
 * \return uniformly distributed double between \p 0.0f and \p 1.0f
 */

QUALIFIERS double lwrand_uniform_double(lwrandStatePhilox4_32_10_t *state)
{
    return _lwrand_uniform_double(lwrand(state));
}


/**
 * \brief Return a uniformly distributed float from a Sobol32 generator.
 *
 * Return a uniformly distributed float between \p 0.0f and \p 1.0f 
 * from the Sobol32 generator in \p state, increment position of generator.
 * Output range excludes \p 0.0f but includes \p 1.0f.  Denormalized floating
 * point outputs are never returned.
 *
 * The implementation is guaranteed to use a single call to \p lwrand().
 *
 * \param state - Pointer to state to update
 *
 * \return uniformly distributed float between \p 0.0f and \p 1.0f
 */
QUALIFIERS float lwrand_uniform(lwrandStateSobol32_t *state)
{
    return _lwrand_uniform(lwrand(state));
}

/**
 * \brief Return a uniformly distributed double from a Sobol32 generator.
 *
 * Return a uniformly distributed double between \p 0.0 and \p 1.0 
 * from the Sobol32 generator in \p state, increment position of generator.
 * Output range excludes \p 0.0 but includes \p 1.0.  Denormalized floating
 * point outputs are never returned.
 *
 * The implementation is guaranteed to use a single call to \p lwrand()
 * to preserve the quasirandom properties of the sequence.
 *
 * \param state - Pointer to state to update
 *
 * \return uniformly distributed double between \p 0.0 and \p 1.0
 */
QUALIFIERS double lwrand_uniform_double(lwrandStateSobol32_t *state)
{
    return _lwrand_uniform_double(lwrand(state));
}
/**
 * \brief Return a uniformly distributed float from a scrambled Sobol32 generator.
 *
 * Return a uniformly distributed float between \p 0.0f and \p 1.0f 
 * from the scrambled Sobol32 generator in \p state, increment position of generator.
 * Output range excludes \p 0.0f but includes \p 1.0f.  Denormalized floating
 * point outputs are never returned.
 *
 * The implementation is guaranteed to use a single call to \p lwrand().
 *
 * \param state - Pointer to state to update
 *
 * \return uniformly distributed float between \p 0.0f and \p 1.0f
 */
QUALIFIERS float lwrand_uniform(lwrandStateScrambledSobol32_t *state)
{
    return _lwrand_uniform(lwrand(state));
}

/**
 * \brief Return a uniformly distributed double from a scrambled Sobol32 generator.
 *
 * Return a uniformly distributed double between \p 0.0 and \p 1.0 
 * from the scrambled Sobol32 generator in \p state, increment position of generator.
 * Output range excludes \p 0.0 but includes \p 1.0.  Denormalized floating
 * point outputs are never returned.
 *
 * The implementation is guaranteed to use a single call to \p lwrand()
 * to preserve the quasirandom properties of the sequence.
 *
 * \param state - Pointer to state to update
 *
 * \return uniformly distributed double between \p 0.0 and \p 1.0
 */
QUALIFIERS double lwrand_uniform_double(lwrandStateScrambledSobol32_t *state)
{
    return _lwrand_uniform_double(lwrand(state));
}
/**
 * \brief Return a uniformly distributed float from a Sobol64 generator.
 *
 * Return a uniformly distributed float between \p 0.0f and \p 1.0f 
 * from the Sobol64 generator in \p state, increment position of generator.
 * Output range excludes \p 0.0f but includes \p 1.0f.  Denormalized floating
 * point outputs are never returned.
 *
 * The implementation is guaranteed to use a single call to \p lwrand().
 *
 * \param state - Pointer to state to update
 *
 * \return uniformly distributed float between \p 0.0f and \p 1.0f
 */
QUALIFIERS float lwrand_uniform(lwrandStateSobol64_t *state)
{
    return _lwrand_uniform(lwrand(state));
}

/**
 * \brief Return a uniformly distributed double from a Sobol64 generator.
 *
 * Return a uniformly distributed double between \p 0.0 and \p 1.0 
 * from the Sobol64 generator in \p state, increment position of generator.
 * Output range excludes \p 0.0 but includes \p 1.0.  Denormalized floating
 * point outputs are never returned.
 *
 * The implementation is guaranteed to use a single call to \p lwrand()
 * to preserve the quasirandom properties of the sequence.
 *
 * \param state - Pointer to state to update
 *
 * \return uniformly distributed double between \p 0.0 and \p 1.0
 */
QUALIFIERS double lwrand_uniform_double(lwrandStateSobol64_t *state)
{
    return _lwrand_uniform_double(lwrand(state));
}
/**
 * \brief Return a uniformly distributed float from a scrambled Sobol64 generator.
 *
 * Return a uniformly distributed float between \p 0.0f and \p 1.0f 
 * from the scrambled Sobol64 generator in \p state, increment position of generator.
 * Output range excludes \p 0.0f but includes \p 1.0f.  Denormalized floating
 * point outputs are never returned.
 *
 * The implementation is guaranteed to use a single call to \p lwrand().
 *
 * \param state - Pointer to state to update
 *
 * \return uniformly distributed float between \p 0.0f and \p 1.0f
 */
QUALIFIERS float lwrand_uniform(lwrandStateScrambledSobol64_t *state)
{
    return _lwrand_uniform(lwrand(state));
}

/**
 * \brief Return a uniformly distributed double from a scrambled Sobol64 generator.
 *
 * Return a uniformly distributed double between \p 0.0 and \p 1.0 
 * from the scrambled Sobol64 generator in \p state, increment position of generator.
 * Output range excludes \p 0.0 but includes \p 1.0.  Denormalized floating
 * point outputs are never returned.
 *
 * The implementation is guaranteed to use a single call to \p lwrand()
 * to preserve the quasirandom properties of the sequence.
 *
 * \param state - Pointer to state to update
 *
 * \return uniformly distributed double between \p 0.0 and \p 1.0
 */
QUALIFIERS double lwrand_uniform_double(lwrandStateScrambledSobol64_t *state)
{
    return _lwrand_uniform_double(lwrand(state));
}

#endif // !defined(LWRAND_UNIFORM_H_)
