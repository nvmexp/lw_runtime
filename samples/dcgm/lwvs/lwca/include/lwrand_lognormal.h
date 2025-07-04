
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


#if !defined(LWRAND_LOGNORMAL_H_)
#define LWRAND_LOGNORMAL_H_

/**
 * \defgroup DEVICE Device API
 *
 * @{
 */

#include "lwrand_mrg32k3a.h"
#include "lwrand_mtgp32_kernel.h"
#include <math.h>

#include "lwrand_philox4x32_x.h" 


/**
 * \brief Return a log-normally distributed float from an XORWOW generator.
 *
 * Return a single log-normally distributed float derived from a normal
 * distribution with mean \p mean and standard deviation \p stddev 
 * from the XORWOW generator in \p state, 
 * increment position of generator by one.
 *
 * The implementation uses a Box-Muller transform to generate two
 * normally distributed results, transforms them to log-normal distribution,
 * then returns them one at a time.
 * See ::lwrand_log_normal2() for a more efficient version that returns
 * both results at once.
 *
 * \param state  - Pointer to state to update
 * \param mean   - Mean of the related normal distribution
 * \param stddev - Standard deviation of the related normal distribution
 *
 * \return Log-normally distributed float with mean \p mean and standard deviation \p stddev
 */
QUALIFIERS float lwrand_log_normal(lwrandStateXORWOW_t *state, float mean, float stddev)
{
    if(state->boxmuller_flag != EXTRA_FLAG_LOG_NORMAL) {
        unsigned int x, y;
        x = lwrand(state);
        y = lwrand(state);
        float2 v = _lwrand_box_muller(x, y);
        state->boxmuller_extra = expf(mean + (stddev * v.y));
        state->boxmuller_flag = EXTRA_FLAG_LOG_NORMAL;
        return expf(mean + (stddev * v.x));
    }
    state->boxmuller_flag = 0;
    return state->boxmuller_extra;
}

/**
 * \brief Return a log-normally distributed float from an Philox4_32_10 generator.
 *
 * Return a single log-normally distributed float derived from a normal
 * distribution with mean \p mean and standard deviation \p stddev 
 * from the Philox4_32_10 generator in \p state, 
 * increment position of generator by one.
 *
 * The implementation uses a Box-Muller transform to generate two
 * normally distributed results, transforms them to log-normal distribution,
 * then returns them one at a time.
 * See ::lwrand_log_normal2() for a more efficient version that returns
 * both results at once.
 *
 * \param state  - Pointer to state to update
 * \param mean   - Mean of the related normal distribution
 * \param stddev - Standard deviation of the related normal distribution
 *
 * \return Log-normally distributed float with mean \p mean and standard deviation \p stddev
 */

QUALIFIERS float lwrand_log_normal(lwrandStatePhilox4_32_10_t *state, float mean, float stddev)
{
    if(state->boxmuller_flag != EXTRA_FLAG_LOG_NORMAL) {
        unsigned int x, y;
        x = lwrand(state);
        y = lwrand(state);
        float2 v = _lwrand_box_muller(x, y);
        state->boxmuller_extra = expf(mean + (stddev * v.y));
        state->boxmuller_flag = EXTRA_FLAG_LOG_NORMAL;
        return expf(mean + (stddev * v.x));
    }
    state->boxmuller_flag = 0;
    return state->boxmuller_extra;
}

/**
 * \brief Return two normally distributed floats from an XORWOW generator.
 *
 * Return two log-normally distributed floats derived from a normal
 * distribution with mean \p mean and standard deviation \p stddev 
 * from the XORWOW generator in \p state,
 * increment position of generator by two.
 *
 * The implementation uses a Box-Muller transform to generate two
 * normally distributed results, then transforms them to log-normal.
 *
 * \param state - Pointer to state to update
 * \param mean   - Mean of the related normal distribution
 * \param stddev - Standard deviation of the related normal distribution
 *
 * \return Log-normally distributed float2 where each element is from a
 * distribution with mean \p mean and standard deviation \p stddev
 */
QUALIFIERS float2 lwrand_log_normal2(lwrandStateXORWOW_t *state, float mean, float stddev)
{
    float2 v = lwrand_box_muller(state);
    v.x = expf(mean + (stddev * v.x));
    v.y = expf(mean + (stddev * v.y));
    return v;
}

/**
 * \brief Return two normally distributed floats from an Philox4_32_10 generator.
 *
 * Return two log-normally distributed floats derived from a normal
 * distribution with mean \p mean and standard deviation \p stddev 
 * from the Philox4_32_10 generator in \p state,
 * increment position of generator by two.
 *
 * The implementation uses a Box-Muller transform to generate two
 * normally distributed results, then transforms them to log-normal.
 *
 * \param state - Pointer to state to update
 * \param mean   - Mean of the related normal distribution
 * \param stddev - Standard deviation of the related normal distribution
 *
 * \return Log-normally distributed float2 where each element is from a
 * distribution with mean \p mean and standard deviation \p stddev
 */
QUALIFIERS float2 lwrand_log_normal2(lwrandStatePhilox4_32_10_t *state, float mean, float stddev)
{
    float2 v = lwrand_box_muller(state);
    v.x = expf(mean + (stddev * v.x));
    v.y = expf(mean + (stddev * v.y));
    return v;
}
/**
 * \brief Return four normally distributed floats from an Philox4_32_10 generator.
 *
 * Return four log-normally distributed floats derived from a normal
 * distribution with mean \p mean and standard deviation \p stddev 
 * from the Philox4_32_10 generator in \p state,
 * increment position of generator by four.
 *
 * The implementation uses a Box-Muller transform to generate two
 * normally distributed results, then transforms them to log-normal.
 *
 * \param state - Pointer to state to update
 * \param mean   - Mean of the related normal distribution
 * \param stddev - Standard deviation of the related normal distribution
 *
 * \return Log-normally distributed float4 where each element is from a
 * distribution with mean \p mean and standard deviation \p stddev
 */
QUALIFIERS float4 lwrand_log_normal4(lwrandStatePhilox4_32_10_t *state, float mean, float stddev)
{
    float4 v = lwrand_box_muller4(state);
    v.x = expf(mean + (stddev * v.x));
    v.y = expf(mean + (stddev * v.y));
    v.z = expf(mean + (stddev * v.z));
    v.w = expf(mean + (stddev * v.w));
    return v;
}

/**
 * \brief Return a log-normally distributed float from an MRG32k3a generator.
 *
 * Return a single log-normally distributed float derived from a normal
 * distribution with mean \p mean and standard deviation \p stddev 
 * from the MRG32k3a generator in \p state, 
 * increment position of generator by one.
 *
 * The implementation uses a Box-Muller transform to generate two
 * normally distributed results, transforms them to log-normal distribution,
 * then returns them one at a time.
 * See ::lwrand_log_normal2() for a more efficient version that returns
 * both results at once.
 *
 * \param state  - Pointer to state to update
 * \param mean   - Mean of the related normal distribution
 * \param stddev - Standard deviation of the related normal distribution
 *
 * \return Log-normally distributed float with mean \p mean and standard deviation \p stddev
 */
QUALIFIERS float lwrand_log_normal(lwrandStateMRG32k3a_t *state, float mean, float stddev)
{
    if(state->boxmuller_flag != EXTRA_FLAG_LOG_NORMAL) {
        float2 v = lwrand_box_muller_mrg(state);
        state->boxmuller_extra = expf(mean + (stddev * v.y));
        state->boxmuller_flag = EXTRA_FLAG_LOG_NORMAL;
        return expf(mean + (stddev * v.x));
    }
    state->boxmuller_flag = 0;
    return state->boxmuller_extra;
}

/**
 * \brief Return two normally distributed floats from an MRG32k3a generator.
 *
 * Return two log-normally distributed floats derived from a normal
 * distribution with mean \p mean and standard deviation \p stddev 
 * from the MRG32k3a generator in \p state,
 * increment position of generator by two.
 *
 * The implementation uses a Box-Muller transform to generate two
 * normally distributed results, then transforms them to log-normal.
 *
 * \param state - Pointer to state to update
 * \param mean   - Mean of the related normal distribution
 * \param stddev - Standard deviation of the related normal distribution
 *
 * \return Log-normally distributed float2 where each element is from a
 * distribution with mean \p mean and standard deviation \p stddev
 */
QUALIFIERS float2 lwrand_log_normal2(lwrandStateMRG32k3a_t *state, float mean, float stddev)
{
    float2 v = lwrand_box_muller_mrg(state);
    v.x = expf(mean + (stddev * v.x));
    v.y = expf(mean + (stddev * v.y));
    return v;
}

/**
 * \brief Return a log-normally distributed float from an MTGP32 generator.
 *
 * Return a single log-normally distributed float derived from a normal
 * distribution with mean \p mean and standard deviation \p stddev 
 * from the MTGP32 generator in \p state,
 * increment position of generator.
 *
 * The implementation uses the ilwerse cumulative distribution function
 * to generate a normally distributed result, then transforms the result
 * to log-normal.
 *
 * \param state - Pointer to state to update
 * \param mean   - Mean of the related normal distribution
 * \param stddev - Standard deviation of the related normal distribution
 *
 * \return Log-normally distributed float with mean \p mean and standard deviation \p stddev
 */
QUALIFIERS float lwrand_log_normal(lwrandStateMtgp32_t *state, float mean, float stddev)
{
    return expf(mean + (stddev * _lwrand_normal_icdf(lwrand(state))));
}

/**
 * \brief Return a log-normally distributed float from a Sobol32 generator.
 *
 * Return a single log-normally distributed float derived from a normal
 * distribution with mean \p mean and standard deviation \p stddev 
 * from the Sobol32 generator in \p state,
 * increment position of generator by one.
 *
 * The implementation uses the ilwerse cumulative distribution function
 * to generate a normally distributed result, then transforms the result
 * to log-normal.
 *
 * \param state - Pointer to state to update
 * \param mean   - Mean of the related normal distribution
 * \param stddev - Standard deviation of the related normal distribution
 *
 * \return Log-normally distributed float with mean \p mean and standard deviation \p stddev
 */
QUALIFIERS float lwrand_log_normal(lwrandStateSobol32_t *state, float mean, float stddev)
{
    return expf(mean + (stddev * _lwrand_normal_icdf(lwrand(state))));
}
/**
 * \brief Return a log-normally distributed float from a scrambled Sobol32 generator.
 *
 * Return a single log-normally distributed float derived from a normal
 * distribution with mean \p mean and standard deviation \p stddev 
 * from the scrambled Sobol32 generator in \p state,
 * increment position of generator by one.
 *
 * The implementation uses the ilwerse cumulative distribution function
 * to generate a normally distributed result, then transforms the result
 * to log-normal.
 *
 * \param state - Pointer to state to update
 * \param mean   - Mean of the related normal distribution
 * \param stddev - Standard deviation of the related normal distribution
 *
 * \return Log-normally distributed float with mean \p mean and standard deviation \p stddev
 */
QUALIFIERS float lwrand_log_normal(lwrandStateScrambledSobol32_t *state, float mean, float stddev)
{
    return expf(mean + (stddev * _lwrand_normal_icdf(lwrand(state))));
}

/**
 * \brief Return a log-normally distributed float from a Sobol64 generator.
 *
 * Return a single log-normally distributed float derived from a normal
 * distribution with mean \p mean and standard deviation \p stddev 
 * from the Sobol64 generator in \p state,
 * increment position of generator by one.
 *
 * The implementation uses the ilwerse cumulative distribution function
 * to generate normally distributed results, then colwerts to log-normal
 * distribution.
 *
 * \param state - Pointer to state to update
 * \param mean   - Mean of the related normal distribution
 * \param stddev - Standard deviation of the related normal distribution
 *
 * \return Log-normally distributed float with mean \p mean and standard deviation \p stddev
 */
QUALIFIERS float lwrand_log_normal(lwrandStateSobol64_t *state, float mean, float stddev)
{
    return expf(mean + (stddev * _lwrand_normal_icdf(lwrand(state))));
}

/**
 * \brief Return a log-normally distributed float from a scrambled Sobol64 generator.
 *
 * Return a single log-normally distributed float derived from a normal
 * distribution with mean \p mean and standard deviation \p stddev 
 * from the scrambled Sobol64 generator in \p state,
 * increment position of generator by one.
 *
 * The implementation uses the ilwerse cumulative distribution function
 * to generate normally distributed results, then colwerts to log-normal
 * distribution.
 *
 * \param state - Pointer to state to update
 * \param mean   - Mean of the related normal distribution
 * \param stddev - Standard deviation of the related normal distribution
 *
 * \return Log-normally distributed float with mean \p mean and standard deviation \p stddev
 */
QUALIFIERS float lwrand_log_normal(lwrandStateScrambledSobol64_t *state, float mean, float stddev)
{
    return expf(mean + (stddev * _lwrand_normal_icdf(lwrand(state))));
}

/**
 * \brief Return a log-normally distributed double from an XORWOW generator.
 *
 * Return a single normally distributed double derived from a normal
 * distribution with mean \p mean and standard deviation \p stddev 
 * from the XORWOW generator in \p state,
 * increment position of generator.
 *
 * The implementation uses a Box-Muller transform to generate two
 * normally distributed results, transforms them to log-normal distribution,
 * then returns them one at a time.
 * See ::lwrand_log_normal2_double() for a more efficient version that returns
 * both results at once.
 *
 * \param state - Pointer to state to update
 * \param mean   - Mean of the related normal distribution
 * \param stddev - Standard deviation of the related normal distribution
 *
 * \return Log-normally distributed double with mean \p mean and standard deviation \p stddev
 */

QUALIFIERS double lwrand_log_normal_double(lwrandStateXORWOW_t *state, double mean, double stddev)
{
    if(state->boxmuller_flag_double != EXTRA_FLAG_LOG_NORMAL) {
        unsigned int x0, x1, y0, y1;
        x0 = lwrand(state);
        x1 = lwrand(state);
        y0 = lwrand(state);
        y1 = lwrand(state);
        double2 v = _lwrand_box_muller_double(x0, x1, y0, y1);
        state->boxmuller_extra_double = exp(mean + (stddev * v.y));
        state->boxmuller_flag_double = EXTRA_FLAG_LOG_NORMAL;
        return exp(mean + (stddev * v.x));
    }
    state->boxmuller_flag_double = 0;
    return state->boxmuller_extra_double;
}

/**
 * \brief Return a log-normally distributed double from an Philox4_32_10 generator.
 *
 * Return a single normally distributed double derived from a normal
 * distribution with mean \p mean and standard deviation \p stddev 
 * from the Philox4_32_10 generator in \p state,
 * increment position of generator.
 *
 * The implementation uses a Box-Muller transform to generate two
 * normally distributed results, transforms them to log-normal distribution,
 * then returns them one at a time.
 * See ::lwrand_log_normal2_double() for a more efficient version that returns
 * both results at once.
 *
 * \param state - Pointer to state to update
 * \param mean   - Mean of the related normal distribution
 * \param stddev - Standard deviation of the related normal distribution
 *
 * \return Log-normally distributed double with mean \p mean and standard deviation \p stddev
 */

QUALIFIERS double lwrand_log_normal_double(lwrandStatePhilox4_32_10_t *state, double mean, double stddev)
{
    if(state->boxmuller_flag_double != EXTRA_FLAG_LOG_NORMAL) {
        uint4 _x;
        _x = lwrand4(state);
        double2 v = _lwrand_box_muller_double(_x.x, _x.y, _x.z, _x.w);
        state->boxmuller_extra_double = exp(mean + (stddev * v.y));
        state->boxmuller_flag_double = EXTRA_FLAG_LOG_NORMAL;
        return exp(mean + (stddev * v.x));
    }
    state->boxmuller_flag_double = 0;
    return state->boxmuller_extra_double;
}


/**
 * \brief Return two log-normally distributed doubles from an XORWOW generator.
 *
 * Return two log-normally distributed doubles derived from a normal
 * distribution with mean \p mean and standard deviation \p stddev 
 * from the XORWOW generator in \p state,
 * increment position of generator by two.
 *
 * The implementation uses a Box-Muller transform to generate two
 * normally distributed results, and transforms them to log-normal distribution,.
 *
 * \param state - Pointer to state to update
 * \param mean   - Mean of the related normal distribution
 * \param stddev - Standard deviation of the related normal distribution
 *
 * \return Log-normally distributed double2 where each element is from a
 * distribution with mean \p mean and standard deviation \p stddev
 */
QUALIFIERS double2 lwrand_log_normal2_double(lwrandStateXORWOW_t *state, double mean, double stddev)
{
    double2 v = lwrand_box_muller_double(state);
    v.x = exp(mean + (stddev * v.x));
    v.y = exp(mean + (stddev * v.y));
    return v;
}

/**
 * \brief Return two log-normally distributed doubles from an Philox4_32_10 generator.
 *
 * Return two log-normally distributed doubles derived from a normal
 * distribution with mean \p mean and standard deviation \p stddev 
 * from the Philox4_32_10 generator in \p state,
 * increment position of generator by four.
 *
 * The implementation uses a Box-Muller transform to generate two
 * normally distributed results, and transforms them to log-normal distribution,.
 *
 * \param state - Pointer to state to update
 * \param mean   - Mean of the related normal distribution
 * \param stddev - Standard deviation of the related normal distribution
 *
 * \return Log-normally distributed double4 where each element is from a
 * distribution with mean \p mean and standard deviation \p stddev
 */
QUALIFIERS double2 lwrand_log_normal2_double(lwrandStatePhilox4_32_10_t *state, double mean, double stddev)
{
    double2 v = lwrand_box_muller2_double(state);
    v.x = exp(mean + (stddev * v.x));
    v.y = exp(mean + (stddev * v.y));
    return v;
}
// nor part of API
QUALIFIERS double4 lwrand_log_normal4_double(lwrandStatePhilox4_32_10_t *state, double mean, double stddev)
{
    double4 v = lwrand_box_muller4_double(state);
    v.x = exp(mean + (stddev * v.x));
    v.y = exp(mean + (stddev * v.y));
    v.z = exp(mean + (stddev * v.z));
    v.w = exp(mean + (stddev * v.w));
    return v;
}

/**
 * \brief Return a log-normally distributed double from an MRG32k3a generator.
 *
 * Return a single normally distributed double derived from a normal
 * distribution with mean \p mean and standard deviation \p stddev 
 * from the MRG32k3a generator in \p state,
 * increment position of generator.
 *
 * The implementation uses a Box-Muller transform to generate two
 * normally distributed results, transforms them to log-normal distribution,
 * then returns them one at a time.
 * See ::lwrand_log_normal2_double() for a more efficient version that returns
 * both results at once.
 *
 * \param state - Pointer to state to update
 * \param mean   - Mean of the related normal distribution
 * \param stddev - Standard deviation of the related normal distribution
 *
 * \return Log-normally distributed double with mean \p mean and standard deviation \p stddev
 */
QUALIFIERS double lwrand_log_normal_double(lwrandStateMRG32k3a_t *state, double mean, double stddev)
{
    if(state->boxmuller_flag_double != EXTRA_FLAG_LOG_NORMAL) {
        double2 v = lwrand_box_muller_mrg_double(state);
        state->boxmuller_extra_double = exp(mean + (stddev * v.y));
        state->boxmuller_flag_double = EXTRA_FLAG_LOG_NORMAL;
        return exp(mean + (stddev * v.x));
    }
    state->boxmuller_flag_double = 0;
    return state->boxmuller_extra_double;
}

/**
 * \brief Return two log-normally distributed doubles from an MRG32k3a generator.
 *
 * Return two log-normally distributed doubles derived from a normal
 * distribution with mean \p mean and standard deviation \p stddev 
 * from the MRG32k3a generator in \p state,
 * increment position of generator by two.
 *
 * The implementation uses a Box-Muller transform to generate two
 * normally distributed results, and transforms them to log-normal distribution,.
 *
 * \param state - Pointer to state to update
 * \param mean   - Mean of the related normal distribution
 * \param stddev - Standard deviation of the related normal distribution
 *
 * \return Log-normally distributed double2 where each element is from a
 * distribution with mean \p mean and standard deviation \p stddev
 */
QUALIFIERS double2 lwrand_log_normal2_double(lwrandStateMRG32k3a_t *state, double mean, double stddev)
{
    double2 v = lwrand_box_muller_mrg_double(state);
    v.x = exp(mean + (stddev * v.x));
    v.y = exp(mean + (stddev * v.y));
    return v;
}

/**
 * \brief Return a log-normally distributed double from an MTGP32 generator.
 *
 * Return a single log-normally distributed double derived from a normal
 * distribution with mean \p mean and standard deviation \p stddev 
 * from the MTGP32 generator in \p state,
 * increment position of generator.
 *
 * The implementation uses the ilwerse cumulative distribution function
 * to generate normally distributed results, and transforms them into
 * log-normal distribution.
 *
 * \param state - Pointer to state to update
 * \param mean   - Mean of the related normal distribution
 * \param stddev - Standard deviation of the related normal distribution
 *
 * \return Log-normally distributed double with mean \p mean and standard deviation \p stddev
 */
QUALIFIERS double lwrand_log_normal_double(lwrandStateMtgp32_t *state, double mean, double stddev)
{
    return exp(mean + (stddev * _lwrand_normal_icdf_double(lwrand(state))));
}

/**
 * \brief Return a log-normally distributed double from a Sobol32 generator.
 *
 * Return a single log-normally distributed double derived from a normal
 * distribution with mean \p mean and standard deviation \p stddev 
 * from the Sobol32 generator in \p state,
 * increment position of generator by one.
 *
 * The implementation uses the ilwerse cumulative distribution function
 * to generate normally distributed results, and transforms them into
 * log-normal distribution.
 *
 * \param state - Pointer to state to update
 * \param mean   - Mean of the related normal distribution
 * \param stddev - Standard deviation of the related normal distribution
 *
 * \return Log-normally distributed double with mean \p mean and standard deviation \p stddev
 */
QUALIFIERS double lwrand_log_normal_double(lwrandStateSobol32_t *state, double mean, double stddev)
{
    return exp(mean + (stddev * _lwrand_normal_icdf_double(lwrand(state))));
}

/**
 * \brief Return a log-normally distributed double from a scrambled Sobol32 generator.
 *
 * Return a single log-normally distributed double derived from a normal
 * distribution with mean \p mean and standard deviation \p stddev 
 * from the scrambled Sobol32 generator in \p state,
 * increment position of generator by one.
 *
 * The implementation uses the ilwerse cumulative distribution function
 * to generate normally distributed results, and transforms them into
 * log-normal distribution.
 *
 * \param state - Pointer to state to update
 * \param mean   - Mean of the related normal distribution
 * \param stddev - Standard deviation of the related normal distribution
 *
 * \return Log-normally distributed double with mean \p mean and standard deviation \p stddev
 */
QUALIFIERS double lwrand_log_normal_double(lwrandStateScrambledSobol32_t *state, double mean, double stddev)
{
    return exp(mean + (stddev * _lwrand_normal_icdf_double(lwrand(state))));
}

/**
 * \brief Return a log-normally distributed double from a Sobol64 generator.
 *
 * Return a single normally distributed double derived from a normal
 * distribution with mean \p mean and standard deviation \p stddev 
 * from the Sobol64 generator in \p state,
 * increment position of generator by one.
 *
 * The implementation uses the ilwerse cumulative distribution function
 * to generate normally distributed results.
 *
 * \param state - Pointer to state to update
 * \param mean   - Mean of the related normal distribution
 * \param stddev - Standard deviation of the related normal distribution
 *
 * \return Log-normally distributed double with mean \p mean and standard deviation \p stddev
 */
QUALIFIERS double lwrand_log_normal_double(lwrandStateSobol64_t *state, double mean, double stddev)
{
    return exp(mean + (stddev * _lwrand_normal_icdf_double(lwrand(state))));
}

/**
 * \brief Return a log-normally distributed double from a scrambled Sobol64 generator.
 *
 * Return a single normally distributed double derived from a normal
 * distribution with mean \p mean and standard deviation \p stddev 
 * from the scrambled Sobol64 generator in \p state,
 * increment position of generator by one.
 *
 * The implementation uses the ilwerse cumulative distribution function
 * to generate normally distributed results.
 *
 * \param state - Pointer to state to update
 * \param mean   - Mean of the related normal distribution
 * \param stddev - Standard deviation of the related normal distribution
 *
 * \return Log-normally distributed double with mean \p mean and standard deviation \p stddev
 */
QUALIFIERS double lwrand_log_normal_double(lwrandStateScrambledSobol64_t *state, double mean, double stddev)
{
    return exp(mean + (stddev * _lwrand_normal_icdf_double(lwrand(state))));
}

#endif // !defined(LWRAND_LOGNORMAL_H_)
