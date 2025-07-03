
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


#if !defined(LWRAND_DISCRETE_H_)
#define LWRAND_DISCRETE_H_

/**
 * \defgroup DEVICE Device API
 *
 * @{
 */

#include "lwrand_mrg32k3a.h"
#include "lwrand_mtgp32_kernel.h"
#include <math.h>

#include "lwrand_philox4x32_x.h" 


template <typename T>
QUALIFIERS unsigned int _lwrand_discrete(T x, lwrandDiscreteDistribution_t discrete_distribution){
    if (discrete_distribution->method == LWRAND_M2){
        return _lwrand_M2_double(x, discrete_distribution->M2);
    }
    return (unsigned int)((discrete_distribution->stddev * _lwrand_normal_icdf_double(x)) + discrete_distribution->mean + 0.5);
}


template <typename STATE>
QUALIFIERS unsigned int lwrand__discrete(STATE state, lwrandDiscreteDistribution_t discrete_distribution){
    if (discrete_distribution->method == LWRAND_M2){
        return lwrand_M2_double(state, discrete_distribution->M2);
    }
    return (unsigned int)((discrete_distribution->stddev * lwrand_normal_double(state)) + discrete_distribution->mean + 0.5); //Round to nearest
}

template <typename STATE>
QUALIFIERS uint4 lwrand__discrete4(STATE state, lwrandDiscreteDistribution_t discrete_distribution){
    if (discrete_distribution->method == LWRAND_M2){
        return lwrand_M2_double4(state, discrete_distribution->M2);
    }
    double4 _res;
    uint4 result;
    _res = lwrand_normal4_double(state);
    result.x = (unsigned int)((discrete_distribution->stddev * _res.x) + discrete_distribution->mean + 0.5); //Round to nearest
    result.y = (unsigned int)((discrete_distribution->stddev * _res.y) + discrete_distribution->mean + 0.5); //Round to nearest
    result.z = (unsigned int)((discrete_distribution->stddev * _res.z) + discrete_distribution->mean + 0.5); //Round to nearest
    result.w = (unsigned int)((discrete_distribution->stddev * _res.w) + discrete_distribution->mean + 0.5); //Round to nearest
    return result;
}

/*
 * \brief Return a discrete distributed unsigned int from a XORWOW generator.
 *
 * Return a single discrete distributed unsigned int derived from a
 * distribution defined by \p discrete_distribution from the XORWOW generator in \p state,
 * increment position of generator by one.
 *
 * \param state - Pointer to state to update
 * \param discrete_distribution - ancillary structure for discrete distribution
 *
 * \return unsigned int distributed by distribution defined by \p discrete_distribution.
 */
QUALIFIERS unsigned int lwrand_discrete(lwrandStateXORWOW_t *state, lwrandDiscreteDistribution_t discrete_distribution)
{
    return lwrand__discrete(state, discrete_distribution);
}

/*
 * \brief Return a discrete distributed unsigned int from a Philox4_32_10 generator.
 *
 * Return a single discrete distributed unsigned int derived from a
 * distribution defined by \p discrete_distribution from the Philox4_32_10 generator in \p state,
 * increment position of generator by one.
 *
 * \param state - Pointer to state to update
 * \param discrete_distribution - ancillary structure for discrete distribution
 *
 * \return unsigned int distributed by distribution defined by \p discrete_distribution.
 */
QUALIFIERS unsigned int lwrand_discrete(lwrandStatePhilox4_32_10_t *state, lwrandDiscreteDistribution_t discrete_distribution)
{
    return lwrand__discrete(state, discrete_distribution);
}

/*
 * \brief Return four discrete distributed unsigned ints from a Philox4_32_10 generator.
 *
 * Return four single discrete distributed unsigned ints derived from a
 * distribution defined by \p discrete_distribution from the Philox4_32_10 generator in \p state,
 * increment position of generator by one.
 *
 * \param state - Pointer to state to update
 * \param discrete_distribution - ancillary structure for discrete distribution
 *
 * \return unsigned int distributed by distribution defined by \p discrete_distribution.
 */
QUALIFIERS uint4 lwrand_discrete4(lwrandStatePhilox4_32_10_t *state, lwrandDiscreteDistribution_t discrete_distribution)
{
    return lwrand__discrete4(state, discrete_distribution);
}
/*
 * \brief Return a discrete distributed unsigned int from a MRG32k3a generator.
 *
 * Re turn a single discrete distributed unsigned int derived from a
 * distribution defined by \p discrete_distribution from the MRG32k3a generator in \p state,
 * increment position of generator by one.
 *
 * \param state - Pointer to state to update
 * \param discrete_distribution - ancillary structure for discrete distribution
 *
 * \return unsigned int distributed by distribution defined by \p discrete_distribution.
 */
QUALIFIERS unsigned int lwrand_discrete(lwrandStateMRG32k3a_t *state, lwrandDiscreteDistribution_t discrete_distribution)
{
    return lwrand__discrete(state, discrete_distribution);
}

/*
 * \brief Return a discrete distributed unsigned int from a MTGP32 generator.
 *
 * Return a single discrete distributed unsigned int derived from a
 * distribution defined by \p discrete_distribution from the MTGP32 generator in \p state,
 * increment position of generator by one.
 *
 * \param state - Pointer to state to update
 * \param discrete_distribution - ancillary structure for discrete distribution
 *
 * \return unsigned int distributed by distribution defined by \p discrete_distribution.
 */
QUALIFIERS unsigned int lwrand_discrete(lwrandStateMtgp32_t *state, lwrandDiscreteDistribution_t discrete_distribution)
{
    return lwrand__discrete(state, discrete_distribution);
}

/*
 * \brief Return a discrete distributed unsigned int from a Sobol32 generator.
 *
 * Return a single discrete distributed unsigned int derived from a
 * distribution defined by \p discrete_distribution from the Sobol32 generator in \p state,
 * increment position of generator by one.
 *
 * \param state - Pointer to state to update
 * \param discrete_distribution - ancillary structure for discrete distribution
 *
 * \return unsigned int distributed by distribution defined by \p discrete_distribution.
 */
QUALIFIERS unsigned int lwrand_discrete(lwrandStateSobol32_t *state, lwrandDiscreteDistribution_t discrete_distribution)
{
    return lwrand__discrete(state, discrete_distribution);
}

/*
 * \brief Return a discrete distributed unsigned int from a scrambled Sobol32 generator.
 *
 * Return a single discrete distributed unsigned int derived from a
 * distribution defined by \p discrete_distribution from the scrambled Sobol32 generator in \p state,
 * increment position of generator by one.
 *
 * \param state - Pointer to state to update
 * \param discrete_distribution - ancillary structure for discrete distribution
 *
 * \return unsigned int distributed by distribution defined by \p discrete_distribution.
 */
QUALIFIERS unsigned int lwrand_discrete(lwrandStateScrambledSobol32_t *state, lwrandDiscreteDistribution_t discrete_distribution)
{
    return lwrand__discrete(state, discrete_distribution);
}

/*
 * \brief Return a discrete distributed unsigned int from a Sobol64 generator.
 *
 * Return a single discrete distributed unsigned int derived from a
 * distribution defined by \p discrete_distribution from the Sobol64 generator in \p state,
 * increment position of generator by one.
 *
 * \param state - Pointer to state to update
 * \param discrete_distribution - ancillary structure for discrete distribution
 *
 * \return unsigned int distributed by distribution defined by \p discrete_distribution.
 */
QUALIFIERS unsigned int lwrand_discrete(lwrandStateSobol64_t *state, lwrandDiscreteDistribution_t discrete_distribution)
{
    return lwrand__discrete(state, discrete_distribution);
}

/*
 * \brief Return a discrete distributed unsigned int from a scrambled Sobol64 generator.
 *
 * Return a single discrete distributed unsigned int derived from a
 * distribution defined by \p discrete_distribution from the scrambled Sobol64 generator in \p state,
 * increment position of generator by one.
 *
 * \param state - Pointer to state to update
 * \param discrete_distribution - ancillary structure for discrete distribution
 *
 * \return unsigned int distributed by distribution defined by \p discrete_distribution.
 */
QUALIFIERS unsigned int lwrand_discrete(lwrandStateScrambledSobol64_t *state, lwrandDiscreteDistribution_t discrete_distribution)
{
    return lwrand__discrete(state, discrete_distribution);
}

#endif // !defined(LWRAND_DISCRETE_H_)
