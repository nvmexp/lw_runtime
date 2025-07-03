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
#ifndef LWRAND_NORMAL_STATIC_H
#define LWRAND_NORMAL_STATIC_H

#define QUALIFIERS_STATIC __host__ __device__ __forceinline__

QUALIFIERS_STATIC float _lwrand_normal_icdf(unsigned int x)
{
#if __LWDA_ARCH__ > 0 || defined(HOST_HAVE_ERFCILWF)
    float s = LWRAND_SQRT2;
    // Mirror to avoid loss of precision
    if(x > 0x80000000UL) {
        x = 0xffffffffUL - x;
        s = -s;
    }
    float p = x * LWRAND_2POW32_ILW + (LWRAND_2POW32_ILW/2.0f);
    // p is in (0, 0.5], 2p is in (0, 1]
    return s * erfcilwf(2.0f * p);
#else
    x++;    //suppress warnings
    return 0.0f;
#endif
}

QUALIFIERS_STATIC float _lwrand_normal_icdf(unsigned long long x)
{
#if __LWDA_ARCH__ > 0 || defined(HOST_HAVE_ERFCILWF)
    unsigned int t = (unsigned int)(x >> 32);
    float s = LWRAND_SQRT2;
    // Mirror to avoid loss of precision
    if(t > 0x80000000UL) {
        t = 0xffffffffUL - t;
        s = -s;
    }
    float p = t * LWRAND_2POW32_ILW + (LWRAND_2POW32_ILW/2.0f);
    // p is in (0, 0.5], 2p is in (0, 1]
    return s * erfcilwf(2.0f * p);
#else
    x++;
    return 0.0f;
#endif
}

QUALIFIERS_STATIC double _lwrand_normal_icdf_double(unsigned int x)
{
#if __LWDA_ARCH__ > 0 || defined(HOST_HAVE_ERFCILWF)
    double s = LWRAND_SQRT2_DOUBLE;
    // Mirror to avoid loss of precision
    if(x > 0x80000000UL) {
        x = 0xffffffffUL - x;
        s = -s;
    }
    double p = x * LWRAND_2POW32_ILW_DOUBLE + (LWRAND_2POW32_ILW_DOUBLE/2.0);
    // p is in (0, 0.5], 2p is in (0, 1]
    return s * erfcilw(2.0 * p);
#else
    x++;
    return 0.0;
#endif
}

QUALIFIERS_STATIC double _lwrand_normal_icdf_double(unsigned long long x)
{
#if __LWDA_ARCH__ > 0 || defined(HOST_HAVE_ERFCILWF)
    double s = LWRAND_SQRT2_DOUBLE;
    x >>= 11;
    // Mirror to avoid loss of precision
    if(x > 0x10000000000000UL) {
        x = 0x1fffffffffffffUL - x;
        s = -s;
    }
    double p = x * LWRAND_2POW53_ILW_DOUBLE + (LWRAND_2POW53_ILW_DOUBLE/2.0);
    // p is in (0, 0.5], 2p is in (0, 1]
    return s * erfcilw(2.0 * p);
#else
    x++;
    return 0.0;
#endif
}
#undef QUALIFIERS_STATIC
#endif
