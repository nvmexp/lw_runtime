/***************************************************************************************************
 * Copyright (c) 2019, LWPU CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted
 * provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright notice, this list of
 *       conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright notice, this list of
 *       conditions and the following disclaimer in the documentation and/or other materials
 *       provided with the distribution.
 *     * Neither the name of the LWPU CORPORATION nor the names of its contributors may be used
 *       to endorse or promote products derived from this software without specific prior written
 *       permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL LWPU CORPORATION BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROLWREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TOR (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

#pragma once

#include <xmma/xmma.h>

#ifdef max
#undef max
#endif

/**
* \defgroup XMMA_NUMERIC_TYPES Numeric data types
* This section describes numeric data types(half, float, int4/8/16)
* and supporting colwersion function between different types.
* Support both in device and host side colwersion
*/

// FIXME: Lwtlass has jetfire at top level include which causes confliction of jetfire
//   Before lwtlass remove jetfire include from top level don't use lwtlass type now
// #include <lwtlass/half.h>
// FIXME: use below alias to process transition to lwtlass type
namespace lwtlass {

#if !defined(LWTLASS_ENABLE_F16C)
// #include <lwtlass/half.h>
using half_t = uint16_t;
#endif
// #include <lwtlass/float_tf32_t.h>
using float_tf32_t = uint32_t;
using float_bf16_t = uint16_t;

// #include <lwtlass/integer_subbyte.h>
struct int2_t  {};
struct uint2_t {};
struct int4_t  {};
struct uint4_t {};

}

namespace xmma {

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T_> struct Type_traits {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<> struct Type_traits<float> {
    enum { SIZE_IN_BYTES = 4 };
    enum { SIZE_IN_BITS = 32 };
    enum { ALIGNMENT = 4 };
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<> struct Type_traits<int32_t> {
    enum { SIZE_IN_BYTES = 4 };
    enum { SIZE_IN_BITS = 32 };
    enum { ALIGNMENT = 4 };
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<> struct Type_traits<lwtlass::half_t> {
    enum { SIZE_IN_BYTES = 2 };
    enum { SIZE_IN_BITS = 16 };
    enum { ALIGNMENT = 4 };
};

////////////////////////////////////////////////////////////////////////////////////////////////////

enum class Rounding_mode {
    NONE,    // Different colwersion may need different default rounding mode
    RTZ,
    RD,
    RU,
    RTN,
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Dst_type_, typename Src_type_, Rounding_mode MODE = Rounding_mode::NONE >
XMMA_HOST_DEVICE Dst_type_ colwert(const Src_type_ src);

////////////////////////////////////////////////////////////////////////////////////////////////////

// TODO: need to support different rounding mode
template<>
XMMA_HOST_DEVICE float colwert<float, double>(const double src) {
#if defined(__LWDA_ARCH__)
    return __double2float_rz(src);
#else
    return (float)src;
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
XMMA_HOST_DEVICE double colwert<double, float>(const float src) {
    return (double)src;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// FIXME: this may not be necessary.
template<>
XMMA_HOST_DEVICE int32_t colwert<int32_t, double>(const double src) {
    return (int32_t)src;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
XMMA_HOST_DEVICE float colwert<float, float>(float src) {
    return src;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
XMMA_HOST_DEVICE double colwert<double, double>(double src) {
    return src;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
XMMA_HOST_DEVICE float colwert<float, lwtlass::half_t>(lwtlass::half_t src) {
    return __half2float(reinterpret_cast<const __half &>(src));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
XMMA_HOST_DEVICE lwtlass::half_t colwert<lwtlass::half_t, double>(double src) {
    __half value = __float2half_rn(colwert<float, double>(src));
    return reinterpret_cast<lwtlass::half_t &>(value);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
XMMA_HOST_DEVICE lwtlass::half_t colwert<lwtlass::half_t, float>(float src) {
    __half value = __float2half(src);
    return reinterpret_cast<lwtlass::half_t &>(value);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
XMMA_HOST_DEVICE lwtlass::half_t colwert<lwtlass::half_t, float, Rounding_mode::RTN>(float src) {
    __half value = __float2half_rn(src);
    return reinterpret_cast<lwtlass::half_t &>(value);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
XMMA_HOST_DEVICE lwtlass::half_t colwert<lwtlass::half_t, float, Rounding_mode::RTZ>(float src) {
    __half value = __float2half_rz(src);
    return reinterpret_cast<lwtlass::half_t &>(value);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
XMMA_HOST_DEVICE lwtlass::half_t colwert<lwtlass::half_t, float, Rounding_mode::RD>(float src) {
    __half value = __float2half_rd(src);
    return reinterpret_cast<lwtlass::half_t &>(value);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
XMMA_HOST_DEVICE lwtlass::half_t colwert<lwtlass::half_t, float, Rounding_mode::RU>(float src) {
    __half value = __float2half_ru(src);
    return reinterpret_cast<lwtlass::half_t &>(value);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
XMMA_HOST_DEVICE lwtlass::half_t colwert<lwtlass::half_t, lwtlass::half_t>(lwtlass::half_t src) {
    return src;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Numeric Colwersion: float <-> int32_t
//
////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Dst_, typename Src_, Rounding_mode mode>
inline Dst_ __safe_colwert(const Src_ value) {
#if !defined(__LWDACC_RTC__)
    if (static_cast<Dst_>(value) > std::numeric_limits<Dst_>::max()) {
        throw std::bad_cast();
    }
#endif

    return static_cast<Dst_>(value);
}

template<>
XMMA_HOST_DEVICE int32_t colwert<int32_t, float>(float src) {
#if defined(__LWDA_ARCH__)
    return __float2int_rz(src);
#else
    return __safe_colwert<int32_t, float, Rounding_mode::NONE>(src);
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
XMMA_HOST_DEVICE int32_t colwert<int32_t, float, Rounding_mode::RU>(float src) {
#if defined(__LWDA_ARCH__)
    return __float2int_ru(src);
#else
    return __safe_colwert<int32_t, float, Rounding_mode::RU>(src);
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
XMMA_HOST_DEVICE int32_t colwert<int32_t, float, Rounding_mode::RD>(float src) {
#if defined(__LWDA_ARCH__)
    return __float2int_rd(src);
#else
    return __safe_colwert<int32_t, float, Rounding_mode::RD>(src);
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
XMMA_HOST_DEVICE int32_t colwert<int32_t, float, Rounding_mode::RTZ>(float src) {
#if defined(__LWDA_ARCH__)
    return __float2int_rz(src);
#else
    return __safe_colwert<int32_t, float, Rounding_mode::RU>(src);
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
XMMA_HOST_DEVICE int32_t colwert<int32_t, float, Rounding_mode::RTN>(float src) {
#if defined(__LWDA_ARCH__)
    return __float2int_rn(src);
#else
    return __safe_colwert<int32_t, float, Rounding_mode::RTN>(src);
#endif
}

template<>
XMMA_HOST_DEVICE float colwert<float, int32_t>(int32_t src) {
#if defined(__LWDA_ARCH__)
    return __int2float_rz(src);
#else
    return __safe_colwert<float, int32_t, Rounding_mode::RTZ>(src);
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
XMMA_HOST_DEVICE float colwert<float, int32_t, Rounding_mode::RU>(int32_t src) {
#if defined(__LWDA_ARCH__)
    return __int2float_ru(src);
#else
    return __safe_colwert<float, int32_t, Rounding_mode::RU>(src);
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
XMMA_HOST_DEVICE float colwert<float, int32_t, Rounding_mode::RD>(int32_t src) {
#if defined(__LWDA_ARCH__)
    return __int2float_rd(src);
#else
    return __safe_colwert<float, int32_t, Rounding_mode::RD>(src);
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
XMMA_HOST_DEVICE float colwert<float, int32_t, Rounding_mode::RTZ>(int32_t src) {
#if defined(__LWDA_ARCH__)
    return __int2float_rz(src);
#else
    return __safe_colwert<float, int32_t, Rounding_mode::RTZ>(src);
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
XMMA_HOST_DEVICE float colwert<float, int32_t, Rounding_mode::RTN>(int32_t src) {
#if defined(__LWDA_ARCH__)
    return __int2float_rn(src);
#else
    return __safe_colwert<float, int32_t, Rounding_mode::RTN>(src);
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
XMMA_HOST_DEVICE half2 colwert<half2, float>(float src) {
    return __float2half2_rn(src);
}

} // namespace xmma
