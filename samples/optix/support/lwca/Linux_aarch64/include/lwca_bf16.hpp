/*
* Copyright 1993-2020 LWPU Corporation.  All rights reserved.
*
* NOTICE TO LICENSEE:
*
* This source code and/or documentation ("Licensed Deliverables") are
* subject to LWPU intellectual property rights under U.S. and
* international Copyright laws.
*
* These Licensed Deliverables contained herein is PROPRIETARY and
* CONFIDENTIAL to LWPU and is being provided under the terms and
* conditions of a form of LWPU software license agreement by and
* between LWPU and Licensee ("License Agreement") or electronically
* accepted by Licensee.  Notwithstanding any terms or conditions to
* the contrary in the License Agreement, reproduction or disclosure
* of the Licensed Deliverables to any third party without the express
* written consent of LWPU is prohibited.
*
* NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
* LICENSE AGREEMENT, LWPU MAKES NO REPRESENTATION ABOUT THE
* SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
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
* C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
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

#if !defined(__LWDA_BF16_HPP__)
#define __LWDA_BF16_HPP__

#if !defined(__LWDA_BF16_H__)
#error "Do not include this file directly. Instead, include lwda_bf16.h."
#endif

#if !defined(_MSC_VER) && __cplusplus >= 201103L
#   define __CPP_VERSION_AT_LEAST_11_BF16
#elif _MSC_FULL_VER >= 190024210 && _MSVC_LANG >= 201103L
#   define __CPP_VERSION_AT_LEAST_11_BF16
#endif

/* C++11 header for std::move. 
 * In RTC mode, std::move is provided implicitly; don't include the header
 */
#if defined(__CPP_VERSION_AT_LEAST_11_BF16) && !defined(__LWDACC_RTC__)
#include <utility>
#endif /* defined(__CPP_VERSION_AT_LEAST_11_BF16) && !defined(__LWDACC_RTC__) */

/* C++ header for std::memcpy (used for type punning in host-side implementations).
 * When compiling as a LWCA source file memcpy is provided implicitly.
 * !defined(__LWDACC__) implies !defined(__LWDACC_RTC__).
 */
#if defined(__cplusplus) && !defined(__LWDACC__)
#include <cstring>
#endif /* defined(__cplusplus) && !defined(__LWDACC__) */


/* Set up function decorations */
#if defined(__LWDACC__)
#define __LWDA_BF16_DECL__ static __device__ __inline__
#define __LWDA_HOSTDEVICE_BF16_DECL__ static __host__ __device__ __inline__
#define __VECTOR_FUNCTIONS_DECL__ static __inline__ __host__ __device__
#define __LWDA_HOSTDEVICE__ __host__ __device__
#else /* !defined(__LWDACC__) */
#if defined(__GNUC__)
#define __LWDA_HOSTDEVICE_BF16_DECL__ static __attribute__ ((unused))
#else
#define __LWDA_HOSTDEVICE_BF16_DECL__ static
#endif /* defined(__GNUC__) */
#define __LWDA_HOSTDEVICE__
#endif /* defined(__LWDACC_) */

/* Set up structure-alignment attribute */
#if defined(__LWDACC__)
#define __LWDA_ALIGN__(align) __align__(align)
#else
/* Define alignment macro based on compiler type (cannot assume C11 "_Alignas" is available) */
#if defined(__CPP_VERSION_AT_LEAST_11_BF16)
#define __LWDA_ALIGN__(n) alignas(n)    /* C++11 kindly gives us a keyword for this */
#else /* defined(__CPP_VERSION_AT_LEAST_11_BF16)*/
#if defined(__GNUC__)
#define __LWDA_ALIGN__(n) __attribute__ ((aligned(n)))
#elif defined(_MSC_VER)
#define __LWDA_ALIGN__(n) __declspec(align(n))
#else
#define __LWDA_ALIGN__(n)
#endif /* defined(__GNUC__) */
#endif /* defined(__CPP_VERSION_AT_LEAST_11_BF16) */
#endif /* defined(__LWDACC__) */

/* Macros to allow lw_bfloat16 & lw_bfloat162 to be used by inline assembly */
#define __BFLOAT16_TO_US(var) *(reinterpret_cast<unsigned short *>(&(var)))
#define __BFLOAT16_TO_LWS(var) *(reinterpret_cast<const unsigned short *>(&(var)))
#define __BFLOAT162_TO_UI(var) *(reinterpret_cast<unsigned int *>(&(var)))
#define __BFLOAT162_TO_LWI(var) *(reinterpret_cast<const unsigned int *>(&(var)))

/**
* Types which allow static initialization of "lw_bfloat16" and "lw_bfloat162" until
* these become an actual builtin. Note this initialization is as a
* bitfield representation of "lw_bfloat16", and not a colwersion from short->lw_bfloat16.
* Such a representation will be deprecated in a future version of LWCA. 
* (Note these are visible to non-lwcc compilers, including C-only compilation)
*/
typedef struct __LWDA_ALIGN__(2) {
    unsigned short x;
} __lw_bfloat16_raw;

typedef struct __LWDA_ALIGN__(4) {
    unsigned short x;
    unsigned short y;
} __lw_bfloat162_raw;

/* All other definitions in this file are only visible to C++ compilers */
#if defined(__cplusplus)

/* Hide GCC member initialization list warnings because of host/device in-function init requirement */
#if defined(__GNUC__)
#if __GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 6)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#pragma GCC diagnostic ignored "-Weffc++"
#endif /* __GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 6) */
#endif /* defined(__GNUC__) */

/* class' : multiple assignment operators specified
   The class has multiple assignment operators of a single type. This warning is informational */
#if defined(_MSC_VER) && _MSC_VER >= 1500
#pragma warning( push )
#pragma warning( disable:4522 )
#endif /* defined(__GNUC__) */

struct __LWDA_ALIGN__(2) __lw_bfloat16 {
protected:
    unsigned short __x;

public:
#if defined(__CPP_VERSION_AT_LEAST_11_BF16)
    __lw_bfloat16() = default;
#else
    __LWDA_HOSTDEVICE__ __lw_bfloat16() { }
#endif /* defined(__CPP_VERSION_AT_LEAST_11_BF16) */

    /* Colwert to/from __lw_bfloat16_raw */
    __LWDA_HOSTDEVICE__ __lw_bfloat16(const __lw_bfloat16_raw &hr) : __x(hr.x) { }
    __LWDA_HOSTDEVICE__ __lw_bfloat16 &operator=(const __lw_bfloat16_raw &hr) { __x = hr.x; return *this; }
    __LWDA_HOSTDEVICE__ volatile __lw_bfloat16 &operator=(const __lw_bfloat16_raw &hr) volatile { __x = hr.x; return *this; }
    __LWDA_HOSTDEVICE__ volatile __lw_bfloat16 &operator=(const volatile __lw_bfloat16_raw &hr) volatile { __x = hr.x; return *this; }
    __LWDA_HOSTDEVICE__ operator __lw_bfloat16_raw() const { __lw_bfloat16_raw ret; ret.x = __x; return ret; }
    __LWDA_HOSTDEVICE__ operator __lw_bfloat16_raw() const volatile { __lw_bfloat16_raw ret; ret.x = __x; return ret; }

#if !defined(__LWDA_NO_BFLOAT16_COLWERSIONS__)
    /* Construct from float/double */
    __LWDA_HOSTDEVICE__ __lw_bfloat16(const float f) { __x = __float2bfloat16(f).__x;  }
    __LWDA_HOSTDEVICE__ __lw_bfloat16(const double f) { __x = __double2bfloat16(f).__x;  }

    __LWDA_HOSTDEVICE__ operator float() const { return __bfloat162float(*this); }
    __LWDA_HOSTDEVICE__ __lw_bfloat16 &operator=(const float f) { __x = __float2bfloat16(f).__x; return *this; }

    /* We omit "cast to double" operator, so as to not be ambiguous about up-cast */
    __LWDA_HOSTDEVICE__ __lw_bfloat16 &operator=(const double f) { __x = __double2bfloat16(f).__x; return *this; }

/* Member functions only available to lwcc compilation so far */
#if defined(__LWDACC__) && (__LWDA_ARCH__ >= 800 || !defined(__LWDA_ARCH__))
    /* Allow automatic construction from types supported natively in hardware */
    /* Note we do avoid constructor init-list because of special host/device compilation rules */
    __LWDA_HOSTDEVICE__ __lw_bfloat16(short val) { __x = __short2bfloat16_rn(val).__x;  }
    __LWDA_HOSTDEVICE__ __lw_bfloat16(unsigned short val) { __x = __ushort2bfloat16_rn(val).__x;  }
    __LWDA_HOSTDEVICE__ __lw_bfloat16(int val) { __x = __int2bfloat16_rn(val).__x;  }
    __LWDA_HOSTDEVICE__ __lw_bfloat16(unsigned int val) { __x = __uint2bfloat16_rn(val).__x;  }
    __LWDA_HOSTDEVICE__ __lw_bfloat16(long long val) { __x = __ll2bfloat16_rn(val).__x;  }
    __LWDA_HOSTDEVICE__ __lw_bfloat16(unsigned long long val) { __x = __ull2bfloat16_rn(val).__x; }

    /* Allow automatic casts to supported builtin types, matching all that are permitted with float */
    __LWDA_HOSTDEVICE__ operator short() const { return __bfloat162short_rz(*this); }
    __LWDA_HOSTDEVICE__ __lw_bfloat16 &operator=(short val) { __x = __short2bfloat16_rn(val).__x; return *this; }

    __LWDA_HOSTDEVICE__ operator unsigned short() const { return __bfloat162ushort_rz(*this); }
    __LWDA_HOSTDEVICE__ __lw_bfloat16 &operator=(unsigned short val) { __x = __ushort2bfloat16_rn(val).__x; return *this; }

    __LWDA_HOSTDEVICE__ operator int() const { return __bfloat162int_rz(*this); }
    __LWDA_HOSTDEVICE__ __lw_bfloat16 &operator=(int val) { __x = __int2bfloat16_rn(val).__x; return *this; }

    __LWDA_HOSTDEVICE__ operator unsigned int() const { return __bfloat162uint_rz(*this); }
    __LWDA_HOSTDEVICE__ __lw_bfloat16 &operator=(unsigned int val) { __x = __uint2bfloat16_rn(val).__x; return *this; }

    __LWDA_HOSTDEVICE__ operator long long() const { return __bfloat162ll_rz(*this); }
    __LWDA_HOSTDEVICE__ __lw_bfloat16 &operator=(long long val) { __x = __ll2bfloat16_rn(val).__x; return *this; }

    __LWDA_HOSTDEVICE__ operator unsigned long long() const { return __bfloat162ull_rz(*this); }
    __LWDA_HOSTDEVICE__ __lw_bfloat16 &operator=(unsigned long long val) { __x = __ull2bfloat16_rn(val).__x; return *this; }

    /* Boolean colwersion - note both 0 and -0 must return false */
    __LWDA_HOSTDEVICE__ operator bool() const { return (__x & 0x7FFF) != 0; }
#endif /* defined(__LWDACC__) && (__LWDA_ARCH__ >= 800 || !defined(__LWDA_ARCH__)) */
#endif /* !defined(__LWDA_NO_BFLOAT16_COLWERSIONS__) */
};

/* Global-space operator functions are only available to lwcc compilation */
#if defined(__LWDACC__)

#if __LWDA_ARCH__ >= 800 || !defined(__LWDA_ARCH__)
#if !defined(__LWDA_NO_BFLOAT16_OPERATORS__)
/* Some basic arithmetic operations expected of a builtin */
__device__ __forceinline__ __lw_bfloat16 operator+(const __lw_bfloat16 &lh, const __lw_bfloat16 &rh) { return __hadd(lh, rh); }
__device__ __forceinline__ __lw_bfloat16 operator-(const __lw_bfloat16 &lh, const __lw_bfloat16 &rh) { return __hsub(lh, rh); }
__device__ __forceinline__ __lw_bfloat16 operator*(const __lw_bfloat16 &lh, const __lw_bfloat16 &rh) { return __hmul(lh, rh); }
__device__ __forceinline__ __lw_bfloat16 operator/(const __lw_bfloat16 &lh, const __lw_bfloat16 &rh) { return __hdiv(lh, rh); }

__device__ __forceinline__ __lw_bfloat16 &operator+=(__lw_bfloat16 &lh, const __lw_bfloat16 &rh) { lh = __hadd(lh, rh); return lh; }
__device__ __forceinline__ __lw_bfloat16 &operator-=(__lw_bfloat16 &lh, const __lw_bfloat16 &rh) { lh = __hsub(lh, rh); return lh; }
__device__ __forceinline__ __lw_bfloat16 &operator*=(__lw_bfloat16 &lh, const __lw_bfloat16 &rh) { lh = __hmul(lh, rh); return lh; }
__device__ __forceinline__ __lw_bfloat16 &operator/=(__lw_bfloat16 &lh, const __lw_bfloat16 &rh) { lh = __hdiv(lh, rh); return lh; }

/* Note for increment and decrement we use the raw value 0x3F80 equating to lw_bfloat16(1.0f), to avoid the extra colwersion */
__device__ __forceinline__ __lw_bfloat16 &operator++(__lw_bfloat16 &h)      { __lw_bfloat16_raw one; one.x = 0x3F80; h += one; return h; }
__device__ __forceinline__ __lw_bfloat16 &operator--(__lw_bfloat16 &h)      { __lw_bfloat16_raw one; one.x = 0x3F80; h -= one; return h; }
__device__ __forceinline__ __lw_bfloat16  operator++(__lw_bfloat16 &h, const int ignored)
{
    // ignored on purpose. Parameter only needed to distinguish the function declaration from other types of operators.
    static_cast<void>(ignored);

    const __lw_bfloat16 ret = h;
    __lw_bfloat16_raw one;
    one.x = 0x3F80;
    h += one;
    return ret;
}
__device__ __forceinline__ __lw_bfloat16  operator--(__lw_bfloat16 &h, const int ignored)
{
    // ignored on purpose. Parameter only needed to distinguish the function declaration from other types of operators.
    static_cast<void>(ignored);

    const __lw_bfloat16 ret = h;
    __lw_bfloat16_raw one;
    one.x = 0x3F80;
    h -= one;
    return ret;
}
/* Unary plus and ilwerse operators */
__device__ __forceinline__ __lw_bfloat16 operator+(const __lw_bfloat16 &h) { return h; }
__device__ __forceinline__ __lw_bfloat16 operator-(const __lw_bfloat16 &h) { return __hneg(h); }

/* Some basic comparison operations to make it look like a builtin */
__device__ __forceinline__ bool operator==(const __lw_bfloat16 &lh, const __lw_bfloat16 &rh) { return __heq(lh, rh); }
__device__ __forceinline__ bool operator!=(const __lw_bfloat16 &lh, const __lw_bfloat16 &rh) { return __hneu(lh, rh); }
__device__ __forceinline__ bool operator> (const __lw_bfloat16 &lh, const __lw_bfloat16 &rh) { return __hgt(lh, rh); }
__device__ __forceinline__ bool operator< (const __lw_bfloat16 &lh, const __lw_bfloat16 &rh) { return __hlt(lh, rh); }
__device__ __forceinline__ bool operator>=(const __lw_bfloat16 &lh, const __lw_bfloat16 &rh) { return __hge(lh, rh); }
__device__ __forceinline__ bool operator<=(const __lw_bfloat16 &lh, const __lw_bfloat16 &rh) { return __hle(lh, rh); }
#endif /* !defined(__LWDA_NO_BFLOAT16_OPERATORS__) */
#endif /* __LWDA_ARCH__ >= 800 || !defined(__LWDA_ARCH__) */
#endif /* defined(__LWDACC__) */

/* __lw_bfloat162 is visible to non-lwcc host compilers */
struct __LWDA_ALIGN__(4) __lw_bfloat162 {
    __lw_bfloat16 x;
    __lw_bfloat16 y;

    // All construct/copy/assign/move
public:
#if defined(__CPP_VERSION_AT_LEAST_11_BF16)
    __lw_bfloat162() = default;
    __LWDA_HOSTDEVICE__ __lw_bfloat162(__lw_bfloat162 &&src) { __BFLOAT162_TO_UI(*this) = std::move(__BFLOAT162_TO_LWI(src)); }
    __LWDA_HOSTDEVICE__ __lw_bfloat162 &operator=(__lw_bfloat162 &&src) { __BFLOAT162_TO_UI(*this) = std::move(__BFLOAT162_TO_LWI(src)); return *this; }
#else
    __LWDA_HOSTDEVICE__ __lw_bfloat162() { }
#endif /* defined(__CPP_VERSION_AT_LEAST_11_BF16) */
    __LWDA_HOSTDEVICE__ __lw_bfloat162(const __lw_bfloat16 &a, const __lw_bfloat16 &b) : x(a), y(b) { }
    __LWDA_HOSTDEVICE__ __lw_bfloat162(const __lw_bfloat162 &src) { __BFLOAT162_TO_UI(*this) = __BFLOAT162_TO_LWI(src); }
    __LWDA_HOSTDEVICE__ __lw_bfloat162 &operator=(const __lw_bfloat162 &src) { __BFLOAT162_TO_UI(*this) = __BFLOAT162_TO_LWI(src); return *this; }

    /* Colwert to/from __lw_bfloat162_raw */
    __LWDA_HOSTDEVICE__ __lw_bfloat162(const __lw_bfloat162_raw &h2r ) { __BFLOAT162_TO_UI(*this) = __BFLOAT162_TO_LWI(h2r); }
    __LWDA_HOSTDEVICE__ __lw_bfloat162 &operator=(const __lw_bfloat162_raw &h2r) { __BFLOAT162_TO_UI(*this) = __BFLOAT162_TO_LWI(h2r); return *this; }
    __LWDA_HOSTDEVICE__ operator __lw_bfloat162_raw() const { __lw_bfloat162_raw ret; ret.x = 0U; ret.y = 0U; __BFLOAT162_TO_UI(ret) = __BFLOAT162_TO_LWI(*this); return ret; }
};

/* Global-space operator functions are only available to lwcc compilation */
#if defined(__LWDACC__)

#if (__LWDA_ARCH__ >= 800 || !defined(__LWDA_ARCH__)) && !defined(__LWDA_NO_BFLOAT162_OPERATORS__)

__device__ __forceinline__ __lw_bfloat162 operator+(const __lw_bfloat162 &lh, const __lw_bfloat162 &rh) { return __hadd2(lh, rh); }
__device__ __forceinline__ __lw_bfloat162 operator-(const __lw_bfloat162 &lh, const __lw_bfloat162 &rh) { return __hsub2(lh, rh); }
__device__ __forceinline__ __lw_bfloat162 operator*(const __lw_bfloat162 &lh, const __lw_bfloat162 &rh) { return __hmul2(lh, rh); }
__device__ __forceinline__ __lw_bfloat162 operator/(const __lw_bfloat162 &lh, const __lw_bfloat162 &rh) { return __h2div(lh, rh); }

__device__ __forceinline__ __lw_bfloat162& operator+=(__lw_bfloat162 &lh, const __lw_bfloat162 &rh) { lh = __hadd2(lh, rh); return lh; }
__device__ __forceinline__ __lw_bfloat162& operator-=(__lw_bfloat162 &lh, const __lw_bfloat162 &rh) { lh = __hsub2(lh, rh); return lh; }
__device__ __forceinline__ __lw_bfloat162& operator*=(__lw_bfloat162 &lh, const __lw_bfloat162 &rh) { lh = __hmul2(lh, rh); return lh; }
__device__ __forceinline__ __lw_bfloat162& operator/=(__lw_bfloat162 &lh, const __lw_bfloat162 &rh) { lh = __h2div(lh, rh); return lh; }

__device__ __forceinline__ __lw_bfloat162 &operator++(__lw_bfloat162 &h)      { __lw_bfloat162_raw one; one.x = 0x3F80; one.y = 0x3F80; h = __hadd2(h, one); return h; }
__device__ __forceinline__ __lw_bfloat162 &operator--(__lw_bfloat162 &h)      { __lw_bfloat162_raw one; one.x = 0x3F80; one.y = 0x3F80; h = __hsub2(h, one); return h; }
__device__ __forceinline__ __lw_bfloat162  operator++(__lw_bfloat162 &h, const int ignored)
{
    // ignored on purpose. Parameter only needed to distinguish the function declaration from other types of operators.
    static_cast<void>(ignored);

    const __lw_bfloat162 ret = h;
    __lw_bfloat162_raw one;
    one.x = 0x3F80;
    one.y = 0x3F80;
    h = __hadd2(h, one);
    return ret;
}
__device__ __forceinline__ __lw_bfloat162  operator--(__lw_bfloat162 &h, const int ignored)
{
    // ignored on purpose. Parameter only needed to distinguish the function declaration from other types of operators.
    static_cast<void>(ignored);

    const __lw_bfloat162 ret = h;
    __lw_bfloat162_raw one;
    one.x = 0x3F80;
    one.y = 0x3F80;
    h = __hsub2(h, one);
    return ret;
}
__device__ __forceinline__ __lw_bfloat162 operator+(const __lw_bfloat162 &h) { return h; }
__device__ __forceinline__ __lw_bfloat162 operator-(const __lw_bfloat162 &h) { return __hneg2(h); }

__device__ __forceinline__ bool operator==(const __lw_bfloat162 &lh, const __lw_bfloat162 &rh) { return __hbeq2(lh, rh); }
__device__ __forceinline__ bool operator!=(const __lw_bfloat162 &lh, const __lw_bfloat162 &rh) { return __hbneu2(lh, rh); }
__device__ __forceinline__ bool operator>(const __lw_bfloat162 &lh, const __lw_bfloat162 &rh) { return __hbgt2(lh, rh); }
__device__ __forceinline__ bool operator<(const __lw_bfloat162 &lh, const __lw_bfloat162 &rh) { return __hblt2(lh, rh); }
__device__ __forceinline__ bool operator>=(const __lw_bfloat162 &lh, const __lw_bfloat162 &rh) { return __hbge2(lh, rh); }
__device__ __forceinline__ bool operator<=(const __lw_bfloat162 &lh, const __lw_bfloat162 &rh) { return __hble2(lh, rh); }

#endif /* __LWDA_ARCH__ >= 800 || !defined(__LWDA_ARCH__) */
#endif /* defined(__LWDACC__) */

/* Restore warning for multiple assignment operators */
#if defined(_MSC_VER) && _MSC_VER >= 1500
#pragma warning( pop )
#endif /* defined(_MSC_VER) && _MSC_VER >= 1500 */

/* Restore -Weffc++ warnings from here on */
#if defined(__GNUC__)
#if __GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 6)
#pragma GCC diagnostic pop
#endif /* __GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 6) */
#endif /* defined(__GNUC__) */

#undef __LWDA_HOSTDEVICE__
#undef __LWDA_ALIGN__

__LWDA_HOSTDEVICE_BF16_DECL__ unsigned short __internal_float2bfloat16(const float f, unsigned int &sign, unsigned int &remainder)
{
    unsigned int x;

#if defined(__LWDA_ARCH__)
    x = __float_as_uint(f);
#elif defined(__LWDACC__)
    (void)memcpy(&x, &f, sizeof(f));
#else
    (void)std::memcpy(&x, &f, sizeof(f));
#endif

    if ((x & 0x7fffffffU) > 0x7f800000U) {
        sign = 0U;
        remainder = 0U;
        return static_cast<unsigned short>(0x7fffU);
    }
    sign = x >> 31U;
    remainder = x << 16U;
    return static_cast<unsigned short>(x >> 16U);
}

__LWDA_HOSTDEVICE_BF16_DECL__ __lw_bfloat16 __double2bfloat16(const double x)
{
    float f = static_cast<float>(x);
    const double d = static_cast<double>(f);
    unsigned int u;

#if defined(__LWDA_ARCH__)
    u = __float_as_uint(f);
#elif defined(__LWDACC__)
    (void)memcpy(&u, &f, sizeof(f));
#else
    (void)std::memcpy(&u, &f, sizeof(f));
#endif

    if ((x > 0.0) && (d > x)) {
        u--;
    }
    if ((x < 0.0) && (d < x)) {
        u--;
    }
    if ((d != x) && (x == x)) {
        u |= 1U;
    }

#if defined(__LWDA_ARCH__)
    f = __int_as_float(static_cast<int>(u));
#elif defined(__LWDACC__)
    (void)memcpy(&f, &u, sizeof(f));
#else
    (void)std::memcpy(&f, &u, sizeof(f));
#endif

    return __float2bfloat16(f);
}

__LWDA_HOSTDEVICE_BF16_DECL__ __lw_bfloat16 __float2bfloat16(const float a)
{
    __lw_bfloat16 val;
#if __LWDA_ARCH__ >= 800
    asm("{  cvt.rn.bf16.f32 %0, %1;}\n" : "=h"(__BFLOAT16_TO_US(val)) : "f"(a));
#else
    __lw_bfloat16_raw r;
    unsigned int sign = 0U;
    unsigned int remainder = 0U;
    r.x = __internal_float2bfloat16(a, sign, remainder);
    if ((remainder > 0x80000000U) || ((remainder == 0x80000000U) && ((r.x & 0x1U) != 0U))) {
        r.x++;
    }
    val = r;
#endif
    return val;
}
__LWDA_HOSTDEVICE_BF16_DECL__ __lw_bfloat16 __float2bfloat16_rn(const float a)
{
    __lw_bfloat16 val;
#if __LWDA_ARCH__ >= 800
    asm("{  cvt.rn.bf16.f32 %0, %1;}\n" : "=h"(__BFLOAT16_TO_US(val)) : "f"(a));
#else
    __lw_bfloat16_raw r;
    unsigned int sign = 0U;
    unsigned int remainder = 0U;
    r.x = __internal_float2bfloat16(a, sign, remainder);
    if ((remainder > 0x80000000U) || ((remainder == 0x80000000U) && ((r.x & 0x1U) != 0U))) {
        r.x++;
    }
    val = r;
#endif
    return val;
}
__LWDA_HOSTDEVICE_BF16_DECL__ __lw_bfloat16 __float2bfloat16_rz(const float a)
{
    __lw_bfloat16 val;
#if __LWDA_ARCH__ >= 800
    asm("{  cvt.rz.bf16.f32 %0, %1;}\n" : "=h"(__BFLOAT16_TO_US(val)) : "f"(a));
#else
    __lw_bfloat16_raw r;
    unsigned int sign = 0U;
    unsigned int remainder = 0U;
    r.x = __internal_float2bfloat16(a, sign, remainder);
    val = r;
#endif
    return val;
}
__LWDA_HOSTDEVICE_BF16_DECL__ __lw_bfloat16 __float2bfloat16_rd(const float a)
{
    __lw_bfloat16 val;
    __lw_bfloat16_raw r;
    unsigned int sign = 0U;
    unsigned int remainder = 0U;
    r.x = __internal_float2bfloat16(a, sign, remainder);
    if ((remainder != 0U) && (sign != 0U)) {
        r.x++;
    }
    val = r;
    return val;
}
__LWDA_HOSTDEVICE_BF16_DECL__ __lw_bfloat16 __float2bfloat16_ru(const float a)
{
    __lw_bfloat16 val;
    __lw_bfloat16_raw r;
    unsigned int sign = 0U;
    unsigned int remainder = 0U;
    r.x = __internal_float2bfloat16(a, sign, remainder);
    if ((remainder != 0U) && (sign == 0U)) {
        r.x++;
    }
    val = r;
    return val;
}
__LWDA_HOSTDEVICE_BF16_DECL__ __lw_bfloat162 __float2bfloat162_rn(const float a)
{
    __lw_bfloat162 val;
#if __LWDA_ARCH__ >= 800
    asm("{.reg .b16 low;\n"
        "  cvt.rn.bf16.f32 low, %1;\n"
        "  mov.b32 %0, {low,low};}\n" : "=r"(__BFLOAT162_TO_UI(val)) : "f"(a));
#else
    val = __lw_bfloat162(__float2bfloat16_rn(a), __float2bfloat16_rn(a));
#endif
    return val;
}
__LWDA_HOSTDEVICE_BF16_DECL__ __lw_bfloat162 __floats2bfloat162_rn(const float a, const float b)
{
    __lw_bfloat162 val;
#if __LWDA_ARCH__ >= 800
    asm("{.reg .b16 low,high;\n"
        "  cvt.rn.bf16.f32 low, %1;\n"
        "  cvt.rn.bf16.f32 high, %2;\n"
        "  mov.b32 %0, {low,high};}\n" : "=r"(__BFLOAT162_TO_UI(val)) : "f"(a), "f"(b));
#else
    val = __lw_bfloat162(__float2bfloat16_rn(a), __float2bfloat16_rn(b));
#endif
    return val;
}

__LWDA_HOSTDEVICE_BF16_DECL__ float __internal_bfloat162float(const unsigned short h)
{
    float f;
#if defined(__LWDA_ARCH__)
    asm("{ mov.b32 %0, {0,%1};}\n" : "=f"(f) : "h"(h));
#else
    unsigned int u = static_cast<unsigned int>(h) << 16;
#if defined(__LWDACC__)
    (void)memcpy(&f, &u, sizeof(f));
#else
    (void)std::memcpy(&f, &u, sizeof(f));
#endif
#endif
    return f;
}

__LWDA_HOSTDEVICE_BF16_DECL__ float __bfloat162float(const __lw_bfloat16 a)
{
    return __internal_bfloat162float(static_cast<__lw_bfloat16_raw>(a).x);
}
__LWDA_HOSTDEVICE_BF16_DECL__ float __low2float(const __lw_bfloat162 a)
{
    return __internal_bfloat162float(static_cast<__lw_bfloat162_raw>(a).x);
}

__LWDA_HOSTDEVICE_BF16_DECL__ float __high2float(const __lw_bfloat162 a)
{
    return __internal_bfloat162float(static_cast<__lw_bfloat162_raw>(a).y);
}

#if defined(__LWDACC__) && (__LWDA_ARCH__ >= 800 || !defined(__LWDA_ARCH__))

/* LWCA vector-types compatible vector creation function (note returns __lw_bfloat162, not lw_bfloat162) */
__VECTOR_FUNCTIONS_DECL__ __lw_bfloat162 make_bfloat162(const __lw_bfloat16 x, const __lw_bfloat16 y)
{
    __lw_bfloat162 t; t.x = x; t.y = y; return t;
}
#undef __VECTOR_FUNCTIONS_DECL__


/* Definitions of intrinsics */
__LWDA_HOSTDEVICE_BF16_DECL__ __lw_bfloat162 __float22bfloat162_rn(const float2 a)
{
    __lw_bfloat162 val = __floats2bfloat162_rn(a.x, a.y);
    return val;
}
__LWDA_HOSTDEVICE_BF16_DECL__ float2 __bfloat1622float2(const __lw_bfloat162 a)
{
    float hi_float;
    float lo_float;
    lo_float = __internal_bfloat162float(((__lw_bfloat162_raw)a).x);
    hi_float = __internal_bfloat162float(((__lw_bfloat162_raw)a).y);
    return make_float2(lo_float, hi_float);
}
__LWDA_BF16_DECL__ int __bfloat162int_rn(const __lw_bfloat16 h)
{
    return __float2int_rn(__bfloat162float(h));
}
__LWDA_HOSTDEVICE_BF16_DECL__ int __bfloat162int_rz(const __lw_bfloat16 h)
{
    const float f = __bfloat162float(h);
    int   i;
    i = static_cast<int>(f);
#if !(defined __LWDA_ARCH__)
    const int max_val = (int)0x7fffffffU;
    const int min_val = (int)0x80000000U;
    // saturation fixup
    if (f != f) {
        // NaN
        i = 0;
    } else if (f >= static_cast<float>(max_val)) {
        // saturate maximum
        i = max_val;
    } else if (f < static_cast<float>(min_val)) {
        // saturate minimum
        i = min_val;
    }
#endif
    return i;
}
__LWDA_BF16_DECL__ int __bfloat162int_rd(const __lw_bfloat16 h)
{
    return __float2int_rd(__bfloat162float(h));
}
__LWDA_BF16_DECL__ int __bfloat162int_ru(const __lw_bfloat16 h)
{
    return __float2int_ru(__bfloat162float(h));
}
__LWDA_HOSTDEVICE_BF16_DECL__ __lw_bfloat16 __int2bfloat16_rn(const int i)
{
#if (defined __LWDA_ARCH__)
    const float ru = __int2float_ru(i);
    const float rd = __int2float_rd(i);
    float rz = __int2float_rz(i);
    if (ru != rd) {
        rz = __uint_as_float(__float_as_uint(rz) | 1U);
    }
    return __float2bfloat16_rn(rz);
#else
    const double d = static_cast<double>(i);
    return __double2bfloat16(d);
#endif
}
__LWDA_BF16_DECL__ __lw_bfloat16 __int2bfloat16_rz(const int i)
{
    return __float2bfloat16_rz(__int2float_rz(i));
}
__LWDA_BF16_DECL__ __lw_bfloat16 __int2bfloat16_rd(const int i)
{
    return __float2bfloat16_rd(__int2float_rd(i));
}
__LWDA_BF16_DECL__ __lw_bfloat16 __int2bfloat16_ru(const int i)
{
    return __float2bfloat16_ru(__int2float_ru(i));
}

__LWDA_BF16_DECL__ short int __bfloat162short_rn(const __lw_bfloat16 h)
{
   short int val;
   asm("{ .reg.f32 f;\n"
       "  mov.b32 f, {0,%1};\n"
       "  cvt.rni.s16.f32 %0,f;\n}"
        :"=h"(__BFLOAT16_TO_US(val)) : "h"(__BFLOAT16_TO_LWS(h)));
   return val;
}
__LWDA_HOSTDEVICE_BF16_DECL__ short int __bfloat162short_rz(const __lw_bfloat16 h)
{
   short int val;
#if (defined __LWDA_ARCH__)
   asm("{ .reg.f32 f;\n"
       "  mov.b32 f, {0,%1};\n"
       "  cvt.rzi.s16.f32 %0,f;\n}"
        :"=h"(__BFLOAT16_TO_US(val)) : "h"(__BFLOAT16_TO_LWS(h)));
#else
    const float f = __bfloat162float(h);
    val = static_cast<short int>(f);
    const short int max_val = (short int)0x7fffU;
    const short int min_val = (short int)0x8000U;
    // saturation fixup
    if (f != f) {
        // NaN
        val = 0;
    } else if (f > static_cast<float>(max_val)) {
        // saturate maximum
        val = max_val;
    } else if (f < static_cast<float>(min_val)) {
        // saturate minimum
        val = min_val;
    }
#endif
   return val;
}
__LWDA_BF16_DECL__ short int __bfloat162short_rd(const __lw_bfloat16 h)
{
   short int val;
   asm("{ .reg.f32 f;\n"
       "  mov.b32 f, {0,%1};\n"
       "  cvt.rmi.s16.f32 %0,f;\n}"
        :"=h"(__BFLOAT16_TO_US(val)) : "h"(__BFLOAT16_TO_LWS(h)));
   return val;
}
__LWDA_BF16_DECL__ short int __bfloat162short_ru(const __lw_bfloat16 h)
{
   short int val;
   asm("{ .reg.f32 f;\n"
       "  mov.b32 f, {0,%1};\n"
       "  cvt.rpi.s16.f32 %0,f;\n}"
        :"=h"(__BFLOAT16_TO_US(val)) : "h"(__BFLOAT16_TO_LWS(h)));
   return val;
}
__LWDA_HOSTDEVICE_BF16_DECL__ __lw_bfloat16 __short2bfloat16_rn(const short int i)
{
    const float f = static_cast<float>(i);
    return __float2bfloat16_rn(f);
}
__LWDA_BF16_DECL__ __lw_bfloat16 __short2bfloat16_rz(const short int i)
{
    return __float2bfloat16_rz(__int2float_rz(static_cast<int>(i)));
}
__LWDA_BF16_DECL__ __lw_bfloat16 __short2bfloat16_rd(const short int i)
{
    return __float2bfloat16_rd(__int2float_rd(static_cast<int>(i)));
}
__LWDA_BF16_DECL__ __lw_bfloat16 __short2bfloat16_ru(const short int i)
{
    return __float2bfloat16_ru(__int2float_ru(static_cast<int>(i)));
}

__LWDA_BF16_DECL__ unsigned int __bfloat162uint_rn(const __lw_bfloat16 h)
{
    return __float2uint_rn(__bfloat162float(h));
}
__LWDA_HOSTDEVICE_BF16_DECL__ unsigned int __bfloat162uint_rz(const __lw_bfloat16 h)
{
    const float f = __bfloat162float(h);
    unsigned int i;
    i = static_cast<unsigned int>(f);
#if !(defined __LWDA_ARCH__)
    const unsigned int max_val = 0xffffffffU;
    const unsigned int min_val = 0U;
    // saturation fixup
    if (f != f) {
        // NaN
        i = 0U;
    } else if (f >= static_cast<float>(max_val)) {
        // saturate maximum
        i = max_val;
    } else if (f < static_cast<float>(min_val)) {
        // saturate minimum
        i = min_val;
    }
#endif
    return i;
}
__LWDA_BF16_DECL__ unsigned int __bfloat162uint_rd(const __lw_bfloat16 h)
{
    return __float2uint_rd(__bfloat162float(h));
}
__LWDA_BF16_DECL__ unsigned int __bfloat162uint_ru(const __lw_bfloat16 h)
{
    return __float2uint_ru(__bfloat162float(h));
}
__LWDA_HOSTDEVICE_BF16_DECL__ __lw_bfloat16 __uint2bfloat16_rn(const unsigned int i)
{
#if (defined __LWDA_ARCH__)
    const float ru = __uint2float_ru(i);
    const float rd = __uint2float_rd(i);
    float rz = __uint2float_rz(i);
    if (ru != rd) {
        rz = __uint_as_float(__float_as_uint(rz) | 1U);
    }
    return __float2bfloat16_rn(rz);
#else
    const double d = static_cast<double>(i);
    return __double2bfloat16(d);
#endif
}
__LWDA_BF16_DECL__ __lw_bfloat16 __uint2bfloat16_rz(const unsigned int i)
{
    return __float2bfloat16_rz(__uint2float_rz(i));
}
__LWDA_BF16_DECL__ __lw_bfloat16 __uint2bfloat16_rd(const unsigned int i)
{
    return __float2bfloat16_rd(__uint2float_rd(i));
}
__LWDA_BF16_DECL__ __lw_bfloat16 __uint2bfloat16_ru(const unsigned int i)
{
    return __float2bfloat16_ru(__uint2float_ru(i));
}

__LWDA_BF16_DECL__ unsigned short int __bfloat162ushort_rn(const __lw_bfloat16 h)
{
   unsigned short int val;
   asm("{ .reg.f32 f;\n"
       "  mov.b32 f, {0,%1};\n"
       "  cvt.rni.u16.f32 %0,f;\n}"
        :"=h"(__BFLOAT16_TO_US(val)) : "h"(__BFLOAT16_TO_LWS(h)));
   return val;
}
__LWDA_HOSTDEVICE_BF16_DECL__ unsigned short int __bfloat162ushort_rz(const __lw_bfloat16 h)
{
   unsigned short int val;
#if (defined __LWDA_ARCH__)
   asm("{ .reg.f32 f;\n"
       "  mov.b32 f, {0,%1};\n"
       "  cvt.rzi.u16.f32 %0,f;\n}"
        :"=h"(__BFLOAT16_TO_US(val)) : "h"(__BFLOAT16_TO_LWS(h)));
#else
    const float f = __bfloat162float(h);
    val = static_cast<unsigned short int>(f);
    const unsigned short int max_val = 0xffffU;
    const unsigned short int min_val = 0U;
    // saturation fixup
    if (f != f) {
        // NaN
        val = 0U;
    } else if (f > static_cast<float>(max_val)) {
        // saturate maximum
        val = max_val;
    } else if (f < static_cast<float>(min_val)) {
        // saturate minimum
        val = min_val;
    }
#endif
   return val;
}
__LWDA_BF16_DECL__ unsigned short int __bfloat162ushort_rd(const __lw_bfloat16 h)
{
   unsigned short int val;
   asm("{ .reg.f32 f;\n"
       "  mov.b32 f, {0,%1};\n"
       "  cvt.rmi.u16.f32 %0,f;\n}"
        :"=h"(__BFLOAT16_TO_US(val)) : "h"(__BFLOAT16_TO_LWS(h)));
   return val;
}
__LWDA_BF16_DECL__ unsigned short int __bfloat162ushort_ru(const __lw_bfloat16 h)
{
   unsigned short int val;
   asm("{ .reg.f32 f;\n"
       "  mov.b32 f, {0,%1};\n"
       "  cvt.rpi.u16.f32 %0,f;\n}"
        :"=h"(__BFLOAT16_TO_US(val)) : "h"(__BFLOAT16_TO_LWS(h)));
   return val;
}
__LWDA_HOSTDEVICE_BF16_DECL__ __lw_bfloat16 __ushort2bfloat16_rn(const unsigned short int i)
{
    const float f = static_cast<float>(i);
    return __float2bfloat16_rn(f);
}
__LWDA_BF16_DECL__ __lw_bfloat16 __ushort2bfloat16_rz(const unsigned short int i)
{
    return __float2bfloat16_rz(__uint2float_rz(static_cast<unsigned int>(i)));
}
__LWDA_BF16_DECL__ __lw_bfloat16 __ushort2bfloat16_rd(const unsigned short int i)
{
    return __float2bfloat16_rd(__uint2float_rd(static_cast<unsigned int>(i)));
}
__LWDA_BF16_DECL__ __lw_bfloat16 __ushort2bfloat16_ru(const unsigned short int i)
{
    return __float2bfloat16_ru(__uint2float_ru(static_cast<unsigned int>(i)));
}

__LWDA_BF16_DECL__ unsigned long long int __bfloat162ull_rn(const __lw_bfloat16 h)
{
    return __float2ull_rn(__bfloat162float(h));
}
__LWDA_HOSTDEVICE_BF16_DECL__ unsigned long long int __bfloat162ull_rz(const __lw_bfloat16 h)
{
    const float f = __bfloat162float(h);
    unsigned long long int i;
    i = static_cast<unsigned long long int>(f);
#if !(defined __LWDA_ARCH__)
    const unsigned long long int max_val = 0xffffffffffffffffULL;
    const unsigned long long int min_val = 0ULL;
    // saturation fixup
    if (f != f) {
        // NaN
        i = 0x8000000000000000ULL;
    } else if (f >= static_cast<float>(max_val)) {
        // saturate maximum
        i = max_val;
    } else if (f < static_cast<float>(min_val)) {
        // saturate minimum
        i = min_val;
    }
#endif
    return i;
}
__LWDA_BF16_DECL__ unsigned long long int __bfloat162ull_rd(const __lw_bfloat16 h)
{
    return __float2ull_rd(__bfloat162float(h));
}
__LWDA_BF16_DECL__ unsigned long long int __bfloat162ull_ru(const __lw_bfloat16 h)
{
    return __float2ull_ru(__bfloat162float(h));
}
__LWDA_HOSTDEVICE_BF16_DECL__ __lw_bfloat16 __ull2bfloat16_rn(const unsigned long long int i)
{
#if (defined __LWDA_ARCH__)
    const float ru = __ull2float_ru(i);
    const float rd = __ull2float_rd(i);
    float rz = __ull2float_rz(i);
    if (ru != rd) {
        rz = __uint_as_float(__float_as_uint(rz) | 1U);
    }
    return __float2bfloat16_rn(rz);
#else
    float f = static_cast<float>(i);
    const unsigned long long int uf = static_cast<unsigned long long int>(f);
    unsigned int u;

    #if defined(__LWDA_ARCH__)
        u = __float_as_uint(f);
    #elif defined(__LWDACC__)
        (void)memcpy(&u, &f, sizeof(f));
    #else
        (void)std::memcpy(&u, &f, sizeof(f));
    #endif

    // round up happened here
    // note: no need to handle round up to f == 0x1.p64 specially
    if (uf > i) {
        u--;
    }
    if (uf != i) {
        u |= 1U;
    }

    #if defined(__LWDA_ARCH__)
        f = __int_as_float(static_cast<int>(u));
    #elif defined(__LWDACC__)
        (void)memcpy(&f, &u, sizeof(f));
    #else
        (void)std::memcpy(&f, &u, sizeof(f));
    #endif

    return __float2bfloat16_rn(f);
#endif
}
__LWDA_BF16_DECL__ __lw_bfloat16 __ull2bfloat16_rz(const unsigned long long int i)
{
    return __float2bfloat16_rz(__ull2float_rz(i));
}
__LWDA_BF16_DECL__ __lw_bfloat16 __ull2bfloat16_rd(const unsigned long long int i)
{
    return __float2bfloat16_rd(__ull2float_rd(i));
}
__LWDA_BF16_DECL__ __lw_bfloat16 __ull2bfloat16_ru(const unsigned long long int i)
{
    return __float2bfloat16_ru(__ull2float_ru(i));
}
__LWDA_BF16_DECL__ long long int __bfloat162ll_rn(const __lw_bfloat16 h)
{
    return __float2ll_rn(__bfloat162float(h));
}
__LWDA_HOSTDEVICE_BF16_DECL__ long long int __bfloat162ll_rz(const __lw_bfloat16 h)
{
    const float f = __bfloat162float(h);
    long long int i;
    i = static_cast<long long int>(f);
#if !(defined __LWDA_ARCH__)
    const long long int max_val = (long long int)0x7fffffffffffffffULL;
    const long long int min_val = (long long int)0x8000000000000000ULL;
    // saturation fixup
    if (f != f) {
        // NaN
        i = min_val;
    } else if (f >= static_cast<float>(max_val)) {
        // saturate maximum
        i = max_val;
    } else if (f < static_cast<float>(min_val)) {
        // saturate minimum
        i = min_val;
    }
#endif
    return i;
}
__LWDA_BF16_DECL__ long long int __bfloat162ll_rd(const __lw_bfloat16 h)
{
    return __float2ll_rd(__bfloat162float(h));
}
__LWDA_BF16_DECL__ long long int __bfloat162ll_ru(const __lw_bfloat16 h)
{
    return __float2ll_ru(__bfloat162float(h));
}
__LWDA_HOSTDEVICE_BF16_DECL__ __lw_bfloat16 __ll2bfloat16_rn(const long long int i)
{
#if (defined __LWDA_ARCH__)
    const float ru = __ll2float_ru(i);
    const float rd = __ll2float_rd(i);
    float rz = __ll2float_rz(i);
    if (ru != rd) {
        rz = __uint_as_float(__float_as_uint(rz) | 1U);
    }
    return __float2bfloat16_rn(rz);
#else
    float f = static_cast<float>(i);
    const long long int lf = static_cast<long long int>(f);
    unsigned int u;

    #if defined(__LWDA_ARCH__)
        u = __float_as_uint(f);
    #elif defined(__LWDACC__)
        (void)memcpy(&u, &f, sizeof(f));
    #else
        (void)std::memcpy(&u, &f, sizeof(f));
    #endif

    if ((f > 0.0f) && (lf > i)) {
        u--;
    }
    if ((f < 0.0f) && (lf < i)) {
        u--;
    }
    if (lf != i) {
        u |= 1U;
    }

    #if defined(__LWDA_ARCH__)
        f = __int_as_float(static_cast<int>(u));
    #elif defined(__LWDACC__)
        (void)memcpy(&f, &u, sizeof(f));
    #else
        (void)std::memcpy(&f, &u, sizeof(f));
    #endif

    return __float2bfloat16_rn(f);
#endif
}
__LWDA_BF16_DECL__ __lw_bfloat16 __ll2bfloat16_rz(const long long int i)
{
    return __float2bfloat16_rz(__ll2float_rz(i));
}
__LWDA_BF16_DECL__ __lw_bfloat16 __ll2bfloat16_rd(const long long int i)
{
    return __float2bfloat16_rd(__ll2float_rd(i));
}
__LWDA_BF16_DECL__ __lw_bfloat16 __ll2bfloat16_ru(const long long int i)
{
    return __float2bfloat16_ru(__ll2float_ru(i));
}

__LWDA_BF16_DECL__ __lw_bfloat16 htrunc(const __lw_bfloat16 h)
{
    return __float2bfloat16_rz(truncf(__bfloat162float(h)));
}
__LWDA_BF16_DECL__ __lw_bfloat16 hceil(const __lw_bfloat16 h)
{
    return __float2bfloat16_ru(ceilf(__bfloat162float(h)));
}
__LWDA_BF16_DECL__ __lw_bfloat16 hfloor(const __lw_bfloat16 h)
{
    return __float2bfloat16_rd(floorf(__bfloat162float(h)));
}
__LWDA_BF16_DECL__ __lw_bfloat16 hrint(const __lw_bfloat16 h)
{
    return __float2bfloat16_rn(rintf(__bfloat162float(h)));
}

__LWDA_BF16_DECL__ __lw_bfloat162 h2trunc(const __lw_bfloat162 h)
{
    const __lw_bfloat16 low = __float2bfloat16_rz(truncf(__low2float(h)));
    const __lw_bfloat16 high = __float2bfloat16_rz(truncf(__high2float(h)));
    return __lw_bfloat162(low, high);
}
__LWDA_BF16_DECL__ __lw_bfloat162 h2ceil(const __lw_bfloat162 h)
{
    const __lw_bfloat16 low = __float2bfloat16_ru(ceilf(__low2float(h)));
    const __lw_bfloat16 high = __float2bfloat16_ru(ceilf(__high2float(h)));
    return __lw_bfloat162(low, high);
}
__LWDA_BF16_DECL__ __lw_bfloat162 h2floor(const __lw_bfloat162 h)
{
    const __lw_bfloat16 low = __float2bfloat16_rd(floorf(__low2float(h)));
    const __lw_bfloat16 high = __float2bfloat16_rd(floorf(__high2float(h)));
    return __lw_bfloat162(low, high);
}

__LWDA_BF16_DECL__ __lw_bfloat162 h2rint(const __lw_bfloat162 h)
{
    return __halves2bfloat162(hrint(__low2bfloat16(h)), hrint(__high2bfloat16(h)));
}
__LWDA_BF16_DECL__ __lw_bfloat162 __lows2bfloat162(const __lw_bfloat162 a, const __lw_bfloat162 b)
{
    __lw_bfloat162 val;
    asm("{.reg .b16 alow,ahigh,blow,bhigh;\n"
        "  mov.b32 {alow,ahigh}, %1;\n"
        "  mov.b32 {blow,bhigh}, %2;\n"
        "  mov.b32 %0, {alow,blow};}\n" : "=r"(__BFLOAT162_TO_UI(val)) : "r"(__BFLOAT162_TO_LWI(a)), "r"(__BFLOAT162_TO_LWI(b)));
    return val;
}
__LWDA_BF16_DECL__ __lw_bfloat162 __highs2bfloat162(const __lw_bfloat162 a, const __lw_bfloat162 b)
{
    __lw_bfloat162 val;
    asm("{.reg .b16 alow,ahigh,blow,bhigh;\n"
        "  mov.b32 {alow,ahigh}, %1;\n"
        "  mov.b32 {blow,bhigh}, %2;\n"
        "  mov.b32 %0, {ahigh,bhigh};}\n" : "=r"(__BFLOAT162_TO_UI(val)) : "r"(__BFLOAT162_TO_LWI(a)), "r"(__BFLOAT162_TO_LWI(b)));
    return val;
}
__LWDA_BF16_DECL__ __lw_bfloat16 __low2bfloat16(const __lw_bfloat162 a)
{
    __lw_bfloat16 ret;
    asm("{.reg .b16 low,high;\n"
        " mov.b32 {low,high}, %1;\n"
        " mov.b16 %0, low;}" : "=h"(__BFLOAT16_TO_US(ret)) : "r"(__BFLOAT162_TO_LWI(a)));
    return ret;
}
__LWDA_BF16_DECL__ int __hisinf(const __lw_bfloat16 a)
{
    int retval;
    if (__BFLOAT16_TO_LWS(a) == 0xFF80U) {
        retval = -1;
    } else if (__BFLOAT16_TO_LWS(a) == 0x7F80U) {
        retval = 1;
    } else {
        retval = 0;
    }
    return retval;
}
__LWDA_BF16_DECL__ __lw_bfloat162 __low2bfloat162(const __lw_bfloat162 a)
{
    __lw_bfloat162 val;
    asm("{.reg .b16 low,high;\n"
        "  mov.b32 {low,high}, %1;\n"
        "  mov.b32 %0, {low,low};}\n" : "=r"(__BFLOAT162_TO_UI(val)) : "r"(__BFLOAT162_TO_LWI(a)));
    return val;
}
__LWDA_BF16_DECL__ __lw_bfloat162 __high2bfloat162(const __lw_bfloat162 a)
{
    __lw_bfloat162 val;
    asm("{.reg .b16 low,high;\n"
        "  mov.b32 {low,high}, %1;\n"
        "  mov.b32 %0, {high,high};}\n" : "=r"(__BFLOAT162_TO_UI(val)) : "r"(__BFLOAT162_TO_LWI(a)));
    return val;
}
__LWDA_BF16_DECL__ __lw_bfloat16 __high2bfloat16(const __lw_bfloat162 a)
{
    __lw_bfloat16 ret;
    asm("{.reg .b16 low,high;\n"
        " mov.b32 {low,high}, %1;\n"
        " mov.b16 %0, high;}" : "=h"(__BFLOAT16_TO_US(ret)) : "r"(__BFLOAT162_TO_LWI(a)));
    return ret;
}
__LWDA_BF16_DECL__ __lw_bfloat162 __halves2bfloat162(const __lw_bfloat16 a, const __lw_bfloat16 b)
{
    __lw_bfloat162 val;
    asm("{  mov.b32 %0, {%1,%2};}\n"
        : "=r"(__BFLOAT162_TO_UI(val)) : "h"(__BFLOAT16_TO_LWS(a)), "h"(__BFLOAT16_TO_LWS(b)));
    return val;
}
__LWDA_BF16_DECL__ __lw_bfloat162 __bfloat162bfloat162(const __lw_bfloat16 a)
{
    __lw_bfloat162 val;
    asm("{  mov.b32 %0, {%1,%1};}\n"
        : "=r"(__BFLOAT162_TO_UI(val)) : "h"(__BFLOAT16_TO_LWS(a)));
    return val;
}
__LWDA_BF16_DECL__ __lw_bfloat162 __lowhigh2highlow(const __lw_bfloat162 a)
{
    __lw_bfloat162 val;
    asm("{.reg .b16 low,high;\n"
        "  mov.b32 {low,high}, %1;\n"
        "  mov.b32 %0, {high,low};}\n" : "=r"(__BFLOAT162_TO_UI(val)) : "r"(__BFLOAT162_TO_LWI(a)));
    return val;
}
__LWDA_BF16_DECL__ short int __bfloat16_as_short(const __lw_bfloat16 h)
{
    return static_cast<short int>(__BFLOAT16_TO_LWS(h));
}
__LWDA_BF16_DECL__ unsigned short int __bfloat16_as_ushort(const __lw_bfloat16 h)
{
    return __BFLOAT16_TO_LWS(h);
}
__LWDA_BF16_DECL__ __lw_bfloat16 __short_as_bfloat16(const short int i)
{
    __lw_bfloat16 h;
    __BFLOAT16_TO_US(h) = static_cast<unsigned short int>(i);
    return h;
}
__LWDA_BF16_DECL__ __lw_bfloat16 __ushort_as_bfloat16(const unsigned short int i)
{
    __lw_bfloat16 h;
    __BFLOAT16_TO_US(h) = i;
    return h;
}

/******************************************************************************
*                           __lw_bfloat16, __lw_bfloat162 warp shuffle                     *
******************************************************************************/
#define __SHUFFLE_SYNC_BFLOAT162_MACRO(name) /* do */ {\
   __lw_bfloat162 r; \
   asm volatile ("{"#name" %0,%1,%2,%3,%4;\n}" \
       :"=r"(__BFLOAT162_TO_UI(r)): "r"(__BFLOAT162_TO_LWI(var)), "r"(delta), "r"(c), "r"(mask)); \
   return r; \
} /* while(0) */

__LWDA_BF16_DECL__ __lw_bfloat162 __shfl_sync(const unsigned mask, const __lw_bfloat162 var, const int delta, const int width)
{
    unsigned int warp_size;
    asm("{mov.u32 %0, WARP_SZ;\n}" : "=r"(warp_size));
    const unsigned int c = ((warp_size - static_cast<unsigned>(width)) << 8U) | 0x1fU;
    __SHUFFLE_SYNC_BFLOAT162_MACRO(shfl.sync.idx.b32)
}
__LWDA_BF16_DECL__ __lw_bfloat162 __shfl_up_sync(const unsigned mask, const __lw_bfloat162 var, const unsigned int delta, const int width)
{
    unsigned int warp_size;
    asm("{mov.u32 %0, WARP_SZ;\n}" : "=r"(warp_size));
    const unsigned int c = (warp_size - static_cast<unsigned>(width)) << 8U;
    __SHUFFLE_SYNC_BFLOAT162_MACRO(shfl.sync.up.b32)
}
__LWDA_BF16_DECL__ __lw_bfloat162 __shfl_down_sync(const unsigned mask, const __lw_bfloat162 var, const unsigned int delta, const int width)
{
    unsigned int warp_size;
    asm("{mov.u32 %0, WARP_SZ;\n}" : "=r"(warp_size));
    const unsigned int c = ((warp_size - static_cast<unsigned>(width)) << 8U) | 0x1fU;
    __SHUFFLE_SYNC_BFLOAT162_MACRO(shfl.sync.down.b32)
}
__LWDA_BF16_DECL__ __lw_bfloat162 __shfl_xor_sync(const unsigned mask, const __lw_bfloat162 var, const int delta, const int width)
{
    unsigned int warp_size;
    asm("{mov.u32 %0, WARP_SZ;\n}" : "=r"(warp_size));
    const unsigned int c = ((warp_size - static_cast<unsigned>(width)) << 8U) | 0x1fU;
    __SHUFFLE_SYNC_BFLOAT162_MACRO(shfl.sync.bfly.b32)
}

#undef __SHUFFLE_SYNC_BFLOAT162_MACRO

__LWDA_BF16_DECL__ __lw_bfloat16 __shfl_sync(const unsigned mask, const __lw_bfloat16 var, const int delta, const int width)
{
    const __lw_bfloat162 temp1 = __halves2bfloat162(var, var);
    const __lw_bfloat162 temp2 = __shfl_sync(mask, temp1, delta, width);
    return __low2bfloat16(temp2);
}
__LWDA_BF16_DECL__ __lw_bfloat16 __shfl_up_sync(const unsigned mask, const __lw_bfloat16 var, const unsigned int delta, const int width)
{
    const __lw_bfloat162 temp1 = __halves2bfloat162(var, var);
    const __lw_bfloat162 temp2 = __shfl_up_sync(mask, temp1, delta, width);
    return __low2bfloat16(temp2);
}
__LWDA_BF16_DECL__ __lw_bfloat16 __shfl_down_sync(const unsigned mask, const __lw_bfloat16 var, const unsigned int delta, const int width)
{
    const __lw_bfloat162 temp1 = __halves2bfloat162(var, var);
    const __lw_bfloat162 temp2 = __shfl_down_sync(mask, temp1, delta, width);
    return __low2bfloat16(temp2);
}
__LWDA_BF16_DECL__ __lw_bfloat16 __shfl_xor_sync(const unsigned mask, const __lw_bfloat16 var, const int delta, const int width)
{
    const __lw_bfloat162 temp1 = __halves2bfloat162(var, var);
    const __lw_bfloat162 temp2 = __shfl_xor_sync(mask, temp1, delta, width);
    return __low2bfloat16(temp2);
}

/******************************************************************************
*               __lw_bfloat16 and __lw_bfloat162 __ldg,__ldcg,__ldca,__ldcs                *
******************************************************************************/

#if defined(__cplusplus)
#if (defined(_MSC_VER) && defined(_WIN64)) || defined(__LP64__) || defined(__LWDACC_RTC__)
#define __LDG_PTR   "l"
#else
#define __LDG_PTR   "r"
#endif /*(defined(_MSC_VER) && defined(_WIN64)) || defined(__LP64__) || defined(__LWDACC_RTC__)*/
__LWDA_BF16_DECL__ __lw_bfloat162 __ldg(const  __lw_bfloat162 *const ptr)
{
    __lw_bfloat162 ret;
    asm ("ld.global.nc.b32 %0, [%1];"  : "=r"(__BFLOAT162_TO_UI(ret)) : __LDG_PTR(ptr));
    return ret;
}
__LWDA_BF16_DECL__ __lw_bfloat16 __ldg(const __lw_bfloat16 *const ptr)
{
    __lw_bfloat16 ret;
    asm ("ld.global.nc.b16 %0, [%1];"  : "=h"(__BFLOAT16_TO_US(ret)) : __LDG_PTR(ptr));
    return ret;
}
__LWDA_BF16_DECL__ __lw_bfloat162 __ldcg(const  __lw_bfloat162 *const ptr)
{
    __lw_bfloat162 ret;
    asm ("ld.global.cg.b32 %0, [%1];"  : "=r"(__BFLOAT162_TO_UI(ret)) : __LDG_PTR(ptr));
    return ret;
}
__LWDA_BF16_DECL__ __lw_bfloat16 __ldcg(const __lw_bfloat16 *const ptr)
{
    __lw_bfloat16 ret;
    asm ("ld.global.cg.b16 %0, [%1];"  : "=h"(__BFLOAT16_TO_US(ret)) : __LDG_PTR(ptr));
    return ret;
}
__LWDA_BF16_DECL__ __lw_bfloat162 __ldca(const  __lw_bfloat162 *const ptr)
{
    __lw_bfloat162 ret;
    asm ("ld.global.ca.b32 %0, [%1];"  : "=r"(__BFLOAT162_TO_UI(ret)) : __LDG_PTR(ptr));
    return ret;
}
__LWDA_BF16_DECL__ __lw_bfloat16 __ldca(const __lw_bfloat16 *const ptr)
{
    __lw_bfloat16 ret;
    asm ("ld.global.ca.b16 %0, [%1];"  : "=h"(__BFLOAT16_TO_US(ret)) : __LDG_PTR(ptr));
    return ret;
}
__LWDA_BF16_DECL__ __lw_bfloat162 __ldcs(const  __lw_bfloat162 *const ptr)
{
    __lw_bfloat162 ret;
    asm ("ld.global.cs.b32 %0, [%1];"  : "=r"(__BFLOAT162_TO_UI(ret)) : __LDG_PTR(ptr));
    return ret;
}
__LWDA_BF16_DECL__ __lw_bfloat16 __ldcs(const __lw_bfloat16 *const ptr)
{
    __lw_bfloat16 ret;
    asm ("ld.global.cs.b16 %0, [%1];"  : "=h"(__BFLOAT16_TO_US(ret)) : __LDG_PTR(ptr));
    return ret;
}
__LWDA_BF16_DECL__ __lw_bfloat162 __ldlu(const  __lw_bfloat162 *const ptr)
{
    __lw_bfloat162 ret;
    asm ("ld.global.lu.b32 %0, [%1];"  : "=r"(__BFLOAT162_TO_UI(ret)) : __LDG_PTR(ptr) : "memory");
    return ret;
}
__LWDA_BF16_DECL__ __lw_bfloat16 __ldlu(const __lw_bfloat16 *const ptr)
{
    __lw_bfloat16 ret;
    asm ("ld.global.lu.b16 %0, [%1];"  : "=h"(__BFLOAT16_TO_US(ret)) : __LDG_PTR(ptr) : "memory");
    return ret;
}
__LWDA_BF16_DECL__ __lw_bfloat162 __ldcv(const  __lw_bfloat162 *const ptr)
{
    __lw_bfloat162 ret;
    asm ("ld.global.cv.b32 %0, [%1];"  : "=r"(__BFLOAT162_TO_UI(ret)) : __LDG_PTR(ptr) : "memory");
    return ret;
}
__LWDA_BF16_DECL__ __lw_bfloat16 __ldcv(const __lw_bfloat16 *const ptr)
{
    __lw_bfloat16 ret;
    asm ("ld.global.cv.b16 %0, [%1];"  : "=h"(__BFLOAT16_TO_US(ret)) : __LDG_PTR(ptr) : "memory");
    return ret;
}

__LWDA_BF16_DECL__ void __stwb(__lw_bfloat162 *const ptr, const __lw_bfloat162 value)
{
    asm ("st.global.wb.b32 [%0], %1;"  :: __LDG_PTR(ptr), "r"(__BFLOAT162_TO_LWI(value)) : "memory");
}
__LWDA_BF16_DECL__ void __stwb(__lw_bfloat16 *const ptr, const __lw_bfloat16 value)
{
    asm ("st.global.wb.b16 [%0], %1;"  :: __LDG_PTR(ptr),  "h"(__BFLOAT16_TO_LWS(value)) : "memory");
}
__LWDA_BF16_DECL__ void __stcg(__lw_bfloat162 *const ptr, const __lw_bfloat162 value)
{
    asm ("st.global.cg.b32 [%0], %1;"  :: __LDG_PTR(ptr), "r"(__BFLOAT162_TO_LWI(value)) : "memory");
}
__LWDA_BF16_DECL__ void __stcg(__lw_bfloat16 *const ptr, const __lw_bfloat16 value)
{
    asm ("st.global.cg.b16 [%0], %1;"  :: __LDG_PTR(ptr),  "h"(__BFLOAT16_TO_LWS(value)) : "memory");
}
__LWDA_BF16_DECL__ void __stcs(__lw_bfloat162 *const ptr, const __lw_bfloat162 value)
{
    asm ("st.global.cs.b32 [%0], %1;"  :: __LDG_PTR(ptr), "r"(__BFLOAT162_TO_LWI(value)) : "memory");
}
__LWDA_BF16_DECL__ void __stcs(__lw_bfloat16 *const ptr, const __lw_bfloat16 value)
{
    asm ("st.global.cs.b16 [%0], %1;"  :: __LDG_PTR(ptr),  "h"(__BFLOAT16_TO_LWS(value)) : "memory");
}
__LWDA_BF16_DECL__ void __stwt(__lw_bfloat162 *const ptr, const __lw_bfloat162 value)
{
    asm ("st.global.wt.b32 [%0], %1;"  :: __LDG_PTR(ptr), "r"(__BFLOAT162_TO_LWI(value)) : "memory");
}
__LWDA_BF16_DECL__ void __stwt(__lw_bfloat16 *const ptr, const __lw_bfloat16 value)
{
    asm ("st.global.wt.b16 [%0], %1;"  :: __LDG_PTR(ptr),  "h"(__BFLOAT16_TO_LWS(value)) : "memory");
}

#undef __LDG_PTR
#endif /*defined(__cplusplus) */
/******************************************************************************
*                             __lw_bfloat162 comparison                             *
******************************************************************************/
#define __COMPARISON_OP_BFLOAT162_MACRO(name) /* do */ {\
   __lw_bfloat162 val; \
   asm( "{.reg .b32 low_a,low_b,high_a,high_b,high_res,low_res;\n"\
        "  and.b32 high_a, %1, 0xffff0000U;\n"\
        "  and.b32 high_b, %2, 0xffff0000U;\n"\
        "  shl.b32 low_a, %1, 16;\n"\
        "  shl.b32 low_b, %2, 16;\n"\
        "  "#name".f32.f32 low_res, low_a, low_b;\n"\
        "  "#name".f32.f32 high_res, high_a, high_b;\n"\
        "  shr.u32 low_res, low_res, 16;\n"\
        "  or.b32  %0, high_res, low_res;}\n"\
        :"=r"(__BFLOAT162_TO_UI(val)) : "r"(__BFLOAT162_TO_LWI(a)),"r"(__BFLOAT162_TO_LWI(b))); \
   return val; \
} /* while(0) */
__LWDA_BF16_DECL__ __lw_bfloat162 __heq2(const __lw_bfloat162 a, const __lw_bfloat162 b)
{
    __COMPARISON_OP_BFLOAT162_MACRO(set.eq)
}
__LWDA_BF16_DECL__ __lw_bfloat162 __hne2(const __lw_bfloat162 a, const __lw_bfloat162 b)
{
    __COMPARISON_OP_BFLOAT162_MACRO(set.ne)
}
__LWDA_BF16_DECL__ __lw_bfloat162 __hle2(const __lw_bfloat162 a, const __lw_bfloat162 b)
{
    __COMPARISON_OP_BFLOAT162_MACRO(set.le)
}
__LWDA_BF16_DECL__ __lw_bfloat162 __hge2(const __lw_bfloat162 a, const __lw_bfloat162 b)
{
    __COMPARISON_OP_BFLOAT162_MACRO(set.ge)
}
__LWDA_BF16_DECL__ __lw_bfloat162 __hlt2(const __lw_bfloat162 a, const __lw_bfloat162 b)
{
    __COMPARISON_OP_BFLOAT162_MACRO(set.lt)
}
__LWDA_BF16_DECL__ __lw_bfloat162 __hgt2(const __lw_bfloat162 a, const __lw_bfloat162 b)
{
    __COMPARISON_OP_BFLOAT162_MACRO(set.gt)
}
__LWDA_BF16_DECL__ __lw_bfloat162 __hequ2(const __lw_bfloat162 a, const __lw_bfloat162 b)
{
    __COMPARISON_OP_BFLOAT162_MACRO(set.equ)
}
__LWDA_BF16_DECL__ __lw_bfloat162 __hneu2(const __lw_bfloat162 a, const __lw_bfloat162 b)
{
    __COMPARISON_OP_BFLOAT162_MACRO(set.neu)
}
__LWDA_BF16_DECL__ __lw_bfloat162 __hleu2(const __lw_bfloat162 a, const __lw_bfloat162 b)
{
    __COMPARISON_OP_BFLOAT162_MACRO(set.leu)
}
__LWDA_BF16_DECL__ __lw_bfloat162 __hgeu2(const __lw_bfloat162 a, const __lw_bfloat162 b)
{
    __COMPARISON_OP_BFLOAT162_MACRO(set.geu)
}
__LWDA_BF16_DECL__ __lw_bfloat162 __hltu2(const __lw_bfloat162 a, const __lw_bfloat162 b)
{
    __COMPARISON_OP_BFLOAT162_MACRO(set.ltu)
}
__LWDA_BF16_DECL__ __lw_bfloat162 __hgtu2(const __lw_bfloat162 a, const __lw_bfloat162 b)
{
    __COMPARISON_OP_BFLOAT162_MACRO(set.gtu)
}
#undef __COMPARISON_OP_BFLOAT162_MACRO
#define __BOOL_COMPARISON_OP_BFLOAT162_MACRO(name) /* do */ {\
   unsigned int val; \
   asm( "{.reg .b32 low_a,low_b,high_a,high_b,high_res,low_res;\n"\
        "  and.b32 high_a, %1, 0xffff0000U;\n"\
        "  and.b32 high_b, %2, 0xffff0000U;\n"\
        "  shl.b32 low_a, %1, 16;\n"\
        "  shl.b32 low_b, %2, 16;\n"\
        "  "#name".f32.f32 low_res, low_a, low_b;\n"\
        "  "#name".f32.f32 high_res, high_a, high_b;\n"\
        "  and.b32 %0, high_res, low_res;}\n"\
        :"=r"(val) : "r"(__BFLOAT162_TO_LWI(a)),"r"(__BFLOAT162_TO_LWI(b))); \
   return (val != 0U) ? true : false; \
} /* while(0) */
__LWDA_BF16_DECL__ bool __hbeq2(const __lw_bfloat162 a, const __lw_bfloat162 b)
{
    __BOOL_COMPARISON_OP_BFLOAT162_MACRO(set.eq)
}
__LWDA_BF16_DECL__ bool __hbne2(const __lw_bfloat162 a, const __lw_bfloat162 b)
{
    __BOOL_COMPARISON_OP_BFLOAT162_MACRO(set.ne)
}
__LWDA_BF16_DECL__ bool __hble2(const __lw_bfloat162 a, const __lw_bfloat162 b)
{
    __BOOL_COMPARISON_OP_BFLOAT162_MACRO(set.le)
}
__LWDA_BF16_DECL__ bool __hbge2(const __lw_bfloat162 a, const __lw_bfloat162 b)
{
    __BOOL_COMPARISON_OP_BFLOAT162_MACRO(set.ge)
}
__LWDA_BF16_DECL__ bool __hblt2(const __lw_bfloat162 a, const __lw_bfloat162 b)
{
    __BOOL_COMPARISON_OP_BFLOAT162_MACRO(set.lt)
}
__LWDA_BF16_DECL__ bool __hbgt2(const __lw_bfloat162 a, const __lw_bfloat162 b)
{
    __BOOL_COMPARISON_OP_BFLOAT162_MACRO(set.gt)
}
__LWDA_BF16_DECL__ bool __hbequ2(const __lw_bfloat162 a, const __lw_bfloat162 b)
{
    __BOOL_COMPARISON_OP_BFLOAT162_MACRO(set.equ)
}
__LWDA_BF16_DECL__ bool __hbneu2(const __lw_bfloat162 a, const __lw_bfloat162 b)
{
    __BOOL_COMPARISON_OP_BFLOAT162_MACRO(set.neu)
}
__LWDA_BF16_DECL__ bool __hbleu2(const __lw_bfloat162 a, const __lw_bfloat162 b)
{
    __BOOL_COMPARISON_OP_BFLOAT162_MACRO(set.leu)
}
__LWDA_BF16_DECL__ bool __hbgeu2(const __lw_bfloat162 a, const __lw_bfloat162 b)
{
    __BOOL_COMPARISON_OP_BFLOAT162_MACRO(set.geu)
}
__LWDA_BF16_DECL__ bool __hbltu2(const __lw_bfloat162 a, const __lw_bfloat162 b)
{
    __BOOL_COMPARISON_OP_BFLOAT162_MACRO(set.ltu)
}
__LWDA_BF16_DECL__ bool __hbgtu2(const __lw_bfloat162 a, const __lw_bfloat162 b)
{
    __BOOL_COMPARISON_OP_BFLOAT162_MACRO(set.gtu)
}
#undef __BOOL_COMPARISON_OP_BFLOAT162_MACRO
/******************************************************************************
*                             __lw_bfloat16 comparison                              *
******************************************************************************/
#define __COMPARISON_OP_BFLOAT16_MACRO(name) /* do */ {\
   unsigned int val; \
   asm( "{.reg .b32 a,b;\n"\
        "  mov.b32 a, {0, %1};\n"\
        "  mov.b32 b, {0, %2};\n"\
        "  set."#name".f32.f32 %0, a, b;}\n"\
        :"=r"(val) : "h"(__BFLOAT16_TO_LWS(a)),"h"(__BFLOAT16_TO_LWS(b))); \
   return (val != 0U) ? true : false; \
} /* while(0) */
__LWDA_BF16_DECL__ bool __heq(const __lw_bfloat16 a, const __lw_bfloat16 b)
{
    __COMPARISON_OP_BFLOAT16_MACRO(eq)
}
__LWDA_BF16_DECL__ bool __hne(const __lw_bfloat16 a, const __lw_bfloat16 b)
{
    __COMPARISON_OP_BFLOAT16_MACRO(ne)
}
__LWDA_BF16_DECL__ bool __hle(const __lw_bfloat16 a, const __lw_bfloat16 b)
{
    __COMPARISON_OP_BFLOAT16_MACRO(le)
}
__LWDA_BF16_DECL__ bool __hge(const __lw_bfloat16 a, const __lw_bfloat16 b)
{
    __COMPARISON_OP_BFLOAT16_MACRO(ge)
}
__LWDA_BF16_DECL__ bool __hlt(const __lw_bfloat16 a, const __lw_bfloat16 b)
{
    __COMPARISON_OP_BFLOAT16_MACRO(lt)
}
__LWDA_BF16_DECL__ bool __hgt(const __lw_bfloat16 a, const __lw_bfloat16 b)
{
    __COMPARISON_OP_BFLOAT16_MACRO(gt)
}
__LWDA_BF16_DECL__ bool __hequ(const __lw_bfloat16 a, const __lw_bfloat16 b)
{
    __COMPARISON_OP_BFLOAT16_MACRO(equ)
}
__LWDA_BF16_DECL__ bool __hneu(const __lw_bfloat16 a, const __lw_bfloat16 b)
{
    __COMPARISON_OP_BFLOAT16_MACRO(neu)
}
__LWDA_BF16_DECL__ bool __hleu(const __lw_bfloat16 a, const __lw_bfloat16 b)
{
    __COMPARISON_OP_BFLOAT16_MACRO(leu)
}
__LWDA_BF16_DECL__ bool __hgeu(const __lw_bfloat16 a, const __lw_bfloat16 b)
{
    __COMPARISON_OP_BFLOAT16_MACRO(geu)
}
__LWDA_BF16_DECL__ bool __hltu(const __lw_bfloat16 a, const __lw_bfloat16 b)
{
    __COMPARISON_OP_BFLOAT16_MACRO(ltu)
}
__LWDA_BF16_DECL__ bool __hgtu(const __lw_bfloat16 a, const __lw_bfloat16 b)
{
    __COMPARISON_OP_BFLOAT16_MACRO(gtu)
}
#undef __COMPARISON_OP_BFLOAT16_MACRO
/******************************************************************************
*                            __lw_bfloat162 arithmetic                             *
******************************************************************************/
#define __BINARY_OP_BFLOAT162_MACRO(name) /* do */ {\
   __lw_bfloat162 val; \
   asm( "{.reg .b32 low_a,low_b,high_a,high_b,high_res,low_res;\n"\
        " .reg .b16 low,high;\n"\
        "  and.b32 high_a, %1, 0xffff0000U;\n"\
        "  and.b32 high_b, %2, 0xffff0000U;\n"\
        "  shl.b32 low_a, %1, 16;\n"\
        "  shl.b32 low_b, %2, 16;\n"\
        "  "#name".f32 low_res, low_a, low_b;\n"\
        "  "#name".f32 high_res, high_a, high_b;\n"\
        "  cvt.rn.bf16.f32 low, low_res;\n"\
        "  cvt.rn.bf16.f32 high, high_res;\n"\
        "  mov.b32 %0, {low,high};}\n"\
        :"=r"(__BFLOAT162_TO_UI(val)) : "r"(__BFLOAT162_TO_LWI(a)),"r"(__BFLOAT162_TO_LWI(b))); \
   return val; \
} /* while(0) */

__LWDA_BF16_DECL__ __lw_bfloat162 __hadd2(const __lw_bfloat162 a, const __lw_bfloat162 b)
{
   __lw_bfloat162 val;
   asm( "{.reg .b32 c;\n"
        "  mov.b32 c, 0x3f803f80U;\n"
        "  fma.rn.bf16x2 %0,%1,c,%2;}\n"
        :"=r"(__BFLOAT162_TO_UI(val)) : "r"(__BFLOAT162_TO_LWI(a)),"r"(__BFLOAT162_TO_LWI(b))); \
   return val;
}
__LWDA_BF16_DECL__ __lw_bfloat162 __hsub2(const __lw_bfloat162 a, const __lw_bfloat162 b)
{
   __lw_bfloat162 val;
   asm( "{.reg .b32 c;\n"
        "  mov.b32 c, 0xbf80bf80U;\n"
        "  fma.rn.bf16x2 %0,%2,c,%1;}\n"
        :"=r"(__BFLOAT162_TO_UI(val)) : "r"(__BFLOAT162_TO_LWI(a)),"r"(__BFLOAT162_TO_LWI(b))); \
   return val;
}
__LWDA_BF16_DECL__ __lw_bfloat162 __hmul2(const __lw_bfloat162 a, const __lw_bfloat162 b)
{
   __lw_bfloat162 val;
   asm( "{.reg .b32 c;\n"
        "  mov.b32 c, 0x80008000U;\n"
        "  fma.rn.bf16x2 %0,%1,%2,c;}\n"
        :"=r"(__BFLOAT162_TO_UI(val)) : "r"(__BFLOAT162_TO_LWI(a)),"r"(__BFLOAT162_TO_LWI(b))); \
   return val;
}
__LWDA_BF16_DECL__ __lw_bfloat162 __hadd2_sat(const __lw_bfloat162 a, const __lw_bfloat162 b)
{
   __lw_bfloat162 val;
   asm( "{.reg .b32 f, one, zero;\n"
        "  mov.b32 one, 0x3f803f80U;\n"
        "  mov.b32 zero, 0;\n"
        "  fma.rn.bf16x2 f,%1,one,%2;\n"
        "  max.bf16x2 f, f, zero;\n"
        "  min.bf16x2 %0, f, one;\n}"
        :"=r"(__BFLOAT162_TO_UI(val)) : "r"(__BFLOAT162_TO_LWI(a)),"r"(__BFLOAT162_TO_LWI(b))); \
   return val;
}
__LWDA_BF16_DECL__ __lw_bfloat162 __hsub2_sat(const __lw_bfloat162 a, const __lw_bfloat162 b)
{
   __lw_bfloat162 val;
   asm( "{.reg .b32 f, one, zero, mone;\n"
        "  mov.b32 one, 0x3f803f80U;\n"
        "  mov.b32 zero, 0;\n"
        "  mov.b32 mone, 0xbf80bf80U;\n"
        "  fma.rn.bf16x2 f,%2,mone,%1;\n"
        "  max.bf16x2 f, f, zero;\n"
        "  min.bf16x2 %0, f, one;\n}"
        :"=r"(__BFLOAT162_TO_UI(val)) : "r"(__BFLOAT162_TO_LWI(a)),"r"(__BFLOAT162_TO_LWI(b))); \
   return val;
}
__LWDA_BF16_DECL__ __lw_bfloat162 __hmul2_sat(const __lw_bfloat162 a, const __lw_bfloat162 b)
{
   __lw_bfloat162 val;
   asm( "{.reg .b32 f, one, zero, mzero;\n"
        "  mov.b32 one, 0x3f803f80U;\n"
        "  mov.b32 zero, 0;\n"
        "  mov.b32 mzero, 0x80008000U;\n"
        "  fma.rn.bf16x2 f,%1,%2,mzero;\n"
        "  max.bf16x2 f, f, zero;\n"
        "  min.bf16x2 %0, f, one;\n}"
        :"=r"(__BFLOAT162_TO_UI(val)) : "r"(__BFLOAT162_TO_LWI(a)),"r"(__BFLOAT162_TO_LWI(b))); \
   return val;
}
__LWDA_BF16_DECL__ __lw_bfloat162 __hfma2(const __lw_bfloat162 a, const __lw_bfloat162 b, const __lw_bfloat162 c)
{
    __lw_bfloat162 val;
    asm( "{fma.rn.bf16x2 %0,%1,%2,%3;\n}"
         :"=r"(__BFLOAT162_TO_UI(val)) : "r"(__BFLOAT162_TO_LWI(a)),"r"(__BFLOAT162_TO_LWI(b)),"r"(__BFLOAT162_TO_LWI(c)));
    return val;
}
__LWDA_BF16_DECL__ __lw_bfloat162 __hfma2_sat(const __lw_bfloat162 a, const __lw_bfloat162 b, const __lw_bfloat162 c)
{
    __lw_bfloat162 val;
    asm( "{ .reg .b32 f, one, zero;\n"
         "  mov.b32 one, 0x3f803f80U;\n"
         "  mov.b32 zero, 0;\n"
         "  fma.rn.bf16x2 f, %1, %2, %3;\n"
         "  max.bf16x2 f, f, zero;\n"
         "  min.bf16x2 %0, f, one;\n}"
         :"=r"(__BFLOAT162_TO_UI(val)) : "r"(__BFLOAT162_TO_LWI(a)),"r"(__BFLOAT162_TO_LWI(b)),"r"(__BFLOAT162_TO_LWI(c)));
    return val;
}
__LWDA_BF16_DECL__ __lw_bfloat162 __h2div(const __lw_bfloat162 a, const __lw_bfloat162 b) {
    __lw_bfloat16 ha, hb;

    ha = __low2bfloat16(a);
    hb = __low2bfloat16(b);

    const __lw_bfloat16 v1 = __hdiv(ha, hb);

    ha = __high2bfloat16(a);
    hb = __high2bfloat16(b);

    const __lw_bfloat16 v2 = __hdiv(ha, hb);

    return __halves2bfloat162(v1, v2);
}
/******************************************************************************
*                             __lw_bfloat16 arithmetic                             *
******************************************************************************/
#define __BINARY_OP_BFLOAT16_MACRO(name) /* do */ {\
   __lw_bfloat16 val; \
   asm( "{.reg .b32 a,b,res;\n"\
        "  mov.b32 a, {0,%1};\n"\
        "  mov.b32 b, {0,%2};\n"\
        "  "#name".f32 res, a, b;\n"\
        "  cvt.rn.bf16.f32 %0, res;}\n"\
        :"=h"(__BFLOAT16_TO_US(val)) : "h"(__BFLOAT16_TO_LWS(a)),"h"(__BFLOAT16_TO_LWS(b))); \
   return val; \
} /* while(0) */

__LWDA_BF16_DECL__ __lw_bfloat16 __hadd(const __lw_bfloat16 a, const __lw_bfloat16 b)
{
   __lw_bfloat16 val;
   asm( "{.reg .b16 c;\n"
        "  mov.b16 c, 0x3f80U;\n"
        "  fma.rn.bf16 %0,%1,c,%2;}\n"
        :"=h"(__BFLOAT16_TO_US(val)) : "h"(__BFLOAT16_TO_LWS(a)),"h"(__BFLOAT16_TO_LWS(b))); \
   return val;
}
__LWDA_BF16_DECL__ __lw_bfloat16 __hsub(const __lw_bfloat16 a, const __lw_bfloat16 b)
{
   __lw_bfloat16 val;
   asm( "{.reg .b16 c;\n"
        "  mov.b16 c, 0xbf80U;\n"
        "  fma.rn.bf16 %0,%2,c,%1;}\n"
        :"=h"(__BFLOAT16_TO_US(val)) : "h"(__BFLOAT16_TO_LWS(a)),"h"(__BFLOAT16_TO_LWS(b))); \
   return val;
}
__LWDA_BF16_DECL__ __lw_bfloat16 __hmul(const __lw_bfloat16 a, const __lw_bfloat16 b)
{
   __lw_bfloat16 val;
   asm( "{.reg .b16 c;\n"
        "  mov.b16 c, 0x8000U;\n"
        "  fma.rn.bf16 %0,%1,%2,c;}\n"
        :"=h"(__BFLOAT16_TO_US(val)) : "h"(__BFLOAT16_TO_LWS(a)),"h"(__BFLOAT16_TO_LWS(b))); \
   return val;
}
__LWDA_BF16_DECL__ __lw_bfloat16 __hadd_sat(const __lw_bfloat16 a, const __lw_bfloat16 b)
{
    __lw_bfloat16 val;
    asm( "{ .reg .b16 f, one, zero;\n"
         "  mov.b16 one, 0x3f80U;\n"
         "  mov.b16 zero, 0;\n"
         "  fma.rn.bf16 f, %1, one, %2;\n"
         "  max.bf16 f, f, zero;\n"
         "  min.bf16 %0, f, one;\n}"
         :"=h"(__BFLOAT16_TO_US(val)) : "h"(__BFLOAT16_TO_LWS(a)),"h"(__BFLOAT16_TO_LWS(b)));
    return val;
}
__LWDA_BF16_DECL__ __lw_bfloat16 __hsub_sat(const __lw_bfloat16 a, const __lw_bfloat16 b)
{
    __lw_bfloat16 val;
    asm( "{ .reg .b16 f, one, zero, mone;\n"
         "  mov.b16 one, 0x3f80U;\n"
         "  mov.b16 zero, 0;\n"
         "  mov.b16 mone, 0xbf80U;\n"
         "  fma.rn.bf16 f, %2, mone, %1;\n"
         "  max.bf16 f, f, zero;\n"
         "  min.bf16 %0, f, one;\n}"
         :"=h"(__BFLOAT16_TO_US(val)) : "h"(__BFLOAT16_TO_LWS(a)),"h"(__BFLOAT16_TO_LWS(b)));
    return val;
}
__LWDA_BF16_DECL__ __lw_bfloat16 __hmul_sat(const __lw_bfloat16 a, const __lw_bfloat16 b)
{
    __lw_bfloat16 val;
    asm( "{ .reg .b16 f, one, zero, mzero;\n"
         "  mov.b16 one, 0x3f80U;\n"
         "  mov.b16 zero, 0;\n"
         "  mov.b16 mzero, 0x8000U;\n"
         "  fma.rn.bf16 f, %1, %2, mzero;\n"
         "  max.bf16 f, f, zero;\n"
         "  min.bf16 %0, f, one;\n}"
         :"=h"(__BFLOAT16_TO_US(val)) : "h"(__BFLOAT16_TO_LWS(a)),"h"(__BFLOAT16_TO_LWS(b)));
    return val;
}
__LWDA_BF16_DECL__ __lw_bfloat16 __hfma(const __lw_bfloat16 a, const __lw_bfloat16 b, const __lw_bfloat16 c)
{
    __lw_bfloat16 val;
    asm( "{fma.rn.bf16 %0,%1,%2,%3;\n}"
         :"=h"(__BFLOAT16_TO_US(val)) : "h"(__BFLOAT16_TO_LWS(a)),"h"(__BFLOAT16_TO_LWS(b)),"h"(__BFLOAT16_TO_LWS(c)));
    return val;
}
__LWDA_BF16_DECL__ __lw_bfloat16 __hfma_sat(const __lw_bfloat16 a, const __lw_bfloat16 b, const __lw_bfloat16 c)
{
    __lw_bfloat16 val;
    asm( "{ .reg .b16 f, one, zero;\n"
         "  mov.b16 one, 0x3f80U;\n"
         "  mov.b16 zero, 0;\n"
         "  fma.rn.bf16 f, %1, %2, %3;\n"
         "  max.bf16 f, f, zero;\n"
         "  min.bf16 %0, f, one;\n}"
         :"=h"(__BFLOAT16_TO_US(val)) : "h"(__BFLOAT16_TO_LWS(a)),"h"(__BFLOAT16_TO_LWS(b)),"h"(__BFLOAT16_TO_LWS(c)));
    return val;
}
__LWDA_BF16_DECL__ __lw_bfloat16 __hdiv(const __lw_bfloat16 a, const __lw_bfloat16 b) {
    __BINARY_OP_BFLOAT16_MACRO(div.rn)
}

/******************************************************************************
*                             __lw_bfloat162 functions                  *
******************************************************************************/
#define __APPROX_FCAST(fun) /* do */ {\
   __lw_bfloat16 val;\
   asm("{.reg.b32         f;        \n"\
                " .reg.b16         r;        \n"\
                "  mov.b16         r,%1;     \n"\
                "  mov.b32         f,{0,r};  \n"\
                "  "#fun".approx.f32   f,f;  \n"\
                "  cvt.rn.bf16.f32    r,f;  \n"\
                "  mov.b16         %0,r;     \n"\
                "}": "=h"(__BFLOAT16_TO_US(val)) : "h"(__BFLOAT16_TO_LWS(a)));\
   return val;\
} /* while(0) */
#define __APPROX_FCAST2(fun) /* do */ {\
   __lw_bfloat162 val;\
   asm("{.reg.b16         hl, hu;         \n"\
                " .reg.b32         fl, fu;         \n"\
                "  mov.b32         {hl, hu}, %1;   \n"\
                "  mov.b32         fl, {0,hl};     \n"\
                "  mov.b32         fu, {0,hu};     \n"\
                "  "#fun".approx.f32   fl, fl;     \n"\
                "  "#fun".approx.f32   fu, fu;     \n"\
                "  cvt.rn.bf16.f32    hl, fl;     \n"\
                "  cvt.rn.bf16.f32    hu, fu;     \n"\
                "  mov.b32         %0, {hl, hu};   \n"\
                "}":"=r"(__BFLOAT162_TO_UI(val)) : "r"(__BFLOAT162_TO_LWI(a)));       \
   return val;\
} /* while(0) */
__LWDA_BF16_DECL__ __lw_bfloat16 __hsin_internal(const __lw_bfloat16 a) {
    float f = __bfloat162float(a);
    f = sinf(f);
    return __float2bfloat16_rn(f);
}
__LWDA_BF16_DECL__ __lw_bfloat16 hsin(const __lw_bfloat16 a) {
    return __hsin_internal(a);
}
__LWDA_BF16_DECL__ __lw_bfloat162 h2sin(const __lw_bfloat162 a) {
    const __lw_bfloat16 l = __low2bfloat16(a);
    const __lw_bfloat16 h = __high2bfloat16(a);
    return __halves2bfloat162(__hsin_internal(l), __hsin_internal(h));
}
__LWDA_BF16_DECL__ __lw_bfloat16 __hcos_internal(const __lw_bfloat16 a) {
    float f = __bfloat162float(a);
    f = cosf(f);
    return __float2bfloat16_rn(f);
}
__LWDA_BF16_DECL__ __lw_bfloat16 hcos(const __lw_bfloat16 a) {
    return __hcos_internal(a);
}
__LWDA_BF16_DECL__ __lw_bfloat162 h2cos(const __lw_bfloat162 a) {
    const __lw_bfloat16 l = __low2bfloat16(a);
    const __lw_bfloat16 h = __high2bfloat16(a);
    return __halves2bfloat162(__hcos_internal(l), __hcos_internal(h));
}

__LWDA_BF16_DECL__ __lw_bfloat16 hexp(const __lw_bfloat16 a) {
    __lw_bfloat16 val;
    asm("{.reg.b32          f, C;           \n"
        " .reg.b16          h,r;            \n"
        "  mov.b16          h,%1;           \n"
        "  mov.b32          f,{0,h};        \n"
        "  mov.b32          C, 0x3fb8aa3lw;  \n"
        "  mul.f32          f,f,C;          \n"
        "  ex2.approx.f32   f,f;            \n"
        "  cvt.rn.bf16.f32 r,f;            \n"
        "  mov.b16          %0,r;           \n"
        "}": "=h"(__BFLOAT16_TO_US(val)) : "h"(__BFLOAT16_TO_LWS(a)));
    return val;
}
__LWDA_BF16_DECL__ __lw_bfloat162 h2exp(const __lw_bfloat162 a) {
    __lw_bfloat162 val;
    asm("{.reg.b16         hl, hu;         \n"
        " .reg.b32         h,r,fl,fu, C;   \n"
        "  mov.b32         {hl, hu}, %1;   \n"
        "  mov.b32         h, %1;          \n"
        "  mov.b32         fl, {0,hl};     \n"
        "  mov.b32         fu, {0,hu};     \n"
        "  mov.b32         C, 0x3fb8aa3lw;  \n"
        "  mul.f32         fl,fl,C;        \n"
        "  mul.f32         fu,fu,C;        \n"
        "  ex2.approx.f32      fl, fl;     \n"
        "  ex2.approx.f32      fu, fu;     \n"
        "  cvt.rn.bf16.f32    hl, fl;     \n"
        "  cvt.rn.bf16.f32    hu, fu;     \n"
        "  mov.b32         r, {hl, hu};    \n"
        "  mov.b32         %0, r;  \n"
        "}":"=r"(__BFLOAT162_TO_UI(val)) : "r"(__BFLOAT162_TO_LWI(a)));
    return val;
}
__LWDA_BF16_DECL__ __lw_bfloat16 hexp2(const __lw_bfloat16 a) {
    __lw_bfloat16 val;
    asm("{.reg.b32         f, ULP;         \n"
        " .reg.b16         r;              \n"
        "  mov.b16         r,%1;           \n"
        "  mov.b32         f,{0,r};        \n"
        "  ex2.approx.f32      f,f;        \n"
        "  mov.b32         ULP, 0x33800000U;\n"
        "  fma.rn.f32      f,f,ULP,f;      \n"
        "  cvt.rn.bf16.f32    r,f;        \n"
        "  mov.b16         %0,r;           \n"
        "}": "=h"(__BFLOAT16_TO_US(val)) : "h"(__BFLOAT16_TO_LWS(a)));
    return val;
}
__LWDA_BF16_DECL__ __lw_bfloat162 h2exp2(const __lw_bfloat162 a) {
    __lw_bfloat162 val;
    asm("{.reg.b16         hl, hu;         \n"
        " .reg.b32         fl, fu, ULP;    \n"
        "  mov.b32         {hl, hu}, %1;   \n"
        "  mov.b32         fl, {0,hl};     \n"
        "  mov.b32         fu, {0,hu};     \n"
        "  ex2.approx.f32      fl, fl;     \n"
        "  ex2.approx.f32      fu, fu;     \n"
        "  mov.b32         ULP, 0x33800000U;\n"
        "  fma.rn.f32      fl,fl,ULP,fl;   \n"
        "  fma.rn.f32      fu,fu,ULP,fu;   \n"
        "  cvt.rn.bf16.f32    hl, fl;     \n"
        "  cvt.rn.bf16.f32    hu, fu;     \n"
        "  mov.b32         %0, {hl, hu};   \n"
        "}":"=r"(__BFLOAT162_TO_UI(val)) : "r"(__BFLOAT162_TO_LWI(a)));
    return val;
}
__LWDA_BF16_DECL__ __lw_bfloat16 hexp10(const __lw_bfloat16 a) {
    __lw_bfloat16 val;
    asm("{.reg.b16         h, r;           \n"
        " .reg.b32         f, C;           \n"
        " .reg.pred        p;              \n"
        "  mov.b16         h, %1;          \n"
        "  setp.eq.b16     p, h, 0XBC95U;   \n"
        "  mov.b32         f, {0,h};       \n"
        "  mov.b32         C, 0x40549A78U;  \n"
        "  mul.f32         f,f,C;          \n"
        "  ex2.approx.f32      f, f;       \n"
        "  cvt.rn.bf16.f32    r, f;       \n"
        "  selp.b16        r, 0X3F75U, r, p;\n"
        "  mov.b16         %0, r;          \n"
        "}":"=h"(__BFLOAT16_TO_US(val)) : "h"(__BFLOAT16_TO_LWS(a)));
    return val;
}
__LWDA_BF16_DECL__ __lw_bfloat162 h2exp10(const __lw_bfloat162 a) {
    __lw_bfloat162 val;
    asm("{.reg.b16         hl, hu;         \n"
        " .reg.b32         h,r,fl,fu, C;   \n"
        " .reg.pred        pl, pu;         \n"
        "  mov.b32         {hl, hu}, %1;   \n"
        "  setp.eq.b16     pl, hl, 0XBC95U; \n"
        "  setp.eq.b16     pu, hu, 0XBC95U; \n"
        "  mov.b32         fl, {0,hl};     \n"
        "  mov.b32         fu, {0,hu};     \n"
        "  mov.b32         C, 0x40549A78U;  \n"
        "  mul.f32         fl,fl,C;        \n"
        "  mul.f32         fu,fu,C;        \n"
        "  ex2.approx.f32      fl, fl;     \n"
        "  ex2.approx.f32      fu, fu;     \n"
        "  cvt.rn.bf16.f32    hl, fl;     \n"
        "  cvt.rn.bf16.f32    hu, fu;     \n"
        "  selp.b16        hl,0X3F75U,hl,pl;\n"
        "  selp.b16        hu,0X3F75U,hu,pu;\n"
        "  mov.b32         r, {hl, hu};    \n"
        "  mov.b32         %0, r;  \n"
        "}":"=r"(__BFLOAT162_TO_UI(val)) : "r"(__BFLOAT162_TO_LWI(a)));
    return val;
}
__LWDA_BF16_DECL__ __lw_bfloat16 hlog2(const __lw_bfloat16 a) {
    __lw_bfloat16 val;
    asm("{.reg.b16         h, r;           \n"
        " .reg.b32         f;              \n"
        "  mov.b16         h, %1;          \n"
        "  mov.b32         f, {0,h};       \n"
        "  lg2.approx.f32      f, f;       \n"
        "  cvt.rn.bf16.f32    r, f;       \n"
        "  mov.b16         %0, r;          \n"
        "}":"=h"(__BFLOAT16_TO_US(val)) : "h"(__BFLOAT16_TO_LWS(a)));
    return val;
}
__LWDA_BF16_DECL__ __lw_bfloat162 h2log2(const __lw_bfloat162 a) {
    __lw_bfloat162 val;
    asm("{.reg.b16         hl, hu;         \n"
        " .reg.b32         fl, fu, r, p;   \n"
        "  mov.b32         {hl, hu}, %1;   \n"
        "  mov.b32         fl, {0,hl};     \n"
        "  mov.b32         fu, {0,hu};     \n"
        "  lg2.approx.f32      fl, fl;     \n"
        "  lg2.approx.f32      fu, fu;     \n"
        "  cvt.rn.bf16.f32    hl, fl;     \n"
        "  cvt.rn.bf16.f32    hu, fu;     \n"
        "  mov.b32         r, {hl, hu};    \n"
        "  mov.b32         %0, r;          \n"
        "}":"=r"(__BFLOAT162_TO_UI(val)) : "r"(__BFLOAT162_TO_LWI(a)));
    return val;
}
__LWDA_BF16_DECL__ __lw_bfloat16 hlog(const __lw_bfloat16 a) {
    __lw_bfloat16 val;
    asm("{.reg.b32         f, C;           \n"
        " .reg.b16         r,h;            \n"
        "  mov.b16         h,%1;           \n"
        "  mov.b32         f,{0,h};        \n"
        "  lg2.approx.f32      f,f;        \n"
        "  mov.b32         C, 0x3f317218U; \n"
        "  mul.f32         f,f,C;          \n"
        "  cvt.rn.bf16.f32    r,f;        \n"
        "  mov.b16         %0,r;           \n"
        "}": "=h"(__BFLOAT16_TO_US(val)) : "h"(__BFLOAT16_TO_LWS(a)));
    return val;
}
__LWDA_BF16_DECL__ __lw_bfloat162 h2log(const __lw_bfloat162 a) {
    __lw_bfloat162 val;
    asm("{.reg.b16         hl, hu;             \n"
        " .reg.b32         r, fl, fu, C, h;    \n"
        "  mov.b32         {hl, hu}, %1;       \n"
        "  mov.b32         h, %1;              \n"
        "  mov.b32         fl, {0,hl};         \n"
        "  mov.b32         fu, {0,hu};         \n"
        "  lg2.approx.f32      fl, fl;         \n"
        "  lg2.approx.f32      fu, fu;         \n"
        "  mov.b32         C, 0x3f317218U;     \n"
        "  mul.f32         fl,fl,C;            \n"
        "  mul.f32         fu,fu,C;            \n"
        "  cvt.rn.bf16.f32    hl, fl;         \n"
        "  cvt.rn.bf16.f32    hu, fu;         \n"
        "  mov.b32         r, {hl, hu};        \n"
        "  mov.b32         %0, r;              \n"
        "}":"=r"(__BFLOAT162_TO_UI(val)) : "r"(__BFLOAT162_TO_LWI(a)));
    return val;
}
__LWDA_BF16_DECL__ __lw_bfloat16 hlog10(const __lw_bfloat16 a) {
    __lw_bfloat16 val;
    asm("{.reg.b16         h, r;           \n"
        " .reg.b32         f, C;           \n"
        "  mov.b16         h, %1;          \n"
        "  mov.b32         f, {0,h};           \n"
        "  lg2.approx.f32      f, f;       \n"
        "  mov.b32         C, 0x3E9A209BU;  \n"
        "  mul.f32         f,f,C;          \n"
        "  cvt.rn.bf16.f32    r, f;       \n"
        "  mov.b16         %0, r;          \n"
        "}":"=h"(__BFLOAT16_TO_US(val)) : "h"(__BFLOAT16_TO_LWS(a)));
    return val;
}
__LWDA_BF16_DECL__ __lw_bfloat162 h2log10(const __lw_bfloat162 a) {
    __lw_bfloat162 val;
    asm("{.reg.b16         hl, hu;             \n"
        " .reg.b32         r, fl, fu, C, h;    \n"
        "  mov.b32         {hl, hu}, %1;       \n"
        "  mov.b32         h, %1;              \n"
        "  mov.b32         fl, {0,hl};         \n"
        "  mov.b32         fu, {0,hu};         \n"
        "  lg2.approx.f32      fl, fl;         \n"
        "  lg2.approx.f32      fu, fu;         \n"
        "  mov.b32         C, 0x3E9A209BU;      \n"
        "  mul.f32         fl,fl,C;            \n"
        "  mul.f32         fu,fu,C;            \n"
        "  cvt.rn.bf16.f32    hl, fl;         \n"
        "  cvt.rn.bf16.f32    hu, fu;         \n"
        "  mov.b32         r, {hl, hu};        \n"
        "  mov.b32         %0, r;              \n"
        "}":"=r"(__BFLOAT162_TO_UI(val)) : "r"(__BFLOAT162_TO_LWI(a)));
    return val;
}
#undef __SPEC_CASE2
#undef __SPEC_CASE
__LWDA_BF16_DECL__ __lw_bfloat162 h2rcp(const __lw_bfloat162 a) {
    __APPROX_FCAST2(rcp)
}
__LWDA_BF16_DECL__ __lw_bfloat16 hrcp(const __lw_bfloat16 a) {
    __APPROX_FCAST(rcp)
}
__LWDA_BF16_DECL__ __lw_bfloat162 h2rsqrt(const __lw_bfloat162 a) {
    __APPROX_FCAST2(rsqrt)
}
__LWDA_BF16_DECL__ __lw_bfloat16 hrsqrt(const __lw_bfloat16 a) {
    __APPROX_FCAST(rsqrt)
}
__LWDA_BF16_DECL__ __lw_bfloat162 h2sqrt(const __lw_bfloat162 a) {
    __APPROX_FCAST2(sqrt)
}
__LWDA_BF16_DECL__ __lw_bfloat16 hsqrt(const __lw_bfloat16 a) {
    __APPROX_FCAST(sqrt)
}
#undef __APPROX_FCAST
#undef __APPROX_FCAST2
__LWDA_BF16_DECL__ __lw_bfloat162 __hisnan2(const __lw_bfloat162 a)
{
    const __lw_bfloat162 b = a;
    __BINARY_OP_BFLOAT162_MACRO(set.nan.f32)
}
__LWDA_BF16_DECL__ bool __hisnan(const __lw_bfloat16 a)
{
    unsigned int r;
    asm( "{.reg .b32 a;\n"
         "  mov.b32 a, {0,%1};\n"
         "  set.nan.f32.f32 %0, a, a;}\n"
         :"=r"(r) : "h"(__BFLOAT16_TO_LWS(a)));
    return r != 0U;
}
__LWDA_BF16_DECL__ __lw_bfloat162 __hneg2(const __lw_bfloat162 a)
{
    __lw_bfloat162 r;
    asm("{neg.bf16x2 %0,%1;\n}"
        :"=r"(__BFLOAT162_TO_UI(r)) : "r"(__BFLOAT162_TO_LWI(a)));
    return r;
}
__LWDA_BF16_DECL__ __lw_bfloat16 __hneg(const __lw_bfloat16 a)
{
    __lw_bfloat16 r;
    asm("{neg.bf16 %0,%1;\n}"
        :"=h"(__BFLOAT16_TO_US(r)) : "h"(__BFLOAT16_TO_LWS(a)));
    return r;
}
__LWDA_BF16_DECL__ __lw_bfloat162 __habs2(const __lw_bfloat162 a)
{
    __lw_bfloat162 r;
    asm("{abs.bf16x2 %0,%1;\n}"
        :"=r"(__BFLOAT162_TO_UI(r)) : "r"(__BFLOAT162_TO_LWI(a)));
    return r;
}
__LWDA_BF16_DECL__ __lw_bfloat16 __habs(const __lw_bfloat16 a)
{
    __lw_bfloat16 r;
    asm("{abs.bf16 %0,%1;\n}"
        :"=h"(__BFLOAT16_TO_US(r)) : "h"(__BFLOAT16_TO_LWS(a)));
    return r;
}
/******************************************************************************
*                             __lw_bfloat16 arithmetic                             *
******************************************************************************/
__LWDA_BF16_DECL__ __lw_bfloat16 __hmax(const __lw_bfloat16 a, const __lw_bfloat16 b)
{
   __lw_bfloat16 val;
   asm( "{ max.bf16 %0,%1,%2;\n}"
        :"=h"(__BFLOAT16_TO_US(val)) : "h"(__BFLOAT16_TO_LWS(a)),"h"(__BFLOAT16_TO_LWS(b)));
   return val;
}
__LWDA_BF16_DECL__ __lw_bfloat16 __hmin(const __lw_bfloat16 a, const __lw_bfloat16 b)
{
    __lw_bfloat16 val;
    asm( "{ min.bf16 %0,%1,%2;\n}"
         :"=h"(__BFLOAT16_TO_US(val)) : "h"(__BFLOAT16_TO_LWS(a)),"h"(__BFLOAT16_TO_LWS(b)));
    return val;
}
__LWDA_BF16_DECL__ __lw_bfloat16 __hmax_nan(const __lw_bfloat16 a, const __lw_bfloat16 b)
{
    __lw_bfloat16 val;
    asm( "{ max.NaN.bf16 %0,%1,%2;\n}"
         :"=h"(__BFLOAT16_TO_US(val)) : "h"(__BFLOAT16_TO_LWS(a)),"h"(__BFLOAT16_TO_LWS(b)));
    return val;
}
__LWDA_BF16_DECL__ __lw_bfloat16 __hmin_nan(const __lw_bfloat16 a, const __lw_bfloat16 b)
{
    __lw_bfloat16 val;
    asm( "{ min.NaN.bf16 %0,%1,%2;\n}"
         :"=h"(__BFLOAT16_TO_US(val)) : "h"(__BFLOAT16_TO_LWS(a)),"h"(__BFLOAT16_TO_LWS(b)));
    return val;
}
__LWDA_BF16_DECL__ __lw_bfloat16 __hfma_relu(const __lw_bfloat16 a, const __lw_bfloat16 b, const __lw_bfloat16 c)
{
    __lw_bfloat16 val;
    asm( "{ fma.rn.relu.bf16 %0,%1,%2,%3;\n}"
         :"=h"(__BFLOAT16_TO_US(val)) : "h"(__BFLOAT16_TO_LWS(a)),"h"(__BFLOAT16_TO_LWS(b)),"h"(__BFLOAT16_TO_LWS(c)));
    return val;
}
/******************************************************************************
*                            __lw_bfloat162 arithmetic                             *
******************************************************************************/
__LWDA_BF16_DECL__ __lw_bfloat162 __hmax2(const __lw_bfloat162 a, const __lw_bfloat162 b)
{
    __lw_bfloat162 val;
    asm( "{ max.bf16x2 %0,%1,%2;\n}"
         :"=r"(__BFLOAT162_TO_UI(val)) : "r"(__BFLOAT162_TO_LWI(a)),"r"(__BFLOAT162_TO_LWI(b)));
    return val;
}
__LWDA_BF16_DECL__ __lw_bfloat162 __hmin2(const __lw_bfloat162 a, const __lw_bfloat162 b)
{
    __lw_bfloat162 val;
    asm( "{ min.bf16x2 %0,%1,%2;\n}"
         :"=r"(__BFLOAT162_TO_UI(val)) : "r"(__BFLOAT162_TO_LWI(a)),"r"(__BFLOAT162_TO_LWI(b)));
    return val;
}
__LWDA_BF16_DECL__ __lw_bfloat162 __hmax2_nan(const __lw_bfloat162 a, const __lw_bfloat162 b)
{
    __lw_bfloat162 val;
    asm( "{ max.NaN.bf16x2 %0,%1,%2;\n}"
         :"=r"(__BFLOAT162_TO_UI(val)) : "r"(__BFLOAT162_TO_LWI(a)),"r"(__BFLOAT162_TO_LWI(b)));
    return val;
}
__LWDA_BF16_DECL__ __lw_bfloat162 __hmin2_nan(const __lw_bfloat162 a, const __lw_bfloat162 b)
{
    __lw_bfloat162 val;
    asm( "{ min.NaN.bf16x2 %0,%1,%2;\n}"
         :"=r"(__BFLOAT162_TO_UI(val)) : "r"(__BFLOAT162_TO_LWI(a)),"r"(__BFLOAT162_TO_LWI(b)));
    return val;
}
__LWDA_BF16_DECL__ __lw_bfloat162 __hfma2_relu(const __lw_bfloat162 a, const __lw_bfloat162 b, const __lw_bfloat162 c)
{
    __lw_bfloat162 val;
    asm( "{ fma.rn.relu.bf16x2 %0,%1,%2,%3;\n}"
         :"=r"(__BFLOAT162_TO_UI(val)) : "r"(__BFLOAT162_TO_LWI(a)),"r"(__BFLOAT162_TO_LWI(b)),"r"(__BFLOAT162_TO_LWI(c)));
    return val;
}

__LWDA_BF16_DECL__ __lw_bfloat162 __hcmadd(const __lw_bfloat162 a, const __lw_bfloat162 b, const __lw_bfloat162 c)
{
    // fast version of complex multiply-accumulate
    // (a.re, a.im) * (b.re, b.im) + (c.re, c.im)
    // acc.re = (c.re + a.re*b.re) - a.im*b.im
    // acc.im = (c.im + a.re*b.im) + a.im*b.re
    __lw_bfloat16 real_tmp = __hfma(a.x, b.x, c.x);
    __lw_bfloat16 img_tmp  = __hfma(a.x, b.y, c.y);
    real_tmp = __hfma(__hneg(a.y), b.y, real_tmp);
    img_tmp  = __hfma(a.y,         b.x, img_tmp);
    return make_bfloat162(real_tmp, img_tmp);
}

__LWDA_BF16_DECL__ __lw_bfloat162 atomicAdd(__lw_bfloat162 *const address, const __lw_bfloat162 val)
{
    unsigned int* address_as_uint = (unsigned int*)address;
    unsigned int old = *address_as_uint, assumed;
    do {
        assumed = old;
        __lw_bfloat162 new_val = __hadd2(val, *(__lw_bfloat162*)&assumed);
        old = atomicCAS(address_as_uint, assumed, *(unsigned int*)&new_val);
    } while (assumed != old);
    return *(__lw_bfloat162*)&old;
}

__LWDA_BF16_DECL__ __lw_bfloat16 atomicAdd(__lw_bfloat16 *const address, const __lw_bfloat16 val)
{
    unsigned short int* address_as_us = (unsigned short int*)address;
    unsigned short int old = *address_as_us, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_us, assumed,
            __bfloat16_as_ushort(__hadd(val, __ushort_as_bfloat16(assumed))));
    } while (assumed != old);
    return __ushort_as_bfloat16(old);
}

#undef __LWDA_BF16_DECL__
#endif /* defined(__LWDACC__) && (__LWDA_ARCH__ >= 800 || !defined(__LWDA_ARCH__)) */
#endif /* defined(__cplusplus) */

#undef __BINARY_OP_BFLOAT162_MACRO
#undef __BINARY_OP_BFLOAT16_MACRO

#undef __LWDA_HOSTDEVICE_BF16_DECL__
#undef __LWDA_BF16_DECL__

/* Define first-class types "lw_bfloat16" and "lw_bfloat162", unless user specifies otherwise via "#define LWDA_NO_BFLOAT16" */
/* C cannot ever have these types defined here, because __lw_bfloat16 and __lw_bfloat162 are C++ classes */
#if defined(__cplusplus) && !defined(LWDA_NO_BFLOAT16)
typedef __lw_bfloat16  lw_bfloat16;
typedef __lw_bfloat162 lw_bfloat162;

#endif /* defined(__cplusplus) && !defined(LWDA_NO_BFLOAT16) */
 
#if defined(__CPP_VERSION_AT_LEAST_11_BF16)
#undef __CPP_VERSION_AT_LEAST_11_BF16
#endif /* defined(__CPP_VERSION_AT_LEAST_11_BF16) */

#endif /* end of include guard: __LWDA_BF16_HPP__ */
