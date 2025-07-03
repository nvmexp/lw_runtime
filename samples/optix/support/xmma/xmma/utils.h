/***************************************************************************************************
 * Copyright (c) 2011-2019, LWPU CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are not permit-
 * ted.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL LWPU CORPORATION BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROLWREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
#pragma once

#include <xmma/xmma.h>
#include <xmma/numeric_types.h>
#include <xmma/named_barrier.h>

#include "xmma/hopper/emu/utmaldg.h"
#include "xmma/hopper/emu/stsm.h"
#include "xmma/hopper/emu/utmastg.h"

// CP ASYNC FEATURES ///////////////////////////////////////////////////////////////////////////////

#ifndef XMMA_INTERNAL_LWVM_ENABLED
#define XMMA_INTERNAL_LWVM_ENABLED 0
#endif

#ifndef XMMA_LWDA_RP2RP_ENABLED
#define XMMA_LWDA_RP2RP_ENABLED 0
#endif

#ifndef LWDA_CP_ASYNC_SUPPORTED
#define LWDA_CP_ASYNC_SUPPORTED ( \
    (__LWDACC_VER_MAJOR__ >= 11) || \
    ((__LWDACC_VER_MAJOR__ == 10) && (__LWDACC_VER_MINOR__ >= 2) && \
        defined(XMMA_INTERNAL_LWVM_ENABLED)) \
  )
#endif

#ifndef LWDA_CP_ASYNC_ENABLED
#define LWDA_CP_ASYNC_ENABLED LWDA_CP_ASYNC_SUPPORTED
#endif

#if LWDA_CP_ASYNC_ENABLED && defined(__LWDA_ARCH__) && (__LWDA_ARCH__ >= 800)
  #define LWDA_CP_ASYNC_ACTIVATED 1
#endif

#ifndef LWDA_CP_ASYNC_GROUP_POLICY_SUPPORTED
#define LWDA_CP_ASYNC_GROUP_POLICY_SUPPORTED \
    ((LWDA_CP_ASYNC_SUPPORTED) && (__LWDACC_VER_MAJOR__ >= 11))
#endif

#ifndef LWDA_CP_ASYNC_GROUP_POLICY_ENABLED
#define LWDA_CP_ASYNC_GROUP_POLICY_ENABLED LWDA_CP_ASYNC_GROUP_POLICY_SUPPORTED
#endif

#if LWDA_CP_ASYNC_GROUP_POLICY_ENABLED && defined(__LWDA_ARCH__) && (__LWDA_ARCH__ >= 800)
  #define LWDA_CP_ASYNC_GROUP_POLICY_ACTIVATED 1
#endif

#ifndef XMMA_CP_ASYNC_EXPLICIT_DESCRIPTORS_SUPPORTED
  #define XMMA_CP_ASYNC_EXPLICIT_DESCRIPTORS_SUPPORTED 0
#endif

#ifndef XMMA_L2_SECTOR_PROMOTION_SUPPORTED
#define XMMA_L2_SECTOR_PROMOTION_SUPPORTED 0
#endif

namespace xmma {

#if ((__LWDACC_VER_MAJOR__ >= 11) || (__LWDACC_VER_MAJOR__ == 10 && __LWDACC_VER_MINOR__ >= 2))
extern "C" {
    __device__ uint32_t __lwvm_get_smem_pointer(void *ptr);
}
#endif

#if ((defined(__LWDA_ARCH__) && (__LWDA_ARCH__ >= 900)) && \
    (__LWDACC_VER_MAJOR__ == 11 && __LWDACC_VER_MINOR__ >= 3))
extern "C" {
    __device__ void __lw_ptx_builtin_ocg_acqblk(void);
    __device__ void __lw_ptx_builtin_ocg_preexit(void);
}
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

// Certain build targets don't use C++11 flag, so we need these
// But a different namespace keeps it safe from interfering
namespace lwca {

    struct false_type {
        static constexpr bool value = false;
        constexpr operator bool() const noexcept { return value; }
    };

    struct true_type {
        static constexpr bool value = true;
        constexpr operator bool() const noexcept { return value; }
    };

    template< bool B, typename T, typename U >
    struct conditional { typedef T type; };


    template< typename T, typename U >
    struct conditional<false, T, U> { typedef U type; };

    template< typename T, typename U >
    struct is_same : public false_type {};

    template< typename T >
    struct is_same<T, T> : public true_type {};

} // namespace lwca

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ unsigned get_smem_pointer(void *ptr) {
#if (defined(__LWDA_ARCH__) && \
     ((__LWDACC_VER_MAJOR__ >= 11 ) || (__LWDACC_VER_MAJOR__ == 10 && __LWDACC_VER_MINOR__ >= 2)))
    return __lwvm_get_smem_pointer(ptr);
#else

  uint32_t smem_ptr;

  asm(
  "{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 %0, smem_ptr; }\n"
    : "=r"(smem_ptr) : "l"(ptr));

  return smem_ptr;
#endif

}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline __device__ char* align_128(char *ptr) {
    uint64_t address_bit = reinterpret_cast<uint64_t>(ptr);
    uint64_t offset = address_bit % 128;
    if(offset == 0) {
        return ptr;
    } else {
        return ptr + (128 - offset);
    }
}


////////////////////////////////////////////////////////////////////////////////////////////////////

// The default value for Ampere memory descriptor.
static const uint64_t AMPERE_MEM_DESC_DEFAULT = uint64_t(0x1000000000000000ul);
// The default value for memory descriptor.
static const uint64_t MEM_DESC_DEFAULT = AMPERE_MEM_DESC_DEFAULT;

////////////////////////////////////////////////////////////////////////////////////////////////////

typedef enum { CE_DEFAULT = 0, CE_LDGSTS, CE_UTMALDG } Copy_engine;

template <
    // The instruction traits.
    typename Traits,
    // The class with LDG+STS.
    typename Class_with_ldg_and_sts,
    // The class with LDGSTS.
    typename Class_with_ldgsts,
    // The class with UTMALDG.
    typename Class_with_utmaldg,
    // Copy engine
    Copy_engine COPY_ENGINE,
    // Do we use UTMALDG
    bool USE_UTMALDG = Traits::Gpu_arch::HAS_UTMALDG &&COPY_ENGINE == Copy_engine::CE_UTMALDG,
    // Do we use LDGSTS?
    bool USE_LDGSTS = Traits::Gpu_arch::HAS_LDGSTS &&COPY_ENGINE == Copy_engine::CE_LDGSTS>
struct Copy_engine_selector {
    using Class = Class_with_ldg_and_sts;
};

template <
    // The instruction traits.
    typename Traits,
    // The class with LDG+STS.
    typename Class_with_ldg_and_sts,
    // The class with LDGSTS.
    typename Class_with_ldgsts,
    // The class with UTMALDG.
    typename Class_with_utmaldg>
struct Copy_engine_selector<Traits,
                            Class_with_ldg_and_sts,
                            Class_with_ldgsts,
                            Class_with_utmaldg,
                            Copy_engine::CE_UTMALDG,
                            true,
                            true> {
    using Class = Class_with_utmaldg;
};

template <
    // The instruction traits.
    typename Traits,
    // The class with LDG+STS.
    typename Class_with_ldg_and_sts,
    // The class with LDGSTS.
    typename Class_with_ldgsts,
    // The class with UTMALDG.
    typename Class_with_utmaldg>
struct Copy_engine_selector<Traits,
                            Class_with_ldg_and_sts,
                            Class_with_ldgsts,
                            Class_with_utmaldg,
                            Copy_engine::CE_UTMALDG,
                            true,
                            false> {
    using Class = Class_with_utmaldg;
};

template <
    // The instruction traits.
    typename Traits,
    // The class with LDG+STS.
    typename Class_with_ldg_and_sts,
    // The class with LDGSTS.
    typename Class_with_ldgsts,
    // The class with UTMALDG.
    typename Class_with_utmaldg>
struct Copy_engine_selector<Traits,
                            Class_with_ldg_and_sts,
                            Class_with_ldgsts,
                            Class_with_utmaldg,
                            Copy_engine::CE_LDGSTS,
                            false,
                            true> {
    using Class = Class_with_ldgsts;
};

////////////////////////////////////////////////////////////////////////////////////////////////////


template< int N_, int H_, int W_ >
struct Tile_nhw {
    enum { N = N_, H = H_, W = W_ };
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< int N_, int D_, int H_, int W_ >
struct Tile_ndhw {
    enum { N = N_, D = D_, H = H_, W = W_ };
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< int M, bool = (M & (M-1)) == 0 >
struct Next_power_of_two {
};

template< int M >
struct Next_power_of_two<  M, true > { enum { VALUE =   M }; };
template<>
struct Next_power_of_two<  3, false> { enum { VALUE =   4 }; };
template<>
struct Next_power_of_two<  5, false> { enum { VALUE =   8 }; };
template<>
struct Next_power_of_two<  6, false> { enum { VALUE =   8 }; };
template<>
struct Next_power_of_two<  7, false> { enum { VALUE =   8 }; };
template<>
struct Next_power_of_two<  9, false> { enum { VALUE =  16 }; };
template<>
struct Next_power_of_two< 10, false> { enum { VALUE =  16 }; };
template<>
struct Next_power_of_two< 11, false> { enum { VALUE =  16 }; };
template<>
struct Next_power_of_two< 12, false> { enum { VALUE =  16 }; };
template<>
struct Next_power_of_two< 13, false> { enum { VALUE =  16 }; };
template<>
struct Next_power_of_two< 14, false> { enum { VALUE =  16 }; };
template<>
struct Next_power_of_two< 15, false> { enum { VALUE =  16 }; };
template<>
struct Next_power_of_two< 24, false> { enum { VALUE =  32 }; };
template<>
struct Next_power_of_two< 48, false> { enum { VALUE =  64 }; };
template<>
struct Next_power_of_two< 80, false> { enum { VALUE = 128 }; };
template<>
struct Next_power_of_two< 96, false> { enum { VALUE = 128 }; };
template<>
struct Next_power_of_two<112, false> { enum { VALUE = 128 }; };
template<>
struct Next_power_of_two<144, false> { enum { VALUE = 256 }; };

////////////////////////////////////////////////////////////////////////////////////////////////////

template< int M, int N >
struct Div_up {
    enum { VALUE = (M + N-1) / N };
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< int A, int B >
struct Max {
    enum { VALUE = A >= B ? A : B };
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< int A, int B >
struct Min {
    enum { VALUE = A <= B ? A : B };
};

////////////////////////////////////////////////////////////////////////////////////////////////////

inline int clz(int x) {
    for( int i = 31; i >= 0; --i ) {
        if( (1 << i) & x ) {
            return 31 - i;
        }
    }
    return 32;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline int find_log_2(int x, bool round_up = false) {
    int a = 31 - clz(x);
    if( round_up ) {
        a += (x & (x-1)) ? 1 : 0;
    }
    return a;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline void find_divisor(uint32_t &mul, uint32_t &shr, int x) {
    assert(x != 0);
    if( x == 1 ) {
        // If dividing by 1, reduced math doesn't work because mul_coeff would need to be 2^32,
        // which doesn't fit into unsigned int.  the div() routine handles this special case
        // separately.
        mul = 0;
        shr = 0;
    } else {
        // To express the division N/D in terms of a multiplication, what we first
        // imagine is simply N*(1/D).  However, 1/D will always evaluate to 0 (for D>1),
        // so we need another way.  There's nothing that says we have to use exactly
        // the fraction 1/D; instead it could be any X/Y that reduces to 1/D (i.e.,
        // Y=X*D), or at least to "close enough" to it.  If we pick Y that is a power
        // of two, then the N*(X/Y) can be N*X followed by a right-shift by some amount.
        // The power of two we should pick should be at least 2^32, because in the
        // div() routine we'll use umulhi(), which returns only the upper 32 bits --
        // this being equivalent to a right-shift by 32.  But we might want a higher
        // power of two for better accuracy depending on the magnitude of the denominator.
        // Once we've picked Y, then X [our mul_coeff value] is simply Y/D, rounding up,
        // and we save shift_coeff as whatever further shift we have to do beyond
        // what the umulhi() implies.
        uint32_t p = 31 + find_log_2(x, true);
        uint32_t m = (uint32_t)(((1ull << p) + (uint32_t) x - 1) / (uint32_t) x);

        mul = m;
        shr = p - 32;
    }
}

inline void find_divisor_v2(uint32_t &mul, uint32_t &shr, int x) {

    uint32_t p = 31 + find_log_2(2*x, true);
    uint32_t m = (uint32_t)(((1ull << p) + (uint32_t) (2*x) - 1) / (uint32_t) (2*x));

    mul = m;
    shr = p - 32;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void fast_divmod(int &div, int &mod, int x, int y, uint32_t mul, uint32_t shr) {
    if( y == 1 ) {
        div = x;
        mod = 0;
    } else {
        div = __umulhi((uint32_t) x, mul) >> shr;
        mod = x - div*y;
    }
}

inline __device__ void fast_divmod_v2(uint32_t &div, uint32_t &mod, int x, int y, uint32_t mul, uint32_t shr) {
    div = __umulhi((uint32_t) 2*x, mul) >> shr;
    mod = x - div*y;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename T >
inline __device__ T clamp(T x, T lb, T ub) {
  return x < lb ? lb : (x > ub ? ub : x);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
inline __device__ int32_t idp_4a(const int32_t a, const int32_t b, const int32_t c) {
    int32_t r;

    asm volatile("dp4a.s32.s32 %0, %1, %2, %3;\n" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    return r;
}

inline __device__ float4 s8x4_to_float4(uint32_t x) {
#if defined __LWDA_ARCH__ && __LWDA_ARCH__ <= 800
    int32_t t[4];
    float4 f;

    t[0] = __byte_perm(0, x, 0x4) ^ 0x4b7fff80;
    t[1] = __byte_perm(0, x, 0x5) ^ 0x4b7fff80;
    t[2] = __byte_perm(0, x, 0x6) ^ 0x4b7fff80;
    t[3] = __byte_perm(0, x, 0x7) ^ 0x4b7fff80;

    f.x = (reinterpret_cast<float&>(t[0]) - 16777088.f);
    f.y = (reinterpret_cast<float&>(t[1]) - 16777088.f);
    f.z = (reinterpret_cast<float&>(t[2]) - 16777088.f);
    f.w = (reinterpret_cast<float&>(t[3]) - 16777088.f);
    return f;
#else
    int32_t c = 0;
    int32_t t[4];
    const int32_t b3 = 0x01000000;
    const int32_t b2 = 0x00010000;
    const int32_t b1 = 0x00000100;
    const int32_t b0 = 0x00000001;
    float4 f;

    t[0] = idp_4a(x, b0, c);
    t[1] = idp_4a(x, b1, c);
    t[2] = idp_4a(x, b2, c);
    t[3] = idp_4a(x, b3, c);

    f.x = (float)(t[0]);
    f.y = (float)(t[1]);
    f.z = (float)(t[2]);
    f.w = (float)(t[3]);
    return f;
#endif

}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ uint32_t float_to_tf32_rn(float x) {
    uint32_t tmp = reinterpret_cast<const uint32_t&>(x);
    if( (tmp & 0x7f800000) != 0x7f800000 ) {

        uint32_t mantissa_bit = tmp & 0x00002000u; // (1 << 13)
        uint32_t round_bit    = tmp & 0x00001000u; // (1 << 12)
        uint32_t sticky_bit   = tmp & 0x00000fffu; // ((1 << 12) - 1)

        if( (round_bit && sticky_bit) || (round_bit && mantissa_bit) ) {
            tmp += 0x00002000u;
        }
    } else if( tmp & ~0xff800000 ) {
        tmp = 0x7fffffff;
    }
    return tmp;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ uint32_t float2_to_half2(float a, float b) {
    uint32_t c;
#if !defined(__LWDACC_RTC__) && (__LWDA_ARCH__ >= 750 && \
            ((__LWDACC_VER_MAJOR__ >= 11) || \
             (__LWDACC_VER_MAJOR__ == 10 && __LWDACC_VER_MINOR__ >= 2)))
    asm volatile("cvt.rn.f16x2.f32 %0, %1, %2;\n" : "=r"(c) : "f"(b), "f"(a));
#else
    asm volatile( \
        "{\n" \
        "    .reg .f16 lo, hi;\n" \
        "    cvt.rn.f16.f32 lo, %1;\n" \
        "    cvt.rn.f16.f32 hi, %2;\n" \
        "    mov.b32 %0, {lo, hi};\n" \
        "}\n" : "=r"(c) : "f"(a), "f"(b));
#endif
    return c;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline __device__ uint32_t float2_to_bf16_2(float a, float b) {
    uint32_t c;
#if !defined(__LWDACC_RTC__) && __LWDA_ARCH__ >= 750
    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;\n" : "=r"(c) : "f"(b), "f"(a));
#else
    uint16_t* px = reinterpret_cast<uint16_t*>(&a);
    uint16_t* py = reinterpret_cast<uint16_t*>(&b);
    uint16_t value = px[1];
    uint16_t value2 = py[1];

    if (px[0] == 0x8000) {
       if ((value & 0x1) == 1)
            value++;
    } else if (px[0] > 0x8000) {
        value++;
    }

    if (py[0] == 0x8000) {
        if ((value2 & 0x1) == 1)
            value2++;
    } else if (py[0] > 0x8000) {
        value2++;
    }

    uint32_t high = reinterpret_cast<uint32_t &>(value2);
    c = (high << 16) | value;
#endif
    return c;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ uint32_t hadd2(uint32_t a, uint32_t b) {
    uint32_t c;
    asm volatile("add.f16x2 %0, %1, %2;\n" : "=r"(c) : "r"(a), "r"(b));
    return c;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ uint32_t bfadd2(uint32_t a, uint32_t b) {
    uint32_t c;
    uint32_t one = 0x3f803f80;;
    asm volatile("fma.rn.bf16x2 %0, %1, %3, %2;\n" : "=r"(c) : "r"(a), "r"(b), "r"(one));
    return c;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ uint2 hadd4(uint2 a, uint2 b) {
    uint2 c;
    c.x = hadd2(a.x, b.x);
    c.y = hadd2(a.y, b.y);
    return c;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ uint2 hadd4(uint32_t a, uint2 b) {
    uint2 c;
    asm volatile("add.f16x2 %0, %1, %2;\n" : "=r"(c.x) : "r"(a), "r"(b.x));
    asm volatile("add.f16x2 %0, %1, %2;\n" : "=r"(c.y) : "r"(a), "r"(b.y));
    return c;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ uint4 hadd8(uint32_t a, uint4 b) {
    uint4 c;
    asm volatile("add.f16x2 %0, %1, %2;\n" : "=r"(c.x) : "r"(a), "r"(b.x));
    asm volatile("add.f16x2 %0, %1, %2;\n" : "=r"(c.y) : "r"(a), "r"(b.y));
    asm volatile("add.f16x2 %0, %1, %2;\n" : "=r"(c.z) : "r"(a), "r"(b.z));
    asm volatile("add.f16x2 %0, %1, %2;\n" : "=r"(c.w) : "r"(a), "r"(b.w));
    return c;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ uint4 hadd8(uint4 a, uint4 b) {
    uint4 c;
    asm volatile("add.f16x2 %0, %1, %2;\n" : "=r"(c.x) : "r"(a.x), "r"(b.x));
    asm volatile("add.f16x2 %0, %1, %2;\n" : "=r"(c.y) : "r"(a.y), "r"(b.y));
    asm volatile("add.f16x2 %0, %1, %2;\n" : "=r"(c.z) : "r"(a.z), "r"(b.z));
    asm volatile("add.f16x2 %0, %1, %2;\n" : "=r"(c.w) : "r"(a.w), "r"(b.w));
    return c;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ uint32_t half_to_half2(lwtlass::half_t x) {
    uint32_t res;
    asm volatile( \
        "{\n" \
        "    mov.b32 %0, {%1, %1};\n" \
        "}\n" : "=r"(res) : "h"(x));
    return res;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float half_to_float(lwtlass::half_t x) {
    float res;
    asm volatile( \
        "{\n" \
        "    cvt.f32.f16 %0, %1;\n" \
        "}\n" : "=f"(res) : "h"(x));
    return res;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float bf16_to_float(lwtlass::float_bf16_t x) {
    float res;
    asm volatile( \
        "{\n" \
        "    mov.b32 %0, {0, %1};\n" \
        "}\n" : "=f"(res) : "h"(x));
    return res;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float2 bf16_2_to_float2(uint32_t x) {
    float2 res;
    asm volatile( \
        "{\n" \
        "    .reg .b16 lo, hi;\n" \
        "    mov.b32 {lo, hi}, %2;\n" \
        "    mov.b32 %0, {0, lo};\n" \
        "    mov.b32 %1, {0, hi};\n" \
        "}\n" : "=f"(res.x), "=f"(res.y) : "r"(x));
    return res;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline __device__ void unpack_half2(uint32_t x, uint16_t &half_lo, uint16_t &half_hi) {
    asm volatile( \
        "{\n" \
        "    mov.b32 {%0, %1}, %2;\n" \
        "}\n" : "=h"(half_lo), "=h"(half_hi) : "r"(x));
    return;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float2 half2_to_float2(uint32_t x) {
    float2 res;
    asm volatile( \
        "{\n" \
        "    .reg .f16 lo, hi;\n" \
        "    mov.b32 {lo, hi}, %2;\n" \
        "    cvt.f32.f16 %0, lo;\n" \
        "    cvt.f32.f16 %1, hi;\n" \
        "}\n" : "=f"(res.x), "=f"(res.y) : "r"(x));
    return res;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ uint32_t hfma2(uint32_t a, uint32_t b, uint32_t c) {
    uint32_t d;
    asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(d) : "r"(a), "r"(b), "r"(c));
    return d;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ uint32_t bfma2(uint32_t a, uint32_t b, uint32_t c) {
    uint32_t d;
    asm volatile("fma.rn.bf16x2 %0, %1, %2, %3;\n" : "=r"(d) : "r"(a), "r"(b), "r"(c));
    return d;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ uint2 hfma4(uint32_t a, uint2 b, uint2 c) {
    uint2 d;
    asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(d.x) : "r"(a), "r"(b.x), "r"(c.x));
    asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(d.y) : "r"(a), "r"(b.y), "r"(c.y));
    return d;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ uint4 hfma8(uint32_t a, uint4 b, uint4 c) {
    uint4 d;
    asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(d.x) : "r"(a), "r"(b.x), "r"(c.x));
    asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(d.y) : "r"(a), "r"(b.y), "r"(c.y));
    asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(d.z) : "r"(a), "r"(b.z), "r"(c.z));
    asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(d.w) : "r"(a), "r"(b.w), "r"(c.w));
    return d;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ uint32_t hmax2(uint32_t a, uint32_t b) {
    uint32_t c;
    asm volatile( \
        "{\n" \
        "\t .reg .f16x2 sela, selb;\n" \
        "\n" \
        "\t set.ge.f16x2.f16x2 sela, %1, %2;\n" \
        "\t set.gt.f16x2.f16x2 selb, %2, %1;\n" \
        "\n" \
        "\t mul.f16x2 %0, sela, %1;\n" \
        "\t fma.rn.f16x2 %0, selb, %2, %0;\n" \
        "}\n" : "=r"(c) : "r"(a), "r"(b));
    return c;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ uint2 hmax4(uint2 a, uint2 b) {
    uint2 c;
    c.x = hmax2(a.x, b.x);
    c.y = hmax2(a.y, b.y);
    return c;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ uint4 hmax8(uint4 a, uint4 b) {
    uint4 c;
    c.x = hmax2(a.x, b.x);
    c.y = hmax2(a.y, b.y);
    c.z = hmax2(a.z, b.z);
    c.w = hmax2(a.w, b.w);
    return c;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ uint32_t hmul2(uint32_t a, uint32_t b) {
    uint32_t c;
    asm volatile("mul.f16x2 %0, %1, %2;\n" : "=r"(c) : "r"(a), "r"(b));
    return c;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ uint2 hmul4(uint32_t a, uint2 b) {
    uint2 c;
    asm volatile("mul.f16x2 %0, %1, %2;\n" : "=r"(c.x) : "r"(a), "r"(b.x));
    asm volatile("mul.f16x2 %0, %1, %2;\n" : "=r"(c.y) : "r"(a), "r"(b.y));
    return c;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ uint4 hmul8(uint32_t a, uint4 b) {
    uint4 c;
    asm volatile("mul.f16x2 %0, %1, %2;\n" : "=r"(c.x) : "r"(a), "r"(b.x));
    asm volatile("mul.f16x2 %0, %1, %2;\n" : "=r"(c.y) : "r"(a), "r"(b.y));
    asm volatile("mul.f16x2 %0, %1, %2;\n" : "=r"(c.z) : "r"(a), "r"(b.z));
    asm volatile("mul.f16x2 %0, %1, %2;\n" : "=r"(c.w) : "r"(a), "r"(b.w));
    return c;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ int32_t f2i(float a) {
    int32_t res;
#if (defined(__LWDA_ARCH__) &&  \
    ((__LWDACC_VER_MAJOR__ >= 11) || (__LWDACC_VER_MAJOR__ == 10 && __LWDACC_VER_MINOR__ >= 2)))
    asm volatile("cvt.rni.sat.s32.f32 %0, %1;" : "=r"(res) : "f"(a));
#else
    asm volatile("cvt.rni.sat.s8.f32 %0, %1;" : "=r"(res) : "f"(a));
#endif
    return res;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float i2f( int32_t a ) {
    float res;
    asm volatile( "cvt.rn.f32.s32 %0, %1;" : "=f"( res ) : "r"( a ) );
    return res;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ int32_t pack_int8x4(int32_t (&a)[4]) {
    int32_t res;
#if (defined(__LWDA_ARCH__) && (__LWDA_ARCH__ >= 720) && \
    ((__LWDACC_VER_MAJOR__ >= 11) || (__LWDACC_VER_MAJOR__ == 10 && __LWDACC_VER_MINOR__ >= 2)))
    asm volatile(
        "{\n" \
        ".reg .u32 r4;\n" \
        "cvt.pack.sat.s8.s32.b32   r4, %4, %3,  0;\n" \
        "cvt.pack.sat.s8.s32.b32   %0, %2, %1, r4;\n" \
        "}" \
        : "=r"(res) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]));
#else
    asm volatile(
        "{\n" \
        "prmt.b32 %2, %1, %2, 0x1140;\n" \
        "prmt.b32 %4, %3, %4, 0x1140;\n" \
        "prmt.b32 %0, %2, %4, 0x5410;\n" \
        "}" \
        : "=r"(res) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]));
#endif
    return res;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ uint32_t float4_to_s8x4( float4 in ) {
    uint32_t ret;
#if defined( __LWDA_ARCH__ ) && ( __LWDA_ARCH__ <= 800 )
    // Users need to guarantee the value in [-128.f, 127.f].
    float x = in.x + 12582912.f;
    float y = in.y + 12582912.f;
    float z = in.z + 12582912.f;
    float w = in.w + 12582912.f;

    uint32_t reg_lo =
        __byte_perm( reinterpret_cast<uint32_t &>( x ), reinterpret_cast<uint32_t &>( y ), 0x40 );
    uint32_t reg_hi =
        __byte_perm( reinterpret_cast<uint32_t &>( z ), reinterpret_cast<uint32_t &>( w ), 0x40 );
    ret = __byte_perm( reg_lo, reg_hi, 0x5410 );
#else
    int32_t tmp[4];
    tmp[0] = f2i( in.x );
    tmp[1] = f2i( in.y );
    tmp[2] = f2i( in.z );
    tmp[3] = f2i( in.w );
    ret = pack_int8x4( tmp );
#endif
    return ret;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// C L E A R   R A W   D A T A   C H U N K S
//
////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void clear(uint8_t &dst) {
    dst = uint8_t(0);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void clear(uint16_t &dst) {
    dst = uint16_t(0);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void clear(uint32_t &dst) {
    dst = uint32_t(0);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void clear(uint2 &dst) {
    dst = make_uint2(0u, 0u);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void clear(uint4 &dst) {
    dst = make_uint4(0u, 0u, 0u, 0u);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Data_type >
inline __device__ void clear_if_pnz_is_not_supported(Data_type &dst) {
#if __LWDA_ARCH__ <= 750
    clear(dst);
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// P R E D I C A T E   P A C K I N G
//
////////////////////////////////////////////////////////////////////////////////////////////////////

enum { BYTES_PER_REG = 4, PREDS_PER_BYTE = 4, PREDS_PER_REG = BYTES_PER_REG * PREDS_PER_BYTE };

////////////////////////////////////////////////////////////////////////////////////////////////////

template< int LDGS >
struct Compute_number_of_pred_regs {
    enum { VALUE = Div_up<LDGS, PREDS_PER_REG>::VALUE };
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< int M, int N >
inline __device__ void pack_predicates(uint32_t (&preds)[M], const uint32_t (&p)[N]) {

    // Make sure the values match.
    static_assert(Compute_number_of_pred_regs<N>::VALUE == M, "");

    // The number of complete steps (where we use all the predicates in a byte).
    enum { COMPLETE_BYTES = N / PREDS_PER_BYTE };
    // Make sure we allocated enough predicate registers.
    static_assert(Div_up<COMPLETE_BYTES, BYTES_PER_REG>::VALUE <= M, "");
    // The remainder.
    enum { REMAINDER = N - COMPLETE_BYTES * PREDS_PER_BYTE };
    // Make sure we got the math right and the remainder is between 0 and 3.
    static_assert(REMAINDER >= 0 && REMAINDER <= 3, "");
    // The mask to extract the predicates.
    enum { COMPLETE_MASK = (1 << PREDS_PER_BYTE) - 1 };

    // Run complete steps.
    //
    // NOTE (Julien): I use a separate "reg" variable to WAR a compiler limitation (or what seems
    // to be). The compiler does not seem to be able to deal with
    //
    // __lw_p2r(X, tmp, MASK, &preds[ii / BYTES_PER_REG]);
    //
    // where I would take the correct predicate with the "&preds[...]" syntax.

    #pragma unroll
    for( int ii = 0; ii < M; ++ii ) {

        // The number of complete bytes for that register. Be careful it can be > than 4 ;)
        const int COMPLETE = (N - ii * PREDS_PER_REG) / PREDS_PER_BYTE;

        // Pack the predicates in a register.
        uint32_t reg = 0u;
        #pragma unroll
        for( int jj = 0; jj < 4; ++jj ) {

            // Early exit.
            if( jj >= COMPLETE ) {
                break;
            }

            // Prepare the array of predicates.
            bool tmp[PREDS_PER_BYTE];
            #pragma unroll
            for( int kk = 0; kk < PREDS_PER_BYTE; ++kk ) {
                tmp[kk] = p[ii * PREDS_PER_REG + jj * PREDS_PER_BYTE + kk] != 0;
            }

#if (!defined(__LWDACC_RTC__) && defined(XMMA_LWDA_RP2RP_ENABLED) && XMMA_LWDA_RP2RP_ENABLED == 1)
            // Store the predicates.
            if( jj == 0 ) {
                __lw_p2r(0, tmp, COMPLETE_MASK, &reg);
            } else if( jj == 1 ) {
                __lw_p2r(1, tmp, COMPLETE_MASK, &reg);
            } else if( jj == 2 ) {
                __lw_p2r(2, tmp, COMPLETE_MASK, &reg);
            } else if( jj == 3 ) {
                __lw_p2r(3, tmp, COMPLETE_MASK, &reg);
            }
#else
            #pragma unroll
            for( int kk = 0; kk < PREDS_PER_BYTE; ++kk ) {
              reg |= (tmp[kk] ? 1u : 0u) << (jj*8 + kk);
            }
#endif
        }

        // Skip the rest of the code if we do not have a remainder.
        if( COMPLETE < 4 && REMAINDER > 0 ) {

            // The mask to extract the predicates.
            enum { REMAINDER_MASK = (1 << REMAINDER) - 1 };

            // Prepare the array of predicates.
            bool tmp[PREDS_PER_BYTE];
            #pragma unroll
            for( int jj = 0; jj < REMAINDER; ++jj ) {
                tmp[jj] = p[COMPLETE_BYTES * PREDS_PER_BYTE + jj] != 0;
            }

#if (!defined(__LWDACC_RTC__) && defined(XMMA_LWDA_RP2RP_ENABLED) && XMMA_LWDA_RP2RP_ENABLED == 1)
            // Store the predicates.
            if( COMPLETE == 0 ) {
                __lw_p2r(0, tmp, REMAINDER_MASK, &reg);
            } else if( COMPLETE == 1 ) {
                __lw_p2r(1, tmp, REMAINDER_MASK, &reg);
            } else if( COMPLETE == 2 ) {
                __lw_p2r(2, tmp, REMAINDER_MASK, &reg);
            } else if( COMPLETE == 3 ) {
                __lw_p2r(3, tmp, REMAINDER_MASK, &reg);
            }
#else
            #pragma unroll
            for( int jj = 0; jj < REMAINDER; ++jj ) {
              reg |= (tmp[jj] ? 1u : 0u) << (COMPLETE*8 + jj);
            }
#endif
        }

        // Store the predicate register.
        preds[ii] = reg;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template< int N >
inline __device__ uint32_t pack_predicates(const uint32_t (&p)[N]) {
    uint32_t tmp[1];
    pack_predicates(tmp, p);
    return tmp[0];
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// G E N E R I C   P R E D I C A T E D   L D G S T S
//
////////////////////////////////////////////////////////////////////////////////////////////////////

template< int N, int M, typename Functor >
inline __device__ void ldgsts_(Functor &fct, const uint32_t (&preds)[M], uint64_t mem_desc) {

    // The number of complete bytes (where we use all the predicates in a byte).
    enum { COMPLETE = N / PREDS_PER_BYTE };
    // Make sure we did allocate enough predicates.
    static_assert(Div_up<COMPLETE, BYTES_PER_REG>::VALUE <= M, "");
    // The remainder.
    enum { REMAINDER = N - COMPLETE * PREDS_PER_BYTE };
    // Make sure we got the math right and the remainder is between 0 and 3.
    static_assert(REMAINDER >= 0 && REMAINDER <= 3, "");
    // The mask to extract the predicates.
    enum { COMPLETE_MASK = (1 << PREDS_PER_BYTE) - 1 };

    // Clear the fetch registers.
    #pragma unroll
    for( int ii = 0; ii < N; ++ii ) {
        fct.clear(ii);
    }

    // Run complete steps.
    bool p[PREDS_PER_BYTE];
    #pragma unroll
    for( int ii = 0; ii < COMPLETE; ++ii ) {

        // The predicate.
        uint32_t reg = preds[ii / BYTES_PER_REG];

#if (!defined(__LWDACC_RTC__) && defined(XMMA_LWDA_RP2RP_ENABLED) && XMMA_LWDA_RP2RP_ENABLED == 1)
        // Extract the predicates.
        if( ii % BYTES_PER_REG == 0 ) {
            __lw_r2p(0, p, COMPLETE_MASK, reg);
        } else if( ii % BYTES_PER_REG == 1 ) {
            __lw_r2p(1, p, COMPLETE_MASK, reg);
        } else if( ii % BYTES_PER_REG == 2 ) {
            __lw_r2p(2, p, COMPLETE_MASK, reg);
        } else if( ii % BYTES_PER_REG == 3 ) {
            __lw_r2p(3, p, COMPLETE_MASK, reg);
        }
#else
        const uint32_t R2P_MASK = 1u << (ii % BYTES_PER_REG * 8);
        #pragma unroll
        for( int jj = 0; jj < PREDS_PER_BYTE; ++jj ) {
          p[jj] = reg & (R2P_MASK << jj);
        }
#endif

        // Issue the loads.
        #pragma unroll
        for( int jj = 0; jj < PREDS_PER_BYTE; ++jj ) {
            fct.ldgsts(ii * PREDS_PER_BYTE + jj, p[jj], mem_desc);
        }
    }

    // Skip the rest of the code if we do not have a remainder.
    if( REMAINDER > 0 ) {

        // The mask to extract the predicates.
        enum { REMAINDER_MASK = (1 << REMAINDER) - 1 };

        // The predicate register.
        uint32_t reg = preds[COMPLETE / BYTES_PER_REG];

#if (!defined(__LWDACC_RTC__) && defined(XMMA_LWDA_RP2RP_ENABLED) && XMMA_LWDA_RP2RP_ENABLED == 1)
        // Extract the predicates.
        if( COMPLETE % BYTES_PER_REG == 0 ) {
            __lw_r2p(0, p, REMAINDER_MASK, reg);
        } else if( COMPLETE % BYTES_PER_REG == 1 ) {
            __lw_r2p(1, p, REMAINDER_MASK, reg);
        } else if( COMPLETE % BYTES_PER_REG == 2 ) {
            __lw_r2p(2, p, REMAINDER_MASK, reg);
        } else if( COMPLETE % BYTES_PER_REG == 3 ) {
            __lw_r2p(3, p, REMAINDER_MASK, reg);
        }
#else
        const uint32_t R2P_MASK = 1u << (COMPLETE % BYTES_PER_REG * 8);
        #pragma unroll
        for( int ii = 0; ii < REMAINDER; ++ii ) {
          p[ii] = reg & (R2P_MASK << ii);
        }
#endif

        // Issue the loads.
        #pragma unroll
        for( int ii = 0; ii < REMAINDER; ++ii ) {
            fct.ldgsts(COMPLETE * PREDS_PER_BYTE + ii, p[ii], mem_desc);
        }
    }
}

template< int N, int PHASE, int M, typename Functor >
inline __device__ void ldgsts_per_phase_(Functor &fct, const uint32_t (&preds)[M], uint64_t mem_desc) {

    // The number of complete bytes (where we use all the predicates in a byte).
    enum { COMPLETE_0 = N / PREDS_PER_BYTE };
    // Make sure we did allocate enough predicates.
    static_assert(Div_up<COMPLETE_0, BYTES_PER_REG>::VALUE <= M, "");
    // The remainder.
    enum { REMAINDER = N - COMPLETE_0 * PREDS_PER_BYTE };
    // Make sure we got the math right and the remainder is between 0 and 3.
    static_assert(REMAINDER >= 0 && REMAINDER <= 3, "");
    // The mask to extract the predicates.
    enum { COMPLETE_MASK = (1 << PREDS_PER_BYTE) - 1 };
    constexpr int START = PHASE * N;
    constexpr int START_REG = START / 16;
    constexpr int START_BYTE = START % 16 / 4;
    constexpr int REMINDER_0 = START % 16 % 4;
    constexpr int COMPLETE = Max<N - REMINDER_0, 0>::VALUE / 4;
    constexpr int COMPLETE_REG = (START + REMINDER_0) / 16;
    //constexpr int complete_byte = (start + REMINDER_0) % 16 / 4;
    int complete_byte = (START + REMINDER_0) % 16 / 4;
    constexpr int REMINDER_1 = Max<N - COMPLETE * 4 - REMINDER_0,0>::VALUE;
    constexpr int REMINDER_REG = (START + REMINDER_0 + COMPLETE * 4)/16;
    constexpr int REMINDER_BYTE = (START + REMINDER_0 + COMPLETE * 4) % 16 / 4;


    // Clear the fetch registers.
    #pragma unroll
    for( int ii = 0; ii < N; ++ii ) {
        fct.clear(ii);
    }

    // Run complete steps.
    bool p[PREDS_PER_BYTE];

    if( REMINDER_0 > 0 ) {

        // The mask to extract the predicates.
        enum { REMAINDER_MASK = ((1 << (N % 4)) - 1) << REMINDER_0 };
        // The predicate register.
        uint32_t reg = preds[START_REG];

#if (!defined(__LWDACC_RTC__) && defined(XMMA_LWDA_RP2RP_ENABLED) && XMMA_LWDA_RP2RP_ENABLED == 1)
        // Extract the predicates.
        if( START_BYTE  == 0 ) {
            __lw_r2p(0, p, REMAINDER_MASK, reg);
        } else if( START_BYTE == 1 ) {
            __lw_r2p(1, p, REMAINDER_MASK, reg);
        } else if( START_BYTE == 2 ) {
            __lw_r2p(2, p, REMAINDER_MASK, reg);
        } else if( START_BYTE == 3 ) {
            __lw_r2p(3, p, REMAINDER_MASK, reg);
        }
#else
        const uint32_t R2P_MASK = 1u << (START_BYTE * 8);
        #pragma unroll
        for( int ii = 0; ii < REMINDER_0; ++ii ) {
          p[ii] = reg & (R2P_MASK << ii);
        }
#endif

        // Issue the loads.
        #pragma unroll
        for( int ii = 0; ii < N; ++ii ) {
            fct.ldgsts(ii, p[ii], mem_desc);
        }
    }

    if(COMPLETE > 0){
        #pragma unroll
        for( int ii = 0; ii < COMPLETE; ++ii ) {

            // The predicate.
            uint32_t reg = preds[COMPLETE_REG];

#if (!defined(__LWDACC_RTC__) && defined(XMMA_LWDA_RP2RP_ENABLED) && XMMA_LWDA_RP2RP_ENABLED == 1)
            // Extract the predicates.
            if( complete_byte == 0 ) {
                __lw_r2p(0, p, COMPLETE_MASK, reg);
            } else if( complete_byte == 1 ) {
                __lw_r2p(1, p, COMPLETE_MASK, reg);
            } else if( complete_byte == 2 ) {
                __lw_r2p(2, p, COMPLETE_MASK, reg);
            } else if( complete_byte == 3 ) {
                __lw_r2p(3, p, COMPLETE_MASK, reg);
            }

            if(COMPLETE > 1) {
                complete_byte=(complete_byte + 1) % 4;
            }
#else
            const uint32_t R2P_MASK = 1u << (complete_byte * 8);
#pragma unroll
            for( int jj = 0; jj < PREDS_PER_BYTE; ++jj ) {
                p[jj] = reg & (R2P_MASK << jj);
            }
#endif

            // Issue the loads.
#pragma unroll
            for( int jj = 0; jj < PREDS_PER_BYTE; ++jj ) {
                fct.ldgsts(REMINDER_0 + ii * PREDS_PER_BYTE + jj, p[jj], mem_desc);
            }
        }
    }

    // Skip the rest of the code if we do not have a remainder.
    if( REMINDER_1 > 0 ) {

        // The mask to extract the predicates.
        enum { REMAINDER_MASK = (1 << REMINDER_1) - 1 };

        // The predicate register.
        uint32_t reg = preds[REMINDER_REG];

#if (!defined(__LWDACC_RTC__) && defined(XMMA_LWDA_RP2RP_ENABLED) && XMMA_LWDA_RP2RP_ENABLED == 1)
        // Extract the predicates.
        if( REMINDER_BYTE == 0 ) {
            __lw_r2p(0, p, REMAINDER_MASK, reg);
        } else if( REMINDER_BYTE == 1 ) {
            __lw_r2p(1, p, REMAINDER_MASK, reg);
        } else if( REMINDER_BYTE == 2 ) {
            __lw_r2p(2, p, REMAINDER_MASK, reg);
        } else if( REMINDER_BYTE == 3 ) {
            __lw_r2p(3, p, REMAINDER_MASK, reg);
        }
#else
        const uint32_t R2P_MASK = 1u << (REMINDER_BYTE * 8);
        #pragma unroll
        for( int ii = 0; ii < REMINDER_1; ++ii ) {
          p[ii] = reg & (R2P_MASK << ii);
        }
#endif

        // Issue the loads.
        #pragma unroll
        for( int ii = 0; ii < REMINDER_1; ++ii ) {
            fct.ldgsts(REMINDER_0 + COMPLETE_0 * PREDS_PER_BYTE + ii, p[ii], mem_desc);
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template< int M, typename Functor >
inline __device__ void ldgsts_(Functor &fct, uint32_t preds, uint64_t mem_desc) {
    uint32_t tmp[1] = { preds };
    ldgsts_<M>(fct, tmp, mem_desc);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// L D G
//
////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void ldg(uint8_t &dst, const void *ptr, uint64_t mem_desc = MEM_DESC_DEFAULT) {
    dst = *reinterpret_cast<const uint8_t*>(ptr);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void ldg(uint16_t &dst, const void *ptr, uint64_t mem_desc = MEM_DESC_DEFAULT) {
#if defined(__LWDA_ARCH__) && __LWDA_ARCH__ >= 800 && XMMA_CP_ASYNC_EXPLICIT_DESCRIPTORS_SUPPORTED
    asm volatile(
        "ld.global.desc.b16 %0, [%1], %2;\n" \
            : "=h"(dst)
            :  "l"(ptr)
            ,  "l"(mem_desc));
#else
    dst = *reinterpret_cast<const uint16_t*>(ptr);
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void ldg(uint32_t &dst, const void *ptr, uint64_t mem_desc = MEM_DESC_DEFAULT) {
#if defined(__LWDA_ARCH__) && __LWDA_ARCH__ >= 800 && XMMA_CP_ASYNC_EXPLICIT_DESCRIPTORS_SUPPORTED
    asm volatile(
        "ld.global.desc.b32 %0, [%1], %2;\n" \
            : "=r"(dst)
            :  "l"(ptr)
            ,  "l"(mem_desc));
#else
    dst = *reinterpret_cast<const uint32_t*>(ptr);
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void ldg(uint2 &dst, const void *ptr, uint64_t mem_desc = MEM_DESC_DEFAULT) {
#if defined(__LWDA_ARCH__) && __LWDA_ARCH__ >= 800 && XMMA_CP_ASYNC_EXPLICIT_DESCRIPTORS_SUPPORTED
    asm volatile(
        "ld.global.desc.v2.b32 {%0, %1}, [%2], %3;\n" \
            : "=r"(dst.x)
            , "=r"(dst.y)
            :  "l"(ptr)
            ,  "l"(mem_desc));
#else
    dst = *reinterpret_cast<const uint2*>(ptr);
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void ldg(uint4 &dst, const void *ptr, uint64_t mem_desc = MEM_DESC_DEFAULT) {
#if defined(__LWDA_ARCH__) && __LWDA_ARCH__ >= 800 && XMMA_CP_ASYNC_EXPLICIT_DESCRIPTORS_SUPPORTED
    asm volatile(
        "ld.global.desc.v4.b32 {%0, %1, %2, %3}, [%4], %5;\n" \
            : "=r"(dst.x)
            , "=r"(dst.y)
            , "=r"(dst.z)
            , "=r"(dst.w)
            :  "l"(ptr)
            ,  "l"(mem_desc));
#else
    dst = *reinterpret_cast<const uint4*>(ptr);
#endif
}
////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void ldg_cs( uint4 &dst,
                               const char *ptr,
                               uint64_t mem_desc = MEM_DESC_DEFAULT ) {
#if defined(__LWDA_ARCH__) && __LWDA_ARCH__ >= 800 && XMMA_CP_ASYNC_EXPLICIT_DESCRIPTORS_SUPPORTED
    asm volatile(
        "ld.global.cs.desc.v4.b32 {%0, %1, %2, %3}, [%4], %5;\n" \
            : "=r"(dst.x)
            , "=r"(dst.y)
            , "=r"(dst.z)
            , "=r"(dst.w)
            :  "l"(ptr)
            ,  "l"(mem_desc));
#else
    asm volatile(
        "ld.global.cs.v4.b32 {%0, %1, %2, %3}, [%4];\n" \
            : "=r"(dst.x)
            , "=r"(dst.y)
            , "=r"(dst.z)
            , "=r"(dst.w)
            :  "l"(ptr));
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void ldg_with_pnz(uint8_t &dst, const void *ptr, bool p) {
    dst = p ? *reinterpret_cast<const uint8_t*>(ptr) : uint8_t(0);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void ldg_with_pnz(uint16_t &dst, const void *ptr, bool p) {
    dst = p ? *reinterpret_cast<const uint16_t*>(ptr) : uint16_t(0);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void ldg_with_pnz(uint32_t &dst, const void *ptr, bool p) {
    dst = p ? *reinterpret_cast<const uint32_t*>(ptr) : uint32_t(0);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void ldg_with_pnz(uint2 &dst, const void *ptr, bool p) {
    dst = p ? *reinterpret_cast<const uint2*>(ptr) : make_uint2(0u, 0u);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void ldg_with_pnz(uint4 &dst, const void *ptr, bool p) {
    dst = p ? *reinterpret_cast<const uint4*>(ptr) : make_uint4(0u, 0u, 0u, 0u);
}

inline __device__ void ldg_cg( uint4 &dst,
                               const char *ptr,
                               uint64_t mem_desc = MEM_DESC_DEFAULT ) {
#if defined(__LWDA_ARCH__) && __LWDA_ARCH__ >= 800 && XMMA_CP_ASYNC_EXPLICIT_DESCRIPTORS_SUPPORTED
    asm volatile(
        "ld.global.cg.desc.v4.b32 {%0, %1, %2, %3}, [%4], %5;\n" \
            : "=r"(dst.x)
            , "=r"(dst.y)
            , "=r"(dst.z)
            , "=r"(dst.w)
            :  "l"(ptr)
            ,  "l"(mem_desc));
#else
    asm volatile(
        "ld.global.cg.v4.b32 {%0, %1, %2, %3}, [%4];\n" \
            : "=r"(dst.x)
            , "=r"(dst.y)
            , "=r"(dst.z)
            , "=r"(dst.w)
            :  "l"(ptr));
#endif
}

inline __device__ void ldg_cg( uint2 &dst,
                               const char *ptr,
                               uint64_t mem_desc = MEM_DESC_DEFAULT ) {
#if defined(__LWDA_ARCH__) && __LWDA_ARCH__ >= 800 && XMMA_CP_ASYNC_EXPLICIT_DESCRIPTORS_SUPPORTED
    asm volatile(
        "ld.global.cg.desc.v2.b32 {%0, %1}, [%2], %3;\n" \
            : "=r"(dst.x)
            , "=r"(dst.y)
            :  "l"(ptr)
            ,  "l"(mem_desc));
#else
    asm volatile(
        "ld.global.cg.v2.b32 {%0, %1}, [%2];\n" \
            : "=r"(dst.x)
            , "=r"(dst.y)
            :  "l"(ptr));
#endif
}

inline __device__ void ldg_cg( uint32_t &dst,
                               const char *ptr,
                               uint64_t mem_desc = MEM_DESC_DEFAULT ) {
#if defined(__LWDA_ARCH__) && __LWDA_ARCH__ >= 800 && XMMA_CP_ASYNC_EXPLICIT_DESCRIPTORS_SUPPORTED
    asm volatile(
        "ld.global.cg.desc.b32 %0, [%1], %2;\n" \
            : "=r"(dst)
            :  "l"(ptr)
            ,  "l"(mem_desc));
#else
    asm volatile(
        "ld.global.cg.b32 %0, [%1];\n" \
            : "=r"(dst)
            :  "l"(ptr));
#endif
}
////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Data_type, int N >
struct Ldg_functor {
    // Ctor.
    inline __device__ Ldg_functor(Data_type (&fetch)[N], const void* (&ptrs)[N])
        : fetch_(fetch), ptrs_(ptrs) {
    }

    // Clear the element.
    inline __device__ void clear(int ii) {
        xmma::clear_if_pnz_is_not_supported(fetch_[ii]);
    }

    // Trigger the loads.
    inline __device__ void ldgsts(int ii, bool p, uint64_t mem_desc) {
        // if( p ) {
        //     xmma::ldg(fetch_[ii], ptrs_[ii], mem_desc);
        // }
        ldg_with_pnz(fetch_[ii], ptrs_[ii], p);
    }

    // The fetch registers.
    Data_type (&fetch_)[N];
    // The pointers.
    const void* (&ptrs_)[N];
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Data_type, int N, int M >
inline __device__ void ldg_(Data_type (&fetch)[N],
                            const void* (&ptrs)[N],
                            uint32_t (&preds)[M],
                            uint64_t mem_desc = MEM_DESC_DEFAULT) {
    Ldg_functor<Data_type, N> fct(fetch, ptrs);
    ldgsts_<N>(fct, preds, mem_desc);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template< int N, int M >
inline __device__ void ldg(uint8_t (&fetch)[N],
                           const void* (&ptrs)[N],
                           uint32_t (&preds)[M],
                           uint64_t mem_desc = MEM_DESC_DEFAULT) {
    ldg_<uint8_t, N>(fetch, ptrs, preds, mem_desc);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template< int N, int M >
inline __device__ void ldg(uint16_t (&fetch)[N],
                           const void* (&ptrs)[N],
                           uint32_t (&preds)[M],
                           uint64_t mem_desc = MEM_DESC_DEFAULT) {
    ldg_<uint16_t, N>(fetch, ptrs, preds, mem_desc);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template< int N, int M >
inline __device__ void ldg(uint32_t (&fetch)[N],
                           const void* (&ptrs)[N],
                           uint32_t (&preds)[M],
                           uint64_t mem_desc = MEM_DESC_DEFAULT) {
    ldg_<uint32_t, N>(fetch, ptrs, preds, mem_desc);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template< int N, int M >
inline __device__ void ldg(uint2 (&fetch)[N],
                           const void* (&ptrs)[N],
                           uint32_t (&preds)[M],
                           uint64_t mem_desc = MEM_DESC_DEFAULT) {
    ldg_<uint2, N>(fetch, ptrs, preds, mem_desc);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template< int N, int M >
inline __device__ void ldg(uint4 (&fetch)[N],
                           const void* (&ptrs)[N],
                           uint32_t (&preds)[M],
                           uint64_t mem_desc = MEM_DESC_DEFAULT) {
    ldg_<uint4, N>(fetch, ptrs, preds, mem_desc);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Data_type, int N >
inline __device__ void ldg_(Data_type (&fetch)[N],
                            const void* (&ptrs)[N],
                            uint32_t preds,
                            uint64_t mem_desc = MEM_DESC_DEFAULT) {
    uint32_t tmp[1] = { preds };
    ldg_(fetch, ptrs, tmp, mem_desc);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template< int N >
inline __device__ void ldg(uint32_t (&fetch)[N],
                           const void* (&ptrs)[N],
                           uint32_t preds,
                           uint64_t mem_desc = MEM_DESC_DEFAULT) {
    ldg_<uint32_t, N>(fetch, ptrs, preds, mem_desc);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template< int N >
inline __device__ void ldg(uint2 (&fetch)[N],
                           const void* (&ptrs)[N],
                           uint32_t preds,
                           uint64_t mem_desc = MEM_DESC_DEFAULT) {
    ldg_<uint2, N>(fetch, ptrs, preds, mem_desc);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template< int N >
inline __device__ void ldg(uint4 (&fetch)[N],
                           const void* (&ptrs)[N],
                           uint32_t preds,
                           uint64_t mem_desc = MEM_DESC_DEFAULT) {
    ldg_<uint4, N>(fetch, ptrs, preds, mem_desc);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// L D G S T S
//
////////////////////////////////////////////////////////////////////////////////////////////////////
template<bool ZFILL_ = true, bool BYPASS_ = true, bool NAN_FILL_OOB_ = false>
struct Ldgsts_config {
    static const bool BYPASS = BYPASS_;
    static const bool ZFILL = ZFILL_;
    static const bool NAN_FILL_OOB = NAN_FILL_OOB_;
};

inline __device__ void ldgsts128(uint32_t dst,
                                 const void *src,
                                 bool p = true,
                                 uint64_t mem_desc = MEM_DESC_DEFAULT) {
#if defined(__LWDA_ARCH__) && __LWDA_ARCH__ >= 800
    uint32_t m = p ? 16u : 0u;
#if XMMA_CP_ASYNC_EXPLICIT_DESCRIPTORS_SUPPORTED
    asm volatile(
#if XMMA_L2_SECTOR_PROMOTION_SUPPORTED
            ".pragma \"next knob sectorpromotion=128\";\n"
#endif
            "cp.async.cg.shared.global.desc [%0], [%1], 16, %2, %3;\n" \
        :: "r"(dst), "l"(src), "r"(m), "l"(mem_desc));
#else
    asm volatile(
#if XMMA_L2_SECTOR_PROMOTION_SUPPORTED
            ".pragma \"next knob sectorpromotion=128\";\n"
#endif
            "cp.async.cg.shared.global [%0], [%1], 16, %2;\n" \
        :: "r"(dst), "l"(src), "r"(m));
#endif
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void ldgsts128_access(uint32_t dst,
                                        const void *src,
                                        bool p = true,
                                        uint64_t mem_desc = MEM_DESC_DEFAULT) {
#if defined(__LWDA_ARCH__) && __LWDA_ARCH__ >= 800
    uint32_t m = p ? 16u : 0u;
#if XMMA_CP_ASYNC_EXPLICIT_DESCRIPTORS_SUPPORTED
    asm volatile(
#if XMMA_L2_SECTOR_PROMOTION_SUPPORTED
            ".pragma \"next knob sectorpromotion=128\";\n"
#endif
            "cp.async.ca.shared.global.desc [%0], [%1], 16, %2, %3;\n" \
        :: "r"(dst), "l"(src), "r"(m), "l"(mem_desc));
#else
    asm volatile(
#if XMMA_L2_SECTOR_PROMOTION_SUPPORTED
            ".pragma \"next knob sectorpromotion=128\";\n"
#endif
            "cp.async.ca.shared.global [%0], [%1], 16, %2;\n" \
        :: "r"(dst), "l"(src), "r"(m));
#endif
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void ldgsts128_nozfill(uint32_t dst,
                                 const void *src,
                                 bool p = true,
                                 uint64_t mem_desc = MEM_DESC_DEFAULT) {
#if defined(__LWDA_ARCH__) && __LWDA_ARCH__ >= 800
#if XMMA_CP_ASYNC_EXPLICIT_DESCRIPTORS_SUPPORTED
    if( p ) {
        asm volatile(
#if XMMA_L2_SECTOR_PROMOTION_SUPPORTED
                ".pragma \"next knob sectorpromotion=128\";\n"
#endif
                "cp.async.cg.shared.global.desc [%0], [%1], 16, %2;\n" \
            :: "r"(dst), "l"(src), "l"(mem_desc));
    }
#else
    if( p ) {
        asm volatile(
#if XMMA_L2_SECTOR_PROMOTION_SUPPORTED
                ".pragma \"next knob sectorpromotion=128\";\n"
#endif
                "cp.async.cg.shared.global [%0], [%1], 16;\n" \
            :: "r"(dst), "l"(src));
    }
#endif
#endif
}

inline __device__ void ldgsts128_nozfill_access(
                                 uint32_t dst,
                                 const void *src,
                                 bool p = true,
                                 uint64_t mem_desc = MEM_DESC_DEFAULT) {
#if defined(__LWDA_ARCH__) && __LWDA_ARCH__ >= 800
#if XMMA_CP_ASYNC_EXPLICIT_DESCRIPTORS_SUPPORTED
    if( p ) {
        asm volatile(
#if XMMA_L2_SECTOR_PROMOTION_SUPPORTED
                ".pragma \"next knob sectorpromotion=128\";\n"
#endif
                "cp.async.ca.shared.global.desc [%0], [%1], 16, %2;\n" \
            :: "r"(dst), "l"(src), "l"(mem_desc));
    }
#else
    if( p ) {
        asm volatile(
#if XMMA_L2_SECTOR_PROMOTION_SUPPORTED
                ".pragma \"next knob sectorpromotion=128\";\n"
#endif
                "cp.async.ca.shared.global [%0], [%1], 16;\n" \
            :: "r"(dst), "l"(src));
    }
#endif
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void ldgsts128_nopreds(uint32_t dst, const void *src) {
#if defined(__LWDA_ARCH__) && __LWDA_ARCH__ >= 800
    asm volatile(
#if XMMA_L2_SECTOR_PROMOTION_SUPPORTED
            ".pragma \"next knob sectorpromotion=128\";\n"
#endif
            "cp.async.cg.shared.global [%0], [%1], 16;\n" \
        :: "r"(dst), "l"(src));
#endif
}

inline __device__ void ldgsts32_nopreds(uint32_t dst, const void *src) {
#if defined(__LWDA_ARCH__) && __LWDA_ARCH__ >= 800
    asm volatile(
#if XMMA_L2_SECTOR_PROMOTION_SUPPORTED
            ".pragma \"next knob sectorpromotion=128\";\n"
#endif
            "cp.async.ca.shared.global [%0], [%1], 4;\n" \
        :: "r"(dst), "l"(src));
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void ldgsts64(uint32_t dst,
                                const void *src,
                                bool p = true,
                                uint64_t mem_desc = MEM_DESC_DEFAULT) {
#if defined(__LWDA_ARCH__) && __LWDA_ARCH__ >= 800
    uint32_t m = p ? 8u : 0u;
#if XMMA_CP_ASYNC_EXPLICIT_DESCRIPTORS_SUPPORTED
    asm volatile(
#if XMMA_L2_SECTOR_PROMOTION_SUPPORTED
            ".pragma \"next knob sectorpromotion=128\";\n"
#endif
            "cp.async.ca.shared.global.desc [%0], [%1], 8, %2, %3;\n" \
        :: "r"(dst), "l"(src), "r"(m), "l"(mem_desc));
#else
    asm volatile(
#if XMMA_L2_SECTOR_PROMOTION_SUPPORTED
            ".pragma \"next knob sectorpromotion=128\";\n"
#endif
            "cp.async.ca.shared.global [%0], [%1], 8, %2;\n" \
        :: "r"(dst), "l"(src), "r"(m));
#endif
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void ldgsts64_nozfill(uint32_t dst,
                                        const void *src,
                                        bool p = true,
                                        uint64_t mem_desc = MEM_DESC_DEFAULT) {
#if defined(__LWDA_ARCH__) && __LWDA_ARCH__ >= 800
#if XMMA_CP_ASYNC_EXPLICIT_DESCRIPTORS_SUPPORTED
    if( p ) {
        asm volatile(
#if XMMA_L2_SECTOR_PROMOTION_SUPPORTED
                ".pragma \"next knob sectorpromotion=128\";\n"
#endif
                "cp.async.ca.shared.global.desc [%0], [%1], 8, %2;\n" \
            :: "r"(dst), "l"(src), "l"(mem_desc));
    }
#else
    if( p ) {
        asm volatile(
#if XMMA_L2_SECTOR_PROMOTION_SUPPORTED
                ".pragma \"next knob sectorpromotion=128\";\n"
#endif
                "cp.async.ca.shared.global [%0], [%1], 8;\n" \
            :: "r"(dst), "l"(src));
    }
#endif
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void ldgsts32(uint32_t dst,
                                const void *src,
                                bool p = true,
                                uint64_t mem_desc = MEM_DESC_DEFAULT) {
#if defined(__LWDA_ARCH__) && __LWDA_ARCH__ >= 800
    uint32_t m = p ? 4u : 0u;
#if XMMA_CP_ASYNC_EXPLICIT_DESCRIPTORS_SUPPORTED
    asm volatile(
#if XMMA_L2_SECTOR_PROMOTION_SUPPORTED
            ".pragma \"next knob sectorpromotion=128\";\n"
#endif
            "cp.async.ca.shared.global.desc [%0], [%1], 4, %2, %3;\n" \
        :: "r"(dst), "l"(src), "r"(m), "l"(mem_desc));
#else
    asm volatile(
#if XMMA_L2_SECTOR_PROMOTION_SUPPORTED
            ".pragma \"next knob sectorpromotion=128\";\n"
#endif
            "cp.async.ca.shared.global [%0], [%1], 4, %2;\n" \
        :: "r"(dst), "l"(src), "r"(m));
#endif
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void ldgsts32_nozfill(uint32_t dst,
                                const void *src,
                                bool p = true,
                                uint64_t mem_desc = MEM_DESC_DEFAULT) {
#if defined(__LWDA_ARCH__) && __LWDA_ARCH__ >= 800
#if XMMA_CP_ASYNC_EXPLICIT_DESCRIPTORS_SUPPORTED
    if( p ) {
    asm volatile(
#if XMMA_L2_SECTOR_PROMOTION_SUPPORTED
            ".pragma \"next knob sectorpromotion=128\";\n"
#endif
            "cp.async.ca.shared.global.desc [%0], [%1], 4, %2;\n" \
        :: "r"(dst), "l"(src), "l"(mem_desc));
    }
#else
    if( p ) {
    asm volatile(
#if XMMA_L2_SECTOR_PROMOTION_SUPPORTED
            ".pragma \"next knob sectorpromotion=128\";\n"
#endif
            "cp.async.ca.shared.global [%0], [%1], 4;\n" \
        :: "r"(dst), "l"(src) );
    }
#endif
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template< int N, int SZ, bool ZFILL = true, bool BYPASS = true >
struct Ldgsts_functor {
    // Ctor.
    inline __device__ Ldgsts_functor(uint32_t (&smem_ptrs)[N], const void* (&gmem_ptrs)[N])
        : smem_ptrs_(smem_ptrs), gmem_ptrs_(gmem_ptrs) {
    }

    // Does nothing.
    inline __device__ void clear(int ii) {
    }

    // Trigger the load-store instruction.
    inline __device__ void ldgsts(int ii, bool p, uint64_t mem_desc) {
        if( ZFILL ) {
            if( SZ == 16 ) {
                if( BYPASS ) {
                    xmma::ldgsts128(smem_ptrs_[ii], gmem_ptrs_[ii], p, mem_desc);
                } else {
                    xmma::ldgsts128_access(smem_ptrs_[ii], gmem_ptrs_[ii], p, mem_desc);
                }
            } else if( SZ == 8 ) {
                xmma::ldgsts64(smem_ptrs_[ii], gmem_ptrs_[ii], p, mem_desc);
            } else if( SZ == 4 ) {
                xmma::ldgsts32(smem_ptrs_[ii], gmem_ptrs_[ii], p, mem_desc);
            }
        } else {
            if( SZ == 16 ) {
                if( BYPASS ) {
                    xmma::ldgsts128_nozfill(smem_ptrs_[ii], gmem_ptrs_[ii], p, mem_desc);
                } else {
                    xmma::ldgsts128_nozfill_access(smem_ptrs_[ii], gmem_ptrs_[ii], p, mem_desc);
                }
            } else if( SZ == 8 ) {
                xmma::ldgsts64_nozfill(smem_ptrs_[ii], gmem_ptrs_[ii], p, mem_desc);
            } else if( SZ == 4 ) {
                xmma::ldgsts32_nozfill(smem_ptrs_[ii], gmem_ptrs_[ii], p, mem_desc);
            }
        }
    }

    // The shared memory pointers.
    uint32_t (&smem_ptrs_)[N];
    // The global memory pointers.
    const void* (&gmem_ptrs_)[N];
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// Pixel value to be used for out of bounds pixels
template< uint32_t OOB_VAL_H_ = 0x7eff, uint32_t OOB_VAL_H2_ = 0x7eff7eff >
struct Oob_val {
    static constexpr uint16_t OOB_VAL_H = OOB_VAL_H_;
    static constexpr uint32_t OOB_VAL_H2 = OOB_VAL_H2_;
};

// Just function delaration - search below for definition
template< int N, int M >
inline __device__ void sts(uint32_t (&ptrs)[N], const uint4& data, uint32_t (&preds)[M],
                           bool ilwerse_preds);

template< int N,
          int M,
          int SZ = 16,
          bool ZFILL = true,
          bool NAN_FILL_OOB = false,
          bool BYPASS = true >
inline __device__ void ldgsts(uint32_t (&dst)[N],
                              const void* (&src)[N],
                              uint32_t (&preds)[M],
                              uint64_t mem_desc = MEM_DESC_DEFAULT) {

    if( NAN_FILL_OOB ) {
        //static_assert(ZFILL == false, "Can't do Nan-Fill + Zfill - race condition failure");

        Ldgsts_functor<N, SZ, ZFILL> fct(dst, src);
        ldgsts_<N>(fct, preds, mem_desc);

        // now that we have filled without ZFILL - lets fill the other portion
        // Note this is a SPECIAL NAN_FILL(the name is to distingiush with NAN)
        // - and not the LWPU canonical NAN. This has been vetted by the SM Arch.
        // team as a good choice to signal out of bound pixels.
        const uint32_t NAN_FILL = Oob_val<>::OOB_VAL_H2;

        // Make sure we ilwert the predicates while doing STS
        sts(dst, make_uint4(NAN_FILL, NAN_FILL, NAN_FILL, NAN_FILL), preds, true);

    } else {
        Ldgsts_functor<N, SZ, ZFILL, BYPASS> fct(dst, src);
        ldgsts_<N>(fct, preds, mem_desc);
    }
}

template< int N,
          int M,
          int PHASE,
          int SZ = 16,
          bool ZFILL = true,
          bool NAN_FILL_OOB = false,
          bool BYPASS = true >
inline __device__ void ldgsts_per_phase(uint32_t (&dst)[N],
                              const void* (&src)[N],
                              uint32_t (&preds)[M],
                              uint64_t mem_desc = MEM_DESC_DEFAULT) {

    if( NAN_FILL_OOB ) {
        //static_assert(ZFILL == false, "Can't do Nan-Fill + Zfill - race condition failure");

        Ldgsts_functor<N, SZ, ZFILL> fct(dst, src);
        ldgsts_<N>(fct, preds, mem_desc);

        // now that we have filled without ZFILL - lets fill the other portion
        // Note this is a SPECIAL NAN_FILL - and not the LWPU canonical NAN
        //  This has been vetted by the SM Arch. team as a good choice to signal
        // Out of bound pixels
        const uint32_t NAN_FILL = Oob_val<>::OOB_VAL_H2;

        // Make sure we ilwert the predicates while doing STS
        sts(dst, make_uint4(NAN_FILL, NAN_FILL, NAN_FILL, NAN_FILL), preds, true);

    } else {
        Ldgsts_functor<N, SZ, ZFILL, BYPASS> fct(dst, src);
        ldgsts_per_phase_<N, PHASE>(fct, preds, mem_desc);
    }
}

template< int N,
          int M,
          int SZ,
          typename LDGSTS_CFG>
inline __device__ void ldgsts(uint32_t (&dst)[N],
                              const void* (&src)[N],
                              uint32_t (&preds)[M],
                              uint64_t mem_desc = MEM_DESC_DEFAULT) {
    ldgsts<N,
           M,
           SZ,
           LDGSTS_CFG::ZFILL,
           LDGSTS_CFG::NAN_FILL_OOB,
           LDGSTS_CFG::BYPASS> (dst, src, preds, mem_desc);
}

template< int N,
          int M,
          int PHASE,
          int SZ,
          typename LDGSTS_CFG>
inline __device__ void ldgsts_per_phase(uint32_t (&dst)[N],
                                        const void* (&src)[N],
                                        uint32_t (&preds)[M],
                                        uint64_t mem_desc = MEM_DESC_DEFAULT) {
    ldgsts_per_phase<N,
                     M,
                     PHASE,
                     SZ,
                     LDGSTS_CFG::ZFILL,
                     LDGSTS_CFG::NAN_FILL_OOB,
                     LDGSTS_CFG::BYPASS> (dst, src, preds, mem_desc);
}
////////////////////////////////////////////////////////////////////////////////////////////////////

template< int N, int SZ = 16 >
inline __device__ void ldgsts(uint32_t (&dst)[N],
                              const void* (&src)[N],
                              uint32_t preds,
                              uint64_t mem_desc = MEM_DESC_DEFAULT) {
    uint32_t tmp[1] = { preds };
    ldgsts<N, 1, SZ>(dst, src, tmp, mem_desc);
}

template<int SZ = 4, bool ZFILL = true >
inline __device__ void ldgsts(uint32_t &dst,
                              const void* &src,
                              uint32_t preds,
                              uint64_t mem_desc = MEM_DESC_DEFAULT) {
    if ( ZFILL ) {
        if( SZ == 16 ) {
            xmma::ldgsts128(dst, src, preds, mem_desc);
        } else if( SZ == 8 ) {
            xmma::ldgsts64(dst, src, preds, mem_desc);
        } else if( SZ == 4 ) {
            xmma::ldgsts32(dst, src, preds, mem_desc);
        }
    } else {
        if( SZ == 16 ) {
            xmma::ldgsts128_nozfill(dst, src, preds, mem_desc);
        } else if( SZ == 8 ) {
            xmma::ldgsts64_nozfill(dst, src, preds, mem_desc);
        } else if( SZ == 4 ) {
            xmma::ldgsts32_nozfill(dst, src, preds, mem_desc);
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// E L E C T - O N E - S Y N C
//
////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ uint32_t elect_one_sync() {
    uint32_t pred = 0;
#if __LWDA_ARCH__ >= 900
    uint32_t laneid = 0;
    asm volatile( "\n\
    {\n\
        .reg .b32 %rx;\n\
        .reg .pred %px;\n\
        elect.one.sync %rx|%px, %2;\n\
        @%px mov.s32 %1, 1;\n\
        mov.s32 %0, %rx;\n\
    }\n\
  "
                  : "+r"( laneid ), "+r"( pred )
                  : "r"( 0xFFFFFFFF ) );
#endif
    return pred;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// U T M A L D G
//
////////////////////////////////////////////////////////////////////////////////////////////////////

template <uint8_t DIM, lwdaTmaDescType DESC_TYPE, int USE_TMA_MULTICAST>
inline __device__ void utmaldg( const lwdaTmaDescv2 *p_desc,
                                uint32_t urb0,
                                uint32_t urb1,
                                int32_t urb2,
                                int32_t urb3,
                                int32_t urb4,
                                int32_t urb5,
                                int32_t urb6,
                                uint32_t off_w = 0,
                                uint32_t off_h = 0,
                                uint32_t off_d = 0,
                                uint16_t mcast_mask = 0 ) {
#if defined( __LWDA_ARCH__ ) && __LWDA_ARCH__ >= 900
    if( ! USE_TMA_MULTICAST ) {
        if( DIM == 2 && DESC_TYPE == TILED ) {
            asm volatile( "cp.async.bulk.tensor.2d.shared.global.mbarrier [%0], [%1, {%2, %3}], [%4];\n"
                          :
                          : "r"( urb0 ),
                            "l"( reinterpret_cast<uint64_t>( p_desc ) ),
                            "r"( urb2 ),
                            "r"( urb3 ),
                            "r"( urb1 )
                          : "memory" );
        }
        if( DIM == 5 && DESC_TYPE == TILED ) {
            asm volatile( "cp.async.bulk.tensor.5d.shared.global.mbarrier [%0], [%1, {%2, %3, %4, %5, "
                          "%6}], [%7];\n"
                          :
                          : "r"( urb0 ),
                            "l"( reinterpret_cast<uint64_t>( p_desc ) ),
                            "r"( urb2 ),
                            "r"( urb3 ),
                            "r"( urb4 ),
                            "r"( urb5 ),
                            "r"( urb6 ),
                            "r"( urb1 )
                          : "memory" );
        }
        if( DIM == 5 && DESC_TYPE == IM2COL ) {
            asm volatile( "cp.async.bulk.tensor.5d.shared.global.im2col.mbarrier [%0], [%1, {%2, %3, "
                          "%4, %5, %6}], [%7], {%8, %9, %10};\n"
                          :
                          : "r"( urb0 ),
                            "l"( reinterpret_cast<uint64_t>( p_desc ) ),
                            "r"( urb2 ),
                            "r"( urb3 ),
                            "r"( urb4 ),
                            "r"( urb5 ),
                            "r"( urb6 ),
                            "r"( urb1 ),
                            "h"( static_cast<uint16_t>( off_w ) ),
                            "h"( static_cast<uint16_t>( off_h ) ),
                            "h"( static_cast<uint16_t>( off_d ) )
                          : "memory" );
        }

    // Multicast enabled
    } else {
        if( DIM == 2 && DESC_TYPE == TILED ) {
            asm volatile( "cp.async.bulk.tensor.2d.multicast.shared.global.mbarrier [%0], [%1, {%2, %3}], [%4], [%5];\n"
                          :
                          : "r"( urb0 ),
                            "l"( reinterpret_cast<uint64_t>( p_desc ) ),
                            "r"( urb2 ),
                            "r"( urb3 ),
                            "r"( urb1 ),
                            "h"(mcast_mask)
                          : "memory" );
        }
        //
        // TODO : Add multicast support for other modes (3D, 4D, 5D, IM2COL)
        //
    }
#endif
}

template<uint8_t DIM>
inline __device__ void utmaldg_tiled(const lwdaTmaDesc *p_desc, uint32_t urb0, uint32_t urb1, int32_t urb2, int32_t urb3, int32_t urb4, int32_t urb5, int32_t urb6) {
    UTMALDG<DIM, TILED, false>(p_desc, urb0, urb1, urb2, urb3, urb4, urb5, urb6, 0);
    // __utmaldg<k_dim, TILED, false>(p_desc, urb0, urb1, urb2, urb3, urb4, urb5, urb6);
}

template<uint8_t DIM>
inline __device__ void utmaldg_im2col(const lwdaTmaDesc *p_desc,
                                     uint32_t urb0,
                                     uint32_t urb1,
                                     int32_t urb2,
                                     int32_t urb3,
                                     int32_t urb4,
                                     int32_t urb5,
                                     int32_t urb6,
                                     uint32_t urc) {
    UTMALDG<DIM, IM2COL, false>(p_desc, urb0, urb1, urb2, urb3, urb4, urb5, urb6, urc);
}

template <uint8_t DIM, lwdaTmaDescType DESC_TYPE = TILED>
__device__ void utmapf2( const lwdaTmaDesc *p_desc_,
                         int32_t urb0,
                         int32_t urb1,
                         int32_t urb2,
                         int32_t urb3,
                         int32_t urb4 ) {
    __utmapf2<DIM, DESC_TYPE>(p_desc_, urb0, urb1, urb2, urb3, urb4);
}


template <uint8_t DIM, lwdaTmaDescType DESC_TYPE = TILED>
__device__ void utmapf2( const lwdaTmaDescv2 *p_desc_,
                         int32_t urb0,
                         int32_t urb1,
                         int32_t urb2,
                         int32_t urb3,
                         int32_t urb4 ) {
}

template<lwdaCacheCtrl CTL>
__device__ void utmacctl(const lwdaTmaDesc *p_desc_) {
    __utmacctl<CTL>(p_desc_);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// L D S
//
////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void lds(uint16_t &dst, uint32_t ptr) {
    asm volatile("ld.shared.b16 %0, [%1];\n"
        : "=h"(dst)
        :  "r"(ptr));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void lds(uint32_t &dst, uint32_t ptr) {
    asm volatile("ld.shared.b32 %0, [%1];\n"
        : "=r"(dst)
        :  "r"(ptr));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void lds(uint2 &dst, uint32_t ptr) {
    asm volatile("ld.shared.v2.b32 {%0, %1}, [%2];\n"
        : "=r"(dst.x)
        , "=r"(dst.y)
        :  "r"(ptr));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void lds(uint4 &dst, uint32_t ptr) {
    asm volatile("ld.shared.v4.b32 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(dst.x)
        , "=r"(dst.y)
        , "=r"(dst.z)
        , "=r"(dst.w)
        :  "r"(ptr));
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// L D S M
//
////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void ldsm(uint32_t &dst, uint32_t ptr) {
#if defined(__LWDA_ARCH__) && __LWDA_ARCH__ >= 730
#if (XMMA_EXTENDED_PTX_ENABLED)
    asm volatile("_ldsm.m8n8.b16 %0, [%1];\n" : "=r"(dst) : "r"(ptr));
#else
    asm volatile("ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];\n"
        : "=r"(dst) : "r"(ptr));
#endif
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void ldsmt(uint32_t &dst, uint32_t ptr) {
#if defined(__LWDA_ARCH__) && __LWDA_ARCH__ >= 730
#if (XMMA_EXTENDED_PTX_ENABLED)
    asm volatile("_ldsm.m8n8.trans.b16 %0, [%1];\n" : "=r"(dst) : "r"(ptr));
#else
    asm volatile("ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16 {%0}, [%1];\n"
        : "=r"(dst) : "r"(ptr));
#endif
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void ldsm(uint2 &dst, uint32_t ptr) {
#if defined(__LWDA_ARCH__) && __LWDA_ARCH__ >= 730
#if (XMMA_EXTENDED_PTX_ENABLED)
    asm volatile("_ldsm.m8n8.x2.b16 {%0, %1}, [%2];\n"
        : "=r"(dst.x), "=r"(dst.y) : "r"(ptr));
#else
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
        : "=r"(dst.x), "=r"(dst.y) : "r"(ptr));
#endif
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void ldsmt(uint2 &dst, uint32_t ptr) {
#if defined(__LWDA_ARCH__) && __LWDA_ARCH__ >= 730
#if (XMMA_EXTENDED_PTX_ENABLED)
    asm volatile("_ldsm.m8n8.x2.trans.b16 {%0, %1}, [%2];\n"
        : "=r"(dst.x), "=r"(dst.y) : "r"(ptr));
#else
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];\n"
        : "=r"(dst.x), "=r"(dst.y) : "r"(ptr));
#endif
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void ldsm(uint4 &dst, uint32_t ptr) {
#if defined(__LWDA_ARCH__) && __LWDA_ARCH__ >= 730
#if (XMMA_EXTENDED_PTX_ENABLED)
    asm volatile("_ldsm.m8n8.x4.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(dst.x), "=r"(dst.y), "=r"(dst.z), "=r"(dst.w) : "r"(ptr));
#else
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(dst.x), "=r"(dst.y), "=r"(dst.z), "=r"(dst.w) : "r"(ptr));
#endif
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void ldsmt(uint4 &dst, uint32_t ptr) {
#if defined(__LWDA_ARCH__) && __LWDA_ARCH__ >= 730
#if (XMMA_EXTENDED_PTX_ENABLED)
    asm volatile("_ldsm.m8n8.x4.trans.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(dst.x), "=r"(dst.y), "=r"(dst.z), "=r"(dst.w) : "r"(ptr));
#else
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(dst.x), "=r"(dst.y), "=r"(dst.z), "=r"(dst.w) : "r"(ptr));
#endif
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// U T M A S T G
//
////////////////////////////////////////////////////////////////////////////////////////////////////

template<uint8_t DIM, lwdaTmaDescType DESC_TYPE>
inline __device__ void utmastg(const lwdaTmaDesc *p_desc, unsigned urb0, unsigned urb1, int urb2, int urb3, int urb4, int urb5, int urb6, int urc) {
    using namespace xmma::hopper::emu;
    UTMASTG<DIM, DESC_TYPE, false>(p_desc, urb0, urb1, urb2, urb3, urb4, urb5, urb6, urc);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// S T G
//
////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void stg(void *ptr, uint8_t val, uint64_t mem_desc = MEM_DESC_DEFAULT) {
    *reinterpret_cast<uint8_t*>(ptr) = val;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void stg(void *ptr, uint16_t val, uint64_t mem_desc = MEM_DESC_DEFAULT) {
#if defined(__LWDA_ARCH__) && __LWDA_ARCH__ >= 800 && XMMA_CP_ASYNC_EXPLICIT_DESCRIPTORS_SUPPORTED
    asm volatile( \
        "st.global.desc.b16 [%0], %2, %1;" \
            :
            : "l"(ptr)
            , "l"(mem_desc)
            , "h"(val));
#else
    *reinterpret_cast<uint16_t*>(ptr) = val;
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void stg(void *ptr, uint32_t val, uint64_t mem_desc = MEM_DESC_DEFAULT) {
#if defined(__LWDA_ARCH__) && __LWDA_ARCH__ >= 800 && XMMA_CP_ASYNC_EXPLICIT_DESCRIPTORS_SUPPORTED
    asm volatile( \
        "st.global.desc.b32 [%0], %2, %1;" \
            :
            : "l"(ptr)
            , "l"(mem_desc)
            , "r"(val));
#else
    *reinterpret_cast<uint32_t*>(ptr) = val;
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void stg(void *ptr, uint2 val, uint64_t mem_desc = MEM_DESC_DEFAULT) {
#if defined(__LWDA_ARCH__) && __LWDA_ARCH__ >= 800 && XMMA_CP_ASYNC_EXPLICIT_DESCRIPTORS_SUPPORTED
    asm volatile( \
        "st.global.desc.v2.b32 [%0], {%2, %3}, %1;" \
            :
            : "l"(ptr)
            , "l"(mem_desc)
            , "r"(val.x)
            , "r"(val.y));
#else
    *reinterpret_cast<uint2*>(ptr) = val;
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void stg(void *ptr, uint4 val, uint64_t mem_desc = MEM_DESC_DEFAULT) {
#if defined(__LWDA_ARCH__) && __LWDA_ARCH__ >= 800 && XMMA_CP_ASYNC_EXPLICIT_DESCRIPTORS_SUPPORTED
    asm volatile( \
        "st.global.desc.v4.b32 [%0], {%2, %3, %4, %5}, %1;" \
            :
            : "l"(ptr)
            , "l"(mem_desc)
            , "r"(val.x)
            , "r"(val.y)
            , "r"(val.z)
            , "r"(val.w));
#else
    *reinterpret_cast<uint4*>(ptr) = val;
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// REDUDCTION
//
////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void red_add_f32(char* ptr, uint32_t val) {
      asm volatile ("red.relaxed.cta.global.add.f32 [%0], %1;" :: "l"(ptr),"r"(val));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void red_add_f16x2(__half2 *ptr, uint32_t val) {
      asm volatile ("red.relaxed.cta.global.add.noftz.f16x2 [%0], %1;" :: "l"(ptr),"r"(val));

}

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// S T S
//
////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void sts(uint32_t ptr, uint16_t val) {
    asm volatile("st.shared.b16 [%0], %1;\n"
        :
        : "r"(ptr)
        , "h"(val));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void sts(uint32_t ptr, uint32_t val) {
    asm volatile("st.shared.b32 [%0], %1;\n"
        :
        : "r"(ptr)
        , "r"(val));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void sts(uint32_t ptr, uint2 val) {
    asm volatile("st.shared.v2.b32 [%0], {%1, %2};\n"
        :
        : "r"(ptr)
        , "r"(val.x)
        , "r"(val.y));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void sts(uint32_t ptr, uint4 val) {
    asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};\n"
        :
        : "r"(ptr)
        , "r"(val.x)
        , "r"(val.y)
        , "r"(val.z)
        , "r"(val.w));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Data_type, int N >
inline __device__ void sts_(uint32_t (&ptrs)[N], const Data_type (&data)[N]) {
    #pragma unroll
    for( int ii = 0; ii < N; ++ii ) {
        sts(ptrs[ii], data[ii]);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// STS.U16

template< int N >
inline __device__ void sts(uint32_t (&ptrs)[N], const uint16_t (&data)[N]) {
    sts_<uint16_t, N>(ptrs, data);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// STS.32

template< int N >
inline __device__ void sts(uint32_t (&ptrs)[N], const uint32_t (&data)[N]) {
    sts_<uint32_t, N>(ptrs, data);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// STS.64

template< int N >
inline __device__ void sts(uint32_t (&ptrs)[N], const uint2 (&data)[N]) {
    sts_<uint2, N>(ptrs, data);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// STS.128

template< int N >
inline __device__ void sts(uint32_t (&ptrs)[N], const uint4 (&data)[N]) {
    sts_<uint4, N>(ptrs, data);
}

template< int N >
inline __device__ void sts_force_64(uint32_t (&ptrs)[N], const uint4 (&data)[N]) {
    #pragma unroll
    for( int ii = 0; ii < N; ++ii ) {
        uint2 tmp = make_uint2(data[ii].x,data[ii].y);
        sts(ptrs[ii], tmp);
    }
}

template< int N >
inline __device__ void sts_force_32(uint32_t (&ptrs)[N], const uint4 (&data)[N]) {
    #pragma unroll
    for( int ii = 0; ii < N; ++ii ) {
        sts(ptrs[ii], data[ii].x);
    }
}
////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Data_type, int N_PTR, int N_DATA>
struct Sts_functor_base {
    // Ctor.
    inline __device__ Sts_functor_base(uint32_t (&ptrs)[N_PTR], const Data_type (&data)[N_DATA])
        : ptrs_(ptrs), data_(data) {
    }

    // Does nothing.
    inline __device__ void clear(int ii) {
    }

    // The pointers.
    uint32_t (&ptrs_)[N_PTR];
    // The data registers.
    const Data_type (&data_)[N_DATA];
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Data_type, int N, bool REUSE_DATA = false, bool ILWERSE_PRED = false >
struct Sts_functor : Sts_functor_base<Data_type, N, N>{

    using Base = Sts_functor_base<Data_type, N, N>;

    // Ctor.
    inline __device__ Sts_functor(uint32_t (&ptrs)[N], const Data_type (&data)[N])
        : Base(ptrs, data) {
    }

    // Trigger the store.
    inline __device__ void ldgsts(int ii, bool p, uint64_t) {
      if( p ) {
          xmma::sts(this->ptrs_[ii], this->data_[ii]);
      }
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Data_type, int N>
struct Sts_functor<Data_type, N, true, true> : Sts_functor_base <Data_type, N, 1>{

    using Base = Sts_functor_base<Data_type, N, 1>;

    // Ctor.
    inline __device__ Sts_functor(uint32_t (&ptrs)[N], const Data_type (&data)[1])
        : Base(ptrs, data) {
    }

    // Trigger the store.
    inline __device__ void ldgsts(int ii, bool p, uint64_t) {
      if( !p ) {
          xmma::sts(this->ptrs_[ii], this->data_[0]);
      }
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Data_type, int N>
struct Sts_functor<Data_type, N, false, true> : Sts_functor_base <Data_type, N, N>{

    using Base = Sts_functor_base<Data_type, N, N>;

    // Ctor.
    inline __device__ Sts_functor(uint32_t (&ptrs)[N], const Data_type (&data)[N])
        : Base(ptrs, data) {
    }

    // Trigger the store.
    inline __device__ void ldgsts(int ii, bool p, uint64_t) {
      if( !p ) {
          xmma::sts(this->ptrs_[ii], this->data_[ii]);
      }
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Data_type, int N>
struct Sts_functor<Data_type, N, true, false> : Sts_functor_base <Data_type, N, 1>{

    using Base = Sts_functor_base<Data_type, N, 1>;

    // Ctor.
    inline __device__ Sts_functor(uint32_t (&ptrs)[N], const Data_type (&data)[1])
        : Base(ptrs, data) {
    }

    // Trigger the store.
    inline __device__ void ldgsts(int ii, bool p, uint64_t) {
      if( p ) {
          xmma::sts(this->ptrs_[ii], this->data_[0]);
      }
    }
};


////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Data_type, int N, int M >
inline __device__ void sts_(uint32_t (&ptrs)[N], const Data_type (&data)[N], uint32_t (&preds)[M]) {
    Sts_functor<Data_type, N> fct(ptrs, data);
    ldgsts_<N>(fct, preds, 0u);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template< int N, int M >
inline __device__ void sts(uint32_t (&ptrs)[N], const uint8_t (&data)[N], uint32_t (&preds)[M]) {
    sts_<uint8_t, N>(ptrs, data, preds);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template< int N, int M >
inline __device__ void sts(uint32_t (&ptrs)[N], const uint16_t (&data)[N], uint32_t (&preds)[M]) {
    sts_<uint16_t, N>(ptrs, data, preds);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template< int N, int M >
inline __device__ void sts(uint32_t (&ptrs)[N], const uint32_t (&data)[N], uint32_t (&preds)[M]) {
    sts_<uint32_t, N>(ptrs, data, preds);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template< int N, int M >
inline __device__ void sts(uint32_t (&ptrs)[N], const uint2 (&data)[N], uint32_t (&preds)[M]) {
    sts_<uint2, N>(ptrs, data, preds);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template< int N, int M >
inline __device__ void sts(uint32_t (&ptrs)[N], const uint4 (&data)[N], uint32_t (&preds)[M]) {
    sts_<uint4, N>(ptrs, data, preds);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Data_type, int N >
inline __device__ void sts_(uint32_t (&ptrs)[N], const Data_type (&data)[N], uint32_t preds) {
    uint32_t tmp[1] = { preds };
    ldgsts_<N>(ptrs, data, tmp, 0u);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template< int N >
inline __device__ void sts(uint32_t (&ptrs)[N], const uint8_t (&data)[N], uint32_t preds) {
    sts_<uint8_t, N>(ptrs, data, preds);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template< int N >
inline __device__ void sts(uint32_t (&ptrs)[N], const uint16_t (&data)[N], uint32_t preds) {
    sts_<uint16_t, N>(ptrs, data, preds);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template< int N >
inline __device__ void sts(uint32_t (&ptrs)[N], const uint32_t (&data)[N], uint32_t preds) {
    sts_<uint32_t, N>(ptrs, data, preds);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template< int N >
inline __device__ void sts(uint32_t (&ptrs)[N], const uint2 (&data)[N], uint32_t preds) {
    sts_<uint2, N>(ptrs, data, preds);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template< int N >
inline __device__ void sts(uint32_t (&ptrs)[N], const uint4 (&data)[N], uint32_t preds) {
    sts_<uint4, N>(ptrs, data, preds);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Data_type, int N, int M, bool REUSE_DATA=true >
inline __device__ void sts_(uint32_t (&ptrs)[N], const Data_type &data, uint32_t (&preds)[M],
                            bool ilwerse_preds) {

    Data_type tmp[1] = { data };

    if ( ilwerse_preds ) {
        Sts_functor<Data_type, N, REUSE_DATA, true> fct(ptrs, tmp);
        ldgsts_<N>(fct, preds, 0u);
    } else {
        Sts_functor<Data_type, N, REUSE_DATA, false> fct(ptrs, tmp);
        ldgsts_<N>(fct, preds, 0u);
    }

}

////////////////////////////////////////////////////////////////////////////////////////////////////

template< int N, int M >
inline __device__ void sts(uint32_t (&ptrs)[N], const uint4& data, uint32_t (&preds)[M],
                           bool ilwerse_preds) {
    sts_<uint4, N, M, true>(ptrs, data, preds, ilwerse_preds);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// S T S M
//
////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void stsm(uint32_t ptr, uint32_t &src) {
#if defined( __LWDA_ARCH__ ) && __LWDA_ARCH__ >= 900
    asm volatile( "stmatrix.sync.aligned.m8n8.x1.shared.b16 [%0], %1;\n" ::"r"( ptr ), "r"( src ) );
#endif
}

inline __device__ void stsmt(uint32_t ptr, uint32_t &src) {
#if defined( __LWDA_ARCH__ ) && __LWDA_ARCH__ >= 900
    asm volatile( "stmatrix.sync.aligned.m8n8.x1.trans.shared.b16 [%0], %1;\n" ::"r"( ptr ),
                  "r"( src ) );
#endif
}

inline __device__ void stsm(uint32_t ptr, uint2 &src) {
#if defined( __LWDA_ARCH__ ) && __LWDA_ARCH__ >= 900
    asm volatile( "stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};\n" ::"r"( ptr ),
                  "r"( src.x ),
                  "r"( src.y ) );
#endif
}

inline __device__ void stsmt(uint32_t ptr, uint2 &src) {
#if defined( __LWDA_ARCH__ ) && __LWDA_ARCH__ >= 900
    asm volatile( "stmatrix.sync.aligned.m8n8.x2.trans.shared.b16 [%0], {%1, %2};\n" ::"r"( ptr ),
                  "r"( src.x ),
                  "r"( src.y ) );
#endif
}

inline __device__ void stsm(uint32_t ptr, uint4 &src) {
#if defined( __LWDA_ARCH__ ) && __LWDA_ARCH__ >= 900
    asm volatile( "stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n" ::"r"( ptr ),
                  "r"( src.x ),
                  "r"( src.y ),
                  "r"( src.z ),
                  "r"( src.w ) );
#endif
}

inline __device__ void stsmt(uint32_t ptr, uint4 &src) {
#if defined( __LWDA_ARCH__ ) && __LWDA_ARCH__ >= 900
    asm volatile(
        "stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n" ::"r"( ptr ),
        "r"( src.x ),
        "r"( src.y ),
        "r"( src.z ),
        "r"( src.w ) );
#endif
}


/*
inline __device__ __half2 divide( const __half2 &x, const __half2 &y ) {
    return __h2div(x, y);
}

inline __device__ __half2 negate(const __half2 &in) {
    return __hneg2(in);
}
*/

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// H E L P E R S   T O   G E T   U I N T s   F R O M   S I Z E   I N   B Y T E S
//
////////////////////////////////////////////////////////////////////////////////////////////////////

template< int SIZE_IN_BYTES >
struct Uint_from_size_in_bytes {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
struct Uint_from_size_in_bytes<1> {
    using Type = uint8_t;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
struct Uint_from_size_in_bytes<2> {
    using Type = uint16_t;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
struct Uint_from_size_in_bytes<4> {
    using Type = uint32_t;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
struct Uint_from_size_in_bytes<8> {
    using Type = uint2;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
struct Uint_from_size_in_bytes<16> {
    using Type = uint4;
};

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Swish
//
////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float swish(float x) {
    float exp = __expf(x);
    return x *  ( exp / (1.0 + exp) );
}

inline __device__ float dswish(float x) {
    float exp = __expf(x);
    float sig = exp / (1.0 + exp);
    return sig * (1.0 - x * (1 - sig));
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// R E L U
//
////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float reluTh_fp32(float x, float lb = 0.f) {
    float res;

    // relu = x > lb ? x : 0
    asm volatile( \
        "{\n" \
        "\t .reg .f32 sela;\n" \
        "\n" \
        "\t set.gtu.f32.f32 sela, %1, %2;\n" \
        "\t mul.f32 %0, sela, %1;\n"
        "}\n" : "=f"(res) : "f"(x), "f"(lb));
    return res;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float relu_fp32(float x, float lb = 0.f) {
    return fmax(x, lb);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float relu_ub_fp32(float x, float ub) {
    return fmin(x, ub);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ double relu_fp64(double x, double lb = 0.f) {
    return fmax(x, lb);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ double relu_ub_fp64(double x, double ub) {
    return fmin(x, ub);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ uint32_t reluTh_fp16x2(uint32_t x, uint32_t lb = 0u) {
    uint32_t res;

    // relu = x > lb ? x : 0
#if (defined(__LWDA_ARCH__) && ((__LWDACC_VER_MAJOR__ == 10 && __LWDACC_VER_MINOR__ >= 2) || __LWDACC_VER_MAJOR__ == 11))
    asm volatile( \
        "{\n" \
        "\t .reg .f16x2 sela;\n" \
        "\n" \
        "\t set.gtu.u32.f16x2 sela, %1, %2;\n" \
        "\t and.b32 %0, sela, %1;\n"
        "}\n" : "=r"(res) : "r"(x), "r"(lb));
#else
    // For compiler version <=10.1
    uint32_t tmp = 0;
    asm volatile("set.gtu.f16x2.f16x2 %0, %1, %2;\n" : "=r"(tmp) : "r"(x), "r"(lb));
    if( tmp == 0x3c003c00 ) { tmp = 0xffffffff; }
    else if( tmp == 0x3c000000 ) { tmp = 0xffff0000; }
    else if( tmp == 0x00003c00 ) { tmp = 0x0000ffff; }
    asm volatile("and.b32 %0, %1, %2;\n" : "=r"(res) : "r"(x), "r"(tmp));

#endif

    return res;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ uint32_t relu_fp16x2(uint32_t x, uint32_t lb = 0u) {
    uint32_t res;

#if defined(__LWDA_ARCH__) && __LWDA_ARCH__ >= 800
    asm volatile( "max.f16x2 %0, %1, %2;\n" : "=r"(res) : "r"(x), "r"(lb));
#else
    // Threshold assigned to 0, equivalent to general ReLU
    res = reluTh_fp16x2(x, lb);
#endif

    return res;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ uint32_t relu_bf16x2(uint32_t x, uint32_t lb = 0u) {

#if defined(__LWDA_ARCH__) && __LWDA_ARCH__ >= 800
    uint32_t res;
    asm volatile( "max.bf16x2 %0, %1, %2;\n" : "=r"(res) : "r"(x), "r"(lb));
    return res;
#else
    assert(0);
    return 0;
#endif

}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ uint32_t relu_ub_bf16x2(uint32_t x, uint32_t ub) {

#if defined(__LWDA_ARCH__) && __LWDA_ARCH__ >= 800
    uint32_t res;
    asm volatile( "min.bf16x2 %0, %1, %2;\n" : "=r"(res) : "r"(x), "r"(ub));
    return res;
#else
    assert(0);
    return 0;
#endif

}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ uint32_t relu_ub_fp16x2(uint32_t x, uint32_t ub) {
    uint32_t res;

#if defined(__LWDA_ARCH__) && __LWDA_ARCH__ >= 800
    asm volatile( "min.f16x2 %0, %1, %2;\n" : "=r"(res) : "r"(x), "r"(ub));
#else
#if (defined(__LWDA_ARCH__) && \
    ((__LWDACC_VER_MAJOR__ == 10 && __LWDACC_VER_MINOR__ >= 2) || __LWDACC_VER_MAJOR__ >= 11))
    // The logic of lop3 is (x&sela)|(ub&~sela)
    asm volatile( \
        "{\n" \
        "\t .reg .f16x2 sela;\n" \
        "\n" \
        "\t set.leu.u32.f16x2 sela, %1, %2;\n" \
        "\t lop3.b32 %0, %1, %2, sela, 0xe4;\n"
        "}\n" : "=r"(res) : "r"(x), "r"(ub));
#else
    // For compiler version <=10.1
    uint32_t tmp = 0;
    asm volatile("set.leu.f16x2.f16x2 %0, %1, %2;\n" : "=r"(tmp) : "r"(x), "r"(ub));
    if( tmp == 0x3c003c00 ) { tmp = 0xffffffff; }
    else if( tmp == 0x3c000000 ) { tmp = 0xffff0000; }
    else if( tmp == 0x00003c00 ) { tmp = 0x0000ffff; }
    // The logic of lop3 is (x&sela)|(ub&~sela)
    asm volatile("lop3.b32 %0, %1, %2, %3, 0xe4;\n" : "=r"(res) : "r"(x), "r"(ub), "r"(tmp));
#if 0
    // r10.1 will complie set.leu to HSET2.BF instead of BM, and
    // lop3 cannot work correctly with BF in sela, thus using the
    // the above ugly implementation to get r10.1 kernel pass ref check.
    asm volatile( \
        "{\n" \
        "\t .reg .f16x2 sela;\n" \
        "\n" \
        "\t set.leu.f16x2.f16x2 sela, %1, %2;\n" \
        "\t lop3.b32 %0, %1, %2, sela, 0xe4;\n"
        "}\n" : "=r"(res) : "r"(x), "r"(ub));
#endif
#endif
#endif

    return res;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ uint32_t relu_fp16(uint32_t x, uint32_t lb = 0u) {
    return relu_fp16x2(x, lb);  // Packing is identical.
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ uint32_t bfma2_relu(uint32_t a, uint32_t b, uint32_t c,
                                      uint32_t with_relu, uint32_t lb) {
    uint32_t d;

#if defined(__LWDA_ARCH__) && __LWDA_ARCH__ >= 800
    if( with_relu ) {
        asm volatile("fma.rn.relu.bf16x2 %0, %1, %2, %3;\n" : "=r"(d) : "r"(a), "r"(b), "r"(c));
    } else {
        asm volatile("fma.rn.bf16x2 %0, %1, %2, %3;\n" : "=r"(d) : "r"(a), "r"(b), "r"(c));
    }
#else
    d = bfma2(a, b, c);
    d = relu_bf16x2(d, lb);
#endif
    return d;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ uint32_t hfma2_relu(uint32_t a, uint32_t b, uint32_t c,
                                      uint32_t with_relu, uint32_t lb) {
    uint32_t d;

#if defined(__LWDA_ARCH__) && __LWDA_ARCH__ >= 800
    if( with_relu ) {
        asm volatile("fma.rn.relu.f16x2 %0, %1, %2, %3;\n" : "=r"(d) : "r"(a), "r"(b), "r"(c));
    } else {
        asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(d) : "r"(a), "r"(b), "r"(c));
    }
#else
    d = hfma2(a, b, c);
    d = relu_fp16x2(d, lb);
#endif
    return d;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ uint32_t relu_s32(int32_t x, int32_t lb = 0) {
    uint32_t res;
    asm volatile( \
        "{\n" \
        "\t .reg .s32 sela;\n" \
        "\n" \
        "\t set.ge.s32.s32 sela, %1, %2;\n" \
        "\t mul.lo.s32 %0, sela, %1;\n"
        "}\n" : "=r"(res) : "r"(x), "r"(lb));
    return res;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ uint32_t relu_s8x4(uint32_t x, uint32_t lb = 0) {
    // Undo the transformation from s8x4 to make the arithmetic easier
    int8_t lb_s8 = static_cast<int8_t>(lb & 0xff);

    // x = {a,b,c,d}
    int8_t a = (x >>  0) & 0xff;
    int8_t b = (x >>  8) & 0xff;
    int8_t c = (x >> 16) & 0xff;
    int8_t d = (x >> 24) & 0xff;

    uint32_t mask = 0x0;
    if (a >= lb_s8) mask = mask |       0xff;
    if (b >= lb_s8) mask = mask |     0xff00;
    if (c >= lb_s8) mask = mask |   0xff0000;
    if (d >= lb_s8) mask = mask | 0xff000000;

    return (x & mask);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ uint32_t relu_ub_s8x4( uint32_t x, uint32_t ub ) {
    // Undo the transformation from s8x4 to make the arithmetic easier
    int8_t ub_s8 = static_cast<int8_t>( ub & 0xff );
    uint32_t ub_s8x4 = ( static_cast<uint32_t>( ub_s8 ) << 24 )
                     | ( static_cast<uint32_t>( ub_s8 ) << 16 )
                     | ( static_cast<uint32_t>( ub_s8 ) <<  8 )
                     | ( static_cast<uint32_t>( ub_s8 ) );

    // x = {a,b,c,d}
    int8_t a = ( x >> 0 ) & 0xff;
    int8_t b = ( x >> 8 ) & 0xff;
    int8_t c = ( x >> 16 ) & 0xff;
    int8_t d = ( x >> 24 ) & 0xff;

    uint32_t mask = 0x0;
    if( a <= ub_s8 )
        mask = mask | 0xff;
    if( b <= ub_s8 )
        mask = mask | 0xff00;
    if( c <= ub_s8 )
        mask = mask | 0xff0000;
    if( d <= ub_s8 )
        mask = mask | 0xff000000;

    return ( x & mask ) | ( ub_s8x4 & ( ~mask ) );
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void spin_lock(int *counter_gmem,
                                 int expected,
                                 int init = 0,
                                 int tidx = threadIdx.x,
                                 Named_barrier epi_sync = Named_barrier()) {

    int found = init;
    if( tidx == 0 ) {
        while( found != expected ) {
            asm volatile("ld.global.acquire.gpu.b32 %0, [%1];" : "=r"(found) : "l"(counter_gmem));
        }
    }
    __syncwarp();
    if( epi_sync.invalid() ) {
        __syncthreads();
    } else {
        epi_sync.wait();
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ __host__ int div_up(int m, int n) {
    return (m + n-1) / n;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ __host__ int min(int m, int n) {
    return (m < n) ? m : n;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template< bool USE_LDGSTS >
inline __device__ void ldgdepbar() {
    if( USE_LDGSTS ) {
        #if LWDA_CP_ASYNC_GROUP_POLICY_ACTIVATED
            asm volatile("cp.async.commit_group;\n" ::);
        #elif LWDA_CP_ASYNC_ACTIVATED
            asm volatile("cp.async.wait.defer;\n" ::);
        #else
            assert(0);
        #endif
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template< bool USE_LDGSTS, int STAGES >
inline __device__ void depbar() {
    if( USE_LDGSTS ) {
        #if LWDA_CP_ASYNC_GROUP_POLICY_ACTIVATED
            const int VALUE = Max<STAGES - 2, 0>::VALUE;
            asm volatile("cp.async.wait_group %0;\n" ::"n"(VALUE));
        #elif LWDA_CP_ASYNC_ACTIVATED
            const int VALUE = Max<STAGES - 2, 0>::VALUE;
            asm volatile("cp.async.wait %0;\n" ::"n"(VALUE));
        #else
            assert(0);
        #endif
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void ldg_force_64(uint4 &dst, const void* ptr, bool p) {
    const uint64_t zero = 0ull;
    asm volatile("mov.b64 {%0, %1 }, %2;\n" : "=r"(dst.x), "=r"(dst.y) : "l"(zero));
    if( p ) {
        asm volatile("ld.global.v2.b32 {%0, %1}, [%2];\n"
            : "=r"(dst.x), "=r"(dst.y)
            :  "l"(ptr));
    }

}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void ldg_force_64(uint4 (&dst)[1], const void* (&ptr)[1], uint32_t preds) {
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void ldg_force_64(uint4 (&dst)[2], const void* (&ptr)[2], uint32_t preds) {
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void ldg_force_64(uint4 (&dst)[3], const void* (&ptr)[3], uint32_t preds) {
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void ldg_force_64(uint4 (&dst)[4], const void* (&ptr)[4], uint32_t preds) {
#if !defined(__LWDACC_RTC__)
    const uint64_t zero = 0ull;
    asm volatile( \
        "{\n" \
        "\t.reg .pred p<4>;\n" \
        "\tr2p.b32 {p3, p2, p1, p0}, %12.b0, 0x0f;\n" \
        "\n" \
        "\t    mov.b64 {%0, %1 }, %13;\n" \
        "\t@p0 ld.global.v2.b32 {%0, %1}, [%8];\n" \
        "\n" \
        "\t    mov.b64 {%2, %3 }, %13;\n" \
        "\t@p1 ld.global.v2.b32 {%2, %3}, [%9];\n" \
        "\n" \
        "\t    mov.b64 {%4, %5 }, %13;\n" \
        "\t@p2 ld.global.v2.b32 {%4, %5}, [%10];\n" \
        "\n" \
        "\t    mov.b64 {%6, %7}, %13;\n" \
        "\t@p3 ld.global.v2.b32 {%6, %7}, [%11];\n" \
        "}\n"
            : "=r"(dst[0].x), "=r"(dst[0].y)
            , "=r"(dst[1].x), "=r"(dst[1].y)
            , "=r"(dst[2].x), "=r"(dst[2].y)
            , "=r"(dst[3].x), "=r"(dst[3].y)
            :  "l"(ptr[0])
            ,  "l"(ptr[1])
            ,  "l"(ptr[2])
            ,  "l"(ptr[3])
            ,  "r"(preds)
            ,  "l"(zero));
#else
    bool p[4];
    #pragma unroll
    for( int i = 0; i < 4; ++i ) {
        p[i] = ((1u << i) & preds);
        ldg_force_64(dst[i], ptr[i], p[i]);
    }
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void ldg_force_64(uint4 (&dst)[5], const void* (&ptr)[5], uint32_t preds) {
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void ldg_force_64(uint4 (&dst)[6], const void* (&ptr)[6], uint32_t preds) {
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void ldg_force_64(uint4 (&dst)[7], const void* (&ptr)[7], uint32_t preds) {
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void ldg_force_64(uint4 (&dst)[8], const void* (&ptr)[8], uint32_t preds) {
#if !defined(__LWDACC_RTC__)
    const uint64_t zero = 0ull;
    asm volatile( \
        "{\n" \
        "\t.reg .pred p<4>;\n" \
        "\tr2p.b32 {p3, p2, p1, p0}, %12.b0, 0x0f;\n" \
        "\n" \
        "\t    mov.b64 {%0, %1 }, %13;\n" \
        "\t@p0 ld.global.v2.b32 {%0, %1}, [%8];\n" \
        "\n" \
        "\t    mov.b64 {%2, %3 }, %13;\n" \
        "\t@p1 ld.global.v2.b32 {%2, %3}, [%9];\n" \
        "\n" \
        "\t    mov.b64 {%4, %5 }, %13;\n" \
        "\t@p2 ld.global.v2.b32 {%4, %5}, [%10];\n" \
        "\n" \
        "\t    mov.b64 {%6, %7}, %13;\n" \
        "\t@p3 ld.global.v2.b32 {%6, %7}, [%11];\n" \
        "}\n"
            : "=r"(dst[0].x), "=r"(dst[0].y)
            , "=r"(dst[1].x), "=r"(dst[1].y)
            , "=r"(dst[2].x), "=r"(dst[2].y)
            , "=r"(dst[3].x), "=r"(dst[3].y)
            :  "l"(ptr[0])
            ,  "l"(ptr[1])
            ,  "l"(ptr[2])
            ,  "l"(ptr[3])
            ,  "r"(preds)
            ,  "l"(zero));

    asm volatile( \
        "{\n" \
        "\t.reg .pred p<4>;\n" \
        "\tr2p.b32 {p3, p2, p1, p0}, %12.b1, 0x0f;\n" \
        "\n" \
        "\t    mov.b64 {%0, %1 }, %13;\n" \
        "\t@p0 ld.global.v2.b32 {%0, %1}, [%8];\n" \
        "\n" \
        "\t    mov.b64 {%2, %3 }, %13;\n" \
        "\t@p1 ld.global.v2.b32 {%2, %3}, [%9];\n" \
        "\n" \
        "\t    mov.b64 {%4, %5 }, %13;\n" \
        "\t@p2 ld.global.v2.b32 {%4, %5}, [%10];\n" \
        "\n" \
        "\t    mov.b64 {%6, %7}, %13;\n" \
        "\t@p3 ld.global.v2.b32 {%6, %7}, [%11];\n" \
        "}\n"
            : "=r"(dst[4].x), "=r"(dst[4].y)
            , "=r"(dst[5].x), "=r"(dst[5].y)
            , "=r"(dst[6].x), "=r"(dst[6].y)
            , "=r"(dst[7].x), "=r"(dst[7].y)
            :  "l"(ptr[4])
            ,  "l"(ptr[5])
            ,  "l"(ptr[6])
            ,  "l"(ptr[7])
            ,  "r"(preds)
            ,  "l"(zero));
#else
    bool p[4];
    #pragma unroll
    for( int i = 0; i < 4; ++i ) {
        p[i] = ((1u << i) & preds);
        ldg_force_64(dst[i], ptr[i], p[i]);
    }

    #pragma unroll
    for( int i = 0; i < 4; ++i ) {
        p[i] = ((1u << (i+8)) & preds);
        ldg_force_64(dst[i + 4], ptr[i + 4], p[i]);
    }
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void ldg_force_64(uint4 (&dst)[9], const void* (&ptr)[9], uint32_t preds) {
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void ldg_force_64(uint4 (&dst)[10], const void* (&ptr)[10], uint32_t preds) {
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void ldg_force_64(uint4 (&dst)[11], const void* (&ptr)[11], uint32_t preds) {
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void ldg_force_64(uint4 (&dst)[12], const void* (&ptr)[12], uint32_t preds) {
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void ldg_force_64(uint4 (&dst)[13], const void* (&ptr)[13], uint32_t preds) {
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void ldg_force_64(uint4 (&dst)[16], const void* (&ptr)[16], uint32_t preds) {
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__
    uint32_t scale_bias(const uint32_t &fetch, const uint32_t &scale, const uint32_t &bias) {
    uint32_t d;
    asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n"
                    : "=r"(d)
                    : "r"(scale), "r"(fetch), "r"(bias));
    return d;
}

inline __device__
    uint32_t scale_bias(const uint32_t &fetch, const uint16_t &scale, const uint16_t &bias) {
    uint32_t d;
    asm volatile( "{ \n\t"
                  " .reg .b32 s;\n\t"    \
                  " .reg .b32 b;\n\t"    \
                  " mov.b32 s, {%1,%1};\n\t" \
                  " mov.b32 b, {%3,%3};\n\t" \
                  " fma.rn.f16x2 %0, %2, s, b;\n\t" \
                  "}"
                  : "=r"(d)
                  : "h"(scale), "r"(fetch), "h"(bias));
    return d;
}

template <bool WITH_RELU>
inline __device__
    uint32_t scale_bias_relu(const uint32_t &fetch, const uint32_t &scale, const uint32_t &bias) {
    uint32_t d;
#if defined(__LWDA_ARCH__) && __LWDA_ARCH__ >= 800
    if ( WITH_RELU ) {
        asm volatile("fma.rn.f16x2.relu %0, %1, %2, %3;\n"
                        : "=r"(d)
                        : "r"(scale), "r"(fetch), "r"(bias));
    } else {
        asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n"
                        : "=r"(d)
                        : "r"(scale), "r"(fetch), "r"(bias));
    }
#else
    d = scale_bias(fetch, scale, bias);
    if ( WITH_RELU ) {
        d = relu_fp16x2(d);
    }
#endif
    return d;
}

// Share the same __half value of scale and bias across the FP16x2 (uint32_t) input (fetch)
template <bool WITH_RELU>
inline __device__
    uint32_t scale_bias_relu(const uint32_t &fetch, const uint16_t &scale, const uint16_t &bias) {

#if defined(__LWDA_ARCH__) && __LWDA_ARCH__ >= 800
    uint32_t result;

    if( WITH_RELU ) {
        asm volatile(   \
           "{\n\t"     \
           " .reg .b32 s;\n\t"    \
           " .reg .b32 b;\n\t"    \
           " mov.b32 s, {%1,%1};\n\t" \
           " mov.b32 b, {%3,%3};\n\t" \
           " fma.rn.f16x2.relu %0, %2, s, b;\n\t" \
           "}"
           : "=r"(result)
           : "h"(scale), "r"(fetch), "h"(bias));
    } else {
        asm volatile(   \
           "{\n\t"     \
           " .reg .b32 s;\n\t"    \
           " .reg .b32 b;\n\t"    \
           " mov.b32 s, {%1,%1};\n\t" \
           " mov.b32 b, {%3,%3};\n\t" \
           " fma.rn.f16x2 %0, %2, s, b;\n\t" \
           "}"
           : "=r"(result)
           : "h"(scale), "r"(fetch), "h"(bias));
    }

    return result;
#else
    __half2 d;

    // This should translate into hfma2 H0_H0 instructions
    __half2 in = reinterpret_cast<const __half2&>(fetch);
    __half scale_h = reinterpret_cast<const __half&>(scale);
    __half bias_h = reinterpret_cast<const __half&>(bias);

    d.x = in.x * scale_h + bias_h;
    d.y = in.y * scale_h + bias_h;

    uint32_t d_ = reinterpret_cast<uint32_t&>(d);

    if ( WITH_RELU ) {
        d_ = relu_fp16x2(d_);
    }

    return d_;
#endif
}

// Check if the value is a special NaN, if so - set it to zero, else do the scale and bias
// application
template <bool WITH_RELU>
inline __device__
    uint32_t guarded_scale_bias_relu_a(const uint32_t &fetch, const uint32_t &scale,
                                       const uint32_t &bias) {
    constexpr uint32_t oob_val = Oob_val<>::OOB_VAL_H2;

#if defined(__LWDA_ARCH__) && __LWDA_ARCH__ >= 800
    uint32_t d;
    if( WITH_RELU ) {
        asm volatile(   \
            "{\n\t"     \
            " .reg .pred %p;\n\t"   \
            " .reg .b32 t1;\n\t"     \
            " setp.eq.u32 %p, %2, %4;\n\t" \
            " fma.rn.f16x2.relu t1, %1, %2, %3;\n"
            " selp.u32 %0, 0, t1, %p;\n\t"   \
            "}"
            : "=r"(d)
            : "r"(scale), "r"(fetch), "r"(bias)
            , "r"(oob_val));
    } else {
        asm volatile(   \
            "{\n\t"     \
            " .reg .pred %p;\n\t"   \
            " .reg .b32 t1;\n\t"     \
            " setp.eq.u32 %p, %2, %4;\n\t" \
            " fma.rn.f16x2 t1, %1, %2, %3;\n"
            " selp.u32 %0, 0, t1, %p;\n\t"   \
            "}"
            : "=r"(d)
            : "r"(scale), "r"(fetch), "r"(bias)
            , "r"(oob_val));
    }
    return d;
#else
    if( fetch == oob_val ) {
        return 0;
    } else {
        return scale_bias_relu<WITH_RELU>(fetch, scale, bias);
    }
#endif
}

// Check if the value is a special NaN, if so - set it to zero, else do the scale and bias
// application. But When doing it for the B matrix, need extra care / checks - since only 1 out
// of the 2 half values can special be NaN
template <bool WITH_RELU>
inline __device__
    uint32_t guarded_scale_bias_relu_b(const uint32_t &fetch, const uint16_t &scale,
                                       const uint16_t &bias) {
    uint32_t tmp_fetch = fetch;
    __half2 &fetch_h = reinterpret_cast<__half2&>(tmp_fetch);

    uint16_t &h1 = reinterpret_cast<uint16_t&>(fetch_h.x);
    uint16_t &h2 = reinterpret_cast<uint16_t&>(fetch_h.y);

    bool h1_oob = h1 == Oob_val<>::OOB_VAL_H;
    bool h2_oob = h2 == Oob_val<>::OOB_VAL_H;

    tmp_fetch = scale_bias_relu<WITH_RELU>(tmp_fetch, scale, bias);

    // Mask the result based on OOB value
    tmp_fetch = h1_oob ? tmp_fetch & 0xFFFF0000 : tmp_fetch;
    tmp_fetch = h2_oob ? tmp_fetch & 0x0000FFFF : tmp_fetch;

    return tmp_fetch;

    // To be enabled only if necessary - make sure to change OOB val
    // To a non-NAN value
    #if 0
        uint32_t tmp_fetch = fetch;
        __half2 &fetch_h = reinterpret_cast<__half2&>(tmp_fetch);
        const uint32_t oob_val_ = Oob_val<>::OOB_VAL_H2;
        const __half2 oob_val = reinterpret_cast<const __half2&>(oob_val_);
        __half2 nan_chk = __hneu2 ( fetch_h, oob_val );

        uint32_t res = scale_bias_relu<WITH_RELU>(tmp_fetch, scale, bias);

        __half2 &res_h = reinterpret_cast<__half2&>(res);
        res_h = __hmul2( res_h , nan_chk );

        return reinterpret_cast<uint32_t&>(res);
    #endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__
    void scale_bias(uint4 (&fetch)[2], const uint4 &scale, const uint4 &bias, uint32_t preds) {
#if !defined(__LWDACC_RTC__)
    asm volatile( \
        "{\n" \
        "\t.reg .pred p<2>;\n" \
        "\tr2p.b32 {p1, p0}, %16.b0, 0x03;\n" \
        "\n" \
        "\t@p0 fma.rn.f16x2 %0, %8 , %0, %12;\n" \
        "\t@p0 fma.rn.f16x2 %1, %9 , %1, %13;\n" \
        "\t@p0 fma.rn.f16x2 %2, %10, %2, %14;\n" \
        "\t@p0 fma.rn.f16x2 %3, %11, %3, %15;\n" \
        "\n" \
        "\t@p1 fma.rn.f16x2 %4, %8 , %4, %12;\n" \
        "\t@p1 fma.rn.f16x2 %5, %9 , %5, %13;\n" \
        "\t@p1 fma.rn.f16x2 %6, %10, %6, %14;\n" \
        "\t@p1 fma.rn.f16x2 %7, %11, %7, %15;\n" \
        "}\n"
            : "+r"(fetch[0].x), "+r"(fetch[0].y), "+r"(fetch[0].z), "+r"(fetch[0].w)
            , "+r"(fetch[1].x), "+r"(fetch[1].y), "+r"(fetch[1].z), "+r"(fetch[1].w)
            :  "r"(scale.x),     "r"(scale.y),     "r"(scale.z),     "r"(scale.w)
            ,  "r"(bias.x),      "r"(bias.y),      "r"(bias.z),      "r"(bias.w)
            ,  "r"(preds));
#else
    bool p[2];
    #pragma unroll
    for( int i = 0; i < 2; ++i ) {
        p[i] = ((1u << i) & preds);
        if( p[i] ) {
            fetch[i].x = hfma2(fetch[i].x, scale.x, bias.x);
            fetch[i].y = hfma2(fetch[i].y, scale.y, bias.y);
            fetch[i].z = hfma2(fetch[i].z, scale.z, bias.z);
            fetch[i].w = hfma2(fetch[i].w, scale.w, bias.w);
        }
    }
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__
    void scale_bias(uint4 (&fetch)[4], const uint4 &scale, const uint4 &bias, uint32_t preds) {
#if !defined(__LWDACC_RTC__)
    asm volatile( \
        "{\n" \
        "\t.reg .pred p<4>;\n" \
        "\tr2p.b32 {p3, p2, p1, p0}, %24.b0, 0x0f;\n" \
        "\n" \
        "\t@p0 fma.rn.f16x2 %0 , %16, %0 , %20;\n" \
        "\t@p0 fma.rn.f16x2 %1 , %17, %1 , %21;\n" \
        "\t@p0 fma.rn.f16x2 %2 , %18, %2 , %22;\n" \
        "\t@p0 fma.rn.f16x2 %3 , %19, %3 , %23;\n" \
        "\n" \
        "\t@p1 fma.rn.f16x2 %4 , %16, %4 , %20;\n" \
        "\t@p1 fma.rn.f16x2 %5 , %17, %5 , %21;\n" \
        "\t@p1 fma.rn.f16x2 %6 , %18, %6 , %22;\n" \
        "\t@p1 fma.rn.f16x2 %7 , %19, %7 , %23;\n" \
        "\n" \
        "\t@p2 fma.rn.f16x2 %8 , %16, %8 , %20;\n" \
        "\t@p2 fma.rn.f16x2 %9 , %17, %9 , %21;\n" \
        "\t@p2 fma.rn.f16x2 %10, %18, %10, %22;\n" \
        "\t@p2 fma.rn.f16x2 %11, %19, %11, %23;\n" \
        "\n" \
        "\t@p3 fma.rn.f16x2 %12, %16, %12, %20;\n" \
        "\t@p3 fma.rn.f16x2 %13, %17, %13, %21;\n" \
        "\t@p3 fma.rn.f16x2 %14, %18, %14, %22;\n" \
        "\t@p3 fma.rn.f16x2 %15, %19, %15, %23;\n" \
        "}\n"
            : "+r"(fetch[0].x), "+r"(fetch[0].y), "+r"(fetch[0].z), "+r"(fetch[0].w)
            , "+r"(fetch[1].x), "+r"(fetch[1].y), "+r"(fetch[1].z), "+r"(fetch[1].w)
            , "+r"(fetch[2].x), "+r"(fetch[2].y), "+r"(fetch[2].z), "+r"(fetch[2].w)
            , "+r"(fetch[3].x), "+r"(fetch[3].y), "+r"(fetch[3].z), "+r"(fetch[3].w)
            :  "r"(scale.x),     "r"(scale.y),     "r"(scale.z),     "r"(scale.w)
            ,  "r"(bias.x),      "r"(bias.y),      "r"(bias.z),      "r"(bias.w)
            ,  "r"(preds));
#else
    bool p[4];
    #pragma unroll
    for( int i = 0; i < 4; ++i ) {
        p[i] = ((1u << i) & preds);
        if( p[i] ) {
            fetch[i].x = hfma2(fetch[i].x, scale.x, bias.x);
            fetch[i].y = hfma2(fetch[i].y, scale.y, bias.y);
            fetch[i].z = hfma2(fetch[i].z, scale.z, bias.z);
            fetch[i].w = hfma2(fetch[i].w, scale.w, bias.w);
        }
    }
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__
    void scale_bias(uint4 (&fetch)[8], const uint4 &scale, const uint4 &bias, uint32_t preds) {
#if !defined(__LWDACC_RTC__)
    asm volatile( \
        "{\n" \
        "\t.reg .pred p<4>;\n" \
        "\tr2p.b32 {p3, p2, p1, p0}, %24.b0, 0x0f;\n" \
        "\n" \
        "\t@p0 fma.rn.f16x2 %0 , %16, %0 , %20;\n" \
        "\t@p0 fma.rn.f16x2 %1 , %17, %1 , %21;\n" \
        "\t@p0 fma.rn.f16x2 %2 , %18, %2 , %22;\n" \
        "\t@p0 fma.rn.f16x2 %3 , %19, %3 , %23;\n" \
        "\n" \
        "\t@p1 fma.rn.f16x2 %4 , %16, %4 , %20;\n" \
        "\t@p1 fma.rn.f16x2 %5 , %17, %5 , %21;\n" \
        "\t@p1 fma.rn.f16x2 %6 , %18, %6 , %22;\n" \
        "\t@p1 fma.rn.f16x2 %7 , %19, %7 , %23;\n" \
        "\n" \
        "\t@p2 fma.rn.f16x2 %8 , %16, %8 , %20;\n" \
        "\t@p2 fma.rn.f16x2 %9 , %17, %9 , %21;\n" \
        "\t@p2 fma.rn.f16x2 %10, %18, %10, %22;\n" \
        "\t@p2 fma.rn.f16x2 %11, %19, %11, %23;\n" \
        "\n" \
        "\t@p3 fma.rn.f16x2 %12, %16, %12, %20;\n" \
        "\t@p3 fma.rn.f16x2 %13, %17, %13, %21;\n" \
        "\t@p3 fma.rn.f16x2 %14, %18, %14, %22;\n" \
        "\t@p3 fma.rn.f16x2 %15, %19, %15, %23;\n" \
        "}\n"
            : "+r"(fetch[0].x), "+r"(fetch[0].y), "+r"(fetch[0].z), "+r"(fetch[0].w)
            , "+r"(fetch[1].x), "+r"(fetch[1].y), "+r"(fetch[1].z), "+r"(fetch[1].w)
            , "+r"(fetch[2].x), "+r"(fetch[2].y), "+r"(fetch[2].z), "+r"(fetch[2].w)
            , "+r"(fetch[3].x), "+r"(fetch[3].y), "+r"(fetch[3].z), "+r"(fetch[3].w)
            :  "r"(scale.x),     "r"(scale.y),     "r"(scale.z),     "r"(scale.w)
            ,  "r"(bias.x),      "r"(bias.y),      "r"(bias.z),      "r"(bias.w)
            ,  "r"(preds));

    asm volatile( \
        "{\n" \
        "\t.reg .pred p<4>;\n" \
        "\tr2p.b32 {p3, p2, p1, p0}, %24.b1, 0x0f;\n" \
        "\n" \
        "\t@p0 fma.rn.f16x2 %0 , %16, %0 , %20;\n" \
        "\t@p0 fma.rn.f16x2 %1 , %17, %1 , %21;\n" \
        "\t@p0 fma.rn.f16x2 %2 , %18, %2 , %22;\n" \
        "\t@p0 fma.rn.f16x2 %3 , %19, %3 , %23;\n" \
        "\n" \
        "\t@p1 fma.rn.f16x2 %4 , %16, %4 , %20;\n" \
        "\t@p1 fma.rn.f16x2 %5 , %17, %5 , %21;\n" \
        "\t@p1 fma.rn.f16x2 %6 , %18, %6 , %22;\n" \
        "\t@p1 fma.rn.f16x2 %7 , %19, %7 , %23;\n" \
        "\n" \
        "\t@p2 fma.rn.f16x2 %8 , %16, %8 , %20;\n" \
        "\t@p2 fma.rn.f16x2 %9 , %17, %9 , %21;\n" \
        "\t@p2 fma.rn.f16x2 %10, %18, %10, %22;\n" \
        "\t@p2 fma.rn.f16x2 %11, %19, %11, %23;\n" \
        "\n" \
        "\t@p3 fma.rn.f16x2 %12, %16, %12, %20;\n" \
        "\t@p3 fma.rn.f16x2 %13, %17, %13, %21;\n" \
        "\t@p3 fma.rn.f16x2 %14, %18, %14, %22;\n" \
        "\t@p3 fma.rn.f16x2 %15, %19, %15, %23;\n" \
        "}\n"
            : "+r"(fetch[4].x), "+r"(fetch[4].y), "+r"(fetch[4].z), "+r"(fetch[4].w)
            , "+r"(fetch[5].x), "+r"(fetch[5].y), "+r"(fetch[5].z), "+r"(fetch[5].w)
            , "+r"(fetch[6].x), "+r"(fetch[6].y), "+r"(fetch[6].z), "+r"(fetch[6].w)
            , "+r"(fetch[7].x), "+r"(fetch[7].y), "+r"(fetch[7].z), "+r"(fetch[7].w)
            :  "r"(scale.x),     "r"(scale.y),     "r"(scale.z),     "r"(scale.w)
            ,  "r"(bias.x),      "r"(bias.y),      "r"(bias.z),      "r"(bias.w)
            ,  "r"(preds));
#else
    bool p[4];
    #pragma unroll
    for( int i = 0; i < 4; ++i ) {
        p[i] = ((1u << i) & preds);
        if( p[i] ) {
            fetch[i].x = hfma2(fetch[i].x, scale.x, bias.x);
            fetch[i].y = hfma2(fetch[i].y, scale.y, bias.y);
            fetch[i].z = hfma2(fetch[i].z, scale.z, bias.z);
            fetch[i].w = hfma2(fetch[i].w, scale.w, bias.w);
        }
    }

    #pragma unroll
    for( int i = 0; i < 4; ++i ) {
        p[i] = ((1u << (i + 8)) & preds);
        if( p[i] ) {
            fetch[4 + i].x = hfma2(fetch[4 + i].x, scale.x, bias.x);
            fetch[4 + i].y = hfma2(fetch[4 + i].y, scale.y, bias.y);
            fetch[4 + i].z = hfma2(fetch[4 + i].z, scale.z, bias.z);
            fetch[4 + i].w = hfma2(fetch[4 + i].w, scale.w, bias.w);
        }
    }
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// S E L E C T O R   T O   P I C K   T H E   C O R R E C T   B A S E   C L A S S
//
////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    // The instruction traits.
    typename Traits,
    // The class with LDGSTS.
    typename Class_with_ldgsts,
    // The class without LDGSTS.
    typename Class_without_ldgsts,
    // Do we disable LDGSTS on an architecture that supports it?
    bool DISABLE_LDGSTS,
    // Do we use LDGSTS?
    bool USE_LDGSTS = Traits::Gpu_arch::HAS_LDGSTS && !DISABLE_LDGSTS
>
struct Ldgsts_selector {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    // The instruction traits.
    typename Traits,
    // The class with LDGSTS.
    typename Class_with_ldgsts,
    // The class without LDGSTS.
    typename Class_without_ldgsts
>
struct Ldgsts_selector<Traits, Class_with_ldgsts, Class_without_ldgsts, false, true> {
    using Class = Class_with_ldgsts;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    // The instruction traits.
    typename Traits,
    // The class with LDGSTS.
    typename Class_with_ldgsts,
    // The class without LDGSTS.
    typename Class_without_ldgsts,
    // Do we disable LDGSTS on an architecture that supports it?
    bool DISABLE_LDGSTS
>
struct Ldgsts_selector<Traits, Class_with_ldgsts, Class_without_ldgsts, DISABLE_LDGSTS, false> {
    using Class = Class_without_ldgsts;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<int N>
static inline __device__ void prefetch_l1(const char* (&ptrs)[N], bool (&mask)[N]) {
    #pragma unroll
    for( int i = 0; i < N; i++) {
        if( mask[N] ) {
            asm volatile("prefetch.global.L1 [%0];" : :"l"(ptrs[i]));
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static int
get_ctas_per_sm(lwdaFuncAttributes *attr, int shared_memory_size_per_sm, int registers_per_sm ) {
    int registers_per_thread = attr->numRegs;
    int threads_per_cta = attr->maxThreadsPerBlock;
    int shared_memory_size_per_cta = attr->maxDynamicSharedSizeBytes;
    int ctas_per_sm_register(registers_per_sm / (registers_per_thread * threads_per_cta));
    int ctas_per_sm;
    if (shared_memory_size_per_cta == 0) {
        ctas_per_sm = ctas_per_sm_register;
    } else {
        int ctas_per_sm_shared_memory(shared_memory_size_per_sm / shared_memory_size_per_cta);
        ctas_per_sm = (ctas_per_sm_register < ctas_per_sm_shared_memory
            ? ctas_per_sm_register : ctas_per_sm_shared_memory);
    }
    return ctas_per_sm;
}

static int get_ctas_per_sm(lwdaFuncAttributes *attr, lwdaDeviceProp *dev_prop) {
    int registers_per_thread = attr->numRegs;
    int threads_per_cta = attr->maxThreadsPerBlock;
    if( threads_per_cta > dev_prop->maxThreadsPerBlock ){
        return -1;
    }

    int shared_memory_size_per_cta = attr->maxDynamicSharedSizeBytes;
    int shared_memory_size_per_sm = static_cast<int>(dev_prop->sharedMemPerMultiprocessor);
    int registers_per_sm = dev_prop->regsPerMultiprocessor;
    int ctas_per_sm_register(registers_per_sm / (registers_per_thread * threads_per_cta));
    int ctas_per_sm_shared_memory(shared_memory_size_per_sm / shared_memory_size_per_cta);
    int ctas_per_sm = (ctas_per_sm_register < ctas_per_sm_shared_memory
            ? ctas_per_sm_register : ctas_per_sm_shared_memory);
    if( ctas_per_sm * threads_per_cta > dev_prop->maxThreadsPerMultiProcessor ) {
        ctas_per_sm = dev_prop->maxThreadsPerMultiProcessor / threads_per_cta;
    }
    return ctas_per_sm;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static int get_ctas_per_wave(lwdaFuncAttributes *attr,
                             int sm_count,
                             int shared_memory_size_per_sm,
                             int registers_per_sm) {
    return sm_count*get_ctas_per_sm(attr, shared_memory_size_per_sm, registers_per_sm);
}

static int get_ctas_per_wave(lwdaFuncAttributes *attr, lwdaDeviceProp *dev_prop) {
    return dev_prop->multiProcessorCount*get_ctas_per_sm(attr,dev_prop);
}

XMMA_HOST_DEVICE
static uint32_t colwert_tf32(float const &s) {
    uint32_t x = reinterpret_cast<uint32_t const &>(s);

#if defined(__LWDA_ARCH__)
    if (::isfinite(s)){
        x += 0x1000u;
    }
#else
    if (std::isfinite(s)){
        x += 0x1000u;
    }
#endif

    return (x);
}
////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float gelu(float ele, float literal0, float literal1, float literal2, float gelu_scale_){
   // origin: 0.5f * x * (1.0f + tanh(0.797885f * (x + 0.044715f * x * x * x)));
   // new: 0.5f * (x + x * tanh(x * (0.797885f + 0.0356774f * x * x)));
   // reduce two op
   float v0 = literal0 * ele;
   float v1 = __fmaf_rn(ele, v0, literal1);
   float v2 = v1       * ele;
   float v3;
   asm volatile ("tanh.approx.f32 %0, %1;" : "=f"(v3) : "f"(v2));
   float v4 = __fmaf_rn(ele, v3, ele);
   float v5 = literal2 * v4;
   ele = v5    * gelu_scale_;
   return ele;

}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float gelu_erf(float ele, float scale_){
	// scale * 0.5 * ele * (1.0 + erf( ele / sqrt(2.0)))
	constexpr float literal0 = 0.70710678118f;
	float var = 0.5f * ele * (1.0f + erff(ele * literal0));
    return scale_ * var;
}

} // namespace xmma

