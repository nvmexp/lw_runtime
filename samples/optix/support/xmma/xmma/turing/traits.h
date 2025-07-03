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

#include <xmma/utils.h>
#include <xmma/traits.h>
#include <xmma/numeric_types.h>

namespace xmma {

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Turing : public Gpu_arch_base {

    enum { HAS_SUPER_HMMA = 1 };

    enum { MAX_DYNAMIC_SMEM_SIZE_BYTES = 48 * 1024 };
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Cta_tile, int K_PER_XMMA_ >
struct Turing_mma_tile {

    // The number of elements computed with a single warp-MMA.
    enum { M_PER_XMMA = 16, N_PER_XMMA = 16, K_PER_XMMA = K_PER_XMMA_ };

    // The number of elements computed with a single CTA-MMA.
    enum {
        M_PER_XMMA_PER_CTA = M_PER_XMMA * Cta_tile::WARPS_M,
        N_PER_XMMA_PER_CTA = N_PER_XMMA * Cta_tile::WARPS_N,
        K_PER_XMMA_PER_CTA = K_PER_XMMA * Cta_tile::WARPS_K *
                             (Cta_tile::GROUPS > 1 ? Cta_tile::WARPS_N : 1)
    };

    // The number of MMAs needed to compute the GEMM.
    enum {
        XMMAS_M = Div_up<Cta_tile::M, M_PER_XMMA_PER_CTA>::VALUE,
        XMMAS_N = Div_up<Cta_tile::N, N_PER_XMMA_PER_CTA>::VALUE,
        XMMAS_K = Div_up<Cta_tile::K, K_PER_XMMA_PER_CTA>::VALUE,

        XMMAS_GROUPS = (Cta_tile::GROUPS < 4 ? Cta_tile::GROUPS : 4)
    };

    // The number of elements computed per warp.
    enum {
        M_PER_WARP = XMMAS_M * M_PER_XMMA,
        N_PER_WARP = XMMAS_N * N_PER_XMMA,
        K_PER_WARP = XMMAS_K * K_PER_XMMA,
    };

    // The distribution of threads in the output tile.
    enum {
        THREADS_PER_XMMA_M = 8,
        THREADS_PER_XMMA_N = 4,
    };
};

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// H M M A . F 1 6 / F 3 2
//
////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Cta_tile >
struct Turing_hmma_tile : public Turing_mma_tile<Cta_tile, 8> {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Turing_hmma_fp16_traits
    : public Traits<Turing, lwtlass::half_t,
                            lwtlass::half_t,
                            lwtlass::half_t,
                            lwtlass::half_t,
                            lwtlass::half_t> {
    
    static const bool IS_GELU_ERF = false;

    enum { USE_SPLIT_K_WITH_OUTPUT_PRECISION = 1 };
    
    // The Cta tile.
    template< int M, int N, int K, int GROUPS = 1 >
    using Cta_tile = xmma::Cta_tile<
        Turing, M, N, K, XMMA_DIV_UP(M, 64), XMMA_DIV_UP(N, 64), XMMA_DIV_UP(K, 32), GROUPS>;

    // The XMMA tile.
    template< typename Cta_tile >
    using Xmma_tile = Turing_hmma_tile<Cta_tile>;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Turing_hmma_fp32_traits
    : public Traits<Turing, lwtlass::half_t, lwtlass::half_t, lwtlass::half_t, float, float> {

    static const bool IS_GELU_ERF = false;

    // The Cta tile.
    template< int M, int N, int K, int GROUPS = 1 >
    using Cta_tile = xmma::Cta_tile<
        Turing, M, N, K, XMMA_DIV_UP(M, 64), XMMA_DIV_UP(N, 64), XMMA_DIV_UP(K, 32), GROUPS>;

    // The XMMA tile.
    template< typename Cta_tile >
    using Xmma_tile = Turing_hmma_tile<Cta_tile>;
};

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// I M M A . 8
//
////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Cta_tile >
struct Turing_imma_int8_tile : public Turing_mma_tile<Cta_tile, 16> {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template < bool IS_GELU_ERF_ >
struct Turing_imma_int8_int32_traits
    : public Traits<Turing, int8_t, int8_t, int8_t, int32_t, float> {

    static const bool IS_GELU_ERF = IS_GELU_ERF_;

    // The Cta tile.
    template< int M, int N, int K, int GROUPS = 1 >
    using Cta_tile = xmma::Cta_tile<
        Turing, M, N, K, XMMA_DIV_UP(M, 64), XMMA_DIV_UP(N, 64), XMMA_DIV_UP(K, 64), GROUPS>;

    // The XMMA tile.
    template< typename Cta_tile >
    using Xmma_tile = Turing_imma_int8_tile<Cta_tile>;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Cta_tile, int K_PER_XMMA_ >
struct Turing_imma_interleaved_int8_tile {

    // The number of elements computed with a single warp-MMA.
    enum { M_PER_XMMA = 16, N_PER_XMMA = 32, K_PER_XMMA = K_PER_XMMA_ };

    // The number of elements computed with a single CTA-MMA.
    enum {
        M_PER_XMMA_PER_CTA = M_PER_XMMA * Cta_tile::WARPS_M,
        N_PER_XMMA_PER_CTA = N_PER_XMMA * Cta_tile::WARPS_N,
        K_PER_XMMA_PER_CTA = K_PER_XMMA * Cta_tile::WARPS_K
    };

    // The number of MMAs needed to compute the GEMM.
    enum {
        XMMAS_M = (Cta_tile::M + M_PER_XMMA_PER_CTA-1) / M_PER_XMMA_PER_CTA,
        XMMAS_N = (Cta_tile::N + N_PER_XMMA_PER_CTA-1) / N_PER_XMMA_PER_CTA,
        XMMAS_K = (Cta_tile::K / Cta_tile::GROUPS + K_PER_XMMA_PER_CTA-1) / K_PER_XMMA_PER_CTA,
        XMMAS_GROUPS = (Cta_tile::GROUPS < 2 ? Cta_tile::GROUPS : 2)
    };

    // The number of elements computed per warp.
    enum {
        M_PER_WARP = XMMAS_M * M_PER_XMMA,
        N_PER_WARP = XMMAS_N * N_PER_XMMA,
        K_PER_WARP = XMMAS_K * K_PER_XMMA,
    };

    // The distribution of threads in the output tile.
    enum {
        THREADS_PER_XMMA_M = 8,
        THREADS_PER_XMMA_N = 4,
    };
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Turing_imma_interleaved_int8_int32_traits
    : public Traits<Turing, int8_t, int8_t, int8_t, int32_t, float> {

    // Whether fuse operation is gelu.
    static const bool IS_GELU = false;
    // Whether to use epilogue fadd.
    static const bool IS_EPIFADD = false;
    //Whether fuse operation is gelu.
    static const bool IS_SWISH = false;

    // The Cta tile.
    template< int M, int N, int K, int GROUPS = 1 >
    using Cta_tile = xmma::Cta_tile<
        Turing, M, N, K, XMMA_DIV_UP(M, 64), XMMA_DIV_UP(N, 64), XMMA_DIV_UP(K, 64), GROUPS>;

    // The XMMA tile.
    template< typename Cta_tile >
    using Xmma_tile = Turing_imma_interleaved_int8_tile<Cta_tile, 32>;
};

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// I M M A . 4
//
////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Cta_tile >
struct Turing_imma_int4_tile : public Turing_mma_tile<Cta_tile, 32> {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Turing_imma_int4_int32_traits
    : public Traits<Turing, lwtlass::int4_t, lwtlass::int4_t, int32_t, int32_t, int32_t> {

    // The Cta tile.
    template< int M, int N, int K, int GROUPS = 1 >
    using Cta_tile = xmma::Cta_tile<
        Turing, M, N, K, XMMA_DIV_UP(M, 64), XMMA_DIV_UP(N, 64), XMMA_DIV_UP(K, 128), GROUPS>;

    // The XMMA tile.
    template< typename Cta_tile >
    using Xmma_tile = Turing_imma_int4_tile<Cta_tile>;

    // The number of bits per element of A.
    enum { BITS_PER_ELEMENT_A = 4 };

    // An offset in bytes for A.
    static inline __host__ __device__ int64_t offset_in_bytes_a(int64_t offset) {
        return offset / 2;
    }

    // The number of bits per element of B.
    enum { BITS_PER_ELEMENT_B = 4 };

    // An offset in bytes for B.
    static inline __host__ __device__ int64_t offset_in_bytes_b(int64_t offset) {
        return offset / 2;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// B M M A
//
////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Cta_tile >
struct Turing_bmma_tile : public Turing_mma_tile<Cta_tile, 128> {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Turing_bmma_int32_traits
    : public Traits<Turing, bool, bool, int32_t, int32_t, int32_t> {

    // The Cta tile.
    template< int M, int N, int K, int GROUPS = 1 >
    using Cta_tile = xmma::Cta_tile<
        Turing, M, N, K, XMMA_DIV_UP(M, 64), XMMA_DIV_UP(N, 64), XMMA_DIV_UP(K, 512), GROUPS>;

    // The XMMA tile.
    template< typename Cta_tile >
    using Xmma_tile = Turing_bmma_tile<Cta_tile>;

    // The number of bits per element of A.
    enum { BITS_PER_ELEMENT_A = 1 };

    // An offset in bytes for A.
    static inline __host__ __device__ size_t offset_in_bytes_a(size_t offset) {
        return offset / 8;
    }

    // The number of bits per element of B.
    enum { BITS_PER_ELEMENT_B = 1 };

    // An offset in bytes for B.
    static inline __host__ __device__ size_t offset_in_bytes_b(size_t offset) {
        return offset / 8;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace xmma

