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

template <int HAS_SUPER_HMMA_ = 0>
struct Volta : public Gpu_arch_base {
    // It has super HMMA
    enum { HAS_SUPER_HMMA = HAS_SUPER_HMMA_ };

    enum { MAX_DYNAMIC_SMEM_SIZE_BYTES = 64 * 1024 };
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Cta_tile, int K_PER_XMMA_ = 8 >
struct Volta_mma_tile {

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
        XMMAS_M = (Cta_tile::M + M_PER_XMMA_PER_CTA-1) / M_PER_XMMA_PER_CTA,
        XMMAS_N = (Cta_tile::N + N_PER_XMMA_PER_CTA-1) / N_PER_XMMA_PER_CTA,
        XMMAS_K = (Cta_tile::K + K_PER_XMMA_PER_CTA-1) / K_PER_XMMA_PER_CTA,
        XMMAS_GROUPS = (Cta_tile::GROUPS < 4 ? Cta_tile::GROUPS : 4)
    };

    // The number of elements computed per warp.
    enum {
        M_PER_WARP = XMMAS_M * M_PER_XMMA,
        N_PER_WARP = XMMAS_N * N_PER_XMMA,
        K_PER_WARP = XMMAS_K * K_PER_XMMA,
    };

    // Do we enable the fast path for LDS.
    enum { ENABLE_LDS_FAST_PATH = 0 };
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Cta_tile >
struct Volta_hmma_fp16_tile : public Volta_mma_tile<Cta_tile> {

    // The distribution of threads in the output tile.
    enum {
        THREADS_PER_XMMA_M = 16,
        THREADS_PER_XMMA_N = 2,
    };
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Volta_hmma_fp16_traits
    : public Traits<Volta<0>, lwtlass::half_t,
                           lwtlass::half_t,
                           lwtlass::half_t,
                           lwtlass::half_t,
                           lwtlass::half_t> {

    static const bool IS_GELU_ERF = false;

    enum { USE_SPLIT_K_WITH_OUTPUT_PRECISION = 1 };

    // The Cta tile.
    template< int M, int N, int K, int GROUPS = 1 >
    using Cta_tile = xmma::Cta_tile<
        Volta<0>, M, N, K, XMMA_DIV_UP(M, 64), XMMA_DIV_UP(N, 64), XMMA_DIV_UP(K, 32), GROUPS>;

    // The XMMA tile.
    template< typename Cta_tile >
    using Xmma_tile = Volta_hmma_fp16_tile<Cta_tile>;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Cta_tile >
struct Volta_hmma_fp32_tile : public Volta_mma_tile<Cta_tile> {

    // The distribution of threads in the output tile.
    enum {
        THREADS_PER_XMMA_M = 8,
        THREADS_PER_XMMA_N = 4,
    };
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Volta_hmma_fp32_traits
    : public Traits<Volta<0>, lwtlass::half_t, lwtlass::half_t, lwtlass::half_t, float, float> {
    
    static const bool IS_GELU_ERF = false;

    // The Cta tile.
    template< int M, int N, int K, int GROUPS = 1 >
    using Cta_tile = xmma::Cta_tile<
        Volta<0>, M, N, K, XMMA_DIV_UP(M, 64), XMMA_DIV_UP(N, 64), XMMA_DIV_UP(K, 32), GROUPS>;

    // The XMMA tile.
    template< typename Cta_tile >
    using Xmma_tile = Volta_hmma_fp32_tile<Cta_tile>;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Volta_hmma_fp32_interleaved_traits 
    : public Volta_hmma_fp32_traits {
    // Whether fuse operation is gelu.
    static const bool IS_GELU = false;
    // Whether to use epilogue fadd.
    static const bool IS_EPIFADD = false;
    // Whether fuse operation is swish.
    static const bool IS_SWISH = false;
};

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// I M M A . 8 NHWC/TN layout
//
////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Cta_tile >
struct Volta_imma_int8_tile : public Volta_mma_tile<Cta_tile, 16> {
    // The distribution of threads in the output tile.
    enum {
        THREADS_PER_XMMA_M = 8,
        THREADS_PER_XMMA_N = 4,
    };
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< bool IS_GELU_ERF_ >
struct Volta_imma_int8_int32_traits
    : public Traits<Volta<1>, int8_t, int8_t, int8_t, int32_t, float> {

    static const int32_t IS_GELU_ERF = IS_GELU_ERF_;

    // The Cta tile.
    template< int M, int N, int K, int GROUPS = 1 >
    using Cta_tile = xmma::Cta_tile<
        Volta<1>, M, N, K, XMMA_DIV_UP(M, 64), XMMA_DIV_UP(N, 64), XMMA_DIV_UP(K, 64), GROUPS>;

    // The XMMA tile.
    template< typename Cta_tile >
    using Xmma_tile = Volta_imma_int8_tile<Cta_tile>;
};

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// I M M A . 8 NC/32HW32 layout
//
////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Cta_tile, int K_PER_XMMA_ >
struct Volta_imma_interleaved_int8_tile {

    // The number of elements computed with a single warp-MMA.
    enum { M_PER_XMMA = 16, N_PER_XMMA = 32, K_PER_XMMA = K_PER_XMMA_ };

    // The number of elements computed with a single CTA-MMA.
    enum {
        M_PER_XMMA_PER_CTA = M_PER_XMMA * Cta_tile::WARPS_M,
        N_PER_XMMA_PER_CTA = N_PER_XMMA * Cta_tile::WARPS_N,
        K_PER_XMMA_PER_CTA = K_PER_XMMA * Cta_tile::WARPS_K *
                             (Cta_tile::GROUPS > 1 ? Cta_tile::WARPS_N : 1)
    };

    // The number of MMAs needed to compute the GEMM.
    enum {
        XMMAS_M = (Cta_tile::M + M_PER_XMMA_PER_CTA-1) / M_PER_XMMA_PER_CTA,
        XMMAS_N = (Cta_tile::N + N_PER_XMMA_PER_CTA-1) / N_PER_XMMA_PER_CTA,
        XMMAS_K = (Cta_tile::K + K_PER_XMMA_PER_CTA-1) / K_PER_XMMA_PER_CTA,
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

    // Do we enable the fast path for LDS.
    enum { ENABLE_LDS_FAST_PATH = 0 };
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Volta_imma_interleaved_int8_int32_traits
    : public Traits<Volta<1>, int8_t, int8_t, int8_t, int32_t, float> {

    // Whether fuse operation is gelu.
    static const bool IS_GELU = false;
    // Whether to use epilogue fadd.
    static const bool IS_EPIFADD = false;
    // Whether fuse operation is swish.
    static const bool IS_SWISH = false;

    // The Cta tile.
    template< int M, int N, int K, int GROUPS = 1 >
    using Cta_tile = xmma::Cta_tile<
        Volta<1>, M, N, K, XMMA_DIV_UP(M, 64), XMMA_DIV_UP(N, 64), XMMA_DIV_UP(K, 64), GROUPS>;

    // The XMMA tile.
    template< typename Cta_tile >
    using Xmma_tile = Volta_imma_interleaved_int8_tile<Cta_tile, 32>;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace xmma

