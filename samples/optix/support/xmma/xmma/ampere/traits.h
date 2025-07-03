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
#include <xmma/turing/traits.h>

#include <xmma/numeric_types.h>

namespace xmma {

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Ampere : public Gpu_arch_base {
    // It has LDGSTS.
    enum { HAS_LDGSTS = 1 };
    // It has super HMMA
    enum { HAS_SUPER_HMMA = 1 };

    enum { MAX_DYNAMIC_SMEM_SIZE_BYTES = 164 * 1024 };
};

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// H M M A . F 1 6 / F 3 2
//
////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Cta_tile, int K_PER_XMMA = 16 >
struct Ampere_hmma_tile : public Turing_mma_tile<Cta_tile, K_PER_XMMA> {
};

////////////////////////////////////////////////////////////////////////////////////////////////////
struct Ampere_hmma_fp16_traits
    : public Traits<Ampere, lwtlass::half_t,
                            lwtlass::half_t,
                            lwtlass::half_t,
                            lwtlass::half_t,
                            lwtlass::half_t> {

    enum { USE_SPLIT_K_WITH_OUTPUT_PRECISION = 1 };
    static const bool IS_RT_FUSE = true;
    static const bool IS_GELU_ERF = false;

    // The Cta tile.
    template< int M, int N, int K, int GROUPS = 1 >
    using Cta_tile = xmma::Cta_tile<
        Ampere, M, N, K, XMMA_DIV_UP(M, 64), XMMA_DIV_UP(N, 64), XMMA_DIV_UP(K, 64), GROUPS>;

    // The XMMA tile.
    template< typename Cta_tile >
    using Xmma_tile = Ampere_hmma_tile<Cta_tile>;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Ampere_hmma_fp32_traits
    : public Traits<Ampere, lwtlass::half_t, lwtlass::half_t, lwtlass::half_t, float, float> {

    static const bool IS_RT_FUSE = true;
    static const bool IS_GELU_ERF = false;

    // The Cta tile.
    template< int M, int N, int K, int GROUPS = 1 >
    using Cta_tile = xmma::Cta_tile<
        Ampere, M, N, K, XMMA_DIV_UP(M, 64), XMMA_DIV_UP(N, 64), XMMA_DIV_UP(K, 64), GROUPS>;

    // The XMMA tile.
    template< typename Cta_tile >
    using Xmma_tile = Ampere_hmma_tile<Cta_tile>;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Input_type, typename Output_type >
struct Ampere_hmma_bf16_traits
    : public Traits<Ampere, Input_type, Input_type, Output_type, float, float> {

    static const bool IS_GELU_ERF = false;
    
    // The Cta tile.
    template< int M, int N, int K, int GROUPS = 1 >
    using Cta_tile = xmma::Cta_tile<
        Ampere, M, N, K, XMMA_DIV_UP(M, 64), XMMA_DIV_UP(N, 64), XMMA_DIV_UP(K, 64), GROUPS>;

    // The XMMA tile.
    template< typename Cta_tile >
    using Xmma_tile = Ampere_hmma_tile<Cta_tile>;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Input_type, typename Output_type >
struct Ampere_hmma_tf32_traits
    : public Traits<Ampere, Input_type, Input_type, Output_type, float, float> {

    static const bool IS_GELU_ERF = false;

    // The Cta tile.
    template< int M, int N, int K, int GROUPS = 1 >
    using Cta_tile = xmma::Cta_tile<
        Ampere, M, N, K, XMMA_DIV_UP(M, 64), XMMA_DIV_UP(N, 64), XMMA_DIV_UP(K, 32), GROUPS>;

    // The XMMA tile.
    template< typename Cta_tile >
    using Xmma_tile = Ampere_hmma_tile<Cta_tile, 8>;
};

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// I M M A . 8
//
////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Cta_tile >
struct Ampere_imma_int8_tile : public Turing_mma_tile<Cta_tile, 32> {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< bool IS_GELU_ERF_ >
struct Ampere_imma_int8_int32_traits
    : public Traits<Ampere, int8_t, int8_t, int8_t, int32_t, float> {

    static const bool IS_GELU_ERF = IS_GELU_ERF_;

    // The Cta tile.
    template< int M, int N, int K, int GROUPS = 1 >
    using Cta_tile = xmma::Cta_tile<
        Ampere, M, N, K, XMMA_DIV_UP(M, 64), XMMA_DIV_UP(N, 64), XMMA_DIV_UP(K, 64), GROUPS>;

    // The XMMA tile.
    template< typename Cta_tile >
    using Xmma_tile = Ampere_imma_int8_tile<Cta_tile>;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Ampere_imma_wo_epi_swizzle_int8_int32_traits
    : public Ampere_imma_int8_int32_traits<false> {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Cta_tile, int K_PER_XMMA_ >
struct Ampere_imma_interleaved_int8_tile
    : public Turing_imma_interleaved_int8_tile<Cta_tile, K_PER_XMMA_> {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Input_type,
         typename Output_type,
         bool IS_GELU_ = false,
         bool IS_EPIFADD_ = false,
         bool IS_SWISH_ = false,
         bool IS_RT_FUSE_ = false>
struct Ampere_imma_interleaved_traits
    : public Traits<Ampere, Input_type, Input_type, Output_type, int32_t, float> {

    // Whether fuse operation is gelu.
    static const bool IS_GELU = IS_GELU_;
    // Whether to use epilogue fadd.
    static const bool IS_EPIFADD = IS_EPIFADD_;
    // Whether fuse operation is swish.
    static const bool IS_SWISH = IS_SWISH_;
    // Whether fuse operation is runtime generated.
    static const bool IS_RT_FUSE = true;

    // The Cta tile.
    template< int M, int N, int K, int GROUPS = 1 >
    using Cta_tile = xmma::Cta_tile<
    Ampere, M, N, K, (M == 256) ? 4 : 2, (N == 256) ? 4 : 2, XMMA_DIV_UP(K, 64), GROUPS>;

    // The XMMA tile.
    template< typename Cta_tile >
    using Xmma_tile = Ampere_imma_interleaved_int8_tile<Cta_tile, 32>;
};

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// D M M A . F P 6 4
//
////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Cta_tile, int K_PER_XMMA_=4 >
struct Ampere_dmma_tile {

    // The number of elements computed with a single warp-MMA.
    enum { M_PER_XMMA = 8, N_PER_XMMA = 8, K_PER_XMMA = K_PER_XMMA_ };

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
        XMMAS_K = (Cta_tile::K + K_PER_XMMA_PER_CTA-1) / K_PER_XMMA_PER_CTA,
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


struct Ampere_dmma_fp64_traits
    : public Traits<Ampere, double, double, double, double, double> {

    static const bool IS_GELU_ERF = false;

    // The Cta tile.
    template< int M, int N, int K, int GROUPS = 1 >
    using Cta_tile = xmma::Cta_tile<
        Ampere, M, N, K, XMMA_DIV_UP(M, 32), M == 64 ? XMMA_DIV_UP(N, 32) : 2, XMMA_DIV_UP(K, 16), GROUPS>;

    // The XMMA tile.
    template< typename Cta_tile >
    using Xmma_tile = Ampere_dmma_tile<Cta_tile>;
};

} // namespace xmma

