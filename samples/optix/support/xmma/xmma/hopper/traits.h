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

struct Hopper : public Gpu_arch_base {
    // It has LDGSTS.
    enum { HAS_LDGSTS = 1 };
    // It has super HMMA
    enum { HAS_SUPER_HMMA = 1 };
    // It has HGMMA
    enum { HAS_HGMMA = 1 };
    // 4 warps per warp group
    enum { WARPS_PER_WARP_GROUP = 4 };

    enum { HAS_UTMALDG = 1 };

    enum { MAX_DYNAMIC_SMEM_SIZE_BYTES = 228 * 1024 };
};

template <typename Cta_tile, int K_PER_XMMA = 16>
struct Hopper_hmma_tile : public Turing_mma_tile<Cta_tile, K_PER_XMMA> {};

struct Hopper_hmma_fp16_traits : public Traits<Hopper,
                                               lwtlass::half_t,
                                               lwtlass::half_t,
                                               lwtlass::half_t,
                                               lwtlass::half_t,
                                               lwtlass::half_t> {
    enum { USER_SPLIT_K_WITH_OUTPUT_PRECISION = 1 };
    static const bool IS_GELU_ERF = false;

    template <int M, int N, int K, int GROUPS = 1>
    using Cta_tile = xmma::Cta_tile<Hopper,
                                    M,
                                    N,
                                    K,
                                    XMMA_DIV_UP( M, 64 ),
                                    XMMA_DIV_UP( N, 64 ),
                                    XMMA_DIV_UP( K, 64 ),
                                    GROUPS>;

    template <typename Cta_tile> using Xmma_tile = Hopper_hmma_tile<Cta_tile>;
};

struct Hopper_hmma_fp32_traits
    : public Traits<Hopper, lwtlass::half_t, lwtlass::half_t, lwtlass::half_t, float, float> {

    static const bool IS_GELU_ERF = false;

    template <int M, int N, int K, int GROUPS = 1>
    using Cta_tile = xmma::Cta_tile<Hopper,
                                    M,
                                    N,
                                    K,
                                    XMMA_DIV_UP( M, 64 ),
                                    XMMA_DIV_UP( N, 64 ),
                                    XMMA_DIV_UP( K, 64 ),
                                    GROUPS>;

    template <typename Cta_tile> using Xmma_tile = Hopper_hmma_tile<Cta_tile>;
};

template <typename Input_type, typename Output_type>
struct Hopper_hmma_bf16_traits
    : public Traits<Hopper, Input_type, Input_type, Output_type, float, float> {

    static const bool IS_GELU_ERF = false;

    // The Cta tile.
    template <int M, int N, int K, int GROUPS = 1>
    using Cta_tile = xmma::Cta_tile<Hopper,
                                    M,
                                    N,
                                    K,
                                    XMMA_DIV_UP( M, 64 ),
                                    XMMA_DIV_UP( N, 64 ),
                                    XMMA_DIV_UP( K, 64 ),
                                    GROUPS>;

    // The XMMA tile.
    template <typename Cta_tile> using Xmma_tile = Hopper_hmma_tile<Cta_tile>;
};

template <typename Input_type, typename Output_type>
struct Hopper_hmma_tf32_traits
    : public Traits<Hopper, Input_type, Input_type, Output_type, float, float> {

    static const bool IS_GELU_ERF = false;
    
    // The Cta tile.
    template <int M, int N, int K, int GROUPS = 1>
    using Cta_tile = xmma::Cta_tile<Hopper,
                                    M,
                                    N,
                                    K,
                                    XMMA_DIV_UP( M, 64 ),
                                    XMMA_DIV_UP( N, 64 ),
                                    XMMA_DIV_UP( K, 32 ),
                                    GROUPS>;

    // The XMMA tile.
    template <typename Cta_tile> using Xmma_tile = Hopper_hmma_tile<Cta_tile, 8>;
};

////////////////////////////////////////////////////////////////////////////////////////////////////
template <
          int ROW_ = 1,
          int COL_ = 1>
struct Hopper_cga_tile {

    // The size of the CGA
    enum { CLUSTER_ROW = ROW_, CLUSTER_COL = COL_ };

};

template <
          typename Cta_tile_,
          typename Cga_tile_,
          int USE_TMA_MCAST_ = 0>
struct Hopper_tile_traits {

    using Gpu_arch = typename Cta_tile_::Gpu_arch;
    
    using Cta_tile = Cta_tile_;

    using Cga_tile = Cga_tile_;

    // Enable TMA Multi-cast ?
    enum { USE_TMA_MULTICAST = USE_TMA_MCAST_ };

    static constexpr int M = Cta_tile::M;
    static constexpr int N = Cta_tile::N;
    static constexpr int K = Cta_tile::K;
    static constexpr int THREADS_PER_CTA = Cta_tile::THREADS_PER_CTA;
    static constexpr int CLUSTER_ROW = Cga_tile::CLUSTER_ROW;
    static constexpr int CLUSTER_COL = Cga_tile::CLUSTER_COL;
};

////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename Gpu_arch_,
          int M_,             // CTA tile M
          int N_,             // CTA tile N
          int K_,             // CTA tile K
          int WARP_GROUP_M_,  // Number of warp group along M dim
          int WARP_GROUP_N_,  // Number of warp group along N dim
          int WARP_GROUP_K_,  // Number of warp group along K dim
          int GROUPS_ = 1,
          int USE_PREDICATES_ = 1>
struct Hopper_cta_tile {

    using Gpu_arch = Gpu_arch_;

    // Make sure M and N are multiples of the group size.
    static_assert( ( M_ % GROUPS_ == 0 ) && ( N_ % GROUPS_ == 0 ),
                   "M/N must be multiple of GROUPS" );

    // The size of the CTA tile.
    enum { M = M_, N = N_, K = K_ };
    // The number of warp groups.
    enum {
        WARP_GROUP_M = WARP_GROUP_M_,
        WARP_GROUP_N = WARP_GROUP_N_,
        WARP_GROUP_K = WARP_GROUP_K_
    };
    // need to think more about this. a lot of code has default value CtaTile::warp_k
    enum { WARPS_K = WARP_GROUP_K_ };

    // The number of groups.
    enum { GROUPS = GROUPS_ };
    // The number of warps per CTA.
    enum {
        WARPS_PER_CTA = WARP_GROUP_M * WARP_GROUP_N * WARP_GROUP_K * Gpu_arch::WARPS_PER_WARP_GROUP
    };
    // The number of warps per warpgroup.
    enum { WARPS_PER_WARP_GROUP = Gpu_arch::WARPS_PER_WARP_GROUP };
    // The number of threads per warp.
    enum { THREADS_PER_WARP = Gpu_arch::THREADS_PER_WARP };
    // the number of threads per warpgroup.
    enum { THREADS_PER_WARP_GROUP = THREADS_PER_WARP * WARPS_PER_WARP_GROUP };
    // The number of threads per CTA.
    enum { THREADS_PER_CTA = WARPS_PER_CTA * THREADS_PER_WARP };
    // Half K dimension
    enum { HALF_K = K_ / 2 };
    // Do we use predicates for loads
    enum { USE_PREDICATES = USE_PREDICATES_ };
};

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// H M M A . F 1 6 / F 3 2
//
////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Cta_tile, int GMMA_M, int GMMA_N, int GMMA_K> struct Hopper_hgmma_tile {

    // The number of elements computed with a single warp group mma.
    enum { M_PER_XMMA = GMMA_M, N_PER_XMMA = GMMA_N, K_PER_XMMA = GMMA_K };

    // The number of warp groups
    enum {
        NUM_WARP_GROUPS = Cta_tile::WARP_GROUP_M * Cta_tile::WARP_GROUP_N * Cta_tile::WARP_GROUP_K
    };

    // The number of elements computed with a single CTA-MMA.
    enum {
        M_PER_XMMA_PER_CTA = M_PER_XMMA * Cta_tile::WARP_GROUP_M,
        N_PER_XMMA_PER_CTA = N_PER_XMMA * Cta_tile::WARP_GROUP_N,
        K_PER_XMMA_PER_CTA = K_PER_XMMA * Cta_tile::WARP_GROUP_K
    };

    // The number of MMAs needed to compute the GEMM.
    enum {
        XMMAS_M = ( Cta_tile::M + M_PER_XMMA_PER_CTA - 1 ) / M_PER_XMMA_PER_CTA,
        XMMAS_N = ( Cta_tile::N + N_PER_XMMA_PER_CTA - 1 ) / N_PER_XMMA_PER_CTA,
        XMMAS_K = ( Cta_tile::K + K_PER_XMMA_PER_CTA - 1 ) / K_PER_XMMA_PER_CTA,
        XMMAS_GROUPS = ( Cta_tile::GROUPS < 4 ? Cta_tile::GROUPS : 4 )
    };

    // The number of elements computed per warp group.
    enum {
        M_PER_WARP_GROUP = XMMAS_M * M_PER_XMMA,
        N_PER_WARP_GROUP = XMMAS_N * N_PER_XMMA,
        K_PER_WARP_GROUP = XMMAS_K * K_PER_XMMA,
    };

    // the size of GMMA group, which is GMMA_M x GMMA_N x Kblock
    enum {
        M_PER_GMMA_GROUP = GMMA_M,
        N_PER_GMMA_GROUP = GMMA_N,
        K_PER_GMMA_GROUP = Cta_tile::K,
    };

    // The distribution of threads in the output tile.
    // TODO
    enum {
        THREADS_PER_XMMA_M = 8,
        THREADS_PER_XMMA_N = 4,
    };
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int GMMA_M_,      // GMMA instruction shape in M dim
          int GMMA_N_,      // GMMA instruction shape in N dim
          int GMMA_K_,      // GMMA instruction shape in K dim
          bool GMMA_A_RF_,  // GMMA A operand coming from RF?
          bool GMMA_B_RF_   // GMMA B operand coming from RF?
          >
struct Hopper_hgmma_fp16_traits : public Traits<Hopper,
                                                lwtlass::half_t,
                                                lwtlass::half_t,
                                                lwtlass::half_t,
                                                lwtlass::half_t,
                                                lwtlass::half_t> {

    enum { USE_SPLIT_K_WITH_OUTPUT_PRECISION = 1 };

    // The GMMA shape
    enum { GMMA_M = GMMA_M_, GMMA_N = GMMA_N_, GMMA_K = GMMA_K_ };

    // is A operand in RF for GMMA?
    static constexpr bool GMMA_A_RF = GMMA_A_RF_;

    // is B operand in RF for GMMA?
    static constexpr bool GMMA_B_RF = GMMA_B_RF_;

    // GMMA shape has certain requirement
    static_assert( GMMA_K == 16, "GMMA K must be 16; this might change" );
    static_assert( GMMA_M == 64, "GMMA M must be 64; this might change" );
    static_assert( GMMA_N % 8 == 0, "GMMA N must be multiple of 8; this might change" );
    static_assert( GMMA_N <= 256, "GMMA N must be no larger than 256; this might change" );

    // GMMA does not allow both operands coming from RF.
    static_assert( ( GMMA_A_RF && GMMA_B_RF ) != true,
                   "GMMA does not allow both operands coming from RF." );

    // The Cta tile.
    template <int M, int N, int K, int Warpgroup_M, int Warpgroup_N, int Warpgroup_K>
    using Cta_tile = Hopper_cta_tile<Hopper, M, N, K, Warpgroup_M, Warpgroup_N, Warpgroup_K, 1, 1>;

    // The CGA Tile
    using Cga_tile = Hopper_cga_tile<1, 1>;

    // Hopper Tile Traits
    template <typename Cta_tile_, typename Cga_tile_>
    using Tile_traits = Hopper_tile_traits<Cta_tile_, Cga_tile_>;

    // The XMMA tile.
    template <typename Cta_tile>
    using Xmma_tile = Hopper_hgmma_tile<Cta_tile, GMMA_M, GMMA_N, GMMA_K>;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int GMMA_M_,      // GMMA instruction shape in M dim
          int GMMA_N_,      // GMMA instruction shape in N dim
          int GMMA_K_,      // GMMA instruction shape in K dim
          bool GMMA_A_RF_,  // GMMA A operand coming from RF?
          bool GMMA_B_RF_   // GMMA B operand coming from RF?
          >
struct Hopper_hgmma_fp32_traits
    : public Traits<Hopper, lwtlass::half_t, lwtlass::half_t, lwtlass::half_t, float, float> {

    // The GMMA shape
    enum { GMMA_M = GMMA_M_, GMMA_N = GMMA_N_, GMMA_K = GMMA_K_ };

    // is A operand in RF for GMMA?
    static constexpr bool GMMA_A_RF = GMMA_A_RF_;

    // is B operand in RF for GMMA?
    static constexpr bool GMMA_B_RF = GMMA_B_RF_;

    // GMMA shape has certain requirement
    static_assert( GMMA_K == 16, "GMMA K must be 16; this might change" );
    static_assert( GMMA_M == 64, "GMMA M must be 64; this might change" );
    static_assert( GMMA_N % 8 == 0, "GMMA N must be multiple of 8; this might change" );
    static_assert( GMMA_N <= 256, "GMMA N must be no larger than 256; this might change" );

    // GMMA does not allow both operands coming from RF.
    static_assert( ( GMMA_A_RF && GMMA_B_RF ) != true,
                   "GMMA does not allow both operands coming from RF." );

    // The Cta tile.
    template <int M, int N, int K, int Warpgroup_M, int Warpgroup_N, int Warpgroup_K>
    using Cta_tile = Hopper_cta_tile<Hopper, M, N, K, Warpgroup_M, Warpgroup_N, Warpgroup_K, 1, 1>;

    // The CGA Tile
    using Cga_tile = Hopper_cga_tile<1, 1>;

    // Hopper Tile Traits
    template <typename Cta_tile_, typename Cga_tile_>
    using Tile_traits = Hopper_tile_traits<Cta_tile_, Cga_tile_>;

    // The XMMA tile.
    template <typename Cta_tile>
    using Xmma_tile = Hopper_hgmma_tile<Cta_tile, GMMA_M, GMMA_N, GMMA_K>;
};

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// D M M A . F P 6 4
//
////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Cta_tile, int K_PER_XMMA_ = 16> struct Hopper_dmma_tile {

    // The number of elements computed with a single warp-MMA.
    enum { M_PER_XMMA = 16, N_PER_XMMA = 16, K_PER_XMMA = K_PER_XMMA_ };

    // The number of elements computed with a single CTA-MMA.
    enum {
        M_PER_XMMA_PER_CTA = M_PER_XMMA * Cta_tile::WARPS_M,
        N_PER_XMMA_PER_CTA = N_PER_XMMA * Cta_tile::WARPS_N,
        K_PER_XMMA_PER_CTA = K_PER_XMMA * Cta_tile::WARPS_K
    };

    // The number of MMAs needed to compute the GEMM.
    enum {
        XMMAS_M = ( Cta_tile::M + M_PER_XMMA_PER_CTA - 1 ) / M_PER_XMMA_PER_CTA,
        XMMAS_N = ( Cta_tile::N + N_PER_XMMA_PER_CTA - 1 ) / N_PER_XMMA_PER_CTA,
        XMMAS_K = ( Cta_tile::K + K_PER_XMMA_PER_CTA - 1 ) / K_PER_XMMA_PER_CTA,
        XMMAS_GROUPS = ( Cta_tile::GROUPS < 4 ? Cta_tile::GROUPS : 4 )
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

struct Hopper_dmma_fp64_traits : public Traits<Hopper, double, double, double, double, double> {

    static const bool IS_GELU_ERF = false;

    // The Cta tile.
    template <int M, int N, int K, int GROUPS = 1>
    using Cta_tile = xmma::Cta_tile<Hopper,
                                    M,
                                    N,
                                    K,
                                    XMMA_DIV_UP( M, 32 ),
                                    M == 64 ? XMMA_DIV_UP( N, 32 ) : 2,
                                    1,
                                    GROUPS>;

    // The XMMA tile.
    template <typename Cta_tile> using Xmma_tile = Hopper_dmma_tile<Cta_tile>;
};

}  // namespace xmma
