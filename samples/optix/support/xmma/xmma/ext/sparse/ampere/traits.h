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

#include <xmma/ampere/traits.h>

namespace xmma {

template<
    typename Gpu_arch_,
    typename A_type_,
    typename B_type_,
    typename C_type_,
    typename Aclwmulator_type_,
    typename Epilogue_type_,
    typename E_type_,
    int Alignment_E = 8
>
struct sparse_traits :
    public Traits<Gpu_arch_, A_type_, B_type_, C_type_, Aclwmulator_type_, Epilogue_type_> {
        // Metadata type is uint16_t;
        using E_type = E_type_;
        // The number of bits per element of E.
        enum { BITS_PER_ELEMENT_E = Alignment_E * sizeof(E_type) };

        enum { ROW_STRIDE_GROUP = 64 };

        enum { ELEMENTS_PER_UINT16 = 8 };
        // An offset in bytes for E.
        static inline __host__ __device__ int64_t offset_in_bytes_e(int64_t offset) {
            return offset * static_cast<int64_t>(sizeof(E_type));
        }

        template<
            typename Fragment_aclwmulators,
            typename Fragment_a,
            typename Fragment_b,
            typename Fragment_e,
            int M,
            int N,
            int N_PER_GROUP
        >
        static inline __device__ void spgemm(Fragment_aclwmulators (&acc)[M][N_PER_GROUP],
            const Fragment_a (&a)[M],
            const Fragment_b (&b)[N],
            const Fragment_e (&e)[1],
            int nBegin=0) {
                xmma::helpers::spgemm(acc, a, b, e);
        }

        template<
            typename Fragment_aclwmulators,
            typename Fragment_a,
            typename Fragment_b,
            typename Fragment_e,
            int M,
            int N,
            int N_PER_GROUP
        >
        static inline __device__ void spgemm_pipeline(Fragment_aclwmulators (&acc)[M][N_PER_GROUP],
            const Fragment_a (&a)[M],
            const Fragment_b (&b)[N],
            const Fragment_e (&e)[1],
            int pipe_stage
            ) {
                xmma::helpers::spgemm_pipeline(acc, a, b, e, pipe_stage);
        }
/*
        template<
            typename Fragment_aclwmulators,
            typename Fragment_a,
            typename Fragment_b,
            typename Fragment_e,
            int M,
            int N,
            int N_PER_GROUP
        >
        static inline __device__ void sparse_gemm(Fragment_aclwmulators (&acc)[M][N_PER_GROUP],
            const Fragment_a (&a)[M],
            const Fragment_b (&b)[N],
            const Fragment_e (&e)[1],
            int nBegin=0) {
            xmma::helpers::sparse_gemm(acc, a, b, e);
        }
*/
};

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// S P A R S E  H M M A . F 3 2
//
////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Cta_tile, int K_PER_XMMA = 32 >
struct Ampere_sphmma_tile : public Turing_mma_tile<Cta_tile, K_PER_XMMA> {
    enum {
        XMMAS_N_HALF = Ampere_sphmma_tile::XMMAS_N / 2,
        XMMAS_N_QUAD = Ampere_sphmma_tile::XMMAS_N / 4
    };
    enum {
        XMMAS_N_DIV_2 = 2,
        XMMAS_N_DIV_4 = 4
    };
    enum {
        XMMAS_N_PIPE_STAGE = (XMMAS_N_QUAD != 0) ? XMMAS_N_QUAD : XMMAS_N_HALF
    };
    enum {
        XMMAS_N_DIV = (XMMAS_N_QUAD != 0) ? XMMAS_N_DIV_4 : XMMAS_N_DIV_2
    };
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Ampere_sphmma_fp32_traits :
    public sparse_traits<Ampere, uint16_t, uint16_t, uint16_t, float, float, uint16_t> {

    // The Cta tile.
    template< int M, int N, int K, int GROUPS = 1>
    using Cta_tile = xmma::Cta_tile<
        Ampere, M, N, K, XMMA_DIV_UP(M, 64),
        XMMA_DIV_UP(N, (N == 160 ? 80 : M == 64 ? 32 : 64)),
        XMMA_DIV_UP(K, 64), GROUPS>;

    // The XMMA tile.
    template< typename Cta_tile >
    using Xmma_tile = Ampere_sphmma_tile<Cta_tile>;

    enum { ACLWMULATOR_32BIT = 1 };
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Ampere_sphmma_bf16_fp32_bf16_traits :
    public sparse_traits<Ampere, lwtlass::float_bf16_t,
                                 lwtlass::float_bf16_t,
                                 lwtlass::float_bf16_t,
                                 float,
                                 float,
                                 uint16_t > {

    // The Cta tile.
    template< int M, int N, int K, int GROUPS = 1>
    using Cta_tile = xmma::Cta_tile<
        Ampere, M, N, K, XMMA_DIV_UP(M, 64), XMMA_DIV_UP(N, (M == 64 ? 32 : 64)),
        XMMA_DIV_UP(K, 64), GROUPS>;

    // The XMMA tile.
    template< typename Cta_tile >
    using Xmma_tile = Ampere_sphmma_tile<Cta_tile>;

    enum { ACLWMULATOR_32BIT = 1 };
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Ampere_sphmma_fp16_traits :
    public sparse_traits<Ampere, lwtlass::half_t, lwtlass::half_t, lwtlass::half_t, lwtlass::half_t, lwtlass::half_t, uint16_t> {

    // The Cta tile.
    template< int M, int N, int K, int GROUPS = 1>
    using Cta_tile = xmma::Cta_tile<
        Ampere, M, N, K, XMMA_DIV_UP(M, 64),
        XMMA_DIV_UP(N, (N == 160 ? 80 : M == 64 ? 32 : 64)),
        XMMA_DIV_UP(K, 64), GROUPS>;

    // The XMMA tile.
    template< typename Cta_tile >
    using Xmma_tile = Ampere_sphmma_tile<Cta_tile>;

    enum { ACLWMULATOR_32BIT = 0 };
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Input_type, typename Output_type >
struct Ampere_sphmma_tf32_traits
    : public sparse_traits<Ampere, Input_type, Input_type, Output_type, float, float, uint16_t, 16> {

    // The Cta tile.
    template< int M, int N, int K, int GROUPS = 1 >
    using Cta_tile = xmma::Cta_tile<
        Ampere, M, N, K, XMMA_DIV_UP(M, 64),
        XMMA_DIV_UP(N, (N == 160 ? 80 : M == 64 ? 32 : 64)),
        XMMA_DIV_UP(K, 32), GROUPS>;

    // The XMMA tile.
    template< typename Cta_tile >
    using Xmma_tile = Ampere_sphmma_tile<Cta_tile, 16>;

    enum { ACLWMULATOR_32BIT = 1 };
};

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// S P A R S E  I M M A . 8
//
////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Cta_tile >
struct Ampere_spimma_int8_tile : public Turing_mma_tile<Cta_tile, 64> {
    enum {
        XMMAS_N_HALF = Ampere_spimma_int8_tile::XMMAS_N / 2,
        XMMAS_N_QUAD = Ampere_spimma_int8_tile::XMMAS_N / 4
    };
    enum {
        XMMAS_N_DIV_2 = 2,
        XMMAS_N_DIV_4 = 4
    };
    enum {
        XMMAS_N_PIPE_STAGE = (XMMAS_N_QUAD != 0) ? XMMAS_N_QUAD : XMMAS_N_HALF
    };
    enum {
        XMMAS_N_DIV = (XMMAS_N_QUAD != 0) ? XMMAS_N_DIV_4 : XMMAS_N_DIV_2
    };
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Ampere_spimma_int8_traits :
    public sparse_traits<Ampere, int8_t, int8_t, int8_t, int32_t, float, uint16_t> {

    // The Cta tile.
    template< int M, int N, int K, int GROUPS = 1>
    using Cta_tile = xmma::Cta_tile<
        Ampere, M, N, K, XMMA_DIV_UP(M, 64),
        XMMA_DIV_UP(N, (N == 160 ? 80 : M == 64 ? 32 : 64)),
        XMMA_DIV_UP(K, 128), GROUPS>;

    // The XMMA tile.
    template< typename Cta_tile >
    using Xmma_tile = Ampere_spimma_int8_tile<Cta_tile>;

    enum { ACLWMULATOR_32BIT = 1 };
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Ampere_spimma_gelu_int8_traits : Ampere_spimma_int8_traits {
};

struct Ampere_spimma_int8_rt_fuse_traits : Ampere_spimma_int8_traits {
};
////////////////////////////////////////////////////////////////////////////////////////////////////
//
// S P A R S E  I M M A . 8   I N T E R L E A V E D
//
////////////////////////////////////////////////////////////////////////////////////////////////////

struct Ampere_spimma_interleaved_int8_traits :
    public sparse_traits<Ampere, int8_t, int8_t, int8_t, int32_t, float, uint16_t> {

    // The Cta tile.
    template< int M, int N, int K, int GROUPS = 1>
    using Cta_tile = xmma::Cta_tile<
        Ampere, M, N, K, XMMA_DIV_UP(M, 64),
        XMMA_DIV_UP(N, (N == 160 ? 80 : M == 64 ? 32 : 64)),
        XMMA_DIV_UP(K, 128), GROUPS>;

    // The XMMA tile.
    template< typename Cta_tile >
    using Xmma_tile = Ampere_spimma_int8_tile<Cta_tile>;

    enum { ACLWMULATOR_32BIT = 1 };
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Ampere_spimma_interleaved_gelu_int8_traits : Ampere_spimma_interleaved_int8_traits {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace xmma

