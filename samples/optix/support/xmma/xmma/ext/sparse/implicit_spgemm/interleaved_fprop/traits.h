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

#include <xmma/volta/traits.h>
#include <xmma/turing/traits.h>
#include <xmma/ampere/traits.h>

#include <xmma/ext/sparse/helpers/gemm.h>
#include <xmma/helpers/epilogue_with_split_k.h>
#include <xmma/params.h>
#include <xmma/tile_distribution.h>
#include <xmma/ext/sparse/implicit_spgemm/interleaved_fprop/params.h>
#include <xmma/ext/sparse/implicit_spgemm/interleaved_fprop/gmem_tile.h>
#include <xmma/ext/sparse/ampere/smem_tile_sparse.h>
#include <xmma/ext/sparse/helpers/epilogue_sparse.h>
#include <xmma/ext/sparse/implicit_spgemm/kernel.h>

///////////////////////////////////////////////////////////////////////////////////////////////////

namespace xmma {
namespace ext {
namespace implicit_gemm {
namespace interleaved_fprop {

///////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Traits_, typename Cta_tile_, typename Input_related_, int VECT_SIZE_, int STAGES_ = 1, int ELTWISE = xmma::RELU >
struct Sparse_kernel_traits : public Traits_ {

    // Colwolution type.
    static const xmma::Operation_type OPERATION_TYPE = xmma::Operation_type::FPROP;
    // Colwolution algorithm.
    static const xmma::Colwolution_algorithm COLW_ALGO =
        xmma::Colwolution_algorithm::PRECOMPUTED;
    // Colwolution layout.
    static const xmma::Colwolution_layout COLW_LAYOUT =
        xmma::Colwolution_layout::NCHW_VECT_C_32;

    // The instruction traits.
    using Traits = Traits_;
    // The Cta tile.
    using Cta_tile = Cta_tile_;
    // Template parameters which are related to input parameters like filter size.
    using Input_related = Input_related_;

    enum { FLT_T = Input_related::FLT_T };
    enum { FLT_R = Input_related::FLT_R };
    enum { FLT_S = Input_related::FLT_S };
    enum { USE_PREDICATES = Input_related::USE_PREDICATES };
    enum { STATIC_FILTER_SIZE = Input_related::STATIC_FILTER_SIZE };

    // The number of stages.
    enum { STAGES = STAGES_ };

    enum { VECT_SIZE = VECT_SIZE_ };

    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;
    // The kernel parameters.
    using Params = xmma::ext::implicit_gemm::interleaved_fprop::Params<Traits, Cta_tile, STAGES>;
    // Tile distribution
    using Tile_distribution = xmma::Tile_distribution;

    // The global memory loader for A.
    using Gmem_tile_a = xmma::ext::implicit_gemm::interleaved_fprop::Gmem_tile_a<Traits, Cta_tile, USE_PREDICATES>;
    // The number of buffers in shared memory. It must be at least 2 for LDGSTS.
    enum { BUFFERS_PER_SMEM_TILE_A = Gmem_tile_a::USE_LDGSTS ? xmma::Max<2, STAGES>::VALUE : STAGES };
    // The shared memory loader for A.
    using Smem_tile_a = xmma::Smem_tile_a<Traits,
                                              Cta_tile,
                                              xmma::Row,
                                              Gmem_tile_a::BYTES_PER_LDG,
                                              BUFFERS_PER_SMEM_TILE_A,
                                              USE_PREDICATES>;

    // The global memory loader for B.
    using Gmem_tile_b = xmma::ext::implicit_gemm::interleaved_fprop::Gmem_tile_b<Traits, Cta_tile, Input_related, VECT_SIZE, USE_PREDICATES>;
    // The number of buffers in shared memory. It must be at least 2 for LDGSTS.
    enum { BUFFERS_PER_SMEM_TILE_B = Gmem_tile_b::USE_LDGSTS ? xmma::Max<2, STAGES>::VALUE : STAGES };
    // The shared memory loader for B.
    using Smem_tile_b = xmma::Smem_tile_b<Traits,
                                              Cta_tile,
                                              xmma::Col,
                                              Gmem_tile_b::BYTES_PER_LDG,
                                              BUFFERS_PER_SMEM_TILE_B,
                                              USE_PREDICATES>;

    // The global memory loader for METADATA.
    using Gmem_tile_e = xmma::ext::implicit_gemm::interleaved_fprop::Gmem_tile_e<Traits, Cta_tile>;
    // The number of buffers in shared memory. It must be at least 2 for LDGSTS.
    enum { BUFFERS_PER_SMEM_TILE_E = STAGES };
    // The shared memory loader for E.
    using Smem_tile_e = xmma::Smem_tile_lds_e<Traits, Cta_tile, xmma::Row, Gmem_tile_e::BYTES_PER_LDG, STAGES>;

    // The global memory epilogue.
    using Gmem_tile_epilogue = xmma::ext::implicit_gemm::interleaved_fprop::Gmem_tile_epilogue<Traits, Cta_tile, VECT_SIZE, USE_PREDICATES>;
    // The shared memory epilogue tile.
    using Swizzle_epilogue = xmma::Swizzle_epilogue_empty<Traits, Cta_tile, xmma::Row>;
    // The callbacks.
    using Callbacks_epilogue = xmma::ext::helpers::Callbacks_epilogue<
                                                              Traits,
                                                              Cta_tile,
                                                              ELTWISE>;
    // The epilogue.
    using Epilogue = xmma::ext::helpers::Epilogue<Traits,
                                                              Cta_tile,
                                                              xmma::Row,
                                                              Gmem_tile_epilogue,
                                                              ELTWISE,
                                                              Callbacks_epilogue,
                                                              Swizzle_epilogue>;
    // The residue prefetch.
    using Gmem_tile_epilogue_prefetch = xmma::ext::implicit_gemm::interleaved_fprop::Gmem_tile_imma_epilogue_prefetch<Traits, Cta_tile>;
    
#if !defined(__LWDACC_RTC__)
    typedef void (*Kernel_type)(Params params);

    // Return device kernel function pointer.
    static XMMA_HOST Kernel_type kernel_ptr(const Params params = Params()) {
        return &::xmma::ext::implicit_gemm::kernel<Sparse_kernel_traits>;
    }

    // Return split k kernel function pointer.
    static XMMA_HOST Kernel_type split_k_kernel_ptr() {
        return &::xmma::ext::implicit_gemm::split_k_kernel<Sparse_kernel_traits>;
    }
#endif

    // The number of threads in the CTA.
    static int threads_per_cta() {
        return Cta_tile::THREADS_PER_CTA;
    }

    // The amount of shared memory.
    static int dynamic_smem_size_per_cta() {
        // The amount of shared memory needed for the main loop.
        const int LOOP_SIZE_IN_BYTES = Smem_tile_a::BYTES_PER_TILE + Smem_tile_b::BYTES_PER_TILE
            + Smem_tile_e::BYTES_PER_TILE
            + max(Cta_tile::N, Cta_tile::THREADS_PER_CTA) * 4 * 2;
            // + Cta_tile::THREADS_PER_CTA * 4 * 4;

        // The amount of shared memory needed by the epilogue.
        const int EPILOGUE_SIZE_IN_BYTES = 0;//Smem_tile_epilogue::BYTES_PER_TILE;

        // The amount of shared memory to launch the kernel.
        return max(LOOP_SIZE_IN_BYTES, EPILOGUE_SIZE_IN_BYTES);
    }

    // Initialize the filter position.
    template <typename Params>
    static inline __device__ int initialize_filter_position( const Params& params ) {
        int32_t filter_trs_per_cta =
            ( STATIC_FILTER_SIZE ? FLT_T * FLT_R * FLT_S : params.filter_trs_per_cta );
        if( filter_trs_per_cta == 1 ) {
            return 0;
        } else {
            return (threadIdx.x / ( 2 * Gmem_tile_b::ROWS_PER_LDG )) % filter_trs_per_cta;
        }
    }

    // Load the delta values and move the filter position.
    template <typename Params>
    static inline __device__ int
    load_deltas_and_move_filter_position( int64_t& a_delta, int64_t& b_delta, const Params& params,
                                          int trsi ) {
        int32_t filter_trs_per_cta =
            ( STATIC_FILTER_SIZE ? FLT_T * FLT_R * FLT_S : params.filter_trs_per_cta );
        // Early exit for 1x1x1 filters.
        if( filter_trs_per_cta == 1 ) {
            b_delta = params.a_delta[0];
            return 0;
        }

        // Are we moving to a new channel?
        int reset_trsi = trsi >= filter_trs_per_cta - 4;

        // Load the updates.
        b_delta = params.a_delta[trsi];

        // Update the filter position.
        // return (( reset_trsi ? trsi - filter_trs_per_cta : trsi ) + 4) ;
        return (( reset_trsi ? trsi - filter_trs_per_cta : trsi ) + 4) % filter_trs_per_cta;
    }

    // Gemm.
    template<
    int M_FIRST,
    int M_LAST,
    int N_FIRST,
    int N_LAST,
        typename Fragment_aclwmulators,
        typename Fragment_a,
        typename Fragment_b,
        typename Fragment_e,
        int M,
        int N,
        int N_PER_GROUP
    >
    static inline __device__ void gemm(Fragment_aclwmulators (&acc)[M][N_PER_GROUP],
              const Fragment_a (&a)[M],
              const Fragment_b (&b)[N],
              const Fragment_e (&e)[1],
              int ki=0) {
        if (Traits::BITS_PER_ELEMENT_C == 8) {
            xmma::helpers::sparse_igemm<M_FIRST,M_LAST,N_FIRST,N_LAST>(acc, a, b, e);
        } else {
            xmma::helpers::sparse_gemm<M_FIRST,M_LAST,N_FIRST,N_LAST>(acc, a, b, e);
        }
    }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Traits, typename Cta_tile, typename Input_related, int VECT_SIZE, int STAGES >
struct Sparse_kernel_traits_adapter {
};

///////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Cta_tile, typename Input_related, int VECT_SIZE, int STAGES >
struct Sparse_kernel_traits_adapter<xmma::Ampere_sphmma_fp32_traits,
                             Cta_tile,
                             Input_related,
                             VECT_SIZE,
                             STAGES> {
    using Type = Sparse_kernel_traits<xmma::Ampere_sphmma_fp32_traits,
                                      Cta_tile,
                                      Input_related,
                                      VECT_SIZE,
                                      STAGES,
                                      xmma::RELU>;
};

///////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Cta_tile, typename Input_related, int VECT_SIZE, int STAGES >
struct Sparse_kernel_traits_adapter<xmma::Ampere_sphmma_fp16_traits,
                             Cta_tile,
                             Input_related,
                             VECT_SIZE,
                             STAGES> {
    using Type = Sparse_kernel_traits<xmma::Ampere_sphmma_fp16_traits,
                                      Cta_tile,
                                      Input_related,
                                      VECT_SIZE,
                                      STAGES,
                                      xmma::RELU>;
};

///////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Cta_tile, typename Input_related, int VECT_SIZE, int STAGES >
struct Sparse_kernel_traits_adapter<xmma::Ampere_spimma_interleaved_int8_traits,
                             Cta_tile,
                             Input_related,
                             VECT_SIZE,
                             STAGES> {
    using Type = Sparse_kernel_traits<xmma::Ampere_spimma_interleaved_int8_traits,
                                      Cta_tile,
                                      Input_related,
                                      VECT_SIZE,
                                      STAGES,
                                      xmma::RELU>;
};

///////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Cta_tile, typename Input_related, int VECT_SIZE, int STAGES >
struct Sparse_kernel_traits_adapter<xmma::Ampere_spimma_interleaved_gelu_int8_traits,
                             Cta_tile,
                             Input_related,
                             VECT_SIZE,
                             STAGES> {
    using Type = Sparse_kernel_traits<xmma::Ampere_spimma_interleaved_int8_traits,
                                           Cta_tile,
                                           Input_related,
                                           VECT_SIZE,
                                           STAGES,
                                           xmma::GELU>;
};

///////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Traits, typename Cta_tile, typename Input_related, int VECT_SIZE, int STAGES >
using Kernel_traits = typename Sparse_kernel_traits_adapter<Traits, Cta_tile, Input_related, VECT_SIZE, STAGES>::Type;

///////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace interleaved_fprop
} // namespace implicit_gemm
} // namespace ext
} // namespace xmma

