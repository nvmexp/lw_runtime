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

#include <xmma/helpers/epilogue_with_split_k.h>
#include <xmma/volta/traits.h>
#include <xmma/turing/traits.h>
#include <xmma/ampere/traits.h>
#include <xmma/params.h>
#include <xmma/tile_distribution.h>
#include <xmma/compute_tile.h>
#include <xmma/smem_tile.h>

#include <xmma/gemm/kernel.h>
#include <xmma/implicit_gemm/interleaved_dgrad/gmem_tile.h>
#include <xmma/implicit_gemm/interleaved_dgrad/params.h>
#include <xmma/implicit_gemm/interleaved_dgrad/callbacks.h>

namespace xmma {
namespace implicit_gemm {
namespace interleaved_dgrad {

///////////////////////////////////////////////////////////////////////////////////////////////////

template <
    // The instruction traits.
    typename Traits_,
    // The CTA tile descriptor.
    typename Cta_tile_,
    // The global memory tile for Epilogue (transposed or not).
    typename Gmem_tile_epilogue_,
    // Input related params
    typename Input_related_,
    int32_t BYTES_PER_PACKET,
    // The number of stages in the prefetch pipeline.
    int32_t STAGES_ = 1>
struct Interleaved_kernel_traits : public Traits_ {

    // Colwolution type.
    static const xmma::Operation_type OPERATION_TYPE = xmma::Operation_type::DGRAD;
    // Colwolution algorithm.
    static const xmma::Colwolution_algorithm COLW_ALGO = xmma::Colwolution_algorithm::PRECOMPUTED;
    // Colwolution layout.
    static const xmma::Colwolution_layout COLW_LAYOUT = xmma::Colwolution_layout::NCHW_VECT_C_32;
    // Whether use warp specialized
    static const bool USE_WARP_SPECIALIZATION = false;

    // The instruction traits.
    using Traits = Traits_;
    // The Cta tile.
    using Cta_tile = Cta_tile_;
    // Template parameters which are related to input parameters like filter size.
    using Input_related = Input_related_;

    enum { FLT_T = Input_related::FLT_T };
    enum { FLT_R = Input_related::FLT_R };
    enum { FLT_S = Input_related::FLT_S };
    enum { STATIC_FILTER_SIZE = Input_related::STATIC_FILTER_SIZE };

    // The number of stages.
    enum { STAGES = STAGES_ };

    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;

    // The kernel parameters.
    using Params = xmma::implicit_gemm::interleaved_dgrad::Params<Traits, Cta_tile, STAGES>;
    // Tile distribution
    using Tile_distribution = xmma::Tile_distribution;
    // The global memory loader for A.
    using Gmem_tile_a = xmma::implicit_gemm::interleaved_dgrad::
        Gmem_tile_a<Traits, Cta_tile, Input_related, BYTES_PER_PACKET>;
    // The number of buffers in shared memory. It must be at least 2 for LDGSTS.
    enum {
        BUFFERS_PER_SMEM_TILE_A = Gmem_tile_a::USE_LDGSTS ? xmma::Max<2, STAGES>::VALUE : STAGES
    };
    // The shared memory loader for A.
    using Smem_tile_a = xmma::Smem_tile_a<Traits,
                                          Cta_tile,
                                          xmma::Col_interleaved,
                                          Gmem_tile_a::BYTES_PER_LDG,
                                          BUFFERS_PER_SMEM_TILE_A>;

    // The global memory loader for B.
    using Gmem_tile_b = xmma::implicit_gemm::interleaved_dgrad::
        Gmem_tile_b<Traits, Cta_tile, Input_related, BYTES_PER_PACKET>;
    // The number of buffers in shared memory. It must be at least 2 for LDGSTS.
    enum {
        BUFFERS_PER_SMEM_TILE_B = Gmem_tile_b::USE_LDGSTS ? xmma::Max<2, STAGES>::VALUE : STAGES
    };
    // The shared memory loader for B.
    using Smem_tile_b = xmma::Smem_tile_b<Traits,
                                          Cta_tile,
                                          xmma::Row_interleaved,
                                          Gmem_tile_b::BYTES_PER_LDG,
                                          BUFFERS_PER_SMEM_TILE_B>;

    // The compute tile.
    using Compute_tile = xmma::Compute_tile<Traits, Cta_tile, Smem_tile_a, Smem_tile_b>;

    // The global memory epilogue.
    using Gmem_tile_epilogue = Gmem_tile_epilogue_;

    // The shared memory epilogue tile.
    using Swizzle_epilogue =
        xmma::Swizzle_epilogue_interleaved<Traits, Cta_tile, xmma::Col_interleaved>;

    // The callbacks.
    using Callbacks_epilogue =
        Callbacks_epilogue_fuse<Traits,
                                Cta_tile,
                                Gmem_tile_epilogue,
                                xmma::Fragment_epilogue_pre_swizzle<Traits, Cta_tile>,
                                xmma::Fragment_epilogue_interleaved_post_swizzle<Traits, Cta_tile>,
                                typename Gmem_tile_epilogue::Fragment_c>;

    // The epilogue.
    using Epilogue = xmma::helpers::Epilogue<Traits,
                                             Cta_tile,
                                             xmma::Col_interleaved,
                                             Gmem_tile_epilogue,
                                             Callbacks_epilogue,
                                             Swizzle_epilogue>;

    /* NOTE: Only FP64 GEMM supports gmem_wo_smem kernel */
    // The global memory loader for A without smem.
    using Gmem_wo_smem_tile_a =
        xmma::Gmem_wo_smem_tile_a<Traits, Cta_tile, xmma::Row, Gmem_tile_a::BYTES_PER_LDG>;
    // The global memory loader for A without smem.
    using Gmem_wo_smem_tile_b =
        xmma::Gmem_wo_smem_tile_b<Traits, Cta_tile, xmma::Col, Gmem_tile_b::BYTES_PER_LDG>;

    using Gmem_tile_wo_smem_epilogue = Gmem_tile_epilogue;
    // The callbacks.
    using Callbacks_wo_smem_epilogue = Callbacks_epilogue;
    // The epilogue.
    using Epilogue_wo_smem = Epilogue;
/* NOTE: end. */

#if !defined( __LWDACC_RTC__ )
    typedef void ( *Kernel_type )( Params params );

    // Return device kernel function pointer.
    static XMMA_HOST Kernel_type kernel_ptr( const Params params = Params() ) {
        return &::xmma::gemm::kernel<Interleaved_kernel_traits>;
    }

    // Return split k kernel function pointer.
    static XMMA_HOST Kernel_type split_k_kernel_ptr() {
        return &::xmma::gemm::split_k_kernel<Interleaved_kernel_traits>;
    }
#endif

    // The number of threads in the CTA.
    static int32_t threads_per_cta( const Params params = Params() ) {
        return Cta_tile::THREADS_PER_CTA;
    }

    // The amount of shared memory per CTA.
    static int32_t dynamic_smem_size_per_cta() {
        // The amount of shared memory needed for the main loop.
        const int32_t LOOP_SIZE_IN_BYTES = Smem_tile_a::BYTES_PER_TILE +
                                           Smem_tile_b::BYTES_PER_TILE +
                                           Gmem_tile_a::BYTES_PER_EXTRA_SMEM;

        // The amount of shared memory needed by the epilogue.
        const int32_t EPILOGUE_SIZE_IN_BYTES = Swizzle_epilogue::BYTES_PER_TILE;

        // The amount of shared memory to launch the kernel.
        return max( LOOP_SIZE_IN_BYTES, EPILOGUE_SIZE_IN_BYTES );
    }

    // The amount of epilogue shared memory per CTA.
    static int32_t epilogue_smem_size_per_cta() {
        return Swizzle_epilogue::BYTES_PER_TILE;
    }

    // Initialize the filter position.
    template <typename Params>
    static inline __device__ int32_t initialize_filter_position( const Params &params ) {
        int32_t filter_trs_per_cta =
            ( STATIC_FILTER_SIZE ? FLT_T * FLT_R * FLT_S : params.filter_trs_per_cta );
        if( filter_trs_per_cta == 1 ) {
            return 0;
        } else {
            return threadIdx.x / ( Gmem_tile_a::THREADS_PER_PACKET * Gmem_tile_a::ROWS_PER_LDG );
        }
    }

    // Load the delta values and move the filter position.
    template <typename Params>
    static inline __device__ int load_deltas_and_move_filter_position( int64_t &a_delta,
                                                                       int64_t &b_delta,
                                                                       const Params &params,
                                                                       int32_t trsi ) {
        int32_t filter_trs_per_cta =
            ( STATIC_FILTER_SIZE ? FLT_T * FLT_R * FLT_S : params.filter_trs_per_cta );
        // Early exit for 1x1x1 filters.
        if( filter_trs_per_cta == 1 ) {
            a_delta = params.a_delta[0];
            b_delta = params.b_delta[0];
            return 0;
        }

        // Are we moving to a new channel?
        int32_t reset_trsi = trsi >= filter_trs_per_cta - Gmem_tile_a::COLS;

        // Load the updates.
        a_delta = params.a_delta[trsi];
        b_delta = params.b_delta[0];

        // Update the filter position.
        return ( reset_trsi ? trsi - filter_trs_per_cta : trsi ) + Gmem_tile_a::COLS;
    }

    // Gemm.
    template <typename Fragment_aclwmulators,
              typename Fragment_a,
              typename Fragment_b,
              int32_t M,
              int32_t N,
              int32_t N_PER_GROUP>
    static inline __device__ void gemm( Fragment_aclwmulators ( &acc )[M][N_PER_GROUP],
                                        const Fragment_a ( &a )[M],
                                        const Fragment_b ( &b )[N],
                                        int32_t = 0 ) {
        xmma::helpers::gemm( acc, a, b );
    }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits,
          typename Cta_tile,
          typename Gmem_tile_a,
          typename Gmem_tile_epilogue,
          typename Input_related,
          int32_t STAGES>
using Kernel_traits =
    Interleaved_kernel_traits<Traits, Cta_tile, Gmem_tile_epilogue, Input_related, 32, STAGES>;

///////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace interleaved_dgrad
}  // namespace implicit_gemm
}  // namespace xmma
