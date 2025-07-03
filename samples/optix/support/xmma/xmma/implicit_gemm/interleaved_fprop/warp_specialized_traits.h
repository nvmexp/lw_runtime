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

#include <xmma/helpers/epilogue_with_split_k.h>
#include <xmma/params.h>
#include <xmma/tile_distribution.h>
#include <xmma/compute_tile.h>

#include <xmma/smem_tile.h>
#include <xmma/implicit_gemm/interleaved_fprop/gmem_tile.h>
#include <xmma/implicit_gemm/interleaved_fprop/callbacks.h>

#include <xmma/implicit_gemm/interleaved_fprop/warp_specialized_params.h>
#include <xmma/warp_specialized_traits.h>

namespace xmma {
namespace implicit_gemm {
namespace interleaved_fprop {

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
    int BYTES_PER_PACKET,
    // The arch being compiled for this warp specialized kernel.
    int32_t ARCH_ = 80>  // int STAGES_ = 1>
struct Warp_specialized_interleaved_kernel_traits : public Traits_ {

    // Colwolution type.
    static const xmma::Operation_type OPERATION_TYPE = xmma::Operation_type::FPROP;
    // Colwolution algorithm.
    static const xmma::Colwolution_algorithm COLW_ALGO = xmma::Colwolution_algorithm::PRECOMPUTED;
    // Colwolution layout.
    static const xmma::Colwolution_layout COLW_LAYOUT = xmma::Colwolution_layout::NCHW_VECT_C_32;
    // Whether use warp specialized
    static const bool USE_WARP_SPECIALIZATION = true;

    // The warp specialized kernel traits type.
    enum { ARRIVE_WAIT = 0, NAMED_BARRIER = 1 };
    // The arch for smem allocation.
    enum { ARCH = ARCH_ };

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
    enum { STAGES = 1 };

    // The XMMA tile.
    using Xmma_tile = typename Traits::Xmma_tile<Cta_tile>;

    // The kernel parameters.
    using Params =
        xmma::implicit_gemm::interleaved_fprop::Warp_specialized_params<Traits,
                                                                        Cta_tile>;  // STAGES = 1>;

    // The global memory loader for A.
    using Gmem_tile_a = xmma::implicit_gemm::interleaved_fprop::
        Gmem_tile_a<Traits, Cta_tile, Input_related, BYTES_PER_PACKET>;

    // The global memory loader for B.
    using Gmem_tile_b = xmma::implicit_gemm::interleaved_fprop::
        Gmem_tile_b<Traits, Cta_tile, Input_related, BYTES_PER_PACKET>;

    // The warp specialized kernel traits.
    using Warp_specialized_traits =
        typename xmma::Warp_specialized_traits_selector<Traits,
                                                        Cta_tile,
                                                        xmma::Col_interleaved,
                                                        xmma::Row_interleaved,
                                                        xmma::Col_interleaved,
                                                        Gmem_tile_a::BYTES_PER_LDG,
                                                        Gmem_tile_b::BYTES_PER_LDG,
                                                        ARCH,
                                                        ARRIVE_WAIT,
                                                        0,
                                                        0,
                                                        true>::Class;

    // Tile distribution_persisitent
    using Tile_distribution_persistent =
        typename Warp_specialized_traits::Tile_distribution_persistent;

    // Tile distribution
    using Tile_distribution = xmma::Tile_distribution;

    enum { WARP_SPECIALIZED_CONFIG = Warp_specialized_traits::WARP_SPECIALIZED_CONFIG };
    enum { BUFFERS_PER_SMEM_TILE_A = Warp_specialized_traits::BUFFERS_PER_SMEM_TILE_A };
    enum { BUFFERS_PER_SMEM_TILE_B = Warp_specialized_traits::BUFFERS_PER_SMEM_TILE_B };

    // The shared memory loader for A.
    using Smem_tile_a = xmma::Smem_tile_a<Traits,
                                          Cta_tile,
                                          xmma::Col_interleaved,
                                          Gmem_tile_a::BYTES_PER_LDG,
                                          BUFFERS_PER_SMEM_TILE_A>;

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

    // The callbacks for ReLU lowerbound == 0.
    using Callbacks_epilogue_lb_zero =
        xmma::implicit_gemm::interleaved_fprop::Callbacks_epilogue_fuse<
            Traits,
            Cta_tile,
            Gmem_tile_epilogue,
            xmma::Fragment_epilogue_pre_swizzle<Traits, Cta_tile>,
            xmma::Fragment_epilogue_interleaved_post_swizzle<Traits, Cta_tile>,
            typename Gmem_tile_epilogue::Fragment_c,
            false>;

    // The callbacks for ReLU lowerbound != 0.
    using Callbacks_epilogue_lb_nonzero =
        xmma::implicit_gemm::interleaved_fprop::Callbacks_epilogue_fuse<
            Traits,
            Cta_tile,
            Gmem_tile_epilogue,
            xmma::Fragment_epilogue_pre_swizzle<Traits, Cta_tile>,
            xmma::Fragment_epilogue_interleaved_post_swizzle<Traits, Cta_tile>,
            typename Gmem_tile_epilogue::Fragment_c,
            true>;

    // The callbacks.
    // We use non-zero because it will disable RELU by default
    using Callbacks_epilogue = Callbacks_epilogue_lb_nonzero;

    // The epilogue.
    using Epilogue_wosplitk = xmma::helpers::Epilogue<Traits,
                                                      Cta_tile,
                                                      xmma::Col_interleaved,
                                                      Gmem_tile_epilogue,
                                                      Callbacks_epilogue,
                                                      Swizzle_epilogue>;

    using Epilogue = Epilogue_wosplitk;

    // WARNING : SPLIT-K Is not supported with IMMA
    // This is just to make the compilation work fine.
    using Epilogue_withsplitk = Epilogue_wosplitk;

    // The epilogue for ReLU lowerbound != 0.
    using Epilogue_lb_nonzero = xmma::helpers::Epilogue<Traits,
                                                        Cta_tile,
                                                        xmma::Col_interleaved,
                                                        Gmem_tile_epilogue,
                                                        Callbacks_epilogue_lb_nonzero,
                                                        Swizzle_epilogue>;

    enum {
        SMEM_BYTES_PER_CTA =
            Smem_tile_a::BYTES_PER_TILE +
            Smem_tile_b::BYTES_PER_TILE + 
            Warp_specialized_traits::EPILOGUE_SIZE_IN_BYTES +
            Warp_specialized_traits::ARRIVE_WAIT_SMEM_SIZE
    };

    static_assert( (int)SMEM_BYTES_PER_CTA <= (int)Warp_specialized_traits::SMEM_BYTES_PER_SM,
                   "error: Shared memory needed exceeds capacity" );

    // The number of threads per CTA.
    static int threads_per_cta() {
        return Cta_tile::THREADS_PER_CTA;
    }

    // Initialize the filter position.
    template <typename Params>
    static inline __device__ int initialize_filter_position( const Params& params ) {
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
    static inline __device__ int load_deltas_and_move_filter_position( int64_t& a_delta,
                                                                       int64_t& b_delta,
                                                                       const Params& params,
                                                                       int trsi ) {
        int32_t filter_trs_per_cta =
            ( STATIC_FILTER_SIZE ? FLT_T * FLT_R * FLT_S : params.filter_trs_per_cta );
        // Early exit for 1x1x1 filters.
        if( filter_trs_per_cta == 1 ) {
            a_delta = params.a_delta[0];
            b_delta = params.b_delta[0];
            return 0;
        }

        // Are we moving to a new channel?
        int reset_trsi = trsi >= filter_trs_per_cta - Gmem_tile_a::COLS;

        // Load the updates.
        a_delta = params.a_delta[trsi];
        b_delta = params.b_delta[trsi];

        // Update the filter position.
        return ( reset_trsi ? trsi - filter_trs_per_cta : trsi ) + Gmem_tile_a::COLS;
    }

    // Gemm.
    template <typename Fragment_aclwmulators,
              typename Fragment_a,
              typename Fragment_b,
              int M,
              int N,
              int N_PER_GROUP>
    static inline __device__ void gemm( Fragment_aclwmulators ( &acc )[M][N_PER_GROUP],
                                        const Fragment_a ( &a )[M],
                                        const Fragment_b ( &b )[N],
                                        int = 0 ) {
        xmma::helpers::gemm( acc, a, b );
    }

    // The amount of shared memory.
    static int dynamic_smem_size_per_cta() {
        // The amount of shared memory to launch the kernel.
        return SMEM_BYTES_PER_CTA;
    }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits,
          typename Cta_tile,
          typename Gmem_tile_epilogue,
          typename Input_related,
          int32_t ARCH = 80>
using Warp_specialized_kernel_traits =
    Warp_specialized_interleaved_kernel_traits<Traits,
                                               Cta_tile,
                                               Gmem_tile_epilogue,
                                               Input_related,
                                               32,
                                               ARCH>;

///////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace interleaved_fprop
}  // namespace implicit_gemm
}  // namespace xmma
