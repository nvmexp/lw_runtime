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
#include <xmma/params.h>
#include <xmma/tile_distribution.h>
#include <xmma/compute_tile.h>
#include <xmma/gemm/kernel.h>
#include <xmma/implicit_gemm/wgrad_indexed/params.h>
#include <xmma/implicit_gemm/wgrad_indexed/gmem_tile.h>
#include <xmma/implicit_gemm/wgrad_indexed/fragment_epilogue.h>

namespace xmma {
namespace implicit_gemm {
namespace wgrad_indexed {

///////////////////////////////////////////////////////////////////////////////////////////////////

template <
    // The instruction traits.
    typename Traits_,
    // The CTA tile descriptor.
    typename Cta_tile_,
    // The global memory tile for A (transposed or not).
    typename Gmem_tile_a_,
    // The global memory tile for Epilogue (transposed or not).
    typename Gmem_tile_b_,
    // SIMPLE_1x1x1
    bool SIMPLE_1x1x1,
    // The number of stages in the prefetch pipeline.
    int32_t STAGES_ = 1>

struct Kernel_traits : public Traits_ {

    // This is a workaround. TODO: Use capital letters!
    enum { is_simple_1x1x1 = SIMPLE_1x1x1 };

    // Colwolution type.
    static const xmma::Operation_type OPERATION_TYPE = xmma::Operation_type::WGRAD;
    // Colwolution algorithm.
    static const xmma::Colwolution_algorithm COLW_ALGO = xmma::Colwolution_algorithm::INDEX;
    // Colwolution layout.
    static const xmma::Colwolution_layout COLW_LAYOUT = xmma::Colwolution_layout::NHWC;
    // Whether use warp specialized
    static const bool USE_WARP_SPECIALIZATION = false;

    // The instruction traits.
    using Traits = Traits_;
    // The Cta tile.
    using Cta_tile = Cta_tile_;
    // The number of stages.
    enum { STAGES = STAGES_ };

    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;
    // The kernel parameters.
    using Params = xmma::implicit_gemm::wgrad_indexed::Params<Traits, Cta_tile_, STAGES>;
    // Tile distribution
    using Tile_distribution = xmma::Tile_distribution;

    // The global memory loader for A.
    using Gmem_tile_a = Gmem_tile_a_;
    // The number of buffers in shared memory. It must be at least 2 for LDGSTS.
    enum { BUFFERS_PER_SMEM_TILE_A = Gmem_tile_a::USE_LDGSTS ? Max<2, STAGES>::VALUE : STAGES };
    // The shared memory loader for A.
    using Smem_tile_a = xmma::Smem_tile_a<Traits,
                                          Cta_tile,
                                          typename Gmem_tile_a::Smem_layout,
                                          Gmem_tile_a::BYTES_PER_LDG,
                                          BUFFERS_PER_SMEM_TILE_A>;

    // The global memory loader for B.
    using Gmem_tile_b = Gmem_tile_b_;
    // The number of buffers in shared memory. It must be at least 2 for LDGSTS.
    enum { BUFFERS_PER_SMEM_TILE_B = Gmem_tile_b::USE_LDGSTS ? Max<2, STAGES>::VALUE : STAGES };
    // The shared memory loader for B.
    using Smem_tile_b = xmma::Smem_tile_b<Traits,
                                          Cta_tile,
                                          typename Gmem_tile_b::Smem_layout,
                                          Gmem_tile_b::BYTES_PER_LDG,
                                          BUFFERS_PER_SMEM_TILE_B>;

    // The compute tile.
    using Compute_tile = typename Compute_tile_selector<Traits,
                                                        Cta_tile,
                                                        Smem_tile_a,
                                                        Smem_tile_b,
                                                        OPERATION_TYPE,
                                                        true>::Class;

    // The global memory epilogue.
    using Gmem_tile_epilogue =
        xmma::implicit_gemm::wgrad_indexed::Gmem_tile_epilogue<Traits, Cta_tile, SIMPLE_1x1x1>;
    // The fragment to store.
    using Fragment_epilogue_pre_swizzle =
        xmma::implicit_gemm::wgrad_indexed::Fragment_epilogue_pre_swizzle<Traits, Cta_tile>;
    // The shared memory epilogue tile.
    using Swizzle_epilogue = xmma::Swizzle_epilogue<Traits, Cta_tile, xmma::Row>;
    // The callbacks.
    using Callbacks_epilogue =
        xmma::helpers::Empty_callbacks_epilogue<Traits, Cta_tile, Fragment_epilogue_pre_swizzle>;
    // The epilogue.
    using Epilogue = xmma::helpers::Epilogue_with_split_k<Traits,
                                                          Cta_tile,
                                                          xmma::Row,
                                                          Gmem_tile_epilogue,
                                                          Callbacks_epilogue,
                                                          Swizzle_epilogue,
                                                          Fragment_epilogue_pre_swizzle>;

    /* NOTE: Only FP64 GEMM supports gmem_wo_smem kernel */
    // The global memory loader for A without smem.
    using Gmem_wo_smem_tile_a = xmma::Gmem_wo_smem_tile_a<Traits,
                                                          Cta_tile,
                                                          typename Gmem_tile_a::Smem_layout,
                                                          Gmem_tile_a::BYTES_PER_LDG>;
    // The global memory loader for A without smem.
    using Gmem_wo_smem_tile_b = xmma::Gmem_wo_smem_tile_b<Traits,
                                                          Cta_tile,
                                                          typename Gmem_tile_b::Smem_layout,
                                                          Gmem_tile_b::BYTES_PER_LDG>;

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
        return &::xmma::gemm::kernel<Kernel_traits>;
    }

    // Return split k kernel function pointer.
    static XMMA_HOST Kernel_type split_k_kernel_ptr() {
        return &::xmma::gemm::split_k_kernel<Kernel_traits>;
    }
#endif

    // The number of threads in the CTA.
    static int32_t threads_per_cta( const Params params = Params() ) {
        return Cta_tile::THREADS_PER_CTA;
    }

    // The amount of shared memory per CTA.
    static int32_t dynamic_smem_size_per_cta() {
        // The amount of shared memory needed for the main loop.
        const int32_t LOOP_SIZE_IN_BYTES =
            Smem_tile_a::BYTES_PER_TILE + Smem_tile_b::BYTES_PER_TILE;

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
    static inline __device__ int32_t initialize_filter_position( const Params & ) {
        return 0;
    }

    // Load the delta values and move the filter position.
    template <typename Params>
    static inline __device__ int32_t load_deltas_and_move_filter_position( int64_t &a_delta,
                                                                           int64_t &b_delta,
                                                                           const Params &params,
                                                                           int32_t ) {
        // Load the updates. For A, let the GMEM tile deal with that.
        a_delta = params.a_delta[0];
        b_delta = uint64_t( 0 );
        if( SIMPLE_1x1x1 ) {
            b_delta = params.b_delta[0];
        }

        // Update the filter position so to say :).
        return 0;
    }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace wgrad_indexed
}  // namespace implicit_gemm
}  // namespace xmma
