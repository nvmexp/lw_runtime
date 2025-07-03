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
#include <xmma/implicit_gemm/strided_dgrad_indexed/params.h>
#include <xmma/implicit_gemm/strided_dgrad_indexed/gmem_tile.h>
#include <xmma/implicit_gemm/strided_dgrad_indexed/kernel.h>
#include <xmma/helpers/callbacks.h>

namespace xmma {
namespace implicit_gemm {
namespace strided_dgrad_indexed {

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    // The instruction traits.
    typename Traits_,
    // The CTA tile descriptor.
    typename Cta_tile_,
    // The global memory tile for A (transposed or not).
    typename Gmem_tile_a_,
    // The global memory tile for Epilogue (transposed or not).
    typename Gmem_tile_epilogue_,
    // Input related params
    typename Input_related_,
    // The number of stages in the prefetch pipeline.
    int32_t STAGES_ = 1>
struct Kernel_traits : public Traits_ {

    // Colwolution type.
    static const xmma::Operation_type OPERATION_TYPE = xmma::Operation_type::STRIDED_DGRAD;
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
    // Template parameters which are related to input parameters like filter size.
    using Input_related = Input_related_;
    // Idx kernels don't need the Input_related
    static_assert( Input_related::STATIC_FILTER_SIZE == 0, "Input_related::STATIC_FILTER_SIZE==0" );
    // The number of stages.
    enum { STAGES = STAGES_ };

    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;
    // The kernel parameters.
    using Params = xmma::implicit_gemm::strided_dgrad_indexed::Params<Traits, Cta_tile, STAGES>;
    // Tile distribution
    using Tile_distribution = xmma::Tile_distribution;

    // The global memory loader for A.
    using Gmem_tile_a = xmma::implicit_gemm::strided_dgrad_indexed::Gmem_tile_a<Traits, Cta_tile>;
    // The shared memory loader for A.
    using Smem_tile_a = xmma::Smem_tile_a<Traits, Cta_tile, xmma::Row, 16, STAGES>;

    // The global memory loader for B.
    using Gmem_tile_b = xmma::implicit_gemm::strided_dgrad_indexed::Gmem_tile_b<Traits, Cta_tile>;
    // The shared memory loader for B.
    using Smem_tile_b = xmma::Smem_tile_b<Traits, Cta_tile, xmma::Row, 16, STAGES>;

    // The compute tile.
    using Compute_tile = typename xmma::
        Compute_tile_selector<Traits, Cta_tile, Smem_tile_a, Smem_tile_b, OPERATION_TYPE>::Class;
    // The global memory epilogue.
    using Gmem_tile_epilogue = Gmem_tile_epilogue_;
    // The shared memory epilogue tile.
    using Swizzle_epilogue = xmma::Swizzle_epilogue<Traits, Cta_tile, xmma::Row>;
    // The callbacks.
    using Callbacks_epilogue =
        xmma::helpers::Callbacks_epilogue_with_bias_and_relu<Traits, Cta_tile>;
    // The epilogue.
    using Epilogue = xmma::helpers::
        Epilogue_with_split_k<Traits, Cta_tile, xmma::Row, Gmem_tile_epilogue, Callbacks_epilogue>;

#if !defined( __LWDACC_RTC__ )
    typedef void ( *Kernel_type )( Params params );

    // Return device kernel function pointer.
    static XMMA_HOST Kernel_type kernel_ptr( const Params params = Params() ) {
        return &::xmma::implicit_gemm::strided_dgrad_indexed::kernel<Kernel_traits>;
    }

    // Return split k kernel function pointer.
    static XMMA_HOST Kernel_type split_k_kernel_ptr() {
        return &::xmma::gemm::split_k_kernel<Kernel_traits>;
    }

    // Return helper stage kernel funtion pointers.
    static XMMA_HOST Kernel_type kernel_helper_stage_ptr( const int32_t stage ) {
        switch( stage ) {
        case 0:
            return &::xmma::implicit_gemm::strided_dgrad_indexed::kernel_helper_stage_0<
                Kernel_traits>;
        case 1:
            return &::xmma::implicit_gemm::strided_dgrad_indexed::kernel_helper_stage_1<
                Kernel_traits>;
        case 2:
            return &::xmma::implicit_gemm::strided_dgrad_indexed::kernel_helper_stage_2<
                Kernel_traits>;
        case 3:
            return &::xmma::implicit_gemm::strided_dgrad_indexed::kernel_helper_stage_3<
                Kernel_traits>;
        default:
            return nullptr;
        };
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
                                           4 * static_cast<int32_t>( sizeof( int32_t ) ) * Cta_tile::M;

        // The amount of shared memory needed by the epilogue.
        const int32_t EPILOGUE_SIZE_IN_BYTES = Swizzle_epilogue::BYTES_PER_TILE;

        const int32_t SINGLE_EXTRA_BUFFER_SIZE_IN_BYTES = 
            Cta_tile::M * static_cast<int32_t>( sizeof( int32_t ) );

        // The amount of shared memory to launch the kernel.
        return xmma::div_up( max( LOOP_SIZE_IN_BYTES, EPILOGUE_SIZE_IN_BYTES ), 4 ) * 4 +
               SINGLE_EXTRA_BUFFER_SIZE_IN_BYTES;
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

#if 0
    // Load the delta values and move the filter position.
    template< typename Params >
    static inline __device__ int32_t load_deltas_and_move_filter_position(int64_t &a_delta,
                                                                      int64_t &b_delta,
                                                                      const Params &params,
                                                                      int32_t trsi) {
        // Are we moving to the next channel?
        int32_t reset_trsi = trsi == params.filter_trs_per_cta - 1;

        // Load the updates. For A, it is complicated so we let the GMEM tile deal with it.
        a_delta = uint64_t(0);
        b_delta = params.b_delta[reset_trsi ? 1 : 0];

        // Update the filter position.
        return reset_trsi ? 0 : trsi + 1;
    }
#endif
};

///////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace strided_dgrad_indexed
}  // namespace implicit_gemm
}  // namespace xmma
