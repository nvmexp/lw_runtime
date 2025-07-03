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
#include <xmma/implicit_gemm/fprop_indexed/params.h>
#include <xmma/implicit_gemm/fprop_indexed/gmem_tile.h>
#include <xmma/helpers/callbacks.h>

#include <xmma/implicit_gemm/fprop/traits.h>

namespace xmma {
namespace implicit_gemm {
namespace fprop_indexed {

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
struct Kernel_traits : public fprop::Kernel_traits<Traits_,
                                                   Cta_tile_,
                                                   Gmem_tile_a_,
                                                   Gmem_tile_epilogue_,
                                                   Input_related_,
                                                   STAGES_> {
    using Base = fprop::Kernel_traits<Traits_,
                                      Cta_tile_,
                                      Gmem_tile_a_,
                                      Gmem_tile_epilogue_,
                                      Input_related_,
                                      STAGES_>;

    using Traits = typename Base::Traits;
    using Cta_tile = typename Base::Cta_tile;
    using Input_related = typename Base::Input_related;

    static const xmma::Colwolution_algorithm COLW_ALGO = xmma::Colwolution_algorithm::INDEX;

    // Idx kernels don't need the Input_related
    static_assert( Base::Input_related::STATIC_FILTER_SIZE == 0,
                   "Input_related::STATIC_FILTER_SIZE==0" );

    using Params = fprop_indexed::template Params<typename Base::Traits,
                                                  typename Base::Cta_tile,
                                                  Base::STAGES>;

    static int32_t dynamic_smem_size_per_cta() {
        // The amount of shared memory needed for the main loop.
        const int32_t LOOP_SIZE_IN_BYTES =
            Base::Smem_tile_a::BYTES_PER_TILE + Base::Smem_tile_b::BYTES_PER_TILE +
            Base::Gmem_tile_a::BYTES_PER_EXTRA_SMEM + Base::Gmem_tile_b::BYTES_PER_EXTRA_SMEM;

        // The amount of shared memory needed by the epilogue.
        const int32_t EPILOGUE_SIZE_IN_BYTES = Base::Swizzle_epilogue::BYTES_PER_TILE;

        // The amount of shared memory to launch the kernel.
        return max( LOOP_SIZE_IN_BYTES, EPILOGUE_SIZE_IN_BYTES );
    }

    // Load the delta values and move the filter position.
    template <typename Params>
    static inline __device__ int load_deltas_and_move_filter_position( int64_t &a_delta,
                                                                       int64_t &b_delta,
                                                                       const Params &params,
                                                                       int32_t trsi ) {
        // Are we moving to the next channel?
        int32_t reset_trsi = trsi == params.filter_trs_per_cta - 1;

        // Load the updates. For A, it is complicated so we let the GMEM tile deal with it.
        a_delta = uint64_t( 0 );
        b_delta = params.b_delta[reset_trsi ? 1 : 0];

        // Update the filter position.
        return reset_trsi ? 0 : trsi + 1;
    }

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
};

///////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace fprop_indexed
}  // namespace implicit_gemm
}  // namespace xmma
