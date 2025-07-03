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
#include <xmma/helpers/callbacks.h>

#include <xmma/params.h>
#include <xmma/tile_distribution.h>
#include <xmma/compute_tile.h>

#include <xmma/implicit_gemm/utils.h>
#include <xmma/implicit_gemm/fprop/traits.h>
#include <xmma/implicit_gemm/dgrad_indexed/params.h>
#include <xmma/implicit_gemm/dgrad_indexed/gmem_tile.h>

namespace xmma {
namespace implicit_gemm {
namespace dgrad_indexed {

///////////////////////////////////////////////////////////////////////////////////////////////////

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

    // Colwolution type.
    static const xmma::Operation_type OPERATION_TYPE = xmma::Operation_type::DGRAD;
    // Colwolution algorithm.
    static const xmma::Colwolution_algorithm COLW_ALGO = xmma::Colwolution_algorithm::INDEX;

    // Idx kernels don't need the Input_related
    static_assert( Base::Input_related::STATIC_FILTER_SIZE == 0,
                   "Input_related::STATIC_FILTER_SIZE==0" );
    // The number of stages.
    enum { STAGES = STAGES_ };

    // The kernel parameters.
    using Params = dgrad_indexed::Params<typename Base::Traits, typename Base::Cta_tile, STAGES>;

    // The global memory loader for B.
    using Gmem_tile_b = dgrad::
        Gmem_tile_b<typename Base::Traits, typename Base::Cta_tile, typename Base::Input_related>;
    // The number of buffers in shared memory. It must be at least 2 for LDGSTS.
    static const int32_t BUFFERS_PER_SMEM_TILE_B =
        Gmem_tile_b::USE_LDGSTS ? (int)Max<2, STAGES>::VALUE : STAGES;
    // The shared memory loader for B.
    using Smem_tile_b = xmma::Smem_tile_b<typename Base::Traits,
                                          typename Base::Cta_tile,
                                          xmma::Row,
                                          Gmem_tile_b::BYTES_PER_LDG,
                                          BUFFERS_PER_SMEM_TILE_B>;

    // The compute tile.
    using Compute_tile = typename Compute_tile_selector<typename Base::Traits,
                                                        typename Base::Cta_tile,
                                                        typename Base::Smem_tile_a,
                                                        Smem_tile_b,
                                                        OPERATION_TYPE>::Class;

    /* NOTE: Only FP64 GEMM supports gmem_wo_smem kernel */
    // The global memory loader for A without smem.
    using Gmem_wo_smem_tile_b = xmma::Gmem_wo_smem_tile_b<typename Base::Traits,
                                                          typename Base::Cta_tile,
                                                          typename Gmem_tile_b::Smem_layout,
                                                          Gmem_tile_b::BYTES_PER_LDG>;

    /* NOTE: end. */

    // The amount of shared memory per CTA.
    static int32_t dynamic_smem_size_per_cta() {
        // The amount of shared memory needed for the main loop.
        const int32_t LOOP_SIZE_IN_BYTES =
            Base::Smem_tile_a::BYTES_PER_TILE + Smem_tile_b::BYTES_PER_TILE +
            Base::Gmem_tile_a::BYTES_PER_EXTRA_SMEM + Gmem_tile_b::BYTES_PER_EXTRA_SMEM;

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
};

///////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace dgrad_indexed
}  // namespace implicit_gemm
}  // namespace xmma
