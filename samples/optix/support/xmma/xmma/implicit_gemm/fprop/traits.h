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
#include <xmma/volta/fragment.h>
#include <xmma/volta/smem_tile.h>

#include <xmma/turing/traits.h>
#include <xmma/turing/fragment.h>
#include <xmma/turing/smem_tile.h>

#include <xmma/ampere/traits.h>
#include <xmma/ampere/fragment.h>
#include <xmma/ampere/smem_tile.h>
#include <xmma/ampere/gmem_wo_smem_tile.h>

#include <xmma/helpers/epilogue_with_split_k.h>
#include <xmma/helpers/epilogue_gmma.h>
#include <xmma/helpers/callbacks.h>

#include <xmma/params.h>
#include <xmma/tile_distribution.h>
#include <xmma/compute_tile.h>
#include <xmma/gemm/kernel.h>

#include <xmma/implicit_gemm/fprop/params.h>
#include <xmma/implicit_gemm/fprop/gmem_tile.h>
#include <xmma/implicit_gemm/fprop/gmem_tile_hopper.h>

///////////////////////////////////////////////////////////////////////////////////////////////////

namespace xmma {
namespace implicit_gemm {
namespace fprop {

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
struct Kernel_traits : public Traits_ {

    // Colwolution type.
    static const xmma::Operation_type OPERATION_TYPE = xmma::Operation_type::FPROP;
    // Colwolution algorithm.
    static const xmma::Colwolution_algorithm COLW_ALGO = xmma::Colwolution_algorithm::PRECOMPUTED;
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

    enum { FLT_T = Input_related::FLT_T };
    enum { FLT_R = Input_related::FLT_R };
    enum { FLT_S = Input_related::FLT_S };
    enum { STATIC_FILTER_SIZE = Input_related::STATIC_FILTER_SIZE };

    // The number of stages.
    enum { STAGES = STAGES_ };

    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;
    // The kernel parameters.
    using Params = xmma::implicit_gemm::fprop::Params<Traits, Cta_tile, STAGES>;
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
    using Gmem_tile_b = xmma::implicit_gemm::fprop::Gmem_tile_b<Traits, Cta_tile, Input_related>;
    // The number of buffers in shared memory. It must be at least 2 for LDGSTS.
    enum { BUFFERS_PER_SMEM_TILE_B = Gmem_tile_b::USE_LDGSTS ? Max<2, STAGES>::VALUE : STAGES };
    // The shared memory loader for B.
    using Smem_tile_b = xmma::Smem_tile_b<Traits,
                                          Cta_tile,
                                          xmma::Col,
                                          Gmem_tile_b::BYTES_PER_LDG,
                                          BUFFERS_PER_SMEM_TILE_B>;

    // The compute tile.
    using Compute_tile =
        typename Compute_tile_selector<Traits, Cta_tile, Smem_tile_a, Smem_tile_b, OPERATION_TYPE>::
            Class;

    // The global memory epilogue.
    using Gmem_tile_epilogue = Gmem_tile_epilogue_;
    // The shared memory epilogue tile.
    using Swizzle_epilogue = xmma::Swizzle_epilogue<Traits,
                                                    Cta_tile,
                                                    typename Gmem_tile_epilogue::Layout,
                                                    Gmem_tile_epilogue::BYTES_PER_STG>;
    // The callbacks.
    using Callbacks_epilogue = xmma::helpers::Callbacks_epilogue_with_bias_and_relu<
        Traits,
        Cta_tile,
        xmma::Fragment_epilogue_pre_swizzle<Traits, Cta_tile>,
        xmma::Fragment_epilogue_post_swizzle<Traits, Cta_tile>,
        xmma::Fragment_c<Traits, Cta_tile>,
        false,
        false,
        typename Gmem_tile_epilogue::Layout,
        Gmem_tile_epilogue::BYTES_PER_STG>;
    // The epilogue.
    using Epilogue = xmma::helpers::Epilogue_with_split_k<Traits,
                                                          Cta_tile,
                                                          typename Gmem_tile_epilogue::Layout,
                                                          Gmem_tile_epilogue,
                                                          Callbacks_epilogue>;

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

    // The amount of shared memory.
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
    static inline __device__ int load_deltas_and_move_filter_position( int64_t &a_delta,
                                                                       int64_t &b_delta,
                                                                       const Params &params,
                                                                       int32_t trsi ) {
        // Load the updates.
        a_delta = params.a_delta[trsi];
        b_delta = params.b_delta[trsi];

        // Update the filter position.
        int32_t filter_trs_per_cta =
            ( STATIC_FILTER_SIZE ? FLT_T * FLT_R * FLT_S : params.filter_trs_per_cta );
        return trsi == filter_trs_per_cta - 1 ? 0 : trsi + 1;
    }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

template <
    // The instruction traits.
    typename Traits_,
    // The CTA tile descriptor.
    typename Cta_tile_,
    // The global memory tile for A (transposed or not).
    typename Gmem_tile_a_,
    // The global memory tile for B (transposed or not).
    typename Gmem_tile_b_,
    // The global memory tile for Epilogue (transposed or not).
    typename Gmem_tile_epilogue_,
    // Input related params
    int32_t STAGES_ = 1>
struct Kernel_traits_tma : public Traits_ {

    // Colwolution type.
    static const xmma::Operation_type OPERATION_TYPE = xmma::Operation_type::FPROP;
    // Colwolution algorithm.
    static const xmma::Colwolution_algorithm COLW_ALGO = xmma::Colwolution_algorithm::PRECOMPUTED;
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
    using Params = xmma::implicit_gemm::fprop::Params<Traits, Cta_tile, STAGES>;
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
                                          BUFFERS_PER_SMEM_TILE_A,
                                          true,
                                          Gmem_tile_a::USE_TMA>;

    // The global memory loader for B.
    using Gmem_tile_b = Gmem_tile_b_;
    // The number of buffers in shared memory. It must be at least 2 for LDGSTS.
    enum { BUFFERS_PER_SMEM_TILE_B = Gmem_tile_b::USE_LDGSTS ? Max<2, STAGES>::VALUE : STAGES };
    // The shared memory loader for B.
    using Smem_tile_b = xmma::Smem_tile_b<Traits,
                                          Cta_tile,
                                          xmma::Col,
                                          Gmem_tile_b::BYTES_PER_LDG,
                                          BUFFERS_PER_SMEM_TILE_B,
                                          true,
                                          Gmem_tile_b::USE_TMA>;

    // The compute tile.
    using Compute_tile =
        typename Compute_tile_selector<Traits, Cta_tile, Smem_tile_a, Smem_tile_b, OPERATION_TYPE>::
            Class;

    // The global memory epilogue.
    using Gmem_tile_epilogue = Gmem_tile_epilogue_;
    // The shared memory epilogue tile.
    using Swizzle_epilogue = xmma::Swizzle_epilogue<Traits,
                                                    Cta_tile,
                                                    typename Gmem_tile_epilogue::Layout,
                                                    Gmem_tile_epilogue::BYTES_PER_STG>;
    // The callbacks.
    using Callbacks_epilogue = xmma::helpers::Callbacks_epilogue_with_bias_and_relu<
        Traits,
        Cta_tile,
        xmma::Fragment_epilogue_pre_swizzle<Traits, Cta_tile>,
        xmma::Fragment_epilogue_post_swizzle<Traits, Cta_tile>,
        xmma::Fragment_c<Traits, Cta_tile>,
        false,
        false,
        typename Gmem_tile_epilogue::Layout,
        Gmem_tile_epilogue::BYTES_PER_STG>;
    // The epilogue.
    using Epilogue = xmma::helpers::Epilogue_with_split_k<Traits,
                                                          Cta_tile,
                                                          typename Gmem_tile_epilogue::Layout,
                                                          Gmem_tile_epilogue,
                                                          Callbacks_epilogue>;

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
    // The epilogue.
    using Epilogue_wo_smem = Epilogue;
    /* NOTE: end. */

    // The number of threads in the CTA.
    static int32_t threads_per_cta() {
        return Cta_tile::THREADS_PER_CTA;
    }

    // The amount of shared memory.
    static int32_t dynamic_smem_size_per_cta() {
        // The amount of shared memory needed for the main loop.
        const int32_t LOOP_SIZE_IN_BYTES =
            Smem_tile_a::BYTES_PER_TILE + Smem_tile_b::BYTES_PER_TILE;

        // The amount of shared memory needed by the epilogue.
        const int32_t EPILOGUE_SIZE_IN_BYTES = Swizzle_epilogue::BYTES_PER_TILE;

        // The amount of shared memory to launch the kernel.
        return max( LOOP_SIZE_IN_BYTES, EPILOGUE_SIZE_IN_BYTES );
    }

    // Initialize the filter position.
    template <typename Params>
    static inline __device__ int32_t initialize_filter_position( const Params & ) {
        return 0;
    }

    // Load the delta values and move the filter position.
    template <typename Params>
    static inline __device__ int load_deltas_and_move_filter_position( int64_t &a_delta,
                                                                       int64_t &b_delta,
                                                                       const Params &params,
                                                                       int32_t trsi ) {
        int32_t next_trsi = trsi == params.filter_trs_per_cta - 1 ? 0 : trsi + 1;
        // Load the updates.
        a_delta = params.filter_coord_a[next_trsi];
        b_delta = params.filter_coord_a[next_trsi];

        // Update the filter position.
        return next_trsi;
    }
};

#ifdef USE_GMMA
// gmma specific compute_tile, gmma specific smem_tile
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
    int32_t STAGES_ = 1,
    // The number of GMMA Stages to be issued in the prologue
    int32_t GMMA_STAGES_ = 1>
struct Gmma_kernel_traits : public Traits_ {

    // Colwolution type.
    static const xmma::Operation_type OPERATION_TYPE = xmma::Operation_type::FPROP;
    // Colwolution algorithm.
    static const xmma::Colwolution_algorithm COLW_ALGO = xmma::Colwolution_algorithm::PRECOMPUTED;
    // Colwolution layout.
    static const xmma::Colwolution_layout COLW_LAYOUT = xmma::Colwolution_layout::NHWC;
    // Whether use warp specialized
    static const bool USE_WARP_SPECIALIZATION = false;

    // The instruction traits.
    using Traits = Traits_;
    // is A operand in RF for GMMA?
    static constexpr bool GMMA_A_RF = Traits::GMMA_A_RF;
    // is B operand in RF for GMMA?
    static constexpr bool GMMA_B_RF = Traits::GMMA_B_RF;
    // The Cta tile.
    using Cta_tile = Cta_tile_;
    // Template parameters which are related to input parameters like filter size.
    using Input_related = Input_related_;

    enum { FLT_T = Input_related::FLT_T };
    enum { FLT_R = Input_related::FLT_R };
    enum { FLT_S = Input_related::FLT_S };
    enum { STATIC_FILTER_SIZE = Input_related::STATIC_FILTER_SIZE };

    // is syncthreads needed after ldgdepbar?
    // For now, if there is only 1 warpgroup, syncthreads is not needed.
    static const bool SYNCTHREADS_NEEDED = Cta_tile::THREADS_PER_CTA > 128 ? true : false;

    // The number of stages.
    enum { STAGES = STAGES_ };
    // The number of GMMA Stages to be issued in the prologue
    enum { GMMA_STAGES = GMMA_STAGES_ };

    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;
    // The kernel parameters.
    using Params = xmma::implicit_gemm::fprop::Params<Traits, Cta_tile, STAGES>;
    // Tile distribution
    using Tile_distribution = xmma::Tile_distribution;

    // The global memory loader for A.
    using Gmem_tile_a = Gmem_tile_a_;
    // The number of buffers in shared memory. It must be at least 2 for LDGSTS.
    enum { BUFFERS_PER_SMEM_TILE_A = Gmem_tile_a::USE_LDGSTS ? Max<2, STAGES>::VALUE : STAGES };
    // The shared memory loader for A.
    using Smem_tile_a = xmma::Smem_tile_hopper_a<Traits,
                                                 Cta_tile,
                                                 typename Gmem_tile_a::Smem_layout,
                                                 BUFFERS_PER_SMEM_TILE_A,
                                                 Gmem_tile_a::GMMA_DESC_MODE>;

    // The global memory loader for B.
    static constexpr xmma::Gmma_descriptor_mode gmma_desc_mode_b =
        Cta_tile::K * sizeof( typename Traits::B_type ) >= 128
            ? xmma::Gmma_descriptor_mode::SWIZZLE_128B
            : xmma::Gmma_descriptor_mode::SWIZZLE_64B;

    using Gmem_tile_b = xmma::implicit_gemm::fprop::
        Gmem_tile_gmma_b<Traits, Cta_tile, Input_related, gmma_desc_mode_b>;
    // The number of buffers in shared memory. It must be at least 2 for LDGSTS.
    enum { BUFFERS_PER_SMEM_TILE_B = Gmem_tile_b::USE_LDGSTS ? Max<2, STAGES>::VALUE : STAGES };
    // The shared memory loader for B.
    using Smem_tile_b = xmma::Smem_tile_hopper_b<Traits,
                                                 Cta_tile,
                                                 xmma::Col,
                                                 BUFFERS_PER_SMEM_TILE_B,
                                                 Gmem_tile_b::GMMA_DESC_MODE>;

    // The compute tile.
    using Compute_tile = wip_do_not_use::Compute_tile_with_gmma<Traits,
                                                                Cta_tile,
                                                                Smem_tile_a,
                                                                Smem_tile_b,
                                                                GMMA_A_RF,
                                                                GMMA_B_RF,
                                                                STAGES>;

    // The global memory epilogue.
    using Gmem_tile_epilogue = Gmem_tile_epilogue_;
    // The shared memory epilogue tile.
    using Swizzle_epilogue = xmma::Swizzle_epilogue<Traits,
                                                    Cta_tile,
                                                    typename Gmem_tile_epilogue::Layout,
                                                    Gmem_tile_epilogue::BYTES_PER_STG>;
    // The callbacks.
    using Epilogue_layout = typename Gmem_tile_epilogue::Layout;
    using Callbacks_epilogue = xmma::helpers::Callbacks_gmma_epilogue_with_bias_and_relu<
        Traits,
        Cta_tile,
        xmma::Fragment_gmma_epilogue_pre_swizzle<Traits, Cta_tile, Epilogue_layout>,
        xmma::Fragment_gmma_epilogue_post_swizzle<Traits, Cta_tile, Epilogue_layout>,
        xmma::Fragment_gmma_c<Traits, Cta_tile, Epilogue_layout>,
        false,
        false,
        Epilogue_layout,
        Gmem_tile_epilogue::BYTES_PER_STG>;
    // The epilogue.
    using Epilogue = xmma::helpers::Epilogue_gmma_with_split_k<Traits,
                                                               Cta_tile,
                                                               Epilogue_layout,
                                                               Gmem_tile_epilogue,
                                                               Callbacks_epilogue,
                                                               Swizzle_epilogue>;
    // The number of threads in the CTA.
    static int32_t threads_per_cta() {
        return Cta_tile::THREADS_PER_CTA;
    }

    // The amount of shared memory.
    static int32_t dynamic_smem_size_per_cta() {
        // The amount of shared memory needed for the main loop.
        const int32_t LOOP_SIZE_IN_BYTES =
            Smem_tile_a::BYTES_PER_TILE + Smem_tile_b::BYTES_PER_TILE;

        // The amount of shared memory needed by the epilogue.
        const int32_t EPILOGUE_SIZE_IN_BYTES = Swizzle_epilogue::BYTES_PER_TILE;

        // The amount of shared memory to launch the kernel.
        return max( LOOP_SIZE_IN_BYTES, EPILOGUE_SIZE_IN_BYTES );
    }

    // Initialize the filter position.
    template <typename Params>
    static inline __device__ int32_t initialize_filter_position( const Params & ) {
        return 0;
    }

    // Load the delta values and move the filter position.
    template <typename Params>
    static inline __device__ int load_deltas_and_move_filter_position( int64_t &a_delta,
                                                                       int64_t &b_delta,
                                                                       const Params &params,
                                                                       int32_t trsi ) {
        // Load the updates.
        a_delta = params.a_delta[trsi];
        b_delta = params.b_delta[trsi];

        // Update the filter position.
        int32_t filter_trs_per_cta =
            ( STATIC_FILTER_SIZE ? FLT_T * FLT_R * FLT_S : params.filter_trs_per_cta );
        return trsi == filter_trs_per_cta - 1 ? 0 : trsi + 1;
    }
};
#endif

}  // namespace fprop
}  // namespace implicit_gemm
}  // namespace xmma
