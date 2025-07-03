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
#include <xmma/tile_distribution.h>
#include <xmma/compute_tile.h>

#include <xmma/helpers/epilogue_with_split_k.h>
#include <xmma/helpers/fragment.h>
#include <xmma/helpers/gemm.h>
#include <xmma/helpers/epilogue_gmma.h>
#include <xmma/helpers/callbacks.h>

#include <xmma/gemm/params.h>
#include <xmma/gemm/gmem_tile.h>
#include <xmma/gemm/kernel.h>

#include <xmma/ampere/gmem_wo_smem_tile.h>

namespace xmma {
namespace gemm {

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    // The instruction traits.
    typename Traits_,
    // The CTA tile descriptor.
    typename Cta_tile_,
    // The global memory tile for A (transposed or not).
    typename Gmem_tile_a_,
    // The global memory tile for B (transposed or not).
    typename Gmem_tile_b_,
    // The number of stages in the prefetch pipeline.
    int32_t STAGES_ = 1>
struct Kernel_traits : public Traits_ {

    // Gemm type.
    static const xmma::Operation_type OPERATION_TYPE = xmma::Operation_type::GEMM;
    // Whether use warp specialized
    static const bool USE_WARP_SPECIALIZATION = false;

    // The instruction traits.
    using Traits = Traits_;
    // The Cta tile.
    using Cta_tile = Cta_tile_;
    // The number of stages in the multi-stage pipeline for prefetching.
    enum { STAGES = STAGES_ };

    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;
    // The kernel parameters.
    using Params = xmma::gemm::Params<Traits, Cta_tile, STAGES>;
    // Tile distribution
    using Tile_distribution = xmma::Tile_distribution;

    // The global memory loader for A.
    using Gmem_tile_a = Gmem_tile_a_;
    // The number of buffers in shared memory. It must be at least 2 for LDGSTS.
    enum { BUFFERS_PER_SMEM_TILE_A = Gmem_tile_a::USE_LDGSTS ? Max<2, STAGES>::VALUE : STAGES };
    // The shared memory layout for A.
    using Smem_layout_a = typename Gmem_tile_a::Smem_layout;
    // The shared memory loader for A.
    using Smem_tile_a = xmma::Smem_tile_a<Traits, Cta_tile, Smem_layout_a, Gmem_tile_a::BYTES_PER_LDG, BUFFERS_PER_SMEM_TILE_A,
        true, Gmem_tile_a::USE_UTMALDG>;
    // The global memory loader for A without smem.
    using Gmem_wo_smem_tile_a = xmma::Gmem_wo_smem_tile_a<Traits, Cta_tile, Smem_layout_a, Gmem_tile_a::BYTES_PER_LDG>;

    // The global memory loader for B.
    using Gmem_tile_b = Gmem_tile_b_;
    // The number of buffers in shared memory. It must be at least 2 for LDGSTS.
    enum { BUFFERS_PER_SMEM_TILE_B = Gmem_tile_b::USE_LDGSTS ? Max<2, STAGES>::VALUE : STAGES };
    // The shared memory layout for B.
    using Smem_layout_b = typename Gmem_tile_b::Smem_layout;
    // The shared memory loader for B.
    using Smem_tile_b = xmma::Smem_tile_b<Traits, Cta_tile, Smem_layout_b, Gmem_tile_b::BYTES_PER_LDG, BUFFERS_PER_SMEM_TILE_B,
        true, Gmem_tile_b::USE_UTMALDG>;
    // The global memory loader for A without smem.
    using Gmem_wo_smem_tile_b = xmma::Gmem_wo_smem_tile_b<Traits, Cta_tile, Smem_layout_b,
                                                          Gmem_tile_b::BYTES_PER_LDG>;

    // The compute tile.
    using Compute_tile = xmma::Compute_tile<Traits, Cta_tile, Smem_tile_a, Smem_tile_b>;

    // The global memory epilogue.
    using Gmem_tile_epilogue = xmma::gemm::Gmem_tile_epilogue<Traits, Cta_tile>;
    // The global memory epilogue without smem.
    using Gmem_tile_wo_smem_epilogue = xmma::gemm::Gmem_tile_wo_smem_epilogue<Traits, Cta_tile>;
    // The shared memory epilogue tile.
    using Swizzle_epilogue = xmma::Swizzle_epilogue<Traits, Cta_tile, xmma::Row>;

    // The callbacks for epilogue
    enum {
        IS_C_I8_EPILOGUE_F32 =
            lwca::is_same<int8_t, typename Traits::C_type>::value && std::is_same<float, typename Traits::Epilogue_type>::value
    };
    using Callbacks_epilogue_bias_relu = xmma::helpers::Callbacks_epilogue_with_bias_and_relu<
        Traits,
        Cta_tile,
        xmma::Fragment_epilogue_pre_swizzle<Traits, Cta_tile>,
        xmma::Fragment_epilogue_post_swizzle<Traits, Cta_tile>,
        xmma::Fragment_c<Traits, Cta_tile>,
        false,
        Traits::IS_GELU_ERF>;

    using Callbacks_epilogue_bias_relu_per_channel_scaling =
        xmma::helpers::Callbacks_epilogue_with_bias_relu_per_channel_scaling<Traits, Cta_tile,
        xmma::Fragment_epilogue_pre_swizzle<Traits, Cta_tile>,
        xmma::Fragment_epilogue_post_swizzle<Traits, Cta_tile>,
        xmma::Fragment_c<Traits, Cta_tile>,
        false,
        Traits::IS_GELU_ERF>;
    // Enable per_channel_scaling for int8 kernels.
    using Callbacks_epilogue = typename lwca::conditional<IS_C_I8_EPILOGUE_F32, //FIXME: per_channel with gelu_erf
                                                 Callbacks_epilogue_bias_relu_per_channel_scaling,
                                                 Callbacks_epilogue_bias_relu>::type;

    using Callbacks_wo_smem_epilogue = xmma::helpers::Callbacks_wo_smem_epilogue<Traits, Cta_tile>;

    // The epilogue.
    using Epilogue =
        xmma::helpers::Epilogue_with_split_k<Traits, Cta_tile, xmma::Row,
                                             Gmem_tile_epilogue, Callbacks_epilogue>;
    using Epilogue_wo_smem =
        xmma::helpers::Epilogue_wo_smem<Traits, Cta_tile, xmma::Row,
                                        Gmem_tile_wo_smem_epilogue, Callbacks_wo_smem_epilogue>;

    // The callbacks for input fusion
    using Callback_fuse_a = xmma::gemm::Callback_fuse_input_empty;
    using Callback_fuse_b = xmma::gemm::Callback_fuse_input_empty;

#if !defined(__LWDACC_RTC__)
    typedef void (*Kernel_type)(Params params);

    // Return device kernel function pointer.
    static XMMA_HOST Kernel_type kernel_ptr(const Params params = Params()) {
        return &::xmma::gemm::kernel<Kernel_traits>;
    }

    // Return split k kernel function pointer.
    static XMMA_HOST Kernel_type split_k_kernel_ptr() {
        return &::xmma::gemm::split_k_kernel<Kernel_traits>;
    }
#endif

    // Number of threads per CTA.
    static int32_t threads_per_cta(const Params params = Params()) {
        return Cta_tile::THREADS_PER_CTA;
    }

    // The amount of dynamic shared memory per CTA.
    static int32_t dynamic_smem_size_per_cta() {
        // The amount of shared memory needed for the main loop.
        const int32_t LOOP_SIZE_IN_BYTES = STAGES == 0
            ? 0
            : Smem_tile_a::BYTES_PER_TILE + Smem_tile_b::BYTES_PER_TILE;

        // The amount of shared memory needed by the epilogue.
        const int32_t EPILOGUE_SIZE_IN_BYTES = STAGES == 0
            ? 0
            : Swizzle_epilogue::BYTES_PER_TILE;

        // The amount of shared memory needed to store tma barriers
        const int32_t BARRIER_FOR_TMA_SIZE_IN_BYTES =
            ( Gmem_tile_a::USE_UTMALDG || Gmem_tile_b::USE_UTMALDG ? BUFFERS_PER_SMEM_TILE_A * 8
                                                                   : 0 );

        // The amount of shared memory to launch the kernel.
        return max( LOOP_SIZE_IN_BYTES + BARRIER_FOR_TMA_SIZE_IN_BYTES,
                    EPILOGUE_SIZE_IN_BYTES + BARRIER_FOR_TMA_SIZE_IN_BYTES );
    }

    // The amount of epilogue shared memory per CTA.
    static int32_t epilogue_smem_size_per_cta() {
        // The amount of shared memory needed by the epilogue.
        const int32_t EPILOGUE_SIZE_IN_BYTES = STAGES == 0
            ? 0
            : Swizzle_epilogue::BYTES_PER_TILE;

        return EPILOGUE_SIZE_IN_BYTES;
    }

    // Initialize the filter position.
    template <typename Params>
    static inline __device__ int32_t initialize_filter_position( const Params& ) {
        return 0;
    }

    // Load the delta values and move the filter position.
    template <typename Params>
    static inline __device__ int32_t load_deltas_and_move_filter_position( int64_t& a_delta,
                                                                       int64_t& b_delta,
                                                                       const Params& params, int32_t ) {
        a_delta = params.a_delta[0];
        b_delta = params.b_delta[0];
        return 0;
    }
};

template < typename Kernel_traits >
struct Kernel_info {

};

////////////////////////////////////////////////////////////////////////////////////////////////////
// probably should be consolidated
#ifdef USE_GMMA
// gmma specific compute_tile, gmma specific smem_tile
template <
    // The instruction traits.
    typename Traits_,
    // The CTA tile descriptor.
    typename Cta_tile_,
    // The global memory tile for A (transposed or not).
    typename Gmem_tile_a_,
    // The global memory tile for B (transposed or not).
    typename Gmem_tile_b_,
    // The layout of C/D, could be row major or column major
    typename Layout_c_,
    // The number of stages in the prefetch pipeline.
    int32_t STAGES_ = 1,
    // The number of GMMA Stages to be issued in the prologue
    int32_t GMMA_STAGES_ = 1
    >
struct Gmma_kernel_traits : public Traits_ {

    // Gemm type.
    static const xmma::Operation_type OPERATION_TYPE = xmma::Operation_type::GEMM;

    // The instruction traits.
    using Traits = Traits_;
    // is A operand in RF for GMMA?
    static constexpr bool GMMA_A_RF = Traits::GMMA_A_RF;
    // is B operand in RF for GMMA?
    static constexpr bool GMMA_B_RF = Traits::GMMA_B_RF;
    // The Cta tile.
    using Cta_tile = Cta_tile_;
    // is syncthreads needed after ldgdepbar?
    // For now, if there is only 1 warpgroup, syncthreads is not needed.
    static const bool SYNCTHREADS_NEEDED = Cta_tile::THREADS_PER_CTA > 128 ? true : false;
    // The number of stages in the multi-stage pipeline for prefetching.
    enum { STAGES = STAGES_ };
    // The number of GMMA Stages to be issued in the prologue
    enum { GMMA_STAGES = GMMA_STAGES_ };

    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;
    // The kernel parameters.
    using Params = Params<Traits, Cta_tile, STAGES>;
    // Tile distribution
    using Tile_distribution = Tile_distribution;

    // The global memory loader for A.
    using Gmem_tile_a = Gmem_tile_a_;
    // The number of buffers in shared memory. It must be at least 2 for LDGSTS.
    enum { BUFFERS_PER_SMEM_TILE_A = Gmem_tile_a::USE_LDGSTS ? Max<2, STAGES>::VALUE : STAGES };
    // The shared memory layout for A.
    using Smem_layout_a = typename Gmem_tile_a::Smem_layout;
    // The shared memory loader for A.
    //using Smem_tile_a = Smem_tile_a<Traits, Cta_tile, Smem_layout_a, Gmem_tile_a::BYTES_PER_LDG,
    //                                BUFFERS_PER_SMEM_TILE_A>;
    using Smem_tile_a = xmma::Smem_tile_hopper_a<
        Traits, Cta_tile, Smem_layout_a, BUFFERS_PER_SMEM_TILE_A,
        Gmem_tile_a::GMMA_DESC_MODE, GMMA_A_RF,
        xmma::Gmma_fusion_mode::NO_FUSION, Gmem_tile_a::USE_UTMALDG>;

    // The global memory loader for B.
    using Gmem_tile_b = Gmem_tile_b_;
    // The number of buffers in shared memory. It must be at least 2 for LDGSTS.
    enum { BUFFERS_PER_SMEM_TILE_B = Gmem_tile_b::USE_LDGSTS ? Max<2, STAGES>::VALUE : STAGES };
    // The shared memory layout for B.
    using Smem_layout_b = typename Gmem_tile_b::Smem_layout;
    // The shared memory loader for B.
    //using Smem_tile_b = Smem_tile_b<Traits, Cta_tile, Smem_layout_b, Gmem_tile_b::BYTES_PER_LDG,
    //                                BUFFERS_PER_SMEM_TILE_B>;
    using Smem_tile_b = xmma::Smem_tile_hopper_b<
        Traits, Cta_tile, Smem_layout_b, BUFFERS_PER_SMEM_TILE_A,
        Gmem_tile_b::GMMA_DESC_MODE, Gmem_tile_b::USE_UTMALDG>;

    // The compute tile.
    using Compute_tile = wip_do_not_use::Compute_tile_with_gmma<Traits,
                                                                Cta_tile,
                                                                Smem_tile_a,
                                                                Smem_tile_b,
                                                                Traits::GMMA_A_RF,
                                                                Traits::GMMA_B_RF,
                                                                STAGES>;


    // The global memory epilogue.
    using Gmem_tile_epilogue = Gmem_tile_gmma_epilogue<Traits, Cta_tile, Layout_c_>;
    // The shared memory epilogue tile.
    using Swizzle_epilogue = xmma::Swizzle_epilogue<Traits, Cta_tile, Layout_c_>;
    // The callbacks.
    using Callbacks_epilogue = xmma::helpers::Callbacks_gmma_epilogue_with_bias_and_relu<
        Traits,
        Cta_tile,
        xmma::Fragment_gmma_epilogue_pre_swizzle<Traits, Cta_tile, Layout_c_>,
        xmma::Fragment_gmma_epilogue_post_swizzle<Traits, Cta_tile, Layout_c_>,
        xmma::Fragment_gmma_c<Traits, Cta_tile, Layout_c_>,
        false,
        false,
        Layout_c_,
        Gmem_tile_epilogue::BYTES_PER_STG>;
    // The epilogue.
    using Epilogue = xmma::helpers::Epilogue_gmma_with_split_k<Traits,
                                                               Cta_tile,
                                                               Layout_c_,
                                                               Gmem_tile_epilogue,
                                                               Callbacks_epilogue,
                                                               Swizzle_epilogue>;

    // The number of threads per CTA.
    static int32_t threads_per_cta() {
        return Cta_tile::THREADS_PER_CTA;
    }

    // The amount of dynamic shared memory per CTA.
    static int32_t dynamic_smem_size_per_cta() {
        // The amount of shared memory needed for the main loop.
        const int32_t LOOP_SIZE_IN_BYTES = Smem_tile_a::BYTES_PER_TILE + Smem_tile_b::BYTES_PER_TILE;

        // The amount of shared memory needed by the epilogue.
        const int32_t EPILOGUE_SIZE_IN_BYTES = Swizzle_epilogue::BYTES_PER_TILE;

        // The amount of shared memory needed to store tma barriers
        const int32_t BARRIER_FOR_TMA_SIZE_IN_BYTES =
            ( Gmem_tile_a::USE_UTMALDG || Gmem_tile_b::USE_UTMALDG ? BUFFERS_PER_SMEM_TILE_A * 8
                                                                   : 0 );

        // The amount of shared memory to launch the kernel.
        return max( LOOP_SIZE_IN_BYTES + BARRIER_FOR_TMA_SIZE_IN_BYTES,
                    EPILOGUE_SIZE_IN_BYTES + BARRIER_FOR_TMA_SIZE_IN_BYTES );
    }

    // Initialize the filter position.
    template <typename Params>
    static inline __device__ int32_t initialize_filter_position( const Params& ) {
        return 0;
    }

    // Load the delta values and move the filter position.
    template <typename Params>
    static inline __device__ int32_t load_deltas_and_move_filter_position( int64_t& a_delta,
                                                                       int64_t& b_delta,
                                                                       const Params& params, int32_t ) {
        a_delta = params.a_delta[0];
        b_delta = params.b_delta[0];
        return 0;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// gmma specific compute_tile, gmma specific smem_tile
template <
    // The instruction traits.
    typename Traits_,
    // The CTA tile descriptor.
    typename Cta_tile_,
    // The global memory tile for A (transposed or not).
    typename Gmem_tile_a_,
    // The global memory tile for B (transposed or not).
    typename Gmem_tile_b_,
    // The layout of C/D, could be row major or column major
    typename Layout_c_,
    // The number of stages in the prefetch pipeline.
    int32_t STAGES_ = 1,
    // The number of GMMA Stages to be issued in the prologue
    int32_t GMMA_STAGES_ = 1
    >
struct Gmma_bn_apply_kernel_traits :
  public Gmma_kernel_traits< Traits_,
                             Cta_tile_,
                             Gmem_tile_a_,
                             Gmem_tile_b_,
                             Layout_c_,
                             STAGES_,
                             GMMA_STAGES_ > {

    // Gemm type.
    static const xmma::Operation_type OPERATION_TYPE = xmma::Operation_type::GEMM;

    // Base class
    using Base = Gmma_kernel_traits< Traits_,
                             Cta_tile_,
                             Gmem_tile_a_,
                             Gmem_tile_b_,
                             Layout_c_,
                             STAGES_,
                             GMMA_STAGES_ >;

    // The instruction traits.
    using Traits = typename Base::Traits;
    // is A operand in RF for GMMA?
    static constexpr bool GMMA_A_RF = Base::GMMA_A_RF;
    // is B operand in RF for GMMA?
    static constexpr bool GMMA_B_RF = Base::GMMA_B_RF;
    // for now, we always fuse A
    static constexpr xmma::Gmma_fusion_mode GMMA_FUSION_MODE_A = xmma::Gmma_fusion_mode::BN_APPLY;

    // The Cta tile.
    using Cta_tile = typename Base::Cta_tile;
    // is syncthreads needed after ldgdepbar?
    // For now, if there is only 1 warpgroup, syncthreads is not needed.
    static const bool SYNCTHREADS_NEEDED = Base::SYNCTHREADS_NEEDED;
    // The number of stages in the multi-stage pipeline for prefetching.
    enum { STAGES = Base::STAGES };
    // The number of GMMA Stages to be issued in the prologue
    enum { GMMA_STAGES = Base::GMMA_STAGES };

    // The kernel parameters.
    using Params = Bn_apply_with_scale_bias_relu_params<Traits, Cta_tile, STAGES>;

    // The global memory loader for A.
    using Gmem_tile_a = typename Base::Gmem_tile_a;
    // The number of buffers in shared memory. It must be at least 2 for LDGSTS.
    enum { BUFFERS_PER_SMEM_TILE_A = Base::BUFFERS_PER_SMEM_TILE_A };
    // The shared memory layout for A.
    using Smem_layout_a = typename Base::Smem_layout_a;
    // The shared memory loader for A.
    //using Smem_tile_a = Smem_tile_a<Traits, Cta_tile, Smem_layout_a, Gmem_tile_a::BYTES_PER_LDG,
    //                                BUFFERS_PER_SMEM_TILE_A>;
    using Smem_tile_a = xmma::Smem_tile_hopper_a<Traits, Cta_tile, Smem_layout_a,
        BUFFERS_PER_SMEM_TILE_A, Gmem_tile_a::GMMA_DESC_MODE, GMMA_A_RF, GMMA_FUSION_MODE_A >;


    // The shared memory layout for B.
    using Smem_tile_b = typename Base::Smem_tile_b;


    // The compute tile.
    // cannot use base class compute_tile since we have a different smem_tile_a
    using Compute_tile = wip_do_not_use::Compute_tile_with_gmma<Traits,
                                                                Cta_tile,
                                                                Smem_tile_a,
                                                                Smem_tile_b,
                                                                Traits::GMMA_A_RF,
                                                                Traits::GMMA_B_RF,
                                                                STAGES>;

    // The shared memory epilogue tile.
    using Swizzle_epilogue = typename Base::Swizzle_epilogue;

    // The amount of dynamic shared memory per CTA.
    static int32_t dynamic_smem_size_per_cta() {
        // The amount of shared memory needed for the main loop.
        // note that we save some SMEM for scale and bias.
        const int32_t LOOP_SIZE_IN_BYTES = Smem_tile_a::BYTES_PER_TILE
                                     + Smem_tile_b::BYTES_PER_TILE
                                     + Smem_tile_a::BYTES_PER_SCALE_BIAS_PER_TILE;

        // The amount of shared memory needed by the epilogue.
        const int32_t EPILOGUE_SIZE_IN_BYTES = Swizzle_epilogue::BYTES_PER_TILE;

        // The amount of shared memory needed to store tma barriers
        const int32_t BARRIER_FOR_TMA_SIZE_IN_BYTES =
            Gmem_tile_a::USE_UTMALDG ? BUFFERS_PER_SMEM_TILE_A * 8 : 0;

        // The amount of shared memory to launch the kernel.
        return max( LOOP_SIZE_IN_BYTES + BARRIER_FOR_TMA_SIZE_IN_BYTES,
                    EPILOGUE_SIZE_IN_BYTES + BARRIER_FOR_TMA_SIZE_IN_BYTES );
    }

};


#endif
///////////////////////////////////////////////////////////////////////////////////////////////////

template <
    // The instruction traits.
    typename Traits,
    // The CTA tile descriptor.
    typename Cta_tile,
    // The global memory tile for A (transposed or not).
    typename Gmem_tile_a,
    // The global memory tile for B (transposed or not).
    typename Gmem_tile_b,
    // The number of stages in the prefetch pipeline.
    int32_t STAGES>
using Gemm_traits = Kernel_traits<Traits, Cta_tile, Gmem_tile_a, Gmem_tile_b, STAGES>;

///////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace gemm
}  // namespace xmma
