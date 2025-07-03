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

#include <xmma/ext/batchnorm/bn_apply/wgrad/params.h>
#include <xmma/ext/batchnorm/bn_apply/volta/gmem_tile.h>
#include <xmma/ext/batchnorm/bn_apply/turing/gmem_tile.h>
#include <xmma/implicit_gemm/wgrad_indexed/traits.h>
#include <xmma/ext/batchnorm/bn_apply/ampere/gmem_tile.h>
#include <xmma/ext/batchnorm/bn_apply/ampere/smem_tile.h>

////////////////////////////////////////////////////////////////////////////////////////////////////
namespace xmma {
namespace ext {
namespace batchnorm {
namespace bn_apply {
namespace wgrad {
///////////////////////////////////////////////////////////////////////////////////////////////////

// Volta / Turing Traits
template <typename Traits_, typename Cta_tile_, bool SIMPLE_1x1x1, int STAGES_ = 1,
          bool WITH_RELU = true, bool WITH_FUSED_DBNA_DGRAD = false >
struct Kernel_traits
    : public xmma::implicit_gemm::wgrad_indexed::Kernel_traits<Traits_, Cta_tile_,
        xmma::implicit_gemm::wgrad_indexed::Gmem_tile_a_n<Traits_, Cta_tile_>,
        Gmem_tile_b_volta_turing<Traits_, Cta_tile_, SIMPLE_1x1x1, WITH_RELU>,
        SIMPLE_1x1x1, STAGES_> {

    // This is a workaround. TODO: Use capital letters!
    enum { is_simple_1x1x1 = SIMPLE_1x1x1 };

    // The traits class.
    using Traits = Traits_;

    // The Cta tile.
    using Cta_tile = Cta_tile_;

    // The kernel parameters.
    using Params = Bn_apply_with_scale_bias_relu_wgrad_params<Traits, Cta_tile, STAGES_>;

    // Base class
    using Base = xmma::implicit_gemm::wgrad_indexed::Kernel_traits<Traits, Cta_tile,
        xmma::implicit_gemm::wgrad_indexed::Gmem_tile_a_n<Traits, Cta_tile>,
        Gmem_tile_b_volta_turing<Traits, Cta_tile, SIMPLE_1x1x1, WITH_RELU>,
        SIMPLE_1x1x1, STAGES_>;

    // The global memory loader for B.
    using Gmem_tile_b = Gmem_tile_b_volta_turing<Traits, Cta_tile, SIMPLE_1x1x1, WITH_RELU>;

    // The number of buffers in shared memory. It must be at least 2 for LDGSTS.
    enum { BUFFERS_PER_SMEM_TILE_B = Gmem_tile_b::USE_LDGSTS ? xmma::Max<2, STAGES_>::VALUE : STAGES_ };

    // The shared memory loader for B.
    using Smem_tile_b = xmma::Smem_tile_b<Traits, Cta_tile, xmma::Row,
                                              Gmem_tile_b::BYTES_PER_LDG, BUFFERS_PER_SMEM_TILE_B>;

};

///////////////////////////////////////////////////////////////////////////////////////////////////
//
//	Ampere Specialization
//
///////////////////////////////////////////////////////////////////////////////////////////////////

template < typename Cta_tile_, bool SIMPLE_1x1x1, int STAGES_, 
           bool WITH_RELU, bool WITH_FUSED_DBNA_DGRAD >
struct Kernel_traits< xmma::Ampere_hmma_fp32_traits, Cta_tile_, SIMPLE_1x1x1, STAGES_, 
                      WITH_RELU, WITH_FUSED_DBNA_DGRAD>
    : public xmma::implicit_gemm::wgrad_indexed::Kernel_traits<xmma::Ampere_hmma_fp32_traits,
        Cta_tile_,
        xmma::implicit_gemm::wgrad_indexed::Gmem_tile_a_n<xmma::Ampere_hmma_fp32_traits, Cta_tile_>,
        Gmem_tile_b<xmma::Ampere_hmma_fp32_traits, Cta_tile_, SIMPLE_1x1x1>,
        SIMPLE_1x1x1, STAGES_> {

    // Is simple 1x1x1
    enum { is_simple_1x1x1 = SIMPLE_1x1x1 };

    // The traits class.
    using Traits = xmma::Ampere_hmma_fp32_traits;

    // The Cta tile.
    using Cta_tile = Cta_tile_;

    // The kernel parameters.
    using Params = Bn_apply_with_scale_bias_relu_wgrad_params<Traits, Cta_tile, STAGES_>;

    // The number of stages.
    enum { STAGES = STAGES_ };

    // Base class
    using Base = xmma::implicit_gemm::wgrad_indexed::Kernel_traits<Traits, Cta_tile,
        xmma::implicit_gemm::wgrad_indexed::Gmem_tile_a_n<Traits, Cta_tile>,
        bn_apply::Gmem_tile_b<Traits, Cta_tile, SIMPLE_1x1x1>,
        SIMPLE_1x1x1, STAGES>;

    // Colwolution type.
    static const xmma::Operation_type OPERATION_TYPE = xmma::Operation_type::WGRAD;
    // Colwolution algorithm.
    static const xmma::Colwolution_algorithm COLW_ALGO = xmma::Colwolution_algorithm::INDEX;
    // Colwolution layout.
    static const xmma::Colwolution_layout COLW_LAYOUT = xmma::Colwolution_layout::NHWC;
    // Whether use warp specialized
    static const bool USE_WARP_SPECIALIZATION = false;

    // The global memory loader for B.
    // Do B Tile first, since it allows for A-tile may need to know B-Tile size
    using Gmem_tile_b = bn_apply::Gmem_tile_b<Traits, Cta_tile, SIMPLE_1x1x1>;

    // The number of buffers in shared memory. It must be at least 2 for LDGSTS.
    enum { BUFFERS_PER_SMEM_TILE_B = Gmem_tile_b::USE_LDGSTS ? xmma::Max<2, STAGES>::VALUE 
                                                             : STAGES };

    // The shared memory loader for B.
    using Smem_tile_b = bn_apply::Smem_tile_b<Traits, Cta_tile, xmma::Row,
                                    Gmem_tile_b::BYTES_PER_LDG, BUFFERS_PER_SMEM_TILE_B,
                                    WITH_RELU>;

    // The global memory loader for A.
    using Gmem_tile_a_ = xmma::implicit_gemm::wgrad_indexed::Gmem_tile_a_n<Traits, Cta_tile>;

    using Gmem_tile_a_with_dbna_dy_ = bn_apply::Gmem_tile_a_wgrad<Traits, Cta_tile, SIMPLE_1x1x1, 
                                        WITH_FUSED_DBNA_DGRAD>;

    using Gmem_tile_a = typename std::conditional< WITH_FUSED_DBNA_DGRAD, 
                            Gmem_tile_a_with_dbna_dy_, Gmem_tile_a_ >::type;

    // The number of buffers in shared memory. It must be at least 2 for LDGSTS.
    enum { BUFFERS_PER_SMEM_TILE_A = Gmem_tile_a::USE_LDGSTS ? xmma::Max<2, STAGES>::VALUE 
                                                             : STAGES };

    // The shared memory loader for A.
    using Smem_tile_a_ = xmma::Smem_tile_a<Traits, Cta_tile, xmma::Col,
                                              Gmem_tile_a::BYTES_PER_LDG, BUFFERS_PER_SMEM_TILE_A>;

    using Smem_tile_a_with_dbna_dy_ = bn_apply::Smem_tile_a_wgrad< Traits, Cta_tile, xmma::Col,
                                              Gmem_tile_a::BYTES_PER_LDG, BUFFERS_PER_SMEM_TILE_A,
                                              Gmem_tile_a::LDGS, WITH_FUSED_DBNA_DGRAD,
                                              Smem_tile_b::BYTES_PER_TILE >;

    using Smem_tile_a = typename std::conditional< WITH_FUSED_DBNA_DGRAD, 
                            Smem_tile_a_with_dbna_dy_, Smem_tile_a_ >::type;

    // The compute tile.
    using Compute_tile = typename xmma::Compute_tile_selector<Traits, 
                                                                  Cta_tile, 
                                                                  Smem_tile_a, 
                                                                  Smem_tile_b,
                                                                  OPERATION_TYPE,
                                                                  true>::Class;
    // The amount of shared memory per CTA.
    static int dynamic_smem_size_per_cta() {

        // If we do dgrad fusion - we need this extra space in wgrad
        const int EXTRA_FPROP_TENSOR_BYTES = WITH_FUSED_DBNA_DGRAD ? Smem_tile_a::BYTES_PER_TILE : 0;

        // The amount of shared memory needed for the main loop.
        const int LOOP_SIZE_IN_BYTES = Smem_tile_a::BYTES_PER_TILE + Smem_tile_b::BYTES_PER_TILE 
                                       + EXTRA_FPROP_TENSOR_BYTES;

        // The amount of shared memory needed by the epilogue.
        const int EPILOGUE_SIZE_IN_BYTES = Base::Swizzle_epilogue::BYTES_PER_TILE;

        // The amount of shared memory to launch the kernel.
        return max( LOOP_SIZE_IN_BYTES, EPILOGUE_SIZE_IN_BYTES );
    }
};

///////////////////////////////////////////////////////////////////////////////////////////////////
} // namespace wgrad
} // namespace bn_apply
} // namespace batchnorm
} // namespace ext
} // namespace xmma
///////////////////////////////////////////////////////////////////////////////////////////////////
