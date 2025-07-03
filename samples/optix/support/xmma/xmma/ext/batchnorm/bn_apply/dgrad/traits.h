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
#include <xmma/xmma.h>
#include <xmma/traits.h>
#include <xmma/ampere/traits.h>
#include <xmma/implicit_gemm/dgrad/traits.h>
#include <xmma/ext/batchnorm/epilogue_dbns.h>
#include <xmma/ext/batchnorm/bn_apply/dgrad/params.h>
#include <xmma/ext/batchnorm/bn_apply/dgrad/gmem_tile.h>
#include <xmma/ext/batchnorm/bn_stats/dgrad/gmem_tile.h>
#include <xmma/ext/batchnorm/bn_apply/dgrad/smem_tile.h>
#include <xmma/ext/batchnorm/bn_apply/dgrad/ampere/gmem_tile.h>
#include <xmma/ext/batchnorm/bn_apply/dgrad/ampere/smem_tile.h>
#include <xmma/ext/batchnorm/bn_stats/dgrad/activation.h>
#include <xmma/ext/batchnorm/bn_stats/dgrad/callbacks_epilogue.h>
#include <xmma/ext/batchnorm/bn_stats/dgrad/ampere/callbacks_epilogue.h>
#include <xmma/ext/batchnorm/bn_stats/dgrad/ampere/smem_tile.h>

////////////////////////////////////////////////////////////////////////////////////////////////////
namespace xmma {
namespace ext {
namespace batchnorm {
namespace bn_apply {
namespace dgrad {
///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////
//
//	Ampere Specialization
//
///////////////////////////////////////////////////////////////////////////////////////////////////
template <typename Traits_,
          typename Cta_tile_,
          typename Input_related_,
          typename Activation_,
          xmma::ext::batchnorm::ReluBitmaskFormat ReluBitMaskFormat_,
          bool DUAL_DBNS_ = false,
          int STAGES_ = 1>
struct Kernel_traits
    : public xmma::implicit_gemm::dgrad::Kernel_traits<
          xmma::Ampere_hmma_fp32_traits,
          Cta_tile_,
          xmma::ext::batchnorm::bn_apply::dgrad::Gmem_tile_a_dbna_dgrad<
              xmma::Ampere_hmma_fp32_traits,
              Cta_tile_,
              Input_related_,
              STAGES_>,
          xmma::implicit_gemm::dgrad::Gmem_tile_c_t<xmma::Ampere_hmma_fp32_traits, Cta_tile_, 16>,
          Input_related_,
          STAGES_> {

    // The traits class.
    using Traits = xmma::Ampere_hmma_fp32_traits;

    // Activation.
    using Activation = Activation_;

    // Base class
    using Base = xmma::implicit_gemm::dgrad::Kernel_traits<
        xmma::Ampere_hmma_fp32_traits,
        Cta_tile_,
        xmma::ext::batchnorm::bn_apply::dgrad::Gmem_tile_a_dbna_dgrad<xmma::Ampere_hmma_fp32_traits,
                                                                      Cta_tile_,
                                                                      Input_related_,
                                                                      STAGES_>,
        xmma::implicit_gemm::dgrad::Gmem_tile_c_t<xmma::Ampere_hmma_fp32_traits, Cta_tile_, 16>,
        Input_related_,
        STAGES_>;

    // The Cta tile.
    using Cta_tile = Cta_tile_;

    // The kernel parameters.
    using Params = xmma::ext::batchnorm::bn_apply::dgrad::
        Bn_apply_with_scale_bias_relu_dgrad_params<Traits, Cta_tile, STAGES_>;

    // The global memory loader for A.
    using Gmem_tile_a = xmma::ext::batchnorm::bn_apply::dgrad::
        Gmem_tile_a_dbna_dgrad<Traits, Cta_tile, Input_related_, STAGES_>;

    // The number of buffers in shared memory. It must be at least 2 for LDGSTS.
    enum {
        BUFFERS_PER_SMEM_TILE_A = Gmem_tile_a::USE_LDGSTS ? xmma::Max<2, STAGES_>::VALUE : STAGES_
    };

    // The shared memory loader for B.
    using Smem_tile_b = typename Base::Smem_tile_b;

    // The shared memory loader for A.
    using Smem_tile_a =
        xmma::ext::batchnorm::bn_apply::dgrad::Smem_tile_a_dbna_dgrad<Traits,
                                                                      Cta_tile,
                                                                      xmma::Row,
                                                                      Gmem_tile_a::BYTES_PER_LDG,
                                                                      BUFFERS_PER_SMEM_TILE_A,
                                                                      Gmem_tile_a::LDGS,
                                                                      Smem_tile_b::BYTES_PER_TILE,
                                                                      STAGES_>;

    // The compute tile.
    using Compute_tile = xmma::Compute_tile<Traits, Cta_tile, Smem_tile_a, Smem_tile_b>;

    // Smem fro swizzle
    using Swizzle_epilogue =
        xmma::Swizzle_turing_hmma_fp32_epilogue_bn_stats<Traits_, Cta_tile_, xmma::Row>;

    // The global memory epilogue.
    using Gmem_tile_epilogue =
        xmma::ext::batchnorm::bn_stats::dgrad::Gmem_tile_c_t<Traits, Cta_tile, 16>;

    // The callbacks.
    using Callbacks_epilogue =
        xmma::ext::batchnorm::bn_stats::dgrad::Batch_norm_dgrad_callbacks_epilogue<
            Traits,
            Cta_tile,
            Activation,
            ReluBitMaskFormat_,
            DUAL_DBNS_>;

    using Epilogue = xmma::helpers::Epilogue_dbns<Traits,
                                                  Cta_tile,
                                                  xmma::Row,
                                                  Gmem_tile_epilogue,
                                                  ReluBitMaskFormat_,
                                                  Callbacks_epilogue,
                                                  Swizzle_epilogue,
                                                  DUAL_DBNS_>;

    /* NOTE: Only FP64 GEMM supports gmem_wo_smem kernel */
    using Gmem_tile_wo_smem_epilogue = Gmem_tile_epilogue;
    // The callbacks.
    using Callbacks_wo_smem_epilogue = Callbacks_epilogue;
    // The epilogue.
    using Epilogue_wo_smem = Epilogue;
    /* NOTE: end. */

    // The amount of shared memory.
    static int dynamic_smem_size_per_cta() {

        // Compute the space needed for the main loop: tile_a + tile_b + scale + bias + residual

        const int BN_SCALE_BIAS_BYTES = Smem_tile_a::BYTES_PER_TILE_SCALE * 2;

        const int RESIDUAL_BYTES = Smem_tile_a::BYTES_PER_TILE;

        const int MAIN_LOOP_BYTES = Smem_tile_a::BYTES_PER_TILE + Smem_tile_b::BYTES_PER_TILE +
                                    BN_SCALE_BIAS_BYTES + RESIDUAL_BYTES;

        // Compute the space needed in the epilogue
        // First we do swizzle, then bn_stats

        const int SWIZZLE_BYTES = Base::Swizzle_epilogue::BYTES_PER_TILE;

        const int BN_STATS_BYTES = Callbacks_epilogue::BYTES_PER_TILE;

        const int EPILOGUE_BYTES = max( SWIZZLE_BYTES, BN_STATS_BYTES );

        // The amount of shared memory to launch the kernel.
        return max( MAIN_LOOP_BYTES, EPILOGUE_BYTES );
    }
};

///////////////////////////////////////////////////////////////////////////////////////////////////
}  // namespace dgrad
}  // namespace bn_apply
}  // namespace batchnorm
}  // namespace ext
}  // namespace xmma
///////////////////////////////////////////////////////////////////////////////////////////////////
