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

#include <xmma/ext/batchnorm/bn_apply/ampere/gmem_tile.h>
#include <xmma/ext/batchnorm/bn_apply/ampere/smem_tile.h>
#include <xmma/ext/batchnorm/bn_apply/turing/gmem_tile.h>
#include <xmma/ext/batchnorm/bn_apply/volta/gmem_tile.h>
#include <xmma/ext/batchnorm/bn_stats/ampere/callbacks_epilogue.h>
#include <xmma/ext/batchnorm/bn_stats/turing/callbacks_epilogue.h>
#include <xmma/ext/batchnorm/bn_stats/volta/callbacks_epilogue.h>

////////////////////////////////////////////////////////////////////////////////////////////////////
namespace xmma {
namespace ext {
namespace batchnorm {
namespace bn_apply {
namespace fprop {
///////////////////////////////////////////////////////////////////////////////////////////////////

////// Volta / Turing Traits
template <typename Traits_, typename Cta_tile_, typename Input_related_, int STAGES_ = 1,
          bool WITH_RELU = true, bool WITH_RESIDUAL = false, bool WITH_BNA_RESIDUAL = false,
          bool SIMPLE_1x1x1 = false, bool WITH_BITMASK_RELU_WRITE = false>
struct Kernel_traits
    : public xmma::implicit_gemm::fprop::Kernel_traits<Traits_, Cta_tile_,
        Gmem_tile_a_volta_turing<Traits_, Cta_tile_, Input_related_, WITH_RESIDUAL, STAGES_, 
            WITH_RELU>,
        xmma::implicit_gemm::fprop::Gmem_tile_c_t<Traits_, Cta_tile_, 16>,
        Input_related_, STAGES_> {

    // The traits class.
    using Traits = Traits_;

    // The Cta tile.
    using Cta_tile = Cta_tile_;

    // The kernel parameters.
    using Params = Bn_apply_with_scale_bias_relu_fprop_params<Traits, Cta_tile, STAGES_>;

    // The global memory loader for A.
    using Gmem_tile_a =
        Gmem_tile_a_volta_turing<Traits, Cta_tile, Input_related_, WITH_RESIDUAL, STAGES_, 
            WITH_RELU>;

    // The global memory epilogue.
    using Gmem_tile_epilogue = xmma::implicit_gemm::fprop::Gmem_tile_c_t<Traits, Cta_tile, 16>;

    // The callbacks.
    using Callbacks_epilogue = bn_stats::Batch_norm_fprop_callbacks_epilogue<Traits, Cta_tile>;
    // using Callbacks_epilogue = xmma::helpers::Empty_callbacks_epilogue<Traits, Cta_tile>;

    using Epilogue =
        xmma::helpers::Epilogue_with_split_k<Traits, Cta_tile, xmma::Row,
                                                 Gmem_tile_epilogue, Callbacks_epilogue>;
    /* NOTE: Only FP64 GEMM supports gmem_wo_smem kernel */
    using Gmem_tile_wo_smem_epilogue = Gmem_tile_epilogue;
    // The callbacks.
    using Callbacks_wo_smem_epilogue = Callbacks_epilogue;
    // The epilogue.
    using Epilogue_wo_smem = Epilogue;
    /* NOTE: end. */

    // The amount of shared memory.
    static int dynamic_smem_size_per_cta() {

        // The amount of shared memory needed to compute the BN stats.
        const int BN_SIZE_IN_BYTES = Callbacks_epilogue::BYTES_PER_TILE;

        // Max(main loop, epilogue) used in the FPROP portion
        const int FPROP_SIZE_IN_BYTES = xmma::implicit_gemm::fprop::Kernel_traits<
            Traits, Cta_tile_,
            Gmem_tile_a_volta_turing<Traits, Cta_tile_, Input_related_, WITH_RESIDUAL, STAGES_, 
                WITH_RELU>,
            xmma::implicit_gemm::fprop::Gmem_tile_c_t<Traits, Cta_tile_, 16>,
            Input_related_, STAGES_>::dynamic_smem_size_per_cta();

        // The amount of shared memory to launch the kernel.
        return max(FPROP_SIZE_IN_BYTES, BN_SIZE_IN_BYTES);
    }
};

///////////////////////////////////////////////////////////////////////////////////////////////////
//
//	Ampere Specialization
//
///////////////////////////////////////////////////////////////////////////////////////////////////
template <typename Cta_tile_, typename Input_related_, int STAGES_, bool WITH_RELU, 
          bool WITH_RESIDUAL, bool WITH_BNA_RESIDUAL, bool SIMPLE_1x1x1, 
          bool WITH_BITMASK_RELU_WRITE>
struct Kernel_traits<xmma::Ampere_hmma_fp32_traits, Cta_tile_, Input_related_, STAGES_,
                     WITH_RELU, WITH_RESIDUAL, WITH_BNA_RESIDUAL, SIMPLE_1x1x1, 
                     WITH_BITMASK_RELU_WRITE>
    : public xmma::implicit_gemm::fprop::Kernel_traits<xmma::Ampere_hmma_fp32_traits,
        Cta_tile_,
        Gmem_tile_a<xmma::Ampere_hmma_fp32_traits, Cta_tile_, Input_related_, WITH_RESIDUAL, 
                    WITH_BNA_RESIDUAL, STAGES_>,
        xmma::implicit_gemm::fprop::Gmem_tile_c_t<xmma::Ampere_hmma_fp32_traits, Cta_tile_, 16>,
        Input_related_,
        STAGES_> {

    // WITH_BNA_RESIDUAL has to be false when WITH_RESIDUAL == false
    static_assert( WITH_BNA_RESIDUAL ? WITH_RESIDUAL : 1 , 
                   "Illegal Template Args with DUAL_BNA" );

    // Bitmask RELU Write out is supported only for simple 1x1x1 and with residual flag set
    static_assert( WITH_BITMASK_RELU_WRITE ? WITH_RESIDUAL & SIMPLE_1x1x1 : 1, 
                   "Illegal Template Args with BITMASK_RELU");

    // The traits class.
    using Traits = xmma::Ampere_hmma_fp32_traits;

    // Base class
    using Base = xmma::implicit_gemm::fprop::Kernel_traits< xmma::Ampere_hmma_fp32_traits,
        Cta_tile_,
        bn_apply::Gmem_tile_a<xmma::Ampere_hmma_fp32_traits, Cta_tile_, Input_related_, 
                              WITH_RESIDUAL, WITH_BNA_RESIDUAL, STAGES_>,
        xmma::implicit_gemm::fprop::Gmem_tile_c_t<xmma::Ampere_hmma_fp32_traits, Cta_tile_, 16>,
        Input_related_,
        STAGES_ >;

    // The Cta tile.
    using Cta_tile = Cta_tile_;

    // The kernel parameters.
    using Params = Bn_apply_with_scale_bias_relu_fprop_params<Traits, Cta_tile, STAGES_>;

    // The global memory loader for A.
    using Gmem_tile_a = bn_apply::Gmem_tile_a<Traits, Cta_tile, Input_related_, 
                                              WITH_RESIDUAL, WITH_BNA_RESIDUAL, STAGES_>;

    // The number of buffers in shared memory. It must be at least 2 for LDGSTS.
    enum {
        BUFFERS_PER_SMEM_TILE_A =
            Gmem_tile_a::USE_LDGSTS ? xmma::Max<2, STAGES_>::VALUE : STAGES_
    };

    // The shared memory loader for B.
    using Smem_tile_b = typename Base::Smem_tile_b;

    // The shared memory loader for A.
    using Smem_tile_a =
        bn_apply::Smem_tile_a<Traits, Cta_tile, xmma::Row, Gmem_tile_a::BYTES_PER_LDG,
                              BUFFERS_PER_SMEM_TILE_A, WITH_RESIDUAL, WITH_BNA_RESIDUAL,
                              Gmem_tile_a::LDGS, Smem_tile_b::BYTES_PER_TILE, STAGES_, 
                              WITH_RELU, SIMPLE_1x1x1, WITH_BITMASK_RELU_WRITE>;

    // The compute tile.
    using Compute_tile = xmma::Compute_tile<Traits, Cta_tile, Smem_tile_a, Smem_tile_b>;

    // The global memory epilogue.
    using Gmem_tile_epilogue = xmma::implicit_gemm::fprop::Gmem_tile_c_t<Traits, Cta_tile, 16>;

    // The callbacks.
    using Callbacks_epilogue = bn_stats::Batch_norm_fprop_callbacks_epilogue<Traits, Cta_tile>;

    using Epilogue =
        xmma::helpers::Epilogue_with_split_k<Traits, Cta_tile, xmma::Row,
                                                 Gmem_tile_epilogue, Callbacks_epilogue>;

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

        // If residual add tensor needs its own scale and bias - allocate SMEM for it
        const int RESIDUAL_BYTES = WITH_RESIDUAL ? 
                                       WITH_BNA_RESIDUAL ? 
                                           Smem_tile_a::BYTES_PER_TILE + BN_SCALE_BIAS_BYTES
                                           : Smem_tile_a::BYTES_PER_TILE  
                                       : 0;

        const int MAIN_LOOP_BYTES = Smem_tile_a::BYTES_PER_TILE + Smem_tile_b::BYTES_PER_TILE
                                          + BN_SCALE_BIAS_BYTES + RESIDUAL_BYTES;

        // Compute the space needed in the epilogue
        // First we do swizzle, then bn_stats

        const int SWIZZLE_BYTES = Base::Swizzle_epilogue::BYTES_PER_TILE;

        const int BN_STATS_BYTES = Callbacks_epilogue::BYTES_PER_TILE;

        const int EPILOGUE_BYTES = max(SWIZZLE_BYTES, BN_STATS_BYTES);

        // The amount of shared memory to launch the kernel.
        return max(MAIN_LOOP_BYTES, EPILOGUE_BYTES);
    }
};

///////////////////////////////////////////////////////////////////////////////////////////////////
} // namespace fprop
} // namespace bn_apply
} // namespace batchnorm
} // namespace ext
} // namespace xmma
///////////////////////////////////////////////////////////////////////////////////////////////////
