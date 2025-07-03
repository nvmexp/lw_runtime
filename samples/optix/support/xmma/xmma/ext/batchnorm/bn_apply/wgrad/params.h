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

#include <xmma/implicit_gemm/fprop/params.h>

////////////////////////////////////////////////////////////////////////////////////////////////////
namespace xmma {
namespace ext {
namespace batchnorm {
namespace bn_apply {
namespace wgrad {
////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits, typename Cta_tile, int32_t STAGES>
struct Bn_apply_with_scale_bias_relu_wgrad_params
    : public xmma::implicit_gemm::wgrad_indexed::Params<Traits, Cta_tile, STAGES> {

    using Base = xmma::implicit_gemm::wgrad_indexed::Params<Traits, Cta_tile, STAGES>;

    // The residual (to be added in the main loop, it may be nullptr).
    const void *bn_res_gmem;

    // The scale.
    typename Traits::Epilogue_type alpha;
    typename Traits::Epilogue_type beta;

    // Do we have a residual in the epilogue?
    int32_t with_residual;

    // Do we add a residual in the main loop?
    int32_t with_residual_in_main_loop;

    // The scale and bias
    const void *scale_gmem;
    const void *bias_gmem;

    // Do we finalize the BN stats?
    int32_t bn_finalize_stats;
    // The buffer to store partial sums.
    void *bn_partial_sums_gmem, *bn_partial_sums_of_squares_gmem;
    // The mean and the ilw-stddev computed by the kernel.
    void *bn_sum_gmem, *bn_mean_gmem, *bn_sum_of_squares_gmem, *bn_ilw_stddev_gmem;
    // The constant to compute the sqrt of the variance.
    float bn_epsilon;

    // Do we apply BN?
    int32_t bn_apply;
    // The scale and bias.
    void *bn_scale_gmem, *bn_bias_gmem;
    // The output of BN.
    void *bn_out_gmem;

    // The number of channels.
    int32_t num_channels;
    // The number of partial sums.
    int32_t num_partial_sums;
    // The number of pixels.
    float ilw_count;

    // Disable writing the stats output
    int32_t bn_disable_stats_output;

    // Amount of memory to allocate for partial sums
    int32_t bn_partial_sums_sz;

    // Dgrad Fusion Arrays
    void *bna_fprop_tensor_gmem;
    void *bna_bias_gmem;
    void *bna_fprop_tensor_scale_gmem;
    void *bna_grad_scale_gmem;

    // Ctor.
    Bn_apply_with_scale_bias_relu_wgrad_params() : Base() {
    }

    // TODO:deprecate in future.
    template <typename Bn_traits>
    xmma::Error initialize( xmma::Host_workspace<Bn_traits> *workspace ) {

        int32_t xmmas_m = Bn_traits::Gmem_tile_epilogue::Layout::ROW
                              ? Bn_traits::Xmma_tile::XMMAS_M
                              : Bn_traits::Xmma_tile::XMMAS_N;

        XMMA_CALL(
            this->callwlate_grid_dimensions( workspace->grid, workspace->split_k_grid, xmmas_m ) );

        XMMA_CALL( this->finalize( workspace->grid ) );
        // Initialize workspace

        // Do we need a sequential reduction?
        workspace->split_k_with_reduction = this->split_k.with_reduction();
        workspace->device_workspace_size = this->split_k.size_in_bytes();
        workspace->smem_size = Bn_traits::dynamic_smem_size_per_cta();
        const int32_t EPILOGUE_SIZE_IN_BYTES = Bn_traits::Swizzle_epilogue::BYTES_PER_TILE;
        workspace->epilogue_size_in_bytes = EPILOGUE_SIZE_IN_BYTES;

        workspace->xmma_params = *this;

        return xmma::Error::SUCCESS;
    }

    template <typename Bn_traits>
    xmma::Error set_partial_sums_of_squares_ptr( xmma::Host_workspace<Bn_traits> *workspace ) {
        return xmma::Error::SUCCESS;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
}  // wgrad
}  // bn_apply
}  // batchnorm
}  // ext
}  // xmma
////////////////////////////////////////////////////////////////////////////////////////////////////