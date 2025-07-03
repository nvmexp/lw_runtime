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

#include <xmma/implicit_gemm/dgrad/params.h>

////////////////////////////////////////////////////////////////////////////////////////////////////
namespace xmma {
namespace ext {
namespace batchnorm {
namespace bn_apply {
namespace dgrad {
////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits, typename Cta_tile, int32_t STAGES>
struct Bn_apply_with_scale_bias_relu_dgrad_params
    : public xmma::implicit_gemm::dgrad::Params<Traits, Cta_tile, STAGES> {

    // Base class
    using Base = xmma::implicit_gemm::dgrad::Params<Traits, Cta_tile, STAGES>;

    // In a sequence for fprop like bn1 + relu1 + colw + bn2
    // This kernel models dbn2(a) + dgrad + dRelu1 + dbn1(s)
    // dbn2(a) is computed using input of bn2 dusing fprop and input of dbn2 during dgrad
    // Input of fprop bn2
    const void *bna_fprop_tensor_gmem;
    const void *bna_fprop_tensor_scale_gmem;
    const void *bna_gradient_scale_gmem;
    const void *bna_bias_gmem;

    // Input of fprop bn1
    const void *bn_fprop_tensor_gmem;
    const void *bn_fprop_mean_gmem;
    const void *bn_fprop_ilw_stddev_gmem;
    const void *bn_fprop_alpha_gmem;

    // dual dbn(s)
    bool dual_dbns;
    const void *dual_bn_fprop_mean_gmem;
    const void *dual_bn_fprop_tensor_gmem;
    const void *dual_bn_fprop_ilw_stddev_gmem;
    const void *dual_bn_fprop_alpha_gmem;

    // Do dgrad + add
    int32_t with_residual;

    // dReLu bitmask
    // Only used when with_residual is set
    const void *bn_drelu_bitmask;

    // Standalone dbn(a)
    bool standalone_dbna;
    bool dual_standalone_dbna;
    // if  overwrite_out_gmem = true then we overwrite out_gmem
    // else we write to standalone_dbna_output
    bool overwrite_out_gmem;
    void *standalone_dbna_output;
    void *dual_standalone_dbna_output;

    // The scale.
    typename Traits::Epilogue_type alpha;
    typename Traits::Epilogue_type beta;

    // Do we finalize the BN stats?
    int32_t bn_finalize_stats;

    // The buffer to store partial sums.
    void *bn_partial_sums_gmem, *bn_partial_sums_of_squares_gmem;
    void *bn_partial_dual_sums_of_squares_gmem;

    // The mean and the ilw-stddev computed by the kernel.
    void *bn_sum_gmem, *bn_mean_gmem, *bn_sum_of_squares_gmem, *bn_ilw_stddev_gmem;
    void *dual_bn_sum_gmem, *dual_bn_mean_gmem, *dual_bn_sum_of_squares_gmem,
        *dual_bn_ilw_stddev_gmem;

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

    // Ctor
    Bn_apply_with_scale_bias_relu_dgrad_params() : Base() {
    }

    // Finalize params data.
    XMMA_HOST xmma::Error
    finalize( const dim3 &grid_, int32_t split_k_t = 0, int32_t split_k_r = 0 ) {
        this->split_k_t = split_k_t;
        this->split_k_r = split_k_r;

        this->ampere = std::is_same<typename Traits::Traits::Gpu_arch, xmma::Ampere>::value;
        this->hopper = std::is_same<typename Traits::Traits::Gpu_arch, xmma::Hopper>::value;

        XMMA_CALL( this->finalize_performance( grid_ ) );
        XMMA_CALL( this->finalize_problem( grid_ ) );

        // Also initialize some BN params

        // Determine the size of the biggest "output matrix"/grid we compute.
        size_t max_partial_sums = 0;
        if( this->use_horizontal_cta_rasterization && ( grid_.y > max_partial_sums ) ) {
            max_partial_sums = grid_.y;
        } else if( !this->use_horizontal_cta_rasterization && ( grid_.x > max_partial_sums ) ) {
            max_partial_sums = grid_.x;
        }

        if( this->dual_dbns ) {
            // *4 because we also need to allocated for sum of squares and sums of dual
            this->bn_partial_sums_sz = 4 * max_partial_sums * this->g * this->c * sizeof( float );
        } else {
            // *2 because we also need to allocated for sum of squares
            this->bn_partial_sums_sz = 2 * max_partial_sums * this->g * this->c * sizeof( float );
        }

        // TODO : Find a cleaner fix, but for now we can init partial sums later
        const int32_t partial_sums = this->use_horizontal_cta_rasterization ? grid_.y : grid_.x;

        // remaining parameters
        this->num_channels = this->g * this->c;
        this->num_partial_sums = partial_sums;
        this->ilw_count = 1.f / (float)this->ndhw;

        return xmma::Error::SUCCESS;
    }

    // TODO:deprecate in future.
    template <typename Implicit_gemm_traits>
    xmma::Error initialize( xmma::Host_workspace<Implicit_gemm_traits> *workspace ) {

        int32_t xmmas_m = Implicit_gemm_traits::Gmem_tile_epilogue::Layout::ROW
                              ? Implicit_gemm_traits::Xmma_tile::XMMAS_M
                              : Implicit_gemm_traits::Xmma_tile::XMMAS_N;

        XMMA_CALL(
            this->callwlate_grid_dimensions( workspace->grid, workspace->split_k_grid, xmmas_m ) );

        XMMA_CALL( this->finalize( workspace->grid ) );
        // Initialize workspace

        // Do we need a sequential reduction?
        workspace->split_k_with_reduction = this->split_k.with_reduction();
        workspace->device_workspace_size = this->split_k.size_in_bytes();
        workspace->smem_size = Implicit_gemm_traits::dynamic_smem_size_per_cta();
        const int32_t EPILOGUE_SIZE_IN_BYTES =
            Implicit_gemm_traits::Swizzle_epilogue::BYTES_PER_TILE;
        workspace->epilogue_size_in_bytes = EPILOGUE_SIZE_IN_BYTES;

        workspace->xmma_params = *this;

        return xmma::Error::SUCCESS;
    }

    template <typename Bn_traits>
    xmma::Error set_partial_sums_of_squares_ptr( xmma::Host_workspace<Bn_traits> *workspace ) {
        const int32_t partial_sums =
            this->use_horizontal_cta_rasterization ? workspace->grid.y : workspace->grid.x;
        float *partial_sums_gmem = (float *)bn_partial_sums_gmem;

        float *partial_sums_of_squares_gmem = partial_sums_gmem + partial_sums * this->g * this->c;
        this->bn_partial_sums_of_squares_gmem = partial_sums_of_squares_gmem;
        if( this->dual_dbns ) {
            // 3 because the layout is sum, sum^2, dual_sum (which is same as sum), dual_sum^2
            float *partial_dual_sums_of_squares_gmem =
                partial_sums_gmem + 3 * partial_sums * this->g * this->c;
            this->bn_partial_dual_sums_of_squares_gmem = partial_dual_sums_of_squares_gmem;
        }

        return xmma::Error::SUCCESS;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
}  // namespace dgrad
}  // namespace bn_apply
}  // namespace batchnorm
}  // namespace ext
}  // namespace xmma
////////////////////////////////////////////////////////////////////////////////////////////////////
