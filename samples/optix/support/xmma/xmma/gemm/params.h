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

#include <xmma/integer.h>
#include <xmma/params.h>
#include <xmma/cta_swizzle.h>
//#include <xmma/utils.h>

#include <xmma/ampere/traits.h>
#include <xmma/hopper/traits.h>

namespace xmma {
namespace gemm {

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Traits, typename Cta_tile, int32_t STAGES = 1 >
struct Params : public Gemm_params_base {
    // Precomputed deltas for the matrices.
    int64_t a_delta[1], b_delta[1];

    // The loop count.
    int32_t loop_start;
    // The index of the loop count where we trigger the residue.
    int32_t loop_residue;
    // The number of K elements consumed when we enter the resiude.
    int32_t loop_residue_k;
    // Whether or not we are trying to run Ampere kernels.
    bool ampere;
    // Whether or not we are tyring to run hopper kernels.
    bool hopper;

    // Do we use horizontal rasterization of CTAs?
    int32_t use_horizontal_cta_rasterization;
    // Best group col width(the log to the base 2) for CTA swizzling
    unsigned best_log2_group_cols;
    // The number of CTA tiles in each dimension.
    int32_t tiles_m, tiles_n, tiles_k;
    // The number of CTA tiles in the grid.
    int32_t tiles_x, tiles_y;


    // FIXME: need to initialize default fields properly
    Params() {}

#if !defined(__LWDACC_RTC__)
    XMMA_HOST xmma::Error finalize(const dim3 &grid_) {
        this->ampere = std::is_same<typename Traits::Traits::Gpu_arch, xmma::Ampere>::value;
        this->hopper = std::is_same<typename Traits::Traits::Gpu_arch, xmma::Hopper>::value;

        XMMA_CALL(this->finalize_problem(grid_));
        XMMA_CALL(this->finalize_performance(grid_));

        return xmma::Error::SUCCESS;
    }

    // FIXME: depcreated
    template<typename Kernel_traits>
    XMMA_HOST xmma::Error initialize( const Host_workspace<Kernel_traits> *workspace ) {
        return this->finalize(workspace->grid);
    }

    XMMA_HOST xmma::Error callwlate_grid_dimensions( dim3 &grid, dim3 &split_k_grid, const int32_t xmmas_m ) {
        this->tiles_m = xmma::div_round_up(m, Cta_tile::M);
        this->tiles_n = xmma::div_round_up(n, Cta_tile::N);
        this->tiles_k = this->split_k.slices;
        if( use_horizontal_cta_rasterization ) {
            grid.y = this->tiles_m;
            grid.x = this->tiles_n;
            this->tiles_y = this->tiles_m;
            this->tiles_x = this->tiles_n;
        } else {
            grid.x = this->tiles_m;
            grid.y = this->tiles_n;
            this->tiles_x = this->tiles_m;
            this->tiles_y = this->tiles_n;
        }

        if( this->batch.is_batched ) {
            grid.z = this->batch.batches;
        } else {
            grid.z = this->tiles_k;
        }

        // FIXME: We should treat this as canImplment error and we shouldn't change params underneath
        if( this->use_horizontal_cta_rasterization == 1 && grid.y >= 65536 ) {
            this->use_horizontal_cta_rasterization = 0;
            int32_t tmp = grid.x;
            grid.x = grid.y;
            grid.y = tmp;
        }

        split_k_grid = grid;
        split_k_grid.z = xmmas_m;

        return xmma::Error::SUCCESS;
    }
#endif // __LWDACC_RTC__

protected:
    xmma::Error finalize_problem(const dim3 &grid);
    xmma::Error finalize_performance(const dim3 &grid);
};


////////////////////////////////////////////////////////////////////////////////////////////////////
// BN APPLY PARAMS. Used by proxy bn apply gemm kernel.
template< typename Traits, typename Cta_tile, int32_t STAGES = 1 >
struct Bn_apply_with_scale_bias_relu_params : public Params<Traits, Cta_tile, STAGES> {
    // The scale and bias pointer.
    // Assuming scale and bias can be accessed by the same pointer.
    const void* scale_bias_gmem;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

#if !defined(__LWDACC_RTC__)

template< typename Traits, typename Cta_tile, int32_t STAGES >
xmma::Error Params<Traits, Cta_tile, STAGES>::finalize_problem(const dim3 &grid) {

    // The number of elements from GEMM-k that we consume in one iteration of the loop.
    const int32_t k_per_iteration = this->split_k.slices * Cta_tile::K;

    // The pointer update for A (keep in mind that we transpose A).
    int32_t delta_a;
    if( this->ta ) {
        delta_a = k_per_iteration;
    } else {
        delta_a = k_per_iteration * this->lda;
    }

    // The pointer update for B (keep in mind that we transpose B).
    int32_t delta_b;
    if( this->tb ) {
        delta_b = k_per_iteration * this->ldb;
    } else {
        delta_b = k_per_iteration;
    }

    // Recompute the gmem deltas for interleaved addresses.
    if( this->is_interleaved ) {
        delta_a = delta_a * this->m;
        delta_b = delta_b * this->n;
    }

    // Construct the delta tables.
    this->a_delta[0] = Traits::offset_in_bytes_a( delta_a );
    this->b_delta[0] = Traits::offset_in_bytes_b( delta_b );

    // The number of loop iterations to cover C elements.
    int32_t loop_count_k = xmma::div_round_up( this->k, k_per_iteration );
    // The first iteration of the loop.
    this->loop_start = loop_count_k - 1;
    // The iteration where we trigger the residue.
    this->loop_residue = max(2, STAGES);
    // The number of elements read when we enter the residue.
    this->loop_residue_k = (loop_count_k - 1) * k_per_iteration;

    // Do we have a residual? I.e. do we run with beta != 0.
    this->with_residual = ( this->beta != 0.f );

    // Bias element number.
    // Since we do B^T*A^T=(AB)^T and A will be filter in fullyConnected,
    // it should be n, which is --m in command line.
    if( this->with_bias ) {
        this->with_bias = this->n;
    }

    // Choose best groupCols for CTA swizzling
    xmma::Cta_swizzle::Pos3 grid_dim = xmma::Cta_swizzle::Pos3(
            this->use_horizontal_cta_rasterization ? this->tiles_n : this->tiles_m,
            this->use_horizontal_cta_rasterization ? this->tiles_m : this->tiles_n,
            grid.z );
    xmma::Cta_swizzle::Pos2 cta_tile = xmma::Cta_swizzle::Pos2(
            this->use_horizontal_cta_rasterization ? Cta_tile::N : Cta_tile::M,
            this->use_horizontal_cta_rasterization ? Cta_tile::M : Cta_tile::N );

    // When CGAs are enabled, and size > 1, this rasterization scheme must to be used
    if( this->hopper && ((this->cluster_width * this->cluster_height) > 1) ) {
        xmma::Cta_swizzle::Pos2 cga_tile{
                this->use_horizontal_cta_rasterization ? this->cluster_width : this->cluster_height,
                this->use_horizontal_cta_rasterization ? this->cluster_height : this->cluster_width};

        xmma::Cga_swizzle cs { grid_dim, cta_tile, cga_tile, this->use_horizontal_cta_rasterization };
        this->best_log2_group_cols = cs.choose_best_log2_group_cols( this->ctas_per_wave );
    } else {
        xmma::Cta_swizzle cs =
            xmma::Cta_swizzle( grid_dim, cta_tile, this->use_horizontal_cta_rasterization );
        this->best_log2_group_cols = cs.choose_best_log2_group_cols( this->ctas_per_wave );
    }
 
    return xmma::Error::SUCCESS;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits, typename Cta_tile, int32_t STAGES>
xmma::Error Params<Traits, Cta_tile, STAGES>::finalize_performance(const dim3 &grid) {
    using xmma::integer_cast;

    // Make sure the split-k params are consistent.
    if (this->split_k.slices > 1) {
        this->split_k.buffers = max(this->split_k.buffers, 1);
    }

    // Allocate buffers to do split-k (if needed).
    if (this->split_k.buffers > 0) {
        const int32_t tile_m = Cta_tile::M;
        const int32_t tile_n = Cta_tile::N;

        size_t max_grid = integer_cast<size_t>(max(0, this->tiles_m * this->tiles_n));
        size_t max_data = integer_cast<size_t>(max(0, this->tiles_m * this->tiles_n * tile_m * tile_n));

        // Size to allocate the buffers.
        using Acc_type = typename Traits::Aclwmulator_type;
        this->split_k.buffer_size = integer_cast<int64_t>(sizeof(Acc_type) * max_data);

        // Size to allocate the counters/locks.
        this->split_k.counters_ctas_size = (int)(max_grid * this->split_k.buffers * sizeof(int32_t));
        this->split_k.retired_ctas_size = (int)(max_grid * sizeof(int32_t));
    }

    return xmma::Error::SUCCESS;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

#endif // __LWDACC_RTC__

} // namespace gemm
} // namespace xmma

