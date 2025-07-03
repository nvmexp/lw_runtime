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

#include <xmma/params.h>
#include <xmma/cta_swizzle.h>
#include <xmma/implicit_gemm/fprop/utils.h>

#include <xmma/ampere/traits.h>
#include <xmma/hopper/traits.h>

namespace xmma {
namespace ext {
namespace implicit_gemm {
namespace indexed_wo_smem {
namespace fprop {

///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits, typename Cta_tile, int STAGES = 1>
struct Params : public xmma::Colwolution_params_base {
    // Do we have a residual?
    int with_residual;

    // The knobs to control how we split the filter for split-k.
    int split_k_t, split_k_r, split_k_c;
    // Precomputed values.
    int split_k_trs, split_k_rs;

    // Precomputed values. Reserved!
    int dhwc, hwc, wc, nopq, opq, pq, trsc, trs, rs;
    // Precomputed values for fast divisions.
    uint32_t mul_opq, shr_opq, mul_pq, shr_pq, mul_q, shr_q;
    uint32_t mul_trs, shr_trs, mul_rs, shr_rs, mul_s, shr_s;
    uint32_t mul_k, shr_k, kn;

    // Precomputed values for fast divisions for the kernel without L1 replications.
    uint32_t ctas_pq, mul_ctas_pq, shr_ctas_pq, ctas_q, mul_ctas_q, shr_ctas_q;
    // Precomputed deltas for the image and the filter.
    int64_t a_delta[32], b_delta[32];

    int filter_trs_per_cta, filter_rs_per_cta;
    uint32_t mask_t, mask_r, mask_s;

    // Filter 1x1x1, no padding, unit stride, no dilation
    bool simple1x1x1;
    // The loop count.
    int loop_start;
    // The index of the loop count where we trigger the residue.
    int loop_residue;
    // The number of elements read before we enter the residue in the GEMM-K dimension.
    int loop_residue_k;
    // Whether or not we are trying to run Ampere kernels.
    bool ampere;
    // Whether or not we are trying to run Hopperkernels.
    bool hopper;

    int pool_factor;

    // Do we use horizontal rasterization of CTAs?
    int use_horizontal_cta_rasterization;
    // Best group col width(the log to the base 2) for CTA swizzling
    unsigned best_log2_group_cols;
    // Num of CTAs per wave, if > 0 then use CTA swizzling
    unsigned ctas_per_wave;
    // The number of CTA tiles in each dimension.
    int tiles_m, tiles_n, tiles_k;
    // The number of CTA tiles in the grid.
    int tiles_x, tiles_y;

    // Initialize params from base params
    template <typename Implicit_gemm_traits>
    xmma::Error
    initialize( xmma::Host_workspace<Implicit_gemm_traits>* workspace ) {

        int32_t xmmas_m = Implicit_gemm_traits::Xmma_tile::XMMAS_M;

        XMMA_CALL(
            this->callwlate_grid_dimensions( workspace->grid, workspace->split_k_grid, xmmas_m ) );

        XMMA_CALL( this->finalize( workspace->grid ) );

        // Initialize workspace

        // Do we need a sequential reduction?
        workspace->split_k_with_reduction = this->split_k.with_reduction();
        workspace->device_workspace_size = this->split_k.size_in_bytes();
        workspace->smem_size = Implicit_gemm_traits::dynamic_smem_size_per_cta();
        const int EPILOGUE_SIZE_IN_BYTES = 
            Implicit_gemm_traits::Swizzle_epilogue::BYTES_PER_TILE;
        workspace->epilogue_size_in_bytes = EPILOGUE_SIZE_IN_BYTES;

        workspace->xmma_params = *this;

        return xmma::Error::SUCCESS;
    }

    // Finalize params data.
    XMMA_HOST xmma::Error
    finalize( const dim3 &grid_ ) {
        this->ampere = std::is_same<typename Traits::Traits::Gpu_arch, xmma::Ampere>::value;
        this->hopper = std::is_same<typename Traits::Traits::Gpu_arch, xmma::Hopper>::value;

        XMMA_CALL( this->finalize_performance( grid_ ) );
        XMMA_CALL( this->finalize_problem( grid_ ) );

        return xmma::Error::SUCCESS;
    }

    // Callwlate grid and split_k grid dimensions.
    XMMA_HOST xmma::Error callwlate_grid_dimensions( dim3 &grid,
                                                     dim3 &split_k_grid,
                                                     const int32_t xmmas_m,
                                                     bool without_l1_replication = false ) {
    
        this->use_horizontal_cta_rasterization = 1;

        if( without_l1_replication ) {
            const int32_t n_per_cta = 1;
            const int32_t p_per_cta = Cta_tile::M == 256 ? 16 : 8;
            const int32_t q_per_cta = Cta_tile::M == 64 ? 8 : 16;

            const int32_t cta_n = xmma::div_up( this->n, n_per_cta );
            const int32_t cta_p = xmma::div_up( this->p, p_per_cta );
            const int32_t cta_q = xmma::div_up( this->q, q_per_cta );

            this->tiles_m = cta_n * cta_p * cta_q;
            this->tiles_n = xmma::div_up( this->k, Cta_tile::N ) * this->g;
        } else {
            this->tiles_m = xmma::div_up( this->n * this->o * this->p * this->q, Cta_tile::M );
            this->tiles_n = xmma::div_up( this->k, Cta_tile::N )* this->g;
        }
        this->tiles_k = this->split_k.slices;

        if( this->use_horizontal_cta_rasterization ) {
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
        grid.z = this->tiles_k;

        // Hardware limitation
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
  protected:
    // Finalize problem related params.
    xmma::Error finalize_problem( const dim3 &grid );
    // Finalize performance related params.
    xmma::Error finalize_performance( const dim3 &grid );
};

///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits, typename Cta_tile, int32_t STAGES>
xmma::Error Params<Traits, Cta_tile, STAGES>::finalize_problem( const dim3 &grid ) {

    this->with_residual = xmma::colwert<float>( this->beta ) == 0.f ? 0 : 1;
    this->pool_factor = 1;

    // Compute precomputed values.
    this->dhwc = this->d * this->h * this->w * this->g * this->c;
    this->hwc = this->h * this->w * this->g * this->c;
    this->wc = this->w * this->g * this->c;
    this->nopq = this->n * this->o * this->p * this->q;
    this->opq = this->o * this->p * this->q;
    this->pq = this->p * this->q;
    this->trsc = this->t * this->r * this->s * this->c;
    this->trs = this->r * this->s * this->c;
    this->rs = this->s * this->c;

    // The fast division params.
    xmma::find_divisor( this->mul_opq, this->shr_opq, this->opq );
    xmma::find_divisor( this->mul_pq, this->shr_pq, this->pq );
    xmma::find_divisor( this->mul_q, this->shr_q, this->q );
    xmma::find_divisor( this->mul_trs, this->shr_trs, this->trs );
    xmma::find_divisor( this->mul_rs, this->shr_rs, this->rs );
    xmma::find_divisor( this->mul_s, this->shr_s, this->c );

    this->kn = xmma::div_up(this->k, Cta_tile::N);
    xmma::find_divisor( this->mul_k, this->shr_k, this->kn );

    this->filter_t_per_cta = ( this->split_k_t == 1 ? 1 : this->t );
    this->filter_r_per_cta = ( this->split_k_r == 1 ? 1 : this->r );
    this->filter_s_per_cta = this->s;
    this->filter_trs_per_cta =
        this->filter_t_per_cta * this->filter_r_per_cta * this->filter_s_per_cta;
    this->filter_rs_per_cta = this->filter_r_per_cta * this->filter_s_per_cta;

    // The number of elements in the C dimension that are used per iteration.
    int c_per_iteration = Cta_tile::K;

    // The number of loop iterations to cover C elements.
    int loop_count_k = xmma::div_up(
        Cta_tile::GROUPS * this->c * this->filter_trs_per_cta, c_per_iteration);
    // The first iteration of the loop.
    this->loop_start = loop_count_k - 1;
    // The iteration where we trigger the residue.
    this->loop_residue = 1;
    // The number of elements read when we enter the residue.
    this->loop_residue_k = ( loop_count_k - 1 ) * c_per_iteration;

    // Bias element number
    if( this->with_bias ) {
        this->with_bias = this->g * this->k;
    }

    // Choose best groupCols for CTA swizzling
    xmma::Cta_swizzle::Pos3 grid_dim = xmma::Cta_swizzle::Pos3(
        this->use_horizontal_cta_rasterization ? this->tiles_n : this->tiles_m,
        this->use_horizontal_cta_rasterization ? this->tiles_m : this->tiles_n,
        grid.z );
    xmma::Cta_swizzle::Pos2 cta_tile = xmma::Cta_swizzle::Pos2(
        this->use_horizontal_cta_rasterization ? Cta_tile::N : Cta_tile::M,
        this->use_horizontal_cta_rasterization ? Cta_tile::M : Cta_tile::N );
    xmma::Cta_swizzle::Pos2 filter = xmma::Cta_swizzle::Pos2( this->r, this->s );
    xmma::Cta_swizzle::Pos2 colw_stride =
        xmma::Cta_swizzle::Pos2( this->stride[1], this->stride[2] );
    xmma::Cta_swizzle::Pos2 output = xmma::Cta_swizzle::Pos2( this->p, this->q );
    xmma::Cta_swizzle cs = xmma::Cta_swizzle(
        grid_dim, cta_tile, filter, colw_stride, output, this->use_horizontal_cta_rasterization );
    this->best_log2_group_cols = cs.choose_best_log2_group_cols( this->ctas_per_wave );

    return xmma::Error::SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits, typename Cta_tile, int32_t STAGES>
xmma::Error Params<Traits, Cta_tile, STAGES>::finalize_performance( const dim3 &grid ) {

    this->split_k_c = ( this->split_k.slices > 1 && !this->split_k_t && !this->split_k_r );

    this->split_k_trs = this->split_k_t * this->r * this->s;
    this->split_k_rs = this->split_k_r * this->s;

    // Make sure that if we do split-k in the C dimension, we use Cta_tile::K.
    this->split_k_c *= Cta_tile::K;

    // Make sure the split-k params are consistent.
    if( this->split_k.slices > 1 ) {
        this->split_k.buffers = max( this->split_k.buffers, 1 );
    }

    // Allocate buffers to do split-k (if needed).
    if( this->split_k.buffers > 0 ) {
        size_t max_grid = 0, max_data = 0;

        const int32_t tile_m = Cta_tile::M;
        const int32_t tile_n = Cta_tile::N;

        max_grid = max( max_grid, (size_t)this->tiles_m * this->tiles_n );
        max_data = max( max_data, (size_t)this->tiles_m * this->tiles_n * tile_m * tile_n );

        // Size to allocate the buffers.
        using Acc_type = typename Traits::Aclwmulator_type;
        this->split_k.buffer_size = ( int64_t )( sizeof( Acc_type ) * (int32_t)max_data );

        // Size to allocate the counters/locks.
        this->split_k.counters_ctas_size =
            (int32_t)( max_grid * this->split_k.buffers * sizeof( int32_t ) );
        this->split_k.retired_ctas_size = (int32_t)( max_grid * sizeof( int32_t ) );
    }

    return xmma::Error::SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace fprop
}  // namespace indexed_wo_smem
}  // namespace implicit_gemm
}  // namespace ext
} // namespace xmma
