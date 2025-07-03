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
#include <xmma/implicit_gemm/dgrad/utils.h>
#include <xmma/implicit_gemm/utils.h>

namespace xmma {
namespace implicit_gemm {
namespace dgrad {

///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits, typename Cta_tile, int32_t STAGES = 1>
struct Params : public Colwolution_params_base {
    // Do we have a residual?
    int32_t with_residual;

    // The knobs to control how we split the filter for split-k.
    int32_t split_k_t, split_k_r, split_k_k;
    // Precomputed values.
    int32_t split_k_trs, split_k_rs;

    // Precomputed values. Reserved!
    int32_t opqk, pqk, qk, ndhw, dhw, hw, trsc, trs, rs;
    // Precomputed values for fast divisions.
    uint32_t mul_dhw, shr_dhw, mul_hw, shr_hw, mul_w, shr_w;
    uint32_t mul_stride[3], shr_stride[3];
    // Precomputed deltas for the image and the filter.
    int64_t a_delta[32], b_delta[32];

    // The size of the filter computed per CTA.
    int32_t filter_trs_per_cta, filter_rs_per_cta;
    uint32_t mask_t, mask_r, mask_s;

    // Filter 1x1x1, no padding, unit stride, no dilation
    bool simple1x1x1;
    // The loop count.
    int32_t loop_start;
    // The index of the loop count where we trigger the residue.
    int32_t loop_residue;
    // The number of elements read before we enter the residue in the GEMM-K dimension.
    int32_t loop_residue_k;
    // Whether or not we are trying to run Ampere kernels.
    bool ampere;
    // Whether or not we are trying to run Hopperkernels.
    bool hopper;

    // Do we use horizontal rasterization of CTAs?
    int32_t use_horizontal_cta_rasterization;
    // Best group col width(the log to the base 2) for CTA swizzling
    unsigned best_log2_group_cols;
    // The number of CTA tiles in each dimension.
    int32_t tiles_m, tiles_n, tiles_k;
    // The number of CTA tiles in the grid.
    int32_t tiles_x, tiles_y;

    // Ctor
    Params() : Colwolution_params_base() {
        memset( a_delta, 0, sizeof( a_delta ) );
        memset( b_delta, 0, sizeof( b_delta ) );
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

    // Finalize params data.
    XMMA_HOST xmma::Error
    finalize( const dim3 &grid_ ) {
        this->ampere = std::is_same<typename Traits::Traits::Gpu_arch, xmma::Ampere>::value;

        XMMA_CALL( this->finalize_performance( grid_ ) );
        XMMA_CALL( this->finalize_problem( grid_ ) );

        return xmma::Error::SUCCESS;
    }

    // Callwlate grid and split_k grid dimensions.
    XMMA_HOST xmma::Error
    callwlate_grid_dimensions( dim3 &grid, dim3 &split_k_grid, const int32_t xmmas_m ) {
        this->use_horizontal_cta_rasterization = 1;
        this->tiles_m = xmma::div_up( this->n * this->d * this->h * this->w, Cta_tile::M );
        this->tiles_n = xmma::div_up( this->c, Cta_tile::N / Cta_tile::GROUPS ) *
                        xmma::div_up( this->g, Cta_tile::GROUPS );
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
    this->with_residual = this->with_residual || (xmma::colwert<float>( this->beta ) == 0.f ? 0 : 1);

    // Compute precomputed values.
    this->opqk = this->o * this->p * this->q * this->g * this->k;
    this->pqk = this->p * this->q * this->g * this->k;
    this->qk = this->q * this->g * this->k;
    this->ndhw = this->n * this->d * this->h * this->w;
    this->dhw = this->d * this->h * this->w;
    this->hw = this->h * this->w;
    this->trsc = this->t * this->r * this->s * this->c;
    this->trs = this->t * this->r * this->s;
    this->rs = this->r * this->s;

    // The fast division params.
    xmma::find_divisor( this->mul_dhw, this->shr_dhw, this->dhw );
    xmma::find_divisor( this->mul_hw, this->shr_hw, this->hw );
    xmma::find_divisor( this->mul_w, this->shr_w, this->w );

    xmma::find_divisor_v2( this->mul_stride[0], this->shr_stride[0], this->stride[0] );
    xmma::find_divisor_v2( this->mul_stride[1], this->shr_stride[1], this->stride[1] );
    xmma::find_divisor_v2( this->mul_stride[2], this->shr_stride[2], this->stride[2] );

    this->filter_t_per_cta = ( this->split_k_t == 1 ? 1 : this->t );
    this->filter_r_per_cta = ( this->split_k_r == 1 ? 1 : this->r );
    this->filter_s_per_cta = this->s;
    this->filter_trs_per_cta =
        this->filter_t_per_cta * this->filter_r_per_cta * this->filter_s_per_cta;
    this->filter_rs_per_cta = this->filter_r_per_cta * this->filter_s_per_cta;
    this->simple1x1x1 = ( this->filter_trs_per_cta == 1 && this->pad[0][0] == 0 &&
                          this->pad[1][0] == 0 && this->pad[2][0] == 0 && this->pad[0][1] == 0 &&
                          this->pad[1][1] == 0 && this->pad[2][1] == 0 && this->stride[0] == 1 &&
                          this->stride[1] == 1 && this->stride[2] == 1 && this->dilation[0] == 1 &&
                          this->dilation[1] == 1 && this->dilation[2] == 1 );

    // Set masks.
    xmma::implicit_gemm::build_masks( this->mask_t,
                                      this->mask_r,
                                      this->mask_s,
                                      this->filter_t_per_cta,
                                      this->filter_r_per_cta,
                                      this->filter_s_per_cta );

    // The update in the C dimension.
    int32_t move_k = Cta_tile::K;
    if( this->split_k.slices > 1 && this->split_k_k > 0 ) {
        move_k *= this->split_k.slices;
    }

    int32_t delta = 0, total_delta = 0;
    // The deltas for the image.
    for( int32_t ii = 0; ii < this->filter_trs_per_cta - 1; ++ii ) {
        int32_t rsi = ( ii + 1 ) % this->filter_rs_per_cta;
        int32_t ti = ( ii + 1 ) / this->filter_rs_per_cta;
        int32_t ri = rsi / this->filter_s_per_cta;
        int32_t si = rsi % this->filter_s_per_cta;
        delta = 0 - this->out_stride_d * ( ti * this->dilation[0] / this->stride[0] ) -
                this->out_stride_h * ( ri * this->dilation[1] / this->stride[1] ) -
                this->out_stride_w * ( si * this->dilation[2] / this->stride[2] ) - total_delta;
        this->a_delta[ii] = Traits::offset_in_bytes_a( delta );
        total_delta += delta;
    }
    this->a_delta[this->filter_trs_per_cta - 1] = Traits::offset_in_bytes_a(
        move_k * static_cast<int64_t>( this->out_stride_c ) - total_delta );

    // The update in the K dimension.
    int32_t move_flt_k = Cta_tile::K * this->trs * this->c;
    if( this->split_k.slices > 1 && this->split_k_k > 0 ) {
        move_flt_k *= this->split_k.slices;
    }

    // The deltas for the filter.
    int32_t b_delta = this->c;
    if( !this->cross_correlation ) {
        b_delta = -this->c;
    }
    for( int32_t ii = 0; ii < this->filter_trs_per_cta - 1; ++ii ) {
        this->b_delta[ii] = Traits::offset_in_bytes_b( b_delta );
    }
    b_delta = -( this->filter_trs_per_cta - 1 ) * this->c;
    if( !this->cross_correlation ) {
        b_delta = ( this->filter_trs_per_cta - 1 ) * this->c;
    }
    this->b_delta[this->filter_trs_per_cta - 1] = Traits::offset_in_bytes_b( move_flt_k + b_delta );

    // The number of elements in the C dimension that are used per iteration.
    int32_t k_per_iteration = Cta_tile::K;
    if( this->split_k.slices > 1 && this->split_k_k > 0 ) {
        k_per_iteration *= this->split_k.slices;
    }

    // The number of loop iterations to cover C elements.
    int32_t loop_count_k = xmma::div_up( Cta_tile::GROUPS * this->k, k_per_iteration );
    // The first iteration of the loop.
    this->loop_start = this->filter_trs_per_cta * loop_count_k - 1;
    // The iteration where we trigger the residue.
    this->loop_residue = this->filter_trs_per_cta + max( 1, STAGES - 1 );
    // The number of elements read when we enter the residue.
    this->loop_residue_k = ( loop_count_k - 1 ) * k_per_iteration;

    // Bias element number
    if( this->with_bias ) {
        this->with_bias = this->g * this->c;
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
    xmma::Cta_swizzle::Pos2 output = xmma::Cta_swizzle::Pos2( this->h, this->w );
    xmma::Cta_swizzle cs = xmma::Cta_swizzle(
        grid_dim, cta_tile, filter, colw_stride, output, this->use_horizontal_cta_rasterization );
    this->best_log2_group_cols = cs.choose_best_log2_group_cols( this->ctas_per_wave );

    return xmma::Error::SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits, typename Cta_tile, int32_t STAGES>
xmma::Error Params<Traits, Cta_tile, STAGES>::finalize_performance( const dim3 &grid ) {
    this->split_k_k = ( this->split_k.slices > 1 && !this->split_k_t && !this->split_k_r );

    this->split_k_trs = this->split_k_t * this->r * this->s;
    this->split_k_rs = this->split_k_r * this->s;

    // Make sure that if we do split-k in the C dimension, we use Cta_tile::K.
    this->split_k_k *= Cta_tile::K;

    // Make sure the split-k params are consistent.
    if( this->split_k.slices > 1 ) {
        this->split_k.buffers = max( this->split_k.buffers, 1 );
    }

    // Allocate buffers to do split-k (if needed).
    if( this->split_k.buffers > 0 ) {
        size_t max_grid = 0, max_data = 0;

        const int32_t tile_m = Cta_tile::M;
        const int32_t tile_n = Cta_tile::N;
        const int32_t tile_g = Cta_tile::GROUPS;

        max_grid = max( max_grid, (size_t)this->tiles_m * this->tiles_n );
        max_data = max( max_data, (size_t)this->tiles_m * this->tiles_n * tile_m * tile_n );

        // Size to allocate the buffers.
        using Acc_type = typename Traits::Aclwmulator_type;
        this->split_k.buffer_size = ( int64_t )( sizeof( Acc_type ) * (int)max_data );

        // Size to allocate the counters/locks.
        this->split_k.counters_ctas_size =
            (int)( max_grid * this->split_k.buffers * sizeof( int32_t ) );
        this->split_k.retired_ctas_size = (int)( max_grid * sizeof( int32_t ) );
    }

    return xmma::Error::SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace dgrad
}  // namespace implicit_gemm
}  // namespace xmma
