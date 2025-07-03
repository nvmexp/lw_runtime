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

#include <vector>
#include <xmma/params.h>
#include <xmma/cta_swizzle.h>

#include <xmma/traits.h>
#include <xmma/ampere/traits.h>
#include <xmma/implicit_gemm/strided_dgrad_indexed/utils.h>

namespace xmma {
namespace implicit_gemm {
namespace strided_dgrad_indexed {

///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits, typename Cta_tile, int32_t STAGES = 1>
struct Params : public xmma::Colwolution_params_base {

    // Do we have a residual?
    int32_t with_residual;

    // The knobs to control how we split the filter for split-k.
    int32_t split_k_k;
    int32_t split_k_t;
    int32_t split_k_r;

    // Precomputed values. Reserved!
    int32_t opqk, pqk, qk, ndhw, dhw, hw, trsc, trs, rs, sc, rsc;
    // Precomputed values for fast divisions.
    uint32_t mul_rs, shr_rs, mul_s, shr_s;
    uint32_t mul_dhw, shr_dhw, mul_hw, shr_hw, mul_w, shr_w;
    uint32_t mul_stride_d, mul_stride_h, mul_stride_w;
    uint32_t shr_stride_d, shr_stride_h, shr_stride_w;
    int32_t step_o, step_p, step_q;
    int32_t step_t, step_r, step_s;
    uint32_t mul_step_t, mul_step_r, mul_step_s;
    uint32_t shr_step_t, shr_step_r, shr_step_s;

    // Store the ndhw indices of each filter pattern in global memory.
    int *ndhw_indices_of_each_filter_pattern_gmem;

    // The total nhdw indices, which is large than or equal to ndhw.
    int32_t sum_of_round_up_ndhw_number_of_each_filter_pattern;

    int *valid_t;
    int *valid_r;
    int *valid_s;

    // Size should be equal to txrxs+1.
    // Use this to tell a cta which filter pattern it is belong to.
    int *start_cta_id_in_ndhw_dimension_of_each_filter_pattern;
    int *dhw_count_of_each_filter_pattern;

    // The number of loop iterations to cover K elements.
    int32_t loop_count_k;
    // The number of K elements consumed when we enter the resiude.
    int32_t loop_residue_k;
    // Whether or not we are trying to run Ampere kernels.
    bool ampere;

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
        this->valid_t = nullptr;
        this->valid_r = nullptr;
        this->valid_s = nullptr;
        this->ndhw_indices_of_each_filter_pattern_gmem = nullptr;
        this->start_cta_id_in_ndhw_dimension_of_each_filter_pattern = nullptr;
        this->dhw_count_of_each_filter_pattern = nullptr;
    }

    // TODO:deprecate in future.
    template <typename Implicit_gemm_traits>
    xmma::Error initialize( xmma::Host_workspace<Implicit_gemm_traits> *workspace ) {

        // Prepare the filter pattern in 3 dimensions.
        std::vector<int> valid_count_d( this->t + 1, 0 );
        std::vector<int> valid_count_h( this->r + 1, 0 );
        std::vector<int> valid_count_w( this->s + 1, 0 );

        for( int32_t di = 0; di < this->d; ++di ) {
            int32_t valid_tmp = -1;
            for( int32_t ti = this->t - 1; ti >= 0; --ti ) {
                int32_t oi = di + this->pad[0][0] - ti * this->dilation[0];
                if( oi >= 0 && oi % this->stride[0] == 0 && oi / this->stride[0] < this->o ) {
                    valid_tmp = ti;
                    break;
                }
            }
            valid_count_d[valid_tmp + 1] += 1;
        }

        for( int32_t hi = 0; hi < this->h; ++hi ) {
            int32_t valid_tmp = -1;
            for( int32_t ri = this->r - 1; ri >= 0; --ri ) {
                int32_t pi = hi + this->pad[1][0] - ri * this->dilation[1];
                if( pi >= 0 && pi % this->stride[1] == 0 && pi / this->stride[1] < this->p ) {
                    valid_tmp = ri;
                    break;
                }
            }
            valid_count_h[valid_tmp + 1] += 1;
        }

        for( int32_t wi = 0; wi < this->w; ++wi ) {
            int32_t valid_tmp = -1;
            for( int32_t si = this->s - 1; si >= 0; --si ) {
                int32_t qi = wi + this->pad[2][0] - si * this->dilation[2];
                if( qi >= 0 && qi % this->stride[2] == 0 && qi / this->stride[2] < this->q ) {
                    valid_tmp = si;
                    break;
                }
            }
            valid_count_w[valid_tmp + 1] += 1;
        }

        // Prepare the filter pattern for all dhw elements.
        this->trs = this->t * this->r * this->s;
        this->rs = this->r * this->s;
        std::vector<int> dhw_indices( this->trs + 1 );
        for( int32_t i = 0; i < this->trs; ++i ) {
            int32_t expect_trs = this->trs - 1 - i;
            int32_t expect_t = expect_trs / this->rs;
            int32_t expect_rs = expect_trs % this->rs;
            int32_t expect_r = expect_rs / this->s;
            int32_t expect_s = expect_rs % this->s;
            ++expect_t;
            ++expect_r;
            ++expect_s;
            dhw_indices[i] =
                valid_count_d[expect_t] * valid_count_h[expect_r] * valid_count_w[expect_s];
        }
        dhw_indices[this->trs] = this->d * this->h * this->w -
                                 ( this->d - valid_count_d[0] ) * ( this->h - valid_count_h[0] ) *
                                     ( this->w - valid_count_w[0] );

        // Get the buffer size for the current configuration.
        const int32_t DHW_PER_CTA = Cta_tile::M;
        this->sum_of_round_up_ndhw_number_of_each_filter_pattern = 0;
        for( int32_t i = 0; i <= this->trs; ++i ) {
            if( dhw_indices[i] > 0 ) {
                this->sum_of_round_up_ndhw_number_of_each_filter_pattern +=
                    xmma::div_round_up( this->n * dhw_indices[i], DHW_PER_CTA );
            }
        }
        this->sum_of_round_up_ndhw_number_of_each_filter_pattern *= DHW_PER_CTA;

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
        this->tiles_m =
            xmma::div_up( this->sum_of_round_up_ndhw_number_of_each_filter_pattern, Cta_tile::M );
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

    this->with_residual = xmma::colwert<float>( this->beta ) == 0.f ? 0 : 1;

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
    this->rsc = this->r * this->s * this->c;
    this->sc = this->s * this->c;

    // The fast division params.
    xmma::find_divisor( this->mul_dhw, this->shr_dhw, this->dhw );
    xmma::find_divisor( this->mul_hw, this->shr_hw, this->hw );
    xmma::find_divisor( this->mul_w, this->shr_w, this->w );
    xmma::find_divisor( this->mul_rs, this->shr_rs, this->rs );
    xmma::find_divisor( this->mul_s, this->shr_s, this->s );
    xmma::find_divisor( this->mul_stride_d, this->shr_stride_d, this->stride[0] );
    xmma::find_divisor( this->mul_stride_h, this->shr_stride_h, this->stride[1] );
    xmma::find_divisor( this->mul_stride_w, this->shr_stride_w, this->stride[2] );

    this->step_o = lcm( this->stride[0], this->dilation[0] ) / this->stride[0];
    this->step_t = lcm( this->stride[0], this->dilation[0] ) / this->dilation[0];
    this->step_p = lcm( this->stride[1], this->dilation[1] ) / this->stride[1];
    this->step_r = lcm( this->stride[1], this->dilation[1] ) / this->dilation[1];
    this->step_q = lcm( this->stride[2], this->dilation[2] ) / this->stride[2];
    this->step_s = lcm( this->stride[2], this->dilation[2] ) / this->dilation[2];
    xmma::find_divisor( this->mul_step_t, this->shr_step_t, this->step_t );
    xmma::find_divisor( this->mul_step_r, this->shr_step_r, this->step_r );
    xmma::find_divisor( this->mul_step_s, this->shr_step_s, this->step_s );

    // The number of elements in the C dimension that are used per iteration.
    int32_t k_per_iteration = Cta_tile::K;
    if( this->split_k.slices > 1 && this->split_k_k > 0 ) {
        k_per_iteration *= this->split_k.slices;
    }

    // The number of loop iterations to cover C elements.
    this->loop_count_k = xmma::div_up( Cta_tile::GROUPS * this->k, k_per_iteration );
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
    this->split_k_k = ( this->split_k.slices > 1 );

    // Make sure that if we do split-k in the C dimension, we use Cta_tile::K.
    this->split_k_k *= Cta_tile::K;

    // Make sure the split-k params are consistent.
    if( this->split_k.slices > 1 ) {
        this->split_k.buffers = max( this->split_k.buffers, 1 );
    }

    // Allocate buffers to do split-k (if needed).
    if( this->split_k.buffers > 0 ) {
        size_t grid_size = 0, data_size = 0;

        const int32_t tile_m = Cta_tile::M;
        const int32_t tile_n = Cta_tile::N;
        const int32_t tile_g = Cta_tile::GROUPS;

        grid_size = (size_t)grid.x * grid.y;
        data_size = (size_t)grid.x * grid.y * tile_m * tile_n;

        // Size to allocate the buffers.
        using Acc_type = typename Traits::Aclwmulator_type;
        this->split_k.buffer_size = ( int64_t )( sizeof( Acc_type ) * (int32_t)data_size );

        // Size to allocate the counters/locks.
        this->split_k.counters_ctas_size =
            ( int32_t )( grid_size * this->split_k.buffers * sizeof( int32_t ) );
        this->split_k.retired_ctas_size = ( int32_t )( grid_size * sizeof( int32_t ) );
    }

    return xmma::Error::SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace dgrad_indexed
}  // namespace implicit_gemm
}  // namespace xmma
