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
#include <xmma/params.h>
#include <xmma/cta_swizzle.h>
#include <xmma/implicit_gemm/strided_dgrad_indexed/warp_specialized_utils.h>
#include <xmma/implicit_gemm/strided_dgrad_indexed/params.h>

namespace xmma {
namespace implicit_gemm {
namespace strided_dgrad_indexed {

///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits, typename Cta_tile>  //, int STAGES = 1 >
struct Warp_specialized_params
    : public xmma::implicit_gemm::strided_dgrad_indexed::Params<Traits, Cta_tile, 1> {
    // warp specialized parameters
    // which warp specialization mode
    int specialize;
    // used in ping-pong mode
    int buffers_img, buffers_flt, buffers_epilog;
    // int delta_flt_head, delta_img_head;
    // steps for tile move
    int tile_move_step;
    // sm id used for shared memory capacity.
    int sm;
    // The number of CTA tiles in each dimension.
    int tiles_mn, tiles_all;
    // precomputed values for fast_divmod
    uint32_t mul_grid_yx, shr_grid_yx, mul_grid_x, shr_grid_x;

    // Initialize params from base params
    template <typename Implicit_gemm_traits>
    xmma::Error initialize( xmma::Host_workspace<Implicit_gemm_traits>* workspace ) {
        // Init warp specialization related params.
        this->specialize = Implicit_gemm_traits::WARP_SPECIALIZED_CONFIG;
        // Device info.
        lwdaDeviceProp props;
        int dev = 0;
        XMMA_LWDA_CALL( lwdaGetDeviceProperties( &props, dev ) );
        // persistent CTA:  1CTA/SM.
        this->tile_move_step = props.multiProcessorCount;
        int sm = props.major * 10 + props.minor;
        this->sm = sm;

        // ping-pong mode
        // if(this->specialize == 3) {
        //    int buffers_a = Implicit_gemm_traits::BUFFERS_PER_SMEM_TILE_A;
        //    int buffers_b = Implicit_gemm_traits::BUFFERS_PER_SMEM_TILE_B;
        //    this->delta_img_head = buffers_a > 0 ? (this->loop_start + 1) % buffers_a : 0;
        //    this->delta_flt_head = buffers_b > 0 ? (this->loop_start + 1) % buffers_b : 0;
        //}

        // Init strided dgrad related params.
        this->valid_t = nullptr;
        this->valid_r = nullptr;
        this->valid_s = nullptr;
        this->ndhw_indices_of_each_filter_pattern_gmem = nullptr;
        this->start_cta_id_in_ndhw_dimension_of_each_filter_pattern = nullptr;
        this->dhw_count_of_each_filter_pattern = nullptr;

        // Prepare the filter pattern in 3 dimensions.
        std::vector<int> valid_count_d( this->t + 1, 0 );
        std::vector<int> valid_count_h( this->r + 1, 0 );
        std::vector<int> valid_count_w( this->s + 1, 0 );

        for( int di = 0; di < this->d; ++di ) {
            int valid_tmp = -1;
            for( int ti = this->t - 1; ti >= 0; --ti ) {
                int oi = di + this->pad[0][0] - ti * this->dilation[0];
                if( oi >= 0 && oi % this->stride[0] == 0 && oi / this->stride[0] < this->o ) {
                    valid_tmp = ti;
                    break;
                }
            }
            valid_count_d[valid_tmp + 1] += 1;
        }

        for( int hi = 0; hi < this->h; ++hi ) {
            int valid_tmp = -1;
            for( int ri = this->r - 1; ri >= 0; --ri ) {
                int pi = hi + this->pad[1][0] - ri * this->dilation[1];
                if( pi >= 0 && pi % this->stride[1] == 0 && pi / this->stride[1] < this->p ) {
                    valid_tmp = ri;
                    break;
                }
            }
            valid_count_h[valid_tmp + 1] += 1;
        }

        for( int wi = 0; wi < this->w; ++wi ) {
            int valid_tmp = -1;
            for( int si = this->s - 1; si >= 0; --si ) {
                int qi = wi + this->pad[2][0] - si * this->dilation[2];
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
        for( int i = 0; i < this->trs; ++i ) {
            int expect_trs = this->trs - 1 - i;
            int expect_t = expect_trs / this->rs;
            int expect_rs = expect_trs % this->rs;
            int expect_r = expect_rs / this->s;
            int expect_s = expect_rs % this->s;
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
        const int DHW_PER_CTA = Implicit_gemm_traits::Cta_tile::M;
        this->sum_of_round_up_ndhw_number_of_each_filter_pattern = 0;
        for( int i = 0; i <= this->trs; ++i ) {
            if( dhw_indices[i] > 0 ) {
                this->sum_of_round_up_ndhw_number_of_each_filter_pattern +=
                    xmma::div_round_up( this->n * dhw_indices[i], DHW_PER_CTA );
            }
        }
        this->sum_of_round_up_ndhw_number_of_each_filter_pattern *= DHW_PER_CTA;

        if( callwlate_splitk_params() != xmma::Error::SUCCESS ) {
            return xmma::Error::ERROR_ILWALID_PARAMS;
        }

        if( callwlate_strided_dgrad_params() != xmma::Error::SUCCESS ) {
            return xmma::Error::ERROR_ILWALID_PARAMS;
        }

        // Do we need a sequential reduction?
        workspace->split_k_with_reduction = this->split_k.with_reduction();
        workspace->device_workspace_size = this->split_k.size_in_bytes();

        dim3 grid;
        xmma::implicit_gemm::strided_dgrad_indexed::warp_specialized_compute_grid_dimensions(
            grid,
            *this,
            Implicit_gemm_traits::Cta_tile::M,
            Implicit_gemm_traits::Cta_tile::N,
            Implicit_gemm_traits::Cta_tile::GROUPS );

        // Hardware limitation
        if( this->use_horizontal_cta_rasterization == 1 && grid.y >= 65536 ) {
            this->use_horizontal_cta_rasterization = 0;
            int tmp = grid.x;
            grid.x = grid.y;
            grid.y = tmp;
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
        xmma::Cta_swizzle cs = xmma::Cta_swizzle( grid_dim,
                                                  cta_tile,
                                                  filter,
                                                  colw_stride,
                                                  output,
                                                  this->use_horizontal_cta_rasterization );
        this->best_log2_group_cols = cs.choose_best_log2_group_cols( this->ctas_per_wave );

        workspace->grid = grid;
        workspace->smem_size = Implicit_gemm_traits::dynamic_smem_size_per_cta();
        const int EPILOGUE_SIZE_IN_BYTES = Implicit_gemm_traits::Swizzle_epilogue::BYTES_PER_TILE;
        workspace->epilogue_size_in_bytes = EPILOGUE_SIZE_IN_BYTES;

        workspace->xmma_params = *this;

        return xmma::Error::SUCCESS;
    }

  protected:
    xmma::Error callwlate_splitk_params();
    xmma::Error callwlate_strided_dgrad_params();
};

///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits, typename Cta_tile>  //, int STAGES >
xmma::Error Warp_specialized_params<Traits, Cta_tile>::callwlate_splitk_params() {
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

        const int tile_m = Cta_tile::M;
        const int tile_n = Cta_tile::N;
        const int tile_g = Cta_tile::GROUPS;

        dim3 grid;
        xmma::implicit_gemm::strided_dgrad_indexed::warp_specialized_compute_grid_dimensions(
            grid, *this, tile_m, tile_n, tile_g );

        grid_size = (size_t) this->tiles_m * this->tiles_n;
        data_size = (size_t) this->tiles_m * this->tiles_n * tile_m * tile_n;

        // Size to allocate the buffers.
        using Acc_type = typename Traits::Aclwmulator_type;
        this->split_k.buffer_size = ( int64_t )( sizeof( Acc_type ) * (int)data_size );

        // Size to allocate the counters/locks.
        this->split_k.counters_ctas_size =
            (int)( grid_size * this->split_k.buffers * sizeof( int32_t ) );
        this->split_k.retired_ctas_size = (int)( grid_size * sizeof( int32_t ) );
    }

    return xmma::Error::SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits, typename Cta_tile>
xmma::Error Warp_specialized_params<Traits, Cta_tile>::callwlate_strided_dgrad_params() {
    this->with_residual = xmma::colwert<float>( this->beta ) == 0.f ? 0 : 1;
    this->use_horizontal_cta_rasterization = 1;

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
    int k_per_iteration = Cta_tile::K;
    if( this->split_k.slices > 1 && this->split_k_k > 0 ) {
        k_per_iteration *= this->split_k.slices;
    }

    // The number of loop iterations to cover C elements.
    this->loop_count_k = xmma::div_up( Cta_tile::GROUPS * this->k, k_per_iteration );
    // The number of elements read when we enter the residue.
    this->loop_residue_k = ( this->loop_count_k - 1 ) * k_per_iteration;

    // Bias element number
    if( this->with_bias ) {
        this->with_bias = this->g * this->c;
    }

    return xmma::Error::SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace strided_dgrad_indexed
}  // namespace implicit_gemm
}  // namespace xmma
