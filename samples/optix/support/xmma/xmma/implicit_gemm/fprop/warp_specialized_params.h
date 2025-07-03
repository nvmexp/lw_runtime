/***************************************************************************************************
 * Copyright (c) 2011-2019, LWPU CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 *modification, are not permit-
 * ted.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 *MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL LWPU
 *CORPORATION BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 *DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROLWREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 *DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 *WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY
 *WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
#pragma once

#include <xmma/xmma.h>
#include <xmma/cta_swizzle.h>
#include <xmma/implicit_gemm/fprop/params.h>
#include <xmma/implicit_gemm/fprop/warp_specialized_utils.h>
#include <xmma/implicit_gemm/utils.h>
#include <xmma/params.h>
#include <xmma/warp_specialized_traits.h>

namespace xmma {
namespace implicit_gemm {
namespace fprop {

///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits, typename Cta_tile>  //, int STAGES = 1 >
struct Warp_specialized_params : public xmma::implicit_gemm::fprop::Params<Traits, Cta_tile, 1> {
    // lic+
    // warp specialized parameters
    // which warp specialization mode
    int specialize;
    // used in ping-pong mode
    int buffers_img, buffers_flt, buffers_epilog;
    int delta_flt_head, delta_img_head;
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
        this->specialize = Implicit_gemm_traits::WARP_SPECIALIZED_CONFIG;
        // Device info.
        lwdaDeviceProp props;
        int dev = 0;
        XMMA_LWDA_CALL( lwdaGetDeviceProperties( &props, dev ) );
        // persistent CTA:  1CTA/SM.
        this->tile_move_step = props.multiProcessorCount;
        int sm = props.major * 10 + props.minor;
        this->sm = sm;
        if( callwlate_splitk_params() != xmma::Error::SUCCESS ) {
            return xmma::Error::ERROR_ILWALID_PARAMS;
        }

        if( callwlate_fprop_params() != xmma::Error::SUCCESS ) {
            return xmma::Error::ERROR_ILWALID_PARAMS;
        }

        // 2math config
        if( this->specialize == xmma::CONFIG_1DMA_2MATH ) {
            int buffers_a = Implicit_gemm_traits::BUFFERS_PER_SMEM_TILE_A;
            int buffers_b = Implicit_gemm_traits::BUFFERS_PER_SMEM_TILE_B;
            this->delta_img_head = buffers_a > 0 ? ( this->loop_start + 1 ) % buffers_a : 0;
            this->delta_flt_head = buffers_b > 0 ? ( this->loop_start + 1 ) % buffers_b : 0;
        }
        // Initialize workspace

        // Do we need a sequential reduction?
        workspace->split_k_with_reduction = this->split_k.with_reduction();
        workspace->device_workspace_size = this->split_k.size_in_bytes();

        dim3 grid;
        xmma::implicit_gemm::fprop::warp_specialized_compute_grid_dimensions(
            grid, *this, Implicit_gemm_traits::Cta_tile::M, Implicit_gemm_traits::Cta_tile::N );

        // The split K kernel needs a non-specialized grid, so specialize is set to
        // 0
        // here. It gets reset after compute_grid_dimensions is called.
        this->specialize = 0;
        int tmp = this->split_k.kernels;
        this->split_k.kernels = 1;
        dim3 split_k_grid;
        xmma::implicit_gemm::fprop::warp_specialized_compute_grid_dimensions(
            split_k_grid,
            *this,
            Implicit_gemm_traits::Cta_tile::M,
            Implicit_gemm_traits::Cta_tile::N );

        split_k_grid.z = Implicit_gemm_traits::Gmem_tile_epilogue::Layout::ROW
                             ? Implicit_gemm_traits::Xmma_tile::XMMAS_M
                             : Implicit_gemm_traits::Xmma_tile::XMMAS_N;
        this->specialize = Implicit_gemm_traits::WARP_SPECIALIZED_CONFIG;
        this->split_k.kernels = tmp;

        // Hardware limitation
        if( this->use_horizontal_cta_rasterization == 1 && grid.y >= 65536 ) {
            this->use_horizontal_cta_rasterization = 0;
            int tmp = grid.x;
            grid.x = grid.y;
            grid.y = tmp;
            tmp = split_k_grid.x;
            split_k_grid.x = split_k_grid.y;
            split_k_grid.y = tmp;
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
        xmma::Cta_swizzle cs = xmma::Cta_swizzle( grid_dim,
                                                  cta_tile,
                                                  filter,
                                                  colw_stride,
                                                  output,
                                                  this->use_horizontal_cta_rasterization );
        this->best_log2_group_cols = cs.choose_best_log2_group_cols( this->ctas_per_wave );

        workspace->grid = grid;
        workspace->split_k_grid = split_k_grid;
        workspace->smem_size = Implicit_gemm_traits::dynamic_smem_size_per_cta();

        const int EPILOGUE_SIZE_IN_BYTES = Implicit_gemm_traits::Swizzle_epilogue::BYTES_PER_TILE;

        workspace->epilogue_size_in_bytes = EPILOGUE_SIZE_IN_BYTES;

        workspace->xmma_params = *this;

        return xmma::Error::SUCCESS;
    }

  protected:
    xmma::Error callwlate_splitk_params();
    xmma::Error callwlate_fprop_params();
};

///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits, typename Cta_tile>  //, int STAGES >
xmma::Error Warp_specialized_params<Traits, Cta_tile>::callwlate_splitk_params() {

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

        const int tile_m = Cta_tile::M;
        const int tile_n = Cta_tile::N;

        dim3 grid;
        xmma::implicit_gemm::fprop::warp_specialized_compute_grid_dimensions(
            grid, *this, tile_m, tile_n );

        max_grid = max( max_grid, ( size_t ) this->tiles_m * this->tiles_n );
        max_data = max( max_data, ( size_t ) this->tiles_m * this->tiles_n * tile_m * tile_n );

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

template <typename Traits, typename Cta_tile>  //, int STAGES >
xmma::Error Warp_specialized_params<Traits, Cta_tile>::callwlate_fprop_params() {
    this->with_residual = xmma::colwert<float>( this->beta ) == 0.f ? 0 : 1;
    this->use_horizontal_cta_rasterization = 1;
    this->pool_factor = 1;

    // The size in the C dimension in bits.
    const int a_c_in_bits = this->g * this->c * Traits::BITS_PER_ELEMENT_A;
    const int b_c_in_bits = this->g * this->c * Traits::BITS_PER_ELEMENT_B;

    // If the number of filters is not a multiple of K, just skip the kernel.
    if( a_c_in_bits % 8 != 0 || b_c_in_bits % 8 != 0 || this->g * this->k % 8 != 0 ) {
        return xmma::Error::ERROR_ILWALID_PARAMS;
    }

    // Compute precomputed values.
    this->dhwc = this->d * this->h * this->w * this->g * this->c;
    this->hwc = this->h * this->w * this->g * this->c;
    this->wc = this->w * this->g * this->c;
    this->nopq = this->n * this->o * this->p * this->q;
    this->opq = this->o * this->p * this->q;
    this->pq = this->p * this->q;
    this->trsc = this->t * this->r * this->s * this->c;
    this->trs = this->t * this->r * this->s;

    // The fast division params.
    xmma::find_divisor( this->mul_opq, this->shr_opq, this->opq );
    xmma::find_divisor( this->mul_pq, this->shr_pq, this->pq );
    xmma::find_divisor( this->mul_q, this->shr_q, this->q );

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
    int move_c = Cta_tile::K;
    if( this->split_k.slices > 1 && this->split_k_c > 0 ) {
        move_c *= this->split_k.slices;
    }

    // The deltas for the image.
    for( int ti = 0; ti < this->filter_t_per_cta; ++ti ) {
        for( int ri = 0; ri < this->filter_r_per_cta; ++ri ) {
            for( int si = 0; si < this->filter_s_per_cta; ++si ) {
                int delta = 0;
                if( ti == this->filter_t_per_cta - 1 && ri == this->filter_r_per_cta - 1 &&
                    si == this->filter_s_per_cta - 1 ) {
                    delta = move_c -
                            ( this->filter_t_per_cta - 1 ) * this->hwc * this->dilation[0] -
                            ( this->filter_r_per_cta - 1 ) * this->wc * this->dilation[1] -
                            ( this->filter_s_per_cta - 1 ) * this->g * this->c * this->dilation[2];
                } else if( ri == this->filter_r_per_cta - 1 && si == this->filter_s_per_cta - 1 ) {
                    delta = this->hwc * this->dilation[0] -
                            ( this->filter_r_per_cta - 1 ) * this->wc * this->dilation[1] -
                            ( this->filter_s_per_cta - 1 ) * this->g * this->c * this->dilation[2];
                } else if( si == this->filter_s_per_cta - 1 ) {
                    delta = this->wc * this->dilation[1] -
                            ( this->filter_s_per_cta - 1 ) * this->g * this->c * this->dilation[2];
                } else {
                    delta = this->g * this->c * this->dilation[2];
                }
                this->a_delta[ti * this->filter_r_per_cta * this->filter_s_per_cta +
                              ri * this->filter_s_per_cta + si] =
                    Traits::offset_in_bytes_a( delta );
            }
        }
    }

    // The deltas for the filter.
    int b_delta = 0;
    if( this->cross_correlation ) {
        b_delta = this->c;
    } else {
        b_delta = -this->c;
    }
    for( int ii = 0; ii < this->filter_trs_per_cta - 1; ++ii ) {
        this->b_delta[ii] = Traits::offset_in_bytes_b( b_delta );
    }

    // Change the channel.
    if( this->cross_correlation ) {
        b_delta = -( this->filter_trs_per_cta - 1 ) * this->c;
    } else {
        b_delta = ( this->filter_trs_per_cta - 1 ) * this->c;
    }
    this->b_delta[this->filter_trs_per_cta - 1] = Traits::offset_in_bytes_b( move_c + b_delta );

    // The number of elements in the C dimension that are used per iteration.
    int c_per_iteration = Cta_tile::K;
    if( this->split_k.slices > 1 && this->split_k_c > 0 ) {
        c_per_iteration *= this->split_k.slices;
    }

    // The number of loop iterations to cover C elements.
    int loop_count_k = xmma::div_up( Cta_tile::GROUPS * this->c, c_per_iteration );
    // The first iteration of the loop.
    this->loop_start = this->filter_trs_per_cta * loop_count_k - 1;
    // The iteration where we trigger the residue.
    this->loop_residue = this->filter_trs_per_cta;
    // The number of elements read when we enter the residue.
    this->loop_residue_k = ( loop_count_k - 1 ) * c_per_iteration;

    // Bias element number
    if( this->with_bias ) {
        this->with_bias = this->g * this->k;
    }

    return xmma::Error::SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace fprop
}  // namespace implicit_gemm
}  // namespace xmma
