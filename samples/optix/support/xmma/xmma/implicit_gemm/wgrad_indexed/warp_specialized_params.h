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
#include <xmma/implicit_gemm/wgrad_indexed/warp_specialized_utils.h>
#include <xmma/implicit_gemm/wgrad_indexed/params.h>
#include <xmma/warp_specialized_traits.h>

namespace xmma {
namespace implicit_gemm {
namespace wgrad_indexed {

///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits, typename Cta_tile>  //, int STAGES = 1 >
struct Warp_specialized_params
    : public xmma::implicit_gemm::wgrad_indexed::Params<Traits, Cta_tile, 1> {
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
    // filter_trs added (no real meaning) to keep API CALLED IN KERNEL the same
    int filter_trs_per_cta;
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

        if( callwlate_wgrad_indexed_params() != xmma::Error::SUCCESS ) {
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
        xmma::implicit_gemm::wgrad_indexed::warp_specialized_compute_grid_dimensions(
            grid,
            *this,
            Implicit_gemm_traits::Cta_tile::M,
            Implicit_gemm_traits::Cta_tile::N,
            Implicit_gemm_traits::Cta_tile::K,
            Implicit_gemm_traits::Cta_tile::GROUPS );

        // The split K kernel needs a non-specialized grid, so specialize is set to 0
        // here. It gets reset after compute_grid_dimensions is called.
        this->specialize = 0;
        int tmp = this->split_k.kernels;
        this->split_k.kernels = 1;
        dim3 split_k_grid;
        xmma::implicit_gemm::wgrad_indexed::warp_specialized_compute_grid_dimensions(
            split_k_grid,
            *this,
            Implicit_gemm_traits::Cta_tile::M,
            Implicit_gemm_traits::Cta_tile::N,
            Implicit_gemm_traits::Cta_tile::K,
            Implicit_gemm_traits::Cta_tile::GROUPS );
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
        xmma::Cta_swizzle cs =
            xmma::Cta_swizzle( grid_dim, cta_tile, this->use_horizontal_cta_rasterization );
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
    xmma::Error callwlate_wgrad_indexed_params();
};

///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits, typename Cta_tile>  //, int STAGES>
xmma::Error Warp_specialized_params<Traits, Cta_tile>::callwlate_splitk_params() {

    // Make sure the split-k params are consistent.
    if( this->split_k.slices > 1 ) {
        this->split_k.buffers = max( this->split_k.buffers, 1 );
    }

    // Allocate buffers to do split-k (if needed).
    if( this->split_k.buffers > 0 ) {
        size_t max_grid = 0, max_data = 0;

        const int tile_m = Cta_tile::M;
        const int tile_n = Cta_tile::N;
        const int tile_k = Cta_tile::K;
        const int tile_g = Cta_tile::GROUPS;

        dim3 grid;
        xmma::implicit_gemm::wgrad_indexed::warp_specialized_compute_grid_dimensions(
            grid, *this, tile_m, tile_n, tile_k, tile_g );

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
xmma::Error Warp_specialized_params<Traits, Cta_tile>::callwlate_wgrad_indexed_params() {

    this->with_residual = xmma::colwert<float>( this->beta ) == 0.f ? 0 : 1;
    this->use_horizontal_cta_rasterization = 1;

    // Compute precomputed values.
    this->dhwc = this->d * this->h * this->w * this->g * this->c;
    this->hwc = this->h * this->w * this->g * this->c;
    this->wc = this->w * this->g * this->c;
    this->nopq = this->n * this->o * this->p * this->q;
    this->opq = this->o * this->p * this->q;
    this->pq = this->p * this->q;
    this->pqk = this->p * this->q * this->g * this->k;
    this->qk = this->q * this->g * this->k;
    this->nhw = this->n * this->h * this->w;
    this->trsc = this->t * this->r * this->s * this->c;
    this->rs = this->r * this->s;

    // The fast division params.
    xmma::find_divisor( this->mul_opq, this->shr_opq, this->opq );
    xmma::find_divisor( this->mul_pq, this->shr_pq, this->pq );
    xmma::find_divisor( this->mul_q, this->shr_q, this->q );
    xmma::find_divisor( this->mul_rs, this->shr_rs, this->rs );
    xmma::find_divisor( this->mul_s, this->shr_s, this->s );

    // The number of CTAs per C.
    this->c_per_ctas = Cta_tile::GROUPS * this->c;
    xmma::find_divisor( this->mul_c_per_ctas, this->shr_c_per_ctas, this->c_per_ctas );

    // The deltas for the pointers.
    int64_t a_delta = this->split_k.slices * Cta_tile::K * ( this->g * this->k );
    this->a_delta[0] = Traits::offset_in_bytes_a( a_delta );

    int64_t b_delta = this->split_k.slices * Cta_tile::K * ( this->g * this->c );
    this->b_delta[0] = Traits::offset_in_bytes_b( b_delta );

    // The number of loop iterations to cover NOPQ elements.
    int loop_count_nopq = xmma::div_up( this->nopq, this->split_k.slices * Cta_tile::K );
    // The first iteration of the loop.
    this->loop_start = loop_count_nopq - 1;
    // The iteration where we trigger the residue.
    this->loop_residue = 1;
    // The number of elements read when we enter the residue.
    this->loop_residue_k = ( loop_count_nopq - 1 ) * this->split_k.slices * Cta_tile::K;

    return xmma::Error::SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace wgrad_indexed
}  // namespace implicit_gemm
}  // namespace xmma
