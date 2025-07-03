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
#include <xmma/numeric_types.h>
#include <xmma/cta_swizzle.h>
#include <xmma/params.h>
#include <xmma/implicit_gemm/interleaved_fprop/warp_specialized_utils.h>

namespace xmma {
namespace implicit_gemm {
namespace interleaved_fprop {

///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits, typename Cta_tile>  // int STAGES = 1>
struct Warp_specialized_params : public xmma::Colwolution_params_base {
    // Do we have a residual?
    int with_residual;

    // The knobs to control how we split the filter for split-k.
    int split_k_t, split_k_r, split_k_c;
    // Precomputed values.
    int split_k_trs, split_k_rs;

    // Precomputed values. Reserved!
    int dhwc, dhw, hw, nopq, opq, pq, trsc, trs;
    // Precomputed values for fast divisions.
    uint32_t mul_opq, shr_opq, mul_pq, shr_pq, mul_q, shr_q;
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
    int pool_factor;

    // Do we use horizontal rasterization of CTAs?
    int use_horizontal_cta_rasterization;
    // Best group col width(the log to the base 2) for CTA swizzling
    unsigned best_log2_group_cols;
    // The number of CTA tiles in each dimension.
    int tiles_m, tiles_n, tiles_k;
    // The number of CTA tiles in the grid.
    int tiles_x, tiles_y;

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

    // Precomputed values for fast divisions of filter_trs_per_cta.
    uint32_t mul_filter_trs_per_cta, shr_filter_trs_per_cta;
    // Precomputed values for fast divisions of filter_rs_per_cta.
    uint32_t mul_filter_rs_per_cta, shr_filter_rs_per_cta;
    // Precomputed values for fast divisions of filter_s_per_cta.
    uint32_t mul_filter_s_per_cta, shr_filter_s_per_cta;

    // The number of stages.
    enum { STAGES = 1 };

    // Initialize params from base params
    template <typename Implicit_gemm_traits>
    xmma::Error initialize( xmma::Host_workspace<Implicit_gemm_traits>* workspace ) {

        // Warp specialization related
        this->specialize = Implicit_gemm_traits::WARP_SPECIALIZED_CONFIG;
        // Device info.
        lwdaDeviceProp props;
        int dev = 0;
        XMMA_LWDA_CALL( lwdaGetDeviceProperties( &props, dev ) );
        // persistent CTA:  1CTA/SM.
        this->tile_move_step = props.multiProcessorCount;
        int sm = props.major * 10 + props.minor;
        this->sm = sm;
        // 2math config
        if( this->specialize == xmma::CONFIG_1DMA_2MATH ) {
            int buffers_a = Implicit_gemm_traits::BUFFERS_PER_SMEM_TILE_A;
            int buffers_b = Implicit_gemm_traits::BUFFERS_PER_SMEM_TILE_B;
            this->delta_img_head = buffers_a > 0 ? ( this->loop_start + 1 ) % buffers_a : 0;
            this->delta_flt_head = buffers_b > 0 ? ( this->loop_start + 1 ) % buffers_b : 0;
        }

        // Regular IMMA
        if( callwlate_splitk_params() != xmma::Error::SUCCESS ) {
            return xmma::Error::ERROR_ILWALID_PARAMS;
        }

        if( callwlate_fprop_params() != xmma::Error::SUCCESS ) {
            return xmma::Error::ERROR_ILWALID_PARAMS;
        }

        // Initialize workspace

        // Do we need a sequential reduction?
        workspace->split_k_with_reduction = this->split_k.with_reduction();
        workspace->device_workspace_size = this->split_k.size_in_bytes();

        dim3 grid;
        xmma::implicit_gemm::interleaved_fprop::warp_specialized_compute_grid_dimensions(
            grid, *this, Implicit_gemm_traits::Cta_tile::M, Implicit_gemm_traits::Cta_tile::N );

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
        xmma::Cta_swizzle cs = xmma::Cta_swizzle( grid_dim, cta_tile );
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
    xmma::Error callwlate_fprop_params();
};

///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits, typename Cta_tile>  // int STAGES = 1>
xmma::Error Warp_specialized_params<Traits, Cta_tile>::callwlate_splitk_params() {
    // IMMA kernels doesn't support split-k.
    return xmma::Error::SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits, typename Cta_tile>  // int STAGES = 1>
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

    // TODO: needs to better to set this
    const int ELEMENTS_PER_PACKET = 32;

    // Update stride info.
    // this->img_stride_n *= ELEMENTS_PER_PACKET;
    // this->img_stride_d *= ELEMENTS_PER_PACKET;
    // this->img_stride_h *= ELEMENTS_PER_PACKET;
    // this->img_stride_w *= ELEMENTS_PER_PACKET;
    // this->img_stride_c *= ELEMENTS_PER_PACKET;
    // this->out_stride_n *= ELEMENTS_PER_PACKET;
    // this->out_stride_d *= ELEMENTS_PER_PACKET;
    // this->out_stride_h *= ELEMENTS_PER_PACKET;
    // this->out_stride_w *= ELEMENTS_PER_PACKET;
    // this->out_stride_c *= ELEMENTS_PER_PACKET;

    // Compute precomputed values.
    this->dhwc = this->d * this->h * this->w * this->g * this->c;
    this->dhw = this->d * this->h * this->w;
    this->hw = this->h * this->w;
    this->w = this->w;
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

    xmma::find_divisor(
        this->mul_filter_trs_per_cta, this->shr_filter_trs_per_cta, this->filter_trs_per_cta );
    xmma::find_divisor(
        this->mul_filter_rs_per_cta, this->shr_filter_rs_per_cta, this->filter_rs_per_cta );
    xmma::find_divisor(
        this->mul_filter_s_per_cta, this->shr_filter_s_per_cta, this->filter_s_per_cta );

    // Set masks.
    xmma::implicit_gemm::build_masks( this->mask_t,
                                      this->mask_r,
                                      this->mask_s,
                                      this->filter_t_per_cta,
                                      this->filter_r_per_cta,
                                      this->filter_s_per_cta );

    // The deltas for the image.
    // TODO: needs a better way to set this
    const int FILTER_TAPS_PER_ITERATION = 2;
    int flt_t = this->filter_t_per_cta;
    int flt_r = this->filter_r_per_cta;
    int flt_s = this->filter_s_per_cta;
    for( int i = 0; i < flt_t * flt_r * flt_s; ++i ) {
        // The position in the filter.
        int t = i / ( flt_r * flt_s );
        int r = i % ( flt_r * flt_s ) / flt_s;
        int s = i % ( flt_r * flt_s ) % flt_s;

        // The next position in the filter.
        int next_i = i + FILTER_TAPS_PER_ITERATION;

        // Decompose the next position in the filter.
        int next_c = next_i / ( flt_t * flt_r * flt_s );
        int next_t = next_i % ( flt_t * flt_r * flt_s ) / ( flt_r * flt_s );
        int next_r = next_i % ( flt_t * flt_r * flt_s ) % ( flt_r * flt_s ) / flt_s;
        int next_s = next_i % ( flt_t * flt_r * flt_s ) % ( flt_r * flt_s ) % flt_s;

        // The offset.
        int offset = ( next_c - 0 ) * this->img_stride_c +
                     ( next_t - t ) * this->dilation[0] * this->img_stride_d +
                     ( next_r - r ) * this->dilation[1] * this->img_stride_h +
                     ( next_s - s ) * this->dilation[2] * this->img_stride_w;
        offset *= ELEMENTS_PER_PACKET;

        // Compute the delta offset from one position to the next one.
        this->a_delta[i] = Traits::offset_in_bytes_a( offset );
    }

    // The deltas for the filter.
    for( int i = 0; i < flt_t * flt_r * flt_s; ++i ) {
        this->b_delta[i] = Traits::offset_in_bytes_b( this->k * Cta_tile::K );
    }

    // The number of elements in the C dimension that are used per iteration.
    int c_per_iteration = Cta_tile::K;
    if( this->split_k.slices > 1 && this->split_k_c > 0 ) {
        c_per_iteration *= this->split_k.slices;
    }

    constexpr int XMMAS_GROUPS = Cta_tile::GROUPS > 1 ? 2 : 1;

    // The number of loop iterations to cover C elements.
    int loop_count_k =
        xmma::div_up( this->c * this->filter_trs_per_cta, Cta_tile::K / XMMAS_GROUPS );
    // The first iteration of the loop.
    this->loop_start = loop_count_k - 1;
    // The iteration where we trigger the residue
    this->loop_residue = 1;
    // The number of elements read when we enter the residue.
    this->loop_residue_k = ( loop_count_k - 1 ) * c_per_iteration;

    return xmma::Error::SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace interleaved_fprop
}  // namespace implicit_gemm
}  // namespace xmma
