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

#include <xmma/numeric_types.h>
#include <xmma/cta_swizzle.h>
#include <xmma/params.h>
#include <xmma/implicit_gemm/fprop/utils.h>
#include <xmma/implicit_gemm/utils.h>

namespace xmma {
namespace implicit_gemm {
namespace interleaved_fprop {

///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits, typename Cta_tile, int32_t STAGES = 1>
struct Params : public xmma::Colwolution_params_base {
    // Do we have a residual?
    int32_t with_residual;

    // The knobs to control how we split the filter for split-k.
    int32_t split_k_t, split_k_r, split_k_c;
    // Precomputed values.
    int32_t split_k_trs, split_k_rs;

    // Precomputed values. Reserved!
    int dhwc, dhw, hw, nopq, opq, pq, trsc, trs;
    // Precomputed values for fast divisions.
    uint32_t mul_opq, shr_opq, mul_pq, shr_pq, mul_q, shr_q;
    // Precomputed values for fast divisions for the kernel without L1 replications.
    uint32_t ctas_pq, mul_ctas_pq, shr_ctas_pq, ctas_q, mul_ctas_q, shr_ctas_q;
    // Precomputed deltas for the image and the filter.
    int64_t a_delta[32], b_delta[1];

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
    int32_t pool_factor;

    // Do we use horizontal rasterization of CTAs?
    int32_t use_horizontal_cta_rasterization;
    // Best group col width(the log to the base 2) for CTA swizzling
    unsigned best_log2_group_cols;
    // The number of CTA tiles in each dimension.
    int32_t tiles_m, tiles_n, tiles_k;
    // The number of CTA tiles in the grid.
    int32_t tiles_x, tiles_y;

    // Precomputed values for fast divisions of filter_trs_per_cta.
    uint32_t mul_filter_trs_per_cta, shr_filter_trs_per_cta;
    // Precomputed values for fast divisions of filter_rs_per_cta.
    uint32_t mul_filter_rs_per_cta, shr_filter_rs_per_cta;
    // Precomputed values for fast divisions of filter_s_per_cta.
    uint32_t mul_filter_s_per_cta, shr_filter_s_per_cta;

    // Ctor
    Params() : Colwolution_params_base() {
        ctas_pq = 0;
        mul_ctas_pq = 0;
        shr_ctas_pq = 0;
        ctas_q = 0;
        mul_ctas_q = 0;
        shr_ctas_q = 0;
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
            this->tiles_n = xmma::div_up( this->k, Cta_tile::N );
        } else {
            this->tiles_m = xmma::div_up( this->n * this->o * this->p * this->q, Cta_tile::M );
            this->tiles_n = xmma::div_up( this->k * this->g, Cta_tile::N );
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

    // The size in the C dimension in bits.
    const int32_t a_c_in_bits = this->g * this->c * Traits::BITS_PER_ELEMENT_A;
    const int32_t b_c_in_bits = this->g * this->c * Traits::BITS_PER_ELEMENT_B;

    // If the number of filters is not a multiple of K, just skip the kernel.
    if( a_c_in_bits % 8 != 0 || b_c_in_bits % 8 != 0 || this->g * this->k % 8 != 0 ) {
        return xmma::Error::ERROR_ILWALID_PARAMS;
    }

    // TODO: needs to better to set this
    const int32_t ELEMENTS_PER_PACKET = Traits::BITS_PER_ELEMENT_A == 16 ? 8 : 32; //8 for hmma Nc8hw8

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
    const int32_t FILTER_TAPS_PER_ITERATION = Traits::BITS_PER_ELEMENT_A == 16 ? 4 : Cta_tile::GROUPS > 1 ? 1 : 2;
    int32_t flt_t = this->filter_t_per_cta;
    int32_t flt_r = this->filter_r_per_cta;
    int32_t flt_s = this->filter_s_per_cta;
    for( int i = 0; i < flt_t * flt_r * flt_s; ++i ) {
        // The position in the filter.
        int32_t t = i / ( flt_r * flt_s );
        int32_t r = i % ( flt_r * flt_s ) / flt_s;
        int32_t s = i % ( flt_r * flt_s ) % flt_s;

        // The next position in the filter.
        int32_t next_i = i + FILTER_TAPS_PER_ITERATION;

        // Decompose the next position in the filter.
        int32_t next_c = next_i / ( flt_t * flt_r * flt_s );
        int32_t next_t = next_i % ( flt_t * flt_r * flt_s ) / ( flt_r * flt_s );
        int32_t next_r = next_i % ( flt_t * flt_r * flt_s ) % ( flt_r * flt_s ) / flt_s;
        int32_t next_s = next_i % ( flt_t * flt_r * flt_s ) % ( flt_r * flt_s ) % flt_s;

        // The offset.
        int32_t offset = ( next_c - 0 ) * this->img_stride_c +
                         ( next_t - t ) * this->dilation[0] * this->img_stride_d +
                         ( next_r - r ) * this->dilation[1] * this->img_stride_h +
                         ( next_s - s ) * this->dilation[2] * this->img_stride_w;
        offset *= ELEMENTS_PER_PACKET;

        // Compute the delta offset from one position to the next one.
        this->a_delta[i] = Traits::offset_in_bytes_a( offset );
    }

    constexpr int32_t per_group_c_k = Cta_tile::K / Cta_tile::GROUPS;
    // for group colw, if per-group c/k < 32, b tensor will be padded to 32

    // The deltas for the filter.
    int32_t round_k = max( 32, per_group_c_k );
    this->b_delta[0] = Traits::offset_in_bytes_b( this->k * this->g * ( round_k ) );

    // The number of elements in the C dimension that are used per iteration.
    int32_t c_per_iteration = per_group_c_k;
    if( this->split_k.slices > 1 && this->split_k_c > 0 ) {
        c_per_iteration *= this->split_k.slices;
    }
    // The number of loop iterations to cover C elements.
    int32_t loop_count_k = xmma::div_up( this->c * this->filter_trs_per_cta, per_group_c_k );
    // The first iteration of the loop.
    this->loop_start = loop_count_k - 1;
    // The iteration where we trigger the residue.
    this->loop_residue = max( 2, STAGES );
    // The number of elements read when we enter the residue.
    this->loop_residue_k = ( loop_count_k - 1 ) * c_per_iteration;

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
    // IMMA kernels doesn't support split-k.
    return xmma::Error::SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace interleaved_fprop
}  // namespace implicit_gemm
}  // namespace xmma
