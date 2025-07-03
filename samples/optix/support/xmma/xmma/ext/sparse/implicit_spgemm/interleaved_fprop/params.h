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
#include <xmma/ext/sparse/implicit_spgemm/interleaved_fprop/utils.h>
#include <xmma/implicit_gemm/utils.h>

namespace xmma {
namespace ext {
namespace implicit_gemm {
namespace interleaved_fprop {

///////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Traits, typename Cta_tile, int STAGES = 1 >
struct Params : public xmma::Colwolution_params_base {
  // Do we have a residual?
  int with_residual;

  // The knobs to control how we split the filter for split-k.
  int split_k_t, split_k_r, split_k_c;
  // Precomputed values.
  int split_k_trs, split_k_rs;

  // Precomputed values. Reserved!
  int dhwc, dhw, hwc, wc, nopq, opq, pq, trsc, trs, rsc, sc;
  int hw32c, w32c;
  // Precomputed values for fast divisions.
  uint32_t mul_opq, shr_opq, mul_pq, shr_pq, mul_q, shr_q;
  uint32_t mul_rsc, shr_rsc, mul_sc, shr_sc, mul_c, shr_c;
  uint32_t mul_k, shr_k, kn;
  // Precomputed values for fast divisions for the kernel without L1 replications.
  uint32_t ctas_pq, mul_ctas_pq, shr_ctas_pq, ctas_q, mul_ctas_q, shr_ctas_q;
  // Precomputed deltas for the image and the filter.
  int64_t a_delta[32], b_delta[32];

  uint32_t padded_c;
  uint32_t padded_k;

  int filter_trs_per_cta, filter_rs_per_cta;
  uint32_t mask_t, mask_r, mask_s;

  const void *e_gmem;

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

  // Precomputed values for fast divisions of filter_trs_per_cta.
  uint32_t mul_filter_trs_per_cta, shr_filter_trs_per_cta;
  // Precomputed values for fast divisions of filter_rs_per_cta.
  uint32_t mul_filter_rs_per_cta, shr_filter_rs_per_cta;
  // Precomputed values for fast divisions of filter_s_per_cta.
  uint32_t mul_filter_s_per_cta, shr_filter_s_per_cta;

  // Initialize params from base params
  template<typename Implicit_gemm_traits>
  xmma::Error initialize(xmma::Host_workspace<Implicit_gemm_traits> *workspace)
  {
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
    compute_grid_dimensions(grid, *this, Implicit_gemm_traits::Cta_tile::M,
                            Implicit_gemm_traits::Cta_tile::N);

    // Hardware limitation
    if( this->use_horizontal_cta_rasterization == 0 && grid.y >= 65536 ) {
      this->use_horizontal_cta_rasterization = 1;
      int tmp = grid.x;
      grid.x = grid.y;
      grid.y = tmp;
    }

    // Choose best groupCols for CTA swizzling
    xmma::Cta_swizzle::Pos3 grid_dim = xmma::Cta_swizzle::Pos3(
        this->use_horizontal_cta_rasterization ? this->tiles_n : this->tiles_m,
        this->use_horizontal_cta_rasterization ? this->tiles_m : this->tiles_n, grid.z);
    xmma::Cta_swizzle::Pos2 cta_tile = xmma::Cta_swizzle::Pos2(
        this->use_horizontal_cta_rasterization ? Cta_tile::N : Cta_tile::M,
        this->use_horizontal_cta_rasterization ? Cta_tile::M : Cta_tile::N);
    xmma::Cta_swizzle::Pos2 filter = xmma::Cta_swizzle::Pos2( this->r, this->s );
    xmma::Cta_swizzle::Pos2 colw_stride = xmma::Cta_swizzle::Pos2( this->stride[1], this->stride[2] );
    xmma::Cta_swizzle::Pos2 output = xmma::Cta_swizzle::Pos2( this->p, this->q );
    xmma::Cta_swizzle cs = xmma::Cta_swizzle( grid_dim, cta_tile,
        filter, colw_stride, output, this->use_horizontal_cta_rasterization );
    this->best_log2_group_cols = cs.choose_best_log2_group_cols( this->ctas_per_wave );

    workspace->grid = grid;
    workspace->smem_size = Implicit_gemm_traits::dynamic_smem_size_per_cta();
    const int EPILOGUE_SIZE_IN_BYTES = 0;
    workspace->epilogue_size_in_bytes = EPILOGUE_SIZE_IN_BYTES;

    workspace->xmma_params = *this;

    return xmma::Error::SUCCESS;
  }

  // WAR Finalize
  xmma::Error finalize(dim3& grid)
  {
    if( callwlate_splitk_params() != xmma::Error::SUCCESS ) {
      return xmma::Error::ERROR_ILWALID_PARAMS;
    }

    if( callwlate_fprop_params() != xmma::Error::SUCCESS ) {
      return xmma::Error::ERROR_ILWALID_PARAMS;
    }

    // Initialize workspace

    // Do we need a sequential reduction?
    //workspace->split_k_with_reduction = this->split_k.with_reduction();
    //workspace->device_workspace_size = this->split_k.size_in_bytes();
    
    // Hardware limitation
    if( this->use_horizontal_cta_rasterization == 0 && grid.y >= 65536 ) {
      this->use_horizontal_cta_rasterization = 1;
      int tmp = grid.x;
      grid.x = grid.y;
      grid.y = tmp;
    }

    // Choose best groupCols for CTA swizzling
    xmma::Cta_swizzle::Pos3 grid_dim = xmma::Cta_swizzle::Pos3(
        this->use_horizontal_cta_rasterization ? this->tiles_n : this->tiles_m,
        this->use_horizontal_cta_rasterization ? this->tiles_m : this->tiles_n, grid.z);
    xmma::Cta_swizzle::Pos2 cta_tile = xmma::Cta_swizzle::Pos2(
        this->use_horizontal_cta_rasterization ? Cta_tile::N : Cta_tile::M,
        this->use_horizontal_cta_rasterization ? Cta_tile::M : Cta_tile::N);
    xmma::Cta_swizzle::Pos2 filter = xmma::Cta_swizzle::Pos2( this->r, this->s );
    xmma::Cta_swizzle::Pos2 colw_stride = xmma::Cta_swizzle::Pos2( this->stride[1], this->stride[2] );
    xmma::Cta_swizzle::Pos2 output = xmma::Cta_swizzle::Pos2( this->p, this->q );
    xmma::Cta_swizzle cs = xmma::Cta_swizzle( grid_dim, cta_tile,
        filter, colw_stride, output, this->use_horizontal_cta_rasterization );
    this->best_log2_group_cols = cs.choose_best_log2_group_cols( this->ctas_per_wave );

    //workspace->grid = grid;
    //workspace->smem_size = Implicit_gemm_traits::dynamic_smem_size_per_cta();
    //const int EPILOGUE_SIZE_IN_BYTES = 0;
    //workspace->epilogue_size_in_bytes = EPILOGUE_SIZE_IN_BYTES;

    //workspace->xmma_params = *this;

    return xmma::Error::SUCCESS;
  }

protected:
  xmma::Error callwlate_splitk_params();
  xmma::Error callwlate_fprop_params();
};

///////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Traits, typename Cta_tile, int STAGES >
xmma::Error Params<Traits, Cta_tile, STAGES>::callwlate_splitk_params() {

  this->split_k_c = (this->split_k.slices > 1 && !this->split_k_t && !this->split_k_r);

  this->split_k_trs = this->split_k_t * this->r * this->s;
  this->split_k_rs = this->split_k_r * this->s;

  // Make sure that if we do split-k in the C dimension, we use Cta_tile::K.
  this->split_k_c *= Cta_tile::K;

  // Make sure the split-k params are consistent.
  if (this->split_k.slices > 1) {
    this->split_k.buffers = max(this->split_k.buffers, 1);
  }

  // Allocate buffers to do split-k (if needed).
  if( this->split_k.buffers > 0 ) {
    size_t max_grid = 0, max_data = 0;

    const int tile_m = Cta_tile::M;
    const int tile_n = Cta_tile::N;

    dim3 grid;
    compute_grid_dimensions(grid,
                            *this,
                            tile_m,
                            tile_n);

    max_grid = max(max_grid, (size_t) this->tiles_m * this->tiles_n);
    max_data = max(max_data, (size_t) this->tiles_m * this->tiles_n * tile_m * tile_n);

    // Size to allocate the buffers.
    using Acc_type = typename Traits::Aclwmulator_type;
    this->split_k.buffer_size = (int64_t)(sizeof(Acc_type) * (int)max_data);

    // Size to allocate the counters/locks.
    this->split_k.counters_ctas_size = (int)(max_grid * this->split_k.buffers * sizeof(int32_t));
    this->split_k.retired_ctas_size = (int)(max_grid * sizeof(int32_t));
  }

  return xmma::Error::SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Traits, typename Cta_tile, int STAGES >
xmma::Error Params<Traits, Cta_tile, STAGES>::callwlate_fprop_params() {

  this->with_residual = xmma::colwert<float>(this->beta) == 0.f ? 0 : 1;
  this->use_horizontal_cta_rasterization = 0;
  this->pool_factor = 1;

  // The size in the C dimension in bits.
  const int a_c_in_bits = this->g * this->c * Traits::BITS_PER_ELEMENT_A;
  const int b_c_in_bits = this->g * this->c * Traits::BITS_PER_ELEMENT_B;

  // If the number of filters is not a multiple of K, just skip the kernel.
  if( a_c_in_bits % 8 != 0 || b_c_in_bits % 8 != 0 ||
      this->g * this->k % 8 != 0 ) {
    return xmma::Error::ERROR_ILWALID_PARAMS;
  }

  // Compute precomputed values.
  this->padded_c = xmma::div_up(this->c * this->t * this->r * this->s, Cta_tile::K) * Cta_tile::K;
  this->padded_k = xmma::div_up(this->k, 64) * 64;

  this->dhwc = this->d * this->h * this->w * this->g * this->c;
  this->dhw  = this->d * this->h * this->w;
  this->hwc  = this->h * this->w * this->g * this->c;
  this->wc   = this->w * this->g * this->c;
  this->hw32c  = this->h * this->w * 32;
  this->w32c   = this->w * 32;
  this->nopq = this->n * this->o * this->p * this->q;
  this->opq  = this->o * this->p * this->q;
  this->pq   = this->p * this->q;
  this->trsc = this->t * this->r * this->s * this->c / 2;
  this->trs  = this->t * this->r * this->s;
  this->rsc  = this->r * this->s * this->c;
  this->sc   = this->s * this->c;

  // The fast division params.
  xmma::find_divisor(this->mul_opq, this->shr_opq, this->opq);
  xmma::find_divisor(this->mul_pq,  this->shr_pq, this->pq);
  xmma::find_divisor(this->mul_q,   this->shr_q, this->q);

  this->filter_t_per_cta   = this->t;
  this->filter_r_per_cta   = this->r;
  this->filter_s_per_cta   = this->s;
  this->filter_rs_per_cta  = this->r * this->s;
  this->filter_trs_per_cta = this->t * this->r * this->s;

  xmma::find_divisor( this->mul_filter_trs_per_cta, this->shr_filter_trs_per_cta,
                          this->filter_trs_per_cta );
  xmma::find_divisor( this->mul_filter_rs_per_cta, this->shr_filter_rs_per_cta,
                          this->filter_rs_per_cta );
  xmma::find_divisor( this->mul_filter_s_per_cta, this->shr_filter_s_per_cta,
                          this->filter_s_per_cta );

  this->kn = xmma::div_up(this->k, Cta_tile::M);
  xmma::find_divisor( this->mul_k, this->shr_k, this->kn );

  this->simple1x1x1 = (this->t * this->r * this->s == 1);

    // Set masks.
    xmma::implicit_gemm::build_masks(this->mask_t, this->mask_r, this->mask_s,
                this->filter_t_per_cta, this->filter_r_per_cta, this->filter_s_per_cta);

    // The deltas for the image.
    // TODO: needs a better way to set this
    const int FILTER_TAPS_PER_ITERATION = 4;
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
        int offset = ( next_c - 0 ) * this->dhw*32 +
                     ( next_t - t ) * this->dilation[0] * this->hw32c +
                     ( next_r - r ) * this->dilation[1] * this->w32c +
                     ( next_s - s ) * this->dilation[2] * 32;
        // Compute the delta offset from one position to the next one.
        this->a_delta[i] = Traits::offset_in_bytes_a( offset );
    }

  // The number of elements in the C dimension that are used per iteration.
  int c_per_iteration = Cta_tile::K;

  // The number of loop iterations to cover C elements.
  int loop_count_k = xmma::div_up(
      Cta_tile::GROUPS * this->c * this->filter_trs_per_cta, c_per_iteration);
  // The first iteration of the loop.
  this->loop_start = loop_count_k - 1;
  // The iteration where we trigger the residue.
  this->loop_residue = max( 2, STAGES );
  // The number of elements read when we enter the residue.
  this->loop_residue_k = (loop_count_k - 1) * c_per_iteration;

  return xmma::Error::SUCCESS;

}

///////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace interleaved_fprop
} // namespace implicit_gemm
} // namespace ext
} // namespace xmma

