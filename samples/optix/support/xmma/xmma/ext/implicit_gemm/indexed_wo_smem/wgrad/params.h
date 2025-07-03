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
#include <xmma/ext/implicit_gemm/indexed_wo_smem/wgrad/utils.h>

namespace xmma {
namespace ext {
namespace implicit_gemm {
namespace indexed_wo_smem {
namespace wgrad {

///////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Traits, typename Cta_tile, int STAGES = 1 >
struct Params : public xmma::Colwolution_params_base {

  // Do we have a residual?
  int with_residual;

  // Precomputed values. Reserved!
  int nopq, opq, pq, pqk, qk, nhw, dhwc, dhw, hw, hwc, wc, trsc, rsc, sc, rs;
  // The number of CTAs in the C dimension.
  int ctas_per_c, ctas_per_pq, ctas_per_q;
  // Precomputed values for fast divisions.
  uint32_t mul_opq, shr_opq, mul_pq, shr_pq, mul_q, shr_q, mul_rsc, shr_rsc, mul_sc, shr_sc, mul_c, shr_c;
  uint32_t mul_rs, shr_rs, mul_s, shr_s;
  // Precomputed values for fast divisions.
  uint32_t mul_ctas_per_c, shr_ctas_per_c;
  // Precomputed values for fast divisions (only for split along N).
  uint32_t mul_ctas_per_pq, shr_ctas_per_pq, mul_ctas_per_q, shr_ctas_per_q;
  uint32_t mul_k, shr_k, kn;
  // Precomputed deltas for the image and the filter.
  int64_t a_delta[1], b_delta[1];

  // The loop count.
  int loop_start;
  // The index of the loop count where we trigger the residue.
  int loop_residue;
  // The number of elements read before we enter the residue in the GEMM-K dimension.
  int loop_residue_k;
  // Whether or not we are trying to run Ampere kernels.
  bool ampere;

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
  template<typename Implicit_gemm_traits>
  xmma::Error initialize(xmma::Host_workspace<Implicit_gemm_traits> *workspace)
  {
    if( callwlate_splitk_params() != xmma::Error::SUCCESS ) {
      return xmma::Error::ERROR_ILWALID_PARAMS;
    }

    if( callwlate_wgrad_params() != xmma::Error::SUCCESS ) {
      return xmma::Error::ERROR_ILWALID_PARAMS;
    }

    // Initialize workspace

    // Do we need a sequential reduction?
    workspace->split_k_with_reduction = this->split_k.with_reduction();
    workspace->device_workspace_size = this->split_k.size_in_bytes();

    dim3 grid;
    compute_grid_dimensions(grid,
                            *this,
                            Implicit_gemm_traits::Cta_tile::M,
                            Implicit_gemm_traits::Cta_tile::N,
                            Implicit_gemm_traits::Cta_tile::K,
			    Implicit_gemm_traits::Cta_tile::GROUPS);

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
        this->use_horizontal_cta_rasterization ? this->tiles_m : this->tiles_n, grid.z);
    xmma::Cta_swizzle::Pos2 cta_tile = xmma::Cta_swizzle::Pos2(
        this->use_horizontal_cta_rasterization ? Cta_tile::N : Cta_tile::M,
        this->use_horizontal_cta_rasterization ? Cta_tile::M : Cta_tile::N);
    xmma::Cta_swizzle cs = xmma::Cta_swizzle( grid_dim, cta_tile, this->use_horizontal_cta_rasterization );
    this->best_log2_group_cols = cs.choose_best_log2_group_cols( this->ctas_per_wave );

    workspace->grid = grid;
    workspace->smem_size = Implicit_gemm_traits::dynamic_smem_size_per_cta();
    const int EPILOGUE_SIZE_IN_BYTES = 0;//Implicit_gemm_traits::Swizzle_epilogue::BYTES_PER_TILE;
    workspace->epilogue_size_in_bytes = EPILOGUE_SIZE_IN_BYTES;

    workspace->xmma_params = *this;

    return xmma::Error::SUCCESS;
  }

protected:
  xmma::Error callwlate_splitk_params();
  xmma::Error callwlate_wgrad_params();
};

///////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Traits, typename Cta_tile, int STAGES>
xmma::Error Params<Traits, Cta_tile, STAGES>::callwlate_splitk_params() {

  // Make sure the split-k params are consistent.
  if (this->split_k.slices > 1) {
    this->split_k.buffers = max(this->split_k.buffers, 1);
  }

  // Allocate buffers to do split-k (if needed).
  if( this->split_k.buffers > 0 ) {
    size_t max_grid = 0, max_data = 0;

    const int tile_m = Cta_tile::M;
    const int tile_n = Cta_tile::N;
    const int tile_k = Cta_tile::K;
    const int tile_g = Cta_tile::GROUPS;

    dim3 grid;
    compute_grid_dimensions(grid,
                            *this,
                            tile_m,
                            tile_n,
                            tile_k,
                            tile_g);

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
xmma::Error Params<Traits, Cta_tile, STAGES>::callwlate_wgrad_params() {

  this->with_residual = xmma::colwert<float>(this->beta) == 0.f ? 0 : 1;
  this->use_horizontal_cta_rasterization = 1;

  // Compute precomputed values.
  this->dhwc = this->d * this->h * this->w * this->g * this->c;
  this->dhw  = this->d * this->h * this->w;
  this->hw   = this->h * this->w;
  this->hwc  = this->h * this->w * this->g * this->c;
  this->wc   = this->w * this->g * this->c;
  this->nopq = this->n * this->o * this->p * this->q;
  this->opq  = this->o * this->p * this->q;
  this->pq   = this->p * this->q;
  this->pqk  = this->p * this->q * this->g * this->k;
  this->qk   = this->q * this->g * this->k;
  this->nhw  = this->n * this->h * this->w;
  this->trsc = this->t * this->r * this->s * this->c;
  this->rsc  = this->r * this->s * this->c;
  this->sc   = this->s * this->c;
  this->rs   = this->r * this->s;

  // The fast division params.
  xmma::find_divisor(this->mul_opq, this->shr_opq, this->opq);
  xmma::find_divisor(this->mul_pq,  this->shr_pq, this->pq);
  xmma::find_divisor(this->mul_q,   this->shr_q, this->q);

  xmma::find_divisor(this->mul_rsc,  this->shr_rsc, this->rsc);
  xmma::find_divisor(this->mul_sc,  this->shr_sc, this->sc);
  xmma::find_divisor(this->mul_c,   this->shr_c, this->c);

  this->kn = 1;
  xmma::find_divisor( this->mul_k, this->shr_k, this->kn );

  // The number of CTAs per C.
  this->ctas_per_c = xmma::div_up(Cta_tile::GROUPS * this->c, Cta_tile::N);
  xmma::find_divisor(this->mul_ctas_per_c, this->shr_ctas_per_c, this->ctas_per_c);

  // The deltas for the pointers.
  int64_t a_delta = this->split_k.slices * Cta_tile::K * (this->g * this->k);
  this->a_delta[0] = Traits::offset_in_bytes_a(a_delta);

  int64_t b_delta = this->split_k.slices * Cta_tile::K * (this->g * this->c);
  this->b_delta[0] = Traits::offset_in_bytes_b(b_delta);

  // The number of loop iterations to cover NOPQ elements.
  int loop_count_nopq = xmma::div_up(this->nopq, this->split_k.slices * Cta_tile::K);
  // The first iteration of the loop.
  this->loop_start = loop_count_nopq - 1;
  // The iteration where we trigger the residue.
  this->loop_residue = max(2, STAGES);
  // The number of elements read when we enter the residue.
  this->loop_residue_k = (loop_count_nopq - 1) * this->split_k.slices * Cta_tile::K;

  return xmma::Error::SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace wgrad
} // namespace indexed_wo_smem
} // namespace implicit_gemm
} // namespace ext
} // namespace xmma

