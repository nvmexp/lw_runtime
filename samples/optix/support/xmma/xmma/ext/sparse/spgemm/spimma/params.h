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

#include <stdint.h>
#include <xmma/params.h>
#include <xmma/cta_swizzle.h>
#include <xmma/ext/sparse/spgemm/utils.h>

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace xmma {
namespace ext {
namespace gemm {
namespace sparse_imma_gemm {
////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Traits, typename Cta_tile, int STAGE = 1 >
struct Xmma_sparse_gemm_params : public xmma::Gemm_params_base {
  // Precomputed deltas for the matrices.
  int64_t a_delta, b_delta;

  // Do we use horizontal rasterization of CTAs?
  int use_horizontal_cta_rasterization;
  // The loop count.
  int loop_start;
  // The index of the loop count where we trigger the residue.
  int loop_residue;
  // The number of K elements consumed when we enter the resiude.
  int loop_residue_k;
  // Whether or not we are trying to run Ampere kernels.
  bool ampere;
  // The number of CTA tiles in each dimension.
  int tiles_m, tiles_n, tiles_k;
  // The number of CTA tiles in the grid.
  int tiles_x, tiles_y;

  // Best group col width(the log to the base 2) for CTA swizzling
  unsigned best_log2_group_cols;

  // Beta
  bool has_beta;
  // The Metedata matrix.
  const void *e_gmem;
  // Precomputed deltas for the metadata in device memory
  int64_t e_delta;
  // Sparse factor
  int sparse_factor;
  // Padded colwerted k dimension for metadata
  int meta_k_pad;
  // Sparse or not
  bool is_sparse;

  // Initialize params from base params
  template<typename Gemm_traits>
  xmma::Error initialize(xmma::Host_workspace<Gemm_traits> *workspace)
  {
    if( callwlate_splitk_params() != xmma::Error::SUCCESS) {
      return xmma::Error::ERROR_ILWALID_PARAMS;
    }

    if( callwlate_gemm_params() != xmma::Error::SUCCESS) {
      return xmma::Error::ERROR_ILWALID_PARAMS;
    }

    // Enable beta load or not
    if ( this->beta > 0.0f ) {
      this->has_beta = true;
    } else {
      this->has_beta = false;
    }

    // Initialize workspace

    // Do we need a sequential reduction?
    workspace->split_k_with_reduction = this->split_k.with_reduction();
    workspace->device_workspace_size = this->split_k.size_in_bytes();

    dim3 grid;
    compute_grid_dimensions(grid, *this, Gemm_traits::Cta_tile::M, Gemm_traits::Cta_tile::N);

    ///////////////////////////////////////////////////////////////////////////////////////////////////
    // Enable CTA swizzle
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
    ///////////////////////////////////////////////////////////////////////////////////////////////////

    workspace->grid = grid;

    // The amount of shared memory needed by the epilogue.
    //const int EPILOGUE_SIZE_IN_BYTES = Gemm_traits::Swizzle_epilogue::BYTES_PER_TILE;
    //workspace->epilogue_size_in_bytes = EPILOGUE_SIZE_IN_BYTES;
    // The amount of shared memory needed for the main loop.
    workspace->smem_size = Gemm_traits::dynamic_smem_size_per_cta();

    workspace->xmma_params = *this;

    return xmma::Error::SUCCESS;
  }

  xmma::Error finalize(dim3& grid)
  {
    if( callwlate_splitk_params() != xmma::Error::SUCCESS) {
      return xmma::Error::ERROR_ILWALID_PARAMS;
    }

    if( callwlate_gemm_params() != xmma::Error::SUCCESS) {
      return xmma::Error::ERROR_ILWALID_PARAMS;
    }

    // Enable beta load or not
    if ( this->beta > 0.0f ) {
      this->has_beta = true;
    } else {
      this->has_beta = false;
    }

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

    return xmma::Error::SUCCESS;
  }

protected:

protected:
  xmma::Error callwlate_splitk_params();
  xmma::Error callwlate_gemm_params();

};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Traits, typename Cta_tile, int STAGE>
xmma::Error Xmma_sparse_gemm_params<Traits, Cta_tile, STAGE>::callwlate_splitk_params()
{
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

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Traits, typename Cta_tile, int STAGE>
xmma::Error Xmma_sparse_gemm_params<Traits, Cta_tile, STAGE>::callwlate_gemm_params()
{
  this->sparse_factor = 2;

  int uint16_t_b = 16; //uint16 is a pack for id2
  int id2_b = 2; // Id2 is 2 bits
  int min_cta_k = 64;  // Ideal min k size (LWRRENTLY!)
  int num_meta = uint16_t_b / id2_b;  // Number of metadata elements in a unit16_t
  // Colwert the CTA K dimension to metadata K dimension
  int meta_k_size = (min_cta_k / this->sparse_factor) / num_meta;

  int meta_k = this->k / this->sparse_factor;
  meta_k = (meta_k * 2 + uint16_t_b - 1) / uint16_t_b;

  if((meta_k % meta_k_size) != 0){
      //printf("meta_k = %d is not multiple of 4\n", meta_k);
      // Pad the meta K to ideal size, now multiple of 4 based on CTA K = 64
      meta_k = meta_k_size * ((meta_k + meta_k_size - 1) / meta_k_size);
  }
  this->meta_k_pad = meta_k;

  // The number of elements from GEMM-k that we consume in one iteration of the loop.
  const int k_per_iteration = this->split_k.slices * Cta_tile::K;

  // The pointer update for A (keep in mind that we transpose A).
  int delta_a;
  if( this->ta ) {
    delta_a = k_per_iteration;
  } else {
    delta_a = k_per_iteration * this->lda;
  }
  this->a_delta = Traits::offset_in_bytes_a(delta_a) / this->sparse_factor;

  // The pointer update for B (keep in mind that we transpose B).
  int delta_b;
  if( this->tb ) {
    delta_b = k_per_iteration * this->ldb;
  } else {
    delta_b = k_per_iteration;
  }
  this->b_delta = Traits::offset_in_bytes_b(delta_b);

  // e_delta is the gmem offset while moving along kBlock
  // Now metadata is reordered and stored linearly, however we can still move the gmem potinter accorinding to K block iteration
  // That's why I multiply m by half_K (and divide by 8 to colwer to metadata k length)
  // 8 here is to colwent to metadata element
  int pad_m = (this->m % 64 != 0) ? ((this->m / 64) * 64 + 64) : this->m;
  //this->e_delta = Traits::offset_in_bytes_e(this->split_k.slices * pad_m * Cta_tile::K / 2 / 8);
  this->e_delta = this->split_k.slices * pad_m * Cta_tile::K / 2 / 16 * 4;

  // The number of loop iterations to cover C elements.
  int loop_count_k = xmma::div_up(this->k, k_per_iteration);
  // The first iteration of the loop.
  this->loop_start = loop_count_k - 1;
  // The iteration where we trigger the residue.
  this->loop_residue = this->ampere ? STAGE : 2;
  // The number of elements read when we enter the residue.
  this->loop_residue_k = (loop_count_k - 1) * k_per_iteration;

  return xmma::Error::SUCCESS;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
}
} // namespace gemm
} // namespace ext
} // namespace xmma
