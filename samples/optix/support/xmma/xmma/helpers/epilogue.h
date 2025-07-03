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

#include <xmma/fragment.h>
#include <xmma/named_barrier.h>
#include <xmma/smem_tile.h>
#include <xmma/warp_masks.h>
#include <xmma/arrive_wait.h>

namespace xmma {
namespace helpers {
    
template<
    // The instruction traits.
    typename Traits, 
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The layout.
    typename Layout = xmma::Row,
    // The number of bytes per STG.
    int BYTES_PER_STG_ = 16
>
struct Gmem_tile_epilogue_distribution {

    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;

    // The number of bytes per STG.
    enum { BYTES_PER_STG = BYTES_PER_STG_ };
    // The number of elements per STG per thread.
    enum { ELEMENTS_PER_STG = BYTES_PER_STG * 8 / Traits::BITS_PER_ELEMENT_C };
    // The number of threads per output row 
    enum { THREADS_PER_ROW = Min<Cta_tile::THREADS_PER_CTA,
        (Layout::ROW ? Cta_tile::N : Cta_tile::M)  / ELEMENTS_PER_STG>::VALUE };
    // The number of rows that are written with a single STG (accross the CTA).
    enum { ROWS_PER_STG = Cta_tile::THREADS_PER_CTA / THREADS_PER_ROW };
    // Select the number of rows per XMMA
    enum { ROWS_PER_XMMA_PER_CTA = Layout::ROW ? Xmma_tile::M_PER_XMMA_PER_CTA 
                                               : Xmma_tile::N_PER_XMMA_PER_CTA };

    // Compute the row.
    static inline __device__ int compute_col(int tidx) {
        return tidx % THREADS_PER_ROW;
    }

    // Compute the row.
    static inline __device__ int compute_row(int tidx) {
        return tidx / THREADS_PER_ROW;
    }

    // Compute the row offset.
    static inline __device__ constexpr int compute_offset(int mi, int ii) {
        return mi * ROWS_PER_XMMA_PER_CTA + ii * ROWS_PER_STG;
    }   

};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    // The instruction traits.
    typename Traits,
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The layout.
    typename Layout = xmma::Row,
    // The number of bytes per STG.
    int BYTES_PER_STG_ = 16>
struct Gmem_tile_gmma_epilogue_distribution {

    // To help you understand the distribution, please refer to spreadsheet in
    // https://gitlab-master.lwpu.com/jiliu/document/-/raw/master/GMMA/
    // HGMMA_epilogue_fp16acc_fp16out_rowOut_0815.xlsx

    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;

    // Bytes per element
    enum { BYTES_PER_ELEMENT = Traits::BITS_PER_ELEMENT_C / 8 };
    // The number of bytes per STG.
    enum { BYTES_PER_STG = BYTES_PER_STG_ };
    // The number of elements per STG per thread.
    enum { ELEMENTS_PER_STG = BYTES_PER_STG * 8 / Traits::BITS_PER_ELEMENT_C };
    // The number of column loaded per STG
    // There are always 8 threads per row for STG in epilogue
    enum { COLUMNS_PER_STG = BYTES_PER_STG * 8 / BYTES_PER_ELEMENT };
    enum { MIN_TILE_N = Cta_tile::N < COLUMNS_PER_STG ? Cta_tile::N : COLUMNS_PER_STG };
    // tile_m is limited such that every thread can participate, 8 rows per warp
    enum { M_PER_WARP = 8 };
    enum { TILE_M = 8 * Cta_tile::THREADS_PER_CTA / 32, TILE_N = MIN_TILE_N };
    // The number of threads per output row
    enum { THREADS_PER_ROW = TILE_N * BYTES_PER_ELEMENT / BYTES_PER_STG };
    // the number of rows per STG instruction by one warp.
    enum { ROWS_PER_STG_PER_WARP = Cta_tile::THREADS_PER_WARP / THREADS_PER_ROW };
    // Select the number of rows per XMMA
    enum { ROWS_PER_XMMA_PER_CTA = Xmma_tile::M_PER_XMMA_PER_CTA };

    // Compute the row.
    static inline __device__ int compute_col( int tidx ) {
        int lane = tidx % Cta_tile::THREADS_PER_WARP;
        return lane % THREADS_PER_ROW;
    }

    // Compute the row.
    static inline __device__ int compute_row( int tidx ) {
        int lane = tidx % Cta_tile::THREADS_PER_WARP;
        int warp_m = tidx / Cta_tile::THREADS_PER_WARP;
        return lane / THREADS_PER_ROW + warp_m * 16;
    }

    // Compute the row offset.
    static inline __device__ constexpr int compute_offset( int xmmas_mi, int mi, int ii ) {
        return xmmas_mi * ROWS_PER_XMMA_PER_CTA + mi * M_PER_WARP + ii * ROWS_PER_STG_PER_WARP;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    // The instruction traits.
    typename Traits,
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The number of bytes per STG.
    int BYTES_PER_STG_>
struct Gmem_tile_gmma_epilogue_distribution<Traits, Cta_tile, xmma::Col, BYTES_PER_STG_> {

    // To help you understand the distribution, please refer to spreadsheet in
    // https://gitlab-master.lwpu.com/jiliu/document/-/blob/master/GMMA/
    // HGMMA_epilogue_fp16acc_fp16out_colOut_0815.xlsx

    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;

    // Bytes per element
    enum { BYTES_PER_ELEMENT = Traits::BITS_PER_ELEMENT_C / 8 };
    // The number of bytes per STG.
    enum { BYTES_PER_STG = BYTES_PER_STG_ };
    // The number of elements per STG per thread.
    enum { ELEMENTS_PER_STG = BYTES_PER_STG * 8 / Traits::BITS_PER_ELEMENT_C };
    // CTA tile size
    enum { CTA_M = Cta_tile::M, CTA_N = Cta_tile::N };
    // Bytes per column
    enum { BYTES_PER_COLUMN = CTA_M * BYTES_PER_ELEMENT };
    // The number of threads
    enum { THREADS_PER_CTA = Cta_tile::THREADS_PER_CTA };
    // Threads for STG per column
    enum { STG_THREADS_PER_COLUMN = CTA_M * BYTES_PER_ELEMENT / BYTES_PER_STG };
    static_assert( STG_THREADS_PER_COLUMN >= 8,
                   "STG_THREADS_PER_COLUMN should be larger than 8\n" );
    // the number of columns can be store by all threads per STG instruction
    enum { COLUMNS_PER_STG = THREADS_PER_CTA / STG_THREADS_PER_COLUMN };
    // we can probably reduce the tile M to MIN_TILE_M, but for simplicity we set tile_M = cta_M
    enum { TILE_M = CTA_M, TILE_N = COLUMNS_PER_STG < 8 ? 8 : COLUMNS_PER_STG };
    // the min tile in N dim is 8 such that every thread can participate in sts
    static_assert( TILE_N % 8 == 0, "TILE_N should be multiple of 8" );
    // Select the number of rows per XMMA
    enum { COLS_PER_XMMA_PER_CTA = Xmma_tile::N_PER_XMMA_PER_CTA };

    // Compute the row.
    static inline __device__ int compute_col( int tidx ) {
        return tidx / STG_THREADS_PER_COLUMN;
    }

    // Compute the row.
    static inline __device__ int compute_row( int tidx ) {
        return ( tidx % STG_THREADS_PER_COLUMN ) * ELEMENTS_PER_STG;
    }

    // Compute the col offset.
    static inline __device__ constexpr int compute_offset( int xmmas_ni, int ni, int ii ) {
        return xmmas_ni * COLS_PER_XMMA_PER_CTA + ni * TILE_N + ii * COLUMNS_PER_STG;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// Load the data from global memory to shared memory
template< typename Implicit_gemm_traits,
          typename Gmem_tile_epilogue,
          typename Smem_tile_a,
          typename Smem_tile_b       
>
inline __device__ void warp_specialized_early_load(
              Gmem_tile_epilogue gmem_epilogue,
              Smem_tile_a &smem_tile_a,
              Smem_tile_b &smem_tile_b,
              int &buffer_head,
              xmma::Arrive_wait &buffer_empty,
              xmma::Arrive_wait &buffer_full,
              int &cnt,
              unsigned int &lwrrent_phase_buffer_empty,
              int tidx,
              uint64_t mem_desc = MEM_DESC_DEFAULT) {
    // The CTA tile.
    using Cta_tile = typename Implicit_gemm_traits::Cta_tile; 
    // The XMMA tile.
    using Xmma_tile = typename Implicit_gemm_traits::Xmma_tile;    

    const int BYTES_PER_XMMA = Cta_tile::THREADS_PER_CTA * 
                               Gmem_tile_epilogue::BYTES_PER_STG;
    // Number of xmma_tiles(of all threads) a_smem buffer can hold rounded down.
    const int xmma_tiles_per_a = (Smem_tile_a::BYTES_PER_TILE / 
                                  Implicit_gemm_traits::BUFFERS_PER_SMEM_TILE_A) /
                                  BYTES_PER_XMMA;                      
    // Number of xmma_tiles(of all threads) b_smem buffer can hold rounded down.
    const int xmma_tiles_per_b = (Smem_tile_b::BYTES_PER_TILE / 
                                  Implicit_gemm_traits::BUFFERS_PER_SMEM_TILE_A) /
                                  BYTES_PER_XMMA;                             
    unsigned int phase_bit;                                    
    int xmma_tile_counter = 0;

    #pragma unroll
    for( int mi = 0; mi < Xmma_tile::XMMAS_M; ++mi )
    {
      int out_masks[Gmem_tile_epilogue::STGS];
      #pragma unroll
      for( int ii = 0; ii < Gmem_tile_epilogue::STGS; ++ii )
      {
          out_masks[ii] = gmem_epilogue.compute_output_mask(mi, ii);
      }

      #pragma unroll
      for( int ii = 0; ii < Gmem_tile_epilogue::STGS; ++ii )
      {    
          //Wait on a new A/B buffer 
          if (xmma_tile_counter == 0 && cnt >= Implicit_gemm_traits::BUFFERS_PER_SMEM_TILE_A)
          {
              phase_bit = ( lwrrent_phase_buffer_empty >> buffer_head ) & 1;
              buffer_empty.bar_wait( buffer_head, phase_bit );
              lwrrent_phase_buffer_empty ^= ( 1 << buffer_head ) ^ ( 0 );                        
          }
          //Load data for (mi,ii)th tile
          gmem_epilogue.load_residual_to_smem(
                                      smem_tile_a,
                                      smem_tile_b,
                                      mi, 
                                      ii, 
                                      out_masks[ii],
                                      xmma_tiles_per_a,
                                      xmma_tiles_per_b,
                                      tidx,
                                      mem_desc);
          //Increment (mi,ii) counter
          xmma_tile_counter++;
         
          //One A/B buffer is fully filled, move on to the next.
          if (xmma_tile_counter == xmma_tiles_per_a + xmma_tiles_per_b)
          {
             xmma_tile_counter = 0;
             buffer_full.bar_arrive_ldgsts( buffer_head );
             buffer_head = ( buffer_head < Implicit_gemm_traits::BUFFERS_PER_SMEM_TILE_A - 1 )
                               ? ( buffer_head + 1 )
                               : 0;
             smem_tile_a.move_next_write_buffer();
             smem_tile_b.move_next_write_buffer();    
             cnt++;                        
          }   
      }// ii
    }//mi
   
   /*
   Edge case when xmma_tiles_per_c cannot be divided by
   (xmma_tiles_per_a + xmma_tiles_per_b)
   */
   if (xmma_tile_counter != 0)
   {
     buffer_full.bar_arrive_ldgsts( buffer_head );
     buffer_head = ( buffer_head < Implicit_gemm_traits::BUFFERS_PER_SMEM_TILE_A - 1 )
                       ? ( buffer_head + 1 )
                       : 0;
     smem_tile_a.move_next_write_buffer();
     smem_tile_b.move_next_write_buffer();    
     cnt++;                 
  }

}//warp_specialized_early_load


////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    // The instruction traits.
    typename Traits, 
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile, 
    // The layout.
    typename Layout = xmma::Row,
    // The fragment class before writing data to global memory.
    typename Fragment_c = xmma::Fragment_c<Traits, Cta_tile>
>
struct Gmem_tile_epilogue {    
    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;
    // The helper class to compute the row offset (in the tile).
    using Tile_distribution = Gmem_tile_epilogue_distribution<Traits, Cta_tile, Layout>;

    // The number of bytes per STG.
    enum { BYTES_PER_STG = 16 };
    // The number of elements per STG per thread.
    enum { ELEMENTS_PER_STG = BYTES_PER_STG * 8 / Traits::BITS_PER_ELEMENT_C };
    // The number of elements per row.
    enum { ELEMENTS_PER_ROW = Layout::ROW ? Cta_tile::N : Cta_tile::M };
    // The number of threads needed to store a row.
    enum { THREADS_PER_ROW = ELEMENTS_PER_ROW / ELEMENTS_PER_STG };
    // The number of rows that are written with a single STG (accross the CTA).
    enum { ROWS_PER_STG = Cta_tile::THREADS_PER_CTA / THREADS_PER_ROW };
    // The number of rows to store per XMMA per CTA.
    enum { ROWS_PER_XMMA_PER_CTA = Layout::ROW ? Xmma_tile::M_PER_XMMA_PER_CTA 
                                               : Xmma_tile::N_PER_XMMA_PER_CTA };
    // The number of STGs needed to store the elements per iteration.
    enum { STGS = ROWS_PER_XMMA_PER_CTA / ROWS_PER_STG };

    // Ctor.
    inline __device__ Gmem_tile_epilogue(int m, int n, int stride_n)
        : params_m_(m)
        , params_n_(n)
        , params_stride_n_(stride_n)
        ,tidx_(threadIdx.x % Cta_tile::THREADS_PER_CTA) {
    }

    // Ctor.
    inline __device__ Gmem_tile_epilogue(
        int m, int n, int stride_n, char *out_ptr, const char* res_ptr,
        int bidm, int bidn, int bidz, int tidx)
        : params_m_(m)
        , params_n_(n)
        , params_stride_n_(stride_n)
        , tidx_(tidx) {

        // The location of the tile.
        int row = Tile_distribution::compute_row(tidx);
        int col = Tile_distribution::compute_col(tidx);

        // Compute the output position for each thread.
        if( Layout::ROW ) {
            m_ = bidm * Cta_tile::M + row;
            n_ = bidn * Cta_tile::N + col * ELEMENTS_PER_STG;
        } else {
            m_ = bidn * Cta_tile::N + row;
            n_ = bidm * Cta_tile::M + col * ELEMENTS_PER_STG;

        }        
        // The pointer.
        const int64_t offset = Traits::offset_in_bytes_c(m_*params_stride_n_ + n_);
        out_ptr_ = &out_ptr[offset];
        res_ptr_ = &res_ptr[offset];
    }


    // Is a given output valid?
    inline __device__ int compute_output_mask(int mi, int ii) const {
        const int offset = Tile_distribution::compute_offset(mi, ii);
        return (m_ + offset) < params_m_ && n_ < params_n_;
    }

    // Load the data from global memory.
    inline __device__ void load(Fragment_c &data, 
                                int mi, 
                                int ii, 
                                int mask,
                                uint64_t mem_desc = MEM_DESC_DEFAULT) {
        const int offset = Tile_distribution::compute_offset(mi, ii);
        const char *ptr = res_ptr_ + Traits::offset_in_bytes_c(offset*params_stride_n_);
        if( mask ) {
            uint4 tmp;
            xmma::ldg(tmp, ptr, mem_desc);
            data.from_int4(tmp);
        }
    }
    
    // Load residual from gmem to smem buffers.
    template<typename Smem_tile_a, typename Smem_tile_b>
    inline __device__ void load_residual_to_smem(
                                Smem_tile_a &smem_tile_a,
                                Smem_tile_b &smem_tile_b,
                                int mi, 
                                int ii, 
                                int mask,
                                int xmma_tiles_per_a,
                                int xmma_tiles_per_b,
                                int tidx,
                                uint64_t mem_desc = MEM_DESC_DEFAULT) {
        const int offset = Tile_distribution::compute_offset(mi, ii);
        const char *ptr = res_ptr_ + Traits::offset_in_bytes_c(offset*params_stride_n_);
        if( mask ) {
          
          int xmma_tile_idx = (mi * STGS + ii) % (xmma_tiles_per_a + xmma_tiles_per_b);
          uint32_t smem_ptr;
          
          if (xmma_tile_idx < xmma_tiles_per_a) 
            smem_ptr = smem_tile_a.smem_ + smem_tile_a.smem_write_buffer_ + 
            xmma_tile_idx *  Cta_tile::THREADS_PER_CTA * BYTES_PER_STG + tidx *BYTES_PER_STG;
          else
            smem_ptr = smem_tile_b.smem_ + smem_tile_b.smem_write_buffer_ + 
            (xmma_tile_idx - xmma_tiles_per_a) *  Cta_tile::THREADS_PER_CTA * BYTES_PER_STG + 
            tidx * BYTES_PER_STG;

            ldgsts128(smem_ptr, ptr, true, mem_desc);
        }      
    }
    
    // Store the data to global memory.
    inline __device__ void store(int mi, 
                                 int ii, 
                                 const Fragment_c &data, int mask,
                                 uint64_t mem_desc = MEM_DESC_DEFAULT) {
        const int offset = Tile_distribution::compute_offset(mi, ii);
        char *ptr = out_ptr_ + Traits::offset_in_bytes_c(offset*params_stride_n_);
        if( mask ) {
            xmma::stg(ptr, data.to_int4(), mem_desc);
        }
    }

    // The dimensions of the matrix.
    const int params_m_, params_n_, params_stride_n_;
    // The position of the tile.
    int m_, n_;
    // The pointer to global memory.
    char *out_ptr_;
    const char *res_ptr_;
    // thread index
    const int tidx_;
};

template<
    // The instruction traits.
    typename Traits,
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The layout.
    typename Layout = xmma::Row,
    // The fragment class before writing data to global memory.
    typename Fragment_c = xmma::Fragment_c<Traits, Cta_tile>
>
struct Gmem_tile_wo_smem_epilogue {
    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;

    // The number of bytes per STG.
    enum { BYTES_PER_STG = 16 };
    // The number of elements per STG per thread.
    enum { ELEMENTS_PER_STG = BYTES_PER_STG * 8 / Traits::BITS_PER_ELEMENT_C };
    // The number of elements per row.
    enum { ELEMENTS_PER_ROW = Layout::ROW ? Cta_tile::N : Cta_tile::M };
    // The number of threads needed to store a row.
    enum { THREADS_PER_ROW = ELEMENTS_PER_ROW / ELEMENTS_PER_STG };
    // The number of rows that are written with a single STG (accross the CTA).
    enum { ROWS_PER_STG = Cta_tile::THREADS_PER_CTA / THREADS_PER_ROW };
    // The number of rows to store per XMMA per CTA.
    enum { ROWS_PER_XMMA_PER_CTA = Layout::ROW ? Xmma_tile::M_PER_XMMA_PER_CTA 
                                               : Xmma_tile::N_PER_XMMA_PER_CTA };
    // The number of STGs needed to store the elements per iteration.
    enum { STGS = ROWS_PER_XMMA_PER_CTA / ROWS_PER_STG };

    // Ctor.
    inline __device__ Gmem_tile_wo_smem_epilogue(int m, int n, int stride_n)
        : params_m_(m)
        , params_n_(n)
        , params_stride_n_(stride_n) {
    }

    // Ctor.
    inline __device__ Gmem_tile_wo_smem_epilogue(
        int m, int n, int stride_n, char *out_ptr, const char* res_ptr,
        int bidm, int bidn, int bidz, int tidx)
        : params_m_(m)
        , params_n_(n)
        , params_stride_n_(stride_n) {

        // The number of warps.
        const int WARPS_M = Cta_tile::WARPS_M;
        const int WARPS_N = Cta_tile::WARPS_N;
        const int WARPS_K = Cta_tile::WARPS_K;

        // The masks to select the warps.
        const int WARP_MASK_M = xmma::Warp_masks<WARPS_M, WARPS_N, WARPS_K>::M;
        const int WARP_MASK_N = xmma::Warp_masks<WARPS_M, WARPS_N, WARPS_K>::N;

        // The divisor for the warps.
        const int WARP_DIV_M =       1 *       1 * Cta_tile::THREADS_PER_WARP;
        const int WARP_DIV_N = WARPS_M *       1 * Cta_tile::THREADS_PER_WARP;

        // The location of the tile.
        // The second part after addition seems only work for f64.
        // Need double check if want supporting other type.
        int row = ((tidx & WARP_MASK_M) / WARP_DIV_M) * Xmma_tile::M_PER_XMMA
            + (tidx % 32) / 4;
        int col = ((tidx & WARP_MASK_N) / WARP_DIV_N) * Xmma_tile::N_PER_XMMA
            + (tidx % 4) * 2;

        // Compute the output position for each thread.
        m_ = bidm * Cta_tile::M + row;
        n_ = bidn * Cta_tile::N + col;
        // The pointer.
        const int64_t offset = Traits::offset_in_bytes_c(m_*params_stride_n_ + n_);
        out_ptr_ = &out_ptr[offset];
        res_ptr_ = &res_ptr[offset];
    }


    // Is a given output valid?
    inline __device__ int compute_output_mask(int mi, int ni) const {
        const int row = mi * Xmma_tile::M_PER_XMMA_PER_CTA;
        const int col = ni * Xmma_tile::N_PER_XMMA_PER_CTA;
        return m_ + row < params_m_ && n_ + col < params_n_;
    }

    inline __device__ int compute_output_offset(int mi, int ni) const {
        const int row = mi * Xmma_tile::M_PER_XMMA_PER_CTA;
        const int col = ni * Xmma_tile::N_PER_XMMA_PER_CTA;
        const int offset = row * params_stride_n_ + col;
        return offset;
    }


    // Load the data from global memory.
    inline __device__ void load(Fragment_c &data,
                                int mi,
                                int ni,
                                uint64_t mem_desc = MEM_DESC_DEFAULT) {
        const int mask = compute_output_mask(mi, ni);
        const int offset = compute_output_offset(mi, ni);

        const char *ptr = res_ptr_ + Traits::offset_in_bytes_c(offset);
        if( mask ) {
            uint4 tmp;
            xmma::ldg(tmp, ptr, mem_desc);
            data.from_int4(tmp);
        }
    }
    
    // Load residual from gmem to smem buffers.
    template<typename Smem_tile_a, typename Smem_tile_b>
    inline __device__ void load_residual_to_smem(
                                Smem_tile_a &smem_tile_a,
                                Smem_tile_b &smem_tile_b,
                                int mi, 
                                int ii, 
                                int mask,
                                int xmma_tiles_per_a,
                                int xmma_tiles_per_b,
                                int tidx,
                                uint64_t mem_desc = MEM_DESC_DEFAULT) {
        const int mask_ = compute_output_mask(mi, ii);
        const int offset = compute_output_offset(mi, ii);

        const char *ptr = res_ptr_ + Traits::offset_in_bytes_c(offset);
        if( mask_ ) {
          int xmma_tile_idx = (mi * STGS + ii) % (xmma_tiles_per_a + xmma_tiles_per_b);
          uint32_t smem_ptr;
          
          if (xmma_tile_idx < xmma_tiles_per_a) 
            smem_ptr = smem_tile_a.smem_ + smem_tile_a.smem_write_buffer_ + 
            xmma_tile_idx *  Cta_tile::THREADS_PER_CTA * BYTES_PER_STG + tidx *BYTES_PER_STG;
          else
            smem_ptr = smem_tile_b.smem_ + smem_tile_b.smem_write_buffer_ + 
            (xmma_tile_idx - xmma_tiles_per_a) *  Cta_tile::THREADS_PER_CTA * BYTES_PER_STG + 
            tidx * BYTES_PER_STG;

            ldgsts128(smem_ptr, ptr, true, mem_desc);
        }                            
    }
    
    // Store the data to global memory.
    inline __device__ void store(int mi,
                                 int ni,
                                 const Fragment_c &data,
                                 uint64_t mem_desc = MEM_DESC_DEFAULT) {

        const int mask = compute_output_mask(mi, ni);
        const int offset = compute_output_offset(mi, ni);

        char *ptr = out_ptr_ + Traits::offset_in_bytes_c(offset);
        if( mask ) {
            xmma::stg(ptr, data.to_int4(), mem_desc);
        }
    }
    
    // The dimensions of the matrix.
    const int params_m_, params_n_, params_stride_n_;
    // The position of the tile.
    int m_, n_;
    // The pointer to global memory.
    char *out_ptr_;
    const char *res_ptr_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Traits, typename Cta_tile, typename Fragment_c, bool = (Cta_tile::WARPS_K > 1) >
struct Gmem_tile_epilogue_interleaved {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    // The instruction traits.
    typename Traits, 
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile, 
    // The fragment class before writing data to shared memory for swizzling.
    typename Fragment_pre_swizzle_ = xmma::Fragment_epilogue_pre_swizzle<Traits, Cta_tile>,
    // The fragment class after reading data from shared memory for swizzling.
    typename Fragment_post_swizzle_ = xmma::Fragment_epilogue_post_swizzle<Traits, Cta_tile>,
    // The fragment class before writing data to global memory.
    typename Fragment_c_ = xmma::Fragment_c<Traits, Cta_tile>
>
struct Empty_callbacks_epilogue {

    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;
    // The aclwmulators.
    using Fragment_aclwmulator = xmma::Fragment_aclwmulator<Traits>;
    // The fragment before swizzling through shared memory.
    using Fragment_pre_swizzle = Fragment_pre_swizzle_;
    // The fragment after swizzling through shared memory.
    using Fragment_post_swizzle = Fragment_post_swizzle_;
    // The fragment before writing to global memory.
    using Fragment_c = Fragment_c_;
    // The alpha fragment before swizzle
    using Fragment_alpha_pre_swizzle = typename Traits::Epilogue_type;
    // The alpha fragment after swizzle
    using Fragment_alpha_post_swizzle = typename Traits::Epilogue_type;
    // The beta fragment
    using Fragment_beta = typename Traits::Epilogue_type;

    // This callback objetc needs 0 shared memory.
    enum { BYTES_PER_TILE = 0 };

    // Ctor.
    template< typename Params >
    inline __device__ Empty_callbacks_epilogue(const Params &params, void*, int, int, int, int) {
        alpha_ = colwert<typename Traits::Epilogue_type>(params.alpha);
        beta_  = colwert<typename Traits::Epilogue_type>(params.beta);
    }

    // A callback function to get alpha.
    template< typename Epilogue >
    inline __device__ void alpha_pre_swizzle(Epilogue&, int, int, Fragment_alpha_pre_swizzle &f) {
        f = this->alpha_;
    }

    // A callback function to get alpha.
    template< typename Epilogue >
    inline __device__ void alpha_post_swizzle(Epilogue&, int, int, Fragment_alpha_post_swizzle &f) {
        f = this->alpha_;
    }

    // A callback function to get beta.
    template< typename Epilogue >
    inline __device__ void beta(Epilogue&, int, int, Fragment_beta &f) {
        f = this->beta_;
    }

    // Pre epilogue.
    inline __device__ void pre_epilogue() {
    }

    // Pre-swizzle on aclwmulators.
    template< typename Epilogue >
    inline __device__ void pre_colwert(Epilogue&, int, int, Fragment_aclwmulator&) {
    }

    // Pre-swizzle on the fragment with elements colwerted to the epilogue type.
    template< typename Epilogue >
    inline __device__ void pre_swizzle(Epilogue&, int, int, const Fragment_pre_swizzle&) {
    }

    // Post swizzle.
    template< typename Epilogue >
    inline __device__ void post_swizzle(Epilogue&, int, int, const Fragment_post_swizzle&, int) {
    }

    // Before packing to Fragment_c.
    template< typename Epilogue >
    inline __device__ void pre_pack(Epilogue&, int, int, Fragment_post_swizzle&) {
    }

    // Before storing to global memory.
    template< typename Epilogue >
    inline __device__ void pre_store(Epilogue&, int, int, const Fragment_c&, int) {
    }

    // Post epilogue.
    inline __device__ void post_epilogue() {
    }

    // Register swizzeling
    template< typename Fragment_pre_swizzle, typename Fragment_post_swizzle >
    inline __device__ void reg_swizzle(Fragment_pre_swizzle &, Fragment_post_swizzle &) {
    }

    // Global alpha/beta.
    typename Traits::Epilogue_type alpha_, beta_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    // The instruction traits.
    typename Traits, 
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile, 
    // The fragment class before writing data to shared memory for swizzling.
    typename Fragment_pre_swizzle_ = xmma::Fragment_epilogue_pre_swizzle<Traits, Cta_tile>,
    // The fragment class after reading data from shared memory for swizzling.
    typename Fragment_post_swizzle_ = xmma::Fragment_epilogue_post_swizzle<Traits, Cta_tile>,
    // The fragment class before writing data to global memory.
    typename Fragment_c_ = xmma::Fragment_c<Traits, Cta_tile>
>
struct Empty_callbacks_epilogue_with_per_channel_alpha_beta
    : public Empty_callbacks_epilogue<Traits, 
                                      Cta_tile, 
                                      Fragment_pre_swizzle_, 
                                      Fragment_post_swizzle_, 
                                      Fragment_c_> {
    // The base class.
    using Base = Empty_callbacks_epilogue<Traits, 
                                          Cta_tile, 
                                          Fragment_pre_swizzle_, 
                                          Fragment_post_swizzle_, 
                                          Fragment_c_>;
    // The alpha fragment before swizzle.
    using Fragment_alpha_pre_swizzle = xmma::Fragment<float,
                                                          Base::Fragment_pre_swizzle::NUM_ELTS>;
    // The alpha fragment after swizzle.
    using Fragment_alpha_post_swizzle = xmma::Fragment<float,
                                                           Base::Fragment_post_swizzle::NUM_ELTS>;
    // The beta fragment.
    using Fragment_beta = xmma::Fragment<float, Base::Fragment_post_swizzle::NUM_ELTS>;

    // Ctor.
    template< typename Params >
    inline __device__ Empty_callbacks_epilogue_with_per_channel_alpha_beta(const Params &params, 
                                                                           char *smem, 
                                                                           int bidm, 
                                                                           int bidn, 
                                                                           int bidz, 
                                                                           int tidx) 
        : Base(params, smem, bidm, bidn, bidz, tidx) {
    }

    // A callback function to get alpha.
    template< typename Epilogue >
    inline __device__ void alpha_pre_swizzle(Epilogue&, int, int, Fragment_alpha_pre_swizzle &f) {
        #pragma unroll
        for( int ii = 0; ii < Fragment_alpha_pre_swizzle::NUM_ELTS; ++ii ) {
            f.elt(ii) = this->alpha_;
        }
    }

    // A callback function to get alpha.
    template< typename Epilogue >
    inline __device__ void alpha_post_swizzle(Epilogue&, int, int, Fragment_alpha_post_swizzle &f) {
        #pragma unroll
        for( int ii = 0; ii < Fragment_alpha_post_swizzle::NUM_ELTS; ++ii ) {
            f.elt(ii) = this->alpha_;
        }
    }

    // A callback function to get beta.
    template< typename Epilogue >
    inline __device__ void beta(Epilogue&, int, int, Fragment_beta &f) {
        #pragma unroll
        for( int ii = 0; ii < Fragment_beta::NUM_ELTS; ++ii ) {
            f.elt(ii) = this->beta_;
        }
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    // The instruction traits.
    typename Traits_,
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile_,
    // The layout of the tile.
    typename Layout_,
    // The global memory tile to store the output.
    typename Gmem_tile_,
    // The callbacks to lwstomize the behaviour of the epilogue.
    typename Callbacks = Empty_callbacks_epilogue<Traits_, Cta_tile_>,
    // The class to swizzle the data.
    typename Swizzle_ = xmma::Swizzle_epilogue<Traits_, Cta_tile_, Layout_, Gmem_tile_::BYTES_PER_STG>,
    // The fragment class before the swizzling.
    typename Fragment_pre_swizzle_ = typename Callbacks::Fragment_pre_swizzle,
    // The fragment class after the swizzling.
    typename Fragment_post_swizzle_ = typename Callbacks::Fragment_post_swizzle,
    // The output fragment.
    typename Fragment_c_ = typename Callbacks::Fragment_c
>
struct Epilogue {
    // The instruction traits.
    using Traits = Traits_;
    // The dimensions of the tile computed by the CTA.
    using Cta_tile = Cta_tile_;
    // The layout of the tile.
    using Layout = Layout_;
    // The global memory tile to store the output.
    using Gmem_tile = Gmem_tile_;
    // The class to swizzle the data.
    using Swizzle = Swizzle_;

    // The fragment class before the swizzling.
    using Fragment_pre_swizzle = Fragment_pre_swizzle_;
    // The fragment class after the swizzling.
    using Fragment_post_swizzle = Fragment_post_swizzle_;
    // The output fragment.
    using Fragment_c = Fragment_c_;

    // The fragment for alpha (used before swizzling).
    using Fragment_alpha_pre_swizzle = typename Callbacks::Fragment_alpha_pre_swizzle;
    // The fragment for alpha (used after swizzling).
    using Fragment_alpha_post_swizzle = typename Callbacks::Fragment_alpha_post_swizzle;
    // The fragment for beta.
    using Fragment_beta = typename Callbacks::Fragment_beta;

    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;

    // Ctor.
    template< typename Params >
    inline __device__ Epilogue(const Params &params,
                               Gmem_tile &gmem_tile,
                               Swizzle &swizzle,
                               Callbacks &callbacks,
                               const Named_barrier &epi_sync = Named_barrier(),
                               const int bidm = blockIdx.x,
                               const int bidn = blockIdx.y,
                               const int bidz = blockIdx.z,
                               const int tidx = threadIdx.x,
                               const bool is_warp_specialized = false)
        : gmem_tile_(gmem_tile)
        , swizzle_(swizzle)
        , callbacks_(callbacks)
        , epi_sync_(epi_sync)
        , mem_desc_c_(params.mem_descriptors.descriptor_c)
        , mem_desc_d_(params.mem_descriptors.descriptor_d)
        , tidx_(tidx) {
    }

    // Do the epilogue.
    template< bool WITH_RESIDUAL, typename Fragment_aclwmulator, int M, int N >
    inline __device__ void execute(Fragment_aclwmulator (&acc)[M][N]) {

        #pragma unroll
        for( int mi = 0; mi < M; ++mi ) {
            //this->step<WITH_RESIDUAL>(mi, acc[mi]);

            // The output masks.
            int32_t out_masks[Gmem_tile::STGS];
            callwlate_out_mask(out_masks, mi);

            // Load valid values if beta is not zero.
            Fragment_c res_fetch[Gmem_tile::STGS];
            if( WITH_RESIDUAL ) {
                #pragma unroll
                for( int ii = 0; ii < Gmem_tile::STGS; ++ii ) {
                    this->gmem_tile_.load(res_fetch[ii], mi, ii, out_masks[ii], mem_desc_c_);
                }
            }

            // Colwert aclwmulator to post swizzle fragment;
            Fragment_post_swizzle post_swizzle[Gmem_tile::STGS];
            Fragment_alpha_post_swizzle alpha_post_swizzle[Gmem_tile::STGS];
            acc_to_post_swizzle(post_swizzle, alpha_post_swizzle, acc[mi], out_masks, mi);

            // Load beta. TODO: We should not need a loop.
            Fragment_beta beta[Gmem_tile::STGS];
            if( WITH_RESIDUAL ) {
                #pragma unroll
                for( int ii = 0; ii < Gmem_tile::STGS; ++ii ) {
                    callbacks_.beta(*this, mi, ii, beta[ii]);
                }
            }

            // Add the residual value before packing. TODO: We should be able to pass a single beta.
            if( WITH_RESIDUAL ) {
                #pragma unroll
                for( int ii = 0; ii < Gmem_tile::STGS; ++ii ) {
                    post_swizzle[ii].add_residual(res_fetch[ii], beta[ii]);
                }
            }

            // Colwert post swizzle fragment to output fragment.
            Fragment_c out_regs[Gmem_tile::STGS];
            post_swizzle_to_output(out_regs, post_swizzle, alpha_post_swizzle, mi);

            // Add the residual value.
            if( WITH_RESIDUAL ) {
                #pragma unroll
                for( int ii = 0; ii < Gmem_tile::STGS; ++ii ) {
                    out_regs[ii].add_residual(res_fetch[ii], beta[ii]);
                }
            }

            // Store output fragments to global memory.
            store(out_regs, out_masks, mi);
        }
    }
    
    // Do only split-k for a 2-kernel split-k.
    template< int N >
    inline __device__ void exelwte_split_k() {
    }

    // Callwlate output_mask.
    inline __device__ void callwlate_out_mask(int32_t out_masks[Gmem_tile::STGS], int32_t mi) {
        #pragma unroll
        for( int ii = 0; ii < Gmem_tile::STGS; ++ii ) {
            out_masks[ii] = this->gmem_tile_.compute_output_mask(mi, ii);
        }
    }

    // Colwert aclwmulator to post swizzle fragment.
    template< typename Fragment_aclwmulator, int N >
    inline __device__
    void acc_to_post_swizzle(Fragment_post_swizzle (&post_swizzle)[Gmem_tile::STGS],
                             Fragment_alpha_post_swizzle (&alpha_post_swizzle)[Gmem_tile::STGS],
                             Fragment_aclwmulator (&acc)[N],
                             int32_t out_masks[Gmem_tile::STGS],
                             int32_t mi) {
        // Do something before we colwert.
        #pragma unroll
        for( int ni = 0; ni < N; ++ni ) {
            callbacks_.pre_colwert(*this, mi, ni, acc[ni]);
        }

        // Colwert the aclwmulators to the epilogue format (or keep them as-is).
        Fragment_pre_swizzle pre_swizzle[N];
        #pragma unroll
        for( int ni = 0; ni < N; ++ni ) {
            pre_swizzle[ni].shuffle_groups(acc[ni]);
        }

        // Load alpha.
        Fragment_alpha_pre_swizzle alpha_pre_swizzle[N];
        #pragma unroll
        for( int ni = 0; ni < N; ++ni ) {
            callbacks_.alpha_pre_swizzle(*this, mi, ni, alpha_pre_swizzle[ni]);
        }

        // Do the colwersion.
        #pragma unroll
        for( int ni = 0; ni < N; ++ni ) {
            pre_swizzle[ni].colwert(alpha_pre_swizzle[ni], acc[ni]);
        }

        // Do something before we swizzle.
        #pragma unroll
        for( int ni = 0; ni < N; ++ni ) {
            callbacks_.pre_swizzle(*this, mi, ni, pre_swizzle[ni]);
        }

        // Make sure the main loop or the previous loop of the epilogue are finished.
        if( !Swizzle::SKIP_SYNCTHREADS ) {
            if( epi_sync_.invalid() ) {
                __syncthreads();
            } else {
                epi_sync_.wait();
            }
        }

        // Store the data in shared memory to produce more friendly stores.
        #pragma unroll
        for( int ni = 0; ni < N; ++ni ) {
            this->swizzle_.store(ni, pre_swizzle[ni]);
        }

        // Make sure the data is in SMEM.
        if( !Swizzle::SKIP_SYNCTHREADS ) {
            if( epi_sync_.invalid() ) {
                __syncthreads();
            } else {
                epi_sync_.wait();
            }
        }

        // The fragments after the swizzle. One fragment per STG.128.
        #pragma unroll
        for( int ii = 0; ii < Gmem_tile::STGS; ++ii ) {
            this->swizzle_.load(post_swizzle[ii], ii);
        }

        // Swizzling via register
        callbacks_.reg_swizzle(pre_swizzle, post_swizzle);

        // Load alpha post swizzle.
        #pragma unroll
        for( int ii = 0; ii < Gmem_tile::STGS; ++ii ) {
            callbacks_.alpha_post_swizzle(*this, mi, ii, alpha_post_swizzle[ii]);
        }

        // Do the parallel reduction, if needed.
        #pragma unroll
        for( int ii = 0; ii < Gmem_tile::STGS; ++ii ) {
            post_swizzle[ii].reduce(alpha_post_swizzle[ii]);
        }

        // Do something now that the data has been swizzled.
        #pragma unroll
        for( int ii = 0; ii < Gmem_tile::STGS; ++ii ) {
            callbacks_.post_swizzle(*this, mi, ii, post_swizzle[ii], out_masks[ii]);
        }
    }

    inline __device__
    void post_swizzle_to_output(Fragment_c (&out_regs)[Gmem_tile::STGS],
                                Fragment_post_swizzle post_swizzle[Gmem_tile::STGS],
                                Fragment_alpha_post_swizzle alpha_post_swizzle[Gmem_tile::STGS],
                                int32_t mi) {
        // Do something before packing and pack to produce a STG.128.
        // Put in one loop for F2IP.RELU optimization.
        #pragma unroll
        for( int ii = 0; ii < Gmem_tile::STGS; ++ii ) {
            callbacks_.pre_pack(*this, mi, ii, post_swizzle[ii]);
            out_regs[ii].pack(alpha_post_swizzle[ii], post_swizzle[ii]);
        }
    }

    inline __device__ void store(Fragment_c out_regs[Gmem_tile::STGS],
                                 int32_t out_masks[Gmem_tile::STGS],
                                 int32_t mi) {
        // Do something before we store.
        #pragma unroll
        for( int ii = 0; ii < Gmem_tile::STGS; ++ii ) {
            callbacks_.pre_store(*this, mi, ii, out_regs[ii], out_masks[ii]);
        }

        // Write valid values.
        #pragma unroll
        for( int ii = 0; ii < Gmem_tile::STGS; ++ii ) {
            this->gmem_tile_.store(mi, ii, out_regs[ii], out_masks[ii], mem_desc_d_);
        }
    }

    // Execute a single iteration of the loop.
    template< bool WITH_RESIDUAL, typename Fragment_aclwmulator, int N >
    inline __device__ void step(int mi, Fragment_aclwmulator (&acc)[N] ) {         
        // The output masks.
        int out_masks[Gmem_tile::STGS];
        #pragma unroll
        for( int ii = 0; ii < Gmem_tile::STGS; ++ii ) {
            out_masks[ii] = this->gmem_tile_.compute_output_mask(mi, ii);
        }

        // Load valid values if beta is not zero.
        Fragment_c res_fetch[Gmem_tile::STGS];
        if( WITH_RESIDUAL ) {
            #pragma unroll
            for( int ii = 0; ii < Gmem_tile::STGS; ++ii ) {
                this->gmem_tile_.load(res_fetch[ii], mi, ii, out_masks[ii], mem_desc_c_);
            }
        }

        // Do something before we colwert.
        #pragma unroll
        for( int ni = 0; ni < N; ++ni ) {
            callbacks_.pre_colwert(*this, mi, ni, acc[ni]);
        }

        // Colwert the aclwmulators to the epilogue format (or keep them as-is).
        Fragment_pre_swizzle pre_swizzle[N];
        #pragma unroll
        for( int ni = 0; ni < N; ++ni ) {
            pre_swizzle[ni].shuffle_groups(acc[ni]);
        }

        // Load alpha.
        Fragment_alpha_pre_swizzle alpha_pre_swizzle[N];
        #pragma unroll
        for( int ni = 0; ni < N; ++ni ) {
            callbacks_.alpha_pre_swizzle(*this, mi, ni, alpha_pre_swizzle[ni]);
        }

        // Do the colwersion.
        #pragma unroll
        for( int ni = 0; ni < N; ++ni ) {
            pre_swizzle[ni].colwert(alpha_pre_swizzle[ni], acc[ni]);
        }

        // Do something before we swizzle.
        #pragma unroll
        for( int ni = 0; ni < N; ++ni ) {
            callbacks_.pre_swizzle(*this, mi, ni, pre_swizzle[ni]);
        }

        // Make sure the main loop or the previous loop of the epilogue are finished.
        if( !Swizzle::SKIP_SYNCTHREADS ) {
            if( epi_sync_.invalid() ) {
                __syncthreads();
            } else {
                epi_sync_.wait();
            }
        }

        // Store the data in shared memory to produce more friendly stores.
        #pragma unroll
        for( int ni = 0; ni < N; ++ni ) {
            this->swizzle_.store(ni, pre_swizzle[ni]);
        }

        // Make sure the data is in SMEM.
        if( !Swizzle::SKIP_SYNCTHREADS ) {
            if( epi_sync_.invalid() ) {
                __syncthreads();
            } else {
                epi_sync_.wait();
            }
        }

        // The fragments after the swizzle. One fragment per STG.128.
        Fragment_post_swizzle post_swizzle[Gmem_tile::STGS];
        #pragma unroll
        for( int ii = 0; ii < Gmem_tile::STGS; ++ii ) {
            this->swizzle_.load(post_swizzle[ii], ii);
        }

        // Swizzling via register
        callbacks_.reg_swizzle(pre_swizzle, post_swizzle);

        // Load alpha post swizzle.
        Fragment_alpha_post_swizzle alpha_post_swizzle[Gmem_tile::STGS];
        #pragma unroll
        for( int ii = 0; ii < Gmem_tile::STGS; ++ii ) {
            callbacks_.alpha_post_swizzle(*this, mi, ii, alpha_post_swizzle[ii]);
        }

        // Do the parallel reduction, if needed.
        #pragma unroll
        for( int ii = 0; ii < Gmem_tile::STGS; ++ii ) {
            post_swizzle[ii].reduce(alpha_post_swizzle[ii]);
        }

        // Do something now that the data has been swizzled.
        #pragma unroll
        for( int ii = 0; ii < Gmem_tile::STGS; ++ii ) {
            callbacks_.post_swizzle(*this, mi, ii, post_swizzle[ii], out_masks[ii]);
        }

        // Load beta. TODO: We should not need a loop.
        Fragment_beta beta[Gmem_tile::STGS];
        if( WITH_RESIDUAL ) {
            #pragma unroll
            for( int ii = 0; ii < Gmem_tile::STGS; ++ii ) {
                callbacks_.beta(*this, mi, ii, beta[ii]);
            }
        }

        // Add the residual value before packing. TODO: We should be able to pass a single beta.
        if( WITH_RESIDUAL ) {
            #pragma unroll
            for( int ii = 0; ii < Gmem_tile::STGS; ++ii ) {
                post_swizzle[ii].add_residual(res_fetch[ii], beta[ii]);
            }
        }

        // Do something before packing and pack to produce a STG.128.
        // Put in one loop for F2IP.RELU optimization.
        Fragment_c out_regs[Gmem_tile::STGS];
        #pragma unroll
        for( int ii = 0; ii < Gmem_tile::STGS; ++ii ) {
            callbacks_.pre_pack(*this, mi, ii, post_swizzle[ii]);
            out_regs[ii].pack(alpha_post_swizzle[ii], post_swizzle[ii]);
        }

        // Add the residual value.
        if( WITH_RESIDUAL ) {
            #pragma unroll
            for( int ii = 0; ii < Gmem_tile::STGS; ++ii ) {
                out_regs[ii].add_residual(res_fetch[ii], beta[ii]);
            }
        }

        // Do something before we store.
        #pragma unroll
        for( int ii = 0; ii < Gmem_tile::STGS; ++ii ) {
            callbacks_.pre_store(*this, mi, ii, out_regs[ii], out_masks[ii]);
        }

        // Write valid values.
        #pragma unroll
        for( int ii = 0; ii < Gmem_tile::STGS; ++ii ) {
            this->gmem_tile_.store(mi, ii, out_regs[ii], out_masks[ii], mem_desc_d_);
        }
    }
    
    template< bool WITH_RESIDUAL, typename Fragment_aclwmulator, int N >
    inline __device__ void step(int mi, 
                                Fragment_aclwmulator (&acc)[N],
                                Fragment_c (&res_fetch)[Gmem_tile::STGS]) {
      // The output masks.
      int out_masks[Gmem_tile::STGS];
      #pragma unroll
      for( int ii = 0; ii < Gmem_tile::STGS; ++ii ) {
          out_masks[ii] = this->gmem_tile_.compute_output_mask(mi, ii);
      }
      
      #pragma unroll
      for( int ni = 0; ni < N; ++ni ) {
          callbacks_.pre_colwert(*this, mi, ni, acc[ni]);
      }
      
      // Colwert the aclwmulators to the epilogue format (or keep them as-is).
      Fragment_pre_swizzle pre_swizzle[N];
      #pragma unroll
      for( int ni = 0; ni < N; ++ni ) {
          pre_swizzle[ni].shuffle_groups(acc[ni]);
      }
      
      // Load alpha.
      Fragment_alpha_pre_swizzle alpha_pre_swizzle[N];
      #pragma unroll
      for( int ni = 0; ni < N; ++ni ) {
          callbacks_.alpha_pre_swizzle(*this, mi, ni, alpha_pre_swizzle[ni]);
      }
      
      // Do the colwersion.
      #pragma unroll
      for( int ni = 0; ni < N; ++ni ) {
          pre_swizzle[ni].colwert(alpha_pre_swizzle[ni], acc[ni]);
      }
      
      // Do something before we swizzle.
      #pragma unroll
      for( int ni = 0; ni < N; ++ni ) {
          callbacks_.pre_swizzle(*this, mi, ni, pre_swizzle[ni]);
      }
      
      // Make sure the main loop or the previous loop of the epilogue are finished.
      if( !Swizzle::SKIP_SYNCTHREADS ) {
          if( epi_sync_.invalid() ) {
              __syncthreads();
          } else {
              epi_sync_.wait();
          }
      }
      
      // Store the data in shared memory to produce more friendly stores.
      #pragma unroll
      for( int ni = 0; ni < N; ++ni ) {
          this->swizzle_.store(ni, pre_swizzle[ni]);
      }
      
      // Make sure the data is in SMEM.
      if( !Swizzle::SKIP_SYNCTHREADS ) {
          if( epi_sync_.invalid() ) {
              __syncthreads();
          } else {
              epi_sync_.wait();
          }
      }
      
      // The fragments after the swizzle. One fragment per STG.128.
      Fragment_post_swizzle post_swizzle[Gmem_tile::STGS];
      #pragma unroll
      for( int ii = 0; ii < Gmem_tile::STGS; ++ii ) {
          this->swizzle_.load(post_swizzle[ii], ii);
      }
      
      // Swizzling via register
      callbacks_.reg_swizzle(pre_swizzle, post_swizzle);
      
      // Load alpha post swizzle.
      Fragment_alpha_post_swizzle alpha_post_swizzle[Gmem_tile::STGS];
      #pragma unroll
      for( int ii = 0; ii < Gmem_tile::STGS; ++ii ) {
          callbacks_.alpha_post_swizzle(*this, mi, ii, alpha_post_swizzle[ii]);
      }
      
      // Do the parallel reduction, if needed.
      #pragma unroll
      for( int ii = 0; ii < Gmem_tile::STGS; ++ii ) {
          post_swizzle[ii].reduce(alpha_post_swizzle[ii]);
      }
      
      // Do something now that the data has been swizzled.
      #pragma unroll
      for( int ii = 0; ii < Gmem_tile::STGS; ++ii ) {
          callbacks_.post_swizzle(*this, mi, ii, post_swizzle[ii], out_masks[ii]);
      }
      
      // Load beta. TODO: We should not need a loop.
      Fragment_beta beta[Gmem_tile::STGS];
      if( WITH_RESIDUAL ) {
          #pragma unroll
          for( int ii = 0; ii < Gmem_tile::STGS; ++ii ) {
              callbacks_.beta(*this, mi, ii, beta[ii]);
          }
      }
      
      // Add the residual value before packing. TODO: We should be able to pass a single beta.
      if( WITH_RESIDUAL ) {
          #pragma unroll
          for( int ii = 0; ii < Gmem_tile::STGS; ++ii ) {
              post_swizzle[ii].add_residual(res_fetch[ii], beta[ii]);
          }
      }
      
      // Do something before packing and pack to produce a STG.128.
      // Put in one loop for F2IP.RELU optimization.
      Fragment_c out_regs[Gmem_tile::STGS];
      #pragma unroll
      for( int ii = 0; ii < Gmem_tile::STGS; ++ii ) {
          callbacks_.pre_pack(*this, mi, ii, post_swizzle[ii]);
          out_regs[ii].pack(alpha_post_swizzle[ii], post_swizzle[ii]);
      }
      
      // Add the residual value.
      if( WITH_RESIDUAL ) {
          #pragma unroll
          for( int ii = 0; ii < Gmem_tile::STGS; ++ii ) {
              out_regs[ii].add_residual(res_fetch[ii], beta[ii]);
          }
      }
      
      // Do something before we store.
      #pragma unroll
      for( int ii = 0; ii < Gmem_tile::STGS; ++ii ) {
          callbacks_.pre_store(*this, mi, ii, out_regs[ii], out_masks[ii]);
      }
      
      // Write valid values.
      #pragma unroll
      for( int ii = 0; ii < Gmem_tile::STGS; ++ii ) {
          this->gmem_tile_.store(mi, ii, out_regs[ii], out_masks[ii], mem_desc_d_);
      }
    
    }
         
    // Execute a single iteration of the loop.
    template< bool WITH_RESIDUAL, typename Fragment_aclwmulator, int M, int N,typename Smem_tile_a ,typename Smem_tile_b>
    inline __device__ void execute(
                                Fragment_aclwmulator (&acc)[M][N], 
                                Smem_tile_a &smem_tile_a,
                                Smem_tile_b &smem_tile_b,
                                int &buffer_head,
                                xmma::Arrive_wait &buffer_empty,
                                xmma::Arrive_wait &buffer_full,
                                unsigned int &lwrrent_phase_buffer_full,                      
                                int buffer_count  ) {         
    
        const int BYTES_PER_XMMA = Cta_tile::THREADS_PER_CTA * 
                                   Gmem_tile::BYTES_PER_STG;
        // Number of xmma_tiles(of all threads) a_smem buffer can hold
        //TODO: Take care of the case when no. of xmma_tiles per a is not integer 
        const int xmma_tiles_per_a = (Smem_tile_a::BYTES_PER_TILE / 
                               buffer_count) /
                               BYTES_PER_XMMA;                      
        // Number of xmma_tiles(of all threads) b_smem buffer can hol
        // TODO: Take care of the case when no. of xmma_tiles per b is not integer                     
        const int xmma_tiles_per_b = (Smem_tile_b::BYTES_PER_TILE / 
                               buffer_count) /
                               BYTES_PER_XMMA;
                                                            
         unsigned int phase_bit;                                    
         int xmma_tile_counter = 0;
                            
         #pragma unroll
         for( int mi = 0; mi < Xmma_tile::XMMAS_M; ++mi )
         {
           
           Fragment_c res_fetch[Gmem_tile::STGS];
           #pragma unroll
           for( int ii = 0; ii < Gmem_tile::STGS; ++ii )
           {    
               //Wait on a new A/B buffer 
               if (xmma_tile_counter == 0 )
               {
                   phase_bit = ( lwrrent_phase_buffer_full >> buffer_head ) & 1;
                   buffer_full.bar_wait( buffer_head, phase_bit );
                   lwrrent_phase_buffer_full ^= ( 1 << buffer_head ) ^ ( 0 );                    
               }
              
               // Load data for (mi,ii)th tile from smem
               if( WITH_RESIDUAL ) 
               {
                    int xmma_tile_idx = (mi * Gmem_tile::STGS + ii) % (xmma_tiles_per_a + xmma_tiles_per_b);
                    uint32_t smem_ptr;
                    if (xmma_tile_idx < xmma_tiles_per_a) 
                      smem_ptr = smem_tile_a.smem_ + smem_tile_a.smem_read_buffer_ + 
                      xmma_tile_idx *  Cta_tile::THREADS_PER_CTA * Gmem_tile::BYTES_PER_STG + 
                      tidx_ * Gmem_tile::BYTES_PER_STG;
                    else
                      smem_ptr = smem_tile_b.smem_ + smem_tile_b.smem_read_buffer_ +
                      (xmma_tile_idx - xmma_tiles_per_a) *  Cta_tile::THREADS_PER_CTA * Gmem_tile::BYTES_PER_STG +
                      tidx_ * Gmem_tile::BYTES_PER_STG;
                     
                     if (Gmem_tile::BYTES_PER_STG == 16)
                     {
                       uint4 dst;
                       lds(dst, smem_ptr);
                       res_fetch[ii].regs_[0] = dst.x;
                       res_fetch[ii].regs_[1] = dst.y;
                       res_fetch[ii].regs_[2] = dst.z;
                       res_fetch[ii].regs_[3] = dst.w;
                     } else if (Gmem_tile::BYTES_PER_STG == 8)
                     {
                       uint2 dst;
                       lds(dst, smem_ptr);
                       res_fetch[ii].regs_[0] = dst.x;
                       res_fetch[ii].regs_[1] = dst.y;
                     } else 
                     {
                       uint32_t dst;
                       lds(dst, smem_ptr);
                       res_fetch[ii].regs_[0] = dst;
                     }
                      
              
                }
               //Increment (mi,ii) counter
               xmma_tile_counter++;
               
               //One A/B buffer is fully filled, move on to the next.
               if (xmma_tile_counter == xmma_tiles_per_a + xmma_tiles_per_b)
               {
                   xmma_tile_counter = 0;
                   buffer_empty.bar_arrive_normal( buffer_head );
                   buffer_head = ( buffer_head < buffer_count - 1 )
                                     ? ( buffer_head + 1 )
                                     : 0;
                   smem_tile_a.move_next_read_buffer();
                   smem_tile_b.move_next_read_buffer();                            
               }
           }//ii           
           this->step<WITH_RESIDUAL>(mi, acc[mi], res_fetch);              
         }//mi
        
        /*
        Edge case when xmma_tiles_per_c cannot be divided by
        (xmma_tiles_per_a + xmma_tiles_per_b)
        */
        if (xmma_tile_counter != 0)
        {
          buffer_empty.bar_arrive_normal( buffer_head );
          buffer_head = ( buffer_head < buffer_count - 1 )
                            ? ( buffer_head + 1 )
                            : 0;
          smem_tile_a.move_next_read_buffer();
          smem_tile_b.move_next_read_buffer();                     
        }
        
    }
    
    // The output tile.
    Gmem_tile &gmem_tile_;
    // The shared memory tile.
    Swizzle &swizzle_;
    // The callbacks.
    Callbacks &callbacks_;
    // The named barrier object used for epilog sync.
    const Named_barrier epi_sync_;
    // Ampere memory descriptors
    const uint64_t mem_desc_c_, mem_desc_d_;  
    //thread index
    const int tidx_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// Epilogue structure that writes back at the very end
// after bn_apply
// friendlier to bn_fusion with cta sync

template<
    typename Traits,
    typename Cta_tile,
    typename Layout,
    typename Gmem_tile,
    typename Callbacks,
    typename Swizzle = xmma::Swizzle_epilogue<Traits, Cta_tile, Layout, Gmem_tile::BYTES_PER_STG>,
    typename Fragment_pre_swizzle = typename Callbacks::Fragment_pre_swizzle,
    typename Fragment_post_swizzle = typename Callbacks::Fragment_post_swizzle,
    typename Fragment_c = typename Callbacks::Fragment_c
>
struct Epilogue_bn_apply {

    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;

    // Ctor.
    template< typename Params >
    inline __device__ Epilogue_bn_apply(const Params &params,
                                        Gmem_tile &gmem_tile,
                                        Swizzle &swizzle,
                                        Callbacks &callbacks,
                                        const Named_barrier &epi_sync = Named_barrier())
        : gmem_tile_(gmem_tile)
        , swizzle_(swizzle)
        , callbacks_(callbacks)
        , epi_sync_(epi_sync) {

        // Make sure alpha/beta are of the correct type.
        this->alpha_ = colwert<typename Traits::Epilogue_type>(params.alpha);
        this->beta_  = colwert<typename Traits::Epilogue_type>(params.beta);

        // need to update smem ptr if bn_final uses smem
        if( Callbacks::BN_final_sum_in_smem_ == true ) {
            swizzle.smem_ += Callbacks::BYTES_PER_TILE;
        }
    }

    // Do the epilogue.
    template< bool WITH_RESIDUAL, typename Fragment_aclwmulator, int M, int N >
    inline __device__ void execute(Fragment_aclwmulator (&acc)[M][N]) {
        // The output masks.
        int out_masks[M][Gmem_tile::STGS];
        
        #pragma unroll
        for( int mi = 0; mi < M; ++mi ){
            #pragma unroll
            for( int ii = 0; ii < Gmem_tile::STGS; ++ii ) {
                out_masks[mi][ii] = this->gmem_tile_.compute_output_mask(mi, ii);
            }
        }
        
        // Do something before we colwert.
        // some of the partial reduction is done at this step
        #pragma unroll
        for( int mi = 0; mi < M; ++mi ) {
          #pragma unroll
          for( int ni = 0; ni < N; ++ni ) {
              callbacks_.pre_colwert(*this, mi, ni, acc[mi][ni]);
          }
        }
        
        // compute the final scale and bias, and store to gmem/smem
        callbacks_.pre_colwert_bn_stats();

        // step will do the entire pipeline of the epilogue
        // without writting the data back to gmem
        #pragma unroll
        for( int mi = 0; mi < M; ++mi ) {
            this->step<WITH_RESIDUAL>(mi, acc[mi], out_masks[mi]);
        }
        
        // pre_store will do batchnorm + relu apply on out_regs
        // the reason we choose to do pre_store outside the M loop
        // and not inside step() is such that we can reuse scale and bias 
        // as much as possible. It is possible this does not matter much. 
        callbacks_.pre_store(*this, out_regs, out_masks);

        // after batchnorm + relu fusion, the results will be written back to gmem
        #pragma unroll
        for( int mi = 0; mi < M; ++mi ) {
            this->store_to_gmem(mi, out_masks[mi]);
        }
    }

    // Do only split-k for a 2-kernel split-k.
    template< int N >
    inline __device__ void exelwte_split_k() {
    }

    // Execute a single iteration of the loop.
    template< bool WITH_RESIDUAL, typename Fragment_aclwmulator, int N >
    inline __device__ void step(int mi, Fragment_aclwmulator (&acc)[N], int (&out_masks)[Gmem_tile::STGS]) {

/*         // The output masks.
        int out_masks[Gmem_tile::STGS];
        #pragma unroll
        for( int ii = 0; ii < Gmem_tile::STGS; ++ii ) {
            out_masks[ii] = this->gmem_tile_.compute_output_mask(mi, ii);
        } */

        // Load valid values if beta is not zero.
        Fragment_c res_fetch[Gmem_tile::STGS];
        if( WITH_RESIDUAL ) {
            #pragma unroll
            for( int ii = 0; ii < Gmem_tile::STGS; ++ii ) {
                this->gmem_tile_.load(res_fetch[ii], mi, ii, out_masks[ii]);
            }
        }

/*         // Do something before we colwert.
        #pragma unroll
        for( int ni = 0; ni < N; ++ni ) {
            callbacks_.pre_colwert(*this, mi, ni, acc[ni]);
        } */

        // Colwert the aclwmulators to the epilogue format (or keep them as-is).
        //Fragment_pre_swizzle pre_swizzle[N];

        #pragma unroll
        for( int ni = 0; ni < N; ++ni ) {
            pre_swizzle[ni].shuffle_groups(acc[ni]);
        }

        #pragma unroll
        for( int ni = 0; ni < N; ++ni ) {
            pre_swizzle[ni].colwert(alpha_, acc[ni]);
        }

        // Do something before we swizzle.
        #pragma unroll
        for( int ni = 0; ni < N; ++ni ) {
            callbacks_.pre_swizzle(*this, mi, ni, pre_swizzle[ni]);
        }

        // Make sure the main loop or the previous loop of the epilogue are finished.
        if( !Swizzle::SKIP_SYNCTHREADS ) {
            if( epi_sync_.invalid() ) {
                __syncthreads();
            } else {
                epi_sync_.wait();
            }
        }

        // Store the data in shared memory to produce more friendly stores.
        #pragma unroll
        for( int ni = 0; ni < N; ++ni ) {
            this->swizzle_.store(ni, pre_swizzle[ni]);
        }

        // Make sure the data is in SMEM.
        if( !Swizzle::SKIP_SYNCTHREADS ) {
            if( epi_sync_.invalid() ) {
                __syncthreads();
            } else {
                epi_sync_.wait();
            }
        }

        // The fragments after the swizzle. One fragment per STG.128.
        Fragment_post_swizzle post_swizzle[Gmem_tile::STGS];
        #pragma unroll
        for( int ii = 0; ii < Gmem_tile::STGS; ++ii ) {
            this->swizzle_.load(post_swizzle[ii], ii);
        }

        // Do the parallel reduction, if needed.
        #pragma unroll
        for( int ii = 0; ii < Gmem_tile::STGS; ++ii ) {
            post_swizzle[ii].reduce(alpha_);
        }

        // Do something now that the data has been swizzled.
        #pragma unroll
        for( int ii = 0; ii < Gmem_tile::STGS; ++ii ) {
            callbacks_.post_swizzle(*this, mi, ii, post_swizzle[ii], out_masks[ii]);
        }

        // Add the residual value before packing.
        if( WITH_RESIDUAL ) {
            #pragma unroll
            for( int ii = 0; ii < Gmem_tile::STGS; ++ii ) {
                post_swizzle[ii].add_residual(res_fetch[ii], beta_);
            }
        }

        // Pack the elements to produce a STG.128.
        // Fragment_c out_regs[Gmem_tile::STGS];
        #pragma unroll
        for( int ii = 0; ii < Gmem_tile::STGS; ++ii ) {
            out_regs[mi][ii].pack(alpha_, post_swizzle[ii]);
        }

        // Add the residual value.
        if( WITH_RESIDUAL ) {
            #pragma unroll
            for( int ii = 0; ii < Gmem_tile::STGS; ++ii ) {
                out_regs[mi][ii].add_residual(res_fetch[ii], beta_);
            }
        }
    }

    // The output tile.
    Gmem_tile &gmem_tile_;
    // The shared memory tile.
    Swizzle &swizzle_;
    // The callbacks.
    Callbacks &callbacks_;
    // The named barrier object used for epilog sync.
    const Named_barrier epi_sync_;
    // The alpha for alpha-scaling.
    typename Traits::Epilogue_type alpha_;
    // The beta for beta-scaling.
    typename Traits::Epilogue_type beta_;
    // Move this as a data member for epilogue, thus callback function
    // can use this register.
    Fragment_pre_swizzle pre_swizzle[Xmma_tile::XMMAS_N];
    // Move the colwerted, swizzled, and packed data member out so the callback
    // function can manipulate these values
    Fragment_c out_regs[Xmma_tile::XMMAS_M][Gmem_tile::STGS];

private:
    inline __device__ void store_to_gmem(int mi, int (&out_masks)[Gmem_tile::STGS]) {
          // Write valid values.
          #pragma unroll
          for( int ii = 0; ii < Gmem_tile::STGS; ++ii ) {
              this->gmem_tile_.store(mi, ii, out_regs[mi][ii], out_masks[ii]);
          }
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    // The instruction traits.
    typename Traits_,
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile_,
    // The layout of the tile.
    typename Layout_,
    // The global memory tile to store the output.
    typename Gmem_tile_,
    // The callbacks to lwstomize the behaviour of the epilogue.
    typename Callbacks = Empty_callbacks_epilogue<Traits_, Cta_tile_>,
    // The class to swizzle the data.
    typename Swizzle_ = xmma::Swizzle_epilogue<Traits_, Cta_tile_, Layout_, Gmem_tile_::BYTES_PER_STG>,
    // The fragment class before the swizzling.
    typename Fragment_pre_swizzle_ = typename Callbacks::Fragment_pre_swizzle,
    // The fragment class after the swizzling.
    typename Fragment_post_swizzle_ = typename Callbacks::Fragment_post_swizzle,
    // The output fragment.
    typename Fragment_c_ = typename Callbacks::Fragment_c
>
struct Epilogue_wo_smem {
    // The instruction traits.
    using Traits = Traits_;
    // The dimensions of the tile computed by the CTA.
    using Cta_tile = Cta_tile_;
    // The layout of the tile.
    using Layout = Layout_;
    // The global memory tile to store the output.
    using Gmem_tile = Gmem_tile_;
    // The class to swizzle the data.
    using Swizzle = Swizzle_;

    // The fragment class before the swizzling.
    using Fragment_pre_swizzle = Fragment_pre_swizzle_;
    // The fragment class after the swizzling.
    using Fragment_post_swizzle = Fragment_post_swizzle_;
    // The output fragment.
    using Fragment_c = Fragment_c_;

    // The fragment for alpha (used before swizzling).
    using Fragment_alpha_pre_swizzle = typename Callbacks::Fragment_alpha_pre_swizzle;
    // The fragment for alpha (used after swizzling).
    using Fragment_alpha_post_swizzle = typename Callbacks::Fragment_alpha_post_swizzle;
    // The fragment for beta.
    using Fragment_beta = typename Callbacks::Fragment_beta;

    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;

    // Ctor.
    template< typename Params >
    inline __device__ Epilogue_wo_smem(const Params &params,
                               Gmem_tile &gmem_tile,
                               Swizzle &swizzle,
                               Callbacks &callbacks,
                               const Named_barrier &epi_sync = Named_barrier(),
                               const int bidm = blockIdx.x,
                               const int bidn = blockIdx.y,
                               const int bidz = blockIdx.z,
                               const int tidx = threadIdx.x,
                               const bool is_warp_specialized = false)
        : gmem_tile_(gmem_tile)
        , swizzle_(swizzle)
        , callbacks_(callbacks)
        , epi_sync_(epi_sync)
        , mem_desc_c_(params.mem_descriptors.descriptor_c)
        , mem_desc_d_(params.mem_descriptors.descriptor_d) {
    }

    // Do the epilogue.
    template< bool WITH_RESIDUAL, typename Fragment_aclwmulator, int M, int N >
    inline __device__ void execute(Fragment_aclwmulator (&acc)[M][N]) {

        #pragma unroll
        for( int mi = 0; mi < M; ++mi ) {
            // reg swizzle for col major (non-interleaved) epilogue
            Fragment_aclwmulator cont_acc[N];
            if ( Layout::COL && !Layout::INTERLEAVED ) {
                // col major epilogue
                #pragma unroll
                for( int ni = 0; ni < N; ++ni ) {
                    cont_acc[ni] = acc[ni][mi];
                }
                this->step<WITH_RESIDUAL>(mi, cont_acc);
            } else {
                // row major epilogue
                this->step<WITH_RESIDUAL>(mi, acc[mi]);
            }
        }
    }

    // Do only split-k for a 2-kernel split-k.
    template< int N >
    inline __device__ void exelwte_split_k() {
    }

    // Execute a single iteration of the loop.
    template< bool WITH_RESIDUAL, typename Fragment_aclwmulator, int N >
    inline __device__ void step(int mi, Fragment_aclwmulator (&acc)[N]) {

        // Write valid values.
        #pragma unroll
        for( int ni = 0; ni < N; ++ni ) {
            Fragment_c out_regs;
            out_regs.pack(callbacks_.alpha_, acc[ni]);

            if (WITH_RESIDUAL) {
                Fragment_c res_fetch;
                this->gmem_tile_.load(res_fetch, mi, ni, mem_desc_d_);
                out_regs.add_residual(res_fetch, callbacks_.beta_);
            }

            // handle bias and relu
            callbacks_.pre_store(*this, mi, ni, out_regs);

            this->gmem_tile_.store(mi, ni, out_regs, mem_desc_d_);
        }
    }

    // The output tile.
    Gmem_tile &gmem_tile_;
    // The shared memory tile.
    Swizzle &swizzle_;
    // The callbacks.
    Callbacks &callbacks_;
    // The named barrier object used for epilog sync.
    const Named_barrier epi_sync_;
    // Ampere memory descriptors
    const uint64_t mem_desc_c_, mem_desc_d_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace helpers
} // namespace xmma

