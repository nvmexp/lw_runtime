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

#include <xmma/helpers/epilogue.h>

#include <xmma/utils.h>
#include <xmma/implicit_gemm/utils.h>
#include <xmma/hopper/gmma_descriptor.h>
#include <xmma/hopper/fragment.h>
#include <xmma/gemm/gmem_tile_hopper.h>
#include <xmma/implicit_gemm/fprop/gmem_tile.h>
#include "xmma/hopper/emu/lwda_tma_utils_internal.h"

namespace xmma {
namespace implicit_gemm {
namespace fprop {

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// T I L E   A
//
///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits,
          typename Cta_tile,
          typename Input_related,
          xmma::Gmma_descriptor_mode desc_mode>
struct Gmem_tile_gmma_a_t
    : public Gmem_tile_base_a<Traits,
                              Cta_tile,
                              Input_related,
                              16,
                              xmma::Row,
                              Cta_tile::K,
                              Cta_tile::M,
                              gemm::Gmem_tile_gmma_base<Traits,
                                                        Cta_tile,
                                                        Cta_tile::K,
                                                        Cta_tile::M,
                                                        Traits::BITS_PER_ELEMENT_A,
                                                        desc_mode>> {
    // The base class.
    using Base_ = Gmem_tile_base_a<Traits,
                                   Cta_tile,
                                   Input_related,
                                   16,
                                   xmma::Row,
                                   Cta_tile::K,
                                   Cta_tile::M,
                                   gemm::Gmem_tile_gmma_base<Traits,
                                                             Cta_tile,
                                                             Cta_tile::K,
                                                             Cta_tile::M,
                                                             Traits::BITS_PER_ELEMENT_A,
                                                             desc_mode>>;

    // The expected shared memory layout.
    using Smem_layout = xmma::Row;

    // Ctor.
    template <typename Params>
    inline __device__
    Gmem_tile_gmma_a_t( const Params &params, void *smem, const dim3 &bidx, int tidx )
        : Base_( params, smem, bidx, tidx ) {
    }
};

template <typename Traits,
          typename Cta_tile,
          typename Input_related,
          xmma::Gmma_descriptor_mode desc_mode>
struct Gmem_tile_gmma_a_n
    : public Gmem_tile_base_a<Traits,
                              Cta_tile,
                              Input_related,
                              16,
                              xmma::Col,
                              Cta_tile::M,
                              Cta_tile::K,
                              gemm::Gmem_tile_gmma_base<Traits,
                                                        Cta_tile,
                                                        Cta_tile::M,
                                                        Cta_tile::K,
                                                        Traits::BITS_PER_ELEMENT_A,
                                                        desc_mode>> {
    static_assert( desc_mode != xmma::Gmma_descriptor_mode::SWIZZLE_64B,
                   "Lwrrently, for SWIZZLE_64B mode, a_n is not needed/implemented" );

    // The base class.
    using Base_ = Gmem_tile_base_a<Traits,
                                   Cta_tile,
                                   Input_related,
                                   16,
                                   xmma::Col,
                                   Cta_tile::M,
                                   Cta_tile::K,
                                   gemm::Gmem_tile_gmma_base<Traits,
                                                             Cta_tile,
                                                             Cta_tile::M,
                                                             Cta_tile::K,
                                                             Traits::BITS_PER_ELEMENT_A,
                                                             desc_mode>>;

    // The expected shared memory layout.
    using Smem_layout = xmma::Col;

    // Ctor.
    template <typename Params>
    inline __device__
    Gmem_tile_gmma_a_n( const Params &params, void *smem, const dim3 &bidx, int tidx )
        : Base_( params, smem, bidx, tidx ) {
    }
};

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// T I L E   B
//
///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits,
          typename Cta_tile,
          typename Input_related,
          xmma::Gmma_descriptor_mode desc_mode>
struct Gmem_tile_gmma_b
    : public Gmem_tile_base_b<Traits,
                              Cta_tile,
                              Input_related,
                              16,
                              gemm::Gmem_tile_gmma_base<Traits,
                                                        Cta_tile,
                                                        Cta_tile::K,
                                                        Cta_tile::N,
                                                        Traits::BITS_PER_ELEMENT_B,
                                                        desc_mode>> {
    // The base class.
    using Base_ = Gmem_tile_base_b<Traits,
                                   Cta_tile,
                                   Input_related,
                                   16,
                                   gemm::Gmem_tile_gmma_base<Traits,
                                                             Cta_tile,
                                                             Cta_tile::K,
                                                             Cta_tile::N,
                                                             Traits::BITS_PER_ELEMENT_B,
                                                             desc_mode>>;

    // The expected shared memory layout.
    using Smem_layout = xmma::Col;

    // Ctor.
    template <typename Params>
    inline __device__
    Gmem_tile_gmma_b( const Params &params, void *smem, const dim3 &bidx, int tidx )
        : Base_( params, smem, bidx, tidx ) {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// T I L E   C
//
////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    // The instruction traits.
    typename Traits,
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // Layout_C
    typename Layout_,
    // The fragment class before writing data to global memory.
    typename Fragment_c = xmma::Fragment_gmma_c<Traits, Cta_tile, Layout_>,
    // Assuming tensor is packed and tensor strides won't be used.
    bool DISABLE_STRIDES = false>
struct Gmem_tile_implicit_gemm_gmma_epilogue_base {
    using Layout = Layout_;

    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;
    // The helper class to compute the row offset (in the tile).
    using Tile_distribution =
        xmma::helpers::Gmem_tile_gmma_epilogue_distribution<Traits, Cta_tile, Layout, 16>;

    // Bytes per element
    enum { BYTES_PER_ELEMENT = Traits::BITS_PER_ELEMENT_C / 8 };

    // The size for each STG.128
    enum { BYTES_PER_STG = 16 };
    // The number of elements per STG per thread.
    enum { ELEMENTS_PER_STG = BYTES_PER_STG / BYTES_PER_ELEMENT };
    // The number of elements per row.
    enum { ELEMENTS_PER_ROW = Layout::ROW ? Cta_tile::N : Cta_tile::M };

    // The number of column loaded per STG
    enum { COLUMNS_PER_STG = BYTES_PER_STG * 8 / BYTES_PER_ELEMENT };
    enum { MIN_TILE_N = Cta_tile::N < COLUMNS_PER_STG ? Cta_tile::N : COLUMNS_PER_STG };
    // tile_m is limited such that every thread can participate, 8 rows per warp
    enum { TILE_M = 8 * Cta_tile::THREADS_PER_CTA / 32, TILE_N = MIN_TILE_N };

    // Threads for STG per row
    enum { STG_THREADS_PER_ROW = TILE_N * BYTES_PER_ELEMENT / BYTES_PER_STG };

    // The number of rows that are written with a single STG (accross the CTA).
    enum { ROWS_PER_STG = Cta_tile::THREADS_PER_CTA / STG_THREADS_PER_ROW };
    // the number of rows per STG instruction by one warp.
    enum { ROWS_PER_STG_PER_WARP = Cta_tile::THREADS_PER_WARP / STG_THREADS_PER_ROW };
    // the number of inner iterations
    enum { STGS_PER_ROW_PER_TILE = xmma::Div_up<TILE_N, COLUMNS_PER_STG>::VALUE };
    enum { STGS_PER_COL_PER_TILE = TILE_M / ROWS_PER_STG };
    enum { STG_ITERATIONS_PER_TILE = STGS_PER_ROW_PER_TILE * STGS_PER_COL_PER_TILE };
    enum { M_PER_WARP = 8 };
    // the number of inner iteration to cover a GMMA M. should always be 2.
    enum { STG_ITERATIONS_PER_GMMA_M = 16 / M_PER_WARP };
    // The number of STGS needed to load a complete row.
    enum { STGS_PER_ROW = ELEMENTS_PER_ROW / COLUMNS_PER_STG };
    static_assert( STGS_PER_ROW > 0, "" );

    // The number of rows to store per XMMA per CTA.
    enum {
        ROWS_PER_XMMA_PER_CTA =
            Layout::ROW ? Xmma_tile::M_PER_XMMA_PER_CTA : Xmma_tile::N_PER_XMMA_PER_CTA
    };
    // The number of steps needed to load the columns.
    enum { STGS_PER_COLUMN = ROWS_PER_XMMA_PER_CTA / ROWS_PER_STG };

    // The number of STGs needed to store the elements per iteration.
    enum { STGS = STGS_PER_COLUMN * STGS_PER_ROW };
    static_assert( STGS > 0, "" );

    // Ctor.
    template <typename Params>
    inline __device__
    Gmem_tile_implicit_gemm_gmma_epilogue_base( const Params &params, int bidm, int bidn, int tidx )
        : Gmem_tile_implicit_gemm_gmma_epilogue_base( params.out_gmem,
                                                      params.res_gmem,
                                                      params.n,
                                                      params.o,
                                                      params.p,
                                                      params.q,
                                                      params.k * params.g,
                                                      params.nopq,
                                                      params.out_stride_n,
                                                      params.out_stride_d,
                                                      params.out_stride_h,
                                                      params.out_stride_w,
                                                      params.out_stride_c,
                                                      params.q,
                                                      params.mul_q,
                                                      params.shr_q,
                                                      params.pq,
                                                      params.mul_pq,
                                                      params.shr_pq,
                                                      params.opq,
                                                      params.mul_opq,
                                                      params.shr_opq,
                                                      bidm,
                                                      bidn,
                                                      tidx ) {
    }

    // Ctor.
    inline __device__ Gmem_tile_implicit_gemm_gmma_epilogue_base( void *out_ptr,
                                                                  const void *res_ptr,
                                                                  int n,
                                                                  int d,
                                                                  int h,
                                                                  int w,
                                                                  int c,
                                                                  int ndhw,
                                                                  int stride_n,
                                                                  int stride_d,
                                                                  int stride_h,
                                                                  int stride_w,
                                                                  int stride_c,
                                                                  int div_w,
                                                                  int mul_w,
                                                                  int shr_w,
                                                                  int div_hw,
                                                                  int mul_hw,
                                                                  int shr_hw,
                                                                  int div_dhw,
                                                                  int mul_dhw,
                                                                  int shr_dhw,
                                                                  int bidm,
                                                                  int bidn,
                                                                  int tidx )
        : params_out_ptr_( out_ptr ), params_res_ptr_( res_ptr ), params_n_( n ), params_d_( d ),
          params_h_( h ), params_w_( w ), params_c_( c ), params_ndhw_( ndhw ),
          params_stride_n_( stride_n ), params_stride_d_( stride_d ), params_stride_h_( stride_h ),
          params_stride_w_( stride_w ), params_stride_c_( stride_c ), params_div_w_( div_w ),
          params_mul_w_( mul_w ), params_shr_w_( shr_w ), params_div_hw_( div_hw ),
          params_mul_hw_( mul_hw ), params_shr_hw_( shr_hw ), params_div_dhw_( div_dhw ),
          params_mul_dhw_( mul_dhw ), params_shr_dhw_( shr_dhw ) {

        if( Layout::ROW ) {
            // Compute the output position for each thread.
            int row = Tile_distribution::compute_row(tidx);
            int col = Tile_distribution::compute_col(tidx);
            ndhw_ = bidm * Cta_tile::M + row;
            c_ = bidn * Cta_tile::N + col * ELEMENTS_PER_STG;
        } else {
        }
    }

    // store to gmem
    template <typename Fragment_pre_stg>
    inline __device__ void store( int mi, int ni, const Fragment_pre_stg &acc_pre_stg ) {
        const int offset =
            ni * TILE_N + ( mi % STG_ITERATIONS_PER_GMMA_M ) * M_PER_WARP * params_c_ +
            ( mi / STG_ITERATIONS_PER_GMMA_M ) * ( TILE_M * STG_ITERATIONS_PER_GMMA_M ) * params_c_;

        char *ptr = &ptr_[Traits::offset_in_bytes_c( offset )];

        #pragma unroll
        for( int stg_idx = 0; stg_idx < STG_ITERATIONS_PER_TILE; ++stg_idx ) {
            ptr += stg_idx * ROWS_PER_STG_PER_WARP * params_c_ * BYTES_PER_ELEMENT;
            int acc_idx = stg_idx * 4;
            uint4 tmp = make_uint4( acc_pre_stg.regs_[acc_idx],
                                    acc_pre_stg.regs_[acc_idx + 1],
                                    acc_pre_stg.regs_[acc_idx + 2],
                                    acc_pre_stg.regs_[acc_idx + 3] );
            // if( mask ) {
            xmma::stg( ptr, tmp );
            //}
        }
    }

    // Is a given output valid?
    inline __device__ int compute_output_mask( int xmmas_mi, int mi, int ni, int stgs_ni ) {
        // Compute the new offsets for that iteration.
        if( stgs_ni == 0 ) {
            #pragma unroll
            for( int ii = 0; ii < STGS_PER_ROW_PER_TILE; ++ii )
                #pragma unroll
                for( int jj = 0; jj < STGS_PER_COL_PER_TILE; ++jj ) {
                    int ndhw = ndhw_;
                    int c = c_;
                    if( Layout::ROW ) {
                        // The row index..
                        ndhw += Tile_distribution::compute_offset( xmmas_mi, mi, jj );
                        c += ni * TILE_N + ii * COLUMNS_PER_STG;
                    } else {
                        // ndhw += ni * TILE_N + ii * COLUMNS_PER_STG;
                        // c += xmmas_mi * Xmma_tile::N_PER_XMMA_PER_CTA + mi * M_PER_WARP + jj *
                        // ROWS_PER_STG;
                    }

                    // Finally assemble the offset.
                    int offset = -1;

                    if( DISABLE_STRIDES ) {
                        if( Layout::ROW ) {
                            if( ndhw < params_ndhw_ && c < params_c_ ) {
                                offset = ndhw * params_c_ + c;
                            }
                        } else {
                            // Decompose the position into N and OPQ.
                            // int n, dhw;
                            // xmma::fast_divmod(n, dhw, ndhw, params_div_dhw_,
                            //                                     params_mul_dhw_,
                            //                                     params_shr_dhw_);
                            // if( ndhw < params_ndhw_ && c < params_c_ ) {
                            //     offset = n * params_c_ * params_d_ * params_h_ * params_w_ +
                            //              c * params_d_ * params_h_ * params_w_ +
                            //              dhw;
                            // }
                        }
                    } else {
                        // Decompose the position into N and OPQ.
                        int n, dhw;
                        xmma::fast_divmod(
                            n, dhw, ndhw, params_div_dhw_, params_mul_dhw_, params_shr_dhw_ );

                        // Decompose the position into D and HW.
                        int d, hw;
                        xmma::fast_divmod(
                            d, hw, dhw, params_div_hw_, params_mul_hw_, params_shr_hw_ );

                        // Decompose the position into H and W.
                        int h, w;
                        xmma::fast_divmod( h, w, hw, params_div_w_, params_mul_w_, params_shr_w_ );

                        if( ndhw < params_ndhw_ && c < params_c_ ) {
                            offset = n * params_stride_n_ + d * params_stride_d_ +
                                     h * params_stride_h_ + w * params_stride_w_ +
                                     c * params_stride_c_;
                        }
                    }
                    offsets_[ii * STGS_PER_COL_PER_TILE + jj] = offset;
                }
        }

        // Is it a valid mask?
        return offsets_[stgs_ni] >= 0;
    }

    // Load the data from global memory.
    inline __device__ void load( Fragment_c &data,
                                 int xmmas_mi,
                                 int mi,
                                 int ni,
                                 int ii,
                                 int mask,
                                 uint64_t mem_desc = MEM_DESC_DEFAULT ) {
        const char *ptr = reinterpret_cast<const char *>( params_res_ptr_ );
        if( mask ) {
            if( BYTES_PER_STG == 16 ) {
                uint4 tmp;
                xmma::ldg( tmp, ptr + Traits::offset_in_bytes_c( offsets_[ii] ), mem_desc );
                data.from_int4( tmp );
            } else if( BYTES_PER_STG == 8 ) {
                uint2 tmp;
                xmma::ldg( tmp, ptr + Traits::offset_in_bytes_c( offsets_[ii] ), mem_desc );
                data.from_int2( tmp );
            } else {
                uint32_t tmp;
                xmma::ldg( tmp, ptr + Traits::offset_in_bytes_c( offsets_[ii] ), mem_desc );
                data.reg( 0 ) = tmp;
            }
        }
    }

    // store to gmem
    inline __device__ void store( int xmmas_mi,
                                  int mi,
                                  int ni,
                                  int ii,
                                  const Fragment_c &data,
                                  int mask,
                                  uint64_t mem_desc = MEM_DESC_DEFAULT ) {
        char *ptr = reinterpret_cast<char *>( params_out_ptr_ );
        if( mask ) {
            if( BYTES_PER_STG == 16 ) {
                xmma::stg(
                    ptr + Traits::offset_in_bytes_c( offsets_[ii] ), data.to_int4(), mem_desc );
            } else if( BYTES_PER_STG == 8 ) {
                xmma::stg( ptr + Traits::offset_in_bytes_c( offsets_[ii] ),
                           make_uint2( data.reg( 0 ), data.reg( 1 ) ),
                           mem_desc );
            } else {
                xmma::stg(
                    ptr + Traits::offset_in_bytes_c( offsets_[ii] ), data.reg( 0 ), mem_desc );
            }
        }
    }

    // The pointer to global memory.
    char *ptr_;
    // The pointer to the output buffer.
    void *const params_out_ptr_;
    // The pointer to the input residual buffer.
    const void *const params_res_ptr_;
    // The dimensions of the output.
    const int params_n_, params_d_, params_h_, params_w_, params_c_;
    // The strides.
    const int params_stride_n_, params_stride_d_, params_stride_h_, params_stride_w_,
        params_stride_c_;
    // The constants to help with faster division.
    const int params_div_w_;
    const unsigned params_mul_w_, params_shr_w_;
    // The constants to help with faster division.
    const int params_div_hw_;
    const unsigned params_mul_hw_, params_shr_hw_;
    // The constants to help with faster division.
    const int params_div_dhw_;
    const unsigned params_mul_dhw_, params_shr_dhw_;
    // The position of the thread in the 2D output matrix.
    int ndhw_, c_, params_ndhw_;
    // Offsets.
    int offsets_[STG_ITERATIONS_PER_TILE];
};

template <
    // The instruction traits.
    typename Traits,
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The fragment class before writing data to global memory.
    typename Fragment_c = xmma::Fragment_gmma_c<Traits, Cta_tile, xmma::Row>,
    bool DISABLE_STRIDES = false>
struct Gmem_tile_gmma_c_t : public Gmem_tile_implicit_gemm_gmma_epilogue_base<Traits,
                                                                              Cta_tile,
                                                                              xmma::Row,
                                                                              Fragment_c,
                                                                              DISABLE_STRIDES> {

    // The base class
    using Base = Gmem_tile_implicit_gemm_gmma_epilogue_base<Traits,
                                                            Cta_tile,
                                                            xmma::Row,
                                                            Fragment_c,
                                                            DISABLE_STRIDES>;

    template <typename Params>
    inline __device__
    Gmem_tile_gmma_c_t( const Params &params, int bidm, int bidn, int bidz, int tidx )
        : Base( params, bidm, bidn, tidx ) {
    }
};

template <
    // The instruction traits.
    typename Traits,
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The fragment class before writing data to global memory.
    typename Fragment_c = xmma::Fragment_gmma_c<Traits, Cta_tile, xmma::Col>,
    bool DISABLE_STRIDES = false>
struct Gmem_tile_gmma_c_n : public Gmem_tile_implicit_gemm_gmma_epilogue_base<Traits,
                                                                              Cta_tile,
                                                                              xmma::Col,
                                                                              Fragment_c,
                                                                              DISABLE_STRIDES> {

    // The base class
    using Base = Gmem_tile_implicit_gemm_gmma_epilogue_base<Traits,
                                                            Cta_tile,
                                                            xmma::Col,
                                                            Fragment_c,
                                                            DISABLE_STRIDES>;

    template <typename Params>
    inline __device__
    Gmem_tile_gmma_c_n( const Params &params, int bidm, int bidn, int bidz, int tidx )
        : Base( params, bidm, bidn, tidx ) {
    }
};

template<
    typename Traits_,
    typename Cta_tile_
    // typename Layout
>
struct Gmem_tile_tma_a {

    using Traits = Traits_;
    using Cta_tile = Cta_tile_;

   // The expected shared memory layout.
    using Smem_layout = xmma::Row;


    enum { USE_IM2COL = 1 };
    enum { BYTES = Traits::BITS_PER_ELEMENT_A };
    enum { USE_LDGSTS = 1 };
    enum { BYTES_PER_LDG = 16 };
    enum { BYTES_PER_EXTRA_SMEM = 0 };
    enum { USE_TMA = 1 };

    enum { NUM_TILE_BLOCKS_ROW = xmma::tma::kNumTileBlocksRow(Cta_tile::K, BYTES)};
    enum { NUM_TILE_BLOCKS_COL = xmma::tma::kNumTileBlocksCol(Cta_tile::M)};
    enum { TILE_BLOCK_ROW = xmma::tma::kTileBlockRow(Cta_tile::K, BYTES)};
    enum { TILE_BLOCK_COL = xmma::tma::kTileBlockCol(Cta_tile::M)};
    enum { TILE_BLOCK_ROW_ALIGNED_OFFSET_BYTES = 
            xmma::tma::kTileBlockRowAlignedOffsetBytes(Cta_tile::K, Cta_tile::M, BYTES)};
    enum { TILE_BLOCK_COL_ALIGNED_OFFSET_BYTES = 
            xmma::tma::kTileBlockColAlignedOffsetBytes(Cta_tile::K, Cta_tile::M, BYTES)};


    // Ctor.
    template <typename Params>
    inline __device__ Gmem_tile_tma_a( const Params &params,
                                       const lwdaTmaDescv2 *p_desc,
                                       char *extra_smem,
                                       const dim3 &bidx,
                                       int tidx )
        : p_desc_( p_desc ), params_split_k_c_( params.split_k_c ),
          params_filter_trs_per_cta_( params.filter_trs_per_cta ), dont_issue_utmaldg_( false ) {

        // Prefetch tma descriptors.
        if( tidx == 0 ) {
            utmapf2<2, TILED>(p_desc_, bidx.x, bidx.y, 0, 0, 0);
        }

        // params.load_range_ndhw;
        // params.load_range_c;
        // Below are done in uniform registers.
        // The 1st warp loads gmem_tile_a.
        if( tidx == 0 ) {
            int nopq_base = bidx.x * Cta_tile::M;

            #pragma unroll
            for( int mi = 0; mi < NUM_TILE_BLOCKS_COL; mi ++) {
                int nopq, opq, n;
                nopq = nopq_base + mi * TILE_BLOCK_COL;
                xmma::fast_divmod( n, opq, nopq, params.opq, params.mul_opq,
                                params.shr_opq );
                // Decompose opq into o and pq.
                int pq, o, p, q;
                xmma::fast_divmod( o, pq, opq, params.pq, params.mul_pq, params.shr_pq );
                // Decompose pq into p and q.
                xmma::fast_divmod( p, q, pq, params.q, params.mul_q, params.shr_q );

                // d[mi] += bidx.z * params.split_k_t * params.dilation[0];
                // h[mi] += bidx.z * params.split_k_r * params.dilation[1];
                n_base_[mi] = n;
                d_base_[mi] = o * params.stride[0] - params.pad[0][0];
                h_base_[mi] = p * params.stride[1] - params.pad[1][0];
                w_base_[mi] = q * params.stride[2] - params.pad[2][0];
            }

            c_base_ = bidx.z * params_split_k_c_;
            // Initialize filter coordinate.
            filter_coord_ = params.filter_coord_a[0];
        }
    }

    template<typename Smem_tile>
    inline __device__ void load(Smem_tile &smem_tile, uint64_t) {
        if( !dont_issue_utmaldg_ ) {
            #pragma unroll
            for(unsigned c = 0; c < NUM_TILE_BLOCKS_COL; c++) {
                #pragma unroll
                for(unsigned r = 0; r < NUM_TILE_BLOCKS_ROW; r++) {
                    unsigned smem_offset = r * TILE_BLOCK_ROW_ALIGNED_OFFSET_BYTES +
                                         c * TILE_BLOCK_COL_ALIGNED_OFFSET_BYTES;
                    // smem_tile.template store<5, IM2COL>(reinterpret_cast<const void*>(p_desc_),
                    // smem_offset, c_base_, w_base_[c], h_base_[c], d_base_[c], n_base_[c],
                    // filter_coord_);
                }
            }
        }
    }

    template<typename Smem_tile>
    inline __device__ void commit(Smem_tile &smem_tile_) {}

    // Reuse next_trsi here.
    inline __device__ void move( int next_trsi, int64_t delta ) {
        filter_coord_ = delta;
        if(next_trsi == 0) c_base_ += Cta_tile::K;
    }

    inline __device__ void residue() {}

    inline __device__ void disable_loads() {
        dont_issue_utmaldg_ = true;
    }

    const int params_filter_trs_per_cta_;
    const lwdaTmaDescv2 *p_desc_;
    // The split-k argument (TODO: move to base class).
    const int params_split_k_c_;
    bool dont_issue_utmaldg_;
    uint32_t filter_coord_;
    int32_t n_base_[NUM_TILE_BLOCKS_ROW];
    int32_t d_base_[NUM_TILE_BLOCKS_ROW];
    int32_t h_base_[NUM_TILE_BLOCKS_ROW];
    int32_t w_base_[NUM_TILE_BLOCKS_ROW];
    int32_t c_base_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    typename Traits_,
    typename Cta_tile_
    // typename Layout_
>
struct Gmem_tile_tma_b {

    using Traits = Traits_;
    using Cta_tile = Cta_tile_;
    // using Gmem_layout = Layout_;

    // The expected shared memory layout.
    using Smem_layout = xmma::Col;

    enum { USE_IM2COL = 0 };
    enum { BYTES = Traits::BITS_PER_ELEMENT_B };
    enum { USE_LDGSTS = 1};
    enum { BYTES_PER_LDG = 16};
    enum { BYTESPER_EXTRA_SMEM = 0 };
    enum { USE_TMA = 1 };

    enum { NUM_TILE_BLOCKS_ROW = xmma::tma::kNumTileBlocksRow(Cta_tile::K, BYTES)};
    enum { NUM_TILE_BLOCKS_COL = xmma::tma::kNumTileBlocksCol(Cta_tile::N)};
    enum { TILE_BLOCK_ROW = xmma::tma::kTileBlockRow(Cta_tile::K, BYTES)};
    enum { TILE_BLOCK_COL = xmma::tma::kTileBlockCol(Cta_tile::N)};
    enum { TILE_BLOCK_ROW_ALIGNED_OFFSET_BYTES =
            xmma::tma::kTileBlockRowAlignedOffsetBytes(Cta_tile::K, Cta_tile::N, BYTES)};
    enum { TILE_BLOCK_COL_ALIGNED_OFFSET_BYTES =
            xmma::tma::kTileBlockColAlignedOffsetBytes(Cta_tile::K, Cta_tile::N, BYTES)};


    // Ctor.
    template <typename Params>
    inline __device__ Gmem_tile_tma_b( const Params &params,
                                       const lwdaTmaDescv2 *p_desc,
                                       char *extra_smem,
                                       const dim3 &bidx,
                                       int tidx )
        : p_desc_( p_desc ), params_split_k_c_( params.split_k_c ),
          params_filter_trs_per_cta_( params.filter_trs_per_cta ), dont_issue_utmaldg_( false ),
          flt_rs_( params.filter_rs_per_cta ), mul_rs_( params.mul_filter_rs_per_cta ),
          shr_rs_( params.shr_filter_rs_per_cta ), flt_s_( params.filter_s_per_cta ),
          mul_s_( params.mul_filter_s_per_cta ), shr_s_( params.shr_filter_s_per_cta ) {

        // Prefetch tma descriptors.
        if( tidx == 0 ) {
            utmapf2<2, TILED>(p_desc_, bidx.x, bidx.y, 0, 0, 0);
        }

        // Below are done in uniform registers.
        if( tidx == 0 ) {
            k_ = bidx.y * Cta_tile::N;
            c_ = bidx.z * params_split_k_c_;
            filter_coord_ = 0;
        }
    }

    template<typename Smem_tile>
    inline __device__ void load(Smem_tile &smem_tile, uint64_t) {
        if( !dont_issue_utmaldg_ ) {
            if(threadIdx.x == 0) {
                #pragma unroll
                for(uint32_t col = 0; col < NUM_TILE_BLOCKS_COL; col++) {
                    #pragma unroll
                    for(uint32_t row = 0; row < NUM_TILE_BLOCKS_ROW; row++) {
                        uint32_t smem_ = smem_tile.smem_ + smem_tile.smem_write_buffer_ +
                                        row * TILE_BLOCK_ROW_ALIGNED_OFFSET_BYTES +
                                        col * TILE_BLOCK_COL_ALIGNED_OFFSET_BYTES;

                        int32_t rs, ti, ri, si;
                        xmma::fast_divmod( ti, rs, filter_coord_, flt_rs_, mul_rs_, shr_rs_ );
                        xmma::fast_divmod( ri, si, rs, flt_s_, mul_s_, shr_s_ );

                        int32_t ki = k_ + col * TILE_BLOCK_COL;
                        int32_t ci = c_ + row * TILE_BLOCK_ROW;
                        // Update with correct instruction
                        // xmma::utmaldg_tiled<5>(p_desc_,
                        // xmma::hopper::emu::set_shared_data_address(smem_), uint32_t(0), ci, si,
                        // ri, ti, ki);
                    }
                }
            }
        }
    }

    template<typename Smem_tile>
    inline __device__ void commit(Smem_tile &smem_tile_) {}

    // Reuse next_trsi here.
    inline __device__ void move( int next_trsi, int64_t delta ) {
        filter_coord_ = next_trsi;
        if(next_trsi == 0) c_ += Cta_tile::K;
    }

    inline __device__ void residue() {}

    inline __device__ void disable_loads() {
        dont_issue_utmaldg_ = true;
    }

    const int params_filter_trs_per_cta_;
    const lwdaTmaDescv2 *p_desc_;
    // The split-k argument (TODO: move to base class).
    const int params_split_k_c_;
    bool dont_issue_utmaldg_;
    int32_t k_, c_;
    int32_t filter_coord_;
    int32_t flt_rs_, flt_s_;
    int32_t c_base_;
    uint32_t mul_rs_, shr_rs_, mul_s_, shr_s_;
};

} // namespace fprop
} // namespace implicit_gemm
} // namespace xmma
