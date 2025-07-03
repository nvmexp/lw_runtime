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

#include <xmma/warp_masks.h>
#include <xmma/utils.h>
#include <xmma/implicit_gemm/fprop/utils.h>
#include <xmma/implicit_gemm/utils.h>
#include <xmma/gemm/gmem_tile.h>

namespace xmma {
namespace implicit_gemm {
namespace interleaved_fprop {

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// T I L E   A
//
///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits_,
          typename Cta_tile_,
          typename Input_related_,
          int BYTES_PER_PACKET_,
          int BYTES_PER_LDG_>
struct Gmem_tile_base_a {

    // Make sure we use 16B per LDG for the moment.
    static_assert( BYTES_PER_LDG_ == 16, "" );

    // The traits class.
    using Traits = Traits_;
    // The CTA tile descriptor.
    using Cta_tile = Cta_tile_;
    // Template parameters which are related to input parameters like filter size.
    using Input_related = Input_related_;
    // The dimensions of the tile.
    enum { M = Cta_tile::M, N = Cta_tile::K };
    // The size in bits of each element.
    enum { BITS_PER_ELT = Traits::BITS_PER_ELEMENT_A };

    // The size of a packet of interleaved elements.
    enum { BYTES_PER_PACKET = BYTES_PER_PACKET_ };
    // The number of elements per packet.
    enum { ELTS_PER_PACKET = BYTES_PER_PACKET * 8 / BITS_PER_ELT };
    // Make sure a single loop iteration consumes at least one packet.
    static_assert( Cta_tile::K % ELTS_PER_PACKET == 0, "" );
    // The number of valid elements per packet for group colw.
    enum { ELTS_PER_GROUP = Cta_tile::K / Cta_tile::GROUPS };

    // The unroll factor for LDGS
    enum { LDGS_UNROLL = 16 / BYTES_PER_LDG_ };
    // The size in bytes of each LDG.
    enum { BYTES_PER_LDG = LDGS_UNROLL * BYTES_PER_LDG_ };
    // The number of elements per LDG.128.
    enum { ELTS_PER_LDG = BYTES_PER_LDG_ * 8 / BITS_PER_ELT };
    // Make sure there are more elements in a packet than loaded per LDG.
    static_assert( BYTES_PER_PACKET % BYTES_PER_LDG == 0, "" );
    // The number of threads needed to load a single packet.
    enum { THREADS_PER_PACKET = BYTES_PER_PACKET / BYTES_PER_LDG };

    // The number of columns in the matrix. One element is a packet.
    enum { COLS = Cta_tile::K / ELTS_PER_PACKET };
    // The number of packets loaded per LDG.
    enum { PACKETS_PER_LDG = Cta_tile::THREADS_PER_CTA / THREADS_PER_PACKET };
    // The number of columns loaded per LDG. It must be COLS or it won't work (move/preds).
    enum { COLS_PER_LDG = COLS };
    // The number of rows loaded per LDG.
    enum { ROWS_PER_LDG = PACKETS_PER_LDG / COLS_PER_LDG };

    // The number of LDGs needed to load a column.
    enum { LDGS_PER_COL = Cta_tile::M / ROWS_PER_LDG };
    // The number of LDGs needed to load a row.
    enum { LDGS_PER_ROW = COLS / COLS_PER_LDG };
    // Make sure we have at most one LDG per row. Or, we have to use multiple predicate regs.
    static_assert( LDGS_PER_ROW == 1, "" );
    // The total number of LDGS.
    enum { LDGS = LDGS_PER_COL * LDGS_PER_ROW * LDGS_UNROLL };
    // The number of predicate registers.
    enum { PRED_REGS = xmma::Compute_number_of_pred_regs<LDGS>::VALUE };

    // The extra amount of shared memory needed by that tile.
    enum { BYTES_PER_EXTRA_SMEM = 0 };

    enum { FLT_T = Input_related::FLT_T };
    enum { FLT_R = Input_related::FLT_R };
    enum { FLT_S = Input_related::FLT_S };
    enum { STATIC_FILTER_SIZE = Input_related::STATIC_FILTER_SIZE };

    // LDGSTS config
    static const bool USE_BYPASS = ( FLT_T * FLT_R * FLT_S == 1 );
    static const bool IS_GROUP_COLW = (Cta_tile::GROUPS >=4 );
    using LDGSTS_CFG = xmma::Ldgsts_config<true, USE_BYPASS || IS_GROUP_COLW>;

    // Ctor.
    template <typename Params>
    inline __device__ Gmem_tile_base_a( const Params& params, void* )
        : params_filter_trs_per_cta_( params.filter_trs_per_cta ),
          ptr_( reinterpret_cast<const char*>( params.img_gmem ) ) {
    }

    // Ctor.
    template <typename Params>
    inline __device__ Gmem_tile_base_a( const Params& params, void*, const dim3& bidx, int tidx )
        : Gmem_tile_base_a( params, nullptr ) {

        // The coordinates of the thread.
        int row = bidx.x * Cta_tile::M + tidx / THREADS_PER_PACKET % ROWS_PER_LDG;
        int col = tidx / THREADS_PER_PACKET / ROWS_PER_LDG;

        int c_base = 0;
        if( params.g > 1 ) {
            c_base = bidx.y * ( Cta_tile::N / ELTS_PER_PACKET );
        }

        // The following code works only because LDGS_PER_ROW == 1.
        int c = Cta_tile::GROUPS > 1 ? col : 0;
        int trs = Cta_tile::GROUPS > 1 ? 0 : col;
        // Extract C and RS. If the filter is smaller than the number of "columns", then we have to
        // split between C and RS as we work on multiple "filter taps".
        int filter_t_per_cta, filter_r_per_cta, filter_s_per_cta;
        int filter_trs_per_cta, filter_rs_per_cta;
        if( STATIC_FILTER_SIZE ) {
            filter_s_per_cta = FLT_S;
            filter_rs_per_cta = FLT_R * FLT_S;
            filter_trs_per_cta = FLT_T * FLT_R * FLT_S;
        } else {
            filter_s_per_cta = params.filter_s_per_cta;
            filter_rs_per_cta = params.filter_rs_per_cta;
            filter_trs_per_cta = params.filter_trs_per_cta;
        }

        if( Cta_tile::GROUPS == 1 && filter_trs_per_cta < COLS ) {
            xmma::fast_divmod( c,
                               trs,
                               col,
                               filter_trs_per_cta,
                               params.mul_filter_trs_per_cta,
                               params.shr_filter_trs_per_cta );
        }

        c += c_base;

        // Extract R and S. Of course, if the filter is 1, R = S = 0.
        int t, r, s, rs;
        xmma::fast_divmod( t,
                           rs,
                           trs,
                           filter_rs_per_cta,
                           params.mul_filter_rs_per_cta,
                           params.shr_filter_rs_per_cta );
        xmma::fast_divmod(
            r, s, rs, filter_s_per_cta, params.mul_filter_s_per_cta, params.shr_filter_s_per_cta );

        // The masks for the predicates.
        const uint32_t MASK_T = xmma::implicit_gemm::Build_mask_t<FLT_T, FLT_R, FLT_S>::VALUE;
        const uint32_t MASK_R = xmma::implicit_gemm::Build_mask_r<FLT_T, FLT_R, FLT_S>::VALUE;
        const uint32_t MASK_S = xmma::implicit_gemm::Build_mask_s<FLT_T, FLT_R, FLT_S>::VALUE;

        // For each LDG, compute the NPQ decomposition, the pointer and issue the 1st LDG.
        int d[LDGS], h[LDGS], w[LDGS];
#pragma unroll
        for( int mi = 0; mi < LDGS_PER_COL; ++mi ) {

            // The position in the row/col.
            int nopq = row + mi * ROWS_PER_LDG;

            // The masks -- initialization to -1 if it is valid.
            masks_[mi] = ( nopq < params.nopq && ( c * ELTS_PER_PACKET ) < params.c * params.g )
                             ? uint32_t( -1 )
                             : 0u;

            // Account for the fact that we may have more than a single thread per packet.
            int ele_offset_in_packet = tidx % THREADS_PER_PACKET * ELTS_PER_LDG;


            if( Input_related::IS_SIMPLE_1x1x1 ) {
                // Decompose nopq into n and opq.
                int n, opq;
                xmma::fast_divmod( n, opq, nopq, params.opq, params.mul_opq, params.shr_opq );
                this->offsets_[mi] =
                    ( n * params.img_stride_n + c * params.img_stride_c + opq ) * ELTS_PER_PACKET;
                offsets_[mi] += ele_offset_in_packet;
            } else {
                // Decompose nopq into n and opq.
                int n, opq;
                xmma::fast_divmod( n, opq, nopq, params.opq, params.mul_opq, params.shr_opq );

                // Decompose into opq into o, pq.
                int o, pq;
                xmma::fast_divmod( o, pq, opq, params.pq, params.mul_pq, params.shr_pq );

                // Decompose pq into p and q.
                int p, q;
                xmma::fast_divmod( p, q, pq, params.q, params.mul_q, params.shr_q );

                // Compute h and w. We implement a cross-correlation.
                d[mi] = o * params.stride[0] - params.pad[0][0];
                h[mi] = p * params.stride[1] - params.pad[1][0];
                w[mi] = q * params.stride[2] - params.pad[2][0];

                // Compute the offset for N/C (the C dimension is already multiplied by
                // ELTS_PER_LDG).
                offsets_[mi] = n * params.img_stride_n + c * params.img_stride_c;

                // Cast ELTS_PER_LDG to make sure it is treated as signed (h/w may be < 0).
                offsets_[mi] += ( d[mi] + t * params.dilation[0] ) * params.img_stride_d;
                offsets_[mi] += ( h[mi] + r * params.dilation[1] ) * params.img_stride_h;
                offsets_[mi] += ( w[mi] + s * params.dilation[2] ) * params.img_stride_w;
                offsets_[mi] *= ELTS_PER_PACKET;

                offsets_[mi] += ele_offset_in_packet;

                // Assemble the 1st bit of the masks for T.
                uint32_t mask_t;
                if( STATIC_FILTER_SIZE ) {
                    mask_t = MASK_T;
                } else {
                    mask_t = params.mask_t;
                }
                mask_t ^= uint32_t( -1 );
                if( (unsigned)( d[mi] ) >= params.d ) {
                    masks_[mi] = masks_[mi] & mask_t;
                }

                // Assemble the 1st bit of the masks for R.
                uint32_t mask_r;
                if( STATIC_FILTER_SIZE ) {
                    mask_r = MASK_R;
                } else {
                    mask_r = params.mask_r;
                }
                mask_r ^= uint32_t( -1 );
                if( (unsigned)( h[mi] ) >= params.h ) {
                    masks_[mi] = masks_[mi] & mask_r;
                }

                // Assemble the 1st bit of the masks for S.
                uint32_t mask_s;
                if( STATIC_FILTER_SIZE ) {
                    mask_s = MASK_S;
                } else {
                    mask_s = params.mask_s;
                }
                mask_s ^= uint32_t( -1 );
                if( (unsigned)( w[mi] ) >= params.w ) {
                    masks_[mi] = masks_[mi] & mask_s;
                }
            }
        }

        // Compute the masks for R.
        if( STATIC_FILTER_SIZE ) {
            filter_t_per_cta = FLT_T;
            filter_r_per_cta = FLT_R;
        } else {
            filter_t_per_cta = params.filter_t_per_cta;
            filter_r_per_cta = params.filter_r_per_cta;
        }

// Compute the masks for T.
#pragma unroll
        for( int ti = 1; ti < filter_t_per_cta; ++ti ) {
            uint32_t mask_t;
            if( STATIC_FILTER_SIZE ) {
                mask_t = ( MASK_T << ( ti * FLT_R * FLT_S ) );
            } else {
                mask_t = ( params.mask_t << ( ti * params.filter_rs_per_cta ) );
            }
            mask_t ^= uint32_t( -1 );
#pragma unroll
            for( int mi = 0; mi < LDGS; ++mi ) {
                if( (unsigned)( d[mi] + ti * params.dilation[0] ) >= params.d ) {
                    masks_[mi] = masks_[mi] & mask_t;
                }
            }
        }

// Compute the masks for R.
#pragma unroll
        for( int ri = 1; ri < filter_r_per_cta; ++ri ) {
            uint32_t mask_r;
            if( STATIC_FILTER_SIZE ) {
                mask_r = ( MASK_R << ( ri * FLT_S ) );
            } else {
                mask_r = ( params.mask_r << ( ri * params.filter_s_per_cta ) );
            }
            mask_r ^= uint32_t( -1 );
#pragma unroll
            for( int mi = 0; mi < LDGS; ++mi ) {
                if( (unsigned)( h[mi] + ri * params.dilation[1] ) >= params.h ) {
                    masks_[mi] = masks_[mi] & mask_r;
                }
            }
        }

// Compute the masks for S.
#pragma unroll
        for( int si = 1; si < filter_s_per_cta; ++si ) {
            uint32_t mask_s;
            if( STATIC_FILTER_SIZE ) {
                mask_s = ( MASK_S << si );
            } else {
                mask_s = ( params.mask_s << si );
            }
            mask_s ^= uint32_t( -1 );
#pragma unroll
            for( int mi = 0; mi < LDGS; ++mi ) {
                if( (unsigned)( w[mi] + si * params.dilation[2] ) >= params.w ) {
                    masks_[mi] = masks_[mi] & mask_s;
                }
            }
        }

        // Pack the predicates.
        xmma::implicit_gemm::pack_predicates( this->preds_, masks_, 1u << trs );

        // Precompute the masks for the residue.
        int gemm_k = params.loop_residue_k + col * ELTS_PER_PACKET;
        this->residue_mask_ = 0u;
        if( gemm_k < params.c * params.g * filter_trs_per_cta ) {
            this->residue_mask_ = uint32_t( -1 );
        }
    }

    // Compute the global memory pointers.
    inline __device__ void compute_load_pointers( const void* ( &ptrs )[LDGS] ) const {
#pragma unroll
        for( int ii = 0; ii < LDGS; ++ii ) {
            ptrs[ii] = this->ptr_ + Traits::offset_in_bytes_a( this->offsets_[ii] );
        }
    }

    // Disable the loads.
    inline __device__ void disable_loads() {
        preds_[0] = 0u;
    }

    // Move the pointers and update the predicates for R2P/P2R.
    inline __device__ void move( int next_trsi, int64_t delta ) {
        // Update the pointer.
        ptr_ += delta;

        // Update the predicates and store them in a register using P2R.
        int filter_trs_per_cta;
        if( STATIC_FILTER_SIZE ) {
            filter_trs_per_cta = FLT_T * FLT_R * FLT_S;
        } else {
            filter_trs_per_cta = params_filter_trs_per_cta_;
        }
        if( filter_trs_per_cta > 1 ) {
            xmma::implicit_gemm::pack_predicates( this->preds_, masks_, 1u << next_trsi );
        }
    }

    // The residue to "fix" the predicates.
    inline __device__ void residue() {
        this->preds_[0] &= this->residue_mask_;
    }

    // The product of each dim of filter
    const int params_filter_trs_per_cta_;
    // The pointer.
    const char* ptr_;
    // The associated offsets.
    int offsets_[LDGS];
    // The masks.
    uint32_t masks_[LDGS], preds_[PRED_REGS], residue_mask_;
};

///////////////////////////////////////////////////////////////////////////////////////////////////

template <
    // The instruction traits.
    typename Traits,
    // The CTA tile description.
    typename Cta_tile,
    // Template parameters which are related to input parameters like filter size.
    typename Input_related,
    // The size in bytes of each packet.
    int BYTES_PER_PACKET,
    // The size in bytes of the LDGs.
    int BYTES_PER_LDG = 16,
    // Do we disable LDGSTS even on an architecture that has it?
    bool DISABLE_LDGSTS = false,
    // The ancestor/base class. See docs/gmem_tile.md.
    typename Ancestor =
        Gmem_tile_base_a<Traits, Cta_tile, Input_related, BYTES_PER_PACKET, BYTES_PER_LDG>>
struct Gmem_tile_a
    : public xmma::Ldgsts_selector<
          Traits,
          xmma::gemm::Rebase_gmem_tile_with_ldgsts<Ancestor, typename Ancestor::LDGSTS_CFG>,
          xmma::gemm::Rebase_gmem_tile_with_ldg_and_sts<Ancestor>,
          DISABLE_LDGSTS>::Class {

    // The base class.
    using Base_ = typename xmma::Ldgsts_selector<
        Traits,
        xmma::gemm::Rebase_gmem_tile_with_ldgsts<Ancestor, typename Ancestor::LDGSTS_CFG>,
        xmma::gemm::Rebase_gmem_tile_with_ldg_and_sts<Ancestor>,
        DISABLE_LDGSTS>::Class;

    // Ctor.
    template <typename Params>
    inline __device__ Gmem_tile_a( const Params& params, void* smem, const dim3& bidx, int tidx )
        : Base_( params, smem, bidx, tidx ) {
    }
};

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// T I L E   B
//
////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits_,
          typename Cta_tile_,
          typename Input_related_,
          int BYTES_PER_PACKET_,
          int BYTES_PER_LDG_>
struct Gmem_tile_base_b {

    // Make sure we use 16B per LDG for the moment.
    static_assert( BYTES_PER_LDG_ == 16, "" );

    // The traits class.
    using Traits = Traits_;
    // The CTA tile descriptor.
    using Cta_tile = Cta_tile_;
    // The Input_related
    using Input_related = Input_related_;
    // The dimensions of the tile.
    enum { M = Cta_tile::N, N = Cta_tile::K };
    // The size in bits of each element.
    enum { BITS_PER_ELT = Traits::BITS_PER_ELEMENT_B };

    // The size of a packet of interleaved elements.
    // For group colw each thread load one pack ( only 4/8/16 elems vaild)
    enum { BYTES_PER_PACKET = Cta_tile::GROUPS > 2 ? 16 : BYTES_PER_PACKET_ };
    // The number of elements per packet.
    enum { ELTS_PER_PACKET = BYTES_PER_PACKET_ * 8 / BITS_PER_ELT };
    // The number of valid elements per packet for group colw.
    enum { ELTS_PER_GROUP = Cta_tile::K / Cta_tile::GROUPS };
    // Make sure a single loop iteration consumes at least one packet.
    static_assert( Cta_tile::K % ELTS_PER_PACKET == 0, "" );

    // The unroll factor for LDGS
    enum { LDGS_UNROLL = 16 / BYTES_PER_LDG_ };
    // The size in bytes of each LDG.
    enum { BYTES_PER_LDG = LDGS_UNROLL * BYTES_PER_LDG_ };
    // The number of elements per LDG.128.
    enum { ELTS_PER_LDG = BYTES_PER_LDG_ * 8 / BITS_PER_ELT };
    // Make sure there are more elements in a packet than loaded per LDG.
    static_assert( BYTES_PER_PACKET % BYTES_PER_LDG == 0, "" );
    // The number of threads needed to load a single packet.
    enum { THREADS_PER_PACKET = BYTES_PER_PACKET / BYTES_PER_LDG };

    // The number of columns in the matrix. One element is a packet.
    enum { COLS = Cta_tile::K / ELTS_PER_PACKET };
    // The number of packets loaded per LDG.
    enum { PACKETS_PER_LDG = Cta_tile::THREADS_PER_CTA / THREADS_PER_PACKET };
    // The number of columns loaded per LDG. It must be COLS or it won't work (move/preds).
    enum { COLS_PER_LDG = COLS };
    // The number of rows loaded per LDG.
    enum { ROWS_PER_LDG = PACKETS_PER_LDG / COLS_PER_LDG };

    // The number of LDGs needed to load a column.
    enum { LDGS_PER_COL = Cta_tile::N / ROWS_PER_LDG };
    // The number of LDGs needed to load a row.
    enum { LDGS_PER_ROW = COLS / COLS_PER_LDG };
    // Make sure we have at most one LDG per row. Or, we have to use multiple predicate regs.
    static_assert( LDGS_PER_ROW == 1, "" );
    // The total number of LDGS.
    enum { LDGS = LDGS_PER_COL * LDGS_PER_ROW * LDGS_UNROLL };
    // The number of predicate registers.
    enum { PRED_REGS = xmma::Compute_number_of_pred_regs<LDGS>::VALUE };
    // The extra amount of shared memory needed by that tile.
    enum { BYTES_PER_EXTRA_SMEM = 0 };
    // LDGSTS config.
    using LDGSTS_CFG = xmma::Ldgsts_config<true>;

    // Ctor.
    template <typename Params>
    inline __device__ Gmem_tile_base_b( const Params& params, void* )
        : ptr_( reinterpret_cast<const char*>( params.flt_gmem ) ) {
    }

    enum { FLT_T = Input_related::FLT_T };
    enum { FLT_R = Input_related::FLT_R };
    enum { FLT_S = Input_related::FLT_S };
    enum { STATIC_FILTER_SIZE = Input_related::STATIC_FILTER_SIZE };

    // Ctor.
    template <typename Params>
    inline __device__ Gmem_tile_base_b( const Params& params, void*, const dim3& bidx, int tidx )
        : Gmem_tile_base_b( params, nullptr ) {

        // The coordinates of the thread.
        int row = bidx.y * Cta_tile::N + tidx / THREADS_PER_PACKET % ROWS_PER_LDG;
        int col = tidx / THREADS_PER_PACKET / ROWS_PER_LDG;
        int ele_offset_in_packet = tidx % THREADS_PER_PACKET * ELTS_PER_LDG;
        // The C dimension. The filter is packed as C/Cta_tile::K * RS * K * Cta_tile::K.
        int c = ele_offset_in_packet + ( Cta_tile::GROUPS > 1 ? col * ELTS_PER_PACKET : 0 );

        // For each channel, a given thread loads one or more filters.
        int k[LDGS];
#pragma unroll
        for( int ni = 0; ni < LDGS; ++ni ) {
            k[ni] = row + ni * ROWS_PER_LDG;
        }

// Compute the offsets for the N dimension.
#pragma unroll
        for( int ni = 0; ni < LDGS; ++ni ) {
            offsets_[ni] =
                ( Cta_tile::GROUPS > 1 ? 0 : col * params.k * params.g * ELTS_PER_PACKET ) +
                k[ni] * ELTS_PER_PACKET + c;
        }

        // Compute the mask for the filters.
        int filter_trs_per_cta;
        if( STATIC_FILTER_SIZE ) {
            filter_trs_per_cta = FLT_T * FLT_R * FLT_S;
        } else {
            filter_trs_per_cta = params.filter_trs_per_cta;
        }

        const int col_is_valid = ( Cta_tile::GROUPS > 1 )
                                     ? ( c < params.c && ele_offset_in_packet < ELTS_PER_GROUP )
                                     : ( col * ELTS_PER_PACKET < filter_trs_per_cta * params.c );

        uint32_t preds[LDGS];
        for( int ni = 0; ni < LDGS; ++ni ) {
            preds[ni] = col_is_valid && k[ni] < params.k * params.g;
        }
        preds_[0] = xmma::pack_predicates( preds );

        // Precompute the masks for the residue.
        int gemm_k = params.loop_residue_k + col * ELTS_PER_PACKET;
        this->residue_mask_ = 0u;
        if( gemm_k < params.c * filter_trs_per_cta ) {
            this->residue_mask_ = uint32_t( -1 );
        }
    }

    // Compute the global memory pointers.
    inline __device__ void compute_load_pointers( const void* ( &ptrs )[LDGS] ) const {
#pragma unroll
        for( int ii = 0; ii < LDGS; ++ii ) {
            ptrs[ii] = this->ptr_ + Traits::offset_in_bytes_b( this->offsets_[ii] );
        }
    }

    // Disable the loads.
    inline __device__ void disable_loads() {
        preds_[0] = 0u;
    }

    // Move the pointers and update the predicates for R2P/P2R.
    inline __device__ void move( int next_trsi, int64_t delta ) {
        ptr_ += delta;
    }

    // The residue to "fix" the predicates.
    inline __device__ void residue() {
        this->preds_[0] &= this->residue_mask_;
    }

    // The base pointer.
    const char* ptr_;
    // The associated offsets.
    int offsets_[LDGS];
    // The masks.
    uint32_t preds_[PRED_REGS], residue_mask_;
};

///////////////////////////////////////////////////////////////////////////////////////////////////

template <
    // The instruction traits.
    typename Traits,
    // The CTA tile description.
    typename Cta_tile,
    // Template parameters which are related to input parameters like filter size.
    typename Input_related,
    // The size in bytes of the packets.
    int BYTES_PER_PACKET,
    // The size in bytes of the LDGs.
    int BYTES_PER_LDG = 16,
    // Do we disable LDGSTS even on an architecture that has it?
    bool DISABLE_LDGSTS = false,
    // The ancestor/base class. See docs/gmem_tile.md.
    typename Ancestor =
        Gmem_tile_base_b<Traits, Cta_tile, Input_related, BYTES_PER_PACKET, BYTES_PER_LDG>>
struct Gmem_tile_b
    : public xmma::Ldgsts_selector<
          Traits,
          xmma::gemm::Rebase_gmem_tile_with_ldgsts<Ancestor, typename Ancestor::LDGSTS_CFG>,
          xmma::gemm::Rebase_gmem_tile_with_ldg_and_sts<Ancestor>,
          DISABLE_LDGSTS>::Class {

    // The base class.
    using Base_ = typename xmma::Ldgsts_selector<
        Traits,
        xmma::gemm::Rebase_gmem_tile_with_ldgsts<Ancestor, typename Ancestor::LDGSTS_CFG>,
        xmma::gemm::Rebase_gmem_tile_with_ldg_and_sts<Ancestor>,
        DISABLE_LDGSTS>::Class;

    // Ctor.
    template <typename Params>
    inline __device__ Gmem_tile_b( const Params& params, void* smem, const dim3& bidx, int tidx )
        : Base_( params, smem, bidx, tidx ) {
    }
};

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// T I L E   C
//
///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits, typename Cta_tile, bool FAST_EPILOGUE = true>
struct Gmem_tile_epilogue {};

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// H M M A: N C/8 H W 8 (16  B Y T E S  I N T E R L E A V E)
//
///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Cta_tile>
struct Gmem_tile_epilogue<xmma::Volta_hmma_fp32_interleaved_traits, Cta_tile, false> {
    
    using Layout = xmma::Col_interleaved;

    // The traits class.
    using Traits = xmma::Volta_hmma_fp32_interleaved_traits;

    // The XMMA tile.
    using Xmma_tile = typename Traits::Xmma_tile<Cta_tile>;
    // The fragment.
    using Fragment_c = xmma::Fragment_interleaved_c<Traits, Cta_tile>;

    // The size of a packet of interleaved elements.
    enum { BYTES_PER_PACKET = 16 };
    // The number of elements per packet.
    enum { ELEMENTS_PER_PACKET = BYTES_PER_PACKET * 8 / Traits::BITS_PER_ELEMENT_C };
    // It must be 8.
    static_assert( ELEMENTS_PER_PACKET == 8, "" );
    // The number of packets per XMMA.
    enum {
        PACKETS_PER_XMMA_M = Xmma_tile::M_PER_XMMA / Xmma_tile::THREADS_PER_XMMA_M,
        PACKETS_PER_XMMA_N = Xmma_tile::N_PER_XMMA / ELEMENTS_PER_PACKET
    };
    // The number of packets per CTA in the N dimension.
    enum { PACKETS_PER_N = Cta_tile::N / ELEMENTS_PER_PACKET };

    // The number of bytes per STG.
    enum { BYTES_PER_STG = Fragment_c::NUM_REGS * 4 };
    // It must be 4.
    static_assert( BYTES_PER_STG == 4, "" );

    // How many elements per STG.
    enum { ELEMENTS_PER_STG = BYTES_PER_STG * 8 / Traits::BITS_PER_ELEMENT_C };

    // // The number of threads needed to output a single packet.
    // enum { THREADS_PER_PACKET = BYTES_PER_PACKET / BYTES_PER_STG };
    // // It must be 4.
    // static_assert(THREADS_PER_PACKET == 4, "");

    // The number of STGS needed to output the rows produced by a CTA-wide XMMA.
    enum { STGS = PACKETS_PER_XMMA_M * PACKETS_PER_XMMA_N * Xmma_tile::XMMAS_N };

    // Ctor.
    template <typename Params>
    inline __device__
    Gmem_tile_epilogue( const Params& params, int bidm, int bidn, int bidz, int tidx )
        : params_pq_( params.pq ), params_k_( params.k ),
          params_out_ptr_( reinterpret_cast<char*>( params.out_gmem ) ),
          params_res_ptr_( reinterpret_cast<const char*>( params.res_gmem ) ) {

        // For HMMA.F32 on Volta, the aclwmulators are distributed as shown on the following dia-
        // gram. It is for a 16x16 XMMA tile and the numbers represent the owning threads:
        //
        //       0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15
        //    --------------------------------------------------
        //  0 |  0  0  2  2  8  8 10 10  0  0  2  2  8  8 10 10
        //  1 |  1  1  3  3  9  9 11 11  1  1  3  3  9  9 11 11
        //  2 |  0  0  2  2  8  8 10 10  0  0  2  2  8  8 10 10
        //  3 |  1  1  3  3  9  9 11 11  1  1  3  3  9  9 11 11
        //  4 |  4  4  6  6 12 12 14 14  4  4  6  6 12 12 14 14
        //  5 |  5  5  7  7 13 13 15 15  5  5  7  7 13 13 15 15
        //  6 |  4  4  6  6 12 12 14 14  4  4  6  6 12 12 14 14
        //  7 |  5  5  7  7 13 13 15 15  5  5  7  7 13 13 15 15
        //  8 | 16 16 18 18 24 24 26 26 16 16 18 18 24 24 26 26
        //  9 | 17 17 19 19 25 25 27 27 17 17 19 19 25 25 27 27
        // 10 | 16 16 18 18 24 24 26 26 16 16 18 18 24 24 26 26
        // 11 | 17 17 19 19 25 25 27 27 17 17 19 19 25 25 27 27
        // 12 | 20 20 22 22 28 28 30 30 20 20 22 22 28 28 30 30
        // 13 | 21 21 23 23 29 29 31 31 21 21 23 23 29 29 31 31
        // 14 | 20 20 22 22 28 28 30 30 20 20 22 22 28 28 30 30
        // 15 | 21 21 23 23 29 29 31 31 21 21 23 23 29 29 31 31
        //

        // The number of warps.
        const int WARPS_M = Cta_tile::WARPS_M;
        const int WARPS_N = Cta_tile::WARPS_N;

        // The masks to select the warps.
        const int WARP_MASK_M = xmma::Warp_masks<WARPS_M, WARPS_N, 1>::M;
        const int WARP_MASK_N = xmma::Warp_masks<WARPS_M, WARPS_N, 1>::N;

        // The divisor for the warps.
        const int WARP_DIV_M = 1 * Cta_tile::THREADS_PER_WARP;
        const int WARP_DIV_N = WARPS_M * Cta_tile::THREADS_PER_WARP;

        // The coordinates of the 1st row written by the thread.
        int npq = ( bidm * Cta_tile::M ) + ( tidx & WARP_MASK_M ) / WARP_DIV_M * 16 +
                  ( tidx & 0x10 ) / 2 + ( tidx & 0x05 );

        // The offset inside a packet for each thread.
        int packet_k = ( tidx & 0x08 ) / 2 + ( tidx & 0x02 );

// Compute the offset for all the stores.
#pragma unroll
        for( int ii = 0; ii < Xmma_tile::XMMAS_M; ++ii ) {
#pragma unroll
            for( int jj = 0; jj < PACKETS_PER_XMMA_M; ++jj ) {
                // The location written by the thread.
                int idx = npq + ii * Xmma_tile::M_PER_XMMA_PER_CTA + 2 * jj;

                // Decompose the position into N and PQ.
                int n, pq;
                xmma::fast_divmod( n, pq, idx, params.pq, params.mul_pq, params.shr_pq );

                // Compute the offset in to add to the pointer.
                int offset = -1;
                if( n < params.n && pq < params.pq ) {
                    offset = n * params.pq * params.k + pq * ELEMENTS_PER_PACKET + packet_k;
                }
                npq_[ii * PACKETS_PER_XMMA_M + jj] = offset;
            }
        }

        // Compute the output packet.
        int k = bidn * PACKETS_PER_N + ( tidx & WARP_MASK_N ) / WARP_DIV_N * PACKETS_PER_XMMA_N;
        // Scale by the number of elements per packet.
        k_ = k * ELEMENTS_PER_PACKET;

        // // DEBUG.
        // #pragma unroll
        // for( int mi = 0; mi < Xmma_tile::XMMAS_M*PACKETS_PER_XMMA_M; ++mi ) {
        //     printf("tidx=%3d mi=%d npq=%3d k=%3d\n", tidx, mi, npq_[mi], k_);
        // }
        // // END OF DEBUG.
    }

    // Is a given output valid?
    inline __device__ int compute_output_mask( int mi, int oi ) const {
        // Decompose the output position inside the XMMA.
        const int ki = oi / ( PACKETS_PER_XMMA_M * PACKETS_PER_XMMA_N );
        const int mj = oi % ( PACKETS_PER_XMMA_M * PACKETS_PER_XMMA_N ) / PACKETS_PER_XMMA_N;
        const int kj = oi % ( PACKETS_PER_XMMA_M * PACKETS_PER_XMMA_N ) % PACKETS_PER_XMMA_N;

        // The channel offset.
        const int m = mi * PACKETS_PER_XMMA_M + mj;
        const int k = ki * Cta_tile::WARPS_N * PACKETS_PER_XMMA_N + kj;

        // Is it a valid position?
        return npq_[m] >= 0 && ( k_ + k * ELEMENTS_PER_PACKET ) < params_k_;
    }

    // Load the data from global memory.
    inline __device__ void load( Fragment_c& data, int mi, int oi, int mask, 
                                    const uint64_t mem_desc ) {
        // Decompose the output position inside the XMMA.
        const int ki = oi / ( PACKETS_PER_XMMA_M * PACKETS_PER_XMMA_N );
        const int mj = oi % ( PACKETS_PER_XMMA_M * PACKETS_PER_XMMA_N ) / PACKETS_PER_XMMA_N;
        const int kj = oi % ( PACKETS_PER_XMMA_M * PACKETS_PER_XMMA_N ) % PACKETS_PER_XMMA_N;

        // The position of the XMMA in the output tile.
        const int m = mi * PACKETS_PER_XMMA_M + mj;
        const int k = ki * Cta_tile::WARPS_N * PACKETS_PER_XMMA_N + kj;

        // Load the data.
        const int offset = npq_[m] + ( k_ + k * ELEMENTS_PER_PACKET ) * params_pq_;
        if( mask ) {
            data.ldg( &params_res_ptr_[Traits::offset_in_bytes_c( offset )], mem_desc );
        }
    }
    
    // Load residual from gmem to smem buffers.
    template<typename Smem_tile_a, typename Smem_tile_b>
    inline __device__ void load_residual_to_smem(
                                Smem_tile_a &smem_tile_a,
                                Smem_tile_b &smem_tile_b,
                                int mi, 
                                int oi, 
                                int mask,
                                int xmma_tiles_per_a,
                                int xmma_tiles_per_b,
                                int tidx,
                                uint64_t mem_desc = MEM_DESC_DEFAULT) {
        // Decompose the output position inside the XMMA.
        const int ki = oi / ( PACKETS_PER_XMMA_M * PACKETS_PER_XMMA_N );
        const int mj = oi % ( PACKETS_PER_XMMA_M * PACKETS_PER_XMMA_N ) / PACKETS_PER_XMMA_N;
        const int kj = oi % ( PACKETS_PER_XMMA_M * PACKETS_PER_XMMA_N ) % PACKETS_PER_XMMA_N;

        // The position of the XMMA in the output tile.
        const int m = mi * PACKETS_PER_XMMA_M + mj;
        const int k = ki * Cta_tile::WARPS_N * PACKETS_PER_XMMA_N + kj;

        // Load the data.
        const int offset = npq_[m] + ( k_ + k * ELEMENTS_PER_PACKET ) * params_pq_;
        if( mask ) {
          int xmma_tile_idx = (mi * STGS + oi) % (xmma_tiles_per_a + xmma_tiles_per_b);
          uint32_t smem_ptr;
          
          if (xmma_tile_idx < xmma_tiles_per_a) 
            smem_ptr = smem_tile_a.smem_ + smem_tile_a.smem_write_buffer_ + 
            xmma_tile_idx *  Cta_tile::THREADS_PER_CTA * BYTES_PER_STG + tidx *BYTES_PER_STG;
          else
            smem_ptr = smem_tile_b.smem_ + smem_tile_b.smem_write_buffer_ + 
            (xmma_tile_idx - xmma_tiles_per_a) *  Cta_tile::THREADS_PER_CTA * BYTES_PER_STG + 
            tidx * BYTES_PER_STG;
            
          if (BYTES_PER_STG == 16)
              ldgsts128(smem_ptr, &params_res_ptr_[Traits::offset_in_bytes_c( offset )], true, mem_desc);
          else if (BYTES_PER_STG == 8)
              ldgsts64(smem_ptr, &params_res_ptr_[Traits::offset_in_bytes_c( offset )], true, mem_desc);
          else if (BYTES_PER_STG == 4)
              ldgsts32(smem_ptr, &params_res_ptr_[Traits::offset_in_bytes_c( offset )], true, mem_desc);
        }  
    }

    // Store the data to global memory.
    inline __device__ void store( int mi, int oi, const Fragment_c& data, int mask, const uint64_t mem_desc ) {
        // Decompose the output position inside the XMMA.
        const int ki = oi / ( PACKETS_PER_XMMA_M * PACKETS_PER_XMMA_N );
        const int mj = oi % ( PACKETS_PER_XMMA_M * PACKETS_PER_XMMA_N ) / PACKETS_PER_XMMA_N;
        const int kj = oi % ( PACKETS_PER_XMMA_M * PACKETS_PER_XMMA_N ) % PACKETS_PER_XMMA_N;

        // The position of the XMMA in the output tile.
        const int m = mi * PACKETS_PER_XMMA_M + mj;
        const int k = ki * Cta_tile::WARPS_N * PACKETS_PER_XMMA_N + kj;

        // // DEBUG.
        // printf("STG: tidx=%3d, mi=%d, oi=%d, offset=%4d value=0x%08x\n",
        //     threadIdx.x,
        //     mi,
        //     oi,
        //     npq_[m] + (k_ + k*ELEMENTS_PER_PACKET)*params_pq_,
        //     data.regs[0]);
        // // END OF DEBUG.

        // Store the data.
        const int offset = npq_[m] + ( k_ + k * ELEMENTS_PER_PACKET ) * params_pq_;
        if( mask ) {
            data.stg( &params_out_ptr_[Traits::offset_in_bytes_c( offset )] );
        }
    }

    // The number of output channels.
    const int params_pq_, params_k_;
    // The pointer to global memory.
    char* const params_out_ptr_;
    const char* params_res_ptr_;
    // The offsets for the thread to output its values.
    int npq_[Xmma_tile::XMMAS_M * PACKETS_PER_XMMA_M], k_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename Traits, typename Cta_tile> struct Gmem_tile_imma_epilogue_prefetch {

    // The XMMA tile.
    using Xmma_tile = typename Traits::Xmma_tile<Cta_tile>;
    // The fragment.
    using Fragment_c = xmma::Fragment_interleaved_c<Traits, Cta_tile>;
    // The number of elements per packet.
    enum { ELEMENTS_PER_PACKET = 32 };
    // The size of a packet of interleaved elements.
    enum { BYTES_PER_PACKET = ELEMENTS_PER_PACKET * Traits::BITS_PER_ELEMENT_C / 8 };
    // The number of packets per CTA in the N dimension.
    enum { PACKETS_PER_N = Cta_tile::N / ELEMENTS_PER_PACKET };
    // Cacheline prefetch
    enum { CACHE_LINE_SIZE = 128 };
    enum { PACKETS_PER_THREAD = 128 / BYTES_PER_PACKET };
    enum {
        CACHE_LINE_NUM =
            Cta_tile::M * Cta_tile::N * ( Traits::BITS_PER_ELEMENT_C / 8 ) / CACHE_LINE_SIZE
    };
    enum { CACHE_LINE_NUM_PER_THREAD = Div_up<CACHE_LINE_NUM, Cta_tile::THREADS_PER_CTA>::VALUE };
    enum { COLS = Cta_tile::N / ELEMENTS_PER_PACKET };
    enum { COLS_PER_PREFETCH = COLS / CACHE_LINE_NUM_PER_THREAD };
    enum { THREADS_PER_COL = Cta_tile::M / PACKETS_PER_THREAD };

    // Ctor.
    template <typename Params>
    inline __device__
    Gmem_tile_imma_epilogue_prefetch( const Params& params, int bidm, int bidn, int bidz, int tidx )
        : params_res_ptr_( reinterpret_cast<const char*>( params.res_gmem ) ) {

        int nopq = ( bidm * Cta_tile::M ) + ( tidx % THREADS_PER_COL ) * PACKETS_PER_THREAD;
        int n, opq;
        xmma::fast_divmod( n, opq, nopq, params.opq, params.mul_opq, params.shr_opq );
        // Decompose the position into O and PQ.
        int o, pq;
        xmma::fast_divmod( o, pq, opq, params.pq, params.mul_pq, params.shr_pq );
        // Decompose the position into P and Q.
        int p, q;
        xmma::fast_divmod( p, q, pq, params.q, params.mul_q, params.shr_q );
        // Compute the offset in to add to the pointer.
        int base = 0;
        if( nopq < params.nopq ) {
            base = ( n * params.out_stride_n + o * params.out_stride_d + p * params.out_stride_h +
                     q * params.out_stride_w ) *
                   ELEMENTS_PER_PACKET;
        }

#pragma unroll
        for( int i = 0; i < CACHE_LINE_NUM_PER_THREAD; i++ ) {
            int k = bidn * PACKETS_PER_N + tidx / THREADS_PER_COL + i * COLS_PER_PREFETCH;
            int offset = base + k * params.out_stride_c * ELEMENTS_PER_PACKET;
            prefetch_ptr_[i] = &params_res_ptr_[Traits::offset_in_bytes_c( offset )];

            mask_[i] = ( k * ELEMENTS_PER_PACKET < params.k );
            if( tidx >= CACHE_LINE_NUM ) {
                mask_[i] = false;
            }
        }
    }

    inline __device__ void prefetch() {
        xmma::prefetch_l1( prefetch_ptr_, mask_ );
    }

    const char* params_res_ptr_;
    const char* prefetch_ptr_[CACHE_LINE_NUM_PER_THREAD];
    bool mask_[CACHE_LINE_NUM_PER_THREAD];
};

template <typename Traits, typename Cta_tile, bool FAST_EPILOGUE = false>
struct Gmem_tile_imma_epilogue {

    using Layout = xmma::Row;
    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;
    // The fragment.
    using Fragment_c = xmma::Fragment_interleaved_c<Traits, Cta_tile>;

    // The number of elements per packet.
    enum { ELEMENTS_PER_PACKET = 32 };
    // The size of a packet of interleaved elements.
    enum { BYTES_PER_PACKET = ELEMENTS_PER_PACKET * Traits::BITS_PER_ELEMENT_C / 8 };
    // The number of packets per XMMA.
    enum {
        PACKETS_PER_XMMA_M = Xmma_tile::M_PER_XMMA / Xmma_tile::THREADS_PER_XMMA_M,
        PACKETS_PER_XMMA_N = Xmma_tile::N_PER_XMMA / ELEMENTS_PER_PACKET
    };
    // The number of packets per CTA in the N dimension.
    enum { PACKETS_PER_N = Cta_tile::N / ELEMENTS_PER_PACKET };

    // The number of bytes per STG.
    enum { BYTES_PER_STG = Fragment_c::NUM_REGS * 4 };
    // How many elements per STG.
    enum { ELEMENTS_PER_STG = BYTES_PER_STG * 8 / Traits::BITS_PER_ELEMENT_C };
    // It must be 8.
    static_assert( BYTES_PER_STG == 8 || BYTES_PER_STG == 16 || BYTES_PER_STG == 32, "" );

    // The number of threads needed to output a single packet.
    enum { THREADS_PER_PACKET = BYTES_PER_PACKET / BYTES_PER_STG };
    // It must be 4.
    static_assert( THREADS_PER_PACKET == 4, "" );

    // The number of STGS needed to output the rows produced by a CTA-wide XMMA.
    enum { STGS = PACKETS_PER_XMMA_M * PACKETS_PER_XMMA_N * Xmma_tile::XMMAS_N };

    // Ctor.
    template <typename Params>
    inline __device__
    Gmem_tile_imma_epilogue( const Params& params, int bidm, int bidn, int bidz, int tidx )
        : params_k_( params.k * params.g ), params_stride_k_( params.out_stride_c ),
          params_out_ptr_( reinterpret_cast<char*>( params.out_gmem ) ),
          params_res_ptr_( reinterpret_cast<const char*>( params.res_gmem ) ) {

        // For IMMA, the aclwmulators are distributed as shown on the following dia-
        // gram. It is for a 1/4 of 16x32 XMMA tile and the numbers represent the owning threads:
        //
        //       0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15
        //    --------------------------------------------------
        //  0 |  0  0  1  1  2  2  3  3  0  0  1  1  2  2  3  3
        //  1 |  4  4  5  5  6  6  7  7  4  4  5  5  6  6  7  7
        //  2 |  8  8  9  9 10 10 11 11  8  8  9  9 10 10 11 11
        //  3 | 12 12 13 13 14 14 15 15 12 12 13 13 14 14 15 15
        //  4 | 16 16 17 17 18 18 19 19 16 16 17 17 18 18 19 19
        //  5 | 20 20 21 21 22 22 23 23 20 20 21 21 22 22 23 23
        //  6 | 24 24 25 25 26 26 27 27 24 24 25 25 26 26 27 27
        //  7 | 28 28 29 29 30 30 31 31 28 28 29 29 30 30 31 31
        //

        // The number of warps.
        const int WARPS_M = Cta_tile::WARPS_M;
        const int WARPS_N = Cta_tile::WARPS_N;

        // The masks to select the warps.
        const int WARP_MASK_M = xmma::Warp_masks<WARPS_M, WARPS_N, 1>::M;
        const int WARP_MASK_N = xmma::Warp_masks<WARPS_M, WARPS_N, 1>::N;

        // The divisor for the warps.
        const int WARP_DIV_M = 1 * Cta_tile::THREADS_PER_WARP;
        const int WARP_DIV_N = WARPS_M * Cta_tile::THREADS_PER_WARP;

        // The coordinates of the 1st row written by the thread.
        int nopq = ( bidm * Cta_tile::M ) +
                   ( tidx & WARP_MASK_M ) / WARP_DIV_M * Xmma_tile::M_PER_XMMA +
                   ( tidx & 0x1f ) / Xmma_tile::THREADS_PER_XMMA_N;

        // The offset inside a packet for each thread.
        int packet_k = ( tidx & 0x03 ) * ELEMENTS_PER_STG;

// Compute the offset for all the stores.
#pragma unroll
        for( int ii = 0; ii < Xmma_tile::XMMAS_M; ++ii ) {
#pragma unroll
            for( int jj = 0; jj < PACKETS_PER_XMMA_M; ++jj ) {
                // The location written by the thread.
                int idx = nopq + ii * Xmma_tile::M_PER_XMMA_PER_CTA + 8 * jj;
                if( FAST_EPILOGUE ) {
                    // Decompose the position into N and OPQ.
                    int n, opq;
                    xmma::fast_divmod( n, opq, idx, params.opq, params.mul_opq, params.shr_opq );
                    // Compute the offset in to add to the pointer.
                    int offset = -1;
                    if( idx < params.nopq ) {
                        offset = ( n * params.out_stride_n + opq ) * ELEMENTS_PER_PACKET + packet_k;
                    }
                    nopq_[ii * PACKETS_PER_XMMA_M + jj] = offset;
                } else {
                    // Decompose the position into N and OPQ.
                    int n, opq;
                    xmma::fast_divmod( n, opq, idx, params.opq, params.mul_opq, params.shr_opq );
                    // Decompose the position into O and PQ.
                    int o, pq;
                    xmma::fast_divmod( o, pq, opq, params.pq, params.mul_pq, params.shr_pq );
                    // Decompose the position into P and Q.
                    int p, q;
                    xmma::fast_divmod( p, q, pq, params.q, params.mul_q, params.shr_q );

                    // Compute the offset in to add to the pointer.
                    int offset = -1;
                    if( idx < params.nopq ) {
                        offset = ( n * params.out_stride_n + o * params.out_stride_d +
                                   p * params.out_stride_h + q * params.out_stride_w ) *
                                 ELEMENTS_PER_PACKET;
                        offset += packet_k;
                    }
                    nopq_[ii * PACKETS_PER_XMMA_M + jj] = offset;
                }
            }
        }

        // Compute the output packet.
        k_ = bidn * PACKETS_PER_N + ( tidx & WARP_MASK_N ) / WARP_DIV_N * PACKETS_PER_XMMA_N;
    }

    // Is a given output valid?
    inline __device__ int compute_output_mask( int mi, int oi ) const {
        // Decompose the output position inside the XMMA.
        const int ki = oi / ( PACKETS_PER_XMMA_M * PACKETS_PER_XMMA_N );
        const int mj = oi % ( PACKETS_PER_XMMA_M * PACKETS_PER_XMMA_N ) / PACKETS_PER_XMMA_N;
        const int kj = oi % ( PACKETS_PER_XMMA_M * PACKETS_PER_XMMA_N ) % PACKETS_PER_XMMA_N;

        // The channel offset.
        const int m = mi * PACKETS_PER_XMMA_M + mj;
        const int k = ki * Cta_tile::WARPS_N * PACKETS_PER_XMMA_N + kj;

        // Is it a valid position?
        return nopq_[m] >= 0 && ( ( k_ + k ) * ELEMENTS_PER_PACKET ) < params_k_;
    }

    // Load the data from global memory.
    inline __device__ void
    load( Fragment_c& data, int mi, int oi, int mask, const uint64_t mem_desc ) {
        // Decompose the output position inside the XMMA.
        const int ki = oi / ( PACKETS_PER_XMMA_M * PACKETS_PER_XMMA_N );
        const int mj = oi % ( PACKETS_PER_XMMA_M * PACKETS_PER_XMMA_N ) / PACKETS_PER_XMMA_N;
        const int kj = oi % ( PACKETS_PER_XMMA_M * PACKETS_PER_XMMA_N ) % PACKETS_PER_XMMA_N;

        // The position of the XMMA in the output tile.
        const int m = mi * PACKETS_PER_XMMA_M + mj;
        const int k = ki * Cta_tile::WARPS_N * PACKETS_PER_XMMA_N + kj;

        // Load the data.
        const int offset = nopq_[m] + ( k_ + k ) * params_stride_k_ * ELEMENTS_PER_PACKET;
        if( mask ) {
            data.ldg( &params_res_ptr_[Traits::offset_in_bytes_c( offset )], mem_desc );
        }
    }
    
    // Load residual from gmem to smem buffers.
    template<typename Smem_tile_a, typename Smem_tile_b>
    inline __device__ void load_residual_to_smem(
                                Smem_tile_a &smem_tile_a,
                                Smem_tile_b &smem_tile_b,
                                int mi, 
                                int oi, 
                                int mask,
                                int xmma_tiles_per_a,
                                int xmma_tiles_per_b,
                                int tidx,
                                uint64_t mem_desc = MEM_DESC_DEFAULT) {
    // Decompose the output position inside the XMMA.
    const int ki = oi / ( PACKETS_PER_XMMA_M * PACKETS_PER_XMMA_N );
    const int mj = oi % ( PACKETS_PER_XMMA_M * PACKETS_PER_XMMA_N ) / PACKETS_PER_XMMA_N;
    const int kj = oi % ( PACKETS_PER_XMMA_M * PACKETS_PER_XMMA_N ) % PACKETS_PER_XMMA_N;

    // The position of the XMMA in the output tile.
    const int m = mi * PACKETS_PER_XMMA_M + mj;
    const int k = ki * Cta_tile::WARPS_N * PACKETS_PER_XMMA_N + kj;  
    // Load the data.
    const int offset = nopq_[m] + ( k_ + k ) * params_stride_k_ * ELEMENTS_PER_PACKET;
    if( mask ) {
        int xmma_tile_idx = (mi * STGS + oi) % (xmma_tiles_per_a + xmma_tiles_per_b);
        uint32_t smem_ptr;
        
        if (xmma_tile_idx < xmma_tiles_per_a) 
          smem_ptr = smem_tile_a.smem_ + smem_tile_a.smem_write_buffer_ + 
          xmma_tile_idx *  Cta_tile::THREADS_PER_CTA * BYTES_PER_STG + tidx *BYTES_PER_STG;
        else
          smem_ptr = smem_tile_b.smem_ + smem_tile_b.smem_write_buffer_ + 
          (xmma_tile_idx - xmma_tiles_per_a) *  Cta_tile::THREADS_PER_CTA * BYTES_PER_STG + 
          tidx * BYTES_PER_STG;
          
        if (BYTES_PER_STG == 16)
            ldgsts128(smem_ptr, &params_res_ptr_[Traits::offset_in_bytes_c( offset )], true, mem_desc);
        else if (BYTES_PER_STG == 8)
            ldgsts64(smem_ptr, &params_res_ptr_[Traits::offset_in_bytes_c( offset )], true, mem_desc);
        else if (BYTES_PER_STG == 4)
            ldgsts32(smem_ptr, &params_res_ptr_[Traits::offset_in_bytes_c( offset )], true, mem_desc);
    }    
}

    // Store the data to global memory.
    inline __device__ void
    store( int mi, int oi, const Fragment_c& data, int mask, const uint64_t mem_desc ) {
        // Decompose the output position inside the XMMA.
        const int ki = oi / ( PACKETS_PER_XMMA_M * PACKETS_PER_XMMA_N );
        const int mj = oi % ( PACKETS_PER_XMMA_M * PACKETS_PER_XMMA_N ) / PACKETS_PER_XMMA_N;
        const int kj = oi % ( PACKETS_PER_XMMA_M * PACKETS_PER_XMMA_N ) % PACKETS_PER_XMMA_N;

        // The position of the XMMA in the output tile.
        const int m = mi * PACKETS_PER_XMMA_M + mj;
        const int k = ki * Cta_tile::WARPS_N * PACKETS_PER_XMMA_N + kj;

        // Store the data.
        const int offset = nopq_[m] + ( k_ + k ) * params_stride_k_ * ELEMENTS_PER_PACKET;

        if( mask ) {
            data.stg( &params_out_ptr_[Traits::offset_in_bytes_c( offset )], mem_desc );
        }
    }

    // The number and stride of output channels.
    const int params_stride_k_, params_k_;
    // The pointer to global memory.
    char* const params_out_ptr_;
    const char* params_res_ptr_;
    // The offsets for the thread to output its values.
    int nopq_[Xmma_tile::XMMAS_M * PACKETS_PER_XMMA_M], k_;
};

///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Cta_tile, bool FAST_EPILOGUE>
struct Gmem_tile_epilogue<xmma::Volta_imma_interleaved_int8_int32_traits, Cta_tile, FAST_EPILOGUE>
    : Gmem_tile_imma_epilogue<xmma::Volta_imma_interleaved_int8_int32_traits,
                              Cta_tile,
                              FAST_EPILOGUE> {
    using Base = Gmem_tile_imma_epilogue<xmma::Volta_imma_interleaved_int8_int32_traits,
                                         Cta_tile,
                                         FAST_EPILOGUE>;
    template <typename Params>
    inline __device__
    Gmem_tile_epilogue( const Params& params, int bidm, int bidn, int bidz, int tidx )
        : Base( params, bidm, bidn, bidz, tidx ) {
    }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Cta_tile, bool FAST_EPILOGUE>
struct Gmem_tile_epilogue<xmma::Turing_imma_interleaved_int8_int32_traits, Cta_tile, FAST_EPILOGUE>
    : Gmem_tile_imma_epilogue<xmma::Turing_imma_interleaved_int8_int32_traits,
                              Cta_tile,
                              FAST_EPILOGUE> {
    using Base = Gmem_tile_imma_epilogue<xmma::Turing_imma_interleaved_int8_int32_traits,
                                         Cta_tile,
                                         FAST_EPILOGUE>;
    template <typename Params>
    inline __device__
    Gmem_tile_epilogue( const Params& params, int bidm, int bidn, int bidz, int tidx )
        : Base( params, bidm, bidn, bidz, tidx ) {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Cta_tile,
          typename Input_type,
          typename Output_type,
          bool IS_GELU,
          bool IS_EPIFADD,
          bool IS_SWISH,
          bool FAST_EPILOGUE>
struct Gmem_tile_epilogue<xmma::Ampere_imma_interleaved_traits<Input_type,
                                                               Output_type,
                                                               IS_GELU,
                                                               IS_EPIFADD,
                                                               IS_SWISH>,
                          Cta_tile,
                          FAST_EPILOGUE>
    : Gmem_tile_imma_epilogue<xmma::Ampere_imma_interleaved_traits<Input_type,
                                                                   Output_type,
                                                                   IS_GELU,
                                                                   IS_EPIFADD,
                                                                   IS_SWISH>,
                              Cta_tile,
                              FAST_EPILOGUE> {
    using Base = Gmem_tile_imma_epilogue<xmma::Ampere_imma_interleaved_traits<Input_type,
                                                                              Output_type,
                                                                              IS_GELU,
                                                                              IS_EPIFADD,
                                                                              IS_SWISH>,
                                         Cta_tile,
                                         FAST_EPILOGUE>;
    template <typename Params>
    inline __device__
    Gmem_tile_epilogue( const Params& params, int bidm, int bidn, int bidz, int tidx )
        : Base( params, bidm, bidn, bidz, tidx ) {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace interleaved_fprop
}  // namespace implicit_gemm
}  // namespace xmma
