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
#include <xmma/implicit_gemm/dgrad/utils.h>
#include <xmma/implicit_gemm/utils.h>
#include <xmma/gemm/gmem_tile.h>

#define MIN(m, n) ((m < n) ? m : n)
namespace xmma {
namespace implicit_gemm {
namespace interleaved_dgrad {

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
struct Gmem_tile_base_a : public gemm::Gmem_tile_base<Traits_, Cta_tile_, Cta_tile_::M, Cta_tile_::K,
                                                      Traits_::BITS_PER_ELEMENT_A, BYTES_PER_LDG_> {

    // Make sure we use 16B per LDG for the moment.
    static_assert( BYTES_PER_LDG_ == 16, "" );

    // The traits class.
    using Traits = Traits_;
    // The CTA tile descriptor.
    using Cta_tile = Cta_tile_;
    // Template parameters which are related to input parameters like filter size.
    using Input_related = Input_related_;
    using Base_ = gemm::Gmem_tile_base<Traits, Cta_tile, Cta_tile::M, Cta_tile::K,
                                       Traits::BITS_PER_ELEMENT_A, BYTES_PER_LDG_>;
    // The dimensions of the tile.
    //enum { M = Cta_tile::M, N = Cta_tile::K };
    // The size in bits of each element.
    enum { BITS_PER_ELT = Traits::BITS_PER_ELEMENT_A };

    // The size of a packet of interleaved elements.
    enum { BYTES_PER_PACKET = BYTES_PER_PACKET_ };
    // The number of elements per packet.
    enum { ELTS_PER_PACKET = BYTES_PER_PACKET * 8 / BITS_PER_ELT };
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
    enum {
        BYTES_PER_EXTRA_SMEM =
            xmma::Max<Cta_tile::THREADS_PER_CTA, Cta_tile::M>::VALUE * sizeof( uint2 )
    };

    enum { FLT_T = Input_related::FLT_T };
    enum { FLT_R = Input_related::FLT_R };
    enum { FLT_S = Input_related::FLT_S };
    enum { STATIC_FILTER_SIZE = Input_related::STATIC_FILTER_SIZE };

    // Ctor.
    template <typename Params>
    inline __device__ Gmem_tile_base_a( const Params& params, void* smem, int bidz )
        : Base_( params, smem, params.c * params.g, params.img_gmem, bidz ),
          params_filter_trs_per_cta_( params.filter_trs_per_cta ),
          ptr_( reinterpret_cast<const char*>( params.img_gmem ) ),
          smem_( xmma::get_smem_pointer( smem ) ) {
    }

    // Ctor.
    template <typename Params>
    inline __device__
    Gmem_tile_base_a( const Params& params, void* smem, const dim3& bidx, int tidx )
        : Gmem_tile_base_a( params, smem, bidx.z ) {

        // split between C and RS as we work on multiple "filter taps".
        int filter_t_per_cta, filter_r_per_cta, filter_s_per_cta;
        int filter_trs_per_cta, filter_rs_per_cta;
        if( STATIC_FILTER_SIZE ) {
            filter_t_per_cta = FLT_T;
            filter_r_per_cta = FLT_R;
            filter_s_per_cta = FLT_S;
            filter_rs_per_cta = FLT_R * FLT_S;
            filter_trs_per_cta = FLT_T * FLT_R * FLT_S;
        } else {
            filter_t_per_cta = params.filter_t_per_cta;
            filter_r_per_cta = params.filter_r_per_cta;
            filter_s_per_cta = params.filter_s_per_cta;
            filter_rs_per_cta = params.filter_rs_per_cta;
            filter_trs_per_cta = params.filter_trs_per_cta;
        }

        // The coordinates of the thread.
        int col = tidx / THREADS_PER_PACKET / ROWS_PER_LDG;

        // The following code works only because LDGS_PER_ROW == 1.
        int trs, rs, k, t, r, s;
        xmma::fast_divmod( k,
                           trs,
                           col,
                           filter_trs_per_cta,
                           params.mul_filter_trs_per_cta,
                           params.shr_filter_trs_per_cta );
        xmma::fast_divmod( t,
                           rs,
                           trs,
                           filter_rs_per_cta,
                           params.mul_filter_rs_per_cta,
                           params.shr_filter_rs_per_cta );
        xmma::fast_divmod(
            r, s, rs, filter_s_per_cta, params.mul_filter_s_per_cta, params.shr_filter_s_per_cta );

        t = ( t * params.dilation[0] ) / params.stride[0];
        r = ( r * params.dilation[1] ) / params.stride[1];
        s = ( s * params.dilation[2] ) / params.stride[2];
        int delta = Traits::offset_in_bytes_a( -t * static_cast<int64_t>( params.out_stride_d ) -
                                               r * static_cast<int64_t>( params.out_stride_h ) -
                                               s * static_cast<int64_t>( params.out_stride_w ) );

        if( Input_related::IS_SIMPLE_1x1x1 ) {
#pragma unroll
            for( int mi = 0; mi < LDGS; mi++ ) {
                // The position in the row/col.
                int ndhw = bidx.x * Cta_tile::M + tidx / THREADS_PER_PACKET % ROWS_PER_LDG +
                           mi * ROWS_PER_LDG;

                // Decompose ndhw into n and dhw.
                int dhw, n;
                xmma::fast_divmod( n, dhw, ndhw, params.dhw, params.mul_dhw, params.shr_dhw );
                offsets_[mi] =
                    n * params.out_stride_n + dhw * params.out_stride_w + k * params.out_stride_c;

                masks_[mi] =
                    ( ndhw < params.ndhw && k * ELTS_PER_PACKET < params.k ) ? uint32_t( -1 ) : 0u;
            }
        } else {
            // We first compute the coordinates of all the pixels loaded by this CTA.
            const int STEPS = xmma::Div_up<Cta_tile::M, Cta_tile::THREADS_PER_CTA>::VALUE;
#pragma unroll
            for( int step = 0; step < STEPS; ++step ) {

                int idx_in_cta = tidx + step * Cta_tile::THREADS_PER_CTA;

                // The masks for the predicates.
                const uint32_t MASK_T = xmma::implicit_gemm::Build_mask_t<FLT_T, FLT_R, FLT_S>::VALUE;
                const uint32_t MASK_R = xmma::implicit_gemm::Build_mask_r<FLT_T, FLT_R, FLT_S>::VALUE;
                const uint32_t MASK_S = xmma::implicit_gemm::Build_mask_s<FLT_T, FLT_R, FLT_S>::VALUE;

                // For each LDG, compute the NPQ decomposition, the pointer and issue the 1st LDG.
                int n, o, p, q;

                // The position in the row/col.
                int ndhw = bidx.x * Cta_tile::M + idx_in_cta;

                uint32_t mask = ( ndhw < params.ndhw ) ? uint32_t( -1 ) : 0u;
                uint32_t offset;

                // Decompose ndhw into n and dhw.
                int dhw, hw, d, h, w;
                xmma::fast_divmod( n, dhw, ndhw, params.dhw, params.mul_dhw, params.shr_dhw );
                xmma::fast_divmod( d, hw, dhw, params.hw, params.mul_hw, params.shr_hw );
                xmma::fast_divmod( h, w, hw, params.w, params.mul_w, params.shr_w );

                // Compute h and w. We implement a cross-correlation.
                o = d + params.pad[0][0];
                p = h + params.pad[1][0];
                q = w + params.pad[2][0];

                uint32_t o_div, o_mod;
                if( o >= 0 ) {
                    xmma::fast_divmod_v2( o_div,
                                          o_mod,
                                          o,
                                          params.stride[0],
                                          params.mul_stride[0],
                                          params.shr_stride[0] );
                } else {
                    o_mod = 1;
                }
                uint32_t p_div, p_mod;
                if( p >= 0 ) {
                    xmma::fast_divmod_v2( p_div,
                                          p_mod,
                                          p,
                                          params.stride[1],
                                          params.mul_stride[1],
                                          params.shr_stride[1] );
                } else {
                    p_mod = 1;
                }
                uint32_t q_div, q_mod;
                if( q >= 0 ) {
                    xmma::fast_divmod_v2( q_div,
                                          q_mod,
                                          q,
                                          params.stride[2],
                                          params.mul_stride[2],
                                          params.shr_stride[2] );
                } else {
                    q_mod = 1;
                }

                offset = n * params.out_stride_n + o_div * params.out_stride_d +
                         p_div * params.out_stride_h + q_div * params.out_stride_w;

// Finalize the masks for T.
#pragma unroll
                for( int ti = 0; ti < filter_t_per_cta; ++ti ) {
                    uint32_t mask_t;
                    if( STATIC_FILTER_SIZE ) {
                        mask_t = ( MASK_T << ( ti * FLT_R * FLT_S ) );
                    } else {
                        mask_t = ( params.mask_t << ( ti * params.filter_rs_per_cta ) );
                    }
                    mask_t ^= uint32_t( -1 );
                    uint32_t o_div, o_mod;
                    xmma::fast_divmod_v2( o_div,
                                          o_mod,
                                          o - ti * params.dilation[0],
                                          params.stride[0],
                                          params.mul_stride[0],
                                          params.shr_stride[0] );
                    if( ( o_div ) >= params.o || o_mod ) {
                        mask = mask & mask_t;
                    }
                }

// Finalize the masks for R.
#pragma unroll
                for( int ri = 0; ri < filter_r_per_cta; ++ri ) {
                    uint32_t mask_r;
                    if( STATIC_FILTER_SIZE ) {
                        mask_r = ( MASK_R << ( ri * FLT_S ) );
                    } else {
                        mask_r = ( params.mask_r << ( ri * params.filter_s_per_cta ) );
                    }
                    mask_r ^= uint32_t( -1 );
                    uint32_t p_div, p_mod;
                    xmma::fast_divmod_v2( p_div,
                                          p_mod,
                                          p - ri * params.dilation[1],
                                          params.stride[1],
                                          params.mul_stride[1],
                                          params.shr_stride[1] );
                    if( ( p_div ) >= params.p || p_mod ) {
                        mask = mask & mask_r;
                    }
                }

// Finalize the masks for S.
#pragma unroll
                for( int si = 0; si < filter_s_per_cta; ++si ) {
                    uint32_t mask_s;
                    if( STATIC_FILTER_SIZE ) {
                        mask_s = ( MASK_S << si );
                    } else {
                        mask_s = ( params.mask_s << si );
                    }
                    mask_s ^= uint32_t( -1 );
                    uint32_t q_div, q_mod;
                    xmma::fast_divmod_v2( q_div,
                                          q_mod,
                                          q - si * params.dilation[2],
                                          params.stride[2],
                                          params.mul_stride[2],
                                          params.shr_stride[2] );
                    if( ( q_div ) >= params.q || q_mod ) {
                        mask = mask & mask_s;
                    }
                }

                uint32_t ptr = this->smem_ + idx_in_cta * sizeof( uint2 );
                uint2 tmp;
                tmp.x = mask;
                tmp.y = offset;
                xmma::sts( ptr, tmp );

            }  // STEPS

            __syncthreads();

#pragma unroll
            for( int mi = 0; mi < LDGS; mi++ ) {
                int idx = tidx / THREADS_PER_PACKET % ROWS_PER_LDG + mi * ROWS_PER_LDG;
                uint2 p;
                xmma::lds( p, this->smem_ + idx * sizeof( uint2 ) );

                // Extract the coordinates of that pixel.
                if( k * ELTS_PER_PACKET >= params.k ) {
                    masks_[mi] = 0u;
                } else {
                    masks_[mi] = p.x;
                }
                offsets_[mi] = p.y + k * params.out_stride_c;
            }

        }  // SIMPLE_1X1X1

        ptr_ += Traits::offset_in_bytes_a( tidx % THREADS_PER_PACKET * ELTS_PER_LDG ) + delta;
        xmma::implicit_gemm::pack_predicates( this->preds_, masks_, 1u << trs );

        // Precompute the masks for the residue.
        k = params.loop_residue_k + col * ELTS_PER_PACKET;
        this->residue_mask_ = 0u;
        if( k < params.k * filter_trs_per_cta ) {
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
    // Compute the global memory pointers.
    template< int phase >
    inline __device__ void compute_load_pointers_per_phase( const void* ( &ptrs )[Base_::LDGS_PER_PHASE] ) const {
#pragma unroll
        for( int ii = phase * Base_::LDGS_PER_PHASE; ii < MIN((phase + 1) * Base_::LDGS_PER_PHASE, Base_::LDGS); ++ii ) {
            ptrs[ii - phase*Base_::LDGS_PER_PHASE] = this->ptr_ + Traits::offset_in_bytes_a( this->offsets_[ii] );
            
        }
    }
    // Disable the loads.
    inline __device__ void disable_loads() {
        this->preds_[0] = 0u;
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

    /////////////////////////////////////////////////////////////////
    // Main loop fusion support
    //   d = matmul(f(a, x0, ...), g(b, y0, ...)) + c
    /////////////////////////////////////////////////////////////////
    template < typename Callback_fuse >
    inline __device__ void apply_fuse( Callback_fuse &fuse ) {  }

    template < typename Callback_fuse >
    inline __device__ void load_vectors_m( Callback_fuse &fuse ) {  }

    template < typename Callback_fuse >
    inline __device__ void load_vectors_n( Callback_fuse &fuse ) {  }

    // The product of each dim of filter
    const int params_filter_trs_per_cta_;
    // The pointer.
    const char* ptr_;
    uint32_t smem_;
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
    : public xmma::Ldgsts_selector<Traits,
                                   xmma::gemm::Rebase_gmem_tile_with_ldgsts<Ancestor>,
                                   xmma::gemm::Rebase_gmem_tile_with_ldg_and_sts<Ancestor>,
                                   DISABLE_LDGSTS>::Class {

    // The base class.
    using Base_ =
        typename xmma::Ldgsts_selector<Traits,
                                       xmma::gemm::Rebase_gmem_tile_with_ldgsts<Ancestor>,
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
struct Gmem_tile_base_b : public gemm::Gmem_tile_base<Traits_, Cta_tile_, Cta_tile_::K, Cta_tile_::N, Traits_::BITS_PER_ELEMENT_B, BYTES_PER_LDG_> {

    // Make sure we use 16B per LDG for the moment.
    static_assert( BYTES_PER_LDG_ == 16, "" );

    // The traits class.
    using Traits = Traits_;
    // The CTA tile descriptor.
    using Cta_tile = Cta_tile_;
    // The Input_related
    using Input_related = Input_related_;
    // The base class.
    using Base_ = gemm::Gmem_tile_base<Traits, Cta_tile, Cta_tile::K, Cta_tile::N,
                                       Traits::BITS_PER_ELEMENT_B, BYTES_PER_LDG_>;

    // The dimensions of the tile.
    //enum { M = Cta_tile::N, N = Cta_tile::K };
    // The size in bits of each element.
    enum { BITS_PER_ELT = Traits::BITS_PER_ELEMENT_B };

    // The size of a packet of interleaved elements.
    enum { BYTES_PER_PACKET = BYTES_PER_PACKET_ };
    // The number of elements per packet.
    enum { ELTS_PER_PACKET = BYTES_PER_PACKET * 8 / BITS_PER_ELT };
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

    // Ctor.
    template <typename Params>
    inline __device__ Gmem_tile_base_b( const Params& params, void* smem, int bidz)
        : Base_( params, smem, params.c * params.g, params.flt_gmem, bidz )
        , ptr_( reinterpret_cast<const char*>( params.flt_gmem ) )
        , params_c_( params.c ) {
    }

    enum { FLT_T = Input_related::FLT_T };
    enum { FLT_R = Input_related::FLT_R };
    enum { FLT_S = Input_related::FLT_S };
    enum { STATIC_FILTER_SIZE = Input_related::STATIC_FILTER_SIZE };

    // Ctor.
    template <typename Params>
    inline __device__ Gmem_tile_base_b( const Params& params, void* smem, const dim3& bidx, int tidx )
        : Gmem_tile_base_b( params, smem, bidx.z ) {

        // The coordinates of the thread.
        int row = bidx.y * Cta_tile::N + tidx / THREADS_PER_PACKET % ROWS_PER_LDG;
        int col = tidx / THREADS_PER_PACKET / ROWS_PER_LDG;

        // The C dimension. The filter is packed as C/Cta_tile::K * RS * K * Cta_tile::K.
        int c = tidx % THREADS_PER_PACKET * ELTS_PER_LDG;

        // For each channel, a given thread loads one or more filters.
        int k[LDGS];
#pragma unroll
        for( int ni = 0; ni < LDGS; ++ni ) {
            k[ni] = row + ni * ROWS_PER_LDG;
        }

// Compute the offsets for the N dimension.
#pragma unroll
        for( int ni = 0; ni < LDGS; ++ni ) {
            offsets_[ni] = col * params.c * ELTS_PER_PACKET + k[ni] * ELTS_PER_PACKET + c;
        }

        // Compute the mask for the filters.
        int filter_trs_per_cta;
        if( STATIC_FILTER_SIZE ) {
            filter_trs_per_cta = FLT_T * FLT_R * FLT_S;
        } else {
            filter_trs_per_cta = params.filter_trs_per_cta;
        }
        const int col_is_valid = ( col * ELTS_PER_PACKET < filter_trs_per_cta * params.k );
        uint32_t preds[LDGS];
        for( int ni = 0; ni < LDGS; ++ni ) {
            preds[ni] = col_is_valid && k[ni] < params.c;
        }
        preds_[0] = xmma::pack_predicates( preds );

        // Precompute the masks for the residue.
        int gemm_k = params.loop_residue_k + col * ELTS_PER_PACKET;
        this->residue_mask_ = 0u;
        if( gemm_k < params.k * filter_trs_per_cta ) {
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

    // Compute the global memory pointers.
    template< int phase >
    inline __device__ void compute_load_pointers_per_phase( const void* ( &ptrs )[Base_::LDGS_PER_PHASE] ) const {
#pragma unroll
        for( int ii = phase * Base_::LDGS_PER_PHASE; ii < MIN((phase + 1) * Base_::LDGS_PER_PHASE, Base_::LDGS); ++ii ) {
            ptrs[ii - phase*Base_::LDGS_PER_PHASE] = this->ptr_ + Traits::offset_in_bytes_b( this->offsets_[ii] );
        }
    }

    // Disable the loads.
    inline __device__ void disable_loads() {
        preds_[0] = 0u;
    }

    // Move the pointers and update the predicates for R2P/P2R.
    inline __device__ void move( int next_trsi, int64_t delta ) {
        // ptr_ += delta;
        ptr_ += Traits::offset_in_bytes_b( params_c_ * Cta_tile::K );
    }

    // The residue to "fix" the predicates.
    inline __device__ void residue() {
        this->preds_[0] &= this->residue_mask_;
    }

    /////////////////////////////////////////////////////////////////
    // Main loop fusion support
    //   d = matmul(f(a, x0, ...), g(b, y0, ...)) + c
    /////////////////////////////////////////////////////////////////
    template < typename Callback_fuse >
    inline __device__ void apply_fuse( Callback_fuse &fuse ) {  }

    template < typename Callback_fuse >
    inline __device__ void load_vectors_m( Callback_fuse &fuse ) {  }

    template < typename Callback_fuse >
    inline __device__ void load_vectors_n( Callback_fuse &fuse ) {  }

    // The base pointer.
    const char* ptr_;
    // The associated offsets.
    int offsets_[LDGS];
    // The masks.
    uint32_t preds_[PRED_REGS], residue_mask_;
    int params_c_;
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
    : public xmma::Ldgsts_selector<Traits,
                                   xmma::gemm::Rebase_gmem_tile_with_ldgsts<Ancestor>,
                                   xmma::gemm::Rebase_gmem_tile_with_ldg_and_sts<Ancestor>,
                                   DISABLE_LDGSTS>::Class {

    // The base class.
    using Base_ =
        typename xmma::Ldgsts_selector<Traits,
                                       xmma::gemm::Rebase_gmem_tile_with_ldgsts<Ancestor>,
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
struct Gmem_tile_epilogue<xmma::Volta_hmma_fp32_traits, Cta_tile, false> {

    // The traits class.
    using Traits = xmma::Volta_hmma_fp32_traits;
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
        : params_hw_( params.hw ), params_c_( params.c ),
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
        int nhw = ( bidm * Cta_tile::M ) + ( tidx & WARP_MASK_M ) / WARP_DIV_M * 16 +
                  ( tidx & 0x10 ) / 2 + ( tidx & 0x05 );

        // The offset inside a packet for each thread.
        int packet_k = ( tidx & 0x08 ) / 2 + ( tidx & 0x02 );

// Compute the offset for all the stores.
#pragma unroll
        for( int ii = 0; ii < Xmma_tile::XMMAS_M; ++ii ) {
#pragma unroll
            for( int jj = 0; jj < PACKETS_PER_XMMA_M; ++jj ) {
                // The location written by the thread.
                int idx = nhw + ii * Xmma_tile::M_PER_XMMA_PER_CTA + 2 * jj;

                // Decompose the position into N and PQ.
                int n, hw;
                xmma::fast_divmod( n, hw, idx, params.hw, params.mul_hw, params.shr_hw );

                // Compute the offset in to add to the pointer.
                int offset = -1;
                if( n < params.n && hw < params.hw ) {
                    offset = n * params.hw * params.c + hw * ELEMENTS_PER_PACKET + packet_k;
                }
                nhw_[ii * PACKETS_PER_XMMA_M + jj] = offset;
            }
        }

        // Compute the output packet.
        int k = bidn * PACKETS_PER_N + ( tidx & WARP_MASK_N ) / WARP_DIV_N * PACKETS_PER_XMMA_N;
        // Scale by the number of elements per packet.
        k_ = k * ELEMENTS_PER_PACKET;

        // // DEBUG.
        // #pragma unroll
        // for( int mi = 0; mi < Xmma_tile::XMMAS_M*PACKETS_PER_XMMA_M; ++mi ) {
        //     printf("tidx=%3d mi=%d nhw=%3d k=%3d\n", tidx, mi, nhw_[mi], k_);
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
        return nhw_[m] >= 0 && ( k_ + k * ELEMENTS_PER_PACKET ) < params_c_;
    }

    // Load the data from global memory.
    inline __device__ void load( Fragment_c& data, int mi, int oi, int mask ) {
        // Decompose the output position inside the XMMA.
        const int ki = oi / ( PACKETS_PER_XMMA_M * PACKETS_PER_XMMA_N );
        const int mj = oi % ( PACKETS_PER_XMMA_M * PACKETS_PER_XMMA_N ) / PACKETS_PER_XMMA_N;
        const int kj = oi % ( PACKETS_PER_XMMA_M * PACKETS_PER_XMMA_N ) % PACKETS_PER_XMMA_N;

        // The position of the XMMA in the output tile.
        const int m = mi * PACKETS_PER_XMMA_M + mj;
        const int k = ki * Cta_tile::WARPS_N * PACKETS_PER_XMMA_N + kj;

        // Load the data.
        const int offset = nhw_[m] + ( k_ + k * ELEMENTS_PER_PACKET ) * params_hw_;
        if( mask ) {
            data.ldg( &params_res_ptr_[Traits::offset_in_bytes_c( offset )] );
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
    const int offset = nhw_[m] + ( k_ + k * ELEMENTS_PER_PACKET ) * params_hw_;
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
    inline __device__ void store( int mi, int oi, const Fragment_c& data, int mask ) {
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
        //     nhw_[m] + (k_ + k*ELEMENTS_PER_PACKET)*params_hw_,
        //     data.regs[0]);
        // // END OF DEBUG.

        // Store the data.
        const int offset = nhw_[m] + ( k_ + k * ELEMENTS_PER_PACKET ) * params_hw_;
        if( mask ) {
            data.stg( &params_out_ptr_[Traits::offset_in_bytes_c( offset )] );
        }
    }

    // The number of output channels.
    const int params_hw_, params_c_;
    // The pointer to global memory.
    char* const params_out_ptr_;
    const char* params_res_ptr_;
    // The offsets for the thread to output its values.
    int nhw_[Xmma_tile::XMMAS_M * PACKETS_PER_XMMA_M], k_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

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
    static_assert( BYTES_PER_STG == 8 || BYTES_PER_STG == 16, "" );

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
        : params_c_( params.c ), params_stride_k_( params.img_stride_c ),
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
        int ndhw = ( bidm * Cta_tile::M ) +
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
                int idx = ndhw + ii * Xmma_tile::M_PER_XMMA_PER_CTA + 8 * jj;
                if( FAST_EPILOGUE ) {
                    // Decompose the position into N and OPQ.
                    int n, dhw;
                    xmma::fast_divmod( n, dhw, idx, params.dhw, params.mul_dhw, params.shr_dhw );
                    // Compute the offset in to add to the pointer.
                    int offset = -1;
                    if( idx < params.ndhw ) {
                        offset = n * params.img_stride_n + dhw * BYTES_PER_PACKET + packet_k;
                    }
                    ndhw_[ii * PACKETS_PER_XMMA_M + jj] = offset;
                } else {
                    // Decompose the position into N and OPQ.
                    int n, dhw;
                    xmma::fast_divmod( n, dhw, idx, params.dhw, params.mul_dhw, params.shr_dhw );
                    // Decompose the position into O and PQ.
                    int d, hw;
                    xmma::fast_divmod( d, hw, dhw, params.hw, params.mul_hw, params.shr_hw );
                    // Decompose the position into P and Q.
                    int h, w;
                    xmma::fast_divmod( h, w, hw, params.w, params.mul_w, params.shr_w );

                    // Compute the offset in to add to the pointer.
                    int offset = -1;
                    if( idx < params.ndhw ) {
                        offset = n * params.img_stride_n + d * params.img_stride_d +
                                 h * params.img_stride_h + w * params.img_stride_w + packet_k;
                    }
                    ndhw_[ii * PACKETS_PER_XMMA_M + jj] = offset;
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
        return ndhw_[m] >= 0 && ( ( k_ + k ) * ELEMENTS_PER_PACKET ) < params_c_;
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
        const int offset = ndhw_[m] + ( k_ + k ) * params_stride_k_;
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
      const int offset = ndhw_[m] + ( k_ + k ) * params_stride_k_;
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
        const int offset = ndhw_[m] + ( k_ + k ) * params_stride_k_;

        if( mask ) {
            data.stg( &params_out_ptr_[Traits::offset_in_bytes_c( offset )], mem_desc );
        }
    }

    // The number and stride of output channels.
    const int params_stride_k_, params_c_;
    // The pointer to global memory.
    char* const params_out_ptr_;
    const char* params_res_ptr_;
    // The offsets for the thread to output its values.
    int ndhw_[Xmma_tile::XMMAS_M * PACKETS_PER_XMMA_M], k_;
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

}  // namespace interleaved_dgrad
}  // namespace implicit_gemm
}  // namespace xmma
