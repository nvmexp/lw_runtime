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

#include <xmma/gemm/gmem_tile.h>

namespace xmma {
namespace implicit_gemm {
namespace wgrad_indexed {

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// T I L E   A
//
///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits, typename Cta_tile, int BYTES_PER_LDG_,
    typename Layout, int M, int N>
struct Gmem_tile_base_a : public gemm::Gmem_tile_base<Traits, Cta_tile, M, N,
                                                      Traits::BITS_PER_ELEMENT_A, BYTES_PER_LDG_> {
    using Gmem_layout = Layout;

    // The base class.
    using Base_ = gemm::Gmem_tile_base<Traits, Cta_tile, M, N,
                                       Traits::BITS_PER_ELEMENT_A, BYTES_PER_LDG_>;

    // The number of threads needed to load a pixel.
    enum { THREADS_PER_PIXEL = Base_::THREADS_PER_COLUMN };
    // The number of images loaded per LDG.
    enum { PIXELS_PER_LDG = Base_::COLUMNS_PER_LDG };

    // Ctor.
    template <typename Params>
    inline __device__ Gmem_tile_base_a( const Params& params,
                                        void*,
                                        int bidz )
        : Base_( params, nullptr, params.nopq, params.flt_gmem, bidz ),
          params_delta_k_( PIXELS_PER_LDG * params.g * params.k ),
          params_delta_(params.split_k.slices * Cta_tile::K),
          params_opqk_(params.opq * params.g * params.k),
          params_opq_(params.opq),
          params_mul_opq_(params.mul_opq),
          params_shr_opq_(params.shr_opq) {
              precompute_residue_predicates();
    }

    // Ctor.
    template <typename Params>
    inline __device__ Gmem_tile_base_a( const Params& params,
                                        void*,
                                        const dim3& bidx,
                                        int tidx )
        : Gmem_tile_base_a( params, nullptr, bidx.z ) {

        // The position in the K dimension (GEMM-M).
        const int k_per_tile_m = ( Cta_tile::M > Cta_tile::N && params.g > 1 ) 
                           ? Cta_tile::N : Cta_tile::M;
        int k_base = bidx.x * k_per_tile_m;
        // The position in the NPQ dimension.
        int nopq_base = bidx.z * Cta_tile::K;

        int k_in_tile;

        if( Gmem_layout::ROW ) {
            nopq_base += tidx % THREADS_PER_PIXEL * Base_::ELTS_PER_LDG;
            k_in_tile = tidx / THREADS_PER_PIXEL;

            #pragma unroll
            for( int ii = 0; ii < Base_::LDGS_UNROLL; ++ii ) {
                nopq_[ii] = nopq_base + ii;
                // Decompose nopq into n and opq.
                int n, opq;
                xmma::fast_divmod( n, opq, nopq_[ii], params.opq,
                    params.mul_opq, params.shr_opq );

                nopq_offsets_[ii] = n * params_opqk_ + opq;
            }

        } else {
            nopq_base += tidx / THREADS_PER_PIXEL;
            k_in_tile = tidx % THREADS_PER_PIXEL * Base_::ELTS_PER_LDG;
        }

        // Initialize the predicates.
        uint32_t preds[Base_::LDGS];
#pragma unroll
        for( int ii = 0; ii < Base_::LDGS; ++ii ) {

            int k, nopq, k_offset;
            if( Gmem_layout::ROW ) {
                nopq = nopq_[ii%Base_::LDGS_UNROLL];
                k_offset = k_in_tile + (ii / Base_::LDGS_UNROLL) * PIXELS_PER_LDG;
                k = k_base + k_offset;

                this->offsets_[ii] = k * params.opq;

            } else {
                nopq = nopq_base + (ii / Base_::LDGS_UNROLL) * PIXELS_PER_LDG;
                k_offset = k_in_tile + ii % Base_::LDGS_UNROLL;
                k = k_base + k_offset;

                this->offsets_[ii] = nopq * params.g * params.k + k;
            }


            preds[ii] = nopq < params.nopq &&  k < (params.g * params.k);

            if ( Cta_tile::M > Cta_tile::N && params.g > 1 ) {
                preds[ii] &= ( k_offset < Cta_tile::N );
        }
        }

        xmma::pack_predicates( this->preds_, preds );
    }

    // Compute the global memory pointers.
    inline __device__ void compute_load_pointers( const void* ( &ptrs )[Base_::LDGS] ) const {
        if( Gmem_layout::ROW ) {
            #pragma unroll
            for( int ii = 0; ii < Base_::LDGS; ++ii ) {
                ptrs[ii] = &this->ptr_[Traits::offset_in_bytes_a(
                    this->offsets_[ii] + nopq_offsets_[ii%Base_::LDGS_UNROLL])];
            }
        } else {
            #pragma unroll
            for( int ii = 0; ii < Base_::LDGS; ++ii ) {
                ptrs[ii] = &this->ptr_[Traits::offset_in_bytes_a( this->offsets_[ii] )];
            }
        }
    }

    // Compute the global memory pointers.
    template< int PHASE >
    inline __device__ void compute_load_pointers_per_phase( const void* ( &ptrs )[Base_::LDGS_PER_PHASE] ) const {
        if( Gmem_layout::ROW ) {
            #pragma unroll
            for( int ii = PHASE * Base_::LDGS_PER_PHASE; ii < MIN((PHASE + 1) * Base_::LDGS_PER_PHASE, Base_::LDGS); ++ii ) {
                ptrs[ii - PHASE * Base_::LDGS_PER_PHASE] = &this->ptr_[Traits::offset_in_bytes_a(
                    this->offsets_[ii] + nopq_offsets_[ii%Base_::LDGS_UNROLL])];
            }
        } else {
            #pragma unroll
            for( int ii = PHASE * Base_::LDGS_PER_PHASE; ii < MIN((PHASE + 1) * Base_::LDGS_PER_PHASE, Base_::LDGS); ++ii ) {
                ptrs[ii - PHASE * Base_::LDGS_PER_PHASE] = &this->ptr_[Traits::offset_in_bytes_a( this->offsets_[ii] )];
            }
        }
    }
    // Precompute predicates for residue.
    inline __device__ void precompute_residue_predicates() {
        if( Gmem_layout::ROW ) {
        } else {
            Base_::precompute_residue_predicates_a_n_b_t();
        }
    }


    // The residue to "fix" the predicates.
    inline __device__ void residue( int masks_to_clear = Base_::LDGS ) {
        if( Gmem_layout::ROW ) {
        } else {
            Base_::residue_a_n_b_t();
        }
    }

    inline __device__ void move( int next_trsi, int64_t delta ) {
        if( Gmem_layout::ROW ) {

            uint32_t preds[Base_::LDGS];
            #pragma unroll
            for (int ii = 0; ii < Base_::LDGS_UNROLL; ii++) {
                nopq_[ii] += params_delta_;
                // Decompose nopq into n and opq.
                int n, opq;
                xmma::fast_divmod( n, opq, nopq_[ii], params_opq_,
                    params_mul_opq_, params_shr_opq_ );

                nopq_offsets_[ii] = n * params_opqk_ + opq;
            }
            #pragma unroll
            for( int ii = 0; ii < Base_::LDGS; ++ii ) {
                preds[ii] = nopq_[ii%Base_::LDGS_UNROLL] < this->params_k_;
            }
            uint32_t tmp[Base_::PRED_REGS];
            xmma::pack_predicates( tmp, preds );
            #pragma unroll
            for( int ii = 0; ii < Base_::PRED_REGS; ++ii ) {
                this->preds_[ii] &= tmp[ii];
            }
        } else {
            Base_::move(next_trsi, delta);
        }
    }

    // The offset to move the pointers.
    const int params_delta_k_;
    const int params_delta_, params_opqk_;
    const int params_opq_,  params_mul_opq_, params_shr_opq_;
    uint32_t nopq_[Base_::LDGS_UNROLL];
    int nopq_offsets_[Base_::LDGS_UNROLL];
};

///////////////////////////////////////////////////////////////////////////////////////////////////

template <
    // The instruction traits.
    typename Traits,
    // The CTA tile description.
    typename Cta_tile,
    // The size in bytes of the LDGs.
    int BYTES_PER_LDG_ = 16,
    // Do we disable LDGSTS even on an architecture that has it?
    bool DISABLE_LDGSTS = false,
    // The ancestor/base class. See docs/gmem_tile.md.
    typename Ancestor = Gmem_tile_base_a<Traits, Cta_tile, BYTES_PER_LDG_,
        xmma::Col,  Cta_tile::M, Cta_tile::K>>
struct Gmem_tile_a_n : public Ldgsts_selector<Traits, gemm::Rebase_gmem_tile_with_ldgsts<Ancestor>,
                                            gemm::Rebase_gmem_tile_with_ldg_and_sts<Ancestor>,
                                            DISABLE_LDGSTS>::Class {

    // The base class.
    using Base_ = typename Ldgsts_selector<Traits, gemm::Rebase_gmem_tile_with_ldgsts<Ancestor>,
                                           gemm::Rebase_gmem_tile_with_ldg_and_sts<Ancestor>,
                                           DISABLE_LDGSTS>::Class;
    // The expected shared memory layout.
    using Smem_layout = xmma::Col;

    // Ctor.
    template <typename Params>
    inline __device__ Gmem_tile_a_n( const Params& params,
                                     void* smem,
                                     const dim3& bidx,
                                     int tidx )
        : Base_( params, smem, bidx, tidx ) {
    }
};

template <
    // The instruction traits.
    typename Traits,
    // The CTA tile description.
    typename Cta_tile,
    // The size in bytes of the LDGs.
    int BYTES_PER_LDG_ = 16,
    // Do we disable LDGSTS even on an architecture that has it?
    bool DISABLE_LDGSTS = false,
    // The ancestor/base class. See docs/gmem_tile.md.
    typename Ancestor = Gmem_tile_base_a<Traits, Cta_tile, BYTES_PER_LDG_,
        xmma::Row, Cta_tile::K, Cta_tile::M>>
struct Gmem_tile_a_t : public Ldgsts_selector<Traits, gemm::Rebase_gmem_tile_with_ldgsts<Ancestor>,
                                            gemm::Rebase_gmem_tile_with_ldg_and_sts<Ancestor>,
                                            DISABLE_LDGSTS>::Class {

    // The base class.
    using Base_ = typename Ldgsts_selector<Traits, gemm::Rebase_gmem_tile_with_ldgsts<Ancestor>,
                                           gemm::Rebase_gmem_tile_with_ldg_and_sts<Ancestor>,
                                           DISABLE_LDGSTS>::Class;
    // The expected shared memory layout.
    using Smem_layout = xmma::Row;

    // Ctor.
    template <typename Params>
    inline __device__ Gmem_tile_a_t( const Params& params,
                                     void* smem,
                                     const dim3& bidx,
                                     int tidx )
        : Base_( params, smem, bidx, tidx ) {
    }
};

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// T I L E   B
//
////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits, typename Cta_tile, bool SIMPLE_1x1x1, int BYTES_PER_LDG,
    typename Layout, int M, int N>
struct Gmem_tile_base_b {};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits, typename Cta_tile, int BYTES_PER_LDG_,
    typename Layout, int M, int N>
struct Gmem_tile_base_b<Traits, Cta_tile, false, BYTES_PER_LDG_,
    Layout, M, N>
    : public gemm::Gmem_tile_base<Traits, Cta_tile, M, N,
                                  Traits::BITS_PER_ELEMENT_B, BYTES_PER_LDG_> {
    using Gmem_layout = Layout;

    // The base class.
    using Base_ = gemm::Gmem_tile_base<Traits, Cta_tile, M, N,
                                       Traits::BITS_PER_ELEMENT_B, BYTES_PER_LDG_>;

    // The number of threads needed to load a pixel.
    enum { THREADS_PER_PIXEL = Base_::THREADS_PER_COLUMN };
    // The number of images loaded per LDG.
    enum { PIXELS_PER_LDG = Base_::COLUMNS_PER_LDG };
    // The ratio of tile_N to tile_M
    enum { TILE_N_DIV_M = Cta_tile::N / Cta_tile::M };

    // Ctor.
    template <typename Params>
    inline __device__ Gmem_tile_base_b( const Params& params,
                                        void*,
                                        const dim3& bidx,
                                        int tidx )
        : Base_( params, nullptr, params.nopq, params.img_gmem, bidx.z ), params_n_( params.n ),
          params_d_( params.d ), params_h_( params.h ), params_w_( params.w ),
          params_c_( params.g * params.c ), params_dhwc_( params.dhwc ), params_hwc_( params.hwc ),
          params_hw_( params.hw),
          params_wc_( params.wc ), params_opq_( params.opq ), params_mul_opq_( params.mul_opq ),
          params_shr_opq_( params.shr_opq ), params_pq_( params.pq ),
          params_mul_pq_( params.mul_pq ), params_shr_pq_( params.shr_pq ), params_q_( params.q ),
          params_mul_q_( params.mul_q ), params_shr_q_( params.shr_q ),
          params_stride_d_( params.stride[0] ), params_stride_h_( params.stride[1] ),
          params_stride_w_( params.stride[2] ), params_dilation_d_( params.dilation[0] ),
          params_dilation_h_( params.dilation[1] ), params_dilation_w_( params.dilation[2] ),
          params_pad_d_( params.pad[0][0] ), params_pad_h_( params.pad[1][0] ),
          params_pad_w_( params.pad[2][0] ), tiles_k_( params.tiles_k ),
          params_dhw_( params.dhw ) {

        // Extract TxRxS and C.
        int trs;
        int layout_tid_offset;
	if( Gmem_layout::ROW ) {
            layout_tid_offset = (tidx % THREADS_PER_PIXEL)*Base_::ELTS_PER_LDG;
        } else {
	    layout_tid_offset = tidx / THREADS_PER_PIXEL;	
        }
	int gemm_n = bidx.y*Cta_tile::N + layout_tid_offset;
        xmma::fast_divmod( trs, c_, gemm_n, params.c_per_ctas, params.mul_c_per_ctas,
	params.shr_c_per_ctas );

        // The T, R and S coordinates.
        int rs;
        xmma::fast_divmod( t_, rs, trs, params.rs, params.mul_rs, params.shr_rs );
        xmma::fast_divmod( r_, s_, rs, params.s, params.mul_s, params.shr_s );

        // Deal with cross-correlation and colwolution.
        if( !params.cross_correlation ) {
            t_ = params.t - 1 - t_;
            r_ = params.r - 1 - r_;
            s_ = params.s - 1 - s_;
        }

        // Special case for the group colwolution.
        if( params.g > 1 ) {
            auto ratio = TILE_N_DIV_M <= 1 ? 1: TILE_N_DIV_M;
            c_ += (bidx.x / ratio) * Cta_tile::N;
        }

        // The starting point in the gemm-K dimension.
        nopq_[0] = bidx.z * Cta_tile::K;

        if( Gmem_layout::ROW ) {
            nopq_[0] += tidx / THREADS_PER_PIXEL;
        } else {
            nopq_[0] += tidx % THREADS_PER_PIXEL * Base_::ELTS_PER_LDG;
        }

        // Compute the offsets and predicates.
        compute_offsets_and_predicates();
    }

    // Compute the predicates.
    inline __device__ void compute_offsets_and_predicates() {

        // For each load, extract N, P and Q.
        uint32_t preds[Base_::LDGS];
        uint32_t preds_nopq[Base_::LDGS_UNROLL];

        if( Gmem_layout::COL ) {
            #pragma unroll
            for( int ii = 0; ii < Base_::LDGS_UNROLL; ++ii ) {
                nopq_[ii] = nopq_[0] + ii;
                // Decompose nopq into n and opq.
                int n, opq;
                xmma::fast_divmod( n, opq, nopq_[ii], params_opq_,
                    params_mul_opq_, params_shr_opq_ );
                int o, pq;
                xmma::fast_divmod( o, pq, opq, params_pq_, params_mul_pq_, params_shr_pq_ );
                int p, q;
                xmma::fast_divmod( p, q, pq, params_q_, params_mul_q_, params_shr_q_ );

                // Compute the H and W coordinates.
                int d = o * params_stride_d_ - params_pad_d_ + t_ * params_dilation_d_;
                int h = p * params_stride_h_ - params_pad_h_ + r_ * params_dilation_h_;
                int w = q * params_stride_w_ - params_pad_w_ + s_ * params_dilation_w_;

                // The offset to add to the pointer.
                this->nopq_offsets_[ii] =
                    n * params_dhwc_ + d * params_hw_ + h * params_w_ + w;

                // Update the predicates.
                preds_nopq[ii] = n < params_n_ && (unsigned)d < params_d_ &&
                    (unsigned)h < params_h_ && (unsigned)w < params_w_
                    && c_ < params_c_;
            }
        }

        #pragma unroll
        for( int ii = 0; ii < Base_::LDGS; ++ii ) {
            if( Gmem_layout::ROW ) {
                // The NPQ position.
                int nopq = nopq_[0] + ii * PIXELS_PER_LDG;

                // Extract the N, O, P and Q coordinates.
                int n, opq;
                xmma::fast_divmod( n, opq, nopq, params_opq_, params_mul_opq_, params_shr_opq_ );
                int o, pq;
                xmma::fast_divmod( o, pq, opq, params_pq_, params_mul_pq_, params_shr_pq_ );
                int p, q;
                xmma::fast_divmod( p, q, pq, params_q_, params_mul_q_, params_shr_q_ );

                // Compute the H and W coordinates.
                int d = o * params_stride_d_ - params_pad_d_ + t_ * params_dilation_d_;
                int h = p * params_stride_h_ - params_pad_h_ + r_ * params_dilation_h_;
                int w = q * params_stride_w_ - params_pad_w_ + s_ * params_dilation_w_;

                // The offset to add to the pointer.
                this->offsets_[ii] =
                n * params_dhwc_ + d * params_hwc_ + h * params_wc_ + w * params_c_ + c_;

                // Update the predicates.
                preds[ii] = n < params_n_ && (unsigned)d < params_d_ && (unsigned)h < params_h_ &&
                        (unsigned)w < params_w_ && c_ < params_c_;
            } else {
                int c = c_ + (ii / Base_::LDGS_UNROLL) * PIXELS_PER_LDG;

                this->offsets_[ii] = c * params_dhw_ + nopq_offsets_[ii%Base_::LDGS_UNROLL];

                preds[ii] = preds_nopq[ii%Base_::LDGS_UNROLL] && c < params_c_;
            }
        }

        // Pack the predicates.
        xmma::pack_predicates( this->preds_, preds );
    }

    // Move the pointers.
    inline __device__ void move( int, int64_t ) {
        if( Gmem_layout::ROW ) {
            nopq_[0] += tiles_k_ * Cta_tile::K;
            compute_offsets_and_predicates();
        } else {
            #pragma unroll
            for (int ii = 0; ii < Base_::LDGS_UNROLL; ii++) {
                nopq_[ii] += tiles_k_ * Cta_tile::K;
                compute_offsets_and_predicates();
            }
        }
    }

    // Update the predicates.
    inline __device__ void residue() {
    }

    // The two key dimensions of the tensor.
    const int params_n_, params_d_, params_h_, params_w_, params_c_;
    // The precomputed values.
    const int params_dhwc_, params_hwc_, params_hw_, params_wc_;
    // The parameters to decompose NOPQ.
    const int params_opq_, params_mul_opq_, params_shr_opq_;
    // The parameters to decompose OPQ.
    const int params_pq_, params_mul_pq_, params_shr_pq_;
    // The parameters to decompose PQ.
    const int params_q_, params_mul_q_, params_shr_q_;
    // The stride.
    const int params_stride_d_, params_stride_h_, params_stride_w_;
    // The dilation.
    const int params_dilation_d_, params_dilation_h_, params_dilation_w_;
    // The padding.
    const int params_pad_d_, params_pad_h_, params_pad_w_;
    // The position.
    int t_, r_, s_, c_, nopq_[Base_::LDGS_UNROLL];
    const int params_dhw_;
    // The number of tiles in the k dimension.
    int tiles_k_;
    int nopq_offsets_[Base_::LDGS_UNROLL];
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits, typename Cta_tile, int BYTES_PER_LDG_,
    typename Layout, int M, int N>
struct Gmem_tile_base_b<Traits, Cta_tile, true, BYTES_PER_LDG_,
    Layout, M, N>
    : public gemm::Gmem_tile_base<Traits, Cta_tile, M, N,
                                  Traits::BITS_PER_ELEMENT_B, BYTES_PER_LDG_> {
    using Gmem_layout = Layout;

    // The base class.
    using Base_ = gemm::Gmem_tile_base<Traits, Cta_tile, M, N,
                                       Traits::BITS_PER_ELEMENT_B, BYTES_PER_LDG_>;

    // The number of threads needed to load a pixel.
    enum { THREADS_PER_PIXEL = Base_::THREADS_PER_COLUMN };
    // The number of images loaded per LDG.
    enum { PIXELS_PER_LDG = Base_::COLUMNS_PER_LDG };
    // The ratio of tile_N to tile_M
    enum { TILE_N_DIV_M = Cta_tile::N / Cta_tile::M };

    // Ctor.
    template <typename Params>
    inline __device__ Gmem_tile_base_b( const Params& params,
                                        void* smem,
                                        const dim3& bidx,
                                        int tidx )
        : Base_( params, smem, params.nopq, params.img_gmem, bidx.z ), params_nopq_( params.nopq ),
          params_delta_nopq_( PIXELS_PER_LDG * params.g * params.c ),
          params_residue_nopq_( params.loop_residue_k ),
          params_delta_(params.split_k.slices * Cta_tile::K),
          params_opqk_(params.opq * params.g * params.c),
          params_opq_(params.opq),
          params_mul_opq_(params.mul_opq),
          params_shr_opq_(params.shr_opq) {

        // Deal with the special case for groups.
        int c_base = bidx.y * Cta_tile::N;
        if( params.g > 1 ) {
            auto ratio = TILE_N_DIV_M <= 1 ? 1: TILE_N_DIV_M;
            c_base += (bidx.x / ratio) * Cta_tile::N;
        }

        // The coordinate in the NPQ dimension.
        int nopq_base = bidx.z * Cta_tile::K;

        if( Gmem_layout::ROW ) {
            nopq_base += tidx / THREADS_PER_PIXEL;
            c_base += tidx % THREADS_PER_PIXEL * Base_::ELTS_PER_LDG;
        } else {
            nopq_base += tidx % THREADS_PER_PIXEL * Base_::ELTS_PER_LDG;
            c_base += tidx / THREADS_PER_PIXEL;

            #pragma unroll
            for( int ii = 0; ii < Base_::LDGS_UNROLL; ++ii ) {
                nopq_[ii] = nopq_base + ii;
                // Decompose nopq into n and opq.
                int n, opq;
                xmma::fast_divmod( n, opq, nopq_[ii], params.opq,
                    params.mul_opq, params.shr_opq );

                nopq_offsets_[ii] = n * params_opqk_ + opq;
            }
        }

        c_ = c_base;

        // For each LDG, make sure the load is valid.
        uint32_t preds[Base_::LDGS];
        #pragma unroll
        for( int ii = 0; ii < Base_::LDGS; ++ii ) {

            int c, nopq;
            if( Gmem_layout::ROW ) {
                nopq = nopq_base + (ii / Base_::LDGS_UNROLL) * PIXELS_PER_LDG;
                c = c_base + ii % Base_::LDGS_UNROLL;

                this->offsets_[ii] = nopq * params.g * params.c + c;
            } else {
                nopq = nopq_[ii%Base_::LDGS_UNROLL];
                c = c_base + (ii / Base_::LDGS_UNROLL) * PIXELS_PER_LDG;

                this->offsets_[ii] = c * params.opq;
            }

            preds[ii] = nopq < params_nopq_ && c < (params.g * params.c);
        }

        xmma::pack_predicates( this->preds_, preds );

        precompute_residue_predicates();
    }

    // Compute the global memory pointers.
    inline __device__ void compute_load_pointers( const void* ( &ptrs )[Base_::LDGS] ) const {
        if( Gmem_layout::ROW ) {
            #pragma unroll
            for( int ii = 0; ii < Base_::LDGS; ++ii ) {
                ptrs[ii] = &this->ptr_[Traits::offset_in_bytes_b( this->offsets_[ii] )];
            }
        } else {
            #pragma unroll
            for( int ii = 0; ii < Base_::LDGS; ++ii ) {
                ptrs[ii] = &this->ptr_[Traits::offset_in_bytes_b(
                    this->offsets_[ii] + nopq_offsets_[ii%Base_::LDGS_UNROLL])];
            }
        }
    }

    // Compute the global memory pointers.
    template< int PHASE >
    inline __device__ void compute_load_pointers_per_phase( const void* ( &ptrs )[Base_::LDGS_PER_PHASE] ) const {
        if( Gmem_layout::ROW ) {
            #pragma unroll
            for( int ii = PHASE * Base_::LDGS_PER_PHASE; ii < MIN((PHASE + 1) * Base_::LDGS_PER_PHASE, Base_::LDGS); ++ii ) {
                ptrs[ii - PHASE * Base_::LDGS_PER_PHASE] = &this->ptr_[Traits::offset_in_bytes_b( this->offsets_[ii] )];
            }
        } else {
            #pragma unroll
            for( int ii = PHASE * Base_::LDGS_PER_PHASE; ii < MIN((PHASE + 1) * Base_::LDGS_PER_PHASE, Base_::LDGS); ++ii ) {
                ptrs[ii - PHASE * Base_::LDGS_PER_PHASE] = &this->ptr_[Traits::offset_in_bytes_b(
                    this->offsets_[ii] + nopq_offsets_[ii%Base_::LDGS_UNROLL])];
            }
        }
    }

    // Precompute predicates for residue.
    inline __device__ void precompute_residue_predicates() {
        if( Gmem_layout::ROW ) {
            Base_::precompute_residue_predicates_a_n_b_t();
        } else {
        }
    }

    // Update the predicates.
    inline __device__ void residue() {
        if( Gmem_layout::ROW ) {
            Base_::residue_a_n_b_t();
        } else {
        }
    }

    inline __device__ void move( int next_trsi, int64_t delta ) {
        if( Gmem_layout::ROW ) {
            Base_::move(next_trsi, delta);
        } else {
            #pragma unroll
            for (int ii = 0; ii < Base_::LDGS_UNROLL; ii++) {
                nopq_[ii] += params_delta_;
                // Decompose nopq into n and opq.
                int n, opq;
                xmma::fast_divmod( n, opq, nopq_[ii], params_opq_,
                    params_mul_opq_, params_shr_opq_ );

                nopq_offsets_[ii] = n * params_opqk_ + opq;
            }
            uint32_t preds[Base_::LDGS];
            #pragma unroll
            for( int ii = 0; ii < Base_::LDGS; ++ii ) {
                preds[ii] = nopq_[ii%Base_::LDGS_UNROLL] < params_nopq_;
            }
            uint32_t tmp[Base_::PRED_REGS];
            xmma::pack_predicates( tmp, preds );
            #pragma unroll
            for( int ii = 0; ii < Base_::PRED_REGS; ++ii ) {
                this->preds_[ii] &= tmp[ii];
            }
        }
    }


    // The two key dimensions of the tensor.
    const int params_nopq_, params_delta_nopq_, params_residue_nopq_;
    // The position in C.
    int c_;
    const int params_delta_, params_opqk_;
    const int params_opq_,  params_mul_opq_, params_shr_opq_;
    uint32_t nopq_[Base_::LDGS_UNROLL];
    int nopq_offsets_[Base_::LDGS_UNROLL];
};

///////////////////////////////////////////////////////////////////////////////////////////////////

template <
    // The instruction traits.
    typename Traits,
    // The CTA tile description.
    typename Cta_tile,
    // Do we use a 1x1x1 filter?
    bool SIMPLE_1x1x1,
    // The size in bytes of the LDGs.
    int BYTES_PER_LDG_ = 16,
    // Do we disable LDGSTS even on an architecture that has it?
    bool DISABLE_LDGSTS = false,
    // The ancestor/base class. See docs/gmem_tile.md.
    typename Ancestor = Gmem_tile_base_b<Traits, Cta_tile, SIMPLE_1x1x1, BYTES_PER_LDG_,
        xmma::Row, Cta_tile::N, Cta_tile::K>>
struct Gmem_tile_b_t : public Ldgsts_selector<Traits, gemm::Rebase_gmem_tile_with_ldgsts<Ancestor>,
                                            gemm::Rebase_gmem_tile_with_ldg_and_sts<Ancestor>,
                                            DISABLE_LDGSTS>::Class {

    // The base class.
    using Base_ = typename Ldgsts_selector<Traits, gemm::Rebase_gmem_tile_with_ldgsts<Ancestor>,
                                           gemm::Rebase_gmem_tile_with_ldg_and_sts<Ancestor>,
                                           DISABLE_LDGSTS>::Class;
    // The expected shared memory layout.
    using Smem_layout = xmma::Row;

    // Ctor.
    template <typename Params>
    inline __device__ Gmem_tile_b_t( const Params& params,
                                     void* smem,
                                     const dim3& bidx,
                                     int tidx )
        : Base_( params, smem, bidx, tidx ) {
    }
};

template <
    // The instruction traits.
    typename Traits,
    // The CTA tile description.
    typename Cta_tile,
    // Do we use a 1x1x1 filter?
    bool SIMPLE_1x1x1,
    // The size in bytes of the LDGs.
    int BYTES_PER_LDG_ = 16,
    // Do we disable LDGSTS even on an architecture that has it?
    bool DISABLE_LDGSTS = false,
    // The ancestor/base class. See docs/gmem_tile.md.
    typename Ancestor = Gmem_tile_base_b<Traits, Cta_tile, SIMPLE_1x1x1, BYTES_PER_LDG_,
    xmma::Col, Cta_tile::K, Cta_tile::N>>
struct Gmem_tile_b_n : public Ldgsts_selector<Traits, gemm::Rebase_gmem_tile_with_ldgsts<Ancestor>,
                                            gemm::Rebase_gmem_tile_with_ldg_and_sts<Ancestor>,
                                            DISABLE_LDGSTS>::Class {

    // The base class.
    using Base_ = typename Ldgsts_selector<Traits, gemm::Rebase_gmem_tile_with_ldgsts<Ancestor>,
                                           gemm::Rebase_gmem_tile_with_ldg_and_sts<Ancestor>,
                                           DISABLE_LDGSTS>::Class;
    // The expected shared memory layout.
    using Smem_layout = xmma::Col;

    // Ctor.
    template <typename Params>
    inline __device__ Gmem_tile_b_n( const Params& params,
                                     void* smem,
                                     const dim3& bidx,
                                     int tidx )
        : Base_( params, smem, bidx, tidx ) {
    }
};

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// T I L E   C
//
////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    typename Traits,
    typename Cta_tile,
    typename Fragment_c = xmma::Fragment_c<Traits, Cta_tile>>
struct Gmem_tile_epilogue_base
    : public xmma::helpers::Gmem_tile_epilogue<Traits, Cta_tile, xmma::Row, Fragment_c> {

    using Layout = xmma::Row;
    // The base class.
    using Base = xmma::helpers::Gmem_tile_epilogue<Traits, Cta_tile, xmma::Row, Fragment_c>;
    // The helper class to compute the row offset (in the tile).
    using Tile_distribution = xmma::helpers::Gmem_tile_epilogue_distribution<Traits, Cta_tile>;

    // Ctor.
    inline __device__ Gmem_tile_epilogue_base( int m,
                                               int n,
                                               int g,
                                               int stride_n )
        : Base( m, n, stride_n )
        , params_g_( g ) {
    }

    // Ctor.
    inline __device__ Gmem_tile_epilogue_base( int m,
                                               int n,
                                               int g,
                                               int stride_n,
                                               char* out_ptr,
                                               const char* res_ptr,
                                               int bidm,
                                               int bidn,
                                               int bidz,
                                               int tidx )
        : Base( m, n, stride_n, out_ptr, res_ptr, bidm, bidn, bidz, tidx )
        , params_g_( g ) {
    }

    // Load the data from global memory.
    inline __device__ void load( Fragment_c& data, int mi, int ii, int mask,
                                 const uint64_t mem_desc ) {
        const int offset = Tile_distribution::compute_offset( mi, ii );
        const char* ptr = &this->res_ptr_[Traits::offset_in_bytes_c( offset * this->params_stride_n_ )];

        if ( Cta_tile::M > Cta_tile::N && params_g_ > 1 ) {
            mask &= ( offset < Cta_tile::N );
        }

        if( mask ) {
            if( Cta_tile::N / Cta_tile::GROUPS * sizeof(typename Traits::C_type) >= 16 ) {
                uint4 tmp;
                xmma::ldg( tmp, ptr, mem_desc );
                data.from_int4( tmp );
            } else if( Cta_tile::N / Cta_tile::GROUPS * sizeof(typename Traits::C_type) == 8 ) {
                // For GROUPS = 16 and output type is fp16, we use LDG.64
                uint2 tmp;
                xmma::ldg( tmp, ptr, mem_desc );
                data.from_int2( tmp );
            }
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
        const int offset = Tile_distribution::compute_offset( mi, ii );
        const char* ptr = &this->res_ptr_[Traits::offset_in_bytes_c( offset * this->params_stride_n_ )];

        if ( Cta_tile::M > Cta_tile::N && params_g_ > 1 ) {
            mask &= ( offset < Cta_tile::N );
        }

        if( mask ) {
          int xmma_tile_idx = (mi * Base::STGS + ii) % (xmma_tiles_per_a + xmma_tiles_per_b);
          uint32_t smem_ptr;

          if (xmma_tile_idx < xmma_tiles_per_a) 
            smem_ptr = smem_tile_a.smem_ + smem_tile_a.smem_write_buffer_ + 
            xmma_tile_idx *  Cta_tile::THREADS_PER_CTA * Base::BYTES_PER_STG + tidx * Base::BYTES_PER_STG;
          else
            smem_ptr = smem_tile_b.smem_ + smem_tile_b.smem_write_buffer_ + 
            (xmma_tile_idx - xmma_tiles_per_a) *  Cta_tile::THREADS_PER_CTA * Base::BYTES_PER_STG + 
            tidx * Base::BYTES_PER_STG;
          
            if( Cta_tile::N / Cta_tile::GROUPS * sizeof(typename Traits::C_type) >= 16 ) {
              ldgsts128(smem_ptr, ptr, true, mem_desc);
            } else if( Cta_tile::N / Cta_tile::GROUPS * sizeof(typename Traits::C_type) == 8 ) {
              ldgsts64(smem_ptr, ptr, true, mem_desc);
            }
        }                             
    }

    // Store the data to global memory.
    inline __device__ void store( int mi, int ii, const Fragment_c& data, int mask,
                                  const uint64_t mem_desc ) {
        const int offset = Tile_distribution::compute_offset( mi, ii );
        char* ptr = &this->out_ptr_[Traits::offset_in_bytes_c( offset * this->params_stride_n_ )];

        if ( Cta_tile::M > Cta_tile::N && params_g_ > 1 ) {
            mask &= ( offset < Cta_tile::N );
        }

        if( mask ) {
            if( Cta_tile::N / Cta_tile::GROUPS * sizeof(typename Traits::C_type) >= 16 ) {
                xmma::stg( ptr, data.to_int4(), mem_desc );
            } else if( Cta_tile::N / Cta_tile::GROUPS * sizeof(typename Traits::C_type) == 8 ) {
                // For GROUPS = 16 and output type is fp16, we use STG.64
                xmma::stg( ptr, data.to_int2(), mem_desc );
            }
        }
    }

    const int params_g_;
};

///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits, typename Cta_tile, bool SIMPLE_1x1x1>
struct Gmem_tile_epilogue : public Gmem_tile_epilogue_base<Traits, Cta_tile> {};

///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits, typename Cta_tile>
struct Gmem_tile_epilogue<Traits, Cta_tile, false>
    : public Gmem_tile_epilogue_base<Traits, Cta_tile> {

    // The base class.
    using Base = Gmem_tile_epilogue_base<Traits, Cta_tile>;

    // Ctor.
    template <typename Params>
    inline __device__ Gmem_tile_epilogue( const Params& params,
                                          int bidm,
                                          int bidn,
                                          int bidz,
                                          int tidx )
        : Base( params.g * params.k, Cta_tile::GROUPS >1 ? params.c : params.trsc, params.g, params.trsc ) {

	int layout_tid_offset, bidn_offset;
	layout_tid_offset = (tidx % Base::THREADS_PER_ROW)*Base::ELEMENTS_PER_STG;
	bidn_offset = bidn * (Cta_tile::N / Cta_tile::GROUPS);
        // The n dimension.
	this->n_ = layout_tid_offset;
	if(Cta_tile::GROUPS == 1) {
		this->n_ += bidn_offset;
	}

        // The m dimension.
        const int k_per_tile_m = ( Cta_tile::M > Cta_tile::N && params.g > 1 ) 
                           ? Cta_tile::N : Cta_tile::M;
        this->m_ = bidm * k_per_tile_m + tidx / Base::THREADS_PER_ROW;

        // The pointers.
        char* out_ptr = reinterpret_cast<char*>( params.out_gmem );
        const char* res_ptr = reinterpret_cast<const char*>( params.res_gmem );
        int64_t offset = (Cta_tile::GROUPS > 1) 
			? Traits::offset_in_bytes_c( this->m_ * params.trsc + bidn_offset + this->n_)
			: Traits::offset_in_bytes_c( this->m_ * params.trsc + this->n_);
        this->out_ptr_ = &out_ptr[offset];
        this->res_ptr_ = &res_ptr[offset];
    }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits, typename Cta_tile>
struct Gmem_tile_epilogue<Traits, Cta_tile, true>
    : public Gmem_tile_epilogue_base<Traits, Cta_tile> {

    // The base class.
    using Base = Gmem_tile_epilogue_base<Traits, Cta_tile>;

    // Ctor.
    template <typename Params>
    inline __device__ Gmem_tile_epilogue( const Params& params,
                                          int bidm,
                                          int bidn,
                                          int bidz,
                                          int tidx )
        : Base( params.g * params.k, params.c, params.g, params.c,
                reinterpret_cast<char*>( params.out_gmem ),
                reinterpret_cast<const char*>( params.res_gmem ),
                bidm, bidn, bidz, tidx ) {
                    
        // Add the thread index to C.
        this->n_ = bidn * Cta_tile::N + tidx % Base::THREADS_PER_ROW * Base::ELEMENTS_PER_STG;

        // The k dimension.
        const int k_per_tile_m = ( Cta_tile::M > Cta_tile::N && params.g > 1 ) 
                           ? Cta_tile::N : Cta_tile::M;
        this->m_ = bidm * k_per_tile_m + tidx / Base::THREADS_PER_ROW;

        // The pointers.
        char* out_ptr = reinterpret_cast<char*>( params.out_gmem );
        const char* res_ptr = reinterpret_cast<const char*>( params.res_gmem );
        int64_t offset =
            Traits::offset_in_bytes_c( this->m_ * params.c + this->n_ );
        this->out_ptr_ = &out_ptr[offset];
        this->res_ptr_ = &res_ptr[offset];
    }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace wgrad_indexed
}  // namespace implicit_gemm
}  // namespace xmma
