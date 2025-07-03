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
#include <xmma/implicit_gemm/utils.h>
#include <xmma/implicit_gemm/fprop/gmem_tile.h>

#define MIN(m, n) ((m < n) ? m : n)
///////////////////////////////////////////////////////////////////////////////////////////////////

namespace xmma {
namespace implicit_gemm {
namespace dgrad {

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// T I L E   A
//
///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits, typename Cta_tile, typename Input_related, int BYTES_PER_LDG_,
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

    enum { FLT_T = Input_related::FLT_T };
    enum { FLT_R = Input_related::FLT_R };
    enum { FLT_S = Input_related::FLT_S };
    enum { STATIC_FILTER_SIZE = Input_related::STATIC_FILTER_SIZE };

    // Ctor.
    template <typename Params>
    inline __device__ Gmem_tile_base_a( const Params& params,
                                        void* smem,
                                        int bidz )
        : Base_( params, smem, params.k * params.g, params.img_gmem, bidz ),
          params_split_k_k_( params.split_k_k ),
          params_filter_trs_per_cta_( params.filter_trs_per_cta ) {
              precompute_residue_predicates( params_split_k_k_ );
    }

    // Ctor.
    template <typename Params>
    inline __device__ Gmem_tile_base_a( const Params& params,
                                        void* smem,
                                        const dim3& bidx,
                                        int tidx )
        : Gmem_tile_base_a( params, smem, bidx.z ) {

        int k, ndhw;
        int first_k_in_cta = bidx.z * params_split_k_k_;
        int k_base = 0;
        int ndhw_base = bidx.x * Cta_tile::M;

        // The position in the K dimension. It is the "row" of the matrix.
        if( Gmem_layout::ROW ) {
            ndhw_base += tidx / THREADS_PER_PIXEL;
            first_k_in_cta += tidx % THREADS_PER_PIXEL * Base_::ELTS_PER_LDG;
        } else {
            ndhw_base += tidx % THREADS_PER_PIXEL * Base_::ELTS_PER_LDG;
            first_k_in_cta += tidx / THREADS_PER_PIXEL;
        }

        if( params.g > 1 ) {
            k_base = bidx.y * Cta_tile::N;
        }

        // The masks for the predicates.
        const uint32_t MASK_T = Build_mask_t<FLT_T, FLT_R, FLT_S>::VALUE;
        const uint32_t MASK_R = Build_mask_r<FLT_T, FLT_R, FLT_S>::VALUE;
        const uint32_t MASK_S = Build_mask_s<FLT_T, FLT_R, FLT_S>::VALUE;

        // For each LDG, compute the NPQ decomposition, the pointer and issue the 1st LDG.
        int n[Base_::LDGS], o[Base_::LDGS], p[Base_::LDGS], q[Base_::LDGS];
        #pragma unroll
        for( int mi = 0; mi < Base_::LDGS; ++mi ) {

            int k_in_cta;
            // The index of the element loaded by this thread. That's the column.
            if( Gmem_layout::ROW ) {
                ndhw = ndhw_base + (mi / Base_::LDGS_UNROLL) * PIXELS_PER_LDG;
                k_in_cta = first_k_in_cta + mi % Base_::LDGS_UNROLL;
            } else {
                ndhw = ndhw_base + mi % Base_::LDGS_UNROLL;
                k_in_cta = first_k_in_cta + (mi / Base_::LDGS_UNROLL) * PIXELS_PER_LDG;
            }

            k = k_base + k_in_cta;

            // The masks -- initialization to -1 if it is valid.
            masks_[mi] = ( ndhw < params.ndhw && k < this->params_k_ ) ? uint32_t( -1 ) : 0u;

            if ( Cta_tile::N < Cta_tile::K && params.g > 1 ) {
                masks_[mi] &= ( k_in_cta < Cta_tile::N ? uint32_t( -1 ) : 0u );
            }

            if( Input_related::IS_SIMPLE_1x1x1 ) {
                if( Gmem_layout::ROW ) {
                    this->offsets_[mi] = ndhw * this->params_k_ + k;
                } else {
                    // Decompose ndhw into n and dhw.
                    int dhw;
                    xmma::fast_divmod( n[mi], dhw, ndhw, params.dhw, params.mul_dhw,
                                       params.shr_dhw );

                    this->offsets_[mi] = n[mi] * params.out_stride_n +
                                         dhw * params.out_stride_w +
                                         k * params.out_stride_c;
                }
            } else {
                // Decompose ndhw into n and dhw.
                int dhw;
                xmma::fast_divmod( n[mi], dhw, ndhw, params.dhw, params.mul_dhw,
                                       params.shr_dhw );
                // Decompose dhw into o and pq.
                int d, hw;
                xmma::fast_divmod( d, hw, dhw, params.hw, params.mul_hw, params.shr_hw );
                // Decompose pq into p and q.
                int h, w;
                xmma::fast_divmod( h, w, hw, params.w, params.mul_w, params.shr_w );

                // Compute d, h and w. We do a cross-correlation and tweak filter indices for colw.
                o[mi] = d + params.pad[0][0];
                p[mi] = h + params.pad[1][0];
                q[mi] = w + params.pad[2][0];

                // Take into account the inter-CTA split-K.
                o[mi] -= bidx.z * params.split_k_t * params.dilation[0];
                p[mi] -= bidx.z * params.split_k_r * params.dilation[1];

                // Assemble the 1st bit of the masks for T.
                uint32_t mask_t;
                if( STATIC_FILTER_SIZE ) {
                    mask_t = MASK_T;
                } else {
                    mask_t = params.mask_t;
                }
                uint32_t ilwalidation_mask_t = uint32_t( -1 ) ^ mask_t;
                uint32_t o_div, o_mod;
                xmma::fast_divmod_v2(o_div, o_mod, o[mi], params.stride[0], params.mul_stride[0], params.shr_stride[0]);
                if( (o_div) >= params.o
                    || o_mod ) {
                    masks_[mi] = masks_[mi] & ilwalidation_mask_t;
                }
                // Assemble the 1st bit of the masks for R.
                uint32_t mask_r;
                if( STATIC_FILTER_SIZE ) {
                    mask_r = MASK_R;
                } else {
                    mask_r = params.mask_r;
                }
                uint32_t ilwalidation_mask_r = uint32_t( -1 ) ^ mask_r;
                uint32_t p_div, p_mod;
                xmma::fast_divmod_v2(p_div, p_mod, p[mi], params.stride[1], params.mul_stride[1], params.shr_stride[1]);
                if( (p_div) >= params.p
                    || p_mod ) {
                    masks_[mi] = masks_[mi] & ilwalidation_mask_r;
                }

                // Assemble the 1st bit of the masks for S.
                uint32_t mask_s;
                if( STATIC_FILTER_SIZE ) {
                    mask_s = MASK_S;
                } else {
                    mask_s = params.mask_s;
                }
                uint32_t ilwalidation_mask_s = uint32_t( -1 ) ^ mask_s;
                uint32_t q_div, q_mod;
                xmma::fast_divmod_v2(q_div, q_mod, q[mi], params.stride[2], params.mul_stride[2], params.shr_stride[2]);
                if( (q_div) >= params.q
                    || q_mod ) {
                    masks_[mi] = masks_[mi] & ilwalidation_mask_s;
                }

                this->offsets_[mi] = n[mi] * params.out_stride_n +
                                     o_div * params.out_stride_d +
                                     p_div * params.out_stride_h +
                                     q_div * params.out_stride_w +
                                     k * params.out_stride_c;

            }
        }

        // Pack the predicates.
        xmma::implicit_gemm::pack_predicates( this->preds_, masks_, 1u );

        // Finalize the masks for T.
        int filter_t_per_cta;
        if( STATIC_FILTER_SIZE ) {
            filter_t_per_cta = FLT_T;
        } else {
            filter_t_per_cta = params.filter_t_per_cta;
        }
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
            for( int mi = 0; mi < Base_::LDGS; ++mi ) {
                uint32_t o_div, o_mod;
                xmma::fast_divmod_v2(o_div, o_mod, o[mi] - ti * params.dilation[0], params.stride[0], params.mul_stride[0], params.shr_stride[0]);
                if( (o_div) >= params.o
                    || o_mod ) {
                    masks_[mi] = masks_[mi] & mask_t;
                }
            }
        }

        // Finalize the masks for R.
        int filter_r_per_cta;
        if( STATIC_FILTER_SIZE ) {
            filter_r_per_cta = FLT_R;
        } else {
            filter_r_per_cta = params.filter_r_per_cta;
        }
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
            for( int mi = 0; mi < Base_::LDGS; ++mi ) {
                uint32_t p_div, p_mod;
                xmma::fast_divmod_v2(p_div, p_mod, p[mi] - ri * params.dilation[1], params.stride[1], params.mul_stride[1], params.shr_stride[1]);
                if( (p_div) >= params.p
                    || p_mod ) {
                    masks_[mi] = masks_[mi] & mask_r;
                }
            }
        }

        // Finalize the masks for S.
        int filter_s_per_cta;
        if( STATIC_FILTER_SIZE ) {
            filter_s_per_cta = FLT_S;
        } else {
            filter_s_per_cta = params.filter_s_per_cta;
        }
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
            for( int mi = 0; mi < Base_::LDGS; ++mi ) {
                uint32_t q_div, q_mod;
                xmma::fast_divmod_v2(q_div, q_mod, q[mi] - si * params.dilation[2], params.stride[2], params.mul_stride[2], params.shr_stride[2]);
                if( (q_div) >= params.q
                    || q_mod ) {
                    masks_[mi] = masks_[mi] & mask_s;
                }
            }
        }
    }

    // Move the pointers and update the predicates for R2P/P2R.
    inline __device__ void move( int next_trsi, int64_t delta ) {
        // Update the pointer.
        Base_::move( next_trsi, delta );

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

    // Precompute predicates for residue.
    inline __device__ void precompute_residue_predicates( int params_split_k_k_ ) {
        if( Gmem_layout::ROW ) {
            this->precompute_residue_predicates_a_t_b_n( params_split_k_k_ );
        } else {
            this->precompute_residue_predicates_a_n_b_t( params_split_k_k_ );
        }
    }

    // The residue to "fix" the predicates.
    inline __device__ void residue( int masks_to_clear = Base_::LDGS ) {
        // We are using a similar code as A^T.
        if( Gmem_layout::ROW ) {
            if( Base_::residue_a_t_b_n( this->params_split_k_k_ ) ) {
                return;
            }
        } else {
            if( Base_::residue_a_n_b_t( this->params_split_k_k_ ) ) {
                return;
            }
        }

    }

    // The split-k argument (TODO: move to base class).
    const int params_split_k_k_;
    // The part of the filter computed by that CTA.
    const int params_filter_trs_per_cta_;
    // The masks.
    uint32_t masks_[Base_::LDGS];
};

///////////////////////////////////////////////////////////////////////////////////////////////////

template <
    // The instruction traits.
    typename Traits,
    // The CTA tile description.
    typename Cta_tile,
    // Template parameters which are related to input parameters like filter size.
    typename Input_related,
    // The size in bytes of the LDGs.
    int BYTES_PER_LDG_ = 16,
    // Do we disable LDGSTS even on an architecture that has it?
    bool DISABLE_LDGSTS = false,
    // The ancestor/base class. See docs/gmem_tile.md.
    typename Ancestor = Gmem_tile_base_a<Traits, Cta_tile, Input_related, BYTES_PER_LDG_,
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

template <
    // The instruction traits.
    typename Traits,
    // The CTA tile description.
    typename Cta_tile,
    // Template parameters which are related to input parameters like filter size.
    typename Input_related,
    // The size in bytes of the LDGs.
    int BYTES_PER_LDG_ = 16,
    // Do we disable LDGSTS even on an architecture that has it?
    bool DISABLE_LDGSTS = false,
    // The ancestor/base class. See docs/gmem_tile.md.
    typename Ancestor = Gmem_tile_base_a<Traits, Cta_tile, Input_related, BYTES_PER_LDG_,
        xmma::Col, Cta_tile::M, Cta_tile::K>>
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

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// T I L E   B
//
////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits, typename Cta_tile, typename Input_related_, int BYTES_PER_LDG>
struct Gmem_tile_base_b : public gemm::Gmem_tile_base<Traits, Cta_tile, Cta_tile::N, Cta_tile::K,
                                                      Traits::BITS_PER_ELEMENT_B, BYTES_PER_LDG> {

    // The Input_related
    using Input_related = Input_related_;

    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;

    // The base class.
    using Base_ = gemm::Gmem_tile_base<Traits, Cta_tile, Cta_tile::N, Cta_tile::K,
                                       Traits::BITS_PER_ELEMENT_B, BYTES_PER_LDG>;

    // The number of threads needed to load a pixel.
    enum { THREADS_PER_FILTER = Base_::THREADS_PER_COLUMN };
    // The number of images loaded per LDG.
    enum { FILTERS_PER_LDG = Base_::COLUMNS_PER_LDG };

    enum { FLT_T = Input_related::FLT_T };
    enum { FLT_R = Input_related::FLT_R };
    enum { FLT_S = Input_related::FLT_S };
    enum { STATIC_FILTER_SIZE = Input_related::STATIC_FILTER_SIZE };
    //LDGSTS config
    static const bool USE_BYPASS = (!(Cta_tile::M == 256 && Cta_tile::N == 32)) || (FLT_T * FLT_R * FLT_S == 1);
    using LDGSTS_CFG = xmma::Ldgsts_config<true, USE_BYPASS>;
    // Ctor.
    template <typename Params>
    inline __device__ Gmem_tile_base_b( const Params& params,
                                        void* smem,
                                        const dim3& bidx,
                                        int tidx )
        : Base_( params, smem, params.k * params.g, params.flt_gmem, bidx.z ),
          params_delta_k_( FILTERS_PER_LDG * params.trsc ), params_split_k_k_( params.split_k_k ) {

        // The coordinate in the K dimension.
        int k_base = bidx.z * params_split_k_k_ + tidx / THREADS_PER_FILTER;
        int k = k_base;
        // The coordinates of the elements loaded by the thread.
        int c_in_tile = tidx % THREADS_PER_FILTER * Base_::ELTS_PER_LDG;
        int c = bidx.y * Cta_tile::N + c_in_tile;

        // Restrict C to the index in the group if we have more than 1 group.
        int c_in_group = c;

        if( params.g > 1 ) {
            k += bidx.y * Cta_tile::N;
            c_in_group = c & ( Cta_tile::N / Cta_tile::GROUPS - 1 );
        }

        // We treat the filter as a KRS x C matrix.
        int ktrs = k * params.trs;
        if( params.cross_correlation ) {
            ktrs += ( bidx.z ) * params.split_k_trs + ( bidx.z ) * params.split_k_rs;
        } else {
            ktrs += ( params.t - 1 - bidx.z ) * params.split_k_trs +
                    ( params.r - 1 - bidx.z ) * params.split_k_rs;
        }

        // The image tile does a cross-correlation, and we modify the filter to implement the colw.
        if( !params.cross_correlation ) {
            int filter_trs_per_cta;
            if( STATIC_FILTER_SIZE ) {
                filter_trs_per_cta = FLT_T * FLT_R * FLT_S;
            } else {
                filter_trs_per_cta = params.filter_trs_per_cta;
            }
            ktrs += filter_trs_per_cta - 1;
        }

        // Assemble the offset.
        this->offsets_[0] = ktrs * params.c + c_in_group;

        // Compute the predicates.
        uint32_t preds[Base_::LDGS];
#pragma unroll
        for( int ii = 0; ii < Base_::LDGS; ++ii ) {
            preds[ii] = k + ii*FILTERS_PER_LDG < this->params_k_;
        }

        if ( Cta_tile::N < Cta_tile::K && params.g > 1 ) {
#pragma unroll
            for( int ii = 0; ii < Base_::LDGS; ++ii ) {
                preds[ii] &= ( k_base + ii*FILTERS_PER_LDG < Cta_tile::N );
            }
        }

        // Finalize the predicates.
        int gc = params.g * params.c;
        asm volatile( "set.lt.u32.u32 %0, %1, %2;"
                      : "=r"( this->preds_[0] )
                      : "r"( c ), "r"( gc ) );
        this->preds_[0] &= xmma::pack_predicates( preds );

        if( Cta_tile::GROUPS > 1 ) {
            this->preds_[0] &= ( c_in_tile < Cta_tile::N / Xmma_tile::XMMAS_GROUPS
                                 ? 0xffffffffu : 0x0u );
        }

        precompute_residue_predicates( params_split_k_k_ );
    }

    // Compute the global memory pointers.
    inline __device__ void compute_load_pointers( const void* ( &ptrs )[Base_::LDGS] ) const {
#pragma unroll
        for( int ii = 0; ii < Base_::LDGS; ++ii ) {
            ptrs[ii] =
                &this->ptr_[Traits::offset_in_bytes_b( this->offsets_[0] + ii * params_delta_k_ )];
        }

    }

    // Compute the global memory pointers.
    template< int phase >
    inline __device__ void compute_load_pointers_per_phase( const void* ( &ptrs )[Base_::LDGS_PER_PHASE] ) const {
#pragma unroll
        for( int ii = phase * Base_::LDGS_PER_PHASE; ii < MIN((phase + 1) * Base_::LDGS_PER_PHASE, Base_::LDGS); ++ii ) {
            ptrs[ii - phase*Base_::LDGS_PER_PHASE] =
                &this->ptr_[Traits::offset_in_bytes_b( this->offsets_[0] + ii * params_delta_k_ )];
        }

    }

    // Precompute predicates for residue.
    inline __device__ void precompute_residue_predicates( int params_split_k_k_ ) {
        this->precompute_residue_predicates_a_n_b_t( params_split_k_k_ );
    }

    // The residue to "fix" the predicates.
    inline __device__ void residue() {
        this->residue_a_n_b_t( params_split_k_k_ );
    }

    // The constant C dimension and the delta in the k dimension.
    const int params_delta_k_, params_split_k_k_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    // The instruction traits.
    typename Traits,
    // The CTA tile description.
    typename Cta_tile,
    // Template parameters which are related to input parameters like filter size.
    typename Input_related,
    // The size in bytes of the LDGs.
    int BYTES_PER_LDG = 16,
    // Do we disable LDGSTS even on an architecture that has it?
    bool DISABLE_LDGSTS = false,
    // The ancestor/base class. See docs/gmem_tile.md.
    typename Ancestor = Gmem_tile_base_b<Traits, Cta_tile, Input_related, BYTES_PER_LDG>
>
struct Gmem_tile_b : public Ldgsts_selector<Traits,
                                            fprop::Rebase_gmem_tile_with_ldgsts_b<Ancestor, typename Ancestor::LDGSTS_CFG>,
                                            fprop::Rebase_gmem_tile_with_ldg_and_sts_b<Ancestor>,
                                            DISABLE_LDGSTS>::Class {

    // The base class.
    using Base_ = typename Ldgsts_selector<Traits,
                                           fprop::Rebase_gmem_tile_with_ldgsts_b<Ancestor, typename Ancestor::LDGSTS_CFG>,
                                           fprop::Rebase_gmem_tile_with_ldg_and_sts_b<Ancestor>,
                                           DISABLE_LDGSTS>::Class;

    // The expected shared memory layout.
    using Smem_layout = xmma::Row;

    // Ctor.
    template <typename Params>
    inline __device__ Gmem_tile_b( const Params& params,
                                   void* smem,
                                   const dim3& bidx,
                                   int tidx )
        : Base_( params, smem, bidx, tidx ) {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// T I L E   C
//
////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    // The instruction traits.
    typename Traits,
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // Layout_C
    typename Layout,
    // The number of bytes per STG.
    int BYTES_PER_STG = 16,
    // The fragment class before writing data to global memory.
    typename Fragment_c = xmma::Fragment_c<Traits, Cta_tile>,
    // Assuming tensor is packed and tensor strides won't be used.
    bool DISABLE_STRIDES = false
>
struct Gmem_tile_epilogue_base
    : public xmma::implicit_gemm::fprop::Gmem_tile_epilogue_base<Traits,
        Cta_tile, Layout, BYTES_PER_STG, Fragment_c, DISABLE_STRIDES> {

    // The base class.
    using Base = xmma::implicit_gemm::fprop::Gmem_tile_epilogue_base<Traits,
        Cta_tile, Layout, BYTES_PER_STG, Fragment_c, DISABLE_STRIDES>;

    // Ctor.
    template <typename Params>
    inline __device__ Gmem_tile_epilogue_base( const Params& params,
                                               int bidm,
                                               int bidn,
                                               int bidz,
                                               int tidx )
        : Base(params.out_gmem,
                params.res_gmem,
                params.n,
                params.d,
                params.h,
                params.w,
                params.c * params.g,
                params.ndhw,
                params.img_stride_n,
                params.img_stride_d,
                params.img_stride_h,
                params.img_stride_w,
                params.img_stride_c,
                params.w,
                params.mul_w,
                params.shr_w,
                params.hw,
                params.mul_hw,
                params.shr_hw,
                params.dhw,
                params.mul_dhw,
                params.shr_dhw,
                bidm,
                bidn,
                bidz,
                tidx ) {
    }
};

template<
    // The instruction traits.
    typename Traits,
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The number of bytes per STG.
    int BYTES_PER_STG = 16,
    // The fragment class before writing data to global memory.
    typename Fragment_c = xmma::Fragment_c<Traits, Cta_tile>,
    // Assuming tensor is packed and tensor strides won't be used.
    bool DISABLE_STRIDES = false
>
struct Gmem_tile_c_t : public Gmem_tile_epilogue_base<Traits, Cta_tile,
    xmma::Row, BYTES_PER_STG, Fragment_c, DISABLE_STRIDES> {
    using Base = Gmem_tile_epilogue_base<Traits, Cta_tile, xmma::Row,
        BYTES_PER_STG, Fragment_c, DISABLE_STRIDES>;

    template <typename Params>
    inline __device__ Gmem_tile_c_t( const Params& params,
                                     int bidm,
                                     int bidn,
                                     int bidz,
                                     int tidx )
       : Base ( params, bidm, bidn, bidz, tidx ) {
    }

};

template<
    // The instruction traits.
    typename Traits,
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The number of bytes per STG.
    int BYTES_PER_STG = 16,
    // The fragment class before writing data to global memory.
    typename Fragment_c = xmma::Fragment_c<Traits, Cta_tile>,
    // Assuming tensor is packed and tensor strides won't be used.
    bool DISABLE_STRIDES = false
>
struct Gmem_tile_c_n : public Gmem_tile_epilogue_base<Traits, Cta_tile,
    xmma::Col, BYTES_PER_STG, Fragment_c, DISABLE_STRIDES> {
    using Base = Gmem_tile_epilogue_base<Traits, Cta_tile, xmma::Col,
        BYTES_PER_STG, Fragment_c, DISABLE_STRIDES>;

    template <typename Params>
    inline __device__ Gmem_tile_c_n( const Params& params,
                                     int bidm,
                                     int bidn,
                                     int bidz,
                                     int tidx )
       : Base ( params, bidm, bidn, bidz, tidx ) {
    }

};


///////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace dgrad
}  // namespace implicit_gemm
}  // namespace xmma
