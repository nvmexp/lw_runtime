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
#include <xmma/ext/sparse/helpers/epilogue.h>

#include <xmma/ext/sparse/implicit_spgemm/utils.h>
#include <xmma/gemm/gmem_tile.h>

namespace xmma {
namespace ext {
namespace implicit_gemm {
namespace fprop {

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// T I L E   M E T A D A T A
//
///////////////////////////////////////////////////////////////////////////////////////////////////

template<
    // The instruction traits.
    typename Traits,
    // The CTA tile description.
    typename Cta_tile,
    // The size in bytes of the LDGs.
    int BYTES_PER_LDG_ = 4,
    // Do we disable LDGSTS even on an architecture that has it?
    bool DISABLE_LDGSTS = false
>
struct Gmem_tile_e {

    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;

    enum { BYTES_PER_LDG = BYTES_PER_LDG_ };

    enum { BYTES_PER_ELEMENT = (Cta_tile::K == 128
        ? (Cta_tile::WARPS_N != 4 ? 16 : 8)
        : (Cta_tile::WARPS_N != 4 ? 8 : 4))};

    enum { ELTS_PER_UINT = (Cta_tile::K == 128
        ? 32 : Cta_tile::K == 32 ? 8 : 16) };

    enum { BITS_PER_MD = Cta_tile::K == 32 ? 4 : 2 }; //TF32 1:2 FP16,INT8 2:4

    enum { MD_SCALE = Cta_tile::K == 128 ? 2 : 1 };

    // The expected shared memory layout.
    using Smem_layout = xmma::Row;

    // Ctor.
    template< typename Params >
    inline __device__ Gmem_tile_e (const Params &params,
            void *smem, const dim3 &bidx, int tidx) {

        static_assert((Xmma_tile::M_PER_WARP == 64), "");

        ptr_ = reinterpret_cast<const char*>(params.e_gmem);

        const int ty = (tidx / 32) / Cta_tile::WARPS_N;
        const int tz = (tidx / 32) % Cta_tile::WARPS_N;
        const int warp_id = tidx % 32;

        const uint32_t c = params.padded_c;

        int kg, g;
        xmma::fast_divmod( g, kg, bidx.x, params.kn, params.mul_k, params.shr_k);
        //this->ptr_ += (g * params.padded_k * (c / 2)) / (8 / BITS_PER_MD);
        const int n = kg * Cta_tile::M + ty * Xmma_tile::M_PER_WARP;
        // (8 / BITS_PER_MD), e.g., 4, which means to translate blocked 2 bit metadata offset to bytes
        uint32_t offset = (g * params.padded_k * (c / 2)) / (8 / BITS_PER_MD) + 
                          ((n / 64) * (c / ELTS_PER_UINT) * ELTS_PER_UINT * BITS_PER_MD) * BYTES_PER_LDG + 
                          (tz * 32 + warp_id * 1)*BYTES_PER_ELEMENT;
/*                          
        uint32_t effctive_offset = xmma::div_up( params.g * params.padded_k, Xmma_tile::M_PER_WARP) * 
                                   Xmma_tile::M_PER_WARP * 
                                   (((c / 2) / ELTS_PER_UINT) * 4) * MD_SCALE;
*/
        uint32_t effctive_offset = params.g * params.padded_k * 
                                   (((c / 2) / ELTS_PER_UINT) * 4) * MD_SCALE;

        if( offset >= effctive_offset ) {
            uint32_t dummy_offset = (g * params.padded_k * (c / 2)) / (8 / BITS_PER_MD) + 
                                    (((kg * Cta_tile::M) / 64) * (c / ELTS_PER_UINT) * ELTS_PER_UINT * BITS_PER_MD) * BYTES_PER_LDG + 
                                    warp_id * BYTES_PER_ELEMENT;
            offset = dummy_offset;
        }

        this->ptr_ += offset;
/*        
        this->ptr_ += offset;
        // Only 1 predication var needed since we always pad the gemm-k to full cta-k
        uint32_t preds[1];
        // (c / 2) / ELTS_PER_UINT -- derive needed uint metadata element
        // * 4 -- Each unit is 4 bytes
        // MD_SCALE -- For int8 spmma inst, we use 2 32-bit regs for a row; rest we use 1 32-bit regs for a row
        uint32_t effctive_offset = xmma::div_up( params.g * params.padded_k, Xmma_tile::M_PER_WARP) * 
                                   Xmma_tile::M_PER_WARP * 
                                   (( (c / 2) / ELTS_PER_UINT) * 4) * MD_SCALE;        
        preds[0] = offset < effctive_offset;
        this->preds_ = xmma::pack_predicates(preds);
*/
    }

    inline __device__ void move () {
        ptr_ += Cta_tile::K*4*BITS_PER_MD;
    }

    // Load a tile from global memory.
    template< typename Xmma_smem_tile >
    inline __device__ void load(Xmma_smem_tile &smem) {
        smem.store(ptr_);
        //Need second
        if (Cta_tile::WARPS_N == 1) {
            smem.store(ptr_ + 32 * BYTES_PER_ELEMENT, 1);
        }
    }

    const char* ptr_;
};

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// T I L E   B
//
///////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Traits, typename Cta_tile, int BYTES_PER_LDG, bool USE_PREDICATES_,
    typename Input_related >
struct Gmem_tile_base_b : public xmma::gemm::Gmem_tile_base<Traits,
                                                      Cta_tile,
                                                      Cta_tile::K,
                                                      Cta_tile::N,
                                                      Traits::BITS_PER_ELEMENT_B,
                                                      BYTES_PER_LDG> {

    // The base class.
    using Base_ = xmma::gemm::Gmem_tile_base<Traits,
                                       Cta_tile,
                                       Cta_tile::K,
                                       Cta_tile::N,
                                       Traits::BITS_PER_ELEMENT_B,
                                       BYTES_PER_LDG>;

    // Make sure we use a single predicate register (for the moment).
    static_assert(Base_::PRED_REGS == 1, "");

    // The number of threads needed to load a pixel.
    enum { THREADS_PER_PIXEL = Base_::THREADS_PER_COLUMN };
    // The number of images loaded per LDG.
    enum { PIXELS_PER_LDG = Base_::COLUMNS_PER_LDG };

    enum { USE_PREDICATES = USE_PREDICATES_ };

    enum { IS_SIMPLE_1x1x1 = Input_related::IS_SIMPLE_1x1x1 };

    enum { IS_FLT_1x1x1 = (Input_related::FLT_T == 1 &&
                           Input_related::FLT_R == 1 &&
                           Input_related::FLT_S == 1) };

    enum { STEPS = Div_up<Cta_tile::N, Cta_tile::THREADS_PER_CTA>::VALUE };

    // Ctor.
    template< typename Params >
    inline __device__ Gmem_tile_base_b(const Params &params, void *smem)
        : Base_(params, smem, params.c*params.g, params.img_gmem)
        , params_split_k_c_(params.split_k_c)
        , params_filter_trs_per_cta_(params.filter_trs_per_cta)
        , params_trsc_(params.trsc*2)
        , params_rsc_(params.rsc), params_mul_rsc_(params.mul_rsc)
        , params_shr_rsc_(params.shr_rsc)
        , params_sc_(params.sc), params_mul_sc_(params.mul_sc), params_shr_sc_(params.shr_sc)
        , params_c_(params.c), params_mul_c_(params.mul_c), params_shr_c_(params.shr_c)
        , params_dilation_d_(params.dilation[0])
        , params_dilation_h_(params.dilation[1])
        , params_dilation_w_(params.dilation[2])
        , params_n_(params.n)
        , params_d_(params.d)
        , params_h_(params.h)
        , params_w_(params.w)
        , smem_(xmma::get_smem_pointer(smem)) {
    }

    // Ctor.
    template< typename Params >
    inline __device__ Gmem_tile_base_b(const Params &params, void *smem, const dim3 &bidx, int tidx)
        : Gmem_tile_base_b(params, smem) {
        // The position in the C dimension. It is the "row" of the matrix.
        trsc_ = tidx % THREADS_PER_PIXEL * Base_::ELTS_PER_LDG;

        int k, g;
        xmma::fast_divmod( g, k, bidx.x, params.kn, params.mul_k, params.shr_k);
        this->ptr_ += Traits::offset_in_bytes_b(g * params.c);

        int trs, rs, t, r, s, c;
        if (!IS_SIMPLE_1x1x1 && !IS_FLT_1x1x1) {
            xmma::fast_divmod( t, trs, trsc_, params.rsc, params.mul_rsc,  params.shr_rsc );
            xmma::fast_divmod( r, rs, trs, params.sc, params.mul_sc, params.shr_sc );
            xmma::fast_divmod( s, c, rs, params.c, params.mul_c, params.shr_c );

            t *= params.dilation[0];
            r *= params.dilation[1];
            s *= params.dilation[2];
            c *= 1;//params.img_stride_c;
            if(trsc_ >= params_trsc_) c = params_trsc_;
        } else {
            t = 0; r = 0; s = 0; c = trsc_;
        }

        //const int STEPS = Div_up<Cta_tile::N, Cta_tile::THREADS_PER_CTA>::VALUE;
        int opq, pq, n, o, p, q;
        #pragma unroll
        for( int ii = 0; ii < STEPS; ++ii ) {
            // The index of the element loaded by this thread. That's the column.
            int nopq = bidx.y * Cta_tile::N + tidx + ii * Cta_tile::THREADS_PER_CTA;

            xmma::fast_divmod(n, opq, nopq, params.opq, params.mul_opq, params.shr_opq);
            // Decompose opq into o and pq.
            xmma::fast_divmod(o, pq, opq, params.pq, params.mul_pq, params.shr_pq);
            // Decompose pq into p and q.
            xmma::fast_divmod(p, q, pq, params.q, params.mul_q, params.shr_q);

            // Compute d, h and w. We do a cross-correlation and tweak filter indices for colw.
            if (!IS_SIMPLE_1x1x1) {
                o = o*params.stride[0] - params.pad[0][0];
                p = p*params.stride[1] - params.pad[1][0];
                q = q*params.stride[2] - params.pad[2][0];
            }
            if (nopq >= params.nopq) n = params.n+1;

            if( tidx + ii * Cta_tile::THREADS_PER_CTA < Cta_tile::N ) {
                int4 indices = make_int4( n, o, p, q );
                xmma::sts( smem_ + (tidx + ii * Cta_tile::THREADS_PER_CTA) * sizeof( int4 ),
                    reinterpret_cast<const uint4&>( indices ) );
            }
        }

        smem_ += (tidx / THREADS_PER_PIXEL) * sizeof( uint4 );

        __syncthreads();

        // For each LDG, compute the NPQ decomposition, the pointer and issue the 1st LDG.
        uint32_t preds[Base_::LDGS];
        #pragma unroll
        for( int mi = 0; mi < Base_::LDGS; ++mi ) {

            // The index of the element loaded by this thread. That's the column.
            int idx = mi * PIXELS_PER_LDG;
            uint4 nopq;
            xmma::lds( nopq, smem_ + idx * sizeof( uint4 ) );

            n = reinterpret_cast<const int&>( nopq.x );
            const int d = reinterpret_cast<const int&>( nopq.y ) + t;
            const int h = reinterpret_cast<const int&>( nopq.z ) + r;
            const int w = reinterpret_cast<const int&>( nopq.w ) + s;
            if(!(IS_SIMPLE_1x1x1 == 0 && mi>=2 &&
                (((Cta_tile::M==128 && Cta_tile::N==128) ||
                (Cta_tile::M==256 && Cta_tile::N==128)) &&
                Traits::ACLWMULATOR_32BIT))) { //To reduce register pressure
                n_[mi] = n;
                d_[mi] = reinterpret_cast<const int&>( nopq.y );
                h_[mi] = reinterpret_cast<const int&>( nopq.z );
                w_[mi] = reinterpret_cast<const int&>( nopq.w );
            }

            if (IS_SIMPLE_1x1x1) {
                const uint32_t offset = bidx.y * Cta_tile::N +
                    tidx / THREADS_PER_PIXEL + mi * PIXELS_PER_LDG;
                if (USE_PREDICATES) {
                    preds[mi] = (offset < params.nopq && c < params.c);
                    ndhw_preds_[mi] = offset < params.nopq;
                }
                // Compute the final pointers.
                this->offsets_[mi] = offset*params.c*params.g;
            } else {
                // Compute the final pointers.
                this->offsets_[mi] = n*params.dhwc +
                                     d*params.hwc +
                                     h*params.wc +
                                     w*params.c*params.g;
                if (!IS_FLT_1x1x1) {
                    this->offsets_[mi] += c;
                }
                if (USE_PREDICATES) {
                    if (IS_FLT_1x1x1) {
                        ndhw_preds_[mi] = n < params.n &&
                                 (unsigned) d < params.d &&
                                 (unsigned) h < params.h &&
                                 (unsigned) w < params.w;
                    }
                    preds[mi] = (n < params.n &&
                                 (unsigned) d < params.d &&
                                 (unsigned) h < params.h &&
                                 (unsigned) w < params.w &&
                                 c < params.c);
                }
            }
        }
        if (IS_FLT_1x1x1 || IS_SIMPLE_1x1x1) {
            this->ptr_ += Traits::offset_in_bytes_b(c);
        }

        // Pack the predicates.
        if (USE_PREDICATES) {
            this->preds_[0] = xmma::pack_predicates(preds);
        } else {
            this->preds_[0] = 0xffffffffu;
        }
    }

    // Move the pointers and update the predicates for R2P/P2R.
    template< typename Params >
    inline __device__ void move_b(int next_trsi, int64_t delta, const Params &params) {
        // Update the pointer.
        trsc_ += Cta_tile::K;
        int trs, rs, t, r, s, c;

        if(params.simple1x1x1 || IS_SIMPLE_1x1x1 || IS_FLT_1x1x1) {
        t = 0;
        r = 0;
        s = 0;
        c = trsc_;
        } else {
        if(trsc_ >= params_trsc_) {
            c = params_trsc_;
        } else {
        xmma::fast_divmod( t, trs, trsc_, params_rsc_, params_mul_rsc_,  params_shr_rsc_ );
        xmma::fast_divmod( r, rs, trs, params_sc_, params_mul_sc_, params_shr_sc_ );
        xmma::fast_divmod( s, c, rs, params_c_, params_mul_c_, params_shr_c_ );

        t *= params_dilation_d_;
        r *= params_dilation_h_;
        s *= params_dilation_w_;
        }
        }

        if (IS_SIMPLE_1x1x1 || IS_FLT_1x1x1) {
            this->ptr_ += Traits::offset_in_bytes_b(Cta_tile::K);
        }

        // For each LDG, compute the NPQ decomposition, the pointer and issue the 1st LDG.
        uint32_t preds[Base_::LDGS];
        #pragma unroll
        for( int mi = 0; mi < Base_::LDGS; ++mi ) {

            int n, d, h, w;
            if(IS_SIMPLE_1x1x1 == 0 && mi>=2 &&
                (((Cta_tile::M==128 && Cta_tile::N==128) ||
                (Cta_tile::M==256 && Cta_tile::N==128)) &&
                Traits::ACLWMULATOR_32BIT)) {
                int idx = mi * PIXELS_PER_LDG;
                uint4 nopq;
                xmma::lds( nopq, smem_ + idx * sizeof( uint4 ) );

                n = reinterpret_cast<int&>( nopq.x );
                d = reinterpret_cast<int&>( nopq.y ) + t;
                h = reinterpret_cast<int&>( nopq.z ) + r;
                w = reinterpret_cast<int&>( nopq.w ) + s;

            } else {
                n = n_[mi];
                d = d_[mi] + t;
                h = h_[mi] + r;
                w = w_[mi] + s;
            }
            if (IS_SIMPLE_1x1x1) {
                if (USE_PREDICATES) {
                    preds[mi] = (ndhw_preds_[mi] && c < params.c);
                }
            } else {
                if (USE_PREDICATES) {
                    if (IS_FLT_1x1x1) {
                        preds[mi] = ndhw_preds_[mi] && c < params_c_;
                    } else {
                        preds[mi] = ( n < params_n_ &&
                             (uint16_t) d < params_d_ &&
                             (uint16_t) h < params_h_ &&
                             (uint16_t) w < params_w_ &&
                             c < params_c_
                             );
                    }
                }
                if (!IS_FLT_1x1x1) {
                    // Compute the final pointers.
                    this->offsets_[mi] = n*params.dhwc +
                                     d*params.hwc +
                                     h*params.wc +
                                     w*params.c*params.g +
                                     c;
                }
            }
        }

        // Pack the predicates.
        if (USE_PREDICATES) {
            this->preds_[0] = xmma::pack_predicates(preds);
        }
    }

    // The residue to "fix" the predicates.
    inline __device__ void residue() {
    }

    // The split-k argument (TODO: move to base class).
    const int params_split_k_c_;
    // The part of the filter computed by that CTA.
    const int params_filter_trs_per_cta_;
    uint32_t params_rsc_, params_mul_rsc_, params_shr_rsc_;
    uint32_t params_sc_, params_mul_sc_, params_shr_sc_;
    uint32_t params_c_, params_mul_c_, params_shr_c_;
    uint16_t params_n_;
    uint16_t params_d_;
    uint16_t params_h_;
    uint16_t params_w_;

    int16_t params_dilation_d_, params_dilation_h_, params_dilation_w_;
    bool ndhw_preds_[Base_::LDGS];

    int16_t n_[Base_::LDGS];
    int16_t d_[Base_::LDGS];
    int16_t h_[Base_::LDGS];
    int16_t w_[Base_::LDGS];

    // The masks.
    uint32_t smem_;
    int trsc_, params_trsc_;
};

///////////////////////////////////////////////////////////////////////////////////////////////////

template<
    // The instruction traits.
    typename Traits,
    // The CTA tile description.
    typename Cta_tile,
    // Input_related
    typename Input_related,
    // Use or not predicates
    bool USE_PREDICATES = true,
    // The size in bytes of the LDGs.
    int BYTES_PER_LDG = 16,
    // Do we disable LDGSTS even on an architecture that has it?
    bool DISABLE_LDGSTS = false,
    // The ancestor/base class. See docs/gmem_tile.md.
    typename Ancestor = Gmem_tile_base_b<Traits, Cta_tile, BYTES_PER_LDG, USE_PREDICATES,
        Input_related>
>
struct Gmem_tile_b : public xmma::Ldgsts_selector<Traits,
                                            xmma::gemm::Rebase_gmem_tile_with_ldgsts<Ancestor>,
                                            xmma::gemm::Rebase_gmem_tile_with_ldg_and_sts<Ancestor>,
                                            DISABLE_LDGSTS>::Class {

    // The base class.
    using Base_ = typename xmma::Ldgsts_selector<Traits,
                                           xmma::gemm::Rebase_gmem_tile_with_ldgsts<Ancestor>,
                                           xmma::gemm::Rebase_gmem_tile_with_ldg_and_sts<Ancestor>,
                                           DISABLE_LDGSTS>::Class;
    // The expected shared memory layout.
    using Smem_layout = xmma::Col;

    // Ctor.
    template< typename Params >
    inline __device__ Gmem_tile_b(const Params &params, void *smem, const dim3 &bidx, int tidx)
        : Base_(params, smem, bidx, tidx) {
    }
};

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// T I L E   A
//
///////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Traits, typename Cta_tile, int BYTES_PER_LDG, bool USE_PREDICATES_>
struct Gmem_tile_base_a : public xmma::gemm::Gmem_tile_base<Traits,
                                                      Cta_tile,
                                                      Cta_tile::HALF_K,
                                                      Cta_tile::M,
                                                      Traits::BITS_PER_ELEMENT_A,
                                                      BYTES_PER_LDG> {

    using Base_ = xmma::gemm::Gmem_tile_base<Traits,
                                       Cta_tile,
                                       Cta_tile::HALF_K,
                                       Cta_tile::M,
                                       Traits::BITS_PER_ELEMENT_A,
                                       BYTES_PER_LDG>;

    // The number of threads needed to load a pixel.
    enum { THREADS_PER_FILTER = Base_::THREADS_PER_COLUMN };
    // The number of images loaded per LDG.
    enum { FILTERS_PER_LDG = Base_::COLUMNS_PER_LDG };

    enum { FILTERS_PER_LOAD = 64 };

    enum { USE_PREDICATES = USE_PREDICATES_ };

    // Ctor.
    template< typename Params >
    inline __device__ Gmem_tile_base_a(const Params &params, void *smem, const dim3 &bidx, int tidx)
        : Base_(params, smem, (params.c / 2)*params.g, params.flt_gmem)
        , params_delta_k_(params.trsc)
        , params_split_k_c_(params.split_k_c)
        , params_trsc_(params.trs * (params.c / 2)) {

        uint32_t k, c = 0, offset=0;

        const uint32_t row_id = tidx / THREADS_PER_FILTER;
        const uint32_t col_id = tidx % THREADS_PER_FILTER;

        int kg, g;
        xmma::fast_divmod( g, kg, bidx.x, params.kn, params.mul_k, params.shr_k);
        this->ptr_ += Traits::offset_in_bytes_a(g * params.k * params.trsc);

        k = kg * Cta_tile::M + row_id / 8 + (row_id % 8) * 8;

        c = col_id * Base_::ELTS_PER_LDG;

        offset = k * (params.c / 2) * params.trs + c;

        const int off1 = (Cta_tile::THREADS_PER_CTA / THREADS_PER_FILTER) / 8;
        const int off2 = 8 / off1;

        this->ptr_ += Traits::offset_in_bytes_a(static_cast<int64_t>(offset));

        #pragma unroll
        for (int ii = 0; ii < Base_::LDGS; ii++) {
            this->offsets_[ii] =
                ((ii/off2)*FILTERS_PER_LOAD + (ii%off2)*off1) * params_delta_k_;
        }

        // Finalize the predicates.
        if (USE_PREDICATES == 0) {
            this->preds_[0] = 0xffffffffu;
        } else {
            params_trsc_ -= c;
            // Compute the predicates.
            uint32_t preds[Base_::LDGS];
            #pragma unroll
            for( int ii = 0; ii < Base_::LDGS; ++ii ) {
                preds[ii] =
                    (k + ((ii/off2)*FILTERS_PER_LOAD + (ii%off2)*off1) < params.k)
                    && (params_trsc_ > 0);
            }
            this->preds_[0] = xmma::pack_predicates(preds);
        }
    }

    inline __device__ void move (int rsi, int64_t delta) {
        this->ptr_ += Traits::offset_in_bytes_a(Cta_tile::K / 2);
        if (USE_PREDICATES == 1) {
        params_trsc_ -= Cta_tile::K / 2;
        if (params_trsc_ <= 0) {
            this->preds_[0] = 0u;
        }
        }
    }

    // The residue to "fix" the predicates.
    inline __device__ void residue() {
    }

    // The constant C dimension and the delta in the k dimension.
    const int params_delta_k_, params_split_k_c_;
    int params_trsc_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    // The instruction traits.
    typename Traits,
    // The CTA descriptor.
    typename Cta_tile,
    // The size in bytes of the LDG.
    int BYTES_PER_LDG,
    // The base class.
    typename Base = Gmem_tile_base_a<Traits, Cta_tile, BYTES_PER_LDG, true>
>
struct Gmem_tile_with_ldg_and_sts_a : public Base {

    // It does not use LDGSTS.
    enum { USE_LDGSTS = 0 };

    // DEBUG: The group implementation assumes LDG.128 and K == 64.
    static_assert(Cta_tile::GROUPS == 1 || (BYTES_PER_LDG == 16 && Cta_tile::K == 64), "");
    // END OF DEBUG.

    // Ctor.
    template< typename Params >
    inline __device__ Gmem_tile_with_ldg_and_sts_a(const Params &params,
                                                   void *smem,
                                                   const dim3 &bidx,
                                                   int tidx)
        : Base(params, smem, bidx, tidx) {
    }

    // Load a tile from global memory.
    template< typename Smem_tile >
    inline __device__ void load(Smem_tile &, const uint64_t mem_desc) {
        // Prepare the pointers.
        const void *ptrs[Base::LDGS];
        this->compute_load_pointers(ptrs);

        // Issue the loads.
        if( Cta_tile::GROUPS == 16 ) {
            xmma::ldg_force_64(this->fetch_, ptrs, this->preds_[0]);
        } else {
            xmma::ldg(this->fetch_, ptrs, this->preds_);
        }
    }

    // The fetch registers.
    typename xmma::Uint_from_size_in_bytes<Base::BYTES_PER_LDG>::Type fetch_[Base::LDGS];
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Base >
using Rebase_gmem_tile_with_ldg_and_sts_a = Gmem_tile_with_ldg_and_sts_a<typename Base::Traits,
                                                                         typename Base::Cta_tile,
                                                                         Base::BYTES_PER_LDG,
                                                                         Base>;

////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    // The instruction traits.
    typename Traits,
    // The CTA tile description.
    typename Cta_tile,
    // Use or not predicates
    bool USE_PREDICATES = true,
    // The size in bytes of the LDGs.
    int BYTES_PER_LDG = 16,
    // Do we disable LDGSTS even on an architecture that has it?
    bool DISABLE_LDGSTS = false,
    // The ancestor/base class. See docs/gmem_tile.md.
    typename Ancestor = Gmem_tile_base_a<Traits, Cta_tile, BYTES_PER_LDG, USE_PREDICATES>
>
struct Gmem_tile_a : public xmma::Ldgsts_selector<Traits,
                             xmma::gemm::Rebase_gmem_tile_with_ldgsts<Ancestor>,
                             Gmem_tile_with_ldg_and_sts_a<Traits, Cta_tile, BYTES_PER_LDG>,
                             DISABLE_LDGSTS>::Class {

    // The base class.
    using Base_ =
        typename xmma::Ldgsts_selector<Traits,
                                 xmma::gemm::Rebase_gmem_tile_with_ldgsts<Ancestor>,
                                 Gmem_tile_with_ldg_and_sts_a<Traits, Cta_tile, BYTES_PER_LDG>,
                                 DISABLE_LDGSTS>::Class;

    // The expected shared memory layout.
    using Smem_layout = xmma::Row;

    // Ctor.
    template< typename Params >
    inline __device__ Gmem_tile_a(const Params &params, void *smem, const dim3 &bidx, int tidx)
        : Base_(params, smem, bidx, tidx) {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// T I L E   C
//
////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    typename Traits,
    typename Cta_tile,
    typename Fragment_c = xmma::Fragment_c<Traits, Cta_tile>
>
struct Gmem_tile_epilogue
    : public xmma::helpers::Gmem_tile_epilogue<Traits, Cta_tile, xmma::Row, Fragment_c> {

    enum { STGS = 2 };

    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;

    // The base class.
    using Base = xmma::helpers::Gmem_tile_epilogue<Traits, Cta_tile, xmma::Row, Fragment_c>;
    // Fragment for dst if beta != 0.0
    using Fragment_aclwmulator = xmma::Fragment_aclwmulator<Traits>;

    // Ctor.
    template< typename Params >
    inline __device__ Gmem_tile_epilogue(const Params &params, int bidm, int bidn, int bidz, int tidx)
        : Base(params.nopq,
               params.k * params.g,
               params.k * params.g,
               reinterpret_cast<char*>(params.out_gmem),
               reinterpret_cast<const char*>( params.res_gmem ),
               bidm,
               bidn,
               bidz,
               tidx),
            params_m_(params.nopq),
            params_n_(params.k),
            params_stride_n_(params.k*params.g) {

        char* ptr = reinterpret_cast<char*>(params.out_gmem);
        const char* ptr_res = reinterpret_cast<const char*>(params.res_gmem);

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
        int row = ((tidx & WARP_MASK_N) / WARP_DIV_N) * Xmma_tile::N_PER_XMMA
            + (tidx % 4) * 2;

        int col = ((tidx & WARP_MASK_M) / WARP_DIV_M) * Xmma_tile::M_PER_WARP
            + ((tidx % Cta_tile::THREADS_PER_WARP)/4) * 8;

        int k, g;
        xmma::fast_divmod( g, k, bidm, params.kn, params.mul_k, params.shr_k);

        // Compute the output position for each thread.
        m_ = bidn * Cta_tile::N + row;
        n_ = k * Cta_tile::M + col;

        // The pointer.
        ptr_ = &ptr[Traits::offset_in_bytes_c(m_*params_stride_n_ + n_ + g * params.k)];
        ptr_res_ = &ptr_res[Traits::offset_in_bytes_c(m_*params_stride_n_ + n_ + g * params.k)];
    }

    // Compute the row offset.
    static inline __device__ constexpr int compute_offset(int mi, int i) {
        return (mi % 2) * Xmma_tile::N_PER_XMMA / 2
         + (mi / 2) * Xmma_tile::N_PER_XMMA_PER_CTA + i;
    }

    // Is a given output valid?
    inline __device__ int compute_output_mask(const int offset) const {
        return (offset + m_) < params_m_ && n_ < params_n_;
    }

    // Store the data to global memory.
    inline __device__ void store(int mi, int ii, const Fragment_c data) {
        int offset = compute_offset(mi, ii);
        char *ptr = &ptr_[Traits::offset_in_bytes_c(offset*params_stride_n_)];

        int mask = compute_output_mask(offset);

        if (mask) {
            if (Traits::BITS_PER_ELEMENT_C == 8) {
                xmma::stg(ptr, data.to_int2());
            } else if (Traits::BITS_PER_ELEMENT_C == 32) {
                xmma::stg(ptr,
                make_uint4(data.reg(0),data.reg(1), data.reg(2),data.reg(3)));

                xmma::stg(ptr + Traits::offset_in_bytes_c(4),
                make_uint4(data.reg(4),data.reg(5), data.reg(6),data.reg(7)));
            } else {
                xmma::stg(ptr,
                make_uint4(data.reg(0),data.reg(1), data.reg(2),data.reg(3)));
            }
        }
    }

    // Load the data from global memory.
    inline __device__ void load(Fragment_c &data, int mi, int i) {

        int offset = compute_offset(mi, i);
        const char *ptr = &ptr_res_[Traits::offset_in_bytes_c(offset*params_stride_n_)];

        int mask = compute_output_mask(offset);

        if (mask) {
            if (Traits::BITS_PER_ELEMENT_C == 8) {
                uint2 tmp;
                xmma::ldg(tmp, ptr);
                data.from_int2(tmp);
            } else if (Traits::BITS_PER_ELEMENT_C == 32) {
                uint4 tmp;
                xmma::ldg(tmp, ptr);
                data.from_int4(tmp);

                xmma::ldg(tmp, ptr + Traits::offset_in_bytes_c(4));
                data.reg(4) = tmp.x;
                data.reg(5) = tmp.y;
                data.reg(6) = tmp.z;
                data.reg(7) = tmp.w;
            } else {
                uint4 tmp;
                xmma::ldg(tmp, ptr);
                data.from_int4(tmp);
            }
        }
    }

    // The dimensions of the matrix.
    const int params_m_, params_n_, params_stride_n_;
    // The pointer to global memory.
    char *ptr_;
    const char *ptr_res_;
    // The position of the tile.
    int m_, n_;
};

///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits, 
          typename Cta_tile
> 
struct Gmem_tile_imma_epilogue_prefetch {
    // Ctor.
    template <typename Params>
    inline __device__
    Gmem_tile_imma_epilogue_prefetch( const Params& params, int bidm, int bidn, int bidz, int tidx ) { }

    inline __device__ void prefetch() { }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace fprop
} // namespace implicit_gemm
} // namespace ext
} // namespace xmma
