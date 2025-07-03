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
namespace interleaved_fprop {

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

    enum { ELTS_PER_UINT = (Cta_tile::K == 128 ? 32 : 16) };

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
        //this->ptr_ += (g * params.padded_k * (c / 2)) / 4;

        const int n = kg * Cta_tile::M + ty * Xmma_tile::M_PER_WARP;
        uint32_t offset = (g * params.padded_k * (c / 2)) / 4 + 
                          ((n / 64) * (c / ELTS_PER_UINT) * ELTS_PER_UINT * 2) * BYTES_PER_LDG + 
                          (tz * 32 + warp_id * 1)*BYTES_PER_ELEMENT;
/*        
        uint32_t effctive_offset = xmma::div_up( params.g * params.padded_k, Xmma_tile::M_PER_WARP) * 
                                   Xmma_tile::M_PER_WARP * 
                                   (( (c / 2) / ELTS_PER_UINT) * 4) * 2;
*/
        uint32_t effctive_offset = params.g * params.padded_k * 
                                   (( (c / 2) / ELTS_PER_UINT) * 4) * 2;

        if( offset >= effctive_offset ) {
            uint32_t dummy_offset = (g * params.padded_k * (c / 2)) / 4 + 
                                    (((kg * Cta_tile::M) / 64) * (c / ELTS_PER_UINT) * ELTS_PER_UINT * 2) * BYTES_PER_LDG + 
                                    warp_id * BYTES_PER_ELEMENT;
            offset = dummy_offset;
        }

        this->ptr_ += offset;
/*        
        // Same logic in NHWC, the 2 here is the MD_SCALE
        uint32_t preds[1];
        uint32_t effctive_offset = xmma::div_up( params.g * params.padded_k, Xmma_tile::M_PER_WARP) * 
                                   Xmma_tile::M_PER_WARP * 
                                   (( (c / 2) / ELTS_PER_UINT) * 4) * 2;

        preds[0] = offset < effctive_offset;
        this->preds_ = xmma::pack_predicates(preds);    
*/        
    }

    inline __device__ void move () {
        ptr_ += Cta_tile::K*4*2;
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
    typename Input_related, int VECT_SIZE_ >
struct Gmem_tile_base_b : public xmma::gemm::Gmem_tile_base<Traits,
                                                      Cta_tile,
                                                      Cta_tile::K,
                                                      Cta_tile::N,
                                                      Traits::BITS_PER_ELEMENT_B,
                                                      BYTES_PER_LDG> {

    enum { VECT_SIZE = VECT_SIZE_ };

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

    enum { FLT_T = Input_related::FLT_T };
    enum { FLT_R = Input_related::FLT_R };
    enum { FLT_S = Input_related::FLT_S };
    enum { STATIC_FILTER_SIZE = Input_related::STATIC_FILTER_SIZE };

    enum { ELTS_PER_PACKET = 32 };
    enum { THREADS_PER_PACKET = 2 };
    enum { ROWS_PER_LDG = PIXELS_PER_LDG };

    enum { STEPS = Div_up<Cta_tile::N, Cta_tile::THREADS_PER_CTA>::VALUE };

    // Ctor.
    template< typename Params >
    inline __device__ Gmem_tile_base_b(const Params &params, void *smem)
        : Base_(params, smem, params.c*params.g, params.img_gmem)
        , params_filter_trs_per_cta_( params.filter_trs_per_cta )
        , params_dhw_(params.dhw)
        , smem_(xmma::get_smem_pointer(smem)) {
    }

    // Ctor.
    template< typename Params >
    inline __device__ Gmem_tile_base_b(const Params &params, void *smem, const dim3 &bidx, int tidx)
        : Gmem_tile_base_b(params, smem) {

        int pack_c = tidx % THREADS_PER_PACKET * Base_::ELTS_PER_LDG;
        int row    = (tidx / 2) % (ROWS_PER_LDG);
        trsc_      = (tidx / 2) / (ROWS_PER_LDG);

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

        int trs = 0, rs, t, r, s, c;
        if (!IS_SIMPLE_1x1x1 && !IS_FLT_1x1x1) {
            xmma::fast_divmod( c, trs, trsc_, filter_trs_per_cta,
                params.mul_filter_trs_per_cta, params.shr_filter_trs_per_cta );

            xmma::fast_divmod( t, rs, trs, filter_rs_per_cta,
                params.mul_filter_rs_per_cta, params.shr_filter_rs_per_cta );
            xmma::fast_divmod( r, s, rs, filter_s_per_cta,
                params.mul_filter_s_per_cta, params.shr_filter_s_per_cta );
        } else {
            t = 0; r = 0; s = 0; c = trsc_;
        }
        c *= ELTS_PER_PACKET;

        int delta = t * params.dilation[0] * params.hw32c
                  + r * params.dilation[1] * params.w32c
                  + s * params.dilation[2] * 32
                  + c * params.dhw;

        int k, g;
        xmma::fast_divmod( g, k, bidx.x, params.kn, params.mul_k, params.shr_k);
        this->ptr_ += Traits::offset_in_bytes_b(g * params.c * params.dhw + pack_c + delta);

        if(IS_SIMPLE_1x1x1) {
            #pragma unroll
            for( int mi = 0; mi < Base_::LDGS; ++mi ) {
                int nopq = bidx.y * Cta_tile::N + row + mi * PIXELS_PER_LDG;
                int opq, n;
                xmma::fast_divmod(n, opq, nopq, params.opq, params.mul_opq, params.shr_opq);

                masks_[mi] = ( nopq < params.nopq && c < params.c) ? uint32_t( -1 ) : 0u;;
                this->offsets_[mi] = n * params.dhwc + opq * ELTS_PER_PACKET;
            }
        } else {

            //const int STEPS = Div_up<Cta_tile::N, Cta_tile::THREADS_PER_CTA>::VALUE;
            // The index of the element loaded by this thread. That's the column.
            #pragma unroll
            for( int ii = 0; ii < STEPS; ++ii ) {

                int nopq = bidx.y * Cta_tile::N + tidx + ii * Cta_tile::THREADS_PER_CTA;
                // Decompose nopq into n and opq.
                int opq, pq, n, o, p, q;
                xmma::fast_divmod(n, opq, nopq, params.opq, params.mul_opq, params.shr_opq);
                xmma::fast_divmod(o, pq, opq, params.pq, params.mul_pq, params.shr_pq);
                xmma::fast_divmod(p, q, pq, params.q, params.mul_q, params.shr_q);

                int d = o*params.stride[0] - params.pad[0][0];
                int h = p*params.stride[1] - params.pad[1][0];
                int w = q*params.stride[2] - params.pad[2][0];

                // The masks for the predicates.
                const uint32_t MASK_T = xmma::ext::implicit_gemm::Build_mask_t<FLT_T, FLT_R, FLT_S>::VALUE;
                const uint32_t MASK_R = xmma::ext::implicit_gemm::Build_mask_r<FLT_T, FLT_R, FLT_S>::VALUE;
                const uint32_t MASK_S = xmma::ext::implicit_gemm::Build_mask_s<FLT_T, FLT_R, FLT_S>::VALUE;

                uint32_t mask = uint32_t( -1 );
                int offset = n * params.dhwc + d * params.hw32c + h * params.w32c +
                    w * ELTS_PER_PACKET;

                if (USE_PREDICATES) {
                    mask = ( nopq < params.nopq ) ? uint32_t( -1 ) : 0u;

                    // Compute the masks for T.
                    #pragma unroll
                    for( int ti = 0; ti < filter_t_per_cta; ++ti ) {
                        uint32_t mask_t;
                        if( STATIC_FILTER_SIZE ) {
                            mask_t = ( MASK_T << ( ti * FLT_R * FLT_S ) );
                        } else {
                            mask_t = ( params.mask_t << ( ti * params.filter_rs_per_cta ) );
                        }
                        mask_t ^= uint32_t( -1 );
                        if( (unsigned)( d + ti * params.dilation[0] ) >= params.d ) {
                            mask = mask & mask_t;
                        }
                    }

                    // Compute the masks for R.
                    #pragma unroll
                    for( int ri = 0; ri < filter_r_per_cta; ++ri ) {
                        uint32_t mask_r;
                        if( STATIC_FILTER_SIZE ) {
                            mask_r = ( MASK_R << ( ri * FLT_S ) );
                        } else {
                            mask_r = ( params.mask_r << ( ri * params.filter_s_per_cta ) );
                        }
                        mask_r ^= uint32_t( -1 );
                        if( (unsigned)( h + ri * params.dilation[1] ) >= params.h ) {
                            mask = mask & mask_r;
                        }
                    }

                    // Compute the masks for S.
                    #pragma unroll
                    for( int si = 0; si < filter_s_per_cta; ++si ) {
                        uint32_t mask_s;
                        if( STATIC_FILTER_SIZE ) {
                            mask_s = ( MASK_S << si );
                        } else {
                            mask_s = ( params.mask_s << si );
                        }
                        mask_s ^= uint32_t( -1 );
                        if( (unsigned)( w + si * params.dilation[2] ) >= params.w ) {
                            mask = mask & mask_s;
                        }
                    }
                }

                uint2 tmp;
                tmp.x = mask;
                tmp.y = offset;
                xmma::sts( smem_ + (tidx + ii * Cta_tile::THREADS_PER_CTA) * sizeof( uint2 ), tmp );
            }
            __syncthreads();

            #pragma unroll
            for( int mi = 0; mi < Base_::LDGS; ++mi ) {
                int idx = row + mi * PIXELS_PER_LDG;
                uint2 p;
                xmma::lds( p, smem_ + idx * sizeof( uint2 ) );

                if (c >= params.c) {
                    masks_[mi]   = 0u;
                } else {
                    masks_[mi]   = p.x;
                }
                this->offsets_[mi] = p.y;
            }
        }

        // Pack the predicates.
        if (USE_PREDICATES) {
            xmma::ext::implicit_gemm::pack_predicates( this->preds_, masks_, 1u << trs );

            // Precompute the masks for the residue.
            int c = params.loop_residue_k + trsc_ * ELTS_PER_PACKET;
            residue_mask_ = 0u;
            if( c < params.c * filter_trs_per_cta ) {
               residue_mask_ = uint32_t( -1 );
            }
        } else {
            this->preds_[0] = 0xffffffffu;
        }
    }

    // Move the pointers and update the predicates for R2P/P2R.
    template< typename Params >
    inline __device__ void move_b(int next_trsi, int64_t delta, const Params &params) {
        // Update the pointer.
        this->ptr_ += delta;

        // Update the predicates and store them in a register using P2R.
        int filter_trs_per_cta;
        if( STATIC_FILTER_SIZE ) {
            filter_trs_per_cta = FLT_T * FLT_R * FLT_S;
        } else {
            filter_trs_per_cta = params_filter_trs_per_cta_;
        }
        if( filter_trs_per_cta > 1 && USE_PREDICATES) {
            xmma::ext::implicit_gemm::pack_predicates( this->preds_, masks_, 1u << next_trsi );
        }

    }

    // The residue to "fix" the predicates.
    inline __device__ void residue() {
        if (USE_PREDICATES) {
            this->preds_[0] &= this->residue_mask_;
        }
    }

    const int params_filter_trs_per_cta_;
    // The masks.
    uint32_t smem_;
    int trsc_;
    int params_dhw_;
    uint32_t masks_[Base_::LDGS], residue_mask_;
};

///////////////////////////////////////////////////////////////////////////////////////////////////

template<
    // The instruction traits.
    typename Traits,
    // The CTA tile description.
    typename Cta_tile,
    // Input_related
    typename Input_related,
    // VECTOR SIZE (if interleave)
    int VECT_SIZE,
    // Use or not predicates
    bool USE_PREDICATES = true,
    // The size in bytes of the LDGs.
    int BYTES_PER_LDG = 16,
    // Do we disable LDGSTS even on an architecture that has it?
    bool DISABLE_LDGSTS = false,
    // The ancestor/base class. See docs/gmem_tile.md.
    typename Ancestor = Gmem_tile_base_b<Traits, Cta_tile, BYTES_PER_LDG, USE_PREDICATES,
        Input_related, VECT_SIZE>
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
    const int params_delta_k_;
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
    int VECT_SIZE_,
    int USE_PREDICATES,
    typename Fragment_c = xmma::Fragment_c<Traits, Cta_tile>
>
struct Gmem_tile_epilogue
    : public xmma::helpers::Gmem_tile_epilogue<Traits, Cta_tile, xmma::Row, Fragment_c> {

    enum { STGS = 2 };

//    enum { VECT_SIZE = VECT_SIZE_ };
    enum { VECT_SIZE = 32 };

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
            params_opq_(params.opq),
            params_mul_opq_(params.mul_opq),
            params_shr_opq_(params.shr_opq),
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
        if (VECT_SIZE == 1) {
            ptr_ = &ptr[Traits::offset_in_bytes_c(m_*params_stride_n_ + n_ + g * params.k * params_opq_)];
            ptr_res_ = &ptr_res[Traits::offset_in_bytes_c(m_*params_stride_n_ + n_ + g * params.k * params_opq_)];
        } else {
        k = (n_ / VECT_SIZE) * params.opq * VECT_SIZE + (n_ % VECT_SIZE);
        uint32_t offset = k + g * params.k * params_opq_;
        ptr_ = &ptr[Traits::offset_in_bytes_c(static_cast<int64_t>(offset))];
        ptr_res_ = &ptr_res[Traits::offset_in_bytes_c(static_cast<int64_t>(offset))];
        }
    }

    // Compute the row offset.
    inline __device__ int compute_offset(int mi, int i) {
        if (VECT_SIZE == 1) {
            return (mi % 2) * Xmma_tile::N_PER_XMMA / 2
                + (mi / 2) * Xmma_tile::N_PER_XMMA_PER_CTA + i;
        } else {
        int nopq = m_ + (mi % 2) * Xmma_tile::N_PER_XMMA / 2
         + (mi / 2) * Xmma_tile::N_PER_XMMA_PER_CTA + i;
        int opq, n;
        xmma::fast_divmod(n, opq, nopq, params_opq_, params_mul_opq_, params_shr_opq_);

        return n * params_opq_ * params_stride_n_ + opq * VECT_SIZE;
        }
    }

    // Is a given output valid?
    inline __device__ int compute_output_mask(const int mi, int i) const {
            int nopq = m_ + (mi % 2) * Xmma_tile::N_PER_XMMA / 2
                 + (mi / 2) * Xmma_tile::N_PER_XMMA_PER_CTA + i;

            return nopq < params_m_ && n_ < params_n_;;
    }

    // Store the data to global memory.
    inline __device__ void store(int mi, int ii, const Fragment_c data) {

        int offset = compute_offset(mi, ii);
        char *ptr = (VECT_SIZE == 1)
            ? &ptr_[Traits::offset_in_bytes_c(offset*params_stride_n_)]
            : &ptr_[Traits::offset_in_bytes_c(offset)];

        int mask;
        if (USE_PREDICATES) {
            mask = compute_output_mask(mi, ii);
        } else {
            mask = 1;
        }

        if (mask) {
            if (Traits::BITS_PER_ELEMENT_C == 8) {
                xmma::stg(ptr, data.to_int2());
            } else {
                xmma::stg(ptr, data.to_int4());
            }
        }
    }

    // Load the data from global memory.
    inline __device__ void load(Fragment_c &data, int mi, int i) {

        int offset = compute_offset(mi, i);
        const char *ptr = (VECT_SIZE == 1)
            ? &ptr_res_[Traits::offset_in_bytes_c(offset*params_stride_n_)]
            : &ptr_res_[Traits::offset_in_bytes_c(offset)];

        int mask = compute_output_mask(mi, i);

        if (mask) {
            if (Traits::BITS_PER_ELEMENT_C == 8) {
                uint2 tmp;
                xmma::ldg(tmp, ptr);
                data.from_int2(tmp);
            } else {
                uint4 tmp;
                xmma::ldg(tmp, ptr);
                data.from_int4(tmp);
            }
        }
    }

    // The dimensions of the matrix.
    const int params_m_, params_n_, params_stride_n_;
    const int params_opq_, params_mul_opq_, params_shr_opq_;
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

    // The XMMA tile.
    using Xmma_tile = typename Traits::Xmma_tile<Cta_tile>;
    // The fragment.
    using Fragment_c = xmma::Fragment_interleaved_c<Traits, Cta_tile>;
    // The number of elements per packet.
    enum { ELEMENTS_PER_PACKET = 32 };
    // The size of a packet of interleaved elements.
    enum { BYTES_PER_PACKET = ELEMENTS_PER_PACKET * Traits::BITS_PER_ELEMENT_C / 8 };
    // The number of packets per CTA in the N dimension.
    enum { PACKETS_PER_M = Cta_tile::M / ELEMENTS_PER_PACKET };
    // Cacheline prefetch
    enum { CACHE_LINE_SIZE = 128 };
    enum { PACKETS_PER_THREAD = 128 / BYTES_PER_PACKET };
    enum {
        CACHE_LINE_NUM =
            Cta_tile::M * Cta_tile::N * ( Traits::BITS_PER_ELEMENT_C / 8 ) / CACHE_LINE_SIZE
    };
    enum { CACHE_LINE_NUM_PER_THREAD = Div_up<CACHE_LINE_NUM, Cta_tile::THREADS_PER_CTA>::VALUE };
    enum { COLS = Cta_tile::M / ELEMENTS_PER_PACKET };
    enum { COLS_PER_PREFETCH = COLS / CACHE_LINE_NUM_PER_THREAD };
    enum { THREADS_PER_COL = Cta_tile::N / PACKETS_PER_THREAD };

    // Ctor.
    template <typename Params>
    inline __device__
    Gmem_tile_imma_epilogue_prefetch( const Params& params, int bidm, int bidn, int bidz, int tidx )
        : params_res_ptr_( reinterpret_cast<const char*>( params.res_gmem ) ) {

        int nopq = (bidn * Cta_tile::N) + ( tidx % THREADS_PER_COL ) * PACKETS_PER_THREAD;
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
            int k = bidm * PACKETS_PER_M + tidx / THREADS_PER_COL + i * COLS_PER_PREFETCH;
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

///////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace interleaved_fprop
} // namespace implicit_gemm
} // namespace ext
} // namespace xmma
