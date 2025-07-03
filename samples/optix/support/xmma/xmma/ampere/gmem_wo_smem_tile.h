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

#include <xmma/ampere/fragment.h>
#include <xmma/warp_masks.h>

namespace xmma {

template<
    // The instruction traits.
    typename Traits_,
    // The CTA descriptor.
    typename Cta_tile_,
    // The M dimension of the tile (depends on A/B and whether it is transposed or not).
    int M_,
    // The N dimension of the tile (depends on A/B and whether it is transposed or not).
    int N_,
    // The size if bits of each element.
    int BITS_PER_ELT_,
    // The size in bytes of the LDG.
    int BYTES_PER_LDG_,
    // 1 if it is A and 0 if it is B
    bool IS_A_
>
struct Gmem_wo_smem_tile_base {

    // The traits class.
    using Traits = Traits_;
    // The CTA tile descriptor.
    using Cta_tile = Cta_tile_;
    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;

    // The dimensions of the tile.
    enum { M = M_, N = N_ };
    // The size in bits of each element.
    enum { BITS_PER_ELT = BITS_PER_ELT_ };
    // The size in bytes of each element.
    enum { BYTES_PER_ELT = BITS_PER_ELT / 8 };
    // The size in bytes of each LDG.
    enum { BYTES_PER_LDG = BYTES_PER_LDG_ };
    // 1 if it is A and 0 if it is B
    enum { IS_A = IS_A_ };

    // The number of elements per LDG.
    enum { ELTS_PER_LDG = 2 };

    // The total number of LDGs.
    enum { LDGS = IS_A ? Xmma_tile::XMMAS_M : Xmma_tile::XMMAS_N };

    enum { WARPS_M = Cta_tile::WARPS_M };
    enum { WARPS_N = Cta_tile::WARPS_N };
    enum { WARPS_K = Cta_tile::WARPS_K };

    enum { WARP_MASK = IS_A
        ? xmma::Warp_masks<WARPS_M, WARPS_N, WARPS_K>::M
        : xmma::Warp_masks<WARPS_M, WARPS_N, WARPS_K>::N };

    enum { WARP_DIV = IS_A
        ?       1 *       1 * Cta_tile::THREADS_PER_WARP
        : WARPS_M *       1 * Cta_tile::THREADS_PER_WARP };

    enum { ROWS_PER_XMMA = IS_A
        ? Xmma_tile::M_PER_XMMA
        : Xmma_tile::N_PER_XMMA };

    enum { ROWS_PER_XMMA_PER_CTA = IS_A
        ? Xmma_tile::M_PER_XMMA_PER_CTA
        : Xmma_tile::N_PER_XMMA_PER_CTA };

    enum { PRED_REGS = 2 };

    // Ctor.
    template< typename Params >
    inline __device__ Gmem_wo_smem_tile_base(const Params &params, void *, int k, const void *ptr)
        : params_k_(k)
        , params_residue_k_(params.loop_residue_k)
        , ptr_(reinterpret_cast<const char*>(ptr)) {
    }

    // Ctor.
    template< typename Params >
    inline __device__ Gmem_wo_smem_tile_base(const Params &params,
                                     int m,
                                     int n,
                                     int stride_m,
                                     int stride_n,
                                     int64_t delta,
                                     const void *ptr,
                                     int bidm,
                                     int bidn,
                                     int tidx)
        : Gmem_wo_smem_tile_base(params, nullptr, params.k, ptr) {

        stride_n_ = stride_n;
        delta_ = delta;

        // The location of the tile.
        int row = bidm * M
            + ((tidx & WARP_MASK) / WARP_DIV) * ROWS_PER_XMMA
            + (tidx % 32) / 4;
        int col = (tidx % 4) * ELTS_PER_LDG;

        params_residue_k_ += col;

        uint32_t preds[LDGS];
        uint32_t preds1[LDGS]; 
        uint32_t preds_residue[LDGS];
        uint32_t preds_residue1[LDGS];
        #pragma unroll
        for (int ii = 0; ii < LDGS; ii++) {
            offsets_[ii] = ii * ROWS_PER_XMMA_PER_CTA * stride_m;
            preds[ii] =
                row + ii * ROWS_PER_XMMA_PER_CTA < m && col < n;
            preds1[ii] =
                row + ii * ROWS_PER_XMMA_PER_CTA < m && col + 1 < n;

            preds_residue[ii] =
                row + ii * ROWS_PER_XMMA_PER_CTA < m && params_residue_k_ < n;
            preds_residue1[ii] =
                row + ii * ROWS_PER_XMMA_PER_CTA < m && params_residue_k_ + 1  < n;
        }
        preds_[0] = xmma::pack_predicates(preds);
        preds_[1] = xmma::pack_predicates(preds1);
        preds_residue_[0] = xmma::pack_predicates(preds_residue);
        preds_residue_[1] = xmma::pack_predicates(preds_residue1);
        ptr_ += Traits::offset_in_bytes_b(col * stride_n + row * stride_m);
    }

    template < typename Fragment, int K >
    inline __device__ void load (
        Fragment (&a)[K][LDGS], int ki) {

        uint4 tmp[LDGS];
        uint2 tmp_2[2][LDGS];

        for (int jj = 0; jj < 16 / BYTES_PER_LDG; jj++) {

            const void *ptrs[LDGS];
            #pragma unroll
            for (int ii = 0; ii < LDGS; ii++) {
                ptrs[ii] = ptr_ + Traits::offset_in_bytes_a(offsets_[ii]
                    + (ki * 2 * Xmma_tile::K_PER_XMMA + jj) * stride_n_);
            }

            if (BYTES_PER_LDG == 16) {
                xmma::ldg(tmp, ptrs, preds_[jj]);
            } else if (BYTES_PER_LDG == 8) {
                xmma::ldg(tmp_2[jj], ptrs, preds_[jj]);
            } else {
                assert(false);
            }
        }

        #pragma unroll
        for (int ii = 0; ii < LDGS; ii++) {
            if (BYTES_PER_LDG == 16) {
                a[2*ki][ii].reg(0) = tmp[ii].x;
                a[2*ki][ii].reg(1) = tmp[ii].y;
                a[2*ki+1][ii].reg(0) = tmp[ii].z;
                a[2*ki+1][ii].reg(1) = tmp[ii].w;
            } else if (BYTES_PER_LDG == 8) {
                a[2*ki][ii].reg(0) = tmp_2[0][ii].x;
                a[2*ki][ii].reg(1) = tmp_2[0][ii].y;
                a[2*ki+1][ii].reg(0) = tmp_2[1][ii].x;
                a[2*ki+1][ii].reg(1) = tmp_2[1][ii].y;
            } else {
                assert(false);
            }
        }
    }


    inline __device__ void residue() {
        preds_[0] = preds_residue_[0];
        preds_[1] = preds_residue_[1];
    }

    inline __device__ void residue(int ki) {
        uint32_t preds[LDGS];
        uint32_t preds1[LDGS];
        #pragma unroll
        for (int ii = 0; ii < LDGS; ii++) {
            preds[ii] = params_residue_k_
                + ki * 2 * Xmma_tile::K_PER_XMMA < params_k_;
            preds1[ii] = params_residue_k_ 
                + ki * 2 * Xmma_tile::K_PER_XMMA + 1 < params_k_;
        }
        preds_[0] &= xmma::pack_predicates(preds);
        preds_[1] &= xmma::pack_predicates(preds1);
    }

    // Move the pointers and update the predicates for R2P/P2R (if needed).
    inline __device__ void move() {
        ptr_ += delta_;
    }

    // The K dimension.
    int params_k_, params_residue_k_;
    // stride_k_
    int stride_n_;
    // Delta
    int64_t delta_;
    // The pointer.
    const char *ptr_;
    // The associated offsets.
    int offsets_[LDGS];
    // The predicates.
    uint32_t preds_[PRED_REGS];
    uint32_t preds_residue_[PRED_REGS];

};


template<
    // The instruction traits.
    typename Traits,
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The layout of the tile.
    typename Layout,
    // The size of the LDG.
    int BYTES_PER_LDG = 16
>
struct Gmem_wo_smem_tile_a
    : public Gmem_wo_smem_tile_base<Traits,
                                    Cta_tile,
                                    Cta_tile::M,
                                    Cta_tile::K,
                                    64,
                                    8,
                                    1>
    {

    using Gmem_layout = Layout;

    // Base class
    using Base = Gmem_wo_smem_tile_base<Traits,
                                    Cta_tile,
                                    Cta_tile::M,
                                    Cta_tile::K,
                                    64,
                                    8,
                                    1>;

    // Ctor.
    template< typename Params >
    inline __device__ Gmem_wo_smem_tile_a(const Params &params, void *smem,
        const dim3 &bidx, int tidx)
        : Base(params,
               1,
               1,
               1,
               1,
               1,
               NULL,
               bidx.x,
               bidx.z,
               tidx) {
    }

    template < typename Fragment, int K >
    inline __device__ void load (
        Fragment (&a)[K][Base::LDGS], int ki) {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    // The instruction traits.
    typename Traits,
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The layout of the tile.
    typename Layout,
    // The size of the LDG.
    int BYTES_PER_LDG = 16
>
struct Gmem_wo_smem_tile_b
    : public Gmem_wo_smem_tile_base<Traits,
                                    Cta_tile,
                                    Cta_tile::N,
                                    Cta_tile::K,
                                    64,
                                    8,
                                    0>
    {

    using Gmem_layout = Layout;

    // Base class
    using Base = Gmem_wo_smem_tile_base<Traits,
                                    Cta_tile,
                                    Cta_tile::N,
                                    Cta_tile::K,
                                    64,
                                    8,
                                    0>;

    // Ctor.
    template< typename Params >
    inline __device__ Gmem_wo_smem_tile_b(const Params &params, void *smem,
        const dim3 &bidx, int tidx)
        : Base(params,
               1,
               1,
               1,
               1,
               1,
               NULL,
               bidx.y,
               bidx.z,
               tidx) {
    }

    template < typename Fragment, int K >
    inline __device__ void load (
        Fragment (&a)[K][Base::LDGS], int ki) {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// D M M A . F 6 4
//
////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The size of the LDG.
    int BYTES_PER_LDG
>
struct Gmem_wo_smem_tile_a<Ampere_dmma_fp64_traits, Cta_tile, Col, BYTES_PER_LDG>
    : public Gmem_wo_smem_tile_base<Ampere_dmma_fp64_traits,
                                    Cta_tile,
                                    Cta_tile::M,
                                    Cta_tile::K,
                                    64,
                                    8,
                                    1>
    {

    using Gmem_layout = Col;

    // Base class
    using Base = Gmem_wo_smem_tile_base<Ampere_dmma_fp64_traits,
                                    Cta_tile,
                                    Cta_tile::M,
                                    Cta_tile::K,
                                    64,
                                    8,
                                    1>;
    // The traits class.
    using Traits = Ampere_dmma_fp64_traits;

    // Ctor.
    template <typename Params>
    inline __device__
    Gmem_wo_smem_tile_a( const Params &params, void *smem, const dim3 &bidx, int tidx )
        : Base( params,
                params.m,
                params.k,
                1,
                params.lda,
                params.a_delta[0],
                params.a_gmem,
                bidx.x,
                bidx.z,
                tidx ) {
        if( params.batch.is_batched ) {
            this->ptr_ +=
                Traits::offset_in_bytes_a( static_cast<int64_t>( bidx.z ) *
                                           static_cast<int64_t>( params.a_stride_batches ) );
        }
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////


template<
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The size of the LDG.
    int BYTES_PER_LDG
>
struct Gmem_wo_smem_tile_a<Ampere_dmma_fp64_traits, Cta_tile, Row, BYTES_PER_LDG>
    : public Gmem_wo_smem_tile_base<Ampere_dmma_fp64_traits,
                                    Cta_tile,
                                    Cta_tile::M,
                                    Cta_tile::K,
                                    64,
                                    16,
                                    1>
    {

    using Gmem_layout = Row;

    // Base class
    using Base = Gmem_wo_smem_tile_base<Ampere_dmma_fp64_traits,
                                    Cta_tile,
                                    Cta_tile::M,
                                    Cta_tile::K,
                                    64,
                                    16,
                                    1>;
    // The traits class.
    using Traits = Ampere_dmma_fp64_traits;

    // Ctor.
    template <typename Params>
    inline __device__
    Gmem_wo_smem_tile_a( const Params &params, void *smem, const dim3 &bidx, int tidx )
        : Base( params,
                params.m,
                params.k,
                params.lda,
                1,
                Traits::offset_in_bytes_a( Cta_tile::K ),
                params.a_gmem,
                bidx.x,
                bidx.z,
                tidx ) {
        if( params.batch.is_batched ) {
            this->ptr_ +=
                Traits::offset_in_bytes_a( static_cast<int64_t>( bidx.z ) *
                                           static_cast<int64_t>( params.a_stride_batches ) );
        }
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The size of the LDG.
    int BYTES_PER_LDG
>
struct Gmem_wo_smem_tile_b<Ampere_dmma_fp64_traits, Cta_tile, Col, BYTES_PER_LDG>
    : public Gmem_wo_smem_tile_base<Ampere_dmma_fp64_traits,
                                    Cta_tile,
                                    Cta_tile::N,
                                    Cta_tile::K,
                                    64,
                                    16,
                                    0>
    {

    using Gmem_layout = Col;

    // Base class
    using Base = Gmem_wo_smem_tile_base<Ampere_dmma_fp64_traits,
                                    Cta_tile,
                                    Cta_tile::N,
                                    Cta_tile::K,
                                    64,
                                    16,
                                    0>;
    // The traits class.
    using Traits = Ampere_dmma_fp64_traits;

    // Ctor.
    template <typename Params>
    inline __device__
    Gmem_wo_smem_tile_b( const Params &params, void *smem, const dim3 &bidx, int tidx )
        : Base( params,
                params.n,
                params.k,
                params.ldb,
                1,
                Traits::offset_in_bytes_b( Cta_tile::K ),
                params.b_gmem,
                bidx.y,
                bidx.z,
                tidx ) {
        if( params.batch.is_batched ) {
            this->ptr_ +=
                Traits::offset_in_bytes_b( static_cast<int64_t>( bidx.z ) *
                                           static_cast<int64_t>( params.b_stride_batches ) );
        }
    }
};

template<
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The size of the LDG.
    int BYTES_PER_LDG
>
struct Gmem_wo_smem_tile_b<Ampere_dmma_fp64_traits, Cta_tile, Row, BYTES_PER_LDG>
    : public Gmem_wo_smem_tile_base<Ampere_dmma_fp64_traits,
                                    Cta_tile,
                                    Cta_tile::N,
                                    Cta_tile::K,
                                    64,
                                    8,
                                    0>
    {

    using Gmem_layout = Row;

    // Base class
    using Base = Gmem_wo_smem_tile_base<Ampere_dmma_fp64_traits,
                                    Cta_tile,
                                    Cta_tile::N,
                                    Cta_tile::K,
                                    64,
                                    8,
                                    0>;
    // The traits class.
    using Traits = Ampere_dmma_fp64_traits;

    // Ctor.
    template <typename Params>
    inline __device__
    Gmem_wo_smem_tile_b( const Params &params, void *smem, const dim3 &bidx, int tidx )
        : Base( params,
                params.n,
                params.k,
                1,
                params.ldb,
                params.b_delta[0],
                params.b_gmem,
                bidx.y,
                bidx.z,
                tidx ) {
        if( params.batch.is_batched ) {
            this->ptr_ += Traits::offset_in_bytes_b(
                static_cast<int64_t>( bidx.z ) * static_cast<int64_t>( params.b_stride_batches ) );
        }
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace xmma

