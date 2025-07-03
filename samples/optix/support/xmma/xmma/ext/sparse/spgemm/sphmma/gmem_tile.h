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
#include <xmma/ext/sparse/utils.h>
#include <xmma/ext/sparse/helpers/epilogue.h>

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace xmma {
namespace ext {
namespace gemm {
namespace sparse_hmma_gemm {

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Traits, typename Cta_tile, int M, int N, int BITS_PER_ELEMENT >
struct Gemm_gmem_meta_tile_base {

    // Number of LDGSTS32 needed
    enum { LDGSTS32_NEEDED = ((Cta_tile::M * Cta_tile::HALF_K / Traits::ELEMENTS_PER_UINT16 )
                            * BITS_PER_ELEMENT) / 32 };
    // Number of LDGSTS64 needed
    enum { LDGSTS64_NEEDED = ((Cta_tile::M * Cta_tile::HALF_K / Traits::ELEMENTS_PER_UINT16 )
                            * BITS_PER_ELEMENT) / 64 };
    // LDGS by LDGSTS64
    enum { LDGS_by_LDGSTS32 = (LDGSTS32_NEEDED / Cta_tile::THREADS_PER_CTA) };
    // LDGS by LDGSTS32
    enum { LDGS_by_LDGSTS64 = (LDGSTS64_NEEDED / Cta_tile::THREADS_PER_CTA) };
    // Use LDGSTS64 or not
    enum { SEL_64 = ((LDGS_by_LDGSTS64 < LDGS_by_LDGSTS32) && LDGS_by_LDGSTS64 > 0) ? 1 : 0 };
    // The number of steps needed to load the metedata.
    enum { LDGS = (SEL_64) ? LDGS_by_LDGSTS64 : LDGS_by_LDGSTS32 };
    // Make sure we have a "nice" number of LDGs.
    static_assert(LDGS > 0, "");
    // Selected LDGSTS bytes
    enum { LDGSTS_BYTES = (SEL_64) ? 8 : 4 };

    //enum { OFFSET_STRIDE = LDGS };
    // Ctor.
    template< typename Params >
    inline __device__ Gemm_gmem_meta_tile_base(const Params &params, const void *ptr)
        : ptr_(reinterpret_cast<const char*>(ptr)) {
    }

    // Store the pixels to shared memory.
    template< typename Xmma_smem_tile >
    inline __device__ void commit(Xmma_smem_tile &smem) {
    }

    // Disable the loads.
    inline __device__ void disable_loads() {
        preds_ = 0u;
    }

    // Move the pointers and update the predicates for R2P/P2R.
    inline __device__ void move(int64_t delta) {
        ptr_ += delta;
    }

    // Restore predicate
    inline __device__ void restore_predicate() {
        preds_ = pre_store_preds_;
    }

    // The pointer.
    const char *ptr_;
    // The associated offsets.
    int offsets_[LDGS];
    // The predicates.
    uint32_t preds_;
    // The pre-stroed predicates.
    uint32_t pre_store_preds_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Traits, typename Cta_tile, int M, int N >
struct Gemm_gmem_tile_e
    : public Gemm_gmem_meta_tile_base<Traits, Cta_tile, M, N, Traits::BITS_PER_ELEMENT_E> {

    // The base class.
    using Base = Gemm_gmem_meta_tile_base<Traits, Cta_tile, M, N, Traits::BITS_PER_ELEMENT_E>;

    // Ctor.
    template< typename Params >
    inline __device__ Gemm_gmem_tile_e(const Params &params)
        : Base(params, params.e_gmem) {
    }

    // Load a tile from global memory.
    template< typename Xmma_smem_tile >
    inline __device__ void load(Xmma_smem_tile &smem) {
        const char *ptrs[Base::LDGS];
        #pragma unroll
        for( int ii = 0; ii < Base::LDGS; ++ii ) {
            ptrs[ii] = this->ptr_ + this->offsets_[ii];
        }
        smem.store_e(ptrs, this->preds_, Base::SEL_64);
    }

    inline __device__ void disable() {
        this->preds_ = 0u;
    }

};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    typename Traits,
    typename Cta_tile,
    typename mma_type
>
struct Gemm_gmem_tile_e_linear {

};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    typename Traits,
    typename Cta_tile
>
struct Gemm_gmem_tile_e_linear_32b
    : public Gemm_gmem_tile_e<Traits, Cta_tile, 4, Cta_tile::M> {

    // The base class.
    using Base = Gemm_gmem_tile_e<Traits, Cta_tile, 4, Cta_tile::M>;

    // The expected shared memory layout.
    using Smem_layout = xmma::Row;

    // Ctor.
    template< typename Params >
    inline __device__ Gemm_gmem_tile_e_linear_32b(const Params &params, int bidm, int bidz, int tidx)
        : Base(params) {

        // The number of warps.
        const int WARPS_M = Cta_tile::WARPS_M;
        const int WARPS_N = Cta_tile::WARPS_N;
        const int WARPS_K = Cta_tile::WARPS_K;
        // The masks to select the warps.
        const int WARP_MASK_N = xmma::Warp_masks<WARPS_M, WARPS_N, WARPS_K>::N;
        // The divisor for the warps.
        const int WARP_DIV_N = WARPS_M *       1 * Cta_tile::THREADS_PER_WARP;

        const int warp_n = (tidx & WARP_MASK_N) / WARP_DIV_N;

        // M="32" is to support reorderin algo
        int pad_m = (params.m % 32 != 0) ? ((params.m / 32) * 32 + 32) : params.m;

        // Offset for inter-cta split-k
        const int inter_split_offset = params.batch.is_batched ? 0 : (bidz *
            Traits::offset_in_bytes_e(pad_m * Cta_tile::HALF_K / (Traits::ELEMENTS_PER_UINT16 / 2)));
        // Offset across block
        const int stride_offset = bidm *
            Traits::offset_in_bytes_e(Cta_tile::M * Cta_tile::HALF_K / (Traits::ELEMENTS_PER_UINT16 / 2));
        // Address of specific batch
        if( params.batch.is_batched ) {
            // (k / 2) / (Traits::ELEMENTS_PER_UINT16 / 2)
            int pad_col = xmma::div_up(params.k, Traits::ELEMENTS_PER_UINT16);
            pad_col = (pad_col % 4 != 0) ? (pad_col / 4 * 4 + 4) : pad_col;
            this->ptr_ += Traits::offset_in_bytes_e(static_cast<int64_t>(bidz) * pad_m * pad_col);
        }

        // The goal it to split the meatadata tile into 2 parts.
        // 1st k=2(32) is loaded by half threads
        // 2nd k=2(32) is loaded by another half threads
        // Split the warps to half, responsible for each M * 2 (4 -- kBlock = 64, 2 -- half kBlock)

        // Colwert the thread id to effective row
        int row_effective = 0;
        if(Base::SEL_64)
            row_effective = (tidx * Base::LDGS * 2) % Cta_tile::M;
        else
            row_effective = (tidx * Base::LDGS) % Cta_tile::M;

        // Dertermine if the CTA/Block is the last block at M dimension
        int grid_dim = 0;
        if(params.use_horizontal_cta_rasterization) {
            grid_dim = gridDim.y;
        } else {
            grid_dim = gridDim.x;
        }
        bool is_residue = ((grid_dim - 1) == bidm) ? true : false;

        // Compute amount of bytes in current CTA
        int in_cta_data_bytes = (is_residue) ? ( (pad_m - Cta_tile::M * bidm) * 4 * 2) :
            Traits::offset_in_bytes_e(Cta_tile::M * Cta_tile::HALF_K / (Traits::ELEMENTS_PER_UINT16 / 2));

        // Divide the in-cta-byte by 2 to determine the gmem offset for valid warp
        int half_bytes = in_cta_data_bytes / 2;
        if(Cta_tile::M == 128) {
            // GMEM offset for each thread
            this->offsets_[0] = inter_split_offset + stride_offset + half_bytes * warp_n + 
                                (tidx % WARP_DIV_N) * Base::LDGSTS_BYTES * Base::LDGS;
        } else {
            // Check this https://confluence.lwpu.com/display/~yujungc/Metadata+Addressing+Logic+WAR
            int warp_idx = tidx / Cta_tile::THREADS_PER_WARP;
            int sec_set = warp_idx / 4;
            // int cta_1st = ((warp_idx & 0x3) < 2) ? 1 : 0 ;
            // int cta_2nd = (warp_idx & 0x3) / 2 ;
            
            // cta_set_1 means w0/w1/w4/w5; cta_set_2 means w0/w1/w4/w5 in original warp distribution
            int cta_set_1 = 0;
            int cta_set_2 = 0;

            if( ((warp_idx & 0x3) / 2) == 0) {
                cta_set_1 = 1;
                cta_set_2 = 0;
            } else {
                cta_set_1 = 0;
                cta_set_2 = 1;
            }

            // Translate to new thread id distribution, w0145 -- tid0 ~ tid127; w2367 -- tid128 ~ tid255
            int new_tid = (tidx % 64) + sec_set * 64 + cta_set_2 * 128;
            
            // % 128 to limit the effective row range to 128, for cta_set_2 we need to add offset 128
            row_effective = (new_tid * Base::LDGS * 2) % 128 + cta_set_2 * 128;

            // % 128 to remap to CTA-M=128 distirbution and divide by 64 which is half CTA thread number
            const int new_warp_n = (new_tid % 128) / 64;

            if ((pad_m - Cta_tile::M * bidm) <= 128 ) {
                half_bytes = half_bytes;
            } else {
                // 512 is the half of 128 x 4 unit16 metadata size
                // (half_bytes - 512) is the residue half of the 2nd half cta
                half_bytes = cta_set_1 * 512 + cta_set_2 * (half_bytes - 512);
            }
            // cta_set_2 * 1024 -- offset to the 2nd warp
            this->offsets_[0] = inter_split_offset + stride_offset + half_bytes * new_warp_n + cta_set_2 * 1024 +
                                (new_tid % 64) * Base::LDGSTS_BYTES * Base::LDGS;
        }

        #pragma unroll
        for( int mi = 1 ; mi < Base::LDGS; ++mi ) {
            this->offsets_[mi] = this->offsets_[0] + mi * Base::LDGSTS_BYTES;
        }

        int row[Base::LDGS];
        row[0] = bidm * Cta_tile::M + row_effective;

        // Need to revisit this loop
        #pragma unroll
        for( int ii = 1 ; ii < Base::LDGS; ++ii ) {
            //Derive the effective row number
            row[ii] = row[0] + 1;
        }

        // Compute the predicates.
        uint32_t preds[Base::LDGS];
        // Need to revisit this loop
        #pragma unroll
        for( int ii = 0; ii < Base::LDGS; ++ii ) {
            preds[ii] = row[ii] < pad_m;
        }

        this->preds_ = xmma::pack_predicates(preds);

        this->pre_store_preds_ = this->preds_ ;
    }

    // The residue to "fix" the predicates.
    inline __device__ void residue() {
        // No need to perform residue since we'll pad metedata first at host
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    typename Traits,
    typename Cta_tile
>
struct Gemm_gmem_tile_e_linear <Traits, Cta_tile, float>
    : public Gemm_gmem_tile_e_linear_32b <Traits, Cta_tile> {

    // The base class.
    using Base = Gemm_gmem_tile_e_linear_32b<Traits, Cta_tile>;

    using Smem_layout = typename Base::Smem_layout;

    // Ctor.
    template< typename Params >
    inline __device__ Gemm_gmem_tile_e_linear(const Params &params, int bidm, int bidz, int tidx)
        : Base(params, bidm, bidz, tidx) {
    }
};

template<
    typename Traits,
    typename Cta_tile
>
struct Gemm_gmem_tile_e_linear <Traits, Cta_tile, uint32_t>
    : public Gemm_gmem_tile_e_linear_32b <Traits, Cta_tile> {

    // The base class.
    using Base = Gemm_gmem_tile_e_linear_32b<Traits, Cta_tile>;

    using Smem_layout = typename Base::Smem_layout;

    // Ctor.
    template< typename Params >
    inline __device__ Gemm_gmem_tile_e_linear(const Params &params, int bidm, int bidz, int tidx)
        : Base(params, bidm, bidz, tidx) {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    typename Traits,
    typename Cta_tile
>
struct Gemm_gmem_tile_e_linear <Traits, Cta_tile, uint16_t>
    : public Gemm_gmem_tile_e<Traits, Cta_tile,
                            Cta_tile::HALF_K / Traits::ELEMENTS_PER_UINT16, Cta_tile::M> {

    // The base class.
    using Base = Gemm_gmem_tile_e<Traits, Cta_tile,
                            Cta_tile::HALF_K / Traits::ELEMENTS_PER_UINT16, Cta_tile::M>;

    // The expected shared memory layout.
    using Smem_layout = xmma::Row;

    // Ctor.
    template< typename Params >
    inline __device__ Gemm_gmem_tile_e_linear(const Params &params, int bidm, int bidz, int tidx)
        : Base(params) {

        // The number of warps.
        const int WARPS_M = Cta_tile::WARPS_M;
        const int WARPS_N = Cta_tile::WARPS_N;
        const int WARPS_K = Cta_tile::WARPS_K;
        // The masks to select the warps.
        const int WARP_MASK_N = xmma::Warp_masks<WARPS_M, WARPS_N, WARPS_K>::N;
        // The divisor for the warps.
        const int WARP_DIV_N = WARPS_M *       1 * Cta_tile::THREADS_PER_WARP;

        const int warp_n = (tidx & WARP_MASK_N) / WARP_DIV_N;

        // M="32" is to support reorderin algo
        int pad_m = (params.m % 32 != 0) ? ((params.m / 32) * 32 + 32) : params.m;

        // Offset for inter-cta split-k
        const int inter_split_offset = params.batch.is_batched ? 0 : (bidz *
            Traits::offset_in_bytes_e(pad_m * Cta_tile::HALF_K / Traits::ELEMENTS_PER_UINT16));
        // Offset across block
        const int stride_offset = bidm *
            Traits::offset_in_bytes_e(Cta_tile::M * Cta_tile::HALF_K / Traits::ELEMENTS_PER_UINT16);
        // Address of specific batch
        if( params.batch.is_batched ) {
            // (k / 2) / (Traits::ELEMENTS_PER_UINT16)
            int pad_col = xmma::div_up(params.k, Traits::ELEMENTS_PER_UINT16 * 2);
            pad_col = (pad_col % 4 != 0) ? (pad_col / 4 * 4 + 4) : pad_col;
            this->ptr_ += Traits::offset_in_bytes_e(static_cast<int64_t>(bidz) * pad_m * pad_col);
        }

        // The goal it to split the meatadata tile into 2 parts.
        // 1st k=2(32) is loaded by half threads
        // 2nd k=2(32) is loaded by another half threads
        // Split the warps to half, responsible for each M * 2 (4 -- kBlock = 64, 2 -- half kBlock)

        // Colwert the thread id to effective row
        int row_effective = 0;
        if(Base::SEL_64)
            row_effective = (tidx * Base::LDGS * 2) % Cta_tile::M;
        else
            row_effective = (tidx * Base::LDGS) % Cta_tile::M;

        // Dertermine if the CTA/Block is the last block at M dimension
        int grid_dim = 0;
        if(params.use_horizontal_cta_rasterization) {
            grid_dim = gridDim.y;
        } else {
            grid_dim = gridDim.x;
        }
        bool is_residue = ((grid_dim - 1) == bidm) ? true : false;

        // Compute amount of bytes in current CTA
        int in_cta_data_bytes = (is_residue) ? ( (pad_m - Cta_tile::M * bidm) * 4 * 2) :
            Traits::offset_in_bytes_e(Cta_tile::M * Cta_tile::HALF_K / Traits::ELEMENTS_PER_UINT16);

        // Divide the in-cta-byte by 2 to determine the gmem offset for valid warp
        int half_bytes = in_cta_data_bytes / 2;
        if(Cta_tile::M == 128) {
            // GMEM offset for each thread
            this->offsets_[0] = inter_split_offset + stride_offset + half_bytes * warp_n + 
                                (tidx % WARP_DIV_N) * Base::LDGSTS_BYTES * Base::LDGS;
        } else {
            // Check this https://confluence.lwpu.com/display/~yujungc/Metadata+Addressing+Logic+WAR
            int warp_idx = tidx / Cta_tile::THREADS_PER_WARP;
            int sec_set = warp_idx / 4;
            //int cta_1st = ((warp_idx & 0x3) < 2) ? 1 : 0 ;
            //int cta_2nd = (warp_idx & 0x3) / 2 ;
            
            // cta_set_1 means w0/w1/w4/w5; cta_set_2 means w0/w1/w4/w5 in original warp distribution
            int cta_set_1 = 0;
            int cta_set_2 = 0;

            if( ((warp_idx & 0x3) / 2) == 0) {
                cta_set_1 = 1;
                cta_set_2 = 0;
            } else {
                cta_set_1 = 0;
                cta_set_2 = 1;
            }
            
            // Translate to new thread id distribution, w0145 -- tid0 ~ tid127; w2367 -- tid128 ~ tid255
            int new_tid = (tidx % 64) + sec_set * 64 + cta_set_2 * 128;
            
            // % 128 to limit the effective row range to 128, for cta_set_2 we need to add offset 128
            row_effective = (new_tid * Base::LDGS * 2) % 128 + cta_set_2 * 128;

            // % 128 to remap to CTA-M=128 distirbution and divide by 64 which is half CTA thread number
            const int new_warp_n = (new_tid % 128) / 64;

            if ((pad_m - Cta_tile::M * bidm) <= 128 ) {
                half_bytes = half_bytes;
            } else {
                // 512 is the half of 128 x 4 unit16 metadata size
                // (half_bytes - 512) is the residue half of the 2nd half cta
                half_bytes = cta_set_1 * 512 + cta_set_2 * (half_bytes - 512);
            }
            // cta_set_2 * 1024 -- offset to the 2nd warp
            this->offsets_[0] = inter_split_offset + stride_offset + half_bytes * new_warp_n + cta_set_2 * 1024 +
                                (new_tid % 64) * Base::LDGSTS_BYTES * Base::LDGS;
        }

        #pragma unroll
        for( int mi = 1 ; mi < Base::LDGS; ++mi ) {
            this->offsets_[mi] = this->offsets_[0] + mi * Base::LDGSTS_BYTES;
        }

        int row[Base::LDGS];
        row[0] = bidm * Cta_tile::M + row_effective;

        // Need to revisit this loop
        #pragma unroll
        for( int ii = 1 ; ii < Base::LDGS; ++ii ) {
            //Derive the effective row number
            row[ii] = row[0] + 1;
        }

        // Compute the predicates.
        uint32_t preds[Base::LDGS];
        // Need to revisit this loop
        #pragma unroll
        for( int ii = 0; ii < Base::LDGS; ++ii ) {
            preds[ii] = row[ii] < pad_m;
        }

        this->preds_ = xmma::pack_predicates(preds);

        this->pre_store_preds_ = this->preds_ ;
    }

    // The residue to "fix" the predicates.
    inline __device__ void residue() {
        // No need to perform residue since we'll pad metedata first at host
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Traits, typename Cta_tile, int M, int N, int BITS_PER_ELEMENT >
struct Gemm_gmem_tile_base {

    // The number of elements per LDG.128.
    enum { ELEMENTS_PER_LDG = 128 / BITS_PER_ELEMENT };
    // Make sure we have a "nice" number of elements per LDG.
    static_assert(ELEMENTS_PER_LDG > 0, "");

    // The number of threads needed to load a column. Each thread does LDG.128.
    enum { THREADS_PER_COLUMN = M / ELEMENTS_PER_LDG };
    // Make sure we have a "nice" number of pixels.
    static_assert(THREADS_PER_COLUMN > 0, "");

    // The number of columns loaded per LDG.
    enum { COLUMNS_PER_LDG = Cta_tile::THREADS_PER_CTA / THREADS_PER_COLUMN };
    // Make sure we have a "nice" number of columns.
    static_assert(N % COLUMNS_PER_LDG == 0, "");

    // The number of steps needed to load the columns.
    enum { LDGS = N / COLUMNS_PER_LDG };
    // Make sure we have a "nice" number of LDGs.
    static_assert(LDGS > 0, "");

    // The number of predicates that we store per register.
    enum { PREDS_PER_REG = 4 };

    // Ctor.
    template< typename Params >
    inline __device__ Gemm_gmem_tile_base(const Params &params, const void *ptr)
        : params_k_(params.k)
        , params_residue_k_(params.loop_residue_k)
        , params_half_k_ (params.k / 2)
        , params_residue_half_k_(params.loop_residue_k / 2)
        , is_batched_(params.batch.is_batched)
        , ptr_(reinterpret_cast<const char*>(ptr)) {
    }

    // Store the pixels to shared memory.
    template< typename Xmma_smem_tile >
    inline __device__ void commit(Xmma_smem_tile &smem) {
#if defined(__LWDA_ARCH__) && __LWDA_ARCH__ < 800
        smem.store(fetch_);
#endif
    }

    // Disable the loads.
    inline __device__ void disable_loads() {
        preds_ = 0u;
    }

    // Move the pointers and update the predicates for R2P/P2R.
    inline __device__ void move(int64_t delta) {
        ptr_ += delta;
    }

    // The residue to "fix" the predicates.
    inline __device__ void residue_a_n_b_t() {
        // The coordinates -- use inline PTX to avoid LWVM's rematerialization.
        int bidz;
        asm volatile("mov.b32 %0, %%ctaid.z;" : "=r"(bidz));
        int tidx;
        asm volatile("mov.b32 %0, %%tid.x;" : "=r"(tidx));

        // The position in the K dimension.
        int k[LDGS];
        k[0] = this->params_residue_k_ + (this->is_batched_ ? 0 : bidz*Cta_tile::K) + tidx / THREADS_PER_COLUMN;
        #pragma unroll
        for( int ki = 1; ki < LDGS; ++ki ) {
            k[ki] = k[0] + ki*COLUMNS_PER_LDG;
        }

        // The predicates.
        uint32_t preds[LDGS];
        #pragma unroll
        for( int ki = 0; ki < LDGS; ++ki ) {
            preds[ki] = k[ki] < this->params_k_;
        }

        // Update the predicates.
        this->preds_ &= xmma::pack_predicates(preds);
    }

    // The residue to "fix" the predicates.
    inline __device__ void residue_a_t_b_n() {
        // The coordinates -- use inline PTX to avoid LWVM's rematerialization.
        int bidz;
        asm volatile("mov.b32 %0, %%ctaid.z;" : "=r"(bidz));
        int tidx;
        asm volatile("mov.b32 %0, %%tid.x;" : "=r"(tidx));

        // The position in the K dimension.
        int k = (this->is_batched_ ? 0 : bidz*Cta_tile::K) + tidx % THREADS_PER_COLUMN * ELEMENTS_PER_LDG;

        // Jump back to the loop if we have nothing to do.
        if( this->params_residue_k_ + k < this->params_k_ ) {
            return;
        }

        // Disable the predicates.
        this->preds_ = 0u;
    }

    // The residue to "fix" the predicates.
    inline __device__ void residue_a_t() {
        // The coordinates -- use inline PTX to avoid LWVM's rematerialization.
        int bidz;
        asm volatile("mov.b32 %0, %%ctaid.z;" : "=r"(bidz));
        int tidx;
        asm volatile("mov.b32 %0, %%tid.x;" : "=r"(tidx));

        // The position in the K dimension.
        int k = (this->is_batched_ ? 0 : bidz * Cta_tile::K / 2) + tidx % THREADS_PER_COLUMN * ELEMENTS_PER_LDG;

        // Jump back to the loop if we have nothing to do.
        if( this->params_residue_half_k_ + k < this->params_half_k_ ) {
            return;
        }

        // Disable the predicates.
        this->preds_ = 0u;
    }

    // The residue to "fix" the predicates.
    inline __device__ void residue_a_n() {
        // The coordinates -- use inline PTX to avoid LWVM's rematerialization.
        int bidz;
        asm volatile("mov.b32 %0, %%ctaid.z;" : "=r"(bidz));
        int tidx;
        asm volatile("mov.b32 %0, %%tid.x;" : "=r"(tidx));

        // The position in the K dimension.
        int k[LDGS];
        k[0] = this->params_residue_half_k_ + (this->is_batched_ ? 0 : bidz * Cta_tile::K / 2) + tidx / THREADS_PER_COLUMN;
        #pragma unroll
        for( int ki = 1; ki < LDGS; ++ki ) {
            k[ki] = k[0] + ki*COLUMNS_PER_LDG;
        }

        // The predicates.
        uint32_t preds[LDGS];
        #pragma unroll
        for( int ki = 0; ki < LDGS; ++ki ) {
            preds[ki] = k[ki] < this->params_half_k_;
        }

        // Update the predicates.
        this->preds_ &= xmma::pack_predicates(preds);
    }

    // Move the pointers and update the predicates for R2P/P2R.
    inline __device__ void restore_predicate() {
        preds_ = pre_store_preds_;
    }

    // The K dimension.
    const int params_k_, params_residue_k_;
    // The half K dimension.
    const int params_half_k_, params_residue_half_k_;
    // The enablement of Batched GEMM.
    const bool is_batched_;
    // The pointer.
    const char *ptr_;
    // The associated offsets.
    int offsets_[LDGS];
    // The predicates.
    uint32_t preds_;
    // The pre-stored predicates.
    uint32_t pre_store_preds_;
    // The fetch registers.
    int4 fetch_[LDGS];
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Traits, typename Cta_tile, int M, int N >
struct Gemm_gmem_tile_a
    : public Gemm_gmem_tile_base<Traits, Cta_tile, M, N, Traits::BITS_PER_ELEMENT_A> {

    // The base class.
    using Base = Gemm_gmem_tile_base<Traits, Cta_tile, M, N, Traits::BITS_PER_ELEMENT_A>;

    // Ctor.
    template< typename Params >
    inline __device__ Gemm_gmem_tile_a(const Params &params)
        : Base(params, params.a_gmem) {
    }

    // Load a tile from global memory.
    inline __device__ void load() {
        const char *ptrs[Base::LDGS];
        #pragma unroll
        for( int ii = 0; ii < Base::LDGS; ++ii ) {
            ptrs[ii] = this->ptr_ + Traits::offset_in_bytes_a(this->offsets_[ii]);
        }
        xmma::ldg(this->fetch_, ptrs, this->preds_);
    }

    // Load a tile from global memory.
    template< typename Xmma_smem_tile >
    inline __device__ void load(Xmma_smem_tile &smem, const uint64_t mem_desc) {
        const void *ptrs[Base::LDGS];
        #pragma unroll
        for( int ii = 0; ii < Base::LDGS; ++ii ) {
            ptrs[ii] = this->ptr_ + Traits::offset_in_bytes_a(this->offsets_[ii]);
        }
        // Issue the ldgsts.
        smem.store(ptrs, this->preds_, mem_desc);
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    typename Traits,
    typename Cta_tile,
    typename mma_type,
    typename output_layout
>
struct Gemm_gmem_tile_a_t {

};

template<
    typename Traits,
    typename Cta_tile,
    typename mma_type,
    typename output_layout
>
struct Gemm_gmem_tile_a_n {

};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    typename Traits,
    typename Cta_tile
    //typename mma_type,
    //typename output_layout
>
struct Gemm_gmem_tile_a_n <Traits,
                           Cta_tile,
                           uint16_t,
                           xmma::Row>
    : public Gemm_gmem_tile_a<Traits, Cta_tile, Cta_tile::M, Cta_tile::K / 2> {

    // The base class.
    using Base = Gemm_gmem_tile_a<Traits, Cta_tile, Cta_tile::M, Cta_tile::K / 2>;
    // The expected shared memory layout.
    using Smem_layout = xmma::Col;

    // Ctor.
    template< typename Params >
    inline __device__ Gemm_gmem_tile_a_n(const Params &params, int bidm, int bidz, int tidx)
        : Base(params) {

        // The position in the M dimension.
        const int m = bidm*Cta_tile::M + tidx % Base::THREADS_PER_COLUMN * Base::ELEMENTS_PER_LDG;
        const int half_k = params.k / 2;

        // Address of specific batch
        if( params.batch.is_batched ) {
            this->ptr_ += Traits::offset_in_bytes_a(static_cast<int64_t>(bidz) * params.m * half_k);
        }

        // The K position for the thread.
        int k[Base::LDGS];
        k[0] = (params.batch.is_batched ? 0 : bidz * Cta_tile::K / 2) + tidx / Base::THREADS_PER_COLUMN;
        #pragma unroll
        for( int ki = 1; ki < Base::LDGS; ++ki ) {
            k[ki] = k[0] + ki*Base::COLUMNS_PER_LDG;
        }

        // Compute the offsets.
        #pragma unroll
        for( int ki = 0; ki < Base::LDGS; ++ki ) {
            this->offsets_[ki] = k[ki]*params.lda + m;
        }

        // Compute the predicates.
        uint32_t preds[Base::LDGS];
        #pragma unroll
        for( int ki = 0; ki < Base::LDGS; ++ki ) {
            //preds[ki] = k[ki] < params.k;
            preds[ki] = k[ki] < half_k;
        }

        // Finalize the predicates.
        asm volatile("set.lt.u32.u32 %0, %1, %2;" : "=r"(this->preds_) : "r"(m), "r"(params.m));
        this->preds_ &= xmma::pack_predicates(preds);

        this->pre_store_preds_ = this->preds_ ;
    }

    // The residue to "fix" the predicates.
    inline __device__ void residue() {
        this->residue_a_n();
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    typename Traits,
    typename Cta_tile
    //typename mma_type,
    //typename output_layout
>
struct Gemm_gmem_tile_a_t <Traits,
                           Cta_tile,
                           uint16_t,
                           xmma::Row>
    : public Gemm_gmem_tile_a<Traits, Cta_tile, Cta_tile::K / 2, Cta_tile::M> {

    // The base class.
    using Base = Gemm_gmem_tile_a<Traits, Cta_tile, Cta_tile::K / 2, Cta_tile::M>;
    // The expected shared memory layout.
    using Smem_layout = xmma::Row;

    // Ctor.
    template< typename Params >
    inline __device__ Gemm_gmem_tile_a_t(const Params &params, int bidm, int bidz, int tidx)
        : Base(params) {

        // The position in the K dimension.
        const int k = (params.batch.is_batched ? 0 : bidz*Cta_tile::K / 2) + tidx % Base::THREADS_PER_COLUMN * Base::ELEMENTS_PER_LDG;
        const int half_k = params.k / 2;

        // Address of specific batch
        if( params.batch.is_batched ) {
            this->ptr_ += Traits::offset_in_bytes_a(static_cast<int64_t>(bidz) * params.m * half_k);
        }

        // For each LDG, compute the M position.
        int m[Base::LDGS];
        m[0] = bidm*Cta_tile::M + tidx / Base::THREADS_PER_COLUMN;
        #pragma unroll
        for( int mi = 1; mi < Base::LDGS; ++mi ) {
            m[mi] = m[0] + mi*Base::COLUMNS_PER_LDG;
        }

        // Compute the offsets.
        #pragma unroll
        for( int mi = 0; mi < Base::LDGS; ++mi ) {
            this->offsets_[mi] = m[mi]*(params.lda/2) + k;
        }

        // Compute the predicates.
        uint32_t preds[Base::LDGS];
        #pragma unroll
        for( int mi = 0; mi < Base::LDGS; ++mi ) {
            preds[mi] = m[mi] < params.m;
        }

        // Finalize the predicates.
        asm volatile("set.lt.u32.u32 %0, %1, %2;" : "=r"(this->preds_) : "r"(k), "r"(half_k));
        this->preds_ &= xmma::pack_predicates(preds);
        
        this->pre_store_preds_ = this->preds_ ;
    }

    // The residue to "fix" the predicates.
    inline __device__ void residue() {
        this->residue_a_t();
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Traits, typename Cta_tile, typename output_layout >
struct Gemm_gmem_tile_a_n_32b
    : public Gemm_gmem_tile_a<Traits, Cta_tile, Cta_tile::M, Cta_tile::K / 2> {

    // The base class.
    using Base = Gemm_gmem_tile_a<Traits, Cta_tile, Cta_tile::M, Cta_tile::K / 2>;
    // The expected shared memory layout.
    using Smem_layout = xmma::Col;

    // Ctor.
    template< typename Params >
    inline __device__ Gemm_gmem_tile_a_n_32b(const Params &params, int bidm, int bidz, int tidx)
        : Base(params) {

        // The position in the M dimension.
        const int m = bidm*Cta_tile::M + tidx % Base::THREADS_PER_COLUMN * Base::ELEMENTS_PER_LDG;
        const int half_k = params.k / 2;

        // Address of specific batch
        if( params.batch.is_batched ) {
            this->ptr_ += Traits::offset_in_bytes_a(static_cast<int64_t>(bidz) * params.m * half_k);
        }

        // The K position for the thread.
        int k[Base::LDGS];
        k[0] = (params.batch.is_batched ? 0 : bidz * Cta_tile::K / 2) + tidx / Base::THREADS_PER_COLUMN;
        #pragma unroll
        for( int ki = 1; ki < Base::LDGS; ++ki ) {
            k[ki] = k[0] + ki*Base::COLUMNS_PER_LDG;
        }

        // Compute the offsets.
        #pragma unroll
        for( int ki = 0; ki < Base::LDGS; ++ki ) {
            this->offsets_[ki] = k[ki]*params.lda + m;
        }

        // Compute the predicates.
        uint32_t preds[Base::LDGS];
        #pragma unroll
        for( int ki = 0; ki < Base::LDGS; ++ki ) {
            //preds[ki] = k[ki] < params.k;
            preds[ki] = k[ki] < half_k;
        }

        // Finalize the predicates.
        asm volatile("set.lt.u32.u32 %0, %1, %2;" : "=r"(this->preds_) : "r"(m), "r"(params.m));
        this->preds_ &= xmma::pack_predicates(preds);

        this->pre_store_preds_ = this->preds_ ;
    }

    // The residue to "fix" the predicates.
    inline __device__ void residue() {
        this->residue_a_n();
    }
};

template<
    typename Traits,
    typename Cta_tile
    //typename mma_type,
    //typename output_layout
>
struct Gemm_gmem_tile_a_n <Traits,
                           Cta_tile,
                           uint32_t,
                           xmma::Row>
    : public Gemm_gmem_tile_a_n_32b <Traits, Cta_tile, xmma::Row> {

    using Smem_layout = xmma::Col;

    // The base class.
    using Base = Gemm_gmem_tile_a_n_32b<Traits, Cta_tile, xmma::Row>;

    // Ctor.
    template< typename Params >
    inline __device__ Gemm_gmem_tile_a_n(const Params &params, int bidm, int bidz, int tidx)
        : Base(params, bidm, bidz, tidx) {
    }

};

template<
    typename Traits,
    typename Cta_tile
    //typename mma_type,
    //typename output_layout
>
struct Gemm_gmem_tile_a_n <Traits,
                           Cta_tile,
                           float,
                           xmma::Row>
    : public Gemm_gmem_tile_a_n_32b <Traits, Cta_tile, xmma::Row> {

    using Smem_layout = xmma::Col;

    // The base class.
    using Base = Gemm_gmem_tile_a_n_32b<Traits, Cta_tile, xmma::Row>;

    // Ctor.
    template< typename Params >
    inline __device__ Gemm_gmem_tile_a_n(const Params &params, int bidm, int bidz, int tidx)
        : Base(params, bidm, bidz, tidx) {
    }
};


template<
    typename Traits,
    typename Cta_tile
    //typename mma_type,
    //typename output_layout
>
struct Gemm_gmem_tile_a_n <Traits,
                           Cta_tile,
                           uint32_t,
                           xmma::Col>
    : public Gemm_gmem_tile_a_n_32b <Traits, Cta_tile, xmma::Col> {

    using Smem_layout = xmma::Col;

    // The base class.
    using Base = Gemm_gmem_tile_a_n_32b<Traits, Cta_tile, xmma::Col>;

    // Ctor.
    template< typename Params >
    inline __device__ Gemm_gmem_tile_a_n(const Params &params, int bidm, int bidz, int tidx)
        : Base(params, bidm, bidz, tidx) {
    }
};

template<
    typename Traits,
    typename Cta_tile
    //typename mma_type,
    //typename output_layout
>
struct Gemm_gmem_tile_a_n <Traits,
                           Cta_tile,
                           float,
                           xmma::Col>
    : public Gemm_gmem_tile_a_n_32b <Traits, Cta_tile, xmma::Col> {

    using Smem_layout = xmma::Col;

    // The base class.
    using Base = Gemm_gmem_tile_a_n_32b<Traits, Cta_tile, xmma::Col>;

    // Ctor.
    template< typename Params >
    inline __device__ Gemm_gmem_tile_a_n(const Params &params, int bidm, int bidz, int tidx)
        : Base(params, bidm, bidz, tidx) {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Traits, typename Cta_tile, typename output_layout >
struct Gemm_gmem_tile_a_t_32b
    : public Gemm_gmem_tile_a<Traits, Cta_tile, Cta_tile::K / 2, Cta_tile::M> {

    // The base class.
    using Base = Gemm_gmem_tile_a<Traits, Cta_tile, Cta_tile::K / 2, Cta_tile::M>;
    // The expected shared memory layout.
    using Smem_layout = xmma::Row;

    // Ctor.
    template< typename Params >
    inline __device__ Gemm_gmem_tile_a_t_32b(const Params &params, int bidm, int bidz, int tidx)
        : Base(params) {

        // The position in the K dimension.
        const int k = (params.batch.is_batched ? 0 : bidz*Cta_tile::K / 2) + tidx % Base::THREADS_PER_COLUMN * Base::ELEMENTS_PER_LDG;
        const int half_k = params.k / 2;

        // Address of specific batch
        if( params.batch.is_batched ) {
            this->ptr_ += Traits::offset_in_bytes_a(static_cast<int64_t>(bidz) * params.m * half_k);
        }

        // For each LDG, compute the M position.
        int m[Base::LDGS];
        m[0] = bidm*Cta_tile::M + tidx / Base::THREADS_PER_COLUMN;
        #pragma unroll
        for( int mi = 1; mi < Base::LDGS; ++mi ) {
            m[mi] = m[0] + mi*Base::COLUMNS_PER_LDG;
        }

        // Compute the offsets.
        #pragma unroll
        for( int mi = 0; mi < Base::LDGS; ++mi ) {
            this->offsets_[mi] = m[mi]*(params.lda/2) + k;
        }

        // Compute the predicates.
        uint32_t preds[Base::LDGS];
        #pragma unroll
        for( int mi = 0; mi < Base::LDGS; ++mi ) {
            preds[mi] = m[mi] < params.m;
        }

        // Finalize the predicates.
        asm volatile("set.lt.u32.u32 %0, %1, %2;" : "=r"(this->preds_) : "r"(k), "r"(half_k));
        this->preds_ &= xmma::pack_predicates(preds);

        this->pre_store_preds_ = this->preds_ ;
    }

    // The residue to "fix" the predicates.
    inline __device__ void residue() {
        this->residue_a_t();
    }
};

template<
    typename Traits,
    typename Cta_tile
    //typename mma_type,
    //typename output_layout
>
struct Gemm_gmem_tile_a_t <Traits,
                           Cta_tile,
                           uint32_t,
                           xmma::Row>
    : public Gemm_gmem_tile_a_t_32b <Traits, Cta_tile, xmma::Row> {

    using Smem_layout = xmma::Row;

    // The base class.
    using Base = Gemm_gmem_tile_a_t_32b<Traits, Cta_tile, xmma::Row>;

    // Ctor.
    template< typename Params >
    inline __device__ Gemm_gmem_tile_a_t(const Params &params, int bidm, int bidz, int tidx)
        : Base(params, bidm, bidz, tidx) {
    }
};

template<
    typename Traits,
    typename Cta_tile
    //typename mma_type,
    //typename output_layout
>
struct Gemm_gmem_tile_a_t <Traits,
                           Cta_tile,
                           float,
                           xmma::Row>
    : public Gemm_gmem_tile_a_t_32b <Traits, Cta_tile, xmma::Row> {

    using Smem_layout = xmma::Row;

    // The base class.
    using Base = Gemm_gmem_tile_a_t_32b<Traits, Cta_tile, xmma::Row>;

    // Ctor.
    template< typename Params >
    inline __device__ Gemm_gmem_tile_a_t(const Params &params, int bidm, int bidz, int tidx)
        : Base(params, bidm, bidz, tidx) {
    }
};


template<
    typename Traits,
    typename Cta_tile
    //typename mma_type,
    //typename output_layout
>
struct Gemm_gmem_tile_a_t <Traits,
                           Cta_tile,
                           uint32_t,
                           xmma::Col>
    : public Gemm_gmem_tile_a_t_32b <Traits, Cta_tile, xmma::Col> {

    using Smem_layout = xmma::Row;

    // The base class.
    using Base = Gemm_gmem_tile_a_t_32b<Traits, Cta_tile, xmma::Col>;

    // Ctor.
    template< typename Params >
    inline __device__ Gemm_gmem_tile_a_t(const Params &params, int bidm, int bidz, int tidx)
        : Base(params, bidm, bidz, tidx) {
    }
};

template<
    typename Traits,
    typename Cta_tile
    //typename mma_type,
    //typename output_layout
>
struct Gemm_gmem_tile_a_t <Traits,
                           Cta_tile,
                           float,
                           xmma::Col>
    : public Gemm_gmem_tile_a_t_32b <Traits, Cta_tile, xmma::Col> {

    using Smem_layout = xmma::Row;

    // The base class.
    using Base = Gemm_gmem_tile_a_t_32b<Traits, Cta_tile, xmma::Col>;

    // Ctor.
    template< typename Params >
    inline __device__ Gemm_gmem_tile_a_t(const Params &params, int bidm, int bidz, int tidx)
        : Base(params, bidm, bidz, tidx) {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Traits, typename Cta_tile, int M, int N >
struct Gemm_gmem_tile_b
    : public Gemm_gmem_tile_base<Traits, Cta_tile, M, N, Traits::BITS_PER_ELEMENT_B> {

    // The base class.
    using Base = Gemm_gmem_tile_base<Traits, Cta_tile, M, N, Traits::BITS_PER_ELEMENT_B>;

    // Ctor.
    template< typename Params >
    inline __device__ Gemm_gmem_tile_b(const Params &params)
        : Base(params, params.b_gmem) {
    }

    // Load a tile from global memory.
    inline __device__ void load() {
        const char *ptrs[Base::LDGS];
        #pragma unroll
        for( int ii = 0; ii < Base::LDGS; ++ii ) {
            ptrs[ii] = this->ptr_ + Traits::offset_in_bytes_b(this->offsets_[ii]);
        }
        xmma::ldg(this->fetch_, ptrs, this->preds_);
    }

    // Load a tile from global memory.
    template< typename Xmma_smem_tile >
    inline __device__ void load(Xmma_smem_tile &smem, const uint64_t mem_desc) {
        const void *ptrs[Base::LDGS];
        #pragma unroll
        for( int ii = 0; ii < Base::LDGS; ++ii ) {
            ptrs[ii] = this->ptr_ + Traits::offset_in_bytes_b(this->offsets_[ii]);
        }
        // Issue the ldgsts.
        smem.store(ptrs, this->preds_, mem_desc);
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    typename Traits,
    typename Cta_tile,
    typename mma_type,
    typename output_layout
>
struct Gemm_gmem_tile_b_t {

};

template<
    typename Traits,
    typename Cta_tile,
    typename mma_type,
    typename output_layout
>
struct Gemm_gmem_tile_b_n {

};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    typename Traits,
    typename Cta_tile
    //typename mma_type,
    //typename output_layout
>
struct Gemm_gmem_tile_b_n <Traits,
                           Cta_tile,
                           uint16_t,
                           xmma::Row>
    : public Gemm_gmem_tile_b<Traits, Cta_tile, Cta_tile::K, Cta_tile::N> {

    // The base class.
    using Base = Gemm_gmem_tile_b<Traits, Cta_tile, Cta_tile::K, Cta_tile::N>;
    // The expected shared memory layout.
    using Smem_layout = xmma::Col;

    // Ctor.
    template< typename Params >
    inline __device__ Gemm_gmem_tile_b_n(const Params &params, int bidn, int bidz, int tidx)
        : Base(params) {

        // The position in the K dimension.
        const int k = (params.batch.is_batched ? 0 : bidz*Cta_tile::K) + tidx % Base::THREADS_PER_COLUMN * Base::ELEMENTS_PER_LDG;

        // Address of specific batch
        if( params.batch.is_batched ) {
            this->ptr_ += Traits::offset_in_bytes_b(static_cast<int64_t>(bidz) * params.k * params.n);
        }

        // For each LDG, compute the M position.
        int n[Base::LDGS];
        n[0] = bidn*Cta_tile::N + tidx / Base::THREADS_PER_COLUMN;
        #pragma unroll
        for( int ni = 1; ni < Base::LDGS; ++ni ) {
            n[ni] = n[0] + ni*Base::COLUMNS_PER_LDG;
        }

        // Compute the offsets.
        #pragma unroll
        for( int ni = 0; ni < Base::LDGS; ++ni ) {
            this->offsets_[ni] = n[ni]*params.ldb + k;
        }

        // Compute the predicates.
        uint32_t preds[Base::LDGS];
        #pragma unroll
        for( int ni = 0; ni < Base::LDGS; ++ni ) {
            preds[ni] = n[ni] < params.n;
        }

        // Finalize the predicates.
        asm volatile("set.lt.u32.u32 %0, %1, %2;" : "=r"(this->preds_) : "r"(k), "r"(params.k));
        this->preds_ &= xmma::pack_predicates(preds);

        this->pre_store_preds_ = this->preds_ ;
    }

    // The residue to "fix" the predicates.
    inline __device__ void residue() {
        this->residue_a_t_b_n();
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    typename Traits,
    typename Cta_tile
    //typename mma_type,
    //typename output_layout
>
struct Gemm_gmem_tile_b_t <Traits,
                           Cta_tile,
                           uint16_t,
                           xmma::Row>
    : public Gemm_gmem_tile_b<Traits, Cta_tile, Cta_tile::N, Cta_tile::K> {

    // The base class.
    using Base = Gemm_gmem_tile_b<Traits, Cta_tile, Cta_tile::N, Cta_tile::K>;
    // The expected shared memory layout.
    using Smem_layout = xmma::Row;

    // Ctor.
    template< typename Params >
    inline __device__ Gemm_gmem_tile_b_t(const Params &params, int bidn, int bidz, int tidx)
        : Base(params) {

        // The position in the N dimension.
        const int n = bidn*Cta_tile::N + tidx % Base::THREADS_PER_COLUMN * Base::ELEMENTS_PER_LDG;

        // Address of specific batch
        if( params.batch.is_batched ) {
            this->ptr_ += Traits::offset_in_bytes_b(static_cast<int64_t>(bidz) * params.k * params.n);
        }

        // The K position for the thread.
        int k[Base::LDGS];
        k[0] = (params.batch.is_batched ? 0 : bidz*Cta_tile::K) + tidx / Base::THREADS_PER_COLUMN;
        #pragma unroll
        for( int ki = 1; ki < Base::LDGS; ++ki ) {
            k[ki] = k[0] + ki*Base::COLUMNS_PER_LDG;
        }

        // Compute the offsets.
        #pragma unroll
        for( int ki = 0; ki < Base::LDGS; ++ki ) {
            this->offsets_[ki] = k[ki]*params.ldb + n;
        }

        // Compute the predicates.
        uint32_t preds[Base::LDGS];
        #pragma unroll
        for( int ki = 0; ki < Base::LDGS; ++ki ) {
            preds[ki] = k[ki] < params.k;
        }

        // Finalize the predicates.
        asm volatile("set.lt.u32.u32 %0, %1, %2;" : "=r"(this->preds_) : "r"(n), "r"(params.n));
        this->preds_ &= xmma::pack_predicates(preds);

        this->pre_store_preds_ = this->preds_ ;
    }

    // The residue to "fix" the predicates.
    inline __device__ void residue() {
        this->residue_a_n_b_t();
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Traits, typename Cta_tile, typename output_layout >
struct Gemm_gmem_tile_b_n_32b
    : public Gemm_gmem_tile_b<Traits, Cta_tile, Cta_tile::K, Cta_tile::N> {

    // The base class.
    using Base = Gemm_gmem_tile_b<Traits, Cta_tile, Cta_tile::K, Cta_tile::N>;
    // The expected shared memory layout.
    using Smem_layout = xmma::Col;

    // Ctor.
    template< typename Params >
    inline __device__ Gemm_gmem_tile_b_n_32b(const Params &params, int bidn, int bidz, int tidx)
        : Base(params) {

        // The position in the K dimension.
        const int k = (params.batch.is_batched ? 0 : bidz*Cta_tile::K) + tidx % Base::THREADS_PER_COLUMN * Base::ELEMENTS_PER_LDG;

        // Address of specific batch
        if( params.batch.is_batched ) {
            this->ptr_ += Traits::offset_in_bytes_b(static_cast<int64_t>(bidz) * params.k * params.n);
        }

        // For each LDG, compute the M position.
        int n[Base::LDGS];
        n[0] = bidn*Cta_tile::N + tidx / Base::THREADS_PER_COLUMN;
        #pragma unroll
        for( int ni = 1; ni < Base::LDGS; ++ni ) {
            n[ni] = n[0] + ni*Base::COLUMNS_PER_LDG;
        }

        // Compute the offsets.
        #pragma unroll
        for( int ni = 0; ni < Base::LDGS; ++ni ) {
            this->offsets_[ni] = n[ni]*params.ldb + k;
        }

        // Compute the predicates.
        uint32_t preds[Base::LDGS];
        #pragma unroll
        for( int ni = 0; ni < Base::LDGS; ++ni ) {
            preds[ni] = n[ni] < params.n;
        }

        // Finalize the predicates.
        asm volatile("set.lt.u32.u32 %0, %1, %2;" : "=r"(this->preds_) : "r"(k), "r"(params.k));
        this->preds_ &= xmma::pack_predicates(preds);

        this->pre_store_preds_ = this->preds_ ;
    }

    // The residue to "fix" the predicates.
    inline __device__ void residue() {
        this->residue_a_t_b_n();
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    typename Traits,
    typename Cta_tile
    //typename mma_type,
    //typename output_layout
>
struct Gemm_gmem_tile_b_n <Traits,
                           Cta_tile,
                           uint32_t,
                           xmma::Row>
    : public Gemm_gmem_tile_b_n_32b <Traits, Cta_tile, xmma::Row> {

    // The base class.
    using Base = Gemm_gmem_tile_b_n_32b<Traits, Cta_tile, xmma::Row>;

    using Smem_layout = typename Base::Smem_layout;

    // Ctor.
    template< typename Params >
    inline __device__ Gemm_gmem_tile_b_n(const Params &params, int bidn, int bidz, int tidx)
        : Base(params, bidn, bidz, tidx) {
    }

};

template<
    typename Traits,
    typename Cta_tile
    //typename mma_type,
    //typename output_layout
>
struct Gemm_gmem_tile_b_n <Traits,
                           Cta_tile,
                           float,
                           xmma::Row>
    : public Gemm_gmem_tile_b_n_32b <Traits, Cta_tile, xmma::Row> {

    // The base class.
    using Base = Gemm_gmem_tile_b_n_32b<Traits, Cta_tile, xmma::Row>;

    using Smem_layout = typename Base::Smem_layout;

    // Ctor.
    template< typename Params >
    inline __device__ Gemm_gmem_tile_b_n(const Params &params, int bidn, int bidz, int tidx)
        : Base(params, bidn, bidz, tidx) {
    }
};


template<
    typename Traits,
    typename Cta_tile
    //typename mma_type,
    //typename output_layout
>
struct Gemm_gmem_tile_b_n <Traits,
                           Cta_tile,
                           uint32_t,
                           xmma::Col>
    : public Gemm_gmem_tile_b_n_32b <Traits, Cta_tile, xmma::Col> {

    // The base class.
    using Base = Gemm_gmem_tile_b_n_32b<Traits, Cta_tile, xmma::Col>;

    using Smem_layout = typename Base::Smem_layout;

    // Ctor.
    template< typename Params >
    inline __device__ Gemm_gmem_tile_b_n(const Params &params, int bidn, int bidz, int tidx)
        : Base(params, bidn, bidz, tidx) {
    }
};

template<
    typename Traits,
    typename Cta_tile
    //typename mma_type,
    //typename output_layout
>
struct Gemm_gmem_tile_b_n <Traits,
                           Cta_tile,
                           float,
                           xmma::Col>
    : public Gemm_gmem_tile_b_n_32b <Traits, Cta_tile, xmma::Col> {

    // The base class.
    using Base = Gemm_gmem_tile_b_n_32b<Traits, Cta_tile, xmma::Col>;

    using Smem_layout = typename Base::Smem_layout;

    // Ctor.
    template< typename Params >
    inline __device__ Gemm_gmem_tile_b_n(const Params &params, int bidn, int bidz, int tidx)
        : Base(params, bidn, bidz, tidx) {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Traits, typename Cta_tile, typename output_layout >
struct Gemm_gmem_tile_b_t_32b
    : public Gemm_gmem_tile_b<Traits, Cta_tile, Cta_tile::N, Cta_tile::K> {

    // The base class.
    using Base = Gemm_gmem_tile_b<Traits, Cta_tile, Cta_tile::N, Cta_tile::K>;
    // The expected shared memory layout.
    using Smem_layout = xmma::Row;

    // Ctor.
    template< typename Params >
    inline __device__ Gemm_gmem_tile_b_t_32b(const Params &params, int bidn, int bidz, int tidx)
        : Base(params) {

        // The position in the N dimension.
        const int n = bidn*Cta_tile::N + tidx % Base::THREADS_PER_COLUMN * Base::ELEMENTS_PER_LDG;

        // Address of specific batch
        if( params.batch.is_batched ) {
            this->ptr_ += Traits::offset_in_bytes_b(static_cast<int64_t>(bidz) * params.k * params.n);
        }

        // The K position for the thread.
        int k[Base::LDGS];
        k[0] = (params.batch.is_batched ? 0 : bidz*Cta_tile::K) + tidx / Base::THREADS_PER_COLUMN;

        #pragma unroll
        for( int ki = 1; ki < Base::LDGS; ++ki ) {
            k[ki] = k[0] + ki*Base::COLUMNS_PER_LDG;
        }

        // Compute the offsets.
        #pragma unroll
        for( int ki = 0; ki < Base::LDGS; ++ki ) {
            this->offsets_[ki] = k[ki]*params.ldb + n;
        }

        // Compute the predicates.
        uint32_t preds[Base::LDGS];
        #pragma unroll
        for( int ki = 0; ki < Base::LDGS; ++ki ) {
            preds[ki] = k[ki] < params.k;
        }

        // Finalize the predicates.
        asm volatile("set.lt.u32.u32 %0, %1, %2;" : "=r"(this->preds_) : "r"(n), "r"(params.n));
        this->preds_ &= xmma::pack_predicates(preds);

        this->pre_store_preds_ = this->preds_ ;
    }

    // The residue to "fix" the predicates.
    inline __device__ void residue() {
        this->residue_a_n_b_t();
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    typename Traits,
    typename Cta_tile
    //typename mma_type,
    //typename output_layout
>
struct Gemm_gmem_tile_b_t <Traits,
                           Cta_tile,
                           uint32_t,
                           xmma::Row>
    : public Gemm_gmem_tile_b_t_32b <Traits, Cta_tile, xmma::Row> {

    // The base class.
    using Base = Gemm_gmem_tile_b_t_32b<Traits, Cta_tile, xmma::Row>;

    using Smem_layout = typename Base::Smem_layout;

    // Ctor.
    template< typename Params >
    inline __device__ Gemm_gmem_tile_b_t(const Params &params, int bidn, int bidz, int tidx)
        : Base(params, bidn, bidz, tidx) {
    }
};

template<
    typename Traits,
    typename Cta_tile
    //typename mma_type,
    //typename output_layout
>
struct Gemm_gmem_tile_b_t <Traits,
                           Cta_tile,
                           float,
                           xmma::Row>
    : public Gemm_gmem_tile_b_t_32b <Traits, Cta_tile, xmma::Row> {

    // The base class.
    using Base = Gemm_gmem_tile_b_t_32b<Traits, Cta_tile, xmma::Row>;

    using Smem_layout = typename Base::Smem_layout;

    // Ctor.
    template< typename Params >
    inline __device__ Gemm_gmem_tile_b_t(const Params &params, int bidn, int bidz, int tidx)
        : Base(params, bidn, bidz, tidx) {
    }
};


template<
    typename Traits,
    typename Cta_tile
    //typename mma_type,
    //typename output_layout
>
struct Gemm_gmem_tile_b_t <Traits,
                           Cta_tile,
                           uint32_t,
                           xmma::Col>
    : public Gemm_gmem_tile_b_t_32b <Traits, Cta_tile, xmma::Col> {

    // The base class.
    using Base = Gemm_gmem_tile_b_t_32b<Traits, Cta_tile, xmma::Col>;

    using Smem_layout = typename Base::Smem_layout;

    // Ctor.
    template< typename Params >
    inline __device__ Gemm_gmem_tile_b_t(const Params &params, int bidn, int bidz, int tidx)
        : Base(params, bidn, bidz, tidx) {
    }
};

template<
    typename Traits,
    typename Cta_tile
    //typename mma_type,
    //typename output_layout
>
struct Gemm_gmem_tile_b_t <Traits,
                           Cta_tile,
                           float,
                           xmma::Col>
    : public Gemm_gmem_tile_b_t_32b <Traits, Cta_tile, xmma::Col> {

    // The base class.
    using Base = Gemm_gmem_tile_b_t_32b<Traits, Cta_tile, xmma::Col>;

    using Smem_layout = typename Base::Smem_layout;

    // Ctor.
    template< typename Params >
    inline __device__ Gemm_gmem_tile_b_t(const Params &params, int bidn, int bidz, int tidx)
        : Base(params, bidn, bidz, tidx) {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    typename Traits,
    typename Cta_tile,
    typename Layout,
    typename Fragment_c = xmma::Fragment_c<Traits, Cta_tile>
>
struct Gemm_gmem_tile_epilogue {

};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    typename Traits,
    typename Cta_tile,
    //typename Layout,
    typename Fragment_c
>
struct Gemm_gmem_tile_epilogue <Traits,
                                Cta_tile,
                                xmma::Col,
                                Fragment_c>
    : public xmma::helpers::Gmem_tile_epilogue<Traits, Cta_tile, xmma::Col, Fragment_c> {

    // The base class.
    using Base = xmma::helpers::Gmem_tile_epilogue<Traits, Cta_tile, xmma::Col, Fragment_c>;

    // The base class.
    using output_layout = xmma::Col;

    // Ctor.
    template< typename Params >
    inline __device__ Gemm_gmem_tile_epilogue(const Params &params, int bidm, int bidn, int bidz, int tidx)
        : Base(params.n,
               params.m,
               params.m,
               reinterpret_cast<char*>(params.d_gmem),
               reinterpret_cast<const char*>(params.c_gmem),
               bidm,
               bidn,
               bidz,
               tidx) {
            if( params.batch.is_batched ) {
                const int64_t batch_offset = static_cast<int64_t>(blockIdx.z) * params.m * params.n;
                this->out_ptr_ += Traits::offset_in_bytes_c(batch_offset);
                this->res_ptr_ += Traits::offset_in_bytes_c(batch_offset);
            }
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    typename Traits,
    typename Cta_tile,
    //typename Layout,
    typename Fragment_c
>
struct Gemm_gmem_tile_epilogue<Traits,
                                Cta_tile,
                                xmma::Row,
                                Fragment_c>
    : public xmma::helpers::Gmem_row_stride_tile_epilogue<Traits, Cta_tile, Fragment_c> {

    // The base class.
    using Base = xmma::helpers::Gmem_row_stride_tile_epilogue<Traits, Cta_tile, Fragment_c>;

    using output_layout = xmma::Row;

    // Ctor.
    template< typename Params >
    inline __device__ Gemm_gmem_tile_epilogue(const Params &params, int bidm, int bidn, int bidz, int tidx)
        : Base(params.m,
               params.n,
               params.n,
               reinterpret_cast<char*>(params.d_gmem),
               reinterpret_cast<const char*>(params.c_gmem),
               bidm,
               bidn,
               bidz,
               tidx) {
            if( params.batch.is_batched ) {
                const int64_t batch_offset = static_cast<int64_t>(blockIdx.z) * params.m * params.n;
                this->out_ptr_ += Traits::offset_in_bytes_c(batch_offset);
                this->res_ptr_ += Traits::offset_in_bytes_c(batch_offset);
            }
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

}
} // namespace gemm
} // namespace ext
} // namespace xmma
