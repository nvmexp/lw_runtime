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

#include <xmma/ext/sparse/ampere/fragment.h>
#include <xmma/warp_masks.h>
#include <xmma/ampere/smem_tile.h>
#include <xmma/ext/sparse/utils.h>
#include <xmma/ext/sparse/smem_tile.h>

namespace xmma {

/***************************************************************************************************
[WARNING] : The current sparse smem tile load implementation is based on 64x16 XMMA tile.
It's different from general 16x16 XMMA tile desing. We use it for now. After the sparse feature is
stablized we might alter the design into 16x16 tile in the future.
**************************************************************************************************/


////////////////////////////////////////////////////////////////////////////////////////////////////
//
// x M M A
//
////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Traits, int N >
struct Rows_per_xor_pattern_ampere_e {
    // The size in bits.
    enum { N_IN_BITS = N * Traits::BITS_PER_ELEMENT_E / Traits::ELEMENTS_PER_UINT16};
    // The number of rows.
    enum { VALUE = N_IN_BITS <= 256 ? 2 : (N_IN_BITS <= 512 ? 4 : 8) };
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    // The instruction traits.
    typename Traits,
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The size of the STS.
    int BYTES_PER_STS,
    // The number of buffers per tile.
    int BUFFERS_PER_TILE,
    // USE LDS OR LDSM
    bool USE_LDS_ = false,
    // How many rows to use for the XOR pattern to avoid bank conflicts?
    int ROWS_PER_XOR_PATTERN_ = Rows_per_xor_pattern_ampere_e<Traits, Cta_tile::HALF_K>::VALUE
>
struct Smem_tile_sparse_ampere_row_e
    : public Smem_metadata_tile_linear<Cta_tile,
                                     Cta_tile::M,
                                     Cta_tile::HALF_K/Traits::ELEMENTS_PER_UINT16,
                                     Traits::BITS_PER_ELEMENT_E,
                                     BYTES_PER_STS,
                                     BUFFERS_PER_TILE,
                                     0,
                                     ROWS_PER_XOR_PATTERN_,
                                     1> {

    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;
    // The base class.
    using Base = Smem_metadata_tile_linear<Cta_tile,
                                         Cta_tile::M,
                                         Cta_tile::HALF_K/Traits::ELEMENTS_PER_UINT16,
                                         Traits::BITS_PER_ELEMENT_E,
                                         BYTES_PER_STS,
                                         BUFFERS_PER_TILE,
                                         0,
                                         ROWS_PER_XOR_PATTERN_,
                                         1>;
    // The fragment.
    using Fragment = Fragment_e<Traits, Row>;

    // The size of a single LDS in bytes.
    enum { BYTES_PER_LDS = 16 };

    enum { USE_LDS = USE_LDS_ };

    // Ctor.
    inline __device__ Smem_tile_sparse_ampere_row_e(char *smem, int tidx)
        : Base(smem, tidx) {

        // The number of warps.
        const int WARPS_M = Cta_tile::WARPS_M;

        // Sparse GEMM
        // Apply LDSM.2, so the address of half warp is valid
        int smem_base_linear = 0;
        if(Cta_tile::K == 64 || Cta_tile::K == 32) {
            const int LDSM_2_OFFSET = 16;
            const int VALID_THREADS = 16;
            const int MOD_IDX = tidx % (WARPS_M * Cta_tile::THREADS_PER_WARP);
            const int WARP_M_IDX = MOD_IDX / Cta_tile::THREADS_PER_WARP;
            smem_base_linear = (MOD_IDX % VALID_THREADS + WARP_M_IDX * VALID_THREADS) * LDSM_2_OFFSET;
        } else {
            const int LDSM_4_OFFSET = 16;
            const int VALID_THREADS = 32;
            const int MOD_IDX = tidx % (WARPS_M * Cta_tile::THREADS_PER_WARP);
            const int WARP_M_IDX = MOD_IDX / Cta_tile::THREADS_PER_WARP;
            smem_base_linear = (MOD_IDX % VALID_THREADS + WARP_M_IDX * VALID_THREADS) * LDSM_4_OFFSET;
        }

        // Sparse Colw
        const int ty = (tidx / 32) % Cta_tile::WARPS_M;
        const int warp_id = tidx % 32;

        if (Cta_tile::K == 64 || Cta_tile::K == 32) {
            const int read_offset  = 4*(ty * 128 + warp_id * 4);
            this->smem_read_ptr_ = xmma::get_smem_pointer(&smem[read_offset]);
        } else {
            const int read_offset  = 4*(ty * 256 + warp_id * 4);
            this->smem_read_ptr_ = xmma::get_smem_pointer(&smem[read_offset]);
        }

        this->smem_read_offset_ = smem_base_linear;
    }
    // Load from shared memory.
    inline __device__ void load(Fragment (&e)[1], int ki) {
        // 2 is half meta_k, another 2 is 2 bytes
        uint2 tmp;
        uint4 tmp4;
        if (USE_LDS) {
            if (Cta_tile::K == 64 || Cta_tile::K == 32) {
                ldsm(tmp, this->smem_read_ptr_ + this->smem_read_buffer_ + ki * 256);
                e[0].reg(0) = tmp.x;
                e[0].reg(1) = tmp.y;
            } else {
                ldsm(tmp4, this->smem_read_ptr_ + this->smem_read_buffer_ + ki * 512);
                e[0].reg(0) = tmp4.x;
                e[0].reg(1) = tmp4.y;
                e[0].reg(2) = tmp4.z;
                e[0].reg(3) = tmp4.w;
            }
        } else {
            if (Cta_tile::K == 64 || Cta_tile::K == 32) {
                uint32_t offset = this->smem_read_offset_ + ki * Cta_tile::M * 2 * 2;
                ldsm(tmp, this->smem_ + offset + this->smem_read_buffer_ );
                e[0].reg(0) = tmp.x;
                e[0].reg(1) = tmp.y;
            } else {
                uint32_t offset = this->smem_read_offset_ + ki * Cta_tile::M * 2 * 4;
                ldsm(tmp4, this->smem_ + offset + this->smem_read_buffer_ );
                e[0].reg(0) = tmp4.x;
                e[0].reg(1) = tmp4.y;
                e[0].reg(2) = tmp4.z;
                e[0].reg(3) = tmp4.w;
            }

        }
    }

    // The pointer for LDSM consumption. That's a WAR to deal with a compiler limitation.
    uint32_t smem_read_ptr_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Input_type, typename Output_type >
struct Cols_per_xor_pattern_ampere<Ampere_sphmma_tf32_traits<Input_type, Output_type>> {
    enum { VALUE = 2 };
};

template< typename Input_type, typename Output_type, int N >
struct Rows_per_xor_pattern_ampere_col_a<Ampere_sphmma_tf32_traits<Input_type, Output_type>, N> {
    enum { VALUE = 4 };
};

template< typename Input_type, typename Output_type, int N >
struct Rows_per_xor_pattern_ampere_row_b<Ampere_sphmma_tf32_traits<Input_type, Output_type>, N> {
    // The number of rows.
    enum { VALUE = 4 };
};

template<
    // The instruction traits.
    typename Traits,
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The size of the STS.
    int BYTES_PER_STS,
    // The number of buffers per tile.
    int BUFFERS_PER_TILE,
    // How many rows to use for the XOR pattern to avoid bank conflicts?
    int ROWS_PER_XOR_PATTERN_ = Rows_per_xor_pattern_ampere_col_a<Traits, Cta_tile::M>::VALUE,
    // How many cols to use for the XOR pattern to avoid bank conflicts?
    int COLS_PER_XOR_PATTERN_ = Cols_per_xor_pattern_ampere<Traits>::VALUE
>
struct Smem_tile_sparse_ampere_col_a : public Smem_tile_without_skews<Cta_tile,
                                                               Cta_tile::HALF_K,
                                                               Cta_tile::M,
                                                               Traits::BITS_PER_ELEMENT_A,
                                                               BYTES_PER_STS,
                                                               BUFFERS_PER_TILE,
                                                               0,
                                                               ROWS_PER_XOR_PATTERN_,
                                                               COLS_PER_XOR_PATTERN_> {

    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;
    // The base class.
    using Base = Smem_tile_without_skews<Cta_tile,
                                         Cta_tile::HALF_K,
                                         Cta_tile::M,
                                         Traits::BITS_PER_ELEMENT_A,
                                         BYTES_PER_STS,
                                         BUFFERS_PER_TILE,
                                         0,
                                         ROWS_PER_XOR_PATTERN_,
                                         COLS_PER_XOR_PATTERN_>;
    // The fragment.
    using Fragment = Fragment_a<Traits, Col>;

    // Can we use LDSM? No if the data type is 32-bit large.
    enum { USE_LDSMT = Traits::BITS_PER_ELEMENT_A == 16 };
    // The size of a single LDS in bytes.
    enum { BYTES_PER_LDS = USE_LDSMT ? 16 : 4 };
    // The number of elements per LDS.
    enum { ELEMENTS_PER_LDS = BYTES_PER_LDS * 8 / Traits::BITS_PER_ELEMENT_A };

    // Ctor.
    inline __device__ Smem_tile_sparse_ampere_col_a(char *smem, int tidx)
        : Base(smem, tidx) {

        // The number of warps.
        const int WARPS_M = Cta_tile::WARPS_M;
        const int WARPS_N = Cta_tile::WARPS_N;
        const int WARPS_K = Cta_tile::WARPS_K;

        // The masks to select the warps.
        const int WARP_MASK_M = Warp_masks<WARPS_M, WARPS_N, WARPS_K>::M;
        const int WARP_MASK_K = Warp_masks<WARPS_M, WARPS_N, WARPS_K>::K;

        // The divisor for the warps.
        const int WARP_DIV_M =       1 *       1 * Cta_tile::THREADS_PER_WARP;
        const int WARP_DIV_K = WARPS_M * WARPS_N * Cta_tile::THREADS_PER_WARP;

        // The row/col read by the thread.
        int smem_read_row, smem_read_col;
        if( USE_LDSMT && Base::ROWS_PER_XOR_PATTERN == 8 ) {
            smem_read_row = (tidx & WARP_MASK_K) / WARP_DIV_K * Xmma_tile::XMMAS_K * 16 +
                            (tidx & 0x07) + ((tidx & 0x10) >> 1);
            //smem_read_col = (tidx & 0x07);
            smem_read_col = (tidx & 0x07) ^ ((tidx & 0x08) >> 3);
        } else if( USE_LDSMT && Base::ROWS_PER_XOR_PATTERN == 4 ) {
            smem_read_row = (tidx & WARP_MASK_K) / WARP_DIV_K * Xmma_tile::XMMAS_K * 8 +
                            (tidx & 0x06) / 2 +  + ((tidx & 0x10) >> 2);
            smem_read_col = (tidx & 0x01) * 4 + (tidx & 0x06) / 2;
        } else if( USE_LDSMT && Base::ROWS_PER_XOR_PATTERN == 2 ) {
            smem_read_row = (tidx & WARP_MASK_K) / WARP_DIV_K * Xmma_tile::XMMAS_K * 4 +
                            (tidx & 0x04) / 4 + ((tidx & 0x10) >> 3);
            smem_read_col = (tidx & 0x03) * 2 + (tidx & 0x04) / 4;
        } else if( Base::ROWS_PER_XOR_PATTERN == 4 && Base::COLS_PER_XOR_PATTERN == 2 ) {
            smem_read_row = (tidx & WARP_MASK_K) / WARP_DIV_K * Xmma_tile::XMMAS_K * 8 +
                            (tidx & 0x03);
            smem_read_col = (tidx & 0x1c) / 4 + (tidx & 0x03) * 8;
        } else {
            assert(false);
        }

        if( USE_LDSMT ) {
            if(WARPS_M == 2){
                int stride = (tidx & WARP_MASK_M) / WARP_DIV_M * Xmma_tile::M_PER_XMMA_PER_CTA * 4;
                this->smem_read_offset_ = smem_read_row*Base::BYTES_PER_ROW + smem_read_col*BYTES_PER_LDS + stride;
            } else if(WARPS_M == 4){
                int stride = (tidx & WARP_MASK_M) / WARP_DIV_M * Xmma_tile::M_PER_XMMA_PER_CTA * 2;
                this->smem_read_offset_ = smem_read_row*Base::BYTES_PER_ROW + smem_read_col*BYTES_PER_LDS + stride;
            } else {
                assert(false);
            }
        } else {
            smem_read_col ^= (tidx & WARP_MASK_M) / WARP_DIV_M * 64;
            this->smem_read_offset_ = smem_read_row*Base::BYTES_PER_ROW + smem_read_col*BYTES_PER_LDS;
        }

        // Swizzle the column for the 2nd halfwarp.
        //smem_read_col ^= (tidx & WARP_MASK_M) / WARP_DIV_M * 2 + (tidx & 0x08) / 8;
        // The shared memory offset.
        //this->smem_read_offset_ = smem_read_row*Base::BYTES_PER_ROW + smem_read_col*BYTES_PER_LDS;
    }

    // Load from shared memory.
    inline __device__ void load(Fragment (&a)[Xmma_tile::XMMAS_M], int ki) {
        // The size of each element in bits.
        const int BITS_PER_ELT = Traits::BITS_PER_ELEMENT_A;
        // The size in bytes of the data needed to compute an XMMA per CTA.
        const int BYTES_PER_XMMA = Xmma_tile::M_PER_XMMA * BITS_PER_ELT / 8;

        #pragma unroll
        for( int mi = 0; mi < Xmma_tile::XMMAS_M; ++mi ) {

            if(!USE_LDSMT) {
                // Prepare the offset.
                int offset = ki * Base::ROWS_PER_XOR_PATTERN * 2 * Base::BYTES_PER_ROW;
                offset += this->smem_read_offset_ ^ (mi  ) * BYTES_PER_XMMA;

                // Load the data using LDSM.MT88.4 or 4x LDS.32.
                uint32_t ptr = this->smem_ + this->smem_read_buffer_ + offset;
                uint4 tmp;
                lds(tmp.x, (ptr     ) + 0*Base::BYTES_PER_ROW);
                lds(tmp.y, (ptr ^ 32) + 0*Base::BYTES_PER_ROW);
                lds(tmp.z, (ptr     ) + 4*Base::BYTES_PER_ROW);
                lds(tmp.w, (ptr ^ 32) + 4*Base::BYTES_PER_ROW);

                // Store those values in the fragment.
                a[mi].reg(0) = tmp.x;
                a[mi].reg(1) = tmp.y;
                a[mi].reg(2) = tmp.z;
                a[mi].reg(3) = tmp.w;

            } else{
                // Prepare the offset.
                int offset = ki * Base::ROWS_PER_XOR_PATTERN * 2 * Base::BYTES_PER_ROW;

                offset += (this->smem_read_offset_ ^ (BYTES_PER_LDS * (2 * mi)));

                // Load the data using LDSM.MT88.2.
                uint4 tmp;
                ldsmt(tmp, this->smem_ + this->smem_read_buffer_ + offset);
                a[mi].reg(0) = tmp.x;
                a[mi].reg(1) = tmp.y;
                a[mi].reg(2) = tmp.z;
                a[mi].reg(3) = tmp.w;
            }
        }
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    // The instruction traits.
    typename Traits,
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The size of the STS.
    int BYTES_PER_STS,
    // The number of buffers per tile.
    int BUFFERS_PER_TILE,
    // Us or not predicates
    bool USE_PREDICATES = true,
    // How many rows to use for the XOR pattern to avoid bank conflicts?
    int ROWS_PER_XOR_PATTERN_ = Rows_per_xor_pattern_ampere_a<Traits, Cta_tile::HALF_K>::VALUE
>
struct Smem_tile_sparse_ampere_row_a
    : public Smem_tile_without_skews<Cta_tile,
                                     Cta_tile::M,
                                     Cta_tile::HALF_K,
                                     Traits::BITS_PER_ELEMENT_A,
                                     BYTES_PER_STS,
                                     BUFFERS_PER_TILE,
                                     0,
                                     ROWS_PER_XOR_PATTERN_,
                                     1,
                                     USE_PREDICATES> {

    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;
    // The base class.
    using Base = Smem_tile_without_skews<Cta_tile,
                                         Cta_tile::M,
                                         Cta_tile::HALF_K,
                                         Traits::BITS_PER_ELEMENT_A,
                                         BYTES_PER_STS,
                                         BUFFERS_PER_TILE,
                                         0,
                                         ROWS_PER_XOR_PATTERN_,
                                         1,
                                         USE_PREDICATES>;
    // The fragment.
    using Fragment = Fragment_a<Traits, Row>;

    // The size of a single LDS in bytes.
    enum { BYTES_PER_LDS = 16 };

    // Ctor.
    inline __device__ Smem_tile_sparse_ampere_row_a(char *smem, int tidx)
        : Base(smem, tidx) {

        // For documentation on the layout, see doc/xmma_smem_layout.xlsx.

        // The number of warps.
        const int WARPS_M = Cta_tile::WARPS_M;
        const int WARPS_N = Cta_tile::WARPS_N;
        const int WARPS_K = Cta_tile::WARPS_K;

        // The masks to select the warps.
        const int WARP_MASK_M = Warp_masks<WARPS_M, WARPS_N, WARPS_K>::M;
        const int WARP_MASK_K = Warp_masks<WARPS_M, WARPS_N, WARPS_K>::K;

        // The divisor for the warps.
        const int WARP_DIV_M =       1 *       1 * Cta_tile::THREADS_PER_WARP;
        const int WARP_DIV_K = WARPS_M * WARPS_N * Cta_tile::THREADS_PER_WARP;

        // The row and column read by the thread.
        int smem_read_row, smem_read_col;
        if( Base::ROWS_PER_XOR_PATTERN == 8 ) {
            smem_read_row = (tidx & WARP_MASK_M) / WARP_DIV_M * Xmma_tile::M_PER_XMMA / 1 +
                            (tidx & 0x0f);
            smem_read_col = ((tidx & 0x07) ^ ((tidx & 0x10) >> 4));
        } else if( Base::ROWS_PER_XOR_PATTERN == 4 ) {
            smem_read_row = (tidx & 0x0e) / 2;
            smem_read_col = (((tidx & 0x06) / 2 + (tidx & 0x01) * 4) ^ ((tidx & 0x10) >> 4));
        } else if( Base::ROWS_PER_XOR_PATTERN == 2 ) {
            smem_read_row = (tidx & WARP_MASK_M) / WARP_DIV_M * Xmma_tile::M_PER_WARP / 4 +
                   (tidx & 0x0c) / 4;
            smem_read_col = (((tidx & 0x04) / 4 + (tidx & 0x03) * 2) ^ ((tidx & 0x10) >> 4));
        } else {
            assert(false);
        }

        // We "swap" the block for the second warp working on the in-CTA split-K.
        if( WARPS_K == 2 ) {
            assert(Base::ROWS_PER_XOR_PATTERN != 2);
            smem_read_col ^= (tidx & WARP_MASK_K) / WARP_DIV_K * Xmma_tile::XMMAS_K * 2;
        } else if( WARPS_K > 1 ) {
            assert(false);
        }

        // The shared memory offset.
        int xmma_stride = ((tidx / Cta_tile::THREADS_PER_WARP) % WARPS_M) *
            Traits::ROW_STRIDE_GROUP * Cta_tile::K / 2 *
            Base::BITS_PER_ELEMENT / 8;
        //smem_read_row = smem_read_row % 8;

        // The shared memory offset.
        this->smem_read_offset_ = smem_read_row*Base::BYTES_PER_ROW
            + smem_read_col*BYTES_PER_LDS + xmma_stride;
    }

    // Load from shared memory.
    inline __device__ void load(Fragment (&a)[Xmma_tile::XMMAS_M], int ki) {

        #pragma unroll
        for( int mi = 0; mi < Xmma_tile::XMMAS_M; ++mi ) {
            int offset = mi * Xmma_tile::M_PER_XMMA *
                              Cta_tile::HALF_K *
                              Base::BITS_PER_ELEMENT / 8;

            // Load using LDSM.M88.4.
            uint4 tmp;
            ldsm(tmp, this->smem_ + (this->smem_read_offset_
                + this->smem_read_buffer_ + offset));

            a[mi].reg(0) = tmp.x;
            a[mi].reg(1) = tmp.y;
            a[mi].reg(2) = tmp.z;
            a[mi].reg(3) = tmp.w;
   
        }

        // Move the offset to the next possition. See doc/xmma_smem_layout.xlsx.
        if( Base::ROWS_PER_XOR_PATTERN == 8 ) {
            if( Xmma_tile::XMMAS_K == 4 ) {
                this->smem_read_offset_ ^= ((ki % 2 == 0) ? 1 : 3) * 2 * BYTES_PER_LDS;
            } else if( Xmma_tile::XMMAS_K == 2) {
                this->smem_read_offset_ ^= 2 * BYTES_PER_LDS;
            }
        } else if( Base::ROWS_PER_XOR_PATTERN == 4 ) {
            this->smem_read_offset_ ^= 2 * BYTES_PER_LDS;
        }

    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    // The instruction traits.
    typename Traits,
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The size of the STS.
    int BYTES_PER_STS,
    // The number of buffers per tile.
    int BUFFERS_PER_TILE,
    // Us or not predicates
    bool USE_PREDICATES = true,
    // How many rows to use for the XOR pattern to avoid bank conflicts?
    int ROWS_PER_XOR_PATTERN_ = Rows_per_xor_pattern_ampere_b<Traits, Cta_tile::K>::VALUE
>
struct Smem_tile_sparse_ampere_col_b : public Smem_tile_without_skews<Cta_tile,
                                                               Cta_tile::N,
                                                               Cta_tile::K,
                                                               Traits::BITS_PER_ELEMENT_B,
                                                               BYTES_PER_STS,
                                                               BUFFERS_PER_TILE,
                                                               0,
                                                               ROWS_PER_XOR_PATTERN_,
                                                               1,
                                                               USE_PREDICATES> {

    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;
    // The base class.
    using Base = Smem_tile_without_skews<Cta_tile,
                                         Cta_tile::N,
                                         Cta_tile::K,
                                         Traits::BITS_PER_ELEMENT_B,
                                         BYTES_PER_STS,
                                         BUFFERS_PER_TILE,
                                         0,
                                         ROWS_PER_XOR_PATTERN_,
                                         1,
                                         USE_PREDICATES>;
    // The fragment.
    using Fragment = Fragment_b<Traits, Col>;

    // The size of a single LDS in bytes.
    enum { BYTES_PER_LDS = 16 };

    // Ctor.
    inline __device__ Smem_tile_sparse_ampere_col_b(char *smem, int tidx)
        : Base(smem, tidx) {

        // For documentation on the layout, see doc/xmma_smem_layout.xlsx.

        // The number of warps.
        const int WARPS_M = Cta_tile::WARPS_M;
        const int WARPS_N = Cta_tile::WARPS_N;
        const int WARPS_K = Cta_tile::WARPS_K;

        // The masks to select the warps.
        const int WARP_MASK_N = Warp_masks<WARPS_M, WARPS_N, WARPS_K>::N;
        const int WARP_MASK_K = Warp_masks<WARPS_M, WARPS_N, WARPS_K>::K;

        // The divisor for the warps.
        const int WARP_DIV_N = WARPS_M *       1 * Cta_tile::THREADS_PER_WARP;
        const int WARP_DIV_K = WARPS_M * WARPS_N * Cta_tile::THREADS_PER_WARP;

        // The row and column read by the thread.
        int smem_read_row, smem_read_col;
        if( Base::ROWS_PER_XOR_PATTERN == 8 ) {
            smem_read_row = (tidx & WARP_MASK_N) / WARP_DIV_N * Xmma_tile::N_PER_XMMA / 1 +
                            (tidx & 0x07);
            smem_read_col = ((tidx & 0x07) ^ ((tidx & 0x18) >> 3));
            //smem_read_row = (tidx & WARP_MASK_N) / WARP_DIV_N * Xmma_tile::N_PER_XMMA / 1 +
            //(tidx & 0x07) + ((tidx & 0x10) >> 1);
            //smem_read_col = ((tidx & 0x07) ^ ((tidx & 0x08) >> 3));
        } else if( Base::ROWS_PER_XOR_PATTERN == 4 ) {
            smem_read_row = (tidx & WARP_MASK_N) / WARP_DIV_N * Xmma_tile::N_PER_XMMA / 2 +
                            (tidx & 0x06) / 2 + ((tidx & 0x10) >> 2);
            smem_read_col = (((tidx & 0x06) / 2 + (tidx & 0x01) * 4) ^ ((tidx & 0x08) >> 3));
        } else if( Base::ROWS_PER_XOR_PATTERN == 2 ) {
            smem_read_row = (tidx & WARP_MASK_N) / WARP_DIV_N * Xmma_tile::N_PER_XMMA / 4 +
                            (tidx & 0x04) / 4 + ((tidx & 0x10) >> 3);
            smem_read_col = (((tidx & 0x04) / 4 + (tidx & 0x03) * 2) ^ ((tidx & 0x08) >> 3));
        } else {
            assert(false);
        }

        // We "swap" the block for the second warp working on the in-CTA split-K.
        if( WARPS_K == 2 ) {
            assert(Base::ROWS_PER_XOR_PATTERN != 2);
            smem_read_col ^= (tidx & WARP_MASK_K) / WARP_DIV_K * 2 * Xmma_tile::XMMAS_K;
        } else if( WARPS_K > 1 ) {
            assert(false);
        }

        // The shared memory offset.
        this->smem_read_offset_ = smem_read_row*Base::BYTES_PER_ROW + smem_read_col*BYTES_PER_LDS;
    }

    // Load from shared memory.
    inline __device__ void load(Fragment (&b)[Xmma_tile::XMMAS_N], int ki) {

        int shift_offset = 8 * Base::BYTES_PER_ROW;

        #pragma unroll
        for( int ni = 0; ni < Xmma_tile::XMMAS_N; ++ ni ) {
            int offset = ni * Xmma_tile::N_PER_XMMA_PER_CTA *
                              Cta_tile::K *
                              Base::BITS_PER_ELEMENT / 8;
            // Load using LDSM.M88.4.
            uint4 tmp;
            ldsm(tmp, this->smem_ + (this->smem_read_offset_ +
                            this->smem_read_buffer_ + offset));
            b[ni].reg(0) = tmp.x;
            b[ni].reg(1) = tmp.y;
            b[ni].reg(2) = tmp.z;
            b[ni].reg(3) = tmp.w;

            ldsm(tmp, this->smem_ + this->smem_read_offset_ +
                            shift_offset + this->smem_read_buffer_ + offset);

            b[ni].reg(4) = tmp.x;
            b[ni].reg(5) = tmp.y;
            b[ni].reg(6) = tmp.z;
            b[ni].reg(7) = tmp.w;

        }

        if (Base::ROWS_PER_XOR_PATTERN == 8) {
            this->smem_read_offset_ = (this->smem_read_offset_
                ^ (2 * BYTES_PER_LDS)) ^ (6 * BYTES_PER_LDS);
        }

        // These are the original offset manipulation logic, I leave here as referernce
/*
        // Move the offset to the next possition. See doc/xmma_smem_layout.xlsx.
        if( Base::ROWS_PER_XOR_PATTERN == 8 ) {
            if( Xmma_tile::XMMAS_K == 2 ) {
                this->smem_read_offset_ ^= 2 * BYTES_PER_LDS;
            } else if( Xmma_tile::XMMAS_K == 4 ) {
                this->smem_read_offset_ ^= ((ki % 2 == 0) ? 1 : 3) * 2 * BYTES_PER_LDS;
            }
        } else if( Base::ROWS_PER_XOR_PATTERN == 4 ) {
            this->smem_read_offset_ ^= 2 * BYTES_PER_LDS;
        }
*/
        // These are the original offset manipulation logic, I leave here as referernce
    }
    inline __device__ void tf32_pipe_colwert(Fragment (&b)[Xmma_tile::XMMAS_N][1]){
    }

    inline __device__ void pipe_load(Fragment (&b)[1], int ki, int idx) {

        int shift_offset = 8 * Base::BYTES_PER_ROW ;
        int offset = idx * Xmma_tile::N_PER_XMMA_PER_CTA *
                           Cta_tile::K *
                           Base::BITS_PER_ELEMENT / 8;

        // Load using LDSM.M88.4.
        uint4 tmp;
        ldsm(tmp, this->smem_ + (this->smem_read_offset_ +
                        this->smem_read_buffer_ + offset));
        b[0].reg(0) = tmp.x;
        b[0].reg(1) = tmp.y;
        b[0].reg(2) = tmp.z;
        b[0].reg(3) = tmp.w;

        ldsm(tmp, this->smem_ + this->smem_read_offset_ +
                    shift_offset + this->smem_read_buffer_ + offset);

        b[0].reg(4) = tmp.x;
        b[0].reg(5) = tmp.y;
        b[0].reg(6) = tmp.z;
        b[0].reg(7) = tmp.w;

        if (idx == (Xmma_tile::XMMAS_N_DIV - 1))
            this->smem_read_offset_ = (this->smem_read_offset_ ^ (2 * BYTES_PER_LDS)) ^ (6 * BYTES_PER_LDS);
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    // The instruction traits.
    typename Traits,
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The size of the STS.
    int BYTES_PER_STS,
    // The number of buffers per tile.
    int BUFFERS_PER_TILE,
    // Us or not predicates
    bool USE_PREDICATES = true,
    // How many rows to use for the XOR pattern to avoid bank conflicts?
    int ROWS_PER_XOR_PATTERN_ = 2
>
struct Smem_tile_sparse_ampere_interleaved_col_b
    : public Smem_tile_sparse_interleaved<Cta_tile,
                                    Cta_tile::N,
                                          Cta_tile::K,
                                          Traits::BITS_PER_ELEMENT_B,
                                          BYTES_PER_STS,
                                          BUFFERS_PER_TILE,
                                          0,
                                          ROWS_PER_XOR_PATTERN_,
                                          1,
                                          USE_PREDICATES> {

    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;
    // The base class.
    using Base = Smem_tile_sparse_interleaved<Cta_tile,
                                         Cta_tile::N,
                                         Cta_tile::K,
                                         Traits::BITS_PER_ELEMENT_B,
                                         BYTES_PER_STS,
                                         BUFFERS_PER_TILE,
                                         0,
                                         ROWS_PER_XOR_PATTERN_,
                                         1,
                                         USE_PREDICATES>;
    // The fragment.
    using Fragment = Fragment_b<Traits, Col>;

    // The size of a single LDS in bytes.
    enum { BYTES_PER_LDS = 16 };

    // Ctor.
    inline __device__ Smem_tile_sparse_ampere_interleaved_col_b(char *smem, int tidx)
        : Base(smem, tidx) {

        // For documentation on the layout, see doc/xmma_smem_layout.xlsx.

        // The number of warps.
        const int WARPS_M = Cta_tile::WARPS_M;
        const int WARPS_N = Cta_tile::WARPS_N;
        const int WARPS_K = Cta_tile::WARPS_K;

        // The masks to select the warps.
        const int WARP_MASK_N = Warp_masks<WARPS_M, WARPS_N, WARPS_K>::N;

        // The divisor for the warps.
        const int WARP_DIV_N = WARPS_M *       1 * Cta_tile::THREADS_PER_WARP;

        // The row and column read by the thread.
        int smem_read_row, smem_read_col;
        if( Base::ROWS_PER_XOR_PATTERN == 2 ) {

            if (Base::ROWS_PER_STS == 4) {
                smem_read_row = ((tidx & WARP_MASK_N) / WARP_DIV_N) *
                    (Xmma_tile::N_PER_XMMA / 4) * 4;

                smem_read_row += (tidx & 0x04) / 4
                            + ((tidx & 0x10) >> 4) * (8 / 4);
            } else if (Base::ROWS_PER_STS == 8) {
                smem_read_row = ((tidx & WARP_MASK_N) / WARP_DIV_N) *
                    (Xmma_tile::N_PER_XMMA / 4) * 4;

                smem_read_row += (tidx & 0x04) / 4
                            + ((tidx & 0x10) >> 4) * (Base::ROWS_PER_STS / 4);
            } else {
                int tmp = Base::ROWS_PER_STS / Xmma_tile::N_PER_XMMA;
                smem_read_row = (((tidx & WARP_MASK_N) / WARP_DIV_N) / tmp) *
                    (Base::ROWS_PER_STS / 4) * 4;

                smem_read_row += (((tidx & WARP_MASK_N) / WARP_DIV_N) % tmp) *
                    (Xmma_tile::N_PER_XMMA / 4);

                smem_read_row += (tidx & 0x04) / 4
                            + ((tidx & 0x10) >> 4) * (Base::ROWS_PER_STS / 4);
            }

            smem_read_col = (((tidx & 0x04) / 4 + (tidx & 0x03) * 2) ^ ((tidx & 0x08) >> 3));

        } else {
            assert(false);
        }

        // The shared memory offset.
        this->smem_read_offset_ = smem_read_row*Base::BYTES_PER_ROW +
            smem_read_col*BYTES_PER_LDS;
    }

    // Load from shared memory.
    inline __device__ void load(Fragment (&b)[Xmma_tile::XMMAS_N], int ki) {

        int shift_offset = 0;
        if (Base::ROWS_PER_STS == 8 || Base::ROWS_PER_STS == 4) {
            shift_offset = 8 * Cta_tile::K * Base::BITS_PER_ELEMENT / 8;
        } else {
            shift_offset = 2 * Base::BYTES_PER_ROW;
        }

        #pragma unroll
        for( int ni = 0; ni < Xmma_tile::XMMAS_N; ++ ni ) {
            int offset = ni * Xmma_tile::N_PER_XMMA_PER_CTA *
                        Cta_tile::K *
                        Base::BITS_PER_ELEMENT / 8;
            if (Base::ROWS_PER_STS == 4) {
                offset += ki * 2 * (2*Base::ROWS_PER_STS / 4) * Base::BYTES_PER_ROW;
            } else {
                offset += ki * 2 * (Base::ROWS_PER_STS / 4) * Base::BYTES_PER_ROW;
            }
            // Load using LDSM.M88.4.
            uint4 tmp;
            ldsm(tmp, this->smem_ + (this->smem_read_offset_ +
                            this->smem_read_buffer_ + offset));
            b[ni].reg(0) = tmp.x;
            b[ni].reg(1) = tmp.y;
            b[ni].reg(2) = tmp.z;
            b[ni].reg(3) = tmp.w;

            ldsm(tmp, this->smem_ + this->smem_read_offset_ +
                            shift_offset + this->smem_read_buffer_ + offset);

            b[ni].reg(4) = tmp.x;
            b[ni].reg(5) = tmp.y;
            b[ni].reg(6) = tmp.z;
            b[ni].reg(7) = tmp.w;
        }
    }

};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    // The instruction traits.
    typename Traits,
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The size of the STS.
    int BYTES_PER_STS,
    // The number of buffers per tile.
    int BUFFERS_PER_TILE,
    // How many rows to use for the XOR pattern to avoid bank conflicts?
    int ROWS_PER_XOR_PATTERN_ = Rows_per_xor_pattern_ampere_row_b<Traits, Cta_tile::N>::VALUE,
    // How many cols to use for the XOR pattern to avoid bank conflicts?
    int COLS_PER_XOR_PATTERN_ = Cols_per_xor_pattern_ampere<Traits>::VALUE
>
struct Smem_tile_sparse_ampere_row_b : public Smem_tile_without_skews<Cta_tile,
                                                               Cta_tile::K,
                                                               Cta_tile::N,
                                                               Traits::BITS_PER_ELEMENT_B,
                                                               BYTES_PER_STS,
                                                               BUFFERS_PER_TILE,
                                                               0,
                                                               ROWS_PER_XOR_PATTERN_,
                                                               COLS_PER_XOR_PATTERN_> {

    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;
    // The base class.
    using Base = Smem_tile_without_skews<Cta_tile,
                                         Cta_tile::K,
                                         Cta_tile::N,
                                         Traits::BITS_PER_ELEMENT_B,
                                         BYTES_PER_STS,
                                         BUFFERS_PER_TILE,
                                         0,
                                         ROWS_PER_XOR_PATTERN_,
                                         COLS_PER_XOR_PATTERN_>;
    // The fragment.
    using Fragment = Fragment_b<Traits, Row>;

    // Can we use LDSM? No if the data type is 32-bit large.
    enum { USE_LDSMT = Traits::BITS_PER_ELEMENT_B == 16 };
    // The size of a single LDS in bytes.
    enum { BYTES_PER_LDS = USE_LDSMT ? 16 : 4 };
    //enum { BYTES_PER_LDS = 16 };
    // The number of elements per LDS.
    enum { ELEMENTS_PER_LDS = BYTES_PER_LDS * 8 / Traits::BITS_PER_ELEMENT_B };


    // Ctor.
    inline __device__ Smem_tile_sparse_ampere_row_b(char *smem, int tidx)
        : Base(smem, tidx) {

        // For documentation on the layout, see doc/xmma_smem_layout.xlsx.

        // The number of warps.
        const int WARPS_M = Cta_tile::WARPS_M;
        const int WARPS_N = Cta_tile::WARPS_N;
        const int WARPS_K = Cta_tile::WARPS_K;

        // The masks to select the warps.
        const int WARP_MASK_N = Warp_masks<WARPS_M, WARPS_N, WARPS_K>::N;
        const int WARP_MASK_K = Warp_masks<WARPS_M, WARPS_N, WARPS_K>::K;

        // The divisor for the warps.
        const int WARP_DIV_N = WARPS_M *       1 * Cta_tile::THREADS_PER_WARP;
        const int WARP_DIV_K = WARPS_M * WARPS_N * Cta_tile::THREADS_PER_WARP;

        // The row/col read by the thread.
        int smem_read_row, smem_read_col;
        if( USE_LDSMT && Base::ROWS_PER_XOR_PATTERN == 8 ) {
            smem_read_row = (tidx & WARP_MASK_K) / WARP_DIV_K * Xmma_tile::XMMAS_K * 16 +
                            (tidx & 0x1F);
            smem_read_col = (tidx & 0x07) ^ (tidx & WARP_MASK_N) / WARP_DIV_N * 2;
            //smem_read_row = (tidx & WARP_MASK_K) / WARP_DIV_K * Xmma_tile::XMMAS_K * 16 +
                            //(tidx & 0x07) + (tidx & 0x08);
            //smem_read_col = (tidx & 0x07);
        } else if( USE_LDSMT && Base::ROWS_PER_XOR_PATTERN == 4 ) {
            smem_read_row = (tidx & WARP_MASK_K) / WARP_DIV_K * Xmma_tile::XMMAS_K * 8 +
                            (tidx & 0x06) / 2 + (tidx & 0x08) / 2;
            smem_read_col = (tidx & 0x01) * 4 + (tidx & 0x06) / 2;
        } else if( USE_LDSMT && Base::ROWS_PER_XOR_PATTERN == 2 ) {
            smem_read_row = (tidx & WARP_MASK_K) / WARP_DIV_K * Xmma_tile::XMMAS_K * 4 +
                            (tidx & 0x04) / 4 + (tidx & 0x08) / 4;
            smem_read_col = (tidx & 0x03) * 2 + (tidx & 0x04) / 4;
        } else if( Base::ROWS_PER_XOR_PATTERN == 4 && Base::COLS_PER_XOR_PATTERN == 2 ) {
            if ( Cta_tile::GROUPS > 1 && WARPS_K == 1 && WARPS_N > 1 ) {
                smem_read_row = (tidx & WARP_MASK_N) / WARP_DIV_N * (Cta_tile::N / WARPS_N) +
                                (tidx & 0x03);
            } else {
                smem_read_row = (tidx & WARP_MASK_K) / WARP_DIV_K * Xmma_tile::XMMAS_K * 8 +
                                (tidx & 0x03);
            }
            smem_read_col = (tidx & 0x1c) / 4 + (tidx & 0x03) * 8;

            // Only for non-group.
            if ( Cta_tile::GROUPS == 1 ) {
                smem_read_col ^= (tidx & WARP_MASK_N) / WARP_DIV_N * 16;
            }

        } else {
            assert(false);
        }
        
        // Each half-warp applies a different XOR pattern -- see the Excel document.
        //smem_read_col ^= (tidx & WARP_MASK_N) / WARP_DIV_N * 2 + (tidx & 0x10) / 16;
        //smem_read_row = smem_read_row + ((tidx % Cta_tile::THREADS_PER_WARP) / (Cta_tile::THREADS_PER_WARP / 2)) * 16;
        //smem_read_col = smem_read_col ^ ((tidx % Cta_tile::THREADS_PER_WARP) / (Cta_tile::THREADS_PER_WARP / 2));

        // The shared memory offset.
        this->smem_read_offset_ = smem_read_row*Base::BYTES_PER_ROW + smem_read_col*BYTES_PER_LDS;
    }

    // Load from shared memory.
    inline __device__ void load(Fragment (&b)[Xmma_tile::XMMAS_N], int ki) {
        #pragma unroll
        for( int ni = 0; ni < Xmma_tile::XMMAS_N; ++ni ) {
            // Prepare the offset.
            int offset = 2 * ki * Base::ROWS_PER_XOR_PATTERN * 2 * Base::BYTES_PER_ROW;
            if( Cta_tile::WARPS_N == 1 ) {
                offset += this->smem_read_offset_;
            } else if( Cta_tile::WARPS_N == 2 ) {
                offset += this->smem_read_offset_ + (ni/2) * Xmma_tile::N_PER_XMMA_PER_CTA*4;
            } else {
                offset += this->smem_read_offset_ + (ni  ) * Xmma_tile::N_PER_XMMA_PER_CTA*2;
            }

            // Load the data using LDSM.MT88.2.
            uint4 tmp, tmp1;
            ldsmt(tmp, this->smem_ + this->smem_read_buffer_ + offset);
            b[ni].reg(0) = tmp.x;
            b[ni].reg(1) = tmp.y;
            b[ni].reg(2) = tmp.z;
            b[ni].reg(3) = tmp.w;
            ldsmt(tmp1, this->smem_ + this->smem_read_buffer_ + offset ^ BYTES_PER_LDS);
            b[ni].reg(4) = tmp1.x;
            b[ni].reg(5) = tmp1.y;
            b[ni].reg(6) = tmp1.z;
            b[ni].reg(7) = tmp1.w;

            // Move the pointer for the next ni. I expect the compiler to not recompute those.
            if( Cta_tile::WARPS_N == 4 ) {
                // Nothing to do!
            } else if( Cta_tile::WARPS_N == 2 ) {
                if( Xmma_tile::XMMAS_N > 1 ) {
                    this->smem_read_offset_ ^= BYTES_PER_LDS * (4);
                }
            } else if( Xmma_tile::XMMAS_N == 4 ) {
                this->smem_read_offset_ ^= BYTES_PER_LDS * (ni % 2 == 0 ? 2 : 6);
            } else if( Xmma_tile::XMMAS_N == 2 ) {
                this->smem_read_offset_ ^= BYTES_PER_LDS * 2;
            } else if( Xmma_tile::XMMAS_N != 1 ) {
                assert(false); // Not implemented!
            }
        }
    }

    inline __device__ void tf32_pipe_colwert(Fragment (&b)[Xmma_tile::XMMAS_N][1]){
    }

    inline __device__ void pipe_load(Fragment (&b)[1], int ki, int idx) {
        // The size of each element in bits.
        const int BITS_PER_ELT = Traits::BITS_PER_ELEMENT_B;

        if(!USE_LDSMT) {
            // The size in bytes of the data needed to compute an XMMA per CTA.
            const int BYTES_PER_XMMA_PER_CTA = Xmma_tile::N_PER_XMMA_PER_CTA * BITS_PER_ELT / 8;
            int offset = ki * Base::ROWS_PER_XOR_PATTERN * 2 * Base::BYTES_PER_ROW * 2;
            offset += this->smem_read_offset_ + (idx  ) * BYTES_PER_XMMA_PER_CTA;
            //printf("tid %d idx %d BYTES_PER_XMMA_PER_CTA %d offset %d\n", threadIdx.x, idx, BYTES_PER_XMMA_PER_CTA, offset);
            uint32_t ptr = this->smem_ + this->smem_read_buffer_ + offset;
            uint4 tmp, tmp1;
            lds(tmp.x, (ptr     ) + 0*Base::BYTES_PER_ROW);
            lds(tmp.y, (ptr     ) + 4*Base::BYTES_PER_ROW);
            lds(tmp.z, (ptr     ) + 8*Base::BYTES_PER_ROW);
            lds(tmp.w, (ptr     ) + 12*Base::BYTES_PER_ROW);

            lds(tmp1.x, (ptr ^ 32) + 0*Base::BYTES_PER_ROW);
            lds(tmp1.y, (ptr ^ 32) + 4*Base::BYTES_PER_ROW);
            lds(tmp1.z, (ptr ^ 32) + 8*Base::BYTES_PER_ROW);
            lds(tmp1.w, (ptr ^ 32) + 12*Base::BYTES_PER_ROW);
            
            // Store those values in the fragment.
            b[0].reg(0) = tmp.x;
            b[0].reg(1) = tmp.y;
            b[0].reg(2) = tmp.z;
            b[0].reg(3) = tmp.w;

            b[0].reg(4) = tmp1.x;
            b[0].reg(5) = tmp1.y;
            b[0].reg(6) = tmp1.z;
            b[0].reg(7) = tmp1.w;         
            
        } else {
                int offset = 2 * ki * Base::ROWS_PER_XOR_PATTERN * 2 * Base::BYTES_PER_ROW;
                if( Cta_tile::WARPS_N == 1 ) {
                    offset += this->smem_read_offset_;
                } else if( Cta_tile::WARPS_N == 2 ) {
                    offset += this->smem_read_offset_ + (idx/2) * Xmma_tile::N_PER_XMMA_PER_CTA*4;
                } else {
                    offset += this->smem_read_offset_ + (idx  ) * Xmma_tile::N_PER_XMMA_PER_CTA*2;
                }

                // Load the data using LDSM.MT88.2.
                uint4 tmp, tmp1;
                ldsmt(tmp, this->smem_ + this->smem_read_buffer_ + offset);

                b[0].reg(0) = tmp.x;
                b[0].reg(1) = tmp.y;
                b[0].reg(2) = tmp.z;
                b[0].reg(3) = tmp.w;

                ldsmt(tmp1, this->smem_ + this->smem_read_buffer_ + offset ^ BYTES_PER_LDS);
                b[0].reg(4) = tmp1.x;
                b[0].reg(5) = tmp1.y;
                b[0].reg(6) = tmp1.z;
                b[0].reg(7) = tmp1.w;

                if( Cta_tile::WARPS_N == 4 ) {
                    // Nothing to do!
                } else if( Cta_tile::WARPS_N == 2 ) {
                    if( Xmma_tile::XMMAS_N > 1 ) {
                        this->smem_read_offset_ ^= BYTES_PER_LDS * (4);
                    }
                } else if( Xmma_tile::XMMAS_N == 4 ) {
                    this->smem_read_offset_ ^= BYTES_PER_LDS * (idx % 2 == 0 ? 2 : 6);
                } else if( Xmma_tile::XMMAS_N == 2 ) {
                    this->smem_read_offset_ ^= BYTES_PER_LDS * 2;
                } else if( Xmma_tile::XMMAS_N != 1 ) {
                    assert(false); // Not implemented!
                }
        }
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Cta_tile>
struct Swizzle_epilogue<Ampere_sphmma_fp32_traits, Cta_tile, Row, 16, true> {

    // The traits.
    using Traits = Ampere_sphmma_fp32_traits;
    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;

    // The number of bytes for key memory instruction.
    enum { BYTES_PER_LDS = 16, BYTES_PER_STS = 8 };
    // The number of rows in shared memory.
    enum { ROWS = Xmma_tile::M_PER_XMMA_PER_CTA * 2 };
    // The size of a single row in bytes.
    enum { BYTES_PER_ROW = Cta_tile::N / 2 * Cta_tile::WARPS_K * sizeof(float) };
    // The skew to avoid bank conflicts.
    enum { BYTES_PER_SKEW = 16 };
    // The number of columns with the skew.
    enum { BYTES_PER_ROW_WITH_SKEW = BYTES_PER_ROW + BYTES_PER_SKEW };
    // The size in bytes in shared memory.
    enum { BYTES_PER_TILE = ROWS * BYTES_PER_ROW_WITH_SKEW };

    // The number of threads per output row. Each thread writes 8 elements.
    enum { THREADS_PER_ROW = Cta_tile::N / 8 };
    // The number of rows written in one STG.128.
    enum { ROWS_PER_STG = Cta_tile::THREADS_PER_CTA / THREADS_PER_ROW };

    // How we see the distribution of data.
    enum { THREADS_PER_XMMA_M = 8, THREADS_PER_XMMA_N = 4 };
    // The number of elements stored per thread.
    enum { M_PER_XMMA_PER_THREAD = 2, N_PER_XMMA_PER_THREAD = 4 };

    // Do we skip the syncthreads in the epilogue? Of course, not :)
    enum { SKIP_SYNCTHREADS = 0 };

    // Ctor.
    inline __device__ Swizzle_epilogue(char *smem, int tidx) : smem_(smem) {

        // The number of warps.
        const int WARPS_M = Cta_tile::WARPS_M;
        const int WARPS_N = Cta_tile::WARPS_N;
        const int WARPS_K = Cta_tile::WARPS_K;

        //
        //     tidx   0: row =  0, col = 0
        //     tidx   1: row =  0, col = 8
        //     tidx   2: row =  1, col = 0
        //     tidx   3: row =  1, col = 8
        //     tidx   4: row =  2, col = 0
        //     tidx   5: row =  2, col = 8
        //     tidx   6: row =  3, col = 0
        //     tidx   7: row =  3, col = 8
        //     tidx   8: row =  4, col = 0
        //     tidx   9: row =  4, col = 8
        //     tidx  10: row =  5, col = 0
        //     tidx  11: row =  5, col = 8
        //     tidx  12: row =  6, col = 0
        //     tidx  13: row =  6, col = 8
        //     tidx  14: row =  7, col = 0
        //     tidx  15: row =  7, col = 8
        //     tidx  16: row =  8, col = 0
        //     tidx  17: row =  8, col = 8
        //     tidx  18: row =  9, col = 0
        //     tidx  19: row =  9, col = 8
        //     tidx  20: row = 10, col = 0
        //     tidx  21: row = 10, col = 8
        //     tidx  22: row = 11, col = 0
        //     tidx  23: row = 11, col = 8
        //     tidx  24: row = 12, col = 0
        //     tidx  25: row = 12, col = 8
        //     tidx  26: row = 13, col = 0
        //     tidx  27: row = 13, col = 8
        //     tidx  28: row = 14, col = 0
        //     tidx  29: row = 14, col = 8
        //     tidx  30: row = 15, col = 0
        //     tidx  31: row = 15, col = 8 

	// The masks to select the warps.
        const int WARP_MASK_M = Warp_masks<WARPS_M, WARPS_N, WARPS_K>::M;
        const int WARP_MASK_N = Warp_masks<WARPS_M, WARPS_N, WARPS_K>::N;
        const int WARP_MASK_K = Warp_masks<WARPS_M, WARPS_N, WARPS_K>::K;

        // The divisor for the warps.
        const int WARP_DIV_M =       1 *       1 * Cta_tile::THREADS_PER_WARP;
        const int WARP_DIV_N = WARPS_M *       1 * Cta_tile::THREADS_PER_WARP;
        const int WARP_DIV_K = WARPS_M * WARPS_N * Cta_tile::THREADS_PER_WARP;
        
        // Compute the row and the column in shared memory. Each warp reads from a 16*16B segment.
        const int smem_write_row = (tidx & WARP_MASK_M) / WARP_DIV_M * 32 +
                                   (tidx & 0x1e) / 2;
        const int smem_write_col = (tidx & WARP_MASK_N) / WARP_DIV_N * Xmma_tile::N_PER_XMMA / 2 * 4 + 
                                   (tidx & WARP_MASK_K) / WARP_DIV_K * Cta_tile::N / 2 * 4 + 
                                   (tidx & 0x01) * 8;

        // The corresponding offset.
        smem_write_offset_ = smem_write_row*BYTES_PER_ROW_WITH_SKEW + smem_write_col;

        // The row and column read by a single thread.
        const int smem_read_row = tidx / THREADS_PER_ROW * 2;
        const int smem_read_col = tidx % THREADS_PER_ROW * BYTES_PER_LDS;

        // The corresponding offset.
        smem_read_offset_ = smem_read_row*BYTES_PER_ROW_WITH_SKEW + smem_read_col;
    }

    // Load from the tile in shared memory.
    template<typename Fragment_post_swizzle>
    inline __device__ void load(Fragment_post_swizzle &dst, int oi) const {
        int offset = oi * ROWS_PER_STG * 2 * BYTES_PER_ROW_WITH_SKEW;
        #pragma unroll
        for( int ki = 0; ki < Cta_tile::WARPS_K; ++ki ) {
            
            // Add the WARP_K factor.
            int offset_0 = offset + ki * Cta_tile::N * 2;

            // The 1st group of 4 floats.
            float4 tmp_0 = *reinterpret_cast<const float4*>(&smem_[smem_read_offset_ + offset_0]);
            dst.reg(ki * 8 + 0) = tmp_0.x;
            dst.reg(ki * 8 + 1) = tmp_0.y;
            dst.reg(ki * 8 + 2) = tmp_0.z;
            dst.reg(ki * 8 + 3) = tmp_0.w;

            // The 2nd group of 4 floats.
            int offset_1 = offset_0 + 1*BYTES_PER_ROW_WITH_SKEW;
            float4 tmp_1 = *reinterpret_cast<const float4*>(&smem_[smem_read_offset_ + offset_1]);
            dst.reg(ki * 8 + 4) = tmp_1.x;
            dst.reg(ki * 8 + 5) = tmp_1.y;
            dst.reg(ki * 8 + 6) = tmp_1.z;
            dst.reg(ki * 8 + 7) = tmp_1.w;
        }
    }

    // Store to the tile in shared memory.
    template<typename Fragment_pre_swizzle>
    inline __device__ void store(int ni, const Fragment_pre_swizzle &c) {
        #pragma unroll
        for( int mi = 0; mi < M_PER_XMMA_PER_THREAD; ++mi ) {
            int offset = mi * 16 * BYTES_PER_ROW_WITH_SKEW +
                         ni * Xmma_tile::N_PER_XMMA_PER_CTA / 2 * sizeof(float);
            reinterpret_cast<float2*>(&smem_[smem_write_offset_ + offset +  0])[0] = 
                make_float2(c.reg(4 * mi + 0), c.reg(4 * mi + 1));
            reinterpret_cast<float2*>(&smem_[smem_write_offset_ + offset + 16])[0] = 
                make_float2(c.reg(4 * mi + 2), c.reg(4 * mi + 3));
        }
    }

    // The shared memory pointer in bytes.
    char *smem_;
    // The write offset.
    int smem_write_offset_;
    // The read offset.
    int smem_read_offset_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// S P A R S E  H M M A . F 3 2
//
////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The size of the STS.
    int BYTES_PER_STS,
    // The number of buffers per tile.
    int BUFFERS_PER_TILE
>
struct Smem_tile_e<Ampere_sphmma_fp32_traits, Cta_tile, Row, BYTES_PER_STS, BUFFERS_PER_TILE>
    : public Smem_tile_sparse_ampere_row_e<Ampere_sphmma_fp32_traits,
                                    Cta_tile,
                                    BYTES_PER_STS,
                                    BUFFERS_PER_TILE,
                                    false> {

    // The traits class.
    using Traits = Ampere_sphmma_fp32_traits;
    // The base class.
    using Base = Smem_tile_sparse_ampere_row_e<Traits, Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE, false>;

    // Ctor.
    inline __device__ Smem_tile_e(char *smem, int tidx)
        : Base(smem, tidx) {
    }
};

template<
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The size of the STS.
    int BYTES_PER_STS,
    // The number of buffers per tile.
    int BUFFERS_PER_TILE
>
struct Smem_tile_lds_e<Ampere_sphmma_fp32_traits, Cta_tile, Row, BYTES_PER_STS, BUFFERS_PER_TILE>
    : public Smem_tile_sparse_ampere_row_e<Ampere_sphmma_fp32_traits,
                                    Cta_tile,
                                    BYTES_PER_STS,
                                    BUFFERS_PER_TILE,
                                    true> {

    // The traits class.
    using Traits = Ampere_sphmma_fp32_traits;
    // The base class.
    using Base = Smem_tile_sparse_ampere_row_e<Traits, Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE, true>;

    // Ctor.
    inline __device__ Smem_tile_lds_e(char *smem, int tidx)
        : Base(smem, tidx) {
    }
};


////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The size of the STS.
    int BYTES_PER_STS,
    // The number of buffers per tile.
    int BUFFERS_PER_TILE,
    // Use or not predicates
    bool USE_PREDICATES
>
struct Smem_tile_a<Ampere_sphmma_fp32_traits, Cta_tile, Row, BYTES_PER_STS, BUFFERS_PER_TILE, USE_PREDICATES>
    : public Smem_tile_sparse_ampere_row_a<Ampere_sphmma_fp32_traits,
                                    Cta_tile,
                                    BYTES_PER_STS,
                                    BUFFERS_PER_TILE,
                                    USE_PREDICATES> {
    // The traits class.
    using Traits = Ampere_sphmma_fp32_traits;
    // The base class.
    using Base = Smem_tile_sparse_ampere_row_a<Traits, Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE, USE_PREDICATES>;

    // Ctor.
    inline __device__ Smem_tile_a(char *smem, int tidx) : Base(smem, tidx) {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The size of the STS.
    int BYTES_PER_STS,
    // The number of buffers per tile.
    int BUFFERS_PER_TILE
>
struct Smem_tile_a<Ampere_sphmma_fp32_traits, Cta_tile, Col, BYTES_PER_STS, BUFFERS_PER_TILE>
    : public Smem_tile_sparse_ampere_col_a<Ampere_sphmma_fp32_traits,
                                    Cta_tile,
                                    BYTES_PER_STS,
                                    BUFFERS_PER_TILE> {

    // The traits class.
    using Traits = Ampere_sphmma_fp32_traits;
    // The base class.
    using Base = Smem_tile_sparse_ampere_col_a<Traits, Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE>;

    // Ctor.
    inline __device__ Smem_tile_a(char *smem, int tidx) : Base(smem, tidx) {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The size of the STS.
    int BYTES_PER_STS,
    // The number of buffers per tile.
    int BUFFERS_PER_TILE,
    // Use or not predicates
    bool USE_PREDICATES
>
struct Smem_tile_b<Ampere_sphmma_fp32_traits, Cta_tile, Col, BYTES_PER_STS, BUFFERS_PER_TILE, USE_PREDICATES>
    : public Smem_tile_sparse_ampere_col_b<Ampere_sphmma_fp32_traits,
                                    Cta_tile,
                                    BYTES_PER_STS,
                                    BUFFERS_PER_TILE,
                                    USE_PREDICATES> {
    // The traits class.
    using Traits = Ampere_sphmma_fp32_traits;
    // The base class.
    using Base = Smem_tile_sparse_ampere_col_b<Traits, Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE, USE_PREDICATES>;

    // Ctor.
    inline __device__ Smem_tile_b(char *smem, int tidx) : Base(smem, tidx) {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The size of the STS.
    int BYTES_PER_STS,
    // The number of buffers per tile.
    int BUFFERS_PER_TILE
>
struct Smem_tile_b<Ampere_sphmma_fp32_traits, Cta_tile, Row, BYTES_PER_STS, BUFFERS_PER_TILE>
    : public Smem_tile_sparse_ampere_row_b<Ampere_sphmma_fp32_traits,
                                    Cta_tile,
                                    BYTES_PER_STS,
                                    BUFFERS_PER_TILE> {

    // The traits class.
    using Traits = Ampere_sphmma_fp32_traits;
    // The base class.
    using Base = Smem_tile_sparse_ampere_row_b<Traits, Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE>;

    // Ctor.
    inline __device__ Smem_tile_b(char *smem, int tidx) : Base(smem, tidx) {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// S P A R S E  H M M A . F 1 6
//
////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The size of the STS.
    int BYTES_PER_STS,
    // The number of buffers per tile.
    int BUFFERS_PER_TILE
>
struct Smem_tile_e<Ampere_sphmma_fp16_traits, Cta_tile, Row, BYTES_PER_STS, BUFFERS_PER_TILE>
    : public Smem_tile_sparse_ampere_row_e<Ampere_sphmma_fp16_traits,
                                    Cta_tile,
                                    BYTES_PER_STS,
                                    BUFFERS_PER_TILE,
                                    false> {

    // The traits class.
    using Traits = Ampere_sphmma_fp16_traits;
    // The base class.
    using Base = Smem_tile_sparse_ampere_row_e<Traits, Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE, false>;

    // Ctor.
    inline __device__ Smem_tile_e(char *smem, int tidx)
        : Base(smem, tidx) {
    }
};

template<
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The size of the STS.
    int BYTES_PER_STS,
    // The number of buffers per tile.
    int BUFFERS_PER_TILE
>
struct Smem_tile_lds_e<Ampere_sphmma_fp16_traits, Cta_tile, Row, BYTES_PER_STS, BUFFERS_PER_TILE>
    : public Smem_tile_sparse_ampere_row_e<Ampere_sphmma_fp16_traits,
                                    Cta_tile,
                                    BYTES_PER_STS,
                                    BUFFERS_PER_TILE,
                                    true> {

    // The traits class.
    using Traits = Ampere_sphmma_fp16_traits;
    // The base class.
    using Base = Smem_tile_sparse_ampere_row_e<Traits, Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE, true>;

    // Ctor.
    inline __device__ Smem_tile_lds_e(char *smem, int tidx)
        : Base(smem, tidx) {
    }
};


////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The size of the STS.
    int BYTES_PER_STS,
    // The number of buffers per tile.
    int BUFFERS_PER_TILE,
    // Use or not predicates
    bool USE_PREDICATES
>
struct Smem_tile_a<Ampere_sphmma_fp16_traits, Cta_tile, Row, BYTES_PER_STS, BUFFERS_PER_TILE, USE_PREDICATES>
    : public Smem_tile_sparse_ampere_row_a<Ampere_sphmma_fp16_traits,
                                    Cta_tile,
                                    BYTES_PER_STS,
                                    BUFFERS_PER_TILE,
                                    USE_PREDICATES> {
    // The traits class.
    using Traits = Ampere_sphmma_fp16_traits;
    // The base class.
    using Base = Smem_tile_sparse_ampere_row_a<Traits, Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE, USE_PREDICATES>;

    // Ctor.
    inline __device__ Smem_tile_a(char *smem, int tidx) : Base(smem, tidx) {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The size of the STS.
    int BYTES_PER_STS,
    // The number of buffers per tile.
    int BUFFERS_PER_TILE
>
struct Smem_tile_a<Ampere_sphmma_fp16_traits, Cta_tile, Col, BYTES_PER_STS, BUFFERS_PER_TILE>
    : public Smem_tile_sparse_ampere_col_a<Ampere_sphmma_fp16_traits,
                                    Cta_tile,
                                    BYTES_PER_STS,
                                    BUFFERS_PER_TILE> {

    // The traits class.
    using Traits = Ampere_sphmma_fp16_traits;
    // The base class.
    using Base = Smem_tile_sparse_ampere_col_a<Traits, Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE>;

    // Ctor.
    inline __device__ Smem_tile_a(char *smem, int tidx) : Base(smem, tidx) {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The size of the STS.
    int BYTES_PER_STS,
    // The number of buffers per tile.
    int BUFFERS_PER_TILE,
    // Use or not predicates
    bool USE_PREDICATES
>
struct Smem_tile_b<Ampere_sphmma_fp16_traits, Cta_tile, Col, BYTES_PER_STS, BUFFERS_PER_TILE, USE_PREDICATES>
    : public Smem_tile_sparse_ampere_col_b<Ampere_sphmma_fp16_traits,
                                    Cta_tile,
                                    BYTES_PER_STS,
                                    BUFFERS_PER_TILE,
                                    USE_PREDICATES> {
    // The traits class.
    using Traits = Ampere_sphmma_fp16_traits;
    // The base class.
    using Base = Smem_tile_sparse_ampere_col_b<Traits, Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE, USE_PREDICATES>;

    // Ctor.
    inline __device__ Smem_tile_b(char *smem, int tidx) : Base(smem, tidx) {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The size of the STS.
    int BYTES_PER_STS,
    // The number of buffers per tile.
    int BUFFERS_PER_TILE
>
struct Smem_tile_b<Ampere_sphmma_fp16_traits, Cta_tile, Row, BYTES_PER_STS, BUFFERS_PER_TILE>
    : public Smem_tile_sparse_ampere_row_b<Ampere_sphmma_fp16_traits,
                                    Cta_tile,
                                    BYTES_PER_STS,
                                    BUFFERS_PER_TILE> {

    // The traits class.
    using Traits = Ampere_sphmma_fp16_traits;
    // The base class.
    using Base = Smem_tile_sparse_ampere_row_b<Traits, Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE>;

    // Ctor.
    inline __device__ Smem_tile_b(char *smem, int tidx) : Base(smem, tidx) {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// S P A R S E  H M M A . T F 3 2
//
////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    typename Input_type,
    typename Output_type,
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The size of the STS.
    int BYTES_PER_STS,
    // The number of buffers per tile.
    int BUFFERS_PER_TILE
>
struct Smem_tile_e<Ampere_sphmma_tf32_traits<Input_type, Output_type>,
    Cta_tile, Row, BYTES_PER_STS, BUFFERS_PER_TILE>
    : public Smem_tile_sparse_ampere_row_e<Ampere_sphmma_tf32_traits<Input_type, Output_type>,
                                    Cta_tile,
                                    BYTES_PER_STS,
                                    BUFFERS_PER_TILE,
                                    false> {

    // The traits class.
    using Traits = Ampere_sphmma_tf32_traits<Input_type, Output_type>;
    // The base class.
    using Base = Smem_tile_sparse_ampere_row_e<Traits, Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE, false>;

    // Ctor.
    inline __device__ Smem_tile_e(char *smem, int tidx)
        : Base(smem, tidx) {
    }
};

template<
    typename Input_type,
    typename Output_type,
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The size of the STS.
    int BYTES_PER_STS,
    // The number of buffers per tile.
    int BUFFERS_PER_TILE
>
struct Smem_tile_lds_e<Ampere_sphmma_tf32_traits<Input_type, Output_type>,
    Cta_tile, Row, BYTES_PER_STS, BUFFERS_PER_TILE>
    : public Smem_tile_sparse_ampere_row_e<Ampere_sphmma_tf32_traits<Input_type, Output_type>,
                                    Cta_tile,
                                    BYTES_PER_STS,
                                    BUFFERS_PER_TILE,
                                    true> {

    // The traits class.
    using Traits = Ampere_sphmma_tf32_traits<Input_type, Output_type>;
    // The base class.
    using Base = Smem_tile_sparse_ampere_row_e<Traits, Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE, true>;

    // Ctor.
    inline __device__ Smem_tile_lds_e(char *smem, int tidx)
        : Base(smem, tidx) {
    }
};


////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    typename Output_type,
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The size of the STS.
    int BYTES_PER_STS,
    // The number of buffers per tile.
    int BUFFERS_PER_TILE,
    // Use or not predicates
    bool USE_PREDICATES
>
struct Smem_tile_a<Ampere_sphmma_tf32_traits<float, Output_type>,
    Cta_tile, Row, BYTES_PER_STS, BUFFERS_PER_TILE, USE_PREDICATES>
    : public Smem_tile_sparse_ampere_row_a<Ampere_sphmma_tf32_traits<float, Output_type>,
                                    Cta_tile,
                                    BYTES_PER_STS,
                                    BUFFERS_PER_TILE,
                                    USE_PREDICATES> {
    // The traits class.
    using Traits = Ampere_sphmma_tf32_traits<float, Output_type>;
    // The base class.
    using Base = Smem_tile_sparse_ampere_row_a<Traits, Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE, USE_PREDICATES>;
    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;
    // The fragment.
    using Fragment = Fragment_a<Traits, Row>;

    // Ctor.
    inline __device__ Smem_tile_a(char *smem, int tidx) : Base(smem, tidx) {
    }

    inline __device__ void load(Fragment (&a)[Xmma_tile::XMMAS_M], int ki) {
        Base::load(a, ki);
        #pragma unroll
        for( int mi = 0; mi < Xmma_tile::XMMAS_M; ++mi ) {
            #pragma unroll
            for( int jj = 0; jj < Fragment::NUM_REGS; ++jj ) {
                float x = reinterpret_cast<float const &>(a[mi].reg(jj)); 
                a[mi].reg(jj) = colwert_tf32(x);
            }
        }

    }
};

template<
    typename Output_type,
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The size of the STS.
    int BYTES_PER_STS,
    // The number of buffers per tile.
    int BUFFERS_PER_TILE
>
struct Smem_tile_a<Ampere_sphmma_tf32_traits<float, Output_type>,
    Cta_tile, Col, BYTES_PER_STS, BUFFERS_PER_TILE>
    : public Smem_tile_sparse_ampere_col_a<Ampere_sphmma_tf32_traits<float, Output_type>,
                                    Cta_tile,
                                    BYTES_PER_STS,
                                    BUFFERS_PER_TILE> {

    // The traits class.
    using Traits = Ampere_sphmma_tf32_traits<float, Output_type>;
    // The base class.
    using Base = Smem_tile_sparse_ampere_col_a<Traits, Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE>;
    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;
    // The fragment.
    using Fragment = Fragment_a<Traits, Col>;

    // Ctor.
    inline __device__ Smem_tile_a(char *smem, int tidx) : Base(smem, tidx) {
    }

    inline __device__ void load(Fragment (&a)[Xmma_tile::XMMAS_M], int ki) {
        Base::load(a, ki);
        #pragma unroll
        for( int mi = 0; mi < Xmma_tile::XMMAS_M; ++mi ) {
            #pragma unroll
            for( int jj = 0; jj < Fragment::NUM_REGS; ++jj ) {
                float x = reinterpret_cast<float const &>(a[mi].reg(jj)); 
                a[mi].reg(jj) = colwert_tf32(x);
            }
        }

    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    typename Output_type,
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The size of the STS.
    int BYTES_PER_STS,
    // The number of buffers per tile.
    int BUFFERS_PER_TILE,
    // Use or not predicates
    bool USE_PREDICATES
>
struct Smem_tile_a<Ampere_sphmma_tf32_traits<lwtlass::float_tf32_t, Output_type>,
    Cta_tile, Row, BYTES_PER_STS, BUFFERS_PER_TILE, USE_PREDICATES>
    : public Smem_tile_sparse_ampere_row_a<Ampere_sphmma_tf32_traits<
                                    lwtlass::float_tf32_t, Output_type>,
                                    Cta_tile,
                                    BYTES_PER_STS,
                                    BUFFERS_PER_TILE,
                                    USE_PREDICATES> {
    // The traits class.
    using Traits = Ampere_sphmma_tf32_traits<lwtlass::float_tf32_t, Output_type>;
    // The base class.
    using Base = Smem_tile_sparse_ampere_row_a<Traits, Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE, USE_PREDICATES>;

    // Ctor.
    inline __device__ Smem_tile_a(char *smem, int tidx) : Base(smem, tidx) {
    }
};

template<
    typename Output_type,
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The size of the STS.
    int BYTES_PER_STS,
    // The number of buffers per tile.
    int BUFFERS_PER_TILE
>
struct Smem_tile_a<Ampere_sphmma_tf32_traits<lwtlass::float_tf32_t, Output_type>,
    Cta_tile, Col, BYTES_PER_STS, BUFFERS_PER_TILE>
    : public Smem_tile_sparse_ampere_col_a<Ampere_sphmma_tf32_traits<lwtlass::float_tf32_t, Output_type>,
                                    Cta_tile,
                                    BYTES_PER_STS,
                                    BUFFERS_PER_TILE> {

    // The traits class.
    using Traits = Ampere_sphmma_tf32_traits<lwtlass::float_tf32_t, Output_type>;
    // The base class.
    using Base = Smem_tile_sparse_ampere_col_a<Traits, Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE>;

    // Ctor.
    inline __device__ Smem_tile_a(char *smem, int tidx) : Base(smem, tidx) {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
/*
template<
    typename Input_type,
    typename Output_type,
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The size of the STS.
    int BYTES_PER_STS,
    // The number of buffers per tile.
    int BUFFERS_PER_TILE
>
struct Smem_tile_a<Ampere_sphmma_tf32_traits<Input_type, Output_type>,
    Cta_tile, Col, BYTES_PER_STS, BUFFERS_PER_TILE>
    : public Smem_tile_sparse_ampere_col_a<Ampere_sphmma_tf32_traits<Input_type, Output_type>,
                                    Cta_tile,
                                    BYTES_PER_STS,
                                    BUFFERS_PER_TILE> {

    // The traits class.
    using Traits = Ampere_sphmma_tf32_traits<Input_type, Output_type>;
    // The base class.
    using Base = Smem_tile_sparse_ampere_col_a<Traits, Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE>;

    // Ctor.
    inline __device__ Smem_tile_a(char *smem, int tidx) : Base(smem, tidx) {
    }
};
*/
////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    typename Output_type,
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The size of the STS.
    int BYTES_PER_STS,
    // The number of buffers per tile.
    int BUFFERS_PER_TILE,
    // Use or not predicates
    bool USE_PREDICATES
>
struct Smem_tile_b<Ampere_sphmma_tf32_traits<float, Output_type>,
    Cta_tile, Col, BYTES_PER_STS, BUFFERS_PER_TILE, USE_PREDICATES>
    : public Smem_tile_sparse_ampere_col_b<Ampere_sphmma_tf32_traits<float, Output_type>,
                                    Cta_tile,
                                    BYTES_PER_STS,
                                    BUFFERS_PER_TILE,
                                    USE_PREDICATES> {
    // The traits class.
    using Traits = Ampere_sphmma_tf32_traits<float, Output_type>;
    // The base class.
    using Base = Smem_tile_sparse_ampere_col_b<Traits, Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE, USE_PREDICATES>;
    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;
    // The fragment.
    using Fragment = Fragment_b<Traits, Col>;

    // Ctor.
    inline __device__ Smem_tile_b(char *smem, int tidx) : Base(smem, tidx) {
    }

    inline __device__ void load(Fragment (&b)[Xmma_tile::XMMAS_N], int ki) {
        Base::load(b, ki);
        #pragma unroll
        for( int ni = 0; ni < Xmma_tile::XMMAS_N; ++ ni ) {
            #pragma unroll
            for( int jj = 0; jj < Fragment::NUM_REGS; ++jj ) {
                float x = reinterpret_cast<float const &>(b[ni].reg(jj)); 
                b[ni].reg(jj) = colwert_tf32(x);
            }
        }
    }

    inline __device__ void pipe_load(Fragment (&b)[1], int ki, int idx) {
        Base::pipe_load(b, ki, idx);
    }

    inline __device__ void tf32_pipe_colwert(Fragment (&b)[Xmma_tile::XMMAS_N][1]){
        #pragma unroll
        for( int ni = 0; ni < Xmma_tile::XMMAS_N; ++ ni ) {
            #pragma unroll
            for( int jj = 0; jj < Fragment::NUM_REGS; ++jj ) {
                float x = reinterpret_cast<float const &>(b[ni][0].reg(jj)); 
                b[ni][0].reg(jj) = colwert_tf32(x);
            }
        }
    }
};


template<
    typename Output_type,
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The size of the STS.
    int BYTES_PER_STS,
    // The number of buffers per tile.
    int BUFFERS_PER_TILE
>
struct Smem_tile_b<Ampere_sphmma_tf32_traits<float, Output_type>,
    Cta_tile, Row, BYTES_PER_STS, BUFFERS_PER_TILE>
    : public Smem_tile_sparse_ampere_row_b<Ampere_sphmma_tf32_traits<float, Output_type>,
                                    Cta_tile,
                                    BYTES_PER_STS,
                                    BUFFERS_PER_TILE> {

    // The traits class.
    using Traits = Ampere_sphmma_tf32_traits<float, Output_type>;
    // The base class.
    using Base = Smem_tile_sparse_ampere_row_b<Traits, Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE>;
    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;
    // The fragment.
    using Fragment = Fragment_b<Traits, Row>;

    // Ctor.
    inline __device__ Smem_tile_b(char *smem, int tidx) : Base(smem, tidx) {
    }

    inline __device__ void load(Fragment (&b)[Xmma_tile::XMMAS_N], int ki) {

        Base::load(b, ki);
        #pragma unroll
        for( int ni = 0; ni < Xmma_tile::XMMAS_N; ++ ni ) {
            #pragma unroll
            for( int jj = 0; jj < Fragment::NUM_REGS; ++jj ) {
                float x = reinterpret_cast<float const &>(b[ni].reg(jj)); 
                b[ni].reg(jj) = colwert_tf32(x);
            }
        }
    }

    inline __device__ void pipe_load(Fragment (&b)[1], int ki, int idx) {
        Base::pipe_load(b, ki, idx);
    }

    inline __device__ void tf32_pipe_colwert(Fragment (&b)[Xmma_tile::XMMAS_N][1]){
        #pragma unroll
        for( int ni = 0; ni < Xmma_tile::XMMAS_N; ++ ni ) {
            #pragma unroll
            for( int jj = 0; jj < Fragment::NUM_REGS; ++jj ) {
                float x = reinterpret_cast<float const &>(b[ni][0].reg(jj)); 
                b[ni][0].reg(jj) = colwert_tf32(x);
            }
        }
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    typename Output_type,
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The size of the STS.
    int BYTES_PER_STS,
    // The number of buffers per tile.
    int BUFFERS_PER_TILE,
    // Use or not predicates
    bool USE_PREDICATES
>
struct Smem_tile_b<Ampere_sphmma_tf32_traits<lwtlass::float_tf32_t, Output_type>,
    Cta_tile, Col, BYTES_PER_STS, BUFFERS_PER_TILE, USE_PREDICATES>
    : public Smem_tile_sparse_ampere_col_b<Ampere_sphmma_tf32_traits<
                                    lwtlass::float_tf32_t, Output_type>,
                                    Cta_tile,
                                    BYTES_PER_STS,
                                    BUFFERS_PER_TILE,
                                    USE_PREDICATES> {
    // The traits class.
    using Traits = Ampere_sphmma_tf32_traits<lwtlass::float_tf32_t, Output_type>;
    // The base class.
    using Base = Smem_tile_sparse_ampere_col_b<Traits, Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE, USE_PREDICATES>;

    // Ctor.
    inline __device__ Smem_tile_b(char *smem, int tidx) : Base(smem, tidx) {
    }
};

template<
    typename Output_type,
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The size of the STS.
    int BYTES_PER_STS,
    // The number of buffers per tile.
    int BUFFERS_PER_TILE
    // Use or not predicates
    //bool USE_PREDICATES
>
struct Smem_tile_b<Ampere_sphmma_tf32_traits<lwtlass::float_tf32_t, Output_type>,
    Cta_tile, Row, BYTES_PER_STS, BUFFERS_PER_TILE>
    : public Smem_tile_sparse_ampere_row_b<Ampere_sphmma_tf32_traits<
                                    lwtlass::float_tf32_t, Output_type>,
                                    Cta_tile,
                                    BYTES_PER_STS,
                                    BUFFERS_PER_TILE
                                    > {
    // The traits class.
    using Traits = Ampere_sphmma_tf32_traits<lwtlass::float_tf32_t, Output_type>;
    // The base class.
    using Base = Smem_tile_sparse_ampere_row_b<Traits, Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE>;

    // Ctor.
    inline __device__ Smem_tile_b(char *smem, int tidx) : Base(smem, tidx) {
    }
};


////////////////////////////////////////////////////////////////////////////////////////////////////
/*
template<
    typename Input_type,
    typename Output_type,
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The size of the STS.
    int BYTES_PER_STS,
    // The number of buffers per tile.
    int BUFFERS_PER_TILE
>
struct Smem_tile_b<Ampere_sphmma_tf32_traits<Input_type, Output_type>,
    Cta_tile, Row, BYTES_PER_STS, BUFFERS_PER_TILE>
    : public Smem_tile_sparse_ampere_row_b<Ampere_sphmma_tf32_traits<Input_type, Output_type>,
                                    Cta_tile,
                                    BYTES_PER_STS,
                                    BUFFERS_PER_TILE> {

    // The traits class.
    using Traits = Ampere_sphmma_tf32_traits<Input_type, Output_type>;
    // The base class.
    using Base = Smem_tile_sparse_ampere_row_b<Traits, Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE>;

    // Ctor.
    inline __device__ Smem_tile_b(char *smem, int tidx) : Base(smem, tidx) {
    }
};
*/
////////////////////////////////////////////////////////////////////////////////////////////////////
//
// S P A R S E  I M M A . 8
//
////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The size of the STS.
    int BYTES_PER_STS,
    // The number of buffers per tile.
    int BUFFERS_PER_TILE
>
struct Smem_tile_e<Ampere_spimma_int8_traits, Cta_tile, Row, BYTES_PER_STS, BUFFERS_PER_TILE>
    : public Smem_tile_sparse_ampere_row_e<Ampere_spimma_int8_traits,
                                    Cta_tile,
                                    BYTES_PER_STS,
                                    BUFFERS_PER_TILE,
                                    false> {

    // The traits class.
    using Traits = Ampere_spimma_int8_traits;
    // The base class.
    using Base = Smem_tile_sparse_ampere_row_e<Traits, Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE, false>;

    // Ctor.
    inline __device__ Smem_tile_e(char *smem, int tidx)
        : Base(smem, tidx) {
    }
};

template<
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The size of the STS.
    int BYTES_PER_STS,
    // The number of buffers per tile.
    int BUFFERS_PER_TILE
>
struct Smem_tile_lds_e<Ampere_spimma_int8_traits, Cta_tile, Row, BYTES_PER_STS, BUFFERS_PER_TILE>
    : public Smem_tile_sparse_ampere_row_e<Ampere_spimma_int8_traits,
                                    Cta_tile,
                                    BYTES_PER_STS,
                                    BUFFERS_PER_TILE,
                                    true> {

    // The traits class.
    using Traits = Ampere_spimma_int8_traits;
    // The base class.
    using Base = Smem_tile_sparse_ampere_row_e<Traits, Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE, true>;

    // Ctor.
    inline __device__ Smem_tile_lds_e(char *smem, int tidx)
        : Base(smem, tidx) {
    }
};


////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The size of the STS.
    int BYTES_PER_STS,
    // The number of buffers per tile.
    int BUFFERS_PER_TILE,
    // Use or not predicates
    bool USE_PREDICATES
>
struct Smem_tile_a<Ampere_spimma_int8_traits, Cta_tile, Row, BYTES_PER_STS, BUFFERS_PER_TILE, USE_PREDICATES>
    : public Smem_tile_sparse_ampere_row_a<Ampere_spimma_int8_traits,
                                    Cta_tile,
                                    BYTES_PER_STS,
                                    BUFFERS_PER_TILE,
                                    USE_PREDICATES> {
    // The traits class.
    using Traits = Ampere_spimma_int8_traits;
    // The base class.
    using Base = Smem_tile_sparse_ampere_row_a<Traits, Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE, USE_PREDICATES>;

    // Ctor.
    inline __device__ Smem_tile_a(char *smem, int tidx) : Base(smem, tidx) {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The size of the STS.
    int BYTES_PER_STS,
    // The number of buffers per tile.
    int BUFFERS_PER_TILE
>
struct Smem_tile_a<Ampere_spimma_int8_traits, Cta_tile, Col, BYTES_PER_STS, BUFFERS_PER_TILE>
    : public Smem_tile_sparse_ampere_col_a<Ampere_spimma_int8_traits,
                                    Cta_tile,
                                    BYTES_PER_STS,
                                    BUFFERS_PER_TILE> {

    // The traits class.
    using Traits = Ampere_spimma_int8_traits;
    // The base class.
    using Base = Smem_tile_sparse_ampere_col_a<Traits, Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE>;

    // Ctor.
    inline __device__ Smem_tile_a(char *smem, int tidx) : Base(smem, tidx) {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The size of the STS.
    int BYTES_PER_STS,
    // The number of buffers per tile.
    int BUFFERS_PER_TILE,
    // Use or not predicates
    bool USE_PREDICATES
>
struct Smem_tile_b<Ampere_spimma_int8_traits, Cta_tile, Col, BYTES_PER_STS, BUFFERS_PER_TILE, USE_PREDICATES>
    : public Smem_tile_sparse_ampere_col_b<Ampere_spimma_int8_traits,
                                    Cta_tile,
                                    BYTES_PER_STS,
                                    BUFFERS_PER_TILE,
                                    USE_PREDICATES> {
    // The traits class.
    using Traits = Ampere_spimma_int8_traits;
    // The base class.
    using Base = Smem_tile_sparse_ampere_col_b<Traits, Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE, USE_PREDICATES>;

    // Ctor.
    inline __device__ Smem_tile_b(char *smem, int tidx) : Base(smem, tidx) {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The size of the STS.
    int BYTES_PER_STS,
    // The number of buffers per tile.
    int BUFFERS_PER_TILE
>
struct Smem_tile_b<Ampere_spimma_int8_traits, Cta_tile, Row, BYTES_PER_STS, BUFFERS_PER_TILE>
    : public Smem_tile_sparse_ampere_row_b<Ampere_spimma_int8_traits,
                                    Cta_tile,
                                    BYTES_PER_STS,
                                    BUFFERS_PER_TILE> {

    // The traits class.
    using Traits = Ampere_spimma_int8_traits;
    // The base class.
    using Base = Smem_tile_sparse_ampere_row_b<Traits, Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE>;

    // Ctor.
    inline __device__ Smem_tile_b(char *smem, int tidx) : Base(smem, tidx) {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// S P A R S E  I M M A . 8   I N T E R L E A V E D
//
////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The size of the STS.
    int BYTES_PER_STS,
    // The number of buffers per tile.
    int BUFFERS_PER_TILE
>
struct Smem_tile_e<Ampere_spimma_interleaved_int8_traits, Cta_tile, Row, BYTES_PER_STS, BUFFERS_PER_TILE>
    : public Smem_tile_sparse_ampere_row_e<Ampere_spimma_interleaved_int8_traits,
                                    Cta_tile,
                                    BYTES_PER_STS,
                                    BUFFERS_PER_TILE,
                                    false> {

    // The traits class.
    using Traits = Ampere_spimma_interleaved_int8_traits;
    // The base class.
    using Base = Smem_tile_sparse_ampere_row_e<Traits, Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE, false>;

    // Ctor.
    inline __device__ Smem_tile_e(char *smem, int tidx)
        : Base(smem, tidx) {
    }
};

template<
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The size of the STS.
    int BYTES_PER_STS,
    // The number of buffers per tile.
    int BUFFERS_PER_TILE
>
struct Smem_tile_lds_e<Ampere_spimma_interleaved_int8_traits, Cta_tile, Row, BYTES_PER_STS, BUFFERS_PER_TILE>
    : public Smem_tile_sparse_ampere_row_e<Ampere_spimma_interleaved_int8_traits,
                                    Cta_tile,
                                    BYTES_PER_STS,
                                    BUFFERS_PER_TILE,
                                    true> {

    // The traits class.
    using Traits = Ampere_spimma_interleaved_int8_traits;
    // The base class.
    using Base = Smem_tile_sparse_ampere_row_e<Traits, Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE, true>;

    // Ctor.
    inline __device__ Smem_tile_lds_e(char *smem, int tidx)
        : Base(smem, tidx) {
    }
};


////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The size of the STS.
    int BYTES_PER_STS,
    // The number of buffers per tile.
    int BUFFERS_PER_TILE,
    // Use or not predicates
    bool USE_PREDICATES
>
struct Smem_tile_a<Ampere_spimma_interleaved_int8_traits, Cta_tile, Row, BYTES_PER_STS, BUFFERS_PER_TILE, USE_PREDICATES>
    : public Smem_tile_sparse_ampere_row_a<Ampere_spimma_interleaved_int8_traits,
                                    Cta_tile,
                                    BYTES_PER_STS,
                                    BUFFERS_PER_TILE,
                                    USE_PREDICATES> {
    // The traits class.
    using Traits = Ampere_spimma_interleaved_int8_traits;
    // The base class.
    using Base = Smem_tile_sparse_ampere_row_a<Traits, Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE, USE_PREDICATES>;

    // Ctor.
    inline __device__ Smem_tile_a(char *smem, int tidx) : Base(smem, tidx) {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The size of the STS.
    int BYTES_PER_STS,
    // The number of buffers per tile.
    int BUFFERS_PER_TILE
>
struct Smem_tile_a<Ampere_spimma_interleaved_int8_traits, Cta_tile, Col, BYTES_PER_STS, BUFFERS_PER_TILE>
    : public Smem_tile_sparse_ampere_col_a<Ampere_spimma_interleaved_int8_traits,
                                    Cta_tile,
                                    BYTES_PER_STS,
                                    BUFFERS_PER_TILE> {

    // The traits class.
    using Traits = Ampere_spimma_interleaved_int8_traits;
    // The base class.
    using Base = Smem_tile_sparse_ampere_col_a<Traits, Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE>;

    // Ctor.
    inline __device__ Smem_tile_a(char *smem, int tidx) : Base(smem, tidx) {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The size of the STS.
    int BYTES_PER_STS,
    // The number of buffers per tile.
    int BUFFERS_PER_TILE,
    // Use or not predicates
    bool USE_PREDICATES
>
struct Smem_tile_b<Ampere_spimma_interleaved_int8_traits, Cta_tile, Col, BYTES_PER_STS, BUFFERS_PER_TILE, USE_PREDICATES>
    : public Smem_tile_sparse_ampere_interleaved_col_b<Ampere_spimma_interleaved_int8_traits,
                                    Cta_tile,
                                    BYTES_PER_STS,
                                    BUFFERS_PER_TILE,
                                    USE_PREDICATES> {
    // The traits class.
    using Traits = Ampere_spimma_interleaved_int8_traits;
    // The base class.
    using Base = Smem_tile_sparse_ampere_interleaved_col_b<Traits, Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE, USE_PREDICATES>;

    // Ctor.
    inline __device__ Smem_tile_b(char *smem, int tidx) : Base(smem, tidx) {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The size of the STS.
    int BYTES_PER_STS,
    // The number of buffers per tile.
    int BUFFERS_PER_TILE
>
struct Smem_tile_b<Ampere_spimma_interleaved_int8_traits, Cta_tile, Row, BYTES_PER_STS, BUFFERS_PER_TILE>
    : public Smem_tile_sparse_ampere_row_b<Ampere_spimma_interleaved_int8_traits,
                                    Cta_tile,
                                    BYTES_PER_STS,
                                    BUFFERS_PER_TILE> {

    // The traits class.
    using Traits = Ampere_spimma_interleaved_int8_traits;
    // The base class.
    using Base = Smem_tile_sparse_ampere_row_b<Traits, Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE>;

    // Ctor.
    inline __device__ Smem_tile_b(char *smem, int tidx) : Base(smem, tidx) {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< 
    typename Traits_, 
    typename Cta_tile_, 
    typename Storage_type_, 
    typename Layout_, 
    typename Fragment_pre_swizzle_ = Fragment_epilogue_pre_swizzle<Traits_, Cta_tile_>,
    typename Fragment_post_swizzle_ = Fragment_epilogue_post_swizzle<Traits_, Cta_tile_> 
>
struct Swizzle_sparse_ampere_epilogue {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< 
    typename Traits_, 
    typename Cta_tile_, 
    typename Storage_type_, 
    typename Fragment_pre_swizzle_,
    typename Fragment_post_swizzle_
>
struct Swizzle_sparse_ampere_epilogue <Traits_,
                                       Cta_tile_,
                                       Storage_type_,
                                       xmma::Row,
                                       Fragment_pre_swizzle_, 
                                       Fragment_post_swizzle_> {

    using Traits = Traits_;
    using Cta_tile = Cta_tile_;
    using Storage_type = Storage_type_;
    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;

    // The number of bytes for key memory instruction.
    enum { BYTES_PER_LDS = 16, BYTES_PER_STG = 16, BYTES_PER_STS = 2*sizeof(Storage_type) };
    // The amount of bytes per element.
    enum { BYTES_PER_ELEMENT = sizeof(Storage_type) };
    // The number of rows in shared memory.
    enum { ROWS = Xmma_tile::M_PER_XMMA_PER_CTA };
    // The size of a single row in bytes.
    enum { BYTES_PER_ROW = Cta_tile::N * Cta_tile::WARPS_K * sizeof(Storage_type) };
    // The skew to avoid bank conflicts.
    enum { BYTES_PER_SKEW = BYTES_PER_STS*4 };
    // The number of columns with the skew.
    enum { BYTES_PER_ROW_WITH_SKEW = BYTES_PER_ROW + BYTES_PER_SKEW };
    // The size in bytes in shared memory.
    enum { BYTES_PER_TILE = ROWS * BYTES_PER_ROW_WITH_SKEW };

    // The number of threads per "pixel".
    enum { THREADS_PER_PIXEL = Cta_tile::N * sizeof(typename Traits::C_type) / BYTES_PER_STG };
    // The number of "pixels" written in one STS.128.
    enum { PIXELS_PER_STG = Cta_tile::THREADS_PER_CTA / THREADS_PER_PIXEL };

    // How we see the distribution of data.
    enum { THREADS_PER_XMMA_M = 8, THREADS_PER_XMMA_N = 4 };
    // The number of elements stored per thread.
    enum { M_PER_XMMA_PER_THREAD = 2, N_PER_XMMA_PER_THREAD = 4 };

    // Do we skip the syncthreads in the epilogue? Of course, not :)
    enum { SKIP_SYNCTHREADS = 0 };

    // Ctor.
    inline __device__ Swizzle_sparse_ampere_epilogue(char *smem, int tidx) 
        : smem_(smem) {

        // Extract the number of warps.
        const int WARPS_M = Cta_tile::WARPS_M;
        const int WARPS_N = Cta_tile::WARPS_N;
        const int WARPS_K = Cta_tile::WARPS_K;

        //
        //     tidx   0: row =  0, col = 0*BYTES_PER_STS
        //     tidx   1: row =  0, col = 1*BYTES_PER_STS
        //     tidx   2: row =  0, col = 2*BYTES_PER_STS
        //     tidx   3: row =  0, col = 3*BYTES_PER_STS
        //     tidx   4: row =  1, col = 0*BYTES_PER_STS
        //     tidx   5: row =  1, col = 1*BYTES_PER_STS
        //     tidx   6: row =  1, col = 2*BYTES_PER_STS
        //     tidx   7: row =  1, col = 3*BYTES_PER_STS
        //     tidx   8: row =  2, col = 0*BYTES_PER_STS
        //     tidx   9: row =  2, col = 1*BYTES_PER_STS
        //     tidx  10: row =  2, col = 2*BYTES_PER_STS
        //     tidx  11: row =  2, col = 3*BYTES_PER_STS
        //     tidx  12: row =  3, col = 0*BYTES_PER_STS
        //     tidx  13: row =  3, col = 1*BYTES_PER_STS
        //     tidx  14: row =  3, col = 2*BYTES_PER_STS
        //     tidx  15: row =  3, col = 3*BYTES_PER_STS
        //     tidx  16: row =  4, col = 0*BYTES_PER_STS
        //     tidx  17: row =  4, col = 1*BYTES_PER_STS
        //     tidx  18: row =  4, col = 2*BYTES_PER_STS
        //     tidx  19: row =  4, col = 3*BYTES_PER_STS
        //     tidx  20: row =  5, col = 0*BYTES_PER_STS
        //     tidx  21: row =  5, col = 1*BYTES_PER_STS
        //     tidx  22: row =  5, col = 2*BYTES_PER_STS
        //     tidx  23: row =  5, col = 3*BYTES_PER_STS
        //     tidx  24: row =  6, col = 0*BYTES_PER_STS
        //     tidx  25: row =  6, col = 1*BYTES_PER_STS
        //     tidx  26: row =  6, col = 2*BYTES_PER_STS
        //     tidx  27: row =  6, col = 3*BYTES_PER_STS
        //     tidx  28: row =  7, col = 0*BYTES_PER_STS
        //     tidx  29: row =  7, col = 1*BYTES_PER_STS
        //     tidx  30: row =  7, col = 2*BYTES_PER_STS
        //     tidx  31: row =  7, col = 3*BYTES_PER_STS
        //

	    // The masks to select the warps.
        const int WARP_MASK_M = Warp_masks<WARPS_M, WARPS_N, WARPS_K>::M;
        const int WARP_MASK_N = Warp_masks<WARPS_M, WARPS_N, WARPS_K>::N;
        const int WARP_MASK_K = Warp_masks<WARPS_M, WARPS_N, WARPS_K>::K;

        // The divisor for the warps.
        const int WARP_DIV_M =       1 *       1 * Cta_tile::THREADS_PER_WARP;
        const int WARP_DIV_N = WARPS_M *       1 * Cta_tile::THREADS_PER_WARP;
        const int WARP_DIV_K = WARPS_M * WARPS_N * Cta_tile::THREADS_PER_WARP;

        // The number of bytes in the N dimension.
        const int BYTES_PER_TILE_N = Cta_tile::N * sizeof(Storage_type);

        // Compute the row and the column in shared memory. Each warp reads from a 16*16B segment.
        const int smem_write_row = (tidx & WARP_MASK_M) / WARP_DIV_M * 16 + 
                                   (tidx & 0x1c) / 4;
        const int smem_write_col = (tidx & WARP_MASK_K) / WARP_DIV_K * BYTES_PER_TILE_N +
                                   (tidx & WARP_MASK_N) / WARP_DIV_N * BYTES_PER_STS * 4 * 2 + 
                                   (tidx & 0x03)                     * BYTES_PER_STS;

        // The corresponding offset.
        smem_write_offset_ = smem_write_row*BYTES_PER_ROW_WITH_SKEW + smem_write_col;

        // Decompose into groups of size "THREADS_PER_PIXEL".
        const int tidx_div_tpp = tidx / THREADS_PER_PIXEL;
        const int tidx_mod_tpp = tidx % THREADS_PER_PIXEL;

        // The row and column read by a single thread.
        const int smem_read_row = tidx_div_tpp;
        const int smem_read_col = tidx_mod_tpp * BYTES_PER_LDS;

        // The corresponding offset.
        smem_read_offset_ = smem_read_row*BYTES_PER_ROW_WITH_SKEW + smem_read_col;
    }

    // The shared memory pointer in bytes.
    char *smem_;
    // The write offset.
    int smem_write_offset_;
    // The read offset.
    int smem_read_offset_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// Keep here for fp16/bf16 row-major swizzle epilogue reference
/*
template< typename Cta_tile >
struct Swizzle_epilogue<Ampere_sphmma_fp32_traits, Cta_tile, xmma::Row, 16, false> 
    : public Swizzle_sparse_ampere_epilogue<Ampere_sphmma_fp32_traits, 
                                            Cta_tile, 
                                            uint16_t, 
                                            xmma::Row> {

    // The traits.
    using Traits = Ampere_sphmma_fp32_traits;
    // The base class.
    using Base = Swizzle_sparse_ampere_epilogue<Traits, 
                                                Cta_tile, 
                                                uint16_t, 
                                                xmma::Row>;
    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;

    // Ctor.
    inline __device__ Swizzle_epilogue(char *smem, int tidx) : Base(smem, tidx) {
    }

    // Load from the tile in shared memory.
    template<typename Fragment_post_swizzle>
    inline __device__ void load(Fragment_post_swizzle &dst, int oi) const {
        const int offset = oi * Base::PIXELS_PER_STG * Base::BYTES_PER_ROW_WITH_SKEW;
        uint4 tmp;
        lds(tmp, get_smem_pointer(this->smem_ + this->smem_read_offset_ + offset));
        dst.reg(0) = tmp.x;
        dst.reg(1) = tmp.y;
        dst.reg(2) = tmp.z;
        dst.reg(3) = tmp.w;
    }

    // Store to the tile in shared memory.
    template<typename Fragment_pre_swizzle>
    inline __device__ void store(int ni, const Fragment_pre_swizzle &c) {
        #pragma unroll
        for( int mi = 0; mi < Base::M_PER_XMMA_PER_THREAD; ++mi ) {
            int offset = mi * Base::THREADS_PER_XMMA_M * Base::BYTES_PER_ROW_WITH_SKEW +
                         ni * Xmma_tile::N_PER_XMMA_PER_CTA * sizeof(uint16_t);
            reinterpret_cast<int*>(&this->smem_[this->smem_write_offset_ + offset +  0])[0] = 
                c.reg(2 * mi + 0);
            reinterpret_cast<int*>(&this->smem_[this->smem_write_offset_ + offset + 16])[0] = 
                c.reg(2 * mi + 1);
        }
    }
};
*/
///////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Cta_tile >
struct Swizzle_epilogue<Ampere_spimma_int8_traits, Cta_tile, xmma::Row, 16, false> 
    : public Swizzle_sparse_ampere_epilogue<Ampere_spimma_int8_traits, 
                                            Cta_tile, 
                                            uint16_t, 
                                            xmma::Row> {

    // The traits.
    using Traits = Ampere_spimma_int8_traits;
    // The base class.
    using Base = Swizzle_sparse_ampere_epilogue<Traits, 
                                                Cta_tile, 
                                                uint16_t, 
                                                xmma::Row>;
    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;

    // Ctor.
    inline __device__ Swizzle_epilogue(char *smem, int tidx) : Base(smem, tidx) {
    }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Cta_tile >
struct Swizzle_epilogue<Ampere_spimma_interleaved_int8_traits, Cta_tile, xmma::Row, 16, false> 
    : public Swizzle_sparse_ampere_epilogue<Ampere_spimma_interleaved_int8_traits, 
                                            Cta_tile, 
                                            uint16_t, 
                                            xmma::Row> {

    // The traits.
    using Traits = Ampere_spimma_interleaved_int8_traits;
    // The base class.
    using Base = Swizzle_sparse_ampere_epilogue<Traits, 
                                                Cta_tile, 
                                                uint16_t, 
                                                xmma::Row>;
    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;

    // Ctor.
    inline __device__ Swizzle_epilogue(char *smem, int tidx) : Base(smem, tidx) {
    }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

template< 
    typename Traits_, 
    typename Cta_tile_, 
    typename Storage_type_, 
    typename Fragment_pre_swizzle_,
    typename Fragment_post_swizzle_
>
struct Swizzle_sparse_ampere_epilogue <Traits_,
                                       Cta_tile_,
                                       Storage_type_,
                                       xmma::Col,
                                       Fragment_pre_swizzle_, 
                                       Fragment_post_swizzle_> {

    using Traits = Traits_;
    using Cta_tile = Cta_tile_;
    using Storage_type = Storage_type_;
    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;

    // The number of bytes for key memory instruction.
    enum { BYTES_PER_LDS = 16, BYTES_PER_STG = 16, BYTES_PER_STS = 2*sizeof(Storage_type) };
    // The amount of bytes per element.
    enum { BYTES_PER_ELEMENT = sizeof(Storage_type) };
    // The number of rows in shared memory.
    //enum { ROWS = Xmma_tile::M_PER_XMMA_PER_CTA };
    enum { ROWS = Xmma_tile::N_PER_XMMA_PER_CTA };
    // The size of a single row in bytes.
    //enum { BYTES_PER_ROW = Cta_tile::N * Cta_tile::WARPS_K * sizeof(Storage_type) };
    enum { BYTES_PER_ROW = Cta_tile::M * Cta_tile::WARPS_K * sizeof(Storage_type) };
    // The skew to avoid bank conflicts.
    enum { BYTES_PER_SKEW = BYTES_PER_STS*4 };
    // The number of columns with the skew.
    enum { BYTES_PER_ROW_WITH_SKEW = BYTES_PER_ROW + BYTES_PER_SKEW };
    // The size in bytes in shared memory.
    enum { BYTES_PER_TILE = ROWS * BYTES_PER_ROW_WITH_SKEW };

    // The number of threads per "pixel".
    //enum { THREADS_PER_PIXEL = Cta_tile::N * sizeof(typename Traits::C_type) / BYTES_PER_STG };
    enum { THREADS_PER_PIXEL = Cta_tile::M * sizeof(typename Traits::C_type) / BYTES_PER_STG };
    // The number of "pixels" written in one STS.128.
    enum { PIXELS_PER_STG = Cta_tile::THREADS_PER_CTA / THREADS_PER_PIXEL };

    // How we see the distribution of data.
    enum { THREADS_PER_XMMA_M = 8, THREADS_PER_XMMA_N = 4 };
    // The number of elements stored per thread.
    enum { M_PER_XMMA_PER_THREAD = 2, N_PER_XMMA_PER_THREAD = 4 };

    // Do we skip the syncthreads in the epilogue? Of course, not :)
    enum { SKIP_SYNCTHREADS = 0 };

    // Ctor.
    inline __device__ Swizzle_sparse_ampere_epilogue(char *smem, int tidx) 
        : smem_(smem) {

        // Extract the number of warps.
        const int WARPS_M = Cta_tile::WARPS_M;
        const int WARPS_N = Cta_tile::WARPS_N;
        const int WARPS_K = Cta_tile::WARPS_K;

	    // The masks to select the warps.
        const int WARP_MASK_M = Warp_masks<WARPS_M, WARPS_N, WARPS_K>::M;
        //const int WARP_MASK_N = Warp_masks<WARPS_M, WARPS_N, WARPS_K>::N;
        //const int WARP_MASK_K = Warp_masks<WARPS_M, WARPS_N, WARPS_K>::K;

        // The divisor for the warps.
        const int WARP_DIV_M =       1 *       1 * Cta_tile::THREADS_PER_WARP;
        const int WARP_DIV_N = WARPS_M *       1 * Cta_tile::THREADS_PER_WARP;
        //const int WARP_DIV_K = WARPS_M * WARPS_N * Cta_tile::THREADS_PER_WARP;

        // The number of bytes in the N dimension.
        //const int BYTES_PER_TILE_N = Cta_tile::N * sizeof(Storage_type);

        //const int smem_write_row = (tidx / WARP_DIV_N) * (ROWS / 2) + 
        const int smem_write_row = (tidx / WARP_DIV_N) * (ROWS / WARPS_N) + 
                                (tidx % WARP_DIV_N) % N_PER_XMMA_PER_THREAD * 2; 
                                
        const int smem_write_col = ( (tidx % Cta_tile::THREADS_PER_WARP) / THREADS_PER_XMMA_N + 
                                    (tidx & WARP_MASK_M) / WARP_DIV_M * Traits::ROW_STRIDE_GROUP) * 2;
                                    //(tidx % WARP_DIV_N) / ROWS * Traits::ROW_STRIDE_GROUP) * 2;
        // The corresponding offset.
        smem_write_offset_ = smem_write_row*BYTES_PER_ROW_WITH_SKEW + smem_write_col;

        // Decompose into groups of size "THREADS_PER_PIXEL".
        const int tidx_div_tpp = tidx / THREADS_PER_PIXEL;
        const int tidx_mod_tpp = tidx % THREADS_PER_PIXEL;

        // The row and column read by a single thread.
        const int smem_read_row = tidx_div_tpp;
        const int smem_read_col = tidx_mod_tpp * BYTES_PER_LDS;

        // The corresponding offset.
        smem_read_offset_ = smem_read_row*BYTES_PER_ROW_WITH_SKEW + smem_read_col;
    }

    // The shared memory pointer in bytes.
    char *smem_;
    // The write offset.
    int smem_write_offset_;
    // The read offset.
    int smem_read_offset_;
};

///////////////////////////////////////////////////////////////////////////////////////////////////
// Keep here for fp16/bf16 col-major swizzle epilogue reference
/*
template< typename Cta_tile >
struct Swizzle_epilogue<Ampere_sphmma_fp32_traits, Cta_tile, xmma::Col, 16, false> 
    : public Swizzle_sparse_ampere_epilogue<Ampere_sphmma_fp32_traits, 
                                            Cta_tile, 
                                            uint16_t, 
                                            xmma::Col> {

    // The traits.
    using Traits = Ampere_sphmma_fp32_traits;
    // The base class.
    using Base = Swizzle_sparse_ampere_epilogue<Traits, 
                                                Cta_tile, 
                                                uint16_t, 
                                                xmma::Col>;
    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;

    // Ctor.
    inline __device__ Swizzle_epilogue(char *smem, int tidx) : Base(smem, tidx) {
    }

    // Load from the tile in shared memory.
    template<typename Fragment_post_swizzle>
    inline __device__ void load(Fragment_post_swizzle &dst, int oi) const {
        const int offset = oi * Base::PIXELS_PER_STG * Base::BYTES_PER_ROW_WITH_SKEW;
        uint4 tmp;
        lds(tmp, get_smem_pointer(this->smem_ + this->smem_read_offset_ + offset));
        dst.reg(0) = tmp.x;
        dst.reg(1) = tmp.y;
        dst.reg(2) = tmp.z;
        dst.reg(3) = tmp.w;
    }

    // Store to the tile in shared memory.
    template<typename Fragment_pre_swizzle>
    inline __device__ void store(int ni, const Fragment_pre_swizzle &c) {

        #pragma unroll
        for( int mi = 0; mi < Base::M_PER_XMMA_PER_THREAD; ++mi ) {               
            int offset = (mi * Base::THREADS_PER_XMMA_M + ni * Xmma_tile::M_PER_XMMA) * sizeof(uint16_t);

            uint16_t reg_0 = 0;
            uint16_t reg_1 = 0;
            reg_0 = c.reg(2*mi + 0) & 0xFFFF;
            reg_1 = (c.reg(2*mi + 0)>>16);// & 0xFFFF;

            reinterpret_cast<uint16_t*>(&this->smem_[this->smem_write_offset_ + offset])[0] = reg_0;
            reinterpret_cast<uint16_t*>(&this->smem_[this->smem_write_offset_ + offset + Base::BYTES_PER_ROW_WITH_SKEW])[0] = reg_1;
  
            offset = offset + ((Base::ROWS / Cta_tile::WARPS_N) / 2) * Base::BYTES_PER_ROW_WITH_SKEW;
            //offset = offset + ((Base::ROWS / 2) / 2) * Base::BYTES_PER_ROW_WITH_SKEW;
            
            uint16_t reg_2 = 0;
            uint16_t reg_3 = 0;
            reg_2 = c.reg(2*mi + 1) & 0xFFFF;
            reg_3 = (c.reg(2*mi + 1)>>16);// & 0xFFFF;

            reinterpret_cast<uint16_t*>(&this->smem_[this->smem_write_offset_ + offset])[0] = reg_2;
            reinterpret_cast<uint16_t*>(&this->smem_[this->smem_write_offset_ + offset + Base::BYTES_PER_ROW_WITH_SKEW])[0] = reg_3;

        }
    }
};
*/
////////////////////////////////////////////////////////////////////////////////////////////////////
//
// S P A R S E  I M M A
//
////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Cta_tile >
struct Swizzle_epilogue<Ampere_spimma_int8_traits, Cta_tile, xmma::Col, 16, false> 
    : public Swizzle_sparse_ampere_epilogue<Ampere_spimma_int8_traits, 
                                            Cta_tile, 
                                            uint16_t, 
                                            xmma::Col> {

    // The traits.
    using Traits = Ampere_spimma_int8_traits;
    // The base class.
    using Base = Swizzle_sparse_ampere_epilogue<Traits, 
                                                Cta_tile, 
                                                uint16_t, 
                                                xmma::Col>;
    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;

    // Ctor.
    inline __device__ Swizzle_epilogue(char *smem, int tidx) : Base(smem, tidx) {
    }
};

///////////////////////////////////////////////////////////////////////////////////////////////////
template< typename Cta_tile >
struct Swizzle_epilogue<Ampere_spimma_gelu_int8_traits , Cta_tile, xmma::Col, 16, false> 
    : public Swizzle_sparse_ampere_epilogue<Ampere_spimma_gelu_int8_traits , 
                                            Cta_tile, 
                                            uint16_t, 
                                            xmma::Col> {

    // The traits.
    using Traits = Ampere_spimma_gelu_int8_traits ;
    // The base class.
    using Base = Swizzle_sparse_ampere_epilogue<Traits, 
                                                Cta_tile, 
                                                uint16_t, 
                                                xmma::Col>;
    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;

    // Ctor.
    inline __device__ Swizzle_epilogue(char *smem, int tidx) : Base(smem, tidx) {
    }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Cta_tile >
struct Swizzle_epilogue<Ampere_spimma_gelu_int8_traits , Cta_tile, xmma::Row, 16, false> 
    : public Swizzle_sparse_ampere_epilogue<Ampere_spimma_gelu_int8_traits , 
                                            Cta_tile, 
                                            uint16_t, 
                                            xmma::Row> {

    // The traits.
    using Traits = Ampere_spimma_gelu_int8_traits ;
    // The base class.
    using Base = Swizzle_sparse_ampere_epilogue<Traits, 
                                                Cta_tile, 
                                                uint16_t, 
                                                xmma::Row>;
    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;

    // Ctor.
    inline __device__ Swizzle_epilogue(char *smem, int tidx) : Base(smem, tidx) {
    }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Cta_tile >
struct Swizzle_epilogue<Ampere_spimma_interleaved_int8_traits, Cta_tile, xmma::Col, 16, false> 
    : public Swizzle_sparse_ampere_epilogue<Ampere_spimma_interleaved_int8_traits, 
                                            Cta_tile, 
                                            uint16_t, 
                                            xmma::Col> {

    // The traits.
    using Traits = Ampere_spimma_interleaved_int8_traits;
    // The base class.
    using Base = Swizzle_sparse_ampere_epilogue<Traits, 
                                                Cta_tile, 
                                                uint16_t, 
                                                xmma::Col>;
    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;

    // Ctor.
    inline __device__ Swizzle_epilogue(char *smem, int tidx) : Base(smem, tidx) {
    }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Cta_tile >
struct Swizzle_epilogue<Ampere_spimma_interleaved_gelu_int8_traits , Cta_tile, xmma::Col, 16, false> 
    : public Swizzle_sparse_ampere_epilogue<Ampere_spimma_interleaved_gelu_int8_traits , 
                                            Cta_tile, 
                                            uint16_t, 
                                            xmma::Col> {

    // The traits.
    using Traits = Ampere_spimma_interleaved_gelu_int8_traits ;
    // The base class.
    using Base = Swizzle_sparse_ampere_epilogue<Traits, 
                                                Cta_tile, 
                                                uint16_t, 
                                                xmma::Col>;
    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;

    // Ctor.
    inline __device__ Swizzle_epilogue(char *smem, int tidx) : Base(smem, tidx) {
    }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Cta_tile >
struct Swizzle_epilogue<Ampere_spimma_interleaved_gelu_int8_traits , Cta_tile, xmma::Row, 16, false> 
    : public Swizzle_sparse_ampere_epilogue<Ampere_spimma_interleaved_gelu_int8_traits , 
                                            Cta_tile, 
                                            uint16_t, 
                                            xmma::Row> {

    // The traits.
    using Traits = Ampere_spimma_interleaved_gelu_int8_traits ;
    // The base class.
    using Base = Swizzle_sparse_ampere_epilogue<Traits, 
                                                Cta_tile, 
                                                uint16_t, 
                                                xmma::Row>;
    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;

    // Ctor.
    inline __device__ Swizzle_epilogue(char *smem, int tidx) : Base(smem, tidx) {
    }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

template<
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The size of the STS.
    int BYTES_PER_STS,
    // The number of buffers per tile.
    int BUFFERS_PER_TILE
>
struct Smem_tile_e<Ampere_spimma_gelu_int8_traits, Cta_tile, Row, BYTES_PER_STS, BUFFERS_PER_TILE>
    : public Smem_tile_sparse_ampere_row_e<Ampere_spimma_gelu_int8_traits,
                                    Cta_tile,
                                    BYTES_PER_STS,
                                    BUFFERS_PER_TILE,
                                    false> {

    // The traits class.
    using Traits = Ampere_spimma_gelu_int8_traits;
    // The base class.
    using Base = Smem_tile_sparse_ampere_row_e<Traits, Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE, false>;

    // Ctor.
    inline __device__ Smem_tile_e(char *smem, int tidx)
        : Base(smem, tidx) {
    }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

template<
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The size of the STS.
    int BYTES_PER_STS,
    // The number of buffers per tile.
    int BUFFERS_PER_TILE
>
struct Smem_tile_lds_e<Ampere_spimma_gelu_int8_traits, Cta_tile, Row, BYTES_PER_STS, BUFFERS_PER_TILE>
    : public Smem_tile_sparse_ampere_row_e<Ampere_spimma_gelu_int8_traits,
                                    Cta_tile,
                                    BYTES_PER_STS,
                                    BUFFERS_PER_TILE,
                                    true> {

    // The traits class.
    using Traits = Ampere_spimma_gelu_int8_traits;
    // The base class.
    using Base = Smem_tile_sparse_ampere_row_e<Traits, Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE, true>;

    // Ctor.
    inline __device__ Smem_tile_lds_e(char *smem, int tidx)
        : Base(smem, tidx) {
    }
};


////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The size of the STS.
    int BYTES_PER_STS,
    // The number of buffers per tile.
    int BUFFERS_PER_TILE,
    // Use or not predicates
    bool USE_PREDICATES
>
struct Smem_tile_a<Ampere_spimma_gelu_int8_traits, Cta_tile, Row, BYTES_PER_STS, BUFFERS_PER_TILE, USE_PREDICATES>
    : public Smem_tile_sparse_ampere_row_a<Ampere_spimma_gelu_int8_traits,
                                    Cta_tile,
                                    BYTES_PER_STS,
                                    BUFFERS_PER_TILE,
                                    USE_PREDICATES> {
    // The traits class.
    using Traits = Ampere_spimma_gelu_int8_traits;
    // The base class.
    using Base = Smem_tile_sparse_ampere_row_a<Traits, Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE, USE_PREDICATES>;

    // Ctor.
    inline __device__ Smem_tile_a(char *smem, int tidx) : Base(smem, tidx) {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The size of the STS.
    int BYTES_PER_STS,
    // The number of buffers per tile.
    int BUFFERS_PER_TILE
>
struct Smem_tile_a<Ampere_spimma_gelu_int8_traits, Cta_tile, Col, BYTES_PER_STS, BUFFERS_PER_TILE>
    : public Smem_tile_sparse_ampere_col_a<Ampere_spimma_gelu_int8_traits,
                                    Cta_tile,
                                    BYTES_PER_STS,
                                    BUFFERS_PER_TILE> {

    // The traits class.
    using Traits = Ampere_spimma_gelu_int8_traits;
    // The base class.
    using Base = Smem_tile_sparse_ampere_col_a<Traits, Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE>;

    // Ctor.
    inline __device__ Smem_tile_a(char *smem, int tidx) : Base(smem, tidx) {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The size of the STS.
    int BYTES_PER_STS,
    // The number of buffers per tile.
    int BUFFERS_PER_TILE,
    // Use or not predicates
    bool USE_PREDICATES
>
struct Smem_tile_b<Ampere_spimma_gelu_int8_traits, Cta_tile, Col, BYTES_PER_STS, BUFFERS_PER_TILE, USE_PREDICATES>
    : public Smem_tile_sparse_ampere_col_b<Ampere_spimma_gelu_int8_traits,
                                    Cta_tile,
                                    BYTES_PER_STS,
                                    BUFFERS_PER_TILE,
                                    USE_PREDICATES> {
    // The traits class.
    using Traits = Ampere_spimma_gelu_int8_traits;
    // The base class.
    using Base = Smem_tile_sparse_ampere_col_b<Traits, Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE, USE_PREDICATES>;

    // Ctor.
    inline __device__ Smem_tile_b(char *smem, int tidx) : Base(smem, tidx) {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The size of the STS.
    int BYTES_PER_STS,
    // The number of buffers per tile.
    int BUFFERS_PER_TILE
>
struct Smem_tile_b<Ampere_spimma_gelu_int8_traits, Cta_tile, Row, BYTES_PER_STS, BUFFERS_PER_TILE>
    : public Smem_tile_sparse_ampere_row_b<Ampere_spimma_gelu_int8_traits,
                                    Cta_tile,
                                    BYTES_PER_STS,
                                    BUFFERS_PER_TILE> {

    // The traits class.
    using Traits = Ampere_spimma_gelu_int8_traits;
    // The base class.
    using Base = Smem_tile_sparse_ampere_row_b<Traits, Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE>;

    // Ctor.
    inline __device__ Smem_tile_b(char *smem, int tidx) : Base(smem, tidx) {
    }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

template<
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The size of the STS.
    int BYTES_PER_STS,
    // The number of buffers per tile.
    int BUFFERS_PER_TILE
>
struct Smem_tile_e<Ampere_spimma_interleaved_gelu_int8_traits, Cta_tile, Row, BYTES_PER_STS, BUFFERS_PER_TILE>
    : public Smem_tile_sparse_ampere_row_e<Ampere_spimma_interleaved_gelu_int8_traits,
                                    Cta_tile,
                                    BYTES_PER_STS,
                                    BUFFERS_PER_TILE,
                                    false> {

    // The traits class.
    using Traits = Ampere_spimma_interleaved_gelu_int8_traits;
    // The base class.
    using Base = Smem_tile_sparse_ampere_row_e<Traits, Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE, false>;

    // Ctor.
    inline __device__ Smem_tile_e(char *smem, int tidx)
        : Base(smem, tidx) {
    }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

template<
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The size of the STS.
    int BYTES_PER_STS,
    // The number of buffers per tile.
    int BUFFERS_PER_TILE
>
struct Smem_tile_lds_e<Ampere_spimma_interleaved_gelu_int8_traits, Cta_tile, Row, BYTES_PER_STS, BUFFERS_PER_TILE>
    : public Smem_tile_sparse_ampere_row_e<Ampere_spimma_interleaved_gelu_int8_traits,
                                    Cta_tile,
                                    BYTES_PER_STS,
                                    BUFFERS_PER_TILE,
                                    true> {

    // The traits class.
    using Traits = Ampere_spimma_interleaved_gelu_int8_traits;
    // The base class.
    using Base = Smem_tile_sparse_ampere_row_e<Traits, Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE, true>;

    // Ctor.
    inline __device__ Smem_tile_lds_e(char *smem, int tidx)
        : Base(smem, tidx) {
    }
};


////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The size of the STS.
    int BYTES_PER_STS,
    // The number of buffers per tile.
    int BUFFERS_PER_TILE,
    // Use or not predicates
    bool USE_PREDICATES
>
struct Smem_tile_a<Ampere_spimma_interleaved_gelu_int8_traits, Cta_tile, Row, BYTES_PER_STS, BUFFERS_PER_TILE, USE_PREDICATES>
    : public Smem_tile_sparse_ampere_row_a<Ampere_spimma_interleaved_gelu_int8_traits,
                                    Cta_tile,
                                    BYTES_PER_STS,
                                    BUFFERS_PER_TILE,
                                    USE_PREDICATES> {
    // The traits class.
    using Traits = Ampere_spimma_interleaved_gelu_int8_traits;
    // The base class.
    using Base = Smem_tile_sparse_ampere_row_a<Traits, Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE, USE_PREDICATES>;

    // Ctor.
    inline __device__ Smem_tile_a(char *smem, int tidx) : Base(smem, tidx) {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The size of the STS.
    int BYTES_PER_STS,
    // The number of buffers per tile.
    int BUFFERS_PER_TILE
>
struct Smem_tile_a<Ampere_spimma_interleaved_gelu_int8_traits, Cta_tile, Col, BYTES_PER_STS, BUFFERS_PER_TILE>
    : public Smem_tile_sparse_ampere_col_a<Ampere_spimma_interleaved_gelu_int8_traits,
                                    Cta_tile,
                                    BYTES_PER_STS,
                                    BUFFERS_PER_TILE> {

    // The traits class.
    using Traits = Ampere_spimma_interleaved_gelu_int8_traits;
    // The base class.
    using Base = Smem_tile_sparse_ampere_col_a<Traits, Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE>;

    // Ctor.
    inline __device__ Smem_tile_a(char *smem, int tidx) : Base(smem, tidx) {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The size of the STS.
    int BYTES_PER_STS,
    // The number of buffers per tile.
    int BUFFERS_PER_TILE,
    // Use or not predicates
    bool USE_PREDICATES
>
struct Smem_tile_b<Ampere_spimma_interleaved_gelu_int8_traits, Cta_tile, Col, BYTES_PER_STS, BUFFERS_PER_TILE, USE_PREDICATES>
    : public Smem_tile_sparse_ampere_interleaved_col_b<Ampere_spimma_interleaved_gelu_int8_traits,
                                    Cta_tile,
                                    BYTES_PER_STS,
                                    BUFFERS_PER_TILE,
                                    USE_PREDICATES> {
    // The traits class.
    using Traits = Ampere_spimma_interleaved_gelu_int8_traits;
    // The base class.
    using Base = Smem_tile_sparse_ampere_interleaved_col_b<Traits, Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE, USE_PREDICATES>;

    // Ctor.
    inline __device__ Smem_tile_b(char *smem, int tidx) : Base(smem, tidx) {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The size of the STS.
    int BYTES_PER_STS,
    // The number of buffers per tile.
    int BUFFERS_PER_TILE
>
struct Smem_tile_b<Ampere_spimma_interleaved_gelu_int8_traits, Cta_tile, Row, BYTES_PER_STS, BUFFERS_PER_TILE>
    : public Smem_tile_sparse_ampere_row_b<Ampere_spimma_interleaved_gelu_int8_traits,
                                    Cta_tile,
                                    BYTES_PER_STS,
                                    BUFFERS_PER_TILE> {

    // The traits class.
    using Traits = Ampere_spimma_interleaved_gelu_int8_traits;
    // The base class.
    using Base = Smem_tile_sparse_ampere_row_b<Traits, Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE>;

    // Ctor.
    inline __device__ Smem_tile_b(char *smem, int tidx) : Base(smem, tidx) {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// S P A R S E  H M M A . B F 1 6 . F 3 2 . B F 1 6
//
////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The size of the STS.
    int BYTES_PER_STS,
    // The number of buffers per tile.
    int BUFFERS_PER_TILE,
    // Use or not predicates
    bool USE_PREDICATES
>
struct Smem_tile_a<Ampere_sphmma_bf16_fp32_bf16_traits, Cta_tile, Row, BYTES_PER_STS, BUFFERS_PER_TILE, USE_PREDICATES>
    : public Smem_tile_sparse_ampere_row_a<Ampere_sphmma_bf16_fp32_bf16_traits,
                                    Cta_tile,
                                    BYTES_PER_STS,
                                    BUFFERS_PER_TILE,
                                    USE_PREDICATES> {
    // The traits class.
    using Traits = Ampere_sphmma_bf16_fp32_bf16_traits;
    // The base class.
    using Base = Smem_tile_sparse_ampere_row_a<Traits, Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE, USE_PREDICATES>;

    // Ctor.
    inline __device__ Smem_tile_a(char *smem, int tidx) : Base(smem, tidx) {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The size of the STS.
    int BYTES_PER_STS,
    // The number of buffers per tile.
    int BUFFERS_PER_TILE
>
struct Smem_tile_a<Ampere_sphmma_bf16_fp32_bf16_traits, Cta_tile, Col, BYTES_PER_STS, BUFFERS_PER_TILE>
    : public Smem_tile_sparse_ampere_col_a<Ampere_sphmma_bf16_fp32_bf16_traits,
                                    Cta_tile,
                                    BYTES_PER_STS,
                                    BUFFERS_PER_TILE> {

    // The traits class.
    using Traits = Ampere_sphmma_bf16_fp32_bf16_traits;
    // The base class.
    using Base = Smem_tile_sparse_ampere_col_a<Traits, Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE>;

    // Ctor.
    inline __device__ Smem_tile_a(char *smem, int tidx) : Base(smem, tidx) {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The size of the STS.
    int BYTES_PER_STS,
    // The number of buffers per tile.
    int BUFFERS_PER_TILE,
    // Use or not predicates
    bool USE_PREDICATES
>
struct Smem_tile_b<Ampere_sphmma_bf16_fp32_bf16_traits, Cta_tile, Col, BYTES_PER_STS, BUFFERS_PER_TILE, USE_PREDICATES>
    : public Smem_tile_sparse_ampere_col_b<Ampere_sphmma_bf16_fp32_bf16_traits,
                                    Cta_tile,
                                    BYTES_PER_STS,
                                    BUFFERS_PER_TILE,
                                    USE_PREDICATES> {
    // The traits class.
    using Traits = Ampere_sphmma_bf16_fp32_bf16_traits;
    // The base class.
    using Base = Smem_tile_sparse_ampere_col_b<Traits, Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE, USE_PREDICATES>;

    // Ctor.
    inline __device__ Smem_tile_b(char *smem, int tidx) : Base(smem, tidx) {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The size of the STS.
    int BYTES_PER_STS,
    // The number of buffers per tile.
    int BUFFERS_PER_TILE
>
struct Smem_tile_b<Ampere_sphmma_bf16_fp32_bf16_traits, Cta_tile, Row, BYTES_PER_STS, BUFFERS_PER_TILE>
    : public Smem_tile_sparse_ampere_row_b<Ampere_sphmma_bf16_fp32_bf16_traits,
                                    Cta_tile,
                                    BYTES_PER_STS,
                                    BUFFERS_PER_TILE> {

    // The traits class.
    using Traits = Ampere_sphmma_bf16_fp32_bf16_traits;
    // The base class.
    using Base = Smem_tile_sparse_ampere_row_b<Traits, Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE>;

    // Ctor.
    inline __device__ Smem_tile_b(char *smem, int tidx) : Base(smem, tidx) {
    }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

template<
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The size of the STS.
    int BYTES_PER_STS,
    // The number of buffers per tile.
    int BUFFERS_PER_TILE
>
struct Smem_tile_e<Ampere_sphmma_bf16_fp32_bf16_traits, Cta_tile, Row, BYTES_PER_STS, BUFFERS_PER_TILE>
    : public Smem_tile_sparse_ampere_row_e<Ampere_sphmma_bf16_fp32_bf16_traits,
                                    Cta_tile,
                                    BYTES_PER_STS,
                                    BUFFERS_PER_TILE,
                                    false> {

    // The traits class.
    using Traits = Ampere_sphmma_bf16_fp32_bf16_traits;
    // The base class.
    using Base = Smem_tile_sparse_ampere_row_e<Traits, Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE, false>;

    // Ctor.
    inline __device__ Smem_tile_e(char *smem, int tidx)
        : Base(smem, tidx) {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    // Traits of tensor-core.
    typename Traits,
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // Acc storage type
    typename Storage_type, 
    // Output storage type
    typename Output_storage_type, 
    // Layout of storage.
    typename Layout,
    // Bytes per LDS.
    int BYTES_PER_LDS_ = 16
>
struct Swizzle_ampere_hmma_32b_epilogue {};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    // Traits of tensor-core.
    typename Traits,
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // Acc storage type
    typename Storage_type, 
    // Output storage type
    typename Output_storage_type, 
    // Bytes per lds.
    int BYTES_PER_LDS_
>
struct Swizzle_ampere_hmma_32b_epilogue<Traits, 
                                        Cta_tile, 
                                        Col, 
                                        Storage_type, 
                                        Output_storage_type, 
                                        BYTES_PER_LDS_> {

    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;

    // ACC count per thread
    enum { ACC_COUNT = 2 };
    // ACC Stride
    enum { ACC_STRIDE = 8 };
    // Stored Element Size
    enum { ELEMENT_SIZE = sizeof(Storage_type) };
    // STG output element size
    enum { STG_OUT_ELEMENT_SIZE = sizeof(Output_storage_type) };
    // Acc/output sizing ratio
    enum { ACC_OUT_RATIO = ELEMENT_SIZE / STG_OUT_ELEMENT_SIZE };
    // The number of bytes for key memory instruction.
    enum { BYTES_PER_LDS = BYTES_PER_LDS_, BYTES_PER_STS = ACC_COUNT * ELEMENT_SIZE };
    // The number of elements per LDS.
    enum { ELEMENTS_PER_LDS = BYTES_PER_LDS / ELEMENT_SIZE };
    // The number of rows in shared memory.
    enum { ROWS = Xmma_tile::N_PER_XMMA_PER_CTA };
    // The size of a single row in bytes.
    enum { BYTES_PER_ROW = Cta_tile::M * Cta_tile::WARPS_K * ELEMENT_SIZE };
    // The skew to avoid bank conflicts.
    //enum { BYTES_PER_SKEW = BYTES_PER_STS * 4 };
    enum { BYTES_PER_SKEW = 16 };
    // The number of columns with the skew.
    enum { BYTES_PER_ROW_WITH_SKEW = BYTES_PER_ROW + BYTES_PER_SKEW };
    // The size in bytes in shared memory.
    enum { BYTES_PER_TILE = ROWS * BYTES_PER_ROW_WITH_SKEW };

    // The number of threads per output row. Each thread writes 8 elements.
    enum { THREADS_PER_ROW =
        Min<Cta_tile::THREADS_PER_CTA, Cta_tile::M / (ELEMENTS_PER_LDS * ACC_OUT_RATIO) >::VALUE };

    // The number of column loaded per STG
    enum { COLUMNS_PER_STG = THREADS_PER_ROW * ELEMENTS_PER_LDS };

    // The number of rows written in one STG.128.
    enum { ROWS_PER_STG = Cta_tile::THREADS_PER_CTA / THREADS_PER_ROW };
    static_assert(ROWS_PER_STG > 0, "");

    // The number of steps needed to load the columns.
    enum { STGS_PER_COLUMN = Xmma_tile::N_PER_XMMA_PER_CTA / ROWS_PER_STG };
    // Do we skip the syncthreads in the epilogue? Of course, not :)
    enum { SKIP_SYNCTHREADS = 0 };


    // Ctor.
    inline __device__ Swizzle_ampere_hmma_32b_epilogue(void *smem, int tidx)
        : smem_(get_smem_pointer(smem)) {
      
        // The number of warps.
        const int WARPS_M = Cta_tile::WARPS_M;
        const int WARPS_N = Cta_tile::WARPS_N;
        const int WARPS_K = Cta_tile::WARPS_K;

        // The masks to select the warps.
        const int WARP_MASK_M = Warp_masks<WARPS_M, WARPS_N, WARPS_K>::M;
        const int WARP_MASK_N = Warp_masks<WARPS_M, WARPS_N, WARPS_K>::N;
        //const int WARP_MASK_K = Warp_masks<WARPS_M, WARPS_N, WARPS_K>::K;

        // The divisor for the warps.
        const int WARP_DIV_M =       1 *       1 * Cta_tile::THREADS_PER_WARP;
        const int WARP_DIV_N = WARPS_M *       1 * Cta_tile::THREADS_PER_WARP;
        //const int WARP_DIV_K = WARPS_M * WARPS_N * Cta_tile::THREADS_PER_WARP;

        // Compute the row and the column in shared memory. Each warp reads from a 16*16B segment.
        const int smem_write_row = (tidx & WARP_MASK_N) / WARP_DIV_N * Xmma_tile::N_PER_XMMA +
                                   (tidx % 4) * 2;

        const int smem_write_col = (tidx & WARP_MASK_M) / WARP_DIV_M * Traits::ROW_STRIDE_GROUP + 
                                   ((tidx % 32) / 4) ;

        smem_write_offset_ = smem_write_row*BYTES_PER_ROW_WITH_SKEW + smem_write_col*ELEMENT_SIZE;

        // The row and column read by a single thread.
        const int smem_read_row = tidx / THREADS_PER_ROW;
        const int smem_read_col = tidx % THREADS_PER_ROW * BYTES_PER_LDS * ACC_OUT_RATIO;

        // The corresponding offset.
        smem_read_offset_ = smem_read_row*BYTES_PER_ROW_WITH_SKEW + smem_read_col;

    }

    // Load from the tile in shared memory.
    template< typename Fragment_post_swizzle >
    inline __device__ void load(Fragment_post_swizzle &dst, int oi) const {
        int offset =
            (oi % STGS_PER_COLUMN) * ROWS_PER_STG * BYTES_PER_ROW_WITH_SKEW
            + (oi / STGS_PER_COLUMN) * COLUMNS_PER_STG * ELEMENT_SIZE;

        #pragma unroll
        for( int ki = 0; ki < Cta_tile::WARPS_K; ++ki ) {
            // Compute the address to load from.
            uint32_t ptr = smem_ + smem_read_offset_ + offset + ki * Cta_tile::M * ELEMENT_SIZE;

            if (BYTES_PER_LDS == 4) {
                // Load from shared memory.
                uint32_t tmp;
                lds(tmp, ptr);

                // Add the elements to the fragment.
                dst.reg(ki*4 + 0) = tmp;
            } else if (BYTES_PER_LDS == 8) {
                // Load from shared memory.
                uint2 tmp;
                lds(tmp, ptr);

                // Add the elements to the fragment.
                dst.reg(ki*4 + 0) = tmp.x;
                dst.reg(ki*4 + 1) = tmp.y;
            } else {
                // Load from shared memory.
                uint4 tmp;
                lds(tmp, ptr);

                // Add the elements to the fragment.
                dst.reg(ki*4 + 0) = tmp.x;
                dst.reg(ki*4 + 1) = tmp.y;
                dst.reg(ki*4 + 2) = tmp.z;
                dst.reg(ki*4 + 3) = tmp.w;
            }
        }

    }

    // Store to the tile in shared memory.
    template<typename Fragment_pre_swizzle>
    inline __device__ void store(int ni, const Fragment_pre_swizzle &c) {

        #pragma unroll
        for( int mi = 0; mi < 2; ++mi ) {
            // The row offset. 8 rows are written per iteration of mi.
            int row_offset = mi * 8 * ELEMENT_SIZE;

            // The column offset. As many columns as the CTA-wide XMMA in the N dimension.
            int stride = Xmma_tile::M_PER_XMMA; // Not Xmma_tile::M_PER_XMMA_PER_CTA
            int col_offset = ni * stride * ELEMENT_SIZE;

            // The base pointer.
            uint32_t ptr = smem_ + smem_write_offset_ + (row_offset + col_offset);

            // Store the 4 elements per thread in 2 STS.64.
            sts(ptr +  0 * BYTES_PER_ROW_WITH_SKEW, c.reg(4*mi+0));
            sts(ptr +  1 * BYTES_PER_ROW_WITH_SKEW, c.reg(4*mi+1));
            sts(ptr +  8 * BYTES_PER_ROW_WITH_SKEW, c.reg(4*mi+2));
            sts(ptr +  9 * BYTES_PER_ROW_WITH_SKEW, c.reg(4*mi+3));

        }

    }

    // The shared memory pointer in bytes.
    uint32_t smem_;
    // The write offset.
    int smem_write_offset_;
    // The read offset.
    int smem_read_offset_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    // Traits of tensor-core.
    typename Traits,
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // Acc storage type
    typename Storage_type, 
    // Output storage type
    typename Output_storage_type, 
    // Bytes per lds.
    int BYTES_PER_LDS_
>
struct Swizzle_ampere_hmma_32b_epilogue<Traits, 
                                        Cta_tile, 
                                        Row, 
                                        Storage_type, 
                                        Output_storage_type, 
                                        BYTES_PER_LDS_> {
    
    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;

    // ACC count per thread
    enum { ACC_COUNT = 2 };
    // ACC Stride
    enum { ACC_STRIDE = 8 };
    // Stored Element Size
    enum { ELEMENT_SIZE = sizeof(Storage_type) };
    // STG output element size
    enum { STG_OUT_ELEMENT_SIZE = sizeof(Output_storage_type) };
    // Acc/output sizing ratio
    enum { ACC_OUT_RATIO = ELEMENT_SIZE / STG_OUT_ELEMENT_SIZE };
    // The number of bytes for key memory instruction.
    enum { BYTES_PER_LDS = BYTES_PER_LDS_, BYTES_PER_STS = ACC_COUNT * ELEMENT_SIZE };
    // The number of elements per LDS.
    enum { ELEMENTS_PER_LDS = BYTES_PER_LDS / ELEMENT_SIZE };
    // The number of rows in shared memory.
    enum { ROWS = Xmma_tile::M_PER_XMMA_PER_CTA };
    // The size of a single row in bytes.
    enum { BYTES_PER_ROW = Cta_tile::N * ELEMENT_SIZE };
    // The skew to avoid bank conflicts.
    enum { BYTES_PER_SKEW = BYTES_PER_STS * 4 };
    // The number of columns with the skew.
    enum { BYTES_PER_ROW_WITH_SKEW = BYTES_PER_ROW + BYTES_PER_SKEW };
    // The size in bytes in shared memory.
    enum { BYTES_PER_TILE = ROWS * BYTES_PER_ROW_WITH_SKEW };

    // The number of threads per output row. 
    enum { THREADS_PER_ROW_ = Cta_tile::N / (ELEMENTS_PER_LDS * ACC_OUT_RATIO) };
    // The number of threads per output row. 
    enum { THREADS_PER_ROW = Min<Cta_tile::THREADS_PER_CTA, THREADS_PER_ROW_>::VALUE };

    // The number of rows written in one STG.128.
    enum { ROWS_PER_STG = Cta_tile::THREADS_PER_CTA / THREADS_PER_ROW };
    // Make sure we store at least 1 row per STG.
    static_assert(ROWS_PER_STG > 0, "");

    // Do we skip the syncthreads in the epilogue? Of course, not :)
    enum { SKIP_SYNCTHREADS = 0 };

    // Ctor.
    inline __device__ Swizzle_ampere_hmma_32b_epilogue(void *smem, int tidx)
        : smem_(get_smem_pointer(smem)) {

        // The number of warps.
        const int WARPS_M = Cta_tile::WARPS_M;
        const int WARPS_N = Cta_tile::WARPS_N;
        const int WARPS_K = Cta_tile::WARPS_K;

        // The masks to select the warps.
        const int WARP_MASK_M = Warp_masks<WARPS_M, WARPS_N, WARPS_K>::M;
        const int WARP_MASK_N = Warp_masks<WARPS_M, WARPS_N, WARPS_K>::N;
        //const int WARP_MASK_K = Warp_masks<WARPS_M, WARPS_N, WARPS_K>::K;

        // The divisor for the warps.
        const int WARP_DIV_M =       1 *       1 * Cta_tile::THREADS_PER_WARP;
        const int WARP_DIV_N = WARPS_M *       1 * Cta_tile::THREADS_PER_WARP;
        //const int WARP_DIV_K = WARPS_M * WARPS_N * Cta_tile::THREADS_PER_WARP;

        const int smem_write_row = (tidx & WARP_MASK_M) / WARP_DIV_M * Xmma_tile::M_PER_XMMA 
                                   + (tidx & 0x1c) / 4;
        const int smem_write_col = (tidx & WARP_MASK_N) / WARP_DIV_N * Xmma_tile::N_PER_XMMA
                                   + (tidx & 0x03) * 2;

        // The corresponding offset.
        smem_write_offset_ = smem_write_row*BYTES_PER_ROW_WITH_SKEW + smem_write_col * ELEMENT_SIZE;

        // The row and column read by a single thread.
        const int smem_read_row = tidx / THREADS_PER_ROW;
        const int smem_read_col = tidx % THREADS_PER_ROW * BYTES_PER_LDS * ACC_OUT_RATIO;

        // The corresponding offset.
        smem_read_offset_ = smem_read_row*BYTES_PER_ROW_WITH_SKEW + smem_read_col;
    }

    // Load from the tile in shared memory.
    template< typename Fragment_post_swizzle >
    inline __device__ void load(Fragment_post_swizzle &dst, int oi) const {
        int offset = oi * ROWS_PER_STG * BYTES_PER_ROW_WITH_SKEW;

        #pragma unroll
        for( int ki = 0; ki < Cta_tile::WARPS_K; ++ki ) {
            // Compute the address to load from.
            uint32_t ptr = smem_ + smem_read_offset_ + offset + ki * Cta_tile::N * ELEMENT_SIZE;

            // Load from shared memory.
            uint4 tmp;
            lds(tmp, ptr);

            // Add the elements to the fragment.
            dst.reg(ki*4 + 0) = tmp.x;
            dst.reg(ki*4 + 1) = tmp.y;
            dst.reg(ki*4 + 2) = tmp.z;
            dst.reg(ki*4 + 3) = tmp.w;
        }
    }

    // Store to the tile in shared memory.
    template<typename Fragment_pre_swizzle>
    inline __device__ void store(int ni, const Fragment_pre_swizzle &c) {

        #pragma unroll
        for( int mi = 0; mi < 2; ++mi ) {
            // The row offset. 8 rows are written per iteration of mi.
            int row_offset = mi * ACC_STRIDE * BYTES_PER_ROW_WITH_SKEW;
            // The column offset. As many columns as the CTA-wide XMMA in the N dimension.
            int stride = Xmma_tile::N_PER_XMMA_PER_CTA;
            int col_offset = ni * stride * ELEMENT_SIZE;

            // The base pointer.
            uint32_t ptr = smem_ + smem_write_offset_ + (row_offset + col_offset);

            // Store the 4 elements per thread in 2 STS.64.
            sts(ptr +  0, make_uint2(c.reg(4*mi+0), c.reg(4*mi+1)));
            sts(ptr + 32, make_uint2(c.reg(4*mi+2), c.reg(4*mi+3)));
        }

    }

    // The shared memory pointer in bytes.
    uint32_t smem_;
    // The write offset.
    int smem_write_offset_;
    // The read offset.
    int smem_read_offset_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Cta_tile >
struct Swizzle_epilogue<Ampere_sphmma_fp32_traits, Cta_tile, xmma::Row, 16, false> 
    : public Swizzle_ampere_hmma_32b_epilogue<Ampere_sphmma_fp32_traits, 
                                              Cta_tile, 
                                              xmma::Row, 
                                              float,
                                              lwtlass::half_t,
                                              16> {
    // The base class.
    using Base = Swizzle_ampere_hmma_32b_epilogue<Ampere_sphmma_fp32_traits, 
                                                  Cta_tile, 
                                                  xmma::Row, 
                                                  float,
                                                  lwtlass::half_t,
                                                  16>;

    // Ctor.
    inline __device__ Swizzle_epilogue(char *smem, int tidx) : Base(smem, tidx) {
    }
    
    // Load from the tile in shared memory.
    template< typename Fragment_post_swizzle >
    inline __device__ void load(Fragment_post_swizzle &dst, int oi) const {
        int offset = oi * Base::ROWS_PER_STG * Base::BYTES_PER_ROW_WITH_SKEW;

        #pragma unroll
        for( int ki = 0; ki < Cta_tile::WARPS_K; ++ki ) {
            // Compute the address to load from.
            uint32_t ptr = Base::smem_ + Base::smem_read_offset_ + 
                           offset + 
                           ki * Cta_tile::N * Base::ELEMENT_SIZE;

            // Load from shared memory.
            uint4 tmp;
            lds(tmp, ptr);

            // Add the elements to the fragment.
            dst.reg(ki*8 + 0) = tmp.x;
            dst.reg(ki*8 + 1) = tmp.y;
            dst.reg(ki*8 + 2) = tmp.z;
            dst.reg(ki*8 + 3) = tmp.w;
            //Steve

            uint4 tmp1;
            lds(tmp1, ptr + 16);

            // Add the elements to the fragment.
            dst.reg(ki*8 + 4) = tmp1.x;
            dst.reg(ki*8 + 5) = tmp1.y;
            dst.reg(ki*8 + 6) = tmp1.z;
            dst.reg(ki*8 + 7) = tmp1.w;

        }
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Cta_tile >
struct Swizzle_epilogue<Ampere_sphmma_fp32_traits, Cta_tile, xmma::Col, 16, false> 
    : public Swizzle_ampere_hmma_32b_epilogue<Ampere_sphmma_fp32_traits, 
                                              Cta_tile, 
                                              xmma::Col, 
                                              float,
                                              lwtlass::half_t,
                                              16> {
    // The base class.
    using Base = Swizzle_ampere_hmma_32b_epilogue<Ampere_sphmma_fp32_traits, 
                                                  Cta_tile, 
                                                  xmma::Col, 
                                                  float,
                                                  lwtlass::half_t,
                                                  16>;

    // Ctor.
    inline __device__ Swizzle_epilogue(char *smem, int tidx) : Base(smem, tidx) {
    }
    
    // Load from the tile in shared memory.
    template< typename Fragment_post_swizzle >
    inline __device__ void load(Fragment_post_swizzle &dst, int oi) const {
        const int offset = oi * Base::ROWS_PER_STG * Base::BYTES_PER_ROW_WITH_SKEW;

        #pragma unroll
        for( int ki = 0; ki < Cta_tile::WARPS_K; ++ki ) {
            // Compute the address to load from.
            uint32_t ptr = Base::smem_ + Base::smem_read_offset_ + 
                           offset + 
                           ki * Cta_tile::M * Base::ELEMENT_SIZE;

            uint4 tmp;
            lds(tmp, ptr);

            // Add the elements to the fragment.
            dst.reg(ki*8 + 0) = tmp.x;
            dst.reg(ki*8 + 1) = tmp.y;
            dst.reg(ki*8 + 2) = tmp.z;
            dst.reg(ki*8 + 3) = tmp.w;
                
            // Load from shared memory.
            uint4 tmp1;
            lds(tmp1, ptr + 16);

            // Add the elements to the fragment.
            dst.reg(ki*8 + 4) = tmp1.x;
            dst.reg(ki*8 + 5) = tmp1.y;
            dst.reg(ki*8 + 6) = tmp1.z;
            dst.reg(ki*8 + 7) = tmp1.w;
        }
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Cta_tile >
struct Swizzle_epilogue<Ampere_sphmma_bf16_fp32_bf16_traits , Cta_tile, xmma::Row, 16, false> 
    : public Swizzle_epilogue<Ampere_sphmma_fp32_traits, Cta_tile, xmma::Row, 16, false> {
    
    // The base class.
    using Base = Swizzle_epilogue<Ampere_sphmma_fp32_traits, Cta_tile, xmma::Row, 16, false>;

    // Ctor.
    inline __device__ Swizzle_epilogue(char *smem, int tidx) : Base(smem, tidx) {
    }
    
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Cta_tile >
struct Swizzle_epilogue<Ampere_sphmma_bf16_fp32_bf16_traits , Cta_tile, xmma::Col, 16, false> 
    : public Swizzle_epilogue<Ampere_sphmma_fp32_traits, Cta_tile, xmma::Col, 16, false> {
    
    // The base class.
    using Base = Swizzle_epilogue<Ampere_sphmma_fp32_traits, Cta_tile, xmma::Col, 16, false>;

    // Ctor.
    inline __device__ Swizzle_epilogue(char *smem, int tidx) : Base(smem, tidx) {
    }

};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Cta_tile >
struct Swizzle_epilogue<xmma::Ampere_sphmma_tf32_traits<float, float>, 
                        Cta_tile, 
                        xmma::Col, 
                        16, 
                        false> 
    : public Swizzle_ampere_hmma_32b_epilogue<xmma::Ampere_sphmma_tf32_traits<float, float>, 
                                              Cta_tile, 
                                              xmma::Col, 
                                              float,
                                              float,
                                              16> {
    
    // The base class.
    using Base = Swizzle_ampere_hmma_32b_epilogue<xmma::Ampere_sphmma_tf32_traits<float, float>, 
                                                  Cta_tile, 
                                                  xmma::Col, 
                                                  float,
                                                  float,
                                                  16>;

    // Ctor.
    inline __device__ Swizzle_epilogue(char *smem, int tidx) : Base(smem, tidx) {
    }

};

template< typename Cta_tile >
struct Swizzle_epilogue<xmma::Ampere_sphmma_tf32_traits<float, float>, 
                        Cta_tile, 
                        xmma::Row, 
                        16, 
                        false> 
    : public Swizzle_ampere_hmma_32b_epilogue<xmma::Ampere_sphmma_tf32_traits<float, float>, 
                                              Cta_tile, 
                                              xmma::Row, 
                                              float,
                                              float,
                                              16> {
    
    // The base class.
    using Base = Swizzle_ampere_hmma_32b_epilogue<xmma::Ampere_sphmma_tf32_traits<float, float>, 
                                                  Cta_tile, 
                                                  xmma::Row, 
                                                  float,
                                                  float,
                                                  16>;

    // Ctor.
    inline __device__ Swizzle_epilogue(char *smem, int tidx) : Base(smem, tidx) {
    }

};

///////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Cta_tile >
struct Swizzle_epilogue<xmma::Ampere_sphmma_tf32_traits<lwtlass::float_tf32_t, lwtlass::float_tf32_t>,
                        Cta_tile,
                        xmma::Col,
                        16,
                        false>
    : public Swizzle_ampere_hmma_32b_epilogue<xmma::Ampere_sphmma_tf32_traits<lwtlass::float_tf32_t, lwtlass::float_tf32_t>, 
                                              Cta_tile, 
                                              xmma::Col, 
                                              float,
                                              lwtlass::float_tf32_t,
                                              16> {
    
    // The base class.
    using Base = Swizzle_ampere_hmma_32b_epilogue<xmma::Ampere_sphmma_tf32_traits<lwtlass::float_tf32_t, lwtlass::float_tf32_t>, 
                                                  Cta_tile,                                                  
                                                  xmma::Col, 
                                                  float,
                                                  lwtlass::float_tf32_t,
                                                  16>;

    // Ctor.
    inline __device__ Swizzle_epilogue(char *smem, int tidx) : Base(smem, tidx) {
    }

};

///////////////////////////////////////////////////////////////////////////////////////////////////


template< typename Cta_tile >
struct Swizzle_epilogue<xmma::Ampere_sphmma_tf32_traits<lwtlass::float_tf32_t, lwtlass::float_tf32_t>,
                        Cta_tile,
                        xmma::Row,
                        16,
                        false>
    : public Swizzle_ampere_hmma_32b_epilogue<xmma::Ampere_sphmma_tf32_traits<lwtlass::float_tf32_t, lwtlass::float_tf32_t>, 
                                              Cta_tile, 
                                              xmma::Row, 
                                              float,
                                              lwtlass::float_tf32_t,
                                              16> {
    
    // The base class.
    using Base = Swizzle_ampere_hmma_32b_epilogue<xmma::Ampere_sphmma_tf32_traits<lwtlass::float_tf32_t, lwtlass::float_tf32_t>, 
                                                  Cta_tile, 
                                                  xmma::Row, 
                                                  float,
                                                  lwtlass::float_tf32_t,
                                                  16>;

    // Ctor.
    inline __device__ Swizzle_epilogue(char *smem, int tidx) : Base(smem, tidx) {
    }

};

///////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Cta_tile >
struct Swizzle_epilogue<Ampere_spimma_int8_rt_fuse_traits , Cta_tile, xmma::Col, 16, false>
    : public Swizzle_sparse_ampere_epilogue<Ampere_spimma_int8_rt_fuse_traits ,
                                            Cta_tile,
                                            uint16_t,
                                            xmma::Col> {

    // The traits.
    using Traits = Ampere_spimma_int8_rt_fuse_traits ;
    // The base class.
    using Base = Swizzle_sparse_ampere_epilogue<Traits,
                                                Cta_tile,
                                                uint16_t,
                                                xmma::Col>;
    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;

    // Ctor.
    inline __device__ Swizzle_epilogue(char *smem, int tidx) : Base(smem, tidx) {
    }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Cta_tile >
struct Swizzle_epilogue<Ampere_spimma_int8_rt_fuse_traits , Cta_tile, xmma::Row, 16, false>
    : public Swizzle_sparse_ampere_epilogue<Ampere_spimma_int8_rt_fuse_traits ,
                                            Cta_tile,
                                            uint16_t,
                                            xmma::Row> {

    // The traits.
    using Traits = Ampere_spimma_int8_rt_fuse_traits ;
    // The base class.
    using Base = Swizzle_sparse_ampere_epilogue<Traits,
                                                Cta_tile,
                                                uint16_t,
                                                xmma::Row>;
    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;

    // Ctor.
    inline __device__ Swizzle_epilogue(char *smem, int tidx) : Base(smem, tidx) {
    }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

template<
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The size of the STS.
    int BYTES_PER_STS,
    // The number of buffers per tile.
    int BUFFERS_PER_TILE
>
struct Smem_tile_e<Ampere_spimma_int8_rt_fuse_traits, Cta_tile, Row, BYTES_PER_STS, BUFFERS_PER_TILE>
    : public Smem_tile_sparse_ampere_row_e<Ampere_spimma_int8_rt_fuse_traits,
                                    Cta_tile,
                                    BYTES_PER_STS,
                                    BUFFERS_PER_TILE,
                                    false> {

    // The traits class.
    using Traits = Ampere_spimma_int8_rt_fuse_traits;
    // The base class.
    using Base = Smem_tile_sparse_ampere_row_e<Traits, Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE, false>;

    // Ctor.
    inline __device__ Smem_tile_e(char *smem, int tidx)
        : Base(smem, tidx) {
    }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

template<
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The size of the STS.
    int BYTES_PER_STS,
    // The number of buffers per tile.
    int BUFFERS_PER_TILE
>
struct Smem_tile_lds_e<Ampere_spimma_int8_rt_fuse_traits, Cta_tile, Row, BYTES_PER_STS, BUFFERS_PER_TILE>
    : public Smem_tile_sparse_ampere_row_e<Ampere_spimma_int8_rt_fuse_traits,
                                    Cta_tile,
                                    BYTES_PER_STS,
                                    BUFFERS_PER_TILE,
                                    true> {

    // The traits class.
    using Traits = Ampere_spimma_int8_rt_fuse_traits;
    // The base class.
    using Base = Smem_tile_sparse_ampere_row_e<Traits, Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE, true>;

    // Ctor.
    inline __device__ Smem_tile_lds_e(char *smem, int tidx)
        : Base(smem, tidx) {
    }
};


////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The size of the STS.
    int BYTES_PER_STS,
    // The number of buffers per tile.
    int BUFFERS_PER_TILE,
    // Use or not predicates
    bool USE_PREDICATES
>
struct Smem_tile_a<Ampere_spimma_int8_rt_fuse_traits, Cta_tile, Row, BYTES_PER_STS, BUFFERS_PER_TILE, USE_PREDICATES>
    : public Smem_tile_sparse_ampere_row_a<Ampere_spimma_int8_rt_fuse_traits,
                                    Cta_tile,
                                    BYTES_PER_STS,
                                    BUFFERS_PER_TILE,
                                    USE_PREDICATES> {
    // The traits class.
    using Traits = Ampere_spimma_int8_rt_fuse_traits;
    // The base class.
    using Base = Smem_tile_sparse_ampere_row_a<Traits, Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE, USE_PREDICATES>;

    // Ctor.
    inline __device__ Smem_tile_a(char *smem, int tidx) : Base(smem, tidx) {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The size of the STS.
    int BYTES_PER_STS,
    // The number of buffers per tile.
    int BUFFERS_PER_TILE
>
struct Smem_tile_a<Ampere_spimma_int8_rt_fuse_traits, Cta_tile, Col, BYTES_PER_STS, BUFFERS_PER_TILE>
    : public Smem_tile_sparse_ampere_col_a<Ampere_spimma_int8_rt_fuse_traits,
                                    Cta_tile,
                                    BYTES_PER_STS,
                                    BUFFERS_PER_TILE> {

    // The traits class.
    using Traits = Ampere_spimma_int8_rt_fuse_traits;
    // The base class.
    using Base = Smem_tile_sparse_ampere_col_a<Traits, Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE>;

    // Ctor.
    inline __device__ Smem_tile_a(char *smem, int tidx) : Base(smem, tidx) {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The size of the STS.
    int BYTES_PER_STS,
    // The number of buffers per tile.
    int BUFFERS_PER_TILE,
    // Use or not predicates
    bool USE_PREDICATES
>
struct Smem_tile_b<Ampere_spimma_int8_rt_fuse_traits, Cta_tile, Col, BYTES_PER_STS, BUFFERS_PER_TILE, USE_PREDICATES>
    : public Smem_tile_sparse_ampere_col_b<Ampere_spimma_int8_rt_fuse_traits,
                                    Cta_tile,
                                    BYTES_PER_STS,
                                    BUFFERS_PER_TILE,
                                    USE_PREDICATES> {
    // The traits class.
    using Traits = Ampere_spimma_int8_rt_fuse_traits;
    // The base class.
    using Base = Smem_tile_sparse_ampere_col_b<Traits, Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE, USE_PREDICATES>;

    // Ctor.
    inline __device__ Smem_tile_b(char *smem, int tidx) : Base(smem, tidx) {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The size of the STS.
    int BYTES_PER_STS,
    // The number of buffers per tile.
    int BUFFERS_PER_TILE
>
struct Smem_tile_b<Ampere_spimma_int8_rt_fuse_traits, Cta_tile, Row, BYTES_PER_STS, BUFFERS_PER_TILE>
    : public Smem_tile_sparse_ampere_row_b<Ampere_spimma_int8_rt_fuse_traits,
                                    Cta_tile,
                                    BYTES_PER_STS,
                                    BUFFERS_PER_TILE> {

    // The traits class.
    using Traits = Ampere_spimma_int8_rt_fuse_traits;
    // The base class.
    using Base = Smem_tile_sparse_ampere_row_b<Traits, Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE>;

    // Ctor.
    inline __device__ Smem_tile_b(char *smem, int tidx) : Base(smem, tidx) {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace xmma
