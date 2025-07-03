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

#include <xmma/volta/fragment.h>
#include <xmma/warp_masks.h>
#include <xmma/smem_tile.h>

namespace xmma {

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// H M M A . F 1 6 / 3 2
//
////////////////////////////////////////////////////////////////////////////////////////////////////

template< int N_ >
struct Rows_per_xor_pattern_volta {
    enum { VALUE = N_ == 16 ? 1 : (N_ == 32 ? 2 : 4) };
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Traits, typename Cta_tile >
struct Cols_per_xor_pattern_volta {
    enum { VALUE = Traits::template Xmma_tile<Cta_tile>::ENABLE_LDS_FAST_PATH ? 2 : 1 };
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
    int ROWS_PER_XOR_PATTERN_ = Rows_per_xor_pattern_volta<Cta_tile::M>::VALUE,
    // How many cols to use for the XOR pattern to avoid bank conflicts?
    int COLS_PER_XOR_PATTERN_ = Cols_per_xor_pattern_volta<Traits, Cta_tile>::VALUE
>
struct Smem_tile_volta_col_a : public Smem_tile_without_skews<Cta_tile, 
                                                              Cta_tile::K, 
                                                              Cta_tile::M, 
                                                              16, 
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
                                         Cta_tile::M, 
                                         16, 
                                         BYTES_PER_STS,
                                         BUFFERS_PER_TILE,
                                         0, 
                                         ROWS_PER_XOR_PATTERN_,
                                         COLS_PER_XOR_PATTERN_>;
    // The fragment.
    using Fragment = Fragment_a<Traits, Col>;

    // The size of a single LDS in bytes.
    enum { BYTES_PER_LDS = Xmma_tile::ENABLE_LDS_FAST_PATH ? 16 : 8 };
    // The number of elements per LDS.
    enum { ELEMENTS_PER_LDS = BYTES_PER_LDS * 8 / Traits::BITS_PER_ELEMENT_A };

    // Ctor.
    inline __device__ Smem_tile_volta_col_a(void *smem, int tidx) : Base(smem, tidx) {

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

        // The row/col read by the thread.
        int smem_read_row, smem_read_col;

        static_assert(Base::ROWS_PER_XOR_PATTERN == 4 ||
                      Base::ROWS_PER_XOR_PATTERN == 2 || 
                      Base::ROWS_PER_XOR_PATTERN == 1, "");
            
        if( Base::ROWS_PER_XOR_PATTERN == 4 ) {
            smem_read_row = (tidx & WARP_MASK_K) / WARP_DIV_K * Xmma_tile::XMMAS_K * 8 + 
                            (tidx & 0x03);
            smem_read_col = (tidx & 0x03) * 2 + (tidx & 0x04) / 4;
        } else if( Base::ROWS_PER_XOR_PATTERN == 2 ) {
            smem_read_row = (tidx & WARP_MASK_K) / WARP_DIV_K * Xmma_tile::XMMAS_K * 4 + 
                            (tidx & 0x03) / 2;
            smem_read_col = (tidx & 0x02) * 1 + (tidx & 0x04) / 4 + (tidx & 0x01) * 8;
        } else if( Base::ROWS_PER_XOR_PATTERN == 1 ) {
            smem_read_row = (tidx & WARP_MASK_K) / WARP_DIV_K * Xmma_tile::XMMAS_K * 2;
            smem_read_col = (tidx & 0x03) * 4 + (tidx & 0x04) / 4;
        }

        // Each half-warp applies a different XOR pattern -- see the Excel document.
        smem_read_col ^= (tidx & WARP_MASK_M) / WARP_DIV_M * 4 + (tidx & 0x10) / 8;

        // The shared memory offset.
        this->smem_read_offset_ = smem_read_row*Base::BYTES_PER_ROW + smem_read_col*BYTES_PER_LDS;
    }

    // Rewind smem_read_offset for last LDS phase in main loop.-
    inline __device__ void reverse_smem_read_offset(int ki = 0) {
    }

    // Load from shared memory.
    inline __device__ void load(Fragment (&a)[Xmma_tile::XMMAS_M], int ki) {
        #pragma unroll
        for( int mi = 0; mi < Xmma_tile::XMMAS_M; ++mi ) {
            // Prepare the offset.
            int offset = ki * Base::ROWS_PER_XOR_PATTERN * Base::BYTES_PER_ROW * 2;
            if( Cta_tile::WARPS_M == 1 ) {
                offset += this->smem_read_offset_ ^ (mi % 2 == 0 ? 0 : (BYTES_PER_LDS * 4)); 
                offset += (mi >> 1) * (BYTES_PER_LDS * 4) * 2;
            } else {
                offset += this->smem_read_offset_ + (mi  ) * Xmma_tile::M_PER_XMMA_PER_CTA * 2;
            }

            // The load pointer.
            uint32_t ptr = this->smem_ + this->smem_read_buffer_ + offset;

            // Load the data using 2x LDS.64.
            uint2 tmp_0;
            lds(tmp_0, ptr);
            a[mi].reg(0) = tmp_0.x;
            a[mi].reg(1) = tmp_0.y;

            uint2 tmp_1; 
            lds(tmp_1, ptr + Base::ROWS_PER_XOR_PATTERN * Base::BYTES_PER_ROW);
            a[mi].reg(2) = tmp_1.x;
            a[mi].reg(3) = tmp_1.y;

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
    // Do we enable a faster LDS path to use LDS.128.
    int ENABLE_LDS_FAST_PATH = Traits::template Xmma_tile<Cta_tile>::ENABLE_LDS_FAST_PATH
>
struct Smem_tile_volta_row_a : public Smem_tile_without_skews<Cta_tile, 
                                                              Cta_tile::M, 
                                                              Cta_tile::K, 
                                                              16, 
                                                              BYTES_PER_STS,
                                                              BUFFERS_PER_TILE,
                                                              ENABLE_LDS_FAST_PATH,
                                                              8, 
                                                              1> {

    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;
    // The base class.
    using Base = Smem_tile_without_skews<Cta_tile, 
                                         Cta_tile::M, 
                                         Cta_tile::K, 
                                         16, 
                                         BYTES_PER_STS,
                                         BUFFERS_PER_TILE,
                                         ENABLE_LDS_FAST_PATH, 
                                         8, 
                                         1>;
    // The fragment.
    using Fragment = Fragment_a<Traits, Row>;

    // When we use padding to reach a power of two, special care has to be taken.
    using Cta_tile_with_padding = Cta_tile_with_k_with_padding<Traits, Cta_tile>; 
    // The number of XMMAs.
    using Xmma_tile_with_padding = typename Traits::template Xmma_tile<Cta_tile_with_padding>;

    // The size of a single LDS in bytes.
    enum { BYTES_PER_LDS = 16 };

    // Ctor.
    inline __device__ Smem_tile_volta_row_a(void *smem, int tidx) : Base(smem, tidx) {

        // For documentation on the layout, see doc/xmma_smem_layout.xlsx.

        // The number of warps.
        const int WARPS_M = Cta_tile::WARPS_M;
        const int WARPS_N = Cta_tile::WARPS_N;
        const int WARPS_K = Cta_tile::WARPS_K;

        // The masks to select the warps.
        const int WARP_MASK_M = Warp_masks<WARPS_M, WARPS_N, WARPS_K>::M;
        const int WARP_MASK_N = Warp_masks<WARPS_M, WARPS_N, WARPS_K>::N;
        const int WARP_MASK_K = Warp_masks<WARPS_M, WARPS_N, WARPS_K>::K;

        // The divisor for the warps.
        const int WARP_DIV_M =       1 *       1 * Cta_tile::THREADS_PER_WARP;
        const int WARP_DIV_N = WARPS_M *       1 * Cta_tile::THREADS_PER_WARP;
        const int WARP_DIV_K = WARPS_M * WARPS_N * Cta_tile::THREADS_PER_WARP;

        // The row and column read by the thread.
        int smem_read_row, smem_read_col;

        static_assert(Base::N_WITH_PADDING >= 64 || 
                      Base::N_WITH_PADDING == 32 || 
                      Base::N_WITH_PADDING == 16, "");

        if( Base::N_WITH_PADDING >= 64 ) {
            const int HALF_WARP_DIV = Xmma_tile::ENABLE_LDS_FAST_PATH ? 1 : 2;
            smem_read_row  = (tidx & WARP_MASK_M) / WARP_DIV_M * Xmma_tile::M_PER_XMMA / 1 + 
                             (tidx & 0x10) / HALF_WARP_DIV + (tidx & 0x07);
            // For group fprop/dgrd. A is divided into 2 halves along K dimension.
            // The fist warp takes the first half and the second warp takes the second half.
            if ( Cta_tile::GROUPS > 1 && WARPS_K == 1 && WARPS_N > 1 ) {
                smem_read_col = (tidx & 0x07) ^ 
                                (tidx & WARP_MASK_N) / WARP_DIV_N * (Cta_tile::K / WARPS_N) / 
                                (BYTES_PER_LDS * 8 / Base::BITS_PER_ELEMENT);
            } else {
                smem_read_col  = (tidx & 0x07);
            }
        } else if( Base::N_WITH_PADDING == 32 ) {
            const int HALF_WARP_DIV = Xmma_tile::ENABLE_LDS_FAST_PATH ? 2 : 4;
            smem_read_row  = (tidx & WARP_MASK_M) / WARP_DIV_M * Xmma_tile::M_PER_XMMA / 2 + 
                             (tidx & 0x10) / HALF_WARP_DIV + (tidx & 0x06) / 2;
            smem_read_col  = (tidx & 0x10) / HALF_WARP_DIV + (tidx & 0x06) / 2; 
            smem_read_col ^= (tidx & 0x01) * 4;
        } else if( Base::N_WITH_PADDING == 16 ) {
            const int HALF_WARP_DIV = Xmma_tile::ENABLE_LDS_FAST_PATH ? 4 : 8;
            smem_read_row  = (tidx & WARP_MASK_M) / WARP_DIV_M * Xmma_tile::M_PER_XMMA / 4 + 
                             (tidx & 0x10) / HALF_WARP_DIV + (tidx & 0x04) / 4;
            smem_read_col  = (tidx & 0x10) / HALF_WARP_DIV + (tidx & 0x04) / 4;;
            smem_read_col ^= (tidx & 0x03) * 2;
        }

        // For WARPS_K > 1, we do not support Base::N_WITH_PADDING < 64 for the moment.
        static_assert(WARPS_K <= 2 && (WARPS_K == 1 || Base::N_WITH_PADDING >= 64), "");

        // We "swap" the block for the second warp working on the in-CTA split-K.
        if( WARPS_K == 2 ) {
            smem_read_col ^= (tidx & WARP_MASK_K) / WARP_DIV_K * Xmma_tile_with_padding::XMMAS_K;
        }

        // The shared memory offset.
        this->smem_read_offset_ = smem_read_row*Base::BYTES_PER_ROW + smem_read_col*BYTES_PER_LDS;
    }

    // Rewind smem_read_offset for last LDS phase in main loop.-
    inline __device__ void reverse_smem_read_offset(int ki = 0) {

        // Move the offset to the next possition. See doc/xmma_smem_layout.xlsx.
        this->smem_read_offset_ ^= ((ki % 2 == 0) ? 1 : 3) * BYTES_PER_LDS;
    }

    // Load from shared memory.
    inline __device__ void load(Fragment (&a)[Xmma_tile::XMMAS_M], int ki) {
        #pragma unroll
        for( int mi = 0; mi < Xmma_tile::XMMAS_M; ++mi ) {
            int offset = mi * Xmma_tile::M_PER_XMMA_PER_CTA * Base::BYTES_PER_ROW_BEFORE_PACKING;

            // TODO: Could we fuse smem_read_buffer and smem_read_offset?
            uint4 tmp;
            lds(tmp, this->smem_ + this->smem_read_offset_ + this->smem_read_buffer_ + offset);
            a[mi].reg(0) = tmp.x;
            a[mi].reg(1) = tmp.y;
            a[mi].reg(2) = tmp.z;
            a[mi].reg(3) = tmp.w;
        }

        // Move the offset to the next possition. See doc/xmma_smem_layout.xlsx.
        static_assert(Xmma_tile_with_padding::XMMAS_K < 64, "Not implemented"); 
        if(        Xmma_tile_with_padding::XMMAS_K >= 32 && ki % 16 == 15 ) {
            this->smem_read_offset_ ^= 31 * BYTES_PER_LDS;
        } else if( Xmma_tile_with_padding::XMMAS_K >= 16 && ki %  8 ==  7 ) {
            this->smem_read_offset_ ^= 15 * BYTES_PER_LDS;
        } else if( Xmma_tile_with_padding::XMMAS_K >=  8 && ki %  4 ==  3 ) {
            this->smem_read_offset_ ^=  7 * BYTES_PER_LDS;
        } else if( Xmma_tile_with_padding::XMMAS_K >=  4 && ki %  2 ==  1 ) {
            this->smem_read_offset_ ^=  3 * BYTES_PER_LDS;
        } else if( Xmma_tile_with_padding::XMMAS_K >=  2 ) {
            this->smem_read_offset_ ^=  1 * BYTES_PER_LDS;
        }
    }

    // Reset the read offset.
    inline __device__ void reset_read_offset() {
        // The number of XMMAs in the K dimension.
        enum { XMMAS_K = Xmma_tile::XMMAS_K };
        // The number of XMMAs in the K dimension when we include padding.
        enum { XMMAS_K_WITH_PADDING = Xmma_tile_with_padding::XMMAS_K };
        // Assemble the mask.
        enum { MASK = Compute_reset_mask<XMMAS_K, XMMAS_K_WITH_PADDING>::VALUE };

        // Reset the read offset.
        this->smem_read_offset_ ^= MASK * BYTES_PER_LDS;
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
    // Do we enable a faster LDS path to use LDS.128.
    int ENABLE_LDS_FAST_PATH = Traits::template Xmma_tile<Cta_tile>::ENABLE_LDS_FAST_PATH
>
struct Smem_tile_volta_col_b : public Smem_tile_without_skews<Cta_tile, 
                                                              Cta_tile::N, 
                                                              Cta_tile::K, 
                                                              16, 
                                                              BYTES_PER_STS,
                                                              BUFFERS_PER_TILE, 
                                                              ENABLE_LDS_FAST_PATH, 
                                                              8, 
                                                              1> {

    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;
    // The base class.
    using Base = Smem_tile_without_skews<Cta_tile, 
                                         Cta_tile::N, 
                                         Cta_tile::K, 
                                         16, 
                                         BYTES_PER_STS,
                                         BUFFERS_PER_TILE, 
                                         ENABLE_LDS_FAST_PATH, 
                                         8, 
                                         1>;
    // The fragment.
    using Fragment = Fragment_b<Traits, Col>;

    // When we use padding to reach a power of two, special care has to be taken.
    using Cta_tile_with_padding = Cta_tile_with_k_with_padding<Traits, Cta_tile>; 
    // The number of XMMAs.
    using Xmma_tile_with_padding = typename Traits::template Xmma_tile<Cta_tile_with_padding>;

    // The size of a single LDS in bytes.
    enum { BYTES_PER_LDS = 16 };

    // Ctor.
    inline __device__ Smem_tile_volta_col_b(void *smem, int tidx) : Base(smem, tidx) {

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

        static_assert(Base::N_WITH_PADDING >= 64 || 
                      Base::N_WITH_PADDING == 32 || 
                      Base::N_WITH_PADDING == 16, "");

        if( Base::N_WITH_PADDING >= 64 ) {
            const int HALF_WARP_DIV = Xmma_tile::ENABLE_LDS_FAST_PATH ? 1 : 2;
            // For group fprop. B is divided into 2 halves along N dimension.
            // The fist warp takes the first half and the second warp takes the second half.
            if ( Cta_tile::GROUPS > 1 && WARPS_K == 1 && WARPS_N > 1 ) {
                smem_read_row = (tidx & WARP_MASK_N) / WARP_DIV_N * (Cta_tile::N / WARPS_N) / 1 + 
                                (tidx & 0x10) / HALF_WARP_DIV + (tidx & 0x08) / 2 + (tidx & 0x03);
            } else {
                smem_read_row = (tidx & WARP_MASK_N) / WARP_DIV_N * Xmma_tile::N_PER_XMMA / 1 + 
                                (tidx & 0x10) / HALF_WARP_DIV + (tidx & 0x08) / 2 + (tidx & 0x03);
            }
            smem_read_col = (tidx & 0x08) / 2 + (tidx & 0x03);
        } else if( Base::N_WITH_PADDING == 32 ) {
            const int HALF_WARP_DIV = Xmma_tile::ENABLE_LDS_FAST_PATH ? 2 : 4;
            smem_read_row  = (tidx & WARP_MASK_N) / WARP_DIV_N * Xmma_tile::N_PER_XMMA / 2 + 
                             (tidx & 0x10) / HALF_WARP_DIV + (tidx & 0x08) / 4 + (tidx & 0x02) / 2;
            smem_read_col  = (tidx & 0x10) / HALF_WARP_DIV + (tidx & 0x08) / 4 + (tidx & 0x02) / 2;
            smem_read_col ^= (tidx & 0x01) * 4;
        } else if( Base::N_WITH_PADDING == 16 ) {
            const int HALF_WARP_DIV = Xmma_tile::ENABLE_LDS_FAST_PATH ? 4 : 8;
            smem_read_row  = (tidx & WARP_MASK_N) / WARP_DIV_N * Xmma_tile::N_PER_XMMA / 4 + 
                             (tidx & 0x10) / HALF_WARP_DIV + (tidx & 0x08) / 8;
            smem_read_col  = (tidx & 0x10) / HALF_WARP_DIV + (tidx & 0x08) / 8;
            smem_read_col ^= (tidx & 0x03) * 2;
        }

        // For WARPS_K > 1, we do not support Base::N_WITH_PADDING < 64 for the moment.
        static_assert(WARPS_K <= 2 && (WARPS_K == 1 || Base::N_WITH_PADDING >= 64), "");

        // We "swap" the block for the second warp working on the in-CTA split-K.
        if( WARPS_K == 2 ) {
            smem_read_col ^= (tidx & WARP_MASK_K) / WARP_DIV_K * Xmma_tile_with_padding::XMMAS_K;
        }

        // The shared memory offset.
        this->smem_read_offset_ = smem_read_row*Base::BYTES_PER_ROW + smem_read_col*BYTES_PER_LDS;
    }

    // Rewind smem_read_offset for last LDS phase in main loop.-
    inline __device__ void reverse_smem_read_offset(int ki = 0) {

        // Move the offset to the next possition. See doc/xmma_smem_layout.xlsx.
        this->smem_read_offset_ ^= ((ki % 2 == 0) ? 1 : 3) * BYTES_PER_LDS;
    }

    // Load from shared memory.
    inline __device__ void load(Fragment (&b)[Xmma_tile::XMMAS_N], int ki) {
        #pragma unroll
        for( int ni = 0; ni < Xmma_tile::XMMAS_N; ++ ni ) {
            // For group fprop. B is divided into 2 halves along N dimension.
            // The fist warp takes the first half and the second warp takes the second half.
            int offset = ni * (Cta_tile::GROUPS > 1 ? Xmma_tile::N_PER_XMMA : 
                         Xmma_tile::N_PER_XMMA_PER_CTA) * Base::BYTES_PER_ROW_BEFORE_PACKING;

            // TODO: Can we fuse read_offset and read_buffer?
            uint4 tmp;
            lds(tmp, this->smem_ + this->smem_read_offset_ + this->smem_read_buffer_ + offset);
            b[ni].reg(0) = tmp.x;
            b[ni].reg(1) = tmp.y;
            b[ni].reg(2) = tmp.z;
            b[ni].reg(3) = tmp.w;
        }

        // Move the offset to the next possition. See doc/xmma_smem_layout.xlsx.
        static_assert(Xmma_tile_with_padding::XMMAS_K < 64, "Not implemented"); 
        if(        Xmma_tile_with_padding::XMMAS_K >= 32 && ki % 16 == 15 ) {
            this->smem_read_offset_ ^= 31 * BYTES_PER_LDS;
        } else if( Xmma_tile_with_padding::XMMAS_K >= 16 && ki %  8 ==  7 ) {
            this->smem_read_offset_ ^= 15 * BYTES_PER_LDS;
        } else if( Xmma_tile_with_padding::XMMAS_K >=  8 && ki %  4 ==  3 ) {
            this->smem_read_offset_ ^=  7 * BYTES_PER_LDS;
        } else if( Xmma_tile_with_padding::XMMAS_K >=  4 && ki %  2 ==  1 ) {
            this->smem_read_offset_ ^=  3 * BYTES_PER_LDS;
        } else if( Xmma_tile_with_padding::XMMAS_K >=  2 ) {
            this->smem_read_offset_ ^=  1 * BYTES_PER_LDS;
        } 
    }

    // Reset the read offset.
    inline __device__ void reset_read_offset() {
        // The number of XMMAs in the K dimension.
        enum { XMMAS_K = Xmma_tile::XMMAS_K };
        // The number of XMMAs in the K dimension when we include padding.
        enum { XMMAS_K_WITH_PADDING = Xmma_tile_with_padding::XMMAS_K };
        // Assemble the mask.
        enum { MASK = Compute_reset_mask<XMMAS_K, XMMAS_K_WITH_PADDING>::VALUE };

        // Reset the read offset.
        this->smem_read_offset_ ^= MASK * BYTES_PER_LDS;
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
    int ROWS_PER_XOR_PATTERN_ = Rows_per_xor_pattern_volta<Cta_tile::N>::VALUE,
    // How many rows to use for the XOR pattern to avoid bank conflicts?
    int COLS_PER_XOR_PATTERN_ = Cols_per_xor_pattern_volta<Traits, Cta_tile>::VALUE
>
struct Smem_tile_volta_row_b : public Smem_tile_without_skews<Cta_tile, 
                                                              Cta_tile::K, 
                                                              Cta_tile::N, 
                                                              16, 
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
                                         16,
                                         BYTES_PER_STS,
                                         BUFFERS_PER_TILE,
                                         0,
                                         ROWS_PER_XOR_PATTERN_,
                                         COLS_PER_XOR_PATTERN_>;
    // The fragment.
    using Fragment = Fragment_b<Traits, Row>;

    // The size of a single LDS in bytes.
    enum { BYTES_PER_LDS = Xmma_tile::ENABLE_LDS_FAST_PATH ? 16 : 8 };
    // The number of elements per LDS.
    enum { ELEMENTS_PER_LDS = BYTES_PER_LDS * 8 / Traits::BITS_PER_ELEMENT_B };

    // The size in bytes of the data needed to compute an XMMA per CTA.
    enum { BYTES_PER_XMMA_PER_CTA = Xmma_tile::N_PER_XMMA_PER_CTA * 
                                    (Traits::BITS_PER_ELEMENT_B / 8) };

    // Ctor.
    inline __device__ Smem_tile_volta_row_b(void *smem, int tidx) : Base(smem, tidx) {

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

        static_assert(Base::ROWS_PER_XOR_PATTERN == 4 || 
                      Base::ROWS_PER_XOR_PATTERN == 2 || 
                      Base::ROWS_PER_XOR_PATTERN == 1, "");

        if( Base::ROWS_PER_XOR_PATTERN == 4 ) {
            // For group dgrd. B is divided into 2 halves along K dimension.
            // The fist warp takes the first half and the second warp takes the second half.
            if ( Cta_tile::GROUPS > 1 && WARPS_K == 1 && WARPS_N > 1 ) {
                smem_read_row = (tidx & WARP_MASK_N) / WARP_DIV_N * (Cta_tile::N / WARPS_N) +
                                (tidx & 0x03);
            } else {
                smem_read_row = (tidx & WARP_MASK_K) / WARP_DIV_K * Xmma_tile::XMMAS_K * 8 + 
                                (tidx & 0x03);
            }
            smem_read_col = (tidx & 0x03) * 2 + (tidx & 0x08) / 8;
        } else if( Base::ROWS_PER_XOR_PATTERN == 2 ) {
            smem_read_row = (tidx & WARP_MASK_K) / WARP_DIV_K * Xmma_tile::XMMAS_K * 4 + 
                (tidx & 0x03) / 2;
            smem_read_col = (tidx & 0x02) * 1 + (tidx & 0x08) / 8 + (tidx & 0x01) * 8;
        } else if( Base::ROWS_PER_XOR_PATTERN == 1 ) {
            smem_read_row = (tidx & WARP_MASK_K) / WARP_DIV_K * Xmma_tile::XMMAS_K * 2;
            smem_read_col = (tidx & 0x03) * 4 + (tidx & 0x08) / 8;
        }

        // Each half-warp applies a different XOR pattern -- see the Excel document.
        if ( Cta_tile::GROUPS > 1 && WARPS_K == 1 && WARPS_N > 1 ) {
            smem_read_col ^= (tidx & 0x10) / 8;
        }
        else {
            smem_read_col ^= (tidx & WARP_MASK_N) / WARP_DIV_N * 4 +
                             (tidx & 0x10) / 8;
        }

        // The shared memory offset.
        this->smem_read_offset_ = 
            smem_read_row*Base::BYTES_PER_ROW + smem_read_col*BYTES_PER_LDS;
    }

    // Rewind smem_read_offset for last LDS phase in main loop.-
    inline __device__ void reverse_smem_read_offset(int ki = 0) {
    }

    // Load from shared memory.
    inline __device__ void load(Fragment (&b)[Xmma_tile::XMMAS_N], int ki) {
        #pragma unroll
        for( int ni = 0; ni < Xmma_tile::XMMAS_N; ++ ni ) {
            // Prepare the offset.
            int offset = ki * Base::ROWS_PER_XOR_PATTERN * Base::BYTES_PER_ROW * 2;
            if( Cta_tile::WARPS_N == 1 || (Cta_tile::N == 64 && Cta_tile::GROUPS > 1) ) {
                offset += this->smem_read_offset_ ^ (ni % 2 == 0 ? 0 : (BYTES_PER_LDS * 4)); 
                offset += (ni >>1 ) * (BYTES_PER_LDS * 4) * 2;
            } else {
                offset += this->smem_read_offset_ + (ni  ) * Xmma_tile::N_PER_XMMA_PER_CTA * 2;
            }

            // The load pointer.
            uint32_t ptr = this->smem_ + this->smem_read_buffer_ + offset;

            // Load the data using 2x LDS.64.
            uint2 tmp_0; 
            lds(tmp_0, ptr);
            b[ni].reg(0) = tmp_0.x;
            b[ni].reg(1) = tmp_0.y;

            uint2 tmp_1; 
            lds(tmp_1, ptr + Base::ROWS_PER_XOR_PATTERN * Base::BYTES_PER_ROW);
            b[ni].reg(2) = tmp_1.x;
            b[ni].reg(3) = tmp_1.y;
        }
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Traits, typename Cta_tile, int BUFFERS_PER_TILE >
struct Smem_tile_volta_hmma_col_interleaved_a 
    : public Smem_tile_interleaved<Cta_tile, Cta_tile::K, Cta_tile::M, 16, BUFFERS_PER_TILE> {

    // The base class.
    using Base = Smem_tile_interleaved<Cta_tile, Cta_tile::K, Cta_tile::M, 16, BUFFERS_PER_TILE>;
    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;
    // The fragment.
    using Fragment = Fragment_a<Traits, Row>;

    // The size of a single LDS in bytes.
    enum { BYTES_PER_LDS = 16 };

    // Ctor.
    inline __device__ Smem_tile_volta_hmma_col_interleaved_a(void *smem, int tidx) 
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
        int smem_read_row = (tidx & WARP_MASK_K) / WARP_DIV_K * Base::ROWS / WARPS_K;
        int smem_read_col = (tidx & WARP_MASK_M) / WARP_DIV_M * Xmma_tile::M_PER_XMMA +
                            (tidx & 0x10) / 2 + 
                            (tidx & 0x07);

        // The shared memory offset.
        this->smem_read_offset_ = smem_read_row*Base::BYTES_PER_ROW + smem_read_col*BYTES_PER_LDS;
    }

    // Rewind smem_read_offset for last LDS phase in main loop.-
    inline __device__ void reverse_smem_read_offset(int ki = 0) {
    }

    // Load from shared memory.
    inline __device__ void load(Fragment (&a)[Xmma_tile::XMMAS_M], int ki) {
        #pragma unroll
        for( int mi = 0; mi < Xmma_tile::XMMAS_M; ++mi ) {
            int offset = mi*Xmma_tile::M_PER_XMMA_PER_CTA*BYTES_PER_LDS + ki*Base::BYTES_PER_ROW;
            uint4 tmp; 
            lds(tmp, this->smem_ + this->smem_read_offset_ + this->smem_read_buffer_ + offset);
            a[mi].reg(0) = tmp.x;
            a[mi].reg(1) = tmp.y;
            a[mi].reg(2) = tmp.z;
            a[mi].reg(3) = tmp.w;
        }
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Cta_tile, int BUFFERS_PER_TILE >
struct Smem_tile_volta_hmma_col_interleaved_a <Volta_hmma_fp32_interleaved_traits,Cta_tile,BUFFERS_PER_TILE>
    : public Smem_tile_interleaved<Cta_tile, Cta_tile::K, Cta_tile::M, 16, 16, BUFFERS_PER_TILE> {

    using Traits = Volta_hmma_fp32_interleaved_traits;
    // The base class.
    using Base = Smem_tile_interleaved<Cta_tile, Cta_tile::K, Cta_tile::M, 16, 16, BUFFERS_PER_TILE>;
    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;
    // The fragment.
    using Fragment = Fragment_a<Traits, Row>;
    

    // The size of a single LDS in bytes.
    enum { BYTES_PER_LDS = 16 };

    // Ctor.
    inline __device__ Smem_tile_volta_hmma_col_interleaved_a(void *smem, int tidx) 
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
        int smem_read_row = (tidx & WARP_MASK_K) / WARP_DIV_K * Base::ROWS / WARPS_K;
        int smem_read_col = (tidx & WARP_MASK_M) / WARP_DIV_M * Xmma_tile::M_PER_XMMA +
                            (tidx & 0x10) / 2 + 
                            (tidx & 0x07);

        // The shared memory offset.
        this->smem_read_offset_ = smem_read_row*Base::BYTES_PER_ROW + smem_read_col*BYTES_PER_LDS;
    }

    // Rewind smem_read_offset for last LDS phase in main loop.-
    inline __device__ void reverse_smem_read_offset(int ki = 0) {
    }

    // Load from shared memory.
    inline __device__ void load(Fragment (&a)[Xmma_tile::XMMAS_M], int ki) {
        #pragma unroll
        for( int mi = 0; mi < Xmma_tile::XMMAS_M; ++mi ) {
            int offset = mi*Xmma_tile::M_PER_XMMA_PER_CTA*BYTES_PER_LDS + ki*Base::BYTES_PER_ROW;
            uint4 tmp; 
            lds(tmp, this->smem_ + this->smem_read_offset_ + this->smem_read_buffer_ + offset);
            a[mi].reg(0) = tmp.x;
            a[mi].reg(1) = tmp.y;
            a[mi].reg(2) = tmp.z;
            a[mi].reg(3) = tmp.w;
        }
    }

};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Traits, typename Cta_tile, int BUFFERS_PER_TILE >
struct Smem_tile_volta_hmma_row_interleaved_b 
    : public Smem_tile_interleaved<Cta_tile, Cta_tile::K, Cta_tile::N, 16, BUFFERS_PER_TILE> {

    // The base class.
    using Base = Smem_tile_interleaved<Cta_tile, Cta_tile::K, Cta_tile::N, 16, BUFFERS_PER_TILE>;
    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;
    // The fragment.
    using Fragment = Fragment_b<Traits, Col>;

    // The size of a single LDS in bytes.
    enum { BYTES_PER_LDS = 16 };

    // Ctor.
    inline __device__ Smem_tile_volta_hmma_row_interleaved_b(void *smem, int tidx) 
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
        int smem_read_row = (tidx & WARP_MASK_K) / WARP_DIV_K * Base::ROWS / WARPS_K;
        int smem_read_col = (tidx & WARP_MASK_N) / WARP_DIV_N * Xmma_tile::N_PER_XMMA +
                            (tidx & 0x18) / 2 + 
                            (tidx & 0x03);

        // The shared memory offset.
        this->smem_read_offset_ = smem_read_row*Base::BYTES_PER_ROW + smem_read_col*BYTES_PER_LDS;
    }

    // Rewind smem_read_offset for last LDS phase in main loop.-
    inline __device__ void reverse_smem_read_offset(int ki = 0) {
    }

    // Load from shared memory.
    inline __device__ void load(Fragment (&b)[Xmma_tile::XMMAS_N], int ki) {
        #pragma unroll
        for( int ni = 0; ni < Xmma_tile::XMMAS_N; ++ni ) {
            int offset = ni*Xmma_tile::N_PER_XMMA_PER_CTA*BYTES_PER_LDS + ki*Base::BYTES_PER_ROW;
            uint4 tmp; 
            lds(tmp, this->smem_ + this->smem_read_offset_ + this->smem_read_buffer_ + offset);
            b[ni].reg(0) = tmp.x;
            b[ni].reg(1) = tmp.y;
            b[ni].reg(2) = tmp.z;
            b[ni].reg(3) = tmp.w;
        }
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Cta_tile, int BUFFERS_PER_TILE >
struct Smem_tile_volta_hmma_row_interleaved_b <Volta_hmma_fp32_interleaved_traits,Cta_tile,BUFFERS_PER_TILE>
    : public Smem_tile_interleaved<Cta_tile, Cta_tile::K, Cta_tile::N, 16, 16, BUFFERS_PER_TILE> {

    using Traits = Volta_hmma_fp32_interleaved_traits;
    // The base class.
    using Base = Smem_tile_interleaved<Cta_tile, Cta_tile::K, Cta_tile::N, 16, 16, BUFFERS_PER_TILE>;
    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;
    // The fragment.
    using Fragment = Fragment_b<Traits, Col>;

    // The size of a single LDS in bytes.
    enum { BYTES_PER_LDS = 16 };


    // Ctor.
    inline __device__ Smem_tile_volta_hmma_row_interleaved_b(void *smem, int tidx) 
        : Base(smem, tidx){
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
        int smem_read_row = (tidx & WARP_MASK_K) / WARP_DIV_K * Base::ROWS / WARPS_K;
        int smem_read_col = (tidx & WARP_MASK_N) / WARP_DIV_N * Xmma_tile::N_PER_XMMA +
                            (tidx & 0x18) / 2 + 
                            (tidx & 0x03);

        // The shared memory offset.
        this->smem_read_offset_ = smem_read_row*Base::BYTES_PER_ROW + smem_read_col*BYTES_PER_LDS;
    }

    // Rewind smem_read_offset for last LDS phase in main loop.-
    inline __device__ void reverse_smem_read_offset(int ki = 0) {
    }

    // Load from shared memory.
    inline __device__ void load(Fragment (&b)[Xmma_tile::XMMAS_N], int ki) {
        #pragma unroll
        for( int ni = 0; ni < Xmma_tile::XMMAS_N; ++ni ) {
            int offset = ni*Xmma_tile::N_PER_XMMA_PER_CTA*BYTES_PER_LDS + ki*Base::BYTES_PER_ROW;
            uint4 tmp; 
            lds(tmp, this->smem_ + this->smem_read_offset_ + this->smem_read_buffer_ + offset);
            b[ni].reg(0) = tmp.x;
            b[ni].reg(1) = tmp.y;
            b[ni].reg(2) = tmp.z;
            b[ni].reg(3) = tmp.w;
        }
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// H M M A . F 1 6
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
struct Smem_tile_a<Volta_hmma_fp16_traits, Cta_tile, Row, BYTES_PER_STS, BUFFERS_PER_TILE> 
    : public Smem_tile_volta_row_a<Volta_hmma_fp16_traits, 
                                   Cta_tile, 
                                   BYTES_PER_STS,
                                   BUFFERS_PER_TILE> {

    // The traits class.
    using Traits = Volta_hmma_fp16_traits;
    // The XMMA tile.
    using Xmma_tile = Traits::template Xmma_tile<Cta_tile>;
    // The base class.
    using Base = Smem_tile_volta_row_a<Traits, Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE>;

    // Ctor.
    inline __device__ Smem_tile_a(void *smem, int tidx) : Base(smem, tidx) {
    }

    // Rewind smem_read_offset for last LDS phase in main loop.-
    inline __device__ void reverse_smem_read_offset(int ki = 0) {

        // Move the offset to the next possition. See doc/xmma_smem_layout.xlsx.
        if( Base::N_WITH_PADDING / Cta_tile::WARPS_K == 64 && ki % 4 == 3 ) {
            this->smem_read_offset_ ^= 7 * Base::BYTES_PER_LDS;
        } else if( ki % 2 == 1 ) {
            this->smem_read_offset_ ^= 3 * Base::BYTES_PER_LDS;
        } else {
            this->smem_read_offset_ ^= 1 * Base::BYTES_PER_LDS;
        }
    }

    // Load from shared memory.
    inline __device__ void load(typename Base::Fragment (&a)[Xmma_tile::XMMAS_M], int ki) {
        #pragma unroll
        for( int mi = 0; mi < Xmma_tile::XMMAS_M; ++mi ) {
            int offset = mi * Xmma_tile::M_PER_XMMA_PER_CTA * Base::BYTES_PER_ROW_BEFORE_PACKING;
            uint4 tmp; 
            lds(tmp, this->smem_ + this->smem_read_offset_ + this->smem_read_buffer_ + offset);
            a[mi].reg(0) = tmp.x;
            a[mi].reg(1) = tmp.y;
            a[mi].reg(2) = tmp.z;
            a[mi].reg(3) = tmp.w;
        }

        // Move the offset to the next possition. See doc/xmma_smem_layout.xlsx.
        if ( Cta_tile::GROUPS == 1 ) {
            if ( Base::N_WITH_PADDING / Cta_tile::WARPS_K == 64 && ki % 4 == 3 ) {
                this->smem_read_offset_ ^= 7 * Base::BYTES_PER_LDS;
            } else if ( ki % 2 == 1 ) {
                this->smem_read_offset_ ^= 3 * Base::BYTES_PER_LDS;
            } else {
                this->smem_read_offset_ ^= 1 * Base::BYTES_PER_LDS;
            }
        } else {
            if ( ki % 2 == 1 && Cta_tile::WARPS_N == 2 ) {
                this->smem_read_offset_ ^= 3 * Base::BYTES_PER_LDS;
            } else {
                this->smem_read_offset_ ^= 1 * Base::BYTES_PER_LDS;
            }
        }
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
struct Smem_tile_a<Volta_hmma_fp16_traits, Cta_tile, Col, BYTES_PER_STS, BUFFERS_PER_TILE> 
    : public Smem_tile_volta_col_a<Volta_hmma_fp16_traits, 
                                   Cta_tile, 
                                   BYTES_PER_STS,
                                   BUFFERS_PER_TILE> {

    // The traits class.
    using Traits = Volta_hmma_fp16_traits;
    // The base class.
    using Base = Smem_tile_volta_col_a<Traits, Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE>;

    // Ctor.
    inline __device__ Smem_tile_a(void *smem, int tidx) : Base(smem, tidx) {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< 
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile, 
    // The number of buffers per tile.
    int BUFFERS_PER_TILE
>
struct Smem_tile_a<Volta_hmma_fp16_traits, Cta_tile, Col_interleaved, 16, BUFFERS_PER_TILE> 
    : public Smem_tile_volta_hmma_col_interleaved_a<Volta_hmma_fp16_traits, 
                                                    Cta_tile, 
                                                    BUFFERS_PER_TILE> {

    // The traits class.
    using Traits = Volta_hmma_fp16_traits;
    // The base class.
    using Base = Smem_tile_volta_hmma_col_interleaved_a<Traits, Cta_tile, BUFFERS_PER_TILE>;

    // Ctor.
    inline __device__ Smem_tile_a(void *smem, int tidx) : Base(smem, tidx) {
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
struct Smem_tile_b<Volta_hmma_fp16_traits, Cta_tile, Row, BYTES_PER_STS, BUFFERS_PER_TILE>
    : public Smem_tile_volta_row_b<Volta_hmma_fp16_traits, 
                                   Cta_tile, 
                                   BYTES_PER_STS,
                                   BUFFERS_PER_TILE> {

    // The traits class.
    using Traits = Volta_hmma_fp16_traits;
    // The base class.
    using Base = Smem_tile_volta_row_b<Traits, Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE>;

    // Ctor.
    inline __device__ Smem_tile_b(void *smem, int tidx) : Base(smem, tidx) {
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
struct Smem_tile_b<Volta_hmma_fp16_traits, Cta_tile, Col, BYTES_PER_STS, BUFFERS_PER_TILE>
    : public Smem_tile_volta_col_b<Volta_hmma_fp16_traits, 
                                   Cta_tile, 
                                   BYTES_PER_STS,
                                   BUFFERS_PER_TILE> {

    // The traits class.
    using Traits = Volta_hmma_fp16_traits;
    // The XMMA tile.
    using Xmma_tile = Traits::template Xmma_tile<Cta_tile>;
    // The base class.
    using Base = Smem_tile_volta_col_b<Traits, Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE>;

    // Ctor.
    inline __device__ Smem_tile_b(void *smem, int tidx) : Base(smem, tidx) {
    }

    // Rewind smem_read_offset for last LDS phase in main loop.-
    inline __device__ void reverse_smem_read_offset(int ki = 0) {

        // Move the offset to the next possition. See doc/xmma_smem_layout.xlsx.
        if( Base::N_WITH_PADDING / Cta_tile::WARPS_K == 64 && ki % 4 == 3 ) {
            this->smem_read_offset_ ^= 7 * Base::BYTES_PER_LDS;
        } else if( ki % 2 == 1 ) {
            this->smem_read_offset_ ^= 3 * Base::BYTES_PER_LDS;
        } else {
            this->smem_read_offset_ ^= 1 * Base::BYTES_PER_LDS;
        }
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< 
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile, 
    // The number of buffers per tile.
    int BUFFERS_PER_TILE
>
struct Smem_tile_b<Volta_hmma_fp16_traits, Cta_tile, Row_interleaved, 16, BUFFERS_PER_TILE>
    : public Smem_tile_volta_hmma_row_interleaved_b<Volta_hmma_fp16_traits, 
                                                    Cta_tile, 
                                                    BUFFERS_PER_TILE> {

    // The traits class.
    using Traits = Volta_hmma_fp16_traits;
    // The base class.
    using Base = Smem_tile_volta_hmma_row_interleaved_b<Traits, Cta_tile, BUFFERS_PER_TILE>;

    // Ctor.
    inline __device__ Smem_tile_b(void *smem, int tidx) : Base(smem, tidx) {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< 
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile, 
    // Do we use the special split-k trick inside the CTA?
    bool IN_CTA_SPLIT_K
>
struct Swizzle_epilogue<Volta_hmma_fp16_traits, 
                        Cta_tile, 
                        Row, 
                        16,
                        IN_CTA_SPLIT_K
                        > {

    // The traits.
    using Traits = Volta_hmma_fp16_traits;
    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;

    // To support arbitrary N, we pad some values to a power-of-2.
    enum { N_WITH_PADDING = Next_power_of_two<Cta_tile::N>::VALUE }; 
    // The number of bytes for key memory instruction.
    enum { BYTES_PER_LDS = 16, BYTES_PER_STG = 16, BYTES_PER_STS = 8 };
    // The number of rows in shared memory.
    enum { ROWS = Xmma_tile::M_PER_XMMA_PER_CTA };
    // The size of a single row in bytes.
    enum { BYTES_PER_ROW = N_WITH_PADDING * Cta_tile::WARPS_K * sizeof(Traits::C_type) };
    // The skew to avoid bank conflicts.
    enum { BYTES_PER_SKEW = BYTES_PER_STS * 2 };
    // The number of columns with the skew.
    enum { BYTES_PER_ROW_WITH_SKEW = BYTES_PER_ROW + BYTES_PER_SKEW };
    // The size in bytes in shared memory.
    enum { BYTES_PER_TILE = ROWS * BYTES_PER_ROW_WITH_SKEW };

    // The number of threads per "pixel".
    enum { THREADS_PER_PIXEL = N_WITH_PADDING * sizeof(Traits::C_type) / BYTES_PER_STG };
    // The number of "pixels" written in one STS.128.
    enum { PIXELS_PER_STG = Cta_tile::THREADS_PER_CTA / THREADS_PER_PIXEL };

    // How we see the distribution of data.
    enum { THREADS_PER_XMMA_M = 16, THREADS_PER_XMMA_N = 2 };
    // The number of elements stored per thread.
    enum { M_PER_XMMA_PER_THREAD = 1, N_PER_XMMA_PER_THREAD = 8 };

    // Do we skip the syncthreads in the epilogue? Of course, not :)
    enum { SKIP_SYNCTHREADS = 0 };

    // Ctor.
    inline __device__ Swizzle_epilogue(void *smem, int tidx) 
        : smem_(get_smem_pointer(smem)) {

        // Extract the number of warps.
        const int WARPS_M = Cta_tile::WARPS_M;
        const int WARPS_N = Cta_tile::WARPS_N;
        const int WARPS_K = Cta_tile::WARPS_K;

        //
        //     tidx   0: row =  0, col = 0
        //     tidx   1: row =  1, col = 0
        //     tidx   2: row =  2, col = 0
        //     tidx   3: row =  3, col = 0
        //     tidx   4: row =  4, col = 0
        //     tidx   5: row =  5, col = 0
        //     tidx   6: row =  6, col = 0
        //     tidx   7: row =  7, col = 0
        //     tidx   8: row =  0, col = 8
        //     tidx   9: row =  1, col = 8
        //     tidx  10: row =  2, col = 8
        //     tidx  11: row =  3, col = 8
        //     tidx  12: row =  4, col = 8
        //     tidx  13: row =  5, col = 8
        //     tidx  14: row =  6, col = 8
        //     tidx  15: row =  7, col = 8
        //     tidx  16: row =  8, col = 0
        //     tidx  17: row =  9, col = 0
        //     tidx  18: row = 10, col = 0
        //     tidx  19: row = 11, col = 0
        //     tidx  20: row = 12, col = 0
        //     tidx  21: row = 13, col = 0
        //     tidx  22: row = 14, col = 0
        //     tidx  23: row = 15, col = 0
        //     tidx  24: row =  8, col = 8
        //     tidx  25: row =  9, col = 8
        //     tidx  26: row = 10, col = 8
        //     tidx  27: row = 11, col = 8
        //     tidx  28: row = 12, col = 8
        //     tidx  29: row = 13, col = 8
        //     tidx  30: row = 14, col = 8
        //     tidx  31: row = 15, col = 8
        //

	// The masks to select the warps.
        const int WARP_MASK_M = Warp_masks<WARPS_M, WARPS_N, WARPS_K>::M;
        const int WARP_MASK_N = Warp_masks<WARPS_M, WARPS_N, WARPS_K>::N;
        const int WARP_MASK_K = Warp_masks<WARPS_M, WARPS_N, WARPS_K>::K;

        // The divisor for the warps.
        const int WARP_DIV_M =       1 *       1 * Cta_tile::THREADS_PER_WARP;
        const int WARP_DIV_N = WARPS_M *       1 * Cta_tile::THREADS_PER_WARP;
        const int WARP_DIV_K = WARPS_M * WARPS_N * Cta_tile::THREADS_PER_WARP;

        // Compute the row and the column in shared memory. Each warp reads from a 16*16B segment.
        const int smem_write_row = (tidx & WARP_MASK_M) / WARP_DIV_M * 16 +
                                   (tidx & 0x10) / 2 +
                                   (tidx & 0x07);

        // For group fprop/dgrd. C is divided into 2 halves along N dimension.
        // The fist warp stores the first half and the second warp stores the second half.
        const int smem_write_col = (tidx & WARP_MASK_K) / WARP_DIV_K * N_WITH_PADDING * 2 + 
                                   (tidx & WARP_MASK_N) / WARP_DIV_N * (Cta_tile::GROUPS > 1 ? 
                                   Cta_tile::N / WARPS_N : Xmma_tile::N_PER_XMMA) * 
                                   sizeof(Traits::C_type) + (tidx & 0x08);

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

    // Load from the tile in shared memory.
    template<typename Fragment_post_swizzle>
    inline __device__ void load(Fragment_post_swizzle &dst, int oi) const {
        #pragma unroll
        for( int ki = 0; ki < Cta_tile::WARPS_K; ++ki ) {
            const int offset = oi * PIXELS_PER_STG * BYTES_PER_ROW_WITH_SKEW +
                               ki * N_WITH_PADDING * sizeof(Traits::C_type);
            uint4 tmp;
            lds(tmp, smem_ + smem_read_offset_ + offset);
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
        for( int mi = 0; mi < M_PER_XMMA_PER_THREAD; ++mi ) {
            int offset = mi * Xmma_tile::M_PER_XMMA_PER_CTA * BYTES_PER_ROW_WITH_SKEW +
                         ni * (Cta_tile::GROUPS > 1 ? Xmma_tile::N_PER_XMMA : 
                         Xmma_tile::N_PER_XMMA_PER_CTA) * sizeof(Traits::C_type);

            uint32_t ptr = smem_ + smem_write_offset_ + offset;
            sts(ptr +  0, make_uint2(c.reg(4*mi+0), c.reg(4*mi+1)));
            sts(ptr + 16, make_uint2(c.reg(4*mi+2), c.reg(4*mi+3)));
        }
    }

    inline __device__ void debug_print() const {
        // Not implemented
    }

    // The shared memory pointer in bytes.
    uint32_t smem_;
    // The write offset.
    int smem_write_offset_;
    // The read offset.
    int smem_read_offset_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// H M M A . F 3 2
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
struct Smem_tile_a<Volta_hmma_fp32_traits, Cta_tile, Col, BYTES_PER_STS, BUFFERS_PER_TILE> 
    : public Smem_tile_volta_col_a<Volta_hmma_fp32_traits, 
                                   Cta_tile, 
                                   BYTES_PER_STS,
                                   BUFFERS_PER_TILE> {

    // The traits class.
    using Traits = Volta_hmma_fp32_traits;
    // The base class.
    using Base = Smem_tile_volta_col_a<Traits, Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE>;

    // Ctor.
    inline __device__ Smem_tile_a(void *smem, int tidx) : Base(smem, tidx) {
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
struct Smem_tile_a<Volta_hmma_fp32_traits, Cta_tile, Row, BYTES_PER_STS, BUFFERS_PER_TILE> 
    : public Smem_tile_volta_row_a<Volta_hmma_fp32_traits, 
                                   Cta_tile, 
                                   BYTES_PER_STS,
                                   BUFFERS_PER_TILE> {

    // The traits class.
    using Traits = Volta_hmma_fp32_traits;
    // The base class.
    using Base = Smem_tile_volta_row_a<Traits, Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE>;

    // Ctor.
    inline __device__ Smem_tile_a(void *smem, int tidx) : Base(smem, tidx) {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< 
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile, 
    // The number of buffers per tile.
    int BUFFERS_PER_TILE
>
struct Smem_tile_a<Volta_hmma_fp32_traits, Cta_tile, Col_interleaved, 16, BUFFERS_PER_TILE> 
    : public Smem_tile_volta_hmma_col_interleaved_a<Volta_hmma_fp32_traits, 
                                                    Cta_tile, 
                                                    BUFFERS_PER_TILE> {

    // The traits class.
    using Traits = Volta_hmma_fp32_traits;
    // The base class.
    using Base = Smem_tile_volta_hmma_col_interleaved_a<Traits, Cta_tile, BUFFERS_PER_TILE>;

    // Ctor.
    inline __device__ Smem_tile_a(void *smem, int tidx) : Base(smem, tidx) {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< 
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile, 
    // The number of buffers per tile.
    int BUFFERS_PER_TILE
>
struct Smem_tile_a<Volta_hmma_fp32_interleaved_traits, Cta_tile, Col_interleaved, 16, BUFFERS_PER_TILE> 
    : public Smem_tile_volta_hmma_col_interleaved_a<Volta_hmma_fp32_interleaved_traits, 
                                                    Cta_tile, 
                                                    BUFFERS_PER_TILE> {

    // The traits class.
    using Traits = Volta_hmma_fp32_interleaved_traits;
    // The base class.
    using Base = Smem_tile_volta_hmma_col_interleaved_a<Traits, Cta_tile, BUFFERS_PER_TILE>;

    // Ctor.
    inline __device__ Smem_tile_a(void *smem, int tidx) : Base(smem, tidx) {
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
struct Smem_tile_b<Volta_hmma_fp32_traits, Cta_tile, Row, BYTES_PER_STS, BUFFERS_PER_TILE> 
    : public Smem_tile_volta_row_b<Volta_hmma_fp32_traits, 
                                   Cta_tile, 
                                   BYTES_PER_STS,
                                   BUFFERS_PER_TILE> {

    // The traits class.
    using Traits = Volta_hmma_fp32_traits;
    // The base class.
    using Base = Smem_tile_volta_row_b<Traits, Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE>;

    // Ctor.
    inline __device__ Smem_tile_b(void *smem, int tidx) : Base(smem, tidx) {
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
struct Smem_tile_b<Volta_hmma_fp32_traits, Cta_tile, Col, BYTES_PER_STS, BUFFERS_PER_TILE> 
    : public Smem_tile_volta_col_b<Volta_hmma_fp32_traits, 
                                   Cta_tile, 
                                   BYTES_PER_STS,
                                   BUFFERS_PER_TILE> {

    // The traits class.
    using Traits = Volta_hmma_fp32_traits;
    // The base class.
    using Base = Smem_tile_volta_col_b<Traits, Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE>;

    // Ctor.
    inline __device__ Smem_tile_b(void *smem, int tidx) : Base(smem, tidx) {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< 
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile, 
    // The number of buffers per tile.
    int BUFFERS_PER_TILE
>
struct Smem_tile_b<Volta_hmma_fp32_traits, Cta_tile, Row_interleaved, 16, BUFFERS_PER_TILE> 
    : public Smem_tile_volta_hmma_row_interleaved_b<Volta_hmma_fp32_traits, 
                                                    Cta_tile, 
                                                    BUFFERS_PER_TILE> {

    // The traits class.
    using Traits = Volta_hmma_fp32_traits;
    // The base class.
    using Base = Smem_tile_volta_hmma_row_interleaved_b<Traits, Cta_tile, BUFFERS_PER_TILE>;

    // Ctor.
    inline __device__ Smem_tile_b(void *smem, int tidx) : Base(smem, tidx) {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< 
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile, 
    // The number of buffers per tile.
    int BUFFERS_PER_TILE
>
struct Smem_tile_b<Volta_hmma_fp32_interleaved_traits, Cta_tile, Row_interleaved, 16, BUFFERS_PER_TILE> 
    : public Smem_tile_volta_hmma_row_interleaved_b<Volta_hmma_fp32_interleaved_traits, 
                                                    Cta_tile, 
                                                    BUFFERS_PER_TILE> {

    // The traits class.
    using Traits = Volta_hmma_fp32_interleaved_traits;
    // The base class.
    using Base = Smem_tile_volta_hmma_row_interleaved_b<Traits, Cta_tile, BUFFERS_PER_TILE>;

    // Ctor.
    inline __device__ Smem_tile_b(void *smem, int tidx) : Base(smem, tidx) {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Traits, typename Cta_tile, typename Layout, bool IN_CTA_SPLIT_K >
struct Swizzle_volta_hmma_fp32_epilogue {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Traits, typename Cta_tile >
struct Swizzle_volta_hmma_fp32_epilogue<Traits,
                          Cta_tile,
                          Row,
                          true
                          > {

    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;

    // To support arbitrary N, we pad some values to a power-of-2.
    enum { N_WITH_PADDING = Next_power_of_two<Cta_tile::N>::VALUE }; 
    // The number of bytes for key memory instruction.
    enum { BYTES_PER_LDS = 16, BYTES_PER_STS = 8 };
    // The number of rows in shared memory.
    enum { ROWS = Xmma_tile::M_PER_XMMA_PER_CTA * 2 };
    // The size of a single row in bytes.
    enum { BYTES_PER_ROW = N_WITH_PADDING / 2 * Cta_tile::WARPS_K * sizeof(float) };
    // The skew to avoid bank conflicts.
    enum { BYTES_PER_SKEW = 16 };
    // The number of columns with the skew.
    enum { BYTES_PER_ROW_WITH_SKEW = BYTES_PER_ROW + BYTES_PER_SKEW };
    // The size in bytes in shared memory.
    enum { BYTES_PER_TILE = ROWS * BYTES_PER_ROW_WITH_SKEW };

    // The number of threads per output row. Each thread writes 8 elements.
    enum { THREADS_PER_ROW = N_WITH_PADDING / 8 };
    // The number of rows written in one STG.128.
    enum { ROWS_PER_STG = Cta_tile::THREADS_PER_CTA / THREADS_PER_ROW };

    // How we see the distribution of data.
    enum { THREADS_PER_XMMA_M = 8, THREADS_PER_XMMA_N = 4 };
    // The number of elements stored per thread.
    enum { M_PER_XMMA_PER_THREAD = 2, N_PER_XMMA_PER_THREAD = 4 };

    // Do we skip the syncthreads in the epilogue? Of course, not :)
    enum { SKIP_SYNCTHREADS = 0 };

    // Ctor.
    inline __device__ Swizzle_volta_hmma_fp32_epilogue(void *smem, int tidx) 
        : smem_(get_smem_pointer(smem)) {

        // The size of a single XMMA instruction is 16x16.
        static_assert(Xmma_tile::M_PER_XMMA == 16 && Xmma_tile::N_PER_XMMA == 16, "");

        // The number of warps.
        const int WARPS_M = Cta_tile::WARPS_M;
        const int WARPS_N = Cta_tile::WARPS_N;
        const int WARPS_K = Cta_tile::WARPS_K;

	// The masks to select the warps.
        const int WARP_MASK_M = Warp_masks<WARPS_M, WARPS_N, WARPS_K>::M;
        const int WARP_MASK_N = Warp_masks<WARPS_M, WARPS_N, WARPS_K>::N;
        const int WARP_MASK_K = Warp_masks<WARPS_M, WARPS_N, WARPS_K>::K;

        // The divisor for the warps.
        const int WARP_DIV_M =       1 *       1 * Cta_tile::THREADS_PER_WARP;
        const int WARP_DIV_N = WARPS_M *       1 * Cta_tile::THREADS_PER_WARP;
        const int WARP_DIV_K = WARPS_M * WARPS_N * Cta_tile::THREADS_PER_WARP;
        
        //     tidx   0: row =  0, col = 0
        //     tidx   1: row =  1, col = 0
        //     tidx   2: row =  0, col = 8
        //     tidx   3: row =  1, col = 8
        //     tidx   4: row =  2, col = 0
        //     tidx   5: row =  3, col = 0
        //     tidx   6: row =  2, col = 8
        //     tidx   7: row =  3, col = 8
        //     tidx   8: row =  4, col = 0
        //     tidx   9: row =  5, col = 0
        //     tidx  10: row =  4, col = 8
        //     tidx  11: row =  5, col = 8
        //     tidx  12: row =  6, col = 0
        //     tidx  13: row =  7, col = 0
        //     tidx  14: row =  6, col = 8
        //     tidx  15: row =  7, col = 8
        //     tidx  16: row =  8, col = 0
        //     tidx  17: row =  9, col = 0
        //     tidx  18: row =  8, col = 8
        //     tidx  19: row =  9, col = 8
        //     tidx  20: row = 10, col = 0
        //     tidx  21: row = 11, col = 0
        //     tidx  22: row = 10, col = 8
        //     tidx  23: row = 11, col = 8
        //     tidx  24: row = 12, col = 0
        //     tidx  25: row = 13, col = 0
        //     tidx  26: row = 12, col = 8
        //     tidx  27: row = 13, col = 8
        //     tidx  28: row = 14, col = 0
        //     tidx  29: row = 15, col = 0
        //     tidx  30: row = 14, col = 8
        //     tidx  31: row = 15, col = 8 

        // Compute the row and the column in shared memory. 
        int smem_write_row, smem_write_col;
        if( Xmma_tile::ENABLE_LDS_FAST_PATH ) {
        } else {
            smem_write_row = (tidx & WARP_MASK_M) / WARP_DIV_M * Xmma_tile::M_PER_XMMA*2 +
                             (tidx & 0x1c) / 2 +
                             (tidx & 0x01);

            // For group fprop/dgrd. C is divided into 2 halves along N dimension.
            // The fist warp stores the first half and the second warp stores the second half.
            smem_write_col = (tidx & WARP_MASK_N) / WARP_DIV_N *
                             (Cta_tile::GROUPS > 1 ? Cta_tile::N / WARPS_N / 2 :
                             Xmma_tile::N_PER_XMMA / 2) +
                             (tidx & WARP_MASK_K) / WARP_DIV_K * Cta_tile::N / 2 +
                             (tidx & 0x02);
        }

        // The corresponding offset.
        smem_write_offset_ = smem_write_row*BYTES_PER_ROW_WITH_SKEW + smem_write_col*sizeof(float);

        // The mapping of rows is "ugly"; we have:
        //
        //     tidx_div_tpr =  0 -> row =  0
        //     tidx_div_tpr =  1 -> row =  1
        //     tidx_div_tpr =  2 -> row = 16
        //     tidx_div_tpr =  3 -> row = 17
        //     tidx_div_tpr =  4 -> row =  2
        //     tidx_div_tpr =  5 -> row =  3
        //     tidx_div_tpr =  6 -> row = 18
        //     tidx_div_tpr =  7 -> row = 19
        //     tidx_div_tpr =  8 -> row =  8
        //     tidx_div_tpr =  9 -> row =  9
        //     tidx_div_tpr = 10 -> row = 24
        //     tidx_div_tpr = 11 -> row = 25
        //     tidx_div_tpr = 12 -> row = 10
        //     tidx_div_tpr = 13 -> row = 11
        //     tidx_div_tpr = 14 -> row = 26
        //     tidx_div_tpr = 15 -> row = 27
        //     ...
        //     tidx_div_tpr = 16 -> row = 32
        //

        // Each thread produces 8x fp16s -- how many threads do we need per row.
        const int tidx_div_tpr = tidx / THREADS_PER_ROW;
        const int tidx_mod_tpr = tidx % THREADS_PER_ROW;

        // The row and column read by a single thread.
        const int smem_read_row = (tidx_div_tpr & 0xf0)*2 + 
                                  (tidx_div_tpr & 0x09)*1 +
                                  (tidx_div_tpr & 0x04)/2 +
                                  (tidx_div_tpr & 0x02)*8;
        const int smem_read_col = (tidx_mod_tpr       )*BYTES_PER_LDS;

        // The corresponding offset.
        smem_read_offset_ = smem_read_row*BYTES_PER_ROW_WITH_SKEW + smem_read_col;
    }

    // Load from the tile in shared memory.
    template<typename Fragment_post_swizzle>
    inline __device__ void load(Fragment_post_swizzle &dst, int oi) const {
        // Compute our very ugly offset to jump from one row to the next one.
        int row;
        if( ROWS_PER_STG == 1 ) {
            // i =  0 -> row  0 || i =  4 -> row  2 || i =  8 -> row  8 || i = 12 -> row 10
            // i =  1 -> row  1 || i =  5 -> row  3 || i =  9 -> row  9 || i = 13 -> row 11
            // i =  2 -> row 16 || i =  6 -> row 18 || i = 10 -> row 24 || i = 14 -> row 26
            // i =  3 -> row 17 || i =  7 -> row 19 || i = 11 -> row 25 || i = 15 -> row 27
            //
            // i = 16 -> row 32 || ...
            row = (oi & 0xf0)*2 + (oi & 0x04)/2 + (oi & 0x02)*8 + (oi & 0x09);
        } else if( ROWS_PER_STG == 2 ) {
            // i =  0 -> row  0 || i =  2 -> row  2 || i =  4 -> row  8 || i =  6 -> row 10
            // i =  1 -> row 16 || i =  3 -> row 18 || i =  5 -> row 24 || i =  7 -> row 26
            //
            // i =  8 -> row 32 || ...
            row = (oi & 0xf8)*4 + (oi & 0x04)*2 + (oi & 0x02)*1 + (oi & 0x01)*16;
        } else if( ROWS_PER_STG == 4 ) {
            // i =  0 -> row  0 || i =  1 -> row  2 || i =  2 -> row  8 || i =  3 -> row 10
            //
            // i =  4 -> row 32 || ...
            row = (oi & 0xfc)*8 + (oi & 0x02)*4 + (oi & 0x01)*2;
        } else if( ROWS_PER_STG == 8 ) {
            // i =  0 -> row  0 || i =  1 -> row  8
            //
            // i =  2 -> row 32 || ...
            row = (oi & 0xfe)*16 + (oi & 0x01)*8;
        } else {
            row = (oi) * ROWS_PER_STG * 2;
        }

        #pragma unroll
        for( int ki = 0; ki < Cta_tile::WARPS_K; ++ki ) {
            
            // Add the WARP_K factor.
            int offset_0 = row*BYTES_PER_ROW_WITH_SKEW + ki * N_WITH_PADDING * 2;

            // The 1st group of 4 floats.
            uint4 tmp_0; 
            lds(tmp_0, smem_ + smem_read_offset_ + offset_0);
            dst.reg(ki*8 + 0) = tmp_0.x;
            dst.reg(ki*8 + 1) = tmp_0.y;
            dst.reg(ki*8 + 2) = tmp_0.z;
            dst.reg(ki*8 + 3) = tmp_0.w;

            // The 2nd group of 4 floats.
            int offset_1 = offset_0 + 4*BYTES_PER_ROW_WITH_SKEW;
            uint4 tmp_1;
            lds(tmp_1, smem_ + smem_read_offset_ + offset_1);
            dst.reg(ki*8 + 4) = tmp_1.x;
            dst.reg(ki*8 + 5) = tmp_1.y;
            dst.reg(ki*8 + 6) = tmp_1.z;
            dst.reg(ki*8 + 7) = tmp_1.w;
        }
    }

    // Store to the tile in shared memory.
    template<typename Fragment_pre_swizzle>
    inline __device__ void store(int ni, const Fragment_pre_swizzle &c) {
        #pragma unroll
        for( int mi = 0; mi < M_PER_XMMA_PER_THREAD; ++mi ) {
            int offset = mi * 16 * BYTES_PER_ROW_WITH_SKEW +
                         ni * (Cta_tile::GROUPS > 1 ? Xmma_tile::N_PER_XMMA / 2:
                         Xmma_tile::N_PER_XMMA_PER_CTA / 2) * sizeof(float);

            uint32_t ptr = smem_ + smem_write_offset_ + offset;
            sts(ptr +  0, make_uint2(c.reg(4*mi+0), c.reg(4*mi+1)));
            sts(ptr + 16, make_uint2(c.reg(4*mi+2), c.reg(4*mi+3)));
        }
    }

    inline __device__ void debug_print() const {
        // Not implemented
    }

    // The shared memory pointer in bytes.
    uint32_t smem_;
    // The write offset.
    int smem_write_offset_;
    // The read offset.
    int smem_read_offset_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// Not used, remain as reference due to special swizzle pattern
template< typename Traits, typename Cta_tile >
struct Swizzle_volta_hmma_fp32_epilogue<Traits,
                          Cta_tile,
                          Row,
                          false
                          > {

    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;

    // The number of bytes for key memory instruction.
    enum { BYTES_PER_LDS = 16, BYTES_PER_STS = 4 };
    // To support arbitrary N, we pad some values to a power-of-2.
    enum { N_WITH_PADDING = Next_power_of_two<Cta_tile::N>::VALUE }; 
    // The number of rows in shared memory.
    enum { ROWS = Xmma_tile::M_PER_XMMA_PER_CTA };
    // The size of a single row in bytes.
    enum { BYTES_PER_ROW = N_WITH_PADDING * sizeof(lwtlass::half_t) };
    // The skew to avoid bank conflicts.
    enum { BYTES_PER_SKEW = 16 };
    // The number of columns with the skew.
    enum { BYTES_PER_ROW_WITH_SKEW = BYTES_PER_ROW + BYTES_PER_SKEW };
    // The size in bytes in shared memory.
    enum { BYTES_PER_TILE = ROWS * BYTES_PER_ROW_WITH_SKEW };

    // The number of threads per output row. Each thread writes 8 elements.
    enum { THREADS_PER_ROW = N_WITH_PADDING / 8 };
    // The number of rows written in one STG.128.
    enum { ROWS_PER_STG = Cta_tile::THREADS_PER_CTA / THREADS_PER_ROW };

    // How we see the distribution of data.
    enum { THREADS_PER_XMMA_M = 8, THREADS_PER_XMMA_N = 4 };
    // The number of elements stored per thread.
    enum { M_PER_XMMA_PER_THREAD = 2, N_PER_XMMA_PER_THREAD = 4 };

    // Do we skip the syncthreads in the epilogue? Of course, not :)
    enum { SKIP_SYNCTHREADS = 0 };

    // Ctor.
    inline __device__ Swizzle_volta_hmma_fp32_epilogue(void *smem, int tidx) 
        : smem_(get_smem_pointer(smem)) {

        // The number of warps.
        const int WARPS_M = Cta_tile::WARPS_M;
        const int WARPS_N = Cta_tile::WARPS_N;
        const int WARPS_K = Cta_tile::WARPS_K;

	// The masks to select the warps.
        const int WARP_MASK_M = Warp_masks<WARPS_M, WARPS_N, WARPS_K>::M;
        const int WARP_MASK_N = Warp_masks<WARPS_M, WARPS_N, WARPS_K>::N;

        // The divisor for the warps.
        const int WARP_DIV_M =       1 *       1 * Cta_tile::THREADS_PER_WARP;
        const int WARP_DIV_N = WARPS_M *       1 * Cta_tile::THREADS_PER_WARP;

        //     tidx   0: row =  0, col =  0
        //     tidx   1: row =  1, col =  0
        //     tidx   2: row =  0, col =  4
        //     tidx   3: row =  1, col =  4
        //     tidx   4: row =  2, col =  0
        //     tidx   5: row =  3, col =  0
        //     tidx   6: row =  2, col =  4
        //     tidx   7: row =  3, col =  4
        //     tidx   8: row =  0, col =  8
        //     tidx   9: row =  1, col =  8
        //     tidx  10: row =  0, col = 12 
        //     tidx  11: row =  1, col = 12 
        //     tidx  12: row =  2, col =  8
        //     tidx  13: row =  3, col =  8
        //     tidx  14: row =  2, col = 12
        //     tidx  15: row =  3, col = 12
        //     tidx  16: row =  4, col =  0
        //     tidx  17: row =  5, col =  0
        //     tidx  18: row =  4, col =  4
        //     tidx  19: row =  5, col =  4
        //     tidx  20: row =  6, col =  0
        //     tidx  21: row =  7, col =  0
        //     tidx  22: row =  6, col =  4
        //     tidx  23: row =  7, col =  4
        //     tidx  24: row =  4, col =  8
        //     tidx  25: row =  5, col =  8
        //     tidx  26: row =  4, col = 12 
        //     tidx  27: row =  5, col = 12 
        //     tidx  28: row =  6, col =  8
        //     tidx  29: row =  7, col =  8
        //     tidx  30: row =  6, col = 12
        //     tidx  31: row =  7, col = 12
        //
        //     tidx  32: row =  8, col =  0

        // Compute the row and the column in shared memory. Each warp reads from a 16*16B segment.
        int smem_write_row, smem_write_col;
        if( Xmma_tile::ENABLE_LDS_FAST_PATH ) {
        } else {
            smem_write_row = (tidx & WARP_MASK_M) / WARP_DIV_M * Xmma_tile::M_PER_XMMA +
                             (tidx & 0x10) / 4 +
                             (tidx & 0x04) / 2 +
                             (tidx & 0x01);

            // For group fprop/dgrd. C is divided into 2 halves along N dimension.
            // The fist warp stores the first half and the second warp stores the second half.
            smem_write_col = (tidx & WARP_MASK_N) / WARP_DIV_N * (Cta_tile::GROUPS > 1 ? 
                             Cta_tile::N / WARPS_N : Xmma_tile::N_PER_XMMA) + 
                             (tidx & 0x08) / 2 +
                             (tidx & 0x02);
        }

        // The corresponding offset.
        smem_write_offset_ = smem_write_row*BYTES_PER_ROW_WITH_SKEW + 
                             smem_write_col*sizeof(lwtlass::half_t);

        // The mapping of rows is "ugly"; we have:
        //
        //     tidx_div_tpr =  0 -> row =  0
        //     tidx_div_tpr =  1 -> row =  1
        //     tidx_div_tpr =  2 -> row =  8
        //     tidx_div_tpr =  3 -> row =  9
        //     tidx_div_tpr =  4 -> row =  2
        //     tidx_div_tpr =  5 -> row =  3
        //     tidx_div_tpr =  6 -> row = 10
        //     tidx_div_tpr =  7 -> row = 11
        //     tidx_div_tpr =  8 -> row =  4
        //     tidx_div_tpr =  9 -> row =  5
        //     tidx_div_tpr = 10 -> row = 12
        //     tidx_div_tpr = 11 -> row = 13
        //     tidx_div_tpr = 12 -> row =  6
        //     tidx_div_tpr = 13 -> row =  7
        //     tidx_div_tpr = 14 -> row = 14
        //     tidx_div_tpr = 15 -> row = 15
        //     ...
        //     tidx_div_tpr = 16 -> row = 16
        //

        // Each thread produces 8x fp16s -- how many threads do we need per row.
        const int tidx_div_tpr = tidx / THREADS_PER_ROW;
        const int tidx_mod_tpr = tidx % THREADS_PER_ROW;

        // The row and column read by a single thread.
        const int smem_read_row = (tidx_div_tpr & 0xf0)*1 + 
                                  (tidx_div_tpr & 0x0c)/2 +
                                  (tidx_div_tpr & 0x02)*4 +
                                  (tidx_div_tpr & 0x01)*1;
        const int smem_read_col = (tidx_mod_tpr       )*BYTES_PER_LDS;

        // The corresponding offset.
        smem_read_offset_ = smem_read_row*BYTES_PER_ROW_WITH_SKEW + smem_read_col;
    }

    // Load from the tile in shared memory.
    template<typename Fragment_post_swizzle>
    inline __device__ void load(Fragment_post_swizzle &dst, int oi) const {

        // Compute our very ugly offset to jump from one row to the next one.
        int row;
        if( ROWS_PER_STG == 1 ) {
            // i =  0 -> row  0 || i =  4 -> row  2 || i =  8 -> row  4 || i = 12 -> row  6
            // i =  1 -> row  1 || i =  5 -> row  3 || i =  9 -> row  5 || i = 13 -> row  7
            // i =  2 -> row  8 || i =  6 -> row 10 || i = 10 -> row 12 || i = 14 -> row 14
            // i =  3 -> row  9 || i =  7 -> row 11 || i = 11 -> row 13 || i = 15 -> row 15
            //
            // i = 16 -> row 16 || ...
            row = (oi & 0xf1) + (oi & 0x0c)/2 + (oi & 0x02)*4;
        } else if( ROWS_PER_STG == 2 ) {
            // i =  0 -> row  0 || i =  2 -> row  2 || i =  4 -> row  4 || i =  6 -> row  6
            // i =  1 -> row  8 || i =  3 -> row 10 || i =  5 -> row 12 || i =  7 -> row 14
            //
            // i =  8 -> row 16 || ...
            row = (oi & 0xf8)*2 + (oi & 0x06)*1 + (oi & 0x01)*8;
        } else if( ROWS_PER_STG == 4 ) {
            // i =  0 -> row  0 || i =  1 -> row  2 || i =  2 -> row  4 || i =  3 -> row  6
            //
            // i =  4 -> row 16 || ...
            row = (oi & 0xfc)*4 + (oi & 0x03)*2;
        } else if( ROWS_PER_STG == 8 ) {
            // i =  0 -> row  0 || i =  1 -> row  4
            //
            // i =  2 -> row 16 || ...
            row = (oi & 0xfe)*8 + (oi & 0x01)*4;
        } else {
            row = (oi) * ROWS_PER_STG;
        }

        // The 1st group of 8 fp16s.
        uint4 tmp; 
        lds(tmp, smem_ + smem_read_offset_ + row*BYTES_PER_ROW_WITH_SKEW);
        dst.reg(0) = tmp.x;
        dst.reg(1) = tmp.y;
        dst.reg(2) = tmp.z;
        dst.reg(3) = tmp.w;
    }

    // Store to the tile in shared memory.
    template<typename Fragment_pre_swizzle>
    inline __device__ void store(int ni, const Fragment_pre_swizzle &c) {
        #pragma unroll
        for( int mi = 0; mi < M_PER_XMMA_PER_THREAD; ++mi ) {
            int offset = mi * 8 * BYTES_PER_ROW_WITH_SKEW +
                         ni * (Cta_tile::GROUPS > 1 ? Xmma_tile::N_PER_XMMA : 
                         Xmma_tile::N_PER_XMMA_PER_CTA) * sizeof(lwtlass::half_t);

            uint32_t ptr = smem_ + smem_write_offset_ + offset;
            sts(ptr +  0, c.reg(2*mi+0));
            sts(ptr + 16, c.reg(2*mi+1));
        }
    }

    inline __device__ void debug_print() const {
        // Not implemented
    }

    // The shared memory pointer in bytes.
    uint32_t smem_;
    // The write offset.
    int smem_write_offset_;
    // The read offset.
    int smem_read_offset_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Cta_tile, bool IN_CTA_SPLIT_K >
struct Swizzle_epilogue<Volta_hmma_fp32_traits, Cta_tile, Row, 16, IN_CTA_SPLIT_K>
    : public Swizzle_volta_hmma_fp32_epilogue<Volta_hmma_fp32_traits,
                                               Cta_tile,
                                               Row,
                                               true> {
    // The traits class.
    using Traits = Volta_hmma_fp32_traits;
    // The base class.
    using Base = Swizzle_volta_hmma_fp32_epilogue<Volta_hmma_fp32_traits, 
                                                   Cta_tile, 
                                                   Row,
                                                   true>;

    // Ctor.
    inline __device__ Swizzle_epilogue(void *smem, int tidx) : Base(smem, tidx) {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile
>
struct Swizzle_epilogue_interleaved<Volta_hmma_fp32_traits,
                                      Cta_tile,
                                      Col_interleaved,
                                      true> {
    inline __device__ void debug_print() const {
        // Not implemented
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile
>
struct Swizzle_epilogue_interleaved<Volta_hmma_fp32_interleaved_traits,
                                      Cta_tile,
                                      Col_interleaved,
                                      true> {
    inline __device__ void debug_print() const {
        // Not implemented
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile
>
struct Swizzle_epilogue_interleaved<Volta_hmma_fp32_traits,
                                      Cta_tile,
                                      Col_interleaved,
                                      false> {

    // The traits.
    using Traits = Volta_hmma_fp32_traits;
    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;

    // The size in bytes in shared memory.
    enum { BYTES_PER_TILE = 0 };
    // Do we skip the syncthreads in the epilogue? Of course, not :)
    enum { SKIP_SYNCTHREADS = 1 };

    enum { NUM_REGS = 4 };

    inline __device__ Swizzle_epilogue_interleaved(void *, int) { }

    // Load from the tile in shared memory.
    template<typename Fragment_post_swizzle>
    inline __device__ void load(Fragment_post_swizzle &dst, int oi) const {
        // Make sure that the number of register in the post-swizzle fragments is as expected.
        static_assert(Fragment_post_swizzle::NUM_REGS == 1, "");
        dst.reg(0) = regs_[oi];
    }

    // Store to the tile in shared memory.
    template<typename Fragment_pre_swizzle>
    inline __device__ void store(int ni, const Fragment_pre_swizzle &c) {
        // Make sure that the number of register in the pre-swizzle fragments is as expected.
        static_assert(Fragment_pre_swizzle::NUM_REGS == NUM_REGS, "");

        #pragma unroll
        for( int ii = 0; ii < Fragment_pre_swizzle::NUM_REGS; ++ii ) {
            regs_[ni * Fragment_pre_swizzle::NUM_REGS + ii] = c.reg(ii);
        }
    }

    inline __device__ void debug_print() const {
        // Not implemented
    }

    // Storage for the input registers.
    uint32_t regs_[Xmma_tile::XMMAS_N * NUM_REGS];
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile
>
struct Swizzle_epilogue_interleaved<Volta_hmma_fp32_interleaved_traits,
                                      Cta_tile,
                                      Col_interleaved,
                                      false> {

    // The traits.
    using Traits = Volta_hmma_fp32_interleaved_traits;
    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;

    // The size in bytes in shared memory.
    enum { BYTES_PER_TILE = 0 };
    // Do we skip the syncthreads in the epilogue? Of course, not :)
    enum { SKIP_SYNCTHREADS = 1 };

    enum { NUM_REGS = 4 };

    inline __device__ Swizzle_epilogue_interleaved(void *, int) { }

    // Load from the tile in shared memory.
    template<typename Fragment_post_swizzle>
    inline __device__ void load(Fragment_post_swizzle &dst, int oi) const {
        // Make sure that the number of register in the post-swizzle fragments is as expected.
        static_assert(Fragment_post_swizzle::NUM_REGS == 1, "");
        dst.reg(0) = regs_[oi];
    }

    // Store to the tile in shared memory.
    template<typename Fragment_pre_swizzle>
    inline __device__ void store(int ni, const Fragment_pre_swizzle &c) {
        // Make sure that the number of register in the pre-swizzle fragments is as expected.
        static_assert(Fragment_pre_swizzle::NUM_REGS == NUM_REGS, "");

        #pragma unroll
        for( int ii = 0; ii < Fragment_pre_swizzle::NUM_REGS; ++ii ) {
            regs_[ni * Fragment_pre_swizzle::NUM_REGS + ii] = c.reg(ii);
        }
    }

    inline __device__ void debug_print() const {
        // Not implemented
    }

    // Storage for the input registers.
    uint32_t regs_[Xmma_tile::XMMAS_N * NUM_REGS];
};

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// I M M A . 8 NHWC/TN layout
//
////////////////////////////////////////////////////////////////////////////////////////////////////

template< 
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile, 
    // The size of the STS.
    int BYTES_PER_STS,
    // The number of buffers per tile.
    int BUFFERS_PER_TILE,
    // Support gelu_erf or not
    bool IS_GELU_ERF
>
struct Smem_tile_a<Volta_imma_int8_int32_traits<IS_GELU_ERF>, Cta_tile, Row, BYTES_PER_STS, BUFFERS_PER_TILE> 
    : public Smem_tile_volta_row_a<Volta_imma_int8_int32_traits<IS_GELU_ERF>, 
                                   Cta_tile, 
                                   BYTES_PER_STS,
                                   BUFFERS_PER_TILE> {

    // The traits class.
    using Traits = Volta_imma_int8_int32_traits<IS_GELU_ERF>;
    // The base class.
    using Base = Smem_tile_volta_row_a<Volta_imma_int8_int32_traits<IS_GELU_ERF>, 
                                       Cta_tile, 
                                       BYTES_PER_STS,
                                       BUFFERS_PER_TILE>;
    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;
    // The fragment.
    using Fragment = Fragment_a<Traits, Row>;

    // The size of a single LDS in bytes.
    enum { BYTES_PER_LDS = 4 };
    // The number of bytes per row.
    enum { BYTES_PER_ROW = Base::N_WITH_PADDING };

    // Ctor.
    inline __device__ Smem_tile_a(void *smem, int tidx) : Base(smem, tidx) {

        // For documentation on the layout, see doc/xmma_smem_layout.xlsx.

        // The number of warps.
        const int WARPS_M = Cta_tile::WARPS_M;
        const int WARPS_N = Cta_tile::WARPS_N;
        const int WARPS_K = Cta_tile::WARPS_K;

        // The masks to select the warps.
        const int WARP_MASK_M = Warp_masks<WARPS_M, WARPS_N, WARPS_K>::M;
        //const int WARP_MASK_N = Warp_masks<WARPS_M, WARPS_N, WARPS_K>::N;
        //const int WARP_MASK_K = Warp_masks<WARPS_M, WARPS_N, WARPS_K>::K;

        // The divisor for the warps.
        const int WARP_DIV_M =       1 *       1 * Cta_tile::THREADS_PER_WARP;
        //const int WARP_DIV_N = WARPS_M *       1 * Cta_tile::THREADS_PER_WARP;
        //const int WARP_DIV_K = WARPS_M * WARPS_N * Cta_tile::THREADS_PER_WARP;

        // The row and column read by the thread.
        int smem_read_row, smem_read_col;

        static_assert(Base::N_WITH_PADDING >= 64 || 
                      Base::N_WITH_PADDING == 32 || 
                      Base::N_WITH_PADDING == 16, "");

        smem_read_row = (tidx & WARP_MASK_M) / WARP_DIV_M * Xmma_tile::M_PER_XMMA / 2 +
                        (tidx & 0x18) / 8;
        smem_read_col = (((tidx & 0x1f) >> 3) + (tidx & 0x4)) * BYTES_PER_STS +
                        (tidx & 0x3) * BYTES_PER_LDS;

        // The shared memory offset.
        this->smem_read_offset_ = smem_read_row*Base::BYTES_PER_ROW + smem_read_col;
    }

    // Load from shared memory.
    inline __device__ void load(Fragment (&a)[Xmma_tile::XMMAS_M], int ki) {
        // TODO: we may want to move this to ctor.
        int smem_read_offset0 = (this->smem_read_offset_ ^ (ki * 16));
        int smem_read_offset1 = (smem_read_offset0 ^ 64);

        #pragma unroll
        for( int mi = 0; mi < Xmma_tile::XMMAS_M; ++mi ) {
            int offset = mi * Xmma_tile::M_PER_XMMA_PER_CTA * BYTES_PER_ROW;

            uint32_t tmp0, tmp1;
            lds(tmp0, this->smem_ + this->smem_read_buffer_ + smem_read_offset0 + offset);
            a[mi].reg(0) = tmp0;
            lds(tmp1, this->smem_ + this->smem_read_buffer_ + smem_read_offset1 + offset + 512);
            a[mi].reg(1) = tmp1;
        }
    }

    // Rewind smem_read_offset for last LDS phase in main loop.-
    inline __device__ void reverse_smem_read_offset(int ki = 0) {
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
    // Support gelu_erf or not
    bool IS_GELU_ERF
>
struct Smem_tile_b<Volta_imma_int8_int32_traits<IS_GELU_ERF>, Cta_tile, Col, BYTES_PER_STS, BUFFERS_PER_TILE>
    : public Smem_tile_volta_col_b<Volta_imma_int8_int32_traits<IS_GELU_ERF>, 
                                   Cta_tile, 
                                   BYTES_PER_STS,
                                   BUFFERS_PER_TILE> {

    // The traits class.
    using Traits = Volta_imma_int8_int32_traits<IS_GELU_ERF>;
    // The base class.
    using Base = Smem_tile_volta_col_b<Volta_imma_int8_int32_traits<IS_GELU_ERF>, 
                                       Cta_tile, 
                                       BYTES_PER_STS,
                                       BUFFERS_PER_TILE>;
    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;
    // The fragment.
    using Fragment = Fragment_b<Traits, Col>;

    // The size of a single LDS in bytes.
    enum { BYTES_PER_LDS = 4 };
    // The number of bytes per row.
    enum { BYTES_PER_ROW = Base::N_WITH_PADDING };

    // Ctor.
    inline __device__ Smem_tile_b(void *smem, int tidx) : Base(smem, tidx) {

        // For documentation on the layout, see doc/xmma_smem_layout.xlsx.

        // The number of warps.
        const int WARPS_M = Cta_tile::WARPS_M;
        const int WARPS_N = Cta_tile::WARPS_N;
        const int WARPS_K = Cta_tile::WARPS_K;

        // The masks to select the warps.
        const int WARP_MASK_N = Warp_masks<WARPS_M, WARPS_N, WARPS_K>::N;
        //const int WARP_MASK_K = Warp_masks<WARPS_M, WARPS_N, WARPS_K>::K;

        // The divisor for the warps.
        const int WARP_DIV_N = WARPS_M *       1 * Cta_tile::THREADS_PER_WARP;
        //const int WARP_DIV_K = WARPS_M * WARPS_N * Cta_tile::THREADS_PER_WARP;

        // The row and column read by the thread.
        int smem_read_row, smem_read_col;

        static_assert(Base::N_WITH_PADDING >= 64 || 
                      Base::N_WITH_PADDING == 32 || 
                      Base::N_WITH_PADDING == 16, "");

        smem_read_row = (tidx & WARP_MASK_N) / WARP_DIV_N * Xmma_tile::N_PER_XMMA / 2 +
                        (tidx & 0x18) / 8;
        smem_read_col = (((tidx & 0x1f) >> 3) + (tidx & 0x4)) * BYTES_PER_STS +
                        (tidx & 0x3) * BYTES_PER_LDS;

        // The shared memory offset.
        this->smem_read_offset_ = smem_read_row*Base::BYTES_PER_ROW + smem_read_col;
    }

    // Rewind smem_read_offset for last LDS phase in main loop.-
    inline __device__ void reverse_smem_read_offset(int ki = 0) {
    }

    // Load from shared memory.
    inline __device__ void load(Fragment (&b)[Xmma_tile::XMMAS_N], int ki) {
        // TODO: We may want to put this in ctor.
        int smem_read_offset0 = (this->smem_read_offset_ ^ (ki * 16));
        int smem_read_offset1 = (smem_read_offset0 ^ 64);

        #pragma unroll
        for( int ni = 0; ni < Xmma_tile::XMMAS_N; ++ ni ) {
            int offset = ni * Xmma_tile::N_PER_XMMA_PER_CTA * BYTES_PER_ROW;

            // TODO: Can we fuse read_offset and read_buffer?
            uint32_t tmp0, tmp1;
            lds(tmp0, this->smem_ + this->smem_read_buffer_ + smem_read_offset0 + offset);
            b[ni].reg(0) = tmp0;
            lds(tmp1, this->smem_ + this->smem_read_buffer_ + smem_read_offset1 + offset + 512);
            b[ni].reg(1) = tmp1;
        }
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// TODO: this function is the same with Turing implementation
//       need figure out a way to merge the code
template< 
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile, 
    // Do we enable split-k?
    bool IN_CTA_SPLIT_K,
    // Support gelu_erf or not
    bool IS_GELU_ERF
>
struct Swizzle_epilogue<Volta_imma_int8_int32_traits<IS_GELU_ERF>, 
                        Cta_tile, 
                        Row, 
                        16,
                        IN_CTA_SPLIT_K> { 

    // The traits.
    using Traits = Volta_imma_int8_int32_traits<IS_GELU_ERF>;
    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;

    // To support arbitrary N, we pad some values to a power-of-2.
    enum { N_WITH_PADDING = Next_power_of_two<Cta_tile::N>::VALUE }; 
    // The number of bytes for key memory instruction.
    enum { BYTES_PER_LDS = 16, BYTES_PER_STG = 16, BYTES_PER_STS = 8 };
    // The number of rows in shared memory.
    enum { ROWS = Xmma_tile::M_PER_XMMA_PER_CTA };
    // The size of a single row in bytes.
    enum { BYTES_PER_ROW = N_WITH_PADDING * Cta_tile::WARPS_K * sizeof(float) };
    // The skew to avoid bank conflicts (4 means 4 threads).
    enum { BYTES_PER_SKEW = BYTES_PER_STS * 4 };
    // The number of columns with the skew.
    enum { BYTES_PER_ROW_WITH_SKEW = BYTES_PER_ROW + BYTES_PER_SKEW };
    // The size in bytes in shared memory.
    enum { BYTES_PER_TILE = ROWS * BYTES_PER_ROW_WITH_SKEW };

    // The number of threads per "pixel".
    enum { THREADS_PER_PIXEL = N_WITH_PADDING * sizeof(typename Traits::C_type) / BYTES_PER_STG };
    // The number of "pixels" written in one STS.128.
    enum { PIXELS_PER_STG = Cta_tile::THREADS_PER_CTA / THREADS_PER_PIXEL };

    // How we see the distribution of data.
    enum { THREADS_PER_XMMA_M = 8, THREADS_PER_XMMA_N = 4 };
    // The number of elements stored per thread.
    enum { M_PER_XMMA_PER_THREAD = 2, N_PER_XMMA_PER_THREAD = 4 };

    // Do we skip the syncthreads in the epilogue? Of course, not :)
    enum { SKIP_SYNCTHREADS = 0 };


    // Ctor.
    inline __device__ Swizzle_epilogue(void *smem, int tidx)
        : smem_(get_smem_pointer(smem)) {

        // Extract the number of warps.
        const int WARPS_M = Cta_tile::WARPS_M;
        const int WARPS_N = Cta_tile::WARPS_N;
        const int WARPS_K = Cta_tile::WARPS_K;

	// The masks to select the warps.
        const int WARP_MASK_M = Warp_masks<WARPS_M, WARPS_N, WARPS_K>::M;
        const int WARP_MASK_N = Warp_masks<WARPS_M, WARPS_N, WARPS_K>::N;
        const int WARP_MASK_K = Warp_masks<WARPS_M, WARPS_N, WARPS_K>::K;

        // The divisor for the warps.
        const int WARP_DIV_M =       1 *       1 * Cta_tile::THREADS_PER_WARP;
        const int WARP_DIV_N = WARPS_M *       1 * Cta_tile::THREADS_PER_WARP;
        const int WARP_DIV_K = WARPS_M * WARPS_N * Cta_tile::THREADS_PER_WARP;

        // The number of bytes in the N dimension.
        const int BYTES_PER_TILE_N = N_WITH_PADDING * sizeof(Traits::C_type);

        // Compute the row and the column in shared memory. Each warp reads from a 16*16B segment.
        const int smem_write_row = (tidx & WARP_MASK_M) / WARP_DIV_M * Xmma_tile::M_PER_XMMA + 
                                   (tidx & 0x1c) / 4;

        // For group fprop/dgrd. C is divided into 2 halves along N dimension.
        // The fist warp stores the first half and the second warp stores the second half.
        const int smem_write_col = (tidx & WARP_MASK_K) / WARP_DIV_K * BYTES_PER_TILE_N +
                                   (tidx & WARP_MASK_N) / WARP_DIV_N * 
                                   (Cta_tile::GROUPS > 1 ? Cta_tile::N / WARPS_N * sizeof(Traits::C_type) : 
                                   BYTES_PER_STS * Xmma_tile::N_PER_XMMA / 2 ) + (tidx & 0x03) * BYTES_PER_STS;

        // The correspondng offset.
        smem_write_offset_ = smem_write_row*BYTES_PER_ROW_WITH_SKEW + smem_write_col;

        // Elements loaded by each thread.
        const int elem_per_thd = Cta_tile::N / THREADS_PER_PIXEL;

        // Decompose into groups of size "THREADS_PER_PIXEL".
        const int tidx_div_tpp = tidx / THREADS_PER_PIXEL;
        const int tidx_mod_tpp = tidx % THREADS_PER_PIXEL;

        // The row and column read by a single thread.
        const int smem_read_row = tidx_div_tpp;
        const int smem_read_col = tidx_mod_tpp * elem_per_thd * sizeof(float);

        smem_read_offset_ = smem_read_row * BYTES_PER_ROW_WITH_SKEW +
                            smem_read_col; 
    }

    // Load from the tile in shared memory.
    template<typename Fragment_post_swizzle>
    inline __device__ void load(Fragment_post_swizzle &dst, int oi) const {
        #pragma unroll
        for( int ki = 0; ki < Fragment_post_swizzle::NUM_REGS * sizeof(float) /
            BYTES_PER_LDS; ++ki ) {
            int offset = oi * PIXELS_PER_STG * BYTES_PER_ROW_WITH_SKEW + 
                         ki * BYTES_PER_LDS;
            uint4 tmp;
            lds(tmp, smem_ + smem_read_offset_ + offset);
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
        for( int mi = 0; mi < M_PER_XMMA_PER_THREAD; ++mi ) {
            int offset = mi * THREADS_PER_XMMA_M * BYTES_PER_ROW_WITH_SKEW +
                         ni * Xmma_tile::N_PER_XMMA_PER_CTA * sizeof(int32_t); 

            uint32_t ptr = smem_ + smem_write_offset_ + offset; 
            sts(ptr +  0, make_uint2(c.reg(4*mi + 0), c.reg(4*mi + 1)));
            sts(ptr + 32, make_uint2(c.reg(4*mi + 2), c.reg(4*mi + 3)));
        }
    }

    inline __device__ void debug_print() const {
        // Not implemented
    }

    // The shared memory pointer in bytes.
    uint32_t smem_;
    // The write offset.
    int smem_write_offset_;
    // The read offset.
    int smem_read_offset_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// I M M A . 8 NC/32HW32 layout
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
struct Smem_tile_a<Volta_imma_interleaved_int8_int32_traits, 
                   Cta_tile, 
                   Col_interleaved, 
                   BYTES_PER_STS,
                   BUFFERS_PER_TILE> 
    : public Smem_tile_interleaved<Cta_tile, 
                                   Cta_tile::K,
                                   Cta_tile::M,
                                   8, // bits_per_elem
                                   BYTES_PER_STS,
                                   BUFFERS_PER_TILE> {

    // The traits class.
    using Traits = Volta_imma_interleaved_int8_int32_traits;
    // The base class.
    using Base = Smem_tile_interleaved<Cta_tile, 
                                       Cta_tile::K, 
                                       Cta_tile::M, 
                                       8,
                                       BYTES_PER_STS,
                                       BUFFERS_PER_TILE>;
 
    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;
    // The fragment.
    using Fragment = Fragment_a<Traits, Row>;

    // The size of a single LDS in bytes.
    enum { BYTES_PER_LDS = 8 };
    // The number of elements that are stored with a single STS.
    enum { ELEMENTS_PER_STS = BYTES_PER_STS * 8 / Base::BITS_PER_ELEMENT };
    // Interleaved elements.
    enum { ELEMENTS_PER_PACKET = 32 };
    // Bytes per packed.
    enum { BYTES_PER_PACKET = 32 };
    // The number of rows that are needed.
    enum { ROWS = Cta_tile::K / ELEMENTS_PER_PACKET };
    // The number of threads per row.
    enum { THREADS_PER_ROW = Cta_tile::THREADS_PER_CTA / ROWS };
    // The number of bytes per row.
    enum { BYTES_PER_ROW = Cta_tile::M * BYTES_PER_PACKET };
    // The size in bytes in shared memory.
    enum { BYTES_PER_TILE = ROWS * BYTES_PER_ROW };
    // The size of shared memory catch line.
    enum { BYTES_PER_SMEM_LINE = 128 };
    // Smem stride between two xmma
    //enum { LDSM_STRIDE_PER_XMMA = Cta_tile::GROUPS > 1 ? Xmma_tile::N_PER_XMMA * BYTES_PER_PACKET 
      //  : Xmma_tile::N_PER_XMMA_PER_CTA * BYTES_PER_PACKET };

    // Ctor.
    inline __device__ Smem_tile_a(void *smem, int tidx) : Base(smem, tidx) {
        // The number of warps.
        const int WARPS_M = Cta_tile::WARPS_M;
        const int WARPS_N = Cta_tile::WARPS_N;
        const int WARPS_K = Cta_tile::WARPS_K;

        // The masks to select the warps.
        const int WARP_MASK_M = Warp_masks<WARPS_M, WARPS_N, WARPS_K>::M;
        const int WARP_MASK_N = Warp_masks<WARPS_M, WARPS_N, WARPS_K>::N;
        const int WARP_MASK_K = Warp_masks<WARPS_M, WARPS_N, WARPS_K>::K;

        // The divisor for the warps.
        const int WARP_DIV_M =       1 *       1 * Cta_tile::THREADS_PER_WARP;
        const int WARP_DIV_N = WARPS_M *       1 * Cta_tile::THREADS_PER_WARP;
        const int WARP_DIV_K = WARPS_M * WARPS_N * Cta_tile::THREADS_PER_WARP;

        // The row and column read by the thread.
        int smem_read_row = Cta_tile::GROUPS > 1 
                            ? ((tidx & WARP_MASK_N) / WARP_DIV_N * ROWS / WARPS_N)
                            : ((tidx & WARP_MASK_K) / WARP_DIV_K * ROWS / WARPS_K);
        // 4 threads load one 16Byte with LDS>32
        int smem_read_col = (tidx & WARP_MASK_M) / WARP_DIV_M * BYTES_PER_PACKET * 
                            Xmma_tile::M_PER_XMMA + (tidx & 0x1f) * BYTES_PER_LDS;

        this->smem_read_offset_ = smem_read_row * BYTES_PER_ROW + smem_read_col;

        // The row/col written by the thread (8: 8 threads per 128B with 16 STS instruction).
        int smem_write_row = (tidx / THREADS_PER_ROW);
        int smem_write_col = (tidx % THREADS_PER_ROW);

        // The location where the thread writes its elements.
        this->smem_write_offset_ = smem_write_row * BYTES_PER_ROW + 
                                   smem_write_col * BYTES_PER_STS;
    }

    // Rewind smem_read_offset for last LDS phase in main loop.-
    inline __device__ void reverse_smem_read_offset(int ki = 0) {
    }

    // Load from shared memory.
    inline __device__ void load(Fragment (&a)[Xmma_tile::XMMAS_M], int ki) {
        #pragma unroll
        for( int mi = 0; mi < Xmma_tile::XMMAS_M; ++mi ) {
            #pragma unroll
            for( int i = 0; i < Xmma_tile::M_PER_XMMA/8; ++i ) {
                int offset = this->smem_read_offset_ + 
                             ki * BYTES_PER_ROW +
                             mi * Xmma_tile::M_PER_XMMA_PER_CTA * BYTES_PER_PACKET +
                             i  * 8 * BYTES_PER_PACKET;
                uint2 tmp;
                lds(tmp, this->smem_ + this->smem_read_buffer_ + offset);
                a[mi].reg(i*2  ) = tmp.x;
                a[mi].reg(i*2+1) = tmp.y;
            }// end i
        }// end mi
    }

    // Compute the store pointers.
    template< int N >
    inline __device__ void compute_store_pointers(uint32_t (&ptrs)[N]) {
        #pragma unroll
        for( int ii = 0; ii < N; ++ii ) {
            int offset = this->smem_write_offset_ + this->smem_write_buffer_ 
                + ii * THREADS_PER_ROW * BYTES_PER_STS;
            ptrs[ii] = this->smem_ + offset;
        }
    }

    // Store to the tile in shared memory.
    template< int N >
    inline __device__ void store(const uint4 (&data)[N]) {
        uint32_t smem_ptrs[N];
        this->compute_store_pointers(smem_ptrs);
        sts(smem_ptrs, data);
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
struct Smem_tile_b<Volta_imma_interleaved_int8_int32_traits, 
                   Cta_tile, 
                   Row_interleaved, 
                   BYTES_PER_STS,
                   BUFFERS_PER_TILE>
    : public Smem_tile_interleaved<Cta_tile, 
                                   Cta_tile::K, 
                                   Cta_tile::N, 
                                   8, // bits_per_elem 
                                   BYTES_PER_STS,
                                   BUFFERS_PER_TILE> {

    // The traits class.
    using Traits = Volta_imma_interleaved_int8_int32_traits;
    // The base class.
    using Base = Smem_tile_interleaved<Cta_tile,
                                       Cta_tile::K, 
                                       Cta_tile::N, 
                                       8, 
                                       BYTES_PER_STS,
                                       BUFFERS_PER_TILE>;
    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;
    // The fragment.
    using Fragment = Fragment_b<Traits, Col>;

    // The size of a single LDS in bytes.
    enum { BYTES_PER_LDS = 8 };
    // The number of elements that are stored with a single STS.
    enum { ELEMENTS_PER_STS = BYTES_PER_STS * 8 / Base::BITS_PER_ELEMENT };
    // Interleaved elements.
    enum { ELEMENTS_PER_PACKET = 32 };
    // Bytes per packed.
    enum { BYTES_PER_PACKET = 32 };
    // The number of rows that are needed.
    enum { ROWS = Cta_tile::K / ELEMENTS_PER_PACKET };
    // The number of threads per row.
    enum { THREADS_PER_ROW = Cta_tile::THREADS_PER_CTA / ROWS };
    // The number of bytes per row.
    enum { BYTES_PER_ROW = Cta_tile::N * BYTES_PER_PACKET };
    // The size in bytes in shared memory.
    enum { BYTES_PER_TILE = ROWS * BYTES_PER_ROW };

    // STS_PER_THREAD
    enum { STS_PER_THREAD = BYTES_PER_ROW / THREADS_PER_ROW / BYTES_PER_STS};
    // Ctor.
    inline __device__ Smem_tile_b(void *smem, int tidx) : Base(smem, tidx) {
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
        int smem_read_row = (tidx & WARP_MASK_K) / WARP_DIV_K * ROWS / WARPS_K;
        int smem_read_col = (tidx & WARP_MASK_N) / WARP_DIV_N *
                            (Cta_tile::GROUPS > 1 ? Cta_tile::N / WARPS_N : Xmma_tile::N_PER_XMMA) *
                            (BYTES_PER_STS / BYTES_PER_LDS) +
                            (tidx & 0x1f) / 2;


        int smem_read_col_inner_id  = tidx % 2;
        // The shared memory offset.
        this->smem_read_offset_ = smem_read_row*BYTES_PER_ROW +
                                  smem_read_col*BYTES_PER_STS +
                                  smem_read_col_inner_id*BYTES_PER_LDS;

        // The row/col written b the thread (8: 8 threads per 128B with 16 STS instruction).
        int smem_write_row = (tidx / THREADS_PER_ROW);
        int smem_write_col = (tidx % THREADS_PER_ROW);

        // The location where the thread writes its elements.
        this->smem_write_offset_ = smem_write_row * BYTES_PER_ROW +
                                   smem_write_col * BYTES_PER_STS;

        if ( Cta_tile::GROUPS >=4){
            int per_group_k = Cta_tile::K / Cta_tile::GROUPS;
            // for C/K = 16 use ldg/sts128, for c/k=8 use ldg/sts64 for c/k = 4, use ldg/sts32
            // FIXME: hack here, in NC32HW32 Gmem tile, when we callwlate address,
            //  suppose we use ldg.128, so each packet(col) use 2 threads

            // fill zero since we don't use LDG PNZ
            uint32_t smem_ptrs[STS_PER_THREAD];
            #pragma unroll
            for ( int i = 0; i < BUFFERS_PER_TILE; ++i ) {
                this->compute_store_pointers(smem_ptrs);
                uint4 zero = make_uint4(0, 0, 0, 0);
                #pragma unroll
                for( int ii = 0; ii < STS_PER_THREAD; ++ii ) {
                    sts(smem_ptrs[ii], zero);
                }
                this->move_next_write_buffer();
            }
            // find the origin k
            static const int n_cdiv32_hw_c32_reorder_col[32] = {
              0,  1,  8,  9, 16, 17, 24, 25,  2,  3, 10, 11, 18, 19, 26, 27,
              4,  5, 12, 13, 20, 21, 28, 29,  6,  7, 14, 15, 22, 23, 30, 31
            };
            int gmem_col_id = smem_read_col % 64;
            gmem_col_id = gmem_col_id / (BYTES_PER_STS/BYTES_PER_LDS);
            int xform_k = n_cdiv32_hw_c32_reorder_col[gmem_col_id];

            // for each warp move b using diagonal pattern,
            // same with ampere, see ampere code for detail
            int group_id = xform_k / per_group_k;

            if (Cta_tile::GROUPS == 4) smem_read_col ^= group_id & 1;
            if (Cta_tile::GROUPS == 8) {
                smem_read_col ^= (group_id / 2) & 1;
                smem_read_col_inner_id ^= (group_id % 2);
            }
            if (Cta_tile::GROUPS == 16) {
                smem_read_col ^= (group_id / 4) & 1;
                smem_read_col_inner_id ^= ((group_id & 0x2) / 2);
            }
           // The location where the thread read its elements.
            this->smem_read_offset_ = smem_read_row * BYTES_PER_ROW +
                                  smem_read_col * BYTES_PER_STS +
                                  smem_read_col_inner_id * BYTES_PER_LDS;

            // each pack use one thread(one STS128) rather than two
            smem_write_col = (tidx % THREADS_PER_ROW) * 2;

            this->smem_write_offset_ = smem_write_row * BYTES_PER_ROW +
                                       smem_write_col * BYTES_PER_STS;
        }
    }

    // Rewind smem_read_offset for last LDS phase in main loop.-
    inline __device__ void reverse_smem_read_offset(int ki = 0) {
    }

    // Load from shared memory.
    inline __device__ void load(Fragment (&b)[Xmma_tile::XMMAS_N], int ki) {
        #pragma unroll
        for( int ni = 0; ni < Xmma_tile::XMMAS_N; ++ni ) {
            #pragma unroll
            for( int j = 0; j < Xmma_tile::N_PER_XMMA/8; ++j ) { 
                int offset = this->smem_read_offset_ +
                             ki * BYTES_PER_ROW + 
                             ni * Xmma_tile::N_PER_XMMA_PER_CTA * BYTES_PER_PACKET + 
                             j  * 8 * BYTES_PER_PACKET;

                if (Cta_tile::GROUPS == 16){
                    uint32_t tmp;
                    lds(tmp, this->smem_ + this->smem_read_buffer_ + offset);
                    b[ni].reg(j*2  ) = j < 2 ? tmp : 0;;
                    b[ni].reg(j*2+1) = j < 2 ? 0 : tmp;

                }
                else {
                    uint2 tmp;
                    lds(tmp, this->smem_ + this->smem_read_buffer_ + offset);
                    b[ni].reg(j*2  ) = tmp.x;
                    b[ni].reg(j*2+1) = tmp.y;
                }
            }// end j
        }// end ni
    }

    // Compute the store pointers.
    template< int N >
    inline __device__ void compute_store_pointers(uint32_t (&ptrs)[N]) {
        #pragma unroll
        for( int ii = 0; ii < N; ++ii ) {
            int offset = this->smem_write_offset_ + this->smem_write_buffer_ 
                + ii * THREADS_PER_ROW * BYTES_PER_STS;
            ptrs[ii] = this->smem_ + offset;
        }
    }

    // Store to the tile in shared memory.
    template< int N >
    inline __device__ void store(const uint4 (&data)[N]) {
        uint32_t smem_ptrs[N];
        this->compute_store_pointers(smem_ptrs);
        if(Cta_tile::GROUPS == 8){
            sts_force_64(smem_ptrs, data);
        }
        else if(Cta_tile::GROUPS == 16) {
            sts_force_32(smem_ptrs,data);
        }
        else{
            sts(smem_ptrs, data);
        }
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile
>
struct Swizzle_epilogue_interleaved<Volta_imma_interleaved_int8_int32_traits,
                                      Cta_tile,
                                      Col_interleaved,
                                      true> {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile
>
struct Swizzle_epilogue_interleaved<Volta_imma_interleaved_int8_int32_traits,
                                      Cta_tile,
                                      Col_interleaved,
                                      false> {

    // The traits.
    using Traits = Volta_imma_interleaved_int8_int32_traits;
    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;

    // The size in bytes in shared memory.
    enum { BYTES_PER_TILE = 0 };
    // Do we skip the syncthreads in the epilogue? Of course, not :)
    enum { SKIP_SYNCTHREADS = 1 };
    // Bytes per lds, sts.
    enum { BYTES_PER_LDS = 16, BYTES_PER_STS = 8 };

    enum { NUM_REGS = 16 };

    // Ctor.
    inline __device__ Swizzle_epilogue_interleaved(void *smem, int tidx) {
    }

    // Load from the tile in shared memory.
    template<typename Fragment_post_swizzle>
    inline __device__ void load(Fragment_post_swizzle &dst, int oi) const {
        // Make sure that the number of register in the post-swizzle fragments is as expected.
        static_assert(Fragment_post_swizzle::NUM_REGS == 8, "");

        dst.elt(0) = regs_[oi*8];
        dst.elt(1) = regs_[oi*8+1];
        dst.elt(2) = regs_[oi*8+2];
        dst.elt(3) = regs_[oi*8+3];
        dst.elt(4) = regs_[oi*8+4];
        dst.elt(5) = regs_[oi*8+5];
        dst.elt(6) = regs_[oi*8+6];
        dst.elt(7) = regs_[oi*8+7];
    }

    // Store to the tile in shared memory.
    template<typename Fragment_pre_swizzle>
    inline __device__ void store(int ni, const Fragment_pre_swizzle &c) {
        // Make sure that the number of register in the pre-swizzle fragments is as expected.
        static_assert(Fragment_pre_swizzle::NUM_REGS == 16, "");

        #pragma unroll
        for( int ii = 0; ii < Fragment_pre_swizzle::NUM_REGS; ++ii ) {
            regs_[ni*Fragment_pre_swizzle::NUM_REGS + ii] = c.elt(ii);
        }
    }

    inline __device__ void debug_print() const {
        // Not implemented
    }

    // Storage for the input registers.
    float regs_[Xmma_tile::XMMAS_N * NUM_REGS];
};

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace xmma

