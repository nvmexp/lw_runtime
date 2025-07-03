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

#include <xmma/turing/fragment.h>
#include <xmma/warp_masks.h>
#include <xmma/smem_tile.h>

namespace xmma {

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// x M M A 
//
////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Traits, int N >
struct Rows_per_xor_pattern_turing_a {
    // The size in bits.
    enum { N_IN_BITS = N * Traits::BITS_PER_ELEMENT_A };
    // The number of rows.
    enum { VALUE = N_IN_BITS <= 256 ? 2 : (N_IN_BITS <= 512 ? 4 : 8) };
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Traits, int N >
struct Rows_per_xor_pattern_turing_b {
    // The size in bits.
    enum { N_IN_BITS = N * Traits::BITS_PER_ELEMENT_B };
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
    // How many rows to use for the XOR pattern to avoid bank conflicts?
    int ROWS_PER_XOR_PATTERN_ = Rows_per_xor_pattern_turing_a<Traits, Cta_tile::M>::VALUE
>
struct Smem_tile_turing_col_a : public Smem_tile_without_skews<Cta_tile, 
                                                               Cta_tile::K, 
                                                               Cta_tile::M, 
                                                               Traits::BITS_PER_ELEMENT_A, 
                                                               BYTES_PER_STS,
                                                               BUFFERS_PER_TILE,
                                                               0,
                                                               ROWS_PER_XOR_PATTERN_,
                                                               1> {

    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;
    // The base class.
    using Base = Smem_tile_without_skews<Cta_tile, 
                                         Cta_tile::K, 
                                         Cta_tile::M, 
                                         Traits::BITS_PER_ELEMENT_A, 
                                         BYTES_PER_STS,
                                         BUFFERS_PER_TILE,
                                         0,
                                         ROWS_PER_XOR_PATTERN_,
                                         1>;
    // The fragment.
    using Fragment = Fragment_a<Traits, Col>;

    // The size of a single LDS in bytes.
    enum { BYTES_PER_LDS = 16 };
    // The number of elements per LDS.
    enum { ELEMENTS_PER_LDS = BYTES_PER_LDS * 8 / Traits::BITS_PER_ELEMENT_A };

    // Ctor.
    inline __device__ Smem_tile_turing_col_a(void *smem, int tidx) : Base(smem, tidx) {

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

        static_assert(Base::ROWS_PER_XOR_PATTERN == 8 || 
                      Base::ROWS_PER_XOR_PATTERN == 4 || 
                      Base::ROWS_PER_XOR_PATTERN == 2, "");

        if( Base::ROWS_PER_XOR_PATTERN == 8 ) {
            smem_read_row = (tidx & WARP_MASK_K) / WARP_DIV_K * Xmma_tile::XMMAS_K * 8 + 
                            (tidx & 0x07);
            smem_read_col = (tidx & 0x07);
        } else if( Base::ROWS_PER_XOR_PATTERN == 4 ) {
            smem_read_row = (tidx & WARP_MASK_K) / WARP_DIV_K * Xmma_tile::XMMAS_K * 4 + 
                            (tidx & 0x06) / 2;
            smem_read_col = (tidx & 0x01) * 4 + (tidx & 0x06) / 2;
        } else if( Base::ROWS_PER_XOR_PATTERN == 2 ) {
            smem_read_row = (tidx & WARP_MASK_K) / WARP_DIV_K * Xmma_tile::XMMAS_K * 2 +
                            (tidx & 0x04) / 4;
            smem_read_col = (tidx & 0x03) * 2 + (tidx & 0x04) / 4;
        }

        // Swizzle the column for the 2nd halfwarp.
        smem_read_col ^= (tidx & WARP_MASK_M) / WARP_DIV_M * 2 + (tidx & 0x08) / 8;
        
        // The shared memory offset.
        this->smem_read_offset_ = smem_read_row*Base::BYTES_PER_ROW + smem_read_col*BYTES_PER_LDS;
    }

    // Rewind smem_read_offset for last LDS phase in main loop.-
    inline __device__ void reverse_smem_read_offset(int ki = 0) {

        static_assert(Cta_tile::WARPS_M == 4 ||
                      Cta_tile::WARPS_M == 2 ||
                      Xmma_tile::XMMAS_M <= 4, "");

        #pragma unroll
        for( int mi = 0; mi < Xmma_tile::XMMAS_M; ++mi ) {

            // Move the pointer for the next ni. I expect the compiler to not recompute those.
            if( Cta_tile::WARPS_M == 4 ) {
                // Nothing to do!
            } else if( Cta_tile::WARPS_M == 2 && Xmma_tile::XMMAS_M > 1 ) {
                this->smem_read_offset_ ^= BYTES_PER_LDS * (4);
            } else if( Xmma_tile::XMMAS_M == 4 ) {
                this->smem_read_offset_ ^= BYTES_PER_LDS * (mi % 2 == 0 ? 2 : 6);
            } else if( Xmma_tile::XMMAS_M == 3 && mi == 0 ) {
                this->smem_read_offset_ ^= BYTES_PER_LDS * 2;
            } else if( Xmma_tile::XMMAS_M == 3 && mi == 1 ) {
                this->smem_read_offset_ ^= BYTES_PER_LDS * 6;
            } else if( Xmma_tile::XMMAS_M == 3 && mi == 2 ) {
                this->smem_read_offset_ ^= BYTES_PER_LDS * 4;
            } else if( Xmma_tile::XMMAS_M == 2 ) { 
                this->smem_read_offset_ ^= BYTES_PER_LDS * 2;
            }
        }
    }

    // Load from shared memory.
    inline __device__ void load(Fragment (&a)[Xmma_tile::XMMAS_M], int ki) {
        #pragma unroll
        for( int mi = 0; mi < Xmma_tile::XMMAS_M; ++mi ) {

            // Prepare the offset.
            int offset = ki * Base::ROWS_PER_XOR_PATTERN * Base::BYTES_PER_ROW;
            if( Cta_tile::WARPS_M == 1 ) {
                offset += this->smem_read_offset_;
            } else if( Cta_tile::WARPS_M == 2 && Xmma_tile::XMMAS_M > 1 ) {
                offset += this->smem_read_offset_ + (mi/2) * Xmma_tile::M_PER_XMMA_PER_CTA*4;
            } else {
                offset += this->smem_read_offset_ + (mi  ) * Xmma_tile::M_PER_XMMA_PER_CTA*2;
            }
            // Load the data using LDSM.MT88.2.
            uint2 tmp;
            ldsmt(tmp, this->smem_ + this->smem_read_buffer_ + offset);
            a[mi].reg(0) = tmp.x;
            a[mi].reg(1) = tmp.y;

            static_assert(Cta_tile::WARPS_M == 4 ||
                          Cta_tile::WARPS_M == 2 ||
                          Xmma_tile::XMMAS_M <= 4, "");

            // Move the pointer for the next ni. I expect the compiler to not recompute those.
            if( Cta_tile::WARPS_M == 4 ) {
                // Nothing to do!
            } else if( Cta_tile::WARPS_M == 2 && Xmma_tile::XMMAS_M > 1 ) {
                this->smem_read_offset_ ^= BYTES_PER_LDS * (4);
            } else if( Xmma_tile::XMMAS_M == 4 ) {
                this->smem_read_offset_ ^= BYTES_PER_LDS * (mi % 2 == 0 ? 2 : 6);
            } else if( Xmma_tile::XMMAS_M == 3 && mi == 0 ) {
                this->smem_read_offset_ ^= BYTES_PER_LDS * 2;
            } else if( Xmma_tile::XMMAS_M == 3 && mi == 1 ) {
                this->smem_read_offset_ ^= BYTES_PER_LDS * 6;
            } else if( Xmma_tile::XMMAS_M == 3 && mi == 2 ) {
                this->smem_read_offset_ ^= BYTES_PER_LDS * 4;
            } else if( Xmma_tile::XMMAS_M == 2 ) { 
                this->smem_read_offset_ ^= BYTES_PER_LDS * 2;
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
    // How many rows to use for the XOR pattern to avoid bank conflicts?
    int ROWS_PER_XOR_PATTERN_ = Rows_per_xor_pattern_turing_a<Traits, Cta_tile::K>::VALUE
>
struct Smem_tile_turing_row_a : public Smem_tile_without_skews<Cta_tile, 
                                                               Cta_tile::M, 
                                                               Cta_tile::K, 
                                                               Traits::BITS_PER_ELEMENT_A, 
                                                               BYTES_PER_STS,
                                                               BUFFERS_PER_TILE,
                                                               0,
                                                               ROWS_PER_XOR_PATTERN_,
                                                               1> {

    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;
    // The base class.
    using Base = Smem_tile_without_skews<Cta_tile, 
                                         Cta_tile::M, 
                                         Cta_tile::K, 
                                         Traits::BITS_PER_ELEMENT_A, 
                                         BYTES_PER_STS,
                                         BUFFERS_PER_TILE,
                                         0,
                                         ROWS_PER_XOR_PATTERN_, 
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
    inline __device__ Smem_tile_turing_row_a(void *smem, int tidx) : Base(smem, tidx) {

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

        static_assert(Base::ROWS_PER_XOR_PATTERN == 8 || 
                      Base::ROWS_PER_XOR_PATTERN == 4 || 
                      Base::ROWS_PER_XOR_PATTERN == 2, "");

        if( Base::ROWS_PER_XOR_PATTERN == 8 ) {
            smem_read_row = (tidx & WARP_MASK_M) / WARP_DIV_M * Xmma_tile::M_PER_XMMA / 1 + 
                            (tidx & 0x0f);
            smem_read_col = (tidx & 0x07);
            // For group fprop/dgrd. A is divided into 2 halves along K dimension.
            // The fist warp takes the first half and the second warp takes the second half.
            if ( Cta_tile::GROUPS > 1 && WARPS_K == 1 && WARPS_N > 1 ) {
                smem_read_col ^= (tidx & WARP_MASK_N) / WARP_DIV_N * (Cta_tile::K / WARPS_N) / 
                                 (BYTES_PER_LDS * 8 / Base::BITS_PER_ELEMENT);
            }
        } else if( Base::ROWS_PER_XOR_PATTERN == 4 ) {
            smem_read_row = (tidx & WARP_MASK_M) / WARP_DIV_M * Xmma_tile::M_PER_XMMA / 2 + 
                            (tidx & 0x0e) / 2;
            smem_read_col = (tidx & 0x06) / 2 + (tidx & 0x01) * 4; 
        } else if( Base::ROWS_PER_XOR_PATTERN == 2 ) {
            smem_read_row = (tidx & WARP_MASK_M) / WARP_DIV_M * Xmma_tile::M_PER_XMMA / 4 + 
                            (tidx & 0x0c) / 4;
            smem_read_col = (tidx & 0x04) / 4 + (tidx & 0x03) * 2; 
        }

        static_assert(WARPS_K <= 2, "");

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
            uint2 tmp;
            ldsm(tmp, this->smem_ + this->smem_read_offset_ + this->smem_read_buffer_ + offset);
            a[mi].reg(0) = tmp.x;
            a[mi].reg(1) = tmp.y;
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
    int ROWS_PER_XOR_PATTERN_ = Rows_per_xor_pattern_turing_b<Traits, Cta_tile::K>::VALUE
>
struct Smem_tile_turing_col_b : public Smem_tile_without_skews<Cta_tile, 
                                                               Cta_tile::N, 
                                                               Cta_tile::K, 
                                                               Traits::BITS_PER_ELEMENT_B,
                                                               BYTES_PER_STS,
                                                               BUFFERS_PER_TILE,
                                                               0,
                                                               ROWS_PER_XOR_PATTERN_,
                                                               1> {

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
    inline __device__ Smem_tile_turing_col_b(void *smem, int tidx) : Base(smem, tidx) {

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

        static_assert(Base::ROWS_PER_XOR_PATTERN == 8 || 
                      Base::ROWS_PER_XOR_PATTERN == 4 || 
                      Base::ROWS_PER_XOR_PATTERN == 2, "");

        if( Base::ROWS_PER_XOR_PATTERN == 8 ) {
            // For group fprop. B is divided into 2 halves along N dimension.
            // The fist warp takes the first half and the second warp takes the second half.
            if ( Cta_tile::GROUPS > 1 && WARPS_K == 1 && WARPS_N > 1 ) {
                smem_read_row = (tidx & WARP_MASK_N) / WARP_DIV_N * (Cta_tile::N / WARPS_N) / 1 + 
                                (tidx & 0x0f);
            } else {
                smem_read_row = (tidx & WARP_MASK_N) / WARP_DIV_N * Xmma_tile::N_PER_XMMA / 1 + 
                                (tidx & 0x0f);
            }    
            smem_read_col = (tidx & 0x07);
        } else if( Base::ROWS_PER_XOR_PATTERN == 4 ) {
            smem_read_row = (tidx & WARP_MASK_N) / WARP_DIV_N * Xmma_tile::N_PER_XMMA / 2 + 
                            (tidx & 0x0e) / 2;
            smem_read_col = (tidx & 0x06) / 2 + (tidx & 0x01) * 4; 
        } else if( Base::ROWS_PER_XOR_PATTERN == 2 ) {
            smem_read_row = (tidx & WARP_MASK_N) / WARP_DIV_N * Xmma_tile::N_PER_XMMA / 4 + 
                            (tidx & 0x0c) / 4;
            smem_read_col = (tidx & 0x04) / 4 + (tidx & 0x03) * 2; 
        }

        static_assert(WARPS_K <= 2, "");

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
            int offset = ni * (Cta_tile::GROUPS > 1 ? Xmma_tile::N_PER_XMMA : 
                         Xmma_tile::N_PER_XMMA_PER_CTA) * Base::BYTES_PER_ROW_BEFORE_PACKING;
            uint2 tmp; 
            ldsm(tmp, this->smem_ + this->smem_read_offset_ + this->smem_read_buffer_ + offset);
            b[ni].reg(0) = tmp.x;
            b[ni].reg(1) = tmp.y;
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
    int ROWS_PER_XOR_PATTERN_ = Rows_per_xor_pattern_turing_b<Traits, Cta_tile::N>::VALUE
>
struct Smem_tile_turing_row_b : public Smem_tile_without_skews<Cta_tile, 
                                                               Cta_tile::K, 
                                                               Cta_tile::N, 
                                                               Traits::BITS_PER_ELEMENT_B, 
                                                               BYTES_PER_STS,
                                                               BUFFERS_PER_TILE,
                                                               0,
                                                               ROWS_PER_XOR_PATTERN_,
                                                               1> {

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
                                         1>;
    // The fragment.
    using Fragment = Fragment_b<Traits, Row>;

    // The size of a single LDS in bytes.
    enum { BYTES_PER_LDS = 16 };
    // The number of elements per LDS.
    enum { ELEMENTS_PER_LDS = BYTES_PER_LDS * 8 / Traits::BITS_PER_ELEMENT_B };

    // Ctor.
    inline __device__ Smem_tile_turing_row_b(void *smem, int tidx) : Base(smem, tidx) {

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

        static_assert(Base::ROWS_PER_XOR_PATTERN == 8 || 
                      Base::ROWS_PER_XOR_PATTERN == 4 || 
                      Base::ROWS_PER_XOR_PATTERN == 2, "");

        if( Base::ROWS_PER_XOR_PATTERN == 8 ) {
            // For group dgrd. B is divided into 2 halves along K dimension.
            // The fist warp takes the first half and the second warp takes the second half.
            if ( Cta_tile::GROUPS > 1 && WARPS_K == 1 && WARPS_N > 1 ) {
                smem_read_row = (tidx & WARP_MASK_N) / WARP_DIV_N * (Cta_tile::N / WARPS_N) +
                                (tidx & 0x07);
            } else {
                smem_read_row = (tidx & WARP_MASK_K) / WARP_DIV_K * Xmma_tile::XMMAS_K * 8 + 
                                (tidx & 0x07);
            }
            smem_read_col = (tidx & 0x07);
        } else if( Base::ROWS_PER_XOR_PATTERN == 4 ) {
            smem_read_row = (tidx & WARP_MASK_K) / WARP_DIV_K * Xmma_tile::XMMAS_K * 4 + 
                            (tidx & 0x06) / 2;
            smem_read_col = (tidx & 0x01) * 4 + (tidx & 0x06) / 2;
        } else if( Base::ROWS_PER_XOR_PATTERN == 2 ) {
            smem_read_row = (tidx & WARP_MASK_K) / WARP_DIV_K * Xmma_tile::XMMAS_K * 2 +
                            (tidx & 0x04) / 4;
            smem_read_col = (tidx & 0x03) * 2 + (tidx & 0x04) / 4;
        }
        
        // Each half-warp applies a different XOR pattern -- see the Excel document.
        smem_read_col ^= (Cta_tile::GROUPS > 1 ? 0 : (tidx & WARP_MASK_N) / WARP_DIV_N * 2) +
                         (tidx & 0x08) / 8;

        // The shared memory offset.
        this->smem_read_offset_ = smem_read_row*Base::BYTES_PER_ROW + smem_read_col*BYTES_PER_LDS;
    }

    // Rewind smem_read_offset for last LDS phase in main loop.-
    inline __device__ void reverse_smem_read_offset(int ki = 0) {

        static_assert(Cta_tile::WARPS_M == 4 ||
                      Cta_tile::WARPS_M == 2 ||
                      Xmma_tile::XMMAS_M <= 4, "");

        #pragma unroll
        for( int ni = 0; ni < Xmma_tile::XMMAS_N; ++ni ) {

            // Move the pointer for the next ni. I expect the compiler to not recompute those.
            if( Cta_tile::WARPS_N == 4 ) {
                // Nothing to do!
            } else if( Cta_tile::WARPS_N == 2 && Xmma_tile::XMMAS_N > 1 ) {
                this->smem_read_offset_ ^= BYTES_PER_LDS * (4);
            } else if( Xmma_tile::XMMAS_N == 4 ) {
                this->smem_read_offset_ ^= BYTES_PER_LDS * (ni % 2 == 0 ? 2 : 6);
            } else if( Xmma_tile::XMMAS_N == 3 && ni == 0 ) {
                this->smem_read_offset_ ^= BYTES_PER_LDS * 2;
            } else if( Xmma_tile::XMMAS_N == 3 && ni == 1 ) {
                this->smem_read_offset_ ^= BYTES_PER_LDS * 6;
            } else if( Xmma_tile::XMMAS_N == 3 && ni == 2 ) {
                this->smem_read_offset_ ^= BYTES_PER_LDS * 4;
            } else if( Xmma_tile::XMMAS_N == 2 ) { 
                this->smem_read_offset_ ^= BYTES_PER_LDS * 2;
            }
        }
    }

    // Load from shared memory.
    inline __device__ void load(Fragment (&b)[Xmma_tile::XMMAS_N], int ki) {
        #pragma unroll
        for( int ni = 0; ni < Xmma_tile::XMMAS_N; ++ni ) {
            // Prepare the offset.
            int offset = ki * Base::ROWS_PER_XOR_PATTERN * Base::BYTES_PER_ROW;
            if( Cta_tile::WARPS_N == 1 || Cta_tile::GROUPS > 1 ) {
                offset += this->smem_read_offset_;
            } else if( Cta_tile::WARPS_N == 2 ) {
                offset += this->smem_read_offset_ + (ni/2) * Xmma_tile::N_PER_XMMA_PER_CTA*4;
            } else {
                offset += this->smem_read_offset_ + (ni  ) * Xmma_tile::N_PER_XMMA_PER_CTA*2;
            }

            // Load the data using LDSM.MT88.2.
            uint2 tmp; 
            ldsmt(tmp, this->smem_ + this->smem_read_buffer_ + offset);
            b[ni].reg(0) = tmp.x;
            b[ni].reg(1) = tmp.y;

            static_assert(Cta_tile::WARPS_M == 4 ||
                          Cta_tile::WARPS_M == 2 ||
                          Xmma_tile::XMMAS_M <= 4, "");

            // Move the pointer for the next ni. I expect the compiler to not recompute those.
            if( Cta_tile::WARPS_N == 4 ) {
                // Nothing to do!
            } else if( Cta_tile::WARPS_N == 2 && Xmma_tile::XMMAS_N > 1 ) {
                this->smem_read_offset_ ^= BYTES_PER_LDS * (Cta_tile::GROUPS > 1 ? 2 : 4);
            } else if( Xmma_tile::XMMAS_N == 4 ) {
                this->smem_read_offset_ ^= BYTES_PER_LDS * (ni % 2 == 0 ? 2 : 6);
            } else if( Xmma_tile::XMMAS_N == 3 && ni == 0 ) {
                this->smem_read_offset_ ^= BYTES_PER_LDS * 2;
            } else if( Xmma_tile::XMMAS_N == 3 && ni == 1 ) {
                this->smem_read_offset_ ^= BYTES_PER_LDS * 6;
            } else if( Xmma_tile::XMMAS_N == 3 && ni == 2 ) {
                this->smem_read_offset_ ^= BYTES_PER_LDS * 4;
            } else if( Xmma_tile::XMMAS_N == 2 ) { 
                this->smem_read_offset_ ^= BYTES_PER_LDS * 2;
            }
        }
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< 
    // The instruction traits.
    typename Traits, 
    // The dimensions of the CTA.
    typename Cta_tile, 
    // The size of each STS.
    int BYTES_PER_STS,
    // The number of buffers in the tile. 
    int BUFFERS_PER_TILE 
>
struct Smem_tile_turing_col_interleaved_a : public Smem_tile_interleaved<Cta_tile, 
                                                                         Cta_tile::K, 
                                                                         Cta_tile::M, 
                                                                         Traits::BITS_PER_ELEMENT_A, 
                                                                         BYTES_PER_STS,
                                                                         BUFFERS_PER_TILE> {

    // The base class.
    using Base = Smem_tile_interleaved<Cta_tile, 
                                       Cta_tile::K, 
                                       Cta_tile::M, 
                                       Traits::BITS_PER_ELEMENT_A, 
                                       BYTES_PER_STS,
                                       BUFFERS_PER_TILE>;
    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;
    // The fragment.
    using Fragment = Fragment_a<Traits, Row>;

    // The size of a single LDS in bytes.
    enum { BYTES_PER_LDS = 16 };

    // Ctor.
    inline __device__ Smem_tile_turing_col_interleaved_a(void *smem, int tidx) : Base(smem, tidx) {

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
                            (tidx & 0x0f);

        // The shared memory offset.
        this->smem_read_offset_ = smem_read_row*Base::BYTES_PER_ROW + smem_read_col*BYTES_PER_LDS;
    }

    // Load from shared memory.
    inline __device__ void load(Fragment &a, int mi, int ki) {
        int offset = mi * Xmma_tile::M_PER_XMMA_PER_CTA * BYTES_PER_LDS + ki * Base::BYTES_PER_ROW;
        uint2 tmp;
        ldsm(tmp, this->smem_ + this->smem_read_offset_ + this->smem_read_buffer_ + offset);
        a.reg(0) = tmp.x;
        a.reg(1) = tmp.y;
    }

    // Load from shared memory.
    template< int M >
    inline __device__ void load(Fragment (&a)[M], int ki) {
        #pragma unroll
        for( int mi = 0; mi < M; ++mi ) {
            load(a[mi], mi, ki);
        }
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< 
    // The instruction traits.
    typename Traits, 
    // The dimensions of the CTA.
    typename Cta_tile, 
    // The size of each STS.
    int BYTES_PER_STS,
    // The number of buffers in the tile. 
    int BUFFERS_PER_TILE 
>
struct Smem_tile_turing_row_interleaved_b : public Smem_tile_interleaved<Cta_tile, 
                                                                         Cta_tile::K, 
                                                                         Cta_tile::N, 
                                                                         Traits::BITS_PER_ELEMENT_B, 
                                                                         BYTES_PER_STS,
                                                                         BUFFERS_PER_TILE> {

    // The base class.
    using Base = Smem_tile_interleaved<Cta_tile, 
                                       Cta_tile::K, 
                                       Cta_tile::N, 
                                       Traits::BITS_PER_ELEMENT_B,
                                       BYTES_PER_STS, 
                                       BUFFERS_PER_TILE>;
    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;
    // The fragment.
    using Fragment = Fragment_b<Traits, Col>;

    // The size of a single LDS in bytes.
    enum { BYTES_PER_LDS = 16 };

    // Ctor.
    inline __device__ Smem_tile_turing_row_interleaved_b(void *smem, int tidx) : Base(smem, tidx) {

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
                            (tidx & 0x0f);

        // The shared memory offset.
        this->smem_read_offset_ = smem_read_row*Base::BYTES_PER_ROW + smem_read_col*BYTES_PER_LDS;
    }

    // Load from shared memory.
    inline __device__ void load(Fragment &b, int ni, int ki) {
        int offset = ni * Xmma_tile::N_PER_XMMA_PER_CTA * BYTES_PER_LDS + ki * Base::BYTES_PER_ROW;
        uint2 tmp;
        ldsm(tmp, this->smem_ + this->smem_read_offset_ + this->smem_read_buffer_ + offset);
        b.reg(0) = tmp.x;
        b.reg(1) = tmp.y;
    }

    // Load from shared memory.
    template< int N >
    inline __device__ void load(Fragment (&b)[N], int ki) {
        #pragma unroll
        for( int ni = 0; ni < N; ++ni ) {
            load(b[ni], ni, ki);
        }
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< 
    // The instruction traits.
    typename Traits, 
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile, 
    // The type of storage.
    typename Storage_type, 
    // Fragment before the swizzling.
    typename Fragment_pre_swizzle = Fragment_epilogue_pre_swizzle<Traits, Cta_tile>,
    // Fragment after the swizzling.
    typename Fragment_post_swizzle = Fragment_epilogue_post_swizzle<Traits, Cta_tile> 
>
struct Swizzle_turing_epilogue {

    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;

    // To support arbitrary N, we pad some values to a power-of-2.
    enum { N_WITH_PADDING = Next_power_of_two<Cta_tile::N>::VALUE }; 
    // The number of bytes for key memory instruction.
    enum { BYTES_PER_LDS = 16, BYTES_PER_STG = 16, BYTES_PER_STS = 2*sizeof(Storage_type) };
    // The amount of bytes per element.
    enum { BYTES_PER_ELEMENT = sizeof(Storage_type) };
    // The number of rows in shared memory.
    enum { ROWS = Xmma_tile::M_PER_XMMA_PER_CTA };
    // The size of a single row in bytes.
    enum { BYTES_PER_ROW = N_WITH_PADDING * Cta_tile::WARPS_K * sizeof(Storage_type) };
    // The skew to avoid bank conflicts.
    enum { BYTES_PER_SKEW = BYTES_PER_STS*4 };
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
    inline __device__ Swizzle_turing_epilogue(void *smem, int tidx) 
        : smem_(get_smem_pointer(smem)) {

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
        const int BYTES_PER_TILE_N = N_WITH_PADDING * sizeof(Storage_type);

        // Compute the row and the column in shared memory. Each warp reads from a 16*16B segment.
        const int smem_write_row = (tidx & WARP_MASK_M) / WARP_DIV_M * Xmma_tile::M_PER_XMMA + 
                                   (tidx & 0x1c) / 4;

        // For group fprop/dgrd. C is divided into 2 halves along N dimension.
        // The fist warp stores the first half and the second warp stores the second half.
        const int smem_write_col = (tidx & WARP_MASK_K) / WARP_DIV_K * BYTES_PER_TILE_N +
                                   (tidx & WARP_MASK_N) / WARP_DIV_N * 
                                   (Cta_tile::GROUPS > 1 ? Cta_tile::N / WARPS_N * sizeof(Storage_type) : 
                                   BYTES_PER_STS * Xmma_tile::N_PER_XMMA / 2 ) + (tidx & 0x03) * BYTES_PER_STS;

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

    // Print the content of the tile (only for debug ;)).
    inline __device__ void debug_print() const {
        for( int row = 0; row < ROWS; ++row ) {
            for( int col = 0; col < BYTES_PER_ROW; col += 4 ) {
                if( threadIdx.x == 0 ) {
                    uint32_t val;
                    lds(val, smem_ + row*BYTES_PER_ROW_WITH_SKEW + col);
                    printf("block=(x=%2d, y=%2d, z=%2d) (row=%2d, byte=%4d)=0x%08x\n",
                        blockIdx.x,
                        blockIdx.y,
                        blockIdx.z,
                        row,
                        col,
                        val);
                }
            }
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
struct Smem_tile_a<Turing_hmma_fp16_traits, Cta_tile, Row, BYTES_PER_STS, BUFFERS_PER_TILE> 
    : public Smem_tile_turing_row_a<Turing_hmma_fp16_traits, 
                                    Cta_tile, 
                                    BYTES_PER_STS,
                                    BUFFERS_PER_TILE> {

    // The traits class.
    using Traits = Turing_hmma_fp16_traits;
    // The base class.
    using Base = Smem_tile_turing_row_a<Traits, Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE>;

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
struct Smem_tile_a<Turing_hmma_fp16_traits, Cta_tile, Col, BYTES_PER_STS, BUFFERS_PER_TILE> 
    : public Smem_tile_turing_col_a<Turing_hmma_fp16_traits, 
                                    Cta_tile, 
                                    BYTES_PER_STS,
                                    BUFFERS_PER_TILE> {

    // The traits class.
    using Traits = Turing_hmma_fp16_traits;
    // The associated XMMA tile.
    using Xmma_tile = Traits::template Xmma_tile<Cta_tile>;
    // The base class.
    using Base = Smem_tile_turing_col_a<Traits, Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE>;

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
struct Smem_tile_a<Turing_hmma_fp16_traits, Cta_tile, 
                   Col_interleaved, 
                   BYTES_PER_STS, 
                   BUFFERS_PER_TILE> 
    : public Smem_tile_turing_col_interleaved_a<Turing_hmma_fp16_traits, 
                                                Cta_tile, 
                                                BYTES_PER_STS,
                                                BUFFERS_PER_TILE> {

    // The traits class.
    using Traits = Turing_hmma_fp16_traits;
    // The associated XMMA tile.
    using Xmma_tile = Traits::template Xmma_tile<Cta_tile>;
    // The base class.
    using Base = Smem_tile_turing_col_interleaved_a<Traits, 
                                                    Cta_tile, 
                                                    BYTES_PER_STS, 
                                                    BUFFERS_PER_TILE>;

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
struct Smem_tile_b<Turing_hmma_fp16_traits, Cta_tile, Row, BYTES_PER_STS, BUFFERS_PER_TILE> 
    : public Smem_tile_turing_row_b<Turing_hmma_fp16_traits, 
                                    Cta_tile, 
                                    BYTES_PER_STS,
                                    BUFFERS_PER_TILE> {

    // The traits class.
    using Traits = Turing_hmma_fp16_traits;
    // The associated XMMA tile.
    using Xmma_tile = Traits::template Xmma_tile<Cta_tile>;
    // The base class.
    using Base = Smem_tile_turing_row_b<Traits, Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE>;

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
struct Smem_tile_b<Turing_hmma_fp16_traits, Cta_tile, Col, BYTES_PER_STS, BUFFERS_PER_TILE> 
    : public Smem_tile_turing_col_b<Turing_hmma_fp16_traits, 
                                    Cta_tile, 
                                    BYTES_PER_STS, 
                                    BUFFERS_PER_TILE> {

    // The traits class.
    using Traits = Turing_hmma_fp16_traits;
    // The base class.
    using Base = Smem_tile_turing_col_b<Traits, Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE>;

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
struct Smem_tile_b<Turing_hmma_fp16_traits, 
                   Cta_tile, 
                   Row_interleaved, 
                   BYTES_PER_STS, 
                   BUFFERS_PER_TILE> 
    : public Smem_tile_turing_row_interleaved_b<Turing_hmma_fp16_traits, 
                                                Cta_tile, 
                                                BYTES_PER_STS,
                                                BUFFERS_PER_TILE> {

    // The traits class.
    using Traits = Turing_hmma_fp16_traits;
    // The associated XMMA tile.
    using Xmma_tile = Traits::template Xmma_tile<Cta_tile>;
    // The base class.
    using Base = Smem_tile_turing_row_interleaved_b<Traits, 
                                                    Cta_tile, 
                                                    BYTES_PER_STS, 
                                                    BUFFERS_PER_TILE>;

    // Ctor.
    inline __device__ Smem_tile_b(void *smem, int tidx) : Base(smem, tidx) {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Traits, typename Cta_tile >
struct Swizzle_turing_hmma_fp16_epilogue 
    : public Swizzle_turing_epilogue<Traits, Cta_tile, lwtlass::half_t> {

    // The base class.
    using Base = Swizzle_turing_epilogue<Traits, Cta_tile, lwtlass::half_t>;
    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;

    // Ctor.
    inline __device__ Swizzle_turing_hmma_fp16_epilogue(void *smem, int tidx) : Base(smem, tidx) {
    }

    // Load from the tile in shared memory.
    template<typename Fragment_post_swizzle>
    inline __device__ void load(Fragment_post_swizzle &dst, int oi) const {
        #pragma unroll
        for( int ki = 0; ki < Cta_tile::WARPS_K; ++ki ) {
            const int offset = oi * Base::PIXELS_PER_STG * Base::BYTES_PER_ROW_WITH_SKEW + 
                               ki * Base::N_WITH_PADDING * sizeof(lwtlass::half_t);
            uint4 tmp;
            lds(tmp, this->smem_ + this->smem_read_offset_ + offset);
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
        for( int mi = 0; mi < Base::M_PER_XMMA_PER_THREAD; ++mi ) {
            int offset = mi * Base::THREADS_PER_XMMA_M * Base::BYTES_PER_ROW_WITH_SKEW +
                         ni * (Cta_tile::GROUPS > 1 ? Xmma_tile::N_PER_XMMA : 
                         Xmma_tile::N_PER_XMMA_PER_CTA) * sizeof(lwtlass::half_t);

            uint32_t ptr = this->smem_ + this->smem_write_offset_ + offset; 
            sts(ptr +  0, c.reg(2*mi + 0));
            sts(ptr + 16, c.reg(2*mi + 1));
        }
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< 
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile, 
    // Do we enable split-k?
    bool IN_CTA_SPLIT_K
>
struct Swizzle_epilogue<Turing_hmma_fp16_traits, Cta_tile, Row, 16, IN_CTA_SPLIT_K> 
    : public Swizzle_turing_hmma_fp16_epilogue<Turing_hmma_fp16_traits, Cta_tile> {

    // The traits.
    using Traits = Turing_hmma_fp16_traits;
    // The base class.
    using Base = Swizzle_turing_hmma_fp16_epilogue<Traits, Cta_tile>;

    // Ctor.
    inline __device__ Swizzle_epilogue(void *smem, int tidx) : Base(smem, tidx) {
    }
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
struct Smem_tile_a<Turing_hmma_fp32_traits, Cta_tile, Row, BYTES_PER_STS, BUFFERS_PER_TILE>
    : public Smem_tile_turing_row_a<Turing_hmma_fp32_traits, 
                                    Cta_tile, 
                                    BYTES_PER_STS,
                                    BUFFERS_PER_TILE> {

    // The traits class.
    using Traits = Turing_hmma_fp32_traits;
    // The base class.
    using Base = Smem_tile_turing_row_a<Traits, Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE>;

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
struct Smem_tile_a<Turing_hmma_fp32_traits, Cta_tile, Col, BYTES_PER_STS, BUFFERS_PER_TILE> 
    : public Smem_tile_turing_col_a<Turing_hmma_fp32_traits, 
                                    Cta_tile, 
                                    BYTES_PER_STS,
                                    BUFFERS_PER_TILE> {

    // The traits class.
    using Traits = Turing_hmma_fp32_traits;
    // The base class.
    using Base = Smem_tile_turing_col_a<Traits, Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE>;

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
struct Smem_tile_b<Turing_hmma_fp32_traits, Cta_tile, Col, BYTES_PER_STS, BUFFERS_PER_TILE>
    : public Smem_tile_turing_col_b<Turing_hmma_fp32_traits, 
                                    Cta_tile, 
                                    BYTES_PER_STS,
                                    BUFFERS_PER_TILE> {

    // The traits class.
    using Traits = Turing_hmma_fp32_traits;
    // The base class.
    using Base = Smem_tile_turing_col_b<Traits, Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE>;

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
struct Smem_tile_b<Turing_hmma_fp32_traits, Cta_tile, Row, BYTES_PER_STS, BUFFERS_PER_TILE>
    : public Smem_tile_turing_row_b<Turing_hmma_fp32_traits, 
                                    Cta_tile, 
                                    BYTES_PER_STS,
                                    BUFFERS_PER_TILE> {

    // The traits class.
    using Traits = Turing_hmma_fp32_traits;
    // The base class.
    using Base = Smem_tile_turing_row_b<Traits, Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE>;

    // Ctor.
    inline __device__ Smem_tile_b(void *smem, int tidx) : Base(smem, tidx) {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Traits, typename Cta_tile, typename Layout>
struct Swizzle_turing_hmma_fp32_epilogue {

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
    inline __device__ Swizzle_turing_hmma_fp32_epilogue(void *smem, int tidx) 
        : smem_(get_smem_pointer(smem)) {

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

        // For group fprop/dgrd. C is divided into 2 halves along N dimension.
        // The fist warp stores the first half and the second warp stores the second half.
        const int smem_write_col = (tidx & WARP_MASK_N) / WARP_DIV_N *
                                   (Cta_tile::GROUPS > 1 ? Cta_tile::N / WARPS_N / 2 * 4
                                   : Xmma_tile::N_PER_XMMA / 2 * 4) +
                                   (tidx & WARP_MASK_K) / WARP_DIV_K * N_WITH_PADDING / 2 * 4 + 
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
            int offset_0 = offset + ki * N_WITH_PADDING * 2;

            // The 1st group of 4 floats.
            uint4 tmp_0;
            lds(tmp_0, smem_ + smem_read_offset_ + offset_0);
            dst.reg(ki*8 + 0) = tmp_0.x;
            dst.reg(ki*8 + 1) = tmp_0.y;
            dst.reg(ki*8 + 2) = tmp_0.z;
            dst.reg(ki*8 + 3) = tmp_0.w;

            // The 2nd group of 4 floats.
            int offset_1 = offset_0 + 1*BYTES_PER_ROW_WITH_SKEW;
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
                         ni * (Cta_tile::GROUPS > 1 ? Xmma_tile::N_PER_XMMA / 2 :
                         Xmma_tile::N_PER_XMMA_PER_CTA / 2) * sizeof(float);

            uint32_t ptr = smem_ + smem_write_offset_ + offset;
            sts(ptr +  0, make_uint2(c.reg(4*mi+0), c.reg(4*mi+1)));
            sts(ptr + 16, make_uint2(c.reg(4*mi+2), c.reg(4*mi+3)));
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

template< typename Cta_tile, bool IN_CTA_SPLIT_K >
struct Swizzle_epilogue<Turing_hmma_fp32_traits, Cta_tile, Row, 16, IN_CTA_SPLIT_K>
    : public Swizzle_turing_hmma_fp32_epilogue<Turing_hmma_fp32_traits, 
                                               Cta_tile, 
                                               Row> {
    // The traits class.
    using Traits = Turing_hmma_fp32_traits;
    // The base class.
    using Base = Swizzle_turing_hmma_fp32_epilogue<Turing_hmma_fp32_traits, 
                                                   Cta_tile, 
                                                   Row>;

    // Ctor.
    inline __device__ Swizzle_epilogue(void *smem, int tidx) : Base(smem, tidx) {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    // The instruction traits.
    typename Traits,
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile
>
struct Swizzle_turing_imma_int32_epilogue 
    : public Swizzle_turing_epilogue<Traits, Cta_tile, int32_t> {

    // The base class.
    using Base = Swizzle_turing_epilogue<Traits, Cta_tile, int32_t>;
    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;

    // Ctor.
    inline __device__ Swizzle_turing_imma_int32_epilogue(void *smem, int tidx) 
        : Base(smem, tidx) {
        const int elem_per_thd = Cta_tile::N / Base::THREADS_PER_PIXEL;

        // Decompose into groups of size "THREADS_PER_PIXEL".
        const int tidx_div_tpp = tidx / Base::THREADS_PER_PIXEL;
        const int tidx_mod_tpp = tidx % Base::THREADS_PER_PIXEL;

        // The row and column read by a single thread.
        const int smem_read_row = tidx_div_tpp;
        const int smem_read_col = tidx_mod_tpp * elem_per_thd * sizeof(float);

        smem_read_offset_ = smem_read_row * Base::BYTES_PER_ROW_WITH_SKEW +
                            smem_read_col; 
    }

    // Load from the tile in shared memory.
    template<typename Fragment_post_swizzle>
    inline __device__ void load(Fragment_post_swizzle &dst, int oi) const {
        #pragma unroll
        for( int ki = 0; ki < Fragment_post_swizzle::NUM_REGS * sizeof(float) /
            Base::BYTES_PER_LDS; ++ki ) {
            int offset = oi * Base::PIXELS_PER_STG * Base::BYTES_PER_ROW_WITH_SKEW + 
                         ki * Base::BYTES_PER_LDS;

            uint4 tmp;
            lds(tmp, this->smem_ + this->smem_read_offset_ + offset);
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
        for( int mi = 0; mi < Base::M_PER_XMMA_PER_THREAD; ++mi ) {
            int offset = mi * Base::THREADS_PER_XMMA_M * Base::BYTES_PER_ROW_WITH_SKEW +
                         ni * Xmma_tile::N_PER_XMMA_PER_CTA * sizeof(int32_t);

            uint32_t ptr = this->smem_ + this->smem_write_offset_ + offset; 
            sts(ptr +  0, make_uint2(c.reg(4*mi + 0), c.reg(4*mi + 1)));
            sts(ptr + 32, make_uint2(c.reg(4*mi + 2), c.reg(4*mi + 3)));
        }
    }

    // The read offset.
    int smem_read_offset_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// I M M A . 8
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
struct Smem_tile_a<Turing_imma_int8_int32_traits<IS_GELU_ERF>, Cta_tile, Row, BYTES_PER_STS, BUFFERS_PER_TILE> 
    : public Smem_tile_turing_row_a<Turing_imma_int8_int32_traits<IS_GELU_ERF>, 
                                    Cta_tile, 
                                    BYTES_PER_STS,
                                    BUFFERS_PER_TILE> {

    // The traits class.
    using Traits = Turing_imma_int8_int32_traits<IS_GELU_ERF>;
    // The base class.
    using Base = Smem_tile_turing_row_a<Turing_imma_int8_int32_traits<IS_GELU_ERF>, 
                                        Cta_tile, 
                                        BYTES_PER_STS,
                                        BUFFERS_PER_TILE>;
    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;

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
    int BUFFERS_PER_TILE,
    // Support gelu_erf or not
    bool IS_GELU_ERF
>
struct Smem_tile_b<Turing_imma_int8_int32_traits<IS_GELU_ERF>, Cta_tile, Col, BYTES_PER_STS, BUFFERS_PER_TILE>
    : public Smem_tile_turing_col_b<Turing_imma_int8_int32_traits<IS_GELU_ERF>, 
                                    Cta_tile, 
                                    BYTES_PER_STS,
                                    BUFFERS_PER_TILE> {

    // The traits class.
    using Traits = Turing_imma_int8_int32_traits<IS_GELU_ERF>;
    // The base class.
    using Base = Smem_tile_turing_col_b<Turing_imma_int8_int32_traits<IS_GELU_ERF>, 
                                        Cta_tile, 
                                        BYTES_PER_STS,
                                        BUFFERS_PER_TILE>;
    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;

    // Ctor.
    inline __device__ Smem_tile_b(void *smem, int tidx) : Base(smem, tidx) {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< 
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile, 
    // Do we enable split-k?
    bool IN_CTA_SPLIT_K,
    // Support gelu_erf or not
    bool IS_GELU_ERF
>
struct Swizzle_epilogue<Turing_imma_int8_int32_traits<IS_GELU_ERF>, 
                          Cta_tile, 
                          Row, 
                          16,
                          IN_CTA_SPLIT_K> 
    : public Swizzle_turing_imma_int32_epilogue<Turing_imma_int8_int32_traits<IS_GELU_ERF>, Cta_tile> {

    // The traits.
    using Traits = Turing_imma_int8_int32_traits<IS_GELU_ERF>;
    // The base class.
    using Base = Swizzle_turing_imma_int32_epilogue<Traits, Cta_tile>;
    // Ctor.
    inline __device__ Swizzle_epilogue(void *smem, int tidx) : Base(smem, tidx) {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// I M M A . 4
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
struct Smem_tile_a<Turing_imma_int4_int32_traits, Cta_tile, Row, BYTES_PER_STS, BUFFERS_PER_TILE> 
    : public Smem_tile_turing_row_a<Turing_imma_int4_int32_traits, 
                                    Cta_tile, 
                                    BYTES_PER_STS,
                                    BUFFERS_PER_TILE> {

    // The traits class.
    using Traits = Turing_imma_int4_int32_traits;
    // The base class.
    using Base = Smem_tile_turing_row_a<Turing_imma_int4_int32_traits, 
                                        Cta_tile, 
                                        BYTES_PER_STS,
                                        BUFFERS_PER_TILE>;
    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;

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
struct Smem_tile_b<Turing_imma_int4_int32_traits, Cta_tile, Col, BYTES_PER_STS, BUFFERS_PER_TILE>
    : public Smem_tile_turing_col_b<Turing_imma_int4_int32_traits, 
                                    Cta_tile, 
                                    BYTES_PER_STS,
                                    BUFFERS_PER_TILE> {

    // The traits class.
    using Traits = Turing_imma_int4_int32_traits;
    // The base class.
    using Base = Smem_tile_turing_col_b<Turing_imma_int4_int32_traits, 
                                        Cta_tile, 
                                        BYTES_PER_STS,
                                        BUFFERS_PER_TILE>;
    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;

    // Ctor.
    inline __device__ Smem_tile_b(void *smem, int tidx) : Base(smem, tidx) {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // Do we enable split-k?
    bool IN_CTA_SPLIT_K
>
struct Swizzle_epilogue<Turing_imma_int4_int32_traits,
                          Cta_tile,
                          Row,
                          16,
                          IN_CTA_SPLIT_K> 
    : public Swizzle_turing_imma_int32_epilogue<Turing_imma_int4_int32_traits, Cta_tile> {
    // The traits.
    using Traits = Turing_imma_int4_int32_traits;
    // The base class.
    using Base = Swizzle_turing_imma_int32_epilogue<Turing_imma_int4_int32_traits, Cta_tile>;
    // Ctor.
    inline __device__ Swizzle_epilogue(void *smem, int tidx) : Base(smem, tidx) {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// B M M A 
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
struct Smem_tile_a<Turing_bmma_int32_traits, Cta_tile, Row, BYTES_PER_STS, BUFFERS_PER_TILE> 
    : public Smem_tile_turing_row_a<Turing_bmma_int32_traits, 
                                    Cta_tile, 
                                    BYTES_PER_STS,
                                    BUFFERS_PER_TILE> {

    // The traits class.
    using Traits = Turing_bmma_int32_traits;
    // The base class.
    using Base = Smem_tile_turing_row_a<Turing_bmma_int32_traits, 
                                        Cta_tile, 
                                        BYTES_PER_STS,
                                        BUFFERS_PER_TILE>;
    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;

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
struct Smem_tile_b<Turing_bmma_int32_traits, Cta_tile, Col, BYTES_PER_STS, BUFFERS_PER_TILE>
    : public Smem_tile_turing_col_b<Turing_bmma_int32_traits, 
                                    Cta_tile, 
                                    BYTES_PER_STS,
                                    BUFFERS_PER_TILE> {

    // The traits class.
    using Traits = Turing_bmma_int32_traits;
    // The base class.
    using Base = Smem_tile_turing_col_b<Turing_bmma_int32_traits, 
                                        Cta_tile, 
                                        BYTES_PER_STS,
                                        BUFFERS_PER_TILE>;
    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;

    // Ctor.
    inline __device__ Smem_tile_b(void *smem, int tidx) : Base(smem, tidx) {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // Do we enable split-k?
    bool IN_CTA_SPLIT_K
>
struct Swizzle_epilogue<Turing_bmma_int32_traits, 
                          Cta_tile, 
                          Row, 
                          16,
                          IN_CTA_SPLIT_K> 
    : public Swizzle_turing_imma_int32_epilogue<Turing_bmma_int32_traits, Cta_tile> {
    // The traits.
    using Traits = Turing_bmma_int32_traits;
    // The base class.
    using Base = Swizzle_turing_imma_int32_epilogue<Turing_bmma_int32_traits, Cta_tile>;
    // Ctor.
    inline __device__ Swizzle_epilogue(void *smem, int tidx) : Base(smem, tidx) {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// I M M A  I N T E R L E A V E (NC/32HW32)
//
////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Traits, typename Cta_tile, int BITS_PER_ELEMENT, int BUFFERS_PER_TILE >
struct Smem_tile_turing_imma_col_interleaved_a : public Smem_tile_interleaved<Cta_tile, 
                                                                              Cta_tile::K, 
                                                                              Cta_tile::M, 
                                                                              BITS_PER_ELEMENT, 
                                                                              16,
                                                                              BUFFERS_PER_TILE> {

    // The base class.
    using Base = Smem_tile_interleaved<Cta_tile, 
                                       Cta_tile::K, 
                                       Cta_tile::M, 
                                       BITS_PER_ELEMENT, 
                                       16,
                                       BUFFERS_PER_TILE>;
    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;
    // The fragment.
    using Fragment = Fragment_a<Traits, Row>;

    // The size of a single LDS in bytes.
    enum { BYTES_PER_LDS = 8, BYTES_PER_STS = 16 };
    // The number of elements that are stored with a single STS.
    enum { ELEMENTS_PER_STS = BYTES_PER_STS * 8 / BITS_PER_ELEMENT };
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
    // Bytes per smem cache line.
    enum { BYTES_PER_SMEM_LINE = 128 };
    // Byers in skew.
    enum { BYTES_PER_SKEW = 0 * 32 };
    // Skew.
    enum { SKEW = 0 };
    // Skew per row.
    enum { SKEW_PER_ROW = (Cta_tile::M * BYTES_PER_PACKET / BYTES_PER_SMEM_LINE) * SKEW * BYTES_PER_SKEW };
    // The size in bytes in shared memory.
    enum { BYTES_PER_TILE = ROWS * (BYTES_PER_ROW + SKEW_PER_ROW) };
    // Smem stride between two xmma
    enum { LDS_STRIDE_PER_XMMA = Cta_tile::WARPS_M * Xmma_tile::M_PER_XMMA *
           BYTES_PER_PACKET / BYTES_PER_SMEM_LINE * 
           (BYTES_PER_SMEM_LINE + BYTES_PER_SKEW) };
    enum { LDS_STRIDE_IN_XMMA = 2 * (BYTES_PER_SMEM_LINE + BYTES_PER_SKEW) };

    // Ctor.
    inline __device__ Smem_tile_turing_imma_col_interleaved_a(void *smem, int tidx) 
        : Base(smem, tidx) {

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
        int thrd_per_line = BYTES_PER_SMEM_LINE / BYTES_PER_LDS;
        int lines_per_warp = Xmma_tile::M_PER_XMMA * BYTES_PER_PACKET / 
                             BYTES_PER_SMEM_LINE;
        int line_idx = lines_per_warp * (tidx & WARP_MASK_M) / WARP_DIV_M +
            (tidx & 0x1f) / thrd_per_line;
        
        this->smem_read_offset_ = smem_read_row*BYTES_PER_ROW +
                                  line_idx * (BYTES_PER_SMEM_LINE + BYTES_PER_SKEW) +
                                  (tidx & (thrd_per_line-1)) * BYTES_PER_LDS;

        // The row/col written by the thread.
        int smem_write_row = (tidx / THREADS_PER_ROW);
        int smem_write_col = (tidx % THREADS_PER_ROW);

        // The location where the thread writes its elements.
        this->smem_write_offset_ = smem_write_row*BYTES_PER_ROW + 
                                   smem_write_col*BYTES_PER_STS;
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
                             ki*(BYTES_PER_ROW + SKEW_PER_ROW) +
                             mi * LDS_STRIDE_PER_XMMA +
                             i * LDS_STRIDE_IN_XMMA;

                uint2 tmp;
                lds(tmp, this->smem_ + this->smem_read_buffer_ + offset);
                a[mi].reg(i*2  ) = tmp.x;
                a[mi].reg(i*2+1) = tmp.y;
            }// end i
        }// end mi
    }

    // Print the content of the tile (only for debug ;)).
    inline __device__ void debug_print() const {
        for( int row = 0; row < ROWS; ++row ) {
            for( int col = 0; col < BYTES_PER_ROW; col += 4 ) {
                if( threadIdx.x == 0 ) {
                    uint32_t val;
                    lds(val, this->smem_ + row*BYTES_PER_ROW + col);
                    printf("img_block=(x=%2d, y=%2d, z=%2d) (row=%2d, byte=%4d)=0x%08x\n",
                        blockIdx.x,
                        blockIdx.y,
                        blockIdx.z,
                        row,
                        col,
                        val);
                }
            }
        }
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

template< typename Traits, typename Cta_tile, int BITS_PER_ELEMENT, int BUFFERS_PER_TILE >
struct Smem_tile_turing_imma_row_interleaved_b : public Smem_tile_interleaved<Cta_tile, 
                                                                              Cta_tile::K, 
                                                                              Cta_tile::N, 
                                                                              BITS_PER_ELEMENT, 
                                                                              16,
                                                                              BUFFERS_PER_TILE> {

    // The base class.
    using Base = Smem_tile_interleaved<Cta_tile, 
                                       Cta_tile::K, 
                                       Cta_tile::N, 
                                       BITS_PER_ELEMENT, 
                                       16,
                                       BUFFERS_PER_TILE>;
    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;
    // The fragment.
    using Fragment = Fragment_b<Traits, Col>;

    // The size of a single LDS in bytes.
    enum { BYTES_PER_LDS = 8, BYTES_PER_STS=16 };
    // The number of elements that are stored with a single STS.
    enum { ELEMENTS_PER_STS = BYTES_PER_STS * 8 / BITS_PER_ELEMENT };
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
    // Smem stride between two xmma
    enum { LDSM_STRIDE_PER_XMMA = Cta_tile::GROUPS > 1 ? Xmma_tile::N_PER_XMMA * BYTES_PER_PACKET 
        : Xmma_tile::N_PER_XMMA_PER_CTA * BYTES_PER_PACKET };

    // STS_PER_THREAD
    enum { STS_PER_THREAD = BYTES_PER_ROW / THREADS_PER_ROW / BYTES_PER_STS};
    // Ctor.
    inline __device__ Smem_tile_turing_imma_row_interleaved_b(void *smem, int tidx) 
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
        int smem_read_row = (tidx & WARP_MASK_K) / WARP_DIV_K * ROWS / WARPS_K;
        int smem_read_col = (tidx & WARP_MASK_N) / WARP_DIV_N * 
                            (Cta_tile::GROUPS > 1 ? Cta_tile::N / WARPS_N : Xmma_tile::N_PER_XMMA) *
                            (BYTES_PER_STS / BYTES_PER_LDS) +
                            (tidx & 0x1f) / 2;

        int smem_read_col_inner_id  = tidx % 2;
        // The shared memory offset.
        this->smem_read_offset_ = smem_read_row * BYTES_PER_ROW +
                                  smem_read_col * BYTES_PER_STS +
                                  smem_read_col_inner_id*BYTES_PER_LDS;

        // The row/col written by the thread.
        int smem_write_row = (tidx / THREADS_PER_ROW);
        int smem_write_col = (tidx % THREADS_PER_ROW);

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
                             ni*LDSM_STRIDE_PER_XMMA + 
                             ki*BYTES_PER_ROW + 
                             j*Cta_tile::THREADS_PER_WARP*BYTES_PER_LDS;

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

    // Print the content of the tile (only for debug ;)).
    inline __device__ void debug_print() const {
        for( int row = 0; row < ROWS; ++row ) {
            for( int col = 0; col < BYTES_PER_ROW; col += 4 ) {
                if( threadIdx.x == 0 ) {
                    uint32_t val;
                    lds(val, this->smem_ + row*BYTES_PER_ROW + col);
                    printf("flt_block=(x=%2d, y=%2d, z=%2d) (row=%2d, byte=%4d)=0x%08x\n",
                        blockIdx.x,
                        blockIdx.y,
                        blockIdx.z,
                        row,
                        col,
                        val);
                }
            }
        }
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
    typename Cta_tile, 
    // The number of buffers per tile.
    int BUFFERS_PER_TILE
>
struct Smem_tile_a<Turing_imma_interleaved_int8_int32_traits, 
                   Cta_tile, 
                   Col_interleaved, 
                   16,
                   BUFFERS_PER_TILE> 
    : public Smem_tile_turing_imma_col_interleaved_a<Turing_imma_interleaved_int8_int32_traits, 
                                                     Cta_tile, 
                                                     8, 
                                                     BUFFERS_PER_TILE> {

    // The traits class.
    using Traits = Turing_imma_interleaved_int8_int32_traits;
    // The base class.
    using Base = Smem_tile_turing_imma_col_interleaved_a<Traits, Cta_tile, 8, BUFFERS_PER_TILE>;

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
struct Smem_tile_b<Turing_imma_interleaved_int8_int32_traits, 
                   Cta_tile, 
                   Row_interleaved, 
                   16,
                   BUFFERS_PER_TILE>
    : public Smem_tile_turing_imma_row_interleaved_b<Turing_imma_interleaved_int8_int32_traits, 
                                                     Cta_tile, 
                                                     8, 
                                                     BUFFERS_PER_TILE> {

    // The traits class.
    using Traits = Turing_imma_interleaved_int8_int32_traits;
    // The base class.
    using Base = Smem_tile_turing_imma_row_interleaved_b<Traits, Cta_tile, 8, BUFFERS_PER_TILE>;

    // Ctor.
    inline __device__ Smem_tile_b(void *smem, int tidx) : Base(smem, tidx) {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile
>
struct Swizzle_epilogue_interleaved<Turing_imma_interleaved_int8_int32_traits,
                                      Cta_tile,
                                      Col_interleaved,
                                      true> {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile
>
struct Swizzle_epilogue_interleaved<Turing_imma_interleaved_int8_int32_traits,
                                      Cta_tile,
                                      Col_interleaved,
                                      false> {

    // The traits.
    using Traits = Turing_imma_interleaved_int8_int32_traits;
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

template<
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The pixels.
    typename Pixel_tile
>
struct Swizzle_epilogue_interleaved_1x1_with_3x3<Turing_imma_interleaved_int8_int32_traits,
                                                   Cta_tile,
                                                   Pixel_tile,
                                                   Col_interleaved,
                                                   true> {

    inline __device__ void debug_print() const {
        // Not implemented
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// Add reorder for B matrix.
template<
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The pixels.
    typename Pixel_tile
>
struct Swizzle_epilogue_interleaved_1x1_with_3x3<Turing_imma_interleaved_int8_int32_traits,
                                                   Cta_tile,
                                                   Pixel_tile,
                                                   Col_interleaved,
                                                   false> {

    // The traits.
    using Traits = Turing_imma_interleaved_int8_int32_traits;
    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;

    enum { NUM_REGS = 16 };
    // How many threads work on the nhw dimension.
    enum { BYTES_PER_PACKET = 32 };
    // Bytes per lds, sts.
    enum { BYTES_PER_LDS = 16, BYTES_PER_STS = 8 };
    // How many threads Per Packet (compute with ldg, which is wider).
    enum { THREADS_PER_PACKET = BYTES_PER_PACKET / 16 };
    // Round factor.
    enum { ROUND_FACTOR = 
        Cta_tile::THREADS_PER_CTA / (Cta_tile::K / BYTES_PER_PACKET) / THREADS_PER_PACKET };
    // The size in bytes in shared memory.
    enum { BYTES_PER_TILE =
           ((Cta_tile::M + ROUND_FACTOR-1) / ROUND_FACTOR) * ROUND_FACTOR * Cta_tile::N };
    // Do we skip the syncthreads in the epilogue? Of course, not :)
    enum { SKIP_SYNCTHREADS = 1 };

    // Ctor.
    inline __device__ Swizzle_epilogue_interleaved_1x1_with_3x3(void *smem, int tidx) {
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

