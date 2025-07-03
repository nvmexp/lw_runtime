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
#include <xmma/smem_tile.h>
#include <xmma/turing/smem_tile.h>

namespace xmma {

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// x M M A
//
////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Traits, int N >
struct Rows_per_xor_pattern_ampere_a {
    // The size in bits.
    enum { N_IN_BITS = N * Traits::BITS_PER_ELEMENT_A };
    // The number of rows.
    enum { VALUE = N_IN_BITS <= 256 ? 2 : (N_IN_BITS <= 512 ? 4 : 8) };
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Traits >
struct Cols_per_xor_pattern_ampere {
    enum { VALUE = 1 };
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Input_type, typename Output_type >
struct Cols_per_xor_pattern_ampere<Ampere_hmma_tf32_traits<Input_type, Output_type>> {
    enum { VALUE = 2 };
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Traits, int N >
struct Rows_per_xor_pattern_ampere_col_a : public Rows_per_xor_pattern_ampere_a<Traits, N> {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Input_type, typename Output_type, int N >
struct Rows_per_xor_pattern_ampere_col_a<Ampere_hmma_tf32_traits<Input_type, Output_type>, N> {
    enum { VALUE = 4 };
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
    int ROWS_PER_XOR_PATTERN_ = Rows_per_xor_pattern_ampere_col_a<Traits, Cta_tile::M>::VALUE,
    // How many cols to use for the XOR pattern to avoid bank conflicts?
    int COLS_PER_XOR_PATTERN_ = Cols_per_xor_pattern_ampere<Traits>::VALUE
>
struct Smem_tile_ampere_col_a : public Smem_tile_without_skews<Cta_tile,
                                                               Cta_tile::K,
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
                                         Cta_tile::K,
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
    inline __device__ Smem_tile_ampere_col_a(void *smem, int tidx) : Base(smem, tidx) {

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

        static_assert((USE_LDSMT && Base::ROWS_PER_XOR_PATTERN == 8) ||
                      Base::ROWS_PER_XOR_PATTERN == 4 ||
                      Base::ROWS_PER_XOR_PATTERN == 2, "");

        if( USE_LDSMT && Base::ROWS_PER_XOR_PATTERN == 8 ) {
            smem_read_row = (tidx & WARP_MASK_K) / WARP_DIV_K * Xmma_tile::XMMAS_K * 16 +
                            (tidx & 0x10) / 2 +
                            (tidx & 0x07);
            smem_read_col = (tidx & 0x07);
        } else if( USE_LDSMT && Base::ROWS_PER_XOR_PATTERN == 4 ) {
            smem_read_row = (tidx & WARP_MASK_K) / WARP_DIV_K * Xmma_tile::XMMAS_K * 8 +
                            (tidx & 0x10) / 4 +
                            (tidx & 0x06) / 2;
            smem_read_col = (tidx & 0x01) * 4 + (tidx & 0x06) / 2;
        } else if( USE_LDSMT && Base::ROWS_PER_XOR_PATTERN == 2 ) {
            smem_read_row = (tidx & WARP_MASK_K) / WARP_DIV_K * Xmma_tile::XMMAS_K * 4 +
                            (tidx & 0x10) / 8 +
                            (tidx & 0x04) / 4;
            smem_read_col = (tidx & 0x03) * 2 + (tidx & 0x04) / 4;
        } else if( Base::ROWS_PER_XOR_PATTERN == 4 && Base::COLS_PER_XOR_PATTERN == 2 ) {
            smem_read_row = (tidx & WARP_MASK_K) / WARP_DIV_K * Xmma_tile::XMMAS_K * 8 +
                            (tidx & 0x03);
            smem_read_col = (tidx & 0x1c) / 4 + (tidx & 0x03) * 8;
        }

        // Swizzle the column for other warps.
        if( USE_LDSMT ) {
            smem_read_col ^= (tidx & WARP_MASK_M) / WARP_DIV_M *  2 + (tidx & 0x08) / 8;
        } else {
            smem_read_col ^= (tidx & WARP_MASK_M) / WARP_DIV_M * 16;
        }

        // The shared memory offset.
        this->smem_read_offset_ = smem_read_row*Base::BYTES_PER_ROW + smem_read_col*BYTES_PER_LDS;
    }

    // Rewind smem_read_offset for last LDS phase in main loop.
    inline __device__ void reverse_smem_read_offset(int ki = 0) {
        // The size of each element in bits.
        const int BITS_PER_ELT = Traits::BITS_PER_ELEMENT_A;
        // The size in bytes of the data needed to compute an XMMA per CTA.
        const int BYTES_PER_XMMA_PER_CTA = Xmma_tile::M_PER_XMMA_PER_CTA * BITS_PER_ELT / 8;

        #pragma unroll
        for( int mi = 0; mi < Xmma_tile::XMMAS_M; ++mi ) {
            // Undo the pointer increment for the next ni.
            // Should match the load function below for ki = 0.
            if( BYTES_PER_XMMA_PER_CTA >= 128 ) {
                // Nothing to do!
            } else if( BYTES_PER_XMMA_PER_CTA == 64 && Xmma_tile::XMMAS_M > 1 ) {
                this->smem_read_offset_ ^= BYTES_PER_XMMA_PER_CTA;
            } else if( BYTES_PER_XMMA_PER_CTA == 64 ) {
                // Nothing to do!
            } else if( BYTES_PER_XMMA_PER_CTA == 32 && Xmma_tile::XMMAS_M == 4 ) {
                this->smem_read_offset_ ^= BYTES_PER_LDS * (mi % 2 == 0 ? 2 : 6);
            } else if( BYTES_PER_XMMA_PER_CTA == 32 && Xmma_tile::XMMAS_M == 2 ) {
                this->smem_read_offset_ ^= BYTES_PER_LDS * 2;
            }
        }
    }

    // Load from shared memory.
    inline __device__ void load(Fragment (&a)[Xmma_tile::XMMAS_M], int ki) {
        // The size of each element in bits.
        const int BITS_PER_ELT = Traits::BITS_PER_ELEMENT_A;
        // The size in bytes of the data needed to compute an XMMA per CTA.
        const int BYTES_PER_XMMA_PER_CTA = Xmma_tile::M_PER_XMMA_PER_CTA * BITS_PER_ELT / 8;

        // Perform the different loads.
        #pragma unroll
        for( int mi = 0; mi < Xmma_tile::XMMAS_M; ++mi ) {
            // Prepare the offset.
            int offset = ki * Base::ROWS_PER_XOR_PATTERN * 2 * Base::BYTES_PER_ROW;
            if( BYTES_PER_XMMA_PER_CTA == 32 ) {
                offset += this->smem_read_offset_;
            } else if( BYTES_PER_XMMA_PER_CTA == 64 ) {
                offset += this->smem_read_offset_ + (mi/2) * BYTES_PER_XMMA_PER_CTA * 2;
            } else if( BYTES_PER_XMMA_PER_CTA >= 128 ) {
                offset += this->smem_read_offset_ + (mi  ) * BYTES_PER_XMMA_PER_CTA;
            } else {
                assert(false);
            }

            // Load the data using LDSM.MT88.4 or 4x LDS.32.
            uint32_t ptr = this->smem_ + this->smem_read_buffer_ + offset;
            uint4 tmp;
            if( USE_LDSMT ) {
                ldsmt(tmp, ptr);
            } else {
                lds(tmp.x, (ptr     ) + 0*Base::BYTES_PER_ROW);
                lds(tmp.y, (ptr ^ 32) + 0*Base::BYTES_PER_ROW);
                lds(tmp.z, (ptr     ) + 4*Base::BYTES_PER_ROW);
                lds(tmp.w, (ptr ^ 32) + 4*Base::BYTES_PER_ROW);
            }

            // Store those values in the fragment.
            a[mi].reg(0) = tmp.x;
            a[mi].reg(1) = tmp.y;
            a[mi].reg(2) = tmp.z;
            a[mi].reg(3) = tmp.w;

            static_assert(BYTES_PER_XMMA_PER_CTA >= 128 ||
                          BYTES_PER_XMMA_PER_CTA ==  64 ||
                          (BYTES_PER_XMMA_PER_CTA == 32 &&
                          (Xmma_tile::XMMAS_M == 4 ||
                          Xmma_tile::XMMAS_M == 2 ||
                          Xmma_tile::XMMAS_M == 1)), "");

            // Move the pointer for the next ni. I expect the compiler to not recompute those.
            if( BYTES_PER_XMMA_PER_CTA >= 128 ) {
                // Nothing to do!
            } else if( BYTES_PER_XMMA_PER_CTA == 64 && Xmma_tile::XMMAS_M > 1 ) {
                this->smem_read_offset_ ^= BYTES_PER_XMMA_PER_CTA;
            } else if( BYTES_PER_XMMA_PER_CTA == 64 ) {
                // Nothing to do!
            } else if( BYTES_PER_XMMA_PER_CTA == 32 && Xmma_tile::XMMAS_M == 4 ) {
                this->smem_read_offset_ ^= BYTES_PER_LDS * (mi % 2 == 0 ? 2 : 6);
            } else if( BYTES_PER_XMMA_PER_CTA == 32 && Xmma_tile::XMMAS_M == 2 ) {
                this->smem_read_offset_ ^= BYTES_PER_LDS * 2;
            }
        }
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Traits, int N >
struct Rows_per_xor_pattern_ampere_row_a : public Rows_per_xor_pattern_ampere_a<Traits, N> {
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
    int ROWS_PER_XOR_PATTERN_ = Rows_per_xor_pattern_ampere_row_a<Traits, Cta_tile::K>::VALUE
>
struct Smem_tile_ampere_row_a : public Smem_tile_without_skews<Cta_tile,
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
    inline __device__ Smem_tile_ampere_row_a(void *smem, int tidx) : Base(smem, tidx) {

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
            smem_read_row  = (tidx & WARP_MASK_M) / WARP_DIV_M * Xmma_tile::M_PER_XMMA / 1 +
                             (tidx & 0x0f);
            smem_read_col  = (tidx & 0x07);
            smem_read_col ^= (tidx & 0x10) / 16;
            // For group fprop/dgrd. A is divided into 2 halves along K dimension.
            // The fist warp takes the first half and the second warp takes the second half.
            if ( Cta_tile::GROUPS > 1 && WARPS_K == 1 && WARPS_N > 1 ) {
                smem_read_col ^= (tidx & WARP_MASK_N) / WARP_DIV_N * (Cta_tile::K / WARPS_N) /
                                 (BYTES_PER_LDS * 8 / Base::BITS_PER_ELEMENT);
            }
        } else if( Base::ROWS_PER_XOR_PATTERN == 4 ) {
            smem_read_row  = (tidx & WARP_MASK_M) / WARP_DIV_M * Xmma_tile::M_PER_XMMA / 2 +
                             (tidx & 0x0e) /  2;
            smem_read_col  = (tidx & 0x06) /  2 + (tidx & 0x01) * 4;
            smem_read_col ^= (tidx & 0x10) / 16;
        } else if( Base::ROWS_PER_XOR_PATTERN == 2 ) {
            smem_read_row  = (tidx & WARP_MASK_M) / WARP_DIV_M * Xmma_tile::M_PER_XMMA / 4 +
                             (tidx & 0x0c) /  4;
            smem_read_col  = (tidx & 0x04) /  4 + (tidx & 0x03) * 2;
            smem_read_col ^= (tidx & 0x10) / 16;
        }

        static_assert(WARPS_K <= 2, "");
        static_assert(WARPS_K != 2 || Base::ROWS_PER_XOR_PATTERN != 2, "");

        // We "swap" the block for the second warp working on the same outputs in-CTA split-K.
        if( WARPS_K == 2 ) {
            smem_read_col ^= (tidx & WARP_MASK_K) / WARP_DIV_K * Xmma_tile_with_padding::XMMAS_K * 2;
        }

        // The shared memory offset.
        this->smem_read_offset_ = smem_read_row*Base::BYTES_PER_ROW + smem_read_col*BYTES_PER_LDS;
    }

    // Rewind smem_read_offset for last LDS phase in main loop.
    inline __device__ void reverse_smem_read_offset(int ki = 0) {
        // Undo the pointer increment for the next ni.
        // Should match the load function below for ki = 0.
        if( Xmma_tile_with_padding::XMMAS_K >=  2 ) {
            this->smem_read_offset_ ^= BYTES_PER_LDS * 2;
        }
    }

    // Load from shared memory.
    inline __device__ void load(Fragment (&a)[Xmma_tile::XMMAS_M], int ki) {
        #pragma unroll
        for( int mi = 0; mi < Xmma_tile::XMMAS_M; ++mi ) {
            // Jump by as many matrix rows as needed (a row in smem may pack multiple matrix rows).
            int offset = mi * Xmma_tile::M_PER_XMMA_PER_CTA * Base::BYTES_PER_ROW_BEFORE_PACKING;

            // Load using LDSM.M88.4.
            uint4 tmp;
            ldsm(tmp, this->smem_ + this->smem_read_offset_ + this->smem_read_buffer_ + offset);

            // Store the value into the fragment.
            a[mi].reg(0) = tmp.x;
            a[mi].reg(1) = tmp.y;
            a[mi].reg(2) = tmp.z;
            a[mi].reg(3) = tmp.w;
        }

        // Move the offset to the next possition. See doc/xmma_smem_layout.xlsx.
        static_assert(Xmma_tile_with_padding::XMMAS_K < 64, "Not implemented");
        if(        Xmma_tile_with_padding::XMMAS_K >= 32 && ki % 16 == 15 ) {
            this->smem_read_offset_ ^= 31 * BYTES_PER_LDS * 2;
        } else if( Xmma_tile_with_padding::XMMAS_K >= 16 && ki %  8 ==  7 ) {
            this->smem_read_offset_ ^= 15 * BYTES_PER_LDS * 2;
        } else if( Xmma_tile_with_padding::XMMAS_K >=  8 && ki %  4 ==  3 ) {
            this->smem_read_offset_ ^=  7 * BYTES_PER_LDS * 2;
        } else if( Xmma_tile_with_padding::XMMAS_K >=  4 && ki %  2 ==  1 ) {
            this->smem_read_offset_ ^=  3 * BYTES_PER_LDS * 2;
        } else if( Xmma_tile_with_padding::XMMAS_K >=  2 ) {
            this->smem_read_offset_ ^=  1 * BYTES_PER_LDS * 2;
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
        this->smem_read_offset_ ^= MASK * BYTES_PER_LDS * 2;
    }

};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Traits, int N >
struct Rows_per_xor_pattern_ampere_b {
    // The size in bits.
    enum { N_IN_BITS = N * Traits::BITS_PER_ELEMENT_B };
    // The number of rows.
    enum { VALUE = N_IN_BITS <= 256 ? 2 : (N_IN_BITS <= 512 ? 4 : 8) };
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Traits, int N >
struct Rows_per_xor_pattern_ampere_col_b : public Rows_per_xor_pattern_ampere_b<Traits, N> {
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
    int ROWS_PER_XOR_PATTERN_ = Rows_per_xor_pattern_ampere_col_b<Traits, Cta_tile::K>::VALUE
>
struct Smem_tile_ampere_col_b : public Smem_tile_without_skews<Cta_tile,
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

    // The number of STS per thread
    enum { STS_PER_THREAD_ = Base::ROWS * Base::THREADS_PER_ROW / Cta_tile::THREADS_PER_CTA };
    // The number of STS per thread must be at least 1.
    enum { STS_PER_THREAD = Max<1, STS_PER_THREAD_>::VALUE };

    // Ctor.
    inline __device__ Smem_tile_ampere_col_b(void *smem, int tidx) : Base(smem, tidx) {

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
                smem_read_row  = (tidx & WARP_MASK_N) / WARP_DIV_N * (Cta_tile::N / WARPS_N) / 1 +
                                 (tidx & 0x07) +
                                 (tidx & 0x10) / 2;
            } else {
                smem_read_row  = (tidx & WARP_MASK_N) / WARP_DIV_N * Xmma_tile::N_PER_XMMA / 1 +
                                 (tidx & 0x07) +
                                 (tidx & 0x10) / 2;
            }
            smem_read_col  = (tidx & 0x07);
            smem_read_col ^= (tidx & 0x08) / 8;
        } else if( Base::ROWS_PER_XOR_PATTERN == 4 ) {
            smem_read_row  = (tidx & WARP_MASK_N) / WARP_DIV_N * Xmma_tile::N_PER_XMMA / 2 +
                             (tidx & 0x06) / 2 +
                             (tidx & 0x10) / 4;
            smem_read_col  = (tidx & 0x06) / 2 + (tidx & 0x01) * 4;
            smem_read_col ^= (tidx & 0x08) / 8;
        } else if( Base::ROWS_PER_XOR_PATTERN == 2 ) {
            smem_read_row  = (tidx & WARP_MASK_N) / WARP_DIV_N * Xmma_tile::N_PER_XMMA / 4 +
                             (tidx & 0x04) / 4 +
                             (tidx & 0x10) / 8;
            smem_read_col  = (tidx & 0x04) / 4 + (tidx & 0x03) * 2;
            smem_read_col ^= (tidx & 0x08) / 8;
        }

        static_assert(WARPS_K <= 2, "");
        static_assert(WARPS_K != 2 || Base::ROWS_PER_XOR_PATTERN != 2, "");

        // We "swap" the block for the second warp working on the in-CTA split-K.
        if( WARPS_K == 2 ) {
            smem_read_col ^= (tidx & WARP_MASK_K) / WARP_DIV_K * Xmma_tile_with_padding::XMMAS_K * 2;
        }

        // The shared memory offset.
        this->smem_read_offset_ = smem_read_row*Base::BYTES_PER_ROW + smem_read_col*BYTES_PER_LDS;

        // Fill zeroes for group colw
        if ( Base::BITS_PER_ELEMENT == 16 && Cta_tile::GROUPS == 16 ) {
            int row_idx = threadIdx.x & (Base::THREADS_PER_ROW - 1);
            if ( row_idx < 2 ) {
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
            }
        }
    }

    // Rewind smem_read_offset for last LDS phase in main loop.
    inline __device__ void reverse_smem_read_offset(int ki = 0) {
        // Undo the pointer increment for the next ni.
        // Should match the load function below for ki = 0.
        if( Xmma_tile_with_padding::XMMAS_K >=  2 ) {
            this->smem_read_offset_ ^= BYTES_PER_LDS * 2;
        }
    }

    // Load from shared memory.
    inline __device__ void load(Fragment (&b)[Xmma_tile::XMMAS_N], int ki) {
        #pragma unroll
        for( int ni = 0; ni < Xmma_tile::XMMAS_N; ++ni ) {
            // Jump by as many matrix rows as needed (a row in smem may pack multiple matrix rows).
            int offset = ni * (Cta_tile::GROUPS > 1 ? Xmma_tile::N_PER_XMMA :
                         Xmma_tile::N_PER_XMMA_PER_CTA) * Base::BYTES_PER_ROW_BEFORE_PACKING;

            // Load using LDSM.M88.4.
            uint4 tmp;
            ldsm(tmp, this->smem_ + this->smem_read_offset_ + this->smem_read_buffer_ + offset);

            // Store the value into the fragment.
            b[ni].reg(0) = tmp.x;
            b[ni].reg(1) = tmp.y;
            b[ni].reg(2) = tmp.z;
            b[ni].reg(3) = tmp.w;
        }

        // Move the offset to the next possition. See doc/xmma_smem_layout.xlsx.
        static_assert(Xmma_tile_with_padding::XMMAS_K < 64, "Not implemented");
        if(        Xmma_tile_with_padding::XMMAS_K >= 32 && ki % 16 == 15 ) {
            this->smem_read_offset_ ^= 31 * BYTES_PER_LDS * 2;
        } else if( Xmma_tile_with_padding::XMMAS_K >= 16 && ki %  8 ==  7 ) {
            this->smem_read_offset_ ^= 15 * BYTES_PER_LDS * 2;
        } else if( Xmma_tile_with_padding::XMMAS_K >=  8 && ki %  4 ==  3 ) {
            this->smem_read_offset_ ^=  7 * BYTES_PER_LDS * 2;
        } else if( Xmma_tile_with_padding::XMMAS_K >=  4 && ki %  2 ==  1 ) {
            this->smem_read_offset_ ^=  3 * BYTES_PER_LDS * 2;
        } else if( Xmma_tile_with_padding::XMMAS_K >=  2 ) {
            this->smem_read_offset_ ^=  1 * BYTES_PER_LDS * 2;
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
        this->smem_read_offset_ ^= MASK * BYTES_PER_LDS * 2;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Traits, int N >
struct Rows_per_xor_pattern_ampere_row_b : public Rows_per_xor_pattern_ampere_b<Traits, N> {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Input_type, typename Output_type, int N >
struct Rows_per_xor_pattern_ampere_row_b<Ampere_hmma_tf32_traits<Input_type, Output_type>, N> {
    // The number of rows.
    enum { VALUE = 4 };
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
struct Smem_tile_ampere_row_b : public Smem_tile_without_skews<Cta_tile,
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
    // The number of elements per LDS.
    enum { ELEMENTS_PER_LDS = BYTES_PER_LDS * 8 / Traits::BITS_PER_ELEMENT_B };

    // The number of STS per thread
    enum { STS_PER_THREAD_ = Base::ROWS * Base::THREADS_PER_ROW / Cta_tile::THREADS_PER_CTA };
    // The number of STS per thread must be at least 1.
    enum { STS_PER_THREAD = Max<1, STS_PER_THREAD_>::VALUE };

    // Ctor.
    inline __device__ Smem_tile_ampere_row_b(void *smem, int tidx) : Base(smem, tidx) {

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

        static_assert((USE_LDSMT && Base::ROWS_PER_XOR_PATTERN == 8) ||
                      Base::ROWS_PER_XOR_PATTERN == 4 ||
                      Base::ROWS_PER_XOR_PATTERN == 2, "");

        if( USE_LDSMT && Base::ROWS_PER_XOR_PATTERN == 8 ) {
            // For group dgrad. B is divided into 2 halves along K dimension.
            // The fist warp takes the first half and the second warp takes the second half.
            if ( Cta_tile::GROUPS > 1 && WARPS_K == 1 && WARPS_N > 1 ) {
                smem_read_row = (tidx & WARP_MASK_N) / WARP_DIV_N * (Cta_tile::N / WARPS_N) +
                                (tidx & 0x07) + (tidx & 0x08);
            } else {
                smem_read_row = (tidx & WARP_MASK_K) / WARP_DIV_K * Xmma_tile::XMMAS_K * 16 +
                                (tidx & 0x07) + (tidx & 0x08);
            }
            smem_read_col = (tidx & 0x07);
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
        }

        // Each half-warp applies a different XOR pattern -- see the Excel document.
        if( USE_LDSMT ) {
            if ( Cta_tile::GROUPS > 1 && WARPS_K == 1 && WARPS_N > 1 )
                smem_read_col ^= (tidx & 0x10) / 16;
            else
                smem_read_col ^= (tidx & WARP_MASK_N) / WARP_DIV_N * 2 + (tidx & 0x10) / 16;
        } else {
            // Only for non-group.
            if ( Cta_tile::GROUPS == 1 ) {
                smem_read_col ^= (tidx & WARP_MASK_N) / WARP_DIV_N * 16;
            }
        }

        // The shared memory offset.
        this->smem_read_offset_ = smem_read_row*Base::BYTES_PER_ROW + smem_read_col*BYTES_PER_LDS;

        // Fill zeroes for group colw
        if ( Base::BITS_PER_ELEMENT == 16 && Cta_tile::GROUPS == 16 && Cta_tile::WARPS_N > 1 ) {
            int row_idx = threadIdx.x & (Base::THREADS_PER_ROW - 1);
            if ( row_idx < 2 ) {
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
            }
        }
    }

    // Rewind smem_read_offset for last LDS phase in main loop.
    inline __device__ void reverse_smem_read_offset(int ki = 0) {
        // The size of each element in bits.
        const int BITS_PER_ELT = Traits::BITS_PER_ELEMENT_B;
        // The size in bytes of the data needed to compute an XMMA per CTA.
        const int BYTES_PER_XMMA_PER_CTA = Xmma_tile::N_PER_XMMA_PER_CTA * BITS_PER_ELT / 8;

        #pragma unroll
        for( int ni = 0; ni < Xmma_tile::XMMAS_N; ++ni ) {
            // Undo the pointer increment for the next ni.
            // Should match the load function below for ki = 0.
            if ( Cta_tile::GROUPS > 1 && Cta_tile::WARPS_N > 1 && Xmma_tile::XMMAS_N > 1 ) {
                this->smem_read_offset_ ^= BYTES_PER_XMMA_PER_CTA / 2;
            } else {
                if( BYTES_PER_XMMA_PER_CTA >= 128 ) {
                    // Nothing to do!
                } else if( BYTES_PER_XMMA_PER_CTA == 64 && Xmma_tile::XMMAS_N > 1 ) {
                    this->smem_read_offset_ ^= BYTES_PER_XMMA_PER_CTA;
                } else if( BYTES_PER_XMMA_PER_CTA == 64 ) {
                    // Nothing to do!
                } else if( BYTES_PER_XMMA_PER_CTA == 32 && Xmma_tile::XMMAS_N == 4 ) {
                    this->smem_read_offset_ ^= BYTES_PER_LDS * (ni % 2 == 0 ? 2 : 6);
                } else if( BYTES_PER_XMMA_PER_CTA == 32 && Xmma_tile::XMMAS_N == 2 ) {
                    this->smem_read_offset_ ^= BYTES_PER_LDS * 2;
                }
            }
        }
    }

    // Load from shared memory.
    inline __device__ void load(Fragment (&b)[Xmma_tile::XMMAS_N], int ki) {
        // The size of each element in bits.
        const int BITS_PER_ELT = Traits::BITS_PER_ELEMENT_B;
        // The size in bytes of the data needed to compute an XMMA per CTA.
        const int BYTES_PER_XMMA_PER_CTA = Xmma_tile::N_PER_XMMA_PER_CTA * BITS_PER_ELT / 8;

        #pragma unroll
        for( int ni = 0; ni < Xmma_tile::XMMAS_N; ++ni ) {
            // Prepare the offset.
            int offset = ki * Base::ROWS_PER_XOR_PATTERN * 2 * Base::BYTES_PER_ROW;
            if ( Cta_tile::GROUPS > 1 && Cta_tile::WARPS_K == 1 && Cta_tile::WARPS_N > 1 ) {
                offset += this->smem_read_offset_;
            } else {
                if ( BYTES_PER_XMMA_PER_CTA == 32 ) {
                    offset += this->smem_read_offset_;
                } else if ( BYTES_PER_XMMA_PER_CTA == 64 ) {
                    offset += this->smem_read_offset_ + (ni/2) * BYTES_PER_XMMA_PER_CTA * 2;
                } else {
                    offset += this->smem_read_offset_ + (ni  ) * BYTES_PER_XMMA_PER_CTA;
                }
            }

            // Load the data using LDSM.MT88.2.
            uint32_t ptr = this->smem_ + this->smem_read_buffer_ + offset;
            uint4 tmp;
            if( USE_LDSMT ) {
                ldsmt(tmp, ptr);
            } else {
                lds(tmp.x, (ptr     ) + 0*Base::BYTES_PER_ROW);
                lds(tmp.y, (ptr     ) + 4*Base::BYTES_PER_ROW);
                lds(tmp.z, (ptr ^ 32) + 0*Base::BYTES_PER_ROW);
                lds(tmp.w, (ptr ^ 32) + 4*Base::BYTES_PER_ROW);
            }

            // Store those values in the fragment.
            b[ni].reg(0) = tmp.x;
            b[ni].reg(1) = tmp.y;
            b[ni].reg(2) = tmp.z;
            b[ni].reg(3) = tmp.w;

            static_assert(BYTES_PER_XMMA_PER_CTA >= 128 ||
                          BYTES_PER_XMMA_PER_CTA ==  64 ||
                          (BYTES_PER_XMMA_PER_CTA == 32 &&
                          (Xmma_tile::XMMAS_M == 4 ||
                          Xmma_tile::XMMAS_M == 2 ||
                          Xmma_tile::XMMAS_M == 1)), "");

            // Move the pointer for the next ni. I expect the compiler to not recompute those.
            if ( Cta_tile::GROUPS > 1 && Cta_tile::WARPS_N > 1 && Xmma_tile::XMMAS_N > 1 ) {
                this->smem_read_offset_ ^= BYTES_PER_XMMA_PER_CTA / 2;
            } else {
                if( BYTES_PER_XMMA_PER_CTA >= 128 ) {
                    // Nothing to do!
                } else if( BYTES_PER_XMMA_PER_CTA == 64 && Xmma_tile::XMMAS_N > 1 ) {
                    this->smem_read_offset_ ^= BYTES_PER_XMMA_PER_CTA;
                } else if( BYTES_PER_XMMA_PER_CTA == 64 ) {
                    // Nothing to do!
                } else if( BYTES_PER_XMMA_PER_CTA == 32 && Xmma_tile::XMMAS_N == 4 ) {
                    this->smem_read_offset_ ^= BYTES_PER_LDS * (ni % 2 == 0 ? 2 : 6);
                } else if( BYTES_PER_XMMA_PER_CTA == 32 && Xmma_tile::XMMAS_N == 2 ) {
                    this->smem_read_offset_ ^= BYTES_PER_LDS * 2;
                }
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
    int BUFFERS_PER_TILE,
    // The step between rows in the ctor.
    int STEP_BETWEEN_ROWS_IN_CTOR = 1,
    // The step between rows in the load member function.
    int STEP_BETWEEN_ROWS_IN_LOAD = 2
>
struct Smem_tile_ampere_col_interleaved_a : public Smem_tile_interleaved<Cta_tile, 
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
    inline __device__ Smem_tile_ampere_col_interleaved_a(void *smem, int tidx) : Base(smem, tidx) {

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
        int smem_read_row = (tidx & WARP_MASK_K) / WARP_DIV_K * Base::ROWS / WARPS_K + 
                            (tidx & 0x10) / 16 * STEP_BETWEEN_ROWS_IN_CTOR;
        int smem_read_col = (tidx & WARP_MASK_M) / WARP_DIV_M * Xmma_tile::M_PER_XMMA +
                            (tidx & 0x0f);

        // The shared memory offset.
        this->smem_read_offset_ = smem_read_row*Base::BYTES_PER_ROW + smem_read_col*BYTES_PER_LDS;
    }

    // Rewind smem_read_offset for last LDS phase in main loop.
    inline __device__ void reverse_read_smem_offset(int ki = 0) {
    }

    // Load from shared memory.
    inline __device__ void load(Fragment &a, int mi, int ki) {
        int offset = mi * Xmma_tile::M_PER_XMMA_PER_CTA * BYTES_PER_LDS + 
                     ki * STEP_BETWEEN_ROWS_IN_LOAD * Base::BYTES_PER_ROW;
        uint4 tmp; 
        ldsm(tmp, this->smem_ + this->smem_read_offset_ + this->smem_read_buffer_ + offset);
        a.reg(0) = tmp.x;
        a.reg(1) = tmp.y;
        a.reg(2) = tmp.z;
        a.reg(3) = tmp.w;
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
    int BUFFERS_PER_TILE,
    // The step between rows in the ctor.
    int STEP_BETWEEN_ROWS_IN_CTOR = 1,
    // The step between rows in the load member function.
    int STEP_BETWEEN_ROWS_IN_LOAD = 2
>
struct Smem_tile_ampere_row_interleaved_b : public Smem_tile_interleaved<Cta_tile, 
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
    inline __device__ Smem_tile_ampere_row_interleaved_b(void *smem, int tidx) : Base(smem, tidx) {

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
        int smem_read_row = (tidx & WARP_MASK_K) / WARP_DIV_K * Base::ROWS / WARPS_K +
                            (tidx & 0x08) / 8 * STEP_BETWEEN_ROWS_IN_CTOR;
        int smem_read_col = (tidx & WARP_MASK_N) / WARP_DIV_N * Xmma_tile::N_PER_XMMA +
                            (tidx & 0x07) + 
                            (tidx & 0x10) / 2;

        // The shared memory offset.
        this->smem_read_offset_ = smem_read_row*Base::BYTES_PER_ROW + smem_read_col*BYTES_PER_LDS;
    }

    // Rewind smem_read_offset for last LDS phase in main loop.
    inline __device__ void reverse_smem_read_offset(int ki = 0) {
    }

    // Load from shared memory.
    inline __device__ void load(Fragment &b, int ni, int ki) {
        int offset = ni * Xmma_tile::N_PER_XMMA_PER_CTA * BYTES_PER_LDS + 
                     ki * STEP_BETWEEN_ROWS_IN_LOAD * Base::BYTES_PER_ROW;
        uint4 tmp; 
        ldsm(tmp, this->smem_ + this->smem_read_offset_ + this->smem_read_buffer_ + offset);
        b.reg(0) = tmp.x;
        b.reg(1) = tmp.y;
        b.reg(2) = tmp.z;
        b.reg(3) = tmp.w;
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
struct Smem_tile_a<Ampere_hmma_fp16_traits, Cta_tile, Row, BYTES_PER_STS, BUFFERS_PER_TILE>
    : public Smem_tile_ampere_row_a<Ampere_hmma_fp16_traits,
                                    Cta_tile,
                                    BYTES_PER_STS,
                                    BUFFERS_PER_TILE> {
    // The traits class.
    using Traits = Ampere_hmma_fp16_traits;
    // The base class.
    using Base = Smem_tile_ampere_row_a<Traits, Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE>;

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
struct Smem_tile_a<Ampere_hmma_fp16_traits, Cta_tile, Col, BYTES_PER_STS, BUFFERS_PER_TILE >
    : public Smem_tile_ampere_col_a<Ampere_hmma_fp16_traits,
                                    Cta_tile,
                                    BYTES_PER_STS,
                                    BUFFERS_PER_TILE> {
    // The traits class.
    using Traits = Ampere_hmma_fp16_traits;
    // The base class.
    using Base = Smem_tile_ampere_col_a<Traits, Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE>;

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
struct Smem_tile_b<Ampere_hmma_fp16_traits, Cta_tile, Col, BYTES_PER_STS, BUFFERS_PER_TILE >
    : public Smem_tile_ampere_col_b<Ampere_hmma_fp16_traits,
                                    Cta_tile,
                                    BYTES_PER_STS,
                                    BUFFERS_PER_TILE> {

    // The traits class.
    using Traits = Ampere_hmma_fp16_traits;
    // The base class.
    using Base = Smem_tile_ampere_col_b<Traits, Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE>;

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
struct Smem_tile_b<Ampere_hmma_fp16_traits, Cta_tile, Row, BYTES_PER_STS, BUFFERS_PER_TILE >
    : public Smem_tile_ampere_row_b<Ampere_hmma_fp16_traits,
                                    Cta_tile,
                                    BYTES_PER_STS,
                                    BUFFERS_PER_TILE> {
    // The traits class.
    using Traits = Ampere_hmma_fp16_traits;
    // The base class.
    using Base = Smem_tile_ampere_row_b<Traits, Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE>;

    // Ctor.
    inline __device__ Smem_tile_b(void *smem, int tidx) : Base(smem, tidx) {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Cta_tile, bool IN_CTA_SPLIT_K >
struct Swizzle_epilogue<Ampere_hmma_fp16_traits, Cta_tile, Row, 16, IN_CTA_SPLIT_K>
    : public Swizzle_turing_hmma_fp16_epilogue<Ampere_hmma_fp16_traits, Cta_tile> {

    // The traits.
    using Traits = Ampere_hmma_fp16_traits;
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
struct Smem_tile_a<Ampere_hmma_fp32_traits, Cta_tile, Row, BYTES_PER_STS, BUFFERS_PER_TILE>
    : public Smem_tile_ampere_row_a<Ampere_hmma_fp32_traits,
                                    Cta_tile,
                                    BYTES_PER_STS,
                                    BUFFERS_PER_TILE> {
    // The traits class.
    using Traits = Ampere_hmma_fp32_traits;
    // The base class.
    using Base = Smem_tile_ampere_row_a<Traits, Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE>;

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
struct Smem_tile_a<Ampere_hmma_fp32_traits, Cta_tile, Col, BYTES_PER_STS, BUFFERS_PER_TILE>
    : public Smem_tile_ampere_col_a<Ampere_hmma_fp32_traits,
                                    Cta_tile,
                                    BYTES_PER_STS,
                                    BUFFERS_PER_TILE> {

    // The traits class.
    using Traits = Ampere_hmma_fp32_traits;
    // The base class.
    using Base = Smem_tile_ampere_col_a<Traits, Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE>;

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
struct Smem_tile_b<Ampere_hmma_fp32_traits, Cta_tile, Col, BYTES_PER_STS, BUFFERS_PER_TILE>
    : public Smem_tile_ampere_col_b<Ampere_hmma_fp32_traits,
                                    Cta_tile,
                                    BYTES_PER_STS,
                                    BUFFERS_PER_TILE> {
    // The traits class.
    using Traits = Ampere_hmma_fp32_traits;
    // The base class.
    using Base = Smem_tile_ampere_col_b<Traits, Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE>;

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
struct Smem_tile_b<Ampere_hmma_fp32_traits, Cta_tile, Row, BYTES_PER_STS, BUFFERS_PER_TILE>
    : public Smem_tile_ampere_row_b<Ampere_hmma_fp32_traits,
                                    Cta_tile,
                                    BYTES_PER_STS,
                                    BUFFERS_PER_TILE> {

    // The traits class.
    using Traits = Ampere_hmma_fp32_traits;
    // The base class.
    using Base = Smem_tile_ampere_row_b<Traits, Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE>;

    // Ctor.
    inline __device__ Smem_tile_b(void *smem, int tidx) : Base(smem, tidx) {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Cta_tile, bool IN_CTA_SPLIT_K >
struct Swizzle_epilogue<Ampere_hmma_fp32_traits, Cta_tile, Row, 16, IN_CTA_SPLIT_K>
    : public Swizzle_turing_hmma_fp32_epilogue<Ampere_hmma_fp32_traits,
                                               Cta_tile,
                                               Row> {
    // The traits class.
    using Traits = Ampere_hmma_fp32_traits;
    // The base class.
    using Base = Swizzle_turing_hmma_fp32_epilogue<Traits, Cta_tile, Row>;

    // Ctor.
    inline __device__ Swizzle_epilogue(void *smem, int tidx) : Base(smem, tidx) {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// H M M A . E 8 M 1 0
//
////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    // The output type.
    typename Output_type,
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The size of the STS.
    int BYTES_PER_STS,
    // The number of buffers per tile.
    int BUFFERS_PER_TILE
>
struct Smem_tile_a<Ampere_hmma_tf32_traits<lwtlass::float_tf32_t, Output_type>,
                   Cta_tile,
                   Row,
                   BYTES_PER_STS,
                   BUFFERS_PER_TILE>
    : public Smem_tile_ampere_row_a<Ampere_hmma_tf32_traits<lwtlass::float_tf32_t, Output_type>,
                                    Cta_tile,
                                    BYTES_PER_STS,
                                    BUFFERS_PER_TILE> {
    // The traits class.
    using Traits = Ampere_hmma_tf32_traits<lwtlass::float_tf32_t, Output_type>;
    // The base class.
    using Base = Smem_tile_ampere_row_a<Traits, Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE>;

    // Ctor.
    inline __device__ Smem_tile_a(void *smem, int tidx) : Base(smem, tidx) {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Fragment, int N >
static inline __device__ void float_to_tf32_rn(Fragment (&f)[N]) {
    #pragma unroll
    for( int ii = 0; ii < N; ++ii ) {
        #pragma unroll
        for( int jj = 0; jj < Fragment::NUM_REGS; ++jj ) {
            f[ii].reg(jj) = float_to_tf32_rn(reinterpret_cast<const float&>(f[ii].reg(jj)));
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    // The output type.
    typename Output_type,
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The size of the STS.
    int BYTES_PER_STS,
    // The number of buffers per tile.
    int BUFFERS_PER_TILE
>
struct Smem_tile_a<Ampere_hmma_tf32_traits<float, Output_type>,
                   Cta_tile,
                   Row,
                   BYTES_PER_STS,
                   BUFFERS_PER_TILE>
    : public Smem_tile_ampere_row_a<Ampere_hmma_tf32_traits<float, Output_type>,
                                    Cta_tile,
                                    BYTES_PER_STS,
                                    BUFFERS_PER_TILE> {
    // The traits class.
    using Traits = Ampere_hmma_tf32_traits<float, Output_type>;
    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;
    // The base class.
    using Base = Smem_tile_ampere_row_a<Traits, Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE>;

    // Ctor.
    inline __device__ Smem_tile_a(void *smem, int tidx) : Base(smem, tidx) {
    }

    inline __device__ void load(typename Base::Fragment (&a)[Xmma_tile::XMMAS_M], int ki) {
        #pragma unroll
        for( int mi = 0; mi < Xmma_tile::XMMAS_M; ++mi ) {
            // Jump by as many matrix rows as needed (a row in smem may pack multiple matrix rows).
            int offset = mi * Xmma_tile::M_PER_XMMA_PER_CTA * Base::BYTES_PER_ROW_BEFORE_PACKING;

            // Load using LDSM.M88.4.
            uint4 tmp;
            ldsm(tmp, this->smem_ + this->smem_read_offset_ + this->smem_read_buffer_ + offset);

            // Store the value into the fragment.
            a[mi].reg(0) = tmp.x;
            a[mi].reg(1) = tmp.y;
            a[mi].reg(2) = tmp.z;
            a[mi].reg(3) = tmp.w;
            
            float x0 = reinterpret_cast<float const &>(a[mi].reg(0)); 
            float x1 = reinterpret_cast<float const &>(a[mi].reg(1)); 
            float x2 = reinterpret_cast<float const &>(a[mi].reg(2)); 
            float x3 = reinterpret_cast<float const &>(a[mi].reg(3)); 
            

            a[mi].reg(0) = colwert_tf32(x0);
            a[mi].reg(1) = colwert_tf32(x1);
            a[mi].reg(2) = colwert_tf32(x2);
            a[mi].reg(3) = colwert_tf32(x3);
        }

        // Move the offset to the next possition. See doc/xmma_smem_layout.xlsx.
        static_assert(Base::Xmma_tile_with_padding::XMMAS_K < 64, "Not implemented");
        if(        Base::Xmma_tile_with_padding::XMMAS_K >= 32 && ki % 16 == 15 ) {
            this->smem_read_offset_ ^= 31 * Base::BYTES_PER_LDS * 2;
        } else if( Base::Xmma_tile_with_padding::XMMAS_K >= 16 && ki %  8 ==  7 ) {
            this->smem_read_offset_ ^= 15 * Base::BYTES_PER_LDS * 2;
        } else if( Base::Xmma_tile_with_padding::XMMAS_K >=  8 && ki %  4 ==  3 ) {
            this->smem_read_offset_ ^=  7 * Base::BYTES_PER_LDS * 2;
        } else if( Base::Xmma_tile_with_padding::XMMAS_K >=  4 && ki %  2 ==  1 ) {
            this->smem_read_offset_ ^=  3 * Base::BYTES_PER_LDS * 2;
        } else if( Base::Xmma_tile_with_padding::XMMAS_K >=  2 ) {
            this->smem_read_offset_ ^=  1 * Base::BYTES_PER_LDS * 2;
        }
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    // The output type.
    typename Output_type,
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The size of the STS.
    int BYTES_PER_STS,
    // The number of buffers per tile.
    int BUFFERS_PER_TILE
>
struct Smem_tile_a<Ampere_hmma_tf32_traits<lwtlass::float_tf32_t, Output_type>,
                   Cta_tile,
                   Col,
                   BYTES_PER_STS,
                   BUFFERS_PER_TILE>
    : public Smem_tile_ampere_col_a<Ampere_hmma_tf32_traits<lwtlass::float_tf32_t, Output_type>,
                                    Cta_tile,
                                    BYTES_PER_STS,
                                    BUFFERS_PER_TILE> {
    // The traits class.
    using Traits = Ampere_hmma_tf32_traits<lwtlass::float_tf32_t, Output_type>;
    // The base class.
    using Base = Smem_tile_ampere_col_a<Traits, Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE>;

    // Ctor.
    inline __device__ Smem_tile_a(void *smem, int tidx) : Base(smem, tidx) {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    // The output type.
    typename Output_type,
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The size of the STS.
    int BYTES_PER_STS,
    // The number of buffers per tile.
    int BUFFERS_PER_TILE
>
struct Smem_tile_a<Ampere_hmma_tf32_traits<float, Output_type>,
                   Cta_tile,
                   Col,
                   BYTES_PER_STS,
                   BUFFERS_PER_TILE>
    : public Smem_tile_ampere_col_a<Ampere_hmma_tf32_traits<float, Output_type>,
                                    Cta_tile,
                                    BYTES_PER_STS,
                                    BUFFERS_PER_TILE> {
    // The traits class.
    using Traits = Ampere_hmma_tf32_traits<float, Output_type>;
    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;
    // The base class.
    using Base = Smem_tile_ampere_col_a<Traits, Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE>;

    // Ctor.
    inline __device__ Smem_tile_a(void *smem, int tidx) : Base(smem, tidx) {
    }

    inline __device__ void load(typename Base::Fragment (&a)[Xmma_tile::XMMAS_M], int ki) {
        Base::load(a, ki);
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    // The output type.
    typename Output_type,
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The size of the STS.
    int BYTES_PER_STS,
    // The number of buffers per tile.
    int BUFFERS_PER_TILE
>
struct Smem_tile_b<Ampere_hmma_tf32_traits<lwtlass::float_tf32_t, Output_type>,
                   Cta_tile,
                   Col,
                   BYTES_PER_STS,
                   BUFFERS_PER_TILE>
    : public Smem_tile_ampere_col_b<Ampere_hmma_tf32_traits<lwtlass::float_tf32_t, Output_type>,
                                    Cta_tile,
                                    BYTES_PER_STS,
                                    BUFFERS_PER_TILE> {
    // The traits class.
    using Traits = Ampere_hmma_tf32_traits<lwtlass::float_tf32_t, Output_type>;
    // The base class.
    using Base = Smem_tile_ampere_col_b<Traits, Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE>;

    // Ctor.
    inline __device__ Smem_tile_b(void *smem, int tidx) : Base(smem, tidx) {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    // The output type.
    typename Output_type,
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The size of the STS.
    int BYTES_PER_STS,
    // The number of buffers per tile.
    int BUFFERS_PER_TILE
>
struct Smem_tile_b<Ampere_hmma_tf32_traits<float, Output_type>,
                   Cta_tile,
                   Col,
                   BYTES_PER_STS,
                   BUFFERS_PER_TILE>
    : public Smem_tile_ampere_col_b<Ampere_hmma_tf32_traits<float, Output_type>,
                                    Cta_tile,
                                    BYTES_PER_STS,
                                    BUFFERS_PER_TILE> {
    // The traits class.
    using Traits = Ampere_hmma_tf32_traits<float, Output_type>;
    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;
    // The base class.
    using Base = Smem_tile_ampere_col_b<Traits, Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE>;

    // Ctor.
    inline __device__ Smem_tile_b(void *smem, int tidx) : Base(smem, tidx) {
    }

    // Load FP32s and colwert to TF32s.
    inline __device__ void load(typename Base::Fragment (&b)[Xmma_tile::XMMAS_N], int ki) {
        #pragma unroll
        for( int ni = 0; ni < Xmma_tile::XMMAS_N; ++ni ) {
            // Jump by as many matrix rows as needed (a row in smem may pack multiple matrix rows).
            int offset = ni * (Cta_tile::GROUPS > 1 ? Xmma_tile::N_PER_XMMA :
                         Xmma_tile::N_PER_XMMA_PER_CTA) * Base::BYTES_PER_ROW_BEFORE_PACKING;

            // Load using LDSM.M88.4.
            uint4 tmp;
            ldsm(tmp, this->smem_ + this->smem_read_offset_ + this->smem_read_buffer_ + offset);

            // Store the value into the fragment.
            b[ni].reg(0) = tmp.x;
            b[ni].reg(1) = tmp.y;
            b[ni].reg(2) = tmp.z;
            b[ni].reg(3) = tmp.w;
            
            float x0 = reinterpret_cast<float const &>(b[ni].reg(0)); 
            float x1 = reinterpret_cast<float const &>(b[ni].reg(1)); 
            float x2 = reinterpret_cast<float const &>(b[ni].reg(2)); 
            float x3 = reinterpret_cast<float const &>(b[ni].reg(3)); 
            

            b[ni].reg(0) = colwert_tf32(x0);
            b[ni].reg(1) = colwert_tf32(x1);
            b[ni].reg(2) = colwert_tf32(x2);
            b[ni].reg(3) = colwert_tf32(x3);
        }

        // Move the offset to the next possition. See doc/xmma_smem_layout.xlsx.
        static_assert(Base::Xmma_tile_with_padding::XMMAS_K < 64, "Not implemented");
        if(        Base::Xmma_tile_with_padding::XMMAS_K >= 32 && ki % 16 == 15 ) {
            this->smem_read_offset_ ^= 31 * Base::BYTES_PER_LDS * 2;
        } else if( Base::Xmma_tile_with_padding::XMMAS_K >= 16 && ki %  8 ==  7 ) {
            this->smem_read_offset_ ^= 15 * Base::BYTES_PER_LDS * 2;
        } else if( Base::Xmma_tile_with_padding::XMMAS_K >=  8 && ki %  4 ==  3 ) {
            this->smem_read_offset_ ^=  7 * Base::BYTES_PER_LDS * 2;
        } else if( Base::Xmma_tile_with_padding::XMMAS_K >=  4 && ki %  2 ==  1 ) {
            this->smem_read_offset_ ^=  3 * Base::BYTES_PER_LDS * 2;
        } else if( Base::Xmma_tile_with_padding::XMMAS_K >=  2 ) {
            this->smem_read_offset_ ^=  1 * Base::BYTES_PER_LDS * 2;
        }
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    // The output type.
    typename Output_type,
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The size of the STS.
    int BYTES_PER_STS,
    // The number of buffers per tile.
    int BUFFERS_PER_TILE
>
struct Smem_tile_b<Ampere_hmma_tf32_traits<lwtlass::float_tf32_t, Output_type>,
                   Cta_tile,
                   Row,
                   BYTES_PER_STS,
                   BUFFERS_PER_TILE>
    : public Smem_tile_ampere_row_b<Ampere_hmma_tf32_traits<lwtlass::float_tf32_t, Output_type>,
                                    Cta_tile,
                                    BYTES_PER_STS,
                                    BUFFERS_PER_TILE> {
    // The traits class.
    using Traits = Ampere_hmma_tf32_traits<lwtlass::float_tf32_t, Output_type>;
    // The base class.
    using Base = Smem_tile_ampere_row_b<Traits, Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE>;

    // Ctor.
    inline __device__ Smem_tile_b(void *smem, int tidx) : Base(smem, tidx) {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    // The output type.
    typename Output_type,
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The size of the STS.
    int BYTES_PER_STS,
    // The number of buffers per tile.
    int BUFFERS_PER_TILE
>
struct Smem_tile_b<Ampere_hmma_tf32_traits<float, Output_type>,
                   Cta_tile,
                   Row,
                   BYTES_PER_STS,
                   BUFFERS_PER_TILE>
    : public Smem_tile_ampere_row_b<Ampere_hmma_tf32_traits<float, Output_type>,
                                    Cta_tile,
                                    BYTES_PER_STS,
                                    BUFFERS_PER_TILE> {
    // The traits class.
    using Traits = Ampere_hmma_tf32_traits<float, Output_type>;
    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;
    // The base class.
    using Base = Smem_tile_ampere_row_b<Traits, Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE>;

    // Ctor.
    inline __device__ Smem_tile_b(void *smem, int tidx) : Base(smem, tidx) {
    }

    inline __device__ void load(typename Base::Fragment (&b)[Xmma_tile::XMMAS_N], int ki) {
        Base::load(b, ki);
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    // Traits of tensor-core.
    typename Traits,
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // Layout of storage.
    typename Layout,
    // Bytes per LDS.
    int BYTES_PER_LDS_ = 16
>
struct Swizzle_ampere_hmma_tf32_epilogue {};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    // Traits of tensor-core.
    typename Traits,
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // Bytes per lds.
    int BYTES_PER_LDS_
>
struct Swizzle_ampere_hmma_tf32_epilogue<Traits, Cta_tile, Col, BYTES_PER_LDS_> {
    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;

    // The number of bytes for key memory instruction.
    enum { BYTES_PER_LDS = BYTES_PER_LDS_, BYTES_PER_STS = 8 };
    // The number of elements per LDS.
    enum { ELEMENTS_PER_LDS = BYTES_PER_LDS / sizeof(float) };
    // The number of rows in shared memory.
    enum { ROWS = Xmma_tile::N_PER_XMMA_PER_CTA };
    // The size of a single row in bytes.
    enum { BYTES_PER_ROW = Cta_tile::M * Cta_tile::WARPS_K * sizeof(float) };
    // The skew to avoid bank conflicts.
    enum { BYTES_PER_SKEW = 32 };
    // The number of columns with the skew.
    enum { BYTES_PER_ROW_WITH_SKEW = BYTES_PER_ROW + BYTES_PER_SKEW };
    // The size in bytes in shared memory.
    enum { BYTES_PER_TILE = ROWS * BYTES_PER_ROW_WITH_SKEW };

    // The number of threads per output row. Each thread writes 8 elements.
    enum { THREADS_PER_ROW =
        Min<Cta_tile::THREADS_PER_CTA, Cta_tile::M / ELEMENTS_PER_LDS>::VALUE };

    // The number of column loaded per STG
    enum { COLUMNS_PER_STG = THREADS_PER_ROW * ELEMENTS_PER_LDS };

    // The number of rows written in one STG.128.
    enum { ROWS_PER_STG = Cta_tile::THREADS_PER_CTA / THREADS_PER_ROW };
    static_assert(ROWS_PER_STG > 0, "");

    // The number of steps needed to load the columns.
    enum { STGS_PER_COLUMN = Xmma_tile::N_PER_XMMA_PER_CTA / ROWS_PER_STG };

    // // How we see the distribution of data.
    // enum { THREADS_PER_XMMA_M = 8, THREADS_PER_XMMA_N = 4 };
    // // The number of elements stored per thread.
    // enum { M_PER_XMMA_PER_THREAD = 2, N_PER_XMMA_PER_THREAD = 4 };

    // Do we skip the syncthreads in the epilogue? Of course, not :)
    enum { SKIP_SYNCTHREADS = 0 };

    // Ctor.
    inline __device__ Swizzle_ampere_hmma_tf32_epilogue(void *smem, int tidx)
        : smem_(get_smem_pointer(smem)) {

        // The number of warps.
        const int WARPS_M = Cta_tile::WARPS_M;
        const int WARPS_N = Cta_tile::WARPS_N;
        const int WARPS_K = Cta_tile::WARPS_K;

        //
        //     tidx   0: row =  0, col = 0
        //     tidx   1: row =  0, col = 2
        //     tidx   2: row =  0, col = 4
        //     tidx   3: row =  0, col = 6
        //     tidx   4: row =  1, col = 0
        //     tidx   5: row =  1, col = 2
        //     tidx   6: row =  1, col = 4
        //     tidx   7: row =  1, col = 6
        //     tidx   8: row =  2, col = 0
        //     tidx   9: row =  2, col = 2
        //     tidx  10: row =  2, col = 4
        //     tidx  11: row =  2, col = 6
        //     tidx  12: row =  3, col = 0
        //     tidx  13: row =  3, col = 2
        //     tidx  14: row =  3, col = 4
        //     tidx  15: row =  3, col = 6
        //     tidx  16: row =  4, col = 0
        //     tidx  17: row =  4, col = 2
        //     tidx  18: row =  4, col = 4
        //     tidx  19: row =  4, col = 6
        //     tidx  20: row =  5, col = 0
        //     tidx  21: row =  5, col = 2
        //     tidx  22: row =  5, col = 4
        //     tidx  23: row =  5, col = 6
        //     tidx  24: row =  6, col = 0
        //     tidx  25: row =  6, col = 2
        //     tidx  26: row =  6, col = 4
        //     tidx  27: row =  6, col = 6
        //     tidx  28: row =  7, col = 0
        //     tidx  29: row =  7, col = 2
        //     tidx  30: row =  7, col = 4
        //     tidx  31: row =  7, col = 6

        // The masks to select the warps.
        const int WARP_MASK_M = Warp_masks<WARPS_M, WARPS_N, WARPS_K>::M;
        const int WARP_MASK_N = Warp_masks<WARPS_M, WARPS_N, WARPS_K>::N;
        const int WARP_MASK_K = Warp_masks<WARPS_M, WARPS_N, WARPS_K>::K;

        // The divisor for the warps.
        const int WARP_DIV_M =       1 *       1 * Cta_tile::THREADS_PER_WARP;
        const int WARP_DIV_N = WARPS_M *       1 * Cta_tile::THREADS_PER_WARP;
        const int WARP_DIV_K = WARPS_M * WARPS_N * Cta_tile::THREADS_PER_WARP;

        // Compute the row and the column in shared memory. Each warp reads from a 16*16B segment.
        const int smem_write_row = (tidx & WARP_MASK_N) / WARP_DIV_N *
                                   (Cta_tile::GROUPS > 1 ? Cta_tile::N / WARPS_N : Xmma_tile::N_PER_XMMA) +
                                   (tidx % 4) * 2;
        const int smem_write_col = (tidx & WARP_MASK_M) / WARP_DIV_M * Xmma_tile::M_PER_XMMA +
                                   (tidx & WARP_MASK_K) / WARP_DIV_K * Cta_tile::M +
                                   ((tidx % 32) / 4) * 1;

        // The corresponding offset.
        smem_write_offset_ = smem_write_row*BYTES_PER_ROW_WITH_SKEW + smem_write_col*sizeof(float);

        // The row and column read by a single thread.
        const int smem_read_row = tidx / THREADS_PER_ROW;
        const int smem_read_col = tidx % THREADS_PER_ROW * BYTES_PER_LDS;

        // The corresponding offset.
        smem_read_offset_ = smem_read_row*BYTES_PER_ROW_WITH_SKEW + smem_read_col;
    }

    // Load from the tile in shared memory.
    template< typename Fragment_post_swizzle >
    inline __device__ void load(Fragment_post_swizzle &dst, int oi) const {
        int offset =
            (oi % STGS_PER_COLUMN) * ROWS_PER_STG * BYTES_PER_ROW_WITH_SKEW
            + (oi / STGS_PER_COLUMN) * COLUMNS_PER_STG * sizeof(float);

        #pragma unroll
        for( int ki = 0; ki < Cta_tile::WARPS_K; ++ki ) {
            // Compute the address to load from.
            uint32_t ptr = smem_ + smem_read_offset_ + offset + ki * Cta_tile::M * sizeof(float);

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
        // Each thread stores 2 rows and 4 columns. Hence an XMMA per warp is (2x8)x(2x8). The
        // first loop (mi) is over the 2 series of 8 rows. The offset depends on 8*mi and the ni
        // index represents the ni-th XMMA in the N dimension.
        #pragma unroll
        for( int mi = 0; mi < 2; ++mi ) {
            // The row offset. 8 rows are written per iteration of mi.
            int row_offset = mi * 8 * sizeof(float);

            // The column offset. As many columns as the CTA-wide XMMA in the N dimension.
            int stride = Cta_tile::GROUPS > 1 ? Xmma_tile::M_PER_XMMA : Xmma_tile::M_PER_XMMA_PER_CTA;
            int col_offset = ni * stride * sizeof(float);

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
    // Bytes per lds.
    int BYTES_PER_LDS_
>
struct Swizzle_ampere_hmma_tf32_epilogue<Traits, Cta_tile, Row, BYTES_PER_LDS_> {
    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;

    // The number of bytes for key memory instruction.
    enum { BYTES_PER_LDS = BYTES_PER_LDS_, BYTES_PER_STS = 8 };
    // The number of elements per LDS.
    enum { ELEMENTS_PER_LDS = BYTES_PER_LDS / sizeof(float) };
    // The number of rows in shared memory.
    enum { ROWS = Xmma_tile::M_PER_XMMA_PER_CTA };
    // The size of a single row in bytes.
    enum { BYTES_PER_ROW = Cta_tile::N * Cta_tile::WARPS_K * sizeof(float) };
    // The skew to avoid bank conflicts.
    enum { BYTES_PER_SKEW = 32 };
    // The number of columns with the skew.
    enum { BYTES_PER_ROW_WITH_SKEW = BYTES_PER_ROW + BYTES_PER_SKEW };
    // The size in bytes in shared memory.
    enum { BYTES_PER_TILE = ROWS * BYTES_PER_ROW_WITH_SKEW };

    // The number of threads per output row. Each thread writes 8 elements.
    enum { THREADS_PER_ROW_ = Cta_tile::N / ELEMENTS_PER_LDS };
    // The number of threads per output row. Each thread writes 8 elements.
    enum { THREADS_PER_ROW = Min<Cta_tile::THREADS_PER_CTA, THREADS_PER_ROW_>::VALUE };

    // The number of rows written in one STG.128.
    enum { ROWS_PER_STG = Cta_tile::THREADS_PER_CTA / THREADS_PER_ROW };
    // Make sure we store at least 1 row per STG.
    static_assert(ROWS_PER_STG > 0, "");

    // Do we skip the syncthreads in the epilogue? Of course, not :)
    enum { SKIP_SYNCTHREADS = 0 };

    // Ctor.
    inline __device__ Swizzle_ampere_hmma_tf32_epilogue(void *smem, int tidx)
        : smem_(get_smem_pointer(smem)) {

        // The number of warps.
        const int WARPS_M = Cta_tile::WARPS_M;
        const int WARPS_N = Cta_tile::WARPS_N;
        const int WARPS_K = Cta_tile::WARPS_K;

        //
        //     tidx   0: row =  0, col = 0
        //     tidx   1: row =  0, col = 2
        //     tidx   2: row =  0, col = 4
        //     tidx   3: row =  0, col = 6
        //     tidx   4: row =  1, col = 0
        //     tidx   5: row =  1, col = 2
        //     tidx   6: row =  1, col = 4
        //     tidx   7: row =  1, col = 6
        //     tidx   8: row =  2, col = 0
        //     tidx   9: row =  2, col = 2
        //     tidx  10: row =  2, col = 4
        //     tidx  11: row =  2, col = 6
        //     tidx  12: row =  3, col = 0
        //     tidx  13: row =  3, col = 2
        //     tidx  14: row =  3, col = 4
        //     tidx  15: row =  3, col = 6
        //     tidx  16: row =  4, col = 0
        //     tidx  17: row =  4, col = 2
        //     tidx  18: row =  4, col = 4
        //     tidx  19: row =  4, col = 6
        //     tidx  20: row =  5, col = 0
        //     tidx  21: row =  5, col = 2
        //     tidx  22: row =  5, col = 4
        //     tidx  23: row =  5, col = 6
        //     tidx  24: row =  6, col = 0
        //     tidx  25: row =  6, col = 2
        //     tidx  26: row =  6, col = 4
        //     tidx  27: row =  6, col = 6
        //     tidx  28: row =  7, col = 0
        //     tidx  29: row =  7, col = 2
        //     tidx  30: row =  7, col = 4
        //     tidx  31: row =  7, col = 6

	// The masks to select the warps.
        const int WARP_MASK_M = Warp_masks<WARPS_M, WARPS_N, WARPS_K>::M;
        const int WARP_MASK_N = Warp_masks<WARPS_M, WARPS_N, WARPS_K>::N;
        const int WARP_MASK_K = Warp_masks<WARPS_M, WARPS_N, WARPS_K>::K;

        // The divisor for the warps.
        const int WARP_DIV_M =       1 *       1 * Cta_tile::THREADS_PER_WARP;
        const int WARP_DIV_N = WARPS_M *       1 * Cta_tile::THREADS_PER_WARP;
        const int WARP_DIV_K = WARPS_M * WARPS_N * Cta_tile::THREADS_PER_WARP;

        // Compute the row and the column in shared memory. Each warp reads from a 16*16B segment.
        const int smem_write_row = (tidx & WARP_MASK_M) / WARP_DIV_M * Xmma_tile::M_PER_XMMA +
                                   (tidx & 0x1c) / 4;
        const int smem_write_col = (tidx & WARP_MASK_N) / WARP_DIV_N *
                                   (Cta_tile::GROUPS > 1 ? Cta_tile::N / WARPS_N : Xmma_tile::N_PER_XMMA) +
                                   (tidx & WARP_MASK_K) / WARP_DIV_K * Cta_tile::N +
                                   (tidx & 0x03) * 2;

        // The corresponding offset.
        smem_write_offset_ = smem_write_row*BYTES_PER_ROW_WITH_SKEW + smem_write_col*sizeof(float);

        // The row and column read by a single thread.
        const int smem_read_row = tidx / THREADS_PER_ROW;
        const int smem_read_col = tidx % THREADS_PER_ROW * BYTES_PER_LDS;

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
            uint32_t ptr = smem_ + smem_read_offset_ + offset + ki * Cta_tile::N * sizeof(float);

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
        // Each thread stores 2 rows and 4 columns. Hence an XMMA per warp is (2x8)x(2x8). The
        // first loop (mi) is over the 2 series of 8 rows. The offset depends on 8*mi and the ni
        // index represents the ni-th XMMA in the N dimension.
        #pragma unroll
        for( int mi = 0; mi < 2; ++mi ) {
            // The row offset. 8 rows are written per iteration of mi.
            int row_offset = mi * 8 * BYTES_PER_ROW_WITH_SKEW;
            // The column offset. As many columns as the CTA-wide XMMA in the N dimension.
            int stride = Cta_tile::GROUPS > 1 ? Xmma_tile::N_PER_XMMA : Xmma_tile::N_PER_XMMA_PER_CTA;
            int col_offset = ni * stride * sizeof(float);

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

template<
    // The instruction traits.
    typename Traits,
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile
>
struct Swizzle_ampere_imma_int32_epilogue
    : public Swizzle_turing_imma_int32_epilogue<Traits, Cta_tile> {

    // The base class.
    using Base = Swizzle_turing_imma_int32_epilogue<Traits, Cta_tile>;
    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;

    // Ctor.
    inline __device__ Swizzle_ampere_imma_int32_epilogue(void *smem, int tidx)
        : Base(smem, tidx) {
    }

    // Store to the tile in shared memory.
    template<typename Fragment_pre_swizzle>
    inline __device__ void store(int ni, const Fragment_pre_swizzle &c) {
        #pragma unroll
        for( int mi = 0; mi < Base::M_PER_XMMA_PER_THREAD; ++mi ) {
            int offset = mi * Base::THREADS_PER_XMMA_M * Base::BYTES_PER_ROW_WITH_SKEW +
                         ni * Xmma_tile::N_PER_XMMA_PER_CTA * sizeof(int32_t);

            uint32_t ptr = this->smem_ + this->smem_write_offset_ + offset;
            sts(ptr +  0, make_uint2(c.reg(2*mi + 0), c.reg(2*mi + 1)));
            sts(ptr + 32, make_uint2(c.reg(2*mi + 4), c.reg(2*mi + 5)));
        }
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    typename Input_type,
    typename Output_type,
    typename Cta_tile,
    int BYTES_PER_LDS_,
    bool IN_CTA_SPLIT_K
>
struct Swizzle_epilogue<Ampere_hmma_tf32_traits<Input_type, Output_type>,
                        Cta_tile,
                        Col,
                        BYTES_PER_LDS_,
                        IN_CTA_SPLIT_K>
    : public Swizzle_ampere_hmma_tf32_epilogue<Ampere_hmma_tf32_traits<Input_type, Output_type>,
                                               Cta_tile,
                                               Col,
                                               BYTES_PER_LDS_> {
    // The traits.
    using Traits = Ampere_hmma_tf32_traits<Input_type, Output_type>;
    // The base class.
    using Base = Swizzle_ampere_hmma_tf32_epilogue<Traits, Cta_tile, Col, BYTES_PER_LDS_>;

    // Ctor.
    inline __device__ Swizzle_epilogue(void *smem, int tidx) : Base(smem, tidx) {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    typename Input_type,
    typename Output_type,
    typename Cta_tile,
    int BYTES_PER_LDS_,
    bool IN_CTA_SPLIT_K
>
struct Swizzle_epilogue<Ampere_hmma_tf32_traits<Input_type, Output_type>,
                        Cta_tile,
                        Row,
                        BYTES_PER_LDS_,
                        IN_CTA_SPLIT_K>
    : public Swizzle_ampere_hmma_tf32_epilogue<Ampere_hmma_tf32_traits<Input_type, Output_type>,
                                               Cta_tile,
                                               Row,
                                               BYTES_PER_LDS_> {
    // The traits.
    using Traits = Ampere_hmma_tf32_traits<Input_type, Output_type>;
    // The base class.
    using Base = Swizzle_ampere_hmma_tf32_epilogue<Traits, Cta_tile, Row, BYTES_PER_LDS_>;

    // Ctor.
    inline __device__ Swizzle_epilogue(void *smem, int tidx) : Base(smem, tidx) {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// H M M A . E 8 M 7
//
////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    // Output data type.
    typename Output_type,
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The size of the STS.
    int BYTES_PER_STS,
    // The number of buffers per tile.
    int BUFFERS_PER_TILE
>
struct Smem_tile_a<Ampere_hmma_bf16_traits<lwtlass::float_bf16_t, Output_type>,
                   Cta_tile,
                   Row,
                   BYTES_PER_STS,
                   BUFFERS_PER_TILE>
    : public Smem_tile_ampere_row_a<Ampere_hmma_bf16_traits<lwtlass::float_bf16_t, Output_type>,
                                    Cta_tile,
                                    BYTES_PER_STS,
                                    BUFFERS_PER_TILE> {
    // The traits class.
    using Traits = Ampere_hmma_bf16_traits<lwtlass::float_bf16_t, Output_type>;
    // The base class.
    using Base = Smem_tile_ampere_row_a<Traits, Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE>;

    // Ctor.
    inline __device__ Smem_tile_a(void *smem, int tidx) : Base(smem, tidx) {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    // Output type name.
    typename Output_type,
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The size of the STS.
    int BYTES_PER_STS,
    // The number of buffers per tile.
    int BUFFERS_PER_TILE
>
struct Smem_tile_a<Ampere_hmma_bf16_traits<lwtlass::float_bf16_t, Output_type>,
                   Cta_tile,
                   Col,
                   BYTES_PER_STS,
                   BUFFERS_PER_TILE>
    : public Smem_tile_ampere_col_a<Ampere_hmma_bf16_traits<lwtlass::float_bf16_t, Output_type>,
                                    Cta_tile,
                                    BYTES_PER_STS,
                                    BUFFERS_PER_TILE> {

    // The traits class.
    using Traits = Ampere_hmma_bf16_traits<lwtlass::float_bf16_t, Output_type>;
    // The base class.
    using Base = Smem_tile_ampere_col_a<Traits, Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE>;

    // Ctor.
    inline __device__ Smem_tile_a(void *smem, int tidx) : Base(smem, tidx) {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    // Output data type.
    typename Output_type,
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The size of the STS.
    int BYTES_PER_STS,
    // The number of buffers per tile.
    int BUFFERS_PER_TILE
>
struct Smem_tile_b<Ampere_hmma_bf16_traits<lwtlass::float_bf16_t, Output_type>,
                   Cta_tile,
                   Col,
                   BYTES_PER_STS,
                   BUFFERS_PER_TILE>
    : public Smem_tile_ampere_col_b<Ampere_hmma_bf16_traits<lwtlass::float_bf16_t, Output_type>,
                                    Cta_tile,
                                    BYTES_PER_STS,
                                    BUFFERS_PER_TILE> {
    // The traits class.
    using Traits = Ampere_hmma_bf16_traits<lwtlass::float_bf16_t, Output_type>;
    // The base class.
    using Base = Smem_tile_ampere_col_b<Traits, Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE>;

    // Ctor.
    inline __device__ Smem_tile_b(void *smem, int tidx) : Base(smem, tidx) {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    // Output data type
    typename Output_type,
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The size of the STS.
    int BYTES_PER_STS,
    // The number of buffers per tile.
    int BUFFERS_PER_TILE
>
struct Smem_tile_b<Ampere_hmma_bf16_traits<lwtlass::float_bf16_t, Output_type>,
                   Cta_tile,
                   Row,
                   BYTES_PER_STS,
                   BUFFERS_PER_TILE>
    : public Smem_tile_ampere_row_b<Ampere_hmma_bf16_traits<lwtlass::float_bf16_t, Output_type>,
                                    Cta_tile,
                                    BYTES_PER_STS,
                                    BUFFERS_PER_TILE> {

    // The traits class.
    using Traits = Ampere_hmma_bf16_traits<lwtlass::float_bf16_t, Output_type>;
    // The base class.
    using Base = Smem_tile_ampere_row_b<Traits, Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE>;

    // Ctor.
    inline __device__ Smem_tile_b(void *smem, int tidx) : Base(smem, tidx) {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Cta_tile, bool IN_CTA_SPLIT_K >
struct Swizzle_epilogue<Ampere_hmma_bf16_traits<lwtlass::float_bf16_t, lwtlass::float_bf16_t>,
                        Cta_tile,
                        Row,
                        16,
                        IN_CTA_SPLIT_K>
    : public Swizzle_turing_hmma_fp32_epilogue<Ampere_hmma_bf16_traits<lwtlass::float_bf16_t,
                                                                       lwtlass::float_bf16_t>,
                                               Cta_tile,
                                               Row> {

    // The traits.
    using Traits = Ampere_hmma_bf16_traits<lwtlass::float_bf16_t, lwtlass::float_bf16_t>;
    // The base class.
    using Base = Swizzle_turing_hmma_fp32_epilogue<Traits, Cta_tile, Row>;

    // Ctor.
    inline __device__ Swizzle_epilogue(void *smem, int tidx) : Base(smem, tidx) {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Cta_tile, bool IN_CTA_SPLIT_K >
struct Swizzle_epilogue<Ampere_hmma_bf16_traits<lwtlass::float_bf16_t, float>,
                        Cta_tile,
                        Row,
                        16,
                        IN_CTA_SPLIT_K>
    : public Swizzle_ampere_hmma_tf32_epilogue<Ampere_hmma_bf16_traits<lwtlass::float_bf16_t,
                                                                       float>,
                                               Cta_tile,
                                               Row,
                                               16> {
    // The traits.
    using Traits = Ampere_hmma_bf16_traits<lwtlass::float_bf16_t, float>;
    // The base class.
    using Base = Swizzle_ampere_hmma_tf32_epilogue<Traits, Cta_tile, Row, 16>;

    // Ctor.
    inline __device__ Swizzle_epilogue(void *smem, int tidx) : Base(smem, tidx) {
    }
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
struct Smem_tile_a<Ampere_imma_int8_int32_traits<IS_GELU_ERF>, Cta_tile, Row, BYTES_PER_STS, BUFFERS_PER_TILE>
    : public Smem_tile_ampere_row_a<Ampere_imma_int8_int32_traits<IS_GELU_ERF>,
                                    Cta_tile,
                                    BYTES_PER_STS,
                                    BUFFERS_PER_TILE> {

    // The traits class.
    using Traits = Ampere_imma_int8_int32_traits<IS_GELU_ERF>;
    // The base class.
    using Base = Smem_tile_ampere_row_a<Ampere_imma_int8_int32_traits<IS_GELU_ERF>,
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
struct Smem_tile_b<Ampere_imma_int8_int32_traits<IS_GELU_ERF>, Cta_tile, Col, BYTES_PER_STS, BUFFERS_PER_TILE>
    : public Smem_tile_ampere_col_b<Ampere_imma_int8_int32_traits<IS_GELU_ERF>,
                                    Cta_tile,
                                    BYTES_PER_STS,
                                    BUFFERS_PER_TILE> {

    // The traits class.
    using Traits = Ampere_imma_int8_int32_traits<IS_GELU_ERF>;
    // The base class.
    using Base = Smem_tile_ampere_col_b<Ampere_imma_int8_int32_traits<IS_GELU_ERF>,
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

template< typename Cta_tile, bool IN_CTA_SPLIT_K, bool IS_GELU_ERF >
struct Swizzle_epilogue<Ampere_imma_int8_int32_traits<IS_GELU_ERF>, Cta_tile, Row, 16, IN_CTA_SPLIT_K>
    : public Swizzle_ampere_imma_int32_epilogue<Ampere_imma_int8_int32_traits<IS_GELU_ERF>, Cta_tile> {

    // The traits.
    using Traits = Ampere_imma_int8_int32_traits<IS_GELU_ERF>;
    // The base class.
    using Base = Swizzle_ampere_imma_int32_epilogue<Traits, Cta_tile>;

    // Ctor.
    inline __device__ Swizzle_epilogue(void *smem, int tidx) : Base(smem, tidx) {
    }
};


////////////////////////////////////////////////////////////////////////////////////////////////////
//
// I M M A . 8 W/O Swizzle Epilogue
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
struct Smem_tile_a<Ampere_imma_wo_epi_swizzle_int8_int32_traits, Cta_tile, Row, BYTES_PER_STS, BUFFERS_PER_TILE> 
    : public Smem_tile_ampere_row_a<Ampere_imma_wo_epi_swizzle_int8_int32_traits, 
                                    Cta_tile, 
                                    BYTES_PER_STS,
                                    BUFFERS_PER_TILE> {

    // The traits class.
    using Traits = Ampere_imma_wo_epi_swizzle_int8_int32_traits;
    // The base class.
    using Base = Smem_tile_ampere_row_a<Traits, 
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
struct Smem_tile_b<Ampere_imma_wo_epi_swizzle_int8_int32_traits, Cta_tile, Col, BYTES_PER_STS, BUFFERS_PER_TILE>
    : public Smem_tile_ampere_col_b<Ampere_imma_wo_epi_swizzle_int8_int32_traits, 
                                    Cta_tile, 
                                    BYTES_PER_STS,
                                    BUFFERS_PER_TILE> {

    // The traits class.
    using Traits = Ampere_imma_wo_epi_swizzle_int8_int32_traits;
    // The base class.
    using Base = Smem_tile_ampere_col_b<Traits, 
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

template< typename Traits, typename Cta_tile, typename Layout >
struct Swizzle_epilogue_bypass 
    : public Swizzle_epilogue_empty <Traits, Cta_tile, Layout> {

    // Skip sync in epilogue
    enum { SKIP_SYNCTHREADS = 1 };
    // To reuse implicit_gemm params.h
    enum { BYTES_PER_TILE = 0};
    
    // The base class.
    using Base = Swizzle_epilogue_empty<Traits, Cta_tile, Layout>;
    // Ctor.
    inline __device__ Swizzle_epilogue_bypass(void *smem, int tidx) : Base(smem, tidx) {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// I M M A  I N T E R L E A V E (NC/32HW32)
//
////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Traits, typename Cta_tile, int BITS_PER_ELEMENT, int BUFFERS_PER_TILE >
struct Smem_tile_ampere_imma_col_interleaved_a
    : public Smem_tile_interleaved<Cta_tile,
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

    // The size of a single STSS in bytes.
    enum { BYTES_PER_STS = 16 };
    // The number of elements that are stored with a single STS.
    enum { ELEMENTS_PER_STS = BYTES_PER_STS * 8 / BITS_PER_ELEMENT };
    // Interleaved elements.
    enum { ELEMENTS_PER_PACKET = 32 };
    // Bytes per packed.
    enum { BYTES_PER_PACKET = 32 };
    // The number of channels that are needed.
    enum { CHANNELS = Cta_tile::K / ELEMENTS_PER_PACKET };
    // The number of threads per channel.
    enum { THREADS_PER_CHANNEL = Cta_tile::THREADS_PER_CTA / CHANNELS };
    // The number of bytes per channel.
    enum { BYTES_PER_CHANNEL = Cta_tile::M * BYTES_PER_PACKET };
    // Bytes per smem cache line.
    enum { BYTES_PER_SMEM_LINE = 128 };
    // Threads per smem cache linevfor STS.
    enum { THREADS_PER_SMEM_LINE = BYTES_PER_SMEM_LINE / BYTES_PER_STS };
    // The numbder of matrix rows per smem cache line.
    enum { ROWS_PER_SMEM_LINE = BYTES_PER_SMEM_LINE / ELEMENTS_PER_PACKET };
    // The size in bytes in shared memory.
    enum { BYTES_PER_TILE = CHANNELS * BYTES_PER_CHANNEL * BUFFERS_PER_TILE};
    // Smem stride between two xmma
    enum { LDSM_STRIDE_PER_XMMA = Xmma_tile::M_PER_XMMA_PER_CTA * BYTES_PER_PACKET };
    // LDSM stride
    enum { LDSM_STRIDE_IN_XMMA = 16 * Cta_tile::THREADS_PER_WARP };

    // Ctor.
    inline __device__ Smem_tile_ampere_imma_col_interleaved_a(void *smem, int tidx)
        : Base(smem, tidx) {

        // The number of warps.
        const int WARPS_M = Cta_tile::WARPS_M;
        const int WARPS_N = Cta_tile::WARPS_N;
        const int WARPS_K = Cta_tile::WARPS_K;

        // The masks to select the warps.
        const int WARP_MASK_N = Warp_masks<WARPS_M, WARPS_N, WARPS_K>::N;

        // The divisor for the warps.
        const int WARP_DIV_N = WARPS_M *       1 * Cta_tile::THREADS_PER_WARP;

        // This is for imma.16832 lwrrently.
        int warp_id = tidx / Cta_tile::THREADS_PER_WARP;
        int lane_id = tidx % Cta_tile::THREADS_PER_WARP;

        int smem_read_row_id = (lane_id % 16) / ROWS_PER_SMEM_LINE ;
        int smem_read_col_id = lane_id % ROWS_PER_SMEM_LINE * 2 + lane_id / 16;

        int swizzle_smem_read_col_id = smem_read_col_id ^ (smem_read_row_id & 1);

        this->smem_read_offset_ = smem_read_row_id * BYTES_PER_SMEM_LINE +
                                  swizzle_smem_read_col_id * 16 +
                                  (warp_id % Cta_tile::WARPS_M) *
                                      (BYTES_PER_PACKET * Xmma_tile::M_PER_XMMA);

        if ( Cta_tile::GROUPS > 1 )
            this->smem_read_offset_ += (tidx & WARP_MASK_N) / WARP_DIV_N
                * (Cta_tile::K / ELEMENTS_PER_PACKET / WARPS_N) * BYTES_PER_CHANNEL;

        int smem_write_channel = tidx / THREADS_PER_CHANNEL;
        int smem_write_tid = tidx % THREADS_PER_CHANNEL;

        int smem_write_row_id = smem_write_tid / THREADS_PER_SMEM_LINE;
        int smem_write_col_id = smem_write_tid % THREADS_PER_SMEM_LINE;

        int swizzle_smem_write_col_id = smem_write_col_id ^ (smem_write_row_id & 1);

        // The location where the thread writes its elements.
        this->smem_write_offset_ = smem_write_channel * BYTES_PER_CHANNEL +
                                   smem_write_row_id * BYTES_PER_SMEM_LINE +
                                   swizzle_smem_write_col_id * BYTES_PER_STS;
    }

    // Rewind smem_read_offset for last LDS phase in main loop.
    inline __device__ void reverse_smem_read_offset(int ki = 0) {
    }

    // Load from shared memory.
    inline __device__ void load(Fragment (&a)[Xmma_tile::XMMAS_M], int ki) {
        #pragma unroll
        for( int mi = 0; mi < Xmma_tile::XMMAS_M; ++mi ) {
            #pragma unroll
            for( int i = 0; i < Xmma_tile::M_PER_XMMA / 16; ++i ) {
                int offset = this->smem_read_offset_ + this->smem_read_buffer_ +
                             ki * BYTES_PER_CHANNEL +
                             mi * LDSM_STRIDE_PER_XMMA +
                             i * LDSM_STRIDE_IN_XMMA;
                uint4 tmp;
                ldsm(tmp, this->smem_ + offset);
                a[mi].reg(i * 4 + 0) = tmp.x;
                a[mi].reg(i * 4 + 1) = tmp.y;
                a[mi].reg(i * 4 + 2) = tmp.z;
                a[mi].reg(i * 4 + 3) = tmp.w;
            } // end i
        } // end mi
    }
    // Print the content of the tile (only for debug ;)).
    inline __device__ void debug_print() const {
        for( int row = 0; row < CHANNELS; ++row ) {
            for( int col = 0; col < BYTES_PER_CHANNEL; col += 4 ) {
                if( threadIdx.x == 0 ) {
                    uint32_t val;
                    lds(val, this->smem_ + row*BYTES_PER_CHANNEL + col);
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
                + ii * THREADS_PER_CHANNEL * BYTES_PER_STS;
            ptrs[ii] = this->smem_ + offset;
        }
    }

    // Compute the store pointers.
    template< int N, int PHASE >
    inline __device__ void compute_store_pointers_per_phase(uint32_t (&ptrs)[N]) {
        #pragma unroll
        for( int ii = N * PHASE; ii < N * (PHASE + 1); ++ii ) {
            int offset = this->smem_write_offset_ + this->smem_write_buffer_
                + ii * THREADS_PER_CHANNEL * BYTES_PER_STS;
            ptrs[ii - N * PHASE] = this->smem_ + offset;
        }
    }
    // Store to the tile in shared memory.
    template< int N >
    inline __device__ void store(const uint4 (&data)[N]) {
        uint32_t smem_ptrs[N];
        this->compute_store_pointers(smem_ptrs);
        sts(smem_ptrs, data);
    }

    template< int N >
    inline __device__ void store(const void* (&gmem_ptrs)[N],
                                 uint32_t preds,
                                 uint64_t mem_desc = MEM_DESC_DEFAULT) {
        uint32_t smem_ptrs[N];
        this->compute_store_pointers(smem_ptrs);
        ldgsts(smem_ptrs, gmem_ptrs, preds, mem_desc);
    }

    // Store to the tile in shared memory.
    template< int N, int M, int K = 1, typename LDGSTS_CFG = xmma::Ldgsts_config<true> >
    inline __device__ void store(const void* (&gmem_ptrs)[N],
                                 uint32_t (&preds)[M],
                                 uint64_t mem_desc = MEM_DESC_DEFAULT) {
        uint32_t smem_ptrs[N];
        this->compute_store_pointers(smem_ptrs);
        ldgsts<N, M, 16/K, LDGSTS_CFG>(smem_ptrs, gmem_ptrs, preds, mem_desc);
    }
    
    // Store to the tile in shared memory.
    template< int N, int M, int PHASE, int K = 1, typename LDGSTS_CFG = xmma::Ldgsts_config<true> >
    inline __device__ void store_per_phase(const void* (&gmem_ptrs)[N],
                                 uint32_t (&preds)[M],
                                 uint64_t mem_desc = MEM_DESC_DEFAULT) {
        uint32_t smem_ptrs[N];
        this->compute_store_pointers_per_phase<N, PHASE>(smem_ptrs);
        ldgsts_per_phase<N, M, PHASE, 16/K, LDGSTS_CFG>(smem_ptrs, gmem_ptrs, preds, mem_desc);
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Traits, typename Cta_tile, int BITS_PER_ELEMENT, int BUFFERS_PER_TILE >
struct Smem_tile_ampere_imma_row_interleaved_b
    : public Smem_tile_interleaved<Cta_tile,
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

    // The size of a single STSS in bytes.
    enum { BYTES_PER_STS = 16 };
    // The number of elements that are stored with a single STS.
    enum { ELEMENTS_PER_STS = BYTES_PER_STS * 8 / BITS_PER_ELEMENT };
    // Interleaved elements.
    enum { ELEMENTS_PER_PACKET = 32 };
    // Bytes per packed.
    enum { BYTES_PER_PACKET = 32 };
    // The number of channels that are needed.
    enum { CHANNELS = Cta_tile::K / ELEMENTS_PER_PACKET };
    // The number of threads per channel.
    enum { THREADS_PER_CHANNEL = Cta_tile::THREADS_PER_CTA / CHANNELS };
    // The number of bytes per channel.
    enum { BYTES_PER_CHANNEL = Cta_tile::N * BYTES_PER_PACKET };
    // Bytes per smem cache line.
    enum { BYTES_PER_SMEM_LINE = 128 };
    // Threads per smem cache linevfor STS.
    enum { THREADS_PER_SMEM_LINE = BYTES_PER_SMEM_LINE / BYTES_PER_STS };
    // The numbder of matrix rows per smem cache line.
    enum { ROWS_PER_SMEM_LINE = BYTES_PER_SMEM_LINE / ELEMENTS_PER_PACKET };
    // The size in bytes in shared memory.
    enum { BYTES_PER_TILE = Cta_tile::GROUPS < 4 ? CHANNELS * BYTES_PER_CHANNEL * BUFFERS_PER_TILE
            : BYTES_PER_CHANNEL * BUFFERS_PER_TILE};
    // The size for each smem buffer
    enum { BYTES_PER_BUFFER = Cta_tile::GROUPS < 4 ? CHANNELS * BYTES_PER_CHANNEL
        : BYTES_PER_CHANNEL};
    // The inc boundary
    enum { BYTES_PER_TILE_INC_BOUNDARY = BYTES_PER_TILE - BYTES_PER_BUFFER };
    // Smem stride between two xmma
    enum { LDSM_STRIDE_PER_XMMA = Cta_tile::GROUPS > 1 ? Xmma_tile::N_PER_XMMA * BYTES_PER_PACKET
        : Xmma_tile::N_PER_XMMA_PER_CTA * BYTES_PER_PACKET };
    // LDSM stride
    enum { LDSM_STRIDE_IN_XMMA = 16 * Cta_tile::THREADS_PER_WARP };

    // STS_PER_THREAD
    enum { STS_PER_THREAD = BYTES_PER_CHANNEL / THREADS_PER_CHANNEL / BYTES_PER_STS};
    // Ctor.
    inline __device__ Smem_tile_ampere_imma_row_interleaved_b(void *smem, int tidx)
        : Base(smem, tidx) {

        // The number of warps.
        const int WARPS_M = Cta_tile::WARPS_M;
        const int WARPS_N = Cta_tile::WARPS_N;
        const int WARPS_K = Cta_tile::WARPS_K;

        // The masks to select the warps.
        const int WARP_MASK_N = Warp_masks<WARPS_M, WARPS_N, WARPS_K>::N;

        // The divisor for the warps.
        const int WARP_DIV_N = WARPS_M *       1 * Cta_tile::THREADS_PER_WARP;

        // This is for imma.16832 lwrrently.
        int lane_id = tidx % Cta_tile::THREADS_PER_WARP;

        int smem_read_row_id = (lane_id % 8 + lane_id / 16 * 8) / ROWS_PER_SMEM_LINE ;
        int smem_read_col_id = lane_id % ROWS_PER_SMEM_LINE * 2 + (lane_id / 8) % 2;
        if (Cta_tile::GROUPS >= 4){
            smem_read_row_id = lane_id / ROWS_PER_SMEM_LINE;
            smem_read_col_id = lane_id % ROWS_PER_SMEM_LINE * 2;

        }

        int swizzle_smem_read_col_id = smem_read_col_id ^ (smem_read_row_id & 1);

        this->smem_read_offset_ = smem_read_row_id * BYTES_PER_SMEM_LINE +
                                  swizzle_smem_read_col_id * 16 +
                                  (tidx & WARP_MASK_N) / WARP_DIV_N *
                                  (Cta_tile::GROUPS > 1 ? Cta_tile::N / WARPS_N : Xmma_tile::N_PER_XMMA)
                                  * BYTES_PER_PACKET;

        int smem_write_channel = tidx / THREADS_PER_CHANNEL;
        int smem_write_tid = tidx % THREADS_PER_CHANNEL;

        int smem_write_row_id = smem_write_tid / THREADS_PER_SMEM_LINE;
        int smem_write_col_id = smem_write_tid % THREADS_PER_SMEM_LINE;

        int swizzle_smem_write_col_id = smem_write_col_id ^ (smem_write_row_id & 1);

        // The location where the thread writes its elements.
        this->smem_write_offset_ = smem_write_channel * BYTES_PER_CHANNEL +
                                   smem_write_row_id * BYTES_PER_SMEM_LINE +
                                   swizzle_smem_write_col_id * BYTES_PER_STS;

        if ( Cta_tile::GROUPS >=4){
            int per_group_k = Cta_tile::K / Cta_tile::GROUPS;
            // for C/K = 16 use ldgsts.128, for c/k=8 use ldgsts.64 for c/k = 4, use ldgsts.32
            // FIXME: hack here, in NC32HW32 Gmem tile, when we callwlate address,
            //  suppose we use ldg.128, so each packet(col) use 2 threads

            // fill zero since we don't use LDG PNZ
            if (smem_write_channel == 0){
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
            }
            // find the origin k
            static const int n_cdiv32_hw_c32_reorder_col[32] = {
              0,  1,  8,  9, 16, 17, 24, 25,  2,  3, 10, 11, 18, 19, 26, 27,
              4,  5, 12, 13, 20, 21, 28, 29,  6,  7, 14, 15, 22, 23, 30, 31
            };
            int gmem_col_id = smem_write_tid % 32;
            int xform_k = n_cdiv32_hw_c32_reorder_col[gmem_col_id];

            group_read_id = n_cdiv32_hw_c32_reorder_col[lane_id / 4] / per_group_k;
            // normal to [0,1]
            // every read row distance is 16
            // per_group_k   16  8    4
            // row0          0   0,1  0,1,2,3
            // row1          1   2,3  4,5,6,7
            // divisor       1   2    4
            group_read_id = group_read_id / (16 / per_group_k);

            // for each warp move b using diagonal pattern,
            int group_id = xform_k / per_group_k;
            // offset inside 16bytes
            int group_offset = 0;
            // for per_group_k == 16, exchange in 16bytes range(STS bytes = 128)
            // c = k = 16
            // g0 g1    g0 x
            // x  x  => x g1

            // for per_group_k == 8, (g0,g2),(g1,g3) exchange in 16 (g0,g1),(g2,g3) exchange in 8bytes range
            // c = k = 8 (STS 64)
            // g0 g1 g2 g3   g0 x
            // x  x  x  x => x g1
            // x  x  x  x    x  x g2
            // x  x  x  x => x  x   g3
            if(Cta_tile::GROUPS == 8) {
                group_offset = (group_id % 2) * 8;
            }
            // c = k = 32 (STS 32)
            // g0 g1 g2 g3 g4 g5 g6 g7  g0 x
            // x  x  x  x  x  x  x  x=> x g1
            // x  x  x  x  x  x  x  x   x  x g2
            // x  x  x  x  x  x  x  x=> x  x   g3
            // x  x  x  x  x  x  x  x=> x  x     g4
            // x  x  x  x  x  x  x  x=> x  x       g5
            // x  x  x  x  x  x  x  x=> x  x         g6
            // x  x  x  x  x  x  x  x=> x  x           g7
            if(Cta_tile::GROUPS == 16) {
                group_offset = ((group_id & 0x2) / 2)  * 8;
                group_offset += (group_id % 2)  * 4;
            }
            smem_write_row_id =  smem_write_tid / (THREADS_PER_SMEM_LINE / 2);
            smem_write_col_id = (smem_write_tid % (THREADS_PER_SMEM_LINE / 2)) * 2;

            swizzle_smem_write_col_id = smem_write_col_id ^ (smem_write_row_id & 1);
            // The location where the thread writes its elements.
            this->smem_write_offset_ = smem_write_channel * BYTES_PER_CHANNEL +
                                       smem_write_row_id * BYTES_PER_SMEM_LINE +
                                       swizzle_smem_write_col_id * BYTES_PER_STS + group_offset;
        }
    }

    // Rewind smem_read_offset for last LDS phase in main loop.
    inline __device__ void reverse_smem_read_offset(int ki = 0) {
    }
    // Move the read offset to next buffer.
    inline __device__ void move_to_next_read_buffer() {
        if( BUFFERS_PER_TILE > 1 && this->smem_read_buffer_ >= BYTES_PER_TILE_INC_BOUNDARY ) {
            this->smem_read_buffer_ -= BYTES_PER_TILE_INC_BOUNDARY;
        } else if( BUFFERS_PER_TILE > 1 ) {
            this->smem_read_buffer_ += BYTES_PER_BUFFER;
        }
    }
    // Move the read offset to next buffer. TODO: Remove this member function!!!
    inline __device__ void move_next_read_buffer() {
        this->move_to_next_read_buffer();
    }

    // Move the read offset to next N buffer (cirlwlar-buffer).
    inline __device__ void move_to_next_read_buffer(int N) {
        if( BUFFERS_PER_TILE > 1 ) {
            this->smem_read_buffer_ += N * BYTES_PER_BUFFER;
            this->smem_read_buffer_ -= this->smem_read_buffer_ >= BYTES_PER_TILE ? BYTES_PER_TILE : 0;
        }
    }
    // Move the read offset to next N buffer (cirlwlar-buffer). TODO: Remove this member function!!!
    inline __device__ void move_next_read_buffer(int N) {
        this->move_to_next_read_buffer(N);
    }

    inline __device__ void move_to_next_write_buffer(){
        if (BITS_PER_ELEMENT == 16) {
             this->smem_write_offset_ +=
                (this->smem_write_offset_ >= BYTES_PER_TILE_INC_BOUNDARY )
                ? -BYTES_PER_TILE_INC_BOUNDARY
                : BYTES_PER_BUFFER ;
        }else{
            if( BUFFERS_PER_TILE > 1 && this->smem_write_buffer_ >= BYTES_PER_TILE_INC_BOUNDARY ) {
                this->smem_write_buffer_ -= BYTES_PER_TILE_INC_BOUNDARY;
            } else if( BUFFERS_PER_TILE > 1 ) {
                this->smem_write_buffer_ += BYTES_PER_BUFFER;
            }
        }
    }

    // Move the write offset to next buffer. TODO: Remove that member function!
    inline __device__ void move_next_write_buffer() {
        this->move_to_next_write_buffer();
    }
    // Load from shared memory.
    inline __device__ void load(Fragment (&b)[Xmma_tile::XMMAS_N], int ki) {
        #pragma unroll
        for( int ni = 0; ni < Xmma_tile::XMMAS_N; ++ni ) {
            #pragma unroll
            for( int j = 0; j < Xmma_tile::N_PER_XMMA / 16; ++j ) {
                int offset = this->smem_read_offset_ + this->smem_read_buffer_ +
                             ki * BYTES_PER_CHANNEL +
                             ni * LDSM_STRIDE_PER_XMMA +
                             j * LDSM_STRIDE_IN_XMMA;

                // use ldsm.88.2 to reduce mio for group colw
                if(Cta_tile::GROUPS >= 4){
                    uint2 tmp;
                    ldsm(tmp, this->smem_ + offset);
                    if(group_read_id == 0){
                        b[ni].reg(j * 4 + 0) = tmp.x;
                        b[ni].reg(j * 4 + 1) = 0;
                        b[ni].reg(j * 4 + 2) = tmp.y;
                        b[ni].reg(j * 4 + 3) = 0;
                    }
                    else{
                        b[ni].reg(j * 4 + 0) = 0;
                        b[ni].reg(j * 4 + 1) = tmp.x;
                        b[ni].reg(j * 4 + 2) = 0;
                        b[ni].reg(j * 4 + 3) = tmp.y;
                    }
                }
                else{
                    uint4 tmp;
                    ldsm(tmp, this->smem_ + offset);
                    b[ni].reg(j * 4 + 0) = tmp.x;
                    b[ni].reg(j * 4 + 1) = tmp.y;
                    b[ni].reg(j * 4 + 2) = tmp.z;
                    b[ni].reg(j * 4 + 3) = tmp.w;
                }
            } // end j
        } // end ni
    }

    // Print the content of the tile (only for debug ;)).
    inline __device__ void debug_print() const {
        for( int row = 0; row < CHANNELS; ++row ) {
            for( int col = 0; col < BYTES_PER_CHANNEL; col += 4 ) {
                if( threadIdx.x == 0 ) {
                    uint32_t val;
                    lds(val, this->smem_ + row*BYTES_PER_CHANNEL + col);
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
                + ii * THREADS_PER_CHANNEL * BYTES_PER_STS;
            ptrs[ii] = this->smem_ + offset;
        }
    }
    // Compute the store pointers.
    template< int N, int PHASE >
    inline __device__ void compute_store_pointers_per_phase(uint32_t (&ptrs)[N]) {
        #pragma unroll
        for( int ii = N * PHASE; ii < N * (PHASE + 1); ++ii ) {
            int offset = this->smem_write_offset_ + this->smem_write_buffer_
                + ii * THREADS_PER_CHANNEL * BYTES_PER_STS;
            ptrs[ii - N * PHASE] = this->smem_ + offset;
        }
    }

    // Store to the tile in shared memory.
    template< int N >
    inline __device__ void store(const uint4 (&data)[N]) {
        uint32_t smem_ptrs[N];
        this->compute_store_pointers(smem_ptrs);
        sts(smem_ptrs, data);
    }

    template< int N >
    inline __device__ void store(const void* (&gmem_ptrs)[N],
                                 uint32_t preds,
                                 uint64_t mem_desc = MEM_DESC_DEFAULT) {
        uint32_t smem_ptrs[N];
        this->compute_store_pointers(smem_ptrs);
        ldgsts(smem_ptrs, gmem_ptrs, preds, mem_desc);
    }

    // Store to the tile in shared memory.
    template< int N, int M, int K = 1, typename LDGSTS_CFG = xmma::Ldgsts_config<true> >
    inline __device__ void store(const void* (&gmem_ptrs)[N],
                                 uint32_t (&preds)[M],
                                 uint64_t mem_desc = MEM_DESC_DEFAULT) {
        uint32_t smem_ptrs[N];
        this->compute_store_pointers(smem_ptrs);
        if (Cta_tile::GROUPS >= 4){
            constexpr int SZ = Cta_tile::K / Cta_tile::GROUPS;
            ldgsts<N, M, SZ/K, xmma::Ldgsts_config<false>>(smem_ptrs, gmem_ptrs, preds, mem_desc);
        }
        else {
            ldgsts<N, M, 16/K, LDGSTS_CFG>(smem_ptrs, gmem_ptrs, preds, mem_desc);
        }
    }
    
    // Store to the tile in shared memory.
    template< int N, int M, int PHASE, int K = 1, typename LDGSTS_CFG = xmma::Ldgsts_config<true> >
    inline __device__ void store_per_phase(const void* (&gmem_ptrs)[N],
                                 uint32_t (&preds)[M],
                                 uint64_t mem_desc = MEM_DESC_DEFAULT) {
        uint32_t smem_ptrs[N];
        this->compute_store_pointers_per_phase<N, PHASE>(smem_ptrs);
        if (Cta_tile::GROUPS >= 4){
            constexpr int SZ = Cta_tile::K / Cta_tile::GROUPS;
            ldgsts_per_phase<N, M, PHASE, SZ/K, xmma::Ldgsts_config<false>>(smem_ptrs, gmem_ptrs, preds, mem_desc);
        }
        else {
            ldgsts_per_phase<N, M, PHASE, 16/K, LDGSTS_CFG>(smem_ptrs, gmem_ptrs, preds, mem_desc);
        }
    }
    int group_read_id; // 0 reg0 reg2 valid, 1 reg1 reg3 valid
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Cta_tile,
          typename Input_type,
          typename Output_type,
          bool IS_GELU,
          bool IS_EPIFADD,
          bool IS_SWISH,
          bool IS_RT_FUSE,
          int BUFFERS_PER_TILE >
struct Smem_tile_a<Ampere_imma_interleaved_traits<Input_type,
                                                  Output_type,
                                                  IS_GELU,
                                                  IS_EPIFADD,
                                                  IS_SWISH,
                                                  IS_RT_FUSE>,
                   Cta_tile,
                   Col_interleaved,
                   16,
                   BUFFERS_PER_TILE>
    : public Smem_tile_ampere_imma_col_interleaved_a<Ampere_imma_interleaved_traits<Input_type,
                                                                                    Output_type,
                                                                                    IS_GELU,
                                                                                    IS_EPIFADD,
                                                                                    IS_SWISH,
                                                                                    IS_RT_FUSE>,
                                                     Cta_tile,
                                                     8,
                                                     BUFFERS_PER_TILE> {

    // The traits class.
    using Traits = Ampere_imma_interleaved_traits<Input_type,
                                                  Output_type,
                                                  IS_GELU,
                                                  IS_EPIFADD,
                                                  IS_SWISH,
                                                  IS_RT_FUSE>;
    // The base class.
    using Base = Smem_tile_ampere_imma_col_interleaved_a<Traits, Cta_tile, 8, BUFFERS_PER_TILE>;

    // Ctor.
    inline __device__ Smem_tile_a(void *smem, int tidx) : Base(smem, tidx) {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Cta_tile,
          typename Input_type,
          typename Output_type,
          bool IS_GELU,
          bool IS_EPIFADD,
          bool IS_SWISH,
          bool IS_RT_FUSE,
          int BUFFERS_PER_TILE >
struct Smem_tile_b<Ampere_imma_interleaved_traits<Input_type,
                                                  Output_type,
                                                  IS_GELU,
                                                  IS_EPIFADD,
                                                  IS_SWISH,
                                                  IS_RT_FUSE>,
                   Cta_tile,
                   Row_interleaved,
                   16,
                   BUFFERS_PER_TILE>
    : public Smem_tile_ampere_imma_row_interleaved_b<Ampere_imma_interleaved_traits<Input_type,
                                                                                    Output_type,
                                                                                    IS_GELU,
                                                                                    IS_EPIFADD,
                                                                                    IS_SWISH,
                                                                                    IS_RT_FUSE>,
                                                     Cta_tile,
                                                     8,
                                                     BUFFERS_PER_TILE> {
    // The traits class.
    using Traits = Ampere_imma_interleaved_traits<Input_type,
                                                  Output_type,
                                                  IS_GELU,
                                                  IS_EPIFADD,
                                                  IS_SWISH,
                                                  IS_RT_FUSE>;
    // The base class.
    using Base = Smem_tile_ampere_imma_row_interleaved_b<Traits, Cta_tile, 8, BUFFERS_PER_TILE>;

    // Ctor.
    inline __device__ Smem_tile_b(void *smem, int tidx) : Base(smem, tidx) {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    // The instruction traits.
    typename Traits,
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // Do we enable CTA split-k?
    bool = (Cta_tile::WARPS_K > 1)
>
struct Swizzle_epilogue_interleaved_ampere {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Traits, typename Cta_tile >
struct Swizzle_epilogue_interleaved_ampere<Traits, Cta_tile, false> {

    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;

    enum { NUM_REGS = 16 };
    // The size in bytes in shared memory.
    enum { BYTES_PER_TILE = 0 };
    // Do we skip the syncthreads in the epilogue? Of course, not :)
    enum { SKIP_SYNCTHREADS = 1 };
    // Bytes per lds, sts.
    enum { BYTES_PER_LDS = 16, BYTES_PER_STS = 8 };

    // Ctor.
    inline __device__ Swizzle_epilogue_interleaved_ampere(void *smem, int tidx) {
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

    // Storage for the input registers.
    float regs_[Xmma_tile::XMMAS_N * NUM_REGS];
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // Ampere IMMA interleaved traits related parameters
    typename Input_type,
    typename Output_type,
    bool IS_GELU,
    bool IS_EPIFADD,
    bool IS_SWISH,
    bool IS_RT_FUSE,
    // Do we enable CTA split-K?
    bool IN_CTA_SPLIT_K
>
struct Swizzle_epilogue_interleaved<Ampere_imma_interleaved_traits<Input_type,
                                                                   Output_type,
                                                                   IS_GELU,
                                                                   IS_EPIFADD,
                                                                   IS_SWISH,
                                                                   IS_RT_FUSE>,
                                    Cta_tile,
                                    Col_interleaved,
                                    IN_CTA_SPLIT_K>
    : Swizzle_epilogue_interleaved_ampere<Ampere_imma_interleaved_traits<Input_type,
                                                                         Output_type,
                                                                         IS_GELU,
                                                                         IS_EPIFADD,
                                                                         IS_SWISH,
                                                                         IS_RT_FUSE>,
                                          Cta_tile,
                                          IN_CTA_SPLIT_K> {

    using Base = Swizzle_epilogue_interleaved_ampere<Ampere_imma_interleaved_traits<Input_type,
                                                                                    Output_type,
                                                                                    IS_GELU,
                                                                                    IS_EPIFADD,
                                                                                    IS_SWISH,
                                                                                    IS_RT_FUSE>,
                                                     Cta_tile,
                                                     IN_CTA_SPLIT_K>;
        // Ctor.
    inline __device__ Swizzle_epilogue_interleaved(void *smem, int tidx) : Base(smem, tidx) {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// D M M A . F P 6 4
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
struct Smem_tile_a<Ampere_dmma_fp64_traits, Cta_tile, Row, BYTES_PER_STS, BUFFERS_PER_TILE>
    : public Smem_tile_without_skews<Cta_tile, Cta_tile::M, Cta_tile::K,
                                    Ampere_dmma_fp64_traits::BITS_PER_ELEMENT_A,
                                    BYTES_PER_STS,
                                    BUFFERS_PER_TILE,
                                    0, //ENABLE_LDS_FAST_PATH_
                                    4, //ROWS_PER_XOR_PATTERN_
                                    2> {
    // The traits class.
    using Traits = Ampere_dmma_fp64_traits;

    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;
    // The base class.
    using Base = Smem_tile_without_skews<Cta_tile, Cta_tile::M, Cta_tile::K,
                                         Traits::BITS_PER_ELEMENT_A,
                                         BYTES_PER_STS,
                                         BUFFERS_PER_TILE,
                                         0,
                                         4,
                                         2>;
    // The fragment.
    using Fragment = Fragment_a<Traits, Row>;

    // When we use padding to reach a power of two, special care has to be taken.
    using Cta_tile_with_padding = Cta_tile_with_k_with_padding<Traits, Cta_tile>;
    // The number of XMMAs.
    using Xmma_tile_with_padding = typename Traits::template Xmma_tile<Cta_tile_with_padding>;

    // The size of a single LDS in bytes.
    enum { BYTES_PER_LDS = 8 };


    // Ctor.
    inline __device__ Smem_tile_a(void *smem, int tidx) : Base(smem, tidx) {

        // The number of warps.
        const int WARPS_M = Cta_tile::WARPS_M;
        const int WARPS_N = Cta_tile::WARPS_N;
        const int WARPS_K = Cta_tile::WARPS_K;

        // The masks to select the warps.
        const int WARP_MASK_M = Warp_masks<WARPS_M, WARPS_N, WARPS_K>::M;

        // The divisor for the warps.
        const int WARP_DIV_M =       1 *       1 * Cta_tile::THREADS_PER_WARP;

        // The row and column read by the thread.
        int smem_read_row, smem_read_col;

        static_assert( Base::ROWS_PER_XOR_PATTERN == 4, "");

        if( Base::ROWS_PER_XOR_PATTERN == 4 ) {
            smem_read_row = (tidx & WARP_MASK_M) / WARP_DIV_M * Xmma_tile::M_PER_XMMA + (tidx & 0x1f ) / 4; // warp_id * M_PER_XMMA + tidx % warp_thread_num / thread_pre_row
            smem_read_col = tidx & 0x0f; // tidx % 16
        }

        static_assert(WARPS_K <= 2, "");

        // The shared memory offset.
        this->smem_read_offset_ = smem_read_row*Base::BYTES_PER_ROW + smem_read_col*BYTES_PER_LDS;
    }

    // Rewind smem_read_offset for last LDS phase in main loop.
    inline __device__ void reverse_smem_read_offset(int ki = 0) {
        // Undo the pointer increment for the next ni.
        // Should match the load function below for ki = 0.
        if (Xmma_tile_with_padding::XMMAS_K >= 2) {
            this->smem_read_offset_ ^= 4 * BYTES_PER_LDS;
        }
    }

    inline __device__ void load(Fragment (&a)[Xmma_tile::XMMAS_M], int ki) {
        #pragma unroll
        for( int mi = 0; mi < Xmma_tile::XMMAS_M; ++mi ) {
            // Jump by as many matrix rows as needed (a row in smem may pack multiple matrix rows).
            int offset = mi * Xmma_tile::M_PER_XMMA_PER_CTA * Base::BYTES_PER_ROW_BEFORE_PACKING;

            // Load using LDS.64
            uint2 tmp;
            lds(tmp, this->smem_ + this->smem_read_offset_ + this->smem_read_buffer_ + offset);

            // Store the value into the fragment.
            a[mi].reg(0) = tmp.x;
            a[mi].reg(1) = tmp.y;
        }

        static_assert(Xmma_tile_with_padding::XMMAS_K <= 8, "Not implemented");

        if (Xmma_tile_with_padding::XMMAS_K >= 8 && ki % 4 == 3) {
            this->smem_read_offset_ ^=  28 * BYTES_PER_LDS;
        } else if (Xmma_tile_with_padding::XMMAS_K >= 4 && ki % 2 == 1) {
            this->smem_read_offset_ ^=  12 * BYTES_PER_LDS;
        } else if (Xmma_tile_with_padding::XMMAS_K >= 2) {
            this->smem_read_offset_ ^=  4 * BYTES_PER_LDS;
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
struct Smem_tile_a<Ampere_dmma_fp64_traits, Cta_tile, Col, BYTES_PER_STS, BUFFERS_PER_TILE>
    : public Smem_tile_without_skews<Cta_tile,
                                    Cta_tile::K,
                                    Cta_tile::M,
                                    Ampere_dmma_fp64_traits::BITS_PER_ELEMENT_A,
                                    BYTES_PER_STS,
                                    BUFFERS_PER_TILE,
                                    0,
                                    4,
                                    2> {
    // The traits class.
    using Traits = Ampere_dmma_fp64_traits;
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
                                         4,
                                         2>;

    // The fragment.
    using Fragment = Fragment_a<Traits, Col>;

    // The size of a single LDS in bytes.
    enum { BYTES_PER_LDS = 8 };
    // The number of elements per LDS.
    enum { ELEMENTS_PER_LDS = BYTES_PER_LDS * 8 / Traits::BITS_PER_ELEMENT_A };

    // Ctor.
    inline __device__ Smem_tile_a(void *smem, int tidx) : Base(smem, tidx) {

        // The number of warps.
        const int WARPS_M = Cta_tile::WARPS_M;
        const int WARPS_N = Cta_tile::WARPS_N;
        const int WARPS_K = Cta_tile::WARPS_K;

        // The masks to select the warps.
        const int WARP_MASK_M = Warp_masks<WARPS_M, WARPS_N, WARPS_K>::M;

        // The divisor for the warps.
        const int WARP_DIV_M =       1 *       1 * Cta_tile::THREADS_PER_WARP;

        // The row and column read by the thread.
        int smem_read_row, smem_read_col;

        static_assert( Base::ROWS_PER_XOR_PATTERN == 4, "");

        if( Base::ROWS_PER_XOR_PATTERN == 4 ) {
            smem_read_row = tidx & 0x03;
            smem_read_col = (tidx & WARP_MASK_M) / WARP_DIV_M * Xmma_tile::M_PER_XMMA + (tidx & 0x1c) / 4;  // (tidx % 32 % 4) /4
            smem_read_col ^= smem_read_row * 4;
        }

        static_assert(WARPS_K <= 2, "");

        // The shared memory offset.
        this->smem_read_offset_ = smem_read_row*Base::BYTES_PER_ROW + smem_read_col*BYTES_PER_LDS;
    }

    // Rewind smem_read_offset for last LDS phase in main loop.
    inline __device__ void reverse_smem_read_offset(int ki = 0) {
        // Nothing to do here, smem_read_offset doesn't need to get reset.
    }

    // Load from shared memory.
    inline __device__ void load(Fragment (&a)[Xmma_tile::XMMAS_M], int ki) {

        #pragma unroll
        for( int mi = 0; mi < Xmma_tile::XMMAS_M; ++mi ) {
            // Jump by as many matrix rows as needed.
            int offset = mi * Xmma_tile::M_PER_XMMA_PER_CTA * BYTES_PER_LDS + ki * Base::BYTES_PER_ROW *Xmma_tile::K_PER_XMMA;
            // Load using LDS.64
            uint2 tmp;
            lds(tmp, this->smem_ + this->smem_read_offset_ + this->smem_read_buffer_ + offset);

            // Store the value into the fragment.
            a[mi].reg(0) = tmp.x;
            a[mi].reg(1) = tmp.y;

        }
        //smem_read_offset is idential for different K.
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
struct Smem_tile_b<Ampere_dmma_fp64_traits, Cta_tile, Row, BYTES_PER_STS, BUFFERS_PER_TILE >
    : public Smem_tile_without_skews<Cta_tile,
                                    Cta_tile::K,
                                    Cta_tile::N,
                                    Ampere_dmma_fp64_traits::BITS_PER_ELEMENT_B,
                                    BYTES_PER_STS,
                                    BUFFERS_PER_TILE,
                                    0,
                                    4,
                                    2> {

    // The traits class.
    using Traits = Ampere_dmma_fp64_traits;
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
                                         4,
                                         2>;
    // The fragment.
    using Fragment = Fragment_b<Traits, Row>;

    // The size of a single LDS in bytes.
    enum { BYTES_PER_LDS = 8 };
    // The number of elements per LDS.
    enum { ELEMENTS_PER_LDS = BYTES_PER_LDS * 8 / Traits::BITS_PER_ELEMENT_B };

    // Ctor.
    inline __device__ Smem_tile_b(void *smem, int tidx) : Base(smem, tidx) {

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

        static_assert( Base::ROWS_PER_XOR_PATTERN == 4, "");

        if( Base::ROWS_PER_XOR_PATTERN == 4 ) {
            smem_read_row = tidx & 0x03;
            smem_read_col = (tidx & WARP_MASK_N) / WARP_DIV_N * Xmma_tile::N_PER_XMMA + (tidx & 0x1c) / 4;
            smem_read_col ^= smem_read_row * 4;
        }

        static_assert(WARPS_K <= 2, "");

        // The shared memory offset.
        this->smem_read_offset_ = smem_read_row * Base::BYTES_PER_ROW + smem_read_col*BYTES_PER_LDS;

    }

    // Rewind smem_read_offset for last LDS phase in main loop.
    inline __device__ void reverse_smem_read_offset(int ki = 0) {
        // Nothing to do here, smem_read_offset doesn't need to get reset.
    }

    // Load from shared memory.
    inline __device__ void load(Fragment (&b)[Xmma_tile::XMMAS_N], int ki) {

        #pragma unroll
        for( int ni = 0; ni < Xmma_tile::XMMAS_N; ++ni ) {
            // Jump by as many matrix rows as needed.
            int offset = ni * Xmma_tile::N_PER_XMMA_PER_CTA * BYTES_PER_LDS + ki * Base::BYTES_PER_ROW * Xmma_tile::K_PER_XMMA;
            uint2 tmp;
            lds(tmp, this->smem_ + this->smem_read_offset_ + this->smem_read_buffer_ + offset);

            // Store the value into the fragment.
            b[ni].reg(0) = tmp.x;
            b[ni].reg(1) = tmp.y;
        }
        //smem_read_offset is idential for different K.
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
struct Smem_tile_b<Ampere_dmma_fp64_traits, Cta_tile, Col, BYTES_PER_STS, BUFFERS_PER_TILE >
    : public Smem_tile_without_skews<Cta_tile,
                                    Cta_tile::N,
                                    Cta_tile::K,
                                    Ampere_dmma_fp64_traits::BITS_PER_ELEMENT_B,
                                    BYTES_PER_STS,
                                    BUFFERS_PER_TILE,
                                    0,
                                    4,
                                    2>{

    // The traits class.
    using Traits = Ampere_dmma_fp64_traits;

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
                                         4,
                                         2>;
    // The fragment.
    using Fragment = Fragment_b<Traits, Col>;

    // When we use padding to reach a power of two, special care has to be taken.
    using Cta_tile_with_padding = Cta_tile_with_k_with_padding<Traits, Cta_tile>;

    // The number of XMMAs.
    using Xmma_tile_with_padding = typename Traits::template Xmma_tile<Cta_tile_with_padding>;

    // The size of a single LDS in bytes.
    enum { BYTES_PER_LDS = 8 };

    // Ctor.
    inline __device__ Smem_tile_b(void *smem, int tidx) : Base(smem, tidx) {

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

        static_assert(Base::ROWS_PER_XOR_PATTERN == 4, "");

        if( Base::ROWS_PER_XOR_PATTERN == 4 ) {
            smem_read_row = (tidx & WARP_MASK_N) / WARP_DIV_N * Xmma_tile::N_PER_XMMA +  (tidx & 0x1f ) / 4;
            smem_read_col = tidx & 0x0f;
        }

        static_assert(WARPS_K <= 2, "");

        // The shared memory offset.
        this->smem_read_offset_ = smem_read_row*Base::BYTES_PER_ROW + smem_read_col*BYTES_PER_LDS;

    }

    // Rewind smem_read_offset for last LDS phase in main loop.
    inline __device__ void reverse_smem_read_offset(int ki = 0) {
        // Undo the pointer increment for the next ni.
        // Should match the load function below for ki = 0.
        if (Xmma_tile_with_padding::XMMAS_K >= 2) {
            this->smem_read_offset_ ^= 4 * BYTES_PER_LDS;
        }
    }

    // Load from shared memory.
    inline __device__ void load(Fragment (&b)[Xmma_tile::XMMAS_N], int ki) {
        #pragma unroll
        for( int ni = 0; ni < Xmma_tile::XMMAS_N; ++ni ) {
            // Jump by as many matrix rows as needed (a row in smem may pack multiple matrix rows).
            int offset = ni * Xmma_tile::N_PER_XMMA_PER_CTA * Base::BYTES_PER_ROW_BEFORE_PACKING;

            // Load using LDS.64
            uint2 tmp;
            lds(tmp, this->smem_ + this->smem_read_offset_ + this->smem_read_buffer_ + offset);

            // Store the value into the fragment.
            b[ni].reg(0) = tmp.x;
            b[ni].reg(1) = tmp.y;
        }

        static_assert(Xmma_tile_with_padding::XMMAS_K <= 8, "Not implemented");

        if (Xmma_tile_with_padding::XMMAS_K >= 8 && ki % 4 == 3) {
            this->smem_read_offset_ ^=  28 * BYTES_PER_LDS;
        } else if (Xmma_tile_with_padding::XMMAS_K >= 4 && ki % 2 == 1) {
            this->smem_read_offset_ ^=  12 * BYTES_PER_LDS;
        } else if (Xmma_tile_with_padding::XMMAS_K >= 2) {
            this->smem_read_offset_ ^=  4 * BYTES_PER_LDS;
        }
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Cta_tile, bool IN_CTA_SPLIT_K >
struct Swizzle_epilogue<Ampere_dmma_fp64_traits, Cta_tile, Row, 16, IN_CTA_SPLIT_K>
    : public Swizzle_turing_epilogue<Ampere_dmma_fp64_traits, Cta_tile, double> {

    // The traits.
    using Traits = Ampere_dmma_fp64_traits;
    // The base class.
    using Base =  Swizzle_turing_epilogue<Ampere_dmma_fp64_traits, Cta_tile, double>;
    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;
    // Ctor.
    inline __device__ Swizzle_epilogue(void *smem, int tidx) : Base(smem, tidx) {
    }

    // Store to the tile in shared memory.
    template<typename Fragment_pre_swizzle>
    inline __device__ void store(int ni, const Fragment_pre_swizzle &c) {   
        int offset = ni * Xmma_tile::N_PER_XMMA_PER_CTA * sizeof(double);
        uint32_t ptr = this->smem_ + this->smem_write_offset_ + offset;

        sts(ptr, make_uint4(c.reg(0), c.reg(1),c.reg(2),c.reg(3)));
    }


    // Load from the tile in shared memory.
    template<typename Fragment_post_swizzle>
    inline __device__ void load(Fragment_post_swizzle &dst, int oi) const {
        int offset = oi * Base::PIXELS_PER_STG * Base::BYTES_PER_ROW_WITH_SKEW;

        uint4 tmp;
        lds(tmp, this->smem_ + this->smem_read_offset_ + offset);
        dst.reg(0) = tmp.x;
        dst.reg(1) = tmp.y;
        dst.reg(2) = tmp.z;
        dst.reg(3) = tmp.w;
    }
};

} // namespace xmma

