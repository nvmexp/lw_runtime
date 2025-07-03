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

#include <xmma/ext/colw_with_2x2_pooling/fragment.h>
#include <xmma/warp_masks.h>
#include <xmma/volta/traits.h>
#include <xmma/turing/traits.h>
#include <xmma/ampere/traits.h>

namespace xmma {
namespace ext {
namespace colw_with_2x2_pooling {

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Traits, typename Cta_tile, typename Gmem_tile_c >
struct Smem_tile_base_c {

    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;

    // The amount of bytes per channel of a pixel.
    enum { BYTES_PER_CHANNEL = Traits::BITS_PER_ELEMENT_C / 8 };

    // The number of rows in shared memory. One row for odd/even pixels.
    enum { ROWS = 2 };
    // The number of channels per pixel.
    enum { CHANNELS_PER_PIXEL = Gmem_tile_c::CHANNELS_PER_PIXEL }; 
    // The number of bytes needed to store a single pixel.
    enum { BYTES_PER_PIXEL = CHANNELS_PER_PIXEL * Cta_tile::WARPS_K * BYTES_PER_CHANNEL };
    // The number of pixels produced by a CTA-wide XMMA.
    enum { PIXELS_PER_XMMA = Xmma_tile::M_PER_XMMA_PER_CTA };
    // The size of a single row in bytes.
    enum { BYTES_PER_ROW = PIXELS_PER_XMMA / ROWS * BYTES_PER_PIXEL };

    // Volta
    // -----
    // The skew to avoid bank conflicts between odd/even threads. We use STS.64 so we need 64B with 
    // the following pattern:
    //
    // T00 T02 T04 T06 T08 T10 T12 T14 ....
    // XXX XXX XXX XXX XXX XXX XXX XXX T01 T03 T05 T07 T09 T11 T13 T15 ...
    //
    // Turing and sequel
    // -----------------
    // The skew to avoid bank conflicts between odd/even quads. We use STS.32 so we need 64B with
    // the following pattern:
    //
    // Q0 Q2 Q4 Q6....
    // XX XX XX XX Q1 Q3 Q5 Q7 ...
    enum { BYTES_PER_SKEW = 64 };
    // The number of columns with the skew.
    enum { BYTES_PER_ROW_WITH_SKEW = BYTES_PER_ROW + BYTES_PER_SKEW };
    // The size in bytes in shared memory.
    enum { BYTES_PER_TILE = ROWS * BYTES_PER_ROW_WITH_SKEW };

    // Do we skip the syncthreads in the epilogue? Of course, not :)
    enum { SKIP_SYNCTHREADS = 0 };

    // The pre-swizzle fragment.
    using Fragment_pre_swizzle = xmma::Fragment_epilogue_pre_swizzle<Traits, Cta_tile>;
    // The post-swizzle fragment.
    using Fragment_post_swizzle = Fragment_post_swizzle<Traits, Cta_tile, Gmem_tile_c>;

    // Ctor.
    inline __device__ Smem_tile_base_c(void *smem, int tidx) 
        : smem_(xmma::get_smem_pointer(smem)) {

        // The row and column read by a single thread.
        int pq = tidx / Gmem_tile_c::THREADS_PER_PIXEL;

        // If we cover more than 1 row, take it into account.
        int p = pq / Gmem_tile_c::PIXELS_PER_ROW_AFTER_POOLING;
        int q = pq % Gmem_tile_c::PIXELS_PER_ROW_AFTER_POOLING;

        // The column.
        int k  = tidx % Gmem_tile_c::THREADS_PER_PIXEL;

        // The corresponding offset.
        read_offset_ = p * 2 * Gmem_tile_c::PIXELS_PER_ROW_AFTER_POOLING * BYTES_PER_PIXEL + 
                       q * BYTES_PER_PIXEL +
                       k * Gmem_tile_c::BYTES_PER_STG;
    }

    // Print the content of the tile (only for debug ;)).
    inline __device__ void debug_print() const {
        for( int row = 0; row < ROWS; ++row ) {
            for( int col = 0; col < BYTES_PER_ROW; col += 4 ) {
                if( threadIdx.x == 0 ) {
                    uint32_t val;
                    xmma::lds(val, smem_ + row * BYTES_PER_ROW_WITH_SKEW + col);
                    printf("epilogue (row=%3d, byte=%4d)=0x%08x\n", row, col, val);
                }
            }
        }
    }

    // Load from the tile in shared memory.
    inline __device__ void load(Fragment_post_swizzle &dst, int oi) const {
        xmma::Fragment_lds<Gmem_tile_c::BYTES_PER_STG> lds_helper;
        #pragma unroll
        for( int ki = 0; ki < Cta_tile::WARPS_K; ++ki ) {
            #pragma unroll
            for( int pi = 0 ; pi < 2 ; ++pi ){
                #pragma unroll
                for( int qi = 0 ; qi < 2 ; ++qi ){
                    int imm = oi * Gmem_tile_c::PIXELS_PER_STG / ROWS * BYTES_PER_PIXEL + 
                              pi * Gmem_tile_c::PIXELS_PER_ROW / ROWS * BYTES_PER_PIXEL +
                              qi * BYTES_PER_ROW_WITH_SKEW +
                              ki * CHANNELS_PER_PIXEL * BYTES_PER_CHANNEL;
                    lds_helper.lds(dst, ki*4 + pi*2 + qi, smem_ + read_offset_ + imm);
                }
            }
        }
    }

    // The shared memory pointer in bytes.
    uint32_t smem_;
    // The write offset.
    uint32_t write_offset_;
    // The read offset.
    uint32_t read_offset_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Traits, typename Cta_tile, typename Gmem_tile_c >
struct Smem_tile_volta_c : public Smem_tile_base_c<Traits, Cta_tile, Gmem_tile_c> {

    // The base class.
    using Base = Smem_tile_base_c<Traits, Cta_tile, Gmem_tile_c>;

    // The number of bytes per STS.64.
    enum { BYTES_PER_STS = 8 };

    // Ctor.
    inline __device__ Smem_tile_volta_c(void *smem, int tidx) : Base(smem, tidx) {

        // Extract the number of warps.
        enum { WARPS_M = Cta_tile::WARPS_M };
        enum { WARPS_N = Cta_tile::WARPS_N };
        enum { WARPS_K = Cta_tile::WARPS_K };

	// The masks to select the warps.
        enum { WARP_MASK_M = xmma::Warp_masks<WARPS_M, WARPS_N, WARPS_K>::M };
        enum { WARP_MASK_N = xmma::Warp_masks<WARPS_M, WARPS_N, WARPS_K>::N };
        enum { WARP_MASK_K = xmma::Warp_masks<WARPS_M, WARPS_N, WARPS_K>::K };

        // The divisor for the warps.
        enum { WARP_DIV_M =       1 *       1 * Cta_tile::THREADS_PER_WARP };
        enum { WARP_DIV_N = WARPS_M *       1 * Cta_tile::THREADS_PER_WARP };
        enum { WARP_DIV_K = WARPS_M * WARPS_N * Cta_tile::THREADS_PER_WARP };

        // The bytes per channel.
        enum { BYTES_PER_CHANNEL = Base::BYTES_PER_CHANNEL };
        // The number of channels per pixel.
        enum { CHANNELS_PER_PIXEL = Base::CHANNELS_PER_PIXEL };

        // The column written by a warp for a given pixel.
        int col = (tidx & WARP_MASK_K) / WARP_DIV_K * CHANNELS_PER_PIXEL * BYTES_PER_CHANNEL +
                  (tidx & WARP_MASK_N) / WARP_DIV_N * 16                 * BYTES_PER_CHANNEL; 

        // The position of the pixel.
        int pix = (tidx & WARP_MASK_M) / WARP_DIV_M * 16 + (tidx & 0x10) / 2 + (tidx & 0x07);

        // Finalize the column.
        col += pix / 2 * Base::BYTES_PER_PIXEL + (tidx & 0x8) / 8 * BYTES_PER_STS;

        // The corresponding offset.
        this->write_offset_ = (pix & 0x1) * Base::BYTES_PER_ROW_WITH_SKEW + col;
    }

    // Store to the tile in shared memory.
    inline __device__ void store(int ni, const typename Base::Fragment_pre_swizzle &c) {
        #pragma unroll
        for( int ii = 0; ii < 2; ++ii ) {
            int imm = ii * 2 * BYTES_PER_STS + 
                      ni * Base::Xmma_tile::N_PER_XMMA_PER_CTA * Base::BYTES_PER_CHANNEL;
            uint2 tmp;
            tmp.x = c.reg(2*ii + 0);
            tmp.y = c.reg(2*ii + 1);
            xmma::sts(this->smem_ + this->write_offset_ + imm, tmp);
        }
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Traits, typename Cta_tile, typename Gmem_tile_c >
struct Smem_tile_turing_c : public Smem_tile_base_c<Traits, Cta_tile, Gmem_tile_c> {

    // The base class.
    using Base = Smem_tile_base_c<Traits, Cta_tile, Gmem_tile_c>;

    // The number of bytes per STS.32.
    enum { BYTES_PER_STS = 4 };

    // Ctor.
    inline __device__ Smem_tile_turing_c(void *smem, int tidx) : Base(smem, tidx) {
        // Extract the number of warps.
        enum { WARPS_M = Cta_tile::WARPS_M };
        enum { WARPS_N = Cta_tile::WARPS_N };
        enum { WARPS_K = Cta_tile::WARPS_K };

	// The masks to select the warps.
        enum { WARP_MASK_M = xmma::Warp_masks<WARPS_M, WARPS_N, WARPS_K>::M };
        enum { WARP_MASK_N = xmma::Warp_masks<WARPS_M, WARPS_N, WARPS_K>::N };
        enum { WARP_MASK_K = xmma::Warp_masks<WARPS_M, WARPS_N, WARPS_K>::K };

        // The divisor for the warps.
        enum { WARP_DIV_M =       1 *       1 * Cta_tile::THREADS_PER_WARP };
        enum { WARP_DIV_N = WARPS_M *       1 * Cta_tile::THREADS_PER_WARP };
        enum { WARP_DIV_K = WARPS_M * WARPS_N * Cta_tile::THREADS_PER_WARP };

        // The bytes per channel.
        enum { BYTES_PER_CHANNEL = Base::BYTES_PER_CHANNEL };
        // The number of channels per pixel.
        enum { CHANNELS_PER_PIXEL = Base::CHANNELS_PER_PIXEL };

        // The column written by a warp for a given pixel.
        int col = (tidx & WARP_MASK_K) / WARP_DIV_K * CHANNELS_PER_PIXEL * BYTES_PER_CHANNEL +
                  (tidx & WARP_MASK_N) / WARP_DIV_N * 16                 * BYTES_PER_CHANNEL; 

        // The position of the pixel.
        int pix = (tidx & WARP_MASK_M) / WARP_DIV_M * 16 + (tidx & 0x1c) / 4;

        // Finalize the column.
        col += pix / 2 * Base::BYTES_PER_PIXEL + (tidx & 0x3) * BYTES_PER_STS;

        // The corresponding offset.
        this->write_offset_ = (pix & 0x1) * Base::BYTES_PER_ROW_WITH_SKEW + col;
    }

    // Store to the tile in shared memory.
    inline __device__ void store(int ni, const typename Base::Fragment_pre_swizzle &c) {
        #pragma unroll
        for( int ii = 0; ii < 2; ++ii ) {
            int imm = ii * 4 * Base::BYTES_PER_PIXEL + 
                      ni * Base::Xmma_tile::N_PER_XMMA_PER_CTA * Base::BYTES_PER_CHANNEL;
            xmma::sts(this->smem_ + this->write_offset_ + imm +  0, c.reg(2*ii + 0));
            xmma::sts(this->smem_ + this->write_offset_ + imm + 16, c.reg(2*ii + 1));
        }
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Traits, typename Cta_tile, typename Gmem_tile_c >
struct Smem_tile_c {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Cta_tile, typename Gmem_tile_c >
struct Smem_tile_c<xmma::Volta_hmma_fp16_traits, Cta_tile, Gmem_tile_c>
    : public Smem_tile_volta_c<xmma::Volta_hmma_fp16_traits, Cta_tile, Gmem_tile_c> {

    // The traits.
    using Traits = xmma::Volta_hmma_fp16_traits;
    // The base class.
    using Base = Smem_tile_volta_c<Traits, Cta_tile, Gmem_tile_c>;

    // Ctor.
    inline __device__ Smem_tile_c(void *smem, int tidx) : Base(smem, tidx) {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Cta_tile, typename Gmem_tile_c >
struct Smem_tile_c<xmma::Turing_hmma_fp16_traits, Cta_tile, Gmem_tile_c>
    : public Smem_tile_turing_c<xmma::Turing_hmma_fp16_traits, Cta_tile, Gmem_tile_c> {

    // The traits.
    using Traits = xmma::Turing_hmma_fp16_traits;
    // The base class.
    using Base = Smem_tile_turing_c<Traits, Cta_tile, Gmem_tile_c>;

    // Ctor.
    inline __device__ Smem_tile_c(void *smem, int tidx) : Base(smem, tidx) {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Cta_tile, typename Gmem_tile_c >
struct Smem_tile_c<xmma::Ampere_hmma_fp16_traits, Cta_tile, Gmem_tile_c>
    : public Smem_tile_turing_c<xmma::Ampere_hmma_fp16_traits, Cta_tile, Gmem_tile_c> {

    // The traits.
    using Traits = xmma::Ampere_hmma_fp16_traits;
    // The base class.
    using Base = Smem_tile_turing_c<Traits, Cta_tile, Gmem_tile_c>;

    // Ctor.
    inline __device__ Smem_tile_c(void *smem, int tidx) : Base(smem, tidx) {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace colw_with_2x2_pooling
} // namespace ext
} // namespace xmma

