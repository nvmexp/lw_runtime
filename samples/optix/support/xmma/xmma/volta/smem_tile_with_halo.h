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

#include <xmma/smem_tile_with_halo.h>

namespace xmma {

////////////////////////////////////////////////////////////////////////////////////////////////////

template< 
    typename Traits, 
    typename Cta_tile, 
    typename Pixel_tile, 
    typename Halo, 
    int BUFFERS_PER_TILE_ 
>
struct Smem_tile_with_halo_volta_hmma_row_a {

    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;
    // The fragment.
    using Fragment = Fragment_a<Traits, Row>;

    // The number of words in 128b.
    enum { BITS_PER_ELEMENT = 16, BYTES_PER_STS = 16 };
    // The size of the tile with halo in the H dimension.
    enum { PIXELS_PER_TILE_WITH_HALO_H = Pixel_tile::H + Halo::H };
    // The size of the tile with halo in the W dimension.
    enum { PIXELS_PER_TILE_WITH_HALO_W = Pixel_tile::W + Halo::W };
    // The number of pixels that are needed.
    enum { PIXELS = Pixel_tile::N * PIXELS_PER_TILE_WITH_HALO_H * PIXELS_PER_TILE_WITH_HALO_W };
    // The number of threads per pixel.
    enum { THREADS_PER_PIXEL = Cta_tile::K * BITS_PER_ELEMENT / 8 / BYTES_PER_STS };
    // The number of pixels written with a single STS.
    enum { PIXELS_PER_STS = Cta_tile::THREADS_PER_CTA / THREADS_PER_PIXEL };
    // The number of STS.
    enum { STS = (PIXELS + PIXELS_PER_STS-1) / PIXELS_PER_STS };

    // The number of rows in shared memory.
    enum { ROWS = THREADS_PER_PIXEL };
    // The size of a single row in bytes.
    enum { BYTES_PER_ROW = STS * PIXELS_PER_STS * BYTES_PER_STS };
    // The number of bytes written per row of shared memory for each "store".
    enum { BYTES_PER_SKEW = ROWS >= 8 ? BYTES_PER_STS : (8 / ROWS) * BYTES_PER_STS };
    // The number of columns with the skew.
    enum { BYTES_PER_ROW_WITH_SKEW = BYTES_PER_ROW + BYTES_PER_SKEW };
    // The size of one buffer in bytes in shared memory.
    enum { BYTES_PER_BUFFER = ROWS * BYTES_PER_ROW_WITH_SKEW };
    // The number of buffers. 
    enum { BUFFERS_PER_TILE = BUFFERS_PER_TILE_ };
    // The size in bytes in of total buffers.
    enum { BYTES_PER_TILE = BYTES_PER_BUFFER * BUFFERS_PER_TILE };
    // The boundary for smem_read_offset and smem_write_offset increment.
    enum { BYTES_PER_TILE_INC_BOUNDARY = BYTES_PER_TILE - BYTES_PER_BUFFER };

    // Ctor.
    inline __device__ Smem_tile_with_halo_volta_hmma_row_a(void *smem, int tidx) 
        : smem_(get_smem_pointer(smem)) {

        // The number of warps.
        const int WARPS_M = Cta_tile::WARPS_M;
        const int WARPS_N = Cta_tile::WARPS_N;
        const int WARPS_K = Cta_tile::WARPS_K;

        //
        //     tidx   0 and   8: row = 0, col =  0*16B
        //     tidx   1 and   9: row = 0, col =  1*16B
        //     tidx   2 and  10: row = 0, col =  2*16B
        //     tidx   3 and  11: row = 0, col =  3*16B
        //     tidx   4 and  12: row = 0, col =  4*16B
        //     tidx   5 and  13: row = 0, col =  5*16B
        //     tidx   6 and  14: row = 0, col =  6*16B
        //     tidx   7 and  15: row = 0, col =  7*16B
        //     tidx  16 and  24: row = 0, col =  8*16B
        //     tidx  17 and  25: row = 0, col =  9*16B
        //     tidx  18 and  26: row = 0, col = 10*16B
        //     tidx  19 and  27: row = 0, col = 11*16B
        //     tidx  20 and  28: row = 0, col = 12*16B
        //     tidx  21 and  29: row = 0, col = 13*16B
        //     tidx  22 and  30: row = 0, col = 14*16B
        //     tidx  23 and  31: row = 0, col = 15*16B
        //

        // The masks to select the warps.
        const int WARP_MASK_M = Warp_masks<WARPS_M, WARPS_N, WARPS_K>::M;
        const int WARP_MASK_K = Warp_masks<WARPS_M, WARPS_N, WARPS_K>::K;

        // The divisor for the warps.
        const int WARP_DIV_M =       1 *       1 * Cta_tile::THREADS_PER_WARP;
        const int WARP_DIV_K = WARPS_M * WARPS_N * Cta_tile::THREADS_PER_WARP;
        
        // Compute the row and the column in shared memory. 
        const int smem_read_row = (tidx & WARP_MASK_K) / WARP_DIV_K * Xmma_tile::XMMAS_K;
        const int smem_read_col = (tidx & WARP_MASK_M) / WARP_DIV_M * 16 + 
                                  (tidx & 0x10) / 2 + 
                                  (tidx & 0x07);

        // The size of the HW tile.
        const int PIXELS_PER_TILE_HW = Pixel_tile::H * Pixel_tile::W; 

        // The column has to be mapped to the location in the tile.
        int smem_read_n = smem_read_col / PIXELS_PER_TILE_HW;
        int smem_read_h = smem_read_col % PIXELS_PER_TILE_HW / Pixel_tile::W;
        int smem_read_w = smem_read_col % PIXELS_PER_TILE_HW % Pixel_tile::W;

        // The offset in the tile.
        int smem_read_pix = smem_read_n * PIXELS_PER_TILE_WITH_HALO_H*PIXELS_PER_TILE_WITH_HALO_W +
                            smem_read_h * PIXELS_PER_TILE_WITH_HALO_W +
                            smem_read_w;

        // The corresponding offset.
        this->smem_read_offset_ = smem_read_row*BYTES_PER_ROW_WITH_SKEW + 
                                  smem_read_pix*BYTES_PER_STS;

        // The layout of threads writing to shared memory.
        int smem_write_row = tidx % ROWS;
        int smem_write_col = tidx / ROWS * BYTES_PER_STS;

        // The offset.
        this->smem_write_offset_ = smem_write_row*BYTES_PER_ROW_WITH_SKEW + smem_write_col;

        // TODO: Why not merge it with the read offset?
        this->smem_read_buffer_  = __shfl_sync(0xffffffff, 0, 0);
    }

    // Compute the store pointers.
    template< int N >
    inline __device__ void compute_store_pointers(uint32_t (&ptrs)[N]) {
        #pragma unroll
        for( int ii = 0; ii < N; ++ii ) {
            int offset = ii * PIXELS_PER_STS * BYTES_PER_STS;
            ptrs[ii] = smem_ + (smem_write_offset_ + offset);
        }
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

    // Load from shared memory.
    inline __device__ void load(Fragment (&a)[Xmma_tile::XMMAS_M], int ki, int ri, int si) const {
        #pragma unroll
        for( int mi = 0; mi < Xmma_tile::XMMAS_M; ++mi ) {
            // The number of XMMAs needed to compute a tile.
            const int XMMAS_PER_IMG = 
                Pixel_tile::H * Pixel_tile::W < Xmma_tile::M_PER_XMMA_PER_CTA ? 
                    1 : (Pixel_tile::H * Pixel_tile::W / Xmma_tile::M_PER_XMMA_PER_CTA);
            const int XMMAS_PER_ROW = 
                Pixel_tile::W < Xmma_tile::M_PER_XMMA_PER_CTA ?
                    1 : (Pixel_tile::W / Xmma_tile::M_PER_XMMA_PER_CTA);

            // The number of elements loaded per XMMA.
            const int IMGS_PER_TILE_PER_XMMA = 
                Pixel_tile::H*Pixel_tile::W >= Xmma_tile::M_PER_XMMA_PER_CTA ? 
                    1 : (Xmma_tile::M_PER_XMMA_PER_CTA / Pixel_tile::H / Pixel_tile::W);
            const int ROWS_PER_TILE_PER_XMMA = 
                Pixel_tile::H*Pixel_tile::W < Xmma_tile::M_PER_XMMA_PER_CTA ? 
                    1 : (Xmma_tile::M_PER_XMMA_PER_CTA / Pixel_tile::W);
            const int COLS_PER_TILE_PER_XMMA = 
                Pixel_tile::W < Xmma_tile::M_PER_XMMA_PER_CTA ? 
                    1 : (Xmma_tile::M_PER_XMMA_PER_CTA);

            // Decompose mi into ni, hi and wi.
            const int ni = mi / XMMAS_PER_IMG;
            const int hi = mi % XMMAS_PER_IMG / XMMAS_PER_ROW;
            const int wi = mi % XMMAS_PER_IMG % XMMAS_PER_ROW;

            // The number of bytes.
            const int PIXELS_PER_IMG = PIXELS_PER_TILE_WITH_HALO_H * PIXELS_PER_TILE_WITH_HALO_W; 
            const int PIXELS_PER_ROW = PIXELS_PER_TILE_WITH_HALO_W; 

            // The immediate in shared memory where to grab the element from.
            const int imm = ki * BYTES_PER_ROW_WITH_SKEW +
                            ni * IMGS_PER_TILE_PER_XMMA * PIXELS_PER_IMG * BYTES_PER_STS +
                            hi * ROWS_PER_TILE_PER_XMMA * PIXELS_PER_ROW * BYTES_PER_STS +
                            wi * COLS_PER_TILE_PER_XMMA * BYTES_PER_STS;

            // The offset - depends on the loop iteration.
            const int offset = (ri * PIXELS_PER_TILE_WITH_HALO_W + si) * BYTES_PER_STS + imm;

            // The load pointer.
            uint32_t ptr = this->smem_ + this->smem_read_offset_ + this->smem_read_buffer_ + offset;
            // Read the element.
            uint4 tmp; 
            lds(tmp, ptr);

            // Store the registers.
            a[mi].reg(0) = tmp.x;
            a[mi].reg(1) = tmp.y;
            a[mi].reg(2) = tmp.z;
            a[mi].reg(3) = tmp.w;
        }
    }

    // Move the read offset to next buffer.
    inline __device__ void move_next_read_buffer() {
        if( BUFFERS_PER_TILE > 1 && smem_read_offset_ >= BYTES_PER_TILE_INC_BOUNDARY ) {
            this->smem_read_offset_ -= BYTES_PER_TILE_INC_BOUNDARY;
        } else if( BUFFERS_PER_TILE > 1 ) {
            this->smem_read_offset_ += BYTES_PER_BUFFER;
        }
    }

    // Move the read offset to next N buffers (cirlwlar-buffer).
    inline __device__ void move_next_read_buffer(int N) {
        if( BUFFERS_PER_TILE > 1 ) {
            this->smem_read_buffer_ += N * BYTES_PER_BUFFER;
            this->smem_read_buffer_ -= smem_read_buffer_ >= BYTES_PER_TILE ? BYTES_PER_TILE : 0;
        }
    }

    // Move the write offset to next buffer.
    inline __device__ void move_next_write_buffer() {
        if( BUFFERS_PER_TILE > 1 && smem_write_offset_ >= BYTES_PER_TILE_INC_BOUNDARY ) {
            this->smem_write_offset_ -= BYTES_PER_TILE_INC_BOUNDARY;
        } else if( BUFFERS_PER_TILE > 1 ) {
            this->smem_write_offset_ += BYTES_PER_BUFFER;
        }
    }

    // Move the read offset.
    inline __device__ void move_read_offset(int delta) {
        this->smem_read_offset_ += delta;
    }

    // Move the write offset.
    inline __device__ void move_write_offset(int delta) {
        this->smem_write_offset_ += delta;
    }

    // Rewind smem_read_offset for last LDS phase in main loop.-
    inline __device__ void reverse_smem_read_offset(int ki = 0) {
    }

    // Store to the tile in shared memory.
    template< int N >
    inline __device__ void store(const uint4 (&data)[N], uint32_t preds) {
        uint32_t smem_ptrs[N];
        this->compute_store_pointers(smem_ptrs);
        sts(smem_ptrs, data);
    }

    // Store to the tile in shared memory.
    template< int N >
    inline __device__ void store(const uint4* (&gmem_ptrs)[N], uint32_t preds) {
        uint32_t smem_ptrs[N];
        this->compute_store_pointers(smem_ptrs);
        ldgsts(smem_ptrs, gmem_ptrs, preds);
    }

    // The shared memory pointer.
    uint32_t smem_;
    // The read offset.
    int smem_read_offset_;
    // The write offset.
    int smem_write_offset_;
    // The buffer base offset for read.
    int smem_read_buffer_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< 
    typename Cta_tile, 
    typename Pixel_tile, 
    typename Halo, 
    int BUFFERS_PER_TILE 
>
struct Smem_tile_with_halo_a<Volta_hmma_fp16_traits, 
                             Cta_tile, 
                             Row, 
                             Pixel_tile,
                             Halo,
                             BUFFERS_PER_TILE>
    : public Smem_tile_with_halo_volta_hmma_row_a<Volta_hmma_fp16_traits, 
                                                  Cta_tile, 
                                                  Pixel_tile, 
                                                  Halo, 
                                                  BUFFERS_PER_TILE> {

    // The traits class.
    using Traits = Volta_hmma_fp16_traits;
    // The base class.
    using Base = Smem_tile_with_halo_volta_hmma_row_a<Traits, 
                                                      Cta_tile, 
                                                      Pixel_tile, 
                                                      Halo, 
                                                      BUFFERS_PER_TILE>;

    // Ctor.
    inline __device__ Smem_tile_with_halo_a(void *smem, int tidx) : Base(smem, tidx) {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< 
    typename Cta_tile, 
    typename Pixel_tile, 
    typename Halo, 
    int BUFFERS_PER_TILE 
>
struct Smem_tile_with_halo_a<Volta_hmma_fp32_traits, 
                             Cta_tile, 
                             Row, 
                             Pixel_tile, 
                             Halo, 
                             BUFFERS_PER_TILE>
    : public Smem_tile_with_halo_volta_hmma_row_a<Volta_hmma_fp32_traits, 
                                                  Cta_tile, 
                                                  Pixel_tile, 
                                                  Halo, 
                                                  BUFFERS_PER_TILE> {

    // The traits class.
    using Traits = Volta_hmma_fp32_traits;
    // The base class.
    using Base = Smem_tile_with_halo_volta_hmma_row_a<Traits, 
                                                      Cta_tile, 
                                                      Pixel_tile, 
                                                      Halo, 
                                                      BUFFERS_PER_TILE>;

    // Ctor.
    inline __device__ Smem_tile_with_halo_a(void *smem, int tidx) : Base(smem, tidx) {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace xmma 

