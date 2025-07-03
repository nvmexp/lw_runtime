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

// This class is the base class for a 2D tile in shared memory. It works for NC/xHWx and NHWC.

template< 
    // The instruction traits.
    typename Traits, 
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile, 
    // The dimensions of the 2D tile of pixels.
    typename Pixel_tile, 
    // The halo.
    typename Halo, 
    // The number of buffers in the tile.
    int BUFFERS_PER_TILE_,
    // The number of elements per "packet" (i.e. NC/8HW8 => 8, NHWC => Cta_tile::K).
    int ELEMENTS_PER_PACKET_
>
struct Smem_tile_with_halo_turing_row_a {

    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;
    // The fragment.
    using Fragment = Fragment_a<Traits, Row>;

    // The size of the tile with halo in the H dimension.
    enum { PIXELS_PER_TILE_WITH_HALO_H = Pixel_tile::H + Halo::H };
    // The size of the tile with halo in the W dimension.
    enum { PIXELS_PER_TILE_WITH_HALO_W = Pixel_tile::W + Halo::W };
    // The number of pixels that are needed.
    enum { PIXELS = Pixel_tile::N * PIXELS_PER_TILE_WITH_HALO_H * PIXELS_PER_TILE_WITH_HALO_W };

    // The number of bits per element.
    enum { BITS_PER_ELEMENT = Traits::BITS_PER_ELEMENT_A };
    // The size of each STS.
    enum { BYTES_PER_STS = 16, BYTES_PER_LDS = 16 };
    // The number of elements per STS.
    enum { ELEMENTS_PER_STS = BYTES_PER_STS * 8 / BITS_PER_ELEMENT };
    // The number of elements per packet. See above.
    enum { ELEMENTS_PER_PACKET = ELEMENTS_PER_PACKET_ };
    // The number of packets per pixel. It's one 1 for NHWC.
    enum { PACKETS_PER_PIXEL = Cta_tile::K / ELEMENTS_PER_PACKET };
    // The number of threads per packet.
    enum { THREADS_PER_PACKET = ELEMENTS_PER_PACKET / ELEMENTS_PER_STS };

    // The number of pixels written with a single STS.
    enum { PIXELS_PER_STS = Cta_tile::THREADS_PER_CTA / THREADS_PER_PACKET };
    // The number of STS.
    enum { STS = Div_up<PIXELS, PIXELS_PER_STS>::VALUE };

    // The number of rows in shared memory.
    enum { ROWS = THREADS_PER_PACKET * PACKETS_PER_PIXEL };
    // The size of a single row in bytes.
    enum { BYTES_PER_ROW = STS * PIXELS_PER_STS * BYTES_PER_STS };
    // The number of bytes written per row of shared memory for each "store".
    enum { BYTES_PER_SKEW_ = Max<1, 8 / THREADS_PER_PACKET>::VALUE * BYTES_PER_STS };
    // The number of bytes written per row of shared memory for each "store".
    enum { BYTES_PER_SKEW = THREADS_PER_PACKET == 1 ? 0 : BYTES_PER_SKEW_ };
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

    // The number of pixels per tile.
    enum { PIXELS_H = Pixel_tile::H, PIXELS_W = Pixel_tile::W };

    // The number of pixels loaded per XMMA.
    enum { PIXELS_PER_XMMA = Xmma_tile::M_PER_XMMA_PER_CTA };

    // The number of XMMAs needed to compute a tile.
    enum { XMMAS_PER_IMG = Max<1, PIXELS_H * PIXELS_W / PIXELS_PER_XMMA>::VALUE };
    enum { XMMAS_PER_ROW = Max<1,            PIXELS_W / PIXELS_PER_XMMA>::VALUE };

    // The number of elements loaded per XMMA.
    enum { IMGS_PER_TILE_PER_XMMA = Max<1, PIXELS_PER_XMMA / PIXELS_H / PIXELS_W>::VALUE };
    enum { ROWS_PER_TILE_PER_XMMA = Max<1, PIXELS_PER_XMMA            / PIXELS_W>::VALUE };
    enum { COLS_PER_TILE_PER_XMMA = Max<1, PIXELS_PER_XMMA                      >::VALUE };

    // The number of pixels per tile.
    enum { PIXELS_PER_IMG = PIXELS_PER_TILE_WITH_HALO_H * PIXELS_PER_TILE_WITH_HALO_W }; 
    enum { PIXELS_PER_ROW =                               PIXELS_PER_TILE_WITH_HALO_W }; 

    // Ctor.
    inline __device__ Smem_tile_with_halo_turing_row_a(void *smem) 
        : smem_(get_smem_pointer(smem)) {
    }

    // Ctor.
    inline __device__ Smem_tile_with_halo_turing_row_a(void *smem, int tidx) 
        : Smem_tile_with_halo_turing_row_a(smem) {

        // The number of warps.
        enum { WARPS_M = Cta_tile::WARPS_M };
        enum { WARPS_N = Cta_tile::WARPS_N };
        enum { WARPS_K = Cta_tile::WARPS_K };

        // The masks to select the warps.
        enum { WARP_MASK_M = Warp_masks<WARPS_M, WARPS_N, WARPS_K>::M };
        enum { WARP_MASK_K = Warp_masks<WARPS_M, WARPS_N, WARPS_K>::K };

        // The divisor for the warps.
        enum { WARP_DIV_M =       1 *       1 * Cta_tile::THREADS_PER_WARP };
        enum { WARP_DIV_K = WARPS_M * WARPS_N * Cta_tile::THREADS_PER_WARP };
        
        // Compute the row and the column in shared memory. 
        int row = (tidx & WARP_MASK_K) / WARP_DIV_K * Xmma_tile::XMMAS_K;
        int col = (tidx & WARP_MASK_M) / WARP_DIV_M * 16 + (tidx & 0x0f);

        // Compute the read offset.
        compute_read_offset_(row, col);
        // Compute the write offset.
        compute_write_offset_(tidx);

        // Use URF for the read/write buffers.
        this->read_buffer_ = this->write_buffer_ = __shfl_sync(0xffffffff, 0, 0);
    }

    // Compute the load offset from mi/ri/si.
    inline __device__ int compute_load_offset_(int mi, int ri, int si) const {
        // Decompose mi into ni, hi and wi.
        int ni = mi / XMMAS_PER_IMG;
        int hi = mi % XMMAS_PER_IMG / XMMAS_PER_ROW;
        int wi = mi % XMMAS_PER_IMG % XMMAS_PER_ROW;

        // Reassemble the offset in shared memory.
        return ni * IMGS_PER_TILE_PER_XMMA * PIXELS_PER_IMG * BYTES_PER_LDS +
               hi * ROWS_PER_TILE_PER_XMMA * PIXELS_PER_ROW * BYTES_PER_LDS +
               ri                          * PIXELS_PER_ROW * BYTES_PER_LDS +
               wi * COLS_PER_TILE_PER_XMMA                  * BYTES_PER_LDS +
               si                                           * BYTES_PER_LDS;
    }

    // Compute the read offset from the row/col.
    inline __device__ void compute_read_offset_(int row, int col) {
        // The column has to be mapped to the location in the tile.
        int n = col / (Pixel_tile::H * Pixel_tile::W);
        int h = col % (Pixel_tile::H * Pixel_tile::W) / Pixel_tile::W;
        int w = col % (Pixel_tile::H * Pixel_tile::W) % Pixel_tile::W;

        // The offset in the tile.
        int pix = n * PIXELS_PER_TILE_WITH_HALO_H*PIXELS_PER_TILE_WITH_HALO_W +
                  h * PIXELS_PER_TILE_WITH_HALO_W +
                  w;

        // The corresponding offset.
        read_offset_ = row*BYTES_PER_ROW_WITH_SKEW + pix*BYTES_PER_LDS; 
    }

    // Compute the store pointers.
    template< int N >
    inline __device__ void compute_store_pointers(uint32_t (&ptrs)[N]) {
        static_assert(PACKETS_PER_PIXEL * STS == N, "");
        #pragma unroll
        for( int ii = 0; ii < PACKETS_PER_PIXEL; ++ii ) {
            #pragma unroll
            for( int jj = 0; jj < STS; ++jj ) {
                int offset = ii * BYTES_PER_ROW_WITH_SKEW + jj * PIXELS_PER_STS * BYTES_PER_STS;
                ptrs[ii*STS + jj] = smem_ + write_offset_ + write_buffer_ + offset;
            }
        }
    }

    // Compute the write offset.
    inline __device__ void compute_write_offset_(int tidx) {
        // The layout of threads writing to shared memory.
        int row = tidx % THREADS_PER_PACKET;
        int col = tidx / THREADS_PER_PACKET;

        // The offset.
        write_offset_ = row*BYTES_PER_ROW_WITH_SKEW + col*BYTES_PER_STS;
    }

    // Move the read offset// Print the content of the tile (only for debug ;)).
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
        // Load the different pixels.
        #pragma unroll
        for( int mi = 0; mi < Xmma_tile::XMMAS_M; ++mi ) {
            // The immediate in shared memory where to grab the element from.
            int imm = compute_load_offset_(mi, ri, si) + ki * BYTES_PER_ROW_WITH_SKEW;

            // Read the element.
            uint2 tmp;
            ldsm(tmp, smem_ + read_offset_ + read_buffer_ + imm);
            a[mi].reg(0) = tmp.x;
            a[mi].reg(1) = tmp.y;
        }
    }

    // Move the read offset to next buffer.
    inline __device__ void move_to_next_read_buffer() {
        if( BUFFERS_PER_TILE > 1 && read_buffer_ >= BYTES_PER_TILE_INC_BOUNDARY ) {
            this->read_buffer_ -= BYTES_PER_TILE_INC_BOUNDARY;
        } else if( BUFFERS_PER_TILE > 1 ) {
            this->read_buffer_ += BYTES_PER_BUFFER;
        }
    }

    // Move the read offset to next buffer.
    inline __device__ void move_next_read_buffer() {
        this->move_to_next_read_buffer();
    }

    // Move the read offset to next N buffers (cirlwlar-buffer).
    inline __device__ void move_next_read_buffer(int N) {
        if( BUFFERS_PER_TILE > 1 ) {
            this->read_buffer_ += N * BYTES_PER_BUFFER;
            this->read_buffer_ -= read_buffer_ >= BYTES_PER_TILE ? BYTES_PER_TILE : 0;
        }
    }

    // Move the write offset to next buffer.
    inline __device__ void move_to_next_write_buffer() {
        if( BUFFERS_PER_TILE > 1 && write_buffer_ >= BYTES_PER_TILE_INC_BOUNDARY ) {
            this->write_buffer_ -= BYTES_PER_TILE_INC_BOUNDARY;
        } else if( BUFFERS_PER_TILE > 1 ) {
            this->write_buffer_ += BYTES_PER_BUFFER;
        }
    }

    // Move the write offset to next buffer.
    inline __device__ void move_next_write_buffer() {
        this->move_to_next_write_buffer();
    }

    // Move the read offset.
    inline __device__ void move_read_offset(int delta) {
        this->read_offset_ += delta;
    }

    // Move the write offset.
    inline __device__ void move_write_offset(int delta) {
        this->write_offset_ += delta;
    }

    // Rewind smem_read_offset for last LDS phase in main loop.-
    inline __device__ void reverse_read_offset(int ki = 0) {
    }

    // Store to the tile in shared memory.
    template< int N >
    inline __device__ void store(const uint4 (&data)[N], uint32_t = uint32_t(-1)) {
        uint32_t ptrs[N];
        this->compute_store_pointers(ptrs);
        sts(ptrs, data);
    }

    // Store to the tile in shared memory.
    template< int N >
    inline __device__ void store(const void* (&gmem_ptrs)[N], uint32_t preds = uint32_t(-1)) {
        uint32_t ptrs[N];
        this->compute_store_pointers(ptrs);
        ldgsts(ptrs, gmem_ptrs, preds);
    }

    // The shared memory pointer.
    uint32_t smem_;
    // The read offset.
    int read_offset_;
    // The write offset.
    int write_offset_;
    // The buffer base offset for read.
    int read_buffer_;
    // The buffer base offset for write.
    int write_buffer_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Cta_tile, typename Pixel_tile, typename Halo, int BUFFERS_PER_TILE >
struct Smem_tile_with_halo_a<Turing_hmma_fp16_traits, 
                             Cta_tile, 
                             Row, 
                             Pixel_tile, 
                             Halo, 
                             BUFFERS_PER_TILE>
    : public Smem_tile_with_halo_turing_row_a<Turing_hmma_fp16_traits, 
                                              Cta_tile, 
                                              Pixel_tile, 
                                              Halo, 
                                              BUFFERS_PER_TILE,
                                              Cta_tile::K> {

    // The traits class.
    using Traits = Turing_hmma_fp16_traits;
    // The base class.
    using Base = Smem_tile_with_halo_turing_row_a<Traits, 
                                                  Cta_tile, 
                                                  Pixel_tile, 
                                                  Halo, 
                                                  BUFFERS_PER_TILE,
                                                  Cta_tile::K>;

    // Ctor.
    inline __device__ Smem_tile_with_halo_a(void *smem, int tidx) : Base(smem, tidx) {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Cta_tile, typename Pixel_tile, typename Halo, int BUFFERS_PER_TILE >
struct Smem_tile_with_halo_a<Turing_hmma_fp16_traits, 
                             Cta_tile, 
                             Col_interleaved, 
                             Pixel_tile, 
                             Halo, 
                             BUFFERS_PER_TILE>
    : public Smem_tile_with_halo_turing_row_a<Turing_hmma_fp16_traits, 
                                              Cta_tile, 
                                              Pixel_tile, 
                                              Halo, 
                                              BUFFERS_PER_TILE,
                                              8> {

    // The traits class.
    using Traits = Turing_hmma_fp16_traits;
    // The base class.
    using Base = Smem_tile_with_halo_turing_row_a<Traits, 
                                                  Cta_tile, 
                                                  Pixel_tile, 
                                                  Halo, 
                                                  BUFFERS_PER_TILE,
                                                  8>;

    // Ctor.
    inline __device__ Smem_tile_with_halo_a(void *smem, int tidx) : Base(smem, tidx) {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Cta_tile, typename Pixel_tile, typename Halo, int BUFFERS_PER_TILE >
struct Smem_tile_with_halo_a<Turing_hmma_fp32_traits, 
                             Cta_tile, 
                             Row, 
                             Pixel_tile, 
                             Halo, 
                             BUFFERS_PER_TILE>
    : public Smem_tile_with_halo_turing_row_a<Turing_hmma_fp32_traits, 
                                              Cta_tile, 
                                              Pixel_tile, 
                                              Halo, 
                                              BUFFERS_PER_TILE,
                                              Cta_tile::K> {

    // The traits class.
    using Traits = Turing_hmma_fp32_traits;
    // The base class.
    using Base = Smem_tile_with_halo_turing_row_a<Traits, 
                                                  Cta_tile, 
                                                  Pixel_tile, 
                                                  Halo, 
                                                  BUFFERS_PER_TILE,
                                                  Cta_tile::K>;

    // Ctor.
    inline __device__ Smem_tile_with_halo_a(void *smem, int tidx) : Base(smem, tidx) {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Cta_tile, typename Pixel_tile, typename Halo, int BUFFERS_PER_TILE >
struct Smem_tile_with_halo_a<Turing_hmma_fp32_traits, 
                             Cta_tile, 
                             Col_interleaved, 
                             Pixel_tile, 
                             Halo, 
                             BUFFERS_PER_TILE>
    : public Smem_tile_with_halo_turing_row_a<Turing_hmma_fp32_traits, 
                                              Cta_tile, 
                                              Pixel_tile, 
                                              Halo, 
                                              BUFFERS_PER_TILE,
                                              8> {

    // The traits class.
    using Traits = Turing_hmma_fp32_traits;
    // The base class.
    using Base = Smem_tile_with_halo_turing_row_a<Traits, 
                                                  Cta_tile, 
                                                  Pixel_tile, 
                                                  Halo, 
                                                  BUFFERS_PER_TILE,
                                                  8>;

    // Ctor.
    inline __device__ Smem_tile_with_halo_a(void *smem, int tidx) : Base(smem, tidx) {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Cta_tile, typename Pixel_tile, typename Halo, int BUFFERS_PER_TILE_ >
struct Smem_tile_with_halo_a<Turing_imma_interleaved_int8_int32_traits, 
                             Cta_tile, 
                             Col_interleaved, 
                             Pixel_tile, 
                             Halo, 
                             BUFFERS_PER_TILE_> {

    // The traits class.
    using Traits = Turing_imma_interleaved_int8_int32_traits;
    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;
    // The fragment.
    using Fragment = Fragment_a<Traits, Row>;

    // The number of words in 128b.
    enum { BITS_PER_ELEMENT = 8, BYTES_PER_STS = 16, BYTES_PER_LDS = 8 };
    // TODO: hard code halo for 3x3.
    // The size of the tile with halo in the H dimension.
    enum { PIXELS_PER_TILE_WITH_HALO_H = Pixel_tile::H + 2 };
    // The size of the tile with halo in the W dimension.
    enum { PIXELS_PER_TILE_WITH_HALO_W = Pixel_tile::W + 2 };
    // The number of pixels that are needed.
    enum { PIXELS = Pixel_tile::N * PIXELS_PER_TILE_WITH_HALO_H * PIXELS_PER_TILE_WITH_HALO_W };
    // The number of threads per pixel.
    enum { THREADS_PER_PIXEL = Cta_tile::K * BITS_PER_ELEMENT / 8 / BYTES_PER_STS };
    // The number of pixels written with a single STS.
    enum { PIXELS_PER_STS = Cta_tile::THREADS_PER_CTA / THREADS_PER_PIXEL };
    // The number of STS.
    enum { STS = (PIXELS + PIXELS_PER_STS-1) / PIXELS_PER_STS };
    // How many shared memory stages for a tile.
    enum { BUFFERS_PER_TILE = BUFFERS_PER_TILE_ };
    // Interleaved elements.
    enum { ELEMENTS_PER_PACKET = 32 };
    // Bytes per packed.
    enum { BYTES_PER_PACKET = 32 };
    // The number of rows that are needed.
    enum { ROWS = Cta_tile::K / ELEMENTS_PER_PACKET };
    // The number of threads per row.
    enum { THREADS_PER_ROW = Cta_tile::THREADS_PER_CTA / ROWS };

    // The size of a single row in bytes.
    enum { BYTES_PER_ROW = STS * THREADS_PER_ROW * BYTES_PER_STS };
    // The number of bytes written per row of shared memory for each "store".
    enum { BYTES_PER_SKEW = 0 };
    // The number of columns with the skew.
    enum { BYTES_PER_ROW_WITH_SKEW = BYTES_PER_ROW + BYTES_PER_SKEW };
    // The size of one buffer in bytes in shared memory.
    enum { BYTES_PER_BUFFER = ROWS * BYTES_PER_ROW_WITH_SKEW };
    // Smem size per tile.
    enum { BYTES_PER_TILE = BYTES_PER_BUFFER * BUFFERS_PER_TILE };
    // The boundary for smem_read_offset and smem_write_offset increment.
    enum { BYTES_PER_TILE_INC_BOUNDARY = BYTES_PER_TILE - BYTES_PER_BUFFER };

    // Ctor.
    inline __device__ Smem_tile_with_halo_a(void *smem, int tidx) 
        : smem_(get_smem_pointer(smem)) {

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
        int smem_read_row = (tidx & WARP_MASK_K) / WARP_DIV_K * ROWS / WARPS_K;
        int smem_read_col = (tidx & WARP_MASK_M) / WARP_DIV_M * Xmma_tile::M_PER_XMMA *
                            (BYTES_PER_STS / BYTES_PER_LDS) + 
                            (tidx & 0x1f) / 2;

        // The shared memory offset.
        this->smem_read_offset_ = smem_read_row*BYTES_PER_ROW + 
                                  smem_read_col*BYTES_PER_STS + 
                                  (tidx % 2) * BYTES_PER_LDS;

        // The row/col written by the thread.
        int smem_write_row = (tidx / THREADS_PER_ROW);
        int smem_write_col = (tidx % THREADS_PER_ROW);

        // The location where the thread writes its elements.
        this->smem_write_offset_ = smem_write_row*BYTES_PER_ROW + 
                                   smem_write_col*BYTES_PER_STS;
        this->smem_read_buffer_  = __shfl_sync(0xffffffff, 0, 0);
        this->smem_write_buffer_ = __shfl_sync(0xffffffff, 0, 0);

    }

    // Compute the store pointers.
    template< int N >
    inline __device__ void compute_store_pointers(uint32_t (&ptrs)[N]) {
        #pragma unroll
        for( int ii = 0; ii < N; ++ii ) {
            int offset = ii * PIXELS_PER_STS * BYTES_PER_PACKET;
            ptrs[ii] = smem_ + smem_write_offset_ + smem_write_buffer_  + offset;
        }
    }

    // Move the read offset// Print the content of the tile (only for debug ;)).
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
            #pragma unroll
            for( int i = 0; i < Xmma_tile::M_PER_XMMA/8; ++i ) {
                int offset = this->smem_read_offset_ +
                             mi*Xmma_tile::M_PER_XMMA_PER_CTA*BYTES_PER_PACKET +
                             ki*BYTES_PER_ROW +
                             i*Cta_tile::THREADS_PER_WARP*BYTES_PER_LDS;
                uint2 tmp;
                xmma::lds(tmp, this->smem_ + this->smem_read_buffer_ + offset);
                a[mi].reg(i*2  ) = tmp.x;
                a[mi].reg(i*2+1) = tmp.y;
            }
        }
    }
 
    // Move the read offset to next buffer.
    inline __device__ void move_next_read_buffer() {
        if( BUFFERS_PER_TILE > 1 && smem_read_buffer_ >= BYTES_PER_TILE_INC_BOUNDARY ) {
            this->smem_read_buffer_ -= BYTES_PER_TILE_INC_BOUNDARY;
        } else if( BUFFERS_PER_TILE > 1 ) {
            this->smem_read_buffer_ += BYTES_PER_BUFFER;
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
        if( BUFFERS_PER_TILE > 1 && smem_write_buffer_ >= BYTES_PER_TILE_INC_BOUNDARY ) {
            this->smem_write_buffer_ -= BYTES_PER_TILE_INC_BOUNDARY;
        } else if( BUFFERS_PER_TILE > 1 ) {
            this->smem_write_buffer_ += BYTES_PER_BUFFER;
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

    // The shared memory pointer.
    uint32_t smem_;
    // The read offset.
    int smem_read_offset_;
    // The write offset.
    int smem_write_offset_;
    // The buffer base offset for read.
    int smem_read_buffer_;
    // The buffer base offset for write.
    int smem_write_buffer_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace xmma 

