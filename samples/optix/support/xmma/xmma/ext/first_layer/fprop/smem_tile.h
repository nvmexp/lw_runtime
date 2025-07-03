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

#include <xmma/utils.h>

namespace xmma {
namespace ext {
namespace first_layer {
namespace fprop {

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Traits, typename Cta_tile, typename Cfg >
struct Smem_tile_base_a {

    // We assume one warp in the N dimension.
    static_assert(Cta_tile::WARPS_N == 1, "");
    // We assume one warp in the K dimension.
    static_assert(Cta_tile::WARPS_K == 1, "");

    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;

    // The number of channels per pixel. The layout is NDHW4.
    enum { CHANNELS_PER_PIXEL = Cfg::CHANNELS_PER_PIXEL };
    // The size of each channel in bytes.
    enum { BYTES_PER_CHANNEL = Traits::BITS_PER_ELEMENT_A / 8 };
    // The size of a pixel in bytes.
    enum { BYTES_PER_PIXEL = CHANNELS_PER_PIXEL * BYTES_PER_CHANNEL };
    // The size of each STS - each thread stores 1 pixel per STS.
    enum { BYTES_PER_STS = BYTES_PER_PIXEL };
    // The number of pixels stored per STS.
    enum { PIXELS_PER_STS = Cta_tile::THREADS_PER_CTA };

    // The size of the input tile in the vertical dimension. 
    enum { IMG_H = Cfg::IMG_H_IN_PROLOGUE };
    // The size of the input tile in the horizontal dimension.
    enum { IMG_W = Cfg::IMG_W };
    // The number of buffers.
    enum { BUFFERS_PER_TILE = 2 };
    // The size of a row in bytes.
    enum { BYTES_PER_ROW = IMG_W * BYTES_PER_PIXEL };
    // The size of a buffer.
    enum { BYTES_PER_BUFFER = IMG_H * BYTES_PER_ROW };
    // The amount of shared memory needed for the tile.  
    enum { BYTES_PER_TILE = BUFFERS_PER_TILE * BYTES_PER_BUFFER };

    // The type of elements that are stored in shared memory by each thread.
    using Store_type = typename xmma::Uint_from_size_in_bytes<BYTES_PER_STS>::Type;

    // Each quad reads data for the computation of one output pixel per LDS.
    enum { THREADS_PER_QUAD = 4 };
    // Each quad loads two input pixels per LDS.
    enum { PIXELS_PER_QUAD = 2 };
    // The number of threads per pixel.
    enum { THREADS_PER_PIXEL = THREADS_PER_QUAD / PIXELS_PER_QUAD };
    // The number of channels loaded per thread.
    enum { CHANNELS_PER_THREAD = CHANNELS_PER_PIXEL / THREADS_PER_PIXEL };
    // Make sure each thread loads 2 channels.
    static_assert(CHANNELS_PER_THREAD == 2, "");
    // The size of each LDS.
    enum { BYTES_PER_LDS = CHANNELS_PER_THREAD * BYTES_PER_CHANNEL };

    // The number of loads per thread in the main loop (in the K dimension).
    enum { LDS = Cfg::FLT_S };
    // Make sure the number of LDSs matches the number of XMMAs in the K dimension.
    static_assert(LDS == (int) Xmma_tile::XMMAS_K, "");

    // The number of rows loaded per LDS.
    enum { ROWS_PER_LDS = Xmma_tile::M_PER_XMMA_PER_CTA / Cfg::OUT_W };

    // Ctor.
    inline __device__ Smem_tile_base_a(void *smem, int tidx) 
        : smem_(xmma::get_smem_pointer(smem)) {

        // We should not have bank conflicts if we store the data "as-is".
        smem_write_ = smem_ + tidx * BYTES_PER_STS;

        // Make sure the number of columns is a multiple of 16 or a warp may span 2 rows :urgh:.
        static_assert(Cfg::OUT_W % Xmma_tile::M_PER_XMMA == 0, "Not supported");

        // Decompose the CTA in warps/lanes.
        int warp = tidx / Cta_tile::THREADS_PER_WARP;
        int lane = tidx % Cta_tile::THREADS_PER_WARP;

        // The 1st index of the 1st pixel loaded by that thread.
        int pix = warp * Xmma_tile::M_PER_XMMA + lane / THREADS_PER_QUAD;

        // Map that output pixel to the corresponding input pixel.
        int h = pix / Cfg::OUT_W * Cfg::STRIDE_H;
        int w = pix % Cfg::OUT_W * Cfg::STRIDE_W;

        // The channel loaded by this thread.
        int c = tidx % THREADS_PER_PIXEL * CHANNELS_PER_THREAD;

        // The position of the thread in the filter.
        rsi_ = tidx % THREADS_PER_QUAD / THREADS_PER_PIXEL;

        // The base offset.
        uint32_t smem_read = smem_ + (h * IMG_W + w) * BYTES_PER_PIXEL + c * BYTES_PER_CHANNEL;

        // Extract the 2D positions of the pixel loaded by this thread for each LDS.
        #pragma unroll
        for( int ii = 0; ii < LDS; ++ii ) {

            // Compute the filter position.
            int ri = (rsi_ + ii * PIXELS_PER_QUAD) / Cfg::FLT_S;
            int si = (rsi_ + ii * PIXELS_PER_QUAD) % Cfg::FLT_S;

            // Assemble the shared memory offset for reads.
            smem_read_[ii] = smem_read + ri * BYTES_PER_ROW + si * BYTES_PER_PIXEL;
        }

        // // DEBUG.
        // for( int ii = 0; ii < LDS; ++ii ) {
        //     printf("tidx=%3d smem_read_[%2d] = %6d\n", tidx, ii, smem_read_[ii]);
        // }
        // // END OF DEBUG.
    }

    // Print the content of the tile (only for debug ;)).
    inline __device__ void debug_print() const {
        for( int row = 0; row < IMG_H * BUFFERS_PER_TILE; ++row ) {
            for( int col = 0; col < BYTES_PER_ROW; col += 16 ) {
                if( threadIdx.x == 0 ) {
                    uint4 val;
                    xmma::lds(val, smem_ + row*BYTES_PER_ROW + col);
                    printf("a[%3d][%2d]=(%8.3f, %8.3f, %8.3f, %8.3f)\n",
                        row,
                        col / 16,
                        reinterpret_cast<const float&>(val.x),
                        reinterpret_cast<const float&>(val.y),
                        reinterpret_cast<const float&>(val.z),
                        reinterpret_cast<const float&>(val.w));
                }
            }
        }
    }

    // Move the read offset.
    inline __device__ void move_read_offset() {
        #pragma unroll
        for( int ii = 0; ii < LDS; ++ii ) {
            smem_read_[ii] += Cfg::FLT_R_PER_ITERATION * BYTES_PER_ROW;
            if( smem_read_[ii] >= smem_ + BYTES_PER_TILE ) {
                smem_read_[ii] -= BYTES_PER_TILE;
            }
        }
    }

    // Move the write offset.
    inline __device__ void move_write_offset(bool is_prologue = false) {
        const int ROWS = is_prologue ? Cfg::IMG_H_IN_PROLOGUE : Cfg::IMG_H_PER_INNER_LOOP;
        smem_write_ += ROWS * BYTES_PER_ROW;
        if( smem_write_ >= smem_ + BYTES_PER_TILE ) {
            smem_write_ -= BYTES_PER_TILE;
        }
    }

    // Reset the read offset after the inner loop.
    inline __device__ void reset_read_offset() {
        // Reset the read offsets to their original position in the buffer.
        enum { BYTES_PER_OUTER_LOOP = Cfg::INNER_LOOPS * Cfg::FLT_R_PER_ITERATION * BYTES_PER_ROW };
        #pragma unroll
        for( int ii = 0; ii < LDS; ++ii ) {
            if( smem_read_[ii] < BYTES_PER_OUTER_LOOP ) {
                smem_read_[ii] += BYTES_PER_TILE;
            }
            smem_read_[ii] = smem_read_[ii] - BYTES_PER_OUTER_LOOP;
        }

        // Switch to the next buffer.
        int move_forward = smem_read_[0] < BYTES_PER_BUFFER;
        #pragma unroll
        for( int ii = 0; ii < LDS; ++ii ) {
            if( move_forward ) {
                smem_read_[ii] += BYTES_PER_BUFFER;
            } else {
                smem_read_[ii] -= BYTES_PER_BUFFER;
            }
        }

        // // DEBUG.
        // if( threadIdx.x / 4 == 16 ) {
        //     printf("tidx=%3d a::smem_read_[0]=%5d\n", threadIdx.x, smem_read_[0]);
        // }
        // // END OF DEBUG.
    }

    // Store to the tile in shared memory.
    template< int N >
    inline __device__ void store(const Store_type (&data)[N], int is_active_for_last_sts) {
        #pragma unroll
        for( int ii = 0; ii < N; ++ii ) {
            if( ii < N - 1 || is_active_for_last_sts ) {
                xmma::sts(smem_write_ + ii * PIXELS_PER_STS * BYTES_PER_PIXEL, data[ii]);
            }
        }
    }

    // The base shared memory pointer.
    uint32_t smem_;
    // The store pointer.
    uint32_t smem_write_;
    // The load pointers for the main loop.
    uint32_t smem_read_[LDS];
    // The position in the filter.
    int rsi_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Traits, typename Cta_tile, typename Cfg >
struct Smem_tile_a {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Cta_tile, typename Cfg >
struct Smem_tile_a<xmma::Ampere_hmma_tf32_traits<lwtlass::float_tf32_t, lwtlass::float_tf32_t>,
                   Cta_tile,
                   Cfg> 
    : public Smem_tile_base_a<xmma::Ampere_hmma_tf32_traits<lwtlass::float_tf32_t,
                                                                lwtlass::float_tf32_t>,
                              Cta_tile,
                              Cfg> {
    // The traits.
    using Traits = xmma::Ampere_hmma_tf32_traits<lwtlass::float_tf32_t, lwtlass::float_tf32_t>;
    // The base class.
    using Base = Smem_tile_base_a<Traits, Cta_tile, Cfg>;
    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;
    // The fragment.
    using Fragment = xmma::Fragment_a<Traits, xmma::Row>;

    // Ctor.
    inline __device__ Smem_tile_a(void *smem, int tidx) : Base(smem, tidx) {
    }

    // Load from shared memory.
    inline __device__ void load(Fragment (&a)[Xmma_tile::XMMAS_M], int rsi, int ki) {
        const int is_valid = this->rsi_ < (int) (Cfg::FLT_R * Cfg::FLT_S) - rsi;
        #pragma unroll
        for( int mi = 0; mi < Xmma_tile::XMMAS_M; ++mi ) {
            // The number of rows per XMMA in the M dimension.
            enum { ROWS_PER_XMMA_PER_CTA = Base::ROWS_PER_LDS * Cfg::STRIDE_H };
            // The amount of data read per XMMA in the M dimension.
            enum { BYTES_PER_XMMA_PER_CTA = ROWS_PER_XMMA_PER_CTA * Base::BYTES_PER_ROW };

            // Compute the base offset.
            int offset = this->smem_read_[ki] + mi * BYTES_PER_XMMA_PER_CTA;

            // Make sure we are still in the buffer.
            if( offset >= Base::BYTES_PER_TILE ) {
                offset -= Base::BYTES_PER_TILE;
            }

            // // DEBUG.
            // printf("tidx=%3d mi=%d ki=%d offset_0=%6d offset_1=%6d rsi=%2d is_valid=%d read=%6d\n", 
            //     threadIdx.x, 
            //     mi, 
            //     ki,
            //     offset_0,
            //     offset_1,
            //     rsi,
            //     is_valid,
            //     this->smem_read_[ki]);

            // The amount of bytes needed to grab the 2nd "half of the data" for an XMMA.
            enum { BYTES_PER_HALF_XMMA = 8 * Cfg::STRIDE_W * Base::BYTES_PER_PIXEL };

            // Load the data using 2x LDS.64.
            uint2 tmp_0 = make_uint2(0u, 0u), tmp_1 = make_uint2(0u, 0u);
            if( is_valid ) {
                xmma::lds(tmp_0, offset + 0*BYTES_PER_HALF_XMMA);
                xmma::lds(tmp_1, offset + 1*BYTES_PER_HALF_XMMA);
            }
            a[mi].reg(0) = tmp_0.x;
            a[mi].reg(1) = tmp_1.x;
            a[mi].reg(2) = tmp_0.y;
            a[mi].reg(3) = tmp_1.y;
        }
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Traits, typename Cta_tile, typename Cfg >
struct Smem_tile_base_b {

    // The number of channels per tap. The layout is NDHW4.
    enum { CHANNELS_PER_TAP = Cfg::CHANNELS_PER_PIXEL };
    // The size of each channel in bytes.
    enum { BYTES_PER_CHANNEL = Traits::BITS_PER_ELEMENT_B / 8 };
    // The size of a tap in bytes.
    enum { BYTES_PER_TAP = CHANNELS_PER_TAP * BYTES_PER_CHANNEL };
    // The size of each STS - each thread stores 1 tap per STS.
    enum { BYTES_PER_STS = BYTES_PER_TAP };
    // The number of taps stored per STS.
    enum { TAPS_PER_STS = Cta_tile::THREADS_PER_CTA };
    // The number of taps.
    enum { TAPS = Cfg::FLT_K * Cfg::FLT_T * Cfg::FLT_R * Cfg::FLT_S };
    // The number of STS.
    enum { STS = xmma::Div_up<TAPS, TAPS_PER_STS>::VALUE };

    // The number of shared memory rows.
    enum { ROWS = Cfg::FLT_K };
    // The size of a row in bytes.
    enum { BYTES_PER_ROW = Cfg::FLT_T * Cfg::FLT_R * Cfg::FLT_S * BYTES_PER_TAP };
    // The skew to avoid bank conflicts on loads (we have conflicts on stores).
    enum { BYTES_PER_SKEW = BYTES_PER_TAP };
    // The size of a row with skews.
    enum { BYTES_PER_ROW_WITH_SKEW = BYTES_PER_ROW + BYTES_PER_SKEW };
    // The size of a tile.
    enum { BYTES_PER_TILE = ROWS * BYTES_PER_ROW_WITH_SKEW };

    // The type of elements that are stored in shared memory by each thread.
    using Store_type = typename xmma::Uint_from_size_in_bytes<BYTES_PER_STS>::Type;

    // The number of threads per quad.
    enum { THREADS_PER_QUAD = 4 };
    // Each quad loads two input taps per LDS.
    enum { TAPS_PER_QUAD = 2 };
    // The number of threads per tap.
    enum { THREADS_PER_TAP = THREADS_PER_QUAD / TAPS_PER_QUAD };
    // The number of channels loaded per thread.
    enum { CHANNELS_PER_THREAD = CHANNELS_PER_TAP / THREADS_PER_TAP };
    // Make sure each thread loads 2 channels.
    static_assert(CHANNELS_PER_THREAD == 2, "");
    // The size of each LDS.
    enum { BYTES_PER_LDS = CHANNELS_PER_THREAD * BYTES_PER_CHANNEL };

    // The number of taps per LDS.
    enum { TAPS_PER_LDS = Cta_tile::THREADS_PER_WARP / THREADS_PER_QUAD };
    // The number of LDS.
    enum { LDS = xmma::Div_up<TAPS, TAPS_PER_LDS>::VALUE };

    // Ctor.
    inline __device__ Smem_tile_base_b(void *smem, int tidx) 
        : smem_(xmma::get_smem_pointer(smem)) {

        // The offsets for the writes.
        #pragma unroll
        for( int ii = 0; ii < STS; ++ii ) {
            // The linear index.
            int idx = tidx + ii * TAPS_PER_STS;

            // Decompose the index into filter and trs.
            int row = idx / (Cfg::FLT_T * Cfg::FLT_R * Cfg::FLT_S);
            int col = idx % (Cfg::FLT_T * Cfg::FLT_R * Cfg::FLT_S);

            // Assemble the ith pointer.
            smem_write_[ii] = smem_ + row * BYTES_PER_ROW_WITH_SKEW + col * BYTES_PER_TAP;
        }

        // We support only one warp in the horizontal dimension for the moment.
        int row, col;
        if( Cta_tile::WARPS_N == 1 && Cta_tile::WARPS_K == 1 ) {
            row = tidx % Cta_tile::THREADS_PER_WARP / THREADS_PER_QUAD;
            col = tidx % Cta_tile::THREADS_PER_WARP % THREADS_PER_QUAD * CHANNELS_PER_THREAD;
        } else {
            assert(false);
        }

        // The position in the filter.
        rsi_ = col / CHANNELS_PER_TAP;
        // The offset for the reads.
        smem_read_ = smem_ + row * BYTES_PER_ROW_WITH_SKEW + col * BYTES_PER_CHANNEL;
    }

    // Print the content of the tile (only for debug ;)).
    inline __device__ void debug_print() const {
        for( int row = 0; row < ROWS; ++row ) {
            for( int col = 0; col < BYTES_PER_ROW; col += 16 ) {
                if( threadIdx.x == 0 ) {
                    uint4 val;
                    xmma::lds(val, smem_ + row*BYTES_PER_ROW_WITH_SKEW + col);
                    printf("b[%3d][%2d]=(%8.3f, %8.3f, %8.3f, %8.3f)\n",
                        row,
                        col / 16,
                        reinterpret_cast<const float&>(val.x),
                        reinterpret_cast<const float&>(val.y),
                        reinterpret_cast<const float&>(val.z),
                        reinterpret_cast<const float&>(val.w));
                }
            }
        }
    }

    // Move the read offset between iterations of the inner loop.
    inline __device__ void move_read_offset() {
        smem_read_ += Cfg::FLT_R_PER_ITERATION * Cfg::FLT_S * BYTES_PER_TAP;
    }

    // Reset the read offset to its original position.
    inline __device__ void reset_read_offset() {
        smem_read_ -= Cfg::INNER_LOOPS * Cfg::FLT_R_PER_ITERATION * Cfg::FLT_S * BYTES_PER_TAP;

        // // DEBUG.
        // if( threadIdx.x / 4 == 16 ) {
        //     printf("tidx=%3d b::smem_read_   =%5d\n", threadIdx.x, smem_read_);
        // }
        // // END OF DEBUG.
    }

    // Store to the tile in shared memory.
    template< int N >
    inline __device__ void store(const Store_type (&data)[N], int is_active_for_last_sts) {
        static_assert(STS == N, "");
        #pragma unroll
        for( int ii = 0; ii < STS; ++ii ) {
            if( ii < STS - 1 || is_active_for_last_sts ) {
                xmma::sts(smem_write_[ii], data[ii]);
            }
        }
    }

    // The base address in shared memory.
    uint32_t smem_;
    // The store pointers.
    uint32_t smem_write_[STS];
    // The load pointers for the main loop.
    uint32_t smem_read_;
    // The position in the filter.
    int rsi_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Traits, typename Cta_tile, typename Cfg >
struct Smem_tile_b {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Cta_tile, typename Cfg >
struct Smem_tile_b<xmma::Ampere_hmma_tf32_traits<lwtlass::float_tf32_t, lwtlass::float_tf32_t>,
                   Cta_tile,
                   Cfg> 
    : public Smem_tile_base_b<xmma::Ampere_hmma_tf32_traits<lwtlass::float_tf32_t,
                                                                lwtlass::float_tf32_t>,
                              Cta_tile,
                              Cfg> {
    // The traits.
    using Traits = xmma::Ampere_hmma_tf32_traits<lwtlass::float_tf32_t, lwtlass::float_tf32_t>;
    // The base class.
    using Base = Smem_tile_base_b<Traits, Cta_tile, Cfg>;
    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;
    // The fragment.
    using Fragment = xmma::Fragment_b<Traits, xmma::Col>;

    // Ctor.
    inline __device__ Smem_tile_b(void *smem, int tidx) : Base(smem, tidx) {
    }

    // Load from shared memory.
    inline __device__ void load(Fragment (&b)[Xmma_tile::XMMAS_N], int rsi, int ki) {
        // The offset in the K dimension.
        const int immk = ki * Base::TAPS_PER_QUAD * Base::BYTES_PER_TAP;
        // Is the thread loading from a valid position?
        const int is_valid = this->rsi_ < (int) (Cfg::FLT_R * Cfg::FLT_S) - rsi;

        #pragma unroll
        for( int ni = 0; ni < Xmma_tile::XMMAS_N; ++ni ) {
            // Compute the offsets.
            int offset_0 = (2*ni + 0) * Base::TAPS_PER_LDS * Base::BYTES_PER_ROW_WITH_SKEW + immk;
            int offset_1 = (2*ni + 1) * Base::TAPS_PER_LDS * Base::BYTES_PER_ROW_WITH_SKEW + immk;

            // Load the data using 2x LDS.64.
            uint2 tmp_0 = make_uint2(0u, 0u), tmp_1 = make_uint2(0u, 0u);
            if( is_valid ) {
                xmma::lds(tmp_0, this->smem_read_ + offset_0);
                xmma::lds(tmp_1, this->smem_read_ + offset_1);
            }
            b[ni].reg(0) = tmp_0.x;
            b[ni].reg(1) = tmp_0.y;
            b[ni].reg(2) = tmp_1.x;
            b[ni].reg(3) = tmp_1.y;
        }
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace fprop
}  // namespace first_layer
}  // namespace ext
} // namespace xmma

