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
#include <xmma/turing/smem_tile_with_halo.h>

namespace xmma {

////////////////////////////////////////////////////////////////////////////////////////////////////

template< 
    // The instruction traits.
    typename Traits, 
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile, 
    // The dimensions of the tile stored in shared memory (in pixels).
    typename Pixel_tile, 
    // The extra halo (in total i.e. 2x2 stands for a halo of 1 pixel on each dimension).
    typename Halo, 
    // The number of buffers per tile.
    int BUFFERS_PER_TILE_,
    // The number of elements per "packet" (i.e. NC/8HW8 => 8, NHWC => Cta_tile::K).
    int ELEMENTS_PER_PACKET_
>
struct Smem_tile_with_halo_ampere_row_a 
    : public Smem_tile_with_halo_turing_row_a<Traits,
                                              Cta_tile,
                                              Pixel_tile,
                                              Halo,
                                              BUFFERS_PER_TILE_,
                                              ELEMENTS_PER_PACKET_> {

    // The base class.
    using Base = Smem_tile_with_halo_turing_row_a<Traits,
                                                  Cta_tile,
                                                  Pixel_tile,
                                                  Halo,
                                                  BUFFERS_PER_TILE_,
                                                  ELEMENTS_PER_PACKET_>;
    // The XMMA tile.
    using Xmma_tile = typename Base::Xmma_tile;
    // The fragment.
    using Fragment = typename Base::Fragment;

    // Ctor.
    inline __device__ Smem_tile_with_halo_ampere_row_a(void *smem, int tidx) : Base(smem) {

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
        int row = (tidx & WARP_MASK_K) / WARP_DIV_K * Xmma_tile::XMMAS_K * 2 +
                  (tidx & 0x10) / 16;
        int col = (tidx & WARP_MASK_M) / WARP_DIV_M * 16 + 
                  (tidx & 0x0f);

        // Compute the read offset.
        this->compute_read_offset_(row, col);
        // Compute the write offset.
        this->compute_write_offset_(tidx);

        // Use URF for the read/write buffers.
        this->read_buffer_ = this->write_buffer_ = __shfl_sync(0xffffffff, 0, 0);
    }

    // Load from shared memory.
    inline __device__ void load(Fragment (&a)[Xmma_tile::XMMAS_M], int ki, int ri, int si) const {
        // Load the different elements.
        #pragma unroll
        for( int mi = 0; mi < Xmma_tile::XMMAS_M; ++mi ) {
            // The immediate in shared memory where to grab the element from.
            int imm = this->compute_load_offset_(mi, ri, si) + ki*2*Base::BYTES_PER_ROW_WITH_SKEW;

            // Read the element.
            uint4 tmp;
            xmma::ldsm(tmp, this->smem_ + this->read_offset_ + this->read_buffer_ + imm);
            a[mi].reg(0) = tmp.x;
            a[mi].reg(1) = tmp.y;
            a[mi].reg(2) = tmp.z;
            a[mi].reg(3) = tmp.w;
        }
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Cta_tile, typename Pixel_tile, typename Halo, int BUFFERS_PER_TILE >
struct Smem_tile_with_halo_a<Ampere_hmma_fp16_traits, 
                             Cta_tile, 
                             Row, 
                             Pixel_tile, 
                             Halo, 
                             BUFFERS_PER_TILE>
    : public Smem_tile_with_halo_ampere_row_a<Ampere_hmma_fp16_traits, 
                                              Cta_tile, 
                                              Pixel_tile, 
                                              Halo, 
                                              BUFFERS_PER_TILE,
                                              Cta_tile::K> {

    // The traits class.
    using Traits = Ampere_hmma_fp16_traits;
    // The base class.
    using Base = Smem_tile_with_halo_ampere_row_a<Traits, 
                                                  Cta_tile, 
                                                  Pixel_tile, 
                                                  Halo, 
                                                  BUFFERS_PER_TILE,
                                                  Cta_tile::K>;

    // Ctor.
    inline __device__ Smem_tile_with_halo_a(void *smem, int tidx) 
        : Base(smem, tidx) {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Cta_tile, typename Pixel_tile, typename Halo, int BUFFERS_PER_TILE >
struct Smem_tile_with_halo_a<Ampere_hmma_fp16_traits, 
                             Cta_tile, 
                             Col_interleaved, 
                             Pixel_tile, 
                             Halo, 
                             BUFFERS_PER_TILE>
    : public Smem_tile_with_halo_ampere_row_a<Ampere_hmma_fp16_traits, 
                                              Cta_tile, 
                                              Pixel_tile, 
                                              Halo, 
                                              BUFFERS_PER_TILE,
                                              8> {

    // The traits class.
    using Traits = Ampere_hmma_fp16_traits;
    // The base class.
    using Base = Smem_tile_with_halo_ampere_row_a<Traits, 
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
struct Smem_tile_with_halo_a<Ampere_hmma_fp32_traits, 
                             Cta_tile, 
                             Row, 
                             Pixel_tile, 
                             Halo, 
                             BUFFERS_PER_TILE>
    : public Smem_tile_with_halo_ampere_row_a<Ampere_hmma_fp32_traits, 
                                              Cta_tile, 
                                              Pixel_tile, 
                                              Halo, 
                                              BUFFERS_PER_TILE,
                                              Cta_tile::K> {

    // The traits class.
    using Traits = Ampere_hmma_fp32_traits;
    // The base class.
    using Base = Smem_tile_with_halo_ampere_row_a<Traits, 
                                                  Cta_tile, 
                                                  Pixel_tile, 
                                                  Halo, 
                                                  BUFFERS_PER_TILE,
                                                  Cta_tile::K>;

    // Ctor.
    inline __device__ Smem_tile_with_halo_a(void *smem, int tidx) 
        : Base(smem, tidx) {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Cta_tile, typename Pixel_tile, typename Halo, int BUFFERS_PER_TILE >
struct Smem_tile_with_halo_a<Ampere_hmma_fp32_traits, 
                             Cta_tile, 
                             Col_interleaved, 
                             Pixel_tile, 
                             Halo, 
                             BUFFERS_PER_TILE>
    : public Smem_tile_with_halo_ampere_row_a<Ampere_hmma_fp32_traits, 
                                              Cta_tile, 
                                              Pixel_tile, 
                                              Halo, 
                                              BUFFERS_PER_TILE,
                                              8> {

    // The traits class.
    using Traits = Ampere_hmma_fp32_traits;
    // The base class.
    using Base = Smem_tile_with_halo_ampere_row_a<Traits, 
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

} // namespace xmma 

