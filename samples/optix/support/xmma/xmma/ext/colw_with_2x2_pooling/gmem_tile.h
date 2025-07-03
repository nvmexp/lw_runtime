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

#include <xmma/fragment.h>

namespace xmma {
namespace ext {
namespace colw_with_2x2_pooling {

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Traits, typename Cta_tile, typename Colw_tile, typename Colw_filter >
struct Gmem_tile_a {

    // Make sure it matches the number of pixels computed by the CTA.
    static_assert(Colw_tile::H * Colw_tile::W == Cta_tile::M, "Size mismatch!!!");

    // The vertical dimension of the padding for the colwolution.
    enum { COLW_PADDING_H = Colw_filter::FLT_R / 2 };
    // The horizontal dimension of the padding for the colwolution.
    enum { COLW_PADDING_W = Colw_filter::FLT_S / 2 };

    // The number of rows in the input tile.
    enum { COLW_IN_H = Colw_tile::H + 2*COLW_PADDING_H };
    // The number of columns in the input tile.
    enum { COLW_IN_W = Colw_tile::W + 2*COLW_PADDING_W };

    // The number of rows in the output of the pooling.
    enum { POOLING_OUT_H = Colw_tile::H / 2 };
    // The number of columns in the output of the pooling.
    enum { POOLING_OUT_W = Colw_tile::W / 2 };

    // The size of each LDG.
    enum { BYTES_PER_LDG = 16 };
    // The number of elements per LDG.
    enum { ELEMENTS_PER_LDG = BYTES_PER_LDG * 8 / Traits::BITS_PER_ELEMENT_A };
    // Make sure the number of elements per LDG is compatible with the Cta_tile::K dimension.
    static_assert(Cta_tile::K % ELEMENTS_PER_LDG == 0, "");

    // The number of threads needed to load a pixel.
    enum { THREADS_PER_PIXEL = Cta_tile::K / ELEMENTS_PER_LDG };
    // The number of images loaded per LDG.
    enum { PIXELS_PER_LDG = Cta_tile::THREADS_PER_CTA / THREADS_PER_PIXEL };
    // The number of steps needed to load a tile.
    enum { LDGS = xmma::Div_up<COLW_IN_H * COLW_IN_W, PIXELS_PER_LDG>::VALUE };

    // We do not use LDGSTS for that Gmem tile.
    enum { USE_LDGSTS = 0 };

    // The layout in shared memory.
    using Smem_layout = xmma::Row;

    // Ctor.
    template< typename Params >
    inline __device__ Gmem_tile_a(const Params &params, void*, const dim3 &bidx, int tidx)
        : params_c_(params.c) {

        // The position in the C dimension.
        c_ = bidx.z * Cta_tile::K + (tidx % THREADS_PER_PIXEL * ELEMENTS_PER_LDG);

        // Extract the "top-left" position for the CTA.
        int cta_p, cta_q;
        xmma::fast_divmod(cta_p, cta_q, bidx.x, params.ctas_q, 
                                                    params.mul_ctas_q, 
                                                    params.shr_ctas_q);

        // The position of the top-left element computed by the CTA after pooling.
        int pooling_cta_p = cta_p * POOLING_OUT_H;
        int pooling_cta_q = cta_q * POOLING_OUT_W;

        // The corresponding position in the output of the colwolution.
        int colw_cta_p = pooling_cta_p * 2;
        int colw_cta_q = pooling_cta_q * 2;

        // The position in the input of the colwolution.
        int colw_cta_h = colw_cta_p - (int) COLW_PADDING_H;
        int colw_cta_w = colw_cta_q - (int) COLW_PADDING_W;

        // For each LDG, compute the location of the element loaded by the thread.
        int colw_h[LDGS], colw_w[LDGS]; uint32_t preds[LDGS];
        #pragma unroll
        for( int ii = 0; ii < LDGS; ++ii ) {

            // Decompose the thread index.
            const int idx = tidx / THREADS_PER_PIXEL + ii * PIXELS_PER_LDG;

            // Compute h and w. We implement a cross-correlation.
            colw_h[ii] = colw_cta_h + idx / COLW_IN_W;
            colw_w[ii] = colw_cta_w + idx % COLW_IN_W;

            // Assemble the predicate.
            preds[ii] = (unsigned) colw_h[ii] < params.h && (unsigned) colw_w[ii] < params.w;

            // Create the pointer.
            offsets_[ii] = colw_h[ii]*params.wc + colw_w[ii]*params.c + c_;
        }

        // Assemble the predicates.
        xmma::pack_predicates(preds_, preds);

        // The base pointer.
        ptr_ = reinterpret_cast<const char*>(params.img_gmem);
    } 

    // Store the pixels to shared memory.
    template< typename Smem_tile >
    inline __device__ void commit(Smem_tile &smem) {
        smem.store(fetch_, preds_[0]);
    }

    // Disable the loads.
    inline __device__ void disable_loads() {
        preds_[0] = 0u;
    }

    // Load a tile from global memory.
    template< typename Smem_tile >
    inline __device__ void load(Smem_tile&, uint64_t = xmma::MEM_DESC_DEFAULT) {
        const void *ptrs[LDGS];
        #pragma unroll
        for( int ii = 0; ii < LDGS; ++ii ) {
            ptrs[ii] = ptr_ + Traits::offset_in_bytes_a(offsets_[ii]);
        }
        
        uint32_t tmp[1];
        tmp[0] = c_ < params_c_ ? preds_[0] : 0;
        xmma::ldg(fetch_, ptrs, tmp);
    }

    // Move the pointers and update the predicates for R2P/P2R.
    inline __device__ void move(int64_t delta, int first = 0) {
        ptr_ += delta;
        c_ += gridDim.z*Cta_tile::K;
    }

    // The residue code.
    inline __device__ void residue() {
    }

    // The C dimension.
    const int params_c_;
    // The pointer.
    const char *ptr_;
    // The offsets to move the pointers.
    int offsets_[LDGS];
    // Make sure we have a single predicate.
    static_assert(xmma::Compute_number_of_pred_regs<LDGS>::VALUE == 1, "");
    // The predicates.
    uint32_t preds_[1];
    // The C dimension to disable loads.
    int c_;
    // The fetch registers.
    typename xmma::Uint_from_size_in_bytes<BYTES_PER_LDG>::Type fetch_[LDGS];
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Traits, typename Cta_tile, typename Colw_tile >
struct Gmem_tile_c {

    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;
    // The number of threads per CTA.
    enum { THREADS_PER_CTA = Cta_tile::THREADS_PER_CTA };

    // The pooling factor is 2 in both dimension.
    enum { POOLING = 2 };
    // The number of columns in the tile before the pooling.
    enum { PIXELS_PER_ROW = Colw_tile::W };
    // The number of rows in the tile before the pooling.
    enum { ROWS = Colw_tile::H };

    // The number of rows after pooling.
    enum { ROWS_AFTER_POOLING = ROWS / POOLING };
    // The number of cols after pooling.
    enum { PIXELS_PER_ROW_AFTER_POOLING = PIXELS_PER_ROW / POOLING };

    // The number of pixels produced by a single XMMA instruction across the CTA after pooling.
    enum { PIXELS_PER_XMMA = Xmma_tile::M_PER_XMMA_PER_CTA };
    // Make sure we produce entire rows.
    static_assert(PIXELS_PER_XMMA % PIXELS_PER_ROW == 0, "");
    // The number of rows produced per XMMA.
    enum { ROWS_PER_XMMA = PIXELS_PER_XMMA / PIXELS_PER_ROW };
    // Make sure we produce enough rows to do pooling.
    static_assert(ROWS_PER_XMMA % POOLING == 0, "");

    // To support arbitrary N, we pad some values to a power-of-2.
    enum { CHANNELS_PER_PIXEL = xmma::Next_power_of_two<Cta_tile::N>::VALUE }; 
    // The total number of elements to store _before_ pooling.
    enum { CHANNELS_PER_XMMA = PIXELS_PER_XMMA * CHANNELS_PER_PIXEL };

    // The size of a single element for output in bits.
    enum { BITS_PER_CHANNEL = Traits::BITS_PER_ELEMENT_A };
    // The number of channels in 16B.
    enum { CHANNELS_IN_16B = 128 / BITS_PER_CHANNEL };
    // The total number of elements to store _after_ pooling.
    enum { CHANNELS_PER_XMMA_AFTER_POOLING = CHANNELS_PER_XMMA / POOLING / POOLING };
    // Figure out how many threads per pixel to use so that all threads have stuff to do.
    enum { CHANNELS_PER_THREAD_ = CHANNELS_PER_XMMA_AFTER_POOLING / THREADS_PER_CTA };
    // Make sure we do not exceed 16B.
    enum { CHANNELS_PER_THREAD = xmma::Min<CHANNELS_PER_THREAD_, CHANNELS_IN_16B>::VALUE };
    // The size of a single STG.
    enum { BYTES_PER_STG = CHANNELS_PER_THREAD * BITS_PER_CHANNEL / 8 };

    // Make sure the amount of bytes per STG is "valid".
    static_assert(BYTES_PER_STG == (int) xmma::Next_power_of_two<BYTES_PER_STG>::VALUE, "");
    // Make sure the amount of bytes per STG is at least 4B.
    static_assert(BYTES_PER_STG >= 4, "");

    // The number fo threads per pixel.
    enum { THREADS_PER_PIXEL = CHANNELS_PER_PIXEL / CHANNELS_PER_THREAD };
    // How many STGs do we need to process the output of a single XMMA.
    enum { PIXELS_PER_STG = THREADS_PER_CTA / THREADS_PER_PIXEL };
    // Make sure we store entire rows per STG.
    static_assert(PIXELS_PER_STG % (PIXELS_PER_ROW / POOLING) == 0, "");
    // The number of STGs. Divide by 4 to account for the pooling.
    enum { STGS = PIXELS_PER_XMMA / POOLING / POOLING / PIXELS_PER_STG };

    // The layout of the output.
    using Layout = xmma::Row;
    // The fragment stored by that tile.
    using Fragment_c = xmma::Fragment_c<Traits, Cta_tile, CHANNELS_PER_THREAD>;

    // The names that are used by other GMEM tiles.
    enum { THREADS_PER_ROW = THREADS_PER_PIXEL, ELEMENTS_PER_STG = CHANNELS_PER_THREAD };

    // Ctor.
    template< typename Params >
    inline __device__ Gmem_tile_c(const Params &params, int bidm, int bidn, int bidz, int tidx) 
        : params_p_(params.p)
        , params_q_(params.q)
        , params_k_(params.k) {

        // Extract the "top-left" position for the CTA.
        int cta_p, cta_q;
        xmma::fast_divmod(cta_p, cta_q, bidm, params.ctas_q, 
                                                  params.mul_ctas_q, 
                                                  params.shr_ctas_q);

        // The position of the top-left element computed by the CTA.
        cta_p = cta_p * ROWS_AFTER_POOLING;
        cta_q = cta_q * PIXELS_PER_ROW_AFTER_POOLING;

        // The location of the thread in the tile.
        int p = cta_p + tidx / THREADS_PER_PIXEL / PIXELS_PER_ROW_AFTER_POOLING;
        int q = cta_q + tidx / THREADS_PER_PIXEL % PIXELS_PER_ROW_AFTER_POOLING;

        // Record the p dimension.
        p_ = p;

        // Compute the output position for each thread in the K dimension inside the CTA.
        int k_in_cta = tidx % THREADS_PER_PIXEL * CHANNELS_PER_THREAD;
        // Compute the output position in the grid.
        int k = bidn * Cta_tile::N + k_in_cta;
        // Is K valid?
        is_valid_ = q < params.q && k_in_cta < Cta_tile::N && k < params.k;

        // The base offset.
        int offset = q * params.k + k;
        // The pointer.
        ptr_ = reinterpret_cast<char*>(params.out_gmem) + Traits::offset_in_bytes_c(offset);
    }

    // Is it a valid output?
    inline __device__ int compute_output_mask(int mi, int ii) {
        // The number of rows produced per XMMA.
        enum { ROWS_PER_XMMA_AFTER_POOLING = ROWS_PER_XMMA / POOLING };

        // The coordinates in the different dimensions.
        int pi = p_ + mi * ROWS_PER_XMMA_AFTER_POOLING + ii * ROWS_PER_XMMA_AFTER_POOLING / STGS;

        // The offset.
        offsets_[ii] = pi*params_q_*params_k_;

        // Output if the coordinates are in bound.
        return pi < params_p_ && is_valid_;
    }

    // Load the data from global memory.
    inline __device__ void load(Fragment_c &data, int mi, int ii, int mask, uint64_t) {
    }

    // Store the data to global memory.
    inline __device__ void store(int mi, 
                                 int ii, 
                                 const Fragment_c &data, 
                                 int mask, 
                                 uint64_t = xmma::MEM_DESC_DEFAULT) {
        if( mask ) {
            data.stg(ptr_ + Traits::offset_in_bytes_c(offsets_[ii]));
        }
    }

    // The dimensions of the output.
    const int params_p_, params_q_, params_k_;
    // The position of the thread in the vertical dimension.
    int p_;
    // Is the thread writing a valid position?
    int is_valid_;
    // The pointers to global memory.
    char *ptr_;
    // The offset.
    int offsets_[STGS];
};

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace colw_with_2x2_pooling
} // namespace ext
} // namespace xmma

