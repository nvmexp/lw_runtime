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

template< typename Traits, typename Cta_tile, typename Cfg, bool IS_PROLOGUE_TILE >
struct Gmem_tile_a {

    // The dimension of the output tile.
    enum { OUT_D = Cfg::OUT_D };
    enum { OUT_H = Cfg::OUT_H };
    enum { OUT_W = Cfg::OUT_W };

    // The dimension of the filters.
    enum { FLT_K = Cfg::FLT_K };
    enum { FLT_T = Cfg::FLT_T }; 
    enum { FLT_R = Cfg::FLT_R }; 
    enum { FLT_S = Cfg::FLT_S };

    // The padding.
    enum { PAD_D = Cfg::PAD_D };
    enum { PAD_H = Cfg::PAD_H };
    enum { PAD_W = Cfg::PAD_W };

    // The strides.
    enum { STRIDE_D = Cfg::STRIDE_D };
    enum { STRIDE_H = Cfg::STRIDE_H };
    enum { STRIDE_W = Cfg::STRIDE_W };

    // The number of channels per pixel. The layout is NDHW4.
    enum { CHANNELS_PER_PIXEL = Cfg::CHANNELS_PER_PIXEL };
    // The size of each channel in bytes.
    enum { BYTES_PER_CHANNEL = Traits::BITS_PER_ELEMENT_A / 8 };
    // The size of a pixel in bytes.
    enum { BYTES_PER_PIXEL = CHANNELS_PER_PIXEL * BYTES_PER_CHANNEL };
    // The size of each LDG - each thread loads 1 pixel per LDG.
    enum { BYTES_PER_LDG = BYTES_PER_PIXEL };
    // The number of threads needed to load a single pixel.
    enum { THREADS_PER_PIXEL = 1 };
    // The number of pixels loaded per LDG.
    enum { PIXELS_PER_LDG = Cta_tile::THREADS_PER_CTA / THREADS_PER_PIXEL };

    // The size of the input tile in the vertical dimension. 
    enum { IMG_H = IS_PROLOGUE_TILE ? Cfg::IMG_H_IN_PROLOGUE : Cfg::IMG_H_PER_INNER_LOOP };
    // The size of the input tile in the horizontal dimension.
    enum { IMG_W = Cfg::IMG_W };
    // The number of LDGs?
    enum { LDGS = xmma::Div_up<IMG_H * IMG_W, PIXELS_PER_LDG>::VALUE };

    // The number of predicate registers.
    enum { PRED_REGS = xmma::Compute_number_of_pred_regs<LDGS>::VALUE };
    // Make sure we do not need more than 1 register.
    static_assert(PRED_REGS == 1, "");

    // No use of LDGSTS.
    enum { USE_LDGSTS = 0 };

    // Ctor.
    template< typename Params >
    inline __device__ Gmem_tile_a(const Params &params, 
                                  void *smem, 
                                  int cta_n,
                                  int cta_p,
                                  int cta_q, 
                                  int tidx) 
        : params_h_(params.h)
        , params_w_(params.w)
        , params_stride_h_(params.img_stride_h)
        , ptr_(reinterpret_cast<const char*>(params.img_gmem)) {

        // Add the batch offset.
        ptr_ += cta_n * params.img_stride_n * BYTES_PER_PIXEL;

        // The position of the top-left input pixel.
        int cta_h = cta_p * params.out_rows_per_cta * STRIDE_H - PAD_H;
        int cta_w = cta_q * OUT_W                   * STRIDE_W - PAD_W;

        // If that's _not_ the prologue tile, move to 1st row of the "steady load regime".
        if( !IS_PROLOGUE_TILE ) {
            cta_h += Cfg::IMG_H_IN_PROLOGUE;
        }

        // The positions of the elements loaded by this thread.
        #pragma unroll
        for( int ii = 0; ii < LDGS; ++ii ) {
            h_[ii] = cta_h + (tidx + ii * PIXELS_PER_LDG) / IMG_W;
            w_[ii] = cta_w + (tidx + ii * PIXELS_PER_LDG) % IMG_W;
        }

        // Is that thread active for the last LDG?
        is_active_for_last_ldg_ = tidx + (LDGS-1) * PIXELS_PER_LDG < IMG_H * IMG_W;

        // Finalize the initialization by computing the predicates.
        compute_predicates();

        // // DEBUG.
        // if( IS_PROLOGUE_TILE ) {
        //     #pragma unroll
        //     for( int ii = 0; ii < LDGS; ++ii ) {
        //         printf("tidx=%3d ii=%d h_=%3d w_=%3d preds_[0]=0x%08x\n", 
        //             tidx, 
        //             ii, 
        //             h_[ii], 
        //             w_[ii], 
        //             preds_[0]);
        //     }
        // }
        // // END OF DEBUG.
    }

    // Store the pixels to shared memory.
    template< typename Smem_tile > 
    inline __device__ void commit(Smem_tile &smem) {
        smem.store(fetch_, is_active_for_last_ldg_);
    }

    // Compute the predicates.
    inline __device__ void compute_predicates() {
        uint32_t preds[LDGS];
        #pragma unroll
        for( int ii = 0; ii < LDGS; ++ii ) {
            preds[ii] = (unsigned) h_[ii] < params_h_ && (unsigned) w_[ii] < params_w_;
        }
        if( !is_active_for_last_ldg_ ) {
            preds[LDGS-1] = 0;
        }
        preds_[0] = xmma::pack_predicates(preds);
    }

    // Load from global memory.
    template< typename Smem_tile >
    inline __device__ void load(Smem_tile&, uint64_t = xmma::MEM_DESC_DEFAULT) {
        const void* ptrs[LDGS];
        #pragma unroll
        for( int ii = 0; ii < LDGS; ++ii ) {
            ptrs[ii] = &ptr_[(h_[ii] * params_stride_h_ + w_[ii]) * BYTES_PER_PIXEL];
        }
        xmma::ldg(fetch_, ptrs, preds_);
    }

    // The move function. Update the position in the H dimension.
    inline __device__ void move() {
        #pragma unroll
        for( int ii = 0; ii < LDGS; ++ii ) {
            h_[ii] += Cfg::IMG_H_PER_INNER_LOOP;
        }
    }

    // The residue.
    inline __device__ void residue() {
        compute_predicates();
    }

    // The params.
    const int params_h_, params_w_, params_stride_h_;
    // The pointer.
    const char *ptr_;
    // The position of the thread in the image. It moves in the H dimension.
    int h_[LDGS], w_[LDGS];
    // The predicates.
    uint32_t preds_[PRED_REGS];
    // Is that thread active for the last LDG?
    int is_active_for_last_ldg_;
    // The fetch registers.
    typename xmma::Uint_from_size_in_bytes<BYTES_PER_LDG>::Type fetch_[LDGS];
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Traits, typename Cta_tile, typename Cfg >
struct Gmem_tile_b {

    // The number of channels per tap. The layout is KTRS4.
    enum { CHANNELS_PER_TAP = Cfg::CHANNELS_PER_PIXEL };
    // The size of each channel in bytes.
    enum { BYTES_PER_CHANNEL = Traits::BITS_PER_ELEMENT_B / 8 };
    // The size of a tap in bytes.
    enum { BYTES_PER_TAP = CHANNELS_PER_TAP * BYTES_PER_CHANNEL };
    // The size of each LDG - each thread loads 1 tap per LDG.
    enum { BYTES_PER_LDG = BYTES_PER_TAP };
    // The number of threads needed to load a single tap.
    enum { THREADS_PER_TAP = 1 };
    // The number of taps loaded per LDG.
    enum { TAPS_PER_LDG = Cta_tile::THREADS_PER_CTA / THREADS_PER_TAP };
    // The number of taps to load.
    enum { TAPS = Cfg::FLT_K * Cfg::FLT_T * Cfg::FLT_R * Cfg::FLT_S };
    // The number of LDGs?
    enum { LDGS = xmma::Div_up<TAPS, TAPS_PER_LDG>::VALUE };

    // The number of predicate registers.
    enum { PRED_REGS = xmma::Compute_number_of_pred_regs<LDGS>::VALUE };
    // Make sure we do not need more than 1 register.
    static_assert(PRED_REGS == 1, "");

    // No use of LDGSTS.
    enum { USE_LDGSTS = 0 };

    // Ctor.
    template< typename Params >
    inline __device__ Gmem_tile_b(const Params &params, void *smem, int cta_k, int tidx) {

        // Construct the predicates. TODO: Produce an immediate for the predicates.
        uint32_t preds[LDGS];
        #pragma unroll
        for( int ii = 0; ii < LDGS-1; ++ii ) {
            preds[ii] = 1;
        }
        preds[LDGS-1] = is_active_for_last_ldg_ = tidx + (LDGS-1) * TAPS_PER_LDG < TAPS;

        // Pack the predicates.
        preds_[0] = xmma::pack_predicates(preds);

        // The position of the 1st tap loaded by the threads.
        int offset = cta_k * TAPS + tidx;

        // The base pointer.
        ptr_ = &reinterpret_cast<const char*>(params.flt_gmem)[offset * BYTES_PER_TAP];
    }

    // Store the taps to shared memory.
    template< typename Smem_tile > 
    inline __device__ void commit(Smem_tile &smem) {
        smem.store(fetch_, is_active_for_last_ldg_);
    }

    // Load from global memory.
    template<typename Smem_tile >
    inline __device__ void load(Smem_tile&, uint64_t = xmma::MEM_DESC_DEFAULT) {
        const void* ptrs[LDGS];
        #pragma unroll
        for( int ii = 0; ii < LDGS; ++ii ) {
            ptrs[ii] = &ptr_[ii * TAPS_PER_LDG * BYTES_PER_TAP];
        }
        xmma::ldg(fetch_, ptrs, preds_);
    }

    // The pointer.
    const char *ptr_;
    // The predicates.
    uint32_t preds_[PRED_REGS];
    // Is that thread active for the last LDG?
    int is_active_for_last_ldg_;
    // The fetch registers.
    typename xmma::Uint_from_size_in_bytes<BYTES_PER_LDG>::Type fetch_[LDGS];
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Traits, typename Cta_tile, typename Layout, typename Cfg >
struct Gmem_tile_c {

    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;
    // The fragment C.
    using Fragment_c = xmma::Fragment_c<Traits, Cta_tile>;

    // The size of each element in bytes.
    enum { BYTES_PER_ELT = Traits::BITS_PER_ELEMENT_C / 8 };
    // Make sure that's at least 1 byte.
    static_assert(BYTES_PER_ELT >= 1, "");
    // The number of bytes per STG.
    enum { BYTES_PER_STG = Layout::ROW ? 16 : BYTES_PER_ELT };
    // The number of elements per STG per thread.
    enum { ELTS_PER_STG = BYTES_PER_STG / BYTES_PER_ELT };
    // The number of elements per row.
    enum { ELTS_PER_ROW = Layout::ROW ? Cta_tile::N : Cta_tile::M };
    // The number of threads needed to store a row.
    enum { THREADS_PER_ROW = ELTS_PER_ROW / ELTS_PER_STG };
    // The number of rows that are written with a single STG (accross the CTA).
    enum { ROWS_PER_STG = xmma::Max<1, Cta_tile::THREADS_PER_CTA / THREADS_PER_ROW>::VALUE };
    // The number of STGs per row.
    enum { STGS_PER_ROW = xmma::Max<1, THREADS_PER_ROW / Cta_tile::THREADS_PER_CTA>::VALUE };
    // The number of rows to store per XMMA per CTA.
    enum { ROWS_PER_XMMA_PER_CTA = Layout::ROW ? Xmma_tile::M_PER_XMMA_PER_CTA 
                                               : Xmma_tile::N_PER_XMMA_PER_CTA };
    // The number of STGs needed to store the elements per iteration.
    enum { STGS = ROWS_PER_XMMA_PER_CTA / ROWS_PER_STG * STGS_PER_ROW };

    // The number of output pixel stored per STG.
    enum { OUT_HW_PER_STG = Layout::ROW ? ROWS_PER_STG : ELTS_PER_ROW / STGS_PER_ROW };
    // The number of rows.
    enum { OUT_H_PER_STG = OUT_HW_PER_STG / Cfg::OUT_W };
    // Make sure we store at least one row.
    static_assert(OUT_H_PER_STG >= 1, "");
    // The number of output pixel cols stored per STG.
    enum { OUT_W_PER_STG = OUT_HW_PER_STG % Cfg::OUT_W };
    // The number of channels stored per STG.
    enum { OUT_K_PER_STG = Layout::ROW ? ELTS_PER_ROW / STGS_PER_ROW : ROWS_PER_STG };
    // Make sure we store at least one channel.
    static_assert(OUT_K_PER_STG >= 1, "");

    // Ctor.
    template< typename Params >
    inline __device__ Gmem_tile_c(const Params &params, 
                                  int cta_n,
                                  int cta_p,
                                  int cta_q, 
                                  int cta_k,
                                  int tidx)
        : params_n_(params.n)
        , params_p_(params.p)
        , params_q_(params.q)
        , params_k_(params.k) 
        , params_stride_n_(params.out_stride_n) 
        , params_stride_p_(params.out_stride_h) 
        , params_stride_q_(params.out_stride_w) {

        // The location of the pixel/channel in the CTA.
        int pix, chn;
        if( Layout::ROW ) {
            pix = tidx / THREADS_PER_ROW;
            chn = tidx % THREADS_PER_ROW * ELTS_PER_STG;
        } else {
            pix = tidx % THREADS_PER_ROW;
            chn = tidx / THREADS_PER_ROW;
        }

        // The N dimension is the same for all the threads.
        n_ = cta_n;
        // The the number of rows computed per CTA is a parameter.
        p_ = cta_p * params.out_rows_per_cta + pix / Cfg::OUT_W;
        // The Q dimension is fixed.
        q_ = cta_q * Cfg::OUT_W + pix % Cfg::OUT_W;
        // The K dimension depends on the CTA position but that's not clear we will use that.
        k_ = cta_k * Cfg::FLT_K + chn;

        // Setup the destination pointer.
        ptr_ = reinterpret_cast<char*>(params.out_gmem);
    }

    // Compute the position.
    inline __device__ void compute_position(int &n, int &p, int &q, int &k, int si, int ii) const {
        // Decompose the STG index into row/col.
        int mi = ii / STGS_PER_ROW;
        int ni = ii % STGS_PER_ROW;

        // Compute the position in the output tile.
        if( Layout::ROW ) {
            p = p_ + mi * OUT_H_PER_STG + si * OUT_H_PER_STG * STGS;
            k = k_ + ni * OUT_K_PER_STG;
        } else {
            p = p_ + ni * OUT_H_PER_STG;
            k = k_ + mi * OUT_K_PER_STG + si * OUT_K_PER_STG * STGS;
        }

        // N and Q do not change.
        n = n_;
        q = q_;
    }

    // Is a given output valid?
    inline __device__ int compute_output_mask(int si, int ii) const {
        int n, p, q, k;
        compute_position(n, p, q, k, si, ii);
        return n < params_n_ && p < params_p_ && q < params_q_ && k < params_k_;
    }

    // Load the data from global memory.
    inline __device__ void load(Fragment_c &data, 
                                int si, 
                                int ii, 
                                int mask,
                                uint64_t mem_desc = xmma::MEM_DESC_DEFAULT) {
        int n, p, q, k;
        compute_position(n, p, q, k, si, ii);
        int64_t offset = Traits::offset_in_bytes_c(n * params_stride_n_ + 
                                                   p * params_stride_p_ + 
                                                   q * params_stride_q_ + 
                                                   k);
        if( mask ) {
            uint4 tmp;
            xmma::ldg(tmp, ptr_ + offset, mem_desc);
            data.from_int4(tmp);
        }
    }

    // Move to the next chunk of output rows.
    inline __device__ void move() {
        p_ += Cfg::OUT_H;
    }

    // Store the data to global memory.
    inline __device__ void store(int si, 
                                 int ii, 
                                 const Fragment_c &data, 
                                 int mask,
                                 uint64_t mem_desc = xmma::MEM_DESC_DEFAULT) {
        int n, p, q, k;
        compute_position(n, p, q, k, si, ii);
        int64_t offset = Traits::offset_in_bytes_c(n * params_stride_n_ + 
                                                   p * params_stride_p_ + 
                                                   q * params_stride_q_ + 
                                                   k);
        if( mask ) {
            xmma::stg(ptr_ + offset, data.to_int4(), mem_desc);
        }
    }

    // The dimensions of the tile.
    const int params_n_, params_p_, params_q_, params_k_;
    // The strides in the output.
    const int params_stride_n_, params_stride_p_, params_stride_q_;
    // The pointer.
    char *ptr_;
    // The position in the tile.
    int n_, p_, q_, k_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace fprop
}  // namespace first_layer
}  // namespace ext
} // namespace xmma

