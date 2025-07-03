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

#include <xmma/ext/batchnorm/bn_stats/callbacks_epilogue.h>
#include <xmma/ext/batchnorm/bn_stats/params.h>
#include <xmma/cta_swizzle.h>
#include <xmma/turing/fragment.h>

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace xmma {
namespace ext {
namespace batchnorm {
namespace bn_stats {

////////////////////////////////////////////////////////////////////////////////////////////////////
template< typename Cta_tile >
struct Batch_norm_fprop_callbacks_epilogue<xmma::Turing_hmma_fp32_traits, Cta_tile, false> {

    // The traits.
    using Traits = xmma::Turing_hmma_fp32_traits;
    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;
    // The aclwmulators.
    using Fragment_aclwmulator = xmma::Fragment_aclwmulator<Traits>;
    // The registers before the swizzle.
    using Fragment_pre_swizzle = xmma::Fragment_epilogue_pre_swizzle<Traits, Cta_tile>;
    // The registers after the swizzle (and the reduction).
    using Fragment_post_swizzle = xmma::Fragment_epilogue_post_swizzle<Traits, Cta_tile>;
    // The registers before storing to GMEM.
    using Fragment_c = xmma::Fragment_c<Traits, Cta_tile>;

    // The number of XMMAs.
    enum { XMMAS_M = Xmma_tile::XMMAS_M, XMMAS_N = Xmma_tile::XMMAS_N };

    // The number of rows in SMEM. Each XMMA computes 16 rows but a thread owns 2 rows that are 
    // reduced before writing to shared memory. Thus, we have 8 rows for the sums and 8 rows for 
    // the sums of squares.
    enum { SMEM_ROWS = 16 };
    // The number of columns per row.
    enum { SMEM_COLS = Cta_tile::N * Cta_tile::WARPS_M };
    // The skew to avoid bank conflicts on stores.
    enum { SMEM_COLS_WITH_SKEW = SMEM_COLS + 8 };
    // The size of a column with skew.
    enum { BYTES_PER_ROW_WITH_SKEW = SMEM_COLS_WITH_SKEW * sizeof(float) };
    // The size of the tile.
    enum { BYTES_PER_TILE = SMEM_ROWS * BYTES_PER_ROW_WITH_SKEW };

    // TODO : Verify this portion ////////////////////
    typename Traits::Epilogue_type alpha_;
    typename Traits::Epilogue_type beta_;
    using Fragment_alpha_pre_swizzle = typename Traits::Epilogue_type;
    using Fragment_alpha_post_swizzle = typename Traits::Epilogue_type;
    using Fragment_beta = typename Traits::Epilogue_type;

    // Before packing to Fragment_c.
    template< typename Epilogue >
    inline __device__
        void pre_pack(Epilogue &epilogue, int mi, int ii, const Fragment_post_swizzle &frag) {
    }

    // A callback function to get alpha.
    template< typename Epilogue >
    inline __device__ void alpha_pre_swizzle(Epilogue &,
                                             int, 
                                             int,
                                             Fragment_alpha_pre_swizzle &alpha) {
        alpha = this->alpha_;
    }   
  
    // A callback function to get alpha.
    template< typename Epilogue >
    inline __device__ void alpha_post_swizzle(Epilogue &,
                                              int, 
                                              int,
                                              Fragment_alpha_post_swizzle &alpha) {
        alpha = this->alpha_;
    }   
  
    // A callback function to get beta.
    template< typename Epilogue >
    inline __device__ void beta(Epilogue &, int, int, Fragment_beta &beta) {
        beta = this->beta_;
    }   
    // TODO : Verify the above portion ////////////////////

    // Ctor.
    template< typename Params >
    inline __device__ Batch_norm_fprop_callbacks_epilogue(const Params &params, 
                                                          char *smem, 
                                                          int bidm,
                                                          int bidn,
                                                          int bidz,
                                                          int tidx,
                                                          int k) 
        : alpha_(params.alpha)
        , beta_(params.beta)
        , params_k_(k)
        , params_bn_partial_sums_ptr_(params.bn_partial_sums_gmem)
        , smem_(smem)
        , bidm_(bidm)
        , bidn_(bidn) {

        // The position of the CTA in the grid.
        if( params.use_horizontal_cta_rasterization ) {
            partial_sums_ = gridDim.y;
        } else {
            partial_sums_ = gridDim.x;
        }

        // Clear the data.
        #pragma unroll
        for( int ni = 0; ni < XMMAS_N; ++ni ) {
            sums_[ni][0] = make_float2(0.f, 0.f);
            sums_[ni][1] = make_float2(0.f, 0.f);

            sums_of_squares_[ni][0] = make_float2(0.f, 0.f);
            sums_of_squares_[ni][1] = make_float2(0.f, 0.f);
        }
    }

    // Ctor.
    template< typename Params >
    inline __device__ Batch_norm_fprop_callbacks_epilogue(const Params &params, 
                                                          char *smem,
                                                          int bidm,
                                                          int bidn,
                                                          int bidz,
                                                          int tidx) 
        : Batch_norm_fprop_callbacks_epilogue(params, smem, bidm, bidn, bidz, tidx, params.g*params.k) {
    }

    // Pre-swizzle aclwmulation (it does nothing in that case).
    template< typename Epilogue >
    inline __device__ 
        void pre_colwert(Epilogue &epilogue, int mi, int ni, const Fragment_aclwmulator &acc) {

        sums_[ni][0].x += acc.elt(0);
        sums_[ni][0].y += acc.elt(1);
        sums_[ni][0].x += acc.elt(2);
        sums_[ni][0].y += acc.elt(3);
        sums_[ni][1].x += acc.elt(4);
        sums_[ni][1].y += acc.elt(5);
        sums_[ni][1].x += acc.elt(6);
        sums_[ni][1].y += acc.elt(7);

        sums_of_squares_[ni][0].x += acc.elt(0) * acc.elt(0);
        sums_of_squares_[ni][0].y += acc.elt(1) * acc.elt(1);
        sums_of_squares_[ni][0].x += acc.elt(2) * acc.elt(2);
        sums_of_squares_[ni][0].y += acc.elt(3) * acc.elt(3);
        sums_of_squares_[ni][1].x += acc.elt(4) * acc.elt(4);
        sums_of_squares_[ni][1].y += acc.elt(5) * acc.elt(5);
        sums_of_squares_[ni][1].x += acc.elt(6) * acc.elt(6);
        sums_of_squares_[ni][1].y += acc.elt(7) * acc.elt(7);
    }

    // Pre-swizzle.
    template< typename Epilogue >
    inline __device__ 
        void pre_swizzle(Epilogue &epilogue, int mi, int ni, const Fragment_pre_swizzle &frag) {
    }

    // Post-swizzle aclwmulation.
    template< typename Epilogue >
    inline __device__ void post_swizzle(Epilogue &epilogue, 
                                        int mi, 
                                        int ii, 
                                        const Fragment_post_swizzle &frag, 
                                        int mask) {
    }

    // Before storing to global memory.
    template< typename Epilogue >
    inline __device__ 
        void pre_store(Epilogue &epilogue, int mi, int ii, const Fragment_c &frag, int) {
    }

    // Do the reduction.
    inline __device__ void post_epilogue() {

        // The number of warps in the CTA.
        const int WARPS_M = Cta_tile::WARPS_M;
        const int WARPS_N = Cta_tile::WARPS_N;
        const int WARPS_K = Cta_tile::WARPS_K;
        
        // The number of warps in the K dimension is 1.
        static_assert(WARPS_K == 1, "Sliced-K not supported");

        // The thread index.
        const int tidx = threadIdx.x;

        // Each XMMA operates on 16 columns/channels. Inside a warp lane 0 owns channels 0 to 7,
        // thread 2 owns 8 to 15, thread 8 owns 16 to 23 and 10 owns 24 to 31. It is possible to
        // write the 16 channels using a single STS.128.

        // Inside a warp we have the following layout of elements for the stores: simple turing format
        //
        // tidx  0 -> row = 0, col =  0B
        // tidx  1 -> row = 0, col =  8B
        // tidx  2 -> row = 0, col = 16B
        // tidx  3 -> row = 0, col = 24B
        // tidx  4 -> row = 1, col =  0B
        // tidx  5 -> row = 1, col =  8B
        // tidx  6 -> row = 1, col = 16B
        // tidx  7 -> row = 1, col = 24B
        // tidx  8 -> row = 2, col =  0B
        // tidx  9 -> row = 2, col =  8B
        // tidx 10 -> row = 2, col = 16B
        // tidx 11 -> row = 2, col = 24B
        // tidx 12 -> row = 3, col =  0B
        // tidx 13 -> row = 3, col =  8B
        // tidx 14 -> row = 3, col = 16B
        // tidx 15 -> row = 3, col = 24B
        // tidx 16 -> row = 4, col =  0B
        // tidx 17 -> row = 4, col =  8B
        // tidx 18 -> row = 4, col = 16B
        // tidx 19 -> row = 4, col = 24B
        // tidx 20 -> row = 5, col =  0B
        // tidx 21 -> row = 5, col =  8B
        // tidx 22 -> row = 5, col = 16B
        // tidx 23 -> row = 5, col = 24B
        // tidx 24 -> row = 6, col =  0B
        // tidx 25 -> row = 6, col =  8B
        // tidx 26 -> row = 6, col = 16B
        // tidx 27 -> row = 6, col = 24B
        // tidx 28 -> row = 7, col =  0B
        // tidx 29 -> row = 7, col =  8B
        // tidx 30 -> row = 7, col = 16B
        // tidx 31 -> row = 7, col = 24B
        //
        
        // The masks to select the warps.
        const int WARP_MASK_M = xmma::Warp_masks<WARPS_M, WARPS_N, WARPS_K>::M;
        const int WARP_MASK_N = xmma::Warp_masks<WARPS_M, WARPS_N, WARPS_K>::N;

        // The divisor for the warps.
        const int WARP_DIV_M =       1 * Cta_tile::THREADS_PER_WARP;
        const int WARP_DIV_N = WARPS_M * Cta_tile::THREADS_PER_WARP;

        // Compute the row and the column in shared memory. Each warp reads from a 16*16B segment.
        const int smem_write_row = (tidx & 0x1f) >> 2;
        const int smem_write_col = (tidx & WARP_MASK_M) / WARP_DIV_M * Cta_tile::N * 4 +
                                   (tidx & WARP_MASK_N) / WARP_DIV_N * 64 +
                                   (tidx & 0x03) * 8;

        // The corresponding offset.
        int smem_write_offset = smem_write_row*BYTES_PER_ROW_WITH_SKEW + smem_write_col;

        // Make sure shared memory can be written.
        __syncthreads();

        // Store the data into shared memory.
        #pragma unroll
        for( int ni = 0; ni < XMMAS_N; ++ni ) {

            // Store the sums.
            int offset_0 = smem_write_offset + ni*Xmma_tile::N_PER_XMMA_PER_CTA*sizeof(float);
            reinterpret_cast<float2*>(&this->smem_[offset_0 +  0])[0] = sums_[ni][0];
            reinterpret_cast<float2*>(&this->smem_[offset_0 + 32])[0] = sums_[ni][1];

            // Store the sums of squares.
            int offset_1 = offset_0 + 8*BYTES_PER_ROW_WITH_SKEW;
            reinterpret_cast<float2*>(&this->smem_[offset_1 +  0])[0] = sums_of_squares_[ni][0];
            reinterpret_cast<float2*>(&this->smem_[offset_1 + 32])[0] = sums_of_squares_[ni][1];

        }

        // Make sure the data is in shared memory.
        __syncthreads();

        // The number of threads working per row. 
        const int THREADS_PER_ROW = Cta_tile::THREADS_PER_CTA / 2;
        // Make sure we never exceed the N dimension.
        static_assert(Cta_tile::N >= THREADS_PER_ROW, "");
        // The number of sums per thread.
        const int SUMS_PER_THREAD = Cta_tile::N / THREADS_PER_ROW;
        // Make sure the number of sums per thread is 1, 2 or 4.
        static_assert(SUMS_PER_THREAD == 1 || SUMS_PER_THREAD == 2 || SUMS_PER_THREAD == 4, "");

        // The fragment to store the running sums.
        using Fragment = xmma::Fragment<float, SUMS_PER_THREAD>;

        // Is the thread computing the sums or the sums of squares.
        const int sums_or_sums_of_sq = tidx / THREADS_PER_ROW;
        // Which channel is that thread computing?
        const int k = tidx % THREADS_PER_ROW;

        // The 1st group of threads works on the sums and the others on the sums of squares.
        const int smem_read_row = sums_or_sums_of_sq * 8;
        // The position in the row (in bytes).
        const int smem_read_col = k * sizeof(Fragment);
        // The starting offset.
        const int smem_read_offset = smem_read_row*BYTES_PER_ROW_WITH_SKEW + smem_read_col;

        // The running sums.
        Fragment res; res.clear();


        // Perfom the reductions.
        // if( sums_or_sums_of_sq < 2 && k*SUMS_PER_THREAD < Cta_tile::N ) {
        for( int i = 0; i < 8; ++i )  {
            for( int j = 0; j < WARPS_M; ++j )  {
                int offset = i*BYTES_PER_ROW_WITH_SKEW + j*Cta_tile::N*sizeof(float);
                const char *ptr = &this->smem_[smem_read_offset + offset];
                res.add(*reinterpret_cast<const Fragment*>(ptr));
            }
        }
        // }


        // The buffer.
        const int buffer = sums_or_sums_of_sq*partial_sums_ + bidm_;
        // The position in the buffer.
        const int idx_in_buffer = bidn_*Cta_tile::N + k*SUMS_PER_THREAD;
        // The offset (in floats) written by this thread.
        const int offset = buffer*params_k_ + idx_in_buffer;

        // Is it in-bound?
        const int is_in_bound = k*SUMS_PER_THREAD < Cta_tile::N && idx_in_buffer < params_k_;
        // Store the partial results.
        char *ptr = reinterpret_cast<char*>(params_bn_partial_sums_ptr_);
        if( sums_or_sums_of_sq < 2 && is_in_bound ) {
            reinterpret_cast<Fragment*>(&ptr[offset*sizeof(float)])[0] = res;
        }
    }

    // The number of output channels.
    const int params_k_;
    // The output pointer.
    void *const params_bn_partial_sums_ptr_;
    // Shared memory.
    char *smem_;
    // The position of the CTA in the grid.
    int bidm_, bidn_, partial_sums_;
    // The sums.
    float2 sums_[XMMAS_N][2];
    // The sums of squares.
    float2 sums_of_squares_[XMMAS_N][2];
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Cta_tile >
struct Batch_norm_fprop_callbacks_epilogue<xmma::Turing_hmma_fp32_traits, Cta_tile, true> {

    // The traits.
    using Traits = xmma::Turing_hmma_fp32_traits;
    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;
    // The aclwmulators.
    using Fragment_aclwmulator = xmma::Fragment_aclwmulator<Traits>;
    // The registers before the swizzle.
    using Fragment_pre_swizzle = xmma::Fragment_epilogue_pre_swizzle<Traits, Cta_tile>;
    // The registers after the swizzle (and the reduction).
    using Fragment_post_swizzle = xmma::Fragment_epilogue_post_swizzle<Traits, Cta_tile>;
    // The registers before storing to GMEM.
    using Fragment_c = xmma::Fragment_c<Traits, Cta_tile>;

    // TODO : Verify this portion ////////////////////
    typename Traits::Epilogue_type alpha_;
    typename Traits::Epilogue_type beta_;
    using Fragment_alpha_pre_swizzle = typename Traits::Epilogue_type;
    using Fragment_alpha_post_swizzle = typename Traits::Epilogue_type;
    using Fragment_beta = typename Traits::Epilogue_type;

    // Before packing to Fragment_c.
    template< typename Epilogue >
    inline __device__
        void pre_pack(Epilogue &epilogue, int mi, int ii, const Fragment_post_swizzle &frag) {
    }

    // A callback function to get alpha.
    template< typename Epilogue >
    inline __device__ void alpha_pre_swizzle(Epilogue &,
                                             int, 
                                             int,
                                             Fragment_alpha_pre_swizzle &alpha) {
        alpha = this->alpha_;
    }   
  
    // A callback function to get alpha.
    template< typename Epilogue >
    inline __device__ void alpha_post_swizzle(Epilogue &,
                                              int, 
                                              int,
                                              Fragment_alpha_post_swizzle &alpha) {
        alpha = this->alpha_;
    }   
  
    // A callback function to get beta.
    template< typename Epilogue >
    inline __device__ void beta(Epilogue &, int, int, Fragment_beta &beta) {
        beta = this->beta_;
    }   
    // TODO : Verify the above portion ////////////////////

    // The number of XMMAs.
    enum { XMMAS_M = Xmma_tile::XMMAS_M, XMMAS_N = Xmma_tile::XMMAS_N };

    // The number of rows in SMEM. That's 4 for the sums and 4 for the sums of squares.
    enum { SMEM_ROWS = 4 };
    // The number of columns per row. Each thread stores 4 floats per row.
    enum { SMEM_COLS = Cta_tile::THREADS_PER_CTA*4 };
    // The skew to avoid bank conflicts on stores.
    enum { SMEM_COLS_WITH_SKEW = SMEM_COLS + 16 };
    // The size of a column with skew.
    enum { BYTES_PER_ROW_WITH_SKEW = SMEM_COLS_WITH_SKEW * sizeof(float) };
    // The size of the tile.
    enum { BYTES_PER_TILE = SMEM_ROWS * BYTES_PER_ROW_WITH_SKEW };

    // Ctor.
    template< typename Params >
    inline __device__ Batch_norm_fprop_callbacks_epilogue(const Params &params, 
                                                          char *smem, 
                                                          int bidm,
                                                          int bidn,
                                                          int bidz,
                                                          int tidx,
                                                          int k) 
        : alpha_(params.alpha)
        , beta_(params.beta)
        , params_k_(k)
        , params_bn_partial_sums_ptr_(params.bn_partial_sums_gmem)
        , smem_(smem)
        , bidm_(bidm)
        , bidn_(bidn) {

        // The position of the CTA in the grid.
        if( params.use_horizontal_cta_rasterization ) {
            partial_sums_ = gridDim.y;
        } else {
            partial_sums_ = gridDim.x;
        }

        // Clear the data.
        sums_[0] = make_float4(0.f, 0.f, 0.f, 0.f);
        sums_[1] = make_float4(0.f, 0.f, 0.f, 0.f);

        sums_of_squares_[0] = make_float4(0.f, 0.f, 0.f, 0.f);
        sums_of_squares_[1] = make_float4(0.f, 0.f, 0.f, 0.f);
    }

    // Ctor.
    template< typename Params >
    inline __device__ Batch_norm_fprop_callbacks_epilogue(const Params &params, 
                                                          char *smem, 
                                                          int bidm,
                                                          int bidn,
                                                          int bidz,
                                                          int tidx)
        : Batch_norm_fprop_callbacks_epilogue(params, smem, bidm, bidn, bidz, tidx, params.g*params.k) {
    }

    // Pre-swizzle aclwmulation (it does nothing in that case).
    template< typename Epilogue >
    inline __device__ 
        void pre_colwert(Epilogue &epilogue, int mi, int ni, const Fragment_aclwmulator &acc) {
    }

    // Pre-swizzle.
    template< typename Epilogue >
    inline __device__ 
        void pre_swizzle(Epilogue &epilogue, int mi, int ni, const Fragment_pre_swizzle &frag) {
    }

    // Post-swizzle aclwmulation.
    template< typename Epilogue >
    inline __device__ void post_swizzle(Epilogue &epilogue, 
                                        int mi, 
                                        int ii, 
                                        const Fragment_post_swizzle &frag, 
                                        int mask) {

        if( mask ) {
            sums_[0].x += frag.elt(0);
            sums_[0].y += frag.elt(1);
            sums_[0].z += frag.elt(2);
            sums_[0].w += frag.elt(3);
            sums_[1].x += frag.elt(4);
            sums_[1].y += frag.elt(5);
            sums_[1].z += frag.elt(6);
            sums_[1].w += frag.elt(7);

            sums_of_squares_[0].x += frag.elt(0) * frag.elt(0);
            sums_of_squares_[0].y += frag.elt(1) * frag.elt(1);
            sums_of_squares_[0].z += frag.elt(2) * frag.elt(2);
            sums_of_squares_[0].w += frag.elt(3) * frag.elt(3);
            sums_of_squares_[1].x += frag.elt(4) * frag.elt(4);
            sums_of_squares_[1].y += frag.elt(5) * frag.elt(5);
            sums_of_squares_[1].z += frag.elt(6) * frag.elt(6);
            sums_of_squares_[1].w += frag.elt(7) * frag.elt(7);
        }
    }

    // Before storing to global memory.
    template< typename Epilogue >
    inline __device__ 
        void pre_store(Epilogue &epilogue, int mi, int ii, const Fragment_c &frag, int) {
    }

    // Store to global memory.
    inline __device__ void post_epilogue() {

        // The thread index.
        const int tidx = threadIdx.x;

        // The shared memory offset.
        int smem_write_offset = tidx*sizeof(float4);

        // Make sure shared memory can be written.
        __syncthreads();

        // The pointer to shared memory.
        char *smem = &this->smem_[smem_write_offset];

        float4 *tmp0 = reinterpret_cast<float4*>(&smem[0*BYTES_PER_ROW_WITH_SKEW]);
        float4 *tmp1 = reinterpret_cast<float4*>(&smem[1*BYTES_PER_ROW_WITH_SKEW]);
        float4 *tmp2 = reinterpret_cast<float4*>(&smem[2*BYTES_PER_ROW_WITH_SKEW]);
        float4 *tmp3 = reinterpret_cast<float4*>(&smem[3*BYTES_PER_ROW_WITH_SKEW]);

        tmp0[0] = sums_[0];
        tmp1[0] = sums_[1];
        tmp2[0] = sums_of_squares_[0];
        tmp3[0] = sums_of_squares_[1];

        // Make sure the data is in shared memory.
        __syncthreads();

        // The number of threads computing a single "row" of outputs.
        const int THREADS_PER_ROW = Cta_tile::THREADS_PER_CTA / 2;
        // The number of sums per thread.
        const int SUMS_PER_THREAD = (Cta_tile::N + THREADS_PER_ROW-1) / THREADS_PER_ROW;
        // Make sure the number of sums per thread is 1, 2 or 4.
        static_assert(SUMS_PER_THREAD == 1 || SUMS_PER_THREAD == 2 || SUMS_PER_THREAD == 4, "");

        // The number of threads needed to cover 4 elements.
        const int THREADS_PER_FLOAT4 = 4 / SUMS_PER_THREAD;

        // The fragment to store the running sums.
        using Fragment = xmma::Fragment<float, SUMS_PER_THREAD>;

        // Make sure the size of the fragment matches our expectations.
        static_assert(sizeof(Fragment) == 4*SUMS_PER_THREAD, "");

        // Is the thread computing the sums or the sums of squares.
        const int sums_or_sums_of_sq = tidx / THREADS_PER_ROW;
        // Which channel is that thread computing?
        const int k = tidx % THREADS_PER_ROW;

        // Where to read the data from. We have to balance between rows as we stored groups of 
        // 4 floats to different rows to avoid bank conflicts.
        int smem_read_row = (k % (THREADS_PER_FLOAT4*2)) / THREADS_PER_FLOAT4;
        int smem_read_col = (k / (THREADS_PER_FLOAT4*2)) * THREADS_PER_FLOAT4 + 
                            (k % (THREADS_PER_FLOAT4  ));

        // The starting offset.
        int smem_read_offset = smem_read_row*BYTES_PER_ROW_WITH_SKEW + 
                               smem_read_col*sizeof(Fragment);

        // Half threads work on the sum of squares.
        smem_read_offset += 2*sums_or_sums_of_sq*BYTES_PER_ROW_WITH_SKEW;

        // The number of reduction steps. Each thread computes 8 channels. Since there are 
        // Cta_tile::N channels per tile, we have Cta_tile::N / 8 threads computing different 
        // channels. In other words, we have THREADS_PER_CTA / Cta_tile::N * 8 different groups of 
        // threads computing the same channels.
        const int STEPS = Cta_tile::THREADS_PER_CTA * 8 / Cta_tile::N;
        // Make sure we have at least one step.
        static_assert(STEPS >= 1, "");

        // The running sums.
        Fragment res; res.clear();
        // Perfom the reductions. TODO: Use a binary tree approach (if possible).
        if( sums_or_sums_of_sq < 2 && k * SUMS_PER_THREAD < Cta_tile::N ) {
            for( int i = 0; i < STEPS; ++i )  {
                int offset = smem_read_offset + i*(Cta_tile::N/2)*sizeof(float);
                Fragment val = reinterpret_cast<const Fragment*>(&this->smem_[offset])[0];
                res.add(val);
            }
        }

        // The buffer.
        const int buffer = sums_or_sums_of_sq*partial_sums_ + bidm_;
        // The position in the buffer.
        const int idx_in_buffer = bidn_*Cta_tile::N + k*SUMS_PER_THREAD;
        // The offset (in floats) written by this thread.
        const int offset = buffer*params_k_ + idx_in_buffer;

        // Is it in-bound?
        const int is_in_bound = k*SUMS_PER_THREAD < Cta_tile::N && idx_in_buffer < params_k_;
        // Store the partial results.
        char *ptr = reinterpret_cast<char*>(params_bn_partial_sums_ptr_);
        if( sums_or_sums_of_sq < 2 && is_in_bound ) {
            reinterpret_cast<Fragment*>(&ptr[offset*sizeof(float)])[0] = res;
        }
    }

    // The number of output channels.
    const int params_k_;
    // The output pointer.
    void *const params_bn_partial_sums_ptr_;
    // Shared memory.
    char *smem_;
    // The position of the CTA in the grid.
    int bidm_, bidn_, partial_sums_;
    // The sums.
    float4 sums_[2];
    // The sums of squares.
    float4 sums_of_squares_[2];
};


////////////////////////////////////////////////////////////////////////////////////////////////////
} // bn_stats
} // namespace batchnorm
} // namespace ext
} // namespace xmma
////////////////////////////////////////////////////////////////////////////////////////////////////
