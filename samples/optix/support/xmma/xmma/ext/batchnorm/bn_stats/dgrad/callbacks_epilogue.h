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

#include <xmma/xmma.h>
#include <xmma/ext/batchnorm/relu_bitmask_format.h>
////////////////////////////////////////////////////////////////////////////////////////////////////

namespace xmma {
namespace ext {
namespace batchnorm {
namespace bn_stats {
namespace dgrad {
////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Cta_tile, typename Activation_, bool DUAL_DBNS_>
struct Batch_norm_dgrad_callbacks_epilogue_base {

    // The traits.`
    using Traits = xmma::Ampere_hmma_fp32_traits;

    // Activation
    using Activation = Activation_;

    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;
    // The aclwmulators.
    using Fragment_aclwmulator = xmma::Fragment_aclwmulator<Traits>;
    // The registers before the swizzle.
    using Fragment_pre_swizzle = xmma::Fragment_epilogue_pre_swizzle_bn_stats<Traits, Cta_tile>;
    // The registers after the swizzle (and the reduction).
    using Fragment_post_swizzle = xmma::Fragment_epilogue_post_swizzle_bn_stats<Traits, Cta_tile>;
    // The registers before storing to GMEM.
    using Fragment_c = xmma::Fragment_hmma_fp32_c_bn_stats<Traits, Cta_tile>;

    // TODO : Verify this portion ////////////////////
    typename Traits::Epilogue_type alpha_;
    typename Traits::Epilogue_type beta_;
    using Fragment_alpha_pre_swizzle = typename Traits::Epilogue_type;
    using Fragment_alpha_post_swizzle = typename Traits::Epilogue_type;
    using Fragment_beta = typename Traits::Epilogue_type;

    // Before packing to Fragment_c.
    template <typename Epilogue>
    inline __device__ void
    pre_pack( Epilogue &epilogue, int mi, int ii, const Fragment_post_swizzle &frag ) {
    }

    // A callback function to get alpha.
    template <typename Epilogue>
    inline __device__ void
    alpha_pre_swizzle( Epilogue &, int, int, Fragment_alpha_pre_swizzle &alpha ) {
        alpha = this->alpha_;
    }

    // A callback function to get alpha.
    template <typename Epilogue>
    inline __device__ void
    alpha_post_swizzle( Epilogue &, int, int, Fragment_alpha_post_swizzle &alpha ) {
        alpha = this->alpha_;
    }

    // A callback function to get beta.
    template <typename Epilogue>
    inline __device__ void beta( Epilogue &, int, int, Fragment_beta &beta ) {
        beta = this->beta_;
    }
    // TODO : Verify the above portion ////////////////////

    // The number of XMMAs.
    enum { XMMAS_M = Xmma_tile::XMMAS_M, XMMAS_N = Xmma_tile::XMMAS_N };

    // The number of rows in SMEM. That's 4 for the sums and 4 for the sums of squares.
    enum { SMEM_ROWS = 4 };
    // The number of columns per row. Each thread stores 4 floats per row.
    enum { SMEM_COLS = Cta_tile::THREADS_PER_CTA * 4 };
    // The skew to avoid bank conflicts on stores.
    enum { SMEM_COLS_WITH_SKEW = SMEM_COLS + 16 };
    // The size of a column with skew.
    enum { BYTES_PER_ROW_WITH_SKEW = SMEM_COLS_WITH_SKEW * sizeof( float ) };
    // The size of the tile.
    enum { BYTES_PER_TILE = SMEM_ROWS * BYTES_PER_ROW_WITH_SKEW };

    // Ctor.
    template <typename Params>
    inline __device__ Batch_norm_dgrad_callbacks_epilogue_base( const Params &params,
                                                                char *smem,
                                                                int bidm,
                                                                int bidn,
                                                                int bidz,
                                                                int tidx,
                                                                int c )
        : alpha_( params.alpha ), beta_( params.beta ), params_c_( c ),
          params_bn_partial_sums_ptr_( params.bn_partial_sums_gmem ), smem_( smem ), bidm_( bidm ),
          bidn_( bidn ) {

        // The position of the CTA in the grid.
        if( params.use_horizontal_cta_rasterization ) {
            partial_sums_ = gridDim.y;
        } else {
            partial_sums_ = gridDim.x;
        }

        // Clear the data.
        sums_[0] = make_float4( 0.f, 0.f, 0.f, 0.f );
        sums_[1] = make_float4( 0.f, 0.f, 0.f, 0.f );

        sums_of_squares_[0] = make_float4( 0.f, 0.f, 0.f, 0.f );
        sums_of_squares_[1] = make_float4( 0.f, 0.f, 0.f, 0.f );
    }

    // Ctor.
    template <typename Params>
    inline __device__ Batch_norm_dgrad_callbacks_epilogue_base( const Params &params,
                                                                char *smem,
                                                                int bidm,
                                                                int bidn,
                                                                int bidz,
                                                                int tidx )
        : Batch_norm_dgrad_callbacks_epilogue_base( params,
                                                    smem,
                                                    bidm,
                                                    bidn,
                                                    bidz,
                                                    tidx,
                                                    params.g * params.c ) {
    }

    // Pre-swizzle aclwmulation (it does nothing in that case).
    template <typename Epilogue>
    inline __device__ void
    pre_colwert( Epilogue &epilogue, int mi, int ni, const Fragment_aclwmulator &acc ) {
    }

    // Pre-swizzle.
    template <typename Epilogue>
    inline __device__ void
    pre_swizzle( Epilogue &epilogue, int mi, int ni, const Fragment_pre_swizzle &frag ) {
    }

    // Post-swizzle aclwmulation.
    template <typename Epilogue>
    inline __device__ void post_swizzle( Epilogue &epilogue,
                                         int mi,
                                         int ii,
                                         const Fragment_post_swizzle &frag,
                                         const Fragment<float, 8> &mean,
                                         const Fragment<float, 8> &ilwstd,
                                         const Fragment_c &xfrag,
                                         int mask ) {

        if( mask ) {

            float2 tmp[frag.NUM_REGS];
            float2 xtmp[xfrag.NUM_REGS];

            Activation::execute( tmp, xtmp, frag, xfrag, mean, ilwstd );

            sums_[0].x += tmp[0].x;
            sums_[0].y += tmp[0].y;
            sums_[0].z += tmp[1].x;
            sums_[0].w += tmp[1].y;

            sums_[1].x += tmp[2].x;
            sums_[1].y += tmp[2].y;
            sums_[1].z += tmp[3].x;
            sums_[1].w += tmp[3].y;

            sums_of_squares_[0].x += tmp[0].x * ( xtmp[0].x - mean.elt( 0 ) );
            sums_of_squares_[0].y += tmp[0].y * ( xtmp[0].y - mean.elt( 1 ) );
            sums_of_squares_[0].z += tmp[1].x * ( xtmp[1].x - mean.elt( 2 ) );
            sums_of_squares_[0].w += tmp[1].y * ( xtmp[1].y - mean.elt( 3 ) );
            sums_of_squares_[1].x += tmp[2].x * ( xtmp[2].x - mean.elt( 4 ) );
            sums_of_squares_[1].y += tmp[2].y * ( xtmp[2].y - mean.elt( 5 ) );
            sums_of_squares_[1].z += tmp[3].x * ( xtmp[3].x - mean.elt( 6 ) );
            sums_of_squares_[1].w += tmp[3].y * ( xtmp[3].y - mean.elt( 7 ) );
        }
    }

    // Before storing to global memory.
    template <typename Epilogue>
    inline __device__ void
    pre_store( Epilogue &epilogue, int mi, int ii, const Fragment_c &frag, int ) {
    }

    // Store to global memory.
    inline __device__ void post_epilogue() {

        // The thread index.
        const int tidx = threadIdx.x;

        // The shared memory offset.
        int smem_write_offset = tidx * sizeof( float4 );

        // Make sure shared memory can be written.
        __syncthreads();

        // The pointer to shared memory.
        char *smem = &this->smem_[smem_write_offset];

        float4 *tmp0 = reinterpret_cast<float4 *>( &smem[0 * BYTES_PER_ROW_WITH_SKEW] );
        float4 *tmp1 = reinterpret_cast<float4 *>( &smem[1 * BYTES_PER_ROW_WITH_SKEW] );
        float4 *tmp2 = reinterpret_cast<float4 *>( &smem[2 * BYTES_PER_ROW_WITH_SKEW] );
        float4 *tmp3 = reinterpret_cast<float4 *>( &smem[3 * BYTES_PER_ROW_WITH_SKEW] );

        tmp0[0] = sums_[0];
        tmp1[0] = sums_[1];
        tmp2[0] = sums_of_squares_[0];
        tmp3[0] = sums_of_squares_[1];

        // Make sure the data is in shared memory.
        __syncthreads();

        // The number of threads computing a single "row" of outputs.
        const int THREADS_PER_ROW = Cta_tile::THREADS_PER_CTA / 2;

        // The number of sums per thread.
        const int SUMS_PER_THREAD = ( Cta_tile::N + THREADS_PER_ROW - 1 ) / THREADS_PER_ROW;
        // Make sure the number of sums per thread is 1, 2 or 4.
        static_assert( SUMS_PER_THREAD == 1 || SUMS_PER_THREAD == 2 || SUMS_PER_THREAD == 4, "" );

        // The number of threads needed to cover 4 elements.
        const int THREADS_PER_FLOAT4 = 4 / SUMS_PER_THREAD;

        // The fragment to store the running sums.
        using Fragment = xmma::Fragment<float, SUMS_PER_THREAD>;

        // Make sure the size of the fragment matches our expectations.
        static_assert( sizeof( Fragment ) == 4 * SUMS_PER_THREAD, "" );

        // Is the thread computing the sums or the sums of squares.
        const int sums_or_sums_of_sq = tidx / THREADS_PER_ROW;
        // Which channel is that thread computing?
        const int k = tidx % THREADS_PER_ROW;

        // Where to read the data from. We have to balance between rows as we stored groups of
        // 4 floats to different rows to avoid bank conflicts.
        int smem_read_row = ( k % ( THREADS_PER_FLOAT4 * 2 ) ) / THREADS_PER_FLOAT4;
        int smem_read_col = ( k / ( THREADS_PER_FLOAT4 * 2 ) ) * THREADS_PER_FLOAT4 +
                            ( k % ( THREADS_PER_FLOAT4 ) );

        // The starting offset.
        int smem_read_offset =
            smem_read_row * BYTES_PER_ROW_WITH_SKEW + smem_read_col * sizeof( Fragment );

        // Half threads work on the sum of squares.
        smem_read_offset += 2 * sums_or_sums_of_sq * BYTES_PER_ROW_WITH_SKEW;

        // The number of reduction steps. Each thread computes 8 channels. Since there are
        // Cta_tile::N channels per tile, we have Cta_tile::N / 8 threads computing different
        // channels. In other words, we have THREADS_PER_CTA / Cta_tile::N * 8 different groups of
        // threads computing the same channels.
        const int STEPS = Cta_tile::THREADS_PER_CTA * 8 / Cta_tile::N;
        // Make sure we have at least one step.
        static_assert( STEPS >= 1, "" );

        // The running sums.
        Fragment res;
        res.clear();
        // Perfom the reductions. TODO: Use a binary tree approach (if possible).
        if( sums_or_sums_of_sq < 2 && k * SUMS_PER_THREAD < Cta_tile::N ) {
            for( int i = 0; i < STEPS; ++i ) {
                int offset = smem_read_offset + i * ( Cta_tile::N / 2 ) * sizeof( float );
                Fragment val = reinterpret_cast<const Fragment *>( &this->smem_[offset] )[0];
                res.add( val );
            }
        }

        // The buffer.
        const int buffer = sums_or_sums_of_sq * partial_sums_ + bidm_;
        // The position in the buffer.
        const int idx_in_buffer = bidn_ * Cta_tile::N + k * SUMS_PER_THREAD;
        // The offset (in floats) written by this thread.
        const int offset = buffer * params_c_ + idx_in_buffer;

        // Is it in-bound?
        const int is_in_bound = k * SUMS_PER_THREAD < Cta_tile::N && idx_in_buffer < params_c_;
        // Store the partial results.
        char *ptr = reinterpret_cast<char *>( params_bn_partial_sums_ptr_ );
        if( sums_or_sums_of_sq < 2 && is_in_bound ) {
            reinterpret_cast<Fragment *>( &ptr[offset * sizeof( float )] )[0] = res;
        }
    }

    // The number of output channels.
    const int params_c_;
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

template <typename Traits,
          typename Cta_tile,
          typename Activation,
          xmma::ext::batchnorm::ReluBitmaskFormat ReluBitMaskFormat_ =
              xmma::ext::batchnorm::ReluBitmaskFormat::NONE,
          bool DUAL_DBNS_ = false,
          bool IN_CTA_SPLIT_K = ( Cta_tile::WARPS_K > 1 )>
struct Batch_norm_dgrad_callbacks_epilogue
    : public Batch_norm_dgrad_callbacks_epilogue_base<Cta_tile, Activation, DUAL_DBNS_> {};

////////////////////////////////////////////////////////////////////////////////////////////////////
}  // namespace dgrad
}  // namespace bn_stats
}  // namespace batchnorm
}  // namespace ext
}  // namespace xmma
////////////////////////////////////////////////////////////////////////////////////////////////////
