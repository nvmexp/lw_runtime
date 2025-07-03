/***************************************************************************************************
 *
 * Redistribution and use in source and binary forms, with or without modification, are not permit-
 * ted.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
#pragma once

#include <xmma/ext/batchnorm/bn_stats/dgrad/callbacks_epilogue.h>
#include <xmma/ext/batchnorm/bn_stats/params.h>
#include <xmma/ext/batchnorm/bn_stats/traits.h>
#include <xmma/ext/batchnorm/bn_stats/dgrad/activation.h>
#include <xmma/cta_swizzle.h>
#include <xmma/ampere/fragment.h>
#include <xmma/ext/batchnorm/fragment.h>

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace xmma {
namespace ext {
namespace batchnorm {
namespace bn_stats {
namespace dgrad {

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Cta_tile, typename Activation_>
struct Batch_norm_dgrad_callbacks_epilogue<xmma::Ampere_hmma_fp32_traits,
                                           Cta_tile,
                                           Activation_,
                                           xmma::ext::batchnorm::ReluBitmaskFormat::NONE,
                                           false,
                                           false>
    : public Batch_norm_dgrad_callbacks_epilogue_base<Cta_tile, Activation_, false> {

    using Base = Batch_norm_dgrad_callbacks_epilogue_base<Cta_tile, Activation_, false>;

    // Ctor.
    template <typename Params>
    inline __device__ Batch_norm_dgrad_callbacks_epilogue( const Params &params,
                                                           char *smem,
                                                           int bidm,
                                                           int bidn,
                                                           int bidz,
                                                           int tidx,
                                                           int c )
        : Base( params, smem, bidm, bidn, bidz, tidx, c ) {
    }

    // Ctor.
    template <typename Params>
    inline __device__ Batch_norm_dgrad_callbacks_epilogue( const Params &params,
                                                           char *smem,
                                                           int bidm,
                                                           int bidn,
                                                           int bidz,
                                                           int tidx )
        : Batch_norm_dgrad_callbacks_epilogue( params,
                                               smem,
                                               bidm,
                                               bidn,
                                               bidz,
                                               tidx,
                                               params.g * params.c ) {
    }
};

template <typename Cta_tile, typename Activation_>
struct Batch_norm_dgrad_callbacks_epilogue<xmma::Ampere_hmma_fp32_traits,
                                           Cta_tile,
                                           Activation_,
                                           xmma::ext::batchnorm::ReluBitmaskFormat::FULL,
                                           false,  // DUAL_DBNS_
                                           false>
    : public Batch_norm_dgrad_callbacks_epilogue_base<Cta_tile, Activation_, false> {
    using Base = Batch_norm_dgrad_callbacks_epilogue_base<Cta_tile, Activation_, false>;
    using Fragment_post_swizzle =
        xmma::Fragment_epilogue_post_swizzle_bn_stats<xmma::Ampere_hmma_fp32_traits, Cta_tile>;
    using Fragment_c = xmma::Fragment_hmma_fp32_c_bn_stats<xmma::Ampere_hmma_fp32_traits, Cta_tile>;

    // Ctor.
    template <typename Params>
    inline __device__ Batch_norm_dgrad_callbacks_epilogue( const Params &params,
                                                           char *smem,
                                                           int bidm,
                                                           int bidn,
                                                           int bidz,
                                                           int tidx,
                                                           int c )
        : Base( params, smem, bidm, bidn, bidz, tidx, c ) {
    }

    // Ctor.
    template <typename Params>
    inline __device__ Batch_norm_dgrad_callbacks_epilogue( const Params &params,
                                                           char *smem,
                                                           int bidm,
                                                           int bidn,
                                                           int bidz,
                                                           int tidx )
        : Batch_norm_dgrad_callbacks_epilogue( params,
                                               smem,
                                               bidm,
                                               bidn,
                                               bidz,
                                               tidx,
                                               params.g * params.c ) {
    }

    // Post-swizzle accumulation.
    template <typename Epilogue>
    inline __device__ void post_swizzle( Epilogue &epilogue,
                                         int mi,
                                         int ii,
                                         const Fragment_post_swizzle &frag,
                                         const Fragment<float, 8> &mean,
                                         const Fragment<float, 8> &invstd,
                                         const Fragment_c &xfrag,
                                         const Fragment_c &bfrag,
                                         int mask ) {

        if( mask ) {

            float2 tmp[frag.NUM_REGS];
            float2 xtmp[xfrag.NUM_REGS];

            Activation_::execute( tmp, xtmp, frag, xfrag, mean, invstd, bfrag );

            Base::sums_[0].x += tmp[0].x;
            Base::sums_[0].y += tmp[0].y;
            Base::sums_[0].z += tmp[1].x;
            Base::sums_[0].w += tmp[1].y;

            Base::sums_[1].x += tmp[2].x;
            Base::sums_[1].y += tmp[2].y;
            Base::sums_[1].z += tmp[3].x;
            Base::sums_[1].w += tmp[3].y;

            Base::sums_of_squares_[0].x += tmp[0].x * ( xtmp[0].x - mean.elt( 0 ) );
            Base::sums_of_squares_[0].y += tmp[0].y * ( xtmp[0].y - mean.elt( 1 ) );
            Base::sums_of_squares_[0].z += tmp[1].x * ( xtmp[1].x - mean.elt( 2 ) );
            Base::sums_of_squares_[0].w += tmp[1].y * ( xtmp[1].y - mean.elt( 3 ) );
            Base::sums_of_squares_[1].x += tmp[2].x * ( xtmp[2].x - mean.elt( 4 ) );
            Base::sums_of_squares_[1].y += tmp[2].y * ( xtmp[2].y - mean.elt( 5 ) );
            Base::sums_of_squares_[1].z += tmp[3].x * ( xtmp[3].x - mean.elt( 6 ) );
            Base::sums_of_squares_[1].w += tmp[3].y * ( xtmp[3].y - mean.elt( 7 ) );
        }
    }
};

template <typename Cta_tile, typename Activation_>
struct Batch_norm_dgrad_callbacks_epilogue<xmma::Ampere_hmma_fp32_traits,
                                           Cta_tile,
                                           Activation_,
                                           xmma::ext::batchnorm::ReluBitmaskFormat::FULL,
                                           true,  // DUAL_DBNS_
                                           false>
    : public Batch_norm_dgrad_callbacks_epilogue_base<Cta_tile, Activation_, false> {

    using Base = Batch_norm_dgrad_callbacks_epilogue_base<Cta_tile, Activation_, false>;
    using Fragment_post_swizzle =
        xmma::Fragment_epilogue_post_swizzle_bn_stats<xmma::Ampere_hmma_fp32_traits, Cta_tile>;
    using Fragment_c = xmma::Fragment_hmma_fp32_c_bn_stats<xmma::Ampere_hmma_fp32_traits, Cta_tile>;

    // Ctor.
    template <typename Params>
    inline __device__ Batch_norm_dgrad_callbacks_epilogue( const Params &params,
                                                           char *smem,
                                                           int bidm,
                                                           int bidn,
                                                           int bidz,
                                                           int tidx,
                                                           int c )
        : Base( params, smem, bidm, bidn, bidz, tidx, c ) {
    }

    // Ctor.
    template <typename Params>
    inline __device__ Batch_norm_dgrad_callbacks_epilogue( const Params &params,
                                                           char *smem,
                                                           int bidm,
                                                           int bidn,
                                                           int bidz,
                                                           int tidx )
        : Batch_norm_dgrad_callbacks_epilogue( params,
                                               smem,
                                               bidm,
                                               bidn,
                                               bidz,
                                               tidx,
                                               params.g * params.c ) {

        dual_sums_of_squares_[0] = make_float4( 0.f, 0.f, 0.f, 0.f );
        dual_sums_of_squares_[1] = make_float4( 0.f, 0.f, 0.f, 0.f );
    }

    // Post-swizzle accumulation.
    template <typename Epilogue>
    inline __device__ void post_swizzle( Epilogue &epilogue,
                                         int mi,
                                         int ii,
                                         const Fragment_post_swizzle &frag,
                                         const Fragment<float, 8> &mean,
                                         const Fragment<float, 8> &invstd,
                                         const Fragment_c &xfrag,
                                         const Fragment<float, 8> &dual_mean,
                                         const Fragment<float, 8> &dual_invstd,
                                         const Fragment_c &dual_xfrag,
                                         const Fragment_c &bfrag,
                                         int mask ) {

        if( mask ) {

            float2 tmp[frag.NUM_REGS];
            float2 xtmp[xfrag.NUM_REGS];

            Activation_::execute( tmp, xtmp, frag, xfrag, mean, invstd, bfrag );

            Base::sums_[0].x += tmp[0].x;
            Base::sums_[0].y += tmp[0].y;
            Base::sums_[0].z += tmp[1].x;
            Base::sums_[0].w += tmp[1].y;

            Base::sums_[1].x += tmp[2].x;
            Base::sums_[1].y += tmp[2].y;
            Base::sums_[1].z += tmp[3].x;
            Base::sums_[1].w += tmp[3].y;

            Base::sums_of_squares_[0].x += tmp[0].x * ( xtmp[0].x - mean.elt( 0 ) );
            Base::sums_of_squares_[0].y += tmp[0].y * ( xtmp[0].y - mean.elt( 1 ) );
            Base::sums_of_squares_[0].z += tmp[1].x * ( xtmp[1].x - mean.elt( 2 ) );
            Base::sums_of_squares_[0].w += tmp[1].y * ( xtmp[1].y - mean.elt( 3 ) );
            Base::sums_of_squares_[1].x += tmp[2].x * ( xtmp[2].x - mean.elt( 4 ) );
            Base::sums_of_squares_[1].y += tmp[2].y * ( xtmp[2].y - mean.elt( 5 ) );
            Base::sums_of_squares_[1].z += tmp[3].x * ( xtmp[3].x - mean.elt( 6 ) );
            Base::sums_of_squares_[1].w += tmp[3].y * ( xtmp[3].y - mean.elt( 7 ) );

            Activation_::execute( tmp, xtmp, frag, dual_xfrag, dual_mean, dual_invstd, bfrag );
            dual_sums_of_squares_[0].x += tmp[0].x * ( xtmp[0].x - dual_mean.elt( 0 ) );
            dual_sums_of_squares_[0].y += tmp[0].y * ( xtmp[0].y - dual_mean.elt( 1 ) );
            dual_sums_of_squares_[0].z += tmp[1].x * ( xtmp[1].x - dual_mean.elt( 2 ) );
            dual_sums_of_squares_[0].w += tmp[1].y * ( xtmp[1].y - dual_mean.elt( 3 ) );
            dual_sums_of_squares_[1].x += tmp[2].x * ( xtmp[2].x - dual_mean.elt( 4 ) );
            dual_sums_of_squares_[1].y += tmp[2].y * ( xtmp[2].y - dual_mean.elt( 5 ) );
            dual_sums_of_squares_[1].z += tmp[3].x * ( xtmp[3].x - dual_mean.elt( 6 ) );
            dual_sums_of_squares_[1].w += tmp[3].y * ( xtmp[3].y - dual_mean.elt( 7 ) );
        }
    }

    // Store to global memory.
    inline __device__ void post_epilogue() {
        Base::post_epilogue();  // writes sums_ and sums_of_squares_ to params_bn_partial_sums_ptr_
        // The thread index.
        const int tidx = threadIdx.x;

        // The shared memory offset.
        int smem_write_offset = tidx * sizeof( float4 );

        // Make sure shared memory can be written.
        __syncthreads();

        // The pointer to shared memory.
        char *smem = &this->smem_[smem_write_offset];

        float4 *tmp0 = reinterpret_cast<float4 *>( &smem[0 * Base::BYTES_PER_ROW_WITH_SKEW] );
        float4 *tmp1 = reinterpret_cast<float4 *>( &smem[1 * Base::BYTES_PER_ROW_WITH_SKEW] );
        float4 *tmp2 = reinterpret_cast<float4 *>( &smem[2 * Base::BYTES_PER_ROW_WITH_SKEW] );
        float4 *tmp3 = reinterpret_cast<float4 *>( &smem[3 * Base::BYTES_PER_ROW_WITH_SKEW] );

        tmp0[0] = Base::sums_[0];
        tmp1[0] = Base::sums_[1];
        tmp2[0] = dual_sums_of_squares_[0];
        tmp3[0] = dual_sums_of_squares_[1];

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
            smem_read_row * Base::BYTES_PER_ROW_WITH_SKEW + smem_read_col * sizeof( Fragment );

        // Half threads work on the sum of squares.
        smem_read_offset += 2 * sums_or_sums_of_sq * Base::BYTES_PER_ROW_WITH_SKEW;

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
        const int INDEX_DUAL = 2;
        const int buffer = ( INDEX_DUAL + sums_or_sums_of_sq ) * Base::partial_sums_ + Base::bidm_;
        // The position in the buffer.
        const int idx_in_buffer = Base::bidn_ * Cta_tile::N + k * SUMS_PER_THREAD;
        // The offset (in floats) written by this thread.
        const int offset = buffer * Base::params_c_ + idx_in_buffer;

        // Is it in-bound?
        const int is_in_bound =
            k * SUMS_PER_THREAD < Cta_tile::N && idx_in_buffer < Base::params_c_;
        // Store the partial results.
        char *ptr = reinterpret_cast<char *>( Base::params_bn_partial_sums_ptr_ );
        if( sums_or_sums_of_sq < 2 && is_in_bound ) {
            reinterpret_cast<Fragment *>( &ptr[offset * sizeof( float )] )[0] = res;
        }
    }
    float4 dual_sums_of_squares_[2];
};

template <typename Cta_tile, typename Activation_>
struct Batch_norm_dgrad_callbacks_epilogue<xmma::Ampere_hmma_fp32_traits,
                                           Cta_tile,
                                           Activation_,
                                           xmma::ext::batchnorm::ReluBitmaskFormat::NONE,
                                           true,  // DUAL_DBNS_
                                           false>
    : public Batch_norm_dgrad_callbacks_epilogue_base<Cta_tile, Activation_, false> {

    using Base = Batch_norm_dgrad_callbacks_epilogue_base<Cta_tile, Activation_, false>;
    using Fragment_post_swizzle =
        xmma::Fragment_epilogue_post_swizzle_bn_stats<xmma::Ampere_hmma_fp32_traits, Cta_tile>;
    using Fragment_c = xmma::Fragment_hmma_fp32_c_bn_stats<xmma::Ampere_hmma_fp32_traits, Cta_tile>;

    // Ctor.
    template <typename Params>
    inline __device__ Batch_norm_dgrad_callbacks_epilogue( const Params &params,
                                                           char *smem,
                                                           int bidm,
                                                           int bidn,
                                                           int bidz,
                                                           int tidx,
                                                           int c )
        : Base( params, smem, bidm, bidn, bidz, tidx, c ) {
    }

    // Ctor.
    template <typename Params>
    inline __device__ Batch_norm_dgrad_callbacks_epilogue( const Params &params,
                                                           char *smem,
                                                           int bidm,
                                                           int bidn,
                                                           int bidz,
                                                           int tidx )
        : Batch_norm_dgrad_callbacks_epilogue( params,
                                               smem,
                                               bidm,
                                               bidn,
                                               bidz,
                                               tidx,
                                               params.g * params.c ) {
        dual_sums_of_squares_[0] = make_float4( 0.f, 0.f, 0.f, 0.f );
        dual_sums_of_squares_[1] = make_float4( 0.f, 0.f, 0.f, 0.f );
    }

    // Post-swizzle accumulation.
    template <typename Epilogue>
    inline __device__ void post_swizzle( Epilogue &epilogue,
                                         int mi,
                                         int ii,
                                         const Fragment_post_swizzle &frag,
                                         const Fragment<float, 8> &mean,
                                         const Fragment<float, 8> &invstd,
                                         const Fragment_c &xfrag,
                                         const Fragment<float, 8> &dual_mean,
                                         const Fragment<float, 8> &dual_invstd,
                                         const Fragment_c &dual_xfrag,
                                         int mask ) {

        if( mask ) {

            float2 tmp[frag.NUM_REGS];
            float2 xtmp[xfrag.NUM_REGS];

            Activation_::execute( tmp, xtmp, frag, xfrag, mean, invstd );

            Base::sums_[0].x += tmp[0].x;
            Base::sums_[0].y += tmp[0].y;
            Base::sums_[0].z += tmp[1].x;
            Base::sums_[0].w += tmp[1].y;

            Base::sums_[1].x += tmp[2].x;
            Base::sums_[1].y += tmp[2].y;
            Base::sums_[1].z += tmp[3].x;
            Base::sums_[1].w += tmp[3].y;

            Base::sums_of_squares_[0].x += tmp[0].x * ( xtmp[0].x - mean.elt( 0 ) );
            Base::sums_of_squares_[0].y += tmp[0].y * ( xtmp[0].y - mean.elt( 1 ) );
            Base::sums_of_squares_[0].z += tmp[1].x * ( xtmp[1].x - mean.elt( 2 ) );
            Base::sums_of_squares_[0].w += tmp[1].y * ( xtmp[1].y - mean.elt( 3 ) );
            Base::sums_of_squares_[1].x += tmp[2].x * ( xtmp[2].x - mean.elt( 4 ) );
            Base::sums_of_squares_[1].y += tmp[2].y * ( xtmp[2].y - mean.elt( 5 ) );
            Base::sums_of_squares_[1].z += tmp[3].x * ( xtmp[3].x - mean.elt( 6 ) );
            Base::sums_of_squares_[1].w += tmp[3].y * ( xtmp[3].y - mean.elt( 7 ) );

            Activation_::execute( tmp, xtmp, frag, dual_xfrag, dual_mean, dual_invstd );

            dual_sums_of_squares_[0].x += tmp[0].x * ( xtmp[0].x - dual_mean.elt( 0 ) );
            dual_sums_of_squares_[0].y += tmp[0].y * ( xtmp[0].y - dual_mean.elt( 1 ) );
            dual_sums_of_squares_[0].z += tmp[1].x * ( xtmp[1].x - dual_mean.elt( 2 ) );
            dual_sums_of_squares_[0].w += tmp[1].y * ( xtmp[1].y - dual_mean.elt( 3 ) );
            dual_sums_of_squares_[1].x += tmp[2].x * ( xtmp[2].x - dual_mean.elt( 4 ) );
            dual_sums_of_squares_[1].y += tmp[2].y * ( xtmp[2].y - dual_mean.elt( 5 ) );
            dual_sums_of_squares_[1].z += tmp[3].x * ( xtmp[3].x - dual_mean.elt( 6 ) );
            dual_sums_of_squares_[1].w += tmp[3].y * ( xtmp[3].y - dual_mean.elt( 7 ) );
        }
    }

    // Store to global memory.
    inline __device__ void post_epilogue() {
        Base::post_epilogue();  // writes sums_ and sums_of_squares_ to params_bn_partial_sums_ptr_
        // The thread index.
        const int tidx = threadIdx.x;

        // The shared memory offset.
        int smem_write_offset = tidx * sizeof( float4 );

        // Make sure shared memory can be written.
        __syncthreads();

        // The pointer to shared memory.
        char *smem = &this->smem_[smem_write_offset];

        float4 *tmp0 = reinterpret_cast<float4 *>( &smem[0 * Base::BYTES_PER_ROW_WITH_SKEW] );
        float4 *tmp1 = reinterpret_cast<float4 *>( &smem[1 * Base::BYTES_PER_ROW_WITH_SKEW] );
        float4 *tmp2 = reinterpret_cast<float4 *>( &smem[2 * Base::BYTES_PER_ROW_WITH_SKEW] );
        float4 *tmp3 = reinterpret_cast<float4 *>( &smem[3 * Base::BYTES_PER_ROW_WITH_SKEW] );

        tmp0[0] = Base::sums_[0];
        tmp1[0] = Base::sums_[1];
        tmp2[0] = dual_sums_of_squares_[0];
        tmp3[0] = dual_sums_of_squares_[1];

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
            smem_read_row * Base::BYTES_PER_ROW_WITH_SKEW + smem_read_col * sizeof( Fragment );

        // Half threads work on the sum of squares.
        smem_read_offset += 2 * sums_or_sums_of_sq * Base::BYTES_PER_ROW_WITH_SKEW;

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
        const int INDEX_DUAL = 2;
        const int buffer = ( INDEX_DUAL + sums_or_sums_of_sq ) * Base::partial_sums_ + Base::bidm_;
        // The position in the buffer.
        const int idx_in_buffer = Base::bidn_ * Cta_tile::N + k * SUMS_PER_THREAD;
        // The offset (in floats) written by this thread.
        const int offset = buffer * Base::params_c_ + idx_in_buffer;

        // Is it in-bound?
        const int is_in_bound =
            k * SUMS_PER_THREAD < Cta_tile::N && idx_in_buffer < Base::params_c_;
        // Store the partial results.
        char *ptr = reinterpret_cast<char *>( Base::params_bn_partial_sums_ptr_ );
        if( sums_or_sums_of_sq < 2 && is_in_bound ) {
            reinterpret_cast<Fragment *>( &ptr[offset * sizeof( float )] )[0] = res;
        }
    }

    // The sums of squares.
    float4 dual_sums_of_squares_[2];
};

////////////////////////////////////////////////////////////////////////////////////////////////////
}  // namespace dgrad
}  // namespace bn_stats
}  // namespace batchnorm
}  // namespace ext
}  // namespace xmma
////////////////////////////////////////////////////////////////////////////////////////////////////
