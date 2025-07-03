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
#include <xmma/named_barrier.h>
#include <xmma/smem_tile.h>
#include <xmma/warp_masks.h>
#include <xmma/arrive_wait.h>
#include <xmma/helpers/epilogue.h>
#include <xmma/ext/batchnorm/relu_bitmask_format.h>

namespace xmma {
namespace helpers {

template <
    // The instruction traits.
    typename Traits_,
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile_,
    // The layout of the tile.
    typename Layout_,
    // The global memory tile to store the output.
    typename Gmem_tile_,
    // ReLuBitmaskFormat
    xmma::ext::batchnorm::ReluBitmaskFormat relu_bitmask_format,
    // The global memory tile to read fprop tensors
    typename Callbacks,
    // The class to swizzle the data.
    typename Swizzle_,
    // dual dbn(s)
    bool DUAL_DBNS_>
struct Epilogue_dbns {};

template <
    // The instruction traits.
    typename Traits_,
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile_,
    // The layout of the tile.
    typename Layout_,
    // The global memory tile to store the output.
    typename Gmem_tile_,
    // The global memory tile to read fprop tensors
    typename Callbacks,
    // The class to swizzle the data.
    typename Swizzle_>
struct Epilogue_dbns<Traits_,
                     Cta_tile_,
                     Layout_,
                     Gmem_tile_,
                     xmma::ext::batchnorm::ReluBitmaskFormat::NONE,
                     Callbacks,
                     Swizzle_,
                     false> {
    // dbias  = sum(dy)
    // dscale = sum(dy * (x-u))

    // Load u [0,N] into smem = N FP32 elements
    // Load x [M, N] into smem = M * N FP32 Elements

    // After MMA need to compute dy * (x-u)
    // For Ampere_hmma_fp32_traits each thread has 8 FP32 elements for 4 channels
    // Read the correct smem index and do a fp32 multiply and store to sum_of_products

    // TODO : 1) Clear sums_ and sums_of_sq in the ctor

    // The instruction traits.
    using Traits = Traits_;
    // The dimensions of the tile computed by the CTA.
    using Cta_tile = Cta_tile_;
    // The layout of the tile.
    using Layout = Layout_;
    // The global memory tile to store the output.
    using Gmem_tile = Gmem_tile_;
    // The class to swizzle the data.
    using Swizzle = Swizzle_;

    // The fragment class before the swizzling.
    using Fragment_pre_swizzle = typename Callbacks::Fragment_pre_swizzle;
    // The fragment class after the swizzling.
    using Fragment_post_swizzle = typename Callbacks::Fragment_post_swizzle;
    // The output fragment.
    using Fragment_c = typename Callbacks::Fragment_c;

    // The fragment for alpha (used before swizzling).
    using Fragment_alpha_pre_swizzle = typename Callbacks::Fragment_alpha_pre_swizzle;
    // The fragment for alpha (used after swizzling).
    using Fragment_alpha_post_swizzle = typename Callbacks::Fragment_alpha_post_swizzle;
    // The fragment for beta.
    using Fragment_beta = typename Callbacks::Fragment_beta;

    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;

    // The number of XMMAs.
    enum { XMMAS_M = Xmma_tile::XMMAS_M, XMMAS_N = Xmma_tile::XMMAS_N };

    // Ctor.
    template <typename Params>
    inline __device__ Epilogue_dbns( const Params &params,
                                     Gmem_tile &gmem_tile,
                                     Swizzle &swizzle,
                                     Callbacks &callbacks,
                                     const Named_barrier &epi_sync = Named_barrier(),
                                     const int bidm = blockIdx.x,
                                     const int bidn = blockIdx.y,
                                     const int bidz = blockIdx.z,
                                     const int tidx = threadIdx.x,
                                     const bool is_warp_specialized = false )
        : gmem_tile_( gmem_tile ), swizzle_( swizzle ), callbacks_( callbacks ),
          epi_sync_( epi_sync ), mem_desc_c_( params.mem_descriptors.descriptor_c ),
          mem_desc_d_( params.mem_descriptors.descriptor_d ) {
    }

    // Do the epilogue.
    template <bool WITH_RESIDUAL, typename Fragment_aclwmulator, int M, int N>
    inline __device__ void execute( Fragment_aclwmulator ( &acc )[M][N] ) {
#pragma unroll
        for( int mi = 0; mi < M; ++mi ) {
            this->step<WITH_RESIDUAL>( mi, acc[mi] );
        }
    }

    // Do only split-k for a 2-kernel split-k.
    template <int N> inline __device__ void exelwte_split_k() {
    }

    // Execute a single iteration of the loop.
    template <bool WITH_RESIDUAL, typename Fragment_aclwmulator, int N>
    inline __device__ void step( int mi, Fragment_aclwmulator ( &acc )[N] ) {

        // The output masks.
        int out_masks[Gmem_tile::STGS];
#pragma unroll
        for( int ii = 0; ii < Gmem_tile::STGS; ++ii ) {
            out_masks[ii] = this->gmem_tile_.compute_output_mask( mi, ii );
        }

        if( mi == 0 ) {
            this->gmem_tile_.load_bn_fprop_mean(
                bn_fprop_mean[0], 0, 0, out_masks[0], mem_desc_c_ );
            this->gmem_tile_.load_bn_fprop_ilw_stddev(
                bn_fprop_ilw_stddev[0], 0, 0, out_masks[0], mem_desc_c_ );
        }

        // Load valid values if beta is not zero.
        Fragment_c res_fetch[Gmem_tile::STGS];
        if( WITH_RESIDUAL ) {
#pragma unroll
            for( int ii = 0; ii < Gmem_tile::STGS; ++ii ) {
                this->gmem_tile_.load( res_fetch[ii], mi, ii, out_masks[ii], mem_desc_c_ );
            }
        }

        // Load fprop values
        Fragment_c x_fetch[Gmem_tile::STGS];
#pragma unroll
        for( int ii = 0; ii < Gmem_tile::STGS; ++ii ) {
            this->gmem_tile_.load_bn_fprop_tensor(
                x_fetch[ii], mi, ii, out_masks[ii], mem_desc_c_ );
        }

// Do something before we colwert
#pragma unroll
        for( int ni = 0; ni < N; ++ni ) {
            callbacks_.pre_colwert( *this, mi, ni, acc[ni] );
        }

        // Colwert the aclwmulators to the epilogue format (or keep them as-is).
        Fragment_pre_swizzle pre_swizzle[N];
#pragma unroll
        for( int ni = 0; ni < N; ++ni ) {
            pre_swizzle[ni].shuffle_groups( acc[ni] );
        }

        // Load alpha.
        Fragment_alpha_pre_swizzle alpha_pre_swizzle[N];
#pragma unroll
        for( int ni = 0; ni < N; ++ni ) {
            callbacks_.alpha_pre_swizzle( *this, mi, ni, alpha_pre_swizzle[ni] );
        }

// Do the colwersion.
#pragma unroll

        for( int ni = 0; ni < N; ++ni ) {
            pre_swizzle[ni].colwert( alpha_pre_swizzle[ni], acc[ni] );
        }

// Do something before we swizzle.
#pragma unroll
        for( int ni = 0; ni < N; ++ni ) {
            callbacks_.pre_swizzle( *this, mi, ni, pre_swizzle[ni] );
        }

        // Make sure the main loop or the previous loop of the epilogue are finished.
        if( !Swizzle::SKIP_SYNCTHREADS ) {
            if( epi_sync_.invalid() ) {
                __syncthreads();
            } else {
                epi_sync_.wait();
            }
        }

// Store the data in shared memory to produce more friendly stores.
#pragma unroll
        for( int ni = 0; ni < N; ++ni ) {
            this->swizzle_.store( ni, pre_swizzle[ni] );
        }

        // Make sure the data is in SMEM.
        if( !Swizzle::SKIP_SYNCTHREADS ) {
            if( epi_sync_.invalid() ) {
                __syncthreads();
            } else {
                epi_sync_.wait();
            }
        }

        // The fragments after the swizzle. One fragment per STG.128.
        Fragment_post_swizzle post_swizzle[Gmem_tile::STGS];
#pragma unroll
        for( int ii = 0; ii < Gmem_tile::STGS; ++ii ) {
            this->swizzle_.load( post_swizzle[ii], ii );
        }

        // Load alpha post swizzle.
        Fragment_alpha_post_swizzle alpha_post_swizzle[Gmem_tile::STGS];
#pragma unroll
        for( int ii = 0; ii < Gmem_tile::STGS; ++ii ) {
            callbacks_.alpha_post_swizzle( *this, mi, ii, alpha_post_swizzle[ii] );
        }

// Do the parallel reduction, if needed.
#pragma unroll
        for( int ii = 0; ii < Gmem_tile::STGS; ++ii ) {
            post_swizzle[ii].reduce( alpha_post_swizzle[ii] );
        }

        // Add the residual value before packing. TODO: We should be able to pass a single beta.
        if( WITH_RESIDUAL ) {
#pragma unroll
            for( int ii = 0; ii < Gmem_tile::STGS; ++ii ) {
                post_swizzle[ii].add( res_fetch[ii] );
            }
        }

// Do something now that the data has been swizzled.
#pragma unroll
        for( int ii = 0; ii < Gmem_tile::STGS; ++ii ) {
            callbacks_.post_swizzle( *this,
                                     mi,
                                     ii,
                                     post_swizzle[ii],
                                     bn_fprop_mean[0],
                                     bn_fprop_ilw_stddev[0],
                                     x_fetch[ii],
                                     out_masks[ii] );
        }

        // Do something before packing and pack to produce a STG.128.
        // Put in one loop for F2IP.RELU optimization.
        Fragment_c out_regs[Gmem_tile::STGS];
#pragma unroll
        for( int ii = 0; ii < Gmem_tile::STGS; ++ii ) {
            callbacks_.pre_pack( *this, mi, ii, post_swizzle[ii] );
            out_regs[ii].pack( alpha_post_swizzle[ii], post_swizzle[ii] );
        }

// Do something before we store.
#pragma unroll
        for( int ii = 0; ii < Gmem_tile::STGS; ++ii ) {
            callbacks_.pre_store( *this, mi, ii, out_regs[ii], out_masks[ii] );
        }

// Write valid values.
#pragma unroll
        for( int ii = 0; ii < Gmem_tile::STGS; ++ii ) {
            this->gmem_tile_.store( mi, ii, out_regs[ii], out_masks[ii], mem_desc_d_ );
        }
    }

    // The output tile.
    Gmem_tile &gmem_tile_;
    // The shared memory tile.
    Swizzle &swizzle_;

    // The callbacks.
    Callbacks &callbacks_;

    char *smem;

    // The named barrier object used for epilog sync.
    const Named_barrier &epi_sync_;

    // Ampere memory descriptors
    const uint64_t mem_desc_c_, mem_desc_d_;

    // BN fprop mean and var
    Fragment<float, 8> bn_fprop_mean[1];
    Fragment<float, 8> bn_fprop_ilw_stddev[1];
};

template <
    // The instruction traits.
    typename Traits_,
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile_,
    // The layout of the tile.
    typename Layout_,
    // The global memory tile to store the output.
    typename Gmem_tile_,
    // The global memory tile to read fprop tensors
    typename Callbacks,
    // The class to swizzle the data.
    typename Swizzle_>
struct Epilogue_dbns<Traits_,
                     Cta_tile_,
                     Layout_,
                     Gmem_tile_,
                     xmma::ext::batchnorm::ReluBitmaskFormat::FULL,
                     Callbacks,
                     Swizzle_,
                     false> {

    // The instruction traits.
    using Traits = Traits_;
    // The dimensions of the tile computed by the CTA.
    using Cta_tile = Cta_tile_;
    // The layout of the tile.
    using Layout = Layout_;
    // The global memory tile to store the output.
    using Gmem_tile = Gmem_tile_;
    // The class to swizzle the data.
    using Swizzle = Swizzle_;

    // The fragment class before the swizzling.
    using Fragment_pre_swizzle = typename Callbacks::Fragment_pre_swizzle;
    // The fragment class after the swizzling.
    using Fragment_post_swizzle = typename Callbacks::Fragment_post_swizzle;
    // The output fragment.
    using Fragment_c = typename Callbacks::Fragment_c;

    // The fragment for alpha (used before swizzling).
    using Fragment_alpha_pre_swizzle = typename Callbacks::Fragment_alpha_pre_swizzle;
    // The fragment for alpha (used after swizzling).
    using Fragment_alpha_post_swizzle = typename Callbacks::Fragment_alpha_post_swizzle;
    // The fragment for beta.
    using Fragment_beta = typename Callbacks::Fragment_beta;

    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;

    // The number of XMMAs.
    enum { XMMAS_M = Xmma_tile::XMMAS_M, XMMAS_N = Xmma_tile::XMMAS_N };

    // Ctor.
    template <typename Params>
    inline __device__ Epilogue_dbns( const Params &params,
                                     Gmem_tile &gmem_tile,
                                     Swizzle &swizzle,
                                     Callbacks &callbacks,
                                     const Named_barrier &epi_sync = Named_barrier(),
                                     const int bidm = blockIdx.x,
                                     const int bidn = blockIdx.y,
                                     const int bidz = blockIdx.z,
                                     const int tidx = threadIdx.x,
                                     const bool is_warp_specialized = false )
        : gmem_tile_( gmem_tile ), swizzle_( swizzle ), callbacks_( callbacks ),
          epi_sync_( epi_sync ), mem_desc_c_( params.mem_descriptors.descriptor_c ),
          mem_desc_d_( params.mem_descriptors.descriptor_d ) {
    }

    // Do the epilogue.
    template <bool WITH_RESIDUAL, typename Fragment_aclwmulator, int M, int N>
    inline __device__ void execute( Fragment_aclwmulator ( &acc )[M][N] ) {
#pragma unroll
        for( int mi = 0; mi < M; ++mi ) {
            this->step<WITH_RESIDUAL>( mi, acc[mi] );
        }
    }

    // Do only split-k for a 2-kernel split-k.
    template <int N> inline __device__ void exelwte_split_k() {
    }

    template <int N, typename dtype>
    inline __device__ void to_float( float ( &dst )[2 * N], dtype ( &src )[N] ) {
#pragma unroll
        for( int i = 0; i < N; ++i ) {
            uint16_t lo, hi;
            asm volatile( "mov.b32 {%0, %1}, %2;" : "=h"( lo ), "=h"( hi ) : "r"( src[i] ) );
            asm volatile( "cvt.f32.f16 %0, %1;" : "=f"( dst[2 * i + 0] ) : "h"( lo ) );
            asm volatile( "cvt.f32.f16 %0, %1;" : "=f"( dst[2 * i + 1] ) : "h"( hi ) );
        }
    }

    // Execute a single iteration of the loop.
    template <bool WITH_RESIDUAL, typename Fragment_aclwmulator, int N>
    inline __device__ void step( int mi, Fragment_aclwmulator ( &acc )[N] ) {

        // The output masks.
        int out_masks[Gmem_tile::STGS];
#pragma unroll
        for( int ii = 0; ii < Gmem_tile::STGS; ++ii ) {
            out_masks[ii] = this->gmem_tile_.compute_output_mask( mi, ii );
        }

        if( mi == 0 ) {
            this->gmem_tile_.load_bn_fprop_mean(
                bn_fprop_mean[0], 0, 0, out_masks[0], mem_desc_c_ );
            this->gmem_tile_.load_bn_fprop_ilw_stddev(
                bn_fprop_ilw_stddev[0], 0, 0, out_masks[0], mem_desc_c_ );
        }

        // Load valid values if beta is not zero.
        Fragment_c res_fetch[Gmem_tile::STGS];
        Fragment_c bitmask[Gmem_tile::STGS];
        if( WITH_RESIDUAL ) {
#pragma unroll
            for( int ii = 0; ii < Gmem_tile::STGS; ++ii ) {
                this->gmem_tile_.load( res_fetch[ii], mi, ii, out_masks[ii], mem_desc_c_ );
            }

#pragma unroll
            for( int ii = 0; ii < Gmem_tile::STGS; ++ii ) {
                this->gmem_tile_.load_bitmask_full(
                    bitmask[ii], mi, ii, out_masks[ii], mem_desc_c_ );
            }
        }

        // Load fprop values
        Fragment_c x_fetch[Gmem_tile::STGS];
#pragma unroll
        for( int ii = 0; ii < Gmem_tile::STGS; ++ii ) {
            this->gmem_tile_.load_bn_fprop_tensor(
                x_fetch[ii], mi, ii, out_masks[ii], mem_desc_c_ );
        }

// Do something before we colwert
#pragma unroll
        for( int ni = 0; ni < N; ++ni ) {
            callbacks_.pre_colwert( *this, mi, ni, acc[ni] );
        }

        // Colwert the aclwmulators to the epilogue format (or keep them as-is).
        Fragment_pre_swizzle pre_swizzle[N];
#pragma unroll
        for( int ni = 0; ni < N; ++ni ) {
            pre_swizzle[ni].shuffle_groups( acc[ni] );
        }

        // Load alpha.
        Fragment_alpha_pre_swizzle alpha_pre_swizzle[N];
#pragma unroll
        for( int ni = 0; ni < N; ++ni ) {
            callbacks_.alpha_pre_swizzle( *this, mi, ni, alpha_pre_swizzle[ni] );
        }

// Do the colwersion.
#pragma unroll

        for( int ni = 0; ni < N; ++ni ) {
            pre_swizzle[ni].colwert( alpha_pre_swizzle[ni], acc[ni] );
        }

// Do something before we swizzle.
#pragma unroll
        for( int ni = 0; ni < N; ++ni ) {
            callbacks_.pre_swizzle( *this, mi, ni, pre_swizzle[ni] );
        }

        // Make sure the main loop or the previous loop of the epilogue are finished.
        if( !Swizzle::SKIP_SYNCTHREADS ) {
            if( epi_sync_.invalid() ) {
                __syncthreads();
            } else {
                epi_sync_.wait();
            }
        }

// Store the data in shared memory to produce more friendly stores.
#pragma unroll
        for( int ni = 0; ni < N; ++ni ) {
            this->swizzle_.store( ni, pre_swizzle[ni] );
        }

        // Make sure the data is in SMEM.
        if( !Swizzle::SKIP_SYNCTHREADS ) {
            if( epi_sync_.invalid() ) {
                __syncthreads();
            } else {
                epi_sync_.wait();
            }
        }

        // The fragments after the swizzle. One fragment per STG.128.
        Fragment_post_swizzle post_swizzle[Gmem_tile::STGS];
#pragma unroll
        for( int ii = 0; ii < Gmem_tile::STGS; ++ii ) {
            this->swizzle_.load( post_swizzle[ii], ii );
        }

        // Load alpha post swizzle.
        Fragment_alpha_post_swizzle alpha_post_swizzle[Gmem_tile::STGS];
#pragma unroll
        for( int ii = 0; ii < Gmem_tile::STGS; ++ii ) {
            callbacks_.alpha_post_swizzle( *this, mi, ii, alpha_post_swizzle[ii] );
        }

// Do the parallel reduction, if needed.
#pragma unroll
        for( int ii = 0; ii < Gmem_tile::STGS; ++ii ) {
            post_swizzle[ii].reduce( alpha_post_swizzle[ii] );
        }

        // Add the residual value before packing. TODO: We should be able to pass a single beta.
        if( WITH_RESIDUAL ) {
#pragma unroll
            for( int ii = 0; ii < Gmem_tile::STGS; ++ii ) {
                post_swizzle[ii].add( res_fetch[ii] );
            }
        }

// Do something now that the data has been swizzled.
#pragma unroll
        for( int ii = 0; ii < Gmem_tile::STGS; ++ii ) {
            callbacks_.post_swizzle( *this,
                                     mi,
                                     ii,
                                     post_swizzle[ii],
                                     bn_fprop_mean[0],
                                     bn_fprop_ilw_stddev[0],
                                     x_fetch[ii],
                                     bitmask[ii],
                                     out_masks[ii] );
        }

        // Do something before packing and pack to produce a STG.128.
        // Put in one loop for F2IP.RELU optimization.
        Fragment_c out_regs[Gmem_tile::STGS];
#pragma unroll
        for( int ii = 0; ii < Gmem_tile::STGS; ++ii ) {
            callbacks_.pre_pack( *this, mi, ii, post_swizzle[ii] );
            out_regs[ii].pack( alpha_post_swizzle[ii], post_swizzle[ii] );
        }

// Do something before we store.
#pragma unroll
        for( int ii = 0; ii < Gmem_tile::STGS; ++ii ) {
            callbacks_.pre_store( *this, mi, ii, out_regs[ii], out_masks[ii] );
        }

// Write valid values.
#pragma unroll
        for( int ii = 0; ii < Gmem_tile::STGS; ++ii ) {
            this->gmem_tile_.store( mi, ii, out_regs[ii], out_masks[ii], mem_desc_d_ );
        }
    }

    // The output tile.
    Gmem_tile &gmem_tile_;
    // The shared memory tile.
    Swizzle &swizzle_;

    // The callbacks.
    Callbacks &callbacks_;

    char *smem;

    // The named barrier object used for epilog sync.
    const Named_barrier &epi_sync_;

    // Ampere memory descriptors
    const uint64_t mem_desc_c_, mem_desc_d_;

    // BN fprop mean and var
    Fragment<float, 8> bn_fprop_mean[1];
    Fragment<float, 8> bn_fprop_ilw_stddev[1];
};

template <
    // The instruction traits.
    typename Traits_,
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile_,
    // The layout of the tile.
    typename Layout_,
    // The global memory tile to store the output.
    typename Gmem_tile_,
    // The global memory tile to read fprop tensors
    typename Callbacks,
    // The class to swizzle the data.
    typename Swizzle_>
struct Epilogue_dbns<Traits_,
                     Cta_tile_,
                     Layout_,
                     Gmem_tile_,
                     xmma::ext::batchnorm::ReluBitmaskFormat::FULL,
                     Callbacks,
                     Swizzle_,
                     true> {

    // The instruction traits.
    using Traits = Traits_;
    // The dimensions of the tile computed by the CTA.
    using Cta_tile = Cta_tile_;
    // The layout of the tile.
    using Layout = Layout_;
    // The global memory tile to store the output.
    using Gmem_tile = Gmem_tile_;
    // The class to swizzle the data.
    using Swizzle = Swizzle_;

    // The fragment class before the swizzling.
    using Fragment_pre_swizzle = typename Callbacks::Fragment_pre_swizzle;
    // The fragment class after the swizzling.
    using Fragment_post_swizzle = typename Callbacks::Fragment_post_swizzle;
    // The output fragment.
    using Fragment_c = typename Callbacks::Fragment_c;

    // The fragment for alpha (used before swizzling).
    using Fragment_alpha_pre_swizzle = typename Callbacks::Fragment_alpha_pre_swizzle;
    // The fragment for alpha (used after swizzling).
    using Fragment_alpha_post_swizzle = typename Callbacks::Fragment_alpha_post_swizzle;
    // The fragment for beta.
    using Fragment_beta = typename Callbacks::Fragment_beta;

    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;

    // The number of XMMAs.
    enum { XMMAS_M = Xmma_tile::XMMAS_M, XMMAS_N = Xmma_tile::XMMAS_N };

    // Ctor.
    template <typename Params>
    inline __device__ Epilogue_dbns( const Params &params,
                                     Gmem_tile &gmem_tile,
                                     Swizzle &swizzle,
                                     Callbacks &callbacks,
                                     const Named_barrier &epi_sync = Named_barrier(),
                                     const int bidm = blockIdx.x,
                                     const int bidn = blockIdx.y,
                                     const int bidz = blockIdx.z,
                                     const int tidx = threadIdx.x,
                                     const bool is_warp_specialized = false )
        : gmem_tile_( gmem_tile ), swizzle_( swizzle ), callbacks_( callbacks ),
          epi_sync_( epi_sync ), mem_desc_c_( params.mem_descriptors.descriptor_c ),
          mem_desc_d_( params.mem_descriptors.descriptor_d ) {
    }

    // Do the epilogue.
    template <bool WITH_RESIDUAL, typename Fragment_aclwmulator, int M, int N>
    inline __device__ void execute( Fragment_aclwmulator ( &acc )[M][N] ) {
#pragma unroll
        for( int mi = 0; mi < M; ++mi ) {
            this->step<WITH_RESIDUAL>( mi, acc[mi] );
        }
    }

    // Do only split-k for a 2-kernel split-k.
    template <int N> inline __device__ void exelwte_split_k() {
    }

    template <int N, typename dtype>
    inline __device__ void to_float( float ( &dst )[2 * N], dtype ( &src )[N] ) {
#pragma unroll
        for( int i = 0; i < N; ++i ) {
            uint16_t lo, hi;
            asm volatile( "mov.b32 {%0, %1}, %2;" : "=h"( lo ), "=h"( hi ) : "r"( src[i] ) );
            asm volatile( "cvt.f32.f16 %0, %1;" : "=f"( dst[2 * i + 0] ) : "h"( lo ) );
            asm volatile( "cvt.f32.f16 %0, %1;" : "=f"( dst[2 * i + 1] ) : "h"( hi ) );
        }
    }

    // Execute a single iteration of the loop.
    template <bool WITH_RESIDUAL, typename Fragment_aclwmulator, int N>
    inline __device__ void step( int mi, Fragment_aclwmulator ( &acc )[N] ) {

        // The output masks.
        int out_masks[Gmem_tile::STGS];
#pragma unroll
        for( int ii = 0; ii < Gmem_tile::STGS; ++ii ) {
            out_masks[ii] = this->gmem_tile_.compute_output_mask( mi, ii );
        }

        if( mi == 0 ) {
            this->gmem_tile_.load_bn_fprop_mean(
                bn_fprop_mean[0], 0, 0, out_masks[0], mem_desc_c_ );
            this->gmem_tile_.load_bn_fprop_ilw_stddev(
                bn_fprop_ilw_stddev[0], 0, 0, out_masks[0], mem_desc_c_ );
            this->gmem_tile_.load_dual_bn_fprop_mean(
                dual_bn_fprop_mean[0], 0, 0, out_masks[0], mem_desc_c_ );
            this->gmem_tile_.load_dual_bn_fprop_ilw_stddev(
                dual_bn_fprop_ilw_stddev[0], 0, 0, out_masks[0], mem_desc_c_ );
        }

        // Load valid values if beta is not zero.
        Fragment_c res_fetch[Gmem_tile::STGS];
        Fragment_c bitmask[Gmem_tile::STGS];
        if( WITH_RESIDUAL ) {
#pragma unroll
            for( int ii = 0; ii < Gmem_tile::STGS; ++ii ) {
                this->gmem_tile_.load( res_fetch[ii], mi, ii, out_masks[ii], mem_desc_c_ );
            }

#pragma unroll
            for( int ii = 0; ii < Gmem_tile::STGS; ++ii ) {
                this->gmem_tile_.load_bitmask_full(
                    bitmask[ii], mi, ii, out_masks[ii], mem_desc_c_ );
            }
        }

        // Load fprop values
        Fragment_c x_fetch[Gmem_tile::STGS];
#pragma unroll
        for( int ii = 0; ii < Gmem_tile::STGS; ++ii ) {
            this->gmem_tile_.load_bn_fprop_tensor(
                x_fetch[ii], mi, ii, out_masks[ii], mem_desc_c_ );
        }

        Fragment_c dual_x_fetch[Gmem_tile::STGS];
#pragma unroll
        for( int ii = 0; ii < Gmem_tile::STGS; ++ii ) {
            this->gmem_tile_.load_dual_bn_fprop_tensor(
                dual_x_fetch[ii], mi, ii, out_masks[ii], mem_desc_c_ );
        }

// Do something before we colwert
#pragma unroll
        for( int ni = 0; ni < N; ++ni ) {
            callbacks_.pre_colwert( *this, mi, ni, acc[ni] );
        }

        // Colwert the aclwmulators to the epilogue format (or keep them as-is).
        Fragment_pre_swizzle pre_swizzle[N];
#pragma unroll
        for( int ni = 0; ni < N; ++ni ) {
            pre_swizzle[ni].shuffle_groups( acc[ni] );
        }

        // Load alpha.
        Fragment_alpha_pre_swizzle alpha_pre_swizzle[N];
#pragma unroll
        for( int ni = 0; ni < N; ++ni ) {
            callbacks_.alpha_pre_swizzle( *this, mi, ni, alpha_pre_swizzle[ni] );
        }

// Do the colwersion.
#pragma unroll

        for( int ni = 0; ni < N; ++ni ) {
            pre_swizzle[ni].colwert( alpha_pre_swizzle[ni], acc[ni] );
        }

// Do something before we swizzle.
#pragma unroll
        for( int ni = 0; ni < N; ++ni ) {
            callbacks_.pre_swizzle( *this, mi, ni, pre_swizzle[ni] );
        }

        // Make sure the main loop or the previous loop of the epilogue are finished.
        if( !Swizzle::SKIP_SYNCTHREADS ) {
            if( epi_sync_.invalid() ) {
                __syncthreads();
            } else {
                epi_sync_.wait();
            }
        }

// Store the data in shared memory to produce more friendly stores.
#pragma unroll
        for( int ni = 0; ni < N; ++ni ) {
            this->swizzle_.store( ni, pre_swizzle[ni] );
        }

        // Make sure the data is in SMEM.
        if( !Swizzle::SKIP_SYNCTHREADS ) {
            if( epi_sync_.invalid() ) {
                __syncthreads();
            } else {
                epi_sync_.wait();
            }
        }

        // The fragments after the swizzle. One fragment per STG.128.
        Fragment_post_swizzle post_swizzle[Gmem_tile::STGS];
#pragma unroll
        for( int ii = 0; ii < Gmem_tile::STGS; ++ii ) {
            this->swizzle_.load( post_swizzle[ii], ii );
        }

        // Load alpha post swizzle.
        Fragment_alpha_post_swizzle alpha_post_swizzle[Gmem_tile::STGS];
#pragma unroll
        for( int ii = 0; ii < Gmem_tile::STGS; ++ii ) {
            callbacks_.alpha_post_swizzle( *this, mi, ii, alpha_post_swizzle[ii] );
        }

// Do the parallel reduction, if needed.
#pragma unroll
        for( int ii = 0; ii < Gmem_tile::STGS; ++ii ) {
            post_swizzle[ii].reduce( alpha_post_swizzle[ii] );
        }

        // Add the residual value before packing. TODO: We should be able to pass a single beta.
        if( WITH_RESIDUAL ) {
#pragma unroll
            for( int ii = 0; ii < Gmem_tile::STGS; ++ii ) {
                post_swizzle[ii].add( res_fetch[ii] );
            }
        }

// Do something now that the data has been swizzled.
#pragma unroll
        for( int ii = 0; ii < Gmem_tile::STGS; ++ii ) {
            callbacks_.post_swizzle( *this,
                                     mi,
                                     ii,
                                     post_swizzle[ii],
                                     bn_fprop_mean[0],
                                     bn_fprop_ilw_stddev[0],
                                     x_fetch[ii],
                                     dual_bn_fprop_mean[0],
                                     dual_bn_fprop_ilw_stddev[0],
                                     dual_x_fetch[ii],
                                     bitmask[ii],
                                     out_masks[ii] );
        }

        // Do something before packing and pack to produce a STG.128.
        // Put in one loop for F2IP.RELU optimization.
        Fragment_c out_regs[Gmem_tile::STGS];
#pragma unroll
        for( int ii = 0; ii < Gmem_tile::STGS; ++ii ) {
            callbacks_.pre_pack( *this, mi, ii, post_swizzle[ii] );
            out_regs[ii].pack( alpha_post_swizzle[ii], post_swizzle[ii] );
        }

// Do something before we store.
#pragma unroll
        for( int ii = 0; ii < Gmem_tile::STGS; ++ii ) {
            callbacks_.pre_store( *this, mi, ii, out_regs[ii], out_masks[ii] );
        }

// Write valid values.
#pragma unroll
        for( int ii = 0; ii < Gmem_tile::STGS; ++ii ) {
            this->gmem_tile_.store( mi, ii, out_regs[ii], out_masks[ii], mem_desc_d_ );
        }
    }

    // The output tile.
    Gmem_tile &gmem_tile_;
    // The shared memory tile.
    Swizzle &swizzle_;

    // The callbacks.
    Callbacks &callbacks_;

    char *smem;

    // The named barrier object used for epilog sync.
    const Named_barrier &epi_sync_;

    // Ampere memory descriptors
    const uint64_t mem_desc_c_, mem_desc_d_;

    // BN fprop mean and var
    Fragment<float, 8> bn_fprop_mean[1];
    Fragment<float, 8> bn_fprop_ilw_stddev[1];
    Fragment<float, 8> dual_bn_fprop_mean[1];
    Fragment<float, 8> dual_bn_fprop_ilw_stddev[1];
};

template <
    // The instruction traits.
    typename Traits_,
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile_,
    // The layout of the tile.
    typename Layout_,
    // The global memory tile to store the output.
    typename Gmem_tile_,
    // The global memory tile to read fprop tensors
    typename Callbacks,
    // The class to swizzle the data.
    typename Swizzle_>
struct Epilogue_dbns<Traits_,
                     Cta_tile_,
                     Layout_,
                     Gmem_tile_,
                     xmma::ext::batchnorm::ReluBitmaskFormat::NONE,
                     Callbacks,
                     Swizzle_,
                     true> {

    // The instruction traits.
    using Traits = Traits_;
    // The dimensions of the tile computed by the CTA.
    using Cta_tile = Cta_tile_;
    // The layout of the tile.
    using Layout = Layout_;
    // The global memory tile to store the output.
    using Gmem_tile = Gmem_tile_;
    // The class to swizzle the data.
    using Swizzle = Swizzle_;

    // The fragment class before the swizzling.
    using Fragment_pre_swizzle = typename Callbacks::Fragment_pre_swizzle;
    // The fragment class after the swizzling.
    using Fragment_post_swizzle = typename Callbacks::Fragment_post_swizzle;
    // The output fragment.
    using Fragment_c = typename Callbacks::Fragment_c;

    // The fragment for alpha (used before swizzling).
    using Fragment_alpha_pre_swizzle = typename Callbacks::Fragment_alpha_pre_swizzle;
    // The fragment for alpha (used after swizzling).
    using Fragment_alpha_post_swizzle = typename Callbacks::Fragment_alpha_post_swizzle;
    // The fragment for beta.
    using Fragment_beta = typename Callbacks::Fragment_beta;

    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;

    // The number of XMMAs.
    enum { XMMAS_M = Xmma_tile::XMMAS_M, XMMAS_N = Xmma_tile::XMMAS_N };

    // Ctor.
    template <typename Params>
    inline __device__ Epilogue_dbns( const Params &params,
                                     Gmem_tile &gmem_tile,
                                     Swizzle &swizzle,
                                     Callbacks &callbacks,
                                     const Named_barrier &epi_sync = Named_barrier(),
                                     const int bidm = blockIdx.x,
                                     const int bidn = blockIdx.y,
                                     const int bidz = blockIdx.z,
                                     const int tidx = threadIdx.x,
                                     const bool is_warp_specialized = false )
        : gmem_tile_( gmem_tile ), swizzle_( swizzle ), callbacks_( callbacks ),
          epi_sync_( epi_sync ), mem_desc_c_( params.mem_descriptors.descriptor_c ),
          mem_desc_d_( params.mem_descriptors.descriptor_d ) {
    }

    // Do the epilogue.
    template <bool WITH_RESIDUAL, typename Fragment_aclwmulator, int M, int N>
    inline __device__ void execute( Fragment_aclwmulator ( &acc )[M][N] ) {
#pragma unroll
        for( int mi = 0; mi < M; ++mi ) {
            this->step<WITH_RESIDUAL>( mi, acc[mi] );
        }
    }

    // Do only split-k for a 2-kernel split-k.
    template <int N> inline __device__ void exelwte_split_k() {
    }

    template <int N, typename dtype>
    inline __device__ void to_float( float ( &dst )[2 * N], dtype ( &src )[N] ) {
#pragma unroll
        for( int i = 0; i < N; ++i ) {
            uint16_t lo, hi;
            asm volatile( "mov.b32 {%0, %1}, %2;" : "=h"( lo ), "=h"( hi ) : "r"( src[i] ) );
            asm volatile( "cvt.f32.f16 %0, %1;" : "=f"( dst[2 * i + 0] ) : "h"( lo ) );
            asm volatile( "cvt.f32.f16 %0, %1;" : "=f"( dst[2 * i + 1] ) : "h"( hi ) );
        }
    }

    // Execute a single iteration of the loop.
    template <bool WITH_RESIDUAL, typename Fragment_aclwmulator, int N>
    inline __device__ void step( int mi, Fragment_aclwmulator ( &acc )[N] ) {

        // The output masks.
        int out_masks[Gmem_tile::STGS];
#pragma unroll
        for( int ii = 0; ii < Gmem_tile::STGS; ++ii ) {
            out_masks[ii] = this->gmem_tile_.compute_output_mask( mi, ii );
        }

        if( mi == 0 ) {
            this->gmem_tile_.load_bn_fprop_mean(
                bn_fprop_mean[0], 0, 0, out_masks[0], mem_desc_c_ );
            this->gmem_tile_.load_bn_fprop_ilw_stddev(
                bn_fprop_ilw_stddev[0], 0, 0, out_masks[0], mem_desc_c_ );
            this->gmem_tile_.load_dual_bn_fprop_mean(
                dual_bn_fprop_mean[0], 0, 0, out_masks[0], mem_desc_c_ );
            this->gmem_tile_.load_dual_bn_fprop_ilw_stddev(
                dual_bn_fprop_ilw_stddev[0], 0, 0, out_masks[0], mem_desc_c_ );
        }

        // Load valid values if beta is not zero.
        Fragment_c res_fetch[Gmem_tile::STGS];
        Fragment_c bitmask[Gmem_tile::STGS];
        if( WITH_RESIDUAL ) {
#pragma unroll
            for( int ii = 0; ii < Gmem_tile::STGS; ++ii ) {
                this->gmem_tile_.load( res_fetch[ii], mi, ii, out_masks[ii], mem_desc_c_ );
            }

#pragma unroll
            for( int ii = 0; ii < Gmem_tile::STGS; ++ii ) {
                this->gmem_tile_.load_bitmask_full(
                    bitmask[ii], mi, ii, out_masks[ii], mem_desc_c_ );
            }
        }

        // Load fprop values
        Fragment_c x_fetch[Gmem_tile::STGS];
#pragma unroll
        for( int ii = 0; ii < Gmem_tile::STGS; ++ii ) {
            this->gmem_tile_.load_bn_fprop_tensor(
                x_fetch[ii], mi, ii, out_masks[ii], mem_desc_c_ );
        }

        Fragment_c dual_x_fetch[Gmem_tile::STGS];
#pragma unroll
        for( int ii = 0; ii < Gmem_tile::STGS; ++ii ) {
            this->gmem_tile_.load_dual_bn_fprop_tensor(
                dual_x_fetch[ii], mi, ii, out_masks[ii], mem_desc_c_ );
        }

// Do something before we colwert
#pragma unroll
        for( int ni = 0; ni < N; ++ni ) {
            callbacks_.pre_colwert( *this, mi, ni, acc[ni] );
        }

        // Colwert the aclwmulators to the epilogue format (or keep them as-is).
        Fragment_pre_swizzle pre_swizzle[N];
#pragma unroll
        for( int ni = 0; ni < N; ++ni ) {
            pre_swizzle[ni].shuffle_groups( acc[ni] );
        }

        // Load alpha.
        Fragment_alpha_pre_swizzle alpha_pre_swizzle[N];
#pragma unroll
        for( int ni = 0; ni < N; ++ni ) {
            callbacks_.alpha_pre_swizzle( *this, mi, ni, alpha_pre_swizzle[ni] );
        }

// Do the colwersion.
#pragma unroll

        for( int ni = 0; ni < N; ++ni ) {
            pre_swizzle[ni].colwert( alpha_pre_swizzle[ni], acc[ni] );
        }

// Do something before we swizzle.
#pragma unroll
        for( int ni = 0; ni < N; ++ni ) {
            callbacks_.pre_swizzle( *this, mi, ni, pre_swizzle[ni] );
        }

        // Make sure the main loop or the previous loop of the epilogue are finished.
        if( !Swizzle::SKIP_SYNCTHREADS ) {
            if( epi_sync_.invalid() ) {
                __syncthreads();
            } else {
                epi_sync_.wait();
            }
        }

// Store the data in shared memory to produce more friendly stores.
#pragma unroll
        for( int ni = 0; ni < N; ++ni ) {
            this->swizzle_.store( ni, pre_swizzle[ni] );
        }

        // Make sure the data is in SMEM.
        if( !Swizzle::SKIP_SYNCTHREADS ) {
            if( epi_sync_.invalid() ) {
                __syncthreads();
            } else {
                epi_sync_.wait();
            }
        }

        // The fragments after the swizzle. One fragment per STG.128.
        Fragment_post_swizzle post_swizzle[Gmem_tile::STGS];
#pragma unroll
        for( int ii = 0; ii < Gmem_tile::STGS; ++ii ) {
            this->swizzle_.load( post_swizzle[ii], ii );
        }

        // Load alpha post swizzle.
        Fragment_alpha_post_swizzle alpha_post_swizzle[Gmem_tile::STGS];
#pragma unroll
        for( int ii = 0; ii < Gmem_tile::STGS; ++ii ) {
            callbacks_.alpha_post_swizzle( *this, mi, ii, alpha_post_swizzle[ii] );
        }

// Do the parallel reduction, if needed.
#pragma unroll
        for( int ii = 0; ii < Gmem_tile::STGS; ++ii ) {
            post_swizzle[ii].reduce( alpha_post_swizzle[ii] );
        }

        // Add the residual value before packing. TODO: We should be able to pass a single beta.
        if( WITH_RESIDUAL ) {
#pragma unroll
            for( int ii = 0; ii < Gmem_tile::STGS; ++ii ) {
                post_swizzle[ii].add( res_fetch[ii] );
            }
        }

// Do something now that the data has been swizzled.
#pragma unroll
        for( int ii = 0; ii < Gmem_tile::STGS; ++ii ) {
            callbacks_.post_swizzle( *this,
                                     mi,
                                     ii,
                                     post_swizzle[ii],
                                     bn_fprop_mean[0],
                                     bn_fprop_ilw_stddev[0],
                                     x_fetch[ii],
                                     dual_bn_fprop_mean[0],
                                     dual_bn_fprop_ilw_stddev[0],
                                     dual_x_fetch[ii],
                                     out_masks[ii] );
        }

        // Do something before packing and pack to produce a STG.128.
        // Put in one loop for F2IP.RELU optimization.
        Fragment_c out_regs[Gmem_tile::STGS];
#pragma unroll
        for( int ii = 0; ii < Gmem_tile::STGS; ++ii ) {
            callbacks_.pre_pack( *this, mi, ii, post_swizzle[ii] );
            out_regs[ii].pack( alpha_post_swizzle[ii], post_swizzle[ii] );
        }

// Do something before we store.
#pragma unroll
        for( int ii = 0; ii < Gmem_tile::STGS; ++ii ) {
            callbacks_.pre_store( *this, mi, ii, out_regs[ii], out_masks[ii] );
        }

// Write valid values.
#pragma unroll
        for( int ii = 0; ii < Gmem_tile::STGS; ++ii ) {
            this->gmem_tile_.store( mi, ii, out_regs[ii], out_masks[ii], mem_desc_d_ );
        }
    }

    // The output tile.
    Gmem_tile &gmem_tile_;
    // The shared memory tile.
    Swizzle &swizzle_;

    // The callbacks.
    Callbacks &callbacks_;

    char *smem;

    // The named barrier object used for epilog sync.
    const Named_barrier &epi_sync_;

    // Ampere memory descriptors
    const uint64_t mem_desc_c_, mem_desc_d_;

    // BN fprop mean and var
    Fragment<float, 8> bn_fprop_mean[1];
    Fragment<float, 8> bn_fprop_ilw_stddev[1];
    Fragment<float, 8> dual_bn_fprop_mean[1];
    Fragment<float, 8> dual_bn_fprop_ilw_stddev[1];
};

}  // namespace helpers
}  // namespace xmma
