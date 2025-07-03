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

#include <xmma/jetfire/jetfire.h>
#include <xmma/utils.h>
#include <xmma/helpers/fragment.h>
#include <xmma/helpers/gemm.h>

namespace xmma {
namespace ext {
namespace implicit_gemm {

template <typename Kernel_traits>
static __global__ __launch_bounds__( Kernel_traits::Cta_tile::THREADS_PER_CTA ) 
void kernel( typename Kernel_traits::Params params ) {

    // The traits class.
    using Traits = typename Kernel_traits::Traits;
    // The CTA tile.
    using Cta_tile = typename Kernel_traits::Cta_tile;
    // The XMMA tile.
    using Xmma_tile = typename Kernel_traits::Xmma_tile;
    // The tile distribution manager.
    using Tile_distribution = typename Kernel_traits::Tile_distribution;

    // The number of XMMAs.
    const int XMMAS_M = Xmma_tile::XMMAS_M;
    const int XMMAS_N = Xmma_tile::XMMAS_N;
    const int XMMAS_K = Xmma_tile::XMMAS_K;

    // Initialize the tile distribution.
    Tile_distribution tile( params, blockIdx );

    // The block/tile indices.
    int bidm = tile.bidm();
    int bidn = tile.bidn();
    int bidz = tile.bidz();

    // The thread index.
    const int tidx = threadIdx.x;

    // The tiles in global memory for the images.
    using Gmem_tile_a = typename Kernel_traits::Gmem_tile_a;
    // The fragment.
    using Fragment_a = typename Kernel_traits::Fragment_a;
    using Data_type = typename Fragment_a::Data_type;
    using Input_type_ = typename Fragment_a::Input_type_;

    // The tiles in global memory for the filters.
    using Gmem_tile_b = typename Kernel_traits::Gmem_tile_b;
    // The fragment.
    using Fragment_b = typename Kernel_traits::Fragment_b;

    // The shared memory. It is allocated at launch time.
    extern __shared__ char smem_[];

    // The shared memory pointers.
    char* i_smem_ = &smem_[0];

    // Create the objects to load from global memory.
    Gmem_tile_a a_gmem( params, i_smem_, tile.bidx(), tidx );
    Gmem_tile_b b_gmem( params, i_smem_, tile.bidx(), tidx );

    // Create the compute tile and clear the aclwmulators.
    xmma::Fragment_aclwmulator<Traits> acc[XMMAS_M][XMMAS_N];
    Fragment_a a[XMMAS_K][XMMAS_M];
    Fragment_b b[XMMAS_K][XMMAS_N];

    xmma::helpers::clear(acc);

#ifdef XMMA_ENABLED_PMTRIG
    __prof_trigger(15);
#endif

    // M A I N   L O O P
    #pragma unroll (1)
    for (int loop = params.loop_start; loop >= 0; loop--) {
        JETFIRE_MAC_LOOP_HEADER

        a_gmem.load(a, params);
        b_gmem.load(b);

        #pragma unroll
        for (int ki = 0; ki < XMMAS_K; ki++) {
            /* GEMM */
            xmma::helpers::gemm(acc, a[ki], b[ki]);
        }

        if (loop==0) break;

        __syncthreads();
        a_gmem.move(params);
        b_gmem.move();
        __syncthreads();
    }

#ifdef XMMA_ENABLED_PMTRIG
    __prof_trigger(15);
#endif

    // The tile in global memory.
    using Gmem_tile_epilogue = typename Kernel_traits::Gmem_tile_epilogue;
    // The tile in shared memory to swizzle the output.
    using Swizzle_epilogue = typename Kernel_traits::Swizzle_epilogue;
    // The callbacks.
    using Callbacks_epilogue = typename Kernel_traits::Callbacks_epilogue;
    // The epilogue.
    using Epilogue = typename Kernel_traits::Epilogue;

    // Do allocate the tile to output in the epilogue.
    Gmem_tile_epilogue gmem_epilogue( params, bidm, bidn, bidz, tidx );
    // Do allocate the tile and compute the offsets. TODO: Merge with the epilogue class!
    Swizzle_epilogue swizzle_epilogue( smem_, tidx );
    // The callbacks.
    Callbacks_epilogue callbacks_epilogue( params, smem_, bidm, bidn, bidz, tidx );

    // Do the epilogue.
    Epilogue epilogue( params, gmem_epilogue, swizzle_epilogue, callbacks_epilogue );

    epilogue.execute( acc );
}

template <typename Kernel_traits>
static __global__ __launch_bounds__( Kernel_traits::Cta_tile::THREADS_PER_CTA )
void split_k_kernel( typename Kernel_traits::Params params ) {
    // The traits class.
    using Traits = typename Kernel_traits::Traits;
    // The CTA tile.
    using Cta_tile = typename Kernel_traits::Cta_tile;
    // The XMMA tile.
    using Xmma_tile = typename Kernel_traits::Xmma_tile;

    // Initialize the tile distribution.
    typename Kernel_traits::Tile_distribution tile( params, blockIdx );

    // The block indices.
    int bidm = tile.bidm();
    int bidn = tile.bidn();
    int bidz = tile.bidz();

    // The thread index.
    const int tidx = threadIdx.x;

    // The output tile for the epilogue.
    using Gmem_tile_epilogue = typename Kernel_traits::Gmem_tile_epilogue;
    // The shared memory swizzle.
    using Swizzle_epilogue = typename Kernel_traits::Swizzle_epilogue;
    // The callbacks.
    using Callbacks_epilogue = typename Kernel_traits::Callbacks_epilogue;
    // The epilogue.
    using Epilogue = typename Kernel_traits::Epilogue;

    // Do allocate the tile to output in the epilogue.
    Gmem_tile_epilogue gmem_epilogue( params, bidm, bidn, bidz, tidx );
    // Do allocate the tile and compute the offsets. TODO: Merge with the epilogue class!
    Swizzle_epilogue swizzle_epilogue( 0, tidx );
    // The callbacks.
    Callbacks_epilogue callbacks( params, 0, bidm, bidn, 0, tidx );

    // Do the epilogue.
    Epilogue epilogue( params, gmem_epilogue, swizzle_epilogue, callbacks );
    epilogue.template exelwte_split_k<Xmma_tile::XMMAS_N>();
}

}  // namespace implicit_gemm
}  // namespace ext
} // namespace xmma
