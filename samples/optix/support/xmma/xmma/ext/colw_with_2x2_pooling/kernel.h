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
#include <xmma/helpers/fragment.h>
#include <xmma/helpers/gemm.h>

namespace xmma {
namespace ext {
namespace colw_with_2x2_pooling {

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Kernel_traits, typename Params >
static inline __device__ void device(const Params &params) {

    // The traits class.
    using Traits = typename Kernel_traits::Traits;
    // The CTA tile.
    using Cta_tile = typename Kernel_traits::Cta_tile;
    // The tile distribution manager.
    using Cta_distribution = typename Kernel_traits::Cta_distribution;
    // The XMMA tile.
    using Xmma_tile = typename Kernel_traits::Xmma_tile;

    // The size of the filter.
    enum { FLT_R = Kernel_traits::Colw_filter::FLT_R };
    enum { FLT_S = Kernel_traits::Colw_filter::FLT_S };

    // The number of XMMAs.
    enum { XMMAS_M = Xmma_tile::XMMAS_M };
    enum { XMMAS_N = Xmma_tile::XMMAS_N };
    enum { XMMAS_K = Xmma_tile::XMMAS_K };

    // Initialize the tile distribution.
    Cta_distribution tile(params, blockIdx);

    // The block/tile indices.
    int bidm = tile.bidm();
    int bidn = tile.bidn();
    int bidz = tile.bidz();

    // The thread index.
    const int tidx = threadIdx.x;

    // The tiles to store the data in shared memory for the images.
    using Smem_tile_a = typename Kernel_traits::Smem_tile_a;
    // The tiles to store the data in shared memory for the filters.
    using Smem_tile_b = typename Kernel_traits::Smem_tile_b;

    // Make sure we do not use double buffering.
    static_assert(Smem_tile_a::BUFFERS_PER_TILE == 1, "");
    // Make sure we do not use double buffering.
    static_assert(Smem_tile_b::BUFFERS_PER_TILE == 1, "");

    // Dynamically allocated shared memory.
    extern __shared__ char smem_[];

    // The shared memory pointers.
    char *a_smem_ = &smem_[0];
    char *b_smem_ = &smem_[Smem_tile_a::BYTES_PER_TILE];

    // The tiles in shared memory.
    Smem_tile_a a_smem(a_smem_, tidx);
    Smem_tile_b b_smem(b_smem_, tidx);

    // Load from global memory.
    typename Kernel_traits::Gmem_tile_a a_gmem(params, nullptr, tile.bidx(), tidx);
    typename Kernel_traits::Gmem_tile_b b_gmem(params, nullptr, tile.bidx(), tidx);

    // Clear the aclwmulators.
    xmma::Fragment_aclwmulator<Traits> acc[XMMAS_M][XMMAS_N];
    xmma::helpers::clear(acc);

    // Issue the 1st loads.
    a_gmem.load(a_smem);
    b_gmem.load(b_smem);

    // Store the pixels and filters to shared memory.
    a_gmem.commit(a_smem);
    b_gmem.commit(b_smem);

    // Move the pointers and assemble the predicates for the next loop.
    a_gmem.move(               params.a_delta[0]);
    b_gmem.move(/*ignored*/ 0, params.b_delta[0]);

    // Make sure the data is in shared memory.
    __syncthreads();

    // Load the image pixels.
    typename Smem_tile_a::Fragment a[2][XMMAS_M];
    a_smem.load(a[0], /*ki*/ 0, /*ri*/ 0, /*si*/ 0);

    // Load the filters.
    typename Smem_tile_b::Fragment b[2][XMMAS_N];
    b_smem.load(b[0], 0); 


    // Only enable fencing if it looks like our desired stage overlap will fit in register budget.
    constexpr bool JETFIRE_FENCING_ENABLED = (255 * 4 > 
            sizeof(acc) + sizeof(a) + sizeof(b) + sizeof(a_gmem) + sizeof(b_gmem));

    // // DEBUG.
    // a_smem.debug_print();
    // b_smem.debug_print();
    // // END OF DEBUG.

    // Iterate over the loop (K dimension).
    JETFIRE_MAC_LOOP_PRAGMA
    for( int loop = params.loop_start ; loop >= 0; --loop ) {

        // Is it the last iteration?
        const int is_last_loop = loop == 0;

        // Trigger the global image loads for the next R x S iterations.
        if( is_last_loop ) {
            a_gmem.disable_loads();
        }

        // Loop over R.
        for( int ri = 0; ri < FLT_R; ++ri ) {
            // Loop over S.
            for( int si = 0; si < FLT_S; ++si ) {

                // The core of the loop.
                JETFIRE_MAC_LOOP_HEADER

                // Is it the last filter coefficient of the row?
                int is_last_filter_s = si == FLT_S-1;
                // Is it the last filter coefficient of the filter?
                int is_last_filter_rs = is_last_filter_s && ri == FLT_R-1;

                // Disable the filter loads if needed.
                if( is_last_loop && is_last_filter_rs ) {
                    b_gmem.disable_loads();
                }

                // Do the stages of math (except for the last one).
                #pragma unroll
                for( int ki = 1; ki < XMMAS_K; ++ki ) {

                    // Load the pixels and filters from shared memory.
                    a_smem.load(a[ki&1], ki, ri, si);
                    b_smem.load(b[ki&1], ki); 

                    // Trigger the global load for the next filter.
                    if( ki == 1 ) {
                        b_gmem.load(b_smem);
                    }

                    // Interference fence after smem and gmem loads.
                    jetfire::ifence(JETFIRE_FENCING_ENABLED);                       

                    // Do the math - The core of the loop does 16x16x8.
                    xmma::helpers::gemm(acc, a[(ki-1)&1], b[(ki-1)&1]);

                    // Interference fence after the MMAs.
                    jetfire::ifence(JETFIRE_FENCING_ENABLED);   

                } // (ki)
                
                // Load from global memory. 
                if( !is_last_loop && ri == 0 && si == 0 ) {
                    a_gmem.load(a_smem);
                }

                // Make sure the data in shared memory was read.
                __syncthreads();

                // If that's the last filter element, we want to store the image.
                if( is_last_filter_rs ) {
                    a_gmem.commit(a_smem);
                }

                // Commit the data to shared memory for the filter.
                b_gmem.commit(b_smem);

                // Make sure the data is in shared memory.
                __syncthreads();

                // The ki.
                const int ki = XMMAS_K;

                // Reset the load pointers for SMEM B. TODO: Find a better solution!
                #pragma unroll
                for( int kj = XMMAS_K; kj < 4; ++kj ) {
                    b_smem.load(b[XMMAS_K & 1], kj);
                }

                // The next filter coefficient.
                int next_ri = ri, next_si = si+1;
                if( is_last_filter_s ) {
                    next_ri = is_last_filter_rs ? 0 : ri+1;
                    next_si = 0;
                }

                // Interference fence after gmem commit a.
                jetfire::ifence(JETFIRE_FENCING_ENABLED);   

                // Interference fence after smem load.
                jetfire::ifence(JETFIRE_FENCING_ENABLED);   

                // Move the image pointer(s).
                if( is_last_filter_rs ) {
                    a_gmem.move(params.a_delta[0]);
                }

                // Move the filter.
                b_gmem.move(/*ignored*/ 0, params.b_delta[next_ri*FLT_S + next_si]);

                // Do the math - The core of the loop does 16x16x8.
                xmma::helpers::gemm(acc, a[(ki-1)&1], b[(ki-1)&1]);

                // Load the next pixels and filters.
                a_smem.load(a[0], 0, next_ri, next_si);
                b_smem.load(b[0], 0); 

            } // (si)
        } // (ri)
    } // (loop)

    // Do allocate the tile to output in the epilogue. 
    typename Kernel_traits::Gmem_tile_epilogue gmem_c(params, bidm, bidn, bidz, tidx);
    // Do allocate the tile and compute the offsets. 
    typename Kernel_traits::Swizzle_epilogue smem_c(smem_, tidx);
    // The callbacks.
    typename Kernel_traits::Callbacks_epilogue callbacks(params, smem_, bidm, bidn, bidz, tidx);

    // Do the epilogue.
    typename Kernel_traits::Epilogue epilogue(params, gmem_c, smem_c, callbacks);
    epilogue.execute<false>(acc);

    // Finalize the callbacks.
    callbacks.post_epilogue();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace colw_with_2x2_pooling
} // namespace ext
} // namespace xmma

