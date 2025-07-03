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
#include <xmma/fragment.h>
#include <xmma/numeric_types.h>

namespace xmma {
namespace helpers {

////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    typename Fragment_aclwmulators, 
    typename Fragment_a, 
    typename Fragment_b, 
    int M, 
    int N, 
    int N_PER_GROUP
>
inline __device__ void gemm(Fragment_aclwmulators (&acc)[M][N_PER_GROUP],
                            const Fragment_a (&a)[M],
                            const Fragment_b (&b)[N]) {
    // The number of groups.
    const int GROUPS = N / N_PER_GROUP;
    // The number of XMMAs per group.
    const int M_PER_GROUP = M / GROUPS;
    // The total number of MMAs
    const int TOTAL = GROUPS * M_PER_GROUP * N_PER_GROUP;
    // The running count of MMAs issued
    int mma_count = 0;

    // Only enable fencing if it looks like our desired stage overlap will fit in register budget.
    constexpr bool JETFIRE_FENCING_ENABLED = (255 * 4 > sizeof(acc) + sizeof(a) + sizeof(b));
    
    // Compute the MMAs.
    #pragma unroll
    for( int gi = 0; gi < GROUPS; ++gi ) {
        #pragma unroll
        for( int mi = 0; mi < M_PER_GROUP; ++mi ) {
            #pragma unroll
            for( int ni = 0; ni < N_PER_GROUP; ++ni ) {
                // Interference fence halfway through to switch to a different LDS scoreboard
                if (mma_count == TOTAL / 2)
                    jetfire::ifence(JETFIRE_FENCING_ENABLED);

                acc[mi + gi*N_PER_GROUP][ni].mma(a[mi + gi*M_PER_GROUP], b[ni + gi*N_PER_GROUP]);
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Fragment_aclwmulators, typename Smem_a, typename Smem_b >
inline __device__ void gemm(
    Fragment_aclwmulators (&acc)[Smem_a::Xmma_tile::XMMAS_M][Smem_b::Xmma_tile::XMMAS_N], 
    Smem_a &smem_a, 
    Smem_b &smem_b) {

    // The XMMA tile.
    const int XMMAS_M = Smem_a::Xmma_tile::XMMAS_M;
    const int XMMAS_N = Smem_b::Xmma_tile::XMMAS_N;
    const int XMMAS_K = Smem_a::Xmma_tile::XMMAS_K;

    // The fragments for A and B.
    using Fragment_a = typename Smem_a::Fragment;
    using Fragment_b = typename Smem_b::Fragment;

    // Load A from shared memory.
    Fragment_a a[2][XMMAS_M];
    smem_a.load(a[0], 0);

    // Load B from shared memory.
    Fragment_b b[2][XMMAS_N];
    smem_b.load(b[0], 0); 

    // Do the main loop.
    #pragma unroll
    for( int ki_ = 1; ki_ <= XMMAS_K; ++ki_ ) {

        // We want the loop index to be ki_ % XMMAS_K.
        int ki = ki_ == XMMAS_K ? 0 : ki_;

        // Load the A fragments from shared memory.
        smem_a.load(a[ki&1], ki);

        // Load the B fragments from shared memory.
        smem_b.load(b[ki&1], ki); 

        // Do the math - The core of the loop does 16x16x8.
        xmma::helpers::gemm(acc, a[(ki-1)&1], b[(ki-1)&1]);

    } // (ki)
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace helpers
} // namespace xmma 

