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

namespace xmma {
namespace helpers {

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Output_type, typename Layout_a, typename Layout_b, int M, int N, int N_PER_GROUP >
inline __device__ void gemm(
    Fragment_aclwmulator<Ampere_hmma_tf32_traits<float, Output_type>> (&acc)[M][N_PER_GROUP],
    const Fragment_a<Ampere_hmma_tf32_traits<float, Output_type>, Layout_a> (&a)[M],
    const Fragment_b<Ampere_hmma_tf32_traits<float, Output_type>, Layout_b> (&b)[N]) {

    // The fragment types.
    using Fragment_a_ = Fragment_a<Ampere_hmma_tf32_traits<float, Output_type>, Layout_a>;
    using Fragment_b_ = Fragment_b<Ampere_hmma_tf32_traits<float, Output_type>, Layout_b>; 

    // The number of groups.
    const int GROUPS = N / N_PER_GROUP;
    // The number of XMMAs per group.
    const int M_PER_GROUP = M / GROUPS;

    // Compute the MMAs.
    #pragma unroll
    for( int gi = 0; gi < GROUPS; ++gi ) {
        #pragma unroll
        for( int mi = 0; mi < M_PER_GROUP; ++mi ) {
            #pragma unroll
            for( int ni = 0; ni < N_PER_GROUP; ++ni ) {
                acc[mi + gi*N_PER_GROUP][ni].mma(a[mi + gi*M_PER_GROUP], 
                                                 b[ni + gi*N_PER_GROUP]);
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace helpers
} // namespace xmma 

