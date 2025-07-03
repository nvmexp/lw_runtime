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

#include <xmma/helpers/gemm.h>

namespace xmma {
namespace helpers {

////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    int M_FIRST,
    int M_LAST,
    int N_FIRST,
    int N_LAST,
    typename Fragment_aclwmulators,
    typename Fragment_a,
    typename Fragment_b,
    typename Fragment_e,
    int M,
    int N,
    int N_PER_GROUP
>
inline __device__
void sparse_gemm(Fragment_aclwmulators (&acc)[M][N_PER_GROUP],
          const Fragment_a (&a)[M],
          const Fragment_b (&b)[N],
          const Fragment_e (&e)[1]) {
    // The number of groups.
    const int GROUPS = N / N_PER_GROUP;
    // The number of XMMAs per group.
    const int M_PER_GROUP = M / GROUPS;

    // Compute the MMAs.
    #pragma unroll
    for( int gi = 0; gi < GROUPS; ++gi ) {

        #pragma unroll
        for( int ni = N_FIRST; ni < N_LAST; ++ni ) {
            #pragma unroll
            for( int mi = M_FIRST; mi < M_LAST; ++mi ) {
                if (mi == 0) {
                    jetfire::warp_switch();
                }
                acc[mi + gi*N_PER_GROUP][ni].mma(a[mi + gi*M_PER_GROUP], b[ni + gi*N_PER_GROUP],
                    e[0].reg(mi/2), mi%2);
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    typename Fragment_aclwmulators,
    typename Fragment_a,
    typename Fragment_b,
    typename Fragment_e,
    int M,
    int N,
    int N_PER_GROUP
>
inline __device__
void sparse_igemm_unroll(Fragment_aclwmulators (&acc)[M][N_PER_GROUP],
          const Fragment_a (&a)[M],
          const Fragment_b (&b)[N],
          const Fragment_e (&e)[1]) {

    #pragma unroll
    for( int mi = 0; mi < M; ++mi ) {
        #pragma unroll
        for( int ni = 0; ni < N; ++ni ) {
            if (ni == 0) {
                jetfire::warp_switch();
            }
            acc[mi][ni].mma(a[mi], b[ni], e[0].reg(mi), 0);
        }
    }


}

////////////////////////////////////////////////////////////////////////////////////////////////////
template<
    int M_FIRST,
    int M_LAST,
    int N_FIRST,
    int N_LAST,
    typename Fragment_aclwmulators,
    typename Fragment_a,
    typename Fragment_b,
    typename Fragment_e,
    int M,
    int N,
    int N_PER_GROUP
>
inline __device__
void sparse_igemm(Fragment_aclwmulators (&acc)[M][N_PER_GROUP],
          const Fragment_a (&a)[M],
          const Fragment_b (&b)[N],
          const Fragment_e (&e)[1]) {
    // The number of groups.
    const int GROUPS = N / N_PER_GROUP;
    // The number of XMMAs per group.
    const int M_PER_GROUP = M / GROUPS;

    // Compute the MMAs.
    #pragma unroll
    for( int gi = 0; gi < GROUPS; ++gi ) {
        #pragma unroll
        for( int mi = M_FIRST; mi < M_LAST; ++mi ) {
            #pragma unroll
            for( int ni = N_FIRST; ni < N_LAST; ++ni ) {
                if (ni == 0) {
                    jetfire::warp_switch();
                }
                acc[mi + gi*N_PER_GROUP][ni].mma(a[mi + gi*M_PER_GROUP], b[ni + gi*N_PER_GROUP],
                    e[0].reg(mi), 0);
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    typename Fragment_aclwmulators, 
    typename Fragment_a, 
    typename Fragment_b, 
    typename Fragment_e, 
    int M, 
    int N, 
    int N_PER_GROUP
>
inline __device__ 
void spgemm(Fragment_aclwmulators (&acc)[M][N_PER_GROUP], 
          const Fragment_a (&a)[M], 
          const Fragment_b (&b)[N],
          const Fragment_e (&e)[1]) {
    // The number of groups.
    const int GROUPS = N / N_PER_GROUP;
    // The number of XMMAs per group.
    const int M_PER_GROUP = M / GROUPS;
    // The total number of MMAs
    const int TOTAL = GROUPS * M_PER_GROUP * N_PER_GROUP;
    // The running count of MMAs issued
    int mma_count = 0;

    // Only enable fencing if it looks like our desired stage overlap will fit in register budget.
    constexpr bool JETFIRE_FENCING_ENABLED = (255 * 4 > sizeof(acc) + sizeof(a) + sizeof(b) 
            + sizeof(e));
    
    // Compute the MMAs.
    #pragma unroll
    for( int gi = 0; gi < GROUPS; ++gi ) {
        //#pragma unroll
        //for( int mi = 0; mi < M_PER_GROUP; ++mi ) {
            #pragma unroll
            for( int ni = 0; ni < N_PER_GROUP; ++ni ) {
                // Interference fence halfway through to switch to a different LDS scoreboard
                if (mma_count == TOTAL / 2)
                    jetfire::ifence(JETFIRE_FENCING_ENABLED);
                // Manually break down the loop in m dimension to get better insts seq
                acc[0 + gi*N_PER_GROUP][ni].spmma_s0(a[0 + gi*M_PER_GROUP], 
                    b[ni + gi*N_PER_GROUP], e[0]);
                
                acc[1 + gi*N_PER_GROUP][ni].spmma_s1(a[1 + gi*M_PER_GROUP], 
                    b[ni + gi*N_PER_GROUP], e[0]);
                
                acc[2 + gi*N_PER_GROUP][ni].spmma_s2(a[2 + gi*M_PER_GROUP], 
                    b[ni + gi*N_PER_GROUP], e[0]);
                
                acc[3 + gi*N_PER_GROUP][ni].spmma_s3(a[3 + gi*M_PER_GROUP], 
                    b[ni + gi*N_PER_GROUP], e[0]);  
            }
        //}
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    typename Fragment_aclwmulators, 
    typename Fragment_a, 
    typename Fragment_b, 
    typename Fragment_e, 
    int M, 
    int N, 
    int N_PER_GROUP
>
inline __device__ 
void spgemm_pipeline(Fragment_aclwmulators (&acc)[M][N_PER_GROUP], 
          const Fragment_a (&a)[M], 
          const Fragment_b (&b)[N],
          const Fragment_e (&e)[1], 
          const int pipe_stage) {

        // Manually break down the loop in m dimension to get better insts seq
        acc[0][pipe_stage].spmma_s0(a[0], b[0], e[0]);
        acc[1][pipe_stage].spmma_s1(a[1], b[0], e[0]);
        acc[2][pipe_stage].spmma_s2(a[2], b[0], e[0]);
        acc[3][pipe_stage].spmma_s3(a[3], b[0], e[0]);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    typename Fragment_aclwmulators, 
    typename Fragment_a, 
    typename Fragment_b, 
    typename Fragment_e, 
    int M, 
    int N, 
    int N_PER_GROUP
>
inline __device__ 
void sparse_igemm_pipeline(Fragment_aclwmulators (&acc)[M][N_PER_GROUP], 
          const Fragment_a (&a)[M], 
          const Fragment_b (&b)[N],
          const Fragment_e (&e)[1], 
          const int pipe_stage) {

        acc[0][pipe_stage].mma(a[0], b[0], e[0].reg(0), 0);
        acc[1][pipe_stage].mma(a[1], b[0], e[0].reg(1), 0);
        acc[2][pipe_stage].mma(a[2], b[0], e[0].reg(2), 0);
        acc[3][pipe_stage].mma(a[3], b[0], e[0].reg(3), 0);
}


////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace helpers
} // namespace xmma 



