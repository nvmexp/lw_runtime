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
#include <xmma/ext/first_layer/fprop/traits.h>

namespace xmma {
namespace ext {
namespace first_layer {
namespace fprop {

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Kernel_traits >
inline __device__ void device(const typename Kernel_traits::Params &params) {

    // The traits class.
    using Traits = typename Kernel_traits::Traits;
    // The CTA tile.
    using Cta_tile = typename Kernel_traits::Cta_tile;
    // The XMMA tile.
    using Xmma_tile = typename Kernel_traits::Xmma_tile;

    // The number of XMMAs in the M dimension.
    enum { XMMAS_M = Xmma_tile::XMMAS_M };
    // The number of XMMAs in the N dimension.
    enum { XMMAS_N = Xmma_tile::XMMAS_N };
    // The number of XMMAs in the K dimension.
    enum { XMMAS_K = Xmma_tile::XMMAS_K };

    // The block/tile indices.
    const int bidm = blockIdx.x;
    const int bidn = blockIdx.y;

    // Decompose the M index into N and PQ.
    int cta_n, cta_pq;
    xmma::fast_divmod(cta_n, cta_pq, bidm, params.ctas_pq,
                                               params.mul_ctas_pq,
                                               params.shr_ctas_pq);

    // Decompose the block index to find the P = PQ / Q and Q = PQ % Q.
    int cta_p, cta_q;
    xmma::fast_divmod(cta_p, cta_q, cta_pq, params.ctas_q,
                                                params.mul_ctas_q,
                                                params.shr_ctas_q);

    // The position in the K dimension.
    const int cta_k = bidn;

    // The thread index.
    const int tidx = threadIdx.x;

    // The tiles in global memory for the images.
    using Gmem_tile_prologue_a = typename Kernel_traits::Gmem_tile_prologue_a;
    // The tiles in global memory for the images.
    using Gmem_tile_loop_a = typename Kernel_traits::Gmem_tile_loop_a;
    // The tiles to store the data in shared memory for the images.
    using Smem_tile_a = typename Kernel_traits::Smem_tile_a;

    // The tiles in global memory for the filters.
    using Gmem_tile_prologue_b = typename Kernel_traits::Gmem_tile_prologue_b;
    // The tiles to store the data in shared memory for the filters.
    using Smem_tile_b = typename Kernel_traits::Smem_tile_b;

    // The tiles in global memory for the filters.
    using Gmem_tile_c = typename Kernel_traits::Gmem_tile_c;
    // The tiles to store the data in shared memory for the filters.
    using Smem_tile_c = typename Kernel_traits::Smem_tile_c;

    // The shared memory. It is allocated at launch time.
    extern __shared__ char smem_[];

    // The shared memory pointers.
    char* smem_a_ = &smem_[0];
    char* smem_b_ = &smem_[Smem_tile_a::BYTES_PER_TILE];
    char* smem_c_ = &smem_[Smem_tile_a::BYTES_PER_TILE + Smem_tile_b::BYTES_PER_TILE];

    // The tiles in shared memory.
    Smem_tile_a smem_a(smem_a_, tidx);
    Smem_tile_b smem_b(smem_b_, tidx);

    // Create the objects to load from global memory in the prologue.
    Gmem_tile_prologue_a gmem_prologue_a(params, nullptr, cta_n, cta_p, cta_q, tidx);
    Gmem_tile_prologue_b gmem_prologue_b(params, nullptr, cta_k, tidx);

    // Do we use LDGSTS?
    enum { USE_LDGSTS = Gmem_tile_prologue_a::USE_LDGSTS || Gmem_tile_prologue_b::USE_LDGSTS };

    // Trigger the loads for A and B. Either LDG or LDGSTS.
    gmem_prologue_a.load(smem_a);
    gmem_prologue_b.load(smem_b);

    // Make sure we insert the corresponding LDGDEPBAR. NOP on Volta/Turing.
    xmma::ldgdepbar<USE_LDGSTS>();

    // Store the pixels and filters to shared memory.
    gmem_prologue_a.commit(smem_a);
    gmem_prologue_b.commit(smem_b);

    // Move to next SMEM buffer for multistage or double buffer.
    smem_a.move_write_offset(true);

    // Make sure the data is in shared memory.
    xmma::depbar<USE_LDGSTS, 1>(); __syncthreads();

    // Create the objects to load from global memory in the loop.
    Gmem_tile_loop_a gmem_loop_a(params, nullptr, cta_n, cta_p, cta_q, tidx);

    // The tile in global memory.
    using Gmem_tile_c = typename Kernel_traits::Gmem_tile_c;
    // The tile in shared memory to swizzle the output.
    using Smem_tile_c = typename Kernel_traits::Smem_tile_c;
    // The callbacks.
    using Callbacks_epilogue = typename Kernel_traits::Callbacks_epilogue;
    // The epilogue.
    using Epilogue = typename Kernel_traits::Epilogue;

    // Do allocate the tile to output in the epilogue.
    Gmem_tile_c gmem_c(params, cta_n, cta_p, cta_q, cta_k, tidx);
    // Do allocate the tile and compute the offsets. TODO: Merge with the epilogue class!
    Smem_tile_c smem_c(smem_c_, tidx);
    // The callbacks.
    Callbacks_epilogue callbacks_epilogue(params, smem_c_, bidm, bidn, 0, tidx);

    // The number of rows computed by that CTA.
    int rows_per_cta = min(params.p - cta_p*params.out_rows_per_cta, params.out_rows_per_cta);
    // The number of outer loops.
    int outer_loop_count = (rows_per_cta + Kernel_traits::OUT_H-1) / Kernel_traits::OUT_H;
    // The outer-most loop.
    for( int outer_loop = outer_loop_count-1; outer_loop >= 0; --outer_loop ) {

        // // DEBUG.
        // smem_a.debug_print();
        // smem_b.debug_print();
        // // END OF DEBUG.

        // The fragments to load A.
        typename Smem_tile_a::Fragment a[2][XMMAS_M];
        smem_a.load(a[0], 0, 0);

        // The fragments to load B.
        typename Smem_tile_b::Fragment b[2][XMMAS_N];
        smem_b.load(b[0], 0, 0);

        // The aclwmulators.
        xmma::Fragment_aclwmulator<Traits> acc[XMMAS_M][XMMAS_N];
        xmma::helpers::clear(acc);


        // Only enable fencing if it looks like our desired stage overlap will fit in register budget.
        constexpr bool JETFIRE_FENCING_ENABLED = (255 * 4 > 
            sizeof(acc) + sizeof(a) + sizeof(b) + sizeof(a_gmem) + sizeof(b_gmem));

        // The position in the filter.
        int rsi = Kernel_traits::TAPS_PER_XMMA_K;

        // Iterate over the inner loop.
        JETFIRE_MAC_LOOP_PRAGMA  
        #pragma unroll 1
        for( int inner_loop = Kernel_traits::INNER_LOOPS-1; inner_loop >= 0; --inner_loop ) {
            JETFIRE_MAC_LOOP_HEADER

            // Make sure the number of XMMAs in the K dimension is odd.
            static_assert(XMMAS_K % 2 == 1, "");

            // The core part of the loop.
            #pragma unroll
            for( int ki = 0; ki < XMMAS_K; ++ki, rsi += Kernel_traits::TAPS_PER_XMMA_K ) {

                // The ki_next variable is "one-step ahead". The ki points to the current step.
                int ki_next = (ki == XMMAS_K-1) ? 0 : ki+1;

                // Interference fence at top of stage
                jetfire::ifence(JETFIRE_FENCING_ENABLED);

                // Trigger the commit of global loads in the last last iteration
                if( ki == XMMAS_K-1 ) {

                    // Make sure the data was read from shared memory.
                    __syncthreads();

                    // Store the data to shared memory for A.
                    gmem_loop_a.commit(smem_a);
                    jetfire::ifence(JETFIRE_FENCING_ENABLED);

                    // Make sure the data is in shared memory and synchronize threads.
                    xmma::depbar<Gmem_tile_loop_a::USE_LDGSTS, 1>(); __syncthreads();

                    // Move the shared memory write offset for the activations.
                    smem_a.move_write_offset();

                    // Move the read offsets for activations and filters.
                    smem_a.move_read_offset();
                    smem_b.move_read_offset();
                }

                // Load from shared memory before the math -- except for the last one.
                if( ki < XMMAS_K-1 ) {
                    smem_a.load(a[ki_next&1], rsi, ki_next);
                    smem_b.load(b[ki_next&1], rsi, ki_next);
                }

                // Inteference fence after smem loads.
                jetfire::ifence(JETFIRE_FENCING_ENABLED);

                // Trigger the global loads on the 1st iteration of that core loop.
                if( ki == 0 ) {

                    // Trigger the loads for the A matrix.
                    gmem_loop_a.load(smem_a);
                    jetfire::ifence(JETFIRE_FENCING_ENABLED);

                    // Push the LDGDEPBAR instruction after the loads for A and B.
                    xmma::ldgdepbar<Gmem_tile_loop_a::USE_LDGSTS>();
                    jetfire::ifence(JETFIRE_FENCING_ENABLED);
                }

                // // DEBUG.
                // if( tidx / 4 == 16 ) {
                //     printf("tidx=%3d rsi=%2d ki=%2d a[0][0]=%8.3f %8.3f b[0][0]=%8.3f %8.3f\n",
                //         tidx,
                //         rsi-2,
                //         ki,
                //         reinterpret_cast<const float&>(a[ki&1][0].reg(0)),
                //         reinterpret_cast<const float&>(a[ki&1][0].reg(2)),
                //         reinterpret_cast<const float&>(b[ki&1][0].reg(0)),
                //         reinterpret_cast<const float&>(b[ki&1][0].reg(1)));
                // }
                // // END OF DEBUG.

                // Do the math - The core of the loop does 16x16x8.
                xmma::helpers::gemm(acc, a[ki&1], b[ki&1]);

                // Load from shared memory after the for the last value since ki % 2 == 0.
                if( inner_loop > 0 && ki == XMMAS_K-1 ) {
                    smem_a.load(a[0], rsi, 0);
                    smem_b.load(b[0], rsi, 0);
                }

            }  // (ki/ki_next)

            // Move the global pointers.
            gmem_loop_a.move();

            // Trigger the residue in the last iteration of the outer loop.
            if( (outer_loop == 1 && inner_loop == 0) || outer_loop == 0 ) {
                gmem_loop_a.residue();
            }

        }  // (inner loop)

        // // DEBUG.
        // if( tidx / 4 == 16 ) {
        //     printf("tidx=%3d acc[0][0]=%.3f %.3f %.3f %.3f\n", 
        //         tidx, 
        //         acc[0][0].elt(0),
        //         acc[0][0].elt(1),
        //         acc[0][0].elt(2),
        //         acc[0][0].elt(3));
        // }
        // // END OF DEBUG.

        // Make sure we can use the shared memory.
        __syncthreads();

        // Do the epilogue.
        Epilogue epilogue(params, gmem_c, smem_c, callbacks_epilogue);
        if( params.beta != 0.f ) {
            epilogue.template execute<true>(acc);
        } else {
            epilogue.template execute<false>(acc);
        }

        // Move the global memory tile.
        gmem_c.move();

        // Reset the shared memory read offsets.
        smem_a.reset_read_offset();
        smem_b.reset_read_offset();

    } // (outer loop)
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace fprop
} // namespace first_layer 
} // namespace ext
} // namespace xmma 

