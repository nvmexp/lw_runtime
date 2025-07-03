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

#include <xmma/ext/sparse/utils.h>

#include <xmma/helpers/epilogue_with_split_k.h>
#include <xmma/helpers/fragment.h>
#include <xmma/ext/sparse/helpers/gemm.h>
#include <xmma/ext/sparse/spgemm/sphmma/gmem_tile.h>
#include <xmma/ext/sparse/ampere/smem_tile_sparse.h>

////////////////////////////////////////////////////////////////////////////////////////////////////

extern __shared__ char smem_[];

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace xmma {
namespace ext {
namespace gemm {
namespace sparse_hmma_gemm {

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Gemm_traits >
static __global__ __launch_bounds__(Gemm_traits::Cta_tile::THREADS_PER_CTA)
void kernel(typename Gemm_traits::Params params) {

    // The traits class.
    using Traits = typename Gemm_traits::Traits;
    // The CTA tile.
    using Cta_tile = typename Gemm_traits::Cta_tile;
    // The XMMA tile.
    using Xmma_tile = typename Gemm_traits::Xmma_tile;

    // The number of XMMAs.
    const int XMMAS_M = Xmma_tile::XMMAS_M;
    const int XMMAS_N = Xmma_tile::XMMAS_N;
    const int XMMAS_K = Xmma_tile::XMMAS_K;

    // The block indices.
    int bidm, bidn;
    if( params.use_horizontal_cta_rasterization ) {
        bidm = blockIdx.y;
        bidn = blockIdx.x;
    } else {
        bidm = blockIdx.x;
        bidn = blockIdx.y;
    }

    const int bidz = blockIdx.z;

    // The thread index.
    const int tidx = threadIdx.x;
    // Ampere memory descritors.
    const uint64_t mem_desc_a = params.mem_descriptors.descriptor_a;
    const uint64_t mem_desc_b = params.mem_descriptors.descriptor_b;

    // The tiles in global memory for the images.
    using Gmem_tile_a = typename Gemm_traits::Gmem_tile_a;
    // The tiles to store the data in shared memory for the images.
    using Smem_tile_a = typename Gemm_traits::Smem_tile_a;
    // The fragment.
    using Fragment_a = typename Smem_tile_a::Fragment;

    // The tiles in global memory for the filters.
    using Gmem_tile_b = typename Gemm_traits::Gmem_tile_b;
    // The tiles to store the data in shared memory for the filters.
    using Smem_tile_b = typename Gemm_traits::Smem_tile_b;
    // The fragment.
    using Fragment_b = typename Smem_tile_b::Fragment;

    // The tiles in global memory for the metadata.
    using Gmem_tile_e = typename Gemm_traits::Gmem_tile_e;
    // The tiles to store the data in shared memory for the filters.
    using Smem_tile_e = typename Gemm_traits::Smem_tile_e;
    // The fragment.
    using Fragment_e = typename Smem_tile_e::Fragment;

    // The tile in global memory.
    using Gmem_tile_epilogue = typename Gemm_traits::Gmem_tile_epilogue;
    // The tile in shared memory to swizzle the output.
    using Swizzle_epilogue = typename Gemm_traits::Swizzle_epilogue;
    // The callbacks.
    using Callbacks_epilogue = typename Gemm_traits::Callbacks_epilogue;
    // The epilogue.
    using Epilogue = typename Gemm_traits::Epilogue;

    // The shared memory pointers.
    char *a_smem_ = &smem_[0];
    char *b_smem_ = &smem_[Smem_tile_a::BYTES_PER_TILE];
    char *e_smem_ = &smem_[Smem_tile_a::BYTES_PER_TILE + Smem_tile_b::BYTES_PER_TILE];

    // The tiles in shared memory.
    Smem_tile_a a_smem(a_smem_, tidx);
    Smem_tile_b b_smem(b_smem_, tidx);
    Smem_tile_e e_smem(e_smem_, tidx);

    // Create the objects to load from global memory.
    Gmem_tile_a a_gmem(params, bidm, bidz, tidx);   
    Gmem_tile_b b_gmem(params, bidn, bidz, tidx);
    Gmem_tile_e e_gmem(params, bidm, bidz, tidx);

    // Clear the aclwmulators. 
    xmma::Fragment_aclwmulator<Traits> acc[XMMAS_M][XMMAS_N];
    xmma::helpers::clear(acc);

    // Pipeline data load in prolog.
    const int PRELOAD_STAGE = xmma::Max<Gemm_traits::STAGES - 1, 1>::VALUE;
    int real_preload_stage = min(PRELOAD_STAGE, params.loop_start + 1);

    // Load last one k-block
    int last_load_idx = params.loop_start;
    int64_t last_load_delta_a = last_load_idx * params.a_delta;
    int64_t last_load_delta_b = last_load_idx * params.b_delta;
    int64_t last_load_delta_e = last_load_idx * params.e_delta;

    a_gmem.move(last_load_delta_a);
    b_gmem.move(last_load_delta_b);
    e_gmem.move(last_load_delta_e);

    a_gmem.residue();
    b_gmem.residue();
    e_gmem.residue();

    a_gmem.load(a_smem, mem_desc_a);
    b_gmem.load(b_smem, mem_desc_b);
    e_gmem.load(e_smem);

    xmma::ldgdepbar<true>();

    // Store the matrices to shared memory.
    a_gmem.commit(a_smem);
    b_gmem.commit(b_smem);
    e_gmem.commit(e_smem); // Apply LDGSTS, so no need to commit

    // Move to next SMEM buffer, if we have multistage or double buffer.
    a_smem.move_next_write_buffer();
    b_smem.move_next_write_buffer();
    e_smem.move_next_write_buffer();

    // PreLoad rest k block
    a_gmem.move((-1) * last_load_delta_a);
    b_gmem.move((-1) * last_load_delta_b);
    e_gmem.move((-1) * last_load_delta_e);

    a_gmem.restore_predicate();
    b_gmem.restore_predicate();
    e_gmem.restore_predicate();

    #pragma unroll
    for( int stage = 0 ; stage < real_preload_stage - 1 ; stage ++ ) {

        a_gmem.load(a_smem, mem_desc_a);
        b_gmem.load(b_smem, mem_desc_b);
        e_gmem.load(e_smem);
        
        xmma::ldgdepbar<true>();

        // Store the matrices to shared memory.
        a_gmem.commit(a_smem);
        b_gmem.commit(b_smem);
        e_gmem.commit(e_smem); // Apply LDGSTS, so no need to commit

        // Move to next SMEM buffer, if we have multistage or double buffer.
        a_smem.move_next_write_buffer();
        b_smem.move_next_write_buffer();
        e_smem.move_next_write_buffer();

        // Move the pointers and assemble the predicates for the next loop.
        a_gmem.move(params.a_delta);
        b_gmem.move(params.b_delta);
        e_gmem.move(params.e_delta);

    } // end for stage

    // Make sure the data is in shared memory.
#if defined(__LWDA_ARCH__) && __LWDA_ARCH__ >= 800
    int missed_ldgdepbar_stage = PRELOAD_STAGE - real_preload_stage;
    for(int i = 0; i < missed_ldgdepbar_stage ; ++i){
        xmma::ldgdepbar<true>();
    }
    xmma::depbar<true, Gemm_traits::STAGES>();
#endif
    __syncthreads();

    // Load the A matrix.
    Fragment_a a[2][XMMAS_M];
    a_smem.load(a[0], 0);

    // Load the B matrix.
    // Fragment_b b[2][XMMAS_N];
    // b_smem.load(b[0], 0);
    Fragment_b b[XMMAS_N][1];
    #pragma unroll
    for(int load_pipe = 0 ; load_pipe < XMMAS_N ;  load_pipe ++){
        b_smem.pipe_load(b[load_pipe], 0, load_pipe);
    }

    // Load the Meta matrix.
    Fragment_e e[2][1];
    e_smem.load(e[0], 0);

    // Enable fencing for sparse GEMM.
    constexpr bool JETFIRE_FENCING_ENABLED = true;

#ifdef XMMA_ENABLED_PMTRIG
    __prof_trigger(15);
#endif
  
    // Iterate over the loop.
    JETFIRE_MAC_LOOP_PRAGMA     // Jetfire loop body
    #pragma unroll 1
    for( int loop = params.loop_start; loop >= 0; --loop ) {
        JETFIRE_MAC_LOOP_HEADER
        #if JETFIRE_ENABLED
            asm volatile (".pragma \"set knob SchedMemNoAlias=LDS+LDSM+LDGSTS\";\n" : : : "memory");
        #endif

        // Disable the loads in the last iteration.
        const int is_last = (loop < xmma::Max<Gemm_traits::STAGES - 1, 1>::VALUE);

        if( is_last ) {
            a_gmem.disable_loads();
            b_gmem.disable_loads();
            e_gmem.disable_loads();
        }

        #pragma unroll
        for( int KI = 1; KI <= XMMAS_K; ++KI ) {

            int ki = (KI == XMMAS_K) ? 0 : KI;

            // Trigger the commit of global loads in last iteration
            if (KI == XMMAS_K)
            {
                // Make sure the data is in shared memory.
                xmma::depbar<true, Gemm_traits::STAGES>();

                __syncthreads();

                // Move to next SMEM buffer, if we have multistage or double buffer.
                a_smem.move_next_write_buffer();
                b_smem.move_next_write_buffer();
                e_smem.move_next_write_buffer();

                // Move the shared memory pointers for double buffering.
                a_smem.move_next_read_buffer();
                b_smem.move_next_read_buffer();
                e_smem.move_next_read_buffer();
            }

            // Load the matrices from shared memory.
            a_smem.load(a[ki&1], ki); 
            //b_smem.load(b[ki&1], ki);
            e_smem.load(e[ki&1], ki);

            jetfire::ifence(JETFIRE_FENCING_ENABLED);  // Disbale original ifence

            // Trigger the global loads on the 1st iteration of that core loop.
            if( KI == 1 ) {
                a_gmem.load(a_smem, mem_desc_a);
                b_gmem.load(b_smem, mem_desc_b);
                e_gmem.load(e_smem);
                xmma::ldgdepbar<true>();
            }

            // Warp context switch halfway through
            if (KI - 1 == XMMAS_K / 2)
                jetfire::warp_switch();
            
            //jetfire::ifence(JETFIRE_FENCING_ENABLED);
            // Gemm_traits::spgemm(acc, a[(ki-1)&1], b[(ki-1)&1], e[(ki-1)&1]);
            #pragma unroll
            for(int pipe = 0 ; pipe < XMMAS_N ;  pipe ++){
                Gemm_traits::spgemm_pipeline(acc, a[(ki-1)&1], b[pipe], e[(ki-1)&1], pipe);
                b_smem.pipe_load(b[pipe], ki, pipe);
                jetfire::ifence(JETFIRE_FENCING_ENABLED);  // Disbale original ifence
            }

            b_smem.tf32_pipe_colwert(b);
        } // (ki)

        // Move the global pointers.
        a_gmem.move(params.a_delta);
        b_gmem.move(params.b_delta);
        e_gmem.move(params.e_delta);
        #if JETFIRE_ENABLED
            asm volatile (".pragma \"reset knob SchedMemNoAlias=LDS+LDSM+LDGSTS\";\n" : : : "memory");
        #endif
    } // (loop)

#ifdef XMMA_ENABLED_PMTRIG
    __prof_trigger(15);
#endif

    // Do allocate the tile to output in the epilogue. 
    Gmem_tile_epilogue gmem_epilogue(params, bidm, bidn, bidz, tidx);

    // Do allocate the tile and compute the offsets. TODO: Merge with the epilogue class!
    Swizzle_epilogue swizzle_epilogue(smem_, tidx);
    
    // The callbacks.
    Callbacks_epilogue callbacks_epilogue(params, smem_, bidm, bidn, bidz, tidx);

    // Do the epilogue.
    Epilogue epilogue(params, gmem_epilogue, swizzle_epilogue, callbacks_epilogue);
    //epilogue.execute<params.has_beta>(acc);

    if( params.has_beta ) {
        epilogue.template execute<true>(acc);
    } else {
        epilogue.template execute<false>(acc);
    }

    // Make sure we can use the shared memory.
    __syncthreads();

    // Finalize the callbacks.
    callbacks_epilogue.post_epilogue();

}

///////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Gemm_traits >
static __global__ __launch_bounds__(Gemm_traits::Cta_tile::THREADS_PER_CTA)
void split_k_kernel(typename Gemm_traits::Params params) {

    // The traits class.
    using Traits = typename Gemm_traits::Traits;
    // The CTA tile.
    using Cta_tile = typename Gemm_traits::Cta_tile;
    // The XMMA tile.
    using Xmma_tile = typename Gemm_traits::Xmma_tile;
    
    // The block indices.
    int bidm, bidn, bidz;
    if( params.use_horizontal_cta_rasterization ) {
        bidm = blockIdx.y;
        bidn = blockIdx.x;
    } else {
        bidm = blockIdx.x;
        bidn = blockIdx.y;
    }
    bidz = blockIdx.z;
    // The thread index.
    const int tidx = threadIdx.x;

    // The output tile for the epilogue.
    using Gmem_tile_epilogue = typename Gemm_traits::Gmem_tile_epilogue;
    // The shared memory swizzle.
    using Swizzle_epilogue = typename Gemm_traits::Swizzle_epilogue;
    // The callbacks.
    using Callbacks_epilogue = typename Gemm_traits::Callbacks_epilogue;
    // The epilogue.
    using Epilogue = typename Gemm_traits::Epilogue;

    // Do allocate the tile to output in the epilogue. 
    Gmem_tile_epilogue gmem_epilogue(params, bidm, bidn, bidz, tidx);
    // Do allocate the tile and compute the offsets. TODO: Merge with the epilogue class!
    Swizzle_epilogue swizzle_epilogue(smem_, tidx);
    // The callbacks.
    Callbacks_epilogue callbacks(params, smem_, bidm, bidn, 0, tidx);

    enum { CONTIGUOUS = Gmem_tile_epilogue::output_layout::COL
        ? Xmma_tile::XMMAS_M : Xmma_tile::XMMAS_N };

    // Do the epilogue.
    Epilogue epilogue(params, gmem_epilogue, swizzle_epilogue, callbacks);
    epilogue.template exelwte_split_k<CONTIGUOUS>();
    
}

////////////////////////////////////////////////////////////////////////////////////////////////////
} // namespace sparse
} // namespace gemm
} // namespace ext
} // namespace xmma
