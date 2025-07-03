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
namespace implicit_gemm {
namespace interleaved_fprop {

////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CTA_SKEW_ENABLED
#define CTA_SKEW_ENABLED 0
#endif

#if defined( __LWDA_ARCH__ ) && __LWDA_ARCH__ == 700
// Enable for GV100 only
#define SM_NUM 80
// Tuned for GV100
#define SKEW_CYCLES 6000
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

#if defined( CTA_SKEW_ENABLED ) && CTA_SKEW_ENABLED
template <typename Kernel_traits>
static inline __device__ void apply_cta_skew( const typename Kernel_traits::Params& params ) {
#if defined( __LWDA_ARCH__ ) && __LWDA_ARCH__ == 700
    // Insert GV100 intra-SM CTA skew code snippet
    if( Kernel_traits::Cta_tile::GROUPS > 1 ) {
        return;
    }

    // Olwpancy: 2cta/sm
    if( Kernel_traits::Cta_tile::THREADS_PER_CTA == 128 ) {
        if( threadIdx.x == 0 ) {
            int ctanum = gridDim.z * gridDim.y * gridDim.x;
            // > 1 wave
            if( ctanum > 2 * SM_NUM ) {
                unsigned int nsmid;
                asm( "mov.u32 %0, %%nsmid;" : "=r"( nsmid ) );
                // only enable on GV100.
                if( nsmid == SM_NUM ) {
                    int ctaid =
                        blockIdx.z * gridDim.y * gridDim.x + blockIdx.y * gridDim.x + blockIdx.x;
                    unsigned int start_time = clock();
                    unsigned int end_time, wait_time = 0;
                    // volta intra-sm in first wave
                    if( ctaid >= SM_NUM && ctaid < 2 * SM_NUM ) {
                        while( wait_time < SKEW_CYCLES ) {
                            end_time = clock();
                            wait_time = end_time - start_time;
                        }
                    }
                }
            }
        }
        __syncthreads();

        // Olwpancy: 4cta/sm
    } else if( Kernel_traits::Cta_tile::THREADS_PER_CTA == 64 ) {
        if( threadIdx.x == 0 ) {
            int ctanum = gridDim.z * gridDim.y * gridDim.x;
            // > 1 wave
            if( ctanum > 4 * SM_NUM ) {
                unsigned int nsmid;
                asm( "mov.u32 %0, %%nsmid;" : "=r"( nsmid ) );
                // only enable on GV100.
                if( nsmid == SM_NUM ) {
                    int ctaid =
                        blockIdx.z * gridDim.y * gridDim.x + blockIdx.y * gridDim.x + blockIdx.x;
                    unsigned int start_time = clock();
                    unsigned int end_time, wait_time = 0;
                    // volta intra-sm in 1st wave
                    if( ctaid >= 2 * SM_NUM && ctaid < 4 * SM_NUM ) {
                        while( wait_time < SKEW_CYCLES ) {
                            end_time = clock();
                            wait_time = end_time - start_time;
                        }
                    }
                }
            }
        }
        __syncthreads();
    }
#else
#endif  // defined(__LWDA_ARCH__) && __LWDA_ARCH__ == 700
}
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline __device__ char* align_128(char *ptr) {
    uint64_t address_bit = reinterpret_cast<uint64_t>(ptr);
    uint64_t offset = address_bit % 128;
    if(offset == 0) {
        return ptr;
    } else {
        return ptr + (128 - offset);
    }
}


////////////////////////////////////////////////////////////////////////////////////////////////////
// Single-stage kernel
template< typename Kernel_traits, typename Params, bool = (Kernel_traits::STAGES > 1) >
struct Matmul {
    static inline __device__ void run(const Params &params,
                              typename Kernel_traits::Compute_tile &compute_tile,
                              typename Kernel_traits::Tile_distribution &tile) {
        // The traits class.
        using Traits = typename Kernel_traits::Traits;
        // The CTA tile.
        using Cta_tile = typename Kernel_traits::Cta_tile;
        // The XMMA tile.
        using Xmma_tile = typename Kernel_traits::Xmma_tile;

        const int XMMAS_K = Xmma_tile::XMMAS_K;

        // Apply CTA skew on Volta, if enabled.
#if defined( CTA_SKEW_ENABLED ) && CTA_SKEW_ENABLED
        apply_cta_skew<Kernel_traits>( params );
#endif

        // The block/tile indices.
        int bidm = tile.bidm();
        int bidn = tile.bidn();
        int bidz = tile.bidz();

        // The thread index.
        const int tidx = threadIdx.x;

        // Ampere memory descritors.
        const uint64_t mem_desc_a = params.mem_descriptors.descriptor_a;
        const uint64_t mem_desc_b = params.mem_descriptors.descriptor_b;

        // The tiles in global memory for the images.
        using Gmem_tile_a = typename Kernel_traits::Gmem_tile_a;
        // The tiles to store the data in shared memory for the images.
        using Smem_tile_a = typename Kernel_traits::Smem_tile_a;

        // The tiles in global memory for the filters.
        using Gmem_tile_b = typename Kernel_traits::Gmem_tile_b;
        // The tiles to store the data in shared memory for the filters.
        using Smem_tile_b = typename Kernel_traits::Smem_tile_b;

        // The shared memory. It is allocated at launch time.
        extern __shared__ char smem_[];

        // The shared memory pointers.
        char* a_smem_ = &smem_[0];
        char* b_smem_ = &smem_[Smem_tile_a::BYTES_PER_TILE];

        // The tiles in shared memory.
        Smem_tile_a a_smem( a_smem_, tidx );
        Smem_tile_b b_smem( b_smem_, tidx );

        // The extra shared memory buffers that could be needed by the kernels.
        char* a_extra_smem_ = &b_smem_[Smem_tile_b::BYTES_PER_TILE];
        char* b_extra_smem_ = &a_extra_smem_[Gmem_tile_a::BYTES_PER_EXTRA_SMEM];

        // Create the objects to load from global memory.
        Gmem_tile_a a_gmem( params, a_extra_smem_, tile.bidx(), tidx );
        Gmem_tile_b b_gmem( params, b_extra_smem_, tile.bidx(), tidx );

        // Only enable fencing if it looks like our desired stage overlap will fit in register budget.
        constexpr bool JETFIRE_FENCING_ENABLED = (255 * 4 > 
                sizeof(compute_tile) + sizeof(a_gmem) + sizeof(b_gmem));
    
        // The filter index.
        int trsi = Kernel_traits::initialize_filter_position( params );

        // Do we use LDGSTS in that kernel?
        enum { USE_LDGSTS = Gmem_tile_a::USE_LDGSTS || Gmem_tile_b::USE_LDGSTS };

        // for group colw , sync here since sme ctor use STS
        if (Cta_tile::GROUPS > 1) __syncthreads();
        //
        // S T A R T   T H E   P I P E L I N E
        //
        // The number of stages to prefetch to start the pipeline.
        enum { PREFETCH_STAGES = xmma::Max<Kernel_traits::STAGES - 1, 1>::VALUE };
        // The number of iterations of the loop.
        const int loop_count = params.loop_start + 1;
        // The iteration to enter residue.
        const int loop_to_enter_residue = params.loop_start - (params.loop_residue - PREFETCH_STAGES);
        // Initialize the prefetching pipeline.
        for( int ii = 0, prefetch = min( loop_count, PREFETCH_STAGES ); ii < prefetch; ++ii ) {

            // Trigger the loads for A and B. Either LDG or LDGSTS.
            a_gmem.load( a_smem, mem_desc_a );
            b_gmem.load( b_smem, mem_desc_b );

            // Make sure we insert the corresponding LDGDEPBAR. NOP on Volta/Turing.
            xmma::ldgdepbar<USE_LDGSTS>();

            // Store the pixels and filters to shared memory.
            a_gmem.commit( a_smem );
            b_gmem.commit( b_smem );

            // Load the deltas and update the filter position.
            int64_t a_delta, b_delta;
            trsi =
                Kernel_traits::load_deltas_and_move_filter_position( a_delta, b_delta, params, trsi );

            // Move to next SMEM buffer for multistage or double buffer.
            a_smem.move_next_write_buffer();
            b_smem.move_next_write_buffer();

            // Move the pointers and assemble the predicates for the next loop.
            a_gmem.move( trsi, a_delta );
            b_gmem.move( trsi, b_delta );

            // Trigger the residue if the next iteration of the prefetch loop is the one to enter residue.
            if( ii == loop_to_enter_residue) {
                a_gmem.residue();
                b_gmem.residue();
            }
        }

        // The # of LDGDEPBARs must be equal to the number of stages. Issue "extra" ones if needed.
#if !defined(__LWDACC_RTC__)
        #pragma unroll 1
#endif
        for( int ii = loop_count; ii < PREFETCH_STAGES; ++ii ) {
            xmma::ldgdepbar<USE_LDGSTS>();
        }

        // Make sure the data is in shared memory.
        xmma::depbar<USE_LDGSTS, Kernel_traits::STAGES>();
        __syncthreads();

        // // DEBUG.
        // a_smem.debug_print();
        // b_smem.debug_print();
        // // END OF DEBUG.

        // Load the image pixels / filters.
        if( XMMAS_K > 1 ) {
            compute_tile.load(a_smem, b_smem, 0, true);
        }

#ifdef XMMA_ENABLED_PMTRIG
        __prof_trigger(15);
#endif

        // Iterate over the loop.
        JETFIRE_MAC_LOOP_PRAGMA  // Jetfire loop body
#if !defined(__LWDACC_RTC__)
        #pragma unroll 1
#endif
        for( int loop = params.loop_start; loop >= 0; --loop ) {
            JETFIRE_MAC_LOOP_HEADER

// The core part of the loop.
#pragma unroll
            for( int ki_next = 1; ki_next <= XMMAS_K; ++ki_next ) {

                // The ki_next variable is "one-step ahead". The ki points to the current step.
                int ki = ( ki_next == XMMAS_K ) ? 0 : ki_next;

                // Interference fence at top of stage
                jetfire::ifence(JETFIRE_FENCING_ENABLED);


                // Trigger the commit of global loads in last iteration
                if( XMMAS_K > 1 && ki_next == XMMAS_K ) {

                    // Make sure the data was read from shared memory.
                    if( Smem_tile_a::BUFFERS_PER_TILE == 1 || Smem_tile_b::BUFFERS_PER_TILE == 1 ) {
                        __syncthreads();
                    }

                    // Store the data to shared memory for A.
                    a_gmem.commit( a_smem );
                    jetfire::ifence(JETFIRE_FENCING_ENABLED);

                    // Store the data to shared memory for B.
                    b_gmem.commit( b_smem );
                    jetfire::ifence(JETFIRE_FENCING_ENABLED);

                    // Make sure the data is in shared memory and synchronize threads.
                    xmma::depbar<USE_LDGSTS, Kernel_traits::STAGES>();
                    __syncthreads();

                    // Move to next SMEM buffer, if we have multistage or double buffer.
                    a_smem.move_next_write_buffer();
                    b_smem.move_next_write_buffer();

                    // Move the shared memory pointers for double buffering.
                    a_smem.move_next_read_buffer();
                    b_smem.move_next_read_buffer();
                }
                // Load the matrices from shared memory.
                compute_tile.load(a_smem, b_smem, ki);

                // Inteference fence after smem loads.
                jetfire::ifence(JETFIRE_FENCING_ENABLED);

                // Trigger the global loads on the 1st iteration of that core loop.
                if( ki_next == 1 ) {

                    // Disable the loads for the last stages of the pipeline.
                    if( loop < PREFETCH_STAGES ) {
                        a_gmem.disable_loads();
                        b_gmem.disable_loads();
                    }

                    // Trigger the loads for the A matrix.
                    a_gmem.load( a_smem, mem_desc_a );
                    jetfire::ifence(JETFIRE_FENCING_ENABLED);

                    // Trigger the loads for the B matrix.
                    b_gmem.load( b_smem, mem_desc_b );

                    // Push the LDGDEPBAR instruction after the loads for A and B.
                    xmma::ldgdepbar<USE_LDGSTS>();
                    jetfire::ifence(JETFIRE_FENCING_ENABLED);
                }

                // Warp context switch halfway through
                if( ki_next - 1 == XMMAS_K / 2 ) {
                    jetfire::warp_switch();
                }

                // Trigger the commit of global loads in last iteration
                if( XMMAS_K == 1 && ki_next == XMMAS_K ) {

                    // Make sure the data was read from shared memory.
                    if( Smem_tile_a::BUFFERS_PER_TILE == 1 || Smem_tile_b::BUFFERS_PER_TILE == 1 ) {
                        __syncthreads();
                    }

                    // Store the data to shared memory for A.
                    a_gmem.commit( a_smem );
                    jetfire::ifence(JETFIRE_FENCING_ENABLED);

                    // Store the data to shared memory for B.
                    b_gmem.commit( b_smem );
                    jetfire::ifence(JETFIRE_FENCING_ENABLED);

                    // Make sure the data is in shared memory and synchronize threads.
                    xmma::depbar<USE_LDGSTS, Kernel_traits::STAGES>();
                    __syncthreads();

                    // Move to next SMEM buffer, if we have multistage or double buffer.
                    a_smem.move_next_write_buffer();
                    b_smem.move_next_write_buffer();

                    // Move the shared memory pointers for double buffering.
                    a_smem.move_next_read_buffer();
                    b_smem.move_next_read_buffer();
                }

                // Do the math - The core of the loop does 16x16x8.
                compute_tile.compute(ki_next);

            }  // (ki/ki_next)

            // Load the deltas and update the filter position.
            int64_t a_delta, b_delta;
            trsi =
                Kernel_traits::load_deltas_and_move_filter_position( a_delta, b_delta, params, trsi );

            // Move the global pointers.
            a_gmem.move( trsi, a_delta );
            b_gmem.move( trsi, b_delta );

            // Execute the residue code. Clear the masks for the image if needed.
            if( loop <= params.loop_residue ) {
                a_gmem.residue();
                b_gmem.residue();
            }

        }  // (loop)

#ifdef XMMA_ENABLED_PMTRIG
        __prof_trigger(15);
#endif

        // The tile in global memory.
        using Gmem_tile_epilogue = typename Kernel_traits::Gmem_tile_epilogue;
        // The tile in shared memory to swizzle the output.
        using Swizzle_epilogue = typename Kernel_traits::Swizzle_epilogue;

        // The callbacks for ReLU lowerbound != 0.
        using Callbacks_epilogue_lb_nonzero = typename Kernel_traits::Callbacks_epilogue_lb_nonzero;

        // The epilogue for ReLU lowerbound != 0.
        using Epilogue_lb_nonzero = typename Kernel_traits::Epilogue_lb_nonzero;

        // Do allocate the tile to output in the epilogue.
        Gmem_tile_epilogue gmem_epilogue( params, bidm, bidn, bidz, tidx );
        // Do allocate the tile and compute the offsets. TODO: Merge with the epilogue class!
        Swizzle_epilogue swizzle_epilogue( smem_, tidx );

        // The callbacks.
        Callbacks_epilogue_lb_nonzero callbacks_epilogue( params, smem_, bidm, bidn, bidz, tidx );

        // Make sure we can use the shared memory.
        __syncthreads();

        // Do the epilogue.
        Epilogue_lb_nonzero epilogue( params, gmem_epilogue, swizzle_epilogue, callbacks_epilogue );
        if( params.with_residual ) {
            epilogue.template execute<true>( compute_tile.acc_ );
        } else {
            epilogue.template execute<false>( compute_tile.acc_ );
        }

        // Make sure we can use the shared memory.
        __syncthreads();

        // Finalize the callbacks.
        callbacks_epilogue.post_epilogue();

    }
};

// Multi-stage kernel
template< typename Kernel_traits, typename Params >
struct Matmul< Kernel_traits, Params, true > {

    // The tile in global memory.
    using Gmem_tile_epilogue = typename Kernel_traits::Gmem_tile_epilogue;
    // The tile in shared memory to swizzle the output.
    using Swizzle_epilogue = typename Kernel_traits::Swizzle_epilogue;

    // The callbacks for ReLU lowerbound != 0.
    using Callbacks_epilogue_lb_nonzero = typename Kernel_traits::Callbacks_epilogue_lb_nonzero;
    // The callbacks for ReLU lowerbound == 0.
    using Callbacks_epilogue_lb_zero = typename Kernel_traits::Callbacks_epilogue_lb_zero;

    // The epilogue for ReLU lowerbound != 0.
    using Epilogue_lb_nonzero = typename Kernel_traits::Epilogue_lb_nonzero;
    // The epilogue for ReLU lowerbound == 0.
    using Epilogue_lb_zero = typename Kernel_traits::Epilogue_lb_zero;

    static inline __device__ void run(const Params &params,
                                      typename Kernel_traits::Compute_tile &compute_tile,
                                      typename Kernel_traits::Tile_distribution &tile) {
        // The traits class.
        using Traits = typename Kernel_traits::Traits;
        // The CTA tile.
        using Cta_tile = typename Kernel_traits::Cta_tile;
        // The XMMA tile.
        using Xmma_tile = typename Kernel_traits::Xmma_tile;

        const int XMMAS_K = Xmma_tile::XMMAS_K;

        // Apply CTA skew on Volta, if enabled.
#if defined( CTA_SKEW_ENABLED ) && CTA_SKEW_ENABLED
        apply_cta_skew<Kernel_traits>( params );
#endif

        // The block/tile indices.
        int bidm = tile.bidm();
        int bidn = tile.bidn();
        int bidz = tile.bidz();

        // The thread index.
        const int tidx = threadIdx.x;

        // Ampere memory descritors.
        const uint64_t mem_desc_a = params.mem_descriptors.descriptor_a;
        const uint64_t mem_desc_b = params.mem_descriptors.descriptor_b;

        // The tiles in global memory for the images.
        using Gmem_tile_a = typename Kernel_traits::Gmem_tile_a;
        // The tiles to store the data in shared memory for the images.
        using Smem_tile_a = typename Kernel_traits::Smem_tile_a;

        // The tiles in global memory for the filters.
        using Gmem_tile_b = typename Kernel_traits::Gmem_tile_b;
        // The tiles to store the data in shared memory for the filters.
        using Smem_tile_b = typename Kernel_traits::Smem_tile_b;

        // The shared memory. It is allocated at launch time.
        extern __shared__ char smem_[];

        // The shared memory pointers.
        char* a_smem_ = &smem_[0];
        char* b_smem_ = &smem_[Smem_tile_a::BYTES_PER_TILE];

        // The tiles in shared memory.
        Smem_tile_a a_smem( a_smem_, tidx );
        Smem_tile_b b_smem( b_smem_, tidx );

        // The extra shared memory buffers that could be needed by the kernels.
        char* a_extra_smem_ = &b_smem_[Smem_tile_b::BYTES_PER_TILE];
        char* b_extra_smem_ = &a_extra_smem_[Gmem_tile_a::BYTES_PER_EXTRA_SMEM];

        // Create the objects to load from global memory.
        Gmem_tile_a a_gmem( params, a_extra_smem_, tile.bidx(), tidx );
        Gmem_tile_b b_gmem( params, b_extra_smem_, tile.bidx(), tidx );

        // Only enable fencing if it looks like our desired stage overlap will fit in register budget.
        constexpr bool JETFIRE_FENCING_ENABLED = (255 * 4 > 
                sizeof(compute_tile) + sizeof(a_gmem) + sizeof(b_gmem));
    
        // The filter index.
        int trsi = Kernel_traits::initialize_filter_position( params );

        // Do we use LDGSTS in that kernel?
        enum { USE_LDGSTS = Gmem_tile_a::USE_LDGSTS || Gmem_tile_b::USE_LDGSTS };
        
        // for group colw , sync here since some ctor use STS
        if (Cta_tile::GROUPS > 1) __syncthreads();

        //
        // S T A R T   T H E   P I P E L I N E
        //

        // The number of stages to prefetch to start the pipeline.
        enum { PREFETCH_STAGES = xmma::Max<Kernel_traits::STAGES - 1, 1>::VALUE };
        // The number of iterations of the loop.
        const int loop_count = params.loop_start + 1;
        // The iteration to enter residue.
        const int loop_to_enter_residue = params.loop_start - (params.loop_residue - PREFETCH_STAGES);
        // Initialize the prefetching pipeline.
        for( int ii = 0, prefetch = min( loop_count, PREFETCH_STAGES ); ii < prefetch; ++ii ) {

            // Trigger the loads for A and B. Either LDG or LDGSTS.
            a_gmem.load( a_smem, mem_desc_a );
            b_gmem.load( b_smem, mem_desc_b );

            // Make sure we insert the corresponding LDGDEPBAR. NOP on Volta/Turing.
            xmma::ldgdepbar<USE_LDGSTS>();

            // Store the pixels and filters to shared memory.
            a_gmem.commit( a_smem );
            b_gmem.commit( b_smem );

            // Load the deltas and update the filter position.
            int64_t a_delta, b_delta;
            trsi =
                Kernel_traits::load_deltas_and_move_filter_position( a_delta, b_delta, params, trsi );

            // Move to next SMEM buffer for multistage or double buffer.
            a_smem.move_next_write_buffer();
            b_smem.move_next_write_buffer();

            // Move the pointers and assemble the predicates for the next loop.
            a_gmem.move( trsi, a_delta );
            b_gmem.move( trsi, b_delta );

            // Trigger the residue if the next iteration of the prefetch loop is the one to enter residue.
            if( ii == loop_to_enter_residue) {
                a_gmem.residue();
                b_gmem.residue();
            }
        }

        // The # of LDGDEPBARs must be equal to the number of stages. Issue "extra" ones if needed.
#if !defined(__LWDACC_RTC__)
        #pragma unroll 1
#endif
        for( int ii = loop_count; ii < PREFETCH_STAGES; ++ii ) {
            xmma::ldgdepbar<USE_LDGSTS>();
        }

        // WAR: prefetch bias/vectorized alpha/beta into the registers, just for small tile sizes!
        // Do allocate the tile to output in the epilogue.
        Gmem_tile_epilogue gmem_epilogue_small_tile( params, bidm, bidn, bidz, tidx );
        // Do allocate the tile and compute the offsets.
        Swizzle_epilogue swizzle_epilogue_small_tile( smem_, tidx );
        Callbacks_epilogue_lb_nonzero callbacks_epilogue_small_tile(
            params, smem_, bidm, bidn, bidz, tidx );
        Epilogue_lb_nonzero epilogue_small_tile( params,
                                                 gmem_epilogue_small_tile,
                                                 swizzle_epilogue_small_tile,
                                                 callbacks_epilogue_small_tile );

        // Make sure the data is in shared memory.
        xmma::depbar<USE_LDGSTS, Kernel_traits::STAGES>();
        __syncthreads();

        // Residual prefetch.
        using Gmem_tile_epilogue_prefetch = typename Kernel_traits::Gmem_tile_epilogue_prefetch;
        if (params.with_residual) {
            Gmem_tile_epilogue_prefetch prefetch(params, bidm, bidn, bidz, tidx);
            prefetch.prefetch();
        }

        // // DEBUG.
        // a_smem.debug_print();
        // b_smem.debug_print();
        // // END OF DEBUG.

        // Disable the loads for the last stages of the pipeline.
        if( loop_count <= PREFETCH_STAGES ) {
            a_gmem.disable_loads();
            b_gmem.disable_loads();
        }

        // Trigger the loads for A and B. Either LDG or LDGSTS.
        a_gmem.load( a_smem, mem_desc_a );
        a_gmem.commit( a_smem );

        // Load the deltas and update the filter position.
        int64_t a_delta, b_delta;
        trsi =
            Kernel_traits::load_deltas_and_move_filter_position( a_delta, b_delta, params, trsi );

        // Move to next SMEM buffer for multistage or double buffer.
        a_smem.move_next_write_buffer();

        // Move the pointers and assemble the predicates for the next loop.
        a_gmem.move( trsi, a_delta );

        // Load the image pixels / filters.
        if( XMMAS_K > 1 ) {
            compute_tile.load(a_smem, b_smem, 0, true);
        }

#ifdef XMMA_ENABLED_PMTRIG
        __prof_trigger(15);
#endif

        // Iterate over the loop.
        JETFIRE_MAC_LOOP_PRAGMA  // Jetfire loop body
#if !defined(__LWDACC_RTC__)
        #pragma unroll 1
#endif
        for( int loop = params.loop_start; loop >= 0; --loop ) {
            JETFIRE_MAC_LOOP_HEADER

#ifdef JETFIRE_ENABLED
            asm volatile (".pragma \"set knob SchedMemNoAlias=LDS+LDSM+LDGSTS\";\n" : : : "memory");
#endif

// The core part of the loop.
#pragma unroll
            for( int ki_next = 1; ki_next <= XMMAS_K; ++ki_next ) {

                // The ki_next variable is "one-step ahead". The ki points to the current step.
                int ki = ( ki_next == XMMAS_K ) ? 0 : ki_next;

                // Interference fence at top of stage
                jetfire::ifence(JETFIRE_FENCING_ENABLED);


                // Trigger the commit of global loads in last iteration
                if( XMMAS_K > 1 && ki_next == XMMAS_K ) {

                    if( loop <= params.loop_residue ) {
                        // Execute the residue code. Clear the masks for the image if needed.
                        a_gmem.residue();
                        b_gmem.residue();
                    }

                    // Disable the loads for the last stages of the pipeline.
                    if( loop <= PREFETCH_STAGES ) {
                        a_gmem.disable_loads();
                        b_gmem.disable_loads();
                    }

                    // Make sure the data was read from shared memory.
                    if( Smem_tile_a::BUFFERS_PER_TILE == 1 || Smem_tile_b::BUFFERS_PER_TILE == 1 ) {
                        __syncthreads();
                    }

                    // Store the data to shared memory for A.
                    a_gmem.commit( a_smem );
                    jetfire::ifence(JETFIRE_FENCING_ENABLED);

                    // Store the data to shared memory for B.
                    b_gmem.commit( b_smem );
                    jetfire::ifence(JETFIRE_FENCING_ENABLED);

                    // Make sure the data is in shared memory and synchronize threads.
                    xmma::depbar<USE_LDGSTS, Kernel_traits::STAGES>();
                    __syncthreads();

                    // Trigger the loads for the A matrix.
                    a_gmem.load( a_smem, mem_desc_a );

                    // Load the deltas.
                    trsi = Kernel_traits::load_deltas_and_move_filter_position( a_delta, b_delta, params, trsi );
                    a_gmem.move( trsi, a_delta );
                    jetfire::ifence(JETFIRE_FENCING_ENABLED);

                    // Move to next SMEM buffer, if we have multistage or double buffer.
                    a_smem.move_next_write_buffer();
                    b_smem.move_next_write_buffer();

                    // Move the shared memory pointers for double buffering.
                    a_smem.move_next_read_buffer();
                    b_smem.move_next_read_buffer();
                }

                // Load the matrices from shared memory.
                compute_tile.load(a_smem, b_smem, ki);

                // Inteference fence after smem loads.
                jetfire::ifence(JETFIRE_FENCING_ENABLED);

                // Trigger the global loads on the 1st iteration of that core loop.
                if( ki_next == 1 ) {

                    // Trigger the loads for the B matrix.
                    b_gmem.load( b_smem, mem_desc_b );
                    b_gmem.move( trsi, b_delta );
                    jetfire::ifence(JETFIRE_FENCING_ENABLED);


                    // Push the LDGDEPBAR instruction after the loads for A and B.
                    xmma::ldgdepbar<USE_LDGSTS>();
                    jetfire::ifence(JETFIRE_FENCING_ENABLED);
                }

                // Warp context switch halfway through
                if( ki_next - 1 == XMMAS_K / 2 ) {
                    jetfire::warp_switch();
                }

                // Trigger the commit of global loads in last iteration
                if( XMMAS_K == 1 && ki_next == XMMAS_K ) {

                    if( loop <= params.loop_residue ) {
                        // Execute the residue code. Clear the masks for the image if needed.
                        a_gmem.residue();
                        b_gmem.residue();
                    }

                    // Disable the loads for the last stages of the pipeline.
                    if( loop <= PREFETCH_STAGES ) {
                        a_gmem.disable_loads();
                        b_gmem.disable_loads();
                    }

                    // Make sure the data was read from shared memory.
                    if( Smem_tile_a::BUFFERS_PER_TILE == 1 || Smem_tile_b::BUFFERS_PER_TILE == 1 ) {
                        __syncthreads();
                    }

                    // Store the data to shared memory for A.
                    a_gmem.commit( a_smem );
                    jetfire::ifence(JETFIRE_FENCING_ENABLED);

                    // Store the data to shared memory for B.
                    b_gmem.commit( b_smem );
                    jetfire::ifence(JETFIRE_FENCING_ENABLED);

                    // Make sure the data is in shared memory and synchronize threads.
                    xmma::depbar<USE_LDGSTS, Kernel_traits::STAGES>();
                    __syncthreads();

                    // Trigger the loads for the A matrix.
                    a_gmem.load( a_smem, mem_desc_a );

                    // Load the deltas.
                    trsi = Kernel_traits::load_deltas_and_move_filter_position( a_delta, b_delta, params, trsi );
                    a_gmem.move( trsi, a_delta );
                    jetfire::ifence(JETFIRE_FENCING_ENABLED);

                    // Move to next SMEM buffer, if we have multistage or double buffer.
                    a_smem.move_next_write_buffer();
                    b_smem.move_next_write_buffer();

                    // Move the shared memory pointers for double buffering.
                    a_smem.move_next_read_buffer();
                    b_smem.move_next_read_buffer();
                }

                // Do the math - The core of the loop does 16x16x8.
                compute_tile.compute(ki_next);

            }  // (ki/ki_next)

#ifdef JETFIRE_ENABLED
            asm volatile (".pragma \"reset knob SchedMemNoAlias=LDS+LDSM+LDGSTS\";\n" : : : "memory");
#endif

        }  // (loop)

#ifdef XMMA_ENABLED_PMTRIG
        __prof_trigger(15);
#endif


        // Do allocate the tile to output in the epilogue.
        Gmem_tile_epilogue gmem_epilogue( params, bidm, bidn, bidz, tidx );
        // Do allocate the tile and compute the offsets. TODO: Merge with the epilogue class!
        Swizzle_epilogue swizzle_epilogue( smem_, tidx );
    #if defined( __LWDA_ARCH__ ) && __LWDA_ARCH__ > 800
        if ( params.relu_lb ) { // ReLU lowerbound != 0 case
            // The callbacks.
            Callbacks_epilogue_lb_nonzero callbacks_epilogue( params, smem_, bidm, bidn, bidz, tidx );

            // Make sure we can use the shared memory.
            xmma::depbar<USE_LDGSTS, 0>();
            __syncthreads();

            // Do the epilogue.
            Epilogue_lb_nonzero epilogue( params, gmem_epilogue, swizzle_epilogue, callbacks_epilogue );
            if( params.with_residual ) {
                epilogue.template execute<true>( compute_tile.acc_ );
            } else {
                epilogue.template execute<false>( compute_tile.acc_ );
            }

            // Make sure we can use the shared memory.
            __syncthreads();

            // Finalize the callbacks.
            callbacks_epilogue.post_epilogue();
        } else { // ReLU lowerbound == 0
            // The callbacks.
            Callbacks_epilogue_lb_zero callbacks_epilogue( params, smem_, bidm, bidn, bidz, tidx );

            // Make sure we can use the shared memory.
            xmma::depbar<USE_LDGSTS, 0>();
            __syncthreads();

            // Do the epilogue.
            Epilogue_lb_zero epilogue( params, gmem_epilogue, swizzle_epilogue, callbacks_epilogue );
            if( params.with_residual ) {
                epilogue.template execute<true>( compute_tile.acc_ );
            } else {
                epilogue.template execute<false>( compute_tile.acc_ );
            }

            // Make sure we can use the shared memory.
            __syncthreads();

            // Finalize the callbacks.
            callbacks_epilogue.post_epilogue();
        }
    #endif
    #if defined( __LWDA_ARCH__ ) && __LWDA_ARCH__ <= 800
        // The callbacks.
            Callbacks_epilogue_lb_nonzero callbacks_epilogue( params, smem_, bidm, bidn, bidz, tidx );

            // Make sure we can use the shared memory.
            __syncthreads();

            // Do the epilogue.
            Epilogue_lb_nonzero epilogue( params, gmem_epilogue, swizzle_epilogue, callbacks_epilogue );
            // WAR: We're instancing different structs for large tile sizes and small tile sizes.
            // Small tile sizes could benifit from the prefetch of bias and alpha/beta, while large
            // tile sizes cannot, due to register spill.
            if( ( Cta_tile::M / Cta_tile::WARPS_M >= 64 ) &&
                ( Cta_tile::N / Cta_tile::WARPS_N >= 64 ) ) {
                if( params.with_residual ) {
                    epilogue.template execute<true>( compute_tile.acc_ );
                } else {
                    epilogue.template execute<false>( compute_tile.acc_ );
                }
            } else {
                if( params.with_residual ) {
                    epilogue_small_tile.template execute<true>( compute_tile.acc_ );
                } else {
                    epilogue_small_tile.template execute<false>( compute_tile.acc_ );
                }
            }

            // Make sure we can use the shared memory.
            __syncthreads();

            // Finalize the callbacks.
            callbacks_epilogue.post_epilogue();

        #endif

    }
};

// TODO: Do we really need Grouped_acc or can we simply use Cta_tile::GROUPS > 1???

template< typename Kernel_traits >
static __global__ __launch_bounds__( Kernel_traits::Cta_tile::THREADS_PER_CTA )
void kernel( typename Kernel_traits::Params params ) {
    // The traits class.
    using Traits = typename Kernel_traits::Traits;
    // The XMMA tile.
    using Xmma_tile = typename Kernel_traits::Xmma_tile;
    // The tile distribution manager.
    using Tile_distribution = typename Kernel_traits::Tile_distribution;
    // The compute tile.
    using Compute_tile = typename Kernel_traits::Compute_tile;

    // Initialize the tile distribution.
    Tile_distribution tile(params, blockIdx);

    // Create the compute tile and clear the aclwmulators.
    Compute_tile compute_tile;
    compute_tile.clear();

    Matmul<Kernel_traits, typename Kernel_traits::Params>::run(params, compute_tile, tile);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

#if 0 // Comment split k kernel since interleaved IMMA doesn't call split k kernel
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

    // The shared memory. It is allocated at launch time.
    extern __shared__ char smem_[];

    // The output tile for the epilogue.
    using Gmem_tile_epilogue = typename Kernel_traits::Gmem_tile_epilogue;
    // The shared memory swizzle.
    using Swizzle_epilogue = typename Kernel_traits::Swizzle_epilogue;

    // The callbacks for ReLU lowerbound != 0.
    using Callbacks_epilogue_lb_nonzero = typename Kernel_traits::Callbacks_epilogue_lb_nonzero;
    // The callbacks for ReLU lowerbound == 0.
    using Callbacks_epilogue_lb_zero = typename Kernel_traits::Callbacks_epilogue_lb_zero;

    // The epilogue for ReLU lowerbound != 0.
    using Epilogue_lb_nonzero = typename Kernel_traits::Epilogue_lb_nonzero;
    // The epilogue for ReLU lowerbound == 0.
    using Epilogue_lb_zero = typename Kernel_traits::Epilogue_lb_zero;

    // Do allocate the tile to output in the epilogue.
    Gmem_tile_epilogue gmem_epilogue( params, bidm, bidn, bidz, tidx );
    // Do allocate the tile and compute the offsets. TODO: Merge with the epilogue class!
    Swizzle_epilogue swizzle_epilogue( smem_, tidx );
    
    if ( params.relu_lb ) { // ReLU lb != 0 case
        // The callbacks.
        Callbacks_epilogue_lb_nonzero callbacks( params, smem_, bidm, bidn, 0, tidx );

        // Do the epilogue.
        Epilogue_lb_nonzero epilogue( params, gmem_epilogue, swizzle_epilogue, callbacks );
        epilogue.template exelwte_split_k<Xmma_tile::XMMAS_N>();
    } else { // ReLU lb == 0 case
        // The callbacks.
        Callbacks_epilogue_lb_zero callbacks( params, smem_, bidm, bidn, 0, tidx );

        // Do the epilogue.
        Epilogue_lb_zero epilogue( params, gmem_epilogue, swizzle_epilogue, callbacks );
        epilogue.template exelwte_split_k<Xmma_tile::XMMAS_N>();
    }
}
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace interleaved_fprop
}  // namespace implicit_gemm
}  // namespace xmma

