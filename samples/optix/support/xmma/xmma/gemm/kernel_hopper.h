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
namespace gemm {

////////////////////////////////////////////////////////////////////////////////////////////////////
// Regular GMMA kernel
////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Kernel_traits>
static __global__  //__launch_bounds__( Kernel_traits::Cta_tile::THREADS_PER_CTA )
void kernel_gmma( typename Kernel_traits::Params params ) {

    // The traits class.
    using Traits = typename Kernel_traits::Traits;
    // The CTA tile.
    using Cta_tile = typename Kernel_traits::Cta_tile;
    // The XMMA tile.
    using Xmma_tile = typename Kernel_traits::Xmma_tile;
    // The tile distribution manager.
    using Tile_distribution = typename Kernel_traits::Tile_distribution;

    // The number of XMMAs.
    const int XMMAS_K = Xmma_tile::XMMAS_K;

// Apply CTA skew on Volta, if enabled.
#if defined( CTA_SKEW_ENABLED ) && CTA_SKEW_ENABLED
    apply_cta_skew<Kernel_traits>( params );
#endif

    // Initialize the tile distribution.
    Tile_distribution tile( params, blockIdx );

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

    // The compute tile.
    using Compute_tile = typename Kernel_traits::Compute_tile;

    // The shared memory. It is allocated at launch time.
    extern __shared__ char smem_[];

    // The shared memory pointers.
    char *a_smem_ = &smem_[0];

#ifdef USE_GMMA
    // for hopper gmma, we need to make sure the smem ptrs are 128B aligned
    a_smem_ = xmma::align_128( a_smem_ );
    char *b_smem_ = a_smem_ + Smem_tile_a::BYTES_PER_TILE;
#else
    char *b_smem_ = &smem_[Smem_tile_a::BYTES_PER_TILE];
#endif

    // The tiles in shared memory.
    Smem_tile_a a_smem( a_smem_, tidx );
    Smem_tile_b b_smem( b_smem_, tidx );

    // The extra shared memory buffers that could be needed by the kernels.
    char *a_extra_smem_ = &b_smem_[Smem_tile_b::BYTES_PER_TILE];
    char *b_extra_smem_ = &a_extra_smem_[Gmem_tile_a::BYTES_PER_EXTRA_SMEM];

    // Create the objects to load from global memory.
    Gmem_tile_a a_gmem( params, a_extra_smem_, tile.bidx(), tidx );
    Gmem_tile_b b_gmem( params, b_extra_smem_, tile.bidx(), tidx );

// Create the compute tile and clear the aclwmulators.
#ifdef USE_GMMA
    Compute_tile compute_tile( a_smem_, b_smem_ );
#else
    Compute_tile compute_tile;
#endif
    compute_tile.clear();

    // Only enable fencing if it looks like our desired stage overlap will fit in register budget.
    constexpr bool JETFIRE_FENCING_ENABLED =
        ( 255 * 4 > sizeof( compute_tile ) + sizeof( a_gmem ) + sizeof( b_gmem ) );

    // The filter index.
    int trsi = Kernel_traits::initialize_filter_position( params );

    // Do we use LDGSTS in that kernel?
    enum {
        USE_LDGSTS = Gmem_tile_a::USE_LDGSTS || Gmem_tile_b::USE_LDGSTS ||
                     Gmem_tile_a::USE_UTMALDG || Gmem_tile_b::USE_UTMALDG
    };

    //
    // S T A R T   T H E   P I P E L I N E
    //

    // The number of stages to prefetch to start the pipeline.
    enum { PREFETCH_STAGES = xmma::Max<Kernel_traits::STAGES - 1, 1>::VALUE };
    // The number of iterations of the loop.
    const int loop_count = params.loop_start + 1;
    // The iteration to enter residue.
    const int loop_to_enter_residue = params.loop_start - ( params.loop_residue - PREFETCH_STAGES );

    // Initialize the prefetching pipeline.
    #pragma unroll
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

        // Trigger the residue if the next iteration of the prefetch loop is the one to enter
        // residue.
        if( ii == loop_to_enter_residue ) {
            a_gmem.residue();
            b_gmem.residue();
        }
    }

// The # of LDGDEPBARs must be equal to the number of stages. Issue "extra" ones if needed.
#if !defined( __LWDACC_RTC__ )
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
    compute_tile.load( a_smem, b_smem, 0, true );

#ifdef USE_GMMA
    xmma::warpgroup_arrive();

    // issue some GMMA for the first kblock in prologue
    #pragma unroll
    for( int gmma_stage = 0; gmma_stage < Kernel_traits::GMMA_STAGES - 1; ++gmma_stage ) {
        compute_tile.compute( gmma_stage, true );
    }
    compute_tile.compute( Kernel_traits::GMMA_STAGES - 1, true );
    const int GMMA_REMAINING_STAGE = XMMAS_K - Kernel_traits::GMMA_STAGES;
#endif

#ifdef GMMA_PMTRIG
    // pmtrig
    asm volatile("pmevent 0;");
#endif
    // Iterate over the loop.
    JETFIRE_MAC_LOOP_PRAGMA  // Jetfire loop body
#if !defined( __LWDACC_RTC__ )
    #pragma unroll 1
#endif
        for( int loop = params.loop_start; loop >= 0; --loop ) {
        JETFIRE_MAC_LOOP_HEADER
        
        // OCG knob to disable 3 predicated off LDS. http://lwbugs/2549067
        asm volatile (".pragma \"set knob SchedMemNoAlias=LDS+LDSM+LDGSTS\";\n" : : : "memory");

        // The core part of the loop.
        #pragma unroll
        for( int ki_next = 1; ki_next <= XMMAS_K; ++ki_next ) {

            // The ki_next variable is "one-step ahead". The ki points to the current step.
            int ki = ( ki_next == XMMAS_K ) ? 0 : ki_next;

            // Interference fence at top of stage
            jetfire::ifence( JETFIRE_FENCING_ENABLED );

            // Trigger the commit of global loads in last iteration
            if( ki_next == XMMAS_K ) {

                // Make sure the data was read from shared memory.
                if( Smem_tile_a::BUFFERS_PER_TILE == 1 || Smem_tile_b::BUFFERS_PER_TILE == 1 ) {
                    __syncthreads();
                }

                // Store the data to shared memory for A.
                a_gmem.commit( a_smem );
                jetfire::ifence( JETFIRE_FENCING_ENABLED );

                // Store the data to shared memory for B.
                b_gmem.commit( b_smem );
                jetfire::ifence( JETFIRE_FENCING_ENABLED );

#ifndef USE_GMMA
                // for gmma this is done at a different location
                // Make sure the data is in shared memory and synchronize threads.
                xmma::depbar<USE_LDGSTS, Kernel_traits::STAGES>();
                __syncthreads();
#endif

                // Move to next SMEM buffer, if we have multistage or double buffer.
                a_smem.move_next_write_buffer();
                b_smem.move_next_write_buffer();

                // Move the shared memory pointers for double buffering.
                a_smem.move_next_read_buffer();
                b_smem.move_next_read_buffer();
            }

            // Load the matrices from shared memory.
            compute_tile.load( a_smem, b_smem, ki );

            // Inteference fence after smem loads.
            jetfire::ifence( JETFIRE_FENCING_ENABLED );

            // Trigger the global loads on the 1st iteration of that core loop.
            if( ki_next == 1 ) {

                // Disable the loads for the last stages of the pipeline.
                if( loop < PREFETCH_STAGES ) {
                    a_gmem.disable_loads();
                    b_gmem.disable_loads();
                }

                // Trigger the loads for the A matrix.
                a_gmem.load( a_smem, mem_desc_a );
                jetfire::ifence( JETFIRE_FENCING_ENABLED );

                // Trigger the loads for the B matrix.
                b_gmem.load( b_smem, mem_desc_b );

                // Push the LDGDEPBAR instruction after the loads for A and B.
                xmma::ldgdepbar<USE_LDGSTS>();
                jetfire::ifence( JETFIRE_FENCING_ENABLED );
            }

            // Warp context switch halfway through
            if( ki_next - 1 == XMMAS_K / 2 ) {
                jetfire::warp_switch();
            }

#ifdef USE_GMMA
            if( ki_next <= GMMA_REMAINING_STAGE ) {
                // issue remaining GMMAs for the current kblock
                int gmma_stage = ki_next + Kernel_traits::GMMA_STAGES - 1;
                if( ki_next == 1 ) {
                    xmma::warpgroup_arrive();
                }

                if( ki_next == GMMA_REMAINING_STAGE ) {
                    compute_tile.compute( gmma_stage, true, true );
                    // increment the entire desc group
                    compute_tile.increment_gmma_desc_group();
                } else {
                    compute_tile.compute( gmma_stage, false );
                }
            } else {  // (ki_next <= GMMA_REMAINING_STAGE)
                if( ki_next == ( GMMA_REMAINING_STAGE + 1 ) ) {
                    // finish the ldgsts issued by previous iteration
                    xmma::depbar<USE_LDGSTS, Kernel_traits::STAGES>();
                    // if there is only one warpgroup, syncthreads is not necessary.
                    if( Kernel_traits::SYNCTHREADS_NEEDED == true )
                        __syncthreads();
                    // for the first loop iter, the GMMAs issued by prologue is finished
                    // The math of those GMMAs in the prologue should hide the LDS latency of
                    // those remaining GMMAs of the kblock
                    xmma::warpgroup_wait<1>();
                    // compute_tile.update_gmma_desc(0, Kernel_traits::GMMA_STAGES);
                }
                // issue the GMMAs for the next kblock
                // if( loop != 0 ) {
                if( ki_next == ( GMMA_REMAINING_STAGE + 1 ) ) {
                    xmma::warpgroup_arrive();
                }
                int gmma_stage = ki_next - GMMA_REMAINING_STAGE - 1;
                if( gmma_stage == ( Kernel_traits::GMMA_STAGES - 1 ) ) {
                    compute_tile.compute( gmma_stage, true );
                    xmma::warpgroup_wait<Kernel_traits::GMMA_STAGES>();
                    // compute_tile.update_gmma_desc(Kernel_traits::GMMA_STAGES, XMMAS_K);
                } else {
                    compute_tile.compute( gmma_stage, true );
                }
                //}
            }
#else
            // Do the math - The core of the loop does 16x16x8.
            compute_tile.compute( ki_next );
#endif

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
        
        // OCG knob to disable 3 predicated off LDS. http://lwbugs/2549067
        asm volatile (".pragma \"reset knob SchedMemNoAlias=LDS+LDSM+LDGSTS\";\n" : : : "memory");

    }  // (loop)
#ifdef GMMA_PMTRIG
    // pmtrig
    asm volatile("pmevent 1;");
#endif

#ifdef USE_GMMA
    // all GMMAs must be finished
    xmma::warpgroup_wait<0>();
#endif

    // Epilogue
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

    // Make sure we can use the shared memory.
    xmma::depbar<USE_LDGSTS, 0>();
    __syncthreads();

    // Do the epilogue.
    Epilogue epilogue( params, gmem_epilogue, swizzle_epilogue, callbacks_epilogue );
    if( params.with_residual ) {
        epilogue.template execute<true>( compute_tile.acc_ );
    } else {
        epilogue.template execute<false>( compute_tile.acc_ );
    }

    // Make sure we can use the shared memory.
    __syncthreads();
    //
    // Finalize the callbacks.
    callbacks_epilogue.post_epilogue();
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// GMMA gemm kernel when one of the operands is coming from RF.
// This kernel can also be fused.
// need to think about integration with regular gmma gemm kernel when both operands are from SMEM.
////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Kernel_traits>
static __global__  //__launch_bounds__( Kernel_traits::Cta_tile::THREADS_PER_CTA )
    void
    kernel_gmma_arf( typename Kernel_traits::Params params ) {

    // The traits class.
    using Traits = typename Kernel_traits::Traits;
    // The CTA tile.
    using Cta_tile = typename Kernel_traits::Cta_tile;
    // The XMMA tile.
    using Xmma_tile = typename Kernel_traits::Xmma_tile;
    // The tile distribution manager.
    using Tile_distribution = typename Kernel_traits::Tile_distribution;

    // The number of XMMAs.
    const int XMMAS_K = Xmma_tile::XMMAS_K;

// Apply CTA skew on Volta, if enabled.
#if defined( CTA_SKEW_ENABLED ) && CTA_SKEW_ENABLED
    apply_cta_skew<Kernel_traits>( params );
#endif
    // Initialize the tile distribution.
    Tile_distribution tile( params, blockIdx );

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

    // The compute tile.
    using Compute_tile = typename Kernel_traits::Compute_tile;

    // The shared memory. It is allocated at launch time.
    extern __shared__ char smem_[];

    // The shared memory pointers.
    char *a_smem_ = &smem_[0];

#ifdef USE_GMMA
    // for hopper gmma, we need to make sure the smem ptrs are 128B aligned
    a_smem_ = xmma::align_128( a_smem_ );
    char *b_smem_ = a_smem_ + Smem_tile_a::BYTES_PER_TILE;
#else
    char *b_smem_ = &smem_[Smem_tile_a::BYTES_PER_TILE];
#endif

    // The tiles in shared memory.
    Smem_tile_a a_smem( a_smem_, tidx );
    // needed by fusion. non-fused kernel will do nothing.
    char *scale_bias_smem_ = b_smem_ + Smem_tile_b::BYTES_PER_TILE;
    a_smem.set_scale_bias_smem_ptr(scale_bias_smem_, tidx, params.k);

    Smem_tile_b b_smem( b_smem_, tidx );

    // The extra shared memory buffers that could be needed by the kernels.
    char *a_extra_smem_ = &b_smem_[Smem_tile_b::BYTES_PER_TILE];
    char *b_extra_smem_ = &a_extra_smem_[Gmem_tile_a::BYTES_PER_EXTRA_SMEM];

    // Create the objects to load from global memory.
    Gmem_tile_a a_gmem( params, a_extra_smem_, tile.bidx(), tidx );
    Gmem_tile_b b_gmem( params, b_extra_smem_, tile.bidx(), tidx );

// Create the compute tile and clear the aclwmulators.
#ifdef USE_GMMA
    Compute_tile compute_tile( a_smem_, b_smem_ );
#else
    Compute_tile compute_tile;
#endif
    compute_tile.clear();

    // Only enable fencing if it looks like our desired stage overlap will fit in register budget.
    constexpr bool JETFIRE_FENCING_ENABLED =
        ( 255 * 4 > sizeof( compute_tile ) + sizeof( a_gmem ) + sizeof( b_gmem ) );

    // The filter index.
    int trsi = Kernel_traits::initialize_filter_position( params );

    // Do we use LDGSTS in that kernel?
    enum { USE_LDGSTS = Gmem_tile_a::USE_LDGSTS || Gmem_tile_b::USE_LDGSTS };

    //
    // S T A R T   T H E   P I P E L I N E
    //

    // The number of stages to prefetch to start the pipeline.
    enum { PREFETCH_STAGES = xmma::Max<Kernel_traits::STAGES - 1, 1>::VALUE };
    // The number of iterations of the loop.
    const int loop_count = params.loop_start + 1;

    // The iteration to enter residue.
    const int loop_to_enter_residue = params.loop_start - ( params.loop_residue - PREFETCH_STAGES );
    // Initialize the prefetching pipeline.
    #pragma unroll
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

        // Trigger the residue if the next iteration of the prefetch loop is the one to enter
        // residue.
        if( ii == loop_to_enter_residue ) {
            a_gmem.residue();
            b_gmem.residue();
        }
    }

// The # of LDGDEPBARs must be equal to the number of stages. Issue "extra" ones if needed.
#if !defined( __LWDACC_RTC__ )
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
    compute_tile.load( a_smem, b_smem, 0 );

#ifdef USE_GMMA

    // issue some GMMA for the first kblock in prologue
    #pragma unroll
    for( int gmma_stage = 0; gmma_stage < Kernel_traits::GMMA_STAGES - 1; ++gmma_stage ) {
        // ldsm for the next kgroup/gmma_stage
        compute_tile.load( a_smem, b_smem, gmma_stage + 1 );
        xmma::warpgroup_arrive();
        compute_tile.compute( gmma_stage, true );
    }
    compute_tile.load( a_smem, b_smem, Kernel_traits::GMMA_STAGES );
    xmma::warpgroup_arrive();
    compute_tile.compute( Kernel_traits::GMMA_STAGES - 1, true );
    const int GMMA_REMAINING_STAGE = XMMAS_K - Kernel_traits::GMMA_STAGES;
#endif

    // Iterate over the loop.
    JETFIRE_MAC_LOOP_PRAGMA  // Jetfire loop body
#if !defined( __LWDACC_RTC__ )
        #pragma unroll 1
#endif
        for( int loop = params.loop_start; loop >= 0; --loop ) {
        JETFIRE_MAC_LOOP_HEADER
        
        // OCG knob to disable 3 predicated off LDS. http://lwbugs/2549067
        asm volatile (".pragma \"set knob SchedMemNoAlias=LDS+LDSM+LDGSTS\";\n" : : : "memory");
        
        // The core part of the loop.
        #pragma unroll
        for( int ki_next = 1; ki_next <= XMMAS_K; ++ki_next ) {

            // The ki_next variable is "one-step ahead". The ki points to the current step.
            int ki = ( ki_next == XMMAS_K ) ? 0 : ki_next;

            // Interference fence at top of stage
            jetfire::ifence( JETFIRE_FENCING_ENABLED );

            // Trigger the commit of global loads in last iteration
            if( ki_next == XMMAS_K ) {

                // Make sure the data was read from shared memory.
                if( Smem_tile_a::BUFFERS_PER_TILE == 1 || Smem_tile_b::BUFFERS_PER_TILE == 1 ) {
                    __syncthreads();
                }

                // Store the data to shared memory for A.
                a_gmem.commit( a_smem );
                jetfire::ifence( JETFIRE_FENCING_ENABLED );

                // Store the data to shared memory for B.
                b_gmem.commit( b_smem );
                jetfire::ifence( JETFIRE_FENCING_ENABLED );

#ifndef USE_GMMA
                // for gmma this is done at a different location
                // Make sure the data is in shared memory and synchronize threads.
                xmma::depbar<USE_LDGSTS, Kernel_traits::STAGES>();
                __syncthreads();
#endif

                // Move to next SMEM buffer, if we have multistage or double buffer.
                a_smem.move_next_write_buffer();
                b_smem.move_next_write_buffer();

                // Move the shared memory pointers for double buffering.
                // a_smem.move_next_read_buffer();
                // b_smem.move_next_read_buffer();
            }

            // Inteference fence after smem loads.
            jetfire::ifence( JETFIRE_FENCING_ENABLED );

            // Trigger the global loads on the 1st iteration of that core loop.
            if( ki_next == 1 ) {

                // Disable the loads for the last stages of the pipeline.
                if( loop < PREFETCH_STAGES ) {
                    a_gmem.disable_loads();
                    b_gmem.disable_loads();
                }

                // Trigger the loads for the A matrix.
                a_gmem.load( a_smem, mem_desc_a );
                jetfire::ifence( JETFIRE_FENCING_ENABLED );

                // Trigger the loads for the B matrix.
                b_gmem.load( b_smem, mem_desc_b );

                // Push the LDGDEPBAR instruction after the loads for A and B.
                xmma::ldgdepbar<USE_LDGSTS>();
                jetfire::ifence( JETFIRE_FENCING_ENABLED );
            }

            // Warp context switch halfway through
            if( ki_next - 1 == XMMAS_K / 2 ) {
                jetfire::warp_switch();
            }

#ifdef USE_GMMA
            if( ki_next <= GMMA_REMAINING_STAGE ) {
                // issue remaining GMMAs for the current kblock
                int gmma_stage = ki_next + Kernel_traits::GMMA_STAGES - 1;

                if( ki_next == GMMA_REMAINING_STAGE ) {
                    // load for the next kgroup/gmma stage, which will be 0
                    // move the LDSM pointer
                    a_smem.move_next_read_buffer();
                    b_smem.move_next_read_buffer();

                    // finish the ldgsts issued by previous iteration
                    // because compute_tile.load(..., 0,...) will load for the next kblock
                    xmma::depbar<USE_LDGSTS, Kernel_traits::STAGES>();
                    // if there is only one warpgroup, syncthreads is not necessary.
                    // The above statement is true if both operands are from SMEM.
                    // if one of the operand is from RF. we need syncthreads before ldsm.
                    // we also intentionally place ldsm before warpgroup.arrive to prevent
                    // compiler from moving ldsm after gmma, which causes worse performance.
                    __syncthreads();

                    // load it into A_temp
                    compute_tile.load( a_smem, b_smem, 0 );
                    xmma::warpgroup_arrive();

                    compute_tile.compute( gmma_stage, true, true );
                    // increment the entire desc group
                    compute_tile.increment_gmma_desc_group();
                    xmma::warpgroup_wait<Kernel_traits::GMMA_STAGES>();
                } else {
                    // load for the next kgroup/gmma stage
                    compute_tile.load( a_smem, b_smem, gmma_stage + 1 );
                    xmma::warpgroup_arrive();
                    compute_tile.compute( gmma_stage, true );
                    // if there are 2 GMMA_STAGES,
                    // the first 2 GMMAs issued in the prologue
                    // or the first 2 from the later half of previous iter
                    // are finished.
                    xmma::warpgroup_wait<Kernel_traits::GMMA_STAGES>();
                }
            } else {  // (ki_next <= GMMA_REMAINING_STAGE)
                // issue the GMMAs for the next kblock
                int gmma_stage = ki_next - GMMA_REMAINING_STAGE - 1;
                if( gmma_stage == ( Kernel_traits::GMMA_STAGES - 1 ) ) {
                    compute_tile.load( a_smem, b_smem, gmma_stage + 1 );
                    xmma::warpgroup_arrive();
                    compute_tile.compute( gmma_stage, true );
                    xmma::warpgroup_wait<Kernel_traits::GMMA_STAGES>();
                } else {
                    compute_tile.load( a_smem, b_smem, gmma_stage + 1 );
                    xmma::warpgroup_arrive();
                    compute_tile.compute( gmma_stage, true );
                    xmma::warpgroup_wait<Kernel_traits::GMMA_STAGES>();
                }
            }
#else
            // Do the math - The core of the loop does 16x16x8.
            compute_tile.compute( ki_next );
#endif

        }  // (ki/ki_next)

        // Load the deltas and update the filter position.
        int64_t a_delta, b_delta;
        trsi =
            Kernel_traits::load_deltas_and_move_filter_position( a_delta, b_delta, params, trsi );

        // Move the global pointers.
        a_gmem.move( trsi, a_delta );
        b_gmem.move( trsi, b_delta );

        // Skip the residue if we are going to run a full loop iteration.
        if( loop != params.loop_residue ) {
            continue;
        }

        // Execute the residue code. Clear the masks for the image if needed.
        a_gmem.residue();
        b_gmem.residue();
        
        // OCG knob to disable 3 predicated off LDS. http://lwbugs/2549067
        asm volatile (".pragma \"reset knob SchedMemNoAlias=LDS+LDSM+LDGSTS\";\n" : : : "memory");

    }  // (loop)
#ifdef USE_GMMA
    // all GMMAs must be finished
    xmma::warpgroup_wait<0>();
#endif

    // Epilogue
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

    // Make sure we can use the shared memory.
    xmma::depbar<USE_LDGSTS, 0>();
    __syncthreads();

    // Do the epilogue.
    Epilogue epilogue( params, gmem_epilogue, swizzle_epilogue, callbacks_epilogue );
    if( params.with_residual ) {
        epilogue.template execute<true>( compute_tile.acc_ );
    } else {
        epilogue.template execute<false>( compute_tile.acc_ );
    }

    // Make sure we can use the shared memory.
    __syncthreads();
    //
    // Finalize the callbacks.
    callbacks_epilogue.post_epilogue();
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Kernel with TMA
////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Kernel_traits, typename Params>
inline __device__ void matmul_tma( const Params &params,
                                   typename Kernel_traits::Compute_tile &compute_tile,
                                   typename Kernel_traits::Tile_distribution &tile ) {
    /**
     * PREFETCH TMA descriptors
     */
    /*
    xmma::utmacctl<PREFETCH>( params.a_desc );
    xmma::utmacctl<PREFETCH>( params.b_desc );
    */
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
    char *a_smem_ = &smem_[0];
    char *b_smem_ = &smem_[Smem_tile_a::BYTES_PER_TILE];

    // The tiles in shared memory.
    Smem_tile_a a_smem( a_smem_, tidx );
    Smem_tile_b b_smem( b_smem_, tidx );

    // The extra shared memory buffers that could be needed by the kernels.
    char *a_extra_smem_ = &b_smem_[Smem_tile_b::BYTES_PER_TILE];
    char *b_extra_smem_ = &a_extra_smem_[Gmem_tile_a::BYTES_PER_EXTRA_SMEM];

    // Create the objects to load from global memory.
    Gmem_tile_a a_gmem( params, params.a_desc, a_extra_smem_, tile.bidx(), tidx );
    Gmem_tile_b b_gmem( params, params.b_desc, b_extra_smem_, tile.bidx(), tidx );

    // Only enable fencing if it looks like our desired stage overlap will fit in register budget.
    constexpr bool JETFIRE_FENCING_ENABLED =
        ( 255 * 4 > sizeof( compute_tile ) + sizeof( a_gmem ) + sizeof( b_gmem ) );

    // The filter index.
    int trsi = Kernel_traits::initialize_filter_position( params );

    // Do we use LDGSTS in that kernel?
    enum { USE_LDGSTS = Gmem_tile_a::USE_LDGSTS || Gmem_tile_b::USE_LDGSTS };

    //
    // S T A R T   T H E   P I P E L I N E
    //

    // The number of stages to prefetch to start the pipeline.
    enum { PREFETCH_STAGES = xmma::Max<Kernel_traits::STAGES - 1, 1>::VALUE };
    // The number of iterations of the loop.
    const int loop_count = params.loop_start + 1;
    // The iteration to enter residue.
    const int loop_to_enter_residue = params.loop_start - ( params.loop_residue - PREFETCH_STAGES );
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

        // Trigger the residue if the next iteration of the prefetch loop is the one to enter
        // residue.
        if( ii == loop_to_enter_residue ) {
            a_gmem.residue();
            b_gmem.residue();
        }
    }

// The # of LDGDEPBARs must be equal to the number of stages. Issue "extra" ones if needed.
#if !defined( __LWDACC_RTC__ )
    #pragma unroll 1
#endif
    for( int ii = loop_count; ii < PREFETCH_STAGES; ++ii ) {
        xmma::ldgdepbar<USE_LDGSTS>();
    }

    // Make sure the data is in shared memory.
    xmma::depbar<USE_LDGSTS, Kernel_traits::STAGES>();
    __syncthreads();

    // // DEBUG.
    // if(blockIdx.x == 0 && blockIdx.y == 0) {
    // a_smem.debug_print();
    // b_smem.debug_print();
    // // END OF DEBUG.
    // }

    // Load the image pixels / filters.
    if( XMMAS_K > 1 ) {
        compute_tile.load( a_smem, b_smem, 0, true );
    }

#ifdef XMMA_ENABLED_PMTRIG
    __prof_trigger( 15 );
#endif

    // Iterate over the loop.
    JETFIRE_MAC_LOOP_PRAGMA  // Jetfire loop body
#if !defined( __LWDACC_RTC__ )
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
            jetfire::ifence( JETFIRE_FENCING_ENABLED );

            // Trigger the commit of global loads in last iteration
            if( XMMAS_K > 1 && ki_next == XMMAS_K ) {

                // Make sure the data was read from shared memory.
                if( Smem_tile_a::BUFFERS_PER_TILE == 1 || Smem_tile_b::BUFFERS_PER_TILE == 1 ) {
                    __syncthreads();
                }

                // Store the data to shared memory for A.
                a_gmem.commit( a_smem );
                jetfire::ifence( JETFIRE_FENCING_ENABLED );

                // Store the data to shared memory for B.
                b_gmem.commit( b_smem );
                jetfire::ifence( JETFIRE_FENCING_ENABLED );

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
            compute_tile.load( a_smem, b_smem, ki );

            // Inteference fence after smem loads.
            jetfire::ifence( JETFIRE_FENCING_ENABLED );

            // Trigger the global loads on the 1st iteration of that core loop.
            if( ki_next == 1 ) {

                // Disable the loads for the last stages of the pipeline.
                if( loop < PREFETCH_STAGES ) {
                    a_gmem.disable_loads();
                    b_gmem.disable_loads();
                }

                // Trigger the loads for the A matrix.
                a_gmem.load( a_smem, mem_desc_a );
                jetfire::ifence( JETFIRE_FENCING_ENABLED );

                // Trigger the loads for the B matrix.
                b_gmem.load( b_smem, mem_desc_b );

                // Push the LDGDEPBAR instruction after the loads for A and B.
                xmma::ldgdepbar<USE_LDGSTS>();
                jetfire::ifence( JETFIRE_FENCING_ENABLED );
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
                jetfire::ifence( JETFIRE_FENCING_ENABLED );

                // Store the data to shared memory for B.
                b_gmem.commit( b_smem );
                jetfire::ifence( JETFIRE_FENCING_ENABLED );

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
            compute_tile.compute( ki_next );

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
    __prof_trigger( 15 );
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

    // Make sure we can use the shared memory.
    __syncthreads();

    // Do the epilogue.
    Epilogue epilogue( params, gmem_epilogue, swizzle_epilogue, callbacks_epilogue );
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

template <typename Kernel_traits>
static __global__ __launch_bounds__( Kernel_traits::Cta_tile::THREADS_PER_CTA ) void kernel_tma(
    typename Kernel_traits::Params params ) {
    // The traits class.
    using Traits = typename Kernel_traits::Traits;
    // The XMMA tile.
    using Xmma_tile = typename Kernel_traits::Xmma_tile;
    // The tile distribution manager.
    using Tile_distribution = typename Kernel_traits::Tile_distribution;
    // The compute tile.
    using Compute_tile = typename Kernel_traits::Compute_tile;

    // Initialize the tile distribution.
    Tile_distribution tile( params, blockIdx );

    // Create the compute tile and clear the aclwmulators.
    Compute_tile compute_tile;
    compute_tile.clear();

    matmul_tma<Kernel_traits, Kernel_traits::Params>( params, compute_tile, tile );
}

}  // namespace gemm
}  // namespace xmma
