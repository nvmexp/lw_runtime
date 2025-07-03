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
#include <xmma/arrive_wait.h>

namespace xmma {
namespace gemm {

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

struct Callback_fuse_input_empty {
    template< typename Params >
    inline __device__ Callback_fuse_input_empty( const Params &params, const dim3 &bidx, int tidx ) { }

    template< int PRED_REGS >
    inline __device__ void load_vectors_m( uint32_t preds_[PRED_REGS] ) { }

    template< int PRED_REGS >
    inline __device__ void load_vectors_n( uint32_t preds_[PRED_REGS] ) { }

    inline __device__ void move( int64_t delta ) {}

    template< typename Data_type, int NUM_ELEMENTS >
    inline __device__ void apply(Data_type data[NUM_ELEMENTS]) { }
};

template< class T, class R = void >
struct enable_if_type { typedef R type; };

template< typename Kernel_traits, class Enable = void >
struct Callback_fuse_a_selector {
    using Class = Callback_fuse_input_empty;
};

template< typename Kernel_traits >
struct Callback_fuse_a_selector<Kernel_traits,
                                typename enable_if_type< typename Kernel_traits::Callback_fuse_a >::type>  {
    using Class = typename Kernel_traits::Callback_fuse_a;
};

template< typename Kernel_traits, class Enable = void >
struct Callback_fuse_b_selector {
    using Class = Callback_fuse_input_empty;
};

template< typename Kernel_traits >
struct Callback_fuse_b_selector<Kernel_traits,
                                typename enable_if_type<typename Kernel_traits::Callback_fuse_b>::type>  {
    using Class = typename Kernel_traits::Callback_fuse_b;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Kernel_traits, typename Params >
inline __device__ void matmul(const Params &params,
                              typename Kernel_traits::Compute_tile &compute_tile,
                              typename Kernel_traits::Tile_distribution &tile)
{
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

    using Callback_fuse_a = typename Callback_fuse_a_selector<Kernel_traits>::Class;
    using Callback_fuse_b = typename Callback_fuse_b_selector<Kernel_traits>::Class;

    // The shared memory. It is allocated at launch time.
    extern __shared__ char smem_[];

    // The shared memory pointers.
    char* a_smem_data = &smem_[0];
    char* b_smem_data = &smem_[Smem_tile_a::BYTES_PER_TILE];

    Callback_fuse_a a_fuse( params, tile.bidx(), tidx );

    // The tiles in shared memory.
    Smem_tile_a a_smem( a_smem_data, tidx );
    Smem_tile_b b_smem( b_smem_data, tidx );

    // The extra shared memory buffers that could be needed by the kernels.
    char* a_extra_smem_ = &b_smem_data[Smem_tile_b::BYTES_PER_TILE];
    char* b_extra_smem_ = &a_extra_smem_[Gmem_tile_a::BYTES_PER_EXTRA_SMEM];

    if( Gmem_tile_a::USE_UTMALDG && Gmem_tile_b::USE_UTMALDG ) {
        uint64_t *smem_barriers_ =
            reinterpret_cast<uint64_t *>( &b_extra_smem_[Gmem_tile_b::BYTES_PER_EXTRA_SMEM] );

        if( threadIdx.x == 0 ) {
            for( int i = 0; i < ( ( Kernel_traits::BUFFERS_PER_SMEM_TILE_A ) ); i++ )
                xmma::bar_create( &smem_barriers_[i], Cta_tile::THREADS_PER_CTA );
        }
        __syncthreads();

        a_smem.add_smem_barrier_base( smem_barriers_ );
        b_smem.add_smem_barrier_base( smem_barriers_ );
    }

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

    //
    // S T A R T   T H E   P I P E L I N E
    //

    // Load vectors aligned to `m' dimension and broadcast to `n' dimension
    a_gmem.load_vectors_m( a_fuse );
    // Load vectors aligned to `n' dimension and broadcast to `m' dimension
    a_gmem.load_vectors_n( a_fuse );

    // The number of stages to prefetch to start the pipeline.
    enum { PREFETCH_STAGES = xmma::Max<Kernel_traits::STAGES - 1, 1>::VALUE };
    // The number of iterations of the loop.
    const int loop_count = params.loop_start + 1;
    // The iteration to enter residue.
    const int loop_to_enter_residue = params.loop_start - (params.loop_residue - PREFETCH_STAGES);
    // Initialize the prefetching pipeline.
    for( int ii = 0, prefetch = min( loop_count, PREFETCH_STAGES ); ii < prefetch; ++ii ) {
        // Trigger the loads for A and B. Either LDG or LDGSTS.

        // Trigger the loads for A and B. Either LDG or LDGSTS.
        a_gmem.load( a_smem, mem_desc_a );
        b_gmem.load( b_smem, mem_desc_b );

        a_gmem.apply_fuse( a_fuse );

        // Make sure we insert the corresponding LDGDEPBAR. NOP on Volta/Turing.
        xmma::ldgdepbar<USE_LDGSTS>();

        // Store the pixels and filters to shared memory.
        a_gmem.commit( a_smem );
        b_gmem.commit( b_smem );

        // Load the deltas and update the filter position.
        int64_t a_delta, b_delta;
        trsi = Kernel_traits::load_deltas_and_move_filter_position( a_delta, b_delta, params, trsi );

        // Move to next SMEM buffer for multistage or double buffer.
        a_smem.move_next_write_buffer();
        b_smem.move_next_write_buffer();

        // Move the pointers and assemble the predicates for the next loop.
        a_gmem.move( trsi, a_delta );
        b_gmem.move( trsi, b_delta );

        a_fuse.move( a_delta );

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
    //a_smem.debug_print();
    //b_smem.debug_print();
    // // END OF DEBUG.
     
    // Load the image pixels / filters.
    if( XMMAS_K > 1 ) {
        compute_tile.load(a_smem, b_smem, 0, true);
        compute_tile.apply_fuse_a( a_fuse );
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

#ifdef HOPPER_DMMA
// This is a WAR for Bug 200653314. It should be removed once the bug is closed.
asm volatile (".pragma \"set knob SchedResBusyDMMA=64\";\n" : : : "memory");
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

                // Make sure the data was read from shared memory.
                if( Smem_tile_a::BUFFERS_PER_TILE == 1 || Smem_tile_b::BUFFERS_PER_TILE == 1 ) {
                    __syncthreads();
                }

                // For main loop fusion:
                // d = matmal(f(a, x0, ...), g(b, y0, ...))) + acc
                // a_gmem.apply_fuse := f(a, x0, ...) where f is a lwstomized functor, x0 is a vector
                // b_gmem.apply_fuse := g(b, y0, ...) where g is a lwstomized functor, y0 is a vector
                a_gmem.apply_fuse( a_fuse );

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
            compute_tile.load( a_smem, b_smem, ki );
            // For main loop fusion:
            //   d = matmal(f(a, x0, ...), g(b, y0, ...))) + acc
            //     a_gmem.apply_fuse := f(a, x0, ...) where f is a lwstomized functor, x0 is a vector
            //     b_gmem.apply_fuse := g(b, y0, ...) where g is a lwstomized functor, y0 is a vector
            compute_tile.apply_fuse_a( a_fuse );

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
                a_gmem.load_vectors_n( a_fuse );

                jetfire::ifence( JETFIRE_FENCING_ENABLED );
                
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

        // a_fuse.move();
        // b_fuse.move();

        // Execute the residue code. Clear the masks for the image if needed.
        if( loop <= params.loop_residue ) {
            a_gmem.residue();
            b_gmem.residue();
        }

#ifdef HOPPER_DMMA
// This is a WAR for Bug 200653314. It should be removed once the bug is closed.
asm volatile (".pragma \"reset knob SchedResBusyDMMA=64\";\n" : : : "memory");
#endif

#ifdef JETFIRE_ENABLED
    asm volatile (".pragma \"reset knob SchedMemNoAlias=LDS+LDSM+LDGSTS\";\n" : : : "memory");
#endif

    }  // (loop)
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

    // Finalize the callbacks.
    callbacks_epilogue.post_epilogue();

}

template< typename Kernel_traits, typename Params >
inline __device__ void matmul_spread_ldgsts(const Params &params,
                              typename Kernel_traits::Compute_tile &compute_tile,
                              typename Kernel_traits::Tile_distribution &tile)
{
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

    using Callback_fuse_a = typename Callback_fuse_a_selector<Kernel_traits>::Class;
    using Callback_fuse_b = typename Callback_fuse_b_selector<Kernel_traits>::Class;

    // The shared memory. It is allocated at launch time.
    extern __shared__ char smem_[];

    // The shared memory pointers.
    char* a_smem_data = &smem_[0];
    char* b_smem_data = &smem_[Smem_tile_a::BYTES_PER_TILE];

    Callback_fuse_a a_fuse( params, tile.bidx(), tidx );

    // The tiles in shared memory.
    Smem_tile_a a_smem( a_smem_data, tidx );
    Smem_tile_b b_smem( b_smem_data, tidx );

    // The extra shared memory buffers that could be needed by the kernels.
    char* a_extra_smem_ = &b_smem_data[Smem_tile_b::BYTES_PER_TILE];
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

    //
    // S T A R T   T H E   P I P E L I N E
    //

    // Load vectors aligned to `m' dimension and broadcast to `n' dimension
    a_gmem.load_vectors_m( a_fuse );
    // Load vectors aligned to `n' dimension and broadcast to `m' dimension
    a_gmem.load_vectors_n( a_fuse );

    // The number of stages to prefetch to start the pipeline.
    enum { PREFETCH_STAGES = xmma::Max<Kernel_traits::STAGES - 1, 1>::VALUE };
    // The number of iterations of the loop.
    const int loop_count = params.loop_start + 1;
    // The iteration to enter residue.
    const int loop_to_enter_residue = params.loop_start - (params.loop_residue - PREFETCH_STAGES);
    // Initialize the prefetching pipeline.
    for( int ii = 0, prefetch = min( loop_count, PREFETCH_STAGES ); ii < prefetch; ++ii ) {
        // Trigger the loads for A and B. Either LDG or LDGSTS.

        // Trigger the loads for A and B. Either LDG or LDGSTS.
        a_gmem.load( a_smem, mem_desc_a );
        b_gmem.load( b_smem, mem_desc_b );

        a_gmem.apply_fuse( a_fuse );

        // Make sure we insert the corresponding LDGDEPBAR. NOP on Volta/Turing.
        xmma::ldgdepbar<USE_LDGSTS>();

        // Store the pixels and filters to shared memory.
        a_gmem.commit( a_smem );
        b_gmem.commit( b_smem );

        // Load the deltas and update the filter position.
        int64_t a_delta, b_delta;
        trsi = Kernel_traits::load_deltas_and_move_filter_position( a_delta, b_delta, params, trsi );

        // Move to next SMEM buffer for multistage or double buffer.
        a_smem.move_next_write_buffer();
        b_smem.move_next_write_buffer();

        // Move the pointers and assemble the predicates for the next loop.
        a_gmem.move( trsi, a_delta );
        b_gmem.move( trsi, b_delta );

        a_fuse.move( a_delta );

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
    //a_smem.debug_print();
    //b_smem.debug_print();
    // // END OF DEBUG.
     
    // Load the image pixels / filters.
    if( XMMAS_K > 1 ) {
        compute_tile.load(a_smem, b_smem, 0, true);
        compute_tile.apply_fuse_a( a_fuse );
    }
#ifdef XMMA_ENABLED_PMTRIG
    __prof_trigger(15);
#endif
    // LOOK AHEAD
    // Trigger the loads for the A matrix.
    a_gmem.template load_per_phase<Smem_tile_a, 0>( a_smem, mem_desc_a );
    // Trigger the loads for the B matrix.
    b_gmem.template load_per_phase<Smem_tile_b, 0>( b_smem, mem_desc_b );
    
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

#ifdef HOPPER_DMMA
// This is a WAR for Bug 200653314. It should be removed once the bug is closed.
asm volatile (".pragma \"set knob SchedResBusyDMMA=64\";\n" : : : "memory");
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

                // Make sure the data was read from shared memory.
                if( Smem_tile_a::BUFFERS_PER_TILE == 1 || Smem_tile_b::BUFFERS_PER_TILE == 1 ) {
                    __syncthreads();
                }
                 // For main loop fusion:
                 // d = matmal(f(a, x0, ...), g(b, y0, ...))) + acc
                 // a_gmem.apply_fuse := f(a, x0, ...) where f is a lwstomized functor, x0 is a vector
                 // b_gmem.apply_fuse := g(b, y0, ...) where g is a lwstomized functor, y0 is a vector
                 a_gmem.apply_fuse( a_fuse );

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
            
            if (ki == 0){
                // Trigger the loads for the A matrix.
                a_gmem.template load_per_phase<Smem_tile_a, 0>( a_smem, mem_desc_a );
                // Trigger the loads for the B matrix.
                b_gmem.template load_per_phase<Smem_tile_b, 0>( b_smem, mem_desc_b );
                if( ki == XMMAS_K - 1 ) {
                    // Push the LDGDEPBAR instruction after the loads for A and B.
                    xmma::ldgdepbar<USE_LDGSTS>();
                }
            } else if (ki == 1) {
                // Trigger the loads for the A matrix.
                a_gmem.template load_per_phase<Smem_tile_a, 1>( a_smem, mem_desc_a );
                // Trigger the loads for the B matrix.
                b_gmem.template load_per_phase<Smem_tile_b, 1>( b_smem, mem_desc_b );
                if( ki == XMMAS_K - 1 ) {
                    // Push the LDGDEPBAR instruction after the loads for A and B.
                    xmma::ldgdepbar<USE_LDGSTS>();
                }
            } else if (ki == 2) {
                // Trigger the loads for the A matrix.
                a_gmem.template load_per_phase<Smem_tile_a, 2>( a_smem, mem_desc_a );
                // Trigger the loads for the B matrix.
                b_gmem.template load_per_phase<Smem_tile_b, 2>( b_smem, mem_desc_b );
                if( ki == XMMAS_K - 1 ) {
                    // Push the LDGDEPBAR instruction after the loads for A and B.
                    xmma::ldgdepbar<USE_LDGSTS>();
                }
            } else if (ki == 3) {
                // Trigger the loads for the A matrix.
                a_gmem.template load_per_phase<Smem_tile_a, 3>( a_smem, mem_desc_a );
                // Trigger the loads for the B matrix.
                b_gmem.template load_per_phase<Smem_tile_b, 3>( b_smem, mem_desc_b );
                if( ki == XMMAS_K - 1 ) {
                    // Push the LDGDEPBAR instruction after the loads for A and B.
                    xmma::ldgdepbar<USE_LDGSTS>();
                }
            } else if (ki == 4) {
                // Trigger the loads for the A matrix.
                a_gmem.template load_per_phase<Smem_tile_a, 4>( a_smem, mem_desc_a );
                // Trigger the loads for the B matrix.
                b_gmem.template load_per_phase<Smem_tile_b, 4>( b_smem, mem_desc_b );
                if( ki == XMMAS_K - 1 ) {
                    // Push the LDGDEPBAR instruction after the loads for A and B.
                    xmma::ldgdepbar<USE_LDGSTS>();
                }
            } else if (ki == 5) {
                // Trigger the loads for the A matrix.
                a_gmem.template load_per_phase<Smem_tile_a, 5>( a_smem, mem_desc_a );
                // Trigger the loads for the B matrix.
                b_gmem.template load_per_phase<Smem_tile_b, 5>( b_smem, mem_desc_b );
                if( ki == XMMAS_K - 1 ) {
                    // Push the LDGDEPBAR instruction after the loads for A and B.
                    xmma::ldgdepbar<USE_LDGSTS>();
                }
            } else if (ki == 6) {
                // Trigger the loads for the A matrix.
                a_gmem.template load_per_phase<Smem_tile_a, 6>( a_smem, mem_desc_a );
                // Trigger the loads for the B matrix.
                b_gmem.template load_per_phase<Smem_tile_b, 6>( b_smem, mem_desc_b );
                if( ki == XMMAS_K - 1 ) {
                    // Push the LDGDEPBAR instruction after the loads for A and B.
                    xmma::ldgdepbar<USE_LDGSTS>();
                }
            } else if (ki == 7) {
                // Trigger the loads for the A matrix.
                a_gmem.template load_per_phase<Smem_tile_a, 7>( a_smem, mem_desc_a );
                // Trigger the loads for the B matrix.
                b_gmem.template load_per_phase<Smem_tile_b, 7>( b_smem, mem_desc_b );
                if( ki == XMMAS_K - 1 ) {
                    // Push the LDGDEPBAR instruction after the loads for A and B.
                    xmma::ldgdepbar<USE_LDGSTS>();
                }
            }
             
            a_gmem.load_vectors_n( a_fuse );
            // Load the matrices from shared memory.
            compute_tile.load(a_smem, b_smem, ki);
            // For main loop fusion:
            // d = matmal(f(a, x0, ...), g(b, y0, ...))) + acc
            // a_gmem.apply_fuse := f(a, x0, ...) where f is a lwstomized functor, x0 is a vector
            // b_gmem.apply_fuse := g(b, y0, ...) where g is a lwstomized functor, y0 is a vector
            compute_tile.apply_fuse_a( a_fuse );
                                                             
            // Inteference fence after smem loads.
            jetfire::ifence(JETFIRE_FENCING_ENABLED);

            if( ki == XMMAS_K - 1 ) {

                // Load the deltas and update the filter position.
                int64_t a_delta, b_delta;
                trsi =
                    Kernel_traits::load_deltas_and_move_filter_position( a_delta, b_delta, params, trsi );

                // Move the global pointers.
                a_gmem.move( trsi, a_delta );
                b_gmem.move( trsi, b_delta );

                // Disable the loads for the last stages of the pipeline.
                if( loop - 1 < PREFETCH_STAGES ) {
                    a_gmem.disable_loads();
                    b_gmem.disable_loads();
                }

                // Skip the residue if we are going to run a full loop iteration.
                if( loop == params.loop_residue ) {
                    // Execute the residue code. Clear the masks for the image if needed.
                    a_gmem.residue();
                    b_gmem.residue();
                }
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

#ifdef HOPPER_DMMA
// This is a WAR for Bug 200653314. It should be removed once the bug is closed.
asm volatile (".pragma \"reset knob SchedResBusyDMMA=64\";\n" : : : "memory");
#endif

#ifdef JETFIRE_ENABLED
    asm volatile (".pragma \"reset knob SchedMemNoAlias=LDS+LDSM+LDGSTS\";\n" : : : "memory");
#endif
    }  // (loop)
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

    // Finalize the callbacks.
    callbacks_epilogue.post_epilogue();

}

template< typename Kernel_traits, typename Params >
inline __device__ void matmul_wo_smem(const Params &params,
                            typename Kernel_traits::Compute_tile &compute_tile,
                            typename Kernel_traits::Tile_distribution &tile)
{
    // The traits class.
    using Traits = typename Kernel_traits::Traits;
    // The CTA tile.
    using Cta_tile = typename Kernel_traits::Cta_tile;
    // The XMMA tile.
    using Xmma_tile = typename Kernel_traits::Xmma_tile;

    const int XMMAS_M = Xmma_tile::XMMAS_M;
    const int XMMAS_N = Xmma_tile::XMMAS_N;
    const int XMMAS_K = Xmma_tile::XMMAS_K;

    // The block/tile indices.
    int bidm = tile.bidm();
    int bidn = tile.bidn();
    int bidz = tile.bidz();

    // The thread index.
    const int tidx = threadIdx.x;

    // The tiles in global memory for the images.
    using Gmem_wo_smem_tile_a = typename Kernel_traits::Gmem_wo_smem_tile_a;

    // The tiles in global memory for the filters.
    using Gmem_wo_smem_tile_b = typename Kernel_traits::Gmem_wo_smem_tile_b;

    // The fragment.
//    using Fragment_a = Fragment_a<Traits, typename Gmem_wo_smem_tile_a::Gmem_layout>;
//    using Fragment_b = Fragment_b<Traits, typename Gmem_wo_smem_tile_b::Gmem_layout>;
    using Fragment_a = typename Kernel_traits::Smem_tile_a::Fragment;
    using Fragment_b = typename Kernel_traits::Smem_tile_b::Fragment;

    // Create the objects to load from global memory.
    Gmem_wo_smem_tile_a a_gmem( params, NULL, tile.bidx(), tidx );
    Gmem_wo_smem_tile_b b_gmem( params, NULL, tile.bidx(), tidx );

    Fragment_aclwmulator<Traits> acc[XMMAS_M][XMMAS_N];
    Fragment_a a[XMMAS_K][XMMAS_M];
    Fragment_b b[XMMAS_K][XMMAS_N];

    // Clear the aclwmulators.
    xmma::helpers::clear(acc);

    //
    // S T A R T   T H E   P I P E L I N E
    //
    a_gmem.load(a, 0);
    b_gmem.load(b, 0);
    jetfire::ifence(1);

    // Iterate over the loop.
    JETFIRE_MAC_LOOP_PRAGMA  // Jetfire loop body
#if !defined(__LWDACC_RTC__)
    #pragma unroll 1
#endif
    for( int loop = params.loop_start - 1; loop >= 0; --loop ) {
        JETFIRE_MAC_LOOP_HEADER

        #pragma unroll
        for (int ki_next = 1; ki_next <= XMMAS_K / 2; ki_next++) {

            int ki = ( ki_next == XMMAS_K / 2 ) ? 0 : ki_next;

            a_gmem.load(a, ki);
            b_gmem.load(b, ki);
            jetfire::ifence(1);

            #pragma unroll
            for (int ni = 0; ni < XMMAS_N; ni++) {
                #pragma unroll
                for (int k = 0; k < 2; k++) {
                    jetfire::ifence(1);
                    #pragma unroll
                    for (int mi = 0; mi < XMMAS_M; mi++) {
                        acc[mi][ni].mma(a[(ki_next-1) * 2 + k][mi],b[(ki_next-1) * 2 + k][ni]);
                    }
                }
            }
            jetfire::ifence(1);

            if (ki_next == 1) {
                a_gmem.move();
                b_gmem.move();

                if (loop == 0) {
                    a_gmem.residue();
                    b_gmem.residue();
                }
            }
        }
    }  // (loop)

    // Residue
    a_gmem.residue(1);
    b_gmem.residue(1);

    #pragma unroll
    for (int ki_next = 1; ki_next <= XMMAS_K / 2; ki_next++) {
        int ki = ( ki_next == XMMAS_K / 2 ) ? 0 : ki_next;
        if(ki == 1) {
            a_gmem.load(a, ki);
            b_gmem.load(b, ki);
        }
        #pragma unroll
        for (int ni = 0; ni < XMMAS_N; ni++) {
            #pragma unroll
            for (int k = 0; k < 2; k++) {
                jetfire::ifence(1);
                #pragma unroll
                for (int mi = 0; mi < XMMAS_M; mi++) {
                    acc[mi][ni].mma(a[(ki_next-1) * 2 + k][mi], b[(ki_next-1) * 2 + k][ni]);
                }
            }
        }
    }

    // The tile in global memory.
    using Gmem_tile_epilogue = typename Kernel_traits::Gmem_tile_wo_smem_epilogue;
    // The tile in shared memory to swizzle the output.
    using Swizzle_epilogue = typename Kernel_traits::Swizzle_epilogue;

    // The callbacks.
    using Callbacks_epilogue = typename Kernel_traits::Callbacks_wo_smem_epilogue;
    // The epilogue.
    using Epilogue = typename Kernel_traits::Epilogue_wo_smem;

    // Do allocate the tile to output in the epilogue.
    Gmem_tile_epilogue gmem_epilogue( params, bidm, bidn, bidz, tidx );
    // Do allocate the tile and compute the offsets. TODO: Merge with the epilogue class!
    Swizzle_epilogue swizzle_epilogue( NULL, tidx );
    // The callbacks.
    Callbacks_epilogue callbacks_epilogue( params, NULL, bidm, bidn, bidz, tidx );

    // Do the epilogue.
    Epilogue epilogue( params, gmem_epilogue, swizzle_epilogue, callbacks_epilogue );
    if( params.with_residual ) {
        epilogue.template execute<true>(acc);
    } else {
        epilogue.template execute<false>(acc);
    }

}

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
    if (Kernel_traits::STAGES == 0) {
        matmul_wo_smem<Kernel_traits>(params, compute_tile, tile);
    }
    else if (Kernel_traits::STAGES > 1) {
#if defined( __LWDA_ARCH__ ) && __LWDA_ARCH__ == 860
        matmul_spread_ldgsts<Kernel_traits>(params, compute_tile, tile);
#else
        matmul<Kernel_traits>(params, compute_tile, tile);
#endif
    }
    else {
        matmul<Kernel_traits>(params, compute_tile, tile);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Kernel_traits>
static __device__ void exelwte_split_k( const typename Kernel_traits::Params &params ) {
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
    // The callbacks.
    using Callbacks_epilogue = typename Kernel_traits::Callbacks_epilogue;
    // The epilogue.
    using Epilogue = typename Kernel_traits::Epilogue;

    // Do allocate the tile to output in the epilogue.
    Gmem_tile_epilogue gmem_epilogue( params, bidm, bidn, bidz, tidx );
    // Do allocate the tile and compute the offsets. TODO: Merge with the epilogue class!
    Swizzle_epilogue swizzle_epilogue( smem_, tidx );
    // The callbacks.
    Callbacks_epilogue callbacks( params, smem_, bidm, bidn, 0, tidx );

    enum { CONTIGUOUS = Gmem_tile_epilogue::Layout::ROW ? Xmma_tile::XMMAS_N : Xmma_tile::XMMAS_M };

    // Do the epilogue.
    Epilogue epilogue( params, gmem_epilogue, swizzle_epilogue, callbacks );
    epilogue.template exelwte_split_k<CONTIGUOUS>();
}

template <typename Kernel_traits>
static __global__ __launch_bounds__( Kernel_traits::Cta_tile::THREADS_PER_CTA )
void split_k_kernel( typename Kernel_traits::Params params ) {
    exelwte_split_k<Kernel_traits>(params);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace gemm
}  // namespace xmma
