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
#include <xmma/helpers/epilogue.h>
#include <xmma/arrive_wait.h>
#include <xmma/named_barrier.h>
#include <xmma/cta_reconfig.h>

namespace xmma {
namespace gemm {

////////////////////////////////////////////////////////////////////////////////////////////////////
// Warp Specialized *persistent* GMMA kernel
// In this kernel: CTA has 3 specialized groups and associated synchonrizations among them.
// Each SM only launches 1 *persistent* CTA. It has 3 warpgroups:
//    - 1 specialized warpgroup for DMA global to shared (input buffer load),
//    - 2 specialzied warpgroups for math->epilog.
// Two synchronizations:
//   -  Buffer synchronization is needed among DMA warpgroup and Mathwarpgroup;
//   -  Mutex synchronization is needed between  Mathwarpgroup 0 and mathwarpgroup 1
//       to control who enters the the mainloop and epilog as  mainlooop and epilog
//       are considered as critical regions that only one will execute exclusively.
// Each mathwarpgroup's mainloop will produce one output (C) tile.
// Diagram for 3 output tiles in a SM.
// DMA warpgroup       |----------- buffers load cirlwlarly --------|
// Math warpgroup 0    |----mainloop--|--Epilog--|        |----mainloop---|--Epilog---|
// Math warpgroup 1                   |------mainloop-----|--Epilog--|
//**Noted warp specialization doesn't change the way on how global data is loaded to smem buffer
//**and how mainloop and epilog is being done.
////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////
// Allocate TMA descriptors in constant memory
//
// For HOPPER:
// TMA instruction in Hopper takes the following as operands
// 1. Global memory address of TMA descriptor
// 2. Coordinates of tile block to load/store data
// 3. 16B aligned shared memory address to load/store data
// 4. Barrier pointer for loading tile blocks to shared memory
//
// The TMA unit is built along constant fetch path
// When TMA unit gets the global memory address of the descriptor, it is loaded from
// global memory -> GCC -> TMA descriptor cache.
// Yes, TMA has its own cache which holds only descriptors.
//
// At kernel launch, kernel arguments and kernel constants aka programmable constant memory
// (like the ones below) will be prefetched to constant banks. In this process, L2 and GCC
// are populated with constants and arguments too
//
// If TMA descriptors are present in global memory, when the first tma load instruction is
// issued, before issuing data loads, tma units issues loads (ldg) to get tma descriptors
// in global memory and store it in tma cache. Once the tma unit has the descriptors,
// it'll issue data loads. This is a high latency path.
//
// The high first-tma-instruction latency can be reduced if the descriptor is 
// already present in tma cache.
//
// This is achieved in two steps:
// 1. Put tma descriptors in constant memory and leverage constant prefetch at kernel launch
// feature.
// This is implemented by, allocating tma descriptors in programmable constant memory.
// The tma instruction takes global memory address to tma descriptor. So, before launching
// the kernel, we get the associated global memory address for each descriptor in constant
// memory. This global memory address for each descriptor is passed to kernel through
// params structure.
// This will populate L2 and GCC at kernel launch throught constant fetch path
// but doesn't populate TMA cache. 
// 2. To get tma descriptor from GCC to tma cache, in prologue, immediately after kernel launch,
// we issue a prefetch instruction which fetches tma descriptor from global memory.
// In this case, the tma descriptor is already in GCC, so it fetches from GCC to tma cache.
// The argument to tma descriptor prefetch instruction is global memory address of the descriptor
////////////////////////////////////////////////////////////////////////////////////////////////////
namespace warp_specialized_kernel {

////////////////////////////////////////////////////////////////////////////////////////////////////
// [0]: TMA descriptor for operand A
// [1]: TMA descriptor for operand B
// [2]: TMA descriptor for operand C
// [3]: TMA descriptor for operand D
////////////////////////////////////////////////////////////////////////////////////////////////////
static __constant__ lwdaTmaDescv2 desc_k[4];

}
////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename Kernel_traits, bool Grouped_acc = false>
static __global__ __launch_bounds__(
    // FIXME: need to refactor Cta_tile::THREADS_PER_CTA  to be Cta_tile::THREADS_PER_WARPGROUP
    Kernel_traits::Cta_tile::THREADS_PER_CTA * 3,
    1 ) void warp_specialized_kernel_gmma( typename Kernel_traits::Params params ) {
#if defined( __LWDA_ARCH__ ) && __LWDA_ARCH__ >= 900
    DisableWar_SW254906_MACRO
    
    // The CTA tile.
    using Cta_tile = typename Kernel_traits::Cta_tile;
    // The traits class.
    using Traits = typename Kernel_traits::Traits;
    // The XMMA tile.
    using Xmma_tile = typename Kernel_traits::Xmma_tile;
    // The tile distribution manager.
    using Tile_distribution_persistent = typename Kernel_traits::Tile_distribution_persistent;
    // The tiles to store the data in shared memory for the images (A).
    using Smem_tile_a = typename Kernel_traits::Smem_tile_a;
    // The tiles to store the data in shared memory for the filters (B).
    using Smem_tile_b = typename Kernel_traits::Smem_tile_b;
    // The tiles to store epilog
    using Swizzle_epilogue = typename Kernel_traits::Swizzle_epilogue;
    // The compute tile
    using Compute_tile = typename Kernel_traits::Compute_tile;
    // The tiles in global memory for the B.
    using Gmem_tile_a = typename Kernel_traits::Gmem_tile_a;
    // The tiles in global memory for the B.
    using Gmem_tile_b = typename Kernel_traits::Gmem_tile_b;
    // The tile in global memory.
    using Gmem_tile_epilogue = typename Kernel_traits::Gmem_tile_epilogue;
    // The callbacks.
    using Callbacks_epilogue = typename Kernel_traits::Callbacks_epilogue;
    // The epilogue.
    using Epilogue = typename Kernel_traits::Epilogue;

    enum { COPY_BYTES = Smem_tile_a::BYTES_PER_BUFFER + Smem_tile_b::BYTES_PER_BUFFER };

    // The shared memory. It is allocated at launch time.
    extern __shared__ char smem_[];

    //*******************buffer synchornization barrier setup****************
    // The shared memory barriers setup
    // The shared memory barrier pointer
    uint64_t *smem_barriers_ = reinterpret_cast<uint64_t *>(
        &smem_[Smem_tile_a::BYTES_PER_TILE + Smem_tile_b::BYTES_PER_TILE +
               Swizzle_epilogue::BYTES_PER_TILE] );
    // The base pointer setup for buffer_full_barriers used for synchronizing on buffer_full.
    // Each A+B buffer owns a buffer_full_barrier,
    // the barrier can be referenced by the base pointer and buffer_index.
    // DMA warps will ARRIVE on buffer_full_barrier and Math warps will WAIT
    // on the buffer_full_barrier.
    xmma::Arrive_wait buffer_full_barriers( &smem_barriers_[0] );
    // The base pointer setup for buffer_empty_barriers used for synchronizing on buffer_empty.
    // Each A+B buffer owns a buffer_full_barrier,
    // the barrier can be referenced by the base pointer and buffer_index.
    // DMA warps will WAIT on buffer_empty_barrier  and Math warps will ARRIVE i
    // on the buffer_empty_barrier.
    xmma::Arrive_wait buffer_empty_barriers(
        &smem_barriers_[Kernel_traits::BUFFERS_PER_SMEM_TILE_A] );

    if( Kernel_traits::TRAITS_USE_UTMALDG ) {
        if( threadIdx.x == 0 ) {
            // Create buffer_full barriers with 1 arrive count
            // This is later used by A1TR which register an arrive count for each transaction as 1
            for( int i = 0; i < Kernel_traits::BUFFERS_PER_SMEM_TILE_A; i++ ) {
                xmma::bar_create( &smem_barriers_[i], 1 );
            }
            // No change in how buffer_empty barriers for hopper as it does not touch tma
            for( int i = Kernel_traits::BUFFERS_PER_SMEM_TILE_A;
                 i < Kernel_traits::BUFFERS_PER_SMEM_TILE_A * 2;
                 i++ ) {
                xmma::bar_create( &smem_barriers_[i], Cta_tile::THREADS_PER_CTA );
            }
        }
        __syncthreads();
    } else {
        // The smem barrier initialization
        // The total smem barriers is twice of the number of A+B buffers.
        if( threadIdx.x == 0 ) {
            for( int i = 0; i < ( (Kernel_traits::BUFFERS_PER_SMEM_TILE_A)*2 ); i++ )
                // FIXME:need to refactor Cta_tile::THREADS_PER_CTA to
                // Cta_tile::THREADS_PER_WARPGROUP
                xmma::bar_create( &smem_barriers_[i], Cta_tile::THREADS_PER_CTA );
        }
        __syncthreads();
    }

    //*******************input smem buffer and epilog smem buffer pointer setup****************
    // The shared memory A/B/epilogue buffer pointers.
    char *a_smem_ = &smem_[0];
#ifdef USE_GMMA
    a_smem_ = xmma::align_128( a_smem_ );
#endif
    char *b_smem_ = &smem_[Smem_tile_a::BYTES_PER_TILE];
    char *epi_smem_ = &smem_[Smem_tile_a::BYTES_PER_TILE + Smem_tile_b::BYTES_PER_TILE];

    // The warp id.
    const int warp_id = __shfl_sync( 0xffffffff, threadIdx.x / 32, 0 );
    // The branch on whether it is DMA warps (warp 0~3) or math warps (warp 4~11).
    // FIXME: need to refactor Cta_tile::THREADS_PER_CTA  to be Cta_tile::THREADS_PER_WARPGROUP

    if( warp_id < Cta_tile::THREADS_PER_CTA / 32 )
    //*******************DMA warpgroup exelwtion region***************************
    // The operations for DMA warpgroup (consist of warp 0~3).
    {
        // Hopper CTA register reconfiguration feature to reduce register count
        // for DMA warps as it needs less.
        xmma::reg_dealloc();

        //*******************Persistent CTA's Tile Setup*************************
        // Block index for first tile in persistent model
        Tile_distribution_persistent tile( params, blockIdx.x );
        // Memory descritors.
        const uint64_t mem_desc_a = params.mem_descriptors.descriptor_a;
        const uint64_t mem_desc_b = params.mem_descriptors.descriptor_b;
        // The first tile dimension
        int bidm = tile.bidm(), bidz = tile.bidz();
        int bidn = tile.bidn();

        // The thread index.
        const int tidx = threadIdx.x;
        // The index of A/B buffer
        int buffer_head = 0;
        // The cnt for loop iteration, used for updating the phase bit
        // for the buffer_empty_barrier.
        int cnt = Kernel_traits::BUFFERS_PER_SMEM_TILE_A * -1;

        //*******************smem buffer constructor*************************
        // The tiles in shared memory.
        Smem_tile_a a_smem( a_smem_, tidx );
        Smem_tile_b b_smem( b_smem_, tidx );

        if( Gmem_tile_a::USE_UTMALDG && Gmem_tile_b::USE_UTMALDG ) {
            if( warp_id == 0 ) {
                if( xmma::elect_one_sync() ) {
                    // TODO @aatluri: Add tma descriptor prefetch for operand A
                    a_smem.add_smem_barrier_base( smem_barriers_ );
                }
            }
            if( warp_id == 2 ) {
                if( xmma::elect_one_sync() ) {
                    // TODO @aatluri: Add tma descriptor prefetch for operand B
                    b_smem.add_smem_barrier_base( smem_barriers_ );
                }
            }
        }

        // TODO @albertx: add lwca toolkit check for release
        // Issue ACQBULK.
        __lw_ptx_builtin_ocg_acqblk();

        // The phase_bit used in buffer_empty_barriers's  wait operation.
        unsigned int phase_bit;
        // Iterate over the persistent cta's output tiles.
        while( !tile.is_last() ) {
            //*******************global tile constructor********************
            // The  global memory tile.
            Gmem_tile_a a_gmem( params, NULL, tile.bidx(), tidx );
            // The global memory tile.
            Gmem_tile_b b_gmem( params, NULL, tile.bidx(), tidx );
            int trsi = Kernel_traits::initialize_filter_position( params );
            // advance to persistent cta's next tile
            tile.move();
            bidm = tile.bidm();
            bidz = tile.bidz();
            bidn = tile.bidn();
            //*******************Loop on ciruclar buffer load**************
// Iterate over the loop on kblocks in persistent cta's tile.
#pragma unroll 1
            for( int loop = params.loop_start; loop >= 0; --loop ) {
                if( cnt >= 0 ) {
                    // The barrier wait for buffer_empty_barrier.
                    // The buffer_empty_barrier will wait until
                    // the buffer_empty_barrier arrival is complete.
                    // The buffer_empty_barrier arrival (done by math warpgroup) is complete by
                    // updating the phase_bit inside the barrier.
                    // The buffer_empty_barrier wait will know it by
                    // checking whether the phase_bit is updated.
                    phase_bit = cnt < Kernel_traits::BUFFERS_PER_SMEM_TILE_A ? 0 : 1;
                    buffer_empty_barriers.bar_wait( buffer_head, phase_bit );
                }
                // The load of A/B smem buffer by async copy operations.
                if( Kernel_traits::TRAITS_USE_UTMALDG ) {
                    if( warp_id == 0 ) {
                        if( xmma::elect_one_sync() ) {
                            a_gmem.load( a_smem, mem_desc_a );
                        }
                    }
                    if( warp_id == 2 ) {
                        if( xmma::elect_one_sync() ) {
                            b_gmem.load( b_smem, mem_desc_b );
                        }
                    }
                } else {
                    a_gmem.load( a_smem, mem_desc_a );
                    b_gmem.load( b_smem, mem_desc_b );
                }
                // The barrier arrive for buffer_full_barrier.
                // The buffer_full_barrier arrival is complete by updating the phase_bit
                // inside the barrier.
                // The buffer_full_barrier will wait (done by math warpgroup) until the
                // buffer_empty_barrier arrival is complete.
                // The buffer_full_barrier wait will know it (done by math warpgroup) by
                // checking whether the phase_bit is updated.
                if( Kernel_traits::TRAITS_USE_UTMALDG == 0 ) {
                    buffer_full_barriers.bar_arrive_ldgsts( buffer_head );
                }
                // Advance the cirlwlar smem buffer id to the next.
                buffer_head = ( buffer_head < Kernel_traits::BUFFERS_PER_SMEM_TILE_A - 1 )
                                  ? ( buffer_head + 1 )
                                  : 0;
                // Move the smem buffer pointer
                if( Kernel_traits::TRAITS_USE_UTMALDG ) {
                    if( warp_id == 0 ) {
                        if( xmma::elect_one_sync() ) {
                            a_smem.move_next_write_buffer( buffer_head );
                        }
                    }
                    if( warp_id == 2 ) {
                        if( xmma::elect_one_sync() ) {
                            b_smem.move_next_write_buffer( buffer_head );
                        }
                    }
                } else {
                    a_smem.move_next_write_buffer();
                    b_smem.move_next_write_buffer();
                }
                // The update of count
                cnt = cnt < ( 2 * Kernel_traits::BUFFERS_PER_SMEM_TILE_A - 1 ) ? ( cnt + 1 ) : 0;

                // Load the deltas and Update the rsi variable.
                int64_t a_delta;
                int64_t b_delta;
                trsi = Kernel_traits::load_deltas_and_move_filter_position(
                    a_delta, b_delta, params, trsi );
                // Move the A pointer
                if( Kernel_traits::TRAITS_USE_UTMALDG ) {
                    if( warp_id == 0 ) {
                        if( xmma::elect_one_sync() ) {
                            a_gmem.move( trsi, a_delta );
                        }
                    }
                    if( warp_id == 1) {
                        // TODO @aatluri: Add tma block prefetch here for operand a
                    }
                    // Move the B pointer.
                    if( warp_id == 2 ) {
                        if( xmma::elect_one_sync() ) {
                            b_gmem.move( trsi, b_delta );
                        }
                    }
                    if( warp_id == 3) {
                        // TODO @aatluri: Add tma block prefetch here for operand b
                    }
                } else {
                    a_gmem.move( trsi, a_delta );
                    b_gmem.move( trsi, b_delta );
                }
                // A and B residual handling
                if( loop <= params.loop_residue ) {
                    a_gmem.residue();
                    b_gmem.residue();
                }
            }  // (loop)
        }
    }
    //*******************Math warpgroup exelwtion region***************************
    // The operations for math warpgroups:  there are two math warpgroups
    // math warpgroup 1: warp 4~7;
    // math warpgroup 2: warp 8~11;
    else {

        // Compute the role (index) of the two math warpgroups.
        // FIXME: need to refactor Cta_tile::THREADS_PER_CTA to Cta_tile::THREADS_PER_WARPGROUP
        const int role = ( warp_id < ( Cta_tile::THREADS_PER_CTA * 2 ) / 32 ) ? 0 : 1;
        // The regular hw barrier (bar.sync) used for epilog swizzle.
        // FIXME: need to refactor Cta_tile::THREADS_PER_CTA to Cta_tile::THREADS_PER_WARPGROUP
        xmma::Named_barrier epilog_sync( ( role == 0 ? 4 : 5 ), Cta_tile::THREADS_PER_CTA );

        //**************Mutex synchornization setup**********************************
        // The setup of mutex synchronization between the two math warpgroup.
        // The mutex synchonization is implemented using bar.arv and bar.sync pair (low cost)
        // The mutex synchronization on entering the mainloop between 2 math groups.
        // math warpgroup 1:   |------loop-----|
        // math warpgroup 2:                   |-------loop---|
        // FIXME: need to refactor Cta_tile::THREADS_PER_CTA to Cta_tile::THREADS_PER_WARPGROUP
        xmma::Named_barrier loop_enter( ( role == 0 ? 0 : 1 ), Cta_tile::THREADS_PER_CTA * 2 );
        xmma::Named_barrier loop_leave( ( role == 0 ? 1 : 0 ), Cta_tile::THREADS_PER_CTA * 2 );
        // The mutex synchronization on entering the epilog between 2 math groups.
        // math warpgroup 1:   |------epilog-----|
        // math warpgroup 2:                     |-------epilog---|
        // FIXME: need to refactor Cta_tile::THREADS_PER_CTA to Cta_tile::THREADS_PER_WARPGROUP
        xmma::Named_barrier epilog_enter( ( role == 0 ? 2 : 3 ), Cta_tile::THREADS_PER_CTA * 2 );
        xmma::Named_barrier epilog_leave( ( role == 0 ? 3 : 2 ), Cta_tile::THREADS_PER_CTA * 2 );

        //*******************Persistent CTA's Tile Setup********************************
        // The number of XMMAs.
        const int XMMAS_K = Xmma_tile::XMMAS_K;
        // The thread index.
        // FIXME: need to refactor Cta_tile::THREADS_PER_CTA to Cta_tile::THREADS_PER_WARPGROUP
        const int tidx = threadIdx.x & ( Cta_tile::THREADS_PER_CTA - 1 );
        // The index (buffer id) of img (A) and flt (B) buffers
        int buffer_head = role == 0 ? 0 : params.delta_img_head;
        // The index for next buffer.
        int buffer_head_next;
        
	// Hopper CTA register reconfiguration feature to increase register count
        // for math warps as it needs more.
        // Put it here to make sure it can be exelwted a bit later than xmma::reg_dealloc() to
        // reduce latency.
        xmma::reg_alloc();
        
	// Block index for first tile in persistent CTA
        Tile_distribution_persistent tile( params, blockIdx.x + role * params.tile_move_step );
        int bidm = tile.bidm(), bidn = tile.bidn();
        int bidz = tile.bidz();


        //*******************compute_tile constructor*******************************
        // The A and B smem buffer pointers  in shared memory.
        char *a_smem_head = a_smem_ + buffer_head * Smem_tile_a::BYTES_PER_BUFFER;
        char *b_smem_head = b_smem_ + buffer_head * Smem_tile_b::BYTES_PER_BUFFER;
#ifdef USE_GMMA
        // initialize the smem pointers in compute_tile for computing next output (C) tile.
        Compute_tile compute_tile(
            a_smem_head, b_smem_head, buffer_head, Kernel_traits::BUFFERS_PER_SMEM_TILE_A );
#endif
        // The count for updating the phase_bit
        int cnt =
            role == 0
                ? 0
                : ( ( params.loop_start + 1 ) % ( 2 * Kernel_traits::BUFFERS_PER_SMEM_TILE_A ) );
        // The result of bar_peek on buffer_full_barriers, if success, bar_wait is skipped.
        bool is_wait_complete;
        // The phase_bit used in buffer_full_barriers's bar_peek and bar_wait.
        unsigned int phase_bit;
        // The first tile
        bool is_first_tile = true;
        // Iterate over the persistent cta's output tiles.
        while( !tile.is_last() ) {
            // Clear the aclwmulators.
            compute_tile.clear();

            // Math 0 and math 1 will enter mainloop and epilog via a mutex synchronization.
            // |-----mainloop-----|--Epi--|            |----------mainloop------|--Epi---|
            //                    |------mainloop------|--Epi--|

            //*******************Mutex synchronization on entering the mainloop**************
            if( role != 0 || !is_first_tile ) {
                // The mutex synchronization on entering the loop between two math warpgroup
                loop_enter.wait();
            }
            // Only enable fencing if it looks like
            // our desired stage overlap will fit in register budget.
            constexpr bool JETFIRE_FENCING_ENABLED = ( 255 * 4 > sizeof( compute_tile ) );

            //*******************Wait on 1st input buffer to be full********************
            // The barrier wait for buffer_full_barrier
            // The buffer_full_barrier will wait until the buffer_full_barrier arrival is complete
            // The buffer_full_barrier arrival (done by dma warpgroup) is complete by updating the
            // phase_bit inside the barrier
            // The buffer_full_barrier wait will know it by checking wheter the phase_bit is
            // updated.
            phase_bit = cnt < Kernel_traits::BUFFERS_PER_SMEM_TILE_A ? 0 : 1;

            if( Kernel_traits::TRAITS_USE_UTMALDG ) {
                if( threadIdx.x == Cta_tile::THREADS_PER_CTA ||
                    threadIdx.x == 2 * Cta_tile::THREADS_PER_CTA ) {
                    buffer_full_barriers.bar_arrive_set_transactioncnt( buffer_head, COPY_BYTES );
                }
            }
            buffer_full_barriers.bar_wait( buffer_head, phase_bit );

            //*******************issue some GMMA for the first kblock in prologue**************
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
            //**************************** mainloop **************************
            // Iterate over the loop on kblocks.
            JETFIRE_MAC_LOOP_PRAGMA  // Jetfire loop body
#pragma unroll 1
                for( int loop = params.loop_start; loop >= 0; --loop ) {
                JETFIRE_MAC_LOOP_HEADER
                // Disable the loads in the last iteration.
                const int is_last = loop == 0;
                // The update of count
                cnt = cnt < ( 2 * Kernel_traits::BUFFERS_PER_SMEM_TILE_A - 1 ) ? ( cnt + 1 ) : 0;
		// The input phase_bit for buffer_full_barriers to wait on
                phase_bit = cnt < Kernel_traits::BUFFERS_PER_SMEM_TILE_A ? 0 : 1;
#pragma unroll
                for( int ki_next = 1; ki_next <= XMMAS_K; ++ki_next ) {
                    // Interference fence.
                    jetfire::ifence( JETFIRE_FENCING_ENABLED );
                    // The mutex synchronization on entering the loop between two math warpgroup
                    // Put it here as early mainloop exit to prefetch for the second group.
                    if( ki_next == ( XMMAS_K - 1 ) && is_last ) {
                        loop_leave.arrive();
                    }
#ifdef USE_GMMA
                    if( ki_next <= GMMA_REMAINING_STAGE ) {
                        // issue remaining GMMAs for the current kblock
                        int gmma_stage = ki_next + Kernel_traits::GMMA_STAGES - 1;
                        if( ki_next == 1 ) {
                            // First check on the buffer_full_barrier at the beginning.
                            // It is defer-blocking until the is_wait_complete being read later.
                            // next buffer index
                            if( !is_last ) {
                                buffer_head_next =
                                    ( buffer_head < Kernel_traits::BUFFERS_PER_SMEM_TILE_A - 1 )
                                        ? ( buffer_head + 1 )
                                        : 0;

                                if( Kernel_traits::TRAITS_USE_UTMALDG ) {
                                    if( threadIdx.x == Cta_tile::THREADS_PER_CTA ||
                                        threadIdx.x == 2 * Cta_tile::THREADS_PER_CTA ) {
                                        buffer_full_barriers.bar_arrive_set_transactioncnt(
                                            buffer_head_next, COPY_BYTES );
                                    }
                                }
                                is_wait_complete =
                                    buffer_full_barriers.bar_peek( buffer_head_next, phase_bit );
                            }
                            xmma::warpgroup_arrive();
                        }
                        if( ki_next == GMMA_REMAINING_STAGE ) {
                            compute_tile.compute( gmma_stage, true, true );
                            // increment the entire desc group
                            compute_tile.increment_gmma_desc_group();
                        } else {
                            compute_tile.compute( gmma_stage, false );
                        }
                    } else {
                        if( ki_next == ( GMMA_REMAINING_STAGE + 1 ) ) {
                            xmma::warpgroup_wait<1>();
                        }
                        if( !is_last ) {
                            if( ki_next == ( GMMA_REMAINING_STAGE + 1 ) ) {
                                // The barrier wait for buffer_full_barrier.
                                // The buffer_full_barrier will wait
                                // until the buffer_full_barrier arrival is complete.
                                // The buffer_full_barrier arrival (done by dma warpgroup) is
                                // complete
                                // by updating the phase_bit inside the barrier.
                                // The buffer_full_barrier wait will know it by
                                // checking wheter the phase_bit is updated.
                                if( !is_wait_complete )
                                    buffer_full_barriers.bar_wait( buffer_head_next, phase_bit );

                                xmma::warpgroup_arrive();
                            }
                            int gmma_stage = ki_next - GMMA_REMAINING_STAGE - 1;
                            if( gmma_stage == ( Kernel_traits::GMMA_STAGES - 1 ) ) {
                                // LAST KPHASE
                                compute_tile.compute( gmma_stage, true );
                                xmma::warpgroup_wait<Kernel_traits::GMMA_STAGES>();

                                // The barrier arrive for buffer_empty_barrier.
                                // The buffer_empty_barrier arrival is complete by
                                // updating the phase_bit inside the barrier.
                                // The buffer_empty_barrier will wait (done by dma warpgroup) until
                                // the buffer_empty_barrier arrival is complete.
                                // The buffer_empty_barrier wait will know it (done by dma
                                // warpgroup)
                                // by checking whether the phase_bit is updated.
                                buffer_empty_barriers.bar_arrive_normal( buffer_head );
                                // Advance the cirlwlar smem buffer id to the next.
                                buffer_head =
                                    ( buffer_head < ( Kernel_traits::BUFFERS_PER_SMEM_TILE_A - 1 ) )
                                        ? ( buffer_head + 1 )
                                        : 0;
                            } else {
                                compute_tile.compute( gmma_stage, true );
                            }
                        }  // if(!is_last)
                    }      // if(ki_next <= GMMA_REMAINING_STAGE)
#endif
                }  // (ki)
            }      // (mainloop)

            // all GMMAs must be finished
            xmma::warpgroup_wait<0>();
            // The barrier arrive for buffer_empty_barrier.
            // The buffer_empty_barrier arrival is complete
            // by updating the phase_bit inside the barrier.
            // The buffer_empty_barrier will wait (done by dma warpgroup) until
            // the buffer_empty_barrier arrival is complete.
            // The buffer_empty_barrier wait will know it (done by dma warpgroup)
            // by checking whether the phase_bit is updated.
            buffer_empty_barriers.bar_arrive_normal( buffer_head );
            buffer_head = ( buffer_head < Kernel_traits::BUFFERS_PER_SMEM_TILE_A - 1 )
                              ? ( buffer_head + 1 )
                              : 0;

            // Update tile with a distance of 2. Tile update happens here so we know if
            // this is the last tile and need to issue PREEXIT. The second move happens
            // after PREEXIT so this only happens for the last warpgroup.
            tile.move();

            // TODO @albertx: add lwca toolkit check for release
            // Issue PREEXIT for the last tile.
            if ( tile.is_last() ) {
                __lw_ptx_builtin_ocg_preexit();
            }

            // Move a second time after the PREEXIT check.
            tile.move();

            //*******************Mutex synchronization on entering the epilog**************
            // The mutex synchronization on entering the epilog between two math warpgroup
            if( role != 0 || !is_first_tile ) {
                epilog_enter.wait();
            }

            //**************************** epilogue **************************
            // Epilogue operations
            // Do allocate the tile to output in the epilogue.
            Gmem_tile_epilogue gmem_epilogue( params, bidm, bidn, bidz, tidx );
            // Do allocate the tile and compute the offsets. TODO: Merge with the epilogue class!
            Swizzle_epilogue smem_epilogue( epi_smem_, tidx );
            // The callbacks.
            Callbacks_epilogue callbacks_epilogue( params, epi_smem_, bidm, bidn, bidz, tidx );
#ifdef USE_GMMA
            Epilogue epilogue(
                params, gmem_epilogue, smem_epilogue, callbacks_epilogue, epilog_sync );
#endif

            if( params.with_residual ) {
                epilogue.execute<true>( compute_tile.acc_ );
            } else {
                epilogue.execute<false>( compute_tile.acc_ );
            }
            // Finalize the callbacks.
            callbacks_epilogue.post_epilogue();

            // The mutex synchronization on entering the epilog between two math warpgroup
            epilog_leave.arrive();

            //**************************** States updated for next ************************
            // Update is_first_tile
            is_first_tile = false;
            // Update the cnt
            cnt = ( cnt + params.loop_start + 1 ) % ( 2 * Kernel_traits::BUFFERS_PER_SMEM_TILE_A );
            // update cirlwlar buffer index for next tile
            buffer_head = buffer_head + params.delta_img_head;
            buffer_head -= buffer_head < Kernel_traits::BUFFERS_PER_SMEM_TILE_A
                               ? 0
                               : Kernel_traits::BUFFERS_PER_SMEM_TILE_A;

            // The GMMA descriptor update for next tile
#ifdef USE_GMMA
            compute_tile.increment_N_gmma_desc_group( params.loop_start + 1 );
#endif
            // Update the bids after the epilogue from the tile update.
            bidm = tile.bidm();
            bidn = tile.bidn();
            bidz = tile.bidz();
        }
    }
#endif  // only compile sm_90 or upward
}

}  // namespace gemm
}  // namespace xmma
