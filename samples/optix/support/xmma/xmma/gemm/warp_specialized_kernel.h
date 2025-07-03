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
#include <xmma/arrive_wait.h>
#include <xmma/named_barrier.h>

#include <xmma/helpers/fragment.h>
#include <xmma/helpers/gemm.h>
#include <xmma/helpers/epilogue.h>

namespace xmma {
namespace gemm {


////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename Implicit_gemm_traits, bool Grouped_acc = false>
static __global__
__launch_bounds__( Implicit_gemm_traits::Cta_tile::THREADS_PER_CTA * 3, 1 )
void xmma_implicit_gemm_specialize_2math_1dma_arrive_wait_kernel( typename Implicit_gemm_traits::Params params ) {
#if defined( __LWDA_ARCH__ ) && __LWDA_ARCH__ >= 800
    DisableWar_SW254906_MACRO
    // The CTA tile.
    using Cta_tile = typename Implicit_gemm_traits::Cta_tile;
    // The traits class.
    using Traits = typename Implicit_gemm_traits::Traits;
    // The XMMA tile.
    using Xmma_tile = typename Implicit_gemm_traits::Xmma_tile;
    // The tile distribution manager.
    using Tile_distribution_persistent =
        typename Implicit_gemm_traits::Tile_distribution_persistent;
    // The tiles to store the data in shared memory for the images.
    using Smem_tile_a = typename Implicit_gemm_traits::Smem_tile_a;
    // The tiles to store the data in shared memory for the filters.
    using Smem_tile_b = typename Implicit_gemm_traits::Smem_tile_b;
    // The tiles to store epilog
    using Swizzle_epilogue = typename Implicit_gemm_traits::Swizzle_epilogue;
    // The compute tile
    using Compute_tile = typename Implicit_gemm_traits::Compute_tile;

    // The shared memory. It is allocated at launch time.
    extern __shared__ char smem_[];

    // arrive-wait barrier init
    uint64_t* counter = reinterpret_cast<uint64_t*>(
        &smem_[Smem_tile_a::BYTES_PER_TILE + Smem_tile_b::BYTES_PER_TILE * 1 +
               Swizzle_epilogue::BYTES_PER_TILE * 1] );
    if( threadIdx.x < ( (Implicit_gemm_traits::BUFFERS_PER_SMEM_TILE_A)*2 ) ) {  // ping-pong needs
                                                                                 // 4 barriers for
                                                                                 // synchronization
        xmma::bar_create( &counter[threadIdx.x], Cta_tile::THREADS_PER_CTA );
    }
    __syncthreads();

    // The shared memory pointers.
    char* a_smem_ = &smem_[0];
    char* b_smem_ = &smem_[Smem_tile_a::BYTES_PER_TILE];
    char* epi_smem_ = &smem_[Smem_tile_a::BYTES_PER_TILE + Smem_tile_b::BYTES_PER_TILE];

    // lane id
    int lane = 0;  // threadIdx.x & 31;

    // The barriers used for P->C communication and P is DMA warp group.
    xmma::Arrive_wait buffer_full( &counter[0], lane );
    // The barriers used for P->C communication, and P is math warp group.
    xmma::Arrive_wait buffer_empty( &counter[Implicit_gemm_traits::BUFFERS_PER_SMEM_TILE_A], lane );
    const int warp_id = __shfl_sync( 0xffffffff, threadIdx.x / 32, 0 );
    if( warp_id < Cta_tile::THREADS_PER_CTA / 32 ) {

        // xmma::reg_dealloc(64);
        // Block index for first tile in persistent model
        Tile_distribution_persistent tile( params, blockIdx.x );
        // Ampere memory descritors.
        const uint64_t mem_desc_a = params.mem_descriptors.descriptor_a;
        const uint64_t mem_desc_b = params.mem_descriptors.descriptor_b;
        //
        int bidm = tile.bidm(), bidz = tile.bidz();
        int bidn = tile.bidn();  //, bidz = tile.bidz();

        // The thread index.
        const int tidx = threadIdx.x;  //& (Cta_tile::THREADS_PER_CTA-1);
        // The index of img buffer
        int buffer_head = 0;
	// The cnt for loop iteration, used for getting the phase bit for the A/B smem buffer barrier.
	int cnt = Implicit_gemm_traits::BUFFERS_PER_SMEM_TILE_A * -1;

        // The tiles in global memory for the images and filters.
        using Gmem_tile_a = typename Implicit_gemm_traits::Gmem_tile_a;
        using Gmem_tile_b = typename Implicit_gemm_traits::Gmem_tile_b;

        // The tiles in shared memory.
        Smem_tile_a a_smem( a_smem_, tidx );
        Smem_tile_b b_smem( b_smem_, tidx );

        if( Gmem_tile_a::USE_UTMALDG && Gmem_tile_b::USE_UTMALDG ) {
            a_smem.add_smem_barrier_base( counter );
            b_smem.add_smem_barrier_base( counter );
        }

        unsigned int phase_bit;

        while( !tile.is_last() ) {
            // Load from global memory.
            Gmem_tile_a a_gmem( params, NULL, tile.bidx(), tidx );
            // Load from global memory.
            Gmem_tile_b b_gmem( params, NULL, tile.bidx(), tidx );
            int trsi = Implicit_gemm_traits::initialize_filter_position( params );
            // update tile
            tile.move();
            bidm = tile.bidm();
            bidz = tile.bidz();
            bidn = tile.bidn();
// Iterate over the loop.
#pragma unroll 1
            for( int loop = params.loop_start; loop >= 0; --loop ) {
                if( cnt >= 0 ) {
                    // wait barrier
                    phase_bit = cnt <  Implicit_gemm_traits::BUFFERS_PER_SMEM_TILE_A ? 0 : 1;
                    buffer_empty.bar_wait( buffer_head, phase_bit );
                }
                a_gmem.load( a_smem, mem_desc_a );
                b_gmem.load( b_smem, mem_desc_b );
                // async copy arrive barrier
                buffer_full.bar_arrive_ldgsts( buffer_head );
                buffer_head = ( buffer_head < Implicit_gemm_traits::BUFFERS_PER_SMEM_TILE_A - 1 )
                                  ? ( buffer_head + 1 )
                                  : 0;
                a_smem.move_next_write_buffer();
                b_smem.move_next_write_buffer();

                cnt = cnt < (2 *  Implicit_gemm_traits::BUFFERS_PER_SMEM_TILE_A -1) ? (cnt +1) : 0 ;

                // Load the deltas and Update the rsi variable.
                int64_t a_delta;
                int64_t b_delta;
                trsi = Implicit_gemm_traits::load_deltas_and_move_filter_position(
                    a_delta, b_delta, params, trsi );
                a_gmem.move( trsi, a_delta );
                // Move the filter pointer.
                b_gmem.move( trsi, b_delta );

                if( loop <= params.loop_residue ) {
                    a_gmem.residue();
                    b_gmem.residue();
                }
            }  // (loop)
        }
    } else {
        // xmma::reg_alloc(224);
        // Compute the role of the two warp groups.
        const int role = ( warp_id < Cta_tile::THREADS_PER_CTA * 2 / 32 ) ? 0 : 1;
        // The named barrier used for epilog swizzle.
        xmma::Named_barrier epilog_sync( ( role == 0 ? 2 : 3 ), Cta_tile::THREADS_PER_CTA );
        // The barriers used for ping-pong communication between 2 math groups.
        xmma::Named_barrier loop_enter( ( role == 0 ? 0 : 1 ), Cta_tile::THREADS_PER_CTA * 2 );
        xmma::Named_barrier loop_leave( ( role == 0 ? 1 : 0 ), Cta_tile::THREADS_PER_CTA * 2 );
        xmma::Named_barrier epilog_enter( ( role == 0 ? 2 : 3 ), Cta_tile::THREADS_PER_CTA * 2 );
        xmma::Named_barrier epilog_leave( ( role == 0 ? 3 : 2 ), Cta_tile::THREADS_PER_CTA * 2 );
        // The number of XMMAs.
        const int XMMAS_K = Xmma_tile::XMMAS_K;

        // The thread index.
        const int tidx = threadIdx.x & ( Cta_tile::THREADS_PER_CTA - 1 );
        // The index of img and flt buffers
        int buffer_head = role == 0 ? 0 : params.delta_img_head;
        // Block index for first tile in persistent model
        Tile_distribution_persistent tile( params, blockIdx.x + role * params.tile_move_step );
        int bidm = tile.bidm(), bidn = tile.bidn();
        int bidz = tile.bidz();

        // The tile in global memory.
        using Gmem_tile_epilogue = typename Implicit_gemm_traits::Gmem_tile_epilogue;
        // The callbacks.
        using Callbacks_epilogue = typename Implicit_gemm_traits::Callbacks_epilogue;
        // The epilogue without splitk.
        using Epilogue_wosplitk = typename Implicit_gemm_traits::Epilogue_wosplitk;
        // The epilogue without splitk.
        using Epilogue_withsplitk = typename Implicit_gemm_traits::Epilogue_withsplitk;
        // The tiles in shared memory.

        Smem_tile_a a_smem( a_smem_, tidx );
        Smem_tile_b b_smem( b_smem_, tidx );

	// The count for loop iteration
        int cnt = role == 0 ? 0 : ((params.loop_start +1) % ( 2 *  Implicit_gemm_traits::BUFFERS_PER_SMEM_TILE_A));
        bool is_wait_complete;
        unsigned int phase_bit;
	// The first iteration
	bool  is_first_iteration = true;
        while( !tile.is_last() ) {
            // Clear the aclwmulators.
            Compute_tile compute_tile;
            compute_tile.clear();
            // Reverse smem_read_offset for last tile's last smem.load().
            if( !is_first_iteration ) {
                a_smem.reverse_smem_read_offset();
                b_smem.reverse_smem_read_offset();
            }

            if( role != 0 || !is_first_iteration ) {
                a_smem.move_next_read_buffer( params.delta_img_head );
                b_smem.move_next_read_buffer( params.delta_flt_head );
                // wait barrier
                loop_enter.wait();
            }
            // Only enable fencing if it looks like our desired stage overlap will fit in register
            // budget.
            constexpr bool JETFIRE_FENCING_ENABLED = ( 255 * 4 > sizeof( compute_tile ) );

            // wait barrier
            phase_bit = cnt < Implicit_gemm_traits::BUFFERS_PER_SMEM_TILE_A ? 0:1;
            buffer_full.bar_wait( buffer_head, phase_bit );

            // Load the image and filter pixels.
            compute_tile.load( a_smem, b_smem, 0, true );

            // Iterate over the loop.
            JETFIRE_MAC_LOOP_PRAGMA  // Jetfire loop body
#pragma unroll 1
                for( int loop = params.loop_start; loop >= 0; --loop ) {
                JETFIRE_MAC_LOOP_HEADER
                // Disable the loads in the last iteration.
                const int is_last = loop == 0;
		cnt = cnt < (2* Implicit_gemm_traits::BUFFERS_PER_SMEM_TILE_A -1) ? (cnt+1) : 0;

#pragma unroll
                for( int ki = 1; ki <= XMMAS_K; ++ki ) {

                    int KI = ( ki == XMMAS_K ) ? 0 : ki;
                    // Interference fence.
                    jetfire::ifence( JETFIRE_FENCING_ENABLED );

                    if( ki == XMMAS_K ) {
                        a_smem.move_next_read_buffer();
                        b_smem.move_next_read_buffer();
                        // Load the image pixels from shared memory.
                        if( !is_last ) {
                            // Wait barrier
                            if( !is_wait_complete )
                                buffer_full.bar_wait( buffer_head, phase_bit );
                        }
                    }
                    // Load the tile A and B from shared memory buffer.
                    compute_tile.load( a_smem, b_smem, KI );

                    // Early mainloop exit to prefetch for the second group.
                    if ( ki == (XMMAS_K -1)  && is_last ) {
                        loop_leave.arrive();
                    }

                    // Barrier  Arrival after SMEM load complete
                    if( ki == XMMAS_K - 1 ) {
                        // thread arrive barrier
                        buffer_empty.bar_arrive_normal( buffer_head );
                        buffer_head =
                            ( buffer_head < Implicit_gemm_traits::BUFFERS_PER_SMEM_TILE_A - 1 )
                                ? ( buffer_head + 1 )
                                : 0;
                    }

                    // Interference fence.
                    jetfire::ifence( JETFIRE_FENCING_ENABLED );

                    // Do the math - The core of the loop does 16x16x8.
                    compute_tile.compute( ki );

                    if( ki == XMMAS_K - 1 ) {
                        // wait barrier peek
                        phase_bit = cnt < Implicit_gemm_traits::BUFFERS_PER_SMEM_TILE_A ? 0:1;
                        is_wait_complete = buffer_full.bar_peek( buffer_head, phase_bit );
                    }
                }  // (ki)
            }  // (loop)
            if( role != 0 || !is_first_iteration ) {
                // wait barrier
                epilog_enter.wait();
            }
	    is_first_iteration = false;
	    cnt = (cnt + params.loop_start +1 ) % (2 * Implicit_gemm_traits::BUFFERS_PER_SMEM_TILE_A);
            // Do allocate the tile to output in the epilogue.
            Gmem_tile_epilogue gmem_epilogue( params, bidm, bidn, bidz, tidx );
            // Do allocate the tile and compute the offsets. TODO: Merge with the epilogue class!
            Swizzle_epilogue smem_epilogue( epi_smem_, tidx );
            // The callbacks.
            Callbacks_epilogue callbacks_epilogue( params, epi_smem_, bidm, bidn, bidz, tidx );

            // without splik
            if( params.split_k.slices == 1 ) {
                // Do the epilogue.
                Epilogue_wosplitk epilogue( params,
                                            gmem_epilogue,
                                            smem_epilogue,
                                            callbacks_epilogue,
                                            epilog_sync,
                                            params.use_horizontal_cta_rasterization ? bidn : bidm,
                                            params.use_horizontal_cta_rasterization ? bidm : bidn,
                                            bidz,
                                            tidx,
                                            Implicit_gemm_traits::USE_WARP_SPECIALIZATION );

                if( params.with_residual ) {
                    epilogue.execute<true>( compute_tile.acc_ );
                } else {
                    epilogue.execute<false>( compute_tile.acc_ );
                }
            } else {
                // with splitk
                // Do the epilogue.
                Epilogue_withsplitk epilogue( params,
                                              gmem_epilogue,
                                              smem_epilogue,
                                              callbacks_epilogue,
                                              epilog_sync,
                                              params.use_horizontal_cta_rasterization ? bidn : bidm,
                                              params.use_horizontal_cta_rasterization ? bidm : bidn,
                                              bidz,
                                              tidx,
                                              Implicit_gemm_traits::USE_WARP_SPECIALIZATION );

                if( params.with_residual ) {
                    epilogue.execute<true>( compute_tile.acc_ );
                } else {
                    epilogue.execute<false>( compute_tile.acc_ );
                }
            }

            // Finalize the callbacks.
            callbacks_epilogue.post_epilogue();
            // update buffer_head for next tile
            buffer_head = buffer_head + params.delta_img_head;
            buffer_head -= buffer_head < Implicit_gemm_traits::BUFFERS_PER_SMEM_TILE_A
                               ? 0
                               : Implicit_gemm_traits::BUFFERS_PER_SMEM_TILE_A;
            epilog_leave.arrive();

            // Update tile with a distance of 2
            tile.move();
            tile.move();
            bidm = tile.bidm();
            bidn = tile.bidn();
            bidz = tile.bidz();
        }
    }
#endif  // only compile sm_80 or upward
}

////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename Implicit_gemm_traits, bool Grouped_acc = false>
static __global__
__launch_bounds__( Implicit_gemm_traits::Cta_tile::THREADS_PER_CTA * 2, 1 )
void xmma_implicit_gemm_specialize_1math_1dma_arrive_wait_kernel( typename Implicit_gemm_traits::Params params ) {
#if defined( __LWDA_ARCH__ ) && __LWDA_ARCH__ >= 800
    DisableWar_SW254906_MACRO
    // The CTA tile.
    using Cta_tile = typename Implicit_gemm_traits::Cta_tile;
    // The traits class.
    using Traits = typename Implicit_gemm_traits::Traits;
    // The XMMA tile.
    using Xmma_tile = typename Implicit_gemm_traits::Xmma_tile;
    // The tile distribution manager.
    using Tile_distribution_persistent =
        typename Implicit_gemm_traits::Tile_distribution_persistent;
    // The tiles to store the data in shared memory for the images.
    using Smem_tile_a = typename Implicit_gemm_traits::Smem_tile_a;
    // The tiles to store the data in shared memory for the filters.
    using Smem_tile_b = typename Implicit_gemm_traits::Smem_tile_b;
    // The tiles to store epilog
    using Swizzle_epilogue = typename Implicit_gemm_traits::Swizzle_epilogue;
    // The compute tile
    using Compute_tile = typename Implicit_gemm_traits::Compute_tile;

    // The shared memory. It is allocated at launch time.
    extern __shared__ char smem_[];

    // arrive-wait barrier init
    uint64_t* counter = reinterpret_cast<uint64_t*>(
        &smem_[Smem_tile_a::BYTES_PER_TILE + Smem_tile_b::BYTES_PER_TILE * 1 +
               Swizzle_epilogue::BYTES_PER_TILE * 1] );
    if( threadIdx.x < ( (Implicit_gemm_traits::BUFFERS_PER_SMEM_TILE_A)*2 ) ) {
        xmma::bar_create( &counter[threadIdx.x], Cta_tile::THREADS_PER_CTA );
    }
    __syncthreads();

    // The shared memory pointers.
    char* a_smem_ = &smem_[0];
    char* b_smem_ = &smem_[Smem_tile_a::BYTES_PER_TILE];
    char* epi_smem_ = &smem_[Smem_tile_a::BYTES_PER_TILE + Smem_tile_b::BYTES_PER_TILE];
    // store one barrier using 1-bit in int32, the buffer_head indicates the bit position (start
    // from LSB).
    unsigned int lwrrent_phase_buffer_full =
        0;  // total bit needs in int32 is Implicit_gemm_traits::BUFFERS_PER_SMEM_TILE_A
    unsigned int lwrrent_phase_buffer_empty =
        0;  // total bit needs in int32 is Implicit_gemm_traits::BUFFERS_PER_SMEM_TILE_A
    // lane id
    int lane = 0;  // threadIdx.x & 31;

    // The barriers used for P->C communication and P is DMA warp group.
    xmma::Arrive_wait buffer_full( &counter[0], lane );
    // The barriers used for P->C communication, and P is math warp group.
    xmma::Arrive_wait buffer_empty( &counter[Implicit_gemm_traits::BUFFERS_PER_SMEM_TILE_A], lane );
    // The named barrier used for epilog swizzle.
    // Note: The named barrier is set to 1 as a WAR. See CFK-2789 for details.
    xmma::Named_barrier epilog_sync[1] = {
        { 1, Cta_tile::THREADS_PER_CTA },
    };

    bool use_early_loading = true;

    const int warp_id = __shfl_sync( 0xffffffff, threadIdx.x / 32, 0 );
    if( warp_id < Cta_tile::THREADS_PER_CTA / 32 ) {// dma warps


        // Block index for first tile in persistent model
        Tile_distribution_persistent tile( params, blockIdx.x );
        // Ampere memory descritors.
        const uint64_t mem_desc_a = params.mem_descriptors.descriptor_a;
        const uint64_t mem_desc_b = params.mem_descriptors.descriptor_b;
        //
        int bidm = tile.bidm(), bidz = tile.bidz();
        int bidn = tile.bidn();  //, bidz = tile.bidz();

        // The thread index.
        const int tidx = threadIdx.x;  //& (Cta_tile::THREADS_PER_CTA-1);
        // The index of img buffer
        int buffer_head = 0, cnt = 0;

        // The tiles in global memory for the images and filters.
        using Gmem_tile_a = typename Implicit_gemm_traits::Gmem_tile_a;
        using Gmem_tile_b = typename Implicit_gemm_traits::Gmem_tile_b;

        // The tiles in shared memory.
        Smem_tile_a a_smem( a_smem_, tidx );
        Smem_tile_b b_smem( b_smem_, tidx );
        unsigned int phase_bit;

        while( !tile.is_last() ) {//loop through no. of waves
            // Load from global memory.
            Gmem_tile_a a_gmem( params, NULL, tile.bidx(), tidx );
            // Load from global memory.
            Gmem_tile_b b_gmem( params, NULL, tile.bidx(), tidx );
            int trsi = Implicit_gemm_traits::initialize_filter_position( params );

// Iterate over the loop.
#pragma unroll 1
            for( int loop = params.loop_start; loop >= 0; --loop ) {//loop through no. of kBlocks
                if( cnt >= Implicit_gemm_traits::BUFFERS_PER_SMEM_TILE_A ) {
                    // wait barrier
                    phase_bit = ( lwrrent_phase_buffer_empty >> buffer_head ) & 1;
                    buffer_empty.bar_wait( buffer_head, phase_bit );
                    lwrrent_phase_buffer_empty ^= ( 1 << buffer_head ) ^ ( 0 );
                }
                a_gmem.load( a_smem, mem_desc_a );
                b_gmem.load( b_smem, mem_desc_b );
                // async copy arrive barrier
                buffer_full.bar_arrive_ldgsts( buffer_head );
                buffer_head = ( buffer_head < Implicit_gemm_traits::BUFFERS_PER_SMEM_TILE_A - 1 )
                                  ? ( buffer_head + 1 )
                                  : 0;
                a_smem.move_next_write_buffer();
                b_smem.move_next_write_buffer();

                cnt++;

                // Load the deltas and Update the rsi variable.
                int64_t a_delta;
                int64_t b_delta;
                trsi = Implicit_gemm_traits::load_deltas_and_move_filter_position(
                    a_delta, b_delta, params, trsi );
                a_gmem.move( trsi, a_delta );
                // Move the filter pointer.
                b_gmem.move( trsi, b_delta );

                if( loop <= params.loop_residue ) {
                    a_gmem.residue();
                    b_gmem.residue();
                }
            }  // (loop)

            // loading entire C from gmem to smem
            // Lwrrently early loading only works with no split k
            // TODO: Implement early loading for split-k
            if( params.with_residual && use_early_loading && params.split_k.slices == 1 && !params.batch.is_batched )
            {
                using Gmem_tile_epilogue = typename Implicit_gemm_traits::Gmem_tile_epilogue;
                Gmem_tile_epilogue gmem_epilogue( params, bidm, bidn, bidz, tidx );


                xmma::helpers::warp_specialized_early_load<Implicit_gemm_traits>(
                                            gmem_epilogue,
                                            a_smem,
                                            b_smem,
                                            buffer_head,
                                            buffer_empty,
                                            buffer_full,
                                            cnt,
                                            lwrrent_phase_buffer_empty,
                                            tidx,
                                            params.mem_descriptors.descriptor_c);
            } // loading c
            // update tile
            tile.move();
            bidm = tile.bidm();
            bidz = tile.bidz();
            bidn = tile.bidn();
        }// while( !tile.is_last() )

    } else { //math

        // The number of XMMAs.
        const int XMMAS_K = Xmma_tile::XMMAS_K;

        // The thread index.
        const int tidx = threadIdx.x & ( Cta_tile::THREADS_PER_CTA - 1 );
        // The index of img and flt buffers
        int buffer_head = 0;

        // Block index for first tile in persistent model
        Tile_distribution_persistent tile( params, blockIdx.x );
        int bidm = tile.bidm(), bidn = tile.bidn();
        int bidz = tile.bidz();

        // The tile in global memory.
        using Gmem_tile_epilogue = typename Implicit_gemm_traits::Gmem_tile_epilogue;
        // The callbacks.
        using Callbacks_epilogue = typename Implicit_gemm_traits::Callbacks_epilogue;
        // The epilogue without splitk.
        using Epilogue_wosplitk = typename Implicit_gemm_traits::Epilogue_wosplitk;
        // The epilogue without splitk.
        using Epilogue_withsplitk = typename Implicit_gemm_traits::Epilogue_withsplitk;

        // The tiles in shared memory.

        Smem_tile_a a_smem( a_smem_, tidx );
        Smem_tile_b b_smem( b_smem_, tidx );

        int cnt = 0;
        bool is_wait_complete = 0;
        unsigned int phase_bit;
        while( !tile.is_last() ) {
            // Clear the aclwmulators.
            Compute_tile compute_tile;
            compute_tile.clear();

            // Only enable fencing if it looks like our desired stage overlap will fit in register
            // budget.
            constexpr bool JETFIRE_FENCING_ENABLED = ( 255 * 4 > sizeof( compute_tile ) );

            // Load smem if not using early loading and the previous tile's
            // prefetch was  unsuccessful
            if ( params.with_residual || !is_wait_complete ) {
                // Reverse smem_read_offset for last tile's last smem.load().
                if( cnt > 0 ) {
                    a_smem.reverse_smem_read_offset();
                    b_smem.reverse_smem_read_offset();
                }
                // wait barrier
                phase_bit = ( lwrrent_phase_buffer_full >> buffer_head ) & 1;
                buffer_full.bar_wait( buffer_head, phase_bit );

                // Load the image and filter pixels.
                compute_tile.load( a_smem, b_smem, 0, true );
            }

            lwrrent_phase_buffer_full ^= ( 1 << buffer_head ) ^ ( 0 );

            // Iterate over the loop.
            JETFIRE_MAC_LOOP_PRAGMA  // Jetfire loop body
#pragma unroll 1
                for( int loop = params.loop_start; loop >= 0; --loop ) {
                JETFIRE_MAC_LOOP_HEADER
                // Disable the loads in the last iteration.
                const int is_last = loop == 0;

#pragma unroll
                for( int ki = 1; ki <= XMMAS_K; ++ki ) {

                    int KI = ( ki == XMMAS_K ) ? 0 : ki;
                    // Interference fence.
                    jetfire::ifence( JETFIRE_FENCING_ENABLED );

                    if( ki == XMMAS_K ) {

                        a_smem.move_next_read_buffer();
                        b_smem.move_next_read_buffer();

                        // Load the image pixels from shared memory.
                        if( !is_last ) {
                            // Wait barrier
                            if( !is_wait_complete )
                                buffer_full.bar_wait( buffer_head, phase_bit );
                        }
                    }
                    // Load the tile A and B from shared memory buffer.
                    compute_tile.load( a_smem, b_smem, KI );

                    // Barrier  Arrival after SMEM load complete
                    if( ki == XMMAS_K - 1 ) {
                        // thread arrive barrier
                        buffer_empty.bar_arrive_normal( buffer_head );
                        buffer_head =
                            ( buffer_head < Implicit_gemm_traits::BUFFERS_PER_SMEM_TILE_A - 1 )
                                ? ( buffer_head + 1 )
                                : 0;
                    }

                    // Interference fence.
                    jetfire::ifence( JETFIRE_FENCING_ENABLED );

                    // Do the math - The core of the loop does 16x16x8.
                    compute_tile.compute( ki );

                    if( ki == XMMAS_K - 1 ) {
                        // wait barrier peek
                        phase_bit = ( lwrrent_phase_buffer_full >> buffer_head ) & 1;
                        is_wait_complete = buffer_full.bar_peek( buffer_head, phase_bit );
                    }
                }  // (ki)
                // Phase update
                if( !is_last ) {
                    lwrrent_phase_buffer_full ^= ( 1 << buffer_head ) ^ ( 0 );
                }

            }  // Main loop

            // Do allocate the tile to output in the epilogue.
            Gmem_tile_epilogue gmem_epilogue( params, bidm, bidn, bidz, tidx );
            // Do allocate the tile and compute the offsets. TODO: Merge with the epilogue class!
            Swizzle_epilogue smem_epilogue( epi_smem_, tidx );
            // The callbacks.
            Callbacks_epilogue callbacks_epilogue( params, epi_smem_, bidm, bidn, bidz, tidx );
            // without splik
            if( params.split_k.slices == 1 && !params.batch.is_batched ) {
                // Do the epilogue.
                Epilogue_wosplitk epilogue( params,
                                            gmem_epilogue,
                                            smem_epilogue,
                                            callbacks_epilogue,
                                            epilog_sync[0],
                                            params.use_horizontal_cta_rasterization ? bidn : bidm,
                                            params.use_horizontal_cta_rasterization ? bidm : bidn,
                                            bidz,
                                            tidx,
                                            Implicit_gemm_traits::USE_WARP_SPECIALIZATION );

                if( params.with_residual ) {

                    if (use_early_loading) {
                      //Execute epilogue, fetching C from smem buffers
                      epilogue.execute<true>( compute_tile.acc_ ,
                                              a_smem,
                                              b_smem,
                                              buffer_head,
                                              buffer_empty,
                                              buffer_full,
                                              lwrrent_phase_buffer_full,
                                              Implicit_gemm_traits::BUFFERS_PER_SMEM_TILE_A);
                    } else {
                      epilogue.execute<true>( compute_tile.acc_);
                    }

                } else {
                    epilogue.execute<false>( compute_tile.acc_ );
                }
            } else {
                // with splitk
                // Do the epilogue.
                Epilogue_withsplitk epilogue( params,
                                              gmem_epilogue,
                                              smem_epilogue,
                                              callbacks_epilogue,
                                              epilog_sync[0],
                                              params.use_horizontal_cta_rasterization ? bidn : bidm,
                                              params.use_horizontal_cta_rasterization ? bidm : bidn,
                                              bidz,
                                              tidx,
                                              Implicit_gemm_traits::USE_WARP_SPECIALIZATION );

                if( params.with_residual ) {
                    epilogue.execute<true>( compute_tile.acc_ );
                    //TODO:
                    //implement early loading with split_k
                } else {
                    epilogue.execute<false>( compute_tile.acc_ );
                }
            }
            // Finalize the callbacks.
            callbacks_epilogue.post_epilogue();

            // Update tile
            tile.move();
            bidm = tile.bidm();
            bidn = tile.bidn();
            bidz = tile.bidz();

            cnt++;
        }
    }
#endif  // only compile sm_80 or upward
}

////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename Implicit_gemm_traits, bool Grouped_acc = false>
static __global__
__launch_bounds__( Implicit_gemm_traits::Cta_tile::THREADS_PER_CTA * 2, 1 )
void xmma_implicit_gemm_specialize_1math_1dma_arrive_wait_kernel_tma( typename Implicit_gemm_traits::Params params ) {
#if defined( __LWDA_ARCH__ ) && __LWDA_ARCH__ >= 800
    // The CTA tile.
    using Cta_tile = typename Implicit_gemm_traits::Cta_tile;
    // The traits class.
    using Traits = typename Implicit_gemm_traits::Traits;
    // The XMMA tile.
    using Xmma_tile = typename Implicit_gemm_traits::Xmma_tile;
    // The tile distribution manager.
    using Tile_distribution_persistent =
        typename Implicit_gemm_traits::Tile_distribution_persistent;
    // The tiles to store the data in shared memory for the images.
    using Smem_tile_a = typename Implicit_gemm_traits::Smem_tile_a;
    // The tiles to store the data in shared memory for the filters.
    using Smem_tile_b = typename Implicit_gemm_traits::Smem_tile_b;
    // The tiles to store epilog
    using Swizzle_epilogue = typename Implicit_gemm_traits::Swizzle_epilogue;
    // The compute tile
    using Compute_tile = typename Implicit_gemm_traits::Compute_tile;

    // The shared memory. It is allocated at launch time.
    extern __shared__ char smem_[];

    // arrive-wait barrier init
    uint64_t* counter = reinterpret_cast<uint64_t*>(
        &smem_[Smem_tile_a::BYTES_PER_TILE + Smem_tile_b::BYTES_PER_TILE * 1 +
               Swizzle_epilogue::BYTES_PER_TILE * 1] );
    if( threadIdx.x < ( (Implicit_gemm_traits::BUFFERS_PER_SMEM_TILE_A)*2 ) ) {
        xmma::bar_create( &counter[threadIdx.x], Cta_tile::THREADS_PER_CTA );
    }
    __syncthreads();

    // The shared memory pointers.
    char* a_smem_ = &smem_[0];
    char* b_smem_ = &smem_[Smem_tile_a::BYTES_PER_TILE];
    char* epi_smem_ = &smem_[Smem_tile_a::BYTES_PER_TILE + Smem_tile_b::BYTES_PER_TILE];
    // store one barrier using 1-bit in int32, the buffer_head indicates the bit position (start
    // from LSB).
    unsigned int lwrrent_phase_buffer_full =
        0;  // total bit needs in int32 is Implicit_gemm_traits::BUFFERS_PER_SMEM_TILE_A
    unsigned int lwrrent_phase_buffer_empty =
        0;  // total bit needs in int32 is Implicit_gemm_traits::BUFFERS_PER_SMEM_TILE_A
    // lane id
    int lane = 0;  // threadIdx.x & 31;

    // The barriers used for P->C communication and P is DMA warp group.
    xmma::Arrive_wait buffer_full( &counter[0], lane );
    // The barriers used for P->C communication, and P is math warp group.
    xmma::Arrive_wait buffer_empty( &counter[Implicit_gemm_traits::BUFFERS_PER_SMEM_TILE_A], lane );
    // The named barrier used for epilog swizzle.
    xmma::Named_barrier epilog_sync[1] = {
        { 0, Cta_tile::THREADS_PER_CTA },
    };

    const int warp_id = __shfl_sync( 0xffffffff, threadIdx.x / 32, 0 );
    if( warp_id < Cta_tile::THREADS_PER_CTA / 32 ) {

        // Block index for first tile in persistent model
        Tile_distribution_persistent tile( params, blockIdx.x );
        // Ampere memory descritors.
        const uint64_t mem_desc_a = params.mem_descriptors.descriptor_a;
        const uint64_t mem_desc_b = params.mem_descriptors.descriptor_b;
        //
        int bidm = tile.bidm(), bidz = tile.bidz();
        int bidn = tile.bidn();  //, bidz = tile.bidz();

        // The thread index.
        const int tidx = threadIdx.x;  //& (Cta_tile::THREADS_PER_CTA-1);
        // The index of img buffer
        int buffer_head = 0, cnt = 0;

        // The tiles in global memory for the images and filters.
        using Gmem_tile_a = typename Implicit_gemm_traits::Gmem_tile_a;
        using Gmem_tile_b = typename Implicit_gemm_traits::Gmem_tile_b;

        // The tiles in shared memory.
        Smem_tile_a a_smem( a_smem_, tidx );
        Smem_tile_b b_smem( b_smem_, tidx );
        unsigned int phase_bit;

        while( !tile.is_last() ) {
            // Load from global memory.
            Gmem_tile_a a_gmem( params, &params.a_desc, NULL, tile.bidx(), tidx );
            // Load from global memory.
            Gmem_tile_b b_gmem( params, &params.b_desc, NULL, tile.bidx(), tidx );
            int trsi = Implicit_gemm_traits::initialize_filter_position( params );
            // update tile
            tile.move();
            bidm = tile.bidm();
            bidz = tile.bidz();
            bidn = tile.bidn();
// Iterate over the loop.
#pragma unroll 1
            for( int loop = params.loop_start; loop >= 0; --loop ) {
                if( cnt >= Implicit_gemm_traits::BUFFERS_PER_SMEM_TILE_A ) {
                    // wait barrier
                    phase_bit = ( lwrrent_phase_buffer_empty >> buffer_head ) & 1;
                    buffer_empty.bar_wait( buffer_head, phase_bit );
                    lwrrent_phase_buffer_empty ^= ( 1 << buffer_head ) ^ ( 0 );
                }
                a_gmem.load( a_smem, mem_desc_a );
                b_gmem.load( b_smem, mem_desc_b );
                // async copy arrive barrier
                buffer_full.bar_arrive_ldgsts( buffer_head );
                buffer_head = ( buffer_head < Implicit_gemm_traits::BUFFERS_PER_SMEM_TILE_A - 1 )
                                  ? ( buffer_head + 1 )
                                  : 0;
                a_smem.move_next_write_buffer();
                b_smem.move_next_write_buffer();

                cnt++;

                // Load the deltas and Update the rsi variable.
                int64_t a_delta;
                int64_t b_delta;
                trsi = Implicit_gemm_traits::load_deltas_and_move_filter_position(
                    a_delta, b_delta, params, trsi );
                a_gmem.move( trsi, a_delta );
                // Move the filter pointer.
                b_gmem.move( trsi, b_delta );

                if( loop <= params.loop_residue ) {
                    a_gmem.residue();
                    b_gmem.residue();
                }
            }  // (loop)
        }
    } else {

        // The number of XMMAs.
        const int XMMAS_K = Xmma_tile::XMMAS_K;

        // The thread index.
        const int tidx = threadIdx.x & ( Cta_tile::THREADS_PER_CTA - 1 );
        // The index of img and flt buffers
        int buffer_head = 0;

        // Block index for first tile in persistent model
        Tile_distribution_persistent tile( params, blockIdx.x );
        int bidm = tile.bidm(), bidn = tile.bidn();
        int bidz = tile.bidz();

        // The tile in global memory.
        using Gmem_tile_epilogue = typename Implicit_gemm_traits::Gmem_tile_epilogue;
        // The callbacks.
        using Callbacks_epilogue = typename Implicit_gemm_traits::Callbacks_epilogue;
        // The epilogue without splitk.
        using Epilogue_wosplitk = typename Implicit_gemm_traits::Epilogue_wosplitk;
        // The epilogue without splitk.
        using Epilogue_withsplitk = typename Implicit_gemm_traits::Epilogue_withsplitk;

        // The tiles in shared memory.

        Smem_tile_a a_smem( a_smem_, tidx );
        Smem_tile_b b_smem( b_smem_, tidx );

        int cnt = 0;
        bool is_wait_complete;
        unsigned int phase_bit;
        while( !tile.is_last() ) {
            // Clear the aclwmulators.
            Compute_tile compute_tile;
            compute_tile.clear();

            // Only enable fencing if it looks like our desired stage overlap will fit in register
            // budget.
            constexpr bool JETFIRE_FENCING_ENABLED = ( 255 * 4 > sizeof( compute_tile ) );

            // Reverse smem_read_offset for last tile's last smem.load().
            if( cnt > 0 ) {
                a_smem.reverse_smem_read_offset();
                b_smem.reverse_smem_read_offset();
            }
            // wait barrier
            phase_bit = ( lwrrent_phase_buffer_full >> buffer_head ) & 1;
            buffer_full.bar_wait( buffer_head, phase_bit );
            lwrrent_phase_buffer_full ^= ( 1 << buffer_head ) ^ ( 0 );

            // Load the image and filter pixels.
            compute_tile.load( a_smem, b_smem, 0, true );

            // Iterate over the loop.
            JETFIRE_MAC_LOOP_PRAGMA  // Jetfire loop body
#pragma unroll 1
                for( int loop = params.loop_start; loop >= 0; --loop ) {
                JETFIRE_MAC_LOOP_HEADER
                // Disable the loads in the last iteration.
                const int is_last = loop == 0;

#pragma unroll
                for( int ki = 1; ki <= XMMAS_K; ++ki ) {

                    int KI = ( ki == XMMAS_K ) ? 0 : ki;
                    // Interference fence.
                    jetfire::ifence( JETFIRE_FENCING_ENABLED );

                    if( ki == XMMAS_K ) {

                        a_smem.move_next_read_buffer();
                        b_smem.move_next_read_buffer();

                        // Load the image pixels from shared memory.
                        if( !is_last ) {
                            // Wait barrier
                            if( !is_wait_complete )
                                buffer_full.bar_wait( buffer_head, phase_bit );
                        }
                    }
                    // Load the tile A and B from shared memory buffer.
                    compute_tile.load( a_smem, b_smem, KI );

                    // Barrier  Arrival after SMEM load complete
                    if( ki == XMMAS_K - 1 ) {
                        // thread arrive barrier
                        buffer_empty.bar_arrive_normal( buffer_head );
                        buffer_head =
                            ( buffer_head < Implicit_gemm_traits::BUFFERS_PER_SMEM_TILE_A - 1 )
                                ? ( buffer_head + 1 )
                                : 0;
                    }

                    // Interference fence.
                    jetfire::ifence( JETFIRE_FENCING_ENABLED );

                    // Do the math - The core of the loop does 16x16x8.
                    compute_tile.compute( ki );

                    if( ki == XMMAS_K - 1 ) {
                        // wait barrier peek
                        phase_bit = ( lwrrent_phase_buffer_full >> buffer_head ) & 1;
                        is_wait_complete = buffer_full.bar_peek( buffer_head, phase_bit );
                    }
                }  // (ki)
                // Phase update
                if( !is_last ) {
                    lwrrent_phase_buffer_full ^= ( 1 << buffer_head ) ^ ( 0 );
                }

            }  // (loop)

            // Do allocate the tile to output in the epilogue.
            Gmem_tile_epilogue gmem_epilogue( params, bidm, bidn, bidz, tidx );
            // Do allocate the tile and compute the offsets. TODO: Merge with the epilogue class!
            Swizzle_epilogue smem_epilogue( epi_smem_, tidx );
            // The callbacks.
            Callbacks_epilogue callbacks_epilogue( params, epi_smem_, bidm, bidn, bidz, tidx );
            // without splik
            if( params.split_k.slices == 1 ) {
                // Do the epilogue.
                Epilogue_wosplitk epilogue( params,
                                            gmem_epilogue,
                                            smem_epilogue,
                                            callbacks_epilogue,
                                            epilog_sync[0],
                                            params.use_horizontal_cta_rasterization ? bidn : bidm,
                                            params.use_horizontal_cta_rasterization ? bidm : bidn,
                                            bidz,
                                            tidx,
                                            Implicit_gemm_traits::USE_WARP_SPECIALIZATION );

                if( params.with_residual ) {
                    epilogue.execute<true>( compute_tile.acc_ );
                } else {
                    epilogue.execute<false>( compute_tile.acc_ );
                }
            } else {
                // with splitk
                // Do the epilogue.
                Epilogue_withsplitk epilogue( params,
                                              gmem_epilogue,
                                              smem_epilogue,
                                              callbacks_epilogue,
                                              epilog_sync[0],
                                              params.use_horizontal_cta_rasterization ? bidn : bidm,
                                              params.use_horizontal_cta_rasterization ? bidm : bidn,
                                              bidz,
                                              tidx,
                                              Implicit_gemm_traits::USE_WARP_SPECIALIZATION );

                if( params.with_residual ) {
                    epilogue.execute<true>( compute_tile.acc_ );
                } else {
                    epilogue.execute<false>( compute_tile.acc_ );
                }
            }
            // Finalize the callbacks.
            callbacks_epilogue.post_epilogue();

            // Update tile
            tile.move();
            bidm = tile.bidm();
            bidn = tile.bidn();
            bidz = tile.bidz();

            cnt++;
        }
    }
#endif  // only compile sm_80 or upward
}

}  // end gemm
}  // end xmma
