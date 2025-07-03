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
#include <xmma/implicit_gemm/strided_dgrad_indexed/utils.h>

namespace xmma {
namespace implicit_gemm {
namespace strided_dgrad_indexed {

////////////////////////////////////////////////////////////////////////////////////////////////////

extern __shared__ char smem_[];

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Implicit_gemm_traits, bool Grouped_acc = false>
static __global__ __launch_bounds__( Implicit_gemm_traits::Cta_tile::THREADS_PER_CTA ) void kernel(
    typename Implicit_gemm_traits::Params params ) {
    // The traits class.
    using Traits = typename Implicit_gemm_traits::Traits;
    // The CTA tile.
    using Cta_tile = typename Implicit_gemm_traits::Cta_tile;
    // The XMMA tile.
    using Xmma_tile = typename Implicit_gemm_traits::Xmma_tile;
    // The tile distribution manager.
    using Tile_distribution_traits = typename Implicit_gemm_traits::Tile_distribution;

    const int32_t HAS_LDGSTS = Traits::Gpu_arch::HAS_LDGSTS;

    // The number of XMMAs.
    const int XMMAS_K = Xmma_tile::XMMAS_K;

    // Make sure we run the main loop at least once :)
    static_assert( XMMAS_K >= 1, "" );

    // The block indices.
    int bidm, bidn, bidz;

    Tile_distribution_traits tile( params, blockIdx );
    bidm = tile.bidm();
    bidn = tile.bidn();
    bidz = tile.bidz();

    // The thread index.
    const int tidx = threadIdx.x;

    // Ampere memory descritors.
    const uint64_t mem_desc_a = params.mem_descriptors.descriptor_a;
    const uint64_t mem_desc_b = params.mem_descriptors.descriptor_b;

    // The tiles in global memory for the images.
    using Gmem_tile_a = typename Implicit_gemm_traits::Gmem_tile_a;
    // The tiles to store the data in shared memory for the images.
    using Smem_tile_a = typename Implicit_gemm_traits::Smem_tile_a;

    // The tiles in global memory for the filters.
    using Gmem_tile_b = typename Implicit_gemm_traits::Gmem_tile_b;
    // The tiles to store the data in shared memory for the filters.
    using Smem_tile_b = typename Implicit_gemm_traits::Smem_tile_b;

    // The compute tile.
    using Compute_tile = typename Implicit_gemm_traits::Compute_tile;

    // The tile in global memory.
    using Gmem_tile_epilogue = typename Implicit_gemm_traits::Gmem_tile_epilogue;
    // The tile in shared memory to swizzle the output.
    using Swizzle_epilogue = typename Implicit_gemm_traits::Swizzle_epilogue;
    // The callbacks.
    using Callbacks_epilogue = typename Implicit_gemm_traits::Callbacks_epilogue;
    // The epilogue.
    using Epilogue = typename Implicit_gemm_traits::Epilogue;

    const int SINGLE_EXTRA_BUFFER_SIZE_IN_BYTES = Cta_tile::M * sizeof( int );

    // The amount of shared memory needed for the main loop.
    const int LOOP_SIZE_IN_BYTES = Implicit_gemm_traits::Smem_tile_a::BYTES_PER_TILE +
                                   Implicit_gemm_traits::Smem_tile_b::BYTES_PER_TILE +
                                   ( 4 * sizeof( int ) * Cta_tile::M );

    // The amount of shared memory needed by the epilogue.
    const int EPILOGUE_SIZE_IN_BYTES = Implicit_gemm_traits::Swizzle_epilogue::BYTES_PER_TILE;

    // The amount of shared memory to launch the kernel.
    const int SMEM_SIZE = xmma::div_up( max( LOOP_SIZE_IN_BYTES, EPILOGUE_SIZE_IN_BYTES ), 4 ) * 4 +
                          SINGLE_EXTRA_BUFFER_SIZE_IN_BYTES;

    // The shared memory pointers.
    char* a_smem_ = &smem_[0];
    char* b_smem_ = &smem_[Smem_tile_a::BYTES_PER_TILE];
    int4* smem_cta_batch_depth_height_width_indices_ =
        (int4*)( b_smem_ + Smem_tile_b::BYTES_PER_TILE );
    int* cta_ndhw_indices = (int*)( &smem_[SMEM_SIZE - SINGLE_EXTRA_BUFFER_SIZE_IN_BYTES] );

    Smem_tile_a a_smem( a_smem_, tidx );
    Smem_tile_b b_smem( b_smem_, tidx );

    // Get the filter pattern index.
    int cta_id_in_dhw_dimension = bidm;

    int filter_pattern_index( -1 );
    for( int i = 0; i <= params.trs; ++i ) {
        if( params.start_cta_id_in_ndhw_dimension_of_each_filter_pattern[i] >= 0 ) {
            if( cta_id_in_dhw_dimension >=
                params.start_cta_id_in_ndhw_dimension_of_each_filter_pattern[i] ) {
                filter_pattern_index = i;
            } else {
                break;
            }
        }
    }

    int strat_trs_index = params.trs - 1 - filter_pattern_index;
    int start_t_index, strat_rs_index;
    int start_r_index, start_s_index;
    if( (unsigned)strat_trs_index < params.trs ) {
        xmma::fast_divmod( start_t_index,
                           strat_rs_index,
                           strat_trs_index,
                           params.rs,
                           params.mul_rs,
                           params.shr_rs );
        xmma::fast_divmod(
            start_r_index, start_s_index, strat_rs_index, params.s, params.mul_s, params.shr_s );
    } else {
        start_t_index = -1;
        start_r_index = -1;
        start_s_index = -1;
    }
    int valid_t_number, valid_t_number_reminder;
    int valid_r_number, valid_r_number_reminder;
    int valid_s_number, valid_s_number_reminder;
    xmma::fast_divmod( valid_t_number,
                       valid_t_number_reminder,
                       start_t_index,
                       params.step_t,
                       params.mul_step_t,
                       params.shr_step_t );
    xmma::fast_divmod( valid_r_number,
                       valid_r_number_reminder,
                       start_r_index,
                       params.step_r,
                       params.mul_step_r,
                       params.shr_step_r );
    xmma::fast_divmod( valid_s_number,
                       valid_s_number_reminder,
                       start_s_index,
                       params.step_s,
                       params.mul_step_s,
                       params.shr_step_s );
    ++valid_t_number;
    ++valid_r_number;
    ++valid_s_number;

    int valid_rs_number = valid_r_number * valid_s_number;
    int valid_trs_number = valid_t_number * valid_rs_number;

    uint32_t mul_valid_rs, shr_valid_rs, mul_valid_s, shr_valid_s;
    lwda_find_divisor( mul_valid_rs, shr_valid_rs, valid_rs_number );
    lwda_find_divisor( mul_valid_s, shr_valid_s, valid_s_number );

    int lwrrent_filter_trs( 0 );
    int lwrrent_filter_t, lwrrent_filter_r, lwrrent_filter_s, lwrrent_filter_rs;

    // Create the objects to load from global memory.
    Gmem_tile_b b_gmem( params,
                        start_t_index,
                        start_r_index,
                        start_s_index,
                        valid_t_number,
                        valid_r_number,
                        valid_s_number,
                        bidn,
                        bidz,
                        tidx );

    // The first iteration of the loop.
    int valid_trs_of_the_cta = ( start_t_index + start_r_index + start_s_index < 0
                                     ? 0
                                     : valid_t_number * valid_r_number * valid_s_number );
    int loop_start = valid_trs_of_the_cta * params.loop_count_k - 1;
    // The iteration where we trigger the residue.
    int loop_residue =
        valid_trs_of_the_cta + ( params.ampere ? ( Implicit_gemm_traits::STAGES - 1 ) : 1 );

    Gmem_tile_a a_gmem( params,
                        cta_ndhw_indices,
                        cta_id_in_dhw_dimension,
                        start_t_index,
                        start_r_index,
                        start_s_index,
                        valid_t_number,
                        valid_r_number,
                        valid_s_number,
                        bidm,
                        bidn,
                        bidz,
                        tidx,
                        smem_cta_batch_depth_height_width_indices_ );

    // Create the compute tile and clear the aclwmulators.
    Compute_tile compute_tile;
    compute_tile.clear();

    // Only enable fencing if it looks like our desired stage overlap will fit in register budget.
    constexpr bool JETFIRE_FENCING_ENABLED =
        ( 255 * 4 > sizeof( compute_tile ) + sizeof( a_gmem ) + sizeof( b_gmem ) );

// Pipeline data load in prolog.
    const int PRELOAD_STAGE = xmma::Max<Implicit_gemm_traits::STAGES - 1, 1>::VALUE;
    int real_preload_stage;
    if ( HAS_LDGSTS ) {
        real_preload_stage = min( PRELOAD_STAGE, loop_start + 1 );
    } else {
        real_preload_stage = 1;
    }
    for( int stage = 0; stage < real_preload_stage; stage++ ) {
        if ( HAS_LDGSTS ) {
            a_gmem.load( a_smem, mem_desc_a );
            b_gmem.load( b_smem, mem_desc_b );
            xmma::ldgdepbar<true>();
        } else {
            a_gmem.load( mem_desc_a );
            b_gmem.load( mem_desc_b );
        }

        // Store the pixels and filters to shared memory.
        a_gmem.commit( a_smem );
        b_gmem.commit( b_smem );

        // Move the pointers and assemble the predicates for the next loop.
        ++lwrrent_filter_trs;
        if( lwrrent_filter_trs == valid_trs_number ) {
            lwrrent_filter_trs = 0;
        }
        xmma::fast_divmod( lwrrent_filter_t,
                           lwrrent_filter_rs,
                           lwrrent_filter_trs,
                           valid_rs_number,
                           mul_valid_rs,
                           shr_valid_rs );
        xmma::fast_divmod( lwrrent_filter_r,
                           lwrrent_filter_s,
                           lwrrent_filter_rs,
                           valid_s_number,
                           mul_valid_s,
                           shr_valid_s );
        a_gmem.move( lwrrent_filter_t,
                     lwrrent_filter_r,
                     lwrrent_filter_s,
                     lwrrent_filter_rs,
                     lwrrent_filter_trs );
        b_gmem.move( lwrrent_filter_s, lwrrent_filter_rs, lwrrent_filter_trs );

        // Move to next SMEM buffer for multistage or double buffer.
        a_smem.move_next_write_buffer();
        b_smem.move_next_write_buffer();

        // Do the residue now if needed.
        if( loop_start < loop_residue ) {
            if ( HAS_LDGSTS ) {
                if ( stage == (loop_start - (loop_residue - Implicit_gemm_traits::STAGES + 1)) ){
                    a_gmem.residue();
                    b_gmem.residue();
                }
            }else{
                a_gmem.residue();
                b_gmem.residue();
            }
        }
    }  // end for stage

    if ( HAS_LDGSTS ) {
        // The code here is to deal with the case when real_preload_stage<PRELOAD_STAGE
        int missed_ldgdepbar_stage = PRELOAD_STAGE - real_preload_stage;
        for( int i = 0; i < missed_ldgdepbar_stage; ++i ) {
            xmma::ldgdepbar<true>();
        }

        // Make sure the data is in shared memory.
        xmma::depbar<true, Implicit_gemm_traits::STAGES>();
    }
    __syncthreads();

    // Load the image pixels / filters.
    if( XMMAS_K > 1 )
        compute_tile.load( a_smem, b_smem, 0, true );

#ifdef XMMA_ENABLED_PMTRIG
    __prof_trigger( 15 );
#endif

    // Iterate over the loop.
    JETFIRE_MAC_LOOP_PRAGMA  // Jetfire loop body
#pragma unroll 1
    for( int loop = loop_start; loop >= 0; --loop ) {
    JETFIRE_MAC_LOOP_HEADER

        // Disable the loads in the last iteration.
        int32_t is_last;
        if ( HAS_LDGSTS ) {
            is_last = ( loop < xmma::Max<Implicit_gemm_traits::STAGES - 1, 1>::VALUE );
        } else {
            is_last = ( loop == 0 );
        }
        if( is_last ) {
            a_gmem.disable_loads();
            b_gmem.disable_loads();
        }

#pragma unroll
        for( int gemm_ki = 1; gemm_ki <= XMMAS_K; ++gemm_ki ) {
            int smem_ki = ( gemm_ki == XMMAS_K ) ? 0 : gemm_ki;

            jetfire::ifence( JETFIRE_FENCING_ENABLED );  // Interference fence at top of stage

            // Trigger the commit of global loads in last iteration
            if( XMMAS_K > 1 && gemm_ki == XMMAS_K ) {
                // Make sure the data was read from shared memory.
                if( Smem_tile_a::BUFFERS_PER_TILE == 1 || Smem_tile_b::BUFFERS_PER_TILE == 1 ) {
                    __syncthreads();
                }

                // Store the data to shared memory.
                a_gmem.commit( a_smem );
                jetfire::ifence(
                    JETFIRE_FENCING_ENABLED );  // Interference fence after gmem commit a
                b_gmem.commit( b_smem );
                jetfire::ifence(
                    JETFIRE_FENCING_ENABLED );  // Interference fence after gmem commit b

                // Make sure the data is in shared memory.
                xmma::depbar<HAS_LDGSTS, Implicit_gemm_traits::STAGES>();
                __syncthreads();

                // Move to next SMEM buffer for multistage or double buffer.
                a_smem.move_next_write_buffer();
                b_smem.move_next_write_buffer();
                // Move the shared memory pointers for double buffering.
                a_smem.move_next_read_buffer();
                b_smem.move_next_read_buffer();
            }

            // Load the matrices from shared memory.
            compute_tile.load( a_smem, b_smem, smem_ki );

            jetfire::ifence( JETFIRE_FENCING_ENABLED );  // Interference fence after smem load

            // Trigger the global loads on the 1st iteration of that core loop.
            if( gemm_ki == 1 ) {
                if ( HAS_LDGSTS ) {
                    a_gmem.load( a_smem, mem_desc_a );
                    jetfire::ifence( JETFIRE_FENCING_ENABLED );  // Interference fence after gmem load a
                    b_gmem.load( b_smem, mem_desc_b );
                    jetfire::ifence( JETFIRE_FENCING_ENABLED );  // Interference fence after gmem load b
                    xmma::ldgdepbar<true>();
                } else {
                    a_gmem.load( mem_desc_a );
                    jetfire::ifence( JETFIRE_FENCING_ENABLED );  // Interference fence after gmem load a
                    b_gmem.load( mem_desc_b );
                    jetfire::ifence( JETFIRE_FENCING_ENABLED );  // Interference fence after gmem load b
                }
            }

            // Warp context switch halfway through
            if( gemm_ki - 1 == XMMAS_K / 2 )
                jetfire::warp_switch();

            // Trigger the commit of global loads in last iteration
            if( XMMAS_K == 1 ) {
                // Make sure the data was read from shared memory.
                if( Smem_tile_a::BUFFERS_PER_TILE == 1 || Smem_tile_b::BUFFERS_PER_TILE == 1 ) {
                    __syncthreads();
                }

                // Store the data to shared memory.
                a_gmem.commit( a_smem );
                jetfire::ifence(
                    JETFIRE_FENCING_ENABLED );  // Interference fence after gmem commit a
                b_gmem.commit( b_smem );
                jetfire::ifence(
                    JETFIRE_FENCING_ENABLED );  // Interference fence after gmem commit b

                // Make sure the data is in shared memory.
                xmma::depbar<HAS_LDGSTS, Implicit_gemm_traits::STAGES>();
                __syncthreads();

                // Move to next SMEM buffer for multistage or double buffer.
                a_smem.move_next_write_buffer();
                b_smem.move_next_write_buffer();
                // Move the shared memory pointers for double buffering.
                a_smem.move_next_read_buffer();
                b_smem.move_next_read_buffer();
            }

            // Do the math - The core of the loop does 16x16x8.
            compute_tile.compute( gemm_ki );

            if( gemm_ki == 1 && !( Cta_tile::M == 64 && Cta_tile::N == 64 ) ) {
                // Move the global pointers.
                ++lwrrent_filter_trs;
                if( lwrrent_filter_trs == valid_trs_number ) {
                    lwrrent_filter_trs = 0;
                }
                xmma::fast_divmod( lwrrent_filter_t,
                                   lwrrent_filter_rs,
                                   lwrrent_filter_trs,
                                   valid_rs_number,
                                   mul_valid_rs,
                                   shr_valid_rs );
                xmma::fast_divmod( lwrrent_filter_r,
                                   lwrrent_filter_s,
                                   lwrrent_filter_rs,
                                   valid_s_number,
                                   mul_valid_s,
                                   shr_valid_s );
                a_gmem.move( lwrrent_filter_t,
                             lwrrent_filter_r,
                             lwrrent_filter_s,
                             lwrrent_filter_rs,
                             lwrrent_filter_trs );
                b_gmem.move( lwrrent_filter_s, lwrrent_filter_rs, lwrrent_filter_trs );
            }

        }  // (gemm_ki)

        if( Cta_tile::M == 64 && Cta_tile::N == 64 ) {
            // Move the global pointers.
            ++lwrrent_filter_trs;
            if( lwrrent_filter_trs == valid_trs_number ) {
                lwrrent_filter_trs = 0;
            }
            xmma::fast_divmod( lwrrent_filter_t,
                               lwrrent_filter_rs,
                               lwrrent_filter_trs,
                               valid_rs_number,
                               mul_valid_rs,
                               shr_valid_rs );
            xmma::fast_divmod( lwrrent_filter_r,
                               lwrrent_filter_s,
                               lwrrent_filter_rs,
                               valid_s_number,
                               mul_valid_s,
                               shr_valid_s );
            a_gmem.move( lwrrent_filter_t,
                         lwrrent_filter_r,
                         lwrrent_filter_s,
                         lwrrent_filter_rs,
                         lwrrent_filter_trs );
            b_gmem.move( lwrrent_filter_s, lwrrent_filter_rs, lwrrent_filter_trs );
        }

        // Execute the residue code. Clear the masks for the image if needed.
        if( loop <= loop_residue ) {
            a_gmem.residue();
            b_gmem.residue();
        }

    }  // (loop)

#ifdef XMMA_ENABLED_PMTRIG
    __prof_trigger( 15 );
#endif

    // Do allocate the tile to output in the epilogue.
    Gmem_tile_epilogue gmem_epilogue( params, cta_ndhw_indices, bidn, tidx );
    // Do allocate the tile and compute the offsets. TODO: Merge with the
    // epilogue class!
    Swizzle_epilogue swizzle_epilogue( smem_, tidx );
    // The callbacks.
    Callbacks_epilogue callbacks_epilogue( params, smem_, bidm, bidn, bidz, tidx );

// Make sure we can use the shared memory.
    xmma::depbar<HAS_LDGSTS, 0>();
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

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // strided_dgrad_indexed
}  // namespace implicit_gemm
}  // namespace xmma
