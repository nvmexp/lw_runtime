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
#include <xmma/named_barrier.h>
#include <xmma/implicit_gemm/strided_dgrad_indexed/warp_specialized_utils.h>

namespace xmma {
namespace implicit_gemm {
namespace strided_dgrad_indexed {

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Implicit_gemm_traits, bool Grouped_acc = false>
static __global__ __launch_bounds__( Implicit_gemm_traits::Cta_tile::THREADS_PER_CTA * 2, 1 )
void xmma_implicit_gemm_strided_dgrad_specialize_1math_1dma_arrive_wait_kernel
( typename Implicit_gemm_traits::Params params ) {
#if defined( __LWDA_ARCH__ ) && __LWDA_ARCH__ >= 800
    // The traits class.
    using Traits = typename Implicit_gemm_traits::Traits;
    // The CTA tile.
    using Cta_tile = typename Implicit_gemm_traits::Cta_tile;
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
    uint64_t* counter =
        reinterpret_cast<uint64_t*>( &smem_[Implicit_gemm_traits::SMEM_BYTES_PER_CTA -
                                            Implicit_gemm_traits::ARRIVE_WAIT_SMEM_SIZE] );
    if( threadIdx.x < ( (Implicit_gemm_traits::BUFFERS_PER_SMEM_TILE_A)*2 ) ) {
        xmma::bar_create( &counter[threadIdx.x], Cta_tile::THREADS_PER_CTA );
    }
    __syncthreads();

    // The shared memory pointers.
    char* a_smem_ = &smem_[0];
    char* b_smem_ = &smem_[Smem_tile_a::BYTES_PER_TILE];

    // store one barrier using 1-bit in int32, the buffer_head indicates the bit position (start
    // from LSB).
    // total bit needs in int32 is Implicit_gemm_traits::BUFFERS_PER_SMEM_TILE_A
    unsigned int lwrrent_phase_buffer_full = 0;
    // total bit needs in int32 is Implicit_gemm_traits::BUFFERS_PER_SMEM_TILE_A
    unsigned int lwrrent_phase_buffer_empty = 0;

    // The barriers used for P->C communication and P is DMA warp group.
    xmma::Arrive_wait buffer_full( &counter[0], 0 );
    // The barriers used for P->C communication, and P is math warp group.
    xmma::Arrive_wait buffer_empty( &counter[Implicit_gemm_traits::BUFFERS_PER_SMEM_TILE_A], 0 );

    const int warp_id = __shfl_sync( 0xffffffff, threadIdx.x / 32, 0 );
    // DMA
    if( warp_id < Cta_tile::THREADS_PER_CTA / 32 ) {
        // The named barrier used for dma sync in prologue.
        xmma::Named_barrier dma_sync( 1, Cta_tile::THREADS_PER_CTA );

        int4* smem_cta_batch_depth_height_width_indices_ =
            (int4*)( b_smem_ + Smem_tile_b::BYTES_PER_TILE );
        int* cta_ndhw_indices =
            (int*)( b_smem_ + Smem_tile_b::BYTES_PER_TILE + 4 * sizeof( int ) * Cta_tile::M );

        Tile_distribution_persistent tile( params, blockIdx.x );
        // The block indices.
        int bidm = tile.bidm();
        int bidn = tile.bidn();
        int bidz = tile.bidz();

        // The thread index.
        const int tidx = threadIdx.x & ( Cta_tile::THREADS_PER_CTA - 1 );

        // The index of img buffer
        int buffer_head = 0, cnt = 0;

        // The tiles in global memory for the images and filters.
        using Gmem_tile_a = typename Implicit_gemm_traits::Gmem_tile_a;
        using Gmem_tile_b = typename Implicit_gemm_traits::Gmem_tile_b;

        Smem_tile_a a_smem( a_smem_, tidx );
        Smem_tile_b b_smem( b_smem_, tidx );

        unsigned int phase_bit;
        while( !tile.is_last() ) {
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
                xmma::fast_divmod( start_r_index,
                                   start_s_index,
                                   strat_rs_index,
                                   params.s,
                                   params.mul_s,
                                   params.shr_s );
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
            // The first iteration of the loop.
            int valid_trs_of_the_cta = ( start_t_index + start_r_index + start_s_index < 0
                                             ? 0
                                             : valid_t_number * valid_r_number * valid_s_number );
            int loop_start = valid_trs_of_the_cta * params.loop_count_k - 1;

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

            // The iteration where we trigger the residue.
            int loop_residue = valid_trs_of_the_cta;

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
                                smem_cta_batch_depth_height_width_indices_,
                                dma_sync );

            // update tile
            tile.move();
            bidm = tile.bidm();
            bidz = tile.bidz();
            bidn = tile.bidn();

#pragma unroll 1
            for( int loop = loop_start; loop >= 0; --loop ) {
                if( cnt >= Implicit_gemm_traits::BUFFERS_PER_SMEM_TILE_A ) {
                    // wait barrier
                    phase_bit = ( lwrrent_phase_buffer_empty >> buffer_head ) & 1;
                    buffer_empty.bar_wait( buffer_head, phase_bit );
                    lwrrent_phase_buffer_empty ^= ( 1 << buffer_head ) ^ ( 0 );
                }
                a_gmem.load( a_smem );
                b_gmem.load( b_smem );

                // Store the pixels and filters to shared memory.
                a_gmem.commit( a_smem );
                b_gmem.commit( b_smem );

                // async copy arrive barrier
                buffer_full.bar_arrive_ldgsts( buffer_head );
                buffer_head = ( buffer_head < Implicit_gemm_traits::BUFFERS_PER_SMEM_TILE_A - 1 )
                                  ? ( buffer_head + 1 )
                                  : 0;

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

                a_smem.move_next_write_buffer();
                b_smem.move_next_write_buffer();

                cnt++;

                // Do the residue now if needed.
                if( loop <= loop_residue ) {
                    a_gmem.residue();
                    b_gmem.residue();
                }
            }
        }
        // MATH
    } else {
        // The named barrier used for epilog swizzle.
        // Note: The named barrier is set to 2 as a WAR. See CFK-2789 for details.
        xmma::Named_barrier epilog_sync( 2, Cta_tile::THREADS_PER_CTA );

        // The number of XMMAs.
        const int XMMAS_K = Xmma_tile::XMMAS_K;

        // The thread index.
        const int tidx = threadIdx.x & ( Cta_tile::THREADS_PER_CTA - 1 );
        // The index of img and flt buffers
        int buffer_head = 0, cnt = 0;

        // Block index for first tile in persistent model
        Tile_distribution_persistent tile( params, blockIdx.x );
        int bidm = tile.bidm();
        int bidn = tile.bidn();
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

        bool is_wait_complete;
        unsigned int phase_bit;
        while( !tile.is_last() ) {
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
                xmma::fast_divmod( start_r_index,
                                   start_s_index,
                                   strat_rs_index,
                                   params.s,
                                   params.mul_s,
                                   params.shr_s );
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

            uint32_t mul_valid_rs, shr_valid_rs, mul_valid_s, shr_valid_s;
            lwda_find_divisor( mul_valid_rs, shr_valid_rs, valid_rs_number );
            lwda_find_divisor( mul_valid_s, shr_valid_s, valid_s_number );

            // The first iteration of the loop.
            int valid_trs_of_the_cta = ( start_t_index + start_r_index + start_s_index < 0
                                             ? 0
                                             : valid_t_number * valid_r_number * valid_s_number );
            int loop_start = valid_trs_of_the_cta * params.loop_count_k - 1;

            // Clear the aclwmulators.
            Compute_tile compute_tile;
            compute_tile.clear();

            // Only enable fencing if it looks like our desired stage overlap will fit in register
            // budget.
            constexpr bool JETFIRE_FENCING_ENABLED = ( 255 * 4 > sizeof( compute_tile ) );

            if( loop_start >= 0 ) {
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
                    for( int loop = loop_start; loop >= 0; --loop ) {
                    JETFIRE_MAC_LOOP_HEADER
                    // Disable the loads in the last iteration.
                    const int is_last = loop == 0;

#pragma unroll
                    for( int gemm_ki = 1; gemm_ki <= XMMAS_K; ++gemm_ki ) {
                        int smem_ki = ( gemm_ki == XMMAS_K ) ? 0 : gemm_ki;

                        jetfire::ifence(
                            JETFIRE_FENCING_ENABLED );  // Interference fence at top of stage
                        // Trigger the commit of global loads in last iteration
                        if( gemm_ki == XMMAS_K ) {

                            // Move the shared memory pointers for double buffering.
                            a_smem.move_next_read_buffer();
                            b_smem.move_next_read_buffer();

                            // Load the image pixels from shared memory.
                            if( !is_last ) {
                                // Wait barrier
                                if( !is_wait_complete ) {
                                    buffer_full.bar_wait( buffer_head, phase_bit );
                                }
                            }
                        }
                        // Load the tiel A and B from shared memory buffer.
                        compute_tile.load( a_smem, b_smem, smem_ki );

                        // Barrier  Arrival after SMEM load complete
                        if( gemm_ki == XMMAS_K - 1 ) {
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
                        compute_tile.compute( gemm_ki );

                        if( gemm_ki == XMMAS_K - 1 ) {
                            // wait barrier peek
                            phase_bit = ( lwrrent_phase_buffer_full >> buffer_head ) & 1;
                            is_wait_complete = buffer_full.bar_peek( buffer_head, phase_bit );
                        }

                    }  // (gemm_ki)
                    // Phase update
                    if( !is_last ) {
                        lwrrent_phase_buffer_full ^= ( 1 << buffer_head ) ^ ( 0 );
                    }
                }  // (loop)

                cnt++;

            }  // (loop_start)

            char* epi_smem_ = &smem_[Smem_tile_a::BYTES_PER_TILE + Smem_tile_b::BYTES_PER_TILE +
                                     Implicit_gemm_traits::SMEM_SIZE_PER_EXTRA_LOOP];
            int* cta_ndhw_indices = (int*)( epi_smem_ + Swizzle_epilogue::BYTES_PER_TILE );

            // Do allocate the tile to output in the epilogue.
            Gmem_tile_epilogue gmem_epilogue(
                params, cta_ndhw_indices, bidm, bidn, tidx, epilog_sync );
            // Do allocate the tile and compute the offsets. TODO: Merge with the
            // epilogue class!
            Swizzle_epilogue swizzle_epilogue( epi_smem_, tidx );
            // The callbacks.
            Callbacks_epilogue callbacks_epilogue( params, epi_smem_, bidm, bidn, bidz, tidx );

            // without splik
            if( params.split_k.slices == 1 ) {
                // Do the epilogue.
                Epilogue_wosplitk epilogue( params,
                                            gmem_epilogue,
                                            swizzle_epilogue,
                                            callbacks_epilogue,
                                            epilog_sync,
                                            params.use_horizontal_cta_rasterization ? bidn : bidm,
                                            params.use_horizontal_cta_rasterization ? bidm : bidn,
                                            bidz,
                                            tidx,
                                            Implicit_gemm_traits::USE_WARP_SPECIALIZATION );

                if( params.with_residual ) {
                    epilogue.template execute<true>( compute_tile.acc_ );
                } else {
                    epilogue.template execute<false>( compute_tile.acc_ );
                }
            } else {
                // Do the epilogue.
                Epilogue_withsplitk epilogue( params,
                                              gmem_epilogue,
                                              swizzle_epilogue,
                                              callbacks_epilogue,
                                              epilog_sync,
                                              params.use_horizontal_cta_rasterization ? bidn : bidm,
                                              params.use_horizontal_cta_rasterization ? bidm : bidn,
                                              bidz,
                                              tidx,
                                              Implicit_gemm_traits::USE_WARP_SPECIALIZATION );

                if( params.with_residual ) {
                    epilogue.template execute<true>( compute_tile.acc_ );
                } else {
                    epilogue.template execute<false>( compute_tile.acc_ );
                }
            }
            // Finalize the callbacks.
            callbacks_epilogue.post_epilogue();
            // update tile
            tile.move();
            bidm = tile.bidm();
            bidn = tile.bidn();
            bidz = tile.bidz();

        }  // (while)
    }      // (else)
#endif     // only compile sm_80 or upward
}

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // strided_dgrad_indexed
}  // namespace implicit_gemm
}  // namespace xmma
