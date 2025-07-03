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

#include <xmma/fragment.h>
#include <xmma/named_barrier.h>
#include <xmma/hopper/traits.h>

namespace xmma {
namespace helpers {

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    // The instruction traits.
    typename Traits_,
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile_,
    // The layout of the tile.
    typename Layout_,
    // The global memory tile to store the output.
    typename Gmem_tile_,
    // The callbacks to lwstomize the behaviour of the epilogue.
    typename Callbacks_ = Empty_callbacks_epilogue<Traits_, Cta_tile_>,
    // The class to swizzle the data.
    typename Swizzle_ =
        xmma::Swizzle_epilogue<Traits_, Cta_tile_, Layout_, Gmem_tile_::BYTES_PER_STG>,
    // The fragment class before the swizzling.
    typename Fragment_pre_swizzle_ = typename Callbacks_::Fragment_pre_swizzle,
    // The fragment class after the swizzling.
    typename Fragment_post_swizzle_ = typename Callbacks_::Fragment_post_swizzle,
    // The output fragment.
    typename Fragment_c_ = typename Callbacks_::Fragment_c>
struct Epilogue_gmma_with_split_k_base {

    // The instruction traits.
    using Traits = Traits_;
    // The dimensions of the tile computed by the CTA.
    using Cta_tile = Cta_tile_;
    // The layout of the tile.
    using Layout = Layout_;
    // The global memory tile to store the output.
    using Gmem_tile = Gmem_tile_;
    // The callbacks.
    using Callbacks = Callbacks_;
    // The class to swizzle the data.
    using Swizzle = Swizzle_;

    // The fragment class before the swizzling.
    using Fragment_pre_swizzle = Fragment_pre_swizzle_;
    // The fragment class after the swizzling.
    using Fragment_post_swizzle = Fragment_post_swizzle_;
    // The output fragment.
    using Fragment_c = Fragment_c_;

    // The fragment for alpha (used before swizzling).
    using Fragment_alpha_pre_swizzle = typename Callbacks::Fragment_alpha_pre_swizzle;
    // The fragment for alpha (used after swizzling).
    using Fragment_alpha_post_swizzle = typename Callbacks::Fragment_alpha_post_swizzle;
    // The fragment for beta.
    using Fragment_beta = typename Callbacks::Fragment_beta;

    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;

    // Tile size of swizzle smem
    enum { TILE_M = Swizzle::TILE_M, TILE_N = Swizzle::TILE_N };
    // Number of LDS/STG iterations for one XMMA
    enum { LDS_ITERATIONS_PER_XMMA_M = Swizzle::LDS_ITERATIONS_PER_GMMA_M };
    enum { LDS_ITERATIONS_PER_XMMA_N = Swizzle::LDS_ITERATIONS_PER_GMMA_N };
    enum { STGS = Gmem_tile::STG_ITERATIONS_PER_TILE };

    // Ctor.
    inline __device__ Epilogue_gmma_with_split_k_base() {
    }

    // Ctor.
    template <typename Params>
    inline __device__ Epilogue_gmma_with_split_k_base( const Params &params,
                                                       Gmem_tile &gmem_tile,
                                                       Swizzle &swizzle,
                                                       Callbacks &callbacks,
                                                       const Named_barrier &epi_sync = Named_barrier(),
                                                       const int bidm = blockIdx.x,
                                                       const int bidn = blockIdx.y,
                                                       const int bidz = blockIdx.z,
                                                       const int tidx = threadIdx.x,
                                                       const bool is_warp_specialized = false )
        : gmem_tile_( gmem_tile ), swizzle_( swizzle ), callbacks_( callbacks ),
          split_k_slices_( params.split_k.slices ), split_k_buffers_( params.split_k.buffers ),
          split_k_kernels_( params.split_k.kernels ),
          split_k_buffer_size_( params.split_k.buffer_size ),
          split_k_buffers_ptr_( params.split_k.buffers_gmem ),
          split_k_counters_ptr_( params.split_k.counters_gmem ),
          split_k_retired_ctas_ptr_( params.split_k.retired_ctas_gmem ),
          split_k_fragment_offset_( 0 ), is_batched_( params.batch.is_batched ),
          epi_sync_( epi_sync ), bidm_( bidm ), bidn_( bidn ), bidz_( bidz ), tidx_( tidx ),
          tiles_x_( params.tiles_x ), tiles_y_( params.tiles_y ),
          mem_desc_c_( params.mem_descriptors.descriptor_c ),
          mem_desc_d_( params.mem_descriptors.descriptor_d ),
          initializer_id_( params.split_k_atomic.initializer_id ),
          initializer_done_( params.split_k_atomic.initializer_done ) {
    }

    // Execute a single iteration of the loop.
    template <bool WITH_RESIDUAL, int CONTIGUOUS>
    inline __device__ void
    step( int si, int mi, int ni, Fragment_pre_swizzle ( &pre_swizzle )[CONTIGUOUS] ) {

        // The output masks.
        int out_masks[STGS];
        #pragma unroll
        for( int ii = 0; ii < STGS; ++ii ) {
            out_masks[ii] = this->gmem_tile_.compute_output_mask( si, mi, ni, ii );
        }

        // Load valid values if beta is not zero.
        Fragment_c res_fetch[STGS];
        if( WITH_RESIDUAL ) {
            #pragma unroll
            for( int ii = 0; ii < STGS; ++ii ) {
                this->gmem_tile_.load( res_fetch[ii], si, mi, ni, ii, out_masks[ii], mem_desc_c_ );
            }
        }

        // Use the precision of the output to do split-k.
        if( Traits::USE_SPLIT_K_WITH_OUTPUT_PRECISION ) {
            // Do the split-k before the swizzle (when WARPS_K == 1).
            if( Cta_tile::WARPS_K == 1 && split_k_slices_ > 1 ) {
                this->split_k( si, mi, ni, pre_swizzle );
            }

            // Early-exit if the CTA does not have extra work to do.
            if( Cta_tile::WARPS_K == 1 && !split_k_must_swizzle_output_ ) {
                return;
            }
        }

        // Store the data in shared memory to produce more friendly stores.
        #pragma unroll
        for( int ci = 0; ci < CONTIGUOUS; ++ci ) {
            this->swizzle_.store( ci, pre_swizzle[ci] );
        }

        // Make sure the data is in SMEM.
        if( epi_sync_.invalid() ) {
            __syncthreads();
        } else {
            epi_sync_.wait();
        }

        // The fragments after the swizzle. One fragment per STG.128.
        Fragment_post_swizzle post_swizzle[STGS];
        #pragma unroll
        for( int ii = 0; ii < STGS; ++ii ) {
            this->swizzle_.load( post_swizzle[ii], ii );
        }

        // Load alpha post swizzle.
        Fragment_alpha_post_swizzle alpha_post_swizzle[STGS];
        #pragma unroll
        for( int ii = 0; ii < STGS; ++ii ) {
            callbacks_.alpha_post_swizzle( *this, si, ii, alpha_post_swizzle[ii] );
        }

        // Do the parallel reduction, if needed.
        #pragma unroll
        for( int ii = 0; ii < STGS; ++ii ) {
            post_swizzle[ii].reduce( alpha_post_swizzle[ii] );
        }

        // Do something now that the data has been swizzled.
        #pragma unroll
        for( int ii = 0; ii < STGS; ++ii ) {
            callbacks_.post_swizzle( *this, si, ni, ii, post_swizzle[ii], out_masks[ii] );
        }

        // Load beta. TODO: We should not need a loop.
        Fragment_beta beta[STGS];
        if( WITH_RESIDUAL ) {
            #pragma unroll
            for( int ii = 0; ii < STGS; ++ii ) {
                callbacks_.beta( *this, si, ii, beta[ii] );
            }
        }

        // Add the residual value before packing.
        if( WITH_RESIDUAL ) {
            #pragma unroll
            for( int ii = 0; ii < STGS; ++ii ) {
                post_swizzle[ii].add_residual( res_fetch[ii], beta[ii] );
            }
        }

        if( !Traits::USE_SPLIT_K_WITH_OUTPUT_PRECISION ) {  // if split_k doesn't require
                                                            // colwersion,  do split_k reduction
                                                            // first.
            // Do the split-k after the swizzle when WARPS_K >= 2.
            if( Cta_tile::WARPS_K >= 2 && split_k_slices_ > 1 ) {
                this->split_k( si, mi, ni, post_swizzle );
            }

            // Early-exit if the CTA does not have extra work to do.
            if( Cta_tile::WARPS_K >= 2 && !split_k_must_swizzle_output_ ) {
                return;
            }
        }

        // Do something before packing and pack to produce a STG.128.
        // Put in one loop for F2IP.RELU optimization.
        // Pack and colwert
        Fragment_c out_regs[STGS];
        #pragma unroll
        for( int ii = 0; ii < STGS; ++ii ) {
            callbacks_.pre_pack( *this, si, ni, ii, post_swizzle[ii] );
            out_regs[ii].pack( alpha_post_swizzle[ii], post_swizzle[ii] );
        }

        if( Traits::USE_SPLIT_K_WITH_OUTPUT_PRECISION ) {
            // Do the split-k after the swizzle when WARPS_K >= 2.
            if( Cta_tile::WARPS_K >= 2 && split_k_slices_ > 1 ) {
                this->split_k( si, mi, ni, out_regs );
            }

            // Early-exit if the CTA does not have extra work to do.
            if( Cta_tile::WARPS_K >= 2 && !split_k_must_swizzle_output_ ) {
                return;
            }
        }

        // Add the residual value.
        if( WITH_RESIDUAL ) {
            #pragma unroll
            for( int ii = 0; ii < STGS; ++ii ) {
                out_regs[ii].add_residual( res_fetch[ii], beta[ii] );
            }
        }

        // Do something before we store.
        #pragma unroll
        for( int ii = 0; ii < STGS; ++ii ) {
            callbacks_.pre_store( *this, si, ni, ii, out_regs[ii], out_masks[ii] );
        }

        #pragma unroll
        for( int ii = 0; ii < STGS; ++ii ) {
            this->gmem_tile_.store( si, mi, ni, ii, out_regs[ii], out_masks[ii], mem_desc_d_ );
        }
    }

    // The split-k function.
    template <typename Fragment, int CONTIGUOUS>
    inline __device__ void split_k( int si, int mi, int ni, Fragment ( &fragments )[CONTIGUOUS] ) {

        // The number of CTAs per slice.
        const int ctas_per_slice = tiles_x_ * tiles_y_;
        // The position of the CTA in the X*Y slice.
        const int cta_in_slice = bidn_ * tiles_x_ + bidm_;
        // The number of threads in the X*Y dimension of the grid.
        const int threads_per_slice = ctas_per_slice * Cta_tile::THREADS_PER_CTA;
        // The position of the thread in that X*Y slice.
        const int thread_in_slice = cta_in_slice * Cta_tile::THREADS_PER_CTA + tidx_;

        // The number of bytes per fragment stored in memory.
        const int64_t bytes_per_fragment =
            (int64_t)threads_per_slice * Fragment::BYTES_PER_LOAD_STORE;
        // The base pointer.
        char *base_ptr = &split_k_buffer_ptr_[split_k_fragment_offset_ * bytes_per_fragment];

        // Do we reduce on fragment_acc or fragment_pre_swizzle
        const bool reduce_on_acc = ( !Traits::USE_SPLIT_K_WITH_OUTPUT_PRECISION ) &&
                                   ( Cta_tile::WARPS_K == 1 ) && ( split_k_slices_ > 1 );
        // The base index of fragment
        const int fragment_idx_base =
            reduce_on_acc
                ? ( si * CONTIGUOUS )
                : ( si * LDS_ITERATIONS_PER_XMMA_N * LDS_ITERATIONS_PER_XMMA_M * CONTIGUOUS +
                    ni * LDS_ITERATIONS_PER_XMMA_M * CONTIGUOUS + mi * CONTIGUOUS );

        // Perform the reduction steps.
        for( int step = 0; step < split_k_reduction_steps_; ++step ) {

            // The address of the the ith buffer.
            const char *ptr = &base_ptr[step * split_k_buffer_size_];

            // Read the old values (if any).
            Fragment old[CONTIGUOUS];
            #pragma unroll
            for( int ci = 0; ci < CONTIGUOUS; ++ci ) {
                const char *buffer = &ptr[( fragment_idx_base + ci ) * bytes_per_fragment];
                old[ci].template deserialize<Gmem_tile::BYTES_PER_STG>(
                    buffer, thread_in_slice, threads_per_slice );
            }

            // Add the values to the current ones.
            #pragma unroll
            for( int ci = 0; ci < CONTIGUOUS; ++ci ) {
                fragments[ci].add( old[ci] );
            }
        }

        // Store data in a swizzled format as "someone" else will fix the swizzling later.
        if( !split_k_must_swizzle_output_ ) {
            #pragma unroll
            for( int ci = 0; ci < CONTIGUOUS; ++ci ) {
                char *ptr = &base_ptr[( fragment_idx_base + ci ) * bytes_per_fragment];
                fragments[ci].template serialize<Gmem_tile::BYTES_PER_STG>(
                    ptr, thread_in_slice, threads_per_slice );
            }
        }
    }

    // The output tile.
    Gmem_tile &gmem_tile_;
    // The shared memory tile.
    Swizzle &swizzle_;
    // The callbacks.
    Callbacks &callbacks_;
    // The named barrier object used for epilog sync.
    const Named_barrier epi_sync_;

    // The number of slices for split-k.
    const int split_k_slices_;
    // The number of buffers for split-k.
    const int split_k_buffers_;
    // The number of kernels for split-k.
    const int split_k_kernels_;
    // The size of a single buffer in bytes.
    const int64_t split_k_buffer_size_;
    // The buffer to store the data.
    void *const split_k_buffers_ptr_;
    // The buffer to keep the counters (one per buffer + one in total).
    int32_t *const split_k_counters_ptr_;
    // The buffer to keep the number of retired CTAs per tile.
    int32_t *const split_k_retired_ctas_ptr_;

    // The atomic buffer to keep the index of the CTA that is initializing output buffer
    int32_t *initializer_id_;

    // The atomic counter for no. of CTAs that finished initialization.
    int32_t *initializer_done_;

    // The enablement of Batched GEMM
    const bool is_batched_;

    // The block ids computed from tile distribution.
    int bidm_, bidn_, bidz_;
    // The thread index.
    int tidx_;
    // The number of CTA tiles in each dimension.
    int tiles_x_, tiles_y_;
    // The split-k buffer used by that CTA.
    char *split_k_buffer_ptr_;
    // The number of split-k steps.
    int split_k_reduction_steps_;
    // Do we need to swizzle the output?
    int split_k_must_swizzle_output_;
    // An offset to issue two epilogues back to back.
    int split_k_fragment_offset_;
    // Ampere memory descriptors
    const uint64_t mem_desc_c_, mem_desc_d_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    // The instruction traits.
    typename Traits_,
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile_,
    // The layout of the tile.
    typename Layout_,
    // The global memory tile to store the output.
    typename Gmem_tile_,
    // The callbacks to lwstomize the behaviour of the epilogue.
    typename Callbacks = Empty_callbacks_epilogue<Traits_, Cta_tile_>,
    // The class to swizzle the data.
    typename Swizzle_ =
        xmma::Swizzle_epilogue<Traits_, Cta_tile_, Layout_, Gmem_tile_::BYTES_PER_STG>,
    // The fragment class before the swizzling.
    typename Fragment_pre_swizzle_ = typename Callbacks::Fragment_pre_swizzle,
    // The fragment class after the swizzling.
    typename Fragment_post_swizzle_ = typename Callbacks::Fragment_post_swizzle,
    // The output fragment.
    typename Fragment_c_ = typename Callbacks::Fragment_c>
struct Epilogue_gmma_with_split_k {};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    // The instruction traits.
    typename Traits_,
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile_,
    // The layout of the tile. Specialized for xmma::Row.
    // typename Layout_,
    // The global memory tile to store the output.
    typename Gmem_tile_,
    // The callbacks to lwstomize the behaviour of the epilogue.
    typename Callbacks,
    // The class to swizzle the data.
    typename Swizzle_,
    // The fragment class before the swizzling.
    typename Fragment_pre_swizzle_,
    // The fragment class after the swizzling.
    typename Fragment_post_swizzle_,
    // The output fragment.
    typename Fragment_c_>
struct Epilogue_gmma_with_split_k<Traits_,
                                  Cta_tile_,
                                  xmma::Row,
                                  Gmem_tile_,
                                  Callbacks,
                                  Swizzle_,
                                  Fragment_pre_swizzle_,
                                  Fragment_post_swizzle_,
                                  Fragment_c_>
    : public Epilogue_gmma_with_split_k_base<Traits_,
                                             Cta_tile_,
                                             xmma::Row,
                                             Gmem_tile_,
                                             Callbacks,
                                             Swizzle_,
                                             Fragment_pre_swizzle_,
                                             Fragment_post_swizzle_,
                                             Fragment_c_> {
    // The base class.
    using Base = Epilogue_gmma_with_split_k_base<Traits_,
                                                 Cta_tile_,
                                                 xmma::Row,
                                                 Gmem_tile_,
                                                 Callbacks,
                                                 Swizzle_,
                                                 Fragment_pre_swizzle_,
                                                 Fragment_post_swizzle_,
                                                 Fragment_c_>;

    // Ctor.
    template <typename Params>
    inline __device__ Epilogue_gmma_with_split_k( const Params &params,
                                                  typename Base::Gmem_tile &gmem_tile,
                                                  typename Base::Swizzle &swizzle,
                                                  typename Base::Callbacks &callbacks,
                                                  const Named_barrier &epi_sync = Named_barrier(),
                                                  const int bidm = blockIdx.x,
                                                  const int bidn = blockIdx.y,
                                                  const int bidz = blockIdx.z,
                                                  const int tidx = threadIdx.x,
                                                  const bool is_warp_specialized = false )
        : Base( params,
                gmem_tile,
                swizzle,
                callbacks,
                epi_sync,
                bidm,
                bidn,
                bidz,
                tidx,
                is_warp_specialized ) {
    }

    // Deterministic reduction. 1st kernel.
    template <bool WITH_RESIDUAL, typename Fragment_aclwmulator, int STRIDED, int CONTIGUOUS>
    inline __device__ void execute( Fragment_aclwmulator ( &acc )[STRIDED][CONTIGUOUS],
                                    int with_lock = 1,
                                    int with_unlock = 1 ) {

        // The number of CTAs per slice.
        const int ctas_per_slice = Base::tiles_x_ * Base::tiles_y_;
        // The position of the CTA in the X*Y slice.
        const int cta_in_slice = Base::bidn_ * Base::tiles_x_ + Base::bidm_;

        // Is it the last CTA working on a given tile?
        int split_k_is_last_slice =
            ( Base::bidz_ == Base::split_k_slices_ - 1 ) || ( Base::is_batched_ );
        // The number of reduction steps for split-k.
        Base::split_k_reduction_steps_ = 0;
        // The buffer in global memory where to put the data for split-k.
        int split_k_buffer = 0;
        // Do we skip the atomics.
        int split_k_skip_atomics = 0;

        // The counter. One CTA at a time (we could also do it at the warp level).
        int32_t *split_k_counter_ptr = 0;
        // The offset to the number of retired CTAs.
        int32_t *split_k_retired_ctas_ptr = 0;

        // No reduction.
        if( Base::split_k_slices_ == 1 ) {

            split_k_skip_atomics = 1;

            // Each slice has its own buffer.
        } else if( Base::split_k_kernels_ == 2 &&
                   Base::split_k_slices_ == Base::split_k_buffers_ ) {

            // The buffer. It may change later if slices > buffers (see below).
            split_k_buffer = Base::bidz_;
            // No need to increase the counters.
            split_k_skip_atomics = 1;

            // If we enable split-k, the last slice does the final reduction.
        } else if( split_k_is_last_slice && Base::split_k_kernels_ == 1 ) {  // the last slice K

            // The starting buffer is 0.
            split_k_buffer = 0;
            // Usually we have split_k_buffers_ <= split_k_slices_. When split_k_buffers_ ==
            // split_k_slices_, the last slice holds its data in shared memory. So, it doesn't need
            // to load its buffer and it shouldn't load the buffer as the buffer is not initialized.
            Base::split_k_reduction_steps_ = Base::split_k_buffers_;
            if( Base::split_k_slices_ == Base::split_k_buffers_ ) {
                Base::split_k_reduction_steps_ -= 1;
            }
            // The total number of retired CTAs per tile.
            split_k_retired_ctas_ptr = &( Base::split_k_retired_ctas_ptr_ )[cta_in_slice];

            xmma::spin_lock( split_k_retired_ctas_ptr,
                             with_lock ? Base::split_k_slices_ - 1 : 0,
                             0,
                             Base::tidx_,
                             Base::epi_sync_ );

            // If CTAs do a sequential reduction first, acquire the lock on the buffer.
        } else {  // slices 0,1,... (K-1)

            // The corresponding buffer (when we have multiple buffers).
            split_k_buffer = Base::bidz_ % Base::split_k_buffers_;
            // The number of CTAs before this one.
            int predecessors = Base::bidz_ / Base::split_k_buffers_;
            // Do 1 parallel step unless that's the 1st CTA to write to the buffer.
            Base::split_k_reduction_steps_ = predecessors == 0 ? 0 : 1;
            // The total number of retired CTAs per tile.
            split_k_retired_ctas_ptr = &( Base::split_k_retired_ctas_ptr_ )[cta_in_slice];

            // The counter in global memory.
            int counter = split_k_buffer * ctas_per_slice + cta_in_slice;
            // The counter. One CTA at a time (we could also do it at the warp level).
            split_k_counter_ptr = &( Base::split_k_counters_ptr_ )[counter];
            // Wait for all the CTAs preceding this one to be done with the buffer.

            xmma::spin_lock( split_k_counter_ptr,
                             with_lock ? predecessors : 0,
                             0,
                             Base::tidx_,
                             Base::epi_sync_ );
        }

        // Do we do the swizzle of the output.
        Base::split_k_must_swizzle_output_ = split_k_is_last_slice && Base::split_k_kernels_ == 1;

        // The pointer to the buffer.
        char *ptr = reinterpret_cast<char *>( Base::split_k_buffers_ptr_ );
        Base::split_k_buffer_ptr_ = &ptr[split_k_buffer * Base::split_k_buffer_size_];

        // Output to memory. Should we make this loop part of the epilogue object?

        // Make sure the main loop are finished.
        if( Base::epi_sync_.invalid() ) {
            __syncthreads();
        } else {
            Base::epi_sync_.wait();
        }

        #pragma unroll
        for( int si = 0; si < STRIDED; ++si ) {

            if( !Base::Traits::USE_SPLIT_K_WITH_OUTPUT_PRECISION ) {  // no need to colwert before
                                                                      // split_k
                // Do the split-k before the swizzle (when WARPS_K == 1).
                if( Base::Cta_tile::WARPS_K == 1 && Base::split_k_slices_ > 1 ) {
                    this->split_k( si, 0, 0, acc[si] );
                }
                
                // Early-exit if the CTA does not have extra work to do.
                if( Base::Cta_tile::WARPS_K == 1 && !Base::split_k_must_swizzle_output_ ) {
                    continue;
                }
            }

            // Do something before we colwert.
            #pragma unroll
            for( int ci = 0; ci < CONTIGUOUS; ++ci ) {
                Base::callbacks_.pre_colwert( *this, si, ci, acc[si][ci] );
            }

            #pragma unroll
            for( int ni = 0; ni < Base::LDS_ITERATIONS_PER_XMMA_N; ++ni ) {
                #pragma unroll
                for( int mi = 0; mi < Base::LDS_ITERATIONS_PER_XMMA_M; ++mi ) {

                    // Take groups into account and shuffle data.
                    typename Base::Fragment_pre_swizzle pre_swizzle[CONTIGUOUS];
                    #pragma unroll
                    for( int ci = 0; ci < CONTIGUOUS; ++ci ) {
                        pre_swizzle[ci].shuffle_groups( acc[si][ci] );
                    }

                    // Load alpha.
                    typename Base::Fragment_alpha_pre_swizzle alpha_pre_swizzle[CONTIGUOUS];
                    #pragma unroll
                    for( int ci = 0; ci < CONTIGUOUS; ++ci ) {
                        Base::callbacks_.alpha_pre_swizzle( *this, si, ci, alpha_pre_swizzle[ci] );
                    }
                    int offset = mi + ni * ( Base::TILE_N / 8 ) * 2;
                    // Colwert the aclwmulators to the epilogue format (or keep them as-is).
                    #pragma unroll
                    for( int ci = 0; ci < CONTIGUOUS; ++ci ) {
                        pre_swizzle[ci].colwert( offset, alpha_pre_swizzle[ci], acc[si][ci] );
                    }

                    // Do something before we swizzle.
                    #pragma unroll
                    for( int ci = 0; ci < CONTIGUOUS; ++ci ) {
                        Base::callbacks_.pre_swizzle( *this, si, ci, pre_swizzle[ci] );
                    }

                    // Do one iteration of the loop.
                    this->template step<WITH_RESIDUAL>( si, mi, ni, pre_swizzle );

                    // Make sure this loop of the epilogue are finished.
                    if( Base::epi_sync_.invalid() ) {
                        __syncthreads();
                    } else {
                        Base::epi_sync_.wait();
                    }
                }
            }
        }

        // If we do not need the counters for split-k, we are good to go.
        if( split_k_skip_atomics || !with_unlock ) {
            return;
        }

        // Make sure all threads are done issueing.
        if( Base::epi_sync_.invalid() ) {
            __syncthreads();
        } else {
            Base::epi_sync_.wait();
        }

        // Update the counters -- release the locks.
        if( Base::tidx_ != 0 ) {
            return;
        }

        // Before we update the lock, we need all writes to be issued and visible.
        __threadfence();

        // We can update the buffer lock and quit.
        if( !split_k_is_last_slice ) {
            atomicAdd( split_k_counter_ptr, 1 );
        }

        // That's the sum of CTAs that are done.
        atomicAdd( split_k_retired_ctas_ptr, 1 );
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    // The instruction traits.
    typename Traits_,
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile_,
    // The layout of the tile. Specialized for xmma::Row.
    // typename Layout_,
    // The global memory tile to store the output.
    typename Gmem_tile_,
    // The callbacks to lwstomize the behaviour of the epilogue.
    typename Callbacks,
    // The class to swizzle the data.
    typename Swizzle_,
    // The fragment class before the swizzling.
    typename Fragment_pre_swizzle_,
    // The fragment class after the swizzling.
    typename Fragment_post_swizzle_,
    // The output fragment.
    typename Fragment_c_>
struct Epilogue_gmma_with_split_k<Traits_,
                                  Cta_tile_,
                                  xmma::Col,
                                  Gmem_tile_,
                                  Callbacks,
                                  Swizzle_,
                                  Fragment_pre_swizzle_,
                                  Fragment_post_swizzle_,
                                  Fragment_c_>
    : public Epilogue_gmma_with_split_k_base<Traits_,
                                             Cta_tile_,
                                             xmma::Col,
                                             Gmem_tile_,
                                             Callbacks,
                                             Swizzle_,
                                             Fragment_pre_swizzle_,
                                             Fragment_post_swizzle_,
                                             Fragment_c_> {
    // The base class.
    using Base = Epilogue_gmma_with_split_k_base<Traits_,
                                                 Cta_tile_,
                                                 xmma::Col,
                                                 Gmem_tile_,
                                                 Callbacks,
                                                 Swizzle_,
                                                 Fragment_pre_swizzle_,
                                                 Fragment_post_swizzle_,
                                                 Fragment_c_>;

    // Ctor.
    template <typename Params>
    inline __device__ Epilogue_gmma_with_split_k( const Params &params,
                                                  typename Base::Gmem_tile &gmem_tile,
                                                  typename Base::Swizzle &swizzle,
                                                  typename Base::Callbacks &callbacks,
                                                  const Named_barrier &epi_sync = Named_barrier(),
                                                  const int bidm = blockIdx.x,
                                                  const int bidn = blockIdx.y,
                                                  const int bidz = blockIdx.z,
                                                  const int tidx = threadIdx.x,
                                                  const bool is_warp_specialized = false )
        : Base( params,
                gmem_tile,
                swizzle,
                callbacks,
                epi_sync,
                bidm,
                bidn,
                bidz,
                tidx,
                is_warp_specialized ) {
    }

    // Deterministic reduction. 1st kernel.
    template <bool WITH_RESIDUAL, typename Fragment_aclwmulator, int STRIDED, int CONTIGUOUS>
    inline __device__ void execute( Fragment_aclwmulator ( &acc )[CONTIGUOUS][STRIDED],
                                    int with_lock = 1,
                                    int with_unlock = 1 ) {

        // The number of CTAs per slice.
        const int ctas_per_slice = Base::tiles_x_ * Base::tiles_y_;
        // The position of the CTA in the X*Y slice.
        const int cta_in_slice = Base::bidn_ * Base::tiles_x_ + Base::bidm_;

        // Is it the last CTA working on a given tile?
        int split_k_is_last_slice =
            ( Base::bidz_ == Base::split_k_slices_ - 1 ) || ( Base::is_batched_ );
        // The number of reduction steps for split-k.
        Base::split_k_reduction_steps_ = 0;
        // The buffer in global memory where to put the data for split-k.
        int split_k_buffer = 0;
        // Do we skip the atomics.
        int split_k_skip_atomics = 0;

        // The counter. One CTA at a time (we could also do it at the warp level).
        int32_t *split_k_counter_ptr = 0;
        // The offset to the number of retired CTAs.
        int32_t *split_k_retired_ctas_ptr = 0;

        // No reduction.
        if( Base::split_k_slices_ == 1 ) {

            split_k_skip_atomics = 1;

            // Each slice has its own buffer.
        } else if( Base::split_k_kernels_ == 2 &&
                   Base::split_k_slices_ == Base::split_k_buffers_ ) {

            // The buffer. It may change later if slices > buffers (see below).
            split_k_buffer = Base::bidz_;
            // No need to increase the counters.
            split_k_skip_atomics = 1;

            // If we enable split-k, the last slice does the final reduction.
        } else if( split_k_is_last_slice && Base::split_k_kernels_ == 1 ) {

            // The starting buffer is 0.
            split_k_buffer = 0;
            // Usually we have split_k_buffers_ <= split_k_slices_. When split_k_buffers_ ==
            // split_k_slices_, the last slice holds its data in shared memory. So, it doesn't need
            // to load its buffer and it shouldn't load the buffer as the buffer is not initialized.
            Base::split_k_reduction_steps_ = Base::split_k_buffers_;
            if( Base::split_k_slices_ == Base::split_k_buffers_ ) {
                Base::split_k_reduction_steps_ -= 1;
            }
            // The total number of retired CTAs per tile.
            split_k_retired_ctas_ptr = &( Base::split_k_retired_ctas_ptr_ )[cta_in_slice];

            // Wait for all the CTAs to be done with their steps.
            xmma::spin_lock( split_k_retired_ctas_ptr,
                             with_lock ? Base::split_k_slices_ - 1 : 0,
                             0,
                             Base::tidx_,
                             Base::epi_sync_ );

            // If CTAs do a sequential reduction first, acquire the lock on the buffer.
        } else {

            // The corresponding buffer (when we have multiple buffers).
            split_k_buffer = Base::bidz_ % Base::split_k_buffers_;
            // The number of CTAs before this one.
            int predecessors = Base::bidz_ / Base::split_k_buffers_;
            // Do 1 parallel step unless that's the 1st CTA to write to the buffer.
            Base::split_k_reduction_steps_ = predecessors == 0 ? 0 : 1;
            // The total number of retired CTAs per tile.
            split_k_retired_ctas_ptr = &( Base::split_k_retired_ctas_ptr_ )[cta_in_slice];

            // The counter in global memory.
            int counter = split_k_buffer * ctas_per_slice + cta_in_slice;
            // The counter. One CTA at a time (we could also do it at the warp level).
            split_k_counter_ptr = &( Base::split_k_counters_ptr_ )[counter];
            // Wait for all the CTAs preceding this one to be done with the buffer.
            xmma::spin_lock( split_k_counter_ptr,
                             with_lock ? predecessors : 0,
                             0,
                             Base::tidx_,
                             Base::epi_sync_ );
        }

        // Do we do the swizzle of the output.
        Base::split_k_must_swizzle_output_ = split_k_is_last_slice && Base::split_k_kernels_ == 1;

        // The pointer to the buffer.
        char *ptr = reinterpret_cast<char *>( Base::split_k_buffers_ptr_ );
        Base::split_k_buffer_ptr_ = &ptr[split_k_buffer * Base::split_k_buffer_size_];

        // Output to memory. Should we make this loop part of the epilogue object?

        // Make sure the main loop are finished.
        if( Base::epi_sync_.invalid() ) {
            __syncthreads();
        } else {
            Base::epi_sync_.wait();
        }

        #pragma unroll
        for( int si = 0; si < STRIDED; ++si ) {

            Fragment_aclwmulator cont_acc[CONTIGUOUS];
            #pragma unroll
            for( int ci = 0; ci < CONTIGUOUS; ++ci ) {
                cont_acc[ci] = acc[ci][si];
            }

            if( !Base::Traits::USE_SPLIT_K_WITH_OUTPUT_PRECISION ) {
                // Do the split-k before the swizzle (when WARPS_K == 1).
                if( Base::Cta_tile::WARPS_K == 1 && Base::split_k_slices_ > 1 ) {
                    this->split_k( si, 0, 0, cont_acc );
                }

                // Early-exit if the CTA does not have extra work to do.
                if( Base::Cta_tile::WARPS_K == 1 && !Base::split_k_must_swizzle_output_ ) {
                    continue;
                }
            }

            // Do something before we colwert.
            #pragma unroll
            for( int ci = 0; ci < CONTIGUOUS; ++ci ) {
                Base::callbacks_.pre_colwert( *this, ci, si, cont_acc[ci] );
            }

            #pragma unroll
            for( int ni = 0; ni < Base::LDS_ITERATIONS_PER_XMMA_N; ++ni ) {
                #pragma unroll
                for( int mi = 0; mi < Base::LDS_ITERATIONS_PER_XMMA_M; ++mi ) {

                    // Take groups into account and shuffle data.
                    typename Base::Fragment_pre_swizzle pre_swizzle[CONTIGUOUS];
                    #pragma unroll
                    for( int ci = 0; ci < CONTIGUOUS; ++ci ) {
                        pre_swizzle[ci].shuffle_groups( cont_acc[ci] );
                    }

                    // Load alpha.
                    typename Base::Fragment_alpha_pre_swizzle alpha_pre_swizzle[CONTIGUOUS];
                    #pragma unroll
                    for( int ci = 0; ci < CONTIGUOUS; ++ci ) {
                        Base::callbacks_.alpha_pre_swizzle( *this, si, ci, alpha_pre_swizzle[ci] );
                    }

                    int offset = mi + ni * ( Base::TILE_N / 8 ) * 2;
                    // Colwert the aclwmulators to the epilogue format (or keep them as-is).
                    #pragma unroll
                    for( int ci = 0; ci < CONTIGUOUS; ++ci ) {
                        pre_swizzle[ci].colwert( offset, alpha_pre_swizzle[ci], cont_acc[ci] );
                    }

                    // Do something before we swizzle.
                    #pragma unroll
                    for( int ci = 0; ci < CONTIGUOUS; ++ci ) {
                        Base::callbacks_.pre_swizzle( *this, ci, si, pre_swizzle[ci] );
                    }

                    // Do one iteration of the loop.
                    this->template step<WITH_RESIDUAL>( si, mi, ni, pre_swizzle );

                    // Make sure this loop of the epilogue are finished.
                    if( Base::epi_sync_.invalid() ) {
                        __syncthreads();
                    } else {
                        Base::epi_sync_.wait();
                    }
                }
            }
        }

        // If we do not need the counters for split-k, we are good to go.
        if( split_k_skip_atomics || !with_unlock ) {
            return;
        }

        // Make sure all threads are done issueing.
        if( Base::epi_sync_.invalid() ) {
            __syncthreads();
        } else {
            Base::epi_sync_.wait();
        }

        // Update the counters -- release the locks.
        if( Base::tidx_ != 0 ) {
            return;
        }

        // Before we update the lock, we need all writes to be issued and visible.
        __threadfence();

        // We can update the buffer lock and quit.
        if( !split_k_is_last_slice ) {
            atomicAdd( split_k_counter_ptr, 1 );
        }

        // That's the sum of CTAs that are done.
        atomicAdd( split_k_retired_ctas_ptr, 1 );
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace helpers
} // namepsace xmma
