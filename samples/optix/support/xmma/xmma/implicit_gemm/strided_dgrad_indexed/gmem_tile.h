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
#include <xmma/named_barrier.h>

///////////////////////////////////////////////////////////////////////////////////////////////////

namespace xmma {
namespace implicit_gemm {
namespace strided_dgrad_indexed {

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// T I L E   A
//
///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits, typename Cta_tile> struct Gmem_tile_a {
    // The number of bytes per LDG.
    enum { BYTES_PER_LDG = 16 };

    // The number of elements per LDG.128.
    enum { ELEMENTS_PER_LDG = BYTES_PER_LDG * 8 / Traits::BITS_PER_ELEMENT_A };
    // Make sure we have a "nice" number of elements per LDG.
    static_assert( ELEMENTS_PER_LDG > 0, "" );

    // The number of threads needed to load a pixel. Each thread does LDG.128.
    enum { THREADS_PER_PIXEL = Cta_tile::K / ELEMENTS_PER_LDG };
    // Make sure we have a "nice" number of pixels.
    static_assert( THREADS_PER_PIXEL > 0, "" );

    // The number of images loaded per LDG.
    enum { PIXELS_PER_LDG = Cta_tile::THREADS_PER_CTA / THREADS_PER_PIXEL };
    // Make sure we have a "nice" number of pixels.
    static_assert( Cta_tile::M % PIXELS_PER_LDG == 0, "" );

    // The number of steps needed to load the pixels.
    enum { LDGS = Cta_tile::M / PIXELS_PER_LDG };
    // Make sure we have a "nice" number of LDGs.
    static_assert( LDGS > 0, "" );

    // The number of predicates that we store per register.
    enum { PREDS_PER_REG = 4 };

    enum { HAS_LDGSTS = Traits::Gpu_arch::HAS_LDGSTS };

    // Ctor.
    template <typename Params>
    inline __device__
    Gmem_tile_a( const Params& params,
                 int* cta_ndhw_indices,
                 int cta_id_in_dhw_dimension,
                 int start_t_index,
                 int start_r_index,
                 int start_s_index,
                 int valid_t_number,
                 int valid_r_number,
                 int valid_s_number,
                 int bidm,
                 int bidn,
                 int bidz,
                 int tidx,
                 int4* smem_cta_batch_depth_height_width_indices,
                 xmma::Named_barrier intra_dma_group_sync = xmma::Named_barrier() )
        : params_k_( params.g * params.k ), params_residue_k_( params.loop_residue_k ),
          params_split_k_k_( params.split_k_k ), params_o_( params.o ), params_p_( params.p ),
          params_q_( params.q ), params_step_o_( params.step_o ), params_step_p_( params.step_p ),
          params_step_q_( params.step_q ),
          smem_cta_batch_depth_height_width_indices_( smem_cta_batch_depth_height_width_indices ),
          bidz_( bidz ), tidx_( tidx ) {
        // The position in the K dimension.
        int k_in_tile = bidz * params_split_k_k_ + tidx % THREADS_PER_PIXEL * ELEMENTS_PER_LDG;
        int k = k_in_tile;
        if( params.g > 1 ) {
            k += bidn * Cta_tile::N;
        }

        const int DHW_PER_CTA = Cta_tile::M;
        const int DHW_PER_THREAD =
            ( DHW_PER_CTA + Cta_tile::THREADS_PER_CTA - 1 ) / Cta_tile::THREADS_PER_CTA;
        static_assert( DHW_PER_THREAD <= LDGS, "DHW_PER_THREAD<=LDGS" );
        for( int i = 0; i < DHW_PER_THREAD; ++i ) {
            int dhw_index = tidx + i * Cta_tile::THREADS_PER_CTA;
            if( dhw_index < Cta_tile::M ) {
                cta_ndhw_indices[dhw_index] =
                    params.ndhw_indices_of_each_filter_pattern_gmem[cta_id_in_dhw_dimension *
                                                                        DHW_PER_CTA +
                                                                    dhw_index];
            }
        }

        // The update in the K dimension.
        int move_k = Cta_tile::K;
        if( params.split_k.slices > 1 && params.split_k_k > 0 ) {
            move_k *= params.split_k.slices;
        }
        params_delta_a_opqk_ = Traits::offset_in_bytes_a(
            move_k - ( valid_t_number - 1 ) * params.step_o * params.pqk -
            ( valid_r_number - 1 ) * params.step_p * params.qk -
            ( valid_s_number - 1 ) * params.step_q * params.g * params.k );
        params_delta_a_opq_ = Traits::offset_in_bytes_a(
            params.step_o * params.pqk - ( valid_r_number - 1 ) * params.step_p * params.qk -
            ( valid_s_number - 1 ) * params.step_q * params.g * params.k );
        params_delta_a_pq_ = Traits::offset_in_bytes_a( params.step_p * params.qk -
                                                        ( valid_s_number - 1 ) * params.step_q *
                                                            params.g * params.k );
        params_delta_a_q_ = Traits::offset_in_bytes_a( params.step_q * params.g * params.k );

        // Define the single pointer.
        ptr_ = reinterpret_cast<const char*>( params.img_gmem );

#pragma unroll
        for( int mi = 0; mi < LDGS; ++mi ) {
            if( start_t_index + start_r_index + start_s_index < 0 ) {
                masks_base_[mi] = 0u;
            } else {
                masks_base_[mi] = 0xffffffffu;
            }
        }
        // For cta_ndhw_indices.
        //__syncthreads();
        int err_n[LDGS], err_o[LDGS], err_p[LDGS], err_q[LDGS];
        int4 packed_err_o_err_p_and_err_q;
        for( int i = 0; i < DHW_PER_THREAD; ++i ) {
            int dhw_index = tidx + i * Cta_tile::THREADS_PER_CTA;
            int img_ndhw( -1 );
            if( dhw_index < Cta_tile::M ) {
                img_ndhw = cta_ndhw_indices[dhw_index];
            }
            // Decompose NDHW into N and DHW.
            int img_dhw;
            xmma::fast_divmod(
                err_n[i], img_dhw, img_ndhw, params.dhw, params.mul_dhw, params.shr_dhw );
            // Decompose DHW into D and HW.
            int img_d, img_hw;
            xmma::fast_divmod( img_d, img_hw, img_dhw, params.hw, params.mul_hw, params.shr_hw );
            // Decompose HW into H and W.
            int img_h, img_w;
            xmma::fast_divmod( img_h, img_w, img_hw, params.w, params.mul_w, params.shr_w );

            // Compute o, p and q.
            int tmp_err_o = img_d + params.pad[0][0] - start_t_index * params.dilation[0];
            int tmp_err_p = img_h + params.pad[1][0] - start_r_index * params.dilation[1];
            int tmp_err_q = img_w + params.pad[2][0] - start_s_index * params.dilation[2];

            int err_o_reminder, err_p_reminder, err_q_reminder;
            xmma::fast_divmod( err_o[i],
                               err_o_reminder,
                               tmp_err_o,
                               params.stride[0],
                               params.mul_stride_d,
                               params.shr_stride_d );
            xmma::fast_divmod( err_p[i],
                               err_p_reminder,
                               tmp_err_p,
                               params.stride[1],
                               params.mul_stride_h,
                               params.shr_stride_h );
            xmma::fast_divmod( err_q[i],
                               err_q_reminder,
                               tmp_err_q,
                               params.stride[2],
                               params.mul_stride_w,
                               params.shr_stride_w );

            packed_err_o_err_p_and_err_q = make_int4( err_n[i], err_o[i], err_p[i], err_q[i] );
            if( dhw_index < Cta_tile::M ) {
                smem_cta_batch_depth_height_width_indices_[dhw_index] =
                    packed_err_o_err_p_and_err_q;
            }
        }
        // For each LDG, compute the NPQ decomposition, the masks and the
        // pointer.
        start_index_in_smem_cta_batch_depth_height_width_indices_ = tidx / THREADS_PER_PIXEL;
        uint32_t masks[LDGS];
        // For smem_cta_batch_depth_height_width_indices_, we won't write data to the buffer
        // anymore,
        // we only read data from it (here and the move() function).
        //__syncthreads();
        if( intra_dma_group_sync.invalid() ) {
            __syncthreads();
        } else {
            intra_dma_group_sync.wait();
        }

#pragma unroll
        for( int mi = 0; mi < LDGS; ++mi ) {
            int row_index_in_a_cta =
                start_index_in_smem_cta_batch_depth_height_width_indices_ + mi * PIXELS_PER_LDG;
            int img_ndhw = cta_ndhw_indices[row_index_in_a_cta];
            packed_err_o_err_p_and_err_q =
                smem_cta_batch_depth_height_width_indices_[row_index_in_a_cta];

            // The masks -- initialization.
            if( k >= params_k_ || img_ndhw < 0 ) {
                masks_base_[mi] = 0u;
            }

            if ( Cta_tile::N < Cta_tile::K && params.g > 1 ) {
                if ( k_in_tile > Cta_tile::N ) {
                    masks_base_[mi] = 0u;
                }
            }

            // Compute h and w.
            err_n[mi] = packed_err_o_err_p_and_err_q.x;
            err_o[mi] = packed_err_o_err_p_and_err_q.y;
            err_p[mi] = packed_err_o_err_p_and_err_q.z;
            err_q[mi] = packed_err_o_err_p_and_err_q.w;

            // Compute the offset.
            offsets_[mi] = err_n[mi] * params.opqk + err_o[mi] * params.pqk +
                           err_p[mi] * params.qk + err_q[mi] * params.g * params.k + k;

            masks[mi] = 0u;
            if( ( (unsigned)err_q[mi] < params.q ) && ( (unsigned)err_p[mi] < params.p ) &&
                ( (unsigned)err_o[mi] < params.o ) ) {
                masks[mi] = masks_base_[mi];
            }
        }
        // Pack the predicates.
        preds_ = xmma::pack_predicates( masks );
    }

    // Store the pixels to shared memory.
    template <typename Xmma_smem_tile> inline __device__ void commit( Xmma_smem_tile& smem ) {
        if ( !HAS_LDGSTS ) {
            smem.store( fetch_ );
        }
    }

    // Disable the loads.
    inline __device__ void disable_loads() {
        preds_ = 0u;
    }

    // Load a tile from global memory.
    inline __device__ void load( uint64_t mem_desc = MEM_DESC_DEFAULT ) {
        const void* ptrs[LDGS];
#pragma unroll
        for( int ii = 0; ii < LDGS; ++ii ) {
            ptrs[ii] = ptr_ + Traits::offset_in_bytes_a( offsets_[ii] );
        }
        xmma::ldg( fetch_, ptrs, preds_, mem_desc );
    }

    // Load a tile from global memory.
    template <typename Xmma_smem_tile>
    inline __device__ void load( Xmma_smem_tile &smem, uint64_t mem_desc = MEM_DESC_DEFAULT ) {
        const void* ptrs[LDGS];
#pragma unroll
        for( int ii = 0; ii < LDGS; ++ii ) {
            ptrs[ii] = ptr_ + Traits::offset_in_bytes_a( offsets_[ii] );
        }
        // Issue the ldgsts.
        smem.store( ptrs, preds_, mem_desc );
    }

    // Move the pointers and update the predicates for R2P/P2R.
    inline __device__ void move( int lwrrent_filter_t,
                                 int lwrrent_filter_r,
                                 int lwrrent_filter_s,
                                 int lwrrent_filter_rs,
                                 int lwrrent_filter_trs ) {
        int64_t delta = params_delta_a_q_;
        if( lwrrent_filter_s == 0 ) {
            delta = params_delta_a_pq_;
        }
        if( lwrrent_filter_rs == 0 ) {
            delta = params_delta_a_opq_;
        }
        if( lwrrent_filter_trs == 0 ) {
            delta = params_delta_a_opqk_;
        }
        ptr_ += delta;
        int err_o, err_p, err_q;
        int4 packed_err_o_err_p_and_err_q;
        uint32_t masks[LDGS];
        int row_index_in_a_cta;

        lwrrent_filter_t *= params_step_o_;
        lwrrent_filter_r *= params_step_p_;
        lwrrent_filter_s *= params_step_q_;

#pragma unroll
        for( int mi = 0; mi < LDGS; ++mi ) {
            row_index_in_a_cta =
                start_index_in_smem_cta_batch_depth_height_width_indices_ + mi * PIXELS_PER_LDG;
            packed_err_o_err_p_and_err_q =
                smem_cta_batch_depth_height_width_indices_[row_index_in_a_cta];
            // Compute h and w. We implement a cross-correlation.
            err_o = packed_err_o_err_p_and_err_q.y + lwrrent_filter_t;
            err_p = packed_err_o_err_p_and_err_q.z + lwrrent_filter_r;
            err_q = packed_err_o_err_p_and_err_q.w + lwrrent_filter_s;

            masks[mi] = 0u;
            // Assemble the 1st bit of the masks.
            if( ( (unsigned)err_o < params_o_ ) && ( (unsigned)err_p < params_p_ ) &&
                ( (unsigned)err_q < params_q_ ) ) {
                masks[mi] = masks_base_[mi];
            }
        }
        preds_ = xmma::pack_predicates( masks );
    }

    // The residue to "fix" the predicates.
    inline __device__ void residue() {

        // The position in the K dimension.
        int k = bidz_ * params_split_k_k_ + tidx_ % THREADS_PER_PIXEL * ELEMENTS_PER_LDG;

        // Jump back to the loop if we have nothing to do.
        if( params_residue_k_ + k < params_k_ ) {
            return;
        }

        // Disable the predicates.
        preds_ = 0u;
    }

    const int tidx_, bidz_;
    // The K dimension.
    const int params_k_, params_residue_k_, params_split_k_k_;
    // The pointer.
    const char* ptr_;
    // The associated offsets.
    int offsets_[LDGS];
    int params_o_, params_p_, params_q_;
    int params_step_o_, params_step_p_, params_step_q_;
    int64_t params_delta_a_q_, params_delta_a_pq_, params_delta_a_opq_, params_delta_a_opqk_;
    int4* smem_cta_batch_depth_height_width_indices_;
    int start_index_in_smem_cta_batch_depth_height_width_indices_;
    // The masks.
    uint32_t masks_base_[LDGS], preds_;
    // The fetch registers.
    uint4 fetch_[LDGS];
};

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// T I L E   B
//
////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits, typename Cta_tile> struct Gmem_tile_b {

    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;

    // The number of bytes per LDG.
    enum { BYTES_PER_LDG = 16 };
    // The number of elements per LDG.128.
    enum { ELEMENTS_PER_LDG = BYTES_PER_LDG * 8 / Traits::BITS_PER_ELEMENT_B };
    // The number of threads needed to load a channel. Each thread does LDG.128.
    enum { THREADS_PER_FILTER = Cta_tile::N / ELEMENTS_PER_LDG };
    // The number of filters loaded per LDG.
    enum { FILTERS_PER_LDG = Cta_tile::THREADS_PER_CTA / THREADS_PER_FILTER };

    // Make sure we have a "nice" number of channels.
    static_assert( Cta_tile::K % FILTERS_PER_LDG == 0, "" );
    // The number of steps needed to load the filters.
    enum { LDGS = Cta_tile::K / FILTERS_PER_LDG };

    enum { HAS_LDGSTS = Traits::Gpu_arch::HAS_LDGSTS };
    
    // Ctor.
    template <typename Params>
    inline __device__ Gmem_tile_b( const Params& params,
                                   int start_t_index,
                                   int start_r_index,
                                   int start_s_index,
                                   int valid_t_number,
                                   int valid_r_number,
                                   int valid_s_number,
                                   int bidn,
                                   int bidz,
                                   int tidx )
        : params_k_( params.g * params.k ), params_delta_k_( FILTERS_PER_LDG * params.trsc ),
          params_residue_k_( params.loop_residue_k ), params_split_k_k_( params.split_k_k ),
          bidz_( bidz ), tidx_( tidx ) {
        // The coordinate in the K dimension.
        int k_in_tile = bidz * params_split_k_k_ + tidx / THREADS_PER_FILTER;
        int k = k_in_tile;
        // The coordinates of the elements loaded by the thread.
        int c_in_tile = ( tidx & ( THREADS_PER_FILTER - 1 ) ) * ELEMENTS_PER_LDG;
        int c = bidn * Cta_tile::N + c_in_tile;

        int c_in_a_group = c;
        if( params.g > 1 ) {
            k += bidn * Cta_tile::N;
            c_in_a_group = c % ( Cta_tile::N / Cta_tile::GROUPS );
        }

        int move_flt_k = Cta_tile::K * params.trs * params.c;
        if( params.split_k.slices > 1 && params.split_k_k > 0 ) {
            move_flt_k *= params.split_k.slices;
        }
        int filter_direction = ( params.cross_correlation ? 1 : -1 );

        params_delta_b_trsk_ =
            Traits::offset_in_bytes_b( move_flt_k +
                                       ( params.step_t * params.rsc * ( valid_t_number - 1 ) +
                                         params.step_r * params.sc * ( valid_r_number - 1 ) +
                                         params.step_s * params.c * ( valid_s_number - 1 ) ) *
                                           filter_direction );
        params_delta_b_trs_ = Traits::offset_in_bytes_b(
            ( params.step_s * params.c * ( valid_s_number - 1 ) +
              params.step_r * params.sc * ( valid_r_number - 1 ) - params.step_t * params.rsc ) *
            filter_direction );
        params_delta_b_rs_ = Traits::offset_in_bytes_b(
            ( params.step_s * params.c * ( valid_s_number - 1 ) - params.step_r * params.sc ) *
            filter_direction );
        params_delta_b_s_ =
            Traits::offset_in_bytes_b( -params.step_s * params.c * filter_direction );

        // We treat the filter as a KRS x C matrix.
        int ktrs = k * params.trs;
        int real_t_index =
            params.cross_correlation ? start_t_index : ( params.t - 1 - start_t_index );
        int real_r_index =
            params.cross_correlation ? start_r_index : ( params.r - 1 - start_r_index );
        int real_s_index =
            params.cross_correlation ? start_s_index : ( params.s - 1 - start_s_index );
        ktrs += real_t_index * params.rs + real_r_index * params.s + real_s_index;

        // Assemble the base pointer.
        const char* ptr = reinterpret_cast<const char*>( params.flt_gmem );
        ptr_ = &ptr[Traits::offset_in_bytes_b( ktrs * params.c + c_in_a_group )];

        // Compute the predicates.
        uint32_t preds[LDGS];
#pragma unroll
        for( int ii = 0; ii < LDGS; ++ii ) {
            preds[ii] = k + ii * FILTERS_PER_LDG < params_k_;
        }

        if ( Cta_tile::N < Cta_tile::K && params.g > 1 ) {
#pragma unroll
            for( int ii = 0; ii < LDGS; ++ii ) {
                preds[ii] &= ( k_in_tile + ii*FILTERS_PER_LDG < Cta_tile::N );
            }
        }
        if ( HAS_LDGSTS ) {
            // C=K=8 or C=K=4 - Each character is a 8x8 block.
            //
            // a 0 c 0 e 0 g 0
            // 0 b 0 d 0 f 0 h
            // 0 0 0 0 0 0 0 0
            // 0 0 0 0 0 0 0 0
            // 0 0 0 0 0 0 0 0
            // 0 0 0 0 0 0 0 0
            // 0 0 0 0 0 0 0 0
            // 0 0 0 0 0 0 0 0
            //
            if( Traits::BITS_PER_ELEMENT_B == 16 ) {
                if( Cta_tile::GROUPS == 8 || Cta_tile::GROUPS == 16 ) {
                    int row_idx = tidx_ % THREADS_PER_FILTER;
                    int col_idx = tidx_ / THREADS_PER_FILTER / 8;
                    if( ( row_idx & 1 ) ^ ( col_idx & 1 ) ) {
#pragma unroll
                        for( int ii = 0; ii < LDGS; ++ii ) {
                            preds[ii] = 0;
                        }
                    }
                }
            } else if( Traits::BITS_PER_ELEMENT_B == 32 ) {
                int row_idx = tidx_ % THREADS_PER_FILTER;
                if( Cta_tile::GROUPS == 8 ) {
#pragma unroll
                    for( int ii = 0; ii < LDGS; ++ii ) {
                        if( row_idx % 4 < 2 ) {
                            preds[ii] = ii % 2 == 0;
                        } else {
                            preds[ii] = ii % 2 == 1;
                        }
                    }
                } else if( Cta_tile::GROUPS == 16 ) {
                    // C=K=4 - Each character is a 4x4 block.
                    // We use the format directly.
                    // a 0 0 0
                    // 0 b 0 0
                    // 0 0 c 0
                    // 0 0 0 d
                    //
                    int col_idx = tidx_ / THREADS_PER_FILTER / 4;
#pragma unroll
                    for( int ii = 0; ii < LDGS; ++ii ) {
                        preds[ii] = 0;
                        if( ( row_idx == 0 && col_idx == 0 ) || ( row_idx == 1 && col_idx == 1 ) ) {
                            preds[ii] = ii % 2 == 0;
                        } else if( ( row_idx == 2 && col_idx == 0 ) ||
                                   ( row_idx == 3 && col_idx == 1 ) ) {
                            preds[ii] = ii % 2 == 1;
                        }
                    }
                }
            }
        }


        // Finalize the predicates.
        if( start_t_index + start_r_index + start_s_index >= 0 ) {
            preds_ = ( ( c < params.g * params.c ) &&
                               ( c_in_tile < Cta_tile::N / Xmma_tile::XMMAS_GROUPS )
                           ? 0xffffffffu
                           : 0x0u ) &
                     xmma::pack_predicates( preds );
        } else {
            preds_ = 0u;
        }
    }

        // Set redundant data to zero.
    inline __device__ void remove_group_redundant() {
        if ( !HAS_LDGSTS ) {
            // C=K=4
            // Each character is a 4x4 block.
            // a 0 b 0 -> a a b b
            // c 0 d 0 -> c c d d
            // e 0 f 0 -> e e f f
            // g 0 h 0 -> g g h h
            if( Cta_tile::GROUPS == 16 ) {
#pragma unroll
                for( int ii = 0; ii < LDGS; ++ii ) {
                    fetch_[ii].z = fetch_[ii].x;
                    fetch_[ii].w = fetch_[ii].y;
                }
            }
            // C=K=4 and C=K=8
            // Each character is a 8x8 block.
            // a b -> a 0
            // c d -> 0 d
            if( Cta_tile::GROUPS == 8 || Cta_tile::GROUPS == 16 ) {
                if( ( tidx_ % THREADS_PER_FILTER & 1 ) ^ ( tidx_ / THREADS_PER_FILTER / 8 & 1 ) ) {
#pragma unroll
                    for( int ii = 0; ii < LDGS; ++ii ) {
                        fetch_[ii] = make_uint4( 0, 0, 0, 0 );
                    }
                }
            }
            // C=K=4
            // Each character is a 4x4 block.
            // warp0 a a 0 0 -> a 0 0 0
            // warp1 b b 0 0 -> 0 b 0 0
            // warp0 0 0 c c -> 0 0 c 0
            // warp1 0 0 d d -> 0 0 0 d
            if( Cta_tile::GROUPS == 16 ) {
                if( ( tidx_ / Cta_tile::THREADS_PER_WARP & 1 ) == 0 ) {
#pragma unroll
                    for( int ii = 0; ii < LDGS; ++ii ) {
                        fetch_[ii].z = 0;
                        fetch_[ii].w = 0;
                    }
                } else {
#pragma unroll
                    for( int ii = 0; ii < LDGS; ++ii ) {
                        fetch_[ii].x = 0;
                        fetch_[ii].y = 0;
                    }
                }
            }
        }
    }

    // Store the pixels to shared memory.
    template <typename Xmma_smem_tile> inline __device__ void commit( Xmma_smem_tile& smem ) {
        if ( !HAS_LDGSTS ) {
            remove_group_redundant();
            smem.store( fetch_ );
        }
    }

    // Disable the loads.
    inline __device__ void disable_loads() {
        preds_ = 0u;
    }

    // Load a tile from global memory.
    inline __device__ void load( uint64_t mem_desc = MEM_DESC_DEFAULT ) {
        // Assemble the pointers.
        const void* ptrs[LDGS];
#pragma unroll
        for( int ii = 0; ii < LDGS; ++ii ) {
            int64_t offset = Traits::offset_in_bytes_b( ii * params_delta_k_ );
            ptrs[ii] = &ptr_[offset];
        }

        // Issue the loads.
        if( Cta_tile::GROUPS == 16 ) {
            // C=K=4, so we need to use LDG.64.
            xmma::ldg_force_64( fetch_, ptrs, preds_ );
        } else {
            xmma::ldg( fetch_, ptrs, preds_, mem_desc );
        }
    }

    // Load a tile from global memory.
    template <typename Xmma_smem_tile>
    inline __device__ void load( Xmma_smem_tile &smem, uint64_t mem_desc = MEM_DESC_DEFAULT ) {
        // Assemble the pointers.
        const void* ptrs[LDGS];
#pragma unroll
        for( int ii = 0; ii < LDGS; ++ii ) {
            int64_t offset = Traits::offset_in_bytes_b( ii * params_delta_k_ );
            ptrs[ii] = &ptr_[offset];
        }

        // Issue the ldgsts.
        if( Cta_tile::GROUPS == 16 && Traits::BITS_PER_ELEMENT_B == 16 ) {
            // C=K=4 - Each character is a 4x4 block.
            //
            // a b 0 0     a 0 0 0
            // 0 0 0 0 --> 0 b 0 0
            // 0 0 c d     0 0 c 0
            // 0 0 0 0     0 0 0 d
            //
            uint32_t smem_str_ptrs[LDGS];

            smem.compute_store_pointers( smem_str_ptrs );
            int is_odd_group = tidx_ / Cta_tile::THREADS_PER_WARP % 2 == 1;
#pragma unroll
            for( int i = 0; i < LDGS; ++i ) {
                smem_str_ptrs[i] = smem_str_ptrs[i] + ( is_odd_group ? 8 : 0 );
            }
            xmma::ldgsts<LDGS, 8>( smem_str_ptrs, ptrs, preds_, mem_desc );
        } else {
            smem.store( ptrs, preds_, mem_desc );
        }
    }

    // Move the pointers.
    inline __device__ void
    move( int lwrrent_filter_s, int lwrrent_filter_rs, int lwrrent_filter_trs ) {
        int64_t delta = params_delta_b_s_;
        if( lwrrent_filter_s == 0 ) {
            delta = params_delta_b_rs_;
        }
        if( lwrrent_filter_rs == 0 ) {
            delta = params_delta_b_trs_;
        }
        if( lwrrent_filter_trs == 0 ) {
            delta = params_delta_b_trsk_;
        }
        ptr_ += delta;
    }

    // The residue to "fix" the predicates.
    inline __device__ void residue() {

        // The position in the K dimension.
        int k = bidz_ * params_split_k_k_ + tidx_ / THREADS_PER_FILTER;

        // Compute the new predicates.
        uint32_t preds[LDGS];
#pragma unroll
        for( int ii = 0; ii < LDGS; ++ii ) {
            preds[ii] = params_residue_k_ + k + ii * FILTERS_PER_LDG < params_k_;
        }

        // Update the predicates.
        preds_ &= xmma::pack_predicates( preds );
    }

    const int bidz_, tidx_;
    // The constant jump between loads.
    const int params_k_, params_delta_k_, params_residue_k_, params_split_k_k_;
    // The pointer.
    const char* ptr_;
    // The predicates to decide if we load.
    int preds_;
    int64_t params_delta_b_s_, params_delta_b_rs_, params_delta_b_trs_, params_delta_b_trsk_;
    // The fetch registers.
    uint4 fetch_[LDGS];
};

///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits,
          typename Cta_tile,
          typename Fragment_c = xmma::Fragment_c<Traits, Cta_tile>>
struct Gmem_tile_epilogue {

    using Layout = xmma::Row;
    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;
    // The helper class to compute the row offset (in the tile).
    using Tile_distribution = xmma::helpers::Gmem_tile_epilogue_distribution<Traits, Cta_tile>;

    // The number of bytes per STG.
    enum { BYTES_PER_STG = 16 };
    // The number of elements per STG per thread.
    enum { ELEMENTS_PER_STG = BYTES_PER_STG * 8 / Traits::BITS_PER_ELEMENT_C };
    // The number of threads per output row (row-major).
    enum { THREADS_PER_ROW = Cta_tile::N / ELEMENTS_PER_STG };
    // The number of rows that are written with a single STG (accross the CTA).
    enum { ROWS_PER_STG = Cta_tile::THREADS_PER_CTA / THREADS_PER_ROW };
    // The number of STGS needed to output the rows produced by a CTA-wide XMMA.
    enum { STGS = Xmma_tile::M_PER_XMMA_PER_CTA / ROWS_PER_STG };

    // Ctor.
    template <typename Params>
    inline __device__
    Gmem_tile_epilogue( const Params& params, int* cta_ndhw_indices, int bidn, int tidx )
        : params_m_( params.ndhw ), params_n_( params.g * params.c ),
          params_stride_n_( params.g * params.c ), params_dhw_( params.dhw ),
          cta_ndhw_indices_( cta_ndhw_indices ) {
        // The location of the tile.
        int row = Tile_distribution::compute_row( tidx );
        int col = Tile_distribution::compute_col( tidx );

        // Compute the output position for each thread.
        m_ = row;
        n_ = bidn * Cta_tile::N + col * ELEMENTS_PER_STG;

        // The pointer.
        out_ptr_ = &( reinterpret_cast<char*>( params.out_gmem ) )[Traits::offset_in_bytes_c( n_ )];
        res_ptr_ =
            &( reinterpret_cast<const char*>( params.res_gmem ) )[Traits::offset_in_bytes_c( n_ )];
    }

    // Is a given output valid?
    inline __device__ int compute_output_mask( int mi, int ii ) const {
        const int offset = Tile_distribution::compute_offset( mi, ii );
        const int row_index_in_a_cta = m_ + offset;
        return cta_ndhw_indices_[row_index_in_a_cta] >= 0 && n_ < params_n_;
    }

    // Load the data from global memory.
    inline __device__ void
    load( Fragment_c& data, int mi, int ii, int mask, const uint64_t& mem_desc ) {
        const int offset = Tile_distribution::compute_offset( mi, ii );
        const int row_index_in_a_cta = m_ + offset;
        const int ndhw_index = cta_ndhw_indices_[row_index_in_a_cta];
        const void* ptr = &res_ptr_[Traits::offset_in_bytes_c( ndhw_index * params_stride_n_ )];
        if( mask ) {
            uint4 tmp;
            xmma::ldg( tmp, ptr );
            data.from_int4( tmp );
        }
    }
    
    // Load residual from gmem to smem buffers.
    template<typename Smem_tile_a, typename Smem_tile_b>
    inline __device__ void load_residual_to_smem(
                                Smem_tile_a &smem_tile_a,
                                Smem_tile_b &smem_tile_b,
                                int mi, 
                                int ii, 
                                int mask,
                                int xmma_tiles_per_a,
                                int xmma_tiles_per_b,
                                int tidx,
                                uint64_t mem_desc = MEM_DESC_DEFAULT) {
      const int offset = Tile_distribution::compute_offset( mi, ii );
      const int row_index_in_a_cta = m_ + offset;
      const int ndhw_index = cta_ndhw_indices_[row_index_in_a_cta];
      const void* ptr = &res_ptr_[Traits::offset_in_bytes_c( ndhw_index * params_stride_n_ )];
      if( mask ) {
        int xmma_tile_idx = (mi * STGS + ii) % (xmma_tiles_per_a + xmma_tiles_per_b);
        uint32_t smem_ptr;

        if (xmma_tile_idx < xmma_tiles_per_a) 
          smem_ptr = smem_tile_a.smem_ + smem_tile_a.smem_write_buffer_ + 
          xmma_tile_idx *  Cta_tile::THREADS_PER_CTA * BYTES_PER_STG + tidx *BYTES_PER_STG;
        else
          smem_ptr = smem_tile_b.smem_ + smem_tile_b.smem_write_buffer_ + 
          (xmma_tile_idx - xmma_tiles_per_a) *  Cta_tile::THREADS_PER_CTA * BYTES_PER_STG + 
          tidx * BYTES_PER_STG;

          ldgsts128(smem_ptr, ptr, true, mem_desc);
      }                                  
    }

    // Store the data to global memory.
    inline __device__ void
    store( int mi, int ii, const Fragment_c& data, int mask, const uint64_t& mem_desc ) {
        const int offset = Tile_distribution::compute_offset( mi, ii );
        const int row_index_in_a_cta = m_ + offset;
        const int ndhw_index = cta_ndhw_indices_[row_index_in_a_cta];
        char* ptr = &out_ptr_[Traits::offset_in_bytes_c( ndhw_index * params_stride_n_ )];
        if( mask ) {
            xmma::stg( ptr, data.to_int4() );
        }
    }

    // The dimensions of the matrix.
    const int params_m_, params_n_, params_stride_n_;
    const int params_dhw_;
    // The position of the tile.
    int m_, n_;
    // The pointer to global memory.
    char* out_ptr_;
    const char* res_ptr_;
    int* cta_ndhw_indices_;
};

///////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace strided_dgrad_indexed
}  // namespace implicit_gemm
}  // namespace xmma
