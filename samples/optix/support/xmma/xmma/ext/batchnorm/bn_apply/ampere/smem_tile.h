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

#include <xmma/ext/batchnorm/bn_apply/smem_tile.h>

namespace xmma {
namespace ext {
namespace batchnorm {
namespace bn_apply {

////////////////////////////////////////////////////////////////////////////////////////////////////

template < typename Cta_tile, int BYTES_PER_STS, int BUFFERS_PER_TILE, bool WITH_RESIDUAL, 
     bool WITH_BNA_RESIDUAL, int LDGS, uint32_t SMEM_B_SIZE, int STAGES_, bool WITH_RELU, 
     bool SIMPLE_1x1x1, bool WITH_BITMASK_RELU_WRITE >
struct Smem_tile_a< xmma::Ampere_hmma_fp32_traits, Cta_tile, xmma::Row, BYTES_PER_STS,
    BUFFERS_PER_TILE, WITH_RESIDUAL, WITH_BNA_RESIDUAL, LDGS, SMEM_B_SIZE, STAGES_, WITH_RELU, 
    SIMPLE_1x1x1, WITH_BITMASK_RELU_WRITE >
    : public xmma::Smem_tile_a<xmma::Ampere_hmma_fp32_traits, Cta_tile, xmma::Row,
                                   BYTES_PER_STS, BUFFERS_PER_TILE> {

    // Residual Add kernel ALWAYS writes out the result of RES_ADD_RELU to memory
    // We support only SIMPLE 1x1x1 for this write out
    static_assert( WITH_RESIDUAL ? SIMPLE_1x1x1 & WITH_RELU : 1, 
                   "ERROR : Residual add valid only for SIMPLE_1x1x1 +  RELU" );

    static_assert( WITH_BITMASK_RELU_WRITE ? WITH_RESIDUAL : 1, 
                   "ERROR : Bitmask Write is valid only with RES_ADD" );
 
    // The traits class.
    using Traits = xmma::Ampere_hmma_fp32_traits;

    // Row layout
    using Row = xmma::Row;

    // The base class.
    using Base = xmma::Smem_tile_a<Traits, Cta_tile, Row, BYTES_PER_STS, BUFFERS_PER_TILE>;

    // Number of stages in the multistage pipeline
    enum { STAGES = xmma::Max<STAGES_, 2>::VALUE };

    // The fragment.
    using Fragment = typename Base::Fragment;

    // The XMMA tile.
    using Xmma_tile = typename Base::Xmma_tile;

    // The type of elements that are stored in shared memory by each thread.
    using Store_type = typename Base::Store_type;

    // The size of a single LDS in bytes.
    enum { BYTES_PER_LDS = 16 };

    // Number of threads participating in loading scale and bias to Shmem
    enum { NUM_THREADS_LOADING = 4 };

    // Elements loaded per thread into Shmem in each K-block iteration
    enum { ELEMENTS_PER_LOAD_SCALE = 4 };

    // Size of each element (fp16)
    enum { BYTES_PER_ELEMENT_SCALE = 2 };

    // Number of interations in K-block
    enum { ITER_PER_K_BLOCK = Xmma_tile::XMMAS_K };

    // Size of scale buffer for all stages
    // uint2 scale_[STAGES][Xmma_tile::XMMAS_K] for 4 threads
    enum { BYTES_PER_TILE_SCALE = STAGES * ITER_PER_K_BLOCK * ELEMENTS_PER_LOAD_SCALE 
                                    * BYTES_PER_ELEMENT_SCALE * NUM_THREADS_LOADING };

    // Offset for the bias 
    enum { BIAS_OFFSET = BYTES_PER_TILE_SCALE };

    // Size of the scale buffer - per stage
    enum { BYTES_PER_STAGE_SCALE = ITER_PER_K_BLOCK * ELEMENTS_PER_LOAD_SCALE 
                                    * BYTES_PER_ELEMENT_SCALE * NUM_THREADS_LOADING };

    // This represents the size of 1 row of scale in shmem
    enum { BYTES_PER_ROW_SCALE = NUM_THREADS_LOADING * sizeof(uint32_t)};

    // The number of XMMAs.
    using Xmma_tile_with_padding = typename Base::Xmma_tile_with_padding;

    // Ctor.
    inline __device__ Smem_tile_a(char *smem, const int tidx)
        : Base(smem, tidx), k_idx(0) {

        // Since only the first 4 threads will be loading the scale and bias
        // In order to get both the scale and bias using a single LDSM.88.4
        // Threads 0-7   will point to Scale.x
        // Threads 8-15  will point to Scale.y
        // Threads 16-23 will point to Bias.x
        // Threads 24-31 will point to Bias.y
        int thread_id_in_warp = threadIdx.x % 32;
        int octect_id = thread_id_in_warp / 8;

        uint32_t ldsm_offset;
        if( octect_id < 2 ) {
            ldsm_offset = octect_id * BYTES_PER_ROW_SCALE;
        } else {
            ldsm_offset = BIAS_OFFSET + (octect_id - 2) * BYTES_PER_ROW_SCALE;
        }

        const uint32_t smem_base = Base::BYTES_PER_TILE + SMEM_B_SIZE; 
        smem_rd_base_ = smem_base + ldsm_offset;
        smem_wr_base_ = smem_base + ((tidx % NUM_THREADS_LOADING) * sizeof(uint1));

        // Initially Base and Addresses are same
        smem_rd_addr_ = smem_rd_base_;
        smem_wr_addr_ = smem_wr_base_;

        lwrr_stage_ = 0;
        lwrr_loop_ = 0;

        // the shared memory layout is: 
        // [ tile_a | tile_b | scale | bias | residual | res_scale | res_bias ]
        if (WITH_RESIDUAL) {
            smem_residual_offset_ = Base::BYTES_PER_TILE + SMEM_B_SIZE + 
                                    2 * BYTES_PER_TILE_SCALE;

            uint32_t warp_id = threadIdx.x / 32;
            warp_id_offset = warp_id * 16;
            thread_in_warp_row = (threadIdx.x % 32) / 4;
            thread_in_warp_col = (threadIdx.x % 32) % 4;
            warp_col_id = warp_id / Cta_tile::WARPS_M;

            if( WITH_BITMASK_RELU_WRITE ) {
                #pragma unroll
                for (int mi = 0; mi < Xmma_tile::XMMAS_M; ++mi) {
                    bitmask_relu[mi] = 0;
                }
                enable_bitmask_write_out_ = 0;
            }
        }
    }

    // Seperate the LDSM and offset move, since the residual doesn't need it
    inline __device__ void move_next_offset(int ki) {
        static_assert(Xmma_tile_with_padding::XMMAS_K < 64, "Not implemented"); 
        if( Xmma_tile_with_padding::XMMAS_K >= 32 && ki % 16 == 15 ) {
            this->smem_read_offset_ ^= 31 * BYTES_PER_LDS * 2;
        } else if( Xmma_tile_with_padding::XMMAS_K >= 16 && ki %  8 ==  7 ) {
            this->smem_read_offset_ ^= 15 * BYTES_PER_LDS * 2;
        } else if( Xmma_tile_with_padding::XMMAS_K >=  8 && ki %  4 ==  3 ) {
            this->smem_read_offset_ ^=  7 * BYTES_PER_LDS * 2;
        } else if( Xmma_tile_with_padding::XMMAS_K >=  4 && ki %  2 ==  1 ) {
            this->smem_read_offset_ ^=  3 * BYTES_PER_LDS * 2;
        } else if( Xmma_tile_with_padding::XMMAS_K >=  2 ) {
            this->smem_read_offset_ ^=  1 * BYTES_PER_LDS * 2;
        }
    }

    // Load from shared memory using LDSM
    inline __device__ void load_tile(Fragment (&a)[Xmma_tile::XMMAS_M], int ki) {

        #pragma unroll
        for (int mi = 0; mi < Xmma_tile::XMMAS_M; ++mi) {
            // Jump by as many matrix rows as needed (a row in smem may pack multiple matrix rows).
            int offset = mi * Xmma_tile::M_PER_XMMA_PER_CTA * Base::BYTES_PER_ROW_BEFORE_PACKING;

            // Load using LDSM.M88.4.
            uint4 tmp;
            xmma::ldsm(tmp, this->smem_ + this->smem_read_offset_ + this->smem_read_buffer_ +
                                    offset);

            // Store the value into the fragment.
            a[mi].reg(0) = tmp.x;
            a[mi].reg(1) = tmp.y;
            a[mi].reg(2) = tmp.z;
            a[mi].reg(3) = tmp.w;
        }
    }

    // Load from shared memory.
    inline __device__ void load(Fragment (&a)[Xmma_tile::XMMAS_M], int ki) {

        // Load the residual and the activations, move the offsets
        Fragment a_res[Xmma_tile::XMMAS_M];
        if (WITH_RESIDUAL) {
            this->smem_ += smem_residual_offset_;
            load_tile(a_res, ki);
            this->smem_ -= smem_residual_offset_;
        }

        load_tile(a, ki);

        move_next_offset(ki);

        // load the bias and scale
        // Optimization : Load the scale and bias is one go instead of getting them separately
        smem_rd_addr_ = smem_rd_base_ + 2 * ki * BYTES_PER_ROW_SCALE + 
                            lwrr_stage_ * BYTES_PER_ROW_SCALE * Xmma_tile::XMMAS_K * 2; 

        uint4 scale_bias;

        // Load the scale and bias from SMEM
        xmma::ldsm( scale_bias, smem_rd_addr_ );
        scale_.x = scale_bias.x;
        scale_.y = scale_bias.y;
        bias_.x  = scale_bias.z;
        bias_.y  = scale_bias.w;

        // If Residual also needs a BNa - load the scale and bias
        // Same offsets, but need to move by size 2* Scale + RES_ADD tensor tile =  A tensor tile size
        if( WITH_BNA_RESIDUAL ) {
            xmma::ldsm( scale_bias, smem_rd_addr_ +  Base::BYTES_PER_TILE + 2 * BIAS_OFFSET);
            res_scale_.x = scale_bias.x;
            res_scale_.y = scale_bias.y;
            res_bias_.x  = scale_bias.z;
            res_bias_.y  = scale_bias.w;
        }

        // BN Apply : val = scale * img + bias + res
        #pragma unroll
        for (int mi = 0; mi < Xmma_tile::XMMAS_M; ++mi) {
            uint32_t bias_final[4] = { bias_.x, bias_.x, bias_.y, bias_.y };

            // Residual Addition
            if( WITH_RESIDUAL )
            {
                // a_res[mi] needs a BNa
                if( WITH_BNA_RESIDUAL ) {
                    a_res[mi].reg(0) = xmma::guarded_scale_bias_relu_a<false>( a_res[mi].reg(0), 
                                                                       res_scale_.x, res_bias_.x );
                    a_res[mi].reg(1) = xmma::guarded_scale_bias_relu_a<false>( a_res[mi].reg(1), 
                                                                       res_scale_.x, res_bias_.x );
                    a_res[mi].reg(2) = xmma::guarded_scale_bias_relu_a<false>( a_res[mi].reg(2), 
                                                                       res_scale_.y, res_bias_.y );
                    a_res[mi].reg(3) = xmma::guarded_scale_bias_relu_a<false>( a_res[mi].reg(3), 
                                                                       res_scale_.y, res_bias_.y );
                }

                bias_final[0] = xmma::hadd2( bias_final[0], a_res[mi].reg(0) );
                bias_final[1] = xmma::hadd2( bias_final[1], a_res[mi].reg(1) );
                bias_final[2] = xmma::hadd2( bias_final[2], a_res[mi].reg(2) );
                bias_final[3] = xmma::hadd2( bias_final[3], a_res[mi].reg(3) );
            }

            // When there is residual add being fused, and simple 1x1x1
            // We may need to write out the bitmask relu vector to perform d(relu) during bprop
            // Hence we disable RELU based on this condition and do an exta step to perform RELU
            constexpr bool RELU_ENABLED =  WITH_RESIDUAL ? (!WITH_BITMASK_RELU_WRITE) : WITH_RELU;

            a[mi].reg(0) = xmma::guarded_scale_bias_relu_a<RELU_ENABLED>( a[mi].reg(0), 
                                                                   scale_.x, bias_final[0]);
            a[mi].reg(1) = xmma::guarded_scale_bias_relu_a<RELU_ENABLED>( a[mi].reg(1), 
                                                                   scale_.x, bias_final[1]);
            a[mi].reg(2) = xmma::guarded_scale_bias_relu_a<RELU_ENABLED>( a[mi].reg(2), 
                                                                   scale_.y, bias_final[2]);
            a[mi].reg(3) = xmma::guarded_scale_bias_relu_a<RELU_ENABLED>( a[mi].reg(3), 
                                                                   scale_.y, bias_final[3]);

            if( WITH_BITMASK_RELU_WRITE ) {
                // We support only 2 CTA::K on Ampere
                uint32_t r_shift;
                if ( Cta_tile::K == 64 ) {
                    r_shift  = 0;
                } else if ( Cta_tile::K == 32 ) {
                    r_shift = (k_idx & 1) * 8;
                }

                calc_bitmask_relu(a[mi].reg(0), bitmask_relu[mi], r_shift + ki * 4 + 0);
                calc_bitmask_relu(a[mi].reg(1), bitmask_relu[mi], r_shift + ki * 4 + 1);
                calc_bitmask_relu(a[mi].reg(2), bitmask_relu[mi], r_shift + ki * 4 + 2);
                calc_bitmask_relu(a[mi].reg(3), bitmask_relu[mi], r_shift + ki * 4 + 3);

                a[mi].reg(0) = xmma::relu_fp16x2(a[mi].reg(0));
                a[mi].reg(1) = xmma::relu_fp16x2(a[mi].reg(1));
                a[mi].reg(2) = xmma::relu_fp16x2(a[mi].reg(2));
                a[mi].reg(3) = xmma::relu_fp16x2(a[mi].reg(3));
            }

            // Writing out the results
            if( WITH_RESIDUAL ) {

                // Its enough only if the First Column of CTAs write the result out
                if( bid_n==0 && warp_col_id==0 ) {
                    constexpr uint32_t ROWS_PER_WARP = 16;
                    constexpr uint32_t LDSM_WIDTH = 8;
                    constexpr uint32_t LDSM_HEIGHT= 8;

                    uint32_t out_row_base = bid_m * Cta_tile::M + warp_id_offset + 
                                          mi * Cta_tile::WARPS_M * ROWS_PER_WARP + 
                                          thread_in_warp_row;
                    uint32_t out_col_base = k_idx * Cta_tile::K + ki * 16 + 2 * thread_in_warp_col;

                    // One XMMA Tile has a 2x2 tile of 8x8s
                    #pragma unroll
                    for(int n =0; n < 2; n++) {
                        #pragma unroll
                        for(int m =0; m < 2; m++) {
                            //// Note : we need to write out the result of residual add + RELU
                            uint32_t out_row = out_row_base + m * LDSM_HEIGHT;
                            uint32_t out_col = out_col_base + n * LDSM_WIDTH;
                            uint32_t out_idx = out_row * max_col + out_col;

                            if((out_col < max_col) && (out_row < max_row)) {
                                res_add_relu_out_ptr_[out_idx/2] = a[mi].reg(m + 2 * n);
                            }
                        }
                    }

                    // Write out bitmask relu at the end of 64 values
                    // This is because we have 32 bits / thread and can handle 64 channels / integer
                    // Refer this diagram : 
                    // https://docs.google.com/spreadsheets/d/1-LFGs42jk0h3jwxYRWhf5XtL_OuWywzxs7-7aq6_eng/edit?usp=sharing 
                    if( WITH_BITMASK_RELU_WRITE ) {
                        if( enable_bitmask_write_out_ ) {

                            // 16 rows in original tensor become 8 rows of bitmask relu
                            // 64 cols in original tensor become 4 cols of bitmask relu (4 integers)
                            constexpr uint32_t ROWS_PRE_MERGE = 16;
                            constexpr uint32_t ROWS_POST_MERGE = 8;
                            constexpr uint32_t COLS_PRE_MERGE = 64;
                            constexpr uint32_t COLS_POST_MERGE = 4;

                            // For fp16 data, 8 columns of data is loaded per LDSM 
                            // And 2 columns of data is loaded per thread in a LDSM
                            constexpr uint32_t COLS_PER_LDSM = 8;
                            constexpr uint32_t COLS_PER_THREAD_LDSM = 2;

                            uint32_t k_idx_tmp;
                            if ( Cta_tile::K == 64 ) {
                                k_idx_tmp = k_idx;
                            // The 0xFFFFFFFE ensures that when C is multiple of 64, 
                            // we use even value of k_idx i.e previous iteration => k_idx - 1
                            // And when C is not multiple of 64, we use it as is in the last iteration 
                            } else if ( Cta_tile::K == 32 ) {
                                k_idx_tmp = k_idx & 0xFFFFFFFE;
                            }

                            uint32_t out_col_base = k_idx_tmp * Cta_tile::K + 2 * thread_in_warp_col;
                            uint32_t out_row = int(out_row_base / ROWS_PRE_MERGE) * ROWS_POST_MERGE + 
                                               out_row_base % ROWS_POST_MERGE;
                            uint32_t out_col = int(out_col_base / COLS_PRE_MERGE) * COLS_POST_MERGE + 
                                               (out_col_base % COLS_PER_LDSM) / COLS_PER_THREAD_LDSM;

                            if((out_col < max_bitmask_col) && (out_row < max_bitmask_row)) {
                                uint32_t out_idx = out_row * max_bitmask_col + out_col;
                                bitmask_relu_out_ptr_[out_idx] = bitmask_relu[mi];
                            }

                            // Reset the bitmask
                            bitmask_relu[mi] = 0;
                        }
                    }
                }
            }
        }
    
        // move along the cirlwlar buffer
        ++lwrr_loop_;

        if (lwrr_loop_ % loop_count_ == 0) {
            ++lwrr_stage_;
            lwrr_stage_ %= STAGES;
            lwrr_loop_ = 0;
            smem_rd_addr_ = lwrr_stage_ == 0 ? smem_rd_base_ : smem_rd_addr_;
            if( WITH_RESIDUAL && SIMPLE_1x1x1 ) {
                k_idx++;
            }
        }

        // Handles the case where C is not a multiple of 64 for bitmask relu write out
        // Can be removed for RN-50, since C is always a multiple of 64
        if( WITH_BITMASK_RELU_WRITE ) {
            constexpr uint32_t KI_READY = Xmma_tile::XMMAS_K - 2;
            const uint32_t last_k = (max_col / Cta_tile::K) - 1;
            uint32_t k_idx_ready = 1;
    
            // We support only 2 CTA::K on Ampere
            if ( Cta_tile::K == 64 ) {
                k_idx_ready  = 1;
            } else if ( Cta_tile::K == 32 ) {
                k_idx_ready = (k_idx % 2 == 1) || (k_idx == last_k);
            }
                
            if( WITH_BITMASK_RELU_WRITE ) {
                enable_bitmask_write_out_ = k_idx_ready && (ki == KI_READY);
            }
        }
    }

    // Store to the tile in shared memory.
    template <int N, int M, int K = 1, typename LDGSTS_CFG = xmma::Ldgsts_config<true>>
    inline __device__ void store(const void *(&gmem_ptrs)[N], uint32_t (&preds)[M],
                                 uint64_t mem_desc = xmma::MEM_DESC_DEFAULT) {

        // We duplicate the Base class function here - because we specifically avoid ZFILL
        // Make sure we ilwert the predicates for STS
        uint32_t smem_ptrs[N];
        this->compute_store_pointers(smem_ptrs);
        if (Base::USE_PREDICATES) {
            // true here sets NAN
            xmma::ldgsts<N, M, 16, false, true>(smem_ptrs, gmem_ptrs, preds, mem_desc);
        } else {
            #pragma unroll
            for (int ii = 0; ii < N; ii++) {
                xmma::ldgsts128_nopreds(smem_ptrs[ii], gmem_ptrs[ii]);
            }
        }

        // Do the same with the residual, but in the default manner (with ZFILL)
        if (WITH_RESIDUAL) {
            this->smem_ += smem_residual_offset_;
            Base::store(gmem_residual_ptrs_, preds, mem_desc);
            this->smem_ -= smem_residual_offset_;
        }
    }

    // Store the scale bias in shared memory.
    inline __device__ void scale_bias_load(const void *gmem_ptrs_scale, 
                                           const void *gmem_ptrs_bias, 
                                           const uint32_t &pred) {

        uint32_t smem_ptr_scale = smem_wr_addr_;
        uint32_t smem_ptr_bias = smem_wr_addr_ + BIAS_OFFSET;

        xmma::ldgsts<4>( smem_ptr_scale, gmem_ptrs_scale, pred );
        xmma::ldgsts<4>( smem_ptr_bias, gmem_ptrs_bias, pred );


        // When the residual tensor also needs a BNa
        // we don't update the smem_write potinters as we reuse them in res_scale, res_bias load
        // so res_scale_bias_load() will update the smem_write pointer
        if( ! WITH_BNA_RESIDUAL ) {
            // Size of one Row of SCALE or BIAS TILE
            const int BYTES_PER_ROW_SCALE = NUM_THREADS_LOADING * sizeof(uint32_t);

            // Move the smem write offset by 1 row
            smem_wr_addr_ += BYTES_PER_ROW_SCALE;
        }
    }

    // Store the residual scale bias in shared memory.
    inline __device__ void res_scale_bias_load(const void *gmem_ptrs_scale, 
                                               const void *gmem_ptrs_bias, 
                                               const uint32_t &pred) {

        // Add extra A-Tensor offset to get the res_scale and res_bias
        // [ tile_a | tile_b | scale | bias | residual | res_scale | res_bias ]
        uint32_t smem_ptr_scale = smem_wr_addr_ + 2 * BIAS_OFFSET + Base::BYTES_PER_TILE;
        uint32_t smem_ptr_bias  = smem_ptr_scale + BIAS_OFFSET;

        xmma::ldgsts<4>( smem_ptr_scale, gmem_ptrs_scale, pred );
        xmma::ldgsts<4>( smem_ptr_bias, gmem_ptrs_bias, pred );

        // Size of one Row of SCALE or BIAS TILE
        const int BYTES_PER_ROW_SCALE = NUM_THREADS_LOADING * sizeof(uint32_t);

        // Move the smem write offset by 1 row
        smem_wr_addr_ += BYTES_PER_ROW_SCALE;
    }

    // Reset smem write pointer to start of cirlwlar buffer
    inline __device__ void reset_smem_wr_offset() { 
        smem_wr_addr_ = smem_wr_base_; 
    }

    // Store the residue pointers
    template <int N> inline __device__ void set_residue_pointers(const char *(&gmem_ptrs)[N]) {
        #pragma unroll
        for (int i = 0; i < N; ++i) {
            gmem_residual_ptrs_[i] = reinterpret_cast<const void *>(gmem_ptrs[i]);
        }
    }

    // Set the loop count per stage
    inline __device__ void setup(const int &loop_cnt) {
        loop_count_ = loop_cnt; 
    }

    // Set the loop count per stage
    inline __device__ void setup(const int &loop_cnt, const dim3 &bid, const uint32_t& x_max, const uint32_t& y_max, uint32_t* add_out_ptr, uint32_t* bitmask_out_ptr) { 
        loop_count_ = loop_cnt; 
        if( WITH_RESIDUAL && SIMPLE_1x1x1 ) {
            bid_m = bid.x; 
            bid_n = bid.y; 
            max_col = x_max;
            max_row = y_max;

            res_add_relu_out_ptr_ = add_out_ptr;
            
            // Refer samples bn_apply_fprop_bn_stats line 400 for details
            // The bitmask relu combines 2 Rows and 16 Columns of Values to form a 32bit value
            // The total number of elements is a bit complicated, since we pack 16 rows into 8
            // So the dimensions are ~ceil(NHW / 2) x ceil(CRS / 16) bits - but not exactly
            // For example if nhw = 14x14 scenario, last 4 rows don't collapse
            if( WITH_BITMASK_RELU_WRITE ) {
                // Code below is the exact math
                max_bitmask_col = xmma::div_up(max_col, 64) * 4;
                max_bitmask_row = int(max_row / 16) * 16;
                uint32_t reminder_rows = max_row - max_bitmask_row;
                max_bitmask_row = (max_bitmask_row / 2) + min(8, reminder_rows);
                bitmask_relu_out_ptr_ = bitmask_out_ptr;
            }
        }
    }

    // Input is a register which has data in half2 format
    // It can be reinterpreted as uint32_t
    // The sign bit for fp16 is the 16th bit
    // so we can use that to find the bit mask (i.e is it negative)
    // Only positive values need to set this bit (so we flip it)
    inline __device__  void calc_bitmask_relu(const uint32_t &x, uint32_t &mask, 
                                              const uint32_t &shift) {

        uint32_t sign_bits = x & 0x80008000;

        // This makes sure that "zero" does not trigger the bitmask
        // This is useful since OOB pixels will also have zero value, 
        // so we don't want to trigger for them
        // And zero anyway does not enable RELU gradient (TODO : Confirm with @kgoyal)
        if ( (x & 0xFFFF0000) == 0)
            sign_bits |= 0x80000000;

        if ( (x & 0x0000FFFF) == 0)
            sign_bits |= 0x00008000;

        uint32_t bitmask_relu = sign_bits ^ 0x80008000;
        mask |= bitmask_relu >> shift;
    } 

    // Scale and bias
    uint2 scale_, bias_;

    // Scale and bias for residual
    uint2 res_scale_, res_bias_;

    // Residual pointers
    const void *gmem_residual_ptrs_[LDGS];

    // Offset for the residual in shmem
    uint32_t smem_residual_offset_;

    // Current stage bring processed in the multistage pipeline
    int lwrr_stage_;

    // Current loop count being processed
    int lwrr_loop_;

    // Total number of calls per stage
    int loop_count_;

    // Smem base address for reading and writing scale
    uint32_t smem_rd_base_;
    uint32_t smem_wr_base_;

    // Smem address for reading and writing scale
    uint32_t smem_rd_addr_;
    uint32_t smem_wr_addr_;

    // Current Block Id (accounting for swizzle)
    uint32_t bid_m;
    uint32_t bid_n;

    // K loop index - increments for every stage loaded
    uint32_t k_idx;

    // Max, #cols, #rows of the res_add_relu output tensor
    uint32_t max_col, max_row;

    // Max, #cols, #rows of the bitmask_relu output tensor
    uint32_t max_bitmask_col, max_bitmask_row;

    // Decides if we are ready to write out bitmask relu
    uint32_t enable_bitmask_write_out_;

    // Re_Add Output pointer
    uint32_t* res_add_relu_out_ptr_;

    // Bitmask RELU pointer
    uint32_t* bitmask_relu_out_ptr_;

    // Warp, thread locations
    uint32_t warp_id_offset;
    uint32_t thread_in_warp_row;
    uint32_t thread_in_warp_col;
    uint32_t warp_col_id;

    // Bitmask RELU output
    uint32_t bitmask_relu[Xmma_tile::XMMAS_M];
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// Used when we have fused DBNA + DGRAD kernels used in the backward pass
template < typename Cta_tile, int BYTES_PER_STS, int BUFFERS_PER_TILE, int GMEM_LDGS, bool WITH_FUSED_DBNA_DGRAD, uint32_t SMEM_B_SIZE >
struct Smem_tile_a_wgrad< xmma::Ampere_hmma_fp32_traits, Cta_tile, xmma::Col, BYTES_PER_STS,
                          BUFFERS_PER_TILE, GMEM_LDGS, WITH_FUSED_DBNA_DGRAD, SMEM_B_SIZE >
    : public xmma::Smem_tile_a< xmma::Ampere_hmma_fp32_traits, Cta_tile, xmma::Col, 
                                BYTES_PER_STS, BUFFERS_PER_TILE > {

    // The traits class.
    using Traits = xmma::Ampere_hmma_fp32_traits;

    // Row layout
    using Col = xmma::Col;

    // The base class.
    using Base = xmma::Smem_tile_a<Traits, Cta_tile, Col, BYTES_PER_STS, BUFFERS_PER_TILE>;

    // The fragment.
    using Fragment = typename Base::Fragment;

    // The XMMA tile.
    using Xmma_tile = typename Base::Xmma_tile;

    // The type of elements that are stored in shared memory by each thread.
    using Store_type = typename Base::Store_type;

    // Ctor.
    inline __device__ Smem_tile_a_wgrad(char *smem, const int tidx)
        : Base(smem, tidx) { 

        if( WITH_FUSED_DBNA_DGRAD ) {
            // Since the scale and bias will be loaded in gmem_tile constructor
            // We just need to load the smem_tile for the residual aka fprop tensor
            // The shared memory layout is: [ tile_a | tile_b | Fprop_tensor ]
            smem_fprop_tensor_offset_ = Base::BYTES_PER_TILE + SMEM_B_SIZE;
        }
    }

    // Load from shared memory using LDSM
    // Unfortunately we have to duplicate this class from Base
    // This is because the Base::load updates the Smem ptr, but we don't want to.
    // We reuse the same base pointer for Extra Fprop Tensor Load and Dy tensor
    inline __device__ void load_tile(Fragment (&a)[Xmma_tile::XMMAS_M], int ki) {
        // The size of each element in bits.
        const int BITS_PER_ELT = Traits::BITS_PER_ELEMENT_A;
        // The size in bytes of the data needed to compute an XMMA per CTA.
        const int BYTES_PER_XMMA_PER_CTA = Xmma_tile::M_PER_XMMA_PER_CTA * BITS_PER_ELT / 8;

        // Perform the different loads.
        int smem_read_offset_local_ = this->smem_read_offset_;
        #pragma unroll
        for( int mi = 0; mi < Xmma_tile::XMMAS_M; ++mi ) {
            // Prepare the offset.
            int offset = ki * Base::ROWS_PER_XOR_PATTERN * 2 * Base::BYTES_PER_ROW;
            if( BYTES_PER_XMMA_PER_CTA == 32 ) {
                offset += smem_read_offset_local_;
            } else if( BYTES_PER_XMMA_PER_CTA == 64 ) {
                offset += smem_read_offset_local_ + (mi/2) * BYTES_PER_XMMA_PER_CTA * 2;
            } else if( BYTES_PER_XMMA_PER_CTA >= 128 ) {
                offset += smem_read_offset_local_ + (mi) * BYTES_PER_XMMA_PER_CTA;
            } else {
                assert(false);
            }

            // Load the data using LDSM.MT88.4 or 4x LDS.32.
            uint32_t ptr = this->smem_ + this->smem_read_buffer_ + offset;
            uint4 tmp;
            ldsmt(tmp, ptr);

            // Store those values in the fragment.
            a[mi].reg(0) = tmp.x;
            a[mi].reg(1) = tmp.y;
            a[mi].reg(2) = tmp.z;
            a[mi].reg(3) = tmp.w;

            // Move the pointer for the next ni. I expect the compiler to not recompute those.
            if( BYTES_PER_XMMA_PER_CTA >= 128 ) {
                // Nothing to do!
            } else if( BYTES_PER_XMMA_PER_CTA == 64 && Xmma_tile::XMMAS_M > 1 ) {
                smem_read_offset_local_ ^= BYTES_PER_XMMA_PER_CTA;
            } else if( BYTES_PER_XMMA_PER_CTA == 64 ) {
                // Nothing to do!
            } else if( BYTES_PER_XMMA_PER_CTA == 32 && Xmma_tile::XMMAS_M == 4 ) {
                smem_read_offset_local_ ^= Base::BYTES_PER_LDS * (mi % 2 == 0 ? 2 : 6);
            } else if( BYTES_PER_XMMA_PER_CTA == 32 && Xmma_tile::XMMAS_M == 2 ) {
                smem_read_offset_local_ ^= Base::BYTES_PER_LDS * 2;
            }
        }
    }

    // Load from shared memory - also apply the scale, bias and RELU
    inline __device__ void load(Fragment (&a)[Xmma_tile::XMMAS_M], int ki) {

        // With fused dgrad, - wgrad needs to load extra tensor
        // When not in use, hopefully the compiler should not alloc any reg. for it
        Fragment fprop_tensor[Xmma_tile::XMMAS_M];
        if( WITH_FUSED_DBNA_DGRAD ) {
            // Load the Extra fprop tensor
            this->smem_ += smem_fprop_tensor_offset_;
            load_tile(fprop_tensor, ki);
            this->smem_ -= smem_fprop_tensor_offset_;
        }

        // Load the dY tensor, this also moves the smem pointers.
        Base::load(a, ki);

        // With fused dgrad, - wgrad needs to load extra tensor, and apply scale, bias
        if( WITH_FUSED_DBNA_DGRAD ) {

            // Apply the scale and bias
            #pragma unroll
            for (int mi = 0; mi < Xmma_tile::XMMAS_M; ++mi) {

                __half2 tmp_fprop_scale = reinterpret_cast<__half2&>(fprop_scale_[mi]);
                __half2 tmp_fprop_bias  = reinterpret_cast<__half2&>(fprop_bias_[mi]);
                __half2 tmp_dgrad_scale = reinterpret_cast<__half2&>(dgrad_scale_[mi]);

                a[mi].reg(0) = guarded_dbna_scale_bias( a[mi].reg(0),
                                   fprop_tensor[mi].reg(0), 
                                   reinterpret_cast<uint16_t&>(tmp_fprop_scale.x), 
                                   reinterpret_cast<uint16_t&>(tmp_fprop_bias.x),
                                   reinterpret_cast<uint16_t&>(tmp_dgrad_scale.x));

                a[mi].reg(1) = guarded_dbna_scale_bias( a[mi].reg(1), 
                                   fprop_tensor[mi].reg(1), 
                                   reinterpret_cast<uint16_t&>(tmp_fprop_scale.y), 
                                   reinterpret_cast<uint16_t&>(tmp_fprop_bias.y),
                                   reinterpret_cast<uint16_t&>(tmp_dgrad_scale.y));

                a[mi].reg(2) = guarded_dbna_scale_bias( a[mi].reg(2), 
                                   fprop_tensor[mi].reg(2), 
                                   reinterpret_cast<uint16_t&>(tmp_fprop_scale.x), 
                                   reinterpret_cast<uint16_t&>(tmp_fprop_bias.x),
                                   reinterpret_cast<uint16_t&>(tmp_dgrad_scale.x));

                a[mi].reg(3) = guarded_dbna_scale_bias( a[mi].reg(3), 
                                   fprop_tensor[mi].reg(3), 
                                   reinterpret_cast<uint16_t&>(tmp_fprop_scale.y), 
                                   reinterpret_cast<uint16_t&>(tmp_fprop_bias.y),
                                   reinterpret_cast<uint16_t&>(tmp_dgrad_scale.y));
            }
        }
    }

    inline __device__ 
    uint32_t guarded_dbna_scale_bias(const uint32_t &dy, const uint32_t &fprop_tensor, 
            const uint16_t &fprop_scale, const uint16_t &fprop_bias, const uint16_t dy_scale) {


        // dBNA + Wrad (with dgrad fusion) =>
        // A Matrix =  Fprop_tensor * Fprop_Scale + Fprop_Bias + Dgrad_scale * dY
        // Dy tensor doesn't need NaN Fill since it is only multiplied by scale
        // So we need guarded scale only for Fprop tensor
        uint32_t scaled_fprop = guarded_scale_bias_relu_b<false>( fprop_tensor, fprop_scale, fprop_bias);
        // Using FMA with bias of 0 may be faster than FMUL (?)
        uint32_t scaled_dy = xmma::scale_bias(dy, dy_scale, 0);
        
        return  hadd2(scaled_fprop, scaled_dy);
    }

    // Store to the tile in shared memory.
    template <int N, int M, int K = 0, typename LDGSTS_CFG = xmma::Ldgsts_config<true>>
    inline __device__ void store(const void *(&gmem_ptrs)[N], uint32_t (&preds)[M],
                                 uint64_t mem_desc = xmma::MEM_DESC_DEFAULT) {

        // We Implement the function here - because we specifically avoid ZFILL on Fprop Tensor
        Base::store(gmem_ptrs, preds, mem_desc);

        if( WITH_FUSED_DBNA_DGRAD ) {
            uint32_t smem_ptrs[N];
            this->smem_ += smem_fprop_tensor_offset_;
            this->compute_store_pointers(smem_ptrs);
            xmma::ldgsts<N, M, 16, false, true>(smem_ptrs, fprop_tensor_ptr_, preds, mem_desc);
            this->smem_ -= smem_fprop_tensor_offset_;
        }
    }

    // Store the residue pointers
    template <int N> 
    inline __device__ void set_residue_pointers(const char *(&gmem_ptrs)[N]) {
        #pragma unroll
        for (int i = 0; i < N; ++i) {
            fprop_tensor_ptr_[i] = reinterpret_cast<const void *>(gmem_ptrs[i]);
        }
    }

    // Store to the tile in shared memory.
    inline __device__ void store_scale_bias(const uint32_t *fprop_scale_data, 
                                            const uint32_t *fprop_bias_data,
                                            const uint32_t *dgrad_scale_data,
                                            const int &ni) {
        fprop_scale_[ni] = fprop_scale_data[0];
        fprop_bias_[ni]  = fprop_bias_data[0];
        dgrad_scale_[ni] = dgrad_scale_data[0];
    }

    // storage for scale and bias
    uint32_t fprop_scale_[Xmma_tile::XMMAS_M];
    uint32_t dgrad_scale_[Xmma_tile::XMMAS_M];
    uint32_t fprop_bias_[Xmma_tile::XMMAS_M];
    // Residual pointers
    const void *fprop_tensor_ptr_[GMEM_LDGS];
    // Offset for the residual Fprop tensor in shmem
    uint32_t smem_fprop_tensor_offset_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Cta_tile, int BYTES_PER_STS, int BUFFERS_PER_TILE, bool WITH_RELU>
struct Smem_tile_b<xmma::Ampere_hmma_fp32_traits, Cta_tile, xmma::Row, BYTES_PER_STS,
                   BUFFERS_PER_TILE, WITH_RELU>
    : public xmma::Smem_tile_b<xmma::Ampere_hmma_fp32_traits, Cta_tile, xmma::Row,
                                   BYTES_PER_STS, BUFFERS_PER_TILE> {

    // The traits class.
    using Traits = xmma::Ampere_hmma_fp32_traits;

    // Row layout
    using Row = xmma::Row;

    // The base class.
    using Base = xmma::Smem_tile_b<Traits, Cta_tile, Row, BYTES_PER_STS, BUFFERS_PER_TILE>;

    // The fragment.
    using Fragment = typename Base::Fragment;

    // The XMMA tile.
    using Xmma_tile = typename Base::Xmma_tile;

    // The type of elements that are stored in shared memory by each thread.
    using Store_type = typename Base::Store_type;

    // Ctor.
    inline __device__ Smem_tile_b(char *smem, const int tidx)
        : Base(smem, tidx) { }

    // Load from shared memory - also apply the scale, bias and RELU
    inline __device__ void load(Fragment (&a)[Xmma_tile::XMMAS_N], int ki) {

        Base::load(a, ki);

        // Apply the scale and bias
        #pragma unroll
        for (int ni = 0; ni < Xmma_tile::XMMAS_N; ++ni) {
            __half2 tmp_scale = reinterpret_cast<__half2&>(scale_[ni]);
            __half2 tmp_bias = reinterpret_cast<__half2&>(bias_[ni]);

            a[ni].reg(0) = xmma::guarded_scale_bias_relu_b<WITH_RELU>(a[ni].reg(0), 
                               reinterpret_cast<uint16_t&>(tmp_scale.x), 
                               reinterpret_cast<uint16_t&>(tmp_bias.x));

            a[ni].reg(1) = xmma::guarded_scale_bias_relu_b<WITH_RELU>(a[ni].reg(1), 
                               reinterpret_cast<uint16_t&>(tmp_scale.x), 
                               reinterpret_cast<uint16_t&>(tmp_bias.x));

            a[ni].reg(2) = xmma::guarded_scale_bias_relu_b<WITH_RELU>(a[ni].reg(2), 
                               reinterpret_cast<uint16_t&>(tmp_scale.y), 
                               reinterpret_cast<uint16_t&>(tmp_bias.y));

            a[ni].reg(3) = xmma::guarded_scale_bias_relu_b<WITH_RELU>(a[ni].reg(3), 
                               reinterpret_cast<uint16_t&>(tmp_scale.y), 
                               reinterpret_cast<uint16_t&>(tmp_bias.y));
        }
    }

    // Store to the tile in shared memory.
    template <int N, int M, int K = 1, typename LDGSTS_CFG = xmma::Ldgsts_config<true>>
    inline __device__ void store(const void *(&gmem_ptrs)[N], uint32_t (&preds)[M],
                                 uint64_t mem_desc = xmma::MEM_DESC_DEFAULT) {

        // We duplicate the function here - because we specifically avoid ZFILL
        uint32_t smem_ptrs[N];
        this->compute_store_pointers(smem_ptrs);
        if (Base::USE_PREDICATES) {
            xmma::ldgsts<N, M, 16, false, true>(smem_ptrs, gmem_ptrs, preds, mem_desc);
        } else {
            #pragma unroll
            for (int ii = 0; ii < N; ii++) {
                xmma::ldgsts128_nopreds(smem_ptrs[ii], gmem_ptrs[ii]);
            }
        }
    }

    // Store to the tile in shared memory.
    inline __device__ void store_scale_bias(const uint32_t *scale_data, 
                                            const uint32_t *bias_data,
                                            const uint32_t &ni) {
        scale_[ni] = scale_data[0];
        bias_[ni] = bias_data[0];
    }

    // storage for scale and bias
    uint32_t scale_[Xmma_tile::XMMAS_N];
    uint32_t bias_[Xmma_tile::XMMAS_N];
};

////////////////////////////////////////////////////////////////////////////////////////////////////
} // namespace bn_apply
} // namespace batchnorm
} // namespace ext
} // namespace xmma
////////////////////////////////////////////////////////////////////////////////////////////////////
