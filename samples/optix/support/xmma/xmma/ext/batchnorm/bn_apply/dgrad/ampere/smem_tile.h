/***************************************************************************************************
 *
 * Redistribution and use in source and binary forms, with or without modification, are not permit-
 * ted.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
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
namespace dgrad {

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Cta_tile, int BYTES_PER_STS, int BUFFERS_PER_TILE, int LDGS,
          uint32_t SMEM_B_SIZE, int STAGES_>
struct Smem_tile_a_dbna_dgrad<xmma::Ampere_hmma_fp32_traits, Cta_tile, xmma::Row, BYTES_PER_STS,
                   BUFFERS_PER_TILE, LDGS, SMEM_B_SIZE, STAGES_>
    : public xmma::Smem_tile_a<xmma::Ampere_hmma_fp32_traits, Cta_tile, xmma::Row,
                                   BYTES_PER_STS, BUFFERS_PER_TILE> {

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
    inline __device__ Smem_tile_a_dbna_dgrad(char *smem, const int tidx)
        : Base(smem, tidx), k_idx(0) {

        // Since only the first 4 threads will be loading the scale and bias
        // We have to load 6 values : 2 bna_fprop_scale, 2 bias, 2 grad_scale
        // We load bna_fprop_scale and bias together using one LDSM.88.4
        // Threads 0-7   will point to Scale.x --> bna_fprop_tensor_scale
        // Threads 8-15  will point to Scale.y
        // Threads 16-23 will point to Bias.x  --> dbns_bias
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

        // the shared memory layout is: [ tile_a | tile_b | scale_fprop | bias | scale_grad | bn_fprop_tensor ]
        smem_residual_offset_ = Base::BYTES_PER_TILE + SMEM_B_SIZE + 3 * BYTES_PER_TILE_SCALE;

        curr_stage_ = 0;
        curr_loop_ = 0;

        uint32_t warp_id = threadIdx.x / 32;
        warp_id_offset = warp_id * 16;
        thread_in_warp_row = (threadIdx.x % 32) / 4;
        thread_in_warp_col = (threadIdx.x % 32) % 4;
        warp_col_id = warp_id / Cta_tile::WARPS_M;

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
        Fragment bn_fprop[Xmma_tile::XMMAS_M];
        this->smem_ += smem_residual_offset_;
        load_tile(bn_fprop, ki);
        this->smem_ -= smem_residual_offset_;

        load_tile(a, ki);

        move_next_offset(ki);

        // load the bias and scale
        // Optimization : Load the scale and bias is one go instead of getting them separately
        smem_rd_addr_ = smem_rd_base_ + 2 * ki * BYTES_PER_ROW_SCALE +
                            curr_stage_ * BYTES_PER_ROW_SCALE * Xmma_tile::XMMAS_K * 2;

        uint4 scale_bias;
        xmma::ldsm(scale_bias, smem_rd_addr_);

        uint2 scale_grad;
        xmma::ldsm(scale_grad, smem_rd_addr_ + 2 * BIAS_OFFSET);

        scale_.x = scale_bias.x;
        scale_.y = scale_bias.y;
        bias_.x  = scale_bias.z;
        bias_.y  = scale_bias.w;
        scale_grad_.x = scale_grad.x;
        scale_grad_.y = scale_grad.y;

        // dBN Apply : val = scale_grad * grad + scale_bna_fprop * bna_fprop_tensor + bias

        #pragma unroll
        for (int mi = 0; mi < Xmma_tile::XMMAS_M; ++mi) {
            uint32_t bias_res[4] = { bias_.x, bias_.x, bias_.y, bias_.y };

            // Residual Addition

            bias_res[0] = xmma::hadd2( bias_res[0], hmul2(bn_fprop[mi].reg(0), scale_.x));
            bias_res[1] = xmma::hadd2( bias_res[1], hmul2(bn_fprop[mi].reg(1), scale_.x));
            bias_res[2] = xmma::hadd2( bias_res[2], hmul2(bn_fprop[mi].reg(2), scale_.y));
            bias_res[3] = xmma::hadd2( bias_res[3], hmul2(bn_fprop[mi].reg(3), scale_.y));

            a[mi].reg(0) = xmma::guarded_scale_bias_relu_a<false>( a[mi].reg(0),
                                                                   scale_grad.x, bias_res[0]);
            a[mi].reg(1) = xmma::guarded_scale_bias_relu_a<false>( a[mi].reg(1),
                                                                   scale_grad.x, bias_res[1]);
            a[mi].reg(2) = xmma::guarded_scale_bias_relu_a<false>( a[mi].reg(2),
                                                                   scale_grad.y, bias_res[2]);
            a[mi].reg(3) = xmma::guarded_scale_bias_relu_a<false>( a[mi].reg(3),
                                                                   scale_grad.y, bias_res[3]);
        }

        // move along the circular buffer
        ++curr_loop_;

        if (curr_loop_ % loop_count_ == 0) {
            ++curr_stage_;
            curr_stage_ %= STAGES;
            curr_loop_ = 0;
            smem_rd_addr_ = curr_stage_ == 0 ? smem_rd_base_ : smem_rd_addr_;
            // TODO (kgoyal) Review this
            //if( WITH_RESIDUAL && SIMPLE_1x1x1 ) {
                k_idx++;
            //}
        }
    }

    // Store to the tile in shared memory.
    template <int N, int M, int K = 1, typename LDGSTS_CFG = xmma::Ldgsts_config<true>>
    inline __device__ void store(const void *(&gmem_ptrs)[N], uint32_t (&preds)[M],
                                 uint64_t mem_desc = xmma::MEM_DESC_DEFAULT) {

        // We duplicate the Base class function here - because we specifically avoid ZFILL
        // Make sure we invert the predicates for STS
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

        // Do the same with the bna fporp tensor, but in the default manner (with ZFILL)
        this->smem_ += smem_residual_offset_;
        Base::store(gmem_residual_ptrs_, preds, mem_desc);
        this->smem_ -= smem_residual_offset_;

    }

    // Store to the tile in shared memory.
    inline __device__ void scale_bias_load(const void *gmem_ptrs_scale_grad,
                                           const void *gmem_ptrs_scale_fprop,
                                           const void *gmem_ptrs_bias, const uint32_t &pred) {

        uint32_t smem_ptr_scale_fprop = smem_wr_addr_;
        uint32_t smem_ptr_bias = smem_wr_addr_ + BIAS_OFFSET;
        uint32_t smem_ptr_scale_grad = smem_wr_addr_ + 2 * BIAS_OFFSET;

        // 16816 => K = 16. FP16 elements. LDGSTS = 128Bits = 16B = 8 FP16 elements
        // LDGSTS32 => 2 elements => 8 elements;
        xmma::ldgsts<4>( smem_ptr_scale_grad, gmem_ptrs_scale_grad, pred);
        xmma::ldgsts<4>( smem_ptr_scale_fprop, gmem_ptrs_scale_fprop, pred);
        xmma::ldgsts<4>( smem_ptr_bias, gmem_ptrs_bias, pred );

        // Size of one Row of SCALE or BIAS TILE
        const int BYTES_PER_ROW_SCALE = NUM_THREADS_LOADING * sizeof(uint32_t);

        // Move the smem write offset by 1 row
        smem_wr_addr_ += BYTES_PER_ROW_SCALE;
    }

    // Reset smem write pointer to start of circular buffer
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
    inline __device__ void setup(const int &loop_cnt, const dim3 &bid, const uint32_t& x_max, const uint32_t& y_max) {
        loop_count_ = loop_cnt;
        bid_m = bid.x;
        bid_n = bid.y;
        max_col = x_max;
        max_row = y_max;
    }

    // Scale and bias
    uint2 scale_, bias_, scale_grad_;

    // Residual pointers
    const void *gmem_residual_ptrs_[LDGS];

    // Offset for the residual in shmem
    uint32_t smem_residual_offset_;

    // Current stage bring processed in the multistage pipeline
    int curr_stage_;

    // Current loop count being processed
    int curr_loop_;

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

    // Warp, thread locations
    uint32_t warp_id_offset;
    uint32_t thread_in_warp_row;
    uint32_t thread_in_warp_col;
    uint32_t warp_col_id;

};

////////////////////////////////////////////////////////////////////////////////////////////////////
} // namespace dgrad
} // namespace bn_apply
} // namespace batchnorm
} // namespace ext
} // namespace xmma
////////////////////////////////////////////////////////////////////////////////////////////////////
