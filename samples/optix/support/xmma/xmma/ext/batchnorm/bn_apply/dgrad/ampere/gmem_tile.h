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

#include <xmma/ext/batchnorm/bn_apply/gmem_tile.h>
////////////////////////////////////////////////////////////////////////////////////////////////////

namespace xmma {
namespace ext {
namespace batchnorm {
namespace bn_apply {
namespace dgrad {

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Cta_tile, typename Input_related_, int STAGES_>
struct Gmem_tile_a_dbna_dgrad<xmma::Ampere_hmma_fp32_traits, Cta_tile, Input_related_, STAGES_>
    : public xmma::implicit_gemm::dgrad::Gmem_tile_a_t<xmma::Ampere_hmma_fp32_traits,
                                                         Cta_tile, Input_related_, 16> {

    // The traits class.
    using Traits = xmma::Ampere_hmma_fp32_traits;

    // The base class.
    using Base = xmma::implicit_gemm::dgrad::Gmem_tile_a_t<Traits, Cta_tile, Input_related_>;

    // Xmma tile in Ampere
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;

    // Number of stages in the multistage pipeline
    enum { STAGES = xmma::Max<STAGES_, 2>::VALUE };

    // Ctor.
    template <typename Params>
    inline __device__ Gmem_tile_a_dbna_dgrad(const Params &params, void *smem, const dim3 &bidx, int tidx)
        : Base(params, smem, bidx, tidx),
          params_filter_trs_per_cta_(params.filter_trs_per_cta), c_max_(params.g * params.k),
          setup_done_(0), residue_flag(0), bidx(bidx), x_max(params.trs * params.k),
          y_max(params.n * params.p * params.q) {

        // Always start at stage 0
        curr_stage_ = 0;
        curr_rs_ = 0;

        // Only the first 4 threads will be loading scale and bias
        loading_needed = int(threadIdx.x < 4);

        const int bidz = bidx.z;

        // Total number of loops per stage - used in shmem iterations
        const int TOTAL_NUM_LOOPS = params.loop_start + 1;
        const int NUM_STAGES_TOTAL =
            xmma::div_up(params.k, Xmma_tile::XMMAS_K * Xmma_tile::K_PER_XMMA);
        const int NUM_LOOPS_PER_STAGE = TOTAL_NUM_LOOPS * Xmma_tile::XMMAS_K / NUM_STAGES_TOTAL;
        loop_count_ = NUM_LOOPS_PER_STAGE;

        if( loading_needed ) {
            // The current position in the C dimension.
            int c = bidz * Cta_tile::K + (tidx % 4) * 2;
            c_ = c;

            // Add a bit to track c < params.c in the preds_ register.
            if (c < params.g * params.k) {
                this->preds_[0] = this->preds_[0] | 0x80000000;
            }

            // The fprop tensor's scale.
            const char *bna_fprop_tensor_scale_ptr = reinterpret_cast<const char *>(params.bna_fprop_tensor_scale_gmem);
            bna_fprop_tensor_scale_ptr_ = &bna_fprop_tensor_scale_ptr[c * sizeof(uint16_t)];

            // The gradient tensor's scale.
            const char *bna_gradient_scale_ptr = reinterpret_cast<const char *>(params.bna_gradient_scale_gmem);
            bna_gradient_scale_ptr_ = &bna_gradient_scale_ptr[c * sizeof(uint16_t)];

            const char *bna_bias_ptr = reinterpret_cast<const char *>(params.bna_bias_gmem);
            bna_bias_ptr_ = &bna_bias_ptr[c * sizeof(uint16_t)];
        }

        // Define the residual pointer.
        bna_fprop_tensor_ptr_ = reinterpret_cast<const char *>(params.bna_fprop_tensor_gmem);
    }

    // The residue to "fix" the predicates.
    inline __device__ void residue( int masks_to_clear = Base::LDGS ) {
        Base::residue(masks_to_clear);
        residue_flag++;
    }

    // Load a tile from global memory.
    template <typename Smem_tile>
    inline __device__ void load(Smem_tile &s, const uint64_t mem_desc) {

        // Load the residual first, since we need to notify the offsets
        // needed for LDGSTS of residuals
        const char *bna_fprop_tensor_ptrs[Base::LDGS];
        #pragma unroll
        for (int ii = 0; ii < Base::LDGS; ++ii) {
            bna_fprop_tensor_ptrs[ii] = bna_fprop_tensor_ptr_ + Traits::offset_in_bytes_a(this->offsets_[ii]);
        }
        s.set_residue_pointers<Base::LDGS>(bna_fprop_tensor_ptrs);

        // Issue the loads = LDGSTS of activation and residuals
        // This will atuomatically make sure the OOB pixels are also set to a special NaN
        Base::load(s, mem_desc);

        // Load scale and bias into a circular buffer in smem
        if ((params_filter_trs_per_cta_ == 1 || curr_rs_ == 0) &&
                (this->preds_[0] || (residue_flag==1)) && loading_needed) {

            constexpr int LDSM_WIDTH = 8;
            constexpr int instr_k = Xmma_tile::K_PER_XMMA;
            constexpr int loop_ki = instr_k / LDSM_WIDTH;
            constexpr int num_ldgs = Xmma_tile::XMMAS_K * loop_ki;

            const char *bna_fprop_tensor_scale_ptr_cpy, *bna_gradient_scale_ptr_cpy, *bna_bias_ptr_cpy;
            bna_fprop_tensor_scale_ptr_cpy = bna_fprop_tensor_scale_ptr_;
            bna_gradient_scale_ptr_cpy = bna_gradient_scale_ptr_;
            bna_bias_ptr_cpy = bna_bias_ptr_;
            uint32_t pred;
            constexpr int offset = LDSM_WIDTH * sizeof(uint16_t);
            #pragma unroll
            for (int i = 0; i < num_ldgs; ++i) {
                pred = static_cast<int>(c_ < c_max_);

                // LDGSTS of scale and bias into circular buffer in shmem
                s.scale_bias_load(bna_fprop_tensor_scale_ptr_cpy, bna_gradient_scale_ptr_cpy, bna_bias_ptr_cpy, pred);

                if( pred ) {
                    bna_bias_ptr_cpy += offset;
                    bna_fprop_tensor_scale_ptr_cpy += offset;
                    bna_gradient_scale_ptr_cpy += offset;
                    c_ += LDSM_WIDTH;
                }
            }
        }

        if ((params_filter_trs_per_cta_ == 1 || curr_rs_ == 0) && this->preds_[0] &&
                loading_needed) {
            // Every time load is called - move to the next stage
            ++curr_stage_;
            curr_stage_ %= STAGES;

            if (curr_stage_ == 0) {
                s.reset_smem_wr_offset();
            }
        }

        // pass the parameters to shmem object at the start
        if ( !setup_done_ ) {
            s.setup(loop_count_, bidx, x_max, y_max);
            setup_done_ = 1;
        }
    }

    // Move the pointers and update the predicates for R2P/P2R.
    inline __device__ void move(int next_rs, int64_t delta) {

        // Move the pointer.
        Base::move(next_rs, delta);

        // Add the bit to enable the loads. The residue code will clear it if needed.
        this->preds_[0] = this->preds_[0] | 0x80000000;

        // Update the pointer.
        bna_fprop_tensor_ptr_ += delta;

        // Move scale and bias pointers
        if ((params_filter_trs_per_cta_ == 1 || next_rs == 0) && loading_needed) {

            int offset = gridDim.z * Cta_tile::K * sizeof(uint16_t);

            bna_fprop_tensor_scale_ptr_ += offset;
            bna_gradient_scale_ptr_ += offset;
            bna_bias_ptr_ += offset;
        }

        // Update the RS (used only in 3x3 conv)
        curr_rs_ = next_rs;
    }

    // RS for the filter parameter
    const int params_filter_trs_per_cta_;

    // Current index in the C-dimension
    int c_;

    // Maximum C dimension for the input
    const int c_max_;

    // Curent stage being processed/loaded
    int curr_stage_;

    // Current RS being loaded
    int curr_rs_;

    // Is setup done for the smem tile
    int setup_done_;

    // Loops per stage - used by shmem tile
    int loop_count_;

    // The pointers for scale and bias - used in normalization
    const char *bna_gradient_scale_ptr_, *bna_bias_ptr_, *bna_fprop_tensor_scale_ptr_;

    // bna fprop input ptr
    const char *bna_fprop_tensor_ptr_;

    // Flag for residue iteration
    int residue_flag;

    // Current block Id (accounting for cta-swizzle)
    const dim3 bidx;

    // Maximum height and width of input
    const int x_max, y_max;

    // Does scale, bias need to be loaded by this thread ?
    // If not other othreads will load and store in shmem
    int loading_needed;
};

////////////////////////////////////////////////////////////////////////////////////////////////////
} // namespace dgrad
} // namespace bn_apply
} // namespace batchnorm
} // namespace ext
} // namespace xmma
////////////////////////////////////////////////////////////////////////////////////////////////////
