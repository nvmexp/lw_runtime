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

#include <xmma/ext/batchnorm/bn_apply/gmem_tile.h>
////////////////////////////////////////////////////////////////////////////////////////////////////

namespace xmma {
namespace ext {
namespace batchnorm {
namespace bn_apply {

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Cta_tile, typename Input_related_, bool WITH_RESIDUAL, 
           bool WITH_BNA_RESIDUAL, int STAGES_>
struct Gmem_tile_a<xmma::Ampere_hmma_fp32_traits, Cta_tile, Input_related_, WITH_RESIDUAL,
                   WITH_BNA_RESIDUAL, STAGES_>
    : public xmma::implicit_gemm::fprop::Gmem_tile_a_t<xmma::Ampere_hmma_fp32_traits,
                                                         Cta_tile, Input_related_> {

    static_assert( WITH_RESIDUAL ? 1 : !WITH_BNA_RESIDUAL, "Illegal combination of template args"); 

    // The traits class.
    using Traits = xmma::Ampere_hmma_fp32_traits;

    // The base class.
    using Base = xmma::implicit_gemm::fprop::Gmem_tile_a_t<Traits, Cta_tile, Input_related_>;

    // Xmma tile in Ampere
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;

    // Number of stages in the multistage pipeline
    enum { STAGES = xmma::Max<STAGES_, 2>::VALUE };

    // Ctor.
    template <typename Params>
    inline __device__ Gmem_tile_a(const Params &params, void *smem, const dim3 &bidx, int tidx)
        : Base(params, smem, bidx, tidx),
          params_filter_trs_per_cta_(params.filter_trs_per_cta), c_max_(params.g * params.c),
          setup_done_(0), residue_flag(0), bidx(bidx), x_max(params.trsc),
          y_max(params.n * params.h * params.w) {

        // Always start at stage 0
        lwrr_stage_ = 0;
        lwrr_rs_ = 0;

        // Only the first 4 threads will be loading scale and bias
        // TODO : We can go wider here based on the K dim. Here we only cater to LDSM. Since, we cater to LDSM we do a LDGSTS.32
        //        instead of LDGSTS.128.
        loading_needed = int(threadIdx.x < 4);

        const int bidz = bidx.z;

        // Total number of loops per stage - used in shmem iterations
        const int TOTAL_NUM_LOOPS = params.loop_start + 1;
        const int NUM_STAGES_TOTAL =
            xmma::div_up(params.c, Xmma_tile::XMMAS_K * Xmma_tile::K_PER_XMMA);
        const int NUM_LOOPS_PER_STAGE = TOTAL_NUM_LOOPS * Xmma_tile::XMMAS_K / NUM_STAGES_TOTAL;
        loop_count_ = NUM_LOOPS_PER_STAGE;

        if( loading_needed ) {
            // The current position in the C dimension.
            int c = bidz * Cta_tile::K + (tidx % 4) * 2;
            c_ = c;

            // Add a bit to track c < params.c in the preds_ register.
            if (c < params.g * params.c) {
                this->preds_[0] = this->preds_[0] | 0x80000000;
            }

            // The scale.
            const char *scale_ptr = reinterpret_cast<const char *>(params.bn_scale_gmem);
            scale_ptr_ = &scale_ptr[c * sizeof(uint16_t)];

            const char *bias_ptr = reinterpret_cast<const char *>(params.bn_bias_gmem);
            bias_ptr_ = &bias_ptr[c * sizeof(uint16_t)];

            // If Residual also needs a BNa
            if( WITH_BNA_RESIDUAL ) {
                const char *scale_ptr = reinterpret_cast<const char *>(params.bn_res_scale_gmem);
                res_scale_ptr_ = &scale_ptr[c * sizeof(uint16_t)];

                const char *bias_ptr = reinterpret_cast<const char *>(params.bn_res_bias_gmem);
                res_bias_ptr_ = &bias_ptr[c * sizeof(uint16_t)];
            }
        }

        // Define the residual pointer.
        if (WITH_RESIDUAL) {
            res_ptr_ = reinterpret_cast<const char *>(params.bn_res_gmem);
            res_add_relu_out_ptr_ = reinterpret_cast<uint32_t *>(params.bn_res_add_relu_out_gmem);
            bitmask_relu_out_ptr_ = reinterpret_cast<uint32_t *>(params.bn_bitmask_relu_out_gmem);
        }
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
        if (WITH_RESIDUAL) {
            const char *res_ptrs[Base::LDGS];
            #pragma unroll
            for (int ii = 0; ii < Base::LDGS; ++ii) {
                res_ptrs[ii] = res_ptr_ + Traits::offset_in_bytes_a(this->offsets_[ii]);
            }

            s.set_residue_pointers<Base::LDGS>(res_ptrs);
        }

        // Issue the loads = LDGSTS of activation and residuals
        // This will atuomatically make sure the OOB pixels are also set to a special NaN
        Base::load(s, mem_desc);

        // Load scale and bias into a cirlwlar buffer in smem
        if ((params_filter_trs_per_cta_ == 1 || lwrr_rs_ == 0) &&
                (this->preds_[0] || (residue_flag==1)) && loading_needed) {

            constexpr int ldsm_width = 8;
            constexpr int instr_k = Xmma_tile::K_PER_XMMA;
            constexpr int loop_ki = instr_k / ldsm_width;
            constexpr int num_ldgs = Xmma_tile::XMMAS_K * loop_ki;

            const char *scale_ptr_cpy, *bias_ptr_cpy;
            const char *res_scale_ptr_cpy, *res_bias_ptr_cpy;
            scale_ptr_cpy = scale_ptr_;
            bias_ptr_cpy = bias_ptr_;
            if( WITH_BNA_RESIDUAL ){
                res_scale_ptr_cpy = res_scale_ptr_;
                res_bias_ptr_cpy = res_bias_ptr_;
            }
            uint32_t pred;
            constexpr int offset = ldsm_width * sizeof(uint16_t);
            #pragma unroll
            for (int i = 0; i < num_ldgs; ++i) {
                pred = static_cast<int>(c_ < c_max_);

                // LDGSTS of scale and bias into cirlwlar buffer in shmem
                s.scale_bias_load(scale_ptr_cpy, bias_ptr_cpy, pred);

                if( WITH_BNA_RESIDUAL ){
                    s.res_scale_bias_load(res_scale_ptr_cpy, res_bias_ptr_cpy, pred);
                }

                if( pred ) {
                    bias_ptr_cpy += offset;
                    scale_ptr_cpy += offset;
                    if( WITH_BNA_RESIDUAL ){
                        res_bias_ptr_cpy += offset;
                        res_scale_ptr_cpy += offset;
                    }
                    c_ += ldsm_width;
                }
            }
        }

        if ((params_filter_trs_per_cta_ == 1 || lwrr_rs_ == 0) && this->preds_[0] &&
                loading_needed) {
            // Every time load is called - move to the next stage
            ++lwrr_stage_;
            lwrr_stage_ %= STAGES;

            if (lwrr_stage_ == 0) {
                s.reset_smem_wr_offset();
            }
        }

        // pass the parameters to shmem object at the start
        if ( !setup_done_ ) {
            if( WITH_RESIDUAL ) {
                s.setup(loop_count_, bidx, x_max, y_max,
                         res_add_relu_out_ptr_, bitmask_relu_out_ptr_);
            } else {
                s.setup(loop_count_);
            }
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
        if (WITH_RESIDUAL) {
            res_ptr_ += delta;
        }

        // Move scale and bias pointers
        if ((params_filter_trs_per_cta_ == 1 || next_rs == 0) && loading_needed) {

            int offset = gridDim.z * Cta_tile::K * sizeof(uint16_t);

            scale_ptr_ += offset;
            bias_ptr_ += offset;
            if( WITH_BNA_RESIDUAL ){
                res_scale_ptr_ += offset;
                res_bias_ptr_ += offset;
            }
        }

        // Update the RS (used only in 3x3 colw)
        lwrr_rs_ = next_rs;
    }

    // RS for the filter parameter
    const int params_filter_trs_per_cta_;

    // Current index in the C-dimension
    int c_;

    // Maximum C dimension for the input
    const int c_max_;

    // Lwrent stage being processed/loaded
    int lwrr_stage_;

    // Current RS being loaded
    int lwrr_rs_;

    // Is setup done for the smem tile
    int setup_done_;

    // Loops per stage - used by shmem tile
    int loop_count_;

    // The pointers for scale and bias - used in normalization
    const char *scale_ptr_, *bias_ptr_;
    const char *res_scale_ptr_, *res_bias_ptr_;

    // Residual add input ptr
    const char *res_ptr_;

    // Residual_add output ptr
    uint32_t *res_add_relu_out_ptr_;

    // Bitmask_relu output ptr
    uint32_t *bitmask_relu_out_ptr_;

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

template <typename Cta_tile, bool SIMPLE_1x1x1>
struct Gmem_tile_b<xmma::Ampere_hmma_fp32_traits, Cta_tile, SIMPLE_1x1x1>
    : public xmma::implicit_gemm::wgrad_indexed::Gmem_tile_b_t<xmma::Ampere_hmma_fp32_traits,
                                                         Cta_tile, SIMPLE_1x1x1> {

    // The traits class.
    using Traits = xmma::Ampere_hmma_fp32_traits;

    // The base class.
    using Base = xmma::implicit_gemm::wgrad_indexed::Gmem_tile_b_t<Traits, Cta_tile, SIMPLE_1x1x1>;

    // Xmma tile in Ampere
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;

    // Ctor.
    template <typename Params>
    inline __device__ Gmem_tile_b(const Params &params, void *smem, const dim3 &bidx, int tidx)
        : Base(params, smem, bidx, tidx)
        , setup_done_(0)
        , c_max_(params.c)
        , crs_max_(params.c * params.r * params.s) {

        const int bidy = bidx.y;
        const int warp_id = threadIdx.x / 32;
        const int warp_col_id = warp_id / Cta_tile::WARPS_M;

        constexpr int LDSM_WIDTH = 4;
        constexpr int LDSM_HEIGHT = 8;

        // Each CTA can hold multiple RS Values - so we first figure out where we are in CRS
        // crs_start = cta_width * cta_id_n + warp_id_n * pixels_per_XMMA_TILE_N
        //                 + Location inside LDSM.MT
        crs_start_ = bidy * Cta_tile::N + warp_col_id * Xmma_tile::N_PER_XMMA
                        + (tidx / LDSM_WIDTH) % LDSM_HEIGHT;

        valid_pixel_ = crs_start_ < crs_max_;

        // Add a bit to track in the preds_ register.
        if( (! SIMPLE_1x1x1) && valid_pixel_ ) {
            this->preds_[0] = this->preds_[0] | 0x80000000;
        }

        // Scale and bias pointer
        scale_ptr_ = reinterpret_cast<const char *>(params.bn_scale_gmem);
        bias_ptr_ = reinterpret_cast<const char *>(params.bn_bias_gmem);
    }

    // Loads the scale and bias from gmem
    template <typename Smem_tile>
    inline __device__ void load_scale_bias(Smem_tile &s) {
        constexpr int ldsm_width = 8;
        constexpr int instr_n = Xmma_tile::N_PER_XMMA;
        constexpr int loop_ni = instr_n / ldsm_width;

        constexpr uint32_t outer_loop_offset = (Cta_tile::WARPS_N-1) * ldsm_width * loop_ni;

        uint16_t tmp_scale[2];
        uint16_t tmp_bias[2];
        const uint32_t c_start = crs_start_ % c_max_;

        int crs_lwrr = crs_start_;
        int c_offset = c_start;

        #pragma unroll
        for(int i = 0; i < Xmma_tile::XMMAS_N; ++i) {

            #pragma unroll
            for(int j = 0; j < loop_ni; ++j) {

                // Pixel Valid can be figured out only using CRS - since CTAs have min overcompute
                // i.e one CTA can have numtiple RSC blocks in it - so C can loop around
                valid_pixel_ = crs_lwrr < crs_max_;

                if (valid_pixel_) {
                    // C can loop around based on where in CRS we are
                    c_offset %= c_max_;
                    const char* bias_addr = bias_ptr_ + c_offset * sizeof(uint16_t);
                    const char* scale_addr = scale_ptr_ + c_offset * sizeof(uint16_t);

                    xmma::ldg(tmp_scale[j], scale_addr, this->preds_[0]);
                    xmma::ldg(tmp_bias[j], bias_addr, this->preds_[0]);
                }

                crs_lwrr += ldsm_width;
                c_offset += ldsm_width;
            }

            uint32_t *tmp_scale_ptr = reinterpret_cast<uint32_t*>(tmp_scale);
            uint32_t *tmp_bias_ptr = reinterpret_cast<uint32_t*>(tmp_bias);

            s.store_scale_bias(tmp_scale_ptr, tmp_bias_ptr, i);

            crs_lwrr += outer_loop_offset;
            c_offset += outer_loop_offset;
        }
    }

    // Load a tile from global memory.
    template <typename Smem_tile>
    inline __device__ void load(Smem_tile &s, const uint64_t mem_desc) {

        // Issue the loads = LDGSTS of activation and residuals
        // this will also set OOB pixels to a special NaN value
        Base::load(s, mem_desc);

        // pass the parameters to shmem object at the start
        if ( !setup_done_ ) {
            load_scale_bias(s);
            setup_done_ = 1;
        }
    }

    // Move the pointers and update the predicates for R2P/P2R.
    inline __device__ void move(int next_rs, int64_t delta) {

        // Move the pointer.
        Base::move(next_rs, delta);

        // Add the bit to enable the loads. The residue code will clear it if needed.
        if( ! SIMPLE_1x1x1 ) {
            this->preds_[0] = this->preds_[0] | 0x80000000;
        }
    }

    // Is setup done for the smem tile
    int setup_done_;

    // The pointers for scale and bias - used in normalization
    const char *scale_ptr_, *bias_ptr_;

    // Is pixel valid / In bounds
    uint32_t valid_pixel_;

    // Tensor dimension C
    const uint32_t c_max_;

    // Tensor dimension CRS
    const uint32_t crs_max_;

    // Starting CRS
    int crs_start_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Cta_tile, bool SIMPLE_1x1x1, bool WITH_FUSED_DBNA_DGRAD>
struct Gmem_tile_a_wgrad<xmma::Ampere_hmma_fp32_traits, Cta_tile, SIMPLE_1x1x1, WITH_FUSED_DBNA_DGRAD>
    : public xmma::implicit_gemm::wgrad_indexed::Gmem_tile_a_n<xmma::Ampere_hmma_fp32_traits,
                                                         Cta_tile > {

    // The traits class.
    using Traits = xmma::Ampere_hmma_fp32_traits;

    // The base class.
    using Base = xmma::implicit_gemm::wgrad_indexed::Gmem_tile_a_n<Traits, Cta_tile>;

    // Xmma tile in Ampere
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;

    // Ctor.
    template <typename Params>
    inline __device__ Gmem_tile_a_wgrad(const Params &params, void *smem, const dim3 &bidx, int tidx)
        : Base(params, smem, bidx, tidx) 
        , setup_done_(0)
        , k_max_(params.k) {

        if( WITH_FUSED_DBNA_DGRAD ) {
            const int bid = bidx.x; // along M
            // Scale and bias pointer, and fprop tensor ptr
            fprop_scale_ptr_ = reinterpret_cast<const char *>(params.bna_fprop_tensor_scale_gmem);
            fprop_bias_ptr_ = reinterpret_cast<const char *>(params.bna_bias_gmem);
            dgrad_scale_ptr_ = reinterpret_cast<const char *>(params.bna_grad_scale_gmem);
            fprop_tensor_ptr_ = reinterpret_cast<const char *>(params.bna_fprop_tensor_gmem);

            // The current position in the K dimension.
            uint32_t warp_row_id = (threadIdx.x / 32) % Cta_tile::WARPS_M;
            k_start_ = bid * Cta_tile::M + warp_row_id * Xmma_tile::M_PER_XMMA + (threadIdx.x % 32) / 4;
            valid_pixel_ = k_start_ < k_max_;

            // Add a bit to track in the preds_ register.
            if( (! SIMPLE_1x1x1) && valid_pixel_ ) {
                this->preds_[0] = this->preds_[0] | 0x80000000;
            }
        }
    }

    // Loads the scale and bias from gmem
    template <typename Smem_tile>
    inline __device__ void load_scale_bias(Smem_tile &s) {
        constexpr int ldsm_height = 8;
        constexpr int instr_m = Xmma_tile::M_PER_XMMA;
        constexpr int loop_mi = instr_m / ldsm_height;

        constexpr uint32_t outer_loop_offset = (Cta_tile::WARPS_M-1) * ldsm_height * loop_mi;

        // Every XMMA is a 16x16x16
        // Thus we will have 2 Scale and 2 Biases per thread
        // Unfortunately we have to do either a LDG.16 here or LDS.16 in the mainloop
        // TODO : If perf. is affected because of this - we might need to prefer LDG.128 + LDS.16
        uint16_t tmp_fprop_scale[2];
        uint16_t tmp_fprop_bias[2];
        uint16_t tmp_dgrad_scale[2];

        // Starting offset
        int k_offset = k_start_;

        // We have to use LDG.32, since Thread zero needs 
        #pragma unroll
        for(int i = 0; i < Xmma_tile::XMMAS_M; ++i) {

            #pragma unroll
            for(int j = 0; j < loop_mi; ++j) {

                // Pixel Valid can be figured out only using CRS - since CTAs have min overcompute
                // i.e one CTA can have numtiple RSC blocks in it - so C can loop around
                valid_pixel_ = k_offset < k_max_;

                if( valid_pixel_ ) {
                    const char* fprop_bias_addr  = fprop_bias_ptr_  + k_offset * sizeof(uint16_t);
                    const char* fprop_scale_addr = fprop_scale_ptr_ + k_offset * sizeof(uint16_t);
                    const char* dgrad_scale_addr = dgrad_scale_ptr_ + k_offset * sizeof(uint16_t);

                    xmma::ldg(tmp_fprop_scale[j], fprop_scale_addr, this->preds_[0]);
                    xmma::ldg(tmp_fprop_bias[j] , fprop_bias_addr , this->preds_[0]);
                    xmma::ldg(tmp_dgrad_scale[j], dgrad_scale_addr, this->preds_[0]);
                }
                k_offset += ldsm_height;
            }

            uint32_t *tmp_fprop_scale_ptr = reinterpret_cast<uint32_t*>(tmp_fprop_scale);
            uint32_t *tmp_fprop_bias_ptr  = reinterpret_cast<uint32_t*>(tmp_fprop_bias);
            uint32_t *tmp_dgrad_scale_ptr = reinterpret_cast<uint32_t*>(tmp_dgrad_scale);

            // Save the scale and biases
            s.store_scale_bias(tmp_fprop_scale_ptr, tmp_fprop_bias_ptr, tmp_dgrad_scale_ptr, i);
            k_offset += outer_loop_offset;
        }
    }

    // Load a tile from global memory.
    template <typename Smem_tile>
    inline __device__ void load(Smem_tile &s, const uint64_t mem_desc) {

        // Load the Fusion stuff first
        if( WITH_FUSED_DBNA_DGRAD ) {
            // TODO : Similarly load Extra Fprop tensor - aka residual
            const char *res_ptrs[Base::LDGS];
            #pragma unroll
            for (int ii = 0; ii < Base::LDGS; ++ii) {
                res_ptrs[ii] = fprop_tensor_ptr_ + Traits::offset_in_bytes_a(this->offsets_[ii]);
            }

            // Save the pointer offsets
            s.set_residue_pointers<Base::LDGS>( res_ptrs );

            // pass the parameters to shmem object at the start
            if ( !setup_done_ ) {
                load_scale_bias(s);
                setup_done_ = 1;
            }
        }

        // Issue the loads = LDGSTS of activation and residuals
        // this will also set OOB pixels to a special NaN value
        Base::load(s, mem_desc);
    }

    // Move the pointers and update the predicates for R2P/P2R.
    inline __device__ void move(int next_rs, int64_t delta) {

        // Move the pointer.
        Base::move(next_rs, delta);

        // Add the bit to enable the loads. The residue code will clear it if needed.
        if( !SIMPLE_1x1x1 && WITH_FUSED_DBNA_DGRAD) {
            this->preds_[0] = this->preds_[0] | 0x80000000;
        }

        if ( WITH_FUSED_DBNA_DGRAD ) {
            fprop_tensor_ptr_ += delta;
        }
    }

    // Is setup done for the smem tile
    int setup_done_;

    // The pointers for scale and bias - used in normalization
    const char *fprop_scale_ptr_, *fprop_bias_ptr_, *dgrad_scale_ptr_;

    // Fprop input tensor - used only when Dgrad is fused
    const char *fprop_tensor_ptr_;

    // Is pixel valid / In bounds
    uint32_t valid_pixel_;

    // Tensor dimension C
    const uint32_t k_max_;

    // Starting K
    int k_start_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////
} // namespace bn_apply
} // namespace batchnorm
} // namespace ext
} // namespace xmma
////////////////////////////////////////////////////////////////////////////////////////////////////
