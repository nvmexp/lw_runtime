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

#include <xmma/implicit_gemm/fprop/gmem_tile.h>

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace xmma {
namespace ext {
namespace batchnorm {
namespace bn_apply {

////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  GMEM TILE A
//
////////////////////////////////////////////////////////////////////////////////////////////////////

// Base class for Volta and Turing, since they share common code
template <typename Traits, typename Cta_tile, typename Input_related, bool WITH_RESIDUAL,
          int STAGES, bool WITH_RELU>
struct Gmem_tile_a_base
    : public xmma::implicit_gemm::fprop::Gmem_tile_a_t<Traits, Cta_tile, Input_related> {

    // The base class.
    using Base = xmma::implicit_gemm::fprop::Gmem_tile_a_t<Traits, Cta_tile, Input_related>;

    // Ctor.
    template <typename Params>
    inline __device__ Gmem_tile_a_base(const Params &params, void *smem, const dim3 &bidx, int tidx)
        : Base(params, smem, bidx, tidx),
          params_filter_trs_per_cta_(params.filter_trs_per_cta) {

        const int bidz = bidx.z;

        // Define the residual pointer.
        if (WITH_RESIDUAL) {
            res_ptr_ = reinterpret_cast<const char *>(params.bn_res_gmem);
        }

        // The position in the C dimension.
        int c = bidz * Cta_tile::K + tidx % Base::THREADS_PER_PIXEL * Base::ELTS_PER_LDG;

        // Add a bit to track c < params.c in the preds_ register.
        if (c < params.g * params.c) {
            this->preds_[0] = this->preds_[0] | 0x80000000;
        }

        // The scale.
        const char *scale_ptr = reinterpret_cast<const char *>(params.bn_scale_gmem);
        scale_ptr_ = &scale_ptr[c * sizeof(uint16_t)];
        scale_ = make_uint4(0, 0, 0, 0);
        if (c < params.g * params.c) {
            xmma::ldg(scale_, scale_ptr_);
        }

        // The bias.
        const char *bias_ptr = reinterpret_cast<const char *>(params.bn_bias_gmem);
        bias_ptr_ = &bias_ptr[c * sizeof(uint16_t)];

        bias_ = make_uint4(0, 0, 0, 0);
        if (c < params.g * params.c) {
            xmma::ldg(bias_, bias_ptr_);
        }
    }

    // Store the pixels to shared memory.
    template <typename Xmma_smem_tile> inline __device__ void commit(Xmma_smem_tile &smem) {
        // print_vals();

        // Scale and add the bias.
        xmma::scale_bias(reinterpret_cast<uint4(&)[Base::LDGS]>(this->fetch_), scale_, bias_,
                             this->preds_[0]);

        // Add the residual if needed.
        if ( WITH_RESIDUAL ) {
            #pragma unroll
            for (int ii = 0; ii < Base::LDGS; ++ii) {
                this->fetch_[ii].x = xmma::hadd2(this->fetch_[ii].x, res_fetch_[ii].x);
                this->fetch_[ii].y = xmma::hadd2(this->fetch_[ii].y, res_fetch_[ii].y);
                this->fetch_[ii].z = xmma::hadd2(this->fetch_[ii].z, res_fetch_[ii].z);
                this->fetch_[ii].w = xmma::hadd2(this->fetch_[ii].w, res_fetch_[ii].w);
            }
        }

        // Apply RELU. if needed
        if ( WITH_RELU ) {
        #pragma unroll
        for (int mi = 0; mi < Base::LDGS; ++mi) {
                this->fetch_[mi].x = xmma::relu_fp16x2(this->fetch_[mi].x);
                this->fetch_[mi].y = xmma::relu_fp16x2(this->fetch_[mi].y);
                this->fetch_[mi].z = xmma::relu_fp16x2(this->fetch_[mi].z);
                this->fetch_[mi].w = xmma::relu_fp16x2(this->fetch_[mi].w);
            }
        }

        // Store to shared memory.
        Base::commit(smem);
    }

    // Load a tile from global memory.
    template <typename Smem_tile>
    inline __device__ void load(Smem_tile &s, const uint64_t mem_desc) {
        // Issue the loads.
        Base::load(s, mem_desc);

        int rs = 0;

        // Load the residual.
        if (WITH_RESIDUAL) {
            const char *res_ptrs[Base::LDGS];
            #pragma unroll
            for (int ii = 0; ii < Base::LDGS; ++ii) {
                res_ptrs[ii] = res_ptr_ + Traits::offset_in_bytes_a(this->offsets_[ii]);
            }
            xmma::ldg<Base::LDGS>(reinterpret_cast<uint4(&)[Base::LDGS]>(res_fetch_),
                                  reinterpret_cast<const void *(&)[Base::LDGS]>(res_ptrs),
                                  this->preds_);
        }

        // Load the new bias/scale.
        if (params_filter_trs_per_cta_ == 1 || rs == 0) {
            if (this->preds_[0]) {
                xmma::ldg(scale_, scale_ptr_);
                xmma::ldg(bias_, bias_ptr_);
            }
        }
    }

    // Move the pointers and update the predicates for R2P/P2R.
    inline __device__ void move(int next_rs, int64_t delta, int first = 0) {
        // Move the pointer.
        Base::move(next_rs, delta);

        // Add the bit to enable the loads. The residue code will clear it if
        // needed.
        this->preds_[0] = this->preds_[0] | 0x80000000;

        // Update the pointer.
        if (WITH_RESIDUAL) {
            res_ptr_ += delta;
        }

        // Move scale and bias.
        if (params_filter_trs_per_cta_ == 1 || next_rs == 0) {
            scale_ptr_ += gridDim.z * Cta_tile::K * sizeof(uint16_t);
            bias_ptr_ += gridDim.z * Cta_tile::K * sizeof(uint16_t);
        }
    }

    // Filter TRS.
    const int params_filter_trs_per_cta_;

    // The pointers for scale and bias.
    const char *scale_ptr_, *bias_ptr_;

    // The bias and scale.
    uint4 scale_, bias_;

    // The pointer for the residuals.
    const char *res_ptr_;

    // The fetch registers.
    int4 res_fetch_[Base::LDGS];
};

template <typename Traits, typename Cta_tile, typename Input_related, bool WITH_RESIDUAL,
          int STAGES, bool WITH_RELU>
struct Gmem_tile_a_volta_turing
    : public Gmem_tile_a_base<Traits, Cta_tile, Input_related, WITH_RESIDUAL, STAGES, WITH_RELU> {
};

template <typename Cta_tile, typename Input_related, bool WITH_RESIDUAL, int STAGES, bool WITH_RELU>
struct Gmem_tile_a_volta_turing<xmma::Volta_hmma_fp32_traits, Cta_tile, Input_related,
                                WITH_RESIDUAL, STAGES, WITH_RELU>
    : public Gmem_tile_a_base<xmma::Volta_hmma_fp32_traits, Cta_tile, Input_related,
                              WITH_RESIDUAL, STAGES, WITH_RELU> {

    // The base class.
    using Base_gmem = Gmem_tile_a_base<xmma::Volta_hmma_fp32_traits, Cta_tile, Input_related,
                                       WITH_RESIDUAL, STAGES, WITH_RELU>;

    // Ctor.
    template <typename Params>
    inline __device__ Gmem_tile_a_volta_turing(const Params &params, void *smem, const dim3 &bidx,
                                               int tidx)
        : Base_gmem(params, smem, bidx, tidx) {}
};

////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  GMEM TILE B
//
////////////////////////////////////////////////////////////////////////////////////////////////////
// Base class for Volta and Turing, since they share common code
template <typename Traits, typename Cta_tile, bool SIMPLE_1x1x1, bool WITH_RELU>
struct Gmem_tile_b_base
    : public xmma::implicit_gemm::wgrad_indexed::Gmem_tile_b_t<Traits, Cta_tile, SIMPLE_1x1x1> {

    // The base class.
    using Base = xmma::implicit_gemm::wgrad_indexed::Gmem_tile_b_t<Traits, Cta_tile, SIMPLE_1x1x1>;

    // Ctor.
    template <typename Params>
    inline __device__ Gmem_tile_b_base(const Params &params, void *smem, const dim3 &bidx, int tidx)
        : Base(params, smem, bidx, tidx) {

        // The position in the C dimension.
        int c = this->c_;

        // The scale.
        const char *scale_ptr = reinterpret_cast<const char *>(params.bn_scale_gmem);
        scale_ptr_ = &scale_ptr[c * sizeof(uint16_t)];
        scale_ = make_uint4(0, 0, 0, 0);
        if (c < params.g * params.c) {
            xmma::ldg(scale_, scale_ptr_);
        }

        // The bias.
        const char *bias_ptr = reinterpret_cast<const char *>(params.bn_bias_gmem);
        bias_ptr_ = &bias_ptr[c * sizeof(uint16_t)];

        bias_ = make_uint4(0, 0, 0, 0);
        if (c < params.g * params.c) {
            xmma::ldg(bias_, bias_ptr_);
        }
    }

    // Store the pixels to shared memory.
    template <typename Xmma_smem_tile> inline __device__ void commit(Xmma_smem_tile &smem) {

        // Scale and add the bias.
        xmma::scale_bias(reinterpret_cast<uint4(&)[Base::LDGS]>(this->fetch_), scale_, bias_,
                             this->preds_[0]);

        // Apply RELU. if needed
        if ( WITH_RELU ) {
            #pragma unroll
            for (int mi = 0; mi < Base::LDGS; ++mi) {
                this->fetch_[mi].x = xmma::relu_fp16x2(this->fetch_[mi].x);
                this->fetch_[mi].y = xmma::relu_fp16x2(this->fetch_[mi].y);
                this->fetch_[mi].z = xmma::relu_fp16x2(this->fetch_[mi].z);
                this->fetch_[mi].w = xmma::relu_fp16x2(this->fetch_[mi].w);
            }
        }

        // Store to shared memory.
        Base::commit(smem);
    }

    // The pointers for scale and bias.
    const char *scale_ptr_, *bias_ptr_;

    // The bias and scale.
    uint4 scale_, bias_;
};

template <typename Traits, typename Cta_tile, bool SIMPLE_1x1x1, bool WITH_RELU>
struct Gmem_tile_b_volta_turing
    : public Gmem_tile_b_base<Traits, Cta_tile, SIMPLE_1x1x1, WITH_RELU> {
};

template <typename Cta_tile, bool SIMPLE_1x1x1, bool WITH_RELU>
struct Gmem_tile_b_volta_turing<xmma::Volta_hmma_fp32_traits, Cta_tile, SIMPLE_1x1x1, WITH_RELU>
    : public Gmem_tile_b_base<xmma::Volta_hmma_fp32_traits, Cta_tile, SIMPLE_1x1x1, WITH_RELU> {

    // The base class.
    using Base_gmem = Gmem_tile_b_base<xmma::Volta_hmma_fp32_traits, Cta_tile, SIMPLE_1x1x1,
                                      WITH_RELU>;

    // Ctor.
    template <typename Params>
    inline __device__ Gmem_tile_b_volta_turing(const Params &params, void *smem, const dim3 &bidx,
                                               int tidx)
        : Base_gmem(params, smem, bidx, tidx) {}
};

////////////////////////////////////////////////////////////////////////////////////////////////////
} // namespace bn_apply
} // namespace batchnorm
} // namespace ext
} // namespace xmma
////////////////////////////////////////////////////////////////////////////////////////////////////
