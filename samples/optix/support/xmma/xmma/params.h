/***************************************************************************************************
 * Copyright (c) 2019, LWPU CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted
 * provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright notice, this list of
 *       conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright notice, this list of
 *       conditions and the following disclaimer in the documentation and/or other materials
 *       provided with the distribution.
 *     * Neither the name of the LWPU CORPORATION nor the names of its contributors may be used
 *       to endorse or promote products derived from this software without specific prior written
 *       permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL LWPU CORPORATION BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROLWREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TOR (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

#pragma once

#include <xmma/xmma.h>

#include <xmma/integer.h>
#include <xmma/numeric_types.h>

#include <xmma/hopper/emu/lwda_tma_types.h>

#if !defined(__LWDACC_RTC__)
#include <lwca.h>
#endif

namespace xmma {

///////////////////////////////////////////////////////////////////////////////////////////////////

enum class Colwolution_layout {
    NCHW = 0,
    NCHW_VECT_C_32,
    NCHW_VECT_C_16,
    NCHW_VECT_C_8,
    NHWC,
};

///////////////////////////////////////////////////////////////////////////////////////////////////

enum class Operation_type {
    GEMM = 0,
    FPROP,
    DGRAD,
    WGRAD,
    STRIDED_DGRAD,
};

///////////////////////////////////////////////////////////////////////////////////////////////////

enum class Colwolution_algorithm {
    PRECOMPUTED = 0,
    INDEX,
    TWOD_TILING,
    WINOGRAD,
};

///////////////////////////////////////////////////////////////////////////////////////////////////

// Ampere mempry descriptors
struct Memory_descriptors {
    uint64_t descriptor_a;
    uint64_t descriptor_b;
    uint64_t descriptor_c;
    uint64_t descriptor_d;
};

///////////////////////////////////////////////////////////////////////////////////////////////////

struct Runtime_reserved_params {
    //float gelu_scale;
    int32_t runtime_param0;
    int32_t runtime_param1;
    int32_t runtime_param2;
    int32_t runtime_param3;
    int32_t runtime_param4;
    int32_t runtime_param5;
};

///////////////////////////////////////////////////////////////////////////////////////////////////

struct Batched_gemm_params {
    // The enablement of batched GEMM
    bool is_batched;
    // The number of batches
    int32_t batches;
    // The memory layout is contiguous or pointer-based
    bool contiguous;
    // alpha/beta scaling per batch sample
    bool batch_scaling;
    // enables batched bias vector
    bool batch_bias;
};

///////////////////////////////////////////////////////////////////////////////////////////////////

struct Split_k_params {
    enum { ALIGNMENT = 16 };

    // Split-k buffers.
    int32_t slices, buffers, kernels;

    // The size of a single buffer in bytes.
    int64_t buffer_size;
    // The buffer to store the data.
    void* buffers_gmem;

    // The size of the counter CTAs buffer.
    int32_t counters_ctas_size;
    // The size of the retired CTAs buffer.
    int32_t retired_ctas_size;

    // The buffer to keep the counters (one per buffer + one in total).
    int32_t* counters_gmem;
    // The buffer to keep the number of retired CTAs per tile.
    int32_t* retired_ctas_gmem;

    XMMA_HOST int64_t size_in_bytes() const {
        int64_t size = buffer_size * buffers + counters_ctas_size + retired_ctas_size;
        if (size != 0) { size += ALIGNMENT; }
        return size;
    }

    XMMA_HOST bool with_reduction() const { return this->slices > 1 && (this->slices > buffers || this->kernels == 1); }

    XMMA_HOST void *buffers_data(void *base) const {
        const auto aligned_base = xmma::ptr_to_int64(base) + ALIGNMENT - xmma::ptr_to_int64(base) % ALIGNMENT;
        return (void*)aligned_base;
    }

    XMMA_HOST int32_t *counters_data(void *base) const {
        return (int32_t *)(xmma::ptr_to_int64(buffers_data(base)) + buffer_size * buffers);
    }

    XMMA_HOST int32_t *retired_data(void *base) const {
        return (int32_t *)(xmma::ptr_to_int64(counters_data(base)) + counters_ctas_size);
    }

    XMMA_HOST void set_base_ptr(void *base) {
        if(nullptr != base) {
            buffers_gmem = this->buffers_data(base);
            counters_gmem = this->counters_data(base);
            retired_ctas_gmem = this->retired_data(base);
        }
    }

    XMMA_HOST void set_params(int32_t slices_, int32_t buffers_, int32_t kernels_) {
        slices = slices_;
        buffers = buffers_;
        kernels = kernels_;

        buffer_size = 0;
        buffers_gmem = nullptr;
        counters_ctas_size = 0;
        retired_ctas_size = 0;
        counters_gmem = nullptr;
        retired_ctas_gmem = nullptr;
    }

#if !defined(__LWDACC_RTC__)
    lwdaError_t clear_buffers(void *base, lwdaStream_t stream) const {
        if (this->with_reduction()) {
            return lwdaMemsetAsync(this->counters_data(base), 0, counters_ctas_size + retired_ctas_size, stream);
        }

        return lwdaSuccess;
    }
#endif
};

// To call set_base_ptr from a void* which points to a Split_k_params struct.
// This avoids introducing Split_k_params to KernelRunner.
XMMA_HOST void split_k_params_set_base_ptr(void* params, void* base)
{
    if(!params)
        return;
    Split_k_params* split_k = reinterpret_cast<Split_k_params*>(params);
    split_k->set_base_ptr(base);
}

// To call clear_buffers from a void* which points to a Split_k_params struct.
// This avoids introducing Split_k_params to KernelRunner.
XMMA_HOST lwdaError_t split_k_params_clear_buffers(void* params, void *base, lwdaStream_t stream)
{
    if(!params)
        return lwdaSuccess;
    Split_k_params* split_k = reinterpret_cast<Split_k_params*>(params);
    lwdaError_t result = lwdaSuccess;
#if !defined(__LWDACC_RTC__)
    result = split_k->clear_buffers(base, stream); 
#endif
    return result;
}

// To set "kernels" from a void* which points to a Split_k_params struct.
// This avoids introducing Split_k_params to KernelRunner.
XMMA_HOST void split_k_params_set_kernels(void* params, int kernels)
{
    if(!params)
        return;
    Split_k_params* split_k = reinterpret_cast<Split_k_params*>(params);
    split_k->kernels = kernels;
}

// To get "kernels" from a void* which points to a Split_k_params struct.
// This avoids introducing Split_k_params to KernelRunner.
XMMA_HOST int split_k_params_get_kernels(void* params)
{
    if(!params)
        return 0;
    Split_k_params* split_k = reinterpret_cast<Split_k_params*>(params);
    return split_k->kernels;
}

struct Split_k_atomic_params {
    // The atomic buffer to keep the index of the CTA that is initializing output buffer
    int32_t* initializer_id;

    // The atomic counter for no. of CTAs that finished initialization.
    int32_t* initializer_done;
};

///////////////////////////////////////////////////////////////////////////////////////////////////

struct Colwolution_params_base {
    enum { DEPTH = 0, HEIGHT, WIDTH, MAX_DIMENSION };

    const void* img_gmem;
    const void* flt_gmem;
    void* out_gmem;
    // Residual data.
    const void* res_gmem;
    const void* bias_gmem;
    const void *alpha_gmem, *beta_gmem;
    // The dimensions of the layer.
    int32_t n, d, h, w, c, o, p, q, k, t, r, s, g;
    int32_t filter_t_per_cta, filter_r_per_cta, filter_s_per_cta;
    // Epilogue scaling. Colwerted to epilogue type in the kernel.
    double alpha, beta;
    // ReLu.
    int32_t with_relu;
    float relu_lb, relu_ub;
    // Tensor strides.
    uint32_t img_stride_n, img_stride_d, img_stride_h, img_stride_w, img_stride_c;
    uint32_t out_stride_n, out_stride_d, out_stride_h, out_stride_w, out_stride_c;
    // The parameters of the colwolution.
    int32_t pad[MAX_DIMENSION][2];
    int32_t stride[MAX_DIMENSION];
    int32_t dilation[MAX_DIMENSION];

    bool cross_correlation;

    Split_k_params split_k;

    Split_k_atomic_params split_k_atomic;
    // L2 descriptors
    Memory_descriptors mem_descriptors;
    // Reserved runtime parameter
    Runtime_reserved_params runtime_params;
    // Number of ctas per wave
    unsigned ctas_per_wave;

    bool per_channel_scaling;
    // Do we enable bias? If we do, with_bias contains the number of bias elements.
    int32_t with_bias;
    // Interleaved kernels
    bool is_interleaved;
    // The batched GEMM params, unused in Colwolution
    Batched_gemm_params batch;
    float one;

    // For Hopper, we also have cluster dimensions as a part of the params
    // This can be setup at runtime or via compile-time (Traits)
    // If setup at runtime-time - it will match Compile-Time value (in Traits)
    uint32_t cluster_height, cluster_width;

    // Ctor
    Colwolution_params_base() {}

#if !defined(__LWDACC_RTC__)

    // Set temsor size.
    XMMA_HOST void set_tensor_desc(int32_t n_, int32_t d_, int32_t h_, int32_t w_, int32_t c_,
                                   const std::array<uint32_t, 5> &img_strides_,
                                   int32_t o_, int32_t p_, int32_t q_, int32_t k_,
                                   const std::array<uint32_t, 5> &out_strides_,
                                   int32_t t_, int32_t r_, int32_t s_, int32_t g_) {
        this->n = n_; this->d = d_; this->h = h_; this->w = w_; this->c = c_;
        this->img_stride_n = img_strides_[0];
        this->img_stride_d = img_strides_[1];
        this->img_stride_h = img_strides_[2];
        this->img_stride_w = img_strides_[3];
        this->img_stride_c = img_strides_[4];
        this->o = o_; this->p = p_; this->q = q_; this->k = k_;
        this->out_stride_n = out_strides_[0];
        this->out_stride_d = out_strides_[1];
        this->out_stride_h = out_strides_[2];
        this->out_stride_w = out_strides_[3];
        this->out_stride_c = out_strides_[4];
        this->t = t_; this->r = r_; this->s = s_; this->g = g_;
    }

    // Set paddings.
    XMMA_HOST void set_padding(const std::array<std::array<int32_t, 2>, MAX_DIMENSION> &paddings_) {
        std::copy(paddings_[0].begin(), paddings_[0].end(), std::begin(this->pad[0]));
        std::copy(paddings_[1].begin(), paddings_[1].end(), std::begin(this->pad[1]));
        std::copy(paddings_[2].begin(), paddings_[2].end(), std::begin(this->pad[2]));
    }

    // Set strides.
    XMMA_HOST void set_strides(const std::array<int32_t, MAX_DIMENSION> &strides_) {
        std::copy(strides_.begin(), strides_.end(), std::begin(this->stride));
    }

    // Set dilations.
    XMMA_HOST void set_dilation(const std::array<int32_t, MAX_DIMENSION> &dilation_) {
        std::copy(dilation_.begin(), dilation_.end(), std::begin(this->dilation));
    }

    // Set relu data.
    XMMA_HOST void set_relu(bool apply_relu_, float relu_upper_bound_) {
        this->relu_lb = -std::numeric_limits<float>::infinity();
        this->relu_ub = std::numeric_limits<float>::infinity();
        this->with_relu = apply_relu_;
        if (apply_relu_) {
            this->relu_lb = 0.f;
            if (relu_upper_bound_ > 0) { this->relu_ub = relu_upper_bound_; }
        }
    }

    // Set split k params.
    XMMA_HOST void set_split_k(int32_t slices_, int32_t buffers_, int32_t kernels_) {
        this->split_k.set_params(slices_, buffers_, kernels_);
    }

    // Interface to set all params.
    XMMA_HOST void set_problem(
        int32_t n_, int32_t d_, int32_t h_, int32_t w_, int32_t c_,
        const std::array<uint32_t, 5> &img_strides_,
        int32_t o_, int32_t p_, int32_t q_, int32_t k_,
        const std::array<uint32_t, 5> &out_strides_,
        int32_t t_, int32_t r_, int32_t s_, int32_t g_,
        double alpha_, double beta_,
        bool cross_correlation_, bool is_interleaved_, uint32_t ctas_per_wave_,
        int32_t with_bias_, bool per_channel_scaling_,
        const std::array<std::array<int32_t, 2>, MAX_DIMENSION> &paddings_,
        const std::array<int32_t, MAX_DIMENSION> &strides_,
        const std::array<int32_t, MAX_DIMENSION> &dilation_,
        bool apply_relu_, float relu_upper_bound_) {

        set_tensor_desc(n_, d_, h_, w_, c_, img_strides_,
                        o_, p_, q_, k_, out_strides_,
                        t_, r_, s_, g_);

        this->alpha = alpha_ ; this->beta = beta_;

        this->cross_correlation = cross_correlation_;
        this->is_interleaved = is_interleaved_;
        this->ctas_per_wave = ctas_per_wave_;
        this->per_channel_scaling = per_channel_scaling_;

        if (with_bias_) {
            this->with_bias = this->k;
        } else {
            this->with_bias = 0;
        }

        memset(&(this->runtime_params), 0, sizeof(this->runtime_params));

        memset(&(this->batch), 0, sizeof(this->batch));

        set_padding(paddings_);
        set_strides(strides_);
        set_dilation(dilation_);

        set_relu(apply_relu_, relu_upper_bound_);
    }

    XMMA_HOST void print() {
        printf("g=%d n=%d d=%d h=%d, w=%d, c=%d, k=%d, t=%d r=%d, s=%d "
               "(o=%d, p=%d, q=%d), alpha=%0.1f, beta=%0.1f, pad=%d, %d, %d "
               "stride=[%d, %d, %d], dilation=[%d, %d, %d]\n",
                g, n, d, h, w, c, k, t, r, s, o, p, q,
                xmma::colwert<float>( alpha ), xmma::colwert<float>( beta ),
                pad[0][0], pad[1][0], pad[2][0],
                stride[0], stride[1], stride[2],
                dilation[0], dilation[1], dilation[2] );
    }
#endif
};

///////////////////////////////////////////////////////////////////////////////////////////////////

struct Gemm_params_base {
    // The A matrix.
    const void* a_gmem;
    // The B matrix.
    const void* b_gmem;
    // The C matrix.
    const void* c_gmem;
    // The Bias vector.
    const void* bias_gmem;
    // The per-batch scaling.
    const void *alpha_gmem, *beta_gmem;
    // The D matrix (output).
    void* d_gmem;

    // FIXME: @ftse I think those shouldn't be put into Gemm_param_base
    // A matrix TMA descriptor
    lwdaTmaDescv2 *a_desc;
    // B matrix TMA descriptor
    lwdaTmaDescv2 *b_desc;
    // C matrix TMA descriptor
    lwdaTmaDescv2 *c_desc;
    // Bias vector TMA descriptor
    lwdaTmaDescv2 *bias_desc;
    // D matrix TMA descriptor
    lwdaTmaDescv2 *d_desc;

    // For Hopper, we also have cluster dimensions as a part of the params
    // This can be setup at runtime or via compile-time (Traits)
    // If setup at runtime-time - it will match Compile-Time value (in Traits)
    uint32_t cluster_height, cluster_width;

    // The dimensions of the product.
    int32_t m, n, k;
    // The strides.
    int32_t lda, ldb, ldc, ldd;
    uint32_t a_stride_rows, a_stride_cols, a_stride_batches;
    uint32_t b_stride_rows, b_stride_cols, b_stride_batches;
    uint32_t c_stride_rows, c_stride_cols, c_stride_batches;
    // Are the matrices transposed?
    int32_t ta, tb;
    // The split-k params.
    Split_k_params split_k;
    // The split-k-atomic params.
    Split_k_atomic_params split_k_atomic;
    // L2 descriptors
    Memory_descriptors mem_descriptors;
    // Reserved runtime parameter
    Runtime_reserved_params runtime_params;
    // Number of ctas per wave
    uint32_t ctas_per_wave;
    // Interleaved kernels. TODO: Do we need that???
    bool is_interleaved;
    // Is there a residual to be added? I.e. is beta != 0?
    bool with_residual;
    // Alpha/beta.
    double alpha, beta;
    // Do we enable bias? If we do, with_bias contains the number of bias elements.
    int32_t with_bias;
    // ReLu.
    bool with_relu;
    float relu_lb, relu_ub;
    bool per_channel_scaling;

    // The batched GEMM params
    Batched_gemm_params batch;

    // For fusing HADD2+relu
    float one;

#if !defined(__LWDACC_RTC__)
    XMMA_HOST Gemm_params_base() {}

    XMMA_HOST void set_problem( int32_t m_,
                                int32_t n_,
                                int32_t k_,
                                int32_t batches_,
                                const std::array<uint32_t, 3> &a_strides_,
                                const std::array<uint32_t, 3> &b_strides_,
                                const std::array<uint32_t, 3> &c_strides_,
                                bool ta_,
                                bool tb_,
                                bool is_interleaved_,
                                uint32_t ctas_per_wave_,
                                double alpha_,
                                double beta_,
                                bool apply_relu_,
                                float relu_ub_,
                                bool with_bias_,
                                bool batch_bias_,
                                bool batch_scaling_,
                                bool per_channel_scaling_ ) {
        this->m = m_; this->n = n_; this->k = k_;
        this->ta = ta_; this->tb = tb_;
        this->alpha = alpha_; this->beta = beta_;
        this->with_relu = apply_relu_; this->relu_ub = relu_ub_;
        this->is_interleaved = is_interleaved_;
        this->ctas_per_wave = ctas_per_wave_;
        this->per_channel_scaling = per_channel_scaling_;
        // FIXME: this one is confusing, need to rename with_bias to better name
        this->with_bias = with_bias_ ? n : 0;

        memset(&runtime_params, 0, sizeof(runtime_params));

        // Set tensor strides.
        this->a_stride_batches = a_strides_[0];
        this->a_stride_rows = a_strides_[1];
        this->a_stride_cols = a_strides_[2];

        this->b_stride_batches = b_strides_[0];
        this->b_stride_rows = b_strides_[1];
        this->b_stride_cols = b_strides_[2];

        this->c_stride_batches = c_strides_[0];
        this->c_stride_rows = c_strides_[1];
        this->c_stride_cols = c_strides_[2];

        // FIXME: ld{a,b,c,d} must be set using stride? Is ldx still needed?
        this->lda = this->ta ? this->k : this->m;
        this->ldb = this->tb ? this->n : this->k;
        this->ldc = this->n;
        this->ldd = this->n;

        this->batch.batches = batches_;
        if( this->batch.batches > 1 ) {
            this->batch.is_batched = true;
            this->batch.contiguous = true;
            this->batch.batch_scaling = batch_scaling_;
            this->batch.batch_bias = batch_bias_;
        } else {
            this->batch.is_batched = false;
            this->batch.contiguous = false;
            this->batch.batch_scaling = false;
            this->batch.batch_bias = false;
        }

        this->relu_lb = -std::numeric_limits<float>::infinity();
        this->relu_ub = std::numeric_limits<float>::infinity();

        if (apply_relu_) {
            this->relu_lb = 0.f;
            if (relu_ub_ > 0) { relu_ub = relu_ub_; }
        }

        // Ported from: https://gitlab-master.lwpu.com/dlarch-fastkernels/cask_sdk/-/commit/7a66883d95508fb36b175ee2d1c99567c30b1bee#4dca468baec106637496557ecc6606bcc55ff81f_117_118
        this->one = 1.f;
    }

    XMMA_HOST void set_split_k(int32_t slices_, int32_t buffers_, int32_t kernels_) {
        this->split_k.set_params(slices_, buffers_, kernels_);
    }

    XMMA_HOST void print() {
        printf("m=%d n=%d k=%d lda=%d, ldb=%d, ldc=%d, ldd=%d, "
               "ta=%d tb=%d, split_k_slices=%d, alpha=%0.1f, beta=%0.1f\n",
               m, n, k, lda, ldb, ldc, ldd, ta, tb, split_k.slices,
               xmma::colwert<float>(alpha), xmma::colwert<float>(beta));
    }
#endif
};

///////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace xmma
