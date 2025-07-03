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

#include <xmma/xmma.h>

namespace xmma {
namespace ext {
namespace helpers {

template<
    typename Traits,
    typename Cta_tile,
    int ELTWISE,
    typename Fragment_pre_swizzle_ = xmma::Fragment_epilogue_pre_swizzle<Traits, Cta_tile>,
    typename Fragment_post_swizzle_ = xmma::Fragment_epilogue_post_swizzle<Traits, Cta_tile, true>,
    typename Fragment_c_ = xmma::Fragment_c<Traits, Cta_tile>
>
struct Callbacks_epilogue
    : public xmma::helpers::Empty_callbacks_epilogue<Traits,
                                                         Cta_tile,
                                                         Fragment_pre_swizzle_,
                                                         Fragment_post_swizzle_,
                                                         Fragment_c_> {
    // The base class.
    using Base = xmma::helpers::Empty_callbacks_epilogue<Traits,
                                                             Cta_tile,
                                                             Fragment_pre_swizzle_,
                                                             Fragment_post_swizzle_,
                                                             Fragment_c_>;

    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;

    enum { ELEMENTS_PER_LDG = 8 };

    // The different fragments.
    using Fragment_post_swizzle = Fragment_post_swizzle_;
    using Fragment_pre_swizzle = Fragment_pre_swizzle_;
    using Fragment_c = Fragment_c_;
    using Fragment_aclwmulator = xmma::Fragment_aclwmulator<Traits>;
    using Fragment_bias = xmma::Fragment<float, ELEMENTS_PER_LDG>;
    using C_type = typename Traits::C_type;

    template< typename Params >
    inline __device__ Callbacks_epilogue(const Params &params,
                                                   char *smem,
                                                   int bidm,
                                                   int bidn,
                                                   int bidz,
                                                   int tidx)
        : Base(params, smem, bidm, bidn, bidz, tidx)
        , relu_lb_(params.relu_lb)
        , relu_ub_(params.relu_ub)
        , params_m_(params.k)
        , with_bias_(params.with_bias)
        , with_residual_(params.with_residual)
        , bias_ptr_(reinterpret_cast<const char*>(params.bias_gmem)) {

        alpha_ = xmma::colwert<typename Traits::Epilogue_type>(params.alpha);
        beta_  = xmma::colwert<typename Traits::Epilogue_type>(params.beta);
        alpha_ptr_ = reinterpret_cast<const char*>(params.alpha_gmem);
        beta_ptr_ = reinterpret_cast<const char*>(params.beta_gmem);
        per_channel_ = params.per_channel_scaling;

        // The number of warps.
        const int WARPS_M = Cta_tile::WARPS_M;
        const int WARPS_N = Cta_tile::WARPS_N;
        const int WARPS_K = Cta_tile::WARPS_K;

        // The masks to select the warps.
        const int WARP_MASK_M = xmma::Warp_masks<WARPS_M, WARPS_N, WARPS_K>::M;

        // The divisor for the warps.
        const int WARP_DIV_M =       1 *       1 * Cta_tile::THREADS_PER_WARP;

        // The location of the tile.
        const int row =
            ((tidx & WARP_MASK_M) / WARP_DIV_M) * Xmma_tile::M_PER_WARP
           + ((tidx % Cta_tile::THREADS_PER_WARP)/4) * ELEMENTS_PER_LDG;

        int k, g;
        xmma::fast_divmod( g, k, bidm, params.kn, params.mul_k, params.shr_k);

        // Compute the output position for each thread.
        bias_m_ = k * Cta_tile::M + row;

        // The pointer.
        if (Traits::BITS_PER_ELEMENT_C == 8 || Traits::BITS_PER_ELEMENT_C == 32) {
            bias_ptr_ += (bias_m_ + g * params.k) * sizeof(float);
            alpha_ptr_ += (bias_m_ + g * params.k) * sizeof(float);
            beta_ptr_ += (bias_m_ + g * params.k) * sizeof(float);
        } else {
            bias_ptr_ += Traits::offset_in_bytes_c(bias_m_ + g * params.k);
            alpha_ptr_ += (bias_m_ + g * params.k) * sizeof(float);
            beta_ptr_ += (bias_m_ + g * params.k) * sizeof(float);
        }

        if (ELTWISE == xmma::GELU) {
            memcpy(&gelu_scale_, &params.runtime_params.runtime_param0, sizeof(float));
            literal0 = 0.044715f * 0.797885f;
            literal1 = 0.797885f;
            literal2 = 0.500000f * gelu_scale_;
        }
    }

    template <int M, int N>
    inline __device__ void colwert(
        Fragment_aclwmulator acc[M][N],
        Fragment_post_swizzle &frag,
        int mi, int i) {
        frag.elt(0) = acc[0][mi/2].elt(0+i+(mi%2)*4);
        frag.elt(1) = acc[0][mi/2].elt(2+i+(mi%2)*4);
        frag.elt(2) = acc[1][mi/2].elt(0+i+(mi%2)*4);
        frag.elt(3) = acc[1][mi/2].elt(2+i+(mi%2)*4);
        frag.elt(4) = acc[2][mi/2].elt(0+i+(mi%2)*4);
        frag.elt(5) = acc[2][mi/2].elt(2+i+(mi%2)*4);
        frag.elt(6) = acc[3][mi/2].elt(0+i+(mi%2)*4);
        frag.elt(7) = acc[3][mi/2].elt(2+i+(mi%2)*4);
    }

    inline __device__ void load_alpha(Fragment_post_swizzle &data) {
        if (Traits::BITS_PER_ELEMENT_C == 8 && per_channel_) {
        if (bias_m_ < params_m_ ) {
            uint4 tmp;
            xmma::ldg(tmp, alpha_ptr_);
            data.reg(0) = tmp.x;
            data.reg(1) = tmp.y;
            data.reg(2) = tmp.z;
            data.reg(3) = tmp.w;

            if (Traits::BITS_PER_ELEMENT_C == 8) {
                xmma::ldg(tmp, alpha_ptr_ + sizeof(float) * 4);
                data.reg(4) = tmp.x;
                data.reg(5) = tmp.y;
                data.reg(6) = tmp.z;
                data.reg(7) = tmp.w;
            }
        }
        } else {
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                data.elt(i) = alpha_;
            }
        }
    }

    inline __device__ void load_beta(Fragment_post_swizzle &data) {
        if (Traits::BITS_PER_ELEMENT_C == 8 && per_channel_) {
        if (bias_m_ < params_m_ ) {
            uint4 tmp;
            xmma::ldg(tmp, beta_ptr_);
            data.reg(0) = tmp.x;
            data.reg(1) = tmp.y;
            data.reg(2) = tmp.z;
            data.reg(3) = tmp.w;

            if (Traits::BITS_PER_ELEMENT_C == 8) {
                xmma::ldg(tmp, beta_ptr_ + sizeof(float) * 4);
                data.reg(4) = tmp.x;
                data.reg(5) = tmp.y;
                data.reg(6) = tmp.z;
                data.reg(7) = tmp.w;
            }
        }
        } else {
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                data.elt(i) = beta_;
            }
        }
    }

    inline __device__ void load_bias(Fragment_post_swizzle &data) {
        if (with_bias_ && bias_m_ < params_m_ ) {
            uint4 tmp;
            xmma::ldg(tmp, bias_ptr_);
            data.reg(0) = tmp.x;
            data.reg(1) = tmp.y;
            data.reg(2) = tmp.z;
            data.reg(3) = tmp.w;

            if (Traits::BITS_PER_ELEMENT_C == 8 || Traits::BITS_PER_ELEMENT_C == 32) {
                xmma::ldg(tmp, bias_ptr_ + sizeof(float) * 4);
                data.reg(4) = tmp.x;
                data.reg(5) = tmp.y;
                data.reg(6) = tmp.z;
                data.reg(7) = tmp.w;
            }

        }
    }

    inline __device__ void do_relu_fp32(Fragment_c &frag) {
        #pragma unroll
        for( int i = 0; i < Fragment_c::NUM_REGS; ++i ) {
            frag.elt(i) = xmma::relu_fp32(frag.elt(i), relu_lb_);
            frag.elt(i) = xmma::relu_ub_fp32(frag.elt(i), relu_ub_);
        }
    }

    inline __device__ void do_relu_fp32_(Fragment_post_swizzle &frag) {
        if (ELTWISE == xmma::RELU) {
        #pragma unroll
        for( int i = 0; i < Fragment_post_swizzle::NUM_REGS; ++i ) {
            frag.elt(i) = xmma::relu_fp32(frag.elt(i), relu_lb_);
        }
        }
    }

    inline __device__ void do_relu(Fragment_c &frag) {
        // FP16x2 ReLu.
        half2 tmp = xmma::colwert<half2>(relu_lb_);
        uint32_t relu_lb_v2 = reinterpret_cast<uint32_t&>(tmp);

        tmp = xmma::colwert<half2>(relu_ub_);
        uint32_t relu_ub_v2 = reinterpret_cast<uint32_t&>(tmp);

        #pragma unroll
        for( int i = 0; i < Fragment_c::NUM_REGS; ++i ) {
            frag.reg(i) = xmma::reluTh_fp16x2(frag.reg(i), relu_lb_v2);
            frag.reg(i) = xmma::relu_ub_fp16x2(frag.reg(i), relu_ub_v2);
        }
    }

    inline __device__ void do_gelu(Fragment_post_swizzle &frag) {

        if (ELTWISE == xmma::GELU) {
            // origin: 0.5f * x * (1.0f + tanh(0.797885f * (x + 0.044715f * x * x * x)));
            // new: 0.5f * (x + x * tanh(x * (0.797885f + 0.0356774f * x * x)));
            // reduce two op
            //constexpr auto literal0 = 0.044715f * 0.797885f;
            //constexpr auto literal1 = 0.797885f;
            //constexpr auto literal2 = 0.500000f;
            //#pragma unroll
            //for( int i = 0; i < ELEMENTS_PER_LDG; ++i ) {
            //    frag.elt(i) = xmma::gelu(frag.elt(i), literal0, literal1, literal2, gelu_scale_);
            //}
            #pragma unroll
            for( int i = 0; i < ELEMENTS_PER_LDG; ++i ) {
                float ele = frag.elt(i);
                float v0 = literal0 * ele;
                float v1 = __fmaf_rn(ele, v0, literal1);
                float v2 = v1       * ele;
                float v3;
                asm volatile ("tanh.approx.f32 %0, %1;" : "=f"(v3) : "f"(v2));
                float v4 = __fmaf_rn(ele, v3, ele);
                frag.elt(i) = literal2 * v4;
            }
        }
    }

    inline __device__ void do_relu_s8(Fragment_c &frag) {
        if (ELTWISE == xmma::RELU) {
            // s8x4 ReLu.
            int8_t tmp = relu_lb_;
            uint32_t relu_lb = reinterpret_cast<uint32_t&>(tmp);
            #pragma unroll
            for( int i = 0; i < ELEMENTS_PER_LDG/4; ++i ) {
                frag.reg(i) = xmma::relu_s8x4(frag.reg(i), relu_lb);
            }
        }
    }

    float relu_lb_, relu_ub_;
    typename Traits::Epilogue_type alpha_, beta_;
    const char *bias_ptr_, *alpha_ptr_, *beta_ptr_;
    const int params_m_;
    bool with_bias_, with_residual_, per_channel_;
    int bias_m_;
    float gelu_scale_;
    float literal0, literal1, literal2;
};

///////////////////////////////////////////////////////////////////////////////////////////////////

template<
    // The instruction traits.
    typename Traits_,
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile_,
    // The layout of the tile.
    typename Layout_,
    // The global memory tile to store the output.
    typename Gmem_tile_,
    // Elementwise operation
    int ELTWISE,
    // The callbacks to lwstomize the behaviour of the epilogue.
    typename Callbacks = xmma::helpers::Empty_callbacks_epilogue<Traits_, Cta_tile_>,
    // The class to swizzle the data.
    typename Swizzle_ = xmma::Swizzle_epilogue<Traits_, Cta_tile_, Layout_>,
    // The fragment class before the swizzling.
    typename Fragment_pre_swizzle_ = typename Callbacks::Fragment_pre_swizzle,
    // The fragment class after the swizzling.
    typename Fragment_post_swizzle_ = typename Callbacks::Fragment_post_swizzle,
    // The output fragment.
    typename Fragment_c_ = typename Callbacks::Fragment_c
>
struct Epilogue {
    // The instruction traits.
    using Traits = Traits_;
    // The dimensions of the tile computed by the CTA.
    using Cta_tile = Cta_tile_;
    // The layout of the tile.
    using Layout = Layout_;
    // The global memory tile to store the output.
    using Gmem_tile = Gmem_tile_;
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
    // The fragment for bias.
    using Fragment_bias = typename Callbacks::Fragment_bias;

    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;

    // Ctor.
    template< typename Params >
    inline __device__ Epilogue(const Params &params,
                               Gmem_tile &gmem_tile,
                               Swizzle &swizzle,
                               Callbacks &callbacks)
        : gmem_tile_(gmem_tile)
        , swizzle_(swizzle)
        , callbacks_(callbacks) {
    }

    template<bool WITH_RESIDUAL, typename Fragment_aclwmulator, int M, int N >
    inline __device__ void execute(Fragment_aclwmulator (&acc)[M][N], Fragment_post_swizzle &bias_regs) {

        Fragment_post_swizzle alpha_regs, beta_regs;

        if (Traits::BITS_PER_ELEMENT_C == 8) {
            if (WITH_RESIDUAL) {
            callbacks_.load_beta(beta_regs);
            }
            callbacks_.load_alpha(alpha_regs);
        }

        #pragma unroll
        for (int mi = 0; mi < 2*N; mi++) {
            if (ELTWISE == xmma::RELU) {
                Fragment_c out_regs[Gmem_tile::STGS];
                Fragment_post_swizzle post_swizzle[Gmem_tile::STGS];

                Fragment_c tmp_residual[Gmem_tile::STGS];
                if(WITH_RESIDUAL) {
                    // Do beta
                    //if (callbacks_.with_residual_) {
                        #pragma unroll
                        for (int i = 0; i < Gmem_tile::STGS; i++) {
                            gmem_tile_.load(tmp_residual[i], mi, i);
                        }
                    //}
                }
                //jetfire::ifence(1);

                #pragma unroll
                for (int i = 0; i < Gmem_tile::STGS; i++) {
                    callbacks_.template colwert<M,N>(acc, post_swizzle[i], mi, i);

                    // Do alpha
                    post_swizzle[i].scale(callbacks_.alpha_, alpha_regs);

                    //if(WITH_RESIDUAL) {
                    //// Do beta
                    //    Fragment_c tmp;
                    //    gmem_tile_.load(tmp, mi, i);
                    //    post_swizzle[i].residual(callbacks_.beta_, tmp, beta_regs);
                    //}

                    // Add bias
                    if (callbacks_.with_bias_) {
                        post_swizzle[i].add_bias(bias_regs);
                    }

                    if(WITH_RESIDUAL) {
                        post_swizzle[i].residual(callbacks_.beta_, tmp_residual[i], beta_regs);
                    }

                    // Do Gelu
                    // callbacks_.do_gelu(post_swizzle[i]);
                    if (Traits::BITS_PER_ELEMENT_C == 8) {
                        callbacks_.do_relu_fp32_(post_swizzle[i]);
                    }

                    // Colwert aclwmulator to ouput (float to half )
                    out_regs[i].pack_sparse(1.0, post_swizzle[i]);

                    // Do Relu
                    if (Traits::BITS_PER_ELEMENT_C == 8) {
                    } else if (Traits::BITS_PER_ELEMENT_C == 32){
                        callbacks_.do_relu_fp32(out_regs[i]);
                    } else {
                        callbacks_.do_relu(out_regs[i]);
                    }

                    // Store results
                    gmem_tile_.store(mi, i, out_regs[i]);
                    // jetfire::ifence(1);
                }
            } else {
                Fragment_c tmp_residual[Gmem_tile::STGS];
                if(WITH_RESIDUAL) {
                    // Do beta
                    if (callbacks_.with_residual_) {
                        #pragma unroll
                        for (int i = 0; i < Gmem_tile::STGS; i++) {
                            gmem_tile_.load(tmp_residual[i], mi, i);
                        }
                    }
                }

                /* Leave this part for now. Just in case. */
                Fragment_post_swizzle post_swizzle[Gmem_tile::STGS];
                #pragma unroll
                for (int i = 0; i < Gmem_tile::STGS; i++) {
                    callbacks_.template colwert<M,N>(acc, post_swizzle[i], mi, i);
                }

                // Do alpha
                #pragma unroll
                for (int i = 0; i < Gmem_tile::STGS; i++) {
                    post_swizzle[i].scale(callbacks_.alpha_, alpha_regs);
                }

                //if(WITH_RESIDUAL) {
                //// Do beta
                //if (callbacks_.with_residual_) {
                //    Fragment_c tmp;
                //    #pragma unroll
                //    for (int i = 0; i < Gmem_tile::STGS; i++) {
                //        gmem_tile_.load(tmp, mi, i);
                //        post_swizzle[i].residual(callbacks_.beta_, tmp, beta_regs);
                //    }
                //}
                //}

                // Add bias
                if (callbacks_.with_bias_) {
                    #pragma unroll
                    for (int i = 0; i < Gmem_tile::STGS; i++) {
                        post_swizzle[i].add_bias(bias_regs);
                    }
                }

                if(WITH_RESIDUAL) {
                    // Do beta
                    if (callbacks_.with_residual_) {
                        #pragma unroll
                        for (int i = 0; i < Gmem_tile::STGS; i++) {
                            post_swizzle[i].residual(callbacks_.beta_, tmp_residual[i], beta_regs);
                        }
                    }
                }

                // Do Gelu
                #pragma unroll
                for (int i = 0; i < Gmem_tile::STGS; i++) {
                    callbacks_.do_gelu(post_swizzle[i]);
                }

                Fragment_c out_regs[Gmem_tile::STGS];
                // Colwert aclwmulator to ouput (float to half )
                #pragma unroll
                for (int i = 0; i < Gmem_tile::STGS; i++) {
                    out_regs[i].pack_sparse(1.0, post_swizzle[i]);

                }
                
                // Do Relu
                //#pragma unroll
                //for (int i = 0; i < Gmem_tile::STGS; i++) {
                //    if (Traits::BITS_PER_ELEMENT_C == 8) {
                //        callbacks_.do_relu_s8(out_regs[i]);
                //    } else if (Traits::BITS_PER_ELEMENT_C == 32){
                //        callbacks_.do_relu_fp32(out_regs[i]);
                //    } else {
                //        callbacks_.do_relu(out_regs[i]);
                //    }
                //}

                // Store results
                #pragma unroll
                for (int i = 0; i < Gmem_tile::STGS; i++) {
                    gmem_tile_.store(mi, i, out_regs[i]);
                }
            }

        }
    }

    // The output tile.
    Gmem_tile &gmem_tile_;
    // The shared memory tile.
    Swizzle &swizzle_;
    // The callbacks.
    Callbacks &callbacks_;
};

///////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace helpers
}  // namespace ext
} // namespace xmma
