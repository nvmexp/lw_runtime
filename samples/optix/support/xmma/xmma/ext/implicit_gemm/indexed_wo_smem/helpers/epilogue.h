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
    int ELTS_PER_STG_,
    typename Fragment_pre_swizzle_ = xmma::Fragment_epilogue_pre_swizzle<Traits, Cta_tile, true>,
    typename Fragment_post_swizzle_ = xmma::Fragment_epilogue_post_swizzle<Traits, Cta_tile, true>,
    typename Fragment_c_ = xmma::Fragment_c<Traits, Cta_tile, true>
>
struct Callbacks_epilogue {

    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;

    // The GPU arch
    using Gpu_arch = typename Traits::Gpu_arch;

    // The number of elements per STG.
    enum { ELTS_PER_STG = ELTS_PER_STG_ };

    enum { BYTES_PER_STG = ELTS_PER_STG * Traits::BITS_PER_ELEMENT_C / 8 };

    enum { ELEMENTS_PER_LDG = Xmma_tile::THREADS_PER_XMMA_N == 4 ? 4 : 8 };

    using Epilogue_type_ = typename Traits::Epilogue_type;

    // The different fragments.
    using Fragment_post_swizzle = Fragment_post_swizzle_;
    using Fragment_pre_swizzle = Fragment_pre_swizzle_;
    using Fragment_c = Fragment_c_;
    using Fragment_aclwmulator = xmma::Fragment_aclwmulator<Traits>;
    using Fragment_bias = xmma::Fragment<Epilogue_type_, ELEMENTS_PER_LDG>;
    using Fragment_per_channel_alpha = Fragment_bias;
    using C_type = typename Traits::C_type;

    enum { HAS_SUPER_HMMA = Gpu_arch::HAS_SUPER_HMMA };
    enum { COLS_OFFSET = Xmma_tile::THREADS_PER_XMMA_N == 4 ? 8 : 2 };

    template< typename Params >
    inline __device__ Callbacks_epilogue(const Params &params,
                                                   char *smem,
                                                   int bidm,
                                                   int bidn,
                                                   int bidz,
                                                   int tidx)
        : relu_lb_(params.relu_lb)
        , relu_ub_(params.relu_ub)
        , params_m_(params.k)
        , with_bias_(params.with_bias)
        , bias_ptr_(reinterpret_cast<const char*>(params.bias_gmem)) {

        per_channel_ = params.per_channel_scaling;
        alpha_ = per_channel_ ? 1.f : xmma::colwert<typename Traits::Epilogue_type>( params.alpha );
        beta_  = xmma::colwert<typename Traits::Epilogue_type>(params.beta);
        alpha_ptr_ = reinterpret_cast<const char*>(params.alpha_gmem);
        beta_ptr_ = reinterpret_cast<const char*>(params.beta_gmem);

        // The number of warps.
        const int WARPS_M = Cta_tile::WARPS_M;
        const int WARPS_N = Cta_tile::WARPS_N;
        const int WARPS_K = Cta_tile::WARPS_K;

        // The masks to select the warps.
        const int WARP_MASK_N = xmma::Warp_masks<WARPS_M, WARPS_N, WARPS_K>::N;

        // The divisor for the warps.
        const int WARP_DIV_N = WARPS_M *       1 * Cta_tile::THREADS_PER_WARP;

        // The location of the tile.
        int row = ((tidx & WARP_MASK_N) / WARP_DIV_N) * Xmma_tile::N_PER_XMMA;

        if ( ELTS_PER_STG == 4 ) {
            if (HAS_SUPER_HMMA) {
                row += (tidx % 4) * ELTS_PER_STG;
            } else {
                row += (Xmma_tile::THREADS_PER_XMMA_N == 4
                    ? ((tidx % 4) / 2) * 4 + ((tidx / 4) % 2) * 8
                    : ((tidx / 4) % 2) * 4);
            }
        } else {
            if (HAS_SUPER_HMMA) {
                row += (tidx % 4) * 2;
            } else {
                row += (Xmma_tile::THREADS_PER_XMMA_N == 4
                    ? ((tidx % 4) / 2) * 2 + ((tidx / 4) % 2) * 4
                    : ((tidx / 4) % 2) * 4);
            }
        }
        int k, g;
        xmma::fast_divmod( g, k, bidn, params.kn, params.mul_k, params.shr_k);

        // Compute the output position for each thread.
        bias_m_ = k * Cta_tile::N + row;
        bias_g_ = g;

    }
    inline __device__ void load_bias_fp16(int ni, Fragment_bias &data) {

        #pragma unroll
        for(int i = 0; i < ELEMENTS_PER_LDG / 4; i++) {

        const char *ptr = bias_ptr_
            + Traits::offset_in_bytes_c(bias_m_ + bias_g_ * params_m_
                + ni * Xmma_tile::N_PER_XMMA_PER_CTA + i * 8);

        if (ELTS_PER_STG == 1) {
            if (sizeof(Epilogue_type_) == 2) {
            lwtlass::half_t tmp;
            if (bias_m_ + ni * Xmma_tile::N_PER_XMMA_PER_CTA + i*8 < params_m_) {
                xmma::ldg((uint16_t&)tmp, ptr);
                data.elt(i*4+0) = tmp;
            }
            if (bias_m_ + ni * Xmma_tile::N_PER_XMMA_PER_CTA + 1 + i*8 < params_m_) {
                xmma::ldg((uint16_t&)tmp, ptr + Traits::offset_in_bytes_c(1));
                data.elt(i*4+1) = tmp;
            }
            if (bias_m_ + ni * Xmma_tile::N_PER_XMMA_PER_CTA +
                COLS_OFFSET + i*8 < params_m_) {
                xmma::ldg((uint16_t&)tmp, ptr + Traits::offset_in_bytes_c(COLS_OFFSET));
                data.elt(i*4+2) = tmp;
            }
            if (bias_m_ + ni * Xmma_tile::N_PER_XMMA_PER_CTA +
                COLS_OFFSET + 1 + i*8 < params_m_) {
                xmma::ldg((uint16_t&)tmp, ptr + Traits::offset_in_bytes_c(COLS_OFFSET+1));
                data.elt(i*4+3) = tmp;
            }
            } else {
            half tmp;
            if (bias_m_ + ni * Xmma_tile::N_PER_XMMA_PER_CTA + i*8 < params_m_) {
                xmma::ldg((uint16_t&)tmp, ptr);
                data.elt(i*4+0) = (float)tmp;
            }
            if (bias_m_ + ni * Xmma_tile::N_PER_XMMA_PER_CTA + 1 + i*8 < params_m_) {
                xmma::ldg((uint16_t&)tmp, ptr + Traits::offset_in_bytes_c(1));
                data.elt(i*4+1) = (float)tmp;
            }
            if (bias_m_ + ni * Xmma_tile::N_PER_XMMA_PER_CTA +
                COLS_OFFSET + i*8 < params_m_) {
                xmma::ldg((uint16_t&)tmp, ptr + Traits::offset_in_bytes_c(COLS_OFFSET));
                data.elt(i*4+2) = (float)tmp;
            }
            if (bias_m_ + ni * Xmma_tile::N_PER_XMMA_PER_CTA +
                COLS_OFFSET + 1 + i*8 < params_m_) {
                xmma::ldg((uint16_t&)tmp, ptr + Traits::offset_in_bytes_c(COLS_OFFSET+1));
                data.elt(i*4+3) = (float)tmp;
            }
            }
        } else if (ELTS_PER_STG == 2) {
            uint32_t tmp;
            float2 bias_f;
            if (bias_m_ + ni * Xmma_tile::N_PER_XMMA_PER_CTA + i*8 < params_m_) {
                xmma::ldg(tmp, ptr);
                if (sizeof(Epilogue_type_) == 2) {
                    data.reg(i*2+0) = tmp;
                } else {
                    bias_f = xmma::half2_to_float2(tmp);
                    data.elt(i*4+0) = bias_f.x;
                    data.elt(i*4+1) = bias_f.y;
                }
            }
            if (bias_m_ + ni * Xmma_tile::N_PER_XMMA_PER_CTA + COLS_OFFSET + i*8 < params_m_) {
                xmma::ldg(tmp, ptr + Traits::offset_in_bytes_c(COLS_OFFSET));
                if (sizeof(Epilogue_type_) == 2) {
                    data.reg(i*2+1) = tmp;
                } else {
                    bias_f = xmma::half2_to_float2(tmp);
                    data.elt(i*4+2) = bias_f.x;
                    data.elt(i*4+3) = bias_f.y;
                }
            }
        } else if (ELTS_PER_STG == 4) {
            uint2 tmp;
            float2 bias_f;
            if (bias_m_ + ni * Xmma_tile::N_PER_XMMA_PER_CTA + i*8 < params_m_) {
                xmma::ldg(tmp, ptr);
                if (sizeof(Epilogue_type_) == 2) {
                    data.reg(i*2+0) = tmp.x;
                    data.reg(i*2+1) = tmp.y;
                } else {
                    bias_f = xmma::half2_to_float2(tmp.x);
                    data.elt(i*4+0) = bias_f.x;
                    data.elt(i*4+1) = bias_f.y;
                    bias_f = xmma::half2_to_float2(tmp.y);
                    data.elt(i*4+2) = bias_f.x;
                    data.elt(i*4+3) = bias_f.y;
                }
            }
        } else {
        }
        }
    }
    inline __device__ void load_scalar_fp32(int ni, Fragment_bias &data,
        const char *ptr_in) {
        const char *ptr = ptr_in
            + sizeof(float) * (bias_m_ + bias_g_ * params_m_
                + ni * Xmma_tile::N_PER_XMMA_PER_CTA);

        if (ELTS_PER_STG == 1) {
            if (bias_m_ + ni * Xmma_tile::N_PER_XMMA_PER_CTA < params_m_) {
                xmma::ldg(data.reg(0), ptr);
            }
            if (bias_m_ + ni * Xmma_tile::N_PER_XMMA_PER_CTA + 1 < params_m_) {
                xmma::ldg(data.reg(1), ptr + sizeof(float)*(1));
            }
            if (bias_m_ + ni * Xmma_tile::N_PER_XMMA_PER_CTA + 8 < params_m_) {
                xmma::ldg(data.reg(2), ptr + sizeof(float)*(8));
            }
            if (bias_m_ + ni * Xmma_tile::N_PER_XMMA_PER_CTA + 9 < params_m_) {
                xmma::ldg(data.reg(3), ptr + sizeof(float)*(9));
            }
        } else if (ELTS_PER_STG == 2) {
            uint2 tmp;
            if (bias_m_ + ni * Xmma_tile::N_PER_XMMA_PER_CTA < params_m_) {
                xmma::ldg(tmp, ptr);
                data.reg(0) = tmp.x;
                data.reg(1) = tmp.y;
            }
            if (bias_m_ + ni * Xmma_tile::N_PER_XMMA_PER_CTA + 8 < params_m_) {
                xmma::ldg(tmp, ptr + sizeof(float)*(8));
                data.reg(2) = tmp.x;
                data.reg(3) = tmp.y;
            }
        } else if (ELTS_PER_STG == 4) {
            uint4 tmp;
            if (bias_m_ + ni * Xmma_tile::N_PER_XMMA_PER_CTA < params_m_) {
                xmma::ldg(tmp, ptr);
                data.reg(0) = tmp.x;
                data.reg(1) = tmp.y;
                data.reg(2) = tmp.z;
                data.reg(3) = tmp.w;
            }
        } else {
        }
    }

    inline __device__ void per_channel_scaling( Fragment_pre_swizzle &frag,
                                                Fragment_per_channel_alpha alpha ) {
        if( sizeof( Epilogue_type_ ) == 2 ) {
            if( Xmma_tile::THREADS_PER_XMMA_N == 4 ) {
                frag.reg( 0 ) = xmma::hmul2( alpha.reg( 0 ), frag.reg( 0 ) );
                frag.reg( 1 ) = xmma::hmul2( alpha.reg( 1 ), frag.reg( 1 ) );
                frag.reg( 2 ) = xmma::hmul2( alpha.reg( 0 ), frag.reg( 2 ) );
                frag.reg( 3 ) = xmma::hmul2( alpha.reg( 1 ), frag.reg( 3 ) );
            } else {
                frag.reg( 0 ) = xmma::hmul2( alpha.reg( 0 ), frag.reg( 0 ) );
                frag.reg( 1 ) = xmma::hmul2( alpha.reg( 1 ), frag.reg( 1 ) );
                frag.reg( 2 ) = xmma::hmul2( alpha.reg( 2 ), frag.reg( 2 ) );
                frag.reg( 3 ) = xmma::hmul2( alpha.reg( 3 ), frag.reg( 3 ) );
            }
        } else {
            if( Xmma_tile::THREADS_PER_XMMA_N == 4 ) {
                #pragma unroll
                for( int i = 0; i < ELEMENTS_PER_LDG; i++ ) {
                    frag.elt( i ) *= alpha.elt( i );
                    frag.elt( i + 4 ) *= alpha.elt( i );
                }
            } else {
                #pragma unroll
                for( int i = 0; i < ELEMENTS_PER_LDG; i++ ) {
                    frag.elt( i ) *= alpha.elt( i );
                }
            }
        }
    }

    inline __device__ void add_bias(
        Fragment_pre_swizzle &frag,
        Fragment_bias bias) {
        if (sizeof(Epilogue_type_) == 2) {
            if (Xmma_tile::THREADS_PER_XMMA_N == 4) {
                frag.reg(0) = xmma::hadd2(bias.reg(0), frag.reg(0));
                frag.reg(1) = xmma::hadd2(bias.reg(1), frag.reg(1));
                frag.reg(2) = xmma::hadd2(bias.reg(0), frag.reg(2));
                frag.reg(3) = xmma::hadd2(bias.reg(1), frag.reg(3));
            } else {
                frag.reg(0) = xmma::hadd2(bias.reg(0), frag.reg(0));
                frag.reg(1) = xmma::hadd2(bias.reg(1), frag.reg(1));
                frag.reg(2) = xmma::hadd2(bias.reg(2), frag.reg(2));
                frag.reg(3) = xmma::hadd2(bias.reg(3), frag.reg(3));
            }
        } else {
            if (Xmma_tile::THREADS_PER_XMMA_N == 4) {
                #pragma unroll
                for (int i = 0; i < ELEMENTS_PER_LDG; i++) {
                    frag.elt(i) += bias.elt(i);
                    frag.elt(i+4) += bias.elt(i);
                }
            } else {
                #pragma unroll
                for (int i = 0; i < ELEMENTS_PER_LDG; i++) {
                    frag.elt(i) += bias.elt(i);
                }
            }
        }
    }

    template <int N>
    inline __device__ void do_relu_fp16(int ni, Fragment_c (&frag)[N]) {
        // FP16x2 ReLu.
        half2 tmp = xmma::colwert<half2>(relu_lb_);
        uint32_t relu_lb_v2 = reinterpret_cast<uint32_t&>(tmp);

        tmp = xmma::colwert<half2>(relu_ub_);
        uint32_t relu_ub_v2 = reinterpret_cast<uint32_t&>(tmp);

        #pragma unroll
        for( int i = 0; i < Fragment_c::NUM_REGS; ++i ) {
            frag[ni].reg(i) = xmma::reluTh_fp16x2(frag[ni].reg(i), relu_lb_v2);
            frag[ni].reg(i) = xmma::relu_ub_fp16x2(frag[ni].reg(i), relu_ub_v2);
        }
    }

    template <int N>
    inline __device__ void do_relu_fp32(int ni, Fragment_c (&frag)[N]) {
        #pragma unroll
        for( int i = 0; i < Fragment_c::NUM_REGS; ++i ) {
            frag[2*ni].elt(i) = xmma::relu_fp32(frag[2*ni].elt(i), relu_lb_);
            frag[2*ni].elt(i) = xmma::relu_ub_fp32(frag[2*ni].elt(i), relu_ub_);
            frag[2*ni+1].elt(i) = xmma::relu_fp32(frag[2*ni+1].elt(i), relu_lb_);
            frag[2*ni+1].elt(i) = xmma::relu_ub_fp32(frag[2*ni+1].elt(i), relu_ub_);
        }
    }

    template <int N>
    inline __device__ void do_relu_int8(int ni, Fragment_c (&frag)[N]) {
        // s8x4 ReLu.
        int8_t tmp_lb = relu_lb_;
        uint32_t relu_lb = reinterpret_cast<uint32_t &>( tmp_lb );
        int8_t tmp_ub = relu_ub_;
        uint32_t relu_ub = reinterpret_cast<uint32_t &>( tmp_ub );
        #pragma unroll
        for( int i = 0; i < 2; ++i ) {
            frag[ni].reg( i ) = xmma::relu_s8x4( frag[ni].reg( i ), relu_lb );
            frag[ni].reg( i ) = xmma::relu_ub_s8x4( frag[ni].reg( i ), relu_ub );
        }
    }

    template <int N>
    inline __device__ void colwert_tf32 (
        int ni,
        Fragment_pre_swizzle frag,
        Fragment_c (&out)[N]) {

        if (std::is_same<Input_type_, lwtlass::float_tf32_t>::value) {
            out[2*ni].reg(0) = xmma::float_to_tf32_rn(frag.elt(0));
            out[2*ni].reg(1) = xmma::float_to_tf32_rn(frag.elt(1));
            out[2*ni].reg(2) = xmma::float_to_tf32_rn(frag.elt(2));
            out[2*ni].reg(3) = xmma::float_to_tf32_rn(frag.elt(3));
            out[2*ni+1].reg(0) = xmma::float_to_tf32_rn(frag.elt(4));
            out[2*ni+1].reg(1) = xmma::float_to_tf32_rn(frag.elt(5));
            out[2*ni+1].reg(2) = xmma::float_to_tf32_rn(frag.elt(6));
            out[2*ni+1].reg(3) = xmma::float_to_tf32_rn(frag.elt(7));
        } else {
            out[2*ni].reg(0) = frag.reg(0);
            out[2*ni].reg(1) = frag.reg(1);
            out[2*ni].reg(2) = frag.reg(2);
            out[2*ni].reg(3) = frag.reg(3);
            out[2*ni+1].reg(0) = frag.reg(4);
            out[2*ni+1].reg(1) = frag.reg(5);
            out[2*ni+1].reg(2) = frag.reg(6);
            out[2*ni+1].reg(3) = frag.reg(7);
        }
    }

    template <int N>
    inline __device__ void colwert (
        int ni,
        const Fragment_aclwmulator acc,
        Fragment_pre_swizzle (&out)[N]) {
        out[ni].scaled_colwert(alpha_, acc);
    }

    template <int N>
    inline __device__ void colwert_no_scale (
        int ni,
        const Fragment_aclwmulator acc,
        Fragment_pre_swizzle (&out)[N]) {
        out[ni].scaled_colwert(1.0, acc);
    }

    template <int N>
    inline __device__ void colwert_int8(int ni,
        Fragment_pre_swizzle frag,
        Fragment_c (&out)[N]) {

        int32_t tmp[4];
        #pragma unroll
        for( int ii = 0; ii < 2; ++ii ) {
            tmp[0] = xmma::f2i(frag.elt(4 * ii    ));
            tmp[1] = xmma::f2i(frag.elt(4 * ii + 1));
            tmp[2] = xmma::f2i(frag.elt(4 * ii + 2));
            tmp[3] = xmma::f2i(frag.elt(4 * ii + 3));
            out[ni].reg(ii) = xmma::pack_int8x4(tmp);
        }
    }

    template <int N>
    inline __device__ void colwert_fp16(
        int ni,
        Fragment_pre_swizzle frag,
        Fragment_c (&out)[N]) {
        out[ni].pack(0.0, frag);
    }

    float relu_lb_, relu_ub_;
    typename Traits::Epilogue_type alpha_, beta_;
    const char *bias_ptr_, *alpha_ptr_, *beta_ptr_;
    const int params_m_;
    bool with_bias_, per_channel_;
    int bias_m_, bias_g_;
    using Input_type_ = typename Fragment_c::Input_type_;
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
    // The callbacks to lwstomize the behaviour of the epilogue.
    typename Callbacks = xmma::helpers::Empty_callbacks_epilogue<Traits_, Cta_tile_>,
    // The class to swizzle the data.
    typename Swizzle_ = xmma::Swizzle_epilogue<Traits_, Cta_tile_, Layout_>,
    // The fragment class before the swizzling.
    typename Fragment_pre_swizzle_ = typename Callbacks::Fragment_pre_swizzle,
    // The fragment class after the swizzling.
    typename Fragment_post_swizzle_ = xmma::Fragment_epilogue_post_swizzle<Traits_, Cta_tile_, true>,
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

    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;

    enum { BYTES_PER_ELT = Traits::BITS_PER_ELEMENT_C / 8 };

    enum { STGS = Xmma_tile::XMMAS_N * 2 };

    using Epilogue_type_ = typename Traits::Epilogue_type;
    using Fragment_bias = typename Callbacks::Fragment_bias;
    using Fragment_per_channel_alpha = typename Callbacks::Fragment_per_channel_alpha;

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

    template<int N >
    inline __device__ void exelwte_split_k() {
    }

    template<typename Fragment_aclwmulator, int M, int N >
    inline __device__ void execute(Fragment_aclwmulator (&acc)[M][N]) {

        Fragment_bias bias_regs[N];
        Fragment_per_channel_alpha alpha_regs[N];

        // Load bias
        if (callbacks_.with_bias_) {
            #pragma unroll
            for (int ni = 0; ni < N; ni++) {
                if (BYTES_PER_ELT == 4) {
                    callbacks_.load_scalar_fp32(ni, bias_regs[ni], callbacks_.bias_ptr_);
                } else if (BYTES_PER_ELT == 2) {
                    callbacks_.load_bias_fp16(ni, bias_regs[ni]);
                } else if (BYTES_PER_ELT == 1) {
                    callbacks_.load_scalar_fp32(ni, bias_regs[ni], callbacks_.bias_ptr_);
                }
            }
        }
        if (callbacks_.per_channel_) {
            #pragma unroll
            for (int ni = 0; ni < N; ni++) {
                callbacks_.load_scalar_fp32(ni, alpha_regs[ni], callbacks_.alpha_ptr_);
            }
        }

        #pragma unroll
        for (int mi = 0; mi < M; mi++) {
            Fragment_pre_swizzle epi_reg[N];
            #pragma unroll
            for (int ni = 0; ni < N; ni++) {
                callbacks_.colwert(ni, acc[mi][ni], epi_reg);
            }

            // Only int8 kernels support per channel scaling
            if( std::is_same<int8_t, typename Traits::C_type>::value ) {
                if( callbacks_.per_channel_ ) {
                    #pragma unroll
                    for( int ni = 0; ni < N; ni++ ) {
                        callbacks_.per_channel_scaling( epi_reg[ni], alpha_regs[ni] );
                    }
                }
            }

            // Add bias
            if (callbacks_.with_bias_) {
                #pragma unroll
                for (int ni = 0; ni < N; ni++) {
                    callbacks_.add_bias(epi_reg[ni], bias_regs[ni]);
                }
            }

            Fragment_c out_reg[STGS];
            #pragma unroll
            for (int ni = 0; ni < N; ni++) {
                if (BYTES_PER_ELT == 4) {
                    callbacks_.colwert_tf32(ni, epi_reg[ni], out_reg);
                } else if (BYTES_PER_ELT == 2) {
                    callbacks_.colwert_fp16(ni, epi_reg[ni], out_reg);
                } else if (BYTES_PER_ELT == 1) {
                    callbacks_.colwert_int8(ni, epi_reg[ni], out_reg);
                }
            }

            // Do Relu
            #pragma unroll
            for (int ni = 0; ni < N; ni++) {
                if (BYTES_PER_ELT == 4) {
                    callbacks_.do_relu_fp32(ni, out_reg);
                } else if (BYTES_PER_ELT == 2) {
                    callbacks_.do_relu_fp16(ni, out_reg);
                } else if (BYTES_PER_ELT == 1) {
                    callbacks_.do_relu_int8(ni, out_reg);
                }
            }

            #pragma unroll
            for (int ni = 0; ni < N; ni++) {
                gmem_tile_.store(mi, ni, out_reg);
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

template<
    // The instruction traits.
    typename Traits_,
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile_,
    // The layout of the tile.
    typename Layout_,
    // The global memory tile to store the output.
    typename Gmem_tile_,
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
struct Epilogue_with_split_k {
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

    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;

    enum { BYTES_PER_ELT = Traits::BITS_PER_ELEMENT_C / 8 };

    enum { STGS = Xmma_tile::XMMAS_N * 2 };

    using Epilogue_type_ = typename Traits::Epilogue_type;
    using Fragment_bias = typename Callbacks::Fragment_bias;

    // Ctor.
    template< typename Params >
    inline __device__ Epilogue_with_split_k(const Params &params,
                               Gmem_tile &gmem_tile,
                               Swizzle &swizzle,
                               Callbacks &callbacks,
                               const xmma::Named_barrier &epi_sync = xmma::Named_barrier())
        : gmem_tile_(gmem_tile)
        , swizzle_(swizzle)
        , callbacks_(callbacks)
        , split_k_slices_(params.split_k.slices)
        , split_k_buffers_(params.split_k.buffers)
        , split_k_kernels_(params.split_k.kernels)
        , split_k_buffer_size_(params.split_k.buffer_size)
        , split_k_buffers_ptr_(params.split_k.buffers_gmem)
        , split_k_counters_ptr_(params.split_k.counters_gmem)
        , split_k_retired_ctas_ptr_(params.split_k.retired_ctas_gmem)
        , split_k_fragment_offset_(0)
        , epi_sync_(epi_sync)
        , bidm_(blockIdx.x)
        , bidn_(blockIdx.y)
        , bidz_(blockIdx.z)
        , tidx_(threadIdx.x)
        , tiles_x_(params.tiles_x)
        , tiles_y_(params.tiles_y)
        , mem_desc_c_(params.mem_descriptors.descriptor_c)
        , mem_desc_d_(params.mem_descriptors.descriptor_d) {
    }

    // The split-k function.
    template< typename Fragment_aclwmulator, int N >
    inline __device__ void split_k(int mi, Fragment_aclwmulator (&fragments)[N]) {

        // The number of CTAs per slice.
        const int ctas_per_slice = tiles_x_*tiles_y_;
        // The position of the CTA in the X*Y slice.
        const int cta_in_slice = bidn_*tiles_x_ + bidm_;
        // The number of threads in the X*Y dimension of the grid.
        const int threads_per_slice = ctas_per_slice*Cta_tile::THREADS_PER_CTA;
        // The position of the thread in that X*Y slice.
        const int thread_in_slice = cta_in_slice*Cta_tile::THREADS_PER_CTA + tidx_;

        // The number of bytes per fragment stored in memory.
        const int64_t bytes_per_fragment = (int64_t) threads_per_slice*8*4;
        // The base pointer.
        char *base_ptr = &split_k_buffer_ptr_[split_k_fragment_offset_*bytes_per_fragment];

        // Perform the reduction steps.
        for( int step = 0; step < split_k_reduction_steps_; ++step ) {

            // The address of the the ith buffer.
            const char *ptr = &base_ptr[step*split_k_buffer_size_];

            // Read the old values (if any).
            Fragment_aclwmulator old[N];
            #pragma unroll
            for( int ni = 0; ni < N; ++ni ) {
                const char *buffer = &ptr[(mi*N + ni)*bytes_per_fragment];
                xmma::deserialize_<4, 8>(old[ni].regs_,buffer,thread_in_slice, threads_per_slice);
            }

            // Add the values to the current ones.
            #pragma unroll
            for( int ni = 0; ni < N; ++ni ) {
                fragments[ni].add(old[ni]);
            }
        }

        if( !split_k_must_swizzle_output_ ) {
            #pragma unroll
            for( int ni = 0; ni < N; ++ni ) {
                char *ptr = &base_ptr[(mi*N + ni)*bytes_per_fragment];
                xmma::serialize_<4, 8>(ptr, fragments[ni].regs_, thread_in_slice, threads_per_slice);
            }
        }
    }
    template<int N >
    inline __device__ void exelwte_split_k() {

        // Is it the last CTA working on a given tile?
        split_k_must_swizzle_output_ = 1;
        // The number of reduction steps for split-k.
        split_k_reduction_steps_ = split_k_buffers_;
        // The pointer to the reduction buffer.
        split_k_buffer_ptr_ = reinterpret_cast<char*>(split_k_buffers_ptr_);

        // The loop iterator in the "regular" kernel.
        int mi = bidz_;

        // A clean fragment of aclwmulators.
        xmma::Fragment_aclwmulator<Traits> acc[N];
        #pragma unroll
        for( int ni = 0; ni < N; ++ni ) {
            acc[ni].clear();
        }

        // Do split-k.
        split_k(mi, acc);

        Fragment_pre_swizzle epi_reg[N];
        #pragma unroll
        for (int ni = 0; ni < N; ni++) {
            callbacks_.colwert_no_scale(ni, acc[ni], epi_reg);
        }

        Fragment_c out_reg[STGS];

        #pragma unroll
        for (int ni = 0; ni < N; ni++) {
            if (BYTES_PER_ELT == 4) {
                callbacks_.colwert_tf32(ni, epi_reg[ni], out_reg);
            } else if (BYTES_PER_ELT == 2) {
                callbacks_.colwert_fp16(ni, epi_reg[ni], out_reg);
            } else {
            }
        }

        #pragma unroll
        for (int ni = 0; ni < N; ni++) {
            gmem_tile_.store(mi, ni, out_reg);
        }
    }

    template<typename Fragment_aclwmulator, int M, int N >
    inline __device__ void execute(Fragment_aclwmulator (&acc)[M][N]) {

        // The number of CTAs per slice.
        const int ctas_per_slice = tiles_x_*tiles_y_;
        // The position of the CTA in the X*Y slice.
        const int cta_in_slice = bidn_*tiles_x_ + bidm_;

        // Is it the last CTA working on a given tile?
        int split_k_is_last_slice = bidz_ == split_k_slices_ - 1;
        // The number of reduction steps for split-k.
        split_k_reduction_steps_ = 0;
        // The buffer in global memory where to put the data for split-k.
        int split_k_buffer = 0;
        // Do we skip the atomics.
        int split_k_skip_atomics = 0;

        // The counter. One CTA at a time (we could also do it at the warp level).
        int32_t *split_k_counter_ptr = 0;
        // The offset to the number of retired CTAs.
        int32_t *split_k_retired_ctas_ptr = 0;

        int with_lock = 1;
        int with_unlock = 1;

        // No reduction.
        if( split_k_slices_ == 1 ) {
            split_k_skip_atomics = 1;
        // Each slice has its own buffer.
        } else if( split_k_kernels_ == 2 && split_k_slices_ == split_k_buffers_ ) {
            // The buffer. It may change later if slices > buffers (see below).
            split_k_buffer = bidz_;
            // No need to increase the counters.
            split_k_skip_atomics = 1;
        // If we enable split-k, the last slice does the final reduction.
        } else if( split_k_is_last_slice && split_k_kernels_ == 1 ) {

            // The starting buffer is 0.
            split_k_buffer = 0;
            // Usually we have split_k_buffers_ <= split_k_slices_. When split_k_buffers_ ==
            // split_k_slices_, the last slice holds its data in shared memory. So, it doesn't need
            // to load its buffer and it shouldn't load the buffer as the buffer is not initialized.
            split_k_reduction_steps_ = split_k_buffers_;
            if( split_k_slices_ == split_k_buffers_ ) {
                split_k_reduction_steps_ -= 1;
            }
            // The total number of retired CTAs per tile.
            split_k_retired_ctas_ptr = &(split_k_retired_ctas_ptr_)[cta_in_slice];

            // Wait for all the CTAs to be done with their steps.
            xmma::spin_lock(split_k_retired_ctas_ptr, with_lock ? split_k_slices_-1 : 0,
                                0, tidx_, epi_sync_);

        // If CTAs do a sequential reduction first, acquire the lock on the buffer.
        } else {

            // The corresponding buffer (when we have multiple buffers).
            split_k_buffer = bidz_ % split_k_buffers_;
            // The number of CTAs before this one.
            int predecessors = bidz_ / split_k_buffers_;
            // Do 1 parallel step unless that's the 1st CTA to write to the buffer.
            split_k_reduction_steps_ = predecessors == 0 ? 0 : 1;
            // The total number of retired CTAs per tile.
            split_k_retired_ctas_ptr = &(split_k_retired_ctas_ptr_)[cta_in_slice];

            // The counter in global memory.
            int counter = split_k_buffer*ctas_per_slice + cta_in_slice;
            // The counter. One CTA at a time (we could also do it at the warp level).
            split_k_counter_ptr = &(split_k_counters_ptr_)[counter];
            // Wait for all the CTAs preceding this one to be done with the buffer.
            xmma::spin_lock(split_k_counter_ptr, with_lock ? predecessors : 0,
                                0, tidx_, epi_sync_);
        }

        // Do we do the swizzle of the output.
        split_k_must_swizzle_output_ = split_k_is_last_slice && split_k_kernels_ == 1;

        // The pointer to the buffer.
        char *ptr = reinterpret_cast<char*>(split_k_buffers_ptr_);
        split_k_buffer_ptr_ = &ptr[split_k_buffer*split_k_buffer_size_];

        #pragma unroll
        for (int mi = 0; mi < M; mi++) {

            // Do the split-k before the swizzle (when WARPS_K == 1).
            if( Cta_tile::WARPS_K == 1 && split_k_slices_ > 1 ) {
                split_k(mi, acc[mi]);
            }

            // Early-exit if the CTA does not have extra work to do.
            if( Cta_tile::WARPS_K == 1 && !split_k_must_swizzle_output_ ) {
                continue;
            }

            Fragment_pre_swizzle epi_reg[N];
            #pragma unroll
            for (int ni = 0; ni < N; ni++) {
                callbacks_.colwert(ni, acc[mi][ni], epi_reg);
            }

            Fragment_c out_reg[STGS];
            #pragma unroll
            for (int ni = 0; ni < N; ni++) {
                if (BYTES_PER_ELT == 4) {
                    callbacks_.colwert_tf32(ni, epi_reg[ni], out_reg);
                } else if (BYTES_PER_ELT == 2) {
                    callbacks_.colwert_fp16(ni, epi_reg[ni], out_reg);
                } else {
                }
            }

            #pragma unroll
            for (int ni = 0; ni < N; ni++) {
                gmem_tile_.store(mi, ni, out_reg);
            }
        }
        // If we do not need the counters for split-k, we are good to go.
        if( split_k_skip_atomics || !with_unlock ) {
            return;
        }

        // Make sure all threads are done issueing.
        if( epi_sync_.invalid() ) {
            __syncthreads();
        } else {
            epi_sync_.wait();
        }

        // Update the counters -- release the locks.
        if( tidx_ != 0 ) {
            return;
        }

        // Before we update the lock, we need all writes to be issued and visible.
        __threadfence();

        // We can update the buffer lock and quit.
        if( !split_k_is_last_slice ) {
            atomicAdd(split_k_counter_ptr, 1);
        }

        // That's the sum of CTAs that are done.
        atomicAdd(split_k_retired_ctas_ptr, 1);

    }

    // The output tile.
    Gmem_tile &gmem_tile_;
    // The shared memory tile.
    Swizzle &swizzle_;
    // The callbacks.
    Callbacks &callbacks_;
    // The named barrier object used for epilog sync.
    const xmma::Named_barrier epi_sync_;

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

///////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace helpers
}  // namespace ext
} // namespace xmma
