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
#include <xmma/ext/batchnorm/fragment.h>
#include <xmma/gemm/gmem_tile.h>
#include <xmma/implicit_gemm/dgrad/gmem_tile.h>
#include <xmma/ext/batchnorm/relu_bitmask_format.h>
///////////////////////////////////////////////////////////////////////////////////////////////////

namespace xmma {
namespace ext {
namespace batchnorm {
namespace bn_stats {
namespace dgrad {

template<
    // The instruction traits.
    typename Traits,
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile
>
struct Activation {
    template<typename Fragment_c>
    inline __device__ void execute(Fragment_c &data);
};

template<
    // The instruction traits.
    typename Traits,
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile
>
struct Identity : public Activation<Traits, Cta_tile> {
    template<typename Fragment_c>
    static inline __device__ void execute(float2 *grad_dy, float2 *fprop_x,
    const xmma::Fragment_epilogue_post_swizzle_bn_stats<Traits, Cta_tile> &frag,
    const Fragment_c &xfrag,
    const xmma::Fragment<float, 8> &mean,
    const xmma::Fragment<float, 8> &ilwstd) {
        #pragma unroll
            for (int i=0; i<frag.NUM_REGS; i++) {
                grad_dy[i] =  xmma::half2_to_float2(frag.reg(i));
            }
            #pragma unroll
            for (int i=0; i<xfrag.NUM_REGS; i++) {
                fprop_x[i] =  xmma::half2_to_float2(xfrag.reg(i));
            }
    }
};

template<
     // The instruction traits.
    typename Traits,

    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,

    // ReLuBitmaskFormat
    xmma::ext::batchnorm::ReluBitmaskFormat relu_bitmask_format
>
struct ReLu : public Activation<Traits, Cta_tile> {
};

template<
    // The instruction traits.
    typename Traits,
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile
>
struct ReLu<Traits, Cta_tile, xmma::ext::batchnorm::ReluBitmaskFormat::NONE> : public Activation<Traits, Cta_tile> {
    // dRelu is implemented using fprop_x, fprop_mean, fprop_ilwstd on grad_dy
    // if (fprop_x - fprop_mean) * fprop_ilwstd >=0 :
    //   output = grad_dy
    // else :
    //   output = 0.0
    template<typename Fragment_c>
    static inline __device__ void execute(float2 *grad_dy, float2 *fprop_x,
    const xmma::Fragment_epilogue_post_swizzle_bn_stats<Traits, Cta_tile> &frag,
    const Fragment_c &xfrag,
    const xmma::Fragment<float, 8> &mean,
    const xmma::Fragment<float, 8> &ilwstd) {

        #pragma unroll
        for (int i=0; i<frag.NUM_REGS; i++) {
            grad_dy[i] =  xmma::half2_to_float2(frag.reg(i));
        }
        #pragma unroll
        for (int i=0; i<xfrag.NUM_REGS; i++) {
            fprop_x[i] =  xmma::half2_to_float2(xfrag.reg(i));
        }
        float2 tmp[frag.NUM_REGS];
        tmp[0].x = (fprop_x[0].x - mean.elt(0)) * ilwstd.elt(0);
        tmp[0].y = (fprop_x[0].y - mean.elt(1)) * ilwstd.elt(1);
        tmp[1].x = (fprop_x[1].x - mean.elt(2)) * ilwstd.elt(2);
        tmp[1].y = (fprop_x[1].y - mean.elt(3)) * ilwstd.elt(3);
        tmp[2].x = (fprop_x[2].x - mean.elt(4)) * ilwstd.elt(4);
        tmp[2].y = (fprop_x[2].y - mean.elt(5)) * ilwstd.elt(5);
        tmp[3].x = (fprop_x[3].x - mean.elt(6)) * ilwstd.elt(6);
        tmp[3].y = (fprop_x[3].y - mean.elt(7)) * ilwstd.elt(7);
        #pragma unroll
        for (int i=0; i<frag.NUM_REGS; i++) {
            if (tmp[i].x < 0.0) {
                grad_dy[i].x = 0.0;
            }
            if (tmp[i].y < 0.0) {
                grad_dy[i].y = 0.0;
            }
        }
    }
};

template<
    // The instruction traits.
    typename Traits,
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile
>
struct ReLu<Traits, Cta_tile, xmma::ext::batchnorm::ReluBitmaskFormat::FULL> : public Activation<Traits, Cta_tile> {
    // dRelu is implemented using bitmask, fprop_x, fprop_mean, fprop_ilwstd on grad_dy
    // Here bitmask is a full tensor which was used in fprop
    // if ((fprop_x - fprop_mean) * fprop_ilwstd) + bitmask >=0 :
    //   output = grad_dy
    // else :
    //   output = 0.0
    template<typename Fragment_c>
    static inline __device__ void execute(float2 *grad_dy, float2 *fprop_x,
    const xmma::Fragment_epilogue_post_swizzle_bn_stats<Traits, Cta_tile> &frag,
    const Fragment_c &xfrag,
    const xmma::Fragment<float, 8> &mean,
    const xmma::Fragment<float, 8> &ilwstd,
    const Fragment_c &bfrag) {

        float2 bitmask[frag.NUM_REGS];
        #pragma unroll
        for (int i=0; i<frag.NUM_REGS; i++) {
            grad_dy[i] =  xmma::half2_to_float2(frag.reg(i));
        }
        #pragma unroll
        for (int i=0; i<xfrag.NUM_REGS; i++) {
            fprop_x[i] =  xmma::half2_to_float2(xfrag.reg(i));
        }
        #pragma unroll
        for (int i=0; i<bfrag.NUM_REGS; i++) {
            bitmask[i] =  xmma::half2_to_float2(bfrag.reg(i));
        }
        bitmask[0].x += (fprop_x[0].x - mean.elt(0)) * ilwstd.elt(0);
        bitmask[0].y += (fprop_x[0].y - mean.elt(1)) * ilwstd.elt(1);
        bitmask[1].x += (fprop_x[1].x - mean.elt(2)) * ilwstd.elt(2);
        bitmask[1].y += (fprop_x[1].y - mean.elt(3)) * ilwstd.elt(3);
        bitmask[2].x += (fprop_x[2].x - mean.elt(4)) * ilwstd.elt(4);
        bitmask[2].y += (fprop_x[2].y - mean.elt(5)) * ilwstd.elt(5);
        bitmask[3].x += (fprop_x[3].x - mean.elt(6)) * ilwstd.elt(6);
        bitmask[3].y += (fprop_x[3].y - mean.elt(7)) * ilwstd.elt(7);
        #pragma unroll
        for (int i=0; i<frag.NUM_REGS; i++) {
            if (bitmask[i].x < 0.0) {
                grad_dy[i].x = 0.0;
            }
            if (bitmask[i].y < 0.0) {
                grad_dy[i].y = 0.0;
            }
        }
    }
};

template<
    // The instruction traits.
    typename Traits,
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile
>
struct Swish : public Activation<Traits, Cta_tile> {
    // dSwish is S(x) * (1 - x * (1 - S(x)))
    template<typename Fragment_c>
    static inline __device__ void execute(float2 *grad_dy, float2 *fprop_x,
    const xmma::Fragment_epilogue_post_swizzle_bn_stats<Traits, Cta_tile> &frag,
    Fragment_c &xfrag,
    const xmma::Fragment<float, 8> &mean,
    const xmma::Fragment<float, 8> &ilwstd
    ) {
        #pragma unroll
        for (int i=0; i<frag.NUM_REGS; i++) {
            grad_dy[i] =  xmma::half2_to_float2(frag.reg(i));
        }
        #pragma unroll
        for (int i=0; i<xfrag.NUM_REGS; i++) {
            fprop_x[i] =  xmma::half2_to_float2(xfrag.reg(i));
        }
        float2 tmp[frag.NUM_REGS];
        tmp[0].x = (fprop_x[0].x - mean.elt(0)) * ilwstd.elt(0);
        tmp[0].y = (fprop_x[0].y - mean.elt(1)) * ilwstd.elt(1);
        tmp[1].x = (fprop_x[1].x - mean.elt(2)) * ilwstd.elt(2);
        tmp[1].y = (fprop_x[1].y - mean.elt(3)) * ilwstd.elt(3);
        tmp[2].x = (fprop_x[2].x - mean.elt(4)) * ilwstd.elt(4);
        tmp[2].y = (fprop_x[2].y - mean.elt(5)) * ilwstd.elt(5);
        tmp[3].x = (fprop_x[3].x - mean.elt(6)) * ilwstd.elt(6);
        tmp[3].y = (fprop_x[3].y - mean.elt(7)) * ilwstd.elt(7);
        #pragma unroll
        for (int i=0; i<frag.NUM_REGS; i++) {
            grad_dy[i].x *= xmma::dswish(tmp[i].x);
            grad_dy[i].y *= xmma::dswish(tmp[i].y);
        }
    }
};

///////////////////////////////////////////////////////////////////////////////////////////////////
}  // namespace dgrad
}  // namespace bn_stats
}  // namespace batchnorm
}  // namespace ext
}  // namespace xmma
