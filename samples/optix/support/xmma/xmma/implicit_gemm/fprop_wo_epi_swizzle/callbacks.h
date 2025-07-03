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

#include <xmma/volta/traits.h>
#include <xmma/turing/traits.h>
#include <xmma/ampere/traits.h>
#include <xmma/helpers/epilogue.h>
#include <xmma/warp_masks.h>

///////////////////////////////////////////////////////////////////////////////////////////////////

namespace xmma {
namespace implicit_gemm {
namespace fprop_wo_epi_swizzle {

///////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Traits, typename Cta_tile, typename Gmem_tile >
struct Fragment_epilogue_activation {
    
    enum { ELEMENTS_PER_LOAD = (Cta_tile::N == 128) ? 16 : 8 };
    enum { PER_CHANNEL_ELEMENT_BITS = 32 };
    enum { THREADS_PER_ROW_PER_WARP = 4 };
    enum { LDGS = (ELEMENTS_PER_LOAD * PER_CHANNEL_ELEMENT_BITS) / 128 };
    enum { ELEMENTS_PER_LDG = 128 / PER_CHANNEL_ELEMENT_BITS };

    using Epi_element = xmma::Fragment<float, ELEMENTS_PER_LOAD>;
    
    // Load the fragment from global memory.
    template< typename Params >
    inline __device__ void load(const Params &params,
                                const char *bias_ptr,
                                const char *alpha_ptr,
                                const char *beta_ptr,
                                int bidn, int tidx, int params_k) {

        params_k_ = params_k;

        // The masks to select the warps.
        const int WARPS_M = Cta_tile::WARPS_M;
        const int WARPS_N = Cta_tile::WARPS_N;
        const int WARP_MASK_N = xmma::Warp_masks<WARPS_M, WARPS_N, 1>::N;
        // The divisor for the warps.
        const int WARP_DIV_N = WARPS_M * Cta_tile::THREADS_PER_WARP;
        
        const int warp_n = (tidx & WARP_MASK_N) / WARP_DIV_N;
        base_k = bidn * Cta_tile::N +
                 warp_n * (Cta_tile::N / 2) + 
                 (tidx % THREADS_PER_ROW_PER_WARP) * ELEMENTS_PER_LOAD;

        bias.clear();
        alpha.clear();
        beta.clear();

        // Set bias.
        if ( params.with_bias ) {
            load_from_gmem(bias_ptr, bias);
        } 

        // Set alpha/beta.
        if ( params.per_channel_scaling ) {
            load_from_gmem(alpha_ptr, alpha);
            if (params.with_residual) 
                load_from_gmem(beta_ptr, beta);
        } else {
            for( int i = 0; i < ELEMENTS_PER_LOAD; i++){
                alpha.elt(i) = float(params.alpha);
                beta.elt(i) = float(params.beta);
            }
        }

    }

    inline __device__ void load_from_gmem(const char *ptr, Epi_element &frag) {
        #pragma unroll
        for (int i = 0; i < LDGS ; i++) {
            int k = i * ELEMENTS_PER_LDG + base_k;
                    
            if (k < params_k_) {
                const char *addr = &ptr[k * sizeof(float)];
                uint4 tmp;
                xmma::ldg(tmp, addr);
                frag.reg(i * 4 + 0) = tmp.x;
                frag.reg(i * 4 + 1) = tmp.y;
                frag.reg(i * 4 + 2) = tmp.z;
                frag.reg(i * 4 + 3) = tmp.w;
            }
        }
    }

    Epi_element alpha, beta, bias;
    int params_k_;
    int base_k;
};

///////////////////////////////////////////////////////////////////////////////////////////////////

template<
    typename Traits,
    typename Cta_tile,
    typename Gmem_tile,
    typename Fragment_pre_swizzle_ = xmma::Fragment_epilogue_pre_swizzle<Traits, Cta_tile>,
    typename Fragment_post_swizzle_ = xmma::Fragment_epilogue_post_swizzle<Traits, Cta_tile>,
    typename Fragment_c_ = xmma::Fragment_c<Traits, Cta_tile>
>
struct Callbacks_epilogue
    : public xmma::helpers::Empty_callbacks_epilogue_with_per_channel_alpha_beta<
        Traits,
        Cta_tile,
        Fragment_pre_swizzle_,
        Fragment_post_swizzle_,
        Fragment_c_> {
    
    // The base class.
    using Base = xmma::helpers::Empty_callbacks_epilogue_with_per_channel_alpha_beta<
        Traits,
        Cta_tile,
        Fragment_pre_swizzle_,
        Fragment_post_swizzle_,
        Fragment_c_>;

    // The different fragments.
    using Fragment_alpha_pre_swizzle = typename Base::Fragment_alpha_pre_swizzle;
    using Fragment_alpha_post_swizzle = typename Base::Fragment_alpha_post_swizzle;
    //using Fragment_beta = typename Base::Fragment_beta;
    using Fragment_beta = typename Fragment_epilogue_activation<Traits, Cta_tile, Gmem_tile>::Epi_element;
    using Fragment_pre_swizzle = Fragment_pre_swizzle_;
    using Fragment_post_swizzle = Fragment_post_swizzle_;
    using Fragment_c = Fragment_c_;
    using Fragment_bias = xmma::Fragment<float, Fragment_post_swizzle::NUM_ELTS>;

    enum { BETA_ELEMENTS = (Cta_tile::N == 128) ? 16 : 8 };
    enum { SWIZZLE_ITERATION = Fragment_post_swizzle::NUM_REGS / 4};

    template< typename Params >
    inline __device__ Callbacks_epilogue(const Params &params,
                                         char *smem,
                                         int bidm,
                                         int bidn,
                                         int bidz,
                                         int tidx)
        : Base(params, smem, bidm, bidn, bidz, tidx)
        , relu_lb_(params.relu_lb)
        , relu_ub_(params.relu_ub) {

        // The fragment to load the bias from global memory.
        const char *bias_ptr = reinterpret_cast<const char*>(params.bias_gmem);
        const char *alpha_ptr = reinterpret_cast<const char*>(params.alpha_gmem);
        const char *beta_ptr = reinterpret_cast<const char*>(params.beta_gmem);

        epi_element.load(params, bias_ptr, alpha_ptr, beta_ptr, bidn, tidx, params.k * params.g);

    }
    
    // A callback function to get alpha.
    template< typename Epilogue >
    inline __device__ void alpha_pre_swizzle(Epilogue &epilogue,
                                             int mi, int ni,
                                             Fragment_alpha_pre_swizzle &frag) {

        frag.reg(0) = epi_element.alpha.reg(ni * 4 + 0);
        frag.reg(1) = epi_element.alpha.reg(ni * 4 + 1);
        frag.reg(2) = epi_element.alpha.reg(ni * 4 + 0);
        frag.reg(3) = epi_element.alpha.reg(ni * 4 + 1);

        frag.reg(4) = epi_element.alpha.reg(ni * 4 + 2);
        frag.reg(5) = epi_element.alpha.reg(ni * 4 + 3);
        frag.reg(6) = epi_element.alpha.reg(ni * 4 + 2);
        frag.reg(7) = epi_element.alpha.reg(ni * 4 + 3);
    }

    // Post swizzle.
    template< typename Epilogue >
    inline __device__ void post_swizzle(Epilogue &epilogue,
                                        int mi, int ii,
                                        Fragment_post_swizzle &frag,
                                        int mask) {
        Fragment_bias bias_;
        #pragma unroll
        for( int i = 0; i < Fragment_bias::NUM_REGS; ++i ) {
            bias_.reg(i) = epi_element.bias.reg(i);
        }
        frag.add_bias(bias_);
    }


    // A callback function to get beta.
    template< typename Epilogue >
    inline __device__ void beta(Epilogue &epilogue,
                                int mi, int ii,
                                Fragment_beta &frag) {
        #pragma unroll
        for( int i = 0; i < Fragment_beta::NUM_REGS ; ++i ) {
            frag.reg(i) = epi_element.beta.reg(i);
        }
    }

    template< int CONTIGUOUS >
    inline __device__ void reg_swizzle(Fragment_pre_swizzle (&pre_swizzle)[CONTIGUOUS], 
                                       Fragment_post_swizzle (&post_swizzle)[Gmem_tile::STGS]) {
        #pragma unroll
        for( int ii = 0; ii < Gmem_tile::STGS; ++ii ) {
            #pragma unroll
            for( int jj = 0; jj < SWIZZLE_ITERATION ; ++jj ) {
                post_swizzle[ii].elt(jj * 4) = pre_swizzle[jj].elt(0 + ii * 2);
                post_swizzle[ii].elt(jj * 4 + 1) = pre_swizzle[jj].elt(1 + ii * 2);
                post_swizzle[ii].elt(jj * 4 + 2) = pre_swizzle[jj].elt(4 + ii * 2);
                post_swizzle[ii].elt(jj * 4 + 3) = pre_swizzle[jj].elt(5 + ii * 2);
            }
        }
    }

    // We do ReLU here.
    template< typename Epilogue >
    inline __device__ void pre_pack(Epilogue &epilogue,
                                    int mi, int ii,
                                    Fragment_post_swizzle &frag) {                       
        #pragma unroll
        for( int i = 0; i < Fragment_post_swizzle::NUM_ELTS; ++i ) {
            frag.elt(i) = fmin(relu_ub_, fmax(relu_lb_, frag.elt(i)));
        }
    }

    float relu_lb_, relu_ub_;
    Fragment_epilogue_activation<Traits, Cta_tile, Gmem_tile> epi_element;
};

///////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace fprop_wo_epi_swizzle
} // namespace implicit_gemm
} // namespace xmma