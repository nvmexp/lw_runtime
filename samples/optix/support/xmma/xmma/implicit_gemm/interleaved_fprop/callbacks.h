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

#include <xmma/helpers/epilogue_with_split_k.h>
#include <xmma/volta/traits.h>
#include <xmma/turing/traits.h>
#include <xmma/ampere/traits.h>
#include <xmma/device_call.h>
namespace xmma {
namespace implicit_gemm {
namespace interleaved_fprop {

///////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Traits, typename Cta_tile, typename Gmem_tile >
struct Fragment_imma_fp32_epilogue_data {

    // The number of elements per packet.
    enum { ELEMENTS_PER_PACKET = Gmem_tile::ELEMENTS_PER_PACKET };
    enum { BYTES_PER_PACKET = ELEMENTS_PER_PACKET * sizeof(float) };
    // The number of packets per CTA in the N dimension.
    enum { PACKETS_PER_TILE_N = Cta_tile::N / ELEMENTS_PER_PACKET,
           PACKETS_PER_WARP_N = PACKETS_PER_TILE_N / Cta_tile::WARPS_N
    };
    enum { ELEMENTS_NUM = Gmem_tile::ELEMENTS_PER_STG * PACKETS_PER_WARP_N };
    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;
    // The base class.
    using Base = xmma::Fragment<float, ELEMENTS_NUM>;
    enum { ELEMENTS_PER_LDG = Gmem_tile::ELEMENTS_PER_STG };
    using LDG_TYPE = xmma::Fragment<float, 4>;

    // The number of packets per XMMA.
    enum {
        PACKETS_PER_XMMA_M = Xmma_tile::M_PER_XMMA / Xmma_tile::THREADS_PER_XMMA_M,
        PACKETS_PER_XMMA_N = Xmma_tile::N_PER_XMMA / ELEMENTS_PER_PACKET
    };
    // The number of threads needed to output a single packet.
    enum { THREADS_PER_PACKET = ELEMENTS_PER_PACKET / ELEMENTS_PER_LDG };

    // Load the fragment from global memory.
    template< typename Params >
    inline __device__ void load(const Params &params,
                                const char *bias_ptr,
                                const char *alpha_ptr,
                                const char *beta_ptr,
                                int bidn, int tidx, int params_k) {
        // The masks to select the warps.
        const int WARPS_M = Cta_tile::WARPS_M;
        const int WARPS_N = Cta_tile::WARPS_N;
        const int WARP_MASK_N = xmma::Warp_masks<WARPS_M, WARPS_N, 1>::N;
        // The divisor for the warps.
        const int WARP_DIV_N = WARPS_M * Cta_tile::THREADS_PER_WARP;

        base_k_ = bidn * Cta_tile::N +
                (tidx & WARP_MASK_N) / WARP_DIV_N * PACKETS_PER_XMMA_N * ELEMENTS_PER_PACKET +
                (tidx % THREADS_PER_PACKET) * ELEMENTS_PER_LDG;

        params_k_ = params_k;
        // Set bias.
        if ( params.with_bias ) {
            load_from_gmem(bias_ptr, bias);
        } else {
            bias.clear();
        }

        // Set alpha/beta.
        for( int i = 0; i < ELEMENTS_NUM; i++){
            alpha.elt(i) = float(params.alpha);
            beta.elt(i) = float(params.beta);
        }
        if ( params.per_channel_scaling ) {
            load_from_gmem(alpha_ptr, alpha);
            if (params.with_residual)
                load_from_gmem(beta_ptr, beta);
        }

    }

    inline __device__ void load_from_gmem(const char *ptr, Base & frag) {

        #pragma unroll
        for (int i = 0; i < PACKETS_PER_WARP_N; i++) {
            int k = i * ELEMENTS_PER_PACKET * Cta_tile::WARPS_N * PACKETS_PER_XMMA_N +
                    base_k_;
            if (k < params_k_) {
                for (int j = 0; j < ELEMENTS_PER_LDG / 4; j++) {
                    const char *addr = &ptr[(k + j * 4) * sizeof(float)];
                    uint4 tmp;
                    xmma::ldg(tmp, addr);
                    frag.reg(i * ELEMENTS_PER_LDG + j * 4 + 0) = tmp.x;
                    frag.reg(i * ELEMENTS_PER_LDG + j * 4 + 1) = tmp.y;
                    frag.reg(i * ELEMENTS_PER_LDG + j * 4 + 2) = tmp.z;
                    frag.reg(i * ELEMENTS_PER_LDG + j * 4 + 3) = tmp.w;
                }
            }
        }
    }

    // Apply relu-bias.
    template< typename Fragment_pre_swizzle >
    inline __device__ void add_bias(Fragment_pre_swizzle &f, int mi, int ni) const {
        #pragma unroll
        for( int i = 0; i < Fragment_pre_swizzle::NUM_ELTS; ++i ) {
            int index = i % 8 + ni * 8;
            f.elt(i) += bias.elt(index);
        }
    }

    Base bias, alpha, beta;
    int params_k_;
    int base_k_;
};

///////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Traits, typename Cta_tile, typename Gmem_tile >
struct Fragment_hmma_fp32_epilogue_data {

    // The number of elements per packet.
    enum { ELEMENTS_PER_PACKET = Gmem_tile::ELEMENTS_PER_PACKET };
    static_assert( ELEMENTS_PER_PACKET == 8, "" );

    // enum { BYTES_PER_PACKET = ELEMENTS_PER_PACKET * sizeof(float) };
    // The number of packets per CTA in the N dimension.
    enum { PACKETS_PER_TILE_N = Cta_tile::N / ELEMENTS_PER_PACKET,
           PACKETS_PER_WARP_N = PACKETS_PER_TILE_N / Cta_tile::WARPS_N
    };
    // enum { ELEMENTS_NUM = Gmem_tile::ELEMENTS_PER_STG * Gmem_tile::STGS };
    enum { ELEMENTS_NUM = Gmem_tile::ELEMENTS_PER_STG * PACKETS_PER_WARP_N };

    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;
    // The base class.
    using Base = xmma::Fragment<lwtlass::half_t, ELEMENTS_NUM>;
    enum { ELEMENTS_PER_LDG = Gmem_tile::ELEMENTS_PER_STG };
    static_assert( ELEMENTS_PER_LDG == 2, "" );
    // using LDG_TYPE = xmma::Fragment<float, 4>;

    // The number of packets per XMMA.
    enum {
        PACKETS_PER_XMMA_M = Xmma_tile::M_PER_XMMA / Xmma_tile::THREADS_PER_XMMA_M,
        PACKETS_PER_XMMA_N = Xmma_tile::N_PER_XMMA / ELEMENTS_PER_PACKET
    };
    // The number of threads needed to output a single packet.
    enum { THREADS_PER_PACKET = ELEMENTS_PER_PACKET / ELEMENTS_PER_LDG };

    enum { PACKETS_PER_N = Cta_tile::N / ELEMENTS_PER_PACKET };

    // Load the fragment from global memory.
    template< typename Params >
    inline __device__ void load(const Params &params,
                                const char *bias_ptr,
                                const char *alpha_ptr,
                                const char *beta_ptr,
                                int bidn, int tidx, int params_k) {
        // The masks to select the warps.
        const int WARPS_M = Cta_tile::WARPS_M;
        const int WARPS_N = Cta_tile::WARPS_N;
        const int WARP_MASK_N = xmma::Warp_masks<WARPS_M, WARPS_N, 1>::N;
        // The divisor for the warps.
        const int WARP_DIV_N = WARPS_M * Cta_tile::THREADS_PER_WARP;

         // Compute the output packet.
        int k = bidn * PACKETS_PER_N + 
               (tidx & WARP_MASK_N) / WARP_DIV_N * PACKETS_PER_XMMA_N;
        // Scale by the number of elements per packet.
        k_ = k * ELEMENTS_PER_PACKET;

        base_k_ = bidn * Cta_tile::N +
                  (tidx & WARP_MASK_N) / WARP_DIV_N * PACKETS_PER_XMMA_N * ELEMENTS_PER_PACKET;

        int packet_k = ( tidx & 0x08 ) / 2 + ( tidx & 0x02 );

        k_ += packet_k;
        base_k_ += packet_k;

        params_k_ = params_k;
        // Set bias.
        if ( params.with_bias ) {
            load_from_gmem(bias_ptr, bias);
        } else {
            bias.clear();
        }

        // Set alpha/beta.
        for( int i = 0; i < ELEMENTS_NUM / 2; ++i){
            beta.reg(i) = float2_to_half2(params.beta,params.beta); 
        }
        if ( params.per_channel_scaling ) {
            load_from_gmem(alpha_ptr, alpha);
            if (params.with_residual){
                load_from_gmem(beta_ptr, beta);
            }
        }
    }

    inline __device__ void load_from_gmem(const char *ptr, Base & frag) {
        #pragma unroll
        for( int oi = 0; oi < Gmem_tile::STGS; ++oi ) {
            // Decompose the output position inside the XMMA.
            const int ki = oi / ( PACKETS_PER_XMMA_M * PACKETS_PER_XMMA_N );
            const int kj = oi % ( PACKETS_PER_XMMA_M * PACKETS_PER_XMMA_N ) % PACKETS_PER_XMMA_N;

            // The position of the XMMA in the output tile.
            const int k = ki * Cta_tile::WARPS_N * PACKETS_PER_XMMA_N + kj;

            // Store the data.
            const int offset = ( k_ + k * ELEMENTS_PER_PACKET );
            
            if (offset < params_k_) {               
                const char *addr = &ptr[Traits::offset_in_bytes_c(offset)];
                uint32_t tmp;
                xmma::ldg(tmp, addr);
                frag.reg((oi / 4) * 2 + oi % 2) = tmp;
            }
        } 
    }

    Base bias, alpha, beta;
    int params_k_;
    int base_k_;
    int k_;
};

///////////////////////////////////////////////////////////////////////////////////////////////////

template<
    typename Traits,
    typename Cta_tile,
    typename Gmem_tile,
    typename Fragment_pre_swizzle_,
    typename Fragment_post_swizzle_,
    typename Fragment_c_
>
struct Callbacks_epilogue_fuse_base
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
    using Fragment_beta = typename Base::Fragment_beta;
    using Fragment_pre_swizzle = Fragment_pre_swizzle_;
    using Fragment_post_swizzle = Fragment_post_swizzle_;
    using Fragment_c = Fragment_c_;

    template< typename Params >
    inline __device__ Callbacks_epilogue_fuse_base(const Params &params,
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
        epilogue_data_.load(params, bias_ptr, alpha_ptr, beta_ptr, bidn, tidx, params.k * params.g);
    }

    // A callback function to get alpha.
    template< typename Epilogue >
    inline __device__ void alpha_pre_swizzle(Epilogue &epilogue,
                                             int mi, int ni,
                                             Fragment_alpha_pre_swizzle &frag) {
        #pragma unroll
        for( int i = 0; i < Fragment_alpha_pre_swizzle::NUM_REGS; ++i ) {
            int index = i % 8 + ni * 8;
            frag.reg(i) = epilogue_data_.alpha.reg(index);
        }
    }

    template< typename Epilogue >
    inline __device__ void alpha_post_swizzle(Epilogue &epilogue,
                                              int mi, int ni,
                                              Fragment_alpha_post_swizzle &frag) {
    }

    // A callback function to get beta.
    template< typename Epilogue >
    inline __device__ void beta(Epilogue &epilogue,
                                int mi, int ii,
                                Fragment_beta &frag) {
        #pragma unroll
        for( int i = 0; i < Fragment_beta::NUM_REGS; ++i ) {
            int index = i % 8 + ii/2 * 8;
            frag.reg(i) = epilogue_data_.beta.reg(index);
        }
    }

    // Post swizzle.
    template< typename Epilogue >
    inline __device__ void post_swizzle(Epilogue &epilogue,
                                        int mi,
                                        int ii,
                                        Fragment_post_swizzle &frag,
                                        int mask) {
        using Fragment_bias = xmma::Fragment<float, Fragment_post_swizzle::NUM_ELTS>;
        Fragment_bias bias_;

        #pragma unroll
        for( int i = 0; i < Fragment_bias::NUM_ELTS; ++i ) {
            bias_.elt(i) = epilogue_data_.bias.elt(ii / 2 * 8 + i);
        }
        frag.add_bias(bias_);
    }

    // We do ReLU here.
    template< typename Epilogue >
    inline __device__ void pre_pack(Epilogue &epilogue,
                                    int mi, int ii,
                                    Fragment_post_swizzle &frag) {
        #pragma unroll
        for( int i = 0; i < Fragment_post_swizzle::NUM_ELTS; ++i ) {
            frag.relu(relu_lb_);
            frag.relu_ub(relu_ub_);
        }
    }

    float relu_lb_, relu_ub_;
    Fragment_imma_fp32_epilogue_data<Traits, Cta_tile, Gmem_tile> epilogue_data_;
};

///////////////////////////////////////////////////////////////////////////////////////////////////

template<
    typename Cta_tile,
    typename Gmem_tile,
    typename Fragment_pre_swizzle_,
    typename Fragment_post_swizzle_,
    typename Fragment_c_
>
struct Callbacks_epilogue_fuse_base <xmma::Volta_hmma_fp32_interleaved_traits, Cta_tile, Gmem_tile, 
                                        Fragment_pre_swizzle_, Fragment_post_swizzle_, Fragment_c_>
    : public xmma::helpers::Empty_callbacks_epilogue_with_per_channel_alpha_beta<
        xmma::Volta_hmma_fp32_interleaved_traits,
        Cta_tile,
        Fragment_pre_swizzle_,
        Fragment_post_swizzle_,
        Fragment_c_> {
    // The base class.
    using Base = xmma::helpers::Empty_callbacks_epilogue_with_per_channel_alpha_beta<
        xmma::Volta_hmma_fp32_interleaved_traits,
        Cta_tile,
        Fragment_pre_swizzle_,
        Fragment_post_swizzle_,
        Fragment_c_>;

    // The different fragments.
    using Fragment_alpha_pre_swizzle = typename Base::Fragment_alpha_pre_swizzle;
    using Fragment_alpha_post_swizzle = typename Base::Fragment_alpha_post_swizzle;
    using Fragment_beta = xmma::Fragment<lwtlass::half_t, Base::Fragment_post_swizzle::NUM_ELTS>;
    using Fragment_pre_swizzle = Fragment_pre_swizzle_;
    using Fragment_post_swizzle = Fragment_post_swizzle_;
    using Fragment_c = Fragment_c_;

    template< typename Params >
    inline __device__ Callbacks_epilogue_fuse_base(const Params &params,
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
        epilogue_data_.load(params, bias_ptr, alpha_ptr, beta_ptr, bidn, tidx, params.k * params.g);
        per_channel_scaling = params.per_channel_scaling ;
    }

    // A callback function to get alpha.
    template< typename Epilogue >
    inline __device__ void alpha_pre_swizzle(Epilogue &epilogue,
                                             int mi, int ni,
                                             Fragment_alpha_pre_swizzle &frag) {
        
        #pragma unroll
        for( int i = 0; i < Fragment_alpha_pre_swizzle::NUM_REGS; ++i ) {
            if(per_channel_scaling){
                frag.reg(i) =  1065353216; // 1065353216 = half2(1.f,1.f)
            }else{
                frag.elt(i) = this->alpha_;
            }
        }
    }

    template< typename Epilogue >
    inline __device__ void alpha_post_swizzle(Epilogue &epilogue,
                                              int mi, int ni,
                                              Fragment_alpha_post_swizzle &frag) {
    }

    // A callback function to get beta.
    template< typename Epilogue >
    inline __device__ void beta(Epilogue &epilogue,
                                int mi, int ii,
                                Fragment_beta &frag) {
        for( int i = 0; i < Fragment_beta::NUM_ELTS / 2; ++i ) {
            int oi = (ii * Fragment_beta::NUM_ELTS / 2);
            frag.reg(i) = epilogue_data_.beta.reg((oi / 4) * 2 + oi % 2 + i);
        }   
    }

    // Post swizzle.
    template< typename Epilogue >
    inline __device__ void post_swizzle(Epilogue &epilogue,
                                        int mi,
                                        int ii,
                                        Fragment_post_swizzle &frag,
                                        int mask) {
        using Fragment_bias = xmma::Fragment<lwtlass::half_t, Fragment_post_swizzle::NUM_ELTS>;
        Fragment_bias bias_;

        using Fragment_alpha = xmma::Fragment<lwtlass::half_t, Fragment_post_swizzle::NUM_ELTS>;
        Fragment_alpha alpha_;

        #pragma unroll
        for( int i = 0; i < Fragment_bias::NUM_ELTS / 2; ++i ) {
            int oi = (ii * Fragment_bias::NUM_ELTS / 2);
            
            bias_.reg(i) = epilogue_data_.bias.reg((oi / 4) * 2 + oi % 2 + i);
        }
        
        #pragma unroll
        for( int i = 0; i < Fragment_bias::NUM_ELTS / 2; ++i ) {
            int oi = (ii * Fragment_bias::NUM_ELTS / 2);
    
            alpha_.reg(i) = epilogue_data_.alpha.reg((oi / 4) * 2 + oi % 2 + i);
        }
       
        if(per_channel_scaling){
            frag.scale(alpha_);
        }
        frag.add_bias(bias_);
    }

    // We do ReLU here.
    template< typename Epilogue >
    inline __device__ void pre_pack(Epilogue &epilogue,
                                    int mi, int ii,
                                    Fragment_post_swizzle &frag) {
    
        frag.relu(relu_lb_);  
        frag.relu_ub(relu_ub_);
    }
    bool per_channel_scaling;
    float relu_lb_, relu_ub_;
    Fragment_hmma_fp32_epilogue_data<xmma::Volta_hmma_fp32_interleaved_traits, Cta_tile, Gmem_tile> epilogue_data_;
};

///////////////////////////////////////////////////////////////////////////////////////////////////
template<
    typename Traits,
    typename Cta_tile,
    typename Gmem_tile,
    typename Fragment_pre_swizzle_,
    typename Fragment_post_swizzle_,
    typename Fragment_c_,
    bool IS_RELU_LB_NONZERO,
    bool IS_GELU = Traits::IS_GELU,
    bool IS_SWISH = Traits::IS_SWISH
>
struct Callbacks_epilogue_fuse
    : public Callbacks_epilogue_fuse_base<
        Traits,
        Cta_tile,
        Gmem_tile,
        Fragment_pre_swizzle_,
        Fragment_post_swizzle_,
        Fragment_c_> {

    // The base class.
    using Base = Callbacks_epilogue_fuse_base<
        Traits,
        Cta_tile,
        Gmem_tile,
        Fragment_pre_swizzle_,
        Fragment_post_swizzle_,
        Fragment_c_>;

    template< typename Params >
    inline __device__ Callbacks_epilogue_fuse(const Params &params,
                                              char *smem,
                                              int bidm,
                                              int bidn,
                                              int bidz,
                                              int tidx)
        : Base(params, smem, bidm, bidn, bidz, tidx) {
    }

    // We do ReLU and pack here.
    template< typename Epilogue >
    inline __device__ void pre_pack(Epilogue &epilogue,
                                    int mi, int ii,
                                    Fragment_post_swizzle_ &frag) {
        #pragma unroll
        for( int i = 0; i < Fragment_post_swizzle_::NUM_ELTS; ++i ) {
            frag.elt(i) = fmax(0.f, fmin(Base::relu_ub_, frag.elt(i)));
        }
    }

};

///////////////////////////////////////////////////////////////////////////////////////////////////
template<
    typename Cta_tile,
    typename Gmem_tile,
    typename Fragment_pre_swizzle_,
    typename Fragment_post_swizzle_,
    typename Fragment_c_
>
struct Callbacks_epilogue_fuse <xmma::Volta_hmma_fp32_interleaved_traits,
                               Cta_tile,
                               Gmem_tile,
                               Fragment_pre_swizzle_,
                               Fragment_post_swizzle_,
                               Fragment_c_,
                               true,
                               false>
    : public Callbacks_epilogue_fuse_base<
        xmma::Volta_hmma_fp32_interleaved_traits,
        Cta_tile,
        Gmem_tile,
        Fragment_pre_swizzle_,
        Fragment_post_swizzle_,
        Fragment_c_> {

    // The base class.
    using Base = Callbacks_epilogue_fuse_base<
        xmma::Volta_hmma_fp32_interleaved_traits,
        Cta_tile,
        Gmem_tile,
        Fragment_pre_swizzle_,
        Fragment_post_swizzle_,
        Fragment_c_>;

    // The different fragments.
    using Fragment_alpha_pre_swizzle = typename Base::Fragment_alpha_pre_swizzle;
    using Fragment_alpha_post_swizzle = typename Base::Fragment_alpha_post_swizzle;
    using Fragment_beta = typename Base::Fragment_beta;
    using Fragment_post_swizzle = Fragment_post_swizzle_;
    using Fragment_c = Fragment_c_;

    template< typename Params >
    inline __device__ Callbacks_epilogue_fuse(const Params &params,
                                                              char *smem,
                                                              int bidm,
                                                              int bidn,
                                                              int bidz,
                                                              int tidx)
        : Base(params, smem, bidm, bidn, bidz, tidx) {
    }
};

template<
    typename Traits,
    typename Cta_tile,
    typename Gmem_tile,
    typename Fragment_pre_swizzle_,
    typename Fragment_post_swizzle_,
    typename Fragment_c_,
    bool IS_RELU_LB_NONZERO
>
struct Callbacks_epilogue_fuse<Traits,
                               Cta_tile,
                               Gmem_tile,
                               Fragment_pre_swizzle_,
                               Fragment_post_swizzle_,
                               Fragment_c_,
                               IS_RELU_LB_NONZERO,
                               true,
                               false>
    : public Callbacks_epilogue_fuse_base<
        Traits,
        Cta_tile,
        Gmem_tile,
        Fragment_pre_swizzle_,
        Fragment_post_swizzle_,
        Fragment_c_> {

    // The base class.
    using Base = Callbacks_epilogue_fuse_base<
        Traits,
        Cta_tile,
        Gmem_tile,
        Fragment_pre_swizzle_,
        Fragment_post_swizzle_,
        Fragment_c_>;

    template< typename Params >
    inline __device__ Callbacks_epilogue_fuse(const Params &params,
                                              char *smem,
                                              int bidm,
                                              int bidn,
                                              int bidz,
                                              int tidx)
        : Base(params, smem, bidm, bidn, bidz, tidx) {
        int32_t runtime_param = params.runtime_params.runtime_param0;
        gelu_scale_ = reinterpret_cast<float &>( runtime_param );
    }

    // We do ReLU and pack here.
    template< typename Epilogue >
    inline __device__ void pre_pack(Epilogue &epilogue,
                                    int mi, int ii,
                                    Fragment_post_swizzle_ &frag) {
       // origin: 0.5f * x * (1.0f + tanh(0.797885f * (x + 0.044715f * x * x * x)));
       // new: 0.5f * (x + x * tanh(x * (0.797885f + 0.0356774f * x * x)));
       // reduce two op
       constexpr auto literal0 = 0.044715f * 0.797885f;
       constexpr auto literal1 = 0.797885f;
       constexpr auto literal2 = 0.500000f;
        #pragma unroll
        for( int i = 0; i < Fragment_post_swizzle_::NUM_ELTS; ++i ) {
            frag.elt(i) = xmma::gelu(frag.elt(i), literal0, literal1, literal2, gelu_scale_);
        }
    }

    float gelu_scale_;
};

template<
    typename Traits,
    typename Cta_tile,
    typename Gmem_tile,
    typename Fragment_pre_swizzle_,
    typename Fragment_post_swizzle_,
    typename Fragment_c_
>
struct Callbacks_epilogue_fuse<Traits,
                               Cta_tile,
                               Gmem_tile,
                               Fragment_pre_swizzle_,
                               Fragment_post_swizzle_,
                               Fragment_c_,
                               true,
                               false,
                               false>
    : public Callbacks_epilogue_fuse_base<
        Traits,
        Cta_tile,
        Gmem_tile,
        Fragment_pre_swizzle_,
        Fragment_post_swizzle_,
        Fragment_c_> {

    // The base class.
    using Base = Callbacks_epilogue_fuse_base<
        Traits,
        Cta_tile,
        Gmem_tile,
        Fragment_pre_swizzle_,
        Fragment_post_swizzle_,
        Fragment_c_>;

    using Fragment_post_swizzle = Fragment_post_swizzle_;

    template <typename Params>
    inline __device__ Callbacks_epilogue_fuse( const Params &params,
                                               char *smem,
                                               int bidm,
                                               int bidn,
                                               int bidz,
                                               int tidx )
        : Base( params, smem, bidm, bidn, bidz, tidx ) {
    }

    // We do ReLU and pack here.
    template< typename Epilogue >
    inline __device__ void pre_pack(Epilogue &epilogue,
                                    int mi, int ii,
                                    Fragment_post_swizzle_ &frag) {
#ifndef CASK_SDK_CASK_PLUGIN_LINK
        #pragma unroll
        for( int i = 0; i < Fragment_post_swizzle_::NUM_ELTS; ++i ) {
            frag.elt(i) = fmin(Base::relu_ub_, fmax(Base::relu_lb_, frag.elt(i)));
        }
#endif

#ifdef CASK_SDK_CASK_PLUGIN_LINK
        if(Fragment_post_swizzle::NUM_REGS == 4){
            ResultPack<float, 4> res;
            ResultPack<float, 4> in;
            in.getFrag(frag);
            res = activation_4(in);
            res.setFrag(frag);
        }
        else if(Fragment_post_swizzle::NUM_REGS == 8){
            ResultPack<float, 8> res;
            ResultPack<float, 8> in;
            in.getFrag(frag);
            res = activation_8(in);
            res.setFrag(frag);
        }
        else{
            ResultPack<float, 16> res;
            ResultPack<float, 16> in;
            in.getFrag(frag);
            res= activation_16(in);
            res.setFrag(frag);
        }
#endif
    }
};


template<
    typename Traits,
    typename Cta_tile,
    typename Gmem_tile,
    typename Fragment_pre_swizzle_,
    typename Fragment_post_swizzle_,
    typename Fragment_c_,
    bool IS_RELU_LB_NONZERO
>
struct Callbacks_epilogue_fuse<Traits,
                               Cta_tile,
                               Gmem_tile,
                               Fragment_pre_swizzle_,
                               Fragment_post_swizzle_,
                               Fragment_c_,
                               IS_RELU_LB_NONZERO,
                               false,
                               true>
    : public Callbacks_epilogue_fuse_base<
        Traits,
        Cta_tile,
        Gmem_tile,
        Fragment_pre_swizzle_,
        Fragment_post_swizzle_,
        Fragment_c_> {
    // The base class.
    using Base = Callbacks_epilogue_fuse_base<
        Traits,
        Cta_tile,
        Gmem_tile,
        Fragment_pre_swizzle_,
        Fragment_post_swizzle_,
        Fragment_c_>;

    template< typename Params >
    inline __device__ Callbacks_epilogue_fuse(const Params &params,
                                              char *smem,
                                              int bidm,
                                              int bidn,
                                              int bidz,
                                              int tidx)
        : Base(params, smem, bidm, bidn, bidz, tidx) {
        int32_t runtime_param = params.runtime_params.runtime_param0;
        swish_runtime_scale = reinterpret_cast<float &>( runtime_param );
    }

    // We do ReLU and pack here.
    template< typename Epilogue >
    inline __device__ void pre_pack(Epilogue &epilogue,
                                    int mi, int ii,
                                    Fragment_post_swizzle_ &frag) {

        constexpr auto literal0 = 0.500000f;
        #pragma unroll
        for( int i = 0; i < Fragment_post_swizzle_::NUM_ELTS; ++i ) {
            float v0 = literal0 * frag.elt(i);
            float v1;
            asm volatile ("tanh.approx.f32 %0, %1;" : "=f"(v1) : "f"(v0));
            float v2 = literal0 * v1;
            float v3 = v2       + literal0;
            float v4 = frag.elt(i) * v3;
            frag.elt(i) = v4 * swish_runtime_scale;
        }
    }
    float swish_runtime_scale;
};
///////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace interleaved_fprop
} // namespace implicit_gemm
} // namespace xmma

