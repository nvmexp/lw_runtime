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
#include <xmma/ext/sparse/ampere/traits.h>

namespace xmma {
namespace ext {
namespace gemm {
namespace sparse_hmma_gemm {


///////////////////////////////////////////////////////////////////////////////////////////////////

template<
    typename Traits,
    typename Cta_tile,
    typename Fragment_pre_swizzle_,
    typename Fragment_post_swizzle_,
    typename Fragment_c_,
    bool DISABLE_BIAS,
    bool DISABLE_RELU
>
struct Callbacks_epilogue_with_bias_and_relu_base
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

    // The different fragments.
    using Fragment_post_swizzle = Fragment_post_swizzle_;
    using Fragment_c = Fragment_c_;

    template< typename Params >
    inline __device__ Callbacks_epilogue_with_bias_and_relu_base(const Params &params,
                                                                 char *smem,
                                                                 int bidm,
                                                                 int bidn,
                                                                 int bidz,
                                                                 int tidx)
        : Base(params, smem, bidm, bidn, bidz, tidx) {
    }

    // Post swizzle.
    template< typename Epilogue >
    inline __device__ void post_swizzle(Epilogue &epilogue,
                                        int mi,
                                        int ii,
                                        Fragment_post_swizzle &frag,
                                        int mask) {
        // Bias is performed at pre_store stage
    }

    // Pre Pack.
    template< typename Epilogue >
    inline __device__ void pre_pack(Epilogue &epilogue,
                                    int mi, int ii,
                                    Fragment_post_swizzle &frag) {
        // ReLU is performed at pre_store stage
    }

};

///////////////////////////////////////////////////////////////////////////////////////////////////

template<
    typename Input_type,
    typename Output_type,
    typename Traits,
    typename Layout,
    typename Cta_tile,
    typename Fragment_pre_swizzle_ = xmma::Fragment_epilogue_pre_swizzle<Traits, Cta_tile>,
    typename Fragment_post_swizzle_ = xmma::Fragment_epilogue_post_swizzle<Traits, Cta_tile>,
    typename Fragment_c_ = xmma::Fragment_c<Traits, Cta_tile>,
    bool DISABLE_BIAS = false,
    bool DISABLE_RELU = false
> struct Callbacks_epilogue_with_bias_and_relu{
};

///////////////////////////////////////////////////////////////////////////////////////////////////

template<
    typename Input_type,
    typename Output_type,
    typename Cta_tile,
    typename Fragment_pre_swizzle_,
    typename Fragment_post_swizzle_,
    typename Fragment_c_,
    bool DISABLE_BIAS,
    bool DISABLE_RELU
>
struct Callbacks_epilogue_with_bias_and_relu<Input_type,
                                             Output_type,
                                             xmma::Ampere_sphmma_fp32_traits,
                                             xmma::Col,
                                             Cta_tile,
                                             Fragment_pre_swizzle_,
                                             Fragment_post_swizzle_,
                                             Fragment_c_,
                                             DISABLE_BIAS,
                                             DISABLE_RELU>
    : public Callbacks_epilogue_with_bias_and_relu_base<xmma::Ampere_sphmma_fp32_traits,
                                                        Cta_tile,
                                                        Fragment_pre_swizzle_,
                                                        Fragment_post_swizzle_,
                                                        Fragment_c_,
                                                        DISABLE_BIAS,
                                                        DISABLE_RELU> {
    using Traits = xmma::Ampere_sphmma_fp32_traits;
    // The base class.
    using Base = Callbacks_epilogue_with_bias_and_relu_base<Traits,
                                                            Cta_tile,
                                                            Fragment_pre_swizzle_,
                                                            Fragment_post_swizzle_,
                                                            Fragment_c_,
                                                            DISABLE_BIAS,
                                                            DISABLE_RELU>;

    // The different fragments.
    using Fragment_post_swizzle = Fragment_post_swizzle_;
    using Fragment_c = Fragment_c_;
    using C_type = typename Traits::C_type;
    using Fragment_bias = Fragment_c;
    using Gmem_tile_epilogue_distribution = 
        xmma::helpers::Gmem_tile_epilogue_distribution<Traits, Cta_tile, xmma::Col>;

    // The number of bytes per STG.
    enum { BYTES_PER_STG = 16 };
    // The number of elements per STG per thread.
    enum { ELEMENTS_PER_STG = BYTES_PER_STG * 8 / Traits::BITS_PER_ELEMENT_C };

    template< typename Params >
    inline __device__ Callbacks_epilogue_with_bias_and_relu(const Params &params,
                                                            char *smem,
                                                            int bidm,
                                                            int bidn,
                                                            int bidz,
                                                            int tidx)
        : Base(params, smem, bidm, bidn, bidz, tidx) {
            if( !DISABLE_RELU ) {
                relu_lb_ = xmma::colwert<typename Traits::Epilogue_type>(params.relu_lb);
                relu_ub_ = xmma::colwert<typename Traits::Epilogue_type>(params.relu_ub);
            }
            if( !DISABLE_BIAS ) {
                params_with_bias_ = params.with_bias;
                if( params_with_bias_ ) {
                    int32_t row = Gmem_tile_epilogue_distribution::compute_col(tidx);
                    int32_t m = bidm * Cta_tile::M + row * ELEMENTS_PER_STG;
                    const uint4 *bias_ptr =  reinterpret_cast<const uint4 *>(params.bias_gmem);
                    bias_ptr += m / ELEMENTS_PER_STG;
                    if( m < params_with_bias_ ) {
                        uint4 tmp;
                        xmma::ldg(tmp, bias_ptr);
                        bias_.reg(0) = tmp.x;
                        bias_.reg(1) = tmp.y;
                        bias_.reg(2) = tmp.z;
                        bias_.reg(3) = tmp.w;
                    }
                }
            }
    }

    // Pre Pack.
    template< typename Epilogue >
    inline __device__ void pre_pack(Epilogue &epilogue,
                                    int mi, int ii,
                                    Fragment_post_swizzle &frag) {
        if ( !DISABLE_BIAS && params_with_bias_ ) {
            frag.add_bias(this->bias_);
        }
        if ( !DISABLE_RELU ) {
            frag.relu(relu_lb_);
            // frag.relu_ub(relu_ub_);
        }
    }

    // Pre store.
    template< typename Epilogue >
    inline __device__
    void pre_store(Epilogue &epilogue, int mi, int ii, Fragment_c &frag, int mask) {
        //Keep here in case we need fp16/bf16 reference
	    //if ( !DISABLE_BIAS && params_with_bias_ ) {
        //    frag.add_bias(this->bias_);
        //}
        //if ( !DISABLE_RELU ) {
        //    frag.relu(relu_lb_);
        //    // frag.relu_ub(relu_ub_);
        //}
    }

    typename Traits::Epilogue_type relu_lb_, relu_ub_;
    Fragment_bias bias_;
    int32_t params_with_bias_;
};

///////////////////////////////////////////////////////////////////////////////////////////////////

template<
    typename Input_type,
    typename Output_type,
    typename Cta_tile,
    typename Fragment_pre_swizzle_,
    typename Fragment_post_swizzle_,
    typename Fragment_c_,
    bool DISABLE_BIAS,
    bool DISABLE_RELU
>
struct Callbacks_epilogue_with_bias_and_relu<Input_type,
                                             Output_type,
                                             xmma::Ampere_sphmma_fp32_traits,
                                             xmma::Row,
                                             Cta_tile,
                                             Fragment_pre_swizzle_,
                                             Fragment_post_swizzle_,
                                             Fragment_c_,
                                             DISABLE_BIAS,
                                             DISABLE_RELU>
    : public Callbacks_epilogue_with_bias_and_relu_base<xmma::Ampere_sphmma_fp32_traits,
                                                        Cta_tile,
                                                        Fragment_pre_swizzle_,
                                                        Fragment_post_swizzle_,
                                                        Fragment_c_,
                                                        DISABLE_BIAS,
                                                        DISABLE_RELU> {
    using Traits = xmma::Ampere_sphmma_fp32_traits;
    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;
    // The base class.
    using Base = Callbacks_epilogue_with_bias_and_relu_base<Traits,
                                                            Cta_tile,
                                                            Fragment_pre_swizzle_,
                                                            Fragment_post_swizzle_,
                                                            Fragment_c_,
                                                            DISABLE_BIAS,
                                                            DISABLE_RELU>;

    // The different fragments.
    using Fragment_post_swizzle = Fragment_post_swizzle_;
    using Fragment_c = Fragment_c_;
    using C_type = typename Traits::C_type;
    using Fragment_bias = Fragment_c;
    using Tile_Distribution = 
        xmma::helpers::Gmem_row_stride_tile_epilogue_distribution<Traits, Cta_tile>;        

    // The number of bytes per STG.
    enum { BYTES_PER_STG = 16 };
    // The number of elements per STG per thread.
    enum { ELEMENTS_PER_STG = BYTES_PER_STG * 8 / Traits::BITS_PER_ELEMENT_C };



    template< typename Params >
    inline __device__ Callbacks_epilogue_with_bias_and_relu(const Params &params,
                                                            char *smem,
                                                            int bidm,
                                                            int bidn,
                                                            int bidz,
                                                            int tidx)
        : Base(params, smem, bidm, bidn, bidz, tidx) {
            if( !DISABLE_RELU ) {
                relu_lb_ = xmma::colwert<typename Traits::Epilogue_type>(params.relu_lb);
                relu_ub_ = xmma::colwert<typename Traits::Epilogue_type>(params.relu_ub);
            }
            if( !DISABLE_BIAS ) {
                params_with_bias_ = params.with_bias;
                if( params_with_bias_ ) {
                    row_ = Tile_Distribution::compute_row(tidx);
                    m_ = bidm * Cta_tile::M + row_;
                    bias_ptr_f16 =  reinterpret_cast<const uint16_t *>(params.bias_gmem);
                }
            }
    }

    // Pre Pack.
    template< typename Epilogue >
    inline __device__ void pre_pack(Epilogue &epilogue,
                                    int mi, int ii,
                                    Fragment_post_swizzle &frag) {
	    if ( !DISABLE_BIAS && params_with_bias_ ) {
            int32_t offset = Tile_Distribution::compute_offset(mi, ii);
            int32_t bias_row = offset + m_;
            const uint16_t *bias_ptr_f16_tmp = bias_ptr_f16;
            bias_ptr_f16_tmp+=bias_row;

            if(bias_row < params_with_bias_) {
                uint16_t tmp;
                xmma::ldg(tmp, bias_ptr_f16_tmp);
                frag.add_single_bias(tmp);
            }
        }
        if ( !DISABLE_RELU ) {
            frag.relu(relu_lb_);
            // frag.relu_ub(relu_ub_);
        }
    }

    // Pre store.
    template< typename Epilogue >
    inline __device__
    void pre_store(Epilogue &epilogue, int mi, int ii, Fragment_c &frag, int mask) {
        //Keep here in case we need fp16/bf16 reference
	    //if ( !DISABLE_BIAS && params_with_bias_ ) {
        //    int32_t offset = Tile_Distribution::compute_offset(mi, ii);
        //    int32_t bias_row = offset + m_;
        //    const uint16_t *bias_ptr_f16_tmp = bias_ptr_f16;
        //    bias_ptr_f16_tmp+=bias_row;
        //
        //    if(bias_row < params_with_bias_) {
        //        uint16_t tmp;
        //        xmma::ldg(tmp, bias_ptr_f16_tmp);
        //        frag.add_bias_v2(tmp);
        //    }
        //}
        //if ( !DISABLE_RELU ) {
        //    frag.relu(relu_lb_);
        //}
    }

    typename Traits::Epilogue_type relu_lb_, relu_ub_;
    Fragment_bias bias_;
    int32_t params_with_bias_;
    int32_t row_;
    int32_t m_;
    const uint16_t *bias_ptr_f16;
    
};

///////////////////////////////////////////////////////////////////////////////////////////////////

template<
    typename Input_type,
    typename Output_type,
    typename Cta_tile,
    typename Fragment_pre_swizzle_,
    typename Fragment_post_swizzle_,
    typename Fragment_c_,
    bool DISABLE_BIAS,
    bool DISABLE_RELU
>
struct Callbacks_epilogue_with_bias_and_relu<Input_type,
                                             Output_type,
                                             xmma::Ampere_sphmma_bf16_fp32_bf16_traits,
                                             xmma::Col,
                                             Cta_tile,
                                             Fragment_pre_swizzle_,
                                             Fragment_post_swizzle_,
                                             Fragment_c_,
                                             DISABLE_BIAS,
                                             DISABLE_RELU>
    : public Callbacks_epilogue_with_bias_and_relu_base<xmma::Ampere_sphmma_bf16_fp32_bf16_traits,
                                                        Cta_tile,
                                                        Fragment_pre_swizzle_,
                                                        Fragment_post_swizzle_,
                                                        Fragment_c_,
                                                        DISABLE_BIAS,
                                                        DISABLE_RELU> {
    using Traits = xmma::Ampere_sphmma_bf16_fp32_bf16_traits;
    // The base class.
    using Base = Callbacks_epilogue_with_bias_and_relu_base<Traits,
                                                            Cta_tile,
                                                            Fragment_pre_swizzle_,
                                                            Fragment_post_swizzle_,
                                                            Fragment_c_,
                                                            DISABLE_BIAS,
                                                            DISABLE_RELU>;
    // The different fragments.
    using Fragment_post_swizzle = Fragment_post_swizzle_;
    using Fragment_c = Fragment_c_;
    using C_type = typename Traits::C_type;
    using Fragment_bias = Fragment_c;
    using Gmem_tile_epilogue_distribution = 
        xmma::helpers::Gmem_tile_epilogue_distribution<Traits, Cta_tile, xmma::Col>;

    // The number of bytes per STG.
    enum { BYTES_PER_STG = 16 };
    // The number of elements per STG per thread.
    enum { ELEMENTS_PER_STG = BYTES_PER_STG * 8 / Traits::BITS_PER_ELEMENT_C };

    template< typename Params >
    inline __device__ Callbacks_epilogue_with_bias_and_relu(const Params &params,
                                                            char *smem,
                                                            int bidm,
                                                            int bidn,
                                                            int bidz,
                                                            int tidx)
        : Base(params, smem, bidm, bidn, bidz, tidx) {
            if( !DISABLE_RELU ) {
                relu_lb_ = xmma::colwert<typename Traits::Epilogue_type>(params.relu_lb);
                relu_ub_ = xmma::colwert<typename Traits::Epilogue_type>(params.relu_ub);
            }
            if( !DISABLE_BIAS ) {
                params_with_bias_ = params.with_bias;
                if( params_with_bias_ ) {
                    int32_t row = Gmem_tile_epilogue_distribution::compute_col(tidx);
                    int32_t m = bidm * Cta_tile::M + row * ELEMENTS_PER_STG;
                    const uint4 *bias_ptr =  reinterpret_cast<const uint4 *>(params.bias_gmem);
                    bias_ptr += m / ELEMENTS_PER_STG;
                    if( m < params_with_bias_ ) {
                        uint4 tmp;
                        xmma::ldg(tmp, bias_ptr);
                        bias_.reg(0) = tmp.x;
                        bias_.reg(1) = tmp.y;
                        bias_.reg(2) = tmp.z;
                        bias_.reg(3) = tmp.w;
                    }
                }
            }
    }

    // Pre Pack.
    template< typename Epilogue >
    inline __device__ void pre_pack(Epilogue &epilogue,
                                    int mi, int ii,
                                    Fragment_post_swizzle &frag) {

        if ( !DISABLE_BIAS && params_with_bias_ ) {
            frag.add_bias(this->bias_);
        }
        if ( !DISABLE_RELU ) {
            frag.relu(relu_lb_);
        }                            
    }

    // Pre store.
    template< typename Epilogue >
    inline __device__
    void pre_store(Epilogue &epilogue, int mi, int ii, Fragment_c &frag, int mask) {
        //Keep here in case we need fp16/bf16 reference
	    //if ( !DISABLE_BIAS && params_with_bias_ ) {
        //    frag.add_bias(this->bias_);
        //}
        //if ( !DISABLE_RELU ) {
        //    frag.relu(relu_lb_);
        //}
    }

    typename Traits::Epilogue_type relu_lb_, relu_ub_;
    Fragment_bias bias_;
    int32_t params_with_bias_;

};

///////////////////////////////////////////////////////////////////////////////////////////////////

template<
    typename Input_type,
    typename Output_type,
    typename Cta_tile,
    typename Fragment_pre_swizzle_,
    typename Fragment_post_swizzle_,
    typename Fragment_c_,
    bool DISABLE_BIAS,
    bool DISABLE_RELU
>
struct Callbacks_epilogue_with_bias_and_relu<Input_type,
                                             Output_type,
                                             xmma::Ampere_sphmma_bf16_fp32_bf16_traits,
                                             xmma::Row,
                                             Cta_tile,
                                             Fragment_pre_swizzle_,
                                             Fragment_post_swizzle_,
                                             Fragment_c_,
                                             DISABLE_BIAS,
                                             DISABLE_RELU>
    : public Callbacks_epilogue_with_bias_and_relu_base<xmma::Ampere_sphmma_bf16_fp32_bf16_traits,
                                                        Cta_tile,
                                                        Fragment_pre_swizzle_,
                                                        Fragment_post_swizzle_,
                                                        Fragment_c_,
                                                        DISABLE_BIAS,
                                                        DISABLE_RELU> {
    using Traits = xmma::Ampere_sphmma_bf16_fp32_bf16_traits;
    // The base class.
    using Base = Callbacks_epilogue_with_bias_and_relu_base<Traits,
                                                            Cta_tile,
                                                            Fragment_pre_swizzle_,
                                                            Fragment_post_swizzle_,
                                                            Fragment_c_,
                                                            DISABLE_BIAS,
                                                            DISABLE_RELU>;

    // The different fragments.
    using Fragment_post_swizzle = Fragment_post_swizzle_;
    using Fragment_c = Fragment_c_;
    using C_type = typename Traits::C_type;
    using Fragment_bias = Fragment_c;
    using Tile_Distribution = 
        xmma::helpers::Gmem_row_stride_tile_epilogue_distribution<Traits, Cta_tile>;        

    // The number of bytes per STG.
    enum { BYTES_PER_STG = 16 };
    // The number of elements per STG per thread.
    enum { ELEMENTS_PER_STG = BYTES_PER_STG * 8 / Traits::BITS_PER_ELEMENT_C };

    template< typename Params >
    inline __device__ Callbacks_epilogue_with_bias_and_relu(const Params &params,
                                                            char *smem,
                                                            int bidm,
                                                            int bidn,
                                                            int bidz,
                                                            int tidx)
        : Base(params, smem, bidm, bidn, bidz, tidx) {
            if( !DISABLE_RELU ) {
                relu_lb_ = xmma::colwert<typename Traits::Epilogue_type>(params.relu_lb);
                relu_ub_ = xmma::colwert<typename Traits::Epilogue_type>(params.relu_ub);
            }
            if( !DISABLE_BIAS ) {
                params_with_bias_ = params.with_bias;
                if( params_with_bias_ ) {
                    row_ = Tile_Distribution::compute_row(tidx);
                    m_ = bidm * Cta_tile::M + row_;
                    bias_ptr_bf16 =  reinterpret_cast<const uint16_t *>(params.bias_gmem);
                }
            }
    }

    // Pre Pack.
    template< typename Epilogue >
    inline __device__ void pre_pack(Epilogue &epilogue,
                                    int mi, int ii,
                                    Fragment_post_swizzle &frag) {
	    if ( !DISABLE_BIAS && params_with_bias_ ) {
            int32_t offset = Tile_Distribution::compute_offset(mi, ii);
            int32_t bias_row = offset + m_;
            const uint16_t *bias_ptr_bf16_tmp = bias_ptr_bf16;
            bias_ptr_bf16_tmp+=bias_row;

            if(bias_row < params_with_bias_) {
                uint16_t tmp;
                xmma::ldg(tmp, bias_ptr_bf16_tmp);
                frag.add_single_bias(tmp);
            }
        }     
        if ( !DISABLE_RELU ) {
            frag.relu(relu_lb_);
        }                            
    }

    // Pre store.
    template< typename Epilogue >
    inline __device__
    void pre_store(Epilogue &epilogue, int mi, int ii, Fragment_c &frag, int mask) {
        //Keep here in case we need fp16/bf16 reference
	    //if ( !DISABLE_BIAS && params_with_bias_ ) {
        //    int32_t offset = Tile_Distribution::compute_offset(mi, ii);
        //    int32_t bias_row = offset + m_;
        //    const uint16_t *bias_ptr_bf16_tmp = bias_ptr_bf16;
        //    bias_ptr_bf16_tmp+=bias_row;
        //
        //    if(bias_row < params_with_bias_) {
        //        uint16_t tmp;
        //        xmma::ldg(tmp, bias_ptr_bf16_tmp);
        //        frag.add_bias_v2(tmp);
        //    }
        //}
        //if ( !DISABLE_RELU ) {
        //    frag.relu(relu_lb_);
        //}
    }

    typename Traits::Epilogue_type relu_lb_, relu_ub_;
    Fragment_bias bias_;
    int32_t params_with_bias_;
    int32_t row_;
    int32_t m_;
    const uint16_t *bias_ptr_bf16;
};

///////////////////////////////////////////////////////////////////////////////////////////////////

template<
    typename Input_type,
    typename Output_type,
    typename Cta_tile,
    typename Fragment_pre_swizzle_,
    typename Fragment_post_swizzle_,
    typename Fragment_c_,
    bool DISABLE_BIAS,
    bool DISABLE_RELU
>
struct Callbacks_epilogue_with_bias_and_relu<Input_type,
                                             Output_type,
                                             xmma::Ampere_sphmma_tf32_traits<Input_type, Output_type>,
                                             xmma::Col,
                                             Cta_tile,
                                             Fragment_pre_swizzle_,
                                             Fragment_post_swizzle_,
                                             Fragment_c_,
                                             DISABLE_BIAS,
                                             DISABLE_RELU>
    : public Callbacks_epilogue_with_bias_and_relu_base<xmma::Ampere_sphmma_tf32_traits<Input_type, Output_type>,
                                                        Cta_tile,
                                                        Fragment_pre_swizzle_,
                                                        Fragment_post_swizzle_,
                                                        Fragment_c_,
                                                        DISABLE_BIAS,
                                                        DISABLE_RELU> {
    using Traits = xmma::Ampere_sphmma_tf32_traits<Input_type, Output_type>;
    // The base class.
    using Base = Callbacks_epilogue_with_bias_and_relu_base<Traits,
                                                            Cta_tile,
                                                            Fragment_pre_swizzle_,
                                                            Fragment_post_swizzle_,
                                                            Fragment_c_,
                                                            DISABLE_BIAS,
                                                            DISABLE_RELU>;

    // The different fragments.
    using Fragment_post_swizzle = Fragment_post_swizzle_;
    using Fragment_c = Fragment_c_;
    using C_type = typename Traits::C_type;
    using Fragment_bias = Fragment_c;
    using Gmem_tile_epilogue_distribution = 
        xmma::helpers::Gmem_tile_epilogue_distribution<Traits, Cta_tile, xmma::Col>;

    // The number of bytes per STG.
    enum { BYTES_PER_STG = 16 };
    // The number of elements per STG per thread.
    enum { ELEMENTS_PER_STG = BYTES_PER_STG * 8 / Traits::BITS_PER_ELEMENT_C };

    template< typename Params >
    inline __device__ Callbacks_epilogue_with_bias_and_relu(const Params &params,
                                                            char *smem,
                                                            int bidm,
                                                            int bidn,
                                                            int bidz,
                                                            int tidx)
        : Base(params, smem, bidm, bidn, bidz, tidx) {
            if( !DISABLE_RELU ) {
                relu_lb_ = xmma::colwert<typename Traits::Epilogue_type>(params.relu_lb);
                relu_ub_ = xmma::colwert<typename Traits::Epilogue_type>(params.relu_ub);
            }
            if( !DISABLE_BIAS ) {
                params_with_bias_ = params.with_bias;
                if( params_with_bias_ ) {
                    int32_t row = Gmem_tile_epilogue_distribution::compute_col(tidx);
                    int32_t m = bidm * Cta_tile::M + row * ELEMENTS_PER_STG;
                    const uint4 *bias_ptr =  reinterpret_cast<const uint4 *>(params.bias_gmem);

                    bias_ptr += m / ELEMENTS_PER_STG;
                    if( m < params_with_bias_ ) {
                        uint4 tmp;
                        xmma::ldg(tmp, bias_ptr);
                        bias_.reg(0) = tmp.x;
                        bias_.reg(1) = tmp.y;
                        bias_.reg(2) = tmp.z;
                        bias_.reg(3) = tmp.w;
                    }
                    
                }
            }
    }
    // Pre store.
    template< typename Epilogue >
    inline __device__
    void pre_store(Epilogue &epilogue, int mi, int ii, Fragment_c &frag, int mask) {

	    if ( !DISABLE_BIAS && params_with_bias_ ) {
            frag.add_bias(this->bias_);
        }
        if ( !DISABLE_RELU ) {
            frag.relu(relu_lb_);
        }

        frag.output_colwert();
    }

    typename Traits::Epilogue_type relu_lb_, relu_ub_;
    Fragment_bias bias_;
    int32_t params_with_bias_;
};

///////////////////////////////////////////////////////////////////////////////////////////////////

template<
    typename Input_type,
    typename Output_type,
    typename Cta_tile,
    typename Fragment_pre_swizzle_,
    typename Fragment_post_swizzle_,
    typename Fragment_c_,
    bool DISABLE_BIAS,
    bool DISABLE_RELU
>
struct Callbacks_epilogue_with_bias_and_relu<Input_type,
                                             Output_type,
                                             xmma::Ampere_sphmma_tf32_traits<Input_type, Output_type>,
                                             xmma::Row,
                                             Cta_tile,
                                             Fragment_pre_swizzle_,
                                             Fragment_post_swizzle_,
                                             Fragment_c_,
                                             DISABLE_BIAS,
                                             DISABLE_RELU>
    : public Callbacks_epilogue_with_bias_and_relu_base<xmma::Ampere_sphmma_tf32_traits<Input_type, Output_type>,
                                                        Cta_tile,
                                                        Fragment_pre_swizzle_,
                                                        Fragment_post_swizzle_,
                                                        Fragment_c_,
                                                        DISABLE_BIAS,
                                                        DISABLE_RELU> {
    using Traits = xmma::Ampere_sphmma_tf32_traits<Input_type, Output_type>;
    // The base class.
    using Base = Callbacks_epilogue_with_bias_and_relu_base<Traits,
                                                            Cta_tile,
                                                            Fragment_pre_swizzle_,
                                                            Fragment_post_swizzle_,
                                                            Fragment_c_,
                                                            DISABLE_BIAS,
                                                            DISABLE_RELU>;

    // The different fragments.
    using Fragment_post_swizzle = Fragment_post_swizzle_;
    using Fragment_c = Fragment_c_;
    using C_type = typename Traits::C_type;
    using Fragment_bias = Fragment_c;
    using Tile_Distribution = 
        xmma::helpers::Gmem_row_stride_tile_epilogue_distribution<Traits, Cta_tile>;

    // The number of bytes per STG.
    enum { BYTES_PER_STG = 16 };
    // The number of elements per STG per thread.
    enum { ELEMENTS_PER_STG = BYTES_PER_STG * 8 / Traits::BITS_PER_ELEMENT_C };

    template< typename Params >
    inline __device__ Callbacks_epilogue_with_bias_and_relu(const Params &params,
                                                            char *smem,
                                                            int bidm,
                                                            int bidn,
                                                            int bidz,
                                                            int tidx)
        : Base(params, smem, bidm, bidn, bidz, tidx) {

            if( !DISABLE_RELU ) {
                relu_lb_ = xmma::colwert<typename Traits::Epilogue_type>(params.relu_lb);
                relu_ub_ = xmma::colwert<typename Traits::Epilogue_type>(params.relu_ub);
            }
            if( !DISABLE_BIAS ) {
                params_with_bias_ = params.with_bias;           
                if( params_with_bias_ ) {
                    row_ = Tile_Distribution::compute_row(tidx);
                    m_ = bidm * Cta_tile::M + row_;
                    bias_ptr_f32 =  reinterpret_cast<const uint32_t *>(params.bias_gmem);
                }
            }
    }

    // Pre store.
    template< typename Epilogue >
    inline __device__
    void pre_store(Epilogue &epilogue, int mi, int ii, Fragment_c &frag, int mask) {

	    if ( !DISABLE_BIAS && params_with_bias_ ) {
            int32_t offset = Tile_Distribution::compute_offset(mi, ii);
            int32_t bias_row = offset + m_;
            const uint32_t *bias_ptr_f32_tmp = bias_ptr_f32;
            bias_ptr_f32_tmp+=bias_row;

            if(bias_row < params_with_bias_) {
                uint32_t tmp;
                xmma::ldg(tmp, bias_ptr_f32_tmp);
                frag.add_single_bias(tmp);
            }
        }

        if ( !DISABLE_RELU ) {
            frag.relu(relu_lb_);
        }

        frag.output_colwert();
    }

    typename Traits::Epilogue_type relu_lb_, relu_ub_;
    Fragment_bias bias_;
    int32_t params_with_bias_;
    int32_t row_;
    int32_t m_;
    const uint32_t *bias_ptr_f32;
};

///////////////////////////////////////////////////////////////////////////////////////////////////

}
}  // namespace gemm
}  // namespace ext
} // namespace xmma