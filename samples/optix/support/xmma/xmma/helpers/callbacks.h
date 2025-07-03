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
#include <xmma/hopper/traits.h>

#include <type_traits>
#include <xmma/device_call.h>

namespace xmma {
namespace helpers {

template<typename Traits>
struct RT_FUSE_FILTER{
    static const  bool IS_RT_FUSE = false;
};

template<typename Input_type,
         typename Output_type,
         bool IS_GELU_,
         bool IS_EPIFADD_,
         bool IS_SWISH_ ,
         bool IS_RT_FUSE_>
struct RT_FUSE_FILTER<xmma::Ampere_imma_interleaved_traits<Input_type,
                                                    Output_type,
                                                    IS_GELU_,
                                                    IS_EPIFADD_,
                                                    IS_SWISH_,
                                                    IS_RT_FUSE_>>{
    static const bool IS_RT_FUSE = xmma::Ampere_imma_interleaved_traits<Input_type,
                                                    Output_type,
                                                    IS_GELU_,
                                                    IS_EPIFADD_,
                                                    IS_SWISH_,
                                                    IS_RT_FUSE_,
                                                    >::IS_RT_FUSE;
};

template<>
struct RT_FUSE_FILTER<xmma::Ampere_hmma_fp16_traits> {
    static const bool IS_RT_FUSE = xmma::Ampere_hmma_fp16_traits::IS_RT_FUSE;
};

template<>
struct RT_FUSE_FILTER<xmma::Ampere_hmma_fp32_traits> {
    static const bool IS_RT_FUSE = xmma::Ampere_hmma_fp32_traits::IS_RT_FUSE;
};
template<
    typename Traits,
    typename Cta_tile,
    typename Fragment_pre_swizzle_ = xmma::Fragment_epilogue_pre_swizzle<Traits, Cta_tile>,
    typename Fragment_post_swizzle_ = xmma::Fragment_epilogue_post_swizzle<Traits, Cta_tile>,
    typename Fragment_c_ = xmma::Fragment_c<Traits, Cta_tile>,
    bool DISABLE_BIAS = false,
    bool DISABLE_RELU = false,
    typename Layout = xmma::Row,
    // The number of bytes per STG.
    int BYTES_PER_STG_ = 16,
    bool IS_RT_FUSE = RT_FUSE_FILTER<Traits>::IS_RT_FUSE
>
struct Callbacks_epilogue_with_bias_and_relu
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
    // The number of bytes per STG.
    enum { BYTES_PER_STG = BYTES_PER_STG_ };

    // The different fragments.
    using Fragment_post_swizzle = Fragment_post_swizzle_;
    using Fragment_c = Fragment_c_;
    using C_type = typename Traits::C_type;
    using Epilogue_type = typename Traits::Epilogue_type;
    using Gmem_tile_epilogue_distribution =
        xmma::helpers::Gmem_tile_epilogue_distribution<Traits, Cta_tile,
        xmma::Row, BYTES_PER_STG>;
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;

    // The number of elements per STG per thread.
    enum { ELEMENTS_PER_STG = BYTES_PER_STG * 8 / Traits::BITS_PER_ELEMENT_C };

    // How many threads work on the contiguous dimension.
    enum { THREADS_PER_ROW = Min<Cta_tile::THREADS_PER_CTA,
                                 (Layout::ROW ? (Cta_tile::N / ELEMENTS_PER_STG)
                                             : (Cta_tile::M / ELEMENTS_PER_STG))>::VALUE };
    // How many cols covered by each STG for Layout::COL case.
    enum { COLS_PER_STG = Cta_tile::THREADS_PER_CTA / THREADS_PER_ROW };
    // For NCHW layout, bias reg number depends on N size per step and cols_per_stg
    // handled by all the warps.

    enum { BIAS_ELTS = Layout::ROW ? Fragment_c::NUM_ELTS
                                   : Xmma_tile::N_PER_XMMA_PER_CTA / COLS_PER_STG };

    // Whether this kernel is i8i8_i8i32_f32 type
    enum { IS_C_I8_EPILOGUE_F32 = std::is_same<int8_t, C_type>::value &&
                                  std::is_same<float, Epilogue_type>::value };

    // If kernel is i8i8_i8i32_f32 type, Fragment_bias will be set specially.
    // we need manually set type = Epilogue_type = f32 instead of C_type = i8.
    using Fragment_bias = typename std::conditional<IS_C_I8_EPILOGUE_F32,
                                                    Fragment<Epilogue_type, BIAS_ELTS>,
                                                    Fragment<C_type, BIAS_ELTS> >::type;

    template< typename Params >
    inline __device__ Callbacks_epilogue_with_bias_and_relu(const Params &params,
                                                            char *smem,
                                                            int bidm,
                                                            int bidn,
                                                            int bidz,
                                                            int tidx)
        : Base(params, smem, bidm, bidn, bidz, tidx) {
            if( !DISABLE_RELU ) {
                params_with_relu_ = params.with_relu;
                relu_lb_ = colwert<Epilogue_type>(params.relu_lb);
                relu_ub_ = colwert<Epilogue_type>(params.relu_ub);
            } else {
                if ( Traits::IS_GELU_ERF ) {
                    params_gelu_scale_ = *reinterpret_cast<const float *>(&params.runtime_params.runtime_param0);
                }
            }
            if( !DISABLE_BIAS ) {
                params_with_bias_ = params.with_bias;
                one_ = params.one;
                bias_ptr_ = reinterpret_cast<const uint32_t *>(params.bias_gmem);
                if( params.batch.is_batched && params.batch.batch_bias ) {
                    int batch = bidz; // FIXME: warp-specialized batch index
                    bias_ptr_ = reinterpret_cast<const uint32_t *>(
                        reinterpret_cast<const typename Fragment_bias::Data_type *>(params.bias_gmem)
                        + batch * params_with_bias_);
                }
                if( params_with_bias_ ) {
                    if( Layout::ROW ) {
                        col_ = Gmem_tile_epilogue_distribution::compute_col(tidx);
                        int32_t n = bidn * Cta_tile::N + col_ * ELEMENTS_PER_STG;
                        const uint4 *bias_ptr =  reinterpret_cast<const uint4 *>(bias_ptr_);
                        if( IS_C_I8_EPILOGUE_F32 ) {
                            // load bias specially for i8i8_i8i32_f32 kernel
                            // 4 comes from Epilogue_type_bit / C_type_bit = 32 / 8
                            // (C_type_bit determinates ELEMENTS_PER_STG)
                            bias_ptr += 4 * n / ELEMENTS_PER_STG;
                            if( n < params_with_bias_ ) {
                                #pragma unroll
                                for( int i = 0; i < 4; i++ ) {
                                    uint4 tmp;
                                    xmma::ldg(tmp, bias_ptr + i);
                                    bias_.reg(4 * i + 0) = tmp.x;
                                    bias_.reg(4 * i + 1) = tmp.y;
                                    bias_.reg(4 * i + 2) = tmp.z;
                                    bias_.reg(4 * i + 3) = tmp.w;
                                }
                            }
                        } else {
                            bias_ptr += n / ELEMENTS_PER_STG;
                            if( n < params_with_bias_ ) {
                                uint4 tmp;
                                xmma::ldg(tmp, bias_ptr);
                                bias_.reg(0) = tmp.x;
                                bias_.reg(1) = tmp.y;
                                bias_.reg(2) = tmp.z;
                                bias_.reg(3) = tmp.w;
                            }
                        } // end if IS_C_I8_EPILOGUE_F32
                    } else {
                        col_ = tidx / THREADS_PER_ROW + bidn * Cta_tile::N;
                        bias_ptr_ += col_;
                    } // end if Layout::Row
                } else {
                    bias_.clear();
                }// end if params_with_bias_
            } // end if DISABLE_BIAS
            if (params.batch.is_batched && params.batch.batch_scaling) {
                int batch = bidz; // FIXME: Add support for warp-specialized batch
                this->alpha_ = reinterpret_cast<const Epilogue_type *>(params.alpha_gmem)[batch];
                this->beta_  = reinterpret_cast<const Epilogue_type *>(params.beta_gmem)[batch];
            }
    }

    // For column major use case, we need to reload bias for each step.
    // This is only called in column major cases.
    inline __device__ void load_bias_col(int mi) {
        int32_t col = col_ + Fragment_bias::NUM_REGS * COLS_PER_STG * mi;
        const uint32_t* bias_ptr = bias_ptr_ + Fragment_bias::NUM_REGS * COLS_PER_STG * mi;
        for( int i = 0; i < Fragment_bias::NUM_REGS; ++i ) {
            int n = col + i * COLS_PER_STG;
            if( n < params_with_bias_ ) {
                xmma::ldg(bias_.reg(i), bias_ptr);
            }
            bias_ptr += COLS_PER_STG;
        }
    }

    // Post swizzle.
    template< typename Epilogue >
    inline __device__ void post_swizzle(Epilogue &epilogue,
                                        int mi,
                                        int ii,
                                        Fragment_post_swizzle &frag,
                                        int mask) {
        if ( !DISABLE_BIAS ) {
            if( Layout::ROW ) {
                frag.add_bias(this->bias_);
            } else {
                if( ii == 0 ) {
                    // Load once for every step.
                    load_bias_col(mi);
                }
                frag.add_bias_nchw(this->bias_, ii % BIAS_ELTS);
            }
        }
    }

    // Pre Pack.
    template< typename Epilogue >
    inline __device__ void pre_pack(Epilogue &epilogue,
                                    int mi, int ii,
                                    Fragment_post_swizzle &frag) {
        if ( !DISABLE_RELU ) {
            frag.relu(relu_lb_);
            frag.relu_ub(relu_ub_);
        } else {
            if ( Traits::IS_GELU_ERF ) {
                frag.gelu_erf(params_gelu_scale_);
            }
        }

#ifdef CASK_SDK_CASK_PLUGIN_LINK
        if (IS_RT_FUSE){
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
        }
#endif
    }

    // Pre store.
    template< typename Epilogue >
    inline __device__
    void pre_store(Epilogue &epilogue, int mi, int ii, Fragment_c &frag, int mask) {
        // bias+relu fusion.
        if( !DISABLE_BIAS && !DISABLE_RELU ) {
            if( Layout::ROW ) {
                frag.add_bias_relu(this->bias_, params_with_relu_, relu_lb_, one_);
                frag.relu_ub(relu_ub_);
            } else {
                frag.add_bias_nchw(this->bias_, ii % BIAS_ELTS);
                if( params_with_relu_ ) {
                    frag.relu(relu_lb_);
                    frag.relu_ub(relu_ub_);
                }
            }
        } else {
            if( !DISABLE_BIAS && params_with_bias_ ) {
                if( Layout::ROW ) {
                    frag.add_bias(this->bias_);
                } else {
                    frag.add_bias_nchw(this->bias_, ii % BIAS_ELTS);
                }
            }
            if( !DISABLE_RELU && params_with_relu_ ) {
                frag.relu(relu_lb_);
                frag.relu_ub(relu_ub_);
            }
        }
    }
    Epilogue_type relu_lb_, relu_ub_;
    Fragment_bias bias_;
    int32_t params_with_bias_;
    int32_t params_with_relu_;
    float params_gelu_scale_;

    // WAR: To fuse HADD2+relu, we have to feed the explicit half(1.f) to hfma2 ptx.
    // OCG would fix it in LWCA 11.1.
    float one_;
    // Index of bias
    int32_t col_;
    const uint32_t* bias_ptr_;
};

template <typename Traits,
          typename Cta_tile,
          typename Fragment_pre_swizzle_ = xmma::Fragment_epilogue_pre_swizzle<Traits, Cta_tile>,
          typename Fragment_post_swizzle_ = xmma::Fragment_epilogue_post_swizzle<Traits, Cta_tile>,
          typename Fragment_c_ = xmma::Fragment_c<Traits, Cta_tile>,
          bool DISABLE_BIAS = false,
          bool DISABLE_RELU = false,
          typename Layout = xmma::Row,
          // The number of bytes per STG.
          int BYTES_PER_STG_ = 16,
          bool IS_RT_FUSE = RT_FUSE_FILTER<Traits>::IS_RT_FUSE>
struct Callbacks_epilogue_with_bias_relu_per_channel_scaling
    : public Callbacks_epilogue_with_bias_and_relu<Traits,
                                                   Cta_tile,
                                                   Fragment_pre_swizzle_,
                                                   Fragment_post_swizzle_,
                                                   Fragment_c_,
                                                   DISABLE_BIAS,
                                                   DISABLE_RELU,
                                                   Layout,
                                                   BYTES_PER_STG_,
                                                   IS_RT_FUSE> {
    // The base class.
    using Base = Callbacks_epilogue_with_bias_and_relu<Traits,
                                                       Cta_tile,
                                                       Fragment_pre_swizzle_,
                                                       Fragment_post_swizzle_,
                                                       Fragment_c_,
                                                       DISABLE_BIAS,
                                                       DISABLE_RELU,
                                                       Layout,
                                                       BYTES_PER_STG_,
                                                       IS_RT_FUSE>;

    // The number of bytes per STG.
    enum { BYTES_PER_STG = BYTES_PER_STG_ };

    // The different fragments.
    using Fragment_pre_swizzle = Fragment_pre_swizzle_;
    using Fragment_post_swizzle = Fragment_post_swizzle_;
    using Fragment_c = Fragment_c_;
    using C_type = typename Traits::C_type;
    using Epilogue_type = typename Traits::Epilogue_type;
    using Gmem_tile_epilogue_distribution =
        xmma::helpers::Gmem_tile_epilogue_distribution<Traits, Cta_tile, xmma::Row, BYTES_PER_STG>;
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;

    // The number of elements per STG per thread.
    enum { ELEMENTS_PER_STG = BYTES_PER_STG * 8 / Traits::BITS_PER_ELEMENT_C };

    // How many threads work on the contiguous dimension.
    enum {
        THREADS_PER_ROW = Min<Cta_tile::THREADS_PER_CTA,
                              ( Layout::ROW ? ( Cta_tile::N / ELEMENTS_PER_STG )
                                            : ( Cta_tile::M / ELEMENTS_PER_STG ) )>::VALUE
    };
    // How many cols covered by each STG for Layout::COL case.
    enum { COLS_PER_STG = Cta_tile::THREADS_PER_CTA / THREADS_PER_ROW };
    // For NCHW layout, bias reg number depends on N size per step and cols_per_stg
    // handled by all the warps.

    enum {
        BIAS_ELTS =
            Layout::ROW ? Fragment_c::NUM_ELTS : Xmma_tile::N_PER_XMMA_PER_CTA / COLS_PER_STG
    };

    // Whether this kernel is i8i8_i8i32_f32 type
    enum {
        IS_C_I8_EPILOGUE_F32 =
            lwca::is_same<int8_t, C_type>::value && std::is_same<float, Epilogue_type>::value
    };
    // Per channel scaling is only enabled for int8 kernels.
    enum { PER_CHANNEL_SCALING = IS_C_I8_EPILOGUE_F32 };

    // If kernel is i8i8_i8i32_f32 type, Fragment_bias will be set specially.
    // we need manually set type = Epilogue_type = f32 instead of C_type = i8.
    using Fragment_bias = typename lwca::conditional<IS_C_I8_EPILOGUE_F32,
                                                     Fragment<Epilogue_type, BIAS_ELTS>,
                                                     Fragment<C_type, BIAS_ELTS>>::type;
    using Fragment_scaling = Fragment_bias;

    using Fragment_alpha_pre_swizzle = xmma::Fragment<float, Fragment_pre_swizzle::NUM_ELTS>;
    using Fragment_alpha_post_swizzle = xmma::Fragment<float, Fragment_post_swizzle::NUM_ELTS>;
    using Fragment_beta = Fragment_alpha_post_swizzle;

    template <typename Params>
    inline __device__ Callbacks_epilogue_with_bias_relu_per_channel_scaling( const Params &params,
                                                                             char *smem,
                                                                             int bidm,
                                                                             int bidn,
                                                                             int bidz,
                                                                             int tidx )
        : Base( params, smem, bidm, bidn, bidz, tidx ) {
        if( IS_C_I8_EPILOGUE_F32 ) {
            const float *alpha_ptr = reinterpret_cast<const float *>( params.alpha_gmem );
            const float *beta_ptr = reinterpret_cast<const float *>( params.beta_gmem );
            int32_t batch = bidz;  // FIXME: Add support for warp-specialized batch
            if( params.batch.is_batched && params.batch.batch_scaling ) {
                if( params.per_channel_scaling ) {
                    alpha_ptr =
                        &reinterpret_cast<const float *>( params.alpha_gmem )[batch * params.n];
                    beta_ptr =
                        &reinterpret_cast<const float *>( params.beta_gmem )[batch * params.n];
                } else {
                    alpha_ptr = &reinterpret_cast<const float *>( params.alpha_gmem )[batch];
                    beta_ptr = &reinterpret_cast<const float *>( params.beta_gmem )[batch];
                }
            }
            for( int32_t i = 0; i < Fragment_scaling::NUM_ELTS; i++ ) {
                frag_alpha_.elt( i ) = this->alpha_;
                frag_beta_.elt( i ) = this->beta_;
            }
            if( !params.per_channel_scaling ) {
                if( params.batch.is_batched && params.batch.batch_scaling ) {
                    // Set alpha.
                    #pragma unroll
                    for( int32_t i = 0; i < Fragment_scaling::NUM_ELTS; i++ ) {
                        frag_alpha_.elt( i ) = alpha_ptr[0];
                    }
                    // Set beta.
                    if( params.with_residual ) {
                        #pragma unroll
                        for( int32_t i = 0; i < Fragment_scaling::NUM_ELTS; i++ ) {
                            frag_beta_.elt( i ) = beta_ptr[0];
                        }
                    }
                }
            } else {
                if( Layout::ROW ) {
                    int32_t col = Gmem_tile_epilogue_distribution::compute_col( tidx );
                    int32_t n = bidn * Cta_tile::N + col * ELEMENTS_PER_STG;
                    const uint4 *alpha_p = reinterpret_cast<const uint4 *>( &alpha_ptr[n] );
                    const uint4 *beta_p = reinterpret_cast<const uint4 *>( &beta_ptr[n] );
                    if( n < params.n ) {
                        #pragma unroll
                        for( int i = 0; i < 4; i++ ) {
                            uint4 tmp;
                            xmma::ldg( tmp, alpha_p + i );
                            frag_alpha_.reg( 4 * i + 0 ) = tmp.x;
                            frag_alpha_.reg( 4 * i + 1 ) = tmp.y;
                            frag_alpha_.reg( 4 * i + 2 ) = tmp.z;
                            frag_alpha_.reg( 4 * i + 3 ) = tmp.w;
                        }

                        if( params.with_residual ) {
                            #pragma unroll
                            for( int i = 0; i < 4; i++ ) {
                                uint4 tmp;
                                xmma::ldg( tmp, beta_p + i );
                                frag_beta_.reg( 4 * i + 0 ) = tmp.x;
                                frag_beta_.reg( 4 * i + 1 ) = tmp.y;
                                frag_beta_.reg( 4 * i + 2 ) = tmp.z;
                                frag_beta_.reg( 4 * i + 3 ) = tmp.w;
                            }
                        }
                    }
                }
            }
        }
    }

    // A callback function to get alpha.
    template <typename Epilogue>
    inline __device__ void
    alpha_pre_swizzle( Epilogue &epilogue, int mi, int ni, Fragment_alpha_pre_swizzle &frag ) {
    }

    template <typename Epilogue>
    inline __device__ void
    alpha_post_swizzle( Epilogue &epilogue, int mi, int ni, Fragment_alpha_post_swizzle &frag ) {
        #pragma unroll
        for( int i = 0; i < Fragment_alpha_post_swizzle::NUM_REGS; ++i ) {
            frag.elt( i ) = frag_alpha_.elt( i );
        }
    }

    // A callback function to get beta.
    template <typename Epilogue>
    inline __device__ void beta( Epilogue &epilogue, int mi, int ii, Fragment_beta &frag ) {
        #pragma unroll
        for( int i = 0; i < Fragment_beta::NUM_REGS; ++i ) {
            frag.elt( i ) = frag_beta_.elt( i );
        }
    }

    Fragment_scaling frag_alpha_, frag_beta_;
};
////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits,
          typename Cta_tile,
          typename Fragment_pre_swizzle_ = xmma::Fragment_epilogue_pre_swizzle<Traits, Cta_tile>,
          typename Fragment_post_swizzle_ = xmma::Fragment_epilogue_post_swizzle<Traits, Cta_tile>,
          typename Fragment_c_ = xmma::Fragment_c<Traits, Cta_tile>,
          bool DISABLE_BIAS = false,
          bool DISABLE_RELU = false,
          typename Layout = xmma::Row,
          // The number of bytes per STG.
          int BYTES_PER_STG_ = 16>
struct Callbacks_gmma_epilogue_with_bias_and_relu
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
    // The number of bytes per STG.
    enum { BYTES_PER_STG = BYTES_PER_STG_ };

    // The different fragments.
    using Fragment_post_swizzle = Fragment_post_swizzle_;
    using Fragment_c = Fragment_c_;
    using C_type = typename Traits::C_type;
    using Epilogue_type = typename Traits::Epilogue_type;
    using Gmem_tile_epilogue_distribution = xmma::helpers::
        Gmem_tile_gmma_epilogue_distribution<Traits, Cta_tile, xmma::Row, BYTES_PER_STG>;
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;

    // The number of elements per STG per thread.
    enum { ELEMENTS_PER_STG = BYTES_PER_STG * 8 / Traits::BITS_PER_ELEMENT_C };

    // The N dimension of one epilogue tile
    enum { TILE_N = Gmem_tile_epilogue_distribution::TILE_N };
    // How many threads work on the contiguous dimension.
    enum { THREADS_PER_ROW = TILE_N / ELEMENTS_PER_STG };
    // How many cols covered by each STG for Layout::COL case.
    enum { COLS_PER_STG = Cta_tile::THREADS_PER_CTA / THREADS_PER_ROW };
    // Number of bias elements
    enum { BIAS_ELTS = Fragment_c::NUM_ELTS };
    using Fragment_bias = Fragment<C_type, BIAS_ELTS>;

    // How many iterations on N dimension in one GMMA
    enum { ITERATIONS_N = Xmma_tile::N_PER_XMMA_PER_CTA / TILE_N };

    template <typename Params>
    inline __device__ Callbacks_gmma_epilogue_with_bias_and_relu( const Params &params,
                                                                  char *smem,
                                                                  int bidm,
                                                                  int bidn,
                                                                  int bidz,
                                                                  int tidx )
        : Base( params, smem, bidm, bidn, bidz, tidx ) {
        if( !DISABLE_RELU ) {
            params_with_relu_ = params.with_relu;
            relu_lb_ = colwert<Epilogue_type>( params.relu_lb );
            relu_ub_ = colwert<Epilogue_type>( params.relu_ub );
        }
        if( !DISABLE_BIAS ) {
            params_with_bias_ = params.with_bias;
            one_ = params.one;
            bias_ptr_ = reinterpret_cast<const uint32_t *>( params.bias_gmem );
            if( params_with_bias_ ) {
                if( Layout::ROW ) {
                    col_ = Gmem_tile_epilogue_distribution::compute_col( tidx );
                    const uint4 *bias_ptr = reinterpret_cast<const uint4 *>( bias_ptr_ );
                    int32_t n = bidn * Cta_tile::N + col_ * ELEMENTS_PER_STG;
                    bias_ptr += n / ELEMENTS_PER_STG;
                    #pragma unroll
                    for( int ni = 0; ni < ITERATIONS_N; ++ni ) {
                        if( n < params_with_bias_ ) {
                            uint4 tmp;
                            xmma::ldg( tmp, bias_ptr );
                            bias_[ni].reg( 0 ) = tmp.x;
                            bias_[ni].reg( 1 ) = tmp.y;
                            bias_[ni].reg( 2 ) = tmp.z;
                            bias_[ni].reg( 3 ) = tmp.w;
                        }
                        n += TILE_N;
                        bias_ptr += TILE_N / ELEMENTS_PER_STG;
                    }
                } else {
                    col_ = tidx / THREADS_PER_ROW + bidn * Cta_tile::N;
                    bias_ptr_ += col_;
                }  // end if Layout::Row
            } else {
                #pragma unroll
                for( int ni = 0; ni < ITERATIONS_N; ++ni ) {
                    bias_[ni].clear();
                }
            }  // end if params_with_bias_
        }      // end if DISABLE_BIAS
    }

    // For column major use case, we need to reload bias for each step.
    // This is only called in column major cases.
    inline __device__ void load_bias_col( int mi, int ni ) {
        int32_t col = col_ + Fragment_bias::NUM_REGS * COLS_PER_STG * mi;
        const uint32_t *bias_ptr = bias_ptr_ + Fragment_bias::NUM_REGS * COLS_PER_STG * mi;
        for( int i = 0; i < Fragment_bias::NUM_REGS; ++i ) {
            int n = col + i * COLS_PER_STG;
            if( n < params_with_bias_ ) {
                xmma::ldg( bias_[ni].reg( i ), bias_ptr );
            }
            bias_ptr += COLS_PER_STG;
        }
    }

    // Post swizzle.
    template <typename Epilogue>
    inline __device__ void post_swizzle( Epilogue &epilogue,
                                         int mi,
                                         int ni,
                                         int ii,
                                         Fragment_post_swizzle &frag,
                                         int mask ) {
    }

    // Pre Pack.
    template <typename Epilogue>
    inline __device__ void
    pre_pack( Epilogue &epilogue, int mi, int ni, int ii, Fragment_post_swizzle &frag ) {
        if( !DISABLE_BIAS && params_with_bias_ ) {
            if( Layout::ROW ) {
                frag.add_bias( this->bias_[ni] );
            } else {
                if( ii == 0 ) {
                    // Load once for every step.
                    load_bias_col( mi, ni );
                }
                frag.add_bias_nchw( this->bias_, ii % BIAS_ELTS );
            }
        }
        if( !DISABLE_RELU && params_with_relu_ ) {
            frag.relu( relu_lb_ );
            frag.relu_ub( relu_ub_ );
        }
    }

    // Pre store.
    template <typename Epilogue>
    inline __device__ void
    pre_store( Epilogue &epilogue, int mi, int ni, int ii, Fragment_c &frag, int mask ) {
        // bias+relu fusion.
        if( !DISABLE_BIAS && !DISABLE_RELU ) {
            if( Layout::ROW ) {
                frag.add_bias_relu( this->bias_[ni], params_with_relu_, relu_lb_, one_ );
                frag.relu_ub( relu_ub_ );
            } else {
                frag.add_bias_nchw( this->bias_[ni], ii % BIAS_ELTS );
                if( params_with_relu_ ) {
                    frag.relu( relu_lb_ );
                    frag.relu_ub( relu_ub_ );
                }
            }
        } else {
            if( !DISABLE_BIAS && params_with_bias_ ) {
                if( Layout::ROW ) {
                    frag.add_bias( this->bias_[ni] );
                } else {
                    frag.add_bias_nchw( this->bias_, ii % BIAS_ELTS );
                }
            }
            if( !DISABLE_RELU && params_with_relu_ ) {
                frag.relu( relu_lb_ );
                frag.relu_ub( relu_ub_ );
            }
        }
    }
    Epilogue_type relu_lb_, relu_ub_;
    Fragment_bias bias_[ITERATIONS_N];
    int32_t params_with_bias_;
    int32_t params_with_relu_;

    // WAR: To fuse HADD2+relu, we have to feed the explicit half(1.f) to hfma2 ptx.
    // OCG would fix it in LWCA 11.1.
    float one_;
    // Index of bias
    int32_t col_;
    const uint32_t* bias_ptr_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    typename Traits,
    typename Cta_tile,
    typename Fragment_pre_swizzle_ = xmma::Fragment_epilogue_pre_swizzle<Traits, Cta_tile>,
    typename Fragment_post_swizzle_ = xmma::Fragment_epilogue_post_swizzle<Traits, Cta_tile>,
    typename Fragment_c_ = xmma::Fragment_c<Traits, Cta_tile>,
    typename Layout = xmma::Row,
    int BYTES_PER_STG_ = 16
>
struct Callbacks_wo_smem_epilogue
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
    // The number of bytes per STG.
    enum { BYTES_PER_STG = BYTES_PER_STG_ };
    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;
    // The type and num_elts of one Fragment_bias.
    enum { BIAS_ELTS = Fragment_c_::NUM_ELTS };
    // The number of elements per STG per thread.
    // NOTE: Here use BITS_PER_ELEMENT_C, maybe Epilogue better.
    enum { ELEMENTS_PER_STG = BYTES_PER_STG * 8 / Traits::BITS_PER_ELEMENT_C };
    enum { XMMAS_N = Xmma_tile::XMMAS_N };
    enum { N_PER_XMMA_PER_CTA = Xmma_tile::N_PER_XMMA_PER_CTA };

    using Fragment_bias = Fragment<typename Traits::Epilogue_type, BIAS_ELTS>;

    template< typename Params >
    inline __device__ Callbacks_wo_smem_epilogue(const Params &params,
                                                 char *smem,
                                                 int bidm,
                                                 int bidn,
                                                 int bidz,
                                                 int tidx)
        : Base(params, smem, bidm, bidn, bidz, tidx) {
            // load bias
            params_is_with_relu_ = params.with_relu;
            relu_lb_ = colwert<typename Traits::Epilogue_type>(params.relu_lb);
            relu_ub_ = colwert<typename Traits::Epilogue_type>(params.relu_ub);
            params_bias_elt_num_ = params.with_bias;
            bias_ptr_ = reinterpret_cast<const uint32_t *>(params.bias_gmem);
            one_ = params.one;

            if( params.batch.is_batched && params.batch.batch_bias ) {
                int batch = bidz;
                bias_ptr_ = reinterpret_cast<const uint32_t *>(
                    reinterpret_cast<const typename Fragment_bias::Data_type *>(params.bias_gmem)
                    + batch * params_bias_elt_num_);
            }

            if( params_bias_elt_num_ ) {
                if( Layout::ROW ) {
                    const uint4 *bias_ptr =  reinterpret_cast<const uint4 *>(bias_ptr_);
                    // the position of bias is computed the same as col in:
                    // Gmem_tile_wo_smem_epilogue::Gmem_tile_wo_smem_epilogue()
                    const int WARPS_M = Cta_tile::WARPS_M;
                    const int WARPS_N = Cta_tile::WARPS_N;
                    const int WARPS_K = Cta_tile::WARPS_K;
                    const int WARP_MASK_N = xmma::Warp_masks<WARPS_M, WARPS_N, WARPS_K>::N;
                    const int WARP_DIV_N = WARPS_M * 1 * Cta_tile::THREADS_PER_WARP;
                    // the second part after addition seems only work for f64.
                    // need double check if want supporting other type.
                    int col = ((tidx & WARP_MASK_N) / WARP_DIV_N) * Xmma_tile::N_PER_XMMA + (tidx % 4) * 2;
                    int n_ = bidn * Cta_tile::N + col;

                    #pragma unroll
                    for( int i = 0; i < XMMAS_N; i++ ) {
                        int offset = n_ + i * N_PER_XMMA_PER_CTA;
                        if( offset < params_bias_elt_num_ ) {
                            uint4 tmp;
                            xmma::ldg(tmp, bias_ptr + offset / ELEMENTS_PER_STG);
                            bias_[i].reg(0) = tmp.x;
                            bias_[i].reg(1) = tmp.y;
                            bias_[i].reg(2) = tmp.z;
                            bias_[i].reg(3) = tmp.w;
                        }
                    }
                } else {
                    // if necessary, add support for Layout::COL.
                }
            } else {
                #pragma unroll
                for( int i = 0; i < XMMAS_N; i++ ) {
                    bias_[i].clear();
                }
            } // end if params_bias_elt_num_
            if (params.batch.is_batched && params.batch.batch_scaling) {
                int batch = bidz; // FIXME: Add support for warp-specialized batch
                this->alpha_ = reinterpret_cast<const typename Traits::Epilogue_type *>(params.alpha_gmem)[batch];
                this->beta_  = reinterpret_cast<const typename Traits::Epilogue_type *>(params.beta_gmem)[batch];
            }
    }

    // Pre store.
    template< typename Epilogue>
    inline __device__ void pre_store(Epilogue &epilogue, int mi, int ni, Fragment_c_ &frag) {
        if( Layout::ROW ) {
            frag.add_bias_relu(this->bias_[ni], params_is_with_relu_, relu_lb_, one_);
        } else {
            // if necessary, add support for Layout::COL.
        }
    }

    typename Traits::Epilogue_type relu_lb_, relu_ub_;
    int32_t params_bias_elt_num_;
    int32_t params_is_with_relu_;
    const uint32_t* bias_ptr_;
    // each thread holds XMMAS_M*XMMAS_N acc and XMMAS_N different Fragment_bias.
    Fragment_bias bias_[XMMAS_N];
    float one_;
};

///////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace helpers
}  // namespace xmma
