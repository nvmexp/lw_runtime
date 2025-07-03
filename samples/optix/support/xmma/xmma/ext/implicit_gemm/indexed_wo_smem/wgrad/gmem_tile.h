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
#include <xmma/warp_masks.h>
#include <xmma/ext/implicit_gemm/indexed_wo_smem/helpers/epilogue.h>

#include <xmma/ext/implicit_gemm/indexed_wo_smem/wgrad/utils.h>

namespace xmma {
namespace ext {
namespace implicit_gemm {
namespace indexed_wo_smem {
namespace wgrad {

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// T I L E   A
//
///////////////////////////////////////////////////////////////////////////////////////////////////

template <
    // The instruction traits.
    typename Traits,
    // The CTA tile description.
    typename Cta_tile,
    // ELEMENT_PER_LDG
    int ELTS_PER_LDG_,
    // ELEMENT_PER_STG
    int ELTS_PER_STG_>
struct Gmem_tile_a {

    enum { HAS_SUPER_HMMA = Traits::Gpu_arch::HAS_SUPER_HMMA };

    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;
    using Fragment_layout = typename Fragment_layout<HAS_SUPER_HMMA>::Layout_a;
    using Fragment = xmma::Fragment_a<Traits, Fragment_layout>;

    // The number of elements per LDG.
    enum { ELTS_PER_LDG = ELTS_PER_LDG_ };
    // The number of elements per STG.
    enum { ELTS_PER_STG = ELTS_PER_STG_ };

    enum { BYTES_PER_LDG = ELTS_PER_LDG * Traits::BITS_PER_ELEMENT_A / 8 };
    enum { BYTES_PER_STG = ELTS_PER_STG * Traits::BITS_PER_ELEMENT_A / 8 };

    enum { THREADS_PER_ROW = 4 };
    enum { THREADS_PER_COL = HAS_SUPER_HMMA ? 8 : 4 };
    enum { LDGS_K = Xmma_tile::K_PER_XMMA / THREADS_PER_ROW };
    enum { LDGS_M = xmma::Div_up<Xmma_tile::M_PER_XMMA, THREADS_PER_COL * ELTS_PER_STG>::VALUE };

    // Ctor.
    template <typename Params>
    inline __device__ Gmem_tile_a( const Params &params, void *smem, const dim3 &bidx, int tidx )
        : ptr_( reinterpret_cast<const char *>( params.flt_gmem ) ),
          params_k_( params.k * params.g ), params_nopq_( params.nopq ), tiles_k_( params.tiles_k ),
          smem_( xmma::get_smem_pointer( smem ) ) {

        const int WARPS_M = Cta_tile::WARPS_M;
        const int WARPS_N = Cta_tile::WARPS_N;
        const int WARPS_K = Cta_tile::WARPS_K;
        // The masks to select the warps.
        const int WARP_MASK_M = xmma::Warp_masks<WARPS_M, WARPS_N, WARPS_K>::M;
        // The divisor for the warps.
        const int WARP_DIV_M = 1 * 1 * Cta_tile::THREADS_PER_WARP;

        k_ = bidx.x * Cta_tile::M + ( tidx & WARP_MASK_M ) / WARP_DIV_M * Xmma_tile::M_PER_WARP +
             ( HAS_SUPER_HMMA ? ( ( tidx & 0x1c ) / 4 )
                              : ( ( ( tidx & 0x10 ) >> 3 ) + ( ( tidx & 0x04 ) >> 2 ) ) ) *
                 ELTS_PER_STG;

        nopq_ = bidx.z * Cta_tile::K + ( tidx % 4 );

        params_nopq_ -= nopq_;

        ptr_ += Traits::offset_in_bytes_a( nopq_ * params_k_ + k_ );
    }

    // Load a tile from global.
    template <typename Params>
    inline __device__ void load( Fragment ( &a )[Xmma_tile::XMMAS_K][Xmma_tile::XMMAS_M],
                                 const Params &params ) {
        if( ELTS_PER_STG == 1 ) {
            load_1( a );
        } else if( ELTS_PER_STG == 2 ) {
            load_2( a );
        } else if( ELTS_PER_STG == 4 ) {
            load_4( a );
        } else {
        }
    }

    inline __device__ void load_4( Fragment ( &a )[Xmma_tile::XMMAS_K][Xmma_tile::XMMAS_M] ) {
        #pragma unroll
        for( int ki = 0; ki < Xmma_tile::XMMAS_K; ki++ ) {
            #pragma unroll
            for( int j = 0; j < LDGS_K; j++ ) {
                int nopq = j * 4 + ki * Xmma_tile::K_PER_XMMA;

                int offset = nopq * params_k_;
                if( HAS_SUPER_HMMA ) {
                    #pragma unroll
                    for( int mi = 0; mi < Xmma_tile::XMMAS_M; mi += 2 ) {
                        uint16_t *tmp_a_1 = &( a[ki][mi].u16( 0 ) );
                        uint16_t *tmp_a_2 = &( a[ki][mi + 1].u16( 0 ) );
                        const int k = mi * 16;
                        const char *ptr = ptr_ + Traits::offset_in_bytes_a( offset + k );
                        if( nopq < params_nopq_ && ( k_ + k ) < params_k_ ) {
                            if( BYTES_PER_STG == 8 ) {
                                uint2 tmp;
                                xmma::ldg( tmp, ptr );
                                tmp_a_1[( j / 2 ) * 4 + 0 * 2 + j % 2] =
                                    ( (uint16_t *)( &tmp ) )[0];
                                tmp_a_1[( j / 2 ) * 4 + 1 * 2 + j % 2] =
                                    ( (uint16_t *)( &tmp ) )[1];
                                tmp_a_2[( j / 2 ) * 4 + 0 * 2 + j % 2] =
                                    ( (uint16_t *)( &tmp ) )[2];
                                tmp_a_2[( j / 2 ) * 4 + 1 * 2 + j % 2] =
                                    ( (uint16_t *)( &tmp ) )[3];
                            } else if( BYTES_PER_STG == 16 ) {
                                uint4 tmp;
                                xmma::ldg( tmp, ptr );
                                a[ki][mi].reg( j * 2 + 0 ) = tmp.x;
                                a[ki][mi].reg( j * 2 + 1 ) = tmp.y;
                                a[ki][mi + 1].reg( j * 2 + 0 ) = tmp.z;
                                a[ki][mi + 1].reg( j * 2 + 1 ) = tmp.w;
                            }
                        } else {
                            if( BYTES_PER_STG == 8 ) {
                                tmp_a_1[( j / 2 ) * 4 + 0 * 2 + j % 2] = 0;
                                tmp_a_1[( j / 2 ) * 4 + 1 * 2 + j % 2] = 0;
                                tmp_a_2[( j / 2 ) * 4 + 0 * 2 + j % 2] = 0;
                                tmp_a_2[( j / 2 ) * 4 + 1 * 2 + j % 2] = 0;
                            } else if( BYTES_PER_STG == 16 ) {
                                a[ki][mi].reg( j * 2 + 0 ) = 0;
                                a[ki][mi].reg( j * 2 + 1 ) = 0;
                                a[ki][mi + 1].reg( j * 2 + 0 ) = 0;
                                a[ki][mi + 1].reg( j * 2 + 1 ) = 0;
                            }
                        }
                    }
                } else {
                    #pragma unroll
                    for( int mi = 0; mi < Xmma_tile::XMMAS_M; mi++ ) {
                        uint2 *tmp_a = (uint2 *)( &( a[ki][mi].u16( 0 ) ) );
                        const int k = mi * 16;
                        const char *ptr = ptr_ + Traits::offset_in_bytes_a( offset + k );
                        if( nopq < params_nopq_ && ( k_ + k ) < params_k_ ) {
                            xmma::ldg( tmp_a[j], ptr );
                        } else {
                            tmp_a[j] = make_uint2( 0, 0 );
                        }
                    }
                }
            }
        }
    }

    inline __device__ void load_2( Fragment ( &a )[Xmma_tile::XMMAS_K][Xmma_tile::XMMAS_M] ) {
        #pragma unroll
        for( int ki = 0; ki < Xmma_tile::XMMAS_K; ki++ ) {
            #pragma unroll
            for( int j = 0; j < LDGS_K; j++ ) {
                int nopq = j * 4 + ki * Xmma_tile::K_PER_XMMA;
                int offset = nopq * params_k_;
                #pragma unroll
                for( int mi = 0; mi < Xmma_tile::XMMAS_M; mi++ ) {
                    if( HAS_SUPER_HMMA ) {
                        const int k = mi * 8 * ELTS_PER_STG;
                        // const int k = mi * 2;
                        uint16_t *tmp_a = &( a[ki][mi].u16( 0 ) );
                        const char *ptr = ptr_ + Traits::offset_in_bytes_a( offset + k );
                        if( nopq < params_nopq_ && ( k_ + k ) < params_k_ ) {
                            if( BYTES_PER_STG == 4 ) {
                                uint32_t tmp;
                                xmma::ldg( tmp, ptr );
                                tmp_a[( j / 2 ) * 4 + 0 * 2 + j % 2] = ( (uint16_t *)( &tmp ) )[0];
                                tmp_a[( j / 2 ) * 4 + 1 * 2 + j % 2] = ( (uint16_t *)( &tmp ) )[1];
                            } else if( BYTES_PER_STG == 8 ) {
                                uint2 tmp;
                                xmma::ldg( tmp, ptr );
                                a[ki][mi].reg( j * 2 + 0 ) = tmp.x;
                                a[ki][mi].reg( j * 2 + 1 ) = tmp.y;
                            }
                        } else {
                            if( BYTES_PER_STG == 4 ) {
                                tmp_a[( j / 2 ) * 4 + 0 * 2 + j % 2] = 0;
                                tmp_a[( j / 2 ) * 4 + 1 * 2 + j % 2] = 0;
                            } else if( BYTES_PER_STG == 8 ) {
                                a[ki][mi].reg( j * 2 + 0 ) = 0;
                                a[ki][mi].reg( j * 2 + 1 ) = 0;
                            }
                        }
                    } else {
                        uint32_t *tmp_a = (uint32_t *)( &( a[ki][mi].u16( 0 ) ) );
                        for( int i = 0; i < LDGS_M; ++i ) {
                            const int k = mi * 16 + i * 8;
                            const char *ptr = ptr_ + Traits::offset_in_bytes_a( offset + k );
                            if( nopq < params_nopq_ && ( k_ + k ) < params_k_ ) {
                                xmma::ldg( tmp_a[j * 2 + i], ptr );
                            } else {
                                tmp_a[j * 2 + i] = 0;
                            }
                        }
                    }
                }
            }
        }
    }

    inline __device__ void load_1( Fragment ( &a )[Xmma_tile::XMMAS_K][Xmma_tile::XMMAS_M] ) {

        #pragma unroll
        for( int ki = 0; ki < Xmma_tile::XMMAS_K; ki++ ) {
            #pragma unroll
            for( int j = 0; j < LDGS_K; j++ ) {
                int nopq = j * 4 + ki * Xmma_tile::K_PER_XMMA;
                int offset = nopq * params_k_;
                #pragma unroll
                for( int mi = 0; mi < Xmma_tile::XMMAS_M; mi++ ) {
                    half *tmp_a = (half *)&( a[ki][mi].u16( 0 ) );
                    if( HAS_SUPER_HMMA ) {
                        #pragma unroll
                        for( int i = 0; i < 2; i++ ) {
                            const int k = mi * 2 * 8 * ELTS_PER_STG + i * 8;
                            const char *ptr = ptr_ + Traits::offset_in_bytes_a( offset + k );
                            if( nopq < params_nopq_ && ( k_ + k ) < params_k_ ) {
                                if( BYTES_PER_STG == 2 ) {
                                    xmma::ldg( (uint16_t &)tmp_a[( j / 2 ) * 4 + i * 2 + j % 2],
                                               ptr );
                                } else if( BYTES_PER_STG == 4 ) {
                                    xmma::ldg( a[ki][mi].reg( j * 2 + i ), ptr );
                                }
                            } else {
                                if( BYTES_PER_STG == 2 ) {
                                    tmp_a[( j / 2 ) * 4 + i * 2 + j % 2] = 0;
                                } else if( BYTES_PER_STG == 4 ) {
                                    a[ki][mi].reg( j * 2 + i ) = 0;
                                }
                            }
                        }
                    } else {
                        for( int i = 0; i < LDGS_M; ++i ) {
                            const int k = mi * 16 + i * 4;
                            const char *ptr = ptr_ + Traits::offset_in_bytes_a( offset + k );
                            if( nopq < params_nopq_ && ( k_ + k ) < params_k_ ) {
                                xmma::ldg( (uint16_t &)tmp_a[j * 4 + i], ptr );
                            } else {
                                tmp_a[j * 4 + i] = 0;
                            }
                        }
                    }
                }
            }
        }
    }

    // Move the pointers and update the predicates for R2P/P2R.
    template <typename Params> inline __device__ void move( const Params &params ) {
        ptr_ +=
            Traits::offset_in_bytes_a( Cta_tile::K * params_k_ * static_cast<int64_t>( tiles_k_ ) );
        params_nopq_ -= Cta_tile::K * tiles_k_;
    }

    int params_k_, params_nopq_;
    uint32_t tiles_k_;
    const char *ptr_;
    uint32_t smem_;
    uint32_t smem_trsc_;
    uint32_t smem_trsc_lds_;
    int nopq_, k_;
    int n_[Xmma_tile::XMMAS_M][2];
    int o_[Xmma_tile::XMMAS_M][2];
    int p_[Xmma_tile::XMMAS_M][2];
    int q_[Xmma_tile::XMMAS_M][2];
};

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// T I L E   B
//
///////////////////////////////////////////////////////////////////////////////////////////////////

template <
    // The instruction traits.
    typename Traits,
    // The CTA tile description.
    typename Cta_tile,
    // ELEMENT_PER_LDG
    int ELTS_PER_LDG_,
    // ELEMENT_PER_STG
    int ELTS_PER_STG_>
struct Gmem_tile_b {

    enum { HAS_SUPER_HMMA = Traits::Gpu_arch::HAS_SUPER_HMMA };

    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;
    using Fragment_layout = typename Fragment_layout<HAS_SUPER_HMMA>::Layout_b;
    using Fragment = xmma::Fragment_b<Traits, Fragment_layout>;

    // The number of elements per LDG.
    enum { ELTS_PER_LDG = ELTS_PER_LDG_ };
    // The number of elements per STG.
    enum { ELTS_PER_STG = ELTS_PER_STG_ };

    enum { BYTES_PER_LDG = ELTS_PER_LDG * Traits::BITS_PER_ELEMENT_B / 8 };

    enum { THREADS_PER_ROW = HAS_SUPER_HMMA ? 8 : 4 };
    enum { THREADS_PER_COL = 4 };
    enum { LDGS_K = Xmma_tile::K_PER_XMMA / THREADS_PER_COL };
    enum { LDGS_N = xmma::Div_up<Xmma_tile::N_PER_XMMA, THREADS_PER_ROW * ELTS_PER_LDG>::VALUE };

    // Ctor.
    template <typename Params>
    inline __device__ Gmem_tile_b( const Params &params, void *smem, const dim3 &bidx, int tidx )
        : ptr_( reinterpret_cast<const char *>( params.img_gmem ) ),
          params_c_( params.c * params.g ), params_ndhw_( params.n * params.dhw ),
          params_dhwc_( params.dhwc ), params_hwc_( params.hwc ), params_hw_( params_hw_ ),
          params_wc_( params.wc ), params_n_( params.n ), params_d_( params.d ),
          params_h_( params.h ), params_w_( params.w ), params_opq_( params.opq ),
          params_mul_opq_( params.mul_opq ), params_shr_opq_( params.shr_opq ),
          params_pq_( params.pq ), params_mul_pq_( params.mul_pq ), params_shr_pq_( params.shr_pq ),
          params_q_( params.q ), params_mul_q_( params.mul_q ), params_shr_q_( params.shr_q ),
          params_stride_d_( params.stride[0] ), params_stride_h_( params.stride[1] ),
          params_stride_w_( params.stride[2] ), params_dilation_d_( params.dilation[0] ),
          params_dilation_h_( params.dilation[1] ), params_dilation_w_( params.dilation[2] ),
          params_pad_d_( params.pad[0][0] ), params_pad_h_( params.pad[1][0] ),
          params_pad_w_( params.pad[2][0] ), tiles_k_( params.tiles_k ),
          smem_( xmma::get_smem_pointer( smem ) ) {

        const int WARPS_M = Cta_tile::WARPS_M;
        const int WARPS_N = Cta_tile::WARPS_N;
        const int WARPS_K = Cta_tile::WARPS_K;
        // The masks to select the warps.
        const int WARP_MASK_N = xmma::Warp_masks<WARPS_M, WARPS_N, WARPS_K>::N;
        // The divisor for the warps.
        const int WARP_DIV_N = WARPS_M * 1 * Cta_tile::THREADS_PER_WARP;

        #pragma unroll
        for( int ni = 0; ni < Xmma_tile::XMMAS_N; ni++ ) {
            #pragma unroll
            for( int i = 0; i < LDGS_N; i++ ) {

                int trsc = bidx.y * Cta_tile::N +
                           ( tidx & WARP_MASK_N ) / WARP_DIV_N * Xmma_tile::N_PER_WARP +
                           ni * Xmma_tile::N_PER_XMMA +
                           ( HAS_SUPER_HMMA ? ( ( tidx & 0x1c ) / 4 + i * 8 ) * ELTS_PER_LDG
                                            : ( ( ( tidx & 0x18 ) >> 3 ) + i * 4 ) * ELTS_PER_LDG );

                int rsc, rs;
                xmma::fast_divmod(
                    t_[ni][i], rsc, trsc, params.rsc, params.mul_rsc, params.shr_rsc );
                xmma::fast_divmod( r_[ni][i], rs, rsc, params.sc, params.mul_sc, params.shr_sc );
                xmma::fast_divmod( s_[ni][i], c_[ni][i], rs, params.c, params.mul_c, params.shr_c );

                t_[ni][i] *= params_dilation_d_;
                r_[ni][i] *= params_dilation_h_;
                s_[ni][i] *= params_dilation_w_;
            }
        }

        smem_lds_ = smem_ + ( tidx % 4 ) * sizeof( int4 );

        smem_ += tidx * sizeof( int4 );

        nopq_ = bidx.z * Cta_tile::K + tidx;
        int n, opq, pq, o, p, q;
        xmma::fast_divmod( n, opq, nopq_, params_opq_, params_mul_opq_, params_shr_opq_ );
        xmma::fast_divmod( o, pq, opq, params_pq_, params_mul_pq_, params_shr_pq_ );
        xmma::fast_divmod( p, q, pq, params_q_, params_mul_q_, params_shr_q_ );

        o = o * params_stride_d_ - params_pad_d_;
        p = p * params_stride_h_ - params_pad_h_;
        q = q * params_stride_w_ - params_pad_w_;

        int4 indices = make_int4( n, o, p, q );
        xmma::sts( smem_, reinterpret_cast<const uint4 &>( indices ) );
        __syncthreads();
    }

    inline __device__ void move() {
        nopq_ += Cta_tile::K * tiles_k_;

        int n, opq, pq, o, p, q;
        xmma::fast_divmod( n, opq, nopq_, params_opq_, params_mul_opq_, params_shr_opq_ );
        xmma::fast_divmod( o, pq, opq, params_pq_, params_mul_pq_, params_shr_pq_ );
        xmma::fast_divmod( p, q, pq, params_q_, params_mul_q_, params_shr_q_ );

        o = o * params_stride_d_ - params_pad_d_;
        p = p * params_stride_h_ - params_pad_h_;
        q = q * params_stride_w_ - params_pad_w_;

        int4 indices = make_int4( n, o, p, q );
        xmma::sts( smem_, reinterpret_cast<const uint4 &>( indices ) );
    }
    // Load a tile from global.
    inline __device__ void load( Fragment ( &b )[Xmma_tile::XMMAS_K][Xmma_tile::XMMAS_N] ) {
        if( ELTS_PER_LDG == 1 ) {
            load_1( b );
        } else if( ELTS_PER_LDG == 2 ) {
            load_2( b );
        } else if( ELTS_PER_LDG == 4 ) {
        } else {
        }
    }

    // Load a tile from shared.
    inline __device__ void load_2( Fragment ( &b )[Xmma_tile::XMMAS_K][Xmma_tile::XMMAS_N] ) {

        #pragma unroll
        for( int ki = 0; ki < Xmma_tile::XMMAS_K; ki++ ) {
            #pragma unroll
            for( int j = 0; j < LDGS_K; j++ ) {
                int idx = j * 4 + ki * Xmma_tile::K_PER_XMMA;

                uint4 nopq;
                xmma::lds( nopq, smem_lds_ + idx * sizeof( uint4 ) );
                // Extract the coordinates of that pixel.
                int n = reinterpret_cast<const int &>( nopq.x );
                int o = reinterpret_cast<const int &>( nopq.y );
                int p = reinterpret_cast<const int &>( nopq.z );
                int q = reinterpret_cast<const int &>( nopq.w );

                #pragma unroll
                for( int ni = 0; ni < Xmma_tile::XMMAS_N; ni++ ) {
                    if( HAS_SUPER_HMMA ) {
                        uint16_t *tmp_b = &( b[ki][ni].u16( 0 ) );
                        // Compute the H and W coordinates.
                        int d = o + t_[ni][0];
                        int h = p + r_[ni][0];
                        int w = q + s_[ni][0];

                        int offset = n * params_dhwc_ + d * params_hwc_ + h * params_wc_ +
                                     w * params_c_ + c_[ni][0];
                        const char *ptr = ptr_ + Traits::offset_in_bytes_b( offset );

                        if( n < params_n_ && (unsigned)d < params_d_ && (unsigned)h < params_h_ &&
                            (unsigned)w < params_w_ && c_[ni][0] < params_c_ ) {
                            if( BYTES_PER_LDG == 4 ) {
                                uint32_t tmp;
                                xmma::ldg( tmp, ptr );
                                tmp_b[0 * LDGS_K + j] = ( (uint16_t *)( &tmp ) )[0];
                                tmp_b[1 * LDGS_K + j] = ( (uint16_t *)( &tmp ) )[1];
                            } else if( BYTES_PER_LDG == 8 ) {
                                uint2 tmp;
                                xmma::ldg( tmp, ptr );
                                b[ki][ni].reg( j + 0 ) = tmp.x;
                                b[ki][ni].reg( j + 2 ) = tmp.y;
                            }
                        } else {
                            if( BYTES_PER_LDG == 4 ) {
                                tmp_b[0 * LDGS_K + j] = 0;
                                tmp_b[1 * LDGS_K + j] = 0;
                            } else if( BYTES_PER_LDG == 8 ) {
                                b[ki][ni].reg( j + 0 ) = 0;
                                b[ki][ni].reg( j + 2 ) = 0;
                            }
                        }
                    } else {
                        uint32_t *tmp_b = (uint32_t *)( &( b[ki][ni].u16( 0 ) ) );
                        #pragma unroll
                        for( int i = 0; i < LDGS_N; ++i ) {
                            // Compute the H and W coordinates.
                            int d = o + t_[ni][i];
                            int h = p + r_[ni][i];
                            int w = q + s_[ni][i];

                            int offset = n * params_dhwc_ + d * params_hwc_ + h * params_wc_ +
                                         w * params_c_ + c_[ni][i];
                            const char *ptr = ptr_ + Traits::offset_in_bytes_b( offset );

                            if( n < params_n_ && (unsigned)d < params_d_ &&
                                (unsigned)h < params_h_ && (unsigned)w < params_w_ &&
                                c_[ni][i] < params_c_ ) {
                                xmma::ldg( tmp_b[j * 2 + i], ptr );
                            } else {
                                tmp_b[j * LDGS_N + i] = 0;
                            }
                        }
                    }
                }
            }
        }
    }

    // Load a tile from shared.
    inline __device__ void load_1( Fragment ( &b )[Xmma_tile::XMMAS_K][Xmma_tile::XMMAS_N] ) {

        #pragma unroll
        for( int ki = 0; ki < Xmma_tile::XMMAS_K; ki++ ) {
            #pragma unroll
            for( int j = 0; j < LDGS_K; j++ ) {
                int idx = j * 4 + ki * Xmma_tile::K_PER_XMMA;

                uint4 nopq;
                xmma::lds( nopq, smem_lds_ + idx * sizeof( uint4 ) );
                // Extract the coordinates of that pixel.
                int n = reinterpret_cast<const int &>( nopq.x );
                int o = reinterpret_cast<const int &>( nopq.y );
                int p = reinterpret_cast<const int &>( nopq.z );
                int q = reinterpret_cast<const int &>( nopq.w );

                #pragma unroll
                for( int ni = 0; ni < Xmma_tile::XMMAS_N; ni++ ) {
                    half *tmp = (half *)&( b[ki][ni].u16( 0 ) );
                    #pragma unroll
                    for( int i = 0; i < LDGS_N; i++ ) {
                        // Compute the H and W coordinates.
                        int d = o + t_[ni][i];
                        int h = p + r_[ni][i];
                        int w = q + s_[ni][i];

                        int offset = n * params_dhwc_ + d * params_hwc_ + h * params_wc_ +
                                     w * params_c_ + c_[ni][i];
                        const char *ptr = ptr_ + Traits::offset_in_bytes_b( offset );

                        if( n < params_n_ && (unsigned)d < params_d_ && (unsigned)h < params_h_ &&
                            (unsigned)w < params_w_ && c_[ni][i] < params_c_ ) {
                            if( BYTES_PER_LDG == 2 ) {
                                if( HAS_SUPER_HMMA ) {
                                    xmma::ldg( (uint16_t &)tmp[i * LDGS_K + j], ptr );
                                } else {
                                    xmma::ldg( (uint16_t &)tmp[j * 4 + i], ptr );
                                }
                            } else if( BYTES_PER_LDG == 4 ) {
                                xmma::ldg( b[ki][ni].reg( i * 2 + j ), ptr );
                            }
                        } else {
                            if( BYTES_PER_LDG == 2 ) {
                                if( HAS_SUPER_HMMA ) {
                                    tmp[i * LDGS_K + j] = 0;
                                } else {
                                    tmp[j * 4 + i] = 0;
                                }
                            } else if( BYTES_PER_LDG == 4 ) {
                                b[ki][ni].reg( i * 2 + j ) = 0;
                            }
                        }
                    }
                }
            }
        }
    }
    const char *ptr_;
    const int params_dhwc_, params_hwc_, params_hw_, params_wc_;
    // The parameters to decompose NOPQ.
    const int params_opq_, params_mul_opq_, params_shr_opq_;
    // The parameters to decompose OPQ.
    const int params_pq_, params_mul_pq_, params_shr_pq_;
    // The parameters to decompose PQ.
    const int params_q_, params_mul_q_, params_shr_q_;
    // The stride.
    const int params_stride_d_, params_stride_h_, params_stride_w_;
    // The dilation.
    const int params_dilation_d_, params_dilation_h_, params_dilation_w_;
    // The padding.
    const int params_pad_d_, params_pad_h_, params_pad_w_;
    uint32_t params_c_, params_ndhw_, params_d_, params_h_, params_w_, params_n_;
    uint32_t tiles_k_;
    uint32_t smem_;
    uint32_t smem_lds_;
    int t_[Xmma_tile::XMMAS_N][LDGS_N], r_[Xmma_tile::XMMAS_N][LDGS_N];
    int s_[Xmma_tile::XMMAS_N][LDGS_N], c_[Xmma_tile::XMMAS_N][LDGS_N];
    int nopq_;
    int params_trsc_, params_trsc_last_, params_k_last_;
    uint32_t preds_lds[Xmma_tile::XMMAS_K][4];
};

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// T I L E   C
//
////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    // The instruction traits.
    typename Traits,
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // ELEMENT_PER_LDG
    int ELTS_PER_LDG_,
    // ELEMENT_PER_STG
    int ELTS_PER_STG_,
    // The fragment class before writing data to global memory.
    typename Fragment_c = xmma::Fragment_c<Traits, Cta_tile, true>>
struct Gmem_tile_c {

    enum { HAS_SUPER_HMMA = Traits::Gpu_arch::HAS_SUPER_HMMA };

    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;

    // Fragment for dst if beta != 0.0
    using Fragment_aclwmulator = xmma::Fragment_aclwmulator<Traits>;

    // The number of elements per LDG.
    enum { ELTS_PER_LDG = ELTS_PER_LDG_ };
    // The number of elements per STG.
    enum { ELTS_PER_STG = ELTS_PER_STG_ };

    enum { BYTES_PER_STG = ELTS_PER_LDG * Traits::BITS_PER_ELEMENT_C / 8 };

    // Ctor.
    template <typename Params>
    inline __device__ Gmem_tile_c( const Params &params, int bidm, int bidn, int bidz, int tidx )
        : params_m_( params.k * params.g ), params_n_( params.trsc * params.g ),
          params_stride_n_( params.trsc * params.g ) {
        char *ptr = reinterpret_cast<char *>( params.out_gmem );

        // The number of warps.
        const int WARPS_M = Cta_tile::WARPS_M;
        const int WARPS_N = Cta_tile::WARPS_N;
        const int WARPS_K = Cta_tile::WARPS_K;

        // The masks to select the warps.
        const int WARP_MASK_M = xmma::Warp_masks<WARPS_M, WARPS_N, WARPS_K>::M;
        const int WARP_MASK_N = xmma::Warp_masks<WARPS_M, WARPS_N, WARPS_K>::N;

        // The divisor for the warps.
        const int WARP_DIV_M = 1 * 1 * Cta_tile::THREADS_PER_WARP;
        const int WARP_DIV_N = WARPS_M * 1 * Cta_tile::THREADS_PER_WARP;

        // The location of the tile.
        int row = ( ( tidx & WARP_MASK_M ) / WARP_DIV_M ) * Xmma_tile::M_PER_WARP;
        int col = ( ( tidx & WARP_MASK_N ) / WARP_DIV_N ) * Xmma_tile::N_PER_WARP;

        if( HAS_SUPER_HMMA ) {
            row += ( ( tidx & 0x1c ) / 4 ) * ELTS_PER_STG;
            col += ( tidx % 4 ) * ELTS_PER_LDG * 2;
        } else {
            row += ( ( ( tidx & 0x10 ) >> 3 ) + ( ( tidx & 0x04 ) >> 2 ) ) * ELTS_PER_STG +
                   ( tidx & 0x01 ) * ( ELTS_PER_STG > 1 ? 1 : 4 );
            col += ( ( tidx & 0x08 ) >> 3 ) * ELTS_PER_LDG +
                   ( tidx & 0x02 ) * ( ELTS_PER_LDG < 4 ? 4 : 1 );
        }

        // Compute the output position for each thread.
        m_ = bidm * Cta_tile::M + row;
        n_ = bidn * Cta_tile::N + col;

        // The pointer.
        ptr_ = &ptr[Traits::offset_in_bytes_c( m_ * params_stride_n_ + n_ )];
    }
    // Compute the row offset.
    static inline __device__ int compute_offset( int mi, int i ) {
        if( HAS_SUPER_HMMA ) {
            if( ELTS_PER_STG == 1 ) {
                return mi * 2 * 8 * ELTS_PER_STG + i * 8;
            } else if( ELTS_PER_STG == 2 ) {
                return mi * 8 * ELTS_PER_STG + i;
            } else if( ELTS_PER_STG == 4 ) {
                return ( mi / 2 ) * 32 + ( mi % 2 ) * 2 + i;
            } else {
                return 0;
            }
        } else {
            if( ELTS_PER_STG == 1 ) {
                return mi * Xmma_tile::M_PER_XMMA + i * 8;
            } else if( ELTS_PER_STG == 2 ) {
                return mi * Xmma_tile::M_PER_XMMA + i * 8;
            } else if( ELTS_PER_STG == 4 ) {
                return mi * Xmma_tile::M_PER_XMMA + i * 2;
            } else {
                return 0;
            }
        }
    }

    // Is a given output valid?
    inline __device__ int compute_output_mask( int mi, int i, int col ) {
        int row;
        if( HAS_SUPER_HMMA ) {
            if( ELTS_PER_STG == 1 ) {
                row = m_ + mi * 2 * 8 * ELTS_PER_STG + i * 8;
            } else if( ELTS_PER_STG == 2 ) {
                row = m_ + mi * 8 * ELTS_PER_STG + i;
            } else if( ELTS_PER_STG == 4 ) {
                row = m_ + ( mi / 2 ) * 32 + ( mi % 2 ) * 2 + i;
            } else {
                row = 0;
            }
        } else {
            if( ELTS_PER_STG == 1 ) {
                row = m_ + mi * Xmma_tile::M_PER_XMMA + i * 8;
            } else if( ELTS_PER_STG == 2 ) {
                row = m_ + mi * Xmma_tile::M_PER_XMMA + i * 8;
            } else if( ELTS_PER_STG == 4 ) {
                row = m_ + mi * Xmma_tile::M_PER_XMMA + i * 2;
            } else {
                row = 0;
            }
        }

        return ( row < params_m_ ) && ( col < params_n_ );
    }

    // Load the data from global memory.
    inline __device__ void
    load( Fragment_c &data, int mi, int ii, int mask, uint64_t mem_desc = xmma::MEM_DESC_DEFAULT ) {
    }

    // Store the data to global memory.
    template <int N> inline __device__ void store( int mi, int ni, Fragment_c ( &data )[N] ) {
        if( ELTS_PER_LDG == 1 ) {
            store_1( mi, ni, data );
        } else if( ELTS_PER_LDG == 2 ) {
            store_2( mi, ni, data );
        } else if( ELTS_PER_LDG == 4 ) {
        }
    }

    template <int N> inline __device__ void store_2( int mi, int ni, Fragment_c ( &data )[N] ) {
        #pragma unroll
        for( int i = 0; i < 2; i++ ) {
            int mask[4];

            if( HAS_SUPER_HMMA ) {
                int offset = compute_offset( mi, i ) * params_stride_n_ + ni * 2;

                mask[0] = compute_output_mask( mi, i, n_ + ni * 2 );

                mask[1] = compute_output_mask( mi, i, n_ + ni * 2 + 2 * Xmma_tile::XMMAS_N );

                if( BYTES_PER_STG == 4 ) {
                    uint32_t tmp;
                    if( mask[0] ) {
                        ( (uint16_t *)&tmp )[0] = data[ni].u16( 4 * i + 0 );
                        ( (uint16_t *)&tmp )[1] = data[ni].u16( 4 * i + 2 );
                        xmma::stg( ptr_ + Traits::offset_in_bytes_c( offset ), tmp );
                    }
                    if( mask[1] ) {
                        ( (uint16_t *)&tmp )[0] = data[ni].u16( 4 * i + 1 );
                        ( (uint16_t *)&tmp )[1] = data[ni].u16( 4 * i + 3 );
                        xmma::stg( ptr_ +
                                       Traits::offset_in_bytes_c( offset + 2 * Xmma_tile::XMMAS_N ),
                                   tmp );
                    }
                } else if( BYTES_PER_STG == 8 ) {
                    if( mask[0] ) {
                        xmma::stg(
                            ptr_ + Traits::offset_in_bytes_c( offset ),
                            make_uint2( data[2 * ni + i].reg( 0 ), data[2 * ni + i].reg( 2 ) ) );
                    }
                    if( mask[1] ) {
                        xmma::stg(
                            ptr_ + Traits::offset_in_bytes_c( offset + 2 * Xmma_tile::XMMAS_N ),
                            make_uint2( data[2 * ni + i].reg( 1 ), data[2 * ni + i].reg( 3 ) ) );
                    }
                }
            } else {
                int offset =
                    compute_offset( mi, i ) * params_stride_n_ + ni * Xmma_tile::N_PER_XMMA;

                int mask[4];
                mask[0] = compute_output_mask( mi, i, n_ + ni * Xmma_tile::N_PER_XMMA );

                mask[1] = compute_output_mask( mi, i, n_ + ni * Xmma_tile::N_PER_XMMA + 4 );

                uint32_t tmp;
                if( mask[0] ) {
                    ( (uint16_t *)&tmp )[0] = data[ni].u16( 4 * i + 0 );
                    ( (uint16_t *)&tmp )[1] = data[ni].u16( 4 * i + 1 );
                    xmma::stg( ptr_ + Traits::offset_in_bytes_c( offset ), tmp );
                }
                if( mask[1] ) {
                    ( (uint16_t *)&tmp )[0] = data[ni].u16( 4 * i + 2 );
                    ( (uint16_t *)&tmp )[1] = data[ni].u16( 4 * i + 3 );
                    xmma::stg( ptr_ + Traits::offset_in_bytes_c( offset + 4 ), tmp );
                }
            }
        }
    }

    template <int N> inline __device__ void store_1( int mi, int ni, Fragment_c ( &data )[N] ) {
        #pragma unroll
        for( int i = 0; i < 2; i++ ) {
            if( HAS_SUPER_HMMA ) {
                int offset = compute_offset( mi, i ) * params_stride_n_ + ni * 2;

                int mask[4];
                mask[0] = compute_output_mask( mi, i, n_ + ni * 2 );
                mask[1] = compute_output_mask( mi, i, n_ + ni * 2 + 1 );
                mask[2] = compute_output_mask( mi, i, n_ + ni * 2 + 8 );
                mask[3] = compute_output_mask( mi, i, n_ + ni * 2 + 9 );

                if( BYTES_PER_STG == 2 ) {
                    if( mask[0] ) {
                        xmma::stg( ptr_ + Traits::offset_in_bytes_c( offset ),
                                   data[ni].u16( 4 * i + 0 ) );
                    }
                    if( mask[1] ) {
                        xmma::stg( ptr_ + Traits::offset_in_bytes_c( offset + 1 ),
                                   data[ni].u16( 4 * i + 1 ) );
                    }
                    if( mask[2] ) {
                        xmma::stg( ptr_ + Traits::offset_in_bytes_c( offset + 8 ),
                                   data[ni].u16( 4 * i + 2 ) );
                    }
                    if( mask[3] ) {
                        xmma::stg( ptr_ + Traits::offset_in_bytes_c( offset + 9 ),
                                   data[ni].u16( 4 * i + 3 ) );
                    }
                } else if( BYTES_PER_STG == 4 ) {
                    if( mask[0] ) {
                        xmma::stg( ptr_ + Traits::offset_in_bytes_c( offset ),
                                   data[2 * ni + i].reg( 0 ) );
                    }
                    if( mask[1] ) {
                        xmma::stg( ptr_ + Traits::offset_in_bytes_c( offset + 1 ),
                                   data[2 * ni + i].reg( 1 ) );
                    }
                    if( mask[2] ) {
                        xmma::stg( ptr_ + Traits::offset_in_bytes_c( offset + 8 ),
                                   data[2 * ni + i].reg( 2 ) );
                    }
                    if( mask[3] ) {
                        xmma::stg( ptr_ + Traits::offset_in_bytes_c( offset + 9 ),
                                   data[2 * ni + i].reg( 3 ) );
                    }
                }
            } else {
                int offset =
                    compute_offset( mi, i ) * params_stride_n_ + ni * Xmma_tile::N_PER_XMMA;

                int mask[4];
                mask[0] = compute_output_mask( mi, i, n_ + ni * Xmma_tile::N_PER_XMMA );
                mask[1] = compute_output_mask( mi, i, n_ + ni * Xmma_tile::N_PER_XMMA + 4 );
                mask[2] = compute_output_mask( mi, i, n_ + ni * Xmma_tile::N_PER_XMMA + 2 );
                mask[3] = compute_output_mask( mi, i, n_ + ni * Xmma_tile::N_PER_XMMA + 6 );

                if( mask[0] ) {
                    xmma::stg( ptr_ + Traits::offset_in_bytes_c( offset ),
                               data[ni].u16( 4 * i + 0 ) );
                }
                if( mask[1] ) {
                    xmma::stg( ptr_ + Traits::offset_in_bytes_c( offset + 4 ),
                               data[ni].u16( 4 * i + 1 ) );
                }
                if( mask[2] ) {
                    xmma::stg( ptr_ + Traits::offset_in_bytes_c( offset + 2 ),
                               data[ni].u16( 4 * i + 2 ) );
                }
                if( mask[3] ) {
                    xmma::stg( ptr_ + Traits::offset_in_bytes_c( offset + 6 ),
                               data[ni].u16( 4 * i + 3 ) );
                }
            }
        }
    }

    // The dimensions of the matrix.
    const int params_m_, params_n_, params_stride_n_;
    // The pointer to global memory.
    char *ptr_;
    // The position of the tile.
    int m_, n_;
};

///////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace wgrad
}  // namespace indexed_wo_smem
}  // namespace implicit_gemm
}  // namespace ext
}  // namespace xmma
