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

#include <xmma/ext/implicit_gemm/indexed_wo_smem/fprop/utils.h>
#include <xmma/implicit_gemm/fprop/gmem_tile.h>

namespace xmma {
namespace ext {
namespace implicit_gemm {
namespace indexed_wo_smem {
namespace fprop {

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// T I L E   A
//
///////////////////////////////////////////////////////////////////////////////////////////////////

template<
    // The instruction traits.
    typename Traits,
    // The CTA tile description.
    typename Cta_tile,
    // ELEMENT_PER_LDG
    int ELTS_PER_LDG_,
    // ELEMENT_PER_STG
    int ELTS_PER_STG_>
struct Gmem_tile_a {

    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;
    using Fragment = xmma::Fragment_a<Traits, xmma::Row>;
    // The GPU arch
    using Gpu_arch = typename Traits::Gpu_arch;

    // The number of elements per LDG.
    enum { ELTS_PER_LDG = ELTS_PER_LDG_ };
    // The number of elements per STG.
    enum { ELTS_PER_STG = ELTS_PER_STG_ };

    enum { BYTES_PER_LDG = ELTS_PER_LDG * Traits::BITS_PER_ELEMENT_A / 8 };

    enum { HAS_SUPER_HMMA = Gpu_arch::HAS_SUPER_HMMA };
    enum { ROWS_PER_THREADS = HAS_SUPER_HMMA ? 2 : 1 };
    enum { THREADS_PER_XMMA_N = HAS_SUPER_HMMA ? 4 : 1 };

    enum { LDGS = xmma::Max<Xmma_tile::K_PER_XMMA /
        (ELTS_PER_LDG * THREADS_PER_XMMA_N), 1>::VALUE };
    enum { XMMAS_K_PER_LDG = xmma::Max<xmma::Div_up<
        ELTS_PER_LDG * THREADS_PER_XMMA_N,
        Xmma_tile::K_PER_XMMA>::VALUE, 1>::VALUE };
    enum { XMMAS_K_LDGS = xmma::Max<Xmma_tile::XMMAS_K / XMMAS_K_PER_LDG, 1>::VALUE };

    enum { K_PER_THREADS = Xmma_tile::K_PER_XMMA / THREADS_PER_XMMA_N };
    enum { REGS_PER_K_PER_THREADS = (K_PER_THREADS * Traits::BITS_PER_ELEMENT_B) / 32 };


    // Ctor.
    template <typename Params>
    inline __device__ Gmem_tile_a( const Params& params, void* smem, const dim3& bidx,
                                        int tidx )
        : ptr_(reinterpret_cast<const char*>(params.img_gmem)),
          smem_(xmma::get_smem_pointer(smem)) {

    int k, g;
    xmma::fast_divmod( g, k, bidx.y, params.kn, params.mul_k, params.shr_k);
    ptr_ += Traits::offset_in_bytes_a(g * params.c);

    #pragma unroll
    for(int ii = 0; ii < xmma::Max<Cta_tile::M/Cta_tile::THREADS_PER_CTA, 1>::VALUE; ii++) {
    const int row = bidx.x * Cta_tile::M + tidx + ii * Cta_tile::THREADS_PER_CTA;
    int n, opq, pq, o, p, q;
    if (row < params.nopq) {
        xmma::fast_divmod( n, opq, row, params.opq, params.mul_opq, params.shr_opq );
        xmma::fast_divmod( o, pq, opq, params.pq, params.mul_pq, params.shr_pq );
        xmma::fast_divmod( p, q, pq, params.q, params.mul_q, params.shr_q );

        o = o * params.stride[0] - params.pad[0][0];
        p = p * params.stride[1] - params.pad[1][0];
        q = q * params.stride[2] - params.pad[2][0];
        n = n * params.img_stride_n +
            o * params.img_stride_d +
            p * params.img_stride_h +
            q * params.img_stride_w;
    } else {
        n = 0;
        o = params.d + 1;
        p = params.h + 1;
        q = params.w + 1;
    }

    int4 indices = make_int4( n, o, p, q );
    xmma::sts( smem_ + (tidx + ii * Cta_tile::THREADS_PER_CTA)
        * sizeof( int4 ), reinterpret_cast<const uint4&>( indices ) );
    }

    params_trsc_ = params.trsc;

    col_ = tidx;
    int ci = 0, trs, rs, ti, ri, si;
    if (col_ < params_trsc_) {
        xmma::fast_divmod( ti, trs, col_, params.trs, params.mul_trs,  params.shr_trs );
        xmma::fast_divmod( ri, rs, trs, params.rs, params.mul_rs, params.shr_rs );
        xmma::fast_divmod( si, ci, rs, params.c, params.mul_s, params.shr_s );

        ti *= params.dilation[0];
        ri *= params.dilation[1];
        si *= params.dilation[2];
    } else {
        ti = -1;
    }
    ci = ci +
         ti * params.img_stride_d +
         ri * params.img_stride_h +
         si * params.img_stride_w;

    smem_trsc_lds_ = smem_ +
        xmma::Max<Cta_tile::M, Cta_tile::THREADS_PER_CTA>::VALUE * 4 * 4;

    smem_trsc_ = smem_ + xmma::Max<Cta_tile::M, Cta_tile::THREADS_PER_CTA>::VALUE * 4 * 4 + tidx * sizeof( int4 );
    int4 trsc_idx = make_int4( ci, ti, ri, si );
    xmma::sts( smem_trsc_, reinterpret_cast<const uint4&>(trsc_idx));

    __syncthreads();

    const int WARPS_M = Cta_tile::WARPS_M;
    const int WARPS_N = Cta_tile::WARPS_N;
    const int WARPS_K = Cta_tile::WARPS_K;
    // The masks to select the warps.
    const int WARP_MASK_M = xmma::Warp_masks<WARPS_M, WARPS_N, WARPS_K>::M;
    // The divisor for the warps.
    const int WARP_DIV_M =       1 *       1 * Cta_tile::THREADS_PER_WARP;

    #pragma unroll
    for (int mi = 0; mi < Xmma_tile::XMMAS_M; mi++) {
        #pragma unroll
        for (int i = 0; i < ROWS_PER_THREADS; i++) {
            int idx =
                + (tidx & WARP_MASK_M) / WARP_DIV_M * Xmma_tile::M_PER_XMMA
                + mi * Xmma_tile::M_PER_XMMA_PER_CTA
                + (HAS_SUPER_HMMA
                    ? (tidx & 0x1c) / 4 + i * 8
                    : tidx % 4 + ((tidx / 8) % 2) * 4 + ((tidx & 0x1c) / 16) * 8);

            uint4 nopq;
            xmma::lds( nopq, this->smem_ + idx * sizeof( uint4 ) );

            // Extract the coordinates of that pixel.
            n_[mi][i] = reinterpret_cast<const int&>( nopq.x );
            o_[mi][i] = reinterpret_cast<const int&>( nopq.y );
            p_[mi][i] = reinterpret_cast<const int&>( nopq.z );
            q_[mi][i] = reinterpret_cast<const int&>( nopq.w );
        }
    }

    }

    // Load a tile from global.
    template <typename Params>
    inline __device__ void load(
        Fragment (&a)[Xmma_tile::XMMAS_K][Xmma_tile::XMMAS_M],
        const Params& params) {
        int tidx;
        asm volatile("mov.b32 %0, %%tid.x;" : "=r"(tidx));

        #pragma unroll
        for (int ki = 0; ki < XMMAS_K_LDGS; ki++) {

        int c[LDGS], t[LDGS], r[LDGS], s[LDGS];

        #pragma unroll
        for (int j = 0; j < LDGS; j++) {
            int col_idx;
            if (HAS_SUPER_HMMA) {
                col_idx = (tidx % 4) * ELTS_PER_LDG
                    + ki*Xmma_tile::K_PER_XMMA*XMMAS_K_PER_LDG + j*4*ELTS_PER_LDG;
            } else {
                col_idx = ki*Xmma_tile::K_PER_XMMA*XMMAS_K_PER_LDG + j*ELTS_PER_LDG;
            }

            uint4 trsc;
            xmma::lds( trsc, smem_trsc_lds_ + col_idx * sizeof( uint4 ) );

            // Extract the coordinates of that pixel.
            c[j] = reinterpret_cast<const int&>( trsc.x );
            t[j] = reinterpret_cast<const int&>( trsc.y );
            r[j] = reinterpret_cast<const int&>( trsc.z );
            s[j] = reinterpret_cast<const int&>( trsc.w );
        }

        #pragma unroll
        for (int mi = 0; mi < Xmma_tile::XMMAS_M; mi++) {
        #pragma unroll
        for (int i = 0; i < ROWS_PER_THREADS; i++) {
            #pragma unroll
            for (int j = 0; j < LDGS; j++) {

                int d = o_[mi][i] + t[j];
                int h = p_[mi][i] + r[j];
                int w = q_[mi][i] + s[j];

                if (
                    (uint32_t)d < params.d &&
                    (uint32_t)h < params.h &&
                    (uint32_t)w < params.w &&
                    t[j] >= 0
                    ) {

                    const int offset = n_[mi][i] + c[j];

                    const char *ptr = ptr_ + Traits::offset_in_bytes_a(offset);
                    if ( BYTES_PER_LDG == 1) {
                        xmma::ldg(a[ki][mi].u8((j/4)*8 + i*4 + j%4), ptr);
                    } else if ( BYTES_PER_LDG == 2) {
                        if (REGS_PER_K_PER_THREADS == 4) {
                            xmma::ldg(a[ki][mi].u16(j), ptr);
                        } else {
                            xmma::ldg(a[ki][mi].u16((j/2)*4 + i*2 + j%2), ptr);
                        }
                    } else if (BYTES_PER_LDG == 4) {
                        if (REGS_PER_K_PER_THREADS == 4) {
                            xmma::ldg(a[ki][mi].reg(j), ptr);
                        } else {
                            xmma::ldg(a[ki][mi].reg(j*2 + i), ptr);
                        }
                    } else if (BYTES_PER_LDG == 8) {
                        uint2 tmp_dst;
                        xmma::ldg(tmp_dst, ptr);
                        if (REGS_PER_K_PER_THREADS == 4) {
                            a[ki][mi].reg(2*j+0) = tmp_dst.x;
                            a[ki][mi].reg(2*j+1) = tmp_dst.y;
                        } else if (REGS_PER_K_PER_THREADS == 2) {
                            a[ki][mi].reg(i) = tmp_dst.x;
                            a[ki][mi].reg(i+2) = tmp_dst.y;
                        } else {
                            a[2*ki][mi].reg(i) = tmp_dst.x;
                            a[2*ki+1][mi].reg(i) = tmp_dst.y;
                        }
                    } else if (BYTES_PER_LDG == 16) {
                        uint4 tmp_dst;
                        xmma::ldg(tmp_dst, ptr);
                        if (REGS_PER_K_PER_THREADS == 4) {
                            a[ki][mi].reg(4*i+0) = tmp_dst.x;
                            a[ki][mi].reg(4*i+1) = tmp_dst.y;
                            a[ki][mi].reg(4*i+2) = tmp_dst.z;
                            a[ki][mi].reg(4*i+3) = tmp_dst.w;
                        } else if (REGS_PER_K_PER_THREADS == 2) {
                            a[2*ki][mi].reg(i) = tmp_dst.x;
                            a[2*ki][mi].reg(i+2) = tmp_dst.y;
                            a[2*ki+1][mi].reg(i) = tmp_dst.z;
                            a[2*ki+1][mi].reg(i+2) = tmp_dst.w;
                        } else {
                            a[4*ki+0][mi].reg(i) = tmp_dst.x;
                            a[4*ki+1][mi].reg(i) = tmp_dst.y;
                            a[4*ki+2][mi].reg(i) = tmp_dst.z;
                            a[4*ki+3][mi].reg(i) = tmp_dst.w;
                        }
                    } else {
                    }
                } else {
                    if ( BYTES_PER_LDG == 1) {
                        a[ki][mi].u8((j/4)*8 + i*4 + j%4) = 0;
                    } else if (BYTES_PER_LDG == 2) {
                        if (REGS_PER_K_PER_THREADS == 4) {
                            a[ki][mi].u16(j) = 0;
                        } else {
                            a[ki][mi].u16((j/2)*4 + i*2 + j%2) = 0;
                        }
                    } else if (BYTES_PER_LDG == 4) {
                        if (REGS_PER_K_PER_THREADS == 4) {
                            a[ki][mi].reg(j) = 0;
                        } else {
                            a[ki][mi].reg(j*2 + i) = 0;
                        }
                    } else if (BYTES_PER_LDG == 8) {
                        if (REGS_PER_K_PER_THREADS == 4) {
                            a[ki][mi].reg(2*j+0) = 0;
                            a[ki][mi].reg(2*j+1) = 0;
                        } else if (REGS_PER_K_PER_THREADS == 2) {
                            a[ki][mi].reg(i) = 0;
                            a[ki][mi].reg(i+2) = 0;
                        } else {
                            a[2*ki][mi].reg(i) = 0;
                            a[2*ki+1][mi].reg(i) = 0;
                        }
                    } else if (BYTES_PER_LDG == 16) {
                        if (REGS_PER_K_PER_THREADS == 4) {
                            a[ki][mi].reg(4*i+0) = 0;
                            a[ki][mi].reg(4*i+1) = 0;
                            a[ki][mi].reg(4*i+2) = 0;
                            a[ki][mi].reg(4*i+3) = 0;
                        } else if (REGS_PER_K_PER_THREADS == 2) {
                            a[2*ki][mi].reg(i) = 0;
                            a[2*ki][mi].reg(i+2) = 0;
                            a[2*ki+1][mi].reg(i) = 0;
                            a[2*ki+1][mi].reg(i+2) = 0;
                        } else {
                            a[4*ki+0][mi].reg(i) = 0;
                            a[4*ki+1][mi].reg(i) = 0;
                            a[4*ki+2][mi].reg(i) = 0;
                            a[4*ki+3][mi].reg(i) = 0;
                        }
                    } else {
                    }
                }
            }
        } }
        }
    }

    // Move the pointers and update the predicates for R2P/P2R.
    template <typename Params>
    inline __device__ void move( const Params& params ) {

        col_ += Cta_tile::K;
        int ci=0, trs, rs, ti, ri, si;
        if (col_ < params_trsc_) {
            xmma::fast_divmod( ti, trs, col_, params.trs, params.mul_trs,  params.shr_trs );
            xmma::fast_divmod( ri, rs, trs, params.rs, params.mul_rs, params.shr_rs );
            xmma::fast_divmod( si, ci, rs, params.c, params.mul_s, params.shr_s );

            ti *= params.dilation[0];
            ri *= params.dilation[1];
            si *= params.dilation[2];
        } else {
            ti = -1;
        }
        ci = ci +
             ti * params.img_stride_d +
             ri * params.img_stride_h +
             si * params.img_stride_w;

        int4 trsc_idx = make_int4( ci, ti, ri, si );
        xmma::sts( smem_trsc_, reinterpret_cast<const uint4&>(trsc_idx));
    }

    uint32_t params_trsc_;
    const char *ptr_;
    uint32_t smem_;
    uint32_t smem_trsc_;
    uint32_t smem_trsc_lds_;
    int col_;
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

template<
    // The instruction traits.
    typename Traits,
    // The CTA tile description.
    typename Cta_tile,
    // ELEMENT_PER_LDG
    int ELTS_PER_LDG_,
    // ELEMENT_PER_STG
    int ELTS_PER_STG_>
struct Gmem_tile_b {

    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;
    using Fragment = xmma::Fragment_b<Traits, xmma::Col>;
    // The GPU arch
    using Gpu_arch = typename Traits::Gpu_arch;

    // The number of elements per LDG.
    enum { ELTS_PER_LDG = ELTS_PER_LDG_ };
    // The number of elements per STG.
    enum { ELTS_PER_STG = ELTS_PER_STG_ };

    enum { BYTES_PER_LDG = ELTS_PER_LDG * Traits::BITS_PER_ELEMENT_B / 8 };

    enum { HAS_SUPER_HMMA = Gpu_arch::HAS_SUPER_HMMA };
    enum { ROWS_PER_THREADS = HAS_SUPER_HMMA ? 2 : 1 };
    enum { THREADS_PER_XMMA_N = HAS_SUPER_HMMA ? 4 : 1 };

    enum { LDGS = xmma::Max<Xmma_tile::K_PER_XMMA /
        (ELTS_PER_LDG * THREADS_PER_XMMA_N), 1>::VALUE };
    enum { XMMAS_K_PER_LDG = xmma::Max<xmma::Div_up<
        ELTS_PER_LDG * THREADS_PER_XMMA_N,
        Xmma_tile::K_PER_XMMA>::VALUE, 1>::VALUE };
    enum { XMMAS_K_LDGS = xmma::Max<Xmma_tile::XMMAS_K / XMMAS_K_PER_LDG, 1>::VALUE };

    enum { K_PER_THREADS = Xmma_tile::K_PER_XMMA / THREADS_PER_XMMA_N };
    enum { REGS_PER_K_PER_THREADS = (K_PER_THREADS * Traits::BITS_PER_ELEMENT_B) / 32 };

    // Ctor.
    template <typename Params>
    inline __device__ Gmem_tile_b( const Params& params, void* smem, const dim3& bidx,
                                        int tidx )
        : ptr_(reinterpret_cast<const char*>(params.flt_gmem)),
          smem_(xmma::get_smem_pointer(smem)) {

        // The number of warps.
        const int WARPS_M = Cta_tile::WARPS_M;
        const int WARPS_N = Cta_tile::WARPS_N;
        const int WARPS_K = Cta_tile::WARPS_K;

        // The masks to select the warps.
        const int WARP_MASK_N = xmma::Warp_masks<WARPS_M, WARPS_N, WARPS_K>::N;

        // The divisor for the warps.
        const int WARP_DIV_N = WARPS_M *       1 * Cta_tile::THREADS_PER_WARP;

        params_trsc_ = params.trsc;
        params_k_ = params.k;

        int k, g;
        xmma::fast_divmod( g, k, bidx.y, params.kn, params.mul_k, params.shr_k);
        ptr_ += Traits::offset_in_bytes_a(g * params.trsc * params.k);

        const int row = HAS_SUPER_HMMA ? tidx % 4 : 0;

        int col = k * Cta_tile::N
            + ((tidx & WARP_MASK_N) / WARP_DIV_N) * Xmma_tile::N_PER_XMMA;
        if ( ELTS_PER_STG == 4 && Xmma_tile::THREADS_PER_XMMA_N == 4) {
            if (HAS_SUPER_HMMA) {
                col += (((tidx & 0x1c) / 4) / 2 ) * 4 + (tidx / 4) % 2;
            } else {
                col += tidx%2 + ((tidx % 8) / 2) * 4 + ((tidx & 0x1c) / 16) * 2;
            }
        } else {
            if (HAS_SUPER_HMMA) {
                col += (tidx & 0x1c) / 4;
            } else {
                col += tidx % 4 + ((tidx / 4) % 2) * 4 + ((tidx & 0x1c) / 16) * 8;
            }
        }

        params_trsc_last_ = params.trsc - row * ELTS_PER_LDG;
        params_k_last_ = params.k - col;

        ptr_ += Traits::offset_in_bytes_b(col * params_trsc_ + row * ELTS_PER_LDG);

        #pragma unroll
        for (int ki = 0; ki < XMMAS_K_LDGS; ki++) {
            #pragma unroll
            for (int j = 0; j < LDGS; j++) {
                if (HAS_SUPER_HMMA) {
                    preds_lds[ki][j] = j*4*ELTS_PER_LDG
                    + ki*Xmma_tile::K_PER_XMMA*XMMAS_K_PER_LDG < params_trsc_last_;
                } else {
                    preds_lds[ki][j] = j*ELTS_PER_LDG
                    + ki*Xmma_tile::K_PER_XMMA*XMMAS_K_PER_LDG < params_trsc_last_;
                }
            }
        }
    }

    inline __device__ void move() {
        ptr_ += Traits::offset_in_bytes_b(Cta_tile::K);
        params_trsc_last_ -= Cta_tile::K;

        #pragma unroll
        for (int ki = 0; ki < XMMAS_K_LDGS; ki++) {
            #pragma unroll
            for (int j = 0; j < LDGS; j++) {
                if (HAS_SUPER_HMMA) {
                    preds_lds[ki][j] = j*4*ELTS_PER_LDG
                    + ki*Xmma_tile::K_PER_XMMA*XMMAS_K_PER_LDG < params_trsc_last_;
                } else {
                    preds_lds[ki][j] = j*ELTS_PER_LDG
                    + ki*Xmma_tile::K_PER_XMMA*XMMAS_K_PER_LDG < params_trsc_last_;
                }
            }
        }
    }

    // Load a tile from shared.
    inline __device__ void load(Fragment
        (&b)[Xmma_tile::XMMAS_K][Xmma_tile::XMMAS_N], int ki = 0) {
        #pragma unroll
        for (int ki = 0; ki < XMMAS_K_LDGS; ki++) {
        #pragma unroll
        for (int ni = 0; ni < Xmma_tile::XMMAS_N; ni++) {
            #pragma unroll
            for (int i = 0; i < ROWS_PER_THREADS; i++) {
            #pragma unroll
            for (int j = 0; j < LDGS; j++) {
                int col = ni*Xmma_tile::N_PER_XMMA_PER_CTA;

                if ( ELTS_PER_STG == 4 && Xmma_tile::THREADS_PER_XMMA_N == 4) {
                    col += i*2;
                } else {
                    col += i*8;
                }
                uint32_t row =
                    ( HAS_SUPER_HMMA ? j*4*ELTS_PER_LDG : j*ELTS_PER_LDG)
                    + ki*Xmma_tile::K_PER_XMMA*XMMAS_K_PER_LDG;
                uint32_t offset = col * params_trsc_ + row;
                const char *ptr = ptr_ + Traits::offset_in_bytes_b(static_cast<int64_t>(offset));
                if (preds_lds[ki][j] && col < params_k_last_) {
                    if ( BYTES_PER_LDG == 1) {
                        xmma::ldg(b[ki][ni].u8(i*REGS_PER_K_PER_THREADS*4 + j), ptr);
                    } else if (BYTES_PER_LDG == 2) {
                        if (REGS_PER_K_PER_THREADS == 4) {
                            xmma::ldg(b[ki][ni].u16(j), ptr);
                        } else {
                            xmma::ldg(b[ki][ni].u16(i*REGS_PER_K_PER_THREADS*2 + j), ptr);
                        }
                    } else if (BYTES_PER_LDG == 4) {
                        if (REGS_PER_K_PER_THREADS == 4) {
                            xmma::ldg(b[ki][ni].reg(j), ptr);
                        } else {
                            xmma::ldg(b[ki][ni].reg(i*REGS_PER_K_PER_THREADS + j%2), ptr);
                        }
                    } else if (BYTES_PER_LDG == 8) {
                        uint2 tmp_dst;
                        xmma::ldg(tmp_dst, ptr);
                        if (REGS_PER_K_PER_THREADS == 4) {
                            b[ki][ni].reg(4*i+j*2+0) = tmp_dst.x;
                            b[ki][ni].reg(4*i+j*2+1) = tmp_dst.y;
                        } else if (REGS_PER_K_PER_THREADS == 2) {
                            b[ki][ni].reg(2*i) = tmp_dst.x;
                            b[ki][ni].reg(2*i+1) = tmp_dst.y;
                        } else {
                            b[2*ki][ni].reg(i) = tmp_dst.x;
                            b[2*ki+1][ni].reg(i) = tmp_dst.y;
                        }
                    } else if (BYTES_PER_LDG == 16) {
                        uint4 tmp_dst;
                        xmma::ldg(tmp_dst, ptr);
                        if (REGS_PER_K_PER_THREADS == 4) {
                            b[ki][ni].reg(4*i+0) = tmp_dst.x;
                            b[ki][ni].reg(4*i+1) = tmp_dst.y;
                            b[ki][ni].reg(4*i+2) = tmp_dst.z;
                            b[ki][ni].reg(4*i+3) = tmp_dst.w;
                        } else if (REGS_PER_K_PER_THREADS == 2) {
                            b[2*ki][ni].reg(2*i) = tmp_dst.x;
                            b[2*ki][ni].reg(2*i+1) = tmp_dst.y;
                            b[2*ki+1][ni].reg(2*i) = tmp_dst.z;
                            b[2*ki+1][ni].reg(2*i+1) = tmp_dst.w;
                        } else {
                            b[4*ki+0][ni].reg(i) = tmp_dst.x;
                            b[4*ki+1][ni].reg(i) = tmp_dst.y;
                            b[4*ki+2][ni].reg(i) = tmp_dst.z;
                            b[4*ki+3][ni].reg(i) = tmp_dst.w;
                        }
                    } else {
                    }
                } else {
                    if ( BYTES_PER_LDG == 1) {
                        b[ki][ni].u8(i*REGS_PER_K_PER_THREADS*4 + j) = 0;
                    } else if (BYTES_PER_LDG == 2) {
                        if (REGS_PER_K_PER_THREADS == 4) {
                            b[ki][ni].u16(j) = 0;
                        } else {
                            b[ki][ni].u16(i*REGS_PER_K_PER_THREADS*2 + j)= 0;
                        }
                    } else if (BYTES_PER_LDG == 4) {
                        if (REGS_PER_K_PER_THREADS == 4) {
                            b[ki][ni].reg(j) = 0;
                        } else {
                            b[ki][ni].reg(i*REGS_PER_K_PER_THREADS + j%2) = 0;
                        }
                    } else if (BYTES_PER_LDG == 8) {
                        if (REGS_PER_K_PER_THREADS == 4) {
                            b[ki][ni].reg(4*i+j*2+0) = 0;
                            b[ki][ni].reg(4*i+j*2+1) = 0;
                        } else if (REGS_PER_K_PER_THREADS == 2) {
                            b[ki][ni].reg(2*i) = 0;
                            b[ki][ni].reg(2*i+1) = 0;
                        } else {
                            b[2*ki][ni].reg(i) = 0;
                            b[2*ki+1][ni].reg(i) = 0;
                        }
                    } else if (BYTES_PER_LDG == 16) {
                        if (REGS_PER_K_PER_THREADS == 4) {
                            b[ki][ni].reg(4*i+0) = 0;
                            b[ki][ni].reg(4*i+1) = 0;
                            b[ki][ni].reg(4*i+2) = 0;
                            b[ki][ni].reg(4*i+3) = 0;
                        } else if (REGS_PER_K_PER_THREADS == 2) {
                            b[2*ki+0][ni].reg(2*i) = 0;
                            b[2*ki+0][ni].reg(2*i+1) = 0;
                            b[2*ki+1][ni].reg(2*i) = 0;
                            b[2*ki+1][ni].reg(2*i+1) = 0;
                        } else {
                            b[4*ki+0][ni].reg(i) = 0;
                            b[4*ki+1][ni].reg(i) = 0;
                            b[4*ki+2][ni].reg(i) = 0;
                            b[4*ki+3][ni].reg(i) = 0;
                        }
                    } else {
                    }
                }
            }
            }
        }
        }
    }
    const char *ptr_;
    uint32_t smem_;
    uint32_t smem_ldgsts_;
    uint32_t smem_lds_;
    int params_trsc_, params_trsc_last_, params_k_last_, params_k_;
    uint32_t preds_lds[Xmma_tile::XMMAS_K][LDGS];
};

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// T I L E   C
//
////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    // The instruction traits.
    typename Traits,
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // ELEMENT_PER_LDG
    int ELTS_PER_LDG_,
    // ELEMENT_PER_STG
    int ELTS_PER_STG_,
    // The fragment class before writing data to global memory.
    typename Fragment_c = xmma::Fragment_c<Traits, Cta_tile, true>
>
struct Gmem_tile_c {

    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;

    // Fragment for dst if beta != 0.0
    using Fragment_aclwmulator = xmma::Fragment_aclwmulator<Traits>;

    // The GPU arch
    using Gpu_arch = typename Traits::Gpu_arch;

    // The number of elements per LDG.
    enum { ELTS_PER_LDG = ELTS_PER_LDG_ };
    // The number of elements per STG.
    enum { ELTS_PER_STG = ELTS_PER_STG_ };

    enum { BYTES_PER_STG = ELTS_PER_STG * Traits::BITS_PER_ELEMENT_C / 8 };

    enum { HAS_SUPER_HMMA = Gpu_arch::HAS_SUPER_HMMA };
    enum { ROWS_PER_THREADS = HAS_SUPER_HMMA ? 2 : 1 };
    enum { ROWS_OFFSET = HAS_SUPER_HMMA ? 8 : 2 };

    enum { COLS_OFFSET = Xmma_tile::THREADS_PER_XMMA_N == 4 ? 8 : 2 };

    // Ctor.
    template< typename Params >
    inline __device__ Gmem_tile_c(const Params &params, int bidm, int bidn, int bidz, int tidx)
        : params_m_(params.nopq),
          params_n_(params.k),
          params_stride_n_(params.k*params.g) {

        char* ptr = reinterpret_cast<char*>(params.out_gmem);

        // The number of warps.
        const int WARPS_M = Cta_tile::WARPS_M;
        const int WARPS_N = Cta_tile::WARPS_N;
        const int WARPS_K = Cta_tile::WARPS_K;

        // The masks to select the warps.
        const int WARP_MASK_M = xmma::Warp_masks<WARPS_M, WARPS_N, WARPS_K>::M;
        const int WARP_MASK_N = xmma::Warp_masks<WARPS_M, WARPS_N, WARPS_K>::N;

        // The divisor for the warps.
        const int WARP_DIV_M =       1 *       1 * Cta_tile::THREADS_PER_WARP;
        const int WARP_DIV_N = WARPS_M *       1 * Cta_tile::THREADS_PER_WARP;


        // The location of the tile.
        int row = ((tidx & WARP_MASK_M) / WARP_DIV_M) * Xmma_tile::M_PER_XMMA
            + (HAS_SUPER_HMMA
                ? (tidx & 0x1c) / 4
                : (Xmma_tile::THREADS_PER_XMMA_N == 4
                    ? (tidx % 2) + ((tidx / 8) % 2) * 4 + ((tidx & 0x1c) / 16) * 8
                    : tidx % 4 + ((tidx / 8) % 2) * 4 + ((tidx & 0x1c) / 16) * 8)
                );

        int col = ((tidx & WARP_MASK_N) / WARP_DIV_N) * Xmma_tile::N_PER_XMMA;

        if ( ELTS_PER_STG == 4 ) {
            if (HAS_SUPER_HMMA) {
                col += (tidx % 4) * ELTS_PER_STG;
            } else {
                col += (Xmma_tile::THREADS_PER_XMMA_N == 4
                    ? ((tidx % 4) / 2) * 4 + ((tidx / 4) % 2) * 8
                    : ((tidx / 4) % 2) * 4);
            }
        } else {
            if (HAS_SUPER_HMMA) {
                col += (tidx % 4) * 2;
            } else {
                col += (Xmma_tile::THREADS_PER_XMMA_N == 4
                    ? ((tidx % 4) / 2) * 2 + ((tidx / 4) % 2) * 4
                    : ((tidx / 4) % 2) * 4);
            }
        }
        int k, g;
        xmma::fast_divmod( g, k, bidn, params.kn, params.mul_k, params.shr_k);

        // Compute the output position for each thread.
        m_ = bidm * Cta_tile::M + row;
        n_ = k * Cta_tile::N + col;

        // The pointer.
        ptr_ = &ptr[Traits::offset_in_bytes_c(m_*params_stride_n_ + n_
            + g * params.k)];
    }
    // Compute the row offset.
    inline __device__ int compute_offset(int mi, int ni, int i) {
        if (Xmma_tile::THREADS_PER_XMMA_N == 4) {
            return (mi * Xmma_tile::M_PER_XMMA_PER_CTA + i * ROWS_OFFSET)
                * params_stride_n_ + ni * Xmma_tile::N_PER_XMMA_PER_CTA;
        } else {
            return (mi * Xmma_tile::M_PER_XMMA_PER_CTA) * params_stride_n_
                + ni * Xmma_tile::N_PER_XMMA_PER_CTA + i * 8;
        }
    }

    // Is a given output valid?
    inline __device__ int compute_output_mask(int row, int col, int ii) {
        if (Xmma_tile::THREADS_PER_XMMA_N == 4) {
            return (row + ii*ROWS_OFFSET < params_m_) && (col < params_n_);
        } else {
            return (row < params_m_) && (col + ii*8 < params_n_);
        }
    }

    // Load the data from global memory.
    inline __device__ void load(Fragment_c &data,
                                int mi,
                                int ii,
                                int mask,
                                uint64_t mem_desc = xmma::MEM_DESC_DEFAULT) {
    }

    // Store the data to global memory.
    template <int N>
    inline __device__ void store(int mi, int ni,
                                 const Fragment_c (&data)[N]) {
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            int offset = compute_offset(mi, ni, i);

            if (ELTS_PER_STG == 1) {
            int mask[4];
            mask[0] = compute_output_mask(
                m_ + mi * Xmma_tile::M_PER_XMMA_PER_CTA,
                n_ + ni * Xmma_tile::N_PER_XMMA_PER_CTA, i);

            mask[1] = compute_output_mask(
                m_ + mi * Xmma_tile::M_PER_XMMA_PER_CTA,
                n_ + ni * Xmma_tile::N_PER_XMMA_PER_CTA + 1, i);

            mask[2] = compute_output_mask(
                m_ + mi * Xmma_tile::M_PER_XMMA_PER_CTA,
                n_ + ni * Xmma_tile::N_PER_XMMA_PER_CTA + COLS_OFFSET, i);

            mask[3] = compute_output_mask(
                m_ + mi * Xmma_tile::M_PER_XMMA_PER_CTA,
                n_ + ni * Xmma_tile::N_PER_XMMA_PER_CTA + COLS_OFFSET + 1, i);

            if (BYTES_PER_STG == 1) {
            if( mask[0] ) {
                xmma::stg(ptr_ + Traits::offset_in_bytes_c(offset),
                data[ni].u8(4*i+0));
            }
            if( mask[1] ) {
                xmma::stg(ptr_ + Traits::offset_in_bytes_c(offset + 1),
                data[ni].u8(4*i+1));
            }
            if( mask[2] ) {
                xmma::stg(ptr_ + Traits::offset_in_bytes_c(offset + COLS_OFFSET),
                data[ni].u8(4*i+2));
            }
            if( mask[3] ) {
                xmma::stg(ptr_ + Traits::offset_in_bytes_c(offset + COLS_OFFSET + 1),
                data[ni].u8(4*i+3));
            }
            } else if (BYTES_PER_STG == 2) {
            if( mask[0] ) {
                xmma::stg(ptr_ + Traits::offset_in_bytes_c(offset),
                data[ni].u16(4*i+0));
            }
            if( mask[1] ) {
                xmma::stg(ptr_ + Traits::offset_in_bytes_c(offset + 1),
                data[ni].u16(4*i+1));
            }
            if( mask[2] ) {
                xmma::stg(ptr_ + Traits::offset_in_bytes_c(offset + COLS_OFFSET),
                data[ni].u16(4*i+2));
            }
            if( mask[3] ) {
                xmma::stg(ptr_ + Traits::offset_in_bytes_c(offset + COLS_OFFSET + 1),
                data[ni].u16(4*i+3));
            }
            } else if (BYTES_PER_STG == 4) {
            if( mask[0] ) {
                xmma::stg(ptr_ + Traits::offset_in_bytes_c(offset),
                data[2*ni+i].reg(0));
            }
            if( mask[1] ) {
                xmma::stg(ptr_ + Traits::offset_in_bytes_c(offset + 1),
                data[2*ni+i].reg(1));
            }
            if( mask[2] ) {
                xmma::stg(ptr_ + Traits::offset_in_bytes_c(offset + COLS_OFFSET),
                data[2*ni+i].reg(2));
            }
            if( mask[3] ) {
                xmma::stg(ptr_ + Traits::offset_in_bytes_c(offset + COLS_OFFSET + 1),
                data[2*ni+i].reg(3));
            }
            } else {
            }
            } else if (ELTS_PER_STG == 2) {
            int mask[2];
            mask[0] = compute_output_mask(
                m_ + mi * Xmma_tile::M_PER_XMMA_PER_CTA,
                n_ + ni * Xmma_tile::N_PER_XMMA_PER_CTA, i);

            mask[1] = compute_output_mask(
                m_ + mi * Xmma_tile::M_PER_XMMA_PER_CTA,
                n_ + ni * Xmma_tile::N_PER_XMMA_PER_CTA + COLS_OFFSET, i);
            if (BYTES_PER_STG == 2) {
            if( mask[0] ) {
                xmma::stg(ptr_ + Traits::offset_in_bytes_c(offset),
                data[ni].u16(2*i+0));
            }
            if( mask[1] ) {
                xmma::stg(ptr_ + Traits::offset_in_bytes_c(offset + COLS_OFFSET),
                data[ni].u16(2*i+1));
            }
            } else if (BYTES_PER_STG == 4) {
            if( mask[0] ) {
                xmma::stg(ptr_ + Traits::offset_in_bytes_c(offset),
                data[ni].reg(2*i+0));
            }
            if( mask[1] ) {
                xmma::stg(ptr_ + Traits::offset_in_bytes_c(offset + COLS_OFFSET),
                data[ni].reg(2*i+1));
            }
            } else if (BYTES_PER_STG == 8) {
            if( mask[0] ) {
                xmma::stg(ptr_ + Traits::offset_in_bytes_c(offset),
                make_uint2(data[2*ni+i].reg(0),data[2*ni+i].reg(1)));
            }
            if( mask[1] ) {
                xmma::stg(ptr_ + Traits::offset_in_bytes_c(offset + COLS_OFFSET),
                make_uint2(data[2*ni+i].reg(2),data[2*ni+i].reg(3)));
            }
            } else {
            }
            } else if (ELTS_PER_STG == 4) {
            int mask[1];
            mask[0] = compute_output_mask(
                m_ + mi * Xmma_tile::M_PER_XMMA_PER_CTA,
                n_ + ni * Xmma_tile::N_PER_XMMA_PER_CTA, i);
            if( mask[0] ) {
                if (BYTES_PER_STG == 4) {
                xmma::stg(ptr_ + Traits::offset_in_bytes_c(offset),
                    data[ni].reg(i));
                } else if (BYTES_PER_STG == 8) {
                xmma::stg(ptr_ + Traits::offset_in_bytes_c(offset),
                    make_uint2(data[ni].reg(2*i+0),data[ni].reg(2*i+1)));
                } else if (BYTES_PER_STG == 16) {
                xmma::stg(ptr_ + Traits::offset_in_bytes_c(offset),
                    make_uint4(data[2*ni+i].reg(0),data[2*ni+i].reg(1),
                        data[2*ni+i].reg(2),data[2*ni+i].reg(3)));
                } else {
                }
            }
            } else {
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

}  // namespace indexed_wo_smem
}  // namespace fprop
}  // namespace implicit_gemm
}  // namespace ext
} // namespace xmma
