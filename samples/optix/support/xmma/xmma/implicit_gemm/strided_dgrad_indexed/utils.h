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

///////////////////////////////////////////////////////////////////////////////////////////////////

#include <xmma/utils.h>

namespace xmma {
namespace implicit_gemm {
namespace strided_dgrad_indexed {

///////////////////////////////////////////////////////////////////////////////////////////////////

static inline __device__ void lwda_find_divisor( uint32_t& mul, uint32_t& shr, int x ) {
    if( x <= 1 ) {
        mul = 0;
        shr = 0;
    } else {
        int a = 31 - __clz( x );
        a += ( x & ( x - 1 ) ) ? 1 : 0;
        uint32_t p = 31 + a;
        uint32_t m = ( ( 1ull << p ) + (uint32_t)x - 1 ) / (uint32_t)x;

        mul = m;
        shr = p - 32;
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////

static inline int gcd( int a, int b ) {
    if( a < b ) {
        int tmp( a );
        a = b;
        b = tmp;
    }
    if( b == 0 ) {
        return a;
    }
    return gcd( b, a % b );
}

///////////////////////////////////////////////////////////////////////////////////////////////////

static inline int lcm( int a, int b ) {
    return a * b / gcd( a, b );
}

///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Params>
int compute_grid_dimensions( dim3& grid, Params& params, int tile_m, int tile_n, int tile_group ) {
    params.tiles_m =
        xmma::div_up( params.sum_of_round_up_ndhw_number_of_each_filter_pattern, tile_m );
    params.tiles_n =
        xmma::div_up( params.c, tile_n / tile_group ) * xmma::div_up( params.g, tile_group );
    params.tiles_k = params.split_k.slices;
    if( params.use_horizontal_cta_rasterization ) {
        grid.y = params.tiles_m;
        grid.x = params.tiles_n;
        params.tiles_y = params.tiles_m;
        params.tiles_x = params.tiles_n;
    } else {
        grid.x = params.tiles_m;
        grid.y = params.tiles_n;
        params.tiles_x = params.tiles_m;
        params.tiles_y = params.tiles_n;
    }
    grid.z = params.tiles_k;
    return 0;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ int find_first_valid_filter_position( int d,
                                                        int t,
                                                        int o,
                                                        int pad,
                                                        int stride,
                                                        int dilation,
                                                        int mul_stride,
                                                        int shr_stride ) {
    for( int ti = t - 1; ti >= 0; --ti ) {
        int oi = d + pad - ti * dilation;
        if( oi >= 0 ) {
            int div, mod;
            xmma::fast_divmod( div, mod, oi, stride, mul_stride, shr_stride );
            if( mod == 0 && div < o ) {
                return ti;
            }
        }
    }
    return -1;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

// Classifier : d/h/w pixels, (t+1)/(r+1)/(s+1) classes
template <typename Implicit_gemm_traits>
static __global__ void kernel_helper_stage_0( typename Implicit_gemm_traits::Params params ) {
    constexpr int THREADS_PER_CTA = 128;
    int tidx = blockIdx.x * THREADS_PER_CTA + threadIdx.x;
    if( tidx >= params.d + params.h + params.w ) {
        return;
    }
    int offset( 0 );
    int* ptr;
    int output_size, filter_size, pad, stride, dilation, mul_stride, shr_stride;
    if( tidx < params.d ) {
        offset = tidx;
        ptr = params.valid_t;
        output_size = params.o;
        filter_size = params.t;
        pad = params.pad[0][0];
        stride = params.stride[0];
        dilation = params.dilation[0];
        mul_stride = params.mul_stride_d;
        shr_stride = params.shr_stride_d;
    } else {
        if( tidx < params.d + params.h ) {
            offset = tidx - params.d;
            ptr = params.valid_r;
            output_size = params.p;
            filter_size = params.r;
            pad = params.pad[1][0];
            stride = params.stride[1];
            dilation = params.dilation[1];
            mul_stride = params.mul_stride_h;
            shr_stride = params.shr_stride_h;
        } else {
            offset = tidx - params.d - params.h;
            ptr = params.valid_s;
            output_size = params.q;
            filter_size = params.s;
            pad = params.pad[2][0];
            stride = params.stride[2];
            dilation = params.dilation[2];
            mul_stride = params.mul_stride_w;
            shr_stride = params.shr_stride_w;
        }
    }
    ptr[offset] = find_first_valid_filter_position(
        offset, filter_size, output_size, pad, stride, dilation, mul_stride, shr_stride );
    return;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

// Counter : Count the pixels' number of each filter pattern,
// (d x h x w) pixels, (t x r x s + 1) patterns
template <typename Implicit_gemm_traits>
static __global__ void kernel_helper_stage_1( typename Implicit_gemm_traits::Params params ) {
    int expected_idx_trs = params.trs - 1 - blockIdx.x;
    int expected_idx_t, expected_idx_r, expected_idx_s;
    if( expected_idx_trs >= 0 ) {
        int expected_idx_rs;
        xmma::fast_divmod( expected_idx_t,
                           expected_idx_rs,
                           expected_idx_trs,
                           params.rs,
                           params.mul_rs,
                           params.shr_rs );
        xmma::fast_divmod(
            expected_idx_r, expected_idx_s, expected_idx_rs, params.s, params.mul_s, params.shr_s );
    } else {
        expected_idx_t = -1;
        expected_idx_r = -1;
        expected_idx_s = -1;
    }
    __shared__ int total_count[3];
    if( threadIdx.x == 0 ) {
        total_count[0] = 0;
        total_count[1] = 0;
        total_count[2] = 0;
    }
    __syncthreads();
    constexpr int THREADS_PER_CTA = 256;
    int count[3];
    count[0] = 0;
    count[1] = 0;
    count[2] = 0;
    for( int i = threadIdx.x; i < params.d; i += THREADS_PER_CTA ) {
        int valid_t = params.valid_t[i];
        if( valid_t == expected_idx_t ) {
            ++count[0];
        }
    }
    atomicAdd( total_count, count[0] );
    for( int i = threadIdx.x; i < params.h; i += THREADS_PER_CTA ) {
        int valid_r = params.valid_r[i];
        if( valid_r == expected_idx_r ) {
            ++count[1];
        }
    }
    atomicAdd( total_count + 1, count[1] );
    for( int i = threadIdx.x; i < params.w; i += THREADS_PER_CTA ) {
        int valid_s = params.valid_s[i];
        if( valid_s == expected_idx_s ) {
            ++count[2];
        }
    }
    atomicAdd( total_count + 2, count[2] );
    __syncthreads();
    if( threadIdx.x == THREADS_PER_CTA - 1 ) {
        int real_total_count[3];
        real_total_count[0] = total_count[0];
        real_total_count[1] = total_count[1];
        real_total_count[2] = total_count[2];
        if( blockIdx.x == params.trs ) {
            params.dhw_count_of_each_filter_pattern[blockIdx.x] =
                params.dhw -
                ( params.d - real_total_count[0] ) * ( params.h - real_total_count[1] ) *
                    ( params.w - real_total_count[2] );
        } else {
            params.dhw_count_of_each_filter_pattern[blockIdx.x] =
                real_total_count[0] * real_total_count[1] * real_total_count[2];
        }
    }
    return;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

// Aclwmulator : f(n+1) = sum(f(0), f(1), ..., f(n))
template <typename Implicit_gemm_traits>
static __global__ void kernel_helper_stage_2( typename Implicit_gemm_traits::Params params ) {
    // The CTA tile.
    using Cta_tile = typename Implicit_gemm_traits::Cta_tile;

    // Tell cta which filter pattern it is belong to.
    int* ptr_in = params.dhw_count_of_each_filter_pattern;
    int* ptr_out = params.start_cta_id_in_ndhw_dimension_of_each_filter_pattern;
    int cta_number_of_previous_filter_patterns( 0 );
    for( int i = 0; i <= params.trs; ++i ) {
        int dhw_count = ptr_in[i];
        if( dhw_count == 0 ) {
            ptr_out[i] = -1;
        } else {
            ptr_out[i] = cta_number_of_previous_filter_patterns;
            cta_number_of_previous_filter_patterns +=
                ( dhw_count * params.n + Cta_tile::M - 1 ) / Cta_tile::M;
        }
    }
    return;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

// Classifier : (n x d x h x w) pixels, (t x r x s + 1) classes
template <typename Implicit_gemm_traits>
static __global__ void kernel_helper_stage_3( typename Implicit_gemm_traits::Params params ) {
    // The CTA tile.
    using Cta_tile = typename Implicit_gemm_traits::Cta_tile;
    int start_cta_id = params.start_cta_id_in_ndhw_dimension_of_each_filter_pattern[blockIdx.x];
    int dhw_count = params.dhw_count_of_each_filter_pattern[blockIdx.x];
    constexpr int THREADS_PER_CTA = 256;
    __shared__ int count[1];
    if( threadIdx.x == 0 ) {
        count[0] = 0;
    }
    __syncthreads();
    if( start_cta_id >= 0 ) {
        int* ptr = params.ndhw_indices_of_each_filter_pattern_gmem + start_cta_id * Cta_tile::M;
        int expected_idx_trs = params.trs - 1 - blockIdx.x;
        int expected_idx_t, expected_idx_r, expected_idx_s;
        if( expected_idx_trs >= 0 ) {
            int expected_idx_rs;
            xmma::fast_divmod( expected_idx_t,
                               expected_idx_rs,
                               expected_idx_trs,
                               params.rs,
                               params.mul_rs,
                               params.shr_rs );
            xmma::fast_divmod( expected_idx_r,
                               expected_idx_s,
                               expected_idx_rs,
                               params.s,
                               params.mul_s,
                               params.shr_s );
        } else {
            expected_idx_t = -1;
            expected_idx_r = -1;
            expected_idx_s = -1;
        }
        for( int i = threadIdx.x; i < params.dhw; i += THREADS_PER_CTA ) {
            int di, hwi;
            xmma::fast_divmod( di, hwi, i, params.hw, params.mul_hw, params.shr_hw );
            int hi, wi;
            xmma::fast_divmod( hi, wi, hwi, params.w, params.mul_w, params.shr_w );
            int valid_t, valid_r, valid_s;
            valid_t = params.valid_t[di];
            valid_r = params.valid_r[hi];
            valid_s = params.valid_s[wi];
            bool is_a_valid_position( false );
            if( valid_t >= 0 && valid_r >= 0 && valid_s >= 0 ) {
                is_a_valid_position = ( valid_t == expected_idx_t ) &&
                                      ( valid_r == expected_idx_r ) &&
                                      ( valid_s == expected_idx_s );
            } else {
                is_a_valid_position = ( expected_idx_trs < 0 );
            }
            if( is_a_valid_position ) {
                int idx = atomicAdd( count, 1 );
                ptr[idx] = i;
                for( int n_idx = 1; n_idx < params.n; ++n_idx ) {
                    ptr[idx + n_idx * dhw_count] = n_idx * params.dhw + i;
                }
            }
        }
        int idx = dhw_count * params.n + threadIdx.x;
        int cta_count = ( dhw_count * params.n + Cta_tile::M - 1 ) / Cta_tile::M;
        for( ; idx < cta_count * Cta_tile::M; idx += THREADS_PER_CTA ) {
            ptr[idx] = -1;
        }
    }
    return;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace strided_dgrad_indexed
}  // namespace implicit_gemm
}  // namespace xmma

///////////////////////////////////////////////////////////////////////////////////////////////////
