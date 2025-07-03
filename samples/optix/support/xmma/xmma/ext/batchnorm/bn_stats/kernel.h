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

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace xmma {
namespace ext {
namespace batchnorm {
namespace bn_stats {

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int DIM_Y, bool DUAL_DBNS = false, typename Params, int STATS_PER_THREAD>
inline __device__ void xmma_bn_sums( float ( &sums )[STATS_PER_THREAD],
                                     float ( &sums_of_squares )[STATS_PER_THREAD],
                                     const Params &params ) {

    // Shared memory.
    __shared__ float2 smem[DIM_Y][32 * STATS_PER_THREAD];

    // The position of the thread.
    const int idx = blockIdx.x * 32 * STATS_PER_THREAD + threadIdx.x * STATS_PER_THREAD;

    float *bn_partial_sums_gmem;
    float *bn_partial_sums_of_squares_gmem;

    if( DUAL_DBNS ) {
        bn_partial_sums_gmem = (float *)params.bn_partial_sums_gmem;
        bn_partial_sums_of_squares_gmem = (float *)params.bn_partial_dual_sums_of_squares_gmem;
    } else {
        bn_partial_sums_gmem = (float *)params.bn_partial_sums_gmem;
        bn_partial_sums_of_squares_gmem = (float *)params.bn_partial_sums_of_squares_gmem;
    }

    // The fragment of data.
    using Fragment = xmma::Fragment<float, STATS_PER_THREAD>;
    // The array of partial sums.
    const float *partial_sums = &bn_partial_sums_gmem[idx];
    // The array of partial sums of squares.
    const float *partial_sums_of_squares = &bn_partial_sums_of_squares_gmem[idx];

    // Each thread computes 1 mean.
    Fragment sum;
    sum.clear();
    Fragment sum_of_squares;
    sum_of_squares.clear();
#pragma unroll
    for( int i = threadIdx.y; i < params.num_partial_sums; i += DIM_Y ) {
        const int offset = i * params.num_channels;
        if( idx < params.num_channels ) {
            sum.add( reinterpret_cast<const Fragment *>( &partial_sums[offset] )[0] );
            sum_of_squares.add(
                reinterpret_cast<const Fragment *>( &partial_sums_of_squares[offset] )[0] );
        }
    }

    // Create pairs of sums/sums of squares.
    float2 a[STATS_PER_THREAD];
#pragma unroll
    for( int i = 0; i < STATS_PER_THREAD; ++i ) {
        a[i] = make_float2( sum.elt( i ), sum_of_squares.elt( i ) );
    }

// Store data into shared memory.
#pragma unroll
    for( int i = 0; i < STATS_PER_THREAD; ++i ) {
        smem[threadIdx.y][threadIdx.x + 32 * i] = a[i];
    }

    // Make sure the data is in shared memory.
    __syncthreads();

    // Parallel reduction.
    for( int offset = DIM_Y / 2; offset > 0; offset /= 2 ) {
        if( threadIdx.y < offset ) {
            float2 b[STATS_PER_THREAD];
#pragma unroll
            for( int i = 0; i < STATS_PER_THREAD; ++i ) {
                b[i] = smem[threadIdx.y + offset][threadIdx.x + 32 * i];
            }
#pragma unroll
            for( int i = 0; i < STATS_PER_THREAD; ++i ) {
                a[i].x += b[i].x;
                a[i].y += b[i].y;
            }
#pragma unroll
            for( int i = 0; i < STATS_PER_THREAD; ++i ) {
                smem[threadIdx.y][threadIdx.x + 32 * i] = a[i];
            }
        }
        __syncthreads();
    }

// Read the values back from memory.
#pragma unroll
    for( int i = 0; i < STATS_PER_THREAD; ++i ) {
        a[i] = smem[0][threadIdx.x + 32 * i];
    }

// The final sums.
#pragma unroll
    for( int i = 0; i < STATS_PER_THREAD; ++i ) {
        sums[i] = a[i].x;
        sums_of_squares[i] = a[i].y;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int DIM_Y, typename Params, int STATS_PER_THREAD>
inline __device__ void xmma_bn_stats( float ( &sum )[STATS_PER_THREAD],
                                      float ( &mean )[STATS_PER_THREAD],
                                      float ( &sum_of_squares )[STATS_PER_THREAD],
                                      float ( &ilw_stddev )[STATS_PER_THREAD],
                                      const Params &params ) {

    // Compute the sums.
    xmma_bn_sums<DIM_Y>( sum, sum_of_squares, params );

    // Compute the mean and the ilw stddev.
    if( !params.bn_disable_stats_output ) {
#pragma unroll
        for( int i = 0; i < STATS_PER_THREAD; ++i ) {
            mean[i] = sum[i] * params.ilw_count;
            float var = sum_of_squares[i] * params.ilw_count - mean[i] * mean[i];
            ilw_stddev[i] = rsqrtf( var + params.bn_epsilon );
        }
    }
}

template <int DIM_Y, typename Params, int STATS_PER_THREAD>
inline __device__ void xmma_dbn_stats( float ( &sum )[STATS_PER_THREAD],
                                       float ( &mean )[STATS_PER_THREAD],
                                       float ( &sum_of_squares )[STATS_PER_THREAD],
                                       float ( &ilw_stddev )[STATS_PER_THREAD],
                                       const Params &params ) {

    // Compute the sums.
    xmma_bn_sums<DIM_Y>( sum, sum_of_squares, params );

    // Compute the mean and the ilw stddev.
    if( !params.bn_disable_stats_output ) {
#pragma unroll
        for( int i = 0; i < STATS_PER_THREAD; ++i ) {
            mean[i] = sum[i] * params.ilw_count;
            ilw_stddev[i] = sum_of_squares[i] * params.ilw_count;
        }
    }
}

template <int DIM_Y, typename Params, int STATS_PER_THREAD>
inline __device__ void xmma_dual_dbn_stats( float ( &sum )[STATS_PER_THREAD],
                                            float ( &mean )[STATS_PER_THREAD],
                                            float ( &sum_of_squares )[STATS_PER_THREAD],
                                            float ( &ilw_stddev )[STATS_PER_THREAD],
                                            float ( &dual_sum )[STATS_PER_THREAD],
                                            float ( &dual_mean )[STATS_PER_THREAD],
                                            float ( &dual_sum_of_squares )[STATS_PER_THREAD],
                                            float ( &dual_ilw_stddev )[STATS_PER_THREAD],
                                            const Params &params ) {

    // Compute the sums.
    xmma_bn_sums<DIM_Y>( sum, sum_of_squares, params );
    xmma_bn_sums<DIM_Y, true>( dual_sum, dual_sum_of_squares, params );

    // Compute the mean and the ilw stddev.
    if( !params.bn_disable_stats_output ) {
#pragma unroll
        for( int i = 0; i < STATS_PER_THREAD; ++i ) {
            mean[i] = sum[i] * params.ilw_count;
            ilw_stddev[i] = sum_of_squares[i] * params.ilw_count;
            dual_mean[i] = mean[i];
            dual_ilw_stddev[i] = dual_sum_of_squares[i] * params.ilw_count;
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int DIM_Y, typename Params> static __global__ void bn_stats_kernel( Params params ) {

    // Compute the mean and the ilwerse of the stddev.
    float sum[1], mean[1], sum_of_squares[1], ilw_stddev[1];
    xmma_bn_stats<DIM_Y>( sum, mean, sum_of_squares, ilw_stddev, params );

    __syncthreads();

    // The position of the thread.
    const int idx = blockIdx.x * 32 + threadIdx.x;

    // Store the results.
    if( ( threadIdx.y == 0 ) && ( idx < params.num_channels ) &&
        ( !params.bn_disable_stats_output ) ) {
        float *bn_sum_gmem = (float *)params.bn_sum_gmem;
        float *bn_mean_gmem = (float *)params.bn_mean_gmem;
        float *bn_sum_of_squares_gmem = (float *)params.bn_sum_of_squares_gmem;
        float *bn_ilw_stddev_gmem = (float *)params.bn_ilw_stddev_gmem;

        bn_sum_gmem[idx] = sum[0];
        bn_mean_gmem[idx] = mean[0];
        bn_sum_of_squares_gmem[idx] = sum_of_squares[0];
        bn_ilw_stddev_gmem[idx] = ilw_stddev[0];
    }
}

template <int DIM_Y, typename Params> static __global__ void dbn_stats_kernel( Params params ) {

    // Compute the mean and the ilwerse of the stddev.
    float sum[1], mean[1], sum_of_squares[1], ilw_stddev[1];
    xmma_dbn_stats<DIM_Y>( sum, mean, sum_of_squares, ilw_stddev, params );

    __syncthreads();

    // The position of the thread.
    const int idx = blockIdx.x * 32 + threadIdx.x;

    // Store the results.
    if( ( threadIdx.y == 0 ) && ( idx < params.num_channels ) &&
        ( !params.bn_disable_stats_output ) ) {
        float *bn_sum_gmem = (float *)params.bn_sum_gmem;
        float *bn_mean_gmem = (float *)params.bn_mean_gmem;
        float *bn_sum_of_squares_gmem = (float *)params.bn_sum_of_squares_gmem;
        float *bn_ilw_stddev_gmem = (float *)params.bn_ilw_stddev_gmem;

        bn_sum_gmem[idx] = sum[0];
        bn_mean_gmem[idx] = mean[0];
        bn_sum_of_squares_gmem[idx] = sum_of_squares[0];
        bn_ilw_stddev_gmem[idx] = ilw_stddev[0];
    }
}

template <int DIM_Y, typename Params>
static __global__ void dual_dbn_stats_kernel( Params params ) {

    // Compute the mean and the ilwerse of the stddev.
    float sum[1], mean[1], sum_of_squares[1], ilw_stddev[1], dual_sum[1], dual_mean[1],
        dual_sum_of_squares[1], dual_ilw_stddev[1];
    xmma_dual_dbn_stats<DIM_Y>( sum,
                                mean,
                                sum_of_squares,
                                ilw_stddev,
                                dual_sum,
                                dual_mean,
                                dual_sum_of_squares,
                                dual_ilw_stddev,
                                params );

    __syncthreads();

    // The position of the thread.
    const int idx = blockIdx.x * 32 + threadIdx.x;

    // Store the results.
    if( ( threadIdx.y == 0 ) && ( idx < params.num_channels ) &&
        ( !params.bn_disable_stats_output ) ) {
        float *bn_sum_gmem = (float *)params.bn_sum_gmem;
        float *bn_mean_gmem = (float *)params.bn_mean_gmem;
        float *bn_sum_of_squares_gmem = (float *)params.bn_sum_of_squares_gmem;
        float *bn_ilw_stddev_gmem = (float *)params.bn_ilw_stddev_gmem;

        float *dual_bn_sum_gmem = (float *)params.dual_bn_sum_gmem;
        float *dual_bn_mean_gmem = (float *)params.dual_bn_mean_gmem;
        float *dual_bn_sum_of_squares_gmem = (float *)params.dual_bn_sum_of_squares_gmem;
        float *dual_bn_ilw_stddev_gmem = (float *)params.dual_bn_ilw_stddev_gmem;

        bn_sum_gmem[idx] = sum[0];
        bn_mean_gmem[idx] = mean[0];
        bn_sum_of_squares_gmem[idx] = sum_of_squares[0];
        bn_ilw_stddev_gmem[idx] = ilw_stddev[0];

        dual_bn_sum_gmem[idx] = sum[0];
        dual_bn_mean_gmem[idx] = mean[0];
        dual_bn_sum_of_squares_gmem[idx] = dual_sum_of_squares[0];
        dual_bn_ilw_stddev_gmem[idx] = dual_ilw_stddev[0];
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// dbn(a) =>
// value[i,k] = fprop_alpha[k] * fprop_ilw_std[k] * [ { dy[i,k] - dbias[k] } -
//     dscale[k] * fprop_ilw_std[k] * fprop_ilw_std[k] * { fprop_tensor[i,k] - fprop_mean[k] }  ]

// Design Considertaions
// 1) Maximize L2 hit rate for dy and fprop_tensor.
//    During the epilogue dy is written to DRAM / L2 and fprop_tensor is read from DRAM to L2
//    The goal is to match the CTA raster order (ilwerse) to maximize reuse of the L2
// 2) Maximize DRAM B/W -- this should be a fully DRAM B/W bound kernel
////////////////////////////////////////////////////////////////////////////////////////////////////
template <int THREADS_PER_CTA, typename Params>
static __global__ void dbn_apply_kernel( Params params, uint32_t C_ELEMENTS_PER_CTA ) {

    const uint32_t ELEMENTS_PER_LDG = 8;
    uint32_t ndhw = params.n * params.d * params.h * params.w;
    uint32_t ROWS_PER_CTA = xmma::div_up( ndhw, gridDim.x );
    uint32_t THREADS_PER_NDHW = C_ELEMENTS_PER_CTA / ELEMENTS_PER_LDG;
    uint32_t THREADS_PER_C = min( xmma::div_up( THREADS_PER_CTA, THREADS_PER_NDHW ), ROWS_PER_CTA );
    uint32_t ROWS_PER_THREAD = xmma::div_up( ROWS_PER_CTA, THREADS_PER_C );
    uint32_t cta_c = blockIdx.y * C_ELEMENTS_PER_CTA;
    uint32_t cta_ndhw = blockIdx.x * ROWS_PER_CTA;
    uint32_t thread_c = cta_c + ( threadIdx.x % THREADS_PER_NDHW ) * ELEMENTS_PER_LDG;
    uint32_t thread_ndhw_base = cta_ndhw + ( threadIdx.x / THREADS_PER_NDHW ) * ROWS_PER_THREAD;
    uint32_t num_channels = params.c * params.g;
    uint32_t offset = thread_ndhw_base * num_channels + thread_c;

    if( thread_c < num_channels && thread_ndhw_base < ndhw ) {

        float falpha[8];
        ;
        float fmean[8];
        float filwvar[8];
        float dscale[8], dbias[8];

        float *ptr_fmean = (float *)params.bn_fprop_mean_gmem;
        float *ptr_filwvar = (float *)params.bn_fprop_ilw_stddev_gmem;
        float *ptr_falpha = (float *)params.bn_fprop_alpha_gmem;
        float *ptr_dscale = (float *)params.bn_ilw_stddev_gmem;
        float *ptr_dbias = (float *)params.bn_mean_gmem;

#pragma unroll
        for( int jj = 0; jj < ELEMENTS_PER_LDG; jj++ ) {
            fmean[jj] = ptr_fmean[thread_c + jj];
            filwvar[jj] = ptr_filwvar[thread_c + jj];
            falpha[jj] = ptr_falpha[thread_c + jj];
            dscale[jj] = ptr_dscale[thread_c + jj];
            dbias[jj] = ptr_dbias[thread_c + jj];
        }

#pragma unroll
        for( int jj = 0; jj < ELEMENTS_PER_LDG; jj++ ) {
            // dscale[k] * fprop_ilw_std[k] * fprop_ilw_std[k]
            dscale[jj] *= filwvar[jj] * filwvar[jj];

            // fprop_alpha[k] * fprop_ilw_std[k]
            falpha[jj] *= filwvar[jj];
        }
#pragma unroll
        for( int ii = 0; ii < ROWS_PER_THREAD; ii++ ) {

            uint32_t thread_ndhw = thread_ndhw_base + ii;
            uint32_t toffset = offset + ii * num_channels;

            if( thread_ndhw < ndhw && thread_c < params.g * params.c ) {

                // 2 is to get the bytes location
                char *ptr_fx = (char *)params.bn_fprop_tensor_gmem + toffset * 2;
                char *ptr_out = (char *)params.standalone_dbna_output + toffset * 2;
                if( params.overwrite_out_gmem ) {
                    ptr_out = (char *)params.out_gmem + toffset * 2;
                }
                char *ptr_dy = (char *)params.out_gmem + toffset * 2;

                uint4 dy, fx;
                xmma::ldg_cs( dy, ptr_dy );
                xmma::ldg_cs( fx, ptr_fx );

                //  { dy[i,k] - dbias[k] }
                float2 dy_f[4];
                dy_f[0] = xmma::half2_to_float2( dy.x );
                dy_f[1] = xmma::half2_to_float2( dy.y );
                dy_f[2] = xmma::half2_to_float2( dy.w );
                dy_f[3] = xmma::half2_to_float2( dy.z );

                dy_f[0].x -= dbias[0];
                dy_f[0].y -= dbias[1];
                dy_f[1].x -= dbias[2];
                dy_f[1].y -= dbias[3];
                dy_f[2].x -= dbias[4];
                dy_f[2].y -= dbias[5];
                dy_f[3].x -= dbias[6];
                dy_f[3].y -= dbias[7];

                // { fprop_tensor[i,k] - fprop_mean[k] }
                float2 fx_f[4];
                fx_f[0] = xmma::half2_to_float2( fx.x );
                fx_f[1] = xmma::half2_to_float2( fx.y );
                fx_f[2] = xmma::half2_to_float2( fx.w );
                fx_f[3] = xmma::half2_to_float2( fx.z );

                fx_f[0].x -= fmean[0];
                fx_f[0].y -= fmean[1];
                fx_f[1].x -= fmean[2];
                fx_f[1].y -= fmean[3];
                fx_f[2].x -= fmean[4];
                fx_f[2].y -= fmean[5];
                fx_f[3].x -= fmean[6];
                fx_f[3].y -= fmean[7];

                float2 out[4];
                out[0].x = falpha[0] * ( dy_f[0].x - dscale[0] * fx_f[0].x );
                out[0].y = falpha[1] * ( dy_f[0].y - dscale[1] * fx_f[0].y );
                out[1].x = falpha[2] * ( dy_f[1].x - dscale[2] * fx_f[1].x );
                out[1].y = falpha[3] * ( dy_f[1].y - dscale[3] * fx_f[1].y );
                out[2].x = falpha[4] * ( dy_f[2].x - dscale[4] * fx_f[2].x );
                out[2].y = falpha[5] * ( dy_f[2].y - dscale[5] * fx_f[2].y );
                out[3].x = falpha[6] * ( dy_f[3].x - dscale[6] * fx_f[3].x );
                out[3].y = falpha[7] * ( dy_f[3].y - dscale[7] * fx_f[3].y );

                uint4 outc;
                outc.x = xmma::float2_to_half2( out[0].x, out[0].y );
                outc.y = xmma::float2_to_half2( out[1].x, out[1].y );
                outc.w = xmma::float2_to_half2( out[2].x, out[2].y );
                outc.z = xmma::float2_to_half2( out[3].x, out[3].y );

                // Add mem_descriptor
                xmma::stg( ptr_out, outc );

            }  // end if
        }      // end for
    }          // end if
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// dbn(a) =>
// value[i,k] = fprop_alpha[k] * fprop_ilw_std[k] * [ { dy[i,k] - dbias[k] } -
//     dscale[k] * fprop_ilw_std[k] * fprop_ilw_std[k] * { fprop_tensor[i,k] - fprop_mean[k] }  ]

// Design Considertaions
// 1) Maximize L2 hit rate for dy and fprop_tensor.
//    During the epilogue dy is written to DRAM / L2 and fprop_tensor is read from DRAM to L2
//    The goal is to match the CTA raster order (ilwerse) to maximize reuse of the L2
// 2) Maximize DRAM B/W -- this should be a fully DRAM B/W bound kernel
////////////////////////////////////////////////////////////////////////////////////////////////////
template <int THREADS_PER_CTA, typename Params>
static __global__ void dual_dbn_apply_kernel( Params params, uint32_t C_ELEMENTS_PER_CTA ) {

    uint32_t ELEMENTS_PER_LDG = 8;
    uint32_t ndhw = params.n * params.d * params.h * params.w;
    uint32_t ROWS_PER_CTA = xmma::div_up( ndhw, gridDim.x );
    uint32_t THREADS_PER_NDHW = C_ELEMENTS_PER_CTA / ELEMENTS_PER_LDG;
    uint32_t THREADS_PER_C = min( xmma::div_up( THREADS_PER_CTA, THREADS_PER_NDHW ), ROWS_PER_CTA );
    uint32_t ROWS_PER_THREAD = xmma::div_up( ROWS_PER_CTA, THREADS_PER_C );
    uint32_t cta_c = blockIdx.y * C_ELEMENTS_PER_CTA;
    uint32_t cta_ndhw = blockIdx.x * ROWS_PER_CTA;
    uint32_t thread_c = cta_c + ( threadIdx.x % THREADS_PER_NDHW ) * ELEMENTS_PER_LDG;
    uint32_t thread_ndhw_base = cta_ndhw + ( threadIdx.x / THREADS_PER_NDHW ) * ROWS_PER_THREAD;
    uint32_t num_channels = params.c * params.g;
    uint32_t offset = thread_ndhw_base * num_channels + thread_c;

    if( thread_c < num_channels && thread_ndhw_base < ndhw ) {

        float falpha[8];
        ;
        float fmean[8];
        float filwvar[8];
        float dscale[8], dbias[8];

        float dual_falpha[8];
        float dual_fmean[8];
        float dual_filwvar[8];
        float dual_dscale[8];

        float *ptr_fmean = (float *)params.bn_fprop_mean_gmem;
        float *ptr_filwvar = (float *)params.bn_fprop_ilw_stddev_gmem;
        float *ptr_falpha = (float *)params.bn_fprop_alpha_gmem;
        float *ptr_dscale = (float *)params.bn_ilw_stddev_gmem;
        float *ptr_dbias = (float *)params.bn_mean_gmem;

        float *ptr_dual_fmean = (float *)params.dual_bn_fprop_mean_gmem;
        float *ptr_dual_filwvar = (float *)params.dual_bn_fprop_ilw_stddev_gmem;
        float *ptr_dual_falpha = (float *)params.dual_bn_fprop_alpha_gmem;
        float *ptr_dual_dscale = (float *)params.dual_bn_ilw_stddev_gmem;

#pragma unroll
        for( int jj = 0; jj < ELEMENTS_PER_LDG; jj++ ) {
            fmean[jj] = ptr_fmean[thread_c + jj];
            filwvar[jj] = ptr_filwvar[thread_c + jj];
            falpha[jj] = ptr_falpha[thread_c + jj];
            dscale[jj] = ptr_dscale[thread_c + jj];
            dbias[jj] = ptr_dbias[thread_c + jj];
            dual_fmean[jj] = ptr_dual_fmean[thread_c + jj];
            dual_filwvar[jj] = ptr_dual_filwvar[thread_c + jj];
            dual_falpha[jj] = ptr_dual_falpha[thread_c + jj];
            dual_dscale[jj] = ptr_dual_dscale[thread_c + jj];
        }

#pragma unroll
        for( int jj = 0; jj < ELEMENTS_PER_LDG; jj++ ) {
            // dscale[k] * fprop_ilw_std[k] * fprop_ilw_std[k]
            dscale[jj] *= filwvar[jj] * filwvar[jj];
            dual_dscale[jj] *= dual_filwvar[jj] * dual_filwvar[jj];

            // fprop_alpha[k] * fprop_ilw_std[k]
            falpha[jj] *= filwvar[jj];
            dual_falpha[jj] *= dual_filwvar[jj];
            ;
        }
#pragma unroll
        for( int ii = 0; ii < ROWS_PER_THREAD; ii++ ) {

            uint32_t thread_ndhw = thread_ndhw_base + ii;
            uint32_t toffset = offset + ii * num_channels;

            if( thread_ndhw < ndhw && thread_c < params.g * params.c ) {

                // 2 is to get the bytes location
                char *ptr_fx = (char *)params.bn_fprop_tensor_gmem + toffset * 2;
                char *ptr_out = (char *)params.standalone_dbna_output + toffset * 2;
                if( params.overwrite_out_gmem ) {
                    ptr_out = (char *)params.out_gmem + toffset * 2;
                }
                char *ptr_dy = (char *)params.out_gmem + toffset * 2;

                // 2 is to get the bytes location
                char *ptr_dual_fx = (char *)params.bn_fprop_tensor_gmem + toffset * 2;
                char *ptr_dual_out = (char *)params.dual_standalone_dbna_output + toffset * 2;

                uint4 dy, fx, dual_fx;
                xmma::ldg_cs( dy, ptr_dy );
                xmma::ldg_cs( fx, ptr_fx );
                xmma::ldg_cs( dual_fx, ptr_dual_fx );

                //  { dy[i,k] - dbias[k] }
                float2 dy_f[4];
                dy_f[0] = xmma::half2_to_float2( dy.x );
                dy_f[1] = xmma::half2_to_float2( dy.y );
                dy_f[2] = xmma::half2_to_float2( dy.w );
                dy_f[3] = xmma::half2_to_float2( dy.z );

                dy_f[0].x -= dbias[0];
                dy_f[0].y -= dbias[1];
                dy_f[1].x -= dbias[2];
                dy_f[1].y -= dbias[3];
                dy_f[2].x -= dbias[4];
                dy_f[2].y -= dbias[5];
                dy_f[3].x -= dbias[6];
                dy_f[3].y -= dbias[7];

                // { fprop_tensor[i,k] - fprop_mean[k] }
                float2 fx_f[4];
                fx_f[0] = xmma::half2_to_float2( fx.x );
                fx_f[1] = xmma::half2_to_float2( fx.y );
                fx_f[2] = xmma::half2_to_float2( fx.w );
                fx_f[3] = xmma::half2_to_float2( fx.z );

                fx_f[0].x -= fmean[0];
                fx_f[0].y -= fmean[1];
                fx_f[1].x -= fmean[2];
                fx_f[1].y -= fmean[3];
                fx_f[2].x -= fmean[4];
                fx_f[2].y -= fmean[5];
                fx_f[3].x -= fmean[6];
                fx_f[3].y -= fmean[7];

                // { fprop_tensor[i,k] - fprop_mean[k] }
                float2 dual_fx_f[4];
                dual_fx_f[0] = xmma::half2_to_float2( dual_fx.x );
                dual_fx_f[1] = xmma::half2_to_float2( dual_fx.y );
                dual_fx_f[2] = xmma::half2_to_float2( dual_fx.w );
                dual_fx_f[3] = xmma::half2_to_float2( dual_fx.z );

                dual_fx_f[0].x -= dual_fmean[0];
                dual_fx_f[0].y -= dual_fmean[1];
                dual_fx_f[1].x -= dual_fmean[2];
                dual_fx_f[1].y -= dual_fmean[3];
                dual_fx_f[2].x -= dual_fmean[4];
                dual_fx_f[2].y -= dual_fmean[5];
                dual_fx_f[3].x -= dual_fmean[6];
                dual_fx_f[3].y -= dual_fmean[7];

                float2 out[4];
                out[0].x = falpha[0] * ( dy_f[0].x - dscale[0] * fx_f[0].x );
                out[0].y = falpha[1] * ( dy_f[0].y - dscale[1] * fx_f[0].y );
                out[1].x = falpha[2] * ( dy_f[1].x - dscale[2] * fx_f[1].x );
                out[1].y = falpha[3] * ( dy_f[1].y - dscale[3] * fx_f[1].y );
                out[2].x = falpha[4] * ( dy_f[2].x - dscale[4] * fx_f[2].x );
                out[2].y = falpha[5] * ( dy_f[2].y - dscale[5] * fx_f[2].y );
                out[3].x = falpha[6] * ( dy_f[3].x - dscale[6] * fx_f[3].x );
                out[3].y = falpha[7] * ( dy_f[3].y - dscale[7] * fx_f[3].y );

                float2 dual_out[4];
                dual_out[0].x = dual_falpha[0] * ( dy_f[0].x - dual_dscale[0] * dual_fx_f[0].x );
                dual_out[0].y = dual_falpha[1] * ( dy_f[0].y - dual_dscale[1] * dual_fx_f[0].y );
                dual_out[1].x = dual_falpha[2] * ( dy_f[1].x - dual_dscale[2] * dual_fx_f[1].x );
                dual_out[1].y = dual_falpha[3] * ( dy_f[1].y - dual_dscale[3] * dual_fx_f[1].y );
                dual_out[2].x = dual_falpha[4] * ( dy_f[2].x - dual_dscale[4] * dual_fx_f[2].x );
                dual_out[2].y = dual_falpha[5] * ( dy_f[2].y - dual_dscale[5] * dual_fx_f[2].y );
                dual_out[3].x = dual_falpha[6] * ( dy_f[3].x - dual_dscale[6] * dual_fx_f[3].x );
                dual_out[3].y = dual_falpha[7] * ( dy_f[3].y - dual_dscale[7] * dual_fx_f[3].y );

                uint4 outc;
                outc.x = xmma::float2_to_half2( out[0].x, out[0].y );
                outc.y = xmma::float2_to_half2( out[1].x, out[1].y );
                outc.w = xmma::float2_to_half2( out[2].x, out[2].y );
                outc.z = xmma::float2_to_half2( out[3].x, out[3].y );

                uint4 dual_outc;
                dual_outc.x = xmma::float2_to_half2( dual_out[0].x, dual_out[0].y );
                dual_outc.y = xmma::float2_to_half2( dual_out[1].x, dual_out[1].y );
                dual_outc.w = xmma::float2_to_half2( dual_out[2].x, dual_out[2].y );
                dual_outc.z = xmma::float2_to_half2( dual_out[3].x, dual_out[3].y );

                // Add mem_descriptor
                xmma::stg( ptr_out, outc );
                xmma::stg( ptr_out, dual_outc );

            }  // end if
        }      // end for
    }          // end if
}
////////////////////////////////////////////////////////////////////////////////////////////////////
}  // namespace bn_stats
}  // namespace batchnorm
}  // namespace ext
}  // namespace xmma
////////////////////////////////////////////////////////////////////////////////////////////////////
