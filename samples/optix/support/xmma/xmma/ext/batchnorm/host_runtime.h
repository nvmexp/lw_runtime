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

#include <xmma/ext/batchnorm/bn_device_arrays.h>
#include <xmma/ext/batchnorm/bn_stats/kernel.h>
#include <xmma/implicit_gemm/host_runtime.h>

///////////////////////////////////////////////////////////////////////////////////////////////////

namespace xmma {
namespace ext {
namespace batchnorm {

///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Implicit_gemm_traits> struct Runtime_params {
    int32_t descriptor_a;
    int32_t descriptor_b;
    int32_t descriptor_c0;
    int32_t descriptor_c1;
    int32_t descriptor_d0;
    int32_t descriptor_d1;
};

//////////////////////////////// Initialize device pointers ///////////////////////////////////////

template <typename Traits, xmma::Operation_type operation_type> struct Device_pointers {};

template <typename Traits> struct Device_pointers<Traits, xmma::Operation_type::DGRAD> {

    static void init( typename Traits::Params &params,
                      void *x_data,
                      const void *y_data,
                      const void *w_data,
                      const void *res_data,
                      const Bn_Device_Arrays &bn_arrays ) {

        params.img_gmem = w_data;
        params.flt_gmem = y_data;
        params.out_gmem = x_data;
        params.res_gmem = res_data;

        // batchnorm specific parameters
        params.bn_sum_gmem = bn_arrays.bn_sum_gmem;
        params.bn_mean_gmem = bn_arrays.bn_mean_gmem;
        params.bn_sum_of_squares_gmem = bn_arrays.bn_sum_of_squares_gmem;
        params.bn_ilw_stddev_gmem = bn_arrays.bn_ilw_stddev_gmem;

        params.bn_partial_sums_gmem = bn_arrays.bn_partial_sums_gmem;
        params.bn_scale_gmem = bn_arrays.bn_scale_gmem;
        params.bn_bias_gmem = bn_arrays.bn_bias_gmem;
        params.bn_out_gmem = bn_arrays.bn_out_gmem;

        params.bn_fprop_mean_gmem = bn_arrays.bn_fprop_mean_gmem;
        params.bn_fprop_tensor_gmem = bn_arrays.bn_fprop_tensor_gmem;
        params.bn_fprop_ilw_stddev_gmem = bn_arrays.bn_fprop_ilw_stddev_gmem;

        // standalone bn(a)
        params.bn_fprop_alpha_gmem = bn_arrays.bn_fprop_alpha_gmem;
        params.standalone_dbna_output = bn_arrays.standalone_dbna_output;

        // dbn(a)
        params.bna_fprop_tensor_gmem = bn_arrays.bna_fprop_tensor_gmem;
        params.bna_fprop_tensor_scale_gmem = bn_arrays.bna_fprop_tensor_scale_gmem;
        params.bna_gradient_scale_gmem = bn_arrays.bna_grad_scale_gmem;
        params.bna_bias_gmem = bn_arrays.bna_bias_gmem;

        // dRelu bitmask
        params.bn_drelu_bitmask = bn_arrays.bn_drelu_bitmask;

        // dual_dbns
        params.dual_bn_sum_gmem = bn_arrays.dual_bn_sum_gmem;
        params.dual_bn_mean_gmem = bn_arrays.dual_bn_mean_gmem;
        params.dual_bn_sum_of_squares_gmem = bn_arrays.dual_bn_sum_of_squares_gmem;
        params.dual_bn_ilw_stddev_gmem = bn_arrays.dual_bn_ilw_stddev_gmem;

        params.dual_bn_fprop_mean_gmem = bn_arrays.dual_bn_fprop_mean_gmem;
        params.dual_bn_fprop_tensor_gmem = bn_arrays.dual_bn_fprop_tensor_gmem;
        params.dual_bn_fprop_ilw_stddev_gmem = bn_arrays.dual_bn_fprop_ilw_stddev_gmem;
        params.dual_bn_fprop_alpha_gmem = bn_arrays.dual_bn_fprop_alpha_gmem;

        params.dual_standalone_dbna_output = bn_arrays.dual_standalone_dbna_output;
    }
};

template <typename Traits> struct Device_pointers<Traits, xmma::Operation_type::FPROP> {

    static void init( typename Traits::Params &params,
                      const void *x_data,
                      const void *y_data,
                      void *w_data,
                      const void *res_data,
                      const Bn_Device_Arrays &bn_arrays ) {

        params.img_gmem = x_data;
        params.flt_gmem = y_data;
        params.out_gmem = w_data;
        params.res_gmem = res_data;

        // batchnorm specific parameters
        params.bn_sum_gmem = bn_arrays.bn_sum_gmem;
        params.bn_mean_gmem = bn_arrays.bn_mean_gmem;
        params.bn_sum_of_squares_gmem = bn_arrays.bn_sum_of_squares_gmem;
        params.bn_ilw_stddev_gmem = bn_arrays.bn_ilw_stddev_gmem;

        params.bn_partial_sums_gmem = bn_arrays.bn_partial_sums_gmem;
        params.bn_scale_gmem = bn_arrays.bn_scale_gmem;
        params.bn_bias_gmem = bn_arrays.bn_bias_gmem;
        params.bn_out_gmem = bn_arrays.bn_out_gmem;

        // Ressidual add tensor pointer
        params.bn_res_gmem = bn_arrays.bn_residual_gmem;

        // Result of residual add - write out
        params.bn_res_add_relu_out_gmem = bn_arrays.bn_res_add_relu_out_gmem;

        // Bitmask relu - write out
        params.bn_bitmask_relu_out_gmem = bn_arrays.bn_bitmask_relu_out_gmem;

        // Residual tensor's scale and bias if it needs a BNA
        params.bn_res_scale_gmem = bn_arrays.bn_res_scale_gmem;
        params.bn_res_bias_gmem = bn_arrays.bn_res_bias_gmem;
    }
};

template <typename Traits> struct Device_pointers<Traits, xmma::Operation_type::WGRAD> {

    static void init( typename Traits::Params &params,
                      const void *x_data,
                      void *y_data,
                      const void *w_data,
                      const void *res_data,
                      const Bn_Device_Arrays &bn_arrays ) {

        params.img_gmem = x_data;
        params.flt_gmem = w_data;
        params.out_gmem = y_data;
        params.res_gmem = res_data;

        // batchnorm specific parameters
        params.bn_sum_gmem = bn_arrays.bn_sum_gmem;
        params.bn_mean_gmem = bn_arrays.bn_mean_gmem;
        params.bn_sum_of_squares_gmem = bn_arrays.bn_sum_of_squares_gmem;
        params.bn_ilw_stddev_gmem = bn_arrays.bn_ilw_stddev_gmem;

        params.bn_partial_sums_gmem = bn_arrays.bn_partial_sums_gmem;
        params.bn_scale_gmem = bn_arrays.bn_scale_gmem;
        params.bn_bias_gmem = bn_arrays.bn_bias_gmem;
        params.bn_out_gmem = bn_arrays.bn_out_gmem;

        // Used in dgrad fused case
        params.bna_fprop_tensor_gmem = bn_arrays.bna_fprop_tensor_gmem;
        params.bna_bias_gmem = bn_arrays.bna_bias_gmem;
        params.bna_fprop_tensor_scale_gmem = bn_arrays.bna_fprop_tensor_scale_gmem;
        params.bna_grad_scale_gmem = bn_arrays.bna_grad_scale_gmem;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits,
          xmma::Operation_type colw_op,
          xmma::Colwolution_algorithm colw_algo,
          xmma::Colwolution_layout colw_layout,
          bool use_warp_specialization>
struct Device_kernel {};

template <typename Traits,
          xmma::Colwolution_algorithm colw_algo,
          xmma::Colwolution_layout colw_layout>
struct Device_kernel<Traits, xmma::Operation_type::WGRAD, colw_algo, colw_layout, false> {

    static xmma::Error run( xmma::Host_workspace<Traits> *workspace, lwdaStream_t &lwda_stream ) {

        if( workspace->smem_size > 48 * 1024 ) {
            if( workspace->xmma_params.ampere && workspace->smem_size > 164 * 1024 ||
                !workspace->xmma_params.ampere && workspace->smem_size > 64 * 1024 ) {
                return xmma::Error::ERROR_LWDA_RUNTIME;
            }

            XMMA_LWDA_CALL( lwdaFuncSetAttribute( xmma::gemm::kernel<Traits>,
                                                  lwdaFuncAttributeMaxDynamicSharedMemorySize,
                                                  workspace->smem_size ) );
            XMMA_LWDA_CALL( lwdaFuncSetAttribute(
                xmma::gemm::kernel<Traits>, lwdaFuncAttributePreferredSharedMemoryCarveout, 100 ) );
        }

        xmma::gemm::kernel<Traits><<<workspace->grid,
                                     Traits::Cta_tile::THREADS_PER_CTA,
                                     workspace->smem_size,
                                     lwda_stream>>>( workspace->xmma_params );
        XMMA_LWDA_CALL( lwdaGetLastError() );

        // If we need two kernels to run split-k launch the second grid.
        if( workspace->xmma_params.split_k.kernels == 2 ) {
            workspace->xmma_params.split_k.kernels = 1;

            dim3 split_k_grid = workspace->grid;
            split_k_grid.z = Traits::Xmma_tile::XMMAS_M;
            xmma::gemm::split_k_kernel<Traits><<<split_k_grid,
                                                 Traits::Cta_tile::THREADS_PER_CTA,
                                                 workspace->epilogue_size_in_bytes,
                                                 lwda_stream>>>( workspace->xmma_params );
            workspace->xmma_params.split_k.kernels = 2;
        }
        return xmma::Error::SUCCESS;
    }

    static lwdaError_t get_func_attributes( lwdaFuncAttributes *attr ) {
        lwdaError_t lwda_status = lwdaFuncGetAttributes( attr, xmma::gemm::kernel<Traits> );
        attr->maxDynamicSharedSizeBytes = Traits::dynamic_smem_size_per_cta();
        attr->maxThreadsPerBlock = Traits::threads_per_cta();
        return lwda_status;
    }
};

template <typename Traits,
          xmma::Colwolution_algorithm colw_algo,
          xmma::Colwolution_layout colw_layout>
struct Device_kernel<Traits, xmma::Operation_type::FPROP, colw_algo, colw_layout, false> {

    static xmma::Error run( xmma::Host_workspace<Traits> *workspace, lwdaStream_t &lwda_stream ) {

        //
        //  Call the fprop kernel
        //
        if( workspace->smem_size > 48 * 1024 ) {
            if( workspace->xmma_params.ampere && workspace->smem_size > 164 * 1024 ||
                !workspace->xmma_params.ampere && workspace->smem_size > 64 * 1024 ) {
                return xmma::Error::ERROR_LWDA_RUNTIME;
            }

            XMMA_LWDA_CALL( lwdaFuncSetAttribute( xmma::gemm::kernel<Traits>,
                                                  lwdaFuncAttributeMaxDynamicSharedMemorySize,
                                                  workspace->smem_size ) );
            XMMA_LWDA_CALL( lwdaFuncSetAttribute(
                xmma::gemm::kernel<Traits>, lwdaFuncAttributePreferredSharedMemoryCarveout, 100 ) );
        }

        xmma::gemm::kernel<Traits><<<workspace->grid,
                                     Traits::Cta_tile::THREADS_PER_CTA,
                                     workspace->smem_size,
                                     lwda_stream>>>( workspace->xmma_params );
        XMMA_LWDA_CALL( lwdaGetLastError() );

        // If we need two kernels to run split-k launch the second grid.
        if( workspace->xmma_params.split_k.kernels == 2 ) {
            workspace->xmma_params.split_k.kernels = 1;

            dim3 split_k_grid = workspace->grid;
            split_k_grid.z = Traits::Xmma_tile::XMMAS_M;
            xmma::gemm::split_k_kernel<Traits><<<split_k_grid,
                                                 Traits::Cta_tile::THREADS_PER_CTA,
                                                 workspace->epilogue_size_in_bytes,
                                                 lwda_stream>>>( workspace->xmma_params );
            workspace->xmma_params.split_k.kernels = 2;
        }

        if( !workspace->xmma_params.bn_disable_stats_output ) {
            const int DIM_Y = 32;
            const int threadblock_size =
                xmma::div_up( workspace->xmma_params.g * workspace->xmma_params.k, 32 );
            bn_stats::bn_stats_kernel<DIM_Y>
                <<<threadblock_size, dim3( 32, DIM_Y ), 0, lwda_stream>>>( workspace->xmma_params );

            XMMA_LWDA_CALL( lwdaGetLastError() );
        }
        return xmma::Error::SUCCESS;
    }

    static lwdaError_t get_func_attributes( lwdaFuncAttributes *attr ) {
        lwdaError_t lwda_status = lwdaFuncGetAttributes( attr, xmma::gemm::kernel<Traits> );
        attr->maxDynamicSharedSizeBytes = Traits::dynamic_smem_size_per_cta();
        attr->maxThreadsPerBlock = Traits::threads_per_cta();
        return lwda_status;
    }
};

template <typename Traits,
          xmma::Colwolution_algorithm colw_algo,
          xmma::Colwolution_layout colw_layout>
struct Device_kernel<Traits, xmma::Operation_type::DGRAD, colw_algo, colw_layout, false> {

    static xmma::Error run( xmma::Host_workspace<Traits> *workspace, lwdaStream_t &lwda_stream ) {

        //
        //  Call the fprop kernel
        //
        if( workspace->smem_size > 48 * 1024 ) {
            if( workspace->xmma_params.ampere && workspace->smem_size > 164 * 1024 ||
                !workspace->xmma_params.ampere && workspace->smem_size > 64 * 1024 ) {
                return xmma::Error::ERROR_LWDA_RUNTIME;
            }

            XMMA_LWDA_CALL( lwdaFuncSetAttribute( xmma::gemm::kernel<Traits>,
                                                  lwdaFuncAttributeMaxDynamicSharedMemorySize,
                                                  workspace->smem_size ) );
            XMMA_LWDA_CALL( lwdaFuncSetAttribute(
                xmma::gemm::kernel<Traits>, lwdaFuncAttributePreferredSharedMemoryCarveout, 100 ) );
        }

        xmma::gemm::kernel<Traits><<<workspace->grid,
                                     Traits::Cta_tile::THREADS_PER_CTA,
                                     workspace->smem_size,
                                     lwda_stream>>>( workspace->xmma_params );
        XMMA_LWDA_CALL( lwdaGetLastError() );

        // If we need two kernels to run split-k launch the second grid.
        if( workspace->xmma_params.split_k.kernels == 2 ) {
            workspace->xmma_params.split_k.kernels = 1;

            dim3 split_k_grid = workspace->grid;
            split_k_grid.z = Traits::Xmma_tile::XMMAS_M;
            xmma::gemm::split_k_kernel<Traits><<<split_k_grid,
                                                 Traits::Cta_tile::THREADS_PER_CTA,
                                                 workspace->epilogue_size_in_bytes,
                                                 lwda_stream>>>( workspace->xmma_params );
            workspace->xmma_params.split_k.kernels = 2;
        }

        if( !workspace->xmma_params.bn_disable_stats_output ) {
            const int DIM_Y = 32;
            const int threadblock_size =
                xmma::div_up( workspace->xmma_params.g * workspace->xmma_params.c, 32 );
            if( workspace->xmma_params.dual_dbns ) {
                bn_stats::dual_dbn_stats_kernel<DIM_Y>
                    <<<threadblock_size, dim3( 32, DIM_Y ), 0, lwda_stream>>>(
                        workspace->xmma_params );
            } else {
                bn_stats::dbn_stats_kernel<DIM_Y>
                    <<<threadblock_size, dim3( 32, DIM_Y ), 0, lwda_stream>>>(
                        workspace->xmma_params );
            }

            XMMA_LWDA_CALL( lwdaGetLastError() );
        }
        if( workspace->xmma_params.standalone_dbna ) {
            // gridY is along C and gridX is along NDHW
            uint32_t C_ELEMENTS_PER_CTA =
                max( 8, min( 64, workspace->xmma_params.g * workspace->xmma_params.c ) );
            const int threadblock_size_x =
                min( 108,
                     workspace->xmma_params.n * workspace->xmma_params.d *
                         workspace->xmma_params.h * workspace->xmma_params.w );
            const int threadblock_size_y = xmma::div_up(
                workspace->xmma_params.g * workspace->xmma_params.c, C_ELEMENTS_PER_CTA );
            if( workspace->xmma_params.dual_dbns ) {
                bn_stats::dual_dbn_apply_kernel<512>
                    <<<dim3( threadblock_size_x, threadblock_size_y ),
                       dim3( 512, 1 ),
                       0,
                       lwda_stream>>>( workspace->xmma_params, C_ELEMENTS_PER_CTA );
            } else {
                bn_stats::dbn_apply_kernel<512>
                    <<<dim3( threadblock_size_x, threadblock_size_y ),
                       dim3( 512, 1 ),
                       0,
                       lwda_stream>>>( workspace->xmma_params, C_ELEMENTS_PER_CTA );
            }
            XMMA_LWDA_CALL( lwdaGetLastError() );
        }

        return xmma::Error::SUCCESS;
    }

    static lwdaError_t get_func_attributes( lwdaFuncAttributes *attr ) {
        lwdaError_t lwda_status = lwdaFuncGetAttributes( attr, xmma::gemm::kernel<Traits> );
        attr->maxDynamicSharedSizeBytes = Traits::dynamic_smem_size_per_cta();
        attr->maxThreadsPerBlock = Traits::threads_per_cta();
        return lwda_status;
    }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits>
int initialize_xmma_params( typename Traits::Params &xmma_params,
                            xmma::Host_workspace<Traits> *workspace ) {
    // Need to do this to ensure correct swizzling is chosen
    // Set ctas per wave.
    lwdaDeviceProp prop;
    lwdaGetDeviceProperties( &prop, 0 );
    lwdaFuncAttributes attr;
    get_func_attributes<Traits>( &attr );
    xmma_params.ctas_per_wave = xmma::get_ctas_per_wave( &attr,
                                                         prop.multiProcessorCount,
                                                         prop.sharedMemPerMultiprocessor,
                                                         prop.regsPerMultiprocessor );

    return xmma_params.initialize( workspace );
}

///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits> size_t get_host_workspace_size() {
    return sizeof( xmma::Host_workspace<Traits> );
}

///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits>
xmma::Error initialize_host_workspace( typename Traits::Params &xmma_params, void *host_ptr ) {
    xmma::Host_workspace<Traits> *workspace =
        static_cast<xmma::Host_workspace<Traits> *>( host_ptr );

    // return 1 for success and 0 for invalid argument
    return initialize_xmma_params<Traits>( xmma_params, workspace );
}

///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits> size_t get_device_workspace_size( const void *host_ptr ) {
    const xmma::Host_workspace<Traits> *workspace =
        static_cast<const xmma::Host_workspace<Traits> *>( host_ptr );

    size_t device_workspace_size = workspace->device_workspace_size;
    // Additional 16 bytes for alignment
    if( device_workspace_size != 0 ) {
        device_workspace_size += 16;
    }
    return device_workspace_size;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits>
xmma::Error
initialize_device_workspace( const void *host_ptr, void *device_ptr, lwdaStream_t lwda_stream ) {
    return xmma::Error::SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits> size_t get_partial_sums_sz( const void *host_ptr ) {
    const xmma::Host_workspace<Traits> *workspace =
        static_cast<const xmma::Host_workspace<Traits> *>( host_ptr );

    return workspace->xmma_params.bn_partial_sums_sz;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits, typename X_type, typename Y_type, typename W_type>
xmma::Error run_kernel( X_type *x_data,
                        Y_type *y_data,
                        W_type *w_data,
                        const void *res_data,
                        const Bn_Device_Arrays &bn_arrays,
                        const void *bias_data,
                        const void *alpha_data,
                        const void *beta_data,
                        void *host_ptr,
                        void *device_ptr,
                        xmma::implicit_gemm::Runtime_params<Traits> &runtime_params,
                        lwdaStream_t &lwda_stream ) {
    xmma::Host_workspace<Traits> *host_workspace =
        static_cast<xmma::Host_workspace<Traits> *>( host_ptr );

    Device_pointers<Traits, Traits::OPERATION_TYPE>::init(
        host_workspace->xmma_params, x_data, y_data, w_data, res_data, bn_arrays );

    // Fix incorrect sum of squares parameter, since malloc happened just before launch
    host_workspace->xmma_params.set_partial_sums_of_squares_ptr( host_workspace );

    const size_t alignment = 16 - ( (size_t)device_ptr ) % 16;

    host_workspace->xmma_params.split_k.buffers_gmem =
        (void *)( xmma::ptr_to_int64( device_ptr ) + alignment );
    host_workspace->xmma_params.split_k.counters_gmem =
        (int32_t *)( xmma::ptr_to_int64( host_workspace->xmma_params.split_k.buffers_gmem ) +
                     host_workspace->xmma_params.split_k.buffer_size *
                         host_workspace->xmma_params.split_k.buffers );
    host_workspace->xmma_params.split_k.retired_ctas_gmem =
        &( host_workspace->xmma_params.split_k
               .counters_gmem )[host_workspace->xmma_params.split_k.counters_ctas_size /
                                sizeof( int32_t )];

    // Clear the buffer of counters for split-k (if needed).
    if( host_workspace->split_k_with_reduction ) {
        XMMA_LWDA_CALL( lwdaMemsetAsync( host_workspace->xmma_params.split_k.counters_gmem,
                                         0,
                                         host_workspace->xmma_params.split_k.counters_ctas_size +
                                             host_workspace->xmma_params.split_k.retired_ctas_size,
                                         lwda_stream ) );
    }

    host_workspace->xmma_params.bias_gmem = bias_data;
    host_workspace->xmma_params.alpha_gmem = alpha_data;
    host_workspace->xmma_params.beta_gmem = beta_data;

    // Initialize the L2 descriptors
    host_workspace->xmma_params.mem_descriptors.descriptor_a =
        ( (uint64_t)runtime_params.descriptor_a << 32 );
    host_workspace->xmma_params.mem_descriptors.descriptor_b =
        ( (uint64_t)runtime_params.descriptor_b << 32 );
    host_workspace->xmma_params.mem_descriptors.descriptor_c =
        ( (uint64_t)runtime_params.descriptor_c1 << 32 ) + (uint64_t)runtime_params.descriptor_c0;
    host_workspace->xmma_params.mem_descriptors.descriptor_d =
        ( (uint64_t)runtime_params.descriptor_d1 << 32 ) + (uint64_t)runtime_params.descriptor_d0;

    if( Device_kernel<Traits,
                      Traits::OPERATION_TYPE,
                      Traits::COLW_ALGO,
                      Traits::COLW_LAYOUT,
                      false>::run( host_workspace, lwda_stream ) != xmma::Error::SUCCESS ) {
        return xmma::Error::ERROR_LWDA_RUNTIME;
    }

    XMMA_LWDA_CALL( lwdaGetLastError() );

    return xmma::Error::SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Implicit_gemm_traits>
xmma::Error get_func_attributes( lwdaFuncAttributes *attr ) {
    if( Device_kernel<Implicit_gemm_traits,
                      Implicit_gemm_traits::OPERATION_TYPE,
                      Implicit_gemm_traits::COLW_ALGO,
                      Implicit_gemm_traits::COLW_LAYOUT,
                      Implicit_gemm_traits::USE_WARP_SPECIALIZATION>::get_func_attributes( attr ) !=
        lwdaSuccess ) {
        return xmma::Error::ERROR_LWDA_RUNTIME;
    }
    return xmma::Error::SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits> static void print_xmma_params( const typename Traits::Params &params ) {
    printf( "g=%d n=%d d=%d h=%d, w=%d, c=%d, k=%d, t=%d r=%d, s=%d \
            (o=%d, p=%d, q=%d, ampere=%d), alpha = %0.1f, beta = %0.1f, pad = %d, %d, %d \
            stride = %d, %d, %d, dilation = %d, %d, %d\n",
            params.g,
            params.n,
            params.d,
            params.h,
            params.w,
            params.c,
            params.k,
            params.t,
            params.r,
            params.s,
            params.o,
            params.p,
            params.q,
            params.ampere,
            xmma::colwert<float>( params.alpha ),
            xmma::colwert<float>( params.beta ),
            params.pad[0][0],
            params.pad[1][0],
            params.pad[2][0],
            params.stride[0],
            params.stride[1],
            params.stride[2],
            params.dilation[0],
            params.dilation[1],
            params.dilation[2] );
}

///////////////////////////////////////////////////////////////////////////////////////////////////
}  // namespace batchnorm
}  // namespace ext
}  // namespace xmma
///////////////////////////////////////////////////////////////////////////////////////////////////
