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
#include <xmma/params.h>
#include <xmma/ext/implicit_gemm/indexed_wo_smem/kernel.h>

namespace xmma {
namespace ext {
namespace implicit_gemm {

///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Implicit_gemm_traits> struct Runtime_params {
    int32_t descriptor_a;
    int32_t descriptor_b;
    int32_t descriptor_c0;
    int32_t descriptor_c1;
    int32_t descriptor_d0;
    int32_t descriptor_d1;

    // Tensor batchsize.
    int32_t batch_size;
};

//////////////////////////////// Initialize device pointer ////////////////////////////////////////

template <typename Implicit_gemm_traits, xmma::Operation_type operation_type>
struct Device_pointers {

    static void init( typename Implicit_gemm_traits::Params& params, const void* x_data,
                      const void* y_data, void* w_data, const void* res_data ) {
        params.img_gmem = x_data;
        params.flt_gmem = y_data;
        params.out_gmem = w_data;
        params.res_gmem = res_data;
    }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Implicit_gemm_traits>
struct Device_pointers<Implicit_gemm_traits, xmma::Operation_type::WGRAD> {

    static void init( typename Implicit_gemm_traits::Params& params, const void* x_data,
                      void* y_data, const void* w_data, const void* res_data ) {
        params.img_gmem = x_data;
        params.flt_gmem = w_data;
        params.out_gmem = y_data;
        params.res_gmem = res_data;
    }
};

//////////////////////////////// Select device kernel to run //////////////////////////////////////

template <typename Implicit_gemm_traits, xmma::Operation_type operation_type,
          xmma::Colwolution_algorithm colw_algo, xmma::Colwolution_layout colw_layout>
struct Device_kernel {

    static xmma::Error run( xmma::Host_workspace<Implicit_gemm_traits>* workspace,
                                lwdaStream_t& lwda_stream ) {
        if( workspace->smem_size > 48 * 1024 ) {
            if( workspace->xmma_params.ampere && workspace->smem_size > 164 * 1024 ||
                !workspace->xmma_params.ampere && workspace->smem_size > 64 * 1024 ) {
                return xmma::Error::ERROR_LWDA_RUNTIME;
            }
            XMMA_LWDA_CALL( lwdaFuncSetAttribute( xmma::ext::implicit_gemm::kernel<Implicit_gemm_traits>,
                                              lwdaFuncAttributeMaxDynamicSharedMemorySize,
                                              workspace->smem_size ) );
            XMMA_LWDA_CALL( lwdaFuncSetAttribute( xmma::ext::implicit_gemm::kernel<Implicit_gemm_traits>,
                                              lwdaFuncAttributePreferredSharedMemoryCarveout,
                                              100 ) );
        }

        xmma::ext::implicit_gemm::kernel<Implicit_gemm_traits><<<
            workspace->grid, Implicit_gemm_traits::Cta_tile::THREADS_PER_CTA, workspace->smem_size,
            lwda_stream>>>( workspace->xmma_params );
        XMMA_LWDA_CALL( lwdaGetLastError() );

        // If we need two kernels to run split-k launch the second grid.
        if( workspace->xmma_params.split_k.kernels == 2 ) {
            workspace->xmma_params.split_k.kernels = 1;
            workspace->split_k_grid = workspace->grid;
            workspace->split_k_grid.z = Implicit_gemm_traits::Xmma_tile::XMMAS_M;
            xmma::ext::implicit_gemm::split_k_kernel<Implicit_gemm_traits><<<
                workspace->split_k_grid, Implicit_gemm_traits::Cta_tile::THREADS_PER_CTA,
                0, lwda_stream>>>( workspace->xmma_params );
            workspace->xmma_params.split_k.kernels = 2;
        }
        XMMA_LWDA_CALL( lwdaGetLastError() );

        return xmma::Error::SUCCESS;
    }

    static lwdaError_t get_func_attributes( lwdaFuncAttributes* attr ) {
        lwdaError_t lwda_status =
            lwdaFuncGetAttributes( attr, xmma::ext::implicit_gemm::kernel<Implicit_gemm_traits> );
        attr->maxDynamicSharedSizeBytes =
            Implicit_gemm_traits::dynamic_smem_size_per_cta();
        attr->maxThreadsPerBlock = Implicit_gemm_traits::threads_per_cta();
        return lwda_status;
    }
};

template <typename Implicit_gemm_traits>
xmma::Error get_func_attributes(lwdaFuncAttributes* attr) {
    if( Device_kernel<Implicit_gemm_traits, Implicit_gemm_traits::OPERATION_TYPE,
                      Implicit_gemm_traits::COLW_ALGO,
                      Implicit_gemm_traits::COLW_LAYOUT>
        ::get_func_attributes(attr) != lwdaSuccess ) {
        return xmma::Error::ERROR_LWDA_RUNTIME;
    }
    return xmma::Error::SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Implicit_gemm_traits> size_t get_host_workspace_size() {
    return sizeof( xmma::Host_workspace<Implicit_gemm_traits> );
}

///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Implicit_gemm_traits>
xmma::Error initialize_host_workspace( typename Implicit_gemm_traits::Params& xmma_params,
                                           void* host_ptr ) {
    xmma::Host_workspace<Implicit_gemm_traits>* workspace =
        static_cast<xmma::Host_workspace<Implicit_gemm_traits>*>( host_ptr );

    return xmma_params.initialize( workspace );
}

///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Implicit_gemm_traits> size_t get_device_workspace_size( const void* host_ptr ) {
    const xmma::Host_workspace<Implicit_gemm_traits>* workspace =
        static_cast<const xmma::Host_workspace<Implicit_gemm_traits>*>( host_ptr );

    size_t device_workspace_size = workspace->device_workspace_size;
    //Additional 16 bytes for alignment
    if (device_workspace_size != 0) {
        device_workspace_size += 16;
    }
    return device_workspace_size;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Implicit_gemm_traits>
xmma::Error initialize_device_workspace( const void* host_ptr, void* device_ptr,
                                             lwdaStream_t lwda_stream ) {
    return xmma::Error::SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Implicit_gemm_traits, typename X_type, typename Y_type, typename W_type>
xmma::Error run_kernel(
    X_type* x_data, Y_type* y_data, W_type* w_data,
    const void* res_data,  const void* bias_data,
    const void* alpha_data,
    const void* beta_data, void* host_ptr, void* device_ptr,
    Runtime_params<Implicit_gemm_traits>& runtime_params,
    lwdaStream_t& lwda_stream
) {
    auto host_workspace = static_cast<xmma::Host_workspace<Implicit_gemm_traits> *>(host_ptr);
    auto &params = host_workspace->xmma_params;

    Device_pointers<Implicit_gemm_traits, Implicit_gemm_traits::OPERATION_TYPE>::init(
        params, x_data, y_data, w_data, res_data );

    params.bias_gmem = bias_data;
    params.alpha_gmem = alpha_data;
    params.beta_gmem = beta_data;

    // Initialize the L2 descriptors
    params.mem_descriptors.descriptor_a = ( (uint64_t)runtime_params.descriptor_a << 32 );
    params.mem_descriptors.descriptor_b = ( (uint64_t)runtime_params.descriptor_b << 32 );
    params.mem_descriptors.descriptor_c = ( (uint64_t)runtime_params.descriptor_c1 << 32 ) + (uint64_t)runtime_params.descriptor_c0;
    params.mem_descriptors.descriptor_d = ( (uint64_t)runtime_params.descriptor_d1 << 32 ) + (uint64_t)runtime_params.descriptor_d0;

    // Setup & clear(if needed) split-k buffers
    params.split_k.set_base_ptr(device_ptr);
    XMMA_LWDA_CALL(params.split_k.clear_buffers(device_ptr, lwda_stream));

    // Update batchSize.
    if( runtime_params.batch_size > 0 && runtime_params.batch_size != params.n ) {
        params.n = runtime_params.batch_size;

        // Re-initialize xmma_params with new batchSize.
        params.initialize( host_workspace );
    }

    if( Device_kernel<Implicit_gemm_traits, Implicit_gemm_traits::OPERATION_TYPE,
                      Implicit_gemm_traits::COLW_ALGO,
                      Implicit_gemm_traits::COLW_LAYOUT>::run( host_workspace, lwda_stream ) !=
        xmma::Error::SUCCESS ) {
        return xmma::Error::ERROR_LWDA_RUNTIME;
    }

    return xmma::Error::SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace implicit_gemm
}  // namespace ext
} // namespace xmma
