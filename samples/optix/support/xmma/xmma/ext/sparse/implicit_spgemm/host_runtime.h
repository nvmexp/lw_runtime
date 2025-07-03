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
#include <xmma/ext/sparse/implicit_spgemm/kernel.h>

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
    // gelu runtime scale factor
    float gelu_scale;
};

//////////////////////////////// Initialize device pointer ////////////////////////////////////////

template <typename Implicit_gemm_traits, xmma::Operation_type operation_type>
struct Device_pointers {

    static void init( typename Implicit_gemm_traits::Params& params, const void* x_data,
                      const void* y_data, void* w_data, const void* res_data,
                      const void* e_data ) {
        params.img_gmem = x_data;
        params.flt_gmem = y_data;
        params.out_gmem = w_data;
        params.res_gmem = res_data;
        params.e_gmem = e_data;
    }
};

//////////////////////////////// Select device kernel to run //////////////////////////////////////

template <typename Implicit_gemm_traits, xmma::Operation_type operation_type,
          xmma::Colwolution_algorithm colw_algo, xmma::Colwolution_layout colw_layout>
struct Device_kernel {

    static xmma::Error run( xmma::Host_workspace<Implicit_gemm_traits>* workspace,
                                lwdaStream_t& lwda_stream ) {

        if (workspace->xmma_params.with_residual) {
            if( workspace->smem_size > 48 * 1024 ) {
                if( workspace->xmma_params.ampere && workspace->smem_size > 164 * 1024 ||
                    !workspace->xmma_params.ampere && workspace->smem_size > 64 * 1024 ) {
                    return xmma::Error::ERROR_LWDA_RUNTIME;
                }
                XMMA_LWDA_CALL( lwdaFuncSetAttribute(
                    xmma::ext::implicit_gemm::kernel<Implicit_gemm_traits, true>,
                    lwdaFuncAttributeMaxDynamicSharedMemorySize,
                    workspace->smem_size ) );
                XMMA_LWDA_CALL( lwdaFuncSetAttribute(
                    xmma::ext::implicit_gemm::kernel<Implicit_gemm_traits, true>,
                    lwdaFuncAttributePreferredSharedMemoryCarveout,
                    100 ) );
            }

            xmma::ext::implicit_gemm::kernel<Implicit_gemm_traits, true><<<
                workspace->grid, Implicit_gemm_traits::Cta_tile::THREADS_PER_CTA,
                workspace->smem_size,
                lwda_stream>>>( workspace->xmma_params );
        } else {
            if( workspace->smem_size > 48 * 1024 ) {
                if( workspace->xmma_params.ampere && workspace->smem_size > 164 * 1024 ||
                    !workspace->xmma_params.ampere && workspace->smem_size > 64 * 1024 ) {
                    return xmma::Error::ERROR_LWDA_RUNTIME;
                }
                XMMA_LWDA_CALL( lwdaFuncSetAttribute(
                    xmma::ext::implicit_gemm::kernel<Implicit_gemm_traits, false>,
                    lwdaFuncAttributeMaxDynamicSharedMemorySize,
                    workspace->smem_size ) );
                XMMA_LWDA_CALL( lwdaFuncSetAttribute(
                xmma::ext::implicit_gemm::kernel<Implicit_gemm_traits, false>,
                lwdaFuncAttributePreferredSharedMemoryCarveout,
                100 ) );
            }
            xmma::ext::implicit_gemm::kernel<Implicit_gemm_traits, false><<<
                workspace->grid, Implicit_gemm_traits::Cta_tile::THREADS_PER_CTA,
                workspace->smem_size,
                lwda_stream>>>( workspace->xmma_params );
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

///////////////////////////////////////////////////////////////////////////////////////////////////

// Used for IMMA reorder kernel
struct Reorder_imma_filter_params {
    // The size of the filter.
    int k;
    // Fitler CRS.
    int crs;
    // The size of  the xformed filter.
    int xform_filter_k;
    // The input filter.
    const char* filter_gmem;
    // The transformed output filter.
    char* xform_filter_gmem;
};

///////////////////////////////////////////////////////////////////////////////////////////////////

// Lwca kernel for filter reorder
static __global__ void lwda_reorder_imma_filter( Reorder_imma_filter_params params ) {
    // From gemm.cpp, that's the mapping from the column k in the original filter
    // to the column nCdiv32Hw32ReorderCol[k] is the transformed filter.
    const int BYTES_PER_PACKETS = 32;
    // The C*R*S position in both filters.
    int crs = blockIdx.y;
    // The k position in the transformed filter.
    int xform_k = blockIdx.x * blockDim.x + threadIdx.x;
    // The position in the original filter.
    int k = ( xform_k & ~31 ) + ( ( xform_k & ( 3 << 1 ) ) << 2 ) +
            ( ( xform_k & ( 3 << 3 ) ) >> 2 ) + ( xform_k & 1 );

    // The offset in the xformed filter.
    int dst_offset = ( crs * params.xform_filter_k + xform_k ) * BYTES_PER_PACKETS;

    // Read the data.
    int data0[4] = { 0 };
    int data1[4] = { 0 };
    int src_offset = ( k * params.crs + crs ) * BYTES_PER_PACKETS;

    const int* read_ptr = reinterpret_cast<const int*>( &params.filter_gmem[src_offset] );
    if( k < params.k ) {
        for( int i = 0; i < 4; i++ ) {
            data0[i] = read_ptr[i];
            data1[i] = read_ptr[i + 4];
        }
    }

    // Write the data back.
    int* write_ptr = reinterpret_cast<int*>( &params.xform_filter_gmem[dst_offset] );
    if( xform_k < params.xform_filter_k ) {
        for( int i = 0; i < 4; i++ ) {
            write_ptr[i] = data0[i];
            write_ptr[i + 4] = data1[i];
        }
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Implicit_gemm_traits>
xmma::Error reorder_filter( const void* filter_data, void* host_ptr, void* device_ptr,
                                lwdaStream_t& lwda_stream ) {
    xmma::Host_workspace<Implicit_gemm_traits>* host_workspace =
        static_cast<xmma::Host_workspace<Implicit_gemm_traits>*>( host_ptr );

    // Initializing params
    Reorder_imma_filter_params reorder_params;
    reorder_params.k = host_workspace->xmma_params.k;
    reorder_params.crs = host_workspace->xmma_params.c / 32 * host_workspace->xmma_params.t *
                         host_workspace->xmma_params.r * host_workspace->xmma_params.s;
    reorder_params.xform_filter_k = host_workspace->xmma_params.k;
    reorder_params.filter_gmem = reinterpret_cast<const char*>( filter_data );
    reorder_params.xform_filter_gmem = reinterpret_cast<char*>( device_ptr );

    int num_threads = 128;
    dim3 grid( xmma::div_up( reorder_params.k, num_threads ), reorder_params.crs );

    lwda_reorder_imma_filter<<<grid, num_threads, 0, lwda_stream>>>( reorder_params );
    XMMA_LWDA_CALL( lwdaGetLastError() );

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

template <typename Implicit_gemm_traits>
static void print_xmma_params( const typename Implicit_gemm_traits::Params& params ) {
    printf( "g=%d n=%d d=%d h=%d, w=%d, c=%d, k=%d, t=%d r=%d, s=%d \
          (o=%d, p=%d, q=%d, ampere=%d), alpha = %0.1f, beta = %0.1f, pad = %d, %d, %d \
	  stride = %d, %d, %d, dilation = %d, %d, %d\n",
            params.g, params.n, params.d, params.h, params.w, params.c, params.k, params.t,
            params.r, params.s, params.o, params.p, params.q, params.ampere,
            xmma::colwert<float>( params.alpha ), xmma::colwert<float>( params.beta ),
            params.pad[0][0], params.pad[1][0], params.pad[2][0], params.stride[0],
            params.stride[1], params.stride[2], params.dilation[0], params.dilation[1],
            params.dilation[2] );
}

///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Implicit_gemm_traits, typename X_type, typename Y_type, typename W_type>
xmma::Error run_kernel(
    X_type* x_data, Y_type* y_data, W_type* w_data,
    const void* e_data,const void* res_data,  const void* bias_data,
    const void* alpha_data,
    const void* beta_data, void* host_ptr, void* device_ptr,
    Runtime_params<Implicit_gemm_traits>& runtime_params,
    lwdaStream_t& lwda_stream
) {
    auto host_workspace = static_cast<xmma::Host_workspace<Implicit_gemm_traits> *>(host_ptr);
    auto &params = host_workspace->xmma_params;

    Device_pointers<Implicit_gemm_traits, Implicit_gemm_traits::OPERATION_TYPE>::init(
        params, x_data, y_data, w_data, res_data, e_data );

    params.bias_gmem = bias_data;
    params.alpha_gmem = alpha_data;
    params.beta_gmem = beta_data;

    // Initialize the L2 descriptors
    params.mem_descriptors.descriptor_a = ( (uint64_t)runtime_params.descriptor_a << 32 );
    params.mem_descriptors.descriptor_b = ( (uint64_t)runtime_params.descriptor_b << 32 );
    params.mem_descriptors.descriptor_c = ( (uint64_t)runtime_params.descriptor_c1 << 32 ) + (uint64_t)runtime_params.descriptor_c0;
    params.mem_descriptors.descriptor_d = ( (uint64_t)runtime_params.descriptor_d1 << 32 ) + (uint64_t)runtime_params.descriptor_d0;
    float *geluScale = &runtime_params.gelu_scale;
    params.runtime_params.runtime_param0 = *reinterpret_cast<int32_t*>(geluScale);


    // Update batchSize.
    if( runtime_params.batch_size > 0 && runtime_params.batch_size != params.n ) {
        // FIXME : Should use int64_t for runtime_params.batch_size?
        params.n = runtime_params.batch_size;

        // Re-initialize xmma_params with new batchSize.
        params.initialize( host_workspace );
    }

    // Setup & clear(if needed) split-k buffers
    params.split_k.set_base_ptr(device_ptr);
    XMMA_LWDA_CALL(params.split_k.clear_buffers(device_ptr, lwda_stream));

    if( Device_kernel<Implicit_gemm_traits, Implicit_gemm_traits::OPERATION_TYPE,
                      Implicit_gemm_traits::COLW_ALGO,
                      Implicit_gemm_traits::COLW_LAYOUT>::run( host_workspace, lwda_stream ) !=
        xmma::Error::SUCCESS ) {
        return xmma::Error::ERROR_LWDA_RUNTIME;
    }

    return xmma::Error::SUCCESS;
}

template <typename Implicit_gemm_traits>
xmma::Error get_func_attributes( lwdaFuncAttributes* attr ) {
    if( Device_kernel<Implicit_gemm_traits,
                      Implicit_gemm_traits::OPERATION_TYPE,
                      Implicit_gemm_traits::COLW_ALGO,
                      Implicit_gemm_traits::COLW_LAYOUT>::get_func_attributes( attr ) !=
        lwdaSuccess ) {
        return xmma::Error::ERROR_LWDA_RUNTIME;
    }
    return xmma::Error::SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace implicit_gemm
}  // namespace ext
} // namespace xmma
