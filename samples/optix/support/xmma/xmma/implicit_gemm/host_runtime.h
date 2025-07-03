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

#include <xmma/gemm/kernel.h>
#include <xmma/gemm/kernel_hopper.h>

#include <xmma/implicit_gemm/strided_dgrad_indexed/kernel.h>
#include <xmma/implicit_gemm/interleaved_fprop/kernel.h>
#include <xmma/gemm/warp_specialized_kernel.h>
#include <xmma/implicit_gemm/strided_dgrad_indexed/warp_specialized_kernel.h>
#include <xmma/warp_specialized_traits.h>
#include <xmma/implicit_gemm/dgrad/utils.h>
#include <xmma/implicit_gemm/wgrad_indexed/utils.h>

#include "xmma/hopper/emu/lwda_tma_utils.h"
#include "xmma/hopper/emu/xmma_tma_helpers.h"

namespace xmma {
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

    static void init( typename Implicit_gemm_traits::Params& params,
                      const void* x_data,
                      const void* y_data,
                      void* w_data,
                      const void* res_data ) {
        params.img_gmem = x_data;
        params.flt_gmem = y_data;
        params.out_gmem = w_data;
        params.res_gmem = res_data;
    }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Implicit_gemm_traits>
struct Device_pointers<Implicit_gemm_traits, xmma::Operation_type::WGRAD> {

    static void init( typename Implicit_gemm_traits::Params& params,
                      const void* x_data,
                      void* y_data,
                      const void* w_data,
                      const void* res_data ) {
        params.img_gmem = x_data;
        params.flt_gmem = w_data;
        params.out_gmem = y_data;
        params.res_gmem = res_data;
    }
};

//////////////////////////////// Select device kernel to run //////////////////////////////////////

template <typename Implicit_gemm_traits,
          xmma::Operation_type operation_type,
          xmma::Colwolution_algorithm colw_algo,
          xmma::Colwolution_layout colw_layout,
          bool use_warp_specialization>
struct Device_kernel {

    static xmma::Error run( xmma::Host_workspace<Implicit_gemm_traits>* workspace,
                            lwdaStream_t& lwda_stream ) {
        if( workspace->smem_size > 48 * 1024 ) {
            if( workspace->xmma_params.ampere && workspace->smem_size > 164 * 1024 ||
                !workspace->xmma_params.ampere && workspace->smem_size > 64 * 1024 ) {
                printf( "[ ERROR: LWCA Runtime ] %s:%d: Invalid shared memory size!\n",
                        __FILE__,
                        __LINE__ );
                return xmma::Error::ERROR_LWDA_RUNTIME;
            }
            XMMA_LWDA_CALL( lwdaFuncSetAttribute( xmma::gemm::kernel<Implicit_gemm_traits>,
                                              lwdaFuncAttributeMaxDynamicSharedMemorySize,
                                              workspace->smem_size ) );
            XMMA_LWDA_CALL( lwdaFuncSetAttribute( xmma::gemm::kernel<Implicit_gemm_traits>,
                                              lwdaFuncAttributePreferredSharedMemoryCarveout,
                                              100 ) );
        }

        xmma::gemm::kernel<Implicit_gemm_traits><<<workspace->grid,
                                                   Implicit_gemm_traits::Cta_tile::THREADS_PER_CTA,
                                                   workspace->smem_size,
            lwda_stream>>>( workspace->xmma_params );
        XMMA_LWDA_CALL( lwdaGetLastError() );

        // If we need two kernels to run split-k launch the second grid.
        if( workspace->xmma_params.split_k.kernels == 2 ) {
            workspace->xmma_params.split_k.kernels = 1;
            workspace->split_k_grid = workspace->grid;
            workspace->split_k_grid.z = Implicit_gemm_traits::Gmem_tile_epilogue::Layout::ROW
                ? Implicit_gemm_traits::Xmma_tile::XMMAS_M
                : Implicit_gemm_traits::Xmma_tile::XMMAS_N;
            xmma::gemm::split_k_kernel<
                Implicit_gemm_traits><<<workspace->split_k_grid,
                                        Implicit_gemm_traits::Cta_tile::THREADS_PER_CTA,
                                        workspace->epilogue_size_in_bytes,
                                        lwda_stream>>>( workspace->xmma_params );
            workspace->xmma_params.split_k.kernels = 2;
        }
        XMMA_LWDA_CALL( lwdaGetLastError() );

        return xmma::Error::SUCCESS;
    }

    static lwdaError_t get_func_attributes( lwdaFuncAttributes* attr ) {
        lwdaError_t lwda_status =
            lwdaFuncGetAttributes( attr, xmma::gemm::kernel<Implicit_gemm_traits> );
        attr->maxDynamicSharedSizeBytes =
            Implicit_gemm_traits::dynamic_smem_size_per_cta();
        attr->maxThreadsPerBlock = Implicit_gemm_traits::threads_per_cta();
        return lwda_status;
    }
};
//////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Implicit_gemm_traits,
         xmma::Operation_type colw_op,
         xmma::Colwolution_algorithm colw_algo,
         xmma::Colwolution_layout colw_layout>
struct Device_kernel<Implicit_gemm_traits, colw_op, colw_algo, colw_layout, true> {

    static xmma::Error run( xmma::Host_workspace<Implicit_gemm_traits>* workspace,
                            lwdaStream_t& lwda_stream ) {
        void ( *xmma_implicit_gemm_warp_specialized_kernel )(typename Implicit_gemm_traits::Params params ) = nullptr;

        int warp_specialized_factor = 1;
        if( workspace->xmma_params.specialize == xmma::CONFIG_1DMA_1MATH ) {
            xmma_implicit_gemm_warp_specialized_kernel =
                xmma::gemm::xmma_implicit_gemm_specialize_1math_1dma_arrive_wait_kernel<Implicit_gemm_traits>;
            warp_specialized_factor = 2;

        } else if( workspace->xmma_params.specialize == xmma::CONFIG_1DMA_2MATH ) {
            xmma_implicit_gemm_warp_specialized_kernel =
                xmma::gemm::xmma_implicit_gemm_specialize_2math_1dma_arrive_wait_kernel<Implicit_gemm_traits>;
            warp_specialized_factor = 3;
        }

        if( workspace->smem_size > 48 * 1024 ) {
            if( workspace->xmma_params.ampere && workspace->smem_size > 164 * 1024 ||
                !workspace->xmma_params.ampere && workspace->smem_size > 64 * 1024 ) {
                printf( "[ ERROR: LWCA Runtime ] %s:%d: Invalid shared memory size %d!\n",
                        __FILE__, __LINE__, workspace->smem_size );
                return xmma::Error::ERROR_LWDA_RUNTIME;
            }
            XMMA_LWDA_CALL( lwdaFuncSetAttribute( xmma_implicit_gemm_warp_specialized_kernel,
                                                  lwdaFuncAttributeMaxDynamicSharedMemorySize,
                                                  workspace->smem_size ) );
            XMMA_LWDA_CALL( lwdaFuncSetAttribute( xmma_implicit_gemm_warp_specialized_kernel,
                                                  lwdaFuncAttributePreferredSharedMemoryCarveout, 100 ) );
        }

        // warp specialization: now THREADS_PER_CTA = 2 * Cta_tile::THREADS_PER_CTA
        // (the CTA size in non-specialized) : each CTA now is 2x original CTA size,
        // 1 for DMA warp group,  another one for running  MATH warp group.
        xmma_implicit_gemm_warp_specialized_kernel<<<
            workspace->grid,
            Implicit_gemm_traits::Cta_tile::THREADS_PER_CTA * warp_specialized_factor,
            workspace->smem_size,
                lwda_stream>>>( workspace->xmma_params );
            XMMA_LWDA_CALL( lwdaGetLastError() );

        // If we need two kernels to run split-k launch the second grid.
        if( workspace->xmma_params.split_k.kernels == 2 ) {
            workspace->xmma_params.split_k.kernels = 1;
                xmma::gemm::split_k_kernel<
                    Implicit_gemm_traits><<<workspace->split_k_grid,
                                            Implicit_gemm_traits::Cta_tile::THREADS_PER_CTA,
                                            workspace->epilogue_size_in_bytes,
                                            lwda_stream>>>( workspace->xmma_params );
            workspace->xmma_params.split_k.kernels = 2;
        }
        XMMA_LWDA_CALL( lwdaGetLastError() );

        return xmma::Error::SUCCESS;
    }

    static lwdaError_t get_func_attributes( lwdaFuncAttributes* attr ) {
        void ( *xmma_implicit_gemm_warp_specialized_kernel )(
                typename Implicit_gemm_traits::Params params ) = NULL;
        int warp_specialized_factor;

        xmma_implicit_gemm_warp_specialized_kernel =
            xmma::gemm::xmma_implicit_gemm_specialize_1math_1dma_arrive_wait_kernel<Implicit_gemm_traits>;
        warp_specialized_factor = 2;

        lwdaError_t lwda_status =
                lwdaFuncGetAttributes( attr, xmma_implicit_gemm_warp_specialized_kernel );
        attr->maxDynamicSharedSizeBytes = Implicit_gemm_traits::dynamic_smem_size_per_cta();
        attr->maxThreadsPerBlock = Implicit_gemm_traits::threads_per_cta() * warp_specialized_factor;
        return lwda_status;
    }

};

///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Implicit_gemm_traits,
          xmma::Colwolution_layout colw_layout,
          bool use_warp_specialization>
struct Device_kernel<Implicit_gemm_traits,
                     xmma::Operation_type::STRIDED_DGRAD,
                     xmma::Colwolution_algorithm::INDEX,
                     colw_layout,
                     use_warp_specialization> {

    static xmma::Error run( xmma::Host_workspace<Implicit_gemm_traits>* workspace,
                                lwdaStream_t& lwda_stream ) {
        if( workspace->smem_size > 48 * 1024 ) {
            if( workspace->xmma_params.ampere && workspace->smem_size > 164 * 1024 ||
                !workspace->xmma_params.ampere && workspace->smem_size > 64 * 1024 ) {
                printf( "[ ERROR: LWCA Runtime ] %s:%d: Invalid shared memory size!\n",
                        __FILE__,
                        __LINE__ );
                return xmma::Error::ERROR_LWDA_RUNTIME;
            }
            XMMA_LWDA_CALL( lwdaFuncSetAttribute(
                xmma::implicit_gemm::strided_dgrad_indexed::kernel<Implicit_gemm_traits>,
                lwdaFuncAttributeMaxDynamicSharedMemorySize,
                workspace->smem_size ) );
            XMMA_LWDA_CALL( lwdaFuncSetAttribute(
                xmma::implicit_gemm::strided_dgrad_indexed::kernel<Implicit_gemm_traits>,
                lwdaFuncAttributePreferredSharedMemoryCarveout,
                100 ) );
        }

        workspace->xmma_params.ndhw_indices_of_each_filter_pattern_gmem =
            (int32_t*)( xmma::ptr_to_int64( workspace->xmma_params.split_k.buffers_gmem ) +
                        workspace->device_workspace_size );

        workspace->xmma_params.start_cta_id_in_ndhw_dimension_of_each_filter_pattern =
            workspace->xmma_params.ndhw_indices_of_each_filter_pattern_gmem +
            workspace->xmma_params.sum_of_round_up_ndhw_number_of_each_filter_pattern;

        workspace->xmma_params.valid_t =
            workspace->xmma_params.start_cta_id_in_ndhw_dimension_of_each_filter_pattern +
            workspace->xmma_params.trs + 1;

        workspace->xmma_params.valid_r = workspace->xmma_params.valid_t + workspace->xmma_params.d;

        workspace->xmma_params.valid_s = workspace->xmma_params.valid_r + workspace->xmma_params.h;

        workspace->xmma_params.dhw_count_of_each_filter_pattern =
            workspace->xmma_params.valid_s + workspace->xmma_params.w;

        xmma::implicit_gemm::strided_dgrad_indexed::kernel_helper_stage_0<
            Implicit_gemm_traits><<<( workspace->xmma_params.d + workspace->xmma_params.h +
                                      workspace->xmma_params.w + 127 ) /
                                        128,
                                    128,
                                    0,
                                    lwda_stream>>>( workspace->xmma_params );
        if( lwdaGetLastError() != lwdaSuccess ) {
            return xmma::Error::ERROR_LWDA_RUNTIME;
        }

        xmma::implicit_gemm::strided_dgrad_indexed::kernel_helper_stage_1<
            Implicit_gemm_traits><<<workspace->xmma_params.trs + 1, 256, 0, lwda_stream>>>(
            workspace->xmma_params );
        if( lwdaGetLastError() != lwdaSuccess ) {
            return xmma::Error::ERROR_LWDA_RUNTIME;
        }

        xmma::implicit_gemm::strided_dgrad_indexed::kernel_helper_stage_2<
            Implicit_gemm_traits><<<1, 1, 0, lwda_stream>>>( workspace->xmma_params );
        if( lwdaGetLastError() != lwdaSuccess ) {
            return xmma::Error::ERROR_LWDA_RUNTIME;
        }

        xmma::implicit_gemm::strided_dgrad_indexed::kernel_helper_stage_3<
            Implicit_gemm_traits><<<workspace->xmma_params.trs + 1, 256, 0, lwda_stream>>>(
            workspace->xmma_params );
        if( lwdaGetLastError() != lwdaSuccess ) {
            return xmma::Error::ERROR_LWDA_RUNTIME;
        }

        xmma::implicit_gemm::strided_dgrad_indexed::kernel<
            Implicit_gemm_traits><<<workspace->grid,
                                    Implicit_gemm_traits::Cta_tile::THREADS_PER_CTA,
                                    workspace->smem_size,
            lwda_stream>>>( workspace->xmma_params );
        if( lwdaGetLastError() != lwdaSuccess ) {
            return xmma::Error::ERROR_LWDA_RUNTIME;
        }

        return xmma::Error::SUCCESS;
    }

    static lwdaError_t get_func_attributes( lwdaFuncAttributes* attr ) {
        lwdaError_t lwda_status = lwdaFuncGetAttributes(
            attr, xmma::implicit_gemm::strided_dgrad_indexed::kernel<Implicit_gemm_traits> );
        attr->maxDynamicSharedSizeBytes =
            Implicit_gemm_traits::dynamic_smem_size_per_cta();
        attr->maxThreadsPerBlock = Implicit_gemm_traits::threads_per_cta();
        return lwda_status;
    }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Implicit_gemm_traits, xmma::Colwolution_layout colw_layout>
struct Device_kernel<Implicit_gemm_traits,
                     xmma::Operation_type::STRIDED_DGRAD,
                     xmma::Colwolution_algorithm::INDEX,
                     colw_layout,
                     true> {

    static xmma::Error run( xmma::Host_workspace<Implicit_gemm_traits>* workspace,
                                lwdaStream_t& lwda_stream ) {
        if( workspace->smem_size > 48 * 1024 ) {
            if( workspace->xmma_params.ampere && workspace->smem_size > 164 * 1024 ||
                !workspace->xmma_params.ampere && workspace->smem_size > 64 * 1024 ) {
                printf( "[ ERROR: LWCA Runtime ] %s:%d: Invalid shared memory size!\n",
                        __FILE__,
                        __LINE__ );
                return xmma::Error::ERROR_LWDA_RUNTIME;
            }
            XMMA_LWDA_CALL( lwdaFuncSetAttribute(
                xmma::implicit_gemm::strided_dgrad_indexed::
                    xmma_implicit_gemm_strided_dgrad_specialize_1math_1dma_arrive_wait_kernel<
                        Implicit_gemm_traits>,
                lwdaFuncAttributeMaxDynamicSharedMemorySize,
                workspace->smem_size ) );
            XMMA_LWDA_CALL( lwdaFuncSetAttribute(
                xmma::implicit_gemm::strided_dgrad_indexed::
                    xmma_implicit_gemm_strided_dgrad_specialize_1math_1dma_arrive_wait_kernel<
                        Implicit_gemm_traits>,
                lwdaFuncAttributePreferredSharedMemoryCarveout,
                100 ) );
        }

        workspace->xmma_params.ndhw_indices_of_each_filter_pattern_gmem =
            (int32_t*)( xmma::ptr_to_int64( workspace->xmma_params.split_k.buffers_gmem ) +
                        workspace->device_workspace_size );

        workspace->xmma_params.start_cta_id_in_ndhw_dimension_of_each_filter_pattern =
            workspace->xmma_params.ndhw_indices_of_each_filter_pattern_gmem +
            workspace->xmma_params.sum_of_round_up_ndhw_number_of_each_filter_pattern;

        workspace->xmma_params.valid_t =
            workspace->xmma_params.start_cta_id_in_ndhw_dimension_of_each_filter_pattern +
            workspace->xmma_params.trs + 1;

        workspace->xmma_params.valid_r = workspace->xmma_params.valid_t + workspace->xmma_params.d;

        workspace->xmma_params.valid_s = workspace->xmma_params.valid_r + workspace->xmma_params.h;

        workspace->xmma_params.dhw_count_of_each_filter_pattern =
            workspace->xmma_params.valid_s + workspace->xmma_params.w;

        xmma::implicit_gemm::strided_dgrad_indexed::kernel_helper_stage_0<
            Implicit_gemm_traits><<<( workspace->xmma_params.d + workspace->xmma_params.h +
                                      workspace->xmma_params.w + 127 ) /
                                        128,
                                    128,
                                    0,
                                    lwda_stream>>>( workspace->xmma_params );
        if( lwdaGetLastError() != lwdaSuccess ) {
            return xmma::Error::ERROR_LWDA_RUNTIME;
        }

        xmma::implicit_gemm::strided_dgrad_indexed::kernel_helper_stage_1<
            Implicit_gemm_traits><<<workspace->xmma_params.trs + 1, 256, 0, lwda_stream>>>(
            workspace->xmma_params );
        if( lwdaGetLastError() != lwdaSuccess ) {
            return xmma::Error::ERROR_LWDA_RUNTIME;
        }

        xmma::implicit_gemm::strided_dgrad_indexed::kernel_helper_stage_2<
            Implicit_gemm_traits><<<1, 1, 0, lwda_stream>>>( workspace->xmma_params );
        if( lwdaGetLastError() != lwdaSuccess ) {
            return xmma::Error::ERROR_LWDA_RUNTIME;
        }

        xmma::implicit_gemm::strided_dgrad_indexed::kernel_helper_stage_3<
            Implicit_gemm_traits><<<workspace->xmma_params.trs + 1, 256, 0, lwda_stream>>>(
            workspace->xmma_params );
        if( lwdaGetLastError() != lwdaSuccess ) {
            return xmma::Error::ERROR_LWDA_RUNTIME;
        }

        xmma::implicit_gemm::strided_dgrad_indexed::
            xmma_implicit_gemm_strided_dgrad_specialize_1math_1dma_arrive_wait_kernel<
                Implicit_gemm_traits><<<workspace->grid,
                                        Implicit_gemm_traits::Cta_tile::THREADS_PER_CTA * 2,
                                        workspace->smem_size,
            lwda_stream>>>( workspace->xmma_params );
        if( lwdaGetLastError() != lwdaSuccess ) {
            return xmma::Error::ERROR_LWDA_RUNTIME;
        }

        return xmma::Error::SUCCESS;
    }

    static lwdaError_t get_func_attributes( lwdaFuncAttributes* attr ) {
        lwdaError_t lwda_status = lwdaFuncGetAttributes(
            attr,
            xmma::implicit_gemm::strided_dgrad_indexed::
                xmma_implicit_gemm_strided_dgrad_specialize_1math_1dma_arrive_wait_kernel<
                    Implicit_gemm_traits> );
        attr->maxDynamicSharedSizeBytes =
            Implicit_gemm_traits::dynamic_smem_size_per_cta();
        attr->maxThreadsPerBlock = Implicit_gemm_traits::threads_per_cta();
        return lwda_status;
    }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

// Specialized for IMMA kernels with interleaved layout
template <typename Implicit_gemm_traits>
struct Device_kernel<Implicit_gemm_traits,
                     xmma::Operation_type::FPROP,
                     xmma::Colwolution_algorithm::PRECOMPUTED,
                     xmma::Colwolution_layout::NCHW_VECT_C_32,
                     false> {

    static xmma::Error run( xmma::Host_workspace<Implicit_gemm_traits>* workspace,
                                lwdaStream_t& lwda_stream ) {
        if( workspace->smem_size > 48 * 1024 ) {
            if( workspace->xmma_params.ampere && workspace->smem_size > 164 * 1024 ||
                !workspace->xmma_params.ampere && workspace->smem_size > 64 * 1024 ) {
                printf( "[ ERROR: LWCA Runtime ] %s:%d: Invalid shared memory size!\n",
                        __FILE__,
                        __LINE__ );
                return xmma::Error::ERROR_LWDA_RUNTIME;
            }
            XMMA_LWDA_CALL( lwdaFuncSetAttribute(
                xmma::implicit_gemm::interleaved_fprop::kernel<Implicit_gemm_traits>,
                            lwdaFuncAttributeMaxDynamicSharedMemorySize,
                            workspace->smem_size ) );
            XMMA_LWDA_CALL( lwdaFuncSetAttribute(
                xmma::implicit_gemm::interleaved_fprop::kernel<Implicit_gemm_traits>,
                            lwdaFuncAttributePreferredSharedMemoryCarveout,
                            100 ) );
        }

        xmma::implicit_gemm::interleaved_fprop::kernel<
            Implicit_gemm_traits><<<workspace->grid,
                                    Implicit_gemm_traits::Cta_tile::THREADS_PER_CTA,
                                    workspace->smem_size,
            lwda_stream>>>( workspace->xmma_params );
        XMMA_LWDA_CALL( lwdaGetLastError() );

        return xmma::Error::SUCCESS;
    }

    static lwdaError_t get_func_attributes( lwdaFuncAttributes* attr ) {
        lwdaError_t lwda_status = lwdaFuncGetAttributes(
            attr, xmma::implicit_gemm::interleaved_fprop::kernel<Implicit_gemm_traits> );
        attr->maxDynamicSharedSizeBytes =
            Implicit_gemm_traits::dynamic_smem_size_per_cta();
        attr->maxThreadsPerBlock = Implicit_gemm_traits::threads_per_cta();
        return lwda_status;
    }
};


template <typename Implicit_gemm_traits>
struct Device_kernel<Implicit_gemm_traits,
                     xmma::Operation_type::FPROP,
                     xmma::Colwolution_algorithm::PRECOMPUTED,
                     xmma::Colwolution_layout::NCHW_VECT_C_8,
                     false> :  Device_kernel<Implicit_gemm_traits,
                        xmma::Operation_type::FPROP,
                        xmma::Colwolution_algorithm::PRECOMPUTED,
                        xmma::Colwolution_layout::NCHW_VECT_C_32,
                        false> {

};

///////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////

// Used for IMMA reorder kernel
struct Reorder_imma_filter_params {
    // The size of the filter.
    int k;
    // Fitler CRS.
    int crs;
    // Input channels
    int c;
    // TRS
    int trs;
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

// Lwca kernel for filter reorder
static __global__ void lwda_reorder_imma_dgrad_filter( Reorder_imma_filter_params params ) {
    const int BYTES_PER_PACKETS = 32;
    // The k position in both filters.
    int k = blockIdx.y;
    // The trs possition
    int trs = blockIdx.z;
    // The k position in the transformed filter.
    int xform_c = blockIdx.x * blockDim.x + threadIdx.x;
    // The position in the original filter.
    int c = ( xform_c & ~31 ) + ( ( xform_c & ( 3 << 1 ) ) << 2 ) +
            ( ( xform_c & ( 3 << 3 ) ) >> 2 ) + ( xform_c & 1 );

    // The offset in the xformed filter.
    int dst_offset =
        ( ( k * params.trs + trs ) * params.xform_filter_k + xform_c ) * BYTES_PER_PACKETS;

    // Read the data.
    int data0[4] = { 0 };
    int data1[4] = { 0 };
    int src_offset =
        ( k * params.trs * params.xform_filter_k + c * params.trs + trs ) * BYTES_PER_PACKETS;

    const int* read_ptr = reinterpret_cast<const int*>( &params.filter_gmem[src_offset] );

    if( c < params.c ) {
        for( int i = 0; i < 4; i++ ) {
            data0[i] = read_ptr[i];
            data1[i] = read_ptr[i + 4];
        }
    }

    // Write the data back.
    int* write_ptr = reinterpret_cast<int*>( &params.xform_filter_gmem[dst_offset] );
    if( xform_c < params.xform_filter_k ) {
        for( int i = 0; i < 4; i++ ) {
            write_ptr[i] = data0[i];
            write_ptr[i + 4] = data1[i];
        }
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////

static __global__ void lwda_reorder_hmma_filter( Reorder_imma_filter_params params ) {
    // From gemm.cpp, that's the mapping from the column k in the original filter
    // to the column nCdiv8Hw8ReorderCol[k] is the transformed filter.
    const int BYTES_PER_PACKETS = 16;
    // The C*R*S position in both filters.
    int crs = blockIdx.y;
    // The k position in the transformed filter.
    int xform_k = blockIdx.x * blockDim.x + threadIdx.x;

    // Hmm
    int k = xform_k;

    // The offset in the xformed filter.
    int dst_offset = ( crs * params.xform_filter_k + xform_k ) * BYTES_PER_PACKETS;

    // Read the data.
    int data[BYTES_PER_PACKETS / 16][4];

    for( int i = 0; i < 4; i++ ) {
        for ( int j = 0; j < BYTES_PER_PACKETS / 16; j++) {
            data[j][i] = 0;
        }
    }

    int src_offset = ( k * params.crs + crs ) * BYTES_PER_PACKETS;

    const int* read_ptr = reinterpret_cast<const int*>( &params.filter_gmem[src_offset] );
    if( k < params.k ) {
        for( int i = 0; i < 4; i++ ) {
            for ( int j = 0; j < BYTES_PER_PACKETS / 16; j++) {
                data[j][i] = read_ptr[i + j * 4];
            }
        }
    }

    // Write the data back.
    int* write_ptr = reinterpret_cast<int*>( &params.xform_filter_gmem[dst_offset] );
    if( xform_k < params.xform_filter_k ) {
        for( int i = 0; i < 4; i++ ) {
            for ( int j = 0; j < BYTES_PER_PACKETS / 16; j++) {
                write_ptr[i + j * 4] = data[j][i];
            }
        }
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Implicit_gemm_traits>
xmma::Error reorder_filter( const void* filter_data,
                            void* host_ptr,
                            void* device_ptr,
                                lwdaStream_t& lwda_stream ) {
    xmma::Host_workspace<Implicit_gemm_traits>* host_workspace =
        static_cast<xmma::Host_workspace<Implicit_gemm_traits>*>( host_ptr );

    if( host_workspace->xmma_params.is_interleaved ) {
        if( Implicit_gemm_traits::OPERATION_TYPE == xmma::Operation_type::DGRAD ) {
        // Initializing params
        Reorder_imma_filter_params reorder_params;
        reorder_params.c = host_workspace->xmma_params.c;
        reorder_params.k = host_workspace->xmma_params.k / 32;
            reorder_params.trs = host_workspace->xmma_params.t * host_workspace->xmma_params.r *
                             host_workspace->xmma_params.s;

        reorder_params.xform_filter_k = host_workspace->xmma_params.c;
        reorder_params.filter_gmem = reinterpret_cast<const char*>( filter_data );
        reorder_params.xform_filter_gmem = reinterpret_cast<char*>( device_ptr );

        int num_threads = 128;
        dim3 grid( xmma::div_up( reorder_params.c, num_threads ),
                   reorder_params.k,
                       reorder_params.trs );

        lwda_reorder_imma_dgrad_filter<<<grid, num_threads, 0, lwda_stream>>>( reorder_params );
        XMMA_LWDA_CALL( lwdaGetLastError() );

        host_workspace->xmma_params.flt_gmem = reorder_params.xform_filter_gmem;
        } else if (Implicit_gemm_traits::COLW_LAYOUT == xmma::Colwolution_layout::NCHW_VECT_C_8){
            // Initializing params
            Reorder_imma_filter_params reorder_params;
            reorder_params.k = host_workspace->xmma_params.k * host_workspace->xmma_params.g;
            reorder_params.crs = max(host_workspace->xmma_params.c, 8) / 8 *
                                 host_workspace->xmma_params.t * host_workspace->xmma_params.r *
                                 host_workspace->xmma_params.s;
            reorder_params.xform_filter_k =
                host_workspace->xmma_params.k * host_workspace->xmma_params.g;
            reorder_params.filter_gmem = reinterpret_cast<const char*>( filter_data );
            reorder_params.xform_filter_gmem = reinterpret_cast<char*>( device_ptr );
            int num_threads = 128;
            dim3 grid( xmma::div_up( reorder_params.k, num_threads ), reorder_params.crs );

            lwda_reorder_hmma_filter<<<grid, num_threads, 0, lwda_stream>>>( reorder_params );
            XMMA_LWDA_CALL( lwdaGetLastError() );

            host_workspace->xmma_params.flt_gmem = reorder_params.xform_filter_gmem;
        } else {
        // Initializing params
        Reorder_imma_filter_params reorder_params;
        reorder_params.k = host_workspace->xmma_params.k * host_workspace->xmma_params.g;
            reorder_params.crs = max(host_workspace->xmma_params.c, 32) / 32 *
                                 host_workspace->xmma_params.t * host_workspace->xmma_params.r *
                                 host_workspace->xmma_params.s;
            reorder_params.xform_filter_k =
                host_workspace->xmma_params.k * host_workspace->xmma_params.g;
        reorder_params.filter_gmem = reinterpret_cast<const char*>( filter_data );
        reorder_params.xform_filter_gmem = reinterpret_cast<char*>( device_ptr );

        int num_threads = 128;
        dim3 grid( xmma::div_up( reorder_params.k, num_threads ), reorder_params.crs );

        lwda_reorder_imma_filter<<<grid, num_threads, 0, lwda_stream>>>( reorder_params );
        XMMA_LWDA_CALL( lwdaGetLastError() );

        host_workspace->xmma_params.flt_gmem = reorder_params.xform_filter_gmem;
        }
        return xmma::Error::SUCCESS;
    } else {
        return xmma::Error::ERROR_ILWALID_PARAMS;
    }
}
///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Implicit_gemm_traits> size_t get_host_workspace_size() {
    return sizeof( xmma::Host_workspace<Implicit_gemm_traits> );
}

///////////////////////////////////////////////////////////////////////////////////////////////////

// FIXME: deprecated when move to CASK 4.0
template <typename Implicit_gemm_traits>
xmma::Error initialize_host_workspace( typename Implicit_gemm_traits::Params& xmma_params, void* host_ptr ) {
    xmma::Host_workspace<Implicit_gemm_traits>* workspace =
        static_cast<xmma::Host_workspace<Implicit_gemm_traits>*>( host_ptr );

    return xmma_params.initialize( workspace );
}

template <typename Implicit_gemm_traits>
xmma::Error initialize_host_workspace( Host_workspace<Implicit_gemm_traits>* workspace ) {
    typename Implicit_gemm_traits::Params& params = workspace->xmma_params;

    params.initialize(workspace);
    params.ampere = std::is_same<typename Implicit_gemm_traits::Traits::Gpu_arch, xmma::Ampere>::value;

    return xmma::Error::SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Implicit_gemm_traits>
size_t get_device_workspace_size( const void* host_ptr ) {
    auto host_workspace = static_cast<const xmma::Host_workspace<Implicit_gemm_traits>*>( host_ptr );
    return host_workspace->device_workspace_size;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Kernel_traits>
xmma::Error initialize_device_workspace( const xmma::Host_workspace<Kernel_traits>* host_workspace,
                                         void* device_workspace,
                                         lwdaStream_t stream ) {
    return xmma::Error::SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Implicit_gemm_traits, typename X_type, typename Y_type, typename W_type>
xmma::Error run_kernel( X_type* x_data,
                        Y_type* y_data,
                        W_type* w_data,
                        const void* res_data,
                        const void* bias_data,
                        const void* alpha_data,
                        const void* beta_data,
                        void* host_ptr,
                        void* device_ptr,
                        Runtime_params<Implicit_gemm_traits>& runtime_params,
                        lwdaStream_t& lwda_stream ) {
    auto host_workspace = static_cast<xmma::Host_workspace<Implicit_gemm_traits>*>( host_ptr );
    auto& params = host_workspace->xmma_params;

    Device_pointers<Implicit_gemm_traits, Implicit_gemm_traits::OPERATION_TYPE>::init(
        params, x_data, y_data, w_data, res_data );

    params.bias_gmem = bias_data;
    params.alpha_gmem = alpha_data;
    params.beta_gmem = beta_data;

    // Initialize the L2 descriptors
    params.mem_descriptors.descriptor_a = ( (uint64_t)runtime_params.descriptor_a << 32 );
    params.mem_descriptors.descriptor_b = ( (uint64_t)runtime_params.descriptor_b << 32 );
    params.mem_descriptors.descriptor_c =
        ( (uint64_t)runtime_params.descriptor_c1 << 32 ) + (uint64_t)runtime_params.descriptor_c0;
    params.mem_descriptors.descriptor_d =
        ( (uint64_t)runtime_params.descriptor_d1 << 32 ) + (uint64_t)runtime_params.descriptor_d0;
    float *geluScale = &runtime_params.gelu_scale;
    params.runtime_params.runtime_param0 = *reinterpret_cast<int32_t*>(geluScale);

    // Setup split-k buffers
    params.split_k.set_base_ptr( device_ptr );
    XMMA_LWDA_CALL(params.split_k.clear_buffers( device_ptr, lwda_stream ));

    // Update batchSize.
    if( runtime_params.batch_size > 0 && runtime_params.batch_size != params.n ) {
        params.n = runtime_params.batch_size;

        // Re-initialize xmma_params with new batchSize.
        params.initialize( host_workspace );
    }

    if( Device_kernel<Implicit_gemm_traits,
                      Implicit_gemm_traits::OPERATION_TYPE,
                      Implicit_gemm_traits::COLW_ALGO,
                      Implicit_gemm_traits::COLW_LAYOUT,
                      Implicit_gemm_traits::USE_WARP_SPECIALIZATION>::run( host_workspace,
                                                                           lwda_stream ) !=
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
                      Implicit_gemm_traits::COLW_LAYOUT,
                      Implicit_gemm_traits::USE_WARP_SPECIALIZATION>::get_func_attributes( attr ) !=
        lwdaSuccess ) {
        return xmma::Error::ERROR_LWDA_RUNTIME;
    }
    return xmma::Error::SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Implicit_gemm_traits, typename X_type, typename Y_type, typename W_type>
xmma::Error run_kernel_tma( X_type* x_data,
                            Y_type* y_data,
                            W_type* w_data,
                            const void* res_data,
                            const void* bias_data,
                            const void* alpha_data,
                            const void* beta_data,
                            void* host_ptr,
                            void* device_ptr,
                            Runtime_params<Implicit_gemm_traits>& runtime_params,
                            lwdaStream_t& lwda_stream ) {

    xmma::Host_workspace<Implicit_gemm_traits>* host_workspace =
        static_cast<xmma::Host_workspace<Implicit_gemm_traits>*>( host_ptr );
    auto& params = host_workspace->xmma_params;

    Device_pointers<Implicit_gemm_traits, Implicit_gemm_traits::OPERATION_TYPE>::init(
        params, x_data, y_data, w_data, res_data );

    params.bias_gmem = bias_data;
    params.alpha_gmem = alpha_data;
    params.beta_gmem = beta_data;

    uint32_t tensor_size_a[5], tensor_size_b[5];
    uint32_t box_size_b[5];
    uint32_t box_size_ndhw, box_size_c;

    swizzle_t swizzle_a, swizzle_b;

    // constexpr int BYTES_PER_ELEMENT_A = Implicit_gemm_traits::BITS_PER_ELEMENT_A / 8;
    constexpr int BYTES_PER_ELEMENT_B = Implicit_gemm_traits::BITS_PER_ELEMENT_B / 8;

    // Image tensor, layout NDHWC
    tensor_size_a[0] = host_workspace->xmma_params.c;
    tensor_size_a[1] = host_workspace->xmma_params.w;
    tensor_size_a[2] = host_workspace->xmma_params.h;
    tensor_size_a[3] = host_workspace->xmma_params.d;
    tensor_size_a[4] = host_workspace->xmma_params.n;

    box_size_c = xmma::tma::kTileBlockRow( Implicit_gemm_traits::Cta_tile::K,
                                           Implicit_gemm_traits::BITS_PER_ELEMENT_A );
    box_size_ndhw = xmma::tma::kTileBlockCol( Implicit_gemm_traits::Cta_tile::M );

    swizzle_a = xmma::tma::kTileBlockRowSwizzle( Implicit_gemm_traits::Cta_tile::K,
                                                 Implicit_gemm_traits::BITS_PER_ELEMENT_A );

    // KTRSC
    tensor_size_b[0] = host_workspace->xmma_params.c;
    tensor_size_b[1] = host_workspace->xmma_params.s;
    tensor_size_b[2] = host_workspace->xmma_params.r;
    tensor_size_b[3] = host_workspace->xmma_params.t;
    tensor_size_b[4] = host_workspace->xmma_params.k;

    box_size_b[0] = xmma::tma::kTileBlockRow( Implicit_gemm_traits::Cta_tile::K,
                                              Implicit_gemm_traits::BITS_PER_ELEMENT_B );;
    box_size_b[1] = 1; // fit_t
    box_size_b[2] = 1; // flt_r
    box_size_b[3] = 1; // flt_s
    box_size_b[4] = xmma::tma::kTileBlockCol( Implicit_gemm_traits::Cta_tile::N );

    swizzle_b = xmma::tma::kTileBlockRowSwizzle( Implicit_gemm_traits::Cta_tile::K,
                                                 Implicit_gemm_traits::BITS_PER_ELEMENT_B );

    // Set tensor strides.
    uint64_t tensor_stride_a[4], tensor_stride_b[4];
    tensor_stride_a[0] = host_workspace->xmma_params.img_stride_w * BYTES_PER_ELEMENT_B;
    tensor_stride_a[1] = host_workspace->xmma_params.img_stride_h * BYTES_PER_ELEMENT_B;
    tensor_stride_a[2] = host_workspace->xmma_params.img_stride_d * BYTES_PER_ELEMENT_B;
    tensor_stride_a[3] = host_workspace->xmma_params.img_stride_n * BYTES_PER_ELEMENT_B;

    tensor_stride_b[0] = tensor_size_b[0] * BYTES_PER_ELEMENT_B;
    tensor_stride_b[1] = tensor_size_b[1] * tensor_stride_b[0];
    tensor_stride_b[2] = tensor_size_b[2] * tensor_stride_b[1];
    tensor_stride_b[3] = tensor_size_b[3] * tensor_stride_b[2];

    // Initialize TMA descriptors
    using A_type = typename Implicit_gemm_traits::Traits::A_type;
    using B_type = typename Implicit_gemm_traits::Traits::B_type;

    // Set image tensor descriptor.
    using namespace xmma::hopper::emu;
    lwdaTmaDescv2 a_desc, b_desc;
    lwdaTmaDescv2 *a_desc_d, *b_desc_d;
    /*
        XMMA_LWDA_CALL( lwdaSetTmaTensorDescriptor(
            &a_desc,
            x_data,
            xmma::hopper::emu::get_data_type_tma_desc<A_type>(),
            INTERLEAVED_NONE,
            tensor_size_a,
            tensor_stride_a,
            5 ) );

        // Set weight tensor descriptor.
        XMMA_LWDA_CALL( lwdaSetTmaTensorDescriptor(
            &b_desc,
            y_data,
            xmma::hopper::emu::get_data_type_tma_desc<B_type>(),
            INTERLEAVED_NONE,
            tensor_size_b,
            tensor_stride_b,
            5 ) );
    */
    int32_t base_corner[3], far_corner[3];
    base_corner[2] = -host_workspace->xmma_params.pad[0][0];  // d
    base_corner[1] = -host_workspace->xmma_params.pad[1][0];  // h
    base_corner[0] = -host_workspace->xmma_params.pad[2][0];  // w
    far_corner[2] = host_workspace->xmma_params.pad[0][1] -
                    (host_workspace->xmma_params.t - 1) *
                        host_workspace->xmma_params.dilation[0];  // d
    far_corner[1] = host_workspace->xmma_params.pad[1][1] -
                    (host_workspace->xmma_params.r - 1) *
                        host_workspace->xmma_params.dilation[1];  // h
    far_corner[0] = host_workspace->xmma_params.pad[2][1] -
                    (host_workspace->xmma_params.s - 1) *
                        host_workspace->xmma_params.dilation[2];  // w

    uint32_t traversal_stride_b[5] = { 1, 1, 1, 1, 1 };
    uint32_t traversal_stride_a[5] = { 1, 1, 1, 1, 1 };
    traversal_stride_a[3] = (uint32_t)host_workspace->xmma_params.stride[0];  // d
    traversal_stride_a[2] = (uint32_t)host_workspace->xmma_params.stride[1];  // h
    traversal_stride_a[1] = (uint32_t)host_workspace->xmma_params.stride[2];  // w
                                                                              /*
                                                                                  XMMA_LWDA_CALL(lwdaSetTmaImageAccessDescriptor(
                                                                                      &a_desc, swizzle_a, traversal_stride_a, box_size_c, box_size_ndhw,
                                                                                      base_corner, far_corner, 5));
                                                                                  XMMA_LWDA_CALL(lwdaSetTmaTileAccessDescriptor(
                                                                                      &b_desc, swizzle_b, traversal_stride_b, box_size_b, 5));
                                                                              */
    XMMA_LWDA_CALL(
        lwdaSetTmaIm2ColDescriptorv2( &a_desc,
                                      x_data,
                                      5,
                                      xmma::hopper::emu::get_data_type_tma_desc<A_type>(),
                                      INTERLEAVE_DISABLED,
                                      swizzle_a,
                                      PROMOTION_DISABLED,
                                      tensor_size_a,
                                      tensor_stride_a,
                                      traversal_stride_a,
                                      box_size_c,
                                      box_size_ndhw,
                                      base_corner,
                                      far_corner,
                                      0,
                                      0 ) );

    XMMA_LWDA_CALL( lwdaSetTmaTileDescriptorv2( &b_desc,
                                                y_data,
                                                5,
                                                xmma::hopper::emu::get_data_type_tma_desc<B_type>(),
                                                INTERLEAVE_DISABLED,
                                                swizzle_b,
                                                PROMOTION_DISABLED,
                                                tensor_size_b,
                                                tensor_stride_b,
                                                traversal_stride_b,
                                                box_size_b,
                                                0,
                                                0 ) );

    XMMA_LWDA_CALL( lwdaMalloc( &a_desc_d, 64 ) );
    XMMA_LWDA_CALL( lwdaMemcpyAsync( a_desc_d, &a_desc, 64, lwdaMemcpyHostToDevice, lwda_stream ) );
    host_workspace->xmma_params.a_desc = a_desc_d;

    XMMA_LWDA_CALL(
        lwdaMalloc( &b_desc_d, 64 ) );  //< This is not freed later. Memory leak! Fix later
    XMMA_LWDA_CALL( lwdaMemcpyAsync( b_desc_d, &b_desc, 64, lwdaMemcpyHostToDevice, lwda_stream ) );
    host_workspace->xmma_params.b_desc = b_desc_d;

    // Initialize the L2 descriptors
    host_workspace->xmma_params.mem_descriptors.descriptor_a =
        ( (uint64_t)runtime_params.descriptor_a << 32 );
    host_workspace->xmma_params.mem_descriptors.descriptor_b =
        ( (uint64_t)runtime_params.descriptor_b << 32 );
    host_workspace->xmma_params.mem_descriptors.descriptor_c =
        ( (uint64_t)runtime_params.descriptor_c1 << 32 ) + (uint64_t)runtime_params.descriptor_c0;
    host_workspace->xmma_params.mem_descriptors.descriptor_d =
        ( (uint64_t)runtime_params.descriptor_d1 << 32 ) + (uint64_t)runtime_params.descriptor_d0;

    const size_t alignment = 16 - ( (size_t)device_ptr ) % 16;

    host_workspace->xmma_params.split_k.buffers_gmem =
        (void*)( xmma::ptr_to_int64( device_ptr ) + alignment );
    host_workspace->xmma_params.split_k.counters_gmem =
        (int32_t*)( xmma::ptr_to_int64( host_workspace->xmma_params.split_k.buffers_gmem ) +
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

    if( host_workspace->smem_size > 48 * 1024 ) {
        if( ( host_workspace->xmma_params.ampere || host_workspace->xmma_params.hopper ) &&
                host_workspace->smem_size > 164 * 1024 ||
            ( !host_workspace->xmma_params.ampere && !host_workspace->xmma_params.hopper ) &&
                host_workspace->smem_size > 64 * 1024 ) {
            return xmma::Error::ERROR_LWDA_RUNTIME;
        }
        XMMA_LWDA_CALL( lwdaFuncSetAttribute( xmma::gemm::kernel_tma<Implicit_gemm_traits>,
                                              lwdaFuncAttributeMaxDynamicSharedMemorySize,
                                              host_workspace->smem_size ) );
        XMMA_LWDA_CALL( lwdaFuncSetAttribute( xmma::gemm::kernel_tma<Implicit_gemm_traits>,
                                              lwdaFuncAttributePreferredSharedMemoryCarveout,
                                              100 ) );
    }

    xmma::gemm::kernel_tma<Implicit_gemm_traits><<<host_workspace->grid,
                                            Implicit_gemm_traits::Cta_tile::THREADS_PER_CTA,
                                            host_workspace->smem_size,
                                            lwda_stream>>>( host_workspace->xmma_params );
    XMMA_LWDA_CALL( lwdaGetLastError() );

    // If we need two kernels to run split-k launch the second grid.
    if( host_workspace->xmma_params.split_k.kernels == 2 ) {
        host_workspace->xmma_params.split_k.kernels = 1;
        host_workspace->split_k_grid = host_workspace->grid;
        host_workspace->split_k_grid.z = Implicit_gemm_traits::Xmma_tile::XMMAS_M;

        if( host_workspace->epilogue_size_in_bytes > 48 * 1024 ) {
            if( host_workspace->xmma_params.ampere &&
                    host_workspace->epilogue_size_in_bytes > 164 * 1024 ||
                !host_workspace->xmma_params.ampere &&
                    host_workspace->epilogue_size_in_bytes > 64 * 1024 ) {
        return xmma::Error::ERROR_LWDA_RUNTIME;
    }
            XMMA_LWDA_CALL( lwdaFuncSetAttribute( xmma::gemm::split_k_kernel<Implicit_gemm_traits>,
                                                  lwdaFuncAttributeMaxDynamicSharedMemorySize,
                                                  host_workspace->epilogue_size_in_bytes ) );
            XMMA_LWDA_CALL( lwdaFuncSetAttribute( xmma::gemm::split_k_kernel<Implicit_gemm_traits>,
                                                  lwdaFuncAttributePreferredSharedMemoryCarveout,
                                                  100 ) );
        }

        xmma::gemm::split_k_kernel<Implicit_gemm_traits><<<host_workspace->split_k_grid,
                                                    Implicit_gemm_traits::Cta_tile::THREADS_PER_CTA,
                                                    host_workspace->epilogue_size_in_bytes,
                                                    lwda_stream>>>( host_workspace->xmma_params );
        host_workspace->xmma_params.split_k.kernels = 2;
    }
    XMMA_LWDA_CALL( lwdaGetLastError() );

    return xmma::Error::SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef USE_GMMA
template <typename Implicit_gemm_traits, typename X_type, typename Y_type, typename W_type>
xmma::Error run_kernel_gmma( X_type* x_data, Y_type* y_data, W_type* w_data,
                            const void* res_data, const void* bias_data, const void* alpha_data,
                            const void* beta_data, void* host_ptr, void* device_ptr,
                            Runtime_params<Implicit_gemm_traits>& runtime_params,
                            lwdaStream_t& lwda_stream) {

    xmma::Host_workspace<Implicit_gemm_traits>* host_workspace =
        static_cast<xmma::Host_workspace<Implicit_gemm_traits>*>( host_ptr );

    Device_pointers<Implicit_gemm_traits, Implicit_gemm_traits::OPERATION_TYPE>::init(
        host_workspace->xmma_params, x_data, y_data, w_data, res_data );

    const size_t alignment = 16 - ((size_t)device_ptr) % 16;

    host_workspace->xmma_params.split_k.buffers_gmem =
        (void*)(xmma::ptr_to_int64(device_ptr) + alignment);
    host_workspace->xmma_params.split_k.counters_gmem =
        (int32_t*)( xmma::ptr_to_int64( host_workspace->xmma_params.split_k.buffers_gmem ) +
                    host_workspace->xmma_params.split_k.buffer_size *
                        host_workspace->xmma_params.split_k.buffers );
    host_workspace->xmma_params.split_k.retired_ctas_gmem =
        &( host_workspace->xmma_params.split_k
               .counters_gmem )[host_workspace->xmma_params.split_k.counters_ctas_size /
                                sizeof( int32_t )];

    // Clear the buffer of counters for split-k (if needed).
    if( host_workspace->split_k_with_reduction ) {
        XMMA_LWDA_CALL( lwdaMemsetAsync( host_workspace->xmma_params.split_k.counters_gmem, 0,
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

    // Update batchSize.
    if( runtime_params.batch_size > 0 &&
        runtime_params.batch_size != host_workspace->xmma_params.n ) {
        host_workspace->xmma_params.n = runtime_params.batch_size;

        // Re-initialize xmma_params with new batchSize.
        host_workspace->xmma_params.initialize( host_workspace );
    }

  // Note for Hopper we have increased the smem budget to 228 KB
  if( host_workspace->smem_size > 48 * 1024 ) {
    if( (host_workspace->xmma_params.ampere || host_workspace->xmma_params.hopper) && host_workspace->smem_size > 228*1024 ||
       !(host_workspace->xmma_params.ampere || host_workspace->xmma_params.hopper) && host_workspace->smem_size > 64*1024) {
      return xmma::Error::ERROR_LWDA_RUNTIME;
    }
    XMMA_LWDA_CALL(lwdaFuncSetAttribute(
      xmma::gemm::kernel_gmma<Implicit_gemm_traits>,
      lwdaFuncAttributeMaxDynamicSharedMemorySize,
      host_workspace->smem_size));
    XMMA_LWDA_CALL(lwdaFuncSetAttribute(
      xmma::gemm::kernel_gmma<Implicit_gemm_traits>,
      lwdaFuncAttributePreferredSharedMemoryCarveout,
      100));
  }

  xmma::gemm::kernel_gmma<Implicit_gemm_traits><<<
      host_workspace->grid,
      Implicit_gemm_traits::Cta_tile::THREADS_PER_CTA,
      host_workspace->smem_size,
      lwda_stream>>>(host_workspace->xmma_params);
  XMMA_LWDA_CALL(lwdaGetLastError());

    return xmma::Error::SUCCESS;
}

template <typename Implicit_gemm_traits, typename X_type, typename Y_type, typename W_type>
xmma::Error run_kernel_gmma_arf( X_type* x_data, Y_type* y_data, W_type* w_data,
                            const void* res_data, const void* bias_data, const void* alpha_data,
                            const void* beta_data, void* host_ptr, void* device_ptr,
                            Runtime_params<Implicit_gemm_traits>& runtime_params,
                            lwdaStream_t& lwda_stream) {

    xmma::Host_workspace<Implicit_gemm_traits>* host_workspace =
        static_cast<xmma::Host_workspace<Implicit_gemm_traits>*>( host_ptr );

    Device_pointers<Implicit_gemm_traits, Implicit_gemm_traits::OPERATION_TYPE>::init(
        host_workspace->xmma_params, x_data, y_data, w_data, res_data );

    const size_t alignment = 16 - ((size_t)device_ptr) % 16;

    host_workspace->xmma_params.split_k.buffers_gmem =
        (void*)(xmma::ptr_to_int64(device_ptr) + alignment);
    host_workspace->xmma_params.split_k.counters_gmem =
        (int32_t*)( xmma::ptr_to_int64( host_workspace->xmma_params.split_k.buffers_gmem ) +
                    host_workspace->xmma_params.split_k.buffer_size *
                        host_workspace->xmma_params.split_k.buffers );
    host_workspace->xmma_params.split_k.retired_ctas_gmem =
        &( host_workspace->xmma_params.split_k
               .counters_gmem )[host_workspace->xmma_params.split_k.counters_ctas_size /
                                sizeof( int32_t )];

    // Clear the buffer of counters for split-k (if needed).
    if( host_workspace->split_k_with_reduction ) {
        XMMA_LWDA_CALL( lwdaMemsetAsync( host_workspace->xmma_params.split_k.counters_gmem, 0,
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

    // Update batchSize.
    if( runtime_params.batch_size > 0 &&
        runtime_params.batch_size != host_workspace->xmma_params.n ) {
        host_workspace->xmma_params.n = runtime_params.batch_size;

        // Re-initialize xmma_params with new batchSize.
        host_workspace->xmma_params.initialize( host_workspace );
    }

  // Note for Hopper we have increased the smem budget to 228 KB
  if( host_workspace->smem_size > 48 * 1024 ) {
    if( (host_workspace->xmma_params.ampere || host_workspace->xmma_params.hopper) && host_workspace->smem_size > 228*1024 ||
       !(host_workspace->xmma_params.ampere || host_workspace->xmma_params.hopper) && host_workspace->smem_size > 64*1024) {
      return xmma::Error::ERROR_LWDA_RUNTIME;
    }
    XMMA_LWDA_CALL(lwdaFuncSetAttribute(
      xmma::gemm::kernel_gmma_arf<Implicit_gemm_traits>,
      lwdaFuncAttributeMaxDynamicSharedMemorySize,
      host_workspace->smem_size));
    XMMA_LWDA_CALL(lwdaFuncSetAttribute(
      xmma::gemm::kernel_gmma_arf<Implicit_gemm_traits>,
      lwdaFuncAttributePreferredSharedMemoryCarveout,
      100));
  }

  xmma::gemm::kernel_gmma_arf<Implicit_gemm_traits><<<
      host_workspace->grid,
      Implicit_gemm_traits::Cta_tile::THREADS_PER_CTA,
      host_workspace->smem_size,
      lwda_stream>>>(host_workspace->xmma_params);
  XMMA_LWDA_CALL(lwdaGetLastError());

    return xmma::Error::SUCCESS;
}
#endif // USE_GMMA

}  // namespace implicit_gemm
}  // namespace xmma
