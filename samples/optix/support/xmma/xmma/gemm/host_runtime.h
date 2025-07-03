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
#include <xmma/warp_specialized_traits.h>

#include <xmma/gemm/kernel.h>
#include <xmma/gemm/kernel_hopper.h>
#include <xmma/gemm/params.h>

#include <xmma/gemm/warp_specialized_kernel.h>
#include <xmma/gemm/warp_specialized_kernel_hopper.h>
#include <xmma/gemm/warp_specialized_params.h>

#include <xmma/hopper/emu/lwda_tma_utils.h>
#include <xmma/hopper/emu/xmma_tma_helpers.h>

#include <xmma/hopper/traits.h>
#include <xmma/hopper/cluster.h>

namespace xmma {
namespace gemm {

///////////////////////////////////////////////////////////////////////////////////////////////////
template<typename Kernel_traits>
struct Runtime_params {
  int32_t descriptor_a;
  int32_t descriptor_b;
  int32_t descriptor_c0;
  int32_t descriptor_c1;
  int32_t descriptor_d0;
  int32_t descriptor_d1;
  // gelu runtime scale factor
  float gelu_scale;
};

///////////////////////////////////////////////////////////////////////////////////////////////////

// FIXME: deprecated when move to CASK 4.0
template <typename Kernel_traits> size_t get_host_workspace_size() {
    return sizeof( xmma::Host_workspace<Kernel_traits> );
}

///////////////////////////////////////////////////////////////////////////////////////////////////

// FIXME: deprecated when move to CASK 4.0
template <typename Kernel_traits>
xmma::Error initialize_host_workspace( typename Kernel_traits::Params& params, void* host_ptr ) {
    auto workspace = static_cast<xmma::Host_workspace<Kernel_traits>*>( host_ptr );
    workspace->xmma_params = params;
    return initialize_host_workspace(workspace);
}

template <typename Kernel_traits>
xmma::Error initialize_host_workspace( Host_workspace<Kernel_traits>* workspace ) {
    using Cta_tile = typename Kernel_traits::Cta_tile;

    typename Kernel_traits::Params& params = workspace->xmma_params;

    XMMA_CALL(params.callwlate_grid_dimensions(workspace->grid, workspace->split_k_grid, Kernel_traits::Xmma_tile::XMMAS_M));


    //FIXME: need to make sure warp specialized kernel works with finalize
    params.initialize(workspace);
    // params.finalize(workspace->grid, workspace->split_k_grid);

    workspace->smem_size = Kernel_traits::dynamic_smem_size_per_cta();
    const int EPILOGUE_SIZE_IN_BYTES = Kernel_traits::Swizzle_epilogue::BYTES_PER_TILE;
    workspace->epilogue_size_in_bytes = EPILOGUE_SIZE_IN_BYTES;

    // Do we need a sequential reduction?
    workspace->split_k_with_reduction = params.split_k.with_reduction();
    workspace->device_workspace_size = params.split_k.size_in_bytes();

    return xmma::Error::SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Kernel_traits> size_t get_device_workspace_size( const void* host_ptr ) {
    auto workspace = static_cast<const xmma::Host_workspace<Kernel_traits>*>( host_ptr );

    return workspace->device_workspace_size;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Kernel_traits>
size_t get_device_workspace_size( const xmma::Host_workspace<Kernel_traits>* host_workspace ) {
    return host_workspace->device_workspace_size;
}

template <typename Kernel_traits>
xmma::Error initialize_device_workspace( const xmma::Host_workspace<Kernel_traits>* host_workspace,
                                         void* device_workspace,
                                         lwdaStream_t stream ) {
    return xmma::Error::SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Kernel_traits, bool use_warp_specialization> struct Device_kernel {

    static xmma::Error run( xmma::Host_workspace<Kernel_traits> *host_workspace,
                            lwdaStream_t &lwda_stream ) {
        // Non-deterministic  reduction using atomics, split-k-mode == 3
        // zero-init output C if beta == 0
        if( host_workspace->xmma_params.split_k.kernels == 3 &&
            !host_workspace->xmma_params.with_residual ) {

            size_t c_mn = (size_t)host_workspace->xmma_params.n * host_workspace->xmma_params.ldc;
            size_t c_sz = c_mn * Kernel_traits::BITS_PER_ELEMENT_C / 8;
            void *c_d = (void *)host_workspace->xmma_params.c_gmem;
            XMMA_LWDA_CALL( lwdaMemsetAsync( c_d, 0, c_sz, lwda_stream ) );
        }

        // Clear the buffer of counters for split-k (if needed).
        if( host_workspace->split_k_with_reduction ) {
            XMMA_LWDA_CALL(
                lwdaMemsetAsync( host_workspace->xmma_params.split_k.counters_gmem,
                                 0,
                                 host_workspace->xmma_params.split_k.counters_ctas_size +
                                     host_workspace->xmma_params.split_k.retired_ctas_size,
                                 lwda_stream ) );
        }

        if( host_workspace->smem_size > 48 * 1024 ) {
            if( host_workspace->xmma_params.hopper && host_workspace->smem_size > 228 * 1024 ||
                host_workspace->xmma_params.ampere && host_workspace->smem_size > 164 * 1024 ||
                !host_workspace->xmma_params.ampere && !host_workspace->xmma_params.hopper &&
                    host_workspace->smem_size > 64 * 1024 ) {
                return xmma::Error::ERROR_LWDA_RUNTIME;
            }
            XMMA_LWDA_CALL( lwdaFuncSetAttribute( xmma::gemm::kernel<Kernel_traits>,
                                                  lwdaFuncAttributeMaxDynamicSharedMemorySize,
                                                  host_workspace->smem_size ) );
            XMMA_LWDA_CALL( lwdaFuncSetAttribute( xmma::gemm::kernel<Kernel_traits>,
                                                  lwdaFuncAttributePreferredSharedMemoryCarveout,
                                                  100 ) );
        }

        xmma::gemm::kernel<Kernel_traits><<<host_workspace->grid,
                                            Kernel_traits::Cta_tile::THREADS_PER_CTA,
                                            host_workspace->smem_size,
                                            lwda_stream>>>( host_workspace->xmma_params );
        XMMA_LWDA_CALL( lwdaGetLastError() );

        // If we need two kernels to run split-k launch the second grid.
        if( host_workspace->xmma_params.split_k.kernels == 2 ) {
            host_workspace->xmma_params.split_k.kernels = 1;
            // host_workspace->split_k_grid = host_workspace->grid;
            // host_workspace->split_k_grid.z = Kernel_traits::Xmma_tile::XMMAS_M;

            if( host_workspace->epilogue_size_in_bytes > 48 * 1024 ) {
                if( host_workspace->xmma_params.hopper && host_workspace->smem_size > 228 * 1024 ||
                    host_workspace->xmma_params.ampere &&
                        host_workspace->epilogue_size_in_bytes > 164 * 1024 ||
                    !host_workspace->xmma_params.ampere && !host_workspace->xmma_params.hopper &&
                        host_workspace->epilogue_size_in_bytes > 64 * 1024 ) {
                    return xmma::Error::ERROR_LWDA_RUNTIME;
                }
                XMMA_LWDA_CALL( lwdaFuncSetAttribute( xmma::gemm::split_k_kernel<Kernel_traits>,
                                                      lwdaFuncAttributeMaxDynamicSharedMemorySize,
                                                      host_workspace->epilogue_size_in_bytes ) );
                XMMA_LWDA_CALL(
                    lwdaFuncSetAttribute( xmma::gemm::split_k_kernel<Kernel_traits>,
                                          lwdaFuncAttributePreferredSharedMemoryCarveout,
                                          100 ) );
            }

            xmma::gemm::split_k_kernel<Kernel_traits>
                <<<host_workspace->split_k_grid,
                   Kernel_traits::Cta_tile::THREADS_PER_CTA,
                   host_workspace->epilogue_size_in_bytes,
                   lwda_stream>>>( host_workspace->xmma_params );
            host_workspace->xmma_params.split_k.kernels = 2;
        }
        XMMA_LWDA_CALL( lwdaGetLastError() );

        return xmma::Error::SUCCESS;
    }

    static lwdaError_t get_func_attributes( lwdaFuncAttributes *attr ) {
        lwdaError_t lwda_status = lwdaFuncGetAttributes( attr, xmma::gemm::kernel<Kernel_traits> );
        attr->maxDynamicSharedSizeBytes = Kernel_traits::get_dynamic_shared_memory_size_per_cta();
        attr->maxThreadsPerBlock = Kernel_traits::get_threads_per_cta();
        return lwda_status;
    }
};  // end baseline device kernel

///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Kernel_traits> struct Device_kernel<Kernel_traits, true> {

    static xmma::Error run( xmma::Host_workspace<Kernel_traits> *host_workspace,
                            lwdaStream_t &lwda_stream ) {
        void ( *xmma_implicit_gemm_warp_specialized_kernel )(
            typename Kernel_traits::Params params ) = NULL;
        int warp_specialized_factor;

        if( host_workspace->xmma_params.specialize == xmma::CONFIG_1DMA_1MATH ) {
            xmma_implicit_gemm_warp_specialized_kernel =
                &xmma::gemm::xmma_implicit_gemm_specialize_1math_1dma_arrive_wait_kernel<
                    Kernel_traits>;
            warp_specialized_factor = 2;

        } else if( host_workspace->xmma_params.specialize == xmma::CONFIG_1DMA_2MATH ) {
            xmma_implicit_gemm_warp_specialized_kernel =
                &xmma::gemm::xmma_implicit_gemm_specialize_2math_1dma_arrive_wait_kernel<
                    Kernel_traits>;
            warp_specialized_factor = 3;
        }

        if( host_workspace->smem_size > 48 * 1024 ) {
            if( host_workspace->xmma_params.ampere && host_workspace->smem_size > 164 * 1024 ||
                host_workspace->xmma_params.hopper && host_workspace->smem_size > 164 * 1024 ||
                !host_workspace->xmma_params.ampere && !host_workspace->xmma_params.hopper &&
                    host_workspace->smem_size > 64 * 1024 ) {
                return xmma::Error::ERROR_LWDA_RUNTIME;
            }
            XMMA_LWDA_CALL( lwdaFuncSetAttribute( xmma_implicit_gemm_warp_specialized_kernel,
                                                  lwdaFuncAttributeMaxDynamicSharedMemorySize,
                                                  host_workspace->smem_size ) );
            XMMA_LWDA_CALL( lwdaFuncSetAttribute( xmma_implicit_gemm_warp_specialized_kernel,
                                                  lwdaFuncAttributePreferredSharedMemoryCarveout,
                                                  100 ) );
        }
        xmma_implicit_gemm_warp_specialized_kernel<<<host_workspace->grid,
                                                     Kernel_traits::Cta_tile::THREADS_PER_CTA *
                                                         warp_specialized_factor,
                                                     host_workspace->smem_size,
                                                     lwda_stream>>>( host_workspace->xmma_params );
        XMMA_LWDA_CALL( lwdaGetLastError() );

        // If we need two kernels to run split-k launch the second grid.
        if( host_workspace->xmma_params.split_k.kernels == 2 ) {
            host_workspace->xmma_params.split_k.kernels = 1;
            xmma::gemm::split_k_kernel<Kernel_traits>
                <<<host_workspace->split_k_grid,
                   Kernel_traits::Cta_tile::THREADS_PER_CTA,
                   host_workspace->epilogue_size_in_bytes,
                   lwda_stream>>>( host_workspace->xmma_params );
            host_workspace->xmma_params.split_k.kernels = 2;
        }
        XMMA_LWDA_CALL( lwdaGetLastError() );

        return xmma::Error::SUCCESS;
    }

    static lwdaError_t get_func_attributes( lwdaFuncAttributes *attr ) {
        void ( *xmma_implicit_gemm_warp_specialized_kernel )(
            typename Kernel_traits::Params params ) = NULL;
        int warp_specialized_factor;

        xmma_implicit_gemm_warp_specialized_kernel =
            &xmma::gemm::xmma_implicit_gemm_specialize_1math_1dma_arrive_wait_kernel<Kernel_traits>;
        warp_specialized_factor = 2;

        lwdaError_t lwda_status =
            lwdaFuncGetAttributes( attr, xmma_implicit_gemm_warp_specialized_kernel );
        attr->maxDynamicSharedSizeBytes = Kernel_traits::get_dynamic_shared_memory_size_per_cta();
        attr->maxThreadsPerBlock = Kernel_traits::get_threads_per_cta() * warp_specialized_factor;
        return lwda_status;
    }
};  // end warp specialized device kernel

template <typename Kernel_traits>
xmma::Error run_kernel( const void *a_data,
                        const void *b_data,
                        const void *c_data,
                        const void *bias_data,
                        const void* alpha_data,
                        const void* beta_data,
                        void *d_data,
                        void *host_ptr,
                        void *device_ptr,
                        Runtime_params<Kernel_traits> &runtime_params,
                        lwdaStream_t &lwda_stream ) {
    using Gpu_arch = typename Kernel_traits::Traits::Gpu_arch;

    auto host_workspace = static_cast<xmma::Host_workspace<Kernel_traits> *>( host_ptr );
    auto &params = host_workspace->xmma_params;

    params.a_gmem = a_data;
    params.b_gmem = b_data;
    params.c_gmem = c_data;
    params.bias_gmem = bias_data;
    params.alpha_gmem = alpha_data;
    params.beta_gmem = beta_data;
    params.d_gmem = d_data;

    // Initialize the L2 descriptors
    params.mem_descriptors.descriptor_a = ( (uint64_t)runtime_params.descriptor_a << 32 );
    params.mem_descriptors.descriptor_b = ( (uint64_t)runtime_params.descriptor_b << 32 );
    params.mem_descriptors.descriptor_c =
        ( (uint64_t)runtime_params.descriptor_c1 << 32 ) + (uint64_t)runtime_params.descriptor_c0;
    params.mem_descriptors.descriptor_d =
        ( (uint64_t)runtime_params.descriptor_d1 << 32 ) + (uint64_t)runtime_params.descriptor_d0;
    float *geluScale = &runtime_params.gelu_scale;
    params.runtime_params.runtime_param0 = *reinterpret_cast<int32_t *>( geluScale );

    // Setup split-k buffers
    params.split_k.set_base_ptr( device_ptr );
    XMMA_LWDA_CALL( params.split_k.clear_buffers( device_ptr, lwda_stream ) );

    // Non-deterministic  reduction using atomics, split-k-mode == 3
    // zero-init output C if beta == 0
    if( params.split_k.kernels == 3 && !params.with_residual ) {
        size_t c_size = params.n * params.ldc * Kernel_traits::BITS_PER_ELEMENT_C / 8;
        XMMA_LWDA_CALL( lwdaMemsetAsync( (void *)params.c_gmem, 0, c_size, lwda_stream ) );
    }

    auto *kernel = Kernel_traits::kernel_ptr( params );
    auto *split_k_kernel = Kernel_traits::split_k_kernel_ptr();

    XMMA_LWDA_CALL( xmma::set_func_attributes<Gpu_arch>( kernel, host_workspace->smem_size ) );
    ( *kernel )<<<host_workspace->grid,
                  Kernel_traits::threads_per_cta( params ),
                  host_workspace->smem_size,
                  lwda_stream>>>( params );
    XMMA_LWDA_CALL( lwdaGetLastError() );

    // If we need two kernels to run split-k launch the second grid.
    if( params.split_k.kernels == 2 ) {
        params.split_k.kernels = 1;
        XMMA_LWDA_CALL( xmma::set_func_attributes<Gpu_arch>(
            split_k_kernel, host_workspace->epilogue_size_in_bytes ) );
        ( *split_k_kernel )<<<host_workspace->split_k_grid,
                              Kernel_traits::Cta_tile::THREADS_PER_CTA,
                              host_workspace->epilogue_size_in_bytes,
                              lwda_stream>>>( params );
        params.split_k.kernels = 2;
        XMMA_LWDA_CALL( lwdaGetLastError() );
    }

    return xmma::Error::SUCCESS;
}


template <typename Kernel_traits>
xmma::Error get_func_attributes( lwdaFuncAttributes* attr ) {
    // FIXME: warp specialized kernel need to be refactored to work with this API
    if (lwdaFuncGetAttributes( attr, Kernel_traits::kernel_ptr() ) == lwdaSuccess) {
        attr->maxDynamicSharedSizeBytes = Kernel_traits::dynamic_smem_size_per_cta();
        attr->maxThreadsPerBlock = Kernel_traits::threads_per_cta();
    }

    return xmma::Error::SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef USE_GMMA
// warp specialized GMMA 
// both A and B are from SMEM
template <typename Kernel_traits>
xmma::Error run_kernel_specialized_gmma( const void* a_data,
                             const void* b_data,
                             const void* c_data,
                             void* d_data,
                             void* host_ptr,
                             void* device_ptr,
                             Runtime_params<Kernel_traits>& runtime_params,
                             lwdaStream_t& lwda_stream ) {
    auto host_workspace = static_cast<xmma::Host_workspace<Kernel_traits>*>( host_ptr );
    auto& params = host_workspace->xmma_params;

    if( Kernel_traits::Gmem_tile_a::COPY_ENGINE == xmma::Copy_engine::CE_UTMALDG ) {
        uint32_t tensor_size_a[2];
        uint32_t box_size_a[2];
        swizzle_t swizzle_a;

        constexpr int BYTES_PER_ELEMENT_A = Kernel_traits::BITS_PER_ELEMENT_A / 8;

        if( Kernel_traits::Gmem_tile_a::GMMA_DESC_MODE ==
            xmma::Gmma_descriptor_mode::SWIZZLE_128B ) {
            swizzle_a = SWIZZLE_128B;
        }
        if( Kernel_traits::Gmem_tile_a::GMMA_DESC_MODE ==
            xmma::Gmma_descriptor_mode::SWIZZLE_64B ) {
            swizzle_a = SWIZZLE_64B;
        }
        if( Kernel_traits::Gmem_tile_a::GMMA_DESC_MODE ==
            xmma::Gmma_descriptor_mode::SWIZZLE_NONE ) {
            swizzle_a = SWIZZLE_DISABLED;
        }

        // In the case of multicasting, we split loading each tile by a certain factor.
        // This factor depends on the number of CTAs in the CGA what will also fill the tile
        // Eg. for a 4x2 CGA, Loading A-Tile will be split among 2 CTAs (#cluster columns) and 
        // Loading B tile will be split among 4 CTAs (#cluster rows).
        constexpr int USE_TMA_MULTICAST = Kernel_traits::Gmem_tile_a::Tile_traits::USE_TMA_MULTICAST;
        int multi_cast_split_factor = USE_TMA_MULTICAST ? params.cluster_width : 1;

        if( host_workspace->xmma_params.ta ) {
            tensor_size_a[0] = host_workspace->xmma_params.k;
            tensor_size_a[1] = host_workspace->xmma_params.m;

            box_size_a[0] = xmma::tma::kTileBlockRow( Kernel_traits::Cta_tile::K,
                                                      Kernel_traits::BITS_PER_ELEMENT_A );
            box_size_a[1] = xmma::tma::kTileBlockCol( Kernel_traits::Cta_tile::M )
                                /  multi_cast_split_factor;

            // smem_size_a = lwlwda::experimental::fk::kTileSharedMemoryBytes(
            // Kernel_traits::Cta_tile::K, Kernel_traits::Cta_tile::M, BYTES_PER_ELEMENT_A );
        } else {
            tensor_size_a[0] = host_workspace->xmma_params.m;
            tensor_size_a[1] = host_workspace->xmma_params.k;

            box_size_a[0] = xmma::tma::kTileBlockRow( Kernel_traits::Cta_tile::M,
                                                      Kernel_traits::BITS_PER_ELEMENT_A );
            box_size_a[1] = xmma::tma::kTileBlockCol( Kernel_traits::Cta_tile::K )
                                /  multi_cast_split_factor;

            // swizzle_a = xmma::tma::kTileBlockRowSwizzle( Kernel_traits::Cta_tile::M,
            //                                             Kernel_traits::BITS_PER_ELEMENT_A );

            // smem_size_a = lwlwda::experimental::fk::kTileSharedMemoryBytes(
            // Kernel_traits::Cta_tile::M, Kernel_traits::Cta_tile::K, BYTES_PER_ELEMENT_A );
        }

        uint64_t tensor_stride_a[1];
        tensor_stride_a[0] = tensor_size_a[0] * BYTES_PER_ELEMENT_A;
        uint32_t traversal_stride_a[2] = { 1, 1 };

        using namespace xmma::hopper::emu;

        lwdaTmaDescv2 a_desc, *a_desc_d;

        XMMA_LWDA_CALL( lwdaSetTmaTileDescriptorv2(
            &a_desc,
            a_data,
            2,
            xmma::hopper::emu::get_data_type_tma_desc<typename Kernel_traits::Traits::A_type>(),
            INTERLEAVE_DISABLED,
            swizzle_a,
            PROMOTION_DISABLED,
            tensor_size_a,
            tensor_stride_a,
            traversal_stride_a,
            box_size_a,
            0,
            0 ) );

        XMMA_LWDA_CALL(lwdaMemcpyToSymbolAsync(xmma::gemm::warp_specialized_kernel::desc_k, &a_desc, 64, 0, lwdaMemcpyHostToDevice, lwda_stream));
        XMMA_LWDA_CALL(lwdaGetSymbolAddress((void**)&a_desc_d, xmma::gemm::warp_specialized_kernel::desc_k));

        host_workspace->xmma_params.a_desc = a_desc_d;
    }

    if( Kernel_traits::Gmem_tile_b::COPY_ENGINE == xmma::Copy_engine::CE_UTMALDG ) {
        uint32_t tensor_size_b[2];
        uint32_t box_size_b[2];

        swizzle_t swizzle_b;

        constexpr int BYTES_PER_ELEMENT_B = Kernel_traits::BITS_PER_ELEMENT_B / 8;

        if( Kernel_traits::Gmem_tile_b::GMMA_DESC_MODE ==
            xmma::Gmma_descriptor_mode::SWIZZLE_128B ) {
            swizzle_b = SWIZZLE_128B;
        }
        if( Kernel_traits::Gmem_tile_b::GMMA_DESC_MODE ==
            xmma::Gmma_descriptor_mode::SWIZZLE_64B ) {
            swizzle_b = SWIZZLE_64B;
        }
        if( Kernel_traits::Gmem_tile_b::GMMA_DESC_MODE ==
            xmma::Gmma_descriptor_mode::SWIZZLE_NONE ) {
            swizzle_b = SWIZZLE_DISABLED;
        }

        // In the case of multicasting, we split loading each tile by a certain factor.
        // This factor depends on the number of CTAs in the CGA what will also fill the tile
        // Eg. for a 4x2 CGA, Loading A-Tile will be split among 2 CTAs (#cluster columns) and 
        // Loading B tile will be split among 4 CTAs (#cluster rows).
        constexpr int USE_TMA_MULTICAST = Kernel_traits::Gmem_tile_b::Tile_traits::USE_TMA_MULTICAST;
        int multi_cast_split_factor = USE_TMA_MULTICAST ? params.cluster_height : 1;

        if( host_workspace->xmma_params.tb ) {
            tensor_size_b[0] = host_workspace->xmma_params.n;
            tensor_size_b[1] = host_workspace->xmma_params.k;

            box_size_b[0] = xmma::tma::kTileBlockRow( Kernel_traits::Cta_tile::N,
                                                      Kernel_traits::BITS_PER_ELEMENT_B );
            box_size_b[1] = xmma::tma::kTileBlockCol( Kernel_traits::Cta_tile::K )
                                /  multi_cast_split_factor;

            // swizzle_b = xmma::tma::kTileBlockRowSwizzle( Kernel_traits::Cta_tile::N,
            //                                             Kernel_traits::BITS_PER_ELEMENT_B );

            // smem_size_b = lwlwda::experimental::fk::kTileSharedMemoryBytes(
            // Kernel_traits::Cta_tile::N, Kernel_traits::Cta_tile::K, BYTES_PER_ELEMENT_B );

        } else {
            tensor_size_b[0] = host_workspace->xmma_params.k;
            tensor_size_b[1] = host_workspace->xmma_params.n;

            box_size_b[0] = xmma::tma::kTileBlockRow( Kernel_traits::Cta_tile::K,
                                                      Kernel_traits::BITS_PER_ELEMENT_B );
            box_size_b[1] = xmma::tma::kTileBlockCol( Kernel_traits::Cta_tile::N )
                                /  multi_cast_split_factor;

            // swizzle_b = SWIZZLE_128B;

            // smem_size_b = lwlwda::experimental::fk::kTileSharedMemoryBytes(
            // Kernel_traits::Cta_tile::K, Kernel_traits::Cta_tile::N, BYTES_PER_ELEMENT_B );
        }

        uint64_t tensor_stride_b[1];
        tensor_stride_b[0] = tensor_size_b[0] * BYTES_PER_ELEMENT_B;
        uint32_t traversal_stride_b[2] = { 1, 1 };

        using namespace xmma::hopper::emu;

        lwdaTmaDescv2 b_desc, *b_desc_d;

        XMMA_LWDA_CALL( lwdaSetTmaTileDescriptorv2(
            &b_desc,
            b_data,
            2,
            xmma::hopper::emu::get_data_type_tma_desc<typename Kernel_traits::Traits::B_type>(),
            INTERLEAVE_DISABLED,
            swizzle_b,
            PROMOTION_DISABLED,
            tensor_size_b,
            tensor_stride_b,
            traversal_stride_b,
            box_size_b,
            0,
            0 ) );

        XMMA_LWDA_CALL(lwdaMemcpyToSymbolAsync(xmma::gemm::warp_specialized_kernel::desc_k, &b_desc, 64, 64, lwdaMemcpyHostToDevice, lwda_stream));
        XMMA_LWDA_CALL(lwdaGetSymbolAddress((void**)&b_desc_d, xmma::gemm::warp_specialized_kernel::desc_k));

        host_workspace->xmma_params.b_desc = b_desc_d + 1;
    }

    if( Kernel_traits::Gmem_tile_a::COPY_ENGINE != xmma::Copy_engine::CE_UTMALDG ) {
        params.a_gmem = a_data;
    }
    if( Kernel_traits::Gmem_tile_b::COPY_ENGINE != xmma::Copy_engine::CE_UTMALDG ) {
        params.b_gmem = b_data;
    }

    params.c_gmem = c_data;
    params.d_gmem = d_data;

    // Initialize the L2 descriptors
    params.mem_descriptors.descriptor_a = ( (uint64_t)runtime_params.descriptor_a << 32 );
    params.mem_descriptors.descriptor_b = ( (uint64_t)runtime_params.descriptor_b << 32 );
    params.mem_descriptors.descriptor_c =
        ( (uint64_t)runtime_params.descriptor_c1 << 32 ) + (uint64_t)runtime_params.descriptor_c0;
    params.mem_descriptors.descriptor_d =
        ( (uint64_t)runtime_params.descriptor_d1 << 32 ) + (uint64_t)runtime_params.descriptor_d0;

    // Setup split-k buffers
    params.split_k.set_base_ptr( device_ptr );
    XMMA_LWDA_CALL(params.split_k.clear_buffers( device_ptr, lwda_stream ));

    // printf("smem_size = %i, host_workspace->xmma_params.hopper = %i\n",
    // host_workspace->smem_size, host_workspace->xmma_params.hopper);
    // Note for Hopper we have increased the smem budget to 228 KB
    // both operands are from SMEM
    if( host_workspace->smem_size > 48 * 1024 ) {
        if(  params.hopper  && host_workspace->smem_size > 228 * 1024 ||
            ! (params.ampere || params.hopper ) && host_workspace->smem_size > 64 * 1024 ) {
            return xmma::Error::ERROR_LWDA_RUNTIME;
        }
        XMMA_LWDA_CALL( lwdaFuncSetAttribute( xmma::gemm::warp_specialized_kernel_gmma<Kernel_traits>,
                                              lwdaFuncAttributeMaxDynamicSharedMemorySize,
                                              host_workspace->smem_size ) );
        XMMA_LWDA_CALL( lwdaFuncSetAttribute( xmma::gemm::warp_specialized_kernel_gmma<Kernel_traits>,
                                              lwdaFuncAttributePreferredSharedMemoryCarveout,
                                              100 ) );
    }

    //////////////////////////////////
    // CGA Launch (only for hopper) //
    //////////////////////////////////
    bool launch_kernel = true;
    if( std::is_same< typename Kernel_traits::Gpu_arch, Hopper >::value ) {

        dim3 cluster_dims {params.cluster_width, params.cluster_height, 1};
        auto cga_launcher = xmma::create_cga_launcher();
        auto err = cga_launcher.initialize( 
                        (const void*) warp_specialized_kernel_gmma<Kernel_traits>,
                        host_workspace->grid,
                        cluster_dims );

        if( err != xmma::Error::SUCCESS ){
            printf("ERROR : Unable to launch GPC_CGA_GRID, check grid and cluster dims\n");
            printf("Grid (x,y,z) = (%d, %d, %d) ; Cluster (x,y,z) = (%d, %d, %d)\n",
                    host_workspace->grid.x, host_workspace->grid.y, host_workspace->grid.z,
                    params.cluster_width, params.cluster_height, 1);

            launch_kernel = false;
        } else {
            printf("Launching GPC_CGA_GRID GridDims = (%d, %d,%d), And ClusterDims = (%d, %d, %d)\n", 
                    host_workspace->grid.x, host_workspace->grid.y, host_workspace->grid.z, 
                    params.cluster_width, params.cluster_height, 1);
        }
    }

    if( launch_kernel ) {

    xmma::gemm::warp_specialized_kernel_gmma<Kernel_traits><<<host_workspace->grid,
    // FIXME : need to refactor Cta_tile::THREADS_PER_CTA to Cta_tile::THREADS_PER_WARPGROUP
                                             3 * Kernel_traits::Cta_tile::THREADS_PER_CTA,
                                             host_workspace->smem_size,
                                             lwda_stream>>>( params );
    }

    XMMA_LWDA_CALL( lwdaGetLastError() );

    //// If we need two kernels to run split-k launch the second grid.
    // if( host_workspace->xmma_params.split_k.kernels == 2 ) {
    //    host_workspace->xmma_params.split_k.kernels = 1;
    //    host_workspace->grid.z = Kernel_traits::Xmma_tile::XMMAS_M;
    //    xmma::gemm::split_k_kernel<Kernel_traits><<<
    //        host_workspace->grid,
    //        Kernel_traits::Cta_tile::THREADS_PER_CTA,
    //        host_workspace->epilogue_size_in_bytes,
    //        lwda_stream>>>(host_workspace->xmma_params);
    //    host_workspace->xmma_params.split_k.kernels = 2;
    //}
    // XMMA_LWDA_CALL(lwdaGetLastError());

    return xmma::Error::SUCCESS;
}
#endif

///////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef USE_GMMA
// both A and B are from SMEM
template <typename Kernel_traits>
xmma::Error run_kernel_gmma( const void* a_data,
                             const void* b_data,
                             const void* c_data,
                             void* d_data,
                             void* host_ptr,
                             void* device_ptr,
                             Runtime_params<Kernel_traits>& runtime_params,
                             lwdaStream_t& lwda_stream ) {
    auto host_workspace = static_cast<xmma::Host_workspace<Kernel_traits>*>( host_ptr );
    auto& params = host_workspace->xmma_params;

    params.a_gmem = a_data;
    params.b_gmem = b_data;
    params.c_gmem = c_data;
    params.d_gmem = d_data;

    // Initialize the L2 descriptors
    params.mem_descriptors.descriptor_a = ( (uint64_t)runtime_params.descriptor_a << 32 );
    params.mem_descriptors.descriptor_b = ( (uint64_t)runtime_params.descriptor_b << 32 );
    params.mem_descriptors.descriptor_c =
        ( (uint64_t)runtime_params.descriptor_c1 << 32 ) + (uint64_t)runtime_params.descriptor_c0;
    params.mem_descriptors.descriptor_d =
        ( (uint64_t)runtime_params.descriptor_d1 << 32 ) + (uint64_t)runtime_params.descriptor_d0;

    // Setup split-k buffers
    params.split_k.set_base_ptr( device_ptr );
    XMMA_LWDA_CALL(params.split_k.clear_buffers( device_ptr, lwda_stream ));

    // printf("smem_size = %i, host_workspace->xmma_params.hopper = %i\n",
    // host_workspace->smem_size, host_workspace->xmma_params.hopper);
    // Note for Hopper we have increased the smem budget to 228 KB
    // both operands are from SMEM
    if( host_workspace->smem_size > 48 * 1024 ) {
        if( ( params.ampere || params.hopper ) && host_workspace->smem_size > 228 * 1024 ||
            !( params.ampere || params.hopper ) && host_workspace->smem_size > 64 * 1024 ) {
            return xmma::Error::ERROR_LWDA_RUNTIME;
        }
        XMMA_LWDA_CALL( lwdaFuncSetAttribute( xmma::gemm::kernel_gmma<Kernel_traits>,
                                              lwdaFuncAttributeMaxDynamicSharedMemorySize,
                                              host_workspace->smem_size ) );
        XMMA_LWDA_CALL( lwdaFuncSetAttribute( xmma::gemm::kernel_gmma<Kernel_traits>,
                                              lwdaFuncAttributePreferredSharedMemoryCarveout,
                                              100 ) );
    }

    xmma::gemm::kernel_gmma<Kernel_traits><<<host_workspace->grid,
                                             Kernel_traits::Cta_tile::THREADS_PER_CTA,
                                             host_workspace->smem_size,
                                             lwda_stream>>>( params );
    XMMA_LWDA_CALL( lwdaGetLastError() );

    //// If we need two kernels to run split-k launch the second grid.
    // if( host_workspace->xmma_params.split_k.kernels == 2 ) {
    //    host_workspace->xmma_params.split_k.kernels = 1;
    //    host_workspace->grid.z = Kernel_traits::Xmma_tile::XMMAS_M;
    //    xmma::gemm::split_k_kernel<Kernel_traits><<<
    //        host_workspace->grid,
    //        Kernel_traits::Cta_tile::THREADS_PER_CTA,
    //        host_workspace->epilogue_size_in_bytes,
    //        lwda_stream>>>(host_workspace->xmma_params);
    //    host_workspace->xmma_params.split_k.kernels = 2;
    //}
    // XMMA_LWDA_CALL(lwdaGetLastError());

    return xmma::Error::SUCCESS;
}

// A is from RF, B is from SMEM
template<typename Kernel_traits>
xmma::Error run_kernel_gmma_arf(const void *a_data,
                           const void *b_data,
                           const void *c_data,
                           void *d_data,
                           void *host_ptr,
                           void *device_ptr,
                           Runtime_params<Kernel_traits> &runtime_params,
                           lwdaStream_t &lwda_stream) {
  auto host_workspace = static_cast<xmma::Host_workspace<Kernel_traits> *>(host_ptr);
  auto &params = host_workspace->xmma_params;

  params.a_gmem = a_data;
  params.b_gmem = b_data;
  params.c_gmem = c_data;
  params.d_gmem = d_data;

  // Initialize the L2 descriptors
  params.mem_descriptors.descriptor_a = ((uint64_t)runtime_params.descriptor_a << 32);
  params.mem_descriptors.descriptor_b = ((uint64_t)runtime_params.descriptor_b << 32);
  params.mem_descriptors.descriptor_c = ((uint64_t)runtime_params.descriptor_c1 << 32) + (uint64_t)runtime_params.descriptor_c0;
  params.mem_descriptors.descriptor_d = ((uint64_t)runtime_params.descriptor_d1 << 32) + (uint64_t)runtime_params.descriptor_d0;

  // Setup split-k buffers
  params.split_k.set_base_ptr(device_ptr);
  XMMA_LWDA_CALL(params.split_k.clear_buffers( device_ptr, lwda_stream ));

  //printf("smem_size = %i, host_workspace->xmma_params.hopper = %i\n", host_workspace->smem_size, host_workspace->xmma_params.hopper);
  // Note for Hopper we have increased the smem budget to 228 KB
  // both operands are from SMEM
  if( host_workspace->smem_size > 48 * 1024 ) {
    if( (params.ampere || params.hopper) && host_workspace->smem_size > 228*1024 ||
       !(params.ampere || params.hopper) && host_workspace->smem_size > 64*1024) {
      return xmma::Error::ERROR_LWDA_RUNTIME;
    }
    XMMA_LWDA_CALL(lwdaFuncSetAttribute(
      xmma::gemm::kernel_gmma_arf<Kernel_traits>,
      lwdaFuncAttributeMaxDynamicSharedMemorySize,
      host_workspace->smem_size));
    XMMA_LWDA_CALL(lwdaFuncSetAttribute(
      xmma::gemm::kernel_gmma_arf<Kernel_traits>,
      lwdaFuncAttributePreferredSharedMemoryCarveout,
      100));
  }

  xmma::gemm::kernel_gmma_arf<Kernel_traits><<<
      host_workspace->grid,
      Kernel_traits::Cta_tile::THREADS_PER_CTA,
      host_workspace->smem_size,
      lwda_stream>>>(params);
  XMMA_LWDA_CALL(lwdaGetLastError());

  //// If we need two kernels to run split-k launch the second grid.
  //if( host_workspace->xmma_params.split_k.kernels == 2 ) {
  //    host_workspace->xmma_params.split_k.kernels = 1;
  //    host_workspace->grid.z = Kernel_traits::Xmma_tile::XMMAS_M;
  //    xmma::gemm::split_k_kernel<Kernel_traits><<<
  //        host_workspace->grid,
  //        Kernel_traits::Cta_tile::THREADS_PER_CTA,
  //        host_workspace->epilogue_size_in_bytes,
  //        lwda_stream>>>(host_workspace->xmma_params);
  //    host_workspace->xmma_params.split_k.kernels = 2;
  //}
  //CHECK_LWDA(lwdaGetLastError());

  return xmma::Error::SUCCESS;
}

// A is from RF, B is from SMEM, scale and bias also applied to A
template<typename Kernel_traits>
xmma::Error run_kernel_gmma_arf_bn_apply(const void *a_data,
                           const void *b_data,
                           const void *c_data,
                           void *d_data,
                           const void *scale_bias_data,
                           void *host_ptr,
                           void *device_ptr,
                           Runtime_params<Kernel_traits> &runtime_params,
                           lwdaStream_t &lwda_stream) {
  auto host_workspace = static_cast<xmma::Host_workspace<Kernel_traits> *>(host_ptr);
  auto &params = host_workspace->xmma_params;

  params.a_gmem = a_data;
  params.b_gmem = b_data;
  params.c_gmem = c_data;
  params.d_gmem = d_data;
  params.scale_bias_gmem = scale_bias_data;

  // Initialize the L2 descriptors
  params.mem_descriptors.descriptor_a = ((uint64_t)runtime_params.descriptor_a << 32);
  params.mem_descriptors.descriptor_b = ((uint64_t)runtime_params.descriptor_b << 32);
  params.mem_descriptors.descriptor_c = ((uint64_t)runtime_params.descriptor_c1 << 32) + (uint64_t)runtime_params.descriptor_c0;
  params.mem_descriptors.descriptor_d = ((uint64_t)runtime_params.descriptor_d1 << 32) + (uint64_t)runtime_params.descriptor_d0;

  // Setup split-k buffers
  params.split_k.set_base_ptr(device_ptr);
  XMMA_LWDA_CALL(params.split_k.clear_buffers( device_ptr, lwda_stream ));

  // printf("smem_size = %i, host_workspace->xmma_params.hopper = %i\n", host_workspace->smem_size, host_workspace->xmma_params.hopper);
  // Note for Hopper we have increased the smem budget to 228 KB
  // both operands are from SMEM
  if( host_workspace->smem_size > 48 * 1024 ) {
    if( (params.ampere || params.hopper) && host_workspace->smem_size > 228*1024 ||
       !(params.ampere || params.hopper) && host_workspace->smem_size > 64*1024) {
      return xmma::Error::ERROR_LWDA_RUNTIME;
    }
    XMMA_LWDA_CALL(lwdaFuncSetAttribute(
      xmma::gemm::kernel_gmma_arf<Kernel_traits>,
      lwdaFuncAttributeMaxDynamicSharedMemorySize,
      host_workspace->smem_size));
    XMMA_LWDA_CALL(lwdaFuncSetAttribute(
      xmma::gemm::kernel_gmma_arf<Kernel_traits>,
      lwdaFuncAttributePreferredSharedMemoryCarveout,
      100));
  }

  xmma::gemm::kernel_gmma_arf<Kernel_traits><<<
      host_workspace->grid,
      Kernel_traits::Cta_tile::THREADS_PER_CTA,
      host_workspace->smem_size,
      lwda_stream>>>(params);
  XMMA_LWDA_CALL(lwdaGetLastError());

  //// If we need two kernels to run split-k launch the second grid.
  //if( host_workspace->xmma_params.split_k.kernels == 2 ) {
  //    host_workspace->xmma_params.split_k.kernels = 1;
  //    host_workspace->grid.z = Kernel_traits::Xmma_tile::XMMAS_M;
  //    xmma::gemm::split_k_kernel<Kernel_traits><<<
  //        host_workspace->grid,
  //        Kernel_traits::Cta_tile::THREADS_PER_CTA,
  //        host_workspace->epilogue_size_in_bytes,
  //        lwda_stream>>>(host_workspace->xmma_params);
  //    host_workspace->xmma_params.split_k.kernels = 2;
  //}
  //CHECK_LWDA(lwdaGetLastError());

  return xmma::Error::SUCCESS;
}


#endif  // USE_GMMA

///////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace gemm
}  // namespace xmma
