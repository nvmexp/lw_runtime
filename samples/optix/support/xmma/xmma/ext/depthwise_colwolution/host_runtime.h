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

#ifndef _XMMA_EXT_DEPTHWISE_COLWOLUTION_HOST_RUNTIME_H_
#define _XMMA_EXT_DEPTHWISE_COLWOLUTION_HOST_RUNTIME_H_

#include "xmma/ext/depthwise_colwolution/fprop/kernel.h"
#include "xmma/ext/depthwise_colwolution/helper_kernel.h"
#include "xmma/ext/depthwise_colwolution/traits.h"
#include "xmma/ext/depthwise_colwolution/wgrad/kernel.h"
#include "xmma/params.h"
#include "xmma/xmma.h"

namespace xmma
{
namespace ext
{
namespace depthwise_colwolution
{

template <typename Kernel_traits> size_t get_host_workspace_size()
{
    return sizeof(xmma::Host_workspace<Kernel_traits>);
}

template <typename Kernel_traits> size_t get_device_workspace_size(const void *host_ptr)
{
    auto workspace = static_cast<const xmma::Host_workspace<Kernel_traits> *>(host_ptr);
    constexpr size_t MAXIMUM_BYTES_PER_INSTRUCTION = 16;
    return Kernel_traits::get_split_k_total_buffer_size_in_bytes(
        *const_cast<typename Kernel_traits::Params *>(&(workspace->xmma_params)))
        + (MAXIMUM_BYTES_PER_INSTRUCTION - 1);
}

template <typename Kernel_traits>
xmma::Error initialize_device_workspace(const xmma::Host_workspace<Kernel_traits> *host_workspace,
                                        void *device_workspace,
                                        lwdaStream_t stream)
{
    return xmma::Error::SUCCESS;
}

template <typename Kernel_traits> struct Device_kernel {
    public:
    static xmma::Error run(xmma::Host_workspace<Kernel_traits> *workspace,
                           lwdaStream_t &lwda_stream)
    {
        typename Kernel_traits::Params xmma_params = workspace->xmma_params;
        Helper_kernel_param helper_kernel_param;
        if (Kernel_traits::Operation == xmma::Operation_type::FPROP ||
            Kernel_traits::Operation == xmma::Operation_type::DGRAD) {
            helper_kernel_param.src = xmma_params.tmp_gmem_trsg;
            helper_kernel_param.dst = xmma_params.gmem[Tensor_type::B];
        } else if (Kernel_traits::Operation == xmma::Operation_type::WGRAD) {
            helper_kernel_param.src = xmma_params.tmp_gmem_trsg;
            helper_kernel_param.dst = xmma_params.gmem[Tensor_type::C];
        }
        helper_kernel_param.m = static_cast<int64_t>(xmma_params.g);
        helper_kernel_param.n = static_cast<int64_t>(xmma_params.t * xmma_params.r * xmma_params.s);

        dim3 helper_kernel_grid;
        helper_kernel_grid.x = static_cast<int32_t>((helper_kernel_param.m * helper_kernel_param.n +
                                                     Helper_kernel_param::THREADS_PER_CTA - 1) /
                                                    Helper_kernel_param::THREADS_PER_CTA);
        helper_kernel_grid.y = 1;
        helper_kernel_grid.z = 1;

        if (!(Kernel_traits::Operation == xmma::Operation_type::WGRAD && xmma_params.beta == 0)) {
            helper_kernel<<<helper_kernel_grid, Helper_kernel_param::THREADS_PER_CTA, 0, lwda_stream>>>(
                helper_kernel_param);
            XMMA_LWDA_CALL(lwdaGetLastError());
        }

        XMMA_LWDA_CALL(lwdaMemsetAsync(workspace->xmma_params.split_k_gmem_buffer_counter,
                                       0,
                                       workspace->xmma_params.split_k_counter_size_in_bytes,
                                       lwda_stream));

        XMMA_LWDA_CALL(lwdaFuncSetAttribute(Kernel_traits::kernel_ptr(),
                                            lwdaFuncAttributeMaxDynamicSharedMemorySize,
                                            workspace->smem_size));
        XMMA_LWDA_CALL(lwdaFuncSetAttribute(
            Kernel_traits::kernel_ptr(), lwdaFuncAttributePreferredSharedMemoryCarveout, 100));

        Kernel_traits::kernel_ptr()<<<workspace->grid,
                                      Kernel_traits::threads_per_cta(),
                                      workspace->smem_size,
                                      lwda_stream>>>(workspace->xmma_params);
        XMMA_LWDA_CALL(lwdaGetLastError());

        if (Kernel_traits::Operation == xmma::Operation_type::WGRAD) {
            Helper_kernel_param helper_kernel_param;
            helper_kernel_param.src = xmma_params.gmem[Tensor_type::D];
            helper_kernel_param.dst = xmma_params.final_gmem_trsg;

            helper_kernel_param.m =
                static_cast<int64_t>(xmma_params.t * xmma_params.r * xmma_params.s);
            helper_kernel_param.n = static_cast<int64_t>(xmma_params.g);

            dim3 helper_kernel_grid;
            helper_kernel_grid.x =
                static_cast<int32_t>((helper_kernel_param.m * helper_kernel_param.n +
                                      Helper_kernel_param::THREADS_PER_CTA - 1) /
                                     Helper_kernel_param::THREADS_PER_CTA);
            helper_kernel_grid.y = 1;
            helper_kernel_grid.z = 1;

            helper_kernel<<<helper_kernel_grid,
                            Helper_kernel_param::THREADS_PER_CTA,
                            0,
                            lwda_stream>>>(helper_kernel_param);
            XMMA_LWDA_CALL(lwdaGetLastError());
        }

        return xmma::Error::SUCCESS;
    }

    static lwdaError_t get_func_attributes(lwdaFuncAttributes *attr)
    {
        lwdaError_t lwda_status = lwdaFuncGetAttributes(attr, Kernel_traits::kernel_ptr());
        attr->maxDynamicSharedSizeBytes = Kernel_traits::dynamic_smem_size_per_cta();
        attr->maxThreadsPerBlock = Kernel_traits::threads_per_cta();
        return lwda_status;
    }
};

template <typename Kernel_traits>
xmma::Error run_kernel(const void *x_data,
                       const void *y_data,
                       const void *w_data,
                       const void *res_data,
                       const void *bias_data,
                       const void *alpha_data,
                       const void *beta_data,
                       void *host_ptr,
                       void *device_ptr,
                       lwdaStream_t &lwda_stream)
{
    auto host_workspace = static_cast<xmma::Host_workspace<Kernel_traits> *>(host_ptr);
    auto &params = host_workspace->xmma_params;

    constexpr int64_t MAXIMUM_BYTES_PER_INSTRUCTION = 16;
    int64_t bytes_to_add = MAXIMUM_BYTES_PER_INSTRUCTION 
        - (reinterpret_cast<intptr_t>(device_ptr) & (MAXIMUM_BYTES_PER_INSTRUCTION - 1));
    if (bytes_to_add == MAXIMUM_BYTES_PER_INSTRUCTION) {
        bytes_to_add = 0;
    }
    void *aligned_device_ptr = static_cast<void *>(static_cast<char *>(device_ptr) + bytes_to_add);
    if (Kernel_traits::Operation == xmma::Operation_type::FPROP) {
        params.set_a_gmem(const_cast<void *>(x_data));
        params.set_tmp_gmem_trsg(const_cast<void *>(w_data));
        params.set_b_gmem(aligned_device_ptr);
        params.set_c_gmem(const_cast<void *>(res_data));
        params.set_d_gmem(const_cast<void *>(y_data));
    } else if (Kernel_traits::Operation == xmma::Operation_type::DGRAD) {
        params.set_a_gmem(const_cast<void *>(y_data));
        params.set_tmp_gmem_trsg(const_cast<void *>(w_data));
        params.set_b_gmem(aligned_device_ptr);
        params.set_c_gmem(const_cast<void *>(res_data));
        params.set_d_gmem(const_cast<void *>(x_data));
    } else if (Kernel_traits::Operation == xmma::Operation_type::WGRAD) {
        params.set_a_gmem(const_cast<void *>(x_data));
        params.set_b_gmem(const_cast<void *>(w_data));
        params.set_tmp_gmem_trsg(const_cast<void *>(res_data));
        params.set_final_gmem_trsg(const_cast<void *>(y_data));
        params.set_c_gmem(aligned_device_ptr);
        params.set_d_gmem(aligned_device_ptr);
    }
    void *split_k_start_ptr = static_cast<void *>(static_cast<uint8_t *>(aligned_device_ptr) +
                                                  Kernel_traits::get_split_k_start_offset(params));
    params.set_split_k_gmem(split_k_start_ptr);

    dim3 grid;

    Kernel_traits::compute_grid_size(grid, params);

    int32_t single_split_k_buffer_size_in_bytes =
        Kernel_traits::get_single_split_k_buffer_size_in_bytes(grid);
    params.set_single_split_k_buffer_size_in_bytes(single_split_k_buffer_size_in_bytes);
    int32_t split_k_gmem_size_in_bytes =
        Kernel_traits::get_split_k_gmem_size_in_bytes(grid, params.split_k_buffers);
    params.set_split_k_gmem_buffer_counter(split_k_gmem_size_in_bytes);
    int32_t split_k_buffer_counter_size_in_bytes =
        Kernel_traits::get_split_k_buffer_counter_size_in_bytes(grid, params.split_k_buffers);
    params.set_split_k_gmem_final_counter(split_k_buffer_counter_size_in_bytes);
    int32_t split_k_final_counter_size_in_bytes =
        Kernel_traits::get_split_k_final_counter_size_in_bytes(grid);
    params.set_split_k_counter_size_in_bytes(split_k_buffer_counter_size_in_bytes +
                                             split_k_final_counter_size_in_bytes);
    Device_kernel<Kernel_traits>::run(host_workspace, lwda_stream);

    return xmma::Error::SUCCESS;
}

} // namespace depthwise_colwolution
} // namespace ext
} // namespace xmma

#endif
