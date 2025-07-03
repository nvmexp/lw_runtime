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
#include <xmma/ext/sparse/spgemm/sphmma/kernel.h>
#include <xmma/ext/sparse/spgemm/spimma/kernel.h>
#include <xmma/ext/sparse/spgemm/utils.h>

///////////////////////////////////////////////////////////////////////////////////////////////////

namespace xmma {
namespace ext {
namespace gemm {
#ifdef LINK
inline void lwCheckErrorsFunc(LWresult err, const char* file, int line)
{
    if (err != LWDA_SUCCESS){
        const char* pStr = nullptr;
        cask::lwGetErrorName(err, &pStr);
        printf("%s at %s:%d\n", pStr, file, line);
        throw std::runtime_error(pStr);
        }
}
#define lwCheckErrors(expr) do{ lwCheckErrorsFunc((expr), __FILE__, __LINE__); }while(false)
#endif
///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Gemm_traits> struct Runtime_params {
    int32_t descriptor_a;
    int32_t descriptor_b;
    int32_t descriptor_c0;
    int32_t descriptor_c1;
    int32_t descriptor_d0;
    int32_t descriptor_d1;

    // gelu runtime scale factor
    float gelu_scale;
    // runtime fusion params
    bool isRtKernel;
    LWfunction kernel;
};

///////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Gemm_traits >
size_t get_host_workspace_size()
{
  return sizeof(xmma::Host_workspace<Gemm_traits>);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Gemm_traits>
xmma::Error initialize_host_workspace(typename Gemm_traits::Params &xmma_params, void *host_ptr)
{
  xmma::Host_workspace<Gemm_traits> *workspace =
    static_cast<xmma::Host_workspace<Gemm_traits> *>(host_ptr);

  return xmma_params.initialize(workspace);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Gemm_traits >
size_t get_device_workspace_size(const void *host_ptr)
{
  const xmma::Host_workspace<Gemm_traits> *workspace =
    static_cast<const xmma::Host_workspace<Gemm_traits> *>(host_ptr);

  size_t device_workspace_size = workspace->device_workspace_size;
  //Additional 16 bytes for alignment
  if (device_workspace_size != 0) {
      device_workspace_size += 16;
  }
  return device_workspace_size;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Gemm_traits >
xmma::Error initialize_device_workspace(const void *host_ptr,
                                            void *device_ptr,
                                            lwdaStream_t lwda_stream)
{
  return xmma::Error::SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
template <typename Gemm_traits,
          bool use_sp_imma>
struct Device_kernel {
};

///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Gemm_traits>
struct Device_kernel<Gemm_traits, false> {

  static xmma::Error run(xmma::Host_workspace<Gemm_traits> *workspace,
                             lwdaStream_t &lwda_stream) {

    if( workspace->smem_size > 48 * 1024 ) {
      if( workspace->xmma_params.ampere && workspace->smem_size > 164*1024 ||
          !workspace->xmma_params.ampere && workspace->smem_size > 64*1024) {
        return xmma::Error::ERROR_LWDA_RUNTIME;
      }
      XMMA_LWDA_CALL(lwdaFuncSetAttribute(
        xmma::ext::gemm::sparse_hmma_gemm::kernel<Gemm_traits>,
        lwdaFuncAttributeMaxDynamicSharedMemorySize,
        workspace->smem_size));
      XMMA_LWDA_CALL(lwdaFuncSetAttribute(
        xmma::ext::gemm::sparse_hmma_gemm::kernel<Gemm_traits>,
        lwdaFuncAttributePreferredSharedMemoryCarveout,
        100));
    }

    xmma::ext::gemm::sparse_hmma_gemm::kernel<Gemm_traits>
      <<<workspace->grid, Gemm_traits::Cta_tile::THREADS_PER_CTA,
        workspace->smem_size, lwda_stream>>>(workspace->xmma_params);
    XMMA_LWDA_CALL(lwdaGetLastError());

    // If we need two kernels to run split-k launch the second grid.
    if(Gemm_traits::HAS_2_KERNEL_SPLITK == true) {
      if( workspace->xmma_params.split_k.kernels == 2 ) {
        workspace->xmma_params.split_k.kernels = 1;
        // workspace->grid.z = Gemm_traits::Xmma_tile::XMMAS_N;
        workspace->split_k_grid = workspace->grid;
        workspace->split_k_grid.z = (Gemm_traits::Gmem_tile_epilogue::output_layout::COL) ?
            Gemm_traits::Xmma_tile::XMMAS_N : Gemm_traits::Xmma_tile::XMMAS_M;
        xmma::ext::gemm::sparse_hmma_gemm::split_k_kernel<Gemm_traits>
            <<<workspace->split_k_grid, Gemm_traits::Cta_tile::THREADS_PER_CTA,
            workspace->epilogue_size_in_bytes, lwda_stream>>>(workspace->xmma_params);
        workspace->xmma_params.split_k.kernels = 2;
      }
    }
    XMMA_LWDA_CALL(lwdaGetLastError());
    return xmma::Error::SUCCESS;

  }

  static xmma::Error driver_run(xmma::Host_workspace<Gemm_traits> *workspace, LWfunction kernel,
                             lwdaStream_t &lwda_stream) {
      printf("warning: only sparse imma support runtime fusion now!\n");
      return xmma::Error::SUCCESS;
  }
  static lwdaError_t get_func_attributes(lwdaFuncAttributes* attr) {
      lwdaError_t lwda_status = lwdaFuncGetAttributes(attr,
                                            xmma::ext::gemm::sparse_hmma_gemm::kernel<Gemm_traits>);
      attr->maxDynamicSharedSizeBytes =
          Gemm_traits::dynamic_smem_size_per_cta();
      attr->maxThreadsPerBlock = Gemm_traits::threads_per_cta();
      return lwda_status;
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Gemm_traits>
struct Device_kernel<Gemm_traits, true> {

  static xmma::Error run(xmma::Host_workspace<Gemm_traits> *workspace,
                             lwdaStream_t &lwda_stream) {
    if( workspace->smem_size > 48 * 1024 ) {
      if( workspace->xmma_params.ampere && workspace->smem_size > 164*1024 ||
          !workspace->xmma_params.ampere && workspace->smem_size > 64*1024) {
        return xmma::Error::ERROR_LWDA_RUNTIME;
      }
      XMMA_LWDA_CALL(lwdaFuncSetAttribute(
        xmma::ext::gemm::sparse_imma_gemm::kernel<Gemm_traits>,
        lwdaFuncAttributeMaxDynamicSharedMemorySize,
        workspace->smem_size));
      XMMA_LWDA_CALL(lwdaFuncSetAttribute(
        xmma::ext::gemm::sparse_imma_gemm::kernel<Gemm_traits>,
        lwdaFuncAttributePreferredSharedMemoryCarveout,
        100));
    }

    xmma::ext::gemm::sparse_imma_gemm::kernel<Gemm_traits>
      <<<workspace->grid, Gemm_traits::Cta_tile::THREADS_PER_CTA,
        workspace->smem_size, lwda_stream>>>(workspace->xmma_params);
    XMMA_LWDA_CALL(lwdaGetLastError());
    return xmma::Error::SUCCESS;
  }
#ifdef LINK
  static xmma::Error driver_run(xmma::Host_workspace<Gemm_traits> *workspace, LWfunction kernel,
                             lwdaStream_t &lwda_stream) {

    if( workspace->smem_size > 48 * 1024 ) {
      if( workspace->xmma_params.ampere && workspace->smem_size > 164*1024 ||
          !workspace->xmma_params.ampere && workspace->smem_size > 64*1024) {
        return xmma::Error::ERROR_LWDA_RUNTIME;
      }
      lwCheckErrors(cask::lwFuncSetAttribute(kernel,LW_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, workspace->smem_size));
      lwCheckErrors(cask::lwFuncSetAttribute(kernel,LW_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT,100));
    }
    void* launchParamsPtr = (void*)(&workspace->xmma_params);
    lwCheckErrors(cask::lwLaunchKernel(kernel,
                  workspace->grid.x,
                  workspace->grid.y,
                  workspace->grid.z,
                  Gemm_traits::Cta_tile::THREADS_PER_CTA,
                  1,
                  1,
                  workspace->smem_size,
                  //dynmicSMemSize,
                  lwda_stream,
                  &launchParamsPtr,
                  nullptr));

    // If we need two kernels to run split-k launch the second grid.
    //if( workspace->xmma_params.split_k.kernels == 2 ) {
    //  workspace->xmma_params.split_k.kernels = 1;
    //  workspace->grid.z = Gemm_traits::Xmma_tile::XMMAS_N;
    //  xmma::ext::gemm::sparse_hmma_gemm::split_k_kernel<Gemm_traits>
    //      <<<workspace->grid, Gemm_traits::Cta_tile::THREADS_PER_CTA,
    //      workspace->epilogue_size_in_bytes, lwda_stream>>>(workspace->xmma_params);
    //  workspace->xmma_params.split_k.kernels = 2;
    //}
    //XMMA_LWDA_CALL(lwdaGetLastError());
    return xmma::Error::SUCCESS;

  }
#else
  static xmma::Error driver_run(xmma::Host_workspace<Gemm_traits> *workspace, LWfunction kernel,
                             lwdaStream_t &lwda_stream) {
    printf("warning: error call driver launch\n");
    return xmma::Error::SUCCESS;
  }
#endif

  static lwdaError_t get_func_attributes(lwdaFuncAttributes* attr) {
      lwdaError_t lwda_status = lwdaFuncGetAttributes(attr,
                                            xmma::ext::gemm::sparse_imma_gemm::kernel<Gemm_traits>);
      attr->maxDynamicSharedSizeBytes =
          Gemm_traits::dynamic_smem_size_per_cta();
      attr->maxThreadsPerBlock = Gemm_traits::threads_per_cta();
    return lwda_status;
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Gemm_traits>
xmma::Error run_kernel(
  const void *a_data,
  const void *b_data,
  const void *c_data,
  const void *e_data,
  const void *bias_data,
  void *d_data,
  void *host_ptr,
  void *device_ptr,
  Runtime_params<Gemm_traits> &runtime_params,
  lwdaStream_t &lwda_stream)
{
  auto host_workspace = static_cast<xmma::Host_workspace<Gemm_traits> *>(host_ptr);
  auto &params = host_workspace->xmma_params;

  params.a_gmem = a_data;
  params.b_gmem = b_data;
  params.c_gmem = c_data;
  params.d_gmem = d_data;
  params.e_gmem = e_data;
  params.bias_gmem = bias_data;

  float *geluScale = &runtime_params.gelu_scale;
  params.runtime_params.runtime_param0 = *reinterpret_cast<int32_t*>(geluScale);

  // Initialize the L2 descriptors
  params.mem_descriptors.descriptor_a = ((uint64_t)runtime_params.descriptor_a << 32);
  params.mem_descriptors.descriptor_b = ((uint64_t)runtime_params.descriptor_b << 32);
  params.mem_descriptors.descriptor_c = ((uint64_t)runtime_params.descriptor_c1 << 32) + (uint64_t)runtime_params.descriptor_c0;
  params.mem_descriptors.descriptor_d = ((uint64_t)runtime_params.descriptor_d1 << 32) + (uint64_t)runtime_params.descriptor_d0;

  // Setup & clear(if needed) split-k buffers
  params.split_k.set_base_ptr(device_ptr);
  XMMA_LWDA_CALL(params.split_k.clear_buffers(device_ptr, lwda_stream));

  if( runtime_params.isRtKernel){
     if( Device_kernel<Gemm_traits, Gemm_traits::USE_SPARSE_IMMA>
        ::driver_run( host_workspace, runtime_params.kernel, lwda_stream ) != xmma::Error::SUCCESS ) {
        return xmma::Error::ERROR_LWDA_RUNTIME;
        }
  }
  else {
      if( Device_kernel<Gemm_traits, Gemm_traits::USE_SPARSE_IMMA>
          ::run( host_workspace, lwda_stream ) != xmma::Error::SUCCESS ) {
          return xmma::Error::ERROR_LWDA_RUNTIME;
      }
  }
  return xmma::Error::SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Gemm_traits>
xmma::Error get_func_attributes(lwdaFuncAttributes* attr) {
  if( Device_kernel<Gemm_traits,
                    Gemm_traits::USE_SPARSE_IMMA>
      ::get_func_attributes(attr) != lwdaSuccess ) {
      return xmma::Error::ERROR_LWDA_RUNTIME;
  }
  return xmma::Error::SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace gemm
} // namespace ext
} // namespace xmma
///////////////////////////////////////////////////////////////////////////////////////////////////
