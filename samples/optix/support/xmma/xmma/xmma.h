/***************************************************************************************************
 * Copyright (c) 2019, LWPU CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted
 * provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright notice, this list of
 *       conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright notice, this list of
 *       conditions and the following disclaimer in the documentation and/or other materials
 *       provided with the distribution.
 *     * Neither the name of the LWPU CORPORATION nor the names of its contributors may be used
 *       to endorse or promote products derived from this software without specific prior written
 *       permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL LWPU CORPORATION BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROLWREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TOR (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

#pragma once

#if !defined(__LWDA_ARCH__)
#include <stdexcept>
#endif

#if !defined(__LWDACC_RTC__)

#include <cstdlib>
#include <cstdio>
#include <cassert>
#include <cinttypes>

#include <typeinfo>
#include <type_traits>
#include <limits>
#include <array>

#include <lwca.h>
#include <lwda_runtime_api.h>

#else
// #include <lwca/std/type_traits>
// #include <lwca/std/limits>
// #define assert(x)
#endif

#include <cstdint>

#include <lwda_fp16.h>

#if defined(__LWDA_ARCH__)
#define XMMA_HOST_DEVICE inline __device__
#define XMMA_HOST inline __host__
#define XMMA_DEVICE inline __device__
#define XMMA_GLOBAL __global__
#elif defined(__LWCC__)
#define XMMA_HOST_DEVICE inline __device__ __host__
#define XMMA_HOST inline __host__
#define XMMA_DEVICE inline __device__
#define XMMA_GLOBAL __global__
#elif defined(__LWDACC_RTC__)
#define XMMA_HOST_DEVICE inline __device__
#define XMMA_HOST inline __host__
#define XMMA_DEVICE inline __device__
#define XMMA_GLOBAL __global__
#else
#define XMMA_HOST_DEVICE inline
#define XMMA_HOST inline
#endif

#ifdef XMMA_NAMESPACE_SUFFIX
#define concat_tok(a, b) a ## b
#define mkxmmanamespacesuffix(pre, ns) concat_tok(pre, ns)
#define xmma mkxmmanamespacesuffix(xmma_, XMMA_NAMESPACE_SUFFIX)
#endif

namespace xmma {

enum class Error {
    SUCCESS = 0,
    ERROR_LWDA_RUNTIME,
    ERROR_ILWALID_PARAMS,
    ERROR_UNKNOWN,
};

///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Kernel_traits>
struct Host_workspace {
    typename Kernel_traits::Params xmma_params;

    dim3 grid;
    dim3 split_k_grid;
    int smem_size;
    int epilogue_size_in_bytes;
    bool split_k_with_reduction;  // for split K
    size_t device_workspace_size;

#if !defined(__LWDACC_RTC__)
    lwdaError_t error;
#endif
};

template<typename Gpu_arch,
         typename Kernel_type>
lwdaError_t set_func_attributes( Kernel_type *entry, int32_t smem_size ) {
    lwdaError_t err = lwdaSuccess;
    if( smem_size > 48 * 1024 ) {
        if( smem_size > Gpu_arch::MAX_DYNAMIC_SMEM_SIZE_BYTES ) {
            err = lwdaErrorIlwalidValue;
        } else {
            err = lwdaFuncSetAttribute( entry, lwdaFuncAttributeMaxDynamicSharedMemorySize, smem_size );
            err = lwdaFuncSetAttribute( entry, lwdaFuncAttributePreferredSharedMemoryCarveout, 100 );
        }
    }

    return err;
}

template<typename Kernel_traits,
         typename Kernel_type>
lwdaError_t get_func_attributes( Kernel_type *entry, lwdaFuncAttributes* attr ) {
    lwdaError_t err = lwdaFuncGetAttributes( attr, entry );
    if (err == lwdaSuccess) {
        attr->maxDynamicSharedSizeBytes = Kernel_traits::dynamic_smem_size_per_cta();
        attr->maxThreadsPerBlock = Kernel_traits::threads_per_cta();
    }
    return err;
}

}

// Wrapper MACRO to handle any xmma call which may return error
#define XMMA_CALL(call) do {                                            \
        xmma::Error err = call;                                         \
        if( err != xmma::Error::SUCCESS ) {                             \
            return err;                                                 \
        }                                                               \
    } while (0)

#if !defined(__LWDACC_RTC__)

// Wrapper MACRO to handle any call which return lwdaError_t
#define XMMA_LWDA_CALL(call) do {                                       \
        lwdaError_t status_ = call;                                     \
        if( status_ != lwdaSuccess ) {                                  \
            fprintf(stderr, "[ ERROR: LWCA Runtime ] %s:%d: %s\n",      \
                    __FILE__, __LINE__, lwdaGetErrorString(status_));   \
            return xmma::Error::ERROR_LWDA_RUNTIME;                     \
        }                                                               \
    } while (0)

#else

#define XMMA_LWDA_CALL(call)

#endif // !defined(__LWDACC_RTC__)
