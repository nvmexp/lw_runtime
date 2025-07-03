/*
 * Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 */

#if !defined(LDPC_TV_UTIL_LWH_INCLUDED_)
#define LDPC_TV_UTIL_LWH_INCLUDED_

#include "lwphy.hpp"
#include "ldpc_tv.hpp"

lwdaError_t gpu_gen_rand_bit(void* dst, int len, int gpuId = 0, lwdaStream_t lwStrm = 0);
lwdaError_t gpu_colwert_symbol(float* s, int len, int gpuId = 0, lwdaStream_t lwStrm = 0);
lwdaError_t gpu_add_noise(float* s, int len, float snr, int gpuId = 0, lwdaStream_t lwStrm = 0);
lwdaError_t gpu_repmat(float* s, float* d, int len, int rx, int ry, int gpuId = 0, lwdaStream_t lwStrm = 0);
lwdaError_t gpu_init_elem(float* s, int rows, int cols, int rstart, int nr, float v, int gpuId = 0, lwdaStream_t lwStrm = 0);
lwdaError_t colwert_to_bit(void* h_addr, int len, void* d_addr);

#endif // !defined(LDPC_TV_UTIL_LWH_INCLUDED_)