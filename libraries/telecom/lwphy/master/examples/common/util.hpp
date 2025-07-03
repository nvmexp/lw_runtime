/*
 * Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 */

#if !defined(UTIL_HPP_INCLUDED_)
#define UTIL_HPP_INCLUDED_

#include "lwphy.h"

#define LWDA_CHECK(result)                        \
    if((lwdaError_t)result != lwdaSuccess)        \
    {                                             \
        fprintf(stderr,                           \
                "LWCA Runtime Error: %s:%i:%s\n", \
                __FILE__,                         \
                __LINE__,                         \
                lwdaGetErrorString(result));      \
    }

void gpu_ms_delay(uint32_t delay_ms, int gpuId = 0, lwdaStream_t lwStrm = 0);
void gpu_ms_sleep(uint32_t sleep_ms, int gpuId = 0, lwdaStream_t lwStrm = 0);

#endif // !defined(UTIL_HPP_INCLUDED_)
