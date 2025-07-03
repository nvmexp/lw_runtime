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

#ifndef _XMMA_EXT_DEPTHWISE_HELPER_KERNEL_H
#define _XMMA_EXT_DEPTHWISE_HELPER_KERNEL_H

#pragma once

#include <cstdint>
#include <lwda_runtime.h>

namespace xmma{
namespace ext
{
namespace depthwise_colwolution
{

struct Helper_kernel_param{
    public:
    static const int64_t THREADS_PER_CTA = 128;
    void *src;
    void *dst;
    int64_t m;
    int64_t n;
};

__global__ static void helper_kernel(Helper_kernel_param param) {
    int64_t linear_index = blockIdx.x * Helper_kernel_param::THREADS_PER_CTA + threadIdx.x;
    int64_t index_m = linear_index/param.n;
    int64_t index_n = linear_index%param.n;
    int64_t new_linear_index = index_m + index_n * param.m;
    uint16_t *dst_with_type = static_cast<uint16_t *>(param.dst);
    uint16_t *src_with_type = static_cast<uint16_t *>(param.src);
    if(linear_index<param.m*param.n){
    dst_with_type[new_linear_index] = src_with_type[linear_index];}
    return;
}

} // namespace depthwise_colwolution
} // namespace ext
} // namespace xmma

#endif
