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

#ifndef _XMMA_EXT_DEPTHWISE_COLWOLUTION_TYPE_COLWERTER_H
#define _XMMA_EXT_DEPTHWISE_COLWOLUTION_TYPE_COLWERTER_H

#pragma once

#include "utils.h"
#include <cstdint>
#include <lwda_fp16.h>
#include <lwda_runtime.h>

namespace xmma{
namespace ext
{
namespace depthwise_colwolution
{

template <typename Out_,
          typename In_,
          typename Packed_Out_,
          typename Packed_In_,
          int32_t N_Packed_In_>
struct Type_colwerter {
    using Out_t = Out_;
    using In_t = In_;
    using Packed_In_t = Packed_In_;
    using Packed_Out_t = Packed_Out_;
    static const int32_t N_Packed_In = N_Packed_In_;
    static const int32_t N_Packed_Out = N_Packed_In * sizeof(Packed_In_t) / sizeof(Packed_Out_t);

    __device__ inline static void exlwte(Packed_Out_t (&packed_out)[N_Packed_Out],
                                         Packed_In_t packed_in[N_Packed_In]);
};

template <int32_t N_Packed_In_>
struct Type_colwerter<float, __half, uint32_t, uint32_t, N_Packed_In_> {
    using Out_t = float;
    using In_t = __half;
    using Packed_In_t = uint32_t;
    using Packed_Out_t = uint32_t;
    static const int32_t N_Packed_In = N_Packed_In_;
    static const int32_t RATIO_EXPAND = sizeof(Out_t) / sizeof(In_t);
    static const int32_t N_Packed_Out =
        N_Packed_In * sizeof(Packed_In_t) * RATIO_EXPAND / sizeof(Packed_Out_t);

    __device__ inline static void exlwte(Packed_Out_t (&packed_out)[N_Packed_Out],
                                         Packed_In_t packed_in[N_Packed_In])
    {
#pragma unroll
        for (int32_t i = 0; i < N_Packed_In; ++i) {
            packed_out[i * 2] = __float_as_int(__low2float(uint32_as_half2(packed_in[i])));
            packed_out[i * 2 + 1] = __float_as_int(__high2float(uint32_as_half2(packed_in[i])));
        }
    }
};

template <int32_t N_Packed_In_>
struct Type_colwerter<__half, float, uint32_t, uint32_t, N_Packed_In_> {
    using Out_t = __half;
    using In_t = float;
    using Packed_In_t = uint32_t;
    using Packed_Out_t = uint32_t;
    static const int32_t N_Packed_In = N_Packed_In_;

    static const int32_t RATIO_COMPRESS = sizeof(In_t) / sizeof(Out_t);
    static const int32_t N_Packed_Out =
        N_Packed_In * sizeof(Packed_In_t) / RATIO_COMPRESS / sizeof(Packed_Out_t);

    __device__ inline static void exlwte(Packed_Out_t (&packed_out)[N_Packed_Out],
                                         Packed_In_t packed_in[N_Packed_In])
    {
#pragma unroll
        for (int32_t i = 0; i < N_Packed_In; ++i) {
            packed_out[i] = xmma::float2_to_half2(__uint_as_float(packed_in[2 * i]),
                                                  __uint_as_float(packed_in[2 * i + 1]));
        }
    }
};

template <int32_t N_Packed_In_>
struct Type_colwerter<__half, __half, uint16_t, uint16_t, N_Packed_In_> {
    using Out_t = __half;
    using In_t = __half;
    using Packed_In_t = uint16_t;
    using Packed_Out_t = uint16_t;
    static const int32_t N_Packed_In = N_Packed_In_;
    static const int32_t RATIO_EXPAND = sizeof(Out_t) / sizeof(In_t);
    static const int32_t N_Packed_Out =
        N_Packed_In * sizeof(Packed_In_t) * RATIO_EXPAND / sizeof(Packed_Out_t);

    __device__ inline static void exlwte(Packed_Out_t (&packed_out)[N_Packed_Out],
                                         Packed_In_t packed_in[N_Packed_In])
    {
#pragma unroll
        for (int32_t i = 0; i < N_Packed_In; ++i) {
            packed_out[i] = packed_in[i];
        }
    }
};

template <int32_t N_Packed_In_>
struct Type_colwerter<__half, __half, uint32_t, uint32_t, N_Packed_In_> {
    using Out_t = __half;
    using In_t = __half;
    using Packed_In_t = uint32_t;
    using Packed_Out_t = uint32_t;
    static const int32_t N_Packed_In = N_Packed_In_;
    static const int32_t RATIO_EXPAND = sizeof(Out_t) / sizeof(In_t);
    static const int32_t N_Packed_Out =
        N_Packed_In * sizeof(Packed_In_t) * RATIO_EXPAND / sizeof(Packed_Out_t);

    __device__ inline static void exlwte(Packed_Out_t (&packed_out)[N_Packed_Out],
                                         Packed_In_t packed_in[N_Packed_In])
    {
#pragma unroll
        for (int32_t i = 0; i < N_Packed_In; ++i) {
            packed_out[i] = packed_in[i];
        }
    }
};

template <int32_t N_Packed_In_>
struct Type_colwerter<float, float, uint32_t, uint32_t, N_Packed_In_> {
    using Out_t = float;
    using In_t = float;
    using Packed_In_t = uint32_t;
    using Packed_Out_t = uint32_t;
    static const int32_t N_Packed_In = N_Packed_In_;
    static const int32_t RATIO_EXPAND = sizeof(Out_t) / sizeof(In_t);
    static const int32_t N_Packed_Out =
        N_Packed_In * sizeof(Packed_In_t) * RATIO_EXPAND / sizeof(Packed_Out_t);

    __device__ inline static void exlwte(Packed_Out_t (&packed_out)[N_Packed_Out],
                                         Packed_In_t packed_in[N_Packed_In])
    {
#pragma unroll
        for (int32_t i = 0; i < N_Packed_In; ++i) {
            packed_out[i] = packed_in[i];
        }
    }
};

template <int32_t N_Packed_In_> struct Type_colwerter<float, __half, uint4, uint4, N_Packed_In_> {
    using Out_t = float;
    using In_t = __half;
    using Packed_In_t = uint4;
    using Packed_Out_t = uint4;
    static const int32_t N_Packed_In = N_Packed_In_;
    static const int32_t RATIO_EXPAND = sizeof(Out_t) / sizeof(In_t);
    static const int32_t N_Packed_Out =
        N_Packed_In * sizeof(Packed_In_t) * RATIO_EXPAND / sizeof(Packed_Out_t);

    __device__ inline static void exlwte(Packed_Out_t (&packed_out)[N_Packed_Out],
                                         Packed_In_t packed_in[N_Packed_In])
    {
#pragma unroll
        for (int32_t i = 0; i < N_Packed_In; ++i) {
            packed_out[i * 2].x = __float_as_int(__low2float(uint32_as_half2(packed_in[i].x)));
            packed_out[i * 2].y = __float_as_int(__high2float(uint32_as_half2(packed_in[i].x)));
            packed_out[i * 2].z = __float_as_int(__low2float(uint32_as_half2(packed_in[i].y)));
            packed_out[i * 2].w = __float_as_int(__high2float(uint32_as_half2(packed_in[i].y)));
            packed_out[i * 2 + 1].x = __float_as_int(__low2float(uint32_as_half2(packed_in[i].z)));
            packed_out[i * 2 + 1].y = __float_as_int(__high2float(uint32_as_half2(packed_in[i].z)));
            packed_out[i * 2 + 1].z = __float_as_int(__low2float(uint32_as_half2(packed_in[i].w)));
            packed_out[i * 2 + 1].w = __float_as_int(__high2float(uint32_as_half2(packed_in[i].w)));
        }
    }
};

template <int32_t N_Packed_In_> struct Type_colwerter<float, __half, uint4, uint2, N_Packed_In_> {
    using Out_t = float;
    using In_t = __half;
    using Packed_In_t = uint2;
    using Packed_Out_t = uint4;
    static const int32_t N_Packed_In = N_Packed_In_;
    static const int32_t RATIO_EXPAND = sizeof(Out_t) / sizeof(In_t);
    static const int32_t N_Packed_Out =
        N_Packed_In * sizeof(Packed_In_t) * RATIO_EXPAND / sizeof(Packed_Out_t);

    __device__ inline static void exlwte(Packed_Out_t (&packed_out)[N_Packed_Out],
                                         Packed_In_t packed_in[N_Packed_In])
    {
#pragma unroll
        for (int32_t i = 0; i < N_Packed_In; ++i) {
            packed_out[i].x = __float_as_int(__low2float(uint32_as_half2(packed_in[i].x)));
            packed_out[i].y = __float_as_int(__high2float(uint32_as_half2(packed_in[i].x)));
            packed_out[i].z = __float_as_int(__low2float(uint32_as_half2(packed_in[i].y)));
            packed_out[i].w = __float_as_int(__high2float(uint32_as_half2(packed_in[i].y)));
        }
    }
};

template <int32_t N_Packed_In_> struct Type_colwerter<__half, float, uint2, uint4, N_Packed_In_> {
    using Out_t = __half;
    using In_t = float;
    using Packed_In_t = uint4;
    using Packed_Out_t = uint2;
    static const int32_t N_Packed_In = N_Packed_In_;
    static const int32_t RATIO_COMPRESS = sizeof(In_t) / sizeof(Out_t);
    static const int32_t N_Packed_Out =
        N_Packed_In * sizeof(Packed_In_t) / RATIO_COMPRESS / sizeof(Packed_Out_t);

    __device__ inline static void exlwte(Packed_Out_t (&packed_out)[N_Packed_Out],
                                         Packed_In_t packed_in[N_Packed_In])
    {
#pragma unroll
        for (int32_t i = 0; i < N_Packed_In; ++i) {
            packed_out[i].x = xmma::float2_to_half2(__uint_as_float(packed_in[i].x),
                                                    __uint_as_float(packed_in[i].y));
            packed_out[i].y = xmma::float2_to_half2(__uint_as_float(packed_in[i].z),
                                                    __uint_as_float(packed_in[i].w));
        }
    }
};

template <int32_t N_Packed_In_> struct Type_colwerter<__half, __half, uint2, uint2, N_Packed_In_> {
    using Out_t = __half;
    using In_t = __half;
    using Packed_In_t = uint2;
    using Packed_Out_t = uint2;
    static const int32_t N_Packed_In = N_Packed_In_;
    static const int32_t N_Packed_Out = N_Packed_In * sizeof(Packed_In_t) / sizeof(Packed_Out_t);

    __device__ inline static void exlwte(Packed_Out_t (&packed_out)[N_Packed_Out],
                                         Packed_In_t packed_in[N_Packed_In])
    {
#pragma unroll
        for (int32_t i = 0; i < N_Packed_In; ++i) {
            packed_out[i].x = packed_in[i].x;
            packed_out[i].y = packed_in[i].y;
        }
    }
};

template <int32_t N_Packed_In_> struct Type_colwerter<__half, __half, uint4, uint4, N_Packed_In_> {
    using Out_t = __half;
    using In_t = __half;
    using Packed_In_t = uint4;
    using Packed_Out_t = uint4;
    static const int32_t N_Packed_In = N_Packed_In_;
    static const int32_t N_Packed_Out = N_Packed_In * sizeof(Packed_In_t) / sizeof(Packed_Out_t);

    __device__ inline static void exlwte(Packed_Out_t (&packed_out)[N_Packed_Out],
                                         Packed_In_t packed_in[N_Packed_In])
    {
#pragma unroll
        for (int32_t i = 0; i < N_Packed_In; ++i) {
            packed_out[i].x = packed_in[i].x;
            packed_out[i].y = packed_in[i].y;
            packed_out[i].z = packed_in[i].z;
            packed_out[i].w = packed_in[i].w;
        }
    }
};

} // namespace depthwise_colwolution
} // namespace ext
} // namespace xmma

#endif
