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

#ifndef _XMMA_EXT_DEPTHWISE_UTILS_H
#define _XMMA_EXT_DEPTHWISE_UTILS_H

#pragma once

#include "data_type.h"
#include "xmma/utils.h"
#include <cstdint>
#include <cstdio>
#include <lwda_fp16.h>
#include <lwda_runtime.h>

namespace xmma
{
namespace ext
{
namespace depthwise_colwolution
{

__device__ inline int32_t threadIdx_x() { return threadIdx.x; }

__device__ inline int32_t threadIdx_y() { return threadIdx.z; }

__device__ inline int32_t threadIdx_z() { return threadIdx.z; }

__device__ inline int32_t blockIdx_x() { return blockIdx.x; }

__device__ inline int32_t blockIdx_y() { return blockIdx.y; }

__device__ inline int32_t blockIdx_z() { return blockIdx.z; }

__device__ inline int32_t gridDim_x() { return gridDim.x; }

__device__ inline int32_t gridDim_y() { return gridDim.y; }

__device__ inline int32_t gridDim_z() { return gridDim.z; }

template <bool CHOOSE_A, typename A, typename B> struct Type_selector {
    using Type = void;
};

template <typename A, typename B> struct Type_selector<true, A, B> {
    using Type = A;
};

template <typename A, typename B> struct Type_selector<false, A, B> {
    using Type = B;
};

template <typename Packed_type_>
__device__ inline uint32_t select(Packed_type_ data, int32_t index);

template <> __device__ inline uint32_t select<uint4>(uint4 data, int32_t index)
{
    if (index == 0) {
        return data.x;
    } else if (index == 1) {
        return data.y;
    } else if (index == 2) {
        return data.z;
    } else if (index == 3) {
        return data.w;
    } else {
        assert(false);
        return 0xffffffff;
    }
}

template <> __device__ inline uint32_t select<uint2>(uint2 data, int32_t index)
{
    if (index == 0) {
        return data.x;
    } else if (index == 1) {
        return data.y;
    } else {
        assert(false);
        return 0xffffffff;
    }
}

template <> __device__ inline uint32_t select<uint32_t>(uint32_t data, int32_t index)
{
    if (index == 0) {
        return data;
    } else {
        assert(false);
        return 0xffffffff;
    }
}

template <typename Packed_type_>
__device__ inline void assign(Packed_type_ &data, int32_t index, uint32_t in);

template <> __device__ inline void assign<uint4>(uint4 &data, int32_t index, uint32_t in)
{
    if (index == 0) {
        data.x = in;
    } else if (index == 1) {
        data.y = in;
    } else if (index == 2) {
        data.z = in;
    } else if (index == 3) {
        data.w = in;
    } else {
        assert(false);
    }
}

template <> __device__ inline void assign<uint2>(uint2 &data, int32_t index, uint32_t in)
{
    if (index == 0) {
        data.x = in;
    } else if (index == 1) {
        data.y = in;
    } else {
        assert(false);
    }
}

template <> __device__ inline void assign<uint32_t>(uint32_t &data, int32_t index, uint32_t in)
{
    if (index == 0) {
        data = in;
    } else {
        assert(false);
    }
}

__device__ inline void make_packed_uint(uint4 &out, uint32_t in[4])
{
    out.x = in[0];
    out.y = in[1];
    out.z = in[2];
    out.w = in[3];
}

__device__ inline void make_packed_uint(uint2 &out, uint32_t in[2])
{
    out.x = in[0];
    out.y = in[1];
}

__device__ inline void make_packed_uint(uint32_t &out, uint32_t in[1]) { out = in[0]; }

template <int32_t DIM_0, int32_t DIM_1, int32_t DIM_2, int32_t DIM_3>
__device__ inline void copy(uint32_t (&out)[DIM_0][DIM_1][DIM_2][DIM_3],
                            uint32_t in[DIM_0][DIM_1][DIM_2][DIM_3])
{
#pragma unroll
    for (int32_t index_dim_0 = 0; index_dim_0 < DIM_0; ++index_dim_0) {
#pragma unroll
        for (int32_t index_dim_1 = 0; index_dim_1 < DIM_1; ++index_dim_1) {
#pragma unroll
            for (int32_t index_dim_2 = 0; index_dim_2 < DIM_2; ++index_dim_2) {
#pragma unroll
                for (int32_t index_dim_3 = 0; index_dim_3 < DIM_3; ++index_dim_3) {
                    out[index_dim_0][index_dim_1][index_dim_2][index_dim_3] =
                        in[index_dim_0][index_dim_1][index_dim_2][index_dim_3];
                }
            }
        }
    }
}

template <int32_t BOUNDARY> __device__ int32_t inline increase_and_mod(const int32_t a)
{
    int32_t value;
    if (a == BOUNDARY - 1) {
        value = 0;
    } else {
        value = a + 1;
    }
    return value;
}

template <bool USE_TEMPLATE, int32_t A0, int32_t A1>
__device__ int32_t multiply_and_add(
    const int32_t c, const int32_t a0, const int32_t b0, const int32_t a1, const int32_t b1)
{
    if (USE_TEMPLATE) {
        return c + A0 * b0 + A1 * b1;
    } else {
        return c + a0 * b0 + a1 * b1;
    }
}

template <typename Type_c_, typename Type_a_, typename Type_b_> struct Multiply_and_add {
    __device__ inline static uint32_t exlwte(const uint32_t c, const uint32_t a, const uint32_t b);
};

template <> struct Multiply_and_add<Data_type_fp32, Data_type_fp32, Data_type_fp32> {
    __device__ inline static uint32_t exlwte(const uint32_t c, const uint32_t a, const uint32_t b)
    {
        float c_ = __uint_as_float(c);
        float a_ = __uint_as_float(a);
        float b_ = __uint_as_float(b);
        float d_ = a_ * b_ + c_;
        return __float_as_uint(d_);
    }
};

template <> struct Multiply_and_add<Data_type_fp16, Data_type_fp16, Data_type_fp16> {
    __device__ inline static uint32_t exlwte(const uint32_t c, const uint32_t a, const uint32_t b)
    {
        return xmma::hfma2(a, b, c);
    }
};

__host__ __device__ inline int32_t uint32_as_int32(const uint32_t in)
{
    return *reinterpret_cast<const int32_t *>(&in);
}

__host__ __device__ inline uint32_t int32_as_uint32(const int32_t in)
{
    return *reinterpret_cast<const uint32_t *>(&in);
}

//__host__ __device__ inline float uint32_as_float(const uint32_t in)
//{
//    return *reinterpret_cast<const float *>(&in);
//}

__device__ inline __half2 uint32_as_half2(const uint32_t in)
{
    return *reinterpret_cast<const __half2 *>(&in);
}

template <> struct Multiply_and_add<Data_type_int32, Data_type_int32, Data_type_int32> {
    __device__ inline static uint32_t exlwte(const uint32_t c, const uint32_t a, const uint32_t b)
    {
        int32_t c_ = uint32_as_int32(c);
        int32_t a_ = uint32_as_int32(a);
        int32_t b_ = uint32_as_int32(b);
        int32_t d_ = a_ * b_ + c_;
        return int32_as_uint32(d_);
    }
};

template <typename Type_a_, typename Type_b_> struct Add {
    __device__ inline static uint32_t exlwte(const uint32_t a, const uint32_t b);
};

template <> struct Add<Data_type_fp32, Data_type_fp32> {
    __device__ inline static uint32_t exlwte(const uint32_t a, const uint32_t b)
    {
        float a_ = __uint_as_float(a);
        float b_ = __uint_as_float(b);
        float c_ = a_ + b_;
        return __float_as_uint(c_);
    }
};

template <> struct Add<Data_type_fp16, Data_type_fp16> {
    __device__ inline static uint32_t exlwte(const uint32_t a, const uint32_t b)
    {
        return xmma::hadd2(a, b);
    }
};

template <typename Type_element_, typename Type_out_, int32_t N, int32_t M> struct Reduction {
    public:
    using Type_element_t = Type_element_;
    using Type_out_t = Type_out_;
    __device__ inline static void exlwte(Type_out_t (&out)[N], Type_out_t in[N][M]);
};

template <int32_t N, int32_t M> struct Reduction<Data_type_fp32, uint4, N, M> {
    public:
    using Type_element_t = Data_type_fp32;
    using Type_out_t = uint4;
    __device__ inline static void exlwte(Type_out_t (&out)[N], Type_out_t in[N][M])
    {
#pragma unroll
        for (int32_t n = 0; n < N; ++n) {
            out[n].x = in[n][0].x;
            out[n].y = in[n][0].y;
            out[n].z = in[n][0].z;
            out[n].w = in[n][0].w;
#pragma unroll
            for (int32_t m = 1; m < M; ++m) {
                out[n].x = Add<Type_element_t, Type_element_t>::exlwte(out[n].x, in[n][m].x);
                out[n].y = Add<Type_element_t, Type_element_t>::exlwte(out[n].y, in[n][m].y);
                out[n].z = Add<Type_element_t, Type_element_t>::exlwte(out[n].z, in[n][m].z);
                out[n].w = Add<Type_element_t, Type_element_t>::exlwte(out[n].w, in[n][m].w);
            }
        }
    }
};

template <int32_t N, int32_t M> struct Reduction<Data_type_fp16, uint2, N, M> {
    public:
    using Type_element_t = Data_type_fp16;
    using Type_out_t = uint2;
    __device__ inline static void exlwte(Type_out_t (&out)[N], Type_out_t in[N][M])
    {
#pragma unroll
        for (int32_t n = 0; n < N; ++n) {
            out[n].x = in[n][0].x;
            out[n].y = in[n][0].y;
#pragma unroll
            for (int32_t m = 1; m < M; ++m) {
                out[n].x = Add<Type_element_t, Type_element_t>::exlwte(out[n].x, in[n][m].x);
                out[n].y = Add<Type_element_t, Type_element_t>::exlwte(out[n].y, in[n][m].y);
            }
        }
    }
};

template <int32_t N, int32_t M> struct Reduction<Data_type_fp16, uint32_t, N, M> {
    public:
    using Type_element_t = Data_type_fp16;
    using Type_out_t = uint32_t;
    __device__ inline static void exlwte(Type_out_t (&out)[N], Type_out_t in[N][M])
    {
#pragma unroll
        for (int32_t n = 0; n < N; ++n) {
            out[n] = in[n][0];
#pragma unroll
            for (int32_t m = 1; m < M; ++m) {
                out[n] = Add<Type_element_t, Type_element_t>::exlwte(out[n], in[n][m]);
            }
        }
    }
};

template <int32_t N, int32_t M> struct Reduction<Data_type_fp16, uint16_t, N, M> {
    public:
    using Type_element_t = Data_type_fp16;
    using Type_out_t = uint16_t;
    __device__ inline static void exlwte(Type_out_t (&out)[N], Type_out_t in[N][M])
    {
#pragma unroll
        for (int32_t n = 0; n < N; ++n) {
            out[n] = in[n][0];
#pragma unroll
            for (int32_t m = 1; m < M; ++m) {
                out[n] = hadd(out[n], in[n][m]);
            }
        }
    }
};

template <int32_t N, int32_t M> struct Reduction<Data_type_fp16, uint4, N, M> {
    public:
    using Type_element_t = Data_type_fp16;
    using Type_out_t = uint4;
    __device__ inline static void exlwte(Type_out_t (&out)[N], Type_out_t in[N][M])
    {
#pragma unroll
        for (int32_t n = 0; n < N; ++n) {
            out[n].x = in[n][0].x;
            out[n].y = in[n][0].y;
            out[n].z = in[n][0].z;
            out[n].w = in[n][0].w;
#pragma unroll
            for (int32_t m = 1; m < M; ++m) {
                out[n].x = Add<Type_element_t, Type_element_t>::exlwte(out[n].x, in[n][m].x);
                out[n].y = Add<Type_element_t, Type_element_t>::exlwte(out[n].y, in[n][m].y);
                out[n].z = Add<Type_element_t, Type_element_t>::exlwte(out[n].z, in[n][m].z);
                out[n].w = Add<Type_element_t, Type_element_t>::exlwte(out[n].w, in[n][m].w);
            }
        }
    }
};

template <typename Type_element_, typename Type_out_, int32_t N> struct Apply_alpha {
    public:
    using Type_element_t = Type_element_;
    using Type_out_t = Type_out_;
    __device__ inline static void exlwte(Type_out_t (&out)[N], uint32_t alpha);
};

template <int32_t N> struct Apply_alpha<Data_type_fp32, uint4, N> {
    public:
    using Type_element_t = Data_type_fp32;
    using Type_out_t = uint4;
    __device__ inline static void exlwte(Type_out_t (&out)[N], uint32_t alpha)
    {
#pragma unroll
        for (int32_t n = 0; n < N; ++n) {
            out[n].x = Multiply_and_add<Type_element_t, Type_element_t, Type_element_t>::exlwte(
                0, out[n].x, alpha);
            out[n].y = Multiply_and_add<Type_element_t, Type_element_t, Type_element_t>::exlwte(
                0, out[n].y, alpha);
            out[n].z = Multiply_and_add<Type_element_t, Type_element_t, Type_element_t>::exlwte(
                0, out[n].z, alpha);
            out[n].w = Multiply_and_add<Type_element_t, Type_element_t, Type_element_t>::exlwte(
                0, out[n].w, alpha);
        }
    }
};

template <int32_t N> struct Apply_alpha<Data_type_fp32, uint32_t, N> {
    public:
    using Type_element_t = Data_type_fp32;
    using Type_out_t = uint32_t;
    __device__ inline static void exlwte(Type_out_t (&out)[N], uint32_t alpha)
    {
#pragma unroll
        for (int32_t n = 0; n < N; ++n) {
            out[n] = Multiply_and_add<Type_element_t, Type_element_t, Type_element_t>::exlwte(
                0, out[n], alpha);
        }
    }
};

template <int32_t N> struct Apply_alpha<Data_type_fp16, uint32_t, N> {
    public:
    using Type_element_t = Data_type_fp16;
    using Type_out_t = uint32_t;
    __device__ inline static void exlwte(Type_out_t (&out)[N], uint32_t alpha)
    {
        float float_alpha = __uint_as_float(alpha);
        uint32_t alpha_alpha = xmma::float2_to_half2(float_alpha, float_alpha);
#pragma unroll
        for (int32_t n = 0; n < N; ++n) {
            out[n] = Multiply_and_add<Type_element_t, Type_element_t, Type_element_t>::exlwte(
                0, out[n], alpha_alpha);
        }
    }
};

template <int32_t N> struct Apply_alpha<Data_type_fp16, uint2, N> {
    public:
    using Type_element_t = Data_type_fp16;
    using Type_out_t = uint2;
    __device__ inline static void exlwte(Type_out_t (&out)[N], uint32_t alpha)
    {
        float float_alpha = __uint_as_float(alpha);
        uint32_t alpha_alpha = xmma::float2_to_half2(float_alpha, float_alpha);
#pragma unroll
        for (int32_t n = 0; n < N; ++n) {
            out[n].x = Multiply_and_add<Type_element_t, Type_element_t, Type_element_t>::exlwte(
                0, out[n].x, alpha_alpha);
            out[n].y = Multiply_and_add<Type_element_t, Type_element_t, Type_element_t>::exlwte(
                0, out[n].y, alpha_alpha);
        }
    }
};

template <int32_t N> struct Apply_alpha<Data_type_fp16, uint4, N> {
    public:
    using Type_element_t = Data_type_fp16;
    using Type_out_t = uint4;
    __device__ inline static void exlwte(Type_out_t (&out)[N], uint32_t alpha)
    {
        float float_alpha = __uint_as_float(alpha);
        uint32_t alpha_alpha = xmma::float2_to_half2(float_alpha, float_alpha);
#pragma unroll
        for (int32_t n = 0; n < N; ++n) {
            out[n].x = Multiply_and_add<Type_element_t, Type_element_t, Type_element_t>::exlwte(
                0, out[n].x, alpha_alpha);
            out[n].y = Multiply_and_add<Type_element_t, Type_element_t, Type_element_t>::exlwte(
                0, out[n].y, alpha_alpha);
            out[n].z = Multiply_and_add<Type_element_t, Type_element_t, Type_element_t>::exlwte(
                0, out[n].z, alpha_alpha);
            out[n].w = Multiply_and_add<Type_element_t, Type_element_t, Type_element_t>::exlwte(
                0, out[n].w, alpha_alpha);
        }
    }
};

template <typename Type_element_, typename Type_out_, int32_t N> struct Apply_beta {
    public:
    using Type_element_t = Type_element_;
    using Type_out_t = Type_out_;
    __device__ inline static void exlwte(Type_out_t (&c)[N], Type_out_t a[N], uint32_t beta);
};

template <int32_t N> struct Apply_beta<Data_type_fp32, uint32_t, N> {
    public:
    using Type_element_t = Data_type_fp32;
    using Type_out_t = uint32_t;
    __device__ inline static void exlwte(Type_out_t (&c)[N], Type_out_t a[N], uint32_t beta)
    {
#pragma unroll
        for (int32_t n = 0; n < N; ++n) {
            c[n] = Multiply_and_add<Type_element_t, Type_element_t, Type_element_t>::exlwte(
                c[n], a[n], beta);
        }
    }
};

template <int32_t N> struct Apply_beta<Data_type_fp32, uint4, N> {
    public:
    using Type_element_t = Data_type_fp32;
    using Type_out_t = uint4;
    __device__ inline static void exlwte(Type_out_t (&c)[N], Type_out_t a[N], uint32_t beta)
    {
#pragma unroll
        for (int32_t n = 0; n < N; ++n) {
            c[n].x = Multiply_and_add<Type_element_t, Type_element_t, Type_element_t>::exlwte(
                c[n].x, a[n].x, beta);
            c[n].y = Multiply_and_add<Type_element_t, Type_element_t, Type_element_t>::exlwte(
                c[n].y, a[n].y, beta);
            c[n].z = Multiply_and_add<Type_element_t, Type_element_t, Type_element_t>::exlwte(
                c[n].z, a[n].z, beta);
            c[n].w = Multiply_and_add<Type_element_t, Type_element_t, Type_element_t>::exlwte(
                c[n].w, a[n].w, beta);
        }
    }
};

template <int32_t N> struct Apply_beta<Data_type_fp16, uint32_t, N> {
    public:
    using Type_element_t = Data_type_fp16;
    using Type_out_t = uint32_t;
    __device__ inline static void exlwte(Type_out_t (&c)[N], Type_out_t a[N], uint32_t beta)
    {
        float float_beta = __uint_as_float(beta);
        uint32_t beta_beta = xmma::float2_to_half2(float_beta, float_beta);
#pragma unroll
        for (int32_t n = 0; n < N; ++n) {
            c[n] = Multiply_and_add<Type_element_t, Type_element_t, Type_element_t>::exlwte(
                c[n], a[n], beta_beta);
        }
    }
};

template <int32_t N> struct Apply_beta<Data_type_fp16, uint2, N> {
    public:
    using Type_element_t = Data_type_fp16;
    using Type_out_t = uint2;
    __device__ inline static void exlwte(Type_out_t (&c)[N], Type_out_t a[N], uint32_t beta)
    {
        float float_beta = __uint_as_float(beta);
        uint32_t beta_beta = xmma::float2_to_half2(float_beta, float_beta);
#pragma unroll
        for (int32_t n = 0; n < N; ++n) {
            c[n].x = Multiply_and_add<Type_element_t, Type_element_t, Type_element_t>::exlwte(
                c[n].x, a[n].x, beta_beta);
            c[n].y = Multiply_and_add<Type_element_t, Type_element_t, Type_element_t>::exlwte(
                c[n].y, a[n].y, beta_beta);
        }
    }
};

template <int32_t N> struct Apply_beta<Data_type_fp16, uint4, N> {
    public:
    using Type_element_t = Data_type_fp16;
    using Type_out_t = uint4;
    __device__ inline static void exlwte(Type_out_t (&c)[N], Type_out_t a[N], uint32_t beta)
    {
        float float_beta = __uint_as_float(beta);
        uint32_t beta_beta = xmma::float2_to_half2(float_beta, float_beta);
#pragma unroll
        for (int32_t n = 0; n < N; ++n) {
            c[n].x = Multiply_and_add<Type_element_t, Type_element_t, Type_element_t>::exlwte(
                c[n].x, a[n].x, beta_beta);
            c[n].y = Multiply_and_add<Type_element_t, Type_element_t, Type_element_t>::exlwte(
                c[n].y, a[n].y, beta_beta);
            c[n].z = Multiply_and_add<Type_element_t, Type_element_t, Type_element_t>::exlwte(
                c[n].z, a[n].z, beta_beta);
            c[n].w = Multiply_and_add<Type_element_t, Type_element_t, Type_element_t>::exlwte(
                c[n].w, a[n].w, beta_beta);
        }
    }
};

template <typename Packed_type_, typename Type_element_, int32_t N> struct Atomic_add {
    public:
    using Packed_type_t = Packed_type_;
    using Type_element_t = typename Type_element_::Type;
    __device__ inline static void exlwte(void *address[N], Packed_type_t data[N]);
};

template <int32_t N> struct Atomic_add<uint4, Data_type_fp32, N> {
    public:
    using Packed_type_t = uint4;
    using Type_element_t = typename Data_type_fp32::Type;
    __device__ inline static void exlwte(void *address[N], Packed_type_t data[N])
    {
        Type_element_t *type_element_address[N];
#pragma unroll
        for (int32_t n = 0; n < N; ++n) {
            type_element_address[n] = static_cast<Type_element_t *>(address[n]);
        }
#pragma unroll
        for (int32_t n = 0; n < N; ++n) {
            atomicAdd(type_element_address[n] + 0, __uint_as_float(data[n].x));
            atomicAdd(type_element_address[n] + 1, __uint_as_float(data[n].y));
            atomicAdd(type_element_address[n] + 2, __uint_as_float(data[n].z));
            atomicAdd(type_element_address[n] + 3, __uint_as_float(data[n].w));
        }
    }
};

template <int32_t N> struct Atomic_add<uint32_t, Data_type_fp16, N> {
    public:
    using Packed_type_t = uint32_t;
    using Type_element_t = __half2;
    __device__ inline static void exlwte(void *address[N], Packed_type_t data[N])
    {
        Type_element_t *type_element_address[N];
#pragma unroll
        for (int32_t n = 0; n < N; ++n) {
            type_element_address[n] = static_cast<Type_element_t *>(address[n]);
        }
#pragma unroll
        for (int32_t n = 0; n < N; ++n) {
            atomicAdd(type_element_address[n], uint32_as_half2(data[n]));
        }
    }
};

template <int32_t N> struct Atomic_add<uint2, Data_type_fp16, N> {
    public:
    using Packed_type_t = uint2;
    using Type_element_t = __half2;
    __device__ inline static void exlwte(void *address[N], Packed_type_t data[N])
    {
        Type_element_t *type_element_address[N];
#pragma unroll
        for (int32_t n = 0; n < N; ++n) {
            type_element_address[n] = static_cast<Type_element_t *>(address[n]);
        }
#pragma unroll
        for (int32_t n = 0; n < N; ++n) {
            atomicAdd(type_element_address[n] + 0, uint32_as_half2(data[n].x));
            atomicAdd(type_element_address[n] + 1, uint32_as_half2(data[n].y));
        }
    }
};

template <int32_t N> struct Atomic_add<uint4, Data_type_fp16, N> {
    public:
    using Packed_type_t = uint4;
    using Type_element_t = __half2;
    __device__ inline static void exlwte(void *address[N], Packed_type_t data[N])
    {
        Type_element_t *type_element_address[N];
#pragma unroll
        for (int32_t n = 0; n < N; ++n) {
            type_element_address[n] = static_cast<Type_element_t *>(address[n]);
        }
#pragma unroll
        for (int32_t n = 0; n < N; ++n) {
            atomicAdd(type_element_address[n] + 0, uint32_as_half2(data[n].x));
            atomicAdd(type_element_address[n] + 1, uint32_as_half2(data[n].y));
            atomicAdd(type_element_address[n] + 2, uint32_as_half2(data[n].z));
            atomicAdd(type_element_address[n] + 3, uint32_as_half2(data[n].w));
        }
    }
};

template <int32_t BOUNDARY_DEPTH, int32_t BOUNDARY_HEIGHT, int32_t BOUNDARY_WIDTH>
__device__ inline void increase(int32_t &out_index_depth,
                                int32_t &out_index_height,
                                int32_t &out_index_width,
                                const int32_t in_index_depth,
                                const int32_t in_index_height,
                                const int32_t in_index_width)
{
    out_index_width = in_index_width + 1;
    out_index_height = in_index_height;
    out_index_depth = in_index_depth;
    if (out_index_width == BOUNDARY_WIDTH) {
        out_index_width = 0;
        ++out_index_height;
        if (out_index_height == BOUNDARY_HEIGHT) {
            out_index_height = 0;
            ++out_index_depth;
            if (out_index_depth == BOUNDARY_DEPTH) {
                out_index_depth = 0;
            }
        }
    }
}

__host__ __device__ inline void *move_pointer(void *in, int32_t offset_in_bytes)
{
    uint8_t *tmp = static_cast<uint8_t *>(in);
    tmp += offset_in_bytes;
    return static_cast<void *>(tmp);
}

template <int32_t FACTOR>
__device__ inline void divmod(int32_t &div, int32_t &mod, const int32_t in)
{
    div = in / FACTOR;
    mod = in % FACTOR;
}

template <int32_t N, int32_t STEP, int32_t TILE_HEIGHT, int32_t TILE_WIDTH, int32_t TILE_GROUP>
__device__ inline void linear_mapping(int32_t (&offset_depth)[N],
                                      int32_t (&offset_height)[N],
                                      int32_t (&offset_width)[N],
                                      int32_t (&offset_group)[N],
                                      const int32_t linear_index)
{
    constexpr int32_t HWG = TILE_HEIGHT * TILE_WIDTH * TILE_GROUP;
    constexpr int32_t WG = TILE_WIDTH * TILE_GROUP;
#pragma unroll
    for (int32_t i = 0; i < N; ++i) {
        int32_t offset_height_width_group;
        divmod<HWG>(offset_depth[i], offset_height_width_group, linear_index + i * STEP);
        int32_t offset_width_group;
        divmod<WG>(offset_height[i], offset_width_group, offset_height_width_group);
        divmod<TILE_GROUP>(offset_width[i], offset_group[i], offset_width_group);
    }
}

template <int32_t STRIDE_0, int32_t STRIDE_1, int32_t STRIDE_2>
__device__ inline int32_t
get_linear_index(int32_t base, int32_t index_0, int32_t index_1, int32_t index_2)
{
    return base + index_0 * STRIDE_0 + index_1 * STRIDE_1 + index_2 * STRIDE_2;
}

template <int32_t STRIDE_0, int32_t STRIDE_1, int32_t STRIDE_2>
__device__ inline uint32_t
get_linear_index(uint32_t base, int32_t index_0, int32_t index_1, int32_t index_2)
{
    return base + index_0 * STRIDE_0 + index_1 * STRIDE_1 + index_2 * STRIDE_2;
}

template <int32_t N, int32_t STRIDE_0, int32_t STRIDE_1, int32_t STRIDE_2>
__device__ inline void get_linear_index(int32_t (&linear_index)[N],
                                        int32_t base,
                                        int32_t index_0[N],
                                        int32_t index_1[N],
                                        int32_t index_2[N])
{
#pragma unroll
    for (int32_t i = 0; i < N; ++i) {
        linear_index[i] = get_linear_index<STRIDE_0, STRIDE_1, STRIDE_2>(
            base, index_0[i], index_1[i], index_2[i]);
    }
}

template <int32_t N, int32_t STRIDE_0, int32_t STRIDE_1, int32_t STRIDE_2>
__device__ inline void get_linear_index(uint32_t (&linear_index)[N],
                                        int32_t base,
                                        int32_t index_0[N],
                                        int32_t index_1[N],
                                        int32_t index_2[N])
{
#pragma unroll
    for (int32_t i = 0; i < N; ++i) {
        linear_index[i] = get_linear_index<STRIDE_0, STRIDE_1, STRIDE_2>(
            base, index_0[i], index_1[i], index_2[i]);
    }
}

template <int32_t N, int32_t STRIDE_0, int32_t STRIDE_1, int32_t STRIDE_2>
__device__ inline void get_linear_index(uint32_t (&linear_index)[N],
                                        uint32_t base,
                                        int32_t index_0[N],
                                        int32_t index_1[N],
                                        int32_t index_2[N])
{
#pragma unroll
    for (int32_t i = 0; i < N; ++i) {
        linear_index[i] = get_linear_index<STRIDE_0, STRIDE_1, STRIDE_2>(
            base, index_0[i], index_1[i], index_2[i]);
    }
}

__device__ inline void set_memory_no_alias(){
#if defined( __LWDA_ARCH__ ) && __LWDA_ARCH__ >= 800
    asm volatile (".pragma \"set knob SchedMemNoAlias=LDS+LDGSTS\";\n" : : : "memory");
#endif
}

__device__ inline void reset_memory_no_alias(){
#if defined( __LWDA_ARCH__ ) && __LWDA_ARCH__ >= 800
    asm volatile (".pragma \"reset knob SchedMemNoAlias=LDS+LDGSTS\";\n" : : : "memory");
#endif
}

enum Grid_dim { TRS = 0, G };

enum Operation { FPROP = 0, DGRAD, WGRAD };

enum Tensor_type { A = 0, B, C, D, COUNT };

#define CEIL_DIV(a, b) (a + b - 1) / b

} // namespace depthwise_colwolution
} // namespace ext
} // namespace xmma

#endif
