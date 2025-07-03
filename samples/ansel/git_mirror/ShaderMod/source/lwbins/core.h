/*
 * Copyright (c) 2018, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 */

#ifndef CORE_H
#define CORE_H

#ifdef Q_CREATOR_RUN
#include <lwda_runtime.h>
#define __LWDACC__
#include <device_functions.h>
#include <device_launch_parameters.h>
#include <sys/types.h>
#endif

#include <lwda_fp16.h>
#include <device_types.h>
#include <math.h>
#include <vector_types.h>

#define LWDART_CHECK(x)                                                                          \
    {                                                                                            \
        lwdaError_t rval;                                                                        \
        if ((rval = (x)) != lwdaSuccess)                                                         \
        {                                                                                        \
            const char *error_str = lwdaGetErrorString(rval);                                    \
            std::printf("%s():%i: LWCA API error: \"%s\"\n", __FUNCTION__, __LINE__, error_str); \
            exit(0);                                                                             \
        }                                                                                        \
    }

#define LWDADRV_CHECK(x)                                                                                \
    {                                                                                                   \
        LWresult rval;                                                                                  \
        if ((rval = (x)) != LWDA_SUCCESS)                                                               \
        {                                                                                               \
            const char *error_str;                                                                      \
            lwGetErrorString(rval, &error_str);                                                         \
            std::printf("%s():%i: LWCA driver API error: \"%s\"\n", __FUNCTION__, __LINE__, error_str); \
            exit(0);                                                                                    \
        }                                                                                               \
    }

namespace lw {

using uchar = unsigned char;

template<typename T>
class DeviceBuffer;

constexpr uint kWarpSizeShift = 5;
constexpr uint kWarpSize      = 1 << kWarpSizeShift;
constexpr uint kWarpSizeMask  = kWarpSize - 1;

// compile-time GCD & LCM
template<int A, int B, int... Args>
struct GCD
{
    static constexpr int kValue = GCD<GCD<A, B>::kValue, Args...>::kValue;
};

template<int A, int B>
struct GCD<A, B>
{
    static constexpr int kValue = GCD<B, A % B>::kValue;
};
template<int A>
struct GCD<A, 0>
{
    static constexpr int kValue = A;
};
template<int A>
struct GCD<0, A>
{
    static constexpr int kValue = A;
};
template<>
struct GCD<0, 0>
{
};

template<int A, int B, int... Args>
struct LCM
{
    static constexpr int kValue = LCM<LCM<A, B>::kValue, Args...>::kValue;
};

template<int A, int B>
struct LCM<A, B>
{
    static constexpr int kValue = A * B / GCD<A, B>::kValue;
};

// compile-time max
template<int A, int B, int... Args>
struct Max
{
    static constexpr int kValue = Max<Max<A, B>::kValue, Args...>::kValue;
};

template<int A, int B>
struct Max<A, B>
{
    static constexpr int kValue = A > B ? A : B;
};

// compile-time min
template<int A, int B, int... Args>
struct Min
{
    static constexpr int kValue = Min<Min<A, B>::kValue, Args...>::kValue;
};

template<int A, int B>
struct Min<A, B>
{
    static constexpr int kValue = A < B ? A : B;
};

// compile-time round-up
template<int N, int M>
struct RoundUp
{
    static constexpr int kValue = ((N + M - 1) / M) * M;
};

// compile-time round-down
template<int N, int M>
struct RoundDown
{
    static constexpr int kValue = (N / M) * M;
};

// compile-time round up an integer to warp size +/- offset
template<int N, int O = 0>
struct WarpAlignedSize
{
    static constexpr int kValue = RoundUp<N, kWarpSize>::kValue + (RoundUp<N, kWarpSize>::kValue - O > N ? -O : O);
};

// Packed data types helper
template<typename T, int N>
struct Pack
{
    static_assert(true, "unsupported pack type!");
};

template<>
struct Pack<float, 1>
{
    using DataType = uint;
    __device__ void clear()
    {
        raw_data = 0;
    }
    DataType raw_data;
};

template<>
struct Pack<float, 2>
{
    using DataType = uint2;
    __device__ void clear()
    {
        raw_data = {0, 0};
    }
    DataType raw_data;
};

template<>
struct Pack<float, 4>
{
    using DataType = uint4;
    __device__ void clear()
    {
        raw_data = {0, 0, 0, 0};
    }
    DataType raw_data;
};

template<>
struct Pack<half, 1>
{
    using DataType = ushort;
    __device__ void clear()
    {
        raw_data = 0;
    }
    DataType raw_data;
};

template<>
struct Pack<half, 2>
{
    using DataType = uint;
    __device__ void clear()
    {
        raw_data = 0;
    }
    DataType raw_data;
};

template<>
struct Pack<half, 4>
{
    using DataType = uint2;
    __device__ void clear()
    {
        raw_data = {0, 0};
    }
    DataType raw_data;
};

template<>
struct Pack<half, 8>
{
    using DataType = uint4;
    __device__ void clear()
    {
        raw_data = {0, 0, 0, 0};
    }
    DataType raw_data;
};

template<>
struct Pack<uchar, 1>
{
    using DataType = uchar;
    __device__ void clear()
    {
        raw_data = 0;
    }
    DataType raw_data;
};

template<>
struct Pack<uchar, 2>
{
    using DataType = ushort;
    __device__ void clear()
    {
        raw_data = 0;
    }
    DataType raw_data;
};

template<>
struct Pack<uchar, 4>
{
    using DataType = uint;
    __device__ void clear()
    {
        raw_data = 0;
    }
    DataType raw_data;
};

template<>
struct Pack<uchar, 8>
{
    using DataType = uint2;
    __device__ void clear()
    {
        raw_data = {0, 0};
    }
    DataType raw_data;
};

template<>
struct Pack<uchar, 16>
{
    using DataType = uint4;
    __device__ void clear()
    {
        raw_data = {0, 0, 0, 0};
    }
    DataType raw_data;
};

// given the type, what is the maximum vectorized pack size?
template<typename T>
struct MaxPackSize
{
    static constexpr int kValue = 16 / sizeof(T);
};

// 1D array abstraction

template<int N>
class StaticArrayRef;

template<int N>
class __align__(16) StaticArray
{
public:
    static constexpr int kSize = N;

    template<int Offset>
    __device__ StaticArrayRef<N - Offset> offset() const
    {
        return const_cast<uchar1 *>(_data + Offset);
    }

    template<typename T = uchar1>
    __device__ T *data() const
    {
        return const_cast<T *>(reinterpret_cast<const T *>(_data));
    }

private:
    uchar1 _data[N];
};

// specialization for empty arrays
template<>
class StaticArray<0>
{
public:
    static constexpr int kSize = 0;
};

template<int N>
class StaticArrayRef
{
    template<int M>
    friend class StaticArrayRef;
    template<int M>
    friend class StaticArray;

public:
    static constexpr int kSize = N;

    template<int M>
    __device__ StaticArrayRef(const StaticArray<M> &src)
        : _data(src.data())
    {
        static_assert(M >= N, "insufficient space in source array!");
    }

    template<int M>
    __device__ StaticArrayRef(const StaticArrayRef<M> &src)
        : _data(src.data())
    {
        static_assert(M >= N, "insufficient space in source array!");
    }

    template<int M>
    __device__ StaticArrayRef<N - M> offset() const
    {
        static_assert(N >= M, "insufficient space in source array!");
        return _data + M;
    }

    template<typename T = uchar1>
    __device__ T *data() const
    {
        return const_cast<T *>(reinterpret_cast<const T *>(_data));
    }

    template<typename T = uchar1>
    __device__ T *end() const
    {
        return const_cast<T *>(reinterpret_cast<const T *>(_data + kSize));
    }

    __device__ static StaticArrayRef FromPointer(uchar1 *data)
    {
        return {data};
    }

private:
    __device__ StaticArrayRef(uchar1 *data)
        : _data(data)
    {
    }

    uchar1 *__restrict__ _data;
};

// tile buffer abstractions
template<typename T>
class TileBuffer
{
public:
    using PixelType = T;

    __device__ TileBuffer(const DeviceBuffer<T> &buf, int sx = 0, int sy = 0)
        : _stride(buf.stride)
        , _data(const_cast<T *>(&buf(sx, sy)))
    {
    }

    __device__ TileBuffer(const T *buf, int stride, int sx = 0, int sy = 0)
        : _stride(stride)
        , _data(const_cast<T *>(&buf[sy * stride + sx]))
    {
    }

    __device__ T &operator()(int x, int y = 0)
    {
        return _data[y * _stride + x];
    }
    __device__ const T &operator()(int x, int y = 0) const
    {
        return _data[y * _stride + x];
    }

private:
    const int _stride;
    T *__restrict__ _data;
};

template<typename T, int S0, int E0, int S1 = 0, int E1 = 1, int S2 = 0, int E2 = 1, int S3 = 0, int E3 = 1>
class TileBufferStatic
{
    static_assert(E0 > 0 && E1 > 0 && E2 > 0 && E3 > 0, "invalid buffer extents!");

public:
    using PixelType = T;

    static constexpr int kSize = Max<S0 * E0, E1 * S1, E2 * S2, E3 * S3>::kValue * sizeof(T);

    static constexpr int kExtent0 = E0;
    static constexpr int kExtent1 = E1;
    static constexpr int kExtent2 = E2;
    static constexpr int kExtent3 = E3;

    static constexpr int kStride0 = S0;
    static constexpr int kStride1 = S1;
    static constexpr int kStride2 = S2;
    static constexpr int kStride3 = S3;

    __device__ TileBufferStatic(T *buf, int sx = 0, int sy = 0, int sz = 0, int sw = 0)
    {
        reset(buf, sx, sy, sz, sw);
    }

    __device__ TileBufferStatic(const T *buf, int sx = 0, int sy = 0, int sz = 0, int sw = 0)
    {
        reset(buf, sx, sy, sz, sw);
    }

    template<int N>
    __device__ TileBufferStatic(const StaticArray<N> &buf, int sx = 0, int sy = 0, int sz = 0, int sw = 0)
    {
        reset(buf, sx, sy, sz, sw);
    }

    template<int N>
    __device__ TileBufferStatic(const StaticArrayRef<N> &buf, int sx = 0, int sy = 0, int sz = 0, int sw = 0)
    {
        reset(buf, sx, sy, sz, sw);
    }

    __device__ static constexpr int Offset(int sx, int sy = 0, int sz = 0, int sw = 0)
    {
        return sw * S3 + sz * S2 + sy * S1 + sx * S0;
    }

    __device__ static constexpr int Offset(const int2 &p)
    {
        return p.y * S1 + p.x * S0;
    }

    __device__ static constexpr int ByteOffset(int sx, int sy = 0, int sz = 0, int sw = 0)
    {
        return sw * (S3 * sizeof(T)) + sz * (S2 * sizeof(T)) + sy * (S1 * sizeof(T)) + sx * (S0 * sizeof(T));
    }

    __device__ static constexpr int ByteOffset(const int2 &p)
    {
        return p.y * (S1 * sizeof(T)) + p.x * (S0 * sizeof(T));
    }

    __device__ static constexpr int ElemCount()
    {
        return E0 * E1 * E2 * E3;
    }

    __device__ constexpr int offset(int sx, int sy = 0, int sz = 0, int sw = 0) const
    {
        return Offset(sx, sy, sz, sw);
    }

    __device__ constexpr int offset(const int2 &p) const
    {
        return Offset(p);
    }

    __device__ constexpr int byteOffset(int sx, int sy = 0, int sz = 0, int sw = 0) const
    {
        return ByteOffset(sx, sy, sz, sw);
    }

    __device__ constexpr int byteOffset(const int2 &p) const
    {
        return ByteOffset(p);
    }

    __device__ T &operator()(int x, int y = 0, int z = 0, int w = 0)
    {
        return _data[Offset(x, y, z, w)];
    }

    __device__ const T &operator()(int x, int y = 0, int z = 0, int w = 0) const
    {
        return _data[Offset(x, y, z, w)];
    }

    __device__ T &operator()(const int2 &p)
    {
        return _data[Offset(p)];
    }

    __device__ const T &operator()(const int2 &p) const
    {
        return _data[Offset(p)];
    }

    __device__ T &operator[](int i)
    {
        return _data[i];
    }

    __device__ const T &operator[](int i) const
    {
        return _data[i];
    }

    __device__ uchar &byteAt(int i)
    {
        return reinterpret_cast<uchar *>(_data)[i];
    }

    __device__ const uchar &byteAt(int i) const
    {
        return reinterpret_cast<const uchar *>(_data)[i];
    }

    __device__ TileBufferStatic shift(int sx, int sy = 0, int sz = 0, int sw = 0) const
    {
        return {_data, sx, sy, sz, sw};
    }

    __device__ T *data() const
    {
        return _data;
    }

    __device__ void reset(T *buf, int sx = 0, int sy = 0, int sz = 0, int sw = 0)
    {
        _data = &buf[Offset(sx, sy, sz, sw)];
    }

    __device__ void reset(const T *buf, int sx = 0, int sy = 0, int sz = 0, int sw = 0)
    {
        _data = const_cast<T *>(&buf[Offset(sx, sy, sz, sw)]);
    }

    template<int N>
    __device__ void reset(const StaticArray<N> &buf, int sx = 0, int sy = 0, int sz = 0, int sw = 0)
    {
        static_assert(StaticArray<N>::kSize >= kSize, "insufficient static storage!");
        _data = buf.template data<T>() + Offset(sx, sy, sz, sw);
    }

    template<int N>
    __device__ void reset(const StaticArrayRef<N> &buf, int sx = 0, int sy = 0, int sz = 0, int sw = 0)
    {
        static_assert(StaticArrayRef<N>::kSize >= kSize, "insufficient static storage!");
        _data = buf.template data<T>() + Offset(sx, sy, sz, sw);
    }

private:
    T *__restrict__ _data;
};

template<typename T>
__device__ T Clamp(T v, T milw, T maxv)
{
    return min(max(v, milw), maxv);
}

template<ushort N>
__device__ typename std::enable_if<(N & (N - 1)) != 0, int2>::type IndexToCoord(ushort index)
{
    static_assert(N > 0, "invalid stride!");
    ushort d = index / N;
    return {index - d * N, d};
}

template<ushort N>
__device__ typename std::enable_if<(N & (N - 1)) == 0, int2>::type IndexToCoord(uint index)
{
    // power-of-two N optimization
    static_assert(N > 0, "invalid stride!");
    return {static_cast<int>(index % N), static_cast<int>(index / N)};
}

template<ushort N>
__device__ int2 CoordStep(const int2 &base, const int2 &step)
{
    int2 rval;
    rval.x = base.x + step.x;
    rval.y = base.y + step.y;
    if (rval.x >= N)
    {
        rval.x -= N;
        rval.y += 1;
    }
    return rval;
}

template<typename T, typename U, typename V>
__device__ void AssignAs(U &u, const V &v)
{
    reinterpret_cast<T &>(u) = reinterpret_cast<const T &>(v);
}

template<typename T>
__device__ float WarpReduceSum(T v)
{
    for (int j = kWarpSize / 2; j > 0; j /= 2)
    {
        v += __shfl_down_sync(0xffffffff, v, j);
    }
    return v;
}

#ifdef __LWDACC__

__forceinline__ __device__ ushort __half2uchar_rn(__half h)
{
    ushort i;
    asm("cvt.rni.u8.f16 %0, %1;" : "=h"(i) : "h"(reinterpret_cast<ushort &>(h)));
    return i;
}

__forceinline__ __device__ ushort __float2uchar_rn(float v)
{
    ushort i;
    asm("cvt.rni.u8.f32 %0, %1;" : "=h"(i) : "f"(v));
    return i;
}

#endif

} // namespace lw

#endif // CORE_H
