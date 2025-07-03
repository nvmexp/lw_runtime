#pragma once

#include <stdio.h>
#include <float.h>
#include <math.h>
#include <assert.h>

#include <algorithm>
#include <type_traits>

#include <lwda_runtime.h>

#include <lwtensor/types.h>
#include <lwtensor/internal/types.h>
#include <lwtensor/internal/util.h>
#include <lwtensor/internal/operators.h>
#include <lwtensor/internal/context.h>
#include <lwtensor/internal/defines.h>

namespace LWTENSOR_NAMESPACE
{

namespace modern
{

/// Represents an operator, either unary or binary, and how to apply it.
struct Operator
{
    using Op = lwtensorOperator_t;

    static constexpr __device__ __inline__ Op fromStatic(Op opStatic, Op op)
    {
        return opStatic == LWTENSOR_OP_UNKNOWN ? op : opStatic;
    }

    __device__ __inline__ Operator(
        Op opStatic, Op op,
        const ElementwiseParameters::ActivationContext* ctx,
        Op opSecondaryStatic = LWTENSOR_OP_UNKNOWN,
        Op opSecondary = LWTENSOR_OP_IDENTITY
    ) : op_(fromStatic(opStatic, op)), ctx_(ctx),
        opSecondary_(fromStatic(opSecondaryStatic, opSecondary))
    {}

    template<typename Compute>
    __device__ __inline__ Compute binary(Compute a, Compute b) const
    {
        return lwtensorBinaryOp<Compute>(a, b, op_, ctx_, opSecondary_);
    }

    template<typename Compute>
    __device__ __inline__ Compute unary(Compute a) const
    {
        return lwtensorUnaryOp<Compute>(a, op_, ctx_);
    }

    const Op op_;
    const ElementwiseParameters::ActivationContext* ctx_;
    const Op opSecondary_;
};

/// Represents an operator and a scaling opertation.
template<typename Compute>
struct ScaleOperator
{

    __device__ __inline__ ScaleOperator(
        const Operator base, Compute scale
    ) : base_(base), scale_(scale), isZero_(lwIsEqual(scale, lwGet<Compute>(0)))
    {}
   
    __device__ __inline__ Compute unary(Compute a) const
    {
        return isZero_ ? lwGet<Compute>(0) : lwMul<Compute>(scale_, base_.unary(a));
    }

    __device__ __inline__ bool needLoad() const { return isZero_; }

    const Operator base_;
    const Compute scale_;
    const bool isZero_;
};

/// A compile time tuple of values.
template<typename T, T... kElements>
struct MetaTuple {
    using type = T;
};

/// A runtime tuple of values.
template<typename T, int kRank_>
struct Tuple
{
    static const int kRank = kRank_;
    T values[kRank];
    /// Add `value` to the `at` element of the tuple
    __device__ __inline__ Tuple add(int at, int value)
    {
        Tuple result = *this;
        result.values[at] += value;
        return result;
    }
    /// Switch elements `a` and `b` of the tuple if in range
    __device__ __inline__ Tuple transpose(int a, int b)
    {
        assert(a >= 0 && b >= 0 && a < kRank && b < kRank);
        Tuple result = *this;
        result.values[a] = values[b];
        result.values[b] = values[a];
        return result;
    }
    __device__ __inline__ Tuple() {}
    __device__ __inline__ Tuple(const T* memory)
    {
        for (int i = 0; i < kRank; i++)
        {
            values[i] = memory[i];
        }
    }
    __device__ __inline__ Tuple(std::initializer_list<T> init)
    {
        int i = 0;
        for (T elem : init)
        {
            values[i++] = elem;
        }
    }
};

/// Add 1 to the first element of a MetaTuple.
/// Used to get the correct stride for shared memory
template<typename Layout> struct AddOneFirst;
template<typename T, T kFirst, T...kRest>
struct AddOneFirst<MetaTuple<T, kFirst, kRest...>>
{
    using type = MetaTuple<T, kFirst + 1, kRest...>;
};

/// Product of elements of a MetaTuple.
template<typename T> struct Product;
template<typename T, T kFirst, T...kRest>
struct Product<MetaTuple<T, kFirst, kRest...>>
{
    static const T value = kFirst * Product<MetaTuple<T, kRest...>>::value;
};
template<typename T>
struct Product<MetaTuple<T>>
{
    static const T value = 1;
};

/// Prepend `kElement` to a MetaTuple `T`.
template<typename T, typename E, E kElement> struct Prepend;
template<typename T, T kElement, T...kElements>
struct Prepend<MetaTuple<T, kElements...>, T, kElement>
{
    using type = MetaTuple<T, kElement, kElements...>;
};

/// Compute the prefix product (exclusive product scan) of a MetaTuple.
/// Used to callwlate strides for shared memory
template<typename T, typename E = typename T::type, E kPrefix = 1> struct PrefixProduct;
template<typename T, T kPrefix, T kFirst, T...kRest>
struct PrefixProduct<MetaTuple<T, kFirst, kRest...>, T, kPrefix>
{
    using Rest = MetaTuple<T, kRest...>;
    using Relwrse = typename PrefixProduct<Rest, T, kPrefix * kFirst>::type;
    using type = typename Prepend<Relwrse, T, kPrefix>::type;
};
template<typename T, T kPrefix>
struct PrefixProduct<MetaTuple<T>, T, kPrefix>
{
    using type = MetaTuple<T>;
};

/// Rank of a MetaTuple.
template<typename T> struct Rank;
template<typename T, T kFirst, T...kRest>
struct Rank<MetaTuple<T, kFirst, kRest...>>
{
    static const T value = 1 + Rank<MetaTuple<T, kRest...>>::value;
};
template<typename T>
struct Rank<MetaTuple<T>>
{
    static const T value = 0;
};

/// Extract an element at position `A` from MetaTuple `T`.
/// Also supports dynamically extracting elements.
template<int kIndex, typename T, typename Enable = void> struct At;
template<typename T, T kFirst, T...kRest>
struct At<0, MetaTuple<T, kFirst, kRest...>, void>
{
    static const T value = kFirst;
    static constexpr T __host__ __device__ __inline__ get(int i)
    {
        return i == 0 ? kFirst : T{};
    }
};
template<int kIndex, typename T, T kFirst, T...kRest>
struct At<kIndex, MetaTuple<T, kFirst, kRest...>, typename std::enable_if<kIndex != 0>::type>
{
    static const T value = At<kIndex - 1, MetaTuple<T, kRest...>>::value;
    static constexpr __host__ __device__ __inline__ T get(int i)
    {
        return i == 0 ? kFirst : At<kIndex - 1, MetaTuple<T, kRest...>>::get(i - 1);
    }
};

/// Dynamically extract from MetaTuple.
template<typename T>
constexpr  __host__ __device__ __inline__ typename T::type at(int i)
{
    return At<Rank<T>::value - 1, T>::get(i);
}

/// Transpose first two elements of a MetaTuple.
/// Again, used for shared memory layout callwlation.
template<typename T> struct Transpose;
template<typename T, T kFirst, T kSecond, T...kRest>
struct Transpose<MetaTuple<T, kFirst, kSecond, kRest...>>
{
    using type = MetaTuple<T, kSecond, kFirst, kRest...>;
};
template<typename T, T kFirst>
struct Transpose<MetaTuple<T, kFirst>>
{
    using type = MetaTuple<T, kFirst>;
};

/// Represents the shared memory of the elementwise kernel.
/// `T` is the compute type,
/// `Layout` is the tile as a MetaTuple,
/// `Enable` is whether the kernel uses shared memory.
template<typename T, typename Layout, bool kEnable>
struct Smem
{
    static constexpr int kRank = Rank<Layout>::value;
    using TransposeLayout = typename Transpose<Layout>::type;
    using Extents = typename AddOneFirst<TransposeLayout>::type;
    using Strides = typename PrefixProduct<Extents>::type;
    static const int kSize = kEnable ? Product<Extents>::value : 1;
    // The below code for some reason does not work.
    // Workaround: Expose size and have pointer passed in.
    // T memory_[kSize];
    T* memory_;
    template<int kIndex = 0, typename std::enable_if<kIndex != kRank, int>::type = 0>
    __device__ __forceinline__ typename Layout::type index_(const Tuple<typename Layout::type, kRank>& idx)
    {
       return index_<kIndex+1>(idx) + idx.values[kIndex] * At<kIndex, Strides>::value;
    }
    template<int kIndex = 0, typename std::enable_if<kIndex == Rank<Layout>::value, int>::type = 0>
    __device__ __forceinline__  typename Layout::type index_(const Tuple<typename Layout::type, Rank<Layout>::value>& idx)
    {
        return 0;
    }
    /// Index shared memory using a Tuple.
    __device__ __inline__ T& index(const Tuple<typename Layout::type, Rank<Layout>::value>& idx)
    {
       return memory_[index_(idx)];
    }
};

/// Represent an n-dimensional tensor with dynamic strides.
template<typename T, int kRank_, typename S>
struct Tensor
{
    static constexpr int kRank = kRank_;

    T* memory_;
    S ld_[kRank];

    /// Index tensor using Tuple.
    template<typename E>
    __device__ __inline__ T& index(const Tuple<E, kRank>& idx)
    {
        S index_ = 0;
        for (int i = 0; i < kRank; i++)
        {
            index_ += ld_[i] * idx.values[i];
        }
        return memory_[index_];
    }

    __device__ __inline__ Tensor(T* memory, const S* ld) : memory_(memory)
    {
        for (int i = 0; i < kRank; i++)
        {
            ld_[i] = ld[i];
        }
    }

    __device__ __inline__ Tensor(T* memory, std::initializer_list<S> ld) : memory_(memory)
    {
        int i = 0;
        for (auto l : ld)
        {
            ld_[i++] = l;
        }
    }

    __device__ __inline__ Tensor(T* memory, Tuple<S, kRank> ld) : memory_(memory)
    {
        for (int i = 0; i < kRank; i++)
        {
            ld_[i] = ld.values[i];
        }
    }
};

/// Compute minimum of two values.
template<typename T, T kA, T kB>
struct Min
{
    static constexpr T value = kA < kB ? kA : kB;
};

/// Thread layouts, specialized by size.
template<typename Tile, int kNumThreads, int kVec, typename Full, int kVecDim, typename Enable = void>
struct ThreadLayout;

/// A 3D layout tiles threads first along `kVecDim`, and then along the first free dim, and then along the last free dim.
template<typename Tile, int kNumThreads, int kVec, typename Full, int kVecDim>
struct ThreadLayout<Tile, kNumThreads, kVec, Full, kVecDim, typename std::enable_if<Rank<Tile>::value == 3>::type>
{

    static constexpr int kRank = Rank<Tile>::value;
    static constexpr int kNolwecDim0 = kVecDim == 0 ? 1 : 0;
    static constexpr int kNolwecDim1 = kVecDim == 1 ? 2 : 2;
    static constexpr int kIncrement = (kNumThreads * kVec) / At<kVecDim, Tile>::value;

    static constexpr int kThreadsAlongVec = At<kVecDim, Tile>::value / kVec;
    static constexpr int kThreadsAlongNolwec0 = Min<int, At<kNolwecDim0, Tile>::value, kNumThreads / kThreadsAlongVec>::value;
    static constexpr int kThreadsAlongNolwec1 = kNumThreads / (kThreadsAlongVec * kThreadsAlongNolwec0);
    static_assert(kThreadsAlongNolwec1 <= At<kNolwecDim1, Tile>::value, "Tile is too small for number of threads");
    static_assert(kThreadsAlongVec <= kNumThreads, "Not enough threads for this layout");

    static constexpr int kTripsAlongNolwec0 = At<kNolwecDim0, Tile>::value / kThreadsAlongNolwec0;
    static constexpr int kTripsAlongNolwec1 = At<kNolwecDim1, Tile>::value / kThreadsAlongNolwec1;


    __device__ __inline__ static constexpr int count()
    {
        return Product<Tile>::value / kNumThreads / kVec;
    }

    __device__ __inline__ bool predicate(int i)
    {
        int i1 = i / kTripsAlongNolwec0;
        return (At<kNolwecDim1, Full>::value) || ((i1 * kThreadsAlongNolwec1) < limit_.values[kNolwecDim1]);
    }

    __device__ __inline__ bool skip(int i)
    {
        if (At<kNolwecDim0, Full>::value) return false;
        int i0 = i % kTripsAlongNolwec0;
        return (i0 * kThreadsAlongNolwec0 >= limit_.values[kNolwecDim0]);
    }

    __device__ __inline__ Tuple<int, kRank> at(int i)
    {
        int i0 = i % kTripsAlongNolwec0;
        int i1 = i / kTripsAlongNolwec0;
        Tuple<int, kRank> result;
        result.values[0] = offset_.values[kVecDim];
        result.values[1] = offset_.values[kNolwecDim0] + i0 * kThreadsAlongNolwec0;
        result.values[2] = offset_.values[kNolwecDim1] + i1 * kThreadsAlongNolwec1;
        return result;
    }

    __device__ __inline__ ThreadLayout(unsigned int thread, Tuple<int, kRank> limit)
    {
        auto pos = static_cast<int>(thread) * kVec;
        offset_.values[kVecDim] = pos % At<kVecDim, Tile>::value;
        limit_.values[kVecDim] = 0;
        pos /= At<kVecDim, Tile>::value;
        #pragma unroll
        for (int i = 0; i < kRank; i++)
        {
            if (i == kVecDim) continue;
            offset_.values[i] = pos % modern::at<Tile>(i);
            pos /= modern::at<Tile>(i);
            limit_.values[i] = limit.values[i] - offset_.values[i];
        }
    }

    Tuple<int, kRank> offset_;
    Tuple<int, kRank> limit_;
};

/// A 2D layout tiles thread along dimension `kVecDim`.
/// The first element of the returned tuple is always the vectorized one.
template<typename Tile, int kNumThreads, int kVec, typename Full, int kVecDim>
struct ThreadLayout<Tile, kNumThreads, kVec, Full, kVecDim, typename std::enable_if<Rank<Tile>::value == 2>::type>
{

    static constexpr int kRank = Rank<Tile>::value;
    static constexpr int kNolwecDim = 1 - kVecDim;
    static constexpr int kIncrement = (kNumThreads * kVec) / At<kVecDim, Tile>::value;

    __device__ __inline__ static constexpr int count()
    {
        return Product<Tile>::value / kNumThreads / kVec;
    }

    __device__ __inline__ bool predicate(int i)
    {
        return ((At<kNolwecDim, Full>::value) || ((i * kIncrement) < limit_));
    }

    __device__ __inline__ bool skip(int i)
    {
        return false;
    }

    __device__ __inline__ Tuple<int, kRank> at(int i)
    {
        return {offset_.values[0], offset_.values[1] + i * kIncrement};
    }

    __device__ __inline__ ThreadLayout(unsigned int thread, Tuple<int, kRank> limit)
    : offset_({(static_cast<int>(thread)*kVec) % At<kVecDim, Tile>::value, (static_cast<int>(thread)*kVec) / At<kVecDim, Tile>::value})
    , limit_(limit.values[kNolwecDim] - offset_.values[1])
    {}

    Tuple<int, kRank> offset_;
    int limit_;
};

/// A 1D layout tiles threads along that one dimension.
template<typename Tile, int kNumThreads, int kVec, typename Full, int kVecDim>
struct ThreadLayout<Tile, kNumThreads, kVec, Full, kVecDim, typename std::enable_if<Rank<Tile>::value == 1>::type>
{

    static constexpr int kRank = Rank<Tile>::value;

    __device__ __inline__ static constexpr int count()
    {
        return Product<Tile>::value / kNumThreads / kVec;
    }

    __device__ __inline__ bool predicate(int i)
    {
        return true;
    }

    __device__ __inline__ bool skip(int i)
    {
        return false;
    }

    __device__ __inline__ Tuple<int, kRank> at(int i)
    {
        return {offset_.values[0] + i * (kNumThreads*kVec)};
    }

    __device__ __inline__ ThreadLayout(unsigned int thread, Tuple<int, kRank> limit)
    : offset_({static_cast<int>(thread)*kVec})
    {}

    Tuple<int, kRank> offset_;
};

/// Construct a MetaTuple that repeats a value `value` `N` times.
template<typename T, int N, T value, typename Enable = void> struct Repeat;
template<typename T, T value>
struct Repeat<T, 0, value, void>
{
    using type = MetaTuple<T>;
};
template<typename T, int N, T value>
struct Repeat<T, N, value, typename std::enable_if<N != 0>::type>
{
    using type = typename Prepend<typename Repeat<T, N-1, value>::type, T, value>::type;

};

/// Construct a MetaTuple off the first `N` elements of MetaTuple `Tuple`.
template<int N, typename Tuple, typename Enable = void> struct Take;
template<typename T, T... values>
struct Take<0, MetaTuple<T, values...>, void>
{
    using type = MetaTuple<T>;
};
template<int N, typename T, T value, T... values>
struct Take<N, MetaTuple<T, value, values...>, typename std::enable_if<N != 0>::type>
{
    using type = typename Prepend<typename Take<N-1, MetaTuple<T, values...>>::type, T, value>::type;
};

/// Checks if a blocking is complete, i.e. if all elements of `limit` are larger or equal than `Blocking`.
template<typename Blocking, typename Limit>
__device__ __inline__ bool full_blocking(Limit limit)
{
    bool result = true;
    #pragma unroll
    for (int i = 0; i < Limit::kRank; i++)
    {
        result = result && (at<Blocking>(i) <= limit.values[i]);
    }
    return result;
}

}  // namespace modern

}  // LWTENSOR_NAMESPACE
