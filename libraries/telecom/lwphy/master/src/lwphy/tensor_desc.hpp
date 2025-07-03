/*
 * Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 */

#if !defined(TENSOR_DESC_HPP_INCLUDED_)
#define TENSOR_DESC_HPP_INCLUDED_

#include <array>
#include <experimental/optional>
#include <functional>
#include <numeric>
#include "lwComplex.h"
#include "lwda_fp16.h"
#include "lwphy.h"
#include "lwphy_internal.h"

// Empty struct - used as base class for internal tensor descriptor
struct lwphyTensorDescriptor
{};

////////////////////////////////////////////////////////////////////////////
// get_lwphy_type_storage_element_size()
// Returns the size of the storage element for a given type. In general,
// this is the size of the type used to store the given lwphyDataType_t.
// However, for sub-byte types, multiple elements are stored in a machine
// word. For these types, the returned size is the size of a machine word,
// which stores multiple elements.
// Returns 0 for LWPHY_VOID.
int get_lwphy_type_storage_element_size(lwphyDataType_t type);

////////////////////////////////////////////////////////////////////////////
// type_is_sub_byte()
inline bool type_is_sub_byte(lwphyDataType_t type)
{
    return (LWPHY_BIT == type);
}

////////////////////////////////////////////////////////////////////////////
// get_element_multiple_for_alignment()
// Returns the number of elements required to populate the given byte
// alignment. This can be used to round up tensor dimensions so that
// higher dimensions begin at a given alignment.
// Example: For byteAlign = 128 and type = LWPHY_R_32F (which has a size
// of 4 bytes), this function will return 128 / 4 = 32.
// Example: For byteAlign = 4 and type = LWPHY_BIT, this function will
// return 4 / (1/8) = 32.
int get_element_multiple_for_alignment(int byte_align, lwphyDataType_t type);

////////////////////////////////////////////////////////////////////////////
// data_type_traits
// clang-format off
template <lwphyDataType_t T> struct data_type_traits;
template <>                  struct data_type_traits<LWPHY_VOID>  { typedef void type;            };
template <>                  struct data_type_traits<LWPHY_BIT>   { typedef uint32_t type;        };
template <>                  struct data_type_traits<LWPHY_R_8I>  { typedef signed char type;     };
template <>                  struct data_type_traits<LWPHY_C_8I>  { typedef char2 type;           };
template <>                  struct data_type_traits<LWPHY_R_8U>  { typedef unsigned char type;   };
template <>                  struct data_type_traits<LWPHY_C_8U>  { typedef uchar2 type;          };
template <>                  struct data_type_traits<LWPHY_R_16I> { typedef short type;           };
template <>                  struct data_type_traits<LWPHY_C_16I> { typedef short2 type;          };
template <>                  struct data_type_traits<LWPHY_R_16U> { typedef unsigned short type;  };
template <>                  struct data_type_traits<LWPHY_C_16U> { typedef ushort2 type;         };
template <>                  struct data_type_traits<LWPHY_R_32I> { typedef int type;             };
template <>                  struct data_type_traits<LWPHY_C_32I> { typedef int2 type;            };
template <>                  struct data_type_traits<LWPHY_R_32U> { typedef unsigned int type;    };
template <>                  struct data_type_traits<LWPHY_C_32U> { typedef uint2 type;           };
template <>                  struct data_type_traits<LWPHY_R_16F> { typedef __half type;          };
template <>                  struct data_type_traits<LWPHY_C_16F> { typedef __half2 type;         };
template <>                  struct data_type_traits<LWPHY_R_32F> { typedef float type;           };
template <>                  struct data_type_traits<LWPHY_C_32F> { typedef lwComplex type;       };
template <>                  struct data_type_traits<LWPHY_R_64F> { typedef double type;          };
template <>                  struct data_type_traits<LWPHY_C_64F> { typedef lwDoubleComplex type; };
// clang-format on

////////////////////////////////////////////////////////////////////////
// complex_from_scalar
// clang-format off
template <typename T> struct complex_from_scalar;
template <>           struct complex_from_scalar<signed char>    { typedef char2 type;           };
template <>           struct complex_from_scalar<unsigned char>  { typedef uchar2 type;          };
template <>           struct complex_from_scalar<short>          { typedef short2 type;          };
template <>           struct complex_from_scalar<unsigned short> { typedef ushort2 type;         };
template <>           struct complex_from_scalar<int>            { typedef int2 type;            };
template <>           struct complex_from_scalar<unsigned int>   { typedef uint2 type;           };
template <>           struct complex_from_scalar<__half>         { typedef __half2 type;         };
template <>           struct complex_from_scalar<float>          { typedef lwComplex type;       };
template <>           struct complex_from_scalar<double>         { typedef lwDoubleComplex type; };
// clang-format on

////////////////////////////////////////////////////////////////////////
// scalar_from_complex
// clang-format off
template <typename T> struct scalar_from_complex;
template <>           struct scalar_from_complex<char2>           { typedef signed char type;    };
template <>           struct scalar_from_complex<uchar2>          { typedef unsigned char type;  };
template <>           struct scalar_from_complex<short2>          { typedef short type;          };
template <>           struct scalar_from_complex<ushort2>         { typedef unsigned short type; };
template <>           struct scalar_from_complex<int2>            { typedef int type;            };
template <>           struct scalar_from_complex<uint2>           { typedef unsigned int type;   };
template <>           struct scalar_from_complex<__half2>         { typedef __half type;         };
template <>           struct scalar_from_complex<lwComplex>       { typedef float type;          };
template <>           struct scalar_from_complex<lwDoubleComplex> { typedef double type;         };
// clang-format on

////////////////////////////////////////////////////////////////////////
// make_complex
// clang-format off
template <typename Tcomplex> struct make_complex;
template <>                  struct make_complex<char2>
{
    typedef typename scalar_from_complex<char2>::type Tscalar;
    static LWDA_BOTH char2   create(Tscalar r, Tscalar i)  { return make_char2(r, i); }
};
template <>                  struct make_complex<uchar2>
{
    typedef typename scalar_from_complex<uchar2>::type Tscalar;
    static LWDA_BOTH uchar2  create(Tscalar r, Tscalar i) {return make_uchar2(r, i); }
};
template <>                  struct make_complex<short2>
{
    typedef typename scalar_from_complex<short2>::type Tscalar;
    static LWDA_BOTH short2  create(Tscalar r, Tscalar i) { return make_short2(r, i); }
};
template <>                  struct make_complex<ushort2>
{
    typedef typename scalar_from_complex<ushort2>::type Tscalar;
    static LWDA_BOTH ushort2 create(Tscalar r, Tscalar i) { return make_ushort2(r, i); }
};
template <>                  struct make_complex<int2>
{
    typedef typename scalar_from_complex<int2>::type Tscalar;
    static LWDA_BOTH int2   create(Tscalar r, Tscalar i) { return make_int2(r, i); }
};
template <>                 struct make_complex<uint2>
{
    typedef typename scalar_from_complex<uint2>::type Tscalar;
    static LWDA_BOTH uint2  create(Tscalar r, Tscalar i) { return make_uint2(r, i); }
};
#if defined(__LWDACC__)
template <>                 struct make_complex<struct __half2>
{
    typedef typename scalar_from_complex<struct __half2>::type Tscalar;
    static LWDA_INLINE struct __half2 create(Tscalar r, Tscalar i) { return __halves2half2(r, i); }
};
#endif
template <>                 struct make_complex<lwComplex>
{
    typedef typename scalar_from_complex<lwComplex>::type Tscalar;
    static LWDA_BOTH lwComplex create(Tscalar r, Tscalar i) { return make_lwFloatComplex(r, i); }
};
template <>                 struct make_complex<lwDoubleComplex>
{
    typedef typename scalar_from_complex<lwDoubleComplex>::type Tscalar;
    static LWDA_BOTH lwDoubleComplex create(Tscalar r, Tscalar i) { return make_lwDoubleComplex(r, i); }
};
// clang-format on

////////////////////////////////////////////////////////////////////////
// vec
// Array of elements with a size that is fixed at compile time. Similar
// to std::array, but this class differs in that it has function
// decorations for operation on a GPU device as well as the host.
template <typename T, int Dim>
struct vec
{
public:
    vec() = default;
    // Constructor from list of values with compile time checking for
    // list size.
    // See:
    // https://stackoverflow.com/questions/5438671/static-assert-on-initializer-listsize
    // http://www.open-std.org/jtc1/sc22/wg21/docs/cwg_defects.html#1591
    template <int N>
    vec(const T (&list)[N])
    {
        static_assert(N == Dim, "Initializer list does not have enough dimensions");
        for(int i = 0; i < N; ++i) elem_[i] = list[i];
    }
    LWDA_BOTH_INLINE
    void fill(T val)
    {
        for(int i = 0; i < Dim; ++i) elem_[i] = val;
    }
    LWDA_BOTH_INLINE
    T& operator[](int idx) { return elem_[idx]; }
    LWDA_BOTH_INLINE
    const T& operator[](int idx) const { return elem_[idx]; }
    template <typename TProduct>
    LWDA_BOTH_INLINE auto product() const -> TProduct
    {
        TProduct val = 1;
        for(int i = 0; i < Dim; ++i) val *= elem_[i];
    }
    LWDA_BOTH_INLINE
    T*       begin() { return elem_; }
    T*       end() { return elem_ + Dim; }
    const T* begin() const { return elem_; }
    const T* end() const { return elem_ + Dim; }

private:
    T elem_[Dim];
};

template <typename TResult, typename TVector, int Dim>
TResult dot(const vec<TVector, Dim>& a, const vec<TVector, Dim>& b)
{
    TResult sum = static_cast<TResult>(0);
#pragma unroll
    for(size_t i = 0; i < Dim; ++i)
    {
        sum += (a[i] * b[i]);
    }
    return sum;
}

template <typename T, int Dim>
LWDA_BOTH_INLINE bool operator==(const vec<T, Dim>& a, const vec<T, Dim>& b)
{
    for(int i = 0; i < Dim; ++i)
    {
        if(a[i] != b[i]) return false;
    }
    return true;
}

template <typename T, int Dim>
LWDA_BOTH_INLINE bool operator!=(const vec<T, Dim>& a, const vec<T, Dim>& b)
{
    return !(a == b);
}

////////////////////////////////////////////////////////////////////////
// tensor_layout_N
// Data type-independent description of tensor layout for a fixed (at
// compile time) number of dimensions. Supports both host and device
// usage.
//
// Tensor indexing:
// A tensor contains N dimensions. The size of each dimension is
// given by an integer d(i), and the stride for each dimension is
// given by s(i). We refer to a specific element of the tensor
// using a set of integers n(i), where n(i) < d(i) for each
// i = 0..N.
// Here we are assuming that all sizes d(i) > 0, strides s(i) > 0,
// and s(i) >= d(i) for i = 0..(N-1).
// For a give set of N indices, the offset from the beginning of the
// data (in elements) is given by the dot product:
//
// offset(n) = sum(i=0..(N-1)){ n(i) * s(i) }
//
// The maximum offset can be obtained using the maximum index in each
// dimension n(i)_max = d(i) - 1.
//
// offset_max = sum(i=0..(N-1)){ (d(i) - 1) * s(i) }
template <int TRank>
struct tensor_layout_N
{
    vec<int, TRank> dimensions;
    vec<int, TRank> strides;
    LWDA_BOTH_INLINE
    int rank() const { return TRank; }
    //------------------------------------------------------------------
    // tensor_layout_N()
    LWDA_BOTH_INLINE
    tensor_layout_N() noexcept
    {
        dimensions.fill(0);
        strides.fill(0);
    }
    //------------------------------------------------------------------
    // tensor_layout_N()
    LWDA_BOTH_INLINE
    tensor_layout_N(const int* dim, const int* str) noexcept
    {
        for(int i = 0; i < TRank; ++i)
        {
            dimensions[i] = dim[i];
            strides[i]    = str[i];
        }
    }
    //------------------------------------------------------------------
    // offset()
    LWDA_BOTH_INLINE
    size_t offset(const vec<int, TRank>& indices) const
    {
        return dot(indices, strides);
    }
    //------------------------------------------------------------------
    LWDA_BOTH_INLINE
    size_t offset(const int (&n)[TRank]) const
    {
        size_t idx = 0;
#pragma unroll
        for(int i = 0; i < TRank; ++i)
        {
            idx += n[i] * strides[i];
        }
        return idx;
    }
    //------------------------------------------------------------------
    // get_num_elements()
    LWDA_BOTH_INLINE
    size_t get_num_elements() const noexcept
    {
        // Return the product of the dimensions d(0) * d(1) * ... * d(N-1)
        return dimensions.product();
    }
    //------------------------------------------------------------------
    // get_num_elements_with_strides()
    LWDA_BOTH_INLINE
    size_t get_num_elements_with_strides() const noexcept
    {
        size_t max_size = 0;
        for(int i = 0; i < TRank; ++i)
        {
            max_size = std::max(max_size, static_cast<size_t>(dimensions[i] * strides[i]));
        }
        return max_size;
    }
};

template <int Dim>
LWDA_BOTH_INLINE bool operator==(const tensor_layout_N<Dim>& a, const tensor_layout_N<Dim>& b)
{
    return (a.dimensions == b.dimensions) && (a.strides == b.strides);
}

template <typename T, int Dim>
LWDA_BOTH_INLINE bool operator!=(const tensor_layout_N<Dim>& a, const tensor_layout_N<Dim>& b)
{
    return !(a == b);
}

////////////////////////////////////////////////////////////////////////
// tensor_layout_any
// Data type-independent description of tensor layout. Supports any
// number of dimensions, up to the maximum supported by the library.
// Useful for representing "generic" layouts for kernels and functions
// that handle any number of (legal) dimensions, as well as layouts
// before optional colwersion to dimension-specific handling.
struct tensor_layout_any : tensor_layout_N<LWPHY_DIM_MAX>
{
    //------------------------------------------------------------------
    // tensor_layout_any()
    LWDA_BOTH_INLINE
    tensor_layout_any() noexcept :
        rank_(0) {}
    //------------------------------------------------------------------
    // tensor_layout_any()
    LWDA_BOTH_INLINE
    tensor_layout_any(int nrank, const int* dim, const int* str) noexcept :
        rank_(nrank)
    {
        // Explicitly initialize base class dimensions and strides, using
        // sensible values for unused dimensions (1) and strides (0)
        for(int i = 0; i < LWPHY_DIM_MAX; ++i)
        {
            dimensions[i] = (i < nrank) ? dim[i] : 1;
        }
        if(str)
        {
            for(int i = 0; i < LWPHY_DIM_MAX; ++i)
            {
                // Set unused strides to 0
                strides[i] = (i < nrank) ? str[i] : 0;
            }
        }
        else
        {
            // Default to tightly packed when no strides are provided
            strides[0] = 1;
            for(int i = 1; i < LWPHY_DIM_MAX; ++i)
            {
                strides[i] = (i < nrank) ? (dim[i - 1] * strides[i - 1]) : 0;
            }
        }
    }
    //------------------------------------------------------------------
    // validate()
    static bool validate(int             rank,
                         const int*      dim,
                         const int*      str,
                         lwphyDataType_t type) noexcept;
    //------------------------------------------------------------------
    // has_same_size()
    bool has_same_size(const tensor_layout_any& layout) const noexcept
    {
        return (rank() == layout.rank()) &&
               (dimensions == layout.dimensions);
    }
    //------------------------------------------------------------------
    // has_same_strides()
    bool has_same_strides(const tensor_layout_any& layout) const noexcept
    {
        return (rank() == layout.rank()) &&
               (strides == layout.strides);
    }
    int rank() const { return rank_; }

private:
    int rank_; // Explicit count of number of "valid" dimensions
};

////////////////////////////////////////////////////////////////////////
// tensor_layout_contiguous_N
// Data type-independent description of tensor layout for a fixed (at
// compile time) number of dimensions, with a first dimension stride of
// 1.
// Some API functions might only support input tensors with a
// contiguous layout, or we might have specialized kernels for the
// common contiguous case.
//
// This class supports both host and device usage.
//
// Note specialization for size 1, to avoid complications for a zero
// length stride vector.
template <int TRank>
struct tensor_layout_contiguous_N
{
    vec<int, TRank>     dimensions;
    vec<int, TRank - 1> strides;
    int                 rank() const { return TRank; }
    //------------------------------------------------------------------
    // tensor_layout_contiguous_N()
    LWDA_BOTH_INLINE
    tensor_layout_contiguous_N() noexcept
    {
        dimensions.fill(0);
        strides.fill(0);
    }
    //------------------------------------------------------------------
    // tensor_layout_contiguous_N()
    LWDA_BOTH_INLINE
    tensor_layout_contiguous_N(const int* dim, const int* str) noexcept
    {
        // clang-format off
        for(int i = 0; i < TRank;     ++i)   { dimensions[i] = dim[i]; }
        for(int i = 0; i < (TRank-1); ++i)   { strides[i]    = str[i]; }
        // clang-format on
    }
    //------------------------------------------------------------------
    // tensor_layout_contiguous_N()
    LWDA_BOTH_INLINE
    explicit tensor_layout_contiguous_N(const tensor_layout_any& layout_any) noexcept
    {
        // clang-format off
        for(int i = 0; i < TRank;     ++i)   { dimensions[i] = layout_any.dimensions[i];  }
        for(int i = 0; i < (TRank-1); ++i)   { strides[i]    = layout_any.strides[i + 1]; } // skip first stride (contiguous)
        // clang-format on
    }
    //------------------------------------------------------------------
    // offset()
    LWDA_BOTH_INLINE
    size_t offset(const vec<int, TRank>& indices) const
    {
        // Add offset for first index, which has stride 1
        size_t idx = indices[0];
#pragma unroll
        for(size_t i = 1; i < TRank; ++i)
        {
            idx += (indices[i] * strides[i - 1]);
        }
        return idx;
    }
    //------------------------------------------------------------------
    LWDA_BOTH_INLINE
    size_t offset(const int (&n)[TRank]) const
    {
        size_t idx = n[0];
#pragma unroll
        for(int i = 1; i < TRank; ++i)
        {
            idx += (n[i] * strides[i - 1]);
        }
        return idx;
    }
    //------------------------------------------------------------------
    // get_num_elements()
    LWDA_BOTH_INLINE
    size_t get_num_elements() const noexcept
    {
        // Return the product of the dimensions d(0) * d(1) * ... * d(N-1)
        return dimensions.product();
    }
    //------------------------------------------------------------------
    // get_num_elements_with_strides()
    LWDA_BOTH_INLINE
    size_t get_num_elements_with_strides() const noexcept
    {
        // Account for first index, which has stride 1
        size_t max_size = dimensions[0];
        for(int i = 1; i < TRank; ++i)
        {
            max_size = std::max(max_size, static_cast<size_t>(dimensions[i] * strides[i - 1]));
        }
        return max_size;
    }
};

template <>
struct tensor_layout_contiguous_N<1>
{
    vec<int, 1> dimensions;
    int         rank() const { return 1; }
    //------------------------------------------------------------------
    // tensor_layout_contiguous_N()
    LWDA_BOTH_INLINE
    tensor_layout_contiguous_N() noexcept
    {
        dimensions[0] = 0;
    }
    //------------------------------------------------------------------
    // tensor_layout_contiguous_N()
    LWDA_BOTH_INLINE
    tensor_layout_contiguous_N(const int* dim, const int* /*str*/) noexcept
    {
        dimensions[0] = dim[0];
    }
    //------------------------------------------------------------------
    // tensor_layout_contiguous_N()
    LWDA_BOTH_INLINE
    explicit tensor_layout_contiguous_N(const tensor_layout_any& layout_any) noexcept
    {
        dimensions[0] = layout_any.dimensions[0];
    }
    //------------------------------------------------------------------
    // offset()
    LWDA_BOTH_INLINE
    size_t offset(const vec<int, 1>& indices) const
    {
        return indices[0];
    }
    //------------------------------------------------------------------
    LWDA_BOTH_INLINE
    size_t offset(const int (&n)[1]) const
    {
        return n[0];
    }
    //------------------------------------------------------------------
    // get_num_elements()
    LWDA_BOTH_INLINE
    size_t get_num_elements() const noexcept
    {
        // Return the product of the dimensions d(0) * d(1) * ... * d(N-1)
        return dimensions[0];
    }
    //------------------------------------------------------------------
    // get_num_elements_with_strides()
    LWDA_BOTH_INLINE
    size_t get_num_elements_with_strides() const noexcept
    {
        // Account for first index, which has stride 1
        return dimensions[0];
    }
};

////////////////////////////////////////////////////////////////////////
// Function to translate the layout for a tensor of type LWPHY_BIT to
// an equivalent tensor with 32-bit words.
tensor_layout_any word_layout_from_bit_layout(const tensor_layout_any& kLayout);

template <class T>
using lwphy_optional = std::experimental::optional<T>;

////////////////////////////////////////////////////////////////////////
// tensor_ref
// Combination of:
// 1.) a storage element type
// 2.) a layout class
// 3.) an address
// to allow host and device code to access elements of the tensor.
// The TLayout class provides callwlation of the address of a specific
// element of the tensor.
template <lwphyDataType_t TType,
          int             NDim,
          bool            TIsConst     = false,
          template <int> class TLayout = tensor_layout_N>
class tensor_ref //
{
public:
    typedef typename std::conditional<TIsConst,
                                      const typename data_type_traits<TType>::type,
                                      typename data_type_traits<TType>::type>::type element_t;
    typedef typename data_type_traits<TType>::type                                  non_const_element_t;
    typedef TLayout<NDim>                                                           layout_t;

    tensor_ref(void* addr, const int* dim, const int* str) :
        addr_(static_cast<element_t*>(addr)),
        layout_(dim, str)
    {
        static_assert(!TIsConst, "Cannot initialize const tensor with non-const address");
    }
    tensor_ref(const void* addr, const int* dim, const int* str) :
        addr_(static_cast<element_t*>(addr)),
        layout_(dim, str)
    {
    }
    tensor_ref(void* addr, const layout_t& layout) :
        addr_(static_cast<element_t*>(addr)),
        layout_(layout)
    {
        static_assert(!TIsConst, "Cannot initialize const tensor with non-const address");
    }
    LWDA_BOTH_INLINE
    element_t* addr() { return addr_; }
    LWDA_BOTH_INLINE
    element_t& operator()(const int (&idx)[NDim]) { return addr_[layout_.offset(idx)]; }
    LWDA_BOTH_INLINE
    const element_t& operator()(const int (&idx)[NDim]) const { return addr_[layout_.offset(idx)]; }
    //------------------------------------------------------------------
    // Generate a tensor_ref for a specific number of dimensions
    template <int TRank>
    lwphy_optional<tensor_ref<TType, TRank>> get_ref_rank()
    {
        if(layout_.rank() == TRank)
        {
            return tensor_ref<TType, TRank, TIsConst>(addr_,
                                                      layout_.dimensions.begin(),
                                                      layout_.strides.begin());
        }
        else
        {
            return std::experimental::nullopt;
        }
    }
    //------------------------------------------------------------------
    // Generate a contiguous tensor_ref for a specific rank
    template <int TRank>
    lwphy_optional<tensor_ref<TType, TRank, TIsConst, tensor_layout_contiguous_N>>
    get_ref_contig_rank()
    {
        if(1 == layout_.strides[0])
        {
            return tensor_ref<TType, TRank, TIsConst, tensor_layout_contiguous_N>(addr_,
                                                                                  layout_.dimensions.begin(),
                                                                                  layout_.strides.begin() + 1); // skip unit stride
        }
        else
        {
            return std::experimental::nullopt;
        }
    }
    //------------------------------------------------------------------
    // Generate a contiguous tensor_ref for a specific rank (const)
    template <int TRank>
    lwphy_optional<tensor_ref<TType, TRank, TIsConst, tensor_layout_contiguous_N>>
    get_ref_contig_rank() const
    {
        if(1 == layout_.strides[0])
        {
            return tensor_ref<TType, TRank, TIsConst, tensor_layout_contiguous_N>(addr_,
                                                                                  layout_.dimensions.begin(),
                                                                                  layout_.strides.begin() + 1); // skip unit stride
        }
        else
        {
            return std::experimental::nullopt;
        }
    }
    LWDA_BOTH_INLINE
    const layout_t& layout() const { return layout_; }

private:
    element_t* addr_;
    layout_t   layout_;
};

// clang-format off
template <lwphyDataType_t T, int NDim = LWPHY_DIM_MAX>     using optional_tensor_ref        = std::experimental::optional<tensor_ref<T, NDim>>;
template <lwphyDataType_t T, bool TIsConst = false>        using tensor_ref_any             = tensor_ref<T, LWPHY_DIM_MAX, TIsConst, tensor_layout_N>;
template <lwphyDataType_t T, bool TIsConst = false>        using tensor_ref_contig_any      = tensor_ref<T, LWPHY_DIM_MAX, TIsConst, tensor_layout_contiguous_N>;
template <lwphyDataType_t T, int N, bool TIsConst = false> using tensor_ref_contig_N        = tensor_ref<T, N, TIsConst, tensor_layout_contiguous_N>;
template <lwphyDataType_t T, bool TIsConst = false>        using tensor_ref_contig_1D       = tensor_ref<T, 1, TIsConst, tensor_layout_contiguous_N>;
template <lwphyDataType_t T, bool TIsConst = false>        using tensor_ref_contig_2D       = tensor_ref<T, 2, TIsConst, tensor_layout_contiguous_N>;
template <lwphyDataType_t T, bool TIsConst = false>        using tensor_ref_contig_3D       = tensor_ref<T, 3, TIsConst, tensor_layout_contiguous_N>;
template <lwphyDataType_t T>                               using const_tensor_ref_contig_1D = tensor_ref<T, 1, true, tensor_layout_contiguous_N>;
template <lwphyDataType_t T>                               using const_tensor_ref_contig_2D = tensor_ref<T, 2, true, tensor_layout_contiguous_N>;
template <lwphyDataType_t T>                               using const_tensor_ref_contig_3D = tensor_ref<T, 3, true, tensor_layout_contiguous_N>;
// clang-format on

////////////////////////////////////////////////////////////////////////
// tensor_desc
// The tensor_desc class combines a layout specification (i.e dimension
// sizes and strides) along with a specific element type enumeration
// (e.g. 32-bit floating point complex values).
// Upon construction, the element type is LWPHY_VOID. In this state,
// the tensor_desc is invalid, and no operations can make use of the
// tensor_desc.
// Users must call the set() function to place the tensor_desc into a
// usable state. If the arguments to the set() function do not
// represent a valid descriptor, the set() function will return false,
// and the descriptor will remain in its previous state.
class tensor_desc : public lwphyTensorDescriptor //
{
public:
    tensor_desc() noexcept :
        type_(LWPHY_VOID),
        size_bytes_(0) {}
    ~tensor_desc() noexcept = default;
    // Returns true for success, false if the type and/or layout are
    // invalid.
    bool                     set(lwphyDataType_t t,
                                 int             numDim,
                                 const int*      dim,
                                 const int*      strides) noexcept;
    lwphyDataType_t          type() const noexcept { return type_; }
    const tensor_layout_any& layout() const noexcept { return layout_; }
    size_t                   get_size_in_bytes() const noexcept { return size_bytes_; }
    //------------------------------------------------------------------
    // Generate a tensor_ref for a specific type and number of dimensions
    template <lwphyDataType_t TType, int TRank>
    lwphy_optional<tensor_ref<TType, TRank, false>> get_ref_rank(void* pv) const
    {
        if((TType == type()) && (layout().rank() == TRank))
        {
            return tensor_ref<TType, TRank, false>(pv,
                                                   layout().dimensions.begin(),
                                                   layout().strides.begin());
        }
        else
        {
            return std::experimental::nullopt;
        }
    }
    //------------------------------------------------------------------
    // Generate a tensor_ref for a specific type and number of dimensions
    // (const address)
    template <lwphyDataType_t TType, int TRank>
    lwphy_optional<tensor_ref<TType, TRank, true>> get_ref_rank(const void* pv) const
    {
        if((TType == type()) && (layout().rank() == TRank))
        {
            return tensor_ref<TType, TRank, true>(pv,
                                                  layout().dimensions.begin(),
                                                  layout().strides.begin());
        }
        else
        {
            return std::experimental::nullopt;
        }
    }
    //------------------------------------------------------------------
    // Generate a tensor_ref for a specific type, allowing any number of
    // dimensions by promoting to the maximum rank (LWPHY_DIM_MAX).
    template <lwphyDataType_t TType>
    lwphy_optional<tensor_ref<TType, LWPHY_DIM_MAX>> get_ref(void* pv) const
    {
        if(TType == type())
        {
            return tensor_ref<TType, LWPHY_DIM_MAX>(pv,
                                                    layout_.dimensions.begin(),
                                                    layout_.strides.begin());
        }
        else
        {
            return std::experimental::nullopt;
        }
    }
    //------------------------------------------------------------------
    // Generate a tensor_ref for a specific type, allowing any number of
    // dimensions by promoting to the maximum rank (LWPHY_DIM_MAX)
    // (const address)
    template <lwphyDataType_t TType>
    lwphy_optional<tensor_ref<TType, LWPHY_DIM_MAX>> get_ref(const void* pv) const
    {
        if(TType == type())
        {
            return tensor_ref<TType, LWPHY_DIM_MAX, true>(pv,
                                                          layout_.dimensions.begin(),
                                                          layout_.strides.begin());
        }
        else
        {
            return std::experimental::nullopt;
        }
    }
    //------------------------------------------------------------------
    // Generate a tensor_ref that is contiguous in the first dimension
    // for a specific type.
    template <lwphyDataType_t TType, int TRank>
    lwphy_optional<tensor_ref<TType, TRank, false, tensor_layout_contiguous_N>>
    get_ref_contig_rank(void* pv) const
    {
        if((TType == type()) && (1 == layout_.strides[0]))
        {
            return tensor_ref<TType, TRank, false, tensor_layout_contiguous_N>(pv,
                                                                               layout_.dimensions.begin(),
                                                                               layout_.strides.begin() + 1); // skip unit stride
        }
        else
        {
            return std::experimental::nullopt;
        }
    }
    //------------------------------------------------------------------
    // Generate a tensor_ref that is contiguous in the first dimension
    // for a specific type.
    // (const address)
    template <lwphyDataType_t TType, int TRank>
    lwphy_optional<tensor_ref<TType, TRank, true, tensor_layout_contiguous_N>>
    get_ref_contig_rank(const void* pv) const
    {
        if((TType == type()) && (1 == layout_.strides[0]))
        {
            return tensor_ref<TType, TRank, true, tensor_layout_contiguous_N>(pv,
                                                                              layout_.dimensions.begin(),
                                                                              layout_.strides.begin() + 1); // skip unit stride
        }
        else
        {
            return std::experimental::nullopt;
        }
    }

private:
    lwphyDataType_t   type_;
    tensor_layout_any layout_;
    size_t            size_bytes_; // includes padding induced by strides
};

typedef std::pair<std::reference_wrapper<const tensor_desc>, void*>       tensor_pair;
typedef std::pair<std::reference_wrapper<const tensor_desc>, const void*> const_tensor_pair;

#endif // !defined(TENSOR_DESC_HPP_INCLUDED_)
