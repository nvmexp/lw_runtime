/*
 * Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 */

#if !defined(LWPHY_HPP_INCLUDED_)
#define LWPHY_HPP_INCLUDED_

#include "lwphy.h"
#include <string>
#include <array>
#include <algorithm>
#include <cassert>
#include <memory>
#include <vector>

#ifdef __LWDACC__
#define LWPHY_BOTH __host__ __device__
#define LWPHY_BOTH_INLINE __forceinline__ __host__ __device__
#define LWPHY_INLINE __forceinline__ __device__
#else
#define LWPHY_BOTH
#define LWPHY_INLINE
#ifdef WINDOWS
#define LWPHY_BOTH_INLINE __inline
#else
#define LWPHY_BOTH_INLINE __inline__
#endif
#endif

namespace lwphy
{
// clang-format off
////////////////////////////////////////////////////////////////////////
// lwphy::lwda_exception
// Exception class for errors from LWCA
class lwda_exception : public std::exception //
{
public:
    lwda_exception(lwdaError_t s) : status_(s) { }
    virtual ~lwda_exception() = default;
    virtual const char* what() const noexcept { return lwdaGetErrorString(status_); }
private:
    lwdaError_t status_;
};
// clang-format on

// clang-format off
////////////////////////////////////////////////////////////////////////
// lwphy::lwphy_exception
// Exception class for errors from the lwphy library
class lwphy_exception : public std::exception //
{
public:
    lwphy_exception(lwphyStatus_t s) : status_(s) { }
    virtual ~lwphy_exception() = default;
    virtual const char* what() const noexcept { return lwphyGetErrorString(status_); }
private:
    lwphyStatus_t status_;
};
// clang-format on

#define LWPHY_CHECK(c)                                                     \
    do                                                                     \
    {                                                                      \
        lwphyStatus_t s = c;                                               \
        if(s != LWPHY_STATUS_SUCCESS)                                      \
        {                                                                  \
            fprintf(stderr, "LWPHY_ERROR: %s (%i)\n", __FILE__, __LINE__); \
            throw lwphy::lwphy_exception(s);                               \
        }                                                                  \
    } while(0)


struct context_deleter
{
    typedef lwphyContext_t ptr_t;
    void operator()(ptr_t p) const
    {
        lwphyDestroyContext(p);
    }

};

////////////////////////////////////////////////////////////////////////////
// unique_ctx_ptr
using unique_ctx_ptr = std::unique_ptr<lwphyContext, context_deleter>;

////////////////////////////////////////////////////////////////////////////
// context
class context
{
public:
    //----------------------------------------------------------------------
    // context()
    context(unsigned int flags = 0)
    {
        lwphyContext_t p = nullptr;
        lwphyStatus_t  s = lwphyCreateContext(&p, flags);
        if(LWPHY_STATUS_SUCCESS != s)
        {
            throw lwphy::lwphy_exception(s);
        }
        ctx_.reset(p);
    }
    //----------------------------------------------------------------------
    // handle()
    lwphyContext_t handle() { return ctx_.get(); }
private:
    unique_ctx_ptr ctx_;
};

// clang-format off
////////////////////////////////////////////////////////////////////////////
// lwphy::type_traits
template <lwphyDataType_t Ttype> struct type_traits;
template <> struct type_traits<LWPHY_VOID>  { typedef void            type; };
template <> struct type_traits<LWPHY_BIT>   { typedef uint32_t        type; };
template <> struct type_traits<LWPHY_R_8I>  { typedef signed char     type; };
template <> struct type_traits<LWPHY_C_8I>  { typedef char2           type; };
template <> struct type_traits<LWPHY_R_8U>  { typedef unsigned char   type; };
template <> struct type_traits<LWPHY_C_8U>  { typedef uchar2          type; };
template <> struct type_traits<LWPHY_R_16I> { typedef short           type; };
template <> struct type_traits<LWPHY_C_16I> { typedef short2          type; };
template <> struct type_traits<LWPHY_R_16U> { typedef unsigned short  type; };
template <> struct type_traits<LWPHY_C_16U> { typedef ushort2         type; };
template <> struct type_traits<LWPHY_R_32I> { typedef int             type; };
template <> struct type_traits<LWPHY_C_32I> { typedef int2            type; };
template <> struct type_traits<LWPHY_R_32U> { typedef unsigned int    type; };
template <> struct type_traits<LWPHY_C_32U> { typedef uint2           type; };
template <> struct type_traits<LWPHY_R_16F> { typedef __half          type; };
template <> struct type_traits<LWPHY_C_16F> { typedef __half2         type; };
template <> struct type_traits<LWPHY_R_32F> { typedef float           type; };
template <> struct type_traits<LWPHY_C_32F> { typedef lwComplex       type; };
template <> struct type_traits<LWPHY_R_64F> { typedef double          type; };
template <> struct type_traits<LWPHY_C_64F> { typedef lwDoubleComplex type; };
// clang-format on

// clang-format off
////////////////////////////////////////////////////////////////////////////
// lwphy::type_traits
template <typename T> struct type_to_lwphy_type;
template <> struct type_to_lwphy_type<signed char>     { static constexpr lwphyDataType_t value = LWPHY_R_8I;  };
template <> struct type_to_lwphy_type<char2>           { static constexpr lwphyDataType_t value = LWPHY_C_8I;  };
template <> struct type_to_lwphy_type<unsigned char>   { static constexpr lwphyDataType_t value = LWPHY_R_8U;  };
template <> struct type_to_lwphy_type<uchar2>          { static constexpr lwphyDataType_t value = LWPHY_C_8U;  };
template <> struct type_to_lwphy_type<short>           { static constexpr lwphyDataType_t value = LWPHY_R_16I; };
template <> struct type_to_lwphy_type<short2>          { static constexpr lwphyDataType_t value = LWPHY_C_16I; };
template <> struct type_to_lwphy_type<unsigned short>  { static constexpr lwphyDataType_t value = LWPHY_R_16U; };
template <> struct type_to_lwphy_type<ushort2>         { static constexpr lwphyDataType_t value = LWPHY_C_16U; };
template <> struct type_to_lwphy_type<int>             { static constexpr lwphyDataType_t value = LWPHY_R_32I; };
template <> struct type_to_lwphy_type<int2>            { static constexpr lwphyDataType_t value = LWPHY_C_32I; };
template <> struct type_to_lwphy_type<unsigned int>    { static constexpr lwphyDataType_t value = LWPHY_R_32U; };
template <> struct type_to_lwphy_type<uint2>           { static constexpr lwphyDataType_t value = LWPHY_C_32U; };
template <> struct type_to_lwphy_type<__half>          { static constexpr lwphyDataType_t value = LWPHY_R_16F; };
template <> struct type_to_lwphy_type<__half2>         { static constexpr lwphyDataType_t value = LWPHY_C_16F; };
template <> struct type_to_lwphy_type<float>           { static constexpr lwphyDataType_t value = LWPHY_R_32F; };
template <> struct type_to_lwphy_type<lwComplex>       { static constexpr lwphyDataType_t value = LWPHY_C_32F; };
template <> struct type_to_lwphy_type<double>          { static constexpr lwphyDataType_t value = LWPHY_R_64F; };
template <> struct type_to_lwphy_type<lwDoubleComplex> { static constexpr lwphyDataType_t value = LWPHY_C_64F; };
// clang-format on

// clang-format off
////////////////////////////////////////////////////////////////////////
// lwphy::vec
// Array of elements with a size that is fixed at compile time. Similar
// to std::array, but this class differs in that it has function
// decorations for operation on a GPU device as well as the host.
template <typename T, int Dim>
class vec
{
public:
    vec() = default;
    template <int N>
    vec(const T(&list)[N])
    {
        static_assert(N == Dim, "Initializer list does not have enough dimensions");
        for(int i = 0; i < N; ++i) elem_[i] = list[i];
    }
    LWPHY_BOTH_INLINE
    void     fill(T val)               { for(int i = 0; i < Dim; ++i) elem_[i] = val; }
    LWPHY_BOTH_INLINE
    T&       operator[](int idx)       { return elem_[idx]; }
    LWPHY_BOTH_INLINE
    const T& operator[](int idx) const { return elem_[idx]; }
    LWPHY_BOTH_INLINE
    T* begin()             { return elem_;       }
    LWPHY_BOTH_INLINE
    T* end()               { return elem_ + Dim; }
    const T* begin() const { return elem_;       }
    const T* end()   const { return elem_ + Dim; }
    LWPHY_BOTH_INLINE
    bool operator==(const vec& rhs) const
    {
        for(int i = 0; i < Dim; ++i)
        {
            if(elem_[i] != rhs.elem_[i]) return false;
        }
        return true;
    }
    LWPHY_BOTH_INLINE
    bool operator!=(const vec& rhs) const
    {
        return !(*this == rhs);
    }
private:
    T  elem_[Dim];
};
// clang-format on

// clang-format off
////////////////////////////////////////////////////////////////////////
// lwphy::tensor_layout
class tensor_layout
{
public:
    tensor_layout() : rank_(0)
    {
        for(size_t i = 0; i < LWPHY_DIM_MAX; ++i) { dimensions_[i] = strides_[i] = 0; }
    }
    tensor_layout(int nrank, const int* dims, const int* str) : rank_(nrank)
    {
        for(size_t i = 0;     i < LWPHY_DIM_MAX; ++i) { dimensions_[i] = (i < nrank) ? dims[i] : 1; }
        if(str)
        {
            // Set unused strides to zero
            for(size_t i = 0; i < LWPHY_DIM_MAX; ++i) { strides_[i] = (i < nrank) ? str[i] : 0; }
        }
        else
        {
            strides_[0] = 1;
            for(size_t i = 1; i < LWPHY_DIM_MAX; ++i) { strides_[i] = (i < nrank) ? (strides_[i - 1] * dimensions_[i -1]) : 0; }
        }
        
        // Initialize remaining strides to zero
        for(size_t i = nrank; i < LWPHY_DIM_MAX; ++i) { strides_[i] = 0; }
    }
    template <int N>
    tensor_layout(const int(&dims)[N], const int(&strides)[N]) : rank_(N)
    {
        static_assert(N <= LWPHY_DIM_MAX, "Layout initialization must have less than LWPHY_DIM_MAX dimensions");
        for(size_t i = 0;     i < LWPHY_DIM_MAX; ++i) { dimensions_[i] = (i < N) ? dims[i]         : 1; }
        for(size_t i = 0;     i < LWPHY_DIM_MAX; ++i) { strides_[i]    = (i < N) ? strides[i]      : 0; }
    }
    tensor_layout(std::initializer_list<int> ilistDim)
    {
        if(ilistDim.size() > LWPHY_DIM_MAX)
        {
            throw std::runtime_error("Tensor rank exceeds layout maximum (LWPHY_DIM_MAX)");
        }
        rank_ = static_cast<int>(ilistDim.size());
        size_t count = 0;
        for(auto& d : ilistDim)                { dimensions_[count++] = d; }
        while(count < LWPHY_DIM_MAX)           { dimensions_[count++] = 1; }
        strides_[0] = 1;
        for(int i = 1; i < LWPHY_DIM_MAX; ++i) { strides_[i] = (i < rank_) ? (dimensions_[i-1] * strides_[i-1]) : 0; }
    }
    tensor_layout(std::initializer_list<int> ilistDim,
                  std::initializer_list<int> ilistStrides)
    {
        if((ilistDim.size() > LWPHY_DIM_MAX) || (ilistStrides.size() > LWPHY_DIM_MAX))
        {
            throw std::runtime_error("Tensor rank exceeds layout maximum (LWPHY_DIM_MAX)");
        }
        if(ilistDim.size() != ilistStrides.size())
        {
            throw std::runtime_error("Tensor dimension and stride specification mismatch");
        }
        rank_ = static_cast<int>(ilistDim.size());
        size_t count = 0;
        for(auto& d : ilistDim)                        { dimensions_[count++] = d; }
        while(count < LWPHY_DIM_MAX)                   { dimensions_[count++] = 1; }
        count = 0;
        for(auto& s : ilistStrides)                    { strides_[count++]    = s; }
        for(count = 0; count < LWPHY_DIM_MAX; ++count) { strides_[count]      = 0; }
    }
    template <int N>
    size_t offset(const int (&indices)[N]) const
    {
        size_t idx = 0;
        for(size_t i = 0; i < N; ++i)
        {
            idx += (indices[i] * strides_[i]);
        }
        return idx;
    }
    size_t get_offset(int i0)                         { return (i0 * strides_[0]); }
    size_t get_offset(int i0, int i1)                 { return ((i0 * strides_[0]) + (i1 * strides_[1])); }
    size_t get_offset(int i0, int i1, int i2)         { return ((i0 * strides_[0]) + (i1 * strides_[1]) + (i2 * strides_[2])); }
    size_t get_offset(int i0, int i1, int i2, int i3) { return ((i0 * strides_[0]) + (i1 * strides_[1]) + (i2 * strides_[2]) + (i3 * strides_[3])); }
    int rank() const { return rank_; }
    vec<int, LWPHY_DIM_MAX>& dimensions()             { return dimensions_; }
    vec<int, LWPHY_DIM_MAX>& strides()                { return strides_;    }
    const vec<int, LWPHY_DIM_MAX>& dimensions() const { return dimensions_; }
    const vec<int, LWPHY_DIM_MAX>& strides()    const { return strides_;    }
private:
    int rank_;
    vec<int, LWPHY_DIM_MAX> dimensions_; // (Only rank elements are valid)
    vec<int, LWPHY_DIM_MAX> strides_;    // stride in elements (Only rank elements are valid)
};
// clang-format on

// clang-format off
////////////////////////////////////////////////////////////////////////
// lwphy::tensor_info
// Representation of the type, dimensions, and strides that are stored
// internally by a tensor descriptor. (This class can be used to cache
// the values stored by the lwPHY library for a tensor descriptor, or
// to store values to assign to a lwPHY library tensor descriptor.)
class tensor_info
{
public:
    typedef tensor_layout layout_t;
    tensor_info() : data_type_(LWPHY_VOID)
    {
    }
    tensor_info(lwphyDataType_t type, std::initializer_list<int> ilist) :
        data_type_(type),
        layout_(ilist)
    {
    }
    tensor_info(lwphyDataType_t type, const layout_t& layout_in) :
        data_type_(type),
        layout_(layout_in)
    {
    }
    int             rank()   const { return layout_.rank(); }
    lwphyDataType_t type()   const { return data_type_;   }
    const layout_t& layout() const { return layout_;      }
    std::string to_string(bool withStride = true) const
    {
        std::string s("type: ");
        s.append(lwphyGetDataTypeString(data_type_));
        s.append(", dim: (");
        for(int i = 0; i < rank(); ++i)
        {
            if(i > 0) s.append(",");
            s.append(std::to_string(layout_.dimensions()[i]));
        }
        s.append(")");
        if(withStride)
        {
            s.append(", stride: (");
            for(int i = 0; i < rank(); ++i)
            {
                if(i > 0) s.append(",");
                s.append(std::to_string(layout_.strides()[i]));
            }
            s.append(")");
        }
        return s;
    }
private:
    lwphyDataType_t data_type_;
    layout_t        layout_;
};
// clang-format on

// clang-format off
////////////////////////////////////////////////////////////////////////
// lwphy::tensor_desc
// Wrapper class for lwPHY tensor descriptor objects. Capable of
// representing a tensor of any underlying type, and any rank (up to the
// maximum supported by the library).
class tensor_desc
{
public:
    typedef tensor_info tensor_info_t;
    tensor_desc()
    {
        lwphyStatus_t s = lwphyCreateTensorDescriptor(&desc_);
        if(LWPHY_STATUS_SUCCESS != s)
        {
            throw lwphy_exception(s);
        }
    }
    tensor_desc(lwphyDataType_t type, const tensor_layout& layout_in, unsigned int flags = LWPHY_TENSOR_ALIGN_DEFAULT)
    {
        tensor_info_t tinfo(type, layout_in);
        lwphyStatus_t s = lwphyCreateTensorDescriptor(&desc_);
        if(LWPHY_STATUS_SUCCESS != s)
        {
            throw lwphy_exception(s);
        }
        s = lwphySetTensorDescriptor(desc_,
                                     tinfo.type(),
                                     tinfo.rank(),
                                     tinfo.layout().dimensions().begin(),
                                     tinfo.layout().strides().begin(),
                                     flags);
        if(LWPHY_STATUS_SUCCESS != s)
        {
            throw lwphy_exception(s);
        }
    }
    tensor_desc(lwphyDataType_t type, std::initializer_list<int> ilist, unsigned int flags = LWPHY_TENSOR_ALIGN_DEFAULT)
    {
        tensor_info_t tinfo(type, ilist);
        lwphyStatus_t s = lwphyCreateTensorDescriptor(&desc_);
        if(LWPHY_STATUS_SUCCESS != s)
        {
            throw lwphy_exception(s);
        }
        s = lwphySetTensorDescriptor(desc_,
                                     tinfo.type(),
                                     tinfo.rank(),
                                     tinfo.layout().dimensions().begin(),
                                     tinfo.layout().strides().begin(),
                                     flags);
        if(LWPHY_STATUS_SUCCESS != s)
        {
            throw lwphy_exception(s);
        }
    }
    template <class TInfo>
    tensor_desc(const TInfo& tinfo, unsigned int flags = LWPHY_TENSOR_ALIGN_DEFAULT)
    {
        lwphyStatus_t s = lwphyCreateTensorDescriptor(&desc_);
        if(LWPHY_STATUS_SUCCESS != s)
        {
            throw lwphy_exception(s);
        }
        s = lwphySetTensorDescriptor(desc_,
                                     tinfo.type(),
                                     tinfo.rank(),
                                     tinfo.layout().dimensions().begin(),
                                     tinfo.layout().strides().begin(),
                                     flags);
        if(LWPHY_STATUS_SUCCESS != s)
        {
            throw lwphy_exception(s);
        }
    }
    tensor_desc(const tensor_desc& td)
    {
        lwphyStatus_t s = lwphyCreateTensorDescriptor(&desc_);
        if(LWPHY_STATUS_SUCCESS != s)
        {
            throw lwphy_exception(s);
        }
        tensor_info_t tinfo = td.get_info();
        s = lwphySetTensorDescriptor(desc_,
                                     tinfo.type(),
                                     tinfo.rank(),
                                     tinfo.layout().dimensions().begin(),
                                     tinfo.layout().strides().begin(),
                                     LWPHY_TENSOR_ALIGN_DEFAULT);
        if(LWPHY_STATUS_SUCCESS != s)
        {
            throw lwphy_exception(s);
        }
    }
    tensor_desc(tensor_desc&& td) : desc_(td.desc_) { td.desc_ = nullptr; }
    ~tensor_desc() { if(desc_) lwphyDestroyTensorDescriptor(desc_); }
    tensor_desc& operator=(const tensor_desc& td)
    {
        if(desc_) lwphyDestroyTensorDescriptor(desc_);
        desc_           = nullptr;
        lwphyStatus_t s = lwphyCreateTensorDescriptor(&desc_);
        if(LWPHY_STATUS_SUCCESS != s)
        {
            throw lwphy_exception(s);
        }
        tensor_info_t tinfo = td.get_info();
        s = lwphySetTensorDescriptor(desc_,
                                     tinfo.type(),
                                     tinfo.rank(),
                                     tinfo.layout().dimensions().begin(),
                                     tinfo.layout().strides().begin(),
                                     LWPHY_TENSOR_ALIGN_DEFAULT);
        if(LWPHY_STATUS_SUCCESS != s)
        {
            throw lwphy_exception(s);
        }
        return *this;
    }
    tensor_desc& operator=(tensor_desc&& td)
    {
        if(desc_) lwphyDestroyTensorDescriptor(desc_);
        desc_    = td.desc_;
        td.desc_ = nullptr;
        return *this;
    }
    //------------------------------------------------------------------
    // Modify the underlying tensor descriptor
    template <class TInfo>
    void set(const TInfo& tinfo, unsigned int flags = LWPHY_TENSOR_ALIGN_DEFAULT)
    {
        lwphyStatus_t s;
        s = lwphySetTensorDescriptor(desc_,
                                     tinfo.type(),
                                     tinfo.rank(),
                                     tinfo.layout().dimensions().begin(),
                                     tinfo.layout().strides().begin(),
                                     flags);
        if(LWPHY_STATUS_SUCCESS != s)
        {
            throw lwphy_exception(s);
        }
    }
    void set(lwphyDataType_t type, std::initializer_list<int> ilist, unsigned int flags = LWPHY_TENSOR_ALIGN_DEFAULT)
    {
        lwphyStatus_t s;
        tensor_info tinfo(type, ilist);
        s = lwphySetTensorDescriptor(desc_,
                                     tinfo.type(),
                                     tinfo.rank(),
                                     tinfo.layout().dimensions().begin(),
                                     tinfo.layout().strides().begin(),
                                     flags);
        if(LWPHY_STATUS_SUCCESS != s)
        {
            throw lwphy_exception(s);
        }
    }
    //------------------------------------------------------------------
    // Make a copy, optionally overriding the alignment with flags
    tensor_desc clone(unsigned int flags = LWPHY_TENSOR_ALIGN_DEFAULT) const
    {
        return tensor_desc(get_info(), flags);
    }
    //------------------------------------------------------------------
    // Retrieve a copy of the tensor info
    tensor_info_t get_info() const
    {
        lwphyDataType_t         dtype;
        int                     rank;
        vec<int, LWPHY_DIM_MAX> dimensions;
        vec<int, LWPHY_DIM_MAX> strides;
        lwphyStatus_t s = lwphyGetTensorDescriptor(desc_, // descriptor
                                                   LWPHY_DIM_MAX,
                                                   &dtype,
                                                   &rank,
                                                   dimensions.begin(),
                                                   strides.begin());
        if(LWPHY_STATUS_SUCCESS != s)
        {
            throw lwphy_exception(s);
        }
        return tensor_info_t(dtype, tensor_layout(rank, dimensions.begin(), strides.begin()));
    };
    //------------------------------------------------------------------
    // Get the size (in bytes) required for a tensor with this descriptor
    size_t get_size_in_bytes() const
    {
        size_t        sz = 0;
        lwphyStatus_t s  = lwphyGetTensorSizeInBytes(desc_, &sz);
        if(LWPHY_STATUS_SUCCESS != s)
        {
            throw lwphy_exception(s);
        }
        return sz;
    }
    lwphyTensorDescriptor_t handle() const { return desc_; }
private:
    lwphyTensorDescriptor_t desc_;
};
// clang-format on

////////////////////////////////////////////////////////////////////////
// lwphy::device_alloc
struct device_alloc
{
    static void* allocate(size_t nbytes)
    {
        void*       addr;
        lwdaError_t s = lwdaMalloc(&addr, nbytes);
        if(lwdaSuccess != s)
        {
            throw lwda_exception(s);
        }
        return addr;
    }
    static void deallocate(void* addr)
    {
        lwdaFree(addr);
    }
};

////////////////////////////////////////////////////////////////////////
// lwphy::pinned_alloc
struct pinned_alloc
{
    static void* allocate(size_t nbytes)
    {
        void*       addr;
        lwdaError_t s = lwdaHostAlloc(&addr, nbytes, 0);
        if(lwdaSuccess != s)
        {
            throw lwda_exception(s);
        }
        return addr;
    }
    static void deallocate(void* addr)
    {
        lwdaFreeHost(addr);
    }
};

// clang-format off
////////////////////////////////////////////////////////////////////////
// lwphy::tensor
// Class to manage a generic (non-typed) lwPHY tensor descriptor of any
// rank (up to the number supported by the lwPHY library) and an
// assocated memory allocation.
// NOTE: The tensor size is immutable (cannot be changed after construction)
template <class TAllocator = device_alloc>
class tensor //
{
public:
    tensor() : addr_(nullptr), alloc_memory(0) { }
    template <class TInfo>
    tensor(const TInfo& tinfo, unsigned int flags = LWPHY_TENSOR_ALIGN_DEFAULT) :
        desc_(tinfo, flags),
        addr_(TAllocator::allocate(desc_.get_size_in_bytes())),
        layout_(tinfo.layout()),
        alloc_memory(1)
    {
    }
    template <class TInfo>
    tensor(void * pre_alloc_addr, const TInfo& tinfo, unsigned int flags = LWPHY_TENSOR_ALIGN_DEFAULT) :
        desc_(tinfo, flags),
        addr_(pre_alloc_addr), /* No allocation */
        layout_(tinfo.layout()),
        alloc_memory(0)
    {
    }
    tensor(const tensor& t) :
        desc_(t.desc()),
        addr_(TAllocator::allocate(desc_.get_size_in_bytes())),
        layout_(t.layout()),
        alloc_memory(1)
    {
        // Use colwert() function, which will copy() if descs match
        colwert(t);
    }
    tensor(tensor&& t) :
        desc_(std::move(t.desc_)),
        addr_(t.addr_),
        layout_(t.layout_)
    {
        t.addr_ = nullptr;
        alloc_memory = t.alloc_memory;
    }
    ~tensor() { if((addr_) && (alloc_memory == 1)) TAllocator::deallocate(addr_); }

    tensor& operator=(const tensor& t)
    {
        if(this != &t)
        {
            copy(t);
        }
        return *this;
    }
    tensor& operator=(tensor&& t)
    {
        if((addr_) && (alloc_memory == 1)) TAllocator::deallocate(addr_);
        addr_ = t.addr_;
        t.addr_ = nullptr;
        layout_ = t.layout();
        desc_ = t.desc();
        alloc_memory = t.alloc_memory;
        return *this;
    }
    // Explicit copy() for callers that want to provide a stream
    template <class TSrcAlloc>
    void copy(const tensor<TSrcAlloc>& tSrc, lwdaStream_t strm = 0)
    {
        // If dimensions don't match, allocate a new descriptor.
        // If dimensions match but strides don't, assume that the
        // caller wants to keep the destination layout.
        if(dimensions() != tSrc.dimensions())
        {
            if((addr_) && (alloc_memory == 1))
            {
                TAllocator::deallocate(addr_);
                addr_ = nullptr;
            }
            desc_ = tSrc.desc();
            addr_ = TAllocator::allocate(desc_.get_size_in_bytes());
            alloc_memory = 1;
            layout_ = desc_.get_info().layout();
        }
        // Copy data.
        colwert(tSrc, strm);
    }
    template <class TSrcAlloc>
    void colwert(const tensor<TSrcAlloc>& tSrc, lwdaStream_t strm = 0)
    {
        lwphyStatus_t s = lwphyColwertTensor(desc_.handle(),
                                             addr_,
                                             tSrc.desc().handle(),
                                             tSrc.addr(),
                                             strm);
        if(LWPHY_STATUS_SUCCESS != s)
        {
            throw lwphy_exception(s);
        }
    }
    const tensor_desc&             desc() const       { return desc_; }
    void*                          addr() const       { return addr_; }
    int                            rank() const       { return layout_.rank(); }
    const vec<int, LWPHY_DIM_MAX>& dimensions() const { return layout_.dimensions(); }
    const vec<int, LWPHY_DIM_MAX>& strides()    const { return layout_.strides();    }
    const tensor_layout&           layout()     const { return layout_; }
    lwphyDataType_t                type()       const
    {
        tensor_info i = desc_.get_info();
        return i.type();
    }
private:
    tensor_desc   desc_;
    void*         addr_;
    tensor_layout layout_;
    int           alloc_memory;
};
// clang-format on

using tensor_device = tensor<device_alloc>;
using tensor_pinned = tensor<pinned_alloc>;

// clang-format off
////////////////////////////////////////////////////////////////////////
// lwphy::tensor_buffer
// Class to manage a generic (non-typed) lwPHY tensor descriptor of any
// rank (up to the number supported by the lwPHY library) and an
// assocated memory allocation.
// The associated tensor descriptor supports limited "resizing" - the
// internal descriptor can be modified without requiring a new
// memory allocation, as long as size of the buffer required by the new
// tensor descriptor info is less than or equal to the initial buffer
// size. (If it is larger, an exception will be thrown.)
// This class might be used to allocate a "max size" buffer, and then
// modify the tensor descriptor to refer to a subset of that maximum
// size.
// NOTE: Resizing DOES NOT rearrange the contents of the buffer to
// maintain values in previous locations, and it does not clear the
// contents.
template <class TAllocator = device_alloc>
class tensor_buffer //
{
public:
    tensor_buffer() : addr_(nullptr), alloc_size_(0) { }
    //------------------------------------------------------------------
    // Constructor
    // Allocate a buffer large enough to contain a tensor with the
    // given info (type and layout). Subsequent calls to reset() will
    // throw an exception if the destination type and layout are larger
    // than this allocation
    template <class TInfo>
    tensor_buffer(const TInfo& tinfo, unsigned int flags = LWPHY_TENSOR_ALIGN_DEFAULT) :
        desc_(tinfo, flags),
        alloc_size_(desc_.get_size_in_bytes()),
        addr_(TAllocator::allocate(alloc_size_)),
        layout_(desc_.get_info().layout())
    {
    }
    tensor_buffer(lwphyDataType_t type, std::initializer_list<int> ilist, unsigned int flags = LWPHY_TENSOR_ALIGN_DEFAULT) :
        desc_(tensor_info(type, ilist), flags),
        alloc_size_(desc_.get_size_in_bytes()),
        addr_(TAllocator::allocate(alloc_size_)),
        layout_(desc_.get_info().layout())
    {
    }
    tensor_buffer(tensor_buffer&& t) :
        desc_(std::move(t.desc_)),
        alloc_size_(t.alloc_size_),
        addr_(t.addr_),
        layout_(t.layout_)
    {
        t.addr_ = nullptr;
    }
    ~tensor_buffer() { if(addr_) TAllocator::deallocate(addr_); }

    tensor_buffer& operator=(tensor_buffer&& t)
    {
        if(addr_) TAllocator::deallocate(addr_);
        addr_ = t.addr_;
        t.addr_ = nullptr;
        layout_ = t.layout();
        desc_ = t.desc();
        alloc_size_ = t.alloc_size_;
        return *this;
    }
    template <class TInfo>
    void reset(const TInfo& tinfo, unsigned int flags = LWPHY_TENSOR_ALIGN_DEFAULT)
    {
        // Save the previous tensor info, in case we fail
        TInfo infoOld = desc_.get_info();
        // Set the new descriptor
        desc_.set(tinfo, flags);
        // Compare the required descriptor size to the allocation size
        if(desc_.get_size_in_bytes() > alloc_size_)
        {
            // Restore to the previous valid size
            desc_.set(infoOld);
            throw std::runtime_error("tensor_buffer reset() size exceeds original allocation");
        }
        layout_ = desc_.get_info().layout();
    }
    void reset(lwphyDataType_t type, std::initializer_list<int> ilist, unsigned int flags = LWPHY_TENSOR_ALIGN_DEFAULT)
    {
        // Save the previous tensor info, in case we fail
        tensor_info infoOld = desc_.get_info();
        // Set the new descriptor
        desc_.set(type, ilist, flags);
        // Compare the required descriptor size to the allocation size
        if(desc_.get_size_in_bytes() > alloc_size_)
        {
            // Restore to the previous valid size
            desc_.set(infoOld);
            throw std::runtime_error("tensor_buffer reset() size exceeds original allocation");
        }
        layout_ = desc_.get_info().layout();
    }

    const tensor_desc&             desc() const       { return desc_; }
    void*                          addr() const       { return addr_; }
    int                            rank() const       { return layout_.rank(); }
    const vec<int, LWPHY_DIM_MAX>& dimensions() const { return layout_.dimensions(); }
    const vec<int, LWPHY_DIM_MAX>& strides()    const { return layout_.strides();    }
    const tensor_layout&           layout()     const { return layout_; }
    lwphyDataType_t                type()       const
    {
        tensor_info i = desc_.get_info();
        return i.type();
    }
    size_t                         alloc_size() const { return alloc_size_; }
private:
    tensor_desc   desc_;
    size_t        alloc_size_;
    void*         addr_;
    tensor_layout layout_;
};
// clang-format on

using tensor_buffer_device = tensor_buffer<device_alloc>;
using tensor_buffer_pinned = tensor_buffer<pinned_alloc>;

// Tensors of type LWPHY_BIT can't use the standard offset callwlations
// to address bits, so we specialize the offset callwlations for tensors
// of that type
template <lwphyDataType_t TType>
struct offset_generator
{
    template <int N>
    static size_t get(const tensor_layout& layout, const int (&idx)[N])
    {
        return layout.offset(idx);
    }
};

template <>
struct offset_generator<LWPHY_BIT>
{
    template <int N>
    static size_t get(const tensor_layout& layout, const int (&idx)[N])
    {
        // Create a layout for 32-bit words and use that for
        // offset callwlation. Inefficient, but only used for
        // host checks.
        vec<int, LWPHY_DIM_MAX> newDims    = layout.dimensions();
        newDims[0]                         = (newDims[0] + 31) / 32;
        vec<int, LWPHY_DIM_MAX> newStrides = layout.strides();
        for(int i = 1; i < layout.rank(); ++i) // Skip first dim
        {
            newStrides[i] /= 32;
        }
        tensor_layout layout_word(layout.rank(),
                                  newDims.begin(),
                                  newStrides.begin());
        size_t        offset = layout_word.offset(idx);
        return offset;
    }
};

// clang-format off
////////////////////////////////////////////////////////////////////////
// typed_tensor
// Class to manage a lwPHY tensor descriptor of any rank (up to the
// maximum supported by the library) and an assocated memory allocation.
template <lwphyDataType_t TType, class TAllocator = device_alloc>
class typed_tensor
{
public:
    typedef typename type_traits<TType>::type element_t;
    typedef          TAllocator               allocator_t;
    typedef          offset_generator<TType>  offset_gen_t;
    typed_tensor() : addr_(nullptr) { }
    typed_tensor(const tensor_layout& tlayout, unsigned int flags = LWPHY_TENSOR_ALIGN_DEFAULT) :
        desc_(tensor_info(TType, tlayout), flags),
        layout_(desc_.get_info().layout()),
        addr_(static_cast<element_t*>(allocator_t::allocate(desc_.get_size_in_bytes())))
    {
    }
    typed_tensor(const typed_tensor&, lwdaStream_t strm = 0) = delete; // TODO
    typed_tensor(typed_tensor&& t) :
        desc_(std::move(t.desc_)),
        layout_(t.layout()),
        addr_(t.addr_)
    {
        t.addr_ = nullptr;
    }
    ~typed_tensor() { if(addr_) allocator_t::deallocate(addr_); }

    template <class TSrc>
    typed_tensor& operator=(const TSrc& tSrc)
    {
        lwphyStatus_t s = lwphyColwertTensor(desc_.handle(),
                                             addr_,
                                             tSrc.desc().handle(),
                                             tSrc.addr(),
                                             0);
        if(LWPHY_STATUS_SUCCESS != s)
        {
            throw lwphy_exception(s);
        }
        lwdaStreamSynchronize(0);
        return *this;
    }

    template <class TSrc>
    void colwert(const tensor<TSrc>& tSrc, lwdaStream_t strm = 0)
    {
        lwphyStatus_t s = lwphyColwertTensor(desc_.handle(),
                                             addr_,
                                             tSrc.desc().handle(),
                                             tSrc.addr(),
                                             strm);
        if(LWPHY_STATUS_SUCCESS != s)
        {
            throw lwphy_exception(s);
        }
    }

    const tensor_desc&             desc() const       { return desc_; }
    element_t*                     addr() const       { return addr_; }
    int                            rank() const       { return layout_.rank(); }
    const vec<int, LWPHY_DIM_MAX>& dimensions() const { return layout_.dimensions(); }
    const vec<int, LWPHY_DIM_MAX>& strides() const    { return layout_.strides();    }
    const tensor_layout&           layout() const     { return layout_; }
    // operator()
    // Indexed access, only enabled on the host for non-device allocations
    template <int N, typename TAlloc = TAllocator>
    typename std::enable_if<!std::is_same<TAlloc, device_alloc>::value, element_t&>::type
    operator()(const int(&idx)[N])
    {
        return addr_[offset_gen_t::get(layout_, idx)];
    }
    template <int N, typename TAlloc = TAllocator>
    typename std::enable_if<std::is_same<TAlloc, device_alloc>::value, element_t>::type
    operator()(const int(&idx)[N])
    {
        element_t elem;
        lwdaError_t e = lwdaMemcpy(&elem,
                                   addr_ + offset_gen_t::get(layout_, idx),
                                   sizeof(element_t),
                                   lwdaMemcpyDeviceToHost);
        if(e != lwdaSuccess)
        {
            throw lwda_exception(e);
        }
        return elem;
    }
private:
    tensor_desc   desc_;
    tensor_layout layout_;
    element_t*    addr_;
};
// clang-format on

template <class TDstAlloc, class TSrcAlloc>
struct memcpy_helper;

template <>
struct memcpy_helper<device_alloc, device_alloc>
{
    static constexpr lwdaMemcpyKind kind = lwdaMemcpyDeviceToDevice;
};

template <>
struct memcpy_helper<pinned_alloc, device_alloc>
{
    static constexpr lwdaMemcpyKind kind = lwdaMemcpyDeviceToHost;
};

template <>
struct memcpy_helper<device_alloc, pinned_alloc>
{
    static constexpr lwdaMemcpyKind kind = lwdaMemcpyHostToDevice;
};

// clang-format off
////////////////////////////////////////////////////////////////////////
// lwphy::buffer
template <typename T, class TAlloc>
class buffer
{
public:
    typedef T      element_t;
    typedef TAlloc allocator_t;
    
    buffer() : addr_(nullptr), size_(0) {}
    buffer(size_t numElements) :
        addr_(static_cast<element_t*>(allocator_t::allocate(numElements * sizeof(T)))),
        size_(numElements)
    {
    };
    ~buffer() { if(addr_) allocator_t::deallocate(addr_); }
    template <class TAlloc2>
    buffer(const buffer<T, TAlloc2>& b) :
        addr_(static_cast<element_t*>(allocator_t::allocate(b.size() * sizeof(T)))),
        size_(b.size())
    {
        lwdaError_t e = lwdaMemcpy(addr_,
                                   b.addr(),
                                   sizeof(T) * size_,
                                   memcpy_helper<TAlloc, TAlloc2>::kind);
        if(e != lwdaSuccess)
        {
            throw lwda_exception(e);
        }
    }
    buffer& operator=(buffer && b)
    {
       if(addr_) allocator_t::deallocate(addr_);
       addr_   = b.addr();
       size_   = b.size();
       b.addr_ = nullptr;
       return *this;
    }
    element_t*       addr()       { return addr_; }
    const element_t* addr() const { return addr_; }
    size_t           size() const { return size_; }
    // operator()
    // Indexed access, only enabled on the host for non-device allocations.
    // Use dummy template parameter for SFINAE
    template <typename Alloc = TAlloc>
    typename std::enable_if<!std::is_same<device_alloc, Alloc>::value, element_t&>::type
    operator[](size_t idx)
    {
        assert(idx < size_);
        return addr_[idx];
    }
private:
    element_t* addr_;
    size_t     size_;
};
// clang-format on

template <class T>
struct device_deleter
{
    typedef typename std::remove_all_extents<T>::type ptr_t;
    //typedef T ptr_t;
    void operator()(ptr_t* p) const
    {
        lwdaFree(p);
    }
};

template <class T>
struct pinned_deleter
{
    typedef typename std::remove_all_extents<T>::type ptr_t;
    //typedef T ptr_t;
    void operator()(ptr_t* p) const { lwdaFreeHost(p); }
};

template <typename T>
using unique_device_ptr = std::unique_ptr<T, device_deleter<T>>;

template <typename T>
using unique_pinned_ptr = std::unique_ptr<T, pinned_deleter<T>>;

template <typename T>
unique_device_ptr<T> make_unique_device(size_t count = 1)
{
    typedef typename unique_device_ptr<T>::pointer pointer_t;
    pointer_t                                      p = static_cast<pointer_t>(device_alloc::allocate(count * sizeof(T)));
    return unique_device_ptr<T>(p);
}

template <typename T>
unique_pinned_ptr<T> make_unique_pinned(size_t count = 1)
{
    typedef typename unique_pinned_ptr<T>::pointer pointer_t;
    pointer_t                                      p = static_cast<pointer_t>(pinned_alloc::allocate(count * sizeof(T)));
    return unique_pinned_ptr<T>(p);
}

// clang-format off
////////////////////////////////////////////////////////////////////////
// lwphy::variant
class variant : public lwphyVariant_t
{
public:
    variant()
    {
        type = LWPHY_VOID;
    }
    template <typename T>
    variant(T t)
    {
        type = type_to_lwphy_type<T>::value;
        set(t);
    }
    void set(const signed char&     sc)  { type = LWPHY_R_8I;  value.r8i  = sc;  }
    void set(const char2&           c2)  { type = LWPHY_C_8I;  value.c8i  = c2;  }
    void set(const unsigned char&   uc)  { type = LWPHY_R_8U;  value.r8u  = uc;  }
    void set(const uchar2&          uc2) { type = LWPHY_C_8U;  value.c8u  = uc2; }
    void set(const short&           s)   { type = LWPHY_R_16I; value.r16i = s;   }
    void set(const short2&          s2)  { type = LWPHY_C_16I; value.c16i = s2;  }
    void set(const unsigned short&  us)  { type = LWPHY_R_16U; value.r16u = us;  }
    void set(const ushort2&         us2) { type = LWPHY_C_16U; value.c16u = us2; }
    void set(const int&             i)   { type = LWPHY_R_32I; value.r32i = i;   }
    void set(const int2&            i2)  { type = LWPHY_C_32I; value.c32i = i2;  }
    void set(const unsigned int&    u)   { type = LWPHY_R_32U; value.r32u = u;   }
    void set(const uint2&           u2)  { type = LWPHY_C_32U; value.c32u = u2;  }
    void set(const __half&          h)   { type = LWPHY_R_16F; memcpy(&value.r16f, &h, sizeof(__half));   }
    void set(const __half2&         h2)  { type = LWPHY_C_16F; memcpy(&value.c16f, &h2, sizeof(__half2)); }
    void set(const float&           f)   { type = LWPHY_R_32F; value.r32f = f;   }
    void set(const lwComplex&       c)   { type = LWPHY_C_32F; value.c32f = c;   }
    void set(const double&          d)   { type = LWPHY_R_64F; value.r64f = d;   }
    void set(const lwDoubleComplex& dc)  { type = LWPHY_C_64F; value.c64f = dc;  }
    template <typename T> T& as();
};
// clang-format on

// clang-format off
template <> inline signed char&     variant::as<signed char>()     { if(type != LWPHY_R_8I)  throw std::runtime_error("variant type mismatch"); return value.r8i;  }
template <> inline char2&           variant::as<char2>()           { if(type != LWPHY_C_8I)  throw std::runtime_error("variant type mismatch"); return value.c8i;  }
template <> inline unsigned char&   variant::as<unsigned char>()   { if(type != LWPHY_R_8U)  throw std::runtime_error("variant type mismatch"); return value.r8u;  }
template <> inline uchar2&          variant::as<uchar2>()          { if(type != LWPHY_C_8U)  throw std::runtime_error("variant type mismatch"); return value.c8u;  }
template <> inline short&           variant::as<short>()           { if(type != LWPHY_R_16I) throw std::runtime_error("variant type mismatch"); return value.r16i; }
template <> inline short2&          variant::as<short2>()          { if(type != LWPHY_C_16I) throw std::runtime_error("variant type mismatch"); return value.c16i; }
template <> inline unsigned short&  variant::as<unsigned short>()  { if(type != LWPHY_R_16U) throw std::runtime_error("variant type mismatch"); return value.r16u; }
template <> inline ushort2&         variant::as<ushort2>()         { if(type != LWPHY_C_16U) throw std::runtime_error("variant type mismatch"); return value.c16u; }
template <> inline int&             variant::as<int>()             { if(type != LWPHY_R_32I) throw std::runtime_error("variant type mismatch"); return value.r32i; }
template <> inline int2&            variant::as<int2>()            { if(type != LWPHY_C_32I) throw std::runtime_error("variant type mismatch"); return value.c32i; }
template <> inline unsigned int&    variant::as<unsigned int>()    { if(type != LWPHY_R_32U) throw std::runtime_error("variant type mismatch"); return value.r32u; }
template <> inline uint2&           variant::as<uint2>()           { if(type != LWPHY_C_32U) throw std::runtime_error("variant type mismatch"); return value.c32u; }
//template <> inline __half&          variant::as<__half>()          { if(type != LWPHY_R_16F) throw std::runtime_error("variant type mismatch"); return value.r16f; }
//template <> inline __half2&         variant::as<uint2>()           { if(type != LWPHY_C_32F) throw std::runtime_error("variant type mismatch"); return value.c32f; }
template <> inline float&           variant::as<float>()           { if(type != LWPHY_R_32F) throw std::runtime_error("variant type mismatch"); return value.r32f; }
template <> inline lwComplex&       variant::as<lwComplex>()       { if(type != LWPHY_C_32F) throw std::runtime_error("variant type mismatch"); return value.c32f; }
template <> inline double&          variant::as<double>()          { if(type != LWPHY_R_64F) throw std::runtime_error("variant type mismatch"); return value.r64f; }
template <> inline lwDoubleComplex& variant::as<lwDoubleComplex>() { if(type != LWPHY_C_64F) throw std::runtime_error("variant type mismatch"); return value.c64f; }
// clang-format on

////////////////////////////////////////////////////////////////////////
// lwphy::device
class device
{
public:
    typedef int int3_t[3];
    device(int idx = 0) : index_(idx)
    {
        lwdaError_t e = lwdaGetDeviceProperties(&properties_, index_);
        if(lwdaSuccess != e)
        {
            throw lwda_exception(e);
        }
    }
    void set()
    {
        lwdaError_t e = lwdaSetDevice(index_);
        if(lwdaSuccess != e)
        {
            throw lwda_exception(e);
        }
    }
    // clang-format off
    const lwdaDeviceProp& properties() const { return properties_; }
    const char*   name()                         const { return properties_.name;                        }
    size_t        total_global_mem_bytes()       const { return properties_.totalGlobalMem;              }
    size_t        shared_mem_per_block()         const { return properties_.sharedMemPerBlock;           }
    int           registers_per_block()          const { return properties_.regsPerBlock;                }
    int           warp_size()                    const { return properties_.warpSize;                    }
    size_t        memory_pitch()                 const { return properties_.memPitch;                    }
    int           max_threads_per_block()        const { return properties_.maxThreadsPerBlock;          }
    const int3_t& max_threads_dim()              const { return properties_.maxThreadsDim;               }
    const int3_t& max_grid_size()                const { return properties_.maxGridSize;                 }
    int           clock_rate_kHz()               const { return properties_.clockRate;                   }
    size_t        total_const_mem_bytes()        const { return properties_.totalConstMem;               }
    int           major_version()                const { return properties_.major;                       }
    int           minor_version()                const { return properties_.minor;                       }
    int           multiprocessor_count()         const { return properties_.multiProcessorCount;         }
    int           kernel_timeout_enabled()       const { return properties_.kernelExecTimeoutEnabled;    }
    int           integrated()                   const { return properties_.integrated;                  }
    int           can_map_host_memory()          const { return properties_.canMapHostMemory;            }
    int           compute_mode()                 const { return properties_.computeMode;                 }
    int           conlwrrent_kernels()           const { return properties_.conlwrrentKernels;           }
    int           ECC_enabled()                  const { return properties_.ECCEnabled;                  }
    int           pci_bus_ID()                   const { return properties_.pciBusID;                    }
    int           pci_device_ID()                const { return properties_.pciDeviceID;                 }
    int           pci_domain_ID()                const { return properties_.pciDomainID;                 }
    int           tcc_driver()                   const { return properties_.tccDriver;                   }
    int           async_engine_count()           const { return properties_.asyncEngineCount;            }
    int           unified_addressing()           const { return properties_.unifiedAddressing;           }
    int           memory_clock_rate_kHz()        const { return properties_.memoryClockRate;             }
    int           memory_bus_width()             const { return properties_.memoryBusWidth;              }
    int           L2_cache_size_bytes()          const { return properties_.l2CacheSize;                 }
    int           max_threads_per_SM()           const { return properties_.maxThreadsPerMultiProcessor; }
    int           stream_prio_supported()        const { return properties_.streamPrioritiesSupported;   }
    int           global_L1_cache_supported()    const { return properties_.globalL1CacheSupported;      }
    int           local_L1_cache_supported()     const { return properties_.localL1CacheSupported;       }
    size_t        shmem_per_multiprocessor()     const { return properties_.sharedMemPerMultiprocessor;  }
    int           registers_per_multiprocessor() const { return properties_.regsPerMultiprocessor;       }
    // clang-format on
    std::string   desc() const
    {
        char buf[128];
        std::string s(properties_.name);
        snprintf(buf,
                 sizeof(buf) / sizeof(buf[0]),
                 ": %d SMs @ %.0f MHz, %.1f GiB @ %.0f MHz, Compute Capability %d.%d, PCI %04X:%02X:%02X",
                 properties_.multiProcessorCount,
                 properties_.clockRate / 1000.0,
                 properties_.totalGlobalMem / (1024.0 * 1024.0 * 1024.0),
                 properties_.memoryClockRate / 1000.0,
                 properties_.major,
                 properties_.minor,
                 properties_.pciDomainID,
                 properties_.pciBusID,
                 properties_.pciDeviceID);
        s.append(buf);
        return s;
    }
private:
    int            index_;
    lwdaDeviceProp properties_;
};

////////////////////////////////////////////////////////////////////////
// lwphy::stream
class stream
{
public:
    stream(unsigned int flags = lwdaStreamDefault)
    {
        lwdaError_t e = lwdaStreamCreateWithFlags(&stream_, flags);
        if(lwdaSuccess != e)
        {
            throw lwda_exception(e);
        }
    }
    stream(stream&& s) : stream_(s.stream_) { s.stream_ = nullptr; }
    ~stream() { if(stream_) lwdaStreamDestroy(stream_); }

    void synchronize()
    {
        lwdaError_t e = lwdaStreamSynchronize(stream_);
        if(lwdaSuccess != e)
        {
            throw lwda_exception(e);
        }
    }
    void wait_event(lwdaEvent_t ev)
    {
        lwdaError_t e = lwdaStreamWaitEvent(stream_, ev, 0);
        if(lwdaSuccess != e)
        {
            throw lwda_exception(e);
        }
    }
    lwdaError_t query() { return lwdaStreamQuery(stream_); }
    stream& operator==(const stream&) = delete;
    stream(const stream&) = delete;
    lwdaStream_t handle() { return stream_; }
private:
    lwdaStream_t stream_;
};

////////////////////////////////////////////////////////////////////////
// lwphy::event
class event
{
public:
    event(unsigned int flags = lwdaEventDefault)
    {
        lwdaError_t e = lwdaEventCreateWithFlags(&ev_, flags);
        if(lwdaSuccess != e)
        {
            throw lwda_exception(e);
        }
    }
    event(event&& e) : ev_(e.ev_) { e.ev_ = nullptr; }
    ~event() { if(ev_) lwdaEventDestroy(ev_); }

    void synchronize()
    {
        lwdaError_t e = lwdaEventSynchronize(ev_);
        if(lwdaSuccess != e)
        {
            throw lwda_exception(e);
        }
    }
    void record()
    {
        lwdaError_t e = lwdaEventRecord(ev_, 0);
        if(lwdaSuccess != e)
        {
            throw lwda_exception(e);
        }
    }
    void record(lwdaStream_t s)
    {
        lwdaError_t e = lwdaEventRecord(ev_, s);
        if(lwdaSuccess != e)
        {
            throw lwda_exception(e);
        }
    }
    void record(stream& s)
    {
        lwdaError_t e = lwdaEventRecord(ev_, s.handle());
        if(lwdaSuccess != e)
        {
            throw lwda_exception(e);
        }
    }
    lwdaError_t query() { return lwdaEventQuery(ev_); }
    event& operator==(const event&) = delete;
    event(const event&) = delete;
    lwdaEvent_t handle() { return ev_; }
private:
    lwdaEvent_t ev_;
};

////////////////////////////////////////////////////////////////////////
// lwphy::event_timer
// Class to time operations using a pair of LWCA events
//
// Usage:
// event_timer tmr;
// tmr.record_begin();             // Record begin event in stream
//   something interesting...
// tmr.record_end();               // Record end event in stream
// tmr.synchronize();              // Allow operations to finish
// float t =tmr.elapsed_time_ms(); // Retrieve time
class event_timer
{
public:
    void record_begin()               { begin_event_.record();  }
    void record_begin(stream&      s) { begin_event_.record(s); }
    void record_begin(lwdaStream_t s) { begin_event_.record(s); }
    void record_end()                 { end_event_.record();    }
    void record_end(stream&      s)   { end_event_.record(s);   }
    void record_end(lwdaStream_t s)   { end_event_.record(s);   }
    void synchronize() { end_event_.synchronize(); }
    float elapsed_time_ms()
    {
        float time_ms = 0.0f;
        lwdaError_t e = lwdaEventElapsedTime(&time_ms,
                                             begin_event_.handle(),
                                             end_event_.handle());
        if(lwdaSuccess != e)
        {
            throw lwda_exception(e);
        }
        return time_ms;
    }
private:
    event begin_event_, end_event_;
};

struct LDPC_decoder_deleter
{
    typedef lwphyLDPCDecoder_t ptr_t;
    void operator()(ptr_t p) const
    {
        lwphyDestroyLDPCDecoder(p);
    }

};
////////////////////////////////////////////////////////////////////////////
// unique_LDPC_decoder_ptr
using unique_LDPC_decoder_ptr = std::unique_ptr<lwphyLDPCDecoder, LDPC_decoder_deleter>;

class LDPC_decoder
{
public:
    //----------------------------------------------------------------------
    // LDPC_decoder()
    LDPC_decoder(context& ctx, unsigned int flags = 0)
    {
        lwphyLDPCDecoder_t dec = nullptr;
        lwphyStatus_t      s   = lwphyCreateLDPCDecoder(ctx.handle(),
                                                        &dec,
                                                        flags);
        if(LWPHY_STATUS_SUCCESS != s)
        {
            throw lwphy::lwphy_exception(s);
        }
        dec_.reset(dec);
    }
    //----------------------------------------------------------------------
    // get_workspace_size()
    size_t get_workspace_size(int             BG,
                              int             Kb,
                              int             mb,
                              int             Z,
                              int             numCodeWords,
                              lwphyDataType_t LLRtype,
                              int             algoIndex)
    {
        size_t        szBuf = 0;
        lwphyStatus_t s     = lwphyErrorCorrectionLDPCDecodeGetWorkspaceSize(handle(),
                                                                             BG,           // BG
                                                                             Kb,           // Kb
                                                                             mb,           // mb
                                                                             Z,            // Z
                                                                             numCodeWords, // numCodeblocks
                                                                             LLRtype,      // type
                                                                             algoIndex,    // algorithm
                                                                             &szBuf);      // output size
        if(LWPHY_STATUS_SUCCESS != s)
        {
            throw lwphy_exception(s);
        }
        return szBuf;
    }
    //----------------------------------------------------------------------
    // decode()
    void decode(lwphyTensorDescriptor_t       tensorDescDst,
                void*                         dstAddr,
                const lwphyTensorDescriptor_t tensorDescLLR,
                const void*                   LLRAddr,
                int                           BG,
                int                           Kb,
                int                           Z,
                int                           mb,
                int                           maxNumIterations,
                float                         normalization,
                int                           earlyTermination,
                lwphyLDPCResults_t*           results,
                int                           algoIndex,
                void*                         workspace,
                int                           flags,
                lwdaStream_t                  strm,
                void*                         reserved)
    {
        lwphyStatus_t s = lwphyErrorCorrectionLDPCDecode(handle(),
                                                         tensorDescDst,
                                                         dstAddr,
                                                         tensorDescLLR,
                                                         LLRAddr,
                                                         BG,
                                                         Kb,
                                                         Z,
                                                         mb,
                                                         maxNumIterations,
                                                         normalization,
                                                         earlyTermination,
                                                         results,
                                                         algoIndex,
                                                         workspace,
                                                         flags,
                                                         strm,
                                                         reserved);
        if(LWPHY_STATUS_SUCCESS != s)
        {
            throw lwphy_exception(s);
        }
    }
    //----------------------------------------------------------------------
    // handle()
    lwphyLDPCDecoder_t handle() { return dec_.get(); }
private:
    //----------------------------------------------------------------------
    // Data
    unique_LDPC_decoder_ptr dec_;
};

////////////////////////////////////////////////////////////////////////
// lwphy::stream_pool
// Simple round-robin stream pool
class stream_pool
{
public:
    //------------------------------------------------------------------
    // Constructor
    stream_pool(size_t maxSize = 32) :
        fork_event_(lwdaEventDisableTiming),
        lwrrent_stream_idx_(0)
    {
        for(size_t i = 0; i < maxSize; ++i)
        {
            streams_.emplace_back(lwdaStreamNonBlocking);
            join_events_.emplace_back(lwdaEventDisableTiming);
        }
        fork_width_ = streams_.size();
    }
    //------------------------------------------------------------------
    // Returns the maximum size of the thread pool
    size_t max_size() { return streams_.size(); }
    //------------------------------------------------------------------
    // resize()
    void resize(size_t maxSize)
    {
        if(maxSize != streams_.size())
        {
            streams_.clear();
            join_events_.clear();
            for(size_t i = 0; i < maxSize; ++i)
            {
                streams_.emplace_back(lwdaStreamNonBlocking);
                join_events_.emplace_back(lwdaEventDisableTiming);
            }
            fork_width_ = streams_.size();
        }
    }
    //------------------------------------------------------------------
    // fork()
    // The width specifies how many streams to use. If the width is
    // greater than the maximum size (from either construction or if the
    // resize() function is called, an exception is thrown.
    // If 0 is given, the maximum size is used.
    // This function records an event in the given stream. `width`
    // streams in the stream pool will wait on that recorded event, so
    // that the work submitted to those streams will not begin until
    // that event is signaled.
    void fork(lwdaStream_t s, size_t width = 0)
    {
        if(width >= streams_.size()) { throw std::runtime_error("Invalid stream pool fork width"); }
        if(0 == width)               { width = streams_.size(); }
        fork_width_ = width;
        lwrrent_stream_idx_ = 0;
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Record an event in the given stream. Forked stream submissions
        // will wait for this event before continuing.
        fork_event_.record(s);
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        for(size_t i = 0; i < fork_width_; ++i)
        {
            streams_[i].wait_event(fork_event_.handle());
        }
    }
    //------------------------------------------------------------------
    // join()
    void join(lwdaStream_t s)
    {
        for(size_t i = 0; i < fork_width_; ++i)
        {
            // Record an event in each worker stream
            join_events_[i].record(streams_[i]);
            // Force the main stream to wait for events from all worker
            // streams
            lwdaError_t e = lwdaStreamWaitEvent(s, join_events_[i].handle(), 0);
            if(lwdaSuccess != e)
            {
                throw lwda_exception(e);
            }
        }
    }
    //------------------------------------------------------------------
    // advance()
    void advance()
    {
        lwrrent_stream_idx_ = (lwrrent_stream_idx_ + 1) % fork_width_;
    }
    //------------------------------------------------------------------
    // lwrrent_stream()
    stream& lwrrent_stream() { return streams_[lwrrent_stream_idx_]; }
private:
    //------------------------------------------------------------------
    // Data
    std::vector<stream> streams_;
    size_t              lwrrent_stream_idx_;
    event               fork_event_;
    std::vector<event>  join_events_;
    size_t              fork_width_;
};

} // namespace lwphy

#endif // !defined(LWPHY_HPP_INCLUDED_)
