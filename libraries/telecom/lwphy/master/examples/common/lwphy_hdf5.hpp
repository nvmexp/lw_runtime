/*
 * Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 */

#if !defined(LWPHY_HDF5_HPP_INCLUDED_)
#define LWPHY_HDF5_HPP_INCLUDED_

#include "lwphy_hdf5.h"
#include "hdf5hpp.hpp"
#include "lwphy.hpp"
#include <array>
#include <exception>

namespace lwphy
{
////////////////////////////////////////////////////////////////////////
// lwphy::lwphyHDF5_exception
// Exception class for errors from the lwphy_hdf5 library
class lwphyHDF5_exception : public std::exception //
{
public:
    lwphyHDF5_exception(lwphyHDF5Status_t s) :
        status_(s) {}
    virtual ~lwphyHDF5_exception() = default;
    virtual const char* what() const noexcept { return lwphyHDF5GetErrorString(status_); }
    lwphyHDF5Status_t status() const { return status_; }
private:
    lwphyHDF5Status_t status_;
};

////////////////////////////////////////////////////////////////////////
// lwphy::lwphyHDF5_struct
class lwphyHDF5_struct
{
public:
    lwphyHDF5_struct(lwphyHDF5Struct_t s = nullptr) : s_(s) {}
    lwphyHDF5_struct(lwphyHDF5_struct&& hdf5Struct) :
        s_(hdf5Struct.s_)
    {
        hdf5Struct.s_ = nullptr;
    }
    ~lwphyHDF5_struct() { if(s_) lwphyHDF5ReleaseStruct(s_); }
    lwphyHDF5_struct(const lwphyHDF5_struct&)             = delete;
    lwphyHDF5_struct&  operator=(const lwphyHDF5_struct&) = delete;
    lwphyHDF5_struct&  operator=(lwphyHDF5_struct&& hdf5Struct)
    {
        if(s_) lwphyHDF5ReleaseStruct(s_);
        s_ = hdf5Struct.s_;
        hdf5Struct.s_ = nullptr;
        return *this;
    }
    lwphyVariant_t get_value(const char* name) const
    {
        lwphyVariant_t v;
        lwphyHDF5Status_t status = lwphyHDF5GetStructScalar(&v,
                                                            s_,
                                                            name,
                                                            LWPHY_VOID);
        if(LWPHYHDF5_STATUS_SUCCESS != status)
        {
            throw lwphyHDF5_exception(status);
        }
        return v;
    }
    template <typename T>
    T get_value_as(const char* name) const
    {
        variant v;
        lwphyHDF5Status_t status = lwphyHDF5GetStructScalar(&v,
                                                            s_,
                                                            name,
                                                            type_to_lwphy_type<T>::value);
        if(LWPHYHDF5_STATUS_SUCCESS != status)
        {
            throw lwphyHDF5_exception(status);
        }
        return v.as<T>();
    }
private:
    lwphyHDF5Struct_t s_;
};

////////////////////////////////////////////////////////////////////////
// lwphy::get_HDF5_struct()
inline
lwphyHDF5_struct get_HDF5_struct(hdf5hpp::hdf5_dataset& dset,
                                 size_t                 numDim = 0,
                                 const hsize_t*         coords = nullptr)
{
    lwphyHDF5Struct_t s      = nullptr;
    lwphyHDF5Status_t status = lwphyHDF5GetStruct(dset.id(), numDim, coords, &s);
    if(LWPHYHDF5_STATUS_SUCCESS != status)
    {
        throw lwphyHDF5_exception(status);
    }
    return lwphyHDF5_struct(s);
}

////////////////////////////////////////////////////////////////////////
// lwphy::get_HDF5_struct()
inline
lwphyHDF5_struct get_HDF5_struct(hdf5hpp::hdf5_file& f,
                                 const char*         name,
                                 size_t              numDim = 0,
                                 const hsize_t*      coords = nullptr)
{
    hdf5hpp::hdf5_dataset dset = f.open_dataset(name);
    return get_HDF5_struct(dset, numDim, coords);
}

////////////////////////////////////////////////////////////////////////
// lwphy::get_HDF5_struct_index()
inline
lwphyHDF5_struct get_HDF5_struct_index(hdf5hpp::hdf5_dataset& dset,
                                       hsize_t                idx)
{
    const size_t numDim = 1;
    return get_HDF5_struct(dset, numDim, &idx);
}

////////////////////////////////////////////////////////////////////////
// lwphy::get_HDF5_struct_index()
inline
lwphyHDF5_struct get_HDF5_struct_index(hdf5hpp::hdf5_dataset& dset,
                                       hsize_t                idx0,
                                       hsize_t                idx1)
{
    const size_t numDim = 2;
    const hsize_t coords[2] = {idx0, idx1};
    return get_HDF5_struct(dset, numDim, coords);
}

// clang-format off
inline lwphy::tensor_info get_HDF5_dataset_info(const hdf5hpp::hdf5_dataset& dset)
{
    lwphyDataType_t                dtype;
    lwphy::vec<int, LWPHY_DIM_MAX> dim;
    int                            rank;
    lwphyHDF5Status_t              s = lwphyHDF5GetDatasetInfo(dset.id(),
                                                               LWPHY_DIM_MAX,
                                                               &dtype,
                                                               &rank,
                                                               dim.begin());
    if(LWPHYHDF5_STATUS_SUCCESS != s)
    {
        throw lwphyHDF5_exception(s);
    }
    return lwphy::tensor_info(dtype, lwphy::tensor_layout(rank, dim.begin(), nullptr));
}
// clang-format on


template <class TTensor>
inline void read_HDF5_dataset(TTensor&                     t,
                              const hdf5hpp::hdf5_dataset& dset,
                              lwdaStream_t                 strm = 0)
{
    lwphyHDF5Status_t s = lwphyHDF5ReadDataset(t.desc().handle(),
                                               t.addr(),
                                               dset.id(),
                                               strm);
    if(LWPHYHDF5_STATUS_SUCCESS != s)
    {
        throw lwphyHDF5_exception(s);
    }
}


template <class TTensor>
inline void write_HDF5_dataset(hdf5hpp::hdf5_file& f,
                               const TTensor&      t,
                               const lwphy::tensor_desc& desc, 
                               const char*         name,
                               lwdaStream_t        strm = 0)
{
    lwphyHDF5Status_t s = lwphyHDF5WriteDataset(f.id(),
                                                name,
                                                desc.handle(),
                                                t.addr(),
                                                strm);
    if(LWPHYHDF5_STATUS_SUCCESS != s)
    {
        throw lwphyHDF5_exception(s);
    }
}

template <class TTensor>
inline void write_HDF5_dataset(hdf5hpp::hdf5_file& f,
                               const TTensor&      t,
                               const char*         name,
                               lwdaStream_t        strm = 0)
{
    lwphyHDF5Status_t s = lwphyHDF5WriteDataset(f.id(),
                                                name,
                                                t.desc().handle(),
                                                t.addr(),
                                                strm);
    if(LWPHYHDF5_STATUS_SUCCESS != s)
    {
        throw lwphyHDF5_exception(s);
    }
}

// Allocate and return a tensor, initializing data from the given HDF5 dataset
template <class TAlloc = device_alloc>
inline tensor<TAlloc> tensor_from_dataset(const hdf5hpp::hdf5_dataset& dset,
                                          unsigned int                 tensorDescFlags = LWPHY_TENSOR_ALIGN_DEFAULT,
                                          lwdaStream_t                 strm            = 0)
{
    tensor<TAlloc> t(get_HDF5_dataset_info(dset), tensorDescFlags); 
    read_HDF5_dataset(t, dset, strm);
    return t;
}

// Allocate and return a tensor, initializing data from the given HDF5 dataset
// (with colwersion)
template <class TAlloc = device_alloc>
inline tensor<TAlloc> tensor_from_dataset(const hdf5hpp::hdf5_dataset& dset,
                                          lwphyDataType_t              colwertToType,
                                          unsigned int                 tensorDescFlags = LWPHY_TENSOR_ALIGN_DEFAULT,
                                          lwdaStream_t                 strm            = 0)
{
    // Create a tensor info with the requested colwersion type, but the original layout
    tensor<TAlloc> t(tensor_info(colwertToType, get_HDF5_dataset_info(dset).layout()),
                     tensorDescFlags); 
    read_HDF5_dataset(t, dset, strm);
    return t;
}

// Allocate and return a tensor, initializing data from the given HDF5 dataset
template <lwphyDataType_t TType, class TAlloc = device_alloc>
inline typed_tensor<TType, TAlloc> typed_tensor_from_dataset(const hdf5hpp::hdf5_dataset& dset,
                                                             unsigned int                 tensorDescFlags = LWPHY_TENSOR_ALIGN_DEFAULT,
                                                             lwdaStream_t                 strm            = 0)
{
    typed_tensor<TType, TAlloc> t(get_HDF5_dataset_info(dset).layout(), tensorDescFlags); 
    read_HDF5_dataset(t, dset, strm);
    return t;
}

inline void disable_hdf5_error_print()
{
    H5Eset_auto(H5E_DEFAULT, (H5E_auto2_t)nullptr, nullptr);
}

inline void enable_hdf5_error_print()
{
    H5Eset_auto(H5E_DEFAULT, (H5E_auto2_t)H5Eprint, stderr);
}

} // namespace lwphy

#endif // !defined(LWPHY_HDF5_HPP_INCLUDED_)
