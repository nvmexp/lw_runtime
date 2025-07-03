/*
 * Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 */

#if !defined(HDF5HPP_HPP_INCLUDED_)
#define HDF5HPP_HPP_INCLUDED_

#include "hdf5.h"
#include <exception>
#include <utility> // std::forward()
#include <vector>
#include <stdexcept> // std::runtime_error()
#include <numeric>   // std::accumulate()

namespace hdf5hpp
{
////////////////////////////////////////////////////////////////////////
// hdf5hpp::hdf5_exception
// Default HDF5 error handling displays a call trace on stderr. We rely
// on that for information purposes, and just use this exception class
// for control flow.
class hdf5_exception : public std::exception //
{
public:
    virtual ~hdf5_exception() = default;
    virtual const char* what() const noexcept { return "HDF5 Error"; }
};

// clang-format off
//////////////////////////////////////////////////////////////////////
// hdf5hpp::hdf5_object
class hdf5_object //
{
public:
    static const hid_t ilwalid_hid = -1;
    static bool id_is_valid(hid_t h) { return h >= 0; }
    bool        is_valid() const     { return id_is_valid(id_); }
    hid_t       id() const           { return id_; }
protected:
    void set_ilwalid() { id_ = ilwalid_hid; }
    hdf5_object(hid_t id = -1) : id_(id) {}
    hdf5_object(hdf5_object&& o) : id_(o.id_) { o.id_ = ilwalid_hid; }
    hdf5_object& operator=(hdf5_object&& o)
    {
        std::swap(id_, o.id_);
        return *this;
    }
private:
    hid_t  id_;    
};
// clang-format on

// clang-format off
////////////////////////////////////////////////////////////////////////
// hdf5hpp::hdf5_dataspace
class hdf5_dataspace : public hdf5_object //
{
public:
    hdf5_dataspace(hid_t id = ilwalid_hid) : hdf5_object(id) {}
    hdf5_dataspace(hdf5_dataspace&& d) : hdf5_object(std::forward<hdf5_object>(d)) {}
    ~hdf5_dataspace() { if(is_valid()) H5Sclose(id()); }
    int get_rank() const
    {
        int ndims = H5Sget_simple_extent_ndims(id());
        if(ndims < 0) throw std::runtime_error("Invalid dataspace (negative rank)");
        return ndims;
    }
    std::vector<hsize_t> get_dimensions() const
    {
        std::vector<hsize_t> dims(get_rank());
        if(H5Sget_simple_extent_dims(id(), dims.data(), nullptr) < 0)
        {
            throw std::runtime_error("Invalid dataspace (extents invalid)\n");
        }
        return dims;
    }
    hsize_t get_num_elements() const
    {
        std::vector<hsize_t> dims = get_dimensions();
        return std::accumulate(begin(dims), end(dims), 1, std::multiplies<hsize_t>());
    }
};
// clang-format on

// clang-format off
////////////////////////////////////////////////////////////////////////
// hdf5hpp::hdf5_datatype
class hdf5_datatype: public hdf5_object
{
public:
    hdf5_datatype(hid_t id = ilwalid_hid) : hdf5_object(id) {}
    hdf5_datatype(hdf5_datatype&& d) : hdf5_object(std::forward<hdf5_object>(d)) {}
    ~hdf5_datatype() { if(is_valid()) H5Tclose(id()); }
    H5T_class_t get_class() const        { return H5Tget_class(id()); }
    bool        is_compound() const      { return (H5T_COMPOUND == get_class()); }
    bool        is_integer() const       { return (H5T_INTEGER  == get_class()); }
    bool        is_float() const         { return (H5T_FLOAT    == get_class()); }
    bool        is_signed() const        { return (H5T_SGN_NONE != H5Tget_sign(id())); }
    size_t      get_size_bytes() const   { return H5Tget_size(id()); }
    const char* get_class_string() const
    {
        const char* c = "H5T_NO_CLASS";
        switch(get_class())
        {
        case H5T_INTEGER:   c = "H5T_INTEGER";   break;
        case H5T_FLOAT:     c = "H5T_FLOAT";     break;
        case H5T_STRING:    c = "H5T_STRING";    break;
        case H5T_BITFIELD:  c = "H5T_BITFIELD";  break;
        case H5T_OPAQUE:    c = "H5T_OPAQUE";    break;
        case H5T_COMPOUND:  c = "H5T_COMPOUND";  break;
        case H5T_REFERENCE: c = "H5T_REFERENCE"; break;
        case H5T_ENUM:      c = "H5T_ENUM";      break;
        case H5T_VLEN:      c = "H5T_VLEN";      break;
        case H5T_ARRAY:     c = "H5T_ARRAY";     break;
        default:                                 break;
        }
        return c;
    }
};
// clang-format on

// clang-format off
////////////////////////////////////////////////////////////////////////
// hdf5hpp::hdf5_dataset
class hdf5_dataset: public hdf5_object
{
public:
    hdf5_dataset(hid_t id = ilwalid_hid) : hdf5_object(id) {}
    hdf5_dataset(hdf5_dataset&& d) : hdf5_object(std::forward<hdf5_object>(d)) {}
    ~hdf5_dataset() { if(is_valid()) H5Dclose(id()); }
    hdf5_dataspace get_dataspace() const { return hdf5_dataspace(H5Dget_space(id())); }
    hdf5_datatype  get_datatype()  const { return hdf5_datatype(H5Dget_type(id()));   }
    size_t         get_buffer_size_bytes() const
    {
        return (get_dataspace().get_num_elements() * get_datatype().get_size_bytes());
    }
    void read(void* buffer)
    {
        herr_t h = H5Dread(id(),                // dataset ID
                           get_datatype().id(), // in-memory datatype
                           H5S_ALL,             // in-memory dataspace
                           H5S_ALL,             // in-file dataspace
                           H5P_DEFAULT,         // transfer property list
                           buffer);             // destination buffer
        if(h < 0) throw hdf5_exception();
    }
};
// clang-format on

// clang-format off
////////////////////////////////////////////////////////////////////////
// hdf5hpp::hdf5_file
class hdf5_file : public hdf5_object
{
public:
    hdf5_file(hid_t id = ilwalid_hid) : hdf5_object(id) {}
    hdf5_file(hdf5_file&& f) : hdf5_object(std::forward<hdf5_object>(f)) {}
    ~hdf5_file() { if(is_valid()) H5Fclose(id()); }
    hdf5_file& operator=(hdf5_file&& f)
    {
        if(is_valid()) H5Fclose(id());
        set_ilwalid();
        hdf5_object::operator=(std::move(f));
        return *this;
    }
    hdf5_dataset open_dataset(const char* name, hid_t dapl_id = H5P_DEFAULT)
    {
        hid_t h = H5Dopen(id(), name, dapl_id);
        if(!id_is_valid(h)) throw hdf5_exception();
        return hdf5_dataset(h);
    }
    static hdf5_file create(const char* name, unsigned flags = H5F_ACC_TRUNC, hid_t cpl = H5P_DEFAULT, hid_t apl = H5P_DEFAULT)
    {
        hid_t f = H5Fcreate(name, flags, cpl, apl);
        if(!id_is_valid(f)) throw hdf5_exception();
        return hdf5_file(f);
    }
    static hdf5_file open(const char* name, unsigned flags = H5F_ACC_RDONLY, hid_t apl = H5P_DEFAULT)
    {
        hid_t f = H5Fopen(name, flags, apl);
        if(!id_is_valid(f)) throw hdf5_exception();
        return hdf5_file(f);
    }
};
// clang-format on

} // namespace hdf5hpp

#endif // !defined(HDF5HPP_HPP_INCLUDED_)
