/*
 * Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 */

#include "lwphy_hdf5.h"
#include <limits>
#include <array>
#include <algorithm>
#include "tensor_desc.hpp"
#include "type_colwert.hpp"

namespace
{

struct tensor_descriptor_info
{
    lwphyDataType_t                dataType;
    int                            numDims;
    std::array<int, LWPHY_DIM_MAX> dimensions; // only 1st numDims elements are valid
    std::array<int, LWPHY_DIM_MAX> strides;    // only 1st numDims elements are valid
};

////////////////////////////////////////////////////////////////////////
// generate_native_HDF5_complex_type()
// Note: caller should call H5Tclose() on the returned type
hid_t generate_native_HDF5_complex_type(hid_t elementType)
{
    size_t elementSizeBytes = H5Tget_size(elementType);
    if(elementSizeBytes <= 0)
    {
        return -1;
    }
    hid_t  cType = H5Tcreate(H5T_COMPOUND, 2 * elementSizeBytes);
    if(cType < 0)
    {
        return cType;
    }
    if((H5Tinsert(cType, "re", 0, elementType) < 0) ||
       (H5Tinsert(cType, "im", elementSizeBytes, elementType) < 0))
    {
        H5Tclose(cType);
        cType = -1;
    }
    return cType;
}

////////////////////////////////////////////////////////////////////////
// generate_native_HDF5_fp16_type()
// Note: caller should call H5Tclose() on the returned type
hid_t generate_native_HDF5_fp16_type()
{
    //------------------------------------------------------------------
    // Copy an existing floating point type as a starting point.
    hid_t cType = H5Tcopy(H5T_NATIVE_FLOAT);
    if(cType < 0)
    {
        return cType;
    }
    //------------------------------------------------------------------
    // https://en.wikipedia.org/wiki/Half-precision_floating-point_format
    // sign_pos = 15
    // exp_pos  = 10
    // exp_size = 5
    // mantissa_pos = 0
    // mantissa_size = 10
    // Order is important: we should not set the size before adjusting
    // the fields.
    if((H5Tset_fields(cType, 15, 10, 5, 0, 10) < 0) ||
       (H5Tset_precision(cType, 16)            < 0) ||
       (H5Tset_ebias(cType, 15)                < 0) ||
       (H5Tset_size(cType, 2)                  < 0))
    {
        H5Tclose(cType);
        cType = -1;
    }
    return cType;
}

// clang-format off
////////////////////////////////////////////////////////////////////////
// native_HDF5_type_from_lwphy_type()
// Note: caller should call H5Tclose() on the returned type
hid_t native_HDF5_type_from_lwphy_type(lwphyDataType_t t)
{
    switch(t)
    {
    case LWPHY_R_32F:    return H5Tcopy(H5T_NATIVE_FLOAT);
    case LWPHY_R_64F:    return H5Tcopy(H5T_NATIVE_DOUBLE);
    case LWPHY_R_8I:     return H5Tcopy(H5T_NATIVE_INT8);
    case LWPHY_R_8U:     return H5Tcopy(H5T_NATIVE_UINT8);
    case LWPHY_R_16I:    return H5Tcopy(H5T_NATIVE_INT16);
    case LWPHY_R_16U:    return H5Tcopy(H5T_NATIVE_UINT16);
    case LWPHY_R_32I:    return H5Tcopy(H5T_NATIVE_INT32);
    case LWPHY_R_32U:    return H5Tcopy(H5T_NATIVE_UINT32);
        
    case LWPHY_C_8I:     return generate_native_HDF5_complex_type(H5T_NATIVE_INT8);
    case LWPHY_C_8U:     return generate_native_HDF5_complex_type(H5T_NATIVE_UINT8);
    case LWPHY_C_16I:    return generate_native_HDF5_complex_type(H5T_NATIVE_INT16);
    case LWPHY_C_16U:    return generate_native_HDF5_complex_type(H5T_NATIVE_UINT16);
    case LWPHY_C_32I:    return generate_native_HDF5_complex_type(H5T_NATIVE_INT32);
    case LWPHY_C_32U:    return generate_native_HDF5_complex_type(H5T_NATIVE_UINT32);
    case LWPHY_C_32F:    return generate_native_HDF5_complex_type(H5T_NATIVE_FLOAT);
    case LWPHY_C_64F:    return generate_native_HDF5_complex_type(H5T_NATIVE_DOUBLE);

    case LWPHY_R_16F:    return generate_native_HDF5_fp16_type();
    case LWPHY_C_16F:
        {
            hid_t fp16Type        = generate_native_HDF5_fp16_type();
            hid_t fp16ComplexType = generate_native_HDF5_complex_type(fp16Type);
            H5Tclose(fp16Type);
            return fp16ComplexType;
        }
        
    case LWPHY_VOID:
    case LWPHY_BIT:
    default:
        // No valid native HDF5 representation, so we return an invalid hid_t
        return -1;
    }
}
// clang-format on

////////////////////////////////////////////////////////////////////////
// get_storage_type
// Returns a lwPHY type that can be used as the destination type, for
// file storage. This is useful for cases where we want to perform
// implicit colwersion. Example: To store a fp16 value into a file, we
// will implicitly colwert it to fp32 first.
// This type should be used as the destination type for the "colwert
// tensor" operations.
lwphyDataType_t get_storage_type(lwphyDataType_t t)
{
    switch(t)
    {
    case LWPHY_R_32F:
    case LWPHY_R_64F:
    case LWPHY_R_8I:
    case LWPHY_R_8U:
    case LWPHY_R_16I:
    case LWPHY_R_16U:
    case LWPHY_R_32I:
    case LWPHY_R_32U:
    case LWPHY_C_8I:
    case LWPHY_C_8U:
    case LWPHY_C_16I:
    case LWPHY_C_16U:
    case LWPHY_C_32I:
    case LWPHY_C_32U:
    case LWPHY_C_32F:
    case LWPHY_C_64F:
    case LWPHY_R_16F:
    case LWPHY_C_16F:
        // No colwersion necessary for these types
        return t;
    case LWPHY_BIT:
        // Store bits by default as unsigned 8-bit integers
        return LWPHY_R_8U;
    case LWPHY_VOID:
    default:
        // No lwrrently supported HDF5 representation or implementation,
        // so we return LWPHY_VOID
        return LWPHY_VOID;
    }
}

////////////////////////////////////////////////////////////////////////
// lwphy_type_from_HDF5_datatype()
// Maps the HDF5 type to a lwPhyDataType_t. If no mapping is possible,
// the return type will by LWPHY_VOID.
lwphyDataType_t lwphy_type_from_HDF5_datatype(hid_t h5Datatype)
{
    lwphyDataType_t lwphyType = LWPHY_VOID;
    size_t          typeSize  = H5Tget_size(h5Datatype);
    H5T_class_t     H5class   = H5Tget_class(h5Datatype);
    switch(H5class)
    {
    case H5T_INTEGER:
        // clang-format off
        {
            H5T_sign_t sgn      = H5Tget_sign(h5Datatype);
            bool       isSigned = (H5T_SGN_2 == sgn);
            switch(typeSize)
            {
            case 1:
                lwphyType = isSigned ? LWPHY_R_8I  : LWPHY_R_8U;
                break;
            case 2:
                lwphyType = isSigned ? LWPHY_R_16I : LWPHY_R_16U;
                break;
            case 4:
                lwphyType = isSigned ? LWPHY_R_32I : LWPHY_R_32U;
            default:
                break;
            }
        }
        // clang-format on
        break;
    case H5T_FLOAT:
        if(sizeof(float) == typeSize)
        {
            lwphyType = LWPHY_R_32F;
        }
        else if(sizeof(double) == typeSize)
        {
            lwphyType = LWPHY_R_64F;
        }
        else if(sizeof(__half_raw) == typeSize)
        {
            lwphyType = LWPHY_R_16F;
        }
        break;
    case H5T_COMPOUND: // Complex data
        // Verify that the compound structure has two fields, with names
        // "re" and "im".
        {
            int numMembers = H5Tget_nmembers(h5Datatype);
            if((2 == numMembers) &&
               (H5Tget_member_index(h5Datatype, "re") == 0) &&
               (H5Tget_member_index(h5Datatype, "im") == 1))
            {
                H5T_class_t reClass = H5Tget_member_class(h5Datatype, 0);
                H5T_class_t imClass = H5Tget_member_class(h5Datatype, 1);
                // Types must be the same, and must be either H5T_INTEGER or H5T_FLOAT
                if((reClass == imClass) &&
                   ((H5T_INTEGER == reClass) || (H5T_FLOAT == reClass)))
                {
                    hid_t  reType = H5Tget_member_type(h5Datatype, 0);
                    hid_t  imType = H5Tget_member_type(h5Datatype, 1);
                    size_t reSize = H5Tget_size(reType);
                    size_t imSize = H5Tget_size(imType);
                    if(reSize == imSize)
                    {
                        if(H5T_FLOAT == reClass)
                        {
                            if(sizeof(float) == reSize)
                            {
                                lwphyType = LWPHY_C_32F;
                            }
                            else if(sizeof(double) == reSize)
                            {
                                lwphyType = LWPHY_C_64F;
                            }
                            else if(sizeof(__half_raw) == reSize)
                            {
                                lwphyType = LWPHY_C_16F;
                            }
                        }
                        else
                        {
                            H5T_sign_t reSign = H5Tget_sign(reType);
                            H5T_sign_t imSign = H5Tget_sign(imType);
                            if(reSign == imSign)
                            {
                                bool isSigned = (H5T_SGN_2 == reSign);
                                switch(reSize)
                                {
                                case 1:
                                    lwphyType = isSigned ? LWPHY_C_8I : LWPHY_C_8U;
                                    break;
                                case 2:
                                    lwphyType = isSigned ? LWPHY_C_16I : LWPHY_C_16U;
                                    break;
                                case 4:
                                    lwphyType = isSigned ? LWPHY_C_32I : LWPHY_C_32U;
                                default:
                                    break;
                                }
                            }
                        }
                    }
                    H5Tclose(imType);
                    H5Tclose(reType);
                } // if 2 members have the same class, which is INTEGER or FLOAT
            }     // if 2 members are named "re" and "im"
        }
        break;
    default:
        // Class is not one of INTEGER, FLOAT, or COMPOUND
        break;
    }

    return lwphyType;
}

////////////////////////////////////////////////////////////////////////
// lwphy_type_from_HDF5_dataset()
// Maps the HDF5 type to a lwPhyDataType_t. If no mapping is possible,
// the return type will by LWPHY_VOID.
lwphyDataType_t lwphy_type_from_HDF5_dataset(hid_t h5Dataset)
{
    hid_t h5Datatype = H5Dget_type(h5Dataset);
    if(h5Datatype < 0)
    {
        return LWPHY_VOID;
    }
    else
    {
        lwphyDataType_t lwphyType  = lwphy_type_from_HDF5_datatype(h5Datatype);
        H5Tclose(h5Datatype);
        return lwphyType;
    }
}


////////////////////////////////////////////////////////////////////////
// get_HDF5_dataset_info()
// Populates the given tensor_descriptor_info structure with information
// obtained from the HDF5 dataset found in a file.
// Note that the order of lwphy tensors dimensions is opposite that of
// HDF5, so we reverse the order.
// Returns one of the following values:
// LWPHYHDF5_STATUS_SUCCESS
// LWPHYHDF5_STATUS_DATATYPE_ERROR
//     The data type of the HDF5 dataset is not supported by lwPHY
// LWPHYHDF5_STATUS_DATASPACE_ERROR
//     HDF5 library error querying the dataspace structure
// LWPHYHDF5_STATUS_UNSUPPORTED_RANK
//     The rank of the dataset is larger than supported by lwPHY
// LWPHYHDF5_STATUS_DIMENSION_TOO_LARGE
//     The size of the dimension is larger than supported by lwPHY tensors
lwphyHDF5Status_t get_HDF5_dataset_info(tensor_descriptor_info& tdi,
                                        hid_t                   h5Dataset)
{
    tdi.dimensions.fill(0);
    tdi.strides.fill(0);
    //------------------------------------------------------------------
    // Get the lwphyDataType_t that corresponds to the HDF5 Datatype
    tdi.dataType = lwphy_type_from_HDF5_dataset(h5Dataset);
    if(LWPHY_VOID == tdi.dataType)
    {
        return LWPHYHDF5_STATUS_DATATYPE_ERROR;
    }
    //------------------------------------------------------------------
    // Get the HDF5 Dataspace to determine the bounds of the tensor
    hid_t h5Dataspace = H5Dget_space(h5Dataset);
    if(h5Dataspace < 0)
    {
        return LWPHYHDF5_STATUS_DATASPACE_ERROR;
    }
    lwphyHDF5Status_t status = LWPHYHDF5_STATUS_SUCCESS;
    tdi.numDims              = H5Sget_simple_extent_ndims(h5Dataspace);
    if((tdi.numDims <= 0) || (tdi.numDims > LWPHY_DIM_MAX))
    {
        status = LWPHYHDF5_STATUS_UNSUPPORTED_RANK;
    }
    else
    {
        std::array<hsize_t, LWPHY_DIM_MAX> dims;
        if(H5Sget_simple_extent_dims(h5Dataspace, dims.data(), nullptr) < 0)
        {
            status = LWPHYHDF5_STATUS_DATASPACE_ERROR;
        }
        else
        {
            // Check the size of the dimensions to make sure they
            // are not too large to be represented by an integer
            for(int i = 0; i < tdi.numDims; ++i)
            {
                if(dims[i] > std::numeric_limits<int>::max())
                {
                    status = LWPHYHDF5_STATUS_DIMENSION_TOO_LARGE;
                    break;
                }
                tdi.dimensions[i] = static_cast<int>(dims[i]);
            }
            // Reverse the order of the dimensions  to reflect the
            // different colwentions between lwphy and HDF5
            std::reverse(tdi.dimensions.begin(), tdi.dimensions.begin() + tdi.numDims);

            // Callwlate the strides, assuming tightly packed.
            tdi.strides[0] = 1;
            for(size_t i = 1; i < tdi.numDims; ++i)
            {
                tdi.strides[i] = tdi.strides[i - 1] * tdi.dimensions[i - 1];
            }
        }
    }
    H5Sclose(h5Dataspace);
    return status;
}

template <typename T, typename U>
struct have_same_signedness
{
    constexpr static bool value = (std::is_signed<T>::value == std::is_signed<U>::value);
};

template <typename T, typename U> T lwphy_narrow_cast(U u) { return static_cast<T>(u); };
template <> __half lwphy_narrow_cast(signed char s)    { return static_cast<__half>(static_cast<float>(s)); };
template <> __half lwphy_narrow_cast(unsigned char u)  { return static_cast<__half>(static_cast<float>(u)); };
template <> __half lwphy_narrow_cast(short s)          { return static_cast<__half>(static_cast<float>(s)); };
template <> __half lwphy_narrow_cast(unsigned short s) { return static_cast<__half>(static_cast<float>(s)); };
template <> __half lwphy_narrow_cast(int i)            { return static_cast<__half>(static_cast<float>(i)); };
template <> __half lwphy_narrow_cast(unsigned int i)   { return static_cast<__half>(static_cast<float>(i)); };
    
// Narrowing casts
// See also:
// https://stackoverflow.com/questions/52863643/understanding-gslnarrow-implementation
template <typename TSrc, typename TDst>
bool safe_narrow_cast(TDst& dst, TSrc src)
{
    dst = lwphy_narrow_cast<TDst>(src);
    if(lwphy_narrow_cast<TSrc>(dst) != src)
    {
        return false;
    }
    // If the types have different "signedness", and if the signs don't
    // match, we return failure.
    if(!have_same_signedness<TDst, TSrc>::value && ((src < TSrc{}) != (dst < TDst{})))
    {
        return false;
    }
    return true;
}

template <typename TSrc, typename TDst>
bool safe_narrow_complex_cast(TDst& dst, TSrc src)
{
    typedef typename scalar_from_complex<TDst>::type TDstScalar;
    typedef typename scalar_from_complex<TSrc>::type TSrcScalar;
    dst.x = lwphy_narrow_cast<TDstScalar>(src.x);
    dst.y = lwphy_narrow_cast<TDstScalar>(src.y);
    if((lwphy_narrow_cast<TSrcScalar>(dst.x) != src.x) ||
       (lwphy_narrow_cast<TSrcScalar>(dst.y) != src.y))
    {
        return false;
    }
    // If the types have different "signedness", and if the signs don't
    // match, we return failure.
    if(!have_same_signedness<TDstScalar, TSrcScalar>::value &&
       (((src.x < TSrcScalar{}) != (dst.x < TDstScalar{})) || ((src.y < TSrcScalar{}) != (dst.y < TDstScalar{}))))
    {
        return false;
    }
    return true;
}

unsigned short half_to_raw(const __half& h)
{
    unsigned short us;
    memcpy(&us, &h, sizeof(us));
    return us;
}

ushort2 half2_to_raw(const __half2& h2)
{
    ushort2 us2;
    memcpy(&us2, &h2, sizeof(us2));
    return us2;
}

// clang-format off
////////////////////////////////////////////////////////////////////////
// colwert_variant()    
lwphyHDF5Status_t colwert_variant(lwphyVariant_t& var,
                                  lwphyDataType_t colwertToType)
{
    switch(var.type)
    {
    case LWPHY_BIT:
        {
            unsigned char b = (0 == var.value.b1) ? 0 : 1;
            // We use static_cast instead of more generic casts (that
            // might take signed/unsigned considerations into account)
            // because here we know that the value is 0 or 1.
            switch(colwertToType)
            {
            case LWPHY_BIT:    /* src and dst types identical */                      break;
            case LWPHY_R_8I:   var.value.r8i  = static_cast<signed char>(b);          break;
            case LWPHY_R_8U:   var.value.r8u  = static_cast<unsigned char>(b);        break;
            case LWPHY_R_16I:  var.value.r16i = static_cast<short>(b);                break;
            case LWPHY_R_16U:  var.value.r16u = static_cast<unsigned short>(b);       break;
            case LWPHY_R_32I:  var.value.r32i = static_cast<int>(b);                  break;
            case LWPHY_R_32U:  var.value.r32u = static_cast<unsigned int>(b);         break;
            case LWPHY_R_16F:  var.value.r16f = half_to_raw(type_colwert<__half>(b)); break;
            case LWPHY_R_32F:  var.value.r32f = static_cast<float>(b);                break;
            case LWPHY_R_64F:  var.value.r64f = static_cast<double>(b);               break;
            default:
                // Colwersion to complex types not supported
                return LWPHYHDF5_STATUS_COLWERT_ERROR;
            }
        }
        break;
    case LWPHY_R_8I:
        {
            bool bSuccess = true;
            switch(colwertToType)
            {
            case LWPHY_BIT:    var.value.b1 = (var.value.r8i == 0) ? 0 : 1;                       break;
            case LWPHY_R_8I:   /* src and dst types identical */                                  break;
            case LWPHY_R_8U:   bSuccess = safe_narrow_cast(var.value.r8u,  var.value.r8i);        break;
            case LWPHY_R_16I:  bSuccess = safe_narrow_cast(var.value.r16i, var.value.r8i);        break;
            case LWPHY_R_16U:  bSuccess = safe_narrow_cast(var.value.r16u, var.value.r8i);        break;
            case LWPHY_R_32I:  var.value.r32i = type_colwert<int>(var.value.r8i);                 break;
            case LWPHY_R_32U:  bSuccess = safe_narrow_cast(var.value.r32u, var.value.r8i);        break;
            case LWPHY_R_16F:  var.value.r16f = half_to_raw(type_colwert<__half>(var.value.r8i)); break;
            case LWPHY_R_32F:  var.value.r32f = type_colwert<float>(var.value.r8i);               break;
            case LWPHY_R_64F:  var.value.r64f = type_colwert<double>(var.value.r8i);              break;
            default:
                // Colwersion to complex types not supported
                return LWPHYHDF5_STATUS_COLWERT_ERROR;
            }
            if(!bSuccess)
            {
                return LWPHYHDF5_STATUS_VALUE_OUT_OF_RANGE;
            }
        }
        break;
    case LWPHY_C_8I:
        {
            bool bSuccess = true;
            switch(colwertToType)
            {
            case LWPHY_C_8I:   /* src and dst types identical */                                    break;
            case LWPHY_C_8U:   bSuccess = safe_narrow_complex_cast(var.value.c8u,  var.value.c8i);  break;
            case LWPHY_C_16I:  bSuccess = safe_narrow_complex_cast(var.value.c16i, var.value.c8i);  break;
            case LWPHY_C_16U:  bSuccess = safe_narrow_complex_cast(var.value.c16u, var.value.c8i);  break;
            case LWPHY_C_32I:  var.value.c32i = type_colwert<int2>(var.value.c8i);                  break;
            case LWPHY_C_32U:  bSuccess = safe_narrow_complex_cast(var.value.c32u, var.value.c8i);  break;
            case LWPHY_C_16F:  var.value.c16f = half2_to_raw(type_colwert<__half2>(var.value.c8i)); break;
            case LWPHY_C_32F:  var.value.c32f = type_colwert<lwComplex>(var.value.c8i);             break;
            case LWPHY_C_64F:  var.value.c64f = type_colwert<lwDoubleComplex>(var.value.c8i);       break;
            default:
                // Colwersion to bit and real types from complex inputs not supported
                return LWPHYHDF5_STATUS_COLWERT_ERROR;
            }
            if(!bSuccess)
            {
                return LWPHYHDF5_STATUS_VALUE_OUT_OF_RANGE;
            }
        }
        break;
    case LWPHY_R_8U:
        {
            bool bSuccess = true;
            switch(colwertToType)
            {
            case LWPHY_BIT:    var.value.b1   = (var.value.r8u == 0) ? 0 : 1;                     break;
            case LWPHY_R_8I:   bSuccess       = safe_narrow_cast(var.value.r8i,  var.value.r8u);  break;
            case LWPHY_R_8U:   /* src and dst types identical */                                  break;
            case LWPHY_R_16I:  bSuccess       = safe_narrow_cast(var.value.r16i, var.value.r8u);  break;
            case LWPHY_R_16U:  bSuccess       = type_colwert<unsigned short>(var.value.r8u);      break;
            case LWPHY_R_32I:  bSuccess       = safe_narrow_cast(var.value.r32i, var.value.r8u);  break;
            case LWPHY_R_32U:  var.value.r32u = static_cast<unsigned int>(var.value.r8u);         break;
            case LWPHY_R_16F:  var.value.r16f = half_to_raw(type_colwert<__half>(var.value.r8u)); break;
            case LWPHY_R_32F:  var.value.r32f = type_colwert<float>(var.value.r8u);               break;
            case LWPHY_R_64F:  var.value.r64f = type_colwert<double>(var.value.r8u);              break;
            default:
                // Colwersion to complex types not supported
                return LWPHYHDF5_STATUS_COLWERT_ERROR;
            }
            if(!bSuccess)
            {
                return LWPHYHDF5_STATUS_VALUE_OUT_OF_RANGE;
            }
        }
        break;
    case LWPHY_C_8U:
        {
            bool bSuccess = true;
            switch(colwertToType)
            {
            case LWPHY_C_8I:   bSuccess = safe_narrow_complex_cast(var.value.c8i, var.value.c8u);   break;
            case LWPHY_C_8U:   /* src and dst types identical */                                    break;
            case LWPHY_C_16I:  bSuccess = safe_narrow_complex_cast(var.value.c16i, var.value.c8u);  break;
            case LWPHY_C_16U:  bSuccess = safe_narrow_complex_cast(var.value.c16u, var.value.c8u);  break;
            case LWPHY_C_32I:  bSuccess = safe_narrow_complex_cast(var.value.c32i, var.value.c8u);  break;
            case LWPHY_C_32U:  var.value.c32u = type_colwert<uint2>(var.value.c8u);                 break;
            case LWPHY_C_16F:  var.value.c16f = half2_to_raw(type_colwert<__half2>(var.value.c8u)); break;
            case LWPHY_C_32F:  var.value.c32f = type_colwert<lwComplex>(var.value.c8u);             break;
            case LWPHY_C_64F:  var.value.c64f = type_colwert<lwDoubleComplex>(var.value.c8u);       break;
            default:
                // Colwersion to bit and real types from complex inputs not supported
                return LWPHYHDF5_STATUS_COLWERT_ERROR;
            }
            if(!bSuccess)
            {
                return LWPHYHDF5_STATUS_VALUE_OUT_OF_RANGE;
            }
        }
        break;
    case LWPHY_R_16I:
        {
            bool bSuccess = true;
            switch(colwertToType)
            {
            case LWPHY_BIT:    var.value.b1 = (var.value.r16i == 0) ? 0 : 1;                                                  break;
            case LWPHY_R_8I:   bSuccess = safe_narrow_cast(var.value.r8i, var.value.r16i);                                    break;
            case LWPHY_R_8U:   bSuccess = safe_narrow_cast(var.value.r8u, var.value.r16i);                                    break;
            case LWPHY_R_16I:  /* src and dst types identical */                                                              break;
            case LWPHY_R_16U:  bSuccess = safe_narrow_cast(var.value.r16u, var.value.r16i);                                   break;
            case LWPHY_R_32I:  var.value.r32i = type_colwert<int>(var.value.r16i);                                            break;
            case LWPHY_R_32U:  bSuccess = safe_narrow_cast(var.value.r32u, var.value.r16i);                                   break;
            case LWPHY_R_16F:  { __half h; bSuccess = safe_narrow_cast(h, var.value.r16i); var.value.r16f = half_to_raw(h); } break;
            case LWPHY_R_32F:  var.value.r32f = type_colwert<float>(var.value.r16i);                                          break;
            case LWPHY_R_64F:  var.value.r64f = type_colwert<double>(var.value.r16i);                                         break;
            default:
                // Colwersion to complex types not supported
                return LWPHYHDF5_STATUS_COLWERT_ERROR;
            }
            if(!bSuccess)
            {
                return LWPHYHDF5_STATUS_VALUE_OUT_OF_RANGE;
            }
        }
        break;
    case LWPHY_C_16I:
        {
            bool bSuccess = true;
            switch(colwertToType)
            {
            case LWPHY_C_8I:   bSuccess = safe_narrow_complex_cast(var.value.c8i,  var.value.c16i);                                     break;
            case LWPHY_C_8U:   bSuccess = safe_narrow_complex_cast(var.value.c8u,  var.value.c16i);                                     break;
            case LWPHY_C_16I:  /* src and dst types identical */                                                                        break;
            case LWPHY_C_16U:  bSuccess = safe_narrow_complex_cast(var.value.c16u, var.value.c16i);                                     break;
            case LWPHY_C_32I:  var.value.c32i = type_colwert<int2>(var.value.c8i);                                                      break;
            case LWPHY_C_32U:  bSuccess = safe_narrow_complex_cast(var.value.c32u, var.value.c16i);                                     break;
            case LWPHY_C_16F:  { __half2 h; bSuccess = safe_narrow_complex_cast(h, var.value.c16i); var.value.c16f = half2_to_raw(h); } break;
            case LWPHY_C_32F:  var.value.c32f = type_colwert<lwComplex>(var.value.c16i);                                                break;
            case LWPHY_C_64F:  var.value.c64f = type_colwert<lwDoubleComplex>(var.value.c16i);                                          break;
            default:
                // Colwersion to bit and real types from complex inputs not supported
                return LWPHYHDF5_STATUS_COLWERT_ERROR;
            }
            if(!bSuccess)
            {
                return LWPHYHDF5_STATUS_VALUE_OUT_OF_RANGE;
            }
        }
        break;
    case LWPHY_R_16U:
        {
            bool bSuccess = true;
            switch(colwertToType)
            {
            case LWPHY_BIT:    var.value.b1   = (var.value.r16u == 0) ? 0 : 1;                                                break;
            case LWPHY_R_8I:   bSuccess       = safe_narrow_cast(var.value.r8i, var.value.r16u);                              break;
            case LWPHY_R_8U:   bSuccess       = safe_narrow_cast(var.value.r8u, var.value.r16u);                              break;
            case LWPHY_R_16I:  bSuccess       = safe_narrow_cast(var.value.r16i, var.value.r16u);                             break;
            case LWPHY_R_16U:  /* src and dst types identical */                                                              break;
            case LWPHY_R_32I:  bSuccess       = safe_narrow_cast(var.value.r32i, var.value.r16u);                             break;
            case LWPHY_R_32U:  var.value.r32u = static_cast<unsigned int>(var.value.r16u);                                    break;
            case LWPHY_R_16F:  { __half h; bSuccess = safe_narrow_cast(h, var.value.r16u); var.value.r16f = half_to_raw(h); } break;
            case LWPHY_R_32F:  var.value.r32f = type_colwert<float>(var.value.r16u);                                          break;
            case LWPHY_R_64F:  var.value.r64f = type_colwert<double>(var.value.r16u);                                         break;
            default:
                // Colwersion to complex types not supported
                return LWPHYHDF5_STATUS_COLWERT_ERROR;
            }
            if(!bSuccess)
            {
                return LWPHYHDF5_STATUS_VALUE_OUT_OF_RANGE;
            }
        }
        break;
    case LWPHY_C_16U:
        {
            bool bSuccess = true;
            switch(colwertToType)
            {
            case LWPHY_C_8I:   bSuccess = safe_narrow_complex_cast(var.value.c8i, var.value.c16u);                                      break;
            case LWPHY_C_8U:   bSuccess = safe_narrow_complex_cast(var.value.c8i, var.value.c16u);                                      break;
            case LWPHY_C_16I:  bSuccess = safe_narrow_complex_cast(var.value.c16i, var.value.c16u);                                     break;
            case LWPHY_C_16U:  /* src and dst types identical */                                                                        break;
            case LWPHY_C_32I:  bSuccess = safe_narrow_complex_cast(var.value.c32i, var.value.c16u);                                     break;
            case LWPHY_C_32U:  var.value.c32u = type_colwert<uint2>(var.value.c16u);                                                    break;
            case LWPHY_C_16F:  { __half2 h; bSuccess = safe_narrow_complex_cast(h, var.value.c16u); var.value.c16f = half2_to_raw(h); } break;
            case LWPHY_C_32F:  var.value.c32f = type_colwert<lwComplex>(var.value.c16u);                                                break;
            case LWPHY_C_64F:  var.value.c64f = type_colwert<lwDoubleComplex>(var.value.c16u);                                          break;
            default:
                // Colwersion to bit and real types from complex inputs not supported
                return LWPHYHDF5_STATUS_COLWERT_ERROR;
            }
            if(!bSuccess)
            {
                return LWPHYHDF5_STATUS_VALUE_OUT_OF_RANGE;
            }
        }
        break;
    case LWPHY_R_32I:
        {
            bool bSuccess = true;
            switch(colwertToType)
            {
            case LWPHY_BIT:    var.value.b1 = (var.value.r32i == 0) ? 0 : 1;                                                  break;
            case LWPHY_R_8I:   bSuccess = safe_narrow_cast(var.value.r8i, var.value.r32i);                                    break;
            case LWPHY_R_8U:   bSuccess = safe_narrow_cast(var.value.r8u, var.value.r32i);                                    break;
            case LWPHY_R_16I:  bSuccess = safe_narrow_cast(var.value.r16i, var.value.r32i);                                   break;
            case LWPHY_R_16U:  bSuccess = safe_narrow_cast(var.value.r16u, var.value.r32i);                                   break;
            case LWPHY_R_32I:  /* src and dst types identical */                                                              break;
            case LWPHY_R_32U:  bSuccess = safe_narrow_cast(var.value.r32u, var.value.r32i);                                   break;
            case LWPHY_R_16F:  { __half h; bSuccess = safe_narrow_cast(h, var.value.r32i); var.value.r16f = half_to_raw(h); } break;
            case LWPHY_R_32F:  bSuccess = safe_narrow_cast(var.value.r32f, var.value.r32i);                                   break;
            case LWPHY_R_64F:  var.value.r64f = type_colwert<double>(var.value.r32i);                                         break;
            default:
                // Colwersion to complex types not supported
                return LWPHYHDF5_STATUS_COLWERT_ERROR;
            }
            if(!bSuccess)
            {
                return LWPHYHDF5_STATUS_VALUE_OUT_OF_RANGE;
            }
        }
        break;
    case LWPHY_C_32I:
        {
            bool bSuccess = true;
            switch(colwertToType)
            {
            case LWPHY_C_8I:   bSuccess = safe_narrow_complex_cast(var.value.c8i,  var.value.c32i);                                     break;
            case LWPHY_C_8U:   bSuccess = safe_narrow_complex_cast(var.value.c8u,  var.value.c32i);                                     break;
            case LWPHY_C_16I:  bSuccess = safe_narrow_complex_cast(var.value.c16i, var.value.c32i);                                     break;
            case LWPHY_C_16U:  bSuccess = safe_narrow_complex_cast(var.value.c16u, var.value.c32i);                                     break;
            case LWPHY_C_32I:  /* src and dst types identical */                                                                        break;
            case LWPHY_C_32U:  bSuccess = safe_narrow_complex_cast(var.value.c32u, var.value.c32i);                                     break;
            case LWPHY_C_16F:  { __half2 h; bSuccess = safe_narrow_complex_cast(h, var.value.c32i); var.value.c16f = half2_to_raw(h); } break;
            case LWPHY_C_32F:  bSuccess = safe_narrow_complex_cast(var.value.c32f, var.value.c32i);                                     break;
            case LWPHY_C_64F:  var.value.c64f = type_colwert<lwDoubleComplex>(var.value.c32i);                                          break;
            default:
                // Colwersion to bit and real types from complex inputs not supported
                return LWPHYHDF5_STATUS_COLWERT_ERROR;
            }
            if(!bSuccess)
            {
                return LWPHYHDF5_STATUS_VALUE_OUT_OF_RANGE;
            }
        }
        break;
    case LWPHY_R_32U:
        {
            bool bSuccess = true;
            switch(colwertToType)
            {
            case LWPHY_BIT:    var.value.b1   = (var.value.r32u == 0) ? 0 : 1;                                                break;
            case LWPHY_R_8I:   bSuccess       = safe_narrow_cast(var.value.r8i, var.value.r32u);                              break;
            case LWPHY_R_8U:   bSuccess       = safe_narrow_cast(var.value.r8u, var.value.r32u);                              break;
            case LWPHY_R_16I:  bSuccess       = safe_narrow_cast(var.value.r16i, var.value.r32u);                             break;
            case LWPHY_R_16U:  bSuccess       = safe_narrow_cast(var.value.r16u, var.value.r32u);                             break;
            case LWPHY_R_32I:  bSuccess       = safe_narrow_cast(var.value.r32i, var.value.r32u);                             break;
            case LWPHY_R_32U:  /* src and dst types identical */                                                              break;
            case LWPHY_R_16F:  { __half h; bSuccess = safe_narrow_cast(h, var.value.r32u); var.value.r16f = half_to_raw(h); } break;
            case LWPHY_R_32F:  bSuccess       = safe_narrow_cast(var.value.r32f, var.value.r32u);                             break;
            case LWPHY_R_64F:  var.value.r64f = type_colwert<double>(var.value.r32u);                                         break;
            default:
                // Colwersion to complex types not supported
                return LWPHYHDF5_STATUS_COLWERT_ERROR;
            }
            if(!bSuccess)
            {
                return LWPHYHDF5_STATUS_VALUE_OUT_OF_RANGE;
            }
        }
        break;
    case LWPHY_C_32U:
        {
            bool bSuccess = true;
            switch(colwertToType)
            {
            case LWPHY_C_8I:   bSuccess = safe_narrow_complex_cast(var.value.c8i, var.value.c32u);                                      break;
            case LWPHY_C_8U:   bSuccess = safe_narrow_complex_cast(var.value.c8i, var.value.c32u);                                      break;
            case LWPHY_C_16I:  bSuccess = safe_narrow_complex_cast(var.value.c16i, var.value.c32u);                                     break;
            case LWPHY_C_16U:  bSuccess = safe_narrow_complex_cast(var.value.c16i, var.value.c32u);                                     break;
            case LWPHY_C_32I:  bSuccess = safe_narrow_complex_cast(var.value.c32i, var.value.c32u);                                     break;
            case LWPHY_C_32U:  /* src and dst types identical */                                                                        break;
            case LWPHY_C_16F:  { __half2 h; bSuccess = safe_narrow_complex_cast(h, var.value.c32u); var.value.c16f = half2_to_raw(h); } break;
            case LWPHY_C_32F:  bSuccess = safe_narrow_complex_cast(var.value.c32f, var.value.c32u);                                     break;
            case LWPHY_C_64F:  var.value.c64f = type_colwert<lwDoubleComplex>(var.value.c32u);                                          break;
            default:
                // Colwersion to bit and real types from complex inputs not supported
                return LWPHYHDF5_STATUS_COLWERT_ERROR;
            }
            if(!bSuccess)
            {
                return LWPHYHDF5_STATUS_VALUE_OUT_OF_RANGE;
            }
        }
        break;
    case LWPHY_R_16F:
        {
            bool bSuccess = true;
            switch(colwertToType)
            {
            case LWPHY_BIT:    var.value.b1 = (var.value.r16f == 0) ? 0 : 1;                break;
            case LWPHY_R_8I:   bSuccess = safe_narrow_cast(var.value.r8i,  var.value.r16f); break;
            case LWPHY_R_8U:   bSuccess = safe_narrow_cast(var.value.r8u,  var.value.r16f); break;
            case LWPHY_R_16I:  bSuccess = safe_narrow_cast(var.value.r16i, var.value.r16f); break;
            case LWPHY_R_16U:  bSuccess = safe_narrow_cast(var.value.r16u, var.value.r16f); break;
            case LWPHY_R_32I:  bSuccess = safe_narrow_cast(var.value.r32i, var.value.r16f); break;
            case LWPHY_R_32U:  bSuccess = safe_narrow_cast(var.value.r32u, var.value.r16f); break;
            case LWPHY_R_16F:  /* src and dst types identical */                            break;
            case LWPHY_R_32F:  var.value.r32f = type_colwert<float>(var.value.r16f);        break;
            case LWPHY_R_64F:  var.value.r64f = type_colwert<double>(var.value.r16f);       break;
            default:
                // Colwersion to complex types not supported
                return LWPHYHDF5_STATUS_COLWERT_ERROR;
            }
            if(!bSuccess)
            {
                return LWPHYHDF5_STATUS_VALUE_OUT_OF_RANGE;
            }
        }
        break;
    case LWPHY_C_16F:
        {
            bool bSuccess = true;
            switch(colwertToType)
            {
            case LWPHY_C_8I:   bSuccess = safe_narrow_complex_cast(var.value.c8i,  var.value.c16f); break;
            case LWPHY_C_8U:   bSuccess = safe_narrow_complex_cast(var.value.c8u,  var.value.c16f); break;
            case LWPHY_C_16I:  bSuccess = safe_narrow_complex_cast(var.value.c16i, var.value.c16f); break;
            case LWPHY_C_16U:  bSuccess = safe_narrow_complex_cast(var.value.c16u, var.value.c16f); break;
            case LWPHY_C_32I:  bSuccess = safe_narrow_complex_cast(var.value.c32i, var.value.c16f); break;
            case LWPHY_C_32U:  bSuccess = safe_narrow_complex_cast(var.value.c32u, var.value.c16f); break;
            case LWPHY_C_16F:  /* src and dst types identical */                                    break;
            case LWPHY_C_32F:  var.value.c32f = type_colwert<lwComplex>(var.value.c16f);            break;
            case LWPHY_C_64F:  var.value.c64f = type_colwert<lwDoubleComplex>(var.value.c16f);      break;
            default:
                // Colwersion to bit and real types from complex inputs not supported
                return LWPHYHDF5_STATUS_COLWERT_ERROR;
            }
            if(!bSuccess)
            {
                return LWPHYHDF5_STATUS_VALUE_OUT_OF_RANGE;
            }
        }
        break;
    case LWPHY_R_32F:
        {
            bool bSuccess = true;
            switch(colwertToType)
            {
            case LWPHY_BIT:    var.value.b1 = (var.value.r32f == 0) ? 0 : 1;                                                  break;
            case LWPHY_R_8I:   bSuccess = safe_narrow_cast(var.value.r8i,  var.value.r32f);                                   break;
            case LWPHY_R_8U:   bSuccess = safe_narrow_cast(var.value.r8u,  var.value.r32f);                                   break;
            case LWPHY_R_16I:  bSuccess = safe_narrow_cast(var.value.r16i, var.value.r32f);                                   break;
            case LWPHY_R_16U:  bSuccess = safe_narrow_cast(var.value.r16u, var.value.r32f);                                   break;
            case LWPHY_R_32I:  bSuccess = safe_narrow_cast(var.value.r32i, var.value.r32f);                                   break;
            case LWPHY_R_32U:  bSuccess = safe_narrow_cast(var.value.r32u, var.value.r32f);                                   break;
            case LWPHY_R_16F:  { __half h; bSuccess = safe_narrow_cast(h, var.value.r32f); var.value.r16f = half_to_raw(h); } break;
            case LWPHY_R_32F:  /* src and dst types identical */                                                              break;
            case LWPHY_R_64F:  var.value.r64f = type_colwert<double>(var.value.r32f);                                         break;
            default:
                // Colwersion to complex types not supported
                return LWPHYHDF5_STATUS_COLWERT_ERROR;
            }
            if(!bSuccess)
            {
                return LWPHYHDF5_STATUS_VALUE_OUT_OF_RANGE;
            }
        }
        break;
    case LWPHY_C_32F:
        {
            bool bSuccess = true;
            switch(colwertToType)
            {
            case LWPHY_C_8I:   bSuccess = safe_narrow_complex_cast(var.value.c8i,  var.value.c32f);                                     break;
            case LWPHY_C_8U:   bSuccess = safe_narrow_complex_cast(var.value.c8u,  var.value.c32f);                                     break;
            case LWPHY_C_16I:  bSuccess = safe_narrow_complex_cast(var.value.c16i, var.value.c32f);                                     break;
            case LWPHY_C_16U:  bSuccess = safe_narrow_complex_cast(var.value.c16u, var.value.c32f);                                     break;
            case LWPHY_C_32I:  bSuccess = safe_narrow_complex_cast(var.value.c32i, var.value.c32f);                                     break;
            case LWPHY_C_32U:  bSuccess = safe_narrow_complex_cast(var.value.c32u, var.value.c32f);                                     break;
            case LWPHY_C_16F:  { __half2 h; bSuccess = safe_narrow_complex_cast(h, var.value.c32f); var.value.c16f = half2_to_raw(h); } break;
            case LWPHY_C_32F:  /* src and dst types identical */                                                                        break;
            case LWPHY_C_64F:  var.value.c64f = type_colwert<lwDoubleComplex>(var.value.c32f);                                          break;
            default:
                // Colwersion to bit and real types from complex inputs not supported
                return LWPHYHDF5_STATUS_COLWERT_ERROR;
            }
            if(!bSuccess)
            {
                return LWPHYHDF5_STATUS_VALUE_OUT_OF_RANGE;
            }
        }
        break;
    case LWPHY_R_64F:
        {
            bool bSuccess = true;
            switch(colwertToType)
            {
            case LWPHY_BIT:    var.value.b1 = (var.value.r64f == 0) ? 0 : 1;                                                  break;
            case LWPHY_R_8I:   bSuccess = safe_narrow_cast(var.value.r8i,  var.value.r64f);                                   break;
            case LWPHY_R_8U:   bSuccess = safe_narrow_cast(var.value.r8u,  var.value.r64f);                                   break;
            case LWPHY_R_16I:  bSuccess = safe_narrow_cast(var.value.r16i, var.value.r64f);                                   break;
            case LWPHY_R_16U:  bSuccess = safe_narrow_cast(var.value.r16u, var.value.r64f);                                   break;
            case LWPHY_R_32I:  bSuccess = safe_narrow_cast(var.value.r32i, var.value.r64f);                                   break;
            case LWPHY_R_32U:  bSuccess = safe_narrow_cast(var.value.r32u, var.value.r64f);                                   break;
            case LWPHY_R_16F:  { __half h; bSuccess = safe_narrow_cast(h, var.value.r64f); var.value.r16f = half_to_raw(h); } break;
            case LWPHY_R_32F:  bSuccess = safe_narrow_cast(var.value.r32f, var.value.r64f);                                   break;
            case LWPHY_R_64F:  /* src and dst types identical */                                                              break;
            default:
                // Colwersion to complex types not supported
                return LWPHYHDF5_STATUS_COLWERT_ERROR;
            }
            if(!bSuccess)
            {
                return LWPHYHDF5_STATUS_VALUE_OUT_OF_RANGE;
            }
        }
        break;
    case LWPHY_C_64F:
        {
            bool bSuccess = true;
            switch(colwertToType)
            {
            case LWPHY_C_8I:   bSuccess = safe_narrow_complex_cast(var.value.c8i,  var.value.c64f);                                     break;
            case LWPHY_C_8U:   bSuccess = safe_narrow_complex_cast(var.value.c8u,  var.value.c64f);                                     break;
            case LWPHY_C_16I:  bSuccess = safe_narrow_complex_cast(var.value.c16i, var.value.c64f);                                     break;
            case LWPHY_C_16U:  bSuccess = safe_narrow_complex_cast(var.value.c16u, var.value.c64f);                                     break;
            case LWPHY_C_32I:  bSuccess = safe_narrow_complex_cast(var.value.c32i, var.value.c64f);                                     break;
            case LWPHY_C_32U:  bSuccess = safe_narrow_complex_cast(var.value.c32u, var.value.c64f);                                     break;
            case LWPHY_C_16F:  { __half2 h; bSuccess = safe_narrow_complex_cast(h, var.value.c64f); var.value.c16f = half2_to_raw(h); } break;
            case LWPHY_C_32F:  bSuccess = safe_narrow_complex_cast(var.value.c32f, var.value.c64f);                                     break;
            case LWPHY_C_64F:  /* src and dst types identical */                                                                        break;
            default:
                // Colwersion to bit and real types from complex inputs not supported
                return LWPHYHDF5_STATUS_COLWERT_ERROR;
            }
            if(!bSuccess)
            {
                return LWPHYHDF5_STATUS_VALUE_OUT_OF_RANGE;
            }
        }
        break;
    default:
        // Don't expect to be here...
        return LWPHYHDF5_STATUS_DATATYPE_ERROR;
    }
    // On success, set the output type
    var.type = colwertToType;
    return LWPHYHDF5_STATUS_SUCCESS;
}
// clang-format on

} // namespace


// clang-format off
////////////////////////////////////////////////////////////////////////
// lwphyHDF5GetErrorString()
const char* lwphyHDF5GetErrorString(lwphyHDF5Status_t status)
{
    switch(status)
    {
    case LWPHYHDF5_STATUS_SUCCESS:                return "The API call returned with no errors.";
    case LWPHYHDF5_STATUS_ILWALID_ARGUMENT:       return "One or more of the arguments provided to the function was invalid.";
    case LWPHYHDF5_STATUS_ILWALID_DATASET:        return "The HDF5 dataset argument provided was invalid.";
    case LWPHYHDF5_STATUS_DATATYPE_ERROR:         return "The HDF5 datatype is not supported by the lwPHY library.";
    case LWPHYHDF5_STATUS_DATASPACE_ERROR:        return "The HDF5 library returned an error creating or querying the dataspace.";
    case LWPHYHDF5_STATUS_UNSUPPORTED_RANK:       return "The HDF5 dataspace rank is not supported by lwPHY.";
    case LWPHYHDF5_STATUS_DIMENSION_TOO_LARGE:    return "One or more HDF5 dataspace dimensions are larger than lwPHY supports.";
    case LWPHYHDF5_STATUS_ILWALID_TENSOR_DESC:    return "An invalid tensor descriptor was provided.";
    case LWPHYHDF5_STATUS_INADEQUATE_BUFFER_SIZE: return "The provided buffer size was inadequate.";
    case LWPHYHDF5_STATUS_TENSOR_MISMATCH:        return "Tensor descriptor arguments do not match in rank and/or dimension(s).";
    case LWPHYHDF5_STATUS_UNKNOWN_ERROR:          return "Unknown or unexpected internal error.";
    case LWPHYHDF5_STATUS_ALLOC_FAILED:           return "Memory allocation failed.";
    case LWPHYHDF5_STATUS_TENSOR_DESC_FAILURE:    return "Creating or setting the lwPHY tensor descriptor failed.";
    case LWPHYHDF5_STATUS_READ_ERROR:             return "An HDF5 read error oclwrred.";
    case LWPHYHDF5_STATUS_COLWERT_ERROR:          return "A colwersion error oclwrred, or an unsupported colwersion was requested.";
    case LWPHYHDF5_STATUS_WRITE_ERROR:            return "An HDF5 write error oclwrred.";
    case LWPHYHDF5_STATUS_DATASET_ERROR:          return "An HDF5 dataset creation/query error oclwrred.";
    case LWPHYHDF5_STATUS_ILWALID_NAME:           return "No such scalar or structure field with the given name exists.";
    case LWPHYHDF5_STATUS_INCORRECT_OBJ_TYPE:     return "The HDF5 object provided is not of the correct/expected type.";
    case LWPHYHDF5_STATUS_OBJ_CREATE_FAILURE:     return "HDF5 object creation failure.";
    case LWPHYHDF5_STATUS_VALUE_OUT_OF_RANGE:     return "Data colwersion could not occur because an input value was out of range.";
    default:                                      return "Unknown status.";
    }
}
// clang-format on

// clang-format off
////////////////////////////////////////////////////////////////////////
// lwphyHDF5GetErrorName()
const char* lwphyHDF5GetErrorName(lwphyHDF5Status_t status)
{
    switch(status)
    {
    case LWPHYHDF5_STATUS_SUCCESS:                return "LWPHYHDF5_STATUS_SUCCESS";
    case LWPHYHDF5_STATUS_ILWALID_ARGUMENT:       return "LWPHYHDF5_STATUS_ILWALID_ARGUMENT";
    case LWPHYHDF5_STATUS_ILWALID_DATASET:        return "LWPHYHDF5_STATUS_ILWALID_DATASET";
    case LWPHYHDF5_STATUS_DATATYPE_ERROR:         return "LWPHYHDF5_STATUS_DATATYPE_ERROR";
    case LWPHYHDF5_STATUS_DATASPACE_ERROR:        return "LWPHYHDF5_STATUS_DATASPACE_ERROR";
    case LWPHYHDF5_STATUS_UNSUPPORTED_RANK:       return "LWPHYHDF5_STATUS_UNSUPPORTED_RANK";
    case LWPHYHDF5_STATUS_DIMENSION_TOO_LARGE:    return "LWPHYHDF5_STATUS_DIMENSION_TOO_LARGE";
    case LWPHYHDF5_STATUS_ILWALID_TENSOR_DESC:    return "LWPHYHDF5_STATUS_ILWALID_TENSOR_DESC";
    case LWPHYHDF5_STATUS_INADEQUATE_BUFFER_SIZE: return "LWPHYHDF5_STATUS_INADEQUATE_BUFFER_SIZE";
    case LWPHYHDF5_STATUS_TENSOR_MISMATCH:        return "LWPHYHDF5_STATUS_TENSOR_MISMATCH";
    case LWPHYHDF5_STATUS_UNKNOWN_ERROR:          return "LWPHYHDF5_STATUS_UNKNOWN_ERROR";
    case LWPHYHDF5_STATUS_ALLOC_FAILED:           return "LWPHYHDF5_STATUS_ALLOC_FAILED";
    case LWPHYHDF5_STATUS_TENSOR_DESC_FAILURE:    return "LWPHYHDF5_STATUS_TENSOR_DESC_FAILURE";
    case LWPHYHDF5_STATUS_READ_ERROR:             return "LWPHYHDF5_STATUS_READ_ERROR";
    case LWPHYHDF5_STATUS_COLWERT_ERROR:          return "LWPHYHDF5_STATUS_COLWERT_ERROR";
    case LWPHYHDF5_STATUS_WRITE_ERROR:            return "LWPHYHDF5_STATUS_WRITE_ERROR";
    case LWPHYHDF5_STATUS_DATASET_ERROR:          return "LWPHYHDF5_STATUS_DATASET_ERROR";
    case LWPHYHDF5_STATUS_ILWALID_NAME:           return "LWPHYHDF5_STATUS_ILWALID_NAME";
    case LWPHYHDF5_STATUS_INCORRECT_OBJ_TYPE:     return "LWPHYHDF5_STATUS_INCORRECT_OBJ_TYPE";
    case LWPHYHDF5_STATUS_OBJ_CREATE_FAILURE:     return "LWPHYHDF5_STATUS_OBJ_CREATE_FAILURE";
    case LWPHYHDF5_STATUS_VALUE_OUT_OF_RANGE:     return "LWPHYHDF5_STATUS_VALUE_OUT_OF_RANGE";
    default:                                      return "LWPHYHDF5_UNKNOWN_STATUS";
    }
}
// clang-format on

////////////////////////////////////////////////////////////////////////
// lwphyHDF5GetDatasetInfo()
lwphyHDF5Status_t lwphyHDF5GetDatasetInfo(hid_t            h5Dataset,
                                          int              dimBufferSize,
                                          lwphyDataType_t* dataType,
                                          int*             numDims,
                                          int              outputDimensions[])
{
    //------------------------------------------------------------------
    // Validate inputs
    if(h5Dataset < 0) return LWPHYHDF5_STATUS_ILWALID_DATASET;
    if((dimBufferSize > 0) && (nullptr == outputDimensions))
    {
        return LWPHYHDF5_STATUS_ILWALID_ARGUMENT;
    }
    //------------------------------------------------------------------
    // Retrieve tensor info from the data set
    tensor_descriptor_info tdi = {}; // Zero-initialize
    lwphyHDF5Status_t      s   = get_HDF5_dataset_info(tdi, h5Dataset);
    //------------------------------------------------------------------
    // Populate return info
    if(dataType)
    {
        *dataType = tdi.dataType;
    }
    if(numDims)
    {
        *numDims = tdi.numDims;
    }
    if(outputDimensions)
    {
        for(int i = 0; i < dimBufferSize; ++i)
        {
            outputDimensions[i] = tdi.dimensions[i];
        }
        // If the caller-provided buffer is too small, provide an
        // error (but only if there was no other error).
        if((s == LWPHYHDF5_STATUS_SUCCESS) &&
           (dimBufferSize < tdi.numDims))
        {
            s = LWPHYHDF5_STATUS_INADEQUATE_BUFFER_SIZE;
        }
    }
    //------------------------------------------------------------------
    return s;
}

////////////////////////////////////////////////////////////////////////
// lwphyHDF5ReadDataset()
lwphyHDF5Status_t lwphyHDF5ReadDataset(const lwphyTensorDescriptor_t tensorDesc,
                                       void*                         addr,
                                       hid_t                         h5Dataset,
                                       lwdaStream_t                  strm)
{
    //------------------------------------------------------------------
    // Validate arguments
    if((nullptr == tensorDesc) ||
       (nullptr == addr) ||
       (h5Dataset < 0))
    {
        return LWPHYHDF5_STATUS_ILWALID_ARGUMENT;
    }
    // clang-format off
    // Retrieve properties of the caller-provided destination tensor
    tensor_descriptor_info tdi = {}; // Zero-initialize all members
    lwphyStatus_t          s   = lwphyGetTensorDescriptor(tensorDesc,                              // tensorDesc
                                                          static_cast<int>(tdi.dimensions.size()), // numDimsRequested
                                                          &tdi.dataType,                           // dataType
                                                          &tdi.numDims,                            // numDims
                                                          tdi.dimensions.data(),                   // dimensions[]
                                                          tdi.strides.data());                     // strides[]
    // clang-format on
    if(LWPHY_STATUS_SUCCESS != s)
    {
        return LWPHYHDF5_STATUS_ILWALID_TENSOR_DESC;
    }
    //------------------------------------------------------------------
    // Get tensor info from the HDF5 dataset (located in the file)
    tensor_descriptor_info tdiHDF5File = {}; // Zero-initialize
    lwphyHDF5Status_t      sHDF5       = get_HDF5_dataset_info(tdiHDF5File,
                                                    h5Dataset);
    if(LWPHYHDF5_STATUS_SUCCESS != sHDF5)
    {
        return sHDF5;
    }
    //------------------------------------------------------------------
    // Compare the tensor descriptor dimensions to the HDF5 file
    // dimensions. (The strides don't need to match - the caller may
    // adjust the strides based on application requirements.) The number
    // of dimensions and the individual dimensions must match.
    if((tdi.numDims != tdiHDF5File.numDims) ||
       !std::equal(tdi.dimensions.begin(),
                   tdi.dimensions.begin() + tdi.numDims,
                   tdiHDF5File.dimensions.begin()))
    {
        return LWPHYHDF5_STATUS_TENSOR_MISMATCH;
    }
    //------------------------------------------------------------------
    // Create a tensor descriptor to represent the in-memory data
    // retrieved from the HDF5 file.
    lwphyTensorDescriptor_t memTensorDesc;
    if(LWPHY_STATUS_SUCCESS != lwphyCreateTensorDescriptor(&memTensorDesc))
    {
        return LWPHYHDF5_STATUS_TENSOR_DESC_FAILURE;
    }
    if(LWPHY_STATUS_SUCCESS != lwphySetTensorDescriptor(memTensorDesc,                 // tensor desc
                                                        tdiHDF5File.dataType,          // data type
                                                        tdiHDF5File.numDims,           // rank
                                                        tdiHDF5File.dimensions.data(), // dimensions
                                                        tdiHDF5File.strides.data(),    // strides
                                                        LWPHY_TENSOR_ALIGN_TIGHT))     // flags
    {
        lwphyDestroyTensorDescriptor(memTensorDesc);
        return LWPHYHDF5_STATUS_TENSOR_DESC_FAILURE;
    }
    //------------------------------------------------------------------
    // Allocate a pinned host buffer for the destination of the HDF5
    // read operation.
    size_t hdf5Size = 0;
    if(LWPHY_STATUS_SUCCESS != lwphyGetTensorSizeInBytes(memTensorDesc, &hdf5Size))
    {
        lwphyDestroyTensorDescriptor(memTensorDesc);
        return LWPHYHDF5_STATUS_TENSOR_DESC_FAILURE;
    }
    //printf("size in bytes: %lu\n", hdf5Size);
    void* hostBuffer = nullptr;
    if(lwdaSuccess != lwdaHostAlloc(&hostBuffer, hdf5Size, lwdaHostAllocMapped | lwdaHostAllocWriteCombined))
    {
        lwphyDestroyTensorDescriptor(memTensorDesc);
        return LWPHYHDF5_STATUS_ALLOC_FAILED;
    }
    //------------------------------------------------------------------
    // Determine the in-memory HDF5 datatype for reading, based on the
    // datatype found in the file.
    hid_t hdf5MemType = native_HDF5_type_from_lwphy_type(tdiHDF5File.dataType);
    if(hdf5MemType < 0)
    {
        lwdaFreeHost(hostBuffer);
        lwphyDestroyTensorDescriptor(memTensorDesc);
        return LWPHYHDF5_STATUS_DATATYPE_ERROR;
    }
    //------------------------------------------------------------------
    // Synchronize the LWCA stream to make sure that the input tensor
    // can be read
    lwdaStreamSynchronize(strm);
    //------------------------------------------------------------------
    // Ilwoke the HDF5 library read call to read data into the pinned
    // host buffer
    herr_t h5Status = H5Dread(h5Dataset, hdf5MemType, H5S_ALL, H5S_ALL, H5P_DEFAULT, hostBuffer);
    if(h5Status < 0)
    {
        H5Tclose(hdf5MemType);
        lwdaFreeHost(hostBuffer);
        lwphyDestroyTensorDescriptor(memTensorDesc);
        return LWPHYHDF5_STATUS_READ_ERROR;
    }
    //------------------------------------------------------------------
    // Use the lwphyColwertTensor() function to perform a "copy," where
    // in this case the source is the host buffer with the H5Dread()
    // results.
    sHDF5 = LWPHYHDF5_STATUS_SUCCESS;
    if(LWPHY_STATUS_SUCCESS != lwphyColwertTensor(tensorDesc,    // dst tensor
                                                  addr,          // dst address
                                                  memTensorDesc, // src tensor
                                                  hostBuffer,    // src address
                                                  strm))         // LWCA stream
    {
        sHDF5 = LWPHYHDF5_STATUS_COLWERT_ERROR;
    }
    //------------------------------------------------------------------
    H5Tclose(hdf5MemType);
    lwdaFreeHost(hostBuffer);
    lwphyDestroyTensorDescriptor(memTensorDesc);
    return sHDF5;
}

////////////////////////////////////////////////////////////////////////
// lwphyHDF5WriteDataset()
lwphyHDF5Status_t LWPHYWINAPI lwphyHDF5WriteDataset(hid_t                         h5LocationID,
                                                    const char*                   name,
                                                    const lwphyTensorDescriptor_t srcTensorDesc,
                                                    const void*                   srcAddr,
                                                    lwdaStream_t                  strm)
{
    //------------------------------------------------------------------
    // Validate arguments
    if((nullptr == srcTensorDesc) ||
       (nullptr == srcAddr) ||
       (nullptr == name) ||
       (0 == strlen(name)) ||
       (h5LocationID < 0))
    {
        return LWPHYHDF5_STATUS_ILWALID_ARGUMENT;
    }
    // clang-format off
    // Retrieve properties of the caller-provided source tensor
    tensor_descriptor_info tdiSrc = {}; // Zero-initialize all members
    lwphyStatus_t          s      = lwphyGetTensorDescriptor(srcTensorDesc,                              // tensorDesc
                                                             static_cast<int>(tdiSrc.dimensions.size()), // numDimsRequested
                                                             &tdiSrc.dataType,                           // dataType
                                                             &tdiSrc.numDims,                            // numDims
                                                             tdiSrc.dimensions.data(),                   // dimensions[]
                                                             tdiSrc.strides.data());                     // strides[]
    // clang-format on
    if(LWPHY_STATUS_SUCCESS != s)
    {
        return LWPHYHDF5_STATUS_ILWALID_TENSOR_DESC;
    }
    //------------------------------------------------------------------
    // Check for known implicit colwersions
    lwphyDataType_t storageType = get_storage_type(tdiSrc.dataType);
    if(LWPHY_VOID == storageType)
    {
        return LWPHYHDF5_STATUS_DATATYPE_ERROR;
    }
    //------------------------------------------------------------------
    // Create a tensor descriptor to represent the in-memory data
    // that will be provided to the HDF5 library for storage.
    lwphyTensorDescriptor_t memTensorDesc;
    if(LWPHY_STATUS_SUCCESS != lwphyCreateTensorDescriptor(&memTensorDesc))
    {
        return LWPHYHDF5_STATUS_TENSOR_DESC_FAILURE;
    }
    if(LWPHY_STATUS_SUCCESS != lwphySetTensorDescriptor(memTensorDesc,             // tensor desc
                                                        storageType,               // data type
                                                        tdiSrc.numDims,            // rank
                                                        tdiSrc.dimensions.data(),  // dimensions
                                                        nullptr,                   // strides
                                                        LWPHY_TENSOR_ALIGN_TIGHT)) // flags
    {
        lwphyDestroyTensorDescriptor(memTensorDesc);
        return LWPHYHDF5_STATUS_TENSOR_DESC_FAILURE;
    }
    //------------------------------------------------------------------
    // Allocate a pinned host buffer for the destination of the read
    // from the source tensor
    size_t hdf5Size = 0;
    if(LWPHY_STATUS_SUCCESS != lwphyGetTensorSizeInBytes(memTensorDesc, &hdf5Size))
    {
        lwphyDestroyTensorDescriptor(memTensorDesc);
        return LWPHYHDF5_STATUS_TENSOR_DESC_FAILURE;
    }
    //printf("size in bytes: %lu\n", hdf5Size);
    void* hostBuffer = nullptr;
    if(lwdaSuccess != lwdaHostAlloc(&hostBuffer, hdf5Size, lwdaHostAllocMapped | lwdaHostAllocWriteCombined))
    {
        lwphyDestroyTensorDescriptor(memTensorDesc);
        return LWPHYHDF5_STATUS_ALLOC_FAILED;
    }
    //------------------------------------------------------------------
    // Determine the HDF5 datatype that corresponds to the source tensor
    hid_t hdf5MemType = native_HDF5_type_from_lwphy_type(storageType);
    if(hdf5MemType < 0)
    {
        lwdaFreeHost(hostBuffer);
        lwphyDestroyTensorDescriptor(memTensorDesc);
        return LWPHYHDF5_STATUS_DATATYPE_ERROR;
    }
    //------------------------------------------------------------------
    lwphyHDF5Status_t sHDF5 = LWPHYHDF5_STATUS_SUCCESS;
    if(LWPHY_STATUS_SUCCESS != lwphyColwertTensor(memTensorDesc, // dst tensor
                                                  hostBuffer,    // dst address
                                                  srcTensorDesc, // src tensor
                                                  srcAddr,       // src address
                                                  strm))         // LWCA stream
    {
        sHDF5 = LWPHYHDF5_STATUS_COLWERT_ERROR;
    }
    else
    {
        std::array<hsize_t, LWPHY_DIM_MAX> h5Dims;
        // HDF5 order is slowest-changing first, so reverse indices
        for(size_t i = 0; i < tdiSrc.numDims; ++i)
        {
            h5Dims[tdiSrc.numDims - i - 1] = tdiSrc.dimensions[i];
        }
        hid_t h5Dataspace = H5Screate_simple(tdiSrc.numDims, h5Dims.data(), nullptr);
        if(h5Dataspace < 0)
        {
            sHDF5 = LWPHYHDF5_STATUS_DATASPACE_ERROR;
        }
        else
        {
            hid_t h5Dataset = H5Dcreate2(h5LocationID, // loc_id
                                         name,         // name
                                         hdf5MemType,  // datatype_id
                                         h5Dataspace,  // dataspace_id
                                         H5P_DEFAULT,  // link creation prop list
                                         H5P_DEFAULT,  // dataset creation prop list
                                         H5P_DEFAULT); // dataset access prop list
            if(h5Dataset < 0)
            {
                sHDF5 = LWPHYHDF5_STATUS_DATASET_ERROR;
            }
            else
            {
                // Synchronize on the stream used for colwersion to ensure
                // that the result can be read by the host.
                lwdaStreamSynchronize(strm);
                
                herr_t h5Status = H5Dwrite(h5Dataset,
                                           hdf5MemType,
                                           H5S_ALL,
                                           H5S_ALL,
                                           H5P_DEFAULT,
                                           hostBuffer);
                if(h5Status < 0)
                {
                    sHDF5 = LWPHYHDF5_STATUS_WRITE_ERROR;
                }
                H5Dclose(h5Dataset);
            }
            H5Sclose(h5Dataspace);
        }
    }
    //------------------------------------------------------------------
    H5Tclose(hdf5MemType);
    lwdaFreeHost(hostBuffer);
    lwphyDestroyTensorDescriptor(memTensorDesc);
    return sHDF5;
}

////////////////////////////////////////////////////////////////////////
// lwphyHDF5Struct
// Empty struct for forward-declared type from lwphy_hdf5.h header
struct lwphyHDF5Struct
{
};

////////////////////////////////////////////////////////////////////////
// lwphy_HDF5_struct_element
// Internal class to represent an element of an HDF5 dataset that will
// be accessed as a "struct." This amounts to a dataspace with a single
// element corresponding to a dataset with the HDF5 compound type.
class lwphy_HDF5_struct_element : public lwphyHDF5Struct
{
public:
    ~lwphy_HDF5_struct_element()
    {
        if(H5Iis_valid(h5_dataset_) > 0)
        {
            // Seems like we can either close or dec ref. We will
            // use dec ref to be symmetric with the inc ref upon
            // construction.
            //H5Dclose(h5_dataset_);
            H5Idec_ref(h5_dataset_);
        }
        if(H5Iis_valid(h5_dataspace_) > 0)
        {
            H5Sclose(h5_dataspace_);
        }
    }
    lwphy_HDF5_struct_element(const lwphy_HDF5_struct_element&)            = delete;
    lwphy_HDF5_struct_element& operator=(const lwphy_HDF5_struct_element&) = delete;
    //------------------------------------------------------------------
    // create()
    static lwphyHDF5Status_t create(hid_t                       dset,
                                    size_t                      numDim,
                                    const hsize_t*              coord,
                                    lwphy_HDF5_struct_element** p)
    {
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Initialize output value, assuming failure
        *p = nullptr;
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Check for a valid object identifier
        // (See "proper" way to test for truth in H5public.h)
        if(H5Iis_valid(dset) <= 0)
        {
            return LWPHYHDF5_STATUS_ILWALID_DATASET;
        }
        // Make sure the input object is a dataset
        if(H5I_DATASET != H5Iget_type(dset))
        {
            return LWPHYHDF5_STATUS_INCORRECT_OBJ_TYPE;
        }
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Check for the dataset type
        hid_t dtype = H5Dget_type(dset);
        if(dtype >= 0)
        {
            //-  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
            // Check the datatype class - fail if it isn't a compound type
            H5T_class_t dclass = H5Tget_class(dtype);
            H5Tclose(dtype); // No longer used...
            dtype = -1;
            if(dclass != H5T_COMPOUND)
            {
                return LWPHYHDF5_STATUS_ILWALID_DATASET;
            }
            //-  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
            // Get the dataspace to determine the dimensions
            hid_t dspace = H5Dget_space(dset);
            if(dspace < 0)
            {
                return LWPHYHDF5_STATUS_DATASPACE_ERROR;
            }
            //-  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
            // Check the rank and the number of elements
            int rank = H5Sget_simple_extent_ndims(dspace);
            if((0 == numDim) && (nullptr == coord))
            {
                // Only valid if dataspace has rank 1
                if(rank != 1)
                {
                    H5Sclose(dspace);
                    return LWPHYHDF5_STATUS_DIMENSION_TOO_LARGE;
                }
                // Getting a struct with 0 == numDim and nullptr == coord is
                // only valid if the dataspace has 1 element.
                hsize_t dim0;
                if(H5Sget_simple_extent_dims(dspace, &dim0, nullptr) < 0)
                {
                    H5Sclose(dspace);
                    return LWPHYHDF5_STATUS_DATASPACE_ERROR;
                }
                if(dim0 != 1)
                {
                    H5Sclose(dspace);
                    return LWPHYHDF5_STATUS_DIMENSION_TOO_LARGE;
                }
            }
            else
            {
                // Validate coordinates
                if((rank > LWPHY_DIM_MAX) || (numDim != rank))
                {
                    H5Sclose(dspace);
                    return LWPHYHDF5_STATUS_DIMENSION_TOO_LARGE;
                }
                // Get the dataspace dimensions
                hsize_t dims[LWPHY_DIM_MAX] = {};
                if(H5Sget_simple_extent_dims(dspace, dims, nullptr) < 0)
                {
                    H5Sclose(dspace);
                    return LWPHYHDF5_STATUS_DATASPACE_ERROR;
                }
                // Compare provided indices with the dataset dimensions
                for(int i = 0; i < numDim; ++i)
                {
                    if(coord[i] >= dims[i])
                    {
                        H5Sclose(dspace);
                        return LWPHYHDF5_STATUS_VALUE_OUT_OF_RANGE;
                    }
                }
                // Select the specific element requested in the stored
                // dataspace
                if(H5Sselect_elements(dspace, H5S_SELECT_SET, 1, coord) < 0)
                {
                    H5Sclose(dspace);
                    return LWPHYHDF5_STATUS_DATASPACE_ERROR;
                }
            }
            // Create the struct element instance with the dataset and
            // dataspace
            *p = new (std::nothrow) lwphy_HDF5_struct_element(dset, dspace);
            if(!*p)
            {
                H5Sclose(dspace);
                return LWPHYHDF5_STATUS_ALLOC_FAILED;
            }
            return LWPHYHDF5_STATUS_SUCCESS;
        }
        else
        {
            // H5Dget_type() failure
            return LWPHYHDF5_STATUS_ILWALID_DATASET;
        }
    }
    //------------------------------------------------------------------
    // get_field()
    lwphyHDF5Status_t get_field(lwphyVariant_t& res,
                                const char*     name,
                                lwphyDataType_t valueAs)
    {
        res.type = LWPHY_VOID;
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Retrieve the datatype for the dataset
        hid_t dsetType = H5Dget_type(h5_dataset_);
        if(dsetType < 0)
        {
            return LWPHYHDF5_STATUS_ILWALID_DATASET;
        }
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Locate the member index from the name
        int idx = H5Tget_member_index(dsetType, name);
        if(idx < 0)
        {
            H5Tclose(dsetType);
            return LWPHYHDF5_STATUS_ILWALID_NAME;
        }
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Get the HDF5 stored type of the field
        hid_t srcFieldType = H5Tget_member_type(dsetType, static_cast<unsigned>(idx));
        H5Tclose(dsetType);
        if(srcFieldType < 0)
        {
            return LWPHYHDF5_STATUS_ILWALID_DATASET;
        }
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Get the corresponding lwPHY type
        lwphyDataType_t srcType = lwphy_type_from_HDF5_datatype(srcFieldType);
        if(LWPHY_VOID == srcType)
        {
            H5Tclose(srcFieldType);
            return LWPHYHDF5_STATUS_DATATYPE_ERROR;
        }
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Get the corresponding native type for the field
        hid_t nativeFieldType = H5Tget_native_type(srcFieldType, H5T_DIR_ASCEND);
        H5Tclose(srcFieldType);
        if(nativeFieldType < 0)
        {
            return LWPHYHDF5_STATUS_DATATYPE_ERROR;
        }
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Create an in-memory compound type, with a single field having
        // a name that matches the callers requested field.
        hid_t memCompoundType = H5Tcreate(H5T_COMPOUND, H5Tget_size(nativeFieldType));
        if(memCompoundType < 0)
        {
            H5Tclose(nativeFieldType);
            return LWPHYHDF5_STATUS_OBJ_CREATE_FAILURE;
        }
        if(H5Tinsert(memCompoundType, name, 0, nativeFieldType) < 0)
        {
            H5Tclose(nativeFieldType);
            H5Tclose(memCompoundType);
            return LWPHYHDF5_STATUS_OBJ_CREATE_FAILURE;
        }
        H5Tclose(nativeFieldType);
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Perform an HDF5 read into the variant storage area using
        // the in-memory compound type.
        const hsize_t one      = 1;
        hid_t         memSpace = H5Screate_simple(1, &one, nullptr);
        if(memSpace < 0)
        {
            H5Tclose(memCompoundType);
            return LWPHYHDF5_STATUS_OBJ_CREATE_FAILURE;
        }
        herr_t readStatus = H5Dread(h5_dataset_,     // dataset
                                    memCompoundType, // memory type
                                    memSpace,        // memory dataspace
                                    h5_dataspace_,   // file dataspace
                                    H5P_DEFAULT,     // xfer prop list
                                    &(res.value.r8i));
        H5Sclose(memSpace);
        H5Tclose(memCompoundType);
        if(readStatus < 0)
        {
            return LWPHYHDF5_STATUS_READ_ERROR;
        }
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Store the original type in the user-provided variant
        res.type = srcType;
        if((LWPHY_VOID != valueAs) && (srcType != valueAs))
        {
            return colwert_variant(res, valueAs);
        }
        else
        {
            return LWPHYHDF5_STATUS_SUCCESS;
        }
    }
private:
    lwphy_HDF5_struct_element(hid_t dset, hid_t dspace) :
        h5_dataset_(dset),
        h5_dataspace_(dspace)
    {
        // Increment the reference count of the dataset
        H5Iinc_ref(h5_dataset_);
    }
    // Data
    hid_t h5_dataset_;
    hid_t h5_dataspace_;
};

////////////////////////////////////////////////////////////////////////
// lwphyHDF5GetStruct()
lwphyHDF5Status_t lwphyHDF5GetStruct(hid_t              h5Dataset,
                                     size_t             numDim,
                                     const hsize_t*     coord,
                                     lwphyHDF5Struct_t* s)
{
    //------------------------------------------------------------------
    // Validate arguments
    if (nullptr == s)
    {
        return LWPHYHDF5_STATUS_ILWALID_ARGUMENT;
    }
    if((numDim > 0) && (nullptr == coord))
    {
        return LWPHYHDF5_STATUS_DIMENSION_TOO_LARGE;
    }
    *s = nullptr;
    // (Further validation provided by lwphy_HDF5_struct_element::create())
    lwphy_HDF5_struct_element* ssd    = nullptr;
    lwphyHDF5Status_t          status = lwphy_HDF5_struct_element::create(h5Dataset,
                                                                          numDim,
                                                                          coord,
                                                                          &ssd);
    if(LWPHYHDF5_STATUS_SUCCESS == status)
    {
        *s = static_cast<lwphyHDF5Struct_t>(ssd);
    }
    return status;
}

////////////////////////////////////////////////////////////////////////
// lwphyHDF5GetStructScalar()
lwphyHDF5Status_t lwphyHDF5GetStructScalar(lwphyVariant_t*         res,
                                           const lwphyHDF5Struct_t s,
                                           const char*             name,
                                           lwphyDataType_t         valueAs)
{

    //------------------------------------------------------------------
    // Validate arguments
    if(!s    ||
       !name ||
       !res)
    {
        return LWPHYHDF5_STATUS_ILWALID_ARGUMENT;
    }
    //------------------------------------------------------------------
    // Cast the user handle to the internal datatype and retrieve the
    // field value
    lwphy_HDF5_struct_element& sds = static_cast<lwphy_HDF5_struct_element&>(*s);
    return sds.get_field(*res, name, valueAs);
}

////////////////////////////////////////////////////////////////////////
// lwphyHDF5ReleaseStruct()
lwphyHDF5Status_t lwphyHDF5ReleaseStruct(lwphyHDF5Struct_t s)
{
    //------------------------------------------------------------------
    // Validate arguments
    if(nullptr == s)
    {
        return LWPHYHDF5_STATUS_ILWALID_ARGUMENT;
    }
    //------------------------------------------------------------------
    // Free the structure previously allocated by lwphyHDF5GetStruct()
    lwphy_HDF5_struct_element* sds = static_cast<lwphy_HDF5_struct_element*>(s);
    delete sds;
    
    return LWPHYHDF5_STATUS_SUCCESS;
}
