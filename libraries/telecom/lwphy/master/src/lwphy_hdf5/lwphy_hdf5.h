/*
 * Copyright 1993-2020 LWPU Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to LWPU intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to LWPU and is being provided under the terms and
 * conditions of a form of LWPU software license agreement by and
 * between LWPU and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of LWPU is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, LWPU MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * LWPU DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL LWPU BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */

/** \file lwphy_hdf5.h
 *  \brief Physical layer library support for HDF5 file I/O
 *
 *  Header file for the lwPHY HDF5 API
 */

#if !defined(LWPHY_HDF5_H_INCLUDED_)
#define LWPHY_HDF5_H_INCLUDED_

#include "lwphy.h"
#include "hdf5.h"
#include <stdlib.h>

#if defined(__cplusplus)
extern "C" {
#endif /* defined(__cplusplus) */

/**
 * lwphyHDF5 error codes
 */
typedef enum
{
    LWPHYHDF5_STATUS_SUCCESS                = 0,  /*!< The API call returned with no errors.                                    */
    LWPHYHDF5_STATUS_ILWALID_ARGUMENT       = 1,  /*!< One or more of the arguments provided to the function was invalid.       */
    LWPHYHDF5_STATUS_ILWALID_DATASET        = 2,  /*!< The HDF5 dataset argument provided was invalid.                          */
    LWPHYHDF5_STATUS_DATATYPE_ERROR         = 3,  /*!< The HDF5 datatype is not supported by the lwPHY library.                 */
    LWPHYHDF5_STATUS_DATASPACE_ERROR        = 4,  /*!< The HDF5 library returned an error creating or querying the dataspace.   */
    LWPHYHDF5_STATUS_UNSUPPORTED_RANK       = 5,  /*!< The HDF5 dataspace rank is not supported by lwPHY.                       */
    LWPHYHDF5_STATUS_DIMENSION_TOO_LARGE    = 6,  /*!< One or more HDF5 dataspace dimensions are larger than lwPHY supports.    */
    LWPHYHDF5_STATUS_ILWALID_TENSOR_DESC    = 7,  /*!< An invalid tensor descriptor was provided.                               */
    LWPHYHDF5_STATUS_INADEQUATE_BUFFER_SIZE = 8,  /*!< The provided buffer size was inadequate.                                 */
    LWPHYHDF5_STATUS_TENSOR_MISMATCH        = 9,  /*!< Tensor descriptor arguments do not match in rank and/or dimension(s).    */
    LWPHYHDF5_STATUS_UNKNOWN_ERROR          = 10, /*!< Unknown or unexpected internal error.                                    */
    LWPHYHDF5_STATUS_ALLOC_FAILED           = 11, /*!< Memory allocation failed.                                                */
    LWPHYHDF5_STATUS_TENSOR_DESC_FAILURE    = 12, /*!< Creating or setting the lwPHY tensor descriptor failed.                  */
    LWPHYHDF5_STATUS_READ_ERROR             = 13, /*!< An HDF5 read error oclwrred.                                             */
    LWPHYHDF5_STATUS_COLWERT_ERROR          = 14, /*!< A colwersion error oclwrred, or an unsupported colwersion was requested. */
    LWPHYHDF5_STATUS_WRITE_ERROR            = 15, /*!< An HDF5 write error oclwrred.                                            */
    LWPHYHDF5_STATUS_DATASET_ERROR          = 16, /*!< An HDF5 dataset creation/query error oclwrred.                           */
    LWPHYHDF5_STATUS_ILWALID_NAME           = 17, /*!< No such scalar or structure field with the given name exists.            */
    LWPHYHDF5_STATUS_INCORRECT_OBJ_TYPE     = 18, /*!< The HDF5 object provided is not of the correct/expected type.            */
    LWPHYHDF5_STATUS_OBJ_CREATE_FAILURE     = 19, /*!< HDF5 object creation failure.                                            */
    LWPHYHDF5_STATUS_VALUE_OUT_OF_RANGE     = 20  /*!< Data colwersion could not occur because an input value was out of range. */
} lwphyHDF5Status_t;

/******************************************************************/ /**
 * \brief Returns the description string for an error code
 *
 * Returns the description string for an error code.  If the error
 * code is not recognized, "Unknown status code" is returned.
 *
 * \param status - Status code for desired string
 *
 * \return
 * \p char* pointer to a NULL-terminated string
 *
 * \sa ::lwphyHDF5Status_t, ::lwphyHDF5GetErrorName
 */
const char* LWPHYWINAPI lwphyHDF5GetErrorString(lwphyHDF5Status_t status);

/******************************************************************/ /**
 * \brief Returns a string version of an error code enumeration value
 *
 * Returns a string version of an error code.  If the error
 * code is not recognized, "LWPHYHDF5_UNKNOWN_STATUS" is returned.
 *
 * \param status - Status code for desired string
 *
 * \return
 * \p char* pointer to a NULL-terminated string
 *
 * \sa ::lwphyHDF5GetErrorString, ::lwphyStatus_t
 */
const char* LWPHYWINAPI lwphyHDF5GetErrorName(lwphyHDF5Status_t status);

/******************************************************************/ /**
 * \brief Returns information about an HDF5 dataset 
 *
 * Determines the rank, dimensions, and datatype of an HDF5 dataset.
 * The stored type of the dataset is mapped to the tensor element types
 * supported by lwPHY.
 *
 * \param h5Dataset - HDF5 dataset
 * \param dimBufferSize - size of the array for the \p outputDimensions argument
 * \param dataType - address for returned lwPHY datatype (may be NULL)
 * \param numDims - address for returned rank (may be NULL)
 * \param dimensions - array for storage of dataset dimensions (may be NULL
 * if \p dimBufferSize is zero)
 *
 * Returns ::LWPHYHDF5_STATUS_ILWALID_DATASET if \p h5Dataset < 0.
 *
 * Returns ::LWPHYHDF5_STATUS_ILWALID_ARGUMENT if \p dimBufferSize > 0
 * and \p outputDimensions is NULL.
 * 
 * Returns ::LWPHYHDF5_STATUS_INADEQUATE_BUFFER_SIZE if \p dimBufferSize is less
 * than the HDF5 dataspace rank
 * 
 * Returns ::LWPHYHDF5_STATUS_DATATYPE_ERROR if the data type of the HDF5 dataset
 * cannot be represented by a lwPHY tensor element type
 *
 * Returns ::LWPHYHDF5_STATUS_DATASPACE_ERROR if the HDF5 dataspace cannot be
 * queried
 *
 * Returns ::LWPHYHDF5_STATUS_UNSUPPORTED_RANK if the HDF5 dataspace is larger
 * than the maximum rank supported by the lwPHY library
 * 
 * Returns ::LWPHYHDF5_STATUS_DIMENSION_TOO_LARGE if one or more of the HDF5
 * dataspace dimensions is larger than the maximum size supported by the lwPHY library
 * 
 * Returns ::LWPHYHDF5_STATUS_SUCCESS if the query was successful.
 *
 * \return
 * ::LWPHYHDF5_STATUS_SUCCESS,
 * ::LWPHYHDF5_STATUS_ILWALID_DATASET
 * ::LWPHYHDF5_STATUS_ILWALID_ARGUMENT
 * ::LWPHYHDF5_STATUS_DIMENSION_TOO_LARGE
 * ::LWPHYHDF5_STATUS_UNSUPPORTED_RANK
 * ::LWPHYHDF5_STATUS_DATASPACE_ERROR
 * ::LWPHYHDF5_STATUS_DATATYPE_ERROR
 * ::LWPHYHDF5_STATUS_INADEQUATE_BUFFER_SIZE
 *
 * \sa ::lwphyHDF5Status_t,::lwphyHDF5GetErrorString
 */
lwphyHDF5Status_t LWPHYWINAPI lwphyHDF5GetDatasetInfo(hid_t            h5Dataset,
                                                      int              dimBufferSize,
                                                      lwphyDataType_t* dataType,
                                                      int*             numDims,
                                                      int              dimensions[]);

/******************************************************************/ /**
 * \brief Reads data from an HDF5 dataset into a lwPHY tensor
 *
 * Reads data from a source HDF5 dataset into a destination lwPHY tensor.
 * The lwPHY tensor descriptor and an appropriately sized buffer must
 * be allocated before this function is called. The buffer must be
 * GPU addressable (i.e. either in device or pinned host memory). The
 * ::lwphyHDF5GetDatasetInfo function can be called to obtain information
 * on the HDF5 dataset.
 *
 * Note that the destination tensor does not have to be the same as
 * the lwPHY type that maps directly to the stored HDF5 datatype. Any
 * colwersion that is supported by the lwphyColwertTensor function will
 * be supported by this function.
 *
 * \param tensorDesc - descriptor for the destination tensor
 * \param addr - destination address for the loaded data
 * \param h5Dataset - source dataset
 * \param strm - LWCA stream used to perform the read operation
 *
 * Returns ::LWPHYHDF5_STATUS_ILWALID_DATASET if \p h5Dataset < 0.
 * 
 * Returns ::LWPHYHDF5_STATUS_ILWALID_TENSOR_DESC if \p tensorDesc is invalid.
 *
 * Returns ::LWPHYHDF5_STATUS_ILWALID_ARGUMENT if \p dimBufferSize > 0
 * and \p outputDimensions is NULL.
 * 
 * Returns ::LWPHYHDF5_STATUS_INADEQUATE_BUFFER_SIZE if \p dimBufferSize is less
 * than the HDF5 dataspace rank
 * 
 * Returns ::LWPHYHDF5_STATUS_DATATYPE_ERROR if the data type of the HDF5 dataset
 * cannot be represented by a lwPHY tensor element type
 *
 * Returns ::LWPHYHDF5_STATUS_DATASPACE_ERROR if the HDF5 dataspace cannot be
 * queried
 *
 * Returns ::LWPHYHDF5_STATUS_UNSUPPORTED_RANK if the HDF5 dataspace is larger
 * than the maximum rank supported by the lwPHY library
 * 
 * Returns ::LWPHYHDF5_STATUS_DIMENSION_TOO_LARGE if one or more of the HDF5
 * dataspace dimensions is larger than the maximum size supported by the lwPHY library
 *
 * Returns ::LWPHYHDF5_STATUS_TENSOR_MISMATCH if the rank and/or dimensions of
 * the given lwPHY tensor descriptor do not match those of the HDF5 dataset.
 *
 * Returns ::LWPHYHDF5_STATUS_TENSOR_DESC_FAILURE if a lwPHY tensor descriptor
 * could not be created, or if the tensor descriptor fields could not be set.
 *
 * Returns ::LWPHYHDF5_STATUS_READ_ERROR if the HDF5 library returned an error
 * when a read operation was attempted.
 *
 * Returns ::LWPHYHDF5_STATUS_COLWERT_ERROR if colwersion from the HDF5
 * type to the given tensor type is not supported.
 * 
 * Returns ::LWPHYHDF5_STATUS_SUCCESS if the read was successful.
 *
 * \return
 * ::LWPHYHDF5_STATUS_SUCCESS,
 * ::LWPHYHDF5_STATUS_ILWALID_ARGUMENT
 * ::LWPHYHDF5_STATUS_ILWALID_TENSOR_DESC
 * ::LWPHYHDF5_STATUS_ILWALID_DATASET
 * ::LWPHYHDF5_STATUS_DIMENSION_TOO_LARGE
 * ::LWPHYHDF5_STATUS_UNSUPPORTED_RANK
 * ::LWPHYHDF5_STATUS_DATASPACE_ERROR
 * ::LWPHYHDF5_STATUS_DATATYPE_ERROR
 * ::LWPHYHDF5_STATUS_INADEQUATE_BUFFER_SIZE
 * ::LWPHYHDF5_STATUS_TENSOR_MISMATCH
 * ::LWPHYHDF5_STATUS_TENSOR_DESC_FAILURE
 * ::LWPHYHDF5_STATUS_READ_ERROR
 * ::LWPHYHDF5_STATUS_COLWERT_ERROR
 *
 * \sa ::lwphyHDF5Status_t,::lwphyHDF5GetErrorString,::lwphyHDF5GetDatasetInfo
 */
lwphyHDF5Status_t LWPHYWINAPI lwphyHDF5ReadDataset(const lwphyTensorDescriptor_t tensorDesc,
                                                   void*                         addr,
                                                   hid_t                         h5Dataset,
                                                   lwdaStream_t                  strm);

/******************************************************************/ /**
 * \brief Writes data from a lwPHY tensor to an HDF5 dataset
 *
 * Writes data from a source lwPHY tensor to a destination HDF5 dataset.
 *
 * \param h5LocationID - output HDF5 location (HDF5 root/group)
 * \param name - name of output HDF5 dataset
 * \param tensorDesc - input tensor descriptor
 * \param addr - address for input data source
 * \param strm - LWCA stream used to perform the write operation
 *
 * 
 * Returns ::LWPHYHDF5_STATUS_ILWALID_ARGUMENT if an of \p tensorDesc, \p addr, 
 * or \p name is NULL, or if \p h5LocationID < 0.
 * 
 * Returns ::LWPHYHDF5_STATUS_ILWALID_TENSOR_DESC if the tensor descriptor
 * cannot be queried.
 *
 * Returns ::LWPHYHDF5_STATUS_DATATYPE_ERROR if the data type of the lwPHY
 * tensor cannot be stored in an HDF5 dataset (either directly, or via
 * colwersion)
 *
 * Returns ::LWPHYHDF5_STATUS_TENSOR_DESC_FAILURE if a lwPHY tensor descriptor
 * could not be created, or if the tensor descriptor fields could not be set.
 *
 * Returns ::LWPHYHDF5_STATUS_ALLOC_FAILED if memory allocation for a host
 * copy of the input tensor failed.
 *
 * Returns ::LWPHYHDF5_STATUS_COLWERT_ERROR if colwersion from the lwPHY
 * tensor type to the HDF5 datatype failed.
 *
 * Returns ::LWPHYHDF5_STATUS_DATASPACE_ERROR if an HDF5 dataspace could not be
 * created.
 *
 * Returns ::LWPHYHDF5_STATUS_DATASET_ERROR if an HDF5 dataset could not be
 * created.
 *
 * Returns ::LWPHYHDF5_STATUS_WRITE_ERROR if the HDF5 library returned an error
 * when a write operation was attempted.
 *
 * Returns ::LWPHYHDF5_STATUS_SUCCESS if the write was successful.
 *
 * \return
 * ::LWPHYHDF5_STATUS_SUCCESS,
 * ::LWPHYHDF5_STATUS_ILWALID_ARGUMENT
 * ::LWPHYHDF5_STATUS_ILWALID_TENSOR_DESC
 * ::LWPHYHDF5_STATUS_DATATYPE_ERROR
 * ::LWPHYHDF5_STATUS_DATASPACE_ERROR
 * ::LWPHYHDF5_STATUS_DATASET_ERROR
 * ::LWPHYHDF5_STATUS_TENSOR_DESC_FAILURE
 * ::LWPHYHDF5_STATUS_ALLOC_FAILED
 * ::LWPHYHDF5_STATUS_WRITE_ERROR
 * ::LWPHYHDF5_STATUS_COLWERT_ERROR
 *
 * \sa ::lwphyHDF5Status_t,::lwphyHDF5GetErrorString
 */
lwphyHDF5Status_t LWPHYWINAPI lwphyHDF5WriteDataset(hid_t                         h5LocationID,
                                                    const char*                   name,
                                                    const lwphyTensorDescriptor_t tensorDesc,
                                                    const void*                   addr,
                                                    lwdaStream_t                  strm);

struct lwphyHDF5Struct;
typedef struct lwphyHDF5Struct* lwphyHDF5Struct_t;

/******************************************************************/ /**
 * \brief Acquires a lwPHY HDF5 structure handle
 *
 * Creates a lwPHY HDF5 structure handle that references an instance
 * of an HDF5 compound data type in the given HDF5 dataset.
 * The \p numDim and \p coord arguments can be used to identify a
 * specific instance when the dataset argument refers to an array
 * of structures. For datasets with a single struct element, NULL
 * can be specified for the \p coord argument (and the \p numDim
 * value should be zero) to access the lone element. Otherwise,
 * \p numDim should match the rank of the dataset, and one coordinate
 * value for each dimension should be provided in the \p coord array.
 * The (HDF5 library) reference count of the dataset is
 * incremented so that the HDF5 dataset can be accessed during the
 * lifetime of the lwphyHDF5Struct_t instance. The reference count
 * will be decremented (and the object potentially destroyed) when
 * the ::lwphyHDF5ReleaseStruct function is called.
 *
 * \param h5Dataset - HDF5 dataset
 * \param numDim - number of elements in the coord array (or zero)
 * \param coord - array of indices for element to access (or NULL)
 * \param s - address for returned lwphy HDF5 structure handle
 * 
 * Returns ::LWPHYHDF5_STATUS_ILWALID_ARGUMENT if \p h5Dataset < 0, or
 * if s is NULL
 *
 * Returns ::LWPHYHDF5_STATUS_ILWALID_DATASET if the type of the HDF5
 * dataset is not the HDF5 compound datatype
 *
 * Returns ::LWPHYHDF5_STATUS_DATASPACE_ERROR if an error oclwrs
 * querying the dataspace of the given dataset
 *
 * Returns ::LWPHYHDF5_STATUS_DIMENSION_TOO_LARGE if the dataset rank is
 * greater than 1 and \p numDim is zero (or \p coord is NULL), or if
 * \p numDim is greater than 0 and \p coord is NULL
 *
 * Returns ::LWPHYHDF5_STATUS_ALLOC_FAILED if memory for the internal
 * data structure could not be allocated
 *
 * Returns ::LWPHYHDF5_STATUS_INCORRECT_OBJ_TYPE if the given HDF5 id \p h5Dataset
 * does not refer to a dataset
 * 
 * Returns ::LWPHYHDF5_STATUS_SUCCESS if the handle acquisition was successful
 *
 * \return
 * ::LWPHYHDF5_STATUS_SUCCESS,
 * ::LWPHYHDF5_STATUS_ILWALID_ARGUMENT
 * ::LWPHYHDF5_STATUS_ILWALID_DATASET
 * ::LWPHYHDF5_STATUS_DATASPACE_ERROR
 * ::LWPHYHDF5_STATUS_DIMENSION_TOO_LARGE
 * ::LWPHYHDF5_STATUS_ALLOC_FAILED;
 *
 * \sa ::lwphyHDF5Status_t,::lwphyHDF5GetErrorString,::lwphyHDF5ReleaseStruct
 */
lwphyHDF5Status_t LWPHYWINAPI lwphyHDF5GetStruct(hid_t              h5Dataset,
                                                 size_t             numDim,
                                                 const hsize_t*     coord,
                                                 lwphyHDF5Struct_t* s);

/******************************************************************/ /**
 * \brief Retrieves a scalar value from a lwphy HDF5 struct
 *
 * Retrieves a scalar value from a HDF5 compound dataset (structure),
 * optionally colwerting the value to a caller provided type.
 *
 * \param res - output value (variant)
 * \param s - lwphy HDF5 structure handle
 * \param name - name of field to retrieve
 * \param valueAs - desired type of result (or LWPHY_VOID to use the stored type)
 * 
 * Returns ::LWPHYHDF5_STATUS_ILWALID_ARGUMENT if an of \p res, \p s, 
 * or \p name is NULL
 * 
 * Returns ::LWPHYHDF5_STATUS_DIMENSION_TOO_LARGE if the field contains
 * more than a single element
 *
 * Returns ::LWPHYHDF5_STATUS_ILWALID_NAME if the structure does not contain
 * a field with the given name
 *
 * Returns ::LWPHYHDF5_STATUS_COLWERT_ERROR if colwersion to the requested lwPHY
 * element type is not possible, or if colwersion failed (due to overflow/underflow),
 * or if the file datatype cannot be represented by one of the available variant
 * types.
 *
 * Returns ::LWPHYHDF5_STATUS_SUCCESS if the query was successful.
 *
 * \return
 * ::LWPHYHDF5_STATUS_SUCCESS,
 * ::LWPHYHDF5_STATUS_ILWALID_ARGUMENT
 * ::LWPHYHDF5_STATUS_DIMENSION_TOO_LARGE
 * ::LWPHYHDF5_STATUS_COLWERT_ERROR
 * ::LWPHYHDF5_STATUS_ILWALID_NAME
 *
 * \sa ::lwphyHDF5Status_t,::lwphyHDF5GetErrorString,::lwphyHDF5GetStruct
 */
lwphyHDF5Status_t LWPHYWINAPI lwphyHDF5GetStructScalar(lwphyVariant_t*         res,
                                                       const lwphyHDF5Struct_t s,
                                                       const char*             name,
                                                       lwphyDataType_t         valueAs);

/******************************************************************/ /**
 * \brief Releases a lwPHY HDF5 structure handle
 *
 * Releases a reference to a lwPHY HDF5 structure handle
 *
 * \param s - lwphy HDF5 structure handle
 * 
 * Returns ::LWPHYHDF5_STATUS_ILWALID_ARGUMENT if \p s is NULL
 * 
 * Returns ::LWPHYHDF5_STATUS_SUCCESS if the release was successful.
 *
 * \return
 * ::LWPHYHDF5_STATUS_SUCCESS,
 * ::LWPHYHDF5_STATUS_ILWALID_ARGUMENT
 *
 * \sa ::lwphyHDF5Status_t,::lwphyHDF5GetErrorString,::lwphyHDF5GetStruct
 */
lwphyHDF5Status_t LWPHYWINAPI lwphyHDF5ReleaseStruct(lwphyHDF5Struct_t s);

#if defined(__cplusplus)
} /* extern "C" */
#endif /* defined(__cplusplus) */

#endif /* !defined(LWPHY_HDF5_H_INCLUDED_) */
