/*
 * Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 */

#include <stdio.h>
#include <stdlib.h>
#include <memory>
#include "lwphy.h"
#include "lwphy_hdf5.h"
#include "hdf5hpp.hpp"

#define CHECK_LWPHY(stmt)                                                 \
    do {                                                                  \
        lwphyStatus_t s = stmt;                                           \
        if(LWPHY_STATUS_SUCCESS != s)                                     \
        {                                                                 \
            fprintf(stderr, "LWPHY error: %s\n", lwphyGetErrorString(s)); \
            exit(1);                                                      \
        }                                                                 \
    } while(0)

#define CHECK_LWPHYHDF5(stmt)                                                     \
    do {                                                                          \
        lwphyHDF5Status_t s = stmt;                                               \
        if(LWPHYHDF5_STATUS_SUCCESS != s)                                         \
        {                                                                         \
            fprintf(stderr, "LWPHYHDF5 error: %s\n", lwphyHDF5GetErrorString(s)); \
            exit(1);                                                              \
        }                                                                         \
    } while(0)

int main(int argc, char* argv[])
{
    lwphyTensorDescriptor_t tensorDesc;
    int dimensions[3] = {10, 20, 100};
    int strides[3]    = {1,  10, 200};
    CHECK_LWPHY(lwphyCreateTensorDescriptor(&tensorDesc));
    CHECK_LWPHY(lwphySetTensorDescriptor(tensorDesc, LWPHY_C_32F, 3, dimensions, strides, 0));
    CHECK_LWPHY(lwphyDestroyTensorDescriptor(tensorDesc));
#if 0
    int device;
    lwdaGetDevice(&device);
    lwdaDeviceProp prop;
    lwdaGetDeviceProperties(&prop, device);
    printf("texture alignment = %lu, texture pitch alignment = %lu\n", prop.textureAlignment, prop.texturePitchAlignment);
    void*  addr = nullptr;
    size_t pitch = 0;
    for(size_t w = 32; w < 2048; w+=32)
    {
        lwdaMallocPitch(&addr, &pitch, w, 3);
        printf("w = %lu, pitch = %lu\n", w, pitch);
        lwdaFree(addr);
    }
#endif
#if 0
    hid_t h5File = H5Fopen("test.h5",
                           H5F_ACC_RDONLY,
                           H5P_DEFAULT);
    if(h5File < 0)
    {
        fprintf(stderr, "Error opening HDF5 file\n");
    }
    else
    {
        printf("HDF5 file opened successfully\n");
        hid_t h5Dataset = H5Dopen(h5File, "A", H5P_DEFAULT);
        if(h5Dataset < 0)
        {
            fprintf(stderr, "Error opening dataset\n");
        }
        else
        {
            printf("HDF5 dataset opened successfully\n");
            int             dims[LWPHY_DIM_MAX];
            lwphyDataType_t type;
            int             numDims;
            CHECK_LWPHYHDF5(lwphyHDF5GetDatasetInfo(h5Dataset,
                                                    LWPHY_DIM_MAX,
                                                    &type,
                                                    &numDims,
                                                    dims));
            printf("type = %s\n", lwphyGetDataTypeString(type));
            for(int i = 0; i < numDims; ++i)
            {
                printf("dim[%i]: %i\n", i, dims[i]);
            }
            H5Dclose(h5Dataset);
        }
        H5Fclose(h5File);
    }
#endif
#if 0
    hid_t h5File = H5Fopen("test_single.h5",
                           H5F_ACC_RDONLY,
                           H5P_DEFAULT);
    if(h5File < 0)
    {
        fprintf(stderr, "Error opening HDF5 file\n");
    }
    else
    {
        printf("HDF5 file opened successfully\n");
        hid_t h5Dataset = H5Dopen(h5File, "A", H5P_DEFAULT);
        if(h5Dataset < 0)
        {
            fprintf(stderr, "Error opening dataset\n");
        }
        else
        {
            printf("HDF5 dataset opened successfully\n");
            int             dims[LWPHY_DIM_MAX];
            lwphyDataType_t type;
            int             numDims;
            CHECK_LWPHYHDF5(lwphyHDF5GetDatasetInfo(h5Dataset,
                                                    LWPHY_DIM_MAX,
                                                    &type,
                                                    &numDims,
                                                    dims));
            printf("type = %s\n", lwphyGetDataTypeString(type));
            for(int i = 0; i < numDims; ++i)
            {
                printf("dim[%i]: %i\n", i, dims[i]);
            }
            // Allocate a destination tensor
            lwphyTensorDescriptor_t tensorDesc;
            lwphyStatus_t           s = lwphyCreateTensorDescriptor(&tensorDesc); 
            if(LWPHY_STATUS_SUCCESS == s)
            {
                s = lwphySetTensorDescriptor(tensorDesc,
                                             type,
                                             numDims,
                                             dims,
                                             nullptr,
                                             LWPHY_TENSOR_ALIGN_COALESCE/*0*/);
                if(LWPHY_STATUS_SUCCESS == s)
                {
                    size_t numBytes = 0;
                    s = lwphyGetTensorSizeInBytes(tensorDesc, &numBytes);
                    if(LWPHY_STATUS_SUCCESS == s)
                    {
                        void* deviceAddr = nullptr;
                        lwdaError_t sLwda = lwdaMalloc(&deviceAddr, numBytes);
                        if(lwdaSuccess  == sLwda)
                        {
                            lwphyHDF5Status_t sH5 = lwphyHDF5ReadDataset(tensorDesc,
                                                                         deviceAddr,
                                                                         h5Dataset,
                                                                         0);
                            if(LWPHYHDF5_STATUS_SUCCESS == sH5)
                            {
                                printf("Dataset read successfully.\n");
                            }
                            else
                            {
                                fprintf(stderr, "Error reading HDF5 dataset: %s\n", lwphyHDF5GetErrorString(sH5));
                            }
                            lwdaFree(deviceAddr);
                        }
                        else
                        {
                            fprintf(stderr, "Error allocating %lu bytes on device\n", numBytes);
                        }
                    }
                    else
                    {
                        fprintf(stderr, "Error retrieving tensor size: %s\n", lwphyGetErrorString(s));
                    }
                }
                else
                {
                    fprintf(stderr, "Error setting tensor descriptor: %s\n", lwphyGetErrorString(s));
                }
                lwphyDestroyTensorDescriptor(tensorDesc);
            }
            else
            {
                fprintf(stderr, "Error creating tensor descriptor: %s\n", lwphyGetErrorString(s));
            }

            
            H5Dclose(h5Dataset);
        }
        H5Fclose(h5File);
    }    
#endif
#if 0
    try
    {
        hdf5hpp::hdf5_file      f;
        f = hdf5hpp::hdf5_file::open("test_single.h5");
        hdf5hpp::hdf5_dataset   dset   = f.open_dataset("A");
        hdf5hpp::hdf5_dataspace dspace = dset.get_dataspace();
        std::vector<hsize_t>    dims   = dspace.get_dimensions();
        hdf5hpp::hdf5_datatype  dtype  = dset.get_datatype();
        printf("rank(A) = %i\n", dspace.get_rank());
        for(int i = 0; i < dspace.get_rank(); ++i)
        {
            printf("dims[%i] = %llu\n", i, dims[i]);
        }
        printf("class(A) = %s\n", dtype.get_class_string());
        printf("datatype size = %lu bytes\n", dtype.get_size_bytes());
        printf("num_elements(A) = %llu\n", dspace.get_num_elements());
        printf("buffer size = %lu bytes\n", dset.get_buffer_size_bytes());
        std::vector<float> data(dspace.get_num_elements());
        dset.read(data.data());
        for(size_t i = 0; i < data.size(); ++i)
        {
            printf("data[%lu] = %f\n", i, data[i]);
        }
    }
    catch(std::exception& e)
    {
        fprintf(stderr, "EXCEPTION: %s\n", e.what());
    }
#endif
#if 0
    try
    {
        std::unique_ptr<hdf5hpp::hdf5_file> file_ptr;
        file_ptr.reset(new hdf5hpp::hdf5_file(hdf5hpp::hdf5_file::open("test_single.h5")));
        hdf5hpp::hdf5_dataset   dset   = file_ptr->open_dataset("A");
        hdf5hpp::hdf5_dataspace dspace = dset.get_dataspace();
        std::vector<hsize_t>    dims   = dspace.get_dimensions();
        hdf5hpp::hdf5_datatype  dtype  = dset.get_datatype();
        printf("rank(A) = %i\n", dspace.get_rank());
        for(int i = 0; i < dspace.get_rank(); ++i)
        {
            printf("dims[%i] = %llu\n", i, dims[i]);
        }
        printf("class(A) = %s\n", dtype.get_class_string());
        printf("datatype size = %lu bytes\n", dtype.get_size_bytes());
        printf("num_elements(A) = %llu\n", dspace.get_num_elements());
        printf("buffer size = %lu bytes\n", dset.get_buffer_size_bytes());
        std::vector<float> data(dspace.get_num_elements());
        dset.read(data.data());
        for(size_t i = 0; i < data.size(); ++i)
        {
            printf("data[%lu] = %f\n", i, data[i]);
        }
    }
    catch(std::exception& e)
    {
        fprintf(stderr, "EXCEPTION: %s\n", e.what());
    }
#endif
    printf("%s exelwtion completed.\n", argv[0]);
    return 0;
}
