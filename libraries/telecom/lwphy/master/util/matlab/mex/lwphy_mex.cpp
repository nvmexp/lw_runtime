/*
 * Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 */

#include "mex.h"
#include <vector>
#include <string>
#include <string.h>
#include <map>
#include <initializer_list>
#include "lwphy.h"

namespace
{
    

////////////////////////////////////////////////////////////////////////
// get_lwphy_type()
lwphyDataType_t get_lwphy_type(mxClassID classID, bool isComplex)
{
    switch(classID)
    {
    case mxUNKNOWN_CLASS:
    case mxCELL_CLASS:
    case mxSTRUCT_CLASS:
    case mxLOGICAL_CLASS:
    case mxVOID_CLASS:
    case mxCHAR_CLASS:
    case mxFUNCTION_CLASS:
    case mxINT64_CLASS:
    case mxUINT64_CLASS:
    default:
        return LWPHY_VOID;
    case mxDOUBLE_CLASS: return (isComplex ? LWPHY_C_64F : LWPHY_R_64F);
    case mxSINGLE_CLASS: return (isComplex ? LWPHY_C_32F : LWPHY_R_32F);
    case mxINT8_CLASS:   return (isComplex ? LWPHY_C_8I  : LWPHY_R_8I);
    case mxUINT8_CLASS:  return (isComplex ? LWPHY_C_8U  : LWPHY_R_8U);
    case mxINT16_CLASS:  return (isComplex ? LWPHY_C_16I : LWPHY_R_16I);
    case mxUINT16_CLASS: return (isComplex ? LWPHY_C_16U : LWPHY_R_16U);
    case mxINT32_CLASS:  return (isComplex ? LWPHY_C_32I : LWPHY_R_32I);
    case mxUINT32_CLASS: return (isComplex ? LWPHY_C_32U : LWPHY_R_32U);
    }
}

////////////////////////////////////////////////////////////////////////
// get_device_tensor_descriptor()
// Creates a lwPHY tensor descriptor to represent the data in either
// device or host memory, by copying the properties of the existing
// tensor descriptor. A stride flag can be provided to make the created
// descriptor have a different layout.
lwphyTensorDescriptor_t copy_tensor_descriptor(const lwphyTensorDescriptor_t srcDesc,
                                               unsigned int                  flags)
{
    lwphyDataType_t  dataType = LWPHY_VOID;
    std::vector<int> dims(LWPHY_DIM_MAX);
    int              numDims;
    lwphyGetTensorDescriptor(srcDesc,      // source descriptor
                             dims.size(),  // output dimension size
                             &dataType,    // data type (output)
                             &numDims,     // number of dimensions (output)
                             dims.data(),  // dimensions (output)
                             nullptr);     // strides
    lwphyTensorDescriptor_t newDesc = nullptr;
    lwphyCreateTensorDescriptor(&newDesc);
    if(newDesc)
    {
        if(LWPHY_STATUS_SUCCESS != lwphySetTensorDescriptor(newDesc,     // descriptor
                                                            dataType,    // data type
                                                            numDims,     // number of dimensions
                                                            dims.data(), // dimensions,
                                                            nullptr,     // strides
                                                            flags))
        {
            lwphyDestroyTensorDescriptor(newDesc);
            newDesc = nullptr;
        }
    }
    return newDesc;
}

////////////////////////////////////////////////////////////////////////
// get_mx_tensor_descriptor()
// Creates a lwPHY tensor descriptor to represent the data in a mxArray
// (on the host, owned by MATLAB).
lwphyTensorDescriptor_t get_mx_tensor_descriptor(const mxArray* mxA)
{
    lwphyTensorDescriptor_t tensorDesc = nullptr;
    bool                    isComplex = mxIsComplex(mxA);
    if(LWPHY_STATUS_SUCCESS != lwphyCreateTensorDescriptor(&tensorDesc))
    {
        return tensorDesc;
    }
    lwphyDataType_t  dataType = get_lwphy_type(mxGetClassID(mxA), isComplex);
    std::vector<int> dim(mxGetNumberOfDimensions(mxA));
    const mwSize*    mxDim = mxGetDimensions(mxA);
    for(size_t i = 0; i < dim.size(); ++i)
    {
        dim[i] = static_cast<int>(mxDim[i]);
    }
    if(LWPHY_STATUS_SUCCESS != lwphySetTensorDescriptor(tensorDesc,
                                                        dataType,
                                                        static_cast<int>(mxGetNumberOfDimensions(mxA)),
                                                        dim.data(),
                                                        nullptr,
                                                        0))
    {
        fprintf(stderr, "lwphySetTensorDescriptor() failure\n");
        lwphyDestroyTensorDescriptor(tensorDesc);
        return nullptr;
    }
    return tensorDesc;
}

////////////////////////////////////////////////////////////////////////
// allocate_device_tensor()
std::pair<lwphyTensorDescriptor_t, void*> allocate_device_tensor(std::initializer_list<int> ilist,
                                                                 lwphyDataType_t            dataType,
                                                                 unsigned int               alignmentFlags)
{
    std::vector<int>        dims(ilist.begin(), ilist.end());
    lwphyTensorDescriptor_t newDesc = nullptr;
    void*                   pv = nullptr;
    lwphyCreateTensorDescriptor(&newDesc);
    if(newDesc)
    {
        lwphyStatus_t sPHY = lwphySetTensorDescriptor(newDesc,         // descriptor
                                                      dataType,        // data type
                                                      dims.size(),     // number of dimensions
                                                      dims.data(),     // dimensions,
                                                      nullptr,         // strides
                                                      alignmentFlags); // alignment flags
        if(LWPHY_STATUS_SUCCESS != sPHY)
        {
            fprintf(stderr, "lwphySetTensorDescriptor() failure (%s)\n", lwphyGetErrorString(sPHY));
            lwphyDestroyTensorDescriptor(newDesc);
            newDesc = nullptr;
        }
        else
        {
            size_t sz = 0;
            lwphyGetTensorSizeInBytes(newDesc, &sz);
            lwdaError_t sLWDA = lwdaMalloc(&pv, sz);
            if(lwdaSuccess != sLWDA)
            {
                fprintf(stderr, "lwdaMalloc() error (%lu bytes) (%s)\n", sz, lwdaGetErrorString(sLWDA));
            }
        }
    }
    return std::pair<lwphyTensorDescriptor_t, void*>(newDesc, pv);

}

////////////////////////////////////////////////////////////////////////
// initialize_device_tensor()
// Allocates device memory and initializes it with the contents with
// data from the mxArray. Returns the allocated address (which should
// be freed with lwdaFree()) upon success, and nullptr on failure.
// Since we are supporting only the traditional split complex MATLAB
// format for now, this function will return nullptr for complex
// mxArrays.
void* initialize_device_tensor(lwphyTensorDescriptor_t tensorDesc,
                               lwphyTensorDescriptor_t mxtensorDesc,
                               const mxArray*          srcmx)
{
    void*  pvdevice = nullptr;
    void*  pvhost   = nullptr;
    size_t sz       = 0;
    if(mxIsComplex(srcmx))
    {
        return pvdevice;
    }
    if(LWPHY_STATUS_SUCCESS != lwphyGetTensorSizeInBytes(tensorDesc, &sz))
    {
        return pvdevice;
    }
    lwdaError_t sLwda = lwdaHostAlloc(&pvhost,
                                      mxGetNumberOfElements(srcmx) * mxGetElementSize(srcmx),
                                      lwdaHostAllocWriteCombined);
    if(lwdaSuccess != sLwda)
    {
        fprintf(stderr, "lwdaHostAlloc() failure (%s)\n", lwdaGetErrorString(sLwda));
        return pvdevice;
    }
    memcpy(pvhost, mxGetData(srcmx), mxGetNumberOfElements(srcmx) * mxGetElementSize(srcmx));
    if(lwdaSuccess != lwdaMalloc(&pvdevice, sz))
    {
        fprintf(stderr, "lwdaMalloc() failure (%lu bytes)\n", sz);
        lwdaFreeHost(pvhost);
        return pvdevice;
    }
    lwphyStatus_t slwPHY = lwphyColwertTensor(tensorDesc,    // dst tensor desc
                                              pvdevice,      // dst data
                                              mxtensorDesc,  // src tensor desc
                                              pvhost,        // src data address
                                              0);            // stream
    if(LWPHY_STATUS_SUCCESS != slwPHY)
    {
        fprintf(stderr, "lwphyColwertTensor() failure (%s)\n", lwphyGetErrorString(slwPHY));
        lwdaFree(pvdevice);
        pvdevice = nullptr;
    }
    lwdaFreeHost(pvhost);
    return pvdevice;
}

template <typename T>
void memcpy_split_to_interleaved(T* dst, const T* srcReal, const T* srcImag, size_t N)
{
    for(size_t i = 0; i < N; ++i)
    {
        dst[(i * 2) + 0] = srcReal[i];
        dst[(i * 2) + 1] = srcImag[i];
    }
}
    
template <typename T>
void memcpy_interleaved_to_split(T* dstReal, T* dstImag, const T* src, size_t N)
{
    for(size_t i = 0; i < N; ++i)
    {
        dstReal[i] = src[(i * 2) + 0];
        dstImag[i] = src[(i * 2) + 1];
    }
}
    
////////////////////////////////////////////////////////////////////////
// initialize_complex_device_tensor()
// Allocates device memory and initializes it with the contents with
// data from the mxArray. Returns the allocated address (which should
// be freed with lwdaFree()) upon success, and nullptr on failure.
// Since we are supporting only the traditional split complex MATLAB
// format for now, this function will return nullptr for complex
// mxArrays.
void* initialize_complex_device_tensor(lwphyTensorDescriptor_t tensorDesc,
                                       lwphyTensorDescriptor_t mxtensorDesc,
                                       const mxArray*          srcmx)
{
    void*  pvdevice = nullptr;
    void*  pvhost   = nullptr;
    size_t sz       = 0;
    if(LWPHY_STATUS_SUCCESS != lwphyGetTensorSizeInBytes(tensorDesc, &sz))
    {
        return pvdevice;
    }
    // Multiply element size by 2 for complex data
    lwdaError_t sLwda = lwdaHostAlloc(&pvhost,
                                      mxGetNumberOfElements(srcmx) * mxGetElementSize(srcmx) * 2,
                                      lwdaHostAllocWriteCombined);
    if(lwdaSuccess != sLwda)
    {
        fprintf(stderr, "lwdaHostAlloc() failure (%s)\n", lwdaGetErrorString(sLwda));
        return pvdevice;
    }
    void* pr = mxGetPr(srcmx);
    void* pi = mxGetPi(srcmx);
    switch(mxGetClassID(srcmx))
    {
    case mxUNKNOWN_CLASS:
    case mxCELL_CLASS:
    case mxSTRUCT_CLASS:
    case mxLOGICAL_CLASS:
    case mxVOID_CLASS:
    case mxCHAR_CLASS:
    case mxFUNCTION_CLASS:
    case mxINT64_CLASS:
    case mxUINT64_CLASS:
    default:
        break;
    case mxDOUBLE_CLASS:
        memcpy_split_to_interleaved(static_cast<double*>(pvhost), static_cast<double*>(pr), static_cast<double*>(pi), mxGetNumberOfElements(srcmx));
        break;
    case mxSINGLE_CLASS:
        memcpy_split_to_interleaved(static_cast<float*>(pvhost), static_cast<float*>(pr), static_cast<float*>(pi), mxGetNumberOfElements(srcmx));
        break;
    case mxINT8_CLASS:
        memcpy_split_to_interleaved(static_cast<int8_t*>(pvhost), static_cast<int8_t*>(pr), static_cast<int8_t*>(pi), mxGetNumberOfElements(srcmx));
        break;
    case mxUINT8_CLASS:
        memcpy_split_to_interleaved(static_cast<uint8_t*>(pvhost), static_cast<uint8_t*>(pr), static_cast<uint8_t*>(pi), mxGetNumberOfElements(srcmx));
        break;
    case mxINT16_CLASS:
        memcpy_split_to_interleaved(static_cast<int16_t*>(pvhost), static_cast<int16_t*>(pr), static_cast<int16_t*>(pi), mxGetNumberOfElements(srcmx));
        break;
    case mxUINT16_CLASS:
        memcpy_split_to_interleaved(static_cast<uint16_t*>(pvhost), static_cast<uint16_t*>(pr), static_cast<uint16_t*>(pi), mxGetNumberOfElements(srcmx));
        break;
    case mxINT32_CLASS:
        memcpy_split_to_interleaved(static_cast<int32_t*>(pvhost), static_cast<int32_t*>(pr), static_cast<int32_t*>(pi), mxGetNumberOfElements(srcmx));
        break;
    case mxUINT32_CLASS:
        memcpy_split_to_interleaved(static_cast<uint32_t*>(pvhost), static_cast<uint32_t*>(pr), static_cast<uint32_t*>(pi), mxGetNumberOfElements(srcmx));
        break;
    }
    if(lwdaSuccess != lwdaMalloc(&pvdevice, sz))
    {
        fprintf(stderr, "lwdaMalloc() failure (%lu bytes)\n", sz);
        lwdaFreeHost(pvhost);
        return pvdevice;
    }
    lwphyStatus_t slwPHY = lwphyColwertTensor(tensorDesc,    // dst tensor desc
                                              pvdevice,      // dst data
                                              mxtensorDesc,  // src tensor desc
                                              pvhost,        // src data address
                                              0);            // stream
    if(LWPHY_STATUS_SUCCESS != slwPHY)
    {
        fprintf(stderr, "lwphyColwertTensor() failure (%s)\n", lwphyGetErrorString(slwPHY));
        lwdaFree(pvdevice);
        pvdevice = nullptr;
    }
    lwdaFreeHost(pvhost);
    return pvdevice;
}

////////////////////////////////////////////////////////////////////////
// allocate_device_tensor()
std::pair<lwphyTensorDescriptor_t, void*> allocate_device_tensor(const mxArray* mxA,
                                                                 unsigned int   alignmentFlags)
{
    lwphyTensorDescriptor_t newDesc = nullptr;
    void*                   pv      = nullptr;
    //------------------------------------------------------------------
    // Create a tensor descriptor from the mxArray
    lwphyTensorDescriptor_t tmx     = get_mx_tensor_descriptor(mxA);
    if(tmx)
    {
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Make a copy, adding any provide alignment flags
        newDesc = copy_tensor_descriptor(tmx, alignmentFlags);
        if(newDesc)
        {
            pv = initialize_device_tensor(newDesc, tmx, mxA);
            if(!pv)
            {
                lwphyDestroyTensorDescriptor(newDesc);
                newDesc = nullptr;
            }
        }
        lwphyDestroyTensorDescriptor(tmx);
        tmx = nullptr;
    }
    return std::pair<lwphyTensorDescriptor_t, void*>(newDesc, pv);

}

////////////////////////////////////////////////////////////////////////
// allocate_complex_device_tensor()
std::pair<lwphyTensorDescriptor_t, void*> allocate_complex_device_tensor(const mxArray* mxA,
                                                                         unsigned int   alignmentFlags)
{
    lwphyTensorDescriptor_t newDesc = nullptr;
    void*                   pv      = nullptr;
    //------------------------------------------------------------------
    // Create a tensor descriptor from the mxArray
    lwphyTensorDescriptor_t tmx     = get_mx_tensor_descriptor(mxA);
    if(tmx)
    {
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Make a copy, adding any provide alignment flags
        newDesc = copy_tensor_descriptor(tmx, alignmentFlags);
        if(newDesc)
        {
            pv = initialize_complex_device_tensor(newDesc, tmx, mxA);
            if(!pv)
            {
                lwphyDestroyTensorDescriptor(newDesc);
                newDesc = nullptr;
            }
        }
        lwphyDestroyTensorDescriptor(tmx);
        tmx = nullptr;
    }
    return std::pair<lwphyTensorDescriptor_t, void*>(newDesc, pv);
}

////////////////////////////////////////////////////////////////////////
// dimensions_are_not()
bool dimensions_are_not(const mxArray* mxA, std::initializer_list<int> initList)
{
    if(initList.size() != mxGetNumberOfDimensions(mxA))
    {
        return true;
    }
    const mwSize* dims = mxGetDimensions(mxA);
    for(auto it = initList.begin(); it != initList.end(); ++it)
    {
        if(*it != *dims++)
        {
            return true;
        }
    }
    return false;
}

////////////////////////////////////////////////////////////////////////
// device_tensor_to_complex_mxarray()
bool device_tensor_to_complex_mxarray(mxArray*                                  mxA,
                                      std::pair<lwphyTensorDescriptor_t, void*> tensor)
{
    // Create a duplicate tensor descriptor along with a device mapped
    // host buffer so that we can use the lwphyColwertTensor() function
    // to perform the copy.
    lwphyTensorDescriptor_t hostCopyDesc = copy_tensor_descriptor(tensor.first,
                                                                  LWPHY_TENSOR_ALIGN_TIGHT);
    if(hostCopyDesc)
    {
        void*  pvhost   = nullptr;
        size_t sz       = 0;
        if(LWPHY_STATUS_SUCCESS != lwphyGetTensorSizeInBytes(hostCopyDesc, &sz))
        {
            lwphyDestroyTensorDescriptor(hostCopyDesc);
            return false;
        }
        lwdaError_t sLwda = lwdaHostAlloc(&pvhost, sz, lwdaHostAllocDefault);
        if(lwdaSuccess != sLwda)
        {
            fprintf(stderr, "lwdaHostAlloc() failure (%s)\n", lwdaGetErrorString(sLwda));
            lwphyDestroyTensorDescriptor(hostCopyDesc);
            return false;
        }
        lwphyStatus_t slwPHY = lwphyColwertTensor(hostCopyDesc,  // dst tensor desc
                                                  pvhost,        // dst data
                                                  tensor.first,  // src tensor desc
                                                  tensor.second, // src data address
                                                  0);            // stream
        if(LWPHY_STATUS_SUCCESS != slwPHY)
        {
            fprintf(stderr, "lwphyColwertTensor() failure (%s)\n", lwphyGetErrorString(slwPHY));
            lwphyDestroyTensorDescriptor(hostCopyDesc);
            lwdaFreeHost(pvhost);
            pvhost = nullptr;
            return false;
        }

        void* pr = mxGetPr(mxA);
        void* pi = mxGetPi(mxA);
        switch(mxGetClassID(mxA))
        {
        case mxUNKNOWN_CLASS:
        case mxCELL_CLASS:
        case mxSTRUCT_CLASS:
        case mxLOGICAL_CLASS:
        case mxVOID_CLASS:
        case mxCHAR_CLASS:
        case mxFUNCTION_CLASS:
        case mxINT64_CLASS:
        case mxUINT64_CLASS:
        default:
            break;
        case mxDOUBLE_CLASS:
            memcpy_interleaved_to_split(static_cast<double*>(pr), static_cast<double*>(pi), static_cast<double*>(pvhost), mxGetNumberOfElements(mxA));
            break;
        case mxSINGLE_CLASS:
            memcpy_interleaved_to_split(static_cast<float*>(pr), static_cast<float*>(pi), static_cast<float*>(pvhost), mxGetNumberOfElements(mxA));
            break;
        case mxINT8_CLASS:
            memcpy_interleaved_to_split(static_cast<int8_t*>(pr), static_cast<int8_t*>(pi), static_cast<int8_t*>(pvhost), mxGetNumberOfElements(mxA));
            break;
        case mxUINT8_CLASS:
            memcpy_interleaved_to_split(static_cast<uint8_t*>(pr), static_cast<uint8_t*>(pi), static_cast<uint8_t*>(pvhost), mxGetNumberOfElements(mxA));
            break;
        case mxINT16_CLASS:
            memcpy_interleaved_to_split(static_cast<int16_t*>(pr), static_cast<int16_t*>(pi), static_cast<int16_t*>(pvhost), mxGetNumberOfElements(mxA));
            break;
        case mxUINT16_CLASS:
            memcpy_interleaved_to_split(static_cast<uint16_t*>(pr), static_cast<uint16_t*>(pi), static_cast<uint16_t*>(pvhost), mxGetNumberOfElements(mxA));
            break;
        case mxINT32_CLASS:
            memcpy_interleaved_to_split(static_cast<int32_t*>(pr), static_cast<int32_t*>(pi), static_cast<int32_t*>(pvhost), mxGetNumberOfElements(mxA));
            break;
        case mxUINT32_CLASS:
            memcpy_interleaved_to_split(static_cast<uint32_t*>(pr), static_cast<uint32_t*>(pi), static_cast<uint32_t*>(pvhost), mxGetNumberOfElements(mxA));
            break;
        }
        lwdaFreeHost(pvhost);
        lwphyDestroyTensorDescriptor(hostCopyDesc);
        return true;
    }
    return false;
}
    
} // namespace

////////////////////////////////////////////////////////////////////////
// lwphy_mex_create()
void lwphy_mex_create(int            nlhs,   /* number of expected outputs */
                      mxArray*       plhs[], /* array of pointers to output arguments */
                      int            nrhs,   /* number of inputs */
                      const mxArray* prhs[]) /* array of pointers to input arguments */
{
    if(nlhs != 1)
    {
        mexErrMsgTxt("lwphy MEX create: One output expected.");
        return;
    }
    plhs[0] = mxCreateDoubleMatrix(1, 1, mxREAL);
    double* c = nullptr;
    c = mxGetPr(plhs[0]);
    // For now, just return an incrementing number
    static double s_create_count = 1.0;
    *c = s_create_count;
    s_create_count += 1.0f;
}

////////////////////////////////////////////////////////////////////////
// lwphy_mex_delete()
void lwphy_mex_delete(int            nlhs,   /* number of expected outputs */
                      mxArray*       plhs[], /* array of pointers to output arguments */
                      int            nrhs,   /* number of inputs */
                      const mxArray* prhs[]) /* array of pointers to input arguments */
{
    // No-op for now
}
        
////////////////////////////////////////////////////////////////////////
// lwphy_mex_channel_est_MMSE_1D()
void lwphy_mex_channel_est_MMSE_1D(int            nlhs,   /* number of expected outputs */
                                   mxArray*       plhs[], /* array of pointers to output arguments */
                                   int            nrhs,   /* number of inputs */
                                   const mxArray* prhs[]) /* array of pointers to input arguments */
{
    //------------------------------------------------------------------
    // Validate inputs
    if(nlhs != 1)
    {
        mexErrMsgTxt("lwphy MEX channelEstMMSE1D: One output expected.");
        return;
    }
    if(nrhs < 5)
    {
        mexErrMsgTxt("lwphy MEX channelEstMMSE1D: At least 5 inputs expected.");
        return;
    }
    //for(int i = 0; i < nrhs; ++i)
    //{
    //    printf("i: %i, type: %s, num_dim: %i\n", i, mxGetClassName(prhs[i]), mxGetNumberOfDimensions(prhs[i]));
    //}
    const mxArray* mxY               = prhs[0];
    const mxArray* mxWfreq           = prhs[1];
    const mxArray* mxWtime           = prhs[2];
    const mxArray* mxDMRS_index_freq = prhs[3];
    const mxArray* mxDMRS_index_time = prhs[4];
    if((3 != mxGetNumberOfDimensions(mxY))               ||
       (3 != mxGetNumberOfDimensions(mxWfreq))           ||
       (3 != mxGetNumberOfDimensions(mxWtime))           ||
       (2 != mxGetNumberOfDimensions(mxDMRS_index_freq)) ||
       (2 != mxGetNumberOfDimensions(mxDMRS_index_time)) ||
       (!mxIsComplex(mxY))                               ||
       (mxIsComplex(mxWfreq))                            ||
       (mxIsComplex(mxWtime))                            ||
       (mxIsComplex(mxDMRS_index_freq))                  ||
       (mxIsComplex(mxDMRS_index_time)))
    {
        mexErrMsgTxt("lwphy MEX channelEstMMSE1D: Invalid input types.");
        return;
    }
    //------------------------------------------------------------------
    // For now, only handle specific sizes
    if(dimensions_are_not(mxY,               {1248, 14, 16 } )   ||
       dimensions_are_not(mxWfreq,           {96,   32, 156} )   ||
       dimensions_are_not(mxWtime,           {14,   4,  156} )   ||
       dimensions_are_not(mxDMRS_index_freq, {32,   156    } )   ||
       dimensions_are_not(mxDMRS_index_time, {4,    156    } ))
    {
        mexErrMsgTxt("lwphy MEX channelEstMMSE1D: Unsupported sizes.");
    }
    typedef std::pair<lwphyTensorDescriptor_t, void*> tensor_pair_t;
    typedef std::map<const char*, tensor_pair_t>      tensor_map_t;
    tensor_map_t                                      tensors;
    tensors.insert(tensor_map_t::value_type("Hinterp",         allocate_device_tensor({96,14,16,156}, LWPHY_C_32F, LWPHY_TENSOR_ALIGN_COALESCE)));
    tensors.insert(tensor_map_t::value_type("Wfreq",           allocate_device_tensor(mxWfreq,                     LWPHY_TENSOR_ALIGN_COALESCE)));
    tensors.insert(tensor_map_t::value_type("Wtime",           allocate_device_tensor(mxWtime,                     LWPHY_TENSOR_ALIGN_COALESCE)));
    tensors.insert(tensor_map_t::value_type("DMRS_index_freq", allocate_device_tensor(mxDMRS_index_freq,           LWPHY_TENSOR_ALIGN_TIGHT)));
    tensors.insert(tensor_map_t::value_type("DMRS_index_time", allocate_device_tensor(mxDMRS_index_time,           LWPHY_TENSOR_ALIGN_TIGHT)));
    tensors.insert(tensor_map_t::value_type("Y",               allocate_complex_device_tensor(mxY,                 LWPHY_TENSOR_ALIGN_COALESCE)));

    int failCount = 0;
    for(auto it = tensors.begin(); it != tensors.end(); ++it)
    {
        tensor_pair_t& p = it->second;
        if(!p.first || !p.second)
        {
            mexErrMsgTxt("lwphy MEX channelEstMMSE1D: Tensor initialization failure");
            ++failCount;
        }
    }
    lwphyStatus_t s = lwphyChannelEst1DTimeFrequency(tensors["Hinterp"].first,
                                                     tensors["Hinterp"].second,
                                                     tensors["Y"].first,
                                                     tensors["Y"].second,
                                                     tensors["Wfreq"].first,
                                                     tensors["Wfreq"].second,
                                                     tensors["Wtime"].first,
                                                     tensors["Wtime"].second,
                                                     tensors["DMRS_index_freq"].first,
                                                     tensors["DMRS_index_freq"].second,
                                                     tensors["DMRS_index_time"].first,
                                                     tensors["DMRS_index_time"].second,
                                                     0); // stream
    if(LWPHY_STATUS_SUCCESS != s)
    {
        std::string msg("lwphy MEX channelEstMMSE1D: Error performing channel estimation (");
        msg.append(lwphyGetErrorString(s));
        msg.append(")");
        mexErrMsgTxt(msg.c_str());
    }
    else
    {
        // Copy result tensor to mxArray
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Hard coded sizes for now...
        std::vector<int> Hdims{96,14,16,156};
        plhs[0] = mxCreateNumericArray(Hdims.size(), Hdims.data(), mxSINGLE_CLASS, mxCOMPLEX);
        if(!device_tensor_to_complex_mxarray(plhs[0], tensors["Hinterp"]))
        {
            mexErrMsgTxt("lwphy MEX channelEstMMSE1D: Error copying data to mxArray.");
        }
        //double* c = nullptr;
        //c = mxGetPr(plhs[0]);
        //*c = 0.0;
    }

    //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    // Clean up tensors
    for(auto it = tensors.begin(); it != tensors.end(); ++it)
    {
        tensor_pair_t& p = it->second;
        if(p.first)
        {
            lwphyDestroyTensorDescriptor(p.first);
        }
        if(p.second)
        {
            lwdaFree(p.second);
        }
    }
    //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    //printf("SUCCESS\n");
}

////////////////////////////////////////////////////////////////////////
// mexFunction()
// MEX library "gateway" function
void mexFunction(int            nlhs,   /* number of expected outputs */
                 mxArray*       plhs[], /* array of pointers to output arguments */
                 int            nrhs,   /* number of inputs */
                 const mxArray* prhs[]) /* array of pointers to input arguments */
{
    //printf("LWPHY MEX mexFunction: nrhs = %i, nlhs = %i\n", nrhs, nlhs);
    //int        i;
    ///* Examine input (right-hand-side) arguments. */
    //mexPrintf("\nThere are %d right-hand-side argument(s).", nrhs);
    //for(int i = 0; i < nrhs; ++i)
    //{
    //    mexPrintf("\n\tInput Arg %i is of type:\t%s ", i, mxGetClassName(prhs[i]));
    //}
    //
    /* Examine output (left-hand-side) arguments. */
    //mexPrintf("\n\nThere are %d left-hand-side argument(s).\n", nlhs);
    //if (nlhs > nrhs)
    //  mexErrMsgIdAndTxt( "MATLAB:mexfunction:inputOutputMismatch",
    //          "Cannot specify more outputs than inputs.\n");
    //
    //for (i=0; i<nlhs; i++)  {
    //    plhs[i]=mxCreateDoubleMatrix(1,1,mxREAL);
    //    *mxGetPr(plhs[i])=(double)mxGetNumberOfElements(prhs[i]);
    //}
    //------------------------------------------------------------------
    // Validate inputs
    if((nrhs < 1) || (mxCHAR_CLASS != mxGetClassID(prhs[0])))
    {
        mexErrMsgTxt("First input should be a command string.");
        return;
    }
    //------------------------------------------------------------------
    // Get the command string
    size_t arg1N = mxGetN(prhs[0]);
    std::vector<char> commandString(arg1N + 1);
    if(0 != mxGetString(prhs[0], commandString.data(), arg1N + 1))
    {
        std::string msg("Error retrieving command string (length ");
        msg.append(std::to_string(arg1N));
        msg.append(")"); 
        mexErrMsgTxt(msg.c_str());
        return;
    }
    //------------------------------------------------------------------
    // Discard the string function name and the internal handle passed
    // as arguments
    int nrhs_fn             = nrhs - 2;
    const mxArray** prhs_fn = prhs + 2;
    if(0 == strcmp(commandString.data(), "create"))
    {
        lwphy_mex_create(nlhs, plhs, nrhs_fn, prhs_fn);
    }
    else if(0 == strcmp(commandString.data(), "delete"))
    {
        lwphy_mex_delete(nlhs, plhs, nrhs_fn, prhs_fn);
    }
    else if(0 == strcmp(commandString.data(), "channelEstMMSE1D"))
    {
        lwphy_mex_channel_est_MMSE_1D(nlhs, plhs, nrhs_fn, prhs_fn);
    }
    else
    {
        mexErrMsgTxt("Unknown command string.");
    }
    
}
