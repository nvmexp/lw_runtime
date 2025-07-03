/*
 * Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 */

#include "lwphy.h"
#include <algorithm>
#include <new>
#include "ch_est.hpp"
#include "channel_eq.hpp"
#include "channel_est.hpp"
#include "colwert_tensor.lwh"
#include "ldpc.hpp"
#include "polar_encoder.hpp"
#include "tensor_desc.hpp"
#include "lwphy_context.hpp"

////////////////////////////////////////////////////////////////////////
// lwphyGetErrorString()
const char* lwphyGetErrorString(lwphyStatus_t status)
{ // clang-format off
    switch (status)
    {
    case LWPHY_STATUS_SUCCESS:            return "Success";
    case LWPHY_STATUS_INTERNAL_ERROR:     return "Internal error";
    case LWPHY_STATUS_NOT_SUPPORTED:      return "An operation was requested that is not lwrrently supported";
    case LWPHY_STATUS_ILWALID_ARGUMENT:   return "An invalid argument was provided";
    case LWPHY_STATUS_ARCH_MISMATCH:      return "Requested computation not supported on current architecture";
    case LWPHY_STATUS_ALLOC_FAILED:       return "Memory allocation failed";
    case LWPHY_STATUS_SIZE_MISMATCH:      return "Operand size mismatch";
    case LWPHY_STATUS_MEMCPY_ERROR:       return "Error performing memory copy";
    case LWPHY_STATUS_ILWALID_COLWERSION: return "Invalid data colwersion requested";
    case LWPHY_STATUS_UNSUPPORTED_TYPE:   return "Operation requested on unsupported type";
    case LWPHY_STATUS_UNSUPPORTED_LAYOUT: return "Operation requested on unsupported tensor layout";
    case LWPHY_STATUS_UNSUPPORTED_RANK:   return "Operation requested on unsupported rank";
    case LWPHY_STATUS_UNSUPPORTED_CONFIG: return "Operation requested using an unsupported configuration";
    default:                              return "Unknown status value";
    }
} // clang-format on

////////////////////////////////////////////////////////////////////////
// lwphyGetErrorName()
const char* lwphyGetErrorName(lwphyStatus_t status)
{ // clang-format off
    switch (status)
    {
    case LWPHY_STATUS_SUCCESS:            return "LWPHY_STATUS_SUCCESS";
    case LWPHY_STATUS_INTERNAL_ERROR:     return "LWPHY_STATUS_INTERNAL_ERROR";
    case LWPHY_STATUS_NOT_SUPPORTED:      return "LWPHY_STATUS_NOT_SUPPORTED";
    case LWPHY_STATUS_ILWALID_ARGUMENT:   return "LWPHY_STATUS_ILWALID_ARGUMENT";
    case LWPHY_STATUS_ARCH_MISMATCH:      return "LWPHY_STATUS_ARCH_MISMATCH";
    case LWPHY_STATUS_ALLOC_FAILED:       return "LWPHY_STATUS_ALLOC_FAILED";
    case LWPHY_STATUS_SIZE_MISMATCH:      return "LWPHY_STATUS_SIZE_MISMATCH";
    case LWPHY_STATUS_MEMCPY_ERROR:       return "LWPHY_STATUS_MEMCPY_ERROR";
    case LWPHY_STATUS_ILWALID_COLWERSION: return "LWPHY_STATUS_ILWALID_COLWERSION";
    case LWPHY_STATUS_UNSUPPORTED_TYPE:   return "LWPHY_STATUS_UNSUPPORTED_TYPE";
    case LWPHY_STATUS_UNSUPPORTED_LAYOUT: return "LWPHY_STATUS_UNSUPPORTED_LAYOUT";
    case LWPHY_STATUS_UNSUPPORTED_RANK:   return "LWPHY_STATUS_UNSUPPORTED_RANK";
    case LWPHY_STATUS_UNSUPPORTED_CONFIG: return "LWPHY_STATUS_UNSUPPORTED_CONFIG";
    default:                              return "LWPHY_UNKNOWN_STATUS";
    }
} // clang-format on

////////////////////////////////////////////////////////////////////////
// lwphyGetDataTypeString()
const char* LWPHYWINAPI lwphyGetDataTypeString(lwphyDataType_t t)
{ // clang-format off
    switch(t)
    {
    case LWPHY_VOID:  return "LWPHY_VOID";
    case LWPHY_BIT:   return "LWPHY_BIT";
    case LWPHY_R_16F: return "LWPHY_R_16F";
    case LWPHY_C_16F: return "LWPHY_C_16F";
    case LWPHY_R_32F: return "LWPHY_R_32F";
    case LWPHY_C_32F: return "LWPHY_C_32F";
    case LWPHY_R_8I:  return "LWPHY_R_8I";
    case LWPHY_C_8I:  return "LWPHY_C_8I";
    case LWPHY_R_8U:  return "LWPHY_R_8U";
    case LWPHY_C_8U:  return "LWPHY_C_8U";
    case LWPHY_R_16I: return "LWPHY_R_16I";
    case LWPHY_C_16I: return "LWPHY_C_16I";
    case LWPHY_R_16U: return "LWPHY_R_16U";
    case LWPHY_C_16U: return "LWPHY_C_16U";
    case LWPHY_R_32I: return "LWPHY_R_32I";
    case LWPHY_C_32I: return "LWPHY_C_32I";
    case LWPHY_R_32U: return "LWPHY_R_32U";
    case LWPHY_C_32U: return "LWPHY_C_32U";
    case LWPHY_R_64F: return "LWPHY_R_64F";
    case LWPHY_C_64F: return "LWPHY_C_64F";
    default:          return "UNKNOWN_TYPE";
    }
} // clang-format on


////////////////////////////////////////////////////////////////////////
// lwphyCreateContext()
lwphyStatus_t LWPHYWINAPI lwphyCreateContext(lwphyContext_t* pcontext,
                                             unsigned int    flags)
{
    if(!pcontext)
    {
        return LWPHY_STATUS_ILWALID_ARGUMENT;
    }
    *pcontext = nullptr;
    try
    {
        lwphy_i::context* c = new lwphy_i::context;
        *pcontext = static_cast<lwphyContext*>(c);
    }
    catch(std::bad_alloc& eba)
    {
        return LWPHY_STATUS_ALLOC_FAILED;
    }
    catch(...)
    {
        return LWPHY_STATUS_INTERNAL_ERROR;
    }
    return LWPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// lwphyDestroyContext()
lwphyStatus_t LWPHYWINAPI lwphyDestroyContext(lwphyContext_t ctx)
{
    if(!ctx)
    {
        return LWPHY_STATUS_ILWALID_ARGUMENT;
    }
    lwphy_i::context* c = static_cast<lwphy_i::context*>(ctx);
    delete c;
    return LWPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// lwphyCreateTensorDescriptor()
lwphyStatus_t lwphyCreateTensorDescriptor(lwphyTensorDescriptor_t* tensorDesc)
{
    //------------------------------------------------------------------
    // Validate arguments
    if(nullptr == tensorDesc)
    {
        return LWPHY_STATUS_ILWALID_ARGUMENT;
    }
    //------------------------------------------------------------------
    // Allocate the descriptor structure
    tensor_desc* tdesc = new(std::nothrow) tensor_desc;
    if(nullptr == tdesc)
    {
        return LWPHY_STATUS_ALLOC_FAILED;
    }
    //------------------------------------------------------------------
    // Populate the return address
    *tensorDesc = tdesc;
    return LWPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// lwphyDestroyTensorDescriptor()
lwphyStatus_t LWPHYWINAPI lwphyDestroyTensorDescriptor(lwphyTensorDescriptor_t tensorDesc)
{
    //------------------------------------------------------------------
    // Validate arguments
    if(nullptr == tensorDesc)
    {
        return LWPHY_STATUS_ILWALID_ARGUMENT;
    }
    //------------------------------------------------------------------
    // Free the structure previously allocated by lwphyCreateTensorDescriptor()
    tensor_desc* tdesc = static_cast<tensor_desc*>(tensorDesc);
    delete tdesc;
    return LWPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// lwphyGetTensorDescriptor()
lwphyStatus_t LWPHYWINAPI lwphyGetTensorDescriptor(const lwphyTensorDescriptor_t tensorDesc,
                                                   int                           numDimsRequested,
                                                   lwphyDataType_t*              dataType,
                                                   int*                          numDims,
                                                   int                           dimensions[],
                                                   int                           strides[])
{
    //------------------------------------------------------------------
    // Validate arguments
    if((nullptr == tensorDesc) ||
       ((numDimsRequested > 0) && (nullptr == dimensions)))
    {
        return LWPHY_STATUS_ILWALID_ARGUMENT;
    }
    //------------------------------------------------------------------
    const tensor_desc* tdesc = static_cast<const tensor_desc*>(tensorDesc);
    if(dataType)
    {
        *dataType = tdesc->type();
    }
    if(numDims)
    {
        *numDims = tdesc->layout().rank();
    }
    if((numDimsRequested > 0) && dimensions)
    {
        std::copy(tdesc->layout().dimensions.begin(),
                  tdesc->layout().dimensions.begin() + numDimsRequested,
                  dimensions);
    }
    if((numDimsRequested > 0) && strides)
    {
        std::copy(tdesc->layout().strides.begin(),
                  tdesc->layout().strides.begin() + numDimsRequested,
                  strides);
    }
    return LWPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// lwphySetTensorDescriptor()
lwphyStatus_t LWPHYWINAPI lwphySetTensorDescriptor(lwphyTensorDescriptor_t tensorDesc,
                                                   lwphyDataType_t         type,
                                                   int                     numDim,
                                                   const int               dim[],
                                                   const int               str[],
                                                   unsigned int            flags)
{
    //-----------------------------------------------------------------
    // Validate arguments. Validation of dimension/stride values will
    // occur in the call below.
    // Tensor descriptor must be non-NULL.
    // Dimensions array must be non-NULL.
    // If the LWPHY_TENSOR_STRIDES_AS_ORDER flag is given, the strides
    // pointer must be non-NULL.
    if(!tensorDesc ||
       !dim ||
       (is_set(flags, LWPHY_TENSOR_STRIDES_AS_ORDER) && !str))
    {
        return LWPHY_STATUS_ILWALID_ARGUMENT;
    }
    // Validate combinations of flags with the strides arg
    if(is_set(flags, LWPHY_TENSOR_STRIDES_AS_ORDER) &&
       std::any_of(str, str + numDim, [=](int s) { return s >= numDim; }))
    {
        return LWPHY_STATUS_ILWALID_ARGUMENT;
    }
    //-----------------------------------------------------------------
    // If the user passed TIGHT, we use nullptr as an argument to the
    // internal function.
    const int* strArg = is_set(flags, LWPHY_TENSOR_ALIGN_TIGHT) ? nullptr : str;
    //-----------------------------------------------------------------
    // Adjust the strides array using any optional flags. Adjusting for
    // alignment and changing the dimension order for strides only makes
    // sense when the number of dimensions is greater than 1.
    std::array<int, LWPHY_DIM_MAX> userStrides;

    if((is_set(flags, LWPHY_TENSOR_STRIDES_AS_ORDER) ||
        is_set(flags, LWPHY_TENSOR_ALIGN_COALESCE)) &&
       (!is_set(flags, LWPHY_TENSOR_ALIGN_TIGHT)) &&
       (numDim > 1))
    {
        // Create an array with the order of dimensions for striding
        std::array<int, LWPHY_DIM_MAX> dimOrder;
        if(is_set(flags, LWPHY_TENSOR_STRIDES_AS_ORDER))
        {
            // Make a local copy of the user array
            std::copy(str, str + numDim, dimOrder.begin());
        }
        else
        {
            // Fill dimOrder with sequentially increasing values,
            // begining with 0.
            std::iota(dimOrder.begin(), dimOrder.begin() + numDim, 0);
        }
        // Use the given array of strides as indices into the dimension
        // vector to determine the actual strides.
        userStrides[dimOrder[0]] = 1;
        for(int i = 1; i < numDim; ++i)
        {
            userStrides[dimOrder[i]] = dim[dimOrder[i - 1]] * userStrides[dimOrder[i - 1]];
        }
        // Adjust the alignment if necessary
        if(is_set(flags, LWPHY_TENSOR_ALIGN_COALESCE))
        {
            const int COALESCE_BYTES   = 128;
            const int num_elem_aligned = round_up_to_next(dim[dimOrder[0]],
                                                          get_element_multiple_for_alignment(COALESCE_BYTES, type));
            userStrides[dimOrder[1]]   = num_elem_aligned;
            for(int i = 2; i < numDim; ++i)
            {
                userStrides[dimOrder[i]] = dim[dimOrder[i - 1]] * userStrides[dimOrder[i - 1]];
            }
        }
        // Use the local array to set the tensor descriptor
        strArg = userStrides.data();
    }
    //-----------------------------------------------------------------
    // Modify the tensor descriptor using the given arguments
    tensor_desc& tdesc = static_cast<tensor_desc&>(*tensorDesc);
    return tdesc.set(type, numDim, dim, strArg) ? LWPHY_STATUS_SUCCESS : LWPHY_STATUS_ILWALID_ARGUMENT;
}

////////////////////////////////////////////////////////////////////////
// lwphyGetTensorSizeInBytes()
lwphyStatus_t LWPHYWINAPI lwphyGetTensorSizeInBytes(const lwphyTensorDescriptor_t tensorDesc,
                                                    size_t*                       psz)
{
    //-----------------------------------------------------------------
    // Validate arguments
    if(!tensorDesc || !psz)
    {
        return LWPHY_STATUS_ILWALID_ARGUMENT;
    }
    tensor_desc& tdesc = static_cast<tensor_desc&>(*tensorDesc);
    *psz               = tdesc.get_size_in_bytes();
    return LWPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// lwphyColwertTensor()
lwphyStatus_t LWPHYWINAPI lwphyColwertTensor(const lwphyTensorDescriptor_t tensorDescDst,
                                             void*                         dstAddr,
                                             lwphyTensorDescriptor_t       tensorDescSrc,
                                             const void*                   srcAddr,
                                             lwdaStream_t                  strm)
{
    //------------------------------------------------------------------
    // Validate arguments
    if(!tensorDescDst || !tensorDescSrc || !dstAddr || !srcAddr)
    {
        return LWPHY_STATUS_ILWALID_ARGUMENT;
    }
    tensor_desc&       tdDst = static_cast<tensor_desc&>(*tensorDescDst);
    const tensor_desc& tdSrc = static_cast<const tensor_desc&>(*tensorDescSrc);
    // Types don't need to match, but they can't be VOID
    if((tdDst.type() == LWPHY_VOID) || tdSrc.type() == LWPHY_VOID)
    {
        return LWPHY_STATUS_ILWALID_ARGUMENT;
    }
    const tensor_layout_any& layoutDst = tdDst.layout();
    const tensor_layout_any& layoutSrc = tdSrc.layout();
    if(!layoutDst.has_same_size(layoutSrc))
    {
        return LWPHY_STATUS_SIZE_MISMATCH;
    }
    //------------------------------------------------------------------
    // Handle "memcpy" case (same type and strides)
    if((tdDst.type() == tdSrc.type()) &&
       layoutDst.has_same_strides(layoutSrc))
    {
        // Assuming availability of lwdaMemcpyDefault (unified virtual
        // addressing), unifiedAddressing property in lwdaDeviceProperties
        lwdaError_t e = lwdaMemcpyAsync(dstAddr,
                                        srcAddr,
                                        tdDst.get_size_in_bytes(),
                                        lwdaMemcpyDefault,
                                        strm);
        return (lwdaSuccess != e) ? LWPHY_STATUS_MEMCPY_ERROR : LWPHY_STATUS_SUCCESS;
    }
    //------------------------------------------------------------------
    // Handle more complex cases here (different types and/or different
    // layouts).
    if(!colwert_tensor_layout(tdDst, dstAddr, tdSrc, srcAddr, strm))
        return LWPHY_STATUS_ILWALID_COLWERSION;
    return LWPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// lwphyChannelEst1DTimeFrequency()
lwphyStatus_t LWPHYWINAPI lwphyChannelEst1DTimeFrequency(const lwphyTensorDescriptor_t tensorDescDst,
                                                         void*                         dstAddr,
                                                         const lwphyTensorDescriptor_t tensorDescSymbols,
                                                         const void*                   symbolsAddr,
                                                         const lwphyTensorDescriptor_t tensorDescFreqFilters,
                                                         const void*                   freqFiltersAddr,
                                                         const lwphyTensorDescriptor_t tensorDescTimeFilters,
                                                         const void*                   timeFiltersAddr,
                                                         const lwphyTensorDescriptor_t tensorDescFreqIndices,
                                                         const void*                   freqIndicesAddr,
                                                         const lwphyTensorDescriptor_t tensorDescTimeIndices,
                                                         const void*                   timeIndicesAddr,
                                                         lwdaStream_t                  strm)
{
    //------------------------------------------------------------------
    // Validate inputs
    if(!tensorDescDst ||
       !dstAddr ||
       !tensorDescSymbols ||
       !symbolsAddr ||
       !tensorDescFreqFilters ||
       !freqFiltersAddr ||
       !tensorDescTimeFilters ||
       !timeFiltersAddr ||
       !tensorDescFreqIndices ||
       !freqIndicesAddr ||
       !tensorDescTimeIndices ||
       !timeIndicesAddr)
    {
        return LWPHY_STATUS_ILWALID_ARGUMENT;
    }
    //------------------------------------------------------------------
    // clang-format off
    tensor_pair       tDst        (static_cast<const tensor_desc&>(*tensorDescDst),         dstAddr);
    const_tensor_pair tSymbols    (static_cast<const tensor_desc&>(*tensorDescSymbols),     symbolsAddr);
    const_tensor_pair tFreqFilters(static_cast<const tensor_desc&>(*tensorDescFreqFilters), freqFiltersAddr);
    const_tensor_pair tTimeFilters(static_cast<const tensor_desc&>(*tensorDescTimeFilters), timeFiltersAddr);
    const_tensor_pair tFreqIndices(static_cast<const tensor_desc&>(*tensorDescFreqIndices), freqIndicesAddr);
    const_tensor_pair tTimeIndices(static_cast<const tensor_desc&>(*tensorDescTimeIndices), timeIndicesAddr);
    // clang-format on
    channel_est::mmse_1D_time_frequency(tDst,
                                        tSymbols,
                                        tFreqFilters,
                                        tTimeFilters,
                                        tFreqIndices,
                                        tTimeIndices,
                                        strm);

    return LWPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// lwphyChannelEq()
lwphyStatus_t LWPHYWINAPI lwphyChannelEq(unsigned int                  nBSAnts,
                                         unsigned int                  nLayers,
                                         unsigned int                  Nh,
                                         unsigned int                  Nf,
                                         unsigned int                  Nd,
                                         unsigned int                  qam,
                                         const lwphyTensorDescriptor_t tDescDataSymLoc,
                                         const void*                   dataSymLocAddr,
                                         const lwphyTensorDescriptor_t tDescDataRx,
                                         const void*                   dataRxAddr,
                                         const lwphyTensorDescriptor_t tDescH,
                                         const void*                   HAddr,
                                         const lwphyTensorDescriptor_t tDescNoisePwr,
                                         const void*                   noisePwrAddr,
                                         lwphyTensorDescriptor_t       tDescDataEq,
                                         void*                         dataEqAddr,
                                         lwphyTensorDescriptor_t       tDescReeDiag,
                                         void*                         reeDiagAddr,
                                         lwphyTensorDescriptor_t       tDescLLR,
                                         void*                         LLRAddr,
                                         lwdaStream_t                  strm)
{
    //------------------------------------------------------------------
    // Validate inputs
    if(!tDescDataSymLoc ||
       !dataSymLocAddr ||
       !tDescDataRx ||
       !dataRxAddr ||
       !tDescH ||
       !HAddr ||
       !tDescNoisePwr ||
       !noisePwrAddr ||
       !tDescDataEq ||
       !dataEqAddr ||
       !tDescReeDiag ||
       !reeDiagAddr ||
       !tDescLLR ||
       !LLRAddr)
    {
        return LWPHY_STATUS_ILWALID_ARGUMENT;
    }

    //------------------------------------------------------------------
    // clang-format off
    const_tensor_pair tPairDataSymLoc(static_cast<const tensor_desc&>(*tDescDataSymLoc), dataSymLocAddr);
    const_tensor_pair tPairDataRx    (static_cast<const tensor_desc&>(*tDescDataRx),     dataRxAddr);
    const_tensor_pair tPairH         (static_cast<const tensor_desc&>(*tDescH),          HAddr);
    const_tensor_pair tPairNoisePwr  (static_cast<const tensor_desc&>(*tDescNoisePwr),   noisePwrAddr);
    tensor_pair       tPairDataEq    (static_cast<const tensor_desc&>(*tDescDataEq),     dataEqAddr);
    tensor_pair       tPairReeDiag   (static_cast<const tensor_desc&>(*tDescReeDiag),    reeDiagAddr);
    tensor_pair       tPairLLR       (static_cast<const tensor_desc&>(*tDescLLR),        LLRAddr);
    // clang-format on

    channel_eq::equalize(static_cast<uint32_t>(nBSAnts),
                         static_cast<uint32_t>(nLayers),
                         static_cast<uint32_t>(Nh),
                         static_cast<uint32_t>(Nf),
                         static_cast<uint32_t>(Nd),
                         static_cast<channel_eq::QAM_t>(qam),
                         tPairDataSymLoc,
                         tPairDataRx,
                         tPairH,
                         tPairNoisePwr,
                         tPairDataEq,
                         tPairReeDiag,
                         tPairLLR,
                         strm);

    return LWPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// lwphyChannelEst()
lwphyStatus_t LWPHYWINAPI lwphyChannelEst(unsigned int                  cellId,
                                          unsigned int                  slotNum,
                                          unsigned int                  nBSAnts,
                                          unsigned int                  nLayers,
                                          unsigned int                  nDMRSSyms,
                                          unsigned int                  nDMRSGridsPerPRB,
                                          unsigned int                  activeDMRSGridBmsk,
                                          unsigned int                  nTotalDMRSPRB,
                                          unsigned int                  nTotalDataPRB,
                                          unsigned int                  Nh,
                                          const lwphyTensorDescriptor_t tDescDataRx,
                                          const void*                   dataRxAddr,
                                          const lwphyTensorDescriptor_t tDescWFreq,
                                          const void*                   WFreqAddr,
                                          const lwphyTensorDescriptor_t tDescDescrShiftSeq,
                                          const void*                   descrShiftSeqAddr,
                                          const lwphyTensorDescriptor_t tDeslwnShiftSeq,
                                          const void*                   unShiftSeqAddr,
                                          const lwphyTensorDescriptor_t tDescHEst,
                                          void*                         HEstAddr,
                                          const lwphyTensorDescriptor_t tDescDbg,
                                          void*                         dbgAddr,
                                          lwdaStream_t                  strm)
{
    //------------------------------------------------------------------
    // Validate inputs
    if(!tDescDataRx          ||
       !dataRxAddr           ||
       !tDescWFreq           ||
       !WFreqAddr            ||
       !tDescDescrShiftSeq   ||
       !descrShiftSeqAddr    || 
       !tDeslwnShiftSeq      ||
       !unShiftSeqAddr       ||
       !tDescHEst            ||
       !HEstAddr             ||               
       !tDescDbg             ||               
       !dbgAddr)               
    {
        return LWPHY_STATUS_ILWALID_ARGUMENT;
    }

    //------------------------------------------------------------------
    // clang-format off
    const_tensor_pair tPairDataRx       (static_cast<const tensor_desc&>(*tDescDataRx)       , dataRxAddr);
    const_tensor_pair tPairWFreq        (static_cast<const tensor_desc&>(*tDescWFreq)        , WFreqAddr);
    const_tensor_pair tPairDescrShiftSeq(static_cast<const tensor_desc&>(*tDescDescrShiftSeq), descrShiftSeqAddr);
    const_tensor_pair tPairUnShiftSeq   (static_cast<const tensor_desc&>(*tDeslwnShiftSeq)   , unShiftSeqAddr);
    tensor_pair       tPairHEst         (static_cast<const tensor_desc&>(*tDescHEst)         , HEstAddr);
    tensor_pair       tPairDbg          (static_cast<const tensor_desc&>(*tDescDbg)          , dbgAddr);
    // clang-format on

    ch_est::estimate_channel(static_cast<uint32_t>(cellId),
                             static_cast<uint32_t>(slotNum),
                             static_cast<uint32_t>(nBSAnts),
                             static_cast<uint32_t>(nLayers),
                             static_cast<uint32_t>(nDMRSSyms),
                             static_cast<uint32_t>(nDMRSGridsPerPRB),
                             static_cast<uint32_t>(activeDMRSGridBmsk),
                             static_cast<uint32_t>(nTotalDMRSPRB),
                             static_cast<uint32_t>(nTotalDataPRB),
                             static_cast<uint32_t>(Nh),
                             tPairDataRx,
                             tPairWFreq,
                             tPairDescrShiftSeq,
                             tPairUnShiftSeq,
                             tPairHEst,
                             tPairDbg,
                             strm);

    return LWPHY_STATUS_SUCCESS;
}

lwphyStatus_t LWPHYWINAPI lwphyErrorCorrectionLDPCEncode(lwphyTensorDescriptor_t inDesc,
                                                         void*                   inAddr,
                                                         lwphyTensorDescriptor_t outDesc,
                                                         void*                   outAddr,
                                                         int                     BG,
                                                         int                     Kb,
                                                         int                     Z,
                                                         bool                    puncture,
                                                         int                     maxParityNodes, /* if unknown, set to 0 */
                                                         int                     rv, /* redundancy version */
                                                         lwdaStream_t            strm)
{

    if ((rv < 0) || (rv > 3)) {
        return LWPHY_STATUS_ILWALID_ARGUMENT;
    }

    tensor_pair in_pair(static_cast<const tensor_desc&>(*inDesc), inAddr);
    tensor_pair out_pair(static_cast<const tensor_desc&>(*outDesc), outAddr);

    LDPC_config config(BG,
                       Kb,
                       8,
                       Z,
                       LWPHY_BIT,
                       in_pair.first.get().layout().dimensions[1]);

    return ldpc::encode(in_pair, out_pair, config, puncture, maxParityNodes, rv, strm);
}

////////////////////////////////////////////////////////////////////////
// lwphyChannelEqCoefComp()
lwphyStatus_t LWPHYWINAPI lwphyChannelEqCoefCompute(unsigned int                  nBSAnts,
                                                    unsigned int                  nLayers,
                                                    unsigned int                  Nh,
                                                    unsigned int                  Nprb,
                                                    const lwphyTensorDescriptor_t tDescH,
                                                    const void*                   HAddr,
                                                    const lwphyTensorDescriptor_t tDescNoisePwr,
                                                    const void*                   noisePwrAddr,
                                                    lwphyTensorDescriptor_t       tDescCoef,
                                                    void*                         coefAddr,
                                                    lwphyTensorDescriptor_t       tDescReeDiagIlw,
                                                    void*                         reeDiagIlwAddr,
                                                    lwphyTensorDescriptor_t       tDescDbg,
                                                    void*                         dbgAddr,
                                                    lwdaStream_t                  strm)
{
    //------------------------------------------------------------------
    // Validate inputs
    if(!tDescH          ||
       !HAddr           ||
       !tDescNoisePwr   ||
       !noisePwrAddr    ||
       !tDescCoef       ||
       !coefAddr        ||
       !tDescReeDiagIlw ||
       !reeDiagIlwAddr  ||
       !tDescDbg        ||
       !dbgAddr)
    {
        return LWPHY_STATUS_ILWALID_ARGUMENT;
    }

    //------------------------------------------------------------------
    // clang-format off
    const_tensor_pair tPairH         (static_cast<const tensor_desc&>(*tDescH)         ,  HAddr);
    const_tensor_pair tPairNoisePwr  (static_cast<const tensor_desc&>(*tDescNoisePwr)  ,  noisePwrAddr);
    tensor_pair       tPairCoef      (static_cast<const tensor_desc&>(*tDescCoef)      ,  coefAddr);
    tensor_pair       tPairReeDiagIlw(static_cast<const tensor_desc&>(*tDescReeDiagIlw),  reeDiagIlwAddr);
    tensor_pair       tPairDbg       (static_cast<const tensor_desc&>(*tDescDbg)       ,  dbgAddr);
    // clang-format on

    channel_eq::eqCoefCompute(static_cast<uint32_t>(nBSAnts),
                              static_cast<uint32_t>(nLayers),
                              static_cast<uint32_t>(Nh),
                              static_cast<uint32_t>(Nprb),
                              tPairH,
                              tPairNoisePwr,
                              tPairCoef,
                              tPairReeDiagIlw,
                              tPairDbg,
                              strm);

    return LWPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// lwphyChannelEqSoftDemap()
lwphyStatus_t LWPHYWINAPI lwphyChannelEqSoftDemap(unsigned int                  nBSAnts,
                                                  unsigned int                  nLayers,
                                                  unsigned int                  Nh,
                                                  unsigned int                  Nd,
                                                  unsigned int                  Nprb,
                                                  const lwphyTensorDescriptor_t tDescDataSymbLoc,
                                                  const void*                   dataSymbLocAddr,
                                                  const lwphyTensorDescriptor_t tDescQamInfo,
                                                  const void*                   qamInfoAddr,
                                                  const lwphyTensorDescriptor_t tDescCoef,
                                                  const void*                   coefAddr,
                                                  const lwphyTensorDescriptor_t tDescReeDiagIlw,
                                                  const void*                   reeDiagIlwAddr,
                                                  const lwphyTensorDescriptor_t tDescDataRx,
                                                  const void*                   dataRxAddr,
                                                  lwphyTensorDescriptor_t       tDescDataEq,
                                                  void*                         dataEqAddr,
                                                  lwphyTensorDescriptor_t       tDescLlr,
                                                  void*                         llrAddr,
                                                  lwphyTensorDescriptor_t       tDescDbg,
                                                  void*                         dbgAddr,
                                                  lwdaStream_t                  strm)
{
    //------------------------------------------------------------------
    // Validate inputs
    if(!tDescDataSymbLoc ||
       !dataSymbLocAddr  ||
       !tDescQamInfo     || 
       !qamInfoAddr      ||
       !tDescCoef        ||
       !coefAddr         || 
       !tDescReeDiagIlw  ||
       !reeDiagIlwAddr   ||
       !tDescDataRx      ||
       !dataRxAddr       ||
       !tDescDataEq      ||
       !dataEqAddr       ||
       !tDescLlr         ||
       !llrAddr          ||
       !tDescDbg         ||
       !dbgAddr)
    {
        return LWPHY_STATUS_ILWALID_ARGUMENT;
    }

    //------------------------------------------------------------------
    // clang-format off
    const_tensor_pair tPairDataSymbLoc(static_cast<const tensor_desc&>(*tDescDataSymbLoc), dataSymbLocAddr);
    const_tensor_pair tPairQamInfo    (static_cast<const tensor_desc&>(*tDescQamInfo)    , qamInfoAddr);
    const_tensor_pair tPairCoef       (static_cast<const tensor_desc&>(*tDescCoef)       , coefAddr);
    const_tensor_pair tPairReeDiagIlw (static_cast<const tensor_desc&>(*tDescReeDiagIlw) , reeDiagIlwAddr);
    const_tensor_pair tPairDataRx     (static_cast<const tensor_desc&>(*tDescDataRx)     , dataRxAddr);
    tensor_pair       tPairDataEq     (static_cast<const tensor_desc&>(*tDescDataEq)     , dataEqAddr);
    tensor_pair       tPairLlr        (static_cast<const tensor_desc&>(*tDescLlr)        , llrAddr);
    tensor_pair       tPairDbg        (static_cast<const tensor_desc&>(*tDescDbg)        , dbgAddr);
    // clang-format on

    channel_eq::eqSoftDemap(static_cast<uint32_t>(nBSAnts),
                             static_cast<uint32_t>(nLayers),
                             static_cast<uint32_t>(Nh),
                             static_cast<uint32_t>(Nd),
                             static_cast<uint32_t>(Nprb),
                             tPairDataSymbLoc,
                             tPairQamInfo,
                             tPairCoef,
                             tPairReeDiagIlw,
                             tPairDataRx,
                             tPairDataEq,
                             tPairLlr,
                             tPairDbg,
                             strm);

    return LWPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// lwphyErrorCorrectionLDPCDecode()
lwphyStatus_t LWPHYWINAPI lwphyErrorCorrectionLDPCDecode(lwphyLDPCDecoder_t            decoder,
                                                         lwphyTensorDescriptor_t       tensorDescDst,
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
    std::array<int, 4> BG2_Kb = {6, 8, 9, 10};
    if(!decoder ||
       !tensorDescDst ||
       !dstAddr ||
       !tensorDescLLR ||
       !LLRAddr ||
       (maxNumIterations < 0) ||
       (BG < 1) ||
       (BG > 2) ||
       ((1 == BG) ? (Kb != 22) : (BG2_Kb.end() == std::find(BG2_Kb.begin(), BG2_Kb.end(), Kb))) ||
       (Z < 2) ||
       (Z > 384) ||
       (mb < 4) ||
       (mb > 46))
    {
        return LWPHY_STATUS_ILWALID_ARGUMENT;
    }
    // clang-format off
    ldpc::decoder&    d = static_cast<ldpc::decoder&>(*decoder);
    tensor_pair       tDst(static_cast<const tensor_desc&>(*tensorDescDst), dstAddr);
    const_tensor_pair tLLR(static_cast<const tensor_desc&>(*tensorDescLLR), LLRAddr);
    // clang-format on
    //------------------------------------------------------------------
    // Initialize an LDPC configuration
    const tensor_desc& tLLRDesc = tLLR.first.get();
    const tensor_desc& tDstDesc = tDst.first.get();

    LDPC_config config(BG,
                       Kb,
                       mb,
                       Z,
                       tLLR.first.get().type(),
                       tLLR.first.get().layout().dimensions[1],
                       (0 != earlyTermination),
                       maxNumIterations);
    //------------------------------------------------------------------
    return d.decode(tDst,
                    tLLR,
                    config,
                    normalization,
                    results,
                    workspace,
                    algoIndex,
                    flags,
                    strm,
                    reserved);
}

////////////////////////////////////////////////////////////////////////
// lwphyErrorCorrectionLDPCDecodeGetWorkspaceSize()
lwphyStatus_t LWPHYWINAPI lwphyErrorCorrectionLDPCDecodeGetWorkspaceSize(lwphyLDPCDecoder_t decoder,
                                                                         int                BG,
                                                                         int                Kb,
                                                                         int                mb,
                                                                         int                Z,
                                                                         int                numCodeWords,
                                                                         lwphyDataType_t    LLRtype,
                                                                         int                algoIndex,
                                                                         size_t*            sizeInBytes)
{
    static const std::array<int, 2> BG_valid = {1, 2};
    static const std::array<int, 5> Kb_valid = {22, 10, 9, 8, 6};
    // clang-format off
    static const std::array<int, 51> Z_valid =
    {
        2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,  14,
       15,  16,  18,  20,  22,  24,  26,  28,  30,  32,  36,  40,  44,
       48,  52,  56,  60,  64,  72,  80,  88,  96, 104, 112, 120, 128,
      144, 160, 176, 192, 208, 224, 240, 256, 288, 320, 352, 384
    };
    // clang-format on
    if(!decoder ||
       (std::find(BG_valid.begin(), BG_valid.end(), BG) == BG_valid.end()) ||
       (std::find(Kb_valid.begin(), Kb_valid.end(), Kb) == Kb_valid.end()) ||
       (std::find(Z_valid.begin(), Z_valid.end(), Z) == Z_valid.end()) ||
       (numCodeWords <= 0) ||
       !sizeInBytes ||
       (mb < 4) ||
       (mb > 46))
    {
        return LWPHY_STATUS_ILWALID_ARGUMENT;
    }
    ldpc::decoder&          d             = static_cast<ldpc::decoder&>(*decoder);
    std::pair<bool, size_t> workspaceSize = d.workspace_size(BG, Kb, mb, Z, LLRtype, algoIndex, numCodeWords);
    if(workspaceSize.first)
    {
        *sizeInBytes = workspaceSize.second;
        return LWPHY_STATUS_SUCCESS;
    }
    else
    {
        return LWPHY_STATUS_UNSUPPORTED_CONFIG;
    }
}

////////////////////////////////////////////////////////////////////////
// lwphyCreateLDPCDecoder()
lwphyStatus_t LWPHYWINAPI lwphyCreateLDPCDecoder(lwphyContext_t      context,
                                                 lwphyLDPCDecoder_t* pdecoder,
                                                 unsigned int        flags)
{
    if(!pdecoder || !context)
    {
        return LWPHY_STATUS_ILWALID_ARGUMENT;
    }
    *pdecoder = nullptr;
    lwphy_i::context& ctx = static_cast<lwphy_i::context&>(*context);
    try
    {
        ldpc::decoder* d = new ldpc::decoder(ctx);
        *pdecoder = static_cast<lwphyLDPCDecoder_t>(d);
    }
    catch(std::bad_alloc& eba)
    {
        return LWPHY_STATUS_ALLOC_FAILED;
    }
    catch(...)
    {
        return LWPHY_STATUS_INTERNAL_ERROR;
    }
    return LWPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// lwphyDestroyLDPCDecoder()
lwphyStatus_t LWPHYWINAPI lwphyDestroyLDPCDecoder(lwphyLDPCDecoder_t decoder)
{
    if(!decoder)
    {
        return LWPHY_STATUS_ILWALID_ARGUMENT;
    }
    ldpc::decoder* d = static_cast<ldpc::decoder*>(decoder);
    delete d;
    return LWPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// lwphyPolarEncRateMatch()
lwphyStatus_t LWPHYWINAPI lwphyPolarEncRateMatch(unsigned int   nInfoBits,
                                                 unsigned int   nTxBits,
                                                 uint8_t const* pInfoBits,
                                                 uint32_t*      pNCodedBits,
                                                 uint8_t*       pCodedBits,
                                                 uint8_t*       pTxBits,
                                                 lwdaStream_t   strm)
{
    //------------------------------------------------------------------
    // Validate inputs
    if((!pInfoBits) || (!pNCodedBits) || (!pCodedBits) || (!pTxBits) ||
       (nInfoBits < 1) || (nInfoBits > LWPHY_POLAR_ENC_MAX_INFO_BITS) ||
       (nTxBits < 1) || (nTxBits > LWPHY_POLAR_ENC_MAX_TX_BITS))
    {
        return LWPHY_STATUS_ILWALID_ARGUMENT;
    }

    //------------------------------------------------------------------
    polar_encoder::encodeRateMatch(static_cast<uint32_t>(nInfoBits),
                                   static_cast<uint32_t>(nTxBits),
                                   pInfoBits,
                                   pNCodedBits,
                                   pCodedBits,
                                   pTxBits,
                                   strm);

    return LWPHY_STATUS_SUCCESS;
}

