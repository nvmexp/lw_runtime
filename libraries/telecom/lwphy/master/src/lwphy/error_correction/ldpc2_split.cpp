/*
 * Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 */


#include "ldpc2_split.hpp"
#include "lwphy_internal.h"

using namespace ldpc2;

namespace ldpc
{

////////////////////////////////////////////////////////////////////////
// decode_ldpc2_split_index()
lwphyStatus_t decode_ldpc2_split_index(decoder&               dec,
                                       LDPC_output_t&         tDst,
                                       const_tensor_pair&     tLLR,
                                       const LDPC_config&     config,
                                       float                  normalization,
                                       lwphyLDPCResults_t*    results,
                                       void*                  workspace,
                                       lwphyLDPCDiagnostic_t* diag,
                                       lwdaStream_t           strm)
{
    //------------------------------------------------------------------
    lwphyDataType_t llrType = tLLR.first.get().type();
    //------------------------------------------------------------------
    dim3 grdDim(config.num_codewords);
    dim3 blkDim(config.Z);

    //------------------------------------------------------------------
    // Initialize the kernel params struct
    LDPC_kernel_params params(config, tLLR, tDst, normalization, workspace);

    lwphyStatus_t s = LWPHY_STATUS_NOT_SUPPORTED;
    
    if(llrType == LWPHY_R_16F)
    {
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Colwert the normalization value to __half2
        params.norm.f16x2 = __float2half2_rn(params.norm.f32);
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Determine the maximum amount of shared memory that the
        // current device can support
        int32_t device_shmem_max = dec.max_shmem_per_block_optin();
        if(device_shmem_max <= 0)
        {
            return LWPHY_STATUS_INTERNAL_ERROR;
        }
        switch(device_shmem_max)
        {
        case (96*1024): s = decode_ldpc2_split_half_96KB(config, params, grdDim, blkDim, strm); break; // Volta:  96 KiB max (opt-in)
        case (64*1024):                                                                         break; // Turing: 64 KiB max (opt-in)
        default:                                                                                break;
        }
    }
    if(LWPHY_STATUS_SUCCESS != s)
    {
        return s;
    }

#if LWPHY_DEBUG
    lwdaDeviceSynchronize();
#endif
    lwdaError_t e = lwdaGetLastError();
    DEBUG_PRINTF("LWCA STATUS (%s:%i): %s\n", __FILE__, __LINE__, lwdaGetErrorString(e));
    return (e == lwdaSuccess) ? LWPHY_STATUS_SUCCESS : LWPHY_STATUS_INTERNAL_ERROR;
}

//----------------------------------------------------------------------
// decode_ldpc2_split_index_workspace_size()
std::pair<bool, size_t> decode_ldpc2_split_index_workspace_size(const decoder&     dec,
                                                                const LDPC_config& cfg)
{
    // For now, cC2V class in ldp2c.lwh loads and stores from global memory
    // using an offset that assumes ALL cC2Vs are in global memory, even
    // though some are in shared memory.
    // TODO: Modify the global load/store and reduce the amount of
    // allocation here.
    if(LWPHY_R_32F == cfg.type)
    {
        return std::pair<bool, size_t>(true, cfg.num_codewords * cfg.mb * cfg.Z * sizeof(int4));
    }
    else if(LWPHY_R_16F == cfg.type)
    {
        // Assumes all of workspace is used for cC2V messages (i.e. no APP values)
        return std::pair<bool, size_t>(true, cfg.num_codewords * cfg.mb * cfg.Z * sizeof(int2));
    }
    else
    {
        return std::pair<bool, size_t>(false, 0);
    }

    return std::pair<bool, size_t>(true, 0);
}

} // namespace ldpc
