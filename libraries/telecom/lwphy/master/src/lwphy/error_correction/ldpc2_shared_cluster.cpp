/*
 * Copyright (c) 2019, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 */


#include "ldpc2_shared_cluster.hpp"
#include "lwphy_internal.h"

using namespace ldpc2;

namespace ldpc2
{

////////////////////////////////////////////////////////////////////////
// decode_ldpc2_shared_cluster_index_half()
lwphyStatus_t decode_ldpc2_shared_cluster_index_half(const LDPC_config&        config,
                                                     const LDPC_kernel_params& params,
                                                     const dim3&               grdDim,
                                                     const dim3&               blkDim,
                                                     lwdaStream_t              strm)
{
    lwphyStatus_t s = LWPHY_STATUS_NOT_SUPPORTED;
    if(config.BG == 1)
    {
        switch(config.Z)
        {
        case 64:  s = decode_ldpc2_shared_cluster_index_half_BG1_Z64 (config, params, grdDim, blkDim, strm); break;
        case 96:  s = decode_ldpc2_shared_cluster_index_half_BG1_Z96 (config, params, grdDim, blkDim, strm); break;
        case 128: s = decode_ldpc2_shared_cluster_index_half_BG1_Z128(config, params, grdDim, blkDim, strm); break;
        case 160: s = decode_ldpc2_shared_cluster_index_half_BG1_Z160(config, params, grdDim, blkDim, strm); break;
        case 192: s = decode_ldpc2_shared_cluster_index_half_BG1_Z192(config, params, grdDim, blkDim, strm); break;
        case 224: s = decode_ldpc2_shared_cluster_index_half_BG1_Z224(config, params, grdDim, blkDim, strm); break;
        case 256: s = decode_ldpc2_shared_cluster_index_half_BG1_Z256(config, params, grdDim, blkDim, strm); break;
        case 288: s = decode_ldpc2_shared_cluster_index_half_BG1_Z288(config, params, grdDim, blkDim, strm); break;
        case 320: s = decode_ldpc2_shared_cluster_index_half_BG1_Z320(config, params, grdDim, blkDim, strm); break;
        case 352: s = decode_ldpc2_shared_cluster_index_half_BG1_Z352(config, params, grdDim, blkDim, strm); break;
        case 384: s = decode_ldpc2_shared_cluster_index_half_BG1_Z384(config, params, grdDim, blkDim, strm); break;
        default:                                                                                             break;
        }
    }
    else if(config.BG == 2)
    {
        switch(config.Z)
        {
        case 64:  s = decode_ldpc2_shared_cluster_index_half_BG2_Z64 (config, params, grdDim, blkDim, strm); break;
        case 96:  s = decode_ldpc2_shared_cluster_index_half_BG2_Z96 (config, params, grdDim, blkDim, strm); break;
        case 128: s = decode_ldpc2_shared_cluster_index_half_BG2_Z128(config, params, grdDim, blkDim, strm); break;
        case 160: s = decode_ldpc2_shared_cluster_index_half_BG2_Z160(config, params, grdDim, blkDim, strm); break;
        case 192: s = decode_ldpc2_shared_cluster_index_half_BG2_Z192(config, params, grdDim, blkDim, strm); break;
        case 224: s = decode_ldpc2_shared_cluster_index_half_BG2_Z224(config, params, grdDim, blkDim, strm); break;
        case 256: s = decode_ldpc2_shared_cluster_index_half_BG2_Z256(config, params, grdDim, blkDim, strm); break;
        case 288: s = decode_ldpc2_shared_cluster_index_half_BG2_Z288(config, params, grdDim, blkDim, strm); break;
        case 320: s = decode_ldpc2_shared_cluster_index_half_BG2_Z320(config, params, grdDim, blkDim, strm); break;
        case 352: s = decode_ldpc2_shared_cluster_index_half_BG2_Z352(config, params, grdDim, blkDim, strm); break;
        case 384: s = decode_ldpc2_shared_cluster_index_half_BG2_Z384(config, params, grdDim, blkDim, strm); break;
        default:                                                                                             break;
        }
    }
    return s;
}

} // namespace ldpc2

namespace ldpc
{

////////////////////////////////////////////////////////////////////////
// decode_ldpc2_shared_cluster_index()
lwphyStatus_t decode_ldpc2_shared_cluster_index(decoder&               dec,
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
        s = decode_ldpc2_shared_cluster_index_half(config,
                                                   params,
                                                   grdDim,
                                                   blkDim,
                                                   strm);
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
// decode_ldpc2_shared_cluster_index_workspace_size()
std::pair<bool, size_t> decode_ldpc2_shared_cluster_index_workspace_size(const decoder&     dec,
                                                                         const LDPC_config& cfg)
{
    return std::pair<bool, size_t>(true, 0);
}

} // namespace ldpc
