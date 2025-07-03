/*
 * Copyright (c) 2019, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 */

#include "ldpc2_shared_dynamic.hpp"
#include <utility>

using namespace ldpc2;

namespace ldpc2
{

////////////////////////////////////////////////////////////////////////
// decode_ldpc2_shared_dynamic_index_half()
lwphyStatus_t decode_ldpc2_shared_dynamic_index_half(ldpc::decoder&            dec,
                                                     const LDPC_config&        config,
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
        //case 64:  s = decode_ldpc2_shared_dynamic_half_BG1_Z64 (dec, config, params, grdDim, blkDim, strm); break;
        //case 96:  s = decode_ldpc2_shared_dynamic_half_BG1_Z96 (dec, config, params, grdDim, blkDim, strm); break;
        //case 128: s = decode_ldpc2_shared_dynamic_half_BG1_Z128(dec, config, params, grdDim, blkDim, strm); break;
        //case 160: s = decode_ldpc2_shared_dynamic_half_BG1_Z160(dec, config, params, grdDim, blkDim, strm); break;
        //case 192: s = decode_ldpc2_shared_dynamic_half_BG1_Z192(dec, config, params, grdDim, blkDim, strm); break;
        //case 224: s = decode_ldpc2_shared_dynamic_half_BG1_Z224(dec, config, params, grdDim, blkDim, strm); break;
        //case 256: s = decode_ldpc2_shared_dynamic_half_BG1_Z256(dec, config, params, grdDim, blkDim, strm); break;
        //case 288: s = decode_ldpc2_shared_dynamic_half_BG1_Z288(dec, config, params, grdDim, blkDim, strm); break;
        //case 320: s = decode_ldpc2_shared_dynamic_half_BG1_Z320(dec, config, params, grdDim, blkDim, strm); break;
        //case 352: s = decode_ldpc2_shared_dynamic_half_BG1_Z352(dec, config, params, grdDim, blkDim, strm); break;
        case 384: s = decode_ldpc2_shared_dynamic_half_BG1_Z384(dec, config, params, grdDim, blkDim, strm); break;
        default:                                                                               break;
        }
    }
    return s;
}

} // namespace ldpc2

namespace ldpc
{

////////////////////////////////////////////////////////////////////////
// decode_ldpc2_shared_dynamic_index()
lwphyStatus_t decode_ldpc2_shared_dynamic_index(decoder&               dec,
                                                LDPC_output_t&         tDst,
                                                const_tensor_pair&     tLLR,
                                                const LDPC_config&     config,
                                                float                  normalization,
                                                lwphyLDPCResults_t*    results,
                                                void*                  workspace,
                                                lwphyLDPCDiagnostic_t* diag,
                                                lwdaStream_t           strm)
{
    DEBUG_PRINTF("ldpc::decode_ldpc2_shared_dynamic_index()\n");
    //------------------------------------------------------------------
    lwphyDataType_t llrType = tLLR.first.get().type();
    //------------------------------------------------------------------
    dim3 grdDim(config.num_codewords);
    dim3 blkDim(config.Z);

    //------------------------------------------------------------------
    // Initialize the kernel params struct
    LDPC_kernel_params params(config, tLLR, tDst, normalization, workspace);

    lwphyStatus_t s;
    
    if(llrType == LWPHY_R_32F)
    {
        return LWPHY_STATUS_NOT_SUPPORTED;
    }
    else if(llrType == LWPHY_R_16F)
    {
        // Colwert the normalization value to __half2
        params.norm.f16x2 = __float2half2_rn(params.norm.f32);
        s = decode_ldpc2_shared_dynamic_index_half(dec,
                                                   config,
                                                   params,
                                                   grdDim,
                                                   blkDim,
                                                   strm);
    }
    else
    {
        return LWPHY_STATUS_NOT_SUPPORTED;
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
// decode_ldpc2_shared_dynamic_index_workspace_size()
std::pair<bool, size_t> decode_ldpc2_shared_dynamic_index_workspace_size(const decoder&     dec,
                                                                         const LDPC_config& cfg)
{
    return std::pair<bool, size_t>(true, 0);
}

} // namespace ldpc

