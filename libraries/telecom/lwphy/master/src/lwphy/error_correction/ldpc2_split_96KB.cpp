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

namespace ldpc2
{

////////////////////////////////////////////////////////////////////////
// decode_ldpc2_split_half_96KB()
lwphyStatus_t decode_ldpc2_split_half_96KB(const LDPC_config&        cfg,
                                           const LDPC_kernel_params& params,
                                           const dim3&               grdDim,
                                           const dim3&               blkDim,
                                           lwdaStream_t              strm)
{
    lwphyStatus_t s = LWPHY_STATUS_NOT_SUPPORTED;
    if(cfg.BG == 1)
    {
        // Shared memory requirements, assuming 2 bytes APP, 8 bytes cC2V:
        // SHMEM = Z * [(Kb + mb) * sizeof(APP) + mb * sizeof(cC2V)]
        // If Z = 128:
        // SHMEM(Z = 128) = 128*[(22 + 46)*2 + 46*8] = 63 * 1024
        // With a minimum of 64 KB shared memory, all APP and cC2V data
        // should fit in shared memory for Z <= 128.
        switch(cfg.Z)
        {
        //case 64:  s = decode_ldpc2_split_half_96KB_BG1_Z64 (cfg, params, grdDim, blkDim, strm); break;
        //case 96:  s = decode_ldpc2_split_half_96KB_BG1_Z96 (cfg, params, grdDim, blkDim, strm); break;
        //case 128: s = decode_ldpc2_split_half_96KB_BG1_Z128(cfg, params, grdDim, blkDim, strm); break;
        case 160: s = decode_ldpc2_split_half_96KB_BG1_Z160(cfg, params, grdDim, blkDim, strm); break;
        case 192: s = decode_ldpc2_split_half_96KB_BG1_Z192(cfg, params, grdDim, blkDim, strm); break;
        case 224: s = decode_ldpc2_split_half_96KB_BG1_Z224(cfg, params, grdDim, blkDim, strm); break;
        case 256: s = decode_ldpc2_split_half_96KB_BG1_Z256(cfg, params, grdDim, blkDim, strm); break;
        case 288: s = decode_ldpc2_split_half_96KB_BG1_Z288(cfg, params, grdDim, blkDim, strm); break;
        case 320: s = decode_ldpc2_split_half_96KB_BG1_Z320(cfg, params, grdDim, blkDim, strm); break;
        case 352: s = decode_ldpc2_split_half_96KB_BG1_Z352(cfg, params, grdDim, blkDim, strm); break;
        case 384: s = decode_ldpc2_split_half_96KB_BG1_Z384(cfg, params, grdDim, blkDim, strm); break;
        default:                                                                                break;
        }
    }
    else if(cfg.BG == 2)
    {
        switch(cfg.Z)
        {
        case 64:  s = decode_ldpc2_split_half_96KB_BG2_Z64 (cfg, params, grdDim, blkDim, strm); break;
        case 96:  s = decode_ldpc2_split_half_96KB_BG2_Z96 (cfg, params, grdDim, blkDim, strm); break;
        case 128: s = decode_ldpc2_split_half_96KB_BG2_Z128(cfg, params, grdDim, blkDim, strm); break;
        case 160: s = decode_ldpc2_split_half_96KB_BG2_Z160(cfg, params, grdDim, blkDim, strm); break;
        case 192: s = decode_ldpc2_split_half_96KB_BG2_Z192(cfg, params, grdDim, blkDim, strm); break;
        case 224: s = decode_ldpc2_split_half_96KB_BG2_Z224(cfg, params, grdDim, blkDim, strm); break;
        case 256: s = decode_ldpc2_split_half_96KB_BG2_Z256(cfg, params, grdDim, blkDim, strm); break;
        case 288: s = decode_ldpc2_split_half_96KB_BG2_Z288(cfg, params, grdDim, blkDim, strm); break;
        case 320: s = decode_ldpc2_split_half_96KB_BG2_Z320(cfg, params, grdDim, blkDim, strm); break;
        case 352: s = decode_ldpc2_split_half_96KB_BG2_Z352(cfg, params, grdDim, blkDim, strm); break;
        case 384: s = decode_ldpc2_split_half_96KB_BG2_Z384(cfg, params, grdDim, blkDim, strm); break;
        default:                                                                                break;
        }
    }
    return s;
}

} // namespace ldpc2
