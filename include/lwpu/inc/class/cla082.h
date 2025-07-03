/*
* _LWRM_COPYRIGHT_BEGIN_
*
* Copyright 2017-2021 by LWPU Corporation.  All rights reserved.  All
* information contained herein is proprietary and confidential to LWPU
* Corporation.  Any use, reproduction, or disclosure without the writte
* permission of LWPU Corporation is prohibited.
*
* _LWRM_COPYRIGHT_END_
*/


#ifndef _cla082_h_
#define _cla082_h_

#ifdef __cplusplus
extern "C" {
#endif

#include "lw_vgpu_types.h"

#define LWA082_HOST_VGPU_DEVICE                                 (0x0000a082)

/*
 * LWA082_ALLOC_PARAMETERS
 *
 * This structure represents vGPU host device object allocation parameters.
 * gfid -> Used only when SRIOV is enabled otherwise set to 0.
 * swizzId -> Used only when MIG mode is enabled otherwise set
 *            to LW2080_CTRL_GPU_PARTITION_ID_ILWALID.
 * numChannels -> Used only when SRIOV is enabled. Must be a power of 2.
 * bDisableDefaultSmcExecPartRestore - If set to true, SMC default exelwtion partition
 *                                     save/restore will not be done in host-RM
 * vgpuDeviceInstanceId -> Specifies the vGPU device instance per VM to be used
 *                         for supporting multiple vGPUs per VM.
 */
typedef struct
{
    LwU32        vgpuType;
    LwU32        vmPid;
    LwU32        gfid;
    LwU32        swizzId;
    LwU32        numChannels;
    LwU32        numPluginChannels;
    VM_ID_TYPE   vmIdType;
    VM_ID        guestVmId;
    LwBool       bDisableDefaultSmcExecPartRestore;
    LwU32        vgpuDeviceInstanceId;
} LWA082_ALLOC_PARAMETERS;

#ifdef __cplusplus
};     /* extern "C" */
#endif
#endif // _cla082_h
