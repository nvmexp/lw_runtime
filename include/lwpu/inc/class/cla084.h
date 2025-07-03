/*
* _LWRM_COPYRIGHT_BEGIN_
*
* Copyright 2021 by LWPU Corporation.  All rights reserved.  All
* information contained herein is proprietary and confidential to LWPU
* Corporation.  Any use, reproduction, or disclosure without the writte
* permission of LWPU Corporation is prohibited.
*
* _LWRM_COPYRIGHT_END_
*/


#ifndef _cla084_h_
#define _cla084_h_

#ifdef __cplusplus
extern "C" {
#endif

#define LWA084_HOST_VGPU_DEVICE_KERNEL                          (0x0000a084)

#define LWA084_MAX_VMMU_SEGMENTS                                384

/*
 * LWA084_ALLOC_PARAMETERS
 *
 * This structure represents vGPU host device KERNEL object allocation parameters.
 * dbdf -> domain (31:16), bus (15:8), device (7:3), function (2:0)
 * gfid -> Used only when SRIOV is enabled otherwise set to 0.
 * swizzId [IN/OUT] -> Used only when MIG mode is enabled otherwise set
 *                     to LW2080_CTRL_GPU_PARTITION_ID_ILWALID.
 * numChannels -> Used only when SRIOV is enabled. Must be a power of 2.
 * bDisableDefaultSmcExecPartRestore - If set to true, SMC default exelwtion partition
 *                                     save/restore will not be done in host-RM
 * vgpuDeviceInstanceId -> Specifies the vGPU device instance per VM to be used
 *                         for supporting multiple vGPUs per VM.
 * numGuestFbHandles -> number of guest memory handles
 * guestFbHandleList -> handle list to guest memory
 * hPluginHeapMemory -> plugin heap memory handle
 */
typedef struct
{
    LwU32 dbdf;
    LwU32 gfid;
    LwU32 swizzId;
    LwU32 vgpuType;
    LwU32 vmPid;
    LwU32 numChannels;
    LwU32 numPluginChannels;
    VM_ID_TYPE vmIdType;
    VM_ID guestVmId;
    LwBool bDisableDefaultSmcExecPartRestore;
    LwU32 vgpuDeviceInstanceId;
    LwU32 numGuestFbHandles;
    LwHandle guestFbHandleList[LWA084_MAX_VMMU_SEGMENTS];
    LwHandle hPluginHeapMemory;
    LwU64 ctrlBuffOffset;
} LWA084_ALLOC_PARAMETERS;

//
// @todo: We will define the actual event values later based on the use case.
// These event values are only for Test purpose.
//
/* event values */
#define LWA084_NOTIFIERS_EVENT_VGPU_PLUGIN_TASK_BOOTLOADED      (0)
#define LWA084_NOTIFIERS_EVENT_VGPU_PLUGIN_TASK_UNLOADED        (1)
#define LWA084_NOTIFIERS_EVENT_GUEST_RPC_VERSION_NEGOTIATED     (2)
#define LWA084_NOTIFIERS_MAXCOUNT                               (3)

#define LWA084_NOTIFICATION_STATUS_IN_PROGRESS              (0x8000)
#define LWA084_NOTIFICATION_STATUS_BAD_ARGUMENT             (0x4000)
#define LWA084_NOTIFICATION_STATUS_ERROR_ILWALID_STATE      (0x2000)
#define LWA084_NOTIFICATION_STATUS_ERROR_STATE_IN_USE       (0x1000)
#define LWA084_NOTIFICATION_STATUS_DONE_SUCCESS             (0x0000)

#ifdef __cplusplus
};     /* extern "C" */
#endif
#endif // _cla084_h
