/*
 * SPDX-FileCopyrightText: Copyright (c) 1999-2018 LWPU CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#ifndef _LW_HYPERVISOR_H_
#define _LW_HYPERVISOR_H_

#include <lw-kernel-interface-api.h>

// Enums for supported hypervisor types.
// New hypervisor type should be added before OS_HYPERVISOR_LWSTOM_FORCED
typedef enum _HYPERVISOR_TYPE
{
    OS_HYPERVISOR_XEN = 0,
    OS_HYPERVISOR_VMWARE,
    OS_HYPERVISOR_HYPERV,
    OS_HYPERVISOR_KVM,
    OS_HYPERVISOR_PARALLELS,
    OS_HYPERVISOR_LWSTOM_FORCED,
    OS_HYPERVISOR_UNKNOWN
} HYPERVISOR_TYPE;

#define CMD_VGPU_VFIO_WAKE_WAIT_QUEUE         0
#define CMD_VGPU_VFIO_INJECT_INTERRUPT        1
#define CMD_VGPU_VFIO_REGISTER_MDEV           2
#define CMD_VGPU_VFIO_PRESENT                 3

#define MAX_VF_COUNT_PER_GPU 64

typedef enum _VGPU_TYPE_INFO
{
    VGPU_TYPE_NAME = 0,
    VGPU_TYPE_DESCRIPTION,
    VGPU_TYPE_INSTANCES,
} VGPU_TYPE_INFO;

typedef struct
{
    void  *vgpuVfioRef;
    void  *waitQueue;
    void  *lw;
    LwU32 *vgpuTypeIds;
    LwU32  numVgpuTypes;
    LwU32  domain;
    LwU8   bus;
    LwU8   slot;
    LwU8   function;
    LwBool is_virtfn;
} vgpu_vfio_info;

typedef struct
{
    LwU32       domain;
    LwU8        bus;
    LwU8        slot;
    LwU8        function;
    LwBool      isLwidiaAttached;
    LwBool      isMdevAttached;
} vgpu_vf_pci_info;

typedef enum VGPU_CMD_PROCESS_VF_INFO_E
{
    LW_VGPU_SAVE_VF_INFO         = 0,
    LW_VGPU_REMOVE_VF_PCI_INFO   = 1,
    LW_VGPU_REMOVE_VF_MDEV_INFO  = 2,
    LW_VGPU_GET_VF_INFO          = 3
} VGPU_CMD_PROCESS_VF_INFO;

typedef enum VGPU_DEVICE_STATE_E
{
    LW_VGPU_DEV_UNUSED = 0,
    LW_VGPU_DEV_OPENED = 1,
    LW_VGPU_DEV_IN_USE = 2
} VGPU_DEVICE_STATE;

typedef enum _VMBUS_CMD_TYPE
{
    VMBUS_CMD_TYPE_ILWALID    = 0,
    VMBUS_CMD_TYPE_SETUP      = 1,
    VMBUS_CMD_TYPE_SENDPACKET = 2,
    VMBUS_CMD_TYPE_CLEANUP    = 3,
} VMBUS_CMD_TYPE;

typedef struct
{
    LwU32 request_id;
    LwU32 page_count;
    LwU64 *pPfns;
    void *buffer;
    LwU32 bufferlen;
} vmbus_send_packet_cmd_params;


typedef struct
{
    LwU32 override_sint;
    LwU8 *lw_guid;
} vmbus_setup_cmd_params;

/*
 * Function prototypes
 */

HYPERVISOR_TYPE LW_API_CALL lw_get_hypervisor_type(void);

#endif // _LW_HYPERVISOR_H_
