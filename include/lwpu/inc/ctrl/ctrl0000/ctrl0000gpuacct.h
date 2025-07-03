/*
 * SPDX-FileCopyrightText: Copyright (c) 2013-2015 LWPU CORPORATION & AFFILIATES. All rights reserved.
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
#pragma once

#include <lwtypes.h>
#if defined(_MSC_VER)
#pragma warning(disable:4324)
#endif

//
// This file was generated with FINN, an LWPU coding tool.
// Source file: ctrl/ctrl0000/ctrl0000gpuacct.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)



#include "ctrl/ctrl0000/ctrl0000base.h"

/*
 * LW0000_CTRL_CMD_GPUACCT_SET_ACCOUNTING_STATE
 *
 * This command is used to enable or disable the per process GPU accounting.
 * This is part of GPU's software state and will persist if persistent
 * software state is enabled. Refer to the description of
 * LW0080_CTRL_CMD_GPU_MODIFY_SW_STATE_PERSISTENCE for more information.
 *  
 *   gpuId
 *     This parameter should specify a valid GPU ID value. Refer to the
 *     description of LW0000_CTRL_CMD_GPU_GET_ATTACHED_IDS for more
 *     information. If there is no GPU present with the specified ID,
 *     a status of LW_ERR_ILWALID_ARGUMENT is returned.
 *   pid
 *     This input parameter specifies the process id of the process for which
 *     the accounting state needs to be set. 
 *     In case of VGX host, this parameter specifies VGPU plugin(VM) pid. This
 *     parameter is set only when this RM control is called from VGPU plugin, 
 *     otherwise it is zero meaning set/reset the accounting state for the
 *     specified GPU.
 *  newState
 *    This input parameter is used to enable or disable the GPU accounting.
 *    Possible values are:
 *      LW0000_CTRL_GPU_ACCOUNTING_STATE_ENABLED
 *      LW0000_CTRL_GPU_ACCOUNTING_STATE_DISABLED
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 */

#define LW0000_CTRL_CMD_GPUACCT_SET_ACCOUNTING_STATE (0xb01) /* finn: Evaluated from "(FINN_LW01_ROOT_GPUACCT_INTERFACE_ID << 8) | LW0000_CTRL_GPUACCT_SET_ACCOUNTING_STATE_PARAMS_MESSAGE_ID" */

/* Possible values of persistentSwState */
#define LW0000_CTRL_GPU_ACCOUNTING_STATE_ENABLED     (0x00000000)
#define LW0000_CTRL_GPU_ACCOUNTING_STATE_DISABLED    (0x00000001)

#define LW0000_CTRL_GPUACCT_SET_ACCOUNTING_STATE_PARAMS_MESSAGE_ID (0x1U)

typedef struct LW0000_CTRL_GPUACCT_SET_ACCOUNTING_STATE_PARAMS {
    LwU32 gpuId;
    LwU32 pid;
    LwU32 newState;
} LW0000_CTRL_GPUACCT_SET_ACCOUNTING_STATE_PARAMS;

/*
 * LW0000_CTRL_CMD_GPUACCT_GET_ACCOUNTING_STATE
 *
 * This command is used to get the current state of GPU accounting.
 *
 *   gpuId
 *     This parameter should specify a valid GPU ID value. Refer to the
 *     description of LW0000_CTRL_CMD_GPU_GET_ATTACHED_IDS for more
 *     information. If there is no GPU present with the specified ID,
 *     a status of LW_ERR_ILWALID_ARGUMENT is returned.
 *   pid
 *     This input parameter specifies the process id of the process of which the
 *     accounting state needs to be queried. 
 *     In case of VGX host, this parameter specifies VGPU plugin(VM) pid. This
 *     parameter is set only when this RM control is called from VGPU plugin, 
 *     otherwise it is zero meaning the accounting state needs to be queried for
 *     the specified GPU.
 *   state
 *     This parameter returns a value indicating if per process GPU accounting
 *     is lwrrently enabled or not for the specified GPU. See the 
 *     description of LW0000_CTRL_CMD_GPU_SET_ACCOUNTING_STATE.
 *     Possible values are:
 *       LW0000_CTRL_GPU_ACCOUNTING_STATE_ENABLED
 *       LW0000_CTRL_GPU_ACCOUNTING_STATE_DISABLED
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 */

#define LW0000_CTRL_CMD_GPUACCT_GET_ACCOUNTING_STATE (0xb02) /* finn: Evaluated from "(FINN_LW01_ROOT_GPUACCT_INTERFACE_ID << 8) | LW0000_CTRL_GPUACCT_GET_ACCOUNTING_STATE_PARAMS_MESSAGE_ID" */

#define LW0000_CTRL_GPUACCT_GET_ACCOUNTING_STATE_PARAMS_MESSAGE_ID (0x2U)

typedef struct LW0000_CTRL_GPUACCT_GET_ACCOUNTING_STATE_PARAMS {
    LwU32 gpuId;
    LwU32 pid;
    LwU32 state;
} LW0000_CTRL_GPUACCT_GET_ACCOUNTING_STATE_PARAMS;

/*
 * LW0000_CTRL_GPUACCT_GET_PROC_ACCOUNTING_INFO_PARAMS
 *
 * This command returns GPU accounting data for the process.
 *
 *   gpuId
 *     This parameter should specify a valid GPU ID value. Refer to the
 *     description of LW0000_CTRL_CMD_GPU_GET_ATTACHED_IDS for more
 *     information. If there is no GPU present with the specified ID,
 *     a status of LW_ERR_ILWALID_ARGUMENT is returned.
 *   pid
 *     This parameter specifies the PID of the process for which information is
 *     to be queried.
 *     In case of VGX host, this parameter specifies VGPU plugin(VM) pid inside 
 *     which the subPid is running. This parameter is set to VGPU plugin pid 
 *     when this RM control is called from VGPU plugin. 
 *   subPid
 *     In case of VGX host, this parameter specifies the PID of the process for
 *     which information is to be queried. In other cases, it is zero.
 *   gpuUtil
 *     This parameter returns the average GR utilization during the process's
 *     lifetime.
 *   fbUtil
 *     This parameter returns the average FB bandwidth utilization during the
 *     process's lifetime.
 *   maxFbUsage
 *     This parameter returns the maximum FB allocated (in bytes) by the process.
 *   startTime
 *     This parameter returns the time stamp value in micro seconds at the time
 *     process started utilizing GPU.
 *   stopTime
 *     This parameter returns the time stamp value in micro seconds at the time
 *     process stopped utilizing GPU.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW0000_CTRL_CMD_GPUACCT_GET_PROC_ACCOUNTING_INFO (0xb03) /* finn: Evaluated from "(FINN_LW01_ROOT_GPUACCT_INTERFACE_ID << 8) | LW0000_CTRL_GPUACCT_GET_PROC_ACCOUNTING_INFO_PARAMS_MESSAGE_ID" */

#define LW0000_CTRL_GPUACCT_GET_PROC_ACCOUNTING_INFO_PARAMS_MESSAGE_ID (0x3U)

typedef struct LW0000_CTRL_GPUACCT_GET_PROC_ACCOUNTING_INFO_PARAMS {
    LwU32 gpuId;
    LwU32 pid;
    LwU32 subPid;
    LwU32 gpuUtil;
    LwU32 fbUtil;
    LW_DECLARE_ALIGNED(LwU64 maxFbUsage, 8);
    LW_DECLARE_ALIGNED(LwU64 startTime, 8);
    LW_DECLARE_ALIGNED(LwU64 endTime, 8);
} LW0000_CTRL_GPUACCT_GET_PROC_ACCOUNTING_INFO_PARAMS;

/*
 * LW0000_CTRL_CMD_GPUACCT_GET_ACCOUNTING_PIDS
 *
 * This command is used to get the PIDS of processes with accounting
 * information in the driver.
 *
 *   gpuId
 *     This parameter should specify a valid GPU ID value. Refer to the
 *     description of LW0000_CTRL_CMD_GPU_GET_ATTACHED_IDS for more
 *     information. If there is no GPU present with the specified ID,
 *     a status of LW_ERR_ILWALID_ARGUMENT is returned.
 *   pid
 *     This input parameter specifies the process id of the process of which the 
 *     information needs to be queried. 
 *     In case of VGX host, this parameter specifies VGPU plugin(VM) pid. This 
 *     parameter is set only when this RM control is called from VGPU plugin, 
 *     otherwise it is zero meaning get the pid list of the all the processes 
 *     running on the specified GPU.
 *    pidTbl
 *      This parameter returns the table of all PIDs for which driver has
 *      accounting info.
 *    pidCount
 *      This parameter returns the number of entries in the PID table.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 */

#define LW0000_CTRL_CMD_GPUACCT_GET_ACCOUNTING_PIDS (0xb04) /* finn: Evaluated from "(FINN_LW01_ROOT_GPUACCT_INTERFACE_ID << 8) | LW0000_CTRL_GPUACCT_GET_ACCOUNTING_PIDS_PARAMS_MESSAGE_ID" */

/* max size of pidTable */
#define LW0000_GPUACCT_PID_MAX_COUNT                4000

#define LW0000_CTRL_GPUACCT_GET_ACCOUNTING_PIDS_PARAMS_MESSAGE_ID (0x4U)

typedef struct LW0000_CTRL_GPUACCT_GET_ACCOUNTING_PIDS_PARAMS {
    LwU32 gpuId;
    LwU32 pid;
    LwU32 pidTbl[LW0000_GPUACCT_PID_MAX_COUNT];
    LwU32 pidCount;
} LW0000_CTRL_GPUACCT_GET_ACCOUNTING_PIDS_PARAMS;

/*
 * LW0000_CTRL_CMD_GPUACCT_CLEAR_ACCOUNTING_DATA
 *
 * This command is used to clear previously collected GPU accounting data. This
 * will have no affect on data for the running processes, accounting data for
 * these processes will not be cleared and will still be logged for these
 * processes. In order to clear ALL accounting data, accounting needs to be
 * disabled using LW0000_CTRL_CMD_GPUACCT_SET_ACCOUNTING_STATE before exelwting
 * this command.
 *  
 *   gpuId
 *     This parameter should specify a valid GPU ID value. Refer to the
 *     description of LW0000_CTRL_CMD_GPU_GET_ATTACHED_IDS for more
 *     information. If there is no GPU present with the specified ID,
 *     a status of LW_ERR_ILWALID_ARGUMENT is returned.
 *   pid
 *     This input parameter specifies the process id of the process for which 
 *     the accounting data needs to be cleared.
 *     In case of VGX host, this parameter specifies VGPU plugin(VM) pid for
 *     which the accounting data needs to be cleared. This parameter is set only 
 *     when this RM control is called from VGPU plugin, otherwise it is zero 
 *     meaning clear the accounting data of processes running on baremetal 
 *     system.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_ILWALID_STATE
 */

#define LW0000_CTRL_CMD_GPUACCT_CLEAR_ACCOUNTING_DATA (0xb05) /* finn: Evaluated from "(FINN_LW01_ROOT_GPUACCT_INTERFACE_ID << 8) | LW0000_CTRL_GPUACCT_CLEAR_ACCOUNTING_DATA_PARAMS_MESSAGE_ID" */

#define LW0000_CTRL_GPUACCT_CLEAR_ACCOUNTING_DATA_PARAMS_MESSAGE_ID (0x5U)

typedef struct LW0000_CTRL_GPUACCT_CLEAR_ACCOUNTING_DATA_PARAMS {
    LwU32 gpuId;
    LwU32 pid;
} LW0000_CTRL_GPUACCT_CLEAR_ACCOUNTING_DATA_PARAMS;


#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)

