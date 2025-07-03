/*
 * SPDX-FileCopyrightText: Copyright (c) 2006-2021 LWPU CORPORATION & AFFILIATES. All rights reserved.
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
// Source file: ctrl/ctrl0080/ctrl0080fifo.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)



#include "ctrl/ctrl0080/ctrl0080base.h"

/* LW01_DEVICE_XX/LW03_DEVICE fifo control commands and parameters */

/**
 * LW0080_CTRL_FIFO_GET_CAPS
 *
 * This command returns the set of FIFO engine capabilities for the device
 * in the form of an array of unsigned bytes.  FIFO capabilities
 * include supported features and required workarounds for the FIFO
 * engine(s) within the device, each represented by a byte offset into the
 * table and a bit position within that byte.
 *
 *   capsTblSize
 *     This parameter specifies the size in bytes of the caps table.
 *     This value should be set to LW0080_CTRL_FIFO_CAPS_TBL_SIZE.
 *   capsTbl
 *     This parameter specifies a pointer to the client's caps table buffer
 *     into which the framebuffer caps bits will be transferred by the RM.
 *     The caps table is an array of unsigned bytes.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW0080_CTRL_CMD_FIFO_GET_CAPS (0x801701) /* finn: Evaluated from "(FINN_LW01_DEVICE_0_FIFO_INTERFACE_ID << 8) | LW0080_CTRL_FIFO_GET_CAPS_PARAMS_MESSAGE_ID" */

#define LW0080_CTRL_FIFO_GET_CAPS_PARAMS_MESSAGE_ID (0x1U)

typedef struct LW0080_CTRL_FIFO_GET_CAPS_PARAMS {
    LwU32 capsTblSize;
    LW_DECLARE_ALIGNED(LwP64 capsTbl, 8);
} LW0080_CTRL_FIFO_GET_CAPS_PARAMS;

/* extract cap bit setting from tbl */
#define LW0080_CTRL_FIFO_GET_CAP(tbl,c)            (((LwU8)tbl[(1?c)]) & (0?c))

/* caps format is byte_index:bit_mask */
#define LW0080_CTRL_FIFO_CAPS_SUPPORT_SCHED_EVENT                    0:0x01
#define LW0080_CTRL_FIFO_CAPS_SUPPORT_PCI_PB                         0:0x02
#define LW0080_CTRL_FIFO_CAPS_SUPPORT_VID_PB                         0:0x04
#define LW0080_CTRL_FIFO_CAPS_USERD_IN_SYSMEM                        0:0x40
/* do not use pipelined PTE BLITs to update PTEs:  call the RM */
#define LW0080_CTRL_FIFO_CAPS_NO_PIPELINED_PTE_BLIT                  0:0x80
#define LW0080_CTRL_FIFO_CAPS_GPU_MAP_CHANNEL                        1:0x01
#define LW0080_CTRL_FIFO_CAPS_BUFFEREDMODE_SCHEDULING                1:0x02 // Deprecated
#define LW0080_CTRL_FIFO_CAPS_WFI_BUG_898467                         1:0x08 // Deprecated
#define LW0080_CTRL_FIFO_CAPS_HAS_HOST_LB_OVERFLOW_BUG_1667921       1:0x10
/*
 * To indicate Volta subcontext support with multiple VA spaces in a TSG.
 * We are not using "subcontext" tag for the property, since we also use
 * subcontext to represent pre-VOlta SCG feature, which only allows a single
 * VA space in a TSG.
 */
#define LW0080_CTRL_FIFO_CAPS_MULTI_VAS_PER_CHANGRP                  1:0x20
/*
 * Bug 2682961: Introducing the new CAP Bit for KMD to enable realtime runlist
 * This feature is enabled only when the regkey RmEnableWDDMInterleaving is set 
 */
#define LW0080_CTRL_FIFO_CAPS_SUPPORT_WDDM_INTERLEAVING              1:0x40

/* size in bytes of fifo caps table */
#define LW0080_CTRL_FIFO_CAPS_TBL_SIZE           2

/*
 * LW0080_CTRL_CMD_FIFO_ENABLE_SCHED_EVENTS
 *
 * This command enables the GPU to place various scheduling events in the
 * off chip event buffer (with optional interrupt) for those GPUs that support
 * it.
 *
 *   record
 *     This parameter specifies a mask of event types to record.
 *   interrupt
 *     This parameter specifies a mask of event types for which to interrupt
 *     the CPU when the event oclwrs.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW0080_CTRL_CMD_FIFO_ENABLE_SCHED_EVENTS (0x801703) /* finn: Evaluated from "(FINN_LW01_DEVICE_0_FIFO_INTERFACE_ID << 8) | 0x3" */

typedef struct LW0080_CTRL_FIFO_ENABLE_SCHED_EVENTS_PARAMS {
    LwU32 record;
    LwU32 interrupt;
} LW0080_CTRL_FIFO_ENABLE_SCHED_EVENTS_PARAMS;

#define LW0080_CTRL_FIFO_ENABLE_SCHED_EVENTS_START_CTX             0:0
#define LW0080_CTRL_FIFO_ENABLE_SCHED_EVENTS_START_CTX_DISABLE   (0x00000000)
#define LW0080_CTRL_FIFO_ENABLE_SCHED_EVENTS_START_CTX_ENABLE    (0x00000001)
#define LW0080_CTRL_FIFO_ENABLE_SCHED_EVENTS_END_CTX               1:1
#define LW0080_CTRL_FIFO_ENABLE_SCHED_EVENTS_END_CTX_DISABLE     (0x00000000)
#define LW0080_CTRL_FIFO_ENABLE_SCHED_EVENTS_END_CTX_ENABLE      (0x00000001)
#define LW0080_CTRL_FIFO_ENABLE_SCHED_EVENTS_NEW_RUNLIST           2:2
#define LW0080_CTRL_FIFO_ENABLE_SCHED_EVENTS_NEW_RUNLIST_DISABLE (0x00000000)
#define LW0080_CTRL_FIFO_ENABLE_SCHED_EVENTS_NEW_RUNLIST_ENABLE  (0x00000001)
#define LW0080_CTRL_FIFO_ENABLE_SCHED_EVENTS_SEM_ACQUIRE           3:3
#define LW0080_CTRL_FIFO_ENABLE_SCHED_EVENTS_SEM_ACQUIRE_DISABLE (0x00000000)
#define LW0080_CTRL_FIFO_ENABLE_SCHED_EVENTS_SEM_ACQUIRE_ENABLE  (0x00000001)
#define LW0080_CTRL_FIFO_ENABLE_SCHED_EVENTS_PAGE_FAULT            4:4
#define LW0080_CTRL_FIFO_ENABLE_SCHED_EVENTS_PAGE_FAULT_DISABLE  (0x00000000)
#define LW0080_CTRL_FIFO_ENABLE_SCHED_EVENTS_PAGE_FAULT_ENABLE   (0x00000001)
#define LW0080_CTRL_FIFO_ENABLE_SCHED_EVENTS_PREEMPT               5:5
#define LW0080_CTRL_FIFO_ENABLE_SCHED_EVENTS_PREEMPT_DISABLE     (0x00000000)
#define LW0080_CTRL_FIFO_ENABLE_SCHED_EVENTS_PREEMPT_ENABLE      (0x00000001)
#define LW0080_CTRL_FIFO_ENABLE_SCHED_EVENTS_YIELD                 6:6
#define LW0080_CTRL_FIFO_ENABLE_SCHED_EVENTS_YIELD_DISABLE       (0x00000000)
#define LW0080_CTRL_FIFO_ENABLE_SCHED_EVENTS_YIELD_ENABLE        (0x00000001)
#define LW0080_CTRL_FIFO_ENABLE_SCHED_EVENTS_IDLE_CTX              7:7
#define LW0080_CTRL_FIFO_ENABLE_SCHED_EVENTS_IDLE_CTX_DISABLE    (0x00000000)
#define LW0080_CTRL_FIFO_ENABLE_SCHED_EVENTS_IDLE_CTX_ENABLE     (0x00000001)
#define LW0080_CTRL_FIFO_ENABLE_SCHED_EVENTS_HI_PRI                8:8
#define LW0080_CTRL_FIFO_ENABLE_SCHED_EVENTS_HI_PRI_DISABLE      (0x00000000)
#define LW0080_CTRL_FIFO_ENABLE_SCHED_EVENTS_HI_PRI_ENABLE       (0x00000001)
#define LW0080_CTRL_FIFO_ENABLE_SCHED_EVENTS_ENG_STALLED           9:9
#define LW0080_CTRL_FIFO_ENABLE_SCHED_EVENTS_ENG_STALLED_DISABLE (0x00000000)
#define LW0080_CTRL_FIFO_ENABLE_SCHED_EVENTS_ENG_STALLED_ENABLE  (0x00000001)
#define LW0080_CTRL_FIFO_ENABLE_SCHED_EVENTS_VSYNC                 10:10
#define LW0080_CTRL_FIFO_ENABLE_SCHED_EVENTS_VSYNC_DISABLE       (0x00000000)
#define LW0080_CTRL_FIFO_ENABLE_SCHED_EVENTS_VSYNC_ENABLE        (0x00000001)
#define LW0080_CTRL_FIFO_ENABLE_SCHED_EVENTS_FGCS_FAULT            11:11
#define LW0080_CTRL_FIFO_ENABLE_SCHED_EVENTS_FGCS_FAULT_DISABLE  (0x00000000)
#define LW0080_CTRL_FIFO_ENABLE_SCHED_EVENTS_FGCS_FAULT_ENABLE   (0x00000001)
#define LW0080_CTRL_FIFO_ENABLE_SCHED_EVENTS_ALL                   11:0
#define LW0080_CTRL_FIFO_ENABLE_SCHED_EVENTS_ALL_DISABLE         (0x00000000)
#define LW0080_CTRL_FIFO_ENABLE_SCHED_EVENTS_ALL_ENABLE          (0x00000fff)

typedef struct LW0080_CTRL_FIFO_CHANNEL {
    LwHandle hChannel;
} LW0080_CTRL_FIFO_CHANNEL;

/*
 * LW0080_CTRL_CMD_FIFO_START_SELECTED_CHANNELS
 *
 * This command allows the caller to request that a set of channels
 * be removed from the runlist.
 *
 *   hChannel
 *     This is the handle to the channel that is scheduled to be stopped.
 *   fifoStartChannelListSize
 *     Size of the fifoStopChannelList.  The units are in entries, not
 *     bytes.
 *   fifoStartChannelList
 *     This will be a list of LW0080_CTRL_FIFO_CHANNEL data structures, 
 *     one for each channel that is to be stopped.
 *   channelHandle
 *     This array will be filled in with the handle of the channel that
 *     was last running in an engine prior to stopping the channel.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW0080_CTRL_CMD_FIFO_START_SELECTED_CHANNELS (0x801705) /* finn: Evaluated from "(FINN_LW01_DEVICE_0_FIFO_INTERFACE_ID << 8) | LW0080_CTRL_FIFO_START_SELECTED_CHANNELS_PARAMS_MESSAGE_ID" */

#define LW0080_CTRL_FIFO_START_SELECTED_CHANNELS_PARAMS_MESSAGE_ID (0x5U)

typedef struct LW0080_CTRL_FIFO_START_SELECTED_CHANNELS_PARAMS {
    LwU32    fifoStartChannelListSize;
    LwHandle channelHandle[8];
    LW_DECLARE_ALIGNED(LwP64 fifoStartChannelList, 8);
} LW0080_CTRL_FIFO_START_SELECTED_CHANNELS_PARAMS;

#define LW0080_CTRL_FIFO_ENGINE_ID_GRAPHICS                                               (0x00000000)
#define LW0080_CTRL_FIFO_ENGINE_ID_MPEG                                                   (0x00000001)
#define LW0080_CTRL_FIFO_ENGINE_ID_MOTION_ESTIMATION                                      (0x00000002)
#define LW0080_CTRL_FIFO_ENGINE_ID_VIDEO                                                  (0x00000003)
#define LW0080_CTRL_FIFO_ENGINE_ID_BITSTREAM                                              (0x00000004)
#define LW0080_CTRL_FIFO_ENGINE_ID_ENCRYPTION                                             (0x00000005)
#define LW0080_CTRL_FIFO_ENGINE_ID_FGT                                                    (0x00000006)

/*
 * LW0080_CTRL_CMD_FIFO_GET_ENGINE_CONTEXT_PROPERTIES
 *
 * This command is used to provide the caller with the alignment and size
 * of the context save region for an engine
 *
 *   engineId
 *     This parameter is an input parameter specifying the engineId for which
 *     the alignment/size is requested.
 *   alignment
 *     This parameter is an output parameter which will be filled in with the
 *     minimum alignment requirement.
 *   size
 *     This parameter is an output parameter which will be filled in with the
 *     minimum size of the context save region for the engine.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 */

#define LW0080_CTRL_CMD_FIFO_GET_ENGINE_CONTEXT_PROPERTIES                                (0x801707) /* finn: Evaluated from "(FINN_LW01_DEVICE_0_FIFO_INTERFACE_ID << 8) | LW0080_CTRL_FIFO_GET_ENGINE_CONTEXT_PROPERTIES_PARAMS_MESSAGE_ID" */

#define LW0080_CTRL_FIFO_GET_ENGINE_CONTEXT_PROPERTIES_ENGINE_ID                          4:0
#define LW0080_CTRL_FIFO_GET_ENGINE_CONTEXT_PROPERTIES_ENGINE_ID_GRAPHICS                 (0x00000000)
#define LW0080_CTRL_FIFO_GET_ENGINE_CONTEXT_PROPERTIES_ENGINE_ID_VLD                      (0x00000001)
#define LW0080_CTRL_FIFO_GET_ENGINE_CONTEXT_PROPERTIES_ENGINE_ID_VIDEO                    (0x00000002)
#define LW0080_CTRL_FIFO_GET_ENGINE_CONTEXT_PROPERTIES_ENGINE_ID_MPEG                     (0x00000003)
#define LW0080_CTRL_FIFO_GET_ENGINE_CONTEXT_PROPERTIES_ENGINE_ID_CAPTURE                  (0x00000004)
#define LW0080_CTRL_FIFO_GET_ENGINE_CONTEXT_PROPERTIES_ENGINE_ID_DISPLAY                  (0x00000005)
#define LW0080_CTRL_FIFO_GET_ENGINE_CONTEXT_PROPERTIES_ENGINE_ID_ENCRYPTION               (0x00000006)
#define LW0080_CTRL_FIFO_GET_ENGINE_CONTEXT_PROPERTIES_ENGINE_ID_POSTPROCESS              (0x00000007)
#define LW0080_CTRL_FIFO_GET_ENGINE_CONTEXT_PROPERTIES_ENGINE_ID_GRAPHICS_ZLWLL           (0x00000008)
#define LW0080_CTRL_FIFO_GET_ENGINE_CONTEXT_PROPERTIES_ENGINE_ID_GRAPHICS_PM              (0x00000009)
#define LW0080_CTRL_FIFO_GET_ENGINE_CONTEXT_PROPERTIES_ENGINE_ID_COMPUTE_PREEMPT          (0x0000000a)
#define LW0080_CTRL_FIFO_GET_ENGINE_CONTEXT_PROPERTIES_ENGINE_ID_GRAPHICS_PREEMPT         (0x0000000b)
#define LW0080_CTRL_FIFO_GET_ENGINE_CONTEXT_PROPERTIES_ENGINE_ID_GRAPHICS_SPILL           (0x0000000c)
#define LW0080_CTRL_FIFO_GET_ENGINE_CONTEXT_PROPERTIES_ENGINE_ID_GRAPHICS_PAGEPOOL        (0x0000000d)
#define LW0080_CTRL_FIFO_GET_ENGINE_CONTEXT_PROPERTIES_ENGINE_ID_GRAPHICS_BETACB          (0x0000000e)
#define LW0080_CTRL_FIFO_GET_ENGINE_CONTEXT_PROPERTIES_ENGINE_ID_GRAPHICS_RTV             (0x0000000f)
#define LW0080_CTRL_FIFO_GET_ENGINE_CONTEXT_PROPERTIES_ENGINE_ID_GRAPHICS_PATCH           (0x00000010)
#define LW0080_CTRL_FIFO_GET_ENGINE_CONTEXT_PROPERTIES_ENGINE_ID_GRAPHICS_BUNDLE_CB       (0x00000011)
#define LW0080_CTRL_FIFO_GET_ENGINE_CONTEXT_PROPERTIES_ENGINE_ID_GRAPHICS_PAGEPOOL_GLOBAL (0x00000012)
#define LW0080_CTRL_FIFO_GET_ENGINE_CONTEXT_PROPERTIES_ENGINE_ID_GRAPHICS_ATTRIBUTE_CB    (0x00000013)
#define LW0080_CTRL_FIFO_GET_ENGINE_CONTEXT_PROPERTIES_ENGINE_ID_GRAPHICS_RTV_CB_GLOBAL   (0x00000014)
#define LW0080_CTRL_FIFO_GET_ENGINE_CONTEXT_PROPERTIES_ENGINE_ID_GRAPHICS_GFXP_POOL       (0x00000015)
#define LW0080_CTRL_FIFO_GET_ENGINE_CONTEXT_PROPERTIES_ENGINE_ID_GRAPHICS_GFXP_CTRL_BLK   (0x00000016)
#define LW0080_CTRL_FIFO_GET_ENGINE_CONTEXT_PROPERTIES_ENGINE_ID_GRAPHICS_FECS_EVENT      (0x00000017)
#define LW0080_CTRL_FIFO_GET_ENGINE_CONTEXT_PROPERTIES_ENGINE_ID_GRAPHICS_PRIV_ACCESS_MAP (0x00000018)
#define LW0080_CTRL_FIFO_GET_ENGINE_CONTEXT_PROPERTIES_ENGINE_ID_COUNT                    (0x00000019)
#define LW0080_CTRL_FIFO_GET_ENGINE_CONTEXT_PROPERTIES_PARAMS_MESSAGE_ID (0x7U)

typedef struct LW0080_CTRL_FIFO_GET_ENGINE_CONTEXT_PROPERTIES_PARAMS {
    LwU32 engineId;
    LwU32 alignment;
    LwU32 size;
} LW0080_CTRL_FIFO_GET_ENGINE_CONTEXT_PROPERTIES_PARAMS;

/*
 * LW0080_CTRL_CMD_FIFO_RUNLIST_GROUP_CHANNELS      <Deprecated since Fermi+>
 *
 * This command allows the caller to group two sets of channels.  A channel 
 * set includes one or more channels.  After grouping, the grouped channel IDs 
 * are set to next to each other in the runlist.  This command can be used 
 * several times to group more than two channels. 
 *
 * Using a LW0080_CTRL_CMD_FIFO_RUNLIST_DIVIDE_TIMESLICE after 
 * LW0080_CTRL_CMD_FIFO_RUNLIST_GROUP_CHANNELS is the general usage.  A 
 * LW0080_CTRL_CMD_FIFO_RUNLIST_GROUP_CHANNELS after a 
 * LW0080_CTRL_CMD_FIFO_RUNLIST_DIVIDE_TIMESLICE for a channel handle is not 
 * allowed.
 *
 * LW0080_CTRL_FIFO_RUNLIST_GROUP_MAX_CHANNELS defines the max channels in a 
 * group.
 *
 *   hChannel1
 *     This parameter specifies the handle of the channel that belongs to the 
 *     base set of channels.
 *   hChannel2
 *     This parameter specifies the handle of the channel that belongs to the 
 *     additional set of channels.

 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 *   LW_ERR_ILWALID_DEVICE
 *   LW_ERR_ILWALID_CHANNEL
 *   LW_ERR_ILWALID_ARGUMENT
 */

#define LW0080_CTRL_CMD_FIFO_RUNLIST_GROUP_CHANNELS (0x801709) /* finn: Evaluated from "(FINN_LW01_DEVICE_0_FIFO_INTERFACE_ID << 8) | 0x9" */

typedef struct LW0080_CTRL_FIFO_RUNLIST_GROUP_CHANNELS_PARAM {
    LwHandle hChannel1;
    LwHandle hChannel2;
} LW0080_CTRL_FIFO_RUNLIST_GROUP_CHANNELS_PARAM;

#define LW0080_CTRL_FIFO_RUNLIST_GROUP_MAX_CHANNELS   (8)

/*
 * LW0080_CTRL_CMD_FIFO_RUNLIST_DIVIDE_TIMESLICE        <Deprecated since Fermi+>
 *
 * This command allows the caller to divide the timeslice (DMA_TIMESLICE) of a 
 * channel between the channels in the group in which the channel resides.  
 * After applying this command, a timeslice divided channel (group) has a
 * short timeslice and repeats more than once in the runlist.  The total
 * available exelwtion time is not changed.
 *
 * Using this command after LW0080_CTRL_CMD_FIFO_RUNLIST_GROUP_CHANNELS is the 
 * general usage.  A LW0080_CTRL_CMD_FIFO_RUNLIST_GROUP_CHANNELS after a 
 * LW0080_CTRL_CMD_FIFO_RUNLIST_DIVIDE_TIMESLICE for a channel handle is not 
 * allowed.
 *
 *   hChannel
 *     This parameter specifies the handle of the channel for the channel
 *     group to which the divided timeslice operation will apply.
 *   tsDivisor
 *     This parameter specifies the timeslice divisor value.  This value
 *     should not exceed LW0080_CTRL_FIFO_RUNLIST_MAX_TIMESLICE_DIVISOR
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 *   LW_ERR_ILWALID_DEVICE
 *   LW_ERR_ILWALID_CHANNEL
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_INSUFFICIENT_RESOURCES
 */
#define LW0080_CTRL_CMD_FIFO_RUNLIST_DIVIDE_TIMESLICE (0x80170b) /* finn: Evaluated from "(FINN_LW01_DEVICE_0_FIFO_INTERFACE_ID << 8) | 0xB" */

typedef struct LW0080_CTRL_FIFO_RUNLIST_DIVIDE_TIMESLICE_PARAM {
    LwHandle hChannel;
    LwU32    tsDivisor;
} LW0080_CTRL_FIFO_RUNLIST_DIVIDE_TIMESLICE_PARAM;

#define LW0080_CTRL_FIFO_RUNLIST_MAX_TIMESLICE_DIVISOR (12)

/*
 * LW0080_CTRL_CMD_FIFO_PREEMPT_RUNLIST                 <Deprecated since Fermi+>
 *
 * This command preepmts the engine represented by the specified runlist.
 * 
 *   hRunlist
 *     This parameter specifies the per engine runlist handle. This
 *     parameter is being retained to maintain backwards compatibility
 *     with clients that have not transitioned over to using runlists
 *     on a per subdevice basis.
 *
 *   engineID
 *     This parameter specifies the engine to be preempted. Engine defines
 *     can be found in cl2080.h. 
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_OBJECT_HANDLE
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW0080_CTRL_CMD_FIFO_PREEMPT_RUNLIST           (0x80170c) /* finn: Evaluated from "(FINN_LW01_DEVICE_0_FIFO_INTERFACE_ID << 8) | 0xC" */

typedef struct LW0080_CTRL_FIFO_PREEMPT_RUNLIST_PARAMS {
    LwHandle hRunlist;
    LwU32    engineID;
} LW0080_CTRL_FIFO_PREEMPT_RUNLIST_PARAMS;


/*
 * LW0080_CTRL_CMD_FIFO_GET_CHANNELLIST
 *
 * Takes a list of hChannels as input and returns the
 * corresponding Channel IDs that they corresponding to
 * on hw.
 * 
 *   numChannels
 *     Size of input hChannellist
 *   pChannelHandleList
 *     List of input channel handles
 *   pChannelList
 *     List of Channel ID's corresponding to the
 *     each entry in the hChannelList.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_OBJECT_HANDLE
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW0080_CTRL_CMD_FIFO_GET_CHANNELLIST (0x80170d) /* finn: Evaluated from "(FINN_LW01_DEVICE_0_FIFO_INTERFACE_ID << 8) | LW0080_CTRL_FIFO_GET_CHANNELLIST_PARAMS_MESSAGE_ID" */

#define LW0080_CTRL_FIFO_GET_CHANNELLIST_PARAMS_MESSAGE_ID (0xDU)

typedef struct LW0080_CTRL_FIFO_GET_CHANNELLIST_PARAMS {
    LwU32 numChannels;
    LW_DECLARE_ALIGNED(LwP64 pChannelHandleList, 8);
    LW_DECLARE_ALIGNED(LwP64 pChannelList, 8);
} LW0080_CTRL_FIFO_GET_CHANNELLIST_PARAMS;


/*
 * LW0080_CTRL_CMD_FIFO_GET_LATENCY_BUFFER_SIZE
 *
 *  This control call is used to return the number of gp methods(gpsize) and push buffer methods(pbsize)
 *  allocated to each engine.
 *
 *engineID
 *  The engine ID which is an input
 *
 *gpEntries
 *  number of gp entries
 *
 *pbEntries
 *  number of pb entries (in units of 32B rows)
 *
 */


#define LW0080_CTRL_CMD_FIFO_GET_LATENCY_BUFFER_SIZE (0x80170e) /* finn: Evaluated from "(FINN_LW01_DEVICE_0_FIFO_INTERFACE_ID << 8) | LW0080_CTRL_FIFO_GET_LATENCY_BUFFER_SIZE_PARAMS_MESSAGE_ID" */

#define LW0080_CTRL_FIFO_GET_LATENCY_BUFFER_SIZE_PARAMS_MESSAGE_ID (0xEU)

typedef struct LW0080_CTRL_FIFO_GET_LATENCY_BUFFER_SIZE_PARAMS {
    LwU32 engineID;
    LwU32 gpEntries;
    LwU32 pbEntries;
} LW0080_CTRL_FIFO_GET_LATENCY_BUFFER_SIZE_PARAMS;

#define LW0080_CTRL_FIFO_GET_CHANNELLIST_ILWALID_CHANNEL (0xffffffff)

/*
 * LW0080_CTRL_CMD_FIFO_SET_CHANNEL_PROPERTIES
 * 
 * This command allows internal properties of the channel
 * to be modified even when the channel is active. Most of these properties 
 * are not meant to be modified during normal runs hence have been
 * kept separate from channel alloc params. It is the
 * responsibility of the underlying hal routine to make
 * sure the channel properties are changed while the channel
 * is *NOT* in a transient state.
 * 
 *   hChannel
 *     The handle to the channel.
 *
 *   property
 *     The channel property to be modified.
 *     LW0080_CTRL_FIFO_SET_CHANNEL_PROPERTIES_xxx provides the entire list
 *     of properties.
 *
 *   value
 *     The new value for the property.
 *     When property = LW0080_CTRL_FIFO_SET_CHANNEL_PROPERTIES_ENGINETIMESLICEINMICROSECONDS
 *          value    = timeslice in microseconds
 *          desc:      Used to change a channel's engine timeslice in microseconds
 *
 *          property = LW0080_CTRL_FIFO_SET_CHANNEL_PROPERTIES_PBDMATIMESLICEINMICROSECONDS
 *          value    = timeslice in microseconds
 *          desc:      Used to change a channel's pbdma timeslice in microseconds
 *
 *          property = LW0080_CTRL_FIFO_SET_CHANNEL_PROPERTIES_ENGINETIMESLICEDISABLE
 *          value    is ignored
 *          desc:      Disables a channel from being timesliced out from an engine.
 *                     Other scheduling events like explicit yield, acquire failures will
 *                     switch out the channel though.
 *
 *          property = LW0080_CTRL_FIFO_SET_CHANNEL_PROPERTIES_PBDMATIMESLICEDISABLE
 *          value    is ignored
 *          desc:      Disables a channel from being timesliced out from its pbdma.
 *
 *          property = LW0080_CTRL_FIFO_SET_CHANNEL_PROPERTIES_ILWALIDATE_PDB_TARGET
 *          value    is ignored
 *          desc:      Override the channel's page directory pointer table with an
 *                     erroneous aperture value. (TODO: make test calls LW_VERIF_FEATURES
 *                     only)(VERIF ONLY)
 *                     
 *          property = LW0080_CTRL_FIFO_SET_CHANNEL_PROPERTIES_RESETENGINECONTEXT
 *          value    = engineID of engine that will have its context pointer reset.
 *                     engineID defines can be found in cl2080.h
 *                     (e.g., LW2080_ENGINE_TYPE_GRAPHICS)
 *          desc:      Override the channel's engine context pointer with a non existent
 *                     buffer forcing it to fault. (VERIF ONLY)
 *                     
 *          property = LW0080_CTRL_FIFO_SET_CHANNEL_PROPERTIES_RESETENGINECONTEXT_NOPREEMPT
 *          value    = engineID of engine that will have its context pointer reset.
 *                     engineID defines can be found in cl2080.h
 *                     (e.g., LW2080_ENGINE_TYPE_GRAPHICS)
 *          desc:      Override the channel's engine context pointer with a non existent
 *                     buffer forcing it to fault. However the channel will not be preempted
 *                     before having its channel state modified.(VERIF ONLY)
 *
 *          property = LW0080_CTRL_FIFO_SET_CHANNEL_PROPERTIES_NOOP
 *          value    is ignored
 *          desc:      does not change any channel state exercises a full channel preempt/
 *                     unbind/bind op. (VERIF ONLY)
 *          
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_CHANNEL
 *   LW_ERR_NOT_SUPPORTED
 *   LW_ERR_ILWALID_ARGUMENT
 *
 */
#define LW0080_CTRL_CMD_FIFO_SET_CHANNEL_PROPERTIES      (0x80170f) /* finn: Evaluated from "(FINN_LW01_DEVICE_0_FIFO_INTERFACE_ID << 8) | LW0080_CTRL_FIFO_SET_CHANNEL_PROPERTIES_PARAMS_MESSAGE_ID" */

#define LW0080_CTRL_FIFO_SET_CHANNEL_PROPERTIES_PARAMS_MESSAGE_ID (0xFU)

typedef struct LW0080_CTRL_FIFO_SET_CHANNEL_PROPERTIES_PARAMS {
    LwHandle hChannel;
    LwU32    property;
    LW_DECLARE_ALIGNED(LwU64 value, 8);
} LW0080_CTRL_FIFO_SET_CHANNEL_PROPERTIES_PARAMS;

#define LW0080_CTRL_FIFO_SET_CHANNEL_PROPERTIES_ENGINETIMESLICEINMICROSECONDS (0x00000000)
#define LW0080_CTRL_FIFO_SET_CHANNEL_PROPERTIES_PBDMATIMESLICEINMICROSECONDS  (0x00000001)
#define LW0080_CTRL_FIFO_SET_CHANNEL_PROPERTIES_ENGINETIMESLICEDISABLE        (0x00000002)
#define LW0080_CTRL_FIFO_SET_CHANNEL_PROPERTIES_PBDMATIMESLICEDISABLE         (0x00000003)
#define LW0080_CTRL_FIFO_SET_CHANNEL_PROPERTIES_ILWALIDATE_PDB_TARGET         (0x00000004)
#define LW0080_CTRL_FIFO_SET_CHANNEL_PROPERTIES_RESETENGINECONTEXT            (0x00000005)
#define LW0080_CTRL_FIFO_SET_CHANNEL_PROPERTIES_NOOP                          (0x00000007)
#define LW0080_CTRL_FIFO_SET_CHANNEL_PROPERTIES_RESETENGINECONTEXT_NOPREEMPT  (0x00000008)


/*
 * These structs are dummy and no longer supported.
 * Leaving it here for dependency checkin between KMD and RM
 * See ctrl2080fifo.h for the unicast versions
 */

#define LW0080_CTRL_FIFO_SWRUNLIST_MAX_TSGS                                   128
#define LW0080_CTRL_FIFO_SWRUNLIST_MAX_RUNLIST_ENTRIES                        1024

typedef struct LW0080_CTRL_FIFO_TSG_INFO {
    LwHandle hChannel;
    LwHandle hClient;
    LwU32    wfiTimeout;
} LW0080_CTRL_FIFO_TSG_INFO;

typedef struct LW0080_CTRL_FIFO_SWRUNLIST_SUBMIT_PARAMS {
    LwU32                     hRunlist;
    LwU32                     submitRunlistOffset;
    LwBool                    bUpdateRunlist;
    LwU32                     numTSGInfo;
    LW0080_CTRL_FIFO_TSG_INFO tsgInfo[LW0080_CTRL_FIFO_SWRUNLIST_MAX_TSGS];
    LwU8                      numRunlistTSGEntries;
    LwU8                      runlistOrderTsgIndex[LW0080_CTRL_FIFO_SWRUNLIST_MAX_RUNLIST_ENTRIES];
} LW0080_CTRL_FIFO_SWRUNLIST_SUBMIT_PARAMS;

/*
 * LW0080_CTRL_CMD_FIFO_STOP_RUNLIST
 *
 * Stops all processing on the runlist for the given engine.  This is only
 * valid in per-engine round-robin scheduling mode.
 * 
 *   engineID
 *     This parameter specifies the engine to be stopped. Engine defines
 *     can be found in cl2080.h. 
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_ILWALID_STATE
 */
#define LW0080_CTRL_CMD_FIFO_STOP_RUNLIST (0x801711) /* finn: Evaluated from "(FINN_LW01_DEVICE_0_FIFO_INTERFACE_ID << 8) | LW0080_CTRL_FIFO_STOP_RUNLIST_PARAMS_MESSAGE_ID" */

#define LW0080_CTRL_FIFO_STOP_RUNLIST_PARAMS_MESSAGE_ID (0x11U)

typedef struct LW0080_CTRL_FIFO_STOP_RUNLIST_PARAMS {
    LwU32 engineID;
} LW0080_CTRL_FIFO_STOP_RUNLIST_PARAMS;

/*
 * LW0080_CTRL_CMD_FIFO_START_RUNLIST
 *
 * Restarts a runlist previously stopped with LW0080_CTRL_CMD_FIFO_STOP_RUNLIST.
 * This is only valid for per-engine round-robin mode.
 * 
 *   engineID
 *     This parameter specifies the engine to be started. Engine defines
 *     can be found in cl2080.h. 
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_ILWALID_STATE
 */
#define LW0080_CTRL_CMD_FIFO_START_RUNLIST (0x801712) /* finn: Evaluated from "(FINN_LW01_DEVICE_0_FIFO_INTERFACE_ID << 8) | LW0080_CTRL_FIFO_START_RUNLIST_PARAMS_MESSAGE_ID" */

#define LW0080_CTRL_FIFO_START_RUNLIST_PARAMS_MESSAGE_ID (0x12U)

typedef struct LW0080_CTRL_FIFO_START_RUNLIST_PARAMS {
    LwU32 engineID;
} LW0080_CTRL_FIFO_START_RUNLIST_PARAMS;

/**
 * LW0080_CTRL_FIFO_GET_CAPS_V2
 *
 * This command returns the same set of FIFO engine capabilities for the device
 * as @ref LW0080_CTRL_FIFO_GET_CAPS. The difference is in the structure
 * LW0080_CTRL_FIFO_GET_CAPS_V2_PARAMS, which contains a statically sized array,
 * rather than a caps table pointer and a caps table size in
 * LW0080_CTRL_FIFO_GET_CAPS_PARAMS.
 *
 *   capsTbl
 *     This parameter is an array of the client's caps table buffer.
 *     The framebuffer caps bits will be written by the RM.
 *     The caps table is an array of unsigned bytes.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW0080_CTRL_CMD_FIFO_GET_CAPS_V2 (0x801713) /* finn: Evaluated from "(FINN_LW01_DEVICE_0_FIFO_INTERFACE_ID << 8) | LW0080_CTRL_FIFO_GET_CAPS_V2_PARAMS_MESSAGE_ID" */

#define LW0080_CTRL_FIFO_GET_CAPS_V2_PARAMS_MESSAGE_ID (0x13U)

typedef struct LW0080_CTRL_FIFO_GET_CAPS_V2_PARAMS {
    LwU8 capsTbl[LW0080_CTRL_FIFO_CAPS_TBL_SIZE];
} LW0080_CTRL_FIFO_GET_CAPS_V2_PARAMS;


/**
 * LW0080_CTRL_CMD_FIFO_IDLE_CHANNELS
 *
 * @brief This command idles (deschedules and waits for pending work to complete) channels
 *        belonging to a particular device. If the device passed is an SLI device, the
 *        channel idling is a broadcast operation.
 *
 *   numChannels
 *     Number of channels to idle
 *
 *   hChannels
 *     Array of channel handles to idle
 *
 *   flags
 *     LWOS30_FLAGS that control aspects of how the channel is idled
 *
 *   timeout
 *     GPU timeout in microseconds, for each CHID Manager's idling operation
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_OBJECT_NOT_FOUND
 *   LW_ERR_TIMEOUT
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_ILWALID_LOCK_STATE
 */
#define LW0080_CTRL_CMD_FIFO_IDLE_CHANNELS              (0x801714) /* finn: Evaluated from "(FINN_LW01_DEVICE_0_FIFO_INTERFACE_ID << 8) | LW0080_CTRL_FIFO_IDLE_CHANNELS_PARAMS_MESSAGE_ID" */
#define LW0080_CTRL_CMD_FIFO_IDLE_CHANNELS_MAX_CHANNELS 4096

#define LW0080_CTRL_FIFO_IDLE_CHANNELS_PARAMS_MESSAGE_ID (0x14U)

typedef struct LW0080_CTRL_FIFO_IDLE_CHANNELS_PARAMS {
    LwU32    numChannels;
    LwHandle hChannels[LW0080_CTRL_CMD_FIFO_IDLE_CHANNELS_MAX_CHANNELS];
    LwU32    flags;
    LwU32    timeout;
} LW0080_CTRL_FIFO_IDLE_CHANNELS_PARAMS;

/* LWRM_PUBLISHED_PENDING_IP_REVIEW */


/* _ctrl0080fifo_h_ */
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)

