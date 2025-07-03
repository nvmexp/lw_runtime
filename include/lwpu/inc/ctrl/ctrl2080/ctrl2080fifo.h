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
// Source file: ctrl/ctrl2080/ctrl2080fifo.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)



#include "ctrl/ctrl2080/ctrl2080base.h"

/*
 * LW2080_CTRL_CMD_SET_GPFIFO
 *
 * This command set the GPFIFO offset and number of entries for a channel
 * after it has been allocated. The channel must be idle and not pending,
 * otherwise ERROR_IN_USE will be returned.
 *
 *   hChannel
 *     The handle to the channel.
 *   base
 *     The base of the GPFIFO in the channel ctxdma.
 *   numEntries
 *     The number of entries in the GPFIFO.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_DEVICE
 *   LW_ERR_ILWALID_CHANNEL
 *   LW_ERR_STATE_IN_USE
 *   LW_ERR_ILWALID_ARGUMENT
 *
 */
#define LW2080_CTRL_CMD_SET_GPFIFO (0x20801102) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_FIFO_INTERFACE_ID << 8) | LW2080_CTRL_CMD_SET_GPFIFO_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_CMD_SET_GPFIFO_PARAMS_MESSAGE_ID (0x2U)

typedef struct LW2080_CTRL_CMD_SET_GPFIFO_PARAMS {
    LwHandle hChannel;
    LW_DECLARE_ALIGNED(LwU64 base, 8);
    LwU32    numEntries;
} LW2080_CTRL_CMD_SET_GPFIFO_PARAMS;

/*
 * LW2080_CTRL_FIFO_BIND_CHANNEL
 *
 * This structure is used to describe a channel that is to have
 * it's bindable engines bound to those of other channels.
 *
 * hClient
 *  This structure member contains the handle of the client object
 *  that owns the channel object specified by hChannel.
 *
 * hChannel
 *  This structure member contains the channel handle of the channel
 *  object.
 */

typedef struct LW2080_CTRL_FIFO_BIND_CHANNEL {
    LwHandle hClient;
    LwHandle hChannel;
} LW2080_CTRL_FIFO_BIND_CHANNEL;

/*
 * LW2080_CTRL_CMD_FIFO_BIND_ENGINES
 *
 * This command can be used to bind different video engines on G8X from separate
 * channels together for operations such as idling.  The set of bindable engines
 * includes the LW2080_ENGINE_TYPE_BSP, LW2080_ENGINE_TYPE_VP and
 * LW2080_ENGINE_TYPE_PPP engines.
 *
 * bindChannelCount
 *  This parameter specifies the number of channels to bind together.  This
 *  parameter cannot exceed LW2080_CTRL_FIFO_BIND_ENGINES_MAX_CHANNELS.
 *
 * bindChannels
 *  The parameter specifies the array of channels to bind together.  The first
 *  bindChannelCount entries are used in the bind channel operation.
 *
 * Possible status values returned are:
 *  LW_OK
 *  LW_ERR_ILWALID_DEVICE
 *  LW_ERR_ILWALID_CHANNEL
 *  LW_ERR_ILWALID_ARGUMENT
 *  LW_ERR_NOT_SUPPORTED
 */
#define LW2080_CTRL_FIFO_BIND_ENGINES_MAX_CHANNELS (16)

#define LW2080_CTRL_FIFO_BIND_ENGINES_PARAMS_MESSAGE_ID (0x3U)

typedef struct LW2080_CTRL_FIFO_BIND_ENGINES_PARAMS {
    LwU32                         bindChannelCount;
    LW2080_CTRL_FIFO_BIND_CHANNEL bindChannels[LW2080_CTRL_FIFO_BIND_ENGINES_MAX_CHANNELS];
} LW2080_CTRL_FIFO_BIND_ENGINES_PARAMS;

#define LW2080_CTRL_CMD_FIFO_BIND_ENGINES          (0x20801103) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_FIFO_INTERFACE_ID << 8) | LW2080_CTRL_FIFO_BIND_ENGINES_PARAMS_MESSAGE_ID" */

/*
 * LW2080_CTRL_CMD_SET_OPERATIONAL_PROPERTIES
 *
 * This command is used for a client to setup specialized custom operational
 * properties that may be specific to an environment, or properties that
 * should be set generally but are not for reasons of backward compatibility
 * with previous chip generations
 *
 *  flags
 *   This field specifies the operational properties to be applied
 *
 * Possible return status values returned are
 *   LW_OK
 *   LW_ERR_ILWALID_CHANNEL
 *   LW_ERR_ILWALID_ARGUMENT
 *
 */
#define LW2080_CTRL_CMD_SET_OPERATIONAL_PROPERTIES (0x20801104) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_FIFO_INTERFACE_ID << 8) | LW2080_CTRL_CMD_SET_OPERATIONAL_PROPERTIES_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_CMD_SET_OPERATIONAL_PROPERTIES_PARAMS_MESSAGE_ID (0x4U)

typedef struct LW2080_CTRL_CMD_SET_OPERATIONAL_PROPERTIES_PARAMS {
    LwU32 flags;
} LW2080_CTRL_CMD_SET_OPERATIONAL_PROPERTIES_PARAMS;

#define LW2080_CTRL_CMD_SET_OPERATIONAL_PROPERTIES_FLAGS_ERROR_ON_STUCK_SEMAPHORE                 0:0
#define LW2080_CTRL_CMD_SET_OPERATIONAL_PROPERTIES_FLAGS_ERROR_ON_STUCK_SEMAPHORE_FALSE (0x00000000)
#define LW2080_CTRL_CMD_SET_OPERATIONAL_PROPERTIES_FLAGS_ERROR_ON_STUCK_SEMAPHORE_TRUE  (0x00000001)

/*
 * LW2080_CTRL_CMD_FIFO_VC_GET_PHYS_CHANNEL_COUNT
 *
 * This command returns the number of physical channel slots reserved for
 * use by virtual channels.
 *
 *   physChannelCount:
 *     This field returns the count of physical channel slots reserved for
 *     virtual channels.
 *
 * Possible return status values returned are
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 *   LW_ERR_ILWALID_ARGUMENT
 *
 */

#define LW2080_CTRL_CMD_FIFO_VC_GET_PHYS_CHANNEL_COUNT                                  (0x20801105) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_FIFO_INTERFACE_ID << 8) | 0x5" */

typedef struct LW2080_CTRL_FIFO_VC_GET_PHYS_CHANNEL_COUNT_PARAMS {
    LwU32 physChannelCount;
} LW2080_CTRL_FIFO_VC_GET_PHYS_CHANNEL_COUNT_PARAMS;

/*
 * LW2080_CTRL_CMD_FIFO_GET_PHYSICAL_CHANNEL_COUNT
 *
 * This command returns the maximum number of physical channels available for
 * allocation on the current GPU.  This may be less than or equal to the total
 * number of channels supported by the current hardware.
 *
 * physChannelCount
 *   This output parameter contains the maximum physical channel count.
 *
 *   physChannelCountInUse
 *     This output parameter contains the number of physical channels in use
 *
 * Possible return status values returned are
 *   LW_OK
 *
 */
#define LW2080_CTRL_CMD_FIFO_GET_PHYSICAL_CHANNEL_COUNT (0x20801108) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_FIFO_INTERFACE_ID << 8) | LW2080_CTRL_FIFO_GET_PHYSICAL_CHANNEL_COUNT_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_FIFO_GET_PHYSICAL_CHANNEL_COUNT_PARAMS_MESSAGE_ID (0x8U)

typedef struct LW2080_CTRL_FIFO_GET_PHYSICAL_CHANNEL_COUNT_PARAMS {
    LwU32 physChannelCount;
    LwU32 physChannelCountInUse;
} LW2080_CTRL_FIFO_GET_PHYSICAL_CHANNEL_COUNT_PARAMS;

/*
 * LW2080_CTRL_FIFO_INFO
 *
 * This structure represents a single 32bit fifo engine value.  Clients
 * request a particular FIFO engine value by specifying a unique fifo
 * information index.
 *
 * Legal fifo information index values are:
 *   LW2080_CTRL_FIFO_INFO_INDEX_INSTANCE_TOTAL
 *     This index can be used to request the amount of instance space
 *     in kilobytes reserved by the fifo engine.
 *   LW2080_CTRL_FIFO_INFO_INDEX_MAX_CHANNEL_GROUPS
 *     This index can be used to query the maximum number of channel groups
 *     that can be allocated on the GPU.
 *   LW2080_CTRL_FIFO_INFO_INDEX_MAX_CHANNELS_PER_GROUP
 *     This index can be used to query the maximum number of channels that can
 *    be allocated in a single channel group.
 *   LW2080_CTRL_FIFO_INFO_INDEX_MAX_SUBCONTEXT_PER_GROUP
 *     This index can be used to query the maximum number of subcontext that can
 *     be allocated in a single channel group.
 *   LW2080_CTRL_FIFO_INFO_INDEX_BAR1_USERD_START_OFFSET
 *     This index can be used to query the starting offset of the RM
 *     pre-allocated USERD range in BAR1. This index query is honored only
 *     on Legacy-vGPU host RM.
 *   LW2080_CTRL_FIFO_INFO_INDEX_DEFAULT_CHANNEL_TIMESLICE
 *     This index can be used to query the default timeslice value
 *     (microseconds) used for a channel or channel group.
 *   LW2080_CTRL_FIFO_INFO_INDEX_CHANNEL_GROUPS_IN_USE
 *     This index can be used to query the number of channel groups that are
 *     already allocated on the GPU.
 *   LW2080_CTRL_FIFO_INFO_INDEX_IS_PER_RUNLIST_CHANNEL_RAM_SUPPORTED
 *     This index can be used to check if per runlist channel ram is supported, and
 *     to query the supported number of channels per runlist.
 *   LW2080_CTRL_FIFO_INFO_INDEX_MAX_CHANNEL_GROUPS_PER_ENGINE
 *     This index can be used to get max channel groups supported per engine/runlist.
 *   LW2080_CTRL_FIFO_INFO_INDEX_CHANNEL_GROUPS_IN_USE_PER_ENGINE
 *     This index can be used too get channel groups lwrrently in use per engine/runlist.
 *
 */
typedef struct LW2080_CTRL_FIFO_INFO {
    LwU32 index;
    LwU32 data;
} LW2080_CTRL_FIFO_INFO;

/* valid fifo info index values */
#define LW2080_CTRL_FIFO_INFO_INDEX_INSTANCE_TOTAL                       (0x000000000)
#define LW2080_CTRL_FIFO_INFO_INDEX_MAX_CHANNEL_GROUPS                   (0x000000001)
#define LW2080_CTRL_FIFO_INFO_INDEX_MAX_CHANNELS_PER_GROUP               (0x000000002)
#define LW2080_CTRL_FIFO_INFO_INDEX_MAX_SUBCONTEXT_PER_GROUP             (0x000000003)
#define LW2080_CTRL_FIFO_INFO_INDEX_BAR1_USERD_START_OFFSET              (0x000000004)
#define LW2080_CTRL_FIFO_INFO_INDEX_DEFAULT_CHANNEL_TIMESLICE            (0x000000005)
#define LW2080_CTRL_FIFO_INFO_INDEX_CHANNEL_GROUPS_IN_USE                (0x000000006)
#define LW2080_CTRL_FIFO_INFO_INDEX_IS_PER_RUNLIST_CHANNEL_RAM_SUPPORTED (0x000000007)
#define LW2080_CTRL_FIFO_INFO_INDEX_MAX_CHANNEL_GROUPS_PER_ENGINE        (0x000000008)
#define LW2080_CTRL_FIFO_INFO_INDEX_CHANNEL_GROUPS_IN_USE_PER_ENGINE     (0x000000009)


/* set INDEX_MAX to greatest possible index value */
#define LW2080_CTRL_FIFO_INFO_INDEX_MAX                                  LW2080_CTRL_FIFO_INFO_INDEX_DEFAULT_CHANNEL_TIMESLICE

#define LW2080_CTRL_FIFO_GET_INFO_USERD_OFFSET_SHIFT                     (12)

/*
 * LW2080_CTRL_CMD_FIFO_GET_INFO
 *
 * This command returns fifo engine information for the associated GPU.
 * Requests to retrieve fifo information use an array of one or more
 * LW2080_CTRL_FIFO_INFO structures.
 *
 *   fifoInfoTblSize
 *     This field specifies the number of valid entries in the fifoInfoList
 *     array.  This value cannot exceed LW2080_CTRL_FIFO_GET_INFO_MAX_ENTRIES.
 *   fifoInfoTbl
 *     This parameter contains the client's fifo info table into
 *     which the fifo info values will be transferred by the RM.
 *     The fifo info table is an array of LW2080_CTRL_FIFO_INFO structures.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW2080_CTRL_CMD_FIFO_GET_INFO                                    (0x20801109) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_FIFO_INTERFACE_ID << 8) | LW2080_CTRL_FIFO_GET_INFO_PARAMS_MESSAGE_ID" */

/* maximum number of LW2080_CTRL_FIFO_INFO entries per request */
#define LW2080_CTRL_FIFO_GET_INFO_MAX_ENTRIES                            (256)

#define LW2080_CTRL_FIFO_GET_INFO_PARAMS_MESSAGE_ID (0x9U)

typedef struct LW2080_CTRL_FIFO_GET_INFO_PARAMS {
    LwU32                 fifoInfoTblSize;
    /*
     * C form:
     * LW2080_CTRL_FIFO_INFO fifoInfoTbl[LW2080_CTRL_FIFO_GET_INFO_MAX_ENTRIES];
     */
    LW2080_CTRL_FIFO_INFO fifoInfoTbl[LW2080_CTRL_FIFO_GET_INFO_MAX_ENTRIES];
    LwU32                 engineType;
} LW2080_CTRL_FIFO_GET_INFO_PARAMS;



/*
 * LW2080_CTRL_FIFO_CHANNEL_PREEMPTIVE_REMOVAL
 *
 * This command removes the specified channel from the associated GPU's runlist
 * and then initiates RC recovery.  If the channel is active it will first be preempted.
 *   hChannel
 *     The handle to the channel to be preempted.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_CHANNEL
 */
#define LW2080_CTRL_CMD_FIFO_CHANNEL_PREEMPTIVE_REMOVAL (0x2080110a) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_FIFO_INTERFACE_ID << 8) | LW2080_CTRL_FIFO_CHANNEL_PREEMPTIVE_REMOVAL_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_FIFO_CHANNEL_PREEMPTIVE_REMOVAL_PARAMS_MESSAGE_ID (0xAU)

typedef struct LW2080_CTRL_FIFO_CHANNEL_PREEMPTIVE_REMOVAL_PARAMS {
    LwHandle hChannel;
} LW2080_CTRL_FIFO_CHANNEL_PREEMPTIVE_REMOVAL_PARAMS;

/*
 * LW2080_CTRL_CMD_FIFO_DISABLE_CHANNELS
 *
 * This command will disable or enable scheduling of channels described in the
 * list provided. Whether or not the channels are also preempted off the GPU
 * can be controlled by bOnlyDisableScheduling. By default channels are preempted
 * off the GPU.
 *
 *  bDisable
 *      This value determines whether to disable or
 *      enable the set of channels.
 *  numChannels
 *      The number of channels to be stopped.
 *  bOnlyDisableScheduling
 *      When false and bDisable=LW_TRUE,the call will ensure none of the listed
 *      channels are running in hardware and will not run until a call with
 *      bDisable=LW_FALSE is made. When true and bDisable=LW_TRUE, the control
 *      call will ensure that none of the listed channels can be scheduled on the
 *      GPU until a call with bDisable=LW_FALSE is made, but will not remove any
 *      of the listed channels from hardware if they are lwrrently running. When
 *      bDisable=LW_FALSE this field is ignored.
 *  bRewindGpPut
 *      If a channel is being disabled and bRewindGpPut=LW_TRUE, the channel's RAMFC
 *      will be updated so that GP_PUT is reset to the value of GP_GET.
 *  hClientList
 *      An array of LwU32 listing the client handles
 *  hChannelList
 *      An array of LwU32 listing the channel handles
 *      to be stopped.
 *  pRunlistPreemptEvent
 *      KEVENT handle for Async HW runlist preemption (unused on preMaxwell)
 *      When NULL, will revert to synchronous preemption with spinloop
 *
 * Possible status values returned are:
 *    LW_OK
 *    LWOS_ILWALID_STATE
 */

#define LW2080_CTRL_CMD_FIFO_DISABLE_CHANNELS         (0x2080110b) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_FIFO_INTERFACE_ID << 8) | LW2080_CTRL_FIFO_DISABLE_CHANNELS_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_FIFO_DISABLE_CHANNELS_MAX_ENTRIES (64)

#define LW2080_CTRL_FIFO_DISABLE_CHANNELS_PARAMS_MESSAGE_ID (0xBU)

typedef struct LW2080_CTRL_FIFO_DISABLE_CHANNELS_PARAMS {
    LwBool   bDisable;
    LwU32    numChannels;
    LwBool   bOnlyDisableScheduling;
    LwBool   bRewindGpPut;
    LW_DECLARE_ALIGNED(LwP64 pRunlistPreemptEvent, 8);
    // C form:  LwHandle hClientList[LW2080_CTRL_FIFO_DISABLE_CHANNELS_MAX_ENTRIES]
    LwHandle hClientList[LW2080_CTRL_FIFO_DISABLE_CHANNELS_MAX_ENTRIES];
    // C form:  LwHandle hChannelList[LW2080_CTRL_FIFO_DISABLE_CHANNELS_MAX_ENTRIES]
    LwHandle hChannelList[LW2080_CTRL_FIFO_DISABLE_CHANNELS_MAX_ENTRIES];
} LW2080_CTRL_FIFO_DISABLE_CHANNELS_PARAMS;

#define LW2080_CTRL_FIFO_DISABLE_CHANNEL_FALSE         (0x00000000)
#define LW2080_CTRL_FIFO_DISABLE_CHANNEL_TRUE          (0x00000001)
#define LW2080_CTRL_FIFO_ONLY_DISABLE_SCHEDULING_FALSE (0x00000000)
#define LW2080_CTRL_FIFO_ONLY_DISABLE_SCHEDULING_TRUE  (0x00000001)

/*
 * LW2080_CTRL_FIFO_MEM_INFO
 *
 * This structure describes the details of a block of memory. It consists
 * of the following fields
 *
 * aperture
 *   One of the LW2080_CTRL_CMD_FIFO_GET_CHANNEL_MEM_APERTURE_* values
 * base
 *   Physical base address of the memory
 * size
 *   Size in bytes of the memory
*/
typedef struct LW2080_CTRL_FIFO_MEM_INFO {
    LwU32 aperture;
    LW_DECLARE_ALIGNED(LwU64 base, 8);
    LW_DECLARE_ALIGNED(LwU64 size, 8);
} LW2080_CTRL_FIFO_MEM_INFO;

/*
 * LW2080_CTRL_FIFO_CHANNEL_MEM_INFO
 *
 * This structure describes the details of the instance memory, ramfc
 * and method buffers a channel. It consists of the following fields
 *
 *   inst
 *     Structure describing the details of instance memory
 *   ramfc
 *     Structure describing the details of ramfc
 *   methodBuf
 *     Array of structures describing the details of method buffers
 *   methodBufCount
 *     Number of method buffers(one per runqueue)
 */

// max runqueues
#define LW2080_CTRL_FIFO_GET_CHANNEL_MEM_INFO_MAX_COUNT 0x2

typedef struct LW2080_CTRL_FIFO_CHANNEL_MEM_INFO {
    LW_DECLARE_ALIGNED(LW2080_CTRL_FIFO_MEM_INFO inst, 8);
    LW_DECLARE_ALIGNED(LW2080_CTRL_FIFO_MEM_INFO ramfc, 8);
    LW_DECLARE_ALIGNED(LW2080_CTRL_FIFO_MEM_INFO methodBuf[LW2080_CTRL_FIFO_GET_CHANNEL_MEM_INFO_MAX_COUNT], 8);
    LwU32 methodBufCount;
} LW2080_CTRL_FIFO_CHANNEL_MEM_INFO;

/*
 * LW2080_CTRL_CMD_FIFO_GET_CHANNEL_MEM
 *
 * This command returns the memory aperture, physical base address and the
 * size of each of the instance memory, cache1 and ramfc of a channel.
 *
 *   hChannel
 *     The handle to the channel for which the memory information is desired.
 *   chMemInfo
 *     A LW2080_CTRL_FIFO_CHANNEL_MEM_INFO structure
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_CHANNEL
*/

#define LW2080_CTRL_CMD_FIFO_GET_CHANNEL_MEM_INFO (0x2080110c) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_FIFO_INTERFACE_ID << 8) | LW2080_CTRL_CMD_FIFO_GET_CHANNEL_MEM_INFO_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_CMD_FIFO_GET_CHANNEL_MEM_INFO_PARAMS_MESSAGE_ID (0xLW)

typedef struct LW2080_CTRL_CMD_FIFO_GET_CHANNEL_MEM_INFO_PARAMS {
    LwHandle hChannel;
    LW_DECLARE_ALIGNED(LW2080_CTRL_FIFO_CHANNEL_MEM_INFO chMemInfo, 8);
} LW2080_CTRL_CMD_FIFO_GET_CHANNEL_MEM_INFO_PARAMS;

#define LW2080_CTRL_CMD_FIFO_GET_CHANNEL_MEM_APERTURE_ILWALID     0x00000000
#define LW2080_CTRL_CMD_FIFO_GET_CHANNEL_MEM_APERTURE_VIDMEM      0x00000001
#define LW2080_CTRL_CMD_FIFO_GET_CHANNEL_MEM_APERTURE_SYSMEM_COH  0x00000002
#define LW2080_CTRL_CMD_FIFO_GET_CHANNEL_MEM_APERTURE_SYSMEM_NCOH 0x00000003

/*
 * LW2080_CTRL_CMD_FIFO_GET_USERD_LOCATION
 *
 *    This command determines the location (vidmem/sysmem)
 *    and attribute (cached/uncached/write combined) of memory where USERD is located.
 *
 *   aperture
 *     One of the LW2080_CTRL_CMD_FIFO_GET_USERD_LOCATION_APERTURE_* values.
 *
 *   attribute
 *     One of the LW2080_CTRL_CMD_FIFO_GET_USERD_LOCATION_ATTRIBUTE_* values.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_DEVICE
 *   LW_ERR_ILWALID_STATE
 *   LW_ERR_ILWALID_POINTER
*/

#define LW2080_CTRL_CMD_FIFO_GET_USERD_LOCATION                   (0x2080110d) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_FIFO_INTERFACE_ID << 8) | LW2080_CTRL_CMD_FIFO_GET_USERD_LOCATION_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_CMD_FIFO_GET_USERD_LOCATION_PARAMS_MESSAGE_ID (0xDU)

typedef struct LW2080_CTRL_CMD_FIFO_GET_USERD_LOCATION_PARAMS {
    LwU32 aperture;
    LwU32 attribute;
} LW2080_CTRL_CMD_FIFO_GET_USERD_LOCATION_PARAMS;

// support for CPU coherent vidmem (VIDMEM_LWILINK_COH) is not yet available in RM

#define LW2080_CTRL_CMD_FIFO_GET_USERD_LOCATION_APERTURE_VIDMEM         0x00000000
#define LW2080_CTRL_CMD_FIFO_GET_USERD_LOCATION_APERTURE_SYSMEM         0x00000001

#define LW2080_CTRL_CMD_FIFO_GET_USERD_LOCATION_ATTRIBUTE_CACHED        0x00000000
#define LW2080_CTRL_CMD_FIFO_GET_USERD_LOCATION_ATTRIBUTE_UNCACHED      0X00000001
#define LW2080_CTRL_CMD_FIFO_GET_USERD_LOCATION_ATTRIBUTE_WRITECOMBINED 0X00000002

/*
 * LW2080_CTRL_CMD_FIFO_OBJSCHED_SW_GET_LOG
 *
 * This command returns the OBJSCHED_SW log enties.
 *
 *   engineId
 *     This field specifies the LW2080_ENGINE_TYPE_* engine whose SW runlist log
 *     entries are to be fetched.
 *
 *   count
 *     This field returns the count of log entries fetched.
 *
 *   entry
 *     The array of SW runlist log entries.
 *
 *       timestampNs
 *         Timestamp in ns when this SW runlist was preeempted.
 *
 *       timeRunTotalNs
 *         Total time in ns this SW runlist has run as compared to others.
 *
 *       timeRunNs
 *         Time in ns this SW runlist ran before preemption.
 *
 *       swrlId
 *         SW runlist Id.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
*/

#define LW2080_CTRL_CMD_FIFO_OBJSCHED_SW_GET_LOG                        (0x2080110e) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_FIFO_INTERFACE_ID << 8) | LW2080_CTRL_FIFO_OBJSCHED_SW_GET_LOG_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_FIFO_OBJSCHED_SW_COUNT                              32
#define LW2080_CTRL_FIFO_OBJSCHED_SW_NCOUNTERS                          8
#define LW2080_CTRL_FIFO_OBJSCHED_SW_GET_LOG_ENTRIES                    200

#define LW2080_CTRL_FIFO_OBJSCHED_SW_GET_LOG_PARAMS_MESSAGE_ID (0xEU)

typedef struct LW2080_CTRL_FIFO_OBJSCHED_SW_GET_LOG_PARAMS {
    LwU32 engineId;
    LwU32 count;
    struct {
        LW_DECLARE_ALIGNED(LwU64 timestampNs, 8);
        LW_DECLARE_ALIGNED(LwS64 timeRunTotalNs, 8);
        LwU32 timeRunNs;
        LwU32 swrlId;
        LwU32 targetTimeSlice;
        LW_DECLARE_ALIGNED(LwU64 lwmulativePreemptionTime, 8);
        LW_DECLARE_ALIGNED(LwU64 counters[LW2080_CTRL_FIFO_OBJSCHED_SW_NCOUNTERS], 8);
    } entry[LW2080_CTRL_FIFO_OBJSCHED_SW_GET_LOG_ENTRIES];
} LW2080_CTRL_FIFO_OBJSCHED_SW_GET_LOG_PARAMS;

/*
 * LW2080_CTRL_CMD_FIFO_SUBMIT_RUNLIST
 *
 * This command can be used to submit a runlist at a specified offset.
 *
 *   submitOffset
 *     Offset at which runlist is submitted.
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
*/

#define LW2080_CTRL_CMD_FIFO_SUBMIT_RUNLIST         (0x2080110f) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_FIFO_INTERFACE_ID << 8) | LW2080_CTRL_FIFO_SUBMIT_RUNLIST_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_FIFO_SUBMIT_RUNLIST_MAX_ENTRIES (64)

#define LW2080_CTRL_FIFO_SUBMIT_RUNLIST_PARAMS_MESSAGE_ID (0xFU)

typedef struct LW2080_CTRL_FIFO_SUBMIT_RUNLIST_PARAMS {
    LwU32 submitOffset;
} LW2080_CTRL_FIFO_SUBMIT_RUNLIST_PARAMS;

/*
 * LW2080_CTRL_CMD_FIFO_CONFIG_CTXSW_TIMEOUT
 *
 * This command can be used to enable and set the engine
 * context switch timeout
 *
 * timeout: Timeout in number of microsec PTIMER ticks
 * 1 microsec PTIMER tick = 1024 PTIMER nanoseconds
 * bEnable: TRUE/FALSE
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
*/

#define LW2080_CTRL_CMD_FIFO_CONFIG_CTXSW_TIMEOUT (0x20801110) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_FIFO_INTERFACE_ID << 8) | LW2080_CTRL_FIFO_CONFIG_CTXSW_TIMEOUT_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_FIFO_CONFIG_CTXSW_TIMEOUT_PARAMS_MESSAGE_ID (0x10U)

typedef struct LW2080_CTRL_FIFO_CONFIG_CTXSW_TIMEOUT_PARAMS {
    LwU32  timeout;
    LwBool bEnable;
} LW2080_CTRL_FIFO_CONFIG_CTXSW_TIMEOUT_PARAMS;

/*
 * Control call for submitting a runlist. If another runlist of is running on the engine,
 * This call will preempt the other runlist and then submit the new runlist.
 * Please note: The previous runlist will preempt only after its own configured wfiTimeout
 * that was set when it was submitted has expired (it is a hw behavior).
 *
 * Usage:
 * Update the runlist with new tsg entries and submit
 *   hRunlist       = hRunlist1 (SW runlist handle created using rmAlloc)
 *   bUpdateRunlist = 1
 *
 *   hMemRunlistSubmitParams with a valid hMemory, with handleInfo and runlistOrderIndex.
 *    - handleInfo        Starts at offset 0 in memory. Format: LW2080_CTRL_FIFO_HANDLE_INFO
 *                        {hTSG1/hChannel1, hClient1, (opt) wfiTimeout1, (opt) timeSlice1},
 *                        {hTSG2/hChannel2, hClient2, (opt) wfiTimeout2, (opt) timeSlice2}
 *
 *    - runlistOrderIndex Starts at runlistOrderInfoOffset. Format LW2080_CTRL_FIFO_RUNLIST_ORDER_INFO
 *                        Index into handleInfo entries.
 *                        ex: ( { 0, 1, 0, 1} -> RM will generate runlist
 *                            => hTSG1/hChannel1, hTSG2/hChannel2, hTSG1/hChannel1, hTSG2/hChannel2 )
 *
 *   numHandleInfo      number of valid elements of type handleInfo starting at offset 0 in hMemRunlistSubmitParams
 *   numRunlistEntries  num of valid elements in runlistOrderIndex array) here : 4
 *
 * Submit the runlist with existing runlist entries. RM will preempt current runlist on HW, and submit
 * the specified SW runlist, without regenerating the runlist entries.
 *    hRunlist       = hRunlist1
 *    bUpdateRunlist = 0
 *
 * Stop the current runlist if it is on the engine. RM will submit a NULL runlist
 *   bUpdateRunlist            = 1
 *   numRunlistEntries         = 0
 *
 * If the current runlist is not on the engine, stop will return
 * error. One SW runlist can preempt another software runlist but not stop it.
 *
 *
 *
 * LW2080_CTRL_FIFO_SWRUNLIST_SUBMIT_PARAMS:
 * hRunlist            [IN]  Handle of the sw runlist allocated
 * hMemRunlistSubmitParams [IN] Handle to the memory where the runlist submission parameter is stored.
 *                           At offset 0, numHandleInfo number of LW2080_CTRL_FIFO_HANDLE_INFO entries
 *                           is stored. At offset runlistOrderInfoOffset, numRunlistEntries number of
 *                           LW2080_CTRL_FIFO_RUNLIST_ORDER_INFO is stored. Which is an index into
 *                           LW2080_CTRL_FIFO_HANDLE_INFO entry list.
 *
 * runlistOrderInfoOffset [IN} In bytes, offset at which the runlist order table begins.
 * submitRunlistOffset [IN]  For Turing+ submission offset of the submitting runlist.
 * bUpdateRunlist      [IN]  If TRUE, RM will generate new runlist entries from the tsgOrderIndex,
 *                           If FALSE, RM will not generate new runlist entries and
 *                           instead will submit existing runlist entries in the runlist buffer.
 *
 *   numHandleInfo     [IN] Number of valid handleInfo (LW2080_CTRL_FIFO_HANDLE_INFO) elements.
 *   numRunlistEntries [IN] Number of entries in runlistOrderIndex that defines the total TSGs in the runlist
 *
 *
 * LW2080_CTRL_FIFO_HANDLE_INFO: Unique per TSG information, to which runlistOrderIndex index
 * hChannel    [IN]  TSG/channel Handle (Rename this once we get rid of hChannel)
 * hClient     [IN]  hClient corresponding to the hTSG
 * wfiTimeout  [IN]  Time period in 100 usecs/ 100 sysclks for which host should try to WFI before graphics
 *                   preempting this runlist when it gets preempted in the future, on a per TSG basis.
 *                   Please note: This timeout is not for current submission.
 *                   Not applicable for LWCA/CE work.
 *                   Depends on: LW_PGRAPH_DEBUG_2_GFXP_WFI_TIMEOUT_UNIT to be set to USEC/SYCLK on Turing.
 *                   On pre-Turing, only SYSCLK is supported. On Turing WDDM KMD, the unit will default to
 *                   RM will only scale the this value by 100 before configuring the LW_PGRAPH_DEBUG_2_GFXP_WFI_TIMEOUT
 *                   register
 *
 * timeSlice   [IN]  TSG timeslice in 100 usecs. Supported only for TSG supported runlists.
 *
 * bConfigTimeout   [IN] If LW_TRUE: wfiTimeout is valid and will be configured
 * bConfigTimeSlice [IN] If LW_TRUE: timeSlice is valid and tsg timeslice should be
 *                       configured for this TSG
 *
 * pRunlistPreemptEvent [IN] Event handle which will be passed by KMD during runlist switch, and will be used to
 *                           notify KMD about the implicit preempt event.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_ILWALID_STATE
*/
#define LW2080_CTRL_CMD_FIFO_SWRUNLIST_SUBMIT          (0x20801111) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_FIFO_INTERFACE_ID << 8) | LW2080_CTRL_FIFO_SWRUNLIST_SUBMIT_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_FIFO_SWRUNLIST_MAX_TSGS            255
#define LW2080_CTRL_FIFO_SWRUNLIST_MAX_RUNLIST_ENTRIES 1024

typedef struct LW2080_CTRL_FIFO_HANDLE_INFO {
    LwHandle hChannel;
    LwHandle hClient;
    LwU8     wfiTimeout;
    LwU8     timeSlice;
    LwBool   bConfigTimeout;
    LwBool   bConfigTimeSlice;
} LW2080_CTRL_FIFO_HANDLE_INFO;

/*
 * Index into the handleInfo specifying the order in which the TSG is to inserted into the runlist.
 */
typedef LwU16 LW2080_CTRL_FIFO_RUNLIST_ORDER_INFO;

#define LW2080_CTRL_FIFO_SWRUNLIST_SUBMIT_PARAMS_MESSAGE_ID (0x11U)

typedef struct LW2080_CTRL_FIFO_SWRUNLIST_SUBMIT_PARAMS {
    LwU32    hRunlist;
    LwU32    submitRunlistOffset;
    LwHandle hMemRunlistSubmitParams;
    LwU32    numHandleInfo;
    LwU16    runlistOrderInfoOffset;
    LwU32    numRunlistEntries;
    LwBool   bUpdateRunlist;
    LwBool   bSkipSubmitRunlist;        // bSkipSubmitRunlist means it just generate runlist but doesn't submit
    LwBool   bSubmitLastUpdatedRunlist; // this need to set when client need to submit runlist prepare using
                                        // bUpdateRunlist and bSkipSubmitRunlist option.
    LwBool   bUpdateWfiTimeout;
    LW_DECLARE_ALIGNED(LwP64 pRunlistPreemptEvent, 8);      //KEVENT handle
} LW2080_CTRL_FIFO_SWRUNLIST_SUBMIT_PARAMS;

/*
 *  LW2080_CTRL_CMD_FIFO_GET_DEVICE_INFO_TABLE
 *
 *  This command retrieves entries from the SW encoded GPU device info table
 *  from Host RM. 
 *
 *  Parameters:
 *
 *    baseIndex [in]
 *      The starting index to read from the devinfo table. Must be a multiple of
 *      MAX_ENTRIES.
 *
 *    entries [out]
 *      A buffer to store up to MAX_ENTRIES entries of the devinfo table.
 *
 *    numEntries [out]
 *      Number of populated entries in the provided buffer.
 *
 *    bMore [out]
 *      A boolean flag indicating whether more valid entries are available to be
 *      read. A value of LW_TRUE indicates that a further call to this control
 *      with baseIndex incremented by MAX_ENTRIES will yield further valid data.
 *
 *  Possible status values returned are:
 *    LW_OK
 */
#define LW2080_CTRL_CMD_FIFO_GET_DEVICE_INFO_TABLE                 (0x20801112) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_FIFO_INTERFACE_ID << 8) | LW2080_CTRL_FIFO_GET_DEVICE_INFO_TABLE_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_FIFO_GET_DEVICE_INFO_TABLE_MAX_DEVICES         256
#define LW2080_CTRL_FIFO_GET_DEVICE_INFO_TABLE_MAX_ENTRIES         32
#define LW2080_CTRL_FIFO_GET_DEVICE_INFO_TABLE_ENGINE_DATA_TYPES   16
#define LW2080_CTRL_FIFO_GET_DEVICE_INFO_TABLE_ENGINE_MAX_PBDMA    2
#define LW2080_CTRL_FIFO_GET_DEVICE_INFO_TABLE_ENGINE_MAX_NAME_LEN 16

/*
 * LW2080_CTRL_FIFO_DEVICE_ENTRY
 *
 * This structure contains the engine, engine name and
 * push buffers information of FIFO device entry. It consists of the following fields
 *
 *   engineData
 *     Type of the engine
 *   pbdmaIds
 *     List of pbdma ids associated with engine
 *   pbdmaFaultIds
 *     List of pbdma fault ids associated with engine
 *   numPbdmas
 *     Number of pbdmas 
 *   engineName
 *     Name of the engine
 */
typedef struct LW2080_CTRL_FIFO_DEVICE_ENTRY {
    LwU32 engineData[LW2080_CTRL_FIFO_GET_DEVICE_INFO_TABLE_ENGINE_DATA_TYPES];
    LwU32 pbdmaIds[LW2080_CTRL_FIFO_GET_DEVICE_INFO_TABLE_ENGINE_MAX_PBDMA];
    LwU32 pbdmaFaultIds[LW2080_CTRL_FIFO_GET_DEVICE_INFO_TABLE_ENGINE_MAX_PBDMA];
    LwU32 numPbdmas;
    char  engineName[LW2080_CTRL_FIFO_GET_DEVICE_INFO_TABLE_ENGINE_MAX_NAME_LEN];
} LW2080_CTRL_FIFO_DEVICE_ENTRY;

#define LW2080_CTRL_FIFO_GET_DEVICE_INFO_TABLE_PARAMS_MESSAGE_ID (0x12U)

typedef struct LW2080_CTRL_FIFO_GET_DEVICE_INFO_TABLE_PARAMS {
    LwU32                         baseIndex;
    LwU32                         numEntries;
    LwBool                        bMore;
    // C form: LW2080_CTRL_FIFO_DEVICE_ENTRY entries[LW2080_CTRL_FIFO_GET_DEVICE_INFO_TABLE_MAX_ENTRIES];
    LW2080_CTRL_FIFO_DEVICE_ENTRY entries[LW2080_CTRL_FIFO_GET_DEVICE_INFO_TABLE_MAX_ENTRIES];
} LW2080_CTRL_FIFO_GET_DEVICE_INFO_TABLE_PARAMS;

/*
 *  LW2080_CTRL_CMD_FIFO_CLEAR_FAULTED_BIT
 *
 *  This command clears the ENGINE or PBDMA FAULTED bit and reschedules the faulted channel
 *  by ringing channel's doorbell
 *
 *  Parameters:
 *
 *    engineType [in]
 *      The LW2080_ENGINE_TYPE of the engine to which the faulted
 *      channel is bound. This may be a logical id for guest RM in
 *      case of SMC.
 *
 *    vChid [in]
 *      Virtual channel ID on which the fault oclwrred
 *
 *    faultType [in]
 *      Whether fault was triggered by engine (_ENGINE_FAULTED) or PBDMA (_PBDMA_FAULTED)
 *      The value specified must be one of the LW2080_CTRL_FIFO_CLEAR_FAULTED_BIT_FAULT_TYPE_* values
 *
 *  Possible status values returned are:
 *    LW_OK
 *    LW_ERR_ILWALID_STATE
 *    LW_ERR_ILWALID_ARGUMENT
 */
#define LW2080_CTRL_CMD_FIFO_CLEAR_FAULTED_BIT               (0x20801113) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_FIFO_INTERFACE_ID << 8) | LW2080_CTRL_CMD_FIFO_CLEAR_FAULTED_BIT_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_FIFO_CLEAR_FAULTED_BIT_FAULT_TYPE_ENGINE 0x00000001
#define LW2080_CTRL_FIFO_CLEAR_FAULTED_BIT_FAULT_TYPE_PBDMA  0x00000002

#define LW2080_CTRL_CMD_FIFO_CLEAR_FAULTED_BIT_PARAMS_MESSAGE_ID (0x13U)

typedef struct LW2080_CTRL_CMD_FIFO_CLEAR_FAULTED_BIT_PARAMS {
    LwU32 engineType;
    LwU32 vChid;
    LwU32 faultType;
} LW2080_CTRL_CMD_FIFO_CLEAR_FAULTED_BIT_PARAMS;

/*
 *  LW2080_CTRL_CMD_FIFO_GET_SCHID
 *
 *  This command returns the physical SCHID for a given VCHID.
 *
 *  Parameters:
 *
 *    engineType [in]
 *      The LW2080_ENGINE_TYPE of the engine to which the channel
 *      is bound. This may be a logical id for guest RM in case of
 *      SMC
 *
 *    vChid [in]
 *      Virtual channel ID
 *
 *    sChid [out]
 *      System Channel ID
 *
 *  Possible status values returned are:
 *    LW_OK
 *    LW_ERR_ILWALID_STATE
 *    LW_ERR_ILWALID_ARGUMENT
 */
#define LW2080_CTRL_CMD_FIFO_GET_SCHID (0x20801114) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_FIFO_INTERFACE_ID << 8) | LW2080_CTRL_CMD_FIFO_GET_SCHID_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_CMD_FIFO_GET_SCHID_PARAMS_MESSAGE_ID (0x14U)

typedef struct LW2080_CTRL_CMD_FIFO_GET_SCHID_PARAMS {
    LwU32 engineType;
    LwU32 vChid;
    LwU32 sChid;
} LW2080_CTRL_CMD_FIFO_GET_SCHID_PARAMS;

/*
 * LW2080_CTRL_CMD_FIFO_RUNLIST_SET_SCHED_POLICY
 *
 * Allows clients to set the global scheduling policy for all runlists
 * associated to the given subdevice.
 *
 * Lwrrently, this is only supported for HW runlists.
 *
 * Since this is a global setting, only privileged clients will be allowed to
 * set it. Regular clients will get LW_ERR_INSUFFICIENT_PERMISSIONS error.
 *
 * Once a certain scheduling policy is set, that policy cannot be changed to a
 * different one unless all clients which set it have either restored the policy
 * (using the corresponding restore flag) or died. Clients trying to set a
 * policy while a different one is locked by another client will get a
 * LW_ERR_ILWALID_STATE error.
 *
 * The same client can set a scheduling policy and later change to another one
 * only when no other clients have set the same policy. Such sequence will be
 * equivalent to restoring the policy in between.
 *
 * For instance, the following sequence:
 *
 *      1. Set policy A
 *      2. Set policy B
 *
 * is equivalent to:
 *
 *      1. Set policy A
 *      2. Restore policy
 *      3. Set policy B
 *
 * Parameters:
 *
 *   flags
 *     This field specifies the operational properties to be applied:
 *
 *      - LW2080_CTRL_CMD_FIFO_RUNLIST_SET_SCHED_POLICY_FLAGS_RESTORE_FALSE
 *          Try to set the provided 'schedPolicy' scheduling policy. If the
 *          operation succeeds, other clients will be prevented from setting a
 *          different scheduling policy until all clients using it have either
 *          restored it or died.
 *
 *      - LW2080_CTRL_CMD_FIFO_RUNLIST_SET_SCHED_POLICY_FLAGS_RESTORE_TRUE
 *          Let the scheduler know the client no longer requires the current
 *          scheduling policy. This may or may not actually change the
 *          scheduling policy, depending on how many other clients are also
 *          using the current policy.
 *
 *          The 'schedPolicy' parameter is ignored when this flag is set.
 *
 *   schedPolicy
 *     One of:
 *
 *      - LW2080_CTRL_FIFO_RUNLIST_SCHED_POLICY_DEFAULT
 *          Set the default scheduling policy and prevent other clients from
 *          changing it.
 *
 *      - LW2080_CTRL_FIFO_RUNLIST_SCHED_POLICY_CHANNEL_INTERLEAVED
 *          This scheduling policy will make channels to be scheduled according
 *          to their interleave level. See LWA06C_CTRL_CMD_SET_INTERLEAVE_LEVEL
 *          description for more details.
 *      - LW2080_CTRL_FIFO_RUNLIST_SCHED_POLICY_CHANNEL_INTERLEAVED_WDDM
 *          This scheduling policy will make channels to be scheduled according
 *          to their interleave level per WDDM policy. 
 *          See LWA06C_CTRL_CMD_SET_INTERLEAVE_LEVEL description for more details.       
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_DEVICE
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_INSUFFICIENT_PERMISSIONS
 *   LW_ERR_ILWALID_STATE
 *   LW_ERR_NOT_SUPPORTED
 */
#define LW2080_CTRL_CMD_FIFO_RUNLIST_SET_SCHED_POLICY                     (0x20801115) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_FIFO_INTERFACE_ID << 8) | LW2080_CTRL_FIFO_RUNLIST_SET_SCHED_POLICY_PARAMS_MESSAGE_ID" */

/* schedPolicy values */
#define LW2080_CTRL_FIFO_RUNLIST_SCHED_POLICY_DEFAULT                     0x0
#define LW2080_CTRL_FIFO_RUNLIST_SCHED_POLICY_CHANNEL_INTERLEAVED         0x1
#define LW2080_CTRL_FIFO_RUNLIST_SCHED_POLICY_CHANNEL_INTERLEAVED_WDDM    0x2

/* SET_SCHED_POLICY flags */
#define LW2080_CTRL_CMD_FIFO_RUNLIST_SET_SCHED_POLICY_FLAGS_RESTORE        0:0
#define LW2080_CTRL_CMD_FIFO_RUNLIST_SET_SCHED_POLICY_FLAGS_RESTORE_FALSE (0x00000000)
#define LW2080_CTRL_CMD_FIFO_RUNLIST_SET_SCHED_POLICY_FLAGS_RESTORE_TRUE  (0x00000001)

#define LW2080_CTRL_FIFO_RUNLIST_SET_SCHED_POLICY_PARAMS_MESSAGE_ID (0x15U)

typedef struct LW2080_CTRL_FIFO_RUNLIST_SET_SCHED_POLICY_PARAMS {
    LwU32 flags;
    LwU32 schedPolicy;
} LW2080_CTRL_FIFO_RUNLIST_SET_SCHED_POLICY_PARAMS;

/*
 *  LW2080_CTRL_CMD_FIFO_UPDATE_CHANNEL_INFO
 *
 *  This command updates the channel info params for an existing channel
 *
 *  Can be a deferred Api. The control call can be used for migrating a
 *
 *  channel to a new userd and gpfifo
 *
 *  Parameters:
 *     [in] hClient  - Client handle
 *     [in] hChannel - Channel handle
 *     [in] hUserdMemory  - UserD handle
 *     [in] gpFifoEntries - Number of Gpfifo Entries
 *     [in] gpFifoOffset  - Gpfifo Virtual Offset
 *     [in] userdOffset   - UserD offset
 *
 *
 *  Possible status values returned are:
 *    LW_OK
 *    LW_ERR_ILWALID_STATE
 *    LW_ERR_ILWALID_ARGUMENT
 */
#define LW2080_CTRL_CMD_FIFO_UPDATE_CHANNEL_INFO (0x20801116) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_FIFO_INTERFACE_ID << 8) | LW2080_CTRL_FIFO_UPDATE_CHANNEL_INFO_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_FIFO_UPDATE_CHANNEL_INFO_PARAMS_MESSAGE_ID (0x16U)

typedef struct LW2080_CTRL_FIFO_UPDATE_CHANNEL_INFO_PARAMS {
    LwHandle hClient;
    LwHandle hChannel;
    LwHandle hUserdMemory;
    LwU32    gpFifoEntries;
    LW_DECLARE_ALIGNED(LwU64 gpFifoOffset, 8);
    LW_DECLARE_ALIGNED(LwU64 userdOffset, 8);
} LW2080_CTRL_FIFO_UPDATE_CHANNEL_INFO_PARAMS;

/*
 * LW2080_CTRL_CMD_FIFO_DISABLE_USERMODE_CHANNELS
 *
 * This command will disable or enable scheduling of all usermode channels.
 *
 *  bDisable
 *      This value determines whether to disable or enable the usermode channels.
 *
 * Possible status values returned are:
 *    LW_OK
 *    LW_ERR_ILWALID_ARGUMENT
 */
#define LW2080_CTRL_CMD_FIFO_DISABLE_USERMODE_CHANNELS (0x20801117) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_FIFO_INTERFACE_ID << 8) | LW2080_CTRL_FIFO_DISABLE_USERMODE_CHANNELS_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_FIFO_DISABLE_USERMODE_CHANNELS_PARAMS_MESSAGE_ID (0x17U)

typedef struct LW2080_CTRL_FIFO_DISABLE_USERMODE_CHANNELS_PARAMS {
    LwBool bDisable;
} LW2080_CTRL_FIFO_DISABLE_USERMODE_CHANNELS_PARAMS;

/*
 * LW2080_CTRL_CMD_FIFO_SETUP_VF_ZOMBIE_SUBCTX_PDB
 *
 * When a VF subcontext is marked as a zombie, host RM points its PDB to a dummy
 * page allocated by guest RM in GPA space. This command provides the parameters
 * of the guest RMs memory descriptor to be able to create a corresponding 
 * memory descriptor on the host RM. Host RM uses this to program the PDB of a
 * zombie subcontext.
 * 
 *  Parameters:
 *  Input parameters to describe the memory descriptor
 *     [in] base
 *     [in] size
 *     [in] addressSpace
 *     [in] cacheAttrib
 */
#define LW2080_CTRL_CMD_FIFO_SETUP_VF_ZOMBIE_SUBCTX_PDB (0x20801118) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_FIFO_INTERFACE_ID << 8) | LW2080_CTRL_FIFO_SETUP_VF_ZOMBIE_SUBCTX_PDB_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_FIFO_SETUP_VF_ZOMBIE_SUBCTX_PDB_PARAMS_MESSAGE_ID (0x18U)

typedef struct LW2080_CTRL_FIFO_SETUP_VF_ZOMBIE_SUBCTX_PDB_PARAMS {
    LW_DECLARE_ALIGNED(LwU64 base, 8);
    LW_DECLARE_ALIGNED(LwU64 size, 8);
    LwU32 addressSpace;
    LwU32 cacheAttrib;
} LW2080_CTRL_FIFO_SETUP_VF_ZOMBIE_SUBCTX_PDB_PARAMS;

/* _ctrl2080fifo_h_ */
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)

