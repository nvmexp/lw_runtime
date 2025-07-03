/*
 * SPDX-FileCopyrightText: Copyright (c) 2014-2020 LWPU CORPORATION & AFFILIATES. All rights reserved.
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
// Source file: ctrl/ctrl2080/ctrl2080flcn.finn
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


//
// XAPICHK/XAPI_TEST breaks on including "lwmisc.h". Workaround: don't include
// the header in that case and just redefine the macros we need.
//
#include "lwmisc.h"
/*
 * Obsolete Falcon ID type. Use LW2080_ENGINE_TYPE_ instead
 */
#define FALCON_ID_PMU           (LW2080_ENGINE_TYPE_PMU)
#define FALCON_ID_DPU           (LW2080_ENGINE_TYPE_DPU)
#define FALCON_ID_SEC2          (LW2080_ENGINE_TYPE_SEC2)
#define FALCON_ID_FBFLCN        (LW2080_ENGINE_TYPE_FBFLCN)

/*
 * LW2080_CTRL_CMD_FLCN_GET_DMEM_USAGE
 *
 * This command returns total heap size and free heap size of a falcon engine
 *
 *  flcnID
 *    The falcon ID
 *
 *  heapSize
 *    Total heap size in byte
 *
 *  heapFree
 *    Total free heap size in byte
 *
 *  Possible status values returned are:
 *    LW_OK
 *    LW_ERR_NOT_SUPPORTED
 *    LW_ERR_ILWALID_ARGUMENT
 *    LW_ERR_ILWALID_OBJECT
 */
#define LW2080_CTRL_CMD_FLCN_GET_DMEM_USAGE (0x20803101) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_FLCN_INTERFACE_ID << 8) | LW2080_CTRL_FLCN_GET_DMEM_USAGE_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_FLCN_GET_DMEM_USAGE_PARAMS_MESSAGE_ID (0x1U)

typedef struct LW2080_CTRL_FLCN_GET_DMEM_USAGE_PARAMS {
    LwU32 flcnID;
    LwU32 heapSize;
    LwU32 heapFree;
} LW2080_CTRL_FLCN_GET_DMEM_USAGE_PARAMS;

/*!
 * @defgroup LWOS_INST_EVT Instrumentation event types.
 * @{
 */
#define LW2080_CTRL_FLCN_LWOS_INST_EVT_RECALIBRATE                0x00U
#define LW2080_CTRL_FLCN_LWOS_INST_EVT_CTXSW_TICK                 0x01U
#define LW2080_CTRL_FLCN_LWOS_INST_EVT_CTXSW_YIELD                0x02U
#define LW2080_CTRL_FLCN_LWOS_INST_EVT_CTXSW_INT0                 0x03U
#define LW2080_CTRL_FLCN_LWOS_INST_EVT_CTXSW_BLOCK                0x04U
#define LW2080_CTRL_FLCN_LWOS_INST_EVT_UNBLOCK                    0x05U
#define LW2080_CTRL_FLCN_LWOS_INST_EVT_HANDLER_BEGIN              0x06U
#define LW2080_CTRL_FLCN_LWOS_INST_EVT_HANDLER_END                0x07U
#define LW2080_CTRL_FLCN_LWOS_INST_EVT_QUEUE_SEND                 0x08U
#define LW2080_CTRL_FLCN_LWOS_INST_EVT_QUEUE_RECV                 0x09U
#define LW2080_CTRL_FLCN_LWOS_INST_EVT_RPC_BEGIN                  0x0AU
#define LW2080_CTRL_FLCN_LWOS_INST_EVT_RPC_END                    0x0BU
#define LW2080_CTRL_FLCN_LWOS_INST_EVT_SKIPPED                    0x0LW
#define LW2080_CTRL_FLCN_LWOS_INST_EVT_TASK_EXEC_PROFILE_BEGIN    0x0DU
#define LW2080_CTRL_FLCN_LWOS_INST_EVT_TASK_EXEC_PROFILE_END      0x0EU
#define LW2080_CTRL_FLCN_LWOS_INST_EVT_TASK_LOAD_PROFILE_BEGIN    0x0FU
#define LW2080_CTRL_FLCN_LWOS_INST_EVT_TASK_LOAD_PROFILE_END      0x10U
#define LW2080_CTRL_FLCN_LWOS_INST_EVT_TASK_ODP_CODE_BEGIN        0x11U
#define LW2080_CTRL_FLCN_LWOS_INST_EVT_TASK_ODP_CODE_END          0x12U
#define LW2080_CTRL_FLCN_LWOS_INST_EVT_TASK_ODP_DATA_BEGIN        0x13U
#define LW2080_CTRL_FLCN_LWOS_INST_EVT_TASK_ODP_DATA_END          0x14U
#define LW2080_CTRL_FLCN_LWOS_INST_EVT_TASK_DMA_PROFILE_BEGIN     0x15U
#define LW2080_CTRL_FLCN_LWOS_INST_EVT_TASK_DMA_PROFILE_END       0x16U
#define LW2080_CTRL_FLCN_LWOS_INST_EVT_TASK_DMA_ODP_PROFILE_BEGIN 0x17U
#define LW2080_CTRL_FLCN_LWOS_INST_EVT_TASK_DMA_ODP_PROFILE_END   0x18U
#define LW2080_CTRL_FLCN_LWOS_INST_EVT_USER_LWSTOM_BEGIN          0x19U
#define LW2080_CTRL_FLCN_LWOS_INST_EVT_USER_LWSTOM_END            0x1AU
#define LW2080_CTRL_FLCN_LWOS_INST_EVT_USER_LWSTOM_2_BEGIN        0x1BU
#define LW2080_CTRL_FLCN_LWOS_INST_EVT_USER_LWSTOM_2_END          0x1LW
#define LW2080_CTRL_FLCN_LWOS_INST_EVT_USER_LWSTOM_3_BEGIN        0x1DU
#define LW2080_CTRL_FLCN_LWOS_INST_EVT_USER_LWSTOM_3_END          0x1EU
/*!@}*/

#define LW2080_CTRL_FLCN_LWOS_INST_ILWALID_TASK_ID                0xFFU

/*!
 * Interrupts and exceptions both use the same event type. Set the first bit
 * for exceptions to differentiate between the two.
 */
#define LW2080_CTRL_FLCN_LWOS_INST_IRQ_ID(_irqId)                (_irqId)
#define LW2080_CTRL_FLCN_LWOS_INST_EXCI_ID(_exciId)              (_exciId | (1 << 7))

/*!
 * @defgroup LW_INSTRUMENTATION_EVENT_DATA Instrumentation event struct entry.
 *
 * This is a binary-packed representation of event type and additional data,
 * including timing data and tracking IDs.
 *
 * @{
 */

/*!
 * Below DRF needs constants assigned to start and end so it can be represented in FINN properly
 * This is because FINN v1 will not have the ability to represent DRF's and bit fields yet
 */
#define LW_INSTRUMENTATION_EVENT_DATA_EVENT_TYPE_DRF_EXTENT       (31)
#define LW_INSTRUMENTATION_EVENT_DATA_EVENT_TYPE_DRF_BASE         (27)
#define LW_INSTRUMENTATION_EVENT_DATA_EVENT_TYPE \
    (LW_INSTRUMENTATION_EVENT_DATA_EVENT_TYPE_DRF_EXTENT) : \
    (LW_INSTRUMENTATION_EVENT_DATA_EVENT_TYPE_DRF_BASE)

#define LW_INSTRUMENTATION_EVENT_DATA_TASK_ID     26:19
#define LW_INSTRUMENTATION_EVENT_DATA_EXTRA       26:19
#define LW_INSTRUMENTATION_EVENT_DATA_TIME_DELTA  18:0
#define LW_INSTRUMENTATION_EVENT_DATA_TIME_ABS    26:0
/*!@}*/

/*!
 * The maximum number of event types, callwlated from the number of bits in the
 * event structure.
 */
#define LW2080_CTRL_FLCN_LWOS_INST_NUM_EVT_TYPES                  (0x20U) /* finn: Evaluated from "(1 << (LW_INSTRUMENTATION_EVENT_DATA_EVENT_TYPE_DRF_EXTENT - LW_INSTRUMENTATION_EVENT_DATA_EVENT_TYPE_DRF_BASE + 1))" */

/*!
 * The number of bytes required in the event mask to contain all event types.
 */
#define LW2080_CTRL_FLCN_LWOS_INST_MASK_SIZE_BYTES                (0x4U) /* finn: Evaluated from "(LW2080_CTRL_FLCN_LWOS_INST_NUM_EVT_TYPES / 8)" */

/*!
 * Instrumentation event bitfield structure. Exact structure depends on the
 * first five bits, which represent event type.
 *
 * For most event types, the structure is:
 *  - 5 bits of event type
 *  - 8 bits of ID
 *  - 19 bits of delta time (time since last event). If we've missed some
 *    events, it's the amount of time since the last event that was not
 *    skipped. If this time would overflow, a recalibration event is inserted
 *    instead (see below).
 *
 * The main exception is the recalibration event, which has no ID/delta time
 * fields and instead has a 27-bit absolute timestamp. This event is used
 * when the gap between two events is greater than the maximum 20-bit integer.
 *
 * All timestamps are represented in increments of 32ns
 * (the finest possible timer resolution).
 */
typedef struct LWOS_INSTRUMENTATION_EVENT {
    /*!
     * Field containing the event type and data.
     *
     * Bitmask of @ref LW_INSTRUMENTATION_EVENT_DATA.
     */
    LwU32 data;
} LWOS_INSTRUMENTATION_EVENT;
typedef struct LWOS_INSTRUMENTATION_EVENT *PLWOS_INSTRUMENTATION_EVENT;

/*
 * LW2080_CTRL_CMD_FLCN_INSTRUMENTATION_MAP/UNMAP
 *
 * Params for both RmCtrls are the same (out for _MAP, in for _UNMAP)
 */
#define LW2080_CTRL_CMD_FLCN_INSTRUMENTATION_MAP   (0x20803112) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_FLCN_INTERFACE_ID << 8) | 0x12" */

#define LW2080_CTRL_CMD_FLCN_INSTRUMENTATION_UNMAP (0x20803113) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_FLCN_INTERFACE_ID << 8) | 0x13" */

typedef struct LW2080_CTRL_FLCN_INSTRUMENTATION_MAP_PARAMS {
    //! The engine type, from LW2080_ENGINE_TYPE_*
    LwU32 engine;

    //! The beginning of the instrumentation buffer, mapped to user memory.
    LW_DECLARE_ALIGNED(LwP64 begin, 8);

    // Priv pointer for memory mapping.
    LW_DECLARE_ALIGNED(LwP64 pPriv, 8);

    // The size of the user-mapped instrumentation buffer.
    LW_DECLARE_ALIGNED(LwU64 size, 8);
} LW2080_CTRL_FLCN_INSTRUMENTATION_MAP_PARAMS;

/*
 * LW2080_CTRL_CMD_FLCN_INSTRUMENTATION_GET_INFO
 *
 * Get static information about FLCN instrumentation.
 */
#define LW2080_CTRL_CMD_FLCN_INSTRUMENTATION_GET_INFO (0x20803114) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_FLCN_INTERFACE_ID << 8) | LW2080_CTRL_FLCN_INSTRUMENTATION_GET_INFO_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_FLCN_INSTRUMENTATION_GET_INFO_PARAMS_MESSAGE_ID (0x14U)

typedef struct LW2080_CTRL_FLCN_INSTRUMENTATION_GET_INFO_PARAMS {
    //! The engine type, from LW2080_ENGINE_TYPE_*
    LwU32  engine;

    //! Whether or not instrumentation is enabled
    LwBool bEnabled;

    /*!
     * Whether we use PTIMER (resolution 32ns) or the 30us timer tick (LW_TRUE
     * is PTIMER).
     */
    LwBool bIsTimerPrecise;
} LW2080_CTRL_FLCN_INSTRUMENTATION_GET_INFO_PARAMS;

/*
 * LW2080_CTRL_CMD_FLCN_INSTRUMENTATION_GET/SET_CONTROL
 *
 * Get/set the event bitmask.
 */
#define LW2080_CTRL_CMD_FLCN_INSTRUMENTATION_GET_CONTROL (0x20803115) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_FLCN_INTERFACE_ID << 8) | 0x15" */

#define LW2080_CTRL_CMD_FLCN_INSTRUMENTATION_SET_CONTROL (0x20803116) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_FLCN_INTERFACE_ID << 8) | 0x16" */

typedef struct LW2080_CTRL_FLCN_INSTRUMENTATION_CONTROL_PARAMS {
    //! The engine type, from LW2080_ENGINE_TYPE_*
    LwU32 engine;

    /*!
     * The bitmask of which event types to log. An event type corresponding to
     * a bit with a zero will be ignored at the log site, which prevents it
     * from filling up the resident buffer in the PMU. In general, set this to
     * only log the event types you actually want to use.
     * Refer to LWOS_BM_* in lwos_utility.h for usage.
     */
    LwU8  mask[4];
} LW2080_CTRL_FLCN_INSTRUMENTATION_CONTROL_PARAMS;

/*
 * LW2080_CTRL_CMD_FLCN_INSTRUMENTATION_RECALIBRATE
 *
 * Send a recalibrate event to the intstrumentation.
 */
#define LW2080_CTRL_CMD_FLCN_INSTRUMENTATION_RECALIBRATE (0x20803117) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_FLCN_INTERFACE_ID << 8) | LW2080_CTRL_FLCN_INSTRUMENTATION_RECALIBRATE_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_FLCN_INSTRUMENTATION_RECALIBRATE_PARAMS_MESSAGE_ID (0x17U)

typedef struct LW2080_CTRL_FLCN_INSTRUMENTATION_RECALIBRATE_PARAMS {
    //! The engine type, from LW2080_ENGINE_TYPE_*
    LwU32 engine;
} LW2080_CTRL_FLCN_INSTRUMENTATION_RECALIBRATE_PARAMS;


/*
 * LW2080_CTRL_CMD_FLCN_GET_ENGINE_ARCH
 *
 * Get the egine arch i.e FALCON, RISCV etc given the LW2080_ENGINE_TYPE_*.
 *
 */
#define LW2080_CTRL_CMD_FLCN_GET_ENGINE_ARCH (0x20803118) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_FLCN_INTERFACE_ID << 8) | LW2080_CTRL_FLCN_GET_ENGINE_ARCH_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_FLCN_GET_ENGINE_ARCH_PARAMS_MESSAGE_ID (0x18U)

typedef struct LW2080_CTRL_FLCN_GET_ENGINE_ARCH_PARAMS {
    //! The engine type, from LW2080_ENGINE_TYPE_*
    LwU32 engine;

    //! The engine architecture - FALCON or RISC-V
    LwU32 engineArch;
} LW2080_CTRL_FLCN_GET_ENGINE_ARCH_PARAMS;

/*!
 * @defgroup Engine Arch types
 * @{
 */
#define LW2080_CTRL_FLCN_GET_ENGINE_ARCH_DEFAULT                         0x0
#define LW2080_CTRL_FLCN_GET_ENGINE_ARCH_FALCON                          0x1
#define LW2080_CTRL_FLCN_GET_ENGINE_ARCH_RISCV                           0x2
/*!@}*/


/* ----------------------- uStreamer (INST v2) ------------------------------ */
/*!
 * @defgroup LW2080_CTRL_FLCN_USTREAMER_EVENT uStreamer event fields.
 *
 * This is a binary-packed representation of uStreamer events. There are
 * three main types of entry: Head, Payload, and Tail. COMM here is used
 * when a field is shared among multiple event types.
 *
 * @{
 */
#define LW2080_CTRL_FLCN_USTREAMER_EVENT_COMM_FLAG                         31:31
#define LW2080_CTRL_FLCN_USTREAMER_EVENT_COMM_HEAD                         30:30
#define LW2080_CTRL_FLCN_USTREAMER_EVENT_TAIL_VARIABLE                     29:29
#define LW2080_CTRL_FLCN_USTREAMER_EVENT_TAIL_EXTEND                       28:28

/*!
 * Below DRF's need constants assigned to start and end so they can be represented in FINN properly
 * This is because FINN v1 will not have the ability to represent DRF's and bit fields yet.
 */
#define LW2080_CTRL_FLCN_USTREAMER_EVENT_TAIL_EVENTID_DRF_EXTENT         (27)
#define LW2080_CTRL_FLCN_USTREAMER_EVENT_TAIL_EVENTID_DRF_BASE           (20)
#define LW2080_CTRL_FLCN_USTREAMER_EVENT_TAIL_EVENTID \
    (LW2080_CTRL_FLCN_USTREAMER_EVENT_TAIL_EVENTID_DRF_EXTENT) : \
    (LW2080_CTRL_FLCN_USTREAMER_EVENT_TAIL_EVENTID_DRF_BASE)
#define LW2080_CTRL_FLCN_USTREAMER_EVENT_TAIL_EVENTIDCOMPACT_DRF_EXTENT  (28)
#define LW2080_CTRL_FLCN_USTREAMER_EVENT_TAIL_EVENTIDCOMPACT_DRF_BASE    (24)
#define LW2080_CTRL_FLCN_USTREAMER_EVENT_TAIL_EVENTIDCOMPACT \
    (LW2080_CTRL_FLCN_USTREAMER_EVENT_TAIL_EVENTIDCOMPACT_DRF_EXTENT) : \
    (LW2080_CTRL_FLCN_USTREAMER_EVENT_TAIL_EVENTIDCOMPACT_DRF_BASE)

#define LW2080_CTRL_FLCN_USTREAMER_EVENT_TAIL_LENGTH                        19:8
#define LW2080_CTRL_FLCN_USTREAMER_EVENT_TAIL_PAYLOAD                        7:0
#define LW2080_CTRL_FLCN_USTREAMER_EVENT_TAIL_PAYLOADCOMPACT                23:0
#define LW2080_CTRL_FLCN_USTREAMER_EVENT_HEAD_TIME                          29:0
#define LW2080_CTRL_FLCN_USTREAMER_EVENT_DATA_PAYLOAD                       30:0
/*!@}*/


/*!
 * @defgroup LW2080_CTRL_FLCN_USTREAMER_FEATURE
 *
 * This defines all the features lwrrently supported by uStreamer. For a new
 * usecase of uStreamer, a feature should be defined here describing the usecase.
 * This value should be unique for each queue.
 *
 * @{
 */
#define LW2080_CTRL_FLCN_USTREAMER_FEATURE_DEFAULT                       0U
#define LW2080_CTRL_FLCN_USTREAMER_FEATURE_PMUMON                        1U
#define LW2080_CTRL_FLCN_USTREAMER_FEATURE__COUNT                        2U
/*!@}*/

/*!
 * @defgroup LW2080_CTRL_FLCN_USTREAMER_QUEUE_POLICY
 *
 * This defines the DRF used for ustreamer queue policy
 *
 * @{
 */

#define LW2080_CTRL_FLCN_USTREAMER_QUEUE_POLICY_IDLE_FLUSH                   0:0
#define LW2080_CTRL_FLCN_USTREAMER_QUEUE_POLICY_IDLE_FLUSH_DISABLED      0U
#define LW2080_CTRL_FLCN_USTREAMER_QUEUE_POLICY_IDLE_FLUSH_ENABLED       1U

#define LW2080_CTRL_FLCN_USTREAMER_QUEUE_POLICY_FULL_FLUSH                   1:1
#define LW2080_CTRL_FLCN_USTREAMER_QUEUE_POLICY_FULL_FLUSH_DISABLED      0U
#define LW2080_CTRL_FLCN_USTREAMER_QUEUE_POLICY_FULL_FLUSH_ENABLED       1U

#define LW2080_CTRL_FLCN_USTREAMER_QUEUE_POLICY_IMMEDIATE_FLUSH              2:2
#define LW2080_CTRL_FLCN_USTREAMER_QUEUE_POLICY_IMMEDIATE_FLUSH_DISABLED 0U
#define LW2080_CTRL_FLCN_USTREAMER_QUEUE_POLICY_IMMEDIATE_FLUSH_ENABLED  1U

#define LW2080_CTRL_FLCN_USTREAMER_QUEUE_POLICY_IDLE_THRESHOLD              31:8

/*!@}*/

/*!
 * The maximum number of compact event types, callwlated from the number of bits
 * in the event structure.
 */
#define LW2080_CTRL_FLCN_USTREAMER_NUM_EVT_TYPES_COMPACT                 (0x20U) /* finn: Evaluated from "(1 << (LW2080_CTRL_FLCN_USTREAMER_EVENT_TAIL_EVENTIDCOMPACT_DRF_EXTENT - LW2080_CTRL_FLCN_USTREAMER_EVENT_TAIL_EVENTIDCOMPACT_DRF_BASE + 1))" */

/*!
 * The maximum number of event types, callwlated from the number of bits in the
 * event structure.
 */
#define LW2080_CTRL_FLCN_USTREAMER_NUM_EVT_TYPES                         (0x120U) /* finn: Evaluated from "((1 << (LW2080_CTRL_FLCN_USTREAMER_EVENT_TAIL_EVENTID_DRF_EXTENT - LW2080_CTRL_FLCN_USTREAMER_EVENT_TAIL_EVENTID_DRF_BASE + 1)) + LW2080_CTRL_FLCN_USTREAMER_NUM_EVT_TYPES_COMPACT)" */

/*!
 * The number of bytes required in the event mask to contain all event types.
 */
#define LW2080_CTRL_FLCN_USTREAMER_MASK_SIZE_BYTES                       (0x24U) /* finn: Evaluated from "((LW2080_CTRL_FLCN_USTREAMER_NUM_EVT_TYPES + 7) / 8)" */

/*!
 * uStreamer Event Filter type, stored as a bitmask.
 */
typedef struct LW2080_CTRL_FLCN_USTREAMER_EVENT_FILTER {
    LwU8 mask[LW2080_CTRL_FLCN_USTREAMER_MASK_SIZE_BYTES];
} LW2080_CTRL_FLCN_USTREAMER_EVENT_FILTER;

/*!
 * LW2080_CTRL_CMD_FLCN_USTREAMER_QUEUE_INFO
 * Get queue info for mapping / unmapping
 */
#define LW2080_CTRL_CMD_FLCN_USTREAMER_QUEUE_INFO (0x20803120) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_FLCN_INTERFACE_ID << 8) | LW2080_CTRL_FLCN_USTREAMER_QUEUE_INFO_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_FLCN_USTREAMER_QUEUE_INFO_PARAMS_MESSAGE_ID (0x20U)

typedef struct LW2080_CTRL_FLCN_USTREAMER_QUEUE_INFO_PARAMS {
    //! The engine type, from LW2080_ENGINE_TYPE_*
    LwU32 engine;

    //!
    // The page size of the requested queue in bytes.
    //
    LwU32 pageSize;

    //! Offset of the queue buffer in FB.
    LW_DECLARE_ALIGNED(LwUPtr offset, 8);

    //!
    // The size of the user-mapped instrumentation buffer. Measured in bytes.
    //
    LwU32 size;

    //!
    // The feature ID of the queue.
    //
    LwU8  queueFeatureId;
} LW2080_CTRL_FLCN_USTREAMER_QUEUE_INFO_PARAMS;

/*
 * LW2080_CTRL_CMD_FLCN_USTREAMER_CONTROL_GET/SET
 *
 * Get/set the event bitmask for the default queue.
 */
#define LW2080_CTRL_CMD_FLCN_USTREAMER_CONTROL_GET (0x20803122) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_FLCN_INTERFACE_ID << 8) | 0x22" */

#define LW2080_CTRL_CMD_FLCN_USTREAMER_CONTROL_SET (0x20803123) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_FLCN_INTERFACE_ID << 8) | 0x23" */

typedef struct LW2080_CTRL_FLCN_USTREAMER_CONTROL_PARAMS {
    //! The engine type, from LW2080_ENGINE_TYPE_*
    LwU32                                   engine;

    /*!
     * The bitmask of which event types to log. An event type corresponding to
     * a bit with a zero will be ignored at the log site, which prevents it
     * from filling up the resident buffer in the PMU. In general, set this to
     * only log the event types you actually want to use.
     * Refer to LWOS_BM_* in lwos_utility.h for usage.
     */
    LW2080_CTRL_FLCN_USTREAMER_EVENT_FILTER eventFilter;

    //! The queueId of the queue whose eventFilter we want to interact with
    LwU8                                    queueId;
} LW2080_CTRL_FLCN_USTREAMER_CONTROL_PARAMS;

/*
 * LW2080_CTRL_CMD_FLCN_GET_CTX_BUFFER_INFO
 *
 * This command provides the attributes of the falcon engine context buffer
 *
 *   hUserClient [IN]
 *     This parameter specifies the client handle that owns this channel.
 *   hChannel [IN]
 *     This parameter specifies the channel or channel group (TSG) handle
 *   alignment
 *     Specifies the alignment requirement for each context buffer
 *   size
 *     Aligned size of context buffer
 *   bufferHandle
 *     Opaque pointer to memdesc. Used by kernel clients for tracking purpose only.
 *   pageCount
 *     allocation size in the form of pageCount
 *   physAddr
 *     Physical address of the buffer first page
 *   aperture
 *     allocation aperture. Could be SYSMEM, VIDMEM, UNKNOWN
 *   kind
 *     PTE kind of this allocation.
 *   pageSize
 *     Page size of the buffer.
 *   bIsContigous
 *     States if physical allocation for this buffer is contiguous. PageSize will
 *     have no meaning if this flag is set.
 *   bDeviceDescendant
 *     TRUE if the allocation is a constructed under a Device or Subdevice.
 *   uuid
 *     SHA1 UUID of the Device or Subdevice. Valid when bDeviceDescendant is TRUE.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW2080_CTRL_CMD_FLCN_GET_CTX_BUFFER_INFO (0x20803124) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_FLCN_INTERFACE_ID << 8) | LW2080_CTRL_FLCN_GET_CTX_BUFFER_INFO_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_FLCN_GET_CTX_BUFFER_INFO_PARAMS_MESSAGE_ID (0x24U)

typedef struct LW2080_CTRL_FLCN_GET_CTX_BUFFER_INFO_PARAMS {
    LwHandle hUserClient;
    LwHandle hChannel;
    LW_DECLARE_ALIGNED(LwU64 alignment, 8);
    LW_DECLARE_ALIGNED(LwU64 size, 8);
    LW_DECLARE_ALIGNED(LwP64 bufferHandle, 8);
    LW_DECLARE_ALIGNED(LwU64 pageCount, 8);
    LW_DECLARE_ALIGNED(LwU64 physAddr, 8);
    LwU32    aperture;
    LwU32    kind;
    LwU32    pageSize;
    LwBool   bIsContigous;
    LwBool   bDeviceDescendant;
    LwU8     uuid[16];
} LW2080_CTRL_FLCN_GET_CTX_BUFFER_INFO_PARAMS;

// Aperture flags
#define LW2080_CTRL_FLCN_CTX_BUFFER_INFO_APERTURE_UNKNWON ADDR_UNKNOWN
#define LW2080_CTRL_FLCN_CTX_BUFFER_INFO_APERTURE_SYSMEM ADDR_SYSMEM
#define LW2080_CTRL_FLCN_CTX_BUFFER_INFO_APERTURE_FBMEM ADDR_FBMEM

/*
 * LW2080_CTRL_CMD_FLCN_GET_CTX_BUFFER_SIZE
 *
 * This command provides the size of the falcon engine context buffer
 *
 *   hChannel [IN]
 *     This parameter specifies the channel or channel group (TSG) handle
 *   totalBufferSize [OUT]
 *     This parameter returns the total context buffers size.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 */
#define LW2080_CTRL_CMD_FLCN_GET_CTX_BUFFER_SIZE (0x20803125) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_FLCN_INTERFACE_ID << 8) | LW2080_CTRL_FLCN_GET_CTX_BUFFER_SIZE_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_FLCN_GET_CTX_BUFFER_SIZE_PARAMS_MESSAGE_ID (0x25U)

typedef struct LW2080_CTRL_FLCN_GET_CTX_BUFFER_SIZE_PARAMS {
    LwHandle hChannel;
    LW_DECLARE_ALIGNED(LwU64 totalBufferSize, 8);
} LW2080_CTRL_FLCN_GET_CTX_BUFFER_SIZE_PARAMS;

typedef struct LW2080_CTRL_FLCN_PMU_RPC_STATS_RECORD {
    //! Success count tracking number of times RPC exelwtion has succeeded.
    LW_DECLARE_ALIGNED(LwU64 cntSuccess, 8);

    //! Failure count tracking number of times RPC exelwtion has failed.
    LW_DECLARE_ALIGNED(LwU64 cntError, 8);

    /*!
     * Total exelwtion time, used to compute average exelwtion time.
     *
     * @note    This field is updated only when @ref cntSuccess is increased.
     */
    LW_DECLARE_ALIGNED(LwU64 timeTotal, 8);

    //! Min recorded exelwtion time.

    LwU32 timeMin;

    //! Max recorded exelwtion time.
    LwU32 timeMax;
} LW2080_CTRL_FLCN_PMU_RPC_STATS_RECORD;

typedef struct LW2080_CTRL_CMD_FLCN_PMU_RPC_STATS {
    //! RPC exelwtion statistics tracking RM data.
    LW_DECLARE_ALIGNED(LW2080_CTRL_FLCN_PMU_RPC_STATS_RECORD rm, 8);

    //! RPC exelwtion statistics tracking PMU data.
    LW_DECLARE_ALIGNED(LW2080_CTRL_FLCN_PMU_RPC_STATS_RECORD pmu, 8);
} LW2080_CTRL_CMD_FLCN_PMU_RPC_STATS;

/*
 * LW2080_CTRL_CMD_FLCN_PMU_RPC_STATS_GET
 *
 * Get the PMU RPC stats saved by RM.
 */
#define LW2080_CTRL_CMD_FLCN_PMU_RPC_STATS_GET (0x20803126) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_FLCN_INTERFACE_ID << 8) | LW2080_CTRL_FLCN_PMU_RPC_STATS_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_FLCN_PMU_RPC_STATS_PARAMS_MESSAGE_ID (0x26U)

typedef struct LW2080_CTRL_FLCN_PMU_RPC_STATS_PARAMS {
    //! The engine type, from LW2080_ENGINE_TYPE_*
    LwU32 engine;

    //! The destination unit ID
    LwU32 unitId;

    //! RPC function ID
    LwU32 functionId;

    //! RPC stats for specified unit and function ID 
    LW_DECLARE_ALIGNED(LW2080_CTRL_CMD_FLCN_PMU_RPC_STATS rpcStats, 8);
} LW2080_CTRL_FLCN_PMU_RPC_STATS_PARAMS;

/* _ctrl2080flcn_h_ */
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)

