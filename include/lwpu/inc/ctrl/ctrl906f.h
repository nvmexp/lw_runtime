/*
* _LWRM_COPYRIGHT_BEGIN_
*
* Copyright 2007-2021 by LWPU Corporation.  All rights reserved.  All
* information contained herein is proprietary and confidential to LWPU
* Corporation.  Any use, reproduction, or disclosure without the written
* permission of LWPU Corporation is prohibited.
*
* _LWRM_COPYRIGHT_END_
*/

#pragma once

#include <lwtypes.h>
#if defined(_MSC_VER)
#pragma warning(disable:4324)
#endif

//
// This file was generated with FINN, an LWPU coding tool.
// Source file: ctrl/ctrl906f.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)




/* GF100_GPFIFO control commands and parameters */

#include "ctrl/ctrlxxxx.h"
#define LW906F_CTRL_CMD(cat,idx)  \
    LWXXXX_CTRL_CMD(0x906F, LW906F_CTRL_##cat, idx)

/* GF100_GPFIFO command categories (6bits) */
#define LW906F_CTRL_RESERVED (0x00)
#define LW906F_CTRL_GPFIFO   (0x01)
#define LW906F_CTRL_EVENT    (0x02)


/*
 * LW906F_CTRL_CMD_NULL
 *
 * This command does nothing.
 * This command does not take any parameters.
 *
 * Possible status values returned are:
 *   LW_OK
 */
#define LW906F_CTRL_CMD_NULL (0x906f0000) /* finn: Evaluated from "(FINN_GF100_CHANNEL_GPFIFO_RESERVED_INTERFACE_ID << 8) | 0x0" */






/*
 * LW906F_CTRL_GET_CLASS_ENGINEID
 *
 * Takes an object handle as input and returns
 * the Class and Engine that this object uses.
 *
 * hObject
 *   Handle to an object created. For example a
 *   handle to object of type FERMI_A created by
 *   the client. This is supplied by the client
 *   of this call.
 *
 * classEngineID
 *   A concatenation of class and engineid
 *   that the object with handle hObject
 *   belongs to. This is returned by RM. The internal
 *   format of this data structure is opaque to clients.
 *
 * classID
 *   ClassID for object represented by hObject
 *
 * engineID
 *   EngineID for object represented by hObject
 *
 * Possible status values returned are:
 *   LW_OK
 *   If the call was successful.
 *
 *   LW_ERR_ILWALID_OBJECT_HANDLE
 *   No object of handle hObject was found.
 */
#define LW906F_CTRL_GET_CLASS_ENGINEID (0x906f0101) /* finn: Evaluated from "(FINN_GF100_CHANNEL_GPFIFO_GPFIFO_INTERFACE_ID << 8) | LW906F_CTRL_GET_CLASS_ENGINEID_PARAMS_MESSAGE_ID" */

#define LW906F_CTRL_GET_CLASS_ENGINEID_PARAMS_MESSAGE_ID (0x1U)

typedef struct LW906F_CTRL_GET_CLASS_ENGINEID_PARAMS {
    LwHandle hObject;
    LwU32    classEngineID;
    LwU32    classID;
    LwU32    engineID;
} LW906F_CTRL_GET_CLASS_ENGINEID_PARAMS;

/*
 * LW906F_CTRL_RESET_CHANNEL
 *
 * This command resets the channel corresponding to specified engine and also
 * resets the specified engine.
 *
 * Takes an engine ID as input.
 *
 * engineID
 *   This parameter specifies the engine to be reset.  See the description of the
 *   LW2080_ENGINE_TYPE values in cl2080.h for more information.
 * subdeviceInstance
 *   This parameter specifies the subdevice to be reset when in SLI.
 * resetReason
 *   Specifies reason to reset a channel.
 *
 * Possible status values returned are:
 *   LW_OK
 */
#define LW906F_CTRL_CMD_RESET_CHANNEL_REASON_DEFAULT             0
#define LW906F_CTRL_CMD_RESET_CHANNEL_REASON_VERIF               1
#define LW906F_CTRL_CMD_RESET_CHANNEL_REASON_MMU_FLT             2
#define LW906F_CTRL_CMD_RESET_CHANNEL_REASON_ENUM_MAX            3
/*
 * Internal values for LW906F_CTRL_CMD_RESET_REASON. External values will be
 * checked and enforced to be < LW906F_CTRL_CMD_RESET_CHANNEL_REASON_ENUM_MAX
 */
#define LW906F_CTRL_CMD_INTERNAL_RESET_CHANNEL_REASON_FAKE_ERROR (0x4) /* finn: Evaluated from "LW906F_CTRL_CMD_RESET_CHANNEL_REASON_ENUM_MAX + 1" */


#define LW906F_CTRL_CMD_RESET_CHANNEL                            (0x906f0102) /* finn: Evaluated from "((FINN_GF100_CHANNEL_GPFIFO_GPFIFO_INTERFACE_ID << 8) | LW906F_CTRL_CMD_RESET_CHANNEL_PARAMS_MESSAGE_ID)" */

#define LW906F_CTRL_CMD_RESET_CHANNEL_PARAMS_MESSAGE_ID (0x2U)

typedef struct LW906F_CTRL_CMD_RESET_CHANNEL_PARAMS {
    LwU32 engineID;
    LwU32 subdeviceInstance;
    LwU32 resetReason;
} LW906F_CTRL_CMD_RESET_CHANNEL_PARAMS;

/*
 * LW906F_CTRL_CMD_EVENT_SET_NOTIFICATION
 *
 * This command sets event notification state for the associated channel.
 * This command requires that an instance of LW01_EVENT has been previously
 * bound to the associated channel object.
 *
 *   event
 *     This parameter specifies the type of event to which the specified
 *     action is to be applied.  This parameter must specify a valid
 *     LW906F_NOTIFIERS value (see cl906f.h for more details) and should
 *     not exceed one less LW906F_NOTIFIERS_MAXCOUNT.
 *   action
 *     This parameter specifies the desired event notification action.
 *     Valid notification actions include:
 *       LW906F_CTRL_SET_EVENT_NOTIFICATION_ACTION_DISABLE
 *         This action disables event notification for the specified
 *         event for the associated channel object.
 *       LW906F_CTRL_SET_EVENT_NOTIFICATION_ACTION_SINGLE
 *         This action enables single-shot event notification for the
 *         specified event for the associated channel object.
 *       LW906F_CTRL_SET_EVENT_NOTIFICATION_ACTION_REPEAT
 *         This action enables repeated event notification for the specified
 *         event for the associated channel object.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_ILWALID_STATE
 */


#define LW906F_CTRL_CMD_EVENT_SET_NOTIFICATION (0x906f0203) /* finn: Evaluated from "(FINN_GF100_CHANNEL_GPFIFO_EVENT_INTERFACE_ID << 8) | LW906F_CTRL_EVENT_SET_NOTIFICATION_PARAMS_MESSAGE_ID" */

#define LW906F_CTRL_EVENT_SET_NOTIFICATION_PARAMS_MESSAGE_ID (0x3U)

typedef struct LW906F_CTRL_EVENT_SET_NOTIFICATION_PARAMS {
    LwU32 event;
    LwU32 action;
} LW906F_CTRL_EVENT_SET_NOTIFICATION_PARAMS;

/* valid action values */
#define LW906F_CTRL_EVENT_SET_NOTIFICATION_ACTION_DISABLE (0x00000000)
#define LW906F_CTRL_EVENT_SET_NOTIFICATION_ACTION_SINGLE  (0x00000001)
#define LW906F_CTRL_EVENT_SET_NOTIFICATION_ACTION_REPEAT  (0x00000002)

/*
 * LW906F_CTRL_CMD_GET_DEFER_RC_STATE
 *
 * If SM Debugger is attached then on a MMU fault, RM defers the RC error
 * recovery and keeps a flag indicating that RC is deferred. This command
 * checks whether or not deferred RC is pending in RM for the associated
 * channel.
 *
 *   bDeferRCPending
 *     The output are TRUE and FALSE.
 *
 * Possible status values returned are:
 *   LW_OK
 */


#define LW906F_CTRL_CMD_GET_DEFER_RC_STATE (0x906f0105) /* finn: Evaluated from "(FINN_GF100_CHANNEL_GPFIFO_GPFIFO_INTERFACE_ID << 8) | LW906F_CTRL_CMD_GET_DEFER_RC_STATE_PARAMS_MESSAGE_ID" */

#define LW906F_CTRL_CMD_GET_DEFER_RC_STATE_PARAMS_MESSAGE_ID (0x5U)

typedef struct LW906F_CTRL_CMD_GET_DEFER_RC_STATE_PARAMS {
    LwBool bDeferRCPending;
} LW906F_CTRL_CMD_GET_DEFER_RC_STATE_PARAMS;

#define LW906F_CTRL_CMD_GET_MMU_FAULT_INFO                         (0x906f0106) /* finn: Evaluated from "(FINN_GF100_CHANNEL_GPFIFO_GPFIFO_INTERFACE_ID << 8) | LW906F_CTRL_GET_MMU_FAULT_INFO_PARAMS_MESSAGE_ID" */

/*
 * Shader types supported by MMU fault info
 * The types before compute shader refer to LW9097_SET_PIPELINE_SHADER_TYPE
 */
#define LW906F_CTRL_MMU_FAULT_SHADER_TYPE_VERTEX_LWLL_BEFORE_FETCH 0x00000000
#define LW906F_CTRL_MMU_FAULT_SHADER_TYPE_VERTEX                   0x00000001
#define LW906F_CTRL_MMU_FAULT_SHADER_TYPE_TESSELLATION_INIT        0x00000002
#define LW906F_CTRL_MMU_FAULT_SHADER_TYPE_TESSELLATION             0x00000003
#define LW906F_CTRL_MMU_FAULT_SHADER_TYPE_GEOMETRY                 0x00000004
#define LW906F_CTRL_MMU_FAULT_SHADER_TYPE_PIXEL                    0x00000005
#define LW906F_CTRL_MMU_FAULT_SHADER_TYPE_COMPUTE                  0x00000006
#define LW906F_CTRL_MMU_FAULT_SHADER_TYPES                         7

/*
 * LW906F_CTRL_CMD_GET_MMU_FAULT_INFO
 *
 * This command returns MMU fault information for a given channel. The MMU
 * fault information will be cleared once this command is exelwted.
 *
 *   addrHi - [out]
 *      Upper 32 bits of faulting address
 *   addrLo [out]
 *      Lower 32 bits of faulting address
 *   faultType [out]
 *      MMU fault type. Please see LW_PFIFO_INTR_MMU_FAULT_INFO_TYPE_* in
 *      dev_fifo.h for details about MMU fault type.
 *   faultString [out]
 *      String indicating the MMU fault type
 *   shaderProgramVA [out]
 *      an array of shader program virtual addresses to indicate faulted shaders in the pipeline
 *
 * Possible status values returned are:
 *   LW_OK
 */
#define LW906F_CTRL_MMU_FAULT_STRING_LEN                           32
#define LW906F_CTRL_GET_MMU_FAULT_INFO_PARAMS_MESSAGE_ID (0x6U)

typedef struct LW906F_CTRL_GET_MMU_FAULT_INFO_PARAMS {
    LwU32 addrHi;
    LwU32 addrLo;
    LwU32 faultType;
    char  faultString[LW906F_CTRL_MMU_FAULT_STRING_LEN];
    LW_DECLARE_ALIGNED(LwU64 shaderProgramVA[LW906F_CTRL_MMU_FAULT_SHADER_TYPES], 8);
} LW906F_CTRL_GET_MMU_FAULT_INFO_PARAMS;


/* _ctrl906f.h_ */
