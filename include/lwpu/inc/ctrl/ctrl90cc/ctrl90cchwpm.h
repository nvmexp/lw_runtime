/* 
 * _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2017 by LWPU Corporation.  All rights reserved.  All
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
// Source file: ctrl/ctrl90cc/ctrl90cchwpm.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)




#include "ctrl/ctrl90cc/ctrl90ccbase.h"

/* GF100_PROFILER HWPM control commands and parameters */

/*
 * LW90CC_CTRL_CMD_HWPM_RESERVE
 *
 * This command attempts to reserve the perfmon for use by the calling client.
 * If this object was allocated as a child of a subdevice, then the
 * reservation will be global among all contexts on that subdevice. If this
 * object was allocated as a child of a channel group or a channel, then the
 * reservation will only be for the hardware context of that channel group or
 * channel.
 *
 * If the global reservation is held on a subdevice by another client, then
 * this command will fail, regardless of the parent class.
 *
 * If one or more per-context reservations are held by other clients, then
 * this command will fail if the parent object is a subdevice or another
 * client already holds the perfmon reservation for the parent context.
 *
 * This command will return LW_ERR_STATE_IN_USE for all of the failure
 * cases described above. A return status of LW_OK guarantees
 * that the client holds the perfmon reservation of the appropriate scope.
 *
 * This command does not take any parameters.
 *
 * Possible status values returned are:
 *    LW_OK
 *    LW_ERR_STATE_IN_USE
 */
#define LW90CC_CTRL_CMD_HWPM_RESERVE              (0x90cc0101) /* finn: Evaluated from "(FINN_GF100_PROFILER_HWPM_INTERFACE_ID << 8) | 0x1" */

/*
 * LW90CC_CTRL_CMD_HWPM_RELEASE
 *
 * This command releases an existing reservation of the perfmon for the
 * calling client. If the calling client does not lwrrently have the perfmon
 * reservation as acquired by LW90CC_CTRL_CMD_PROFILER_RESERVE_HWPM, this
 * command will return LW_ERR_ILWALID_REQUEST.
 *
 * This command does not take any parameters.
 *
 * Possible status values returned are:
 *    LW_OK
 *    LW_ERR_ILWALID_REQUEST
 */
#define LW90CC_CTRL_CMD_HWPM_RELEASE              (0x90cc0102) /* finn: Evaluated from "(FINN_GF100_PROFILER_HWPM_INTERFACE_ID << 8) | 0x2" */

/*
 * LW90CC_CTRL_CMD_HWPM_GET_RESERVATION_INFO
 *
 * This command returns information about any outstanding perfmon
 * reservations. If this object was allocated as a child of a subdevice, then
 * this command will return information about all reservations on the
 * subdevice (global or per-context). If this object was allocated as a child
 * of a channel group or channel, then this command will only return
 * information about the per-context reservation for that context or the
 * global reservation, if one exists.
 *
 *   reservationCount
 *     This parameter returns the number of outstanding perfmon reservations
 *     in the applicable scope. If the value of the bGlobal parameter is
 *     LW_TRUE, then this parameter will have a value of 1. If this object was
 *     allocated as a child of a channel group or channel, then this parameter
 *     will have a value of either 0 or 1. If this object was allocated as a
 *     child of a subdevice and the bGlobal parameter is LW_FALSE, then this
 *     parameter will return the number of per-context reservations on the
 *     subdevice.
 *   pid
 *     This parameter returns the PID of a process that holds a reservation in
 *     the applicable scope. If the value of the bGlobal parameter is LW_TRUE,
 *     then this parameter will be the PID of the process holding the global
 *     perfmon reservation on the subdevice. Otherwise, if the value of
 *     reservationCount is greater than 0, the value of this parameter will be
 *     the PID of one of the process that holds the per-context lock in the
 *     applicable scope. If the value of the reservationCount parameter is 0,
 *     the value of this parameter is undefined.
 *   bGlobal
 *     This parameter returns whether the outstanding perfmon reservation held
 *     by any client is global or per-context. If the value of this parameter
 *     is LW_TRUE, then the value of the reservationCount parameter should be 1
 *     and the value of the pid parameter should be the pid of the process
 *     that holds the global perfmon reservation. The value of this parameter
 *     will be LW_FALSE when there is no global perfmon reservation.
 *
 * Possible status values returned are:
 *    LW_OK
 *    LW_ERR_ILWALID_ARGUMENT
 */
#define LW90CC_CTRL_CMD_HWPM_GET_RESERVATION_INFO (0x90cc0103) /* finn: Evaluated from "(FINN_GF100_PROFILER_HWPM_INTERFACE_ID << 8) | LW90CC_CTRL_HWPM_GET_RESERVATION_INFO_PARAMS_MESSAGE_ID" */

#define LW90CC_CTRL_HWPM_GET_RESERVATION_INFO_PARAMS_MESSAGE_ID (0x3U)

typedef struct LW90CC_CTRL_HWPM_GET_RESERVATION_INFO_PARAMS {
    LwU32  reservationCount;
    LwU32  pid;
    LwBool bGlobal;
} LW90CC_CTRL_HWPM_GET_RESERVATION_INFO_PARAMS;

typedef enum LW90CC_CTRL_HWPM_COUNTER_ID {
    LW90CC_CTRL_HWPM_COUNTER_ID_EVENTCNT = 0,
    LW90CC_CTRL_HWPM_COUNTER_ID_TRIGGERCNT = 1,
    LW90CC_CTRL_HWPM_COUNTER_ID_THRESHCNT = 2,
    LW90CC_CTRL_HWPM_COUNTER_ID_SAMPLECNT = 3,
    LW90CC_CTRL_HWPM_COUNTER_ID_ELAPSED = 4,
    LW90CC_CTRL_HWPM_COUNTER_ID__COUNT = 5,
} LW90CC_CTRL_HWPM_COUNTER_ID;

typedef enum LW90CC_CTRL_HWPM_TRIGGER_TYPE {
    LW90CC_CTRL_HWPM_TRIGGER_TYPE_START_EXPERIMENT = 0,
    LW90CC_CTRL_HWPM_TRIGGER_TYPE_PMA_TESLA_MODE = 1,
    LW90CC_CTRL_HWPM_TRIGGER_TYPE_PMA_MIXED_MODE_START = 2,
    LW90CC_CTRL_HWPM_TRIGGER_TYPE_PMA_MIXED_MODE_END = 3,
} LW90CC_CTRL_HWPM_TRIGGER_TYPE;

/*
 * LW90CC_CTRL_CMD_HWPM_MODE_B_TRIGGER
 *
 * This command sets the PM trigger and emulates the actions in backend
 * which is performed by the hw when the trigger is set if ctx is resident.
 * This control call will check first if the context is resident/non-resident.
 * If the context is resident, then it will be a normal write to the trigger
 * register.
 * In case if the context is non-resident (and PM ctxsw is enabled), then the
 * following basic steps are taken (additional steps may be needed depending
 * on the trigger which is issued):
 *    - Copy values from the free running counters to the shadow ones in
 *      the PM buffer
 *    - Zero out the free running registers in the PM buffer
 *    - Return the final value of the counters read in counters array
 *
 * NOTE:
 *    This control call is to be used only in cases where PM Context-switch is 
 *    enabled. If this control call is used in global mode profiling, it will
 *    return LW_ERR_ILWALID_REQUEST
 *
 * [in] triggerType
 *    PM trigger type to apply. One of LW90CC_CTRL_HWPM_PERFMON_TRIGGER_TYPE
 *
 * [in] numPerfmons
 *    The number of valid elements in the _perfmons_ array
 *
 * The perfmons structure contains the following fields:
 *    [in] readMask
 *        One bit per counter register 
 *        i.e. (1 << LW90CC_CTRL_HWPM_PERFMON_COUNTER_ID).
 *        Only those counter slots will be read which are marked ON
 *    [in] controlOffset
 *        Uniquely identifies Perfmon. Assign to LW_PERF_PMM_CONTROL
 *    [out] counters[]
 *        Final counter values indexed by LW90CC_CTRL_HWPM_PERFMON_COUNTER_ID
 *
 * Possible status values returned are:
 *    LW_OK
 *    LW_ERR_ILWALID_ARGUMENT
 *    LW_ERR_ILWALID_REQUEST
 */

#define LW90CC_CTRL_CMD_HWPM_MODE_B_TRIGGER (0x90cc0104) /* finn: Evaluated from "(FINN_GF100_PROFILER_HWPM_INTERFACE_ID << 8) | LW90CC_CTRL_HWPM_MODE_B_TRIGGER_PARAMS_MESSAGE_ID" */

#define LW90CC_CTRL_HWPM_MAX_PERFMONS       256
typedef struct LW90CC_CTRL_HWPM_SAMPLE {
    LwU8  readMask;
    LwU32 controlOffset;
    LwU32 counters[LW90CC_CTRL_HWPM_COUNTER_ID__COUNT];
} LW90CC_CTRL_HWPM_SAMPLE;

#define LW90CC_CTRL_HWPM_MODE_B_TRIGGER_PARAMS_MESSAGE_ID (0x4U)

typedef struct LW90CC_CTRL_HWPM_MODE_B_TRIGGER_PARAMS {
    LW90CC_CTRL_HWPM_TRIGGER_TYPE triggerType;
    LwU32                         numPerfmons;
    LW90CC_CTRL_HWPM_SAMPLE       perfmons[LW90CC_CTRL_HWPM_MAX_PERFMONS];
} LW90CC_CTRL_HWPM_MODE_B_TRIGGER_PARAMS;

/* _ctrl90cchwpm_h_ */
