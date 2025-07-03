/*
* _LWRM_COPYRIGHT_BEGIN_
*
* Copyright 2006-2021 by LWPU Corporation.  All rights reserved.  All
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
// Source file: ctrl/ctrl83de/ctrl83dedebug.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)




#include "ctrl/ctrl83de/ctrl83debase.h"
#include "lwstatus.h"

#include "ctrl/ctrl2080/ctrl2080gpu.h"

#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

/*
 * LW83DE_CTRL_CMD_SM_DEBUG_MODE_ENABLE
 *
 * The RmCtrl enables the debug mode for a given context.
 * When enabled:
 *  - The program exelwtion on a SM stops at breakpoints.
 *  - It allows the user to handle the RC recovery process and
 *    exceptions.  (Yet to be supported)
 *  - It allows the user to suspend, resume the context. (Yet to be supported)
 *
 * This command accepts no parameters.
 *
 *  Possible return values:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_ILWALID_OBJECT_HANDLE
 *   LW_ERR_ILWALID_CLIENT
 *   LW_ERR_OBJECT_NOT_FOUND
 *
 */
#define LW83DE_CTRL_CMD_SM_DEBUG_MODE_ENABLE     (0x83de0301) /* finn: Evaluated from "(FINN_GT200_DEBUGGER_DEBUG_INTERFACE_ID << 8) | 0x1" */

/*
 * LW83DE_CTRL_CMD_SM_DEBUG_MODE_DISABLE
 *
 * The RmCtrl disables the debug mode for a given context.
 * When disabled:
 *  - The program exelwtion on a SM ignores the breakpoints.
 *  - RC recovery process and exceptions are handled in the usual way.
 *  - A request to suspend, resume the context will return error
 *    LW_ERR_ILWALID_COMMAND.
 *
 * This command accepts no parameters.
 *
 *  Possible return values:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_ILWALID_OBJECT_HANDLE
 *   LW_ERR_ILWALID_CLIENT
 *   LW_ERR_OBJECT_NOT_FOUND
 */
#define LW83DE_CTRL_CMD_SM_DEBUG_MODE_DISABLE    (0x83de0302) /* finn: Evaluated from "(FINN_GT200_DEBUGGER_DEBUG_INTERFACE_ID << 8) | 0x2" */
/* LWRM_UNPUBLISHED */
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)



/*
 * LW83DE_CTRL_CMD_DEBUG_SET_MODE_MMU_DEBUG
 *
 * This command sets the MMU DEBUG mode.  This is Fermi-onwards feature.
 * If the query is made on an incorrect platform (for example, pre-Fermi)
 * the call will return with an LW_ERR_NOT_SUPPORTED error.
 *
 *   action
 *     The possible action values are:
 *   - LW83DE_CTRL_CMD_DEBUG_SET_MODE_MMU_DEBUG_ENABLE
 *      This enables the MMU debug mode if possible. If however, any another
 *      client has already disabled the mode (via LW83DE call) then this
 *      operation returns  LW_ERR_STATE_IN_USE.
 *
 *   - LW83DE_CTRL_CMD_DEBUG_SET_MODE_MMU_DEBUG_DISABLE
 *      This disables the MMU debug mode if possible. If however, any another
 *      client has already enabled the mode (via LW83DE call) then this
 *      operation returns  LW_ERR_STATE_IN_USE.
 *
 *   - LW83DE_CTRL_CMD_DEBUG_RELEASE_MMU_DEBUG_REQUESTS
 *      This operation releases all the client's outstanding requests to enable
 *      or disable the MMU debug mode.
 *
 * Possible return values:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_ILWALID_OBJECT_HANDLE
 *   LW_ERR_ILWALID_CLIENT
 *   LW_ERR_OBJECT_NOT_FOUND
 */
#define LW83DE_CTRL_CMD_DEBUG_SET_MODE_MMU_DEBUG (0x83de0307) /* finn: Evaluated from "(FINN_GT200_DEBUGGER_DEBUG_INTERFACE_ID << 8) | LW83DE_CTRL_DEBUG_SET_MODE_MMU_DEBUG_PARAMS_MESSAGE_ID" */

#define LW83DE_CTRL_DEBUG_SET_MODE_MMU_DEBUG_PARAMS_MESSAGE_ID (0x7U)

typedef struct LW83DE_CTRL_DEBUG_SET_MODE_MMU_DEBUG_PARAMS {
    LwU32 action;
} LW83DE_CTRL_DEBUG_SET_MODE_MMU_DEBUG_PARAMS;

#define LW83DE_CTRL_CMD_DEBUG_SET_MODE_MMU_DEBUG_ENABLE  (0x00000001)
#define LW83DE_CTRL_CMD_DEBUG_SET_MODE_MMU_DEBUG_DISABLE (0x00000002)
#define LW83DE_CTRL_CMD_DEBUG_RELEASE_MMU_DEBUG_REQUESTS (0x00000003)

/*
 * LW83DE_CTRL_CMD_DEBUG_GET_MODE_MMU_DEBUG
 *
 * This command gets the value of lwrrently configured MMU DEBUG mode.
 * This is Fermi-onwards feature. If the query is made on an incorrect
 * platform (for example, pre-Fermi) the call will return with an
 * LW_ERR_NOT_SUPPORTED error.
 *
 *   value
 *     This parameter returns the configured value.
 *
 * Possible return values:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_ILWALID_OBJECT_HANDLE
 *   LW_ERR_ILWALID_CLIENT
 *   LW_ERR_OBJECT_NOT_FOUND
 */
#define LW83DE_CTRL_CMD_DEBUG_GET_MODE_MMU_DEBUG         (0x83de0308) /* finn: Evaluated from "(FINN_GT200_DEBUGGER_DEBUG_INTERFACE_ID << 8) | LW83DE_CTRL_DEBUG_GET_MODE_MMU_DEBUG_PARAMS_MESSAGE_ID" */

#define LW83DE_CTRL_DEBUG_GET_MODE_MMU_DEBUG_PARAMS_MESSAGE_ID (0x8U)

typedef struct LW83DE_CTRL_DEBUG_GET_MODE_MMU_DEBUG_PARAMS {
    LwU32 value;
} LW83DE_CTRL_DEBUG_GET_MODE_MMU_DEBUG_PARAMS;

#define LW83DE_CTRL_CMD_DEBUG_GET_MODE_MMU_DEBUG_ENABLED  (0x00000001)
#define LW83DE_CTRL_CMD_DEBUG_GET_MODE_MMU_DEBUG_DISABLED (0x00000002)

/*
 * LW83DE_CTRL_CMD_DEBUG_SET_EXCEPTION_MASK
 *
 * This command allows the caller to filter events (which are also referred to
 * as "notifications", not to be confused with true notifiers), in the RM,
 * fairly close to the source of the events. In other words, depending on the
 * value of the exceptionMask, some events may not be raised.
 *
 * The original reason for creating this command is that the LWCA driver needs
 * to place the RM and the GPU(s) into SM debug mode, for some GPUs, in order to
 * activate various features and HW bug WARs. Being in SM debug mode has the
 * side effect of exposing the caller to debug events, which are generally
 * undesirable for the LWCA driver, but desirable for the LWCA debugger. This
 * command allows each client to receive only the events that it is
 * specifically interested in.
 *
 * If this command is never ilwoked, then the RM will behave as if
 * exceptionMask==LW83DE_CTRL_DEBUG_SET_EXCEPTION_MASK_ALL.
 *
 * As with many of the debugger features, this is Fermi-onwards feature. If this
 * API call is issued on an earlier platform, it will return an
 * LW_ERR_NOT_SUPPORTED error.
 *
 *   exceptionMask
 *     This identifies the category of notifications that the debug client
 *     is interested in.
 *
 *     Here are the allowed values for exceptionMask:
 *
 *     - LW83DE_CTRL_DEBUG_SET_EXCEPTION_MASK_FATAL
 *       This means that the caller wishes to receive events for any exceptions
 *       that are classified as fatal. For example,
 *       HWW_WARP_ESR_ERROR_PC_OVERFLOW is one such exception.
 *
 *       If any debug object, in any channel, has registered to receive events
 *       for _FATAL exceptions, then RC recovery will be deferred if such an
 *       exception oclwrs.
 *
 *       Also, if a client is registered for fatal exceptions, RC error recovery
 *       will be deferred.  If not registered for fatal exceptions, then fatal
 *       errors will (as usual) cause RC recovery to run immediately.
 *
 *     - LW83DE_CTRL_DEBUG_SET_EXCEPTION_MASK_TRAP
 *       This means that an event will be raised when an SM exelwtes a bpt.pause
 *       instruction. Note that on Fermi, the SM raises HWW when bpt.trap is
 *       exelwted as well, so this event will also be raised in that situation.
 *
 *     - LW83DE_CTRL_DEBUG_SET_EXCEPTION_MASK_SINGLE_STEP
 *       This means that an event will be raised a single step completion
 *       interrupt is received.
 *
 *     - LW83DE_CTRL_DEBUG_SET_EXCEPTION_MASK_INT
 *       This means that an event will be raised when an SM exelwtes a bpt.int
 *       instruction.
 *
 *     - LW83DE_CTRL_DEBUG_SET_EXCEPTION_MASK_NONE
 *       This means that no debug events will be raised.
 *
 *     - LW83DE_CTRL_DEBUG_SET_EXCEPTION_MASK_ALL
 *       This means that an event will be raised for any and all debug
 *       exceptions. This is the default behavior.
 *
 * Possible return values:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_ILWALID_OBJECT_HANDLE
 *   LW_ERR_ILWALID_CLIENT
 *   LW_ERR_OBJECT_NOT_FOUND
 */
#define LW83DE_CTRL_CMD_DEBUG_SET_EXCEPTION_MASK          (0x83de0309) /* finn: Evaluated from "(FINN_GT200_DEBUGGER_DEBUG_INTERFACE_ID << 8) | LW83DE_CTRL_DEBUG_SET_EXCEPTION_MASK_PARAMS_MESSAGE_ID" */

#define LW83DE_CTRL_DEBUG_SET_EXCEPTION_MASK_PARAMS_MESSAGE_ID (0x9U)

typedef struct LW83DE_CTRL_DEBUG_SET_EXCEPTION_MASK_PARAMS {
    LwU32 exceptionMask;
} LW83DE_CTRL_DEBUG_SET_EXCEPTION_MASK_PARAMS;

#define LW83DE_CTRL_DEBUG_SET_EXCEPTION_MASK_FATAL              (0x00000001)
#define LW83DE_CTRL_DEBUG_SET_EXCEPTION_MASK_TRAP               (0x00000002)
#define LW83DE_CTRL_DEBUG_SET_EXCEPTION_MASK_SINGLE_STEP        (0x00000004)
#define LW83DE_CTRL_DEBUG_SET_EXCEPTION_MASK_INT                (0x00000008)
#define LW83DE_CTRL_DEBUG_SET_EXCEPTION_MASK_CILP               (0x00000010)
#define LW83DE_CTRL_DEBUG_SET_EXCEPTION_MASK_PREEMPTION_STARTED (0x00000020)
#define LW83DE_CTRL_DEBUG_SET_EXCEPTION_MASK_NONE               (0x00000000)
#define LW83DE_CTRL_DEBUG_SET_EXCEPTION_MASK_ALL                (0x0000FFFF)

#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)


/*
 * LW83DE_CTRL_CMD_DEBUG_GET_EXCEPTION_MASK
 *
 * This command retrieves the exception mask that is in effect. Typically, that
 * mask would have been set by an earlier call to
 * LW83DE_CTRL_CMD_DEBUG_GET_EXCEPTION_MASK (above), but there is also a default
 * mask value that is set even if no call to _SET_ has been made.
 *
 * As with many of the debugger features, this is Fermi-onwards feature. If this
 * API call is issued on an earlier platform, it will return an
 * LW_ERR_NOT_SUPPORTED error.
 *
 *   exceptionMask
 *     This identifies the category of notifications that the debug client
 *     is interested in.
 *
 *     For the possible returned values, and meanings, of exceptionMask, please
 *     see LW83DE_CTRL_CMD_DEBUG_GET_EXCEPTION_MASK (above).
 *
 * Possible return values:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_ILWALID_OBJECT_HANDLE
 *   LW_ERR_ILWALID_CLIENT
 *   LW_ERR_OBJECT_NOT_FOUND
 */
#define LW83DE_CTRL_CMD_DEBUG_GET_EXCEPTION_MASK                (0x83de030a) /* finn: Evaluated from "(FINN_GT200_DEBUGGER_DEBUG_INTERFACE_ID << 8) | LW83DE_CTRL_DEBUG_GET_EXCEPTION_MASK_PARAMS_MESSAGE_ID" */

#define LW83DE_CTRL_DEBUG_GET_EXCEPTION_MASK_PARAMS_MESSAGE_ID (0xAU)

typedef struct LW83DE_CTRL_DEBUG_GET_EXCEPTION_MASK_PARAMS {
    LwU32 exceptionMask;
} LW83DE_CTRL_DEBUG_GET_EXCEPTION_MASK_PARAMS;

/* LWRM_UNPUBLISHED */
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)



/*
 * LW83DE_CTRL_CMD_READ_SINGLE_SM_ERROR_STATE
 *
 * This command reads the SM error state of a single SM. The error state
 * consists of several 32-bit values.
 *
 * Note that this acts upon the lwrrently resident GR (graphics) context. It is
 * up to the RM client to ensure that the desired GR context is resident, before
 * making this API call.
 *
 * See also: LW83DE_CTRL_CMD_READ_ALL_SM_ERROR_STATES.
 *
 * This is a Fermi-and-later feature. If this API call is issued on an earlier
 * platform, it will return an LW_ERR_NOT_SUPPORTED error.
 *
 * Parameters:
 *
 *   hTargetChannel (input)
 *     This identifies the channel.
 *
 *   smID (input)
 *     This identifies the SM. Allowed values are any valid SM ID. The RM
 *     grProgramSmIdNumbering_HAL() routines are a good place to look, in order
 *     to see how SM IDs are set up. The main idea is that the RM chooses a
 *     numbering scheme, and then informs the GPU hardware of that scheme, by
 *     actually recording each SM ID into the GPU, via a series of PRI (GPU
 *     register) writes.
 *
 *   smErrorState.hwwGlobalEsr (output)
 *     Value of the Global Error Status Register.
 *
 *   smErrorState.hwwWarpEsr (output)
 *     Value of the Warp Error Status Register.
 *
 *   smErrorState.hwwWarpEsrPc (output) : DEPRECATED for 64b PC below, will hold low 32b for now
 *     Value of the Warp Error Status Register Program Counter.
 *
 *   smErrorState.hwwGlobalEsrReportMask (output)
 *     Value of the Global Error Status Register Report Mask.
 *
 *   smErrorState.hwwWarpEsrReportMask (output)
 *     Value of the Error Status Register Report Mask.
 *
 *   smErrorState.hwwWarpEsrPc64 (output)
 *     Value of the 64b Warp Error Status Register Program Counter.
 *
 * Possible return values:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_ILWALID_OBJECT_HANDLE
 *   LW_ERR_ILWALID_CLIENT
 *   LW_ERR_OBJECT_NOT_FOUND
 */
#define LW83DE_CTRL_CMD_DEBUG_READ_SINGLE_SM_ERROR_STATE (0x83de030b) /* finn: Evaluated from "(FINN_GT200_DEBUGGER_DEBUG_INTERFACE_ID << 8) | LW83DE_CTRL_DEBUG_READ_SINGLE_SM_ERROR_STATE_PARAMS_MESSAGE_ID" */

typedef struct LW83DE_SM_ERROR_STATE_REGISTERS {
    LwU32 hwwGlobalEsr;
    LwU32 hwwWarpEsr;
    LwU32 hwwWarpEsrPc;
    LwU32 hwwGlobalEsrReportMask;
    LwU32 hwwWarpEsrReportMask;
    LW_DECLARE_ALIGNED(LwU64 hwwEsrAddr, 8);
    LW_DECLARE_ALIGNED(LwU64 hwwWarpEsrPc64, 8);
    LwU32 hwwCgaEsr;
    LwU32 hwwCgaEsrReportMask;
} LW83DE_SM_ERROR_STATE_REGISTERS;

#define LW83DE_CTRL_DEBUG_READ_SINGLE_SM_ERROR_STATE_PARAMS_MESSAGE_ID (0xBU)

typedef struct LW83DE_CTRL_DEBUG_READ_SINGLE_SM_ERROR_STATE_PARAMS {
    LwHandle hTargetChannel;
    LwU32    smID;
    LW_DECLARE_ALIGNED(LW83DE_SM_ERROR_STATE_REGISTERS smErrorState, 8);
} LW83DE_CTRL_DEBUG_READ_SINGLE_SM_ERROR_STATE_PARAMS;

/*
 * LW83DE_CTRL_CMD_READ_ALL_SM_ERROR_STATES
 *
 * This command reads the SM error state of all SMs.
 *
 * Note that this acts upon the lwrrently resident GR (graphics) context. It is
 * up to the RM client to ensure that the desired GR context is resident, before
 * making this API call.
 *
 * Parameters:
 *
 *   hTargetChannel (input)
 *     This identifies the channel.
 *
 *   numSMsToRead (input)
 *     This should be set to the number of SMs that the RM is supposed to read.
 *     It will typically be the total number of SMs in the GPU. For best
 *     results, you should not pass in a value that is greater than the number
 *     of SMs that the GPU actually contains.
 *
 *   startingSM (input)
 *      This should be set to the starting index of the first SM to read.
 *      Clients may use this to read data from SMs beyond the maximum specified
 *      in LW83DE_CTRL_DEBUG_MAX_SMS_PER_CALL.
 *
 *   smErrorStateArray (output)
 *     This is an array of LW83DE_SM_ERROR_STATE_REGISTERS structs. Please see
 *     the description of LW83DE_CTRL_CMD_READ_SINGLE_SM_ERROR_STATE, above, for
 *     a description of the individual fields.
 *
 *   mmuFault.valid (output)
 *     This is LW_TRUE if an MMU fault oclwrred on the target channel since the last call to
 *     LW83DE_CTRL_CMD_CLEAR_ALL_SM_ERROR_STATES to this channel.
 *
 *   mmuFault.faultInfo (output)
 *     This is the value of the first LW_PFIFO_INTR_MMU_FAULT_INFO that caused the MMU fault.
 *
 *   mmuFaultInfo (output)
 *     Deprecated field, see mmuFault.faultInfo
 *
 * This is a Fermi-and-later feature. If this API call is issued on an earlier
 * platform, it will return an LW_ERR_NOT_SUPPORTED error.
 *
 * Possible return values:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_ILWALID_OBJECT_HANDLE
 *   LW_ERR_ILWALID_CLIENT
 *   LW_ERR_OBJECT_NOT_FOUND
 */
#define LW83DE_CTRL_CMD_DEBUG_READ_ALL_SM_ERROR_STATES (0x83de030c) /* finn: Evaluated from "(FINN_GT200_DEBUGGER_DEBUG_INTERFACE_ID << 8) | LW83DE_CTRL_DEBUG_READ_ALL_SM_ERROR_STATES_PARAMS_MESSAGE_ID" */

#define LW83DE_CTRL_DEBUG_MAX_SMS_PER_CALL             100

typedef struct LW83DE_MMU_FAULT_INFO {
    LwBool valid;
    LwU32  faultInfo;
} LW83DE_MMU_FAULT_INFO;

#define LW83DE_CTRL_DEBUG_READ_ALL_SM_ERROR_STATES_PARAMS_MESSAGE_ID (0xLW)

typedef struct LW83DE_CTRL_DEBUG_READ_ALL_SM_ERROR_STATES_PARAMS {
    LwHandle              hTargetChannel;
    LwU32                 numSMsToRead;
    LW_DECLARE_ALIGNED(LW83DE_SM_ERROR_STATE_REGISTERS smErrorStateArray[LW83DE_CTRL_DEBUG_MAX_SMS_PER_CALL], 8);
    LwU32                 mmuFaultInfo;       // Deprecated, use mmuFault field instead
    LW83DE_MMU_FAULT_INFO mmuFault;
    LwU32                 startingSM;
} LW83DE_CTRL_DEBUG_READ_ALL_SM_ERROR_STATES_PARAMS;

/*
 * LW83DE_CTRL_CMD_CLEAR_SINGLE_SM_ERROR_STATE
 *
 * This command clears the SM error state of a single SM. The error state
 * consists of several 32-bit values.
 *
 * Note that this acts upon the lwrrently resident GR (graphics) context. It is
 * up to the RM client to ensure that the desired GR context is resident, before
 * making this API call.
 *
 * See also: LW83DE_CTRL_CMD_CLEAR_ALL_SM_ERROR_STATES.
 *
 * This API call has a slightly different effect than what would occur as a
 * result of issuing a read-modify-write via _READ_SINGLE_SM_ERROR_STATE and
 * _WRITE_SINGLE_SM_ERROR_STATE. The difference arises due to the fact that RM
 * is caching the error state, to compensate for the fact that the real GPU
 * error state must be cleared very early on in the exception handling routine.
 *
 * In other words, the _READ data is stale by design, and cannot be used in a
 * read-modify-write routine from user space. Therefore, in order to clear the
 * SM error state, a separate RM API call is required.
 *
 * This is a Fermi-and-later feature. If this API call is issued on an earlier
 * platform, it will return an LW_ERR_NOT_SUPPORTED error.
 *
 * Parameters:
 *
 *   hTargetChannel (input)
 *     This identifies the channel.
 *
 *   smID (input)
 *     This identifies the SM. Allowed values are any valid SM ID. Please see
 *     LW83DE_CTRL_CMD_READ_SINGLE_SM_ERROR_STATE for further details.
 *
 * Possible return values:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_ILWALID_OBJECT_HANDLE
 *   LW_ERR_ILWALID_CLIENT
 *   LW_ERR_OBJECT_NOT_FOUND
 */
#define LW83DE_CTRL_CMD_DEBUG_CLEAR_SINGLE_SM_ERROR_STATE (0x83de030f) /* finn: Evaluated from "(FINN_GT200_DEBUGGER_DEBUG_INTERFACE_ID << 8) | LW83DE_CTRL_DEBUG_CLEAR_SINGLE_SM_ERROR_STATE_PARAMS_MESSAGE_ID" */

#define LW83DE_CTRL_DEBUG_CLEAR_SINGLE_SM_ERROR_STATE_PARAMS_MESSAGE_ID (0xFU)

typedef struct LW83DE_CTRL_DEBUG_CLEAR_SINGLE_SM_ERROR_STATE_PARAMS {
    LwHandle hTargetChannel;
    LwU32    smID;
} LW83DE_CTRL_DEBUG_CLEAR_SINGLE_SM_ERROR_STATE_PARAMS;

/*
 * LW83DE_CTRL_CMD_CLEAR_ALL_SM_ERROR_STATES
 *
 * This command clears the SM error state of all SMs.
 *
 * Note that this acts upon the lwrrently resident GR (graphics) context. It is
 * up to the RM client to ensure that the desired GR context is resident, before
 * making this API call.
 *
 * Parameters:
 *
 *   hTargetChannel (input)
 *     This identifies the channel.
 *
 *   numSMsToClear (input)
 *     This should be set to the number of SMs that the RM is supposed to write.
 *     It will typically be the total number of SMs in the GPU. For best
 *     results, you should not pass in a value that is greater than the number
 *     of SMs that the GPU actually contains.
 *
 *     Please see the description of
 *     LW83DE_CTRL_CMD_CLEAR_SINGLE_SM_ERROR_STATE, above, for a description of
 *     why these two _CLEAR API calls are required.
 *
 * This is a Fermi-and-later feature. If this API call is issued on an earlier
 * platform, it will return an LW_ERR_NOT_SUPPORTED error.
 *
 * Possible return values:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_ILWALID_OBJECT_HANDLE
 *   LW_ERR_ILWALID_CLIENT
 *   LW_ERR_OBJECT_NOT_FOUND
 */
#define LW83DE_CTRL_CMD_DEBUG_CLEAR_ALL_SM_ERROR_STATES (0x83de0310) /* finn: Evaluated from "(FINN_GT200_DEBUGGER_DEBUG_INTERFACE_ID << 8) | LW83DE_CTRL_DEBUG_CLEAR_ALL_SM_ERROR_STATES_PARAMS_MESSAGE_ID" */

#define LW83DE_CTRL_DEBUG_CLEAR_ALL_SM_ERROR_STATES_PARAMS_MESSAGE_ID (0x10U)

typedef struct LW83DE_CTRL_DEBUG_CLEAR_ALL_SM_ERROR_STATES_PARAMS {
    LwHandle hTargetChannel;
    LwU32    numSMsToClear;
} LW83DE_CTRL_DEBUG_CLEAR_ALL_SM_ERROR_STATES_PARAMS;

#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

/*
 * LW83DE_CTRL_CMD_DEBUG_SUSPEND_ALL_CONTEXTS_FOR_CLIENT
 *
 * This command has been removed, and is no longer implemented.
 *
 * Possible return values:
 *   LW_ERR_NOT_SUPPORTED
 */
#define LW83DE_CTRL_CMD_DEBUG_SUSPEND_ALL_CONTEXTS_FOR_CLIENT                      (0x83de0311) /* finn: Evaluated from "(FINN_GT200_DEBUGGER_DEBUG_INTERFACE_ID << 8) | 0x11" */
/* LWRM_UNPUBLISHED */
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)



#define LW83DE_CTRL_CMD_DEBUG_SUSPEND_ALL_CONTEXTS_FOR_CLIENT_PARAMS_DEFINED       1
#define LW83DE_CTRL_CMD_DEBUG_SUSPEND_ALL_CONTEXTS_FOR_CLIENT_HAS_RESIDENT_CHANNEL 1
typedef struct LW83DE_CTRL_CMD_DEBUG_SUSPEND_ALL_CONTEXTS_FOR_CLIENT_PARAMS {
    LwU32    waitForEvent;
    LwHandle hResidentChannel;
} LW83DE_CTRL_CMD_DEBUG_SUSPEND_ALL_CONTEXTS_FOR_CLIENT_PARAMS;

#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

/*
 * LW83DE_CTRL_CMD_DEBUG_RESUME_ALL_CONTEXTS_FOR_CLIENT
 *
 * This command has been removed, and is no longer implemented.
 *
 * Possible return values:
 *   LW_ERR_NOT_SUPPORTED
 */
#define LW83DE_CTRL_CMD_DEBUG_RESUME_ALL_CONTEXTS_FOR_CLIENT (0x83de0312) /* finn: Evaluated from "(FINN_GT200_DEBUGGER_DEBUG_INTERFACE_ID << 8) | 0x12" */
/* LWRM_UNPUBLISHED */
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)



/*
 * LW83DE_CTRL_CMD_DEBUG_SET_NEXT_STOP_TRIGGER_TYPE
 *
 * This command set the type of action we want on RM encountering an error
 * and issuing a STOP_TRIGGER. The action will be to either braodcast the
 * STOP_TRIGGER to all SM's, or just send to the SM hitting an exception.
 *
 *   stopTriggerType
 *     This identifies trigger type to initiate.
 *
 *     Here are the allowed values for stopTriggerType:
 *
 *     - LW83DE_CTRL_DEBUG_SET_NEXT_STOP_TRIGGER_TYPE_SINGLE_SM
 *       This means that we will issue STOP_TRIGGER to the single SM
 *       noted in the exception
 *
 *     - LW83DE_CTRL_DEBUG_SET_NEXT_STOP_TRIGGER_TYPE_BROADCAST
 *       This means that we will issue STOP_TRIGGER to all SM's
 *
 * Possible return values:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_ILWALID_OBJECT_HANDLE
 *   LW_ERR_ILWALID_CLIENT
 *   LW_ERR_OBJECT_NOT_FOUND
 */
#define LW83DE_CTRL_CMD_DEBUG_SET_NEXT_STOP_TRIGGER_TYPE     (0x83de0313) /* finn: Evaluated from "(FINN_GT200_DEBUGGER_DEBUG_INTERFACE_ID << 8) | LW83DE_CTRL_DEBUG_SET_NEXT_STOP_TRIGGER_TYPE_PARAMS_MESSAGE_ID" */

#define LW83DE_CTRL_DEBUG_SET_NEXT_STOP_TRIGGER_TYPE_PARAMS_MESSAGE_ID (0x13U)

typedef struct LW83DE_CTRL_DEBUG_SET_NEXT_STOP_TRIGGER_TYPE_PARAMS {
    LwU32 stopTriggerType;
} LW83DE_CTRL_DEBUG_SET_NEXT_STOP_TRIGGER_TYPE_PARAMS;

#define LW83DE_CTRL_DEBUG_SET_NEXT_STOP_TRIGGER_TYPE_SINGLE_SM   (0x00000001)
#define LW83DE_CTRL_DEBUG_SET_NEXT_STOP_TRIGGER_TYPE_BROADCSAT   (0x00000002)

/*
 * LW83DE_CTRL_CMD_DEBUG_SET_SINGLE_STEP_INTERRUPT_HANDLING
 *
 * This command sets the type of action we want on RM encountering a
 * SINGLE_STEP exception while in CILP debug mode.  In the normal case,
 * non-pausing, we ignore these exceptions as on prior chips.  When the
 * user selects pausing, it will cause the exception to be treated just
 * like we had seen an SM error or BPT_PAUSE.
 *
 *   singleStepHandling
 *     This identifies single step handling type to use.
 *
 *     Here are the allowed values for singleStepHandling:
 *
 *     - LW83DE_CTRL_DEBUG_SET_SINGLE_STEP_INTERRUPT_HANDLING_NONPAUSING
 *       Treat SINGLE_STEP exceptions while in debug mode as non-pausing,
 *       which is the default/normal mode in the interrupt pre-process
 *       function, where they are ignored.
 *
 *     - LW83DE_CTRL_DEBUG_SET_SINGLE_STEP_INTERRUPT_HANDLING_PAUSING
 *       Treat SINGLE_STEP exceptions while in debug mode as pausing,
 *       which means in the interrupt pre-process function they will
 *       be treated like BPT_PAUSE and SM error exceptions
 *
 * Possible return values:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_ILWALID_OBJECT_HANDLE
 *   LW_ERR_ILWALID_CLIENT
 *   LW_ERR_OBJECT_NOT_FOUND
 */
#define LW83DE_CTRL_CMD_DEBUG_SET_SINGLE_STEP_INTERRUPT_HANDLING (0x83de0314) /* finn: Evaluated from "(FINN_GT200_DEBUGGER_DEBUG_INTERFACE_ID << 8) | LW83DE_CTRL_DEBUG_SET_SINGLE_STEP_INTERRUPT_HANDLING_PARAMS_MESSAGE_ID" */

#define LW83DE_CTRL_DEBUG_SET_SINGLE_STEP_INTERRUPT_HANDLING_PARAMS_MESSAGE_ID (0x14U)

typedef struct LW83DE_CTRL_DEBUG_SET_SINGLE_STEP_INTERRUPT_HANDLING_PARAMS {
    LwU32 singleStepHandling;
} LW83DE_CTRL_DEBUG_SET_SINGLE_STEP_INTERRUPT_HANDLING_PARAMS;

#define LW83DE_CTRL_DEBUG_SET_SINGLE_STEP_INTERRUPT_HANDLING_NONPAUSING (0x00000001)
#define LW83DE_CTRL_DEBUG_SET_SINGLE_STEP_INTERRUPT_HANDLING_PAUSING    (0x00000002)

/*
 * LW83DE_CTRL_CMD_DEBUG_READ_MEMORY
 *
 *   This command reads a block of memory.
 *   This command is deprecated in favor of LW83DE_CTRL_CMD_DEBUG_READ_BATCH_MEMORY
 *
 *   hMemory [IN]
 *       The handle to the memory being accessed. If hMemory is not accessible
 *       from the caller's address space, LW_ERR_INSUFFICIENT_PERMISSIONS
 *       is returned.
 *
 *   length [IN/OUT]
 *       Number of bytes to read, as well as the number of bytes actually read
 *       returned.
 *
 *   offset [IN]
 *      The offset into the physical memory region given by the handle above.
 *
 *   buffer [OUT]
 *       The data read is returned in this buffer.
 *
 * Possible status values returned are
 *    LW_OK
 *    LW_ERR_ILWALID_ARGUMENT
 *    LW_ERR_NOT_SUPPORTED
 *    LW_ERR_ILWALID_ACCESS_TYPE
 *    LW_ERR_INSUFFICIENT_PERMISSIONS
 *
 */
#define LW83DE_CTRL_CMD_DEBUG_READ_MEMORY                               (0x83de0315) /* finn: Evaluated from "(FINN_GT200_DEBUGGER_DEBUG_INTERFACE_ID << 8) | LW83DE_CTRL_DEBUG_READ_MEMORY_PARAMS_MESSAGE_ID" */

#define LW83DE_CTRL_DEBUG_READ_MEMORY_PARAMS_MESSAGE_ID (0x15U)

typedef struct LW83DE_CTRL_DEBUG_READ_MEMORY_PARAMS {
    LwU32 hMemory;
    LwU32 length;
    LW_DECLARE_ALIGNED(LwU64 offset, 8);
    LW_DECLARE_ALIGNED(LwP64 buffer, 8);
} LW83DE_CTRL_DEBUG_READ_MEMORY_PARAMS;

/*
 * LW83DE_CTRL_CMD_DEBUG_WRITE_MEMORY
 *
 *   This command writes a block of memory.
 *   This command is deprecated in favor of LW83DE_CTRL_CMD_DEBUG_WRITE_BATCH_MEMORY
 *
 *   hMemory [IN]
 *       The handle to the memory being accessed. If hMemory is not accessible
 *       from the caller's address space, LW_ERR_INSUFFICIENT_PERMISSIONS
 *       is returned.
 *
 *   length [IN/OUT]
 *       Number of bytes to write, as well as the number of bytes actually
 *       written.
 *
 *   offset [IN]
 *      The offset into the physical memory region given by the handle above.
 *
 *   buffer [IN]
 *       The data to be written is sent in this buffer.
 *
 * Possible status values returned are
 *    LW_OK
 *    LW_ERR_ILWALID_ARGUMENT
 *    LW_ERR_NOT_SUPPORTED
 *    LW_ERR_ILWALID_ACCESS_TYPE
 *    LW_ERR_INSUFFICIENT_PERMISSIONS
 */
#define LW83DE_CTRL_CMD_DEBUG_WRITE_MEMORY (0x83de0316) /* finn: Evaluated from "(FINN_GT200_DEBUGGER_DEBUG_INTERFACE_ID << 8) | LW83DE_CTRL_DEBUG_WRITE_MEMORY_PARAMS_MESSAGE_ID" */

#define LW83DE_CTRL_DEBUG_WRITE_MEMORY_PARAMS_MESSAGE_ID (0x16U)

typedef struct LW83DE_CTRL_DEBUG_WRITE_MEMORY_PARAMS {
    LwU32 hMemory;
    LwU32 length;
    LW_DECLARE_ALIGNED(LwU64 offset, 8);
    LW_DECLARE_ALIGNED(LwP64 buffer, 8);
} LW83DE_CTRL_DEBUG_WRITE_MEMORY_PARAMS;

/*
 * LW83DE_CTRL_CMD_DEBUG_SUSPEND_CONTEXT
 *
 * This command suspends a SM context associated with the debugger object.
 *
 * When the suspend call returns, context associated with the debugger object
 * should not be actively exelwting any code on any SM. The channel will have
 * been disabled if not resident on GR, or have its SM suspended if it was resident.
 *
 * This is a Fermi-and-later feature. If this API call is issued on an earlier
 * platform, it will return an LW_ERR_NOT_SUPPORTED error.
 *
 *   waitForEvent
 *     This return param indicates if the call had to issue a Preempt,
 *     therefore it is in process and user may need to wait for it.
 *
 * Possible return values:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_ILWALID_OBJECT_HANDLE
 *   LW_ERR_ILWALID_CLIENT
 *   LW_ERR_OBJECT_NOT_FOUND
 *   LW_ERR_ILWALID_STATE
 */
#define LW83DE_CTRL_CMD_DEBUG_SUSPEND_CONTEXT (0x83de0317) /* finn: Evaluated from "(FINN_GT200_DEBUGGER_DEBUG_INTERFACE_ID << 8) | LW83DE_CTRL_CMD_DEBUG_SUSPEND_CONTEXT_PARAMS_MESSAGE_ID" */

#define LW83DE_CTRL_CMD_DEBUG_SUSPEND_CONTEXT_PARAMS_MESSAGE_ID (0x17U)

typedef LW83DE_CTRL_CMD_DEBUG_SUSPEND_ALL_CONTEXTS_FOR_CLIENT_PARAMS LW83DE_CTRL_CMD_DEBUG_SUSPEND_CONTEXT_PARAMS;

/*
 * LW83DE_CTRL_CMD_DEBUG_RESUME_CONTEXT
 *
 * This command safely resumes the SM context associated with the debugger object.
 *
 * This is a Fermi-and-later feature. If this API call is issued on an earlier
 * platform, it will return an LW_ERR_NOT_SUPPORTED error.
 *
 * Possible return values:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_ILWALID_OBJECT_HANDLE
 *   LW_ERR_ILWALID_CLIENT
 *   LW_ERR_OBJECT_NOT_FOUND
 *   LW_ERR_ILWALID_STATE
 */
#define LW83DE_CTRL_CMD_DEBUG_RESUME_CONTEXT (0x83de0318) /* finn: Evaluated from "(FINN_GT200_DEBUGGER_DEBUG_INTERFACE_ID << 8) | 0x18" */

/*
 * LW83DE_CTRL_CMD_DEBUG_GET_HANDLES
 *
 * This command returns relevant handles for the debug object
 * This command is only available on debug and develop builds.
 *
 * Possible return values:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_ILWALID_CLIENT
 *   LW_ERR_ILWALID_OBJECT_HANDLE
 *
 */
#define LW83DE_CTRL_CMD_DEBUG_GET_HANDLES    (0x83de0319) /* finn: Evaluated from "(FINN_GT200_DEBUGGER_DEBUG_INTERFACE_ID << 8) | 0x19" */

typedef struct LW83DE_CTRL_DEBUG_GET_HANDLES_PARAMS {
    LwHandle hChannel;
    LwHandle hSubdevice;
} LW83DE_CTRL_DEBUG_GET_HANDLES_PARAMS;

/*
 * LW83DE_CTRL_CMD_READ_SURFACE
 *
 * This command allows the caller to copy the data from a specified gpuVA
 * to a usermode buffer. Before copying, this command validates whether or
 * not the virtual address (VA) range provided as input has valid and allocated
 * pages mapped to it in its entirety.
 *
 * This command's input is LW83DE_CTRL_DEBUG_ACCESS_SURFACE_PARAMETERS which
 * contains a buffer of LW83DE_CTRL_DEBUG_ACCESS_OPs
 *
 * Possible return values:
 *     LW_OK
 *     LW_ERR_ILWALID_ARGUMENT
 *     LW_ERR_ILWALID_XLATE
 */
#define LW83DE_CTRL_CMD_READ_SURFACE  (0x83de031a) /* finn: Evaluated from "(FINN_GT200_DEBUGGER_DEBUG_INTERFACE_ID << 8) | 0x1A" */

/*
 * LW83DE_CTRL_CMD_WRITE_SURFACE
 *
 * This command allows the caller to copy the data from a provided usermode
 * buffer to a specified GPU VA. Before copying, this command validates whether or
 * not the virtual address (VA) range provided as input has valid and allocated
 * pages mapped to it in its entirety.
 *
 * This command's input is LW83DE_CTRL_DEBUG_ACCESS_SURFACE_PARAMETERS which
 * contains a buffer of LW83DE_CTRL_DEBUG_ACCESS_OPs
 *
 * Possible return values:
 *     LW_OK
 *     LW_ERR_ILWALID_ARGUMENT
 *     LW_ERR_ILWALID_XLATE
 */
#define LW83DE_CTRL_CMD_WRITE_SURFACE (0x83de031b) /* finn: Evaluated from "(FINN_GT200_DEBUGGER_DEBUG_INTERFACE_ID << 8) | 0x1B" */

#define MAX_ACCESS_OPS                64

typedef struct LW83DE_CTRL_DEBUG_ACCESS_OP {
    LW_DECLARE_ALIGNED(LwU64 gpuVA, 8);   // IN
    LW_DECLARE_ALIGNED(LwP64 pCpuVA, 8);  // IN/OUT  Debugger CPU Pointer of buffer
    LwU32 size;                           // IN      Size in bytes
    LwU32 valid;                          // OUT     Whether the GpuVA is accessible
} LW83DE_CTRL_DEBUG_ACCESS_OP;

typedef struct LW83DE_CTRL_DEBUG_ACCESS_SURFACE_PARAMETERS {
    LwU32 count;          // IN  Number of ops in this call
    LW_DECLARE_ALIGNED(LW83DE_CTRL_DEBUG_ACCESS_OP opsBuffer[MAX_ACCESS_OPS], 8);
} LW83DE_CTRL_DEBUG_ACCESS_SURFACE_PARAMETERS;

/*
 * LW83DE_CTRL_CMD_GET_MAPPINGS
 *
 * This command traverses through the virtual memory page hierarchy and
 * fetches the valid virtual mappings and their sizes for a provided virtual
 * address (VA) range.
 * If a given VA range has more than MAX_GET_MAPPINGS_OPS valid mappings,
 * hasMore is set to 1, and opsBuffer is still filled with MAX_GET_MAPPINGS_OPS
 * valid mappings. In this case, this command should be called again with
 *     vaLo = opsBuffer[MAX_GET_MAPPINGS_OPS - 1].gpuVA +
 *            opsBuffer[MAX_GET_MAPPINGS_OPS - 1].size;
 * and vaHi set to the next desired upper limit.
 *
 * This command's input is LW83DE_CTRL_DEBUG_GET_MAPPINGS_PARAMETERS which
 * contains a buffer of LW83DE_CTRL_DEBUG_GET_MAPPINGS_OP
 *
 * Possible return values:
 *     LW_OK
 *     LW_ERR_ILWALID_ARGUMENT
 *     LW_ERR_ILWALID_XLATE
 */
#define LW83DE_CTRL_CMD_GET_MAPPINGS (0x83de031c) /* finn: Evaluated from "(FINN_GT200_DEBUGGER_DEBUG_INTERFACE_ID << 8) | 0x1C" */

#define MAX_GET_MAPPINGS_OPS         64

typedef struct LW83DE_CTRL_DEBUG_GET_MAPPINGS_OP {
    LW_DECLARE_ALIGNED(LwU64 gpuVA, 8);  // OUT     Start of GPU VA for this mapping
    LwU32 size;                          // OUT     Size in bytes of this mapping
} LW83DE_CTRL_DEBUG_GET_MAPPINGS_OP;

typedef struct LW83DE_CTRL_DEBUG_GET_MAPPINGS_PARAMETERS {
    LW_DECLARE_ALIGNED(LwU64 vaLo, 8);   // IN      Lower VA range, inclusive
    LW_DECLARE_ALIGNED(LwU64 vaHi, 8);   // IN      Upper VA range, inclusive
    LwU32 count;                         // OUT     Number of ops in this call
    LwU32 hasMore;                       // OUT     Whether there are more valid mappings in this range than MAX_GET_MAPPINGS_OPS
    LW_DECLARE_ALIGNED(LW83DE_CTRL_DEBUG_GET_MAPPINGS_OP opsBuffer[MAX_GET_MAPPINGS_OPS], 8);
} LW83DE_CTRL_DEBUG_GET_MAPPINGS_PARAMETERS;

/*
 * LW83DE_CTRL_CMD_DEBUG_EXEC_REG_OPS
 *
 * This command is used to submit a buffer containing one or more
 * LW2080_CTRL_GPU_REG_OP structures for processing.  Each entry in the
 * buffer specifies a single read or write operation.  Each entry is checked
 * for validity in an initial pass over the buffer with the results for
 * each operation stored in the corresponding regStatus field. Unless
 * bNonTransactional flag is set to true, if any invalid entries are found
 * during this initial pass then none of the operations are exelwted. Entries
 * are processed in order within each regType with LW2080_CTRL_GPU_REG_OP_TYPE_GLOBAL
 * entries processed first followed by LW2080_CTRL_GPU_REG_OP_TYPE_GR_CTX entries.
 *
 * [IN]     bNonTransactional
 *     This field specifies if command is non-transactional i.e. if set to
 *     true, all the valid operations will be exelwted.
 *
 * [IN]     regOpCount
 *     This field specifies the number of valid entries in the regops list.
 *
 * [IN/OUT] regOps
 *     This field is to be filled with the desired register information that is
 *     to be retrieved.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_ILWALID_PARAM_STRUCT
 */
#define LW83DE_CTRL_CMD_DEBUG_EXEC_REG_OPS   (0x83de031d) /* finn: Evaluated from "(FINN_GT200_DEBUGGER_DEBUG_INTERFACE_ID << 8) | LW83DE_CTRL_DEBUG_EXEC_REG_OPS_PARAMS_MESSAGE_ID" */

#define LW83DE_CTRL_GPU_EXEC_REG_OPS_MAX_OPS 100
#define LW83DE_CTRL_DEBUG_EXEC_REG_OPS_PARAMS_MESSAGE_ID (0x1DU)

typedef struct LW83DE_CTRL_DEBUG_EXEC_REG_OPS_PARAMS {
    LwBool                 bNonTransactional;
    LwU32                  regOpCount;
    // C form: LW2080_CTRL_GPU_REG_OP regOps[LW2080_CTRL_GPU_EXEC_REG_OPS_MAX_OPS]
#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)
    LW2080_CTRL_GPU_REG_OP regOps[LW83DE_CTRL_GPU_EXEC_REG_OPS_MAX_OPS];
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)

} LW83DE_CTRL_DEBUG_EXEC_REG_OPS_PARAMS;

/*
 * LW83DE_CTRL_CMD_DEBUG_SET_MODE_ERRBAR
 *
 * This command sets the Errbar Debug mode. This is Volta-onwards feature.
 * If the query is made on an incorrect platform (for example, pre-Volta)
 * the call will return with an LW_ERR_NOT_SUPPORTED error.
 *
 *   action
 *     The possible action values are:
 *   - LW83DE_CTRL_CMD_DEBUG_SET_MODE_ERRBAR_ENABLE
 *      This enables the Errbar debug mode.
 *
 *   - LW83DE_CTRL_CMD_DEBUG_SET_MODE_ERRBAR_DISABLE
 *      This disables the Errbar debug mode.
 *
 * Possible return values:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 *   LW_ERR_ILWALID_STATE
 *   LW_ERR_ILWALID_WRITE
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW83DE_CTRL_CMD_DEBUG_SET_MODE_ERRBAR_DEBUG (0x83de031f) /* finn: Evaluated from "(FINN_GT200_DEBUGGER_DEBUG_INTERFACE_ID << 8) | LW83DE_CTRL_DEBUG_SET_MODE_ERRBAR_DEBUG_PARAMS_MESSAGE_ID" */

#define LW83DE_CTRL_DEBUG_SET_MODE_ERRBAR_DEBUG_PARAMS_MESSAGE_ID (0x1FU)

typedef struct LW83DE_CTRL_DEBUG_SET_MODE_ERRBAR_DEBUG_PARAMS {
    LwU32 action;
} LW83DE_CTRL_DEBUG_SET_MODE_ERRBAR_DEBUG_PARAMS;

#define LW83DE_CTRL_CMD_DEBUG_SET_MODE_ERRBAR_DEBUG_DISABLE (0x00000000)
#define LW83DE_CTRL_CMD_DEBUG_SET_MODE_ERRBAR_DEBUG_ENABLE  (0x00000001)

/*
 * LW83DE_CTRL_CMD_DEBUG_GET_MODE_ERRBAR
 *
 * This command gets the value of lwrrently configured Errbar DEBUG mode.
 * This is Volta-onwards feature. If the query is made on an incorrect
 * platform (for example, pre-Volta) the call will return with an
 * LW_ERR_NOT_SUPPORTED error.
 *
 *   value
 *     This parameter returns the configured value.
 *
 * Possible return values:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 *   LW_ERR_ILWALID_STATE
 *   LW_ERR_ILWALID_READ
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW83DE_CTRL_CMD_DEBUG_GET_MODE_ERRBAR_DEBUG         (0x83de0320) /* finn: Evaluated from "(FINN_GT200_DEBUGGER_DEBUG_INTERFACE_ID << 8) | LW83DE_CTRL_DEBUG_GET_MODE_ERRBAR_DEBUG_PARAMS_MESSAGE_ID" */

#define LW83DE_CTRL_DEBUG_GET_MODE_ERRBAR_DEBUG_PARAMS_MESSAGE_ID (0x20U)

typedef struct LW83DE_CTRL_DEBUG_GET_MODE_ERRBAR_DEBUG_PARAMS {
    LwU32 value;
} LW83DE_CTRL_DEBUG_GET_MODE_ERRBAR_DEBUG_PARAMS;

#define LW83DE_CTRL_CMD_DEBUG_GET_MODE_ERRBAR_DEBUG_DISABLED (0x00000000)
#define LW83DE_CTRL_CMD_DEBUG_GET_MODE_ERRBAR_DEBUG_ENABLED  (0x00000001)

/*
 * LW83DE_CTRL_CMD_DEBUG_SET_SINGLE_SM_SINGLE_STEP
 *
 * This command either enables or disables single step mode for the given SM.
 *
 *   smID (input)
 *     This identifies the SM.
 *   bSingleStep (input)
 *     This indicates the single step mode. LW_TRUE for ENABLED.
 */
#define LW83DE_CTRL_CMD_DEBUG_SET_SINGLE_SM_SINGLE_STEP      (0x83de0321) /* finn: Evaluated from "(FINN_GT200_DEBUGGER_DEBUG_INTERFACE_ID << 8) | LW83DE_CTRL_DEBUG_SET_SINGLE_SM_SINGLE_STEP_PARAMS_MESSAGE_ID" */

#define LW83DE_CTRL_DEBUG_SET_SINGLE_SM_SINGLE_STEP_PARAMS_MESSAGE_ID (0x21U)

typedef struct LW83DE_CTRL_DEBUG_SET_SINGLE_SM_SINGLE_STEP_PARAMS {
    LwU32  smID;
    LwBool bSingleStep;
} LW83DE_CTRL_DEBUG_SET_SINGLE_SM_SINGLE_STEP_PARAMS;

/*
 * LW83DE_CTRL_CMD_DEBUG_SET_SINGLE_SM_STOP_TRIGGER
 *
 * This command sets or clears the stop trigger for the given SM.
 *
 *   smID (input)
 *     This identifies the SM.
 *   bStopTrigger (input)
 *     This indicates whether to set or clear the trigger. LW_TRUE for ENABLED.
 */
#define LW83DE_CTRL_CMD_DEBUG_SET_SINGLE_SM_STOP_TRIGGER (0x83de0322) /* finn: Evaluated from "(FINN_GT200_DEBUGGER_DEBUG_INTERFACE_ID << 8) | LW83DE_CTRL_DEBUG_SET_SINGLE_SM_STOP_TRIGGER_PARAMS_MESSAGE_ID" */

#define LW83DE_CTRL_DEBUG_SET_SINGLE_SM_STOP_TRIGGER_BROADCAST ((LwU32)~0)

#define LW83DE_CTRL_DEBUG_SET_SINGLE_SM_STOP_TRIGGER_PARAMS_MESSAGE_ID (0x22U)

typedef struct LW83DE_CTRL_DEBUG_SET_SINGLE_SM_STOP_TRIGGER_PARAMS {
    LwU32  smID;
    LwBool bStopTrigger;
} LW83DE_CTRL_DEBUG_SET_SINGLE_SM_STOP_TRIGGER_PARAMS;

/*
 * LW83DE_CTRL_CMD_DEBUG_SET_SINGLE_SM_RUN_TRIGGER
 *
 * This command sets or clears the run trigger for the given SM.
 *
 *   smID (input)
 *     This identifies the SM.
 *   bRunTrigger (input)
 *     This indicates whether to set or clear the trigger. LW_TRUE for ENABLED.
 */
#define LW83DE_CTRL_CMD_DEBUG_SET_SINGLE_SM_RUN_TRIGGER (0x83de0323) /* finn: Evaluated from "(FINN_GT200_DEBUGGER_DEBUG_INTERFACE_ID << 8) | LW83DE_CTRL_DEBUG_SET_SINGLE_SM_RUN_TRIGGER_PARAMS_MESSAGE_ID" */

#define LW83DE_CTRL_DEBUG_SET_SINGLE_SM_RUN_TRIGGER_PARAMS_MESSAGE_ID (0x23U)

typedef struct LW83DE_CTRL_DEBUG_SET_SINGLE_SM_RUN_TRIGGER_PARAMS {
    LwU32  smID;
    LwBool bRunTrigger;
} LW83DE_CTRL_DEBUG_SET_SINGLE_SM_RUN_TRIGGER_PARAMS;

/*
 * LW83DE_CTRL_CMD_DEBUG_SET_SINGLE_SM_SKIP_IDLE_WARP_DETECT
 *
 * This command enables or disables skip idle warp detect for the given sm.
 *
 *   smID (input)
 *     This identifies the SM.
 *   bSkipIdleWarpDetect (input)
 *     This indicates whether to enable or disable the mode. LW_TRUE for ENABLED.
 */
#define LW83DE_CTRL_CMD_DEBUG_SET_SINGLE_SM_SKIP_IDLE_WARP_DETECT (0x83de0324) /* finn: Evaluated from "(FINN_GT200_DEBUGGER_DEBUG_INTERFACE_ID << 8) | LW83DE_CTRL_DEBUG_SET_SINGLE_SM_SKIP_IDLE_WARP_DETECT_PARAMS_MESSAGE_ID" */

#define LW83DE_CTRL_DEBUG_SET_SINGLE_SM_SKIP_IDLE_WARP_DETECT_PARAMS_MESSAGE_ID (0x24U)

typedef struct LW83DE_CTRL_DEBUG_SET_SINGLE_SM_SKIP_IDLE_WARP_DETECT_PARAMS {
    LwU32  smID;
    LwBool bSkipIdleWarpDetect;
} LW83DE_CTRL_DEBUG_SET_SINGLE_SM_SKIP_IDLE_WARP_DETECT_PARAMS;

/*
 * LW83DE_CTRL_CMD_DEBUG_GET_SINGLE_SM_DEBUGGER_STATUS
 *
 * This command retrieves the debugger status states of the given SM.
 *
 *   smID (input)
 *     This identifies the SM.
 *   bInTrapMode (output)
 *     This indicates whether the SM is in trap mode.
 *   bCrsFlushDone (output)
 *     Deprecated GK110+. Always 0 Volta+.
 *   bRunTriggerInProgress (output)
 *     Deprecated GM10X+. Always 0 Volta+.
 *   bComputeContext (output)
 *     Deprecated GM10X+. Always 0 Volta+.
 *   bLockedDown (output)
 *     This indicates whether the SM is locked down.
 */
#define LW83DE_CTRL_CMD_DEBUG_GET_SINGLE_SM_DEBUGGER_STATUS (0x83de0325) /* finn: Evaluated from "(FINN_GT200_DEBUGGER_DEBUG_INTERFACE_ID << 8) | LW83DE_CTRL_DEBUG_GET_SINGLE_SM_DEBUGGER_STATUS_PARAMS_MESSAGE_ID" */

typedef struct LW83DE_CTRL_DEBUG_SINGLE_SM_DEBUGGER_STATUS {
    LwBool bInTrapMode;
    LwBool bCrsFlushDone;
    LwBool bRunTriggerInProgress;
    LwBool bComputeContext;
    LwBool bLockedDown;
} LW83DE_CTRL_DEBUG_SINGLE_SM_DEBUGGER_STATUS;

#define LW83DE_CTRL_DEBUG_GET_SINGLE_SM_DEBUGGER_STATUS_PARAMS_MESSAGE_ID (0x25U)

typedef struct LW83DE_CTRL_DEBUG_GET_SINGLE_SM_DEBUGGER_STATUS_PARAMS {
    LwU32                                       smID;
    LW83DE_CTRL_DEBUG_SINGLE_SM_DEBUGGER_STATUS smDebuggerStatus;
} LW83DE_CTRL_DEBUG_GET_SINGLE_SM_DEBUGGER_STATUS_PARAMS;

/*!
 * LW83DE_CTRL_CMD_DEBUG_ACCESS_MEMORY_ENTRY
 *
 *   This struct represents a requet to read/write a block of memory.
 *
 *   hMemory [IN]
 *       The handle to the memory being accessed. If hMemory is not accessible
 *       from the caller's address space, LW_ERR_INSUFFICIENT_PERMISSIONS
 *       is returned.
 *
 *   length [IN]
 *       Number of bytes to read/write
 *
 *   memOffset [IN]
 *      The offset into the physical memory region given by the handle above.
 *
 *   dataOffset [IN]
 *      An offset into the usermode memory region provided by the enclosing
 *      params indicating where to read/write data from/to.
 *
 *   status [OUT]
 *       The result status of the operation will be output. If LW_OK, even if
 *       command returned error status, the given operation was successful. If
 *       not LW_OK, it is guaranteed that the command will return error status. 
 */
typedef struct LW83DE_CTRL_DEBUG_ACCESS_MEMORY_ENTRY {
    LwHandle  hMemory;
    LwU32     length;
    LW_DECLARE_ALIGNED(LwU64 memOffset, 8);
    LwU32     dataOffset;
    LW_STATUS status;
} LW83DE_CTRL_DEBUG_ACCESS_MEMORY_ENTRY;

/*!
 * LW83DE_CTRL_CMD_DEBUG_READ_BATCH_MEMORY
 *
 * Execute a batch of read memory operations.
 *
 *   count [IN]
 *     Number of read/write operations to perform.
 *
 *   dataLength [IN]
 *     Length of the usermode buffer passed in, in bytes.
 *
 *   pData [OUT]
 *     Usermode buffer to store the output of the read operations. Each
 *     operation is expected to provide an offset into this buffer.
 *
 *   entries [IN]
 *     List of operations to perform. First `count` entries are used.
 */
#define LW83DE_CTRL_CMD_DEBUG_READ_BATCH_MEMORY  (0x83de0326) /* finn: Evaluated from "(FINN_GT200_DEBUGGER_DEBUG_INTERFACE_ID << 8) | 0x26" */

/*!
 * LW83DE_CTRL_CMD_DEBUG_WRITE_BATCH_MEMORY
 *
 * Execute a batch of write memory operations.
 *
 *   count [IN]
 *     Number of read/write operations to perform.
 *
 *   dataLength [IN]
 *     Length of the usermode buffer passed in, in bytes.
 *
 *   pData [IN]
 *     Usermode buffer to store the input of the write operations. Each
 *     operation is expected to provide an offset into this buffer.
 *
 *   entries [IN]
 *     List of operations to perform. First `count` entries are used.
 */
#define LW83DE_CTRL_CMD_DEBUG_WRITE_BATCH_MEMORY (0x83de0327) /* finn: Evaluated from "(FINN_GT200_DEBUGGER_DEBUG_INTERFACE_ID << 8) | 0x27" */

#define MAX_ACCESS_MEMORY_OPS                    150
typedef struct LW83DE_CTRL_DEBUG_ACCESS_MEMORY_PARAMS {
    LW_DECLARE_ALIGNED(LwP64 pData, 8);
    LwU32 dataLength;
    LwU32 count;
    LW_DECLARE_ALIGNED(LW83DE_CTRL_DEBUG_ACCESS_MEMORY_ENTRY entries[MAX_ACCESS_MEMORY_OPS], 8);
} LW83DE_CTRL_DEBUG_ACCESS_MEMORY_PARAMS;

/* _ctrl83dedebug_h_ */

