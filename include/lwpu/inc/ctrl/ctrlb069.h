/*
* _LWRM_COPYRIGHT_BEGIN_
*
* Copyright 2013-2015 by LWPU Corporation.  All rights reserved.  All
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
// Source file: ctrl/ctrlb069.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)




#include "ctrl/ctrlxxxx.h"
/* MAXWELL_FAULT_BUFFER_A control commands and parameters */

#define LWB069_CTRL_CMD(cat,idx)          LWXXXX_CTRL_CMD(0xB069, LWB069_CTRL_##cat, idx)

/* MAXWELL_FAULT_BUFFER_A command categories (6bits) */
#define LWB069_CTRL_RESERVED    (0x00)
#define LWB069_CTRL_FAULTBUFFER (0x01)

/*
 * LWB069_CTRL_CMD_NULL
 *
 * This command does nothing.
 * This command does not take any parameters.
 *
 * Possible status values returned are:
 *   LW_OK
 */
#define LWB069_CTRL_CMD_NULL    (0xb0690000) /* finn: Evaluated from "(FINN_MAXWELL_FAULT_BUFFER_A_RESERVED_INTERFACE_ID << 8) | 0x0" */






/*
 * LWB069_CTRL_CMD_FAULTBUFFER_READ_GET
 *
 * This command returns the current HW GET pointer for the requested type fault buffer
 *
 *    faultBufferGetOffset
 *      Value of current HW GET pointer
 *    faultBufferType
 *      Type of fault buffer. FAULT_BUFFER_REPLAYABLE or FAULT_BUFFER_NON_REPLAYABLE
 */
#define LWB069_CTRL_CMD_FAULTBUFFER_READ_GET (0xb0690101) /* finn: Evaluated from "(FINN_MAXWELL_FAULT_BUFFER_A_FAULTBUFFER_INTERFACE_ID << 8) | LWB069_CTRL_FAULTBUFFER_READ_GET_PARAMS_MESSAGE_ID" */

#define LWB069_CTRL_FAULTBUFFER_READ_GET_PARAMS_MESSAGE_ID (0x1U)

typedef struct LWB069_CTRL_FAULTBUFFER_READ_GET_PARAMS {
    LwU32 faultBufferGetOffset;
    LwU32 faultBufferType;
} LWB069_CTRL_FAULTBUFFER_READ_GET_PARAMS;

//
// Valid Fault buffer Types
// NON_REPLAYABLE is only supported in Volta+ GPUs.
//
#define LWB069_CTRL_FAULT_BUFFER_NON_REPLAYABLE (0x00000000)
#define LWB069_CTRL_FAULT_BUFFER_REPLAYABLE     (0x00000001)

/*
 * LWB069_CTRL_CMD_FAULTBUFFER_WRITE_GET
 *
 * This command writes the HW GET pointer for the requested type of fault buffer
 *
 * NOTE: The caller must issue a write barrier before this function to
 * ensure modifications to the current buffer entry are committed before
 * the GET pointer is updated.
 * 
 *    faultBufferGetOffset
 *      Value to be written to HW GET pointer
 *    faultBufferType
 *      Type of fault buffer. FAULT_BUFFER_REPLAYABLE or FAULT_BUFFER_NON_REPLAYABLE
 */
#define LWB069_CTRL_CMD_FAULTBUFFER_WRITE_GET   (0xb0690102) /* finn: Evaluated from "(FINN_MAXWELL_FAULT_BUFFER_A_FAULTBUFFER_INTERFACE_ID << 8) | LWB069_CTRL_FAULTBUFFER_WRITE_GET_PARAMS_MESSAGE_ID" */

#define LWB069_CTRL_FAULTBUFFER_WRITE_GET_PARAMS_MESSAGE_ID (0x2U)

typedef struct LWB069_CTRL_FAULTBUFFER_WRITE_GET_PARAMS {
    LwU32 faultBufferGetOffset;
    LwU32 faultBufferType;
} LWB069_CTRL_FAULTBUFFER_WRITE_GET_PARAMS;

/*
 * LWB069_CTRL_CMD_FAULTBUFFER_READ_PUT
 *
 * This command returns the current HW PUT pointer for the requested type fault buffer
 *
 *    faultBufferGetOffset
 *      Value of current HW PUT pointer
 *    faultBufferType
 *      Type of fault buffer. FAULT_BUFFER_REPLAYABLE or FAULT_BUFFER_NON_REPLAYABLE
 */
#define LWB069_CTRL_CMD_FAULTBUFFER_READ_PUT (0xb0690103) /* finn: Evaluated from "(FINN_MAXWELL_FAULT_BUFFER_A_FAULTBUFFER_INTERFACE_ID << 8) | LWB069_CTRL_FAULTBUFFER_READ_PUT_PARAMS_MESSAGE_ID" */

#define LWB069_CTRL_FAULTBUFFER_READ_PUT_PARAMS_MESSAGE_ID (0x3U)

typedef struct LWB069_CTRL_FAULTBUFFER_READ_PUT_PARAMS {
    LwU32 faultBufferPutOffset;
    LwU32 faultBufferType;
} LWB069_CTRL_FAULTBUFFER_READ_PUT_PARAMS;

#define LWB069_CTRL_CMD_FAULTBUFFER_ENABLE_NOTIFICATION (0xb0690104) /* finn: Evaluated from "(FINN_MAXWELL_FAULT_BUFFER_A_FAULTBUFFER_INTERFACE_ID << 8) | LWB069_CTRL_FAULTBUFFER_ENABLE_NOTIFICATION_PARAMS_MESSAGE_ID" */

#define LWB069_CTRL_FAULTBUFFER_ENABLE_NOTIFICATION_PARAMS_MESSAGE_ID (0x4U)

typedef struct LWB069_CTRL_FAULTBUFFER_ENABLE_NOTIFICATION_PARAMS {
    LwBool Enable;
} LWB069_CTRL_FAULTBUFFER_ENABLE_NOTIFICATION_PARAMS;

#define LWB069_CTRL_CMD_FAULTBUFFER_GET_SIZE (0xb0690105) /* finn: Evaluated from "(FINN_MAXWELL_FAULT_BUFFER_A_FAULTBUFFER_INTERFACE_ID << 8) | LWB069_CTRL_FAULTBUFFER_GET_SIZE_PARAMS_MESSAGE_ID" */

#define LWB069_CTRL_FAULTBUFFER_GET_SIZE_PARAMS_MESSAGE_ID (0x5U)

typedef struct LWB069_CTRL_FAULTBUFFER_GET_SIZE_PARAMS {
    LwU32 faultBufferSize;
} LWB069_CTRL_FAULTBUFFER_GET_SIZE_PARAMS;


/*
 * LWB069_CTRL_CMD_FAULTBUFFER_GET_REGISTER_MAPPINGS
 *
 * This command provides kernel mapping to a few registers.
 * These mappings are needed by UVM driver to handle non fatal gpu faults
 *
 *    pFaultBufferGet
 *      Mapping for fault buffer's get pointer (LW_PFIFO_REPLAYABLE_FAULT_BUFFER_GET)
 *    pFaultBufferPut
 *      Mapping for fault buffer's put pointer (LW_PFIFO_REPLAYABLE_FAULT_BUFFER_PUT)
 *    pFaultBufferInfo
 *      Mapping for fault buffer's Info pointer (LW_PFIFO_REPLAYABLE_FAULT_BUFFER_INFO)
 *      Note: this variable is deprecated since buffer overflow is not a seperate register from Volta
 *    pPmcIntr
 *      Mapping for PMC intr register (LW_PMC_INTR(0))
 *    pPmcIntrEnSet
 *      Mapping for PMC intr set register - used to enable an intr (LW_PMC_INTR_EN_SET(0))
 *    pPmcIntrEnClear
 *      Mapping for PMC intr clear register - used to disable an intr (LW_PMC_INTR_EN_CLEAR(0))
 *    replayableFaultMask
 *      Mask for the replayable fault bit(LW_PMC_INTR_REPLAYABLE_FAULT)
 */
#define LWB069_CTRL_CMD_FAULTBUFFER_GET_REGISTER_MAPPINGS (0xb0690106) /* finn: Evaluated from "(FINN_MAXWELL_FAULT_BUFFER_A_FAULTBUFFER_INTERFACE_ID << 8) | LWB069_CTRL_CMD_FAULTBUFFER_GET_REGISTER_MAPPINGS_PARAMS_MESSAGE_ID" */

#define LWB069_CTRL_CMD_FAULTBUFFER_GET_REGISTER_MAPPINGS_PARAMS_MESSAGE_ID (0x6U)

typedef struct LWB069_CTRL_CMD_FAULTBUFFER_GET_REGISTER_MAPPINGS_PARAMS {
    LW_DECLARE_ALIGNED(LwP64 pFaultBufferGet, 8);
    LW_DECLARE_ALIGNED(LwP64 pFaultBufferPut, 8);
    LW_DECLARE_ALIGNED(LwP64 pFaultBufferInfo, 8);
    LW_DECLARE_ALIGNED(LwP64 pPmcIntr, 8);
    LW_DECLARE_ALIGNED(LwP64 pPmcIntrEnSet, 8);
    LW_DECLARE_ALIGNED(LwP64 pPmcIntrEnClear, 8);
    LwU32 replayableFaultMask;
    LW_DECLARE_ALIGNED(LwP64 pPrefetchCtrl, 8);
} LWB069_CTRL_CMD_FAULTBUFFER_GET_REGISTER_MAPPINGS_PARAMS;

/* _ctrlb069_h_ */
