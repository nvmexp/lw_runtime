/*
* _LWRM_COPYRIGHT_BEGIN_
*
* Copyright 2015 by LWPU Corporation.  All rights reserved.  All
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
// Source file: ctrl/ctrlc365.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)




#include "ctrl/ctrlxxxx.h"
#define LWC365_CTRL_CMD(cat,idx)          LWXXXX_CTRL_CMD(0xC365, LWC365_CTRL_##cat, idx)


#define LWC365_CTRL_RESERVED           (0x00)
#define LWC365_CTRL_ACCESS_CNTR_BUFFER (0x01)


/*
 * LWC365_CTRL_CMD_NULL
 *
 * This command does nothing.
 * This command does not take any parameters.
 *
 * Possible status values returned are:
 *   LW_OK
 */
#define LWC365_CTRL_CMD_NULL           (0xc3650000) /* finn: Evaluated from "(FINN_ACCESS_COUNTER_NOTIFY_BUFFER_RESERVED_INTERFACE_ID << 8) | 0x0" */






/*
 * LWC365_CTRL_CMD_ACCESS_CNTR_BUFFER_READ_GET
 *
 * This command provides the value of the GET register 
 *
 *    accessCntrBufferGetOffset [OUT]
 *       This parameter returns the value of the GET register
 *
 * Possible status values returned are:
 *   LW_OK
 */

#define LWC365_CTRL_CMD_ACCESS_CNTR_BUFFER_READ_GET (0xc3650101) /* finn: Evaluated from "(FINN_ACCESS_COUNTER_NOTIFY_BUFFER_ACCESS_CNTR_BUFFER_INTERFACE_ID << 8) | LWC365_CTRL_ACCESS_CNTR_BUFFER_READ_GET_PARAMS_MESSAGE_ID" */

#define LWC365_CTRL_ACCESS_CNTR_BUFFER_READ_GET_PARAMS_MESSAGE_ID (0x1U)

typedef struct LWC365_CTRL_ACCESS_CNTR_BUFFER_READ_GET_PARAMS {
    LwU32 accessCntrBufferGetOffset;
} LWC365_CTRL_ACCESS_CNTR_BUFFER_READ_GET_PARAMS;


/*
 * LWC365_CTRL_CMD_ACCESS_CNTR_BUFFER_WRITE_GET
 *
 * This command writes a value into the GET register 
 *
 *    accessCntrBufferGetValue [IN]
 *       This parameter specifies the new value of the GET register
 *
 * Possible status values returned are:
 *   LW_OK
 */

#define LWC365_CTRL_CMD_ACCESS_CNTR_BUFFER_WRITE_GET (0xc3650102) /* finn: Evaluated from "(FINN_ACCESS_COUNTER_NOTIFY_BUFFER_ACCESS_CNTR_BUFFER_INTERFACE_ID << 8) | LWC365_CTRL_ACCESS_CNTR_BUFFER_WRITE_GET_PARAMS_MESSAGE_ID" */

#define LWC365_CTRL_ACCESS_CNTR_BUFFER_WRITE_GET_PARAMS_MESSAGE_ID (0x2U)

typedef struct LWC365_CTRL_ACCESS_CNTR_BUFFER_WRITE_GET_PARAMS {
    LwU32 accessCntrBufferGetValue;
} LWC365_CTRL_ACCESS_CNTR_BUFFER_WRITE_GET_PARAMS;


/*
 * LWC365_CTRL_CMD_ACCESS_CNTR_BUFFER_READ_PUT
 *
 * This command provides the value of the PUT register 
 *
 *    accessCntrBufferPutOffset [OUT]
 *       This parameter returns the value of the PUT register
 *
 * Possible status values returned are:
 *   LW_OK
 */

#define LWC365_CTRL_CMD_ACCESS_CNTR_BUFFER_READ_PUT (0xc3650103) /* finn: Evaluated from "(FINN_ACCESS_COUNTER_NOTIFY_BUFFER_ACCESS_CNTR_BUFFER_INTERFACE_ID << 8) | LWC365_CTRL_ACCESS_CNTR_BUFFER_READ_PUT_PARAMS_MESSAGE_ID" */

#define LWC365_CTRL_ACCESS_CNTR_BUFFER_READ_PUT_PARAMS_MESSAGE_ID (0x3U)

typedef struct LWC365_CTRL_ACCESS_CNTR_BUFFER_READ_PUT_PARAMS {
    LwU32 accessCntrBufferPutOffset;
} LWC365_CTRL_ACCESS_CNTR_BUFFER_READ_PUT_PARAMS;


/*
 * LWC365_CTRL_CMD_ACCESS_CNTR_BUFFER_ENABLE
 *
 * This command enables/disables the access counters 
 * It also sets up RM to either service or ignore the Access Counter interrupts.
 *
 *    intrOwnership [IN]
 *       This parameter specifies whether RM should own the interrupt upon return
 *    enable [IN]
 *       LW_TRUE  = Access counters will be enabled
 *       LW_FALSE = Access counters will be disabled
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 */

#define LWC365_CTRL_CMD_ACCESS_CNTR_BUFFER_ENABLE (0xc3650104) /* finn: Evaluated from "(FINN_ACCESS_COUNTER_NOTIFY_BUFFER_ACCESS_CNTR_BUFFER_INTERFACE_ID << 8) | LWC365_CTRL_ACCESS_CNTR_BUFFER_ENABLE_PARAMS_MESSAGE_ID" */

#define LWC365_CTRL_ACCESS_CNTR_BUFFER_ENABLE_PARAMS_MESSAGE_ID (0x4U)

typedef struct LWC365_CTRL_ACCESS_CNTR_BUFFER_ENABLE_PARAMS {
    LwU32  intrOwnership;
    LwBool enable;
} LWC365_CTRL_ACCESS_CNTR_BUFFER_ENABLE_PARAMS;

#define LWC365_CTRL_ACCESS_COUNTER_INTERRUPT_OWNERSHIP_NO_CHANGE (0x0)
#define LWC365_CTRL_ACCESS_COUNTER_INTERRUPT_OWNERSHIP_RM        (0x1)
#define LWC365_CTRL_ACCESS_COUNTER_INTERRUPT_OWNERSHIP_NOT_RM    (0x2)

/*
 * LWC365_CTRL_CMD_ACCESS_CNTR_BUFFER_GET_SIZE
 *
 * This command provides the size of the notification buffer
 *
 *    accessCntrBufferSize [OUT]
 *       This parameter returns the size of the notification buffer
 *
 * Possible status values returned are:
 *   LW_OK
 */

#define LWC365_CTRL_CMD_ACCESS_CNTR_BUFFER_GET_SIZE              (0xc3650105) /* finn: Evaluated from "(FINN_ACCESS_COUNTER_NOTIFY_BUFFER_ACCESS_CNTR_BUFFER_INTERFACE_ID << 8) | LWC365_CTRL_ACCESS_CNTR_BUFFER_GET_SIZE_PARAMS_MESSAGE_ID" */

#define LWC365_CTRL_ACCESS_CNTR_BUFFER_GET_SIZE_PARAMS_MESSAGE_ID (0x5U)

typedef struct LWC365_CTRL_ACCESS_CNTR_BUFFER_GET_SIZE_PARAMS {
    LwU32 accessCntrBufferSize;
} LWC365_CTRL_ACCESS_CNTR_BUFFER_GET_SIZE_PARAMS;


/*
 * LWC365_CTRL_CMD_ACCESS_CNTR_BUFFER_GET_REGISTER_MAPPINGS
 *
 * This command provides the access counter register mappings
 *
 *    pAccessCntrBufferGet [OUT]
 *       This parameter returns the pointer to the GET register
 *    pAccessCntrBufferPut [OUT]
 *       This parameter returns the pointer to the PUT register
 *    pAccessCntrlBufferFull [OUT]
 *       This parameter returns the pointer to the FULL register 
 *    pHubIntr [OUT]
 *       This parameter returns the pointer to the hub interrupt register
 *    pHubIntrEnSet [OUT]
 *       This parameter returns the pointer to the set register
 *    pHubIntrEnClear [OUT]
 *       This parameter returns the pointer to the clear register
 *    accessCntrMask [OUT]
 *       This parameter returns the interrupt mask
 * 
 * Possible status values returned are:
 *   LW_OK
 */

#define LWC365_CTRL_CMD_ACCESS_CNTR_BUFFER_GET_REGISTER_MAPPINGS (0xc3650106) /* finn: Evaluated from "(FINN_ACCESS_COUNTER_NOTIFY_BUFFER_ACCESS_CNTR_BUFFER_INTERFACE_ID << 8) | LWC365_CTRL_ACCESS_CNTR_BUFFER_GET_REGISTER_MAPPINGS_PARAMS_MESSAGE_ID" */

#define LWC365_CTRL_ACCESS_CNTR_BUFFER_GET_REGISTER_MAPPINGS_PARAMS_MESSAGE_ID (0x6U)

typedef struct LWC365_CTRL_ACCESS_CNTR_BUFFER_GET_REGISTER_MAPPINGS_PARAMS {
    LW_DECLARE_ALIGNED(LwP64 pAccessCntrBufferGet, 8);
    LW_DECLARE_ALIGNED(LwP64 pAccessCntrBufferPut, 8);
    LW_DECLARE_ALIGNED(LwP64 pAccessCntrBufferFull, 8);
    LW_DECLARE_ALIGNED(LwP64 pHubIntr, 8);
    LW_DECLARE_ALIGNED(LwP64 pHubIntrEnSet, 8);
    LW_DECLARE_ALIGNED(LwP64 pHubIntrEnClear, 8);
    LwU32 accessCntrMask;
} LWC365_CTRL_ACCESS_CNTR_BUFFER_GET_REGISTER_MAPPINGS_PARAMS;


/*
 * LWC365_CTRL_CMD_ACCESS_CNTR_BUFFER_GET_FULL_INFO
 *
 * This command gives information whether the buffer is full
 *
 *    fullFlag [OUT]
 *       This parameter specifies whether the buffer is full
 *
 * Possible status values returned are:
 *   LW_OK
 */

#define LWC365_CTRL_CMD_ACCESS_CNTR_BUFFER_GET_FULL_INFO (0xc3650107) /* finn: Evaluated from "(FINN_ACCESS_COUNTER_NOTIFY_BUFFER_ACCESS_CNTR_BUFFER_INTERFACE_ID << 8) | LWC365_CTRL_ACCESS_CNTR_BUFFER_GET_FULL_INFO_PARAMS_MESSAGE_ID" */

#define LWC365_CTRL_ACCESS_CNTR_BUFFER_GET_FULL_INFO_PARAMS_MESSAGE_ID (0x7U)

typedef struct LWC365_CTRL_ACCESS_CNTR_BUFFER_GET_FULL_INFO_PARAMS {
    LwBool fullFlag;
} LWC365_CTRL_ACCESS_CNTR_BUFFER_GET_FULL_INFO_PARAMS;


/*
 * LWC365_CTRL_CMD_ACCESS_CNTR_BUFFER_RESET_COUNTERS
 *
 * This command resets access counters of specified type
 *
 *    resetFlag [OUT]
 *       This parameter specifies that counters have been reset
 *    counterType [IN]
 *       This parameter specifies the type of counters that should be reset (MIMC, MOMC or ALL)  
 *
 * Possible status values returned are:
 *   LW_OK
 */

#define LWC365_CTRL_CMD_ACCESS_CNTR_BUFFER_RESET_COUNTERS (0xc3650108) /* finn: Evaluated from "(FINN_ACCESS_COUNTER_NOTIFY_BUFFER_ACCESS_CNTR_BUFFER_INTERFACE_ID << 8) | LWC365_CTRL_ACCESS_CNTR_BUFFER_RESET_COUNTERS_PARAMS_MESSAGE_ID" */

#define LWC365_CTRL_ACCESS_CNTR_BUFFER_RESET_COUNTERS_PARAMS_MESSAGE_ID (0x8U)

typedef struct LWC365_CTRL_ACCESS_CNTR_BUFFER_RESET_COUNTERS_PARAMS {
    LwBool resetFlag;
    LwU32  counterType;
} LWC365_CTRL_ACCESS_CNTR_BUFFER_RESET_COUNTERS_PARAMS;

#define LWC365_CTRL_ACCESS_COUNTER_TYPE_MIMC   (0x0)
#define LWC365_CTRL_ACCESS_COUNTER_TYPE_MOMC   (0x1)
#define LWC365_CTRL_ACCESS_COUNTER_TYPE_ALL    (0x2)

/*
 * LWC365_CTRL_CMD_ACCESS_CNTR_SET_CONFIG
 *
 * This command configures the access counters
 *
 *    mimcGranularity [IN]
 *       This parameter specifies the desired granularity for mimc (64K, 2M, 16M, 16G)
 *    momcGranularity [IN]
 *       This parameter specifies the desired granularity for momc (64K, 2M, 16M, 16G)
 *    mimcLimit [IN]
 *       This parameter specifies mimc limit (none, qtr, half, full)
 *    momcLimit [IN]
 *       This parameter specifies momc limit (none, qtr, half, full)
 *    threshold [IN]
 *       This parameter specifies the threshold
 *    flag [IN]
 *       This parameter is a bitmask denoting what configurations should be made
 *
 * Possible status values returned are:
 *   LW_OK
 */

#define LWC365_CTRL_CMD_ACCESS_CNTR_SET_CONFIG (0xc3650109) /* finn: Evaluated from "(FINN_ACCESS_COUNTER_NOTIFY_BUFFER_ACCESS_CNTR_BUFFER_INTERFACE_ID << 8) | LWC365_CTRL_ACCESS_CNTR_SET_CONFIG_PARAMS_MESSAGE_ID" */

#define LWC365_CTRL_ACCESS_CNTR_SET_CONFIG_PARAMS_MESSAGE_ID (0x9U)

typedef struct LWC365_CTRL_ACCESS_CNTR_SET_CONFIG_PARAMS {
    LwU32 mimcGranularity;
    LwU32 momcGranularity;
    LwU32 mimcLimit;
    LwU32 momcLimit;
    LwU32 threshold;
    LwU32 cmd;
} LWC365_CTRL_ACCESS_CNTR_SET_CONFIG_PARAMS;

#define LWC365_CTRL_ACCESS_COUNTER_GRANULARITY_64K      (0x0)
#define LWC365_CTRL_ACCESS_COUNTER_GRANULARITY_2M       (0x1)
#define LWC365_CTRL_ACCESS_COUNTER_GRANULARITY_16M      (0x2)
#define LWC365_CTRL_ACCESS_COUNTER_GRANULARITY_16G      (0x3)

#define LWC365_CTRL_ACCESS_COUNTER_MIMC_LIMIT           (0x0)
#define LWC365_CTRL_ACCESS_COUNTER_MOMC_LIMIT           (0x1)

#define LWC365_CTRL_ACCESS_COUNTER_USE_LIMIT_NONE       (0x0)
#define LWC365_CTRL_ACCESS_COUNTER_USE_LIMIT_QTR        (0x1)
#define LWC365_CTRL_ACCESS_COUNTER_USE_LIMIT_HALF       (0x2)
#define LWC365_CTRL_ACCESS_COUNTER_USE_LIMIT_FULL       (0x3)

#define LWC365_CTRL_ACCESS_COUNTER_SET_MIMC_GRANULARITY (0x1)
#define LWC365_CTRL_ACCESS_COUNTER_SET_MOMC_GRANULARITY (0x2)
#define LWC365_CTRL_ACCESS_COUNTER_SET_MIMC_LIMIT       (0x4)
#define LWC365_CTRL_ACCESS_COUNTER_SET_MOMC_LIMIT       (0x8)
#define LWC365_CTRL_ACCESS_COUNTER_SET_THRESHOLD        (0x10)

/*
 * LWC365_CTRL_CMD_ACCESS_CNTR_BUFFER_ENABLE_INTR
 *
 * This command enables the access counters interrupts 
 *
 *    enable [OUT]
 *       This parameter specifies that the access counters interrupts are enabled
 *
 * Possible status values returned are:
 *   LW_OK
 */

#define LWC365_CTRL_CMD_ACCESS_CNTR_BUFFER_ENABLE_INTR  (0xc365010b) /* finn: Evaluated from "(FINN_ACCESS_COUNTER_NOTIFY_BUFFER_ACCESS_CNTR_BUFFER_INTERFACE_ID << 8) | LWC365_CTRL_ACCESS_CNTR_BUFFER_ENABLE_INTR_PARAMS_MESSAGE_ID" */

#define LWC365_CTRL_ACCESS_CNTR_BUFFER_ENABLE_INTR_PARAMS_MESSAGE_ID (0xBU)

typedef struct LWC365_CTRL_ACCESS_CNTR_BUFFER_ENABLE_INTR_PARAMS {
    LwBool enable;
} LWC365_CTRL_ACCESS_CNTR_BUFFER_ENABLE_INTR_PARAMS;
/* _ctrlc365_h_ */
