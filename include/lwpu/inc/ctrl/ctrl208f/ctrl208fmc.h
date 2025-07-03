/* 
* _LWRM_COPYRIGHT_BEGIN_
*
* Copyright 2009-2015 by LWPU Corporation.  All rights reserved.  All
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
// Source file: ctrl/ctrl208f/ctrl208fmc.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)




#include "ctrl/ctrl208f/ctrl208fbase.h"

/*
 * LW208F_CTRL_CMD_MC_ELPG_INDUCE_FALSE_FAULT
 *
 * This command induce the false Elpg RC fault while ELPG_ON/OFF path
 * by returning from the sequence in between without clearing the requested
 * interrupt. This mainly designed for ilwoking Elpg RC path
 *
 *   engineId
 *     Engine for which we need to induce fault
 *   elpgPath
 *     Elpg path while which we need to induce fault.
 *     Possible options are :
 *       LW208F_CTRL_MC_ELPG_INDUCE_FALSE_FAULT_PG_ON
 *       LW208F_CTRL_MC_ELPG_INDUCE_FALSE_FAULT_PG_OFF
 *   faultStatus
 *     This parameter return the status of Elpg RC fault which earlier induced by
 *     LW208F_CTRL_MC_ELPG_INDUCE_FALSE_FAULT_CMD_SET. This is applicable
 *     with command LW208F_CTRL_MC_ELPG_INDUCE_FALSE_FAULT_CMD_FAULT_STATUS.
 *     possible value are :
 *       LW208F_CTRL_MC_ELPG_INDUCE_FALSE_FAULT_NOFAULT
 *         Never Induced the ELPG RC False Fault or Reset the Fault
 *       LW208F_CTRL_MC_ELPG_INDUCE_FALSE_FAULT_RECOVERED
 *         FAULT recovered, or not induced yet
 *       LW208F_CTRL_MC_ELPG_INDUCE_FALSE_FAULT_PROCESSING
 *         Fault recovery processing started but not completed
 *       LW208F_CTRL_MC_ELPG_INDUCE_FALSE_FAULT_FAILED
 *         Fault Recovery Failed
 *       LW208F_CTRL_MC_ELPG_INDUCE_FALSE_FAULT_PG_ON
 *         Fault is at PG_ON path and its recovery processing not started yet
 *       LW208F_CTRL_MC_ELPG_INDUCE_FALSE_FAULT_PG_OFF
 *         Fault is at PG_OFF path and its recovery processing not started yet
 *   commandId
 *     This parameter is used to specify the purpose of ctrl cmd
 *     Possible values are :
 *       LW208F_CTRL_MC_ELPG_INDUCE_FALSE_FAULT_CMD_SET
 *         Set the variable for specified Engine and Path so that next time if Elpg state
 *         machine reaches that path it will cause RC error and hence its recovery.
 *       LW208F_CTRL_MC_ELPG_INDUCE_FALSE_FAULT_CMD_RESET
 *         Rest the variable which cause false RC error while ELPG path
 *       LW208F_CTRL_MC_ELPG_INDUCE_FALSE_FAULT_CMD_FAULT_STATUS
 *         Probe status of Elpg RC error induced by
 *         LW208F_CTRL_CMD_MC_ELPG_INDUCE_FALSE_FAULT_CMD_SET
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_NOT_SUPPORTED
 */

#define LW208F_CTRL_CMD_MC_ELPG_INDUCE_FALSE_FAULT (0x208f0601) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_DIAG_MC_INTERFACE_ID << 8) | 0x1" */

typedef struct LW208F_CTRL_MC_ELPG_INDUCE_FALSE_FAULT_PARAMS {
    LwU32 engineId;
    LwU8  elpgPath;
    LwU8  faultStatus;
    LwU8  commandId;
} LW208F_CTRL_MC_ELPG_INDUCE_FALSE_FAULT_PARAMS;

#define LW208F_CTRL_MC_ELPG_INDUCE_FALSE_FAULT_NOFAULT          (0x00000000)
#define LW208F_CTRL_MC_ELPG_INDUCE_FALSE_FAULT_RECOVERED        (0x00000001)
#define LW208F_CTRL_MC_ELPG_INDUCE_FALSE_FAULT_PROCESSING       (0x00000002)
#define LW208F_CTRL_MC_ELPG_INDUCE_FALSE_FAULT_FAILED           (0x00000003)
#define LW208F_CTRL_MC_ELPG_INDUCE_FALSE_FAULT_PG_ON            (0x00000004)
#define LW208F_CTRL_MC_ELPG_INDUCE_FALSE_FAULT_PG_OFF           (0x00000005)


#define LW208F_CTRL_MC_ELPG_INDUCE_FALSE_FAULT_CMD_RESET        (0x00000000)
#define LW208F_CTRL_MC_ELPG_INDUCE_FALSE_FAULT_CMD_SET          (0x00000001)
#define LW208F_CTRL_MC_ELPG_INDUCE_FALSE_FAULT_CMD_FAULT_STATUS (0x00000002)

