/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2013 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

/*!
 * @file  flcngdbUtilsCDefines.h
 * @brief falcon gdb utility defines & data structures.
 *
 *  */
#ifndef _FLCNGDBUTILSCDEFINES_H_
#define _FLCNGDBUTILSCDEFINES_H_

#include "flcngdbTypes.h"

// class handle to void type so we can use it in C
typedef void CFlcngdbUtils;

// should keep consistant with number in CmdArray
#define CMD_NUM 12

// Error status returned by FlcngdbUtils and UI
#define FLCGDB_ERR_SESSION_DOES_NOT_EXIST -1
#define FLCGDB_ERR_ILWALID_VALUE 0                // for values that cannot be 0

//should keep consitant with the order of CmdArray
typedef enum
{
    FALCGDB_CMD_LAST_CMD = -2,
    FALCGDB_CMD_NONE_MATCH = -1,
    FALCGDB_CMD_HELP = 0,
    FALCGDB_CMD_BF,
    FALCGDB_CMD_BL,
    FALCGDB_CMD_BP,
    FALCGDB_CMD_P,
    FALCGDB_CMD_QUIT,
    FALCGDB_CMD_CONTINUE,
    FALCGDB_CMD_CLEAR_BP,
    FALCGDB_CMD_WAIT,
    FALCGDB_CMD_STEP_INTO,
    FALCGDB_CMD_STEP_OVER, 
    FALCGDB_CMD_CLEAN_SESSION
} FLCNGDB_CMD;

typedef enum
{
    FLCNGDB_SESSION_ENGINE_NONE = -1, 
    FLCNGDB_SESSION_ENGINE_PMU = 0, 
    FLCNGDB_SESSION_ENGINE_DPU = 1,
}FLCNGDB_SESSION_ENGINE_ID;

// structure defines all the registers important to the flcngdb
typedef struct FLCNGDB_REGISTER_MAP
{
    // mark if this map is valid
    LwBool bValid;

    LwU32 registerBase;

    // ICD registers
    LwU32 icdCmd;
    LwU32 icdAddr;
    LwU32 icdWData;
    LwU32 icdRData;

    // breakpoints
    LwU32 numBreakpoints;
    LwU32 firstIBRK;

} FLCNGDB_REGISTER_MAP;

#endif /* _FLCNGDBUTILSCDEFINES_H_ */




