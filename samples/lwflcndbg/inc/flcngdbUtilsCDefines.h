/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2013-2014 by LWPU Corporation.  All rights reserved.  All
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
#define CMD_NUM 20

// maximum breakpoints supported by HW
#define MAX_NUM_BREAKPOINTS 2

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
    FALCGDB_CMD_CLEAN_SESSION,
    FALCGDB_CMD_READ_REG,
    FALCGDB_CMD_READ_DMEM, 
    FALCGDB_CMD_PRINT_GLOBAL_SYMBOL,
    FALCGDB_CMD_SET_DIR_PATH,
    FALCGDB_CMD_SHOW_BP_INFO,
    FALCGDB_CMD_ENABLE_BP,
    FALCGDB_CMD_DISABLE_BP,
	FALCGDB_CMD_PRINT_STACK
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
    LwU32 bpIBRK[MAX_NUM_BREAKPOINTS];

} FLCNGDB_REGISTER_MAP;

typedef struct _FLCNGDB_FP_TABLE
{
    LwU32  (*flcngdbRegRd32)(LwU32 index);
    void   (*flcngdbReadDMEM)(char *pDataBuff, LwU32 startAddress, LwU32 length);
    LwU32  (*flcngdbReadWordDMEM)(LwU32 address);
    int    (*dbgPrintf)( const char * format, ... );
} FLCNGDB_FP_TABLE;

// these can be extern and called from flcngdbUtils.cpp
#endif /* _FLCNGDBUTILSCDEFINES_H_ */




