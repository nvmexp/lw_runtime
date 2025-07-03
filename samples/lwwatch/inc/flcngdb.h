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
 * @file  flcngdb.h
 * @brief 
 *
 *  */
#ifndef _FLCNGDB_H_
#define _FLCNGDB_H_

/* ------------------------ Includes --------------------------------------- */
#include "os.h"
#include "hal.h"
#include "falcon.h"
#include "string.h"
#include "flcngdbUtilsCDefines.h"

/* ------------------------ Defines ---------------------------------------- */

// the amount of time the flcngdb will wait after issuing a ICD command
#define FLCNGDB_CMD_WAIT_TIME 1

// maximum filename/cmd length
#define FLCNGDB_FILENAME_MAX_LEN 255
#define FLCNGDB_CMD_MAX_LEN 255

// maximum function name length
#define FLCNGDB_FUNCNAME_MAX_LEN 50

// maximum object name length
#define FLCNGDB_OBJNAME_MAX_LEN  50

// number of lines to read before and after a bp source line
#define FLCNGDB_LOAD_SOURCE_LINES 8

// maximum size of the cmd input
#define FLCNGDB_MAX_INPUT_LEN 1024


/* ------------------------ Structures ------------------------------------- */

// saves the CPP Flcngdb class pointer between sessions
extern CFlcngdbUtils* pFlcngdbUtilsCls;

// holds the register mapping for the current falcon
extern FLCNGDB_REGISTER_MAP flcngdbRegisterMap;

// saves the interrupt status between sessions so they can be restored
extern LwU32 flcngdbSavedInterrupts;

/* ------------------------ Function Prototypes ---------------------------- */

// function called by pmu/dpu HALs to generate register maps based on falcon ver
void flcnGetFlcngdbRegisterMap_v04_00( LwU32 engineBase,
                                       FLCNGDB_REGISTER_MAP* registerMap);

// function prototypes for the actual debugger implementation
char* flcngdbReadLine(FILE* fp, LwBool returnData);
void flcngdbMenu(char* sessionID, char* pSymbPath);

#endif





