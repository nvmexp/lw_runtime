/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2014-2017 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

//*****************************************************
//
// Non-Hal public functions for SEC2
// sec2.h
//
//*****************************************************

#ifndef _SEC2_H_
#define _SEC2_H_

/* ------------------------ Defines ---------------------------------------- */
#define SEC2_MUTEX_TIMEOUT_US (0x2000)
#define SEC2_RESET_TIMEOUT_US (0x1000)

/* ------------------------ Includes --------------------------------------- */
#include "os.h"
#include "hal.h"
#include "falcon.h"

#include "g_sec2_hal.h"     // (rmconfig)  public interfaces

/* ------------------------ Static Variables ------------------------------- */

/* ------------------------ Function Prototypes ---------------------------- */
POBJFLCN    sec2GetFalconObject  (void);
const char* sec2GetEngineName    (void);
LwU32       sec2GetDmemAccessPort(void);
const char* sec2GetSymFilePath   (void);
LwU32       sec2GetRegAddr       (LwU32);
LwU32       sec2RegRdAddr        (LwU32);
void        sec2RegWrAddr        (LwU32, LwU32);
void        initSec2ObjBaseAddr();

typedef struct
{
    LwU32 registerBase;
    LwU32 fbifBase;
    LwU32 (*getRegAddr)(LwU32);
    LwU32 (*readRegAddr)(LwU32);
    void  (*writeRegAddr)(LwU32, LwU32);
} OBJSEC2;

OBJSEC2 ObjSec2;
OBJSEC2 *pObjSec2;

#endif // _SEC2_H_
