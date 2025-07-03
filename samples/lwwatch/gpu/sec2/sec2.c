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
// SEC2 Non-Hal Functions
// sec2.c
//

/* ------------------------ Includes --------------------------------------- */
#include "sec2.h"
#include "hal.h"
#include "lwsym.h"

/* ------------------------ Defines ---------------------------------------- */
/* ------------------------ Types definitions ------------------------------ */
/* ------------------------ Static variables ------------------------------- */
static OBJFLCN sec2Flcn[MAX_GPUS];
static int sec2ObjInitialized[MAX_GPUS] = {0};

/* ------------------------ Functions -------------------------------------- */

/*!
 * Initialize and return the SEC2 Falcon object
 *
 * @return SEC2 Falcon object
 */
POBJFLCN                
sec2GetFalconObject()
{
    if (!pSec2[indexGpu].sec2IsSupported())
    {
        // SEC2 is not supported
        return NULL;
    }

    // Initialize the object if it is not done yet
    if (sec2ObjInitialized[indexGpu] == 0)
    {
        sec2Flcn[indexGpu].pFCIF = pSec2[indexGpu].sec2GetFalconCoreIFace();
        sec2Flcn[indexGpu].pFEIF = pSec2[indexGpu].sec2GetFalconEngineIFace();
        if(sec2Flcn[indexGpu].pFEIF)
        {
            sec2Flcn[indexGpu].engineName = sec2Flcn[indexGpu].pFEIF->flcnEngGetEngineName();
            sec2Flcn[indexGpu].engineBase = sec2Flcn[indexGpu].pFEIF->flcnEngGetFalconBase();
        }
        else
        {    
            sec2Flcn[indexGpu].engineName = "SEC2";
        }
        sprintf(sec2Flcn[indexGpu].symPath, "%s%s", LWSYM_VIRUTAL_PATH, "sec2/");
        sec2Flcn[indexGpu].bSympathSet = TRUE;
        sec2ObjInitialized[indexGpu] = 1;
    }

    return &sec2Flcn[indexGpu];
}

/*!
 * Return string of engine name
 *
 * @return Engine Name
 */
const char*
sec2GetEngineName()
{
    return "SEC2";
}

/*!
 * 
 * @return LwU32 DMEM access port for LwWatch
 */
LwU32
sec2GetDmemAccessPort()
{
    return 0;
}

/*!
 * Return symbol file path from directory of LWW_MANUAL_SDK. 
 *
 * @return Symbol file path
 */
const char*
sec2GetSymFilePath()
{
    return DIR_SLASH "sec2" DIR_SLASH "bin";
}

/*!
 * Returns the address of physical register of Falcon offset specified
 */
LwU32
sec2GetRegAddr(LwU32 offset)
{
    return ((pObjSec2->registerBase)+offset);
}

/*!
 * Returns the value stored at the physical address of Falcon offset specified
 */
LwU32
sec2RegRdAddr(LwU32 offset)
{
    return GPU_REG_RD32((pObjSec2->registerBase) + offset);
}

/*!
 * Writes the value at the physical address of Falcon offset specified
 */
void
sec2RegWrAddr(LwU32 offset, LwU32 value)
{
    GPU_REG_WR32(((pObjSec2->registerBase) + offset), value);
}

/*!
 * Initialize ObjSec2 base address
 */
void
initSec2ObjBaseAddr()
{
    if (!pSec2[indexGpu].sec2IsSupported())
    {
        // SEC2 is not supported
        pObjSec2 = NULL;
        dprintf("Sec2 is not Supported.\n");
    }
    else
    {
        pSec2[indexGpu].sec2ObjBaseAddr();
    }

}
