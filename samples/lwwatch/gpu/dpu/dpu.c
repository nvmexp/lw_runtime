/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2011-2018 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */
 
//*****************************************************
//
// DPU Non-Hal Functions
// dpu.c
//
//*****************************************************

/* ------------------------ Includes --------------------------------------- */
#include "dpu.h"
#include "lwsym.h"

/* ------------------------ Defines ---------------------------------------- */
/* ------------------------ Types definitions ------------------------------ */
/* ------------------------ Static variables ------------------------------- */
static OBJFLCN dpuFlcn[MAX_GPUS];
static int dpuObjInitialized[MAX_GPUS] = {0};


/* ------------------------ Functions -------------------------------------- */

/*!
 * Return symbol file path from directory of LWW_MANUAL_SDK. 
 *
 * @return Symbol file path
 */
const char*
dpuGetSymFilePath()
{
    return DIR_SLASH "dpu" DIR_SLASH "bin";
}

/*!
 * Return string of engine name
 *
 * @return Engine Name
 */
const char*
dpuGetEngineName()
{
    return "DPU";
}

/*!
 * Init and return the Falcon object that presents the DPU
 *
 * @return the Falcon object of the DPU
 */
POBJFLCN                
dpuGetFalconObject()
{
    // Init the object if it is not done yet
    if (dpuObjInitialized[indexGpu] == 0)
    {
        dpuFlcn[indexGpu].pFCIF = pDpu[indexGpu].dpuGetFalconCoreIFace();
        dpuFlcn[indexGpu].pFEIF = pDpu[indexGpu].dpuGetFalconEngineIFace();
        if(dpuFlcn[indexGpu].pFEIF)
        {
            dpuFlcn[indexGpu].engineName = dpuFlcn[indexGpu].pFEIF->flcnEngGetEngineName();
            dpuFlcn[indexGpu].engineBase = dpuFlcn[indexGpu].pFEIF->flcnEngGetFalconBase();
        }
        else
        {    
            dpuFlcn[indexGpu].engineName = "DPU";
        }
        dpuObjInitialized[indexGpu] = 1;
        sprintf(dpuFlcn[indexGpu].symPath, "%s%s", LWSYM_VIRUTAL_PATH, "flcnsym/");
        dpuFlcn[indexGpu].bSympathSet = TRUE;
    }

    return &dpuFlcn[indexGpu];
}

/*!
 * 
 * @return LwU32 DMEM access port for LwWatch
 */
LwU32
dpuGetDmemAccessPort()
{
    // Use port 2 for LwWatch (copied from legacy LwWatch code)
    return 2;
}

