/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2012 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */
 
//*****************************************************
//
// FECS Non-Hal Functions
// fecs.c
//
//*****************************************************

/* ------------------------ Includes --------------------------------------- */
#include "fecs.h"
#include "hal.h"

/* ------------------------ Defines ---------------------------------------- */
/* ------------------------ Types definitions ------------------------------ */
/* ------------------------ Static variables ------------------------------- */
static OBJFLCN fecsFlcn[MAX_GPUS];
static int fecsObjInitialized[MAX_GPUS] = {0};


/* ------------------------ Functions -------------------------------------- */

/*!
 * Return string of engine name
 *
 * @return Engine Name
 */
const char*
fecsGetEngineName()
{
    return "FECS";
}

/*!
 * Init and return the Falcon object that presents FECS
 *
 * @return the Falcon object of FECS
 */
POBJFLCN                
fecsGetFalconObject()
{
    // Init the object if it is not done yet
    if (fecsObjInitialized[indexGpu] == 0)
    {
        fecsFlcn[indexGpu].pFCIF = pFecs[indexGpu].fecsGetFalconCoreIFace();
        fecsFlcn[indexGpu].pFEIF = pFecs[indexGpu].fecsGetFalconEngineIFace();
        if(fecsFlcn[indexGpu].pFEIF)
        {
            fecsFlcn[indexGpu].engineName = fecsFlcn[indexGpu].pFEIF->flcnEngGetEngineName();
            fecsFlcn[indexGpu].engineBase = fecsFlcn[indexGpu].pFEIF->flcnEngGetFalconBase();
        }
        else
        {    
            fecsFlcn[indexGpu].engineName = "FECS";
        }
        fecsObjInitialized[indexGpu] = 1;
    }

    return &fecsFlcn[indexGpu];
}

