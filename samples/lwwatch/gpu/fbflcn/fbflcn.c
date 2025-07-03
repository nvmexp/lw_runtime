/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2018 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */
#include "fbflcn.h"
#include "hal.h"
#include "lwsym.h"

static OBJFLCN fbflcnFlcn[MAX_GPUS];
static int fbflcnObjInitialized[MAX_GPUS] = {0};


POBJFLCN fbflcnGetFalconObject(void)
{
    if (!pFbflcn[indexGpu].fbflcnIsSupported())
    {
        // Fbflcn is not supported
        return NULL;
    }

    // Initialize the object if it is not done yet
    if (fbflcnObjInitialized[indexGpu] == 0)
    {
        fbflcnFlcn[indexGpu].pFCIF = pFbflcn[indexGpu].fbflcnGetFalconCoreIFace();
        fbflcnFlcn[indexGpu].pFEIF = pFbflcn[indexGpu].fbflcnGetFalconEngineIFace();
        if(fbflcnFlcn[indexGpu].pFEIF)
        {
            fbflcnFlcn[indexGpu].engineName = fbflcnFlcn[indexGpu].pFEIF->flcnEngGetEngineName();
            fbflcnFlcn[indexGpu].engineBase = fbflcnFlcn[indexGpu].pFEIF->flcnEngGetFalconBase();
        }
        else
        {
            fbflcnFlcn[indexGpu].engineName = "FBFLCN";
        }
        sprintf(fbflcnFlcn[indexGpu].symPath, "%s%s", LWSYM_VIRUTAL_PATH, "fbflcn/");
        fbflcnFlcn[indexGpu].bSympathSet = TRUE;
        fbflcnObjInitialized[indexGpu] = 1;
    }

    return &fbflcnFlcn[indexGpu];
}

const char* fbflcnGetEngineName(void)
{
    return "FBFLCN";
}

LwU32 fbflcnGetDmemAccessPort(void)
{
    return 0;
}

const char* fbflcnGetSymFilePath(void)
{
    return DIR_SLASH "fbflcn" DIR_SLASH "bin";
}
