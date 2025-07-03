/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2018 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */
#include "gsp.h"
#include "hal.h"
#include "lwsym.h"

static OBJFLCN gspFlcn[MAX_GPUS];
static int gspObjInitialized[MAX_GPUS] = {0};


POBJFLCN gspGetFalconObject(void)
{
    if (!pGsp[indexGpu].gspIsSupported())
    {
        // GSP is not supported
        return NULL;
    }

    // Initialize the object if it is not done yet
    if (gspObjInitialized[indexGpu] == 0)
    {
        gspFlcn[indexGpu].pFCIF = pGsp[indexGpu].gspGetFalconCoreIFace();
        gspFlcn[indexGpu].pFEIF = pGsp[indexGpu].gspGetFalconEngineIFace();
        if(gspFlcn[indexGpu].pFEIF)
        {
            gspFlcn[indexGpu].engineName = gspFlcn[indexGpu].pFEIF->flcnEngGetEngineName();
            gspFlcn[indexGpu].engineBase = gspFlcn[indexGpu].pFEIF->flcnEngGetFalconBase();
        }
        pGsp[indexGpu].gspFillSymPath(gspFlcn);
        gspObjInitialized[indexGpu] = 1;
    }

    return &gspFlcn[indexGpu];
}

LwU32 gspGetDmemAccessPort(void)
{
    return 0;
}
