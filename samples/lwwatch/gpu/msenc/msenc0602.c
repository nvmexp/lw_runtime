/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2015-2017 by LWPU Corporation.  All rights reserved.  All information
 * contained herein is proprietary and confidential to LWPU Corporation.  Any
 * use, reproduction, or disclosure without the written permission of LWPU
 * Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

//-----------------------------------------------------
//
// msenc0602.c - LWENC routines
//
//-----------------------------------------------------

#include "pascal/gp102/dev_lwenc_pri_sw.h"
#include "pascal/gp102/dev_falcon_v4.h"
#include "class/clc2b7.h"

#include "msenc.h"
#include "hwref/lwutil.h"
#include "g_msenc_private.h"     // (rmconfig)  implementation prototypes

#define USE_LWENC_6_2

#if defined(USE_LWENC_6_2)
#include "msenc0602.h"
#endif

//-----------------------------------------------------
// msencIsSupported_v06_02
//-----------------------------------------------------
BOOL msencIsSupported_v06_02( LwU32 indexGpu )
{
    if(lwencId != LWWATCH_MSENC_0 && lwencId != LWWATCH_MSENC_1)
    {
        dprintf("Only MSENC0 and MSENC1 supported on this GPU\n");
        return FALSE;
    }

    pMsencPrivReg[0] = msenc0PrivReg_v06_02;
    pMsencPrivReg[1] = msenc1PrivReg_v06_02;
    pMsencMethodTable = msencMethodTable_v06_02;

    cmnMethodArraySize = CMNMETHODARRAYSIZEC1B7;
    appMethodArraySize = APPMETHODARRAYSIZEC1B7;

    engineId = lwencId;

    return TRUE;
}

//-----------------------------------------------------
// msencGetClassId_v06_02 - Returns Class ID supported
//                          for IP 06.2
//-----------------------------------------------------
LwU32
msencGetClassId_v06_02 (void)
{
    return LWC2B7_VIDEO_ENCODER;
}

//-----------------------------------------------------
// msencDumpDmem_v06_02 - Dumps LWENC data memory
//-----------------------------------------------------
LW_STATUS msencDumpDmem_v06_02( LwU32 indexGpu , LwU32 dmemSize)
{
    LW_STATUS status = LW_OK;
    LwU32 dmemSizeMax;
    LwU32 methodIdx;
    // these are the variables defined for use in parsing and printinf the methods and data
    LwU32 addrss, address2, u, i, comMthdOffs = 0, appMthdOffs = 0, classNum;
    LwU32* comMthd = NULL;
    LwU32* appMthd = NULL;

    if((cmnMethodArraySize == 0) || (appMethodArraySize == 0))
    {
        return LW_ERR_GENERIC;
    }

    comMthd = malloc(sizeof(LwU32)*cmnMethodArraySize);
    appMthd = malloc(sizeof(LwU32)*appMethodArraySize);

    if((comMthd == NULL) || (appMthd == NULL))
    {
        status = LW_ERR_NO_MEMORY;
        goto early_exit;
    }

    memset(comMthd, 0, sizeof(LwU32)*cmnMethodArraySize);
    memset(appMthd, 0, sizeof(LwU32)*appMethodArraySize);

    dprintf("lw: Dumping DMEM for LWENC%d\n", engineId);
    dmemSizeMax = (GPU_REG_IDX_RD_DRF(_PLWENC_FALCON, _HWCFG, engineId, _DMEM_SIZE)<<8);

    if(dmemSize > 0)
        dmemSize = min(dmemSizeMax, dmemSize);
    else
       dmemSize = dmemSizeMax;

    addrss      = LW_PLWENC_FALCON_DMEMD(engineId,0);
    address2    = LW_PLWENC_FALCON_DMEMC(engineId,0);
    classNum    = pMsenc[indexGpu].msencGetClassId();

    dprintf("\n");
    dprintf("lw: -- Gpu %u LWENC%d DMEM -- \n", indexGpu,engineId);
    dprintf("lw: -- Gpu %u LWENC%d DMEM SIZE =  0x%08x-- \n", indexGpu,engineId,dmemSize);
    //dprintf("lw:\n");
    dprintf("\nADDR: 03....00 07....04 0B....08 0F....0C 13....10 17....14 1B....18 1F....1C");
    dprintf("\n-----------------------------------------------------------------------------");

    for(u=0;u<(dmemSize+3)/4;u++)
    {
        i = (u<<(0?LW_PLWENC_FALCON_IMEMC_OFFS));
        GPU_REG_WR32(address2,i);
        if((u%8==0))
        {
            dprintf("\n%04X: ", 4*u);
        }
        dprintf("%08X ",  GPU_REG_RD32(addrss));
    }

    // get methods offset are in the DWORD#3 in dmem
    u = (3<<(0?LW_PLWENC_FALCON_IMEMC_OFFS));
    GPU_REG_WR32(address2,u);
    comMthdOffs = (GPU_REG_RD32(addrss)) >> 2;
    appMthdOffs = comMthdOffs + cmnMethodArraySize;

    for(u=0; u<cmnMethodArraySize;u++)
    {
        i = ((u+comMthdOffs)<<(0?LW_PLWENC_FALCON_IMEMC_OFFS));
        GPU_REG_WR32(address2,i);
        comMthd[u] = GPU_REG_RD32(addrss);
    }
    for(u=0; u<appMethodArraySize;u++)
    {
        i = ((u+appMthdOffs)<<(0?LW_PLWENC_FALCON_IMEMC_OFFS));
        GPU_REG_WR32(address2,i);
        appMthd[u] = GPU_REG_RD32(addrss);
    }

    dprintf("\n\n-----------------------------------------------------------------------\n");
    dprintf("%4s, %8s,    %4s, %8s,    %4s, %8s,    %4s, %8s\n",
                                "Mthd", "Data", "Mthd", "Data", "Mthd", "Data", "Mthd", "Data");
    dprintf("[COMMON METHODS]\n");
    for (u=0; u<cmnMethodArraySize; u+=4)
    {
        dprintf("%04X: %08X,    %04X: %08X,    %04X: %08X,    %04X: %08X\n",
        CMNMETHODBASE_v02+4*u, comMthd[u], CMNMETHODBASE_v02+4*(u+1), comMthd[u+1],
        CMNMETHODBASE_v02+4*(u+2), comMthd[u+2], CMNMETHODBASE_v02+4*(u+3), comMthd[u+3]);
    }
    dprintf("\n");
    dprintf("\n[APP METHODS]\n");
    for (u=0; u<appMethodArraySize; u+=4)
    {
        dprintf("%04X: %08X,    %04X: %08X,    %04X: %08X,    %04X: %08X\n",
        APPMETHODBASE_v02+4*u, appMthd[u], APPMETHODBASE_v02+4*(u+1), appMthd[u+1],
        APPMETHODBASE_v02+4*(u+2), appMthd[u+2], APPMETHODBASE_v02+4*(u+3), appMthd[u+3]);
    }

    // common methods
    // if this environment variable is present, parse and print out the methods
    if (getelw("LWW_CLASS_SDK") != NULL)
    {
        dprintf("\n[COMMON METHODS]\n");
        for(u=0;u<cmnMethodArraySize;u++)
        {
            if(parseClassHeader(classNum, CMNMETHODBASE_v02+4*u, comMthd[u]))
                dprintf("\n");
        }
        dprintf("\n");

        // app methods
        dprintf("\n[APP METHODS]\n");
        for(u=0;u<appMethodArraySize;u++)
        {
            if(parseClassHeader(classNum, APPMETHODBASE_v02+4*u, appMthd[u]))
                dprintf("\n");
        }
    }
    else
    {
        dprintf("\n[COMMON METHODS]\n");
        for(u=0;u<cmnMethodArraySize;u++)
        {
            for(methodIdx=0;;methodIdx++)
            {
                if(pMsencMethodTable[methodIdx].m_id == (CMNMETHODBASE_v02+4*u))
                {
                    msencPrintMethodData_v01_00(40,
                                                pMsencMethodTable[methodIdx].m_tag,
                                                pMsencMethodTable[methodIdx].m_id,
                                                comMthd[u]);
                    break;
                }
                else if (pMsencMethodTable[methodIdx].m_id == 0)
                {
                    break;
                }
            }
        }
        dprintf("\n");
        // app methods
        dprintf("\n[APP METHODS]\n");
        for(u=0;u<appMethodArraySize;u++)
        {
            for(methodIdx=0;;methodIdx++)
            {
                if(pMsencMethodTable[methodIdx].m_id == (APPMETHODBASE_v02+4*u))
                {
                    msencPrintMethodData_v01_00(40,
                                                pMsencMethodTable[methodIdx].m_tag,
                                                pMsencMethodTable[methodIdx].m_id,
                                                appMthd[u]);
                    break;
                }
                else if (pMsencMethodTable[methodIdx].m_id == 0)
                {
                    break;
                }
            }
        }
        dprintf("\n");
    }

early_exit:
    //deallocation
    free(comMthd);
    free(appMthd);
    comMthd = NULL;
    appMthd = NULL;

    return status;
}
