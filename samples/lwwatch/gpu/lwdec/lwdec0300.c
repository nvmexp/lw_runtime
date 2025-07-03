/* _LWRM_COPYRIGHT_BEGIN_
*
* Copyright 2014-2015 by LWPU Corporation.  All rights reserved.  All information
* contained herein is proprietary and confidential to LWPU Corporation.  Any
* use, reproduction, or disclosure without the written permission of LWPU
* Corporation is prohibited.
*
* _LWRM_COPYRIGHT_END_
*/

//-----------------------------------------------------
//
// lwdec0300.c - LWDEC 3.0 routines
// 
//-----------------------------------------------------

#include "pascal/gp100/dev_pri_ringstation_sys.h"
#include "pascal/gp100/dev_lwdec_pri.h"
#include "pascal/gp100/dev_falcon_v4.h"
#include "pascal/gp100/dev_fifo.h"

#include "lwdec.h"

#include "g_lwdec_private.h"     // (rmconfig)  implementation prototypes

#include "class/clc1b0.h"

#include "lwdec0300.h"

//-----------------------------------------------------
// lwdecIsSupported_v03_00
//-----------------------------------------------------
BOOL lwdecIsSupported_v03_00(LwU32 indexGpu, LwU32 engineId)
{
    if (engineId != LWWATCH_LWDEC_0)
        return FALSE;

    pLwdecPrivReg[engineId] = lwdecPrivReg_v03_00;
    pLwdecMethodTable = lwdecMethodTable_v03_00;
    return TRUE;
}

//-----------------------------------------------------
// lwdecDumpDmem_v03_00 - Dumps LWDEC data memory
//-----------------------------------------------------
LW_STATUS lwdecDumpDmem_v03_00(LwU32 indexGpu, LwU32 engineId, LwU32 dmemSize)
{
    LW_STATUS status = LW_OK;
    LwU32 dmemSizeMax;
    // these are the variables defined for use in parsing and printing the methods and data
    LwU32 addrss, address2, u, i, comMthdOffs = 0, appMthdOffs = 0, classNum;
    LwU32 comMthd[CMNMETHODARRAYSIZEC1B0] = {0};
    LwU32 appMthd[APPMETHODARRAYSIZEC1B0] = {0};
    LwU32 methodIdx;
    LwU32 appMthdBase = 0;
    LwU32 cmnMthdBase = 0;
    LwU32 appId;

    if (engineId != LWWATCH_LWDEC_0)
        return LW_ERR_NOT_SUPPORTED;

    dmemSizeMax = (GPU_REG_RD_DRF(_PLWDEC_FALCON, _HWCFG, _DMEM_SIZE)<<8) ;

    if(dmemSize > 0)
        dmemSize = min(dmemSizeMax, dmemSize);
    else
       dmemSize = dmemSizeMax;

    addrss      = LW_PLWDEC_FALCON_DMEMD(0);
    address2    = LW_PLWDEC_FALCON_DMEMC(0);
    classNum    = LWC1B0_VIDEO_DECODER;

    dprintf("\n");
    dprintf("lw: -- Gpu %u LWDEC DMEM -- \n", indexGpu);
    dprintf("lw: -- Gpu %u LWDEC DMEM SIZE =  0x%08x-- \n", indexGpu,dmemSize);
    //dprintf("lw:\n");
    dprintf("\nADDR: 03....00 07....04 0B....08 0F....0C 13....10 17....14 1B....18 1F....1C");
    dprintf("\n-----------------------------------------------------------------------------");

    for(u=0;u<(dmemSize+3)/4;u++)
    {
        i = (u<<(0?LW_PLWDEC_FALCON_IMEMC_OFFS));
        GPU_REG_WR32(address2,i);
        if((u%8==0))
        {
            dprintf("\n%04X: ", 4*u);
        }
        dprintf("%08X ",  GPU_REG_RD32(addrss));
    }

    // get methods offset are in the DWORD#3 in dmem
    u = (3<<(0?LW_PLWDEC_FALCON_IMEMC_OFFS));

    GPU_REG_WR32(address2,u);
    comMthdOffs = (GPU_REG_RD32(addrss)) >> 2;
    appMthdOffs = comMthdOffs + CMNMETHODARRAYSIZEC1B0;

    for(u=0; u<CMNMETHODARRAYSIZEC1B0;u++)
    {
        i = ((u+comMthdOffs)<<(0?LW_PLWDEC_FALCON_IMEMC_OFFS));
        GPU_REG_WR32(address2,i);
        comMthd[u] = GPU_REG_RD32(addrss);
    }

    for(u=0; u<APPMETHODARRAYSIZEC1B0;u++)
    {
        i = ((u+appMthdOffs)<<(0?LW_PLWDEC_FALCON_IMEMC_OFFS));
        GPU_REG_WR32(address2,i);
        appMthd[u] = GPU_REG_RD32(addrss);
    }

    GPU_REG_WR32(address2, APP_ID_ADDRESS_IN_DMEM);
    appId = GPU_REG_RD32(addrss);

    switch (appId)
    {
    case LWC1B0_SET_APPLICATION_ID_ID_H264:
        appMthdBase = APPMETHODBASE_LWDEC_v03_H264;
        cmnMthdBase = CMNMETHODBASE_LWDEC_v03_CODECS;
        break;
    case LWC1B0_SET_APPLICATION_ID_ID_VP8:
        appMthdBase = APPMETHODBASE_LWDEC_v03_VP8;
        cmnMthdBase = CMNMETHODBASE_LWDEC_v03_CODECS;
        break;
    case LWC1B0_SET_APPLICATION_ID_ID_HEVC:
    case LWC1B0_SET_APPLICATION_ID_ID_HEVC_PARSER:
        appMthdBase = APPMETHODBASE_LWDEC_v03_HEVC;
        cmnMthdBase = CMNMETHODBASE_LWDEC_v03_CODECS;
        break;
    case LWC1B0_SET_APPLICATION_ID_ID_VP9:
        appMthdBase = APPMETHODBASE_LWDEC_v03_VP9;
        cmnMthdBase = CMNMETHODBASE_LWDEC_v03_CODECS;
        break;
    case LWC1B0_SET_APPLICATION_ID_ID_CTR64:
        appMthdBase = APPMETHODBASE_LWDEC_v03_CTR64;
        cmnMthdBase = CMNMETHODBASE_LWDEC_v03_CTR64;
        break;
    default:
        dprintf("No App Method running\n");
        break;
    }

    dprintf("\n\n-----------------------------------------------------------------------\n");
    dprintf("%4s, %8s,    %4s, %8s,    %4s, %8s,    %4s, %8s\n", 
            "Mthd", "Data", "Mthd", "Data", "Mthd", "Data", "Mthd", "Data");
    dprintf("[COMMON METHODS]\n");
    for (u=0; u<CMNMETHODARRAYSIZEC1B0; u++)
    {
        dprintf("%04X: %08X", cmnMthdBase+4*u, comMthd[u]);
        if (((u % 4) == 3) || u == (CMNMETHODARRAYSIZEC1B0 - 1))
        {
            dprintf("\n");
        }
        else
        {
            dprintf(",    ");
        }
    }

    dprintf("\n[APP METHODS]\n");

    for (u=0; u<APPMETHODARRAYSIZEC1B0; u++)
    {
        dprintf("%04X: %08X", appMthdBase+4*u, appMthd[u]);
        if (((u % 4) == 3) || u == (APPMETHODARRAYSIZEC1B0 - 1))
        {
            dprintf("\n");
        }
        else
        {
            dprintf(",    ");
        }
    }

    // common methods
    // if this environment variable is present, parse and print out the methods
    if (getelw("LWW_CLASS_SDK") != NULL)
    {
        dprintf("\n[COMMON METHODS]\n");
        for(u=0;u<CMNMETHODARRAYSIZEC1B0;u++)
        {
            if(parseClassHeader(classNum, cmnMthdBase+4*u, comMthd[u]))
                dprintf("\n");
        }
        dprintf("\n");

        // app methods
        dprintf("\n[APP METHODS]\n");
        for(u=0;u<APPMETHODARRAYSIZEC1B0;u++)
        {
            if(parseClassHeader(classNum, appMthdBase+4*u, appMthd[u]))
                dprintf("\n");
        }
    }
    else
    {
        dprintf("\n[COMMON METHODS]\n");
        for(u=0;u<CMNMETHODARRAYSIZEC1B0;u++)
        {
            for(methodIdx=0;;methodIdx++)
            {
                if(pLwdecMethodTable[methodIdx].m_id == (cmnMthdBase+4*u))
                {
                    lwdecPrintMethodData_v01_00(40,
                                                pLwdecMethodTable[methodIdx].m_tag, 
                                                pLwdecMethodTable[methodIdx].m_id, 
                                                comMthd[u]);
                    break;
                }
                else if (pLwdecMethodTable[methodIdx].m_id == 0)
                {
                    break;
                }
            }
        }
        dprintf("\n");
        // app methods
        dprintf("\n[APP METHODS]\n");
        for(u=0;u<APPMETHODARRAYSIZEC1B0;u++)
        {
            for(methodIdx=0;;methodIdx++)
            {
                if(pLwdecMethodTable[methodIdx].m_id == (appMthdBase+4*u))
                {
                    lwdecPrintMethodData_v01_00(40,
                                                pLwdecMethodTable[methodIdx].m_tag, 
                                                pLwdecMethodTable[methodIdx].m_id, 
                                                appMthd[u]);
                    break;
                }
                else if (pLwdecMethodTable[methodIdx].m_id == 0)
                {
                    break;
                }
            }
        }
        dprintf("\nDefine the LWW_CLASS_SDK environment variable to the location "
                "of the class header files to view parsed methods and data \n");
    }
    return status;
}
