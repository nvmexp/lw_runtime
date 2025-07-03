/* _LWRM_COPYRIGHT_BEGIN_
*
* Copyright 2017-2018 by LWPU Corporation.  All rights reserved.  All information
* contained herein is proprietary and confidential to LWPU Corporation.  Any
* use, reproduction, or disclosure without the written permission of LWPU
* Corporation is prohibited.
*
* _LWRM_COPYRIGHT_END_
*/

//-----------------------------------------------------
//
// lwdec0401.c - LWDEC 4.1 routines
//
//-----------------------------------------------------

#include "lwdec.h"
#include "chip.h"
#include "g_lwdec_private.h"     // (rmconfig)  implementation prototypes
#include "class/clc4b0.h"
#include "lwdec0401.h"

#include "turing/tu102/dev_pri_ringstation_sys.h"

#include "turing/tu102/dev_lwdec_pri.h"
#include "turing/tu102/dev_falcon_v4.h"
#include "turing/tu102/dev_fifo.h"
#include "turing/tu102/dev_master.h"

#define FALCON_LWDEC_BASE(id)  (LW_FALCON_LWDEC0_BASE + (id * 0x4000))

//-----------------------------------------------------
// lwdecIsValidEngineId_v04_01
//-----------------------------------------------------
BOOL lwdecIsValidEngineId_v04_01(LwU32 indexGpu, LwU32 engineId)
{
    switch(engineId)
    {
        case LWWATCH_LWDEC_0:
             break;
        case LWWATCH_LWDEC_1:
             if (IsTU104() || IsTU106())
                break;
        case LWWATCH_LWDEC_2:
             if (IsTU106())
                break;
        default:
             dprintf("TU102/116/117 supports 1 lwdec, TU104 supports 2 lwdec, 106 supports 3 lwdec\n");
             return FALSE;
    }

    return TRUE;
}

//-----------------------------------------------------
// lwdecIsSupported_v04_01
//-----------------------------------------------------
BOOL lwdecIsSupported_v04_01(LwU32 indexGpu, LwU32 engineId)
{
    if (!pLwdec[indexGpu].lwdecIsValidEngineId(indexGpu, engineId))
        return FALSE;

    switch(engineId)
    {
        case LWWATCH_LWDEC_0:
             pLwdecPrivReg[LWWATCH_LWDEC_0] = lwdecPrivReg_v04_01_eng0;
             break;
        case LWWATCH_LWDEC_1:
             pLwdecPrivReg[LWWATCH_LWDEC_1] = lwdecPrivReg_v04_01_eng1;
             break;
        case LWWATCH_LWDEC_2:
             pLwdecPrivReg[LWWATCH_LWDEC_2] = lwdecPrivReg_v04_01_eng2; 
             break;
        default:
             return FALSE;
             break;
    }

    pLwdecMethodTable = lwdecMethodTable_v04_01;

    return TRUE;
}

//-----------------------------------------------------
// lwdecIsPrivBlocked_v04_01
//-----------------------------------------------------
BOOL lwdecIsPrivBlocked_v04_01(LwU32 indexGpu, LwU32 engineId)
{
    LwU32 idx;
    LwU32 bitmask;
    LwU32 regSysPrivFsConfig;

    if (!pLwdec[indexGpu].lwdecIsValidEngineId(indexGpu, engineId))
        return TRUE;

    // Bit-fields within LW_PPRIV_SYS_PRIV_FS_CONFIG denote priv access for video.
    // All video engines must have priv access for lwdec command support.
    switch(engineId)
    {
        case LWWATCH_LWDEC_0:
             idx = LW_PPRIV_SYS_PRI_MASTER_fecs2lwdec_pri0 >> 5;
             regSysPrivFsConfig = GPU_REG_RD32(LW_PPRIV_SYS_PRIV_FS_CONFIG(idx));
             bitmask = BIT(LW_PPRIV_SYS_PRI_MASTER_fecs2lwdec_pri0 - (idx << 5));
             break;
        case LWWATCH_LWDEC_1:
             idx = LW_PPRIV_SYS_PRI_MASTER_fecs2lwdec_pri1 >> 5;
             regSysPrivFsConfig = GPU_REG_RD32(LW_PPRIV_SYS_PRIV_FS_CONFIG(idx));
             bitmask = BIT(LW_PPRIV_SYS_PRI_MASTER_fecs2lwdec_pri1 - (idx << 5));
             break;
        case LWWATCH_LWDEC_2:
             idx = LW_PPRIV_SYS_PRI_MASTER_fecs2lwdec_pri2 >> 5;
             regSysPrivFsConfig = GPU_REG_RD32(LW_PPRIV_SYS_PRIV_FS_CONFIG(idx));
             bitmask = BIT(LW_PPRIV_SYS_PRI_MASTER_fecs2lwdec_pri2 - (idx << 5));
             break;
        default:
             return TRUE;
    }

    return ((regSysPrivFsConfig & bitmask) != bitmask);
}

//-----------------------------------------------------
// lwdecGetClassId_v04_01
//-----------------------------------------------------
LwU32
lwdecGetClassId_v04_01 (void)
{
    return LWC4B0_VIDEO_DECODER;
}

//-----------------------------------------------------
// lwdecPrintMethodData_v04_01
//-----------------------------------------------------
void lwdecPrintMethodData_v04_01(LwU32 clmn, char *tag, LwU32 method, LwU32 data)
{
    size_t len = strlen(tag);
    
    dprintf("lw: %s",tag);

    if((len>0)&&(len<(clmn+4)))
    {
        LwU32 i;
        for(i=0;i<clmn-len;i++)
        {
            dprintf(" ");
        }
    }
    dprintf("(0x%08X)  = 0x%08X\n",method,data);
}

//-----------------------------------------------------
// lwdecDumpImem_v04_01 - Dumps LWDEC instruction memory
//-----------------------------------------------------
LW_STATUS lwdecDumpImem_v04_01(LwU32 indexGpu, LwU32 engineId, LwU32 imemSize)
{
    LW_STATUS status = LW_OK;
    LwU32  imemSizeMax;
    LwU32 addrssImem=LW_PLWDEC_FALCON_IMEMD(engineId,0);
    LwU32 address2Imem=LW_PLWDEC_FALCON_IMEMC(engineId,0);
    LwU32 address2Imemt = LW_PLWDEC_FALCON_IMEMT(engineId,0);
    LwU32 u;
    LwU32 blk=0;

    if (!pLwdec[indexGpu].lwdecIsValidEngineId(indexGpu, engineId))
        return LW_ERR_NOT_SUPPORTED;

    imemSizeMax = (GPU_REG_IDX_RD_DRF(_PLWDEC_FALCON, _HWCFG, engineId, _IMEM_SIZE)<<8) ;
    if (imemSize > 0)
        imemSize = min(imemSizeMax, imemSize);
    else
        imemSize = imemSizeMax;

    dprintf("\n");
    dprintf("lw: -- Gpu %u LWDEC %d IMEM -- \n", indexGpu, engineId);    
    dprintf("lw: -- Gpu %u LWDEC %d IMEM SIZE =  0x%08x-- \n", indexGpu, engineId, imemSize);
    //dprintf("lw:\n");
    dprintf("\nADDR: 03....00 07....04 0B....08 0F....0C 13....10 17....14 1B....18 1F....1C");
    dprintf("\n-----------------------------------------------------------------------------");
    for(u=0;u<(imemSize+3)/4;u++)
    {
        LwU32 i;
        if((u%64)==0) {
            GPU_REG_WR32(address2Imemt, blk++);
        }
        i = (u<<(0?LW_PLWDEC_FALCON_IMEMC_OFFS));
        GPU_REG_WR32(address2Imem,i);
        if((u%8==0))
        {
            dprintf("\n%04X: ", 4*u);
        }
        dprintf("%08X ",  GPU_REG_RD32(addrssImem));
    }
    dprintf("\n");
    return status;  
}

//-----------------------------------------------------
// lwdecDumpDmem_v03_01 - Dumps LWDEC data memory
//-----------------------------------------------------
LW_STATUS lwdecDumpDmem_v04_01(LwU32 indexGpu, LwU32 engineId, LwU32 dmemSize)
{
    LW_STATUS status = LW_OK;
    LwU32 dmemSizeMax;
    // these are the variables defined for use in parsing and printing the methods and data
    LwU32 addrss, address2, u, i, comMthdOffs = 0, appMthdOffs = 0, classNum;
    LwU32 comMthd[CMNMETHODARRAYSIZEC4B0] = {0};
    LwU32 appMthd[APPMETHODARRAYSIZEC4B0] = {0};
    LwU32 methodIdx;
    LwU32 appMthdBase = 0;
    LwU32 cmnMthdBase = 0;
    LwU32 appId;

    if (!pLwdec[indexGpu].lwdecIsValidEngineId(indexGpu, engineId))
        return LW_ERR_NOT_SUPPORTED;

    dmemSizeMax = (GPU_REG_IDX_RD_DRF(_PLWDEC_FALCON, _HWCFG, engineId,  _DMEM_SIZE)<<8) ;

    if(dmemSize > 0)
        dmemSize = min(dmemSizeMax, dmemSize);
    else
       dmemSize = dmemSizeMax;

    addrss      = LW_PLWDEC_FALCON_DMEMD(engineId,0);
    address2    = LW_PLWDEC_FALCON_DMEMC(engineId,0);
    classNum    = pLwdec[indexGpu].lwdecGetClassId();

    dprintf("\n");
    dprintf("lw: -- Gpu %u LWDEC %d DMEM -- \n", indexGpu, engineId);
    dprintf("lw: -- Gpu %u LWDEC %d DMEM SIZE =  0x%08x-- \n", indexGpu, engineId, dmemSize);
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
    appMthdOffs = comMthdOffs + CMNMETHODARRAYSIZEC4B0;

    for(u=0; u<CMNMETHODARRAYSIZEC4B0;u++)
    {
        i = ((u+comMthdOffs)<<(0?LW_PLWDEC_FALCON_IMEMC_OFFS));
        GPU_REG_WR32(address2,i);
        comMthd[u] = GPU_REG_RD32(addrss);
    }

    for(u=0; u<APPMETHODARRAYSIZEC4B0;u++)
    {
        i = ((u+appMthdOffs)<<(0?LW_PLWDEC_FALCON_IMEMC_OFFS));
        GPU_REG_WR32(address2,i);
        appMthd[u] = GPU_REG_RD32(addrss);
    }

    GPU_REG_WR32(address2, APP_ID_ADDRESS_IN_DMEM);
    appId = GPU_REG_RD32(addrss);

    switch (appId)
    {
    case LWC4B0_SET_APPLICATION_ID_ID_H264:
        appMthdBase = APPMETHODBASE_LWDEC_v03_H264;
        cmnMthdBase = CMNMETHODBASE_LWDEC_v03_CODECS;
        break;
    case LWC4B0_SET_APPLICATION_ID_ID_VP8:
        appMthdBase = APPMETHODBASE_LWDEC_v03_VP8;
        cmnMthdBase = CMNMETHODBASE_LWDEC_v03_CODECS;
        break;
    case LWC4B0_SET_APPLICATION_ID_ID_HEVC:
    case LWC4B0_SET_APPLICATION_ID_ID_HEVC_PARSER:
        appMthdBase = APPMETHODBASE_LWDEC_v03_HEVC;
        cmnMthdBase = CMNMETHODBASE_LWDEC_v03_CODECS;
        break;
    case LWC4B0_SET_APPLICATION_ID_ID_VP9:
        appMthdBase = APPMETHODBASE_LWDEC_v03_VP9;
        cmnMthdBase = CMNMETHODBASE_LWDEC_v03_CODECS;
        break;
    case LWC4B0_SET_APPLICATION_ID_ID_CTR64:
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
    for (u=0; u<CMNMETHODARRAYSIZEC4B0; u++)
    {
        dprintf("%04X: %08X", cmnMthdBase+4*u, comMthd[u]);
        if (((u % 4) == 3) || u == (CMNMETHODARRAYSIZEC4B0 - 1))
        {
            dprintf("\n");
        }
        else
        {
            dprintf(",    ");
        }
    }

    dprintf("\n[APP METHODS]\n");

    for (u=0; u<APPMETHODARRAYSIZEC4B0; u++)
    {
        dprintf("%04X: %08X", appMthdBase+4*u, appMthd[u]);
        if (((u % 4) == 3) || u == (APPMETHODARRAYSIZEC4B0 - 1))
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
        for(u=0;u<CMNMETHODARRAYSIZEC4B0;u++)
        {
            if(parseClassHeader(classNum, cmnMthdBase+4*u, comMthd[u]))
                dprintf("\n");
        }
        dprintf("\n");

        // app methods
        dprintf("\n[APP METHODS]\n");
        for(u=0;u<APPMETHODARRAYSIZEC4B0;u++)
        {
            if(parseClassHeader(classNum, appMthdBase+4*u, appMthd[u]))
                dprintf("\n");
        }
    }
    else
    {
        dprintf("\n[COMMON METHODS]\n");
        for(u=0;u<CMNMETHODARRAYSIZEC4B0;u++)
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
        for(u=0;u<APPMETHODARRAYSIZEC4B0;u++)
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

//-----------------------------------------------------
// lwdecTestState_v04_01 - Test basic lwdec state
//-----------------------------------------------------
LW_STATUS lwdecTestState_v04_01(LwU32 indexGpu, LwU32 engineId)
{
    LW_STATUS    status = LW_OK;
    LwU32   regIntr;
    LwU32   regIntrEn;
    LwU32   data32;

    if (!pLwdec[indexGpu].lwdecIsValidEngineId(indexGpu, engineId))
        return LW_ERR_NOT_SUPPORTED;

    //check falcon interrupts
    regIntr = GPU_REG_RD32(LW_PLWDEC_FALCON_IRQSTAT(engineId));
    regIntrEn = GPU_REG_RD32(LW_PLWDEC_FALCON_IRQMASK(engineId));
    regIntr &= regIntrEn;

    if ( !DRF_VAL(_PLWDEC, _FALCON_IRQMASK, _GPTMR, regIntrEn))
        dprintf("lw: LW_PLWDEC_FALCON_IRQMASK_GPTMR disabled\n");

    if ( !DRF_VAL(_PLWDEC, _FALCON_IRQMASK, _WDTMR, regIntrEn))
        dprintf("lw: LW_PLWDEC_FALCON_IRQMASK_WDTMR disabled\n");

    if ( !DRF_VAL(_PLWDEC, _FALCON_IRQMASK, _MTHD, regIntrEn))
        dprintf("lw: LW_PLWDEC_FALCON_IRQMASK_MTHD disabled\n");

    if ( !DRF_VAL(_PLWDEC, _FALCON_IRQMASK, _CTXSW, regIntrEn))
        dprintf("lw: LW_PLWDEC_FALCON_IRQMASK_CTXSW disabled\n");

    if ( !DRF_VAL(_PLWDEC, _FALCON_IRQMASK, _HALT, regIntrEn))
        dprintf("lw: LW_PLWDEC_FALCON_IRQMASK_HALT disabled\n");

    if ( !DRF_VAL(_PLWDEC, _FALCON_IRQMASK, _EXTERR, regIntrEn))
        dprintf("lw: LW_PLWDEC_FALCON_IRQMASK_EXTERR disabled\n");

    if ( !DRF_VAL(_PLWDEC, _FALCON_IRQMASK, _SWGEN0, regIntrEn))
        dprintf("lw: LW_PLWDEC_FALCON_IRQMASK_SWGEN0 disabled\n");

    if ( !DRF_VAL(_PLWDEC, _FALCON_IRQMASK, _SWGEN1, regIntrEn))
        dprintf("lw: LW_PLWDEC_FALCON_IRQMASK_SWGEN1 disabled\n");

   
    //if any interrupt pending, set error
    if (regIntr != 0)
    {
        addUnitErr("\t LWDEC interrupts are pending\n");
        status = LW_ERR_GENERIC;
    }

    if ( DRF_VAL( _PLWDEC,_FALCON_IRQSTAT, _GPTMR, regIntr))
    {
        dprintf("lw: LW_PLWDEC_FALCON_IRQSTAT_GPTMR pending\n");

        dprintf("lw: LW_PLWDEC_FALCON_GPTMRINT:    0x%08x\n", 
            GPU_REG_RD32(LW_PLWDEC_FALCON_GPTMRINT(engineId)) );
        dprintf("lw: LW_PLWDEC_FALCON_GPTMRVAL:    0x%08x\n", 
            GPU_REG_RD32(LW_PLWDEC_FALCON_GPTMRVAL(engineId)) );
        
    }
    
    if ( DRF_VAL( _PLWDEC,_FALCON_IRQSTAT, _WDTMR, regIntr))
    {
        dprintf("lw: LW_PLWDEC_FALCON_IRQSTAT_WDTMR pending\n");
    }

    if ( DRF_VAL( _PLWDEC,_FALCON_IRQSTAT, _MTHD, regIntr))
    {
        dprintf("lw: LW_PLWDEC_FALCON_IRQSTAT_MTHD pending\n");

        dprintf("lw: LW_PLWDEC_FALCON_MTHDDATA_DATA:    0x%08x\n", 
            GPU_REG_RD32(LW_PLWDEC_FALCON_MTHDDATA(engineId)) );
        
        data32 = GPU_REG_RD32(LW_PLWDEC_FALCON_MTHDID(engineId));
        dprintf("lw: LW_PLWDEC_FALCON_MTHDID_ID:    0x%08x\n", 
           DRF_VAL( _PLWDEC,_FALCON_MTHDID, _ID, data32)  );
        dprintf("lw: LW_PLWDEC_FALCON_MTHDID_SUBCH:    0x%08x\n", 
           DRF_VAL( _PLWDEC,_FALCON_MTHDID, _SUBCH, data32)  );
        dprintf("lw: LW_PLWDEC_FALCON_MTHDID_PRIV:    0x%08x\n", 
           DRF_VAL( _PLWDEC,_FALCON_MTHDID, _PRIV, data32)  );
    }
    
    if ( DRF_VAL( _PLWDEC,_FALCON_IRQSTAT, _CTXSW, regIntr))
    {
        dprintf("lw: LW_PLWDEC_FALCON_IRQSTAT_CTXSW pending\n");
    }
    
    if ( DRF_VAL( _PLWDEC,_FALCON_IRQSTAT, _HALT, regIntr))
    {
        dprintf("lw: LW_PLWDEC_FALCON_IRQSTAT_HALT pending\n");
    }
    
    if ( DRF_VAL( _PLWDEC,_FALCON_IRQSTAT, _EXTERR, regIntr))
    {
        dprintf("lw: LW_PLWDEC_FALCON_IRQSTAT_EXTERR pending\n");
    }
    
    if ( DRF_VAL( _PLWDEC,_FALCON_IRQSTAT, _SWGEN0, regIntr))
    {
        dprintf("lw: LW_PLWDEC_FALCON_IRQSTAT_SWGEN0 pending\n");

        pFalcon[indexGpu].falconPrintMailbox(FALCON_LWDEC_BASE(engineId));
    }

    if ( DRF_VAL( _PLWDEC,_FALCON_IRQSTAT, _SWGEN1, regIntr))
    {
        dprintf("lw: LW_PLWDEC_FALCON_IRQSTAT_SWGEN1 pending\n");
    }

     //
    //print falcon states
    //Bit |  Signal meaning
    //0      FALCON busy
    //

    data32 = GPU_REG_RD32(LW_PLWDEC_FALCON_IDLESTATE(engineId));

    if ( DRF_VAL( _PLWDEC, _FALCON_IDLESTATE, _FALCON_BUSY, data32))
    {
        dprintf("lw: + LW_PLWDEC_FALCON_IDLESTATE_FALCON_BUSY\n");
        addUnitErr("\t LW_PLWDEC_FALCON_IDLESTATE_FALCON_BUSY\n");
        status = LW_ERR_GENERIC;
    }

  
    data32 = GPU_REG_RD32(LW_PLWDEC_FALCON_FHSTATE(engineId));
 
    if ( DRF_VAL( _PLWDEC, _FALCON_FHSTATE, _FALCON_HALTED, data32))
    {
        dprintf("lw: + LW_PLWDEC_FALCON_FHSTATE_FALCON_HALTED\n");
        addUnitErr("\t LW_PLWDEC_FALCON_FHSTATE_FALCON_HALTED\n");
        status = LW_ERR_GENERIC;
    }
    
    if ( DRF_VAL( _PLWDEC, _FALCON_FHSTATE, _ENGINE_FAULTED, data32))
    {
        dprintf("lw: + LW_PLWDEC_FALCON_FHSTATE_ENGINE_FAULTED\n");
        addUnitErr("\t LW_PLWDEC_FALCON_FHSTATE_ENGINE_FAULTED\n");
        status = LW_ERR_GENERIC;
    }
    
    if ( DRF_VAL( _PLWDEC, _FALCON_FHSTATE, _STALL_REQ, data32))
    {
        dprintf("lw: + LW_PLWDEC_FALCON_FHSTATE_STALL_REQ\n");
        addUnitErr("\t LW_PLWDEC_FALCON_FHSTATE_STALL_REQ\n");
        status = LW_ERR_GENERIC;
    }

    //print falcon ctl regs
    data32 = GPU_REG_RD32(LW_PLWDEC_FALCON_ENGCTL(engineId));
    
    if ( DRF_VAL( _PLWDEC, _FALCON_ENGCTL, _ILW_CONTEXT, data32))
    {
        dprintf("lw: + LW_PLWDEC_FALCON_ENGCTL_ILW_CONTEXT\n");
        addUnitErr("\t LW_PLWDEC_FALCON_ENGCTL_ILW_CONTEXT\n");
        status = LW_ERR_GENERIC;
    }

    if ( DRF_VAL( _PLWDEC, _FALCON_ENGCTL, _STALLREQ, data32))
    {
        dprintf("lw: + LW_PLWDEC_FALCON_ENGCTL_STALLREQ\n");
        addUnitErr("\t LW_PLWDEC_FALCON_ENGCTL_STALLREQ\n");
        status = LW_ERR_GENERIC;
    }

    data32 = GPU_REG_RD32(LW_PLWDEC_FALCON_CPUCTL(engineId));

    if ( DRF_VAL( _PLWDEC, _FALCON_CPUCTL, _IILWAL, data32))
    {
        dprintf("lw: + LW_PLWDEC_FALCON_CPUCTL_IILWAL\n");
        addUnitErr("\t LW_PLWDEC_FALCON_CPUCTL_IILWAL\n");
        status = LW_ERR_GENERIC;
    }

    if ( DRF_VAL( _PLWDEC, _FALCON_CPUCTL, _HALTED, data32))
    {
        dprintf("lw: + LW_PLWDEC_FALCON_CPUCTL_HALTED\n");
        addUnitErr("\t LW_PLWDEC_FALCON_CPUCTL_HALTED\n");
        status = LW_ERR_GENERIC;
    }

    if ( DRF_VAL( _PLWDEC, _FALCON_CPUCTL, _STOPPED, data32))
    {
        dprintf("lw: + LW_PLWDEC_FALCON_CPUCTL_STOPPED\n");
        addUnitErr("\t Warning: LW_PLWDEC_FALCON_CPUCTL_STOPPED\n");
        //status = LW_ERR_GENERIC;
    }

    // state of mthd/ctx interface 
    data32 = GPU_REG_RD32(LW_PLWDEC_FALCON_ITFEN(engineId));

    if (DRF_VAL( _PLWDEC, _FALCON_ITFEN, _CTXEN, data32))
    {
        dprintf("lw: + LW_PLWDEC_FALCON_ITFEN_CTXEN enabled\n");
             
        if (pFalcon[indexGpu].falconTestCtxState(FALCON_LWDEC_BASE(engineId), "PLWDEC") == LW_ERR_GENERIC)
        {
            dprintf("lw: Current ctx state invalid\n");
            addUnitErr("\t Current ctx state invalid\n");
            status = LW_ERR_GENERIC;
        }
        else
        {
            dprintf("lw: Current ctx state valid\n");
        }
    }
    else
    {
        dprintf("lw: + LW_PLWDEC_FALCON_ITFEN_CTXEN disabled\n");
    }

    if ( DRF_VAL( _PLWDEC, _FALCON_ITFEN, _MTHDEN, data32))
    {
        dprintf("lw: + LW_PLWDEC_FALCON_ITFEN_MTHDEN enabled\n");
    }
    else
    {
        dprintf("lw: + LW_PLWDEC_FALCON_ITFEN_MTHDEN disabled\n");
    }

    //check if falcon is hung (instr ptr)
    if ( pFalcon[indexGpu].falconTestPC(FALCON_LWDEC_BASE(engineId), "PLWDEC") == LW_ERR_GENERIC )
    {
        dprintf("lw: Falcon instruction pointer is stuck or invalid\n");
        
        //TODO: treat falcon PC errors as warnings now, need to report as error
        addUnitErr("\t Warning: Falcon instruction pointer is stuck or invalid\n");
        //status = LW_ERR_GENERIC;
    }

    return status;  
}

//-----------------------------------------------------
// lwdecPrintPriv_v04_01
//-----------------------------------------------------
void lwdecPrintPriv_v04_01(LwU32 clmn, char *tag, LwU32 id)
{
    size_t len = strlen(tag);
    
    dprintf("lw: %s",tag);

    if((len>0)&&(len<(clmn+4)))
    {
        LwU32 i;
        for(i=0;i<clmn-len;i++)
        {
            dprintf(" ");
        }
    }
    dprintf("(0x%08X)  = 0x%08X\n",id,GPU_REG_RD32(id));
}

//-----------------------------------------------------
// lwdecDumpPriv_v04_01 - Dumps LWDEC priv reg space
//-----------------------------------------------------
LW_STATUS lwdecDumpPriv_v04_01(LwU32 indexGpu, LwU32 engineId)
{
    LwU32 u;

    if (!pLwdec[indexGpu].lwdecIsValidEngineId(indexGpu, engineId))
        return LW_ERR_NOT_SUPPORTED;

    if (!pLwdecPrivReg[engineId])
    {
        dprintf("lw: -- Gpu %u LWDEC %d error: priv reg array uninitialized\n", indexGpu, engineId);
        return LW_ERR_ILWALID_PARAMETER;
    }

    dprintf("lw:\n");
    dprintf("lw: -- Gpu %u LWDEC %d priv registers -- \n", indexGpu, engineId);
    dprintf("lw:\n");

    for(u=0;;u++)
    {
        if(pLwdecPrivReg[engineId][u].m_id==0)
        {
            break;
        }
        pLwdec[indexGpu].lwdecPrintPriv(61,pLwdecPrivReg[engineId][u].m_tag,pLwdecPrivReg[engineId][u].m_id);
    }
    return LW_OK; 
}

//--------------------------------------------------------
// lwdecDisplayHwcfg_v04_01 - Display LWDEC HW config
//--------------------------------------------------------
LW_STATUS lwdecDisplayHwcfg_v04_01(LwU32 indexGpu, LwU32 engineId)
{
    LwU32 hwcfg, hwcfg1;

    if (!pLwdec[indexGpu].lwdecIsValidEngineId(indexGpu, engineId))
        return LW_ERR_NOT_SUPPORTED;

    dprintf("lw:\n");
    dprintf("lw: -- Gpu %u LWDEC %d HWCFG -- \n", indexGpu, engineId);
    dprintf("lw:\n");

    hwcfg  = GPU_REG_RD32(LW_PLWDEC_FALCON_HWCFG(engineId));
    dprintf("lw: LW_PLWDEC_FALCON_HWCFG:  0x%08x\n", hwcfg); 
    dprintf("lw:\n");
    dprintf("lw:  IMEM_SIZE:        0x%08X (or 0x%08X bytes)\n",
            DRF_VAL(_PLWDEC, _FALCON_HWCFG, _IMEM_SIZE, hwcfg),
            DRF_VAL(_PLWDEC, _FALCON_HWCFG, _IMEM_SIZE, hwcfg)<<8); 
    dprintf("lw:  DMEM_SIZE:        0x%08X (or 0x%08X bytes)\n",
            DRF_VAL(_PLWDEC, _FALCON_HWCFG, _DMEM_SIZE, hwcfg), 
            DRF_VAL(_PLWDEC, _FALCON_HWCFG, _DMEM_SIZE, hwcfg)<<8); 
    dprintf("lw:  METHODFIFO_DEPTH: 0x%08X\n", DRF_VAL(_PLWDEC, _FALCON_HWCFG, _METHODFIFO_DEPTH, hwcfg)); 
    dprintf("lw:  DMAQUEUE_DEPTH:   0x%08X\n", DRF_VAL(_PLWDEC, _FALCON_HWCFG, _DMAQUEUE_DEPTH, hwcfg)); 

    dprintf("lw:\n");

    hwcfg1 = GPU_REG_RD32(LW_PLWDEC_FALCON_HWCFG1(engineId));
    dprintf("lw: LW_PLWDEC_FALCON_HWCFG1: 0x%08x\n", hwcfg1); 
    dprintf("lw:\n");
    dprintf("lw:  CORE_REV:         0x%08X\n", DRF_VAL(_PLWDEC, _FALCON_HWCFG1, _CORE_REV, hwcfg1)); 
    dprintf("lw:  SELWRITY_MODEL:   0x%08X\n", DRF_VAL(_PLWDEC, _FALCON_HWCFG1, _SELWRITY_MODEL, hwcfg1)); 
    dprintf("lw:  IMEM_PORTS:       0x%08X\n", DRF_VAL(_PLWDEC, _FALCON_HWCFG1, _IMEM_PORTS, hwcfg1)); 
    dprintf("lw:  DMEM_PORTS:       0x%08X\n", DRF_VAL(_PLWDEC, _FALCON_HWCFG1, _DMEM_PORTS, hwcfg1)); 
    dprintf("lw:  TAG_WIDTH:        0x%08X\n", DRF_VAL(_PLWDEC, _FALCON_HWCFG1, _TAG_WIDTH, hwcfg1)); 

    return LW_OK;  
}

 /*
 Prints Falcon's Special purpose registers
0   IV0
1   IV1
3   EV
4   SP
5   PC
6   IMB
7   DMB
8   CSW
*/
// indx taken from Falcon 4.0 arch Table 3
LW_STATUS  lwdecDisplayFlcnSPR_v04_01(LwU32 indexGpu, LwU32 engineId)
{
    if (!pLwdec[indexGpu].lwdecIsValidEngineId(indexGpu, engineId))
        return LW_ERR_NOT_SUPPORTED;

    dprintf("lw:\n");
    dprintf("lw: -- Gpu %u LWDEC %d Special Purpose Registers -- \n", indexGpu, engineId);
    dprintf("lw:\n");

    GPU_REG_WR32(LW_PLWDEC_FALCON_ICD_CMD(engineId), 0x1008);
    dprintf("lw: LWDEC IV0 :    0x%08x\n", GPU_REG_RD32(LW_PLWDEC_FALCON_ICD_RDATA(engineId))); 
    GPU_REG_WR32(LW_PLWDEC_FALCON_ICD_CMD(engineId), 0x1108);
    dprintf("lw: LWDEC IV1 :    0x%08x\n", GPU_REG_RD32(LW_PLWDEC_FALCON_ICD_RDATA(engineId))); 
    GPU_REG_WR32(LW_PLWDEC_FALCON_ICD_CMD(engineId), 0x1308);
    dprintf("lw: LWDEC EV  :    0x%08x\n", GPU_REG_RD32(LW_PLWDEC_FALCON_ICD_RDATA(engineId))); 
    GPU_REG_WR32(LW_PLWDEC_FALCON_ICD_CMD(engineId), 0x1408);
    dprintf("lw: LWDEC SP  :    0x%08x\n", GPU_REG_RD32(LW_PLWDEC_FALCON_ICD_RDATA(engineId))); 
    GPU_REG_WR32(LW_PLWDEC_FALCON_ICD_CMD(engineId), 0x1508);
    dprintf("lw: LWDEC PC  :    0x%08x\n", GPU_REG_RD32(LW_PLWDEC_FALCON_ICD_RDATA(engineId))); 
    GPU_REG_WR32(LW_PLWDEC_FALCON_ICD_CMD(engineId), 0x1608);
    dprintf("lw: LWDEC IMB :    0x%08x\n", GPU_REG_RD32(LW_PLWDEC_FALCON_ICD_RDATA(engineId))); 
    GPU_REG_WR32(LW_PLWDEC_FALCON_ICD_CMD(engineId), 0x1708);
    dprintf("lw: LWDEC DMB :    0x%08x\n", GPU_REG_RD32(LW_PLWDEC_FALCON_ICD_RDATA(engineId))); 
    GPU_REG_WR32(LW_PLWDEC_FALCON_ICD_CMD(engineId), 0x1808);
    dprintf("lw: LWDEC CSW :    0x%08x\n", GPU_REG_RD32(LW_PLWDEC_FALCON_ICD_RDATA(engineId))); 
    dprintf("lw:\n\n");

    return LW_OK; 
}

/*!
 * @brief Checks if LWDEC DEBUG fuse is blown or not
 *
 */
LwBool
lwdecIsDebugMode_v04_01(LwU32 engineId)
{
    LwU32 ctlStat =  GPU_REG_RD32(LW_PLWDEC_SCP_CTL_STAT(engineId));

    if (!pLwdec[indexGpu].lwdecIsValidEngineId(indexGpu, engineId))
        return FALSE;

    return !FLD_TEST_DRF(_PLWDEC, _SCP_CTL_STAT, _DEBUG_MODE, _DISABLED, ctlStat);
}

