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
// msenc0600.c - LWENC routines
// 
//-----------------------------------------------------

#include "pascal/gp100/dev_lwenc_pri_sw.h"
#include "pascal/gp100/dev_falcon_v4.h"
#include "class/clc1b7.h"

#include "msenc.h"
#include "hwref/lwutil.h"
#include "g_msenc_private.h"     // (rmconfig)  implementation prototypes

#define USE_LWENC_6_0

#if defined(USE_LWENC_6_0)
#include "msenc0600.h"
#endif

//-----------------------------------------------------
// msencIsSupported_v06_00
//-----------------------------------------------------
BOOL msencIsSupported_v06_00( LwU32 indexGpu )
{
    if(lwencId != LWWATCH_MSENC_0 && lwencId != LWWATCH_MSENC_1 && lwencId != LWWATCH_MSENC_2)
    {
        dprintf("Only MSENC0, MSENC1 and MSENC2 supported on this GPU\n");
        return FALSE;
    }

    pMsencPrivReg[0] = msenc0PrivReg_v06_00;
    pMsencPrivReg[1] = msenc1PrivReg_v06_00;
    pMsencPrivReg[2] = msenc2PrivReg_v06_00;
    pMsencMethodTable = msencMethodTable_v06_00;

    engineId = lwencId;

    return TRUE;
}

//-----------------------------------------------------
// msencDumpImem_v06_00 - Dumps LWENC instruction memory
//-----------------------------------------------------
LW_STATUS msencDumpImem_v06_00( LwU32 indexGpu , LwU32 imemSize)
{
    LW_STATUS status = LW_OK;
    LwU32 imemSizeMax;

    LwU32 addrssImem;
    LwU32 address2Imem;
    LwU32 address2Imemt;
    LwU32 u;
    LwU32 blk;

    // IF LWENC is not specified, Operate for all available LWENC engines

    dprintf("Dumping IMEM for LWENC%d\n", engineId);
    addrssImem    = LW_PLWENC_FALCON_IMEMD(engineId,0);
    address2Imem  = LW_PLWENC_FALCON_IMEMC(engineId,0);
    address2Imemt = LW_PLWENC_FALCON_IMEMT(engineId,0);
    blk=0;
    imemSizeMax = (GPU_REG_IDX_RD_DRF(_PLWENC_FALCON, _HWCFG, engineId, _IMEM_SIZE)<<8) ;
    if (imemSize > 0)
        imemSize = min(imemSizeMax, imemSize);
    else
        imemSize = imemSizeMax;

    dprintf("\n");
    dprintf("lw: -- Gpu %u LWENC%d IMEM -- \n", indexGpu, engineId);
    dprintf("lw: -- Gpu %u LWENC%d IMEM SIZE =  0x%08x-- \n", indexGpu,engineId,imemSize);
    //dprintf("lw:\n");
    dprintf("\nADDR: 03....00 07....04 0B....08 0F....0C 13....10 17....14 1B....18 1F....1C");
    dprintf("\n-----------------------------------------------------------------------------");
    for(u=0;u<(imemSize+3)/4;u++)
    {
        LwU32 i;
        if((u%64)==0)
        {
            GPU_REG_WR32(address2Imemt, blk++);
        }
        i = (u<<(0?LW_PLWENC_FALCON_IMEMC_OFFS));
        GPU_REG_WR32(address2Imem,i);
        if((u%8==0))
        {
            dprintf("\n%04X: ", 4*u);
        }
        dprintf("%08X ",  GPU_REG_RD32(addrssImem));
    }
    dprintf("\n\n");
    return status;
}

//-----------------------------------------------------
// msencDumpDmem_v06_00 - Dumps LWENC data memory
//-----------------------------------------------------
LW_STATUS msencDumpDmem_v06_00( LwU32 indexGpu , LwU32 dmemSize)
{
    LW_STATUS status = LW_OK;
    LwU32 dmemSizeMax;
    LwU32 methodIdx;
    // these are the variables defined for use in parsing and printinf the methods and data
    LwU32 addrss, address2, u, i, comMthdOffs = 0, appMthdOffs = 0, classNum;
    LwU32 comMthd[CMNMETHODARRAYSIZEC1B7] = {0};
    LwU32 appMthd[APPMETHODARRAYSIZEC1B7] = {0};

    dprintf("lw: Dumping DMEM for LWENC%d\n", engineId);
    dmemSizeMax = (GPU_REG_IDX_RD_DRF(_PLWENC_FALCON, _HWCFG, engineId, _DMEM_SIZE)<<8);

    if(dmemSize > 0)
        dmemSize = min(dmemSizeMax, dmemSize);
    else
       dmemSize = dmemSizeMax;

    addrss      = LW_PLWENC_FALCON_DMEMD(engineId,0);
    address2    = LW_PLWENC_FALCON_DMEMC(engineId,0);
    classNum    = LWC1B7_VIDEO_ENCODER;

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
    appMthdOffs = comMthdOffs + CMNMETHODARRAYSIZEC1B7;

    for(u=0; u<CMNMETHODARRAYSIZEC1B7;u++)
    {
        i = ((u+comMthdOffs)<<(0?LW_PLWENC_FALCON_IMEMC_OFFS));
        GPU_REG_WR32(address2,i);
        comMthd[u] = GPU_REG_RD32(addrss);
    }
    for(u=0; u<APPMETHODARRAYSIZEC1B7;u++)
    {
        i = ((u+appMthdOffs)<<(0?LW_PLWENC_FALCON_IMEMC_OFFS));
        GPU_REG_WR32(address2,i);
        appMthd[u] = GPU_REG_RD32(addrss);
    }

    dprintf("\n\n-----------------------------------------------------------------------\n");
    dprintf("%4s, %8s,    %4s, %8s,    %4s, %8s,    %4s, %8s\n",
                                "Mthd", "Data", "Mthd", "Data", "Mthd", "Data", "Mthd", "Data");
    dprintf("[COMMON METHODS]\n");
    for (u=0; u<CMNMETHODARRAYSIZEC1B7; u++)
    {
        dprintf("%04X: %08X", CMNMETHODBASE_v02+4*u, comMthd[u]);
        if (((u % 4) == 3) || u == (CMNMETHODARRAYSIZEC1B7 - 1))
        {
            dprintf("\n");
        }
        else
        {
            dprintf(",    ");
        }
    }
    dprintf("\n");
    dprintf("\n[APP METHODS]\n");
    for (u=0; u<APPMETHODARRAYSIZEC1B7; u++)
    {
        dprintf("%04X: %08X", APPMETHODBASE_v02+4*u, appMthd[u]);
        if (((u % 4) == 3) || u == (APPMETHODARRAYSIZEC1B7 - 1))
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
        for(u=0;u<CMNMETHODARRAYSIZEC1B7;u++)
        {
            if(parseClassHeader(classNum, CMNMETHODBASE_v02+4*u, comMthd[u]))
                dprintf("\n");
        }
        dprintf("\n");

        // app methods
        dprintf("\n[APP METHODS]\n");
        for(u=0;u<APPMETHODARRAYSIZEC1B7;u++)
        {
            if(parseClassHeader(classNum, APPMETHODBASE_v02+4*u, appMthd[u]))
                dprintf("\n");
        }
    }
    else
    {
    #if defined(USE_LWENC_6_0)
        dprintf("\n[COMMON METHODS]\n");
        for(u=0;u<CMNMETHODARRAYSIZEC1B7;u++)
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
        for(u=0;u<APPMETHODARRAYSIZEC1B7;u++)
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
    #else
        dprintf("\nDefine the LWW_CLASS_SDK environment variable to the location \
                    of the class header files to view parsed methods and data \n");
    #endif
    }
    return status;
}

//-----------------------------------------------------
// msencTestState_v06_00 - Test basic lwenc state
//-----------------------------------------------------
LW_STATUS msencTestState_v06_00( LwU32 indexGpu )
{
    LW_STATUS    status = LW_OK;
    LwU32   regIntr;
    LwU32   regIntrEn;
    LwU32   data32;

    dprintf("lw: Checking states of LWENC%d\n", engineId);
    //check falcon interrupts
    regIntr = GPU_REG_RD32(LW_PLWENC_FALCON_IRQSTAT(engineId));
    regIntrEn = GPU_REG_RD32(LW_PLWENC_FALCON_IRQMASK(engineId));
    regIntr &= regIntrEn;

    if ( !DRF_VAL(_PLWENC, _FALCON_IRQMASK, _GPTMR, regIntrEn))
        dprintf("lw: LW_PLWENC_FALCON_IRQMASK_GPTMR disabled\n");

    if ( !DRF_VAL(_PLWENC, _FALCON_IRQMASK, _WDTMR, regIntrEn))
        dprintf("lw: LW_PLWENC_FALCON_IRQMASK_WDTMR disabled\n");

    if ( !DRF_VAL(_PLWENC, _FALCON_IRQMASK, _MTHD, regIntrEn))
        dprintf("lw: LW_PLWENC_FALCON_IRQMASK_MTHD disabled\n");

    if ( !DRF_VAL(_PLWENC, _FALCON_IRQMASK, _CTXSW, regIntrEn))
        dprintf("lw: LW_PLWENC_FALCON_IRQMASK_CTXSW disabled\n");

    if ( !DRF_VAL(_PLWENC, _FALCON_IRQMASK, _HALT, regIntrEn))
        dprintf("lw: LW_PLWENC_FALCON_IRQMASK_HALT disabled\n");

    if ( !DRF_VAL(_PLWENC, _FALCON_IRQMASK, _EXTERR, regIntrEn))
        dprintf("lw: LW_PLWENC_FALCON_IRQMASK_EXTERR disabled\n");

    if ( !DRF_VAL(_PLWENC, _FALCON_IRQMASK, _SWGEN0, regIntrEn))
        dprintf("lw: LW_PLWENC_FALCON_IRQMASK_SWGEN0 disabled\n");

    if ( !DRF_VAL(_PLWENC, _FALCON_IRQMASK, _SWGEN1, regIntrEn))
        dprintf("lw: LW_PLWENC_FALCON_IRQMASK_SWGEN1 disabled\n");

    //if any interrupt pending, set error
    if (regIntr != 0)
    {
        addUnitErr("\t LWENC interrupts are pending\n");
        status = LW_ERR_GENERIC;
    }

    if ( DRF_VAL( _PLWENC,_FALCON_IRQSTAT, _GPTMR, regIntr))
    {
        dprintf("lw: LW_PLWENC_FALCON_IRQSTAT_GPTMR pending\n");
        dprintf("lw: LW_PLWENC_FALCON_GPTMRINT:    0x%08x\n", 
        GPU_REG_RD32(LW_PLWENC_FALCON_GPTMRINT(0)) );
        dprintf("lw: LW_PLWENC_FALCON_GPTMRVAL:    0x%08x\n", 
        GPU_REG_RD32(LW_PLWENC_FALCON_GPTMRVAL(0)) );
    }

    if ( DRF_VAL( _PLWENC,_FALCON_IRQSTAT, _WDTMR, regIntr))
    {
        dprintf("lw: LW_PLWENC_FALCON_IRQSTAT_WDTMR pending\n");
    }

    if ( DRF_VAL( _PLWENC,_FALCON_IRQSTAT, _MTHD, regIntr))
    {
        dprintf("lw: LW_PLWENC_FALCON_IRQSTAT_MTHD pending\n");
        dprintf("lw: LW_PLWENC_FALCON_MTHDDATA_DATA:    0x%08x\n", 
        GPU_REG_RD32(LW_PLWENC_FALCON_MTHDDATA(engineId)) );

        data32 = GPU_REG_RD32(LW_PLWENC_FALCON_MTHDID(engineId));
        dprintf("lw: LW_PLWENC_FALCON_MTHDID_ID:    0x%08x\n", 
        DRF_VAL( _PLWENC,_FALCON_MTHDID, _ID, data32)  );
        dprintf("lw: LW_PLWENC_FALCON_MTHDID_SUBCH:    0x%08x\n", 
        DRF_VAL( _PLWENC,_FALCON_MTHDID, _SUBCH, data32)  );
        dprintf("lw: LW_PLWENC_FALCON_MTHDID_PRIV:    0x%08x\n", 
        DRF_VAL( _PLWENC,_FALCON_MTHDID, _PRIV, data32)  );
    }

    if ( DRF_VAL( _PLWENC,_FALCON_IRQSTAT, _CTXSW, regIntr))
    {
        dprintf("lw: LW_PLWENC_FALCON_IRQSTAT_CTXSW pending\n");
    }

    if ( DRF_VAL( _PLWENC,_FALCON_IRQSTAT, _HALT, regIntr))
    {
        dprintf("lw: LW_PLWENC_FALCON_IRQSTAT_HALT pending\n");
    }

    if ( DRF_VAL( _PLWENC,_FALCON_IRQSTAT, _EXTERR, regIntr))
    {
        dprintf("lw: LW_PLWENC_FALCON_IRQSTAT_EXTERR pending\n");
    }

    if ( DRF_VAL( _PLWENC,_FALCON_IRQSTAT, _SWGEN0, regIntr))
    {
        dprintf("lw: LW_PLWENC_FALCON_IRQSTAT_SWGEN0 pending\n");

        pFalcon[indexGpu].falconPrintMailbox(LW_FALCON_LWENC_BASE);
    }

    if ( DRF_VAL( _PLWENC,_FALCON_IRQSTAT, _SWGEN1, regIntr))
    {
        dprintf("lw: LW_PLWENC_FALCON_IRQSTAT_SWGEN1 pending\n");
    }

    //
    //print falcon states
    //Bit |  Signal meaning
    //0      FALCON busy
    //

    data32 = GPU_REG_RD32(LW_PLWENC_FALCON_IDLESTATE(engineId));

    if ( DRF_VAL( _PLWENC, _FALCON_IDLESTATE, _FALCON_BUSY, data32))
    {
        dprintf("lw: + LW_PLWENC_FALCON_IDLESTATE_FALCON_BUSY\n");
        addUnitErr("\t LW_PLWENC_FALCON_IDLESTATE_FALCON_BUSY\n");
        status = LW_ERR_GENERIC;
    }

    data32 = GPU_REG_RD32(LW_PLWENC_FALCON_FHSTATE(engineId));

    if ( DRF_VAL( _PLWENC, _FALCON_FHSTATE, _FALCON_HALTED, data32))
    {
        dprintf("lw: + LW_PLWENC_FALCON_FHSTATE_FALCON_HALTED\n");
        addUnitErr("\t LW_PLWENC_FALCON_FHSTATE_FALCON_HALTED\n");
        status = LW_ERR_GENERIC;
    }

    if ( DRF_VAL( _PLWENC, _FALCON_FHSTATE, _ENGINE_FAULTED, data32))
    {
        dprintf("lw: + LW_PLWENC_FALCON_FHSTATE_ENGINE_FAULTED\n");
        addUnitErr("\t LW_PLWENC_FALCON_FHSTATE_ENGINE_FAULTED\n");
        status = LW_ERR_GENERIC;
    }

    if ( DRF_VAL( _PLWENC, _FALCON_FHSTATE, _STALL_REQ, data32))
    {
        dprintf("lw: + LW_PLWENC_FALCON_FHSTATE_STALL_REQ\n");
        addUnitErr("\t LW_PLWENC_FALCON_FHSTATE_STALL_REQ\n");
        status = LW_ERR_GENERIC;
    }

    //print falcon ctl regs
    data32 = GPU_REG_RD32(LW_PLWENC_FALCON_ENGCTL(engineId));

    if ( DRF_VAL( _PLWENC, _FALCON_ENGCTL, _ILW_CONTEXT, data32))
    {
        dprintf("lw: + LW_PLWENC_FALCON_ENGCTL_ILW_CONTEXT\n");
        addUnitErr("\t LW_PLWENC_FALCON_ENGCTL_ILW_CONTEXT\n");
        status = LW_ERR_GENERIC;
    }

    if ( DRF_VAL( _PLWENC, _FALCON_ENGCTL, _STALLREQ, data32))
    {
        dprintf("lw: + LW_PLWENC_FALCON_ENGCTL_STALLREQ\n");
        addUnitErr("\t LW_PLWENC_FALCON_ENGCTL_STALLREQ\n");
        status = LW_ERR_GENERIC;
    }

    data32 = GPU_REG_RD32(LW_PLWENC_FALCON_CPUCTL(engineId));

    if ( DRF_VAL( _PLWENC, _FALCON_CPUCTL, _IILWAL, data32))
    {
        dprintf("lw: + LW_PLWENC_FALCON_CPUCTL_IILWAL\n");
        addUnitErr("\t LW_PLWENC_FALCON_CPUCTL_IILWAL\n");
        status = LW_ERR_GENERIC;
    }

    if ( DRF_VAL( _PLWENC, _FALCON_CPUCTL, _HALTED, data32))
    {
        dprintf("lw: + LW_PLWENC_FALCON_CPUCTL_HALTED\n");
        addUnitErr("\t LW_PLWENC_FALCON_CPUCTL_HALTED\n");
        status = LW_ERR_GENERIC;
    }

    if ( DRF_VAL( _PLWENC, _FALCON_CPUCTL, _STOPPED, data32))
    {
        dprintf("lw: + LW_PLWENC_FALCON_CPUCTL_STOPPED\n");
        addUnitErr("\t Warning: LW_PLWENC_FALCON_CPUCTL_STOPPED\n");
        //status = LW_ERR_GENERIC;
    }

    // state of mthd/ctx interface 
    data32 = GPU_REG_RD32(LW_PLWENC_FALCON_ITFEN(engineId));

    if (DRF_VAL( _PLWENC, _FALCON_ITFEN, _CTXEN, data32))
    {
        dprintf("lw: + LW_PLWENC_FALCON_ITFEN_CTXEN enabled\n");

        if (pFalcon[indexGpu].falconTestCtxState(LW_FALCON_LWENC_BASE, "PLWENC") == LW_ERR_GENERIC)
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
        dprintf("lw: + LW_PLWENC_FALCON_ITFEN_CTXEN disabled\n");
    }

    if ( DRF_VAL( _PLWENC, _FALCON_ITFEN, _MTHDEN, data32))
    {
        dprintf("lw: + LW_PLWENC_FALCON_ITFEN_MTHDEN enabled\n");
    }
    else
    {
        dprintf("lw: + LW_PLWENC_FALCON_ITFEN_MTHDEN disabled\n");
    }

    //check if falcon is hung (instr ptr)
    if ( pFalcon[indexGpu].falconTestPC(LW_FALCON_LWENC_BASE, "PLWENC") == LW_ERR_GENERIC )
    {
        dprintf("lw: Falcon instruction pointer is stuck or invalid\n");

        //TODO: treat falcon PC errors as warnings now, need to report as error
        addUnitErr("\t Warning: Falcon instruction pointer is stuck or invalid\n");
        //status = LW_ERR_GENERIC;
    }
    return status;
}

//-----------------------------------------------------
// msencPrintPriv_v06_00
//-----------------------------------------------------
void msencPrintPriv_v06_00(LwU32 clmn, char *tag, LwU32 id)
{
    size_t len = strlen(tag);

    dprintf("lw: %s",tag);

    if((len>0)&&(len<clmn))
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
// msencDumpPriv_v06_00 - Dumps LWENC priv reg space
//-----------------------------------------------------
LW_STATUS msencDumpPriv_v06_00(LwU32 indexGpu)
{
    LwU32 u;

    dprintf("lw:\n");
    dprintf("lw: -- Gpu %u LWENC%d priv registers -- \n", indexGpu,engineId);
    dprintf("lw:\n");

    for(u=0;;u++)
    {
        if(pMsencPrivReg[engineId][u].m_id==0)
        {
            break;
        }

        pMsenc[indexGpu].msencPrintPriv(70,pMsencPrivReg[engineId][u].m_tag,
                    pMsencPrivReg[engineId][u].m_id);
    }
    return LW_OK; 
}

//--------------------------------------------------------
// msencDisplayHwcfg_v06_00 - Display LWENC HW config
//--------------------------------------------------------
LW_STATUS msencDisplayHwcfg_v06_00(LwU32 indexGpu)
{
    LwU32 hwcfg, hwcfg1;

    dprintf("lw:\n");
    dprintf("lw: -- Gpu %u LWENC%d HWCFG -- \n", indexGpu,engineId);
    dprintf("lw:\n");

    hwcfg  = GPU_REG_RD32(LW_PLWENC_FALCON_HWCFG(engineId));

    dprintf("lw: LW_PLWENC_FALCON_HWCFG:  0x%08x\n", hwcfg); 
    dprintf("lw:\n");
    dprintf("lw:  IMEM_SIZE:        0x%08X (or 0x%08X bytes)\n",

    DRF_VAL(_PLWENC, _FALCON_HWCFG, _IMEM_SIZE, hwcfg),
    DRF_VAL(_PLWENC, _FALCON_HWCFG, _IMEM_SIZE, hwcfg)<<8);

    dprintf("lw:  DMEM_SIZE:        0x%08X (or 0x%08X bytes)\n",

    DRF_VAL(_PLWENC, _FALCON_HWCFG, _DMEM_SIZE, hwcfg),
    DRF_VAL(_PLWENC, _FALCON_HWCFG, _DMEM_SIZE, hwcfg)<<8);

    dprintf("lw:  METHODFIFO_DEPTH: 0x%08X\n", DRF_VAL(_PLWENC, _FALCON_HWCFG, _METHODFIFO_DEPTH, hwcfg));
    dprintf("lw:  DMAQUEUE_DEPTH:   0x%08X\n", DRF_VAL(_PLWENC, _FALCON_HWCFG, _DMAQUEUE_DEPTH, hwcfg));
    dprintf("lw:\n");

    hwcfg1 = GPU_REG_RD32(LW_PLWENC_FALCON_HWCFG1(engineId));

    dprintf("lw: LW_PLWENC_FALCON_HWCFG1: 0x%08x\n", hwcfg1); 
    dprintf("lw:\n");
    dprintf("lw:  CORE_REV:     0x%08X\n", DRF_VAL(_PLWENC, _FALCON_HWCFG1, _CORE_REV, hwcfg1));
    dprintf("lw:  SELWRITY_MODEL:   0x%08X\n", DRF_VAL(_PLWENC, _FALCON_HWCFG1, _SELWRITY_MODEL, hwcfg1));
    dprintf("lw:  IMEM_PORTS:       0x%08X\n", DRF_VAL(_PLWENC, _FALCON_HWCFG1, _IMEM_PORTS, hwcfg1));
    dprintf("lw:  DMEM_PORTS:       0x%08X\n", DRF_VAL(_PLWENC, _FALCON_HWCFG1, _DMEM_PORTS, hwcfg1));
    dprintf("lw:  TAG_WIDTH:        0x%08X\n", DRF_VAL(_PLWENC, _FALCON_HWCFG1, _TAG_WIDTH, hwcfg1));

    return LW_OK;
}

 /*
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
LW_STATUS  msencDisplayFlcnSPR_v06_00(LwU32 indexGpu)
{
    dprintf("lw:\n");
    dprintf("lw: -- Gpu %u LWENC%d Special Purpose Registers -- \n", indexGpu,engineId);
    dprintf("lw:\n");

    GPU_REG_WR32(LW_PLWENC_FALCON_ICD_CMD(engineId), 0x1008);
    dprintf("lw: LWENC IV0 :    0x%08x\n", GPU_REG_RD32(LW_PLWENC_FALCON_ICD_RDATA(engineId))); 
    GPU_REG_WR32(LW_PLWENC_FALCON_ICD_CMD(engineId), 0x1108);
    dprintf("lw: LWENC IV1 :    0x%08x\n", GPU_REG_RD32(LW_PLWENC_FALCON_ICD_RDATA(engineId))); 
    GPU_REG_WR32(LW_PLWENC_FALCON_ICD_CMD(engineId), 0x1308);
    dprintf("lw: LWENC EV  :    0x%08x\n", GPU_REG_RD32(LW_PLWENC_FALCON_ICD_RDATA(engineId))); 
    GPU_REG_WR32(LW_PLWENC_FALCON_ICD_CMD(engineId), 0x1408);
    dprintf("lw: LWENC SP  :    0x%08x\n", GPU_REG_RD32(LW_PLWENC_FALCON_ICD_RDATA(engineId))); 
    GPU_REG_WR32(LW_PLWENC_FALCON_ICD_CMD(engineId), 0x1508);
    dprintf("lw: LWENC PC  :    0x%08x\n", GPU_REG_RD32(LW_PLWENC_FALCON_ICD_RDATA(engineId))); 
    GPU_REG_WR32(LW_PLWENC_FALCON_ICD_CMD(engineId), 0x1608);
    dprintf("lw: LWENC IMB :    0x%08x\n", GPU_REG_RD32(LW_PLWENC_FALCON_ICD_RDATA(engineId))); 
    GPU_REG_WR32(LW_PLWENC_FALCON_ICD_CMD(engineId), 0x1708);
    dprintf("lw: LWENC DMB :    0x%08x\n", GPU_REG_RD32(LW_PLWENC_FALCON_ICD_RDATA(engineId))); 
    GPU_REG_WR32(LW_PLWENC_FALCON_ICD_CMD(engineId), 0x1808);
    dprintf("lw: LWENC CSW :    0x%08x\n", GPU_REG_RD32(LW_PLWENC_FALCON_ICD_RDATA(engineId))); 
    dprintf("lw:\n\n");
    return LW_OK; 
}
