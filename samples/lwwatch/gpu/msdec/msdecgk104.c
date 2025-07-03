/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2011-2018 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#include "falcon.h"
#include "msdec.h"
#include "methodParse.h"
#include "print.h"
#include "virtOp.h"
#include "kepler/gk104/dev_falcon_v1.h"
#include "kepler/gk104/dev_msvld_pri.h"
#include "kepler/gk104/dev_mspdec_pri.h"
#include "kepler/gk104/dev_msppp_pri.h"
#include "kepler/gk104/dev_pri_ringstation_sys.h"

//-----------------------------------------------------
// msdecGetMsvldPriv_GK104
//-----------------------------------------------------
void msdecGetMsvldPriv_GK104(void *fmt)
{
    int i;
    msdecGetPriv(0, msdec_dbg_vld, 0);
    assert(pMsdecPrivRegs != NULL);
    for (i = 0; i < 233; i++)
    {
        if (pMsdecPrivRegs[i].m_id == 0)
        {
            break;
        }
        msdecPrintPriv(61, pMsdecPrivRegs[i].m_tag, pMsdecPrivRegs[i].m_id);
    }
}

//-----------------------------------------------------
// msdecGetMspppPriv_GK104
//-----------------------------------------------------
void msdecGetMspppPriv_GK104(void *fmt)
{
    int i;
    msdecGetPriv(0,msdec_dbg_ppp, 2);

    for(i = 0; i < LW_PMSPPP_FGT_INTENSITY_INTERVAL__SIZE_1;i++)
    {
        char t[1024];
        sprintf(t,"LW_PMSPPP_FGT_INTENSITY_INTERVAL(%d)",i);
        msdecPrintPriv(40,t,LW_PMSPPP_FGT_INTENSITY_INTERVAL(i));
    }
    for(i = 0; i < LW_PMSPPP_FGT_COMP_MODEL_VALUE__SIZE_1;i++)
    {
        char t[1024];
        sprintf(t,"LW_PMSPPP_FGT_COMP_MODEL_VALUE(%d)",i);
        msdecPrintPriv(40,t,LW_PMSPPP_FGT_COMP_MODEL_VALUE(i));
    }

    for(i = 0; i < LW_PMSPPP_HC_RESULT__SIZE_1;i++)
    {
        char t[1024];
        sprintf(t,"LW_PMSPPP_HC_RESULT(%d)",i);
        msdecPrintPriv(40,t,LW_PMSPPP_HC_RESULT(i));
    }
    dprintf("lw:\n");
    dprintf("lw: --------------------------- \n");
    dprintf("lw:\n");
}

//----------------------------------------------------------------------------------------------------
// this is the callback that control is transferred to, so that FCB data can be parsed and printed out
//----------------------------------------------------------------------------------------------------
static LwU32 ParseFCBInfo(LwU64 pa, LwU64 va,
                          pde_entry_t* pde, pte_entry_t* pte,
                          void* buffer, LwU32 length,
                          VCB_PARAM* pParam)
{
    LwU32   *fcbPtr;
    LwU32   index, value;
    fcbPtr = (LwU32*)buffer;
    dprintf("\n\nMSVLD");

    // VLD info in Flow Control Buffer
    value = *fcbPtr;
    value &= 0xf;
    dprintf("\nLwrrent Picture Index        = 0x%08x", value);
    index = value;
    value = *(++fcbPtr);
    fcbPtr += index + 1;
    dprintf("\nLwrrent Picture Error Code   = 0x%08x", *fcbPtr);
    switch(*fcbPtr)
    {
        case 0:         dprintf(" (NOT_STARTED)");       break;
        case 1:         dprintf(" (STARTED)");           break;
        case 2:         dprintf(" (COMPLETED)");         break;
        default:                                         break;
    }
    dprintf("\nLwrrent Write Pointer        = 0x%08x", value);
    fcbPtr = (LwU32*)buffer;
    fcbPtr += 18 + (index);
    dprintf("\nPrevious Picture End Offset  = 0x%08x", *fcbPtr);
    fcbPtr = (LwU32*)buffer;
    dprintf("\n\nMSPDEC");
    fcbPtr +=64;

    // PDEC info in Flow Control Buffer
    value = *fcbPtr;
    value &= 0xf;
    dprintf("\nLwrrent Picture Index        = 0x%08x", value);
    index = value;
    value = *(++fcbPtr);
    fcbPtr += index + 1;
    dprintf("\nLwrrent Picture Error Code   = 0x%08x", *fcbPtr);
    switch(*fcbPtr)
    {
        case 0:         dprintf(" (NOT_STARTED)");       break;
        case 1:         dprintf(" (STARTED)");           break;
        case 2:         dprintf(" (COMPLETED)");         break;
        default:                                         break;
    }
    dprintf("\nLwrrent Read Pointer         = 0x%08x", value);

    dprintf("\n\n\nFlow Control Buffer Dump\n");
    printBuffer(buffer, 0x200, (LwU32) va, 4);

    return LW_OK;
}

//-----------------------------------------------------------------------------
// function that will call the vmemDoVirtualOp function (return value must be non void)
//-----------------------------------------------------------------------------
LW_STATUS PrintFCB(LwU32 chId, LwU64 addrFCB, LwU32 length, BOOL temp, VCB_PARAM *VcbParam)
{
    VMemSpace vMemSpace;
    VMEM_INPUT_TYPE Id;
    memset(&Id, 0, sizeof(Id));
    Id.ch.chId = chId;

    if (vmemGet(&vMemSpace, VMEM_TYPE_CHANNEL, &Id) != LW_OK)
    {
        dprintf("lw: %s: Could not get a VMEM Space for ChId 0x%x.\n",
                __FUNCTION__, chId);
        return LW_ERR_GENERIC;
    }

    vmemDoVirtualOp(&vMemSpace, addrFCB, length, temp,(virtualCallback)ParseFCBInfo, VcbParam);
    return LW_OK;
}

//-----------------------------------------------------
// msdecGetInfo_GK104
//-----------------------------------------------------
void msdecGetInfo_GK104 (LwU32 iDec,LwU32 iDmemSize,LwU32 iImemSize, BOOL isParse, LwU32 iChId, BOOL isPrintFCB)
{
    LwU32 addrss;
    LwU32 address2;

    LwU32 addrssImem;
    LwU32 address2Imem;
    LwU32 address2Imemt;
    LwU32 bitmask;
    LwU32 regSysPrivFsConfig;

    //
    // Check first if all video engines have priv access.
    //
    // Bit-fields within LW_PPRIV_SYS_PRIV_FS_CONFIG denote this for video. All
    // video engines must have priv access for this command to proceed.
    //
    regSysPrivFsConfig = GPU_REG_RD32(LW_PPRIV_SYS_PRIV_FS_CONFIG(0));

    bitmask = BIT(LW_PPRIV_SYS_PRI_MASTER_fecs2mspdec_pri) |
              BIT(LW_PPRIV_SYS_PRI_MASTER_fecs2mspdec_pri) |
              BIT(LW_PPRIV_SYS_PRI_MASTER_fecs2msvld_pri);

    if ((regSysPrivFsConfig & bitmask) != bitmask)
    {
        dprintf("\n");
        dprintf("====================\n");
        dprintf("lw: Video engine priv access blocked. Cannot read registers.\n");
        dprintf("====================\n");
    }
    else
    {
        pMsdecPrivRegs = msdecPrivReg_v04_00;
        msdecPrintPriv(40,"lwBar0",0);

        // we need to look up MSPDEC methods for address of the FCB
        if(isPrintFCB)
        {
            iDec = 1;
        }
        switch(iDec)
        {
        case 0:
            {
                addrss=LW_PMSVLD_FALCON_DMEMD(0);
                address2=LW_PMSVLD_FALCON_DMEMC(0);
                addrssImem=LW_PMSVLD_FALCON_IMEMD(0);
                address2Imem=LW_PMSVLD_FALCON_IMEMC(0);
                address2Imemt = LW_PMSVLD_FALCON_IMEMT(0);

                if(iDmemSize==0 && iImemSize==0 && !isPrintFCB)
                    pMsdec[indexGpu].msdecGetMsvldPriv(0);
                break;
            }
        case 1:
            {
                addrss=LW_PMSPDEC_FALCON_DMEMD(0);
                address2=LW_PMSPDEC_FALCON_DMEMC(0);
                addrssImem=LW_PMSPDEC_FALCON_IMEMD(0);
                address2Imem=LW_PMSPDEC_FALCON_IMEMC(0);
                address2Imemt = LW_PMSPDEC_FALCON_IMEMT(0);

                if(iDmemSize==0 && iImemSize==0 && !isPrintFCB)
                    pMsdec[indexGpu].msdecGetMspdecPriv(0);
                break;
            }
        case 2:
            {
                addrss=LW_PMSPPP_FALCON_DMEMD(0);
                address2=LW_PMSPPP_FALCON_DMEMC(0);
                addrssImem=LW_PMSPPP_FALCON_IMEMD(0);
                address2Imem=LW_PMSPPP_FALCON_IMEMC(0);
                address2Imemt = LW_PMSPPP_FALCON_IMEMT(0);

                if(iDmemSize==0 && iImemSize==0 && !isPrintFCB)
                    pMsdec[indexGpu].msdecGetMspppPriv(0);
                break;
            }
        default:
            return;
        }

        if(iDmemSize>0 && !isPrintFCB)
        {
            LwU32 u;
            LwU32 dmemSizeMax = (GPU_REG_RD_DRF(_PMSPDEC_FALCON, _HWCFG, _DMEM_SIZE) << 8);

            iDmemSize = min(iDmemSize, dmemSizeMax);

            dprintf("lw: MS%s DMEM dump (first 0x%x bytes):", msdecEng(iDec), iDmemSize);
            dprintf("\nADDR: 03....00 07....04 0B....08 0F....0C 13....10 17....14 1B....18 1F....1C");
            dprintf("\n-----------------------------------------------------------------------------");

            for(u=0;u<(iDmemSize+3)/4;u++)
            {
                LwU32 i = (u<<(0?LW_PMSVLD_FALCON_DMEMC_OFFS));

                GPU_REG_WR32(address2,i);
                if((u%8==0))
                {
                    dprintf("\n%04X: ", 4*u);
                }
                dprintf("%08X ",  GPU_REG_RD32(addrss));
            }
        }

        if(iImemSize>0 && !isPrintFCB)
        {
            LwU32 u;
            LwU32 blk = 0;
            LwU32 imemSizeMax = (GPU_REG_RD_DRF(_PMSPDEC_FALCON, _HWCFG, _IMEM_SIZE) << 8);

            iImemSize = min(iImemSize, imemSizeMax);

            dprintf("\n-----------------------------------------------------------------------------\n\n");
            dprintf("lw: MS%s IMEM dump (first 0x%x bytes):", msdecEng(iDec), iImemSize);
            dprintf("\nADDR: 03....00 07....04 0B....08 0F....0C 13....10 17....14 1B....18 1F....1C");
            dprintf("\n-----------------------------------------------------------------------------");

            for(u=0;u<(iImemSize+3)/4;u++)
            {
                LwU32 i;
                if((u%64)==0) {
                    GPU_REG_WR32(address2Imemt, blk++);
                }
                i = (u<<(0?LW_PMSVLD_FALCON_IMEMC_OFFS));
                GPU_REG_WR32(address2Imem,i);
                if((u%8==0))
                {
                    dprintf("\n%04X: ", 4*u);
                }
                dprintf("%08X ",  GPU_REG_RD32(addrssImem));
            }
        }

        if ((iDmemSize>0 || iImemSize>0) && !isPrintFCB)
        {
            LwU32 u;
            LwU32 appId = 0;
            LwU32 comMthdOffs = 0;
            LwU32 appMthdOffs = 0;
            LwU32 comMthd[16] = {0};
            LwU32 appMthd[16] = {0};
            LwU32 classNum = 0;

            // get methods offset are in the DWORD#3
            u = (3<<(0?LW_PMSVLD_FALCON_DMEMC_OFFS));

            GPU_REG_WR32(address2,u);
            comMthdOffs = (GPU_REG_RD32(addrss)) >> 2;
            appMthdOffs = comMthdOffs + 16;

            // appId is stored 2 DWORDS before common methods offset
            u = ((comMthdOffs - 2)<<(0?LW_PMSVLD_FALCON_DMEMC_OFFS));

            GPU_REG_WR32(address2,u);
            appId = GPU_REG_RD32(addrss);

            // common/app methods
            for(u=0;u<16;u++)
            {
                LwU32 i = ((u+comMthdOffs)<<(0?LW_PMSVLD_FALCON_DMEMC_OFFS));
                GPU_REG_WR32(address2,i);
                comMthd[u] = GPU_REG_RD32(addrss);
                i = ((u+appMthdOffs)<<(0?LW_PMSVLD_FALCON_DMEMC_OFFS));
                GPU_REG_WR32(address2,i);
                appMthd[u] = GPU_REG_RD32(addrss);
            }

            // print common/app methods if appId is valid
            if (appId > 0)
            {
                LwU32 appMthdBase = 0;
                switch (iDec)
                {
                case 0:
                         if (appId==1) appMthdBase = 0x600;
                    else if (appId==2) appMthdBase = 0x500;
                    else if (appId==3) appMthdBase = 0x400;
                    else if (appId==4) appMthdBase = 0xE00;
                    else if (appId>=5 && appId<=9 ) appMthdBase = 0xC00;
                    break;
                case 1:
                    appMthdBase = 0x400;
                    break;
                case 2:
                         if (appId==2) appMthdBase = 0x500;
                    else if (appId==3) appMthdBase = 0x400;
                    break;
                case 3:
                    appMthdBase = 0x400;
                    break;
                default:
                    return;
                    break;
                }

                dprintf("\n\n-----------------------------------------------------------------------\n");
                dprintf("%4s, %8s,    %4s, %8s,    %4s, %8s,    %4s, %8s\n", "Mthd", "Data", "Mthd", "Data", "Mthd", "Data", "Mthd", "Data");
                for (u=0; u<16; u+=4)
                {
                    dprintf("%04X: %08X,    %04X: %08X,    %04X: %08X,    %04X: %08X\n",
                            0x700+4*u,     comMthd[u],   0x700+4*(u+1), comMthd[u+1],
                            0x700+4*(u+2), comMthd[u+2], 0x700+4*(u+3), comMthd[u+3]);
                }
                dprintf("\n");
                for (u=0; u<16; u+=4)
                {
                    dprintf("%04X: %08X,    %04X: %08X,    %04X: %08X,    %04X: %08X\n",
                            appMthdBase+4*u,     appMthd[u],   appMthdBase+4*(u+1), appMthd[u+1],
                            appMthdBase+4*(u+2), appMthd[u+2], appMthdBase+4*(u+3), appMthd[u+3]);
                }

                // print out the methods and data only if the "-p" option has been specified
                if (isParse)
                {
                    dprintf("\n\n\n\n");
                    for (u=0; u<16; u+=1)
                    {
                        BOOL bTemp = parseClassHeader(classNum, 0x700 +4*u, comMthd[u]);
                        if(!bTemp)
                            ;
                        else
                            dprintf("\n");
                    }
                    dprintf("\n\n\n\n");
                    for (u=0; u<16; u+=1)
                    {
                        BOOL bTemp = parseClassHeader(classNum, appMthdBase + 4*u, appMthd[u]);
                        if(!bTemp)
                            ;
                        else
                            dprintf("\n");
                    }
                }
            }
        }

        // flow control buffer is to be printed out
        if(isPrintFCB)
        {
            LwU32 u;
            LwU32 comMthdOffs = 0;
            LwU32 comMthd[16] = {0};
            LwU32 addrFCB = 0;
            LwU32 length = 0x200;
            VCB_PARAM VcbParam;

            // get methods offset are in the DWORD#3
            u = (3 << (0 ? LW_PMSPDEC_FALCON_IMEMC_OFFS));
            GPU_REG_WR32(address2,u);
            comMthdOffs = (GPU_REG_RD32(addrss)) >> 2;

            // common methods
            for(u=0;u<16;u++)
            {
                LwU32 i = ((u + comMthdOffs) << (0 ? LW_PMSPDEC_FALCON_IMEMC_OFFS));
                GPU_REG_WR32(address2,i);
                comMthd[u] = GPU_REG_RD32(addrss);
            }

            // at this point we have the FCB address
            addrFCB = comMthd[9];
            addrFCB <<= 8;
            dprintf("\nLW:: FCB address =   %4x", addrFCB);

            // now that we have the FCB address we need to print out the FCB data
            VcbParam.Id = VCB_ID_READVIRTUAL;
            VcbParam.memType = MT_GPUVIRTUAL;
            VcbParam.bStatus = FALSE;

            PrintFCB(iChId, addrFCB, length, FALSE, &VcbParam);
        }

        dprintf("\n");
    }

}

//-----------------------------------------------------
// printVldBitErrorCode_GK104
//-----------------------------------------------------
void printVldBitErrorCode_GK104(void)
{
    LwU32 beCode;

    beCode = GPU_REG_RD32(LW_PMSVLD_VLD_BIT_ERROR_CODE);
    beCode &= GPU_REG_RD32(LW_PMSVLD_VLD_BIT_ERROR_MASK);

    dprintf("lw: LW_PMSVLD_VLD_BIT_ERROR_CODE:      0x%08x\n", beCode);

    //handle cases of limit errors
    switch (DRF_VAL(_PMSVLD, _VLD_BIT_ERROR_CODE, _CODE, beCode))
    {
        case LW_PMSVLD_VLD_BIT_ERROR_CODE_CODE_EXCEED_SLICE_OFFSET_LIMIT:
            dprintf("lw: + LW_PMSVLD_VLD_BIT_ERROR_CODE_CODE_EXCEED_SLICE_OFFSET_LIMIT\n");
            dprintf("lw: LW_PMSVLD_VLD_SLICE_LIMIT_SLICE_BYTE_COUNT:   0x%08x\n",
                    GPU_REG_RD32(LW_PMSVLD_VLD_SLICE_LIMIT));
            break;

        case LW_PMSVLD_VLD_BIT_ERROR_CODE_CODE_MB_DATA_EXCEED_EDOB_FIFO_LIMIT:
            dprintf("lw: + LW_PMSVLD_VLD_BIT_ERROR_CODE_CODE_MB_DATA_EXCEED_EDOB_FIFO_LIMIT\n");
            dprintf("lw: LW_PMSVLD_VLD_EDOB_LIMIT:   0x%08x\n",
                    GPU_REG_RD32(LW_PMSVLD_VLD_EDOB_LIMIT));
            break;
    }
}


// regs to read PC and SP for msdec
static dbg_msdec msdec_pc_spReg[]=
{
    DBG_MSDEC_REG(LW_PMSVLD_FALCON_ICD_CMD),
    DBG_MSDEC_REG(LW_PMSVLD_FALCON_ICD_RDATA),
    DBG_MSDEC_REG(LW_PMSPDEC_FALCON_ICD_CMD),
    DBG_MSDEC_REG(LW_PMSPDEC_FALCON_ICD_RDATA),
    DBG_MSDEC_REG(LW_PMSPPP_FALCON_ICD_CMD),
    DBG_MSDEC_REG(LW_PMSPPP_FALCON_ICD_RDATA),
};

dbg_msdec msdecPrivReg_v04_00[] =
{
    DBG_MSDEC_REG(LW_PMSVLD_FALCON_IRQSSET),
    DBG_MSDEC_REG(LW_PMSVLD_FALCON_IRQSCLR),
    DBG_MSDEC_REG(LW_PMSVLD_FALCON_IRQSTAT),
    DBG_MSDEC_REG(LW_PMSVLD_FALCON_IRQMODE),
    DBG_MSDEC_REG(LW_PMSVLD_FALCON_IRQMSET),
    DBG_MSDEC_REG(LW_PMSVLD_FALCON_IRQMCLR),
    DBG_MSDEC_REG(LW_PMSVLD_FALCON_IRQMASK),
    DBG_MSDEC_REG(LW_PMSVLD_FALCON_IRQDEST),
    DBG_MSDEC_REG(LW_PMSVLD_FALCON_GPTMRINT),
    DBG_MSDEC_REG(LW_PMSVLD_FALCON_GPTMRVAL),
    DBG_MSDEC_REG(LW_PMSVLD_FALCON_GPTMRCTL),
    DBG_MSDEC_REG(LW_PMSVLD_FALCON_PTIMER0),
    DBG_MSDEC_REG(LW_PMSVLD_FALCON_PTIMER1),
    DBG_MSDEC_REG(LW_PMSVLD_FALCON_WDTMRVAL),
    DBG_MSDEC_REG(LW_PMSVLD_FALCON_WDTMRCTL),
    DBG_MSDEC_REG(LW_PMSVLD_FALCON_MTHDDATA),
    DBG_MSDEC_REG(LW_PMSVLD_FALCON_MTHDID),
    DBG_MSDEC_REG(LW_PMSVLD_FALCON_MTHDCOUNT),
    DBG_MSDEC_REG(LW_PMSVLD_FALCON_MTHDPOP),
    DBG_MSDEC_REG(LW_PMSVLD_FALCON_MTHDRAMSZ),
    DBG_MSDEC_REG(LW_PMSVLD_FALCON_LWRCTX),
    DBG_MSDEC_REG(LW_PMSVLD_FALCON_NXTCTX),
    DBG_MSDEC_REG(LW_PMSVLD_FALCON_CTXACK),
    DBG_MSDEC_REG(LW_PMSVLD_FALCON_MAILBOX0),
    DBG_MSDEC_REG(LW_PMSVLD_FALCON_MAILBOX1),
    DBG_MSDEC_REG(LW_PMSVLD_FALCON_ITFEN),
    DBG_MSDEC_REG(LW_PMSVLD_FALCON_IDLESTATE),
    DBG_MSDEC_REG(LW_PMSVLD_FALCON_FHSTATE),
    DBG_MSDEC_REG(LW_PMSVLD_FALCON_PRIVSTATE),
    DBG_MSDEC_REG(LW_PMSVLD_FALCON_SFTRESET),
    DBG_MSDEC_REG(LW_PMSVLD_FALCON_OS),
    DBG_MSDEC_REG(LW_PMSVLD_FALCON_RM),
    DBG_MSDEC_REG(LW_PMSVLD_FALCON_SOFT_PM),
    DBG_MSDEC_REG(LW_PMSVLD_FALCON_SOFT_MODE),
    DBG_MSDEC_REG(LW_PMSVLD_FALCON_DEBUG1),
    DBG_MSDEC_REG(LW_PMSVLD_FALCON_DEBUGINFO),
    DBG_MSDEC_REG(LW_PMSVLD_FALCON_IBRKPT1),
    DBG_MSDEC_REG(LW_PMSVLD_FALCON_IBRKPT2),
    DBG_MSDEC_REG(LW_PMSVLD_FALCON_CGCTL),
    DBG_MSDEC_REG(LW_PMSVLD_FALCON_ENGCTL),
    DBG_MSDEC_REG(LW_PMSVLD_FALCON_PMM),
    DBG_MSDEC_REG(LW_PMSVLD_FALCON_ADDR),
    DBG_MSDEC_REG(LW_PMSVLD_FALCON_CPUCTL),
    DBG_MSDEC_REG(LW_PMSVLD_FALCON_BOOTVEC),
    DBG_MSDEC_REG(LW_PMSVLD_FALCON_HWCFG),
    DBG_MSDEC_REG(LW_PMSVLD_FALCON_HWCFG1),
    DBG_MSDEC_REG(LW_PMSVLD_FALCON_DMACTL),
    DBG_MSDEC_REG(LW_PMSVLD_FALCON_DMATRFBASE),
    DBG_MSDEC_REG(LW_PMSVLD_FALCON_DMATRFMOFFS),
    DBG_MSDEC_REG(LW_PMSVLD_FALCON_DMATRFCMD),
    DBG_MSDEC_REG(LW_PMSVLD_FALCON_DMATRFFBOFFS),
    DBG_MSDEC_REG(LW_PMSVLD_FALCON_DMAPOLL_FB),
    DBG_MSDEC_REG(LW_PMSVLD_FALCON_DMAPOLL_CP),
    DBG_MSDEC_REG(LW_PMSVLD_FALCON_IMCTL),
    DBG_MSDEC_REG(LW_PMSVLD_FALCON_IMSTAT),
    DBG_MSDEC_REG(LW_PMSVLD_FALCON_TRACEIDX),
    DBG_MSDEC_REG(LW_PMSVLD_FALCON_TRACEPC),
    DBG_MSDEC_REG(LW_PMSVLD_FALCON_IMFILLRNG0),
    DBG_MSDEC_REG(LW_PMSVLD_FALCON_IMFILLRNG1),
    DBG_MSDEC_REG(LW_PMSVLD_FALCON_IMFILLCTL),
    DBG_MSDEC_REG(LW_PMSVLD_FALCON_ICD_CMD),
    DBG_MSDEC_REG(LW_PMSVLD_FALCON_ICD_ADDR),
    DBG_MSDEC_REG(LW_PMSVLD_FALCON_ICD_WDATA),
    DBG_MSDEC_REG(LW_PMSVLD_FALCON_ICD_RDATA),
    DBG_MSDEC_REG(LW_PMSVLD_FALCON_IMEMC(0)),
    DBG_MSDEC_REG(LW_PMSVLD_FALCON_IMEMC(1)),
    DBG_MSDEC_REG(LW_PMSVLD_FALCON_IMEMC(2)),
    DBG_MSDEC_REG(LW_PMSVLD_FALCON_IMEMC(3)),
    DBG_MSDEC_REG(LW_PMSVLD_FALCON_IMEMD(0)),
    DBG_MSDEC_REG(LW_PMSVLD_FALCON_IMEMD(1)),
    DBG_MSDEC_REG(LW_PMSVLD_FALCON_IMEMD(2)),
    DBG_MSDEC_REG(LW_PMSVLD_FALCON_IMEMD(3)),
    DBG_MSDEC_REG(LW_PMSVLD_FALCON_IMEMT(0)),
    DBG_MSDEC_REG(LW_PMSVLD_FALCON_IMEMT(1)),
    DBG_MSDEC_REG(LW_PMSVLD_FALCON_IMEMT(2)),
    DBG_MSDEC_REG(LW_PMSVLD_FALCON_IMEMT(3)),
    DBG_MSDEC_REG(LW_PMSVLD_FALCON_DMEMC(0)),
    DBG_MSDEC_REG(LW_PMSVLD_FALCON_DMEMC(1)),
    DBG_MSDEC_REG(LW_PMSVLD_FALCON_DMEMC(2)),
    DBG_MSDEC_REG(LW_PMSVLD_FALCON_DMEMC(3)),
    DBG_MSDEC_REG(LW_PMSVLD_FALCON_DMEMC(4)),
    DBG_MSDEC_REG(LW_PMSVLD_FALCON_DMEMC(5)),
    DBG_MSDEC_REG(LW_PMSVLD_FALCON_DMEMC(6)),
    DBG_MSDEC_REG(LW_PMSVLD_FALCON_DMEMC(7)),
    DBG_MSDEC_REG(LW_PMSVLD_FALCON_DMEMD(0)),
    DBG_MSDEC_REG(LW_PMSVLD_FALCON_DMEMD(1)),
    DBG_MSDEC_REG(LW_PMSVLD_FALCON_DMEMD(2)),
    DBG_MSDEC_REG(LW_PMSVLD_FALCON_DMEMD(3)),
    DBG_MSDEC_REG(LW_PMSVLD_FALCON_DMEMD(4)),
    DBG_MSDEC_REG(LW_PMSVLD_FALCON_DMEMD(5)),
    DBG_MSDEC_REG(LW_PMSVLD_FALCON_DMEMD(6)),
    DBG_MSDEC_REG(LW_PMSVLD_FALCON_DMEMD(7)),
    DBG_MSDEC_REG(LW_PMSVLD_VERSION),
    DBG_MSDEC_REG(LW_PMSVLD_CAP_REG0),
    DBG_MSDEC_REG(LW_PMSVLD_CAP_REG1),
    DBG_MSDEC_REG(LW_PMSVLD_CAP_REG2),
    DBG_MSDEC_REG(LW_PMSVLD_CAP_REG3),
    DBG_MSDEC_REG(LW_PMSVLD_PMM),
    DBG_MSDEC_REG(LW_PMSVLD_VLD_PIC_INFO_COMMON),
    DBG_MSDEC_REG(LW_PMSVLD_VLD_PIC_INFO_H264),
    DBG_MSDEC_REG(LW_PMSVLD_VLD_PIC_INFO_MPEG2),
    DBG_MSDEC_REG(LW_PMSVLD_VLD_PIC_INFO_VC1GB),
    DBG_MSDEC_REG(LW_PMSVLD_VLD_PIC_INFO_VC1T),
    DBG_MSDEC_REG(LW_PMSVLD_VLD_PIC_INFO_VC1Q),
    DBG_MSDEC_REG(LW_PMSVLD_VLD_PIC_INFO_VC1P),
    DBG_MSDEC_REG(LW_PMSVLD_VLD_SLICE_INFO),
    DBG_MSDEC_REG(LW_PMSVLD_VLD_EDOB_MB_INFO),
    DBG_MSDEC_REG(LW_PMSVLD_VLD_MB_INFO),
    DBG_MSDEC_REG(LW_PMSVLD_VLD_PARSE_CMD),
    DBG_MSDEC_REG(LW_PMSVLD_VLD_STATUS),
    DBG_MSDEC_REG(LW_PMSVLD_VLD_RESULT),
    DBG_MSDEC_REG(LW_PMSVLD_VLD_INTRPT_EN),
    DBG_MSDEC_REG(LW_PMSVLD_VLD_INTRPT_STATUS),
    DBG_MSDEC_REG(LW_PMSVLD_VLD_BIT_ERROR_CODE),
    DBG_MSDEC_REG(LW_PMSVLD_VLD_SCP_CFG),
    DBG_MSDEC_REG(LW_PMSVLD_VLD_TIMEOUT_VALUE),
    DBG_MSDEC_REG(LW_PMSVLD_VLD_RESET_CTL),
    DBG_MSDEC_REG(LW_PMSVLD_VLD_IDMA_BUFF_A(0)),
    DBG_MSDEC_REG(LW_PMSVLD_VLD_IDMA_BUFF_A(1)),
    DBG_MSDEC_REG(LW_PMSVLD_VLD_IDMA_BUFF_A(2)),
    DBG_MSDEC_REG(LW_PMSVLD_VLD_IDMA_BUFF_A(3)),
    DBG_MSDEC_REG(LW_PMSVLD_VLD_IDMA_BUFF_B(0)),
    DBG_MSDEC_REG(LW_PMSVLD_VLD_IDMA_BUFF_B(1)),
    DBG_MSDEC_REG(LW_PMSVLD_VLD_IDMA_BUFF_B(2)),
    DBG_MSDEC_REG(LW_PMSVLD_VLD_IDMA_BUFF_B(3)),
    DBG_MSDEC_REG(LW_PMSVLD_VLD_LWRR_IDMA_INFO),
    DBG_MSDEC_REG(LW_PMSVLD_VLD_CLEAR_INPUT_BUFFERS),
    DBG_MSDEC_REG(LW_PMSVLD_VLD_EDOB_START_ADDR),
    DBG_MSDEC_REG(LW_PMSVLD_VLD_EDOB_LIMIT),
    DBG_MSDEC_REG(LW_PMSVLD_VLD_EDOB_OFFSET),
    DBG_MSDEC_REG(LW_PMSVLD_VLD_EDOB_SIZE),
    DBG_MSDEC_REG(LW_PMSVLD_VLD_EDOB_CTL),
    DBG_MSDEC_REG(LW_PMSVLD_VLD_HIST_START_ADDR),
    DBG_MSDEC_REG(LW_PMSVLD_VLD_HIST_SIZE),
    DBG_MSDEC_REG(LW_PMSVLD_VLD_HIST_CTL),
    DBG_MSDEC_REG(LW_PMSVLD_VLD_ENTROPYDEC_CBC_DEBUG_OP),
    DBG_MSDEC_REG(LW_PMSVLD_VLD_ENTROPYDEC_DEBUG),
    DBG_MSDEC_REG(LW_PMSVLD_VLD_ENTROPYDEC_VC1_DEBUG0),
    DBG_MSDEC_REG(LW_PMSVLD_VLD_ENTROPYDEC_VC1_DEBUG1),
    DBG_MSDEC_REG(LW_PMSVLD_VLD_ENTROPYDEC_MPG_DEBUG),
    DBG_MSDEC_REG(LW_PMSVLD_VLD_BIT_ERROR_MASK),
    DBG_MSDEC_REG(LW_PMSVLD_VLD_SLICE_LIMIT),
    DBG_MSDEC_REG(LW_PMSVLD_VLD_SLICE_ERROR_CTL),
    DBG_MSDEC_REG(LW_PMSVLD_VLD_ENTROPYDEC_MPEG4_DEBUG),
    DBG_MSDEC_REG(LW_PMSVLD_CTL_STAT),
    DBG_MSDEC_REG(LW_PMSVLD_CTL_CFG),
    DBG_MSDEC_REG(LW_PMSVLD_CTL_SCP),
    DBG_MSDEC_REG(LW_PMSVLD_CTL_HDCP0),
    DBG_MSDEC_REG(LW_PMSVLD_CTL_HDCP1),
    DBG_MSDEC_REG(LW_PMSVLD_BAR0_CSR),
    DBG_MSDEC_REG(LW_PMSVLD_BAR0_ADDR),
    DBG_MSDEC_REG(LW_PMSVLD_BAR0_DATA),
    DBG_MSDEC_REG(LW_PMSVLD_BAR0_TMOUT),
    DBG_MSDEC_REG(LW_PMSVLD_FBIF_TRANSCFG(0)),
    DBG_MSDEC_REG(LW_PMSVLD_FBIF_TRANSCFG(1)),
    DBG_MSDEC_REG(LW_PMSVLD_FBIF_TRANSCFG(2)),
    DBG_MSDEC_REG(LW_PMSVLD_FBIF_TRANSCFG(3)),
    DBG_MSDEC_REG(LW_PMSVLD_FBIF_TRANSCFG(4)),
    DBG_MSDEC_REG(LW_PMSVLD_FBIF_TRANSCFG(5)),
    DBG_MSDEC_REG(LW_PMSVLD_FBIF_TRANSCFG(6)),
    DBG_MSDEC_REG(LW_PMSVLD_FBIF_TRANSCFG(7)),
    DBG_MSDEC_REG(LW_PMSVLD_FBIF_INSTBLK),
    DBG_MSDEC_REG(LW_PMSVLD_FBIF_CTL),
    DBG_MSDEC_REG(LW_PMSVLD_FBIF_DBG_STAT(0)),
    DBG_MSDEC_REG(LW_PMSVLD_FBIF_THROTTLE),
    DBG_MSDEC_REG(LW_PMSVLD_FBIF_ACHK_BLK(0)),
    DBG_MSDEC_REG(LW_PMSVLD_FBIF_ACHK_BLK(1)),
    DBG_MSDEC_REG(LW_PMSVLD_FBIF_ACHK_CTL(0)),
    DBG_MSDEC_REG(LW_PMSVLD_FBIF_ACHK_CTL(1)),
    DBG_MSDEC_REG(LW_PMSVLD_SCP_CTL0),
    DBG_MSDEC_REG(LW_PMSVLD_SCP_CTL1),
    DBG_MSDEC_REG(LW_PMSVLD_SCP_CTL_STAT),
    DBG_MSDEC_REG(LW_PMSVLD_SCP_CTL_CFG),
    DBG_MSDEC_REG(LW_PMSVLD_SCP_CFG0),
    DBG_MSDEC_REG(LW_PMSVLD_SCP_CTL_SCP),
    DBG_MSDEC_REG(LW_PMSVLD_SCP_CTL_PKEY),
    DBG_MSDEC_REG(LW_PMSVLD_SCP_CTL_DEBUG),
    DBG_MSDEC_REG(LW_PMSVLD_SCP_DEBUG0),
    DBG_MSDEC_REG(LW_PMSVLD_SCP_DEBUG1),
    DBG_MSDEC_REG(LW_PMSVLD_SCP_DEBUG2),
    DBG_MSDEC_REG(LW_PMSVLD_SCP_DEBUG_CMD),
    DBG_MSDEC_REG(LW_PMSVLD_SCP_ACL_FETCH),
    DBG_MSDEC_REG(LW_PMSVLD_SCP_STATUS),
    DBG_MSDEC_REG(LW_PMSVLD_SCP_STAT0),
    DBG_MSDEC_REG(LW_PMSVLD_SCP_STAT1),
    DBG_MSDEC_REG(LW_PMSVLD_SCP_RNG_STAT0),
    DBG_MSDEC_REG(LW_PMSVLD_SCP_RNG_STAT1),
    DBG_MSDEC_REG(LW_PMSVLD_SCP_INTR),
    DBG_MSDEC_REG(LW_PMSVLD_SCP_ACL_VIO),
    DBG_MSDEC_REG(LW_PMSVLD_SCP_SELWRITY_VIO),
    DBG_MSDEC_REG(LW_PMSVLD_SCP_CMD_ERROR),
    DBG_MSDEC_REG(LW_PMSVLD_SCP_RNDCTL0),
    DBG_MSDEC_REG(LW_PMSVLD_SCP_RNDCTL1),
    DBG_MSDEC_REG(LW_PMSVLD_SCP_RNDCTL2),
    DBG_MSDEC_REG(LW_PMSVLD_SCP_RNDCTL3),
    DBG_MSDEC_REG(LW_PMSVLD_SCP_RNDCTL4),
    DBG_MSDEC_REG(LW_PMSVLD_SCP_RNDCTL5),
    DBG_MSDEC_REG(LW_PMSVLD_SCP_RNDCTL6),
    DBG_MSDEC_REG(LW_PMSVLD_SCP_RNDCTL7),
    DBG_MSDEC_REG(LW_PMSVLD_SCP_RNDCTL8),
    DBG_MSDEC_REG(LW_PMSVLD_SCP_RNDCTL9),
    DBG_MSDEC_REG(LW_PMSVLD_SCP_RNDCTL10),
    DBG_MSDEC_REG(LW_PMSVLD_SCP_RNDCTL11),
    DBG_MSDEC_REG(0),
};

void msdecGetMspdecPriv_GK104(void *fmt)
{
    msdecGetPriv(0,msdec_dbg_pdec, 1);

    dprintf("lw:\n");
    dprintf("lw: --------------------------- \n");
    dprintf("lw:\n");
}

//-----------------------------------------------------------------------------
// Function to get PC and SP values
//-----------------------------------------------------------------------------
void msdecGetPcInfo_GK104(LwU32 iDec)
{
    // PC value
    GPU_REG_WR32(msdec_pc_spReg[(iDec<<1)].m_id, 0x1508);
    msdecPrintPriv(40, msdec_pc_spReg[(iDec<<1)].m_tag, msdec_pc_spReg[(iDec<<1)+1].m_id);

    // SP value
    GPU_REG_WR32(msdec_pc_spReg[(iDec<<1)].m_id, 0x1408);
    msdecPrintPriv(40, msdec_pc_spReg[(iDec<<1)+1].m_tag, msdec_pc_spReg[(iDec<<1)+1].m_id);
}

LW_STATUS msdecTestMsvldState_GK104( void )
{
    LW_STATUS    status = LW_OK;
    LwU32   regIntr;
    LwU32   regIntrEn;
    LwU32   vldIntr;
    LwU32   vldIntrEn;
    LwU32   data32;

    //check falcon interrupts
    regIntr = GPU_REG_RD32(LW_PMSVLD_FALCON_IRQSTAT);
    regIntrEn = GPU_REG_RD32(LW_PMSVLD_FALCON_IRQMASK);
    regIntr &= regIntrEn;

    if ( !DRF_VAL(_PMSVLD, _FALCON_IRQMASK, _GPTMR, regIntrEn))
        dprintf("lw: LW_PMSVLD_FALCON_IRQMASK_GPTMR disabled\n");

    if ( !DRF_VAL(_PMSVLD, _FALCON_IRQMASK, _MTHD, regIntrEn))
        dprintf("lw: LW_PMSVLD_FALCON_IRQMASK_MTHD disabled\n");

    if ( !DRF_VAL(_PMSVLD, _FALCON_IRQMASK, _CTXSW, regIntrEn))
        dprintf("lw: LW_PMSVLD_FALCON_IRQMASK_CTXSW disabled\n");

    if ( !DRF_VAL(_PMSVLD, _FALCON_IRQMASK, _HALT, regIntrEn))
        dprintf("lw: LW_PMSVLD_FALCON_IRQMASK_HALT disabled\n");

    if ( !DRF_VAL(_PMSVLD, _FALCON_IRQMASK, _SWGEN0, regIntrEn))
        dprintf("lw: LW_PMSVLD_FALCON_IRQMASK_SWGEN0 disabled\n");

    if ( !DRF_VAL(_PMSVLD, _FALCON_IRQMASK, _SWGEN1, regIntrEn))
        dprintf("lw: LW_PMSVLD_FALCON_IRQMASK_SWGEN1 disabled\n");

    if ( !(regIntrEn & 0x100))
        dprintf("lw: LW_PMSVLD_FALCON_IRQMASK: FBIF Ctx error interrupt disabled\n");

    if ( !(regIntrEn & 0x200))
        dprintf("lw: LW_PMSVLD_FALCON_IRQMASK: Limit violation interrupt disabled\n");

    if ( !(regIntrEn & 0x400))
        dprintf("lw: LW_PMSVLD_FALCON_IRQMASK: VLD interrupt disabled\n");


    //if any interrupt pending, set error
    if (regIntr != 0)
    {
        addUnitErr("\t MSVLD interrupts are pending\n");
        status = LW_ERR_GENERIC;
    }

    if ( DRF_VAL( _PMSVLD,_FALCON_IRQSTAT, _GPTMR, regIntr))
    {
        dprintf("lw: LW_PMSVLD_FALCON_IRQSTAT_GPTMR pending\n");

        dprintf("lw: LW_PMSVLD_FALCON_GPTMRINT:    0x%08x\n",
            GPU_REG_RD32(LW_PMSVLD_FALCON_GPTMRINT) );
        dprintf("lw: LW_PMSVLD_FALCON_GPTMRVAL:    0x%08x\n",
            GPU_REG_RD32(LW_PMSVLD_FALCON_GPTMRVAL) );

    }

    if ( DRF_VAL( _PMSVLD,_FALCON_IRQSTAT, _MTHD, regIntr))
    {
        dprintf("lw: LW_PMSVLD_FALCON_IRQSTAT_MTHD pending\n");

        dprintf("lw: LW_PMSVLD_FALCON_MTHDDATA_DATA:    0x%08x\n",
            GPU_REG_RD32(LW_PMSVLD_FALCON_MTHDDATA) );

        data32 = GPU_REG_RD32(LW_PMSVLD_FALCON_MTHDID);
        dprintf("lw: LW_PMSVLD_FALCON_MTHDID_ID:    0x%08x\n",
           DRF_VAL( _PMSVLD,_FALCON_MTHDID, _ID, data32)  );
        dprintf("lw: LW_PMSVLD_FALCON_MTHDID_SUBCH:    0x%08x\n",
           DRF_VAL( _PMSVLD,_FALCON_MTHDID, _SUBCH, data32)  );
        dprintf("lw: LW_PMSVLD_FALCON_MTHDID_PRIV:    0x%08x\n",
           DRF_VAL( _PMSVLD,_FALCON_MTHDID, _PRIV, data32)  );
    }

    if ( DRF_VAL( _PMSVLD,_FALCON_IRQSTAT, _CTXSW, regIntr))
    {
        dprintf("lw: LW_PMSVLD_FALCON_IRQSTAT_CTXSW pending\n");
    }

    if ( DRF_VAL( _PMSVLD,_FALCON_IRQSTAT, _HALT, regIntr))
    {
        dprintf("lw: LW_PMSVLD_FALCON_IRQSTAT_HALT pending\n");
    }

    if ( DRF_VAL( _PMSVLD,_FALCON_IRQSTAT, _SWGEN0, regIntr))
    {
        dprintf("lw: LW_PMSVLD_FALCON_IRQSTAT_SWGEN0 pending\n");
        pFalcon[indexGpu].falconPrintMailbox(LW_FALCON_MSVLD_BASE);
    }

    if ( DRF_VAL( _PMSVLD,_FALCON_IRQSTAT, _SWGEN1, regIntr))
    {
        dprintf("lw: LW_PMSVLD_FALCON_IRQSTAT_SWGEN1 pending\n");
    }

    //
    // Bit  |  Signal meaning
    // 8      FBIF Ctx error interrupt.
    // 9      Limit violation interrupt.
    // 10     VLD interrupt
    //
    if ( regIntr & 0x100)
    {
        dprintf("lw: LW_PMSVLD_FALCON_IRQSTAT: FBIF Ctx error interrupt\n");

        data32 = GPU_REG_RD32(LW_PMSVLD_FBIF_CTL);

        if (DRF_VAL(_PMSVLD, _FBIF_CTL, _ENABLE, data32))
        {
            if (DRF_VAL(_PMSVLD, _FBIF_CTL, _ILWAL_CONTEXT, data32))
            {
                dprintf("lw: + LW_PMSVLD_FBIF_CTL_ILWAL_CONTEXT\n");
            }
        }
    }

    if (regIntr & 0x200)
    {
        dprintf("lw: LW_PMSVLD_FALCON_IRQSTAT: Limit violation interrupt\n");
    }

    if (regIntr & 0x400)
    {
        dprintf("lw: LW_PMSVLD_FALCON_IRQSTAT: VLD interrupt\n");

        vldIntr = GPU_REG_RD32(LW_PMSVLD_VLD_INTRPT_STATUS);
        vldIntrEn = GPU_REG_RD32(LW_PMSVLD_VLD_INTRPT_EN);
        vldIntr &= vldIntrEn;

        //print disabled interrupts
        if ( !DRF_VAL(_PMSVLD, _VLD_INTRPT_EN, _WATCHDOG_TIMER_EN, vldIntrEn))
        dprintf("lw: + LW_PMSVLD_VLD_INTRPT_EN_WATCHDOG_TIMER_EN disabled\n");

        if ( !DRF_VAL(_PMSVLD, _VLD_INTRPT_EN, _BUFF_EMPTY_EN, vldIntrEn))
        dprintf("lw: + LW_PMSVLD_VLD_INTRPT_EN_WBUFF_EMPTY_EN disabled\n");

        if ( !DRF_VAL(_PMSVLD, _VLD_INTRPT_EN, _BIT_ERROR_EN, vldIntrEn))
        dprintf("lw: + LW_PMSVLD_VLD_INTRPT_EN_BIT_ERROR_EN disabled\n");

        if ( !DRF_VAL(_PMSVLD, _VLD_INTRPT_EN, _LIMIT_EN, vldIntrEn))
        dprintf("lw: + LW_PMSVLD_VLD_INTRPT_EN_LIMIT_EN disabled\n");

        if ( !DRF_VAL(_PMSVLD, _VLD_INTRPT_EN, _SLICE_DONE_EN, vldIntrEn))
        dprintf("lw: + LW_PMSVLD_VLD_INTRPT_EN_SLICE_DONE_EN disabled\n");

        //check vld interrupts
        if ( DRF_VAL( _PMSVLD,_VLD_INTRPT_STATUS, _WATCHDOG_TIMER, vldIntr))
        {
            dprintf("lw: + LW_PMSVLD_VLD_INTRPT_STATUS_WATCHDOG_TIMER pending\n");

            //printt timeout value on watchdog timer
            dprintf("lw: LW_PMSVLD_VLD_TIMEOUT_VALUE_MB_DECODE_TIMEOUT:    0x%08x\n",
                    GPU_REG_RD32(LW_PMSVLD_VLD_TIMEOUT_VALUE) );
        }

        if ( DRF_VAL( _PMSVLD,_VLD_INTRPT_STATUS, _BUFF_EMPTY, vldIntr))
        {
            dprintf("lw: + LW_PMSVLD_VLD_INTRPT_STATUS_BUFF_EMPTY pending\n");
        }

        if ( DRF_VAL( _PMSVLD,_VLD_INTRPT_STATUS, _BIT_ERROR, vldIntr))
        {
            dprintf("lw: + LW_PMSVLD_VLD_INTRPT_STATUS_BIT_ERROR pending\n");

            printVldBitErrorCode_GK104();
        }

        if ( DRF_VAL( _PMSVLD,_VLD_INTRPT_STATUS, _LIMIT, vldIntr))
        {
            dprintf("lw: + LW_PMSVLD_VLD_INTRPT_STATUS_LIMIT pending\n");
        }

        if ( DRF_VAL( _PMSVLD,_VLD_INTRPT_STATUS, _SLICE_DONE, vldIntr))
        {
            dprintf("lw: + LW_PMSVLD_VLD_INTRPT_STATUS_SLICE_DONE pending\n");
        }
    }

    data32 = GPU_REG_RD32(LW_PMSVLD_VLD_STATUS);

    dprintf("lw: LW_PMSVLD_VLD_STATUS:      0x%08x\n", data32);

    if ( DRF_VAL( _PMSVLD, _VLD_STATUS, _BUSY, data32))
    {
        dprintf("lw: + LW_PMSVLD_VLD_STATUS_BUSY\n");
        addUnitErr("\t LW_PMSVLD_VLD_STATUS_BUSY\n");
        status = LW_ERR_GENERIC;
    }

    if ( DRF_VAL( _PMSVLD, _VLD_STATUS, _SLICE_BUSY, data32))
    {
        dprintf("lw: + LW_PMSVLD_VLD_STATUS_SLICE_BUSY\n");
        addUnitErr("\t LW_PMSVLD_VLD_STATUS_SLICE_BUSY\n");
        status = LW_ERR_GENERIC;
    }

    if ( DRF_VAL( _PMSVLD, _VLD_STATUS, _EDOB_BUSY, data32))
    {
        dprintf("lw: + LW_PMSVLD_VLD_STATUS_EDOB_BUSY\n");
        addUnitErr("\t LW_PMSVLD_VLD_STATUS_EDOB_BUSY\n");
        status = LW_ERR_GENERIC;
    }

    if ( DRF_VAL( _PMSVLD, _VLD_STATUS, _VC1_PIC_BUSY, data32))
    {
        dprintf("lw: + LW_PMSVLD_VLD_STATUS_VC1_PIC_BUSY\n");
        addUnitErr("\t LW_PMSVLD_VLD_STATUS_VC1_PIC_BUSY\n");
        status = LW_ERR_GENERIC;
    }

    if ( DRF_VAL( _PMSVLD, _VLD_STATUS, _EDOB_NON_EMPTY, data32))
    {
        dprintf("lw: + LW_PMSVLD_VLD_STATUS_EDOB_NON_EMPTY\n");
        addUnitErr("\t LW_PMSVLD_VLD_STATUS_EDOB_NON_EMPTY\n");
        status = LW_ERR_GENERIC;
    }

    if ( DRF_VAL( _PMSVLD, _VLD_STATUS, _MEMACC_BUSY, data32)){

        dprintf("lw: + LW_PMSVLD_VLD_STATUS_MEMACC_BUSY\n");
        addUnitErr("\t LW_PMSVLD_VLD_STATUS_MEMACC_BUSY\n");
        status = LW_ERR_GENERIC;
    }

    if ( DRF_VAL( _PMSVLD, _VLD_STATUS, _FIFOCTRL_BUSY, data32))
    {
        dprintf("lw: + LW_PMSVLD_VLD_STATUS_FIFOCTRL_BUSY\n");
        addUnitErr("\t LW_PMSVLD_VLD_STATUS_FIFOCTRL_BUSY\n");
        status = LW_ERR_GENERIC;
    }

    if ( DRF_VAL( _PMSVLD, _VLD_STATUS, _HIST_BUSY, data32))
    {
        dprintf("lw: + LW_PMSVLD_VLD_STATUS_HIST_BUSY\n");
        addUnitErr("\t LW_PMSVLD_VLD_STATUS_HIST_BUSY\n");
        status = LW_ERR_GENERIC;
    }

    //print bar0 csr status
    data32 = GPU_REG_RD32(LW_PMSVLD_BAR0_CSR);

    dprintf("lw: LW_PMSVLD_BAR0_CSR:        0x%08x\n", data32);

     switch ( DRF_VAL( _PMSVLD, _BAR0_CSR,_STATUS, data32))
     {
        case LW_PMSVLD_BAR0_CSR_STATUS_IDLE:
            dprintf("lw: + LW_PMSVLD_BAR0_CSR_STATUS_IDLE\n");
        break;
        case LW_PMSVLD_BAR0_CSR_STATUS_BUSY:
            dprintf("lw: + LW_PMSVLD_BAR0_CSR_STATUS_BUSY\n");
        break;
        case LW_PMSVLD_BAR0_CSR_STATUS_TMOUT:
            dprintf("lw: + LW_PMSVLD_BAR0_CSR_STATUS_TMOUT\n");
        break;
        case LW_PMSVLD_BAR0_CSR_STATUS_DIS:
            dprintf("lw: + LW_PMSVLD_BAR0_CSR_STATUS_DIS\n");
        break;
     }

    //
    //print falcon states
    //     Bit |  Signal meaning
    //     0      FALCON busy
    //     1      FBIF busy
    //     2      VLD busy
    //     3-15   tied to zero.
    //
    data32 = GPU_REG_RD32(LW_PMSVLD_FALCON_IDLESTATE);

    if ( DRF_VAL( _PMSVLD, _FALCON_IDLESTATE, _FALCON_BUSY, data32))
    {
        dprintf("lw: + LW_PMSVLD_FALCON_IDLESTATE_FALCON_BUSY\n");
        addUnitErr("\t LW_PMSVLD_FALCON_IDLESTATE_FALCON_BUSY\n");
        status = LW_ERR_GENERIC;
    }

    if ( data32 & DRF_SHIFTMASK(1:1))
    {
        dprintf("lw: + LW_PMSVLD_FALCON_IDLESTATE_FBIF_BUSY\n");
        addUnitErr("\t LW_PMSVLD_FALCON_IDLESTATE_FBIF_BUSY\n");
        status = LW_ERR_GENERIC;
    }

    if ( data32 & DRF_SHIFTMASK(2:2))
    {
        dprintf("lw: + LW_PMSVLD_FALCON_IDLESTATE_VLD_BUSY\n");
        addUnitErr("\t LW_PMSVLD_FALCON_IDLESTATE_VLD_BUSY\n");

        status = LW_ERR_GENERIC;
    }


    data32 = GPU_REG_RD32(LW_PMSVLD_FALCON_FHSTATE);

    if ( DRF_VAL( _PMSVLD, _FALCON_FHSTATE, _FALCON_HALTED, data32))
    {
        dprintf("lw: + LW_PMSVLD_FALCON_FHSTATE_FALCON_HALTED\n");
        addUnitErr("\t LW_PMSVLD_FALCON_FHSTATE_FALCON_HALTED\n");

        status = LW_ERR_GENERIC;
    }

    if ( data32 & DRF_SHIFTMASK(1:1))
    {
        dprintf("lw: + LW_PMSVLD_FALCON_FHSTATE_FBIF_HALTED\n");
        addUnitErr("\t LW_PMSVLD_FALCON_FHSTATE_FBIF_HALTED\n");
        status = LW_ERR_GENERIC;
    }

    if ( data32 & DRF_SHIFTMASK(2:2))
    {
        dprintf("lw: + LW_PMSVLD_FALCON_FHSTATE_VLD_HALTED\n");
        addUnitErr("\t LW_PMSVLD_FALCON_FHSTATE_VLD_HALTED\n");
        status = LW_ERR_GENERIC;
    }

    if ( DRF_VAL( _PMSVLD, _FALCON_FHSTATE, _ENGINE_FAULTED, data32))
    {
        dprintf("lw: + LW_PMSVLD_FALCON_FHSTATE_ENGINE_FAULTED\n");
        addUnitErr("\t LW_PMSVLD_FALCON_FHSTATE_ENGINE_FAULTED\n");
        status = LW_ERR_GENERIC;
    }

    if ( DRF_VAL( _PMSVLD, _FALCON_FHSTATE, _STALL_REQ, data32))
    {
        dprintf("lw: + LW_PMSVLD_FALCON_FHSTATE_STALL_REQ\n");
        addUnitErr("\t LW_PMSVLD_FALCON_FHSTATE_STALL_REQ\n");
        status = LW_ERR_GENERIC;
    }

    //print falcon ctl regs
    data32 = GPU_REG_RD32(LW_PMSVLD_FALCON_ENGCTL);

    if ( DRF_VAL( _PMSVLD, _FALCON_ENGCTL, _ILW_CONTEXT, data32))
    {
        dprintf("lw: + LW_PMSVLD_FALCON_ENGCTL_ILW_CONTEXT\n");
        addUnitErr("\t LW_PMSVLD_FALCON_ENGCTL_ILW_CONTEXT\n");
        status = LW_ERR_GENERIC;
    }

    if ( DRF_VAL( _PMSVLD, _FALCON_ENGCTL, _STALLREQ, data32))
    {
        dprintf("lw: + LW_PMSVLD_FALCON_ENGCTL_STALLREQ\n");
        addUnitErr("\t LW_PMSVLD_FALCON_ENGCTL_STALLREQ\n");
        status = LW_ERR_GENERIC;
    }

    data32 = GPU_REG_RD32(LW_PMSVLD_FALCON_CPUCTL);

    if ( DRF_VAL( _PMSVLD, _FALCON_CPUCTL, _IILWAL, data32))
    {
        dprintf("lw: + LW_PMSVLD_FALCON_CPUCTL_IILWAL\n");
        addUnitErr("\t LW_PMSVLD_FALCON_CPUCTL_IILWAL\n");
        status = LW_ERR_GENERIC;
    }

    if ( DRF_VAL( _PMSVLD, _FALCON_CPUCTL, _HALTED, data32))
    {
        dprintf("lw: + LW_PMSVLD_FALCON_CPUCTL_HALTED\n");
        addUnitErr("\t LW_PMSVLD_FALCON_CPUCTL_HALTED\n");
        status = LW_ERR_GENERIC;
    }

    if ( DRF_VAL( _PMSVLD, _FALCON_CPUCTL, _STOPPED, data32))
    {
        dprintf("lw: + LW_PMSVLD_FALCON_CPUCTL_STOPPED\n");
        addUnitErr("\t Warning: LW_PMSVLD_FALCON_CPUCTL_STOPPED\n");
        //status = LW_ERR_GENERIC;
    }

    // state of mthd/ctx interface
    data32 = GPU_REG_RD32(LW_PMSVLD_FALCON_ITFEN);

    if (DRF_VAL( _PMSVLD, _FALCON_ITFEN, _CTXEN, data32))
    {
        dprintf("lw: + LW_PMSVLD_FALCON_ITFEN_CTXEN enabled\n");

        if (pFalcon[indexGpu].falconTestCtxState(LW_FALCON_MSVLD_BASE, "PMSVLD") == LW_ERR_GENERIC)
        {
            dprintf("lw: Current ctx state invalid\n");
            addUnitErr("\t Current ctx state is invalid\n");
            status = LW_ERR_GENERIC;
        }
        else
        {
            dprintf("lw: Current ctx state valid\n");
        }
    }
    else
    {
        dprintf("lw: + LW_PMSVLD_FALCON_ITFEN_CTXEN disabled\n");
    }

    if ( DRF_VAL( _PMSVLD, _FALCON_ITFEN, _MTHDEN, data32))
    {
        dprintf("lw: + LW_PMSVLD_FALCON_ITFEN_MTHDEN enabled\n");
    }
    else
    {
        dprintf("lw: + LW_PMSVLD_FALCON_ITFEN_MTHDEN disabled\n");
    }

    //check if falcon is hung (instr ptr)
    if ( pFalcon[indexGpu].falconTestPC(LW_FALCON_MSVLD_BASE, "PMSVLD") == LW_ERR_GENERIC )
    {
        dprintf("lw: Falcon instruction pointer is stuck or invalid\n");

        //TODO: treat falcon PC errors as warnings now, need to report as error
        addUnitErr("\t Warning: Falcon instruction pointer is stuck or invalid\n");
        //status = LW_ERR_GENERIC;
    }

    return status;
}

LW_STATUS msdecTestMspdecState_GK104( void )
{

    LW_STATUS    status = LW_OK;
    LwU32   regIntr;
    LwU32   regIntrEn;
    LwU32   engIntr;
    LwU32   engIntrEn;
    LwU32   data32;

    //check falcon interrupts
    regIntr = GPU_REG_RD32(LW_PMSPDEC_FALCON_IRQSTAT);
    regIntrEn = GPU_REG_RD32(LW_PMSPDEC_FALCON_IRQMASK);
    regIntr &= regIntrEn;

    if ( !DRF_VAL(_PMSPDEC, _FALCON_IRQMASK, _GPTMR, regIntrEn))
        dprintf("lw: LW_PMSPDEC_FALCON_IRQMASK_GPTMR disabled\n");

    if ( !DRF_VAL(_PMSPDEC, _FALCON_IRQMASK, _MTHD, regIntrEn))
        dprintf("lw: LW_PMSPDEC_FALCON_IRQMASK_MTHD disabled\n");

    if ( !DRF_VAL(_PMSPDEC, _FALCON_IRQMASK, _CTXSW, regIntrEn))
        dprintf("lw: LW_PMSPDEC_FALCON_IRQMASK_CTXSW disabled\n");

    if ( !DRF_VAL(_PMSPDEC, _FALCON_IRQMASK, _HALT, regIntrEn))
        dprintf("lw: LW_PMSPDEC_FALCON_IRQMASK_HALT disabled\n");

    if ( !DRF_VAL(_PMSPDEC, _FALCON_IRQMASK, _SWGEN0, regIntrEn))
        dprintf("lw: LW_PMSPDEC_FALCON_IRQMASK_SWGEN0 disabled\n");

    if ( !DRF_VAL(_PMSPDEC, _FALCON_IRQMASK, _SWGEN1, regIntrEn))
        dprintf("lw: LW_PMSPDEC_FALCON_IRQMASK_SWGEN1 disabled\n");

    if ( !(regIntrEn & 0x100))
        dprintf("lw: LW_PMSPDEC_FALCON_IRQMASK: FBIF Ctx error interrupt disabled\n");

    if ( !(regIntrEn & 0x200))
        dprintf("lw: LW_PMSPDEC_FALCON_IRQMASK: Limit violation interrupt disabled\n");

    if ( !(regIntrEn & 0x400))
        dprintf("lw: LW_PMSPDEC_FALCON_IRQMASK: MV interrupt disabled\n");

    if ( !(regIntrEn & 0x800))
        dprintf("lw: LW_PMSPDEC_FALCON_IRQMASK: IQT interrupt disabled\n");

    if ( !(regIntrEn & 0x1000))
        dprintf("lw: LW_PMSPDEC_FALCON_IRQMASK: MC interrupt disabled\n");

    if ( !(regIntrEn & 0x2000))
        dprintf("lw: LW_PMSPDEC_FALCON_IRQMASK: REC interrupt disabled\n");

    if ( !(regIntrEn & 0x4000))
        dprintf("lw: LW_PMSPDEC_FALCON_IRQMASK: DBMF interrupt disabled\n");

    if ( !(regIntrEn & 0x8000))
        dprintf("lw: LW_PMSPDEC_FALCON_IRQMASK: DBFDMA interrupt disabled\n");


    //if any interrupt pending, set error
    if (regIntr != 0)
    {
        addUnitErr("\t MSPDEC interrupts are pending.\n");
        status = LW_ERR_GENERIC;
    }

    if ( DRF_VAL( _PMSPDEC,_FALCON_IRQSTAT, _GPTMR, regIntr))
    {
        dprintf("lw: LW_PMSPDEC_FALCON_IRQSTAT_GPTMR pending\n");

        dprintf("lw: LW_PMSPDEC_FALCON_GPTMRINT:    0x%08x\n",
            GPU_REG_RD32(LW_PMSPDEC_FALCON_GPTMRINT) );
        dprintf("lw: LW_PMSPDEC_FALCON_GPTMRVAL:    0x%08x\n",
            GPU_REG_RD32(LW_PMSPDEC_FALCON_GPTMRVAL) );

    }

    if ( DRF_VAL( _PMSPDEC,_FALCON_IRQSTAT, _MTHD, regIntr))
    {
        dprintf("lw: LW_PMSPDEC_FALCON_IRQSTAT_MTHD pending\n");

        dprintf("lw: LW_PMSPDEC_FALCON_MTHDDATA_DATA:    0x%08x\n",
            GPU_REG_RD32(LW_PMSPDEC_FALCON_MTHDDATA) );

        data32 = GPU_REG_RD32(LW_PMSPDEC_FALCON_MTHDID);
        dprintf("lw: LW_PMSPDEC_FALCON_MTHDID_ID:    0x%08x\n",
           DRF_VAL( _PMSPDEC,_FALCON_MTHDID, _ID, data32)  );
        dprintf("lw: LW_PMSPDEC_FALCON_MTHDID_SUBCH:    0x%08x\n",
           DRF_VAL( _PMSPDEC,_FALCON_MTHDID, _SUBCH, data32)  );
        dprintf("lw: LW_PMSPDEC_FALCON_MTHDID_PRIV:    0x%08x\n",
           DRF_VAL( _PMSPDEC,_FALCON_MTHDID, _PRIV, data32)  );
    }

    if ( DRF_VAL( _PMSPDEC,_FALCON_IRQSTAT, _CTXSW, regIntr))
    {
        dprintf("lw: LW_PMSPDEC_FALCON_IRQSTAT_CTXSW pending\n");
    }

    if ( DRF_VAL( _PMSPDEC,_FALCON_IRQSTAT, _HALT, regIntr))
    {
        dprintf("lw: LW_PMSPDEC_FALCON_IRQSTAT_HALT pending\n");
    }

    if ( DRF_VAL( _PMSPDEC,_FALCON_IRQSTAT, _SWGEN0, regIntr))
    {
        dprintf("lw: LW_PMSPDEC_FALCON_IRQSTAT_SWGEN0 pending\n");

        pFalcon[indexGpu].falconPrintMailbox(LW_FALCON_MSPDEC_BASE);
    }

    if ( DRF_VAL( _PMSPDEC,_FALCON_IRQSTAT, _SWGEN1, regIntr))
    {
        dprintf("lw: LW_PMSPDEC_FALCON_IRQSTAT_SWGEN1 pending\n");
    }

    /*
    Bit |  Signal meaning
    8      FBIF Ctx error interrupt.
    9      Limit violation interrupt.
    10     MV interrupt. See LW_PMSPDEC_MVINTCSW for more details.
    11     IQT interrupt. See LW_PMSPDEC_IQTINTRPTSTAT for more details.
    12     MC interrupt. See LW_PMSPDEC_MCINTRPTSTAT for more details.
    13     REC interrupt. See LW_PMSPDEC_RECINTCSW for more details.
    14     DBF interrupt. See LW_PMSPDEC_DBFINTRPTSTAT for more details.
    15     DBFDMA interrupt. See LW_PMSPDEC_DBFDMA_INTRPTSTAT for more details.
    */

    if ( regIntr & 0x100)
    {
        dprintf("lw: LW_PMSPDEC_FALCON_IRQSTAT: FBIF Ctx error interrupt\n");

        data32 = GPU_REG_RD32(LW_PMSPDEC_FBIF_CTL);

        if (DRF_VAL(_PMSPDEC, _FBIF_CTL, _ENABLE, data32))
        {
            if (DRF_VAL(_PMSPDEC, _FBIF_CTL, _ILWAL_CONTEXT, data32))
            {
                dprintf("lw: + LW_PMSPDEC_FBIF_CTL_ILWAL_CONTEXT\n");
            }
        }
    }

    if (regIntr & 0x200)
    {
        dprintf("lw: LW_PMSPDEC_FALCON_IRQSTAT: Limit violation interrupt\n");
    }

    if (regIntr & 0x400)
    {
        dprintf("lw: LW_PMSPDEC_FALCON_IRQSTAT: MV interrupt\n");

        engIntr = GPU_REG_RD32(LW_PMSPDEC_MVINTCSW);
        engIntrEn = GPU_REG_RD32(LW_PMSPDEC_MVINTEN);
        engIntr &= engIntrEn;

        //print disabled interrupts
        if ( !DRF_VAL(_PMSPDEC, _MVINTEN,_IRQ_WDTMR, engIntrEn))
        dprintf("lw: + LW_PMSPDEC_MVINTEN_IRQ_WDTMR disabled\n");

        if ( !DRF_VAL(_PMSPDEC, _MVINTEN, _IRQ_ISDMA, engIntrEn))
        dprintf("lw: + LW_PMSPDEC_MVINTEN_IRQ_ISDMA disabled\n");

        if ( !DRF_VAL(_PMSPDEC, _MVINTEN, _IRQ_EOSP, engIntrEn))
        dprintf("lw: + LW_PMSPDEC_MVINTEN_IRQ_EOSP disabled\n");

        if ( !DRF_VAL(_PMSPDEC, _MVINTEN, _IRQ_TERR, engIntrEn))
        dprintf("lw: + LW_PMSPDEC_MVINTEN_IRQ_TERR disabled\n");

        if ( !DRF_VAL(_PMSPDEC, _MVINTEN, _IRQ_RESERR, engIntrEn))
        dprintf("lw: + LW_PMSPDEC_MVINTEN_IRQ_RESERR disabled\n");

        if ( !DRF_VAL(_PMSPDEC, _MVINTEN, _IRQ_BERR, engIntrEn))
        dprintf("lw: + LW_PMSPDEC_MVINTEN_IRQ_BERR disabled\n");


        //check mv interrupts
        if ( DRF_VAL( _PMSPDEC,_MVINTCSW, _IRQ_WDTMR, engIntr))
        {
            dprintf("lw: + LW_PMSPDEC_MVINTCSW_IRQ_WDTMR pending\n");

            dprintf("lw:    LW_PMSPDEC_MVWDTIME_WDTIME:     0x%08x\n",
                GPU_REG_RD_DRF(_PMSPDEC, _MVWDTIME, _WDTIME));
        }

        if ( DRF_VAL( _PMSPDEC,_MVINTCSW, _IRQ_ISDMA, engIntr))
        {
            dprintf("lw: + LW_PMSPDEC_MVINTCSW_IRQ_ISDMA pending\n");
        }

        if ( DRF_VAL( _PMSPDEC,_MVINTCSW, _IRQ_EOSP, engIntr))
        {
            dprintf("lw: + LW_PMSPDEC_MVINTCSW_IRQ_EOSP pending\n");
        }

        if ( DRF_VAL( _PMSPDEC,_MVINTCSW, _IRQ_TERR, engIntr))
        {
            dprintf("lw: + LW_PMSPDEC_MVINTCSW_IRQ_TERR pending\n");
        }

        if ( DRF_VAL( _PMSPDEC,_MVINTCSW, _IRQ_RESERR, engIntr))
        {
            dprintf("lw: + LW_PMSPDEC_MVINTCSW_IRQ_RESERR pending\n");
        }

        if ( DRF_VAL( _PMSPDEC,_MVINTCSW, _IRQ_BERR, engIntr))
        {
            dprintf("lw: + LW_PMSPDEC_MVINTCSW_IRQ_BERR pending\n");
        }
    }

    //iqt interrupt
    if (regIntr & 0x800)
    {
        dprintf("lw: LW_PMSPDEC_FALCON_IRQSTAT: IQT interrupt\n");

        engIntr = GPU_REG_RD32(LW_PMSPDEC_IQTINTRPTSTAT);
        engIntrEn = GPU_REG_RD32(LW_PMSPDEC_IQTINTRPTEN);
        engIntr &= engIntrEn;

        //print disabled interrupts
        if ( !DRF_VAL( _PMSPDEC,_IQTINTRPTEN, _IQCOEFERR, engIntr))
            dprintf("lw: + LW_PMSPDEC_IQTINTRPTEN_IQCOEFERR_OFF\n");

        if ( !DRF_VAL( _PMSPDEC,_IQTINTRPTEN, _IQIDXERR, engIntr))
            dprintf("lw: + LW_PMSPDEC_IQTINTRPTEN_IQIDXERR_OFF\n");

        if ( !DRF_VAL( _PMSPDEC,_IQTINTRPTEN, _CFGWRERR, engIntr))
            dprintf("lw: + LW_PMSPDEC_IQTINTRPTEN_CFGWRERR_OFF\n");

        //print pending interrupts
        if ( DRF_VAL( _PMSPDEC,_IQTINTRPTSTAT, _IQCOEFERR, engIntr))
        {
            dprintf("lw: + LW_PMSPDEC_IQTINTRPTSTAT_IQCOEFERR_TRUE\n");
        }

        if ( DRF_VAL( _PMSPDEC,_IQTINTRPTSTAT, _IQIDXERR, engIntr))
        {
            dprintf("lw: + LW_PMSPDEC_IQTINTRPTSTAT_IQIDXERR_TRUE\n");
        }

        if ( DRF_VAL( _PMSPDEC,_IQTINTRPTSTAT, _CFGWRERR, engIntr))
        {
            dprintf("lw: + LW_PMSPDEC_IQTINTRPTSTAT_CFGWRERR_TRUE\n");
        }

    }

    //mc interrupt
    if (regIntr & 0x1000)
    {
        dprintf("lw: LW_PMSPDEC_FALCON_IRQSTAT: MC interrupt\n");

        engIntr = GPU_REG_RD32(LW_PMSPDEC_MCINTRPTSTAT);
        engIntrEn = GPU_REG_RD32(LW_PMSPDEC_MCINTRPTEN);
        engIntr &= engIntrEn;

        //print disabled interrupts
        if ( !DRF_VAL( _PMSPDEC,_MCINTRPTEN, _CONFIGERR, engIntr))
            dprintf("lw: + LW_PMSPDEC_MCINTRPTEN_CONFIGERR_OFF\n");

        //print pending interrupts
        if ( DRF_VAL( _PMSPDEC,_MCINTRPTSTAT, _CONFIGERR, engIntr))
        {
            dprintf("lw: + LW_PMSPDEC_MCINTRPTSTAT_CONFIGERR_TRUE\n");
        }
    }

    //rec interrupt
    if (regIntr & 0x2000)
    {
        dprintf("lw: LW_PMSPDEC_FALCON_IRQSTAT: REC interrupt\n");

        engIntr = GPU_REG_RD32(LW_PMSPDEC_RECINTCSW);
        engIntrEn = GPU_REG_RD32(LW_PMSPDEC_RECINTEN);
        engIntr &= engIntrEn;

        //print disabled interrupts
        if ( !DRF_VAL( _PMSPDEC,_RECINTEN, _IRQ_CFGWR, engIntr))
            dprintf("lw: + LW_PMSPDEC_RECINTEN_IRQ_CFGWR_OFF\n");

        //print pending interrupts
        if ( DRF_VAL( _PMSPDEC,_RECINTCSW, _IRQ_CFGWR, engIntr))
        {
            dprintf("lw: + LW_PMSPDEC_RECINTCSW_IRQ_CFGWR_TRUE\n");
        }
    }

    //dbf interrupt
    if (regIntr & 0x4000)
    {
        dprintf("lw: LW_PMSPDEC_FALCON_IRQSTAT: DBF interrupt\n");

        engIntr = GPU_REG_RD32(LW_PMSPDEC_DBFINTRPTSTAT);
        engIntrEn = GPU_REG_RD32(LW_PMSPDEC_DBFINTRPTEN);
        engIntr &= engIntrEn;

        //print disabled interrupts
        if ( !DRF_VAL( _PMSPDEC,_DBFINTRPTEN, _CFGERR, engIntr))
            dprintf("lw: + LW_PMSPDEC_DBFINTRPTEN_CFGERR_OFF\n");

        if ( !DRF_VAL( _PMSPDEC,_DBFINTRPTEN, _DMAERR, engIntr))
            dprintf("lw: + LW_PMSPDEC_DBFINTRPTEN_DMAERR_OFF\n");

        if ( !DRF_VAL( _PMSPDEC,_DBFINTRPTEN, _MVERR, engIntr))
            dprintf("lw: + LW_PMSPDEC_DBFINTRPTEN_MVERR_OFF\n");


        //print pending interrupts
        if ( DRF_VAL( _PMSPDEC,_DBFINTRPTSTAT, _CFGERR, engIntr))
        {
            dprintf("lw: + LW_PMSPDEC_DBFINTRPTSTAT_CFGERR_TRUE\n");
        }

        if ( DRF_VAL( _PMSPDEC,_DBFINTRPTSTAT, _DMAERR, engIntr))
        {
            dprintf("lw: + LW_PMSPDEC_DBFINTRPTSTAT_DMAERR_TRUE\n");
        }

        if ( DRF_VAL( _PMSPDEC,_DBFINTRPTSTAT, _MVERR, engIntr))
        {
            dprintf("lw: + LW_PMSPDEC_DBFINTRPTSTAT_MVERR_TRUE\n");
        }
    }

    //dbfdma interrupt
    if (regIntr & 0x8000)
    {
        dprintf("lw: LW_PMSPDEC_FALCON_IRQSTAT: DBFDMA interrupt\n");

        engIntr = GPU_REG_RD32(LW_PMSPDEC_DBFDMA_INTRPTSTAT);
        engIntrEn = GPU_REG_RD32(LW_PMSPDEC_DBFDMA_INTRPTEN);
        engIntr &= engIntrEn;

        //print disabled interrupts
        if ( !DRF_VAL( _PMSPDEC,_DBFDMA_INTRPTEN, _OFFSETERR, engIntr))
            dprintf("lw: + LW_PMSPDEC_DBFDMA_INTRPTEN_OFFSETERR_OFF\n");

        if ( !DRF_VAL( _PMSPDEC,_DBFDMA_INTRPTEN, _CFGWRERR, engIntr))
            dprintf("lw: + LW_PMSPDEC_DBFDMA_INTRPTEN_CFGWRERR_OFF\n");

        if ( !DRF_VAL( _PMSPDEC,_DBFDMA_INTRPTEN, _ROWEND, engIntr))
            dprintf("lw: + LW_PMSPDEC_DBFDMA_INTRPTEN_ROWEND_OFF\n");


        //print pending interrupts
        if ( DRF_VAL( _PMSPDEC,_DBFDMA_INTRPTSTAT, _OFFSETERR, engIntr))
        {
            dprintf("lw: + LW_PMSPDEC_DBFDMA_INTRPTSTAT_OFFSETERR_TRUE\n");
        }

        if ( DRF_VAL( _PMSPDEC,_DBFDMA_INTRPTSTAT,_CFGWRERR, engIntr))
        {
            dprintf("lw: + LW_PMSPDEC_DBFDMA_INTRPTSTAT_CFGWRERR_TRUE\n");
        }

        if ( DRF_VAL( _PMSPDEC,_DBFDMA_INTRPTSTAT, _ROWEND, engIntr))
        {
            dprintf("lw: + LW_PMSPDEC_DBFDMA_INTRPTSTAT_ROWEND_TRUE\n");
            dprintf("lw:    LW_PMSPDEC_DBFDMA_ROW_COUNTER_REG:     0x%08x\n",
                GPU_REG_RD_DRF(_PMSPDEC, _DBFDMA_ROW_COUNTER, _REG));
        }
    }

    data32 = GPU_REG_RD32(LW_PMSPDEC_MVSTATUS);


    dprintf("lw: LW_PMSPDEC_MVSTATUS:         0x%08x\n", data32);

    if ( DRF_VAL( _PMSPDEC, _MVSTATUS, _IS_BUSY, data32))
    {
        dprintf("lw: + LW_PMSPDEC_MVSTATUS_IS_BUSY\n");
        addUnitErr("\t LW_PMSPDEC_MVSTATUS_IS_BUSY\n");
        status = LW_ERR_GENERIC;
    }

    if ( DRF_VAL( _PMSPDEC, _MVSTATUS, _WCOL_BUSY, data32))
    {
        dprintf("lw: + LW_PMSPDEC_MVSTATUS_WCOL_BUSY\n");
        addUnitErr("\t LW_PMSPDEC_MVSTATUS_WCOL_BUSY\n");
        status = LW_ERR_GENERIC;
    }

    if ( DRF_VAL( _PMSPDEC, _MVSTATUS, _RCOL_BUSY, data32))
    {
        dprintf("lw: + LW_PMSPDEC_MVSTATUS_RCOL_BUSY\n");
        addUnitErr("\t LW_PMSPDEC_MVSTATUS_RCOL_BUSY\n");
        status = LW_ERR_GENERIC;
    }

    if ( DRF_VAL( _PMSPDEC, _MVSTATUS, _OSPRDS_BUSY, data32))
    {
        dprintf("lw: + LW_PMSPDEC_MVSTATUS_OSPRDS_BUSY\n");
        addUnitErr("\t LW_PMSPDEC_MVSTATUS_OSPRDS_BUSY\n");
        status = LW_ERR_GENERIC;
    }

    if ( DRF_VAL( _PMSPDEC, _MVSTATUS, _OSPDBL_BUSY, data32))
    {
        dprintf("lw: + LW_PMSPDEC_MVSTATUS_OSPDBL_BUSY\n");
        addUnitErr("\t LW_PMSPDEC_MVSTATUS_OSPDBL_BUSY\n");
        status = LW_ERR_GENERIC;
    }

    if ( DRF_VAL( _PMSPDEC, _MVSTATUS, _OSPREC_BUSY, data32))
    {
        dprintf("lw: + LW_PMSPDEC_MVSTATUS_OSPREC_BUSY\n");
        addUnitErr("\t LW_PMSPDEC_MVSTATUS_OSPREC_BUSY\n");
        status = LW_ERR_GENERIC;
    }

    if ( DRF_VAL( _PMSPDEC, _MVSTATUS, _OSPMCS_BUSY, data32))
    {
        dprintf("lw: + LW_PMSPDEC_MVSTATUS_OSPMCS_BUSY\n");\
        addUnitErr("\t LW_PMSPDEC_MVSTATUS_OSPMCS_BUSY\n");
        status = LW_ERR_GENERIC;
    }


    //get mv error status and mask
    data32 = GPU_REG_RD32(LW_PMSPDEC_MVERRORSTAT);
    data32 &= GPU_REG_RD32(LW_PMSPDEC_MVERRORMASK);

    dprintf("lw: LW_PMSPDEC_MVERRORSTAT:      0x%08x\n", data32);

    if (data32 !=0)
    {
        addUnitErr("\t LW_PMSPDEC_MVERRORSTAT reported error\n");
        status = LW_ERR_GENERIC;
    }

    if ( DRF_VAL( _PMSPDEC, _MVERRORSTAT, _MVY_RANGE128, data32))
        dprintf("lw: + LW_PMSPDEC_MVERRORSTAT_MVY_RANGE128\n");

    if ( DRF_VAL( _PMSPDEC, _MVERRORSTAT, _MVY_RANGE256, data32))
        dprintf("lw: + LW_PMSPDEC_MVERRORSTAT_MVY_RANGE256\n");

     if ( DRF_VAL( _PMSPDEC, _MVERRORSTAT, _MVY_RANGE512, data32))
        dprintf("lw: + LW_PMSPDEC_MVERRORSTAT_MVY_RANGE512\n");

     if ( DRF_VAL( _PMSPDEC, _MVERRORSTAT, _VLD_MBERROR, data32))
        dprintf("lw: + LW_PMSPDEC_MVERRORSTAT_VLD_MBERROR\n");

     if ( DRF_VAL( _PMSPDEC, _MVERRORSTAT, _H264_INTRA16X16, data32))
        dprintf("lw: + LW_PMSPDEC_MVERRORSTAT_H264_INTRA16X16\n");

     if ( DRF_VAL( _PMSPDEC, _MVERRORSTAT, _H264_INTRA8X8, data32))
        dprintf("lw: + LW_PMSPDEC_MVERRORSTAT_H264_INTRA8X8\n");

     if ( DRF_VAL( _PMSPDEC, _MVERRORSTAT, _H264_INTRA4X4, data32))
        dprintf("lw: + LW_PMSPDEC_MVERRORSTAT_H264_INTRA4X4\n");

     if ( DRF_VAL( _PMSPDEC, _MVERRORSTAT, _H264_BIPRED, data32))
        dprintf("lw: + LW_PMSPDEC_MVERRORSTAT_H264_BIPRED\n");

     if ( DRF_VAL( _PMSPDEC, _MVERRORSTAT, _H264_NEI_MB, data32))
        dprintf("lw: + LW_PMSPDEC_MVERRORSTAT_H264_NEI_MB\n");

     if ( DRF_VAL( _PMSPDEC, _MVERRORSTAT, _H264_NEI_PART, data32))
        dprintf("lw: + LW_PMSPDEC_MVERRORSTAT_H264_NEI_PART\n");

     if ( DRF_VAL( _PMSPDEC, _MVERRORSTAT, _SW_ERR, data32))
        dprintf("lw: + LW_PMSPDEC_MVERRORSTAT_SW_ERR\n");

     //
     //print falcon states
     //Bit |  Signal meaning
     //0      FALCON busy
     //1      FBIF busy
     //2      MV busy
     //3      IQT busy
     //4      MC busy
     //5      REC busy
     //6      HIST busy
     //7      DBF busy
     //8      DBFDMA busy
     //9      CSBM busy
     //10-15  tied to zero.
    //

    data32 = GPU_REG_RD32(LW_PMSPDEC_FALCON_IDLESTATE);

    if ( DRF_VAL( _PMSPDEC, _FALCON_IDLESTATE, _FALCON_BUSY, data32))
    {
        dprintf("lw: + LW_PMSPDEC_FALCON_IDLESTATE_FALCON_BUSY\n");
        addUnitErr("\t LW_PMSPDEC_FALCON_IDLESTATE_FALCON_BUSY\n");
        status = LW_ERR_GENERIC;
    }

    if ( data32 & DRF_SHIFTMASK(1:1))
    {
        dprintf("lw: + LW_PMSPDEC_FALCON_IDLESTATE_FBIF_BUSY\n");
        addUnitErr("\t LW_PMSPDEC_FALCON_IDLESTATE_FBIF_BUSY\n");
        status = LW_ERR_GENERIC;
    }

    if ( data32 & DRF_SHIFTMASK(2:2))
    {
        dprintf("lw: + LW_PMSPDEC_FALCON_IDLESTATE_MV_BUSY\n");
        addUnitErr("\t LW_PMSPDEC_FALCON_IDLESTATE_MV_BUSY\n");
        status = LW_ERR_GENERIC;
    }

    if ( data32 & DRF_SHIFTMASK(3:3))
    {
        dprintf("lw: + LW_PMSPDEC_FALCON_IDLESTATE_IQT_BUSY\n");
        addUnitErr("\t LW_PMSPDEC_FALCON_IDLESTATE_IQT_BUSY\n");
        status = LW_ERR_GENERIC;
    }

    if ( data32 & DRF_SHIFTMASK(4:4))
    {
        dprintf("lw: + LW_PMSPDEC_FALCON_IDLESTATE_MC_BUSY\n");
        addUnitErr("\t LW_PMSPDEC_FALCON_IDLESTATE_MC_BUSY\n");
        status = LW_ERR_GENERIC;
    }

    if ( data32 & DRF_SHIFTMASK(5:5))
    {
        dprintf("lw: + LW_PMSPDEC_FALCON_IDLESTATE_REC_BUSY\n");
        addUnitErr("\t LW_PMSPDEC_FALCON_IDLESTATE_REC_BUSY\n");
        status = LW_ERR_GENERIC;
    }

    if ( data32 & DRF_SHIFTMASK(6:6))
    {
        dprintf("lw: + LW_PMSPDEC_FALCON_IDLESTATE_HIST_BUSY\n");
        addUnitErr("\t LW_PMSPDEC_FALCON_IDLESTATE_HIST_BUSY\n");
        status = LW_ERR_GENERIC;
    }

    if ( data32 & DRF_SHIFTMASK(7:7))
    {
        dprintf("lw: + LW_PMSPDEC_FALCON_IDLESTATE_DBF_BUSY\n");
        addUnitErr("\t LW_PMSPDEC_FALCON_IDLESTATE_DBF_BUSY\n");
        status = LW_ERR_GENERIC;
    }

    if ( data32 & DRF_SHIFTMASK(8:8))
    {
        dprintf("lw: + LW_PMSPDEC_FALCON_IDLESTATE_DBFDMA_BUSY\n");
        addUnitErr("\t LW_PMSPDEC_FALCON_IDLESTATE_DBFDMA_BUSY\n");
        status = LW_ERR_GENERIC;
    }

    if ( data32 & DRF_SHIFTMASK(9:9))
    {
        dprintf("lw: + LW_PMSPDEC_FALCON_IDLESTATE_CSBM_BUSY\n");
        addUnitErr("\t LW_PMSPDEC_FALCON_IDLESTATE_CSBM_BUSY\n");
        status = LW_ERR_GENERIC;
    }

    data32 = GPU_REG_RD32(LW_PMSPDEC_FALCON_FHSTATE);


    if ( DRF_VAL( _PMSPDEC, _FALCON_FHSTATE, _FALCON_HALTED, data32))
    {
        dprintf("lw: + LW_PMSPDEC_FALCON_FHSTATE_FALCON_HALTED\n");
        addUnitErr("\t LW_PMSPDEC_FALCON_FHSTATE_FALCON_HALTED\n");
        status = LW_ERR_GENERIC;
    }

    if ( data32 & DRF_SHIFTMASK(1:1))
    {
        dprintf("lw: + LW_PMSPDEC_FALCON_FHSTATE_FBIF_HALTED\n");
        addUnitErr("\t LW_PMSPDEC_FALCON_FHSTATE_FBIF_HALTED\n");
        status = LW_ERR_GENERIC;
    }

    if ( data32 & DRF_SHIFTMASK(2:2))
    {
        dprintf("lw: + LW_PMSPDEC_FALCON_FHSTATE_MV_HALTED\n");
        addUnitErr("\t LW_PMSPDEC_FALCON_FHSTATE_MV_HALTED\n");
        status = LW_ERR_GENERIC;
    }

    if ( data32 & DRF_SHIFTMASK(3:3))
    {
        dprintf("lw: + LW_PMSPDEC_FALCON_FHSTATE_IQT_HALTED\n");
        addUnitErr("\t LW_PMSPDEC_FALCON_FHSTATE_IQT_HALTED\n");
        status = LW_ERR_GENERIC;
    }

    if ( data32 & DRF_SHIFTMASK(4:4))
    {
        dprintf("lw: + LW_PMSPDEC_FALCON_FHSTATE_MC_HALTED\n");
        addUnitErr("\t LW_PMSPDEC_FALCON_FHSTATE_MC_HALTED\n");
        status = LW_ERR_GENERIC;
    }

    if ( data32 & DRF_SHIFTMASK(5:5))
    {
        dprintf("lw: + LW_PMSPDEC_FALCON_FHSTATE_REC_HALTED\n");
        addUnitErr("\t LW_PMSPDEC_FALCON_FHSTATE_REC_HALTED\n");
        status = LW_ERR_GENERIC;
    }

    if ( data32 & DRF_SHIFTMASK(6:6))
    {
        dprintf("lw: + LW_PMSPDEC_FALCON_FHSTATE_HIST_HALTED\n");
        addUnitErr("\t LW_PMSPDEC_FALCON_FHSTATE_HIST_HALTED\n");
        status = LW_ERR_GENERIC;
    }

    if ( data32 & DRF_SHIFTMASK(7:7))
    {
        dprintf("lw: + LW_PMSPDEC_FALCON_FHSTATE_DBF_HALTED\n");
        addUnitErr("\t LW_PMSPDEC_FALCON_FHSTATE_DBF_HALTED\n");
        status = LW_ERR_GENERIC;
    }

    if ( data32 & DRF_SHIFTMASK(8:8))
    {
        dprintf("lw: + LW_PMSPDEC_FALCON_FHSTATE_DBFDMA_HALTED\n");
        addUnitErr("\t LW_PMSPDEC_FALCON_FHSTATE_DBFDMA_HALTED\n");
        status = LW_ERR_GENERIC;
    }

    if ( data32 & DRF_SHIFTMASK(9:9))
    {
        dprintf("lw: + LW_PMSPDEC_FALCON_FHSTATE_CSBM_HALTED\n");
        addUnitErr("\t LW_PMSPDEC_FALCON_FHSTATE_CSBM_HALTED\n");
        status = LW_ERR_GENERIC;
    }

    if ( DRF_VAL( _PMSPDEC, _FALCON_FHSTATE, _ENGINE_FAULTED, data32))
    {
        dprintf("lw: + LW_PMSPDEC_FALCON_FHSTATE_ENGINE_FAULTED\n");
        addUnitErr("\t LW_PMSPDEC_FALCON_FHSTATE_ENGINE_FAULTED\n");
        status = LW_ERR_GENERIC;
    }

    if ( DRF_VAL( _PMSPDEC, _FALCON_FHSTATE, _STALL_REQ, data32))
    {
        dprintf("lw: + LW_PMSPDEC_FALCON_FHSTATE_STALL_REQ\n");
        addUnitErr("\t LW_PMSPDEC_FALCON_FHSTATE_STALL_REQ\n");
        status = LW_ERR_GENERIC;
    }

    //print falcon ctl regs
    data32 = GPU_REG_RD32(LW_PMSPDEC_FALCON_ENGCTL);

    if ( DRF_VAL( _PMSPDEC, _FALCON_ENGCTL, _ILW_CONTEXT, data32))
    {
        dprintf("lw: + LW_PMSPDEC_FALCON_ENGCTL_ILW_CONTEXT\n");
        addUnitErr("\t LW_PMSPDEC_FALCON_ENGCTL_ILW_CONTEXT\n");

        status = LW_ERR_GENERIC;
    }

    if ( DRF_VAL( _PMSPDEC, _FALCON_ENGCTL, _STALLREQ, data32))
    {
        dprintf("lw: + LW_PMSPDEC_FALCON_ENGCTL_STALLREQ\n");
        addUnitErr("\t LW_PMSPDEC_FALCON_ENGCTL_STALLREQ\n");
        status = LW_ERR_GENERIC;
    }

    data32 = GPU_REG_RD32(LW_PMSPDEC_FALCON_CPUCTL);

    if ( DRF_VAL( _PMSPDEC, _FALCON_CPUCTL, _IILWAL, data32))
    {
        dprintf("lw: + LW_PMSPDEC_FALCON_CPUCTL_IILWAL\n");
        addUnitErr("\t LW_PMSPDEC_FALCON_CPUCTL_IILWAL\n");
        status = LW_ERR_GENERIC;
    }

    if ( DRF_VAL( _PMSPDEC, _FALCON_CPUCTL, _HALTED, data32))
    {
        dprintf("lw: + LW_PMSPDEC_FALCON_CPUCTL_HALTED\n");
        addUnitErr("\t LW_PMSPDEC_FALCON_CPUCTL_HALTED\n");
        status = LW_ERR_GENERIC;
    }

    if ( DRF_VAL( _PMSPDEC, _FALCON_CPUCTL, _STOPPED, data32))
    {
        dprintf("lw: + LW_PMSPDEC_FALCON_CPUCTL_STOPPED\n");
        addUnitErr("\t Warning: LW_PMSPDEC_FALCON_CPUCTL_STOPPED\n");
        //status = LW_ERR_GENERIC;
    }

    // state of mthd/ctx interface
    data32 = GPU_REG_RD32(LW_PMSPDEC_FALCON_ITFEN);

    if (DRF_VAL( _PMSPDEC, _FALCON_ITFEN, _CTXEN, data32))
    {
        dprintf("lw: + LW_PMSPDEC_FALCON_ITFEN_CTXEN enabled\n");

        if (pFalcon[indexGpu].falconTestCtxState(LW_FALCON_MSPDEC_BASE, "PMSPDEC") == LW_ERR_GENERIC)
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
        dprintf("lw: + LW_PMSPDEC_FALCON_ITFEN_CTXEN disabled\n");
    }

    if ( DRF_VAL( _PMSPDEC, _FALCON_ITFEN, _MTHDEN, data32))
    {
        dprintf("lw: + LW_PMSPDEC_FALCON_ITFEN_MTHDEN enabled\n");
    }
    else
    {
        dprintf("lw: + LW_PMSPDEC_FALCON_ITFEN_MTHDEN disabled\n");
    }

    //check if falcon is hung (instr ptr)
    if ( pFalcon[indexGpu].falconTestPC(LW_FALCON_MSPDEC_BASE, "PMSPDEC") == LW_ERR_GENERIC )
    {


        dprintf("lw: Falcon instruction pointer is stuck or invalid\n");

        //TODO: treat falcon PC errors as warnings now, need to report as error
        addUnitErr("\t Warning: Falcon instruction pointer is stuck or invalid\n");
        //status = LW_ERR_GENERIC;

    }

    return status;
}

LW_STATUS msdecTestMspppState_GK104( void )
{

    LW_STATUS    status = LW_OK;
    LwU32   regIntr;
    LwU32   regIntrEn;
    LwU32   engIntr;
    LwU32   engIntrEn;
    LwU32   engErr;
    LwU32   data32;

    //check falcon interrupts
    regIntr = GPU_REG_RD32(LW_PMSPPP_FALCON_IRQSTAT);
    regIntrEn = GPU_REG_RD32(LW_PMSPPP_FALCON_IRQMASK);
    regIntr &= regIntrEn;

    if ( !DRF_VAL(_PMSPPP, _FALCON_IRQMASK, _GPTMR, regIntrEn))
        dprintf("lw: LW_PMSPPP_FALCON_IRQMASK_GPTMR disabled\n");

    if ( !DRF_VAL(_PMSPPP, _FALCON_IRQMASK, _MTHD, regIntrEn))
        dprintf("lw: LW_PMSPPP_FALCON_IRQMASK_MTHD disabled\n");

    if ( !DRF_VAL(_PMSPPP, _FALCON_IRQMASK, _CTXSW, regIntrEn))
        dprintf("lw: LW_PMSPPP_FALCON_IRQMASK_CTXSW disabled\n");

    if ( !DRF_VAL(_PMSPPP, _FALCON_IRQMASK, _HALT, regIntrEn))
        dprintf("lw: LW_PMSPPP_FALCON_IRQMASK_HALT disabled\n");

    if ( !DRF_VAL(_PMSPPP, _FALCON_IRQMASK, _SWGEN0, regIntrEn))
        dprintf("lw: LW_PMSPPP_FALCON_IRQMASK_SWGEN0 disabled\n");

    if ( !DRF_VAL(_PMSPPP, _FALCON_IRQMASK, _SWGEN1, regIntrEn))
        dprintf("lw: LW_PMSPPP_FALCON_IRQMASK_SWGEN1 disabled\n");

    if ( !(regIntrEn & 0x100))
        dprintf("lw: LW_PMSPPP_FALCON_IRQMASK: FBIF Ctx error interrupt disabled\n");

    if ( !(regIntrEn & 0x200))
        dprintf("lw: LW_PMSPPP_FALCON_IRQMASK: Limit violation interrupt disabled\n");

    if ( !(regIntrEn & 0x400))
        dprintf("lw: LW_PMSPPP_FALCON_IRQMASK: TF done interrupt disabled\n");

    if ( !(regIntrEn & 0x800))
        dprintf("lw: LW_PMSPPP_FALCON_IRQMASK: TF error interrupt disabled\n");

    if ( !(regIntrEn & 0x1000))
        dprintf("lw: LW_PMSPPP_FALCON_IRQMASK: CRB interrupt disabled\n");

    if ( !(regIntrEn & 0x2000))
        dprintf("lw: LW_PMSPPP_FALCON_IRQMASK: VC interrupt disabled\n");

    if ( !(regIntrEn & 0x4000))
        dprintf("lw: LW_PMSPPP_FALCON_IRQMASK: FGT interrupt disabled\n");

    //if any interrupt pending, set error
    if (regIntr != 0)
    {
        addUnitErr("\t MSPPP interrupts are pending\n");
        status = LW_ERR_GENERIC;
    }

    if ( DRF_VAL( _PMSPPP,_FALCON_IRQSTAT, _GPTMR, regIntr))
    {
        dprintf("lw: LW_PMSPPP_FALCON_IRQSTAT_GPTMR pending\n");

        dprintf("lw: LW_PMSPPP_FALCON_GPTMRINT:    0x%08x\n",
            GPU_REG_RD32(LW_PMSPPP_FALCON_GPTMRINT) );
        dprintf("lw: LW_PMSPPP_FALCON_GPTMRVAL:    0x%08x\n",
            GPU_REG_RD32(LW_PMSPPP_FALCON_GPTMRVAL) );

    }

    if ( DRF_VAL( _PMSPPP,_FALCON_IRQSTAT, _MTHD, regIntr))
    {
        dprintf("lw: LW_PMSPPP_FALCON_IRQSTAT_MTHD pending\n");

        dprintf("lw: LW_PMSPPP_FALCON_MTHDDATA_DATA:    0x%08x\n",
            GPU_REG_RD32(LW_PMSPPP_FALCON_MTHDDATA) );

        data32 = GPU_REG_RD32(LW_PMSPPP_FALCON_MTHDID);
        dprintf("lw: LW_PMSPPP_FALCON_MTHDID_ID:    0x%08x\n",
           DRF_VAL( _PMSPPP,_FALCON_MTHDID, _ID, data32)  );
        dprintf("lw: LW_PMSPPP_FALCON_MTHDID_SUBCH:    0x%08x\n",
           DRF_VAL( _PMSPPP,_FALCON_MTHDID, _SUBCH, data32)  );
        dprintf("lw: LW_PMSPPP_FALCON_MTHDID_PRIV:    0x%08x\n",
           DRF_VAL( _PMSPPP,_FALCON_MTHDID, _PRIV, data32)  );
    }

    if ( DRF_VAL( _PMSPPP,_FALCON_IRQSTAT, _CTXSW, regIntr))
    {
        dprintf("lw: LW_PMSPPP_FALCON_IRQSTAT_CTXSW pending\n");
    }

    if ( DRF_VAL( _PMSPPP,_FALCON_IRQSTAT, _HALT, regIntr))
    {
        dprintf("lw: LW_PMSPPP_FALCON_IRQSTAT_HALT pending\n");
    }

    if ( DRF_VAL( _PMSPPP,_FALCON_IRQSTAT, _SWGEN0, regIntr))
    {
        dprintf("lw: LW_PMSPPP_FALCON_IRQSTAT_SWGEN0 pending\n");

        pFalcon[indexGpu].falconPrintMailbox(LW_FALCON_MSPPP_BASE);
    }

    if ( DRF_VAL( _PMSPPP,_FALCON_IRQSTAT, _SWGEN1, regIntr))
    {
        dprintf("lw: LW_PMSPPP_FALCON_IRQSTAT_SWGEN1 pending\n");
    }

     //
     //Bit |  Signal meaning
     //8      FBIF Ctx error interrupt.
     //9      Limit violation interrupt.
     //10     TF (Tiling Formater) done interrupt.
     //11     TF Error interrupt.
     //12     CRB interrupt.
     //13     VC interrupt.
     //14     FGT interrupt.
     //

    if ( regIntr & 0x100)
    {
        dprintf("lw: LW_PMSPPP_FALCON_IRQSTAT: FBIF Ctx error interrupt\n");

        data32 = GPU_REG_RD32(LW_PMSPPP_FBIF_CTL);

        if (DRF_VAL(_PMSPPP, _FBIF_CTL, _ENABLE, data32))
        {
            if (DRF_VAL(_PMSPPP, _FBIF_CTL, _ILWAL_CONTEXT, data32))
            {
                dprintf("lw: + LW_PMSPPP_FBIF_CTL_ILWAL_CONTEXT\n");
            }
        }
    }

    if (regIntr & 0x200)
    {
        dprintf("lw: LW_PMSPPP_FALCON_IRQSTAT: Limit violation interrupt\n");
    }

    //TF done
    if (regIntr & 0x400)
    {
        dprintf("lw: LW_PMSPPP_FALCON_IRQSTAT: TF done interrupt\n");

        engIntr = GPU_REG_RD32(LW_PMSPPP_TF_INTR);
        engIntrEn = GPU_REG_RD32(LW_PMSPPP_TF_INTEN);
        engIntr &= engIntrEn;

        //print disabled interrupts
        if ( !DRF_VAL(_PMSPPP, _TF_INTEN, _LUMA_DONE_EN, engIntrEn))
            dprintf("lw: + LW_PMSPPP_TF_INTEN_LUMA_DONE_EN disabled\n");

        if ( !DRF_VAL(_PMSPPP, _TF_INTEN, _CHROMA_DONE_EN, engIntrEn))
            dprintf("lw: + LW_PMSPPP_TF_INTEN_CHROMA_DONE_EN disabled\n");

        if ( !DRF_VAL(_PMSPPP, _TF_INTEN, _HW_ERR_EN, engIntrEn))
            dprintf("lw: + LW_PMSPPP_TF_INTEN_HW_ERR_EN disabled\n");

        if ( !DRF_VAL(_PMSPPP, _TF_INTEN, _FORCE_INT, engIntrEn))
            dprintf("lw: + LW_PMSPPP_TF_INTEN_FORCE_INT disabled\n");

        //check TF interrupts
        if ( DRF_VAL( _PMSPPP,_TF_INTR, _LUMA_DONE, engIntr))
        {
            dprintf("lw: + LW_PMSPPP_TF_INTR_LUMA_DONE pending\n");
        }

        if ( DRF_VAL( _PMSPPP,_TF_INTR, _CHROMA_DONE, engIntr))
        {
            dprintf("lw: + LW_PMSPPP_TF_INTR_CHROMA_DONE pending\n");
        }

        if ( DRF_VAL( _PMSPPP,_TF_INTR, _HW_ERR, engIntr))
        {
            dprintf("lw: + LW_PMSPPP_TF_INTR_HW_ERR pending\n");
        }
    }


    //TF err interrupt
    if (regIntr & 0x800)
    {
        dprintf("lw: LW_PMSPPP_FALCON_IRQSTAT: TF error interrupt\n");

        engIntr = GPU_REG_RD32(LW_PMSPPP_TF_INTR);
        engIntrEn = GPU_REG_RD32(LW_PMSPPP_TF_INTEN);
        engIntr &= engIntrEn;
        engErr = GPU_REG_RD32(LW_PMSPPP_TF_STATUS);

        //print disabled interrupts
        if ( !DRF_VAL(_PMSPPP, _TF_INTEN, _LUMA_DONE_EN, engIntrEn))
            dprintf("lw: + LW_PMSPPP_TF_INTEN_LUMA_DONE_EN disabled\n");

        if ( !DRF_VAL(_PMSPPP, _TF_INTEN, _CHROMA_DONE_EN, engIntrEn))
            dprintf("lw: + LW_PMSPPP_TF_INTEN_CHROMA_DONE_EN disabled\n");

        if ( !DRF_VAL(_PMSPPP, _TF_INTEN, _HW_ERR_EN, engIntrEn))
            dprintf("lw: + LW_PMSPPP_TF_INTEN_HW_ERR_EN disabled\n");

        if ( !DRF_VAL(_PMSPPP, _TF_INTEN, _FORCE_INT, engIntrEn))
            dprintf("lw: + LW_PMSPPP_TF_INTEN_FORCE_INT disabled\n");

        //check TF interrupts
        if ( DRF_VAL( _PMSPPP,_TF_INTR, _LUMA_DONE, engIntr))
        {
            dprintf("lw: + LW_PMSPPP_TF_INTR_LUMA_DONE pending\n");
        }

        if ( DRF_VAL( _PMSPPP,_TF_INTR, _CHROMA_DONE, engIntr))
        {
            dprintf("lw: + LW_PMSPPP_TF_INTR_CHROMA_DONE pending\n");
        }

        if ( DRF_VAL( _PMSPPP,_TF_INTR, _HW_ERR, engIntr))
        {
            dprintf("lw: + LW_PMSPPP_TF_INTR_HW_ERR pending\n");
        }

        //check TF error status
        switch ( DRF_VAL( _PMSPPP, _TF_STATUS, _ERR_STATUS, engIntr))
        {
            case LW_PMSPPP_TF_STATUS_ERR_STATUS_INIT:
                dprintf("lw: + LW_PMSPPP_TF_STATUS_ERR_STATUS_INIT\n");
                break;
            case LW_PMSPPP_TF_STATUS_ERR_STATUS_IDLEWATCHDOG:
                dprintf("lw: + LW_PMSPPP_TF_STATUS_ERR_STATUS_IDLEWATCHDOG\n");
                break;
            case LW_PMSPPP_TF_STATUS_ERR_STATUS_TIMEOUT:
                dprintf("lw: + LW_PMSPPP_TF_STATUS_ERR_STATUS_TIMEOUT\n");
                break;
            case LW_PMSPPP_TF_STATUS_ERR_STATUS_PROGERR:
                dprintf("lw: + LW_PMSPPP_TF_STATUS_ERR_STATUS_PROGERR\n");
                break;
            case LW_PMSPPP_TF_STATUS_ERR_STATUS_HWERR:
                dprintf("lw: + LW_PMSPPP_TF_STATUS_ERR_STATUS_HWERR\n");
                break;
            default:
                dprintf("lw: + Uknown LW_PMSPPP_TF_STATUS_ERR_STATUS: 0x%x\n",
                    DRF_VAL( _PMSPPP, _TF_STATUS, _ERR_STATUS, engIntr));
                break;
        }

    }

    //CRB interrupt
    if (regIntr & 0x1000)
    {
        dprintf("lw: LW_PMSPPP_FALCON_IRQSTAT: CRB interrupt\n");

        engIntr = GPU_REG_RD32(LW_PMSPPP_CRB_INTR);
        engIntrEn = GPU_REG_RD32(LW_PMSPPP_CRB_INTEN);
        engIntr &= engIntrEn;
        engErr = GPU_REG_RD32(LW_PMSPPP_CRB_STATUS);

        //print disabled interrupts
        if ( !DRF_VAL(_PMSPPP, _CRB_INTEN, _SW_ERR_EN, engIntrEn))
            dprintf("lw: + LW_PMSPPP_CRB_INTEN_SW_ERR_EN disabled\n");

        if ( !DRF_VAL(_PMSPPP, _CRB_INTEN, _FORCE_INT, engIntrEn))
            dprintf("lw: + LW_PMSPPP_CRB_INTEN_FORCE_INT disabled\n");

        //check CRB interrupts
        if ( DRF_VAL( _PMSPPP,_CRB_INTR, _SW_ERR, engIntr))
        {
            dprintf("lw: + LW_PMSPPP_CRB_INTR_SW_ERR pending\n");
        }

        //check CRB error status
        switch ( DRF_VAL( _PMSPPP, _CRB_STATUS, _ERR_STATUS, engIntr))
        {
            case LW_PMSPPP_CRB_STATUS_ERR_STATUS_INIT:
                dprintf("lw: + LW_PMSPPP_CRB_STATUS_ERR_STATUS_INIT\n");
                break;
            case LW_PMSPPP_CRB_STATUS_ERR_STATUS_IDLEWATCHDOG:
                dprintf("lw: + LW_PMSPPP_CRB_STATUS_ERR_STATUS_IDLEWATCHDOG\n");
                break;
            case LW_PMSPPP_CRB_STATUS_ERR_STATUS_TIMEOUT:
                dprintf("lw: + LW_PMSPPP_CRB_STATUS_ERR_STATUS_TIMEOUT\n");
                break;
            case LW_PMSPPP_CRB_STATUS_ERR_STATUS_PROGERR:
                dprintf("lw: + LW_PMSPPP_CRB_STATUS_ERR_STATUS_PROGERR\n");
                break;
            case LW_PMSPPP_CRB_STATUS_ERR_STATUS_HWERR:
                dprintf("lw: + LW_PMSPPP_CRB_STATUS_ERR_STATUS_HWERR\n");
                break;
            default:
                dprintf("lw: + Uknown LW_PMSPPP_CRB_STATUS_ERR_STATUS: 0x%x\n",
                    DRF_VAL( _PMSPPP, _CRB_STATUS, _ERR_STATUS, engIntr));
                break;
        }
    }

    //VC interrupt
    if (regIntr & 0x2000)
    {
        dprintf("lw: LW_PMSPPP_FALCON_IRQSTAT: VC interrupt\n");

        engIntr = GPU_REG_RD32(LW_PMSPPP_VC_INTR);
        engIntrEn = GPU_REG_RD32(LW_PMSPPP_VC_INTEN);
        engIntr &= engIntrEn;
        engErr = GPU_REG_RD32(LW_PMSPPP_VC_STATUS);

        //print disabled interrupts
        if ( !DRF_VAL(_PMSPPP, _VC_INTEN, _HW_ERR_EN, engIntrEn))
            dprintf("lw: + LW_PMSPPP_VC_INTEN_HW_ERR_EN disabled\n");

        if ( !DRF_VAL(_PMSPPP, _VC_INTEN, _FORCE_INT, engIntrEn))
            dprintf("lw: + LW_PMSPPP_VC_INTEN_FORCE_INT disabled\n");

        //check VC interrupts
        if ( DRF_VAL( _PMSPPP,_VC_INTR, _HW_ERR, engIntr))
        {
            dprintf("lw: + LW_PMSPPP_VC_INTR_HW_ERR pending\n");
        }

        //check VC error status
        switch ( DRF_VAL( _PMSPPP, _VC_STATUS, _ERR_STATUS, engIntr))
        {
        case LW_PMSPPP_VC_STATUS_ERR_STATUS_INIT:
            dprintf("lw: + LW_PMSPPP_VC_STATUS_ERR_STATUS_INIT\n");
            break;
        case LW_PMSPPP_VC_STATUS_ERR_STATUS_IDLEWATCHDOG:
            dprintf("lw: + LW_PMSPPP_VC_STATUS_ERR_STATUS_IDLEWATCHDOG\n");
            break;
        case LW_PMSPPP_VC_STATUS_ERR_STATUS_TIMEOUT:
            dprintf("lw: + LW_PMSPPP_VC_STATUS_ERR_STATUS_TIMEOUT\n");
            break;
        case LW_PMSPPP_VC_STATUS_ERR_STATUS_PROGERR:
            dprintf("lw: + LW_PMSPPP_VC_STATUS_ERR_STATUS_PROGERR\n");
            break;
        case LW_PMSPPP_VC_STATUS_ERR_STATUS_HWERR:
            dprintf("lw: + LW_PMSPPP_VC_STATUS_ERR_STATUS_HWERR\n");
            break;
        default:
            dprintf("lw: + Uknown LW_PMSPPP_VC_STATUS_ERR_STATUS: 0x%x\n",
                DRF_VAL( _PMSPPP, _VC_STATUS, _ERR_STATUS, engIntr));
            break;
        }
    }


    //FGT interrupt
    if (regIntr & 0x4000)
    {
        dprintf("lw: LW_PMSPPP_FALCON_IRQSTAT: FGT interrupt\n");

        engIntr = GPU_REG_RD32(LW_PMSPPP_FGT_INTR);
        engIntrEn = GPU_REG_RD32(LW_PMSPPP_FGT_INTEN);
        engIntr &= engIntrEn;
        engErr = GPU_REG_RD32(LW_PMSPPP_FGT_STATUS);

        //print disabled interrupts
        if ( !DRF_VAL(_PMSPPP, _FGT_INTEN, _HW_ERR_EN, engIntrEn))
            dprintf("lw: + LW_PMSPPP_FGT_INTEN_HW_ERR_EN disabled\n");

        if ( !DRF_VAL(_PMSPPP, _FGT_INTEN, _FORCE_INT, engIntrEn))
            dprintf("lw: + LW_PMSPPP_FGT_INTEN_FORCE_INT disabled\n");

        //check FGT interrupts
        if ( DRF_VAL( _PMSPPP,_FGT_INTR, _HW_ERR, engIntr))
        {
            dprintf("lw: + LW_PMSPPP_FGT_INTR_HW_ERR pending\n");
        }

        //check FGT error status
        switch ( DRF_VAL( _PMSPPP, _FGT_STATUS, _ERR_STATUS, engIntr))
        {
        case LW_PMSPPP_FGT_STATUS_ERR_STATUS_INIT:
            dprintf("lw: + LW_PMSPPP_FGT_STATUS_ERR_STATUS_INIT\n");
            break;
        case LW_PMSPPP_FGT_STATUS_ERR_STATUS_IDLEWATCHDOG:
            dprintf("lw: + LW_PMSPPP_FGT_STATUS_ERR_STATUS_IDLEWATCHDOG\n");
            break;
        case LW_PMSPPP_FGT_STATUS_ERR_STATUS_TIMEOUT:
            dprintf("lw: + LW_PMSPPP_FGT_STATUS_ERR_STATUS_TIMEOUT\n");
            break;
        case LW_PMSPPP_FGT_STATUS_ERR_STATUS_PROGERR:
            dprintf("lw: + LW_PMSPPP_FGT_STATUS_ERR_STATUS_PROGERR\n");
            break;
        case LW_PMSPPP_FGT_STATUS_ERR_STATUS_HWERR:
            dprintf("lw: + LW_PMSPPP_FGT_STATUS_ERR_STATUS_HWERR\n");
            break;
        default:
            dprintf("lw: + Uknown LW_PMSPPP_FGT_STATUS_ERR_STATUS: 0x%x\n",
                DRF_VAL( _PMSPPP, _FGT_STATUS, _ERR_STATUS, engIntr));
            break;
        }
    }

    //print subeng statuses
    data32 = GPU_REG_RD32(LW_PMSPPP_TF_STATUS);

    dprintf("lw: LW_PMSPPP_TF_STATUS:       0x%08x\n", data32);

    if ( DRF_VAL( _PMSPPP, _TF_STATUS, _RUNNING, data32))
    {
        dprintf("lw: + LW_PMSPPP_TF_STATUS_RUNNING_WORKING\n");
        addUnitErr("\t LW_PMSPPP_TF_STATUS_RUNNING_WORKING\n");
        status = LW_ERR_GENERIC;
    }

    if ( !DRF_VAL( _PMSPPP, _TF_STATUS, _IDLE, data32))
    {
        dprintf("lw: + LW_PMSPPP_TF_STATUS_IDLE is busy (0x0)\n");
        addUnitErr("\t LW_PMSPPP_TF_STATUS_IDLE is busy (0x0)\n");
        status = LW_ERR_GENERIC;
    }

    data32 = GPU_REG_RD32(LW_PMSPPP_CRB_STATUS);

    dprintf("lw: LW_PMSPPP_CRB_STATUS:      0x%08x\n", data32);

    if ( DRF_VAL( _PMSPPP, _CRB_STATUS, _RUNNING, data32))
    {
        dprintf("lw: + LW_PMSPPP_CRB_STATUS_RUNNING_WORKING\n");
        addUnitErr("\t LW_PMSPPP_CRB_STATUS_RUNNING_WORKING\n");
        status = LW_ERR_GENERIC;
    }

    if ( !DRF_VAL( _PMSPPP, _CRB_STATUS, _IDLE, data32))
    {
        dprintf("lw: + LW_PMSPPP_CRB_STATUS_IDLE is busy (0x0)\n");
         addUnitErr("\t LW_PMSPPP_CRB_STATUS_IDLE is busy (0x0)\n");
        status = LW_ERR_GENERIC;
    }

    data32 = GPU_REG_RD32(LW_PMSPPP_VC_STATUS);

    dprintf("lw: LW_PMSPPP_VC_STATUS:       0x%08x\n", data32);

    if ( DRF_VAL( _PMSPPP, _VC_STATUS, _RUNNING, data32))
    {
        dprintf("lw: + LW_PMSPPP_VC_STATUS_RUNNING_WORKING\n");
        addUnitErr("\t LW_PMSPPP_VC_STATUS_RUNNING_WORKING\n");
        status = LW_ERR_GENERIC;
    }
    if ( !DRF_VAL( _PMSPPP, _VC_STATUS, _IDLE, data32))
    {
        dprintf("lw: + LW_PMSPPP_VC_STATUS_IDLE is busy (0x0)\n");
        addUnitErr("\t LW_PMSPPP_VC_STATUS_IDLE is busy (0x0)\n");
        status = LW_ERR_GENERIC;
    }

    data32 = GPU_REG_RD32(LW_PMSPPP_FGT_STATUS);

    dprintf("lw: LW_PMSPPP_FGT_STATUS:      0x%08x\n", data32);

    if ( DRF_VAL( _PMSPPP, _FGT_STATUS, _RUNNING, data32))
    {
        dprintf("lw: + LW_PMSPPP_FGT_STATUS_RUNNING_WORKING\n");
        addUnitErr("\t LW_PMSPPP_FGT_STATUS_RUNNING_WORKING\n");
        status = LW_ERR_GENERIC;
    }

    if ( !DRF_VAL( _PMSPPP, _FGT_STATUS, _IDLE, data32))
    {
        dprintf("lw: + LW_PMSPPP_FGT_STATUS_IDLE is busy (0x0)\n");
        addUnitErr("\t LW_PMSPPP_FGT_STATUS_IDLE is busy (0x0)\n");
        status = LW_ERR_GENERIC;
    }

    //
    //print falcon states
    //Bit |  Signal meaning
    //0      FALCON busy
    //1      FBIF busy
    //2      TF (Tiling Formater) busy
    //3      CB (Common Buffer) busy
    //4      VC (VC1 Process Unit) busy
    //5      FGT (Film Grain Tech) busy
    //6      CRB (Common Register Block) busy
    //7      HC (Histogram Collection) busy
    //

    data32 = GPU_REG_RD32(LW_PMSPPP_FALCON_IDLESTATE);

    if ( DRF_VAL( _PMSPPP, _FALCON_IDLESTATE, _FALCON_BUSY, data32))
    {
        dprintf("lw: + LW_PMSPPP_FALCON_IDLESTATE_FALCON_BUSY\n");
        addUnitErr("\t LW_PMSPPP_FALCON_IDLESTATE_FALCON_BUSY\n");
        status = LW_ERR_GENERIC;
    }

    if ( data32 & DRF_SHIFTMASK(1:1))
    {
        dprintf("lw: + LW_PMSPPP_FALCON_IDLESTATE_FBIF_BUSY\n");
        addUnitErr("\t LW_PMSPPP_FALCON_IDLESTATE_FBIF_BUSY\n");
        status = LW_ERR_GENERIC;
    }

    if ( data32 & DRF_SHIFTMASK(2:2))
    {
        dprintf("lw: + LW_PMSPPP_FALCON_IDLESTATE_TF_BUSY\n");
        addUnitErr("\t LW_PMSPPP_FALCON_IDLESTATE_TF_BUSY\n");
        status = LW_ERR_GENERIC;
    }

    if ( data32 & DRF_SHIFTMASK(3:3))
    {
        dprintf("lw: + LW_PMSPPP_FALCON_IDLESTATE_CB_BUSY\n");
        addUnitErr("\t LW_PMSPPP_FALCON_IDLESTATE_CB_BUSY\n");
        status = LW_ERR_GENERIC;
    }

    if ( data32 & DRF_SHIFTMASK(4:4))
    {
        dprintf("lw: + LW_PMSPPP_FALCON_IDLESTATE_VC_BUSY\n");
        addUnitErr("\t LW_PMSPPP_FALCON_IDLESTATE_VC_BUSY\n");
        status = LW_ERR_GENERIC;
    }

    if ( data32 & DRF_SHIFTMASK(5:5))
    {
        dprintf("lw: + LW_PMSPPP_FALCON_IDLESTATE_FGT_BUSY\n");
        addUnitErr("\t LW_PMSPPP_FALCON_IDLESTATE_FGT_BUSY\n");
        status = LW_ERR_GENERIC;
    }

    if ( data32 & DRF_SHIFTMASK(6:6))
    {
        dprintf("lw: + LW_PMSPPP_FALCON_IDLESTATE_CRB_BUSY\n");
        addUnitErr("\t LW_PMSPPP_FALCON_IDLESTATE_CRB_BUSY\n");
        status = LW_ERR_GENERIC;
    }

    if ( data32 & DRF_SHIFTMASK(7:7))
    {
        dprintf("lw: + LW_PMSPPP_FALCON_IDLESTATE_HC_BUSY\n");
        addUnitErr("\t LW_PMSPPP_FALCON_IDLESTATE_HC_BUSY\n");
        status = LW_ERR_GENERIC;
    }

    data32 = GPU_REG_RD32(LW_PMSPPP_FALCON_FHSTATE);

    if ( DRF_VAL( _PMSPPP, _FALCON_FHSTATE, _FALCON_HALTED, data32))
    {
        dprintf("lw: + LW_PMSPPP_FALCON_FHSTATE_FALCON_HALTED\n");
        addUnitErr("\t LW_PMSPPP_FALCON_FHSTATE_FALCON_HALTED\n");
        status = LW_ERR_GENERIC;
    }

    if ( data32 & DRF_SHIFTMASK(1:1))
    {
        dprintf("lw: + LW_PMSPPP_FALCON_FHSTATE_FBIF_HALTED\n");
        addUnitErr("\t LW_PMSPPP_FALCON_FHSTATE_FBIF_HALTED\n");
        status = LW_ERR_GENERIC;
    }

    if ( data32 & DRF_SHIFTMASK(2:2))
    {
        dprintf("lw: + LW_PMSPPP_FALCON_FHSTATE_TF_HALTED\n");
        addUnitErr("\t LW_PMSPPP_FALCON_FHSTATE_TF_HALTED\n");
        status = LW_ERR_GENERIC;
    }

    if ( data32 & DRF_SHIFTMASK(3:3))
    {
        dprintf("lw: + LW_PMSPPP_FALCON_FHSTATE_CB_HALTED\n");
        addUnitErr("\t LW_PMSPPP_FALCON_FHSTATE_CB_HALTED\n");
        status = LW_ERR_GENERIC;
    }

    if ( data32 & DRF_SHIFTMASK(4:4))
    {
        dprintf("lw: + LW_PMSPPP_FALCON_FHSTATE_VC_HALTED\n");
        addUnitErr("\t LW_PMSPPP_FALCON_FHSTATE_VC_HALTED\n");
        status = LW_ERR_GENERIC;
    }

    if ( data32 & DRF_SHIFTMASK(5:5))
    {
        dprintf("lw: + LW_PMSPPP_FALCON_FHSTATE_FGT_HALTED\n");
        addUnitErr("\t LW_PMSPPP_FALCON_FHSTATE_FGT_HALTED\n");
        status = LW_ERR_GENERIC;
    }

    if ( data32 & DRF_SHIFTMASK(6:6))
    {
        dprintf("lw: + LW_PMSPPP_FALCON_FHSTATE_CRB_HALTED\n");
        addUnitErr("\t LW_PMSPPP_FALCON_FHSTATE_CRB_HALTED\n");
        status = LW_ERR_GENERIC;
    }

    if ( data32 & DRF_SHIFTMASK(7:7))
    {
        dprintf("lw: + LW_PMSPPP_FALCON_FHSTATE_HC_HALTED\n");
        addUnitErr("\t LW_PMSPPP_FALCON_FHSTATE_HC_HALTED\n");
        status = LW_ERR_GENERIC;
    }

    if ( DRF_VAL( _PMSPPP, _FALCON_FHSTATE, _ENGINE_FAULTED, data32))
    {
        dprintf("lw: + LW_PMSPPP_FALCON_FHSTATE_ENGINE_FAULTED\n");
        addUnitErr("\t LW_PMSPPP_FALCON_FHSTATE_ENGINE_FAULTED\n");
        status = LW_ERR_GENERIC;
    }

    if ( DRF_VAL( _PMSPPP, _FALCON_FHSTATE, _STALL_REQ, data32))
    {
        dprintf("lw: + LW_PMSPPP_FALCON_FHSTATE_STALL_REQ\n");
        addUnitErr("\t LW_PMSPPP_FALCON_FHSTATE_STALL_REQ\n");
        status = LW_ERR_GENERIC;
    }

    //print falcon ctl regs
    data32 = GPU_REG_RD32(LW_PMSPPP_FALCON_ENGCTL);

    if ( DRF_VAL( _PMSPPP, _FALCON_ENGCTL, _ILW_CONTEXT, data32))
    {
        dprintf("lw: + LW_PMSPPP_FALCON_ENGCTL_ILW_CONTEXT\n");
        addUnitErr("\t LW_PMSPPP_FALCON_ENGCTL_ILW_CONTEXT\n");
        status = LW_ERR_GENERIC;
    }

    if ( DRF_VAL( _PMSPPP, _FALCON_ENGCTL, _STALLREQ, data32))
    {
        dprintf("lw: + LW_PMSPPP_FALCON_ENGCTL_STALLREQ\n");
        addUnitErr("\t LW_PMSPPP_FALCON_ENGCTL_STALLREQ\n");
        status = LW_ERR_GENERIC;
    }

    data32 = GPU_REG_RD32(LW_PMSPPP_FALCON_CPUCTL);

    if ( DRF_VAL( _PMSPPP, _FALCON_CPUCTL, _IILWAL, data32))
    {
        dprintf("lw: + LW_PMSPPP_FALCON_CPUCTL_IILWAL\n");
        addUnitErr("\t LW_PMSPPP_FALCON_CPUCTL_IILWAL\n");
        status = LW_ERR_GENERIC;
    }

    if ( DRF_VAL( _PMSPPP, _FALCON_CPUCTL, _HALTED, data32))
    {
        dprintf("lw: + LW_PMSPPP_FALCON_CPUCTL_HALTED\n");
        addUnitErr("\t LW_PMSPPP_FALCON_CPUCTL_HALTED\n");
        status = LW_ERR_GENERIC;
    }

    if ( DRF_VAL( _PMSPPP, _FALCON_CPUCTL, _STOPPED, data32))
    {
        dprintf("lw: + LW_PMSPPP_FALCON_CPUCTL_STOPPED\n");
        addUnitErr("\t Warning: LW_PMSPPP_FALCON_CPUCTL_STOPPED\n");
        //status = LW_ERR_GENERIC;
    }

    // state of mthd/ctx interface
    data32 = GPU_REG_RD32(LW_PMSPPP_FALCON_ITFEN);

    if (DRF_VAL( _PMSPPP, _FALCON_ITFEN, _CTXEN, data32))
    {
        dprintf("lw: + LW_PMSPPP_FALCON_ITFEN_CTXEN enabled\n");

        if (pFalcon[indexGpu].falconTestCtxState(LW_FALCON_MSPPP_BASE, "PMSPPP") == LW_ERR_GENERIC)
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
        dprintf("lw: + LW_PMSPPP_FALCON_ITFEN_CTXEN disabled\n");
    }

    if ( DRF_VAL( _PMSPPP, _FALCON_ITFEN, _MTHDEN, data32))
    {
        dprintf("lw: + LW_PMSPPP_FALCON_ITFEN_MTHDEN enabled\n");
    }
    else
    {
        dprintf("lw: + LW_PMSPPP_FALCON_ITFEN_MTHDEN disabled\n");
    }

    //check if falcon is hung (instr ptr)
    if ( pFalcon[indexGpu].falconTestPC(LW_FALCON_MSPPP_BASE, "PMSPPP") == LW_ERR_GENERIC )
    {
        dprintf("lw: Falcon instruction pointer is stuck or invalid\n");

        //TODO: treat falcon PC errors as warnings now, need to report as error
        addUnitErr("\t Warning: Falcon instruction pointer is stuck or invalid\n");
        //status = LW_ERR_GENERIC;
    }

    return status;
}
