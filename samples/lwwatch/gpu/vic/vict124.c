/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2013-2014 by LWPU Corporation.  All rights reserved.  All information
 * contained herein is proprietary and confidential to LWPU Corporation.  Any
 * use, reproduction, or disclosure without the written permission of LWPU
 * Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

//-----------------------------------------------------
//
// vict124.c - VIC routines
// 
//-----------------------------------------------------

#include "hal.h"
#include "tegrasys.h"
#include "vic.h"
#include "t12x/t124/dev_vic_pri.h"
#include "class/cla0b6.h"

#include "g_vic_private.h"     // (rmconfig)  implementation prototypes

// VIC Device specific register access macros
#define VIC_REG_RD32(reg)           (DEV_REG_RD32((reg - DRF_BASE(LW_PVIC)), "VIC", 0))
#define VIC_REG_WR32(reg,val)       (DEV_REG_WR32((reg - DRF_BASE(LW_PVIC)), val, "VIC", 0))
#define VIC_REG_RD_DRF(d,r,f)       (((VIC_REG_RD32(LW ## d ## r))>>DRF_SHIFT(LW ## d ## r ## f))&DRF_MASK(LW ## d ## r ## f)) 

dbg_vic vicMethodTable_t124[] =
{
    privInfo_vic(LWA0B6_VIDEO_COMPOSITOR_NOP),
    privInfo_vic(LWA0B6_VIDEO_COMPOSITOR_PM_TRIGGER),
    privInfo_vic(LWA0B6_VIDEO_COMPOSITOR_SET_APPLICATION_ID),
    privInfo_vic(LWA0B6_VIDEO_COMPOSITOR_SET_WATCHDOG_TIMER),
    privInfo_vic(LWA0B6_VIDEO_COMPOSITOR_SEMAPHORE_A),
    privInfo_vic(LWA0B6_VIDEO_COMPOSITOR_SEMAPHORE_B),
    privInfo_vic(LWA0B6_VIDEO_COMPOSITOR_SEMAPHORE_C),
    privInfo_vic(LWA0B6_VIDEO_COMPOSITOR_CTX_SAVE_AREA),
    privInfo_vic(LWA0B6_VIDEO_COMPOSITOR_CTX_SWITCH),
    privInfo_vic(LWA0B6_VIDEO_COMPOSITOR_EXELWTE),
    privInfo_vic(LWA0B6_VIDEO_COMPOSITOR_SEMAPHORE_D),
    privInfo_vic(LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE0_SLOT0_LUMA_OFFSET),
    privInfo_vic(LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE0_SLOT0_CHROMA_U_OFFSET),
    privInfo_vic(LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE0_SLOT0_CHROMA_V_OFFSET),
    privInfo_vic(LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE1_SLOT0_LUMA_OFFSET),
    privInfo_vic(LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE1_SLOT0_CHROMA_U_OFFSET),
    privInfo_vic(LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE1_SLOT0_CHROMA_V_OFFSET),
    privInfo_vic(LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE2_SLOT0_LUMA_OFFSET),
    privInfo_vic(LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE2_SLOT0_CHROMA_U_OFFSET),
    privInfo_vic(LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE2_SLOT0_CHROMA_V_OFFSET),
    privInfo_vic(LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE3_SLOT0_LUMA_OFFSET),
    privInfo_vic(LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE3_SLOT0_CHROMA_U_OFFSET),
    privInfo_vic(LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE3_SLOT0_CHROMA_V_OFFSET),
    privInfo_vic(LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE4_SLOT0_LUMA_OFFSET),
    privInfo_vic(LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE4_SLOT0_CHROMA_U_OFFSET),
    privInfo_vic(LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE4_SLOT0_CHROMA_V_OFFSET),
    privInfo_vic(LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE5_SLOT0_LUMA_OFFSET),
    privInfo_vic(LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE5_SLOT0_CHROMA_U_OFFSET),
    privInfo_vic(LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE5_SLOT0_CHROMA_V_OFFSET),
    privInfo_vic(LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE6_SLOT0_LUMA_OFFSET),
    privInfo_vic(LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE6_SLOT0_CHROMA_U_OFFSET),
    privInfo_vic(LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE6_SLOT0_CHROMA_V_OFFSET),
    privInfo_vic(LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE7_SLOT0_LUMA_OFFSET),
    privInfo_vic(LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE7_SLOT0_CHROMA_U_OFFSET),
    privInfo_vic(LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE7_SLOT0_CHROMA_V_OFFSET),
    privInfo_vic(LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE0_SLOT1_LUMA_OFFSET),
    privInfo_vic(LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE0_SLOT1_CHROMA_U_OFFSET),
    privInfo_vic(LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE0_SLOT1_CHROMA_V_OFFSET),
    privInfo_vic(LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE1_SLOT1_LUMA_OFFSET),
    privInfo_vic(LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE1_SLOT1_CHROMA_U_OFFSET),
    privInfo_vic(LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE1_SLOT1_CHROMA_V_OFFSET),
    privInfo_vic(LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE2_SLOT1_LUMA_OFFSET),
    privInfo_vic(LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE2_SLOT1_CHROMA_U_OFFSET),
    privInfo_vic(LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE2_SLOT1_CHROMA_V_OFFSET),
    privInfo_vic(LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE3_SLOT1_LUMA_OFFSET),
    privInfo_vic(LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE3_SLOT1_CHROMA_U_OFFSET),
    privInfo_vic(LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE3_SLOT1_CHROMA_V_OFFSET),
    privInfo_vic(LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE4_SLOT1_LUMA_OFFSET),
    privInfo_vic(LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE4_SLOT1_CHROMA_U_OFFSET),
    privInfo_vic(LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE4_SLOT1_CHROMA_V_OFFSET),
    privInfo_vic(LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE5_SLOT1_LUMA_OFFSET),
    privInfo_vic(LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE5_SLOT1_CHROMA_U_OFFSET),
    privInfo_vic(LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE5_SLOT1_CHROMA_V_OFFSET),
    privInfo_vic(LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE6_SLOT1_LUMA_OFFSET),
    privInfo_vic(LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE6_SLOT1_CHROMA_U_OFFSET),
    privInfo_vic(LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE6_SLOT1_CHROMA_V_OFFSET),
    privInfo_vic(LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE7_SLOT1_LUMA_OFFSET),
    privInfo_vic(LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE7_SLOT1_CHROMA_U_OFFSET),
    privInfo_vic(LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE7_SLOT1_CHROMA_V_OFFSET),
    privInfo_vic(LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE0_SLOT2_LUMA_OFFSET),
    privInfo_vic(LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE0_SLOT2_CHROMA_U_OFFSET),
    privInfo_vic(LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE0_SLOT2_CHROMA_V_OFFSET),
    privInfo_vic(LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE1_SLOT2_LUMA_OFFSET),
    privInfo_vic(LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE1_SLOT2_CHROMA_U_OFFSET),
    privInfo_vic(LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE1_SLOT2_CHROMA_V_OFFSET),
    privInfo_vic(LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE2_SLOT2_LUMA_OFFSET),
    privInfo_vic(LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE2_SLOT2_CHROMA_U_OFFSET),
    privInfo_vic(LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE2_SLOT2_CHROMA_V_OFFSET),
    privInfo_vic(LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE3_SLOT2_LUMA_OFFSET),
    privInfo_vic(LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE3_SLOT2_CHROMA_U_OFFSET),
    privInfo_vic(LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE3_SLOT2_CHROMA_V_OFFSET),
    privInfo_vic(LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE4_SLOT2_LUMA_OFFSET),
    privInfo_vic(LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE4_SLOT2_CHROMA_U_OFFSET),
    privInfo_vic(LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE4_SLOT2_CHROMA_V_OFFSET),
    privInfo_vic(LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE5_SLOT2_LUMA_OFFSET),
    privInfo_vic(LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE5_SLOT2_CHROMA_U_OFFSET),
    privInfo_vic(LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE5_SLOT2_CHROMA_V_OFFSET),
    privInfo_vic(LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE6_SLOT2_LUMA_OFFSET),
    privInfo_vic(LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE6_SLOT2_CHROMA_U_OFFSET),
    privInfo_vic(LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE6_SLOT2_CHROMA_V_OFFSET),
    privInfo_vic(LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE7_SLOT2_LUMA_OFFSET),
    privInfo_vic(LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE7_SLOT2_CHROMA_U_OFFSET),
    privInfo_vic(LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE7_SLOT2_CHROMA_V_OFFSET),
    privInfo_vic(LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE0_SLOT3_LUMA_OFFSET),
    privInfo_vic(LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE0_SLOT3_CHROMA_U_OFFSET),
    privInfo_vic(LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE0_SLOT3_CHROMA_V_OFFSET),
    privInfo_vic(LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE1_SLOT3_LUMA_OFFSET),
    privInfo_vic(LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE1_SLOT3_CHROMA_U_OFFSET),
    privInfo_vic(LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE1_SLOT3_CHROMA_V_OFFSET),
    privInfo_vic(LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE2_SLOT3_LUMA_OFFSET),
    privInfo_vic(LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE2_SLOT3_CHROMA_U_OFFSET),
    privInfo_vic(LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE2_SLOT3_CHROMA_V_OFFSET),
    privInfo_vic(LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE3_SLOT3_LUMA_OFFSET),
    privInfo_vic(LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE3_SLOT3_CHROMA_U_OFFSET),
    privInfo_vic(LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE3_SLOT3_CHROMA_V_OFFSET),
    privInfo_vic(LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE4_SLOT3_LUMA_OFFSET),
    privInfo_vic(LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE4_SLOT3_CHROMA_U_OFFSET),
    privInfo_vic(LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE4_SLOT3_CHROMA_V_OFFSET),
    privInfo_vic(LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE5_SLOT3_LUMA_OFFSET),
    privInfo_vic(LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE5_SLOT3_CHROMA_U_OFFSET),
    privInfo_vic(LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE5_SLOT3_CHROMA_V_OFFSET),
    privInfo_vic(LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE6_SLOT3_LUMA_OFFSET),
    privInfo_vic(LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE6_SLOT3_CHROMA_U_OFFSET),
    privInfo_vic(LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE6_SLOT3_CHROMA_V_OFFSET),
    privInfo_vic(LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE7_SLOT3_LUMA_OFFSET),
    privInfo_vic(LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE7_SLOT3_CHROMA_U_OFFSET),
    privInfo_vic(LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE7_SLOT3_CHROMA_V_OFFSET),
    privInfo_vic(LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE0_SLOT4_LUMA_OFFSET),
    privInfo_vic(LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE0_SLOT4_CHROMA_U_OFFSET),
    privInfo_vic(LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE0_SLOT4_CHROMA_V_OFFSET),
    privInfo_vic(LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE1_SLOT4_LUMA_OFFSET),
    privInfo_vic(LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE1_SLOT4_CHROMA_U_OFFSET),
    privInfo_vic(LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE1_SLOT4_CHROMA_V_OFFSET),
    privInfo_vic(LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE2_SLOT4_LUMA_OFFSET),
    privInfo_vic(LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE2_SLOT4_CHROMA_U_OFFSET),
    privInfo_vic(LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE2_SLOT4_CHROMA_V_OFFSET),
    privInfo_vic(LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE3_SLOT4_LUMA_OFFSET),
    privInfo_vic(LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE3_SLOT4_CHROMA_U_OFFSET),
    privInfo_vic(LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE3_SLOT4_CHROMA_V_OFFSET),
    privInfo_vic(LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE4_SLOT4_LUMA_OFFSET),
    privInfo_vic(LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE4_SLOT4_CHROMA_U_OFFSET),
    privInfo_vic(LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE4_SLOT4_CHROMA_V_OFFSET),
    privInfo_vic(LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE5_SLOT4_LUMA_OFFSET),
    privInfo_vic(LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE5_SLOT4_CHROMA_U_OFFSET),
    privInfo_vic(LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE5_SLOT4_CHROMA_V_OFFSET),
    privInfo_vic(LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE6_SLOT4_LUMA_OFFSET),
    privInfo_vic(LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE6_SLOT4_CHROMA_U_OFFSET),
    privInfo_vic(LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE6_SLOT4_CHROMA_V_OFFSET),
    privInfo_vic(LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE7_SLOT4_LUMA_OFFSET),
    privInfo_vic(LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE7_SLOT4_CHROMA_U_OFFSET),
    privInfo_vic(LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE7_SLOT4_CHROMA_V_OFFSET),
    privInfo_vic(LWA0B6_VIDEO_COMPOSITOR_SET_CONTROL_PARAMS),
    privInfo_vic(LWA0B6_VIDEO_COMPOSITOR_SET_CONTEXT_ID),
    privInfo_vic(LWA0B6_VIDEO_COMPOSITOR_SET_CONTEXT_ID_FOR_SLOT0),
    privInfo_vic(LWA0B6_VIDEO_COMPOSITOR_SET_CONTEXT_ID_FOR_SLOT1),
    privInfo_vic(LWA0B6_VIDEO_COMPOSITOR_SET_CONTEXT_ID_FOR_SLOT2),
    privInfo_vic(LWA0B6_VIDEO_COMPOSITOR_SET_CONTEXT_ID_FOR_SLOT3),
    privInfo_vic(LWA0B6_VIDEO_COMPOSITOR_SET_CONTEXT_ID_FOR_SLOT4),
    privInfo_vic(LWA0B6_VIDEO_COMPOSITOR_SET_FCE_UCODE_SIZE),
    privInfo_vic(LWA0B6_VIDEO_COMPOSITOR_SET_CONFIG_STRUCT_OFFSET),
    privInfo_vic(LWA0B6_VIDEO_COMPOSITOR_SET_PALETTE_OFFSET),
    privInfo_vic(LWA0B6_VIDEO_COMPOSITOR_SET_HIST_OFFSET),
    privInfo_vic(LWA0B6_VIDEO_COMPOSITOR_SET_FCE_UCODE_OFFSET),
    privInfo_vic(LWA0B6_VIDEO_COMPOSITOR_SET_OUTPUT_SURFACE_LUMA_OFFSET),
    privInfo_vic(LWA0B6_VIDEO_COMPOSITOR_SET_OUTPUT_SURFACE_CHROMA_U_OFFSET),
    privInfo_vic(LWA0B6_VIDEO_COMPOSITOR_SET_OUTPUT_SURFACE_CHROMA_V_OFFSET),
    privInfo_vic(LWA0B6_VIDEO_COMPOSITOR_SET_PICTURE_INDEX),
    privInfo_vic(LWA0B6_VIDEO_COMPOSITOR_PM_TRIGGER_END),
    privInfo_vic(0),
};

dbg_vic vicPrivReg_t124[] =
{
    privInfo_vic(LW_PVIC_FALCON_IRQSSET),
    privInfo_vic(LW_PVIC_FALCON_IRQSCLR),
    privInfo_vic(LW_PVIC_FALCON_IRQSTAT),
    privInfo_vic(LW_PVIC_FALCON_IRQMSET),
    privInfo_vic(LW_PVIC_FALCON_IRQMCLR),
    privInfo_vic(LW_PVIC_FALCON_IRQMASK),
    privInfo_vic(LW_PVIC_FALCON_PTIMER0),
    privInfo_vic(LW_PVIC_FALCON_PTIMER1),
    privInfo_vic(LW_PVIC_FALCON_MTHDDATA),
    privInfo_vic(LW_PVIC_FALCON_MTHDCOUNT),
    privInfo_vic(LW_PVIC_FALCON_MTHDPOP),
    privInfo_vic(LW_PVIC_FALCON_CTXACK),
    privInfo_vic(LW_PVIC_FALCON_IDLESTATE),
    privInfo_vic(LW_PVIC_FALCON_FHSTATE),
    privInfo_vic(LW_PVIC_FALCON_HWCFG),
    privInfo_vic(LW_PVIC_FALCON_HWCFG1),
    privInfo_vic(LW_PVIC_FALCON_IMSTAT),
    privInfo_vic(LW_PVIC_FALCON_TRACEPC),
    privInfo_vic(LW_PVIC_FALCON_ICD_RDATA),
    privInfo_vic(LW_PVIC_FC_COMPOSE),
    privInfo_vic(LW_PVIC_FC_DEBUG0),
    privInfo_vic(LW_PVIC_FC_DEBUG1),
    privInfo_vic(LW_PVIC_FC_DEBUG2),
    privInfo_vic(LW_PVIC_FC_DEBUG3),
    privInfo_vic(LW_PVIC_SL_DEBUG),
    privInfo_vic(LW_PVIC_SC_DEBUG0),
    privInfo_vic(LW_PVIC_SC_DEBUG1),
    privInfo_vic(LW_PVIC_SC_DEBUG2),
    privInfo_vic(LW_PVIC_SC_DEBUG3),
    privInfo_vic(LW_PVIC_IT_TN_DEBUG),
    privInfo_vic(LW_PVIC_IT_CD_DEBUG),
    privInfo_vic(LW_PVIC_CC_CTRL),
    privInfo_vic(LW_PVIC_CC_DEBUG),
    privInfo_vic(LW_PVIC_BL_DEBUG),
    privInfo_vic(LW_PVIC_MISC_INTR_STATUS),
    privInfo_vic(LW_PVIC_MISC_CAP0),
    privInfo_vic(LW_PVIC_FC_FCE_CTRL),
    privInfo_vic(LW_PVIC_FC_FCE_STATUS),
    privInfo_vic(LW_PVIC_YS_DEBUG),
    privInfo_vic(LW_PVIC_XS_DEBUG),
    privInfo_vic(LW_PVIC_TFBIF_DBG_STAT0),
    privInfo_vic(LW_PVIC_TFBIF_DBG_STAT1),
    privInfo_vic(LW_PVIC_TFBIF_DBG_RDCOUNT_LO),
    privInfo_vic(LW_PVIC_TFBIF_DBG_RDCOUNT_HI),
    privInfo_vic(LW_PVIC_TFBIF_DBG_WRCOUNT_LO),
    privInfo_vic(LW_PVIC_TFBIF_DBG_WRCOUNT_HI),
    privInfo_vic(LW_PVIC_TFBIF_DBG_R32COUNT),
    privInfo_vic(LW_PVIC_TFBIF_DBG_R64COUNT),
    privInfo_vic(LW_PVIC_TFBIF_DBG_R128COUNT),
    privInfo_vic(0),
};



//-----------------------------------------------------
// vicIsSupported_T124
//-----------------------------------------------------
BOOL vicIsSupported_T124( LwU32 indexGpu )
{
    pVicPrivReg = vicPrivReg_t124;
    pVicMethodTable = vicMethodTable_t124;
    return TRUE;
}

//-----------------------------------------------------
// vicDumpImem_T124 - Dumps VIC instruction memory
//-----------------------------------------------------
LW_STATUS vicDumpImem_T124 (LwU32 indexGpu)
{
    LW_STATUS status        = LW_OK;
    LwU32 addressImem   = LW_PVIC_FALCON_IMEMD(0);
    LwU32 address2Imem  = LW_PVIC_FALCON_IMEMC(0);
    LwU32 address2Imemt = LW_PVIC_FALCON_IMEMT(0);
    LwU32 blk           = 0;
    LwU32 imemSize;
    LwU32 u;
    LwU32 i;

    imemSize = (VIC_REG_RD_DRF(_PVIC_FALCON, _HWCFG, _IMEM_SIZE) << 8);

    dprintf("\n");
    dprintf("lw: -- Gpu %u VIC IMEM -- \n", indexGpu);    
    dprintf("lw: -- Gpu %u VIC IMEM SIZE =  0x%08x-- \n", indexGpu, imemSize);
    dprintf("\nADDR: 03....00 07....04 0B....08 0F....0C 13....10 17....14 1B....18 1F....1C");
    dprintf("\n-----------------------------------------------------------------------------");
    for (u = 0; u < (imemSize + 3) / 4; u++)
    {
        if ((u % 64) == 0)
        {
            VIC_REG_WR32(address2Imemt, blk++);
        }
        i = (u << (0 ? LW_PVIC_FALCON_IMEMC_OFFS));
        VIC_REG_WR32(address2Imem, i);
        if (( u % 8) == 0)
        {
            dprintf("\n%04X: ", 4*u);
        }
        dprintf("%08X ",  VIC_REG_RD32(addressImem));
    }
    return status;  
}

//-----------------------------------------------------
// vicDumpDmem_T124 - Dumps VIC data memory
//-----------------------------------------------------
LW_STATUS vicDumpDmem_T124 (LwU32 indexGpu)
{
    LW_STATUS status                      = LW_OK;
    LwU32 comMthd[CMNMETHODARRAYSIZE] = {0};
    LwU32 appMthd[APPMETHODARRAYSIZE] = {0};
    LwU32 dmemSize;
    LwU32 address, address2, u, i, comMthdOffs = 0, appMthdOffs = 0, classNum;
    LwU32 methodIdx;

    dmemSize = (VIC_REG_RD_DRF(_PVIC_FALCON, _HWCFG, _DMEM_SIZE)<<8) ;

    address     = LW_PVIC_FALCON_DMEMD(0);
    address2    = LW_PVIC_FALCON_DMEMC(0);
    classNum    = 0xA0B6;

    dprintf("\n");
    dprintf("lw: -- Gpu %u VIC DMEM -- \n", indexGpu);
    dprintf("lw: -- Gpu %u VIC DMEM SIZE =  0x%08x-- \n", indexGpu, dmemSize);
    dprintf("\nADDR: 03....00 07....04 0B....08 0F....0C 13....10 17....14 1B....18 1F....1C");
    dprintf("\n-----------------------------------------------------------------------------");
    

    for(u = 0; u < (dmemSize + 3) / 4; u++)
    {
        i = (u << (0 ? LW_PVIC_FALCON_IMEMC_OFFS));
        VIC_REG_WR32(address2, i);
        if((u % 8) == 0)
        {
            dprintf("\n%04X: ", 4*u);
        }
        dprintf("%08X ",  VIC_REG_RD32(address));
    }

    // get methods offset are in the DWORD#3 in dmem
    u = (3 << (0 ? LW_PVIC_FALCON_IMEMC_OFFS));
    VIC_REG_WR32(address2,u);
    comMthdOffs = (VIC_REG_RD32(address)) >> 2;
    appMthdOffs = comMthdOffs + 16;

    for(u = 0; u < CMNMETHODARRAYSIZE; u++)
    {
        i = ((u + comMthdOffs) << (0 ? LW_PVIC_FALCON_IMEMC_OFFS));
        VIC_REG_WR32(address2, i);
        comMthd[u] = VIC_REG_RD32(address);
        i = ((u + appMthdOffs) << (0 ? LW_PVIC_FALCON_IMEMC_OFFS));
        VIC_REG_WR32(address2, i);
        appMthd[u] = VIC_REG_RD32(address);
    }

    dprintf("\n\n-----------------------------------------------------------------------\n");
    dprintf("%4s, %8s,    %4s, %8s,    %4s, %8s,    %4s, %8s\n", "Mthd", "Data", "Mthd", "Data", "Mthd", "Data", "Mthd", "Data");
    dprintf("[COMMON METHODS]\n");
    for (u=0; u<CMNMETHODARRAYSIZE; u+=4)
    {
        dprintf("%04X: %08X,    %04X: %08X,    %04X: %08X,    %04X: %08X\n",
        CMNMETHODBASE+4*u, comMthd[u], CMNMETHODBASE+4*(u+1), comMthd[u+1], 
        CMNMETHODBASE+4*(u+2), comMthd[u+2], CMNMETHODBASE+4*(u+3), comMthd[u+3]);
    }
    dprintf("\n");
    dprintf("\n[APP METHODS]\n");
    for (u=0; u<APPMETHODARRAYSIZE; u+=4)
    {
        dprintf("%04X: %08X,    %04X: %08X,    %04X: %08X,    %04X: %08X\n",
        APPMETHODBASE+4*u, appMthd[u], APPMETHODBASE+4*(u+1), appMthd[u+1],
        APPMETHODBASE+4*(u+2), appMthd[u+2], APPMETHODBASE+4*(u+3), appMthd[u+3]);
    }

    dprintf("\n[COMMON METHODS]\n");
    for(u=0;u<16;u++)
    {
        for(methodIdx=0;;methodIdx++)
        {
            if(pVicMethodTable[methodIdx].m_id == (CMNMETHODBASE+4*u))
            {
                vicPrintMethodData_t124(60, pVicMethodTable[methodIdx].m_tag, pVicMethodTable[methodIdx].m_id, comMthd[u]);
                break;
            }
            else if (pVicMethodTable[methodIdx].m_id == 0)
            {
                break;
            }
        }
    }
    dprintf("\n");

    // app methods
    dprintf("\n[APP METHODS]\n");
    for(u=0;u<16;u++)
    {
        for(methodIdx=0;;methodIdx++)
        {
            if(pVicMethodTable[methodIdx].m_id == (APPMETHODBASE+4*u))
            {
                vicPrintMethodData_t124(60, pVicMethodTable[methodIdx].m_tag, pVicMethodTable[methodIdx].m_id, appMthd[u]);
                break;
            }
            else if (pVicMethodTable[methodIdx].m_id == 0)
            {
                break;
            }
        }
    }
    dprintf("\n");
    return status;  
}

//-----------------------------------------------------
// vicTestState_T124 - Test basic vic state
//-----------------------------------------------------
LW_STATUS vicTestState_T124(LwU32 indexGpu)
{
    LwU32              regIntr;
    LwU32              regIntrEn;
    LwU32              data32;
    LwU32              vicBaseAddress;
    LW_STATUS               status = LW_OK;
    PDEVICE_RELOCATION pDev   = NULL;

    pDev     = tegrasysGetDeviceReloc(&TegraSysObj[indexGpu], "VIC", 0);
    assert(pDev);
    vicBaseAddress = (LwU32)pDev->start;
    
    //check falcon interrupts
    regIntr = VIC_REG_RD32(LW_PVIC_FALCON_IRQSTAT);
    regIntrEn = VIC_REG_RD32(LW_PVIC_FALCON_IRQMASK);
    regIntr &= regIntrEn;

    if ( !DRF_VAL(_PVIC, _FALCON_IRQMASK, _GPTMR, regIntrEn))
        dprintf("lw: LW_PVIC_FALCON_IRQMASK_GPTMR disabled\n");

    if ( !DRF_VAL(_PVIC, _FALCON_IRQMASK, _WDTMR, regIntrEn))
        dprintf("lw: LW_PVIC_FALCON_IRQMASK_WDTMR disabled\n");

    if ( !DRF_VAL(_PVIC, _FALCON_IRQMASK, _MTHD, regIntrEn))
        dprintf("lw: LW_PVIC_FALCON_IRQMASK_MTHD disabled\n");

    if ( !DRF_VAL(_PVIC, _FALCON_IRQMASK, _CTXSW, regIntrEn))
        dprintf("lw: LW_PVIC_FALCON_IRQMASK_CTXSW disabled\n");

    if ( !DRF_VAL(_PVIC, _FALCON_IRQMASK, _HALT, regIntrEn))
        dprintf("lw: LW_PVIC_FALCON_IRQMASK_HALT disabled\n");

    if ( !DRF_VAL(_PVIC, _FALCON_IRQMASK, _EXTERR, regIntrEn))
        dprintf("lw: LW_PVIC_FALCON_IRQMASK_EXTERR disabled\n");

    if ( !DRF_VAL(_PVIC, _FALCON_IRQMASK, _SWGEN0, regIntrEn))
        dprintf("lw: LW_PVIC_FALCON_IRQMASK_SWGEN0 disabled\n");

    if ( !DRF_VAL(_PVIC, _FALCON_IRQMASK, _SWGEN1, regIntrEn))
        dprintf("lw: LW_PVIC_FALCON_IRQMASK_SWGEN1 disabled\n");

    //if any interrupt pending, set error
    if (regIntr != 0)
    {
        addUnitErr("\t VIC interrupts are pending\n");
        status = LW_ERR_GENERIC;
    }

    if ( DRF_VAL( _PVIC,_FALCON_IRQSTAT, _GPTMR, regIntr))
    {
        dprintf("lw: LW_PVIC_FALCON_IRQSTAT_GPTMR pending\n");
        dprintf("lw: LW_PVIC_FALCON_GPTMRINT:    0x%08x\n", 
                VIC_REG_RD32(LW_PVIC_FALCON_GPTMRINT) );
        dprintf("lw: LW_PVIC_FALCON_GPTMRVAL:    0x%08x\n", 
            VIC_REG_RD32(LW_PVIC_FALCON_GPTMRVAL) );
        
    }
    
    if ( DRF_VAL( _PVIC,_FALCON_IRQSTAT, _WDTMR, regIntr))
    {
        dprintf("lw: LW_PVIC_FALCON_IRQSTAT_WDTMR pending\n");
    }

    if ( DRF_VAL( _PVIC,_FALCON_IRQSTAT, _MTHD, regIntr))
    {
        dprintf("lw: LW_PVIC_FALCON_IRQSTAT_MTHD pending\n");

        dprintf("lw: LW_PVIC_FALCON_MTHDDATA_DATA:    0x%08x\n", 
            VIC_REG_RD32(LW_PVIC_FALCON_MTHDDATA) );
        
        data32 = VIC_REG_RD32(LW_PVIC_FALCON_MTHDID);
        dprintf("lw: LW_PVIC_FALCON_MTHDID_ID:    0x%08x\n", 
           DRF_VAL( _PVIC,_FALCON_MTHDID, _ID, data32)  );
        dprintf("lw: LW_PVIC_FALCON_MTHDID_SUBCH:    0x%08x\n", 
           DRF_VAL( _PVIC,_FALCON_MTHDID, _SUBCH, data32)  );
        dprintf("lw: LW_PVIC_FALCON_MTHDID_PRIV:    0x%08x\n", 
           DRF_VAL( _PVIC,_FALCON_MTHDID, _PRIV, data32)  );
    }
    
    if ( DRF_VAL( _PVIC,_FALCON_IRQSTAT, _CTXSW, regIntr))
    {
        dprintf("lw: LW_PVIC_FALCON_IRQSTAT_CTXSW pending\n");
    }
    
    if ( DRF_VAL( _PVIC,_FALCON_IRQSTAT, _HALT, regIntr))
    {
        dprintf("lw: LW_PVIC_FALCON_IRQSTAT_HALT pending\n");
    }
    
    if ( DRF_VAL( _PVIC,_FALCON_IRQSTAT, _EXTERR, regIntr))
    {
        dprintf("lw: LW_PVIC_FALCON_IRQSTAT_EXTERR pending\n");
    }
    
    if ( DRF_VAL( _PVIC,_FALCON_IRQSTAT, _SWGEN0, regIntr))
    {
        dprintf("lw: LW_PVIC_FALCON_IRQSTAT_SWGEN0 pending\n");

        pFalcon[indexGpu].falconPrintMailbox(vicBaseAddress);
    }

    if ( DRF_VAL( _PVIC,_FALCON_IRQSTAT, _SWGEN1, regIntr))
    {
        dprintf("lw: LW_PVIC_FALCON_IRQSTAT_SWGEN1 pending\n");
    }

     //
    //print falcon states
    //Bit |  Signal meaning
    //0      FALCON busy
    //

    data32 = VIC_REG_RD32(LW_PVIC_FALCON_IDLESTATE);

    if ( DRF_VAL( _PVIC, _FALCON_IDLESTATE, _FALCON_BUSY, data32))
    {
        dprintf("lw: + LW_PVIC_FALCON_IDLESTATE_FALCON_BUSY\n");
        addUnitErr("\t LW_PVIC_FALCON_IDLESTATE_FALCON_BUSY\n");
        status = LW_ERR_GENERIC;
    }

  
    data32 = VIC_REG_RD32(LW_PVIC_FALCON_FHSTATE);
 
    if ( DRF_VAL( _PVIC, _FALCON_FHSTATE, _FALCON_HALTED, data32))
    {
        dprintf("lw: + LW_PVIC_FALCON_FHSTATE_FALCON_HALTED\n");
        addUnitErr("\t LW_PVIC_FALCON_FHSTATE_FALCON_HALTED\n");
        status = LW_ERR_GENERIC;
    }
    
    if ( DRF_VAL( _PVIC, _FALCON_FHSTATE, _ENGINE_FAULTED, data32))
    {
        dprintf("lw: + LW_PVIC_FALCON_FHSTATE_ENGINE_FAULTED\n");
        addUnitErr("\t LW_PVIC_FALCON_FHSTATE_ENGINE_FAULTED\n");
        status = LW_ERR_GENERIC;
    }
    
    if ( DRF_VAL( _PVIC, _FALCON_FHSTATE, _STALL_REQ, data32))
    {
        dprintf("lw: + LW_PVIC_FALCON_FHSTATE_STALL_REQ\n");
        addUnitErr("\t LW_PVIC_FALCON_FHSTATE_STALL_REQ\n");
        status = LW_ERR_GENERIC;
    }

    //print falcon ctl regs
    data32 = VIC_REG_RD32(LW_PVIC_FALCON_ENGCTL);
    
    if ( DRF_VAL( _PVIC, _FALCON_ENGCTL, _ILW_CONTEXT, data32))
    {
        dprintf("lw: + LW_PVIC_FALCON_ENGCTL_ILW_CONTEXT\n");
        addUnitErr("\t LW_PVIC_FALCON_ENGCTL_ILW_CONTEXT\n");
        status = LW_ERR_GENERIC;
    }

    if ( DRF_VAL( _PVIC, _FALCON_ENGCTL, _STALLREQ, data32))
    {
        dprintf("lw: + LW_PVIC_FALCON_ENGCTL_STALLREQ\n");
        addUnitErr("\t LW_PVIC_FALCON_ENGCTL_STALLREQ\n");
        status = LW_ERR_GENERIC;
    }

    data32 = VIC_REG_RD32(LW_PVIC_FALCON_CPUCTL);

    if ( DRF_VAL( _PVIC, _FALCON_CPUCTL, _IILWAL, data32))
    {
        dprintf("lw: + LW_PVIC_FALCON_CPUCTL_IILWAL\n");
        addUnitErr("\t LW_PVIC_FALCON_CPUCTL_IILWAL\n");
        status = LW_ERR_GENERIC;
    }

    if ( DRF_VAL( _PVIC, _FALCON_CPUCTL, _HALTED, data32))
    {
        dprintf("lw: + LW_PVIC_FALCON_CPUCTL_HALTED\n");
        addUnitErr("\t LW_PVIC_FALCON_CPUCTL_HALTED\n");
        status = LW_ERR_GENERIC;
    }

    if ( DRF_VAL( _PVIC, _FALCON_CPUCTL, _STOPPED, data32))
    {
        dprintf("lw: + LW_PVIC_FALCON_CPUCTL_STOPPED\n");
        addUnitErr("\t Warning: LW_PVIC_FALCON_CPUCTL_STOPPED\n");
        //status = LW_ERR_GENERIC;
    }

    // state of mthd/ctx interface 
    data32 = VIC_REG_RD32(LW_PVIC_FALCON_ITFEN);

    if (DRF_VAL( _PVIC, _FALCON_ITFEN, _CTXEN, data32))
    {
        dprintf("lw: + LW_PVIC_FALCON_ITFEN_CTXEN enabled\n");
             
        if (pFalcon[indexGpu].falconTestCtxState(vicBaseAddress, "PVIC") == LW_ERR_GENERIC)
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
        dprintf("lw: + LW_PVIC_FALCON_ITFEN_CTXEN disabled\n");
    }

    if ( DRF_VAL( _PVIC, _FALCON_ITFEN, _MTHDEN, data32))
    {
        dprintf("lw: + LW_PVIC_FALCON_ITFEN_MTHDEN enabled\n");
    }
    else
    {
        dprintf("lw: + LW_PVIC_FALCON_ITFEN_MTHDEN disabled\n");
    }

    //check if falcon is hung (instr ptr)
    if ( pFalcon[indexGpu].falconTestPC(vicBaseAddress, "PVIC") == LW_ERR_GENERIC )
    {
        dprintf("lw: Falcon instruction pointer is stuck or invalid\n");
        
        //TODO: treat falcon PC errors as warnings now, need to report as error
        addUnitErr("\t Warning: Falcon instruction pointer is stuck or invalid\n");
        //status = LW_ERR_GENERIC;
    }

    return status;  
}

//-----------------------------------------------------
// vicPrintPriv_T124
//-----------------------------------------------------
void vicPrintPriv_T124(LwU32 clmn, char *tag, LwU32 id)
{
    size_t len = strlen(tag);
    
    dprintf("lw: %s",tag);

    if((len > 0) && (len < (clmn + 4)))
    {
        LwU32 i;
        for(i=0;i<clmn-len;i++)
        {
            dprintf(" ");
        }
    }
    dprintf("(0x%08X)  = 0x%08X\n", id, VIC_REG_RD32(id));
}

//-----------------------------------------------------
// vicDumpPriv_T124 - Dumps VIC priv reg space
//-----------------------------------------------------
LW_STATUS vicDumpPriv_T124(LwU32 indexGpu)
{
    LwU32 u;

    dprintf("lw:\n");
    dprintf("lw: -- Gpu %u VIC priv registers -- \n", indexGpu);
    dprintf("lw:\n");

    for(u = 0; ; u++)
    {
        if(pVicPrivReg[u].m_id==0)
        {
            break;
        }
        
        pVic[indexGpu].vicPrintPriv(40, pVicPrivReg[u].m_tag,pVicPrivReg[u].m_id);
    }
    return LW_OK; 
}

//--------------------------------------------------------
// vicDisplayHwcfg_T124 - Display VIC HW config
//--------------------------------------------------------
LW_STATUS vicDisplayHwcfg_T124(LwU32 indexGpu)
{
    LwU32 hwcfg, hwcfg1;

    dprintf("lw:\n");
    dprintf("lw: -- Gpu %u VIC HWCFG -- \n", indexGpu);
    dprintf("lw:\n");

    hwcfg  = VIC_REG_RD32(LW_PVIC_FALCON_HWCFG);
    dprintf("lw: LW_PVIC_FALCON_HWCFG:  0x%08x\n", hwcfg); 
    dprintf("lw:\n");
    dprintf("lw:  IMEM_SIZE:        0x%08X (or 0x%08X bytes)\n",
            DRF_VAL(_PVIC, _FALCON_HWCFG, _IMEM_SIZE, hwcfg),
            DRF_VAL(_PVIC, _FALCON_HWCFG, _IMEM_SIZE, hwcfg)<<8); 
    dprintf("lw:  DMEM_SIZE:        0x%08X (or 0x%08X bytes)\n",
            DRF_VAL(_PVIC, _FALCON_HWCFG, _DMEM_SIZE, hwcfg), 
            DRF_VAL(_PVIC, _FALCON_HWCFG, _DMEM_SIZE, hwcfg)<<8); 
    dprintf("lw:  METHODFIFO_DEPTH: 0x%08X\n", DRF_VAL(_PVIC, _FALCON_HWCFG, _METHODFIFO_DEPTH, hwcfg)); 
    dprintf("lw:  DMAQUEUE_DEPTH:   0x%08X\n", DRF_VAL(_PVIC, _FALCON_HWCFG, _DMAQUEUE_DEPTH, hwcfg)); 

    dprintf("lw:\n");

    hwcfg1 = VIC_REG_RD32(LW_PVIC_FALCON_HWCFG1);
    dprintf("lw: LW_PVIC_FALCON_HWCFG1: 0x%08x\n", hwcfg1); 
    dprintf("lw:\n");
    dprintf("lw:  CORE_REV:         0x%08X\n", DRF_VAL(_PVIC, _FALCON_HWCFG1, _CORE_REV, hwcfg1)); 
    dprintf("lw:  SELWRITY_MODEL:   0x%08X\n", DRF_VAL(_PVIC, _FALCON_HWCFG1, _SELWRITY_MODEL, hwcfg1)); 
    dprintf("lw:  IMEM_PORTS:       0x%08X\n", DRF_VAL(_PVIC, _FALCON_HWCFG1, _IMEM_PORTS, hwcfg1)); 
    dprintf("lw:  DMEM_PORTS:       0x%08X\n", DRF_VAL(_PVIC, _FALCON_HWCFG1, _DMEM_PORTS, hwcfg1)); 
    dprintf("lw:  TAG_WIDTH:        0x%08X\n", DRF_VAL(_PVIC, _FALCON_HWCFG1, _TAG_WIDTH, hwcfg1)); 

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
LW_STATUS  vicDisplayFlcnSPR_T124(LwU32 indexGpu)
{
    dprintf("lw:\n");
    dprintf("lw: -- Gpu %u VIC Special Purpose Registers -- \n", indexGpu);
    dprintf("lw:\n");

    VIC_REG_WR32(LW_PVIC_FALCON_ICD_CMD, 0x1008);
    dprintf("lw: VIC IV0 :    0x%08x\n", VIC_REG_RD32(LW_PVIC_FALCON_ICD_RDATA)); 
    VIC_REG_WR32(LW_PVIC_FALCON_ICD_CMD, 0x1108);
    dprintf("lw: VIC IV1 :    0x%08x\n", VIC_REG_RD32(LW_PVIC_FALCON_ICD_RDATA)); 
    VIC_REG_WR32(LW_PVIC_FALCON_ICD_CMD, 0x1308);
    dprintf("lw: VIC EV  :    0x%08x\n", VIC_REG_RD32(LW_PVIC_FALCON_ICD_RDATA)); 
    VIC_REG_WR32(LW_PVIC_FALCON_ICD_CMD, 0x1408);
    dprintf("lw: VIC SP  :    0x%08x\n", VIC_REG_RD32(LW_PVIC_FALCON_ICD_RDATA)); 
    VIC_REG_WR32(LW_PVIC_FALCON_ICD_CMD, 0x1508);
    dprintf("lw: VIC PC  :    0x%08x\n", VIC_REG_RD32(LW_PVIC_FALCON_ICD_RDATA)); 
    VIC_REG_WR32(LW_PVIC_FALCON_ICD_CMD, 0x1608);
    dprintf("lw: VIC IMB :    0x%08x\n", VIC_REG_RD32(LW_PVIC_FALCON_ICD_RDATA)); 
    VIC_REG_WR32(LW_PVIC_FALCON_ICD_CMD, 0x1708);
    dprintf("lw: VIC DMB :    0x%08x\n", VIC_REG_RD32(LW_PVIC_FALCON_ICD_RDATA)); 
    VIC_REG_WR32(LW_PVIC_FALCON_ICD_CMD, 0x1808);
    dprintf("lw: VIC CSW :    0x%08x\n", VIC_REG_RD32(LW_PVIC_FALCON_ICD_RDATA)); 
    dprintf("lw:\n\n");

    return LW_OK; 
}

//-----------------------------------------------------
// vicPrintMethodData_t124
//-----------------------------------------------------
void vicPrintMethodData_t124(LwU32 clmn, char *tag, LwU32 method, LwU32 data)
{
    size_t len = strlen(tag);
    
    dprintf("lw: %s", tag);

    if((len > 0) && (len < (clmn + 4)))
    {
        LwU32 i;
        for(i = 0; i < clmn - len; i++)
        {
            dprintf(" ");
        }
    }
    dprintf("(0x%08X)  = 0x%08X\n", method, data);
}
