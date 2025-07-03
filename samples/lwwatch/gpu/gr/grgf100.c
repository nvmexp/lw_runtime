/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2003-2018 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

//*****************************************************
//
// lwwatch WinDbg Extension
// grgf100.c
//
//*****************************************************

//
// includes
//
#include <time.h>

#include "chip.h"
// HW specific files are in //sw/dev/gpu_drv/chips_a/drivers/common/inc/hwref/<family>/<chip>
#include "fermi/gf100/dev_graphics_nobundle.h"
#include "fermi/gf100/dev_ctxsw_prog.h"
#include "fermi/gf100/dev_fifo.h"
#include "fermi/gf100/dev_xbar.h"
#include "fermi/gf100/dev_fb.h"
#include "fermi/gf100/dev_master.h" // For PMC_BOOT_0
#include "fermi/gf100/hwproject.h"
#include "fermi/gf100/dev_pri_ringmaster.h"
#include "fermi/gf100/dev_pri_ringstation_sys.h"
#include "fermi/gf100/dev_top.h"

#include "inst.h"
#include "print.h"
#include "gpuanalyze.h"
#include "gr.h"

#include "g_gr_private.h"       // (rmconfig) implementation prototypes

#define ISO_CTRL_TEST_VAL_MAX 13

void grPrintGpuUnitStatus_GF100(LwU32 status, char *regName)
{
    switch (status) {
        case EMPTY:
            DPRINTF_FIELD( regName, QUOTE_ME(EMPTY), status );
            break;
        case ACTIVE:
            DPRINTF_FIELD( regName, QUOTE_ME(ACTIVE), status );
            break;
        case PAUSED:
            DPRINTF_FIELD( regName, QUOTE_ME(PAUSED), status );
            break;
        case QUIESCENT:
            DPRINTF_FIELD( regName, QUOTE_ME(QUIESCENT), status );
            break;
        case UNKNOWN:
            DPRINTF_FIELD( regName, QUOTE_ME(UNKNOWN), status );
            break;
        case STALLED:
            DPRINTF_FIELD( regName, QUOTE_ME(STALLED), status );
            break;
        case FAULTED:
            DPRINTF_FIELD( regName, QUOTE_ME(FAULTED), status );
            break;
        case HALTED:
            DPRINTF_FIELD( regName, QUOTE_ME(HALTED), status );
            break;
        default:
            DPRINTF_FIELD( regName, QUOTE_ME(UNKNOWN), status );
    }
}

//
// The input parameter gpc here is the local gpc id when SMC is enabled.
// When SMC is disabled or pre-AMPERE, gpc is the logical gpc
// (which is equal to the physical gpc).
//
LwU32 grGetNumTpcForGpc_GF100(LwU32 gpc, LwU32 grIdx)
{
    LwU32 tpc;
    LwU32 reg;
    const LwU32 num_gpcs = pGr[indexGpu].grGetNumGpcs( grIdx );

    if (gpc >= num_gpcs)
    {
        dprintf("**ERROR: Illegal GPC num: %d (%s)\n", gpc, __FUNCTION__);
        return 0;
    }

    reg = GPU_REG_RD32(GPC_REG_ADDR(GPCCS_FS_GPC, gpc));
    tpc = DRF_VAL(_PGRAPH_PRI, _GPC0_GPCCS_FS_GPC, _NUM_AVAILABLE_TPCS, reg);
    return tpc;
}

LwU32 grGetNumTpcPerGpc_GF100(void)
{
   LwU32  val;
   val = GPU_REG_RD32(LW_PTOP_SCAL_NUM_TPC_PER_GPC);
   return DRF_VAL(_PTOP_SCAL, _NUM_TPC_PER_GPC, _VALUE, val);
}

LwU32 grGetNumSmPerTpc_GF100(void)
{
    return 1; // no define in hwproject.h for pre volta, always 1
}

LwU32 grGetNumGpcs_GF100( LwU32 grIdx )
{
    LwU32  val;
    val = GPU_REG_RD32(LW_PTOP_SCAL_NUM_GPCS);
    return DRF_VAL(_PTOP_SCAL, _NUM_GPCS, _VALUE, val);
}

LwU32 grGetMaxTpcPerGpc_GF100(void)
{
    return LW_SCAL_LITTER_NUM_TPC_PER_GPC;
}

LwU32 grGetMaxGpc_GF100(void)
{
    return LW_SCAL_LITTER_NUM_GPCS;
}

LwU32 grGetMaxFbp_GF100(void)
{
    return LW_SCAL_LITTER_NUM_FBPS;
}

//----------------------------------------------------
// grGetInfo_GF100
//-----------------------------------------------------
void grGetInfo_GF100(void)
{
    dprintf("**ERROR %s - Not supported...  Suggest: grstatus\n", __FUNCTION__);
}

//-----------------------------------------------------
// grGetLwrrentAndPendingInfo_GF100
// + returns 0x0 in ptrs if not valid
//-----------------------------------------------------
void grGetLwrrentAndPendingInfo_GF100(LwU32 *lwrCtx, LwU32 *pendCtx)
{
    LwU32 ctx;

    if (lwrCtx)                     // Get current unless param is null
    {
        ctx  = GPU_REG_RD32(LW_PGRAPH_PRI_FECS_LWRRENT_CTX);
        if (DRF_VAL(_PGRAPH, _PRI_FECS_LWRRENT_CTX, _VALID, ctx)
            == LW_PGRAPH_PRI_FECS_LWRRENT_CTX_VALID_TRUE)
        {
            *lwrCtx = DRF_VAL(_PGRAPH, _PRI_FECS_LWRRENT_CTX, _PTR, ctx);
        }
        else
        {
            *lwrCtx = 0;            // Zero if not valid
        }
    }

    if (pendCtx)                    // Get pending unless param is null
    {
        ctx = GPU_REG_RD32(LW_PGRAPH_PRI_FECS_NEW_CTX);
        if (DRF_VAL(_PGRAPH, _PRI_FECS_NEW_CTX, _VALID, ctx)
            == LW_PGRAPH_PRI_FECS_NEW_CTX_VALID_TRUE)
        {
            *pendCtx = DRF_VAL(_PGRAPH, _PRI_FECS_NEW_CTX, _PTR, ctx);
        }
        else
        {
            *pendCtx = 0;
        }
    }
}

void grPrintEngineGraphicsStatus_GF100(void)
{
    char buffer[GR_REG_NAME_BUFFER_LEN];

    // A little tricky: print address of this reg as a string in order
    // to use priv_dump, which will print all of the fields
    // automatically.  It won't print the fact that _ENGINE_GRAPHICS
    // is the register being shown (most likely the value will be 0),
    // so print that beforehand to make it clear.  priv_dump will
    // recognize "LW_PFIFO_ENGINE_STATUS(0)" as an address, but not
    // "LW_PFIFO_ENGINE_STATUS(NF_PFIFO_ENGINE_GRAPHICS)".  Using them
    // this way expands and checks the values at compile time.
    dprintf("LW_PFIFO_ENGINE_STATUS(LW_PFIFO_ENGINE_GRAPHICS):\n");
    sprintf( buffer, "0x%08x", LW_PFIFO_ENGINE_STATUS(LW_PFIFO_ENGINE_GRAPHICS) );
    priv_dump( buffer );
}

void grPrintExceptionStatusRegister_GF100(void)
{
    PRINT_REG_PD(_PGRAPH, _EXCEPTION);
}

LwBool grCheckPrivAccess_GF100(LwBool bForceEnable)
{
    LwU32 regSysPrivFsConfig;

    //
    // Check first if the graphics engine has priv access.
    // Bit 26 of LW_PPRIV_SYS_PRIV_FS_CONFIG denotes this for graphics.
    //
    regSysPrivFsConfig = GPU_REG_RD32(LW_PPRIV_SYS_PRIV_FS_CONFIG(0));
    if (!(regSysPrivFsConfig & BIT(LW_PPRIV_SYS_PRI_MASTER_fecs2gr_pri)))
    {
        // Check for forced enable access
        if (bForceEnable)
        {
            // Try to enable access
            GPU_REG_WR32(LW_PPRIV_SYS_PRIV_FS_CONFIG(0), (regSysPrivFsConfig | BIT(LW_PPRIV_SYS_PRI_MASTER_fecs2gr_pri)));

            // Check for access again
            regSysPrivFsConfig = GPU_REG_RD32(LW_PPRIV_SYS_PRIV_FS_CONFIG(0));
            if (!(regSysPrivFsConfig & BIT(LW_PPRIV_SYS_PRI_MASTER_fecs2gr_pri)))
            {
                return FALSE;
            }
        }
        else
        {
            return FALSE;
        }
    }
    return TRUE;
}

void grGetGpcIdsFromGrIdx_GF100
(
    LwU32       grIdx,
    LwU32       *pGpcPhysId,
    LwU32       *pGpcCount
)
{
    LwU32 numActiveGpc = pGr[indexGpu].grGetNumActiveGpc(grIdx);
    LwU32 i;
    for (i = 0; i < numActiveGpc; i++)
    {
        pGpcPhysId[i] = i;
        *pGpcCount += 1;
    }
}

void grGetStatus_GF100(LwBool bFullPrint, LwBool bForce, LwU32 grIdx)
{
    LwU32 val;
    LwU32 numActiveGpc =  pGr[indexGpu].grGetNumActiveGpc(grIdx);
    LwU32 numActiveFbp = pGr[indexGpu].grGetNumActiveFbp();
    LwU32 gpcId;

    time_t lwrrent_time;

    if (!pGr[indexGpu].grCheckPrivAccess(bForce))
    {
        LwU32 regSysPrivFsConfig = GPU_REG_RD32(LW_PPRIV_SYS_PRIV_FS_CONFIG(0));
        dprintf("\n");
        dprintf("====================\n");
        dprintf("Graphics engine priv access blocked. Cannot read registers.\n");
        dprintf("LW_PPRIV_SYS_PRIV_FS_CONFIG(0) = 0x%.8x\n", regSysPrivFsConfig );
        if (!bForce)
        {
            dprintf("Try using -f option to see if priv access can be force enabled.\n");
        }
        dprintf("====================\n");
        return;
    }

    dprintf("====================\n");
    dprintf("Consolidated GPU status, showing high level, non-zero registers/fields.\n");
    if (!bFullPrint)
    {
        dprintf("To append a more detailed report, please use: lws grstatus \"-a\".\n");
    }
    dprintf("====================\n");

    // Show the time and date into every report
    lwrrent_time = time( NULL );
    dprintf( ctime( &lwrrent_time ) );

    // Chip master boot information
    val = GPU_REG_RD32( LW_PMC_BOOT_0 );
    dprintf( "Chip %s, ARCHITECTURE 0x%x, MAJOR_REVISION 0x%x,  MINOR_REVISION 0x%x, IMPLEMENTATION 0x%x, DEVICE_ID_STRAP 0x%x\n",
             getLwhalImplName(hal.halImpl),
             DRF_VAL( _PMC, _BOOT_0, _ARCHITECTURE, val ),
             DRF_VAL( _PMC, _BOOT_0, _MAJOR_REVISION, val ),
             DRF_VAL( _PMC, _BOOT_0, _MINOR_REVISION, val ),
             DRF_VAL( _PMC, _BOOT_0, _IMPLEMENTATION, val ),
             DRF_VAL( _PMC, _BOOT_0, _DEVICE_ID_STRAP, val )
        );

    // Floorsweeping
    dprintf( "Floorsweeping (GPCs/TPCs per GPC0,1,.../FBPs):   %d/", numActiveGpc );
    for (gpcId = 0 ; gpcId < numActiveGpc ; gpcId++)
    {
        dprintf( "%d%s", pGr[indexGpu].grGetNumTpcForGpc( gpcId, grIdx ),
                 (gpcId < (numActiveGpc-1) ? "," : "" ) );
    }
    dprintf( "/%d\n\n", numActiveFbp );

    // Always print the consolidated report
    pGr[indexGpu].grDumpConsolidatedReport( grIdx );

    if (bFullPrint)
    {
        pGr[indexGpu].grDumpDetailedStatus( bFullPrint, grIdx );
    }
}

void grDumpDetailedStatus_GF100( BOOL bFullPrint, LwU32 grIdx )
{
    LwU32 numActiveFbp, fbpId;
    char buffer[GR_REG_NAME_BUFFER_LEN];

    dprintf("********************************************************************************\n");
    dprintf("====================\n");
    dprintf("Detailed GPU status.\n");
    dprintf("====================\n\n");

    // The same high level info as in the consolidated report, but
    // print all registers/fields even if 0
    PRINT_REG_PD(_PGRAPH_GRFIFO, _STATUS);
    PRINT_REG_PD(_PGRAPH_GRFIFO, _CONTROL);

    PRINT_REG_PD(_PGRAPH, _STATUS);
    PRINT_REG_PD(_PGRAPH, _STATUS1);
    PRINT_REG_PD(_PGRAPH, _STATUS2);
    PRINT_REG_PD(_PGRAPH, _INTR);
    PRINT_REG_PD(_PGRAPH, _PRI_FECS_HOST_INT_STATUS );

    // LW_PGRAPH_ACTIVITY*
    PRINT_REG_PD(_PGRAPH, _ACTIVITY*);

    // program LW_PGRAPH_PRI_FE_PERFMON register to enable the perfmon reporting
    pGr[indexGpu].grEnableFePerfmonReporting();

    pGr[indexGpu].grPrintExceptionStatusRegister();

    // Full pipeline status - large printout
    // FECS and GPCCS detail is reported in grCheckPipelineStatus
    pGr[indexGpu].grCheckPipelineStatus( NULL, bFullPrint, grIdx );

    PRINT_REG_PD(_PGRAPH_FECS, _INTR);
    PRINT_REG_PD(_PGRAPH_PRI_FECS, _HOST_INT_STATUS);

    PRINT_REG_PD(_PGRAPH_PRI_DS, _MPIPE_STATUS);
    PRINT_REG_PD(_PGRAPH_PRI_SSYNC, _FSM_STATES);
    
    // LW_PFIFO_ENGINE_GRAPHICS
    pGr[indexGpu].grPrintEngineGraphicsStatus();

#ifdef LW_PGRAPH_PRI_FE_CTXSW_USER_STATE0
    PRINT_REG_PD(_PGRAPH_PRI_FE, _CTXSW_USER_STATE0);
#endif

#ifdef LW_PGRAPH_CTXSW_STATUS
    PRINT_REG_PD(_PGRAPH_CTXSW, _STATUS);
#endif

    // LW_PGRAPH_PRI_FECS_LWRRENT/NEW_CTX
    PRINT_REG_PD(_PGRAPH_PRI_FECS, _LWRRENT_CTX);
    PRINT_REG_PD(_PGRAPH_PRI_FECS, _NEW_CTX);

    PRINT_REG_PD(_PGRAPH_PRI_FE, _GO_IDLE_ON_STATUS);
    PRINT_REG_PD(_PGRAPH_PRI_FE, _GO_IDLE_TIMEOUT);
    PRINT_REG_PD(_PGRAPH_PRI_FE, _GO_IDLE_CHECK);

    // CROP/ZROP status for each FBP
    dprintf("====================\n");
    dprintf("Per FBP CROP/ZROP detailed status\n");
    dprintf("====================\n");

    numActiveFbp = pGr[indexGpu].grGetNumActiveFbp();
    for (fbpId = 0 ; fbpId < numActiveFbp ; fbpId++)
    {
        PRINT_BE_REG_PD(CROP_STATUS*, fbpId);
        PRINT_BE_REG_PD(ZROP_STATUS*, fbpId);
    }

    //dump xbar related regs
    pGr[indexGpu].grDumpXbarStatus( grIdx );

    //niso hub status
    pGr[indexGpu].grDumpNisoHubStatus();

    //2x Tex Signature Dump
    pGr[indexGpu].grGetTexHangSignatureInfoLww( grIdx );
}

void grDumpIsoHubStatus_GF100(void)
{
    LwU32 data32;
    LwU32 i;

    dprintf("====================\n");
    dprintf("ISO HUB detailed status\n");
    dprintf("====================\n");

    //LW_PFB_ISO_STATUS
    PRINT_REG_PD(_PFB, _ISO_STATUS);

    dprintf("Setting LW_PFB_ISO_CFGSTATUS_CTRL values (0..13) and then reading _STATUS\n");
    for (i=0; i<=ISO_CTRL_TEST_VAL_MAX; i++)
    {
        data32 = GPU_REG_RD32(LW_PFB_ISO_CFGSTATUS);
        data32 &= ~DRF_SHIFTMASK(LW_PFB_ISO_CFGSTATUS_CTRL);
        data32 |= DRF_NUM(_PFB, _ISO_CFGSTATUS, _CTRL, i);
        dprintf("Written %d\n", i);
        GPU_REG_WR32(LW_PFB_ISO_CFGSTATUS, data32);

        PRINT_REG_PD(_PFB, _ISO_CFGSTATUS);
        PRINT_REG_PD(_PFB, _ISO_STATUS);
    }
}

void grDumpNisoHubStatus_GF100(void)
{
    LwU32 data32;
    LwU32 i;
    LwU32 niso_debug_config_max;

    //LW_PFB_ISO_STATUS
    dprintf("====================\n");
    dprintf("NISO HUB detailed status\n");
    dprintf("====================\n");
    PRINT_REG_PD(_PFB_NISO,_DEBUG);

    // A mask for the CONFIG field indicates how many configs to
    // iterate on.  See pri_fb.ref in the chip manuals.
    niso_debug_config_max = DRF_MASK( LW_PFB_NISO_PRI_DEBUG1_CONFIG );
    dprintf("Setting LW_PFB_NISO_PRI_DEBUG1_CONFIG values (0..%d) and then reading _NISO_DEBUG\n",
        niso_debug_config_max);

    for (i=0; i<= niso_debug_config_max; i++)
    {
        data32 = GPU_REG_RD32(LW_PFB_NISO_PRI_DEBUG1);
        data32 &= ~DRF_SHIFTMASK(LW_PFB_NISO_PRI_DEBUG1_CONFIG);
        data32 |= DRF_NUM(_PFB, _NISO_PRI_DEBUG1, _CONFIG, i);
        dprintf("Written %d\n", i );
        GPU_REG_WR32(LW_PFB_NISO_PRI_DEBUG1, data32);

        PRINT_REG_PD(_PFB, _NISO_PRI_DEBUG1);
        PRINT_REG_PD(_PFB, _NISO_DEBUG);
    }
}

void grDumpXbarStatus_GF100( LwU32 grIdx )
{
    LwU32 i;
    LwU32 gpcCount=0, fbpsCount=0;
    char buffer[GR_REG_NAME_BUFFER_LEN];

    dprintf("====================\n");
    dprintf("XBAR detailed status\n");
    dprintf("====================\n");

    pGr[indexGpu].grGetBusInfo(&gpcCount, NULL, &fbpsCount, NULL, NULL, grIdx);

    for (i=0; i<gpcCount; i++)
    {
        sprintf(buffer, "LW_XBAR_MXBAR_CQ_PRI_GPC%d_GXI_PREG_IDLE_STATUS", i );
        priv_dump(buffer);
    }

    for (i=0; i<gpcCount; i++)
    {
        sprintf(buffer, "LW_XBAR_CXBAR_CQ_PRI_GPC%d_GXI_PREG_IDLE_STATUS", i );
        priv_dump(buffer);
    }

    for (i=0; i<gpcCount; i++)
    {
        sprintf(buffer, "LW_XBAR_WXBAR_CQ_PRI_GPC%d_GXI_PREG_IDLE_STATUS", i );
        priv_dump(buffer);
    }

    for (i=0; i<gpcCount; i++)
    {
        sprintf(buffer, "LW_XBAR_CXBAR_CQ_PRI_GPC%d_GXI_PREG_VTX_ERROR_STATUS", i );
        priv_dump(buffer);
    }

    PRINT_REG_PD(_XBAR_MXBAR,_CQ_PRI_SYS0_HXI_PREG_IDLE_STATUS);
    PRINT_REG_PD(_XBAR_CXBAR,_CQ_PRI_SYS0_HXI_PREG_IDLE_STATUS);
    PRINT_REG_PD(_XBAR_WXBAR,_CQ_PRI_SYS0_HXI_PREG_IDLE_STATUS);
    PRINT_REG_PD(_XBAR_CXBAR,_CQ_PRI_SYS0_HXI_PREG_VTX_ERROR_STATUS);

    for (i=0; i<fbpsCount; i++)
    {
        sprintf(buffer, "LW_XBAR_MXBAR_CQ_PRI_FBP%d_FXI_PREG_IDLE_STATUS", i);
        priv_dump(buffer);
    }

    for (i=0; i<fbpsCount; i++)
    {
        sprintf(buffer, "LW_XBAR_CXBAR_CQ_PRI_FBP%d_FXI_PREG_IDLE_STATUS", i);
        priv_dump(buffer);
    }

    for (i=0; i<fbpsCount; i++)
    {
        sprintf(buffer, "LW_XBAR_CXBAR_CQ_PRI_FBP%d_FXI_PREG_VTX_ERROR_STATUS", i);
        priv_dump(buffer);
    }
}

//-----------------------------------------------------
// grGetChannelCtxInfo_GF100
//-----------------------------------------------------

#ifndef LW_PGRAPH_CTXSW_LWRRENT_CTX         // Temporary short-cut
void grGetChannelCtxInfo_GF100(LwU32 chId)
{
    dprintf("%s: Please regen dev_graphics.h and recompile lwWatch\n", __FUNCTION__);
}

#else
void grGetChannelCtxInfo_GF100(LwU32 chId)
{
    LwU32 grStatus;

    if (chId)
    {
        dprintf("Caution: chid 0x%02lx (%lu) ignored by %s\n", chId, chId, __FUNCTION__);
    }
    grStatus = GPU_REG_RD32(LW_PGRAPH_CTXSW_LWRRENT_CTX);
    dprintf("LW_PGRAPH_CTXSW_LWRRENT_CTX:   0x%08x\n", grStatus);

    if ( !(grStatus & DRF_DEF(_PGRAPH, _CTXSW_LWRRENT_CTX, _VALID, _TRUE)))
        dprintf("  +\t_VALID_FALSE\n");

    else            // Context pointer is valid
    {
        LwU32 grLwrCtxInst, grLwrCtxAddr;
        grLwrCtxInst = DRF_VAL(_PGRAPH, _CTXSW_LWRRENT_CTX, _PTR, grStatus);
        grLwrCtxAddr = LW_INST_START_ADDR + (grLwrCtxInst << 4);

        if (verboseLevel)
        {
            dprintf("PGRAPH_CTXSW_LWRRENT_CTX_PTR:  0x%081x\n", grLwrCtxInst);
            dprintf("grLwrCtxAddr:                  0x%081x\n", grLwrCtxAddr);
        }

        dprintf("See /hw/fermi1_gf100/include/gf100/ctx_state_store.h\n");
        dprintf("\n");

        printData(grLwrCtxAddr, 0x200);     // 512 bytes
    }
}
#endif

void grDumpZlwllExceptionState_GF100(LwU32 gpcCounter)
{
    LwU32 hwwEsr;
    char buffer[GR_REG_NAME_BUFFER_LEN];

    hwwEsr = GPU_REG_RD32(GPC_REG_ADDR(ZLWLL_HWW_ESR, gpcCounter));
    dprintf("====================\n");
    dprintf("Graphics ZLWLL Exception State detailed status\n");
    dprintf("====================\n");

    DUMP_GPC_REG(ZLWLL_HWW_ESR, gpcCounter);

    if (hwwEsr & DRF_DEF(_PGRAPH, _PRI_GPC0_ZLWLL_HWW_ESR, _SAVE_RESTORE_ADDR_OOB, _PENDING))
    {
        LwU32 hwwInfo0, hwwInfo1, hwwInfo2;

        hwwInfo0 = GPU_REG_RD32(GPC_REG_ADDR(ZLWLL_HWW_ESR_INFO_0, gpcCounter));
        hwwInfo1 = GPU_REG_RD32(GPC_REG_ADDR(ZLWLL_HWW_ESR_INFO_1, gpcCounter));
        hwwInfo2 = GPU_REG_RD32(GPC_REG_ADDR(ZLWLL_HWW_ESR_INFO_2, gpcCounter));

        dprintf("ZLWLL Exception Type: SAVE_RESTORE_ADDR_OOB\n");

        dprintf("ZLWLL%d: Address Low 0x%x\n",
                gpcCounter, DRF_VAL(_PGRAPH, _PRI_GPC0_ZLWLL_HWW_ESR_INFO_0, _ADDRESS_LOW, hwwInfo0));
        dprintf("ZLWLL%d: Address High 0x%x\n",
                gpcCounter, DRF_VAL(_PGRAPH, _PRI_GPC0_ZLWLL_HWW_ESR_INFO_1, _ADDRESS_HIGH, hwwInfo1));
        dprintf("ZLWLL%d: Address 40 bit address 0x%x%x\n",
                gpcCounter, DRF_VAL(_PGRAPH, _PRI_GPC0_ZLWLL_HWW_ESR_INFO_1, _ADDRESS_HIGH, hwwInfo1),
                DRF_VAL(_PGRAPH, _PRI_GPC0_ZLWLL_HWW_ESR_INFO_0, _ADDRESS_LOW, hwwInfo0));

        dprintf("ZLWLL%d: Limit Address Low 0x%x\n",
                gpcCounter, DRF_VAL(_PGRAPH, _PRI_GPC0_ZLWLL_HWW_ESR_INFO_2, _LIMIT_ADDRESS_LOW, hwwInfo2));
        dprintf("ZLWLL%d: Limit Address High 0x%x\n",
                gpcCounter, DRF_VAL(_PGRAPH, _PRI_GPC0_ZLWLL_HWW_ESR_INFO_1, _LIMIT_ADDRESS_HIGH, hwwInfo1));
        dprintf("ZLWLL%d: Limit Address 40 bit address 0x%x%x\n",
                gpcCounter, DRF_VAL(_PGRAPH, _PRI_GPC0_ZLWLL_HWW_ESR_INFO_1, _LIMIT_ADDRESS_HIGH, hwwInfo1),
                DRF_VAL(_PGRAPH, _PRI_GPC0_ZLWLL_HWW_ESR_INFO_2, _LIMIT_ADDRESS_LOW, hwwInfo2));
    }
}

void grDumpGccExceptionState_GF100(LwU32 gpcCounter)
{
    LwU32 hwwEsr, badHdrIdx, badSmpIdx;
    char buffer[GR_REG_NAME_BUFFER_LEN];

    dprintf("====================\n");
    dprintf("GPCCS Exception State detailed status\n");
    dprintf("====================\n");

    hwwEsr = GPU_REG_RD32(GPC_REG_ADDR(GCC_HWW_ESR, gpcCounter));
    badHdrIdx = GPU_REG_RD32(GPC_REG_ADDR(GCC_BAD_TEX_HDR_INDEX, gpcCounter));
    badSmpIdx = GPU_REG_RD32(GPC_REG_ADDR(GCC_BAD_TEX_SMP_INDEX, gpcCounter));

    DUMP_GPC_REG(GCC_HWW_ESR, gpcCounter );

    if (hwwEsr & DRF_DEF(_PGRAPH, _PRI_GPC0_GCC_HWW_ESR, _TEX_HDR_INDEXV, _PENDING))
    {
        dprintf("LW_PGRAPH_PRI_GPC%d_GCC_HWW_ESR_TEX_HDR_INDEXV\n", gpcCounter);
        dprintf("Graphics GCC Exception Type: TEX_HDR_INDEXV (tex header index violation)\n");
        dprintf("Graphics GCC:  LW_PGRAPH_PRI_GPC%d_GCC_BAD_TEX_HDR_INDEX: 0x%x\n", gpcCounter,
                DRF_VAL(_PGRAPH, _PRI_GPC0_GCC_BAD_TEX_HDR_INDEX, _VALUE, badHdrIdx));
    }

    if (hwwEsr & DRF_DEF(_PGRAPH, _PRI_GPC0_GCC_HWW_ESR, _TEX_SMP_INDEXV, _PENDING))
    {
        dprintf("LW_PGRAPH_PRI_GPC%d_GCC_HWW_ESR_TEX_SMP_INDEXV\n", gpcCounter);
        dprintf("Graphics GCC Exception Type: TEX_SMP_INDEXV (tex sampler index violation)\n");
        dprintf("Graphics GCC:  LW_PGRAPH_PRI_GPC%d_GCC_BAD_TEX_SMP_INDEX: 0x%x\n", gpcCounter,
                DRF_VAL(_PGRAPH, _PRI_GPC0_GCC_BAD_TEX_SMP_INDEX, _VALUE, badSmpIdx));
    }
}

void grDumpSetupExceptionState_GF100(LwU32 gpcCounter)
{
    LwU32 hwwEsr;
    char buffer[GR_REG_NAME_BUFFER_LEN];

    dprintf("====================\n");
    dprintf("Setup Exception State detailed status\n");
    dprintf("====================\n");

    hwwEsr = GPU_REG_RD32(GPC_REG_ADDR(SETUP_HWW_ESR, gpcCounter));
    DUMP_GPC_REG(SETUP_HWW_ESR, gpcCounter);

    if (hwwEsr & DRF_DEF(_PGRAPH, _PRI_GPC0_SETUP_HWW_ESR, _LDCONST_OOB, _PENDING))
    {
        dprintf("LW_PGRAPH_PRI_GPC%d_SETUP_HWW_ESR_LDCONST_OOB\n", gpcCounter);
        dprintf("VRM: Graphics SETUP Exception Type: LDCONST_OOB\n");
    }

    if (hwwEsr & DRF_DEF(_PGRAPH, _PRI_GPC0_SETUP_HWW_ESR, _STATECB_ZEROSIZE, _PENDING))
    {
        dprintf("LW_PGRAPH_PRI_GPC%d_SETUP_HWW_ESR_STATECB_ZEROSIZE\n", gpcCounter);
        dprintf("Graphics SETUP Exception Type: STATECB_ZEROSIZE\n");
    }

    if (hwwEsr & DRF_DEF(_PGRAPH, _PRI_GPC0_SETUP_HWW_ESR, _STATECB_ILWALID, _PENDING))
    {
        dprintf("LW_PGRAPH_PRI_GPC%d_SETUP_HWW_ESR_STATECB_ILWALID\n", gpcCounter);
        dprintf("Graphics SETUP Exception Type: STATECB_ILWALID\n");
    }

    if (hwwEsr & DRF_DEF(_PGRAPH, _PRI_GPC0_SETUP_HWW_ESR, _POOLPAGE_ILWALID, _PENDING))
    {
        dprintf("LW_PGRAPH_PRI_GPC%d_SETUP_HWW_ESR_POOLPAGE_ILWALID\n", gpcCounter);
        dprintf("Graphics SETUP Exception Type: POOLPAGE_ILWALID\n");
    }
}



LW_STATUS grCheckPriFecsStatus_GF100(void)
{
    //1. read _CTXSW_STATUS_FE_0
    PRINT_REG_PD(_PGRAPH_PRI_FECS, _CTXSW_STATUS_FE_0);

    //2. read STATUS_1
    PRINT_REG_PD(_PGRAPH_PRI_FECS, _CTXSW_STATUS_1);

    //3. read MAILBOX (0) ... (7)
    pGr[indexGpu].grDumpFecsMailboxRegisters();

    //just dumping the values
    return LW_OK;
}

LW_STATUS grCheckPriGpccsStatus_GF100( LwU32 grIdx )
{
    LwU32 gpcIdx;
    char  buffer[GR_REG_NAME_BUFFER_LEN];
    LwU32 gpcCount =  pGr[indexGpu].grGetNumActiveGpc( grIdx );

    buffer[0] = '\0';

    for (gpcIdx = 0; gpcIdx < gpcCount; ++gpcIdx)
    {
        //1. read _STATUS_0 and _STATUS_1 (and more if exist in the future)
        PRINT_GPC_REG_PD(GPCCS_CTXSW_STATUS_*, gpcIdx);

        //3. read MAILBOX (0) ... (n) as defined in the manuals by
        //   using a regex for the mailbox #
        PRINT_GPC_REG_PD(GPCCS_CTXSW_MAILBOX*, gpcIdx );
    }
    //just dumping the values
    return LW_OK;
}

//-----------------------------------------------------
// grCheckCtxswStatus_GF100
// check status of ctx switch machine
//-----------------------------------------------------
LW_STATUS grCheckCtxswStatus_GF100(BOOL bFullPrint, LwU32 grIdx)
{
    LW_STATUS status = LW_OK;

    if ( bFullPrint)
    {
        dprintf("====================\n");
        dprintf(" CTXSW status: Fecs and Gpccs\n");
        dprintf("====================\n");

        status = pGr[indexGpu].grCheckPriFecsStatus();
        status = pGr[indexGpu].grCheckPriGpccsStatus( grIdx);
    }

    return status;
}

//-----------------------------------------------------
// grDumpFeSemaphoreAcquireState_GF100
//
//-----------------------------------------------------
void grDumpFeSemaphoreAcquireState_GF100(void)
{
    PRINT_REG_PD(_PGRAPH_FE, _SEMAPHORE_ACQUIRE_0);
    PRINT_REG_PD(_PGRAPH_FE, _SEMAPHORE_ACQUIRE_1);
    PRINT_REG_PD(_PGRAPH_FE, _SEMAPHORE_ACQUIRE2);
    PRINT_REG_PD(_PGRAPH_FE, _SEMAPHORE_TIMESTAMP);
}

LW_STATUS grDumpPriFeSemaphoreState_GF100(void)
{
    LW_STATUS status = LW_OK;

    dprintf("====================\n");
    dprintf(" FE SEMAPHORE STATE detailed status\n");
    dprintf("====================\n");
    PRINT_REG_PD(_PGRAPH_PRI_FE, _SEMAPHORE_STATE_A);
    PRINT_REG_PD(_PGRAPH_PRI_FE, _SEMAPHORE_STATE_B);
    PRINT_REG_PD(_PGRAPH_PRI_FE, _SEMAPHORE_STATE_C);
    PRINT_REG_PD(_PGRAPH_PRI_FE, _SEMAPHORE_STATE_D);
    PRINT_REG_PD(_PGRAPH_PRI_FE, _SEMAPHORE_STATE_REPORT);

    pGr[indexGpu].grDumpFeSemaphoreAcquireState();
    return status;
}

LwU32 grCheckFELwrrentObject_GF100(void)
{
    LwU32 status = LW_OK;
    LwU32 data32 = GPU_REG_RD32(LW_PGRAPH_PRI_FE_LWRRENT_OBJECT_0);

    PRINT_REG_PD(_PGRAPH_PRI_FE, _LWRRENT_OBJECT_0);

    if (DRF_VAL(_PGRAPH, _PRI_FE, _LWRRENT_OBJECT_0_LWCLASS, data32) ==
        LW_PGRAPH_PRI_FE_LWRRENT_OBJECT_0_LWCLASS_INIT )
    {
        dprintf("ERROR: LWCLASS is _INIT (0)\n");
        addUnitErr("\t LWCLASS is _INIT (0)\n");
        status = LW_ERR_GENERIC;
    }

    return status;
}

//-----------------------------------------------------
// grCheckFeMethodStatus_GF100
//
//-----------------------------------------------------
LW_STATUS grCheckFeMethodStatus_GF100(void)
{
    LwU32   data32;
    LW_STATUS    status = LW_OK;
    LwU32   subChId0;
    LwU32   subChId1;
    LwU32   tmpStatus;

    data32 = GPU_REG_RD32(LW_PGRAPH_PRI_FE_METHOD_STATE);
    subChId0 = DRF_VAL(_PGRAPH, _PRI_FE_METHOD_STATE, _SUBCH, data32);
    PRINT_REG_PD(_PGRAPH_PRI_FE, _METHOD_STATE);

    data32 = GPU_REG_RD32(LW_PGRAPH_PRI_FE_LWRRENT_METHOD);
    subChId1 = DRF_VAL(_PGRAPH, _PRI_FE_LWRRENT_METHOD, _SUBCH, data32);
    PRINT_REG_PD(_PGRAPH_PRI_FE, _LWRRENT_METHOD);

    // All register fields have been dumped, so only print error messages
    if (DRF_VAL(_PGRAPH, _PRI_FE, _LWRRENT_METHOD_STATUS, data32) ==
        LW_PGRAPH_PRI_FE_LWRRENT_METHOD_STATUS_ILWALID )
    {
        dprintf("ERROR: LW_PGRAPH_PRI_FE_LWRRENT_METHOD_STATUS_ILWALID\n");
        addUnitErr("\t LW_PGRAPH_PRI_FE_LWRRENT_METHOD_STATUS_ILWALID\n");
        status = LW_ERR_GENERIC;
    }
    else
    {
        // Leave this since the value is left shifted 2 bits and that must somehow
        // be important to understanding the chip state
        dprintf("\t\t%-33s = [0x%04x]\n",
                "+ _FE_LWRRENT_METHOD_MTHD << 2",
                DRF_VAL(_PGRAPH, _PRI_FE, _LWRRENT_METHOD_MTHD, data32) << 2);

        // check METHOD_STATE subchID w/ current method subchID
        if (subChId0 != subChId1)
        {
            dprintf("ERROR: LW_PGRAPH_PRI_FE_METHOD_STATE_SUBCH: 0x%02x != "
                "LW_PGRAPH_PRI_FE_LWRRENT_METHOD_STATE_SUBCH: 0x%02x\n", subChId0, subChId1);
            addUnitErr("\t LW_PGRAPH_PRI_FE_METHOD_STATE_SUBCH: 0x%02x != "
                "LW_PGRAPH_PRI_FE_LWRRENT_METHOD_STATE_SUBCH: 0x%02x\n", subChId0, subChId1);
            status = LW_ERR_GENERIC;
        }

        // indicates if the current method has completed accessing memory (if needed).
        if (data32 & DRF_DEF(_PGRAPH, _PRI_FE, _LWRRENT_METHOD_MEM_XACTION, _TRUE))
        {
            dprintf("ERROR: LW_PGRAPH_PRI_FE_LWRRENT_METHOD_MEM_XACTION_TRUE\n");
            addUnitErr("\t LW_PGRAPH_PRI_FE_LWRRENT_METHOD_MEM_XACTION_TRUE\n");
            status = LW_ERR_GENERIC;
        }

        status = pGr[indexGpu].grDumpFeLwrrentMethodWfi(data32);

        if (data32 & DRF_DEF(_PGRAPH, _PRI_FE, _LWRRENT_METHOD_DATAHIGH, _VALID))
        {
            PRINT_REG_PD(_PGRAPH_PRI_FE, _LWRRENT_METHOD_DATA_HIGH_V);
        }
    }

    tmpStatus = pGr[indexGpu].grCheckFELwrrentObject();
    if (tmpStatus != LW_OK)
    {
        status = tmpStatus;
    }

    return status;
}

LW_STATUS grDumpFeLwrrentMethodWfi_GF100(LwU32 data32)
{
    LW_STATUS    status = LW_OK;
    //report error if method waiting for idle before being decoded
    if (data32 & DRF_DEF(_PGRAPH, _PRI_FE, _LWRRENT_METHOD_WFI, _TRUE))
        {
            dprintf("ERROR: LW_PGRAPH_PRI_FE_LWRRENT_METHOD_WFI_TRUE\n");
            addUnitErr("\t LW_PGRAPH_PRI_FE_LWRRENT_METHOD_MEM_XACTION_TRUE\n");
            status = LW_ERR_GENERIC;
        }
    return status;
}

void grDumpIntrEnSemaphoreTimeout_GF100(LwU32 grIntrEn)
{
   if (DRF_VAL(_PGRAPH, _INTR_EN, _SEMAPHORE_TIMEOUT, grIntrEn) == 0)
        dprintf("          _EN_SEMAPHORE_TIMEOUT_DISABLED\n");
}

void grDumpIntrSingleStep_GF100(LwU32 grIntr)
{
    if (grIntr & DRF_DEF(_PGRAPH, _INTR, _SINGLE_STEP, _PENDING))
    {
        dprintf("LW_PGRAPH_INTR_SINGLE_STEP_PENDING\n");
        addUnitErr("\t LW_PGRAPH_INTR_SINGLE_STEP_PENDING\n");
    }
}

void grDumpIntrSemaphoreTimeout_GF100(LwU32 grIntr)
{
    if (grIntr & DRF_DEF(_PGRAPH, _INTR, _SEMAPHORE_TIMEOUT, _PENDING))
    {
        dprintf("       _SEMAPHORE_TIMEOUT_PENDING\n");
        addUnitErr("\t LW_PGRAPH_INTR_SEMAPHORE_TIMEOUT_PENDING\n");
        pGr[indexGpu].grDumpFeSemaphoreAcquireState();
    }
}

// based on fermi/gf100/grergf100.c
void grDumpDsExceptionState_GF100(void)
{
    LwU32 hwwEsr = 0;

    hwwEsr = GPU_REG_RD32(LW_PGRAPH_PRI_DS_HWW_ESR);

    dprintf(" Dumping DS exception state LW_PGRAPH_PRI_DS_HWW_ESR:0x%x\n", hwwEsr);

    if (hwwEsr & DRF_DEF(_PGRAPH, _PRI_DS_HWW_ESR, _SPH0_ERR, _PENDING))
    {
        dprintf("       _DS_HWW_ESR_SPH0_ERR\n");
    }

    if (hwwEsr & DRF_DEF(_PGRAPH, _PRI_DS_HWW_ESR, _SPH1_ERR, _PENDING))
    {
        dprintf("       _DS_HWW_ESR_SPH1_ERR\n");
    }

    if (hwwEsr & DRF_DEF(_PGRAPH, _PRI_DS_HWW_ESR, _SPH2_ERR, _PENDING))
    {
        dprintf("           _DS_HWW_ESR_SPH2_ERR\n");
    }

    if (hwwEsr & DRF_DEF(_PGRAPH, _PRI_DS_HWW_ESR, _SPH3_ERR, _PENDING))
    {
        dprintf("           _DS_HWW_ESR_SPH3_ERR\n");
    }

    if (hwwEsr & DRF_DEF(_PGRAPH, _PRI_DS_HWW_ESR, _SPH4_ERR, _PENDING))
    {
        dprintf("           _DS_HWW_ESR_SPH4_ERR\n");
    }

    if (hwwEsr & DRF_DEF(_PGRAPH, _PRI_DS_HWW_ESR, _SPH5_ERR, _PENDING))
    {
        dprintf("           _DS_HWW_ESR_SPH5_ERR\n");
    }

    if (hwwEsr & DRF_DEF(_PGRAPH, _PRI_DS_HWW_ESR, _SPH6_ERR, _PENDING))
    {
        dprintf("           _DS_HWW_ESR_SPH6_ERR\n");
    }

    if (hwwEsr & DRF_DEF(_PGRAPH, _PRI_DS_HWW_ESR, _SPH7_ERR, _PENDING))
    {
        dprintf("           _DS_HWW_ESR_SPH7_ERR\n");
    }

    if (hwwEsr & DRF_DEF(_PGRAPH, _PRI_DS_HWW_ESR, _SPH8_ERR, _PENDING))
    {
        dprintf("           _DS_HWW_ESR_SPH8_ERR\n");
    }

    if (hwwEsr & DRF_DEF(_PGRAPH, _PRI_DS_HWW_ESR, _SPH9_ERR, _PENDING))
    {
        dprintf("           _DS_HWW_ESR_SPH9_ERR\n");
    }

    if (hwwEsr & DRF_DEF(_PGRAPH, _PRI_DS_HWW_ESR, _SPH10_ERR, _PENDING))
    {
        dprintf("           _DS_HWW_ESR_SPH10_ERR\n");
    }

    if (hwwEsr & DRF_DEF(_PGRAPH, _PRI_DS_HWW_ESR, _SPH11_ERR, _PENDING))
    {
        dprintf("           _DS_HWW_ESR_SPH11_ERR\n");
    }

    if (hwwEsr & DRF_DEF(_PGRAPH, _PRI_DS_HWW_ESR, _SPH12_ERR, _PENDING))
    {
        dprintf("           _DS_HWW_ESR_SPH12_ERR\n");
    }

    if (hwwEsr & DRF_DEF(_PGRAPH, _PRI_DS_HWW_ESR, _SPH13_ERR, _PENDING))
    {
        dprintf("           _DS_HWW_ESR_SPH13_ERR\n");
    }

    if (hwwEsr & DRF_DEF(_PGRAPH, _PRI_DS_HWW_ESR, _SPH14_ERR, _PENDING))
    {
        dprintf("           _DS_HWW_ESR_SPH14_ERR\n");
    }

    if (hwwEsr & DRF_DEF(_PGRAPH, _PRI_DS_HWW_ESR, _SPH15_ERR, _PENDING))
    {
        dprintf("           _DS_HWW_ESR_SPH15_ERR\n");
    }

    if (hwwEsr & DRF_DEF(_PGRAPH, _PRI_DS_HWW_ESR, _SPH16_ERR, _PENDING))
    {
        dprintf("           _DS_HWW_ESR_SPH16_ERR\n");
    }

    if (hwwEsr & DRF_DEF(_PGRAPH, _PRI_DS_HWW_ESR, _SPH17_ERR, _PENDING))
    {
        dprintf("           _DS_HWW_ESR_SPH17_ERR\n");
    }

    if (hwwEsr & DRF_DEF(_PGRAPH, _PRI_DS_HWW_ESR, _SPH18_ERR, _PENDING))
    {
        dprintf("           _DS_HWW_ESR_SPH18_ERR\n");
    }

    if (hwwEsr & DRF_DEF(_PGRAPH, _PRI_DS_HWW_ESR, _SPH19_ERR, _PENDING))
    {
        dprintf("           _DS_HWW_ESR_SPH19_ERR\n");
    }

    if (hwwEsr & DRF_DEF(_PGRAPH, _PRI_DS_HWW_ESR, _SPH20_ERR, _PENDING))
    {
        dprintf("           _DS_HWW_ESR_SPH20_ERR\n");
    }

    if (hwwEsr & DRF_DEF(_PGRAPH, _PRI_DS_HWW_ESR, _SPH21_ERR, _PENDING))
    {
        dprintf("           _DS_HWW_ESR_SPH21_ERR\n");
    }

    if (hwwEsr & DRF_DEF(_PGRAPH, _PRI_DS_HWW_ESR, _SPH22_ERR, _PENDING))
    {
        dprintf("           _DS_HWW_ESR_SPH22_ERR\n");
    }

    if (hwwEsr & DRF_DEF(_PGRAPH, _PRI_DS_HWW_ESR, _SPH23_ERR, _PENDING))
    {
        dprintf("           _DS_HWW_ESR_SPH23_ERR\n");
    }

    // print the pipe stage
    dprintf("Dumping DS pipe stage \n");
    dprintf("LW_PGRAPH_PRI_DS_HWW_ESR_M_PIPE_STAGE: %d = ",
            DRF_VAL(_PGRAPH, _PRI_DS_HWW_ESR, _M_PIPE_STAGE, hwwEsr));

    if (DRF_VAL(_PGRAPH, _PRI_DS_HWW_ESR, _M_PIPE_STAGE, hwwEsr) & LW_PGRAPH_PRI_DS_HWW_ESR_M_PIPE_STAGE_VSA)
    {
        dprintf("           _DS_HWW_ESR_M_PIPE_STAGE_VSA\n");
    }
    if (DRF_VAL(_PGRAPH, _PRI_DS_HWW_ESR, _M_PIPE_STAGE, hwwEsr) & LW_PGRAPH_PRI_DS_HWW_ESR_M_PIPE_STAGE_VSB)
    {
        dprintf("           _DS_HWW_ESR_M_PIPE_STAGE_VSB\n");
    }
    if (DRF_VAL(_PGRAPH, _PRI_DS_HWW_ESR, _M_PIPE_STAGE, hwwEsr) & LW_PGRAPH_PRI_DS_HWW_ESR_M_PIPE_STAGE_TI)
    {
        dprintf("           _DS_HWW_ESR_M_PIPE_STAGE_TI\n");
    }
    if (DRF_VAL(_PGRAPH, _PRI_DS_HWW_ESR, _M_PIPE_STAGE, hwwEsr) & LW_PGRAPH_PRI_DS_HWW_ESR_M_PIPE_STAGE_TS)
    {
        dprintf("           _DS_HWW_ESR_M_PIPE_STAGE_TS\n");
    }
    if (DRF_VAL(_PGRAPH, _PRI_DS_HWW_ESR, _M_PIPE_STAGE, hwwEsr) & LW_PGRAPH_PRI_DS_HWW_ESR_M_PIPE_STAGE_GS)
    {
        dprintf("           _DS_HWW_ESR_M_PIPE_STAGE_GS\n");
    }
    if (DRF_VAL(_PGRAPH, _PRI_DS_HWW_ESR, _M_PIPE_STAGE, hwwEsr) & LW_PGRAPH_PRI_DS_HWW_ESR_M_PIPE_STAGE_PS)
    {
        dprintf("           _DS_HWW_ESR_M_PIPE_STAGE_PS\n");
    }

}

void grDumpMmeHwwEsr_GF100(LwU32 data32)
{
    if (!(data32 & DRF_DEF(_PGRAPH, _PRI_MME_HWW_ESR, _INFO_PC_VALID,_INIT)))
            {
                dprintf("            _PRI_MME_HWW_ESR_INFO_PC: 0x%x\n",
                     DRF_VAL(_PGRAPH_PRI, _MME_HWW_ESR_INFO, _PC, data32));
            }
}
//-----------------------------------------------------
// grCheckGrFifoStatus_GF100
// - check the status of gr fifo
//-----------------------------------------------------
LW_STATUS grCheckGrFifoStatus_GF100(void)
{
    LwU32   data32;
    LW_STATUS    status = LW_OK;

    data32 = GPU_REG_RD32(LW_PGRAPH_GRFIFO_CONTROL);

    dprintf("LW_PGRAPH_GRFIFO_CONTROL:             0x%08x\n", data32);

    //set error if gr fifo is disabled
    if (data32 & DRF_DEF(_PGRAPH, _GRFIFO_CONTROL, _ACCESS, _ENABLED))
    {
        dprintf(" +\t_CONTROL_ACCESS_ENABLED\n");
    }
    else
    {
        dprintf("ERROR: + LW_PGRAPH_GRFIFO_CONTROL_ACCESS_DISABLED\n");
        addUnitErr("\t LW_PGRAPH_GRFIFO_CONTROL_ACCESS_DISABLED\n");

        status = LW_ERR_GENERIC;
    }

    if (data32 & DRF_DEF(_PGRAPH, _GRFIFO_CONTROL, _SEMAPHORE_ACCESS, _ENABLED))
        dprintf(" +\t_SEMAPHORE_ACCESS_ENABLED\n");
    else
        dprintf(" +\t_SEMAPHORE_ACCESS_DISABLED\n");

    data32 = GPU_REG_RD32(LW_PGRAPH_GRFIFO_STATUS);

    dprintf("LW_PGRAPH_GRFIFO_STATUS:              0x%08x\n", data32);

    if (data32 & DRF_DEF(_PGRAPH, _GRFIFO_STATUS, _EMPTY, _TRUE))
        dprintf(" +\t_EMPTY_TRUE\n");
    else
        dprintf(" +\t_EMPTY_FALSE\n");

    if (data32 & DRF_DEF(_PGRAPH, _GRFIFO_STATUS, _FULL, _TRUE))
        dprintf(" +\t_FULL_TRUE\n");
    else
        dprintf(" +\t_FULL_FALSE\n");

    pGr[indexGpu].grDumpGrfifoStatusPtrInfo(data32);

    return status;
}

void grDumpGrfifoStatusPtrInfo_GF100(LwU32 data32)
{
    dprintf(" +\t%-30s\t: 0x%x\n",
            "_COUNT",
            DRF_VAL(_PGRAPH, _GRFIFO_STATUS, _COUNT, data32));

    dprintf(" +\t%-30s\t: 0x%x\n",
            "_READ_PTR",
            DRF_VAL(_PGRAPH, _GRFIFO_STATUS, _READ_PTR, data32));

    dprintf(" +\t%-30s\t: 0x%x\n",
            "_WRITE_PTR",
            DRF_VAL(_PGRAPH, _GRFIFO_STATUS, _WRITE_PTR, data32));
}

LW_STATUS grGetBusInfo_GF100(LwU32* pGpcCount,
                        LwU32* pMaxGpcCount,
                        LwU32* pFbpsCount,
                        LwU32* pMaxFbpsCount,
                        LwU32* pMaxTpcPerGpcCount,
                        LwU32 grIdx )
{
    LwU32 data32;

    if (pGpcCount)
    {
        *pGpcCount = pGr[indexGpu].grGetNumActiveGpc( grIdx );
    }

    if (pMaxGpcCount)
    {
        *pMaxGpcCount = pGr[indexGpu].grGetNumGpcs( grIdx );
    }

    if (pFbpsCount)
    {
        *pFbpsCount = pGr[indexGpu].grGetNumActiveFbp();
    }

    if (pMaxFbpsCount)
    {
        data32 = GPU_REG_RD32(LW_PTOP_SCAL_NUM_FBPS);
        *pMaxFbpsCount = DRF_VAL(_PTOP, _SCAL_NUM_FBPS, _VALUE, data32);
    }

    if (pMaxTpcPerGpcCount)
    {
        *pMaxTpcPerGpcCount =  grGetNumTpcPerGpc_GF100();
    }

    return LW_OK;
}

void grDumpGpcInfo_GF100(LwU32 gpcId, LwU32 grIdx )
{
    LwU32 tpcId, numActiveTpc;
    char buffer[GR_REG_NAME_BUFFER_LEN] = { 0 };
    buffer[0] = '\0';

    dprintf("====================\n");
    dprintf("GPC %d\n", gpcId);
    dprintf("====================\n");

    // '*' in the regex matches all ACTIVITY# registers for this GPC
    PRINT_GPC_REG_PD(GPCCS_GPC_ACTIVITY*, gpcId);

    DUMP_GPC_REG(PROP_HWW_ESR, gpcId);
    DUMP_GPC_REG(PROP_HWW_ESR_COORD, gpcId);
    DUMP_GPC_REG(PROP_HWW_ESR_FORMAT, gpcId);
    DUMP_GPC_REG(PROP_PS_ILWOCATIONS_HI, gpcId);
    DUMP_GPC_REG(PROP_EZ_ZPASS_FAIL_LO, gpcId);
    DUMP_GPC_REG(PROP_EZ_ZPASS_FAIL_HI, gpcId);
    DUMP_GPC_REG(PROP_ZPASS_CNT_LO, gpcId);
    DUMP_GPC_REG(PROP_ZPASS_CNT_HI, gpcId);
    DUMP_GPC_REG(PROP_IEEE_CLEAN_ZETA_TARGET, gpcId);
    DUMP_GPC_REG(PROP_EZ_ZPASS_CNT_LO, gpcId);
    DUMP_GPC_REG(PROP_EZ_ZPASS_CNT_HI, gpcId);
    // Use priv_dump with regex to print all STATE_* regs.  GM20x added _6 and _7
    PRINT_GPC_REG_PD(PROP_STATE_., gpcId);

    DUMP_GPC_REG(PROP_PM_CTRL, gpcId);
    DUMP_GPC_REG(PROP_CG, gpcId);
    DUMP_GPC_REG(PROP_CG1, gpcId);
    DUMP_GPC_REG(PROP_PG, gpcId);
    DUMP_GPC_REG(PROP_PG1, gpcId);

    DUMP_GPC_REG(FRSTR_DEBUG, gpcId);
    DUMP_GPC_REG(FRSTR_PM_CTRL, gpcId);
    DUMP_GPC_REG(FRSTR_CG1, gpcId);
    DUMP_GPC_REG(FRSTR_PG1, gpcId);

    DUMP_GPC_REG(WIDCLIP_PM_CTRL, gpcId);
    DUMP_GPC_REG(WIDCLIP_DEBUG, gpcId);
    DUMP_GPC_REG(WIDCLIP_CG, gpcId);
    DUMP_GPC_REG(WIDCLIP_CG1, gpcId);
    DUMP_GPC_REG(WIDCLIP_PG, gpcId);
    DUMP_GPC_REG(WIDCLIP_PG1, gpcId);

    DUMP_GPC_REG(SETUP_CG, gpcId);
    DUMP_GPC_REG(SETUP_CG1, gpcId);
    DUMP_GPC_REG(SETUP_PG, gpcId);
    DUMP_GPC_REG(SETUP_RM_BUNDLE_CB_SIZE, gpcId);
    DUMP_GPC_REG(SETUP_DEBUG_LIMIT0, gpcId);
    DUMP_GPC_REG(SETUP_DEBUG_LIMIT1, gpcId);
    DUMP_GPC_REG(SETUP_PM_CTRL, gpcId);
    DUMP_GPC_REG(SETUP_HWW_ESR, gpcId);

    DUMP_GPC_REG(CRSTR_DEBUG, gpcId);
    DUMP_GPC_REG(CRSTR_MAP_TABLE_CONFIG, gpcId);
    DUMP_GPC_REG(CRSTR_CG, gpcId);

    DUMP_GPC_REG(ZLWLL_HWW_ESR, gpcId);
    DUMP_GPC_REG(ZLWLL_HWW_ESR_INFO_2, gpcId);
    DUMP_GPC_REG(ZLWLL_FS, gpcId);
    DUMP_GPC_REG(ZLWLL_CG, gpcId);
    DUMP_GPC_REG(ZLWLL_CG1, gpcId);
    DUMP_GPC_REG(ZLWLL_PG, gpcId);
    DUMP_GPC_REG(ZLWLL_PG1, gpcId);

    PRINT_GPC_REG_PD(GPM_PD_STATUS, gpcId);

    DUMP_GPC_REG(GPM_PD_PM, gpcId);
    DUMP_GPC_REG(GPM_PD_CG1, gpcId);
    DUMP_GPC_REG(GPM_PD_PG, gpcId);
    DUMP_GPC_REG(GPM_PD_PG1, gpcId);
    DUMP_GPC_REG(GPM_SD_CONFIG, gpcId);

    PRINT_GPC_REG_PD(GPM_SD_STATUS, gpcId);

    numActiveTpc = pGr[indexGpu].grGetNumTpcForGpc( gpcId, grIdx );
    for ( tpcId = 0 ; tpcId < numActiveTpc ; tpcId++ )
    {
        sprintf( buffer, "LW_PGRAPH_PRI_GPC%d_GPM_SD_STATUS_TPC(%d)", gpcId, tpcId );

        priv_dump( buffer );
    }

    DUMP_GPC_REG(GPM_SD_PM, gpcId);
    DUMP_GPC_REG(GPM_SD_ACTIVE_TPCS, gpcId);

    PRINT_GPC_REG_PD(GPM_SD_STATUS_COLL, gpcId);
    PRINT_GPC_REG_PD(GPM_SD_STATUS_DIST, gpcId);
    PRINT_GPC_REG_PD(GPM_SD_STATUS_IN, gpcId);
    PRINT_GPC_REG_PD(GPM_SD_STATUS_OUT, gpcId);

    DUMP_GPC_REG(GPM_SD_CG, gpcId);
    DUMP_GPC_REG(GPM_SD_CG1, gpcId);
    DUMP_GPC_REG(GPM_SD_PG, gpcId);
    DUMP_GPC_REG(GPM_SD_PG1, gpcId);
    DUMP_GPC_REG(GPM_RPT_PM, gpcId);

    PRINT_GPC_REG_PD(GPM_RPT_ALPHA_WRITE_POINTERS, gpcId);
    PRINT_GPC_REG_PD(GPM_RPT_BETA_WRITE_POINTERS, gpcId);
    PRINT_GPC_REG_PD(GPM_RPT_SCOREBOARD_0, gpcId);
    PRINT_GPC_REG_PD(GPM_RPT_SCOREBOARD_1, gpcId);
    PRINT_GPC_REG_PD(GPM_RPT_SCOREBOARD_2, gpcId);
    PRINT_GPC_REG_PD(GPM_RPT_SCOREBOARD_3, gpcId);
    PRINT_GPC_REG_PD(GPM_RPT_STATUS, gpcId);

    DUMP_GPC_REG(GPM_RPT_CG, gpcId);
    DUMP_GPC_REG(GPM_RPT_CG1, gpcId);
    DUMP_GPC_REG(GPM_RPT_PG, gpcId);
    DUMP_GPC_REG(GPM_RPT_PG1, gpcId);

    DUMP_GPC_REG(WDXPS_PM, gpcId);
    DUMP_GPC_REG(WDXPS_CG, gpcId);
    DUMP_GPC_REG(WDXPS_CG1, gpcId);
    DUMP_GPC_REG(WDXPS_PG1, gpcId);

    DUMP_GPC_REG(SWDX_CG, gpcId);
    DUMP_GPC_REG(SWDX_PG, gpcId);
    DUMP_GPC_REG(SWDX_PG1, gpcId);

    DUMP_GPC_REG(GCC_HWW_ESR, gpcId);
    DUMP_GPC_REG(GCC_CG1, gpcId);
    DUMP_GPC_REG(GCC_PG, gpcId);
    DUMP_GPC_REG(GCC_PG1, gpcId);
    DUMP_GPC_REG(GCC_TSL2_CG, gpcId);

    DUMP_GPC_REG(MMU_CTRL, gpcId);
    DUMP_GPC_REG(MMU_PM, gpcId);
    DUMP_GPC_REG(MMU_PM_ADDR_LOW, gpcId);
    DUMP_GPC_REG(MMU_PM_UNIT_MASK, gpcId);
    DUMP_GPC_REG(MMU_PM_REQ_MASK, gpcId);
    DUMP_GPC_REG(MMU_CG, gpcId);
    DUMP_GPC_REG(MMU_CG1, gpcId);
    DUMP_GPC_REG(MMU_PG, gpcId);
    DUMP_GPC_REG(MMU_PG1, gpcId);
    DUMP_GPC_REG(MMU_TLB_CTRL, gpcId);
    DUMP_GPC_REG(MMU_RSRVD, gpcId);
}

void grDumpGrGpcTpcInfo_GF100( LwU32 gpcId, LwU32 grIdx )
{
    LwU32   j;
    LwU32   nTPC;
    LwU32   nPPC;

    dprintf("====================\n");
    dprintf("GPC %d detailed status\n", gpcId);
    dprintf("====================\n");

    nTPC = pGr[indexGpu].grGetNumTpcForGpc(gpcId, grIdx );
    nPPC = pGr[indexGpu].grGetNumPpcForGpc(gpcId, grIdx);

    pGr[indexGpu].grDumpGpcInfo(gpcId, grIdx );

    for (j=0;j<nTPC;j++)
    {
       pGr[indexGpu].grDumpTpcInfo(gpcId, j);
    }

    for (j=0;j<nPPC;j++)
    {
       pGr[indexGpu].grDumpPpcInfo(gpcId, j);
    }
}

LW_STATUS grCheckPipelineStatus_GF100(char* eng, BOOL bFullPrint, LwU32 grIdx)
{
    LwU32   grStatus = 0;
    LwU32   act4;
    LW_STATUS    status = LW_OK;
    LwU32 numActiveFbps = 0;
    LwU32 numActiveGpcs = 0; 
    LwU32 gpcId;
    LW_STATUS    retStatus = LW_OK;
    BOOL    bAnalyze = TRUE;

    if (eng == NULL)
    {
        bAnalyze = FALSE;
    }

    pGr[indexGpu].grGetBusInfo(&numActiveGpcs, NULL, &numActiveFbps, NULL, NULL, grIdx);

    grStatus = GPU_REG_RD32(LW_PGRAPH_STATUS);
    act4 = GPU_REG_RD32(LW_PGRAPH_ACTIVITY4);

    // Individual units status
    PRINT_REG_PD(_PGRAPH, _ACTIVITY*);

    //if any engine busy, _STATE will be BUSY;  set error
    if (grStatus & DRF_DEF(_PGRAPH, _STATUS, _STATE, _BUSY))
    {
       retStatus = LW_ERR_GENERIC;
        dprintf("LW_PGRAPH_STATUS_STATE_BUSY\n");
       if (bAnalyze)
           addUnitErr("\t LW_PGRAPH_STATUS_BUSY\n");
    }
    else
    {
       dprintf("** All engines are IDLE\n");
       if (bAnalyze)
           return retStatus;
    }

    dprintf("====================\n");
    dprintf("Status for BUSY pipeline engines\n");
    dprintf("====================\n");
    // FE Upper
    if (grStatus & DRF_DEF(_PGRAPH, _STATUS, _FE_METHOD_UPPER, _BUSY))
    {
        dprintf("LW_PGRAPH_STATUS_FE_METHOD_UPPER_BUSY\n");
        if (bAnalyze)
        {
            addUnitErr("\t LW_PGRAPH_STATUS_FE_METHOD_UPPER_BUSY\n");
            strcpy(eng, "FE (fe)");
        }
        // There is no LW_PGRAPH_ACTIVITY#_ information for FE_METHOD_UPPER

        status = pGr[indexGpu].grCheckFeMethodStatus();

        if (bAnalyze)
        {
            if (status == LW_ERR_GENERIC)
            {
                dprintf("FE is in invalid state.\n");
                addUnitErr("\t FE is in invalid state: method status is invalid\n");
            }
        }
    }

    // FE Lower
    if (grStatus & DRF_DEF(_PGRAPH, _STATUS, _FE_METHOD_LOWER, _BUSY))
    {
        dprintf("LW_PGRAPH_STATUS_FE_METHOD_LOWER_BUSY\n");
        if (bAnalyze)
        {
            addUnitErr("\t LW_PGRAPH_STATUS_FE_METHOD_LOWER_BUSY\n");
            strcpy(eng, "FE (fe)");
        }

        status = pGr[indexGpu].grCheckFeMethodStatus();

        if (bAnalyze)
        {
            if (status == LW_ERR_GENERIC)
            {
                dprintf("FE is in invalid state.\n");
                addUnitErr("\t FE is in invalid state: method status is invalid\n");
            }
        }
    }

    // FE Funnel
    if (grStatus & DRF_DEF(_PGRAPH, _STATUS, _FE_FUNNEL, _BUSY))
    {
        dprintf("LW_PGRAPH_STATUS_FE_FUNNEL_BUSY.\n");
        if (bAnalyze)
        {
            addUnitErr("\t LW_PGRAPH_STATUS_FE_FUNNEL_BUSY.\n");
            strcpy(eng, "FE (fe)");
        }

        PRINT_REG_PD(_PGRAPH_PRI_FE, _MIW_STATUS);
    }

    // FE Notify
    if (grStatus & DRF_DEF(_PGRAPH, _STATUS, _FE_NOTIFY, _BUSY))
    {
        dprintf("LW_PGRAPH_STATUS_FE_NOTIFY_BUSY\n");
        // No per unit _ACTIVITY# to report for FE_NOTIFY
        if (bAnalyze) {

            addUnitErr("\t LW_PGRAPH_STATUS_FE_NOTIFY_BUSY\n");
            strcpy(eng, "FE (fe)");
        }

        status = pGr[indexGpu].grDumpPriFeSemaphoreState();

        if (bAnalyze)
        {
            if (status == LW_ERR_GENERIC)
            {
                dprintf("FE is in invalid state.\n");
                addUnitErr("\t FE is in invalid state: semaphore status is invalid\n");
            }
        }
    }

    // Semaphore
    if (grStatus & DRF_DEF(_PGRAPH, _STATUS, _SEMAPHORE, _BUSY))
    {
        dprintf("LW_PGRAPH_STATUS_SEMAPHORE_BUSY\n");
        if (bAnalyze)
        {
            addUnitErr("\t LW_PGRAPH_STATUS_SEMAPHORE_BUSY\n");
            strcpy(eng, "FE (fe)");
        }

        status = pGr[indexGpu].grDumpPriFeSemaphoreState();

        if (bAnalyze)
        {
            if (status == LW_ERR_GENERIC)
            {
                dprintf("FE is in invalid state.\n");
                addUnitErr("\t FE is in invalid state: semaphore status is invalid\n");
            }
        }
    }

    // MEMFMT
    if (grStatus & DRF_DEF(_PGRAPH, _STATUS, _MEMFMT, _BUSY))
    {
        dprintf("LW_PGRAPH_STATUS_MEMFMT_BUSY\n");
        if (bAnalyze)
        {
            addUnitErr("\t LW_PGRAPH_STATUS_MEMFMT_BUSY\n");
            strcpy(eng, "MEMFMT ()");
        }
    }

    // CONTEXT SWITCH
    if (grStatus & DRF_DEF(_PGRAPH, _STATUS, _CONTEXT_SWITCH, _BUSY))
    {
        dprintf("LW_PGRAPH_STATUS_CONTEXT_SWITCH_BUSY\n");
        // No per unit _ACTIVITY# to report for CONTEXT_SWITCH
        if (bAnalyze)
        {
            addUnitErr("\t LW_PGRAPH_STATUS_CONTEXT_SWITCH_BUSY\n");
            strcpy(eng, "FE (fecs)");

        }
        pGr[indexGpu].grCheckCtxswStatus(bFullPrint, grIdx);
    }
    //Always print MAILBOX registers as part of detailed report.
    pGr[indexGpu].grDumpFecsMailboxRegisters();

    // PD
    if (grStatus & DRF_DEF(_PGRAPH, _STATUS, _PD, _BUSY))
    {
        dprintf("LW_PGRAPH_STATUS_PD_BUSY\n");
        if (bAnalyze) {
            addUnitErr("\t LW_PGRAPH_STATUS_PD_BUSY\n");
            strcpy(eng, "PD ()");
        }
    }

    // PDB
    if (grStatus & DRF_DEF(_PGRAPH, _STATUS, _PDB, _BUSY))
    {
        dprintf("LW_PGRAPH_STATUS_PDB_BUSY\n");
        PRINT_REG_PD(_PGRAPH_PRI_PDB, _STATUS);
        PRINT_REG_PD(_PGRAPH_PRI_PDB, _STATUS_SCC_IN);
        PRINT_REG_PD(_PGRAPH_PRI_PDB, _STATUS_TASK_CTRL);
        PRINT_REG_PD(_PGRAPH_PRI_PDB, _STATUS_PDB);
        if (bAnalyze) {
            addUnitErr("\t LW_PGRAPH_STATUS_PDB_BUSY\n");
            strcpy(eng, "PDB ()");
        }
    }

    //(SCC)
    if (grStatus & DRF_DEF(_PGRAPH, _STATUS, _SCC, _BUSY))
    {
        dprintf("LW_PGRAPH_STATUS_SCC_BUSY\n");
        if (bAnalyze) {
            addUnitErr("\t LW_PGRAPH_STATUS_SCC_BUSY\n");
            strcpy(eng, "SCC (scc)");
        }
    }

    //RASTWOD
    if (grStatus & DRF_DEF(_PGRAPH, _STATUS, _RASTWOD, _BUSY))
    {
        dprintf("LW_PGRAPH_STATUS_RASTWOD_BUSY\n");
        if (bAnalyze) {
            addUnitErr("\t LW_PGRAPH_STATUS_RASTWOD_BUSY\n");
            strcpy(eng, "RASTWOD ()");
        }
    }

    //SMCARB: Add a section SMCARB for GA100
    pGr[indexGpu].grDumpSmcarbRegisters();

    //SSYNC
    if (grStatus & DRF_DEF(_PGRAPH, _STATUS, _SSYNC, _BUSY))
    {
        dprintf("LW_PGRAPH_STATUS_SSYNC_BUSY\n");
        if (bAnalyze) {
            addUnitErr("\t LW_PGRAPH_STATUS_SSYNC_BUSY\n");
            strcpy(eng, "SSYNC ()");
        }
    }

    //CWD
    if (grStatus & DRF_DEF(_PGRAPH, _STATUS, _CWD, _BUSY))
    {
        dprintf("LW_PGRAPH_STATUS_CWD_BUSY\n");
        if (bAnalyze) {
            addUnitErr("\t LW_PGRAPH_STATUS_CWD_BUSY\n");
            strcpy(eng, "CWD ()");
        }
    }

    //HSHUB
    pGr[indexGpu].grDumpHshubIdleRegisters();

    pGr[indexGpu].grCheckPMMStatus(eng, grStatus, bAnalyze);

    pGr[indexGpu].grCheckBePipelineStatus(eng, grStatus, numActiveFbps);

    // Per GPC ACTIVITY information - very important.  GPC_BUSY is a
    // rollup of status information from the GPCCS (GPC Context
    // Switched) blocks, one per GPC.  Print GPC status for every GPC
    // that's not EMPTY
    if (grStatus & DRF_DEF(_PGRAPH, _STATUS, _GPC, _BUSY))
    {
        LwU32 gpcStatus;

        dprintf("LW_PGRAPH_STATUS_GPC_BUSY\n");

        if ( bAnalyze )
        {
            addUnitErr("\t LW_PGRAPH_STATUS_GPC_BUSY\n");
            strcpy(eng, "GPC ()");
        }

        gpcStatus = act4;

        for ( gpcId = 0 ; gpcId < numActiveGpcs ; gpcId++ )
        {
            if (DRF_VAL(_PGRAPH, _ACTIVITY4, _GPC0, gpcStatus) !=
                LW_PGRAPH_ACTIVITY4_GPC0_EMPTY)
            {
                pGr[indexGpu].grDumpGrGpcTpcInfo(gpcId, grIdx);
            }
            gpcStatus >>= DRF_SIZE(LW_PGRAPH_ACTIVITY4_GPC0);
        }
    }

    // FB
    if (grStatus & DRF_DEF(_PGRAPH, _STATUS, _FB, _BUSY))
    {
        dprintf("LW_PGRAPH_STATUS_FB_BUSY\n");
        // No per unit _ACTIVITY# to report for CONTEXT_SWITCH
        if (bAnalyze) {
            addUnitErr("\t LW_PGRAPH_STATUS_FB_BUSY\n");
            strcpy(eng, "FB ()");
        }
    }

    //SKED.  This is new unit on Kepler so check status through a HAL
    //interface
    pGr[indexGpu].grCheckSkedStatus(eng, bAnalyze);

    //FE_CONST.  This is new unit on Kepler so check status through a HAL
    //interface
    pGr[indexGpu].grCheckFeConstStatus(eng,bAnalyze);

    return retStatus;
}

//-----------------------------------------------------
// grCheckPMMStatus_GF100()
// check status of the PMM/PMMSYS unit, record the "error"
// if necessary, consistent with grCheckPipelineStatus()
//-----------------------------------------------------
void grCheckPMMStatus_GF100(char* eng, LwU32 grStatus, BOOL bAnalyze)
{
    //PMA
    if (grStatus & DRF_DEF(_PGRAPH, _STATUS, _PMA, _BUSY))
    {
        dprintf("LW_PGRAPH_STATUS_PMA_BUSY\n");
        if (bAnalyze)
        {
            addUnitErr("\t LW_PGRAPH_STATUS_PMA_BUSY\n");
            strcpy(eng, "PMA ()");
        }
    }

    //PMMSYS
    if (grStatus & DRF_DEF(_PGRAPH, _STATUS, _PMMSYS, _BUSY))
    {
        dprintf("LW_PGRAPH_STATUS_PMMSYS_BUSY\n");
        if (bAnalyze)
        {
            addUnitErr("\t LW_PGRAPH_STATUS_PMMSYS_BUSY\n");
            strcpy(eng, "PMMSYS ()");
        }
    }
}

//-----------------------------------------------------
// grTestGraphicsState_GF100
//
//-----------------------------------------------------
LW_STATUS grTestGraphicsState_GF100( LwU32 grIdx )
{
    LW_STATUS    status = LW_OK;
    LW_STATUS    retStatus = LW_OK;
    char    eng[72];
    memset(eng, 0, sizeof(eng));

    status = pGr[indexGpu].grCheckGrInterrupts( grIdx );
    if (status == LW_ERR_GENERIC)
    {
        dprintf("Graphics state failed: pending interrupts\n");
        addUnitErr("\t Graphics state failed: pending interrupts\n");
        return LW_ERR_GENERIC;
    }
    else
    {

        dprintf("Checking GR fifo status and activity...\n");

        if (pGr[indexGpu].grCheckGrFifoStatus() == LW_ERR_GENERIC)
        {
            dprintf("ERROR: GR fifo status is in invalid state \n");
            addUnitErr("\t GR fifo status is in invalid state\n");
            retStatus = LW_ERR_GENERIC;
        }

        retStatus = pGr[indexGpu].grCheckPipelineStatus(eng, FALSE, grIdx);
    }

    if (retStatus == LW_ERR_GENERIC)
    {
        dprintf("\n\tEngine that is not idle and is furthest away in "
            "pipeline is: %s\n", eng);
        addUnitErr("\t Engine that is not idle and is furthest away in "
            "pipeline is: %s\n", eng);
    }

    return retStatus;
}

void grGetZlwllInfoLww_GF100( LwU32 grIdx )
{
    LwU32 i;
    LwU32 gpcIdx;
    char  buffer[GR_REG_NAME_BUFFER_LEN];
    LwU32 gpcCount = pGr[indexGpu].grGetNumGpcs( grIdx );

    buffer[0] = '\0';

    dprintf("====================\n");
    dprintf("ZLWLL detailed status\n");
    dprintf("====================\n");

    for (gpcIdx = 0; gpcIdx < gpcCount; ++gpcIdx)
    {
        for (i = 0; i < LW_PGRAPH_PRI_GPC0_ZLWLL_ZCREGION__SIZE_1; i++)
        {
            PRINT_GPC_REG_PD( ZLWLL_ZCREGION(i), gpcIdx );
            PRINT_GPC_REG_PD( ZLWLL_ZCSIZE(i), gpcIdx );
            PRINT_GPC_REG_PD( ZLWLL_ZCSIZE(i), gpcIdx );
            PRINT_GPC_REG_PD( ZLWLL_ZCSTATUS(i), gpcIdx );
        }

        PRINT_GPC_REG_PD( ZLWLL_DEBUG_0, gpcIdx );
        PRINT_GPC_REG_PD( ZLWLL_DEBUG_1, gpcIdx );
    }
}

LwU32 grGetNumActiveGpc_GF100( LwU32 grIdx )
{
    LwU32 data32 = GPU_REG_RD32(LW_PPRIV_MASTER_RING_ENUMERATE_RESULTS_GPC);
    return DRF_VAL(_PPRIV_MASTER, _RING_ENUMERATE_RESULTS_GPC, _COUNT, data32);
}

LwU32 grGetNumActiveFbp_GF100(void)
{
    LwU32 data32 = GPU_REG_RD32(LW_PPRIV_MASTER_RING_ENUMERATE_RESULTS_FBP);
    return DRF_VAL(_PPRIV_MASTER, _RING_ENUMERATE_RESULTS_FBP, _COUNT, data32);
}

/*----------------------------------------------------------------------------
 * static void
 * grDumpConsolidatedReport_GF100( grIdx )
 *      Print a tightly formatted report of chip state, allowing quick
 *      review by SW teams when determining if a bug needs to be
 *      looked at by HW, and by HW teams when determining where to
 *      look next.  HALified because new bit fields for Kepler won't
 *      compile if this is placed in grgf100.c, and fields removed
 *      after Fermi won't compile in grgk104.c (even if guarded).
 *
 * Return Value --
 *      void.
 *
 *----------------------------------------------------------------------------
 */

void grDumpConsolidatedReport_GF100( LwU32 grIdx )
{
    LwU32 grStatus, fbpCount, gpcCount, val, pgraphStatus;
    LwU32   act0, act1, act2, act4;
    char    regName[GR_REG_NAME_BUFFER_LEN];
    char    buffer[GR_REG_NAME_BUFFER_LEN];

    PRINT_REG2_Z(_PGRAPH, _STATUS, GF100);
    PRINT_REG_Z(_PGRAPH_GRFIFO, _STATUS );
    PRINT_REG2_Z(_PGRAPH, _INTR, GF100 );
    PRINT_REG_Z(_PGRAPH, _PRI_FECS_HOST_INT_STATUS );
    PRINT_REG2_Z(_PGRAPH, _PRI_FECS_CTXSW_STATUS_FE_0, GF100 );
    PRINT_REG2_Z(_PGRAPH, _PRI_FECS_CTXSW_STATUS_1, GF100 );
    PRINT_REG2_Z(_PGRAPH, _EXCEPTION, GF100 );

    pGr[indexGpu].grGetBusInfo( &gpcCount, NULL, &fbpCount, NULL, NULL, grIdx );
    pgraphStatus = GPU_REG_RD32(LW_PGRAPH_STATUS);
    act0 = GPU_REG_RD32(LW_PGRAPH_ACTIVITY0);
    act1 = GPU_REG_RD32(LW_PGRAPH_ACTIVITY1);
    act2 = GPU_REG_RD32(LW_PGRAPH_ACTIVITY2);
    act4 = GPU_REG_RD32(LW_PGRAPH_ACTIVITY4);

    // For each unit, if it'd not IDLE in LW_PGRAPH_STATUS, print the
    // associated activity value from LW_PGRAPH_ACTIVITY*

    DUMP_REG(ACTIVITY0);
    if (pgraphStatus & DRF_DEF(_PGRAPH, _STATUS, _PD, _BUSY))
    {
        pGr[indexGpu].grPrintGpuUnitStatus(DRF_VAL(_PGRAPH, _ACTIVITY0, _PD, act0),
                        "PD");
    }
    if (pgraphStatus & DRF_DEF(_PGRAPH, _STATUS, _PDB, _BUSY))
    {
        pGr[indexGpu].grPrintGpuUnitStatus(DRF_VAL(_PGRAPH, _ACTIVITY0, _PDB, act0),
                        "PDB");
    }
    if (pgraphStatus & DRF_DEF(_PGRAPH, _STATUS, _SCC, _BUSY))
    {
        pGr[indexGpu].grPrintGpuUnitStatus(DRF_VAL(_PGRAPH, _ACTIVITY0, _SCC, act0),
                        "SCC");
    }
    if (pgraphStatus & DRF_DEF(_PGRAPH, _STATUS, _RASTWOD, _BUSY))
    {
        pGr[indexGpu].grPrintGpuUnitStatus(DRF_VAL(_PGRAPH, _ACTIVITY0, _RASTWOD, act0),
                        "RASTWOD");
    }
    if (pgraphStatus & DRF_DEF(_PGRAPH, _STATUS, _SSYNC, _BUSY))
    {
        pGr[indexGpu].grPrintGpuUnitStatus(DRF_VAL(_PGRAPH, _ACTIVITY0, _SSYNC, act0),
                        "SSYNC");
    }
    if (pgraphStatus & DRF_DEF(_PGRAPH, _STATUS, _CWD, _BUSY))
    {
        pGr[indexGpu].grPrintGpuUnitStatus(DRF_VAL(_PGRAPH, _ACTIVITY0, _CWD, act0),
                        "CWD");
    }

    DUMP_REG(ACTIVITY1);
    if (pgraphStatus & DRF_DEF(_PGRAPH, _STATUS, _PMA, _BUSY))
    {
        pGr[indexGpu].grPrintGpuUnitStatus(DRF_VAL(_PGRAPH, _ACTIVITY1, _PMA, act1),
                        "PMA");
    }
    if (pgraphStatus & DRF_DEF(_PGRAPH, _STATUS, _PMMSYS, _BUSY))
    {
        pGr[indexGpu].grPrintGpuUnitStatus(DRF_VAL(_PGRAPH, _ACTIVITY1, _PMMSYS, act1),
                        "PMMSYS");
    }
    if (pgraphStatus & DRF_DEF(_PGRAPH, _STATUS, _MEMFMT, _BUSY))
    {
        pGr[indexGpu].grPrintGpuUnitStatus(DRF_VAL(_PGRAPH, _ACTIVITY1, _MEMFMT, act1),
                        "MEMFMT");
    }
    if (pgraphStatus & DRF_DEF(_PGRAPH, _STATUS, _SEMAPHORE, _BUSY))
    {
        pGr[indexGpu].grPrintGpuUnitStatus(DRF_VAL(_PGRAPH, _ACTIVITY1, _SEMAPHORE, act1),
                        "SEMAPHORE");
    }
    if (pgraphStatus & DRF_DEF(_PGRAPH, _STATUS, _FE_FUNNEL, _BUSY))
    {
        pGr[indexGpu].grPrintGpuUnitStatus(DRF_VAL(_PGRAPH, _ACTIVITY1, _FUNNEL, act1),
                        "FE_FUNNEL");
    }

    // Per FBP
    DUMP_REG(ACTIVITY2);
    if (pgraphStatus & DRF_DEF(_PGRAPH, _STATUS, _BE, _BUSY))
    {
        LwU32 i;
        LwU32 fbpStatus = act2;
        for (i=0;i<fbpCount;i++)
        {
            sprintf(regName, "BE%d", i);
            pGr[indexGpu].grPrintGpuUnitStatus( DRF_VAL( _PGRAPH, _ACTIVITY2, _BE0, fbpStatus), regName);
        }
        for (i=0;i<fbpCount;i++)
        {
            fbpStatus >>= DRF_SIZE(LW_PGRAPH_ACTIVITY2_BE0);
            PRINT_BE_REG_Z( i, BECS_BE_ACTIVITY0, GF100 );
        }
    }

    // Per GPC
    DUMP_REG(ACTIVITY4);
    if (pgraphStatus & DRF_DEF(_PGRAPH, _STATUS, _GPC, _BUSY))
    {
        LwU32 i;
        LwU32 gpcStatus = act4;
        for (i=0;i<gpcCount;i++)
        {
            sprintf(regName, "GPC%d", i);
            pGr[indexGpu].grPrintGpuUnitStatus( DRF_VAL(_PGRAPH, _ACTIVITY4, _GPC0, gpcStatus), regName);
            gpcStatus >>= DRF_SIZE(LW_PGRAPH_ACTIVITY4_GPC0);
        }
        pGr[indexGpu].grDumpConsolidatedReportGpc( gpcCount, grIdx );
    }
}

void grDumpConsolidatedReportGpc_GF100( LwU32 numActiveGpc, LwU32 grIdx )
{
    LwU32 tpcId, numActiveTpc, gpcId, grStatus, val;
    char buffer[GR_REG_NAME_BUFFER_LEN];

    for (gpcId = 0 ; gpcId < numActiveGpc ; gpcId++)
    {
        dprintf("====================\n");
        dprintf("GPC %d\n", gpcId);
        dprintf("====================\n");

        numActiveTpc = pGr[indexGpu].grGetNumTpcForGpc(gpcId, grIdx);
        PRINT_GPC_REG_Z( gpcId, GPCCS_GPC_ACTIVITY0, GF100 );
        PRINT_GPC_REG_Z( gpcId, GPCCS_GPC_ACTIVITY1, GF100 );
        PRINT_GPC_REG_Z( gpcId, GPCCS_GPC_ACTIVITY2, GF100 );
        PRINT_GPC_REG_Z( gpcId, GPCCS_GPC_ACTIVITY3, GF100 );

        PRINT_GPC_REG_Z( gpcId, GPCCS_CTXSW_STATUS_1, GF100 );
        PRINT_GPC_REG_Z( gpcId, GPCCS_CTXSW_STATUS_GPC_0, GF100 );

        for (tpcId = 0 ; tpcId < numActiveTpc ; tpcId++)
        {
            dprintf("====================\n");
            dprintf("GPC/TPC %d/%d\n", gpcId, tpcId );
            dprintf("====================\n");

            PRINT_TPC_REG_Z( gpcId, tpcId, TPCCS_TPC_ACTIVITY0, GF100 );
            PRINT_TPC_REG_Z( gpcId, tpcId, PE_STATUS, GF100 );
            PRINT_TPC_REG_Z( gpcId, tpcId, PE_STATUS_EXT, GF100 );            
            PRINT_TPC_REG_Z( gpcId, tpcId, MPC_STATUS, GF100 );
            PRINT_TPC_REG_Z( gpcId, tpcId, MPC_VTG_STATUS, GF100 );
            PRINT_TPC_REG_Z( gpcId, tpcId, MPC_PIX_STATUS, GF100 );
            PRINT_TPC_REG_Z( gpcId, tpcId, SM_INFO_SUBUNIT_STATUS, GF100 );
        }
    }
}

/*!
 * @brief   Function to read a 32bit value from a Register
 *
 * @param[IN]   reg           Register
 *
 * @return  Returns           32bit Value read from the register
 *
 */
LwU32 grReadReg32_GF100
(
    PhysAddr reg
)
{
    return osRegRd32(reg);
}

/*!
 * @brief   Function to write a 32bit value to a Register
 *
 * @param[IN]   reg           Register
 * @param[IN]   data          32bit data to be written
 *
 */
void grWriteReg32_GF100
(
    PhysAddr reg,
    LwU32 data
)
{
    osRegWr32(reg, data);
}

/*!
 * @brief   Function to read a byte from a Register
 *
 * @param[IN]   reg           Register
 *
 * @return  Returns           Byte read from the register
 *
 */
LwU8 grReadReg08_GF100
(
    PhysAddr reg
)
{
    return osRegRd08(reg);
}

/*!
 * @brief   Function to write a byte to a Register
 *
 * @param[IN]   reg           Register
 * @param[IN]   data          Byte to be written
 *
 */
void grWriteReg08_GF100
(
    PhysAddr reg,
    LwU8 data
)
{
    osRegWr08(reg, data);
}
