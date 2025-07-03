/* _LWRM_COPYRIGHT_BEGIN_
*
* Copyright 2012-2015 by LWPU Corporation.  All rights reserved.  All information
* contained herein is proprietary and confidential to LWPU Corporation.  Any
* use, reproduction, or disclosure without the written permission of LWPU
* Corporation is prohibited.
*
* _LWRM_COPYRIGHT_END_
*/

//
// includes
//

#include <time.h>

#include "pascal/gp100/hwproject.h"
#include "chip.h"
// HW specific files are in //sw/dev/gpu_drv/chips_a/drivers/common/inc/hwref/<family>/<chip>
#include "pascal/gp100/dev_graphics_nobundle.h"
#include "pascal/gp100/dev_fb.h"
#include "pascal/gp100/dev_master.h" // For PMC_BOOT_0
#include "pascal/gp100/dev_pri_ringstation_sys.h"

#include "gr.h"
#include "g_gr_private.h"       // (rmconfig) implementation prototypes


LwU32 grGetNumberPesPerGpc_GP100(void)
{
    return LW_SCAL_LITTER_NUM_PES_PER_GPC;
}

LwU32 grGetMaxGpc_GP100()
{
    return LW_SCAL_LITTER_NUM_GPCS;
}

LwU32 grGetMaxFbp_GP100(void)
{
    return LW_SCAL_LITTER_NUM_FBPS;
}

LwU32 grGetMaxTpcPerGpc_GP100()
{
    return LW_SCAL_LITTER_NUM_TPC_PER_GPC;
}

void grDumpGpcInfo_GP100 ( LwU32 gpcId, LwU32 grIdx )
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

    if (IsGP100())
    {
        // Exists on GP100, but not on GP10x.
        DUMP_GPC_REG(PROP_EZ_ZPASS_FAIL_LO, gpcId);
        DUMP_GPC_REG(PROP_EZ_ZPASS_FAIL_HI, gpcId);
    }

    DUMP_GPC_REG(PROP_ZPASS_CNT_LO, gpcId);
    DUMP_GPC_REG(PROP_ZPASS_CNT_HI, gpcId);

    if (!IsGV100orLater())
    {
        DUMP_GPC_REG(PROP_IEEE_CLEAN_ZETA_TARGET, gpcId);
    }

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
    PRINT_GPC_REG_PD(GPM_PD_STATUS2, gpcId);

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

    PRINT_GPC_REG_PD(SWDX_DEBUG, gpcId);

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

void grDumpPpcInfo_GP100(LwU32 gpcId, LwU32 ppcId)
{
    char buffer[GR_REG_NAME_BUFFER_LEN];
    LwU32 apInd, apertureRegVal, addr32, data32;
    buffer[0] = '\0';

    dprintf("====================\n");
    dprintf("GPC/PPC %d/%d\n", gpcId, ppcId);
    dprintf("====================\n");

    DUMP_PPC_REG(CBM_DEBUG, gpcId, ppcId);

    /* Implementation of bug #1485936:
     * Set CBM_STATUS_INDEX_READINCR to ENABLED and CBM_STATUS_INDEX_INDEX to MIN.
     * Read the CBM_STATUS_APERTURE register from CBM_STATUS_INDEX_INDEX_MIN to CBM_STATUS_INDEX_INDEX_MAX.
     * Decode the CBM_STATUS_APERTURE register wrt the index during each iteration.
     * The macro GR_DUMP_APERTURE_FIELDS is defined in inc/gr.h for this purpose.
    */

    // Write READINCR to ENABLED
    addr32 = PPC_REG_ADDR( CBM_STATUS_INDEX, gpcId, ppcId );
    data32 = GPU_REG_RD32( addr32 );
    data32 &= ~DRF_SHIFTMASK( LW_PGRAPH_PRI_GPC0_PPC0_CBM_STATUS_INDEX_READINCR);
    data32 |= DRF_NUM( _PGRAPH, _PRI_GPC0_PPC0_CBM_STATUS_INDEX, _READINCR, LW_PGRAPH_PRI_GPC0_PPC0_CBM_STATUS_INDEX_READINCR_ENABLED );
    GPU_REG_WR32( addr32, data32 );

    // Write INDEX to MIN
    data32 = GPU_REG_RD32( addr32 );
    data32 &= ~DRF_SHIFTMASK( LW_PGRAPH_PRI_GPC0_PPC0_CBM_STATUS_INDEX_INDEX);
    data32 |= DRF_NUM( _PGRAPH, _PRI_GPC0_PPC0_CBM_STATUS_INDEX, _INDEX, LW_PGRAPH_PRI_GPC0_PPC0_CBM_STATUS_INDEX_INDEX_MIN );
    GPU_REG_WR32( addr32, data32 );

    PRINT_PPC_REG_PD (CBM_STATUS_INDEX, gpcId, ppcId);
    for (apInd = DRF_DEF ( _PGRAPH, _PRI_GPC0_PPC0_CBM_STATUS_INDEX, _INDEX, _MIN ) ; apInd <= DRF_DEF ( _PGRAPH, _PRI_GPC0_PPC0_CBM_STATUS_INDEX, _INDEX, _MAX ); apInd ++) {
        dprintf ("----------------\nAperture Index %u\n----------------\n", apInd);
        apertureRegVal = GPU_REG_RD32 ( PPC_REG_ADDR ( CBM_STATUS_APERTURE, gpcId, ppcId ) );
        DUMP_PPC_REG (CBM_STATUS_APERTURE, gpcId, ppcId);
        GR_DUMP_APERTURE_FIELDS (apertureRegVal, apInd);
    }

    DUMP_PPC_REG(CBM_PM, gpcId, ppcId);
    DUMP_PPC_REG(CBM_CG, gpcId, ppcId);
    DUMP_PPC_REG(CBM_CG1, gpcId, ppcId);
    DUMP_PPC_REG(CBM_PG, gpcId, ppcId);
    DUMP_PPC_REG(CBM_PG1, gpcId, ppcId);
    if (! IsGM107orLater())
    {
        PRINT_PPC_REG_PD(CBM_STATUS_CBM, gpcId, ppcId);
    }
    PRINT_PPC_REG_PD(PES_STATUS, gpcId, ppcId);
    PRINT_PPC_REG_PD(PES_VSC_STATUS, gpcId, ppcId);
}

void grDumpMmeHwwEsr_GP100(LwU32 data32)
{
    if (!(data32 & DRF_DEF(_PGRAPH, _PRI_MME_HWW_ESR, _INFO_PC_VALID,_INIT)))
            {
                dprintf("            _PRI_MME_HWW_ESR_INFO_PC: 0x%x\n",
                     DRF_VAL(_PGRAPH_PRI, _MME_HWW_ESR_INFO, _PC, data32));
            }
}

// Need a GP100 version since register LW_PFB_NISO_CFG1(field _DEBUG_BUS_CONFIG/EN) which is used to config LW_PRB_NISO_DEBUG is defined starting from pascal.
//
void grDumpNisoHubStatus_GP100(void)
{
    LwU32 data32;
    LwU32 i;
    LwU32 niso_debug_config_max;

    //LW_PFB_NISO_STATUS
    dprintf("====================\n");
    dprintf("NISO HUB detailed status\n");
    dprintf("====================\n");
    PRINT_REG_PD(_PFB_NISO, _DEBUG);


    // A mask for the CONFIG field indicates how many configs to
    // iterate on.  See pri_fb.ref in the chip manuals.
    niso_debug_config_max = DRF_MASK( LW_PFB_NISO_CFG1_DEBUG_BUS_CONFIG );
    dprintf("Setting _CFG1_DEBUG_BUS_CONFIG values (0..%d) and then reading _NISO_DEBUG\n",
        niso_debug_config_max);

    for (i=0; i<= niso_debug_config_max; i++)
    {
        data32 = GPU_REG_RD32(LW_PFB_NISO_CFG1);
        data32 &= ~DRF_SHIFTMASK(LW_PFB_NISO_CFG1_DEBUG_BUS_CONFIG);
        data32 |= DRF_NUM(_PFB, _NISO_CFG1, _DEBUG_BUS_CONFIG, i);
        data32 |= DRF_NUM(_PFB, _NISO_CFG1, _DEBUG_BUS_EN, LW_PFB_NISO_CFG1_DEBUG_BUS_EN_ON );
        dprintf("Writing LW_PFB_NISO_CFG1_DEBUG_BUS_CONFIG=%d\n", i );
        GPU_REG_WR32(LW_PFB_NISO_CFG1, data32);

        PRINT_REG_PD(_PFB, _NISO_CFG1);
        PRINT_REG_PD(_PFB, _NISO_DEBUG);
    }

    // Clear the debug bus enable
    data32 &= ~DRF_SHIFTMASK(LW_PFB_NISO_CFG1_DEBUG_BUS_EN);
    data32 |= DRF_NUM(_PFB, _NISO_CFG1, _DEBUG_BUS_EN, LW_PFB_NISO_CFG1_DEBUG_BUS_EN_OFF );
    GPU_REG_WR32(LW_PFB_NISO_CFG1, data32);
}

//compared to function in grgk104.c, remove check for below two registers as they are removed in PASCAL. LW_PGRAPH_PRI_FE_GO_IDLE_ON_STATUS && LW_PGRAPH_PRI_FE_GO_IDLE_CHECK
void grDumpDetailedStatus_GP100( BOOL bFullPrint, LwU32 grIdx )
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
    
    PRINT_REG_PD(_PGRAPH_PRI_FE, _GO_IDLE_TIMEOUT);
    
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

void grGetStatus_GP100(LwBool bFullPrint, LwBool bForce, LwU32 grIdx)
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
    dprintf( "Chip %s, ARCHITECTURE 0x%x, MAJOR_REVISION 0x%x,  MINOR_REVISION 0x%x, IMPLEMENTATION 0x%x\n",
             getLwhalImplName(hal.halImpl),
             DRF_VAL( _PMC, _BOOT_0, _ARCHITECTURE, val ),
             DRF_VAL( _PMC, _BOOT_0, _MAJOR_REVISION, val ),
             DRF_VAL( _PMC, _BOOT_0, _MINOR_REVISION, val ),
             DRF_VAL( _PMC, _BOOT_0, _IMPLEMENTATION, val )
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




