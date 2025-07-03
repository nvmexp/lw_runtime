/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2016 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

//*****************************************************
//
// lwwatch WinDbg Extension
// grtu102.c
//
//*****************************************************

//
// includes
//
#include <time.h>

#include "chip.h"
// HW specific files are in //sw/dev/gpu_drv/chips_a/drivers/common/inc/hwref/
#include "turing/tu102/hwproject.h"
#include "turing/tu102/dev_graphics_nobundle.h"
#include "turing/tu102/dev_gpc_no_tpc.h"
#include "turing/tu102/dev_tpc.h"

#include "inst.h"
#include "print.h"
#include "utils/lwassert.h"
#include "gpuanalyze.h"
#include "gr.h"

#include "g_gr_private.h"       // (rmconfig) implementation prototypes

LwU32 grGetNumberPesPerGpc_TU102(void)
{
    return LW_SCAL_LITTER_NUM_PES_PER_GPC;
}

LwU32 grGetMaxTpcPerGpc_TU102()
{
    return LW_SCAL_LITTER_NUM_TPC_PER_GPC;
}

/*----------------------------------------------------------------------------
 * static void
 * grDumpConsolidatedReport_TU102()
 *      Print a tightly formatted report of chip state, allowing quick
 *      review by SW teams when determining if a bug needs to be
 *      looked at by HW, and by HW teams when determining where to
 *      look next.
 *
 * Return Value --
 *      void.
 *
 *----------------------------------------------------------------------------
 */

void grDumpConsolidatedReport_TU102( LwU32 grIdx )
{
    LwU32   grStatus, fbpCount, gpcCount, val, pgraphStatus;
    LwU32   act0, act1, act2, act3, act4;
    char    regName[GR_REG_NAME_BUFFER_LEN];
    char    buffer[GR_REG_NAME_BUFFER_LEN];

    PRINT_REG2_Z(_PGRAPH, _STATUS, TU102);
    PRINT_REG_Z(_PGRAPH_GRFIFO, _STATUS );
    PRINT_REG2_Z(_PGRAPH, _INTR, GV100 );
    PRINT_REG2_Z(_PGRAPH, _PRI_FECS_HOST_INT_STATUS, TU102 );
    PRINT_REG2_Z(_PGRAPH, _PRI_FECS_CTXSW_STATUS_FE_0, TU102 );
    PRINT_REG2_Z(_PGRAPH, _PRI_FECS_CTXSW_STATUS_1, GV100 );
    PRINT_REG2_Z(_PGRAPH, _EXCEPTION, GV100 );

    pGr[indexGpu].grGetBusInfo( &gpcCount, NULL, &fbpCount, NULL, NULL, grIdx );
    pgraphStatus = GPU_REG_RD32(LW_PGRAPH_STATUS);
    act0 = GPU_REG_RD32(LW_PGRAPH_ACTIVITY0);
    act1 = GPU_REG_RD32(LW_PGRAPH_ACTIVITY1);
    act2 = GPU_REG_RD32(LW_PGRAPH_ACTIVITY2);
    act3 = GPU_REG_RD32(LW_PGRAPH_ACTIVITY3);
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
    if (pgraphStatus & DRF_DEF(_PGRAPH, _STATUS, _SKED, _BUSY))
    {
        pGr[indexGpu].grPrintGpuUnitStatus(DRF_VAL(_PGRAPH, _ACTIVITY0, _SKED, act0),
                        "SKED");
    }


    DUMP_REG(ACTIVITY1);


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
    if (pgraphStatus & DRF_DEF(_PGRAPH, _STATUS, _FE_CONST, _BUSY))
    {
        pGr[indexGpu].grPrintGpuUnitStatus(DRF_VAL(_PGRAPH, _ACTIVITY1, _FECONST, act1),
                        "FECONST");
    }
    if (pgraphStatus & DRF_DEF(_PGRAPH, _STATUS, _TPC_MGR, _BUSY))
    {
        pGr[indexGpu].grPrintGpuUnitStatus(DRF_VAL(_PGRAPH, _ACTIVITY1, _TPCMGR, act1),
                        "TPCMGR");
    }


    // Per FBP
    DUMP_REG(ACTIVITY2);
    if (pgraphStatus & DRF_DEF(_PGRAPH, _STATUS, _BE, _BUSY))
    {
        LwU32 i;
        LwU32 fbpStatus = act2;
        for (i=0;i<fbpCount*2 && i<8;i++)
        {
            sprintf(regName, "BE%d", i);
            pGr[indexGpu].grPrintGpuUnitStatus( DRF_VAL( _PGRAPH, _ACTIVITY2, _BE0, fbpStatus), regName);
        }
        for (i=0;i<fbpCount*2 && i<8;i++)
        {
            fbpStatus >>= DRF_SIZE(LW_PGRAPH_ACTIVITY2_BE0);
            PRINT_BE_REG_Z( i, BECS_BE_ACTIVITY0, GV100 );
        }
    }

    // Per FBP
    DUMP_REG(ACTIVITY3);
    if (pgraphStatus & DRF_DEF(_PGRAPH, _STATUS, _BE, _BUSY))
    {
        LwU32 i;
        LwU32 fbpStatus = act3;
        for (i=8;i<fbpCount*2 && i<16;i++)
        {
            sprintf(regName, "BE%d", i);
            pGr[indexGpu].grPrintGpuUnitStatus( DRF_VAL( _PGRAPH, _ACTIVITY3, _BE8, fbpStatus), regName);
        }
        for (i=8;i<fbpCount*2 && i<16;i++)
        {

            fbpStatus >>= DRF_SIZE(LW_PGRAPH_ACTIVITY3_BE8);
            PRINT_BE_REG_Z( i, BECS_BE_ACTIVITY0, GV100 );
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

void grDumpConsolidatedReportGpc_TU102( LwU32 numActiveGpc, LwU32 grIdx )
{
    LwU32 tpcId, numActiveTpc, gpcId, grStatus, val;
    LwU32 data32;
    char buffer[GR_REG_NAME_BUFFER_LEN];
    GR_IO_APERTURE *pGpcAperture;
    GR_IO_APERTURE *pTpcAperture;
    LW_STATUS status;

    for (gpcId = 0 ; gpcId < numActiveGpc ; gpcId++)
    {
        LW_ASSERT_OK_OR_ELSE(status,
            GR_GET_APERTURE(&grApertures[indexGpu], &pGpcAperture, GR_UNIT_TYPE_GPC, gpcId),
            return);
        dprintf("====================\n");
        dprintf("GPC %d\n", gpcId);
        dprintf("====================\n");

        numActiveTpc = pGr[indexGpu].grGetNumTpcForGpc(gpcId, grIdx);
        PRINT_GPC_APERTURE_REG_Z( pGpcAperture, GPCCS_GPC_ACTIVITY0, GV100 );
        PRINT_GPC_APERTURE_REG_Z( pGpcAperture, GPCCS_GPC_ACTIVITY1, GV100 );
        PRINT_GPC_APERTURE_REG_Z( pGpcAperture, GPCCS_GPC_ACTIVITY2, GV100 );
        PRINT_GPC_APERTURE_REG_Z( pGpcAperture, GPCCS_GPC_ACTIVITY3, GV100 );

        PRINT_GPC_APERTURE_REG_Z( pGpcAperture, GPCCS_CTXSW_STATUS_1, GV100 );
        PRINT_GPC_APERTURE_REG_Z( pGpcAperture, GPCCS_CTXSW_STATUS_GPC_0, GV100 );

        for (tpcId = 0 ; tpcId < numActiveTpc ; tpcId++)
        {
            LW_ASSERT_OK_OR_ELSE(status,
                GR_GET_APERTURE(&grApertures[indexGpu], &pTpcAperture, GR_UNIT_TYPE_GPC, gpcId,
                    GR_UNIT_TYPE_TPC, tpcId),
                return);
            
            dprintf("====================\n");
            dprintf("GPC/TPC %d/%d\n", gpcId, tpcId );
            dprintf("====================\n");

            PRINT_TPC_APERTURE_REG_Z( pTpcAperture, TPCCS_TPC_ACTIVITY0, GV100 );
            PRINT_TPC_APERTURE_REG_Z( pTpcAperture, PE_STATUS, GM200 );
            PRINT_TPC_APERTURE_REG_Z( pTpcAperture, MPC_STATUS, GV100 );
            PRINT_TPC_APERTURE_REG_Z( pTpcAperture, MPC_VTG_STATUS, GV100 );
            PRINT_TPC_APERTURE_REG_Z( pTpcAperture, MPC_PIX_STATUS, GV100 );

            data32 = REG_RD32( &pTpcAperture->aperture, LW_PTPC_PRI_TEX_M_ROUTING );
            data32 &= ~DRF_SHIFTMASK( LW_PGRAPH_PRI_GPC0_TPC0_TEX_M_ROUTING_SEL );
            data32 |= DRF_NUM( _PGRAPH, _PRI_GPC0_TPC0_TEX_M_ROUTING, _SEL,
                LW_PGRAPH_PRI_GPC0_TPC0_TEX_M_ROUTING_SEL_PIPE0 );
            REG_WR32( &pTpcAperture->aperture, LW_PTPC_PRI_TEX_M_ROUTING, data32 );
            dprintf("====================\n");
            dprintf("GPC/TPC/SM %d/%d/0\n", gpcId, tpcId );
            dprintf("====================\n");
            PRINT_TPC_APERTURE_REG_Z( pTpcAperture, SM_STATUS, TU102 );

            data32 = REG_RD32( &pTpcAperture->aperture, LW_PTPC_PRI_TEX_M_ROUTING );
            data32 &= ~DRF_SHIFTMASK( LW_PGRAPH_PRI_GPC0_TPC0_TEX_M_ROUTING_SEL );
            data32 |= DRF_NUM( _PGRAPH, _PRI_GPC0_TPC0_TEX_M_ROUTING, _SEL,
                LW_PGRAPH_PRI_GPC0_TPC0_TEX_M_ROUTING_SEL_PIPE1 );
            REG_WR32( &pTpcAperture->aperture, LW_PTPC_PRI_TEX_M_ROUTING, data32 );
            dprintf("====================\n");
            dprintf("GPC/TPC/SM %d/%d/1\n", gpcId, tpcId );
            dprintf("====================\n");
            PRINT_TPC_APERTURE_REG_Z( pTpcAperture, SM_STATUS, TU102 );
        }
    }
}


//-----------------------------------------------------
// grCheckFeMethodStatus_TU102
//
//-----------------------------------------------------
LW_STATUS grCheckFeMethodStatus_TU102(void)
{

    PRINT_REG_PD(_PGRAPH_PRI_FE, _SETOBJ_STATE*);
    PRINT_REG_PD(_PGRAPH_PRI_FE, _DECODE_STATE*);

    PRINT_REG_PD(_PGRAPH_PRI_FE, _PERFMON);

    return LW_OK;
}

void grEnableFePerfmonReporting_TU102()
{
    LwU32 fePerfmon;
    //
    // program LW_PGRAPH_PRI_FE_PERFMON register to enable the perfmon
    // reporting if not already enabled
    //
    fePerfmon  =  GPU_REG_RD32(LW_PGRAPH_PRI_FE_PERFMON);
    if (!(fePerfmon & (LW_PGRAPH_PRI_FE_PERFMON_ENABLE_ENABLE <<
               DRF_SHIFT(LW_PGRAPH_PRI_FE_PERFMON_ENABLE))))
    {
        fePerfmon |=  (LW_PGRAPH_PRI_FE_PERFMON_ENABLE_ENABLE <<
                    DRF_SHIFT(LW_PGRAPH_PRI_FE_PERFMON_ENABLE));
        GPU_REG_WR32(LW_PGRAPH_PRI_FE_PERFMON, fePerfmon);
    }
}

LwU32 grGetPteKindFromPropHwwEsr_TU102(LwU32 regVal)
{
    return DRF_VAL(_PGPC, _PRI_PROP_HWW_ESR_STATE, _PTE_KIND, regVal);    
}

/*!
 *  grDumpTPCMpcExceptionState_TU102
 */
void grDumpTpcMpcExceptionState_TU102(LwU32 gpcIdx, LwU32 tpcIdx) 
{
    GR_IO_APERTURE *pTpcAperture;
    LW_STATUS status;
    LwU32 hwwEsr;

    LW_ASSERT_OK_OR_ELSE(status,
        GR_GET_APERTURE(&grApertures[indexGpu], &pTpcAperture, GR_UNIT_TYPE_GPC, gpcIdx,
            GR_UNIT_TYPE_TPC, tpcIdx),
        return);

    hwwEsr = REG_RD32(&pTpcAperture->aperture, LW_PTPC_PRI_MPC_HWW_ESR);

    dprintf("Graphics MPC Exception: LW_PGRAPH_PRI_GPC%d_TPC%d_MPC_HWW_ESR 0x%x\n", gpcIdx, tpcIdx, hwwEsr);

    if (hwwEsr & DRF_DEF(_PTPC, _PRI_MPC_HWW_ESR, _VTG_ERR, _PENDING))
    {
        dprintf("Graphics MPC Exception Type: VTG_ERR\n");
    }

    if (hwwEsr & DRF_DEF(_PTPC, _PRI_MPC_HWW_ESR, _UNEXPECTED_GRAPHICS_WORK, _PENDING))
    {
        dprintf("Graphics MPC Exception Type: UNEXPECTED_GRAPHICS_WORK\n");
    }

    if (hwwEsr & DRF_DEF(_PTPC, _PRI_MPC_HWW_ESR, _UNEXPECTED_COMPUTE_WORK, _PENDING))
    {
        dprintf("Graphics MPC Exception Type: UNEXPECTED_COMPUTE_WORK\n");
    }

    if (hwwEsr & DRF_DEF(_PTPC, _PRI_MPC_HWW_ESR, _COMPUTE_WORK_OVERFLOW, _PENDING))
    {
        dprintf("Graphics MPC Exception Type: COMPUTE_WORK_OVERFLOW\n");
    }

    if (hwwEsr & DRF_DEF(_PTPC, _PRI_MPC_HWW_ESR, _SKEDCHECK19_CTA_REGISTER_CONSUMPTION, _PENDING))
    {
        dprintf("Graphics MPC Exception Type: SKEDCHECK19_CTA_REGISTER_CONSUMPTION\n");
    }

    if (hwwEsr & DRF_DEF(_PTPC, _PRI_MPC_HWW_ESR, _SKEDCHECK34_OOB_SHARED_MEM_CONFIG, _PENDING))
    {
        dprintf("Graphics MPC Exception Type: SKEDCHECK34_OOB_SHARED_MEM_CONFIG\n");
    }
}

/*!
 *  grDumpGpccsExceptionState_TU102
 */
void grDumpGpccsExceptionState_TU102(LwU32 gpcCounter, LwU32 tpcCounter) 
{
    LW_STATUS status;
    GR_IO_APERTURE *pTpcAperture;
    LwU32 tpccsException = 0;
    
    LW_ASSERT_OK_OR_ELSE(status,
        GR_GET_APERTURE(&grApertures[indexGpu], &pTpcAperture, GR_UNIT_TYPE_GPC, gpcCounter,
            GR_UNIT_TYPE_TPC, tpcCounter),
        return);

    tpccsException = REG_RD32(&pTpcAperture->aperture, LW_PTPC_PRI_TPCCS_TPC_EXCEPTION);
    
    dprintf("LW_PGRAPH_PRI_GPC%d_TPC%d_TPCCS_TPC_EXCEPTION:    0x%x\n", gpcCounter, tpcCounter, tpccsException);

    // is it SM exception?
    if (tpccsException & DRF_DEF(_PTPC, _PRI_TPCCS_TPC_EXCEPTION, _SM, _PENDING))
    {
        pGr[indexGpu].grDumpTPCSMExceptionState(gpcCounter, tpcCounter);
    }

    // is it PE exception?
    if (tpccsException & DRF_DEF(_PTPC, _PRI_TPCCS_TPC_EXCEPTION, _PE, _PENDING))
    {
        pGr[indexGpu].grDumpTpcPeExceptionState(gpcCounter, tpcCounter);
    }

    // is it MPC exception?
    if (tpccsException & DRF_DEF(_PTPC, _PRI_TPCCS_TPC_EXCEPTION, _MPC, _PENDING))
    {
        pGr[indexGpu].grDumpTpcMpcExceptionState(gpcCounter, tpcCounter);
    }
}

