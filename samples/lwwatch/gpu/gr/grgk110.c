/*
 * _LWRM_COPYRIGHT_START_
 *
 * Copyright 2011-2014 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

//*****************************************************
//
// lwwatch WinDbg Extension
// grgk110.c
//
//*****************************************************

//
// includes
//
//#include "kepler/gk110/dev_graphics_nobundle.h"
#include "gr.h"
#include "kepler/gk110/hwproject.h"
#include "kepler/gk110/dev_graphics_nobundle.h"
#include "kepler/gk110/dev_pri_ringmaster.h"
//#include "kepler/gk110/dev_top.h"

#include "g_gr_private.h"       // (rmconfig) implementation prototypes
#include "priv.h"

static void _grGetTexParityCounts(LwU32 gpcOffset, LwU32 tpcOffset, LwU32 gpcCount, LwU32 tpcCount);

typedef enum
{
    PIPE0 = 0,
    PIPE1,
    PIPE2,
    PIPE3,
    NUMPIPES,
} GR_PIPE_NAME;


LwU32 grGetMaxTpcPerGpc_GK110()
{
    return LW_SCAL_LITTER_NUM_TPC_PER_GPC;
}

LwU32 grGetMaxGpc_GK110()
{
    return LW_SCAL_LITTER_NUM_GPCS;
}

LwU32 grGetNumberPesPerGpc_GK110(void)
{
    return LW_SCAL_LITTER_NUM_PES_PER_GPC;
}

/*!
 * @brief Print the Parity Retry Statistics for all the  GPC/TPC
 *
 * @returns None
 */

void grGetTexParityInfo_GK110( LwU32 grIdx )
{

    LwU32 i, j;
    LwU32 gpcCount, tpcCount;
    gpcCount = pGr[indexGpu].grGetNumActiveGpc( grIdx );

    for (i = 0; i < gpcCount; i++)
    {
        tpcCount = pGr[indexGpu].grGetNumTpcForGpc(i, grIdx);

        for (j = 0; j < tpcCount; j++)
        {
            _grGetTexParityCounts((LW_GPC_PRI_STRIDE*i), (LW_TPC_IN_GPC_STRIDE*j), i, j);
        }
    }
}

/*!
 * @brief Print the Parity Retry Statistics for all pipes in a TPC
 *
 * @param[in]   LwU32  gpcOffset
 * @param[in]   LwU32  tpcOffset
 * @param[in]   LwU32  gpcCount
 * @param[in]   LwU32  tpcCount
 *
 * @returns None
 */
static void _grGetTexParityCounts(LwU32 gpcOffset, LwU32 tpcOffset, LwU32 gpcCount, LwU32 tpcCount)
{
    LwU32 originalRouting = 0;
    LwU32 routingField = 0;
    LwU32 texInfo = 0;
    LwU32 texIntrStatus;
    LwU32 pipeIndex = 0;
    LwU32 texParityCount = 0;
    GR_PIPE_NAME pipeName = 0;

    texIntrStatus = GPU_REG_RD32((LW_PGRAPH_PRI_GPC0_TPC0_TEX_M_PARITY + gpcOffset + tpcOffset));
    originalRouting = GPU_REG_RD32((LW_PGRAPH_PRI_GPC0_TPC0_TEX_TRM_DBG + gpcOffset + tpcOffset));

    // route to each PIPE and print the count values
    while (pipeName < NUMPIPES)
    {
        switch (pipeName)
        {
            case (PIPE0):
                routingField = FLD_SET_DRF(_PGRAPH, _PRI_GPC0_TPC0_TEX_TRM_DBG,
                               _ROUTING, _PIPE0, originalRouting);
                break;

            case (PIPE1):
                routingField = FLD_SET_DRF(_PGRAPH, _PRI_GPC0_TPC0_TEX_TRM_DBG,
                               _ROUTING, _PIPE1, originalRouting);
                break;

            case (PIPE2):
                routingField = FLD_SET_DRF(_PGRAPH, _PRI_GPC0_TPC0_TEX_TRM_DBG,
                               _ROUTING, _PIPE2, originalRouting);
                break;

            case (PIPE3):
                routingField = FLD_SET_DRF(_PGRAPH, _PRI_GPC0_TPC0_TEX_TRM_DBG,
                               _ROUTING, _PIPE3, originalRouting);
                break;

            // eliminate warning
            case (NUMPIPES):
                break;
        }

        GPU_REG_WR32((LW_PGRAPH_PRI_GPC0_TPC0_TEX_TRM_DBG + gpcOffset + tpcOffset), routingField);
        texInfo = GPU_REG_RD32((LW_PGRAPH_PRI_GPC0_TPC0_TEX_M_PARITY + gpcOffset + tpcOffset));
        texParityCount = DRF_VAL(_PGRAPH_PRI_GPC0_TPC0_TEX_M, _PARITY, _COUNT, texInfo);

        dprintf("lw: LW_LW_PGRAPH_PRI_GPC%d_TPC%d_TEX_M_PARITY: 0x%x\n", gpcCount,
                 tpcCount, texInfo);
        dprintf("lw: GPC                %d \n", gpcCount);
        dprintf("lw: TPC                %d \n", tpcCount);
        dprintf("lw: PIPE               %d \n", pipeIndex);
        dprintf("lw: Parity Corrections %d \n", texParityCount);

        pipeName = ++pipeIndex;
    }
    GPU_REG_WR32((LW_PGRAPH_PRI_GPC0_TPC0_TEX_TRM_DBG + gpcOffset + tpcOffset),
             originalRouting);
}
