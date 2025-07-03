/*
 * _LWRM_COPYRIGHT_START_
 *
 * Copyright 2003-2019 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

//*****************************************************
//
// lwwatch WinDbg Extension
// grgf104.c
//
//*****************************************************

//
// includes
//
#include "fermi/gf104/dev_graphics_nobundle.h"
#include "gr.h"
#include "fermi/gf104/hwproject.h"
#include "fermi/gf104/dev_top.h"
#include "fermi/gf104/dev_pri_ringmaster.h"

#include "g_gr_private.h"       // (rmconfig) implementation prototypes

//-----------------------------------------------------
// grGetTexHangSignatureInfoLww_GF104
//
// We need to inspect the failure cases w.r.t 2x Tex Registers
//      Case 1: Both Texins reporting QUIESCENT / STALLED.
//      Case 2: Both downstream TM modules reporting IDLE.
//      Case 3: GCC stalled.
//-----------------------------------------------------

void grGetTexHangSignatureInfoLww_GF104( LwU32 grIdx )
{
    LwU32   texStatus, tmStatus, gccStatus, val;
    LwU32   i, j, data32;
    LwU32   tpcM2Offset;
    LwU32   gpcCount, tpcCount;
    LwBool  case1Failed = TRUE;     //Denotes Case 1 in description
    LwBool  case2Failed = TRUE;     //Denotes Case 2 in description
    LwBool  case3Failed = TRUE;     //Denotes Case 3 in description

    // Get Gpc Count for this Chip
    data32 = GPU_REG_RD32(LW_PPRIV_MASTER_RING_ENUMERATE_RESULTS_GPC);
    gpcCount = DRF_VAL(_PPRIV_MASTER, _RING_ENUMERATE_RESULTS_GPC, _COUNT, data32);

    tpcM2Offset = (LW_PGRAPH_PRI_GPC0_TPC0_TEX_M_2_TEX_SUBUNITS_STATUS -
                   LW_PGRAPH_PRI_GPC0_TPC0_TEX_M_TEX_SUBUNITS_STATUS);

    dprintf("lw: ---------------------------------\n");
    dprintf("lw: ---------------------------------\n");
    dprintf("lw:    TEX Hang Signature Status\n");
    dprintf("lw: ---------------------------------\n");
    dprintf("lw: ---------------------------------\n");
    for (i = 0; i < gpcCount; i++)
    {
        tpcCount = pGr[indexGpu].grGetNumTpcForGpc(i, grIdx);

        for(j = 0; j < tpcCount; j++)
        {
            texStatus = GPU_REG_RD32(LW_PGRAPH_PRI_GPC0_TPC0_TEX_M_TEX_SUBUNITS_STATUS + LW_GPC_PRI_STRIDE * i + LW_TPC_IN_GPC_STRIDE * j);
            val = DRF_VAL(_PGRAPH, _PRI_GPC0_TPC0_TEX_M_TEX_SUBUNITS_STATUS, _TEXIN, texStatus);

            switch (val)
            {
            case LW_PGRAPH_PRI_GPC0_TPC0_TEX_M_TEX_SUBUNITS_STATUS_TEXIN_EMPTY:
                dprintf("lw:  +  LW_PGRAPH_PRI_GPC%d_TPC%d_TEX_M_TEX_SUBUNITS_STATUS_TEXIN_EMPTY \n", i, j);
                case1Failed = FALSE;
                break;

            case LW_PGRAPH_PRI_GPC0_TPC0_TEX_M_TEX_SUBUNITS_STATUS_TEXIN_ACTIVE:
                dprintf("lw:  +  LW_PGRAPH_PRI_GPC%d_TPC%d_TEX_M_TEX_SUBUNITS_STATUS_TEXIN_ACTIVE \n", i, j);
                case1Failed = FALSE;
                break;

            case LW_PGRAPH_PRI_GPC0_TPC0_TEX_M_TEX_SUBUNITS_STATUS_TEXIN_QUIESCENT:
                dprintf("lw:  +  LW_PGRAPH_PRI_GPC%d_TPC%d_TEX_M_TEX_SUBUNITS_STATUS_TEXIN_QUIESCENT \n", i, j);
                break;

            case LW_PGRAPH_PRI_GPC0_TPC0_TEX_M_TEX_SUBUNITS_STATUS_TEXIN_STALLED:
                dprintf("lw:  +  LW_PGRAPH_PRI_GPC%d_TPC%d_TEX_M_TEX_SUBUNITS_STATUS_TEXIN_STALLED \n", i, j);
                break;

            case LW_PGRAPH_PRI_GPC0_TPC0_TEX_M_TEX_SUBUNITS_STATUS_TEXIN_HALTED:
                dprintf("lw:  +  LW_PGRAPH_PRI_GPC%d_TPC%d_TEX_M_TEX_SUBUNITS_STATUS_TEXIN_HALTED \n", i, j);
                case1Failed = FALSE;
                break;

            default:
                case1Failed = FALSE;
                dprintf("lw:  +   LW_PGRAPH_PRI_GPC%d_TPC%d_TEX_M_TEX_SUBUNITS_STATUS_TEXIN_UNKNOWN: 0x%x\n", i, j,
                        DRF_VAL(_PGRAPH, _PRI_GPC0_TPC0_TEX_M_TEX_SUBUNITS_STATUS, _TEXIN, texStatus));
            }

            texStatus = GPU_REG_RD32(LW_PGRAPH_PRI_GPC0_TPC0_TEX_M_TEX_SUBUNITS_STATUS + LW_GPC_PRI_STRIDE * i + LW_TPC_IN_GPC_STRIDE * j + tpcM2Offset);
            val = DRF_VAL(_PGRAPH, _PRI_GPC0_TPC0_TEX_M_2_TEX_SUBUNITS_STATUS, _TEXIN, texStatus);

            switch (val)
            {
            case LW_PGRAPH_PRI_GPC0_TPC0_TEX_M_2_TEX_SUBUNITS_STATUS_TEXIN_EMPTY:
                dprintf("lw:  +  LW_PGRAPH_PRI_GPC%d_TPC%d_TEX_M_2_TEX_SUBUNITS_STATUS_TEXIN_EMPTY \n", i, j);
                case1Failed = FALSE;
                break;

            case LW_PGRAPH_PRI_GPC0_TPC0_TEX_M_2_TEX_SUBUNITS_STATUS_TEXIN_ACTIVE:
                dprintf("lw:  +  LW_PGRAPH_PRI_GPC%d_TPC%d_TEX_M_2_TEX_SUBUNITS_STATUS_TEXIN_ACTIVE \n", i, j);
                case1Failed = FALSE;
                break;

            case LW_PGRAPH_PRI_GPC0_TPC0_TEX_M_2_TEX_SUBUNITS_STATUS_TEXIN_QUIESCENT:
                dprintf("lw:  +  LW_PGRAPH_PRI_GPC%d_TPC%d_TEX_M_2_TEX_SUBUNITS_STATUS_TEXIN_QUIESCENT \n", i, j);
                break;

            case LW_PGRAPH_PRI_GPC0_TPC0_TEX_M_2_TEX_SUBUNITS_STATUS_TEXIN_STALLED:
                dprintf("lw:  +  LW_PGRAPH_PRI_GPC%d_TPC%d_TEX_M_2_TEX_SUBUNITS_STATUS_TEXIN_STALLED \n", i, j);
                break;

            case LW_PGRAPH_PRI_GPC0_TPC0_TEX_M_2_TEX_SUBUNITS_STATUS_TEXIN_HALTED:
                dprintf("lw:  +  LW_PGRAPH_PRI_GPC%d_TPC%d_TEX_M_2_TEX_SUBUNITS_STATUS_TEXIN_HALTED \n", i, j);
                case1Failed = FALSE;
                break;

            default:
                case1Failed = FALSE;
                dprintf("lw:  +   LW_PGRAPH_PRI_GPC%d_TPC%d_TEX_M_2_TEX_SUBUNITS_STATUS_TEXIN_UNKNOWN: 0x%x\n", i, j,
                        DRF_VAL(_PGRAPH, _PRI_GPC0_TPC0_TEX_M_2_TEX_SUBUNITS_STATUS, _TEXIN, texStatus));
            }

            tmStatus = GPU_REG_RD32(LW_PGRAPH_PRI_GPC0_TPC0_TEX_M_TEX_SUBUNITS_STATUS + LW_GPC_PRI_STRIDE * i + LW_TPC_IN_GPC_STRIDE * j);
            val = DRF_VAL(_PGRAPH, _PRI_GPC0_TPC0_TEX_M_TEX_SUBUNITS_STATUS, _TM, tmStatus);

            switch (val)
            {
            case LW_PGRAPH_PRI_GPC0_TPC0_TEX_M_TEX_SUBUNITS_STATUS_TM_EMPTY:
                dprintf("lw:  +  LW_PGRAPH_PRI_GPC%d_TPC%d_TEX_M_TEX_SUBUNITS_STATUS_TM_EMPTY \n", i, j);
                break;

            case LW_PGRAPH_PRI_GPC0_TPC0_TEX_M_TEX_SUBUNITS_STATUS_TM_ACTIVE:
                dprintf("lw:  +  LW_PGRAPH_PRI_GPC%d_TPC%d_TEX_M_TEX_SUBUNITS_STATUS_TM_ACTIVE \n", i, j);
                case2Failed = FALSE;
                break;

            case LW_PGRAPH_PRI_GPC0_TPC0_TEX_M_TEX_SUBUNITS_STATUS_TM_QUIESCENT:
                dprintf("lw:  +  LW_PGRAPH_PRI_GPC%d_TPC%d_TEX_M_TEX_SUBUNITS_STATUS_TM_QUIESCENT \n", i, j);
                case2Failed = FALSE;
                break;

            case LW_PGRAPH_PRI_GPC0_TPC0_TEX_M_TEX_SUBUNITS_STATUS_TM_STALLED:
                dprintf("lw:  +  LW_PGRAPH_PRI_GPC%d_TPC%d_TEX_M_TEX_SUBUNITS_STATUS_TM_STALLED \n", i, j);
                case2Failed = FALSE;
                break;

            case LW_PGRAPH_PRI_GPC0_TPC0_TEX_M_TEX_SUBUNITS_STATUS_TM_HALTED:
                dprintf("lw:  +  LW_PGRAPH_PRI_GPC%d_TPC%d_TEX_M_TEX_SUBUNITS_STATUS_TM_HALTED \n", i, j);
                case2Failed = FALSE;
                break;

            default:
                case2Failed = FALSE;
                dprintf("lw:  +   LW_PGRAPH_PRI_GPC%d_TPC%d_TEX_M_TEX_SUBUNITS_STATUS_TM_UNKNOWN: 0x%x\n", i, j,
                        DRF_VAL(_PGRAPH, _PRI_GPC0_TPC0_TEX_M_TEX_SUBUNITS_STATUS, _TM, texStatus));
            }

            texStatus = GPU_REG_RD32(LW_PGRAPH_PRI_GPC0_TPC0_TEX_M_TEX_SUBUNITS_STATUS + LW_GPC_PRI_STRIDE * i + LW_TPC_IN_GPC_STRIDE * j + tpcM2Offset);
            val = DRF_VAL(_PGRAPH, _PRI_GPC0_TPC0_TEX_M_2_TEX_SUBUNITS_STATUS, _TM, texStatus);

            switch (val)
            {
            case LW_PGRAPH_PRI_GPC0_TPC0_TEX_M_2_TEX_SUBUNITS_STATUS_TM_EMPTY:
                dprintf("lw:  +  LW_PGRAPH_PRI_GPC%d_TPC%d_TEX_M_2_TEX_SUBUNITS_STATUS_TM_EMPTY \n", i, j);
                break;

            case LW_PGRAPH_PRI_GPC0_TPC0_TEX_M_2_TEX_SUBUNITS_STATUS_TM_ACTIVE:
                dprintf("lw:  +  LW_PGRAPH_PRI_GPC%d_TPC%d_TEX_M_2_TEX_SUBUNITS_STATUS_TM_ACTIVE \n", i, j);
                case2Failed = FALSE;
                break;

            case LW_PGRAPH_PRI_GPC0_TPC0_TEX_M_2_TEX_SUBUNITS_STATUS_TM_QUIESCENT:
                dprintf("lw:  +  LW_PGRAPH_PRI_GPC%d_TPC%d_TEX_M_2_TEX_SUBUNITS_STATUS_TM_QUIESCENT \n", i, j);
                case2Failed = FALSE;
                break;

            case LW_PGRAPH_PRI_GPC0_TPC0_TEX_M_2_TEX_SUBUNITS_STATUS_TM_STALLED:
                dprintf("lw:  +  LW_PGRAPH_PRI_GPC%d_TPC%d_TEX_M_2_TEX_SUBUNITS_STATUS_TM_STALLED \n", i, j);
                case2Failed = FALSE;
                break;

            case LW_PGRAPH_PRI_GPC0_TPC0_TEX_M_2_TEX_SUBUNITS_STATUS_TM_HALTED:
                dprintf("lw:  +  LW_PGRAPH_PRI_GPC%d_TPC%d_TEX_M_2_TEX_SUBUNITS_STATUS_TM_HALTED \n", i, j);
                case2Failed = FALSE;
                break;

            default:
                case2Failed = FALSE;
                dprintf("lw:  +   LW_PGRAPH_PRI_GPC%d_TPC%d_TEX_M_2_TEX_SUBUNITS_STATUS_TM_UNKNOWN: 0x%x\n", i, j,
                        DRF_VAL(_PGRAPH, _PRI_GPC0_TPC0_TEX_M_2_TEX_SUBUNITS_STATUS, _TM, texStatus));
            }
        }

        gccStatus = GPU_REG_RD32(LW_PGRAPH_PRI_GPC0_GPCCS_GPC_ACTIVITY1 + LW_GPC_PRI_STRIDE * i);
        val = DRF_VAL(_PGRAPH, _PRI_GPC0_GPCCS_GPC_ACTIVITY1, _GCC, gccStatus);
        dprintf("\n");

        switch (val)
        {
        case LW_PGRAPH_PRI_GPC0_GPCCS_GPC_ACTIVITY1_GCC_EMPTY:
            dprintf("lw:  +  LW_PGRAPH_PRI_GPC%d_GPCCS_GPC_ACTIVITY1_GCC_EMPTY \n", i);
            case3Failed = FALSE;
            break;

        case LW_PGRAPH_PRI_GPC0_GPCCS_GPC_ACTIVITY1_GCC_ACTIVE:
            dprintf("lw:  +  LW_PGRAPH_PRI_GPC%d_GPCCS_GPC_ACTIVITY1_GCC_ACTIVE \n", i);
            case3Failed = FALSE;
            break;

        case LW_PGRAPH_PRI_GPC0_GPCCS_GPC_ACTIVITY1_GCC_PAUSED:
            dprintf("lw:  +  LW_PGRAPH_PRI_GPC%d_GPCCS_GPC_ACTIVITY1_GCC_PAUSED \n", i);
            case3Failed = FALSE;
            break;

        case LW_PGRAPH_PRI_GPC0_GPCCS_GPC_ACTIVITY1_GCC_QUIESCENT:
            dprintf("lw:  +  LW_PGRAPH_PRI_GPC%d_GPCCS_GPC_ACTIVITY1_GCC_QUIESCENT \n", i);
            case3Failed = FALSE;
            break;

        case LW_PGRAPH_PRI_GPC0_GPCCS_GPC_ACTIVITY1_GCC_STALLED:
            dprintf("lw:  +  LW_PGRAPH_PRI_GPC%d_GPCCS_GPC_ACTIVITY1_GCC_STALLED \n", i);
            break;

        case LW_PGRAPH_PRI_GPC0_GPCCS_GPC_ACTIVITY1_GCC_FAULTED:
            dprintf("lw:  +  LW_PGRAPH_PRI_GPC%d_GPCCS_GPC_ACTIVITY1_GCC_FAULTED \n", i);
            case3Failed = FALSE;
            break;

        case LW_PGRAPH_PRI_GPC0_GPCCS_GPC_ACTIVITY1_GCC_HALTED:
            dprintf("lw:  +  LW_PGRAPH_PRI_GPC%d_GPCCS_GPC_ACTIVITY1_GCC_HALTED \n", i);
            case3Failed = FALSE;
            break;

        default:
            case3Failed = FALSE;
            dprintf("lw:  +   LW_PGRAPH_PRI_GPC%d_GPCCS_GPC_ACTIVITY1_GCC_UNKNOWN: 0x%x\n", i,
                    DRF_VAL(_PGRAPH, _PRI_GPC0_GPCCS_GPC_ACTIVITY1, _GCC, gccStatus));
        }
        dprintf("\n");
    }

    if(case1Failed && case2Failed && case3Failed)
    {
        dprintf("lw: ---------------------------------\n");
        dprintf("lw: 2x TEX HANG DETECTED.(Bug 665749)\n");
        dprintf("lw: ---------------------------------\n");
    }
    else
    {
        dprintf("lw: ---------------------------------------------------------\n");
        dprintf("lw: Signature does not match typical 2X TEX Hang.(Bug 665749)\n");
        dprintf("lw: ---------------------------------------------------------\n");
    }
}
