/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2014 by LWPU Corporation.  All rights reserved.  All information
 * contained herein is proprietary and confidential to LWPU Corporation.  Any
 * use, reproduction, or disclosure without the written permission of LWPU
 * Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

//*****************************************************
//
// disp0203.c - Disp V02_03 display routines 
// 
//*****************************************************

#include <stdio.h>
#include <string.h>
#include <ctype.h>

#include "inc/disp.h"
#include "disp/v02_03/dev_disp.h"
#include "g_disp_private.h"
#include "dpaux.h"


/*!
 *  Function to print configuration of specified displayport.
 *
 *  @param[in]  port        Specified AUX port.
 *  @param[in]  sorIndex    Specified SOR.
 *  @param[in]  dpIndex     Specified sublink.
 */
void dispDisplayPortInfo_v02_03
(
    LwU32 port,
    LwU32 sorIndex,
    LwU32 dpIndex
)
{
    LwU32 reg, headMask;

    if (pDpaux[indexGpu].dpauxGetHpdStatus(port))
    {
        dprintf("================================================================================\n");
        dprintf("%-55s: %s\n\n", "LW_PMGR_DP_AUXSTAT_HPD_STATUS", "PLUGGED");
    }
    else
    {
        dprintf("ERROR: %s: DP not plugged in. Bailing out early\n",
            __FUNCTION__);
        return;
    }

    // Get right SOR & sublink index.
    if (dpIndex == LW_MAX_SUBLINK)
    {
        LwU32 i, j;
        LwU32 link[LW_MAX_SUBLINK];

        for (i = 0; i < LW_MAX_SUBLINK; i++)
            link[i] = PADLINK_NONE;

        for (i = 0; i < pDisp[indexGpu].dispGetNumOrs(LW_OR_SOR); i++)
        {
            if (pDisp[indexGpu].dispGetLinkBySor(i, link))
            {
                for (j = 0; j < LW_MAX_SUBLINK; j++)
                {
                    if (pDisp[indexGpu].dispGetAuxPortByLink(link[j]) == port)
                    {
                        sorIndex = i;
                        dpIndex = j;
                        break;
                    }
                }
                if (dpIndex != LW_MAX_SUBLINK)
                    break;
            }
        }
        if (dpIndex == LW_MAX_SUBLINK)
        {
            dprintf("ERROR: %s: Can't get corrusponding SOR & sublink.\n",
                __FUNCTION__);
            return;
        }
    }

    dprintf("Tx:\n");
    dprintf("%-55s: %d\n", "sorIndex", sorIndex);
    dprintf("%-55s: %d\n", "dpIndex", dpIndex);

    reg = GPU_REG_RD32(LW_PDISP_SOR_DP_LINKCTL(sorIndex, dpIndex));

    dprintf("%-55s: %s\n", "LW_PDISP_SOR_DP_LINKCTL_ENABLE",
        DRF_VAL(_PDISP, _SOR_DP_LINKCTL, _ENABLE, reg) ? "YES" : "NO");

    dprintf("%-55s: %s\n", "LW_PDISP_SOR_DP_LINKCTL_ENHANCEDFRAME",
        DRF_VAL(_PDISP, _SOR_DP_LINKCTL, _ENHANCEDFRAME, reg) ?
        "ENABLED" : "DISABLED");

    switch(DRF_VAL(_PDISP, _SOR_DP_LINKCTL, _LANECOUNT, reg))
    {
        case LW_PDISP_SOR_DP_LINKCTL_LANECOUNT_ZERO:
            dprintf("%-55s: %s\n", "LW_PDISP_SOR_DP_LINKCTL_LANECOUNT",
                "ZERO");
            break;
        case LW_PDISP_SOR_DP_LINKCTL_LANECOUNT_ONE:
            dprintf("%-55s: %s\n", "LW_PDISP_SOR_DP_LINKCTL_LANECOUNT",
                "ONE");
            break;
        case LW_PDISP_SOR_DP_LINKCTL_LANECOUNT_TWO:
            dprintf("%-55s: %s\n", "LW_PDISP_SOR_DP_LINKCTL_LANECOUNT",
                "TWO");
            break;
        case LW_PDISP_SOR_DP_LINKCTL_LANECOUNT_FOUR:
            dprintf("%-55s: %s\n", "LW_PDISP_SOR_DP_LINKCTL_LANECOUNT",
                "FOUR");
            break;
        default:
            dprintf("ERROR: %s: Invalid LW_PDISP_SOR_DP_LINKCTL_LANECOUNT value.\n",
                __FUNCTION__);
    }

    dprintf("%-55s: %s\n", "LW_PDISP_SOR_DP_LINKCTL_FORMAT_MODE",
        DRF_VAL(_PDISP, _SOR_DP_LINKCTL, _FORMAT_MODE, reg) ?
        "MULTI_STREAM" : "SINGLE_STREAM");

    reg = GPU_REG_RD32(LW_PDISP_SOR_DP_PADCTL(sorIndex, dpIndex));

    dprintf("%-55s: %s\n", "LW_PDISP_SOR_DP_PADCTL_PD_TXD_0",
        DRF_VAL(_PDISP, _SOR_DP_PADCTL, _PD_TXD_0, reg) ?
        "NO" : "YES");

    dprintf("%-55s: %s\n", "LW_PDISP_SOR_DP_PADCTL_PD_TXD_1",
        DRF_VAL(_PDISP, _SOR_DP_PADCTL, _PD_TXD_1, reg) ?
        "NO" : "YES");

    dprintf("%-55s: %s\n", "LW_PDISP_SOR_DP_PADCTL_PD_TXD_2",
        DRF_VAL(_PDISP, _SOR_DP_PADCTL, _PD_TXD_2, reg) ?
        "NO" : "YES");

    dprintf("%-55s: %s\n", "LW_PDISP_SOR_DP_PADCTL_PD_TXD_3",
        DRF_VAL(_PDISP, _SOR_DP_PADCTL, _PD_TXD_3, reg) ?
        "NO" : "YES");

    reg = GPU_REG_RD32(LW_PDISP_CLK_REM_SOR(sorIndex));

    switch(DRF_VAL(_PDISP, _CLK_REM_SOR, _DIV, reg))
    {
        case LW_PDISP_CLK_REM_SOR_DIV_BY_1:
            dprintf("%-55s: %s\n", "LW_PDISP_CLK_REM_SOR_DIV",
                "BY_1");
            break;
        case LW_PDISP_CLK_REM_SOR_DIV_BY_2:
            dprintf("%-55s: %s\n", "LW_PDISP_CLK_REM_SOR_DIV",
                "BY_2");
            break;
        case LW_PDISP_CLK_REM_SOR_DIV_BY_4:
            dprintf("%-55s: %s\n", "LW_PDISP_CLK_REM_SOR_DIV",
                "BY_4");
            break;
        case LW_PDISP_CLK_REM_SOR_DIV_BY_8:
            dprintf("%-55s: %s\n", "LW_PDISP_CLK_REM_SOR_DIV",
                "BY_4");
            break;
        case LW_PDISP_CLK_REM_SOR_DIV_BY_16:
            dprintf("%-55s: %s\n", "LW_PDISP_CLK_REM_SOR_DIV",
                "BY_4");
            break;
        default:
            dprintf("ERROR: %s: Invalid LW_PDISP_CLK_REM_SOR_DIV value.\n",
                __FUNCTION__);
    }

    switch(DRF_VAL(_PDISP, _CLK_REM_SOR, _MODE, reg))
    {
        case LW_PDISP_CLK_REM_SOR_MODE_XITION:
            dprintf("%-55s: %s\n", "LW_PDISP_CLK_REM_SOR_DIV",
                "XITION");
            break;
        case LW_PDISP_CLK_REM_SOR_MODE_NORMAL:
            dprintf("%-55s: %s\n", "LW_PDISP_CLK_REM_SOR_MODE",
                "NORMAL");
            break;
        case LW_PDISP_CLK_REM_SOR_MODE_SAFE:
            dprintf("%-55s: %s\n", "LW_PDISP_CLK_REM_SOR_MODE",
                "SAFE");
            break;
        default:
            dprintf("ERROR: %s: Invalid LW_PDISP_CLK_REM_SOR_MODE value.\n",
                __FUNCTION__);
    }

    switch(DRF_VAL(_PDISP, _CLK_REM_SOR, _PLL_REF_DIV, reg))
    {
        case LW_PDISP_CLK_REM_SOR_DIV_BY_1:
            dprintf("%-55s: %s\n", "LW_PDISP_CLK_REM_SOR_PLL_REF_DIV",
                "BY_1");
            break;
        case LW_PDISP_CLK_REM_SOR_DIV_BY_2:
            dprintf("%-55s: %s\n", "LW_PDISP_CLK_REM_SOR_PLL_REF_DIV",
                "BY_2");
            break;
        case LW_PDISP_CLK_REM_SOR_DIV_BY_4:
            dprintf("%-55s: %s\n", "LW_PDISP_CLK_REM_SOR_PLL_REF_DIV",
                "BY_4");
            break;
        case LW_PDISP_CLK_REM_SOR_DIV_BY_8:
            dprintf("%-55s: %s\n", "LW_PDISP_CLK_REM_SOR_PLL_REF_DIV",
                "BY_4");
            break;
        case LW_PDISP_CLK_REM_SOR_DIV_BY_16:
            dprintf("%-55s: %s\n", "LW_PDISP_CLK_REM_SOR_PLL_REF_DIV",
                "BY_4");
            break;
        default:
            dprintf("ERROR: %s: Invalid LW_PDISP_CLK_REM_SOR_PLL_REF_DIV value.\n",
                __FUNCTION__);
    }

    switch(DRF_VAL(_PDISP, _CLK_REM_SOR, _MODE_BYPASS, reg))
    {
        case LW_PDISP_CLK_REM_SOR_MODE_BYPASS_NONE:
            dprintf("%-55s: %s\n", "LW_PDISP_CLK_REM_SOR_MODE_BYPASS",
                "NONE");
            break;
        case LW_PDISP_CLK_REM_SOR_MODE_BYPASS_DP_NORMAL:
            dprintf("%-55s: %s\n", "LW_PDISP_CLK_REM_SOR_MODE_BYPASS",
                "DP_NORMAL");
            break;
        case LW_PDISP_CLK_REM_SOR_MODE_BYPASS_DP_SAFE:
            dprintf("%-55s: %s\n", "LW_PDISP_CLK_REM_SOR_MODE_BYPASS",
                "DP_SAFE");
            break;
        case LW_PDISP_CLK_REM_SOR_MODE_BYPASS_FEEDBACK:
            dprintf("%-55s: %s\n", "LW_PDISP_CLK_REM_SOR_MODE_BYPASS",
                "FEEDBACK");
            break;
        default:
            dprintf("ERROR: %s: Invalid LW_PDISP_CLK_REM_SOR_MODE_BYPASS value.\n",
                __FUNCTION__);
    }

    switch(DRF_VAL(_PDISP, _CLK_REM_SOR, _LINK_SPEED, reg))
    {
        case LW_PDISP_CLK_REM_SOR_LINK_SPEED_DP_1_62GHZ:
            dprintf("%-55s: %s\n", "LW_PDISP_CLK_REM_SOR_LINK_SPEED",
                    "1_62GHZ");
            break;
        case LW_PDISP_CLK_REM_SOR_LINK_SPEED_DP_2_70GHZ:
            dprintf("%-55s: %s\n", "LW_PDISP_CLK_REM_SOR_LINK_SPEED",
                    "2_70GHZ");
            break;
        case LW_PDISP_CLK_REM_SOR_LINK_SPEED_DP_5_40GHZ:
            dprintf("%-55s: %s\n", "LW_PDISP_CLK_REM_SOR_LINK_SPEED",
                    "5_40GHZ");
            break;
        default:
            dprintf("WARNING: %-55s: %x\n",
                "LW_PDISP_CLK_REM_SOR_LINK_SPEED",
                DRF_VAL(_PDISP, _CLK_REM_SOR, _LINK_SPEED, reg));
            break;
    }

    dprintf("%-55s: %s\n", "LW_PDISP_CLK_REM_SOR_STATE",
        DRF_VAL(_PDISP, _CLK_REM_SOR, _STATE, reg) ? "ENABLE" : "DISABLE");

    switch(DRF_VAL(_PDISP, _CLK_REM_SOR, _CLK_SOURCE, reg))
    {
        case LW_PDISP_CLK_REM_SOR_CLK_SOURCE_SINGLE_PCLK:
            dprintf("%-55s: %s\n", "LW_PDISP_CLK_REM_SOR_CLK_SOURCE",
                "SINGLE_PCLK");
            break;
        case LW_PDISP_CLK_REM_SOR_CLK_SOURCE_DIFF_PCLK:
            dprintf("%-55s: %s\n", "LW_PDISP_CLK_REM_SOR_CLK_SOURCE",
                "DIFF_PCLK");
            break;
        case LW_PDISP_CLK_REM_SOR_CLK_SOURCE_SINGLE_DPCLK:
            dprintf("%-55s: %s\n", "LW_PDISP_CLK_REM_SOR_CLK_SOURCE",
                "SINGLE_DPCLK");
            break;
        case LW_PDISP_CLK_REM_SOR_CLK_SOURCE_DIFF_DPCLK:
            dprintf("%-55s: %s\n", "LW_PDISP_CLK_REM_SOR_CLK_SOURCE",
                "DIFF_DPCLK");
            break;
        default:
            dprintf("ERROR: %s: Invalid LW_PDISP_CLK_REM_SOR_CLK_SOURCE value.\n",
                __FUNCTION__);
    }

    reg = GPU_REG_RD32(LW_PDISP_SOR_CAP(sorIndex));

    dprintf("%-55s: %s\n", "LW_PDISP_SOR_CAP_DP_A",
        DRF_VAL(_PDISP, _SOR_CAP, _DP_A, reg) ? "TRUE" : "FALSE");

    dprintf("%-55s: %s\n", "LW_PDISP_SOR_CAP_DP_B",
        DRF_VAL(_PDISP, _SOR_CAP, _DP_B, reg) ? "TRUE" : "FALSE");

    dprintf("%-55s: %s\n", "LW_PDISP_SOR_CAP_DP_INTERFACE",
        DRF_VAL(_PDISP, _SOR_CAP, _DP_INTERLACE, reg) ? "TRUE" : "FALSE");

    reg = GPU_REG_RD32(LW_PDISP_DSI_SOR_CAP(sorIndex));

    dprintf("%-55s: %s\n", "LW_PDISP_DSI_SOR_CAP_DP_A",
        DRF_VAL(_PDISP, _DSI_SOR_CAP, _DP_A, reg) ? "TRUE" : "FALSE");

    dprintf("%-55s: %s\n", "LW_PDISP_DSI_SOR_CAP_DP_B",
        DRF_VAL(_PDISP, _DSI_SOR_CAP, _DP_B, reg) ? "TRUE" : "FALSE");

    dprintf("%-55s: %s\n", "LW_PDISP_DSI_SOR_CAP_DP_INTERLACE",
        DRF_VAL(_PDISP, _DSI_SOR_CAP, _DP_INTERLACE, reg) ? "TRUE" : "FALSE");

    reg = GPU_REG_RD32(LW_PDISP_SOR_DP_HDCP_BKSV_MSB(sorIndex));

    dprintf("%-55s: %x\n", "LW_PDISP_SOR_DP_HDCP_BKSV_MSB_VALUE",
        DRF_VAL(_PDISP, _SOR_DP_HDCP_BKSV_MSB, _VALUE, reg));

    dprintf("%-55s: %x\n", "LW_PDISP_SOR_DP_HDCP_BKSV_MSB_REPEATER",
        DRF_VAL(_PDISP, _SOR_DP_HDCP_BKSV_MSB, _REPEATER, reg));

    dprintf("%-55s: %x\n", "LW_PDISP_SOR_DP_HDCP_BKSV_LSB_VALUE",
        GPU_REG_RD32(LW_PDISP_SOR_DP_HDCP_BKSV_LSB(sorIndex)));

    dprintf("%-55s: %x\n", "LW_PDISP_SOR_DP_HDCP_RI",
            GPU_REG_RD32(LW_PDISP_SOR_DP_HDCP_RI(sorIndex)));

    reg = GPU_REG_RD32(LW_PDISP_SOR_DP_SPARE(sorIndex, dpIndex));

    dprintf("%-55s: %s\n", "LW_PDISP_SOR_DP_SPARE_SEQ_ENABLE",
        DRF_VAL(_PDISP, _SOR_DP_SPARE, _SEQ_ENABLE, reg) ? "YES" : "NO");

    dprintf("%-55s: %s\n", "LW_PDISP_SOR_DP_SPARE_PANEL",
        DRF_VAL(_PDISP, _SOR_DP_SPARE, _PANEL, reg) ? "INTERNAL" : "EXTERNAL");

    reg = GPU_REG_RD32(LW_PDISP_SOR_TEST(sorIndex));
    dprintf("%-55s: ", "LW_PDISP_SOR_TEST");
    headMask = DRF_VAL(_PDISP, _SOR_TEST, _OWNER_MASK, reg);
    if (headMask)
    {
        BOOL bHeadOnce;
        LwU32 i;

        // Print attached HEADs
        bHeadOnce = FALSE;
        for (i = 0; i < pDisp[indexGpu].dispGetNumHeads(); i++)
        {
            if (headMask & (1 << i))
            {
                if (bHeadOnce)
                    dprintf("|%d", i);
                else
                    dprintf("HEAD %d", i);

                bHeadOnce = TRUE;
            }
        }

        if (bHeadOnce)
            dprintf("\n");
        else
            dprintf("ERROR: head index can't be recognized\n");

        // Print relevant SF registers
        for (i = 0; i < pDisp[indexGpu].dispGetNumSfs(); i++)
        {
            LwU32 reg = GPU_REG_RD32(LW_PDISP_SF_TEST(i));

            if (DRF_VAL(_PDISP, _SF_TEST, _OWNER_MASK, reg) & headMask)
            {
                dprintf("%-55s: %d\n", "sfIndex", i);

                reg = GPU_REG_RD32(LW_PDISP_SF_DP_LINKCTL(i));

                dprintf("%-55s: %s\n", "LW_PDISP_SF_DP_LINKCTL_ENABLE",
                    DRF_VAL(_PDISP, _SF_DP_LINKCTL, _ENABLE,
                    reg) ? "YES" : "NO");

                dprintf("%-55s: %s\n", "LW_PDISP_SF_DP_LINKCTL_FORMAT_MODE",
                    DRF_VAL(_PDISP, _SF_DP_LINKCTL, _FORMAT_MODE,
                    reg) ? "MULTI_STREAM" : "SINGLE_STREAM");

                dprintf("%-55s: %d\n", "LW_PDISP_SF_DP_LINKCTL_LANECOUNT",
                    DRF_VAL(_PDISP, _SF_DP_LINKCTL, _LANECOUNT, reg));

                reg = GPU_REG_RD32(LW_PDISP_SF_DP_FLUSH(i));

                dprintf("%-55s: %s\n", "LW_PDISP_SF_DP_FLUSH_ENABLE",
                    DRF_VAL(_PDISP, _SF_DP_FLUSH, _ENABLE,
                    reg) ? "YES" : "NO");

                dprintf("%-55s: %s\n", "LW_PDISP_SF_DP_FLUSH_MODE",
                    DRF_VAL(_PDISP, _SF_DP_FLUSH, _MODE,
                    reg) ? "IMMEDIATE" : "LOADV");

                dprintf("%-55s: %s\n", "LW_PDISP_SF_DP_FLUSH_CNTL",
                    DRF_VAL(_PDISP, _SF_DP_FLUSH, _CNTL,
                    reg) ? "PENDING" : "DONE");

                reg = GPU_REG_RD32(LW_PDISP_SF_DP_STREAM_CTL(i));

                dprintf("%-55s: %d\n", "LW_PDISP_SF_DP_STREAM_CTL_START",
                    DRF_VAL(_PDISP, _SF_DP_STREAM_CTL, _START, reg));

                dprintf("%-55s: %d\n", "LW_PDISP_SF_DP_STREAM_CTL_LENGTH",
                    DRF_VAL(_PDISP, _SF_DP_STREAM_CTL, _LENGTH, reg));

                dprintf("%-55s: %d\n",
                    "LW_PDISP_SF_DP_STREAM_CTL_START_ACTIVE",
                    DRF_VAL(_PDISP, _SF_DP_STREAM_CTL, _START_ACTIVE, reg));

                dprintf("%-55s: %d\n",
                    "LW_PDISP_SF_DP_STREAM_CTL_LENGTH_ACTIVE",
                    DRF_VAL(_PDISP, _SF_DP_STREAM_CTL, _LENGTH_ACTIVE, reg));

                reg = GPU_REG_RD32(LW_PDISP_SF_DP_STREAM_BW(i));

                dprintf("%-55s: %d\n", "LW_PDISP_SF_DP_STREAM_BW_ALLOCATED",
                    DRF_VAL(_PDISP, _SF_DP_STREAM_BW, _ALLOCATED, reg));

                dprintf("%-55s: %d\n", "LW_PDISP_SF_DP_STREAM_BW_TIMESLICE",
                    DRF_VAL(_PDISP, _SF_DP_STREAM_BW, _TIMESLICE, reg));
            }
        }
    }

    dispPrintDpRxInfo(port);
}
