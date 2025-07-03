/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2018-2020 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#include "common_lwswitch.h"
#include "lr10/lr10.h"
#include "lr10/clock_lr10.h"
#include "lr10/soe_lr10.h"
#include "lwswitch/lr10/dev_soe_ip.h"
#include "lwswitch/lr10/dev_pri_ringstation_sys.h"
#include "lwswitch/lr10/dev_trim.h"
#include "lwswitch/lr10/dev_lws.h"
#include "lwswitch/lr10/dev_lwlperf_ip.h"
#include "lwswitch/lr10/dev_npgperf_ip.h"
#include "lwswitch/lr10/dev_lwlctrl_ip.h"
#include "lwswitch/lr10/dev_lw_xp.h"
#include "lwswitch/lr10/dev_lw_xve.h"
#include "lwswitch/lr10/dev_nport_ip.h"
#include "lwswitch/lr10/dev_minion_ip.h"
#include "lwswitch/lr10/dev_timer.h"
#include "lwswitch/lr10/dev_pri_ringmaster.h"
#include "lwswitch/lr10/dev_pri_ringstation_prt.h"

//
// Initialize the software state of the switch PLL
//
LwlStatus
lwswitch_init_pll_config_lr10
(
    lwswitch_device *device
)
{
    LWSWITCH_PLL_LIMITS pll_limits;
    LWSWITCH_PLL_INFO pll;
    LwlStatus retval = LWL_SUCCESS;

    //
    // These parameters could come from schmoo'ing API, settings file or a ROM.
    // If no configuration ROM settings are present, use the PLL documentation
    //
    // Refer to the PLL35G_DYN_PRB_ESD_B2 cell Vbios Table, in the PLL datasheet
    // for restrictions on MDIV, NDIV and PLDIV to satisfy the pll's frequency limitation.
    //
    // PLL35G_DYN_PRB_ESD_B1.doc
    //

    pll_limits.ref_min_mhz = 100;
    pll_limits.ref_max_mhz = 100;
    pll_limits.vco_min_mhz = 1750;
    pll_limits.vco_max_mhz = 3800;
    pll_limits.update_min_mhz = 13;         // 13.5MHz
    pll_limits.update_max_mhz = 38;         // 38.4MHz
    pll_limits.m_min = LW_PCLOCK_LWSW_SWITCHPLL_COEFF_MDIV_MIN;
    pll_limits.m_max = LW_PCLOCK_LWSW_SWITCHPLL_COEFF_MDIV_MAX;
    pll_limits.n_min = LW_PCLOCK_LWSW_SWITCHPLL_COEFF_NDIV_MIN;
    pll_limits.n_max = LW_PCLOCK_LWSW_SWITCHPLL_COEFF_NDIV_MAX;
    pll_limits.pl_min = LW_PCLOCK_LWSW_SWITCHPLL_COEFF_PLDIV_MIN;
    pll_limits.pl_max = LW_PCLOCK_LWSW_SWITCHPLL_COEFF_PLDIV_MAX;
    pll_limits.valid = LW_TRUE;

    //
    // set well known coefficients to achieve frequency
    // MDIV: need to set > 1 to achieve update_rate < 38.4 MHz
    // 100 / 5 = 20 MHz update rate, therefore MDIV = 5
    // NDIV needs to take us all the way to 1640 MHz
    // 1640 / 20 = 82.  But 100*82/5 < 1.75GHz VCOmin,
    // therefore double NDIV to 164 and set PDIV to 2.
    //

    pll.src_freq_khz = 100000;        // 100MHz
    pll.M = 5;
    pll.N = 164;
    pll.PL = 1;
    pll.dist_mode = LW_PCLOCK_LWSW_CLK_DIST_MODE_SWITCH2CLK_DIST_MODE_2XCLK;
    pll.refclk_div = 15;

    retval = lwswitch_validate_pll_config(device, &pll, pll_limits);
    if (retval != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, WARN,
            "Selecting default PLL setting.\n");

        // Select default, safe clock (1.64GHz)
        pll.src_freq_khz = 100000;        // 100MHz
        pll.M = 5;
        pll.N = 164;
        pll.PL = 2;
        pll.dist_mode =
             LW_PCLOCK_LWSW_CLK_DIST_MODE_SWITCH2CLK_DIST_MODE_1XCLK;
        pll.refclk_div = LW_PCLOCK_LWSW_RX_BYPASS_REFCLK_DIV_INIT;

        retval = lwswitch_validate_pll_config(device, &pll, pll_limits);
        if (retval != LWL_SUCCESS)
        {
            LWSWITCH_PRINT(device, ERROR,
                "Default PLL setting failed.\n");
            return retval;
        }
    }

    device->switch_pll = pll;

    return LWL_SUCCESS;
}

//
// Check that the PLLs are initialized. VBIOS is expected to configure PLLs
//
LwlStatus
lwswitch_init_pll_lr10
(
    lwswitch_device *device
)
{
    LwU32   pllRegVal;

    //
    // Clocks should only be initialized on silicon or a clocks netlist on emulation
    // Unfortunately, we don't have a full robust infrastructure for detecting the
    // runtime environment as we do on GPU.
    //
    if (IS_RTLSIM(device) || IS_EMULATION(device) || IS_FMODEL(device))
    {
        LWSWITCH_PRINT(device, WARN,
        "%s: Skipping setup of LWSwitch clocks\n",
            __FUNCTION__);
        return LWL_SUCCESS;
    }

    pllRegVal = LWSWITCH_REG_RD32(device, _PCLOCK, _LWSW_SWITCHPLL_CFG);
    if (!FLD_TEST_DRF(_PCLOCK, _LWSW_SWITCHPLL_CFG, _PLL_LOCK, _TRUE, pllRegVal))
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: _PLL_LOCK failed\n",
            __FUNCTION__);
        return -LWL_INITIALIZATION_TOTAL_FAILURE;
    }
    if (!FLD_TEST_DRF(_PCLOCK, _LWSW_SWITCHPLL_CFG, _PLL_FREQLOCK, _YES, pllRegVal))
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: _PLL_FREQLOCK failed\n",
            __FUNCTION__);
        return -LWL_INITIALIZATION_TOTAL_FAILURE;
    }

    pllRegVal = LWSWITCH_REG_RD32(device, _PCLOCK, _LWSW_SWITCHCLK);
    if (!FLD_TEST_DRF_NUM(_PCLOCK, _LWSW_SWITCHCLK, _RDY_SWITCHPLL, 1, pllRegVal))
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: _RDY_SWITCHPLL failed\n",
            __FUNCTION__);
        return -LWL_INITIALIZATION_TOTAL_FAILURE;
    }

    pllRegVal = LWSWITCH_REG_RD32(device, _PCLOCK, _LWSW_SYSTEMCLK);
    if (!FLD_TEST_DRF_NUM(_PCLOCK, _LWSW_SYSTEMCLK, _SYSTEMCLK_RDY_SWITCHPLL, 1, pllRegVal))
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: _RDY_SWITCHPLL for SYSTEMCLK failed\n",
            __FUNCTION__);
        return -LWL_INITIALIZATION_TOTAL_FAILURE;
    }

    return LWL_SUCCESS;
}

//
// Timer functions
//

void
lwswitch_init_hw_counter_lr10
(
    lwswitch_device *device
)
{
    return;
}

void
lwswitch_hw_counter_shutdown_lr10
(
    lwswitch_device *device
)
{
    return;
}

//
// Reads the 36-bit free running counter
//
LwU64
lwswitch_hw_counter_read_counter_lr10
(
    lwswitch_device *device
)
{
    return -LWL_ERR_NOT_SUPPORTED;
}

//
// Initialize clock gating.
//
void
lwswitch_init_clock_gating_lr10
(
    lwswitch_device *device
)
{
    LwU32 regval;
    LwU32 i;

    // BUS
    LWSWITCH_REG_WR32(device, _PBUS, _EXT_CG1,
        DRF_DEF(_PBUS, _EXT_CG1, _SLCG, __PROD)         |
        DRF_DEF(_PBUS, _EXT_CG1, _SLCG_C11, __PROD)     |
        DRF_DEF(_PBUS, _EXT_CG1, _SLCG_PRI, __PROD)     |
        DRF_DEF(_PBUS, _EXT_CG1, _SLCG_UNROLL, __PROD)  |
        DRF_DEF(_PBUS, _EXT_CG1, _SLCG_ROLL, __PROD)    |
        DRF_DEF(_PBUS, _EXT_CG1, _SLCG_IFR, __PROD)     |
        DRF_DEF(_PBUS, _EXT_CG1, _SLCG_PMC, __PROD));

    // PRI
    LWSWITCH_REG_WR32(device, _PPRIV_MASTER, _CG1,
        DRF_DEF(_PPRIV_MASTER, _CG1, _SLCG, __PROD));

    regval = 
        DRF_DEF(_PPRIV_PRT, _CG1_SLCG, _SLOWCLK, __PROD)              |
        DRF_DEF(_PPRIV_PRT, _CG1_SLCG, _PRIV_CONFIG_REGS, __PROD)     |
        DRF_DEF(_PPRIV_PRT, _CG1_SLCG, _PRIV_FUNNEL_DECODER, __PROD)  |
        DRF_DEF(_PPRIV_PRT, _CG1_SLCG, _PRIV_FUNNEL_ARB, __PROD)      |
        DRF_DEF(_PPRIV_PRT, _CG1_SLCG, _PRIV_HISTORY_BUFFER, __PROD)  |
        DRF_DEF(_PPRIV_PRT, _CG1_SLCG, _PRIV_MASTER, __PROD)          |
        DRF_DEF(_PPRIV_PRT, _CG1_SLCG, _PRIV_SLAVE, __PROD)           |
        DRF_DEF(_PPRIV_PRT, _CG1_SLCG, _PRIV_UCODE_TRAP, __PROD)      |
        DRF_DEF(_PPRIV_PRT, _CG1_SLCG, _PRIV, __PROD)                 |
        DRF_DEF(_PPRIV_PRT, _CG1_SLCG, _LOC_PRIV, __PROD)             |
        DRF_DEF(_PPRIV_PRT, _CG1_SLCG, _PM, __PROD);

    LWSWITCH_REG_WR32(device, _PPRIV_PRT_PRT0, _CG1, regval);
    LWSWITCH_REG_WR32(device, _PPRIV_PRT_PRT1, _CG1, regval);
    LWSWITCH_REG_WR32(device, _PPRIV_PRT_PRT2, _CG1, regval);
    LWSWITCH_REG_WR32(device, _PPRIV_PRT_PRT3, _CG1, regval);
    LWSWITCH_REG_WR32(device, _PPRIV_PRT_PRT4, _CG1, regval);
    LWSWITCH_REG_WR32(device, _PPRIV_PRT_PRT5, _CG1, regval);
    LWSWITCH_REG_WR32(device, _PPRIV_PRT_PRT6, _CG1, regval);
    LWSWITCH_REG_WR32(device, _PPRIV_PRT_PRT7, _CG1, regval);
    LWSWITCH_REG_WR32(device, _PPRIV_PRT_PRT8, _CG1, regval);

    // XP3G
    LWSWITCH_REG_WR32(device, _XP, _PRI_XP3G_CG,
        DRF_DEF(_XP, _PRI_XP3G_CG, _IDLE_CG_DLY_CNT, __PROD)    |
        DRF_DEF(_XP, _PRI_XP3G_CG, _IDLE_CG_EN, __PROD)         |
        DRF_DEF(_XP, _PRI_XP3G_CG, _STATE_CG_EN, __PROD)        |
        DRF_DEF(_XP, _PRI_XP3G_CG, _STALL_CG_DLY_CNT, __PROD)   |
        DRF_DEF(_XP, _PRI_XP3G_CG, _STALL_CG_EN, __PROD)        |
        DRF_DEF(_XP, _PRI_XP3G_CG, _QUIESCENT_CG_EN, __PROD)    |
        DRF_DEF(_XP, _PRI_XP3G_CG, _WAKEUP_DLY_CNT, __PROD)     |
        DRF_DEF(_XP, _PRI_XP3G_CG, _THROT_CLK_CNT, __PROD)      |
        DRF_DEF(_XP, _PRI_XP3G_CG, _DI_DT_SKEW_VAL, __PROD)     |
        DRF_DEF(_XP, _PRI_XP3G_CG, _THROT_CLK_EN, __PROD)       |
        DRF_DEF(_XP, _PRI_XP3G_CG, _THROT_CLK_SW_OVER, __PROD)  |
        DRF_DEF(_XP, _PRI_XP3G_CG, _PAUSE_CG_EN, __PROD)        |
        DRF_DEF(_XP, _PRI_XP3G_CG, _HALT_CG_EN, __PROD));

    LWSWITCH_REG_WR32(device, _XP, _PRI_XP3G_CG1,
        DRF_DEF(_XP, _PRI_XP3G_CG1, _MONITOR_CG_EN, __PROD));

    // XVE
    LWSWITCH_ENG_WR32_LR10(device, XVE, , 0, _XVE, _PRI_XVE_CG,
        DRF_DEF(_XVE, _PRI_XVE_CG, _IDLE_CG_DLY_CNT, __PROD)    |
        DRF_DEF(_XVE, _PRI_XVE_CG, _IDLE_CG_EN, __PROD)         |
        DRF_DEF(_XVE, _PRI_XVE_CG, _STATE_CG_EN, __PROD)        |
        DRF_DEF(_XVE, _PRI_XVE_CG, _STALL_CG_DLY_CNT, __PROD)   |
        DRF_DEF(_XVE, _PRI_XVE_CG, _STALL_CG_EN, __PROD)        |
        DRF_DEF(_XVE, _PRI_XVE_CG, _QUIESCENT_CG_EN, __PROD)    |
        DRF_DEF(_XVE, _PRI_XVE_CG, _WAKEUP_DLY_CNT, __PROD)     |
        DRF_DEF(_XVE, _PRI_XVE_CG, _THROT_CLK_CNT, __PROD)      |
        DRF_DEF(_XVE, _PRI_XVE_CG, _DI_DT_SKEW_VAL, __PROD)     |
        DRF_DEF(_XVE, _PRI_XVE_CG, _THROT_CLK_EN, __PROD)       |
        DRF_DEF(_XVE, _PRI_XVE_CG, _THROT_CLK_SW_OVER, __PROD)  |
        DRF_DEF(_XVE, _PRI_XVE_CG, _PAUSE_CG_EN, __PROD)        |
        DRF_DEF(_XVE, _PRI_XVE_CG, _HALT_CG_EN, __PROD));

    LWSWITCH_ENG_WR32_LR10(device, XVE, , 0, _XVE, _PRI_XVE_CG1,
        DRF_DEF(_XVE, _PRI_XVE_CG1, _MONITOR_CG_EN, __PROD)     |
        DRF_DEF(_XVE, _PRI_XVE_CG1, _SLCG, __PROD));

    // NPORT
    LWSWITCH_NPORT_MC_BCAST_WR32_LR10(device, _NPORT, _CTRL_SLCG,
        DRF_DEF(_NPORT, _CTRL_SLCG_DIS_CG, _INGRESS,  __PROD)  |
        DRF_DEF(_NPORT, _CTRL_SLCG_DIS_CG, _ROUTE,    __PROD)  |
        DRF_DEF(_NPORT, _CTRL_SLCG_DIS_CG, _EGRESS,   __PROD)  |
        DRF_DEF(_NPORT, _CTRL_SLCG_DIS_CG, _STRACK,   __PROD)  |
        DRF_DEF(_NPORT, _CTRL_SLCG_DIS_CG, _TAGSTATE, __PROD)  |
        DRF_DEF(_NPORT, _CTRL_SLCG_DIS_CG, _TREX,     __PROD));

    // NPG_PERFMON
    LWSWITCH_BCAST_WR32_LR10(device, NPG_PERFMON, _NPGPERF, _CTRL_CLOCK_GATING,
        DRF_DEF(_NPGPERF, _CTRL_CLOCK_GATING, _CG1_SLCG, __PROD));

    LWSWITCH_BCAST_WR32_LR10(device, NPG_PERFMON, _NPGPERF, _PERF_CTRL_CLOCK_GATING,
        DRF_DEF(_NPGPERF, _PERF_CTRL_CLOCK_GATING, _CG1_SLCG, __PROD) |
        DRF_DEF(_NPGPERF, _PERF_CTRL_CLOCK_GATING, _CONTEXT_FREEZE, __PROD));

    //
    // LWLW_PERFMON
    //
    // There registers are protected by PRIV_LEVEL_MASK6.
    // PLM6 will not be blown on Production fuses.
    //
    LWSWITCH_BCAST_WR32_LR10(device, LWLW_PERFMON, _LWLPERF, _CTRL_CLOCK_GATING,
        DRF_DEF(_LWLPERF, _CTRL_CLOCK_GATING, _CG1_SLCG, __PROD) |
        DRF_DEF(_LWLPERF, _CTRL_CLOCK_GATING, _CG1_SLCG_CTRL, __PROD));

    LWSWITCH_BCAST_WR32_LR10(device, LWLW_PERFMON, _LWLPERF, _PERF_CTRL_CLOCK_GATING,
        DRF_DEF(_LWLPERF, _PERF_CTRL_CLOCK_GATING, _CG1_SLCG, __PROD) |
        DRF_DEF(_LWLPERF, _PERF_CTRL_CLOCK_GATING, _CONTEXT_FREEZE, __PROD));

    // LWLCTRL
    LWSWITCH_BCAST_WR32_LR10(device, LWLW, _LWLCTRL, _PLL_PRI_CLOCK_GATING,
        DRF_DEF(_LWLCTRL, _PLL_PRI_CLOCK_GATING, _CG1_SLCG, __PROD));

    // MINION
    for (i = 0; i < LWSWITCH_ENG_COUNT(device, MINION, ); i++)
    {
        regval = LWSWITCH_ENG_RD32_LR10(device, MINION, i, _CMINION_FALCON, _CG2);

        LWSWITCH_ENG_WR32_LR10(device, MINION, , i, _CMINION_FALCON, _CG2,
            FLD_SET_DRF(_CMINION_FALCON, _CG2, _SLCG, __PROD, regval));
    }

    // PTIMER
    LWSWITCH_REG_WR32(device, _PTIMER, _PRI_TMR_CG1,
        DRF_DEF(_PTIMER, _PRI_TMR_CG1, _MONITOR_CG_EN, __PROD) |
        DRF_DEF(_PTIMER, _PRI_TMR_CG1, _SLCG, __PROD));

    // SOE
    regval = LWSWITCH_SOE_RD32_LR10(device, 0, _SOE, _FBIF_CG1);
    regval = FLD_SET_DRF(_SOE, _FBIF_CG1, _SLCG, __PROD, regval);
    LWSWITCH_SOE_WR32_LR10(device, 0, _SOE, _FBIF_CG1, regval);

    regval = LWSWITCH_SOE_RD32_LR10(device, 0, _SOE, _FALCON_CG2);
    regval = FLD_SET_DRF(_SOE, _FALCON_CG2, _SLCG, __PROD, regval);
    LWSWITCH_SOE_WR32_LR10(device, 0, _SOE_FALCON, _CG2, regval);

    regval = LWSWITCH_SOE_RD32_LR10(device, 0, _SOE_MISC, _CG1);
    regval = FLD_SET_DRF(_SOE, _MISC_CG1, _SLCG, __PROD, regval);
    LWSWITCH_SOE_WR32_LR10(device, 0, _SOE_MISC, _CG1, regval);

    LWSWITCH_SOE_WR32_LR10(device, 0, _SOE_MISC, _TOP_CG,
        DRF_DEF(_SOE_MISC, _TOP_CG, _IDLE_CG_DLY_CNT, __PROD));

    return;
}
