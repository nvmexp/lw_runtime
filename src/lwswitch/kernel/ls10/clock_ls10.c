/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2020-2021 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#include "g_lwconfig.h"
#include "common_lwswitch.h"
#include "ls10/ls10.h"
#include "ls10/clock_ls10.h"

#include "lwswitch/ls10/dev_trim.h"
#include "lwswitch/ls10/dev_soe_ip.h"
#include "lwswitch/ls10/dev_lwlperf_ip.h"
#include "lwswitch/ls10/dev_npgperf_ip.h"
#include "lwswitch/ls10/dev_lwlw_ip.h"
#include "lwswitch/ls10/dev_nport_ip.h"
#include "lwswitch/ls10/dev_minion_ip.h"
#include "lwswitch/ls10/dev_timer_ip.h"
#include "lwswitch/ls10/dev_minion_ip.h"
#include "lwswitch/ls10/dev_pri_hub_prt_ip.h"
#include "lwswitch/ls10/dev_pri_masterstation_ip.h"

//
// Initialize the software state of the switch PLL
//
LwlStatus
lwswitch_init_pll_config_ls10
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
    // Refer to the *** TODO:TBD *** Vbios Table, in the PLL datasheet
    // for restrictions on MDIV, NDIV and PLDIV to satisfy the pll's frequency limitation.
    //
    // PLL40G_SMALL_ESD.doc
    //

    pll_limits.ref_min_mhz = 100;
    pll_limits.ref_max_mhz = 100;
    pll_limits.vco_min_mhz = 1750;
    pll_limits.vco_max_mhz = 3800;
    pll_limits.update_min_mhz = 13;         // 13.5MHz
    pll_limits.update_max_mhz = 38;         // 38.4MHz

    pll_limits.m_min = LW_CLOCK_LWSW_SYS_SWITCHPLL_COEFF_MDIV_MIN;
    pll_limits.m_max = LW_CLOCK_LWSW_SYS_SWITCHPLL_COEFF_MDIV_MAX;
    pll_limits.n_min = LW_CLOCK_LWSW_SYS_SWITCHPLL_COEFF_NDIV_MIN;
    pll_limits.n_max = LW_CLOCK_LWSW_SYS_SWITCHPLL_COEFF_NDIV_MAX;
    pll_limits.pl_min = LW_CLOCK_LWSW_SYS_SWITCHPLL_COEFF_PLDIV_MIN;
    pll_limits.pl_max = LW_CLOCK_LWSW_SYS_SWITCHPLL_COEFF_PLDIV_MAX;
    pll_limits.valid = LW_TRUE;

    //
    // set well known coefficients to achieve frequency
    // TODO: Document defaults once established
    // TODO: Bug#3075104 to update PLL settings prior to FS, once the PLL
    // verification is complete.
    //

    pll.src_freq_khz = 100000;        // 100MHz
    pll.M = 3;
    pll.N = 80;
    pll.PL = 2;
    pll.dist_mode = 0;      // Ignored.  Only 1x supported
    pll.refclk_div = LW_CLOCK_LWSW_SYS_RX_BYPASS_REFCLK_DIV_INIT;

    retval = lwswitch_validate_pll_config(device, &pll, pll_limits);
    if (retval != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, WARN,
            "Selecting default PLL setting.\n");

        // Select default, safe clock
        pll.src_freq_khz = 100000;        // 100MHz
        pll.M = 3;
        pll.N = 80;
        pll.PL = 2;
        pll.dist_mode = 0;      // Ignored.  Only 1x supported
        pll.refclk_div = LW_CLOCK_LWSW_SYS_RX_BYPASS_REFCLK_DIV_INIT;

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
lwswitch_init_pll_ls10
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

    pllRegVal = LWSWITCH_ENG_RD32(device, CLKS_SYS,  , 0, _CLOCK_LWSW_SYS, _SWITCHPLL_CFG);
    if (!FLD_TEST_DRF(_CLOCK_LWSW_SYS, _SWITCHPLL_CFG, _PLL_LOCK, _TRUE, pllRegVal))
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: _PLL_LOCK failed\n",
            __FUNCTION__);
        return -LWL_INITIALIZATION_TOTAL_FAILURE;
    }

    pllRegVal = LWSWITCH_ENG_RD32(device, CLKS_SYS,  , 0, _CLOCK_LWSW_SYS, _SWITCHPLL_CTRL);
    if (!FLD_TEST_DRF_NUM(_CLOCK_LWSW_SYS, _SWITCHPLL_CTRL, _PLL_FREQLOCK, 1, pllRegVal))
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: _PLL_FREQLOCK failed\n",
            __FUNCTION__);
        return -LWL_INITIALIZATION_TOTAL_FAILURE;
    }

    pllRegVal = LWSWITCH_ENG_RD32(device, CLKS_SYS,  , 0, _CLOCK_LWSW_SYS, _SWITCHCLK_SWITCH_DIVIDER);
    if (!FLD_TEST_DRF_NUM(_CLOCK_LWSW_SYS, _SWITCHCLK_SWITCH_DIVIDER, _SWITCH_DIVIDER_DONE, 1, pllRegVal))
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: _SWITCH_DIVIDER_DONE failed\n",
            __FUNCTION__);
        return -LWL_INITIALIZATION_TOTAL_FAILURE;
    }

    pllRegVal = LWSWITCH_ENG_RD32(device, CLKS_SYS,  , 0, _CLOCK_LWSW_SYS, _SYSTEM_CLK_SWITCH_DIVIDER);
    if (!FLD_TEST_DRF_NUM(_CLOCK_LWSW_SYS, _SYSTEM_CLK_SWITCH_DIVIDER, _SWITCH_DIVIDER_DONE, 1, pllRegVal))
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: _SWITCH_DIVIDER_DONE for SYSTEMCLK failed\n",
            __FUNCTION__);
        return -LWL_INITIALIZATION_TOTAL_FAILURE;
    }

    return LWL_SUCCESS;
}

//
// Initialize clock gating.
//
void
lwswitch_init_clock_gating_ls10
(
    lwswitch_device *device
)
{
    //
    // CG and PROD settings were already handled by:
    //    - lwswitch_lws_top_prod_ls10
    //    - lwswitch_npg_prod_ls10
    //    - lwswitch_apply_prod_lwlw_ls10
    //    - lwswitch_apply_prod_nxbar_ls10
    //
    // which were all called by lwswitch_initialize_ip_wrappers_ls10

    return;
}

