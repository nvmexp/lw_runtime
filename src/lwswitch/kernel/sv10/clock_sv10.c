/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2016-2020 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#include "common_lwswitch.h"
#include "regkey_lwswitch.h"
#include "sv10/sv10.h"
#include "sv10/clock_sv10.h"
#include "sv10/minion_sv10.h"
#include "lwswitch/svnp01/dev_pri_ringmaster.h"
#include "lwswitch/svnp01/dev_pri_ringstation_sys.h"
#include "lwswitch/svnp01/dev_pri_ringstation_prt.h"
#include "lwswitch/svnp01/dev_trim.h"
#include "lwswitch/svnp01/dev_lws.h"
#include "lwswitch/svnp01/dev_pmgr.h"
#include "lwswitch/svnp01/dev_lw_xp.h"
#include "lwswitch/svnp01/dev_lw_xve.h"
#include "lwswitch/svnp01/dev_minion_ip.h"
#include "lwswitch/svnp01/dev_lwlctrl_ip.h"
#include "lwswitch/svnp01/dev_lwl_ip.h"
#include "lwswitch/svnp01/dev_lwl_ip_addendum.h"
#include "lwswitch/svnp01/dev_npg_ip.h"
#include "lwswitch/svnp01/dev_npgperf_ip.h"
#include "lwswitch/svnp01/dev_lwlipt_ip.h"
#include "lwswitch/svnp01/dev_lwltlc_ip.h"

//
// Initialize the software state of the switch PLL
//
LwlStatus
lwswitch_init_pll_config_sv10
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
// Initialize the hardware state of the switch PLL based on the previously
// initialized software state.
//
LwlStatus
lwswitch_init_pll_sv10
(
    lwswitch_device *device
)
{
    LwlStatus retval = LWL_SUCCESS;
    LwU32 swap_clk;
    sv10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_SV10(device);

#if !defined(DISABLE_CLOCK_INIT)
    LWSWITCH_TIMEOUT timeout;
    LwU32   pllCfg;
    LwU32   pllMux;
    LwU32   pllSwitchClk;
#endif // !defined(DISABLE_CLOCK_INIT)

    // TODO: Bracket clock code with *_PRI_FENCE to guarantee ordering sequence?

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
        return retval;
    }

#if !defined(DISABLE_CLOCK_INIT)
    // set iddq = 0
    pllCfg = LWSWITCH_REG_RD32(device, _PCLOCK, _LWSW_SWITCHPLL_CFG);
    pllCfg = FLD_SET_DRF(_PCLOCK, _LWSW_SWITCHPLL_CFG, _IDDQ, _POWER_ON, pllCfg);
    pllCfg = FLD_SET_DRF(_PCLOCK, _LWSW_SWITCHPLL_CFG, _EN_LCKDET, _POWER_ON, pllCfg);
    LWSWITCH_REG_WR32(device, _PCLOCK, _LWSW_SWITCHPLL_CFG, pllCfg);
    LWSWITCH_FLUSH_MMIO(device);

    // wait at least 5us
    LWSWITCH_NSEC_DELAY(5*LWSWITCH_INTERVAL_1USEC_IN_NS);

    LWSWITCH_ASSERT(device->switch_pll.freq_khz != 0);
    LWSWITCH_REG_WR32(device, _PCLOCK, _LWSW_SWITCHPLL_COEFF,
        DRF_NUM(_PCLOCK, _LWSW_SWITCHPLL_COEFF, _MDIV, device->switch_pll.M) |
        DRF_NUM(_PCLOCK, _LWSW_SWITCHPLL_COEFF, _NDIV, device->switch_pll.N) |
        DRF_NUM(_PCLOCK, _LWSW_SWITCHPLL_COEFF, _PLDIV, device->switch_pll.PL) |
        DRF_DEF(_PCLOCK, _LWSW_SWITCHPLL_COEFF, _NDIV_NEW, _INIT) |
        DRF_DEF(_PCLOCK, _LWSW_SWITCHPLL_COEFF, _CLAMP_NDIV_CYA, _INIT)
        );

    LWSWITCH_REG_WR32(device, _PCLOCK, _LWSW_CLK_DIST_MODE,
        DRF_NUM(_PCLOCK, _LWSW_CLK_DIST_MODE, _SWITCH2CLK_DIST_MODE, device->switch_pll.dist_mode));

    LWSWITCH_REG_WR32(device, _PCLOCK, _LWSW_RX_BYPASS_REFCLK,
        DRF_DEF(_PCLOCK, _LWSW_RX_BYPASS_REFCLK, _DISABLE,                _INIT) |
        DRF_DEF(_PCLOCK, _LWSW_RX_BYPASS_REFCLK, _DIV_SYNC_WAIT,          _INIT) |
        DRF_NUM(_PCLOCK, _LWSW_RX_BYPASS_REFCLK, _DIV, device->switch_pll.refclk_div) |
        DRF_DEF(_PCLOCK, _LWSW_RX_BYPASS_REFCLK, _REFCLK_BUF_EN_CYA,      _INIT) |
        DRF_DEF(_PCLOCK, _LWSW_RX_BYPASS_REFCLK, _REFCLK_BUF_EN_OVERRIDE, _INIT));

    LWSWITCH_FLUSH_MMIO(device);

    // enable the PLL
    pllCfg = FLD_SET_DRF(_PCLOCK, _LWSW_SWITCHPLL_CFG, _ENABLE, _YES, pllCfg);
    LWSWITCH_REG_WR32(device, _PCLOCK, _LWSW_SWITCHPLL_CFG, pllCfg);
    LWSWITCH_FLUSH_MMIO(device);

    // poll for PLL_LOCK

    lwswitch_timeout_create(LWSWITCH_INTERVAL_5MSEC_IN_NS, &timeout);
    do
    {
        pllCfg = LWSWITCH_REG_RD32(device, _PCLOCK, _LWSW_SWITCHPLL_CFG);
        if (FLD_TEST_DRF(_PCLOCK, _LWSW_SWITCHPLL_CFG, _PLL_LOCK, _TRUE, pllCfg))
        {
            break;
        }
    }
    while (!lwswitch_timeout_check(&timeout));

    if (!FLD_TEST_DRF(_PCLOCK, _LWSW_SWITCHPLL_CFG, _PLL_LOCK, _TRUE, pllCfg))
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: _PLL_LOCK failed\n",
            __FUNCTION__);
    }
    LWSWITCH_ASSERT(FLD_TEST_DRF(_PCLOCK, _LWSW_SWITCHPLL_CFG, _PLL_LOCK, _TRUE, pllCfg));

    // unbypass the PLL
    pllCfg = FLD_SET_DRF(_PCLOCK, _LWSW_SWITCHPLL_CFG, _BYPASSPLL, _NO, pllCfg);
    LWSWITCH_REG_WR32(device, _PCLOCK, _LWSW_SWITCHPLL_CFG, pllCfg);

    // set the 8:1 switch to switchpll output
    pllMux = LWSWITCH_REG_RD32(device, _PCLOCK, _LWSW_SWITCHCLK);
    pllMux = FLD_SET_DRF(_PCLOCK, _LWSW_SWITCHCLK, _MUX, _SWITCHPLL, pllMux);
    LWSWITCH_REG_WR32(device, _PCLOCK, _LWSW_SWITCHCLK, pllMux);
    LWSWITCH_FLUSH_MMIO(device);

    // poll for switch status

    lwswitch_timeout_create(LWSWITCH_INTERVAL_5MSEC_IN_NS, &timeout);
    do
    {
        pllSwitchClk = LWSWITCH_REG_RD32(device, _PCLOCK, _LWSW_SWITCHCLK);
        if (FLD_TEST_DRF_NUM(_PCLOCK, _LWSW_SWITCHCLK, _RDY_SWITCHPLL, 1, pllSwitchClk))
        {
            break;
        }
    }
    while (!lwswitch_timeout_check(&timeout));

    if (!FLD_TEST_DRF_NUM(_PCLOCK, _LWSW_SWITCHCLK, _RDY_SWITCHPLL, 1, pllSwitchClk))
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: _RDY_SWITCHPLL failed\n",
            __FUNCTION__);
    }
    LWSWITCH_ASSERT(FLD_TEST_DRF_NUM(_PCLOCK, _LWSW_SWITCHCLK, _RDY_SWITCHPLL, 1, pllSwitchClk));

    // Enable repeaters
    {
        LWSWITCH_REG_WR32(device, _PBUS, _BUFLWHSCLK_CTRL_PREEAST_REFCLK,
            DRF_DEF(_PBUS, _BUFLWHSCLK_CTRL_PREEAST_REFCLK, _BIAS_CTRL, _DEFAULT)   |
            DRF_DEF(_PBUS, _BUFLWHSCLK_CTRL_PREEAST_REFCLK, _DRV_I,     _DEFAULT)   |
            DRF_DEF(_PBUS, _BUFLWHSCLK_CTRL_PREEAST_REFCLK, _DRV_R,     _DEFAULT)   |
            DRF_NUM(_PBUS, _BUFLWHSCLK_CTRL_PREEAST_REFCLK, _ENABLE,    0x1)        |   // Enable
            DRF_DEF(_PBUS, _BUFLWHSCLK_CTRL_PREEAST_REFCLK, _MISC_CTRL, __PROD));

        LWSWITCH_REG_WR32(device, _PBUS, _BUFLWHSCLK_CTRL_P0_REFCLK,
            DRF_DEF(_PBUS, _BUFLWHSCLK_CTRL_P0_REFCLK, _BIAS_CTRL, _DEFAULT)    |
            DRF_DEF(_PBUS, _BUFLWHSCLK_CTRL_P0_REFCLK, _DRV_I,     _DEFAULT)    |
            DRF_DEF(_PBUS, _BUFLWHSCLK_CTRL_P0_REFCLK, _DRV_R,     _DEFAULT)    |
            DRF_NUM(_PBUS, _BUFLWHSCLK_CTRL_P0_REFCLK, _ENABLE,    0x1)         |   // Enable
            DRF_DEF(_PBUS, _BUFLWHSCLK_CTRL_P0_REFCLK, _MISC_CTRL, __PROD));

        LWSWITCH_REG_WR32(device, _PBUS, _BUFLWHSCLK_CTRL_P1_REFCLK,
            DRF_DEF(_PBUS, _BUFLWHSCLK_CTRL_P1_REFCLK, _BIAS_CTRL, _DEFAULT)    |
            DRF_DEF(_PBUS, _BUFLWHSCLK_CTRL_P1_REFCLK, _DRV_I,     _DEFAULT)    |
            DRF_DEF(_PBUS, _BUFLWHSCLK_CTRL_P1_REFCLK, _DRV_R,     _DEFAULT)    |
            DRF_NUM(_PBUS, _BUFLWHSCLK_CTRL_P1_REFCLK, _ENABLE,    0x1)         |   // Enable
            DRF_DEF(_PBUS, _BUFLWHSCLK_CTRL_P1_REFCLK, _MISC_CTRL, __PROD));

        LWSWITCH_REG_WR32(device, _PBUS, _BUFLWHSCLK_CTRL_01_REFCLK,
            DRF_DEF(_PBUS, _BUFLWHSCLK_CTRL_01_REFCLK, _BIAS_CTRL, _DEFAULT)    |
            DRF_DEF(_PBUS, _BUFLWHSCLK_CTRL_01_REFCLK, _DRV_I,     _DEFAULT)    |
            DRF_DEF(_PBUS, _BUFLWHSCLK_CTRL_01_REFCLK, _DRV_R,     _DEFAULT)    |
            DRF_NUM(_PBUS, _BUFLWHSCLK_CTRL_01_REFCLK, _ENABLE,
                (chip_device->subengSIOCTRL[0].subengSIOCTRL[0].valid ? 0x1 : 0x0))  |   // Enable
            DRF_DEF(_PBUS, _BUFLWHSCLK_CTRL_01_REFCLK, _MISC_CTRL, __PROD));

        LWSWITCH_REG_WR32(device, _PBUS, _BUFLWHSCLK_CTRL_23_REFCLK,
            DRF_DEF(_PBUS, _BUFLWHSCLK_CTRL_23_REFCLK, _BIAS_CTRL, _DEFAULT)    |
            DRF_DEF(_PBUS, _BUFLWHSCLK_CTRL_23_REFCLK, _DRV_I,     _DEFAULT)    |
            DRF_DEF(_PBUS, _BUFLWHSCLK_CTRL_23_REFCLK, _DRV_R,     _DEFAULT)    |
            DRF_NUM(_PBUS, _BUFLWHSCLK_CTRL_23_REFCLK, _ENABLE,
                (chip_device->subengSIOCTRL[1].subengSIOCTRL[0].valid ? 0x1 : 0x0))  |   // Enable
            DRF_DEF(_PBUS, _BUFLWHSCLK_CTRL_23_REFCLK, _MISC_CTRL, __PROD));

        LWSWITCH_REG_WR32(device, _PBUS, _BUFLWHSCLK_CTRL_45_REFCLK,
            DRF_DEF(_PBUS, _BUFLWHSCLK_CTRL_45_REFCLK, _BIAS_CTRL, _DEFAULT)    |
            DRF_DEF(_PBUS, _BUFLWHSCLK_CTRL_45_REFCLK, _DRV_I,     _DEFAULT)    |
            DRF_DEF(_PBUS, _BUFLWHSCLK_CTRL_45_REFCLK, _DRV_R,     _DEFAULT)    |
            DRF_NUM(_PBUS, _BUFLWHSCLK_CTRL_45_REFCLK, _ENABLE,
                (chip_device->subengSIOCTRL[2].subengSIOCTRL[0].valid ? 0x1 : 0x0))  |   // Enable
            DRF_DEF(_PBUS, _BUFLWHSCLK_CTRL_45_REFCLK, _MISC_CTRL, __PROD));

        LWSWITCH_REG_WR32(device, _PBUS, _BUFLWHSCLK_CTRL_67_REFCLK,
            DRF_DEF(_PBUS, _BUFLWHSCLK_CTRL_67_REFCLK, _BIAS_CTRL, _DEFAULT)    |
            DRF_DEF(_PBUS, _BUFLWHSCLK_CTRL_67_REFCLK, _DRV_I,     _DEFAULT)    |
            DRF_DEF(_PBUS, _BUFLWHSCLK_CTRL_67_REFCLK, _DRV_R,     _DEFAULT)    |
            DRF_NUM(_PBUS, _BUFLWHSCLK_CTRL_67_REFCLK, _ENABLE,
                (chip_device->subengSIOCTRL[3].subengSIOCTRL[0].valid ? 0x1 : 0x0))  |   // Enable
            DRF_DEF(_PBUS, _BUFLWHSCLK_CTRL_67_REFCLK, _MISC_CTRL, __PROD));

        LWSWITCH_REG_WR32(device, _PBUS, _BUFLWHSCLK_CTRL_89_REFCLK,
            DRF_DEF(_PBUS, _BUFLWHSCLK_CTRL_89_REFCLK, _BIAS_CTRL, _DEFAULT)    |
            DRF_DEF(_PBUS, _BUFLWHSCLK_CTRL_89_REFCLK, _DRV_I,     _DEFAULT)    |
            DRF_DEF(_PBUS, _BUFLWHSCLK_CTRL_89_REFCLK, _DRV_R,     _DEFAULT)    |
            DRF_NUM(_PBUS, _BUFLWHSCLK_CTRL_89_REFCLK, _ENABLE,
                (chip_device->subengSIOCTRL[4].subengSIOCTRL[0].valid ? 0x1 : 0x0))  |   // Enable
            DRF_DEF(_PBUS, _BUFLWHSCLK_CTRL_89_REFCLK, _MISC_CTRL, __PROD));

        LWSWITCH_REG_WR32(device, _PBUS, _BUFLWHSCLK_CTRL_1011_REFCLK,
            DRF_DEF(_PBUS, _BUFLWHSCLK_CTRL_1011_REFCLK, _BIAS_CTRL, _DEFAULT)  |
            DRF_DEF(_PBUS, _BUFLWHSCLK_CTRL_1011_REFCLK, _DRV_I,     _DEFAULT)  |
            DRF_DEF(_PBUS, _BUFLWHSCLK_CTRL_1011_REFCLK, _DRV_R,     _DEFAULT)  |
            DRF_NUM(_PBUS, _BUFLWHSCLK_CTRL_1011_REFCLK, _ENABLE,
                (chip_device->subengSIOCTRL[5].subengSIOCTRL[0].valid ? 0x1 : 0x0))  |   // Enable
            DRF_DEF(_PBUS, _BUFLWHSCLK_CTRL_1011_REFCLK, _MISC_CTRL, __PROD));

        LWSWITCH_REG_WR32(device, _PBUS, _BUFLWHSCLK_CTRL_1213_REFCLK,
            DRF_DEF(_PBUS, _BUFLWHSCLK_CTRL_1213_REFCLK, _BIAS_CTRL, _DEFAULT)  |
            DRF_DEF(_PBUS, _BUFLWHSCLK_CTRL_1213_REFCLK, _DRV_I,     _DEFAULT)  |
            DRF_DEF(_PBUS, _BUFLWHSCLK_CTRL_1213_REFCLK, _DRV_R,     _DEFAULT)  |
            DRF_NUM(_PBUS, _BUFLWHSCLK_CTRL_1213_REFCLK, _ENABLE,
                (chip_device->subengSIOCTRL[6].subengSIOCTRL[0].valid ? 0x1 : 0x0))  |   // Enable
            DRF_DEF(_PBUS, _BUFLWHSCLK_CTRL_1213_REFCLK, _MISC_CTRL, __PROD));

        LWSWITCH_REG_WR32(device, _PBUS, _BUFLWHSCLK_CTRL_1415_REFCLK,
            DRF_DEF(_PBUS, _BUFLWHSCLK_CTRL_1415_REFCLK, _BIAS_CTRL, _DEFAULT)  |
            DRF_DEF(_PBUS, _BUFLWHSCLK_CTRL_1415_REFCLK, _DRV_I,     _DEFAULT)  |
            DRF_DEF(_PBUS, _BUFLWHSCLK_CTRL_1415_REFCLK, _DRV_R,     _DEFAULT)  |
            DRF_NUM(_PBUS, _BUFLWHSCLK_CTRL_1415_REFCLK, _ENABLE,
                (chip_device->subengSIOCTRL[7].subengSIOCTRL[0].valid ? 0x1 : 0x0))  |   // Enable
            DRF_DEF(_PBUS, _BUFLWHSCLK_CTRL_1415_REFCLK, _MISC_CTRL, __PROD));

        LWSWITCH_REG_WR32(device, _PBUS, _BUFLWHSCLK_CTRL_1617_REFCLK,
            DRF_DEF(_PBUS, _BUFLWHSCLK_CTRL_1617_REFCLK, _BIAS_CTRL, _DEFAULT)  |
            DRF_DEF(_PBUS, _BUFLWHSCLK_CTRL_1617_REFCLK, _DRV_I,     _DEFAULT)  |
            DRF_DEF(_PBUS, _BUFLWHSCLK_CTRL_1617_REFCLK, _DRV_R,     _DEFAULT)  |
            DRF_NUM(_PBUS, _BUFLWHSCLK_CTRL_1617_REFCLK, _ENABLE,
                (chip_device->subengSIOCTRL[8].subengSIOCTRL[0].valid ? 0x1 : 0x0))  |   // Enable
            DRF_DEF(_PBUS, _BUFLWHSCLK_CTRL_1617_REFCLK, _MISC_CTRL, __PROD));

        swap_clk = DRF_VAL(_SWITCH_REGKEY, _SWAP_CLK_OVERRIDE,
                        _FIELD, device->regkeys.swap_clk);

        LWSWITCH_REG_WR32(device, _PBUS, _LWHS_REFCLK_PAD_CTRL,
            DRF_NUM(_PBUS, _LWHS_REFCLK_PAD_CTRL, _E_CLK,         0xF)      |   // Enable
            DRF_DEF(_PBUS, _LWHS_REFCLK_PAD_CTRL, _E_CLK_CORE,    _DEFAULT) |
            DRF_DEF(_PBUS, _LWHS_REFCLK_PAD_CTRL, _RFU,           _DEFAULT) |
            DRF_NUM(_PBUS, _LWHS_REFCLK_PAD_CTRL, _SWAP_CLK,      swap_clk) |
            DRF_DEF(_PBUS, _LWHS_REFCLK_PAD_CTRL, _SWAP_CLK_CORE, _DEFAULT) |
            DRF_DEF(_PBUS, _LWHS_REFCLK_PAD_CTRL, _E_BG_FORCE,    _DEFAULT) |
            DRF_DEF(_PBUS, _LWHS_REFCLK_PAD_CTRL, _TERM_IMP0,     _DEFAULT) |
            DRF_DEF(_PBUS, _LWHS_REFCLK_PAD_CTRL, _TERM_IMP1,     _DEFAULT) |
            DRF_DEF(_PBUS, _LWHS_REFCLK_PAD_CTRL, _TERM_MODE0,    _DEFAULT) |
            DRF_DEF(_PBUS, _LWHS_REFCLK_PAD_CTRL, _TERM_MODE1,    _DEFAULT));
    }

#else   //defined(DISABLE_CLOCK_INIT)
    LWSWITCH_PRINT(device, WARN,
        "%s: Clock initialization disabled\n",
        __FUNCTION__);
#endif  //defined(DISABLE_CLOCK_INIT)

    return retval;
}

//
// Timer functions
//

void
lwswitch_init_hw_counter_sv10
(
    lwswitch_device *device
)
{
    sv10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_SV10(device);

    //
    // Start the timer free running counter
    //
    LWSWITCH_REG_WR32(device, _PCLOCK, _LWSW_FR_CLK_CNTR_PTIMER_FR_COUNTER_CFG,
        DRF_DEF(_PCLOCK, _LWSW_FR_CLK_CNTR_PTIMER_FR_COUNTER_CFG, _RESET, _ASSERTED));
    LWSWITCH_FLUSH_MMIO(device);

    LWSWITCH_REG_WR32(device, _PCLOCK, _LWSW_FR_CLK_CNTR_PTIMER_FR_COUNTER_CFG,
        DRF_DEF(_PCLOCK, _LWSW_FR_CLK_CNTR_PTIMER_FR_COUNTER_CFG, _COUNT_UPDATE_CYCLES, _EVERY_128) |
        DRF_DEF(_PCLOCK, _LWSW_FR_CLK_CNTR_PTIMER_FR_COUNTER_CFG, _ASYNC_MODE, _ENABLED) |
        DRF_DEF(_PCLOCK, _LWSW_FR_CLK_CNTR_PTIMER_FR_COUNTER_CFG, _RESET, _DEASSERTED) |
        DRF_DEF(_PCLOCK, _LWSW_FR_CLK_CNTR_PTIMER_FR_COUNTER_CFG, _START_COUNT, _ENABLED) |
        DRF_DEF(_PCLOCK, _LWSW_FR_CLK_CNTR_PTIMER_FR_COUNTER_CFG, _CONTINOUS_UPDATE, _ENABLED) |
        DRF_DEF(_PCLOCK, _LWSW_FR_CLK_CNTR_PTIMER_FR_COUNTER_CFG, _SOURCE, _PEX_REFCLK));
    LWSWITCH_FLUSH_MMIO(device);

    chip_device->timer_initialized = LW_TRUE;
}

void
lwswitch_hw_counter_shutdown_sv10
(
    lwswitch_device *device
)
{
    sv10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_SV10(device);
    chip_device->timer_initialized = LW_FALSE;

    //
    // Stop the timer free running counter
    //
    LWSWITCH_REG_WR32(device, _PCLOCK, _LWSW_FR_CLK_CNTR_PTIMER_FR_COUNTER_CFG,
        DRF_DEF(_PCLOCK, _LWSW_FR_CLK_CNTR_PTIMER_FR_COUNTER_CFG, _RESET, _ASSERTED));
    LWSWITCH_FLUSH_MMIO(device);
}

//
// Reads the 36-bit free running counter
//
LwU64
lwswitch_hw_counter_read_counter_sv10
(
    lwswitch_device *device
)
{
    LwU32   timer_cfg;
    LwU64   timer_counter_lo;
    LwU64   timer_counter_hi;
    LwU64   timer_counter = 0;
    sv10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_SV10(device);

    if (chip_device->timer_initialized)
    {
        // Disable counter update
        timer_cfg = LWSWITCH_REG_RD32(device, _PCLOCK, _LWSW_FR_CLK_CNTR_PTIMER_FR_COUNTER_CFG);
        LWSWITCH_REG_WR32(device, _PCLOCK, _LWSW_FR_CLK_CNTR_PTIMER_FR_COUNTER_CFG,
            FLD_SET_DRF(_PCLOCK, _LWSW_FR_CLK_CNTR_PTIMER_FR_COUNTER_CFG, _CONTINOUS_UPDATE, _DISABLED, timer_cfg));

        timer_counter_lo = LWSWITCH_REG_RD32(device, _PCLOCK, _LWSW_FR_CLK_CNTR_PTIMER_FR_COUNTER_CNT0);
        timer_counter_lo = DRF_VAL(_PCLOCK, _LWSW_FR_CLK_CNTR_PTIMER_FR_COUNTER_CNT0, _LSB, timer_counter_lo);
        timer_counter_hi = LWSWITCH_REG_RD32(device, _PCLOCK, _LWSW_FR_CLK_CNTR_PTIMER_FR_COUNTER_CNT1);
        timer_counter_hi = DRF_VAL(_PCLOCK, _LWSW_FR_CLK_CNTR_PTIMER_FR_COUNTER_CNT1, _MSB, timer_counter_hi);

        // Restore counter update
        LWSWITCH_REG_WR32(device, _PCLOCK, _LWSW_FR_CLK_CNTR_PTIMER_FR_COUNTER_CFG, timer_cfg);

        timer_counter = timer_counter_lo | (timer_counter_hi << 32);
    }

    return timer_counter;
}

void
lwswitch_init_clock_gating_sv10
(
    lwswitch_device *device
)
{
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
    LWSWITCH_REG_WR32(device, _PPRIV_PRT_PRT0, _CG1,
        DRF_DEF(_PPRIV_PRT_PRT0, _CG1, _SLCG, __PROD));
    LWSWITCH_REG_WR32(device, _PPRIV_PRT_PRT1, _CG1,
        DRF_DEF(_PPRIV_PRT_PRT1, _CG1, _SLCG, __PROD));
    LWSWITCH_REG_WR32(device, _PPRIV_PRT_PRTS, _CG1,
        DRF_DEF(_PPRIV_PRT_PRTS, _CG1, _SLCG, __PROD));
     LWSWITCH_REG_WR32(device, _PPRIV_SYS, _CG1,
        DRF_DEF(_PPRIV_SYS, _CG1, _SLCG, __PROD));

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

    // XVE
    LWSWITCH_ENG_WR32_SV10(device, XVE, , 0, uc, _XVE, _PRI_XVE_CG,
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
    LWSWITCH_ENG_WR32_SV10(device, XVE, , 0, uc, _XVE, _PRI_XVE_CG1,
        DRF_DEF(_XVE, _PRI_XVE_CG1, _MONITOR_CG_EN, __PROD)     |
        DRF_DEF(_XVE, _PRI_XVE_CG1, _SLCG, __PROD));

    // NPG
    LWSWITCH_NPG_BCAST_WR32_SV10(device, _NPG, _CTRL_CLOCK_GATING,
        DRF_DEF(_NPG, _CTRL_CLOCK_GATING, _CG1_SLCG, __PROD));

    // NPG_PERFMON
    LWSWITCH_NPGPERF_BCAST_WR32_SV10(device, _NPGPERF, _CTRL_CLOCK_GATING,
        DRF_DEF(_NPGPERF, _CTRL_CLOCK_GATING, _CG1_SLCG, __PROD));

    LWSWITCH_NPGPERF_BCAST_WR32_SV10(device, _NPGPERF, _PERF_CTRL_CLOCK_GATING,
        DRF_DEF(_NPGPERF, _PERF_CTRL_CLOCK_GATING, _CG1_SLCG, __PROD));

    // LWLIPT
    LWSWITCH_LWLIPT_BCAST_WR32_SV10(device, _LWLIPT, _CTRL_CLOCK_GATING,
        DRF_DEF(_LWLIPT, _CTRL_CLOCK_GATING, _CG1_SLCG, __PROD) |
        DRF_DEF(_LWLIPT, _CTRL_CLOCK_GATING, _MINION_CG1_SLCG, __PROD));
}
