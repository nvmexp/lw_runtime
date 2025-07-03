/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2016-2019 by LWPU Corporation.  All rights reserved.  All information
 * contained herein is proprietary and confidential to LWPU Corporation.  Any
 * use, reproduction, or disclosure without the written permission of LWPU
 * Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

//*****************************************************
//
// clkgv100.c - GV100+ clock lwwatch routines
//
//*****************************************************

#include "clk.h"
#include "fb.h"
#include "gr.h"
#include "disp.h"
#include "gf10x/fermi_clk.h" // LW_PTRIM_CLK_NAMEMAP_INDEX_HOSTCLK
#include "volta/gv100/dev_fbpa.h"
#include "volta/gv100/dev_trim.h"
#include "volta/gv100/dev_trim_addendum.h"
#include "ctrl/ctrl2080/ctrl2080clkavfs.h"
#include "ctrl/ctrl2080/ctrl2080clk.h"
#include "volta/gv100/dev_timer.h"
#include "volta/gv100/hwproject.h"
#include "g_clk_private.h"           // (rmconfig) implementation prototypes.

/*
 * Mapping between the NAFLL ID and the various LUT registers for that NAFLL
 */
static CLK_NAFLL_ADDRESS_MAP _nafllMap_GV100[] =
{
    {
        CLK_NAFLL_ID_SYS,
        {
            LW_PTRIM_SYS_NAFLL_SYSLUT_READ_ADDR,
            LW_PTRIM_SYS_NAFLL_SYSLUT_READ_DATA,
            LW_PTRIM_SYS_NAFLL_SYSLUT_CFG,
            LW_PTRIM_SYS_NAFLL_SYSLUT_DEBUG2,
            LW_PTRIM_SYS_NAFLL_SYSNAFLL_COEFF,
            CLK_REGISTER_ADDR_UNDEFINED,
            CLK_REGISTER_ADDR_UNDEFINED,
        }
    },
    {
        CLK_NAFLL_ID_XBAR,
        {
            LW_PTRIM_SYS_NAFLL_XBARLUT_READ_ADDR,
            LW_PTRIM_SYS_NAFLL_XBARLUT_READ_DATA,
            LW_PTRIM_SYS_NAFLL_XBARLUT_CFG,
            LW_PTRIM_SYS_NAFLL_XBARLUT_DEBUG2,
            LW_PTRIM_SYS_NAFLL_XBARNAFLL_COEFF,
            CLK_REGISTER_ADDR_UNDEFINED,
            CLK_REGISTER_ADDR_UNDEFINED,
        }
    },
    {
        CLK_NAFLL_ID_GPC0,
        {
            LW_PTRIM_GPC_GPCLUT_READ_ADDR(0),
            LW_PTRIM_GPC_GPCLUT_READ_DATA(0),
            LW_PTRIM_GPC_GPCLUT_CFG(0),
            LW_PTRIM_GPC_GPCLUT_DEBUG2(0),
            LW_PTRIM_GPC_GPCNAFLL_COEFF(0),
            CLK_REGISTER_ADDR_UNDEFINED,
            CLK_REGISTER_ADDR_UNDEFINED,
        }
    },
    {
        CLK_NAFLL_ID_GPC1,
        {
            LW_PTRIM_GPC_GPCLUT_READ_ADDR(1),
            LW_PTRIM_GPC_GPCLUT_READ_DATA(1),
            LW_PTRIM_GPC_GPCLUT_CFG(1),
            LW_PTRIM_GPC_GPCLUT_DEBUG2(1),
            LW_PTRIM_GPC_GPCNAFLL_COEFF(1),
            CLK_REGISTER_ADDR_UNDEFINED,
            CLK_REGISTER_ADDR_UNDEFINED,
        }
    },
    {
        CLK_NAFLL_ID_GPC2,
        {
            LW_PTRIM_GPC_GPCLUT_READ_ADDR(2),
            LW_PTRIM_GPC_GPCLUT_READ_DATA(2),
            LW_PTRIM_GPC_GPCLUT_CFG(2),
            LW_PTRIM_GPC_GPCLUT_DEBUG2(2),
            LW_PTRIM_GPC_GPCNAFLL_COEFF(2),
            CLK_REGISTER_ADDR_UNDEFINED,
            CLK_REGISTER_ADDR_UNDEFINED,
        }
    },
    {
        CLK_NAFLL_ID_GPC3,
        {
            LW_PTRIM_GPC_GPCLUT_READ_ADDR(3),
            LW_PTRIM_GPC_GPCLUT_READ_DATA(3),
            LW_PTRIM_GPC_GPCLUT_CFG(3),
            LW_PTRIM_GPC_GPCLUT_DEBUG2(3),
            LW_PTRIM_GPC_GPCNAFLL_COEFF(3),
            CLK_REGISTER_ADDR_UNDEFINED,
            CLK_REGISTER_ADDR_UNDEFINED,
        }
    },
    {
        CLK_NAFLL_ID_GPC4,
        {
            LW_PTRIM_GPC_GPCLUT_READ_ADDR(4),
            LW_PTRIM_GPC_GPCLUT_READ_DATA(4),
            LW_PTRIM_GPC_GPCLUT_CFG(4),
            LW_PTRIM_GPC_GPCLUT_DEBUG2(4),
            LW_PTRIM_GPC_GPCNAFLL_COEFF(4),
            CLK_REGISTER_ADDR_UNDEFINED,
            CLK_REGISTER_ADDR_UNDEFINED,
        }
    },
    {
        CLK_NAFLL_ID_GPC5,
        {
            LW_PTRIM_GPC_GPCLUT_READ_ADDR(5),
            LW_PTRIM_GPC_GPCLUT_READ_DATA(5),
            LW_PTRIM_GPC_GPCLUT_CFG(5),
            LW_PTRIM_GPC_GPCLUT_DEBUG2(5),
            LW_PTRIM_GPC_GPCNAFLL_COEFF(5),
            CLK_REGISTER_ADDR_UNDEFINED,
            CLK_REGISTER_ADDR_UNDEFINED,
        }
    },
    {
        CLK_NAFLL_ID_GPCS,
        {
            LW_PTRIM_GPC_BCAST_GPCLUT_READ_ADDR,
            LW_PTRIM_GPC_BCAST_GPCLUT_READ_DATA,
            LW_PTRIM_GPC_BCAST_GPCLUT_CFG,
            LW_PTRIM_GPC_BCAST_GPCLUT_DEBUG2,
            LW_PTRIM_GPC_BCAST_GPCNAFLL_COEFF,
            CLK_REGISTER_ADDR_UNDEFINED,
            CLK_REGISTER_ADDR_UNDEFINED,
        }
    },
    {
        CLK_NAFLL_ID_LWD,
        {
            LW_PTRIM_SYS_NAFLL_LWDLUT_READ_ADDR,
            LW_PTRIM_SYS_NAFLL_LWDLUT_READ_DATA,
            LW_PTRIM_SYS_NAFLL_LWDLUT_CFG,
            LW_PTRIM_SYS_NAFLL_LWDLUT_DEBUG2,
            LW_PTRIM_SYS_NAFLL_LWDNAFLL_COEFF,
            CLK_REGISTER_ADDR_UNDEFINED,
            CLK_REGISTER_ADDR_UNDEFINED,
        }
    },
    {
        CLK_NAFLL_ID_HOST,
        {
            LW_PTRIM_SYS_NAFLL_HOSTLUT_READ_ADDR,
            LW_PTRIM_SYS_NAFLL_HOSTLUT_READ_DATA,
            LW_PTRIM_SYS_NAFLL_HOSTLUT_CFG,
            LW_PTRIM_SYS_NAFLL_HOSTLUT_DEBUG2,
            LW_PTRIM_SYS_NAFLL_HOSTNAFLL_COEFF,
            CLK_REGISTER_ADDR_UNDEFINED,
            CLK_REGISTER_ADDR_UNDEFINED,
        }
    },
};

/*!
 * TODO: Generalize to CLK_NAFLL_REG_GET
 * http://lwbugs/2571189
 */
#define CLK_NAFLL_REG_GET_GV100(_nafllIdx,_type)                              \
    (_nafllMap_GV100[_nafllIdx].regAddr[CLK_NAFLL_REG_TYPE_##_type])

static CLK_FR_COUNTER_REG_INFO clkFrCounterRegInfo_GV100 = {
    LW_PTRIM_SYS_FR_CLK_CNTR_PEX_PAD_TCLKS_CFG,
    LW_PTRIM_SYS_PLLS_OUT,
    LW_PTRIM_SYS_FR_CLK_CNTR_PEX_PAD_TCLKS_CNT0,
    LW_PTRIM_SYS_FR_CLK_CNTR_PEX_PAD_TCLKS_CNT1,
};

static CLK_FR_COUNTER_SRC_INFO clkFrCounterSrcInfo_GV100[] = {
    {
        "DRAM",
        LW2080_CTRL_CLK_DOMAIN_MCLK,
        8,
        {
            {LW_PTRIM_SYS_PLLS_OUT_PLLS_O_SRC_SELECT_FBP0_DRAMDIV4_REC_CLK_CMD0_CNTR_PA0,   "MCLK[0]" },
            {LW_PTRIM_SYS_PLLS_OUT_PLLS_O_SRC_SELECT_FBP1_DRAMDIV4_REC_CLK_CMD0_CNTR_PA0,   "MCLK[1]" },
            {LW_PTRIM_SYS_PLLS_OUT_PLLS_O_SRC_SELECT_FBP2_DRAMDIV4_REC_CLK_CMD0_CNTR_PA0,   "MCLK[2]" },
            {LW_PTRIM_SYS_PLLS_OUT_PLLS_O_SRC_SELECT_FBP3_DRAMDIV4_REC_CLK_CMD0_CNTR_PA0,   "MCLK[3]" },
            {LW_PTRIM_SYS_PLLS_OUT_PLLS_O_SRC_SELECT_FBP4_DRAMDIV4_REC_CLK_CMD0_CNTR_PA0,   "MCLK[4]" },
            {LW_PTRIM_SYS_PLLS_OUT_PLLS_O_SRC_SELECT_FBP5_DRAMDIV4_REC_CLK_CMD0_CNTR_PA0,   "MCLK[5]" },
            {LW_PTRIM_SYS_PLLS_OUT_PLLS_O_SRC_SELECT_FBP6_DRAMDIV4_REC_CLK_CMD0_CNTR_PA0,   "MCLK[6]" },
            {LW_PTRIM_SYS_PLLS_OUT_PLLS_O_SRC_SELECT_FBP7_DRAMDIV4_REC_CLK_CMD0_CNTR_PA0,   "MCLK[7]" },
        },
    },
    {
        "GPC",
        LW2080_CTRL_CLK_DOMAIN_GPCCLK,
        6,
        {
            {LW_PTRIM_SYS_PLLS_OUT_PLLS_O_SRC_SELECT_GPC0_GPCCLK_NOBG_PROBE_OUT,        "GPCCLK[0]" },
            {LW_PTRIM_SYS_PLLS_OUT_PLLS_O_SRC_SELECT_GPC1_GPCCLK_NOBG_PROBE_OUT,        "GPCCLK[1]" },
            {LW_PTRIM_SYS_PLLS_OUT_PLLS_O_SRC_SELECT_GPC2_GPCCLK_NOBG_PROBE_OUT,        "GPCCLK[2]" },
            {LW_PTRIM_SYS_PLLS_OUT_PLLS_O_SRC_SELECT_GPC3_GPCCLK_NOBG_PROBE_OUT,        "GPCCLK[3]" },
            {LW_PTRIM_SYS_PLLS_OUT_PLLS_O_SRC_SELECT_GPC4_GPCCLK_NOBG_PROBE_OUT,        "GPCCLK[4]" },
            {LW_PTRIM_SYS_PLLS_OUT_PLLS_O_SRC_SELECT_GPC5_GPCCLK_NOBG_PROBE_OUT,        "GPCCLK[5]" },
        },
    },
    {
        "DISP",
        LW2080_CTRL_CLK_DOMAIN_DISPCLK,
        1,
        {
            {LW_PTRIM_SYS_PLLS_OUT_PLLS_O_SRC_SELECT_SYS_DISPCLK_CTS_PROBE,             "DISPCLK"},
        }
    },
    {
        "HOST",
        LW2080_CTRL_CLK_DOMAIN_HOSTCLK,
        1,
        {
            {LW_PTRIM_SYS_PLLS_OUT_PLLS_O_SRC_SELECT_SYS_HOSTCLK_NOBG_CTS_PROBE,        "HOSTCLK"},
        }
    },
    {
        "HUB",
        LW2080_CTRL_CLK_DOMAIN_HUBCLK,
        1,
        {
            {LW_PTRIM_SYS_PLLS_OUT_PLLS_O_SRC_SELECT_SYS_HUBCLK_NOBG_CTS_PROBE,         "HUBCLK"},
        }
    },
    {
        "LWD",
        LW2080_CTRL_CLK_DOMAIN_LWDCLK,
        1,
        {
            {LW_PTRIM_SYS_PLLS_OUT_PLLS_O_SRC_SELECT_SYS_LWDCLK_NOBG_CTS_PROBE,         "LWDCLK"},
        }
    },
    {
        "PWR",
        LW2080_CTRL_CLK_DOMAIN_PWRCLK,
        1,
        {
            {LW_PTRIM_SYS_PLLS_OUT_PLLS_O_SRC_SELECT_SYS_PWRCLK_NOBG_CTS_PROBE,         "PWRCLK"},
        }
    },
    {
        "SYS",
        LW2080_CTRL_CLK_DOMAIN_SYSCLK,
        1,
        {
            {LW_PTRIM_SYS_PLLS_OUT_PLLS_O_SRC_SELECT_SYS_SYSCLK_NOBG_CTS_PROBE,         "SYSCLK"},
        }
    },
    {
        "SPPLL0",
        LW2080_CTRL_CLK_SOURCE_SPPLL0,
        1,
        {
            {LW_PTRIM_SYS_PLLS_OUT_PLLS_O_SRC_SELECT_SYS_SPPLL0_O,                      "SPPLL0"},
        }
    },
    {
        "SPPLL1",
        LW2080_CTRL_CLK_SOURCE_SPPLL1,
        1,
        {
            {LW_PTRIM_SYS_PLLS_OUT_PLLS_O_SRC_SELECT_SYS_SPPLL1_O,                      "SPPLL1"},
        }
    },
    {
        "UTILS",
        LW2080_CTRL_CLK_DOMAIN_UTILSCLK,
        1,
        {
            {LW_PTRIM_SYS_PLLS_OUT_PLLS_O_SRC_SELECT_SYS_UTILSCLK_CTS_PROBE,            "UTILSCLK"},
        },
    },
    {
        "VCLK",
        LW2080_CTRL_CLK_DOMAIN_VCLK0,
        8,
        {
            {LW_PTRIM_SYS_PLLS_OUT_PLLS_O_SRC_SELECT_SYS_VCLK0_CLK,                     "VCLK0"},
            {LW_PTRIM_SYS_PLLS_OUT_PLLS_O_SRC_SELECT_SYS_VCLK1_CLK,                     "VCLK1"},
            {LW_PTRIM_SYS_PLLS_OUT_PLLS_O_SRC_SELECT_SYS_VCLK2_CLK,                     "VCLK2"},
            {LW_PTRIM_SYS_PLLS_OUT_PLLS_O_SRC_SELECT_SYS_VCLK3_CLK,                     "VCLK3"},
            {LW_PTRIM_SYS_PLLS_OUT_PLLS_O_SRC_SELECT_SYS_VPLL0_O,                       "VPLL0"},
            {LW_PTRIM_SYS_PLLS_OUT_PLLS_O_SRC_SELECT_SYS_VPLL1_O,                       "VPLL1"},
            {LW_PTRIM_SYS_PLLS_OUT_PLLS_O_SRC_SELECT_SYS_VPLL2_O,                       "VPLL2"},
            {LW_PTRIM_SYS_PLLS_OUT_PLLS_O_SRC_SELECT_SYS_VPLL3_O,                       "VPLL3"},
        },
    },
    {
        "XBAR",
        LW2080_CTRL_CLK_DOMAIN_XBARCLK,
        1,
        {
            {LW_PTRIM_SYS_PLLS_OUT_PLLS_O_SRC_SELECT_XBAR_XBARCLK_NOBG_CTS_PROBE,       "XBARCLK"},
        },
    },
};

static LwU32 _clkGetMClkAltFreqKHz_GV100();
static LwU32 _clkGetMClkRefFreqKHz_GV100();
static LwU32 _clkGetHBMPllFreqKHz_GV100();
static LwU32 _clkGetGDDRMPllFreqKHz_GV100();
static LwU32 _clkGetMPllFreqKHz_GV100();

void
clkGetInputSelReg_GV100
(
    LwU32 clkNameMapIndex,
    LwU32 *pReg
)
{
    LwU32 oneSrcInputSelectReg = 0;

    // Find the register address for the Mux Control register.
    switch (clkNameMapIndex)
    {
        case LW_PTRIM_CLK_NAMEMAP_INDEX_LWDCLK:
            oneSrcInputSelectReg = LW_PTRIM_SYS_LWD2CLK_OUT_SWITCH;
            break;
        case LW_PVTRIM_CLK_NAMEMAP_INDEX_DISPCLK:
            oneSrcInputSelectReg = LW_PVTRIM_SYS_DISPCLK_ALT_SWITCH;
            break;
        case LW_PTRIM_CLK_NAMEMAP_INDEX_SYS2CLK:
            oneSrcInputSelectReg = LW_PTRIM_SYS_SYS2CLK_OUT_SWITCH;
            break;
        case LW_PTRIM_CLK_NAMEMAP_INDEX_HUB2CLK:
            oneSrcInputSelectReg = LW_PTRIM_SYS_HUB2CLK_OUT_SWITCH;
            break;
        case LW_PTRIM_CLK_NAMEMAP_INDEX_DRAMCLK:
        case LW_PTRIM_CLK_NAMEMAP_INDEX_REFCLK:
            oneSrcInputSelectReg = LW_PTRIM_SYS_DRAMCLK_ALT_SWITCH;
            break;
        case LW_PVTRIM_CLK_NAMEMAP_INDEX_AZA2BITCLK:
            oneSrcInputSelectReg = LW_PVTRIM_SYS_AZA2XBITCLK_OUT_SWITCH;
            break;
        case LW_PTRIM_CLK_NAMEMAP_INDEX_GPC2CLK:
            oneSrcInputSelectReg = LW_PTRIM_SYS_GPC2CLK_ALT_SWITCH;
            break;
        case LW_PTRIM_CLK_NAMEMAP_INDEX_XBAR2CLK:
            oneSrcInputSelectReg = LW_PTRIM_SYS_XBAR2CLK_ALT_SWITCH;
            break;
        case LW_PTRIM_CLK_NAMEMAP_INDEX_UTILSCLK:
            oneSrcInputSelectReg = LW_PTRIM_SYS_UTILSCLK_OUT_SWITCH;
            break;
        case LW_PTRIM_CLK_NAMEMAP_INDEX_PWRCLK:
            oneSrcInputSelectReg = LW_PTRIM_SYS_PWRCLK_OUT_SWITCH;
            break;
        case LW_PTRIM_CLK_NAMEMAP_INDEX_HOSTCLK:
            oneSrcInputSelectReg = LW_PTRIM_SYS_HOSTCLK_ALT_SWITCH;
            break;

        default:
        {
            LwU32 headNum = FERMI_VCLK_NAME_MAP_INDEX_TO_HEADNUM(clkNameMapIndex);
            if (headNum < pDisp[indexGpu].dispGetRemVpllCfgSize())
            {
                oneSrcInputSelectReg = LW_PVTRIM_SYS_VCLK_ALT_SWITCH(headNum);
            }
            else
            {
                dprintf("lw:   %s: Unsupported Clock(%d) passed in for Bypass Source Select\n", __FUNCTION__, clkNameMapIndex);
                DBG_BREAKPOINT();
                return;
            }
            break;
        }
    }

    if (pReg)
       *pReg = oneSrcInputSelectReg;
}

void
clkGetDividerRegOffset_GV100
(
    LwU32 clkNameMapIndex,
    LwU32 *pRegOffset
)
{
    LwU32 divRegOffset = 0;

    //
    // For root clocks, src LDIV register follows
    // the pattern of OUT_LDIV.
    //
    switch(clkNameMapIndex)
    {
        case LW_PTRIM_CLK_NAMEMAP_INDEX_LWDCLK:
            divRegOffset = LW_PTRIM_SYS_LWD2CLK_OUT_LDIV;
            break;
        case LW_PVTRIM_CLK_NAMEMAP_INDEX_DISPCLK:
            divRegOffset = LW_PVTRIM_SYS_DISPCLK_ALT_LDIV;
            break;
        case LW_PTRIM_CLK_NAMEMAP_INDEX_SYS2CLK:
            divRegOffset = LW_PTRIM_SYS_SYS2CLK_OUT_LDIV;
            break;
        case LW_PTRIM_CLK_NAMEMAP_INDEX_HUB2CLK:
            divRegOffset = LW_PTRIM_SYS_HUB2CLK_OUT_LDIV;
            break;
        case LW_PTRIM_CLK_NAMEMAP_INDEX_GPC2CLK:
            divRegOffset = LW_PTRIM_SYS_GPC2CLK_ALT_LDIV;
            break;
        case LW_PTRIM_CLK_NAMEMAP_INDEX_XBAR2CLK:
            divRegOffset = LW_PTRIM_SYS_XBAR2CLK_ALT_LDIV;
            break;
        case LW_PTRIM_CLK_NAMEMAP_INDEX_UTILSCLK:
            divRegOffset = LW_PTRIM_SYS_UTILSCLK_OUT_LDIV;
            break;
        case LW_PTRIM_CLK_NAMEMAP_INDEX_PWRCLK:
            divRegOffset = LW_PTRIM_SYS_PWRCLK_OUT_LDIV;
            break;
        case LW_PVTRIM_CLK_NAMEMAP_INDEX_AZA2BITCLK:
            divRegOffset = LW_PVTRIM_SYS_AZA2XBITCLK_OUT_LDIV;
            break;
        case LW_PTRIM_CLK_NAMEMAP_INDEX_REFCLK:
        case LW_PTRIM_CLK_NAMEMAP_INDEX_DRAMCLK:
            divRegOffset = LW_PTRIM_SYS_DRAMCLK_ALT_LDIV;
            break;
        case LW_PTRIM_CLK_NAMEMAP_INDEX_HOSTCLK:
            divRegOffset = LW_PTRIM_SYS_HOSTCLK_ALT_LDIV;
            break;
        case LW_PTRIM_CLK_NAMEMAP_INDEX_LEGCLK:
        {
            // These clocks have OSM1 in the alternate path.  Return silently.
            break;
        }
        case LW_PTRIM_CLK_NAMEMAP_INDEX_LTC2CLK:
        case LW_PVTRIM_CLK_NAMEMAP_INDEX_SPDIFCLK:
        {
            dprintf("lw:   %s: Unsupported clock name map index (%d)\n", __FUNCTION__, clkNameMapIndex);
            DBG_BREAKPOINT();
            return;
        }
        default:
        {
            // VCLK registers are handled seperately.
            LwU32 headNum = FERMI_VCLK_NAME_MAP_INDEX_TO_HEADNUM(clkNameMapIndex);
            if (headNum < pDisp[indexGpu].dispGetRemVpllCfgSize())
            {
                divRegOffset = LW_PVTRIM_SYS_VCLK_ALT_LDIV(headNum);
            }
            else
            {
                dprintf("lw:   %s: INFO: Cannot Read the AltSrcDiv for the Name Map Index supplied\n", __FUNCTION__);
                // Return silently as this may get called for derivative clks too.
                return;
            }
            break;
        }
    }

    if (pRegOffset)
       *pRegOffset = divRegOffset;
}

//-----------------------------------------------------
//
// clkReadAltClockSrc_GV100
// This routine helps in determining the source for ALT Path clock.
//
//-----------------------------------------------------
CLKSRCWHICH
clkReadAltClockSrc_GV100(LwU32 clkNameMapIndex)
{
    LwU32 oneSrcInputSelectReg = 0;
    LwU32 oneSrcInputSelectVal = 0;
    LwU32 muxSelect;
    CLKSRCWHICH clkSrc = clkSrcWhich_Default;

    // Default values.
    clkSrc = clkSrcWhich_XTAL;

    //
    // CMOS O/P Is being driven through ALT Path
    // This works for DERIVED  Clock Path too.
    //
    pClk[indexGpu].clkGetInputSelReg(clkNameMapIndex, &oneSrcInputSelectReg);

    oneSrcInputSelectVal = GPU_REG_RD32(oneSrcInputSelectReg);

    // First Read the ALT Path 4:1 Mux
    switch (DRF_VAL(_PTRIM, _SYS_GPC2CLK_REF_SWITCH, _FINALSEL, oneSrcInputSelectVal))
    {
        case LW_PTRIM_SYS_GPC2CLK_REF_SWITCH_FINALSEL_ONESRCCLK:
            //
            // One Src is acting as a source for the BYPASS CLK.
            // In case of One Src Read 4:1 Mux.
            //
            switch (DRF_VAL(_PTRIM, _SYS_GPC2CLK_REF_SWITCH, _ONESRCCLK, oneSrcInputSelectVal))
            {
                case LW_PTRIM_SYS_GPC2CLK_REF_SWITCH_ONESRCCLK_ONESRC0:
                    clkSrc = clkSrcWhich_SPPLL0;
                    break;

                case LW_PTRIM_SYS_GPC2CLK_REF_SWITCH_ONESRCCLK_ONESRC1:
                    clkSrc = clkSrcWhich_SPPLL1;
                    pClk[indexGpu].clkGetOSM1Selection(oneSrcInputSelectVal,
                                                          &clkSrc);
                    break;
                default:
                    dprintf("lw:   %s: Unsupported ONESRC BYPASS SELECT Option\n", __FUNCTION__);
                    clkSrc = clkSrcWhich_Default;
                    break;
            }
            break;

        case LW_PTRIM_SYS_GPC2CLK_REF_SWITCH_FINALSEL_SLOWCLK:
            //
            // Optput is driven from slow 4:1 Mux inputs
            // Read the slow 4:1 Mux input.
            //
            switch (DRF_VAL(_PTRIM, _SYS_GPC2CLK_REF_SWITCH, _SLOWCLK, oneSrcInputSelectVal))
            {
                case LW_PTRIM_SYS_GPC2CLK_REF_SWITCH_SLOWCLK_XTAL_IN:
                    clkSrc = clkSrcWhich_XTAL;
                    break;

                case LW_PTRIM_SYS_GPC2CLK_REF_SWITCH_SLOWCLK_XTAL4X:
                    clkSrc = clkSrcWhich_XTAL4X;
                    break;
                default:
                    dprintf("lw:   %s: Unsupported ONESRC BYPASS SELECT Option\n", __FUNCTION__);
                    clkSrc = clkSrcWhich_Default;
                    break;
            }
            break;

        case LW_PTRIM_SYS_GPC2CLK_REF_SWITCH_FINALSEL_MISCCLK:
            muxSelect = DRF_VAL(_PTRIM, _SYS_GPC2CLK_REF_SWITCH, _MISCCLK, oneSrcInputSelectVal);
            //
            // Read the 8:1 Mux(This is just a CYA in ALT Path and only for symmetry factor in Ref Path)
            // In future give an ASSERT if any of the SYS CORE PLL"s are driving the CYA O/P and hnece
            // the ALT Path.
            //
            switch (muxSelect)
            {
                case LW_PTRIM_SYS_GPC2CLK_REF_SWITCH_MISCCLK_PEX_REFCLK_FAST:
                    clkSrc = clkSrcWhich_PEXREFCLK;
                    break;

                case LW_PTRIM_SYS_GPC2CLK_REF_SWITCH_MISCCLK_HOSTCLK_DIV:
                    if (clkNameMapIndex != LW_PTRIM_CLK_NAMEMAP_INDEX_HOSTCLK)
                    {
                        clkSrc = clkSrcWhich_HOSTCLK;
                    }
                    else
                    {
                        dprintf("lw:   This Clock cannot Drive itself as Source%d\n", clkNameMapIndex);
                        clkSrc = clkSrcWhich_Default;
                        return clkSrcWhich_Ilwalid;
                    }
                    break;

                default:
                    pClk[indexGpu].clkReadAltSwitchMisc(clkNameMapIndex,
                        oneSrcInputSelectVal, &clkSrc);
            }
            break;

        case LW_PTRIM_SYS_GPC2CLK_REF_SWITCH_FINALSEL_TESTORJTAGCLK:
            dprintf("lw:   Error: TESTCLK is not supported.\n");
            break;
        default:
            dprintf("lw:   %s: Unsupported ONESRC BYPASS SELECT Option\n", __FUNCTION__);
            clkSrc = clkSrcWhich_Default;
            break;
    }
    return clkSrc;
}

//-----------------------------------------------------
//
// clkReadRefClockSrc_GV100
// This routine helps in determining the source for Ref Path clock.
//
//-----------------------------------------------------
CLKSRCWHICH
clkReadRefClockSrc_GV100(LwU32 pllNameMapIndex)
{
    LwU32 oneSrcInputSelectVal = 0, oneSrcInputSelectReg = 0;
    LwU32 muxSelect;
    CLKSRCWHICH pllSrc = clkSrcWhich_XTAL;

    if (pllNameMapIndex == LW_PTRIM_PLL_NAMEMAP_INDEX_DRAMPLL)
    {
        pllSrc = clkSrcWhich_REFMPLL;
        return pllSrc;
    }

    switch (pllNameMapIndex)
    {
        case LW_PTRIM_PLL_NAMEMAP_INDEX_GPCPLL:
            oneSrcInputSelectReg = LW_PTRIM_SYS_GPC2CLK_REF_SWITCH;
            break;
        case LW_PTRIM_PLL_NAMEMAP_INDEX_XBARPLL:
            oneSrcInputSelectReg = LW_PTRIM_SYS_XBAR2CLK_REF_SWITCH;
            break;
        case LW_PTRIM_PLL_NAMEMAP_INDEX_SYSPLL:
            oneSrcInputSelectReg = LW_PTRIM_SYS_SYS2CLK_REF_SWITCH;
            break;
        case LW_PTRIM_PLL_NAMEMAP_INDEX_REFMPLL:
            oneSrcInputSelectReg = LW_PTRIM_SYS_REFCLK_REFMPLL_SWITCH;
            break;
        case LW_PVTRIM_CLK_NAMEMAP_INDEX_DISPCLK:
            oneSrcInputSelectReg = LW_PVTRIM_SYS_DISPCLK_REF_SWITCH;
            break;
        default:
        {
            LwU32 headNum = FERMI_VCLK_NAME_MAP_INDEX_TO_HEADNUM(pllNameMapIndex);
            if (headNum < pDisp[indexGpu].dispGetRemVpllCfgSize())
            {
                oneSrcInputSelectReg = LW_PVTRIM_SYS_VCLK_REF_SWITCH(headNum);
            }
            else
            {
                dprintf( "lw:   %s: Unsupported PLL(%d) Option\n", __FUNCTION__, pllNameMapIndex);
                return clkSrcWhich_Ilwalid;
            }
            break;
        }
    }

    oneSrcInputSelectVal = GPU_REG_RD32(oneSrcInputSelectReg);

    // First Read the Ref Path 4:1 Mux
    switch (DRF_VAL(_PTRIM, _SYS_GPC2CLK_REF_SWITCH, _FINALSEL, oneSrcInputSelectVal))
    {
        case LW_PTRIM_SYS_GPC2CLK_REF_SWITCH_FINALSEL_ONESRCCLK:
            //
            // One Src is acting as a source for the BYPASS CLK.
            // In case of One Src Read 4:1 Mux
            //
            switch (DRF_VAL(_PTRIM, _SYS_GPC2CLK_REF_SWITCH, _ONESRCCLK, oneSrcInputSelectVal))
            {
                case LW_PTRIM_SYS_GPC2CLK_REF_SWITCH_ONESRCCLK_ONESRC0:
                    pllSrc = clkSrcWhich_SPPLL0;
                    break;
                case LW_PTRIM_SYS_GPC2CLK_REF_SWITCH_ONESRCCLK_ONESRC1:
                    pllSrc = clkSrcWhich_SPPLL1;
                    pClk[indexGpu].clkGetOSM1Selection(oneSrcInputSelectVal,
                                                          &pllSrc);
                    break;
                default:
                    dprintf("lw:   %s: Unsupported ONESRC REF PATH SELECT Option\n", __FUNCTION__);
                    pllSrc = clkSrcWhich_Default;
                    break;
            }
            break;

        case LW_PTRIM_SYS_GPC2CLK_REF_SWITCH_FINALSEL_SLOWCLK:
            //
            // Optput is driven from slow 4:1 Mux inputs
            // Read the slow 4:1 Mux input.
            //
            switch (DRF_VAL(_PTRIM, _SYS_GPC2CLK_REF_SWITCH, _SLOWCLK, oneSrcInputSelectVal))
            {
                case LW_PTRIM_SYS_GPC2CLK_REF_SWITCH_SLOWCLK_XTAL_IN:
                    pllSrc = clkSrcWhich_XTAL;
                    break;

                case LW_PTRIM_SYS_GPC2CLK_REF_SWITCH_SLOWCLK_XTAL4X:
                    pllSrc = clkSrcWhich_XTAL4X;
                    break;

                default:
                    dprintf("lw:   %s: Unsupported ONESRC REF PATH SELECT Option\n", __FUNCTION__);
                    pllSrc = clkSrcWhich_Default;
                    break;
            }
            break;

        case LW_PTRIM_SYS_GPC2CLK_REF_SWITCH_FINALSEL_MISCCLK:
            muxSelect = DRF_VAL(_PTRIM, _SYS_GPC2CLK_REF_SWITCH, _MISCCLK, oneSrcInputSelectVal);
            //
            // Read the 8:1 Mux(This is just a CYA in ALT Path and only for symmetry factor in Ref Path)
            // In future give an ASSERT if any of the SYS CORE PLL"s are driving the CYA O/P and hnece
            // the ALT Path.
            //
            switch (muxSelect)
            {
                case LW_PTRIM_SYS_GPC2CLK_REF_SWITCH_MISCCLK_PEX_REFCLK_FAST:
                    pllSrc = clkSrcWhich_PEXREFCLK;
                    break;

                case LW_PTRIM_SYS_GPC2CLK_REF_SWITCH_MISCCLK_HOSTCLK_DIV:
                    pllSrc = clkSrcWhich_HOSTCLK;
                    break;

                 // selects the external SLI reference clock
                case LW_PVTRIM_SYS_VCLK_REF_SWITCH_MISCCLK_EXT_REFCLK:
                {
                    LwU32 headNum = FERMI_VCLK_NAME_MAP_INDEX_TO_HEADNUM(pllNameMapIndex);

                    if (headNum >= pDisp[indexGpu].dispGetRemVpllCfgSize())
                    {
                        dprintf("lw:   Invalid Src for the PLL %d\n", pllNameMapIndex);
                        pllSrc = clkSrcWhich_Default;
                        return clkSrcWhich_Ilwalid;
                    }

                   if (FLD_TEST_DRF(_PVTRIM, _SYS_VCLK_REF_SWITCH, _EXT_REFCLK, _NON_QUAL, oneSrcInputSelectVal))
                   {
                        pllSrc = clkSrcWhich_EXTREF;
                   }
                   else
                   {
                       pllSrc = clkSrcWhich_QUALEXTREF;
                   }
                    break;
                }
                default:
                   pClk[indexGpu].clkReadRefSwitchMisc(pllNameMapIndex,
                       oneSrcInputSelectVal, &pllSrc);
            }
            break;
        case LW_PTRIM_SYS_GPC2CLK_REF_SWITCH_FINALSEL_TESTORJTAGCLK:
            dprintf("lw:   Error: TESTCLK is not supported.\n");
            break;
        default:
            dprintf("lw:   %s: Unsupported ONESRC REF PATH SELECT Option\n", __FUNCTION__);
            pllSrc = clkSrcWhich_Default;
            break;
    }

    return pllSrc;
}

/*!
 * @brief Maps a given NAFLL ID to the index in the NAFLL map table
 *
 * @param[in]   nafllId     NAFLL index
 *
 * @return      NAFLL table map index if found
 *              else CLK_NAFLL_ADDRESS_TABLE_IDX_ILWALID
 */
LwU32
clkNafllGetNafllIdx_GV100
(
    LwU32 nafllId
)
{
    LwU32 idx =0;

    for (idx = 0; idx < LW_ARRAY_ELEMENTS(_nafllMap_GV100); idx++)
    {
        if (_nafllMap_GV100[idx].id == nafllId)
        {
            return idx;
        }
    }
    return CLK_NAFLL_ADDRESS_TABLE_IDX_ILWALID;
}

//-----------------------------------------------------
//
// clkNafllLutRead_GV100
// This routine reads the programmed LUT value for a given NAFLL ID.
//
//-----------------------------------------------------
void
clkNafllLutRead_GV100
(
    LwU32 nafllId,
    LwU32 tempIdx
)
{
    LwU32   reg32;
    LwU32   addressVal;
    LwU32   dataVal;
    LwU32   adcStepSizeuV;
    LwU32   lutNumEntries;
    LwU32   lutStride;
    LwU32   lutStrideIdx;
    LwU32   adcCode0;
    LwU32   adcCode1;
    LwU32   ndiv0;
    LwU32   ndiv1;
    LwU32   vfgain0;
    LwU32   vfgain1;
    LwU32   nafllIdx;

    nafllIdx = pClk[indexGpu].clkNafllGetNafllIdx(nafllId);
    if (nafllIdx == CLK_NAFLL_ADDRESS_TABLE_IDX_ILWALID)
    {
        dprintf("NAFLL ID (%d) not found!!!\n", nafllId);
        return;
    }

    // Read the ADC step size
    if (GPU_REG_RD_DRF(_PTRIM, _SYS_NAFLL_SYSNAFLL_LWVDD_ADC_CTRL, _SEL_ADC_LSB) ==
        LW_PTRIM_SYS_NAFLL_SYSNAFLL_LWVDD_ADC_CTRL_SEL_ADC_LSB_6P25MV)
    {
        adcStepSizeuV = LW2080_CTRL_CLK_ADC_STEP_SIZE_6250UV;
    }
    else
    {
        adcStepSizeuV = LW2080_CTRL_CLK_ADC_STEP_SIZE_10000UV;
    }

    // Callwlate the number of LUT entries & LUT stride
    lutNumEntries = ((LW2080_CTRL_CLK_LUT_MAX_VOLTAGE_UV -
                      LW2080_CTRL_CLK_LUT_MIN_VOLTAGE_UV) / adcStepSizeuV);
    lutStride     = lutNumEntries / CLK_LUT_ENTRIES_PER_STRIDE;

    // Get the current temperature index, if not already specified
    if (tempIdx > CLK_LUT_TEMP_IDX_MAX)
    {
        reg32   = CLK_NAFLL_REG_GET_GV100(nafllIdx, LUT_CFG);
        dataVal = GPU_REG_RD32(reg32);
        tempIdx = DRF_VAL(_PTRIM_SYS, _NAFLL_LTCLUT_CFG, _TEMP_INDEX, dataVal);
    }
    dprintf("Temperature Index: %d\n\n", tempIdx);

    // Set the read address now
    addressVal = FLD_SET_DRF_NUM(_PTRIM_SYS, _NAFLL_LTCLUT_READ_ADDR, _OFFSET,
                    (tempIdx * lutStride), 0);
    addressVal = FLD_SET_DRF(_PTRIM_SYS, _NAFLL_LTCLUT_READ_ADDR,
                    _AUTO_INC, _ON, addressVal);
    reg32   = CLK_NAFLL_REG_GET_GV100(nafllIdx, LUT_READ_ADDR);
    GPU_REG_WR32(reg32, addressVal);

    // Now for the actual LUT read
    reg32   = CLK_NAFLL_REG_GET_GV100(nafllIdx, LUT_READ_DATA);
    dprintf("LUT Table: \n");
    dprintf("|===============================================================|\n");
    dprintf("| ADC-code |  Ndiv  |  Vfgain  |   ADC-code |  Ndiv  |  Vfgain  |\n");
    dprintf("|===============================================================|\n");

    // Each DWORD in the LUT can hold two V/F table entries.
    for (lutStrideIdx = 0; lutStrideIdx < lutStride; lutStrideIdx++)
    {
        dataVal = GPU_REG_RD32(reg32);

        adcCode0 = (2 * lutStrideIdx);
        adcCode1 = (2 * lutStrideIdx) + 1;
        ndiv0    = DRF_VAL(_PTRIM_SYS, _NAFLL_LTCLUT_READ_DATA, _VAL0_NDIV,
                           dataVal);
        vfgain0  = DRF_VAL(_PTRIM_SYS, _NAFLL_LTCLUT_READ_DATA, _VAL0_VFGAIN,
                           dataVal);
        ndiv1    = DRF_VAL(_PTRIM_SYS, _NAFLL_LTCLUT_READ_DATA, _VAL1_NDIV,
                           dataVal);
        vfgain1  = DRF_VAL(_PTRIM_SYS, _NAFLL_LTCLUT_READ_DATA, _VAL1_VFGAIN,
                           dataVal);

        dprintf("|    %-4d  |   %-4d |    %-4d  |      %-4d  |   %-4d |    %-4d  |\n",
                   adcCode0,   ndiv0,   vfgain0,    adcCode1,    ndiv1,  vfgain1);
    }
    dprintf("|===============================================================|\n");
}

/**
 * @brief Determines if a given clock output is being driven by a PLL.
 *        i.e. REF PATH or is being driven by ALT PATH.
 *
 * @param[in]   clkMapIndex         Clock namemap index enum
 *
 * @returns TRUE for ALT (Bypass) PATH, FALSE for REF (VCO) PATH
 */
BOOL
clkIsClockDrivenfromBYPASS_GV100
(
    LwU32   clkMapIndex
)
{
    switch (clkMapIndex)
    {
        case LW_PTRIM_CLK_NAMEMAP_INDEX_DRAMCLK:
        {
            // Find the memory type
            LwU32 ddrMode = pFb[indexGpu].fbGetFBIOBroadcastDDRMode();

            // For HBM memory type
            if ((ddrMode == LW_PFB_FBPA_FBIO_BROADCAST_DDR_MODE_HBM1) ||
                (ddrMode == LW_PFB_FBPA_FBIO_BROADCAST_DDR_MODE_HBM2))
            {
                LwU32 cfgData = GPU_REG_RD32(LW_PTRIM_FBPA_HBMPLL_CFG0(0));
                if (FLD_TEST_DRF(_PTRIM, _FBPA_HBMPLL_CFG0, _BYPASSPLL, _ENABLE, cfgData) &&
                    FLD_TEST_DRF(_PTRIM, _FBPA_HBMPLL_CFG0, _SEL_ALT_DRAMCLK, _ENABLE, cfgData))
                {
                    return TRUE;
                }
                else
                {
                    return FALSE;
                }
            }
            else    // GDDR memory type
            {
                return clkIsClockDrivenfromBYPASS_GM200(clkMapIndex);
            }
            break;
        }

        default:
        {
            return clkIsClockDrivenfromBYPASS_GM200(clkMapIndex);
            break;
        }
    }
}

/**
 * @brief To detemine if a given VPLL is enabled
 *
 * @param[in]  vClkNum  VPLL number
 *
 * @return TRUE if the given VPLL is enabled
 */
BOOL clkIsVPLLEnabled_GV100(LwU32 vClkNum)
{
    LwU32 setupCtrl = GPU_REG_RD32(LW_PVTRIM_SYS_VPLL_MISC(vClkNum));

    return FLD_TEST_DRF(_PVTRIM, _SYS_VPLL_MISC, _SETUP_CONTROL_STATUS_ENABLE, _YES, setupCtrl);
}

/**
 * @brief Reads offsets the CFG and COEFF registers of the PLLs/Clocks.
 *
 * @param[in]   PLLorClockIndex     PLL/Clock namemap index enum
 * @param[out]  *pCfgPLLRegOffset   The offset of config PLL register
 * @param[out]  *pCoeffPLLRegOffset The offset of coefficient PLL register
 * @param[out]  *pDivRegoffset      The offset of divider register
 *
 * @returns void
 */
void
clkReadSysCoreRegOffset_GV100
(
    LwU32   PLLorClockIndex,
    LwU32  *pCfgPLLRegOffset,
    LwU32  *pCoeffPLLRegOffset,
    LwU32  *pDivRegoffset
)
{
    // Read the CFG register offset addresss based on Name Map Index.
    if (pCfgPLLRegOffset != NULL)
    {
        *pCfgPLLRegOffset = 0;

        // Operating on Broadcast Mode
        switch(PLLorClockIndex)
        {
            case LW_PTRIM_CLK_NAMEMAP_INDEX_GPC2CLK:
            {
                *pCfgPLLRegOffset = LW_PTRIM_GPC_BCAST_GPCPLL_CFG;
                break;
            }
            case LW_PTRIM_CLK_NAMEMAP_INDEX_XBAR2CLK:
            {
                *pCfgPLLRegOffset = LW_PTRIM_SYS_XBARPLL_CFG;
                break;
            }
            case LW_PTRIM_CLK_NAMEMAP_INDEX_SYS2CLK:
            {
                *pCfgPLLRegOffset = LW_PTRIM_SYS_SYSPLL_CFG;
                break;
            }
            case LW_PTRIM_PLL_NAMEMAP_INDEX_DRAMPLL:
            {
                *pCfgPLLRegOffset = LW_PFB_FBPA_DRAMPLL_CFG;
                break;
            }
            case LW_PTRIM_PLL_NAMEMAP_INDEX_REFMPLL:
            {
                *pCfgPLLRegOffset = LW_PFB_FBPA_REFMPLL_CFG;
                break;
            }
            case LW_PVTRIM_PLL_NAMEMAP_INDEX_SPPLL0:
            {
                *pCfgPLLRegOffset = LW_PVTRIM_SYS_SPPLL0_CFG;
                break;
            }
            case LW_PVTRIM_PLL_NAMEMAP_INDEX_SPPLL1:
            {
                *pCfgPLLRegOffset = LW_PVTRIM_SYS_SPPLL1_CFG;
                break;
            }
            case LW_PVTRIM_CLK_NAMEMAP_INDEX_DISPCLK:
            {
                *pCfgPLLRegOffset = LW_PVTRIM_SYS_DISPPLL_CFG;
                break;
            }
            default:
            {
                // VPLL registers are handled seperately.
                LwU32 headNum = FERMI_VCLK_NAME_MAP_INDEX_TO_HEADNUM(PLLorClockIndex);
                if (headNum < pDisp[indexGpu].dispGetRemVpllCfgSize())
                {
                    *pCfgPLLRegOffset = LW_PVTRIM_SYS_VPLL_CFG(headNum);
                }
                else
                {
                    dprintf("lw:   %s: ERROR: No valid CFG register for  Name Map Index(%d) supplied\n",
                        __FUNCTION__, PLLorClockIndex);
                    DBG_BREAKPOINT();
                    return;
                }
                break;
            }
        }
    }

    // Read the COEFF register offset addresss based on Name Map Index.
    if (pCoeffPLLRegOffset != NULL)
    {
        *pCoeffPLLRegOffset = 0;

        // Operating on Broadcast Mode.
        switch(PLLorClockIndex)
        {
            case LW_PTRIM_CLK_NAMEMAP_INDEX_GPC2CLK:
            {
                *pCoeffPLLRegOffset = LW_PTRIM_GPC_BCAST_GPCPLL_COEFF;
                break;
            }
            case LW_PTRIM_CLK_NAMEMAP_INDEX_XBAR2CLK:
            {
                *pCoeffPLLRegOffset = LW_PTRIM_SYS_XBARPLL_COEFF;
                break;
            }
            case LW_PTRIM_CLK_NAMEMAP_INDEX_SYS2CLK:
            {
                *pCoeffPLLRegOffset = LW_PTRIM_SYS_SYSPLL_COEFF;
                break;
            }
             // FBP registers are handled seperately.
            case LW_PTRIM_PLL_NAMEMAP_INDEX_DRAMPLL:
            {
                *pCoeffPLLRegOffset   = LW_PFB_FBPA_DRAMPLL_COEFF;
                break;
            }
            case LW_PTRIM_PLL_NAMEMAP_INDEX_REFMPLL:
            {
                *pCoeffPLLRegOffset   = LW_PFB_FBPA_REFMPLL_COEFF;
                break;
            }
            // SPPLL registers are handled seperately.
            case LW_PVTRIM_PLL_NAMEMAP_INDEX_SPPLL0:
            {
                *pCoeffPLLRegOffset = LW_PVTRIM_SYS_SPPLL0_COEFF;
                break;
            }
            case LW_PVTRIM_PLL_NAMEMAP_INDEX_SPPLL1:
            {
                *pCoeffPLLRegOffset = LW_PVTRIM_SYS_SPPLL0_COEFF;
                break;
            }
            case LW_PVTRIM_CLK_NAMEMAP_INDEX_DISPCLK:
            {
                *pCoeffPLLRegOffset = LW_PVTRIM_SYS_DISPPLL_COEFF;
                break;
            }
            default:
            {
                // VCLK registers are handled seperately.
                LwU32 headNum = FERMI_VCLK_NAME_MAP_INDEX_TO_HEADNUM(PLLorClockIndex);
                if (headNum < pDisp[indexGpu].dispGetRemVpllCfgSize())
                {
                    *pCoeffPLLRegOffset = LW_PVTRIM_SYS_VPLL_COEFF(headNum);
                }
                else
                {
                    dprintf("lw:   %s: ERROR: No valid COEFF register for  Name Map Index(%d) supplied\n", __FUNCTION__, PLLorClockIndex);
                    DBG_BREAKPOINT();
                    return;
                }
                break;
            }
        }
    }

    // Read the LDIV register offset addresss based on Name Map Index.
    if (pDivRegoffset != NULL)
    {
        *pDivRegoffset = 0;

        // Operating on Broadcast Mode.
        switch(PLLorClockIndex)
        {
            case LW_PTRIM_CLK_NAMEMAP_INDEX_GPC2CLK:
            {
                *pDivRegoffset = LW_PTRIM_GPC_BCAST_GPC2CLK_OUT;
                break;
            }
            case LW_PTRIM_CLK_NAMEMAP_INDEX_XBAR2CLK:
            {
                *pDivRegoffset = LW_PTRIM_SYS_XBAR2CLK_OUT;
                break;
            }
            case LW_PTRIM_CLK_NAMEMAP_INDEX_SYS2CLK:
            {
                *pDivRegoffset = LW_PTRIM_SYS_SYS2CLK_OUT_LDIV;
                break;
            }
            case LW_PTRIM_CLK_NAMEMAP_INDEX_HUB2CLK:
            {
                *pDivRegoffset = LW_PTRIM_SYS_HUB2CLK_OUT_LDIV;
                break;
            }
            case LW_PTRIM_CLK_NAMEMAP_INDEX_LWDCLK:
            {
                *pDivRegoffset = LW_PTRIM_SYS_LWD2CLK_OUT_LDIV;
                break;
            }
            case LW_PVTRIM_CLK_NAMEMAP_INDEX_DISPCLK:
            {
                *pDivRegoffset = LW_PVTRIM_SYS_DISPCLK_OUT;
                break;
            }
            default:
            {
                // VCLK registers are handled seperately.
                LwU32 headNum = FERMI_VCLK_NAME_MAP_INDEX_TO_HEADNUM(PLLorClockIndex);
                if (headNum < pDisp[indexGpu].dispGetRemVpllCfgSize())
                {
                    *pDivRegoffset = LW_PVTRIM_SYS_VCLK_OUT(headNum);
                }
                else
                {
                    dprintf("lw:   %s: ERROR: No valid DIV Value register for  Name Map Index(%d) supplied\n", __FUNCTION__, PLLorClockIndex);
                    DBG_BREAKPOINT();
                    return;
                }
                break;
            }
        }
    }
    return;

}

/**
 * @brief Reads the divider register value for the linear (integral) divider
 *        that sits before the PLL Block inside the OSM(One Src Module) just
 *        after the SPPLL0/1 Sources.
 *
 * @param[in]   pllNameMapIndex     PLL/Clock namemap index enum
 * @param[out]  pllSrc              The PLL src (SPLL0/1)
 *
 * @returns The linear divider value
 */
LwU32
clkReadRefSrcDIV_GV100
(
    LwU32       pllNameMapIndex,
    CLKSRCWHICH pllSrc
)
{
    LwU32 refSrcDiv = 1;
    LwU32 divRegOffset, div;

    if ((pllSrc != clkSrcWhich_SPPLL0) &&  (pllSrc != clkSrcWhich_SPPLL1))
    {
        return refSrcDiv;
    }

    switch(pllNameMapIndex)
    {
        case LW_PTRIM_PLL_NAMEMAP_INDEX_GPCPLL:
            divRegOffset = LW_PTRIM_SYS_GPC2CLK_REF_LDIV;
            break;
        case LW_PTRIM_PLL_NAMEMAP_INDEX_SYSPLL:
            divRegOffset = LW_PTRIM_SYS_SYS2CLK_REF_LDIV;
            break;
        case LW_PTRIM_PLL_NAMEMAP_INDEX_XBARPLL:
            divRegOffset = LW_PTRIM_SYS_XBAR2CLK_REF_LDIV;
            break;
        case LW_PTRIM_PLL_NAMEMAP_INDEX_REFMPLL:
            divRegOffset = LW_PTRIM_SYS_REFCLK_REFMPLL_LDIV;
            break;
        case LW_PVTRIM_CLK_NAMEMAP_INDEX_DISPCLK:
            divRegOffset = LW_PVTRIM_SYS_DISPCLK_REF_LDIV;
            break;
        default:
        {
            LwU32 head;
            for (head = 0; head < pDisp[indexGpu].dispGetRemVpllCfgSize(); ++head)
            {
                if (LW_PVTRIM_PLL_NAMEMAP_INDEX_VPLL(head) == pllNameMapIndex)
                    break;
            }
            if (head < pDisp[indexGpu].dispGetRemVpllCfgSize())
            {
                divRegOffset = LW_PVTRIM_SYS_VCLK_REF_LDIV(head);
            }
            else
            {
                dprintf("lw:   %s: INFO: Cannot Read the RefSrcDiv for the PLL Name Map Index supplied\n", __FUNCTION__);
                // Return silently as this may get called for derivative clks too.
                return refSrcDiv;
            }
            break;
        }
    }

    div = GPU_REG_RD32(divRegOffset);

    // SPPLL0 is input in the OneSrc 0 of all the source muxes.
    if (pllSrc == clkSrcWhich_SPPLL0)
    {
        refSrcDiv = DRF_VAL(_PTRIM, _SYS_GPC2CLK_REF_LDIV, _ONESRC0DIV, div);
    }
    else
    {
        refSrcDiv = DRF_VAL(_PTRIM, _SYS_GPC2CLK_REF_LDIV, _ONESRC1DIV, div);
    }

    // It should not be fractional!
    RM_ASSERT((refSrcDiv) && !(refSrcDiv & 0x01));

    // Colwert it to original Div2x format.
    refSrcDiv = (refSrcDiv + 2) / 2;

    return refSrcDiv;

}

BOOL
clkIsMpllDivBy2Used_GV100
(
    LwU32 mpllCoeffVal
)
{
    if (FLD_TEST_DRF(_PFB, _FBPA_DRAMPLL_COEFF, _SEL_DIVBY2, _ENABLE, mpllCoeffVal))
    {
        return TRUE;
    }

    return FALSE;
}

/**
 * @brief Reads the spread params of the PLL specified, if spread is supported
 *
 * @param[in]   pllNameMapIndex  PLL namemapindex
 * @param[out]  pSpread          Holds the spread settings if available
 */
void clkGetPllSpreadParams_GV100
(   
    LwU32  pllNameMapIndex, 
    void*  pSpread
)
{
    PLLSPREADPARAMS* pSpreadParams = (PLLSPREADPARAMS*) pSpread;
    LwU32 cfg2        = 0;
    LwU32 ssd0        = 0;
    LwU32 ssd1        = 0;
    BOOL bReadSpread = FALSE;

    switch (pllNameMapIndex)
    {
        case LW_PTRIM_PLL_NAMEMAP_INDEX_REFMPLL:
        {
            cfg2  = GPU_REG_RD32(LW_PFB_FBPA_REFMPLL_CFG2);
            ssd0  = GPU_REG_RD32(LW_PFB_FBPA_REFMPLL_SSD0);
            ssd1  = GPU_REG_RD32(LW_PFB_FBPA_REFMPLL_SSD1);
            bReadSpread = TRUE;
            break;
        }
        case LW_PTRIM_PLL_NAMEMAP_INDEX_DRAMPLL:
        {
            bReadSpread = FALSE;
            break;
        }
        default:
        {
            LwU32 headNum = FERMI_VCLK_NAME_MAP_INDEX_TO_HEADNUM(pllNameMapIndex);

            if (headNum < pDisp[indexGpu].dispGetRemVpllCfgSize())
            {
                cfg2 = GPU_REG_RD32(LW_PVTRIM_SYS_VPLL_CFG2(headNum));
                ssd0 = GPU_REG_RD32(LW_PVTRIM_SYS_VPLL_SSD0(headNum));
                ssd1 = GPU_REG_RD32(LW_PVTRIM_SYS_VPLL_SSD1(headNum));
                bReadSpread = TRUE;
            }
        }
    }

    if (bReadSpread)
    {
        pSpreadParams->bSDM    = FLD_TEST_DRF(_PFB, _FBPA_REFMPLL_CFG2, _SSD_EN_SDM, _YES, cfg2);
        pSpreadParams->SDM     = DRF_VAL_SIGNED(_PFB, _FBPA_REFMPLL_SSD0, _SDM_DIN, ssd0);
        pSpreadParams->bSSC    = FLD_TEST_DRF(_PFB, _FBPA_REFMPLL_CFG2, _SSD_EN_SSC, _YES, cfg2);
        pSpreadParams->SSCMin  = DRF_VAL_SIGNED(_PFB, _FBPA_REFMPLL_SSD1, _SDM_SSC_MIN, ssd1);
        pSpreadParams->SSCMax  = DRF_VAL_SIGNED(_PFB, _FBPA_REFMPLL_SSD1, _SDM_SSC_MAX, ssd1);
    }
    else
    {
        pSpreadParams->bSDM    = FALSE;
        pSpreadParams->SDM     = 0;
        pSpreadParams->bSSC    = FALSE;
        pSpreadParams->SSCMin  = 0;
        pSpreadParams->SSCMax  = 0;
    }
}

//-----------------------------------------------------
//
// clkCounterFrequency_GV100
// This function is used to read the clock counters for 
// each clock domain at different tap points in each
// specific clock tree.
//
//-----------------------------------------------------
void
clkCounterFrequency_GV100(LwU32 clkSel, char *pClkDomainName)
{
    LwU32                       tgtClkSrcDef        = 0; // srf def value for _SOURCE field in cfg reg
    LwU32                       clockFreqKhz        = 0;
    LwU32                       i                   = 0;
    LwU32                       j                   = 0;
    CLK_FR_COUNTER_SRC_INFO*    pClkFrCntrSrcInfo   = NULL;
    CLK_FR_COUNTER_SRC_INFO     clkFrCntrSrcInfo;
    CLK_FR_COUNTER_REG_INFO*    pClkFrCntrRegInfo   = NULL;
    LwU32                       numCntrsToRead      = 0;

    if (LW_OK != pClk[indexGpu].clkGetFrCntrInfo(&pClkFrCntrSrcInfo, &pClkFrCntrRegInfo, &numCntrsToRead))
    {
        dprintf("lw: Failed to retrieve free-running counter data\n");
        return;
    }

    if (strncmp(pClkDomainName, "all", CLK_DOMAIN_NAME_STR_LEN))
    {
        for(i = 0; i < numCntrsToRead; i++)
            if (!strncmp(clkFrCounterSrcInfo_GV100[i].clkDomainName, pClkDomainName, CLK_DOMAIN_NAME_STR_LEN))
            {
                numCntrsToRead = i + 1;
                break;
            }
    }

    if (i == numCntrsToRead)
    {
        dprintf("lw: Invalid domain passed: %s\n", pClkDomainName);
    }
    else
    {
        for (;i < numCntrsToRead; i++)
        {
            clkFrCntrSrcInfo = pClkFrCntrSrcInfo[i];

            for (j = 0; j < clkFrCntrSrcInfo.srcNum; j++)
            {
                dprintf("lw: Measured clk frequencies for clk counter %s:\n",
                        clkFrCntrSrcInfo.srcInfo[j].srcName);

                tgtClkSrcDef = DRF_NUM(_PTRIM, _SYS_PLLS_OUT, _PLLS_O_SRC_SELECT,
                                            clkFrCntrSrcInfo.srcInfo[j].srcIdx);

                // Reset/clear the clock counter first
                pClk[indexGpu].clkResetCntr(pClkFrCntrRegInfo->srcReg, pClkFrCntrRegInfo->cfgReg, clkFrCntrSrcInfo.clkDomain, tgtClkSrcDef);

                // Delay for 1us, bug 1953217
                osPerfDelay(1);

                // configure/enable counter now
                pClk[indexGpu].clkEnableCntr(pClkFrCntrRegInfo->srcReg, pClkFrCntrRegInfo->cfgReg, clkFrCntrSrcInfo.clkDomain, tgtClkSrcDef, 0);

                clockFreqKhz = pClk[indexGpu].clkReadFrCounter(pClkFrCntrRegInfo, clkFrCntrSrcInfo.clkDomain, tgtClkSrcDef);

                if (clockFreqKhz)
                {           
                    dprintf("lw:\t\t Source %16s: %d kHz\n",
                            clkFrCntrSrcInfo.srcInfo[j].srcName, clockFreqKhz);
                }
            }
            dprintf("\n");
        }
    }
}

/*!
 * @brief Fetch the FR clock counter data
 *
 * @param[out]   ppClkFrCntrSrcInfo  Clock counter source info
 * @param[out]   ppClkFrCntrRegInfo  Clock counter register info
 * @param[out]   pNumCntrsToRead     Number of FR counters present
 */
LW_STATUS
clkGetFrCntrInfo_GV100
(
    CLK_FR_COUNTER_SRC_INFO** ppClkFrCntrSrcInfo,
    CLK_FR_COUNTER_REG_INFO** ppClkFrCntrRegInfo,
    LwU32*                    pNumCntrsToRead
)
{
    if ((ppClkFrCntrSrcInfo == NULL) &&
        (ppClkFrCntrRegInfo == NULL) &&
        (pNumCntrsToRead == NULL))
    {
        return LW_ERR_GENERIC;
    }

    if (ppClkFrCntrSrcInfo != NULL)
    {
        *ppClkFrCntrSrcInfo = clkFrCounterSrcInfo_GV100;
    }

    if (ppClkFrCntrRegInfo != NULL)
    {
        *ppClkFrCntrRegInfo = &clkFrCounterRegInfo_GV100;
    }

    if (pNumCntrsToRead != NULL)
    {
        *pNumCntrsToRead  = sizeof(clkFrCounterSrcInfo_GV100)/sizeof(clkFrCounterSrcInfo_GV100[0]);
    }

    return LW_OK;
}

/*!
 * @brief Reset clock counter
 *
 * @param[in]   srcReg              Clock counter source register
 * @param[in]   cfgReg              Clock counter config register
 * @param[in]   clkDomain           Clock domain LW2080_CTRL_CLK_DOMAIN_XX - used only Turing+
 * @param[in]   tgtClkSrcDef        Clock source for the counter
 */
LW_STATUS
clkResetCntr_GV100
(
    LwU32 srcReg,
    LwU32 cfgReg,
    LwU32 clkDomain,
    LwU32 tgtClkSrcDef
)
{
    LwU32 data32;

    // Set the clock source
    data32 = GPU_REG_RD32(srcReg);
    data32 = FLD_SET_DRF_NUM(_PTRIM, _SYS_PLLS_OUT, _PLLS_O_SRC_SELECT, tgtClkSrcDef, data32);
    data32 = FLD_SET_DRF(_PTRIM, _SYS_PLLS_OUT, _PLLS_O_SRC_SELECT_CTS_PROBE_ENABLE, _NO, data32);
    data32 = FLD_SET_DRF(_PTRIM, _SYS_PLLS_OUT, _PLLS_O_SRC_SELECT_TCLKOUT_ENABLE, _NO, data32);
    GPU_REG_WR32(srcReg, data32);

    // Now for the reset
    data32 = GPU_REG_RD32(cfgReg);
    data32 = FLD_SET_DRF(_PTRIM, _SYS_FR_CLK_CNTR_PEX_PAD_TCLKS_CFG, _COUNT_UPDATE_CYCLES, _EVERY_64, data32);
    data32 = FLD_SET_DRF(_PTRIM, _SYS_FR_CLK_CNTR_PEX_PAD_TCLKS_CFG, _RESET,               _ASSERTED, data32);
    data32 = FLD_SET_DRF(_PTRIM, _SYS_FR_CLK_CNTR_PEX_PAD_TCLKS_CFG, _START_COUNT,         _DISABLED, data32);
    data32 = FLD_SET_DRF(_PTRIM, _SYS_FR_CLK_CNTR_PEX_PAD_TCLKS_CFG, _CONTINOUS_UPDATE,    _ENABLED, data32);
    GPU_REG_WR32(cfgReg, data32);

    return LW_OK;
}

/*!
 * @brief Enable clock counter
 *
 * @param[in]   srcReg              Clock counter source register
 * @param[in]   cfgReg              Clock counter config register
 * @param[in]   clkDomain           Clock domain LW2080_CTRL_CLK_DOMAIN_XX - used only Turing+
 * @param[in]   tgtClkSrcDef        Clock source for the counter
 * @param[in]   clockInput          Count period in xtal clock cycles (unused)
 */
LW_STATUS
clkEnableCntr_GV100
(
    LwU32 srcReg,
    LwU32 cfgReg,
    LwU32 clkDomain,
    LwU32 tgtClkSrcDef,
    LwU32 clockInput
)
{
    LwU32 data32;

    // Set the clock source
    data32 = GPU_REG_RD32(srcReg);
    data32 = FLD_SET_DRF_NUM(_PTRIM, _SYS_PLLS_OUT, _PLLS_O_SRC_SELECT, tgtClkSrcDef, data32);
    data32 = FLD_SET_DRF(_PTRIM, _SYS_PLLS_OUT, _PLLS_O_SRC_SELECT_CTS_PROBE_ENABLE, _YES, data32);
    data32 = FLD_SET_DRF(_PTRIM, _SYS_PLLS_OUT, _PLLS_O_SRC_SELECT_TCLKOUT_ENABLE, _YES, data32);
    GPU_REG_WR32(srcReg, data32);

    // Now for the actual enable
    data32 = GPU_REG_RD32(cfgReg);
    data32 = FLD_SET_DRF(_PTRIM, _SYS_FR_CLK_CNTR_PEX_PAD_TCLKS_CFG, _RESET, _DEASSERTED, data32);
    GPU_REG_WR32(cfgReg, data32);

    //
    // Enable clock counter.
    // Note : Need to write un-reset and enable signal in different
    // register writes as the source (register block) and destination
    // (FR counter) are on the same clock and far away from each other,
    // so the signals can not reach in the same clock cycle hence some
    // delay is required between signals.
    //
    data32 = GPU_REG_RD32(cfgReg);
    data32 = FLD_SET_DRF(_PTRIM, _SYS_FR_CLK_CNTR_PEX_PAD_TCLKS_CFG, _START_COUNT, _ENABLED, data32);
    GPU_REG_WR32(cfgReg, data32);

    return LW_OK;
}

static LwU64_PARTS
clkGetLwrrentTime(void)
{
    LwU64_PARTS time;
    LwU32 timeLo  = 0;
    LwU32 timeHi  = 0;
    LwU32 timeHi2 = 0;

    do
    {
        timeHi = GPU_REG_RD32(LW_PTIMER_TIME_1);
        timeLo = GPU_REG_RD32(LW_PTIMER_TIME_0);
        // Read TIME_1 again to detect wrap around.
        timeHi2 = GPU_REG_RD32(LW_PTIMER_TIME_1);
    } while (timeHi != timeHi2);

    // Colwert to 64b
    time.parts.lo = timeLo;
    time.parts.hi = timeHi;

    return time;
}

static LwU64_PARTS
clkReadFrCntrVal
(
    LwU32   cfgReg,
    LwU32   cnt0Reg,
    LwU32   cnt1Reg
)
{
    LwU64_PARTS val;
    LwU32 data;

    // Stop updating counter value in register
    data = GPU_REG_RD32(cfgReg);
    data = FLD_SET_DRF(_PTRIM, _SYS_FR_CLK_CNTR_PEX_PAD_TCLKS_CFG, _CONTINOUS_UPDATE, _DISABLED, data);
    GPU_REG_WR32(cfgReg, data);

    // Read counter value
    val.parts.lo = GPU_REG_RD32(cnt0Reg);
    val.parts.hi = GPU_REG_RD32(cnt1Reg);

    // Start updating counter value in register
    data = FLD_SET_DRF(_PTRIM, _SYS_FR_CLK_CNTR_PEX_PAD_TCLKS_CFG, _CONTINOUS_UPDATE, _ENABLED, data);
    GPU_REG_WR32(cfgReg, data);

    return val;
}

/*!
 * @brief Measures frequency from clock counters
 *
 * @param[in]   pClkFrCntrRegInfo   Clock counter register info
 * @param[in]   clkDomain           Clock domain LW2080_CTRL_CLK_DOMAIN_XX
 * @param[in]   tgtClkSrcDef        Clock source for the counter
 *
 * @return measured frequency for given target source definition
 */
LwU32 
clkReadFrCounter_GV100
(
    CLK_FR_COUNTER_REG_INFO *pClkFrCntrRegInfo,
    LwU32                   clkDomain,
    LwU32                   tgtClkSrcDef
)
{
    LwU64_PARTS time0, time1;
    LwU64_PARTS val0, val1;
    LwU64_PARTS timeDiff;
    LwU64_PARTS cntrDiff;
    LwU64_PARTS tmpTimeNs;
    LwU32       result;

    // Read time0
    time0 = clkGetLwrrentTime();

    // Read count0
    val0 = clkReadFrCntrVal(pClkFrCntrRegInfo->cfgReg,
                            pClkFrCntrRegInfo->cntRegLsb,
                            pClkFrCntrRegInfo->cntRegMsb);
    dprintf("lw:   %s: INFO: timeLo=%x timeHi=%x\n", __FUNCTION__, time0.parts.lo, time0.parts.hi);
    dprintf("lw:   %s: INFO: cntrLo=%x cntrHi=%x\n", __FUNCTION__, val0.parts.lo , val0.parts.hi);

    // Configure the wait period to 500 ms
    tmpTimeNs.parts.hi = 0;
    tmpTimeNs.parts.lo = 500 * 1000000;

    do
    {
        // Delay for 1us
        osPerfDelay(1);

        // Read time1
        time1 = clkGetLwrrentTime();
        LwU64_PARTS_SUB(timeDiff, time1, time0);
        LwU64_PARTS_COMPARE(result, tmpTimeNs, timeDiff);

        // Read count1
        val1 = clkReadFrCntrVal(pClkFrCntrRegInfo->cfgReg,
                                pClkFrCntrRegInfo->cntRegLsb,
                                pClkFrCntrRegInfo->cntRegMsb);
    } while (result == 1);
    dprintf("lw:   %s: INFO: timeLo=%x timeHi=%x\n", __FUNCTION__, time1.parts.lo, time1.parts.hi);
    dprintf("lw:   %s: INFO: cntrLo=%x cntrHi=%x\n", __FUNCTION__, val1.parts.lo , val1.parts.hi);

    // Handle overflow case
    LwU64_PARTS_COMPARE(result, val0, val1);
    if (result == 1)
    {
        LwU64_PARTS maxCnt;
        maxCnt.parts.hi = 0xF;
        maxCnt.parts.lo = 0xFFFFFFFF;
        LwU64_PARTS_SUB(cntrDiff, maxCnt, val0);
        LwU64_PARTS_ADD(cntrDiff, cntrDiff, val1);
    }
    else
    {
        LwU64_PARTS_SUB(cntrDiff, val1, val0);
    }

    dprintf("lw: Timer Diff: Hi = %u, Lo = %u\n", timeDiff.parts.hi, timeDiff.parts.lo);
    dprintf("lw: Counter Diff: Hi = %u, Lo = %u\n", cntrDiff.parts.hi, cntrDiff.parts.lo);

    // Callwlate approx freq in MHz
    return (cntrDiff.parts.lo) / (timeDiff.parts.lo / 1000);
}

//-----------------------------------------------------
// clkGetClocks_GV100
//
//-----------------------------------------------------
void clkGetClocks_GV100()
{
    LwU32 i;

    // Dumpout clock frequencies
    dprintf("lw: Crystal  = %4d MHz\n\n", pClk[indexGpu].clkGetClkSrcFreqKHz(clkSrcWhich_XTAL) / 1000);
    dprintf("lw: SPPLL0   = %4d MHz\n\n", pClk[indexGpu].clkGetSppllFreqKHz(0) / 1000);
    dprintf("lw: SPPLL1   = %4d MHz\n\n", pClk[indexGpu].clkGetSppllFreqKHz(1) / 1000);
    dprintf("lw: DispClk  = %4d MHz\n\n", pClk[indexGpu].clkGetDispclkFreqKHz() / 1000);
    dprintf("lw: Hub2Clk  = %4d MHz\n\n", pClk[indexGpu].clkGetHub2ClkFreqKHz() / 1000);
    dprintf("lw: UtilsClk = %4d MHz\n\n", pClk[indexGpu].clkGetUtilsClkFreqKHz() / 1000);
    dprintf("lw: PwrClk   = %4d MHz\n\n", pClk[indexGpu].clkGetPwrClkFreqKHz() / 1000);
    for (i = 0; i < pDisp[indexGpu].dispGetRemVpllCfgSize(); ++i)
    {
         dprintf("lw: VPLL%d   = %4d MHz\n\n", i,
                 pClk[indexGpu].clkGetVClkFreqKHz(i) / 1000);
    }

    dprintf("lw: HostClk    = %4d MHz\n\n", pClk[indexGpu].clkGetHostClkFreqKHz() / 1000);
    dprintf("lw: GpcClk     = %4d MHz\n\n", pClk[indexGpu].clkGetGpc2ClkFreqKHz() / 1000);
    dprintf("lw: XbarClk    = %4d MHz\n\n", pClk[indexGpu].clkGetXbar2ClkFreqKHz() / 1000);
    dprintf("lw: SysClk     = %4d MHz\n\n", pClk[indexGpu].clkGetSys2ClkFreqKHz() / 1000);
    dprintf("lw: LwdClk     = %4d MHz\n\n", pClk[indexGpu].clkGetMsdClkFreqKHz() / 1000);
    dprintf("lw: MClk       = %4d MHz\n\n", pClk[indexGpu].clkGetMClkFreqKHz() / 1000);

    dprintf("\n");
}

//-----------------------------------------------------
//
// clkGetMClkFreqKHz_GV100
//
//-----------------------------------------------------
LwU32 clkGetMClkFreqKHz_GV100()
{
    LwU32 freqKHz = 0;
    LwU32 ddrMode = 0;
    LwU32 nameMapIndex = LW_PTRIM_CLK_NAMEMAP_INDEX_DRAMCLK;

    if (pClk[indexGpu].clkIsClockDrivenfromBYPASS(LW_PTRIM_CLK_NAMEMAP_INDEX_DRAMCLK))
    {
        dprintf("lw: SOURCE MODE: BYPASS\n");
        freqKHz = _clkGetMClkAltFreqKHz_GV100();
    }
    else
    {
        // Need to find the memory type first
        ddrMode = pFb[indexGpu].fbGetFBIOBroadcastDDRMode();

        // Based on the type
        if ((ddrMode == LW_PFB_FBPA_FBIO_BROADCAST_DDR_MODE_HBM1) ||
            (ddrMode == LW_PFB_FBPA_FBIO_BROADCAST_DDR_MODE_HBM2))
        {
            freqKHz = _clkGetHBMPllFreqKHz_GV100();
        }
        else
        {
            freqKHz = _clkGetGDDRMPllFreqKHz_GV100();
        }
    }

    return freqKHz;
}

//-----------------------------------------------------
//
// _clkGetMClkAltFreqKHz_GV100
// This routine reads MCLK ALT OSM frequency
//
//-----------------------------------------------------
static LwU32
_clkGetMClkAltFreqKHz_GV100
(
)
{
    LwU32 srcdiv = 0;
    CLKSRCWHICH clkSrc = clkSrcWhich_Default;

    // N.B. Here source div val is twice the original.
    clkSrc = pClk[indexGpu].clkReadAltClockSrc(LW_PTRIM_CLK_NAMEMAP_INDEX_DRAMCLK);

    dprintf("lw: MCLK SOURCE: %s\n", getClkSrcName(clkSrc));

    srcdiv = pClk[indexGpu].clkReadAltSrcDIV(LW_PTRIM_CLK_NAMEMAP_INDEX_DRAMCLK, clkSrc);
    if (!srcdiv)
    {
        dprintf("lw: %s: MCLK Alt SrcDiv is 0 !!\n", 
                __FUNCTION__);
        srcdiv = 1;
    }

    return (pClk[indexGpu].clkGetClkSrcFreqKHz(clkSrc) * 2) / srcdiv;
}

//-----------------------------------------------------
//
// _clkGetMClkRefFreqKHz_GV100
// This routine reads REFMPLL REFCLK frequency
//
//-----------------------------------------------------
static LwU32
_clkGetMClkRefFreqKHz_GV100
(
)
{
    LwU32 srcdiv = 0;
    CLKSRCWHICH clkSrc = clkSrcWhich_Default;

    // N.B. Here source div val is twice the original.
    clkSrc = pClk[indexGpu].clkReadRefClockSrc(LW_PTRIM_PLL_NAMEMAP_INDEX_REFMPLL);

    dprintf("lw: MCLK SOURCE: %s\n", getClkSrcName(clkSrc));

    srcdiv = pClk[indexGpu].clkReadRefSrcDIV(LW_PTRIM_PLL_NAMEMAP_INDEX_REFMPLL, clkSrc);
    if (!srcdiv)
    {
        dprintf("lw: %s: MCLK Ref SrcDiv is 0 !!\n", 
                __FUNCTION__);
        srcdiv = 1;
    }

    return (pClk[indexGpu].clkGetClkSrcFreqKHz(clkSrc) / srcdiv);
}

//-----------------------------------------------------
//
// _clkGetHBMPllFreqKHz_GV100
// This routine reads MCLK Frequency for HBM Memory type
//
//-----------------------------------------------------
static LwU32
_clkGetHBMPllFreqKHz_GV100
(
)
{
    LwU32 activeFBPA = 0;
    LwU32 numFBPA = 0;
    LwU32 reg32;
    LwU32 mdiv = 0;
    LwU32 ndiv = 0;
    LwU32 pdiv = 0;

    // Read back HBM params
    dprintf("lw: SOURCE: HBM\n");

    //
    // MCLK for HBM (High Bandwidth Memory) is read-only for GV100.
    // Check floorsweeping and replace FBPA broadcast register with a unicast register.
    // We do this because of a bug in reading the broadcast register (Bug 1596385).
    // We can choose any unicast register so long as it is not floorswept, since
    // this is a read-only clock domain and we assume all are programmed the same.
    // We would be in trouble if this were a programmable domain since Clocks 2.x
    // has an affinity for using the same register for reading and writing.
    //
    pGr[indexGpu].grGetActiveFbpaConfig(&activeFBPA, &numFBPA);
    LOWESTBITIDX_32(activeFBPA);

    reg32 = GPU_REG_RD32(LW_PFB_FBPA_0_FBIO_HBMPLL_COEFF + LW_FBPA_PRI_STRIDE * activeFBPA);
    mdiv = DRF_VAL(_PFB, _FBPA_FBIO_HBMPLL_COEFF, _MDIV, reg32);
    ndiv = DRF_VAL(_PFB, _FBPA_FBIO_HBMPLL_COEFF, _NDIV, reg32);
    pdiv = DRF_VAL(_PFB, _FBPA_FBIO_HBMPLL_COEFF, _PLDIV, reg32);

    dprintf("lw: PLL PARAMS for HBMCLK\n");
    dprintf("lw:   MDIV     = %4d\n", mdiv);
    dprintf("lw:   NDIV     = %4d\n", ndiv);
    dprintf("lw:   PLDIV    = %4d\n", pdiv);

    if ((mdiv == 0) || (pdiv == 0)) return 0;

    return (pClk[indexGpu].clkGetClkSrcFreqKHz(clkSrcWhich_XTAL) * ndiv) / (mdiv * pdiv);
}

//-----------------------------------------------------
//
// _clkGetGDDRMPllFreqKHz_GV100
// This routine reads MCLK Frequency for GDDR Memory type
//
//-----------------------------------------------------
static LwU32
_clkGetGDDRMPllFreqKHz_GV100
(
)
{
    LwU32 mclkSource = clkSrcWhich_MPLL;
    LwU32 freqKHz    = 0;

    // Read back DDR params
    dprintf("lw: SOURCE: DDR\n");

    if (LW_OK == pClk[indexGpu].clkGetMClkSrcMode(LW_PTRIM_CLK_NAMEMAP_INDEX_DRAMCLK, &mclkSource))
    {
        if (mclkSource == clkSrcWhich_MPLL)
        {
            dprintf("lw: SOURCE MODE: Cascaded PLL or MPLL\n");
            freqKHz = _clkGetMPllFreqKHz_GV100(LW_PTRIM_CLK_NAMEMAP_INDEX_DRAMCLK);
        }
        else if (mclkSource == clkSrcWhich_REFMPLL)
        {
            dprintf("lw: SOURCE MODE: REFMPLL\n");
            freqKHz = _clkGetMPllFreqKHz_GV100(LW_PTRIM_CLK_NAMEMAP_INDEX_REFCLK);
        }
        else
        {
            dprintf("lw: %s: Invalid Mclk SrcMode (%d). Returning freq zero.\n", 
                    __FUNCTION__, mclkSource);
        }
    }
    else
    {
        dprintf("lw: %s: Error reading DRAMCLK Source Mode. Returning freq zero.\n",
                    __FUNCTION__);
    }

    return freqKHz;
}

//---------------------------------------------------------------
//
// _clkGetMPllFreqKHz_GV100
// This routine reads MPLL Frequency for both REFMPLL and DRAMPLL
//
//---------------------------------------------------------------
static LwU32
_clkGetMPllFreqKHz_GV100
(
    LwU32 pllNameMapIndex
)
{
    LwU32 refFreqKHz;
    LwU32 actualFreqKHz;
    LwU32 reg32;
    LwU32 mdiv = 0;
    LwU32 ndiv = 0;
    LwU32 pdiv = 0;
    PLLSPREADPARAMS spreadParams = {0}; 
    LwU32 cfgRegOffset;
    LwU32 coeffRegOffset;

    if (pllNameMapIndex == LW_PTRIM_CLK_NAMEMAP_INDEX_DRAMCLK)
    {
        dprintf("lw: DRAMCLK SOURCE: REFCLK\n");
        refFreqKHz = _clkGetMPllFreqKHz_GV100(LW_PTRIM_CLK_NAMEMAP_INDEX_REFCLK);
    }
    else
    {
        refFreqKHz = _clkGetMClkRefFreqKHz_GV100();
    }

    dprintf("lw:   Ref freq = %4d MHz\n", refFreqKHz / 1000);

    if (!refFreqKHz)
    {
        dprintf("lw:    Invalid PLL input freq %d for Clock(NameMapIndex =  0x%x). Returning freq zero.\n",
                 refFreqKHz, pllNameMapIndex);
        return 0;
    }

    pClk[indexGpu].clkReadSysCoreRegOffset(pllNameMapIndex, &cfgRegOffset, &coeffRegOffset, NULL);

    //check if PLL is enabled
    reg32 = GPU_REG_RD32(cfgRegOffset);
    if (FLD_TEST_DRF(_PFB, _FBPA_DRAMPLL_CFG, _ENABLE, _NO, reg32))
    {
        dprintf("lw:    PLL is disabled on Ref path for Clock(NameMapIndex =  0x%x). Returning freq zero.\n",
            pllNameMapIndex);
        return 0;
    }

    // read back coefficient values
    reg32 = GPU_REG_RD32(coeffRegOffset);
    mdiv = DRF_VAL(_PFB, _FBPA_DRAMPLL_COEFF, _MDIV, reg32);
    ndiv = DRF_VAL(_PFB, _FBPA_DRAMPLL_COEFF, _NDIV, reg32);
    pdiv = DRF_VAL(_PFB, _FBPA_DRAMPLL_COEFF, _PLDIV, reg32);

    if (pdiv == 0 || mdiv == 0)
    {
        dprintf("lw:    Dividers set to zero for the PLL (NameMapIndex =  0x%x). Returning freq zero.\n", 
            pllNameMapIndex);
        dprintf("lw:    MDIV    = %4d\n", mdiv);
        dprintf("lw:    NDIV    = %4d\n", ndiv);
        return 0;
    }

    if (pllNameMapIndex == LW_PTRIM_PLL_NAMEMAP_INDEX_DRAMPLL)
    {
        if (pClk[indexGpu].clkIsMpllDivBy2Used(reg32))
            pdiv = 2;
        else
            pdiv = 1;
    }

    if (LW_OK != pClk[indexGpu].clkGetPLValue(&pdiv))
    {
        dprintf("lw:    Value of PL divider is invalid (0x%x) for NameMapIndex = 0x%x.\n", 
            pdiv, pllNameMapIndex);
    }

    pClk[indexGpu].clkGetPllSpreadParams(pllNameMapIndex, &spreadParams);

    dprintf("lw: PLL PARAMS for %s\n",
        (pllNameMapIndex == LW_PTRIM_CLK_NAMEMAP_INDEX_DRAMCLK ? "DRAMCLK" : "REFCLK"));
    dprintf("lw:   MDIV     = %4d\n", mdiv);
    dprintf("lw:   NDIV     = %4d\n", ndiv);
    dprintf("lw:   PLDIV    = %4d\n", pdiv);
    dprintf("lw:   bSDM     = %s\n", (spreadParams.bSDM ? "ENABLED" : "DISABLED"));
    dprintf("lw:   SDM      = %4d\n", spreadParams.SDM);

    if (spreadParams.bSDM)
    {
        actualFreqKHz = (refFreqKHz * (LwU64)(ndiv * 8192 + 4096 + spreadParams.SDM)) / (mdiv * pdiv * 8192);

        dprintf("lw:   SSC      = %s\n", (spreadParams.bSSC ? "ENABLED" : "DISABLED"));

        if (spreadParams.bSSC)
        {
            LwS32      minFreqKHz;
            LwS32      maxFreqKHz;
            float      spreadValue;
            SPREADTYPE spread;

            minFreqKHz = (LwS32)((refFreqKHz *
                (LwU64)(ndiv * 8192 + 4096 + spreadParams.SSCMin)) / (mdiv * pdiv * 8192));

            maxFreqKHz = (LwS32)((refFreqKHz *
                (LwU64)(ndiv * 8192 + 4096 + spreadParams.SSCMax)) / (mdiv * pdiv * 8192));

            spreadValue = (float)(spreadParams.SDM - spreadParams.SSCMin) / 
                (ndiv * 8192 + 4096 + spreadParams.SDM);

            if (spreadParams.SDM == spreadParams.SSCMax)
                spread = spread_Down;
            else 
                spread = spread_Center; 

            dprintf("lw:   SSC_MIN      = %4d = %4.3f MHz\n", spreadParams.SSCMin, (float) minFreqKHz/1000);
            dprintf("lw:   SSC_MAX      = %4d = %4.3f MHz\n", spreadParams.SSCMax, (float) maxFreqKHz/1000);
            dprintf("lw:   Actual freq  = %4d MHz\n", actualFreqKHz / 1000);
            dprintf("lw:   SpreadType   = %s\n", (spread == spread_Down ? "Down-spread" : "Center-spread"));
            dprintf("lw:   Spread       = %4.2f%%\n", spreadValue * 100);
        }
    }
    else
    {
        actualFreqKHz = (refFreqKHz * ndiv / (mdiv * pdiv));
    }

    return actualFreqKHz;
}

//-----------------------------------------------------
//
// clkGetGpc2ClkFreqKHz_GV100
//
//-----------------------------------------------------
LwU32 clkGetGpc2ClkFreqKHz_GV100( void )
{
    return pClk[indexGpu].clkGetNafllFreqKHz(CLK_NAFLL_ID_GPCS);
}

//-----------------------------------------------------
//
// clkGetXbar2ClkFreqKHz_GV100
//
//-----------------------------------------------------
LwU32 clkGetXbar2ClkFreqKHz_GV100()
{
    return pClk[indexGpu].clkGetNafllFreqKHz(CLK_NAFLL_ID_XBAR);
}

//-----------------------------------------------------
//
// clkGetSys2ClkFreqKHz_GV100
//
//-----------------------------------------------------
LwU32 clkGetSys2ClkFreqKHz_GV100()
{
    return pClk[indexGpu].clkGetNafllFreqKHz(CLK_NAFLL_ID_SYS);
}

//-----------------------------------------------------
//
// clkGetMsdClkFreqKHz_GV100
//
//-----------------------------------------------------
LwU32 clkGetMsdClkFreqKHz_GV100()
{
    return pClk[indexGpu].clkGetNafllFreqKHz(CLK_NAFLL_ID_LWD);
}

/*!
 * @brief  Read back NAFLL frequency for a given NAFLL ID
 *
 * @param[in] nafllId   NAFLL index
 *
 * @return NAFLL frequency callwlated from dividers read from hardware
 *         0 otherwise
 */
LwU32
clkGetNafllFreqKHz_GV100
(
    LwU32 nafllId
)
{
    LwU32 RefFreqKHz = 405000;     // Hardcode RefFreq for now
    LwU32 Mdiv       = 1;
    LwU32 Ndiv       = 0;
    LwU32 reg32      = 0;
    LwU32 dataVal    = 0;
    LwU32 nafllIdx   = 0xFFFFFFFF;

    nafllIdx = pClk[indexGpu].clkNafllGetNafllIdx(nafllId);

    dprintf("lw: SOURCE: NAFLL\n");
    dprintf("lw:   Ref freq = %4d MHz\n", RefFreqKHz / 1000);

    // Read back the programmed MDiv
    reg32   = CLK_NAFLL_REG_GET_GV100(nafllIdx, NAFLL_COEFF);
    dataVal = GPU_REG_RD32(reg32);
    Mdiv    = DRF_VAL(_PTRIM, _SYS_NAFLL_SYSNAFLL_COEFF, _MDIV, dataVal);
    dprintf("lw:   Mdiv     = %4d\n", Mdiv);

    if (Mdiv == 0) return 0;

    // Read the NDiv
    reg32   = CLK_NAFLL_REG_GET_GV100(nafllIdx, LUT_DEBUG2);
    dataVal = GPU_REG_RD32(reg32);
    Ndiv    = DRF_VAL(_PTRIM, _SYS_NAFLL_SYSLUT_DEBUG2, _NDIV, dataVal);
    dprintf("lw:   Ndiv     = %4d\n", Ndiv);

    // Callwlate back the freq value
    return (RefFreqKHz * Ndiv / Mdiv);
}
