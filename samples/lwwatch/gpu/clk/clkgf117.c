/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2011-2014 by LWPU Corporation.  All rights reserved.  All information
 * contained herein is proprietary and confidential to LWPU Corporation.  Any
 * use, reproduction, or disclosure without the written permission of LWPU
 * Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

//*****************************************************
//
// clkgf117.c - GF117 Clock lwwatch routines 
// 
//*****************************************************
#include "os.h"
#include "inst.h"
#include "hal.h"
#include "clk.h"
#include "inst.h"
#include "print.h"
#include "hwref/lwutil.h"
#include "gpuanalyze.h"
#include "inc/gf10x/fermi_clk.h"
#include "fermi/gf117/hwproject.h"
#include "fermi/gf117/dev_trim.h"
#include "fermi/gf117/dev_host.h"
#include "fermi/gf117/dev_lw_xve.h"

#include "g_clk_private.h"           // (rmconfig) implementation prototypes.

static CLK_COUNTER_INFO clkCounterNcdispInfo_GF117 = {
  LW_PVTRIM_SYS_CLK_CNTR_NCDISP_CFG,
  LW_PVTRIM_SYS_CLK_CNTR_NCDISP_CNT,
  8,
  {
      {LW_PVTRIM_SYS_CLK_CNTR_NCDISP_CFG_SOURCE_SPPLL0_SRC,  "SPPLL0_SRC"}, 
      {LW_PVTRIM_SYS_CLK_CNTR_NCDISP_CFG_SOURCE_SPPLL0_OUT,  "SPPLL0_OUT"}, 
      {LW_PVTRIM_SYS_CLK_CNTR_NCDISP_CFG_SOURCE_SPPLL1_SRC,  "SPPLL1_SRC"}, 
      {LW_PVTRIM_SYS_CLK_CNTR_NCDISP_CFG_SOURCE_SPPLL1_OUT,  "SPPLL1_OUT"}, 
      {LW_PVTRIM_SYS_CLK_CNTR_NCDISP_CFG_SOURCE_XTAL4X_OUT,  "XTAL4X_OUT"}, 
      {LW_PVTRIM_SYS_CLK_CNTR_NCDISP_CFG_SOURCE_XTAL16X_OUT, "XTAL16X_OUT"}, 
      {LW_PVTRIM_SYS_CLK_CNTR_NCDISP_CFG_SOURCE_PIOR0_CLK,   "PIOR0_CLK"}, 
      {LW_PVTRIM_SYS_CLK_CNTR_NCDISP_CFG_SOURCE_DISPCLK,     "DISPCLK"} 
  },
};

static CLK_COUNTER_INFO clkCounterVclksInfo_GF117 = {
  LW_PVTRIM_SYS_CLK_CNTR_VCLKS_CFG,
  LW_PVTRIM_SYS_CLK_CNTR_VCLKS_CNT,
  12,
  {
      {LW_PVTRIM_SYS_CLK_CNTR_VCLKS_CFG_SOURCE_VCLK0,      "VCLK0"}, 
      {LW_PVTRIM_SYS_CLK_CNTR_VCLKS_CFG_SOURCE_VPLL_REF0,  "VPLL_REF0"}, 
      {LW_PVTRIM_SYS_CLK_CNTR_VCLKS_CFG_SOURCE_ALT_VCLK0,  "ALT_VCLK0"}, 
      {LW_PVTRIM_SYS_CLK_CNTR_VCLKS_CFG_SOURCE_VCLK1,      "VCLK1"}, 
      {LW_PVTRIM_SYS_CLK_CNTR_VCLKS_CFG_SOURCE_VPLL_REF1,  "VPLL_REF1"}, 
      {LW_PVTRIM_SYS_CLK_CNTR_VCLKS_CFG_SOURCE_ALT_VCLK1,  "ALT_VCLK1"}, 
      {LW_PVTRIM_SYS_CLK_CNTR_VCLKS_CFG_SOURCE_VCLK2,      "VCLK2"}, 
      {LW_PVTRIM_SYS_CLK_CNTR_VCLKS_CFG_SOURCE_VPLL_REF2,  "VPLL_REF2"}, 
      {LW_PVTRIM_SYS_CLK_CNTR_VCLKS_CFG_SOURCE_ALT_VCLK2,  "ALT_VCLK2"}, 
      {LW_PVTRIM_SYS_CLK_CNTR_VCLKS_CFG_SOURCE_VCLK3,      "VCLK3"}, 
      {LW_PVTRIM_SYS_CLK_CNTR_VCLKS_CFG_SOURCE_VPLL_REF3,  "VPLL_REF3"}, 
      {LW_PVTRIM_SYS_CLK_CNTR_VCLKS_CFG_SOURCE_ALT_VCLK3,  "ALT_VCLK3"}
  },
};

static CLK_COUNTER_INFO clkCounterNcltcclkInfo_GF117 = {
  LW_PTRIM_FBPA_BCAST_CLK_CNTR_NCLTCCLK_CFG,
  LW_PTRIM_FBPA_BCAST_CLK_CNTR_NCLTCCLK_CNT,
  5,
  {
      {LW_PTRIM_FBPA_BCAST_CLK_CNTR_NCLTCCLK_CFG_SOURCE_DRAMDIV2_REC_CLK0, "DRAMDIV2_REC_CLK0"}, 
      {LW_PTRIM_FBPA_BCAST_CLK_CNTR_NCLTCCLK_CFG_SOURCE_DRAMDIV2_REC_CLK1, "DRAMDIV2_REC_CLK1"}, 
      {LW_PTRIM_FBPA_BCAST_CLK_CNTR_NCLTCCLK_CFG_SOURCE_DRAMDIV4_REC_CLK0, "DRAMDIV4_REC_CLK0"}, 
      {LW_PTRIM_FBPA_BCAST_CLK_CNTR_NCLTCCLK_CFG_SOURCE_DRAMDIV4_REC_CLK1, "DRAMDIV4_REC_CLK1"}, 
      {LW_PTRIM_FBPA_BCAST_CLK_CNTR_NCLTCCLK_CFG_SOURCE_LTCCLK,            "LTCCLK"}
  },
};

static CLK_COUNTER_INFO clkCounterNcosmoreInfo_GF117 = {
  LW_PTRIM_SYS_CLK_CNTR_NCOSMCORE_CFG,
  LW_PTRIM_SYS_CLK_CNTR_NCOSMCORE_CNT,
  9,
  {
      {LW_PTRIM_SYS_CLK_CNTR_NCOSMCORE_CFG_SOURCE_ALT_SYS2CLK,    "ALT_SYS2CLK"}, 
      {LW_PTRIM_SYS_CLK_CNTR_NCOSMCORE_CFG_SOURCE_ALT_HUB2CLK,    "ALT_HUB2CLK"}, 
      {LW_PTRIM_SYS_CLK_CNTR_NCOSMCORE_CFG_SOURCE_ALT_LEGCLK,     "ALT_LEGCLK"}, 
      {LW_PTRIM_SYS_CLK_CNTR_NCOSMCORE_CFG_SOURCE_ALT_MSDCLK,     "ALT_MSDCLK"}, 
      {LW_PTRIM_SYS_CLK_CNTR_NCOSMCORE_CFG_SOURCE_SYSPLL_REFCLK,  "SYSPLL_REFCLK"}, 
      {LW_PTRIM_SYS_CLK_CNTR_NCOSMCORE_CFG_SOURCE_REFMPLL_REFCLK, "REFMPLL_REFCLK"}, 
      {LW_PTRIM_SYS_CLK_CNTR_NCOSMCORE_CFG_SOURCE_UTILSCLK,       "UTILSCLK"}, 
      {LW_PTRIM_SYS_CLK_CNTR_NCOSMCORE_CFG_SOURCE_PWRCLK,         "PWRCLK"}, 
      {LW_PTRIM_SYS_CLK_CNTR_NCOSMCORE_CFG_SOURCE_HOSTCLK,        "HOSTCLK"}, 
      {LW_PTRIM_SYS_CLK_CNTR_NCOSMCORE_CFG_SOURCE_XCLK,           "XCLK"}, 
      {LW_PTRIM_SYS_CLK_CNTR_NCOSMCORE_CFG_SOURCE_TXCLK,          "TXCLK"} 
  },
};

static CLK_COUNTER_INFO clkCounterNcltcpllInfo_GF117 = {
  LW_PTRIM_SYS_CLK_CNTR_NCLTCPLL_CFG,
  LW_PTRIM_SYS_CLK_CNTR_NCLTCPLL_CNT,
  9,
  {
      {LW_PTRIM_SYS_CLK_CNTR_NCLTCPLL_CFG_SOURCE_LTC2CLK,         "LTC2CLK"}, 
      {LW_PTRIM_SYS_CLK_CNTR_NCLTCPLL_CFG_SOURCE_XBAR2CLK,        "XBAR2CLK"}, 
      {LW_PTRIM_SYS_CLK_CNTR_NCLTCPLL_CFG_SOURCE_GPCPLL_REFCLK,   "GPCPLL_REFCLK"}, 
      {LW_PTRIM_SYS_CLK_CNTR_NCLTCPLL_CFG_SOURCE_LTCPLL_REFCLK,   "LTCPLL_REFCLK"}, 
      {LW_PTRIM_SYS_CLK_CNTR_NCLTCPLL_CFG_SOURCE_XBARPLL_REFCLK,  "XBARPLL_REFCLK"}, 
      {LW_PTRIM_SYS_CLK_CNTR_NCLTCPLL_CFG_SOURCE_ALT_GPCCLK,      "ALT_GPCCLK"}, 
      {LW_PTRIM_SYS_CLK_CNTR_NCLTCPLL_CFG_SOURCE_ALT_LTCCLK,      "ALT_LTCCLK"}, 
      {LW_PTRIM_SYS_CLK_CNTR_NCLTCPLL_CFG_SOURCE_ALT_XBARCLK,     "ALT_XBARCLK"}, 
      {LW_PTRIM_SYS_CLK_CNTR_NCLTCPLL_CFG_SOURCE_ALT_DRAMCLK,     "ALT_DRAMCLK"} 
  },
};
//-----------------------------------------------------
//
// clkReadHostClk
// Function to read the present state of the HostClock.
//
//-----------------------------------------------------
LwU32 clkGetHostClkFreqKHz_GF117()
{
    LwU32 clkSrc, clkSrcHostClk;
    LwU32 freq = 0;

    clkSrc = GPU_REG_RD32(LW_PHOST_SYS_CLKSRC);
    clkSrcHostClk = DRF_VAL(_PHOST, _SYS_CLKSRC, _HOSTCLK, clkSrc);

    switch (clkSrcHostClk)
    {
        case LW_PHOST_SYS_CLKSRC_HOSTCLK_BYPASSCLK:
            freq = clkGenReadClk_FERMI(LW_PTRIM_CLK_NAMEMAP_INDEX_HOSTCLK);
            return freq;

        case LW_PHOST_SYS_CLKSRC_HOSTCLK_TESTCLK:
            dprintf("lw: %s: TESTCLK is not supported.\n", __FUNCTION__);
            DBG_BREAKPOINT();
            freq = -1;
            break;

        case LW_PHOST_SYS_CLKSRC_HOSTCLK_XCLK:
        case LW_PHOST_SYS_CLKSRC_HOSTCLK_XCLK_GEN2:
        case LW_PHOST_SYS_CLKSRC_HOSTCLK_XCLK_GEN3:
        {
            if (FLD_TEST_DRF(_PHOST, _SYS_CLKSRC, _HOSTCLK_AUTO, _ENABLE, clkSrc))
            {
                LwU32 linkControlReg, linkControlStatus, linkSpeed;

                linkControlReg = DEVICE_BASE(LW_PCFG) + LW_XVE_LINK_CONTROL_STATUS;
                linkControlStatus = GPU_REG_RD32(linkControlReg);

                linkSpeed = DRF_VAL(_XVE, _LINK_CONTROL_STATUS, _LINK_SPEED, 
                                    linkControlStatus);
                
                switch (linkSpeed)
                {
                    case LW_XVE_LINK_CONTROL_STATUS_LINK_SPEED_2P5:
                        clkSrc = clkSrcWhich_XCLK;
                        break;

                    case LW_XVE_LINK_CONTROL_STATUS_LINK_SPEED_5P0:
                        clkSrc = clkSrcWhich_XCLK3XDIV2;
                        break;

                    case LW_XVE_LINK_CONTROL_STATUS_LINK_SPEED_8P0:
                        clkSrc = clkSrcWhich_XCLKGEN3;
                        break;

                    default:
                        dprintf("lw: %s: Unknown Link speed %d read from 0x%08x "
                           "register\n", __FUNCTION__, linkSpeed, linkControlReg); 
                        DBG_BREAKPOINT();
                        freq = -1;
                        break;
                }
            }
            else
            {
                if (clkSrcHostClk != LW_PHOST_SYS_CLKSRC_HOSTCLK_XCLK_GEN3)
                {
                    clkSrc = (clkSrcHostClk == LW_PHOST_SYS_CLKSRC_HOSTCLK_XCLK_GEN2) ?
                        clkSrcWhich_XCLK3XDIV2 : clkSrcWhich_XCLK;
                }
                else
                {
                    clkSrc = clkSrcWhich_XCLKGEN3;
                }
            }

            break;
        }

        case LW_PHOST_SYS_CLKSRC_HOSTCLK_PEX_REFCLK:
            clkSrc = clkSrcWhich_PEXREFCLK;
            break;

        case LW_PHOST_SYS_CLKSRC_HOSTCLK_XCLK_500:
            clkSrc = clkSrcWhich_XCLK500;
            break;

        default:
            dprintf( "lw: %s: We should never hit the default case in the switch on CLKSRC_HOSTCLK since all the possibilities have been exhausted\n", __FUNCTION__);
            DBG_BREAKPOINT();
            freq = -1;
            break;
    }

    freq = pClk[indexGpu].clkGetClkSrcFreqKHz(clkSrc);
    dprintf("lw: HostClock is driven by %s \n", getClkSrcName(clkSrc));

    return freq;
}

LW_STATUS
clkGetMClkSrcMode_GF117
(
    LwU32 clkMapIndex, 
    LwU32* pMclkSource
)
{
    LwU32 srcModeReg, srcMode;

    if (clkMapIndex != LW_PTRIM_CLK_NAMEMAP_INDEX_DRAMCLK)
        return LW_ERR_GENERIC;

    srcModeReg = GPU_REG_RD32(LW_PTRIM_SYS_FBIO_MODE_SWITCH);
    srcMode = DRF_VAL(_PTRIM, _SYS_FBIO_MODE_SWITCH, _DRAMCLK_MODE, srcModeReg);

    switch (srcMode)
    {
        case LW_PTRIM_SYS_FBIO_MODE_SWITCH_DRAMCLK_MODE_ONESOURCE:
        {
            LwU32 mclkAltSrcReg = GPU_REG_RD32(LW_PTRIM_SYS_DRAMCLK_ALT_SWITCH);
            *pMclkSource = DRF_VAL(_PTRIM, _SYS_DRAMCLK_ALT_SWITCH, _ONESRCCLK, 
                                   mclkAltSrcReg) ? clkSrcWhich_SPPLL1 : clkSrcWhich_SPPLL0;
            break;
        }
        case LW_PTRIM_SYS_FBIO_MODE_SWITCH_DRAMCLK_MODE_DRAMPLL:
        {
            *pMclkSource = clkSrcWhich_MPLL;
            break;
        }
        case LW_PTRIM_SYS_FBIO_MODE_SWITCH_DRAMCLK_MODE_REFMPLL:
        {
            *pMclkSource = clkSrcWhich_REFMPLL;
            break;
        }
        default:
        {
            dprintf("lw: %s : Invalid SrcMode in LW_PTRIM_SYS_FBIO_MODE_SWITCH (%d)\n",
                    __FUNCTION__, srcMode);
            *pMclkSource = clkSrcWhich_Default;
            return LW_ERR_GENERIC;
        }
    }

    return LW_OK;
}

BOOL
clkIsMpllDivBy2Used_GF117
(
    LwU32 mpllCoeffVal
)
{
    if (FLD_TEST_DRF(_PTRIM, _FBPA_DRAMPLL_COEFF, _SEL_DIVBY2, _ENABLE, mpllCoeffVal))
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
void clkGetPllSpreadParams_GF117
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

    if (pllNameMapIndex == LW_PTRIM_PLL_NAMEMAP_INDEX_REFMPLL)
    {
        cfg2  = GPU_REG_RD32(LW_PTRIM_FBPA_BCAST_REFMPLL_CFG2);
        ssd0  = GPU_REG_RD32(LW_PTRIM_FBPA_BCAST_REFMPLL_SSD0);
        ssd1  = GPU_REG_RD32(LW_PTRIM_FBPA_BCAST_REFMPLL_SSD1);
        bReadSpread = TRUE;
    }

    if (bReadSpread)
    {
        pSpreadParams->bSDM    = FLD_TEST_DRF(_PTRIM, _FBPA_BCAST_REFMPLL_CFG2, _SSD_EN_SDM, _YES, cfg2);
        pSpreadParams->SDM     = DRF_VAL_SIGNED(_PTRIM, _FBPA_BCAST_REFMPLL_SSD0, _SDM_DIN, ssd0);
        pSpreadParams->bSSC    = FLD_TEST_DRF(_PTRIM, _FBPA_BCAST_REFMPLL_CFG2, _SSD_EN_SSC, _YES, cfg2);
        pSpreadParams->SSCMin  = DRF_VAL_SIGNED(_PTRIM, _FBPA_BCAST_REFMPLL_SSD1, _SDM_SSC_MIN, ssd1);
        pSpreadParams->SSCMax  = DRF_VAL_SIGNED(_PTRIM, _FBPA_BCAST_REFMPLL_SSD1, _SDM_SSC_MAX, ssd1);
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

CLK_COUNTER_INFO*
clkCounterFreqNcdispInfo_GF117()
{
    return &clkCounterNcdispInfo_GF117;
}

CLK_COUNTER_INFO*
clkCounterFreqNcltcclkInfo_GF117()
{
    return &clkCounterNcltcclkInfo_GF117;
}

CLK_COUNTER_INFO*
clkCounterFreqNcltcpllInfo_GF117()
{
    return &clkCounterNcltcpllInfo_GF117;
}

CLK_COUNTER_INFO*
clkCounterFreqVclksInfo_GF117()
{
    return &clkCounterVclksInfo_GF117;
}

CLK_COUNTER_INFO*
clkCounterFreqNcosmoreInfo_GF117()
{
    return &clkCounterNcosmoreInfo_GF117;
}
