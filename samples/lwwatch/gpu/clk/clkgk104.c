/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2010-2018 by LWPU Corporation.  All rights reserved.  All information
 * contained herein is proprietary and confidential to LWPU Corporation.  Any
 * use, reproduction, or disclosure without the written permission of LWPU
 * Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

//*****************************************************
//
// clkgk104.c - GK104 Clock lwwatch routines 
// 
//*****************************************************

#include "os.h"
#include "hal.h"
#include "chip.h"
#include "inst.h"
#include "print.h"
#include "hwref/lwutil.h"
#include "gpuanalyze.h"
#include "clk.h"
#include "fb.h"
#include "disp.h"
#include "inc/gf10x/fermi_clk.h"
#include "kepler/gk104/dev_trim.h"
#include "kepler/gk104/dev_trim_addendum.h"
#include "kepler/gk104/dev_fbpa.h"
#include "disp/v02_03/dev_disp.h"
#include "g_clk_private.h"           // (rmconfig) implementation prototypes.


#define LW_PTRIM_SYS_CORE_SEL_VCO_CLK_VCO    1
#define LW_PTRIM_SYS_CLK_REF_SWITCH(i)                      (LW_PTRIM_SYS_GPC2CLK_REF_SWITCH + (i * (LW_PTRIM_SYS_LTC2CLK_REF_SWITCH - LW_PTRIM_SYS_GPC2CLK_REF_SWITCH)))
#define LW_PTRIM_SYS_CLK_ALT_SWITCH(i)                      (LW_PTRIM_SYS_GPC2CLK_ALT_SWITCH + (i * (LW_PTRIM_SYS_LTC2CLK_ALT_SWITCH - LW_PTRIM_SYS_GPC2CLK_ALT_SWITCH)))
#define LW_PTRIM_SYS_PLL_CFG(i)                             (LW_PTRIM_SYS_LTCPLL_CFG + ((i-1) * (LW_PTRIM_SYS_XBARPLL_CFG - LW_PTRIM_SYS_LTCPLL_CFG)))
#define LW_PTRIM_SYS_PLL_COEFF(i)                           (LW_PTRIM_SYS_LTCPLL_COEFF + ((i-1) * (LW_PTRIM_SYS_XBARPLL_COEFF - LW_PTRIM_SYS_LTCPLL_COEFF)))
#define LW_PTRIM_SYS_CLK_OUT(i)                             (LW_PTRIM_SYS_LTC2CLK_OUT + ((i-1) * (LW_PTRIM_SYS_XBAR2CLK_OUT - LW_PTRIM_SYS_LTC2CLK_OUT)))
// These indexed defines do not work for Root Clocks as there is no ALT/REF LDIV on root clocks. 
#define LW_PTRIM_SYS_CLK_ALT_LDIV(i)                        (LW_PTRIM_SYS_GPC2CLK_ALT_LDIV + (i * (LW_PTRIM_SYS_LTC2CLK_ALT_LDIV - LW_PTRIM_SYS_GPC2CLK_ALT_LDIV)))
#define LW_PTRIM_SYS_CLK_REF_LDIV(i)                        (LW_PTRIM_SYS_GPC2CLK_REF_LDIV + (i * (LW_PTRIM_SYS_LTC2CLK_REF_LDIV - LW_PTRIM_SYS_GPC2CLK_REF_LDIV)))


static CLK_COUNTER_INFO clkCounterNcsyspllInfo_GK104 = {
  LW_PTRIM_SYS_CLK_CNTR_NCSYSPLL_CFG,
  LW_PTRIM_SYS_CLK_CNTR_NCSYSPLL_CNT,
  4,
  {
      {LW_PTRIM_SYS_CLK_CNTR_NCSYSPLL_CFG_SOURCE_SYS2CLK,         "SYS2CLK"},
      {LW_PTRIM_SYS_CLK_CNTR_NCSYSPLL_CFG_SOURCE_HUB2CLK,         "HUB2CLK"},
      {LW_PTRIM_SYS_CLK_CNTR_NCSYSPLL_CFG_SOURCE_LWDCLK,          "LWDCLK"},    // Alias of MSDCLK in Kepler
      {LW_PTRIM_SYS_CLK_CNTR_NCSYSPLL_CFG_SOURCE_PEX_REFCLK_FAST, "PEX_REFCLK_FAST"},
  },
};

static CLK_COUNTER_INFO clkCounterNcosmoreInfo_GK104 = {
  LW_PTRIM_SYS_CLK_CNTR_NCOSMCORE_CFG,
  LW_PTRIM_SYS_CLK_CNTR_NCOSMCORE_CNT,
  7,
  {
      {LW_PTRIM_SYS_CLK_CNTR_NCOSMCORE_CFG_SOURCE_SYSPLL_REFCLK,  "SYSPLL_REFCLK"}, 
      {LW_PTRIM_SYS_CLK_CNTR_NCOSMCORE_CFG_SOURCE_REFMPLL_REFCLK, "REFMPLL_REFCLK"}, 
      {LW_PTRIM_SYS_CLK_CNTR_NCOSMCORE_CFG_SOURCE_UTILSCLK,       "UTILSCLK"}, 
      {LW_PTRIM_SYS_CLK_CNTR_NCOSMCORE_CFG_SOURCE_PWRCLK,         "PWRCLK"}, 
      {LW_PTRIM_SYS_CLK_CNTR_NCOSMCORE_CFG_SOURCE_HOSTCLK,        "HOSTCLK"}, 
      {LW_PTRIM_SYS_CLK_CNTR_NCOSMCORE_CFG_SOURCE_XCLK,           "XCLK"}, 
      {LW_PTRIM_SYS_CLK_CNTR_NCOSMCORE_CFG_SOURCE_TXCLK,          "TXCLK"}, 
  },
};

void
clkGetOSM1Selection_GK104
(
    LwU32          oneSrcRegVal,
    CLKSRCWHICH*   pClkSrc
)
{
    CLKSRCWHICH clkSrc;

    if (NULL == pClkSrc)
    {
        return;
    }

    switch (DRF_VAL(_PTRIM, _SYS_GPC2CLK_REF_SWITCH, _ONESRCCLK1_SELECT,
               oneSrcRegVal))
    {
        case LW_PTRIM_SYS_GPC2CLK_REF_SWITCH_ONESRCCLK1_SELECT_SPPLL1:
            clkSrc = clkSrcWhich_SPPLL1;
            break;
        case LW_PTRIM_SYS_GPC2CLK_REF_SWITCH_ONESRCCLK1_SELECT_SYSPLL:
            clkSrc = clkSrcWhich_SYSPLL;
            break;
        default:
            dprintf("lw: %s: Unsupported ONESRC BYPASS SELECT Option\n",
                    __FUNCTION__);
            clkSrc = clkSrcWhich_Default;
            DBG_BREAKPOINT();
    }

    if (pClkSrc)
       *pClkSrc = clkSrc;
}

void 
clkGetInputSelReg_GK104
(
    LwU32 clkNameMapIndex,
    LwU32 *pReg
)
{
    LwU32 oneSrcInputSelectReg = 0;

    // Find the register address for the Mux Control register.
    switch (clkNameMapIndex)
    {
        case LW_PTRIM_CLK_NAMEMAP_INDEX_SYS2CLK:
            oneSrcInputSelectReg = LW_PTRIM_SYS_SYS2CLK_OUT_SWITCH;
            break;
        case LW_PTRIM_CLK_NAMEMAP_INDEX_HUB2CLK:
            oneSrcInputSelectReg = LW_PTRIM_SYS_HUB2CLK_OUT_SWITCH;
            break;
        case LW_PTRIM_CLK_NAMEMAP_INDEX_LWDCLK:             // Alias of MSDCLK in Kepler
            oneSrcInputSelectReg = LW_PTRIM_SYS_LWDCLK_OUT_SWITCH;
            break;

        case LW_PTRIM_CLK_NAMEMAP_INDEX_DRAMCLK:
        case LW_PTRIM_CLK_NAMEMAP_INDEX_REFCLK:
            oneSrcInputSelectReg = LW_PTRIM_SYS_DRAMCLK_ALT_SWITCH;
            break;
        case LW_PVTRIM_CLK_NAMEMAP_INDEX_DISPCLK:
            oneSrcInputSelectReg = LW_PVTRIM_SYS_DISPCLK_OUT_SWITCH;
            break;
        case LW_PVTRIM_CLK_NAMEMAP_INDEX_AZA2BITCLK:
            oneSrcInputSelectReg = LW_PVTRIM_SYS_AZA2XBITCLK_OUT_SWITCH;
            break;
        case LW_PVTRIM_CLK_NAMEMAP_INDEX_SPDIFCLK:
            if (!IsGV100())
            {
                oneSrcInputSelectReg = LW_PVTRIM_SYS_SPDIFCLK_OUT_SWITCH;
                break;
            }
            else
            {
                dprintf("lw:   %s: Unsupported Clock(%d) passed in for Bypass Source Select\n", __FUNCTION__, clkNameMapIndex);
                DBG_BREAKPOINT();
                return;
            }
        case LW_PTRIM_CLK_NAMEMAP_INDEX_GPC2CLK:
        case LW_PTRIM_CLK_NAMEMAP_INDEX_LTC2CLK:
        case LW_PTRIM_CLK_NAMEMAP_INDEX_XBAR2CLK:
        case LW_PTRIM_CLK_NAMEMAP_INDEX_LEGCLK:
        case LW_PTRIM_CLK_NAMEMAP_INDEX_UTILSCLK:
        case LW_PTRIM_CLK_NAMEMAP_INDEX_PWRCLK:
            oneSrcInputSelectReg = LW_PTRIM_SYS_CLK_ALT_SWITCH(clkNameMapIndex);
            break;
        case LW_PTRIM_CLK_NAMEMAP_INDEX_HOSTCLK:
            oneSrcInputSelectReg = LW_PTRIM_SYS_HOSTCLK_ALT_SWITCH;
            break;

        default:
        {
            if ((clkNameMapIndex >= LW_PVTRIM_CLK_NAMEMAP_INDEX_VCLK(0)) && 
                (clkNameMapIndex <= LW_PVTRIM_CLK_NAMEMAP_INDEX_VCLK(pDisp[indexGpu].dispGetRemVpllCfgSize() - 1)) )
            {
                oneSrcInputSelectReg = LW_PVTRIM_SYS_VCLK_ALT_SWITCH(FERMI_VCLK_NAME_MAP_INDEX_TO_HEADNUM(clkNameMapIndex));
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
clkGetDividerRegOffset_GK104
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
        case LW_PTRIM_CLK_NAMEMAP_INDEX_SYS2CLK:
            divRegOffset = LW_PTRIM_SYS_SYS2CLK_OUT_LDIV;
            break;
        case LW_PTRIM_CLK_NAMEMAP_INDEX_HUB2CLK:
            divRegOffset = LW_PTRIM_SYS_HUB2CLK_OUT_LDIV;
            break;
        case LW_PTRIM_CLK_NAMEMAP_INDEX_LWDCLK:         // Alias of MSDCLK in Kepler
            divRegOffset = LW_PTRIM_SYS_LWDCLK_OUT_LDIV;
            break;

        case LW_PTRIM_CLK_NAMEMAP_INDEX_GPC2CLK:
        case LW_PTRIM_CLK_NAMEMAP_INDEX_LTC2CLK:
        case LW_PTRIM_CLK_NAMEMAP_INDEX_XBAR2CLK:
        {
            divRegOffset = LW_PTRIM_SYS_CLK_ALT_LDIV(clkNameMapIndex);
            break;
        }
        case LW_PTRIM_CLK_NAMEMAP_INDEX_UTILSCLK:
        case LW_PTRIM_CLK_NAMEMAP_INDEX_PWRCLK:
        {
            divRegOffset = LW_PTRIM_SYS_CLK_OUT(clkNameMapIndex);
            break;
        }
        case LW_PVTRIM_CLK_NAMEMAP_INDEX_DISPCLK:
        {
            divRegOffset = LW_PVTRIM_SYS_DISPCLK_OUT_LDIV;
            break;
        }
        case LW_PVTRIM_CLK_NAMEMAP_INDEX_AZA2BITCLK:
        {
            divRegOffset = LW_PVTRIM_SYS_AZA2XBITCLK_OUT_LDIV;
            break;
        }
        case LW_PVTRIM_CLK_NAMEMAP_INDEX_SPDIFCLK:
        {
            if (!IsGV100())
            {
                divRegOffset = LW_PVTRIM_SYS_SPDIFCLK_OUT_LDIV;
                break;
            }
            else
            {
                dprintf("lw:   %s: Unsupported clock name map index (%d)\n", __FUNCTION__, clkNameMapIndex);
                DBG_BREAKPOINT();
                return;
            }
        }
        case LW_PTRIM_CLK_NAMEMAP_INDEX_REFCLK:
        case LW_PTRIM_CLK_NAMEMAP_INDEX_DRAMCLK:
        {
            divRegOffset = LW_PTRIM_SYS_DRAMCLK_ALT_LDIV;
            break;
        }
        case LW_PTRIM_CLK_NAMEMAP_INDEX_HOSTCLK:
        {
            divRegOffset = LW_PTRIM_SYS_HOSTCLK_ALT_LDIV;
            break;
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

/**
 * @brief Reads the spread params of the PLL specified, if spread is supported
 *
 * @param[in]   pllNameMapIndex  PLL namemapindex
 * @param[out]  pSpread          Holds the spread settings if available
 */
void clkGetPllSpreadParams_GK104
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
            cfg2  = GPU_REG_RD32(LW_PTRIM_FBPA_BCAST_REFMPLL_CFG2);
            ssd0  = GPU_REG_RD32(LW_PTRIM_FBPA_BCAST_REFMPLL_SSD0);
            ssd1  = GPU_REG_RD32(LW_PTRIM_FBPA_BCAST_REFMPLL_SSD1);
            bReadSpread = TRUE;
            break;
        }
        default:
        {
            LwU32 headNum = FERMI_VCLK_NAME_MAP_INDEX_TO_HEADNUM(pllNameMapIndex);

            if (headNum < pDisp[indexGpu].dispGetRemVpllCfgSize())
            {
                cfg2  = GPU_REG_RD32(LW_PDISP_CLK_REM_VPLL_CFG2(headNum));
                ssd0  = GPU_REG_RD32(LW_PDISP_CLK_REM_VPLL_SSD0(headNum));
                ssd1  = GPU_REG_RD32(LW_PDISP_CLK_REM_VPLL_SSD1(headNum));
                bReadSpread = TRUE;
            }
        }
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

LwU32 clkGetMsdClkFreqKHz_GK104()
{
    LwU32 freq = 0;
    freq = clkGenReadClk_FERMI(LW_PTRIM_CLK_NAMEMAP_INDEX_LWDCLK);  // Alias of MSDCLK in Kepler
    return freq;
}

//-----------------------------------------------------
//
// clkIsClockDrivenfromBYPASS_GK104
// This is used to determine if a given  clock output is being driven by a PLL
// i.e. REF PATH or is being driven by ALT PATH.
// This is mostly helpful during the reading of the clocks.
//
//-----------------------------------------------------
BOOL clkIsClockDrivenfromBYPASS_GK104(LwU32 clkMapIndex)
{
    LwU32 selVCOdata;
    LwU32 selectOP = 0;

     if ((clkMapIndex == LW_PTRIM_CLK_NAMEMAP_INDEX_UTILSCLK)  ||
        (clkMapIndex == LW_PTRIM_CLK_NAMEMAP_INDEX_PWRCLK)     ||
        (clkMapIndex == LW_PVTRIM_CLK_NAMEMAP_INDEX_DISPCLK)   ||
        (clkMapIndex == LW_PVTRIM_CLK_NAMEMAP_INDEX_AZA2BITCLK)||
        (clkMapIndex == LW_PTRIM_CLK_NAMEMAP_INDEX_HOSTCLK)    ||
        (clkMapIndex == LW_PVTRIM_CLK_NAMEMAP_INDEX_SPDIFCLK))
    {
        return TRUE;
    }

    switch(clkMapIndex)
    {
        case LW_PTRIM_CLK_NAMEMAP_INDEX_GPC2CLK:
        case LW_PTRIM_CLK_NAMEMAP_INDEX_LTC2CLK:
        case LW_PTRIM_CLK_NAMEMAP_INDEX_XBAR2CLK:
        case LW_PTRIM_CLK_NAMEMAP_INDEX_SYS2CLK:
        case LW_PTRIM_CLK_NAMEMAP_INDEX_HUB2CLK:
        case LW_PTRIM_CLK_NAMEMAP_INDEX_LEGCLK:
        case LW_PTRIM_CLK_NAMEMAP_INDEX_LWDCLK:     // Alias of MSDCLK in Kepler
        {
            selVCOdata = GPU_REG_RD32(LW_PTRIM_SYS_SEL_VCO);
            selectOP = (selVCOdata >> clkMapIndex) & LW_PTRIM_SYS_CORE_SEL_VCO_CLK_VCO;
            break;
        }

        case LW_PTRIM_CLK_NAMEMAP_INDEX_DRAMCLK:
        {
            LwU32 bypPllCtrl, mclkSource;

            if (LW_OK == pClk[indexGpu].clkGetMClkSrcMode(clkMapIndex, &mclkSource))
            {
                if ((mclkSource == clkSrcWhich_REFMPLL) ||
                    (mclkSource == clkSrcWhich_MPLL))
                    return FALSE;
                else
                    return TRUE;
            }

            // Read the RAM Type.
            if (pFb[indexGpu].fbGetFBIOBroadcastDDRMode() == LW_PFB_FBPA_FBIO_BROADCAST_DDR_MODE_GDDR5)
            {
                //
                // Note: This mechnaism works only for GDDR5 
                // For GDDR5, CMOS = Bypass, CML = VCO at all times.
                // For LW_PTRIM_SYS_FBIO_CTRL_CMOS_CLK_SEL field, 1 is 
                // _CMOS and 0 is _CML.
                //
                bypPllCtrl = GPU_REG_RD32(LW_PTRIM_SYS_FBIO_CTRL);
                bypPllCtrl = DRF_NUM(_PTRIM, _SYS_FBIO_CTRL, _CMOS_CLK_SEL,
                                     bypPllCtrl);
            }
            else
            {
                LwU32 temp = GPU_REG_RD32(LW_PTRIM_SYS_BYPASSCTRL_BCAST);
                // Use the following code for SDR3
                //
                // For SDDR3, we only have CMOS and this CMOS mux control
                // can indicate whether we're in bypass or not, for GDDR5 this
                // register should not be used. 1 indicates that MCLK is driven
                // from Bypass path.
                //
                bypPllCtrl = DRF_VAL(_PTRIM, _SYS_BYPASSCTRL_BCAST, _DRAMPLL, temp);
            }
            if (bypPllCtrl)
                return TRUE;
            else
                return FALSE;
        }
        break;

        default:
        {
            // VCLK registers are handled seperately.
            LwU32 headNum = FERMI_VCLK_NAME_MAP_INDEX_TO_HEADNUM(clkMapIndex);
            if (headNum < pDisp[indexGpu].dispGetRemVpllCfgSize())
            {
                //
                // For Display clocks like VCLK where HW overrides the SEL_VCO control,
                // reading the SW register LW_PVTRIM_SYS_SEL_VCO does not give the
                // real status. We have a seperate status register which gives the synchronized
                // status of the SEL_VCO control no matter if its changed by HW or SW.
                //
               selVCOdata = GPU_REG_RD32(LW_PVTRIM_SYS_STATUS_SEL_VCO);
               selectOP = (selVCOdata >> headNum) & LW_PTRIM_SYS_CORE_SEL_VCO_CLK_VCO;
            }
            else
            {
                dprintf("lw:   %s: ERROR: Invalid Name Map Index supplied(%d)\n",__FUNCTION__, clkMapIndex);
                DBG_BREAKPOINT();
            }
            break;
        }
    }

    //
    // If 0 driven by BYPASS Path
    // IF 1 driven by VCO.
    //
    if (!selectOP)
        return TRUE;
    else
        return FALSE;
}


/**
 * @brief           This helper function is used to do a read offsets the CFG and COEFF registers of the PLLs
 *                  by sending in the sys-index and name-map clock index(as defined in fermiclk.h).
 *
 * @param[in]       PLLorClockIndex    Stores Index of either PLL or Clock
 * @param[in]      *pCfgPLLRegOffset   Stores Config PLL Register Offset
 * @param[in]      *pCoeffPLLRegOffset Stores Coefficient PLL Register Offset
 * @param[in]      *pDivRegoffset      Divider Register Offser.
 *
 * @returns         void
 */
void 
clkReadSysCoreRegOffset_GK104
(
    LwU32 PLLorClockIndex,
    LwU32 *pCfgPLLRegOffset,
    LwU32 *pCoeffPLLRegOffset,
    LwU32 *pDivRegoffset
)
{
    if(PLLorClockIndex == LW_PTRIM_CLK_NAMEMAP_INDEX_GPC2CLK)
    {
        pClk[indexGpu].clkReadSysCoreRegOffsetGpc(pCfgPLLRegOffset, pCoeffPLLRegOffset, 
                                        pDivRegoffset);
        return;
    }

    // Read the CFG register offset addresss based on Name Map Index.
    if (pCfgPLLRegOffset != NULL)
    {
        *pCfgPLLRegOffset = 0;

        // Operating on Broadcast Mode
        switch(PLLorClockIndex)
        {
            case LW_PTRIM_CLK_NAMEMAP_INDEX_LTC2CLK:
            case LW_PTRIM_CLK_NAMEMAP_INDEX_XBAR2CLK:
            case LW_PTRIM_CLK_NAMEMAP_INDEX_SYS2CLK:
            {
                *pCfgPLLRegOffset = LW_PTRIM_SYS_PLL_CFG(PLLorClockIndex);
                break;
            }
            // FBP registers are handled seperately.
            case LW_PTRIM_PLL_NAMEMAP_INDEX_DRAMPLL:
            {
                *pCfgPLLRegOffset = LW_PTRIM_FBPA_BCAST_DRAMPLL_CFG;
                break;
            }
            case LW_PTRIM_PLL_NAMEMAP_INDEX_REFMPLL:
            {
                *pCfgPLLRegOffset = LW_PTRIM_FBPA_BCAST_REFMPLL_CFG;
                break;
            }
            // SPPLL registers are handled seperately.
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
            default:
            {
                // VPLL registers are handled seperately.
                LwU32 headNum = FERMI_VCLK_NAME_MAP_INDEX_TO_HEADNUM(PLLorClockIndex);
                if (headNum < pDisp[indexGpu].dispGetRemVpllCfgSize())
                {
                    *pCfgPLLRegOffset = LW_PDISP_CLK_REM_VPLL_CFG(headNum);
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
            case LW_PTRIM_CLK_NAMEMAP_INDEX_LTC2CLK:
            case LW_PTRIM_CLK_NAMEMAP_INDEX_XBAR2CLK:
            case LW_PTRIM_CLK_NAMEMAP_INDEX_SYS2CLK:
            {
                *pCoeffPLLRegOffset = LW_PTRIM_SYS_PLL_COEFF(PLLorClockIndex);
                break;
            }

            // FBP registers are handled seperately.
            case LW_PTRIM_PLL_NAMEMAP_INDEX_DRAMPLL:
            {
                *pCoeffPLLRegOffset   = LW_PTRIM_FBPA_BCAST_DRAMPLL_COEFF;
                break;
            }
            case LW_PTRIM_PLL_NAMEMAP_INDEX_REFMPLL:
            {
                *pCoeffPLLRegOffset   = LW_PTRIM_FBPA_BCAST_REFMPLL_COEFF;
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
            default:
            {
                // VCLK registers are handled seperately.
                LwU32 headNum = FERMI_VCLK_NAME_MAP_INDEX_TO_HEADNUM(PLLorClockIndex);
                if (headNum < pDisp[indexGpu].dispGetRemVpllCfgSize())
                {
                    *pCoeffPLLRegOffset = LW_PDISP_CLK_REM_VPLL_COEFF(headNum);
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
            case LW_PTRIM_CLK_NAMEMAP_INDEX_LTC2CLK:
            case LW_PTRIM_CLK_NAMEMAP_INDEX_XBAR2CLK:
            case LW_PTRIM_CLK_NAMEMAP_INDEX_SYS2CLK:
            case LW_PTRIM_CLK_NAMEMAP_INDEX_HUB2CLK:
            case LW_PTRIM_CLK_NAMEMAP_INDEX_LEGCLK:
            case LW_PTRIM_CLK_NAMEMAP_INDEX_LWDCLK:     // Alias of MSDCLK in Kepler
            {
                *pDivRegoffset = LW_PTRIM_SYS_CLK_OUT(PLLorClockIndex);
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


CLK_COUNTER_INFO*
clkCounterFreqNcosmoreInfo_GK104()
{
    return &clkCounterNcosmoreInfo_GK104;
}

CLK_COUNTER_INFO*
clkCounterFreqNcsyspllInfo_GK104()
{
    return &clkCounterNcsyspllInfo_GK104;
}

