/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2016-2018 by LWPU Corporation.  All rights reserved.  All information
 * contained herein is proprietary and confidential to LWPU Corporation.  Any
 * use, reproduction, or disclosure without the written permission of LWPU
 * Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

//*****************************************************
//
// clkgm200.c - GM200+ clock lwwatch routines
//
//*****************************************************

#include "clk.h"
#include "disp.h"
#include "priv.h"
#include "gf10x/fermi_clk.h" // LW_PTRIM_CLK_NAMEMAP_INDEX_HOSTCLK
#include "maxwell/gm200/dev_trim.h"
#include "maxwell/gm200/dev_trim_addendum.h"
#include "disp/v02_05/dev_disp.h"
#include "g_clk_private.h"           // (rmconfig) implementation prototypes.

#define LW_PTRIM_SYS_CLK_REF_SWITCH(i)                      (LW_PTRIM_SYS_GPC2CLK_REF_SWITCH + (i * (LW_PTRIM_SYS_LTC2CLK_REF_SWITCH - LW_PTRIM_SYS_GPC2CLK_REF_SWITCH)))

/**
 * @brief Determines if a given clock output is being driven by a PLL.
 *        i.e. REF PATH or is being driven by ALT PATH.
 *
 * @param[in]   clkMapIndex         Clock namemap index enum
 *
 * @returns TRUE for ALT (Bypass) PATH, FALSE for REF (VCO) PATH
 */
BOOL
clkIsClockDrivenfromBYPASS_GM200
(
    LwU32   clkMapIndex
)
{
    LwU32 selVCOdata;
    LwU32 selectOP = 0;

    switch (clkMapIndex)
    {
        case LW_PVTRIM_CLK_NAMEMAP_INDEX_DISPCLK:
            selVCOdata = GPU_REG_RD32(LW_PVTRIM_SYS_STATUS_SEL_VCO);
            if (FLD_TEST_DRF(_PVTRIM, _SYS_STATUS_SEL_VCO, _DISPCLK_OUT, _VCO_PATH, selVCOdata))
                return FALSE;
            else
                return TRUE;
            break;

        default:
            return clkIsClockDrivenfromBYPASS_GK104(clkMapIndex);
            break;
    }
}

/**
 * @brief Finds the register address for the Mux control register.
 *
 * @param[in]   clkNameMapIndex     Clock namemap index enum
 * @param[out]  *pReg               The offset of Mux control register
 *
 * @returns void
 */
void
clkGetInputSelReg_GM200
(
    LwU32   clkNameMapIndex,
    LwU32  *pReg
)
{
    LwU32 oneSrcInputSelectReg = 0;

    // Find the register address for the Mux control register.
    switch (clkNameMapIndex)
    {
        case LW_PVTRIM_CLK_NAMEMAP_INDEX_DISPCLK:
        {
            oneSrcInputSelectReg = LW_PVTRIM_SYS_DISPCLK_ALT_SWITCH;
            break;
        }
        default:
        {
            clkGetInputSelReg_GK104(clkNameMapIndex, pReg);
            return;
        }
    }

    if (pReg)
       *pReg = oneSrcInputSelectReg;
}

/**
 * @brief Finds the register address for the divider register (OUT_LDIV/ALT_LDIV).
 *
 * @param[in]   clkNameMapIndex     Clock namemap index enum
 * @param[out]  *pRegOffset         The offset of divider register
 *
 * @returns void
 */
void
clkGetDividerRegOffset_GM200
(
    LwU32   clkNameMapIndex,
    LwU32  *pRegOffset
)
{
    LwU32 divRegOffset = 0;

    //
    // For root clocks, src LDIV register follows
    // the pattern of OUT_LDIV.
    //
    switch(clkNameMapIndex)
    {
        case LW_PVTRIM_CLK_NAMEMAP_INDEX_DISPCLK:
        {
            divRegOffset = LW_PVTRIM_SYS_DISPCLK_ALT_LDIV;
            break;
        }
        default:
        {
            clkGetDividerRegOffset_GK104(clkNameMapIndex, pRegOffset);
            return;
        }
    }

    if (pRegOffset)
       *pRegOffset = divRegOffset;
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
clkReadSysCoreRegOffset_GM200
(
    LwU32   PLLorClockIndex,
    LwU32  *pCfgPLLRegOffset,
    LwU32  *pCoeffPLLRegOffset,
    LwU32  *pDivRegoffset
)
{
    if (PLLorClockIndex == LW_PVTRIM_CLK_NAMEMAP_INDEX_DISPCLK)
    {
        // Read the CFG register offset addresss.
        if (pCfgPLLRegOffset != NULL)
        {
            *pCfgPLLRegOffset = LW_PVTRIM_SYS_DISPPLL_CFG;
        }

        // Read the COEFF register offset addresss.
        if (pCoeffPLLRegOffset != NULL)
        {
            *pCoeffPLLRegOffset = LW_PVTRIM_SYS_DISPPLL_COEFF;
        }

        // Read the LDIV register offset addresss based.
        if (pDivRegoffset != NULL)
        {
            *pDivRegoffset = LW_PVTRIM_SYS_DISPCLK_OUT;
        }
    }
    else
    {
        clkReadSysCoreRegOffset_GK104(PLLorClockIndex, pCfgPLLRegOffset, pCoeffPLLRegOffset, pDivRegoffset);
    }
}

/**
 * @brief Determines the source for Ref Path clock.
 *
 * @param[in]   pllNameMapIndex     PLL/Clock namemap index enum
 *
 * @returns PLL/Clock source enum
 */
CLKSRCWHICH
clkReadRefClockSrc_GM200
(
    LwU32   pllNameMapIndex
)
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
        case LW_PTRIM_PLL_NAMEMAP_INDEX_LTCPLL:
        case LW_PTRIM_PLL_NAMEMAP_INDEX_XBARPLL:
        case LW_PTRIM_PLL_NAMEMAP_INDEX_SYSPLL:
            oneSrcInputSelectReg = LW_PTRIM_SYS_CLK_REF_SWITCH(pllNameMapIndex);
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
                DBG_BREAKPOINT();
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
                    DBG_BREAKPOINT();
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

                case LW_PTRIM_SYS_GPC2CLK_REF_SWITCH_SLOWCLK_XTALS_IN:
                    pllSrc = clkSrcWhich_XTALS;
                    break;

                case LW_PTRIM_SYS_GPC2CLK_REF_SWITCH_SLOWCLK_SWCLK:
                    break;

                case LW_PTRIM_SYS_GPC2CLK_REF_SWITCH_SLOWCLK_XTAL4X:
                    pllSrc = clkSrcWhich_XTAL4X;
                    break;

                default:
                    dprintf("lw:   %s: Unsupported ONESRC REF PATH SELECT Option\n", __FUNCTION__);
                    pllSrc = clkSrcWhich_Default;
                    DBG_BREAKPOINT();
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
                        DBG_BREAKPOINT();
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
            DBG_BREAKPOINT();
            break;
    }

    return pllSrc;
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
clkReadRefSrcDIV_GM200
(
    LwU32       pllNameMapIndex,
    CLKSRCWHICH pllSrc
)
{
    LwU32 refSrcDiv = 1;
    LwU32 div;

    if ((pllSrc != clkSrcWhich_SPPLL0) &&  (pllSrc != clkSrcWhich_SPPLL1))
    {
        return refSrcDiv;
    }

    if (pllNameMapIndex == LW_PVTRIM_CLK_NAMEMAP_INDEX_DISPCLK)
    {
        div = GPU_REG_RD32(LW_PVTRIM_SYS_DISPCLK_REF_LDIV);
 
        // SPPLL0 is input in the OneSrc 0 of all the source muxes.
        if (pllSrc == clkSrcWhich_SPPLL0)
        {
            refSrcDiv = DRF_VAL(_PVTRIM, _SYS_DISPCLK_REF_LDIV, _ONESRC0DIV, div);
        }
        else
        {
            refSrcDiv = DRF_VAL(_PVTRIM, _SYS_DISPCLK_REF_LDIV, _ONESRC1DIV, div);
        }

        // It should not be fractional!
        RM_ASSERT((refSrcDiv) && !(refSrcDiv & 0x01));

        // Colwert it to original Div2x format.
        refSrcDiv = (refSrcDiv + 2) / 2;

        return refSrcDiv;
    }
    else
    {
        return clkReadRefSrcDIV_GF100(pllNameMapIndex, pllSrc);
    }
}

/**
 * @brief Reads the spread params of the PLL specified, if spread is supported
 *
 * @param[in]   pllNameMapIndex  PLL/Clock namemap index enum
 * @param[out]  *pSpread         The spread settings if available
 *
 * @returns void
 */
void
clkGetPllSpreadParams_GM200
(   
    LwU32   pllNameMapIndex, 
    void   *pSpread
)
{
    PLLSPREADPARAMS *pSpreadParams = (PLLSPREADPARAMS *)pSpread;
    LwU32 cfg2 = 0;
    LwU32 ssd0 = 0;
    LwU32 ssd1 = 0;

    if (pllNameMapIndex == LW_PVTRIM_CLK_NAMEMAP_INDEX_DISPCLK)
    {
        cfg2 = GPU_REG_RD32(LW_PVTRIM_SYS_DISPPLL_CFG2);
        ssd0 = GPU_REG_RD32(LW_PVTRIM_SYS_DISPPLL_SSD0);
        ssd1 = GPU_REG_RD32(LW_PVTRIM_SYS_DISPPLL_SSD1);

        pSpreadParams->bSDM   = FLD_TEST_DRF(_PVTRIM, _SYS_DISPPLL_CFG2, _SSD_EN_SDM, _YES, cfg2);
        pSpreadParams->SDM    = DRF_VAL_SIGNED(_PVTRIM, _SYS_DISPPLL_SSD0, _SDM_DIN, ssd0);
        pSpreadParams->bSSC   = FLD_TEST_DRF(_PVTRIM, _SYS_DISPPLL_CFG2, _SSD_EN_SSC, _YES, cfg2);
        pSpreadParams->SSCMin = DRF_VAL_SIGNED(_PVTRIM, _SYS_DISPPLL_SSD1, _SDM_SSC_MIN, ssd1);
        pSpreadParams->SSCMax = DRF_VAL_SIGNED(_PVTRIM, _SYS_DISPPLL_SSD1, _SDM_SSC_MAX, ssd1);
    }
    else
    {
        clkGetPllSpreadParams_GK104(pllNameMapIndex, pSpread);
    }
}
