/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2008-2018 by LWPU Corporation.  All rights reserved.  All information
 * contained herein is proprietary and confidential to LWPU Corporation.  Any
 * use, reproduction, or disclosure without the written permission of LWPU
 * Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

//*****************************************************
//
// clkgf100.c - GF100 Clock lwwatch routines 
// 
//*****************************************************
#include "os.h"
#include "inst.h"
#include "hal.h"
#include "chip.h"
#include "clk.h"
#include "disp.h"
#include "fb.h"
#include "inst.h"
#include "pmu.h"
#include "print.h"
#include "hwref/lwutil.h"
#include "gpuanalyze.h"
#include "rmpmucmdif.h"
#include "gf10x/fermi_clk.h"
#include "fermi/gf100/hwproject.h"
#include "fermi/gf100/dev_trim.h"
#include "fermi/gf100/dev_fbpa.h"
#include "fermi/gf100/dev_perf.h"
#include "fermi/gf100/dev_trim_addendum.h"
#include "fermi/gf100/dev_disp.h"
#include "fermi/gf100/dev_host.h"
#include "fermi/gf100/dev_fbpa.h"
#include "fermi/gf100/pm_signals.h"
#include "fermi/gf100/dev_ext_devices.h"
#include "fermi/gf100/dev_lw_xve.h"

#include "g_clk_private.h"           // (rmconfig) implementation prototypes.

static CLK_COUNTER_INFO clkCounterNcsyspllInfo_GF100 = {
  LW_PTRIM_SYS_CLK_CNTR_NCSYSPLL_CFG,
  LW_PTRIM_SYS_CLK_CNTR_NCSYSPLL_CNT,
  5,
  {
      {LW_PTRIM_SYS_CLK_CNTR_NCSYSPLL_CFG_SOURCE_SYS2CLK,         "SYS2CLK"}, 
      {LW_PTRIM_SYS_CLK_CNTR_NCSYSPLL_CFG_SOURCE_HUB2CLK,         "HUB2CLK"}, 
      {LW_PTRIM_SYS_CLK_CNTR_NCSYSPLL_CFG_SOURCE_LEGCLK,          "LEGCLK"}, 
      {LW_PTRIM_SYS_CLK_CNTR_NCSYSPLL_CFG_SOURCE_MSDCLK,          "MSDCLK"}, 
      {LW_PTRIM_SYS_CLK_CNTR_NCSYSPLL_CFG_SOURCE_PEX_REFCLK_FAST, "PEX_REFCLK_FAST"} 
  },
};

static CLK_COUNTER_INFO clkCounterNcltcpllInfo_GF100 = {
  LW_PTRIM_SYS_CLK_CNTR_NCLTCPLL_CFG,
  LW_PTRIM_SYS_CLK_CNTR_NCLTCPLL_CNT,
  9,
  {
      {LW_PTRIM_SYS_CLK_CNTR_NCLTCPLL_CFG_SOURCE_GPC2CLK,         "GPC2CLK"}, 
      {LW_PTRIM_SYS_CLK_CNTR_NCLTCPLL_CFG_SOURCE_LTC2CLK,         "LTC2CLK"}, 
      {LW_PTRIM_SYS_CLK_CNTR_NCLTCPLL_CFG_SOURCE_XBAR2CLK,        "XBAR2CLK"}, 
      {LW_PTRIM_SYS_CLK_CNTR_NCLTCPLL_CFG_SOURCE_GPCPLL_REFCLK,   "GPCPLL_REFCLK"}, 
      {LW_PTRIM_SYS_CLK_CNTR_NCLTCPLL_CFG_SOURCE_LTCPLL_REFCLK,   "LTCPLL_REFCLK"}, 
      {LW_PTRIM_SYS_CLK_CNTR_NCLTCPLL_CFG_SOURCE_XBARPLL_REFCLK,  "XBARPLL_REFCLK"}, 
      {LW_PTRIM_SYS_CLK_CNTR_NCLTCPLL_CFG_SOURCE_ALT_GPCCLK,      "ALT_GPCCLK"}, 
      {LW_PTRIM_SYS_CLK_CNTR_NCLTCPLL_CFG_SOURCE_ALT_LTCCLK,      "ALT_LTCCLK"}, 
      {LW_PTRIM_SYS_CLK_CNTR_NCLTCPLL_CFG_SOURCE_ALT_XBARCLK,     "ALT_XBARCLK"} 
  },
};

static CLK_COUNTER_INFO clkCounterNcosmoreInfo_GF100 = {
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
      {LW_PTRIM_SYS_CLK_CNTR_NCOSMCORE_CFG_SOURCE_HOSTCLK,        "HOSTCLK"} 
  },
};

static CLK_COUNTER_INFO clkCounterNcdispInfo_GF100 = {
  LW_PVTRIM_SYS_CLK_CNTR_NCDISP_CFG,
  LW_PVTRIM_SYS_CLK_CNTR_NCDISP_CNT,
  13,
  {
      {LW_PVTRIM_SYS_CLK_CNTR_NCDISP_CFG_SOURCE_SPPLL0_SRC,  "SPPLL0_SRC"}, 
      {LW_PVTRIM_SYS_CLK_CNTR_NCDISP_CFG_SOURCE_SPPLL0_OUT,  "SPPLL0_OUT"}, 
      {LW_PVTRIM_SYS_CLK_CNTR_NCDISP_CFG_SOURCE_SPPLL1_SRC,  "SPPLL1_SRC"}, 
      {LW_PVTRIM_SYS_CLK_CNTR_NCDISP_CFG_SOURCE_SPPLL1_OUT,  "SPPLL1_OUT"}, 
      {LW_PVTRIM_SYS_CLK_CNTR_NCDISP_CFG_SOURCE_ALT_VCLK0,   "ALT_VCLK0"}, 
      {LW_PVTRIM_SYS_CLK_CNTR_NCDISP_CFG_SOURCE_VPLL0_REF,   "VPLL0_REF"}, 
      {LW_PVTRIM_SYS_CLK_CNTR_NCDISP_CFG_SOURCE_XTAL4X_OUT,  "XTAL4X_OUT"}, 
      {LW_PVTRIM_SYS_CLK_CNTR_NCDISP_CFG_SOURCE_ALT_VCLK1,   "ALT_VCLK1"}, 
      {LW_PVTRIM_SYS_CLK_CNTR_NCDISP_CFG_SOURCE_VPLL1_REF,   "VPLL1_REF"}, 
      {LW_PVTRIM_SYS_CLK_CNTR_NCDISP_CFG_SOURCE_PIOR0_CLK,   "PIOR0_CLK"}, 
      {LW_PVTRIM_SYS_CLK_CNTR_NCDISP_CFG_SOURCE_VCLK0,       "VCLK0"}, 
      {LW_PVTRIM_SYS_CLK_CNTR_NCDISP_CFG_SOURCE_VCLK1,       "VCLK1"}, 
      {LW_PVTRIM_SYS_CLK_CNTR_NCDISP_CFG_SOURCE_DISPCLK,     "DISPCLK"} 
  },
};

static CLK_COUNTER_INFO clkCounterNcltcclkInfo_GF100 = {
  LW_PTRIM_FBP_BCAST_CLK_CNTR_NCLTCCLK_CFG,
  LW_PTRIM_FBP_BCAST_CLK_CNTR_NCLTCCLK_CNT,
  3,
  {
      {LW_PTRIM_FBP_BCAST_CLK_CNTR_NCLTCCLK_CFG_SOURCE_LTCCLK,            "LTCCLK"}, 
      {LW_PTRIM_FBP_BCAST_CLK_CNTR_NCLTCCLK_CFG_SOURCE_DRAMDIV4_REC_CLK,  "DRAMDIV4_REC_CLK"}, 
      {LW_PTRIM_FBP_BCAST_CLK_CNTR_NCLTCCLK_CFG_SOURCE_DRAMDIV2_REC_CLK,  "DRAMDIV2_REC_CLK"}, 
  }
};

static CLK_COUNTER_INFO clkCounterBcastNcgpcclkInfo_GF100 = {
  LW_PTRIM_GPC_BCAST_CLK_CNTR_NCGPCCLK_CFG,
  LW_PTRIM_GPC_BCAST_CLK_CNTR_NCGPCCLK_CNT,
  1,
  {
      {LW_PTRIM_GPC_BCAST_CLK_CNTR_NCGPCCLK_CFG_SOURCE_GPCCLK,  "GPCCLK"} 
  },
};

typedef enum
{
    clkPath_Vco,
    clkPath_Byp,
} CLKPATH;


#define LW_PVTRIM_SYS_SPPLL_COEFF(sppllNum)                 (LW_PVTRIM_SYS_SPPLL0_COEFF + (LW_PVTRIM_SYS_SPPLL1_COEFF - LW_PVTRIM_SYS_SPPLL0_COEFF) * sppllNum)
#ifndef LW_PVTRIM_SYS_SPPLL_CFG
#define LW_PVTRIM_SYS_SPPLL_CFG(sppllNum)                   (LW_PVTRIM_SYS_SPPLL0_CFG + (LW_PVTRIM_SYS_SPPLL1_CFG - LW_PVTRIM_SYS_SPPLL0_CFG) * sppllNum)
#endif
#define LW_PTRIM_SYS_CLK_REF_SWITCH(i)                      (LW_PTRIM_SYS_GPC2CLK_REF_SWITCH + (i * (LW_PTRIM_SYS_LTC2CLK_REF_SWITCH - LW_PTRIM_SYS_GPC2CLK_REF_SWITCH)))
// This indexed define does not work for Root Clocks as there is no ALT/REF LDIV on root clocks. 
#define LW_PTRIM_SYS_CLK_REF_LDIV(i)                        (LW_PTRIM_SYS_GPC2CLK_REF_LDIV + (i * (LW_PTRIM_SYS_LTC2CLK_REF_LDIV - LW_PTRIM_SYS_GPC2CLK_REF_LDIV)))


// Prototypes Ends //

static LwU32 _clkIsVclk_GF100(LwU32 clkNameMapIndex);
static LwU32 _clkReadLDIV_GF100(LwU32 PLLorClockIndex, CLKPATH clkPath);
static LwU32 _clkGetOneSrcPllFreq_FERMI(LwU32 pllNameMapIndex);

//-----------------------------------------------------
//
// clkReadHostClk
// Function to read the present state of the HostClock.
//
//-----------------------------------------------------
LwU32 clkGetHostClkFreqKHz_GF100()
{
    LwU32 clkSrc, clkSrcHostClk;
    LwU32 freq = 0;

    clkSrc = GPU_REG_RD32(LW_PHOST_SYS_CLKSRC);
    clkSrcHostClk = DRF_VAL(_PHOST, _SYS_CLKSRC, _HOSTCLK, clkSrc);

    switch (clkSrcHostClk)
    {
        case LW_PHOST_SYS_CLKSRC_HOSTCLK_BYPASSCLK:
            freq = clkGenReadClk_FERMI(LW_PTRIM_CLK_NAMEMAP_INDEX_HOSTCLK);
            break;

        case LW_PHOST_SYS_CLKSRC_HOSTCLK_TESTCLK:
            dprintf("lw:   %s: TESTCLK is not supported.\n", __FUNCTION__);
            DBG_BREAKPOINT();
            freq = -1;
            break;

        case LW_PHOST_SYS_CLKSRC_HOSTCLK_XCLK:
        case LW_PHOST_SYS_CLKSRC_HOSTCLK_XCLK_GEN2:
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

                    default:
                        dprintf("lw:   %s: Unknown Link speed %d read from 0x%08x "
                           "register\n", __FUNCTION__, linkSpeed, linkControlReg); 
                        DBG_BREAKPOINT();
                        freq = -1;
                        break;
                }
            }
            else
            {
                clkSrc = (clkSrcHostClk == LW_PHOST_SYS_CLKSRC_HOSTCLK_XCLK_GEN2) ?
                    clkSrcWhich_XCLK3XDIV2 : clkSrcWhich_XCLK;
            }
            freq = pClk[indexGpu].clkGetClkSrcFreqKHz((CLKSRCWHICH)clkSrc);
            break;
        }
        default:
            dprintf( "lw:   %s: We should never hit the default case in the switch on CLKSRC_HOSTCLK since all the possibilities have been exhausted\n", __FUNCTION__);
            DBG_BREAKPOINT();
            freq = -1;
            break;
    }

    dprintf("lw:   HostClock is driven by %s \n", getClkSrcName(clkSrc));

    return freq;
}

//-----------------------------------------------------
//
// clkGetGpc2ClkFreqKHz_GF100
//
//-----------------------------------------------------
LwU32 clkGetGpc2ClkFreqKHz_GF100( void )
{
    LwU32 freq = 0;
    freq = clkGenReadClk_FERMI(LW_PTRIM_CLK_NAMEMAP_INDEX_GPC2CLK);
    return freq;
}

//-----------------------------------------------------
//
// clkGetGpuCache2ClkFreqKHz_GF100
//
//-----------------------------------------------------
LwU32 clkGetGpuCache2ClkFreqKHz_GF100()
{
    LwU32 freq = 0;
    freq = clkGenReadClk_FERMI(LW_PTRIM_CLK_NAMEMAP_INDEX_LTC2CLK);
    return freq;
}

//-----------------------------------------------------
//
// clkGetXbar2ClkFreqKHz_GF100
//
//-----------------------------------------------------
LwU32 clkGetXbar2ClkFreqKHz_GF100()
{
    LwU32 freq = 0;
    freq = clkGenReadClk_FERMI(LW_PTRIM_CLK_NAMEMAP_INDEX_XBAR2CLK);
    return freq;
}

//-----------------------------------------------------
//
// clkGetSys2ClkFreqKHz_GF100
//
//-----------------------------------------------------
LwU32 clkGetSys2ClkFreqKHz_GF100()
{
    LwU32 freq = 0;
    freq = clkGenReadClk_FERMI(LW_PTRIM_CLK_NAMEMAP_INDEX_SYS2CLK);
    return freq;
}

//-----------------------------------------------------
//
// clkGetHub2ClkFreqKHz_GF100
//
//-----------------------------------------------------
LwU32 clkGetHub2ClkFreqKHz_GF100()
{
    LwU32 freq = 0;
    freq = clkGenReadClk_FERMI(LW_PTRIM_CLK_NAMEMAP_INDEX_HUB2CLK);
    return freq;
}

//-----------------------------------------------------
//
// clkGetLegClkFreqKHz_GF100
//
//-----------------------------------------------------
LwU32 clkGetLegClkFreqKHz_GF100()
{
    LwU32 freq = 0;
    freq = clkGenReadClk_FERMI(LW_PTRIM_CLK_NAMEMAP_INDEX_LEGCLK);
    return freq;
}

//-----------------------------------------------------
//
// clkGetUtilsClkFreqKHz_GF100
//
//-----------------------------------------------------
LwU32 clkGetUtilsClkFreqKHz_GF100()
{
    LwU32 freq = 0;
    freq = clkGenReadClk_FERMI(LW_PTRIM_CLK_NAMEMAP_INDEX_UTILSCLK);
    return freq;
}

//-----------------------------------------------------
//
// clkGetPwrClkFreqKHz_GF100
//
//-----------------------------------------------------
LwU32 clkGetPwrClkFreqKHz_GF100()
{
    LwU32 freq = 0;
    freq = clkGenReadClk_FERMI(LW_PTRIM_CLK_NAMEMAP_INDEX_PWRCLK);
    return freq;
}

//-----------------------------------------------------
//
// clkGetMClkFreqKHz_GF100
//
//-----------------------------------------------------
LwU32 clkGetMClkFreqKHz_GF100()
{
    LwS32 freq = -1;
    LwU32 srcFreq = 0;
    LwU32 srcDivVal = 0;
    LwU32 LDIV2X = 0;
    LwU32 nameMapIndex = LW_PTRIM_CLK_NAMEMAP_INDEX_DRAMCLK;
    CLKSRCWHICH clkSrc = clkSrcWhich_Default;
    CLKPATH clkPath = clkPath_Byp;

     if (pClk[indexGpu].clkIsClockDrivenfromBYPASS(nameMapIndex))
     {
         // N.B. Here source div val is twice the original.
         clkPath = clkPath_Byp;
         clkSrc = pClk[indexGpu].clkReadAltClockSrc(nameMapIndex);

         // Read source freq first so info is displayed in logical order.
         srcFreq = pClk[indexGpu].clkGetClkSrcFreqKHz(clkSrc);

         srcDivVal = pClk[indexGpu].clkReadAltSrcDIV(nameMapIndex, clkSrc);
         if (!srcDivVal)
         {
             dprintf("lw:   %s: Error - MCLK Alt SrcDiv is 0 !!\n", 
                     __FUNCTION__);
             srcDivVal = 1;
         }
         freq = (srcFreq * 2) / srcDivVal;
     }
     else
     {   
         clkPath = clkPath_Vco;
         freq = _clkGetOneSrcPllFreq_FERMI(nameMapIndex);
     }

     LDIV2X = _clkReadLDIV_GF100(nameMapIndex, clkPath);
     freq = (2 * freq) / LDIV2X;

     return freq;
}

//-----------------------------------------------------
//
// clkGetRefMClkFreqKHz_GF100
//
//-----------------------------------------------------
LwU32 clkGetRefMClkFreqKHz_GF100()
{
    dprintf("lw:   Error: Not Supported Yet\n");
    return -1;
}

//-----------------------------------------------------
//
// clkGetDispclkFreqKHz_GF100
//
//-----------------------------------------------------
LwS32 clkGetDispclkFreqKHz_GF100()
{
    LwU32 freq = 0;
    freq = clkGenReadClk_FERMI(LW_PVTRIM_CLK_NAMEMAP_INDEX_DISPCLK);
    return freq;
}

//-----------------------------------------------------
//
// clkGetVClkFreqKHz_GF100
//
//-----------------------------------------------------
LwU32 clkGetVClkFreqKHz_GF100(LwU32 vClkNum)
{
    LwU32 nameMapIndex;
    LwS32 freq;
    LwU32 srcDivVal = 0;
    CLKSRCWHICH clkSrc = clkSrcWhich_Default;

    nameMapIndex = LW_PVTRIM_CLK_NAMEMAP_INDEX_VCLK(vClkNum);
  
    if (pClk[indexGpu].clkIsClockDrivenfromBYPASS(nameMapIndex))
    {
        clkSrc = pClk[indexGpu].clkReadAltClockSrc(nameMapIndex);
        srcDivVal = pClk[indexGpu].clkReadAltSrcDIV(nameMapIndex, clkSrc);
        freq = (pClk[indexGpu].clkGetClkSrcFreqKHz(clkSrc) * 2) / srcDivVal;
        if (freq < 0)
        {
            dprintf("lw:   Error: VCLK Problem finding input source frequency\n");
            return -1;
        }

        dprintf("lw:   VCLK(%d) is driven by OSM(ALT PATH) with Input Src  as %s \n", vClkNum, getClkSrcName(clkSrc));
    }
    else
    {
        freq = _clkGetOneSrcPllFreq_FERMI(nameMapIndex);
    }

    return freq;
}

//-----------------------------------------------------
//
// clkGetSppllFreqKHz_GF100
//
//-----------------------------------------------------
LwS32 clkGetSppllFreqKHz_GF100(LwU32 sppllNum)
{
    LwS32   freq = -1, infreq = -1;
    LwU32   sppllCoeff;
    LwU32   sppllCfg;
    LwU32   M = 1, N = 1, PL = 1;

    BOOL bEn, bIDDQ;

    sppllCoeff = GPU_REG_RD32(LW_PVTRIM_SYS_SPPLL_COEFF(sppllNum));
    sppllCfg = GPU_REG_RD32(LW_PVTRIM_SYS_SPPLL_CFG(sppllNum));

    bEn = FLD_TEST_DRF(_PVTRIM, _SYS_SPPLL0_CFG, _ENABLE, _YES, sppllCfg);
    bIDDQ = FLD_TEST_DRF(_PVTRIM, _SYS_SPPLL0_CFG, _IDDQ, _POWER_ON, sppllCfg);

    //Error if pll disabled or powered down
    if (!bEn)
    {
        dprintf("lw:   Error: LW_PVTRIM_SPPLL(%d)_CFG_ENABLE_NO\n", sppllNum);
        return -1;
    }

    if (!bIDDQ)
    {
        dprintf("lw:   Error: LW_PVTRIM_SPPLL(%d)_CFG_IDDQ_POWER_OFF\n", sppllNum);
        return -1;
    }

    PL = 1; // clk to onesrc logic doesn't go through PLDIV
    M = DRF_VAL( _PVTRIM, _SYS_SPPLL0_COEFF, _MDIV, sppllCoeff);
    N = DRF_VAL( _PVTRIM, _SYS_SPPLL0_COEFF, _NDIV, sppllCoeff);

    //calc sppll output freq w/ linear div
    if ( M != 0 )
    {
        infreq = pClk[indexGpu].clkGetClkSrcFreqKHz(clkSrcWhich_XTAL);
        if (infreq < 0)
        {
            dprintf("lw:   Error: sppll Problem finding input source frequency\n");
            return -1;
        }
        // SPPLL's frequency is from VCO not PL div, so PL does not appear in this callwlation
        freq = N * infreq / M;
    }
    else
    {
        dprintf("lw:   Error: sppll clk M == 0\n");
        return -1;
    }

    dprintf("lw:   SPPLL%d[ %d MHz, M: %d, N: %d, PL: %d]\n",
        sppllNum, freq/1000, M, N, PL);
 
    return freq;
}

LwU32 _clkIsVclk_GF100(LwU32 clkNameMapIndex)
{
    return ((clkNameMapIndex >= LW_PVTRIM_CLK_NAMEMAP_INDEX_VCLK(0)) && 
            (clkNameMapIndex <= LW_PVTRIM_CLK_NAMEMAP_INDEX_VCLK(pDisp[indexGpu].dispGetRemVpllCfgSize() - 1)) );
}

/**
 * @brief Returns the OneSrc PLL freq in KHz units
 *
 * @param[in]  pllNameMapIndex  PLL for which the freq is to be returned
 *
 * @return The PLL freq, in KHz
 */
LwU32 _clkGetOneSrcPllFreq_FERMI(LwU32 pllNameMapIndex)
{
    LwU32 refFreqKHz;
    LwU32 actualFreqKHz;
    LwU32 clkSrc;
    LwU32 clkSrcFreqKHz;
    LwU32 srcDivVal;
    LwU32 coeffRegOffset = 0;
    LwU32 coeff;
    LwU32 cfgRegOffset;
    LwU32 cfg;
    LwU32 M;
    LwU32 N;
    LwU32 PL;
    PLLSPREADPARAMS spreadParams = {0}; 
    BOOL bReadPll = FALSE;

    if (pllNameMapIndex == LW_PTRIM_PLL_NAMEMAP_INDEX_DRAMPLL)
    {
        // Assume mclk is using drampll unless specified otherwise
        LwU32 mclkSource = clkSrcWhich_MPLL;

        refFreqKHz = _clkGetOneSrcPllFreq_FERMI(LW_PTRIM_PLL_NAMEMAP_INDEX_REFMPLL);

        //
        // Read MPLL regs if mclkSrcMode points to MPLL, or if clkGetMClkSrcMode
        // is stubbed out.
        //
        if (LW_OK != pClk[indexGpu].clkGetMClkSrcMode(pllNameMapIndex, &mclkSource) 
           || mclkSource == clkSrcWhich_MPLL)
        {
            if (mclkSource == clkSrcWhich_MPLL)
            {
                bReadPll = TRUE;
            }
        }
    }
    else
    {
        clkSrc    = pClk[indexGpu].clkReadRefClockSrc(pllNameMapIndex);
        srcDivVal = pClk[indexGpu].clkReadRefSrcDIV(pllNameMapIndex, clkSrc);

        if (!srcDivVal)
        {
            srcDivVal = 1;
            dprintf("lw:   Src Div Val is set to zero for the PLL (NameMapIndex =  0x%x).\n",
                    pllNameMapIndex);
        }

        clkSrcFreqKHz = pClk[indexGpu].clkGetClkSrcFreqKHz(clkSrc);
        refFreqKHz = clkSrcFreqKHz / srcDivVal;
        dprintf("lw:   PLL %d has Src Freq %d KHz, SrcDiv = %d\n",
                pllNameMapIndex, clkSrcFreqKHz, srcDivVal);
    }

    if (!refFreqKHz)
    {
        dprintf("lw:    Invalid PLL input freq %d for Clock(NameMapIndex =  0x%x). Returning freq zero.\n",
                 refFreqKHz, pllNameMapIndex);
        return 0;
    }

    if (bReadPll)
    {
        pClk[indexGpu].clkReadSysCoreRegOffset(pllNameMapIndex, &cfgRegOffset, &coeffRegOffset, NULL);

        cfg    = GPU_REG_RD32(cfgRegOffset);
        coeff  = GPU_REG_RD32(coeffRegOffset);

        if (_clkIsVclk_GF100(pllNameMapIndex))
        {
            LwU32 vClkNum   = FERMI_VCLK_NAME_MAP_INDEX_TO_HEADNUM(pllNameMapIndex);

            if (!pClk[indexGpu].clkIsVPLLEnabled(vClkNum))
            {
                dprintf("lw:   Error: VPLL(%d) is not Enabled. Returning VCLK freq as Zero\n", vClkNum);
                return 0;
            }
        }
        else if (FLD_TEST_DRF(_PTRIM, _SYS_GPCPLL_CFG, _ENABLE, _NO, cfg))
        {
            dprintf("lw:   PLL is disabled on Ref path for Clock(NameMapIndex =  0x%x). Returning freq zero.\n",
                pllNameMapIndex);
            return 0;
        }

        M  = DRF_VAL(_PTRIM, _SYS_GPCPLL_COEFF, _MDIV, coeff);
        N  = DRF_VAL(_PTRIM, _SYS_GPCPLL_COEFF, _NDIV, coeff);
        PL = DRF_VAL(_PTRIM, _SYS_GPCPLL_COEFF, _PLDIV, coeff);

        if (pllNameMapIndex == LW_PTRIM_PLL_NAMEMAP_INDEX_DRAMPLL)
        {
            if (pClk[indexGpu].clkIsMpllDivBy2Used(coeff))
                PL = 2;
            else
                PL = 1;
        }
        if (LW_OK != pClk[indexGpu].clkGetPLValue(&PL))
        {
            dprintf("lw:  Value of PL divider is invalid (0x%x) for NameMapIndex = 0x%x.\n", 
                PL, pllNameMapIndex);
        }

        if (PL == 0)
        {
            PL = 1;
            dprintf("lw:   Value of PL divider is set to zero for the PLL (NameMapIndex =  0x%x).\n", 
                pllNameMapIndex);
        }

        if (M == 0)
        {
            M = 1;
            dprintf("lw:   Value of M divider is set to zero for the PLL (NameMapIndex =  0x%x).\n", 
                pllNameMapIndex);
        }

        pClk[indexGpu].clkGetPllSpreadParams(pllNameMapIndex, &spreadParams);
    }
    else
    {
        //
        // For cases the PLL is absent (eg: REFMPLL on GF108) or 
        // unused (eg: MPLL on GF117 GDDR5 low/mid speed mode)
        //
        M = N = PL = 1;
    }

    if (spreadParams.bSDM)
    {
        actualFreqKHz = (refFreqKHz * (LwU64)(N * 8192 + 4096 + spreadParams.SDM)) / (M * PL * 8192);
    }
    else
    {
        actualFreqKHz = refFreqKHz * N / (M * PL);
    }

    dprintf("lw:   PLL PARAMS for this clk (NameMapIndex = %d)[ %d MHz, M: %d, N: %d, PL: %d",
        pllNameMapIndex, actualFreqKHz/1000, M, N, PL);

    if (spreadParams.bSDM)
    {
        dprintf(", SDM_DIN: %d]\n", spreadParams.SDM);

        if (spreadParams.bSSC)
        {
            LwS32      minFreqKHz;
            LwS32      maxFreqKHz;
            float      spreadValue;
            SPREADTYPE spread;

            minFreqKHz = (LwS32)((refFreqKHz *
                (LwU64)(N * 8192 + 4096 + spreadParams.SSCMin)) / (M * PL * 8192));

            maxFreqKHz = (LwS32)((refFreqKHz *
                (LwU64)(N * 8192 + 4096 + spreadParams.SSCMax)) / (M * PL * 8192));
        
            spreadValue = (float)(spreadParams.SDM - spreadParams.SSCMin) / 
                (N * 8192 + 4096 + spreadParams.SDM);

            if (spreadParams.SDM == spreadParams.SSCMax)
                spread = spread_Down;
            else 
                spread = spread_Center; 

            dprintf("lw:\t\t[SSC_ENABLED, SSC_MIN: %d=%.3f Mhz, SSC_MAX: %d=%.3f Mhz, SpreadType: %s, Spread: %.2f%% ]\n",
                spreadParams.SSCMin, (float) minFreqKHz/1000, spreadParams.SSCMax, (float)maxFreqKHz/1000, 
                (spread == spread_Down ? "Down-spread" : "Center-spread"), spreadValue * 100);
        }
    }
    else
    {
        dprintf(", SDM: Disabled]\n");
    }

    return actualFreqKHz;
}

/**
 * @brief Reads the spread params of the PLL specified, if spread is supported
 *
 * @param[in]   pllNameMapIndex  PLL namemapindex
 * @param[out]  pSpread          Holds the spread settings if available
 */
void clkGetPllSpreadParams_GF100
(   
    LwU32  pllNameMapIndex, 
    void*  pSpread
)
{
    PLLSPREADPARAMS* pSpreadParams = (PLLSPREADPARAMS*) pSpread;
    LwU32 cfg2        = 0;
    LwU32 ssd0        = 0;
    LwU32 ssd1        = 0;
    LwU32 headNum     = FERMI_VCLK_NAME_MAP_INDEX_TO_HEADNUM(pllNameMapIndex);
    BOOL bReadSpread = FALSE;

    if (headNum < pDisp[indexGpu].dispGetRemVpllCfgSize())
    {
        cfg2  = GPU_REG_RD32(LW_PDISP_CLK_REM_VPLL_CFG2(headNum));
        ssd0  = GPU_REG_RD32(LW_PDISP_CLK_REM_VPLL_SSD0(headNum));
        ssd1  = GPU_REG_RD32(LW_PDISP_CLK_REM_VPLL_SSD1(headNum));
        bReadSpread = TRUE;
    }

    if (bReadSpread)
    {
        pSpreadParams->bSDM    = FLD_TEST_DRF(_PDISP, _CLK_REM_VPLL_CFG2, _SSD_EN_SDM, _YES, cfg2);
        pSpreadParams->SDM     = DRF_VAL_SIGNED(_PDISP, _CLK_REM_VPLL_SSD0, _SDM_DIN, ssd0);
        pSpreadParams->bSSC    = FLD_TEST_DRF(_PDISP, _CLK_REM_VPLL_CFG2, _SSD_EN_SSC, _YES, cfg2);
        pSpreadParams->SSCMin  = DRF_VAL_SIGNED(_PDISP, _CLK_REM_VPLL_SSD1, _SDM_SSC_MIN, ssd1);
        pSpreadParams->SSCMax  = DRF_VAL_SIGNED(_PDISP, _CLK_REM_VPLL_SSD1, _SDM_SSC_MAX, ssd1);
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
// clkGenReadClk_FERMI
//
//-----------------------------------------------------
LwU32 clkGenReadClk_FERMI(LwU32 nameMapIndex)
{
    LwS32 freq = -1;
    LwU32 srcFreq = 0;
    LwU32 srcDivVal = 0;
    LwU32 LDIV2X = 0;
    CLKSRCWHICH clkSrc = clkSrcWhich_Default;
    CLKPATH clkPath = clkPath_Byp;

    if (pClk[indexGpu].clkIsClockDrivenfromBYPASS(nameMapIndex))
    {
        // N.B. Here source div val is twice the original.
        clkPath = clkPath_Byp;
        clkSrc = pClk[indexGpu].clkReadAltClockSrc(nameMapIndex);

         // Read source freq first so info is displayed in logical order.
        srcFreq = pClk[indexGpu].clkGetClkSrcFreqKHz(clkSrc);

        srcDivVal = pClk[indexGpu].clkReadAltSrcDIV(nameMapIndex, clkSrc);
        freq = (srcFreq * 2) / srcDivVal;
    }
    else
    {   
        LwU32 tempNameMapIndex = -1;
        clkPath = clkPath_Vco;

        // In case of Sys Derivative clocks read the sys pll params.
        if (pClk[indexGpu].clkIsDerivativeClk(nameMapIndex))
        {
            tempNameMapIndex = LW_PTRIM_CLK_NAMEMAP_INDEX_SYS2CLK;
        }
        else
        {
            tempNameMapIndex = nameMapIndex;
        }

        freq = _clkGetOneSrcPllFreq_FERMI(tempNameMapIndex);
    }

    LDIV2X = _clkReadLDIV_GF100(nameMapIndex, clkPath);
    freq = (2 * freq) / LDIV2X;

    return freq;
}


//-----------------------------------------------------
//
// clkIsDerivativeClk_GF100:
// This function is used to determine if the input name map index belongs to
// a SYS Derivative Clock or not.
// Note: SYS2CLK tree behaves like to be Non-Derivative clock for Ref path(OSM2)
//       and as a derivative clock in ALT Path(OSM1).Hence included here.
//
//-----------------------------------------------------
BOOL clkIsDerivativeClk_GF100(LwU32 nameMapIndex)
{
    if ((nameMapIndex == LW_PTRIM_CLK_NAMEMAP_INDEX_SYS2CLK)  ||
        (nameMapIndex == LW_PTRIM_CLK_NAMEMAP_INDEX_HUB2CLK)  ||
        (nameMapIndex == LW_PTRIM_CLK_NAMEMAP_INDEX_LEGCLK)   ||
        (nameMapIndex == LW_PTRIM_CLK_NAMEMAP_INDEX_MSDCLK))
        return TRUE;
    else
        return FALSE;
}

//-----------------------------------------------------
//
// clkReadAltClockSrc_GF100
// This routine helps in determining the source for ALT Path clock.
//
//-----------------------------------------------------
CLKSRCWHICH
clkReadAltClockSrc_GF100(LwU32 clkNameMapIndex)
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
                    clkSrc = clkSrcWhich_XTAL;
                    break;

                case LW_PTRIM_SYS_GPC2CLK_REF_SWITCH_SLOWCLK_XTALS_IN:
                    clkSrc = clkSrcWhich_XTALS;
                    break;

                case LW_PTRIM_SYS_GPC2CLK_REF_SWITCH_SLOWCLK_SWCLK:
                    break;

                case LW_PTRIM_SYS_GPC2CLK_REF_SWITCH_SLOWCLK_XTAL4X:
                    clkSrc = clkSrcWhich_XTAL4X;
                    break;
                default:
                    dprintf("lw:   %s: Unsupported ONESRC BYPASS SELECT Option\n", __FUNCTION__);
                    clkSrc = clkSrcWhich_Default;
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
                        DBG_BREAKPOINT();;
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
            DBG_BREAKPOINT();
            break;
    }
    return clkSrc;
}

LW_STATUS
clkReadAltSwitchMisc_GF100
(
    LwU32        clkNameMapIndex,
    LwU32        oneSrcInputSelectVal,
    CLKSRCWHICH* pClkSrc
)
{
    LwU32 muxSelect;

    if (NULL == pClkSrc)
    {
        return LW_ERR_GENERIC;
    }

    muxSelect = DRF_VAL(_PTRIM, _SYS_GPC2CLK_REF_SWITCH, _MISCCLK, 
                        oneSrcInputSelectVal);

    // Initialize to default value
    *pClkSrc = clkSrcWhich_Default;

    switch (muxSelect)
    {
        case LW_PTRIM_SYS_GPC2CLK_REF_SWITCH_MISCCLK_LTCPLL_OP:
            if (clkNameMapIndex != LW_PTRIM_CLK_NAMEMAP_INDEX_LTC2CLK)
            {
                *pClkSrc = clkSrcWhich_LTCPLL;
            }
            else
            {
                dprintf("lw:   This Clock cannot Drive itself as Source%d\n",
                        clkNameMapIndex);
                DBG_BREAKPOINT();
                return clkSrcWhich_Ilwalid;
            }
            break;

        case LW_PTRIM_SYS_GPC2CLK_REF_SWITCH_MISCCLK_XBARPLL_OP:
            if (clkNameMapIndex != LW_PTRIM_CLK_NAMEMAP_INDEX_XBAR2CLK)
            {
                *pClkSrc = clkSrcWhich_XBARPLL;
            }
            else
            {
                dprintf("lw:   This Clock cannot Drive itself as Source%d\n", 
                        clkNameMapIndex);
                DBG_BREAKPOINT();
                return clkSrcWhich_Ilwalid;
            }
            break;

        case LW_PTRIM_SYS_GPC2CLK_REF_SWITCH_MISCCLK_SYSPLL_OP:
            if (clkNameMapIndex != LW_PTRIM_CLK_NAMEMAP_INDEX_SYS2CLK)
            {
                *pClkSrc = clkSrcWhich_SYSPLL;
            }
            else
            {
                dprintf("lw:   This Clock cannot Drive itself as Source%d\n", 
                        clkNameMapIndex);
                DBG_BREAKPOINT();
                return clkSrcWhich_Ilwalid;
            }
            break;
        default:
            dprintf("lw:   Unsupported value (%d) for MISCCLK mux\n", muxSelect);
            break;
    }

    return LW_OK;
}

//-----------------------------------------------------
//
// clkReadRefClockSrc_GF100
// This routine helps in determining the source for Ref Path clock.
//
//-----------------------------------------------------
CLKSRCWHICH
clkReadRefClockSrc_GF100(LwU32 pllNameMapIndex)
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

LW_STATUS
clkReadRefSwitchMisc_GF100
(
    LwU32 pllNameMapIndex,
    LwU32 oneSrcInputSelectVal,
    CLKSRCWHICH *pPllSrc
)
{
    LwU32 muxSelect;
    if (NULL == pPllSrc)
    {
        return LW_ERR_GENERIC;
    }

    // Initialize to default value
    *pPllSrc = clkSrcWhich_Default;

    muxSelect = DRF_VAL(_PTRIM, _SYS_GPC2CLK_REF_SWITCH, _MISCCLK, 
                        oneSrcInputSelectVal);

    switch (DRF_VAL(_PTRIM, _SYS_GPC2CLK_REF_SWITCH, _MISCCLK, 
                    oneSrcInputSelectVal))
    {
        case LW_PTRIM_SYS_GPC2CLK_REF_SWITCH_MISCCLK_LTCPLL_OP:
            if (pllNameMapIndex != LW_PTRIM_PLL_NAMEMAP_INDEX_LTCPLL)
            {
                *pPllSrc = clkSrcWhich_LTCPLL;
            }
            else
            {
                dprintf("lw:   This PLL cannot Drive itself as Source%d\n", 
                        pllNameMapIndex);
                DBG_BREAKPOINT();
            }
            break;

        case LW_PTRIM_SYS_GPC2CLK_REF_SWITCH_MISCCLK_XBARPLL_OP:
            if (pllNameMapIndex != LW_PTRIM_PLL_NAMEMAP_INDEX_XBARPLL)
            {
                *pPllSrc = clkSrcWhich_XBARPLL;
            }
            else
            {
                dprintf("lw:   This PLL cannot Drive itself as Source%d\n", 
                        pllNameMapIndex);
                DBG_BREAKPOINT();
            }
            break;

        case LW_PTRIM_SYS_GPC2CLK_REF_SWITCH_MISCCLK_SYSPLL_OP:
            if (pllNameMapIndex != LW_PTRIM_PLL_NAMEMAP_INDEX_SYSPLL)
            {
                *pPllSrc = clkSrcWhich_SYSPLL;
            }
            else
            {
                dprintf("lw:   This PLL cannot Drive itself as Source%d\n", 
                        pllNameMapIndex);
                DBG_BREAKPOINT();
            }
            break;
        default:
            dprintf("lw:   Unsupported value (%d) for MISCCLK mux\n", muxSelect);
            DBG_BREAKPOINT();
            break;
    }

    return LW_OK;
}

//-----------------------------------------------------
//
// clkReadAltSrcDIV_GF100()
// This function is used to read the FP divider value in OSM Module.
// If the clock source is SPPLL0 or SPPLL1 and the clock is a non-derived
// clock or Root Clock only then this FPDiv has any significance.
// Return value is twice the original Div value.
//
//-----------------------------------------------------
LwU32
clkReadAltSrcDIV_GF100(LwU32 clkNameMapIndex, CLKSRCWHICH clkSrc)
{
    LwU32 altSrcDiv2X = 2;
    LwU32 div, divRegOffset;

    if ((clkSrc != clkSrcWhich_SPPLL0) && (clkSrc != clkSrcWhich_SPPLL1))
    {
        return altSrcDiv2X;
    }

    // Note:- For root clocks Src LDIV register actually follows the pattern of OUT_LDIV.
    pClk[indexGpu].clkGetDividerRegOffset(clkNameMapIndex, &divRegOffset);

    if (divRegOffset == 0)
        return altSrcDiv2X;

    div = GPU_REG_RD32(divRegOffset);

    // SPPLL0 is input in the OneSrc 0 of all the source muxes.
    if (clkSrc == clkSrcWhich_SPPLL0)
    {
        altSrcDiv2X = DRF_VAL(_PTRIM, _SYS_GPC2CLK_ALT_LDIV, _ONESRC0DIV, div);
    }
    else
    {
        altSrcDiv2X = DRF_VAL(_PTRIM, _SYS_GPC2CLK_ALT_LDIV, _ONESRC1DIV, div);
    }

    // Colwert the divider value from register format to actual div * 2
    altSrcDiv2X = altSrcDiv2X + 2;
    RM_ASSERT(altSrcDiv2X >= 2);

    if (altSrcDiv2X > 2)
        dprintf("lw:   ALT_LDIV is %d%s\n", altSrcDiv2X / 2, altSrcDiv2X & 1 ? ".5" : "");

    return altSrcDiv2X;
}

//-----------------------------------------------------
//
// clkReadRefSrcDIV_GF100 :
// This routine helps in reading the divider value for the linear (integral)
// divider that sits before the PLL Block inside the OSM(One Src Module) just
// after the SPPLL0/1 Sources.
// Note:- Though this divider is designed to handle non-integral values its not
// advisable to use non integral divider before the PLL Block as PLLs require a
// 50% duty cycle input otherwise they lose lock.
// SW should never read to the BYPASS part of the Src LDIV.
// Note:- Divider value returned is the actual divider value.
// FERMITODO: Enhance the function for Non-Sys Core Clks i.e.MCLK, VClk, DispClk,
// AzaliaClk, SPDIFCLK.
//
//-----------------------------------------------------
LwU32
clkReadRefSrcDIV_GF100(LwU32 pllNameMapIndex, CLKSRCWHICH pllSrc)
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
        case LW_PTRIM_PLL_NAMEMAP_INDEX_LTCPLL:
        case LW_PTRIM_PLL_NAMEMAP_INDEX_SYSPLL:
        case LW_PTRIM_PLL_NAMEMAP_INDEX_XBARPLL:
            divRegOffset = LW_PTRIM_SYS_CLK_REF_LDIV(pllNameMapIndex);
            break;
        case LW_PTRIM_PLL_NAMEMAP_INDEX_REFMPLL:
            divRegOffset = LW_PTRIM_SYS_REFCLK_REFMPLL_LDIV;
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

//-----------------------------------------------------
//
// _clkReadLDIV_GF100
// This routine is used throughout the code for getting the LDIV values.
// Ealier there was only PDIV which was a part fo the PLL and hence was written in program PLL
// But incase of FERMI the LDIV is out of PLL block and can be configured into more possible
// options than before.Hence, this alienation of LDIV from PLL block was done
// Note:- Divider value returned is twice the actual divider value.
//        SYS2CLK ref path is only path where we use both RefSrcDiv and LDIV.
// FERMITODO: Enhance the function for Non-Sys Core Clks i.e.MCLK, VClk, DispClk,
// AzaliaClk, SPDIFCLK.
//
//-----------------------------------------------------
static LwU32
_clkReadLDIV_GF100(LwU32 PLLorClockIndex, CLKPATH clkPath)
{
    LwU32 div, divRegOffset, divVal;

    // WE Read the LDIV only for SYS derivative clocks(including SYS clk on alt path).
    if (!pClk[indexGpu].clkIsDerivativeClk(PLLorClockIndex))
    {
        return 2;
    }

    pClk[indexGpu].clkReadSysCoreRegOffset(PLLorClockIndex, NULL, NULL, &divRegOffset);

    if (divRegOffset == 0)
        return 2;

    div = GPU_REG_RD32(divRegOffset);

    if (clkPath_Vco == clkPath)
    {
        divVal = DRF_VAL(_PTRIM, _SYS_GPC2CLK_OUT, _VCODIV, div);
    }
    else
    {
        divVal = DRF_VAL(_PTRIM, _SYS_GPC2CLK_OUT, _BYPDIV, div);
    }

    // Colwert the divider value from register format to actual div * 2
    divVal = divVal + 2;

    if (divVal > 2)
        dprintf("lw:   OUT_LDIV is %d%s\n", divVal / 2, divVal & 1 ? ".5" : "");

    return divVal;
}


/*!
 * @brief           This helper function is used to do a read of offsets the CFG
 *                  and COEFF registers of the PLLs by sending in the sys-index
 *                  and name-map clock index(as defined in fermiclk.h) to the
 *                  specific index "LW_PTRIM_CLK_NAMEMAP_INDEX_GPC2CLK" so far.
 *                  Since LW_PTRIM_SYS* have been moved to LW_PTRIM_GPC* from 
 *                  onwards GP100.
 *
 *                  In case any future changes happens to any of the Reg Names to 
 *                  any other domain then the name of this function can be renamed and
 *                  particular domain exception can be handled here.
 *
 * @param[in]      *pCfgPLLRegOffset   Stores Config PLL Register Offset
 * @param[in]      *pCoeffPLLRegOffset Stores Coefficient PLL Register Offset
 * @param[in]      *pDivRegoffset      Divider Register Offset.
 *
 * @returns         void
 */ 
void
clkReadSysCoreRegOffsetGpc_GF100
(
    LwU32 *pCfgPLLRegOffset,
    LwU32 *pCoeffPLLRegOffset,
    LwU32 *pDivRegoffset
)
{
    // Read the CFG register offset addresss based on Name Map Index.
    if (pCfgPLLRegOffset != NULL)
        *pCfgPLLRegOffset = LW_PTRIM_SYS_GPCPLL_CFG;

    // Read the COEFF register offset addresss based on Name Map Index.
    if (pCoeffPLLRegOffset != NULL)
        *pCoeffPLLRegOffset = LW_PTRIM_SYS_GPCPLL_COEFF;

    // Read the LDIV register offset addresss based on Name Map Index.
    if (pDivRegoffset != NULL)
        *pDivRegoffset = LW_PTRIM_SYS_GPC2CLK_OUT;

    return;
}


//-----------------------------------------------------
// pClk[indexGpu].clkGetClkSrcFreqKHz
//
//-----------------------------------------------------
LwU32 clkGetClkSrcFreqKHz_GF100(CLKSRCWHICH whichClkSrc)
{
    LwU32 freq = 0;

    switch (whichClkSrc)
    {
        case clkSrcWhich_Default:
            DBG_BREAKPOINT();
            // Intentional fall through

        case clkSrcWhich_XTAL:
        case clkSrcWhich_XTALS:
        case clkSrcWhich_EXTREF:
        case clkSrcWhich_QUALEXTREF:
            switch(DRF_VAL(_PEXTDEV, _BOOT_0, _STRAP_CRYSTAL, whichClkSrc))
            {
                case LW_PEXTDEV_BOOT_0_STRAP_CRYSTAL_27000K:
                    freq = 27000;
                    break;
                case LW_PEXTDEV_BOOT_0_STRAP_CRYSTAL_25000K:
                    freq = 25000;
                    break;
                default:
                    dprintf("lw:   Error: Unknown Strap Crystal\n");
                    freq = -1;
                    break;
            }
            dprintf("lw:   %s[ %d MHz ]\n", getClkSrcName(whichClkSrc), freq/1000);
            break;
            
        case clkSrcWhich_SPPLL0:
            freq = clkGetSppllFreqKHz_GF100(0);
            break;

        case clkSrcWhich_SPPLL1:
            freq = clkGetSppllFreqKHz_GF100(1);
            break;

        case clkSrcWhich_XTAL4X:
            freq = 108000;
            dprintf("lw:   %s[ %d MHz ]\n", getClkSrcName(whichClkSrc), freq/1000);
            break;

        case clkSrcWhich_HDACLK:
            // HD Audio (azalia) counter runs at 54MHz for GT21x chips
            freq = 54000;
            dprintf("lw:   %s[ %d MHz ]\n", getClkSrcName(whichClkSrc), freq/1000);
            break;

        case clkSrcWhich_PEXREFCLK:
            freq = 100000;
            dprintf("lw:   %s[ %d MHz ]\n", getClkSrcName(whichClkSrc), freq/1000);
            break;

        case clkSrcWhich_XCLK:
            freq = 277778;
            dprintf("lw:   %s[ %d MHz ]\n", getClkSrcName(whichClkSrc), freq/1000);
            break;

        case clkSrcWhich_XCLK3XDIV2:
            freq = 416667;
            dprintf("lw:   %s[ %d MHz ]\n", getClkSrcName(whichClkSrc), freq/1000);
            break;

        case clkSrcWhich_HOSTCLK:
            freq = pClk[indexGpu].clkGetHostClkFreqKHz();
            dprintf("lw:   %s[ %d MHz ]\n", getClkSrcName(whichClkSrc), freq/1000);
            break;

        case clkSrcWhich_PWRCLK:
            freq = clkGetPwrClkFreqKHz_GF100();
            break;

        case clkSrcWhich_XCLK500:
            freq = 500000;
            break;

        case clkSrcWhich_XCLKGEN3:
            freq = 571000;
            break;

        case clkSrcWhich_GPCPLL:
            freq = _clkGetOneSrcPllFreq_FERMI(LW_PTRIM_PLL_NAMEMAP_INDEX_GPCPLL);
            break;

        case clkSrcWhich_LTCPLL:
            freq = _clkGetOneSrcPllFreq_FERMI(LW_PTRIM_PLL_NAMEMAP_INDEX_LTCPLL);
            break;

        case clkSrcWhich_XBARPLL:
            freq = _clkGetOneSrcPllFreq_FERMI(LW_PTRIM_PLL_NAMEMAP_INDEX_XBARPLL);
            break;

        case clkSrcWhich_SYSPLL:
            freq = _clkGetOneSrcPllFreq_FERMI(LW_PTRIM_PLL_NAMEMAP_INDEX_SYSPLL);
            break;

        default:
            dprintf("lw:   Error: Unknown src 0x%x in %s\n", whichClkSrc, __FUNCTION__);
            freq = -1;
            break;
    }

    return freq;
}

CLK_COUNTER_INFO*
clkCounterFreqNcsyspllInfo_GF100()
{
    return &clkCounterNcsyspllInfo_GF100;
}

CLK_COUNTER_INFO*
clkCounterFreqNcltcpllInfo_GF100()
{
    return &clkCounterNcltcpllInfo_GF100;
}

CLK_COUNTER_INFO*
clkCounterFreqNcltcclkInfo_GF100()
{
    return &clkCounterNcltcclkInfo_GF100;
}

CLK_COUNTER_INFO*
clkCounterFreqBcastNcgpcclkInfo_GF100()
{
    return &clkCounterBcastNcgpcclkInfo_GF100;
}

CLK_COUNTER_INFO*
clkCounterFreqNcosmoreInfo_GF100()
{
    return &clkCounterNcosmoreInfo_GF100;
}

CLK_COUNTER_INFO*
clkCounterFreqNcdispInfo_GF100()
{
    return &clkCounterNcdispInfo_GF100;
}

//-----------------------------------------------------
//
// clkCounterFrequency_GF100
// This function is used to read the clock counters for 
// each clock domain at different tap points in each
// specific clock tree.
//
// clkDomainName is used only Volta+
//-----------------------------------------------------
void
clkCounterFrequency_GF100(LwU32 clkSel, char *clkDomainName)
{
    LwU32              clockInput      = CLK_IP_XTAL_CYCLES;
    LwU32              tgtClkCntCfgReg = 0; // cfg register for counters
    LwU32              tgtClkSrcDef    = 0; // srf def value for _SOURCE field in cfg reg
    LwU32              tgtClkCntReg    = 0; // reg to actually read clock count from
    LwU32              clockFreqKhz    = 0;
    LwU32              i               = 0;
    LwU32              j               = 0;
    LwU32              sys2ClkFreqKHz  = 0;
    CLK_COUNTER_INFO*  pClkCntrInfo    = NULL;
    CLK_COUNTER_INFO*  clkCntrInfoList[MAX_CLK_CNTR];

    clkCntrInfoList[0] = pClk[indexGpu].clkCounterFreqNcsyspllInfo();
    clkCntrInfoList[1] = pClk[indexGpu].clkCounterFreqNcosmoreInfo();
    clkCntrInfoList[2] = pClk[indexGpu].clkCounterFreqNcdispInfo();
    clkCntrInfoList[3] = pClk[indexGpu].clkCounterFreqNcltcpllInfo();
    clkCntrInfoList[4] = pClk[indexGpu].clkCounterFreqNcltcclkInfo();
    clkCntrInfoList[5] = pClk[indexGpu].clkCounterFreqBcastNcgpcclkInfo();
    clkCntrInfoList[6] = pClk[indexGpu].clkCounterFreqVclksInfo();

    // TODO: Get GPC floorswept mask
    for (i = 0; i < 6; i++)
    {
        clkCntrInfoList[7+i] = pClk[indexGpu].clkCounterFreqUcastNcgpcclkInfo(i);
    }

    for (i = 0; i < MAX_CLK_CNTR; i++)
    {
        pClkCntrInfo = clkCntrInfoList[i];
        if (!pClkCntrInfo)
            continue;

        tgtClkCntCfgReg = pClkCntrInfo->clkCntrCfgReg;
        tgtClkCntReg    = pClkCntrInfo->clkCntrCntReg;
        dprintf("lw: Measured clk frequencies for clk counter 0x%08x:\n",
                tgtClkCntCfgReg);

        for (j = 0; j < pClkCntrInfo->clkCntrSrcNum; j++)
        {
            tgtClkSrcDef = DRF_NUM(_PTRIM, _SYS_CLK_CNTR_NCLTCPLL_CFG, 
                                   _SOURCE, pClkCntrInfo->srcInfo[j].srcIdx);

            if (strcmp(pClkCntrInfo->srcInfo[j].srcName, "HOSTCLK") ||
                LW_OK != pClk[indexGpu].clkMeasureHostClkWAR(sys2ClkFreqKHz, &clockFreqKhz))
            {
                clockFreqKhz = configFreqCounter(tgtClkCntCfgReg, tgtClkSrcDef, 
                                                     tgtClkCntReg, clockInput);
            }

            if (clockFreqKhz)
            {           
                dprintf("lw:\t\t Source %16s: %d KHz\n",
                        pClkCntrInfo->srcInfo[j].srcName, clockFreqKhz);
            }

            if (LW_PTRIM_SYS_CLK_CNTR_NCSYSPLL_CFG_SOURCE_SYS2CLK == pClkCntrInfo->srcInfo[j].srcIdx)
                sys2ClkFreqKHz = clockFreqKhz;
        }
        dprintf("\n");
    }
}

LwU32 configFreqCounter
(
    LwU32 tgtClkCntCfgReg,
    LwU32 tgtClkSrcDef,
    LwU32 tgtClkCntReg,
    LwU32 clockInput
)
{
    LwU32      timer          = 0;
    LwU32      regCnt         = 0;
    LwU32      freq           = 0;
    LwU32      countStable    = 0;
    LwU32      prevCnt        = 0;
    LwU32      token          = PMU_ILWALID_MUTEX_OWNER_ID;
    LW_STATUS  status         = LW_OK;
    LwBool     bMutexAcquired = LW_FALSE;
    LwU32      xtalFreqKHz    = 0;

    if (LW_PTRIM_SYS_CLK_CNTR_NCLTCPLL_CFG == tgtClkCntCfgReg)
    {
        status = pmuAcquireMutex(RM_PMU_MUTEX_ID_CLK, 10, &token);
        if (status != LW_OK)                                               
        {                                                                  
            dprintf("lw: %s: Failed to acquire CLK mutex. Cannot complete " 
                    "operation. Please retry later.\n", __FUNCTION__);                   
            return status;                                                 
        }
        bMutexAcquired = LW_TRUE;
    }

    //
    // Reset/clear the clock counter first
    // Note: every clock has the same field bit definitions
    //
    pClk[indexGpu].clkResetCntr(0, tgtClkCntCfgReg, 0, tgtClkSrcDef);

    // configure/enable counter now
    pClk[indexGpu].clkEnableCntr(0, tgtClkCntCfgReg, 0, tgtClkSrcDef, clockInput);

    // 
    // Retrieve the count and poll until it does not get updated at a non-zero value.
    // Pri reads on Fermi are driven by crystal clock. So it's almost guaranteed that
    // everytime we read the count register, we will see a different value unless
    // we have stopped counting. But to cover the rare cases where the count register,
    // for whatever reason (HW bug, etc) doesn't get updated even when we are still
    // counting, we will wait for 5 conselwtive reads of the same value. This means
    // that unless the clock is 5 times slower than 27MHz, we should be confident
    // enough to say that clock counters have stopped and we have a valid result.
    // If any clock is 5 times slower than 27MHz, we would have serious issues popping
    // up at other places anyway.
    //
    // First, make sure clock counters have started; wait for a non-zero value.
    //
    regCnt = GPU_REG_RD32(tgtClkCntReg);
    timer  = 0;
    while ( (0 == regCnt) && (timer != MAX_TIMEOUT_US))
    {
        regCnt = GPU_REG_RD32(tgtClkCntReg);
        timer++;
        osPerfDelay(1);
    }
    if (0 == regCnt)
    {
        dprintf("lw: ERROR: Timed out while waiting for counter to be non-zero for reg addr 0x%08x\n", tgtClkCntReg);
        return 0;
    }

    //
    // At this point, we know clock counter has started counting.
    // Poll for 3 conselwtive same clock counts.
    //
    timer  = 0;
    countStable = 0;
    prevCnt = 0;
    // if we see the same value in _CNT 3 times, then it's safe to assume that the counter has stopped
    while ( (countStable != MAX_STABLE_COUNT) && (timer != MAX_TIMEOUT_US))
    {
        timer++;
        regCnt = GPU_REG_RD32(tgtClkCntReg);
        if (regCnt == prevCnt)
        {
            countStable++;
        }
        prevCnt = regCnt;
        osPerfDelay(10); // 10 us delay b/w two reads
    }

    if (countStable != MAX_STABLE_COUNT)
    {
        dprintf("lw: ERROR: Timed out while waiting for counter to reach stable value for reg addr 0x%08x\n", tgtClkCntReg);
        return 0;
    }

    //
    // Let's callwlate the frequency
    // freq = count * XtalFreqKHz / clock Input cycles
    //
    // We cant use the assumption that the XTAL is always running @27MHz now
    // that T124 uses OSC_DIV as Crystal and its freq can range from 12 to 
    // 38.4 MHz. So its better we read the Crystal everytime now.
    //
    xtalFreqKHz = pClk[indexGpu].clkReadCrystalFreqKHz();
    freq = (regCnt * xtalFreqKHz) / clockInput;

    if (bMutexAcquired)
    {
        pmuReleaseMutex(RM_PMU_MUTEX_ID_CLK, &token);
    }
    return freq;
}

// SW WAR for host clock cntr bug 545794 using sys2clk
// We use perfmon counters to measure hostclk as hostclk cntrs do not work.
LW_STATUS
clkMeasureHostClkWAR_GF100(LwU32 sys2ClkFreqKHz, LwU32* pFreq)
{
    LwU32     count = 0;
    LwU32     temp = 0, timer = 0, freq = 0;

    if (!pFreq)
    {
        dprintf("lw: %s: Invalid parameter (Null pointer)\n", __FUNCTION__);
        DBG_BREAKPOINT();
        return LW_ERR_GENERIC;
    }

    // 1. Reset the PERF_SYS_CONTROl(6) regsiter before setting the time base cycles. 
    GPU_REG_WR32(LW_PERF_PMMSYS_CONTROL(6), 0);

    //
    // 2. Set the timebase cycles to 64k. Each time we count 64k cycles of sys2clk,
    //    it will generate a pulse, and perfmon will wait for 3 of these pulses (see step 2)
    //    before stopping. We are basically counting for 256k cycles of sys2clk.
    //
    //    Sys2clk is connected to perfmon index 6.
    //    Disable CTXSW_MODE and CTXSW_TIMER so that the clock measurement will happen
    //    independent of context freeze signals in HW.
    //

    GPU_REG_WR32(LW_PERF_PMMSYS_CONTROL(6),
            DRF_DEF(_PERF, _PMMSYS_CONTROL, _MODE, _B) |
            DRF_DEF(_PERF, _PMMSYS_CONTROL, _TIMEBASE_CYCLES, _64K) |
            DRF_DEF(_PERF, _PMMSYS_CONTROL, _CTXSW_MODE, _DISABLED) |
            DRF_DEF(_PERF, _PMMSYS_CONTROL, _CTXSW_TIMER, _DISABLE));
    GPU_REG_WR32(LW_PERF_PMMSYS_EVENT_OP(6), DRF_NUM(_PERF, _PMMSYS_EVENT_OP, _FUNC, 0xAAAA));
    GPU_REG_WR32(LW_PERF_PMMSYS_EVENT_SEL(6), DRF_NUM(_PERF, _PMMSYS_EVENT_SEL, _SEL0, LW_PERF_PMMSYS_SYS0_SIGVAL_T_PULSE));


    // 3. Reset LW_PERF_PMMSYS_CONTROL(0), LW_PERF_PMMSYS_SAMPLECNT(0) & LW_PERF_PMMSYS_TRIGGERCNT(0) to 0.
    GPU_REG_WR32(LW_PERF_PMMSYS_CONTROL(0), 0);
    GPU_REG_WR32(LW_PERF_PMMSYS_SAMPLECNT(0), 0);
    GPU_REG_WR32(LW_PERF_PMMSYS_TRIGGERCNT(0), 0);

    

    GPU_REG_IDX_WR_DRF_NUM(_PERF, _PMMSYS_TRIG0_SEL, 0, _SEL0, 0x51);
    GPU_REG_IDX_WR_DRF_NUM(_PERF, _PMMSYS_SAMPLE_SEL, 0, _SEL0, 0x51);
    GPU_REG_IDX_WR_DRF_NUM(_PERF, _PMMSYS_TRIG0_OP, 0, _FUNC, 0xAAAA);
    GPU_REG_IDX_WR_DRF_NUM(_PERF, _PMMSYS_TRIG1_OP, 0, _FUNC, 0xFFFF);
    GPU_REG_IDX_WR_DRF_NUM(_PERF, _PMMSYS_EVENT_OP, 0, _FUNC, 0xFFFF);
    GPU_REG_IDX_WR_DRF_NUM(_PERF, _PMMSYS_SAMPLE_OP, 0, _FUNC, 0xAAAA);
    GPU_REG_IDX_WR_DRF_NUM(_PERF, _PMMSYS_SAMPLECNT, 0, _VAL, 3);

    //
    // 4. Hostclk is connected to perfmon index 0.
    //    Here, we choose MODE_A perfmon (oldest perfmon logic) and set the perfmon flag bits. 
    //    Through the flag bit settings, we program the Mode A perfmon to count
    //    host clk for the duration of 3 SAMPLECNT's (pulse from every 64k sys2clk cycles).
    //
    GPU_REG_WR32(LW_PERF_PMMSYS_CONTROL(0),
            DRF_DEF(_PERF, _PMMSYS_CONTROL, _MODE, _A)                  |
            DRF_DEF(_PERF, _PMMSYS_CONTROL, _EVENT_SYNC_MODE, _PULSE)   |
            DRF_DEF(_PERF, _PMMSYS_CONTROL, _CTXSW_MODE, _DISABLED)     |
            DRF_DEF(_PERF, _PMMSYS_CONTROL, _CLEAR_EVENT_ONCE, _ENABLE));

    GPU_REG_IDX_WR_DRF_DEF(_PERF, _PMMSYS_STARTEXPERIMENT, 0, _START, _DOIT);



    // 5. Poll until we're done
    timer = 0;
    temp = GPU_REG_IDX_RD_DRF(_PERF, _PMMSYS_ENGINESTATUS, 0, _STATUS);
    while ((temp == LW_PERF_PMMSYS_ENGINESTATUS_STATUS_EMPTY) && (timer != MAX_TIMEOUT_US))
    {
        // poll
        timer++;
        osPerfDelay(1);
        temp = GPU_REG_IDX_RD_DRF(_PERF, _PMMSYS_ENGINESTATUS, 0, _STATUS);
        continue;
    }

    if ((timer == MAX_TIMEOUT_US) && (temp == LW_PERF_PMMSYS_ENGINESTATUS_STATUS_EMPTY))
    {
        dprintf("lw: ERROR: Hostclk SW WAR timed out while waiting on PMMSYS_ENGINESTATUS_STATUS == EMPTY!\n");
        *pFreq = 0; 
        return LW_ERR_GENERIC;
    }
    else
    {
        //
        // Let's callwlate host clock now.
        // hostclk_freq = count / 256k * sys2clk_freq
        //
        count = GPU_REG_IDX_RD_DRF(_PERF, _PMMSYS_EVENTCNT, 0, _VAL);

        //
        // What we got as sys2clk is actually sys2clk, so we need to div by 2.
        // 256k is right-shifting by 18. So in total, we need to right-shift by 19.
        //
        freq = (LwU32)( ( ((LwU64)count) * ((LwU64)sys2ClkFreqKHz) ) >> 19);

        *pFreq = freq;
        return LW_OK;
    }
}

//-----------------------------------------------------
// clkGetClocks_GF100
//
//-----------------------------------------------------
void clkGetClocks_GF100()
{
    LwU32 i;

    // Dumpout clock frequencies
    dprintf("lw: Crystal  = %4d MHz\n\n", pClk[indexGpu].clkGetClkSrcFreqKHz(clkSrcWhich_XTAL)/1000);
    dprintf("lw: SPPLL0   = %4d MHz\n\n", pClk[indexGpu].clkGetSppllFreqKHz(0)/1000);
    dprintf("lw: SPPLL1   = %4d MHz\n\n", pClk[indexGpu].clkGetSppllFreqKHz(1)/1000);
    dprintf("lw: SYSPLL   = %4d MHz\n\n", _clkGetOneSrcPllFreq_FERMI(LW_PTRIM_PLL_NAMEMAP_INDEX_SYSPLL)/1000);
    dprintf("lw: HostClk  = %4d MHz\n\n", pClk[indexGpu].clkGetHostClkFreqKHz()/1000);
    dprintf("lw: DispClk  = %4d MHz\n\n", pClk[indexGpu].clkGetDispclkFreqKHz()/1000);
    dprintf("lw: Gpc2Clk  = %4d MHz\n\n", pClk[indexGpu].clkGetGpc2ClkFreqKHz()/1000);
    dprintf("lw: Ltc2Clk  = %4d MHz\n\n", pClk[indexGpu].clkGetGpuCache2ClkFreqKHz()/1000);
    dprintf("lw: Xbar2Clk = %4d MHz\n\n", pClk[indexGpu].clkGetXbar2ClkFreqKHz()/1000);
    dprintf("lw: Sys2Clk  = %4d MHz\n\n", pClk[indexGpu].clkGetSys2ClkFreqKHz()/1000);
    dprintf("lw: Hub2Clk  = %4d MHz\n\n", pClk[indexGpu].clkGetHub2ClkFreqKHz()/1000);
    dprintf("lw: LegClk   = %4d MHz\n\n", pClk[indexGpu].clkGetLegClkFreqKHz()/1000);
    dprintf("lw: UtilsClk = %4d MHz\n\n", pClk[indexGpu].clkGetUtilsClkFreqKHz()/1000);
    dprintf("lw: PwrClk   = %4d MHz\n\n", pClk[indexGpu].clkGetPwrClkFreqKHz()/1000);
    dprintf("lw: MsdClk   = %4d MHz\n\n", pClk[indexGpu].clkGetMsdClkFreqKHz()/1000);
    dprintf("lw: MClk     = %4d MHz\n\n", clkGetMClkFreqKHz_GF100()/1000);
    for (i = 0; i < pDisp[indexGpu].dispGetRemVpllCfgSize(); ++i)
    {
         dprintf("lw: VPLL%d   = %4d MHz\n\n", i, 
                 pClk[indexGpu].clkGetVClkFreqKHz(i)/1000);
    }

    dprintf("\n");
}

/**
 * @brief Reads back the Crystal Frequency
 *
 * @returns Frequency in KHz
 */
LwU32
clkReadCrystalFreqKHz_GF100()
{
    return pClk[indexGpu].clkGetClkSrcFreqKHz(clkSrcWhich_XTAL);
}

/**
 * @brief Reset clock counter
 *
 * @param[in]   srcReg              Clock counter source register - used only Volta+
 * @param[in]   tgtClkCntCfgReg     Clock counter config register
 * @param[in]   clkDomain           Clock domain LW2080_CTRL_CLK_DOMAIN_XX - used only Turing+
 * @param[in]   tgtClkSrcDef        Clock source for the counter
 *
 * @returns LW_OK
 */
LW_STATUS
clkResetCntr_GF100
(
    LwU32 srcReg,
    LwU32 tgtClkCntCfgReg,
    LwU32 clkDomain,
    LwU32 tgtClkSrcDef
)
{
    // Reset/clear the clock counter
    GPU_REG_WR32(tgtClkCntCfgReg,
        DRF_NUM(_PTRIM, _GPC_CLK_CNTR_NCGPCCLK_CFG, _NOOFIPCLKS, 0)       |
        DRF_DEF(_PTRIM, _GPC_CLK_CNTR_NCGPCCLK_CFG, _WRITE_EN, _ASSERTED) |
        DRF_DEF(_PTRIM, _GPC_CLK_CNTR_NCGPCCLK_CFG, _ENABLE, _DEASSERTED) |
        DRF_DEF(_PTRIM, _GPC_CLK_CNTR_NCGPCCLK_CFG, _RESET, _ASSERTED)    |
        tgtClkSrcDef);

    return LW_OK;
}

/**
 * @brief Enable clock counter
 *
 * @param[in]   srcReg              Clock counter source register - used only Volta+
 * @param[in]   tgtClkCntCfgReg     Clock counter config register
 * @param[in]   clkDomain           Clock domain LW2080_CTRL_CLK_DOMAIN_XX - used only Turing+
 * @param[in]   tgtClkSrcDef        Clock source for the counter
 * @param[in]   clockInput          Count period in xtal clock cycles
 *
 * @returns LW_OK
 */
LW_STATUS
clkEnableCntr_GF100
(
    LwU32 srcReg,
    LwU32 tgtClkCntCfgReg,
    LwU32 clkDomain,
    LwU32 tgtClkSrcDef,
    LwU32 clockInput
)
{
    // Make sure we are in the limit.
    RM_ASSERT(clockInput <  DRF_MASK(LW_PTRIM_GPC_CLK_CNTR_NCGPCCLK_CFG_NOOFIPCLKS));

    // Enable the counters.
    GPU_REG_WR32(tgtClkCntCfgReg,
        DRF_NUM(_PTRIM, _GPC_CLK_CNTR_NCGPCCLK_CFG, _NOOFIPCLKS, clockInput)   |
        DRF_DEF(_PTRIM, _GPC_CLK_CNTR_NCGPCCLK_CFG, _WRITE_EN, _ASSERTED)      |
        DRF_DEF(_PTRIM, _GPC_CLK_CNTR_NCGPCCLK_CFG, _ENABLE, _ASSERTED)        |
        DRF_DEF(_PTRIM, _GPC_CLK_CNTR_NCGPCCLK_CFG, _RESET, _DEASSERTED)       |
        tgtClkSrcDef);

    return LW_OK;
}

/**
 * @brief To detemine if a given VPLL is enabled
 *
 * @param[in]  vClkNum  VPLL number
 *
 * @return TRUE if the given VPLL is enabled
 */
BOOL clkIsVPLLEnabled_GF100(LwU32 vClkNum)
{
    LwU32 setupCtrl = GPU_REG_RD32(LW_PDISP_CLK_REM_VPLL_SETUP_CONTROL(vClkNum));

    return FLD_TEST_DRF(_PDISP, _CLK_REM_VPLL_SETUP_CONTROL, _STATUS_ENABLE, _YES, setupCtrl);
}
