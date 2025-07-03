/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2014-2018 by LWPU Corporation.  All rights reserved.  All information
 * contained herein is proprietary and confidential to LWPU Corporation.  Any
 * use, reproduction, or disclosure without the written permission of LWPU
 * Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

//*****************************************************
//
// clkgp100.c - GP100+ Clock lwwatch routines 
// 
//*****************************************************

#include "clk.h"
#include "inc/gf10x/fermi_clk.h"
#include "pascal/gp100/dev_trim.h"
#include "g_clk_private.h"           // (rmconfig) implementation prototypes.

#ifndef LW_PTRIM_GPC_CLK_CNTR_NCGPCCLK_CFG_MODE_FREE
#define LW_PTRIM_GPC_CLK_CNTR_NCGPCCLK_CFG_MODE_FREE                0x00000000
#endif

#ifndef LW_PTRIM_GPC_CLK_CNTR_NCGPCCLK_CFG_MODE_LEGACY
#define LW_PTRIM_GPC_CLK_CNTR_NCGPCCLK_CFG_MODE_LEGACY              0x00000001
#endif

static CLK_COUNTER_INFO clkCounterNcsyspllInfo_GP100 = {
    LW_PTRIM_SYS_CLK_CNTR_NCSYSPLL_CFG,
    LW_PTRIM_SYS_CLK_CNTR_NCSYSPLL_CNT,
    5,
    {
        {LW_PTRIM_SYS_CLK_CNTR_NCSYSPLL_CFG_SOURCE_SYS2CLK,         "SYS2CLK"},
        {LW_PTRIM_SYS_CLK_CNTR_NCSYSPLL_CFG_SOURCE_HUB2CLK,         "HUB2CLK"},
        {LW_PTRIM_SYS_CLK_CNTR_NCSYSPLL_CFG_SOURCE_LWDCLK,          "LWDCLK"},
        {LW_PTRIM_SYS_CLK_CNTR_NCSYSPLL_CFG_SOURCE_PEX_REFCLK_FAST, "PEX_REFCLK_FAST"},
        {LW_PTRIM_SYS_CLK_CNTR_NCSYSPLL_CFG_SOURCE_SYSPLL_OR_PEX_PAD_TCLKOUT, "SYSPLL_OR_PEX_PAD_TCLKOUT"}
    },
};

static CLK_COUNTER_INFO clkCounterNcdispInfo_GP100 = {
    LW_PVTRIM_SYS_CLK_CNTR_NCDISP_CFG,
    LW_PVTRIM_SYS_CLK_CNTR_NCDISP_CNT,
    12,
    {
        {LW_PVTRIM_SYS_CLK_CNTR_NCDISP_CFG_SOURCE_SPPLL0_SRC,  "SPPLL0_SRC"},
        {LW_PVTRIM_SYS_CLK_CNTR_NCDISP_CFG_SOURCE_SPPLL0_OUT,  "SPPLL0_OUT"},
        {LW_PVTRIM_SYS_CLK_CNTR_NCDISP_CFG_SOURCE_SPPLL1_SRC,  "SPPLL1_SRC"},
        {LW_PVTRIM_SYS_CLK_CNTR_NCDISP_CFG_SOURCE_SPPLL1_OUT,  "SPPLL1_OUT"},
        {LW_PVTRIM_SYS_CLK_CNTR_NCDISP_CFG_SOURCE_XTAL4X_OUT,  "XTAL4X_OUT"},
        {LW_PVTRIM_SYS_CLK_CNTR_NCDISP_CFG_SOURCE_XTAL16X_OUT, "XTAL16X_OUT"},
        {LW_PVTRIM_SYS_CLK_CNTR_NCDISP_CFG_SOURCE_PIOR0_CLK,   "PIOR0_CLK"},
        {LW_PVTRIM_SYS_CLK_CNTR_NCDISP_CFG_SOURCE_SOR0_CLK,    "SOR0_CLK"},
        {LW_PVTRIM_SYS_CLK_CNTR_NCDISP_CFG_SOURCE_SOR1_CLK,    "SOR1_CLK"},
        {LW_PVTRIM_SYS_CLK_CNTR_NCDISP_CFG_SOURCE_SOR2_CLK,    "SOR2_CLK"},
        {LW_PVTRIM_SYS_CLK_CNTR_NCDISP_CFG_SOURCE_SOR3_CLK,    "SOR3_CLK"},
        {LW_PVTRIM_SYS_CLK_CNTR_NCDISP_CFG_SOURCE_DISPCLK,     "DISPCLK"}
    },
};

static CLK_COUNTER_INFO clkCounterNcltcclkInfo_GP100 = {
    LW_PTRIM_FBPA_BCAST_CLK_CNTR_NCLTCCLK_CFG,
    LW_PTRIM_FBPA_BCAST_CLK_CNTR_NCLTCCLK_CNT,
    9,
    {
        {LW_PTRIM_FBPA_BCAST_CLK_CNTR_NCLTCCLK_CFG_SOURCE_DRAMDIV2_REC_CLK0, "DRAMDIV2_REC_CLK0"}, 
        {LW_PTRIM_FBPA_BCAST_CLK_CNTR_NCLTCCLK_CFG_SOURCE_DRAMDIV2_REC_CLK1, "DRAMDIV2_REC_CLK1"}, 
        {LW_PTRIM_FBPA_BCAST_CLK_CNTR_NCLTCCLK_CFG_SOURCE_DRAMDIV4_REC_CLK0, "DRAMDIV4_REC_CLK0"}, 
        {LW_PTRIM_FBPA_BCAST_CLK_CNTR_NCLTCCLK_CFG_SOURCE_DRAMDIV4_REC_CLK1, "DRAMDIV4_REC_CLK1"}, 
        {LW_PTRIM_FBPA_BCAST_CLK_CNTR_NCLTCCLK_CFG_SOURCE_LTCCLK,            "LTCCLK"},
        {LW_PTRIM_FBPA_BCAST_CLK_CNTR_NCLTCCLK_CFG_SOURCE_DRAMDIV2_REC_CLK0_PA1, "DRAMDIV2_REC_CLK0_PA1"},
        {LW_PTRIM_FBPA_BCAST_CLK_CNTR_NCLTCCLK_CFG_SOURCE_DRAMDIV2_REC_CLK1_PA1, "DRAMDIV2_REC_CLK1_PA1"},
        {LW_PTRIM_FBPA_BCAST_CLK_CNTR_NCLTCCLK_CFG_SOURCE_DRAMDIV4_REC_CLK0_PA1, "DRAMDIV4_REC_CLK0_PA1"},
        {LW_PTRIM_FBPA_BCAST_CLK_CNTR_NCLTCCLK_CFG_SOURCE_DRAMDIV4_REC_CLK1_PA1, "DRAMDIV4_REC_CLK1_PA1"}
    },
};

static CLK_COUNTER_INFO clkCounterBcastNcgpcclkInfo_GP100 = {
    LW_PTRIM_GPC_BCAST_CLK_CNTR_NCGPCCLK_CFG,
    LW_PTRIM_GPC_BCAST_CLK_CNTR_NCGPCCLK_CNT,
    8,
    {
        {LW_PTRIM_GPC_BCAST_CLK_CNTR_NCGPCCLK_CFG_SOURCE_GPCCLK,  "GPCCLK"},
        {LW_PTRIM_GPC_BCAST_CLK_CNTR_NCGPCCLK_CFG_SOURCE_GPC2CLK, "GPC2CLK"},
        {LW_PTRIM_GPC_BCAST_CLK_CNTR_NCGPCCLK_CFG_SOURCE_GPCCLK_NPE,  "GPCCLK_NPE"},
        {LW_PTRIM_GPC_BCAST_CLK_CNTR_NCGPCCLK_CFG_SOURCE_GPCCLK_NAFLL, "GPCCLK_NAFLL"},
        {LW_PTRIM_GPC_BCAST_CLK_CNTR_NCGPCCLK_CFG_SOURCE_GPCPLL_CLKOUT, "GPCPLL_CLKOUT"},
        {LW_PTRIM_GPC_BCAST_CLK_CNTR_NCGPCCLK_CFG_SOURCE_GPCNAFLL_CLKOUT_PRE_SKP, "GPCNAFLL_CLKOUT_PRE_SKP"},
        {LW_PTRIM_GPC_BCAST_CLK_CNTR_NCGPCCLK_CFG_SOURCE_GPCNAFLL_UPD_CLK, "GPCNAFLL_UPD_CLK"},
        {LW_PTRIM_GPC_BCAST_CLK_CNTR_NCGPCCLK_CFG_SOURCE_GPC2CLK_LDIV, "GPC2CLK_LDIV"}
    },
};

static CLK_COUNTER_INFO clkCounterUcastNcgpcclkInfo_GP100[] = 
{
    {
        LW_PTRIM_GPC_CLK_CNTR_NCGPCCLK_CFG(0),
        LW_PTRIM_GPC_CLK_CNTR_NCGPCCLK_CNT(0),
        8,
        {
            {LW_PTRIM_GPC_CLK_CNTR_NCGPCCLK_CFG_SOURCE_GPCCLK,  "GPCCLK[0]"},
            {LW_PTRIM_GPC_CLK_CNTR_NCGPCCLK_CFG_SOURCE_GPC2CLK, "GPC2CLK[0]"},
            {LW_PTRIM_GPC_CLK_CNTR_NCGPCCLK_CFG_SOURCE_GPCCLK_NPE,  "GPCCLK_NPE[0]"},
            {LW_PTRIM_GPC_CLK_CNTR_NCGPCCLK_CFG_SOURCE_GPCCLK_NAFLL, "GPCCLK_NAFLL[0]"},
            {LW_PTRIM_GPC_CLK_CNTR_NCGPCCLK_CFG_SOURCE_GPCPLL_CLKOUT, "GPCPLL_CLKOUT[0]"},
            {LW_PTRIM_GPC_CLK_CNTR_NCGPCCLK_CFG_SOURCE_GPCNAFLL_CLKOUT_PRE_SKP, "GPCNAFLL_CLKOUT_PRE_SKP[0]"},
            {LW_PTRIM_GPC_CLK_CNTR_NCGPCCLK_CFG_SOURCE_GPCNAFLL_UPD_CLK, "GPCNAFLL_UPD_CLK[0]"},
            {LW_PTRIM_GPC_CLK_CNTR_NCGPCCLK_CFG_SOURCE_GPC2CLK_LDIV, "GPC2CLK_LDIV[0]"}
        },
    },
    {
        LW_PTRIM_GPC_CLK_CNTR_NCGPCCLK_CFG(1),
        LW_PTRIM_GPC_CLK_CNTR_NCGPCCLK_CNT(1),
        8,
        {
            {LW_PTRIM_GPC_CLK_CNTR_NCGPCCLK_CFG_SOURCE_GPCCLK,  "GPCCLK[1]"},
            {LW_PTRIM_GPC_CLK_CNTR_NCGPCCLK_CFG_SOURCE_GPC2CLK, "GPC2CLK[1]"},
            {LW_PTRIM_GPC_CLK_CNTR_NCGPCCLK_CFG_SOURCE_GPCCLK_NPE,  "GPCCLK_NPE[1]"},
            {LW_PTRIM_GPC_CLK_CNTR_NCGPCCLK_CFG_SOURCE_GPCCLK_NAFLL, "GPCCLK_NAFLL[1]"},
            {LW_PTRIM_GPC_CLK_CNTR_NCGPCCLK_CFG_SOURCE_GPCPLL_CLKOUT, "GPCPLL_CLKOUT[1]"},
            {LW_PTRIM_GPC_CLK_CNTR_NCGPCCLK_CFG_SOURCE_GPCNAFLL_CLKOUT_PRE_SKP, "GPCNAFLL_CLKOUT_PRE_SKP[1]"},
            {LW_PTRIM_GPC_CLK_CNTR_NCGPCCLK_CFG_SOURCE_GPCNAFLL_UPD_CLK, "GPCNAFLL_UPD_CLK[1]"},
            {LW_PTRIM_GPC_CLK_CNTR_NCGPCCLK_CFG_SOURCE_GPC2CLK_LDIV, "GPC2CLK_LDIV[1]"}
        },
    },
    {
        LW_PTRIM_GPC_CLK_CNTR_NCGPCCLK_CFG(2),
        LW_PTRIM_GPC_CLK_CNTR_NCGPCCLK_CNT(2),
        8,
        {
            {LW_PTRIM_GPC_CLK_CNTR_NCGPCCLK_CFG_SOURCE_GPCCLK,  "GPCCLK[2]"},
            {LW_PTRIM_GPC_CLK_CNTR_NCGPCCLK_CFG_SOURCE_GPC2CLK, "GPC2CLK[2]"},
            {LW_PTRIM_GPC_CLK_CNTR_NCGPCCLK_CFG_SOURCE_GPCCLK_NPE,  "GPCCLK_NPE[2]"},
            {LW_PTRIM_GPC_CLK_CNTR_NCGPCCLK_CFG_SOURCE_GPCCLK_NAFLL, "GPCCLK_NAFLL[2]"},
            {LW_PTRIM_GPC_CLK_CNTR_NCGPCCLK_CFG_SOURCE_GPCPLL_CLKOUT, "GPCPLL_CLKOUT[2]"},
            {LW_PTRIM_GPC_CLK_CNTR_NCGPCCLK_CFG_SOURCE_GPCNAFLL_CLKOUT_PRE_SKP, "GPCNAFLL_CLKOUT_PRE_SKP[2]"},
            {LW_PTRIM_GPC_CLK_CNTR_NCGPCCLK_CFG_SOURCE_GPCNAFLL_UPD_CLK, "GPCNAFLL_UPD_CLK[2]"},
            {LW_PTRIM_GPC_CLK_CNTR_NCGPCCLK_CFG_SOURCE_GPC2CLK_LDIV, "GPC2CLK_LDIV[2]"}
        },
    },
    {
        LW_PTRIM_GPC_CLK_CNTR_NCGPCCLK_CFG(3),
        LW_PTRIM_GPC_CLK_CNTR_NCGPCCLK_CNT(3),
        8,
        {
            {LW_PTRIM_GPC_CLK_CNTR_NCGPCCLK_CFG_SOURCE_GPCCLK,  "GPCCLK[3]"},
            {LW_PTRIM_GPC_CLK_CNTR_NCGPCCLK_CFG_SOURCE_GPC2CLK, "GPC2CLK[3]"},
            {LW_PTRIM_GPC_CLK_CNTR_NCGPCCLK_CFG_SOURCE_GPCCLK_NPE,  "GPCCLK_NPE[3]"},
            {LW_PTRIM_GPC_CLK_CNTR_NCGPCCLK_CFG_SOURCE_GPCCLK_NAFLL, "GPCCLK_NAFLL[3]"},
            {LW_PTRIM_GPC_CLK_CNTR_NCGPCCLK_CFG_SOURCE_GPCPLL_CLKOUT, "GPCPLL_CLKOUT[3]"},
            {LW_PTRIM_GPC_CLK_CNTR_NCGPCCLK_CFG_SOURCE_GPCNAFLL_CLKOUT_PRE_SKP, "GPCNAFLL_CLKOUT_PRE_SKP[3]"},
            {LW_PTRIM_GPC_CLK_CNTR_NCGPCCLK_CFG_SOURCE_GPCNAFLL_UPD_CLK, "GPCNAFLL_UPD_CLK[3]"},
            {LW_PTRIM_GPC_CLK_CNTR_NCGPCCLK_CFG_SOURCE_GPC2CLK_LDIV, "GPC2CLK_LDIV[3]"}
        },
    },    
    {
        LW_PTRIM_GPC_CLK_CNTR_NCGPCCLK_CFG(4),
        LW_PTRIM_GPC_CLK_CNTR_NCGPCCLK_CNT(4),
        8,
        {
            {LW_PTRIM_GPC_CLK_CNTR_NCGPCCLK_CFG_SOURCE_GPCCLK,  "GPCCLK[4]"},
            {LW_PTRIM_GPC_CLK_CNTR_NCGPCCLK_CFG_SOURCE_GPC2CLK, "GPC2CLK[4]"},
            {LW_PTRIM_GPC_CLK_CNTR_NCGPCCLK_CFG_SOURCE_GPCCLK_NPE,  "GPCCLK_NPE[4]"},
            {LW_PTRIM_GPC_CLK_CNTR_NCGPCCLK_CFG_SOURCE_GPCCLK_NAFLL, "GPCCLK_NAFLL[4]"},
            {LW_PTRIM_GPC_CLK_CNTR_NCGPCCLK_CFG_SOURCE_GPCPLL_CLKOUT, "GPCPLL_CLKOUT[4]"},
            {LW_PTRIM_GPC_CLK_CNTR_NCGPCCLK_CFG_SOURCE_GPCNAFLL_CLKOUT_PRE_SKP, "GPCNAFLL_CLKOUT_PRE_SKP[4]"},
            {LW_PTRIM_GPC_CLK_CNTR_NCGPCCLK_CFG_SOURCE_GPCNAFLL_UPD_CLK, "GPCNAFLL_UPD_CLK[4]"},
            {LW_PTRIM_GPC_CLK_CNTR_NCGPCCLK_CFG_SOURCE_GPC2CLK_LDIV, "GPC2CLK_LDIV[4]"}
        },
    },
    {
        LW_PTRIM_GPC_CLK_CNTR_NCGPCCLK_CFG(5),
        LW_PTRIM_GPC_CLK_CNTR_NCGPCCLK_CNT(5),
        8,
        {
            {LW_PTRIM_GPC_CLK_CNTR_NCGPCCLK_CFG_SOURCE_GPCCLK,  "GPCCLK[5]"},
            {LW_PTRIM_GPC_CLK_CNTR_NCGPCCLK_CFG_SOURCE_GPC2CLK, "GPC2CLK[5]"},
            {LW_PTRIM_GPC_CLK_CNTR_NCGPCCLK_CFG_SOURCE_GPCCLK_NPE,  "GPCCLK_NPE[5]"},
            {LW_PTRIM_GPC_CLK_CNTR_NCGPCCLK_CFG_SOURCE_GPCCLK_NAFLL, "GPCCLK_NAFLL[5]"},
            {LW_PTRIM_GPC_CLK_CNTR_NCGPCCLK_CFG_SOURCE_GPCPLL_CLKOUT, "GPCPLL_CLKOUT[5]"},
            {LW_PTRIM_GPC_CLK_CNTR_NCGPCCLK_CFG_SOURCE_GPCNAFLL_CLKOUT_PRE_SKP, "GPCNAFLL_CLKOUT_PRE_SKP[5]"},
            {LW_PTRIM_GPC_CLK_CNTR_NCGPCCLK_CFG_SOURCE_GPCNAFLL_UPD_CLK, "GPCNAFLL_UPD_CLK[5]"},
            {LW_PTRIM_GPC_CLK_CNTR_NCGPCCLK_CFG_SOURCE_GPC2CLK_LDIV, "GPC2CLK_LDIV[5]"}
        },
    },    
};

CLK_COUNTER_INFO*
clkCounterFreqNcsyspllInfo_GP100()
{
    return &clkCounterNcsyspllInfo_GP100;
}

CLK_COUNTER_INFO*
clkCounterFreqNcdispInfo_GP100()
{
    return &clkCounterNcdispInfo_GP100;
}

CLK_COUNTER_INFO*
clkCounterFreqNcltcclkInfo_GP100()
{
    return &clkCounterNcltcclkInfo_GP100;
}

CLK_COUNTER_INFO*
clkCounterFreqBcastNcgpcclkInfo_GP100()
{
    return &clkCounterBcastNcgpcclkInfo_GP100;
}

CLK_COUNTER_INFO*
clkCounterFreqUcastNcgpcclkInfo_GP100(LwU32 Idx)
{
    RM_ASSERT(Idx < 6);
    return &clkCounterUcastNcgpcclkInfo_GP100[Idx];
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
clkResetCntr_GP100
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
        DRF_DEF(_PTRIM, _GPC_CLK_CNTR_NCGPCCLK_CFG, _MODE, _LEGACY)       |
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
clkEnableCntr_GP100
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
        DRF_DEF(_PTRIM, _GPC_CLK_CNTR_NCGPCCLK_CFG, _MODE, _LEGACY)            |
        DRF_DEF(_PTRIM, _GPC_CLK_CNTR_NCGPCCLK_CFG, _ENABLE, _ASSERTED)        |
        DRF_DEF(_PTRIM, _GPC_CLK_CNTR_NCGPCCLK_CFG, _RESET, _DEASSERTED)       |
        tgtClkSrcDef);

    return LW_OK;
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
clkReadSysCoreRegOffsetGpc_GP100
(
    LwU32 *pCfgPLLRegOffset,
    LwU32 *pCoeffPLLRegOffset,
    LwU32 *pDivRegoffset
)
{
    // Read the CFG register offset addresss based on Name Map Index.
    if (pCfgPLLRegOffset != NULL)
        *pCfgPLLRegOffset = LW_PTRIM_GPC_BCAST_GPCPLL_CFG;

    // Read the COEFF register offset addresss based on Name Map Index.
    if (pCoeffPLLRegOffset != NULL)
        *pCoeffPLLRegOffset = LW_PTRIM_GPC_BCAST_GPCPLL_COEFF;

    // Read the LDIV register offset addresss based on Name Map Index.
    if (pDivRegoffset != NULL)
        *pDivRegoffset = LW_PTRIM_GPC_BCAST_GPC2CLK_OUT;

    return;
}
