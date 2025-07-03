/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2017-2019 by LWPU Corporation.  All rights reserved.  All information
 * contained herein is proprietary and confidential to LWPU Corporation.  Any
 * use, reproduction, or disclosure without the written permission of LWPU
 * Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

//*****************************************************
//
// clktu102.c - TU102 Clock lwwatch routines 
// 
//*****************************************************
#include "os.h"
#include "hal.h"
#include "clk.h"
#include "print.h"
#include "turing/tu102/dev_trim.h"
#include "turing/tu102/dev_trim_addendum.h"
#include "ctrl/ctrl2080/ctrl2080clk.h"

#include "g_clk_private.h"           // (rmconfig) implementation prototypes.

static CLK_FR_COUNTER_REG_INFO clkFrCounterRegInfo_TU102 = {
    LW_PTRIM_SYS_FR_CLK_CNTR_PEX_PAD_TCLKS_CFG,
    0,
    LW_PTRIM_SYS_FR_CLK_CNTR_PEX_PAD_TCLKS_CNT0,
    LW_PTRIM_SYS_FR_CLK_CNTR_PEX_PAD_TCLKS_CNT1,
};

static CLK_FR_COUNTER_SRC_INFO clkFrCounterSrcInfo_TU102[] = {
    {
        "DRAM",
        LW2080_CTRL_CLK_DOMAIN_MCLK,
        8,
        {
            {0, "MCLK[0]" },
            {1, "MCLK[1]" },
            {2, "MCLK[2]" },
            {3, "MCLK[3]" },
            {4, "MCLK[4]" },
            {5, "MCLK[5]" },
            {6, "MCLK[6]" },
            {7, "MCLK[7]" },
        },
    },
    {
        "GPC",
        LW2080_CTRL_CLK_DOMAIN_GPCCLK,
        6,
        {
            {0, "GPCCLK[0]" },
            {1, "GPCCLK[1]" },
            {2, "GPCCLK[2]" },
            {3, "GPCCLK[3]" },
            {4, "GPCCLK[4]" },
            {5, "GPCCLK[5]" },
        },
    },
    {
        "DISP",
        LW2080_CTRL_CLK_DOMAIN_DISPCLK,
        1,
        {
            {0, "DISPCLK"},
        }
    },
    {
        "HOST",
        LW2080_CTRL_CLK_DOMAIN_HOSTCLK,
        1,
        {
            {0, "HOSTCLK"},
        }
    },
    {
        "HUB",
        LW2080_CTRL_CLK_DOMAIN_HUBCLK,
        1,
        {
            {0,  "HUBCLK"},
        }
    },
    {
        "LWD",
        LW2080_CTRL_CLK_DOMAIN_LWDCLK,
        1,
        {
            {0,  "LWDCLK"},
        }
    },
    {
        "PWR",
        LW2080_CTRL_CLK_DOMAIN_PWRCLK,
        1,
        {
            {0,  "PWRCLK"},
        }
    },
    {
        "SYS",
        LW2080_CTRL_CLK_DOMAIN_SYSCLK,
        1,
        {
            {0,  "SYSCLK"},
        }
    },
    {
        "SPPLL0",
        LW2080_CTRL_CLK_SOURCE_SPPLL0,  // LW2080_CTRL_CLK_SOURCE_SPPLL1 conflicts with LW2080_CTRL_CLK_DOMAIN_HUBCLK
        2,
        {
            {0,  "SPPLL0"},
            {1,  "SPPLL1"},
        }
    },
    {
        "UTILS",
        LW2080_CTRL_CLK_DOMAIN_UTILSCLK,
        1,
        {
            {0,  "UTILSCLK"},
        },
    },
    {
        "VCLK",
        LW2080_CTRL_CLK_DOMAIN_VCLK0,
        4,
        {
            {0,  "VCLK0"},
            {1,  "VCLK1"},
            {2,  "VCLK2"},
            {3,  "VCLK3"},
        },
    },
    {
        "XBAR",
        LW2080_CTRL_CLK_DOMAIN_XBARCLK,
        1,
        {
            {0, "XBARCLK"},
        },
    },
};

/*!
 * @brief Measures frequency from clock counters
 *
 * @param[in]   pClkFrCntrRegInfo   Clock counter register info
 * @param[in]   clkDomain           Clock domain LW2080_CTRL_CLK_DOMAIN_XX
 * @param[in]   srcIdx              Clock source index
 *
 * @return measured frequency for given target source definition
 */
LwU32 
clkReadFrCounter_TU102
(
    CLK_FR_COUNTER_REG_INFO *pClkFrCntrRegInfo,
    LwU32                   clkDomain,
    LwU32                   srcIdx
)
{
    LwU32   timer       = 0;
    LwU32   regCntLsb   = 0;
    LwU32   regCntMsb   = 0;
    LwU32   freq        = 0;
    LwU32   countStable = 0;
    LwU32   prevCntLsb  = 0;
    LwU32   prevCntMsb  = 0;
    LwU32   xtalFreqKHz = 0;

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
    regCntLsb = GPU_REG_RD32(pClkFrCntrRegInfo->cntRegLsb);
    regCntMsb = GPU_REG_RD32(pClkFrCntrRegInfo->cntRegMsb);
    timer  = 0;
    while ( (0 == regCntLsb) && (0 == regCntMsb) && (timer != MAX_TIMEOUT_US))
    {
        regCntLsb = GPU_REG_RD32(pClkFrCntrRegInfo->cntRegLsb);
        regCntMsb = GPU_REG_RD32(pClkFrCntrRegInfo->cntRegMsb);
        timer++;
        osPerfDelay(1);
    }
    if ((0 == regCntLsb) && (0 == regCntMsb))
    {
        dprintf("lw: ERROR: Timed out while waiting for counter to be non-zero for target domain %x source %x\n", clkDomain, srcIdx);
        return 0;
    }

    //
    // At this point, we know clock counter has started counting.
    // Poll for 3 conselwtive same clock counts.
    //
    timer  = 0;
    // if we see the same value in _CNT 3 times, then it's safe to assume that the counter has stopped
    while ((countStable != MAX_STABLE_COUNT) && (timer != MAX_TIMEOUT_US))
    {
        timer++;
        regCntLsb = GPU_REG_RD32(pClkFrCntrRegInfo->cntRegLsb);
        regCntMsb = GPU_REG_RD32(pClkFrCntrRegInfo->cntRegMsb);
        if ((regCntLsb == prevCntLsb) && (regCntMsb == prevCntMsb))
        {
            countStable++;
        }
        else
        {
            countStable = 0;
        }
        prevCntLsb = regCntLsb;
        prevCntMsb = regCntMsb;
        osPerfDelay(10); // 10 us delay b/w two reads
    }

    if (countStable != MAX_STABLE_COUNT)
    {
        dprintf("lw: ERROR: Timed out while waiting for counter to reach stable value for target domain %x source %x\n", clkDomain, srcIdx);
        return 0;
    }

    //
    // Let's callwlate the frequency
    // freq = count * XtalFreqKHz / clock Input cycles
    // We cant use the assumption that the XTAL is always running @27MHz now
    // that T124 uses OSC_DIV as Crystal and its freq can range from 12 to 
    // 38.4 MHz. So its better we read the Crystal everytime now.
    //
    xtalFreqKHz = pClk[indexGpu].clkGetPexPadSource(pClkFrCntrRegInfo->cfgReg);

    freq = (LwU32)(((LwU64)(((LwU64)regCntMsb << 32) | regCntLsb) * xtalFreqKHz) / 1024);

    if (LW2080_CTRL_CLK_DOMAIN_MCLK == clkDomain)
        pClk[indexGpu].clkAdjustMclkScalingFactor(&freq);

    return freq;
}

void clkAdjustMclkScalingFactor_TU102(LwU32 *pFreq)
{
    //
    // br8 clock was used here, hence the x8 factor
    // refer http://lwbugs/200485322 comment #90 to #95 for details.
    //
    *pFreq *= 8;
}

/*!
 * @brief Fetch the FR clock counter data
 *
 * @param[out]   ppClkFrCntrSrcInfo  Clock counter source info
 * @param[out]   ppClkFrCntrRegInfo  Clock counter register info
 * @param[out]   pNumCntrsToRead     Number of FR counters present
 */
LW_STATUS
clkGetFrCntrInfo_TU102
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
        *ppClkFrCntrSrcInfo = clkFrCounterSrcInfo_TU102;
    }

    if (ppClkFrCntrRegInfo != NULL)
    {
        *ppClkFrCntrRegInfo = &clkFrCounterRegInfo_TU102;
    }

    if (pNumCntrsToRead != NULL)
    {
        *pNumCntrsToRead  = sizeof(clkFrCounterSrcInfo_TU102)/sizeof(clkFrCounterSrcInfo_TU102[0]);
    }

    return LW_OK;
}

/*!
 * @brief Reset clock counter
 *
 * @param[in]   srcReg      Clock counter source register (unused)
 * @param[in]   cfgReg      Clock counter config register
 * @param[in]   clkDomain   Clock domain LW2080_CTRL_CLK_DOMAIN_XX
 * @param[in]   srcIdx      Clock source index
 */
LW_STATUS
clkResetCntr_TU102
(
    LwU32 srcReg,
    LwU32 cfgReg,
    LwU32 clkDomain,
    LwU32 srcIdx
)
{
    LW_STATUS   status = LW_OK;
    LwU32       data32;

    // Set the clock source
    if ((status = pClk[indexGpu].clkSetSourceCntr(clkDomain, srcIdx)) != LW_OK)
    {
        return status;
    }

    // Now for the reset
    data32 = GPU_REG_RD32(cfgReg);
    data32 = FLD_SET_DRF(_PTRIM, _SYS_FR_CLK_CNTR_PEX_PAD_TCLKS_CFG, _COUNT_UPDATE_CYCLES, _INIT, data32);
    data32 = FLD_SET_DRF(_PTRIM, _SYS_FR_CLK_CNTR_PEX_PAD_TCLKS_CFG, _COUNT_ONCE,          _INIT, data32);
    data32 = FLD_SET_DRF(_PTRIM, _SYS_FR_CLK_CNTR_PEX_PAD_TCLKS_CFG, _RESET,               _ASSERTED, data32);
    data32 = FLD_SET_DRF(_PTRIM, _SYS_FR_CLK_CNTR_PEX_PAD_TCLKS_CFG, _START_COUNT,         _DISABLED, data32);
    data32 = FLD_SET_DRF(_PTRIM, _SYS_FR_CLK_CNTR_PEX_PAD_TCLKS_CFG, _CONTINOUS_UPDATE,    _ENABLED, data32);
    GPU_REG_WR32(cfgReg, data32);

    return LW_OK;
}

/*!
 * @brief Enable clock counter
 *
 * @param[in]   srcReg          Clock counter source register (unused)
 * @param[in]   cfgReg          Clock counter config register
 * @param[in]   clkDomain       Clock domain LW2080_CTRL_CLK_DOMAIN_XX
 * @param[in]   srcIdx          Clock source index
 * @param[in]   clockInput      Count period in xtal clock cycles (unused)
 */
LW_STATUS
clkEnableCntr_TU102
(
    LwU32 srcReg,
    LwU32 cfgReg,
    LwU32 clkDomain,
    LwU32 srcIdx,
    LwU32 clockInput
)
{
    LW_STATUS status = LW_OK;
    LwU32 data32;

    // Set the clock source
    if ((status = pClk[indexGpu].clkSetSourceCntr(clkDomain, srcIdx)) != LW_OK)
    {
        return status;
    }

    //
    // Now for the actual enable
    // Setup for one time clock frequency prediction
    // Set the number cycles to 1024 (CLK_INPUT_CLK_CNT_CYCLES) value
    // Un-reset clock counter
    //
    data32 = GPU_REG_RD32(cfgReg);
    data32 = FLD_SET_DRF(_PTRIM, _SYS_FR_CLK_CNTR_PEX_PAD_TCLKS_CFG, _COUNT_ONCE,      _ENABLED, data32);
    data32 = FLD_SET_DRF(_PTRIM, _SYS_FR_CLK_CNTR_PEX_PAD_TCLKS_CFG, _COUNT_NUMCYCLES, _1024, data32);
    data32 = FLD_SET_DRF(_PTRIM, _SYS_FR_CLK_CNTR_PEX_PAD_TCLKS_CFG, _RESET, _DEASSERTED, data32);
    data32 = FLD_SET_DRF(_PTRIM, _SYS_FR_CLK_CNTR_PEX_PAD_TCLKS_CFG, _SOURCE, _PEX_PAD_TCLKOUT, data32);
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

/*!
 * @brief Read clock counters for each clock domain at different
 *        tap points in each specific clock tree
 *
 * @param[in]   clkSel          Source select for LW_PCLK_RST_CONTROLLER_PTO_CLK_CNT_CNTL (unused)
 * @param[in]   pClkDomainName  Clock Domain name
 */
void
clkCounterFrequency_TU102
(
    LwU32   clkSel,
    char   *pClkDomainName
)
{
    LwU32                       clockFreqKhz;
    LwU32                       i;
    LwU32                       j;
    CLK_FR_COUNTER_SRC_INFO*    pClkFrCntrSrcInfo;
    CLK_FR_COUNTER_SRC_INFO     clkFrCntrSrcInfo;
    CLK_FR_COUNTER_REG_INFO*    pClkFrCntrRegInfo;
    LwU32                       cntrIdx;
    LwU32                       numCntrsToRead;
    LwU32                       numCntrsPresent;
    LW_STATUS                   status;

    if (LW_OK != pClk[indexGpu].clkGetFrCntrInfo(&pClkFrCntrSrcInfo, &pClkFrCntrRegInfo, &numCntrsPresent))
    {
        dprintf("lw: Failed to retrieve free-running counter data\n");
        return;
    }

    // 'all' domains
    if (strncmp(pClkDomainName, "all", CLK_DOMAIN_NAME_STR_LEN) == 0)
    {
        numCntrsToRead = numCntrsPresent;
        cntrIdx = 0;
    }
    else // requested domain
    {
        numCntrsToRead = 1;
        for(cntrIdx = 0; cntrIdx < numCntrsPresent; cntrIdx++)
        {
            if (!strncmp(pClkFrCntrSrcInfo[cntrIdx].clkDomainName, pClkDomainName, CLK_DOMAIN_NAME_STR_LEN))
            {
                break;
            }
        }
    }

    if (cntrIdx == numCntrsPresent)
    {
        dprintf("lw: Invalid domain passed: %s\n", pClkDomainName);
    }
    else
    {
        for (i = 0; i < numCntrsToRead; i++)
        {
            clkFrCntrSrcInfo = pClkFrCntrSrcInfo[cntrIdx];
            cntrIdx++;

            for (j = 0; j < clkFrCntrSrcInfo.srcNum; j++)
            {
                dprintf("lw: Measured clk frequencies for clk counter %s:\n",
                        clkFrCntrSrcInfo.srcInfo[j].srcName);

                // Reset/clear the clock counter first
                status = pClk[indexGpu].clkResetCntr(pClkFrCntrRegInfo->srcReg,
                                            pClkFrCntrRegInfo->cfgReg, 
                                            clkFrCntrSrcInfo.clkDomain,
                                            clkFrCntrSrcInfo.srcInfo[j].srcIdx);
                if (status != LW_OK)
                    continue;

                // Delay for 1us, bug 1953217
                osPerfDelay(1);

                // configure/enable counter now
                status = pClk[indexGpu].clkEnableCntr(pClkFrCntrRegInfo->srcReg,
                                             pClkFrCntrRegInfo->cfgReg, 
                                             clkFrCntrSrcInfo.clkDomain,
                                             clkFrCntrSrcInfo.srcInfo[j].srcIdx,
                                             0);
                if (status != LW_OK)
                    continue;

                clockFreqKhz = pClk[indexGpu].clkReadFrCounter(pClkFrCntrRegInfo, 
                                                               clkFrCntrSrcInfo.clkDomain,
                                                               clkFrCntrSrcInfo.srcInfo[j].srcIdx);

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
 * @brief Set TCLKOUT source for given clock domain
 *
 * @param[in]   clkDomain   Clock domain LW2080_CTRL_CLK_DOMAIN_XX
 * @param[in]   srcIdx      Clock source index
 *
 * Generated by get_clk_domain_prog_seq_from_tclkout_pm.pl on 
 * chips_a/drivers/common/inc/hwref/<chip_family>/<chip>/tclkout_id.pm.
 * Refer https://confluence.lwpu.com/pages/viewpage.action?spaceKey=GPUC&title=TCLKOUT+Programming+Guide
 * for details.
 *
 * TODO: Auto generation of below chip specific source code by lwwatch-config
 */
LW_STATUS
clkSetSourceCntr_TU102
(
    LwU32 clkDomain,
    LwU32 srcIdx
)
{
    switch (clkDomain)
    {
        case LW2080_CTRL_CLK_DOMAIN_GPCCLK:
            if (srcIdx == 0)
            {
                GPU_REG_WR32(LW_PTRIM_GPC_TCLKOUT_CTRL_GPC(srcIdx), 0x01030101);
                GPU_REG_WR32(LW_PTRIM_SYS_TCLKOUT_CTRL_XBAR, 0x01070201);
                GPU_REG_WR32(LW_PTRIM_SYS_TCLKOUT_CTRL_SYS, 0x010a0101);
            } 
            else if (srcIdx == 1)
            {
                GPU_REG_WR32(LW_PTRIM_GPC_TCLKOUT_CTRL_GPC(srcIdx), 0x01030101);
                GPU_REG_WR32(LW_PTRIM_SYS_TCLKOUT_CTRL_XBAR, 0x01050201);
                GPU_REG_WR32(LW_PTRIM_SYS_TCLKOUT_CTRL_SYS, 0x010a0101);
            }
            else if (srcIdx == 2)
            {
                GPU_REG_WR32(LW_PTRIM_GPC_TCLKOUT_CTRL_GPC(srcIdx), 0x01030101);
                GPU_REG_WR32(LW_PTRIM_SYS_TCLKOUT_CTRL_XBAR, 0x01040201);
                GPU_REG_WR32(LW_PTRIM_SYS_TCLKOUT_CTRL_SYS, 0x010a0101);
            }
            else if (srcIdx == 3)
            {
                GPU_REG_WR32(LW_PTRIM_GPC_TCLKOUT_CTRL_GPC(srcIdx), 0x01030101);
                GPU_REG_WR32(LW_PTRIM_SYS_TCLKOUT_CTRL_XBAR, 0x01040301);
                GPU_REG_WR32(LW_PTRIM_SYS_TCLKOUT_CTRL_SYS, 0x010a0101);
            }
            else if (srcIdx == 4)
            {
                GPU_REG_WR32(LW_PTRIM_GPC_TCLKOUT_CTRL_GPC(srcIdx), 0x01030101);
                GPU_REG_WR32(LW_PTRIM_SYS_TCLKOUT_CTRL_XBAR, 0x01060301);
                GPU_REG_WR32(LW_PTRIM_SYS_TCLKOUT_CTRL_SYS, 0x010a0101);
            }
            else if (srcIdx == 5)
            {
                GPU_REG_WR32(LW_PTRIM_GPC_TCLKOUT_CTRL_GPC(srcIdx), 0x01030101);
                GPU_REG_WR32(LW_PTRIM_SYS_TCLKOUT_CTRL_XBAR, 0x01060401);
                GPU_REG_WR32(LW_PTRIM_SYS_TCLKOUT_CTRL_SYS, 0x010a0101);
            }
            else
            {
                dprintf("lw: ERROR: clk Domain %x source %x not configured \n", clkDomain, srcIdx);
                return LW_ERR_NOT_SUPPORTED;
            }
            break;
        case LW2080_CTRL_CLK_DOMAIN_XBARCLK:
            GPU_REG_WR32(LW_PTRIM_SYS_TCLKOUT_CTRL_XBAR, 0x01060101);
            GPU_REG_WR32(LW_PTRIM_SYS_TCLKOUT_CTRL_SYS, 0x010a0101);
            break;
        case LW2080_CTRL_CLK_DOMAIN_SYSCLK:
            GPU_REG_WR32(LW_PTRIM_SYS_TCLKOUT_CTRL_SYS, 0x010d0301);
            break;
        case LW2080_CTRL_CLK_DOMAIN_HUBCLK:
            GPU_REG_WR32(LW_PTRIM_SYS_TCLKOUT_CTRL_SYS, 0x010d0201);
            break;
        case LW2080_CTRL_CLK_DOMAIN_MCLK:
            if (srcIdx == 0)
            {
                GPU_REG_WR32(LW_PTRIM_FBPA_TCLKOUT_CTRL_FBP(srcIdx), 0x01020204);
                GPU_REG_WR32(LW_PTRIM_SYS_TCLKOUT_CTRL_XBAR, 0x01070101);
                GPU_REG_WR32(LW_PTRIM_SYS_TCLKOUT_CTRL_SYS, 0x010a0101);
            }
            else if (srcIdx == 1)
            {
                GPU_REG_WR32(LW_PTRIM_FBPA_TCLKOUT_CTRL_FBP(srcIdx), 0x01020204);
                GPU_REG_WR32(LW_PTRIM_SYS_TCLKOUT_CTRL_XBAR, 0x01030101);
                GPU_REG_WR32(LW_PTRIM_SYS_TCLKOUT_CTRL_SYS, 0x010a0101);
            }
            else if (srcIdx == 2)
            {
                GPU_REG_WR32(LW_PTRIM_FBPA_TCLKOUT_CTRL_FBP(srcIdx), 0x01020204);
                GPU_REG_WR32(LW_PTRIM_SYS_TCLKOUT_CTRL_XBAR, 0x01050101);
                GPU_REG_WR32(LW_PTRIM_SYS_TCLKOUT_CTRL_SYS, 0x010a0101);
            }
            else if (srcIdx == 3)
            {
                GPU_REG_WR32(LW_PTRIM_FBPA_TCLKOUT_CTRL_FBP(srcIdx), 0x01020204);
                GPU_REG_WR32(LW_PTRIM_SYS_TCLKOUT_CTRL_XBAR, 0x01040101);
                GPU_REG_WR32(LW_PTRIM_SYS_TCLKOUT_CTRL_SYS, 0x010a0101);
            }
            else if (srcIdx == 4)
            {
                GPU_REG_WR32(LW_PTRIM_FBPA_TCLKOUT_CTRL_FBP(srcIdx), 0x01020204);
                GPU_REG_WR32(LW_PTRIM_SYS_TCLKOUT_CTRL_XBAR, 0x01020301);
                GPU_REG_WR32(LW_PTRIM_SYS_TCLKOUT_CTRL_SYS, 0x010a0101);
            }
            else if (srcIdx == 5)
            {
                GPU_REG_WR32(LW_PTRIM_FBPA_TCLKOUT_CTRL_FBP(srcIdx), 0x01020204);
                GPU_REG_WR32(LW_PTRIM_SYS_TCLKOUT_CTRL_XBAR, 0x01060201);
                GPU_REG_WR32(LW_PTRIM_SYS_TCLKOUT_CTRL_SYS, 0x010a0101);
            }
            else
            {
                dprintf("lw: ERROR: clk Domain %x source %x not configured \n", clkDomain, srcIdx);
                return LW_ERR_NOT_SUPPORTED;
            }
            break;
        case LW2080_CTRL_CLK_DOMAIN_HOSTCLK:
            GPU_REG_WR32(LW_PTRIM_SYS_TCLKOUT_CTRL_SYS, 0x010d0101);
            break;
        case LW2080_CTRL_CLK_DOMAIN_DISPCLK:
            GPU_REG_WR32(LW_PTRIM_SYS_TCLKOUT_CTRL_SYS, 0x01060301);
            break;
        case LW2080_CTRL_CLK_DOMAIN_UTILSCLK:
            GPU_REG_WR32(LW_PTRIM_SYS_TCLKOUT_CTRL_SYS, 0x01090201);
            break;
        case LW2080_CTRL_CLK_DOMAIN_PWRCLK:
            GPU_REG_WR32(LW_PTRIM_SYS_TCLKOUT_CTRL_SYS, 0x01050101);
            break;
        case LW2080_CTRL_CLK_DOMAIN_LWDCLK:
            GPU_REG_WR32(LW_PTRIM_SYS_TCLKOUT_CTRL_SYS, 0x01040101);
            break;
        case LW2080_CTRL_CLK_DOMAIN_VCLK0:
            if (srcIdx == 0)
            {
                GPU_REG_WR32(LW_PTRIM_SYS_TCLKOUT_CTRL_SYS, 0x0103011b);
            }
            else if (srcIdx == 1)
            {
                GPU_REG_WR32(LW_PTRIM_SYS_TCLKOUT_CTRL_SYS, 0x0103011c);
            }
            else if (srcIdx == 2)
            {
                GPU_REG_WR32(LW_PTRIM_SYS_TCLKOUT_CTRL_SYS, 0x0103011d);
            }
            else if (srcIdx == 3)
            {
                GPU_REG_WR32(LW_PTRIM_SYS_TCLKOUT_CTRL_SYS, 0x0103011e);
            }
            else
            {
                dprintf("lw: ERROR: clk Domain %x source %x not configured \n", clkDomain, srcIdx);
                return LW_ERR_NOT_SUPPORTED;
            }
            break;
        case LW2080_CTRL_CLK_SOURCE_SPPLL0:
            if (srcIdx == 0)    // SPPLL0
            {
                GPU_REG_WR32(LW_PTRIM_SYS_TCLKOUT_CTRL_SYS, 0x010d0f17);
            }
            else if (srcIdx == 1)   // SPPLL1
            {
                GPU_REG_WR32(LW_PTRIM_SYS_TCLKOUT_CTRL_SYS, 0x010d0f18);
            }
            else
            {
                dprintf("lw: ERROR: clk Domain %x source %x not configured \n", clkDomain, srcIdx);
                return LW_ERR_NOT_SUPPORTED;
            }
            break;
        default:
        {
            dprintf("lw: ERROR: clk Domain %x source %x not configured \n", clkDomain, srcIdx);
            return LW_ERR_NOT_SUPPORTED;
            break;
        }
    }

    return LW_OK;
}

//-----------------------------------------------------
//
// clkGetHostClkFreqKHz_TU102
// Function to read the present state of the HostClock.
//
//-----------------------------------------------------
LwU32 clkGetHostClkFreqKHz_TU102()
{
    return pClk[indexGpu].clkGetNafllFreqKHz(CLK_NAFLL_ID_HOST);
}
