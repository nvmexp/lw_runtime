/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2019 by LWPU Corporation.  All rights reserved.  All information
 * contained herein is proprietary and confidential to LWPU Corporation.  Any
 * use, reproduction, or disclosure without the written permission of LWPU
 * Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

//*****************************************************
//
// clkga100.c - GA100 Clock lwwatch routines 
// 
//*****************************************************
#include "os.h"
#include "hal.h"
#include "clk.h"
#include "print.h"
#include "ampere/ga100/dev_trim.h"
#include "ampere/ga100/dev_trim_addendum.h"
#include "ctrl/ctrl2080/ctrl2080clk.h"

#include "g_clk_private.h"           // (rmconfig) implementation prototypes.

/*
 * Mapping between the NAFLL ID and the various LUT registers for that NAFLL
 */
static CLK_NAFLL_ADDRESS_MAP _nafllMap_GA100[] =
{
    {
        CLK_NAFLL_ID_SYS,
        {
            LW_PTRIM_SYS_NAFLL_SYSLUT_READ_ADDR,
            LW_PTRIM_SYS_NAFLL_SYSLUT_READ_DATA,
            LW_PTRIM_SYS_NAFLL_SYSLUT_CFG,
            LW_PTRIM_SYS_NAFLL_SYSLUT_DEBUG2,
            LW_PTRIM_SYS_NAFLL_SYSNAFLL_COEFF,
            LW_PTRIM_SYS_NAFLL_SYSLUT_ACK,
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
            LW_PTRIM_SYS_NAFLL_XBARLUT_ACK,
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
            LW_PTRIM_GPC_GPCLUT_ACK(0),
            LW_PTRIM_GPC_GPCLUT_READ_OFFSET_DATA(0),
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
            LW_PTRIM_GPC_GPCLUT_ACK(1),
            LW_PTRIM_GPC_GPCLUT_READ_OFFSET_DATA(1),
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
            LW_PTRIM_GPC_GPCLUT_ACK(2),
            LW_PTRIM_GPC_GPCLUT_READ_OFFSET_DATA(2),
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
            LW_PTRIM_GPC_GPCLUT_ACK(3),
            LW_PTRIM_GPC_GPCLUT_READ_OFFSET_DATA(3),
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
            LW_PTRIM_GPC_GPCLUT_ACK(4),
            LW_PTRIM_GPC_GPCLUT_READ_OFFSET_DATA(4),
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
            LW_PTRIM_GPC_GPCLUT_ACK(5),
            LW_PTRIM_GPC_GPCLUT_READ_OFFSET_DATA(5),
        }
    },
    {
        CLK_NAFLL_ID_GPC6,
        {
            LW_PTRIM_GPC_GPCLUT_READ_ADDR(6),
            LW_PTRIM_GPC_GPCLUT_READ_DATA(6),
            LW_PTRIM_GPC_GPCLUT_CFG(6),
            LW_PTRIM_GPC_GPCLUT_DEBUG2(6),
            LW_PTRIM_GPC_GPCNAFLL_COEFF(6),
            LW_PTRIM_GPC_GPCLUT_ACK(6),
            LW_PTRIM_GPC_GPCLUT_READ_OFFSET_DATA(6),
        }
    },
    {
        CLK_NAFLL_ID_GPC7,
        {
            LW_PTRIM_GPC_GPCLUT_READ_ADDR(7),
            LW_PTRIM_GPC_GPCLUT_READ_DATA(7),
            LW_PTRIM_GPC_GPCLUT_CFG(7),
            LW_PTRIM_GPC_GPCLUT_DEBUG2(7),
            LW_PTRIM_GPC_GPCNAFLL_COEFF(7),
            LW_PTRIM_GPC_GPCLUT_ACK(7),
            LW_PTRIM_GPC_GPCLUT_READ_OFFSET_DATA(7),
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
            LW_PTRIM_GPC_BCAST_GPCLUT_ACK,
            LW_PTRIM_GPC_BCAST_GPCLUT_READ_OFFSET_DATA,
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
            LW_PTRIM_SYS_NAFLL_LWDLUT_ACK,
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
            LW_PTRIM_SYS_NAFLL_HOSTLUT_ACK,
            CLK_REGISTER_ADDR_UNDEFINED,
        }
    },
};

/*!
 * TODO: Generalize to CLK_NAFLL_REG_GET
 * http://lwbugs/2571189
 */
#define CLK_NAFLL_REG_GET_GA100(_nafllIdx,_type)                              \
    (_nafllMap_GA100[_nafllIdx].regAddr[CLK_NAFLL_REG_TYPE_##_type])

/*!
 * Timeout for LUT_ACK_RDACK_LUT_DEBUG to reflect expected value
 */
#define CLK_LUT_ACK_POLL_DELAY_US   1

/*!
 * Temporary defines to remove after changes go in to dev_trim or
 * addendum file.
 * http://lwbugs/2571189
 */
#define LW_PTRIM_SYS_NAFLL_LTCLUT_ACK_INIT_READ_LUT_DATA_READY          1
#define LW_PTRIM_SYS_NAFLL_LTCLUT_ACK_INIT_READ_LUT_DATA_DONE           0
#define LW_PTRIM_GPC_BCAST_GPCLUT_ACK_INIT_READ_LUT_OFFSET_DATA_READY   1
#define LW_PTRIM_GPC_BCAST_GPCLUT_ACK_INIT_READ_LUT_OFFSET_DATA_DONE    0

#define LW_PTRIM_SYS_NAFLL_LTCLUT_ACK_RDACK_LUT_DEBUG_READY             1
#define LW_PTRIM_SYS_NAFLL_LTCLUT_ACK_RDACK_LUT_DEBUG_DONE              0
#define LW_PTRIM_GPC_BCAST_GPCLUT_ACK_RDACK_LUT_OFFSET_DATA_READY       1
#define LW_PTRIM_GPC_BCAST_GPCLUT_ACK_RDACK_LUT_OFFSET_DATA_DONE        0

//
// TODO: Remove this when LW_PTRIM_SYS_FR_CLK_CNTR_PEX_PAD_TCLKS_CFG_REGINTFCLK_TYPE_INIT vaue is fixed to
// LW_PTRIM_SYS_FR_CLK_CNTR_PEX_PAD_TCLKS_CFG_REGINTFCLK_TYPE_108M till then work with this war.
//
#define LWWATCH_CNTRFREQ_WAR_200485322

static CLK_FR_COUNTER_REG_INFO clkFrCounterRegInfo_GA100 = {
    LW_PTRIM_SYS_FR_CLK_CNTR_PEX_PAD_TCLKS_CFG,
    0,
    LW_PTRIM_SYS_FR_CLK_CNTR_PEX_PAD_TCLKS_CNT0,
    LW_PTRIM_SYS_FR_CLK_CNTR_PEX_PAD_TCLKS_CNT1,
};

static CLK_FR_COUNTER_SRC_INFO clkFrCounterSrcInfo_GA100[] = {
    {
        "DRAM",
        LW2080_CTRL_CLK_DOMAIN_MCLK,
        12,
        {
            {0, "MCLK[0]" },
            {1, "MCLK[1]" },
            {2, "MCLK[2]" },
            {3, "MCLK[3]" },
            {4, "MCLK[4]" },
            {5, "MCLK[5]" },
            {6, "MCLK[6]" },
            {7, "MCLK[7]" },
            {8, "MCLK[8]" },
            {9, "MCLK[9]" },
            {10, "MCLK[10]" },
            {11, "MCLK[11]" },
        },
    },
    {
        "GPC",
        LW2080_CTRL_CLK_DOMAIN_GPCCLK,
        8,
        {
            {0, "GPCCLK[0]" },
            {1, "GPCCLK[1]" },
            {2, "GPCCLK[2]" },
            {3, "GPCCLK[3]" },
            {4, "GPCCLK[4]" },
            {5, "GPCCLK[5]" },
            {6, "GPCCLK[6]" },
            {7, "GPCCLK[7]" },
        },
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
        1,
        {
            {0,  "SPPLL0"},
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
        "XBAR",
        LW2080_CTRL_CLK_DOMAIN_XBARCLK,
        1,
        {
            {0, "XBARCLK"},
        },
    },
};

/*!
 * @brief Set TCLKOUT source for given clock domain
 *
 * @param[in]   clkDomain   Clock domain LW2080_CTRL_CLK_DOMAIN_XX
 * @param[in]   srcIdx      Clock source srcIdx
 *
 * Generated by //sw/apps/gpu/drivers/resman/clocks/tclkout/get_clk_domain_prog_seq_from_tclkout_pm.pl on
 * chips_a/drivers/common/inc/hwref/<chip_family>/<chip>/tclkout_id.pm.
 * Refer https://confluence.lwpu.com/pages/viewpage.action?spaceKey=GPUC&title=TCLKOUT+Programming+Guide
 * for details.
 *
 * TODO: Auto generation of below chip specific source code by lwwatch-config
 */
LW_STATUS
clkSetSourceCntr_GA100
(
    LwU32 clkDomain,
    LwU32 srcIdx
)
{
    switch (clkDomain)
    {
        case LW2080_CTRL_CLK_DOMAIN_MCLK:
        {
            switch (srcIdx)
            {
                case 0:
                {
                    GPU_REG_WR32(LW_PTRIM_FBPA_TCLKOUT_CTRL_FBP(0), 0xc102040b);
                    GPU_REG_WR32(LW_PTRIM_FBPA_TCLKOUT_CTRL_FBP(2), 0xc102040b);
                    GPU_REG_WR32(LW_PTRIM_FBPA_TCLKOUT_CTRL_FBP(4), 0xc102040b);
                    GPU_REG_WR32(LW_PTRIM_SYS_TCLKOUT_CTRL_XBAR, 0xc10e0101);
                    GPU_REG_WR32(LW_PTRIM_SYS_TCLKOUT_CTRL_SYS, 0xc1030101);
                    break;
                }
                case 1:
                {
                    GPU_REG_WR32(LW_PTRIM_FBPA_TCLKOUT_CTRL_FBP(1), 0xc10a030b);
                    GPU_REG_WR32(LW_PTRIM_FBPA_TCLKOUT_CTRL_FBP(3), 0xc10a030b);
                    GPU_REG_WR32(LW_PTRIM_FBPA_TCLKOUT_CTRL_FBP(5), 0xc10a030b);
                    GPU_REG_WR32(LW_PTRIM_SYS_TCLKOUT_CTRL_XBAR, 0xc1080101);
                    GPU_REG_WR32(LW_PTRIM_SYS_TCLKOUT_CTRL_SYS, 0xc1030101);
                    break;
                }
                case 2:
                {
                    GPU_REG_WR32(LW_PTRIM_FBPA_TCLKOUT_CTRL_FBP(0), 0xc106030b);
                    GPU_REG_WR32(LW_PTRIM_FBPA_TCLKOUT_CTRL_FBP(2), 0xc106030b);
                    GPU_REG_WR32(LW_PTRIM_FBPA_TCLKOUT_CTRL_FBP(4), 0xc106030b);
                    GPU_REG_WR32(LW_PTRIM_SYS_TCLKOUT_CTRL_XBAR, 0xc10e0101);
                    GPU_REG_WR32(LW_PTRIM_SYS_TCLKOUT_CTRL_SYS, 0xc1030101);
                    break;
                }
                case 3:
                {
                    GPU_REG_WR32(LW_PTRIM_FBPA_TCLKOUT_CTRL_FBP(1), 0xc106030b);
                    GPU_REG_WR32(LW_PTRIM_FBPA_TCLKOUT_CTRL_FBP(3), 0xc106030b);
                    GPU_REG_WR32(LW_PTRIM_FBPA_TCLKOUT_CTRL_FBP(5), 0xc106030b);
                    GPU_REG_WR32(LW_PTRIM_SYS_TCLKOUT_CTRL_XBAR, 0xc1080101);
                    GPU_REG_WR32(LW_PTRIM_SYS_TCLKOUT_CTRL_SYS, 0xc1030101);
                    break;
                }
                case 4:
                {
                    GPU_REG_WR32(LW_PTRIM_FBPA_TCLKOUT_CTRL_FBP(0), 0xc10a030b);
                    GPU_REG_WR32(LW_PTRIM_FBPA_TCLKOUT_CTRL_FBP(2), 0xc10a030b);
                    GPU_REG_WR32(LW_PTRIM_FBPA_TCLKOUT_CTRL_FBP(4), 0xc10a030b);
                    GPU_REG_WR32(LW_PTRIM_SYS_TCLKOUT_CTRL_XBAR, 0xc10e0101);
                    GPU_REG_WR32(LW_PTRIM_SYS_TCLKOUT_CTRL_SYS, 0xc1030101);
                    break;
                }
                case 5:
                {
                    GPU_REG_WR32(LW_PTRIM_FBPA_TCLKOUT_CTRL_FBP(1), 0xc102040b);
                    GPU_REG_WR32(LW_PTRIM_FBPA_TCLKOUT_CTRL_FBP(3), 0xc102040b);
                    GPU_REG_WR32(LW_PTRIM_FBPA_TCLKOUT_CTRL_FBP(5), 0xc102040b);
                    GPU_REG_WR32(LW_PTRIM_SYS_TCLKOUT_CTRL_XBAR, 0xc1080101);
                    GPU_REG_WR32(LW_PTRIM_SYS_TCLKOUT_CTRL_SYS, 0xc1030101);
                    break;
                }
                case 6:
                {
                    GPU_REG_WR32(LW_PTRIM_FBPA_TCLKOUT_CTRL_FBP(6), 0xc10a030b);
                    GPU_REG_WR32(LW_PTRIM_FBPA_TCLKOUT_CTRL_FBP(8), 0xc10a030b);
                    GPU_REG_WR32(LW_PTRIM_FBPA_TCLKOUT_CTRL_FBP(10), 0xc10a030b);
                    GPU_REG_WR32(LW_PTRIM_SYS_TCLKOUT_CTRL_XBAR, 0xc10b0101);
                    GPU_REG_WR32(LW_PTRIM_SYS_TCLKOUT_CTRL_SYS, 0xc1030101);
                    break;
                }
                case 7:
                {
                    GPU_REG_WR32(LW_PTRIM_FBPA_TCLKOUT_CTRL_FBP(7), 0xc102040b);
                    GPU_REG_WR32(LW_PTRIM_FBPA_TCLKOUT_CTRL_FBP(9), 0xc102040b);
                    GPU_REG_WR32(LW_PTRIM_FBPA_TCLKOUT_CTRL_FBP(11), 0xc102040b);
                    GPU_REG_WR32(LW_PTRIM_SYS_TCLKOUT_CTRL_XBAR, 0xc1050101);
                    GPU_REG_WR32(LW_PTRIM_SYS_TCLKOUT_CTRL_SYS, 0xc1030101);
                    break;
                }
                case 8:
                {
                    GPU_REG_WR32(LW_PTRIM_FBPA_TCLKOUT_CTRL_FBP(6), 0xc106030b);
                    GPU_REG_WR32(LW_PTRIM_FBPA_TCLKOUT_CTRL_FBP(8), 0xc106030b);
                    GPU_REG_WR32(LW_PTRIM_FBPA_TCLKOUT_CTRL_FBP(10), 0xc106030b);
                    GPU_REG_WR32(LW_PTRIM_SYS_TCLKOUT_CTRL_XBAR, 0xc10b0101);
                    GPU_REG_WR32(LW_PTRIM_SYS_TCLKOUT_CTRL_SYS, 0xc1030101);
                    break;
                }
                case 9:
                {
                    GPU_REG_WR32(LW_PTRIM_FBPA_TCLKOUT_CTRL_FBP(7), 0xc106030b);
                    GPU_REG_WR32(LW_PTRIM_FBPA_TCLKOUT_CTRL_FBP(9), 0xc106030b);
                    GPU_REG_WR32(LW_PTRIM_FBPA_TCLKOUT_CTRL_FBP(11), 0xc106030b);
                    GPU_REG_WR32(LW_PTRIM_SYS_TCLKOUT_CTRL_XBAR, 0xc1050101);
                    GPU_REG_WR32(LW_PTRIM_SYS_TCLKOUT_CTRL_SYS, 0xc1030101);
                    break;
                }
                case 10:
                {
                    GPU_REG_WR32(LW_PTRIM_FBPA_TCLKOUT_CTRL_FBP(6), 0xc102040b);
                    GPU_REG_WR32(LW_PTRIM_FBPA_TCLKOUT_CTRL_FBP(8), 0xc102040b);
                    GPU_REG_WR32(LW_PTRIM_FBPA_TCLKOUT_CTRL_FBP(10), 0xc102040b);
                    GPU_REG_WR32(LW_PTRIM_SYS_TCLKOUT_CTRL_XBAR, 0xc10b0101);
                    GPU_REG_WR32(LW_PTRIM_SYS_TCLKOUT_CTRL_SYS, 0xc1030101);
                    break;
                }
                case 11:
                {
                    GPU_REG_WR32(LW_PTRIM_FBPA_TCLKOUT_CTRL_FBP(7), 0xc10a030b);
                    GPU_REG_WR32(LW_PTRIM_FBPA_TCLKOUT_CTRL_FBP(9), 0xc10a030b);
                    GPU_REG_WR32(LW_PTRIM_FBPA_TCLKOUT_CTRL_FBP(11), 0xc10a030b);
                    GPU_REG_WR32(LW_PTRIM_SYS_TCLKOUT_CTRL_XBAR, 0xc1050101);
                    GPU_REG_WR32(LW_PTRIM_SYS_TCLKOUT_CTRL_SYS, 0xc1030101);
                    break;
                }
                default:
                {
                    dprintf("lw: ERROR: clk Domain %x source %x not configured \n", clkDomain, srcIdx);
                    return LW_ERR_ILWALID_ARGUMENT;
                    break;
                }
            }
            break;
        }
        case LW2080_CTRL_CLK_DOMAIN_XBARCLK:
        {
            GPU_REG_WR32(LW_PTRIM_SYS_TCLKOUT_CTRL_XBAR, 0xc1060101);
            GPU_REG_WR32(LW_PTRIM_SYS_TCLKOUT_CTRL_SYS, 0xc1030101);
            break;
        }
        case LW2080_CTRL_CLK_DOMAIN_GPCCLK:
        {
            switch (srcIdx)
            {
                case 0:
                {
                    GPU_REG_WR32(LW_PTRIM_GPC_TCLKOUT_CTRL_GPC(srcIdx), 0xc1030101);
                    GPU_REG_WR32(LW_PTRIM_SYS_TCLKOUT_CTRL_XBAR, 0xc10d0101);
                    GPU_REG_WR32(LW_PTRIM_SYS_TCLKOUT_CTRL_SYS, 0xc1030101);
                    break;
                }
                case 1:
                {
                    GPU_REG_WR32(LW_PTRIM_GPC_TCLKOUT_CTRL_GPC(srcIdx), 0xc1030101);
                    GPU_REG_WR32(LW_PTRIM_SYS_TCLKOUT_CTRL_XBAR, 0xc10f0101);
                    GPU_REG_WR32(LW_PTRIM_SYS_TCLKOUT_CTRL_SYS, 0xc1030101);
                    break;
                }
                case 2:
                {
                    GPU_REG_WR32(LW_PTRIM_GPC_TCLKOUT_CTRL_GPC(srcIdx), 0xc1030101);
                    GPU_REG_WR32(LW_PTRIM_SYS_TCLKOUT_CTRL_XBAR, 0xc1090101);
                    GPU_REG_WR32(LW_PTRIM_SYS_TCLKOUT_CTRL_SYS, 0xc1030101);
                    break;
                }
                case 3:
                {
                    GPU_REG_WR32(LW_PTRIM_GPC_TCLKOUT_CTRL_GPC(srcIdx), 0xc1030101);
                    GPU_REG_WR32(LW_PTRIM_SYS_TCLKOUT_CTRL_XBAR, 0xc1070301);
                    GPU_REG_WR32(LW_PTRIM_SYS_TCLKOUT_CTRL_SYS, 0xc1030101);
                    break;
                }
                case 4:
                {
                    GPU_REG_WR32(LW_PTRIM_GPC_TCLKOUT_CTRL_GPC(srcIdx), 0xc1030101);
                    GPU_REG_WR32(LW_PTRIM_SYS_TCLKOUT_CTRL_XBAR, 0xc1040101);
                    GPU_REG_WR32(LW_PTRIM_SYS_TCLKOUT_CTRL_SYS, 0xc1030101);
                    break;
                }
                case 5:
                {
                    GPU_REG_WR32(LW_PTRIM_GPC_TCLKOUT_CTRL_GPC(srcIdx), 0xc1030101);
                    GPU_REG_WR32(LW_PTRIM_SYS_TCLKOUT_CTRL_XBAR, 0xc1060101);
                    GPU_REG_WR32(LW_PTRIM_SYS_TCLKOUT_CTRL_SYS, 0xc1030101);
                    break;
                }
                case 6:
                {
                    GPU_REG_WR32(LW_PTRIM_GPC_TCLKOUT_CTRL_GPC(srcIdx), 0xc1030101);
                    GPU_REG_WR32(LW_PTRIM_SYS_TCLKOUT_CTRL_XBAR, 0xc10c0101);
                    GPU_REG_WR32(LW_PTRIM_SYS_TCLKOUT_CTRL_SYS, 0xc1030101);
                    break;
                }
                case 7:
                {
                    GPU_REG_WR32(LW_PTRIM_GPC_TCLKOUT_CTRL_GPC(srcIdx), 0xc1030101);
                    GPU_REG_WR32(LW_PTRIM_SYS_TCLKOUT_CTRL_XBAR, 0xc10a0101);
                    GPU_REG_WR32(LW_PTRIM_SYS_TCLKOUT_CTRL_SYS, 0xc1030101);
                    break;
                }
                default:
                {
                    dprintf("lw: ERROR: clk Domain %x source %x not configured \n", clkDomain, srcIdx);
                    return LW_ERR_ILWALID_ARGUMENT;
                    break;
                }
            }
            break;
        }
        case LW2080_CTRL_CLK_DOMAIN_LWDCLK:
        {
            GPU_REG_WR32(LW_PTRIM_SYS_TCLKOUT_CTRL_SYS, 0xc1040101);
            break;
        }
        case LW2080_CTRL_CLK_DOMAIN_PWRCLK:
        {
            GPU_REG_WR32(LW_PTRIM_SYS_TCLKOUT_CTRL_SYS, 0xc1040201);
            break;
        }
        case LW2080_CTRL_CLK_DOMAIN_UTILSCLK:
        {
            GPU_REG_WR32(LW_PTRIM_SYS_TCLKOUT_CTRL_SYS, 0xc1020201);
            break;
        }
        case LW2080_CTRL_CLK_DOMAIN_HOSTCLK:
        {
            GPU_REG_WR32(LW_PTRIM_SYS_TCLKOUT_CTRL_SYS, 0xc1080101);
            break;
        }
        case LW2080_CTRL_CLK_DOMAIN_SYSCLK:
        {
            GPU_REG_WR32(LW_PTRIM_SYS_TCLKOUT_CTRL_SYS, 0xc1080201);
            break;
        }
        default:
        {
            dprintf("lw: ERROR: clk Domain %x source %x not configured \n", clkDomain, srcIdx);
            return LW_ERR_ILWALID_ARGUMENT;
            break;
        }
    }
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
clkEnableCntr_GA100
(
    LwU32 srcReg,
    LwU32 cfgReg,
    LwU32 clkDomain,
    LwU32 srcIdx,
    LwU32 clockInput
)
{
    LW_STATUS status = LW_OK;
    LwU32     data32;

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

    return status;
}


/*!
 * @brief Fetch the FR clock counter data
 *
 * @param[out]   ppClkFrCntrSrcInfo  Clock counter source info
 * @param[out]   ppClkFrCntrRegInfo  Clock counter register info
 * @param[out]   pNumCntrsToRead     Number of FR counters present
 */
LW_STATUS
clkGetFrCntrInfo_GA100
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
        *ppClkFrCntrSrcInfo = clkFrCounterSrcInfo_GA100;
    }

    if (ppClkFrCntrRegInfo != NULL)
    {
        *ppClkFrCntrRegInfo = &clkFrCounterRegInfo_GA100;
    }

    if (pNumCntrsToRead != NULL)
    {
        *pNumCntrsToRead  = sizeof(clkFrCounterSrcInfo_GA100)/sizeof(clkFrCounterSrcInfo_GA100[0]);
    }

    return LW_OK;
}

/*!
 * @brief Get PEX PAD source to decide if its driven by 27Mhz, 108Mhz, 100Mhz or 540Mhz. 
 *
 * @param[in] cfgReg Config register address
 */ 
LwU32
clkGetPexPadSource_GA100(LwU32 cfgReg)
{
#ifndef LWWATCH_CNTRFREQ_WAR_200485322
    LwU32 data    = GPU_REG_RD32(cfgReg);
    LwU32 freqSrc = DRF_VAL(_PTRIM_SYS_FR_CLK_CNTR, _PEX_PAD_TCLKS_CFG, _REGINTFCLK_TYPE, data);

    switch (freqSrc)
    {
        case LW_PTRIM_SYS_FR_CLK_CNTR_PEX_PAD_TCLKS_CFG_REGINTFCLK_TYPE_27M:
            return 27000;
        case LW_PTRIM_SYS_FR_CLK_CNTR_PEX_PAD_TCLKS_CFG_REGINTFCLK_TYPE_108M:
            return 108000;
        case LW_PTRIM_SYS_FR_CLK_CNTR_PEX_PAD_TCLKS_CFG_REGINTFCLK_TYPE_100M:
            return 100000;
        case LW_PTRIM_SYS_FR_CLK_CNTR_PEX_PAD_TCLKS_CFG_REGINTFCLK_TYPE_540M:
            return 540000;
        default:
            dprintf("lw: ERROR: freqSrc not recognised %d\n", freqSrc);
            return 0;

    }
#endif //LWWATCH_CNTRFREQ_WAR_200485322
    return 108000;
}

/*!
 * @brief  Multiply the given frequency by a scaling factor.
 *
 * @param[out] pFreq frequency to adjust to a scaling factor.
 */
void clkAdjustMclkScalingFactor_GA100(LwU32 *pFreq)
{
    //
    // br4clk clock was used here, hence the x2 factor
    // refer http://lwbugs/200485322 comment #90 to #95 for details.
    //
    *pFreq *= 2;
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
clkNafllGetNafllIdx_GA100
(
    LwU32 nafllId
)
{
    LwU32 idx =0;

    for (idx = 0; idx < LW_ARRAY_ELEMENTS(_nafllMap_GA100); idx++)
    {
        if (_nafllMap_GA100[idx].id == nafllId)
        {
            return idx;
        }
    }
    return CLK_NAFLL_ADDRESS_TABLE_IDX_ILWALID;
}

/*!
 * @brief Poll check for given NAFLL LUT register address
 *
 * @param[in]   addr        NAFLL LUT register address
 * @param[in]   mask        NAFLL LUT register mask
 * @param[in]   value       NAFLL LUT register value
 * @param[in]   delayUs     delay in us for register read to synchronize
 */
void
clkNafllLutPoll_GA100
(
    LwU32   addr,
    LwU32   mask,
    LwU32   value,
    LwU32   delayUs
)
{
    LwU32 data;

    osPerfDelay(delayUs);

    data = GPU_REG_RD32(addr);
    if ((data & mask) == value)
    {
        return;
    }

    dprintf("lw: ERROR: Poll timeout for reg 0x%x data 0x%x with mask 0x%x for value 0x%x delay %d us \n",
                    addr, data, mask, value, delayUs);
}

/*!
 * @brief  Read the programmed LUT value for a given NAFLL ID
 *
 * Steps:
 * 1) Callwlate or read back required parameters from hardware
 *    Number of LUT entries, LUT Stride, ADC Step size, Temperature Index
 * 2) Set LUT read address
 *    1) set OFFSET as (temperature index * LUT stride)
 *    2) toggle AUTO_INC from OFF to ON to reset the address counters properly
 * 3) Read LUT data
 *    For Each LUT stride or address to read
 *    1) Prepare LUT for read
 *       a) set LUT_ACK_INIT_READ_LUT_DATA = 1
 *       b) poll on LUT_ACK_RDACK_LUT_DEBUG = 1
 *    2) Read updated LUT data
 *    3) Prepare LUT for next read
 *       a) set LUT_ACK_INIT_READ_LUT_DATA = 0
 *       b) poll on LUT_ACK_RDACK_LUT_DEBUG = 0
 *
 * @param[in] nafllId   NAFLL index
 * @param[in] tempIdx   current temperature index
 */
void
clkNafllLutRead_GA100
(
    LwU32 nafllId,
    LwU32 tempIdx
)
{
    LwU32   reg32;
    LwU32   lutAckAddr;
    LwU32   lutReadDataAddr;
    LwU32   lutReadOffsetDataAddr;
    LwU32   addressVal;
    LwU32   dataVal;
    LwU32   adcStepSizeuV;
    LwU32   lutNumEntries;
    LwU32   lutStride;
    LwU32   lutStrideIdx;
    LwU32   adcCode0;
    LwU32   adcCode1;
    LwU32   ndiv0;
    LwU32   ndivOffsetA0;
    LwU32   dvcoOffsetA0;
    LwU32   ndivOffsetB0;
    LwU32   dvcoOffsetB0;
    LwU32   ndiv1;
    LwU32   ndivOffsetA1;
    LwU32   dvcoOffsetA1;
    LwU32   ndivOffsetB1;
    LwU32   dvcoOffsetB1;
    LwU32   nafllIdx;

    nafllIdx = pClk[indexGpu].clkNafllGetNafllIdx(nafllId);
    if (nafllIdx == CLK_NAFLL_ADDRESS_TABLE_IDX_ILWALID)
    {
        dprintf("NAFLL ID (%d) not found!!!\n", nafllId);
        return;
    }

    // step 1) Callwlate or read back required parameters from hardware

    //
    // From Ampere and onwards only 6.25 mv step sizes are supported
    // http://lwbugs/2136405/38
    //
    adcStepSizeuV = LW_PTRIM_SYS_NAFLL_SYSNAFLL_LWVDD_ADC_CTRL_STEP_SIZE_UV;

    // Callwlate the number of LUT entries & LUT stride
    lutNumEntries = ((LW2080_CTRL_CLK_LUT_MAX_VOLTAGE_UV -
                      LW2080_CTRL_CLK_LUT_MIN_VOLTAGE_UV) / adcStepSizeuV);
    lutStride     = lutNumEntries / CLK_LUT_ENTRIES_PER_STRIDE;

    // Get the current temperature index, if not already specified
    if (tempIdx > CLK_LUT_TEMP_IDX_MAX)
    {
        reg32   = CLK_NAFLL_REG_GET_GA100(nafllIdx, LUT_CFG);
        dataVal = GPU_REG_RD32(reg32);
        tempIdx = DRF_VAL(_PTRIM_SYS, _NAFLL_LTCLUT_CFG, _TEMP_INDEX, dataVal);
    }
    dprintf("Temperature Index: %d\n\n", tempIdx);

    // step 2) Set the read address now
    reg32   = CLK_NAFLL_REG_GET_GA100(nafllIdx, LUT_READ_ADDR);
    addressVal = FLD_SET_DRF_NUM(_PTRIM_SYS, _NAFLL_LTCLUT_READ_ADDR, _OFFSET,
                    (tempIdx * lutStride), 0);
    addressVal = FLD_SET_DRF(_PTRIM_SYS, _NAFLL_LTCLUT_READ_ADDR,
                    _AUTO_INC, _OFF, addressVal);
    GPU_REG_WR32(reg32, addressVal);

    // step 2.2) Toggle AUTO_INC to 0 then 1 to reset the address counters properly
    addressVal = FLD_SET_DRF(_PTRIM_SYS, _NAFLL_LTCLUT_READ_ADDR,
                    _AUTO_INC, _ON, addressVal);
    GPU_REG_WR32(reg32, addressVal);

    // step 3) Now for the actual LUT read
    lutReadDataAddr = CLK_NAFLL_REG_GET_GA100(nafllIdx, LUT_READ_DATA);
    lutReadOffsetDataAddr = CLK_NAFLL_REG_GET_GA100(nafllIdx, LUT_READ_OFFSET_DATA);
    lutAckAddr = CLK_NAFLL_REG_GET_GA100(nafllIdx, LUT_ACK);
    dprintf("LUT Table: \n");
    dprintf("|===================================================================================================================================================================| \n");
    if (lutReadOffsetDataAddr != CLK_REGISTER_ADDR_UNDEFINED)
    {
        dprintf("| ADC-code | Ndiv | Ndiv Offset A | Dvco Offset A | Ndiv Offset B | Dvco Offset B | ADC-code | Ndiv | Ndiv Offset A | Dvco Offset A | Ndiv Offset B | Dvco Offset B | \n");
    }
    else
    {
        dprintf("| ADC-code | Ndiv | ADC-code | Ndiv | \n");
    }
    dprintf("|===================================================================================================================================================================| \n");

    // Each DWORD in the LUT can hold two V/F table entries.
    for (lutStrideIdx = 0; lutStrideIdx < lutStride; lutStrideIdx++)
    {
        // step 3.1) Prepare LUT for Read
        dataVal = GPU_REG_RD32(lutAckAddr);
        dataVal = FLD_SET_DRF(_PTRIM_SYS, _NAFLL_LTCLUT_ACK,
                        _INIT_READ_LUT_DATA, _READY, dataVal);
        if (lutReadOffsetDataAddr != CLK_REGISTER_ADDR_UNDEFINED)
        {
            dataVal = FLD_SET_DRF(_PTRIM_GPC, _BCAST_GPCLUT_ACK,
                        _INIT_READ_LUT_OFFSET_DATA, _READY, dataVal);
        }
        GPU_REG_WR32(lutAckAddr, dataVal);

        pClk[indexGpu].clkNafllLutPoll(
                lutAckAddr,
                DRF_SHIFTMASK(LW_PTRIM_SYS_NAFLL_LTCLUT_ACK_RDACK_LUT_DEBUG),
                DRF_DEF(_PTRIM_SYS, _NAFLL_LTCLUT_ACK, _RDACK_LUT_DEBUG, _READY),
                CLK_LUT_ACK_POLL_DELAY_US);

        if (lutReadOffsetDataAddr != CLK_REGISTER_ADDR_UNDEFINED)
        {
            pClk[indexGpu].clkNafllLutPoll(
                lutAckAddr,
                DRF_SHIFTMASK(LW_PTRIM_GPC_BCAST_GPCLUT_ACK_RDACK_LUT_OFFSET_DATA),
                DRF_DEF(_PTRIM_GPC, _BCAST_GPCLUT_ACK, _RDACK_LUT_OFFSET_DATA, _READY),
                CLK_LUT_ACK_POLL_DELAY_US);
        }

        // step 3.2) Read LUT Data
        dataVal = GPU_REG_RD32(lutReadDataAddr);

        adcCode0 = (2 * lutStrideIdx);
        adcCode1 = (2 * lutStrideIdx) + 1;
        ndiv0    = DRF_VAL(_PTRIM_SYS, _NAFLL_LTCLUT_READ_DATA, _VAL0_NDIV,
                           dataVal);
        ndiv1    = DRF_VAL(_PTRIM_SYS, _NAFLL_LTCLUT_READ_DATA, _VAL1_NDIV,
                           dataVal);

        // step 3.2) Read LUT Offset Data
        if (lutReadOffsetDataAddr != CLK_REGISTER_ADDR_UNDEFINED)
        {
            dataVal = GPU_REG_RD32(lutReadOffsetDataAddr);

            ndivOffsetA0 = DRF_VAL(_PTRIM_GPC, _BCAST_GPCLUT_READ_OFFSET_DATA,
                                   _VAL0_NDIV_OFFSET_A, dataVal);
            dvcoOffsetA0 = DRF_VAL(_PTRIM_GPC, _BCAST_GPCLUT_READ_OFFSET_DATA,
                                   _VAL0_DVCO_OFFSET_A, dataVal);
            ndivOffsetB0 = DRF_VAL(_PTRIM_GPC, _BCAST_GPCLUT_READ_OFFSET_DATA,
                                   _VAL0_NDIV_OFFSET_B, dataVal);;
            dvcoOffsetB0 = DRF_VAL(_PTRIM_GPC, _BCAST_GPCLUT_READ_OFFSET_DATA,
                                   _VAL0_DVCO_OFFSET_B, dataVal);
            ndivOffsetA1 = DRF_VAL(_PTRIM_GPC, _BCAST_GPCLUT_READ_OFFSET_DATA,
                                   _VAL1_NDIV_OFFSET_A, dataVal);
            dvcoOffsetA1 = DRF_VAL(_PTRIM_GPC, _BCAST_GPCLUT_READ_OFFSET_DATA,
                                   _VAL1_DVCO_OFFSET_A, dataVal);
            ndivOffsetB1 = DRF_VAL(_PTRIM_GPC, _BCAST_GPCLUT_READ_OFFSET_DATA,
                                   _VAL1_NDIV_OFFSET_B, dataVal);;
            dvcoOffsetB1 = DRF_VAL(_PTRIM_GPC, _BCAST_GPCLUT_READ_OFFSET_DATA,
                                   _VAL1_DVCO_OFFSET_B, dataVal);
        }

        if (lutReadOffsetDataAddr != CLK_REGISTER_ADDR_UNDEFINED)
        {
            dprintf("|    %-4d  | %-4d |        %-4d   |        %-4d   |        %-4d   |        %-4d   |    %-4d  | %-4d |         %-4d   |        %-4d   |        %-4d   |        %-4d   |\n",
                        adcCode0, ndiv0,   ndivOffsetA0,   dvcoOffsetA0,   ndivOffsetB0,   dvcoOffsetB0,  adcCode1, ndiv1,    ndivOffsetA1,   dvcoOffsetA1,   ndivOffsetB1,   dvcoOffsetB1);
        }
        else
        {
            dprintf("|    %-4d  | %-4d |    %-4d  | %-4d |\n",
                        adcCode0, ndiv0,  adcCode1, ndiv1);
        }

        // step 3.3) Prepare LUT for next read
        dataVal = GPU_REG_RD32(lutAckAddr);
        dataVal = FLD_SET_DRF(_PTRIM_SYS, _NAFLL_LTCLUT_ACK,
                        _INIT_READ_LUT_DATA, _DONE, dataVal);
        if (lutReadOffsetDataAddr != CLK_REGISTER_ADDR_UNDEFINED)
        {
            dataVal = FLD_SET_DRF(_PTRIM_GPC, _BCAST_GPCLUT_ACK,
                        _INIT_READ_LUT_OFFSET_DATA, _DONE, dataVal);
        }
        GPU_REG_WR32(lutAckAddr, dataVal);

        pClk[indexGpu].clkNafllLutPoll(
                lutAckAddr,
                DRF_SHIFTMASK(LW_PTRIM_SYS_NAFLL_LTCLUT_ACK_RDACK_LUT_DEBUG),
                DRF_DEF(_PTRIM_SYS, _NAFLL_LTCLUT_ACK, _RDACK_LUT_DEBUG, _DONE),
                CLK_LUT_ACK_POLL_DELAY_US);

        if (lutReadOffsetDataAddr != CLK_REGISTER_ADDR_UNDEFINED)
        {
            pClk[indexGpu].clkNafllLutPoll(
                lutAckAddr,
                DRF_SHIFTMASK(LW_PTRIM_GPC_BCAST_GPCLUT_ACK_RDACK_LUT_OFFSET_DATA),
                DRF_DEF(_PTRIM_GPC, _BCAST_GPCLUT_ACK, _RDACK_LUT_OFFSET_DATA, _DONE),
                CLK_LUT_ACK_POLL_DELAY_US);
        }

    }
    dprintf("|===================================================================================================================================================================| \n");
}

/*!
 * @brief  Read back NAFLL frequency for a given NAFLL ID
 *
 * @param[in] nafllId   NAFLL index
 *
 * @return NAFLL frequency callwlated from dividers read from hardware
 *         0 otherwise
 *
 * TODO: remove, as it is exact replica of _GV100, after generalization of
 *       CLK_NAFLL_REG_GET
 * http://lwbugs/2571189
 */
LwU32 
clkGetNafllFreqKHz_GA100
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
    reg32   = CLK_NAFLL_REG_GET_GA100(nafllIdx, NAFLL_COEFF);
    dataVal = GPU_REG_RD32(reg32);
    Mdiv    = DRF_VAL(_PTRIM, _SYS_NAFLL_SYSNAFLL_COEFF, _MDIV, dataVal);
    dprintf("lw:   Mdiv     = %4d\n", Mdiv);

    if (Mdiv == 0) return 0;

    // Read the NDiv
    reg32   = CLK_NAFLL_REG_GET_GA100(nafllIdx, LUT_DEBUG2);
    dataVal = GPU_REG_RD32(reg32);
    Ndiv    = DRF_VAL(_PTRIM, _SYS_NAFLL_SYSLUT_DEBUG2, _NDIV, dataVal);
    dprintf("lw:   Ndiv     = %4d\n", Ndiv);

    // Callwlate back the freq value
    return (RefFreqKHz * Ndiv / Mdiv);
}
