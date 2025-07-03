/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2021 by LWPU Corporation.  All rights reserved.  All information
 * contained herein is proprietary and confidential to LWPU Corporation.  Any
 * use, reproduction, or disclosure without the written permission of LWPU
 * Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

//*****************************************************
//
// clkgh100.c - GH100 Clock lwwatch routines 
// 
//*****************************************************
#include "os.h"
#include "hal.h"
#include "clk.h"
#include "print.h"
#include "hopper/gh100/dev_trim.h"
#include "ctrl/ctrl2080/ctrl2080clk.h"

#include "g_clk_private.h"           // (rmconfig) implementation prototypes.

static CLK_FR_COUNTER_REG_INFO clkFrCounterRegInfo_GH100 = {
    LW_PTRIM_PCIE_FR_CLK_CNTR_PEX_PAD_TCLKS_CFG,
    0,
    LW_PTRIM_PCIE_FR_CLK_CNTR_PEX_PAD_TCLKS_CNT0,
    LW_PTRIM_PCIE_FR_CLK_CNTR_PEX_PAD_TCLKS_CNT1,
};

static CLK_FR_COUNTER_SRC_INFO clkFrCounterSrcInfo_GH100[] = {
    {
        "DRAM",
        LW2080_CTRL_CLK_DOMAIN_MCLK,
        6,
        {
            {0, "MCLK[0]" },
            {1, "MCLK[1]" },
            {2, "MCLK[2]" },
            {3, "MCLK[3]" },
            {4, "MCLK[4]" },
            {5, "MCLK[5]" },
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

/*
 * Mapping between the NAFLL ID and the various LUT registers for that NAFLL
 */
static CLK_NAFLL_ADDRESS_MAP _nafllMap_GH100[] =
{
    {
        CLK_NAFLL_ID_SYS,
        {
            LW_PTRIM_SYS_NAFLL_SYSLUT_READ_ADDR,
            LW_PTRIM_SYS_NAFLL_SYSLUT_READ_DATA,
            LW_PTRIM_SYS_NAFLL_SYSLUT_CFG,
            LW_PTRIM_SYS_NAFLL_SYSLUT_STATUS,
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
            LW_PTRIM_SYS_NAFLL_XBARLUT_STATUS,
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
            LW_PTRIM_GPC_GPCLUT_STATUS(0),
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
            LW_PTRIM_GPC_GPCLUT_STATUS(1),
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
            LW_PTRIM_GPC_GPCLUT_STATUS(2),
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
            LW_PTRIM_GPC_GPCLUT_STATUS(3),
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
            LW_PTRIM_GPC_GPCLUT_STATUS(4),
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
            LW_PTRIM_GPC_GPCLUT_STATUS(5),
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
            LW_PTRIM_GPC_GPCLUT_STATUS(6),
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
            LW_PTRIM_GPC_GPCLUT_STATUS(7),
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
            LW_PTRIM_GPC_BCAST_GPCLUT_STATUS,
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
            LW_PTRIM_SYS_NAFLL_LWDLUT_STATUS,
            LW_PTRIM_SYS_NAFLL_LWDNAFLL_COEFF,
            LW_PTRIM_SYS_NAFLL_LWDLUT_ACK,
            CLK_REGISTER_ADDR_UNDEFINED,
        }
    },
};

/*!
 * Helper MACRO to access the static table above
 */
#define CLK_NAFLL_REG_GET_GH100(_nafllIdx,_type)                              \
    (_nafllMap_GH100[_nafllIdx].regAddr[CLK_NAFLL_REG_TYPE_##_type])

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


/*!
 * @brief Fetch the FR clock counter data
 *
 * @param[out]   ppClkFrCntrSrcInfo  Clock counter source info
 * @param[out]   ppClkFrCntrRegInfo  Clock counter register info
 * @param[out]   pNumCntrsToRead     Number of FR counters present
 */
LW_STATUS
clkGetFrCntrInfo_GH100
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
        *ppClkFrCntrSrcInfo = clkFrCounterSrcInfo_GH100;
    }

    if (ppClkFrCntrRegInfo != NULL)
    {
        *ppClkFrCntrRegInfo = &clkFrCounterRegInfo_GH100;
    }

    if (pNumCntrsToRead != NULL)
    {
        *pNumCntrsToRead  = sizeof(clkFrCounterSrcInfo_GH100)/sizeof(clkFrCounterSrcInfo_GH100[0]);
    }

    return LW_OK;
}

/*!
 * @brief Set TCLKOUT source for given clock domain
 *
 * @param[in]   clkDomain   Clock domain LW2080_CTRL_CLK_DOMAIN_XX
 * @param[in]   srcIdx      Clock source srcIdx
 *
 * Generated by get_clk_domain_prog_seq_from_tclkout_pm.pl on 
 * chips_a/drivers/common/inc/hwref/<chip_family>/<chip>/tclkout_id.pm.
 * Refer https://confluence.lwpu.com/pages/viewpage.action?spaceKey=GPUC&title=TCLKOUT+Programming+Guide
 * for details.
 *
 * TODO: Auto generation of below chip specific source code by lwwatch-config
 */
LW_STATUS
clkSetSourceCntr_GH100
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
                    GPU_REG_WR32(LW_PTRIM_FBPA_TCLKOUT_CTRL_FBPA_TCLKOUT_DEFAULT_FAST_PA(0), 0xc2030f0f);
                    GPU_REG_WR32(LW_PTRIM_FBPA_TCLKOUT_CTRL_FBP_TCLKOUT_DEFAULT_FAST(0), 0xc1120201);
                    GPU_REG_WR32(LW_PTRIM_SYS_TCLKOUT_CTRL_XBAR_TCLKOUT_DEFAULT_FAST_XBAR, 0xc1030101);
                    GPU_REG_WR32(LW_PTRIM_LWLINK_TCLKOUT_CTRL_LWLINK_TCLKOUT_DEFAULT_FAST, 0xc10b2f01);
                    GPU_REG_WR32(LW_PTRIM_PCIE_TCLKOUT_CTRL_MATHS_TCLKOUT_DEFAULT_FAST_MATHS, 0xc1031c01);
                    break;
                }
                case 1:
                {
                    GPU_REG_WR32(LW_PTRIM_FBPA_TCLKOUT_CTRL_FBPA_TCLKOUT_DEFAULT_FAST_PA(10), 0xc2030f0f);
                    GPU_REG_WR32(LW_PTRIM_FBPA_TCLKOUT_CTRL_FBP_TCLKOUT_DEFAULT_FAST(10), 0xc1120201);
                    GPU_REG_WR32(LW_PTRIM_SYS_TCLKOUT_CTRL_XBAR_TCLKOUT_DEFAULT_FAST_XBAR, 0xc1040101);
                    GPU_REG_WR32(LW_PTRIM_LWLINK_TCLKOUT_CTRL_LWLINK_TCLKOUT_DEFAULT_FAST, 0xc10b2f01);
                    GPU_REG_WR32(LW_PTRIM_PCIE_TCLKOUT_CTRL_MATHS_TCLKOUT_DEFAULT_FAST_MATHS, 0xc1031c01);
                    break;
                }
                case 2:
                {
                    GPU_REG_WR32(LW_PTRIM_FBPA_TCLKOUT_CTRL_FBPA_TCLKOUT_DEFAULT_FAST_PA(4), 0xc1060f0f);
                    GPU_REG_WR32(LW_PTRIM_FBPA_TCLKOUT_CTRL_FBP_TCLKOUT_DEFAULT_FAST(5), 0xc1130101);
                    GPU_REG_WR32(LW_PTRIM_SYS_TCLKOUT_CTRL_XBAR_TCLKOUT_DEFAULT_FAST_XBAR, 0xc1060101);
                    GPU_REG_WR32(LW_PTRIM_LWLINK_TCLKOUT_CTRL_LWLINK_TCLKOUT_DEFAULT_FAST, 0xc10b2f01);
                    GPU_REG_WR32(LW_PTRIM_PCIE_TCLKOUT_CTRL_MATHS_TCLKOUT_DEFAULT_FAST_MATHS, 0xc1031c01);
                    break;
                }
                case 3:
                {
                    GPU_REG_WR32(LW_PTRIM_FBPA_TCLKOUT_CTRL_FBPA_TCLKOUT_DEFAULT_FAST_PA(6), 0xc1060f0f);
                    GPU_REG_WR32(LW_PTRIM_FBPA_TCLKOUT_CTRL_FBP_TCLKOUT_DEFAULT_FAST(7), 0xc1130101);
                    GPU_REG_WR32(LW_PTRIM_SYS_TCLKOUT_CTRL_XBAR_TCLKOUT_DEFAULT_FAST_XBAR, 0xc10a0101);
                    GPU_REG_WR32(LW_PTRIM_LWLINK_TCLKOUT_CTRL_LWLINK_TCLKOUT_DEFAULT_FAST, 0xc10b2f01);
                    GPU_REG_WR32(LW_PTRIM_PCIE_TCLKOUT_CTRL_MATHS_TCLKOUT_DEFAULT_FAST_MATHS, 0xc1031c01);
                    break;
                }
                case 4:
                {
                    GPU_REG_WR32(LW_PTRIM_FBPA_TCLKOUT_CTRL_FBPA_TCLKOUT_DEFAULT_FAST_PA(5), 0xc2030f0f);
                    GPU_REG_WR32(LW_PTRIM_FBPA_TCLKOUT_CTRL_FBP_TCLKOUT_DEFAULT_FAST(5), 0xc1120201);
                    GPU_REG_WR32(LW_PTRIM_SYS_TCLKOUT_CTRL_XBAR_TCLKOUT_DEFAULT_FAST_XBAR, 0xc1060101);
                    GPU_REG_WR32(LW_PTRIM_LWLINK_TCLKOUT_CTRL_LWLINK_TCLKOUT_DEFAULT_FAST, 0xc10b2f01);
                    GPU_REG_WR32(LW_PTRIM_PCIE_TCLKOUT_CTRL_MATHS_TCLKOUT_DEFAULT_FAST_MATHS, 0xc1031c01);
                    break;
                }
                case 5:
                {
                    GPU_REG_WR32(LW_PTRIM_FBPA_TCLKOUT_CTRL_FBPA_TCLKOUT_DEFAULT_FAST_PA(7), 0xc2030f0f);
                    GPU_REG_WR32(LW_PTRIM_FBPA_TCLKOUT_CTRL_FBP_TCLKOUT_DEFAULT_FAST(7), 0xc1120201);
                    GPU_REG_WR32(LW_PTRIM_SYS_TCLKOUT_CTRL_XBAR_TCLKOUT_DEFAULT_FAST_XBAR, 0xc10a0101);
                    GPU_REG_WR32(LW_PTRIM_LWLINK_TCLKOUT_CTRL_LWLINK_TCLKOUT_DEFAULT_FAST, 0xc10b2f01);
                    GPU_REG_WR32(LW_PTRIM_PCIE_TCLKOUT_CTRL_MATHS_TCLKOUT_DEFAULT_FAST_MATHS, 0xc1031c01);
                    break;
                }
                default:
                {
                    dprintf("lw: ERROR: clk Domain %x source %x not configured \n", clkDomain, srcIdx);
                    return LW_ERR_NOT_SUPPORTED;
                    break;
                }
            }
            break;
        }
        case LW2080_CTRL_CLK_DOMAIN_GPCCLK:
        {
            switch(srcIdx)
            {
                case 0:
                {
                    GPU_REG_WR32(LW_PTRIM_GPC_TCLKOUT_CTRL_GPC_TCLKOUT_DEFAULT_FAST(srcIdx), 0xc1050401);
                    GPU_REG_WR32(LW_PTRIM_SYS_TCLKOUT_CTRL_XBAR_TCLKOUT_DEFAULT_FAST_XBAR, 0xc10d0401);
                    GPU_REG_WR32(LW_PTRIM_LWLINK_TCLKOUT_CTRL_LWLINK_TCLKOUT_DEFAULT_FAST, 0xc10b2f01);
                    GPU_REG_WR32(LW_PTRIM_PCIE_TCLKOUT_CTRL_MATHS_TCLKOUT_DEFAULT_FAST_MATHS, 0xc1031c01);
                    break;
                } 
                case 1:
                {
                    GPU_REG_WR32(LW_PTRIM_GPC_TCLKOUT_CTRL_GPC_TCLKOUT_DEFAULT_FAST(srcIdx), 0xc1040401);
                    GPU_REG_WR32(LW_PTRIM_SYS_TCLKOUT_CTRL_XBAR_TCLKOUT_DEFAULT_FAST_XBAR, 0xc10c0101);
                    GPU_REG_WR32(LW_PTRIM_LWLINK_TCLKOUT_CTRL_LWLINK_TCLKOUT_DEFAULT_FAST, 0xc10b2f01);
                    GPU_REG_WR32(LW_PTRIM_PCIE_TCLKOUT_CTRL_MATHS_TCLKOUT_DEFAULT_FAST_MATHS, 0xc1031c01);
                    break;
                }
                case 2:
                {
                    GPU_REG_WR32(LW_PTRIM_GPC_TCLKOUT_CTRL_GPC_TCLKOUT_DEFAULT_FAST(srcIdx), 0xc1040401);
                    GPU_REG_WR32(LW_PTRIM_SYS_TCLKOUT_CTRL_XBAR_TCLKOUT_DEFAULT_FAST_XBAR, 0xc1090101);
                    GPU_REG_WR32(LW_PTRIM_LWLINK_TCLKOUT_CTRL_LWLINK_TCLKOUT_DEFAULT_FAST, 0xc10b2f01);
                    GPU_REG_WR32(LW_PTRIM_PCIE_TCLKOUT_CTRL_MATHS_TCLKOUT_DEFAULT_FAST_MATHS, 0xc1031c01);
                    break;
                }
                case 3:
                {
                    GPU_REG_WR32(LW_PTRIM_GPC_TCLKOUT_CTRL_GPC_TCLKOUT_DEFAULT_FAST(srcIdx), 0xc1040401);
                    GPU_REG_WR32(LW_PTRIM_SYS_TCLKOUT_CTRL_XBAR_TCLKOUT_DEFAULT_FAST_XBAR, 0xc1050101);
                    GPU_REG_WR32(LW_PTRIM_LWLINK_TCLKOUT_CTRL_LWLINK_TCLKOUT_DEFAULT_FAST, 0xc10b2f01);
                    GPU_REG_WR32(LW_PTRIM_PCIE_TCLKOUT_CTRL_MATHS_TCLKOUT_DEFAULT_FAST_MATHS, 0xc1031c01);
                    break;
                }
                case 4:
                {
                    GPU_REG_WR32(LW_PTRIM_GPC_TCLKOUT_CTRL_GPC_TCLKOUT_DEFAULT_FAST(srcIdx), 0xc1040401);
                    GPU_REG_WR32(LW_PTRIM_SYS_TCLKOUT_CTRL_XBAR_TCLKOUT_DEFAULT_FAST_XBAR, 0xc1080101);
                    GPU_REG_WR32(LW_PTRIM_LWLINK_TCLKOUT_CTRL_LWLINK_TCLKOUT_DEFAULT_FAST, 0xc10b2f01);
                    GPU_REG_WR32(LW_PTRIM_PCIE_TCLKOUT_CTRL_MATHS_TCLKOUT_DEFAULT_FAST_MATHS, 0xc1031c01);
                    break;
                }
                case 5:
                {
                    GPU_REG_WR32(LW_PTRIM_GPC_TCLKOUT_CTRL_GPC_TCLKOUT_DEFAULT_FAST(srcIdx), 0xc1040401);
                    GPU_REG_WR32(LW_PTRIM_SYS_TCLKOUT_CTRL_XBAR_TCLKOUT_DEFAULT_FAST_XBAR, 0xc1070101);
                    GPU_REG_WR32(LW_PTRIM_LWLINK_TCLKOUT_CTRL_LWLINK_TCLKOUT_DEFAULT_FAST, 0xc10b2f01);
                    GPU_REG_WR32(LW_PTRIM_PCIE_TCLKOUT_CTRL_MATHS_TCLKOUT_DEFAULT_FAST_MATHS, 0xc1031c01);
                    break;
                }
                case 6:
                {
                    GPU_REG_WR32(LW_PTRIM_GPC_TCLKOUT_CTRL_GPC_TCLKOUT_DEFAULT_FAST(srcIdx), 0xc1040401);
                    GPU_REG_WR32(LW_PTRIM_SYS_TCLKOUT_CTRL_XBAR_TCLKOUT_DEFAULT_FAST_XBAR, 0xc1020101);
                    GPU_REG_WR32(LW_PTRIM_LWLINK_TCLKOUT_CTRL_LWLINK_TCLKOUT_DEFAULT_FAST, 0xc10b2f01);
                    GPU_REG_WR32(LW_PTRIM_PCIE_TCLKOUT_CTRL_MATHS_TCLKOUT_DEFAULT_FAST_MATHS, 0xc1031c01);
                    break;
                }
                case 7:
                {
                    GPU_REG_WR32(LW_PTRIM_GPC_TCLKOUT_CTRL_GPC_TCLKOUT_DEFAULT_FAST(srcIdx), 0xc1040401);
                    GPU_REG_WR32(LW_PTRIM_SYS_TCLKOUT_CTRL_XBAR_TCLKOUT_DEFAULT_FAST_XBAR, 0xc10b1501);
                    GPU_REG_WR32(LW_PTRIM_LWLINK_TCLKOUT_CTRL_LWLINK_TCLKOUT_DEFAULT_FAST, 0xc10b2f01);
                    GPU_REG_WR32(LW_PTRIM_PCIE_TCLKOUT_CTRL_MATHS_TCLKOUT_DEFAULT_FAST_MATHS, 0xc1031c01);
                    break;
                }
                default:
                {
                    dprintf("lw: ERROR: clk Domain %x source %x not configured \n", clkDomain, srcIdx);
                    return LW_ERR_NOT_SUPPORTED;
                    break;
                }
            }
            break;
        }
        case LW2080_CTRL_CLK_DOMAIN_LWDCLK:
        {
            GPU_REG_WR32(LW_PTRIM_SYS_TCLKOUT_CTRL_SYS_TCLKOUT_DEFAULT_FAST, 0xc1050301);
            GPU_REG_WR32(LW_PTRIM_SYS_TCLKOUT_CTRL_XBAR_TCLKOUT_DEFAULT_FAST_XBAR, 0xc10d0501);
            GPU_REG_WR32(LW_PTRIM_LWLINK_TCLKOUT_CTRL_LWLINK_TCLKOUT_DEFAULT_FAST, 0xc10b2f01);
            GPU_REG_WR32(LW_PTRIM_PCIE_TCLKOUT_CTRL_MATHS_TCLKOUT_DEFAULT_FAST_MATHS, 0xc1031c01);
            break;
        }
        case LW2080_CTRL_CLK_DOMAIN_PWRCLK:
        {
            GPU_REG_WR32(LW_PTRIM_SYS_TCLKOUT_CTRL_SYS_TCLKOUT_DEFAULT_FAST, 0xc1050501);
            GPU_REG_WR32(LW_PTRIM_SYS_TCLKOUT_CTRL_XBAR_TCLKOUT_DEFAULT_FAST_XBAR, 0xc10d0501);
            GPU_REG_WR32(LW_PTRIM_LWLINK_TCLKOUT_CTRL_LWLINK_TCLKOUT_DEFAULT_FAST, 0xc10b2f01);
            GPU_REG_WR32(LW_PTRIM_PCIE_TCLKOUT_CTRL_MATHS_TCLKOUT_DEFAULT_FAST_MATHS, 0xc1031c01);
            break;
        }
        case LW2080_CTRL_CLK_DOMAIN_SYSCLK:
        {
            GPU_REG_WR32(LW_PTRIM_SYS_TCLKOUT_CTRL_SYS_TCLKOUT_DEFAULT_FAST, 0xc1050a01);
            GPU_REG_WR32(LW_PTRIM_SYS_TCLKOUT_CTRL_XBAR_TCLKOUT_DEFAULT_FAST_XBAR, 0xc10d0501);
            GPU_REG_WR32(LW_PTRIM_LWLINK_TCLKOUT_CTRL_LWLINK_TCLKOUT_DEFAULT_FAST, 0xc10b2f01);
            GPU_REG_WR32(LW_PTRIM_PCIE_TCLKOUT_CTRL_MATHS_TCLKOUT_DEFAULT_FAST_MATHS, 0xc1031c01);
            break;
        }
        case LW2080_CTRL_CLK_DOMAIN_UTILSCLK:
        {
            GPU_REG_WR32(LW_PTRIM_SYS_TCLKOUT_CTRL_SYS_TCLKOUT_DEFAULT_FAST, 0xc1050d01);
            GPU_REG_WR32(LW_PTRIM_SYS_TCLKOUT_CTRL_XBAR_TCLKOUT_DEFAULT_FAST_XBAR, 0xc10d0501);
            GPU_REG_WR32(LW_PTRIM_LWLINK_TCLKOUT_CTRL_LWLINK_TCLKOUT_DEFAULT_FAST, 0xc10b2f01);
            GPU_REG_WR32(LW_PTRIM_PCIE_TCLKOUT_CTRL_MATHS_TCLKOUT_DEFAULT_FAST_MATHS, 0xc1031c01);
            break;
        }
        case LW2080_CTRL_CLK_DOMAIN_XBARCLK:
        {
            GPU_REG_WR32(LW_PTRIM_SYS_TCLKOUT_CTRL_SYS_TCLKOUT_DEFAULT_FAST, 0xc1050e01);
            GPU_REG_WR32(LW_PTRIM_SYS_TCLKOUT_CTRL_XBAR_TCLKOUT_DEFAULT_FAST_XBAR, 0xc10d0501);
            GPU_REG_WR32(LW_PTRIM_LWLINK_TCLKOUT_CTRL_LWLINK_TCLKOUT_DEFAULT_FAST, 0xc10b2f01);
            GPU_REG_WR32(LW_PTRIM_PCIE_TCLKOUT_CTRL_MATHS_TCLKOUT_DEFAULT_FAST_MATHS, 0xc1031c01);
            break;
        }
        default:
        {
            dprintf("lw: ERROR: clk Domain %x source %x not configured \n", clkDomain, srcIdx);
            return LW_ERR_NOT_SUPPORTED;
            break;
        }
    }

    return LW_OK;
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
clkNafllGetNafllIdx_GH100
(
    LwU32 nafllId
)
{
    LwU32 idx =0;

    for (idx = 0; idx < LW_ARRAY_ELEMENTS(_nafllMap_GH100); idx++)
    {
        if (_nafllMap_GH100[idx].id == nafllId)
        {
            return idx;
        }
    }
    return CLK_NAFLL_ADDRESS_TABLE_IDX_ILWALID;
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
clkNafllLutRead_GH100
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
        reg32   = CLK_NAFLL_REG_GET_GH100(nafllIdx, LUT_CFG);
        dataVal = GPU_REG_RD32(reg32);
        tempIdx = DRF_VAL(_PTRIM_SYS, _NAFLL_LTCLUT_CFG, _TEMP_INDEX, dataVal);
    }
    dprintf("Temperature Index: %d\n\n", tempIdx);

    // step 2) Set the read address now
    reg32   = CLK_NAFLL_REG_GET_GH100(nafllIdx, LUT_READ_ADDR);
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
    lutReadDataAddr = CLK_NAFLL_REG_GET_GH100(nafllIdx, LUT_READ_DATA);
    lutReadOffsetDataAddr = CLK_NAFLL_REG_GET_GH100(nafllIdx, LUT_READ_OFFSET_DATA);
    lutAckAddr = CLK_NAFLL_REG_GET_GH100(nafllIdx, LUT_ACK);
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
