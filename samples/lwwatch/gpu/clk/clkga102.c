/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2020-2021 by LWPU Corporation.  All rights reserved.  All information
 * contained herein is proprietary and confidential to LWPU Corporation.  Any
 * use, reproduction, or disclosure without the written permission of LWPU
 * Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

//*****************************************************
//
// clkga102.c - GA102 Clock lwwatch routines 
// 
//*****************************************************
#include "os.h"
#include "hal.h"
#include "clk.h"
#include "print.h"
#include "ampere/ga102/dev_trim.h"
#include "ampere/ga102/dev_trim_addendum.h"
#include "ctrl/ctrl2080/ctrl2080clk.h"

#include "g_clk_private.h"           // (rmconfig) implementation prototypes.

static CLK_FR_COUNTER_REG_INFO clkFrCounterRegInfo_GA102 = {
    LW_PTRIM_SYS_FR_CLK_CNTR_PEX_PAD_TCLKS_CFG,
    0,
    LW_PTRIM_SYS_FR_CLK_CNTR_PEX_PAD_TCLKS_CNT0,
    LW_PTRIM_SYS_FR_CLK_CNTR_PEX_PAD_TCLKS_CNT1,
};

static CLK_FR_COUNTER_SRC_INFO clkFrCounterSrcInfo_GA102[] = {
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
        7,
        {
            {0, "GPCCLK[0]" },
            {1, "GPCCLK[1]" },
            {2, "GPCCLK[2]" },
            {3, "GPCCLK[3]" },
            {4, "GPCCLK[4]" },
            {5, "GPCCLK[5]" },
            {6, "GPCCLK[6]" },
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
    {
        "DISP",
        LW2080_CTRL_CLK_DOMAIN_DISPCLK,
        1,
        {
            {0, "DISPCLK"},
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
};

/*
 * Mapping between the NAFLL ID and the various LUT registers for that NAFLL
 */
static CLK_NAFLL_ADDRESS_MAP _nafllMap_GA102[] =
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
    {
        CLK_NAFLL_ID_HOST,
        {
            LW_PTRIM_SYS_NAFLL_HOSTLUT_READ_ADDR,
            LW_PTRIM_SYS_NAFLL_HOSTLUT_READ_DATA,
            LW_PTRIM_SYS_NAFLL_HOSTLUT_CFG,
            LW_PTRIM_SYS_NAFLL_HOSTLUT_STATUS,
            LW_PTRIM_SYS_NAFLL_HOSTNAFLL_COEFF,
            LW_PTRIM_SYS_NAFLL_HOSTLUT_ACK,
            CLK_REGISTER_ADDR_UNDEFINED,
        }
    },
};

static LwU16 _clkNafllGetRefClkDivider(LwU32 nafllId);

/*!
 * Helper MACRO to access the static table above
 */
#define CLK_NAFLL_REG_GET_GA102(_nafllIdx,_type)                              \
    (_nafllMap_GA102[_nafllIdx].regAddr[CLK_NAFLL_REG_TYPE_##_type])

/*!
 * @brief Fetch the FR clock counter data
 *
 * @param[out]   ppClkFrCntrSrcInfo  Clock counter source info
 * @param[out]   ppClkFrCntrRegInfo  Clock counter register info
 * @param[out]   pNumCntrsToRead     Number of FR counters present
 */
LW_STATUS
clkGetFrCntrInfo_GA102
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
        *ppClkFrCntrSrcInfo = clkFrCounterSrcInfo_GA102;
    }

    if (ppClkFrCntrRegInfo != NULL)
    {
        *ppClkFrCntrRegInfo = &clkFrCounterRegInfo_GA102;
    }

    if (pNumCntrsToRead != NULL)
    {
        *pNumCntrsToRead  = sizeof(clkFrCounterSrcInfo_GA102)/sizeof(clkFrCounterSrcInfo_GA102[0]);
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
clkSetSourceCntr_GA102
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
                    GPU_REG_WR32(LW_PTRIM_FBPA_TCLKOUT_CTRL_FBP(srcIdx), 0xc1020704);
                    GPU_REG_WR32(LW_PTRIM_SYS_TCLKOUT_CTRL_XBAR, 0xc10b0101);
                    GPU_REG_WR32(LW_PTRIM_SYS_TCLKOUT_CTRL_SYS, 0xc10b0101);
                    break;
                }
                case 1:
                {
                    GPU_REG_WR32(LW_PTRIM_FBPA_TCLKOUT_CTRL_FBP(srcIdx), 0xc1020704);
                    GPU_REG_WR32(LW_PTRIM_SYS_TCLKOUT_CTRL_XBAR, 0xc1070201);
                    GPU_REG_WR32(LW_PTRIM_SYS_TCLKOUT_CTRL_SYS, 0xc10b0101);
                    break;
                }
                case 2:
                {
                    GPU_REG_WR32(LW_PTRIM_FBPA_TCLKOUT_CTRL_FBP(srcIdx), 0xc1020704);
                    GPU_REG_WR32(LW_PTRIM_SYS_TCLKOUT_CTRL_XBAR, 0xc1090601);
                    GPU_REG_WR32(LW_PTRIM_SYS_TCLKOUT_CTRL_SYS, 0xc10b0101);
                    break;
                }
                case 3:
                {
                    GPU_REG_WR32(LW_PTRIM_FBPA_TCLKOUT_CTRL_FBP(srcIdx), 0xc1020704);
                    GPU_REG_WR32(LW_PTRIM_SYS_TCLKOUT_CTRL_XBAR, 0xc1080401);
                    GPU_REG_WR32(LW_PTRIM_SYS_TCLKOUT_CTRL_SYS, 0xc10b0101);
                    break;
                }
                case 4:
                {
                    GPU_REG_WR32(LW_PTRIM_FBPA_TCLKOUT_CTRL_FBP(srcIdx), 0xc1020704);
                    GPU_REG_WR32(LW_PTRIM_SYS_TCLKOUT_CTRL_XBAR, 0xc1060101);
                    GPU_REG_WR32(LW_PTRIM_SYS_TCLKOUT_CTRL_SYS, 0xc10b0101);
                    break;
                }
                case 5:
                {
                    GPU_REG_WR32(LW_PTRIM_FBPA_TCLKOUT_CTRL_FBP(srcIdx), 0xc1020704);
                    GPU_REG_WR32(LW_PTRIM_SYS_TCLKOUT_CTRL_XBAR, 0xc10a0501);
                    GPU_REG_WR32(LW_PTRIM_SYS_TCLKOUT_CTRL_SYS, 0xc10b0101);
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
                    GPU_REG_WR32(LW_PTRIM_GPC_TCLKOUT_CTRL_GPC(srcIdx), 0xc1030401);
                    GPU_REG_WR32(LW_PTRIM_SYS_TCLKOUT_CTRL_XBAR, 0xc1090301);
                    GPU_REG_WR32(LW_PTRIM_SYS_TCLKOUT_CTRL_SYS, 0xc10b0101);
                    break;
                } 
                case 1:
                {
                    GPU_REG_WR32(LW_PTRIM_GPC_TCLKOUT_CTRL_GPC(srcIdx), 0xc1030401);
                    GPU_REG_WR32(LW_PTRIM_SYS_TCLKOUT_CTRL_XBAR, 0xc1090401);
                    GPU_REG_WR32(LW_PTRIM_SYS_TCLKOUT_CTRL_SYS, 0xc10b0101);
                    break;
                }
                case 2:
                {
                    GPU_REG_WR32(LW_PTRIM_GPC_TCLKOUT_CTRL_GPC(srcIdx), 0xc1030401);
                    GPU_REG_WR32(LW_PTRIM_SYS_TCLKOUT_CTRL_XBAR, 0xc1090501);
                    GPU_REG_WR32(LW_PTRIM_SYS_TCLKOUT_CTRL_SYS, 0xc10b0101);
                    break;
                }
                case 3:
                {
                    GPU_REG_WR32(LW_PTRIM_GPC_TCLKOUT_CTRL_GPC(srcIdx), 0xc1030401);
                    GPU_REG_WR32(LW_PTRIM_SYS_TCLKOUT_CTRL_XBAR, 0xc1050101);
                    GPU_REG_WR32(LW_PTRIM_SYS_TCLKOUT_CTRL_SYS, 0xc10b0101);
                    break;
                }
                case 4:
                {
                    GPU_REG_WR32(LW_PTRIM_GPC_TCLKOUT_CTRL_GPC(srcIdx), 0xc1030401);
                    GPU_REG_WR32(LW_PTRIM_SYS_TCLKOUT_CTRL_XBAR, 0xc1080101);
                    GPU_REG_WR32(LW_PTRIM_SYS_TCLKOUT_CTRL_SYS, 0xc10b0101);
                    break;
                }
                case 5:
                {
                    GPU_REG_WR32(LW_PTRIM_GPC_TCLKOUT_CTRL_GPC(srcIdx), 0xc1030401);
                    GPU_REG_WR32(LW_PTRIM_SYS_TCLKOUT_CTRL_XBAR, 0xc1080201);
                    GPU_REG_WR32(LW_PTRIM_SYS_TCLKOUT_CTRL_SYS, 0xc10b0101);
                    break;
                }
                case 6:
                {
                    GPU_REG_WR32(LW_PTRIM_GPC_TCLKOUT_CTRL_GPC(srcIdx), 0xc1030401);
                    GPU_REG_WR32(LW_PTRIM_SYS_TCLKOUT_CTRL_XBAR, 0xc1080301);
                    GPU_REG_WR32(LW_PTRIM_SYS_TCLKOUT_CTRL_SYS, 0xc10b0101);
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
        case LW2080_CTRL_CLK_DOMAIN_HOSTCLK:
        {
            GPU_REG_WR32(LW_PTRIM_SYS_TCLKOUT_CTRL_SYS, 0xc10d0501);
            break;
        }
        case LW2080_CTRL_CLK_DOMAIN_LWDCLK:
        {
            GPU_REG_WR32(LW_PTRIM_SYS_TCLKOUT_CTRL_SYS, 0xc10d0701);
            break;
        }
        case LW2080_CTRL_CLK_DOMAIN_PWRCLK:
        {
            GPU_REG_WR32(LW_PTRIM_SYS_TCLKOUT_CTRL_SYS, 0xc10e0201);
            break;
        }
        case LW2080_CTRL_CLK_DOMAIN_SYSCLK:
        {
            GPU_REG_WR32(LW_PTRIM_SYS_TCLKOUT_CTRL_SYS, 0xc10d0901);
            break;
        }
        case LW2080_CTRL_CLK_DOMAIN_UTILSCLK:
        {
            GPU_REG_WR32(LW_PTRIM_SYS_TCLKOUT_CTRL_SYS, 0xc1090101);
            break;
        }
        case LW2080_CTRL_CLK_DOMAIN_XBARCLK:
        {
            GPU_REG_WR32(LW_PTRIM_SYS_TCLKOUT_CTRL_XBAR, 0xc1090101);
            GPU_REG_WR32(LW_PTRIM_SYS_TCLKOUT_CTRL_SYS, 0xc10b0101);
            break;
        }
        case LW2080_CTRL_CLK_DOMAIN_DISPCLK:
        {
            GPU_REG_WR32(LW_PTRIM_SYS_TCLKOUT_CTRL_SYS, 0xc1030323);
            break;
        }
        case LW2080_CTRL_CLK_DOMAIN_HUBCLK:
        {
            GPU_REG_WR32(LW_PTRIM_SYS_TCLKOUT_CTRL_SYS, 0xc1030201);
            break;
        }
        case LW2080_CTRL_CLK_DOMAIN_VCLK0:
        {
            switch(srcIdx)
            {
                case 0:
                {
                    GPU_REG_WR32(LW_PTRIM_SYS_TCLKOUT_CTRL_SYS, 0xc104020d);
                    break;
                }
                case 1:
                {
                    GPU_REG_WR32(LW_PTRIM_SYS_TCLKOUT_CTRL_SYS, 0xc104020e);
                    break;
                }
                case 2:
                {
                    GPU_REG_WR32(LW_PTRIM_SYS_TCLKOUT_CTRL_SYS, 0xc104020f);
                    break;
                }
                case 3:
                {
                    GPU_REG_WR32(LW_PTRIM_SYS_TCLKOUT_CTRL_SYS, 0xc1040210);
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
clkNafllGetNafllIdx_GA102
(
    LwU32 nafllId
)
{
    LwU32 idx =0;

    for (idx = 0; idx < LW_ARRAY_ELEMENTS(_nafllMap_GA102); idx++)
    {
        if (_nafllMap_GA102[idx].id == nafllId)
        {
            return idx;
        }
    }
    return CLK_NAFLL_ADDRESS_TABLE_IDX_ILWALID;
}

/*
 * @brief Get REF CLK divider value for the given NAFLL
 *
 * @details This divider value is used to callwlate the input frequency to the
 * NAFLL. The value is usually the same for all the NAFLL devices but we
 * read it from different registers for all the GPC (GPC0/GPC1...) /
 * SYS (SYS/LWD/XBAR/HOST) NAFLLs to make sure we don't make any assumptions.
 *
 * @note    For chips prior to GA102, the input ref clk frequency for NAFLL
 * specified in the VBIOS (@ref inputRefClkFreqMHz) already accounts for the
 * REF_CLK divider hence the div value is just considered as 1 from our pov for
 * these chips. For GA102 and later, the VBIOS input refclk frequency doesn't
 * account for it hence we have to explicitly read it from the register in that
 * case.
 * As an enhancement, we want to move away from getting the input frequency
 * from the VBIOS and move to PMU being self sufficient for reading it all
 * through registers just like what we do for other clock domains.
 *
 * @param[in]  nafllId   ID of the NAFLL for which we want to get the REF CLK
 *                       divider value. @ref LW2080_CTRL_CLK_NAFLL_ID_<xyz>
 *
 * @return REF_CLK divider value
 */
static LwU16
_clkNafllGetRefClkDivider
(
    LwU32          nafllId
)
{
    LwU32   regVal;

    switch(nafllId)
    {
        case LW2080_CTRL_CLK_NAFLL_ID_SYS:
        case LW2080_CTRL_CLK_NAFLL_ID_XBAR:
        case LW2080_CTRL_CLK_NAFLL_ID_LWD:
        case LW2080_CTRL_CLK_NAFLL_ID_HOST:
        {
            regVal = GPU_REG_RD32(LW_PTRIM_SYS_AVFS_REFCLK_CONTROL);
            return ((DRF_VAL(_PTRIM, _SYS_AVFS_REFCLK_CONTROL,
                             _AVFS_REFCLK_DIVIDE_SYS, regVal)) + 1);
            break;
        }
        case LW2080_CTRL_CLK_NAFLL_ID_GPC0:
        case LW2080_CTRL_CLK_NAFLL_ID_GPC1:
        case LW2080_CTRL_CLK_NAFLL_ID_GPC2:
        case LW2080_CTRL_CLK_NAFLL_ID_GPC3:
        case LW2080_CTRL_CLK_NAFLL_ID_GPC4:
        case LW2080_CTRL_CLK_NAFLL_ID_GPC5:
        case LW2080_CTRL_CLK_NAFLL_ID_GPC6:
        case LW2080_CTRL_CLK_NAFLL_ID_GPC7:
        case LW2080_CTRL_CLK_NAFLL_ID_GPCS:
        {
            regVal = GPU_REG_RD32(LW_PTRIM_GPC_BCAST_CLK_SRC_CONTROL);
            return ((DRF_VAL(_PTRIM, _GPC_BCAST_CLK_SRC_CONTROL,
                             _AVFS_REFCLK_DIVIDE_GPC, regVal)) + 1);
            break;
        }
        default:
        {
            dprintf("Unsupported NAFLL ID = %u\n", nafllId);
            return 0U;
        }
    }
}

/*!
 * @brief  Read back NAFLL frequency for a given NAFLL ID
 *
 * @param[in] nafllId   NAFLL index
 *
 * @return NAFLL frequency callwlated from dividers read from hardware
 *         0 otherwise
 *
 */
LwU32 
clkGetNafllFreqKHz_GA102
(
    LwU32 nafllId
)
{
    LwU32 RefFreqKHz = 810000;     // Hardcode RefFreq for now
    LwU32 Mdiv       = 1;
    LwU32 Ndiv       = 0;
    LwU32 reg32      = 0;
    LwU32 dataVal    = 0;
    LwU32 nafllIdx   = 0xFFFFFFFF;
    LwU16 refClkDivVal = 0;

    nafllIdx = pClk[indexGpu].clkNafllGetNafllIdx(nafllId);
    refClkDivVal = _clkNafllGetRefClkDivider(nafllId);

    dprintf("lw: SOURCE: NAFLL\n");
    dprintf("lw:   Ref Source freq = %4d MHz\n", RefFreqKHz / 1000);
    dprintf("lw:   Ref Clock Div = %d\n", refClkDivVal);

    if (refClkDivVal != 0)
    {
        RefFreqKHz /= refClkDivVal;
        dprintf("lw:   Ref freq = %4d MHz\n", RefFreqKHz / 1000);
    }
    else
    {
        dprintf("Invalid Reference Clock Divider!\n");
        return 0;
    }

    // Read back the programmed MDiv
    reg32   = CLK_NAFLL_REG_GET_GA102(nafllIdx, NAFLL_COEFF);
    dataVal = GPU_REG_RD32(reg32);
    Mdiv    = DRF_VAL(_PTRIM, _SYS_NAFLL_SYSNAFLL_COEFF, _MDIV, dataVal);
    dprintf("lw:   Mdiv     = %4d\n", Mdiv);

    if (Mdiv == 0) return 0;

    // Read the NDiv
    reg32   = CLK_NAFLL_REG_GET_GA102(nafllIdx, LUT_STATUS);
    dataVal = GPU_REG_RD32(reg32);
    Ndiv    = DRF_VAL(_PTRIM, _SYS_NAFLL_SYSLUT_STATUS, _NDIV, dataVal);
    dprintf("lw:   Ndiv     = %4d\n", Ndiv);

    // Callwlate back the freq value
    return (RefFreqKHz * Ndiv / Mdiv);
}
