/* _lw_COPYRIGHT_BEGIN_
 *
 * Copyright 2003-2018 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _lw_COPYRIGHT_END_
 */

//*****************************************************
//
// lwwatch WinDbg Extension
// retodd@lwpu.com - 2.1.2002
// clkt124.c
//
//*****************************************************

//
// includes
//
#include "hal.h"
#include "chip.h"
#include "clk.h"

#include "lwrm_drf.h"
#include "cheetah/tegra_access.h"
#include "t21x/t210/arapbpm.h"
#include "t21x/t210/dev_arclk_rst.h"
#include "t21x/t210/dev_trim.h"
#include "t21x/t210/dev_ardisplay.h"
#include "t21x/t210/ardvfs.h"
#include "t21x/t210/dev_arapb_misc.h"
#include "t21x/t210/dev_arflow_ctlr.h"
#include "t21x/t210/dev_arevp.h"
#include "t21x/t210/dev_armc.h"

#include "g_clk_private.h"           // (rmconfig) implementation prototypes.

/*!
 * Define this variable here because HW manual is using the lower case letter for
 * pllC_out0. This is against the normal convention of using upper case letters.
 */
#ifndef LW_PCLK_RST_CONTROLLER_CLK_SOURCE_HDMI_AUDIO_HDMI_AUDIO_CLK_SRC_PLLC_OUT0
#define LW_PCLK_RST_CONTROLLER_CLK_SOURCE_HDMI_AUDIO_HDMI_AUDIO_CLK_SRC_PLLC_OUT0  0x00000001
#endif

//
// Macro for setting a value to the particular offset in the
// clock register
//
#define CLKREG_ADDR(clkModule) LW_PCLK_RST_CONTROLLER_##clkModule

#define SET_CLKREG_OFFSET_VALUE(clkModule, offset, value)      \
    do                                                         \
    {                                                          \
        LwU32 regaddr = LW_PCLK_RST_CONTROLLER_##clkModule;    \
        LwU32 regData = 0;                                     \
        regData = CLK_REG_RD32(regaddr);                       \
        regData = FLD_SET_DRF_NUM(_PCLK_RST_CONTROLLER,        \
                            _##clkModule,                      \
                            _##offset,                         \
                            value,                             \
                            regData);                          \
        CLK_REG_WR32(regaddr, regData);                        \
    } while (0)

typedef struct
{
    LwU32 regAddr;
    LwU32 regValue;
} PLL_FREQ_DETECT_TABLE;

//
// PLL IDs.
//
// Make sure these line up with the order in which
// the pllTable[] entries are laid out.
//
typedef enum
{
    PLL_ID_PLLM_OUT0,
    PLL_ID_PLLM_OUT1,
    PLL_ID_PLLM_OVERRIDE,
    PLL_ID_PLLM_UD,
    PLL_ID_PLLC_OUT0,
    PLL_ID_PLLC_OUT1,
    PLL_ID_PLLC2_OUT0,
    PLL_ID_PLLC3_OUT0,
    PLL_ID_PLLC4_OUT0,
    PLL_ID_PLLE_OUT0,
    PLL_ID_PLLP_OUT0,
    PLL_ID_PLLP_OUT1,
    PLL_ID_PLLP_OUT2,
    PLL_ID_PLLP_OUT3,
    PLL_ID_PLLP_OUT4,
    PLL_ID_PLLA_OUT0,
    PLL_ID_PLLD_OUT0,
    PLL_ID_PLLD2_OUT0,
    PLL_ID_PLLDP_OUT0,
    PLL_ID_PLLX_OUT0,
    PLL_ID_PLLX_OUT0_LJ,
    PLL_ID_DVFS_CPU_CLK,
    PLL_ID_DVFS_CPU_CLK_LJ,
    PLL_ID_CLKM,
    PLL_ID_CLKS,
    PLL_ID_CLKD,
} PLL_ID;

typedef struct
{
    PLL_ID  pllId;
    char   *pName;
    LwU32   freq;
    BOOL    bEnable;
} PLL_TABLE_ENTRY;

//
// PLL group types.
//
typedef enum
{
    PLL_GROUP_TYPE_ACTMON,
    PLL_GROUP_TYPE_AUDIO,
    PLL_GROUP_TYPE_CPU,
    PLL_GROUP_TYPE_CSITE,
    PLL_GROUP_TYPE_DISP1,
    PLL_GROUP_TYPE_EMC,
    PLL_GROUP_TYPE_HDMI_AUDIO,
    PLL_GROUP_TYPE_HOST1X,
    PLL_GROUP_TYPE_I2S0,
    PLL_GROUP_TYPE_ISP,
    PLL_GROUP_TYPE_SCLK,
    PLL_GROUP_TYPE_SDMMC1,
    PLL_GROUP_TYPE_SE,
    PLL_GROUP_TYPE_VI,
    PLL_GROUP_TYPE_VIC,
    PLL_GROUP_TYPE_MSELECT,
} PLL_GROUP_TYPE;

//
// Structure that maps a <PLL_GROUP_TYPE,clkSrc> pair to corresponding
// PLL_ID.
//
typedef struct
{
    PLL_GROUP_TYPE pllGroupType;
    LwU32          clkSrc;
    PLL_ID         pllId;
} PLL_GROUP_TYPE_SRC_INFO;

//
// Macros that generate PLL_GROUP_TYPE_SRC_INFO table entries.
//
#define PLL_GROUP_TYPE_SRC_INFO(gid,src)                                \
    { PLL_GROUP_TYPE_##gid,                                             \
      LW_PCLK_RST_CONTROLLER_CLK_SOURCE_##gid##_##gid##_CLK_SRC_##src,  \
      PLL_ID_##src }

#define PLL_GROUP_TYPE_SRC_INFO_WEIRD(gid,src,id)                       \
    { PLL_GROUP_TYPE_##gid,                                             \
      LW_PCLK_RST_CONTROLLER_CLK_SOURCE_##gid##_##gid##_CLK_SRC_##src,  \
      PLL_ID_##id }

#define CPU_GROUP_TYPE_SRC_INFO(src)                                    \
    { PLL_GROUP_TYPE_CPU,                                               \
      LW_PCLK_RST_CONTROLLER_CCLK_BURST_POLICY_CWAKEUP_FIQ_SOURCE_##src,\
      PLL_ID_##src }

#define SCLK_GROUP_TYPE_SRC_INFO(src)                                   \
    { PLL_GROUP_TYPE_SCLK,                                              \
      LW_PCLK_RST_CONTROLLER_SCLK_BURST_POLICY_SWAKEUP_FIQ_SOURCE_##src,\
      PLL_ID_##src }

#define EMC_GROUP_TYPE_SRC_INFO(src)                                    \
    { PLL_GROUP_TYPE_EMC,                                               \
      LW_PCLK_RST_CONTROLLER_CLK_SOURCE_EMC_EMC_2X_CLK_SRC_##src,       \
      PLL_ID_##src }

#define EMC_GROUP_TYPE_SRC_INFO_WEIRD(src,id)                           \
    { PLL_GROUP_TYPE_EMC,                                               \
      LW_PCLK_RST_CONTROLLER_CLK_SOURCE_EMC_EMC_2X_CLK_SRC_##src,       \
      PLL_ID_##id }

//
// Clocks.
//
typedef enum
{
    CLK_ID_ACTMON,
    CLK_ID_AUDIO,
    CLK_ID_CPU,
    CLK_ID_CSITE,
    CLK_ID_DAM0,
    CLK_ID_DAM1,
    CLK_ID_DAM2,
    CLK_ID_DISP1,
    CLK_ID_DISP2,
    CLK_ID_EMC,
    CLK_ID_EMC_DLL,
    CLK_ID_HDA,
    CLK_ID_HDA2CODEC_2X,
    CLK_ID_HDMI,
    CLK_ID_HDMI_AUDIO,
    CLK_ID_HOST1X,
    CLK_ID_I2C1,
    CLK_ID_I2C2,
    CLK_ID_I2C3,
    CLK_ID_I2C4,
    CLK_ID_I2C5,
    CLK_ID_I2C6,
    CLK_ID_I2C_SLOW,
    CLK_ID_ISP,
    CLK_ID_LA,
    CLK_ID_MSENC,
    CLK_ID_PWM,
    CLK_ID_SCLK,    // aka COP aka AVP
    CLK_ID_SDMMC1,
    CLK_ID_SDMMC2,
    CLK_ID_SDMMC3,
    CLK_ID_SDMMC4,
    CLK_ID_SE,
    CLK_ID_SPI1,
    CLK_ID_SPI2,
    CLK_ID_SPI3,
    CLK_ID_SPI4,
    CLK_ID_SPI5,
    CLK_ID_SPI6,
    CLK_ID_SOR0,
    CLK_ID_TSEC,
    CLK_ID_UARTA,
    CLK_ID_UARTB,
    CLK_ID_UARTC,
    CLK_ID_UARTD,
    CLK_ID_I2S0,
    CLK_ID_I2S1,
    CLK_ID_I2S2,
    CLK_ID_I2S3,
    CLK_ID_I2S4,
    CLK_ID_VDE,
    CLK_ID_VI,
    CLK_ID_VIC,
    CLK_ID_VI_SENSOR,
    CLK_ID_MSELECT,
} CLK_ID;

//
// Clock divider groups.
//
typedef enum
{
    CLK_DIVIDER_GROUP_TYPE_CPU,
    CLK_DIVIDER_GROUP_TYPE_SCLK,
    CLK_DIVIDER_GROUP_TYPE_LWSTOM,
    CLK_DIVIDER_GROUP_TYPE_EMC,
    CLK_DIVIDER_GROUP_TYPE_DISP,
    CLK_DIVIDER_GROUP_TYPE_U71,
    CLK_DIVIDER_GROUP_TYPE_UART
} CLK_DIVIDER_GROUP_TYPE;

//
// Clock enables groups.
//
typedef enum
{
    CLK_ENB_L,
    CLK_ENB_H,
    CLK_ENB_U,
    CLK_ENB_V,
    CLK_ENB_X,
} CLK_ENB_GROUP_TYPE;

struct CLK_ENB_REG_INFO
{
    LwU32 regOffset;
    LwU32 value;
};

//
// Clock enables groups.
//
typedef enum
{
    CLK_RST_L,
    CLK_RST_H,
    CLK_RST_U,
    CLK_RST_V,
    CLK_RST_X,
    CLK_RST_TOTAL
} CLK_RST_GROUP_TYPE;


struct CLK_RST_REG_INFO
{
    LwU32 regOffset;
    LwU32 value;
};

struct cpuState
{
    LwU32 state;
    char *name;
};

static struct cpuState cpuStateTable[] =
{
    { LW_PCLK_RST_CONTROLLER_CCLK_BURST_POLICY_CPU_STATE_STDBY, "STANDBY", },
    { LW_PCLK_RST_CONTROLLER_CCLK_BURST_POLICY_CPU_STATE_IDLE,  "IDLE",    },
    { LW_PCLK_RST_CONTROLLER_CCLK_BURST_POLICY_CPU_STATE_RUN,   "RUN",     },
    { LW_PCLK_RST_CONTROLLER_CCLK_BURST_POLICY_CPU_STATE_IRQ,   "IRQ",     },
    { LW_PCLK_RST_CONTROLLER_CCLK_BURST_POLICY_CPU_STATE_FIQ,   "FIQ",     },
};

static LwU32 cpuStateTableSize = sizeof(cpuStateTable) / sizeof(cpuStateTable[0]);

typedef struct
{
    CLK_ID                  id;
    char                   *pName;
    PLL_GROUP_TYPE          pllGrp;
    LwU32                   regOffset;
    PLL_ID                  src;
    CLK_ENB_GROUP_TYPE      enbGrp;
    LwU32                   enbBit;
    CLK_RST_GROUP_TYPE      rstGrp;
    LwU32                   rstBit;
    CLK_DIVIDER_GROUP_TYPE  divGrp;
} CLK_INFO;

// colwenience macros
#define CT_ENTRY(clk,nm,pllgrp,enbgrp,enbbit,rstgrp,rstbit,divgrp)       \
 { CLK_ID_##clk,                                                         \
   nm,                                                                   \
   PLL_GROUP_TYPE_##pllgrp,                                              \
   LW_PCLK_RST_CONTROLLER_CLK_SOURCE_##clk,                              \
   0,                                                                    \
   CLK_ENB_##enbgrp,                                                     \
   DRF_SHIFTMASK(LW_PCLK_RST_CONTROLLER_CLK_OUT_ENB_##enbgrp##_CLK_ENB_##enbbit), \
   CLK_RST_##rstgrp,                                                     \
   DRF_SHIFTMASK(LW_PCLK_RST_CONTROLLER_RST_DEVICES_##enbgrp##_SWR_##rstbit##_RST), \
   CLK_DIVIDER_GROUP_TYPE_##divgrp, }

// colwenience macros
#define CT_ENTRY_NR(clk,nm,pllgrp,enbgrp,enbbit,divgrp)       \
 { CLK_ID_##clk,                                                         \
   nm,                                                                   \
   PLL_GROUP_TYPE_##pllgrp,                                              \
   LW_PCLK_RST_CONTROLLER_CLK_SOURCE_##clk,                              \
   0,                                                                    \
   CLK_ENB_##enbgrp,                                                     \
   DRF_SHIFTMASK(LW_PCLK_RST_CONTROLLER_CLK_OUT_ENB_##enbgrp##_CLK_ENB_##enbbit), \
   CLK_RST_TOTAL,                                                        \
   0,                                                                    \
   CLK_DIVIDER_GROUP_TYPE_##divgrp, }



#define CPU_ENTRY(clk,nm,pllgrp,cpunm)                                   \
 { CLK_ID_##clk, nm, PLL_GROUP_TYPE_##pllgrp,                            \
   LW_PCLK_RST_CONTROLLER_##cpunm##_BURST_POLICY,                        \
   0, 0, 0,                                                              \
   CLK_DIVIDER_GROUP_TYPE_CPU, }

// Used for GPU-PLL since the pldiv reg values and divby values differ
static LwU32 plDivVal[] =
{
  1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 12, 16, 20, 24, 32,
};

static LwU32 _clkMeasureGpc2ClkFrequency_T124();
static LwU32 _clkMeasurePwrClkFrequency_T124();
static LwU32 _clkMeasureSocFreq_T124(LwU32 clkSel);

//
// This routine returns the oscillator (aka clk_m) frequency.
//
static LwU32
getOscFreq_T124(void)
{
    LwU32 regData;
    LwU32 inClkKhz;
    LwU32 timeout;

    //
    // Setup osc clock detection.
    //
    regData = (DRF_DEF(_PCLK_RST_CONTROLLER, _OSC_FREQ_DET,
                          _OSC_FREQ_DET_TRIG, _ENABLE) |
               DRF_NUM(_PCLK_RST_CONTROLLER, _OSC_FREQ_DET,
                          _REF_CLK_WIN_CFG, 0x1));

    CLK_REG_WR32(LW_PCLK_RST_CONTROLLER_OSC_FREQ_DET, regData);

    timeout = 0;
    do {

        //
        // Not sure what do if we timeout here ... for now we'll
        // use a frequency of 26MHz.
        //
        if (timeout == 1000000)
        {
            dprintf("timeout during osc check\n");
            break;
        }

        osPerfDelay(20);        // delay 20 us
        timeout += 20;
        regData = CLK_REG_RD32(LW_PCLK_RST_CONTROLLER_OSC_FREQ_DET_STATUS);
    } while (regData & (DRF_NUM(_PCLK_RST_CONTROLLER, _OSC_FREQ_DET_STATUS,
                                _OSC_FREQ_DET_BUSY, 0x1)));

    if (regData < 740)
        inClkKhz = 12000;       // 12MHz
    else if (regData < 800)
        inClkKhz = 13000;       // 13MHz
    else if (regData < 1200)
        inClkKhz = 19200;       // 19.2MHz
    else
        inClkKhz = 26000;       // 26MHz

    return inClkKhz;
}

//
// PLL group types - Possible clock sources for each clock source type
//
//    PLL_GROUP_TYPE_ACTMON/I2C_SLOW/PWM use the same clock sources
//    ACTMON    : PLLP_OUT0 (0), PLLC2_OUT0 (1), PLLC_OUT0 (2), PLLC3_OUT0 (3),
//                CLK_S (4), CLK_M (6)
//
//    PLL_GROUP_TYPE_AUDIO/DAM0/DAM1/DAM2 use the same clock sources
//    AUDIO     : PLLA_OUT0 (0), PLLC2_OUT0 (1), PLLC_OUT0 (2),
//                PLLC3_OUT0 (3), PLLP_OUT0 (4), CLK_M (6), CLK_SRC_ALT (7)
//
//    PLL_GROUP_TYPE_CPU
//    CPU       : CLKM (0), PLLC_OUT0 (1), CLKS (2), PLLM_OUT0 (3), PLLP_OUT0 (4),
//                PLLP_OUT4 (5), PLLC2_OUT0 (6), PLLC3_OUT0 (7), PLLX_OUT0_LJ (8),
//                RESERVED9 (9), PLLX_OUT0 (10), RESERVED15 (15)
//
//    PLL_GROUP_TYPE_CSITE/LA/UARTA/UARTB/UARTC/UARTD
//    VDE/I2C(1-5)/SPI(1-5) - use the same clock sources
//    CSITE     : PLLP_OUT0 (0), PLLC2_OUT0 (1), PLLC_OUT0 (2), PLLC3_OUT0 (3),
//                PLLM_OUT0 (4), CLK_M (6)
//
//    PLL_GROUP_TYPE_SDMMC1/2/3/4 use the same clock sources
//    SDMMC1    : PLLP_OUT0 (0), PLLC2_OUT0 (1), PLLC_OUT0 (2), PLLC3_OUT0 (3),
//                PLLM_OUT0 (4), PLLE_OUT0(5), CLK_M (6)
//
//    PLL_GROUP_TYPE_SE/TSEC use the same clock sources
//    SE        : PLLP_OUT0 (0), PLLC2_OUT0 (1), PLLC_OUT0 (2), PLLC3_OUT0 (3),
//                PLLM_OUT0 (4), PLLA_OUT0(5), CLK_M (6)
//
//    PLL_GROUP_TYPE_DISP1/DISP2/HDMI/SOR0 use the same clock sources
//    DISP1     : PLLP_OUT0 (0), PLLM_OUT0 (1), PLLD_OUT0 (2), PLLA_OUT0 (3),
//                PLLC_OUT0 (4), PLLD2_OUT0 (5), CLK_M (6)
//
//    PLL_GROUP_TYPE_EMC
//    EMC       : PLLM_OUT0 (0), PLLC_OUT0 (1), PLLP_OUT0 (2), CLK_M (3),
//                PLLM_UD (4), PLLC2_OUT0 (5), PLLC3_OUT0 (6)
//
//    PLL_GROUP_TYPE_HOST1X/MSENC/VI_SENSOR use the same clock sources
//    HOST1X    : PLLM_OUT0 (0), PLLC2_OUT0 (1), PLLC_OUT0 (2), PLLC3_OUT0 (3),
//                PLLP_OUT0(4), PLLA_OUT0(6)
//    MSENC     : PLLM_OUT0 (0), PLLC2_OUT0 (1), PLLC_OUT0 (2), PLLC3_OUT0 (3),
//                PLLP_OUT0(4), PLLA_OUT0(6)
//    VI_SENSOR : PLLM_OUT0 (0), PLLC2_OUT0 (1), PLLC_OUT0 (2), PLLC3_OUT0 (3),
//                PLLP_OUT0(4), PLLA_OUT0(6)
//
//    PLL_GROUP_TYPE_ISP/VI use the same clock sources
//    HOST1X    : PLLM_OUT0 (0), PLLC2_OUT0 (1), PLLC_OUT0 (2), PLLC3_OUT0 (3),
//                PLLP_OUT0(4), PLLA_OUT0(6), PLLC4_OUT0(7)
//
//    PLL_GROUP_TYPE_TSENSOR
//    TSENSOR   : PLLP_OUT0 (0), PLLC2_OUT0 (1), PLLC_OUT0 (2),
//                PLLC3_OUT0 (4), CLK_M (5), CLK_S (6)
//
//    PLL_GROUP_TYPE_VI
//    VI        : PLLM_OUT0 (0), PLLC2_OUT0 (1), PLLC_OUT0 (2), PLLC3_OUT0 (3),
//                PLLP_OUT0 (4), PLLA_OUT0 (6), PLLC4_OUT0 (7)
//
//    PLL_GROUP_TYPE_I2S0/1/2/3/4 use the same clock sources
//    I2S0      : PLLA_OUT0 (0), SYNC_CLK (1), PLLP_OUT0 (2), CLK_M (4)
//
//    PLL_GROUP_TYPE_SYSTEM
//    SYSTEM    : CLKM (0), PLLC_OUT1 (1), PLLP_OUT4 (2), PLLP_OUT3 (3),
//                PLLP_OUT2 (4), CLKD (5), CLKS (6), PLLM_OUT1 (7)
//
//    PLL_GROUP_TYPE_THERM
//    SOC_THERM : PLLM_OUT0 (0), PLLC_OUT0 (1), PLLP_OUT0 (2), PLLA_OUT0 (3),
//                PLLC2_OUT0 (4), PLLC3_OUT0 (5)
//
//    PLL_GROUP_TYPE_XUSBH/D/F use the same clock sources
//    XUSBH : CLKM (0), PLLP_OUT0 (1), PLLC2_OUT0 (2), PLLC_OUT0 (3),
//            PLLC3_OUT0 (4), PLLREFE_OUT0 (5)
//
//    PLL_GROUP_TYPE_XUSBFS
//    XUSBFS : CLKM (0), FO_48M (1), PLLP_OUT0 (2), HSIC_480 (3)
//
//    PLL_GROUP_TYPE_XUSBSS
//    XUSBSS : CLKM (0), PLLREFE_OUT (1), CLKS (2), HSIC_480 (3),
//             PLLC_OUT0 (4), PLLC2_OUT0 (5), PLLC3_OUT0 (6), OSC_DIV (7)
//

char *cpuGetStateName_T124(LwU32 state)
{
    LwU32 i;
    char *name = "UNKNOWN";

    for (i = 0; i < cpuStateTableSize; i++)
    {
        if (cpuStateTable[i].state == state)
        {
            name = cpuStateTable[i].name;
            break;
        }
    }

    return name;
}

/**
 * @brief Measures frequency of desired clk using clk counters in SoC
 *
 * @returns Frequency in KHz
 */
static LwU32
_clkMeasureSocFreq_T124(LwU32 clkSel)
{
    LwU32 regData, i, j, freqKHz = 0;

    //
    // Some clks like CIL*, DISPLAY, PEX are gated at the root from coming to
    // this counter to save power on these paths. To measure these clocks, we
    // must first enable the clocks.
    //
    SET_CLKREG_OFFSET_VALUE(PTO_CLK_CNT_CNTL, PTO_CLK_ENABLE, 0x1);

    // Setup PTO_CLK_CNT_CNTL
    SET_CLKREG_OFFSET_VALUE(PTO_CLK_CNT_CNTL, ANALOG_PAD_AND_PTO_SRC_SEL, clkSel);
    SET_CLKREG_OFFSET_VALUE(PTO_CLK_CNT_CNTL, PTO_REF_CLK_WIN_CFG, 0x1);
    SET_CLKREG_OFFSET_VALUE(PTO_CLK_CNT_CNTL, PTO_DIV_SEL, 0x1);

    // Enable reset
    SET_CLKREG_OFFSET_VALUE(PTO_CLK_CNT_CNTL, PTO_CNT_RST, 0x1);

    // Disable reset
    SET_CLKREG_OFFSET_VALUE(PTO_CLK_CNT_CNTL, PTO_CNT_RST, 0);

    // Enable counter
    SET_CLKREG_OFFSET_VALUE(PTO_CLK_CNT_CNTL, PTO_CNT_EN, 0x1);
    //dprintf("lw: PCLK_RST_CONTROLLER_PTO_CLK_CNT_CNTL:      0x%x\n", regData);

    // Wait until finished
    do
    {
        regData = CLK_REG_RD32(LW_PCLK_RST_CONTROLLER_PTO_CLK_CNT_STATUS);

    } while (FLD_TEST_DRF_NUM(_PCLK_RST_CONTROLLER,
                                _PTO_CLK_CNT_STATUS,
                                _PTO_CLK_CNT_BUSY,
                                0x1, regData));
    dprintf("lw: LW_PCLK_RST_CONTROLLER_PTO_CLK_CNT_STATUS: 0x%x\n", regData);

    i = DRF_VAL(_PCLK_RST_CONTROLLER, _PTO_CLK_CNT_STATUS, _PTO_CLK_CNT, regData);
    dprintf("lw:   _PTO_CLK_CNT:                            0x%x\n", i);

    regData = CLK_REG_RD32(LW_PCLK_RST_CONTROLLER_PTO_CLK_CNT_CNTL);

    // Disable counter
    SET_CLKREG_OFFSET_VALUE(PTO_CLK_CNT_CNTL, PTO_CNT_EN, 0);

    // Get the window width
    j = DRF_VAL(_PCLK_RST_CONTROLLER, _PTO_CLK_CNT_CNTL, _PTO_REF_CLK_WIN_CFG, regData) + 1;
    //dprintf("lw:   _PTO_REF_CLK_WIN_CFG:                            0x%x\n", j);

    freqKHz = (32768 * i)/ (j*1000);
    dprintf("lw: clkSel: %d - freqKHz: %d\n", clkSel, freqKHz);

    // Set the clocks input to the counter back to gated state to save power.
    SET_CLKREG_OFFSET_VALUE(PTO_CLK_CNT_CNTL, PTO_CLK_ENABLE, 0x0);

    return freqKHz;
}

/**
 * @brief Reads GPU clocks
 *
 * @returns LW_OK
 */
LW_STATUS
clkGetGpuClocks_T124()
{
    //
    // T124 has a single gpu clock i.e. GPC2CLK which should be sourced from GPCPLL
    // Programmed frequencies.
    //
    dprintf("\n\nGPU CLOCK FREQUENCY DETAILS\n");
    dprintf("lw: Programmed frequencies are:\n");
    dprintf("lw: Gpc2Clk  = %4d KHz\n\n", pClk[indexGpu].clkGetGpc2ClkFreqKHz());
    dprintf("\n");
    dprintf("lw: Measured frequencies are:\n");
    dprintf("lw: Gpc2Clk  = %4d KHz\n\n", _clkMeasureGpc2ClkFrequency_T124());
    dprintf("lw: PwrClk   = %4d KHz\n\n", _clkMeasurePwrClkFrequency_T124());
    dprintf("lw: Crystal  = %4d KHz\n\n", pClk[indexGpu].clkGetClkSrcFreqKHz(clkSrcWhich_XTAL));
    return LW_OK;
}

/**
 * @brief Given the register value for PLDiv, this function finds
 * the divby value for PLDIV.
 *
 * Register value is the index in plDiv table, value at the index
 * is the divby value it corresponds to. While programming the register
 * we need to use the index (register value), while using it for computing
 * frequency, we need to use the divby value.
 *
 * @param[in/out]   *PL  PLDIV register_value/divby_value
 *
 * @returns LW_OK when valid and LW_ERR_GENERIC when invalid
 */
LW_STATUS
clkGetPLValue_T124(LwU32 *PL)
{
    LwU32 i;
    i = *PL;
    if (i >= LW_ARRAY_ELEMENTS(plDivVal))
    {
        dprintf( "lw:   %s: Invalid PL value 0x%x\n", __FUNCTION__, *PL);
        return LW_ERR_GENERIC;
    }
    *PL = plDivVal[i];
    return LW_OK;
}

/**
 * @brief Returns the ref clk src of passed in PLL
 *
 * @param[in]   pllNameMapIndex  PLL namemapindex
 *
 * @returns  Invalid or XTAl as the clk src
 */
CLKSRCWHICH
clkReadRefClockSrc_T124(LwU32 pllNameMapIndex)
{
    // We shouldn't be getting this for anything other than GPCPLL
    if (LW_PTRIM_PLL_NAMEMAP_INDEX_GPCPLL != pllNameMapIndex)
    {
        dprintf( "lw:   %s: Unsupported PLL(%d) Option\n", __FUNCTION__, pllNameMapIndex);
        return clkSrcWhich_Ilwalid;
    }
    return clkSrcWhich_XTAL;
}

/**
 * @brief Returns the alt clk src when PLL is bypassed
 *
 * @param[in]  clkNameMapIndex clkName
 *
 * @returns  Invalid or XTAl as the clk src
 */
CLKSRCWHICH
clkReadAltClockSrc_T124(LwU32 clkNameMapIndex)
{
    // We shouldn't be getting this for anything other than GPC2CLK
    if (LW_PTRIM_CLK_NAMEMAP_INDEX_GPC2CLK != clkNameMapIndex)
    {
        dprintf( "lw:   %s: Unsupported clk 0x%x\n", __FUNCTION__, clkNameMapIndex);
        return clkSrcWhich_Ilwalid;
    }
    return clkSrcWhich_XTAL;
}

/**
 * @brief Provides the frequency at which the given clk src runs at.
 *
 * @param[in] whichClkSrc CLK source to be read
 *
 * @returns Frequency in KHz, 0 on error
 */
LwU32 clkGetClkSrcFreqKHz_T124(CLKSRCWHICH whichClkSrc)
{
    LwU32 freqKHz = 0, data;

    switch (whichClkSrc)
    {
        case clkSrcWhich_XTAL:
            data = CLK_REG_RD32(LW_PCLK_RST_CONTROLLER_OSC_CTRL);
            switch (DRF_VAL(_PCLK_RST_CONTROLLER, _OSC_CTRL, _OSC_FREQ, data))
            {
                case LW_PCLK_RST_CONTROLLER_OSC_CTRL_OSC_FREQ_OSC13:
                {
                    freqKHz = 13000;       // 13 MHz
                    break;
                }
                case LW_PCLK_RST_CONTROLLER_OSC_CTRL_OSC_FREQ_OSC19P2:
                {
                    freqKHz = 19200;       // 19.2 MHz
                    break;
                }
                case LW_PCLK_RST_CONTROLLER_OSC_CTRL_OSC_FREQ_OSC12:
                {
                    freqKHz = 12000;       // 12 MHz
                    break;
                }
                case LW_PCLK_RST_CONTROLLER_OSC_CTRL_OSC_FREQ_OSC26:
                {
                    freqKHz = 26000;       // 26 MHz
                    break;
                }
                case LW_PCLK_RST_CONTROLLER_OSC_CTRL_OSC_FREQ_OSC16P8:
                {
                    freqKHz = 16800;       // 16.8 MHz
                    break;
                }
                case LW_PCLK_RST_CONTROLLER_OSC_CTRL_OSC_FREQ_OSC38P4:
                {
                    freqKHz = 38400;       // 38.4 MHz
                    break;
                }
                case LW_PCLK_RST_CONTROLLER_OSC_CTRL_OSC_FREQ_OSC48:
                {
                    freqKHz = 48000;       // 48 MHz
                    break;
                }
                default:
                {
            dprintf( "lw:   %s: Unsupported frequency, using 12MHz\n", __FUNCTION__);
                    freqKHz = 12000;       // Let's use 12MHz as default.
                    break;
                }
            }
            break;
        case clkSrcWhich_GPCPLL:
            freqKHz = clkGetClkSrcFreqKHz_GF100(whichClkSrc);
            break;
        default:
            dprintf( "lw:   %s: Unsupported clk source 0x%x\n", __FUNCTION__, whichClkSrc);
            break;
    }
    return freqKHz;
}

/**
 * @brief Measures gpc2clk frequency using gpu clk counters
 *
 * @returns Frequency in KHz
 */
static LwU32
_clkMeasureGpc2ClkFrequency_T124()
{
    DEV_REG_WR32(LW_PTRIM_SYS_PLLS_OUT, LW_PTRIM_SYS_PLLS_OUT_PLLS_O_SRC_SELECT_GPCPLL00_SYNCCLOCKOUT, "GPU", 0);
    osPerfDelay(100);
    // Measure using SoC clock counters.
    return _clkMeasureSocFreq_T124(LW_PCLK_RST_CONTROLLER_PTO_CLK_CNT_CNTL_ANALOG_PAD_AND_PTO_SRC_SEL_PEX_PAD_TCLKOUT_IN);
}

/**
 * @brief Measures pwrclk frequency using SoC clk counters
 *
 * @returns Frequency in KHz
 */
static LwU32
_clkMeasurePwrClkFrequency_T124()
{
    DEV_REG_WR32(LW_PTRIM_SYS_PLLS_OUT, LW_PTRIM_SYS_PLLS_OUT_PLLS_O_SRC_SELECT_PWRCLK, "GPU", 0);
    // Measure using SoC clock counters.
    return _clkMeasureSocFreq_T124(LW_PCLK_RST_CONTROLLER_PTO_CLK_CNT_CNTL_ANALOG_PAD_AND_PTO_SRC_SEL_PEX_PAD_TCLKOUT_IN);
}

/**
 * @brief Reads back the Oscillator Frequency
 *
 * @returns Frequency in KHz
 */
LwU32
clkReadCrystalFreqKHz_T124()
{
    return getOscFreq_T124();
}

