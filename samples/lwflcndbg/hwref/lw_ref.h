/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 1993-2001 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

//
/***************************************************************************\
*                                                                           *
*               Hardware Reference Manual extracted defines.                *
*                                                                           *
\***************************************************************************/
#ifndef _LW_REF_H_
#define _LW_REF_H_

//
// These registers can be accessed by chip-independent code as
// well as chip-dependent code.
//



// used by diag.c
#define LW_PMC_ENABLE                                    0x00000200 /* RW-4R */
#define LW_PMC_ENABLE_BUF_RESET                                 0:0 /* RWIVF */
#define LW_PMC_ENABLE_BUF_RESET_DISABLE                  0x00000000 /* RWI-V */
#define LW_PMC_ENABLE_BUF_RESET_ENABLE                   0x00000001 /* RW--V */
#define LW_PMC_ENABLE_MD_RESET                                  1:1 /* RWIVF */
#define LW_PMC_ENABLE_MD_RESET_DISABLE                   0x00000000 /* RWI-V */
#define LW_PMC_ENABLE_MD_RESET_ENABLE                    0x00000001 /* RW--V */
#define LW_PMC_ENABLE_PMPEG                                     1:1 /* RWIVF */
#define LW_PMC_ENABLE_PMPEG_DISABLED                     0x00000000 /* RWI-V */
#define LW_PMC_ENABLE_PMPEG_ENABLED                      0x00000001 /* RW--V */
#define LW_PMC_ENABLE_PMEDIA                                    4:4 /* RWIVF */
#define LW_PMC_ENABLE_PMEDIA_DISABLED                    0x00000000 /* RWI-V */
#define LW_PMC_ENABLE_PMEDIA_ENABLED                     0x00000001 /* RW--V */
#define LW_PMC_ENABLE_PFIFO                                     8:8 /* RWIVF */
#define LW_PMC_ENABLE_PFIFO_DISABLED                     0x00000000 /* RWI-V */
#define LW_PMC_ENABLE_PFIFO_ENABLED                      0x00000001 /* RW--V */
#define LW_PMC_ENABLE_PGRAPH                                  12:12 /* RWIVF */
#define LW_PMC_ENABLE_PGRAPH_DISABLED                    0x00000000 /* RWI-V */
#define LW_PMC_ENABLE_PGRAPH_ENABLED                     0x00000001 /* RW--V */
#define LW_PMC_ENABLE_PPMI                                    16:16 /* RWIVF */
#define LW_PMC_ENABLE_PPMI_DISABLED                      0x00000000 /* RWI-V */
#define LW_PMC_ENABLE_PPMI_ENABLED                       0x00000001 /* RW--V */
#define LW_PMC_ENABLE_PFB                                     20:20 /* RWIVF */
#define LW_PMC_ENABLE_PFB_DISABLED                       0x00000000 /* RW--V */
#define LW_PMC_ENABLE_PFB_ENABLED                        0x00000001 /* RWI-V */
#define LW_PMC_ENABLE_PCRTC                                   24:24 /* RWIVF */
#define LW_PMC_ENABLE_PCRTC_DISABLED                     0x00000000 /* RW--V */
#define LW_PMC_ENABLE_PCRTC_ENABLED                      0x00000001 /* RWI-V */
#define LW_PMC_ENABLE_PCRTC2                                  25:25 /* RWIVF */
#define LW_PMC_ENABLE_PCRTC2_DISABLED                    0x00000000 /* RW--V */
#define LW_PMC_ENABLE_PCRTC2_ENABLED                     0x00000001 /* RWI-V */
#define LW_PMC_ENABLE_TVO                                     26:26 /* RWIVF */
#define LW_PMC_ENABLE_TVO_DISABLED                       0x00000000 /* RWI-V */
#define LW_PMC_ENABLE_TVO_ENABLED                        0x00000001 /* RW--V */
#define LW_PMC_ENABLE_PVIDEO                                  28:28 /* RWIVF */
#define LW_PMC_ENABLE_PVIDEO_DISABLED                    0x00000000 /* RWI-V */
#define LW_PMC_ENABLE_PVIDEO_ENABLED                     0x00000001 /* RW--V */

// used by os.c
#define LW_PEXTDEV_BOOT_0                                0x00101000 /* R--4R */
#define LW_PEXTDEV_BOOT_0_STRAP_CRYSTAL                         6:6 /* R-XVF */
#define LW_PEXTDEV_BOOT_0_STRAP_CRYSTAL_13500K           0x00000000 /* R---V */
#define LW_PEXTDEV_BOOT_0_STRAP_CRYSTAL_14318180         0x00000001 /* R---V */
#define LW_PEXTDEV_BOOT_0_STRAP_CRYSTAL0                        6:6 /* R-XVF */
#define LW_PEXTDEV_BOOT_0_STRAP_CRYSTAL0_27000K          0x00000000 /* R---V */
#define LW_PEXTDEV_BOOT_0_STRAP_CRYSTAL0_13500K          0x00000000 /* R---V */
#define LW_PEXTDEV_BOOT_0_STRAP_CRYSTAL0_14318K          0x00000001 /* R---V */
#define LW_PEXTDEV_BOOT_0_STRAP_CRYSTAL0_UNKWN           0x00000001 /* R---V */
#define LW_PEXTDEV_BOOT_0_STRAP_CRYSTAL1                      22:22 /* R-XVF */
#define LW_PEXTDEV_BOOT_0_STRAP_CRYSTAL1_14318_OR_13500  0x00000000 /* R---V */
#define LW_PEXTDEV_BOOT_0_STRAP_CRYSTAL1_13500K          0x00000000 /* R---V */
#define LW_PEXTDEV_BOOT_0_STRAP_CRYSTAL1_14318K          0x00000000 /* R---V */
#define LW_PEXTDEV_BOOT_0_STRAP_CRYSTAL1_13500K          0x00000000 /* R---V */
#define LW_PEXTDEV_BOOT_0_STRAP_CRYSTAL1_27000K          0x00000001 /* R---V */
#define LW_PEXTDEV_BOOT_0_STRAP_CRYSTAL1_UNKWN           0x00000001 /* R---V */

// used by os.c
#define LW_PBUS_DEBUG_1                                  0x00001084 /* RW-4R */
#define LW_PBUS_DEBUG_1_DISP_MIRROR                           28:28 /* RWIVF */
#define LW_PBUS_DEBUG_1_DISP_MIRROR_DISABLE              0x00000000 /* RWI-V */
#define LW_PBUS_DEBUG_1_DISP_MIRROR_ENABLE               0x00000001 /* RW--V */
#define LW_PBUS_PCI_LW_18                                0x00001848 /* RW-4R */
#define LW_PBUS_PCI_LW_18_AGP_STATUS_RATE                       2:0 /* R--VF */
#define LW_PBUS_PCI_LW_18_AGP_STATUS_RATE_1X             0x00000001 /* R---V */
#define LW_PBUS_PCI_LW_18_AGP_STATUS_RATE_2X             0x00000002 /* ----V */
#define LW_PBUS_PCI_LW_18_AGP_STATUS_RATE_1X_AND_2X      0x00000003 /* R---V */
#define LW_PBUS_PCI_LW_18_AGP_STATUS_RATE_4X             0x00000004 /* R---V */
#define LW_PBUS_PCI_LW_18_AGP_STATUS_RATE_1X_2X_4X       0x00000007 /* R---V */
#define LW_PBUS_PCI_LW_18_AGP_STATUS_RATE_A3_4X          0x00000001 /* R---V */
#define LW_PBUS_PCI_LW_18_AGP_STATUS_RATE_A3_8X          0x00000002 /* R---V */
#define LW_PBUS_PCI_LW_18_AGP_STATUS_RATE_A3_4X_AND_8X   0x00000003 /* R---V */
#define LW_PBUS_PCI_LW_18_AGP_STATUS_AGP3MODE                   3:3 /* R--VF */
#define LW_PBUS_PCI_LW_18_AGP_STATUS_AGP3MODE_DISABLED   0x00000000 /* R---V */
#define LW_PBUS_PCI_LW_18_AGP_STATUS_AGP3MODE_ENABLED    0x00000001 /* R---V */
#define LW_PBUS_PCI_LW_19                                0x0000184C /* RW-4R */
#define LW_PBUS_PCI_LW_19_AGP_COMMAND_DATA_RATE                 2:0 /* RWIVF */
#define LW_PBUS_PCI_LW_19_AGP_COMMAND_DATA_RATE_OFF      0x00000000 /* RWI-V */
#define LW_PBUS_PCI_LW_19_AGP_COMMAND_DATA_RATE_1X       0x00000001 /* RW--V */
#define LW_PBUS_PCI_LW_19_AGP_COMMAND_DATA_RATE_2X       0x00000002 /* RW--V */
#define LW_PBUS_PCI_LW_19_AGP_COMMAND_DATA_RATE_4X       0x00000004 /* RW--V */
#define LW_PBUS_PCI_LW_19_AGP_COMMAND_DATA_RATE_A3_4X    0x00000001 /* RW--V */
#define LW_PBUS_PCI_LW_19_AGP_COMMAND_DATA_RATE_A3_8X    0x00000002 /* RW--V */
#define LW_PBUS_PCI_LW_5                                 0x00001814 /* RW-4R */

// used by dac.c
#define LW_PRAMDAC_PLL_COMPAT                            0x00680528 /* RW-4R */
#define LW_PRAMDAC_PLL_COMPAT_LWPLL_DET_MODE                    1:0 /* RWIVF */
#define LW_PRAMDAC_PLL_COMPAT_LWPLL_DET_MODE_MULTOFF     0x00000000 /* RWI-V */
#define LW_PRAMDAC_PLL_COMPAT_LWPLL_DET_MODE_AUTO        0x00000002 /* RW--V */
#define LW_PRAMDAC_PLL_COMPAT_LWPLL_DET_MODE_MULTON      0x00000003 /* RW--V */
#define LW_PRAMDAC_PLL_COMPAT_LWPLL_DET_STAT                    3:3 /* R--VF */
#define LW_PRAMDAC_PLL_COMPAT_LWPLL_DET_STAT_OFF         0x00000000 /* R---V */
#define LW_PRAMDAC_PLL_COMPAT_LWPLL_DET_STAT_ON          0x00000001 /* R---V */
#define LW_PRAMDAC_PLL_COMPAT_MPLL_DET_MODE                     5:4 /* RWIVF */
#define LW_PRAMDAC_PLL_COMPAT_MPLL_DET_MODE_MULTOFF      0x00000000 /* RWI-V */
#define LW_PRAMDAC_PLL_COMPAT_MPLL_DET_MODE_AUTO         0x00000002 /* RW--V */
#define LW_PRAMDAC_PLL_COMPAT_MPLL_DET_MODE_MULTON       0x00000003 /* RW--V */
#define LW_PRAMDAC_PLL_COMPAT_MPLL_DET_STAT                     7:7 /* R--VF */
#define LW_PRAMDAC_PLL_COMPAT_MPLL_DET_STAT_OFF          0x00000000 /* R---V */
#define LW_PRAMDAC_PLL_COMPAT_MPLL_DET_STAT_ON           0x00000001 /* R---V */
#define LW_PRAMDAC_PLL_COMPAT_VPLL_DET_MODE                     9:8 /* RWIVF */
#define LW_PRAMDAC_PLL_COMPAT_VPLL_DET_MODE_MULTOFF      0x00000000 /* RWI-V */
#define LW_PRAMDAC_PLL_COMPAT_VPLL_DET_MODE_AUTO         0x00000002 /* RW--V */
#define LW_PRAMDAC_PLL_COMPAT_VPLL_DET_MODE_MULTON       0x00000003 /* RW--V */
#define LW_PRAMDAC_PLL_COMPAT_VPLL_DET_STAT                   11:11 /* R--VF */
#define LW_PRAMDAC_PLL_COMPAT_VPLL_DET_STAT_OFF          0x00000000 /* R---V */
#define LW_PRAMDAC_PLL_COMPAT_VPLL_DET_STAT_ON           0x00000001 /* R---V */
#define LW_PRAMDAC_PLL_COMPAT_VPLL2_DET_MODE                  13:12 /* RWIVF */
#define LW_PRAMDAC_PLL_COMPAT_VPLL2_DET_MODE_MULTOFF     0x00000000 /* RWI-V */
#define LW_PRAMDAC_PLL_COMPAT_VPLL2_DET_MODE_AUTO        0x00000002 /* RW--V */
#define LW_PRAMDAC_PLL_COMPAT_VPLL2_DET_MODE_MULTON      0x00000003 /* RW--V */
#define LW_PRAMDAC_PLL_COMPAT_VPLL2_DET_STAT                  15:15 /* R--VF */
#define LW_PRAMDAC_PLL_COMPAT_VPLL2_DET_STAT_OFF         0x00000000 /* R---V */
#define LW_PRAMDAC_PLL_COMPAT_VPLL2_DET_STAT_ON          0x00000001 /* R---V */
#define LW_PRAMDAC_PLL_COMPAT_6B_DITHER                       16:16 /* RWIVF */
#define LW_PRAMDAC_PLL_COMPAT_6B_DITHER_OFF              0x00000000 /* RWI-V */
#define LW_PRAMDAC_PLL_COMPAT_6B_DITHER_ON               0x00000001 /* RW--V */
#define LW_LW17_PRAMDAC_PLL_COMPAT_RESERVEDA                  23:16 /* RWIVF */
#define LW_LW17_PRAMDAC_PLL_COMPAT_RESERVEDA_VAL         0x00000000 /* RWI-V */
#define LW_LW25_PRAMDAC_PLL_COMPAT_RESERVEDA                  23:16 /* RWIVF */
#define LW_LW25_PRAMDAC_PLL_COMPAT_RESERVEDA_VAL         0x00000000 /* RWI-V */
#define LW_PRAMDAC_PLL_COMPAT_DITHER_RB                       17:17 /* RWIVF */
#define LW_PRAMDAC_PLL_COMPAT_DITHER_RB_NORMAL           0x00000000 /* RWI-V */
#define LW_PRAMDAC_PLL_COMPAT_DITHER_RB_ILWERT           0x00000001 /* RW--V */
#define LW_PRAMDAC_PLL_COMPAT_DITHER_G                        18:18 /* RWIVF */
#define LW_PRAMDAC_PLL_COMPAT_DITHER_G_NORMAL            0x00000000 /* RWI-V */
#define LW_PRAMDAC_PLL_COMPAT_DITHER_G_ILWERT            0x00000001 /* RW--V */
#define LW_PRAMDAC_PLL_COMPAT_DITHER_Y                        19:19 /* RWIVF */
#define LW_PRAMDAC_PLL_COMPAT_DITHER_Y_NORMAL            0x00000000 /* RWI-V */
#define LW_PRAMDAC_PLL_COMPAT_DITHER_Y_ILWERT            0x00000001 /* RW--V */
#define LW_PRAMDAC_PLL_COMPAT_RESERVED                        19:16 /* RWIVF */
#define LW_PRAMDAC_PLL_COMPAT_RESERVED_VAL               0x00000000 /* RWI-V */
#define LW_PRAMDAC_PLL_COMPAT_CRVCOUNT_SEL                    20:20 /* RWIVF */
#define LW_PRAMDAC_PLL_COMPAT_CRVCOUNT_SEL_NORMAL        0x00000000 /* RWI-V */
#define LW_PRAMDAC_PLL_COMPAT_CRVCOUNT_SEL_SYNC          0x00000001 /* RW--V */
#define LW_PRAMDAC_PLL_COMPAT_GPIO5_ALTDATA                   21:21 /* RWIVF */
#define LW_PRAMDAC_PLL_COMPAT_GPIO5_ALTDATA_DISABLED     0x00000000 /* RWI-V */
#define LW_PRAMDAC_PLL_COMPAT_GPIO5_ALTDATA_ENABLED      0x00000001 /* RWI-V */
#define LW_PRAMDAC_PLL_COMPAT_GPIO5_SELECT                    22:22 /* RWIVF */
#define LW_PRAMDAC_PLL_COMPAT_GPIO5_SELECT_TVD           0x00000000 /* RWI-V */
#define LW_PRAMDAC_PLL_COMPAT_GPIO5_SELECT_VIDEO         0x00000001 /* RWI-V */
#define LW_PRAMDAC_PLL_COMPAT_BLEND                           23:23 /* RWIVF */
#define LW_PRAMDAC_PLL_COMPAT_BLEND_NORMAL               0x00000000 /* RWI-V */
#define LW_PRAMDAC_PLL_COMPAT_BLEND_LSBBYPASS            0x00000001 /* RW--V */
#define LW_PRAMDAC_PLL_COMPAT_BLEND_RESERVED             0x00000000 /* RWI-V */
#define LW_PRAMDAC_PLL_COMPAT_MPDIV_XOR                       26:24 /* RWIVF */
#define LW_PRAMDAC_PLL_COMPAT_MPDIV_XOR_DISABLED         0x00000000 /* RWI-V */
#define LW_PRAMDAC_PLL_COMPAT_CHIP_REV                        28:28 /* R-IVF */
#define LW_PRAMDAC_PLL_COMPAT_CHIP_REV_B01_OR_B02        0x00000000 /* R---V */
#define LW_PRAMDAC_PLL_COMPAT_CHIP_REV_B03               0x00000001 /* R-I-V */
#define LW_PRAMDAC_PLL_COMPAT_RESERVEDB                       28:28 /* RWIVF */
#define LW_PRAMDAC_PLL_COMPAT_RESERVEDB_VAL              0x00000000 /* RWI-V */
#define LW_PRAMDAC_PLL_COMPAT_XTAL27                          30:30 /* RWIVF */
#define LW_PRAMDAC_PLL_COMPAT_XTAL27_MDOUBLE             0x00000000 /* RWI-V */
#define LW_PRAMDAC_PLL_COMPAT_XTAL27_MPASSTHRU           0x00000001 /* RW--V */

// used by dac.c
#define LW_PBUS_SEQ_PTR                                  0x00001304 /* RW-4R */
#define LW_PBUS_SEQ_PTR_A_FPON                                  5:0 /* RWIVF */
#define LW_PBUS_SEQ_PTR_A_FPON_DEFAULT                   0x00000000 /* RWI-V */
#define LW_PBUS_SEQ_PTR_B_FPOFF                                13:8 /* RWIVF */
#define LW_PBUS_SEQ_PTR_B_FPOFF_DEFAULT                  0x00000010 /* RWI-V */
#define LW_PBUS_SEQ_PTR_C_SUS                                 21:16 /* RWIVF */
#define LW_PBUS_SEQ_PTR_C_SUS_DEFAULT                    0x00000020 /* RWI-V */
#define LW_PBUS_SEQ_PTR_D_RES                                 29:24 /* RWIVF */
#define LW_PBUS_SEQ_PTR_D_RES_DEFAULT                    0x00000030 /* RWI-V */
#define LW_PBUS_SEQ_BYP                                  0x00001310 /* RW-4R */
#define LW_PBUS_SEQ_BYP_0_OVERRIDE_GPIO2_OUT                    0:0 /* RWIVF */
#define LW_PBUS_SEQ_BYP_0_OVERRIDE_GPIO2_OUT_0           0x00000000 /* RWI-V */
#define LW_PBUS_SEQ_BYP_0_OVERRIDE_GPIO2_OUT_1           0x00000001 /* RW--V */
#define LW_PBUS_SEQ_BYP_1_OVERRIDE_GPIO2_EN                     1:1 /* RWIVF */
#define LW_PBUS_SEQ_BYP_1_OVERRIDE_GPIO2_EN_0            0x00000000 /* RWI-V */
#define LW_PBUS_SEQ_BYP_1_OVERRIDE_GPIO2_EN_1            0x00000001 /* RW--V */
#define LW_PBUS_SEQ_BYP_2_OVERRIDE_GPIO3_OUT                    2:2 /* RWIVF */
#define LW_PBUS_SEQ_BYP_2_OVERRIDE_GPIO3_OUT_0           0x00000000 /* RWI-V */
#define LW_PBUS_SEQ_BYP_2_OVERRIDE_GPIO3_OUT_1           0x00000001 /* RW--V */
#define LW_PBUS_SEQ_BYP_3_OVERRIDE_GPIO3_EN                     3:3 /* RWIVF */
#define LW_PBUS_SEQ_BYP_3_OVERRIDE_GPIO3_EN_0            0x00000000 /* RWI-V */
#define LW_PBUS_SEQ_BYP_3_OVERRIDE_GPIO3_EN_1            0x00000001 /* RW--V */
#define LW_PBUS_SEQ_BYP_4_OVERRIDE_PWRDOWN_H1                   4:4 /* RWIVF */
#define LW_PBUS_SEQ_BYP_4_OVERRIDE_PWRDOWN_H1_0          0x00000000 /* RWI-V */
#define LW_PBUS_SEQ_BYP_4_OVERRIDE_PWRDOWN_H1_1          0x00000001 /* RW--V */
#define LW_PBUS_SEQ_BYP_5_OVERRIDE_PWRDOWN_H2                   5:5 /* RWIVF */
#define LW_PBUS_SEQ_BYP_5_OVERRIDE_PWRDOWN_H2_0          0x00000000 /* RWI-V */
#define LW_PBUS_SEQ_BYP_5_OVERRIDE_PWRDOWN_H2_1          0x00000001 /* RW--V */
#define LW_PBUS_SEQ_BYP_6_OVERRIDE_PD_TMDSPLL_H1                6:6 /* RWIVF */
#define LW_PBUS_SEQ_BYP_6_OVERRIDE_PD_TMDSPLL_H1_0       0x00000000 /* RWI-V */
#define LW_PBUS_SEQ_BYP_6_OVERRIDE_PD_TMDSPLL_H1_1       0x00000001 /* RW--V */
#define LW_PBUS_SEQ_BYP_7_OVERRIDE_PD_TMDSPLL_H2                7:7 /* RWIVF */
#define LW_PBUS_SEQ_BYP_7_OVERRIDE_PD_TMDSPLL_H2_0       0x00000000 /* RWI-V */
#define LW_PBUS_SEQ_BYP_7_OVERRIDE_PD_TMDSPLL_H2_1       0x00000001 /* RW--V */
#define LW_PBUS_SEQ_BYP_8_OVERRIDE_AUX3_TMDS1_L0                8:8 /* RWIVF */
#define LW_PBUS_SEQ_BYP_8_OVERRIDE_AUX3_TMDS1_L0_0       0x00000000 /* RWI-V */
#define LW_PBUS_SEQ_BYP_8_OVERRIDE_AUX3_TMDS1_L0_1       0x00000001 /* RW--V */
#define LW_PBUS_SEQ_BYP_9_OVERRIDE_AUX3_TMDS1_L1                9:9 /* RWIVF */
#define LW_PBUS_SEQ_BYP_9_OVERRIDE_AUX3_TMDS1_L1_0       0x00000000 /* RWI-V */
#define LW_PBUS_SEQ_BYP_9_OVERRIDE_AUX3_TMDS1_L1_1       0x00000001 /* RW--V */
#define LW_PBUS_SEQ_BYP_10_OVERRIDE_AUX3_TMDS2_L0             10:10 /* RWIVF */
#define LW_PBUS_SEQ_BYP_10_OVERRIDE_AUX3_TMDS2_L0_0      0x00000000 /* RWI-V */
#define LW_PBUS_SEQ_BYP_10_OVERRIDE_AUX3_TMDS2_L0_1      0x00000001 /* RW--V */
#define LW_PBUS_SEQ_BYP_11_OVERRIDE_AUX3_TMDS2_L1             11:11 /* RWIVF */
#define LW_PBUS_SEQ_BYP_11_OVERRIDE_AUX3_TMDS2_L1_0      0x00000000 /* RWI-V */
#define LW_PBUS_SEQ_BYP_11_OVERRIDE_AUX3_TMDS2_L1_1      0x00000001 /* RW--V */
#define LW_PBUS_SEQ_BYP_12_OVERRIDE_FPBLANK_H1                12:12 /* RWIVF */
#define LW_PBUS_SEQ_BYP_12_OVERRIDE_FPBLANK_H1_0         0x00000000 /* RWI-V */
#define LW_PBUS_SEQ_BYP_12_OVERRIDE_FPBLANK_H1_1         0x00000001 /* RW--V */
#define LW_PBUS_SEQ_BYP_13_OVERRIDE_FPBLANK_H2                13:13 /* RWIVF */
#define LW_PBUS_SEQ_BYP_13_OVERRIDE_FPBLANK_H2_0         0x00000000 /* RWI-V */
#define LW_PBUS_SEQ_BYP_13_OVERRIDE_FPBLANK_H2_1         0x00000001 /* RW--V */
#define LW_PBUS_SEQ_BYP_14_OVERRIDE                           14:14 /* RWIVF */
#define LW_PBUS_SEQ_BYP_14_OVERRIDE_0                    0x00000000 /* RWI-V */
#define LW_PBUS_SEQ_BYP_14_OVERRIDE_1                    0x00000001 /* RW--V */
#define LW_PBUS_SEQ_BYP_15_OVERRIDE                           15:15 /* RWIVF */
#define LW_PBUS_SEQ_BYP_15_OVERRIDE_0                    0x00000000 /* RWI-V */
#define LW_PBUS_SEQ_BYP_15_OVERRIDE_1                    0x00000001 /* RW--V */
#define LW_PBUS_SEQ_BYP_0_ENABLE_GPIO2_OUT                    16:16 /* RWIVF */
#define LW_PBUS_SEQ_BYP_0_ENABLE_GPIO2_OUT_OFF           0x00000000 /* RWI-V */
#define LW_PBUS_SEQ_BYP_0_ENABLE_GPIO2_OUT_ON            0x00000001 /* RW--V */
#define LW_PBUS_SEQ_BYP_1_ENABLE_GPIO2_EN                     17:17 /* RWIVF */
#define LW_PBUS_SEQ_BYP_1_ENABLE_GPIO2_EN_OFF            0x00000000 /* RWI-V */
#define LW_PBUS_SEQ_BYP_1_ENABLE_GPIO2_EN_ON             0x00000001 /* RW--V */
#define LW_PBUS_SEQ_BYP_2_ENABLE_GPIO3_OUT                    18:18 /* RWIVF */
#define LW_PBUS_SEQ_BYP_2_ENABLE_GPIO3_OUT_OFF           0x00000000 /* RWI-V */
#define LW_PBUS_SEQ_BYP_2_ENABLE_GPIO3_OUT_ON            0x00000001 /* RW--V */
#define LW_PBUS_SEQ_BYP_3_ENABLE_GPIO3_EN                     19:19 /* RWIVF */
#define LW_PBUS_SEQ_BYP_3_ENABLE_GPIO3_EN_OFF            0x00000000 /* RWI-V */
#define LW_PBUS_SEQ_BYP_3_ENABLE_GPIO3_EN_ON             0x00000001 /* RW--V */
#define LW_PBUS_SEQ_BYP_4_ENABLE_PWRDOWN_H1                   20:20 /* RWIVF */
#define LW_PBUS_SEQ_BYP_4_ENABLE_PWRDOWN_H1_OFF          0x00000000 /* RWI-V */
#define LW_PBUS_SEQ_BYP_4_ENABLE_PWRDOWN_H1_ON           0x00000001 /* RW--V */
#define LW_PBUS_SEQ_BYP_5_ENABLE_PWRDOWN_H2                   21:21 /* RWIVF */
#define LW_PBUS_SEQ_BYP_5_ENABLE_PWRDOWN_H2_OFF          0x00000000 /* RWI-V */
#define LW_PBUS_SEQ_BYP_5_ENABLE_PWRDOWN_H2_ON           0x00000001 /* RW--V */
#define LW_PBUS_SEQ_BYP_6_ENABLE_PD_TMDSPLL_H1                22:22 /* RWIVF */
#define LW_PBUS_SEQ_BYP_6_ENABLE_PD_TMDSPLL_H1_OFF       0x00000000 /* RWI-V */
#define LW_PBUS_SEQ_BYP_6_ENABLE_PD_TMDSPLL_H1_ON        0x00000001 /* RW--V */
#define LW_PBUS_SEQ_BYP_7_ENABLE_PD_TMDSPLL_H2                23:23 /* RWIVF */
#define LW_PBUS_SEQ_BYP_7_ENABLE_PD_TMDSPLL_H2_OFF       0x00000000 /* RWI-V */
#define LW_PBUS_SEQ_BYP_7_ENABLE_PD_TMDSPLL_H2_ON        0x00000001 /* RW--V */
#define LW_PBUS_SEQ_BYP_8_ENABLE_AUX3_TMDS1_L0                24:24 /* RWIVF */
#define LW_PBUS_SEQ_BYP_8_ENABLE_AUX3_TMDS1_L0_OFF       0x00000000 /* RWI-V */
#define LW_PBUS_SEQ_BYP_8_ENABLE_AUX3_TMDS1_L0_ON        0x00000001 /* RW--V */
#define LW_PBUS_SEQ_BYP_9_ENABLE_AUX3_TMDS1_L1                25:25 /* RWIVF */
#define LW_PBUS_SEQ_BYP_9_ENABLE_AUX3_TMDS1_L1_OFF       0x00000000 /* RWI-V */
#define LW_PBUS_SEQ_BYP_9_ENABLE_AUX3_TMDS1_L1_ON        0x00000001 /* RW--V */
#define LW_PBUS_SEQ_BYP_10_ENABLE_AUX3_TMDS2_L0               26:26 /* RWIVF */
#define LW_PBUS_SEQ_BYP_10_ENABLE_AUX3_TMDS2_L0_OFF      0x00000000 /* RWI-V */
#define LW_PBUS_SEQ_BYP_10_ENABLE_AUX3_TMDS2_L0_ON       0x00000001 /* RW--V */
#define LW_PBUS_SEQ_BYP_11_ENABLE_AUX3_TMDS2_L1               27:27 /* RWIVF */
#define LW_PBUS_SEQ_BYP_11_ENABLE_AUX3_TMDS2_L1_OFF      0x00000000 /* RWI-V */
#define LW_PBUS_SEQ_BYP_11_ENABLE_AUX3_TMDS2_L1_ON       0x00000001 /* RW--V */
#define LW_PBUS_SEQ_BYP_12_ENABLE_FPBLANK_H1                  28:28 /* RWIVF */
#define LW_PBUS_SEQ_BYP_12_ENABLE_FPBLANK_H1_OFF         0x00000000 /* RWI-V */
#define LW_PBUS_SEQ_BYP_12_ENABLE_FPBLANK_H1_ON          0x00000001 /* RW--V */
#define LW_PBUS_SEQ_BYP_13_ENABLE_FPBLANK_H2                  29:29 /* RWIVF */
#define LW_PBUS_SEQ_BYP_13_ENABLE_FPBLANK_H2_OFF         0x00000000 /* RWI-V */
#define LW_PBUS_SEQ_BYP_13_ENABLE_FPBLANK_H2_ON          0x00000001 /* RW--V */
#define LW_PBUS_SEQ_BYP_14_ENABLE                             30:30 /* RWIVF */
#define LW_PBUS_SEQ_BYP_14_ENABLE_OFF                    0x00000000 /* RWI-V */
#define LW_PBUS_SEQ_BYP_14_ENABLE_ON                     0x00000001 /* RW--V */
#define LW_PBUS_SEQ_BYP_15_ENABLE                             31:31 /* RWIVF */
#define LW_PBUS_SEQ_BYP_15_ENABLE_OFF                    0x00000000 /* RWI-V */
#define LW_PBUS_SEQ_BYP_15_ENABLE_ON                     0x00000001 /* RW--V */
#define LW_PBUS_SEQ_STATUS                               0x00001308 /* R--4R */
#define LW_PBUS_SEQ_STATUS_SEQ1_PC                              5:0 /* R--VF */
#define LW_PBUS_SEQ_STATUS_SEQ1_PC_DEFAULT               0x00000000 /* R-I-V */
#define LW_PBUS_SEQ_STATUS_SEQ1_STATUS                          9:8 /* R--VF */
#define LW_PBUS_SEQ_STATUS_SEQ1_STATUS_IDLE              0x00000000 /* R-I-V */
#define LW_PBUS_SEQ_STATUS_SEQ1_STATUS_RUNNING           0x00000001 /* R---V */
#define LW_PBUS_SEQ_STATUS_SEQ1_STATUS_ERROR             0x00000003 /* R---V */
#define LW_PBUS_SEQ_STATUS_SEQ2_PC                            21:16 /* R--VF */
#define LW_PBUS_SEQ_STATUS_SEQ2_PC_DEFAULT               0x00000000 /* R-I-V */
#define LW_PBUS_SEQ_STATUS_SEQ2_STATUS                        25:24 /* R--VF */
#define LW_PBUS_SEQ_STATUS_SEQ2_STATUS_IDLE              0x00000000 /* R-I-V */
#define LW_PBUS_SEQ_STATUS_SEQ2_STATUS_RUNNING           0x00000001 /* R---V */
#define LW_PBUS_SEQ_STATUS_SEQ2_STATUS_ERROR             0x00000003 /* R---V */
#define LW_PBUS_SEQ_RAM(i)                       (0x00001400+(i)*4) /* RW-4A */
#define LW_PBUS_SEQ_RAM__SIZE_1                                  16 /*       */
#define LW_PBUS_SEQ_RAM_VALUE                                  31:0 /* RW-4F */
#define LW_PBUS_SEQ_RAM_RESERVE(i)               (0x00001440+(i)*4) /* RW-4A */
#define LW_PBUS_SEQ_RAM_RESERVE__SIZE_1                          48 /*       */
#define LW_PBUS_SEQ_RAM_RESERVE_VALUE                          31:0 /* RW-4F */

#endif // #define _LW_REF_H_
