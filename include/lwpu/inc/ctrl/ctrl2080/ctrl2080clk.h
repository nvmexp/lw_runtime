/*
 * SPDX-FileCopyrightText: Copyright (c) 2005-2021 LWPU CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#pragma once

#include <lwtypes.h>
#if defined(_MSC_VER)
#pragma warning(disable:4324)
#endif

//
// This file was generated with FINN, an LWPU coding tool.
// Source file: ctrl/ctrl2080/ctrl2080clk.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)


#include "lwfixedtypes.h"
#include "ctrl/ctrl2080/ctrl2080base.h"
#include "ctrl/ctrl2080/ctrl2080boardobj.h"
#include "ctrl/ctrl2080/ctrl2080gpumon.h"
#include "ctrl/ctrl2080/ctrl2080clkavfs.h"
#include "ctrl/ctrl2080/ctrl2080volt.h"
#include "ctrl/ctrl2080/ctrl2080pmumon.h"
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)

#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1) && (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

#include "ctrl/ctrl2080/ctrl2080clk_opaque_non_privileged.h"
#include "ctrl/ctrl2080/ctrl2080clk_opaque_privileged.h"
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1) && (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)


/* LW20_SUBDEVICE_XX clock control commands and parameters */
/*
 * LW2080_CTRL_CLK_GET_DOMAINS
 *
 * This command returns the clock domains supported by the specified GPU
 * in the form of a 32bit mask.
 *
 *   clkDomains
 *     This parameter contains the 32bit mask of clock domains
 *     The type of the list of domains returned is based on
 *     the value of clkGetDomainsType provided by the caller.
 *     Each bit position in the clkDomains mask corresponds to a unique
 *     LW2080_CTRL_CLK_DOMAIN value.
 *
 *     Legal clock domain values are:
 *         LW2080_CTRL_CLK_DOMAIN_MCLK
 *           This value represents the memory clock domain.  The memory
 *           clock domain is supported on all GPUs.  The reported memory
 *           clock domain frequencies represent the external frequency used
 *           to drive the RAMs.  These values can be effectively doubled
 *           if the LW2080_CTRL_CLK_INFO_FLAGS_MCLK_DDR flag is enabled.
 *           The memory clock domain supports both deferred and immediate
 *           mode set operations with LW2080_CTRL_CLK_SET_INFO_FLAGS_WHEN.
 *         LW2080_CTRL_CLK_DOMAIN_HOSTCLK
 *           This value represents the host clock domain.  The host clock
 *           domain may not be supported on all GPUs.
 *         LW2080_CTRL_CLK_DOMAIN_DISPCLK
 *           This value represents the display clock domain.  The display clock
 *           domain may not be supported on all GPUs.
 *         LW2080_CTRL_CLK_DOMAIN_PCLK0
 *           This value represents the pixel clock domain for head0 in the system
 *         LW2080_CTRL_CLK_DOMAIN_PCLK1
 *           This value represents the pixel clock domain for head1 in the system
 *         LW2080_CTRL_CLK_DOMAIN_PCLK2
 *           This value represents the pixel clock domain for head2 in the system
 *         LW2080_CTRL_CLK_DOMAIN_PCLK3
 *           This value represents the pixel clock domain for head3 in the system
 *         LW2080_CTRL_CLK_DOMAIN_XCLK
 *           This value represents the (PEX) xclk clock domain.  The xclk clock
 *           domain may not be supported on all GPUs.  The default xclk clock
 *           domain frequency is bus-dependent.  Additional valid xclk clock
 *           domain frequencies are chipset-dependent.
 *         LW2080_CTRL_CLK_DOMAIN_LTC2CLK
 *           This value represents the 2x GPU cache clock domain. The GPU cache2
 *           clock domain may not be supported on all GPUs.
 *
 *   clkDomainsType
 *     This parameter specifies the type of domains expected from
 *     the get domains operation.
 *
 *     Legal clock domain types are:
 *         LW2080_CTRL_CLK_DOMAINS_TYPE_ALL
 *           This returns all possible supported clock domains.
 *         LW2080_CTRL_CLK_DOMAINS_TYPE_PROGRAMMABALE
 *           This returns all possible programmable clock domains only.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_NOT_SUPPORTED
 */
#define LW2080_CTRL_CMD_CLK_GET_DOMAINS (0x20801001) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_CLK_INTERFACE_ID << 8) | LW2080_CTRL_CLK_GET_DOMAINS_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_CLK_GET_DOMAINS_PARAMS_MESSAGE_ID (0x1U)

typedef struct LW2080_CTRL_CLK_GET_DOMAINS_PARAMS {
    LwU32 clkDomains;
    LwU32 clkDomainsType;
} LW2080_CTRL_CLK_GET_DOMAINS_PARAMS;

/* valid clock domain values */
#define LW2080_CTRL_CLK_DOMAIN                                     31:0
#define LW2080_CTRL_CLK_DOMAIN_UNDEFINED                (0x00000000U)
#define LW2080_CTRL_CLK_DOMAIN_GPCCLK                   (0x00000001U)
#define LW2080_CTRL_CLK_DOMAIN_XBARCLK                  (0x00000002U)
#define LW2080_CTRL_CLK_DOMAIN_SYSCLK                   (0x00000004U)
#define LW2080_CTRL_CLK_DOMAIN_HUBCLK                   (0x00000008U)
#define LW2080_CTRL_CLK_DOMAIN_MCLK                     (0x00000010U)
#define LW2080_CTRL_CLK_DOMAIN_HOSTCLK                  (0x00000020U)
#define LW2080_CTRL_CLK_DOMAIN_DISPCLK                  (0x00000040U)
#define LW2080_CTRL_CLK_DOMAIN_PCLK0                    (0x00000080U)
#define LW2080_CTRL_CLK_DOMAIN_PCLK1                    (0x00000100U)
#define LW2080_CTRL_CLK_DOMAIN_PCLK2                    (0x00000200U)
#define LW2080_CTRL_CLK_DOMAIN_PCLK3                    (0x00000400U)
#define LW2080_CTRL_CLK_DOMAIN_PCLK(i)                             ((1U << (i)) * LW2080_CTRL_CLK_DOMAIN_PCLK0)
#define LW2080_CTRL_CLK_DOMAIN_PCLK__SIZE_1             (0x00000004U)
#define LW2080_CTRL_CLK_DOMAIN_XCLK                     (0x00000800U)
#define LW2080_CTRL_CLK_DOMAIN_GPC2CLK                  (0x00001000U)     /* Deprecated as of Turing */
#define LW2080_CTRL_CLK_DOMAIN_LTC2CLK                  (0x00002000U)     /* Deprecated as of Turing */
#define LW2080_CTRL_CLK_DOMAIN_XBAR2CLK                 (0x00004000U)     /* Deprecated as of Turing */
#define LW2080_CTRL_CLK_DOMAIN_SYS2CLK                  (0x00008000U)     /* Deprecated as of Turing */
#define LW2080_CTRL_CLK_DOMAIN_HUB2CLK                  (0x00010000U)     /* Deprecated as of Turing */
#define LW2080_CTRL_CLK_DOMAIN_LEGCLK                   (0x00020000U)
#define LW2080_CTRL_CLK_DOMAIN_UTILSCLK                 (0x00040000U)
#define LW2080_CTRL_CLK_DOMAIN_PWRCLK                   (0x00080000U)
#define LW2080_CTRL_CLK_DOMAIN_LWDCLK                   (0x00100000U)     /* Maxwell and after */
#define LW2080_CTRL_CLK_DOMAIN_MSDCLK                   LW2080_CTRL_CLK_DOMAIN_LWDCLK   /* MSD is Kepler-only */
#define LW2080_CTRL_CLK_DOMAIN_PCIEGENCLK               (0x00200000U)

#define LW2080_CTRL_CLK_DOMAIN_VCLK0                    (0x00400000U)
#define LW2080_CTRL_CLK_DOMAIN_VCLK1                    (0x00800000U)
#define LW2080_CTRL_CLK_DOMAIN_VCLK2                    (0x01000000U)
#define LW2080_CTRL_CLK_DOMAIN_VCLK3                    (0x02000000U)
#define LW2080_CTRL_CLK_DOMAIN_VCLK(i)                             ((1U << (i)) * LW2080_CTRL_CLK_DOMAIN_VCLK0)
#define LW2080_CTRL_CLK_DOMAIN_VCLK__SIZE_1             (0x00000004U)

#define LW2080_IS_VCLK_DOMAIN(clkDomain) (ONEBITSET((clkDomain)) && \
                                          ((clkDomain) >= LW2080_CTRL_CLK_DOMAIN_VCLK0) && \
                                          ((clkDomain) <= LW2080_CTRL_CLK_DOMAIN_VCLK((LW2080_CTRL_CLK_DOMAIN_VCLK__SIZE_1 - 1))))

// Deprecated clock domains
#define LW2080_CTRL_CLK_DOMAIN_DEPRECATED               (0x80000000U)
// Offsets added to avoid build failures on old MAC cross module branches as LwGlitzGM-Latch
#define LW2080_CTRL_CLK_DOMAIN_LWCLK                    (0x80000001U) /* finn: Evaluated from "(LW2080_CTRL_CLK_DOMAIN_DEPRECATED + 0x00000001)" */
#define LW2080_CTRL_CLK_DOMAIN_GCLK                     (0x80000002U) /* finn: Evaluated from "(LW2080_CTRL_CLK_DOMAIN_DEPRECATED + 0x00000002)" */
#define LW2080_CTRL_CLK_DOMAIN_SCLK                     (0x80000004U) /* finn: Evaluated from "(LW2080_CTRL_CLK_DOMAIN_DEPRECATED + 0x00000004)" */
#define LW2080_CTRL_CLK_DOMAIN_RCLK                     (0x80000008U) /* finn: Evaluated from "(LW2080_CTRL_CLK_DOMAIN_DEPRECATED + 0x00000008)" */
#define LW2080_CTRL_CLK_DOMAIN_HOTCLK                   (0x80000010U) /* finn: Evaluated from "(LW2080_CTRL_CLK_DOMAIN_DEPRECATED + 0x00000010)" */
#define LW2080_CTRL_CLK_DOMAIN_VPSCLK                   (0x80000020U) /* finn: Evaluated from "(LW2080_CTRL_CLK_DOMAIN_DEPRECATED + 0x00000020)" */
#define LW2080_CTRL_CLK_DOMAIN_VPVCLK                   (0x80000040U) /* finn: Evaluated from "(LW2080_CTRL_CLK_DOMAIN_DEPRECATED + 0x00000040)" */
#define LW2080_CTRL_CLK_DOMAIN_BYPCLK                   (0x80000080U) /* finn: Evaluated from "(LW2080_CTRL_CLK_DOMAIN_DEPRECATED + 0x00000080)" */
#define LW2080_CTRL_CLK_DOMAIN_COLD_LWCLK               (0x80000100U) /* finn: Evaluated from "(LW2080_CTRL_CLK_DOMAIN_DEPRECATED + 0x00000100)" */
#define LW2080_CTRL_CLK_DOMAIN_COLD_HOTCLK              (0x80000200U) /* finn: Evaluated from "(LW2080_CTRL_CLK_DOMAIN_DEPRECATED + 0x00000200)" */
#define LW2080_CTRL_CLK_DOMAIN_GPUCACHECLK              (0x80000400U) /* finn: Evaluated from "(LW2080_CTRL_CLK_DOMAIN_DEPRECATED + 0x00000400)" */

/* valid list of clock domain types. */
#define LW2080_CTRL_CLK_DOMAINS_TYPE_ALL                (0x00000000U)
#define LW2080_CTRL_CLK_DOMAINS_TYPE_PROGRAMMABALE_ONLY (0x00000001U)

/* List of partition indexes for a clock domain. */
#define LW2080_CTRL_CLK_DOMAIN_PART_IDX_0               (0x00000000U)
#define LW2080_CTRL_CLK_DOMAIN_PART_IDX_1               (0x00000001U)
#define LW2080_CTRL_CLK_DOMAIN_PART_IDX_2               (0x00000002U)
#define LW2080_CTRL_CLK_DOMAIN_PART_IDX_3               (0x00000003U)
#define LW2080_CTRL_CLK_DOMAIN_PART_IDX_4               (0x00000004U)
#define LW2080_CTRL_CLK_DOMAIN_PART_IDX_5               (0x00000005U)
#define LW2080_CTRL_CLK_DOMAIN_PART_IDX_6               (0x00000006U)
#define LW2080_CTRL_CLK_DOMAIN_PART_IDX_7               (0x00000007U)
#define LW2080_CTRL_CLK_DOMAIN_PART_IDX_8               (0x00000008U)
#define LW2080_CTRL_CLK_DOMAIN_PART_IDX_9               (0x00000009U)
#define LW2080_CTRL_CLK_DOMAIN_PART_IDX_10              (0x0000000AU)
#define LW2080_CTRL_CLK_DOMAIN_PART_IDX_11              (0x0000000BU)

/* Maximum number of partitions for a clock domain. */
#define LW2080_CTRL_CLK_DOMAIN_PART_IDX_MAX             (0xlw) /* finn: Evaluated from "(LW2080_CTRL_CLK_DOMAIN_PART_IDX_11 + 1)" */

/* Invalid/Undefined partition index for a clock domain. */
#define LW2080_CTRL_CLK_DOMAIN_PART_IDX_UNDEFINED       (0x000000FFU)

/*
 * LW2080_CTRL_CLK_INFO
 *
 * This structure describes per-domain clock information:
 *
 *   flags
 *     This field specifies flags for the clock information entry.
 *     These flags can be domain-dependent, and so may not be valid
 *     for all supported domains.  This field is deprecated for use in
 *     LW2080_CTRL_CLK_GET_INFO.
 *     Possible valid flags are:
 *       LW2080_CTRL_CLK_INFO_FLAGS_MCLK_DDR
 *         This flag is used to indicate that the memory clock domain
 *         is driving DDR-style RAMs.  When this flag is set to
 *         LW2080_CTRL_CLK_INFO_FLAGS_MCLK_DDR_ENABLED, then the
 *         frequencies can be doubled to obtain the effective clock rates.
 *       LW2080_CTRL_CLK_INFO_FLAGS_GPUCACHE2CLK_FORCE_LWCLK
 *         This flag has been deprecated and is equivalent to
 *         LW2080_CTRL_CLK_SOURCE_LWCLK.
 *       LW2080_CTRL_CLK_INFO_FLAGS_PATH
 *         This flag controls the path to the source specified in clkSource.
 *         _DEFAULT     RM decides on the best path to clkSource.
 *         _PLL         Use the domain-specific PLL (reference path)
 *         _BYPASS      Bypass the domain-specific PLL (alternate path)
 *         _SEMI_BYPASS Bypass one but not both domain-specific PLLs
 *       LW2080_CTRL_CLK_INFO_FLAGS_FORCE_PLL
 *         Deprecated.  Equivalent to LW2080_CTRL_CLK_INFO_FLAGS_PATH_PLL.
 *         This flag indicates that the clock source should
 *         be forced to a PLL (and not the bypass clock source).
 *       LW2080_CTRL_CLK_INFO_FLAGS_FORCE_BYPASS
 *         Deprecated.  Equivalent to LW2080_CTRL_CLK_INFO_FLAGS_PATH_BYPASS.
 *         This flag indicates that the clock source should
 *         be forced to the bypass clock source.
 *       LW2080_CTRL_CLK_INFO_FLAGS_RESERVED
 *         Do not use.  Reserved for expansion of _PATH flags.
 *       LW2080_CTRL_CLK_INFO_FLAGS_MDIV_RANGE
 *         Controls the range of domain-specific PLL MDIV values.
 *   clkDomain
 *     This field is used to specify the domain for which clock information
 *     is to be processed.  This field should specify a valid
 *     LW2080_CTRL_CLK_DOMAIN value.
 *   actualFreq
 *     This field contains the actual clock frequency for the domain in
 *     units of KHz.  This value may deviate from the desired target
 *     frequency due to PLL constraints.
 *   targetFreq
 *     This field contains the target clock frequency for the domain in
 *     units of KHz and should be specified with a non-zero value to the
 *     LW2080_CTRL_CLK_SET_INFO command.  LW2080_CTRL_CMD_CLK_GET_INFO
 *     reports zero for this field to indicate that the clock domain has not
 *     been programmed beyond its power-on or vbios default; or the same
 *     the same value as actualFreq if it's unable to determine the
 *     target.  Use of this field in LW2080_CTRL_CLK_GET_INFO or
 *     LW2080_CTRL_CLK_GET_EXTENDED_INFO or
 *     LW2080_CTRL_CMD_CLK_GET_GPUMON_CLOCK_SAMPLES is deprecated.
 *   clkSource
 *     This field contains the clock source for the domain.  This field
 *     should specify a valid LW2080_CTRL_CLK_SOURCE value.
 */
typedef struct LW2080_CTRL_CLK_INFO {
    LwU32 flags;
    LwU32 clkDomain;
    LwU32 actualFreq;
    LwU32 targetFreq;
    LwU32 clkSource;
} LW2080_CTRL_CLK_INFO;

/* valid clock info flags */
#define LW2080_CTRL_CLK_INFO_FLAGS_MCLK_DDR                        0:0
#define LW2080_CTRL_CLK_INFO_FLAGS_MCLK_DDR_DISABLE                 (0x00000000U)
#define LW2080_CTRL_CLK_INFO_FLAGS_MCLK_DDR_ENABLE                  (0x00000001U)
#define LW2080_CTRL_CLK_INFO_FLAGS_GPUCACHE2CLK_FORCE_LWCLK        1:1
#define LW2080_CTRL_CLK_INFO_FLAGS_GPUCACHE2CLK_FORCE_LWCLK_DISABLE (0x00000000U)
#define LW2080_CTRL_CLK_INFO_FLAGS_GPUCACHE2CLK_FORCE_LWCLK_ENABLE  (0x00000001U)
#define LW2080_CTRL_CLK_INFO_FLAGS_PATH                            4:2
#define LW2080_CTRL_CLK_INFO_FLAGS_PATH_DEFAULT                     (0x00000000U)
#define LW2080_CTRL_CLK_INFO_FLAGS_PATH_PLL                         (0x00000001U)
#define LW2080_CTRL_CLK_INFO_FLAGS_PATH_BYPASS                      (0x00000002U)
#define LW2080_CTRL_CLK_INFO_FLAGS_PATH_SEMI_BYPASS                 (0x00000003U)
#define LW2080_CTRL_CLK_INFO_FLAGS_PATH_NAFLL                       (0x00000004U)
#define LW2080_CTRL_CLK_INFO_FLAGS_FORCE_PLL                       2:2
#define LW2080_CTRL_CLK_INFO_FLAGS_FORCE_PLL_DISABLE                (0x00000000U)
#define LW2080_CTRL_CLK_INFO_FLAGS_FORCE_PLL_ENABLE                 (0x00000001U)
#define LW2080_CTRL_CLK_INFO_FLAGS_FORCE_BYPASS                    3:3
#define LW2080_CTRL_CLK_INFO_FLAGS_FORCE_BYPASS_DISABLE             (0x00000000U)
#define LW2080_CTRL_CLK_INFO_FLAGS_FORCE_BYPASS_ENABLE              (0x00000001U)
#define LW2080_CTRL_CLK_INFO_FLAGS_FORCE_NAFLL                     4:4
#define LW2080_CTRL_CLK_INFO_FLAGS_FORCE_NAFLL_DISABLE              (0x00000000U)
#define LW2080_CTRL_CLK_INFO_FLAGS_FORCE_NAFLL_ENABLE               (0x00000001U)
#define LW2080_CTRL_CLK_INFO_FLAGS_RESERVED                        7:5 // for expansion of _PATH
#define LW2080_CTRL_CLK_INFO_FLAGS_MDIV_RANGE                      8:8
#define LW2080_CTRL_CLK_INFO_FLAGS_MDIV_RANGE_DEFAULT               (0x00000000U)
#define LW2080_CTRL_CLK_INFO_FLAGS_MDIV_RANGE_FULL                  (0x00000001U)
// NAFLL force regime flags
#define LW2080_CTRL_CLK_INFO_FLAGS_FORCE_FFR_SET                   9:9
#define LW2080_CTRL_CLK_INFO_FLAGS_FORCE_FFR_SET_NO                 (0x00000000U)
#define LW2080_CTRL_CLK_INFO_FLAGS_FORCE_FFR_SET_YES                (0x00000001U)
#define LW2080_CTRL_CLK_INFO_FLAGS_FORCE_CLEAR                     10:10
#define LW2080_CTRL_CLK_INFO_FLAGS_FORCE_CLEAR_NO                   (0x00000000U)
#define LW2080_CTRL_CLK_INFO_FLAGS_FORCE_CLEAR_YES                  (0x00000001U)
#define LW2080_CTRL_CLK_INFO_FLAGS_FORCE_VR_SET                    11:11
#define LW2080_CTRL_CLK_INFO_FLAGS_FORCE_VR_SET_NO                  (0x00000000U)
#define LW2080_CTRL_CLK_INFO_FLAGS_FORCE_VR_SET_YES                 (0x00000001U)
#define LW2080_CTRL_CLK_INFO_FLAGS_FORCE_FR_SET                    12:12
#define LW2080_CTRL_CLK_INFO_FLAGS_FORCE_FR_SET_NO                  (0x00000000U)
#define LW2080_CTRL_CLK_INFO_FLAGS_FORCE_FR_SET_YES                 (0x00000001U)

/* valid clock source values */
#define LW2080_CTRL_CLK_SOURCE_DEFAULT                              (0x00000000U)
#define LW2080_CTRL_CLK_SOURCE_MPLL                                 (0x00000001U)
#define LW2080_CTRL_CLK_SOURCE_DISPPLL                              (0x00000002U)
#define LW2080_CTRL_CLK_SOURCE_VPLL(i)                             (0x00000003 + (i))
#define LW2080_CTRL_CLK_SOURCE_VPLL__SIZE_1                         4U
#define LW2080_CTRL_CLK_SOURCE_VPLL0                               LW2080_CTRL_CLK_SOURCE_VPLL(0)
#define LW2080_CTRL_CLK_SOURCE_VPLL1                               LW2080_CTRL_CLK_SOURCE_VPLL(1)
#define LW2080_CTRL_CLK_SOURCE_VPLL2                               LW2080_CTRL_CLK_SOURCE_VPLL(2)
#define LW2080_CTRL_CLK_SOURCE_VPLL3                               LW2080_CTRL_CLK_SOURCE_VPLL(3)
#define LW2080_CTRL_CLK_SOURCE_SPPLL0                               (0x00000007U)
#define LW2080_CTRL_CLK_SOURCE_SPPLL1                               (0x00000008U)
#define LW2080_CTRL_CLK_SOURCE_XCLK                                 (0x00000009U)
#define LW2080_CTRL_CLK_SOURCE_PEXREFCLK                            (0x0000000AU)
#define LW2080_CTRL_CLK_SOURCE_XTAL                                 (0x0000000BU)
#define LW2080_CTRL_CLK_SOURCE_3XXCLKDIV2                           (0x0000000LW)
#define LW2080_CTRL_CLK_SOURCE_GPCPLL                               (0x0000000DU)
#define LW2080_CTRL_CLK_SOURCE_LTCPLL                               (0x0000000EU)
#define LW2080_CTRL_CLK_SOURCE_XBARPLL                              (0x0000000FU)
#define LW2080_CTRL_CLK_SOURCE_SYSPLL                               (0x00000010U)
#define LW2080_CTRL_CLK_SOURCE_XTAL4X                               (0x00000011U)
#define LW2080_CTRL_CLK_SOURCE_REFMPLL                              (0x00000012U)
#define LW2080_CTRL_CLK_SOURCE_HOSTCLK                              (0x00000013U)
#define LW2080_CTRL_CLK_SOURCE_XCLK500                              (0x00000014U)
#define LW2080_CTRL_CLK_SOURCE_XCLKGEN3                             (0x00000015U)
#define LW2080_CTRL_CLK_SOURCE_HBMPLL                               (0x00000016U)
#define LW2080_CTRL_CLK_SOURCE_LWDPLL                               (0x00000017U)
#define LW2080_CTRL_CLK_SOURCE_DEFROSTCLK                           (0x00000018U)

// Deprecated clock sources
#define LW2080_CTRL_CLK_SOURCE_DEPRECATED                           (0xF0000000U)
// Offsets added to avoid build failures on old MAC cross module branches as LwGlitzGM-Latch
#define LW2080_CTRL_CLK_SOURCE_LWPLL1                               (0xf0000001U) /* finn: Evaluated from "(LW2080_CTRL_CLK_SOURCE_DEPRECATED + 0x00000001)" */
#define LW2080_CTRL_CLK_SOURCE_LWPLL2                               (0xf0000002U) /* finn: Evaluated from "(LW2080_CTRL_CLK_SOURCE_DEPRECATED + 0x00000002)" */
#define LW2080_CTRL_CLK_SOURCE_MPLLA                                (0xf0000003U) /* finn: Evaluated from "(LW2080_CTRL_CLK_SOURCE_DEPRECATED + 0x00000003)" */
#define LW2080_CTRL_CLK_SOURCE_MPLLB                                (0xf0000004U) /* finn: Evaluated from "(LW2080_CTRL_CLK_SOURCE_DEPRECATED + 0x00000004)" */
#define LW2080_CTRL_CLK_SOURCE_MPLLC                                (0xf0000005U) /* finn: Evaluated from "(LW2080_CTRL_CLK_SOURCE_DEPRECATED + 0x00000005)" */
#define LW2080_CTRL_CLK_SOURCE_MPLLD                                (0xf0000006U) /* finn: Evaluated from "(LW2080_CTRL_CLK_SOURCE_DEPRECATED + 0x00000006)" */
#define LW2080_CTRL_CLK_SOURCE_MPLLEXT                              (0xf0000007U) /* finn: Evaluated from "(LW2080_CTRL_CLK_SOURCE_DEPRECATED + 0x00000007)" */
#define LW2080_CTRL_CLK_SOURCE_HPLL                                 (0xf0000008U) /* finn: Evaluated from "(LW2080_CTRL_CLK_SOURCE_DEPRECATED + 0x00000008)" */
#define LW2080_CTRL_CLK_SOURCE_AGPPLL                               (0xf0000009U) /* finn: Evaluated from "(LW2080_CTRL_CLK_SOURCE_DEPRECATED + 0x00000009)" */
#define LW2080_CTRL_CLK_SOURCE_GVPLL                                (0xf000000aU) /* finn: Evaluated from "(LW2080_CTRL_CLK_SOURCE_DEPRECATED + 0x0000000A)" */
#define LW2080_CTRL_CLK_SOURCE_3XXCLK                               (0xf000000bU) /* finn: Evaluated from "(LW2080_CTRL_CLK_SOURCE_DEPRECATED + 0x0000000B)" */
#define LW2080_CTRL_CLK_SOURCE_VPSPLL                               (0xf000000lw) /* finn: Evaluated from "(LW2080_CTRL_CLK_SOURCE_DEPRECATED + 0x0000000C)" */
#define LW2080_CTRL_CLK_SOURCE_LWCLK                                (0xf000000dU) /* finn: Evaluated from "(LW2080_CTRL_CLK_SOURCE_DEPRECATED + 0x0000000D)" */
#define LW2080_CTRL_CLK_SOURCE_GPUREF                               (0xf000000eU) /* finn: Evaluated from "(LW2080_CTRL_CLK_SOURCE_DEPRECATED + 0x0000000E)" */
#define LW2080_CTRL_CLK_SOURCE_XCLKGEN2                             (0xf000000fU) /* finn: Evaluated from "(LW2080_CTRL_CLK_SOURCE_DEPRECATED + 0x0000000F)" */
#define LW2080_CTRL_CLK_SOURCE_BYPASSCLK                            (0xf0000010U) /* finn: Evaluated from "(LW2080_CTRL_CLK_SOURCE_DEPRECATED + 0x00000010)" */
#define LW2080_CTRL_CLK_SOURCE_COREPLL                              (0xf0000011U) /* finn: Evaluated from "(LW2080_CTRL_CLK_SOURCE_DEPRECATED + 0x00000011)" */

/* maximum number of supported domains */
#define LW2080_CTRL_CLK_ARCH_MAX_DOMAINS                            32U

/*
 * Deprecated. Please use LW2080_CTRL_CMD_CLK_GET_INFO_V2 instead.
 */
#define LW2080_CTRL_CMD_CLK_GET_INFO                                (0x20801002) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_CLK_INTERFACE_ID << 8) | 0x2" */

typedef struct LW2080_CTRL_CLK_GET_INFO_PARAMS {
    LwU32 flags;
    LwU32 clkInfoListSize;
    LW_DECLARE_ALIGNED(LwP64 clkInfoList, 8);
} LW2080_CTRL_CLK_GET_INFO_PARAMS;

/*
 * LW2080_CTRL_CLK_SET_INFO
 *
 * This command programs the specified clock domains using the information
 * contained in the corresponding clock info structures.  Not all domains
 * support the set operation.  Note also that on some GPUs it's required
 * that a group of related domains be set in one operation.  The absence
 * of a domain-specifc clock info structure in such cases will lead to an
 * invalid command error.
 *
 *   flags
 *     This field specifies the desired flags for the set clock operation.
 *     Supported set operation flags include:
 *
 *       LW2080_CTRL_CLK_SET_INFO_FLAGS_WHEN
 *         This flag can be used to defer the clock set operation(s).
 *         A value of _DEFERRED indicates that the set operation(s)
 *         should be held off until the next modeset.  This option is
 *         mostly obsolete.  It was useful before dynamic clocking,
 *         when most clock changes caused the screen to glitch.
 *         The default value of _IMMEDIATE indicates the set operation(s)
 *         should occur synchronously.  Not all clock domain support
 *         deferred set operations.
 *       LW2080_CTRL_CLK_SET_INFO_FLAGS_SET_MAXPERF
 *         This flag is OBSOLETE.
 *       LW2080_CTRL_CLK_SET_INFO_FLAGS_VBLANK
 *         This flag indicates the clock change must happen immediately.
 *         The default value of _WAIT causes delays the clock change
 *         until the next vblank when required to prevent screen
 *         corruption.
 *         A value of _NOWAIT indicates that the clock change must
 *         happen ASAP, even if it causes temporary screen corruption.
 *         It is typically reserved for cases where the power supply
 *         is no longer capable of supporting the GPU, such as when
 *         the AC adapter is unplugged on a notebook.
 *
 *   clkInfoListSize
 *     This field specifies the number of entries on the caller's clkInfoList.
 *
 *   clkInfoList
 *     This field specifies a pointer in the caller's address space
 *     to the buffer from which the desired clock information is to be
 *     retrieved.  This buffer must be at least as big as clkInfoListSize
 *     multiplied by the size of the LW2080_CTRL_CLK_INFO structure.
 *
 *     clkInfoList[i].actualFreq is set to the frequency that would actually be
 *     programmed in lieu of the targetFreq, if known, or targetFreq otherwise.
 *     Clients wanting to use this feature should initialize actualFreq.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_ILWALID_COMMAND
 *   LW_ERR_OPERATING_SYSTEM
 *
 * Deprecated, please use LW2080_CTRL_CMD_CLK_SET_INFO_V2.
 */
#define LW2080_CTRL_CMD_CLK_SET_INFO (0x20801003) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_CLK_INTERFACE_ID << 8) | 0x3" */

typedef struct LW2080_CTRL_CLK_SET_INFO_PARAMS {
    LwU32 flags;
    LwU32 clkInfoListSize;
    LW_DECLARE_ALIGNED(LwP64 clkInfoList, 8);
} LW2080_CTRL_CLK_SET_INFO_PARAMS;

#define LW2080_CTRL_CLK_SET_INFO_FLAGS_WHEN                        0:0
#define LW2080_CTRL_CLK_SET_INFO_FLAGS_WHEN_DEFERRED       (0x00000000U)
#define LW2080_CTRL_CLK_SET_INFO_FLAGS_WHEN_IMMEDIATE      (0x00000001U)
#define LW2080_CTRL_CLK_SET_INFO_FLAGS_SET_MAXPERF                 1:1
#define LW2080_CTRL_CLK_SET_INFO_FLAGS_SET_MAXPERF_DISABLE (0x00000000U)
#define LW2080_CTRL_CLK_SET_INFO_FLAGS_SET_MAXPERF_ENABLE  (0x00000001U)
#define LW2080_CTRL_CLK_SET_INFO_FLAGS_VBLANK                      2:2
#define LW2080_CTRL_CLK_SET_INFO_FLAGS_VBLANK_WAIT         (0x00000000U)
#define LW2080_CTRL_CLK_SET_INFO_FLAGS_VBLANK_NOWAIT       (0x00000001U)

/*!
 * LW2080_CTRL_CMD_CLK_SET_INFO_V2
 *
 * Same as LW2080_CTRL_CMD_CLK_SET_INFO but without embedded pointer.
 *
 */
#define LW2080_CTRL_CMD_CLK_SET_INFO_V2                    (0x20801041) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_CLK_INTERFACE_ID << 8) | LW2080_CTRL_CLK_SET_INFO_V2_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_CLK_SET_INFO_V2_PARAMS_MESSAGE_ID (0x41U)

typedef struct LW2080_CTRL_CLK_SET_INFO_V2_PARAMS {
    LwU32                flags;
    LwU32                clkInfoListSize;
    LW2080_CTRL_CLK_INFO clkInfoList[LW2080_CTRL_CLK_ARCH_MAX_DOMAINS];
} LW2080_CTRL_CLK_SET_INFO_V2_PARAMS;

/*
 * LW2080_CTRL_CLK_EXTENDED_INFO
 *
 * This structure describes per-domain clock information, including extended
 * clock information.
 *
 *   clkInfo
 *     This field contains per-domain clock information.  See the
 *     description of LW2080_CTRL_CLK_INFO for more information.
 *   effectiveFreq
 *     This field contains the effective clock frequency for the domain
 *     in units of KHz. Value is equal to actualFreq, or less if some
 *     event (i.e. thermal) requires clock slowdown.
 *   reserved
 *     This space is reserved for binary compatibility in future api
 *     extensions.
 */
typedef struct LW2080_CTRL_CLK_EXTENDED_INFO {
    LW2080_CTRL_CLK_INFO clkInfo;
    LwU32                effectiveFreq;
    LwU32                reserved00[6];
} LW2080_CTRL_CLK_EXTENDED_INFO;

/*
 * LW2080_CTRL_CLK_GET_EXTENDED_INFO
 *
 * This command returns extended clock information for each valid entry
 * in the extended clock information array.
 *
 *   flags
 *     This field specifies the desired flags for the get clock operation.
 *     This field is lwrrently unused.
 *   numClkInfos
 *     This field specifies the number of entries in the caller's
 *     clkInfos array.  The value in this field must not exceed
 *     LW2080_CTRL_CLK_MAX_INFOS.
 *   clkInfos
 *     This field contains the array into which extended clock information
 *     is to be returned.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_OPERATING_SYSTEM
 */
#define LW2080_CTRL_CMD_CLK_GET_EXTENDED_INFO (0x20801004) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_CLK_INTERFACE_ID << 8) | LW2080_CTRL_CLK_GET_EXTENDED_INFO_PARAMS_MESSAGE_ID" */

/* this macro specifies the maximum number of extended info entries */
#define LW2080_CTRL_CLK_MAX_EXTENDED_INFOS    (0x00000020U)

#define LW2080_CTRL_CLK_GET_EXTENDED_INFO_PARAMS_MESSAGE_ID (0x4U)

typedef struct LW2080_CTRL_CLK_GET_EXTENDED_INFO_PARAMS {
    LwU32                         flags;
    LwU32                         numClkInfos;
    LW2080_CTRL_CLK_EXTENDED_INFO clkInfos[LW2080_CTRL_CLK_MAX_EXTENDED_INFOS];
} LW2080_CTRL_CLK_GET_EXTENDED_INFO_PARAMS;

/*
 * LW2080_CTRL_CLK_GET_CLOSEST_BYPASS_FREQ
 *
 * This command is used to get the two closest bypass frequencies achievable for a
 * particular clock architecture greater than and lesser than the reference
 * ferquency passed in.
 *   clkDomain
 *     The clock domain for which we want to find the closest bypass path freq.
 *     For more information on clkDomain see LW2080_CTRL_CLK_GET_DOMAINS above.
 *   targetFreqKHz
 *     This field specifies the target frequency in KHz passed in by the caller.
 *     This frequency servers as reference for callwlation the closest lower
 *     and higher bypass path frequencies.
 *   maxBypassFreqKHz
 *     This field returns the max closest achievable bypass path frequency,
 *     greater than the targetFreqKHz.
 *   minBypassFreqKHz
 *     This field returns the min closest achievable bypass path frequency,
 *     lesser than targetFreqKHz.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_ILWALID_COMMAND
 *   LW_ERR_OPERATING_SYSTEM
 */
#define LW2080_CTRL_CMD_CLK_GET_CLOSEST_BYPASS_FREQ (0x20801005) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_CLK_INTERFACE_ID << 8) | LW2080_CTRL_CLK_GET_CLOSEST_BYPASS_FREQ_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_CLK_GET_CLOSEST_BYPASS_FREQ_PARAMS_MESSAGE_ID (0x5U)

typedef struct LW2080_CTRL_CLK_GET_CLOSEST_BYPASS_FREQ_PARAMS {
    LwU32 clkDomain;
    LwU32 targetFreqKHz;
    LwU32 minBypassFreqKHz;
    LwU32 maxBypassFreqKHz;
} LW2080_CTRL_CLK_GET_CLOSEST_BYPASS_FREQ_PARAMS;

/*
 * LW2080_CTRL_CMD_SET_VPLL_REF
 *
 * This command sets the vpll ref that must be used for programming VPLL in
 * future. Note that VPLL won't be immediately programmed. Only future
 * programming would get affected.
 *
 *   head
 *      The head for which this cmd is intended.
 *
 *   refName
 *      The ref clk that must be used.
 *
 *   refFreq
 *      Frequency of the specified reference source. This field is relevant
 *      only when refName = EXT_REF, QUAL_EXT_REF or EXT_SPREAD.
 *      The unit of frequency is Hz.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_GENERIC
 */
#define LW2080_CTRL_CMD_SET_VPLL_REF                       (0x20801007) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_CLK_INTERFACE_ID << 8) | LW2080_CTRL_CMD_SET_VPLL_REF_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_CMD_SET_VPLL_REF_REF_NAME                             31:0
#define LW2080_CTRL_CMD_SET_VPLL_REF_REF_NAME_XTAL         (0x00000000U)
#define LW2080_CTRL_CMD_SET_VPLL_REF_REF_NAME_SPPLL0       (0x00000001U)
#define LW2080_CTRL_CMD_SET_VPLL_REF_REF_NAME_SPPLL1       (0x00000002U)
#define LW2080_CTRL_CMD_SET_VPLL_REF_REF_NAME_EXT_REF      (0x00000003U)
#define LW2080_CTRL_CMD_SET_VPLL_REF_REF_NAME_QUAL_EXT_REF (0x00000004U)
#define LW2080_CTRL_CMD_SET_VPLL_REF_REF_NAME_EXT_SPREAD   (0x00000005U)

#define LW2080_CTRL_CMD_SET_VPLL_REF_PARAMS_MESSAGE_ID (0x7U)

typedef struct LW2080_CTRL_CMD_SET_VPLL_REF_PARAMS {
    LwU32 head;

    LwU32 refName;
    LwU32 refFreq;
} LW2080_CTRL_CMD_SET_VPLL_REF_PARAMS;

/*
 * LW2080_CTRL_CMD_CLK_SET_PLL
 *
 * This command programs the specified PLL to specified frequency
 *
 *   clkInfoListSize
 *     This field specifies the number of entries on the caller's clkInfoList.
 *   clkInfoList
 *     This field specifies a pointer in the caller's address space
 *     to the buffer from which the desired clock information is to be
 *     retrieved.  This buffer must be at least as big as clkInfoListSize
 *     multiplied by the size of the LW2080_CTRL_CLK_INFO structure.
 *
 *     The caller is expected to fill in targetFreq and clkSource of the
 *     LW2080_CTRL_CLK_INFO struct
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_ILWALID_COMMAND
 *   LW_ERR_OPERATING_SYSTEM
 *
 * Deprecated, please use LW2080_CTRL_CMD_CLK_SET_PLL_V2.
 */
#define LW2080_CTRL_CMD_CLK_SET_PLL (0x20801008) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_CLK_INTERFACE_ID << 8) | 0x8" */

typedef struct LW2080_CTRL_CLK_SET_PLL_PARAMS {
    LwU32 clkInfoListSize;
    LW_DECLARE_ALIGNED(LwP64 clkInfoList, 8);
} LW2080_CTRL_CLK_SET_PLL_PARAMS;

/*!
 * LW2080_CTRL_CMD_CLK_SET_PLL_V2
 *
 * Same as LW2080_CTRL_CMD_CLK_SET_PLL but without embedded pointer.
 *
 */
#define LW2080_CTRL_CMD_CLK_SET_PLL_V2 (0x20801043) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_CLK_INTERFACE_ID << 8) | LW2080_CTRL_CLK_SET_PLL_V2_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_CLK_SET_PLL_V2_PARAMS_MESSAGE_ID (0x43U)

typedef struct LW2080_CTRL_CLK_SET_PLL_V2_PARAMS {
    LwU32                clkInfoListSize;
    LW2080_CTRL_CLK_INFO clkInfoList[LW2080_CTRL_CLK_ARCH_MAX_DOMAINS];
} LW2080_CTRL_CLK_SET_PLL_V2_PARAMS;

/*
 * Following are the list of APIs for getting information on the public exposed
 * domains of clocks. Any client interested in reading or programming the
 * public exposed clock domains should first query for the exposed clock domains
 * to find out which of the LW2080_CTRL_CLK_DOMAIN it is mapped to and how to
 * colwert the frequency from the internal to public domain and vice versa.
 */

/*!
 * @brief   Type reserved for @ref LW2080_CTRL_CLK_PUBLIC_DOMAIN enumerations.
 */
typedef LwU32 LW2080_CTRL_CLK_PUBLIC_DOMAIN;

/*!
 * @defgroup LW2080_CTRL_CLK_PUBLIC_DOMAIN_ENUM
 *
 * Enumeration of public clock domains. Of type
 * @ref LW2080_CTRL_CLK_PUBLIC_DOMAIN.
 *
 * @{
 */
#define LW2080_CTRL_CLK_PUBLIC_DOMAIN_GRAPHICS  (0x00000001U)
#define LW2080_CTRL_CLK_PUBLIC_DOMAIN_PROCESSOR (0x00000002U)
#define LW2080_CTRL_CLK_PUBLIC_DOMAIN_MEMORY    (0x00000004U)
#define LW2080_CTRL_CLK_PUBLIC_DOMAIN_VIDEO     (0x00000008U)

//
// Special value, not to be used in LW2080_CTRL_CMD_CLK_GET_PUBLIC_DOMAINS,
// LW2080_CTRL_CMD_CLK_GET_PUBLIC_DOMAIN_INFO, LW2080_CTRL_CMD_CLK_GET_PUBLIC_DOMAIN_INFO_V2
//
#define LW2080_CTRL_CLK_PUBLIC_DOMAIN_ILWALID   (0xFFFFFFFFU)
/*!@}*/

/*
 * LW2080_CTRL_CMD_CLK_GET_PUBLIC_DOMAINS
 *
 * This command returns the public clock domains supported by the specified GPU
 * in the form of a 32bit mask.
 *
 *   flags
 *     This field is reserved for future use.
 *   publicDomains
 *     This field contains the 32bit mask of clock domains.
 *     Each bit position in the mask corresponds to a unique
 *     LW2080_CTRL_CLK_PUBLIC_DOMAIN value.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 */
#define LW2080_CTRL_CMD_CLK_GET_PUBLIC_DOMAINS  (0x20801009) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_CLK_INTERFACE_ID << 8) | LW2080_CTRL_CLK_GET_PUBLIC_DOMAINS_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_CLK_GET_PUBLIC_DOMAINS_PARAMS_MESSAGE_ID (0x9U)

typedef struct LW2080_CTRL_CLK_GET_PUBLIC_DOMAINS_PARAMS {
    LwU32 flags;
    LwU32 publicDomains;
} LW2080_CTRL_CLK_GET_PUBLIC_DOMAINS_PARAMS;

/*
 * LW2080_CTRL_CLK_PUBLIC_DOMAIN_INFO
 *
 * This structure describes per-domain public clock information:
 *
 *   publicDomain
 *     This field is used to specify the public clock domain for which
 *     information is to be returned. This field should specify a valid
 *     LW2080_CTRL_CLK_PUBLIC_DOMAIN value.
 *   clkDomain
 *     This field contains the internal clock domin the specified public
 *     clock domain is mapped to. This should should return a valid
 *     LW2080_CTRL_CLK_DOMAIN value.
 *   flags
 *     This field specifies flags for the clock information entry.
 *     Possible valid flags are:
 *       LW2080_CTRL_CLK_PUBLIC_DOMAIN_INFO_FLAGS_PROGRAMMABLE
 *         This flag is used to indicate that this public clock is
 *         programmable. Freq of non-programmable public clock domains
 *         cannot be changed directly.
 *   freqAdjPct
 *     This field specifies the frequency relationship between the public
 *     clock domain and the internal clock domain it is mapped to. The
 *     frequency of the public clock = the frequency of the internal
 *     clock x the value contained in this field / 100.
 */

#define LW2080_CTRL_CLK_PUBLIC_DOMAIN_INFO_FLAGS_PROGRAMMABLE              0:0
#define LW2080_CTRL_CLK_PUBLIC_DOMAIN_INFO_FLAGS_PROGRAMMABLE_NO  (0x00000000U)
#define LW2080_CTRL_CLK_PUBLIC_DOMAIN_INFO_FLAGS_PROGRAMMABLE_YES (0x00000001U)

typedef struct LW2080_CTRL_CLK_PUBLIC_DOMAIN_INFO {
    LwU32 publicDomain;
    LwU32 clkDomain;
    LwU32 flags;
    LwU32 freqAdjPct;
} LW2080_CTRL_CLK_PUBLIC_DOMAIN_INFO;

/*
 * Deprecated. Use LW2080_CTRL_CMD_CLK_GET_PUBLIC_DOMAIN_INFO_V2 instead.
 */
#define LW2080_CTRL_CMD_CLK_GET_PUBLIC_DOMAIN_INFO (0x2080100a) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_CLK_INTERFACE_ID << 8) | LW2080_CTRL_CLK_GET_PUBLIC_DOMAIN_INFO_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_CLK_GET_PUBLIC_DOMAIN_INFO_PARAMS_MESSAGE_ID (0xAU)

typedef struct LW2080_CTRL_CLK_GET_PUBLIC_DOMAIN_INFO_PARAMS {
    LwU32 flags;
    LwU32 publicDomainInfoListSize;
    LW_DECLARE_ALIGNED(LwP64 publicDomainInfoList, 8);
} LW2080_CTRL_CLK_GET_PUBLIC_DOMAIN_INFO_PARAMS;


/*
 * LW2080_CTRL_CMD_CLK_GET_PROFILING_DATA
 *
 * This command returns profiling information after every MCLK switch.
 * DMI_POS/RGDpca returned are for the first active head.
 *
 */

#define LW2080_CTRL_CMD_CLK_GET_PROFILING_DATA (0x2080100b) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_CLK_INTERFACE_ID << 8) | LW2080_CTRL_CLK_GET_PROFILING_DATA_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_CLK_GET_PROFILING_DATA_PARAMS_MESSAGE_ID (0xBU)

typedef struct LW2080_CTRL_CLK_GET_PROFILING_DATA_PARAMS {
    LwU32 dmiPos;
    LwU32 RGDpca;
    LwU32 FBStartTime;
    LwU32 FBStopTime;
    LwU32 pmuTimeout;
} LW2080_CTRL_CLK_GET_PROFILING_DATA_PARAMS;

/*!
 * LW2080_CTRL_CMD_CLK_GET_SRC_FREQ_KHZ_V2
 *
 * This command returns the current frequency of the specified clkSrc
 *
 *   clkInfoListSize
 *     This field specifies the number of entries on the caller's clkInfoList.
 *   clkInfoList
 *     This buffer specifies the desired clock information to be
 *     retrieved.
 *
 *     The caller is expected to fill in clkSource of the
 *     LW2080_CTRL_CLK_INFO struct, and the current freq of the clkSource
 *     would be returned in the field actualFreq.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *
 */
#define LW2080_CTRL_CMD_CLK_GET_SRC_FREQ_KHZ_V2 (0x20801042) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_CLK_INTERFACE_ID << 8) | LW2080_CTRL_CLK_GET_SRC_FREQ_KHZ_V2_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_CLK_GET_SRC_FREQ_KHZ_V2_PARAMS_MESSAGE_ID (0x42U)

typedef struct LW2080_CTRL_CLK_GET_SRC_FREQ_KHZ_V2_PARAMS {
    LwU32                clkInfoListSize;
    LW2080_CTRL_CLK_INFO clkInfoList[LW2080_CTRL_CLK_ARCH_MAX_DOMAINS];
} LW2080_CTRL_CLK_GET_SRC_FREQ_KHZ_V2_PARAMS;

/*
 * LW2080_CTRL_CLK_PSTATES2_INFO
 *
 * This structure describes per-domain clock information related to
 * P-states 2.0.
 *
 *   clkDomain
 *     This field is used to specify the domain for which clock information
 *     is to be processed.  This field should specify a valid
 *     LW2080_CTRL_CLK_DOMAIN value.
 *   flags
 *     This field specifies flags for the entry.
 *       LW2080_CTRL_CLK_PSTATES2_INFO_FLAGS_USAGE
 *         This field defines clock domain usage.
 *           _FIXED     - domain is fixed at initial value by devinit
 *           _PSTATE    - domain changes with P-state, just like P-state 1.0
 *           _DECOUPLED - domain is the primary of a decoupled group
 *           _RATIO     - domain is a member of a group programmed at a ratio of the primary
 *       LW2080_CTRL_CLK_PSTATES2_INFO_FLAGS_PSTATEFLOOR
 *         means whether the domain must stay above the nominal value of the current pstate
 *   ratioDomain
 *     If the frequency of clkDomain is programmed as a ratio of another
 *     primary clock domain, this field would return the LW2080_CTRL_CLK_DOMAIN
 *     value of the primary domain. Otherwise, this field would return
 *     LW2080_CTRL_CLK_DOMAIN_UNDEFINED.
 *   ratio
 *     If the frequency of clkDomain is programmed as a ratio of another
 *     primary clock domain, this field would return the frequency ratio of
 *     clkDomain to the primary domain in percent.
 */
typedef struct LW2080_CTRL_CLK_PSTATES2_INFO {
    LwU32 clkDomain;
    LwU32 flags;
    LwU32 ratioDomain;
    LwU8  ratio;
} LW2080_CTRL_CLK_PSTATES2_INFO;

/* valid clock pstates2 info flags */
#define LW2080_CTRL_CLK_PSTATES2_INFO_FLAGS_USAGE                          2:0
#define LW2080_CTRL_CLK_PSTATES2_INFO_FLAGS_USAGE_FIXED       (0x00000000U)
#define LW2080_CTRL_CLK_PSTATES2_INFO_FLAGS_USAGE_PSTATE      (0x00000001U)
#define LW2080_CTRL_CLK_PSTATES2_INFO_FLAGS_USAGE_DECOUPLED   (0x00000002U)
#define LW2080_CTRL_CLK_PSTATES2_INFO_FLAGS_USAGE_RATIO       (0x00000003U)
#define LW2080_CTRL_CLK_PSTATES2_INFO_FLAGS_PSTATEFLOOR                    3:3
#define LW2080_CTRL_CLK_PSTATES2_INFO_FLAGS_PSTATEFLOOR_TRUE  (0x00000001U)
#define LW2080_CTRL_CLK_PSTATES2_INFO_FLAGS_PSTATEFLOOR_FALSE (0x00000000U)

/*
 * Deprecated. Please use LW2080_CTRL_CMD_CLK_GET_PSTATES2_INFO_V2 instead.
 */
#define LW2080_CTRL_CMD_CLK_GET_PSTATES2_INFO                 (0x2080100d) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_CLK_INTERFACE_ID << 8) | 0xD" */

typedef struct LW2080_CTRL_CLK_GET_PSTATES2_INFO_PARAMS {
    LwU32 flags;
    LwU32 clkInfoListSize;
    LW_DECLARE_ALIGNED(LwP64 clkInfoList, 8);
} LW2080_CTRL_CLK_GET_PSTATES2_INFO_PARAMS;


/*
 * LW2080_CTRL_CLK_ARCH_GET_DOMAINS
 *
 * This command returns the per-architecture clock domains supported by the
 * specified GPU in the form of an array.  Note this command is not supported
 * on all GPU types.  Refer to the LW2080_CTRL_CMD_CLK_GET_DOMAINS command
 * for the alternative to use on such GPUs.
 *
 *   clkDomainsType
 *     This parameter specifies the type of domains expected from
 *     the get domains operation.  Legal clock domain types are:
 *         LW2080_CTRL_CLK_DOMAINS_TYPE_ALL
 *           This returns all possible supported clock domains.
 *         LW2080_CTRL_CLK_DOMAINS_TYPE_PROGRAMMABALE
 *           This returns all possible programmable clock domains only.
 *   clkDomainsCount
 *     This parameter returns the number of valid clock domain values returned
 *     in the clkDomains array.
 *   clkDomains
 *     This array returns the set of architecture-specific supported
 *     domains.  The total number of valid entries is returned in the
 *     clkDomainsCount parameter.  Legal per-architecture clock domain values
 *     include:
 *       LW2080_CTRL_CLK_ARCH_DOMAIN_2D
 *         This value represents the 2D graphics clock domain.
 *       LW2080_CTRL_CLK_ARCH_DOMAIN_3D
 *         This value represents the 3D graphics clock domain.
 *       LW2080_CTRL_CLK_ARCH_DOMAIN_HOST1X
 *          This value represents the host1x clock domain.
 *       LW2080_CTRL_CLK_ARCH_DOMAIN_DISP0
 *          This value represents the display clock domain for display controller 0.
 *       LW2080_CTRL_CLK_ARCH_DOMAIN_DISP1
 *          This value represents the display clock domain for display controller 1.
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_NOT_SUPPORTED
 */

#define LW2080_CTRL_CMD_CLK_GET_ARCH_DOMAINS (0x2080100e) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_CLK_INTERFACE_ID << 8) | 0xE" */

typedef struct LW2080_CTRL_CLK_GET_ARCH_DOMAINS_PARAMS {
    LwU32 clkDomainsType;
    LwU32 clkDomainsCount;
    LwU32 clkDomains[LW2080_CTRL_CLK_ARCH_MAX_DOMAINS];
} LW2080_CTRL_CLK_GET_ARCH_DOMAINS_PARAMS;

/* valid clock domain values*/
#define LW2080_CTRL_CLK_ARCH_DOMAIN_UNDEFINED 0x00000000U
#define LW2080_CTRL_CLK_ARCH_DOMAIN_2D        0xE2000001U
#define LW2080_CTRL_CLK_ARCH_DOMAIN_3D        0xE2000002U
#define LW2080_CTRL_CLK_ARCH_DOMAIN_HOST1X    0xE2000003U
#define LW2080_CTRL_CLK_ARCH_DOMAIN_DISP0     0xE2000004U
#define LW2080_CTRL_CLK_ARCH_DOMAIN_DISP1     0xE2000005U

/*
 * LW2080_CTRL_CLK_PLL_INFO
 *
 * This structure describes operational limits and attributes for a PLL.
 *
 * It is used by LW2080_CTRL_CMD_CLK_GET_PLL_INFO and
 * LW2080_CTRL_CMD_CLK_SET_PLL_SPELWLATIVELY.
 *
 *   pll
 *     Denotes which PLL to return information for. See the
 *     LW2080_CTRL_CLK_SOURCE_* macros above.
 *
 *   milwcoFreq
 *   maxVcoFreq
 *     Denotes PLL VCO frequency limits in Hz;
 *
 *   ndiv
 *   mdiv
 *   pldiv
 *     The lwrrently programmed coefficients for the PLL.
 *
 *   bNdivSlidingSupported
 *   ndivLo
 *   ndivMid
 *     Denotes whether NDIV sliding is supported by the PLL and if so, the
 *     lwrrently programmed coefficients.
 */
typedef struct LW2080_CTRL_CLK_PLL_INFO {
    LwU32  pll;

    LwU32  milwcoFreq;
    LwU32  maxVcoFreq;

    LwU32  ndiv;
    LwU32  mdiv;
    LwU32  pldiv;

    LwU32  ndivLo;
    LwU32  ndivMid;
    LwBool bNdivSliding;
} LW2080_CTRL_CLK_PLL_INFO;

/*
 * LW2080_CTRL_CMD_CLK_GET_PLL_INFO
 *
 * This command returns information about the given PLL. This lwrrently
 * includes some operational limits and coefficients. The PLL to return
 * information for is passed-in via "pllInfo.pll" (see LW2080_CTRL_CLK_PLL_INFO
 * above).
 *
 * NOTE: This command lwrrently only works on the GPCPLL. However, it could be
 * extended to support more PLLs.
 *
 *   flags
 *     Denotes desired options to this command. It is lwrrently unused and
 *     should be set to 0.
 *
 *   pllInfo
 *     Returns the given Pll's operational limits and coefficients.
 *
 * Possible status return values are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_NOT_SUPPORTED
 */
#define LW2080_CTRL_CMD_CLK_GET_PLL_INFO (0x2080100f) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_CLK_INTERFACE_ID << 8) | LW2080_CTRL_CLK_GET_PLL_INFO_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_CLK_GET_PLL_INFO_PARAMS_MESSAGE_ID (0xFU)

typedef struct LW2080_CTRL_CLK_GET_PLL_INFO_PARAMS {
    LwU32                    flags;
    LW2080_CTRL_CLK_PLL_INFO pllInfo;
} LW2080_CTRL_CLK_GET_PLL_INFO_PARAMS;

/*
 * LW2080_CTRL_CMD_CLK_SET_PLL_SPELWLATIVELY
 * DEPRECATED
 *
 *
 * This command spelwlatively sets the given PLL to the given frequency. The
 * PLL to set is passed-in via "pllInfo.pll" (see LW2080_CTRL_CLK_PLL_INFO
 * above).
 *
 * Since the operation is spelwlative, no PLL registers are changed.
 * Information about how the PLL would have been set is returned in "pllInfo".
 * The actual frequency that the PLL would have been set to is returned in
 * "actualFreq".
 *
 *   flags
 *     See below for valid values.
 *
 *   targetFreq
 *     Denotes the desired spelwlative frequency for the given PLL in KHz.
 *
 *   actualFreq
 *     Denotes the actual frequency the given PLL would have been set to in
 *     KHz.
 *
 *   pllInfo
 *     Returns the given PLL's spelwlative settings.
 *
 * Possible status return values are:
 *   LW_ERR_NOT_SUPPORTED
 */
#define LW2080_CTRL_CMD_CLK_SET_PLL_SPELWLATIVELY (0x20801010) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_CLK_INTERFACE_ID << 8) | LW2080_CTRL_CLK_SET_PLL_SPELWLATIVELY_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_CLK_SET_PLL_SPELWLATIVELY_PARAMS_MESSAGE_ID (0x10U)

typedef struct LW2080_CTRL_CLK_SET_PLL_SPELWLATIVELY_PARAMS {
    LwU32                    flags;
    LwU32                    targetFreq;
    LwU32                    actualFreq;
    LW2080_CTRL_CLK_PLL_INFO pllInfo;
} LW2080_CTRL_CLK_SET_PLL_SPELWLATIVELY_PARAMS;

// valid flags
#define LW2080_CTRL_CLK_SET_PLL_SPELWLATIVELY_FLAGS_MDIV_RANGE           0:0
#define LW2080_CTRL_CLK_SET_PLL_SPELWLATIVELY_FLAGS_MDIV_RANGE_DEFAULT (0x00000000U)
#define LW2080_CTRL_CLK_SET_PLL_SPELWLATIVELY_FLAGS_MDIV_RANGE_FULL    (0x00000001U)

/*
 * LW2080_CTRL_CMD_CLK_TEST_NDIV_SLOWDOWN
 *
 * This command can be used to test NDIV slowdown on the given clock domain, if
 * supported.
 *
 *   action
 *     Denotes desired action that the command should take.
 *
 *     Supported fields include:
 *       LW2080_CTRL_CLK_TEST_NDIV_SLOWDOWN_ACTION_VAL
 *
 *         When set to _FORCE_SLOWDOWN, NDIV slowdown is forcefully engaged or
 *         disengaged on the given clock domain, depending on whether the
 *         "data" field is set to LW_TRUE or LW_FALSE respectively.
 *
 *         When set to _PREVENT_RAMPUP, frequency ramp-up during normal
 *         programming of the given clock domain is disallowed or allowed after
 *         NDIV slowdown is engaged, depending on whether the "data" field is
 *         set to LW_TRUE or LW_FALSE respectively.
 *
 *   data
 *     Usage depends on "action" field. See the "action" field for details.
 *
 *   clkDomain
 *     Denotes the clock domain to test NDIV slowdown on.
 *
 * Possible status return values are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_NOT_SUPPORTED
 */
#define LW2080_CTRL_CMD_CLK_TEST_NDIV_SLOWDOWN                         (0x20801011) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_CLK_INTERFACE_ID << 8) | LW2080_CTRL_CLK_TEST_NDIV_SLOWDOWN_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_CLK_TEST_NDIV_SLOWDOWN_PARAMS_MESSAGE_ID (0x11U)

typedef struct LW2080_CTRL_CLK_TEST_NDIV_SLOWDOWN_PARAMS {
    LwU32 action;
    LwU32 data;
    LwU32 clkDomain;
} LW2080_CTRL_CLK_TEST_NDIV_SLOWDOWN_PARAMS;

#define LW2080_CTRL_CLK_TEST_NDIV_SLOWDOWN_ACTION_VAL                       31:0
#define LW2080_CTRL_CLK_TEST_NDIV_SLOWDOWN_ACTION_VAL_FORCE_SLOWDOWN (0x00000001U)
#define LW2080_CTRL_CLK_TEST_NDIV_SLOWDOWN_ACTION_VAL_PREVENT_RAMPUP (0x00000002U)

/*
 * LW2080_CTRL_CMD_CLK_NOISE_AWARE_PLL_CALIBRATION
 *
 * This command can be used to calibrate noise-aware pll if supported.
 * Calibration will happen at the voltage(s) based on the VDT entries(2)
 * present in noise-aware pll vbios table.
 *
 * If both the VDT entries are present/valid(!=0XFF), calibration happens
 * at both the voltages.
 *
 * If only one of the VDT entries is present/valid(!=0xFF), calibration
 * happens on only that voltage point.
 *
 * If none of the VDT entries are present/valid, calibration happens on
 * the current voltage point.
 *
 * Note: Lwrrently Noise-aware PLL feature is supported only on GPCPLL
 *
 *   clkSrc
 *      Denotes the pll to be calibrated.
 *
 * Possible status return values are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_NOT_SUPPORTED
 */

#define LW2080_CTRL_CMD_CLK_NOISE_AWARE_PLL_CALIBRATION              (0x20801012) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_CLK_INTERFACE_ID << 8) | LW2080_CTRL_CLK_NOISE_AWARE_PLL_CALIBRATION_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_CLK_NOISE_AWARE_PLL_CALIBRATION_PARAMS_MESSAGE_ID (0x12U)

typedef struct LW2080_CTRL_CLK_NOISE_AWARE_PLL_CALIBRATION_PARAMS {
    LwU32 clkSrc;
} LW2080_CTRL_CLK_NOISE_AWARE_PLL_CALIBRATION_PARAMS;


/*
 * LW2080_CTRL_CLK_CONFIG_INFO_V2
 *
 * This command configures the specified clock domains using the information
 * contained in the corresponding clock info structures.  Essentially, this
 * command computes the same clocks configuration as LW2080_CTRL_CMD_CLK_SET_INFO
 * without changing the hardware state, allowing the client to determine the
 * actual frequency that would result.
 *
 * Not all domains or chips support this operation.  When not supported, the
 * actual frequency is presumed to be the same as the target frequency.
 *
 *   flags
 *     Unused and reserved.  Should be set to zero before calling.
 *
 *   clkInfoListSize
 *     This field specifies the number of entries on the caller's clkInfoList.
 *
 *   clkInfoList
 *     This is an array which contains the desired clock information is to be
 *     retrieved.
 *     clkInfoListSize must be no larger than the size of this array divided by
 *     the size of the LW2080_CTRL_CLK_INFO structure.
 *
 *     clkInfoList[i].actualFreq is set to the frequency that would actually be
 *     programmed in lieu of the targetFreq, if known, or targetFreq otherwise.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_ILWALID_COMMAND
 *   LW_ERR_OPERATING_SYSTEM
 */
#define LW2080_CTRL_CMD_CLK_CONFIG_INFO_V2 (0x20801014) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_CLK_INTERFACE_ID << 8) | LW2080_CTRL_CLK_CONFIG_INFO_V2_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_CLK_CONFIG_INFO_V2_PARAMS_MESSAGE_ID (0x14U)

typedef struct LW2080_CTRL_CLK_CONFIG_INFO_V2_PARAMS {
    LwU32                flags;
    LwU32                clkInfoListSize;
    LW2080_CTRL_CLK_INFO clkInfoList[LW2080_CTRL_CLK_ARCH_MAX_DOMAINS];
} LW2080_CTRL_CLK_CONFIG_INFO_V2_PARAMS;

/*!
 * This struct represents the gpu monitoring sample of clock values.
 */
typedef struct LW2080_CTRL_CLK_GPUMON_CLOCK_SAMPLE {
    /*!
     * Base gpu monitoring sample.
     */
    LW_DECLARE_ALIGNED(LW2080_CTRL_GPUMON_SAMPLE base, 8);
    /*!
     * Clock samples.
     */
    LW2080_CTRL_CLK_INFO processorClkInfo;
    LW2080_CTRL_CLK_INFO memoryClkInfo;
} LW2080_CTRL_CLK_GPUMON_CLOCK_SAMPLE;

/*!
 * This struct represents the gpu monitoring samples of clock values that
 * client wants the access to.
 */
typedef LW2080_CTRL_GPUMON_SAMPLES LW2080_CTRL_CLK_GPUMON_CLOCK_SAMPLES_PARAMS;

/*!
 * Number of GPU monitoring sample in their respective buffers.
 */
#define LW2080_CTRL_CLK_GPUMON_SAMPLE_COUNT_CLOCK       100U

#define LW2080_CTRL_CLK_GPUMON_CLOCK_BUFFER_SIZE         \
    (LW_SIZEOF32(LW2080_CTRL_CLK_GPUMON_CLOCK_SAMPLE) *  \
    LW2080_CTRL_CLK_GPUMON_SAMPLE_COUNT_CLOCK)


/*!
 * LW2080_CTRL_CMD_CLK_GET_GPUMON_CLOCK_SAMPLES_V2
 *
 * This command returns gpu monitoring clock samples.
 *
 * Possible status values returned are
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW2080_CTRL_CMD_CLK_GET_GPUMON_CLOCK_SAMPLES_V2 (0x20801040) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_CLK_INTERFACE_ID << 8) | LW2080_CTRL_CLK_GPUMON_CLOCK_GET_SAMPLES_V2_PARAMS_MESSAGE_ID" */

/*!
 * This struct represents the GPU monitoring samples of clock values that
 * client wants the access to.
 */
#define LW2080_CTRL_CLK_GPUMON_CLOCK_GET_SAMPLES_V2_PARAMS_MESSAGE_ID (0x40U)

typedef struct LW2080_CTRL_CLK_GPUMON_CLOCK_GET_SAMPLES_V2_PARAMS {
    /*!
    * Type of the sample, see LW2080_CTRL_GPUMON_SAMPLE_TYPE_* for reference.
    */
    LwU8  type;
    /*!
    * tracks the offset of the tail in the cirlwlar queue array pSamples.
    */
    LwU32 tracker;
    /*!
    * A cirlwlar queue with size == bufSize.
    *
    * @note This cirlwlar queue wraps around after 10 seconds of sampling,
    * and it is clients' responsibility to query within this time frame in
    * order to avoid losing samples.
    * @note With one exception, this queue contains last 10 seconds of samples
    * with tracker poiniting to oldest entry and entry before tracker as the
    * newest entry. Exception is when queue is not full (i.e. tracker is
    * pointing to a zeroed out entry), in that case valid entries are between 0
    * and tracker.
    * @note Clients can store tracker from previous query in order to provide
    * samples since last read.
    */
    LW_DECLARE_ALIGNED(LW2080_CTRL_CLK_GPUMON_CLOCK_SAMPLE samples[LW2080_CTRL_CLK_GPUMON_SAMPLE_COUNT_CLOCK], 8);
} LW2080_CTRL_CLK_GPUMON_CLOCK_GET_SAMPLES_V2_PARAMS;

/*
 * LW2080_CTRL_CMD_CLK_GET_PLL_COUNT
 *
 * This command is used to get the PLL mask for a particular clock domain.
 *
 *
 *   clkDomain[in]
 *     The clock domain for which we want to find the number of active PLLs.
 *     For more information on clkDomain see LW2080_CTRL_CLK_GET_DOMAINS above.
 *     This API will fail if you pass down any domain that's not supported on the chip.
 *   pllCount[out]
 *     This field returns the active PLL count for the given clkDomain.
 *   reserved
 *     This space is reserved for compatibility in future api extensions.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW2080_CTRL_CMD_CLK_GET_PLL_COUNT (0x20801016) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_CLK_INTERFACE_ID << 8) | LW2080_CTRL_CLK_GET_PLL_COUNT_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_CLK_GET_PLL_COUNT_PARAMS_MESSAGE_ID (0x16U)

typedef struct LW2080_CTRL_CLK_GET_PLL_COUNT_PARAMS {
    LwU32 clkDomain;
    LwU32 pllCount;
    LwU32 reserved[14];
} LW2080_CTRL_CLK_GET_PLL_COUNT_PARAMS;

/*
 * LW2080_CTRL_CLK_MEASURE_PLL_FREQ
 *
 * This command is used to callwlate the real frequency of any particular PLL available for a
 * clock by using clock counters. Note that this API does NOT measure the PLL output, but
 * the ultimate clock output at the lowest point of the stream.
 *   clkDomain[in]
 *     The clock domain for which we want to measure the frequency.
 *     For more information on clkDomain see LW2080_CTRL_CLK_GET_DOMAINS above.
 *     This API will fail if you pass down any domain that's not supported on the chip.
 *   pllIdx[in]
 *     Index of the PLL
 *   freqKHz[out]
 *     This field returns the callwlated frequency in units of KHz.
 *   reserved
 *     This space is reserved for compatibility in future api extensions.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_ILWALID_COMMAND
 *   LW_ERR_OPERATING_SYSTEM
 */
#define LW2080_CTRL_CMD_CLK_MEASURE_PLL_FREQ (0x20801017) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_CLK_INTERFACE_ID << 8) | LW2080_CTRL_CLK_MEASURE_PLL_FREQ_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_CLK_MEASURE_PLL_FREQ_PARAMS_MESSAGE_ID (0x17U)

typedef struct LW2080_CTRL_CLK_MEASURE_PLL_FREQ_PARAMS {
    LwU32 clkDomain;
    LwU32 pllIdx;
    LwU32 freqKHz;
    LwU32 reserved[13];
} LW2080_CTRL_CLK_MEASURE_PLL_FREQ_PARAMS;

/*
 * LW2080_CTRL_CMD_CLK_CNTR_MEASURE_AVG_FREQ
 *
 * This command is used to callwlate the real frequency of a clock by using
 * clock counters. Note that this API does NOT measure the PLL output, but
 * the ultimate clock output at the lowest point of the stream.
 *
 * This RMCTRL works on differential samples of a continuously incrementing
 * counter to compute the average frequency over the time period.  The caller
 * provides the last readings against which the difference will be taken.
 *
 * avgFreq = (tickCnt' - tickCnt) / (timens' - timens)
 *
 * This is in contrast to @ref LW2080_CTRL_CMD_CLK_MEASURE_FREQ which measures
 * the frequency over a blocking 40us window.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_ILWALID_COMMAND
 *   LW_ERR_OPERATING_SYSTEM
 */
#define LW2080_CTRL_CMD_CLK_CNTR_MEASURE_AVG_FREQ (0x20801018) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_CLK_INTERFACE_ID << 8) | LW2080_CTRL_CLK_CNTR_MEASURE_AVG_FREQ_PARAMS_MESSAGE_ID" */

/*!
 * Structure describing the state of a clock counter when it was sampled -
 * i.e. the current tick count and the time at which it was sampled.  By
 * comparing two of these structures, the average frequency of the counted clock
 * over the corresponding time period can be callwlated:
 *
 * avgFreq = (tickCnt' - tickCnt) / (timens' - timens)
 *
 * GIRISH TO-DO - replace all instances of LW2080_CTRL_CLK_CNTR_SAMPLE with a
 *                32-bit aligned type
 */
typedef struct LW2080_CTRL_CLK_CNTR_SAMPLE {
    /*!
     * Timestamp of the sample.
     */
    LW_DECLARE_ALIGNED(LwU64 timens, 8);
    /*!
     * Sampled clock counter value.
     */
    LW_DECLARE_ALIGNED(LwU64 tickCnt, 8);
} LW2080_CTRL_CLK_CNTR_SAMPLE;

/*!
 * @ref LW2080_CTRL_CLK_CNTR_SAMPLE. This structure is 32 bit aligned.
 */
typedef struct LW2080_CTRL_CLK_CNTR_SAMPLE_ALIGNED {
    /*!
     * Timestamp of the sample.
     */
    LwU64_ALIGN32 timens;
    /*!
     * Sampled clock counter value.
     */
    LwU64_ALIGN32 tickCnt;
} LW2080_CTRL_CLK_CNTR_SAMPLE_ALIGNED;

/*!
 * Structure holding partition specific clock counter data.
 */
typedef struct LW2080_CTRL_CLK_CNTR_PART_AVG_FREQ {
    /*!
     * [in] Partition index @ref LW2080_CTRL_CLK_DOMAIN_PART_IDX_<xyz>
     * LW2080_CTRL_CLK_DOMAIN_PART_IDX_UNDEFINED should be used for
     * the broadcast reads or for the clock domains which do not
     * have partitions.
     */
    LwU8  partIdx;
    /*!
     * [in/out] Caller-provided previous sample against which the current values
     * should be compared to compute the frequency.  After the average frequency
     * is computed, the RMCTRL will store the current value in this buffer.
     */
    LW_DECLARE_ALIGNED(LW2080_CTRL_CLK_CNTR_SAMPLE sample, 8);
    /*!
     * [out] This field returns the callwlated frequency in units of KHz.
     */
    LwU32 freqkHz;
} LW2080_CTRL_CLK_CNTR_PART_AVG_FREQ;

/*!
 * Parameter structure for @ref LW2080_CTRL_CMD_CLK_CNTR_MEASURE_AVG_FREQ.
 */
#define LW2080_CTRL_CLK_CNTR_MEASURE_AVG_FREQ_PARAMS_MESSAGE_ID (0x18U)

typedef struct LW2080_CTRL_CLK_CNTR_MEASURE_AVG_FREQ_PARAMS {
    /*!
     * [in] The clock domain for which we want to measure the frequency.  For more
     * information on clkDomain see LW2080_CTRL_CLK_GET_DOMAINS above.  This API
     * will fail if you pass down any domain that's not supported on the chip.
     */
    LwU32  clkDomain;
    /*!
     * [in] Boolean to check if 32 bit counter support is needed.
     * When this flag is true, RM uses 32 LSBs of counter
     * value for frequency callwlation and also pass that value
     * back in parts[i].sample.tickCnt.
     */
    LwBool b32BitCntr;
    /*!
     * [in] Partition count which is the actual size of the array passed.
     * Mask of available paritions for a clock domain should be queried
     * through LW2080_CTRL_CMD_CLK_CLK_DOMAINS_GET_INFO api.
     */
    LwU8   numParts;
    /*!
     * [in/out] Array of partition specific clock counter data.
     */
    LW_DECLARE_ALIGNED(LW2080_CTRL_CLK_CNTR_PART_AVG_FREQ parts[LW2080_CTRL_CLK_DOMAIN_PART_IDX_MAX], 8);
} LW2080_CTRL_CLK_CNTR_MEASURE_AVG_FREQ_PARAMS;

/*!
 * Enumeration of clients which can disable/enable clock counters or
 * clock domains.
 *
 * 0x00 - RM          - External RMCTRL clients (tests, etc)
 * 0x01 - PMU         - Internal PMU clock cases
 * 0x02 - MSCG        - Entering/Exiting MSCG
 * 0x03 - GPCRG       - Entering/Exiting GPC-RG
 * 0x04 - DIFR_SW_ASR - Entering/Exiting DIFR_SW_ASR
 * 0x05 - DIFR_CG     - Entering/Exiting DIFR_CG
 * 0x06 - Maximum number of valid clients - SHOULD ALWAYS BE LAST
 */
#define LW2080_CTRL_CLK_CLIENT_ID_RM                            (0x00U)
#define LW2080_CTRL_CLK_CLIENT_ID_PMU                           (0x01U)
#define LW2080_CTRL_CLK_CLIENT_ID_MSCG                          (0x02U)
#define LW2080_CTRL_CLK_CLIENT_ID_GR_RG                         (0x03U)
#define LW2080_CTRL_CLK_CLIENT_ID_DIFR_SW_ASR                   (0x04U)
#define LW2080_CTRL_CLK_CLIENT_ID_DIFR_CG                       (0x05U)
#define LW2080_CTRL_CLK_CLIENT_ID_MAX_NUM                       (0x06U)

/*!
 * Mask of all valid clients that are listed - @ref
 * LW2080_CTRL_CLK_CLIENT_<xyz>
 */
#define LW2080_CTRL_CLK_CLIENT_ID_VALID_MASK                 \
            (LWBIT(LW2080_CTRL_CLK_CLIENT_ID_MAX_NUM) - 1)

/*!
 * Reset value of the client ID mask - @ref LW2080_CTRL_CLK_CLIENT_<xyz>
 */
#define LW2080_CTRL_CLK_CLIENT_ID_MASK_RESET_VALUE              (0x0U)

/*!
 * Enumeration of CLK objects' VOLTAGE_TYPEs.
 *
 * _POR - The voltage value which will be used in the POR.  Rounded to the
 *     nearest step supported on the given voltage rail.
 * _SOURCE - The voltage value which was used to source a given VF point.  Not
 *     rounded to the nerest step supported on the given voltage rail.
 */
#define LW2080_CTRL_CLK_VOLTAGE_TYPE_POR                        0x00U
#define LW2080_CTRL_CLK_VOLTAGE_TYPE_SOURCE                     0x01U

/*!
 * Special value representing an uninitialized/invalid CLK_ARB value.
 */
#define LW2080_CTRL_CLK_VF_VALUE_ILWALID                        0x0U

/*!
 * Default behavior for VF point mapping bit-field definitions. These define the
 * semantics for how the CLK code should choose the output value when NO
 * CLK_VF_POINT matches the specified input value.
 *
 * _NO   - No default VF point is set in case of no match.
 * _YES  - If the input value is out of range then we try to find the closest
 *         match semantically.
 *         For Volt -> Freq: If the input voltage is less than the lowest voltage
 *         in the VF lwrve, then we pick the first point in the VF lwrve as
 *         default value
 *         For Freq -> Volt: If the input voltage is more than the highest voltage
 *         in the VF lwrve, then we pick the last point in the VF lwrve as
 *         default value.
 *         ***********************************************************
 *         *                                                         *
 *         *                        WARNING                          *
 *         *                                                         *
 *         ***********************************************************
 *         NOTE:
 *         If the input value is outside the VF range - For Volt -> Freq less
 *         than the minimum volt and for Freq -> Volt more than the maximum freq,
 *         then we could potentially return an invalid output, so use this flag
 *         only if you are sure this is what you want.
 *         Ex: For VF lwrve 1V-> 1GHz, 1,1V -> 1.1GHz
 *              with *_DEFAULT_VF_POINT_SET_TRUE
 *         If Input Volt = 0.9V then output freq = 1GHz resulting in
 *              0.9V -> 1GHz VF point which is invalid
 *         If Input Freq = 1.2GHz then output volt = 1.1V resulting in
 *              1.1V -> 1.2GHz which is also invalid
 */
#define LW2080_CTRL_CLK_VF_INPUT_FLAGS_VF_POINT_SET_DEFAULT          0:0
#define LW2080_CTRL_CLK_VF_INPUT_FLAGS_VF_POINT_SET_DEFAULT_NO  0x00U
#define LW2080_CTRL_CLK_VF_INPUT_FLAGS_VF_POINT_SET_DEFAULT_YES 0x01U

/*!
 *  Macro to initialize the Default value of VF Flags
 */
#define LW2080_CTRL_CLK_VF_INPUT_FLAGS_INIT                             \
    DRF_DEF(2080_CTRL_CLK, _VF_INPUT_FLAGS, _VF_POINT_SET_DEFAULT, _NO)

/*!
 * Structure specifying input ranges for CLK_ARB look-up functions.
 */
typedef struct LW2080_CTRL_CLK_VF_INPUT_RANGE {
    /*!
     * Minimum range value.
     */
    LwU32 milwalue;
    /*!
     * Maximum range value.
     */
    LwU32 maxValue;
} LW2080_CTRL_CLK_VF_INPUT_RANGE;
typedef struct LW2080_CTRL_CLK_VF_INPUT_RANGE *PLW2080_CTRL_CLK_VF_INPUT_RANGE;

/*!
 * Helper macro to init a LW2080_CTRL_CLK_VF_INPUT_RANGE structure.
 * Initializes the structure to default/unmatched values. Intended to be used
 * as an assignment.
 */
#define LW2080_CTRL_CLK_VF_INPUT_RANGE_INIT(pInputRange)                    \
do {                                                                        \
    (pInputRange)->milwalue    = (LwU32)LW2080_CTRL_CLK_VF_VALUE_ILWALID;   \
    (pInputRange)->maxValue    = (LwU32)LW2080_CTRL_CLK_VF_VALUE_ILWALID;   \
} while (LW_FALSE)

/*!
 * Structure specifying input value for CLK_ARB look-up functions. This object
 * is used to pass a single Voltage/Frequency value as opposed to a
 * range (Min and Max) passed in LW2080_CTRL_CLK_VF_INPUT_RANGE
 */
typedef struct LW2080_CTRL_CLK_VF_INPUT {
    /*!
     * Additional flags @ref LW2080_CTRL_CLK_VF_INPUT_FLAGS_<XYZ>
     */
    LwU8  flags;
    /*!
     * Input value for VF lookup
     */
    LwU32 value;
} LW2080_CTRL_CLK_VF_INPUT;
typedef struct LW2080_CTRL_CLK_VF_INPUT *PLW2080_CTRL_CLK_VF_INPUT;

/*!
 * Helper macro to init a LW2080_CTRL_CLK_VF_INPUT structure.
 * Initializes the structure to default/unmatched values.  Intended to be used
 * as an assignment.
 */
#define LW2080_CTRL_CLK_VF_INPUT_INIT(pInput)                                   \
do {                                                                            \
    (pInput)->flags = (LwU8)LW2080_CTRL_CLK_VF_INPUT_FLAGS_INIT;                \
    (pInput)->value = (LwU32)LW2080_CTRL_CLK_VF_VALUE_ILWALID;                  \
} while (LW_FALSE)

/*!
 * Structure specifying output ranges for CLK_ARB look-up functions.
 */
typedef struct LW2080_CTRL_CLK_VF_OUTPUT_RANGE {
    /*!
     * The "best" (i.e. closest) value matching the input @ref
     * LW2080_CTRL_CLK_VF_INPUT_RANGE::milwalue which provided @milwalue below.
     */
    LwU32 minInputBestMatch;
    /*!
     * The minimum output value corresponding to the @ref
     * LW2080_CTRL_CLK_VF_INPUT_RANGE::milwalue.
     */
    LwU32 milwalue;
    /*!
     * The "best" (i.e. closest) value matching the input @ref
     * LW2080_CTRL_CLK_VF_INPUT_RANGE::maxValue which provided @maxValue below.
     */
    LwU32 maxInputBestMatch;
    /*!
     * The maximum output value corresponding to the @ref
     * LW2080_CTRL_CLK_VF_INPUT_RANGE::maxValue.
     */
    LwU32 maxValue;
} LW2080_CTRL_CLK_VF_OUTPUT_RANGE;
typedef struct LW2080_CTRL_CLK_VF_OUTPUT_RANGE *PLW2080_CTRL_CLK_VF_OUTPUT_RANGE;

/*!
 * Helper macro to init a LW2080_CTRL_CLK_VF_OUTPUT_RANGE structure.
 * Initializes the structure to default/unmatched values.  Intended to be used
 * as an assignment.
 */
#define LW2080_CTRL_CLK_VF_OUTPUT_RANGE_INIT(pOutputRange)                      \
do {                                                                            \
    (pOutputRange)->minInputBestMatch = LW2080_CTRL_CLK_VF_VALUE_ILWALID;       \
    (pOutputRange)->milwalue          = LW2080_CTRL_CLK_VF_VALUE_ILWALID;       \
    (pOutputRange)->maxInputBestMatch = LW2080_CTRL_CLK_VF_VALUE_ILWALID;       \
    (pOutputRange)->maxValue          = LW2080_CTRL_CLK_VF_VALUE_ILWALID;       \
} while (LW_FALSE)

/*!
 * Structure specifying output value for CLK_ARB look-up functions. This object
 * is used to return a single Voltage/Frequency value that matches the input
 * voltage/frequency passed in LW2080_CTRL_CLK_VF_INPUT
 */
typedef struct LW2080_CTRL_CLK_VF_OUTPUT {
    /*!
     * The "best" (i.e. closest) value matching the input @ref
     * LW2080_CTRL_CLK_VF_INPUT::value which provided @value below.
     */
    LwU32 inputBestMatch;
    /*!
     * The output value corresponding to the @ref
     * LW2080_CTRL_CLK_VF_INPUT::value.
     */
    LwU32 value;
} LW2080_CTRL_CLK_VF_OUTPUT;
typedef struct LW2080_CTRL_CLK_VF_OUTPUT *PLW2080_CTRL_CLK_VF_OUTPUT;

/*!
 * Helper macro to init a LW2080_CTRL_CLK_VF_OUTPUT structure.
 * Initializes the structure to default/unmatched values.  Intended to be used
 * as an assignment.
 */
#define LW2080_CTRL_CLK_VF_OUTPUT_INIT(pOutput)                                 \
do {                                                                            \
    (pOutput)->inputBestMatch   = LW2080_CTRL_CLK_VF_VALUE_ILWALID;             \
    (pOutput)->value            = LW2080_CTRL_CLK_VF_VALUE_ILWALID;             \
} while (LW_FALSE)

/*!
 * Structure describing a voltage-frequency (VF) pair.  This structure is used
 * to pass voltage and frequency pairs as input/output parameters in the RM and
 * PMU CLK functions.
 */
typedef struct LW2080_CTRL_CLK_VF_PAIR {
    /*!
     * Frequency (MHz).
     */
    LwU16 freqMHz;
    /*!
     * Voltage (uV)
     */
    LwU32 voltageuV;
} LW2080_CTRL_CLK_VF_PAIR;
typedef struct LW2080_CTRL_CLK_VF_PAIR *PLW2080_CTRL_CLK_VF_PAIR;

/*!
 * Accessor macro for @refLW2080_CTRL_CLK_VF_PAIR::freqMHz
 *
 * @param[in] pVFPair   LW2080_CTRL_CLK_VF_PAIR pointer
 *
 * @return @refLW2080_CTRL_CLK_VF_PAIR::freqMHz
 */
#define LW2080_CTRL_CLK_VF_PAIR_FREQ_MHZ_GET(pVFPair)                          \
    ((pVFPair)->freqMHz)

/*!
 * Mutator macro for @refLW2080_CTRL_CLK_VF_PAIR::freqMHz
 *
 * @param[in] pVFPair   LW2080_CTRL_CLK_VF_PAIR pointer
 * @param[in] _freqMHz  Frequency (MHz) to set
 */
#define LW2080_CTRL_CLK_VF_PAIR_FREQ_MHZ_SET(pVFPair, _freqMHz)                \
do {                                                                           \
    ((pVFPair)->freqMHz) = (_freqMHz);                                         \
} while (LW_FALSE)

/*!
 * Accessor macro for @refLW2080_CTRL_CLK_VF_PAIR::voltageuV
 *
 * @param[in] pVFPair   LW2080_CTRL_CLK_VF_PAIR pointer
 *
 * @return @refLW2080_CTRL_CLK_VF_PAIR::voltageuV
 */
#define LW2080_CTRL_CLK_VF_PAIR_VOLTAGE_UV_GET(pVFPair)                        \
    ((pVFPair)->voltageuV)

/*!
 * Mutator macro for @refLW2080_CTRL_CLK_VF_PAIR::voltageuV
 *
 * @param[in] pVFPair     LW2080_CTRL_CLK_VF_PAIR pointer
 * @param[in] _voltageuV  Voltage (uV) to set
 */
#define LW2080_CTRL_CLK_VF_PAIR_VOLTAGE_UV_SET(pVFPair, _voltageuV)            \
do {                                                                           \
    ((pVFPair)->voltageuV) = (_voltageuV);                                     \
} while (LW_FALSE)

/*!
 * Macro to perform a MAX operation against a "last" VF_PAIR.  This is a common
 * operation when enforcing montonicity across the VF lwrve, where the RM/PMU
 * must ensure that all successive VF pairs are >= previous VF pairs on the
 * given VF lwrve.
 *
 * @param[in]     pVFPair
 *      Pointer to the "current" LW2080_CTRL_CLK_VF_PAIR structure which must be
 *      >= the last LW2080_CTRL_CLK_VF_PAIR structure (@ref pVFPairLast).
 * @param[in/out] pVFPairLast
 *      Pointer to the "last" LW2080_CTRL_CLK_VF_PAIR structure which will be
 *      used to floor the current LW2080_CTRL_CLK_VF_PAIR structure (@ref
 *      pVFPair).  After flooring, the current LW2080_CTRL_CLK_VF_PAIR structure
 *      will be copied into this structure.
 */
#define LW2080_CTRL_CLK_VF_PAIR_LAST_MAX(pVFPair, pVFPairLast)                 \
do {                                                                           \
    (pVFPair)->voltageuV = LW_MAX((pVFPair)->voltageuV,                        \
                            (pVFPairLast)->voltageuV);                         \
    (pVFPair)->freqMHz = LW_MAX((pVFPair)->freqMHz,                            \
                            (pVFPairLast)->freqMHz);                           \
} while (LW_FALSE)

/*!
 * Macro to perform a copy operation of source VF Pair to its destination the
 * corresponding VF pair.
 */
#define LW2080_CTRL_CLK_VF_PAIR_COPY(pVFPairSource, pVFPairDest)              \
do {                                                                          \
    *(pVFPairDest) = *(pVFPairSource);                                        \
} while (LW_FALSE)

/*!
 * Helper macro to init a LW2080_CTRL_CLK_VF_PAIR structure.
 * Intended to be used as an assignment.
 */
#define LW2080_CTRL_CLK_VF_PAIR_INIT()                                        \
    {                                                                         \
        /* .freqMHz    = */ 0,                                                \
        /* .voltageuV  = */ 0,                                                \
    }

/*!
 * Restrict the max allowed clock domain entries to some safe value.
 */
#define LW2080_CTRL_CLK_CLK_VF_POINT_FREQ_TUPLE_MAX_SIZE 0x5U

/*!
 * Structure describing a frequency tuple containing frequency value of
 * clock domains that belongs to same primary-secondary group.
 */
typedef struct LW2080_CTRL_CLK_CLK_VF_POINT_FREQ_TUPLE {
    /*!
     * Frequency of the Clock Domain.
     *
     * Secondary Clock Domain's frequency will be determined from their primary's
     * frequency based on primary -> secondary relationship.
     */
    LwU16 freqMHz;
} LW2080_CTRL_CLK_CLK_VF_POINT_FREQ_TUPLE;
typedef struct LW2080_CTRL_CLK_CLK_VF_POINT_FREQ_TUPLE *PLW2080_CTRL_CLK_CLK_VF_POINT_FREQ_TUPLE;

/*!
 * Structure describing a base VF tuple contain frequency values of
 * clock domains that belongs to same primary-secondary group and their
 * corresponding base voltage.
 */
typedef struct LW2080_CTRL_CLK_CLK_VF_POINT_BASE_VF_TUPLE {
    /*!
     * CPM Max Frequency offset MHz
     */
    LwU16                                   cpmMaxFreqOffsetMHz;

    /*!
     * @ref LW2080_CTRL_CLK_CLK_VF_POINT_FREQ_TUPLE
     */
    LW2080_CTRL_CLK_CLK_VF_POINT_FREQ_TUPLE freqTuple[LW2080_CTRL_CLK_CLK_VF_POINT_FREQ_TUPLE_MAX_SIZE];

    /*!
     * Voltage (uV)
     */
    LwU32                                   voltageuV;
} LW2080_CTRL_CLK_CLK_VF_POINT_BASE_VF_TUPLE;
typedef struct LW2080_CTRL_CLK_CLK_VF_POINT_BASE_VF_TUPLE *PLW2080_CTRL_CLK_CLK_VF_POINT_BASE_VF_TUPLE;

/*
 * Define _ILWALID value for DVCO Offset Code. This will be used to
 * check if client override the DVCO Offset Code.
 */
#define LW2080_CTRL_CLK_CLK_VF_POINT_DVCO_OFFSET_CODE_ILWALID 0xFFU

/*!
 * Structure describing a base VF tuple for secondary VF lwrve.
 */
typedef struct LW2080_CTRL_CLK_CLK_VF_POINT_BASE_VF_TUPLE_SEC {
    /*!
     * @ref LW2080_CTRL_CLK_CLK_VF_POINT_BASE_VF_TUPLE
     */
    LW2080_CTRL_CLK_CLK_VF_POINT_BASE_VF_TUPLE super;

    /*!
     * DVCO offset in terms of DVCO codes to trigger the fast slowdown while
     * HW switches from reference NDIV point to secondary NDIV.
     */
    LwU8                                       dvcoOffsetCode;
} LW2080_CTRL_CLK_CLK_VF_POINT_BASE_VF_TUPLE_SEC;
typedef struct LW2080_CTRL_CLK_CLK_VF_POINT_BASE_VF_TUPLE_SEC *PLW2080_CTRL_CLK_CLK_VF_POINT_BASE_VF_TUPLE_SEC;

/*!
 * Structure describing a volt-frequency tuple <Frequency, Volt>.
 */
typedef struct LW2080_CTRL_CLK_CLK_VF_POINT_VF_TUPLE {
    /*!
     * Frequency of the Clock Domain @ref clkDomIdx.
     *
     * Secondary Clock Domain's frequency will be determined from their primary's
     * frequency based on primary -> secondary relationship.
     */
    LwU16 freqMHz;

    /*!
     * Voltage of the Clock Domain @ref clkDomIdx.
     *
     * Secondary Clock Domain's voltage will be determined from their primary's
     * voltage based on primary -> secondary relationship.
     */
    LwU32 voltageuV;
} LW2080_CTRL_CLK_CLK_VF_POINT_VF_TUPLE;
typedef struct LW2080_CTRL_CLK_CLK_VF_POINT_VF_TUPLE *PLW2080_CTRL_CLK_CLK_VF_POINT_VF_TUPLE;

/*!
 * Structure describing an offset volt-frequency tuple <Frequency, Volt>.
 */
typedef struct LW2080_CTRL_CLK_OFFSET_VF_TUPLE {
    /*!
     * Frequency of the Clock Domain (signed) @ref clkDomIdx.
     *
     * Secondary Clock Domain's frequency will be determined from their primary's
     * frequency based on primary -> secondary relationship.
     */
    LwS16 freqMHz;

    /*!
     * Voltage of the Clock Domain (signed) @ref clkDomIdx.
     *
     * Secondary Clock Domain's voltage will be determined from their primary's
     * voltage based on primary -> secondary relationship.
     */
    LwS32 voltageuV;
} LW2080_CTRL_CLK_OFFSET_VF_TUPLE;
typedef struct LW2080_CTRL_CLK_OFFSET_VF_TUPLE *PLW2080_CTRL_CLK_OFFSET_VF_TUPLE;

/*!
 * Macro defining different versions of LUT VF frequency tuples.
 */
#define LW2080_CTRL_CLK_CLK_VF_POINT_LUT_FREQ_TUPLE_VERSION_10 0x10U
#define LW2080_CTRL_CLK_CLK_VF_POINT_LUT_FREQ_TUPLE_VERSION_20 0x20U
#define LW2080_CTRL_CLK_CLK_VF_POINT_LUT_FREQ_TUPLE_VERSION_30 0x30U

/*!
 * Structure describing LUT VF frequency tuple for primary VF lwrve.
 */
typedef struct LW2080_CTRL_CLK_CLK_VF_POINT_LUT_FREQ_TUPLE_PRI {
    /*!
     * Frequency of primary VF lwrve (Reference frequency) which will be used
     * to callwlate primary NDIV to be programmed in HW LUT.
     */
    LwU16 freqMHz;
    /*!
     * CPM Max Frequency offset MHz
     * Enabled only with version _VERSION_30. Set to ZERO on previous versions.
     */
    LwU16 cpmMaxFreqOffsetMHz;
} LW2080_CTRL_CLK_CLK_VF_POINT_LUT_FREQ_TUPLE_PRI;
typedef struct LW2080_CTRL_CLK_CLK_VF_POINT_LUT_FREQ_TUPLE_PRI *PLW2080_CTRL_CLK_CLK_VF_POINT_LUT_FREQ_TUPLE_PRI;

/*!
 * Structure describing LUT VF frequency tuple for secondary VF lwrve
 */
typedef struct LW2080_CTRL_CLK_CLK_VF_POINT_LUT_FREQ_TUPLE_SEC {
    /*!
     * Frequency of secondary VF lwrve (secondary frequency) which will be used
     * to callwlate NDIV offset from primary NDIV to be programmed in HW LUT.
     */
    LwU16 freqMHz;

    /*!
     * DVCO offset in terms of DVCO codes to trigger the fast slowdown while
     * HW switches from reference NDIV point to secondary NDIV.
     */
    LwU8  dvcoOffsetCode;
} LW2080_CTRL_CLK_CLK_VF_POINT_LUT_FREQ_TUPLE_SEC;
typedef struct LW2080_CTRL_CLK_CLK_VF_POINT_LUT_FREQ_TUPLE_SEC *PLW2080_CTRL_CLK_CLK_VF_POINT_LUT_FREQ_TUPLE_SEC;

/*!
 * Structure describing LUT VF frequency tuple for version 10.
 */
typedef struct LW2080_CTRL_CLK_CLK_VF_POINT_LUT_FREQ_TUPLE_10 {
    /*!
     * @ref LW2080_CTRL_CLK_CLK_VF_POINT_LUT_FREQ_TUPLE_PRI
     */
    LW2080_CTRL_CLK_CLK_VF_POINT_LUT_FREQ_TUPLE_PRI pri;
} LW2080_CTRL_CLK_CLK_VF_POINT_LUT_FREQ_TUPLE_10;
typedef struct LW2080_CTRL_CLK_CLK_VF_POINT_LUT_FREQ_TUPLE_10 *PLW2080_CTRL_CLK_CLK_VF_POINT_LUT_FREQ_TUPLE_10;

/*!
 * Maximum possible number of secondary VF entries for a _35_PRIMARY class.
 */
#define LW2080_CTRL_CLK_CLK_PROG_35_PRIMARY_SEC_VF_ENTRY_VOLTRAIL_MAX 0x2U

/*!
 * Structure describing LUT VF frequency tuple for version 20.
 */
typedef struct LW2080_CTRL_CLK_CLK_VF_POINT_LUT_FREQ_TUPLE_20 {
    /*!
     * @ref LW2080_CTRL_CLK_CLK_VF_POINT_LUT_FREQ_TUPLE_PRI
     */
    LW2080_CTRL_CLK_CLK_VF_POINT_LUT_FREQ_TUPLE_PRI pri;

    /*!
     * @ref LW2080_CTRL_CLK_CLK_VF_POINT_LUT_FREQ_TUPLE_SEC
     */
    LW2080_CTRL_CLK_CLK_VF_POINT_LUT_FREQ_TUPLE_SEC sec[LW2080_CTRL_CLK_CLK_PROG_35_PRIMARY_SEC_VF_ENTRY_VOLTRAIL_MAX];
} LW2080_CTRL_CLK_CLK_VF_POINT_LUT_FREQ_TUPLE_20;
typedef struct LW2080_CTRL_CLK_CLK_VF_POINT_LUT_FREQ_TUPLE_20 *PLW2080_CTRL_CLK_CLK_VF_POINT_LUT_FREQ_TUPLE_20;

/*!
 * Structure describing LUT VF frequency tuple for version 20.
 */
typedef struct LW2080_CTRL_CLK_CLK_VF_POINT_LUT_FREQ_TUPLE_30 {
    /*!
     * @ref LW2080_CTRL_CLK_CLK_VF_POINT_LUT_FREQ_TUPLE_PRI
     */
    LW2080_CTRL_CLK_CLK_VF_POINT_LUT_FREQ_TUPLE_PRI pri;

    /*!
     * @ref LW2080_CTRL_CLK_CLK_VF_POINT_LUT_FREQ_TUPLE_SEC
     */
    LW2080_CTRL_CLK_CLK_VF_POINT_LUT_FREQ_TUPLE_SEC sec[LW2080_CTRL_CLK_CLK_PROG_35_PRIMARY_SEC_VF_ENTRY_VOLTRAIL_MAX];
} LW2080_CTRL_CLK_CLK_VF_POINT_LUT_FREQ_TUPLE_30;
typedef struct LW2080_CTRL_CLK_CLK_VF_POINT_LUT_FREQ_TUPLE_30 *PLW2080_CTRL_CLK_CLK_VF_POINT_LUT_FREQ_TUPLE_30;

/*!
 * Structure describing union of version based LUT VF entries.
 */


/*!
 * Structure describing LUT VF frequency tuple. NAFLL code will use this
 * tuple to perform VF look up for programming HW LUT.
 */
typedef struct LW2080_CTRL_CLK_CLK_VF_POINT_LUT_FREQ_TUPLE {
    /*!
     * version of LUT frequency tuple to be read / write.
     */
    LwU8  version;

    /*!
     * The "best" (i.e. closest) value matching the input @ref
     * LW2080_CTRL_CLK_VF_INPUT::value (voltage) which provided @data below.
     */
    LwU32 inputBestMatch;

    /*!
     * version-specific data union.
     */
    union {
        LW2080_CTRL_CLK_CLK_VF_POINT_LUT_FREQ_TUPLE_10 v10;
        LW2080_CTRL_CLK_CLK_VF_POINT_LUT_FREQ_TUPLE_20 v20;
        LW2080_CTRL_CLK_CLK_VF_POINT_LUT_FREQ_TUPLE_30 v30;
    } data;
} LW2080_CTRL_CLK_CLK_VF_POINT_LUT_FREQ_TUPLE;
typedef struct LW2080_CTRL_CLK_CLK_VF_POINT_LUT_FREQ_TUPLE *PLW2080_CTRL_CLK_CLK_VF_POINT_LUT_FREQ_TUPLE;

/*!
 * Structure representing the iteration state of a VF look-up.  This can be used
 * to "resume" a look-up from where the last look-up finished, allowing
 * optimized look-ups when searching in order of increasing input criteria.
 */
typedef struct LW2080_CTRL_CLK_VF_ITERATION_STATE {
    /*!
     * Index of last CLK_PROG retrieved by the iteration.
     */
    LwU8                    clkProgIdx;
    /*!
     * Index of last CLK_VF_POINT retrieved by the iteration.
     */
    LwBoardObjIdx           clkVfPointIdx;
    /*!
     * The last VF_PAIR (voltage and frequency value) retrieved by the
     * iteration.
     */
    LW2080_CTRL_CLK_VF_PAIR vfPair;
} LW2080_CTRL_CLK_VF_ITERATION_STATE;
typedef struct LW2080_CTRL_CLK_VF_ITERATION_STATE *PLW2080_CTRL_CLK_VF_ITERATION_STATE;

/*!
 * Helper macro to init a LW2080_CTRL_CLK_VF_ITERATION_STATE structure.
 * Intended to be used as an assignment.
 */
#define LW2080_CTRL_CLK_VF_ITERATION_STATE_INIT()                             \
    {                                                                         \
        /* .clkProgIdx    = */ 0U,                                            \
        /* .clkVfPointIdx = */ 0U,                                            \
        /* .vfPair        = */ { 0 },                                         \
    }


/*!
 * Maximum number of CLK_DOMAINs which are exported to a RMCTRL client.
 * Compile time sanity check is performed on this value.
 */
#define LW2080_CTRL_CLK_CLK_DOMAIN_CLIENT_MAX_DOMAINS          16U

/*!
 * Special define to represent an invalid CLK_DOMAIN index.
 */
#define LW2080_CTRL_CLK_CLK_DOMAIN_INDEX_ILWALID               LW2080_CTRL_BOARDOBJ_IDX_ILWALID_8BIT

/*!
 * Enumeration of the CLK_DOMAIN feature version.
 *
 * _2X - Legacy implementation of CLK_DOMAIN used in pstates 2.0 and earlier.
 * _3X - PP-TODO : Temporary mapped to _30 for backward compatibility
 * _30 - Represent PSTATE 3.0
 * _35 - Represent PSTATE 3.5
 * _40 - Represent PSTATE 4.0+
 */
#define LW2080_CTRL_CLK_CLK_DOMAIN_VERSION_ILWALID             0x00U
#define LW2080_CTRL_CLK_CLK_DOMAIN_VERSION_2X                  0x20U
#define LW2080_CTRL_CLK_CLK_DOMAIN_VERSION_3X                  0x30U
#define LW2080_CTRL_CLK_CLK_DOMAIN_VERSION_30                  0x30U
#define LW2080_CTRL_CLK_CLK_DOMAIN_VERSION_35                  0x35U
#define LW2080_CTRL_CLK_CLK_DOMAIN_VERSION_40                  0x40U

/*!
 * Enumerations of the CLK_DOMAIN HALs, as specified by the VBIOS Clocks Table.
 * These are the sets of Clock Domains which are implemented on a given chip.
 *
 * https://wiki.lwpu.com/engwiki/index.php/Resman/PState/Data_Tables/Clocks_Table/1.0_Spec#Clock_Domains
 */
#define LW2080_CTRL_CLK_CLK_DOMAIN_HAL_GM20X                   0x00U
#define LW2080_CTRL_CLK_CLK_DOMAIN_HAL_GP100                   0x01U
#define LW2080_CTRL_CLK_CLK_DOMAIN_HAL_GP10X                   0x02U
#define LW2080_CTRL_CLK_CLK_DOMAIN_HAL_GV100                   0x03U
#define LW2080_CTRL_CLK_CLK_DOMAIN_HAL_GH100                   0x04U
#define LW2080_CTRL_CLK_CLK_DOMAIN_HAL_GB10X                   0x05U
#define LW2080_CTRL_CLK_CLK_DOMAIN_HAL_ILWALID                 0xFFU

/*!
 * Enumeration of CLK_DOMAIN's BOARDOBJ class types.
 */
#define LW2080_CTRL_CLK_CLK_DOMAIN_TYPE_2X                     0x00U
#define LW2080_CTRL_CLK_CLK_DOMAIN_TYPE_3X                     0x01U
#define LW2080_CTRL_CLK_CLK_DOMAIN_TYPE_3X_FIXED               0x02U
#define LW2080_CTRL_CLK_CLK_DOMAIN_TYPE_3X_PROG                0x03U
#define LW2080_CTRL_CLK_CLK_DOMAIN_TYPE_30_PRIMARY             0x04U
#define LW2080_CTRL_CLK_CLK_DOMAIN_TYPE_30_SECONDARY           0x05U
#define LW2080_CTRL_CLK_CLK_DOMAIN_TYPE_30_PROG                0x06U
#define LW2080_CTRL_CLK_CLK_DOMAIN_TYPE_35_PRIMARY             0x07U
#define LW2080_CTRL_CLK_CLK_DOMAIN_TYPE_35_SECONDARY           0x08U
#define LW2080_CTRL_CLK_CLK_DOMAIN_TYPE_35_PROG                0x09U
#define LW2080_CTRL_CLK_CLK_DOMAIN_TYPE_40_PROG                0x0AU
#define LW2080_CTRL_CLK_CLK_DOMAIN_TYPE_UNKNOWN                0xFFU

// PP-TODO: Temporary mapping for Updating the RMCTRLs
#define LW2080_CTRL_CLK_CLK_DOMAIN_TYPE_3X_PRIMARY             0x0BU
#define LW2080_CTRL_CLK_CLK_DOMAIN_TYPE_3X_SECONDARY           0x0LW

/*!
 * Enumeration of CLK_DOMAIN's INTERFACE types.
 */
#define LW2080_CTRL_CLK_CLK_DOMAIN_INTERFACE_TYPE_ILWALID      0x00U
#define LW2080_CTRL_CLK_CLK_DOMAIN_INTERFACE_TYPE_3X_PRIMARY   0x01U
#define LW2080_CTRL_CLK_CLK_DOMAIN_INTERFACE_TYPE_3X_SECONDARY 0x02U
#define LW2080_CTRL_CLK_CLK_DOMAIN_INTERFACE_TYPE_PROG         0x03U

/*!
 * Structure describing CLK_DOMAIN_2X static information/POR.
 */
typedef struct LW2080_CTRL_CLK_CLK_DOMAIN_INFO_2X {
    LwU8   perfDomainGrpIdx;   //<! Saved by perf object for fast access.
    LwU32  ratioDomain;  //<! Primary domain - @ref LW2080_CTRL_CLK_DOMAIN_<xyz>.
    LwU8   usage;   //<! CLK_DOMAIN_USAGE in 8 bits.
    LwU8   defaultRatio;   //<! The P-state Clock Range Table Ratio
                                //<! Sub-Entry will override this percentage.
    LwBool bStayAbovePstate; //<! Must stay above the nominal value for the current P-state.
    LwBool bAllowNdivSliding; //<! Allow NDIV sliding for the PLL associated with this domain.
    LwU8   constrainedMdiv;   //<! Constrained MDIV coeff to use for NDIV sliding (0 = not-constrained).
} LW2080_CTRL_CLK_CLK_DOMAIN_INFO_2X;

/*!
 * Structure describing CLK_DOMAIN_PROG static information/POR.
 */
typedef struct LW2080_CTRL_CLK_CLK_DOMAIN_INFO_PROG {
    /*!
     * BOARDOBJ_INTERFACE super class.  Must always be first element in the structure.
     */
    LW2080_CTRL_BOARDOBJ_INTERFACE   super;
    /*!
     * Mask of VOLT_RAILs on which the given CLK_DOMAIN_PROG has a
     * Vmin, and for which a VF lwrve has been specified.
     */
    LW2080_CTRL_BOARDOBJGRP_MASK_E32 voltRailVminMask;
} LW2080_CTRL_CLK_CLK_DOMAIN_INFO_PROG;
typedef struct LW2080_CTRL_CLK_CLK_DOMAIN_INFO_PROG *PLW2080_CTRL_CLK_CLK_DOMAIN_INFO_PROG;

/*!
 * Structure describing CLK_DOMAIN_3X static information/POR.
 */
typedef struct LW2080_CTRL_CLK_CLK_DOMAIN_INFO_3X {
    /*!
     * Boolean specifying whether a CLK_DOMAIN is capable of being generated by
     * a voltage-noise-aware generator such as NAFLL.
     */
    LwBool bNoiseAwareCapable;
    /*!
     * Index of corresponding CLIENT_CLK_DOMAIN object, if specified.
     * @ref LW2080_CTRL_CLK_CLIENT_CLK_DOMAIN_IDX_ILWALID indicates no
     * corresponding CLIENT_CLK_DOMAIN.
     */
    LwU8   clientDomainIdx;
} LW2080_CTRL_CLK_CLK_DOMAIN_INFO_3X;

/*!
 * Structure describing CLK_DOMAIN_3X_FIXED static information/POR.
 */
typedef struct LW2080_CTRL_CLK_CLK_DOMAIN_INFO_3X_FIXED {
    /*!
     * CLK_DOMAIN_3X super class.  Must always be first element in the structure.
     */
    LW2080_CTRL_CLK_CLK_DOMAIN_INFO_3X super;

    /*!
     * Fixed frequency of the given CLK_DOMAIN in MHz.  This is the frequency
     * that the VBIOS DEVINIT has programmed for the CLK_DOMAIN.
     */
    LwU16                              freqMHz;
} LW2080_CTRL_CLK_CLK_DOMAIN_INFO_3X_FIXED;

/*!
 * Invalid ordering index.  Means that no ordering has been specified for the
 * clock domain.
 */
#define LW2080_CTRL_CLK_CLK_DOMAIN_3X_PROG_ORDERING_INDEX_ILWALID LW_U8_MAX

/*!
 * Enumeration of FREQ_DELTA offset types.
 */
#define LW2080_CTRL_CLK_CLK_FREQ_DELTA_TYPE_STATIC                0x00U
#define LW2080_CTRL_CLK_CLK_FREQ_DELTA_TYPE_PERCENT               0x01U

/*!
 * Structure describing FREQ_DELTA freq offset in kHz offset.
 * deltakHz value must be in the range of freqDeltaMinMHz to freqDeltaMaxMHz.
 */
typedef struct LW2080_CTRL_CLK_CLK_FREQ_DELTA_OFFSET_STATIC {
    LwS32 deltakHz;
} LW2080_CTRL_CLK_CLK_FREQ_DELTA_OFFSET_STATIC;

/*!
 * Structure describing FREQ_DELTA freq offset in percent.
 * Percentage floating point value before colwersion to FXP has to be in the
 * range of -1.0 to 1.0.
 */
typedef struct LW2080_CTRL_CLK_CLK_FREQ_DELTA_OFFSET_PERCENT {
    LwSFXP4_12 deltaPercent;
} LW2080_CTRL_CLK_CLK_FREQ_DELTA_OFFSET_PERCENT;

/*!
 * FREQ_DELTA type-specific data union. Discriminated by
 * FREQ_DELTA::type.
 */


/*!
 * Structure describing deviation and deviation type (Static, ...)
 */
typedef struct LW2080_CTRL_CLK_FREQ_DELTA {
    /*!
     * XAPI does not support using an element within another structure as a
     * discriminant.  Placing this redundant type value here until that design
     * constraint can be fixed.
     */
    LwU8 type;
    /*!
     * Type-specific data union.
     */
    union {
        LW2080_CTRL_CLK_CLK_FREQ_DELTA_OFFSET_STATIC  staticOffset;
        LW2080_CTRL_CLK_CLK_FREQ_DELTA_OFFSET_PERCENT percentOffset;
    } data;
} LW2080_CTRL_CLK_FREQ_DELTA;
typedef struct LW2080_CTRL_CLK_FREQ_DELTA *PLW2080_CTRL_CLK_FREQ_DELTA;

/*!
 * Helper macro to init a LW2080_CTRL_CLK_FREQ_DELTA structure.
 * Initializes the structure to default type static.
 */
#define LW2080_CTRL_CLK_FREQ_DELTA_INIT(pFreqDelta)                           \
do {                                                                          \
    (pFreqDelta)->type = LW2080_CTRL_CLK_CLK_FREQ_DELTA_TYPE_STATIC;          \
    (pFreqDelta)->data.staticOffset.deltakHz = 0;                             \
} while (LW_FALSE)

/*!
 * Helper macro to set a LW2080_CTRL_CLK_CLK_DELTA structure.
 */
#define LW2080_CTRL_CLK_DELTA_SET(pDeltaDest, pDeltaSrc)                      \
do {                                                                          \
    portMemCopy((pDeltaDest), LW_SIZEOF32(LW2080_CTRL_CLK_CLK_DELTA),         \
                (pDeltaSrc),  LW_SIZEOF32(LW2080_CTRL_CLK_CLK_DELTA));        \
} while (LW_FALSE)

/*!
 * Accessor macro for @refLW2080_CTRL_CLK_FREQ_DELTA::type
 *
 * @param[in] pFreqDelta   LW2080_CTRL_CLK_FREQ_DELTA value
 *
 * @return @ref LW2080_CTRL_CLK_FREQ_DELTA::type
 */
#define LW2080_CTRL_CLK_FREQ_DELTA_TYPE_GET(pFreqDelta)                       \
    (pFreqDelta)->type

/*!
 * Accessor macro for
 * @ref LW2080_CTRL_CLK_FREQ_DELTA::data.staticOffset.deltakHz
 *
 * @param[in] pFreqDelta   LW2080_CTRL_CLK_FREQ_DELTA pointer
 *
 * @return @ref LW2080_CTRL_CLK_FREQ_DELTA::data.staticOffset.deltakHz
 */
#define LW2080_CTRL_CLK_FREQ_DELTA_GET_STATIC(pFreqDelta)                     \
    ((LW2080_CTRL_CLK_FREQ_DELTA_TYPE_GET(pFreqDelta) ==                      \
        LW2080_CTRL_CLK_CLK_FREQ_DELTA_TYPE_STATIC)   ?                       \
        (pFreqDelta)->data.staticOffset.deltakHz : 0)

/*!
 * Accessor macro for
 * @ref LW2080_CTRL_CLK_FREQ_DELTA::data.percentOffset.percent
 *
 * @param[in] pFreqDelta   LW2080_CTRL_CLK_FREQ_DELTA pointer
 *
 * @return @ref LW2080_CTRL_CLK_FREQ_DELTA::data.percentOffset.percent
 */
#define LW2080_CTRL_CLK_FREQ_DELTA_GET_PCT(pFreqDelta)                        \
    ((LW2080_CTRL_CLK_FREQ_DELTA_TYPE_GET(pFreqDelta) ==                      \
        LW2080_CTRL_CLK_CLK_FREQ_DELTA_TYPE_PERCENT)  ?                       \
        (pFreqDelta)->data.percentOffset.deltaPercent : 0)

/*!
 * Structure describing CLK_DOMAIN_3X_PROG static information/POR.
 */
typedef struct LW2080_CTRL_CLK_CLK_DOMAIN_INFO_3X_PROG {
    /*!
     * CLK_DOMAIN_3X super class.  Must always be first element in the structure.
     */
    LW2080_CTRL_CLK_CLK_DOMAIN_INFO_3X   super;
    /*!
     * CLK_DOMAIN_PROG super/interface class.
     */
    LW2080_CTRL_CLK_CLK_DOMAIN_INFO_PROG prog;
    /*!
     * First index into the Clock Programming Table for this CLK_DOMAIN.
     */
    LwU8                                 clkProgIdxFirst;
    /*!
     * Last index into the Clock Programming Table for this CLK_DOMAIN.
     */
    LwU8                                 clkProgIdxLast;
    /*!
     * Noise-unaware ordering index for clock programming changes.
     * PP-TODO : To be removed once LWAPI are updated with version 3.0
     */
    LwU8                                 noiseUnawareOrderingIndex;
    /*!
     * Noise-Aware ordering index for clock programming changes.  Applicable
     * only if @ref CLK_DOMAIN_3X::bNoiseAwareCapable == LW_TRUE.
     * PP-TODO : To be removed once LWAPI are updated with version 3.0
     */
    LwU8                                 noiseAwareOrderingIndex;
    /*!
     * Boolean flag indicating whether Clock Domain should always be changed per
     * the Noise-Unaware Ordering group, even when in the Noise-Aware mode.
     * Applicable only if @ref CLK_DOMAIN_3X::bNoiseAwareCapable == LW_TRUE.
     */
    LwBool                               bForceNoiseUnawareOrdering;
    /*!
     * Factory OC frequency delta. We always apply this delta regardless of the
     * frequency delta range reg@ freqDeltaMinMHz and ref@ freqDeltaMaxMHz as
     * long as the clock programming entry has the OC feature enabled in it.
     * ref@ CLK_PROG_1X_PRIMARY::bOCOVEnabled
     */
    LW2080_CTRL_CLK_FREQ_DELTA           factoryDelta;
    /*!
     * Minimum frequency delta which can be applied to the CLK_DOMAIN.
     */
    LwS16                                freqDeltaMinMHz;
    /*!
     * Maximum frequency delta which can be applied to the CLK_DOMAIN.
     */
    LwS16                                freqDeltaMaxMHz;
} LW2080_CTRL_CLK_CLK_DOMAIN_INFO_3X_PROG;

/*!
 * Structure describing CLK_DOMAIN_3X_PRIMARY static information/POR.
 */
typedef struct LW2080_CTRL_CLK_CLK_DOMAIN_INFO_3X_PRIMARY {
    /*!
     * BOARDOBJ_INTERFACE super class.  Must always be first element in the structure.
     */
    LW2080_CTRL_BOARDOBJ_INTERFACE super;

    /*!
     * Mask indexes of CLK_DOMAINs which are SECONDARYs to this PRIMARY CLK_DOMAIN.
     */
    LwU32                          slaveIdxsMask;
} LW2080_CTRL_CLK_CLK_DOMAIN_INFO_3X_PRIMARY;
typedef struct LW2080_CTRL_CLK_CLK_DOMAIN_INFO_3X_PRIMARY *PLW2080_CTRL_CLK_CLK_DOMAIN_INFO_3X_PRIMARY;

/*!
 * Structure describing CLK_DOMAIN_3X_SECONDARY static information/POR.
 */
typedef struct LW2080_CTRL_CLK_CLK_DOMAIN_INFO_3X_SECONDARY {
    /*!
     * BOARDOBJ_INTERFACE super class.  Must always be first element in the structure.
     */
    LW2080_CTRL_BOARDOBJ_INTERFACE super;

    /*!
     * CLK_DOMAIN index of primary CLK_DOMAIN.
     */
    LwU8                           masterIdx;
} LW2080_CTRL_CLK_CLK_DOMAIN_INFO_3X_SECONDARY;
typedef struct LW2080_CTRL_CLK_CLK_DOMAIN_INFO_3X_SECONDARY *PLW2080_CTRL_CLK_CLK_DOMAIN_INFO_3X_SECONDARY;

/*!
 * Structure describing CLK_DOMAIN_3X_PRIMARY static information/POR.
 * PP-TODO : To be removed once LWAPI are updated with version 3.0
 */
typedef struct LW2080_CTRL_CLK_CLK_DOMAIN_INFO_3X_PRIMARY_WRAPPER {
    /*!
     * CLK_DOMAIN_3X_PROG super class.  Must always be first element in the structure.
     */
    LW2080_CTRL_CLK_CLK_DOMAIN_INFO_3X_PROG super;

    /*!
     * Mask indexes of CLK_DOMAINs which are SECONDARYs to this PRIMARY CLK_DOMAIN.
     */
    LwU32                                   slaveIdxsMask;
} LW2080_CTRL_CLK_CLK_DOMAIN_INFO_3X_PRIMARY_WRAPPER;
typedef struct LW2080_CTRL_CLK_CLK_DOMAIN_INFO_3X_PRIMARY_WRAPPER *PLW2080_CTRL_CLK_CLK_DOMAIN_INFO_3X_PRIMARY_WRAPPER;

/*!
 * Structure describing CLK_DOMAIN_3X_SECONDARY static information/POR.
 * PP-TODO : To be removed once LWAPI are updated with version 3.0
 */
typedef struct LW2080_CTRL_CLK_CLK_DOMAIN_INFO_3X_SECONDARY_WRAPPER {
    /*!
     * CLK_DOMAIN_3X_PROG super class.  Must always be first element in the structure.
     */
    LW2080_CTRL_CLK_CLK_DOMAIN_INFO_3X_PROG super;

    /*!
     * CLK_DOMAIN index of primary CLK_DOMAIN.
     */
    LwU8                                    masterIdx;
} LW2080_CTRL_CLK_CLK_DOMAIN_INFO_3X_SECONDARY_WRAPPER;
typedef struct LW2080_CTRL_CLK_CLK_DOMAIN_INFO_3X_SECONDARY_WRAPPER *PLW2080_CTRL_CLK_CLK_DOMAIN_INFO_3X_SECONDARY_WRAPPER;

/*!
 * Structure describing CLK_DOMAIN_30_PROG static information/POR.
 */
typedef struct LW2080_CTRL_CLK_CLK_DOMAIN_INFO_30_PROG {
    /*!
     * CLK_DOMAIN_3X_PROG super class. Must always be first element in the structure.
     */
    LW2080_CTRL_CLK_CLK_DOMAIN_INFO_3X_PROG super;
    /*!
     * Noise-unaware ordering index for clock programming changes.
     */
    LwU8                                    noiseUnawareOrderingIndex;
    /*!
     * Noise-Aware ordering index for clock programming changes.  Applicable
     * only if @ref CLK_DOMAIN_3X::bNoiseAwareCapable == LW_TRUE.
     */
    LwU8                                    noiseAwareOrderingIndex;
} LW2080_CTRL_CLK_CLK_DOMAIN_INFO_30_PROG;
typedef struct LW2080_CTRL_CLK_CLK_DOMAIN_INFO_30_PROG *PLW2080_CTRL_CLK_CLK_DOMAIN_INFO_30_PROG;

/*!
 * Structure describing CLK_DOMAIN_30_PRIMARY static information/POR.
 */
typedef struct LW2080_CTRL_CLK_CLK_DOMAIN_INFO_30_PRIMARY {
    /*!
     * CLK_DOMAIN_30_PROG super class.  Must always be first element in the structure.
     */
    LW2080_CTRL_CLK_CLK_DOMAIN_INFO_30_PROG    super;

    /*!
     * PRIMARY super class.
     */
    LW2080_CTRL_CLK_CLK_DOMAIN_INFO_3X_PRIMARY master;
} LW2080_CTRL_CLK_CLK_DOMAIN_INFO_30_PRIMARY;
typedef struct LW2080_CTRL_CLK_CLK_DOMAIN_INFO_30_PRIMARY *PLW2080_CTRL_CLK_CLK_DOMAIN_INFO_30_PRIMARY;

/*!
 * Structure describing CLK_DOMAIN_30_SECONDARY static information/POR.
 */
typedef struct LW2080_CTRL_CLK_CLK_DOMAIN_INFO_30_SECONDARY {
    /*!
     * CLK_DOMAIN_30_PROG super class. Must always be first element in the structure.
     */
    LW2080_CTRL_CLK_CLK_DOMAIN_INFO_30_PROG      super;

    /*!
     * CLK_DOMAIN_3X_SECONDARY super class.
     */
    LW2080_CTRL_CLK_CLK_DOMAIN_INFO_3X_SECONDARY slave;
} LW2080_CTRL_CLK_CLK_DOMAIN_INFO_30_SECONDARY;
typedef struct LW2080_CTRL_CLK_CLK_DOMAIN_INFO_30_SECONDARY *PLW2080_CTRL_CLK_CLK_DOMAIN_INFO_30_SECONDARY;

/*!
 * Structure describing CLK_DOMAIN_35_PROG Clock Monitor specific static information/POR.
 */
typedef struct LW2080_CTRL_CLK_CLK_DOMAIN_INFO_35_PROG_CLK_MON {
    /*!
     * Index to VFE equation used to compute the low threshold for the clock monitors.
     */
    LwU8 lowThresholdVfeIdx;
    /*!
     * Index to VFE equation used to compute the high threshold for the clock monitors.
     */
    LwU8 highThresholdVfeIdx;
} LW2080_CTRL_CLK_CLK_DOMAIN_INFO_35_PROG_CLK_MON;
typedef struct LW2080_CTRL_CLK_CLK_DOMAIN_INFO_35_PROG_CLK_MON *PLW2080_CTRL_CLK_CLK_DOMAIN_INFO_35_PROG_CLK_MON;

/*!
 * Structure describing CLK_DOMAIN_35_PROG Clock Monitor specific static information/POR.
 */
typedef struct LW2080_CTRL_CLK_CLK_DOMAIN_INFO_40_PROG_CLK_MON {
    /*!
     * Index to VFE equation used to compute the low threshold for the clock monitors.
     */
    LW2080_CTRL_PERF_VFE_EQU_IDX lowThresholdVfeIdx;
    /*!
     * Index to VFE equation used to compute the high threshold for the clock monitors.
     */
    LW2080_CTRL_PERF_VFE_EQU_IDX highThresholdVfeIdx;
} LW2080_CTRL_CLK_CLK_DOMAIN_INFO_40_PROG_CLK_MON;
typedef struct LW2080_CTRL_CLK_CLK_DOMAIN_INFO_40_PROG_CLK_MON *PLW2080_CTRL_CLK_CLK_DOMAIN_INFO_40_PROG_CLK_MON;

/*!
 * Structure describing CLK_DOMAIN_35_PROG static information/POR.
 */
typedef struct LW2080_CTRL_CLK_CLK_DOMAIN_INFO_35_PROG {
    /*!
     * CLK_DOMAIN_3X_PROG super class. Must always be first element in the structure.
     */
    LW2080_CTRL_CLK_CLK_DOMAIN_INFO_3X_PROG         super;
    /*!
     * Pre-Volt ordering index for clock programming changes.
     */
    LwU8                                            preVoltOrderingIndex;
    /*!
     * Post-Volt ordering index for clock programming changes.
     */
    LwU8                                            postVoltOrderingIndex;
    /*!
     * Position of clock domain in a tightly packed array of primary - secondary clock
     * domain V and/or F tuples.
     */
    LwU8                                            clkPos;
   /*!
     * Count of total number of (primary + secondary) lwrves supported on this clock domain.
     */
    LwU8                                            clkVFLwrveCount;
    /*!
     * Clock Monitor specific static information/POR.
     */
    LW2080_CTRL_CLK_CLK_DOMAIN_INFO_35_PROG_CLK_MON clkMon;
    /*!
     * Represents voltage delta defined in VBIOS clocks table as per LW POR.
     * This will give the deviation of given voltage from it's nominal value.
     */
    LwS32                                           porVoltDeltauV[LW2080_CTRL_VOLT_VOLT_RAIL_CLIENT_MAX_RAILS];
} LW2080_CTRL_CLK_CLK_DOMAIN_INFO_35_PROG;
typedef struct LW2080_CTRL_CLK_CLK_DOMAIN_INFO_35_PROG *PLW2080_CTRL_CLK_CLK_DOMAIN_INFO_35_PROG;

/*!
 * Structure describing CLK_DOMAIN_35_PRIMARY static information/POR.
 */
typedef struct LW2080_CTRL_CLK_CLK_DOMAIN_INFO_35_PRIMARY {
    /*!
     * CLK_DOMAIN_35_PROG super class.  Must always be first element in the structure.
     */
    LW2080_CTRL_CLK_CLK_DOMAIN_INFO_35_PROG    super;

    /*!
     * PRIMARY super class.
     */
    LW2080_CTRL_CLK_CLK_DOMAIN_INFO_3X_PRIMARY master;
} LW2080_CTRL_CLK_CLK_DOMAIN_INFO_35_PRIMARY;
typedef struct LW2080_CTRL_CLK_CLK_DOMAIN_INFO_35_PRIMARY *PLW2080_CTRL_CLK_CLK_DOMAIN_INFO_35_PRIMARY;

/*!
 * Structure describing CLK_DOMAIN_35_SECONDARY static information/POR.
 */
typedef struct LW2080_CTRL_CLK_CLK_DOMAIN_INFO_35_SECONDARY {
    /*!
     * CLK_DOMAIN_35_PROG super class. Must always be first element in the structure.
     */
    LW2080_CTRL_CLK_CLK_DOMAIN_INFO_35_PROG      super;

    /*!
     * CLK_DOMAIN_3X_SECONDARY super class.
     */
    LW2080_CTRL_CLK_CLK_DOMAIN_INFO_3X_SECONDARY slave;
} LW2080_CTRL_CLK_CLK_DOMAIN_INFO_35_SECONDARY;
typedef struct LW2080_CTRL_CLK_CLK_DOMAIN_INFO_35_SECONDARY *PLW2080_CTRL_CLK_CLK_DOMAIN_INFO_35_SECONDARY;

/*!
 * Structure describing CLK_DOMAIN_40_PROG_RAIL_VF_SECONDARY static information/POR.
 */
typedef struct LW2080_CTRL_CLK_CLK_DOMAIN_INFO_40_PROG_RAIL_VF_SECONDARY {
    /*!
     * CLK_DOMAIN index of primary CLK_DOMAIN.
     */
    LwU8 masterIdx;
} LW2080_CTRL_CLK_CLK_DOMAIN_INFO_40_PROG_RAIL_VF_SECONDARY;
typedef struct LW2080_CTRL_CLK_CLK_DOMAIN_INFO_40_PROG_RAIL_VF_SECONDARY *PLW2080_CTRL_CLK_CLK_DOMAIN_INFO_40_PROG_RAIL_VF_SECONDARY;

/*!
 * Structure describing CLK_DOMAIN_40_PROG_RAIL_VF_PRIMARY static information/POR.
 */
typedef struct LW2080_CTRL_CLK_CLK_DOMAIN_INFO_40_PROG_RAIL_VF_PRIMARY {
    /*!
     * First index into the Clock VF Relationship Table for this CLK_DOMAIN.
     */
    LwU8                             clkVfRelIdxFirst;
    /*!
     * Last index into the Clock VF Relationship Table for this CLK_DOMAIN.
     */
    LwU8                             clkVfRelIdxLast;
    /*!
     * Mask indexes of CLK_DOMAINs which are SECONDARYs to this PRIMARY CLK_DOMAIN.
     */
    LW2080_CTRL_BOARDOBJGRP_MASK_E32 slaveDomainsMask;
    /*!
     * Mask indexes of CLK_DOMAINs that belongs to this primary-secondary group.
     */
    LW2080_CTRL_BOARDOBJGRP_MASK_E32 masterSlaveDomainsMask;
} LW2080_CTRL_CLK_CLK_DOMAIN_INFO_40_PROG_RAIL_VF_PRIMARY;
typedef struct LW2080_CTRL_CLK_CLK_DOMAIN_INFO_40_PROG_RAIL_VF_PRIMARY *PLW2080_CTRL_CLK_CLK_DOMAIN_INFO_40_PROG_RAIL_VF_PRIMARY;

/*!
 * Types representing the characteristics of a given clock
 * domain on a given voltage rail.
 */
#define LW2080_CTRL_CLK_CLK_DOMAIN_40_PROG_RAIL_VF_TYPE_NONE      0x0U
#define LW2080_CTRL_CLK_CLK_DOMAIN_40_PROG_RAIL_VF_TYPE_PRIMARY   0x1U
#define LW2080_CTRL_CLK_CLK_DOMAIN_40_PROG_RAIL_VF_TYPE_SECONDARY 0x2U

/*!
 * CLK_DOMAIN_40_PROG_RAIL_VF type-specific data union.  Discriminated by
 * CLK_DOMAIN::super.type.
 */


/*!
 * Structure describing CLK_DOMAIN_40_PROG_RAIL_VF_PRIMARY static information/POR.
 */
typedef struct LW2080_CTRL_CLK_CLK_DOMAIN_INFO_40_PROG_RAIL_VF_ITEM {
    /*!
     * @ref LW2080_CTRL_CLK_CLK_DOMAIN_40_PROG_RAIL_VF_TYPE_<xyz>
     */
    LwU8 type;
    /*!
     * Position of clock domain in a tightly packed array of primary - secondary clock
     * domain V and/or F tuples.
     */
    LwU8 clkPos;
    /*!
     * Type-specific data union.
     */
    union {
        LW2080_CTRL_CLK_CLK_DOMAIN_INFO_40_PROG_RAIL_VF_PRIMARY   master;
        LW2080_CTRL_CLK_CLK_DOMAIN_INFO_40_PROG_RAIL_VF_SECONDARY slave;
    } data;
} LW2080_CTRL_CLK_CLK_DOMAIN_INFO_40_PROG_RAIL_VF_ITEM;
typedef struct LW2080_CTRL_CLK_CLK_DOMAIN_INFO_40_PROG_RAIL_VF_ITEM *PLW2080_CTRL_CLK_CLK_DOMAIN_INFO_40_PROG_RAIL_VF_ITEM;

/*!
 * Macro defining max allowed voltage rail VF item per clk domain.
 */
#define LW2080_CTRL_CLK_CLK_DOMAIN_PROG_RAIL_VF_ITEM_MAX 0x2U

/*!
 * Data that is used to adjust power measurements made on the FBVDD rail.
 */
typedef struct LW2080_CTRL_CLK_CLK_DOMAIN_40_PROG_FBVDD_PWR_ADJUSTMENT {
    /*!
     * Slope of the line that does the adjustment.
     */
    LwUFXP20_12 slope;

    /*!
     * Intercept of the line that does the adjustment.
     */
    LwSFXP20_12 interceptmW;
} LW2080_CTRL_CLK_CLK_DOMAIN_40_PROG_FBVDD_PWR_ADJUSTMENT;

/*!
 * Defines a mapping between a DRAMCLK frequency range and the voltage required
 * for that frequency.
 */
typedef struct LW2080_CTRL_CLK_CLK_DOMAIN_40_PROG_FBVDD_VF_MAPPING {
    /*!
     * Maximum frequency at which this voltage should be used.
     */
    LwU32 maxFreqkHz;

    /*!
     * Voltage to be used with
     * @ref LW2080_CTRL_CLK_CLK_DOMAIN_40_PROG_FBVDD_VF_MAPPING::maxFreqMhz
     */
    LwU32 voltuV;
} LW2080_CTRL_CLK_CLK_DOMAIN_40_PROG_FBVDD_VF_MAPPING;

/*!
 * Maximum number of @ref LW2080_CTRL_CLK_CLK_DOMAIN_40_PROG_FBVDD_VF_MAPPING entries
 * supported in @ref LW2080_CTRL_CLK_CLK_DOMAIN_40_PROG_FBVDD_VF_MAPPING_TABLE
 * lookup table
 */
#define LW2080_CTRL_CLK_CLK_DOMAIN_40_PROG_FBVDD_VF_MAPPING_TABLE_MAX_MAPPINGS 16U

/*!
 * A lookup table from FBVDD frequencies to voltages.
 */
typedef struct LW2080_CTRL_CLK_CLK_DOMAIN_40_PROG_FBVDD_VF_MAPPING_TABLE {
    /*!
     * Number of valid entries in
     * @ref LW2080_CTRL_CLK_CLK_DOMAIN_40_PROG_FBVDD_VF_MAPPING_TABLE
     */
    LwU8                                                numMappings;

    /*!
     * Mappings from a minimum frequency to a voltage.
     */
    LW2080_CTRL_CLK_CLK_DOMAIN_40_PROG_FBVDD_VF_MAPPING mappings[LW2080_CTRL_CLK_CLK_DOMAIN_40_PROG_FBVDD_VF_MAPPING_TABLE_MAX_MAPPINGS];
} LW2080_CTRL_CLK_CLK_DOMAIN_40_PROG_FBVDD_VF_MAPPING_TABLE;

/*!
 * FBVDD data that may be associated with a given clock domain.
 */
typedef struct LW2080_CTRL_CLK_CLK_DOMAIN_40_PROG_FBVDD_DATA {
    /*!
     * Whether this data is valid; not all clock domains will support FBVDD
     * data.
     */
    LwBool                                                    bValid;

    /*!
     * Data for doing power adjustments on the rail.
     */
    LW2080_CTRL_CLK_CLK_DOMAIN_40_PROG_FBVDD_PWR_ADJUSTMENT   pwrAdjustment;

    /*!
     * Mappings between frequencies and voltages on FBVDD.
     */
    LW2080_CTRL_CLK_CLK_DOMAIN_40_PROG_FBVDD_VF_MAPPING_TABLE vfMappingTable;
} LW2080_CTRL_CLK_CLK_DOMAIN_40_PROG_FBVDD_DATA;

/*!
 * Structure describing CLK_DOMAIN_40_PROG static information/POR.
 */
typedef struct LW2080_CTRL_CLK_CLK_DOMAIN_INFO_40_PROG {
    /*!
     * CLK_DOMAIN_3X super class. Must always be first element in the structure.
     */
    LW2080_CTRL_CLK_CLK_DOMAIN_INFO_3X                   super;
    /*!
     * CLK_DOMAIN_PROG super/interface class.
     */
    LW2080_CTRL_CLK_CLK_DOMAIN_INFO_PROG                 prog;
    /*!
     * Pre-Volt ordering index for clock programming changes.
     */
    LwU8                                                 preVoltOrderingIndex;
    /*!
     * Post-Volt ordering index for clock programming changes.
     */
    LwU8                                                 postVoltOrderingIndex;
   /*!
     * Count of total number of (primary + secondary) lwrves supported on this clock domain.
     */
    LwU8                                                 clkVFLwrveCount;
    /*!
     * First index into the Clock Enumeration Table for this CLK_DOMAIN.
     */
    LwU8                                                 clkEnumIdxFirst;
    /*!
     * Last index into the Clock Enumeration Table for this CLK_DOMAIN.
     */
    LwU8                                                 clkEnumIdxLast;
    /*!
     * Minimum frequency delta which can be applied to the CLK_DOMAIN.
     */
    LwS16                                                freqDeltaMinMHz;
    /*!
     * Maximum frequency delta which can be applied to the CLK_DOMAIN.
     */
    LwS16                                                freqDeltaMaxMHz;
    /*!
     * Factory OC frequency delta.This delta is programmed by AIC in
     * overclocking table.
     */
    LW2080_CTRL_CLK_FREQ_DELTA                           factoryDelta;
    /*!
     * GRD OC frequency delta. This delta is programmed by POR team
     * in clocks table. We will respect this delta IFF client explicitly
     * opted for it by setting ref@ CLK_DOAMINS::bGrdFreqOCEnabled
     */
    LW2080_CTRL_CLK_FREQ_DELTA                           grdFreqDelta;
    /*!
     * Mask of volt rails on which this clock domain has its Vmin.
     */
    LwU32                                                railMask;
    /*!
     * Rail specific data for a given programmable clock domain.
     * ref@ LW2080_CTRL_CLK_CLK_DOMAIN_INFO_40_PROG_RAIL_VF_ITEM
     */
    LW2080_CTRL_CLK_CLK_DOMAIN_INFO_40_PROG_RAIL_VF_ITEM railVfItem[LW2080_CTRL_CLK_CLK_DOMAIN_PROG_RAIL_VF_ITEM_MAX];
    /*!
     * Clock Monitor specific static information/POR.
     */
    LW2080_CTRL_CLK_CLK_DOMAIN_INFO_40_PROG_CLK_MON      clkMon;

    /*!
     * Data characterizing the FBVDD rail associated with this CLK_DOMAIN, if
     * valid.
     */
    LW2080_CTRL_CLK_CLK_DOMAIN_40_PROG_FBVDD_DATA        fbvddData;
} LW2080_CTRL_CLK_CLK_DOMAIN_INFO_40_PROG;
typedef struct LW2080_CTRL_CLK_CLK_DOMAIN_INFO_40_PROG *PLW2080_CTRL_CLK_CLK_DOMAIN_INFO_40_PROG;

/*!
 * CLK_DOMAIN type-specific data union.  Discriminated by
 * CLK_DOMAIN::super.type.
 */


/*!
 * Structure describing CLK_DOMAIN static information/POR.  Implements the
 * BOARDOBJ model/interface.
 */
typedef struct LW2080_CTRL_CLK_CLK_DOMAIN_INFO {
    /*!
     * LW2080_CTRL_BOARDOBJ super class.  Must always be first object in
     * structure.
     */
    LW2080_CTRL_BOARDOBJ          super;
    /*!
     * XAPI does not support using an element within another structure as a
     * discriminant.  Placing this redundant type value here until that design
     * constraint can be fixed.
     */
    LwU8                          type;
    /*!
     * @ref LW2080_CTRL_CLK_DOMAIN_<xyz>
     */
    LwU32                         domain;
    /*!
     * Mask of available partition indexes @ref LW2080_CTRL_CLK_DOMAIN_PART_IDX_<xyz>
     * for a clock domain. The value will be 0 in case of a domain does not have
     * partitions.
     */
    LwU32                         partMask;
    /*!
     * @ref LW2080_CTRL_CLK_PUBLIC_DOMAIN_ENUM
     *
     * @note _VIDEO domain is not supported yet in client clk domains
     */
    LW2080_CTRL_CLK_PUBLIC_DOMAIN publicDomain;
    /*!
     * Type-specific data union.
     */
    union {
        LW2080_CTRL_CLK_CLK_DOMAIN_INFO_2X                   v2x;
        LW2080_CTRL_CLK_CLK_DOMAIN_INFO_3X                   v3x;
        LW2080_CTRL_CLK_CLK_DOMAIN_INFO_3X_FIXED             v3xFixed;
        LW2080_CTRL_CLK_CLK_DOMAIN_INFO_3X_PROG              v3xProg;
        LW2080_CTRL_CLK_CLK_DOMAIN_INFO_3X_PRIMARY_WRAPPER   v3xMaster;
        LW2080_CTRL_CLK_CLK_DOMAIN_INFO_3X_SECONDARY_WRAPPER v3xSlave;
        LW2080_CTRL_CLK_CLK_DOMAIN_INFO_30_PROG              v30Prog;
        LW2080_CTRL_CLK_CLK_DOMAIN_INFO_30_PRIMARY           v30Master;
        LW2080_CTRL_CLK_CLK_DOMAIN_INFO_30_SECONDARY         v30Slave;
        LW2080_CTRL_CLK_CLK_DOMAIN_INFO_35_PROG              v35Prog;
        LW2080_CTRL_CLK_CLK_DOMAIN_INFO_35_PRIMARY           v35Master;
        LW2080_CTRL_CLK_CLK_DOMAIN_INFO_35_SECONDARY         v35Slave;
        LW2080_CTRL_CLK_CLK_DOMAIN_INFO_40_PROG              v40Prog;
    } data;
} LW2080_CTRL_CLK_CLK_DOMAIN_INFO;
typedef struct LW2080_CTRL_CLK_CLK_DOMAIN_INFO *PLW2080_CTRL_CLK_CLK_DOMAIN_INFO;

/*!
 * Structure describing CLK_DOMAINS static information/POR.  Implements the
 * BOARDOBJGRP model/interface.
 */
#define LW2080_CTRL_CLK_CLK_DOMAINS_INFO_MESSAGE_ID (0x19U)

typedef struct LW2080_CTRL_CLK_CLK_DOMAINS_INFO {
    /*!
     * LW2080_CTRL_BOARDOBJGRP_E32 super class.  Must always be first object in
     * structure.
     */
    LW2080_CTRL_BOARDOBJGRP_E32      super;

    /*/
     * CLK_DOMAIN version.  @ref LW2080_CTRL_CLK_CLK_DOMAIN_VERSION_<xyz>
     */
    LwU8                             version;
    /*!
     * Clocks HAL value.  Used to interpret/translate the CLK_DOMAIN indexes to
     * Clock Domains.
     */
    LwU8                             clocksHAL;
    /*!
     * Over Clocking Bin value. Represent the silicon quality for OC. Cached
     * directly from the OC table.
     */
    LwU8                             overClockingBin;
    /*!
     * Mask of CLK_DOMAINs specified in the BOARDOBJGRP by the VBIOS.  Mask of
     * @rerf LW2080_CTRL_CLK_DOMAIN_<xyz>.
     */
    LwU32                            vbiosDomains;
    /*!
     * Mask of CLK_DOMAINs which are readyable on this GPU.  Will be a super-set
     * of @ref vbiosDomains.  Mask of @rerf LW2080_CTRL_CLK_DOMAIN_<xyz>.
     */
    LwU32                            readableDomains;
    /*!
     * Mask of CLK_DOMAINs which are programmable on this GPU.  Will be a
     * sub-set of @ref vbiosDomains.  Mask of @rerf LW2080_CTRL_CLK_DOMAIN_<xyz>.
     */
    LwU32                            programmableDomains;
    /*!
     * Mask of domains which implement CLK_DOMAIN_3X_PRIMARY interface.
     */
    LW2080_CTRL_BOARDOBJGRP_MASK_E32 masterDomainsMask;
    /*!
     * Mask of domains which RM will export as CLIENT_CLK_DOMAINs.
     */
    LW2080_CTRL_BOARDOBJGRP_MASK_E32 clientDomainsMask;
    /*!
     * Mask of domains for which clock monitors are supported in RM/PMU.
     */
    LW2080_CTRL_BOARDOBJGRP_MASK_E32 clkMonDomainsMask;
    /*!
     * CLK_CNTR sampling period in the PMU (ms).  Used for the period at which
     * the CLK_DOMAINs will be periodically sampled by the PMU CLK code.
     */
    LwU16                            cntrSamplingPeriodms;
    /*!
     * Boolean indicating whether CLK_MONITOR is enabled.
     */
    LwBool                           bClkMonEnabled;
    /*!
     * CLK_MONITOR reference window period (us). Used for the period at which
     * the CLK_MONITORs will be periodically sampled by the PMU CLK code.
     */
    LwU16                            clkMonRefWinUsec;
    /*!
     * VFE index to get the XBAR boost required for MCLK switch.
     */
    LwBoardObjIdx                    xbarBoostVfeIdx;
    /*!
     * Array of CLK_DOMAIN structures.  Has valid indexes corresponding to the
     * bits set in @ref super.objMask.
     */
    LW2080_CTRL_CLK_CLK_DOMAIN_INFO  domains[LW2080_CTRL_BOARDOBJ_MAX_BOARD_OBJECTS];
} LW2080_CTRL_CLK_CLK_DOMAINS_INFO;
typedef struct LW2080_CTRL_CLK_CLK_DOMAINS_INFO *PLW2080_CTRL_CLK_CLK_DOMAINS_INFO;

/*!
 * LW2080_CTRL_CMD_CLK_CLK_DOMAINS_GET_INFO
 *
 * This command returns CLK_DOMAINS static object information/POR as specified
 * by the VBIOS in either Performance Table->Domain Entries (@ref
 * LW2080_CTRL_CLK_CLK_DOMAIN_VERSION_2X) or Clocks Table (@ref
 * LW2080_CTRL_CLK_CLK_DOMAIN_VERSION_3X).
 *
 * The CLK_DOMAIN objects are indexed per how they are stored in the RM.  For
 * @ref LW2080_CTRL_CLK_CLK_DOMAIN_VERSION_2X, the RM orders them by their
 * change sequence ordreing.  For @ref LW2080_CTRL_CLK_CLK_DOMAIN_VERSION_3X,
 * they are ordered per the Clock Domains Enumeration (per the Clocks Table
 * Spec) specified for the chip.
 *
 * See @ref LW2080_CTRL_CLK_CLK_DOMAINS_INFO for documentation on the
 * parameters.
 *
 * Return values are specified per @ref BoardObjGrpGetInfo.
 */
#define LW2080_CTRL_CMD_CLK_CLK_DOMAINS_GET_INFO (0x20801019) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_CLK_INTERFACE_ID << 8) | LW2080_CTRL_CLK_CLK_DOMAINS_INFO_MESSAGE_ID" */

/*!
 * Maximum number of CLK_DOMAIN frequencies.
 */
#define LW2080_CTRL_CLK_CLK_DOMAIN_MAX_FREQS     512U

/*!
 * Parameter structure for @ref
 * LW2080_CTRL_CMD_CLK_CLK_DOMAIN_FREQS_ENUM.
 */
#define LW2080_CTRL_CLK_CLK_DOMAIN_FREQS_ENUM_MESSAGE_ID (0x1AU)

typedef struct LW2080_CTRL_CLK_CLK_DOMAIN_FREQS_ENUM {
    /*!
     * [in] Index of the CLK_DOMAIN object for which to enumerate frequencies.
     */
    LwU8  clkDomainIdx;
    /*!
     * [out] Number of frequencies returned by RMCTRL to the caller.
     */
    LwU16 numFreqs;
    /*!
     * [out] Array of frequencies (MHz) returned by RMCTRL.  Has valid indexes in the
     * range [0, @ref numFreqs).
     */
    LwU32 freqsMHz[LW2080_CTRL_CLK_CLK_DOMAIN_MAX_FREQS];
} LW2080_CTRL_CLK_CLK_DOMAIN_FREQS_ENUM;
typedef struct LW2080_CTRL_CLK_CLK_DOMAIN_FREQS_ENUM *PLW2080_CTRL_CLK_CLK_DOMAIN_FREQS_ENUM;

/*!
 * LW2080_CTRL_CMD_CLK_CLK_DOMAIN_FREQS_ENUM
 *
 * This command enumerates the frequencies supported on the given CLK_DOMAIN, as
 * specified by a CLK_DOMAIN index in to the RM/VBIOS Clocks Table.  For more
 * information on indexes see @ref LW2080_CTRL_CMD_CLK_CLK_DOMAINS_GET_INFO.
 *
 * See @ref LW2080_CTRL_CLK_CLK_DOMAIN_FREQS_ENUM for documentation on the
 * parameters.
 */
#define LW2080_CTRL_CMD_CLK_CLK_DOMAIN_FREQS_ENUM (0x2080101a) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_CLK_INTERFACE_ID << 8) | LW2080_CTRL_CLK_CLK_DOMAIN_FREQS_ENUM_MESSAGE_ID" */

/*!
 * Structure describing deviation of given parameter (voltage, frequency, ...)
 * from it's nominal value.
 */
typedef struct LW2080_CTRL_CLK_CLK_DELTA {
    /*!
     * This will give the deviation of given freq from it's nominal value.
     * NOTE: we have one single freq delta that will apply to all voltage rails
     */
    LW2080_CTRL_CLK_FREQ_DELTA freqDelta;
    /*!
     * This will give the deviation of given voltage from it's nominal value.
     */
    LwS32                      voltDeltauV[LW2080_CTRL_VOLT_VOLT_RAIL_CLIENT_MAX_RAILS];
} LW2080_CTRL_CLK_CLK_DELTA;

/*!
 * Structure describing CLK_DOMAIN_3X specific control parameters.
 */
typedef struct LW2080_CTRL_CLK_CLK_DOMAIN_CONTROL_3X {
    /*!
     * CLK_DOMAIN_3X does not contain any control parameters as of now.
     */
    LwU8 rsvd;
} LW2080_CTRL_CLK_CLK_DOMAIN_CONTROL_3X;

/*!
 * Structure describing CLK_DOMAIN_3X_PROG specific control parameters.
 */
typedef struct LW2080_CTRL_CLK_CLK_DOMAIN_CONTROL_3X_PROG {
    /*!
     * CLK_DOMAIN_3X super class.  Must always be first element in the structure.
     */
    LW2080_CTRL_CLK_CLK_DOMAIN_CONTROL_3X super;
    /*!
     * Delta for given CLK Domain.
     * Delta are cumulative, i.e. global + local
     */
    LW2080_CTRL_CLK_CLK_DELTA             deltas;
} LW2080_CTRL_CLK_CLK_DOMAIN_CONTROL_3X_PROG;

/*!
 * Structure describing CLK_DOMAIN_35_PROG Clock Monitors specific control parameters.
 */
typedef struct LW2080_CTRL_CLK_CLK_DOMAIN_CONTROL_35_PROG_CLK_MON {
    /*!
     * Flags Clock Monitor control information.
     * @ref LW2080_CTRL_CLK_CLK_DOMAIN_CONTROL_35_PROG_CLK_MON_FLAGS
     */
    LwU32       flags;
    /*!
     * Override Minimum Threshold value set for clock monitors when override
     * flag is set. The expected value is in format 20.12 and in range [0,1].
     */
    LwUFXP20_12 lowThresholdOverride;
    /*!
     * Override Maximum Threshold value set for clock monitors when override
     * flag is set. The expected value is in format 20.12 and in range [0,1].
     */
    LwUFXP20_12 highThresholdOverride;
} LW2080_CTRL_CLK_CLK_DOMAIN_CONTROL_35_PROG_CLK_MON;

/*!
 * Flags for the @ref flags parameter in the
 * @ref LW2080_CTRL_CLK_CLK_DOMAIN_CONTROL_35_PROG_CLK_MON structure
 */
#define LW2080_CTRL_CLK_CLK_DOMAIN_CONTROL_35_PROG_CLK_MON_FLAGS_FAULT                  0:0
#define LW2080_CTRL_CLK_CLK_DOMAIN_CONTROL_35_PROG_CLK_MON_FLAGS_FAULT_DEFAULT       (0x00000000U)
#define LW2080_CTRL_CLK_CLK_DOMAIN_CONTROL_35_PROG_CLK_MON_FLAGS_FAULT_CLEAR         (0x00000001U)

/*!
 * http://lwbugs/1971316
 * The threshold values are expected to be in FXP20_12 format with values in
 * range (0, 1].
 * Threshold override value of 0 is treated as ignore case i.e. the default VFE
 * callwlated threshold value is used.
 */
#define LW2080_CTRL_CLK_CLK_DOMAIN_CONTROL_35_PROG_CLK_MON_THRESHOLD_OVERRIDE_MIN    0x00000U
#define LW2080_CTRL_CLK_CLK_DOMAIN_CONTROL_35_PROG_CLK_MON_THRESHOLD_OVERRIDE_MAX    0x01000U
#define LW2080_CTRL_CLK_CLK_DOMAIN_CONTROL_35_PROG_CLK_MON_THRESHOLD_OVERRIDE_IGNORE LW2080_CTRL_CLK_CLK_DOMAIN_CONTROL_35_PROG_CLK_MON_THRESHOLD_OVERRIDE_MIN

#define LW2080_CTRL_CLK_CLK_DOMAIN_CONTROL_35_PROG_CLK_MON_THRESHOLD_OVERRIDE_WITHIN_RANGE(thresh)  \
        ((thresh) >  LW2080_CTRL_CLK_CLK_DOMAIN_CONTROL_35_PROG_CLK_MON_THRESHOLD_OVERRIDE_MIN &&   \
         (thresh) <= LW2080_CTRL_CLK_CLK_DOMAIN_CONTROL_35_PROG_CLK_MON_THRESHOLD_OVERRIDE_MAX)

/*!
 * Structure describing CLK_DOMAIN_35_PROG specific control parameters.
 */
typedef struct LW2080_CTRL_CLK_CLK_DOMAIN_CONTROL_35_PROG {
    /*!
     * CLK_DOMAIN_3X_PROG super class.  Must always be first element in the structure.
     */
    LW2080_CTRL_CLK_CLK_DOMAIN_CONTROL_3X_PROG         super;
    /*!
     * Clock Monitors specific control parameters.
     */
    LW2080_CTRL_CLK_CLK_DOMAIN_CONTROL_35_PROG_CLK_MON clkMon;
} LW2080_CTRL_CLK_CLK_DOMAIN_CONTROL_35_PROG;

/*!
 * Structure describing CLK_DOMAIN_35_PROG specific control parameters.
 */
typedef struct LW2080_CTRL_CLK_CLK_DOMAIN_CONTROL_40_PROG {
    /*!
     * CLK_DOMAIN_3X super class.  Must always be first element in the structure.
     */
    LW2080_CTRL_CLK_CLK_DOMAIN_CONTROL_3X              super;
    /*!
     * Delta for given CLK Domain.
     * Delta are cumulative, i.e. global + local
     */
    LW2080_CTRL_CLK_CLK_DELTA                          deltas;
    /*!
     * Clock Monitors specific control parameters.
     */
    LW2080_CTRL_CLK_CLK_DOMAIN_CONTROL_35_PROG_CLK_MON clkMon;
} LW2080_CTRL_CLK_CLK_DOMAIN_CONTROL_40_PROG;
typedef struct LW2080_CTRL_CLK_CLK_DOMAIN_CONTROL_40_PROG *PLW2080_CTRL_CLK_CLK_DOMAIN_CONTROL_40_PROG;

/*!
 * CLK_DOMAIN type-specific data union.  Discriminated by
 * CLK_DOMAIN::super.type.
 */


/*!
 * Structure representing the control parameters associated with a CLK_DOMAIN.
 */
typedef struct LW2080_CTRL_CLK_CLK_DOMAIN_CONTROL {
    /*!
     * LW2080_CTRL_BOARDOBJ super class.  Must always be first object in
     * structure.
     */
    LW2080_CTRL_BOARDOBJ super;
    /*!
     * XAPI does not support using an element within another structure as a
     * discriminant.  Placing this redundant type value here until that design
     * constraint can be fixed.
     */
    LwU8                 type;
    /*!
     * Type-specific data union.
     */
    union {
        LW2080_CTRL_CLK_CLK_DOMAIN_CONTROL_3X      v3x;
        LW2080_CTRL_CLK_CLK_DOMAIN_CONTROL_3X_PROG v3xProg;
        LW2080_CTRL_CLK_CLK_DOMAIN_CONTROL_35_PROG v35Prog;
        LW2080_CTRL_CLK_CLK_DOMAIN_CONTROL_40_PROG v40Prog;
    } data;
} LW2080_CTRL_CLK_CLK_DOMAIN_CONTROL;

/*!
 * Structure representing the control parameters associated with a CLK_DOMAINS.
 */
typedef struct LW2080_CTRL_CLK_CLK_DOMAINS_CONTROL_PARAMS {
    /*!
     * LW2080_CTRL_BOARDOBJGRP_E32 super class.  Must always be first object in
     * structure.
     */
    LW2080_CTRL_BOARDOBJGRP_E32        super;
    /*!
     * Boolean flag that will be use by MODS team to override the domain's
     * OV/OC limits.
     */
    LwBool                             bOverrideOVOC;
    /*!
     * Boolean flag that will be use by client to disable the domain's
     * OC offset including the factory OC offset.
     */
    LwBool                             bDebugMode;
    /*!
     * Boolean indicating whether to enforce VF lwrve increasing monotonicity.
     * If LW_TRUE, RM/PMU will post-process all VF lwrve to make sure that both
     * V and F increase with every subsequent point (including accounting for
     * the various offsets).
     */
    LwBool                             bEnforceVfMonotonicity;
    /*!
     * Boolean indicating whether to enforce VF lwrve smoothening to reduce
     * large discountinuities.
     * If LW_TRUE, RM/PMU will post-process all VF lwrve to make sure that large
     * frequency jumps in VF lwrve will be reduced (including accounting for the
     * various offset).
     */
    LwBool                             bEnforceVfSmoothening;
    /*!
     * Boolean indicating whether to respect the CLK_DOMAIN_INFO_40_PROG::grdFreqDelta
     * This boolean will be set to TRUE by LWTOPPs based on client request.
     */
    LwBool                             bGrdFreqOCEnabled;
    /*!
     * Global delta for all the CLK Domains
     */
    LW2080_CTRL_CLK_CLK_DELTA          deltas;
    /*!
     * Array of CLK_DOMAIN structures.  Has valid indexes corresponding to the
     * bits set in @ref super.objMask.
     */
    LW2080_CTRL_CLK_CLK_DOMAIN_CONTROL domains[LW2080_CTRL_BOARDOBJ_MAX_BOARD_OBJECTS];
} LW2080_CTRL_CLK_CLK_DOMAINS_CONTROL_PARAMS;

/*!
 * LW2080_CTRL_CMD_CLK_CLK_DOMAINS_GET_CONTROL
 *
 * This command returns CLK_DOMAINS control parameters as specified by the
 * VBIOS in either Performance Table->Domain Entries (@ref
 * LW2080_CTRL_CLK_CLK_DOMAIN_VERSION_2X) or Clocks Table (@ref
 * LW2080_CTRL_CLK_CLK_DOMAIN_VERSION_3X).
 *
 * See @ref LW2080_CTRL_CLK_CLK_DOMAINS_CONTROL_PARAMS for documentation on the
 * parameters.
 *
 * Return values are specified per @ref BoardObjGrpGetControl.
 */
#define LW2080_CTRL_CMD_CLK_CLK_DOMAINS_GET_CONTROL              (0x2080101b) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_CLK_INTERFACE_ID << 8) | 0x1B" */

/*!
 * LW2080_CTRL_CMD_CLK_CLK_DOMAINS_SET_CONTROL
 *
 * This command accepts client-specified control parameters for a set
 * of CLK_DOMAIN entries in the Clocks Table, and applies these new
 * parameters to the set of CLK_DOMAIN entries.
 *
 *
 * See LW2080_CTRL_CLK_CLK_DOMAINS_CONTROL_PARAMS for documentation on the
 * parameters.
 *
 * Return values are specified per @ref BoardObjGrpSetControl.
 */
#define LW2080_CTRL_CMD_CLK_CLK_DOMAINS_SET_CONTROL              (0x2080101c) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_CLK_INTERFACE_ID << 8) | 0x1C" */

/*!
 * Macro representing an INVALID/UNSUPPORTED CLK_PROG index.
 */
#define LW2080_CTRL_CLK_CLK_PROG_IDX_ILWALID                     LW2080_CTRL_BOARDOBJ_IDX_ILWALID_8BIT

/*!
 * Enumeration of CLK_PROG class types.
 */
#define LW2080_CTRL_CLK_CLK_PROG_TYPE_3X                         0x00U
#define LW2080_CTRL_CLK_CLK_PROG_TYPE_30                         0x01U
#define LW2080_CTRL_CLK_CLK_PROG_TYPE_30_PRIMARY                 0x02U
#define LW2080_CTRL_CLK_CLK_PROG_TYPE_30_PRIMARY_RATIO           0x03U
#define LW2080_CTRL_CLK_CLK_PROG_TYPE_30_PRIMARY_TABLE           0x04U
#define LW2080_CTRL_CLK_CLK_PROG_TYPE_35                         0x05U
#define LW2080_CTRL_CLK_CLK_PROG_TYPE_35_PRIMARY                 0x06U
#define LW2080_CTRL_CLK_CLK_PROG_TYPE_35_PRIMARY_RATIO           0x07U
#define LW2080_CTRL_CLK_CLK_PROG_TYPE_35_PRIMARY_TABLE           0x08U
#define LW2080_CTRL_CLK_CLK_PROG_TYPE_UNKNOWN                    LW_U8_MAX

/*!
 * Enumeration of CLK_DOMAIN's INTERFACE types.
 */
#define LW2080_CTRL_CLK_CLK_PROG_INTERFACE_TYPE_3X_PRIMARY       0x00U
#define LW2080_CTRL_CLK_CLK_PROG_INTERFACE_TYPE_3X_PRIMARY_RATIO 0x01U
#define LW2080_CTRL_CLK_CLK_PROG_INTERFACE_TYPE_3X_PRIMARY_TABLE 0x02U

// PP-TODO : Temporary typedef
#define LW2080_CTRL_CLK_CLK_PROG_TYPE_1X                         0x09U
#define LW2080_CTRL_CLK_CLK_PROG_TYPE_1X_PRIMARY                 0x0AU
#define LW2080_CTRL_CLK_CLK_PROG_TYPE_1X_PRIMARY_RATIO           0x0BU
#define LW2080_CTRL_CLK_CLK_PROG_TYPE_1X_PRIMARY_TABLE           0x0LW

/*!
 * Enumeration of CLK_PROG source types.
 * OSMs (One Source Modules) have been the default source on ALT_PATH for most of
 * the clocks prior to Ampere hence ONE_SOURCE_PATH was often interchangeably used
 * with ALT_PATH. Starting GA100, OSMs have been deprecated and been replaced with
 * SWDIV hence it calls for deprecating ONE_SOURCE_PATH.
 * Aliasing it with ALT_PATH makes sure the existing logic in clocks/perf doesn't
 * break and follows the expected source in spirit!
 *
 * To-do akshatam/avogt: Make sure the SOURCE_DEFAULT is defined as 0 instead of
 * SOURCE_PLL. With PLL_PATH defined as 0, uninitialized/initialized-to-0 source
 * in the target structure would falsely indicate that the client wants to force
 * PLL path when it actually doesn't want to.
 */
#define LW2080_CTRL_CLK_PROG_1X_SOURCE_PLL                       0x00U /* Domain-specific PLL */
#define LW2080_CTRL_CLK_PROG_1X_SOURCE_ONE_SOURCE                0x01U /* Before Ampere */
#define LW2080_CTRL_CLK_PROG_1X_SOURCE_ALT_PATH                  LW2080_CTRL_CLK_PROG_1X_SOURCE_ONE_SOURCE /* Ampere and after */
#define LW2080_CTRL_CLK_PROG_1X_SOURCE_NAFLL                     0x02U /* Domain-specific NAFLL */
#define LW2080_CTRL_CLK_PROG_1X_SOURCE_SPPLL0                    0x04U /* Ampere and after */
#define LW2080_CTRL_CLK_PROG_1X_SOURCE_SPPLL1                    0x05U /* Ampere and after */
#define LW2080_CTRL_CLK_PROG_1X_SOURCE_XTAL                      0x06U /* Ampere and after */
#define LW2080_CTRL_CLK_PROG_1X_SOURCE_ILWALID                   LW_U8_MAX
#define LW2080_CTRL_CLK_PROG_1X_SOURCE_DEFAULT                   LW_U8_MAX

#define LW2080_CTRL_CLK_PROG_1X_FREQ_STEP_SIZE_NONE              0x00U
#define LW2080_CTRL_CLK_PROG_1X_FREQ_STEP_SIZE_ALL               0x01U

/*!
 * Structure of data specific to the _PLL source.
 */
typedef struct LW2080_CTRL_CLK_CLK_PROG_1X_SOURCE_PLL {
    /*!
     * PLL index.
     */
    LwU8 pllIdx;
    /*!
     * Frequency step size for the PLL within the specified range.
     */
    LwU8 freqStepSizeMHz;
} LW2080_CTRL_CLK_CLK_PROG_1X_SOURCE_PLL;
typedef struct LW2080_CTRL_CLK_CLK_PROG_1X_SOURCE_PLL *PLW2080_CTRL_CLK_CLK_PROG_1X_SOURCE_PLL;

/*!
 * Union of source-specific data.
 */
typedef union LW2080_CTRL_CLK_CLK_PROG_1X_SOURCE_DATA {
    LW2080_CTRL_CLK_CLK_PROG_1X_SOURCE_PLL pll;
} LW2080_CTRL_CLK_CLK_PROG_1X_SOURCE_DATA;

typedef union LW2080_CTRL_CLK_CLK_PROG_1X_SOURCE_DATA *PLW2080_CTRL_CLK_CLK_PROG_1X_SOURCE_DATA;

/*!
 * Structure describing the static configuration/POR state of the _1X class.
 */
typedef struct LW2080_CTRL_CLK_CLK_PROG_INFO_3X {
    /*!
     * Source enumeration.  @ref LW2080_CTRL_CLK_CLK_PROG_1X_SOURCE_<XYZ>.
     */
    LwU8                                    source;
    /*!
     * Maximum frequency for this CLK_PROG entry.  Entries for a given domain
     * need to be specified in ascending maxFreqMhz.
     */
    LwU16                                   freqMaxMHz;
    /*!
     * Union of source-specific data.
     */
    LW2080_CTRL_CLK_CLK_PROG_1X_SOURCE_DATA sourceData;
} LW2080_CTRL_CLK_CLK_PROG_INFO_3X;
typedef struct LW2080_CTRL_CLK_CLK_PROG_INFO_3X *PLW2080_CTRL_CLK_CLK_PROG_INFO_3X;

// PP-TODO : For backward compatibility
typedef LW2080_CTRL_CLK_CLK_PROG_INFO_3X LW2080_CTRL_CLK_CLK_PROG_INFO_1X;
typedef struct LW2080_CTRL_CLK_CLK_PROG_INFO_3X *PLW2080_CTRL_CLK_CLK_PROG_INFO_1X;

/*!
 * Structure describing the static configuration/POR state of the _30 class.
 */
typedef struct LW2080_CTRL_CLK_CLK_PROG_INFO_30 {
    /*!
     * _3X super class.  Must always be first element in structure.
     */
    LW2080_CTRL_CLK_CLK_PROG_INFO_3X super;
} LW2080_CTRL_CLK_CLK_PROG_INFO_30;
typedef struct LW2080_CTRL_CLK_CLK_PROG_INFO_30 *PLW2080_CTRL_CLK_CLK_PROG_INFO_30;

/*!
 * Structure describing the static configuration/POR state of the _35 class.
 */
typedef struct LW2080_CTRL_CLK_CLK_PROG_INFO_35 {
    /*!
     * _3X super class.  Must always be first element in structure.
     */
    LW2080_CTRL_CLK_CLK_PROG_INFO_3X super;

    /*!
     * @ref freqMaxMHz + applied OC adjustments
     *
     * @note This is NOT enabled on PASCAL due to RM - PMU sync issues.
     */
    LwU16                            offsettedFreqMaxMHz;
} LW2080_CTRL_CLK_CLK_PROG_INFO_35;
typedef struct LW2080_CTRL_CLK_CLK_PROG_INFO_35 *PLW2080_CTRL_CLK_CLK_PROG_INFO_35;

/*!
 * Structure of data specific to the _PLL source.
 */
typedef struct LW2080_CTRL_CLK_CLK_PROG_1X_PRIMARY_SOURCE_NAFLL {
    /*!
     * Base voltage from where RM will start smoothing the VF lwrve.
     * RM is only smoothing voltage based VF lwrve, for frequency based VF
     * lwrve we will set this to LW_U32_MAX
     */
    LwU32       baseVFSmoothVoltuV;
    /*!
     * Maximum ramp rate for a given voltage based VF lwrve. RM will ensure
     * that the generated VF lwrve will respect this ramp rate. If the VF lwrve
     * has discontinuity in it, RM will smoothen the VF lwrve using this value.
     * For Frequency based VF lwrve this is don't care. It will be initialized
     * to LW_U32_MAX.
     */
    LwUFXP20_12 maxVFRampRate;
    /*!
     * Maximum allowed frequency difference between two conselwtive VF Points.
     * This value is callwlated based on the @ref maxVFRampRate. RM will use
     * this value to remove the discontinuity from the VF lwrve.
     */
    LwU16       maxFreqStepSizeMHz;
} LW2080_CTRL_CLK_CLK_PROG_1X_PRIMARY_SOURCE_NAFLL;
typedef struct LW2080_CTRL_CLK_CLK_PROG_1X_PRIMARY_SOURCE_NAFLL *PLW2080_CTRL_CLK_CLK_PROG_1X_PRIMARY_SOURCE_NAFLL;

/*!
 * Union of source-specific data.
 */
typedef union LW2080_CTRL_CLK_CLK_PROG_1X_PRIMARY_SOURCE_DATA {
    LW2080_CTRL_CLK_CLK_PROG_1X_PRIMARY_SOURCE_NAFLL nafll;
} LW2080_CTRL_CLK_CLK_PROG_1X_PRIMARY_SOURCE_DATA;

typedef union LW2080_CTRL_CLK_CLK_PROG_1X_PRIMARY_SOURCE_DATA *PLW2080_CTRL_CLK_CLK_PROG_1X_PRIMARY_SOURCE_DATA;

/*!
 * Maximum possible number of VF entries for a _1X_PRIMARY class.
 */
#define LW2080_CTRL_CLK_CLK_PROG_1X_PRIMARY_VF_ENTRY_MAX_ENTRIES 0x4U

/*!
 * Structure describing the a PRIMARY Clock Domain's VF lwrve for a given voltage
 * rail.
 */
typedef struct LW2080_CTRL_CLK_CLK_PROG_1X_PRIMARY_VF_ENTRY {
    /*!
     * VFE index which describes the this CLK_PROG entry's VF equation for the
     * given voltage rail.
     *
     * @ref LW2080_CTRL_PERF_VFE_EQU_INDEX_ILWALID means no VF lwrve specified for
     * the given rail.
     */
    LwU8          vfeIdx;
    /*!
     * VFE index which describes this CLK_PROG entry's NAFLL CPM Max freq
     * offset equation for the given voltage rail.
     *
     * Used only when @ref LW2080_CTRL_CLK_CLK_PROG_INFO_1X::source ==
     * LW2080_CTRL_CLK_PROG_1X_SOURCE_NAFLL.
    *
     * @ref LW2080_CTRL_PERF_VFE_EQU_INDEX_ILWALID means no NAFLL CPM Max specified
     * for the given rail.
     */
    LwU8          cpmMaxFreqOffsetVfeIdx;
    /*!
     * Index of the first CLK_VF_POINT of this object.  The set described by the
     * range [vfPointIdxFirst, @ref vfPointIdxLast] represents the VF lwrve
     * described by this object.  This set is in ascending order of the
     * independent variable of the source (i.e. PLL/ONE_SOURCE -> frequency,
     * NAFLL -> voltage).
     */
    LwBoardObjIdx vfPointIdxFirst;
    /*!
     * Index of the last CLK_VF_POINT of this object.  The set described by the
     * range [@ref  vfPointIdxFirst, vfPointIdxLast] represents the VF lwrve
     * described by this object.  This set is in ascending order of the
     * independent variable of the source (i.e. PLL/ONE_SOURCE -> frequency,
     * NAFLL -> voltage).
     */
    LwBoardObjIdx vfPointIdxLast;
} LW2080_CTRL_CLK_CLK_PROG_1X_PRIMARY_VF_ENTRY;
typedef struct LW2080_CTRL_CLK_CLK_PROG_1X_PRIMARY_VF_ENTRY *PLW2080_CTRL_CLK_CLK_PROG_1X_PRIMARY_VF_ENTRY;

/*!
 * Helper macro to initialize a @ref
 * LW2080_CTRL_CLK_CLK_PROG_1X_PRIMARY_VF_ENTRY structure to its
 * default/disabled state.
 *
 * @param[in] _pSecVfEntry
 *     LW2080_CTRL_CLK_CLK_PROG_1X_PRIMARY_VF_ENTRY to init.
 */
#define LW2080_CTRL_CLK_CLK_PROG_1X_PRIMARY_VF_ENTRY_INIT(_pVfEntry)                         \
    do {                                                                                    \
        (_pVfEntry)->vfeIdx                 = LW2080_CTRL_PERF_VFE_EQU_INDEX_ILWALID_8BIT;  \
        (_pVfEntry)->cpmMaxFreqOffsetVfeIdx = LW2080_CTRL_PERF_VFE_EQU_INDEX_ILWALID_8BIT;  \
        (_pVfEntry)->vfPointIdxFirst        = LW2080_CTRL_CLK_CLK_VF_POINT_IDX_ILWALID;     \
        (_pVfEntry)->vfPointIdxLast         = LW2080_CTRL_CLK_CLK_VF_POINT_IDX_ILWALID;     \
    } while (LW_FALSE)

/*!
 * Structure describing the static configuration/POR state of the _1X_PRIMARY class.
 */
typedef struct LW2080_CTRL_CLK_CLK_PROG_INFO_1X_PRIMARY_WRAPPER {
    /*!
     * _1X super class.  Must always be first element in structure.
     */
    LW2080_CTRL_CLK_CLK_PROG_INFO_1X                super;

    /*!
     * Boolean flag indicating whether this entry supports OC/OV when those
     * settings are applied to the corresponding CLK_DOMAIN object.
     */
    LwBool                                          bOCOVEnabled;
    /*!
     * Array of VF entries.  Indexed per the voltage rail enumeration.
     *
     * Has valid indexes in the range [0, @ref
     * LW2080_CTRL_CLK_CLK_PROGS_INFO::numVfEntries).
     */
    LW2080_CTRL_CLK_CLK_PROG_1X_PRIMARY_VF_ENTRY    vfEntries[LW2080_CTRL_CLK_CLK_PROG_1X_PRIMARY_VF_ENTRY_MAX_ENTRIES];
    /*!
     * Source enumeration.  @ref LW2080_CTRL_CLK_CLK_PROG_1X_SOURCE_<XYZ>.
     * This is duplicate field of PROG_1X. We need this due to XAPI requirement
     */
    LwU8                                            source;

    /*!
     * Union of source-specific data.
     */
    LW2080_CTRL_CLK_CLK_PROG_1X_PRIMARY_SOURCE_DATA sourceData;
} LW2080_CTRL_CLK_CLK_PROG_INFO_1X_PRIMARY_WRAPPER;
typedef struct LW2080_CTRL_CLK_CLK_PROG_INFO_1X_PRIMARY_WRAPPER *PLW2080_CTRL_CLK_CLK_PROG_INFO_1X_PRIMARY_WRAPPER;

/*!
 * Structure describing the static configuration/POR state of the _3X_PRIMARY class.
 */
typedef struct LW2080_CTRL_CLK_CLK_PROG_INFO_3X_PRIMARY {
    /*!
     * BOARDOBJ_INTERFACE super class.  Must always be first element in the structure.
     */
    LW2080_CTRL_BOARDOBJ_INTERFACE                  super;

    /*!
     * Boolean flag indicating whether this entry supports OC/OV when those
     * settings are applied to the corresponding CLK_DOMAIN object.
     */
    LwBool                                          bOCOVEnabled;
    /*!
     * Array of VF entries.  Indexed per the voltage rail enumeration.
     *
     * Has valid indexes in the range [0, @ref
     * LW2080_CTRL_CLK_CLK_PROGS_INFO::numVfEntries).
     */
    LW2080_CTRL_CLK_CLK_PROG_1X_PRIMARY_VF_ENTRY    vfEntries[LW2080_CTRL_CLK_CLK_PROG_1X_PRIMARY_VF_ENTRY_MAX_ENTRIES];
    /*!
     * Source enumeration.  @ref LW2080_CTRL_CLK_CLK_PROG_1X_SOURCE_<XYZ>.
     * This is duplicate field of PROG_1X. We need this due to XAPI requirement
     */
    LwU8                                            source;

    /*!
     * Union of source-specific data.
     */
    LW2080_CTRL_CLK_CLK_PROG_1X_PRIMARY_SOURCE_DATA sourceData;
} LW2080_CTRL_CLK_CLK_PROG_INFO_3X_PRIMARY;
typedef struct LW2080_CTRL_CLK_CLK_PROG_INFO_3X_PRIMARY *PLW2080_CTRL_CLK_CLK_PROG_INFO_3X_PRIMARY;

/*!
 * Structure describing the static configuration/POR state of the _30_PRIMARY class.
 */
typedef struct LW2080_CTRL_CLK_CLK_PROG_INFO_30_PRIMARY {
    /*!
     * _30 super class.  Must always be first element in structure.
     */
    LW2080_CTRL_CLK_CLK_PROG_INFO_30         super;

    /*!
     * _1X_PRIMARY super class.
     */
    LW2080_CTRL_CLK_CLK_PROG_INFO_3X_PRIMARY master;
} LW2080_CTRL_CLK_CLK_PROG_INFO_30_PRIMARY;
typedef struct LW2080_CTRL_CLK_CLK_PROG_INFO_30_PRIMARY *PLW2080_CTRL_CLK_CLK_PROG_INFO_30_PRIMARY;

/*!
 * Macro defining the positions of VF lwrves.
 */
#define LW2080_CTRL_CLK_CLK_PROG_35_PRIMARY_VF_LWRVE_IDX_ILWALID 0xFFU
#define LW2080_CTRL_CLK_CLK_PROG_35_PRIMARY_VF_LWRVE_IDX_PRI     0x00U
#define LW2080_CTRL_CLK_CLK_PROG_35_PRIMARY_VF_LWRVE_IDX_SEC_0   0x01U
#define LW2080_CTRL_CLK_CLK_PROG_35_PRIMARY_VF_LWRVE_IDX_MAX     0x02U // MUST be last.

/*!
 * Structure describing secondary VF lwrves for primary clock domains.
 */
typedef struct LW2080_CTRL_CLK_CLK_PROG_35_PRIMARY_SEC_VF_ENTRY {
    /*!
     * VFE index which describes the this CLK_PROG entry's secondary VF equation for
     * the given voltage rail.
     *
     * Used only when @ref LW2080_CTRL_CLK_CLK_PROG_INFO_1X::source ==
     * LW2080_CTRL_CLK_PROG_1X_SOURCE_NAFLL.
     *
     * @ref LW2080_CTRL_PERF_VFE_EQU_INDEX_ILWALID means no VF lwrve specified for
     * the given rail.
     */
    LwU8          vfeIdx;

    /*!
     * VFE index which describes the this CLK_PROG entry's NAFLL DVCO offset equation
     * for the given voltage rail.
     *
     * Used only when @ref LW2080_CTRL_CLK_CLK_PROG_INFO_1X::source ==
     * LW2080_CTRL_CLK_PROG_1X_SOURCE_NAFLL.
     *
     * @ref LW2080_CTRL_PERF_VFE_EQU_INDEX_ILWALID means no NAFLL DVCO offset specified
     * for the given rail.
     */
    LwU8          dvcoOffsetVfeIdx;

    /*!
     * Index of the first CLK_VF_POINT of this object.  The set described by the
     * range [vfPointIdxFirst, @ref vfPointIdxLast] represents the VF lwrve
     * described by this object.  This set is in ascending order of the
     * independent variable of the source (i.e. PLL/ONE_SOURCE -> frequency,
     * NAFLL -> voltage).
     */
    LwBoardObjIdx vfPointIdxFirst;

    /*!
     * Index of the last CLK_VF_POINT of this object.  The set described by the
     * range [@ref  vfPointIdxFirst, vfPointIdxLast] represents the VF lwrve
     * described by this object.  This set is in ascending order of the
     * independent variable of the source (i.e. PLL/ONE_SOURCE -> frequency,
     * NAFLL -> voltage).
     */
    LwBoardObjIdx vfPointIdxLast;
} LW2080_CTRL_CLK_CLK_PROG_35_PRIMARY_SEC_VF_ENTRY;
typedef struct LW2080_CTRL_CLK_CLK_PROG_35_PRIMARY_SEC_VF_ENTRY *PLW2080_CTRL_CLK_CLK_PROG_35_PRIMARY_SEC_VF_ENTRY;

/*!
 * Helper macro to initialize a @ref
 * LW2080_CTRL_CLK_CLK_PROG_35_PRIMARY_SEC_VF_ENTRY structure to its
 * default/disabled state.
 *
 * @param[in] _pSecVfEntry
 *     LW2080_CTRL_CLK_CLK_PROG_35_PRIMARY_SEC_VF_ENTRY to init.
 */
#define LW2080_CTRL_CLK_CLK_PROG_35_PRIMARY_SEC_VF_ENTRY_INIT(_pSecVfEntry)                 \
    do {                                                                                    \
        (_pSecVfEntry)->vfeIdx            = LW2080_CTRL_PERF_VFE_EQU_INDEX_ILWALID_8BIT;    \
        (_pSecVfEntry)->dvcoOffsetVfeIdx  = LW2080_CTRL_PERF_VFE_EQU_INDEX_ILWALID_8BIT;    \
        (_pSecVfEntry)->vfPointIdxFirst   = LW2080_CTRL_CLK_CLK_VF_POINT_IDX_ILWALID;       \
        (_pSecVfEntry)->vfPointIdxLast    = LW2080_CTRL_CLK_CLK_VF_POINT_IDX_ILWALID;       \
    } while (LW_FALSE)


/*!
 * Structure describing secondary VF lwrves for primary clock domains per volt rail.
 */
typedef struct LW2080_CTRL_CLK_CLK_PROG_35_PRIMARY_SEC_VF_ENTRY_VOLTRAIL {
    /*!
     * Array of secondary VF entries. One per secondary VF lwrve.
     *
     * Used only when @ref LW2080_CTRL_CLK_CLK_PROG_INFO_1X::source ==
     * LW2080_CTRL_CLK_PROG_1X_SOURCE_NAFLL.
     */
    LW2080_CTRL_CLK_CLK_PROG_35_PRIMARY_SEC_VF_ENTRY secVfEntries[LW2080_CTRL_CLK_CLK_PROG_35_PRIMARY_SEC_VF_ENTRY_VOLTRAIL_MAX];
} LW2080_CTRL_CLK_CLK_PROG_35_PRIMARY_SEC_VF_ENTRY_VOLTRAIL;
typedef struct LW2080_CTRL_CLK_CLK_PROG_35_PRIMARY_SEC_VF_ENTRY_VOLTRAIL *PLW2080_CTRL_CLK_CLK_PROG_35_PRIMARY_SEC_VF_ENTRY_VOLTRAIL;

/*!
 * Helper macro to initialize a @ref
 * LW2080_CTRL_CLK_CLK_PROG_35_PRIMARY_SEC_VF_ENTRY_VOLTRAIL structure to its
 * default/disabled state.
 *
 * @param[in] _pSecVfEntryVoltRail
 *     LW2080_CTRL_CLK_CLK_PROG_35_PRIMARY_SEC_VF_ENTRY_VOLTRAIL to init.
 */
#define LW2080_CTRL_CLK_CLK_PROG_35_PRIMARY_SEC_VF_ENTRY_VOLTRAIL_INIT(_pSecVfEntryVoltRail) \
    do {                                                                                     \
        LwU8 i;                                                                              \
        for (i = 0; i < LW2080_CTRL_CLK_CLK_PROG_35_PRIMARY_SEC_VF_ENTRY_VOLTRAIL_MAX;        \
                i++)                                                                         \
        {                                                                                    \
            LW2080_CTRL_CLK_CLK_PROG_35_PRIMARY_SEC_VF_ENTRY_INIT(                           \
                &((_pSecVfEntryVoltRail)->secVfEntries[i]));                                 \
        }                                                                                    \
    } while (LW_FALSE)

/*!
 * Structure describing the static configuration/POR state of the _35_PRIMARY class.
 */
typedef struct LW2080_CTRL_CLK_CLK_PROG_INFO_35_PRIMARY {
    /*!
     * _35 super class.  Must always be first element in structure.
     */
    LW2080_CTRL_CLK_CLK_PROG_INFO_35                          super;

    /*!
     * _1X_PRIMARY super class.
     */
    LW2080_CTRL_CLK_CLK_PROG_INFO_3X_PRIMARY                  master;

    /*!
     * Array of secondary VF entries. Indexed per the voltage rail enumeration.
     *
     * Has valid indexes in the range [0, @ref
     * LW2080_CTRL_CLK_CLK_PROGS_INFO::numVfEntries).
     */
    LW2080_CTRL_CLK_CLK_PROG_35_PRIMARY_SEC_VF_ENTRY_VOLTRAIL voltRailSecVfEntries[LW2080_CTRL_CLK_CLK_PROG_1X_PRIMARY_VF_ENTRY_MAX_ENTRIES];
} LW2080_CTRL_CLK_CLK_PROG_INFO_35_PRIMARY;
typedef struct LW2080_CTRL_CLK_CLK_PROG_INFO_35_PRIMARY *PLW2080_CTRL_CLK_CLK_PROG_INFO_35_PRIMARY;

/*!
 */
#define LW2080_CTRL_CLK_PROG_1X_PRIMARY_MAX_SECONDARY_ENTRIES 0x6U

/*!
 * Structure describing a RATIO PRIMARY-SECONDARY relationship which specifies the
 * SECONDARY Clock Domain's VF lwrve as a ratio function of the PRIMARY Clock
 * Domain's VF lwrve.
 *
 * https://wiki.lwpu.com/engwiki/index.php/Resman/PState/Data_Tables/Clock_Programming_Table/1.0_Spec#Ratio
 */
typedef struct LW2080_CTRL_CLK_CLK_PROG_1X_PRIMARY_RATIO_SECONDARY_ENTRY {
    /*!
     * CLK_DOMAIN index for SECONDARY Clock Domain specified in this entry.  @ref
     * LW2080_CTRL_CLK_CLK_DOMAIN_INDEX_ILWALID indicates no CLK_DOMAIN
     * specified.
     */
    LwU8 clkDomIdx;
    /*!
     * Ratio specified for the SECONDARY Clock Domain as a function of the PRIMARY
     * Clock Domain. Must be in the range [0, 100].
     */
    LwU8 ratio;
} LW2080_CTRL_CLK_CLK_PROG_1X_PRIMARY_RATIO_SECONDARY_ENTRY;
typedef struct LW2080_CTRL_CLK_CLK_PROG_1X_PRIMARY_RATIO_SECONDARY_ENTRY *PLW2080_CTRL_CLK_CLK_PROG_1X_PRIMARY_RATIO_SECONDARY_ENTRY;

/*!
 * Structure describing the static configuration/POR state of the _1X_PRIMARY class.
 */
typedef struct LW2080_CTRL_CLK_CLK_PROG_INFO_1X_PRIMARY_RATIO_WRAPPER {
    /*!
     * _1X_PRIMARY super class.  Must always be first element in structure.
     */
    LW2080_CTRL_CLK_CLK_PROG_INFO_1X_PRIMARY_WRAPPER          super;

    /*!
     * Array of ratio entries.
     *
     * Has valid indexes in the range [0, @ref
     * LW2080_CTRL_CLK_CLK_PROGS_INFO::numSecondaryEntries).
     */
    LW2080_CTRL_CLK_CLK_PROG_1X_PRIMARY_RATIO_SECONDARY_ENTRY slaveEntries[LW2080_CTRL_CLK_PROG_1X_PRIMARY_MAX_SECONDARY_ENTRIES];
} LW2080_CTRL_CLK_CLK_PROG_INFO_1X_PRIMARY_RATIO_WRAPPER;
typedef struct LW2080_CTRL_CLK_CLK_PROG_INFO_1X_PRIMARY_RATIO_WRAPPER *PLW2080_CTRL_CLK_CLK_PROG_INFO_1X_PRIMARY_RATIO_WRAPPER;

/*!
 * Structure describing the static configuration/POR state of the _3X_PRIMARY_RATIO class.
 */
typedef struct LW2080_CTRL_CLK_CLK_PROG_INFO_3X_PRIMARY_RATIO {
    /*!
     * BOARDOBJ_INTERFACE super class.  Must always be first element in the structure.
     */
    LW2080_CTRL_BOARDOBJ_INTERFACE                            super;

    /*!
     * Array of ratio entries.
     *
     * Has valid indexes in the range [0, @ref
     * LW2080_CTRL_CLK_CLK_PROGS_INFO::numSecondaryEntries).
     */
    LW2080_CTRL_CLK_CLK_PROG_1X_PRIMARY_RATIO_SECONDARY_ENTRY slaveEntries[LW2080_CTRL_CLK_PROG_1X_PRIMARY_MAX_SECONDARY_ENTRIES];
} LW2080_CTRL_CLK_CLK_PROG_INFO_3X_PRIMARY_RATIO;
typedef struct LW2080_CTRL_CLK_CLK_PROG_INFO_3X_PRIMARY_RATIO *PLW2080_CTRL_CLK_CLK_PROG_INFO_3X_PRIMARY_RATIO;

/*!
 * Structure describing the static configuration/POR state of the _30_PRIMARY_RATIO class.
 */
typedef struct LW2080_CTRL_CLK_CLK_PROG_INFO_30_PRIMARY_RATIO {
    /*!
     * _30_PRIMARY super class. Must always be first element in structure.
     */
    LW2080_CTRL_CLK_CLK_PROG_INFO_30_PRIMARY       super;

    /*!
     * _3X_PRIMARY_RATIO super class.
     */
    LW2080_CTRL_CLK_CLK_PROG_INFO_3X_PRIMARY_RATIO ratio;
} LW2080_CTRL_CLK_CLK_PROG_INFO_30_PRIMARY_RATIO;
typedef struct LW2080_CTRL_CLK_CLK_PROG_INFO_30_PRIMARY_RATIO *PLW2080_CTRL_CLK_CLK_PROG_INFO_30_PRIMARY_RATIO;

/*!
 * Structure describing the static configuration/POR state of the _35_PRIMARY_RATIO class.
 */
typedef struct LW2080_CTRL_CLK_CLK_PROG_INFO_35_PRIMARY_RATIO {
    /*!
     * _35_PRIMARY super class. Must always be first element in structure.
     */
    LW2080_CTRL_CLK_CLK_PROG_INFO_35_PRIMARY       super;

    /*!
     * _3X_PRIMARY_RATIO super class.
     */
    LW2080_CTRL_CLK_CLK_PROG_INFO_3X_PRIMARY_RATIO ratio;
} LW2080_CTRL_CLK_CLK_PROG_INFO_35_PRIMARY_RATIO;
typedef struct LW2080_CTRL_CLK_CLK_PROG_INFO_35_PRIMARY_RATIO *PLW2080_CTRL_CLK_CLK_PROG_INFO_35_PRIMARY_RATIO;

/*!
 * Structure describing a TABLE PRIMARY-SECONDARY relationship which specifies the
 * SECONDARY Clock Domain's VF lwrve is a table-lookup function of the PRIMARY Clock
 * Domain's VF lwrve.
 *
 * https://wiki.lwpu.com/engwiki/index.php/Resman/PState/Data_Tables/Clock_Programming_Table/1.0_Spec#Table
 */
typedef struct LW2080_CTRL_CLK_CLK_PROG_1X_PRIMARY_TABLE_SECONDARY_ENTRY {
    /*!
     * CLK_DOMAIN index for SECONDARY Clock Domain specified in this entry.  @ref
     * LW2080_CTRL_CLK_CLK_DOMAIN_INDEX_ILWALID indicates no CLK_DOMAIN
     * specified.
     */
    LwU8  clkDomIdx;
    /*!
     * Frequency specified for the SECONDARY Clock Domain for the corresponding
     * frequency on the PRIMARY Clock Domain.
     */
    LwU16 freqMHz;
} LW2080_CTRL_CLK_CLK_PROG_1X_PRIMARY_TABLE_SECONDARY_ENTRY;
typedef struct LW2080_CTRL_CLK_CLK_PROG_1X_PRIMARY_TABLE_SECONDARY_ENTRY *PLW2080_CTRL_CLK_CLK_PROG_1X_PRIMARY_TABLE_SECONDARY_ENTRY;

/*!
 * Structure describing the static configuration/POR state of the _1X_PRIMARY class.
 */
typedef struct LW2080_CTRL_CLK_CLK_PROG_INFO_1X_PRIMARY_TABLE_WRAPPER {
    /*!
     * _1X_PRIMARY super class.  Must always be first element in structure.
     */
    LW2080_CTRL_CLK_CLK_PROG_INFO_1X_PRIMARY_WRAPPER          super;

    /*!
     * Array of table entries.
     *
     * Has valid indexes in the range [0, @ref
     * LW2080_CTRL_CLK_CLK_PROGS_INFO::numSecondaryEntries).
     */
    LW2080_CTRL_CLK_CLK_PROG_1X_PRIMARY_TABLE_SECONDARY_ENTRY slaveEntries[LW2080_CTRL_CLK_PROG_1X_PRIMARY_MAX_SECONDARY_ENTRIES];
} LW2080_CTRL_CLK_CLK_PROG_INFO_1X_PRIMARY_TABLE_WRAPPER;
typedef struct LW2080_CTRL_CLK_CLK_PROG_INFO_1X_PRIMARY_TABLE_WRAPPER *PLW2080_CTRL_CLK_CLK_PROG_INFO_1X_PRIMARY_TABLE_WRAPPER;

/*!
 * Structure describing the static configuration/POR state of the _3X_PRIMARY_TABLE class.
 */
typedef struct LW2080_CTRL_CLK_CLK_PROG_INFO_3X_PRIMARY_TABLE {
    /*!
     * BOARDOBJ_INTERFACE super class.  Must always be first element in the structure.
     */
    LW2080_CTRL_BOARDOBJ_INTERFACE                            super;

    /*!
     * Array of table entries.
     *
     * Has valid indexes in the range [0, @ref
     * LW2080_CTRL_CLK_CLK_PROGS_INFO::numSecondaryEntries).
     */
    LW2080_CTRL_CLK_CLK_PROG_1X_PRIMARY_TABLE_SECONDARY_ENTRY slaveEntries[LW2080_CTRL_CLK_PROG_1X_PRIMARY_MAX_SECONDARY_ENTRIES];
} LW2080_CTRL_CLK_CLK_PROG_INFO_3X_PRIMARY_TABLE;
typedef struct LW2080_CTRL_CLK_CLK_PROG_INFO_3X_PRIMARY_TABLE *PLW2080_CTRL_CLK_CLK_PROG_INFO_3X_PRIMARY_TABLE;

/*!
 * Structure describing the static configuration/POR state of the _30_PRIMARY_TABLE class.
 */
typedef struct LW2080_CTRL_CLK_CLK_PROG_INFO_30_PRIMARY_TABLE {
    /*!
     * _30_PRIMARY super class. Must always be first element in structure.
     */
    LW2080_CTRL_CLK_CLK_PROG_INFO_30_PRIMARY       super;

    /*!
     * _3X_PRIMARY_RATIO super class.
     */
    LW2080_CTRL_CLK_CLK_PROG_INFO_3X_PRIMARY_TABLE table;
} LW2080_CTRL_CLK_CLK_PROG_INFO_30_PRIMARY_TABLE;
typedef struct LW2080_CTRL_CLK_CLK_PROG_INFO_30_PRIMARY_TABLE *PLW2080_CTRL_CLK_CLK_PROG_INFO_30_PRIMARY_TABLE;

/*!
 * Structure describing the static configuration/POR state of the _35_PRIMARY_TABLE class.
 */
typedef struct LW2080_CTRL_CLK_CLK_PROG_INFO_35_PRIMARY_TABLE {
    /*!
     * _35_PRIMARY super class. Must always be first element in structure.
     */
    LW2080_CTRL_CLK_CLK_PROG_INFO_35_PRIMARY       super;

    /*!
     * _3X_PRIMARY_RATIO super class.
     */
    LW2080_CTRL_CLK_CLK_PROG_INFO_3X_PRIMARY_TABLE table;
} LW2080_CTRL_CLK_CLK_PROG_INFO_35_PRIMARY_TABLE;
typedef struct LW2080_CTRL_CLK_CLK_PROG_INFO_35_PRIMARY_TABLE *PLW2080_CTRL_CLK_CLK_PROG_INFO_35_PRIMARY_TABLE;

/*!
 * CLK_PROG type-specific data union.  Discriminated by
 * CLK_PROG::super.type.
 */


/*!
 * Structure describing CLK_PROG static information/POR.  Implements the
 * BOARDOBJ model/interface.
 */
typedef struct LW2080_CTRL_CLK_CLK_PROG_INFO {
    /*!
     * LW2080_CTRL_BOARDOBJ super class.  Must always be first object in
     * structure.
     */
    LW2080_CTRL_BOARDOBJ super;
    /*!
     * XAPI does not support using an element within another structure as a
     * discriminant.  Placing this redundant type value here until that design
     * constraint can be fixed.
     */
    LwU8                 type;

    /*!
     * Type-specific data union.
     */
    union {
        LW2080_CTRL_CLK_CLK_PROG_INFO_1X                       v1x;
        LW2080_CTRL_CLK_CLK_PROG_INFO_30                       v30;
        LW2080_CTRL_CLK_CLK_PROG_INFO_35                       v35;
        LW2080_CTRL_CLK_CLK_PROG_INFO_1X_PRIMARY_WRAPPER       v1xMaster;
        LW2080_CTRL_CLK_CLK_PROG_INFO_30_PRIMARY               v30Master;
        LW2080_CTRL_CLK_CLK_PROG_INFO_35_PRIMARY               v35Master;
        LW2080_CTRL_CLK_CLK_PROG_INFO_1X_PRIMARY_RATIO_WRAPPER v1xMasterRatio;
        LW2080_CTRL_CLK_CLK_PROG_INFO_1X_PRIMARY_TABLE_WRAPPER v1xMasterTable;
        LW2080_CTRL_CLK_CLK_PROG_INFO_30_PRIMARY_RATIO         v30MasterRatio;
        LW2080_CTRL_CLK_CLK_PROG_INFO_30_PRIMARY_TABLE         v30MasterTable;
        LW2080_CTRL_CLK_CLK_PROG_INFO_35_PRIMARY_RATIO         v35MasterRatio;
        LW2080_CTRL_CLK_CLK_PROG_INFO_35_PRIMARY_TABLE         v35MasterTable;
    } data;
} LW2080_CTRL_CLK_CLK_PROG_INFO;
typedef struct LW2080_CTRL_CLK_CLK_PROG_INFO *PLW2080_CTRL_CLK_CLK_PROG_INFO;

/*!
 * Structure describing CLK_PROGS static information/POR.  Implements the
 * BOARDOBJGRP model/interface.
 */
#define LW2080_CTRL_CLK_CLK_PROGS_INFO_MESSAGE_ID (0x1DU)

typedef struct LW2080_CTRL_CLK_CLK_PROGS_INFO {
    /*!
     * LW2080_CTRL_BOARDOBJGRP_E255 super class.  Must always be first object in
     * structure.
     */
    LW2080_CTRL_BOARDOBJGRP_E255  super;

    /*!
     * Number of Secondary entries per _PRIMARY entry.
     */
    LwU8                          slaveEntryCount;
    /*!
     * Number of VF entries per each _PRIMARY entry.
     */
    LwU8                          vfEntryCount;
    /*!
     * Number of secondary VF entries per each _PRIMARY entry.
     */
    LwU8                          vfSecEntryCount;
    /*!
     * Array of CLK_PROG structures.  Has valid indexes corresponding to the
     * bits set in @ref super.objMask.
     */
    LW2080_CTRL_CLK_CLK_PROG_INFO progs[LW2080_CTRL_BOARDOBJGRP_E255_MAX_OBJECTS];
} LW2080_CTRL_CLK_CLK_PROGS_INFO;
typedef struct LW2080_CTRL_CLK_CLK_PROGS_INFO *PLW2080_CTRL_CLK_CLK_PROGS_INFO;

/*!
 * LW2080_CTRL_CMD_CLK_CLK_PROGS_GET_INFO
 *
 * This command returns CLK_PROGS static object information/POR as specified
 * by the VBIOS in the Clocks Programming Table.
 *
 * The CLK_PROG objects are indexed per how they are specified in the VBIOS
 * table.
 *
 * See @ref LW2080_CTRL_CLK_CLK_PROGS_INFO for documentation on the
 * parameters.
 *
 * Return values are specified per @ref BoardObjGrpGetInfo.
 */
#define LW2080_CTRL_CMD_CLK_CLK_PROGS_GET_INFO (0x2080101d) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_CLK_INTERFACE_ID << 8) | LW2080_CTRL_CLK_CLK_PROGS_INFO_MESSAGE_ID" */

/*!
 * Structure describing the dynamic state of the _1X class.
 */
typedef struct LW2080_CTRL_CLK_CLK_PROG_STATUS_3X {
    /*!
     * Maximum frequency for this CLK_PROG entry adjusted with the applied
     * frequency offsets on given clock domain for this programming index
     */
    LwU16 freqMaxMHz;
} LW2080_CTRL_CLK_CLK_PROG_STATUS_3X;
typedef struct LW2080_CTRL_CLK_CLK_PROG_STATUS_3X *PLW2080_CTRL_CLK_CLK_PROG_STATUS_3X;

typedef LW2080_CTRL_CLK_CLK_PROG_STATUS_3X LW2080_CTRL_CLK_CLK_PROG_STATUS_1X;
typedef struct LW2080_CTRL_CLK_CLK_PROG_STATUS_3X *PLW2080_CTRL_CLK_CLK_PROG_STATUS_1X;

/*!
 * Structure describing the dynamic state of the _30 class.
 */
typedef struct LW2080_CTRL_CLK_CLK_PROG_STATUS_30 {
    /*!
     * _3X super class. Must always be first element in structure.
     */
    LW2080_CTRL_CLK_CLK_PROG_STATUS_3X super;

    /*!
     * Maximum frequency for this CLK_PROG entry adjusted with the applied
     * frequency offsets on given clock domain for this programming index
     */
    LwU16                              offsettedFreqMaxMHz;
} LW2080_CTRL_CLK_CLK_PROG_STATUS_30;
typedef struct LW2080_CTRL_CLK_CLK_PROG_STATUS_30 *PLW2080_CTRL_CLK_CLK_PROG_STATUS_30;

/*!
 * Structure describing the dynamic state of the _35 class.
 */
typedef struct LW2080_CTRL_CLK_CLK_PROG_STATUS_35 {
    /*!
     * _3X super class. Must always be first element in structure.
     */
    LW2080_CTRL_CLK_CLK_PROG_STATUS_3X super;

    /*!
     * @ref freqMaxMHz + applied OC adjustments
     *
     * @note This is NOT enabled on PASCAL due to RM - PMU sync issues.
     */
    LwU16                              offsettedFreqMaxMHz;
} LW2080_CTRL_CLK_CLK_PROG_STATUS_35;
typedef struct LW2080_CTRL_CLK_CLK_PROG_STATUS_35 *PLW2080_CTRL_CLK_CLK_PROG_STATUS_35;

/*!
 * CLK_PROG type-specific data union.  Discriminated by
 * CLK_PROG::super.type.
 */


/*!
 * Structure describing CLK_PROG dynamic information. Implements the
 * BOARDOBJ model/interface.
 */
typedef struct LW2080_CTRL_CLK_CLK_PROG_STATUS {
    /*!
     * LW2080_CTRL_BOARDOBJ super class.  Must always be first object in
     * structure.
     */
    LW2080_CTRL_BOARDOBJ super;
    /*!
     * XAPI does not support using an element within another structure as a
     * discriminant.  Placing this redundant type value here until that design
     * constraint can be fixed.
     */
    LwU8                 type;

    /*!
     * Type-specific data union.
     */
    union {
        LW2080_CTRL_CLK_CLK_PROG_STATUS_3X v1x;
        LW2080_CTRL_CLK_CLK_PROG_STATUS_30 v30;
        LW2080_CTRL_CLK_CLK_PROG_STATUS_35 v35;
    } data;
} LW2080_CTRL_CLK_CLK_PROG_STATUS;
typedef struct LW2080_CTRL_CLK_CLK_PROG_STATUS *PLW2080_CTRL_CLK_CLK_PROG_STATUS;

/*!
 * Structure describing CLK_PROGS dynamic information. Implements the
 * BOARDOBJGRP model/interface.
 */
#define LW2080_CTRL_CLK_CLK_PROGS_STATUS_MESSAGE_ID (0x1EU)

typedef struct LW2080_CTRL_CLK_CLK_PROGS_STATUS {
    /*!
     * LW2080_CTRL_BOARDOBJGRP_E255 super class.  Must always be first object in
     * structure.
     */
    LW2080_CTRL_BOARDOBJGRP_E255    super;

    /*!
     * Array of CLK_PROG structures.  Has valid indexes corresponding to the
     * bits set in @ref super.objMask.
     */
    LW2080_CTRL_CLK_CLK_PROG_STATUS progs[LW2080_CTRL_BOARDOBJGRP_E255_MAX_OBJECTS];
} LW2080_CTRL_CLK_CLK_PROGS_STATUS;
typedef struct LW2080_CTRL_CLK_CLK_PROGS_STATUS *PLW2080_CTRL_CLK_CLK_PROGS_STATUS;

/*!
 * LW2080_CTRL_CMD_CLK_CLK_PROGS_GET_STATUS
 *
 * This command returns the CLK_PROGS dynamic state information associated by the
 * Clk Programming functionality
 *
 * See @ref LW2080_CTRL_CLK_CLK_PROGS_STATUS for documentation on the
 * parameters.
 *
 * Return values are specified per @ref BoardObjGrpGetStatus.
 */
#define LW2080_CTRL_CMD_CLK_CLK_PROGS_GET_STATUS (0x2080101e) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_CLK_INTERFACE_ID << 8) | LW2080_CTRL_CLK_CLK_PROGS_STATUS_MESSAGE_ID" */

/*!
 * Structure describing the control parameters associated with the _3X class.
 */
typedef struct LW2080_CTRL_CLK_CLK_PROG_CONTROL_3X {
    /*!
     * CLK_PROG_3X does not contain any control parameters as of now.
     */
    LwU8 rsvd;
} LW2080_CTRL_CLK_CLK_PROG_CONTROL_3X;
typedef struct LW2080_CTRL_CLK_CLK_PROG_CONTROL_3X *PLW2080_CTRL_CLK_CLK_PROG_CONTROL_3X;

/*!
 * Structure describing the control parameters associated with the _1X_PRIMARY class.
 */
typedef struct LW2080_CTRL_CLK_CLK_PROG_CONTROL_3X_PRIMARY {
    /*!
     * BOARDOBJ_INTERFACE super class.  Must always be first element in the structure.
     */
    LW2080_CTRL_BOARDOBJ_INTERFACE super;

    /*!
     * Deltas for _1X_PRIMARY
     */
    LW2080_CTRL_CLK_CLK_DELTA      deltas;
} LW2080_CTRL_CLK_CLK_PROG_CONTROL_3X_PRIMARY;
typedef struct LW2080_CTRL_CLK_CLK_PROG_CONTROL_3X_PRIMARY *PLW2080_CTRL_CLK_CLK_PROG_CONTROL_3X_PRIMARY;

// PP-TODO for barkward compatibility.
typedef LW2080_CTRL_CLK_CLK_PROG_CONTROL_3X_PRIMARY LW2080_CTRL_CLK_CLK_PROG_CONTROL_1X_PRIMARY;
typedef struct LW2080_CTRL_CLK_CLK_PROG_CONTROL_3X_PRIMARY *PLW2080_CTRL_CLK_CLK_PROG_CONTROL_1X_PRIMARY;

/*!
 * Structure describing the control parameters associated with the _30_PRIMARY class.
 */
typedef struct LW2080_CTRL_CLK_CLK_PROG_CONTROL_30_PRIMARY {
    /*!
     * _1X_PRIMARY super class.
     */
    LW2080_CTRL_CLK_CLK_PROG_CONTROL_3X_PRIMARY master;
} LW2080_CTRL_CLK_CLK_PROG_CONTROL_30_PRIMARY;
typedef struct LW2080_CTRL_CLK_CLK_PROG_CONTROL_30_PRIMARY *PLW2080_CTRL_CLK_CLK_PROG_CONTROL_30_PRIMARY;

/*!
 * Structure describing the control parameters associated with the _35_PRIMARY class.
 */
typedef struct LW2080_CTRL_CLK_CLK_PROG_CONTROL_35_PRIMARY {
    /*!
     * _1X_PRIMARY super class.
     */
    LW2080_CTRL_CLK_CLK_PROG_CONTROL_3X_PRIMARY master;
} LW2080_CTRL_CLK_CLK_PROG_CONTROL_35_PRIMARY;
typedef struct LW2080_CTRL_CLK_CLK_PROG_CONTROL_35_PRIMARY *PLW2080_CTRL_CLK_CLK_PROG_CONTROL_35_PRIMARY;

/*!
 * Structure describing the control parameters associated with the _1X_PRIMARY class.
 */
typedef struct LW2080_CTRL_CLK_CLK_PROG_CONTROL_1X_PRIMARY_RATIO_WRAPPER {
    /*!
     * _1X_PRIMARY super class.  Must always be first element in structure.
     */
    LW2080_CTRL_CLK_CLK_PROG_CONTROL_1X_PRIMARY               super;
    /*!
     * Array of ratio entries.
     *
     * Has valid indexes in the range [0, @ref
     * LW2080_CTRL_CLK_CLK_PROGS_INFO::numSecondaryEntries).
     */
    LW2080_CTRL_CLK_CLK_PROG_1X_PRIMARY_RATIO_SECONDARY_ENTRY slaveEntries[LW2080_CTRL_CLK_PROG_1X_PRIMARY_MAX_SECONDARY_ENTRIES];
} LW2080_CTRL_CLK_CLK_PROG_CONTROL_1X_PRIMARY_RATIO_WRAPPER;
typedef struct LW2080_CTRL_CLK_CLK_PROG_CONTROL_1X_PRIMARY_RATIO_WRAPPER *PLW2080_CTRL_CLK_CLK_PROG_CONTROL_1X_PRIMARY_RATIO_WRAPPER;

/*!
 * Structure describing the control parameters associated with the _3X_PRIMARY_RATIO class.
 */
typedef struct LW2080_CTRL_CLK_CLK_PROG_CONTROL_3X_PRIMARY_RATIO {
    /*!
     * BOARDOBJ_INTERFACE super class.  Must always be first element in the structure.
     */
    LW2080_CTRL_BOARDOBJ_INTERFACE                            super;

    /*!
     * Array of ratio entries.
     *
     * Has valid indexes in the range [0, @ref
     * LW2080_CTRL_CLK_CLK_PROGS_INFO::numSecondaryEntries).
     */
    LW2080_CTRL_CLK_CLK_PROG_1X_PRIMARY_RATIO_SECONDARY_ENTRY slaveEntries[LW2080_CTRL_CLK_PROG_1X_PRIMARY_MAX_SECONDARY_ENTRIES];
} LW2080_CTRL_CLK_CLK_PROG_CONTROL_3X_PRIMARY_RATIO;
typedef struct LW2080_CTRL_CLK_CLK_PROG_CONTROL_3X_PRIMARY_RATIO *PLW2080_CTRL_CLK_CLK_PROG_CONTROL_3X_PRIMARY_RATIO;

/*!
 * Structure describing the control parameters associated with the _30_PRIMARY_RATIO class.
 */
typedef struct LW2080_CTRL_CLK_CLK_PROG_CONTROL_30_PRIMARY_RATIO {
    /*!
     * _30_PRIMARY super class.  Must always be first element in structure.
     */
    LW2080_CTRL_CLK_CLK_PROG_CONTROL_30_PRIMARY       super;

    /*!
     * _3X_PRIMARY_RATIO super class.
     */
    LW2080_CTRL_CLK_CLK_PROG_CONTROL_3X_PRIMARY_RATIO ratio;
} LW2080_CTRL_CLK_CLK_PROG_CONTROL_30_PRIMARY_RATIO;
typedef struct LW2080_CTRL_CLK_CLK_PROG_CONTROL_30_PRIMARY_RATIO *PLW2080_CTRL_CLK_CLK_PROG_CONTROL_30_PRIMARY_RATIO;

/*!
 * Structure describing the control parameters associated with the _35_PRIMARY_RATIO class.
 */
typedef struct LW2080_CTRL_CLK_CLK_PROG_CONTROL_35_PRIMARY_RATIO {
    /*!
     * _35_PRIMARY super class.  Must always be first element in structure.
     */
    LW2080_CTRL_CLK_CLK_PROG_CONTROL_35_PRIMARY       super;

    /*!
     * _3X_PRIMARY_RATIO super class.
     */
    LW2080_CTRL_CLK_CLK_PROG_CONTROL_3X_PRIMARY_RATIO ratio;
} LW2080_CTRL_CLK_CLK_PROG_CONTROL_35_PRIMARY_RATIO;
typedef struct LW2080_CTRL_CLK_CLK_PROG_CONTROL_35_PRIMARY_RATIO *PLW2080_CTRL_CLK_CLK_PROG_CONTROL_35_PRIMARY_RATIO;

/*!
 * Structure describing the control parameters associated with the _1X_PRIMARY class.
 */
typedef struct LW2080_CTRL_CLK_CLK_PROG_CONTROL_1X_PRIMARY_TABLE_WRAPPER {
    /*!
     * _1X_PRIMARY super class.  Must always be first element in structure.
     */
    LW2080_CTRL_CLK_CLK_PROG_CONTROL_1X_PRIMARY               super;
    /*!
     * Array of table entries.
     *
     * Has valid indexes in the range [0, @ref
     * LW2080_CTRL_CLK_CLK_PROGS_INFO::numSecondaryEntries).
     */
    LW2080_CTRL_CLK_CLK_PROG_1X_PRIMARY_TABLE_SECONDARY_ENTRY slaveEntries[LW2080_CTRL_CLK_PROG_1X_PRIMARY_MAX_SECONDARY_ENTRIES];
} LW2080_CTRL_CLK_CLK_PROG_CONTROL_1X_PRIMARY_TABLE_WRAPPER;
typedef struct LW2080_CTRL_CLK_CLK_PROG_CONTROL_1X_PRIMARY_TABLE_WRAPPER *PLW2080_CTRL_CLK_CLK_PROG_CONTROL_1X_PRIMARY_TABLE_WRAPPER;

/*!
 * Structure describing the control parameters associated with the _3X_PRIMARY_TABLE class.
 */
typedef struct LW2080_CTRL_CLK_CLK_PROG_CONTROL_3X_PRIMARY_TABLE {
    /*!
     * BOARDOBJ_INTERFACE super class.  Must always be first element in the structure.
     */
    LW2080_CTRL_BOARDOBJ_INTERFACE                            super;

    /*!
     * Array of table entries.
     *
     * Has valid indexes in the range [0, @ref
     * LW2080_CTRL_CLK_CLK_PROGS_INFO::numSecondaryEntries).
     */
    LW2080_CTRL_CLK_CLK_PROG_1X_PRIMARY_TABLE_SECONDARY_ENTRY slaveEntries[LW2080_CTRL_CLK_PROG_1X_PRIMARY_MAX_SECONDARY_ENTRIES];
} LW2080_CTRL_CLK_CLK_PROG_CONTROL_3X_PRIMARY_TABLE;
typedef struct LW2080_CTRL_CLK_CLK_PROG_CONTROL_3X_PRIMARY_TABLE *PLW2080_CTRL_CLK_CLK_PROG_CONTROL_3X_PRIMARY_TABLE;

/*!
 * Structure describing the control parameters associated with the _30_PRIMARY_TABLE class.
 */
typedef struct LW2080_CTRL_CLK_CLK_PROG_CONTROL_30_PRIMARY_TABLE {
    /*!
     * _30_PRIMARY super class.  Must always be first element in structure.
     */
    LW2080_CTRL_CLK_CLK_PROG_CONTROL_30_PRIMARY       super;

    /*!
     * _3X_PRIMARY_TABLE super class.
     */
    LW2080_CTRL_CLK_CLK_PROG_CONTROL_3X_PRIMARY_TABLE table;
} LW2080_CTRL_CLK_CLK_PROG_CONTROL_30_PRIMARY_TABLE;
typedef struct LW2080_CTRL_CLK_CLK_PROG_CONTROL_30_PRIMARY_TABLE *PLW2080_CTRL_CLK_CLK_PROG_CONTROL_30_PRIMARY_TABLE;

/*!
 * Structure describing the control parameters associated with the _35_PRIMARY_TABLE class.
 */
typedef struct LW2080_CTRL_CLK_CLK_PROG_CONTROL_35_PRIMARY_TABLE {
    /*!
     * _35_PRIMARY super class.  Must always be first element in structure.
     */
    LW2080_CTRL_CLK_CLK_PROG_CONTROL_35_PRIMARY       super;

    /*!
     * _3X_PRIMARY_TABLE super class.
     */
    LW2080_CTRL_CLK_CLK_PROG_CONTROL_3X_PRIMARY_TABLE table;
} LW2080_CTRL_CLK_CLK_PROG_CONTROL_35_PRIMARY_TABLE;
typedef struct LW2080_CTRL_CLK_CLK_PROG_CONTROL_35_PRIMARY_TABLE *PLW2080_CTRL_CLK_CLK_PROG_CONTROL_35_PRIMARY_TABLE;

/*!
 * CLK_PROG type-specific data union.  Discriminated by
 * CLK_PROG::super.type.
 */


/*!
 * Structure describing CLK_PROG specific control parameters.  Implements the
 * BOARDOBJ model/interface.
 */
typedef struct LW2080_CTRL_CLK_CLK_PROG_CONTROL {
    /*!
     * LW2080_CTRL_BOARDOBJ super class.  Must always be first object in
     * structure.
     */
    LW2080_CTRL_BOARDOBJ super;
    /*!
     * XAPI does not support using an element within another structure as a
     * discriminant.  Placing this redundant type value here until that design
     * constraint can be fixed.
     */
    LwU8                 type;

    /*!
     * Type-specific data union.
     */
    union {
        LW2080_CTRL_CLK_CLK_PROG_CONTROL_3X                       v3x;
        LW2080_CTRL_CLK_CLK_PROG_CONTROL_3X_PRIMARY               v1xMaster;
        LW2080_CTRL_CLK_CLK_PROG_CONTROL_1X_PRIMARY_RATIO_WRAPPER v1xMasterRatio;
        LW2080_CTRL_CLK_CLK_PROG_CONTROL_1X_PRIMARY_TABLE_WRAPPER v1xMasterTable;
        LW2080_CTRL_CLK_CLK_PROG_CONTROL_30_PRIMARY               v30Master;
        LW2080_CTRL_CLK_CLK_PROG_CONTROL_30_PRIMARY_RATIO         v30MasterRatio;
        LW2080_CTRL_CLK_CLK_PROG_CONTROL_30_PRIMARY_TABLE         v30MasterTable;
        LW2080_CTRL_CLK_CLK_PROG_CONTROL_35_PRIMARY               v35Master;
        LW2080_CTRL_CLK_CLK_PROG_CONTROL_35_PRIMARY_RATIO         v35MasterRatio;
        LW2080_CTRL_CLK_CLK_PROG_CONTROL_35_PRIMARY_TABLE         v35MasterTable;
    } data;
} LW2080_CTRL_CLK_CLK_PROG_CONTROL;
typedef struct LW2080_CTRL_CLK_CLK_PROG_CONTROL *PLW2080_CTRL_CLK_CLK_PROG_CONTROL;

/*!
 * Structure describing CLK_PROGS specific control parameters.  Implements the
 * BOARDOBJGRP model/interface.
 */
typedef struct LW2080_CTRL_CLK_CLK_PROGS_CONTROL {
    /*!
     * LW2080_CTRL_BOARDOBJGRP_E255 super class.  Must always be first object in
     * structure.
     */
    LW2080_CTRL_BOARDOBJGRP_E255     super;
    /*!
     * Array of CLK_PROG structures.  Has valid indexes corresponding to the
     * bits set in @ref super.objMask.
     */
    LW2080_CTRL_CLK_CLK_PROG_CONTROL progs[LW2080_CTRL_BOARDOBJGRP_E255_MAX_OBJECTS];
} LW2080_CTRL_CLK_CLK_PROGS_CONTROL;
typedef struct LW2080_CTRL_CLK_CLK_PROGS_CONTROL *PLW2080_CTRL_CLK_CLK_PROGS_CONTROL;

/*!
 * LW2080_CTRL_CMD_CLK_CLK_PROGS_GET_CONTROL
 *
 * This command returns CLK_PROGS control parameters as specified
 * by the VBIOS in the Clocks Programming Table.
 *
 * The CLK_PROG objects are indexed per how they are specified in the VBIOS
 * table.
 *
 * See @ref LW2080_CTRL_CLK_CLK_PROGS_CONTROL for documentation on the
 * parameters.
 *
 * Return values are specified per @ref BoardObjGrpGetControl.
 */
#define LW2080_CTRL_CMD_CLK_CLK_PROGS_GET_CONTROL        (0x2080101f) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_CLK_INTERFACE_ID << 8) | 0x1F" */

/*!
 * LW2080_CTRL_CMD_CLK_CLK_PROGS_SET_CONTROL
 *
 * This command accepts client-specified control parameters for a set
 * of CLK_PROGS entries in the Clocks Table, and applies these new
 * parameters to the set of CLK_PROGS entries.
 *
 * The CLK_PROG objects are indexed per how they are specified in the VBIOS
 * table.
 *
 * See @ref LW2080_CTRL_CLK_CLK_PROGS_CONTROL for documentation on the
 * parameters.
 *
 * Return values are specified per @ref BoardObjGrpSetControl.
 */
#define LW2080_CTRL_CMD_CLK_CLK_PROGS_SET_CONTROL        (0x20801020) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_CLK_INTERFACE_ID << 8) | 0x20" */

/*!
 * Macro representing an INVALID/UNSUPPORTED CLK_VF_POINT index.
 */
#define LW2080_CTRL_CLK_CLK_VF_POINT_IDX_ILWALID         LW2080_CTRL_BOARDOBJ_IDX_ILWALID

/*!
 * Macro representing reserved value of CLK_VF_POINT cache counter for tools.
 * Change sequencer set control will pass this value to skip the validation.
 */
#define LW2080_CTRL_CLK_CLK_VF_POINT_CACHE_COUNTER_TOOLS LW_U32_MAX

/*!
 * Enumeration of CLK_PROG class types.
 */
#define LW2080_CTRL_CLK_CLK_VF_POINT_TYPE_BASE           0U
#define LW2080_CTRL_CLK_CLK_VF_POINT_TYPE_30             1U
#define LW2080_CTRL_CLK_CLK_VF_POINT_TYPE_30_FREQ        2U
#define LW2080_CTRL_CLK_CLK_VF_POINT_TYPE_30_VOLT        3U
#define LW2080_CTRL_CLK_CLK_VF_POINT_TYPE_35             4U
#define LW2080_CTRL_CLK_CLK_VF_POINT_TYPE_35_FREQ        5U
#define LW2080_CTRL_CLK_CLK_VF_POINT_TYPE_35_VOLT        6U
#define LW2080_CTRL_CLK_CLK_VF_POINT_TYPE_35_VOLT_PRI    7U
#define LW2080_CTRL_CLK_CLK_VF_POINT_TYPE_35_VOLT_SEC    8U
#define LW2080_CTRL_CLK_CLK_VF_POINT_TYPE_40             9U
#define LW2080_CTRL_CLK_CLK_VF_POINT_TYPE_40_FREQ        10U
#define LW2080_CTRL_CLK_CLK_VF_POINT_TYPE_40_VOLT        11U
#define LW2080_CTRL_CLK_CLK_VF_POINT_TYPE_40_VOLT_PRI    12U
#define LW2080_CTRL_CLK_CLK_VF_POINT_TYPE_40_VOLT_SEC    13U
#define LW2080_CTRL_CLK_CLK_VF_POINT_TYPE_MAX            14U
#define LW2080_CTRL_CLK_CLK_VF_POINT_TYPE_UNKNOWN        LW_U8_MAX

/*!
 * Structure describing the static configuration/POR state of the _30_FREQ class.
 */
typedef struct LW2080_CTRL_CLK_CLK_VF_POINT_INFO_30_FREQ {
    /*!
     * Frequency (MHz) for this VF point. This value will be determined per the
     * semantics of the child class.
     */
    LwU16 freqMHz;
} LW2080_CTRL_CLK_CLK_VF_POINT_INFO_30_FREQ;
typedef struct LW2080_CTRL_CLK_CLK_VF_POINT_INFO_30_FREQ *PLW2080_CTRL_CLK_CLK_VF_POINT_INFO_30_FREQ;

/*!
 * Structure describing the static configuration/POR state of the _30_VOLT class.
 */
typedef struct LW2080_CTRL_CLK_CLK_VF_POINT_INFO_30_VOLT {
    /*!
     * Source voltage (uV) which was used to specify this CLK_VF_POINT_VOLT.
     * These are the voltage values supported by the ADC/NAFLL.  This value will
     * be rounded to the regulator size supported by the VOLTAGE_RAIL and stored
     * as @ref CLK_VF_POINT::voltageuV.  However, this source voltage value
     * should be used when looking up data corresponding to the original
     * ADC/NAFLL values.
     */
    LwU32 sourceVoltageuV;
} LW2080_CTRL_CLK_CLK_VF_POINT_INFO_30_VOLT;
typedef struct LW2080_CTRL_CLK_CLK_VF_POINT_INFO_30_VOLT *PLW2080_CTRL_CLK_CLK_VF_POINT_INFO_30_VOLT;

/*!
 * Structure describing the static configuration/POR state of the _35_FREQ class.
 */
typedef struct LW2080_CTRL_CLK_CLK_VF_POINT_INFO_35_FREQ {
    /*!
     * Frequency (MHz) for this VF point. This value will be determined per the
     * semantics of the child class.
     */
    LwU16 freqMHz;
} LW2080_CTRL_CLK_CLK_VF_POINT_INFO_35_FREQ;
typedef struct LW2080_CTRL_CLK_CLK_VF_POINT_INFO_35_FREQ *PLW2080_CTRL_CLK_CLK_VF_POINT_INFO_35_FREQ;

/*!
 * Structure describing the static configuration/POR state of the _35_VOLT class.
 */
typedef struct LW2080_CTRL_CLK_CLK_VF_POINT_INFO_35_VOLT {
    /*!
     * Source voltage (uV) which was used to specify this CLK_VF_POINT_VOLT.
     * These are the voltage values supported by the ADC/NAFLL.  This value will
     * be rounded to the regulator size supported by the VOLTAGE_RAIL and stored
     * as @ref CLK_VF_POINT::voltageuV.  However, this source voltage value
     * should be used when looking up data corresponding to the original
     * ADC/NAFLL values.
     */
    LwU32 sourceVoltageuV;
} LW2080_CTRL_CLK_CLK_VF_POINT_INFO_35_VOLT;
typedef struct LW2080_CTRL_CLK_CLK_VF_POINT_INFO_35_VOLT *PLW2080_CTRL_CLK_CLK_VF_POINT_INFO_35_VOLT;

/*!
 * Structure describing the static configuration/POR state of the _40_FREQ class.
 */
typedef struct LW2080_CTRL_CLK_CLK_VF_POINT_INFO_40_FREQ {
    /*!
     * Frequency (MHz) for this VF point. This value will be determined per the
     * semantics of the child class.
     */
    LwU16 freqMHz;
} LW2080_CTRL_CLK_CLK_VF_POINT_INFO_40_FREQ;
typedef struct LW2080_CTRL_CLK_CLK_VF_POINT_INFO_40_FREQ *PLW2080_CTRL_CLK_CLK_VF_POINT_INFO_40_FREQ;

/*!
 * Structure describing the static configuration/POR state of the _40_VOLT class.
 */
typedef struct LW2080_CTRL_CLK_CLK_VF_POINT_INFO_40_VOLT {
    /*!
     * Source voltage (uV) which was used to specify this CLK_VF_POINT_VOLT.
     * These are the voltage values supported by the ADC/NAFLL.  This value will
     * be rounded to the regulator size supported by the VOLTAGE_RAIL and stored
     * as @ref CLK_VF_POINT::voltageuV.  However, this source voltage value
     * should be used when looking up data corresponding to the original
     * ADC/NAFLL values.
     */
    LwU32 sourceVoltageuV;
} LW2080_CTRL_CLK_CLK_VF_POINT_INFO_40_VOLT;
typedef struct LW2080_CTRL_CLK_CLK_VF_POINT_INFO_40_VOLT *PLW2080_CTRL_CLK_CLK_VF_POINT_INFO_40_VOLT;

/*!
 * CLK_VF_POINT type-specific data union.  Discriminated by
 * CLK_VF_POINT::super.type.
 */


/*!
 * Structure describing CLK_PROG static information/POR.  Implements the
 * BOARDOBJ model/interface.
 */
typedef struct LW2080_CTRL_CLK_CLK_VF_POINT_INFO {
    /*!
     * LW2080_CTRL_BOARDOBJ super class.  Must always be first object in
     * structure.
     */
    LW2080_CTRL_BOARDOBJ super;
    /*!
     * XAPI does not support using an element within another structure as a
     * discriminant.  Placing this redundant type value here until that design
     * constraint can be fixed.
     */
    LwU8                 type;

    /*!
     * Index of the VFE Equation corresponding to this CLK_VF_POINT.
     */
    LwU8                 vfeEquIdx;
    /*!
     * Index of the VOLTAGE_RAIL for this CLK_VF_POINT object.  Will be used to
     * quantize @ref voltageuV to a voltage supported on this VOLTAGE_RAIL.
     *
     * @note Deprecated.  Will be removed after references are removed
     * from LWAPI.
     */
    LwU8                 voltRailIdx;

    /*!
     * Type-specific data union.
     */
    union {
        LW2080_CTRL_CLK_CLK_VF_POINT_INFO_30_FREQ v30Freq;
        LW2080_CTRL_CLK_CLK_VF_POINT_INFO_30_VOLT v30Volt;
        LW2080_CTRL_CLK_CLK_VF_POINT_INFO_35_FREQ v35Freq;
        LW2080_CTRL_CLK_CLK_VF_POINT_INFO_35_VOLT v35Volt;
        LW2080_CTRL_CLK_CLK_VF_POINT_INFO_40_FREQ v40Freq;
        LW2080_CTRL_CLK_CLK_VF_POINT_INFO_40_VOLT v40Volt;
    } data;
} LW2080_CTRL_CLK_CLK_VF_POINT_INFO;
typedef struct LW2080_CTRL_CLK_CLK_VF_POINT_INFO *PLW2080_CTRL_CLK_CLK_VF_POINT_INFO;

/*!
 * Structure describing secondary CLK_VF_POINTS static information/POR.
 * Implements the BOARDOBJGRP model/interface.
 */
typedef struct LW2080_CTRL_CLK_CLK_VF_POINTS_SEC_INFO {
    /*!
     * LW2080_CTRL_BOARDOBJGRP_E512 super class.  Must always be first object in
     * structure.
     */
    LW2080_CTRL_BOARDOBJGRP_E512      super;

    /*!
     * Array of CLK_VF_POINT structures.  Has valid indexes corresponding to the
     * bits set in @ref super.objMask.
     */
    LW2080_CTRL_CLK_CLK_VF_POINT_INFO vfPoints[LW2080_CTRL_BOARDOBJGRP_E512_MAX_OBJECTS];
} LW2080_CTRL_CLK_CLK_VF_POINTS_SEC_INFO;
typedef struct LW2080_CTRL_CLK_CLK_VF_POINTS_SEC_INFO *PLW2080_CTRL_CLK_CLK_VF_POINTS_SEC_INFO;

/*!
 * Structure describing CLK_VF_POINTS static information/POR.  Implements the
 * BOARDOBJGRP model/interface.
 */
#define LW2080_CTRL_CLK_CLK_VF_POINTS_INFO_MESSAGE_ID (0x21U)

typedef struct LW2080_CTRL_CLK_CLK_VF_POINTS_INFO {
    /*!
     * LW2080_CTRL_BOARDOBJGRP_E512 super class.  Must always be first object in
     * structure.
     */
    LW2080_CTRL_BOARDOBJGRP_E512           super;

    /*!
     * Array of CLK_VF_POINT structures.  Has valid indexes corresponding to the
     * bits set in @ref super.objMask.
     */
    LW2080_CTRL_CLK_CLK_VF_POINT_INFO      vfPoints[LW2080_CTRL_BOARDOBJGRP_E512_MAX_OBJECTS];

    /*!
     * Secondary VF lwrves data.
     * @ref LW2080_CTRL_CLK_CLK_VF_POINTS_SEC_INFO
     */
    LW2080_CTRL_CLK_CLK_VF_POINTS_SEC_INFO sec;
} LW2080_CTRL_CLK_CLK_VF_POINTS_INFO;
typedef struct LW2080_CTRL_CLK_CLK_VF_POINTS_INFO *PLW2080_CTRL_CLK_CLK_VF_POINTS_INFO;

/*!
 * LW2080_CTRL_CMD_CLK_CLK_VF_POINTS_GET_INFO
 *
 * This command returns CLK_VF_POINTS static object information/POR as populated
 * by the RM from the VF lwrves specified by the VBIOS in the Clocks Table and
 * Clocks Programming Table.
 *
 * The CLK_VF_POINTS objects are indexed in the order by which the RM allocates
 * them.
 *
 * See @ref LW2080_CTRL_CLK_CLK_VF_POINTS_INFO for documentation on the
 * parameters.
 *
 * Return values are specified per @ref BoardObjGrpGetInfo.
 */
#define LW2080_CTRL_CMD_CLK_CLK_VF_POINTS_GET_INFO (0x20801021) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_CLK_INTERFACE_ID << 8) | LW2080_CTRL_CLK_CLK_VF_POINTS_INFO_MESSAGE_ID" */


/*!
 * Structure describing CLK_VF_POINT_30 dynamic state information.
 */
typedef struct LW2080_CTRL_CLK_CLK_VF_POINT_STATUS_30 {
    /*!
     * Voltage and frequency pair for this VF point.  These values will be
     * determined per the semantics of the child class.
     */
    LW2080_CTRL_CLK_VF_PAIR pair;
} LW2080_CTRL_CLK_CLK_VF_POINT_STATUS_30;
typedef struct LW2080_CTRL_CLK_CLK_VF_POINT_STATUS_30 *PLW2080_CTRL_CLK_CLK_VF_POINT_STATUS_30;

/*!
 * Structure describing CLK_VF_POINT_35 dynamic state information.
 * PP-TODO : Need to remove once the LWAPIs are updated.
 */
typedef struct LW2080_CTRL_CLK_CLK_VF_POINT_STATUS_35 {
    /*!
     * @ref LW2080_CTRL_CLK_CLK_VF_POINT_BASE_VF_TUPLE
     * Base VF Tuple represent the values that are input / output of VFE.
     */
    LW2080_CTRL_CLK_CLK_VF_POINT_BASE_VF_TUPLE baseVFTuple;

    /*!
     * @ref LW2080_CTRL_CLK_CLK_VF_POINT_VF_TUPLE
     * Offsetted VF Tuple represent the VF tuple adjusted with the
     * applied offsets.
     */
    LW2080_CTRL_CLK_CLK_VF_POINT_VF_TUPLE      offsetedVFTuple[LW2080_CTRL_CLK_CLK_VF_POINT_FREQ_TUPLE_MAX_SIZE];
} LW2080_CTRL_CLK_CLK_VF_POINT_STATUS_35;
typedef struct LW2080_CTRL_CLK_CLK_VF_POINT_STATUS_35 *PLW2080_CTRL_CLK_CLK_VF_POINT_STATUS_35;

/*!
 * Structure describing CLK_VF_POINT_35_FREQ dynamic state information.
 */
typedef struct LW2080_CTRL_CLK_CLK_VF_POINT_STATUS_35_FREQ {
    /*!
     * @ref LW2080_CTRL_CLK_CLK_VF_POINT_BASE_VF_TUPLE
     * Base VF Tuple represent the values that are input / output of VFE.
     */
    LW2080_CTRL_CLK_CLK_VF_POINT_BASE_VF_TUPLE baseVFTuple;

    /*!
     * @ref LW2080_CTRL_CLK_CLK_VF_POINT_VF_TUPLE
     * Offsetted VF Tuple represent the VF tuple adjusted with the
     * applied offsets.
     */
    LW2080_CTRL_CLK_CLK_VF_POINT_VF_TUPLE      offsetedVFTuple[LW2080_CTRL_CLK_CLK_VF_POINT_FREQ_TUPLE_MAX_SIZE];
} LW2080_CTRL_CLK_CLK_VF_POINT_STATUS_35_FREQ;
typedef struct LW2080_CTRL_CLK_CLK_VF_POINT_STATUS_35_FREQ *PLW2080_CTRL_CLK_CLK_VF_POINT_STATUS_35_FREQ;
/*!
 * Structure describing CLK_VF_POINT_35_VOLT_PRI dynamic state information.
 */
typedef struct LW2080_CTRL_CLK_CLK_VF_POINT_STATUS_35_VOLT_PRI {
    /*!
     * @ref LW2080_CTRL_CLK_CLK_VF_POINT_BASE_VF_TUPLE
     * Base VF Tuple represent the values that are input / output of VFE.
     */
    LW2080_CTRL_CLK_CLK_VF_POINT_BASE_VF_TUPLE baseVFTuple;

    /*!
     * @ref LW2080_CTRL_CLK_CLK_VF_POINT_VF_TUPLE
     * Offsetted VF Tuple represent the VF tuple adjusted with the
     * applied offsets.
     */
    LW2080_CTRL_CLK_CLK_VF_POINT_VF_TUPLE      offsetedVFTuple[LW2080_CTRL_CLK_CLK_VF_POINT_FREQ_TUPLE_MAX_SIZE];
} LW2080_CTRL_CLK_CLK_VF_POINT_STATUS_35_VOLT_PRI;
typedef struct LW2080_CTRL_CLK_CLK_VF_POINT_STATUS_35_VOLT_PRI *PLW2080_CTRL_CLK_CLK_VF_POINT_STATUS_35_VOLT_PRI;

/*!
 * Structure describing CLK_VF_POINT_35_VOLT_SEC dynamic state information.
 */
typedef struct LW2080_CTRL_CLK_CLK_VF_POINT_STATUS_35_VOLT_SEC {
    /*!
     * @ref LW2080_CTRL_CLK_CLK_VF_POINT_BASE_VF_TUPLE_SEC
     * Secondary base VF Tuple represent the values that are input / output of VFE.
     */
    LW2080_CTRL_CLK_CLK_VF_POINT_BASE_VF_TUPLE_SEC baseVFTuple;

    /*!
     * @ref LW2080_CTRL_CLK_CLK_VF_POINT_VF_TUPLE
     * Offsetted VF Tuple represent the VF tuple adjusted with the
     * applied offsets.
     */
    LW2080_CTRL_CLK_CLK_VF_POINT_VF_TUPLE          offsetedVFTuple[LW2080_CTRL_CLK_CLK_VF_POINT_FREQ_TUPLE_MAX_SIZE];
} LW2080_CTRL_CLK_CLK_VF_POINT_STATUS_35_VOLT_SEC;
typedef struct LW2080_CTRL_CLK_CLK_VF_POINT_STATUS_35_VOLT_SEC *PLW2080_CTRL_CLK_CLK_VF_POINT_STATUS_35_VOLT_SEC;


/*!
 * Structure describing CLK_VF_POINT_40_FREQ dynamic state information.
 */
typedef struct LW2080_CTRL_CLK_CLK_VF_POINT_STATUS_40_FREQ {
    /*!
     * @ref LW2080_CTRL_CLK_CLK_VF_POINT_BASE_VF_TUPLE
     * Base VF Tuple represent the values that are input / output of VFE.
     */
    LW2080_CTRL_CLK_CLK_VF_POINT_BASE_VF_TUPLE baseVFTuple;

    /*!
     * @ref LW2080_CTRL_CLK_CLK_VF_POINT_VF_TUPLE
     * Offsetted VF Tuple represent the VF tuple adjusted with the
     * applied offsets.
     */
    LW2080_CTRL_CLK_CLK_VF_POINT_VF_TUPLE      offsetedVFTuple[LW2080_CTRL_CLK_CLK_VF_POINT_FREQ_TUPLE_MAX_SIZE];

    /*!
     * @ref LW2080_CTRL_CLK_OFFSET_VF_TUPLE
     * Offset VF Tuple represent the VF tuple of
     * applied offsets.
     */
    LW2080_CTRL_CLK_OFFSET_VF_TUPLE            offsetVFTuple[LW2080_CTRL_CLK_CLK_VF_POINT_FREQ_TUPLE_MAX_SIZE];
} LW2080_CTRL_CLK_CLK_VF_POINT_STATUS_40_FREQ;
typedef struct LW2080_CTRL_CLK_CLK_VF_POINT_STATUS_40_FREQ *PLW2080_CTRL_CLK_CLK_VF_POINT_STATUS_40_FREQ;
/*!
 * Structure describing CLK_VF_POINT_40_VOLT_PRI dynamic state information.
 */
typedef struct LW2080_CTRL_CLK_CLK_VF_POINT_STATUS_40_VOLT_PRI {
    /*!
     * @ref LW2080_CTRL_CLK_CLK_VF_POINT_BASE_VF_TUPLE
     * Base VF Tuple represent the values that are input / output of VFE.
     */
    LW2080_CTRL_CLK_CLK_VF_POINT_BASE_VF_TUPLE baseVFTuple;

    /*!
     * @ref LW2080_CTRL_CLK_CLK_VF_POINT_VF_TUPLE
     * Offsetted VF Tuple represent the VF tuple adjusted with the
     * applied offsets.
     */
    LW2080_CTRL_CLK_CLK_VF_POINT_VF_TUPLE      offsetedVFTuple[LW2080_CTRL_CLK_CLK_VF_POINT_FREQ_TUPLE_MAX_SIZE];

    /*!
     * @ref LW2080_CTRL_CLK_OFFSET_VF_TUPLE
     * Offset VF Tuple represent the VF tuple of
     * applied offsets.
     */
    LW2080_CTRL_CLK_OFFSET_VF_TUPLE            offsetVFTuple[LW2080_CTRL_CLK_CLK_VF_POINT_FREQ_TUPLE_MAX_SIZE];
} LW2080_CTRL_CLK_CLK_VF_POINT_STATUS_40_VOLT_PRI;
typedef struct LW2080_CTRL_CLK_CLK_VF_POINT_STATUS_40_VOLT_PRI *PLW2080_CTRL_CLK_CLK_VF_POINT_STATUS_40_VOLT_PRI;

/*!
 * Structure describing CLK_VF_POINT_40_VOLT_SEC dynamic state information.
 */
typedef struct LW2080_CTRL_CLK_CLK_VF_POINT_STATUS_40_VOLT_SEC {
    /*!
     * @ref LW2080_CTRL_CLK_CLK_VF_POINT_BASE_VF_TUPLE_SEC
     * Secondary base VF Tuple represent the values that are input / output of VFE.
     */
    LW2080_CTRL_CLK_CLK_VF_POINT_BASE_VF_TUPLE_SEC baseVFTuple;

    /*!
     * @ref LW2080_CTRL_CLK_CLK_VF_POINT_VF_TUPLE
     * Offsetted VF Tuple represent the VF tuple adjusted with the
     * applied offsets.
     */
    LW2080_CTRL_CLK_CLK_VF_POINT_VF_TUPLE          offsetedVFTuple[LW2080_CTRL_CLK_CLK_VF_POINT_FREQ_TUPLE_MAX_SIZE];

    /*!
     * @ref LW2080_CTRL_CLK_OFFSET_VF_TUPLE
     * Offset VF Tuple represent the VF tuple of
     * applied offsets.
     */
    LW2080_CTRL_CLK_OFFSET_VF_TUPLE                offsetVFTuple[LW2080_CTRL_CLK_CLK_VF_POINT_FREQ_TUPLE_MAX_SIZE];
} LW2080_CTRL_CLK_CLK_VF_POINT_STATUS_40_VOLT_SEC;
typedef struct LW2080_CTRL_CLK_CLK_VF_POINT_STATUS_40_VOLT_SEC *PLW2080_CTRL_CLK_CLK_VF_POINT_STATUS_40_VOLT_SEC;

/*!
 * CLK_VF_POINT type-specific data union.  Discriminated by
 * CLK_VF_POINT::super.type.
 */


/*!
 * Structure describing CLK_VF_POINT dynamic state information. Implements the
 * BOARDOBJ model/interface.
 */
typedef struct LW2080_CTRL_CLK_CLK_VF_POINT_STATUS {
    /*!
     * LW2080_CTRL_BOARDOBJ super class.  Must always be first object in
     * structure.
     */
    LW2080_CTRL_BOARDOBJ super;
    /*!
     * XAPI does not support using an element within another structure as a
     * discriminant.  Placing this redundant type value here until that design
     * constraint can be fixed.
     */
    LwU8                 type;

    /*!
     * PP-TODO : Remove @ref voltageuV and @ref freqMHz once LWAPIs are updated
     */

    /*!
     * Voltage (uV) for this VF point.  This value will be determined per the
     * semantics of the child class.
     */
    LwU32                voltageuV;
    /*!
     * Frequency (MHz) for this VF point. This value will be determined per the
     * semantics of the child class.
     */
    LwU16                freqMHz;

    /*!
     * Type-specific data union.
     */
    union {
        LW2080_CTRL_CLK_CLK_VF_POINT_STATUS_30          v30;
                     // PP-TODO : Need to remove once LWAPIs are updated.
        LW2080_CTRL_CLK_CLK_VF_POINT_STATUS_35          v35;
        LW2080_CTRL_CLK_CLK_VF_POINT_STATUS_35_FREQ     v35Freq;
        LW2080_CTRL_CLK_CLK_VF_POINT_STATUS_35_VOLT_PRI v35VoltPri;
        LW2080_CTRL_CLK_CLK_VF_POINT_STATUS_35_VOLT_SEC v35VoltSec;
        LW2080_CTRL_CLK_CLK_VF_POINT_STATUS_40_FREQ     v40Freq;
        LW2080_CTRL_CLK_CLK_VF_POINT_STATUS_40_VOLT_PRI v40VoltPri;
        LW2080_CTRL_CLK_CLK_VF_POINT_STATUS_40_VOLT_SEC v40VoltSec;
    } data;
} LW2080_CTRL_CLK_CLK_VF_POINT_STATUS;
typedef struct LW2080_CTRL_CLK_CLK_VF_POINT_STATUS *PLW2080_CTRL_CLK_CLK_VF_POINT_STATUS;

/*!
 * Structure describing secondary CLK_VF_POINTS dynamic state information.
 * Implements the BOARDOBJGRP model/interface.
 */
typedef struct LW2080_CTRL_CLK_CLK_VF_POINTS_SEC_STATUS {
    /*!
     * LW2080_CTRL_BOARDOBJGRP_E512 super class for secondary VF lwrves.
     */
    LW2080_CTRL_BOARDOBJGRP_E512        super;

    /*!
     * Array of CLK_VF_POINT structures for secondary VF lwrves. Has valid indexes
     * corresponding to the bits set in @ref super.objMask.
     */
    LW2080_CTRL_CLK_CLK_VF_POINT_STATUS vfPoints[LW2080_CTRL_BOARDOBJGRP_E512_MAX_OBJECTS];
} LW2080_CTRL_CLK_CLK_VF_POINTS_SEC_STATUS;
typedef struct LW2080_CTRL_CLK_CLK_VF_POINTS_SEC_STATUS *PLW2080_CTRL_CLK_CLK_VF_POINTS_SEC_STATUS;

/*!
 * Structure describing CLK_VF_POINTS dynamic state information. Implements the
 * BOARDOBJGRP model/interface.
 */
#define LW2080_CTRL_CLK_CLK_VF_POINTS_STATUS_MESSAGE_ID (0x22U)

typedef struct LW2080_CTRL_CLK_CLK_VF_POINTS_STATUS {
    /*!
     * LW2080_CTRL_BOARDOBJGRP_E512 super class.  Must always be first object in
     * structure.
     */
    LW2080_CTRL_BOARDOBJGRP_E512             super;

    /*!
     * Array of CLK_VF_POINT structures.  Has valid indexes corresponding to the
     * bits set in @ref super.objMask.
     */
    LW2080_CTRL_CLK_CLK_VF_POINT_STATUS      vfPoints[LW2080_CTRL_BOARDOBJGRP_E512_MAX_OBJECTS];

    /*!
     * Secondary VF lwrves data.
     * @ref LW2080_CTRL_CLK_CLK_VF_POINTS_SEC_STATUS
     */
    LW2080_CTRL_CLK_CLK_VF_POINTS_SEC_STATUS sec;
} LW2080_CTRL_CLK_CLK_VF_POINTS_STATUS;
typedef struct LW2080_CTRL_CLK_CLK_VF_POINTS_STATUS *PLW2080_CTRL_CLK_CLK_VF_POINTS_STATUS;

/*!
 * LW2080_CTRL_CMD_CLK_CLK_VF_POINTS_GET_STATUS
 *
 * This command returns the VF Points dynamic state information associated by the
 * CLK_VF_POINTS functionality
 *
 * See @ref LW2080_CTRL_CLK_CLK_VF_POINTS_STATUS for documentation on the
 * parameters.
 *
 * Return values are specified per @ref BoardObjGrpGetStatus.
 */
#define LW2080_CTRL_CMD_CLK_CLK_VF_POINTS_GET_STATUS (0x20801022) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_CLK_INTERFACE_ID << 8) | LW2080_CTRL_CLK_CLK_VF_POINTS_STATUS_MESSAGE_ID" */

/*!
 * Structure describing CLK_VF_POINT_30_FREQ specific control parameters.
 */
typedef struct LW2080_CTRL_CLK_CLK_VF_POINT_CONTROL_30_FREQ {
    /*!
     * This will give the deviation of given voltage from it's nominal value.
     */
    LwS32 voltDeltauV;
} LW2080_CTRL_CLK_CLK_VF_POINT_CONTROL_30_FREQ;
typedef struct LW2080_CTRL_CLK_CLK_VF_POINT_CONTROL_30_FREQ *PLW2080_CTRL_CLK_CLK_VF_POINT_CONTROL_30_FREQ;

/*!
 * Structure describing CLK_VF_POINT_30_VOLT specific control parameters.
 */
typedef struct LW2080_CTRL_CLK_CLK_VF_POINT_CONTROL_30_VOLT {
    /*!
     * This will give the deviation of given freq from it's nominal value.
     */
    LW2080_CTRL_CLK_FREQ_DELTA freqDelta;
} LW2080_CTRL_CLK_CLK_VF_POINT_CONTROL_30_VOLT;
typedef struct LW2080_CTRL_CLK_CLK_VF_POINT_CONTROL_30_VOLT *PLW2080_CTRL_CLK_CLK_VF_POINT_CONTROL_30_VOLT;

/*!
 * Structure describing CLK_VF_POINT_35_FREQ specific control parameters.
 */
typedef struct LW2080_CTRL_CLK_CLK_VF_POINT_CONTROL_35_FREQ {
    /*!
     * This will give the deviation of given voltage from it's nominal value.
     */
    LwS32 voltDeltauV;
} LW2080_CTRL_CLK_CLK_VF_POINT_CONTROL_35_FREQ;
typedef struct LW2080_CTRL_CLK_CLK_VF_POINT_CONTROL_35_FREQ *PLW2080_CTRL_CLK_CLK_VF_POINT_CONTROL_35_FREQ;

/*!
 * Structure describing CLK_VF_POINT_35_VOLT specific control parameters.
 */
typedef struct LW2080_CTRL_CLK_CLK_VF_POINT_CONTROL_35_VOLT {
    /*!
     * This will give the deviation of given freq from it's nominal value.
     */
    LW2080_CTRL_CLK_FREQ_DELTA freqDelta;
} LW2080_CTRL_CLK_CLK_VF_POINT_CONTROL_35_VOLT;
typedef struct LW2080_CTRL_CLK_CLK_VF_POINT_CONTROL_35_VOLT *PLW2080_CTRL_CLK_CLK_VF_POINT_CONTROL_35_VOLT;

/*!
 * Structure describing CLK_VF_POINT_35_VOLT specific control parameters.
 */
typedef struct LW2080_CTRL_CLK_CLK_VF_POINT_CONTROL_35_VOLT_SEC {
    /*!
     * SUPER class. Must always be first.
     */
    LW2080_CTRL_CLK_CLK_VF_POINT_CONTROL_35_VOLT super;

    /*!
     * DVCO offset override in terms of DVCO codes to trigger the fast
     * slowdown while HW switches from reference NDIV point to secondary NDIV.
     */
    LwU8                                         dvcoOffsetCodeOverride;
} LW2080_CTRL_CLK_CLK_VF_POINT_CONTROL_35_VOLT_SEC;
typedef struct LW2080_CTRL_CLK_CLK_VF_POINT_CONTROL_35_VOLT_SEC *PLW2080_CTRL_CLK_CLK_VF_POINT_CONTROL_35_VOLT_SEC;

/*!
 * Structure describing CLK_VF_POINT_40_FREQ specific control parameters.
 */
typedef struct LW2080_CTRL_CLK_CLK_VF_POINT_CONTROL_40_FREQ {
    /*!
     * This will give the deviation of given voltage from it's nominal value.
     */
    LwS32 voltDeltauV;
} LW2080_CTRL_CLK_CLK_VF_POINT_CONTROL_40_FREQ;
typedef struct LW2080_CTRL_CLK_CLK_VF_POINT_CONTROL_40_FREQ *PLW2080_CTRL_CLK_CLK_VF_POINT_CONTROL_40_FREQ;

/*!
 * Structure describing CLK_VF_POINT_40_VOLT specific control parameters.
 */
typedef struct LW2080_CTRL_CLK_CLK_VF_POINT_CONTROL_40_VOLT {
    /*!
     * This will give the deviation of given freq from it's nominal value.
     */
    LW2080_CTRL_CLK_FREQ_DELTA freqDelta;
} LW2080_CTRL_CLK_CLK_VF_POINT_CONTROL_40_VOLT;
typedef struct LW2080_CTRL_CLK_CLK_VF_POINT_CONTROL_40_VOLT *PLW2080_CTRL_CLK_CLK_VF_POINT_CONTROL_40_VOLT;

/*!
 * Structure describing CLK_VF_POINT_40_VOLT specific control parameters.
 */
typedef struct LW2080_CTRL_CLK_CLK_VF_POINT_CONTROL_40_VOLT_SEC {
    /*!
     * SUPER class. Must always be first.
     */
    LW2080_CTRL_CLK_CLK_VF_POINT_CONTROL_35_VOLT super;

    /*!
     * DVCO offset override in terms of DVCO codes to trigger the fast
     * slowdown while HW switches from reference NDIV point to secondary NDIV.
     */
    LwU8                                         dvcoOffsetCodeOverride;
} LW2080_CTRL_CLK_CLK_VF_POINT_CONTROL_40_VOLT_SEC;
typedef struct LW2080_CTRL_CLK_CLK_VF_POINT_CONTROL_40_VOLT_SEC *PLW2080_CTRL_CLK_CLK_VF_POINT_CONTROL_40_VOLT_SEC;

/*!
 * CLK_VF_POINT type-specific data union.  Discriminated by
 * CLK_VF_POINT::super.type.
 */


/*!
 * Structure representing the control parameters associated with a CLK_VF_POINT.
 */
typedef struct LW2080_CTRL_CLK_CLK_VF_POINT_CONTROL {
    /*!
     * LW2080_CTRL_BOARDOBJ super class.  Must always be first object in
     * structure.
     */
    LW2080_CTRL_BOARDOBJ super;
    /*!
     * XAPI does not support using an element within another structure as a
     * discriminant.  Placing this redundant type value here until that design
     * constraint can be fixed.
     */
    LwU8                 type;

    /*!
     * Type-specific data union.
     */
    union {
        LW2080_CTRL_CLK_CLK_VF_POINT_CONTROL_30_FREQ     v30Freq;
        LW2080_CTRL_CLK_CLK_VF_POINT_CONTROL_30_VOLT     v30Volt;
        LW2080_CTRL_CLK_CLK_VF_POINT_CONTROL_35_FREQ     v35Freq;
        LW2080_CTRL_CLK_CLK_VF_POINT_CONTROL_35_VOLT     v35Volt;
        LW2080_CTRL_CLK_CLK_VF_POINT_CONTROL_35_VOLT_SEC v35VoltSec;
        LW2080_CTRL_CLK_CLK_VF_POINT_CONTROL_40_FREQ     v40Freq;
        LW2080_CTRL_CLK_CLK_VF_POINT_CONTROL_40_VOLT     v40Volt;
        LW2080_CTRL_CLK_CLK_VF_POINT_CONTROL_40_VOLT_SEC v40VoltSec;
    } data;
} LW2080_CTRL_CLK_CLK_VF_POINT_CONTROL;
typedef struct LW2080_CTRL_CLK_CLK_VF_POINT_CONTROL *PLW2080_CTRL_CLK_CLK_VF_POINT_CONTROL;

/*!
 * Structure representing the control parameters associated with the
 * secondary CLK_VF_POINTS.
 */
typedef struct LW2080_CTRL_CLK_CLK_VF_POINTS_SEC_CONTROL {
    /*!
     * LW2080_CTRL_BOARDOBJGRP_E512 super class.  Must always be first object in
     * structure.
     */
    LW2080_CTRL_BOARDOBJGRP_E512         super;

    /*!
     * Array of CLK_VF_POINT structures.  Has valid indexes corresponding to the
     * bits set in @ref super.objMask.
     */
    LW2080_CTRL_CLK_CLK_VF_POINT_CONTROL vfPoints[LW2080_CTRL_BOARDOBJGRP_E512_MAX_OBJECTS];
} LW2080_CTRL_CLK_CLK_VF_POINTS_SEC_CONTROL;
typedef struct LW2080_CTRL_CLK_CLK_VF_POINTS_SEC_CONTROL *PLW2080_CTRL_CLK_CLK_VF_POINTS_SEC_CONTROL;

/*!
 * Structure representing the control parameters associated with a CLK_VF_POINTS.
 */
typedef struct LW2080_CTRL_CLK_CLK_VF_POINTS_CONTROL {
    /*!
     * LW2080_CTRL_BOARDOBJGRP_E512 super class.  Must always be first object in
     * structure.
     */
    LW2080_CTRL_BOARDOBJGRP_E512              super;

    /*!
     * Array of CLK_VF_POINT structures.  Has valid indexes corresponding to the
     * bits set in @ref super.objMask.
     */
    LW2080_CTRL_CLK_CLK_VF_POINT_CONTROL      vfPoints[LW2080_CTRL_BOARDOBJGRP_E512_MAX_OBJECTS];

    /*!
     * Secondary VF lwrves data.
     * @ref LW2080_CTRL_CLK_CLK_VF_POINTS_SEC_CONTROL
     */
    LW2080_CTRL_CLK_CLK_VF_POINTS_SEC_CONTROL sec;
} LW2080_CTRL_CLK_CLK_VF_POINTS_CONTROL;
typedef struct LW2080_CTRL_CLK_CLK_VF_POINTS_CONTROL *PLW2080_CTRL_CLK_CLK_VF_POINTS_CONTROL;

/*!
 * LW2080_CTRL_CMD_CLK_CLK_VF_POINTS_GET_CONTROL
 *
 * This command returns CLK_VF_POINTS control parameters as populated by the
 * RM from the VF lwrves specified by the VBIOS in the Clocks Table and Clocks
 * Programming Table.
 *
 * The CLK_VF_POINTS objects are indexed in the order by which the RM allocates
 * them.
 *
 * See @ref LW2080_CTRL_CLK_CLK_VF_POINTS_CONTROL for documentation on the
 * parameters.
 *
 * Return values are specified per @ref BoardObjGrpGetInfo.
 */
#define LW2080_CTRL_CMD_CLK_CLK_VF_POINTS_GET_CONTROL (0x20801023) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_CLK_INTERFACE_ID << 8) | 0x23" */

/*!
 * LW2080_CTRL_CMD_CLK_CLK_VF_POINTS_SET_CONTROL
 *
 * This command accepts client-specified control parameters for a set of
 * CLK_VF_POINTS entries in the Clocks Table and Clocks Programming Table and
 * applies these new parameters to the set of CLK_VF_POINTS entries.
 *
 * The CLK_VF_POINTS objects are indexed in the order by which the RM allocates
 * them.
 *
 * See @ref LW2080_CTRL_CLK_CLK_VF_POINTS_CONTROL for documentation on the
 * parameters.
 *
 * Return values are specified per @ref BoardObjGrpGetInfo.
 */
#define LW2080_CTRL_CMD_CLK_CLK_VF_POINTS_SET_CONTROL (0x20801024) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_CLK_INTERFACE_ID << 8) | 0x24" */

/*!
 * Structure containing the target frequency of the corresponding clock domain
 * that will be used in the clock change sequence. Clients may want to pack
 * several items in the list for changing frequency of multiple domains.
 */
typedef struct LW2080_CTRL_CLK_CLK_DOMAIN_LIST_ITEM {
    /*!
     * [In] Clock Domain @ref LW2080_CTRL_CLK_DOMAIN_<xyz>
     */
    LwU32 clkDomain;

    /*!
     * [In] Target Frequency to be set in KHz.
     * This should ideally have been in MHz, but the clocks code and the
     * interfaces in RM takes clock values in KHz. To avoid changing all the
     * interfaces, keep the unit in KHz here.
     */
    LwU32 clkFreqKHz;

    /*!
     * [In/Out] The regime-id @ref LW2080_CTRL_CLK_NAFLL_REGIME_ID_<XYZ>
     * for the clock domain. Note: This field is applicable only for clocks
     * sourced from noise aware clocks (a.k.a NAFLLs).
     * It is input to CLK code while programming and output of CLK code
     * while reading.
     */
    LwU8  regimeId;

    /*!
     * Source enumeration.  @ref LW2080_CTRL_CLK_CLK_PROG_1X_SOURCE_<XYZ>.
     */
    LwU8  source;

    /*!
     * Cache DVCO Min frequency in MHz
     */
    LwU16 dvcoMinFreqMHz;
} LW2080_CTRL_CLK_CLK_DOMAIN_LIST_ITEM;
typedef struct LW2080_CTRL_CLK_CLK_DOMAIN_LIST_ITEM *PLW2080_CTRL_CLK_CLK_DOMAIN_LIST_ITEM;

/*!
 * Structure containing the number and list of clock domains to be set by a
 * client @ref PLW2080_CTRL_CLK_CLK_DOMAIN_LIST_ITEM.
 */
typedef struct LW2080_CTRL_CLK_CLK_DOMAIN_LIST {
    /*!
     * Number of CLK_DOMAINS that require the frequency change.
     */
    LwU8                                 numDomains;

    /*!
     * List of @ref LW2080_CTRL_CLK_CLK_DOMAIN_LIST_ITEM entries.
     */
    LW2080_CTRL_CLK_CLK_DOMAIN_LIST_ITEM clkDomains[LW2080_CTRL_BOARDOBJ_MAX_BOARD_OBJECTS];
} LW2080_CTRL_CLK_CLK_DOMAIN_LIST;
typedef struct LW2080_CTRL_CLK_CLK_DOMAIN_LIST *PLW2080_CTRL_CLK_CLK_DOMAIN_LIST;

/*!
 * Enumeration of the CLK_FREQ_CONTROLLER feature version.
 *
 * _10 - CLFC 1.0 - Legacy implementation of CLK_FREQ_CONTROLLER
 * _20 - CLFC 2.0 - Latest implementation of CLK_FREQ_CONTROLLER
 */
#define LW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_VERSION_ILWALID              0x00U
#define LW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_VERSION_10                   0x10U
#define LW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_VERSION_20                   0x20U

/*!
 * Enumeration of CLK_FREQ_CONTROLLER IDs.
 */
#define LW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_ID_SYS                       0x00U
#define LW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_ID_LTC                       0x01U
#define LW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_ID_XBAR                      0x02U
#define LW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_ID_GPC0                      0x03U
#define LW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_ID_GPC1                      0x04U
#define LW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_ID_GPC2                      0x05U
#define LW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_ID_GPC3                      0x06U
#define LW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_ID_GPC4                      0x07U
#define LW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_ID_GPC5                      0x08U
#define LW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_ID_GPCS                      0x09U
#define LW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_ID_LWD                       0x0AU
#define LW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_ID_HOST                      0x0BU
#define LW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_ID_GPC6                      0x0LW
#define LW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_ID_GPC7                      0x0DU
#define LW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_ID_GPC8                      0x0EU
#define LW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_ID_GPC9                      0x0FU
#define LW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_ID_GPC10                     0x10U
#define LW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_ID_GPC11                     0x11U


/*!
 * Mask of all frequency controller GPC IDs supported by RM
 */
#define    LW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_MASK_UNICAST_GPC                   \
                (LWBIT(LW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_ID_GPC0)  |            \
                 LWBIT(LW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_ID_GPC1)  |            \
                 LWBIT(LW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_ID_GPC2)  |            \
                 LWBIT(LW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_ID_GPC3)  |            \
                 LWBIT(LW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_ID_GPC4)  |            \
                 LWBIT(LW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_ID_GPC5)  |            \
                 LWBIT(LW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_ID_GPC6)  |            \
                 LWBIT(LW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_ID_GPC7)  |            \
                 LWBIT(LW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_ID_GPC8)  |            \
                 LWBIT(LW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_ID_GPC9)  |            \
                 LWBIT(LW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_ID_GPC10) |            \
                 LWBIT(LW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_ID_GPC11))

/*!
 * Macro representing an INVALID/UNSUPPORTED CLK_FREQ_CONTROLLER ID.
 */
#define LW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_ID_ILWALID                   LW_U8_MAX

/*!
 * Enumeration of CLK_FREQ_CONTROLLER class types.
 */
#define LW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_TYPE_DISABLED                0x00U
#define LW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_TYPE_PI                      0x01U // To be deprecated
#define LW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_TYPE_10_PI                   0x02U
#define LW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_TYPE_20_PI                   0x03U
#define LW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_TYPE_10                      0x04U
#define LW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_TYPE_20                      0x05U

/*!
 * Macro representing an INVALID/UNSUPPORTED CLK_FREQ_CONTROLLER index.
 */
#define LW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_IDX_ILWALID                  LW2080_CTRL_BOARDOBJ_IDX_ILWALID_8BIT

/*!
 * Enumeration of CLK_FREQ_CONTROLLER modes.
 */
#define LW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_MODE_MIN                     (0x00U)
#define LW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_MODE_MAX                     (0x01U)
#define LW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_MODE_AVG                     (0x02U)

/*!
 * Enumeration of CLK_FREQ_CONTROLLER partitions frequency modes.
 */
#define LW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_PARTS_FREQ_MODE_BCAST        (0x00U)
#define LW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_PARTS_FREQ_MODE_MIN          (0x01U)
#define LW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_PARTS_FREQ_MODE_MAX          (0x02U)
#define LW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_PARTS_FREQ_MODE_AVG          (0x03U)

/*!
 * Enumeration of clients which can disable/enable frequency controllers.
 */
#define LW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_CLIENT_ID_CPU                (0x00U)
#define LW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_CLIENT_ID_PMU_REGIME_LOGIC   (0x01U)
#define LW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_CLIENT_ID_PMU_DVCO_MIN       (0x02U)
#define LW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_CLIENT_ID_PMU_WORST_DVCO_MIN (0x03U)
#define LW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_CLIENT_ID_VF_SWITCH          (0x04U)

/*!
 * Max number of therm monitors
 */
#define LW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_THERM_MONITOR_MAX            (0x02U)

/*!
 * Structure describing therm monitors
 */
typedef struct LW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_THERM_MONITOR {
    /*!
     * Therm monitor index
     */
    LwU8       thermMonIdx;
    /*!
     * Minimum threshold value above which controller sample is poisoned (0.0 - 1.0)
     */
    LwUFXP4_12 threshold;
} LW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_THERM_MONITOR;
typedef LW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_THERM_MONITOR PLW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_THERM_MONITOR;

/*!
 * Structure describing the static configuration/POR state of the _PI class.
 */
typedef struct LW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_INFO_PI {
    /*!
     * Proportional gain for this frequency controller in uV/MHz
     */
    LwSFXP20_12 propGain;
    /*!
     * Integral gain for this frequency controller in uV/MHz
     */
    LwSFXP20_12 integGain;
    /*!
     * Decay factor for the integral term.
     */
    LwSFXP20_12 integDecay;
    /*!
     * Voltage delta limit range min value.
     */
    LwS32       voltDeltaMin;
    /*!
     * Voltage delta limit range max value.
     */
    LwS32       voltDeltaMax;
    /*!
     * Minimum percentage time of the HW slowdown required in a
     * sampling period to poison the sample.
     */
    LwU8        slowdownPctMin;
    /*!
     * Whether to poison the sample only if the slowdown oclwrred
     * on the clock domain of this frequency controller. FALSE means
     * poison the sample even if slowdown oclwrred on other clock domains.
     */
    LwBool      bPoison;
    /*!
     * Index into the Thermal Monitor Table. To be used for BA to poison
     * the sample per baPctMin value below.
     *
     * Invalid index value means that BA support is not required and will
     * disable the poisoning behavior.
     */
    LwU8        baThermMonIdx;
    /*!
     * Minimum percentage of time droopy should engage to poison the sample.
     */
    LwUFXP4_12  baPctMin;
    /*!
     * Index into the Thermal Monitor Table. To be used for Droopy VR to poison
     * the sample per droopyPctMin value below.
     *
     * Invalid index value means that droopy support is not required and will
     * disable the poisoning behavior.
     */
    LwU8        thermMonIdx;
    /*!
     * Minimum percentage of time droopy should engage to poison the sample.
     */
    LwUFXP4_12  droopyPctMin;
} LW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_INFO_PI;
typedef struct LW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_INFO_PI *PLW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_INFO_PI;

/*!
 * Structure describing CLK_FREQ_CONTROLLER static information/POR.
 * Implements the BOARDOBJ model/interface.
 */
typedef struct LW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_INFO_10 {
    /*!
     * rsvd
     */
    LwU8 rsvd;
} LW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_INFO_10;
typedef struct LW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_INFO_10 *PLW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_INFO_10;

/*!
 * Structure describing the static configuration/POR state of the _PI class.
 */
typedef struct LW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_INFO_10_PI {
    /*!
     * FREQ_CONTROLLER_10 super class
     */
    LW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_INFO_10 super;
    /*!
     * Proportional gain for this frequency controller in uV/MHz
     */
    LwSFXP20_12                                 propGain;
    /*!
     * Integral gain for this frequency controller in uV/MHz
     */
    LwSFXP20_12                                 integGain;
    /*!
     * Decay factor for the integral term.
     */
    LwSFXP20_12                                 integDecay;
    /*!
     * Voltage delta limit range min value.
     */
    LwS32                                       voltDeltaMin;
    /*!
     * Voltage delta limit range max value.
     */
    LwS32                                       voltDeltaMax;
    /*!
     * Minimum percentage time of the HW slowdown required in a
     * sampling period to poison the sample.
     */
    LwU8                                        slowdownPctMin;
    /*!
     * Whether to poison the sample only if the slowdown oclwrred
     * on the clock domain of this frequency controller. FALSE means
     * poison the sample even if slowdown oclwrred on other clock domains.
     */
    LwBool                                      bPoison;
    /*!
     * Index into the Thermal Monitor Table. To be used for BA to poison
     * the sample per baPctMin value below.
     *
     * Invalid index value means that BA support is not required and will
     * disable the poisoning behavior.
     */
    LwU8                                        baThermMonIdx;
    /*!
     * Minimum percentage of time droopy should engage to poison the sample.
     */
    LwUFXP4_12                                  baPctMin;
    /*!
     * Index into the Thermal Monitor Table. To be used for Droopy VR to poison
     * the sample per droopyPctMin value below.
     *
     * Invalid index value means that droopy support is not required and will
     * disable the poisoning behavior.
     */
    LwU8                                        thermMonIdx;
    /*!
     * Minimum percentage of time droopy should engage to poison the sample.
     */
    LwUFXP4_12                                  droopyPctMin;
} LW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_INFO_10_PI;
typedef struct LW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_INFO_10_PI *PLW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_INFO_10_PI;

/*!
 * Structure describing CLK_FREQ_CONTROLLER static information/POR.
 * Implements the BOARDOBJ model/interface.
 */
typedef struct LW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_INFO_20 {
    /*!
     * Mode of operation @ref LW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_MODE_<xyz>.
     */
    LwU8                                              freqMode;
    /*!
     * Clock domain @ref LW2080_CTRL_CLK_DOMAIN_<xyz>.
     */
    LwU8                                              clkDomainIdx;
    /*!
     * Voltage controller index points to valid CLVC controller
     * This will be used to control whether to respect CLFC voltage offset correction.
     * The guideline is to ignore CLFC voltage offset of given sample if CLVC
     * voltage error (Vrequested - Vsensed) of current sample is greater than
     * or equal to global voltage error Threshold defined in controllers table header.
     */
    LwU8                                              voltControllerIdx;
    /*!
     * Absolute CLVC voltage error Threshold
     * Ignore CLFC voltage offset of given sample if CLVC voltage error (Vrequested - Vsensed)
     * of current sample is greater than or equal to voltage error Threshold.
     */
    LwU32                                             voltErrorThresholduV;
    /*!
     * Minimum threshold value above which controller sample is poisoned (0.0 - 1.0)
     */
    LwUFXP4_12                                        hwSlowdownThreshold;
    /*!
     * Voltage offset range min (in uV)
     */
    LwS32                                             voltOffsetMinuV;
    /*!
     * Voltage offset range max (in uV)
     */
    LwS32                                             voltOffsetMaxuV;
    /*!
     * Therm Monitors
     */
    LW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_THERM_MONITOR thermMonitor[LW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_THERM_MONITOR_MAX];
} LW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_INFO_20;
typedef struct LW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_INFO_20 *PLW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_INFO_20;

/*!
 * Structure describing the static configuration/POR state of the _PI class.
 */
typedef struct LW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_INFO_20_PI {
    /*!
     * FREQ_CONTROLLER_20 super class
     */
    LW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_INFO_20 super;
    /*!
     * Proportional gain for this frequency controller in uV/MHz
     */
    LwSFXP20_12                                 propGain;
    /*!
     * Integral gain for this frequency controller in uV/MHz
     */
    LwSFXP20_12                                 integGain;
    /*!
     * Decay factor for the integral term.
     */
    LwSFXP20_12                                 integDecay;
    /*!
     * Absolute value of Positive Frequency Hysteresis in MHz (0 => no hysteresis).
     * (hysteresis to apply when frequency has positive delta)
     */
    LwS32                                       freqHystPosMHz;
    /*!
     * Absolute value of Negative Frequency Hysteresis in MHz (0 => no hysteresis).
     * (hysteresis to apply when frequency has negative delta)
     */
    LwS32                                       freqHystNegMHz;
} LW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_INFO_20_PI;
typedef struct LW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_INFO_20_PI *PLW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_INFO_20_PI;

/*!
 * CLK_FREQ_CONTROLLER type-specific data union. Discriminated by
 * CLK_FREQ_CONTROLLER::super.type.
 */


/*!
 * Structure describing CLK_FREQ_CONTROLLER static information/POR.
 * Implements the BOARDOBJ model/interface.
 */
typedef struct LW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_INFO {
    /*!
     * LW2080_CTRL_BOARDOBJ super class.  Must always be first object in
     * structure.
     */
    LW2080_CTRL_BOARDOBJ super;
    /*!
     * XAPI does not support using an element within another structure as a
     * discriminant.  Placing this redundant type value here until that design
     * constraint can be fixed.
     */
    LwU8                 type;
    /*!
     * Frequency controller ID @ref LW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_ID_<xyz>
     */
    LwU8                 controllerId;
    /*!
     * Mode for the frequency from partitions for a clock domain.
     * @ref LW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_PARTS_FREQ_MODE_<xyz>.
     */
    LwU8                 partsFreqMode;
    /*!
     * Clock domain @ref LW2080_CTRL_CLK_DOMAIN_<xyz>.
     */
    LwU32                clkDomain;
    /*!
     * Type-specific data union.
     */
    union {
        LW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_INFO_PI    pi;
        LW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_INFO_10_PI v10Pi;
        LW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_INFO_20_PI v20Pi;
    } data;
} LW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_INFO;
typedef struct LW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_INFO *PLW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_INFO;

/*!
 * Structure describing CLK_FREQ_CONTROLLERS static information/POR.
 * Implements the BOARDOBJGRP model/interface.
 */
#define LW2080_CTRL_CLK_CLK_FREQ_CONTROLLERS_GET_INFO_PARAMS_MESSAGE_ID (0x25U)

typedef struct LW2080_CTRL_CLK_CLK_FREQ_CONTROLLERS_GET_INFO_PARAMS {
    /*!
     * LW2080_CTRL_BOARDOBJGRP super class. Must always be first object in
     * structure.
     */
    LW2080_CTRL_BOARDOBJGRP_E32              super;
    /*!
     * Sampling period at which the frequency controllers will run.
     */
    LwU32                                    samplingPeriodms;
    /*!
     * Voltage policy table index required for applying voltage delta.
     */
    LwU8                                     voltPolicyIdx;
    /*!
     * Boolean to indicate if continuous mode is enabled.
     */
    LwBool                                   bContinuousMode;
    /*!
     * Array of CLK_FREQ_CONTROLLER structures. Has valid indexes corresponding
     * to the bits set in @ref super.objMask.
     */
    LW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_INFO freqControllers[LW2080_CTRL_BOARDOBJGRP_E32_MAX_OBJECTS];
} LW2080_CTRL_CLK_CLK_FREQ_CONTROLLERS_GET_INFO_PARAMS;
typedef struct LW2080_CTRL_CLK_CLK_FREQ_CONTROLLERS_GET_INFO_PARAMS *PLW2080_CTRL_CLK_CLK_FREQ_CONTROLLERS_GET_INFO_PARAMS;

/*!
 * LW2080_CTRL_CMD_CLK_CLK_FREQ_CONTROLLERS_GET_INFO
 *
 * This command returns CLK_FREQ_CONTROLLERS static object information/POR
 * as populated by the RM from the frequency controller VBIOS table.
 *
 * The CLK_FREQ_CONTROLLER objects are indexed in the order by which the RM
 * allocates them.
 *
 * See @ref LW2080_CTRL_CLK_CLK_FREQ_CONTROLLERS_GET_INFO_PARAMS for the
 * documentation on the parameters.
 *
 * Return values are specified per @ref BoardObjGrpGetInfo.
 */
#define LW2080_CTRL_CMD_CLK_CLK_FREQ_CONTROLLERS_GET_INFO (0x20801025) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_CLK_INTERFACE_ID << 8) | LW2080_CTRL_CLK_CLK_FREQ_CONTROLLERS_GET_INFO_PARAMS_MESSAGE_ID" */

/*!
 * Structure describing the dynamic state of the _PI class.
 */
typedef struct LW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_STATUS_PI {
    /*!
     * [out] Voltage delta after each iteration for this controller.
     */
    LwS32 voltDeltauV;

    /*!
     * [out] Frequency error for the current cycle.
     */
    LwS32 errorkHz;
} LW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_STATUS_PI;
typedef struct LW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_STATUS_PI *PLW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_STATUS_PI;

/*!
 * Structure describing CLK_FREQ_CONTROLLER dynamic information
 * Implements the BOARDOBJ model/interface.
 */
typedef struct LW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_STATUS_10 {
    /*!
     * [out] Mask of clients requested to disable this controller.
     * @ref LW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_CLIENT_ID_<xyz>.
     */
    LwU32 disableClientsMask;
} LW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_STATUS_10;
typedef struct LW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_STATUS_10 *PLW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_STATUS_10;

/*!
 * Structure describing the dynamic state of the CLK_FREQ_CONTROLLER_10_PI class.
 */
typedef struct LW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_STATUS_10_PI {
    /*!
     * LW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_STATUS_10 super class.
     * Must always be first object in structure.
     */
    LW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_STATUS_10 super;
    /*!
     * [out] Voltage delta after each iteration for this controller.
     */
    LwS32                                         voltDeltauV;
    /*!
     * [out] Frequency error for the current cycle.
     */
    LwS32                                         errorkHz;
} LW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_STATUS_10_PI;
typedef struct LW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_STATUS_10_PI *PLW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_STATUS_10_PI;

/*!
 * Structure describing CLK_FREQ_CONTROLLER dynamic information
 * Implements the BOARDOBJ model/interface.
 */
typedef struct LW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_STATUS_20 {
    /*!
     * rsvd
     */
    LwU8 rsvd;
} LW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_STATUS_20;
typedef struct LW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_STATUS_20 *PLW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_STATUS_20;

/*!
 * Structure describing the dynamic state of the CLK_FREQ_CONTROLLER_20_PI class.
 */
typedef struct LW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_STATUS_20_PI {
    /*!
     * LW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_STATUS_20 super class.
     * Must always be first object in structure.
     */
    LW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_STATUS_20 super;

    /*!
     * [out] Voltage delta after each iteration for this controller.
     */
    LwS32                                         voltDeltauV;

    /*!
     * [out] Frequency error for the current cycle.
     */
    LwS32                                         errorkHz;
} LW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_STATUS_20_PI;
typedef struct LW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_STATUS_20_PI *PLW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_STATUS_20_PI;

/*!
 * CLK_FREQ_CONTROLLER type-specific data union. Discriminated by
 * CLK_FREQ_CONTROLLER::super.type.
 */


/*!
 * Structure describing CLK_FREQ_CONTROLLER dynamic information
 * Implements the BOARDOBJ model/interface.
 */
typedef struct LW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_STATUS {
    /*!
     * LW2080_CTRL_BOARDOBJ super class.  Must always be first object in
     * structure.
     */
    LW2080_CTRL_BOARDOBJ super;
    /*!
     * XAPI does not support using an element within another structure as a
     * discriminant.  Placing this redundant type value here until that design
     * constraint can be fixed.
     */
    LwU8                 type;

    /*!
     * [out] Mask of clients requested to disable this controller.
     * @ref LW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_CLIENT_ID_<xyz>.
     */
    LwU32                disableClientsMask;
    /*!
     * Type-specific data union.
     */
    union {
        LW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_STATUS_PI    pi;
        LW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_STATUS_10_PI v10Pi;
        LW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_STATUS_20_PI v20Pi;
    } data;
} LW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_STATUS;
typedef struct LW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_STATUS *PLW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_STATUS;

/*!
 * Structure describing CLK_FREQ_CONTROLLERS static information/POR.
 * Implements the BOARDOBJGRP model/interface.
 */
#define LW2080_CTRL_CLK_CLK_FREQ_CONTROLLERS_GET_STATUS_PARAMS_MESSAGE_ID (0x26U)

typedef struct LW2080_CTRL_CLK_CLK_FREQ_CONTROLLERS_GET_STATUS_PARAMS {
    /*!
     * LW2080_CTRL_BOARDOBJGRP super class. Must always be first object in
     * structure.
     */
    LW2080_CTRL_BOARDOBJGRP_E32                super;

    /*!
     * [out] Final voltage delta applied after each iteration.
     * This is the max voltage delta from all controllers.
     */
    LwS32                                      finalVoltDeltauV;

    /*!
     * [out] Array of CLK_FREQ_CONTROLLER structures. Has valid indexes
     * corresponding to the bits set in @ref super.objMask.
     */
    LW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_STATUS freqControllers[LW2080_CTRL_BOARDOBJGRP_E32_MAX_OBJECTS];
} LW2080_CTRL_CLK_CLK_FREQ_CONTROLLERS_GET_STATUS_PARAMS;
typedef struct LW2080_CTRL_CLK_CLK_FREQ_CONTROLLERS_GET_STATUS_PARAMS *PLW2080_CTRL_CLK_CLK_FREQ_CONTROLLERS_GET_STATUS_PARAMS;

/*!
 * LW2080_CTRL_CMD_CLK_CLK_FREQ_CONTROLLERS_GET_STATUS
 *
 * This command returns CLK_FREQ_CONTROLLERS dynamic object information from the
 * PMU.
 *
 * See @ref LW2080_CTRL_CMD_CLK_CLK_FREQ_CONTROLLERS_GET_STATUS for the
 * documentation on the parameters.
 *
 * Return values are specified per @ref BoardObjGrpGetStatus.
 */
#define LW2080_CTRL_CMD_CLK_CLK_FREQ_CONTROLLERS_GET_STATUS (0x20801026) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_CLK_INTERFACE_ID << 8) | LW2080_CTRL_CLK_CLK_FREQ_CONTROLLERS_GET_STATUS_PARAMS_MESSAGE_ID" */

/*!
 * Structure describing CLK_FREQ_CONTROLLER_PI specific control parameters.
 */
typedef struct LW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_CONTROL_PI {
    /*!
     * Proportional gain for this frequency controller in uV/MHz
     */
    LwSFXP20_12 propGain;
    /*!
     * Integral gain for this frequency controller in uV/MHz
     */
    LwSFXP20_12 integGain;
    /*!
     * Decay factor for the integral term.
     */
    LwSFXP20_12 integDecay;
    /*!
     * Voltage delta limit range min value.
     */
    LwS32       voltDeltaMin;
    /*!
     * Voltage delta limit range max value.
     */
    LwS32       voltDeltaMax;
    /*!
     * Minimum percentage time of the HW slowdown required in a
     * sampling period to poison the sample.
     */
    LwU8        slowdownPctMin;
    /*!
     * Whether to poison the sample only if the slowdown oclwrred
     * on the clock domain of this frequency controller. FALSE means
     * poison the sample even if slowdown oclwrred on other clock domains.
     */
    LwBool      bPoison;
    /*!
     * Minimum percentage of time BA should engage to poison the sample.
     */
    LwUFXP4_12  baPctMin;
    /*!
     * Minimum percentage of time droopy should engage to poison the sample.
     */
    LwUFXP4_12  droopyPctMin;
} LW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_CONTROL_PI;
typedef struct LW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_CONTROL_PI *PLW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_CONTROL_PI;

/*!
 * Structure representing the control parameters associated with a CLK_FREQ_CONTROLLER_10
 */
typedef struct LW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_CONTROL_10 {
    /*!
     * Keep the controller disabled
     */
    LwBool bDisable;
    /*!
     * Mask of clients requested to disable this controller.
     * @ref LW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_CLIENT_ID_<xyz>.
     */
    LwU32  disableClientsMask;
    /*!
     * Mode of operation @ref LW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_MODE_<xyz>.
     */
    LwU8   freqMode;
    /*!
     * Frequency Cap in MHZ to be applied when V/F point is above
     * Noise Unaware Vmin
     */
    LwS16  freqCapNoiseUnawareVminAbove;
    /*!
     * Frequency Cap in MHz to be applied when V/F point is below
     * Noise Unaware Vmin
     */
    LwS16  freqCapNoiseUnawareVminBelow;
    /*!
     * Absolute value of Positive Frequency Hysteresis in MHz (0 => no hysteresis).
     * (hysteresis to apply when frequency has positive delta)
     */
    LwS16  freqHysteresisPositive;
    /*!
     * Absolute value of Negative Frequency Hysteresis in MHz (0 => no hysteresis).
     * (hysteresis to apply when frequency has negative delta)
     */
    LwS16  freqHysteresisNegative;
} LW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_CONTROL_10;
typedef struct LW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_CONTROL_10 *PLW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_CONTROL_10;

/*!
 * Structure describing CLK_FREQ_CONTROLLER_10_PI specific control parameters.
 */
typedef struct LW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_CONTROL_10_PI {
    /*!
     * LW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_CONTROL_10 super class.
     * Must always be first object in structure.
     */
    LW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_CONTROL_10 super;
    /*!
     * Proportional gain for this frequency controller in uV/MHz
     */
    LwSFXP20_12                                    propGain;
    /*!
     * Integral gain for this frequency controller in uV/MHz
     */
    LwSFXP20_12                                    integGain;
    /*!
     * Decay factor for the integral term.
     */
    LwSFXP20_12                                    integDecay;
    /*!
     * Voltage delta limit range min value.
     */
    LwS32                                          voltDeltaMin;
    /*!
     * Voltage delta limit range max value.
     */
    LwS32                                          voltDeltaMax;
    /*!
     * Minimum percentage time of the HW slowdown required in a
     * sampling period to poison the sample.
     */
    LwU8                                           slowdownPctMin;
    /*!
     * Whether to poison the sample only if the slowdown oclwrred
     * on the clock domain of this frequency controller. FALSE means
     * poison the sample even if slowdown oclwrred on other clock domains.
     */
    LwBool                                         bPoison;
    /*!
     * Minimum percentage of time BA should engage to poison the sample.
     */
    LwUFXP4_12                                     baPctMin;
    /*!
     * Minimum percentage of time droopy should engage to poison the sample.
     */
    LwUFXP4_12                                     droopyPctMin;
} LW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_CONTROL_10_PI;
typedef struct LW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_CONTROL_10_PI *PLW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_CONTROL_10_PI;

/*!
 * Structure representing the control parameters associated with a CLK_FREQ_CONTROLLER_20.
 */
typedef struct LW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_CONTROL_20 {
    /*!
     * Keep the controller disabled
     */
    LwBool                                            bDisable;
    /*!
     * Mask of clients requested to disable this controller.
     * @ref LW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_CLIENT_ID_<xyz>.
     */
    LwU32                                             disableClientsMask;
    /*!
     * Mode of operation @ref LW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_MODE_<xyz>.
     */
    LwU8                                              freqMode;
    /*!
     * Absolute CLVC voltage error Threshold
     * Ignore CLFC voltage offset of given sample if CLVC voltage error (Vrequested - Vsensed)
     * of current sample is greater than or equal to voltage error Threshold.
     */
    LwU32                                             voltErrorThresholduV;
    /*!
     * Minimum threshold value above which controller sample is poisoned (0.0 - 1.0)
     */
    LwUFXP4_12                                        hwSlowdownThreshold;
    /*!
     * Voltage offset range min (in uV)
     */
    LwS32                                             voltOffsetMinuV;
    /*!
     * Voltage offset range max (in uV)
     */
    LwS32                                             voltOffsetMaxuV;
    /*!
     * Therm Monitors
     */
    LW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_THERM_MONITOR thermMonitor[LW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_THERM_MONITOR_MAX];
} LW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_CONTROL_20;
typedef struct LW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_CONTROL_20 *PLW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_CONTROL_20;

/*!
 * Structure describing CLK_FREQ_CONTROLLER_20_PI specific control parameters.
 */
typedef struct LW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_CONTROL_20_PI {
    /*!
     * LW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_CONTROL_20 super class.
     * Must always be first object in structure.
     */
    LW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_CONTROL_20 super;
    /*!
     * Proportional gain for this frequency controller in uV/MHz
     */
    LwSFXP20_12                                    propGain;
    /*!
     * Integral gain for this frequency controller in uV/MHz
     */
    LwSFXP20_12                                    integGain;
    /*!
     * Decay factor for the integral term.
     */
    LwSFXP20_12                                    integDecay;
    /*!
     * Absolute value of Positive Frequency Hysteresis in MHz (0 => no hysteresis).
     * (hysteresis to apply when frequency has positive delta)
     */
    LwS32                                          freqHystPosMHz;
    /*!
     * Absolute value of Negative Frequency Hysteresis in MHz (0 => no hysteresis).
     * (hysteresis to apply when frequency has negative delta)
     */
    LwS32                                          freqHystNegMHz;
} LW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_CONTROL_20_PI;
typedef struct LW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_CONTROL_20_PI *PLW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_CONTROL_20_PI;

/*!
 * CLK_FREQ_CONTROLLER type-specific data union.  Discriminated by
 * CLK_FREQ_CONTROLLER::super.type.
 */


/*!
 * Structure representing the control parameters associated with a CLK_FREQ_CONTROLLER
 */
typedef struct LW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_CONTROL {
    /*!
     * LW2080_CTRL_BOARDOBJ super class.  Must always be first object in
     * structure.
     */
    LW2080_CTRL_BOARDOBJ super;
    /*!
     * XAPI does not support using an element within another structure as a
     * discriminant.  Placing this redundant type value here until that design
     * constraint can be fixed.
     */
    LwU8                 type;
    /*!
     * Keep the controller disabled
     */
    LwBool               bDisable;
    /*!
     * Frequency Cap in MHZ to be applied when V/F point is above
     * Noise Unaware Vmin
     */
    LwS16                freqCapNoiseUnawareVminAbove;
    /*!
     * Frequency Cap in MHz to be applied when V/F point is below
     * Noise Unaware Vmin
     */
    LwS16                freqCapNoiseUnawareVminBelow;
    /*!
     * Absolute value of Positive Frequency Hysteresis in MHz (0 => no hysteresis).
     * (hysteresis to apply when frequency has positive delta)
     */
    LwS16                freqHysteresisPositive;
    /*!
     * Absolute value of Negative Frequency Hysteresis in MHz (0 => no hysteresis).
     * (hysteresis to apply when frequency has negative delta)
     */
    LwS16                freqHysteresisNegative;
    /*!
     * Type-specific data union.
     */
    union {
        LW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_CONTROL_PI    pi;
        LW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_CONTROL_10_PI v10Pi;
        LW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_CONTROL_20_PI v20Pi;
    } data;
} LW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_CONTROL;
typedef struct LW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_CONTROL *PLW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_CONTROL;

/*!
 * Structure representing the control parameters associated with a CLK_FREQ_CONTROLLERS.
 */
typedef struct LW2080_CTRL_CLK_CLK_FREQ_CONTROLLERS_CONTROL_PARAMS {
    /*!
     * LW2080_CTRL_BOARDOBJGRP_E32 super class.  Must always be first object in
     * structure.
     */
    LW2080_CTRL_BOARDOBJGRP_E32                 super;

    /*!
     * Sampling period at which the frequency controllers will run.
     */
    LwU32                                       samplingPeriodms;

    /*!
     * Array of CLK_FREQ_CONTROLLER structures.  Has valid indexes corresponding to the
     * bits set in @ref super.objMask.
     */
    LW2080_CTRL_CLK_CLK_FREQ_CONTROLLER_CONTROL freqControllers[LW2080_CTRL_BOARDOBJGRP_E32_MAX_OBJECTS];
} LW2080_CTRL_CLK_CLK_FREQ_CONTROLLERS_CONTROL_PARAMS;
typedef struct LW2080_CTRL_CLK_CLK_FREQ_CONTROLLERS_CONTROL_PARAMS *PLW2080_CTRL_CLK_CLK_FREQ_CONTROLLERS_CONTROL_PARAMS;

/*!
 * LW2080_CTRL_CMD_CLK_CLK_FREQ_CONTROLLERS_GET_CONTROL
 *
 * This command returns CLK_FREQ_CONTROLLERS control parameters as specified by
 * the VBIOS in either Frequency Controller Table.
 *
 * See @ref LW2080_CTRL_CMD_CLK_CLK_FREQ_CONTROLLERS_CONTROL_PARAMS for
 * documentation on the parameters.
 *
 * Return values are specified per @ref BoardObjGrpGetControl.
 */
#define LW2080_CTRL_CMD_CLK_CLK_FREQ_CONTROLLERS_GET_CONTROL (0x20801027) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_CLK_INTERFACE_ID << 8) | 0x27" */

/*!
 * LW2080_CTRL_CMD_CLK_CLK_FREQ_CONTROLLERS_SET_CONTROL
 *
 * This command accepts client-specified control parameters for a set
 * of CLK_FREQ_CONTROLLERS entries in the Clocks Frequency Table, and applies
 * these new parameters to the set of CLK_FREQ_CONTROLLERS entries.
 *
 * See @ref LW2080_CTRL_CMD_CLK_CLK_FREQ_CONTROLLERS_CONTROL_PARAMS for
 * documentation on the parameters.
 *
 * Return values are specified per @ref BoardObjGrpSetControl.
 */
#define LW2080_CTRL_CMD_CLK_CLK_FREQ_CONTROLLERS_SET_CONTROL (0x20801028) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_CLK_INTERFACE_ID << 8) | 0x28" */

/*!
 * Enumeration of CLIENT_CLK_DOMAIN class types.
 */
#define LW2080_CTRL_CLK_CLIENT_CLK_DOMAIN_TYPE_FIXED         0x00U
#define LW2080_CTRL_CLK_CLIENT_CLK_DOMAIN_TYPE_PROG          0x01U
#define LW2080_CTRL_CLK_CLIENT_CLK_DOMAIN_TYPE_UNKNOWN       0xFFU

/*!
 * Enumeration of volt rail types for CLIENT_CLK_DOMAIN
 */
#define LW2080_CTRL_CLK_CLIENT_CLK_DOMAINa_VOLT_RAIL_LWVDD   0x00U
#define LW2080_CTRL_CLK_CLIENT_CLK_DOMAIN_VOLT_RAIL_MSVDD    0x01U

/*!
 * Macro representing an INVALID/UNSUPPORTED CLIENT_CLK_DOMAIN index.
 */
#define LW2080_CTRL_CLK_CLIENT_CLK_DOMAIN_IDX_ILWALID        LW2080_CTRL_BOARDOBJ_IDX_ILWALID_8BIT

/*!
 * Structure describing CLIENT_CLK_DOMAIN_PROG static information/POR.
 */
typedef struct LW2080_CTRL_CLK_CLIENT_CLK_DOMAIN_INFO_PROG {
    /*!
     * Maximum frequency offset which can be applied to any point on the domain.
     *
     * If @ref freqOffsetMaxkHz == @ref freqOffsetMinkHz == 0, overclocking is
     * not enabled on this clock domain.
     */
    LwS32 freqOffsetMaxkHz;
    /*!
     * Minimum frequency offset which can be applied to any point domain.
     *
     * If @ref freqOffsetMaxkHz == @ref freqOffsetMinkHz == 0, overclocking is
     * not enabled on this clock domain.
     */
    LwS32 freqOffsetMinkHz;
    /*!
     * Index into the CLIENT CLK_VF_POINTs for first CLK_VF_POINT belonging to
     * this domain.
     */
    LwU8  vfPointIdxFirst;
    /*!
     * Index into the CLIENT CLK_VF_POINTs for last CLK_VF_POINT belonging to
     * this domain.
     */
    LwU8  vfPointIdxLast;
} LW2080_CTRL_CLK_CLIENT_CLK_DOMAIN_INFO_PROG;
typedef struct LW2080_CTRL_CLK_CLIENT_CLK_DOMAIN_INFO_PROG *PLW2080_CTRL_CLK_CLIENT_CLK_DOMAIN_INFO_PROG;

/*!
 * CLIENT_CLK_DOMAIN type-specific data union.  Discriminated by
 * CLIENT_CLK_DOMAIN::super.type.
 */


/*!
 * Structure describing CLIENT_CLK_DOMAIN static information/POR.  Implements the
 * BOARDOBJ model/interface.
 */
typedef struct LW2080_CTRL_CLK_CLIENT_CLK_DOMAIN_INFO {
    /*!
     * LW2080_CTRL_BOARDOBJ super class.  Must always be first object in
     * structure.
     */
    LW2080_CTRL_BOARDOBJ          super;
    /*!
     * XAPI does not support using an element within another structure as a
     * discriminant.  Placing this redundant type value here until that design
     * constraint can be fixed.
     */
    LwU8                          type;
    /*!
     * @ref LW2080_CTRL_CLK_DOMAIN_<xyz>
     * @deprecated
     */
    LwU32                         clkDomain;
    /*!
     * @ref LW2080_CTRL_CLK_PUBLIC_DOMAIN_ENUM
     *
     * @note _VIDEO domain is not supported yet in client clk domains
     */
    LW2080_CTRL_CLK_PUBLIC_DOMAIN publicDomain;
    /*!
     * @ref LW2080_CTRL_CLK_CLIENT_CLK_DOMAIN_VOLT_RAIL_<xyz>
     * Mask of volt rails on which this clock domain has its Vmin.
     */
    LwU32                         voltDomainMask;
    /*!
     * Type-specific data union.
     */
    union {
        LW2080_CTRL_CLK_CLIENT_CLK_DOMAIN_INFO_PROG prog;
    } data;
} LW2080_CTRL_CLK_CLIENT_CLK_DOMAIN_INFO;
typedef struct LW2080_CTRL_CLK_CLIENT_CLK_DOMAIN_INFO *PLW2080_CTRL_CLK_CLIENT_CLK_DOMAIN_INFO;

/*!
 * Structure describing CLIENT_CLK_DOMAINS static information/POR.  Implements
 * the BOARDOBJGRP model/interface.
 */
#define LW2080_CTRL_CLK_CLIENT_CLK_DOMAINS_INFO_MESSAGE_ID (0x29U)

typedef struct LW2080_CTRL_CLK_CLIENT_CLK_DOMAINS_INFO {
    /*!
     * LW2080_CTRL_BOARDOBJGRP_E32 super class.  Must always be first object in
     * structure.
     */
    LW2080_CTRL_BOARDOBJGRP_E32            super;

    /*!
     * Array of CLIENT_CLK_DOMAIN structures.  Has valid indexes corresponding
     * to the bits set in @ref super.objMask.
     */
    LW2080_CTRL_CLK_CLIENT_CLK_DOMAIN_INFO domains[LW2080_CTRL_BOARDOBJGRP_E32_MAX_OBJECTS];
} LW2080_CTRL_CLK_CLIENT_CLK_DOMAINS_INFO;
typedef struct LW2080_CTRL_CLK_CLIENT_CLK_DOMAINS_INFO *PLW2080_CTRL_CLK_CLIENT_CLK_DOMAINS_INFO;

/*!
 * LW2080_CTRL_CMD_CLK_CLIENT_CLK_DOMAINS_GET_INFO
 *
 * This command returns CLINET_CLK_DOMAINS static object information/POR as
 * generated by the RM from the VBIOS specification.
 *
 * CLIENT_CLK_DOMAINs are CLK_DOMAINs which the RM is exporting to
 * end-users/clients for run-time configuration in production software, e.g. for
 * overclocking.
 *
 * See @ref LW2080_CTRL_CLK_CLIENT_CLK_DOMAINS_INFO for documentation on the
 * parameters.
 *
 * Return values are specified per @ref BoardObjGrpGetInfo.
 */
#define LW2080_CTRL_CMD_CLK_CLIENT_CLK_DOMAINS_GET_INFO (0x20801029) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_CLK_INTERFACE_ID << 8) | LW2080_CTRL_CLK_CLIENT_CLK_DOMAINS_INFO_MESSAGE_ID" */

/*!
 * Structure representing the client controllable state of the CLIENT_CLK_DOMAIN_PROG.
 */
typedef struct LW2080_CTRL_CLK_CLIENT_CLK_DOMAIN_CONTROL_PROG {
    /*!
     * Deltas for CLIENT_CLK_DOMAIN class
     */
    LW2080_CTRL_CLK_CLK_DELTA delta;
    /*!
     * CPM max Freq Offset override value
     */
    LwU16                     cpmMaxFreqOffsetOverrideMHz;
} LW2080_CTRL_CLK_CLIENT_CLK_DOMAIN_CONTROL_PROG;
typedef struct LW2080_CTRL_CLK_CLIENT_CLK_DOMAIN_CONTROL_PROG *PLW2080_CTRL_CLK_CLIENT_CLK_DOMAIN_CONTROL_PROG;

/*!
 * Union of CLIENT_CLK_DOMAIN_TYPE-specific client controllable data.
 */


/*!
 * Structure representing the client controllable state of  CLIENT_CLK_DOMAIN.
 * Implements the BOARDOBJ model/interface.
 */
typedef struct LW2080_CTRL_CLK_CLIENT_CLK_DOMAIN_CONTROL {
    /*!
     * LW2080_CTRL_BOARDOBJ super class.  Must always be first object in
     * structure.
     */
    LW2080_CTRL_BOARDOBJ super;
    /*!
     * XAPI does not support using an element within another structure as a
     * discriminant.  Placing this redundant type value here until that design
     * constraint can be fixed.
     */
    LwU8                 type;

    /*!
     * Type-specific data union.
     */
    union {
        LW2080_CTRL_CLK_CLIENT_CLK_DOMAIN_CONTROL_PROG prog;
    } data;
} LW2080_CTRL_CLK_CLIENT_CLK_DOMAIN_CONTROL;
typedef struct LW2080_CTRL_CLK_CLIENT_CLK_DOMAIN_CONTROL *PLW2080_CTRL_CLK_CLIENT_CLK_DOMAIN_CONTROL;

/*!
 * Structure describing CLIENT_CLK_DOMAINS control values.  Implements
 * the BOARDOBJGRP model/interface.
 */
typedef struct LW2080_CTRL_CLK_CLIENT_CLK_DOMAINS_CONTROL {
    /*!
     * LW2080_CTRL_BOARDOBJGRP_E32 super class.  Must always be first object in
     * structure.
     */
    LW2080_CTRL_BOARDOBJGRP_E32               super;

    /*!
     * Array of CLIENT_CLK_DOMAIN structures.  Has valid indexes corresponding
     * to the bits set in @ref super.objMask.
     */
    LW2080_CTRL_CLK_CLIENT_CLK_DOMAIN_CONTROL domains[LW2080_CTRL_BOARDOBJGRP_E32_MAX_OBJECTS];
} LW2080_CTRL_CLK_CLIENT_CLK_DOMAINS_CONTROL;
typedef struct LW2080_CTRL_CLK_CLIENT_CLK_DOMAINS_CONTROL *PLW2080_CTRL_CLK_CLIENT_CLK_DOMAINS_CONTROL;

/*!
 * RM support one link corresponding to each volt rail. Because we are exposing
 * only single rail to clients, we will always consider rail at index "0" as
 * primary volt rail and expose its VF points information to clients tools
 * through GET INFO/STATUS/CONTROL RMCTRLs.
 */
#define LW2080_CTRL_CLK_CLIENT_CLK_VF_POINT_LINK_PRIMARY       0x00U

/*!
 * LW2080_CTRL_CMD_CLK_CLIENT_CLK_DOMAINS_GET_CONTROL
 *
 * This command returns CLINET_CLK_DOMAINS control values.  These are values
 * hich the client can directly control.  These values will not change except
 * via calls to this interface.
 *
 * @note This interface is expected to be used in tandem with @ref
 * LW2080_CTRL_CMD_CLK_CLIENT_CLK_DOMAINS_GET_CONTROL in a "Read-Modify-Write"
 * manner.
 *
 * See @ref LW2080_CTRL_CLK_CLIENT_CLK_DOMAINS_CONTROL for documentation on the
 * parameters.
 *
 * Return values are specified per @ref BoardObjGrpGetControl.
 */
#define LW2080_CTRL_CMD_CLK_CLIENT_CLK_DOMAINS_GET_CONTROL     (0x20801030) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_CLK_INTERFACE_ID << 8) | 0x30" */

/*!
 * LW2080_CTRL_CMD_CLK_CLIENT_CLK_DOMAINS_SET_CONTROL
 *
 * This command sets CLINET_CLK_DOMAINS control values.  These are values
 * which the client can directly control.  These values will not change except
 * via calls to this interface.
 *
 * @note This interface is expected to be used in tandem with @ref
 * LW2080_CTRL_CMD_CLK_CLIENT_CLK_DOMAINS_GET_CONTROL in a "Read-Modify-Write"
 * manner.
 *
 * See @ref LW2080_CTRL_CLK_CLIENT_CLK_DOMAINS_CONTROL for documentation on the
 * parameters.
 *
 * Return values are specified per @ref BoardObjGrpSetControl.
 */
#define LW2080_CTRL_CMD_CLK_CLIENT_CLK_DOMAINS_SET_CONTROL     (0x20801031) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_CLK_INTERFACE_ID << 8) | 0x31" */

/*!
 * Enumeration of CLIENT_CLK_VF_POINT class types.
 */
#define LW2080_CTRL_CLK_CLIENT_CLK_VF_POINT_TYPE_FIXED         0x00U
#define LW2080_CTRL_CLK_CLIENT_CLK_VF_POINT_TYPE_PROG          0x01U
#define LW2080_CTRL_CLK_CLIENT_CLK_VF_POINT_TYPE_FIXED_10_RSVD 0x02U
#define LW2080_CTRL_CLK_CLIENT_CLK_VF_POINT_TYPE_PROG_10_RSVD  0x03U
#define LW2080_CTRL_CLK_CLIENT_CLK_VF_POINT_TYPE_FIXED_20      0x04U
#define LW2080_CTRL_CLK_CLIENT_CLK_VF_POINT_TYPE_PROG_20       0x05U
#define LW2080_CTRL_CLK_CLIENT_CLK_VF_POINT_TYPE_PROG_20_FREQ  0x06U
#define LW2080_CTRL_CLK_CLIENT_CLK_VF_POINT_TYPE_PROG_20_VOLT  0x07U
#define LW2080_CTRL_CLK_CLIENT_CLK_VF_POINT_TYPE_UNKNOWN       0xFFU

// PP-TODO : Temp change to update LWAPIs.
#define LW2080_CTRL_CLK_CLIENT_CLK_VF_POINT_TYPE_PROG_10       LW2080_CTRL_CLK_CLIENT_CLK_VF_POINT_TYPE_PROG
#define LW2080_CTRL_CLK_CLIENT_CLK_VF_POINT_TYPE_FIXED_10      LW2080_CTRL_CLK_CLIENT_CLK_VF_POINT_TYPE_FIXED

/*!
 * Special define to represent an invalid CLIENT_CLK_VF_POINT index.
 */
#define LW2080_CTRL_CLK_CLIENT_CLK_VF_POINT_INDEX_ILWALID      LW2080_CTRL_BOARDOBJ_IDX_ILWALID_8BIT

/*!
 * Structure describing CLIENT_CLK_VF_POINT static information/POR.  Implements the
 * BOARDOBJ model/interface.
 */
typedef struct LW2080_CTRL_CLK_CLIENT_CLK_VF_POINT_INFO {
    /*!
     * LW2080_CTRL_BOARDOBJ super class.  Must always be first object in
     * structure.
     */
    LW2080_CTRL_BOARDOBJ super;
    /*!
     * XAPI does not support using an element within another structure as a
     * discriminant.  Placing this redundant type value here until that design
     * constraint can be fixed.
     */
    LwU8                 type;

    /*!
     * Boolean indicating if the CLK_VF_POINT is VOLTAGE based (i.e. the GPU
     * will attempt to hit this VF point using a voltage-based/noise-aware clock
     * generator).
     *
     * When LW_TRUE, this VF_POINT should be tuned for OC by setting a VOLTAGE
     * limit at the voltage value for this VF_POINT.  When LW_FALSE, this
     * VF_POINT should be tuned for OC by setting a FREQUENCY limit at the
     * frequency value for this VF_POINT.
     */
    LwBool               bVoltageBased;
} LW2080_CTRL_CLK_CLIENT_CLK_VF_POINT_INFO;
typedef struct LW2080_CTRL_CLK_CLIENT_CLK_VF_POINT_INFO *PLW2080_CTRL_CLK_CLIENT_CLK_VF_POINT_INFO;

/*!
 * Structure describing CLIENT_CLK_VF_POINTS static information/POR.  Implements
 * the BOARDOBJGRP model/interface.
 */
#define LW2080_CTRL_CLK_CLIENT_CLK_VF_POINTS_INFO_MESSAGE_ID (0x2AU)

typedef struct LW2080_CTRL_CLK_CLIENT_CLK_VF_POINTS_INFO {
    /*!
     * LW2080_CTRL_BOARDOBJGRP_E32 super class.  Must always be first object in
     * structure.
     */
    LW2080_CTRL_BOARDOBJGRP_E255             super;

    /*!
     * Array of CLIENT_CLK_VF_POINT structures.  Has valid indexes corresponding
     * to the bits set in @ref super.objMask.
     */
    LW2080_CTRL_CLK_CLIENT_CLK_VF_POINT_INFO vfPoints[LW2080_CTRL_BOARDOBJGRP_E255_MAX_OBJECTS];
} LW2080_CTRL_CLK_CLIENT_CLK_VF_POINTS_INFO;
typedef struct LW2080_CTRL_CLK_CLIENT_CLK_VF_POINTS_INFO *PLW2080_CTRL_CLK_CLIENT_CLK_VF_POINTS_INFO;

/*!
 * LW2080_CTRL_CMD_CLK_CLIENT_CLK_DOMAINS_GET_INFO
 *
 * This command returns CLIENT_CLK_VF_POINTS static object information/POR as
 * generated by the RM from the VBIOS specification.
 *
 * CLIENT_CLK_VF_POINTs are CLK_VF_POINTs which the RM is exporting to
 * end-users/clients for run-time configuration in production software, e.g. for
 * overclocking.
 *
 * See @ref LW2080_CTRL_CLK_CLIENT_CLK_VF_POINTS_INFO for documentation on the
 * parameters.
 *
 * Return values are specified per @ref BoardObjGrpGetInfo.
 */
#define LW2080_CTRL_CMD_CLK_CLIENT_CLK_VF_POINTS_GET_INFO (0x2080102a) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_CLK_INTERFACE_ID << 8) | LW2080_CTRL_CLK_CLIENT_CLK_VF_POINTS_INFO_MESSAGE_ID" */

/*!
 * Structure describing CLIENT_CLK_VF_POINT dynamic state
 */
typedef struct LW2080_CTRL_CLK_CLIENT_CLK_VF_POINT_TUPLE {
    /*!
     * Base/current frequency of the vf point. In case of current value it includes
     * any applicable offsets
     */
    LwU32 freqkHz;

    /*!
     * Base/current voltage of the vf point.
     */
    LwU32 voltageuV;
} LW2080_CTRL_CLK_CLIENT_CLK_VF_POINT_TUPLE;
typedef struct LW2080_CTRL_CLK_CLIENT_CLK_VF_POINT_TUPLE *PLW2080_CTRL_CLK_CLIENT_CLK_VF_POINT_TUPLE;

/*!
 * Structure describing CLIENT_CLK_VF_POINT_PROG dynamic state
 */
typedef struct LW2080_CTRL_CLK_CLIENT_CLK_VF_POINT_STATUS_PROG {
    /*!
     * Reserved for future expansion.
     */
    LwU8 rsvd;
} LW2080_CTRL_CLK_CLIENT_CLK_VF_POINT_STATUS_PROG;
typedef struct LW2080_CTRL_CLK_CLIENT_CLK_VF_POINT_STATUS_PROG *PLW2080_CTRL_CLK_CLIENT_CLK_VF_POINT_STATUS_PROG;

/*!
 * Structure describing CLIENT_VF_POINT_PROG_10 dynamic state
 */
typedef struct LW2080_CTRL_CLK_CLIENT_CLK_VF_POINT_STATUS_PROG_10 {
    /*!
     * LW2080_CTRL_CLK_CLIENT_CLK_VF_POINT_PROG_STATUS super class. Must always be first
     * object in structure
     */
    LW2080_CTRL_CLK_CLIENT_CLK_VF_POINT_STATUS_PROG super;
} LW2080_CTRL_CLK_CLIENT_CLK_VF_POINT_STATUS_PROG_10;
typedef struct LW2080_CTRL_CLK_CLIENT_CLK_VF_POINT_STATUS_PROG_10 *PLW2080_CTRL_CLK_CLIENT_CLK_VF_POINT_STATUS_PROG_10;

/*!
 * Structure describing CLIENT_VF_POINT_PROG_20 dynamic state as well as base values
 */
typedef struct LW2080_CTRL_CLK_CLIENT_CLK_VF_POINT_STATUS_PROG_20 {
    /*!
     * LW2080_CTRL_CLK_CLIENT_CLK_VF_POINT_PROG_STATUS super class. Must always be first
     * object in structure
     */
    LW2080_CTRL_CLK_CLIENT_CLK_VF_POINT_STATUS_PROG super;

    /*!
     * Base values of voltage and frequency of the CLK_VF_POINT.
     */
    LW2080_CTRL_CLK_CLIENT_CLK_VF_POINT_TUPLE       vfTupleBase;
} LW2080_CTRL_CLK_CLIENT_CLK_VF_POINT_STATUS_PROG_20;
typedef struct LW2080_CTRL_CLK_CLIENT_CLK_VF_POINT_STATUS_PROG_20 *PLW2080_CTRL_CLK_CLIENT_CLK_VF_POINT_STATUS_PROG_20;



/*!
 * Structure describing CLIENT_CLK_VF_POINT dynamic state.  Implements the
 * BOARDOBJ model/interface.
 */
typedef struct LW2080_CTRL_CLK_CLIENT_CLK_VF_POINT_STATUS {
    /*!
     * LW2080_CTRL_BOARDOBJ super class.  Must always be first object in
     * structure.
     */
    LW2080_CTRL_BOARDOBJ                      super;
    /*!
     * XAPI does not support using an element within another structure as a
     * discriminant.  Placing this redundant type value here until that design
     * constraint can be fixed.
     */
    LwU8                                      type;
    /*!
     * Current values of voltage and frequency of the CLK_VF_POINT.
     */
    LW2080_CTRL_CLK_CLIENT_CLK_VF_POINT_TUPLE vfTupleOffset;
    /*!
     * Type-specific data union
     */
    union {
        LW2080_CTRL_CLK_CLIENT_CLK_VF_POINT_STATUS_PROG    prog;
        LW2080_CTRL_CLK_CLIENT_CLK_VF_POINT_STATUS_PROG_10 prog10;
        LW2080_CTRL_CLK_CLIENT_CLK_VF_POINT_STATUS_PROG_20 prog20;
    } data;
} LW2080_CTRL_CLK_CLIENT_CLK_VF_POINT_STATUS;
typedef struct LW2080_CTRL_CLK_CLIENT_CLK_VF_POINT_STATUS *PLW2080_CTRL_CLK_CLIENT_CLK_VF_POINT_STATUS;

/*!
 * Structure describing CLIENT_CLK_VF_POINTS static information/POR.  Implements
 * the BOARDOBJGRP model/interface.
 */
#define LW2080_CTRL_CLK_CLIENT_CLK_VF_POINTS_STATUS_MESSAGE_ID (0x2BU)

typedef struct LW2080_CTRL_CLK_CLIENT_CLK_VF_POINTS_STATUS {
    /*!
     * LW2080_CTRL_BOARDOBJGRP_E32 super class.  Must always be first object in
     * structure.
     */
    LW2080_CTRL_BOARDOBJGRP_E255               super;

    /*!
     * Boolean indicating base VF lwrve is supported.
     * It will be true for Ampere and later, and false for Turing and before.
     * @ref LW2080_CTRL_CLK_CLIENT_CLK_VF_POINT_STATUS_PROG_20::vfTupleBase
     */
    LwBool                                     bVfTupleBaseSupported;

    /*!
     * Array of CLIENT_CLK_VF_POINT structures.  Has valid indexes corresponding
     * to the bits set in @ref super.objMask.
     */
    LW2080_CTRL_CLK_CLIENT_CLK_VF_POINT_STATUS vfPoints[LW2080_CTRL_BOARDOBJGRP_E255_MAX_OBJECTS];
} LW2080_CTRL_CLK_CLIENT_CLK_VF_POINTS_STATUS;
typedef struct LW2080_CTRL_CLK_CLIENT_CLK_VF_POINTS_STATUS *PLW2080_CTRL_CLK_CLIENT_CLK_VF_POINTS_STATUS;

/*!
 * LW2080_CTRL_CMD_CLK_CLIENT_CLK_DOMAINS_GET_STATUS
 *
 * This command returns CLIENT_CLK_VF_POINTS dynamic state.  This is dynamic
 * state of the system over which the client has no direct control.
 *
 * See @ref LW2080_CTRL_CLK_CLIENT_CLK_VF_POINTS_STATUS for documentation on the
 * parameters.
 *
 * Return values are specified per @ref BoardObjGrpGetStatus.
 */
#define LW2080_CTRL_CMD_CLK_CLIENT_CLK_VF_POINTS_GET_STATUS (0x2080102b) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_CLK_INTERFACE_ID << 8) | LW2080_CTRL_CLK_CLIENT_CLK_VF_POINTS_STATUS_MESSAGE_ID" */

/*!
 * Structure representing the client controllable state of the
 * CLIENT_CLK_VF_POINT_PROG class.
 */
typedef struct LW2080_CTRL_CLK_CLIENT_CLK_VF_POINT_CONTROL_PROG {
    /*!
     * Frequency offset to apply to this CLIENT_CLK_VF_POINT.
     */
    LW2080_CTRL_CLK_FREQ_DELTA freqDelta;
} LW2080_CTRL_CLK_CLIENT_CLK_VF_POINT_CONTROL_PROG;
typedef struct LW2080_CTRL_CLK_CLIENT_CLK_VF_POINT_CONTROL_PROG *PLW2080_CTRL_CLK_CLIENT_CLK_VF_POINT_CONTROL_PROG;

/*!
 * Structure representing the client controllable state of the
 * CLIENT_CLK_VF_POINT_PROG_20 class.
 */
typedef struct LW2080_CTRL_CLK_CLIENT_CLK_VF_POINT_CONTROL_PROG_20 {
    /*!
     * LW2080_CTRL_CLK_CLIENT_CLK_VF_POINT_CONTROL_PROG super class. Must always
     * be first object in structure
     */
    LW2080_CTRL_CLK_CLIENT_CLK_VF_POINT_CONTROL_PROG super;
} LW2080_CTRL_CLK_CLIENT_CLK_VF_POINT_CONTROL_PROG_20;
typedef struct LW2080_CTRL_CLK_CLIENT_CLK_VF_POINT_CONTROL_PROG_20 *PLW2080_CTRL_CLK_CLIENT_CLK_VF_POINT_CONTROL_PROG_20;

/*!
 * Structure representing the client controllable state of the
 * CLIENT_CLK_VF_POINT_PROG_FREQ class
 */
typedef struct LW2080_CTRL_CLK_CLIENT_CLK_VF_POINT_CONTROL_PROG_20_FREQ {
    /*!
     * LW2080_CTRL_CLK_CLIENT_CLK_VF_POINT_CONTROL_PROG_20 super class. Must always
     * be first object in structure
     */
    LW2080_CTRL_CLK_CLIENT_CLK_VF_POINT_CONTROL_PROG_20 super;

    /*!
     * Voltage offset to apply to this CLIENT_CLK_VF_POINT
     */
    LwS32                                               voltDeltauV;
} LW2080_CTRL_CLK_CLIENT_CLK_VF_POINT_CONTROL_PROG_20_FREQ;
typedef struct LW2080_CTRL_CLK_CLIENT_CLK_VF_POINT_CONTROL_PROG_20_FREQ *PLW2080_CTRL_CLK_CLIENT_CLK_VF_POINT_CONTROL_PROG_20_FREQ;

/*!
 * Structure representing the client controllable state of the
 * CLIENT_CLK_VF_POINT_PROG_VOLT class
 */
typedef struct LW2080_CTRL_CLK_CLIENT_CLK_VF_POINT_CONTROL_PROG_20_VOLT {
    /*!
     * LW2080_CTRL_CLK_CLIENT_CLK_VF_POINT_CONTROL_PROG_20 super class. Must always
     * be first object in structure
     */
    LW2080_CTRL_CLK_CLIENT_CLK_VF_POINT_CONTROL_PROG_20 super;
    /*!
     * CPM max Freq Offset override value
     */
    LwU16                                               cpmMaxFreqOffsetOverrideMHz;
} LW2080_CTRL_CLK_CLIENT_CLK_VF_POINT_CONTROL_PROG_20_VOLT;
typedef struct LW2080_CTRL_CLK_CLIENT_CLK_VF_POINT_CONTROL_PROG_20_VOLT *PLW2080_CTRL_CLK_CLIENT_CLK_VF_POINT_CONTROL_PROG_20_VOLT;

/*!
 * Union of CLIENT_CLK_VF_POINT_TYPE-specific client controllable data.
 */


/*!
 * Structure describing CLIENT_CLK_VF_POINT dynamic state.  Implements the
 * BOARDOBJ model/interface.
 */
typedef struct LW2080_CTRL_CLK_CLIENT_CLK_VF_POINT_CONTROL {
    /*!
     * LW2080_CTRL_BOARDOBJ super class.  Must always be first object in
     * structure.
     */
    LW2080_CTRL_BOARDOBJ super;
    /*!
     * XAPI does not support using an element within another structure as a
     * discriminant.  Placing this redundant type value here until that design
     * constraint can be fixed.
     */
    LwU8                 type;

    /*!
     * Type-specific data union.
     */
    union {
        LW2080_CTRL_CLK_CLIENT_CLK_VF_POINT_CONTROL_PROG         prog;
        LW2080_CTRL_CLK_CLIENT_CLK_VF_POINT_CONTROL_PROG_20      prog20;
        LW2080_CTRL_CLK_CLIENT_CLK_VF_POINT_CONTROL_PROG_20_FREQ prog20Freq;
        LW2080_CTRL_CLK_CLIENT_CLK_VF_POINT_CONTROL_PROG_20_VOLT prog20Volt;
    } data;
} LW2080_CTRL_CLK_CLIENT_CLK_VF_POINT_CONTROL;
typedef struct LW2080_CTRL_CLK_CLIENT_CLK_VF_POINT_CONTROL *PLW2080_CTRL_CLK_CLIENT_CLK_VF_POINT_CONTROL;

/*!
 * Structure describing CLIENT_CLK_VF_POINTS static information/POR.  Implements
 * the BOARDOBJGRP model/interface.
 */
typedef struct LW2080_CTRL_CLK_CLIENT_CLK_VF_POINTS_CONTROL {
    /*!
     * LW2080_CTRL_BOARDOBJGRP_E32 super class.  Must always be first object in
     * structure.
     */
    LW2080_CTRL_BOARDOBJGRP_E255                super;

    /*!
     * Array of CLIENT_CLK_VF_POINT structures.  Has valid indexes corresponding
     * to the bits set in @ref super.objMask.
     */
    LW2080_CTRL_CLK_CLIENT_CLK_VF_POINT_CONTROL vfPoints[LW2080_CTRL_BOARDOBJGRP_E255_MAX_OBJECTS];
} LW2080_CTRL_CLK_CLIENT_CLK_VF_POINTS_CONTROL;
typedef struct LW2080_CTRL_CLK_CLIENT_CLK_VF_POINTS_CONTROL *PLW2080_CTRL_CLK_CLIENT_CLK_VF_POINTS_CONTROL;

/*!
 * LW2080_CTRL_CMD_CLK_CLIENT_CLK_DOMAINS_GET_CONTROL
 *
 * This command returns CLIENT_CLK_VF_POINTS control values.  These are values
 * which the client can directly control.  These values will not change except
 * via calls to this interface.
 *
 * @note This interface is expected to be used in tandem with @ref
 * LW2080_CTRL_CMD_CLK_CLIENT_CLK_DOMAINS_SET_CONTROL in a "Read-Modify-Write"
 * manner.
 *
 * See @ref LW2080_CTRL_CLK_CLIENT_CLK_VF_POINTS_CONTROL for documentation on the
 * parameters.
 *
 * Return values are specified per @ref BoardObjGrpGetControl.
 */
#define LW2080_CTRL_CMD_CLK_CLIENT_CLK_VF_POINTS_GET_CONTROL (0x2080102c) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_CLK_INTERFACE_ID << 8) | 0x2C" */

/*!
 * LW2080_CTRL_CMD_CLK_CLIENT_CLK_DOMAINS_SET_CONTROL
 *
 * This command sets CLIENT_CLK_VF_POINTS control values.  These are values
 * which the client can directly control.  These values will not change except
 * via calls to this interface.
 *
 * @note This interface is expected to be used in tandem with @ref
 * LW2080_CTRL_CMD_CLK_CLIENT_CLK_DOMAINS_GET_CONTROL in a "Read-Modify-Write"
 * manner.
 *
 * See @ref LW2080_CTRL_CLK_CLIENT_CLK_VF_POINTS_CONTROL for documentation on the
 * parameters.
 *
 * Return values are specified per @ref BoardObjGrpSetControl.
 */
#define LW2080_CTRL_CMD_CLK_CLIENT_CLK_VF_POINTS_SET_CONTROL (0x2080102d) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_CLK_INTERFACE_ID << 8) | 0x2D" */

/*!
 * Parameter structure for @ref
 * LW2080_CTRL_CLK_CLK_DOMAIN_3X_PROG_RPC_DATA
 */
typedef struct LW2080_CTRL_CLK_CLK_DOMAIN_3X_FREQ_TO_VOLT_INFO {
    /*!
     * [in] Voltage Domain @ref LW2080_CTRL_VOLT_VOLT_DOMAIN_<xyz>
     * corresponding to the VOLT_RAIL.
     */
    LwU8  voltDomain;
    /*!
     * [in] CLK objects' VOLTAGE_TYPE.
     */
    LwU8  voltageType;
    /*!
     * [in] Frequency point for which the corresponding voltage point will be retrieved.
     */
    LwU16 clkFreqMHz;
    /*!
     * [out] The output voltage for the input frequency.
     */
    LwU32 voltageuV;
} LW2080_CTRL_CLK_CLK_DOMAIN_3X_FREQ_TO_VOLT_INFO;
typedef struct LW2080_CTRL_CLK_CLK_DOMAIN_3X_FREQ_TO_VOLT_INFO *PLW2080_CTRL_CLK_CLK_DOMAIN_3X_FREQ_TO_VOLT_INFO;

/*!
 * Parameter structure for @ref
 * LW2080_CTRL_CLK_CLK_DOMAIN_3X_PROG_RPC_DATA
 */
typedef struct LW2080_CTRL_CLK_CLK_DOMAIN_3X_VOLT_TO_FREQ_INFO {
    /*!
     * [in] Voltage Domain @ref LW2080_CTRL_VOLT_VOLT_DOMAIN_<xyz>
     * corresponding to the VOLT_RAIL.
     */
    LwU8  voltDomain;
    /*!
     * [in] CLK objects' VOLTAGE_TYPE.
     */
    LwU8  voltageType;
    /*!
     * [in]The voltage point for which the corresponding frequency point will be retrieved.
     */
    LwU32 voltageuV;
    /*!
     * [out] The output frequency for the input voltage.
     */
    LwU16 clkFreqMHz;
} LW2080_CTRL_CLK_CLK_DOMAIN_3X_VOLT_TO_FREQ_INFO;
typedef struct LW2080_CTRL_CLK_CLK_DOMAIN_3X_VOLT_TO_FREQ_INFO *PLW2080_CTRL_CLK_CLK_DOMAIN_3X_VOLT_TO_FREQ_INFO;

/*!
 * Parameter structure for @ref
 * LW2080_CTRL_CLK_CLK_DOMAIN_3X_PROG_RPC_DATA
 */
typedef struct LW2080_CTRL_CLK_CLK_DOMAIN_3X_FREQ_QUANTIZE_INFO {
    /*!
     * [in/out] Frequency point for which the quntized point will be returned.
     */
    LwU16  clkFreqMHz;
    /*!
     * [in] Boolean value to check if the frequency value is to be adjusted with the
     *      applied frequency delta.
     */
    LwBool bReqFreqDeltaAdj;
    /*!
     * [in] Boolean value indicating whether the frequency should be quantized via a floor function.
     */
    LwBool bFloor;
} LW2080_CTRL_CLK_CLK_DOMAIN_3X_FREQ_QUANTIZE_INFO;
typedef struct LW2080_CTRL_CLK_CLK_DOMAIN_3X_FREQ_QUANTIZE_INFO *PLW2080_CTRL_CLK_CLK_DOMAIN_3X_FREQ_QUANTIZE_INFO;

/*!
 * Parameter structure for @ref
 * LW2080_CTRL_CLK_CLK_DOMAIN_3X_PROG_RPC_DATA
 */
typedef struct LW2080_CTRL_CLK_CLK_DOMAIN_3X_CLK_SOURCE_INFO {
    /*!
     * [in] Input frequency in MHz
     */
    LwU16 clkFreqMHz;
    /*!
     * [out] The output CLK_PROG source type
     */
    LwU8  clkSourceType;
} LW2080_CTRL_CLK_CLK_DOMAIN_3X_CLK_SOURCE_INFO;
typedef struct LW2080_CTRL_CLK_CLK_DOMAIN_3X_CLK_SOURCE_INFO *PLW2080_CTRL_CLK_CLK_DOMAIN_3X_CLK_SOURCE_INFO;

/*!
 * RPC Ids for CLK_DOMAIN_3X_PROG class
 */
#define LW2080_CTRL_CLK_DOMAIN_3X_PROG_RPC_ID_FREQ_TO_VOLT   (0x0U)
#define LW2080_CTRL_CLK_DOMAIN_3X_PROG_RPC_ID_VOLT_TO_FREQ   (0x1U)
#define LW2080_CTRL_CLK_DOMAIN_3X_PROG_RPC_ID_FREQ_QUANTIZE  (0x2U)
#define LW2080_CTRL_CLK_DOMAIN_3X_PROG_RPC_ID_GET_CLK_SOURCE (0x3U)

#define LW2080_CTRL_CLK_DOMAIN_40_PROG_RPC_ID_FREQ_TO_VOLT   LW2080_CTRL_CLK_DOMAIN_3X_PROG_RPC_ID_FREQ_TO_VOLT
#define LW2080_CTRL_CLK_DOMAIN_40_PROG_RPC_ID_VOLT_TO_FREQ   LW2080_CTRL_CLK_DOMAIN_3X_PROG_RPC_ID_VOLT_TO_FREQ
#define LW2080_CTRL_CLK_DOMAIN_40_PROG_RPC_ID_FREQ_QUANTIZE  LW2080_CTRL_CLK_DOMAIN_3X_PROG_RPC_ID_FREQ_QUANTIZE
#define LW2080_CTRL_CLK_DOMAIN_40_PROG_RPC_ID_GET_CLK_SOURCE LW2080_CTRL_CLK_DOMAIN_3X_PROG_RPC_ID_GET_CLK_SOURCE

/*!
 * Type-specific union for @ref
 * LW2080_CTRL_CLK_CLK_DOMAIN_3X_PROG_RPC_INFO
 * Discriminated by DOMAIN_3X_PROG_RPC_INFO::type
 */


/*!
 * Parameter structure for @ref
 * LW2080_CTRL_CLK_CLK_DOMAIN_3X_PROG_INFO_DATA
 */
typedef struct LW2080_CTRL_CLK_CLK_DOMAIN_3X_PROG_RPC_INFO {
    /*!
     * Opcode Id type for CLK_DOMAIN_3X_PROG class
     * @ref LW2080_CTRL_CLK_DOMAIN_3x_PROG_RPC_ID_<XYZ>
     */
    LwU8  type;
    /*!
     * RPC's actual exec. time (measured on PMU side).
     */
    LwU32 execTimePmuns;
    /*!
     * RPC Type specific union data
     */
    union {
        LW2080_CTRL_CLK_CLK_DOMAIN_3X_FREQ_TO_VOLT_INFO  freqToVoltInfo;
        LW2080_CTRL_CLK_CLK_DOMAIN_3X_VOLT_TO_FREQ_INFO  voltToFreqInfo;
        LW2080_CTRL_CLK_CLK_DOMAIN_3X_FREQ_QUANTIZE_INFO freqQuantizeInfo;
        LW2080_CTRL_CLK_CLK_DOMAIN_3X_CLK_SOURCE_INFO    clkSourceInfo;
    } rpcData;
} LW2080_CTRL_CLK_CLK_DOMAIN_3X_PROG_RPC_INFO;
typedef struct LW2080_CTRL_CLK_CLK_DOMAIN_3X_PROG_RPC_INFO *PLW2080_CTRL_CLK_CLK_DOMAIN_3X_PROG_RPC_INFO;

/*!
 * Type-specific for @ref
 * LW2080_CTRL_CLK_CLK_DOMAIN_RPC
 * Discriminated by CLK_DOMAIN_RPC::classType
 */


/*!
 * Parameter structure for @ref
 * LW2080_CTRL_CMD_CLK_CLK_DOMAIN_RPC
 */
#define LW2080_CTRL_CLK_CLK_DOMAIN_RPC_MESSAGE_ID (0x32U)

typedef struct LW2080_CTRL_CLK_CLK_DOMAIN_RPC {
    /*!
     * [in] Index of the CLK_DOMAIN object for which to get the Voltage point.
     */
    LwU32 clkDomainIdx;
    /*!
     * [in] CLK_DOMAIN class type
     */
    LwU8  classType;
    /*!
     * Domain class type specific union data
     */
    union {
        LW2080_CTRL_CLK_CLK_DOMAIN_3X_PROG_RPC_INFO v3xRpcInfo;
    } infoData;
} LW2080_CTRL_CLK_CLK_DOMAIN_RPC;
typedef struct LW2080_CTRL_CLK_CLK_DOMAIN_RPC *PLW2080_CTRL_CLK_CLK_DOMAIN_RPC;

/*!
 *  LW2080_CTRL_CMD_CLK_CLK_DOMAIN_RPC
 *
 * This command exposes the RPCs for various classes:
 *  @ref LW2080_CTRL_CLK_CLK_DOMAIN_TYPE_<xyz>
 *
 * The client has to select the Clock Domain Class Type and then
 * the RPC type for that class
 *
 * See @ref LW2080_CTRL_CLK_CLK_DOMAIN_RPC for documentation on the
 * parameters.
 */
#define LW2080_CTRL_CMD_CLK_CLK_DOMAIN_RPC (0x20801032) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_CLK_INTERFACE_ID << 8) | LW2080_CTRL_CLK_CLK_DOMAIN_RPC_MESSAGE_ID" */

/*!
 * Structure describing an individual PLL Device
 */
typedef struct LW2080_CTRL_CLK_PLL_DEVICE_INFO {
    /*!
     * Super class representing the board object
     */
    LW2080_CTRL_BOARDOBJ super;
    /*!
     * PLL ID @refLW2080_CTRL_CLK_SOURCE_<xyz>
     */
    LwU8                 id;
    /*!
     * Minimum Reference Clock Frequency in MHz for this PLL Device
     */
    LwU16                MinRef;
    /*!
     * Maximum Reference Clock Frequency in MHz for this PLL Device
     */
    LwU16                MaxRef;
    /*!
     * Minimum VCO Frequency in MHz for this PLL Device
     */
    LwU16                MilwCO;
    /*!
     * Maximum VCO Frequency in MHz for this PLL Device
     */
    LwU16                MaxVCO;
    /*!
     * Minimum Update Rate in MHz for this PLL Device
     */
    LwU16                MinUpdate;
    /*!
     * Maximum Update Rate in MHz for this PLL Device
     */
    LwU16                MaxUpdate;
    /*!
     * Minimum Reference Clock Divider(M) Coefficient for this PLL Device
     */
    LwU8                 MinM;
    /*!
     * Maximum Reference Clock Divider(M) Coefficient for this PLL Device
     */
    LwU8                 MaxM;
    /*!
     * Minimum VCO Feedback Divider(N) Coefficient for this PLL Device
     */
    LwU8                 MinN;
    /*!
     * Maximum VCO Feedback Divider(N) Coefficient for this PLL Device
     */
    LwU8                 MaxN;
    /*!
     * Minimum Linear Post Divider(PL) Coefficient for this PLL Device
     */
    LwU8                 MinPl;
    /*!
     * Maximum Linear Post Divider(PL) Coefficient for this PLL Device
     */
    LwU8                 MaxPl;
} LW2080_CTRL_CLK_PLL_DEVICE_INFO;
typedef struct LW2080_CTRL_CLK_PLL_DEVICE_INFO *PLW2080_CTRL_CLK_PLL_DEVICE_INFO;

/*!
 * Structure containing the global properties and boardobjgrp for PLL devices
 */
#define LW2080_CTRL_CLK_PLL_DEVICES_GET_INFO_PARAMS_MESSAGE_ID (0x2EU)

typedef struct LW2080_CTRL_CLK_PLL_DEVICES_GET_INFO_PARAMS {
    /*!
     * This is the BOARDOBJGRP_E32 super class.  Must always be the first
     * element in the structure.
     */
    LW2080_CTRL_BOARDOBJGRP_E32     super;
    /*!
     * Array of PLL_DEVICE structures. Has valid indexes corresponding
     * to the bits set in @ref super.objMask.
     */
    LW2080_CTRL_CLK_PLL_DEVICE_INFO pllDevices[LW2080_CTRL_BOARDOBJGRP_E32_MAX_OBJECTS];
} LW2080_CTRL_CLK_PLL_DEVICES_GET_INFO_PARAMS;
typedef struct LW2080_CTRL_CLK_PLL_DEVICES_GET_INFO_PARAMS *PLW2080_CTRL_CLK_PLL_DEVICES_GET_INFO_PARAMS;

/*!
 * LW2080_CTRL_CMD_CLK_PLL_DEVICES_GET_INFO
 *
 * This command returns PLL_DEVICES static object information/POR
 * as populated by the RM from the VBIOS tables.
 *
 * The PLL_DEVICE objects are indexed in the order by which the RM
 * allocates them.
 *
 * See @ref LW2080_CTRL_CLK_PLL_DEVICES_GET_INFO_PARAMS for the
 * documentation on the parameters.
 *
 * Return values are specified per @ref BoardObjGrpGetInfo.
 */
#define LW2080_CTRL_CMD_CLK_PLL_DEVICES_GET_INFO (0x2080102e) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_CLK_INTERFACE_ID << 8) | LW2080_CTRL_CLK_PLL_DEVICES_GET_INFO_PARAMS_MESSAGE_ID" */


/*!
 * Enumeration of FREQ_DOMAIN per @ref LW2080_CTRL_CLK_FREQ_DOMAIN_INFO::type.
 * As of Clocks 3.1, there is only one type.
 */
#define LW2080_CTRL_CLK_FREQ_DOMAIN_SCHEMA_CLK3  0x03U

/*!
 * @brief       Frequency Domains without Static Information/POR
 */
typedef struct LW2080_CTRL_CLK_FREQ_DOMAIN_INFO_EMPTY {
    /*!
     * Reserved for the future.  Do not use.
     */
    LwU32 reserved;
} LW2080_CTRL_CLK_FREQ_DOMAIN_INFO_EMPTY;
typedef struct LW2080_CTRL_CLK_FREQ_DOMAIN_INFO_EMPTY *PLW2080_CTRL_CLK_FREQ_DOMAIN_INFO_EMPTY;


/*!
 * @brief       Static Information/POR Common for Specific Frequency Domain Types
 */
typedef LW2080_CTRL_CLK_FREQ_DOMAIN_INFO_EMPTY LW2080_CTRL_CLK_FREQ_DOMAIN_INFO_CLK3;

/*!
 * @brief       Union of Static Information/POR for Frequency Domains
 * @details     Discriminated by @ref LW2080_CTRL_CLK_FREQ_DOMAIN_INFO.type.
 */



/*!
 * @brief       Information for Frequency Domains
 * @extends     LW2080_CTRL_BOARDOBJ
 */
typedef struct LW2080_CTRL_CLK_FREQ_DOMAIN_INFO {
    /*!
     * LW2080_CTRL_BOARDOBJ super class.  Must always be first object in
     * structure.
     */
    LW2080_CTRL_BOARDOBJ super;

    /*!
     * @brief       Clock domain per LW2080_CTRL_CLK_DOMAIN_<xyz>.
     * @see         LW2080_CTRL_CLK_DOMAIN
     */
    LwU32                clkDomain;

    /*!
     * XAPI discriminant per LW2080_CTRL_CLK_FREQ_DOMAIN_SCHEMA_xxx.
     * Each type is a different schema for the clock frequency domain.
     * This is same as BOARDOBJ::type.
     */
    LwU8                 schema;

    /*!
     * Type-specific data union.
     */
    union {
        LW2080_CTRL_CLK_FREQ_DOMAIN_INFO_CLK3 clk3;
    } data;
} LW2080_CTRL_CLK_FREQ_DOMAIN_INFO;
typedef struct LW2080_CTRL_CLK_FREQ_DOMAIN_INFO *PLW2080_CTRL_CLK_FREQ_DOMAIN_INFO;


/*!
 * FREQ_DOMAIN_GRP static information/POR.
 * Implements the BOARDOBJGRP model/interface.
 */
typedef struct LW2080_CTRL_CLK_FREQ_DOMAIN_GRP_GET_INFO_PARAMS {
    /*!
     * LW2080_CTRL_BOARDOBJGRP super class. Must always be first object in
     * structure.
     */
    LW2080_CTRL_BOARDOBJGRP_E32      super;

    /*!
     * Reserved for the future.  Do not use.
     */
    LwU8                             reserved[32];

    LW2080_CTRL_CLK_FREQ_DOMAIN_INFO freqDomain[LW2080_CTRL_BOARDOBJGRP_E32_MAX_OBJECTS];
} LW2080_CTRL_CLK_FREQ_DOMAIN_GRP_GET_INFO_PARAMS;
typedef struct LW2080_CTRL_CLK_FREQ_DOMAIN_GRP_GET_INFO_PARAMS *PLW2080_CTRL_CLK_FREQ_DOMAIN_GRP_GET_INFO_PARAMS;


/*!
 * LW2080_CTRL_CMD_CLK_FREQ_DOMAIN_GRP_GET_INFO
 *
 * This command returns FREQ_DOMAIN_GRP static object information/POR
 * as populated by the RM from the frequency controller VBIOS table.
 *
 * The FREQ_DOMAIN objects are indexed in the order by which the RM
 * allocates them.
 *
 * See @ref LW2080_CTRL_CLK_FREQ_DOMAIN_GRP_GET_INFO_PARAMS for the
 * documentation on the parameters.
 *
 * Return values are specified per @ref BoardObjGrpGetInfo.
 */
#define LW2080_CTRL_CMD_CLK_FREQ_DOMAIN_GRP_GET_INFO (0x20801030) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_CLK_INTERFACE_ID << 8) | 0x30" */


// ============ Signal & Signal Path ===========================================

/*!
 * @brief       Number of bits for each signal path node
 * @see         LW2080_CTRL_CMD_CLK_SIGNAL_PATH
 *
 * @details     This data type is used to index the various phase arrays and
 *              to contain phase counts.
 */
#define LW2080_CTRL_CMD_CLK_SIGNAL_NODE_WIDTH        4U


/*!
 * @brief       Stack of nodes representing the signal path
 * @see         LW2080_CTRL_CMD_CLK_SIGNAL_PATH_EMPTY
 * @see         LW2080_CTRL_CMD_CLK_SIGNAL
 *
 * @detail      The special value LW2080_CTRL_CMD_CLK_SIGNAL_PATH_EMPTY represents
 *              an empty stack.
 *
 *              If the number of bits increases, it may be useful to change the
 *              order of members within LW2080_CTRL_CMD_CLK_SIGNAL.  See comments there.
 */
typedef LwU32 LW2080_CTRL_CMD_CLK_SIGNAL_PATH;


/*!
 * @brief       Value to indicate an empty stack
 * @see         LW2080_CTRL_CMD_CLK_SIGNAL_PATH
 *
 * @detail      The special value LW2080_CTRL_CMD_CLK_SIGNAL_PATH_EMPTY represents
 *              an empty stack, which means that all nodes are indeterminate.
 */
#define LW2080_CTRL_CMD_CLK_SIGNAL_PATH_EMPTY LW_U32_MAX


/*!
 * @brief       Characteristics of a Clock Signal
 * @see         LW2080_CTRL_CMD_CLK_FREQ_TARGET_SIGNAL
 * @see         LW2080_CTRL_CMD_CLK_FREQ_DOMAIN_BOARDOBJ_GET_STATUS::output
 *
 * @details     This struct can be used for both output and target clock signals.
 *
 *              As an output, it is part of LW2080_CTRL_CMD_CLK_FREQ_DOMAIN_BOARDOBJ_GET_STATUS
 *              as well as parameters for 'clkRead' and 'clkConfig'.
 *
 *              As a target signal, it is part of LW2080_CTRL_CMD_CLK_FREQ_TARGET_SIGNAL.
 *
 * @todo        Consider using the Microsoft "pack" pragma and/or the gcc
 *              "packed" variable attribute to reduce stack pressure.
 *              In particular, the Windows x64 calling convention requires
 *              32 bytes of stack space for this class, but 10 or 12 is
 *              reasonable.  Care must then be taken, however, to avoid
 *              EXCEPTION_DATATYPE_MISALIGNMENT.
 *
 * @see         http://msdn.microsoft.com/en-us/library/aa290049%28v=vs.71%29.aspx
 * @see         http://gcc.gnu.org/onlinedocs/gcc-4.6.2/gcc/Variable-Attributes.html
 */
typedef struct LW2080_CTRL_CMD_CLK_SIGNAL {
    /*!
     * @brief       Output/Target Frequency in KHz
     * @see         ClkTargetSignal::finalFreqKHz
     *
     * @details     As an output signal, this member represents the frequency, etc.
     *              of the frequency domains and/or frequency source.
     *
     *              As a target signal, 'clkConfig' computes the configuration closest
     *              to this frequency.  The value may differ from 'finalFreqKHz' to
     *              indicate a target for an intermediate phase.  (That is, at least
     *              one more phase is required.)
     */
    LwU32                           freqKHz;

    /*!
     * @brief       Precise Signal Path
     *
     * @details     LW2080_CTRL_CMD_CLK_SIGNAL_PATH_EMPTY as a target means 'don't-care'.
     *
     *              'path' and 'source' indicate essentially the same information,
     *              how the multiplexers are set along the signal path.  However,
     *              'path' is detailed.  It's a stack with each nibble indicating
     *              the setting of a specific mux.  In contrast, 'source' summarizes
     *              this information.  'path' => 'source' is surjective.
     */
    LW2080_CTRL_CMD_CLK_SIGNAL_PATH path;

    /*!
     * @brief       Fractional Divide Indicator
     * @see         ClkLdivUnit
     *
     * @details     LW_TRISTATE_FALSE           Fractional divide is (should be) disabled
     *              LW_TRISTATE_TRUE            Fractional divide (should be) enabled
     *              LW_TRISTATE_INDETERMINATE   Can't tell (don't care) if fractional
     *                                          divide is enabled or disabled
     *
     *              As an output signal, this flag indicates if fractional divide was
     *              applied.
     *
     *              For target signals, it can be used to enable or disable fractional
     *              divide.
     *
     *              Fractional divide, also known as 'fracdiv', is a function of the
     *              linear dividers.
     *
     *              As of Maxwell, we don't apply fractional divide on more than
     *              one linear divider in the same signal path.  As such, once
     *              applied by 'clkConfig_LdivUnit', this flag is set to _FALSE
     *              in the target passed to its input.
     */
    LwU8                            fracdiv;

    /*!
     * @brief       NAFLL regime ID @ref LW2080_CTRL_CLK_NAFLL_REGIME_ID_<XYZ>
     *
     * @details     This indicates the target regime ID for each NAFLL of the clock
     *              domain that should be set for the given target frequency change
     */
    LwU8                            regimeId;

    /*!
     * @brief       Signal path per @ref LW2080_CTRL_CLK_PROG_1X_SOURCE_<XYZ>
     * @see         path
     * @see         LW2080_CTRL_CLK_PROG_1X_SOURCE_ILWALID
     *
     * @details     In ClkFreqDomain::clkConfig, this field is used to set 'path'
     *              only if 'path' is _EMPTY.  It is othersize ignored.
     */
    LwU8                            source;

    /*!
     * @brief       Cache DVCO Min frequency in MHz
     * @see         path
     * @see         LW2080_CTRL_CLK_PROG_1X_SOURCE_ILWALID
     *
     * @details     This indicates the DVCO min frequency of the NAFLL DVCO
     *              based on target voltage and other process / temperature
     *              condition at the time of clock switch request injection.
     */
    LwU16                           dvcoMinFreqMHz;
} LW2080_CTRL_CMD_CLK_SIGNAL;
typedef struct LW2080_CTRL_CMD_CLK_SIGNAL *PLW2080_CTRL_CMD_CLK_SIGNAL;


// ============ Frequency Sources ==============================================

/*!
 * @brief       Structure describing the dynamic state of a DVCO
 * @version     PASCAL and later
 */
typedef struct LW2080_CTRL_CLK_FREQ_SOURCE_STATUS_NAFLL_DVCO {

    /*!
     * Min Frequency MHz
     */
    LwU16  minFreqMHz;

    /*!
     * Flag to indicate if DVCO min frequency reached
     *
     * @note Unused from Ampere and onwards. Its usage is being replaced with
     *       LW2080_CTRL_CLK_NAFLL_REGIME_ID_FFR_BELOW_DVCO_MIN.
     */
    LwBool bMinReached;
} LW2080_CTRL_CLK_FREQ_SOURCE_STATUS_NAFLL_DVCO;
typedef struct LW2080_CTRL_CLK_FREQ_SOURCE_STATUS_NAFLL_DVCO *PLW2080_CTRL_CLK_FREQ_SOURCE_STATUS_NAFLL_DVCO;

/*!
 * @brief       Structure containing the dynamic state of regime logic.
 * @version     PASCAL and later
 */
typedef struct LW2080_CTRL_CLK_FREQ_SOURCE_STATUS_NAFLL_REGIME {
    /*!
     * Current regime ID @ref LW2080_CTRL_PERF_NAFLL_REGIME_ID_<xyz> of this NAFLL device
     */
    LwU8  lwrrentRegimeId;

    /*!
     * Target regime ID @ref LW2080_CTRL_PERF_NAFLL_REGIME_ID_<xyz> for this NAFLL device
     */
    LwU8  targetRegimeId;

    /*!
     * Current frequency in MHz of this NAFLL device.
     */
    LwU16 lwrrentFreqMHz;

    /*!
     * Target frequency in MHz for this NAFLL device.
     */
    LwU16 targetFreqMHz;

    /*!
     * Current target frequency offset in MHz.
     *
     * @note Unused from Ampere and onwards.
     */
    LwU16 offsetFreqMHz;
} LW2080_CTRL_CLK_FREQ_SOURCE_STATUS_NAFLL_REGIME;
typedef struct LW2080_CTRL_CLK_FREQ_SOURCE_STATUS_NAFLL_REGIME *PLW2080_CTRL_CLK_FREQ_SOURCE_STATUS_NAFLL_REGIME;


/*!
 * @brief       Nafll status
 * @version     Clocks 3.0
 */
typedef struct LW2080_CTRL_CLK_FREQ_SOURCE_STATUS_NAFLL {
    /*!
     * @brief       Indicator for Engaging / Disengaging the PLDIV at the target
     *              frequency for a given NAFLL.
     *
     * @details     LW_TRISTATE_FALSE
     *                  Disengage pldiv PRIOR to changing the target freq and regime.
     *              LW_TRISTATE_TRUE
     *                  Engage pldiv AFTER the target frequency and regime is programmed
     *              LW_TRISTATE_INDETERMINATE
     *                  No change is required.
     *
     * @note        Unused from Ampere and onwards. Its usage is being replaced with
     *              LW2080_CTRL_CLK_NAFLL_REGIME_ID_FFR_BELOW_DVCO_MIN.
     */
    LwU8                                            pldivEngage;

    /*!
     * PL-Divider value for this NAFLL device
     */
    LwU8                                            pldiv;

    /*!
     * @ref LW2080_CTRL_CLK_FREQ_SOURCE_STATUS_NAFLL_DVCO
     */
    LW2080_CTRL_CLK_FREQ_SOURCE_STATUS_NAFLL_DVCO   dvco;

    /*!
     * @ref LW2080_CTRL_CLK_FREQ_SOURCE_STATUS_NAFLL_REGIME
     */
    LW2080_CTRL_CLK_FREQ_SOURCE_STATUS_NAFLL_REGIME regime;
} LW2080_CTRL_CLK_FREQ_SOURCE_STATUS_NAFLL;
typedef struct LW2080_CTRL_CLK_FREQ_SOURCE_STATUS_NAFLL *PLW2080_CTRL_CLK_FREQ_SOURCE_STATUS_NAFLL;


// ============ Frequency Domains ==============================================

/*!
 * @brief       Status shared by all Frequency Domains
 * @see         ClkFreqDomain
 */
typedef struct LW2080_CTRL_CLK_FREQ_DOMAIN_STATUS_BASE {
    /*!
     * @brief       Domain API ID
     * @see         LW2080_CTRL_CLK_DOMAIN_<xyz>
     *
     * @ilwariant   Exactly one bit of this field must be set.
     */
    LwU32                      clkDomain;

    /*!
     * @brief       The output frequency, signal path, and other clock signal details.
     *
     * @details     These data are a cache of the current hardware state.
     *
     *              When 'output.freqKHz' is zero, the cache is in a reset state
     *              and 'clkRead' reads the hardware and updates this member.
     *
     *              Otherwise, 'clkRead' is a no-op and simply returns (except
     *              for volatile domains).
     *
     *              'clkProgram' updates this member when programming is done.
     *              Assuming no error, this means that 'clkRead' is reads hardware
     *              something only after initialization except on volatile domains.
     */
    LW2080_CTRL_CMD_CLK_SIGNAL output;
} LW2080_CTRL_CLK_FREQ_DOMAIN_STATUS_BASE;
typedef struct LW2080_CTRL_CLK_FREQ_DOMAIN_STATUS_BASE *PLW2080_CTRL_CLK_FREQ_DOMAIN_STATUS_BASE;


/*!
 * @brief       Domain Status
 * @version     Clocks 3.0
 * @note        Associated type: LW2080_CTRL_CLK_FREQ_DOMAIN_SCHEMA_CLK3
 */
typedef struct LW2080_CTRL_CLK_FREQ_DOMAIN_STATUS_CLK3 {
    LW2080_CTRL_CLK_FREQ_DOMAIN_STATUS_BASE domain;     // Must be first

} LW2080_CTRL_CLK_FREQ_DOMAIN_STATUS_CLK3;
typedef struct LW2080_CTRL_CLK_FREQ_DOMAIN_STATUS_CLK3 *PLW2080_CTRL_CLK_FREQ_DOMAIN_STATUS_CLK3;


/*!
 * FREQ_DOMAIN type-specific data union.  Discriminated by
 * @ref FREQ_DOMAIN::super.type.
 */



/*!
 * Structure describing FREQ_DOMAIN dynamic information
 * Implements the BOARDOBJ model/interface.
 */
typedef struct LW2080_CTRL_CLK_FREQ_DOMAIN_STATUS {
    /*!
     * LW2080_CTRL_BOARDOBJ super class.  Must always be first object in
     * structure.
     */
    LW2080_CTRL_BOARDOBJ super;

    /*!
     * XAPI does not support using an element within another structure as a
     * discriminant.  Placing this redundant type value here until that design
     * constraint can be fixed.
     */
    LwU8                 type;

    /*!
     * Type-specific data union.
     */
    union {
        LW2080_CTRL_CLK_FREQ_DOMAIN_STATUS_CLK3 clk3;
    } data;
} LW2080_CTRL_CLK_FREQ_DOMAIN_STATUS;
typedef struct LW2080_CTRL_CLK_FREQ_DOMAIN_STATUS *PLW2080_CTRL_CLK_FREQ_DOMAIN_STATUS;


/*!
 * Structure describing FREQ_DOMAIN_GRP static information/POR.
 * Implements the BOARDOBJGRP model/interface.
 */
#define LW2080_CTRL_CLK_FREQ_DOMAIN_GRP_STATUS_PARAMS_MESSAGE_ID (0x61U)

typedef struct LW2080_CTRL_CLK_FREQ_DOMAIN_GRP_STATUS_PARAMS {
    /*!
     * LW2080_CTRL_BOARDOBJGRP super class. Must always be first object in
     * structure.
     */
    LW2080_CTRL_BOARDOBJGRP_E32        super;

    /*!
     * [out] Array of FREQ_DOMAIN structures. Has valid indices
     * corresponding to the bits set in @ref super.objMask.
     */
    LW2080_CTRL_CLK_FREQ_DOMAIN_STATUS freqDomain[LW2080_CTRL_BOARDOBJGRP_E32_MAX_OBJECTS];
} LW2080_CTRL_CLK_FREQ_DOMAIN_GRP_STATUS_PARAMS;
typedef struct LW2080_CTRL_CLK_FREQ_DOMAIN_GRP_STATUS_PARAMS *PLW2080_CTRL_CLK_FREQ_DOMAIN_GRP_STATUS_PARAMS;


/*!
 * LW2080_CTRL_CMD_CLK_FREQ_DOMAIN_GRP_GET_STATUS
 *
 * This command returns FREQ_DOMAIN_GRP dynamic object information from the
 * PMU.
 *
 * See @ref LW2080_CTRL_CMD_CLK_FREQ_DOMAIN_GRP_GET_STATUS for the
 * documentation on the parameters.
 *
 * Return values are specified per @ref BoardObjGrpGetStatus.
 */
#define LW2080_CTRL_CMD_CLK_FREQ_DOMAIN_GRP_GET_STATUS (0x20801061) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_CLK_INTERFACE_ID << 8) | LW2080_CTRL_CLK_FREQ_DOMAIN_GRP_STATUS_PARAMS_MESSAGE_ID" */


/*!
 * LW2080_CTRL_CLK_GET_INFO_V2
 *
 * This command returns clock information for each entry on the specified list
 * of clock info structures.
 *
 *   [in] flags
 *     This field specifies the desired flags for the get clock operation.
 *     This field is lwrrently unused.
 *   [in] clkInfoListSize
 *     This field specifies the number of entries on the caller's
 *     clkInfoList.
 *   [out] clkInfoList
 *     This field specifies a pointer in the caller's address space
 *     to the buffer into which the clock information is to be returned.
 *     This buffer must be at least as big as clkInfoListSize multiplied
 *     by the size of the LW2080_CTRL_CLK_INFO structure.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_OPERATING_SYSTEM
 */
#define LW2080_CTRL_CMD_CLK_GET_INFO_V2                (0x20801062) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_CLK_INTERFACE_ID << 8) | LW2080_CTRL_CLK_GET_INFO_V2_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_CLK_GET_INFO_V2_PARAMS_MESSAGE_ID (0x62U)

typedef struct LW2080_CTRL_CLK_GET_INFO_V2_PARAMS {
    LwU32                flags;
    LwU32                clkInfoListSize;
    LW2080_CTRL_CLK_INFO clkInfoList[LW2080_CTRL_CLK_ARCH_MAX_DOMAINS];
} LW2080_CTRL_CLK_GET_INFO_V2_PARAMS;

/*!
 * LW2080_CTRL_CLK_GET_CLK_PSTATES2_INFO_V2
 *
 * This command returns clock information related to P-states 2.x for each
 * entry on the specified list of clock P-states 2.x info structures. These
 * information do not change after driver initialization.
 *
 *   [in] flags
 *     This field specifies the flags for the operation. It is lwrrently
 *     unused.
 *   [in] clkInfoListSize
 *     This field specifies the number of entries on the caller's clkInfoList.
 *   [out] clkInfoList
 *     This field specifies a pointer in the caller's address space to the
 *     buffer into which the clock P-states 2.x information is to be returned.
 *     This buffer must be at least as big as clkInfoListSize multiplied
 *     by the size of the LW2080_CTRL_CLK_PSTATES2_INFO structure.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW2080_CTRL_CMD_CLK_GET_PSTATES2_INFO_V2 (0x20801063) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_CLK_INTERFACE_ID << 8) | LW2080_CTRL_CLK_GET_PSTATES2_INFO_V2_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_CLK_GET_PSTATES2_INFO_V2_PARAMS_MESSAGE_ID (0x63U)

typedef struct LW2080_CTRL_CLK_GET_PSTATES2_INFO_V2_PARAMS {
    LwU32                         flags;
    LwU32                         clkInfoListSize;
    LW2080_CTRL_CLK_PSTATES2_INFO clkInfoList[LW2080_CTRL_CLK_ARCH_MAX_DOMAINS];
} LW2080_CTRL_CLK_GET_PSTATES2_INFO_V2_PARAMS;

/*!
 * LW2080_CTRL_CMD_CLK_GET_PUBLIC_DOMAIN_INFO_V2
 *
 * This command returns detailed information for each entry on the
 * specified list of public domain info structures.
 *
 *   [in] flags
 *     This field is reserved for future use.
 *   [in] publicDomainInfoListSize
 *     This field specifies the number of entries on the caller's
 *     publicDomainInfoList.
 *   [out] publicDomainInfoList
 *     This field specifies a pointer in the caller's address space
 *     to the buffer into which the clock information is to be returned.
 *     This buffer must be at least as big as publicDomainInfoList multiplied
 *     by the size of the LW2080_CTRL_CLK_PUBLIC_DOMAIN_INFO structure.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW2080_CTRL_CMD_CLK_GET_PUBLIC_DOMAIN_INFO_V2 (0x20801064) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_CLK_INTERFACE_ID << 8) | LW2080_CTRL_CLK_GET_PUBLIC_DOMAIN_INFO_V2_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_CLK_GET_PUBLIC_DOMAIN_INFO_V2_PARAMS_MESSAGE_ID (0x64U)

typedef struct LW2080_CTRL_CLK_GET_PUBLIC_DOMAIN_INFO_V2_PARAMS {
    LwU32                              flags;
    LwU32                              publicDomainInfoListSize;
    LW2080_CTRL_CLK_PUBLIC_DOMAIN_INFO publicDomainInfoList[LW2080_CTRL_CLK_ARCH_MAX_DOMAINS];
} LW2080_CTRL_CLK_GET_PUBLIC_DOMAIN_INFO_V2_PARAMS;

/*
 * LW2080_CTRL_CMD_CLK_EMU_TRAP_REG_SET
 *
 * This command is used to set the EMU trap register for Voltage set.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW2080_CTRL_CMD_CLK_EMU_TRAP_REG_SET (0x20801065) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_CLK_INTERFACE_ID << 8) | LW2080_CTRL_CLK_EMU_TRAP_REG_SET_PARAMS_MESSAGE_ID" */

/*!
 * voltageuV - to set on the EMU
 */
#define LW2080_CTRL_CLK_EMU_TRAP_REG_SET_PARAMS_MESSAGE_ID (0x65U)

typedef struct LW2080_CTRL_CLK_EMU_TRAP_REG_SET_PARAMS {
    /*!
     * voltage to set on the EMU
     */
    LwU32 voltageuV;
} LW2080_CTRL_CLK_EMU_TRAP_REG_SET_PARAMS;

/*!
 * LW2080_CTRL_CLK_SET_INFO_PMU
 *
 * This command programs the specified clock domains using the information
 * contained in the corresponding clock info structures.  Not all domains
 * support the set operation.  Note also that on some GPUs it's required
 * that a group of related domains be set in one operation.  The absence
 * of a domain-specific clock info structure in such cases will lead to an
 * invalid command error.
 *
 *   [in] domainList
 *     This field contains the information about the domains to be programmed
 *     and respective target Freq.
 *   [in] railList
 *     This field contains the information about the rails to be programmed
 *     and respective target voltage.
 *
 * TODO: Consider making this chip specific union as this is HAL.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_ILWALID_COMMAND
 *   LW_ERR_OPERATING_SYSTEM
 */
#define LW2080_CTRL_CLK_SET_INFOS_PMU_PARAMS_MESSAGE_ID (0x66U)

typedef struct LW2080_CTRL_CLK_SET_INFOS_PMU_PARAMS {
    LW2080_CTRL_CLK_CLK_DOMAIN_LIST domainList;
    LW2080_CTRL_VOLT_VOLT_RAIL_LIST railList;
} LW2080_CTRL_CLK_SET_INFOS_PMU_PARAMS;
#define LW2080_CTRL_CMD_CLK_SET_INFOS_PMU (0x20801066) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_CLK_INTERFACE_ID << 8) | LW2080_CTRL_CLK_SET_INFOS_PMU_PARAMS_MESSAGE_ID" */


/*!
 * Structure describing Clock Monitor specific status data.
 */
typedef struct LW2080_CTRL_CLK_DOMAIN_CLK_MON_STATUS {
    /*!
     * clock domain value
     * @ref LW2080_CTRL_CLK_DOMAIN_XXX
     */
    LwU32 clkApiDomain;
    /*!
     * Active Minimum Threshold value set for clock monitors
     */
    LwU32 lowThreshold;
    /*!
     * Active Maximum Threshold value set for clock monitors
     */
    LwU32 highThreshold;
    /*!
     * [in] Bit mask of faults detected after reading LW_PTRIM_XXX_FMON_FAULT_STATUS_XXXCLK
     * @ref LW2080_CTRL_CLK_DOMAIN_CLK_MON_STATUS_MASK_<xyz>.
     */
    LwU32 clkDomainFaultMask;
} LW2080_CTRL_CLK_DOMAIN_CLK_MON_STATUS;

/*!
 * Macros for the @ref clkDomainFaultMask parameter in the
 * @ref LW2080_CTRL_CLK_DOMAIN_CLK_MON_STATUS structure
 */
#define LW2080_CTRL_CLK_DOMAIN_CLK_MON_STATUS_MASK_DC_FAULT            (0x1U)
#define LW2080_CTRL_CLK_DOMAIN_CLK_MON_STATUS_MASK_LOWER_THRESH_FAULT  (0x2U)
#define LW2080_CTRL_CLK_DOMAIN_CLK_MON_STATUS_MASK_HIGHER_THRESH_FAULT (0x4U)
#define LW2080_CTRL_CLK_DOMAIN_CLK_MON_STATUS_MASK_OVERFLOW_ERROR      (0x8U)

#define LW2080_CTRL_CLK_DOMAINS_CLK_MON_STATUS_PARAMS_MESSAGE_ID (0x67U)

typedef struct LW2080_CTRL_CLK_DOMAINS_CLK_MON_STATUS_PARAMS {
    /*!
     * [out] Boolean indicating whether CLK_MON global fault status bit is set.
     */
    LwBool                                bGlobalStatus;
    /*!
     * [out] This field specifies the number of entries on the caller's.
     */
    LwU32                                 clkMonListSize;
    /*!
     * [out] This field specifies a pointer in the caller's address space
     *       to the buffer into which the clock information is to be returned.
     *       This buffer must be at least as big as clkMonListSize multiplied
     *       by the size of the LW2080_CTRL_CLK_DOMAIN_CLK_MON_STATUS structure.
     */
    LW2080_CTRL_CLK_DOMAIN_CLK_MON_STATUS clkMonList[LW2080_CTRL_CLK_ARCH_MAX_DOMAINS];
} LW2080_CTRL_CLK_DOMAINS_CLK_MON_STATUS_PARAMS;
typedef struct LW2080_CTRL_CLK_DOMAINS_CLK_MON_STATUS_PARAMS *PLW2080_CTRL_CLK_DOMAINS_CLK_MON_STATUS_PARAMS;

/*!
 * LW2080_CTRL_CMD_CLK_DOMAINS_CLK_MON_GET_STATUS
 *
 * This command returns the Clock Monitor Fault status information for each entry
 * on the specified list of clock monitor status structures.
 *
 * See @ref LW2080_CTRL_CLK_DOMAIN_CLK_MON_STATUS for documentation on the
 * parameters.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_OPERATING_SYSTEM
 */
#define LW2080_CTRL_CMD_CLK_DOMAINS_CLK_MON_GET_STATUS (0x20801067) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_CLK_INTERFACE_ID << 8) | LW2080_CTRL_CLK_DOMAINS_CLK_MON_STATUS_PARAMS_MESSAGE_ID" */

/*!
 * Structure containing the target frequency of the corresponding clock domain
 * that will be used in the clock change sequence. Clients may want to pack
 * several items in the list for changing frequency of multiple domains.
 */
typedef struct LW2080_CTRL_CLK_DOMAIN_CLK_MON_LIST_ITEM {
    /*!
     * Clock Domain @ref LW2080_CTRL_CLK_DOMAIN_<xyz>
     */
    LwU32       clkApiDomain;

    /*!
     * Monitored (effectively target) clock freq (in MHz)
     */
    LwU32       clkFreqMHz;

    /*!
     * Minimum Clock Monitor Threshold percent value as per VFE evaluation
     */
    LwUFXP20_12 lowThresholdPercent;

    /*!
     * Minimum Clock Monitor Threshold percent value as per VFE evaluation
     */
    LwUFXP20_12 highThresholdPercent;
} LW2080_CTRL_CLK_DOMAIN_CLK_MON_LIST_ITEM;
typedef struct LW2080_CTRL_CLK_DOMAIN_CLK_MON_LIST_ITEM *PLW2080_CTRL_CLK_DOMAIN_CLK_MON_LIST_ITEM;

/*!
 * Structure containing the number and list of clock monitor domains to be set
 * by a client @ref PLW2080_CTRL_CLK_DOMAIN_CLK_MON_LIST_ITEM.
 */
typedef struct LW2080_CTRL_CLK_DOMAIN_CLK_MON_LIST {
    /*!
     * Number of Clock Monitor Domains that require the frequency change.
     */
    LwU8                                     numDomains;

    /*!
     * List of @ref LW2080_CTRL_CLK_DOMAIN_CLK_MON_LIST_ITEM entries.
     */
    LW2080_CTRL_CLK_DOMAIN_CLK_MON_LIST_ITEM clkDomains[LW2080_CTRL_CLK_CLK_DOMAIN_CLIENT_MAX_DOMAINS];
} LW2080_CTRL_CLK_DOMAIN_CLK_MON_LIST;
typedef struct LW2080_CTRL_CLK_DOMAIN_CLK_MON_LIST *PLW2080_CTRL_CLK_DOMAIN_CLK_MON_LIST;


/*!
 * Pstate 4.0  and later RMCTRLs.
 */

/*!
 * Macro representing an INVALID/UNSUPPORTED CLK_VF_REL index.
 */
#define LW2080_CTRL_CLK_CLK_VF_REL_IDX_ILWALID                           LW2080_CTRL_BOARDOBJ_IDX_ILWALID_8BIT

/*!
 * Enumeration of CLK_VF_REL class types.
 */
#define LW2080_CTRL_CLK_CLK_VF_REL_TYPE_RATIO                            0x00U
#define LW2080_CTRL_CLK_CLK_VF_REL_TYPE_TABLE                            0x01U
#define LW2080_CTRL_CLK_CLK_VF_REL_TYPE_TABLE_FREQ                       0x02U
#define LW2080_CTRL_CLK_CLK_VF_REL_TYPE_RATIO_VOLT                       0x03U
#define LW2080_CTRL_CLK_CLK_VF_REL_TYPE_RATIO_FREQ                       0x04U

/*!
 * Macro defining the positions of VF lwrves.
 */
#define LW2080_CTRL_CLK_CLK_VF_REL_VF_LWRVE_IDX_ILWALID                  0xFFU
#define LW2080_CTRL_CLK_CLK_VF_REL_VF_LWRVE_IDX_PRI                      0x00U
#define LW2080_CTRL_CLK_CLK_VF_REL_VF_LWRVE_IDX_SEC_0                    0x01U
#define LW2080_CTRL_CLK_CLK_VF_REL_VF_LWRVE_IDX_SEC_1                    0x02U
#define LW2080_CTRL_CLK_CLK_VF_REL_VF_LWRVE_IDX_MAX                      0x03U // MUST be last.

/*!
 * Macro defining max supported secondary VF lwrves.
 * Must be kept in sync with LW2080_CTRL_CLK_CLK_VF_REL_VF_LWRVE_IDX_SEC_<x>
 */
#define LW2080_CTRL_CLK_CLK_VF_REL_VF_ENTRY_SEC_MAX                      0x02U

/*!
 * Count of max supported secondary entries.
 */
#define LW2080_CTRL_CLK_CLK_VF_REL_RATIO_SECONDARY_ENTRIES_MAX           0x04U
#define LW2080_CTRL_CLK_CLK_VF_REL_TABLE_SECONDARY_ENTRIES_MAX           0x04U

/*!
 * Count of max supported VF smoothing (ramp rate) entries
 */
#define LW2080_CTRL_CLK_CLK_VF_REL_RATIO_VOLT_VF_SMOOTH_DATA_ENTRIES_MAX 0x02U

/*!
 * Structure describing the params required to generate primary VF lwrve of this
 * VF_REL class.
 */
typedef struct LW2080_CTRL_CLK_CLK_VF_REL_VF_ENTRY_PRI {
    /*!
     * VFE index which describes this CLK_VF_REL entry's VF equation.
     *
     * @ref LW2080_CTRL_PERF_VFE_EQU_INDEX_ILWALID means no VF lwrve specified.
     */
    LwBoardObjIdx vfeIdx;
    /*!
     * VFE index which describes this CLK_VF_REL entry's NAFLL max allowed cpm
     * frequency offset.
     *
     * @ref LW2080_CTRL_PERF_VFE_EQU_INDEX_ILWALID means no  NAFLL max allowed cpm
     * frequency offset specified.
     */
    LwBoardObjIdx cpmMaxFreqOffsetVfeIdx;
    /*!
     * Index of the first CLK_VF_POINT of this object. The set described by the
     * range [vfPointIdxFirst, @ref vfPointIdxLast] represents the VF lwrve
     * described by this object.  This set is in ascending order of the
     * independent variable of the source (i.e. PLL/ONE_SOURCE -> frequency,
     * NAFLL -> voltage).
     */
    LwBoardObjIdx vfPointIdxFirst;
    /*!
     * Index of the last CLK_VF_POINT of this object. The set described by the
     * range [@ref  vfPointIdxFirst, vfPointIdxLast] represents the VF lwrve
     * described by this object.  This set is in ascending order of the
     * independent variable of the source (i.e. PLL/ONE_SOURCE -> frequency,
     * NAFLL -> voltage).
     */
    LwBoardObjIdx vfPointIdxLast;
} LW2080_CTRL_CLK_CLK_VF_REL_VF_ENTRY_PRI;
typedef struct LW2080_CTRL_CLK_CLK_VF_REL_VF_ENTRY_PRI *PLW2080_CTRL_CLK_CLK_VF_REL_VF_ENTRY_PRI;

/*!
 * Structure describing secondary VF lwrves for primary clock domains.
 */
typedef struct LW2080_CTRL_CLK_CLK_VF_REL_VF_ENTRY_SEC {
    /*!
     * VFE index which describes the this CLK_VF_REL entry's secondary VF equation.
     *
     * @ref LW2080_CTRL_PERF_VFE_EQU_INDEX_ILWALID means no VF lwrve specified.
     */
    LwBoardObjIdx vfeIdx;

    /*!
     * VFE index which describes the this CLK_VF_REL entry's NAFLL DVCO offset.
     *
     * @ref LW2080_CTRL_PERF_VFE_EQU_INDEX_ILWALID means no NAFLL DVCO offset specified.
     */
    LwBoardObjIdx dvcoOffsetVfeIdx;

    /*!
     * Index of the first CLK_VF_POINT of this object. The set described by the
     * range [vfPointIdxFirst, @ref vfPointIdxLast] represents the VF lwrve
     * described by this object.  This set is in ascending order of the
     * independent variable of the source (i.e. PLL/ONE_SOURCE -> frequency,
     * NAFLL -> voltage).
     */
    LwBoardObjIdx vfPointIdxFirst;

    /*!
     * Index of the last CLK_VF_POINT of this object. The set described by the
     * range [@ref  vfPointIdxFirst, vfPointIdxLast] represents the VF lwrve
     * described by this object.  This set is in ascending order of the
     * independent variable of the source (i.e. PLL/ONE_SOURCE -> frequency,
     * NAFLL -> voltage).
     */
    LwBoardObjIdx vfPointIdxLast;
} LW2080_CTRL_CLK_CLK_VF_REL_VF_ENTRY_SEC;
typedef struct LW2080_CTRL_CLK_CLK_VF_REL_VF_ENTRY_SEC *PLW2080_CTRL_CLK_CLK_VF_REL_VF_ENTRY_SEC;

/*!
 * Structure representing params used to smooth VF lwrve.
 */
typedef struct LW2080_CTRL_CLK_CLK_VF_REL_RATIO_VOLT_VF_SMOOTH_DATA {
    /*!
     * Base voltage from where RM will start smoothing the VF lwrve.
     * RM is only smoothing voltage based VF lwrve, for frequency based VF
     * lwrve we will set this to LW_U32_MAX
     */
    LwU32       baseVFSmoothVoltuV;
    /*!
     * Maximum ramp rate for a given voltage based VF lwrve. RM will ensure
     * that the generated VF lwrve will respect this ramp rate. If the VF lwrve
     * has discontinuity in it, RM will smoothen the VF lwrve using this value.
     * For Frequency based VF lwrve this is don't care. It will be initialized
     * to LW_U32_MAX.
     */
    LwUFXP20_12 maxVFRampRate;
    /*!
     * Maximum allowed frequency difference between two conselwtive VF Points.
     * This value is callwlated based on the ref@ maxVFRampRate. RM will use
     * this value to remove the discontinuity from the VF lwrve.
     */
    LwU16       maxFreqStepSizeMHz;
} LW2080_CTRL_CLK_CLK_VF_REL_RATIO_VOLT_VF_SMOOTH_DATA;
typedef struct LW2080_CTRL_CLK_CLK_VF_REL_RATIO_VOLT_VF_SMOOTH_DATA *PLW2080_CTRL_CLK_CLK_VF_REL_RATIO_VOLT_VF_SMOOTH_DATA;

/*!
 * Structure representing the combination of array and count of valid
 * entries for VF-lwrve smoothing data. To facilitate multiple
 * VF Ramp rates as per RAM Assist.
 */
typedef struct LW2080_CTRL_CLK_CLK_VF_REL_RATIO_VOLT_VF_SMOOTH_DATA_GRP {
    /*!
     * Contains the number of different VF-ramp rates that indicate the
     * maximum step size.
     * 
     * Valid values are [0, @ref 
     * LW2080_CTRL_CLK_CLK_VF_REL_RATIO_VOLT_VF_SMOOTH_DATA_ENTRIES_MAX).
     */
    LwU8                                                 vfSmoothDataEntriesCount;
    /*!
     * @ref LW2080_CTRL_CLK_CLK_VF_REL_RATIO_VOLT_VF_SMOOTH_DATA
     *
     * Array of VF-smoothing data and ramp rates allowing multiple
     * ramp rates for the same VF relationship. Valid entries are from
     * [0, @ref 
     * LW2080_CTRL_CLK_CLK_VF_REL_RATIO_VOLT_VF_SMOOTH_DATA_ENTRIES_MAX)
     */
    LW2080_CTRL_CLK_CLK_VF_REL_RATIO_VOLT_VF_SMOOTH_DATA vfSmoothDataEntries[LW2080_CTRL_CLK_CLK_VF_REL_RATIO_VOLT_VF_SMOOTH_DATA_ENTRIES_MAX];
} LW2080_CTRL_CLK_CLK_VF_REL_RATIO_VOLT_VF_SMOOTH_DATA_GRP;
typedef struct LW2080_CTRL_CLK_CLK_VF_REL_RATIO_VOLT_VF_SMOOTH_DATA_GRP *PLW2080_CTRL_CLK_CLK_VF_REL_RATIO_VOLT_VF_SMOOTH_DATA_GRP;


/*!
 * Structure describing a RATIO PRIMARY-SECONDARY relationship which specifies the
 * SECONDARY Clock Domain's VF lwrve as a ratio function of the PRIMARY Clock
 * Domain's VF lwrve.
 */
typedef struct LW2080_CTRL_CLK_CLK_VF_REL_RATIO_SECONDARY_ENTRY {
    /*!
     * CLK_DOMAIN index for SECONDARY Clock Domain specified in this entry.
     * @ref LW2080_CTRL_CLK_CLK_DOMAIN_INDEX_ILWALID indicates no CLK_DOMAIN
     * specified.
     */
    LwU8 clkDomIdx;
    /*!
     * Ratio specified for the SECONDARY Clock Domain as a function of the PRIMARY
     * Clock Domain.
     */
    LwU8 ratio;
} LW2080_CTRL_CLK_CLK_VF_REL_RATIO_SECONDARY_ENTRY;
typedef struct LW2080_CTRL_CLK_CLK_VF_REL_RATIO_SECONDARY_ENTRY *PLW2080_CTRL_CLK_CLK_VF_REL_RATIO_SECONDARY_ENTRY;

/*!
 * Structure describing the static configuration/POR state of the _RATIO class.
 */
typedef struct LW2080_CTRL_CLK_CLK_VF_REL_INFO_RATIO {
    /*!
     * Array of ratio entries.
     *
     * Has valid indexes in the range [0, @ref
     * LW2080_CTRL_CLK_CLK_VF_RELS_INFO::secondaryEntryCount).
     */
    LW2080_CTRL_CLK_CLK_VF_REL_RATIO_SECONDARY_ENTRY slaveEntries[LW2080_CTRL_CLK_CLK_VF_REL_RATIO_SECONDARY_ENTRIES_MAX];
} LW2080_CTRL_CLK_CLK_VF_REL_INFO_RATIO;
typedef struct LW2080_CTRL_CLK_CLK_VF_REL_INFO_RATIO *PLW2080_CTRL_CLK_CLK_VF_REL_INFO_RATIO;

/*!
 * Structure describing the static configuration/POR state of the _RATIO_VOLT class.
 */
typedef struct LW2080_CTRL_CLK_CLK_VF_REL_INFO_RATIO_VOLT {
    /*!
     * Super Class.
     */
    LW2080_CTRL_CLK_CLK_VF_REL_INFO_RATIO                    super;

    /*!
     * @ref LW2080_CTRL_CLK_CLK_VF_REL_RATIO_VOLT_VF_SMOOTH_DATA
     *
     * To be deprecated in favor of @ref 
     * LW2080_CTRL_CLK_CLK_VF_REL_INFO_RATIO_VOLT::vfSmoothEntries as per
     * new multiple VF ramp rate.
     */
    LW2080_CTRL_CLK_CLK_VF_REL_RATIO_VOLT_VF_SMOOTH_DATA     vfSmoothData;

    /*!
     * Contains the encapsulation of the valid count of smoothing
     * data entries, and the array of smoothing data itself.
     * 
     * @ref LW2080_CTRL_CLK_CLK_VF_REL_RATIO_VOLT_VF_SMOOTH_DATA_GRP
     */
    LW2080_CTRL_CLK_CLK_VF_REL_RATIO_VOLT_VF_SMOOTH_DATA_GRP vfSmoothDataGrp;
} LW2080_CTRL_CLK_CLK_VF_REL_INFO_RATIO_VOLT;
typedef struct LW2080_CTRL_CLK_CLK_VF_REL_INFO_RATIO_VOLT *PLW2080_CTRL_CLK_CLK_VF_REL_INFO_RATIO_VOLT;

/*!
 * Structure describing a TABLE PRIMARY-SECONDARY relationship which specifies the
 * SECONDARY Clock Domain's VF lwrve is a table-lookup function of the PRIMARY Clock
 * Domain's VF lwrve.
 */
typedef struct LW2080_CTRL_CLK_CLK_VF_REL_TABLE_SECONDARY_ENTRY {
    /*!
     * CLK_DOMAIN index for SECONDARY Clock Domain specified in this entry.  @ref
     * LW2080_CTRL_CLK_CLK_DOMAIN_INDEX_ILWALID indicates no CLK_DOMAIN
     * specified.
     */
    LwU8  clkDomIdx;
    /*!
     * Frequency specified for the SECONDARY Clock Domain for the corresponding
     * frequency on the PRIMARY Clock Domain.
     */
    LwU16 freqMHz;
} LW2080_CTRL_CLK_CLK_VF_REL_TABLE_SECONDARY_ENTRY;
typedef struct LW2080_CTRL_CLK_CLK_VF_REL_TABLE_SECONDARY_ENTRY *PLW2080_CTRL_CLK_CLK_VF_REL_TABLE_SECONDARY_ENTRY;

/*!
 * Structure describing the static configuration/POR state of the _TABLE class.
 */
typedef struct LW2080_CTRL_CLK_CLK_VF_REL_INFO_TABLE {
    /*!
     * Array of table entries.
     *
     * Has valid indexes in the range [0, @ref
     * LW2080_CTRL_CLK_CLK_VF_RELS_INFO::secondaryEntryCount).
     */
    LW2080_CTRL_CLK_CLK_VF_REL_TABLE_SECONDARY_ENTRY slaveEntries[LW2080_CTRL_CLK_CLK_VF_REL_TABLE_SECONDARY_ENTRIES_MAX];
} LW2080_CTRL_CLK_CLK_VF_REL_INFO_TABLE;
typedef struct LW2080_CTRL_CLK_CLK_VF_REL_INFO_TABLE *PLW2080_CTRL_CLK_CLK_VF_REL_INFO_TABLE;

/*!
 * CLK_VF_REL type-specific data union. Discriminated by
 * CLK_VF_REL::super.type.
 */


/*!
 * Structure describing CLK_VF_REL static information/POR.  Implements the
 * BOARDOBJ model/interface.
 */
typedef struct LW2080_CTRL_CLK_CLK_VF_REL_INFO {
    /*!
     * LW2080_CTRL_BOARDOBJ super class.  Must always be first object in
     * structure.
     */
    LW2080_CTRL_BOARDOBJ                    super;
    /*!
     * XAPI does not support using an element within another structure as a
     * discriminant.  Placing this redundant type value here until that design
     * constraint can be fixed.
     */
    LwU8                                    type;
    /*!
     * Index of the VOLTAGE_RAIL for this CLK_VF_REL object.
     */
    LwU8                                    railIdx;
    /*!
     * Boolean flag indicating whether this entry supports OC/OV when those
     * settings are applied to the corresponding CLK_DOMAIN object.
     */
    LwBool                                  bOCOVEnabled;
    /*!
     * Maximum frequency for this CLK_VF_REL entry. Entries for a given domain
     * need to be specified in ascending maxFreqMhz.
     */
    LwU16                                   freqMaxMHz;
    /*!
     * Primary VF lwrve params.
     */
    LW2080_CTRL_CLK_CLK_VF_REL_VF_ENTRY_PRI vfEntryPri;

    /*!
     * Array of secondary VF lwrve's param.
     * Indexed per the LW2080_CTRL_CLK_CLK_VF_REL_VF_LWRVE_IDX_SEC_<X enumeration.
     *
     * Has valid indexes in the range [0, @ref
     * LW2080_CTRL_CLK_CLK_VF_RELS_INFO::vfEntryCountSec).
     */
    LW2080_CTRL_CLK_CLK_VF_REL_VF_ENTRY_SEC vfEntriesSec[LW2080_CTRL_CLK_CLK_VF_REL_VF_ENTRY_SEC_MAX];

    /*!
     * Type-specific data union.
     */
    union {
        LW2080_CTRL_CLK_CLK_VF_REL_INFO_RATIO      ratio;
        LW2080_CTRL_CLK_CLK_VF_REL_INFO_RATIO_VOLT ratioVolt;
        LW2080_CTRL_CLK_CLK_VF_REL_INFO_TABLE      table;
    } data;
} LW2080_CTRL_CLK_CLK_VF_REL_INFO;
typedef struct LW2080_CTRL_CLK_CLK_VF_REL_INFO *PLW2080_CTRL_CLK_CLK_VF_REL_INFO;

/*!
 * Structure describing CLK_VF_RELS static information/POR. Implements the
 * BOARDOBJGRP model/interface.
 */
#define LW2080_CTRL_CLK_CLK_VF_RELS_INFO_MESSAGE_ID (0x70U)

typedef struct LW2080_CTRL_CLK_CLK_VF_RELS_INFO {
    /*!
     * LW2080_CTRL_BOARDOBJGRP_E255 super class. Must always be first object in
     * structure.
     */
    LW2080_CTRL_BOARDOBJGRP_E255    super;

    /*!
     * Number of Secondary entries per VF Rel entry.
     */
    LwU8                            slaveEntryCount;
    /*!
     * Count of secondary VF entries.
     */
    LwU8                            vfEntryCountSec;
    /*!
     * Array of CLK_VF_REL structures.  Has valid indexes corresponding to the
     * bits set in @ref super.objMask.
     */
    LW2080_CTRL_CLK_CLK_VF_REL_INFO vfRels[LW2080_CTRL_BOARDOBJGRP_E255_MAX_OBJECTS];
} LW2080_CTRL_CLK_CLK_VF_RELS_INFO;
typedef struct LW2080_CTRL_CLK_CLK_VF_RELS_INFO *PLW2080_CTRL_CLK_CLK_VF_RELS_INFO;

/*!
 * LW2080_CTRL_CMD_CLK_CLK_VF_RELS_GET_INFO
 *
 * This command returns CLK_VF_RELS static object information/POR as specified
 * by the VBIOS in the Clocks VF Relationships Table.
 *
 * The CLK_VF_REL objects are indexed per how they are specified in the VBIOS
 * table.
 *
 * See @ref LW2080_CTRL_CLK_CLK_VF_RELS_INFO for documentation on the
 * parameters.
 *
 * Return values are specified per @ref BoardObjGrpGetInfo.
 */
#define LW2080_CTRL_CMD_CLK_CLK_VF_RELS_GET_INFO (0x20801070) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_CLK_INTERFACE_ID << 8) | LW2080_CTRL_CLK_CLK_VF_RELS_INFO_MESSAGE_ID" */

/*!
 * Structure describing VF_REL dynamic information. Implements the
 * BOARDOBJ model/interface.
 */
typedef struct LW2080_CTRL_CLK_CLK_VF_REL_STATUS {
    /*!
     * LW2080_CTRL_BOARDOBJ super class.  Must always be first object in
     * structure.
     */
    LW2080_CTRL_BOARDOBJ super;

    /*!
     * Maximum frequency for this VF_REL entry adjusted with the applied
     * frequency offsets on given clock domain for this VF Rel index
     */
    LwU16                offsettedFreqMaxMHz;
} LW2080_CTRL_CLK_CLK_VF_REL_STATUS;
typedef struct LW2080_CTRL_CLK_CLK_VF_REL_STATUS *PLW2080_CTRL_CLK_CLK_VF_REL_STATUS;

/*!
 * Structure describing CLK_VF_RELS dynamic information. Implements the
 * BOARDOBJGRP model/interface.
 */
#define LW2080_CTRL_CLK_CLK_VF_RELS_STATUS_MESSAGE_ID (0x71U)

typedef struct LW2080_CTRL_CLK_CLK_VF_RELS_STATUS {
    /*!
     * LW2080_CTRL_BOARDOBJGRP_E255 super class. Must always be first object in
     * structure.
     */
    LW2080_CTRL_BOARDOBJGRP_E255      super;

    /*!
     * Array of CLK_VF_REL structures. Has valid indexes corresponding to the
     * bits set in @ref super.objMask.
     */
    LW2080_CTRL_CLK_CLK_VF_REL_STATUS vfRels[LW2080_CTRL_BOARDOBJGRP_E255_MAX_OBJECTS];
} LW2080_CTRL_CLK_CLK_VF_RELS_STATUS;
typedef struct LW2080_CTRL_CLK_CLK_VF_RELS_STATUS *PLW2080_CTRL_CLK_CLK_VF_RELS_STATUS;

/*!
 * LW2080_CTRL_CMD_CLK_CLK_VF_RELS_GET_STATUS
 *
 * This command returns the CLK_VF_RELS dynamic state information associated by the
 * Clk Vf Relationship functionality
 *
 * See @ref LW2080_CTRL_CLK_CLK_VF_RELS_STATUS for documentation on the
 * parameters.
 *
 * Return values are specified per @ref BoardObjGrpGetStatus.
 */
#define LW2080_CTRL_CMD_CLK_CLK_VF_RELS_GET_STATUS (0x20801071) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_CLK_INTERFACE_ID << 8) | LW2080_CTRL_CLK_CLK_VF_RELS_STATUS_MESSAGE_ID" */

/*!
 * Structure describing the control parameters associated with the _VF_REL_RATIO class.
 */
typedef struct LW2080_CTRL_CLK_CLK_VF_REL_CONTROL_RATIO {
    /*!
     * Array of ratio entries.
     *
     * Has valid indexes in the range [0, @ref
     * LW2080_CTRL_CLK_CLK_PROGS_INFO::secondaryEntryCount).
     */
    LW2080_CTRL_CLK_CLK_VF_REL_RATIO_SECONDARY_ENTRY slaveEntries[LW2080_CTRL_CLK_CLK_VF_REL_RATIO_SECONDARY_ENTRIES_MAX];
} LW2080_CTRL_CLK_CLK_VF_REL_CONTROL_RATIO;
typedef struct LW2080_CTRL_CLK_CLK_VF_REL_CONTROL_RATIO *PLW2080_CTRL_CLK_CLK_VF_REL_CONTROL_RATIO;

/*!
 * CLK_VF_REL type-specific data union. Discriminated by
 * CLK_VF_REL::super.type.
 */


/*!
 * Structure describing CLK_VF_REL specific control parameters. Implements the
 * BOARDOBJ model/interface.
 */
typedef struct LW2080_CTRL_CLK_CLK_VF_REL_CONTROL {
    /*!
     * LW2080_CTRL_BOARDOBJ super class.  Must always be first object in
     * structure.
     */
    LW2080_CTRL_BOARDOBJ      super;

    /*!
     * XAPI does not support using an element within another structure as a
     * discriminant.  Placing this redundant type value here until that design
     * constraint can be fixed.
     */
    LwU8                      type;

    /*!
     * Deltas for VF_REL class
     */
    LW2080_CTRL_CLK_CLK_DELTA delta;

    /*!
     * Type-specific data union.
     */
    union {
        LW2080_CTRL_CLK_CLK_VF_REL_CONTROL_RATIO ratio;
    } data;
} LW2080_CTRL_CLK_CLK_VF_REL_CONTROL;
typedef struct LW2080_CTRL_CLK_CLK_VF_REL_CONTROL *PLW2080_CTRL_CLK_CLK_VF_REL_CONTROL;

/*!
 * Structure describing VF_RELS specific control parameters.  Implements the
 * BOARDOBJGRP model/interface.
 */
typedef struct LW2080_CTRL_CLK_CLK_VF_RELS_CONTROL {
    /*!
     * LW2080_CTRL_BOARDOBJGRP_E255 super class.  Must always be first object in
     * structure.
     */
    LW2080_CTRL_BOARDOBJGRP_E255       super;
    /*!
     * Array of VF_REL structures.  Has valid indexes corresponding to the
     * bits set in @ref super.objMask.
     */
    LW2080_CTRL_CLK_CLK_VF_REL_CONTROL vfRels[LW2080_CTRL_BOARDOBJGRP_E255_MAX_OBJECTS];
} LW2080_CTRL_CLK_CLK_VF_RELS_CONTROL;
typedef struct LW2080_CTRL_CLK_CLK_VF_RELS_CONTROL *PLW2080_CTRL_CLK_CLK_VF_RELS_CONTROL;

/*!
 * LW2080_CTRL_CMD_CLK_CLK_VF_RELS_GET_CONTROL
 *
 * This command returns VF_RELS control parameters as specified
 * by the VBIOS in the Clocks Programming Table.
 *
 * The VF_REL objects are indexed per how they are specified in the VBIOS
 * table.
 *
 * See @ref LW2080_CTRL_CLK_CLK_VF_RELS_CONTROL for documentation on the
 * parameters.
 *
 * Return values are specified per @ref BoardObjGrpGetControl.
 */
#define LW2080_CTRL_CMD_CLK_CLK_VF_RELS_GET_CONTROL (0x20801072) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_CLK_INTERFACE_ID << 8) | 0x72" */

/*!
 * LW2080_CTRL_CMD_CLK_CLK_VF_RELS_SET_CONTROL
 *
 * This command accepts client-specified control parameters for a set
 * of VF_RELS entries in the Clocks Table, and applies these new
 * parameters to the set of VF_RELS entries.
 *
 * The VF_REL objects are indexed per how they are specified in the VBIOS
 * table.
 *
 * See @ref LW2080_CTRL_CLK_CLK_VF_RELS_CONTROL for documentation on the
 * parameters.
 *
 * Return values are specified per @ref BoardObjGrpSetControl.
 */
#define LW2080_CTRL_CMD_CLK_CLK_VF_RELS_SET_CONTROL (0x20801073) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_CLK_INTERFACE_ID << 8) | 0x73" */

/*!
 * Enumeration of CLK_ENUM class types.
 */
#define LW2080_CTRL_CLK_CLK_ENUM_TYPE_1X            0x00U

/*!
 * Structure describing CLK_ENUM static information/POR. Implements the
 * BOARDOBJ model/interface.
 */
typedef struct LW2080_CTRL_CLK_CLK_ENUM_INFO {
    /*!
     * LW2080_CTRL_BOARDOBJ super class.  Must always be first object in
     * structure.
     */
    LW2080_CTRL_BOARDOBJ super;
    /*!
     * XAPI does not support using an element within another structure as a
     * discriminant.  Placing this redundant type value here until that design
     * constraint can be fixed.
     */
    LwU8                 type;
    /*!
     * Boolean flag indicating whether this entry supports OC/OV when those
     * settings are applied to the corresponding CLK_DOMAIN object.
     */
    LwBool               bOCOVEnabled;
    /*!
     * Minimum frequency in MHz which can be programmed on the CLK_DOMAIN.
     */
    LwU16                freqMinMHz;
    /*!
     * Maximum frequency in MHz which can be programmed on the CLK_DOMAIN.
     */
    LwU16                freqMaxMHz;
} LW2080_CTRL_CLK_CLK_ENUM_INFO;
typedef struct LW2080_CTRL_CLK_CLK_ENUM_INFO *PLW2080_CTRL_CLK_CLK_ENUM_INFO;

/*!
 * Structure describing CLK_ENUMS static information/POR. Implements the
 * BOARDOBJGRP model/interface.
 */
#define LW2080_CTRL_CLK_CLK_ENUMS_INFO_MESSAGE_ID (0x74U)

typedef struct LW2080_CTRL_CLK_CLK_ENUMS_INFO {
    /*!
     * LW2080_CTRL_BOARDOBJGRP_E255 super class. Must always be first object in
     * structure.
     */
    LW2080_CTRL_BOARDOBJGRP_E255  super;

    /*!
     * Array of CLK_ENUM structures. Has valid indexes corresponding to the
     * bits set in @ref super.objMask.
     */
    LW2080_CTRL_CLK_CLK_ENUM_INFO enums[LW2080_CTRL_BOARDOBJGRP_E255_MAX_OBJECTS];
} LW2080_CTRL_CLK_CLK_ENUMS_INFO;
typedef struct LW2080_CTRL_CLK_CLK_ENUMS_INFO *PLW2080_CTRL_CLK_CLK_ENUMS_INFO;

/*!
 * LW2080_CTRL_CMD_CLK_CLK_ENUMS_GET_INFO
 *
 * This command returns CLK_ENUMS static object information/POR as specified
 * by the VBIOS in the Clocks VF Relationships Table.
 *
 * The CLK_ENUM objects are indexed per how they are specified in the VBIOS
 * table.
 *
 * See @ref LW2080_CTRL_CLK_CLK_ENUMS_INFO for documentation on the
 * parameters.
 *
 * Return values are specified per @ref BoardObjGrpGetInfo.
 */
#define LW2080_CTRL_CMD_CLK_CLK_ENUMS_GET_INFO               (0x20801074) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_CLK_INTERFACE_ID << 8) | LW2080_CTRL_CLK_CLK_ENUMS_INFO_MESSAGE_ID" */


/*!
 * Enumeration of CLK_PROP_REGIME class types.
 */
#define LW2080_CTRL_CLK_CLK_PROP_REGIME_TYPE_1X              0x00U

/*!
 * Enumerations of the CLK_PROP_REGIME Id HALs, as specified by the VBIOS Propagation Regime Table.
 *
 * https://confluence.lwpu.com/display/RMPER/Clock+Propagation+Tables+Spec#ClockPropagationTablesSpec-ClockPropagationRegimeHAL
 */
#define LW2080_CTRL_CLK_CLK_PROP_REGIME_HAL_GA10X            0x00U
#define LW2080_CTRL_CLK_CLK_PROP_REGIME_HAL_TU10X_AMPERE     0x01U

/*!
 * Init 0 as invalid regime id.
 */
#define LW2080_CTRL_CLK_CLK_PROP_REGIME_ID_ILWALID           0U

/*!
 * Strict propagation to all CLK_DOMAINs.
 */
#define LW2080_CTRL_CLK_CLK_PROP_REGIME_ID_STRICT            1U
/*!
 * In case of Pstate lock, loose propagation to all CLK_DOMAINs
 * In all other cases, no propagation from client's request.
 */
#define LW2080_CTRL_CLK_CLK_PROP_REGIME_ID_LOOSE             2U
/*!
 * Strict propagation to DRAMCLK and all its performance dependencies
 */
#define LW2080_CTRL_CLK_CLK_PROP_REGIME_ID_DRAM_STRICT       3U
/*!
 * Strict propagation to DRAMCLK and its derectly linked clock domains.
 * DRAM and PCIe are 1-1 linked today through PSTATE so this regime is
 * desigend to re-create the legacy behavior of DRAM LOCK.
 */
#define LW2080_CTRL_CLK_CLK_PROP_REGIME_ID_DRAM_LOCK         4U

/*!
 * Strict propagation to GPCCLK and all of its performance dependencies.
 */
#define LW2080_CTRL_CLK_CLK_PROP_REGIME_ID_GPC_STRICT        5U
/*!
 * Strict propagation to DISPCLK and all of its performance dependencies.
 */
#define LW2080_CTRL_CLK_CLK_PROP_REGIME_ID_DISP_STRICT       6U
/*!
 * Strict propagation to PCIEGENCLK and its performance dependencies.
 */
#define LW2080_CTRL_CLK_CLK_PROP_REGIME_ID_PCIE_STRICT       7U
/*!
 * Strict propagation to XBARCLK and its performance dependencies.
 */
#define LW2080_CTRL_CLK_CLK_PROP_REGIME_ID_XBAR_STRICT       8U

/*!
 * Strict propagation to XBARCLK and its directly linked clock domains in terms
 * of performance and SW support. LWD clock has its own SW utilization controller
 * therefore it is not directly linked to XBAR clock.
 */
#define LW2080_CTRL_CLK_CLK_PROP_REGIME_ID_XBAR_LOCK         9U
/*!
 * Strict propagation to all CLK_DOMAINS that directly controls the performance.
 */
#define LW2080_CTRL_CLK_CLK_PROP_REGIME_ID_PERF_STRICT       10U
/*!
 * Legacy name of perf strict regime.
 * Display and HUB clock are removed from this mask threfore the legacy name was
 * strict no display.
 */
#define LW2080_CTRL_CLK_CLK_PROP_REGIME_ID_STRICT_NO_DISP    LW2080_CTRL_CLK_CLK_PROP_REGIME_ID_PERF_STRICT
/*!
 * Strict propagation to all CLK_DOMAINS specified by IMP VPSTATEs.
 */
#define LW2080_CTRL_CLK_CLK_PROP_REGIME_ID_IMP               11U
/*!
 * Strict propagation to all CLK_DOMAINS specified by IMP clients.
 * This limit will have propagation with IMP + DISPCLK.
 */
#define LW2080_CTRL_CLK_CLK_PROP_REGIME_ID_IMP_CLIENT_STRICT 12U
/*!
 * Strict propagation to all clocks driving GPC core power.
 */
#define LW2080_CTRL_CLK_CLK_PROP_REGIME_ID_GPC_POWER_STRICT  13U

/*!
 * Strict propagation to all clocks driving XBAR core power.
 */
#define LW2080_CTRL_CLK_CLK_PROP_REGIME_ID_XBAR_POWER_STRICT 14U

/*!
 * Strict propagation to all clocks driving LWVDD voltag
 * Voltage Controllers like reliability limit will use this regime.
 */
#define LW2080_CTRL_CLK_CLK_PROP_REGIME_ID_LWVDD_STRICT      15U

/*!
 * Strict propagation to all clocks driving MSVDD voltage.
 * Voltage Controllers like reliability limit will use this regime.
 */
#define LW2080_CTRL_CLK_CLK_PROP_REGIME_ID_MSVDD_STRICT      16U

/*!
 * Reserved propagation regimes for bring up.
 */
#define LW2080_CTRL_CLK_CLK_PROP_REGIME_ID_RSVD_0            17U
#define LW2080_CTRL_CLK_CLK_PROP_REGIME_ID_RSVD_1            18U
#define LW2080_CTRL_CLK_CLK_PROP_REGIME_ID_RSVD_2            19U

/*!
 * Add new regime id here.
 */

/*!
 * Must always be the last regime.
 */
#define LW2080_CTRL_CLK_CLK_PROP_REGIME_ID_MAX               20U

/*!
 * Structure describing CLK_PROP_REGIME static information/POR. Implements the
 * BOARDOBJ model/interface.
 */
typedef struct LW2080_CTRL_CLK_CLK_PROP_REGIME_INFO {
    /*!
     * LW2080_CTRL_BOARDOBJ super class.  Must always be first object in
     * structure.
     */
    LW2080_CTRL_BOARDOBJ             super;
    /*!
     * Clock Propagation Regime Id.
     * @ref LW2080_CTRL_CLK_CLK_PROP_REGIME_ID_MAX
     */
    LwU8                             regimeId;
    /*!
     * Mask of clock domains that must be programmed based on their
     * clock propagation relationship.
     */
    LW2080_CTRL_BOARDOBJGRP_MASK_E32 clkDomainMask;
} LW2080_CTRL_CLK_CLK_PROP_REGIME_INFO;
typedef struct LW2080_CTRL_CLK_CLK_PROP_REGIME_INFO *PLW2080_CTRL_CLK_CLK_PROP_REGIME_INFO;

/*!
 * Structure describing CLK_PROP_REGIMES static information/POR. Implements the
 * BOARDOBJGRP model/interface.
 */
#define LW2080_CTRL_CLK_CLK_PROP_REGIMES_INFO_MESSAGE_ID (0x79U)

typedef struct LW2080_CTRL_CLK_CLK_PROP_REGIMES_INFO {
    /*!
     * LW2080_CTRL_BOARDOBJGRP_E32 super class. Must always be first object in
     * structure.
     */
    LW2080_CTRL_BOARDOBJGRP_E32          super;

    /*!
     * Clock Propagation regime HAL.
     * @ref LW2080_CTRL_CLK_CLK_PROP_REGIME_HAL_<xyz>
     */
    LwU8                                 regimeHal;

    /*!
     * Array of CLK_PROP_REGIME structures. Has valid indexes corresponding to the
     * bits set in @ref super.objMask.
     */
    LW2080_CTRL_CLK_CLK_PROP_REGIME_INFO propRegimes[LW2080_CTRL_BOARDOBJGRP_E32_MAX_OBJECTS];
} LW2080_CTRL_CLK_CLK_PROP_REGIMES_INFO;
typedef struct LW2080_CTRL_CLK_CLK_PROP_REGIMES_INFO *PLW2080_CTRL_CLK_CLK_PROP_REGIMES_INFO;

/*!
 * LW2080_CTRL_CMD_CLK_CLK_PROP_REGIMES_GET_INFO
 *
 * This command returns CLK_PROP_REGIMES static object information/POR as specified
 * by the VBIOS in the Clocks VF Relationships Table.
 *
 * The CLK_PROP_REGIME objects are indexed per how they are specified in the VBIOS
 * table.
 *
 * See @ref LW2080_CTRL_CLK_CLK_PROP_REGIMES_INFO for documentation on the
 * parameters.
 *
 * Return values are specified per @ref BoardObjGrpGetInfo.
 */
#define LW2080_CTRL_CMD_CLK_CLK_PROP_REGIMES_GET_INFO (0x20801079) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_CLK_INTERFACE_ID << 8) | LW2080_CTRL_CLK_CLK_PROP_REGIMES_INFO_MESSAGE_ID" */

/*!
 * Structure describing CLK_PROP_REGIME specific control parameters. Implements
 * the BOARDOBJ model/interface.
 */
typedef struct LW2080_CTRL_CLK_CLK_PROP_REGIME_CONTROL {
    /*!
     * LW2080_CTRL_BOARDOBJ super class.  Must always be first object in
     * structure.
     */
    LW2080_CTRL_BOARDOBJ             super;

    /*!
     * Mask of clock domains that must be programmed based on their
     * clock propagation relationship.
     */
    LW2080_CTRL_BOARDOBJGRP_MASK_E32 clkDomainMask;
} LW2080_CTRL_CLK_CLK_PROP_REGIME_CONTROL;
typedef struct LW2080_CTRL_CLK_CLK_PROP_REGIME_CONTROL *PLW2080_CTRL_CLK_CLK_PROP_REGIME_CONTROL;

/*!
 * Structure describing CLK_PROP_REGIME specific control parameters.  Implements
 * the BOARDOBJGRP model/interface.
 */
typedef struct LW2080_CTRL_CLK_CLK_PROP_REGIMES_CONTROL {
    /*!
     * LW2080_CTRL_BOARDOBJGRP_E32 super class. Must always be first object in
     * structure.
     */
    LW2080_CTRL_BOARDOBJGRP_E32             super;

    /*!
     * Array of CLK_PROP_REGIME structures. Has valid indexes corresponding to the
     * bits set in @ref super.objMask.
     */
    LW2080_CTRL_CLK_CLK_PROP_REGIME_CONTROL propRegimes[LW2080_CTRL_BOARDOBJGRP_E32_MAX_OBJECTS];
} LW2080_CTRL_CLK_CLK_PROP_REGIMES_CONTROL;
typedef struct LW2080_CTRL_CLK_CLK_PROP_REGIMES_CONTROL *PLW2080_CTRL_CLK_CLK_PROP_REGIMES_CONTROL;

/*!
 * LW2080_CTRL_CMD_CLK_CLK_PROP_REGIMES_GET_CONTROL
 *
 * This command returns CLK_PROP_REGIMES control parameters as specified
 * by the VBIOS in the Clocks Propagation Regime Table.
 *
 * The CLK_PROP_REGIME objects are indexed per how they are specified in the VBIOS
 * table.
 *
 * See @ref LW2080_CTRL_CLK_CLK_PROP_REGIMES_CONTROL for documentation on the
 * parameters.
 *
 * Return values are specified per @ref BoardObjGrpGetControl.
 */
#define LW2080_CTRL_CMD_CLK_CLK_PROP_REGIMES_GET_CONTROL (0x2080107b) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_CLK_INTERFACE_ID << 8) | 0x7B" */

/*!
 * LW2080_CTRL_CMD_CLK_CLK_PROP_REGIMES_SET_CONTROL
 *
 * This command accepts client-specified control parameters for a set
 * of CLK_PROP_REGIMES entries in the Clocks Propagation Regime Table,
 * and applies these new parameters to the set of CLK_PROP_REGIMES entries.
 *
 * The CLK_PROP_REGIME objects are indexed per how they are specified in the VBIOS
 * table.
 *
 * See @ref LW2080_CTRL_CLK_CLK_PROP_REGIMES_CONTROL for documentation on the
 * parameters.
 *
 * Return values are specified per @ref BoardObjGrpSetControl.
 */
#define LW2080_CTRL_CMD_CLK_CLK_PROP_REGIMES_SET_CONTROL (0x2080107c) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_CLK_INTERFACE_ID << 8) | 0x7C" */


/*!
 * Enumeration of CLK_PROP_TOP class types.
 */
#define LW2080_CTRL_CLK_CLK_PROP_TOP_TYPE_1X             0x00U

/*!
 * Array of source to destination clock propagation topology path for a given
 * programmable clock domain. Each path stores the clock propagation relationship
 * index which represents the next relationship index on the path from source
 * clock domain to desitnation clock domain.
 */
typedef struct LW2080_CTRL_CLK_CLK_PROP_TOP_CLK_DOMAIN_DST_PATH {
    LwBoardObjIdx dstPath[LW2080_CTRL_CLK_CLK_DOMAIN_CLIENT_MAX_DOMAINS];
} LW2080_CTRL_CLK_CLK_PROP_TOP_CLK_DOMAIN_DST_PATH;
typedef struct LW2080_CTRL_CLK_CLK_PROP_TOP_CLK_DOMAIN_DST_PATH *PLW2080_CTRL_CLK_CLK_PROP_TOP_CLK_DOMAIN_DST_PATH;

/*!
 * @brief   Determines whether an index into
 *          @ref LW2080_CTRL_CLK_CLK_PROP_TOP_CLK_DOMAIN_DST_PATH::dstPath
 *          is valid
 *
 * @param[in]   clkDomainIdx    Index to check
 *
 * @return  LW_TRUE     The index is valid
 * @return  LW_FALSE    Otherwise
 */
#define LW2080_CTRL_CLK_CLK_PROP_TOP_CLK_DOMAIN_DST_PATH_IDX_VALID(clkDomainIdx) \
    (clkDomainIdx < LW_ARRAY_ELEMENTS(((LW2080_CTRL_CLK_CLK_PROP_TOP_CLK_DOMAIN_DST_PATH *)NULL)->dstPath))

/*!
 * Array of source to destination clock propagation topology path for all
 * programmable clock domains.
 */
typedef struct LW2080_CTRL_CLK_CLK_PROP_TOP_CLK_DOMAINS_DST_PATH {
    LW2080_CTRL_CLK_CLK_PROP_TOP_CLK_DOMAIN_DST_PATH domainDstPath[LW2080_CTRL_CLK_CLK_DOMAIN_CLIENT_MAX_DOMAINS];
} LW2080_CTRL_CLK_CLK_PROP_TOP_CLK_DOMAINS_DST_PATH;
typedef struct LW2080_CTRL_CLK_CLK_PROP_TOP_CLK_DOMAINS_DST_PATH *PLW2080_CTRL_CLK_CLK_PROP_TOP_CLK_DOMAINS_DST_PATH;

/*!
 * @brief   Determines whether an index into
 *          @ref LW2080_CTRL_CLK_CLK_PROP_TOP_CLK_DOMAINS_DST_PATH::domainDstPath
 *          is valid
 *
 * @param[in]   clkDomainIdx    Index to check
 *
 * @return  LW_TRUE     The index is valid
 * @return  LW_FALSE    Otherwise
 */
#define LW2080_CTRL_CLK_CLK_PROP_TOP_CLK_DOMAINS_DST_PATH_IDX_VALID(clkDomainIdx) \
    (clkDomainIdx < LW_ARRAY_ELEMENTS(((LW2080_CTRL_CLK_CLK_PROP_TOP_CLK_DOMAINS_DST_PATH *)NULL)->domainDstPath))

/*!
 * Enumerations of the CLK_PROP_TOP HALs, as specified by the VBIOS Propagation Topology Table.
 *
 * https://confluence.lwpu.com/display/RMPER/Clock+Propagation+Tables+Spec#ClockPropagationTablesSpec-ClockPropagationTopologyHAL
 */
#define LW2080_CTRL_CLK_CLK_PROP_TOP_HAL_TU10X            0x00U
#define LW2080_CTRL_CLK_CLK_PROP_TOP_HAL_TU10X_AMPERE     0x01U
#define LW2080_CTRL_CLK_CLK_PROP_TOP_HAL_GA10X_TESLA_0    0x02U
#define LW2080_CTRL_CLK_CLK_PROP_TOP_HAL_GA10X_TESLA_1    0x03U
#define LW2080_CTRL_CLK_CLK_PROP_TOP_HAL_GA10X_TESLA_2    0x04U
#define LW2080_CTRL_CLK_CLK_PROP_TOP_HAL_GA10X_TESLA_3    0x05U

#define LW2080_CTRL_CLK_CLK_PROP_TOP_HAL_GA10X_TESLA      LW2080_CTRL_CLK_CLK_PROP_TOP_HAL_GA10X_TESLA_0

/*!
 * Enumeration of clock propagation topology ids.
 */
#define LW2080_CTRL_CLK_CLK_PROP_TOP_ID_ILWALID           0xFFU
#define LW2080_CTRL_CLK_CLK_PROP_TOP_ID_GRAPHICS          0x00U
#define LW2080_CTRL_CLK_CLK_PROP_TOP_ID_COMPUTE           0x01U
#define LW2080_CTRL_CLK_CLK_PROP_TOP_ID_RSVD_0            0x02U
#define LW2080_CTRL_CLK_CLK_PROP_TOP_ID_RSVD_1            0x03U
#define LW2080_CTRL_CLK_CLK_PROP_TOP_ID_RSVD_2            0x04U
#define LW2080_CTRL_CLK_CLK_PROP_TOP_ID_RSVD_3            0x05U
#define LW2080_CTRL_CLK_CLK_PROP_TOP_ID_RSVD_4            0x06U
#define LW2080_CTRL_CLK_CLK_PROP_TOP_ID_RSVD_5            0x07U
#define LW2080_CTRL_CLK_CLK_PROP_TOP_ID_RSVD_6            0x08U
#define LW2080_CTRL_CLK_CLK_PROP_TOP_ID_RSVD_7            0x09U
#define LW2080_CTRL_CLK_CLK_PROP_TOP_ID_RSVD_8            0x0AU
#define LW2080_CTRL_CLK_CLK_PROP_TOP_ID_RSVD_9            0x0BU
#define LW2080_CTRL_CLK_CLK_PROP_TOP_ID_RSVD_10           0x0LW
#define LW2080_CTRL_CLK_CLK_PROP_TOP_ID_RSVD_11           0x0DU
#define LW2080_CTRL_CLK_CLK_PROP_TOP_ID_RSVD_12           0x0EU
#define LW2080_CTRL_CLK_CLK_PROP_TOP_ID_RSVD_13           0x0FU
#define LW2080_CTRL_CLK_CLK_PROP_TOP_ID_MAX               0x10U

#define LW2080_CTRL_CLK_CLK_PROP_TOP_ID_GRAPHICS_MEMORY_0 LW2080_CTRL_CLK_CLK_PROP_TOP_ID_RSVD_5
#define LW2080_CTRL_CLK_CLK_PROP_TOP_ID_GRAPHICS_MEMORY_1 LW2080_CTRL_CLK_CLK_PROP_TOP_ID_RSVD_6
#define LW2080_CTRL_CLK_CLK_PROP_TOP_ID_GRAPHICS_MEMORY_2 LW2080_CTRL_CLK_CLK_PROP_TOP_ID_RSVD_7
#define LW2080_CTRL_CLK_CLK_PROP_TOP_ID_GRAPHICS_MEMORY_3 LW2080_CTRL_CLK_CLK_PROP_TOP_ID_RSVD_8
#define LW2080_CTRL_CLK_CLK_PROP_TOP_ID_GRAPHICS_MEMORY_4 LW2080_CTRL_CLK_CLK_PROP_TOP_ID_RSVD_9

/*!
 * Macro for CLK_PROP_TOP index
 */
#define LW2080_CTRL_CLK_CLK_PROP_TOP_IDX_ILWALID          LW2080_CTRL_BOARDOBJ_IDX_ILWALID

/*!
 * Structure describing CLK_PROP_TOP static information/POR. Implements the
 * BOARDOBJ model/interface.
 */
typedef struct LW2080_CTRL_CLK_CLK_PROP_TOP_INFO {
    /*!
     * LW2080_CTRL_BOARDOBJ super class.  Must always be first object in
     * structure.
     */
    LW2080_CTRL_BOARDOBJ                              super;

    /*!
     * Clock Propagation Topology Id.
     * @ ref LW2080_CTRL_CLK_CLK_PROP_TOP_ID_MAX
     */
    LwU8                                              topId;

    /*!
     * Mask of clock propagation topology relationships that are valid for this clock
     * propagation topology.
     */
    LW2080_CTRL_BOARDOBJGRP_MASK_E255                 clkPropTopRelMask;

    /*!
     * Clock Propagation Topology Graph for all programmable clock domains.
     */
    LW2080_CTRL_CLK_CLK_PROP_TOP_CLK_DOMAINS_DST_PATH domainsDstPath;
} LW2080_CTRL_CLK_CLK_PROP_TOP_INFO;
typedef struct LW2080_CTRL_CLK_CLK_PROP_TOP_INFO *PLW2080_CTRL_CLK_CLK_PROP_TOP_INFO;

/*!
 * Structure describing CLK_PROP_TOPS static information/POR. Implements the
 * BOARDOBJGRP model/interface.
 */
#define LW2080_CTRL_CLK_CLK_PROP_TOPS_INFO_MESSAGE_ID (0x7DU)

typedef struct LW2080_CTRL_CLK_CLK_PROP_TOPS_INFO {
    /*!
     * LW2080_CTRL_BOARDOBJGRP_E32 super class. Must always be first object in
     * structure.
     */
    LW2080_CTRL_BOARDOBJGRP_E32       super;

    /*!
     * Clock Propagation Topology HAL.
     * @ref LW2080_CTRL_CLK_CLK_PROP_TOP_HAL_<xyz>
     */
    LwU8                              topHal;

    /*!
     * Array of CLK_PROP_TOP structures. Has valid indexes corresponding to the
     * bits set in @ref super.objMask.
     */
    LW2080_CTRL_CLK_CLK_PROP_TOP_INFO propTops[LW2080_CTRL_BOARDOBJGRP_E32_MAX_OBJECTS];
} LW2080_CTRL_CLK_CLK_PROP_TOPS_INFO;
typedef struct LW2080_CTRL_CLK_CLK_PROP_TOPS_INFO *PLW2080_CTRL_CLK_CLK_PROP_TOPS_INFO;

/*!
 * LW2080_CTRL_CMD_CLK_CLK_PROP_TOPS_GET_INFO
 *
 * This command returns CLK_PROP_TOPS static object information/POR as specified
 * by the VBIOS in the Clocks VF Relationships Table.
 *
 * The CLK_PROP_TOP objects are indexed per how they are specified in the VBIOS
 * table.
 *
 * See @ref LW2080_CTRL_CLK_CLK_PROP_TOPS_INFO for documentation on the
 * parameters.
 *
 * Return values are specified per @ref BoardObjGrpGetInfo.
 */
#define LW2080_CTRL_CMD_CLK_CLK_PROP_TOPS_GET_INFO (0x2080107d) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_CLK_INTERFACE_ID << 8) | LW2080_CTRL_CLK_CLK_PROP_TOPS_INFO_MESSAGE_ID" */

/*!
 * Structure describing CLK_PROP_TOP dynamic information. Implements the
 * BOARDOBJ model/interface.
 */
typedef struct LW2080_CTRL_CLK_CLK_PROP_TOP_STATUS {
    /*!
     * LW2080_CTRL_BOARDOBJ super class. Must always be first object in
     * structure.
     */
    LW2080_CTRL_BOARDOBJ super;
} LW2080_CTRL_CLK_CLK_PROP_TOP_STATUS;
typedef struct LW2080_CTRL_CLK_CLK_PROP_TOP_STATUS *PLW2080_CTRL_CLK_CLK_PROP_TOP_STATUS;

/*!
 * Structure describing CLK_PROP_TOPS dynamic information. Implements the
 * BOARDOBJGRP model/interface.
 */
#define LW2080_CTRL_CLK_CLK_PROP_TOPS_STATUS_MESSAGE_ID (0x7EU)

typedef struct LW2080_CTRL_CLK_CLK_PROP_TOPS_STATUS {
    /*!
     * LW2080_CTRL_BOARDOBJGRP_E32 super class.  Must always be first object in
     * structure.
     */
    LW2080_CTRL_BOARDOBJGRP_E32         super;

    /*!
     * Active Clock Propagation Topology Id.
     * SW will dynamically select the active topology from set of available
     * topologies based on the active workload hints coming from KMD/DX.
     */
    LwU8                                activeTopId;

    /*!
     * Array of CLK_PROP_TOP structures.  Has valid indexes corresponding to the
     * bits set in @ref super.objMask.
     */
    LW2080_CTRL_CLK_CLK_PROP_TOP_STATUS propTops[LW2080_CTRL_BOARDOBJGRP_E32_MAX_OBJECTS];
} LW2080_CTRL_CLK_CLK_PROP_TOPS_STATUS;
typedef struct LW2080_CTRL_CLK_CLK_PROP_TOPS_STATUS *PLW2080_CTRL_CLK_CLK_PROP_TOPS_STATUS;

/*!
 * LW2080_CTRL_CMD_CLK_CLK_PROP_TOPS_GET_STATUS
 *
 * This command returns the CLK_PROP_TOPS dynamic state information associated by the
 * Clk Programming functionality
 *
 * See @ref LW2080_CTRL_CLK_CLK_PROP_TOPS_STATUS for documentation on the
 * parameters.
 *
 * Return values are specified per @ref BoardObjGrpGetStatus.
 */
#define LW2080_CTRL_CMD_CLK_CLK_PROP_TOPS_GET_STATUS (0x2080107e) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_CLK_INTERFACE_ID << 8) | LW2080_CTRL_CLK_CLK_PROP_TOPS_STATUS_MESSAGE_ID" */

/*!
 * Structure describing CLK_PROP_TOP control params. Implements the
 * BOARDOBJ model/interface.
 */
typedef struct LW2080_CTRL_CLK_CLK_PROP_TOP_CONTROL {
    /*!
     * LW2080_CTRL_BOARDOBJ super class.  Must always be first object in
     * structure.
     */
    LW2080_CTRL_BOARDOBJ super;
} LW2080_CTRL_CLK_CLK_PROP_TOP_CONTROL;
typedef struct LW2080_CTRL_CLK_CLK_PROP_TOP_CONTROL *PLW2080_CTRL_CLK_CLK_PROP_TOP_CONTROL;

/*!
 * Structure describing CLK_PROP_TOPS control params. Implements the
 * BOARDOBJGRP model/interface.
 */
typedef struct LW2080_CTRL_CLK_CLK_PROP_TOPS_CONTROL {
    /*!
     * LW2080_CTRL_BOARDOBJGRP_E32 super class. Must always be first object in
     * structure.
     */
    LW2080_CTRL_BOARDOBJGRP_E32          super;

    /*!
     * Forced active Clock Propagation Topology Id.
     * When client force an active topology id, the SW will respect the forced
     * topology instead of selecting the topology based on workload.
     *
     * Logic:
     * Use forced active topology if @ref activeTopIdForced == VALID
     * otherwise use @ref activeTopId
     */
    LwU8                                 activeTopIdForced;

    /*!
     * Array of CLK_PROP_TOP structures. Has valid indexes corresponding to the
     * bits set in @ref super.objMask.
     */
    LW2080_CTRL_CLK_CLK_PROP_TOP_CONTROL propTops[LW2080_CTRL_BOARDOBJGRP_E32_MAX_OBJECTS];
} LW2080_CTRL_CLK_CLK_PROP_TOPS_CONTROL;
typedef struct LW2080_CTRL_CLK_CLK_PROP_TOPS_CONTROL *PLW2080_CTRL_CLK_CLK_PROP_TOPS_CONTROL;

/*!
 * LW2080_CTRL_CMD_CLK_CLK_PROP_TOPS_GET_CONTROL
 *
 * This command returns CLK_PROP_TOPS control parameters as specified
 * by the VBIOS in the Clocks Propagation Topology Table.
 *
 * The CLK_PROP_TOP objects are indexed per how they are specified in the VBIOS
 * table.
 *
 * See @ref LW2080_CTRL_CLK_CLK_PROP_TOPS_CONTROL for documentation on the
 * parameters.
 *
 * Return values are specified per @ref BoardObjGrpGetControl.
 */
#define LW2080_CTRL_CMD_CLK_CLK_PROP_TOPS_GET_CONTROL                            (0x2080107f) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_CLK_INTERFACE_ID << 8) | 0x7F" */

/*!
 * LW2080_CTRL_CMD_CLK_CLK_PROP_TOPS_SET_CONTROL
 *
 * This command accepts client-specified control parameters for a set
 * of CLK_PROP_TOPS entries in the Clocks Propagation Topology Table,
 * and applies these new parameters to the set of CLK_PROP_TOPS entries.
 *
 * The CLK_PROP_TOP objects are indexed per how they are specified in the VBIOS
 * table.
 *
 * See @ref LW2080_CTRL_CLK_CLK_PROP_TOPS_CONTROL for documentation on the
 * parameters.
 *
 * Return values are specified per @ref BoardObjGrpSetControl.
 */
#define LW2080_CTRL_CMD_CLK_CLK_PROP_TOPS_SET_CONTROL                            (0x20801080) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_CLK_INTERFACE_ID << 8) | 0x80" */


/*!
 * Enumeration of CLIENT_CLK_PROP_TOP_POL class types.
 */
#define LW2080_CTRL_CLK_CLIENT_CLK_PROP_TOP_POL_TYPE_1X                          0x00U
#define LW2080_CTRL_CLK_CLIENT_CLK_PROP_TOP_POL_TYPE_1X_SLIDER                   0x01U


/*!
 * Enumerations of the CLIENT_CLK_PROP_TOP_POL HALs, as specified by the
 * VBIOS Propagation Topology Policy Table.
 *
 * https://confluence.lwpu.com/display/RMPER/Clock+Propagation+Tables+Spec#ClockPropagationTablesSpec-ClockPropagationTopologyHAL
 */
#define LW2080_CTRL_CLK_CLIENT_CLK_PROP_TOP_POL_HAL_GA10X                        0x00U

/*!
 * Enumeration of clock propagation topology policy ids.
 */
#define LW2080_CTRL_CLK_CLIENT_CLK_PROP_TOP_POL_ID_ILWALID                       LW_U8_MAX
#define LW2080_CTRL_CLK_CLIENT_CLK_PROP_TOP_POL_ID_GRAPHICS_MEMORY               0x00U
#define LW2080_CTRL_CLK_CLIENT_CLK_PROP_TOP_POL_ID_MAX                           0x01U

/*!
 * Macro representing invalid slider point index.
 */
#define LW2080_CTRL_CLK_CLIENT_CLK_PROP_TOP_POL_SLIDER_POINT_IDX_ILWALID         LW_U8_MAX

/*!
 * Macro representing max allowed slider points.
 */
#define LW2080_CTRL_CLK_CLIENT_CLK_PROP_TOP_POL_SLIDER_POINT_IDX_MAX             0x10U

/*!
 * Macro representing name associated with the slider point.
 */
#define LW2080_CTRL_CLK_CLIENT_CLK_PROP_TOP_POL_SLIDER_POINT_NAME_BASE           0x00U
#define LW2080_CTRL_CLK_CLIENT_CLK_PROP_TOP_POL_SLIDER_POINT_NAME_EXTREME        0x01U
#define LW2080_CTRL_CLK_CLIENT_CLK_PROP_TOP_POL_SLIDER_POINT_NAME_INTERMEDIATE_1 0x02U
#define LW2080_CTRL_CLK_CLIENT_CLK_PROP_TOP_POL_SLIDER_POINT_NAME_INTERMEDIATE_2 0x03U
#define LW2080_CTRL_CLK_CLIENT_CLK_PROP_TOP_POL_SLIDER_POINT_NAME_INTERMEDIATE_3 0x04U
#define LW2080_CTRL_CLK_CLIENT_CLK_PROP_TOP_POL_SLIDER_POINT_NAME_ILWALID        LW_U8_MAX

/*!
 * Structure describing CLIENT_CLK_PROP_TOP_POL_1X_SLIDER_POINT static information/POR.
 */
typedef struct LW2080_CTRL_CLK_CLIENT_CLK_PROP_TOP_POL_1X_SLIDER_POINT {
    /*!
     * Name associated with this point.
     * @ref LW2080_CTRL_CLK_CLIENT_CLK_PROP_TOP_POL_SLIDER_POINT_NAME_<xyz>
     */
    LwU8 name;
} LW2080_CTRL_CLK_CLIENT_CLK_PROP_TOP_POL_1X_SLIDER_POINT;

/*!
 * Structure describing CLIENT_CLK_PROP_TOP_POL_1X_SLIDER static information/POR.
 * Implements the BOARDOBJ model/interface.
 */
typedef struct LW2080_CTRL_CLK_CLIENT_CLK_PROP_TOP_POL_1X_SLIDER_INFO {
    /*!
     * Total number of discrete points supported for this policy slider.
     */
    LwU8                                                    numPoints;

    /*!
     * Default POR point index on this slider.
     */
    LwU8                                                    defaultPoint;

    /*!
     * Array of points with associated POR information.
     */
    LW2080_CTRL_CLK_CLIENT_CLK_PROP_TOP_POL_1X_SLIDER_POINT points[LW2080_CTRL_CLK_CLIENT_CLK_PROP_TOP_POL_SLIDER_POINT_IDX_MAX];
} LW2080_CTRL_CLK_CLIENT_CLK_PROP_TOP_POL_1X_SLIDER_INFO;

/*!
 * CLIENT_CLK_PROP_TOP_POL type-specific data union. Discriminated by
 * CLIENT_CLK_PROP_TOP_POL::super.type.
 */


/*!
 * Structure describing CLIENT_CLK_PROP_TOP_POL static information/POR. Implements the
 * BOARDOBJ model/interface.
 */
typedef struct LW2080_CTRL_CLK_CLIENT_CLK_PROP_TOP_POL_INFO {
    /*!
     * LW2080_CTRL_BOARDOBJ super class.  Must always be first object in
     * structure.
     */
    LW2080_CTRL_BOARDOBJ super;

    /*!
     * XAPI does not support using an element within another structure as a
     * discriminant.  Placing this redundant type value here until that design
     * constraint can be fixed.
     */
    LwU8                 type;

    /*!
     * Client Clock Propagation Topology Policy Id.
     * @ ref LW2080_CTRL_CLK_CLIENT_CLK_PROP_TOP_POL_ID_MAX
     */
    LwU8                 topPolId;

    /*!
     * Type-specific data union.
     */
    union {
        LW2080_CTRL_CLK_CLIENT_CLK_PROP_TOP_POL_1X_SLIDER_INFO v1xSlider;
    } data;
} LW2080_CTRL_CLK_CLIENT_CLK_PROP_TOP_POL_INFO;

/*!
 * Structure describing CLIENT_CLK_PROP_TOP_POLS static information/POR. Implements the
 * BOARDOBJGRP model/interface.
 */
#define LW2080_CTRL_CLK_CLIENT_CLK_PROP_TOP_POLS_INFO_MESSAGE_ID (0x85U)

typedef struct LW2080_CTRL_CLK_CLIENT_CLK_PROP_TOP_POLS_INFO {
    /*!
     * LW2080_CTRL_BOARDOBJGRP_E32 super class. Must always be first object in
     * structure.
     */
    LW2080_CTRL_BOARDOBJGRP_E32                  super;

    /*!
     * Clock Propagation Topology Policy HAL.
     * @ref LW2080_CTRL_CLK_CLIENT_CLK_PROP_TOP_POL_HAL_<xyz>
     */
    LwU8                                         topPolHal;

    /*!
     * Array of CLIENT_CLK_PROP_TOP_POL structures. Has valid indexes corresponding to the
     * bits set in @ref super.objMask.
     */
    LW2080_CTRL_CLK_CLIENT_CLK_PROP_TOP_POL_INFO propTopPols[LW2080_CTRL_BOARDOBJGRP_E32_MAX_OBJECTS];
} LW2080_CTRL_CLK_CLIENT_CLK_PROP_TOP_POLS_INFO;

/*!
 * LW2080_CTRL_CMD_CLK_CLIENT_CLK_PROP_TOP_POLS_GET_INFO
 *
 * This command returns CLIENT_CLK_PROP_TOP_POLS static object information/POR as specified
 * by the VBIOS in the Client Clocks Propagation Topology Policy Table.
 *
 * The CLIENT_CLK_PROP_TOP_POL objects are indexed per how they are specified in the VBIOS
 * table.
 *
 * See @ref LW2080_CTRL_CLK_CLIENT_CLK_PROP_TOP_POLS_INFO for documentation on the
 * parameters.
 *
 * Return values are specified per @ref BoardObjGrpGetInfo.
 */
#define LW2080_CTRL_CMD_CLK_CLIENT_CLK_PROP_TOP_POLS_GET_INFO (0x20801085) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_CLK_INTERFACE_ID << 8) | LW2080_CTRL_CLK_CLIENT_CLK_PROP_TOP_POLS_INFO_MESSAGE_ID" */

/*!
 * Structure describing CLIENT_CLK_PROP_TOP_POL_1X_SLIDER control params. Implements the
 * BOARDOBJ model/interface.
 */
typedef struct LW2080_CTRL_CLK_CLIENT_CLK_PROP_TOP_POL_1X_SLIDER_CONTROL {
    /*!
     * Client chosen point index on this slider.
     *
     * Each policy exposes set of available points from which client could
     * choose a single point. The internal SW arbitration logic will determine
     * final active topology based on all actively chosen point from all
     * active policys.
     *
     * Logic:
     * Use chosen point index if
     * @ref chosenPoint != LW2080_CTRL_CLK_CLIENT_CLK_PROP_TOP_POL_SLIDER_POINT_IDX_ILWALID
     */
    LwU8 chosenPoint;
} LW2080_CTRL_CLK_CLIENT_CLK_PROP_TOP_POL_1X_SLIDER_CONTROL;

/*!
 * CLIENT_CLK_PROP_TOP_POL type-specific data union. Discriminated by
 * CLIENT_CLK_PROP_TOP_POL::super.type.
 */


/*!
 * Structure describing CLIENT_CLK_PROP_TOP_POL control params. Implements the
 * BOARDOBJ model/interface.
 */
typedef struct LW2080_CTRL_CLK_CLIENT_CLK_PROP_TOP_POL_CONTROL {
    /*!
     * LW2080_CTRL_BOARDOBJ super class.  Must always be first object in
     * structure.
     */
    LW2080_CTRL_BOARDOBJ super;

    /*!
     * XAPI does not support using an element within another structure as a
     * discriminant.  Placing this redundant type value here until that design
     * constraint can be fixed.
     */
    LwU8                 type;

    /*!
     * Type-specific data union.
     */
    union {
        LW2080_CTRL_CLK_CLIENT_CLK_PROP_TOP_POL_1X_SLIDER_CONTROL v1xSlider;
    } data;
} LW2080_CTRL_CLK_CLIENT_CLK_PROP_TOP_POL_CONTROL;

/*!
 * Structure describing CLIENT_CLK_PROP_TOP_POLS control params. Implements the
 * BOARDOBJGRP model/interface.
 */
typedef struct LW2080_CTRL_CLK_CLIENT_CLK_PROP_TOP_POLS_CONTROL {
    /*!
     * LW2080_CTRL_BOARDOBJGRP_E32 super class. Must always be first object in
     * structure.
     */
    LW2080_CTRL_BOARDOBJGRP_E32                     super;

    /*!
     * Array of CLIENT_CLK_PROP_TOP_POL structures. Has valid indexes corresponding to the
     * bits set in @ref super.objMask.
     */
    LW2080_CTRL_CLK_CLIENT_CLK_PROP_TOP_POL_CONTROL propTopPols[LW2080_CTRL_BOARDOBJGRP_E32_MAX_OBJECTS];
} LW2080_CTRL_CLK_CLIENT_CLK_PROP_TOP_POLS_CONTROL;

/*!
 * LW2080_CTRL_CMD_CLK_CLIENT_CLK_PROP_TOP_POLS_GET_CONTROL
 *
 * This command returns CLIENT_CLK_PROP_TOP_POLS control parameters as specified
 * by the VBIOS in the Client Clocks Propagation Topology Policy Table.
 *
 * The CLIENT_CLK_PROP_TOP_POL objects are indexed per how they are specified in the VBIOS
 * table.
 *
 * See @ref LW2080_CTRL_CLK_CLIENT_CLK_PROP_TOP_POLS_CONTROL for documentation on the
 * parameters.
 *
 * Return values are specified per @ref BoardObjGrpGetControl.
 */
#define LW2080_CTRL_CMD_CLK_CLIENT_CLK_PROP_TOP_POLS_GET_CONTROL (0x20801087) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_CLK_INTERFACE_ID << 8) | 0x87" */

/*!
 * LW2080_CTRL_CMD_CLK_CLIENT_CLK_PROP_TOP_POLS_SET_CONTROL
 *
 * This command accepts client-specified control parameters for a set
 * of CLIENT_CLK_PROP_TOP_POLS entries in the Client Clocks Propagation Topology Policy
 * Table and applies these new parameters to the set of CLIENT_CLK_PROP_TOP_POLS entries.
 *
 * The CLIENT_CLK_PROP_TOP_POL objects are indexed per how they are specified in the VBIOS
 * table.
 *
 * See @ref LW2080_CTRL_CLK_CLIENT_CLK_PROP_TOP_POLS_CONTROL for documentation on the
 * parameters.
 *
 * Return values are specified per @ref BoardObjGrpSetControl.
 */
#define LW2080_CTRL_CMD_CLK_CLIENT_CLK_PROP_TOP_POLS_SET_CONTROL (0x20801088) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_CLK_INTERFACE_ID << 8) | 0x88" */

/*!
 * Enumeration of CLK_PROP_TOP_REL class types.
 */
#define LW2080_CTRL_CLK_CLK_PROP_TOP_REL_TYPE_1X                 0x00U
#define LW2080_CTRL_CLK_CLK_PROP_TOP_REL_TYPE_1X_RATIO           0x01U
#define LW2080_CTRL_CLK_CLK_PROP_TOP_REL_TYPE_1X_TABLE           0x02U
#define LW2080_CTRL_CLK_CLK_PROP_TOP_REL_TYPE_1X_VOLT            0x03U
#define LW2080_CTRL_CLK_CLK_PROP_TOP_REL_TYPE_1X_VFE             0x04U

/*!
 * Structure describing CLK_PROP_TOP_REL_1X_RATIO static information/POR. Implements the
 * BOARDOBJ model/interface.
 */
typedef struct LW2080_CTRL_CLK_CLK_PROP_TOP_REL_1X_RATIO_INFO {
    /*!
     * Ratio Relationship between source and destination. (unsigned percentage)
     *
     * @note - To callwlate reciprocal of an FXP X.Y number without losing any
     *         precision, the reciprocal will be of size (1+Y).X. While VBIOS
     *         can store number as 4.12, internally we should store as 16.16 so
     *         we have enough precision to store both ratio and its reciprocal
     *         in the same format w/o any potential loss of precision, and with
     *         same code to compute in either direction.
     */
    LwUFXP16_16 ratio;

    /*!
     * Ilwerse value of @ref ratio which will be callwlated and used by SW
     * if @ref bBiDirectional is TRUE.
     */
    LwUFXP16_16 ratioIlwerse;
} LW2080_CTRL_CLK_CLK_PROP_TOP_REL_1X_RATIO_INFO;
typedef struct LW2080_CTRL_CLK_CLK_PROP_TOP_REL_1X_RATIO_INFO *PLW2080_CTRL_CLK_CLK_PROP_TOP_REL_1X_RATIO_INFO;

/*!
 * Structure describing CLK_PROP_TOP_REL_1X_TABLE static information/POR. Implements the
 * BOARDOBJ model/interface.
 */
typedef struct LW2080_CTRL_CLK_CLK_PROP_TOP_REL_1X_TABLE_INFO {
    /*!
     * First array index into table relationship array.
     */
    LwU8 tableRelIdxFirst;
    /*!
     * Lirst array index into table relationship array.
     */
    LwU8 tableRelIdxLast;
} LW2080_CTRL_CLK_CLK_PROP_TOP_REL_1X_TABLE_INFO;
typedef struct LW2080_CTRL_CLK_CLK_PROP_TOP_REL_1X_TABLE_INFO *PLW2080_CTRL_CLK_CLK_PROP_TOP_REL_1X_TABLE_INFO;

/*!
 * Structure describing CLK_PROP_TOP_REL_1X_VOLT static information/POR. Implements the
 * BOARDOBJ model/interface.
 */
typedef struct LW2080_CTRL_CLK_CLK_PROP_TOP_REL_1X_VOLT_INFO {
    /*!
     * Voltage Rail Index.
     */
    LwU8 voltRailIdx;
} LW2080_CTRL_CLK_CLK_PROP_TOP_REL_1X_VOLT_INFO;
typedef struct LW2080_CTRL_CLK_CLK_PROP_TOP_REL_1X_VOLT_INFO *PLW2080_CTRL_CLK_CLK_PROP_TOP_REL_1X_VOLT_INFO;

/*!
 * Structure describing CLK_PROP_TOP_REL_1X_VFE static information/POR. Implements the
 * BOARDOBJ model/interface.
 */
typedef struct LW2080_CTRL_CLK_CLK_PROP_TOP_REL_1X_VFE_INFO {
    /*!
     * VFE Equation Index.
     */
    LW2080_CTRL_PERF_VFE_EQU_IDX vfeIdx;
} LW2080_CTRL_CLK_CLK_PROP_TOP_REL_1X_VFE_INFO;
typedef struct LW2080_CTRL_CLK_CLK_PROP_TOP_REL_1X_VFE_INFO *PLW2080_CTRL_CLK_CLK_PROP_TOP_REL_1X_VFE_INFO;

/*!
 * CLK_PROP_TOP_REL type-specific data union. Discriminated by
 * CLK_PROP_TOP_REL::super.type.
 */


/*!
 * Structure describing CLK_PROP_TOP_REL static information/POR. Implements the
 * BOARDOBJ model/interface.
 */
typedef struct LW2080_CTRL_CLK_CLK_PROP_TOP_REL_INFO {
    /*!
     * LW2080_CTRL_BOARDOBJ super class.  Must always be first object in
     * structure.
     */
    LW2080_CTRL_BOARDOBJ super;

    /*!
     * XAPI does not support using an element within another structure as a
     * discriminant.  Placing this redundant type value here until that design
     * constraint can be fixed.
     */
    LwU8                 type;

    /*!
     * Source Clock Domain Index.
     */
    LwU8                 clkDomainIdxSrc;

    /*!
     * Destination Clock Domain Index.
     */
    LwU8                 clkDomainIdxDst;

    /*!
     * Boolean tracking whether bidirectional relationship enabled.
     */
    LwBool               bBiDirectional;

    /*!
     * Type-specific data union.
     */
    union {
        LW2080_CTRL_CLK_CLK_PROP_TOP_REL_1X_RATIO_INFO ratio;
        LW2080_CTRL_CLK_CLK_PROP_TOP_REL_1X_TABLE_INFO table;
        LW2080_CTRL_CLK_CLK_PROP_TOP_REL_1X_VOLT_INFO  volt;
        LW2080_CTRL_CLK_CLK_PROP_TOP_REL_1X_VFE_INFO   vfe;
    } data;
} LW2080_CTRL_CLK_CLK_PROP_TOP_REL_INFO;
typedef struct LW2080_CTRL_CLK_CLK_PROP_TOP_REL_INFO *PLW2080_CTRL_CLK_CLK_PROP_TOP_REL_INFO;

/*!
 * Structure describing tuple of source and destination frequency table for
 * table based clock propagation topology relationship.
 *
 * Clock Propagation Topology Relationship header will store the array of
 * this tuple indexed by table based relationship objects.
 */
typedef struct LW2080_CTRL_CLK_CLK_PROP_TOP_REL_TABLE_REL_TUPLE {
    /*!
     * Source Clock Domain frequency in MHz.
     */
    LwU16 freqMHzSrc;

    /*!
     * Destination Clock Domain frequency in MHz.
     */
    LwU16 freqMHzDst;
} LW2080_CTRL_CLK_CLK_PROP_TOP_REL_TABLE_REL_TUPLE;
typedef struct LW2080_CTRL_CLK_CLK_PROP_TOP_REL_TABLE_REL_TUPLE *PLW2080_CTRL_CLK_CLK_PROP_TOP_REL_TABLE_REL_TUPLE;

/*!
 * Macro defining the max allowed table relationship tuple array entries.
 */
#define LW2080_CTRL_CLK_CLK_PROP_TOP_REL_TABLE_REL_TUPLE_MAX 32

/*!
 * Structure describing CLK_PROP_TOP_RELS static information/POR. Implements the
 * BOARDOBJGRP model/interface.
 */
#define LW2080_CTRL_CLK_CLK_PROP_TOP_RELS_INFO_MESSAGE_ID (0x81U)

typedef struct LW2080_CTRL_CLK_CLK_PROP_TOP_RELS_INFO {
    /*!
     * LW2080_CTRL_BOARDOBJGRP_E255 super class. Must always be first object in
     * structure.
     */
    LW2080_CTRL_BOARDOBJGRP_E255                     super;

    /*!
     * Count of valid table relationship tuple array entries.
     */
    LwU8                                             tableRelTupleCount;

    /*!
     * Array of frequency tuple for table based clock propagation topology relationships.
     * Here valid indexes corresponds to [0, tableRelTupleCount]
     */
    LW2080_CTRL_CLK_CLK_PROP_TOP_REL_TABLE_REL_TUPLE tableRelTuple[LW2080_CTRL_CLK_CLK_PROP_TOP_REL_TABLE_REL_TUPLE_MAX];

    /*!
     * Array of CLK_PROP_TOP_REL structures. Has valid indexes corresponding to the
     * bits set in @ref super.objMask.
     */
    LW2080_CTRL_CLK_CLK_PROP_TOP_REL_INFO            propTopRels[LW2080_CTRL_BOARDOBJGRP_E255_MAX_OBJECTS];
} LW2080_CTRL_CLK_CLK_PROP_TOP_RELS_INFO;
typedef struct LW2080_CTRL_CLK_CLK_PROP_TOP_RELS_INFO *PLW2080_CTRL_CLK_CLK_PROP_TOP_RELS_INFO;

/*!
 * LW2080_CTRL_CMD_CLK_CLK_PROP_TOP_RELS_GET_INFO
 *
 * This command returns CLK_PROP_TOP_RELS static object information/POR as specified
 * by the VBIOS in the Clocks VF Relationships Table.
 *
 * The CLK_PROP_TOP_REL objects are indexed per how they are specified in the VBIOS
 * table.
 *
 * See @ref LW2080_CTRL_CLK_CLK_PROP_TOP_RELS_INFO for documentation on the
 * parameters.
 *
 * Return values are specified per @ref BoardObjGrpGetInfo.
 */
#define LW2080_CTRL_CMD_CLK_CLK_PROP_TOP_RELS_GET_INFO (0x20801081) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_CLK_INTERFACE_ID << 8) | LW2080_CTRL_CLK_CLK_PROP_TOP_RELS_INFO_MESSAGE_ID" */

/*!
 * Structure describing CLK_PROP_TOP_REL_1X_RATIO control params. Implements the
 * BOARDOBJ model/interface.
 */
typedef struct LW2080_CTRL_CLK_CLK_PROP_TOP_REL_1X_RATIO_CONTROL {
    /*!
     * Ratio Relationship between source and destination. (unsigned percentage)
     */
    LwUFXP16_16 ratio;
} LW2080_CTRL_CLK_CLK_PROP_TOP_REL_1X_RATIO_CONTROL;
typedef struct LW2080_CTRL_CLK_CLK_PROP_TOP_REL_1X_RATIO_CONTROL *PLW2080_CTRL_CLK_CLK_PROP_TOP_REL_1X_RATIO_CONTROL;

/*!
 * CLK_PROP_TOP_REL type-specific data union. Discriminated by
 * CLK_PROP_TOP_REL::super.type.
 */


/*!
 * Structure describing CLK_PROP_TOP_REL control params. Implements the
 * BOARDOBJ model/interface.
 */
typedef struct LW2080_CTRL_CLK_CLK_PROP_TOP_REL_CONTROL {
    /*!
     * LW2080_CTRL_BOARDOBJ super class.  Must always be first object in
     * structure.
     */
    LW2080_CTRL_BOARDOBJ super;

    /*!
     * XAPI does not support using an element within another structure as a
     * discriminant.  Placing this redundant type value here until that design
     * constraint can be fixed.
     */
    LwU8                 type;

    /*!
     * Type-specific data union.
     */
    union {
        LW2080_CTRL_CLK_CLK_PROP_TOP_REL_1X_RATIO_CONTROL ratio;
    } data;
} LW2080_CTRL_CLK_CLK_PROP_TOP_REL_CONTROL;
typedef struct LW2080_CTRL_CLK_CLK_PROP_TOP_REL_CONTROL *PLW2080_CTRL_CLK_CLK_PROP_TOP_REL_CONTROL;

/*!
 * Structure describing CLK_PROP_TOP_RELS control params. Implements the
 * BOARDOBJGRP model/interface.
 */
typedef struct LW2080_CTRL_CLK_CLK_PROP_TOP_RELS_CONTROL {
    /*!
     * LW2080_CTRL_BOARDOBJGRP_E255 super class. Must always be first object in
     * structure.
     */
    LW2080_CTRL_BOARDOBJGRP_E255             super;

    /*!
     * Array of CLK_PROP_TOP_REL structures. Has valid indexes corresponding to the
     * bits set in @ref super.objMask.
     */
    LW2080_CTRL_CLK_CLK_PROP_TOP_REL_CONTROL propTopRels[LW2080_CTRL_BOARDOBJGRP_E255_MAX_OBJECTS];
} LW2080_CTRL_CLK_CLK_PROP_TOP_RELS_CONTROL;
typedef struct LW2080_CTRL_CLK_CLK_PROP_TOP_RELS_CONTROL *PLW2080_CTRL_CLK_CLK_PROP_TOP_RELS_CONTROL;

/*!
 * LW2080_CTRL_CMD_CLK_CLK_PROP_TOP_RELS_GET_CONTROL
 *
 * This command returns CLK_PROP_TOP_RELS control parameters as specified
 * by the VBIOS in the Clocks Propagation Topology Relationship Table.
 *
 * The CLK_PROP_TOP_REL objects are indexed per how they are specified in the VBIOS
 * table.
 *
 * See @ref LW2080_CTRL_CLK_CLK_PROP_TOP_RELS_CONTROL for documentation on the
 * parameters.
 *
 * Return values are specified per @ref BoardObjGrpGetControl.
 */
#define LW2080_CTRL_CMD_CLK_CLK_PROP_TOP_RELS_GET_CONTROL     (0x20801083) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_CLK_INTERFACE_ID << 8) | 0x83" */

/*!
 * LW2080_CTRL_CMD_CLK_CLK_PROP_TOP_RELS_SET_CONTROL
 *
 * This command accepts client-specified control parameters for a set
 * of CLK_PROP_TOP_RELS entries in the Clocks Propagation Top Rel Table,
 * and applies these new parameters to the set of CLK_PROP_TOP_RELS entries.
 *
 * The CLK_PROP_TOP_REL objects are indexed per how they are specified in the VBIOS
 * table.
 *
 * See @ref LW2080_CTRL_CLK_CLK_PROP_TOP_RELS_CONTROL for documentation on the
 * parameters.
 *
 * Return values are specified per @ref BoardObjGrpSetControl.
 */
#define LW2080_CTRL_CMD_CLK_CLK_PROP_TOP_RELS_SET_CONTROL     (0x20801084) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_CLK_INTERFACE_ID << 8) | 0x84" */

/*!
 * Enumeration of clients which can disable/enable a given voltage controller instance.
 */
#define LW2080_CTRL_CLK_CLK_VOLT_CONTROLLER_CLIENT_ID_API     (0x00U)
#define LW2080_CTRL_CLK_CLK_VOLT_CONTROLLER_CLIENT_ID_LPWR    (0x01U)
#define LW2080_CTRL_CLK_CLK_VOLT_CONTROLLER_CLIENT_ID_INIT    (0x02U)

/*!
 * Enumeration of clients which can disable/enable voltage controllers group
 */
#define LW2080_CTRL_CLK_CLK_VOLT_CONTROLLERS_CLIENT_ID_API    (0x00U)
#define LW2080_CTRL_CLK_CLK_VOLT_CONTROLLERS_CLIENT_ID_REGKEY (0x01U)

/*!
 * Enumeration of CLK_VOLT_CONTROLLER class types.
 */
#define LW2080_CTRL_CLK_CLK_VOLT_CONTROLLER_TYPE_DISABLED     (0x00U)
#define LW2080_CTRL_CLK_CLK_VOLT_CONTROLLER_TYPE_PROP         (0x01U)

/*!
 * Macro representing an INVALID/UNSUPPORTED CLK_VOLT_CONTROLLER index.
 */
#define LW2080_CTRL_CLK_CLK_VOLT_CONTROLLER_IDX_ILWALID       LW2080_CTRL_BOARDOBJ_IDX_ILWALID_8BIT

/*!
 * Structure describing the static configuration/POR state of the _PROP class.
 */
typedef struct LW2080_CTRL_CLK_CLK_VOLT_CONTROLLER_INFO_PROP {
    /*!
     * Absolute value of Positive Voltage Hysteresis in uV (0 => no hysteresis).
     * (hysteresis to apply when voltage has positive delta)
     */
    LwS16       voltHysteresisPositive;
    /*!
     * Absolute value of Negative Voltage Hysteresis in uV (0 => no hysteresis).
     * (hysteresis to apply when voltage has negative delta)
     */
    LwS16       voltHysteresisNegative;
    /*!
     * Proportional gain for this voltage controller in uV
     */
    LwSFXP20_12 propGain;
} LW2080_CTRL_CLK_CLK_VOLT_CONTROLLER_INFO_PROP;
typedef struct LW2080_CTRL_CLK_CLK_VOLT_CONTROLLER_INFO_PROP *PLW2080_CTRL_CLK_CLK_VOLT_CONTROLLER_INFO_PROP;

/*!
 * CLK_VOLT_CONTROLLER type-specific data union. Discriminated by
 * CLK_VOLT_CONTROLLER::super.type.
 */


/*!
 * Structure describing CLK_VOLT_CONTROLLER static information/POR.
 * Implements the BOARDOBJ model/interface.
 */
typedef struct LW2080_CTRL_CLK_CLK_VOLT_CONTROLLER_INFO {
    /*!
     * LW2080_CTRL_BOARDOBJ super class.  Must always be first object in
     * structure.
     */
    LW2080_CTRL_BOARDOBJ super;
    /*!
     * XAPI does not support using an element within another structure as a
     * discriminant.  Placing this redundant type value here until that design
     * constraint can be fixed.
     */
    LwU8                 type;
    /*!
     * Voltage Rail Index corresponding to a VOLT_RAIL.
     */
    LwU8                 voltRailIdx;
    /*!
     * ADC operating mode @ref LW2080_CTRL_CLK_CLK_VOLT_CONTROLLER_ADC_MODE_<XYZ>
     */
    LwU8                 adcMode;
    /*!
     * Index into the Thermal Monitor Table. To be used for Droopy VR to poison
     * the sample per droopyPctMin value below.
     *
     * Invalid index value means that droopy support is not required and will
     * disable the poisoning behavior.
     */
    LwU8                 thermMonIdx;
    /*!
     * Minimum percentage of time droopy should engage to poison the sample.
     */
    LwUFXP4_12           droopyPctMin;
    /*!
     * Mask of ADC devices monitored by this controller instance
     */
    LwU32                adcMask;
    /*!
     * Voltage offset limit range min value (in uV).
     */
    LwS32                voltOffsetMinuV;
    /*!
     * Voltage offset limit range max value (in uV).
     */
    LwS32                voltOffsetMaxuV;
    /*!
     * Type-specific data union.
     */
    union {
        LW2080_CTRL_CLK_CLK_VOLT_CONTROLLER_INFO_PROP prop;
    } data;
} LW2080_CTRL_CLK_CLK_VOLT_CONTROLLER_INFO;
typedef struct LW2080_CTRL_CLK_CLK_VOLT_CONTROLLER_INFO *PLW2080_CTRL_CLK_CLK_VOLT_CONTROLLER_INFO;

/*!
 * Structure describing CLK_VOLT_CONTROLLERS static information/POR.
 * Implements the BOARDOBJGRP model/interface.
 */
#define LW2080_CTRL_CLK_CLK_VOLT_CONTROLLERS_GET_INFO_PARAMS_MESSAGE_ID (0x33U)

typedef struct LW2080_CTRL_CLK_CLK_VOLT_CONTROLLERS_GET_INFO_PARAMS {
    /*!
     * LW2080_CTRL_BOARDOBJGRP super class. Must always be first object in
     * structure.
     */
    LW2080_CTRL_BOARDOBJGRP_E32              super;
    /*!
     * Sleep aware sampling multiplier for winIdle
     */
    LwU8                                     lowSamplingMultiplier;
    /*!
     * Voltage offset threshold (in uV).
     * If offset <= threshold, SW will respect offset from frequency controller.
     */
    LwS32                                    voltOffsetThresholduV;
    /*!
     * Sampling period in milliseconds at which the voltage controllers will run.
     */
    LwU32                                    samplingPeriodms;
    /*!
     * Array of CLK_VOLT_CONTROLLER structures. Has valid indexes corresponding
     * to the bits set in @ref super.objMask.
     */
    LW2080_CTRL_CLK_CLK_VOLT_CONTROLLER_INFO voltControllers[LW2080_CTRL_BOARDOBJGRP_E32_MAX_OBJECTS];
} LW2080_CTRL_CLK_CLK_VOLT_CONTROLLERS_GET_INFO_PARAMS;
typedef struct LW2080_CTRL_CLK_CLK_VOLT_CONTROLLERS_GET_INFO_PARAMS *PLW2080_CTRL_CLK_CLK_VOLT_CONTROLLERS_GET_INFO_PARAMS;

/*!
 * LW2080_CTRL_CMD_CLK_CLK_VOLT_CONTROLLERS_GET_INFO
 *
 * This command returns CLK_VOLT_CONTROLLERS static object information/POR
 * as populated by the RM from the frequency controller VBIOS table.
 *
 * The CLK_VOLT_CONTROLLER objects are indexed in the order by which the RM
 * allocates them.
 *
 * See @ref LW2080_CTRL_CLK_CLK_VOLT_CONTROLLERS_GET_INFO_PARAMS for the
 * documentation on the parameters.
 *
 * Return values are specified per @ref BoardObjGrpGetInfo.
 */
#define LW2080_CTRL_CMD_CLK_CLK_VOLT_CONTROLLERS_GET_INFO (0x20801033) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_CLK_INTERFACE_ID << 8) | LW2080_CTRL_CLK_CLK_VOLT_CONTROLLERS_GET_INFO_PARAMS_MESSAGE_ID" */

/*!
 * Structure describing the dynamic state of the _PROP class.
 */
typedef struct LW2080_CTRL_CLK_CLK_VOLT_CONTROLLER_STATUS_PROP {
    /*!
     * [out] Voltage error for the current cycle (in uV).
     */
    LwS32 erroruV;
    /*!
     * [out] Sensed voltage for the current cycle (in uV).
     */
    LwU32 senseduV;
    /*!
     * [out] Measured voltage for the current cycle (in uV).
     */
    LwU32 measureduV;
} LW2080_CTRL_CLK_CLK_VOLT_CONTROLLER_STATUS_PROP;
typedef struct LW2080_CTRL_CLK_CLK_VOLT_CONTROLLER_STATUS_PROP *PLW2080_CTRL_CLK_CLK_VOLT_CONTROLLER_STATUS_PROP;

/*!
 * CLK_VOLT_CONTROLLER type-specific data union. Discriminated by
 * CLK_VOLT_CONTROLLER::super.type.
 */


/*!
 * Structure describing CLK_VOLT_CONTROLLER dynamic information
 * Implements the BOARDOBJ model/interface.
 */
typedef struct LW2080_CTRL_CLK_CLK_VOLT_CONTROLLER_STATUS {
    /*!
     * LW2080_CTRL_BOARDOBJ super class.  Must always be first object in
     * structure.
     */
    LW2080_CTRL_BOARDOBJ super;
    /*!
     * XAPI does not support using an element within another structure as a
     * discriminant.  Placing this redundant type value here until that design
     * constraint can be fixed.
     */
    LwU8                 type;
    /*!
     * [out] Voltage offset (in uV) after each iteration for this controller.
     */
    LwS32                voltOffsetuV;
    /*!
     * [out] Mask of clients requested to disable this controller.
     * @ref LW2080_CTRL_CLK_CLK_VOLT_CONTROLLER_CLIENT_ID_<xyz>.
     */
    LwU32                disableClientsMask;
    /*!
     * Type-specific data union.
     */
    union {
        LW2080_CTRL_CLK_CLK_VOLT_CONTROLLER_STATUS_PROP prop;
    } data;
} LW2080_CTRL_CLK_CLK_VOLT_CONTROLLER_STATUS;
typedef struct LW2080_CTRL_CLK_CLK_VOLT_CONTROLLER_STATUS *PLW2080_CTRL_CLK_CLK_VOLT_CONTROLLER_STATUS;

/*!
 * Structure describing CLK_VOLT_CONTROLLERS static information/POR.
 * Implements the BOARDOBJGRP model/interface.
 */
#define LW2080_CTRL_CLK_CLK_VOLT_CONTROLLERS_GET_STATUS_PARAMS_MESSAGE_ID (0x34U)

typedef struct LW2080_CTRL_CLK_CLK_VOLT_CONTROLLERS_GET_STATUS_PARAMS {
    /*!
     * LW2080_CTRL_BOARDOBJGRP super class. Must always be first object in
     * structure.
     */
    LW2080_CTRL_BOARDOBJGRP_E32                super;
    /*!
     * [out] Max voltage delta after each iteration.
     * This is the max voltage offset (in uV) per sample for a given rail.
     */
    LwS32                                      sampleMaxVoltOffsetuV[LW2080_CTRL_VOLT_VOLT_RAIL_CLIENT_MAX_RAILS];
    /*!
     * [out] Max voltage delta applied after each iteration.
     * This is the max voltage offset (in uV) applied for a given rail.
     */
    LwS32                                      totalMaxVoltOffsetuV[LW2080_CTRL_VOLT_VOLT_RAIL_CLIENT_MAX_RAILS];
    /*!
     * [out] Array of CLK_VOLT_CONTROLLER structures. Has valid indexes
     * corresponding to the bits set in @ref super.objMask.
     */
    LW2080_CTRL_CLK_CLK_VOLT_CONTROLLER_STATUS voltControllers[LW2080_CTRL_BOARDOBJGRP_E32_MAX_OBJECTS];
} LW2080_CTRL_CLK_CLK_VOLT_CONTROLLERS_GET_STATUS_PARAMS;
typedef struct LW2080_CTRL_CLK_CLK_VOLT_CONTROLLERS_GET_STATUS_PARAMS *PLW2080_CTRL_CLK_CLK_VOLT_CONTROLLERS_GET_STATUS_PARAMS;

/*!
 * LW2080_CTRL_CMD_CLK_CLK_VOLT_CONTROLLERS_GET_STATUS
 *
 * This command returns CLK_VOLT_CONTROLLERS dynamic object information from the
 * PMU.
 *
 * See @ref LW2080_CTRL_CMD_CLK_CLK_VOLT_CONTROLLERS_GET_STATUS for the
 * documentation on the parameters.
 *
 * Return values are specified per @ref BoardObjGrpGetStatus.
 */
#define LW2080_CTRL_CMD_CLK_CLK_VOLT_CONTROLLERS_GET_STATUS (0x20801034) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_CLK_INTERFACE_ID << 8) | LW2080_CTRL_CLK_CLK_VOLT_CONTROLLERS_GET_STATUS_PARAMS_MESSAGE_ID" */

/*!
 * Structure describing CLK_VOLT_CONTROLLER_PROP specific control parameters.
 */
typedef struct LW2080_CTRL_CLK_CLK_VOLT_CONTROLLER_CONTROL_PROP {
    /*!
     * Absolute value of Positive Voltage Hysteresis in uV (0 => no hysteresis).
     * (hysteresis to apply when voltage has positive delta)
     */
    LwS32       voltHysteresisPositive;
    /*!
     * Absolute value of Negative Voltage Hysteresis in uV (0 => no hysteresis).
     * (hysteresis to apply when voltage has negative delta)
     */
    LwS32       voltHysteresisNegative;
    /*!
     * Proportional gain for this voltage controller in uV
     */
    LwSFXP20_12 propGain;
} LW2080_CTRL_CLK_CLK_VOLT_CONTROLLER_CONTROL_PROP;
typedef struct LW2080_CTRL_CLK_CLK_VOLT_CONTROLLER_CONTROL_PROP *PLW2080_CTRL_CLK_CLK_VOLT_CONTROLLER_CONTROL_PROP;

/*!
 * CLK_VOLT_CONTROLLER type-specific data union.  Discriminated by
 * CLK_VOLT_CONTROLLER::super.type.
 */


/*!
 * Structure representing the control parameters associated with a CLK_VOLT_CONTROLLER.
 */
typedef struct LW2080_CTRL_CLK_CLK_VOLT_CONTROLLER_CONTROL {
    /*!
     * LW2080_CTRL_BOARDOBJ super class.  Must always be first object in
     * structure.
     */
    LW2080_CTRL_BOARDOBJ super;
    /*!
     * XAPI does not support using an element within another structure as a
     * discriminant.  Placing this redundant type value here until that design
     * constraint can be fixed.
     */
    LwU8                 type;
    /*!
     * Keep the controller disabled
     */
    LwBool               bDisable;
    /*!
     * ADC operating mode @ref LW2080_CTRL_CLK_CLK_VOLT_CONTROLLER_ADC_MODE_<XYZ>
     */
    LwU8                 adcMode;
    /*!
     * Minimum percentage of time droopy should engage to poison the sample.
     */
    LwUFXP4_12           droopyPctMin;
    /*!
     * Voltage offset limit range min value (in uV).
     */
    LwS32                voltOffsetMinuV;
    /*!
     * Voltage offset limit range max value (in uV).
     */
    LwS32                voltOffsetMaxuV;
    /*!
     * Mask of ADC devices monitored by this controller instance
     */
    LwU32                adcMask;
    /*!
     * Mask of clients requested to disable all controllers in this group.
     * @ref LW2080_CTRL_CLK_CLK_VOLT_CONTROLLER_CLIENT_ID_<xyz>.
     * If any client bit is set in the mask, all controllers in the gorup
     * will remain disabled
     */
    LwU32                disableClientsMask;
    /*!
     * Type-specific data union.
     */
    union {
        LW2080_CTRL_CLK_CLK_VOLT_CONTROLLER_CONTROL_PROP prop;
    } data;
} LW2080_CTRL_CLK_CLK_VOLT_CONTROLLER_CONTROL;
typedef struct LW2080_CTRL_CLK_CLK_VOLT_CONTROLLER_CONTROL *PLW2080_CTRL_CLK_CLK_VOLT_CONTROLLER_CONTROL;

/*!
 * Structure representing the control parameters associated with a CLK_VOLT_CONTROLLERS.
 */
typedef struct LW2080_CTRL_CLK_CLK_VOLT_CONTROLLERS_CONTROL_PARAMS {
    /*!
     * LW2080_CTRL_BOARDOBJGRP_E32 super class.  Must always be first object in
     * structure.
     */
    LW2080_CTRL_BOARDOBJGRP_E32                 super;
    /*!
     * Mask of clients requested to disable all controllers in this group.
     * @ref LW2080_CTRL_CLK_CLK_VOLT_CONTROLLERS_CLIENT_ID_<xyz>.
     * If any client bit is set in the mask, all controllers in the gorup
     * will remain disabled
     */
    LwU32                                       disableClientsMask;
    /*!
     * Array of CLK_VOLT_CONTROLLER structures.  Has valid indexes corresponding to the
     * bits set in @ref super.objMask.
     */
    LW2080_CTRL_CLK_CLK_VOLT_CONTROLLER_CONTROL voltControllers[LW2080_CTRL_BOARDOBJGRP_E32_MAX_OBJECTS];
} LW2080_CTRL_CLK_CLK_VOLT_CONTROLLERS_CONTROL_PARAMS;
typedef struct LW2080_CTRL_CLK_CLK_VOLT_CONTROLLERS_CONTROL_PARAMS *PLW2080_CTRL_CLK_CLK_VOLT_CONTROLLERS_CONTROL_PARAMS;

/*!
 * LW2080_CTRL_CMD_CLK_CLK_VOLT_CONTROLLERS_GET_CONTROL
 *
 * This command returns CLK_VOLT_CONTROLLERS control parameters as specified by
 * the VBIOS in either Voltage Controller Table.
 *
 * See @ref LW2080_CTRL_CMD_CLK_CLK_VOLT_CONTROLLERS_CONTROL_PARAMS for
 * documentation on the parameters.
 *
 * Return values are specified per @ref BoardObjGrpGetControl.
 */
#define LW2080_CTRL_CMD_CLK_CLK_VOLT_CONTROLLERS_GET_CONTROL (0x20801035) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_CLK_INTERFACE_ID << 8) | 0x35" */

/*!
 * LW2080_CTRL_CMD_CLK_CLK_VOLT_CONTROLLERS_SET_CONTROL
 *
 * This command accepts client-specified control parameters for a set
 * of CLK_VOLT_CONTROLLERS entries in the Clocks Frequency Table, and applies
 * these new parameters to the set of CLK_VOLT_CONTROLLERS entries.
 *
 * See @ref LW2080_CTRL_CMD_CLK_CLK_VOLT_CONTROLLERS_CONTROL_PARAMS for
 * documentation on the parameters.
 *
 * Return values are specified per @ref BoardObjGrpSetControl.
 */
#define LW2080_CTRL_CMD_CLK_CLK_VOLT_CONTROLLERS_SET_CONTROL (0x20801036) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_CLK_INTERFACE_ID << 8) | 0x36" */

/*!
 * Structure to hold the ADC device aclwmulator sample
 */
typedef struct LW2080_CTRL_CLK_ADC_ACC_SAMPLE {
    /*!
     * SW cached value of the actual sampled code (0-127)
     */
    LwU8          sampledCode;

    /*!
     * SW cached value of the voltage in uV
     */
    LwU32         actualVoltageuV;

    /*!
     * SW cached value of the corrected voltage in uV
     */
    LwU32         correctedVoltageuV;

    /*!
     * SW cached value of the ADC aclwmulator register
     */
    LwU32         adcAclwmulatorVal;

    /*!
     * SW cached value of the ADC num_samples register
     */
    LwU32         adcNumSamplesVal;

    /*!
     * SW cached value of the last time stamp
     */
    LwU64_ALIGN32 timeNsLast;
} LW2080_CTRL_CLK_ADC_ACC_SAMPLE;
typedef struct LW2080_CTRL_CLK_ADC_ACC_SAMPLE *PLW2080_CTRL_CLK_ADC_ACC_SAMPLE;

/*!
 * LW2080_CTRL_CMD_CLK_PMUMON_CLK_DOMAINS_GET_SAMPLES
 *
 * Control call to query the samples within the PWR_CHANNELS PMUMON queue.
 */
#define LW2080_CTRL_CMD_CLK_PMUMON_CLK_DOMAINS_GET_SAMPLES (0x20801037) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_CLK_INTERFACE_ID << 8) | LW2080_CTRL_CLK_PMUMON_CLK_DOMAINS_GET_SAMPLES_PARAMS_MESSAGE_ID" */

/*!
 * @brief   With sample period being potentially as fast every 20ms, this gives
 *          us 5 seconds worth of data.
 */
#define LW2080_CTRL_CLK_PMUMON_CLK_DOMAINS_SAMPLE_COUNT    (250U)

/*!
 * Temporary until an INFO control call is stubbed out that exposes the supported
 * feature set of the sampling.
 */
#define LW2080_CTRL_CLK_PMUMON_CLK_DOMAINS_SAMPLE_ILWALID  (LW_U32_MAX)

/*!
 * A single sample of the power channels at a particular point in time.
 */
typedef struct LW2080_CTRL_CLK_PMUMON_CLK_DOMAINS_SAMPLE {
    /*!
     * Ptimer timestamp of when this data was collected.
     */
    LW_DECLARE_ALIGNED(LW2080_CTRL_PMUMON_SAMPLE super, 8);

    /*!
     * Point sampled programmed GPCCLK frequency in KHz.
     *
     * LW2080_CTRL_CLK_PMUMON_CLK_DOMAINS_SAMPLE_ILWALID if not supported.
     */
    LwU32 gpcClkFreqKHz;

    /*!
     * Point sampled programmed DRAMCLK frequency in KHz.
     *
     * LW2080_CTRL_CLK_PMUMON_CLK_DOMAINS_SAMPLE_ILWALID if not supported.
     */
    LwU32 dramClkFreqKHz;
} LW2080_CTRL_CLK_PMUMON_CLK_DOMAINS_SAMPLE;
typedef struct LW2080_CTRL_CLK_PMUMON_CLK_DOMAINS_SAMPLE *PLW2080_CTRL_CLK_PMUMON_CLK_DOMAINS_SAMPLE;

/*!
 * Input/Output parameters for @ref LW2080_CTRL_CMD_CLK_PMUMON_CLK_DOMAINS_GET_SAMPLES
 */
#define LW2080_CTRL_CLK_PMUMON_CLK_DOMAINS_GET_SAMPLES_PARAMS_MESSAGE_ID (0x37U)

typedef struct LW2080_CTRL_CLK_PMUMON_CLK_DOMAINS_GET_SAMPLES_PARAMS {
    /*!
     * [in/out] Meta-data for the samples[] below. Will be modified by the
     *          control call on caller's behalf and should be passed back in
     *          un-modified for subsequent calls.
     */
    LW2080_CTRL_PMUMON_GET_SAMPLES_SUPER super;

    /*!
     * [out] Between the last call and current call, samples[0...super.numSamples-1]
     *       have been published to the pmumon queue. Samples are copied into
     *       this buffer in chronological order. Indexes within this buffer do
     *       not represent indexes of samples in the actual PMUMON queue.
     */
    LW_DECLARE_ALIGNED(LW2080_CTRL_CLK_PMUMON_CLK_DOMAINS_SAMPLE samples[LW2080_CTRL_CLK_PMUMON_CLK_DOMAINS_SAMPLE_COUNT], 8);
} LW2080_CTRL_CLK_PMUMON_CLK_DOMAINS_GET_SAMPLES_PARAMS;
typedef struct LW2080_CTRL_CLK_PMUMON_CLK_DOMAINS_GET_SAMPLES_PARAMS *PLW2080_CTRL_CLK_PMUMON_CLK_DOMAINS_GET_SAMPLES_PARAMS;

/* _ctrl2080clk_h_ */
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)

