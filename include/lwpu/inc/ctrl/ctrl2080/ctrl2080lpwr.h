/*
 * SPDX-FileCopyrightText: Copyright (c) 2013-2021 LWPU CORPORATION & AFFILIATES. All rights reserved.
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
// Source file: ctrl/ctrl2080/ctrl2080lpwr.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

#include "ctrl2080mc.h"
#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)



#include "ctrl/ctrl2080/ctrl2080base.h"

/*!
 * @brief Defines power feature IDs
 *
 * Following defines specifies unique IDs to identify power saving feature.
 */
#define LW2080_CTRL_LPWR_FEATURE_ID_ILWALID                                     (0x0000)
#define LW2080_CTRL_LPWR_FEATURE_ID_PSI                                         (0x0001)
#define LW2080_CTRL_LPWR_FEATURE_ID_GC6                                         (0x0002)
#define LW2080_CTRL_LPWR_FEATURE_ID_PG                                          (0x0003)
#define LW2080_CTRL_LPWR_FEATURE_ID_AP                                          (0x0004)
#define LW2080_CTRL_LPWR_FEATURE_ID_DIDLE                                       (0x0005)
#define LW2080_CTRL_LPWR_FEATURE_ID_PGISLAND                                    (0x0006)
#define LW2080_CTRL_LPWR_FEATURE_ID_RPPG                                        (0x0007)
// This ID will be generic to overall Lpwr Features.
#define LW2080_CTRL_LPWR_FEATURE_ID_GENERIC                                     (0x0008)
// This ID will be used for all Pex features which are related to Lpwr
#define LW2080_CTRL_LPWR_FEATURE_ID_PEX                                         (0x0009)
// This ID will be used for ELCG feature.
#define LW2080_CTRL_LPWR_FEATURE_ID_ELCG                                        (0x000A)

/*!
 * @brief Defines Sub power feature IDs
 *
 * Following defines specifies unique IDs to identify sub power saving feature
 * within main feature.
 *
 * @note These IDs are mapped to enum defining ctrl ids RM_PMU_PG_PSI_ID_*.
 * The below #defines need to be in sync with the enum values
 */

// Sub Feature Ids for PSI
#define LW2080_CTRL_LPWR_SUB_FEATURE_ID_PSI_GR_ELPG_COUPLED                     (0x0000)
#define LW2080_CTRL_LPWR_SUB_FEATURE_ID_PSI_MSCG_COUPLED                        (0x0001)
#define LW2080_CTRL_LPWR_SUB_FEATURE_ID_PSI_DI_COUPLED                          (0x0002)
#define LW2080_CTRL_LPWR_SUB_FEATURE_ID_PSI_PSTATE_COUPLED                      (0x0003)
#define LW2080_CTRL_LPWR_SUB_FEATURE_ID_PSI_ALL                                 (0x0004)

// Sub Feature Ids for PG
#define LW2080_CTRL_LPWR_SUB_FEATURE_ID_GRAPHICS                                LW2080_CTRL_MC_POWERGATING_ENGINE_ID_GRAPHICS
#define LW2080_CTRL_LPWR_SUB_FEATURE_ID_GR_PASSIVE                              LW2080_CTRL_MC_POWERGATING_ENGINE_ID_GR_PASSIVE
#define LW2080_CTRL_LPWR_SUB_FEATURE_ID_GR_RG                                   LW2080_CTRL_MC_POWERGATING_ENGINE_ID_GR_RG
#define LW2080_CTRL_LPWR_SUB_FEATURE_ID_MS                                      LW2080_CTRL_MC_POWERGATING_ENGINE_ID_MS
#define LW2080_CTRL_LPWR_SUB_FEATURE_ID_DI                                      LW2080_CTRL_MC_POWERGATING_ENGINE_ID_DI
#define LW2080_CTRL_LPWR_SUB_FEATURE_ID_EI                                      LW2080_CTRL_MC_POWERGATING_ENGINE_ID_EI
#define LW2080_CTRL_LPWR_SUB_FEATURE_ID_MS_PASSIVE                              LW2080_CTRL_MC_POWERGATING_ENGINE_ID_MS_PASSIVE
#define LW2080_CTRL_LPWR_SUB_FEATURE_ID_EI_PASSIVE                              LW2080_CTRL_MC_POWERGATING_ENGINE_ID_EI_PASSIVE
#define LW2080_CTRL_LPWR_SUB_FEATURE_ID_DFPR                                    LW2080_CTRL_MC_POWERGATING_ENGINE_ID_DFPR
#define LW2080_CTRL_LPWR_SUB_FEATURE_ID_MS_DIFR_SW_ASR                          LW2080_CTRL_MC_POWERGATING_ENGINE_ID_MS_DIFR_SW_ASR
#define LW2080_CTRL_LPWR_SUB_FEATURE_ID_MS_DIFR_CG                              LW2080_CTRL_MC_POWERGATING_ENGINE_ID_MS_DIFR_CG

// Sub Feature Ids for AP
#define LW2080_CTRL_LPWR_SUB_FEATURE_ID_AP_GRAPHICS                             0x0
#define LW2080_CTRL_LPWR_SUB_FEATURE_ID_AP_GC5                                  0x1
#define LW2080_CTRL_LPWR_SUB_FEATURE_ID_AP_MSCG                                 0x2

// Sub Feature Ids for Deep Idle
#define LW2080_CTRL_LPWR_SUB_FEATURE_ID_DIDLE_DIOS                              0x0
#define LW2080_CTRL_LPWR_SUB_FEATURE_ID_DIDLE_DISSC                             0x1

// Sub Feature Ids for PGISLAND
#define LW2080_CTRL_LPWR_SUB_FEATURE_ID_PGISLAND_SCI_PMGR_GPIO_SYNC             (0x0000)

// Sub Feature Ids for RPPG
#define LW2080_CTRL_LPWR_SUB_FEATURE_ID_RPPG_GR_COUPLED                         (0x0000)
#define LW2080_CTRL_LPWR_SUB_FEATURE_ID_RPPG_MS_COUPLED                         (0x0001)
#define LW2080_CTRL_LPWR_SUB_FEATURE_ID_RPPG_DI_COUPLED                         (0x0002)
#define LW2080_CTRL_LPWR_SUB_FEATURE_ID_RPPG_MAX                                (0x0003)

// Sub Feature Ids for PEX
#define LW2080_CTRL_LPWR_SUB_FEATURE_ID_PEX_L1                                  (0x0000)
#define LW2080_CTRL_LPWR_SUB_FEATURE_ID_PEX_DEEP_L1                             (0x0001)
#define LW2080_CTRL_LPWR_SUB_FEATURE_ID_PEX_LINK_WIDTH                          (0x0002)

/*!
 * @brief Defines parameters/characteristics of power feature IDs
 *
 * Following defines specifies unique IDs to identify particular parameters or
 * characteristics of main/sub power saving feature.
 *
 * How to define parameters of different power features?
 * #define LW2080_CTRL_LPWR_PARAMETER_ID_FEATURE0_PARAM0               (0x0000)
 * #define LW2080_CTRL_LPWR_PARAMETER_ID_FEATURE0_PARAM1               (0x0001)
 * #define LW2080_CTRL_LPWR_PARAMETER_ID_SUB_FEATURE0_PARAM3           (0x0801)
 * #define LW2080_CTRL_LPWR_PARAMETER_ID_SUB_FEATURE0_PARAM4           (0x0802)
 *
 * #define LW2080_CTRL_LPWR_PARAMETER_ID_FEATURE1_PARAM0               (0x0000)
 * #define LW2080_CTRL_LPWR_PARAMETER_ID_FEATURE1_PARAM1               (0x0001)
 * #define LW2080_CTRL_LPWR_PARAMETER_ID_SUB_FEATURE1_PARAM3           (0x0801)
 * #define LW2080_CTRL_LPWR_PARAMETER_ID_SUB_FEATURE1_PARAM4           (0x0802)
 */



/*!
 * @brief PSI Rail HAL Types
 *
 * We have the following PSI Rail HALs
 *
 * HAL_LEGACY - only One Rail i.e LWVDD
 * HAL_1      - Two Rails   - LWVDD and FBVDD
 * HAL_2      - Three Rails - LWVDD, LWVDD_SRAM, FBVDD
 * HAL_3      - Three Rails - LWVDD, FBVDD, MSVDD
 *
 * More details can be found here:
 * https://confluence.lwpu.com/display/LS/LowPower+VBIOS+Tables#LowPowerVBIOSTables-PSIRAILHAL
 *
 */
#define LW2080_CTRL_LPWR_PSI_RAIL_HAL_LEGACY                                    0x0
#define LW2080_CTRL_LPWR_PSI_RAIL_HAL_1                                         0x1
#define LW2080_CTRL_LPWR_PSI_RAIL_HAL_2                                         0x2
#define LW2080_CTRL_LPWR_PSI_RAIL_HAL_3                                         0x3
#define LW2080_CTRL_LPWR_PSI_RAIL_NUM_HALS                                      0x4

// Parameters/characteristics of PSI
#define LW2080_CTRL_LPWR_PARAMETER_ID_PSI_ILWALID                               (0x0000)
#define LW2080_CTRL_LPWR_PARAMETER_ID_PSI_SUPPORTED                             (0x0001)
#define LW2080_CTRL_LPWR_PARAMETER_ID_PSI_OPTIMAL_ONE_PHASE_LWRRENT_MA          (0x0002)
#define LW2080_CTRL_LPWR_PARAMETER_ID_PSI_OPTIMAL_ONE_PHASE_LWRRENT_MA_LOGIC    LW2080_CTRL_LPWR_PARAMETER_ID_PSI_OPTIMAL_ONE_PHASE_LWRRENT_MA
#define LW2080_CTRL_LPWR_PARAMETER_ID_PSI_OPTIMAL_ONE_PHASE_LWRRENT_MA_SRAM     (0x0003)
#define LW2080_CTRL_LPWR_PARAMETER_ID_PSI_FLAVOUR                               (0x0004)
#define LW2080_CTRL_LPWR_PARAMETER_ID_PSI_ENGAGE_COUNT_RESET                    (0x0005)

//Additional information for a feature
#define LW2080_CTRL_LPWR_PSI_RAIL_ID_LWVDD                                      (0x0)
#define LW2080_CTRL_LPWR_PSI_RAIL_ID_LWVDD_SRAM                                 (0x1)
#define LW2080_CTRL_LPWR_PSI_RAIL_ID_FBVDD                                      (0x2)
#define LW2080_CTRL_LPWR_PSI_RAIL_ID_MSVDD                                      (0x3)
#define LW2080_CTRL_LPWR_PSI_RAIL_ID_MAX                                        (0x4)

// Parameters/characteristics of PSI_CTRLs (Sub Features in PSI)
#define LW2080_CTRL_LPWR_PARAMETER_ID_PSI_CTRL_SUPPORTED                        (0x0800)
#define LW2080_CTRL_LPWR_PARAMETER_ID_PSI_CTRL_ENABLED                          (0x0801)
#define LW2080_CTRL_LPWR_PARAMETER_ID_PSI_CTRL_ENGAGE_COUNT                     (0x0802)
#define LW2080_CTRL_LPWR_PARAMETER_ID_PSI_CTRL_ISLEEP_MA                        (0x0803)
#define LW2080_CTRL_LPWR_PARAMETER_ID_PSI_CTRL_ISLEEP_MA_LOGIC                  LW2080_CTRL_LPWR_PARAMETER_ID_PSI_CTRL_ISLEEP_MA
#define LW2080_CTRL_LPWR_PARAMETER_ID_PSI_CTRL_ISLEEP_MA_SRAM                   (0x0804)
#define LW2080_CTRL_LPWR_PARAMETER_ID_PSI_CTRL_ENGAGE_COUNT_RESET               (0x0805)
#define LW2080_CTRL_LPWR_PARAMETER_ID_PSI_CTRL_PSTATE_SUPPORT_MASK              (0x0806)
#define LW2080_CTRL_LPWR_PARAMETER_ID_PSI_CTRL_ENABLE                           (0x0807)
#define LW2080_CTRL_LPWR_PARAMETER_ID_PSI_CTRL_RAIL_SUPPORT_MASK                (0x0808)
#define LW2080_CTRL_LPWR_PARAMETER_ID_PSI_CTRL_RAIL_PSTATE_SUPPORT_MASK         (0x0809)

//Parameters for GC6 LwAPI Calls
#define LW2080_CTRL_LPWR_PARAMETER_ID_GC6_GET_STATS                             (0x0001)
#define LW2080_CTRL_LPWR_PARAMETER_ID_GC6_CLEAR_STATS                           (0x0002)
#define LW2080_CTRL_LPWR_PARAMETER_ID_GC6_SET_STATS                             (0x0003)

//GC6 Control Call Return Status
#define LW2080_CTRL_LPWR_PARAMETER_ID_GC6_ERROR_NULL_STATS                      (0x0800) // Stats Lwrrently Disabled
#define LW2080_CTRL_LPWR_PARAMETER_ID_GC6_ERROR_TOGGLE_STATS                    (0x0801) // Error in Surface Allocation / Deallocation
#define LW2080_CTRL_LPWR_PARAMETER_ID_GC6_ERROR_MISC                            (0x08ff) // Other Potential errors

// GC6, Latency Types Indexes
#define LW2080_CTRL_LPWR_GC6_ENTRY_TOTAL_MAX_US                                 (0x0001)
#define LW2080_CTRL_LPWR_GC6_ENTRY_TOTAL_MIN_US                                 (0x0002)
#define LW2080_CTRL_LPWR_GC6_ENTRY_TOTAL_TOT_US                                 (0x0003)
#define LW2080_CTRL_LPWR_GC6_EXIT_TOTAL_MAX_US                                  (0x0004)
#define LW2080_CTRL_LPWR_GC6_EXIT_TOTAL_MIN_US                                  (0x0005)
#define LW2080_CTRL_LPWR_GC6_EXIT_TOTAL_TOT_US                                  (0x0006)
#define LW2080_CTRL_LPWR_GC6_STATE_LOAD_MAX_US                                  (0x0007)
#define LW2080_CTRL_LPWR_GC6_STATE_LOAD_MIN_US                                  (0x0008)
#define LW2080_CTRL_LPWR_GC6_STATE_LOAD_TOT_US                                  (0x0009)
#define LW2080_CTRL_LPWR_GC6_STATE_UNLOAD_MAX_US                                (0x000a)
#define LW2080_CTRL_LPWR_GC6_STATE_UNLOAD_MIN_US                                (0x000b)
#define LW2080_CTRL_LPWR_GC6_STATE_UNLOAD_TOT_US                                (0x000c)
#define LW2080_CTRL_LPWR_GC6_POWER_ON_MAX_US                                    (0x000d)
#define LW2080_CTRL_LPWR_GC6_POWER_ON_MIN_US                                    (0x000e)
#define LW2080_CTRL_LPWR_GC6_POWER_ON_TOT_US                                    (0x000f)
#define LW2080_CTRL_LPWR_GC6_POWER_OFF_MAX_US                                   (0x0010)
#define LW2080_CTRL_LPWR_GC6_POWER_OFF_MIN_US                                   (0x0011)
#define LW2080_CTRL_LPWR_GC6_POWER_OFF_TOT_US                                   (0x0012)
#define LW2080_CTRL_LPWR_GC6_SCI_EXIT_MAX_US                                    (0x0013)
#define LW2080_CTRL_LPWR_GC6_SCI_EXIT_MIN_US                                    (0x0014)
#define LW2080_CTRL_LPWR_GC6_SCI_EXIT_TOT_US                                    (0x0015)
#define LW2080_CTRL_LPWR_GC6_SCI_ENTRY_MAX_US                                   (0x0016)
#define LW2080_CTRL_LPWR_GC6_SCI_ENTRY_MIN_US                                   (0x0017)
#define LW2080_CTRL_LPWR_GC6_SCI_ENTRY_TOT_US                                   (0x0018)
#define LW2080_CTRL_LPWR_GC6_PMU_DEVINIT_MAX_US                                 (0x0019)
#define LW2080_CTRL_LPWR_GC6_PMU_DEVINIT_MIN_US                                 (0x001a)
#define LW2080_CTRL_LPWR_GC6_PMU_DEVINIT_TOT_US                                 (0x001b)
#define LW2080_CTRL_LPWR_GC6_PMU_BOOTSTRAP_MAX_US                               (0x001c)
#define LW2080_CTRL_LPWR_GC6_PMU_BOOTSTRAP_MIN_US                               (0x001d)
#define LW2080_CTRL_LPWR_GC6_PMU_BOOTSTRAP_TOT_US                               (0x001e)
#define LW2080_CTRL_LPWR_GC6_EXIT_CPU_MODESET_MAX_US                            (0X001f)
#define LW2080_CTRL_LPWR_GC6_EXIT_CPU_MODESET_MIN_US                            (0X0020)
#define LW2080_CTRL_LPWR_GC6_EXIT_CPU_MODESET_TOT_US                            (0X0021)
#define LW2080_CTRL_LPWR_GC6_NLT_MAX_US                                         (0x0022)
#define LW2080_CTRL_LPWR_GC6_NLT_MIN_US                                         (0x0023)
#define LW2080_CTRL_LPWR_GC6_NLT_TOT_US                                         (0x0024)
#define LW2080_CTRL_LPWR_GC6_PERF_STATE_LOAD_MAX_US                             (0x0025)
#define LW2080_CTRL_LPWR_GC6_PERF_STATE_LOAD_MIN_US                             (0x0026)
#define LW2080_CTRL_LPWR_GC6_PERF_STATE_LOAD_TOT_US                             (0x0027)
#define LW2080_CTRL_LPWR_GC6_DISP_STATE_LOAD_MAX_US                             (0x0028)
#define LW2080_CTRL_LPWR_GC6_DISP_STATE_LOAD_MIN_US                             (0x0029)
#define LW2080_CTRL_LPWR_GC6_DISP_STATE_LOAD_TOT_US                             (0x002a)

/*!
 * The Address Range between 0x0000 to 0x030A are used by legacy RmCtrl.
 * Do not use this Address Range for Parameter IDS.
 */

// Parameters/characteristics of PGCTRL (Sub Features in PG)

// Parameters that are mapped with legacy Parameter IDs
#define LW2080_CTRL_LPWR_PARAMETER_ID_PG_CTRL_GATING_COUNT                      LW2080_CTRL_MC_POWERGATING_PARAMETER_GATINGCOUNT
#define LW2080_CTRL_LPWR_PARAMETER_ID_PG_CTRL_DENY_COUNT                        LW2080_CTRL_MC_POWERGATING_PARAMETER_DENYCOUNT
#define LW2080_CTRL_LPWR_PARAMETER_ID_PG_CTRL_AVG_ENTRY_LATENCY_US              LW2080_CTRL_MC_POWERGATING_PARAMETER_AVG_ENTRYTIME_US
#define LW2080_CTRL_LPWR_PARAMETER_ID_PG_CTRL_AVG_EXIT_LATENCY_US               LW2080_CTRL_MC_POWERGATING_PARAMETER_AVG_EXITTIME_US
#define LW2080_CTRL_LPWR_PARAMETER_ID_PG_CTRL_SLEEP_TIME                        LW2080_CTRL_MC_POWERGATING_PARAMETER_INGATINGTIME_US
#define LW2080_CTRL_LPWR_PARAMETER_ID_PG_CTRL_NON_SLEEP_TIME                    LW2080_CTRL_MC_POWERGATING_PARAMETER_UNGATINGTIME_US
#define LW2080_CTRL_LPWR_PARAMETER_ID_PG_CTRL_EXIT_COUNT                        LW2080_CTRL_MC_POWERGATING_PARAMETER_INGATINGCOUNT
#define LW2080_CTRL_LPWR_PARAMETER_ID_PG_CTRL_MAX_ENTRY_LATENCY_US              LW2080_CTRL_MC_POWERGATING_PARAMETER_MAX_ENTRYTIME_US
#define LW2080_CTRL_LPWR_PARAMETER_ID_PG_CTRL_MAX_EXIT_LATENCY_US               LW2080_CTRL_MC_POWERGATING_PARAMETER_MAX_EXITTIME_US
#define LW2080_CTRL_LPWR_PARAMETER_ID_PG_CTRL_SHARED_STATISTICS_USED_LAST       LW2080_CTRL_LPWR_PARAMETER_ID_PG_CTRL_MAX_EXIT_LATENCY_US

/*!
 * @brief Reserving few shared IDs for future use
 *
 * The following IDs are kept as reserved for future features. Use these ID only when we
 * need to share parameter with LW2080_CTRL_MC_POWERGATING_PARAMETER_*.
 * It is not recommended to add new ID in this section.
 */
#define LW2080_CTRL_LPWR_PARAMETER_ID_PG_CTRL_RESERVED_1                        (0x2c) /* finn: Evaluated from "(LW2080_CTRL_LPWR_PARAMETER_ID_PG_CTRL_SHARED_STATISTICS_USED_LAST + 1)" */
#define LW2080_CTRL_LPWR_PARAMETER_ID_PG_CTRL_RESERVED_2                        (0x2d) /* finn: Evaluated from "(LW2080_CTRL_LPWR_PARAMETER_ID_PG_CTRL_SHARED_STATISTICS_USED_LAST + 2)" */
#define LW2080_CTRL_LPWR_PARAMETER_ID_PG_CTRL_RESERVED_3                        (0x2e) /* finn: Evaluated from "(LW2080_CTRL_LPWR_PARAMETER_ID_PG_CTRL_SHARED_STATISTICS_USED_LAST + 3)" */
#define LW2080_CTRL_LPWR_PARAMETER_ID_PG_CTRL_RESERVED_4                        (0x2f) /* finn: Evaluated from "(LW2080_CTRL_LPWR_PARAMETER_ID_PG_CTRL_SHARED_STATISTICS_USED_LAST + 4)" */
#define LW2080_CTRL_LPWR_PARAMETER_ID_PG_CTRL_RESERVED_5                        (0x30) /* finn: Evaluated from "(LW2080_CTRL_LPWR_PARAMETER_ID_PG_CTRL_SHARED_STATISTICS_USED_LAST + 5)" */
#define LW2080_CTRL_LPWR_PARAMETER_ID_PG_CTRL_RESERVED6                         (0x31) /* finn: Evaluated from "(LW2080_CTRL_LPWR_PARAMETER_ID_PG_CTRL_SHARED_STATISTICS_USED_LAST + 6)" */
#define LW2080_CTRL_LPWR_PARAMETER_ID_PG_CTRL_RESERVED_7                        (0x32) /* finn: Evaluated from "(LW2080_CTRL_LPWR_PARAMETER_ID_PG_CTRL_SHARED_STATISTICS_USED_LAST + 7)" */
#define LW2080_CTRL_LPWR_PARAMETER_ID_PG_CTRL_RESERVED_8                        (0x33) /* finn: Evaluated from "(LW2080_CTRL_LPWR_PARAMETER_ID_PG_CTRL_SHARED_STATISTICS_USED_LAST + 8)" */
#define LW2080_CTRL_LPWR_PARAMETER_ID_PG_CTRL_RESERVED_9                        (0x34) /* finn: Evaluated from "(LW2080_CTRL_LPWR_PARAMETER_ID_PG_CTRL_SHARED_STATISTICS_USED_LAST + 9)" */
#define LW2080_CTRL_LPWR_PARAMETER_ID_PG_CTRL_RESERVED_10                       (0x35) /* finn: Evaluated from "(LW2080_CTRL_LPWR_PARAMETER_ID_PG_CTRL_SHARED_STATISTICS_USED_LAST + 10)" */
#define LW2080_CTRL_LPWR_PARAMETER_ID_PG_CTRL_SHARED_STATISTICS_LAST            (LW2080_CTRL_LPWR_PARAMETER_ID_PG_CTRL_RESERVED_10)

/*!
 * @brief Parameter IDs used by LW2080_CTRL_CMD_LPWR_FEATURE_PARAMETER_GET/_SET control call
 *
 * Following ID are only used by new control call
 * LW2080_CTRL_CMD_LPWR_FEATURE_PARAMETER_GET/_SET
 */
#define LW2080_CTRL_LPWR_PARAMETER_ID_PG_CTRL_PSTATE_SUPPORT_MASK               (0x36) /* finn: Evaluated from "(LW2080_CTRL_LPWR_PARAMETER_ID_PG_CTRL_SHARED_STATISTICS_LAST + 1)" */
#define LW2080_CTRL_LPWR_PARAMETER_ID_PG_CTRL_ENABLE                            (0x37) /* finn: Evaluated from "(LW2080_CTRL_LPWR_PARAMETER_ID_PG_CTRL_SHARED_STATISTICS_LAST + 2)" */
#define LW2080_CTRL_LPWR_PARAMETER_ID_PG_CTRL_SUB_FEATURE_SUPPORT_MASK          (0x38) /* finn: Evaluated from "(LW2080_CTRL_LPWR_PARAMETER_ID_PG_CTRL_SHARED_STATISTICS_LAST + 3)" */
#define LW2080_CTRL_LPWR_PARAMETER_ID_PG_CTRL_FEATURE_SUPPORT                   (0x39) /* finn: Evaluated from "(LW2080_CTRL_LPWR_PARAMETER_ID_PG_CTRL_SHARED_STATISTICS_LAST + 4)" */
#define LW2080_CTRL_LPWR_PARAMETER_ID_PG_CTRL_IDLE_THRESHOLD_US                 (0x3a) /* finn: Evaluated from "(LW2080_CTRL_LPWR_PARAMETER_ID_PG_CTRL_SHARED_STATISTICS_LAST + 5)" */
#define LW2080_CTRL_LPWR_PARAMETER_ID_PG_CTRL_PPU_THRESHOLD_US                  (0x3b) /* finn: Evaluated from "(LW2080_CTRL_LPWR_PARAMETER_ID_PG_CTRL_SHARED_STATISTICS_LAST + 6)" */
#define LW2080_CTRL_LPWR_PARAMETER_ID_PG_CTRL_ABORT_REASON                      (0x3c) /* finn: Evaluated from "(LW2080_CTRL_LPWR_PARAMETER_ID_PG_CTRL_SHARED_STATISTICS_LAST + 7)" */
#define LW2080_CTRL_LPWR_PARAMETER_ID_PG_CTRL_STAT_RESET                        (0x3d) /* finn: Evaluated from "(LW2080_CTRL_LPWR_PARAMETER_ID_PG_CTRL_SHARED_STATISTICS_LAST + 8)" */
#define LW2080_CTRL_LPWR_PARAMETER_ID_PG_CTRL_SUB_FEATURE_ENABLED_MASK          (0x3e) /* finn: Evaluated from "(LW2080_CTRL_LPWR_PARAMETER_ID_PG_CTRL_SHARED_STATISTICS_LAST + 9)" */
#define LW2080_CTRL_LPWR_PARAMETER_ID_PG_CTRL_ENGAGE_STATUS                     (0x3f) /* finn: Evaluated from "(LW2080_CTRL_LPWR_PARAMETER_ID_PG_CTRL_SHARED_STATISTICS_LAST + 10)" */
#define LW2080_CTRL_LPWR_PARAMETER_ID_PG_CTRL_IDLE_MASK_0                       (0x40) /* finn: Evaluated from "(LW2080_CTRL_LPWR_PARAMETER_ID_PG_CTRL_SHARED_STATISTICS_LAST + 11)" */
#define LW2080_CTRL_LPWR_PARAMETER_ID_PG_CTRL_IDLE_MASK_1                       (0x41) /* finn: Evaluated from "(LW2080_CTRL_LPWR_PARAMETER_ID_PG_CTRL_SHARED_STATISTICS_LAST + 12)" */
#define LW2080_CTRL_LPWR_PARAMETER_ID_PG_CTRL_IDLE_MASK_2                       (0x42) /* finn: Evaluated from "(LW2080_CTRL_LPWR_PARAMETER_ID_PG_CTRL_SHARED_STATISTICS_LAST + 13)" */
#define LW2080_CTRL_LPWR_PARAMETER_ID_PG_CTRL_THRESHOLD_RESET                   (0x43) /* finn: Evaluated from "(LW2080_CTRL_LPWR_PARAMETER_ID_PG_CTRL_SHARED_STATISTICS_LAST + 14)" */
#define LW2080_CTRL_LPWR_PARAMETER_ID_PG_CTRL_WAKEUP_REASON_MASK                (0x44) /* finn: Evaluated from "(LW2080_CTRL_LPWR_PARAMETER_ID_PG_CTRL_SHARED_STATISTICS_LAST + 15)" */
#define LW2080_CTRL_LPWR_PARAMETER_ID_PG_CTRL_WAKEUP_TYPE                       (0x45) /* finn: Evaluated from "(LW2080_CTRL_LPWR_PARAMETER_ID_PG_CTRL_SHARED_STATISTICS_LAST + 16)" */
#define LW2080_CTRL_LPWR_PARAMETER_ID_PG_CTRL_ABORT_REASON_MASK                 (0x46) /* finn: Evaluated from "(LW2080_CTRL_LPWR_PARAMETER_ID_PG_CTRL_SHARED_STATISTICS_LAST + 17)" */
#define LW2080_CTRL_LPWR_PARAMETER_ID_PG_CTRL_THRASH_DETECTION_COUNT            (0x47) /* finn: Evaluated from "(LW2080_CTRL_LPWR_PARAMETER_ID_PG_CTRL_SHARED_STATISTICS_LAST + 18)" */
#define LW2080_CTRL_LPWR_PARAMETER_ID_PG_CTRL_SW_DISABLE_REASON_MASK_RM         (0x48) /* finn: Evaluated from "(LW2080_CTRL_LPWR_PARAMETER_ID_PG_CTRL_SHARED_STATISTICS_LAST + 19)" */
#define LW2080_CTRL_LPWR_PARAMETER_ID_PG_CTRL_HW_DISABLE_REASON_MASK_RM         (0x49) /* finn: Evaluated from "(LW2080_CTRL_LPWR_PARAMETER_ID_PG_CTRL_SHARED_STATISTICS_LAST + 20)" */
#define LW2080_CTRL_LPWR_PARAMETER_ID_PG_CTRL_SW_DISABLE_REASON_MASK_PMU        (0x4a) /* finn: Evaluated from "(LW2080_CTRL_LPWR_PARAMETER_ID_PG_CTRL_SHARED_STATISTICS_LAST + 21)" */
#define LW2080_CTRL_LPWR_PARAMETER_ID_PG_CTRL_HW_DISABLE_REASON_MASK_PMU        (0x4b) /* finn: Evaluated from "(LW2080_CTRL_LPWR_PARAMETER_ID_PG_CTRL_SHARED_STATISTICS_LAST + 22)" */

// Parameters/characteristics of RPPG
#define LW2080_CTRL_LPWR_PARAMETER_ID_RPPG_BASE_INDEX                           (0x0000)
#define LW2080_CTRL_LPWR_PARAMETER_ID_RPPG_SUPPORT                              (0x1) /* finn: Evaluated from "(LW2080_CTRL_LPWR_PARAMETER_ID_RPPG_BASE_INDEX + 1)" */

// Parameter/charactersitics of RPPG CTRL
#define LW2080_CTRL_LPWR_PARAMETER_ID_RPPG_CTRL_BASE_INDEX                      (0x0800)
#define LW2080_CTRL_LPWR_PARAMETER_ID_RPPG_CTRL_SUPPORT                         (0x801) /* finn: Evaluated from "(LW2080_CTRL_LPWR_PARAMETER_ID_RPPG_CTRL_BASE_INDEX + 1)" */
#define LW2080_CTRL_LPWR_PARAMETER_ID_RPPG_CTRL_ENABLE                          (0x802) /* finn: Evaluated from "(LW2080_CTRL_LPWR_PARAMETER_ID_RPPG_CTRL_BASE_INDEX + 2)" */
#define LW2080_CTRL_LPWR_PARAMETER_ID_RPPG_CTRL_ENTRY_COUNT                     (0x803) /* finn: Evaluated from "(LW2080_CTRL_LPWR_PARAMETER_ID_RPPG_CTRL_BASE_INDEX + 3)" */
#define LW2080_CTRL_LPWR_PARAMETER_ID_RPPG_CTRL_EXIT_COUNT                      (0x804) /* finn: Evaluated from "(LW2080_CTRL_LPWR_PARAMETER_ID_RPPG_CTRL_BASE_INDEX + 4)" */
#define LW2080_CTRL_LPWR_PARAMETER_ID_RPPG_CTRL_PSTATE_SUPPORT_MASK             (0x805) /* finn: Evaluated from "(LW2080_CTRL_LPWR_PARAMETER_ID_RPPG_CTRL_BASE_INDEX + 5)" */
#define LW2080_CTRL_LPWR_PARAMETER_ID_RPPG_CTRL_STAT_RESET                      (0x806) /* finn: Evaluated from "(LW2080_CTRL_LPWR_PARAMETER_ID_RPPG_CTRL_BASE_INDEX + 6)" */

// Parameters of Generic Group
#define LW2080_CTRL_LPWR_PARAMETER_ID_GENERIC_CTRL_BASE_INDEX                   (0x0000)
#define LW2080_CTRL_LPWR_PARAMETER_ID_GENERIC_CTRL_CACHE_SUPPORT                (0x1) /* finn: Evaluated from "(LW2080_CTRL_LPWR_PARAMETER_ID_GENERIC_CTRL_BASE_INDEX + 1)" */
#define LW2080_CTRL_LPWR_PARAMETER_ID_GENERIC_CTRL_CACHE_ENABLE                 (0x2) /* finn: Evaluated from "(LW2080_CTRL_LPWR_PARAMETER_ID_GENERIC_CTRL_BASE_INDEX + 2)" */
#define LW2080_CTRL_LPWR_PARAMETER_ID_GENERIC_CTRL_IDLE_STATUS_0                (0x3) /* finn: Evaluated from "(LW2080_CTRL_LPWR_PARAMETER_ID_GENERIC_CTRL_BASE_INDEX + 3)" */
#define LW2080_CTRL_LPWR_PARAMETER_ID_GENERIC_CTRL_IDLE_STATUS_1                (0x4) /* finn: Evaluated from "(LW2080_CTRL_LPWR_PARAMETER_ID_GENERIC_CTRL_BASE_INDEX + 4)" */
#define LW2080_CTRL_LPWR_PARAMETER_ID_GENERIC_CTRL_IDLE_STATUS_2                (0x5) /* finn: Evaluated from "(LW2080_CTRL_LPWR_PARAMETER_ID_GENERIC_CTRL_BASE_INDEX + 5)" */
#define LW2080_CTRL_LPWR_PARAMETER_ID_GENERIC_CTRL_LPWR_MODE                    (0x6) /* finn: Evaluated from "(LW2080_CTRL_LPWR_PARAMETER_ID_GENERIC_CTRL_BASE_INDEX + 6)" */
#define LW2080_CTRL_LPWR_PARAMETER_ID_GENERIC_CTRL_LPWR_MODE_RESET              (0x7) /* finn: Evaluated from "(LW2080_CTRL_LPWR_PARAMETER_ID_GENERIC_CTRL_BASE_INDEX + 7)" */
#define LW2080_CTRL_LPWR_PARAMETER_ID_GENERIC_CTRL_LPWR_MODE_SUPPORT_MASK       (0x8) /* finn: Evaluated from "(LW2080_CTRL_LPWR_PARAMETER_ID_GENERIC_CTRL_BASE_INDEX + 8)" */
#define LW2080_CTRL_LPWR_PARAMETER_ID_GENERIC_CTRL_LPWR_TEST_START              (0x9) /* finn: Evaluated from "(LW2080_CTRL_LPWR_PARAMETER_ID_GENERIC_CTRL_BASE_INDEX + 9)" */
#define LW2080_CTRL_LPWR_PARAMETER_ID_GENERIC_CTRL_LPWR_TEST_STOP               (0xa) /* finn: Evaluated from "(LW2080_CTRL_LPWR_PARAMETER_ID_GENERIC_CTRL_BASE_INDEX + 10)" */
//
// LPWR modes: LPWR mode allow us to engage different LPWR features in same
//             scenario. They help us to qualify different POR combinations
//             same test.
//
// Mode 0: This mode is supported only when GR-RG is supported on chip.
//         This mode keeps GR-RG, GR-PG and GR-Passive enabled on chip.
//
// Mode 1: This mode is supported only when GR-PG is supported on chip.
//         This mode disables GR-RG so that GR-PG can get the priority.
//
// Mode 2: This mode is supported only when GR-Passive is supported on chip.
//         This mode disables GR-RG and GR-PG so that GR-Passive can get
//         priority.
//
#define LW2080_CTRL_LPWR_MODE_0                                                 (0x0000)
#define LW2080_CTRL_LPWR_MODE_1                                                 (0x0001)
#define LW2080_CTRL_LPWR_MODE_2                                                 (0x0002)

//
// LPWR Tests: LPWR test to do functional verification of code path of different
//             aspects.
//
// MS_ODP_DMA          : MS ODP DMA Test, validates DMA operation in PMU.
// MS_ODP_HARD_CRITICAL: Hard critical section verification test in PMU.
// MS_ODP_SOFT_CRITICAL: Scheduler suspend/resume verification test in PMU.
// MS_ODP_PAGE_FAULT   : Page fault injection test in PMU.
//
#define LW2080_CTRL_LPWR_TEST_ID_MS_ODP_DMA                                     (0x00000000)
#define LW2080_CTRL_LPWR_TEST_ID_MS_ODP_HARD_CRITICAL_SECTION                   (0x00000001)
#define LW2080_CTRL_LPWR_TEST_ID_MS_ODP_SOFT_CRITICAL_SECTION                   (0x00000002)
#define LW2080_CTRL_LPWR_TEST_ID_MS_ODP_PAGE_FAULT                              (0x00000003)
#define LW2080_CTRL_LPWR_TEST_ID_MAX                                            (0x00000004)

// Parameters of Pex L1/DeepL1
#define LW2080_CTRL_LPWR_PARAMETER_ID_PEX_CTRL_BASE_INDEX                       (0x0000)
#define LW2080_CTRL_LPWR_PARAMETER_ID_PEX_CTRL_ENTRY_COUNT                      (0x1) /* finn: Evaluated from "(LW2080_CTRL_LPWR_PARAMETER_ID_PEX_CTRL_BASE_INDEX + 1)" */
#define LW2080_CTRL_LPWR_PARAMETER_ID_PEX_CTRL_RESET_COUNT                      (0x2) /* finn: Evaluated from "(LW2080_CTRL_LPWR_PARAMETER_ID_PEX_CTRL_BASE_INDEX + 2)" */
#define LW2080_CTRL_LPWR_PARAMETER_ID_PEX_CTRL_ENABLE                           (0x3) /* finn: Evaluated from "(LW2080_CTRL_LPWR_PARAMETER_ID_PEX_CTRL_BASE_INDEX + 3)" */
#define LW2080_CTRL_LPWR_PARAMETER_ID_PEX_CTRL_SUPPORT                          (0x4) /* finn: Evaluated from "(LW2080_CTRL_LPWR_PARAMETER_ID_PEX_CTRL_BASE_INDEX + 4)" */
#define LW2080_CTRL_LPWR_PARAMETER_ID_PEX_CTRL_PSTATE_SUPPORT_MASK              (0x5) /* finn: Evaluated from "(LW2080_CTRL_LPWR_PARAMETER_ID_PEX_CTRL_BASE_INDEX + 5)" */

// Parameters of ELCG
#define LW2080_CTRL_LPWR_PARAMETER_ID_ELCG_BASE_INDEX                           (0x0000)
#define LW2080_CTRL_LPWR_PARAMETER_ID_ELCG_SUPPORT                              (0x1) /* finn: Evaluated from "(LW2080_CTRL_LPWR_PARAMETER_ID_ELCG_BASE_INDEX + 1)" */

// Parameters of ELCG CTRL
#define LW2080_CTRL_LPWR_PARAMETER_ID_ELCG_CTRL_BASE_INDEX                      (0x0800)
#define LW2080_CTRL_LPWR_PARAMETER_ID_ELCG_CTRL_ENABLED_MASK                    (0x801) /* finn: Evaluated from "(LW2080_CTRL_LPWR_PARAMETER_ID_ELCG_CTRL_BASE_INDEX + 1)" */
#define LW2080_CTRL_LPWR_PARAMETER_ID_ELCG_CTRL_ENABLE                          (0x802) /* finn: Evaluated from "(LW2080_CTRL_LPWR_PARAMETER_ID_ELCG_CTRL_BASE_INDEX + 2)" */

/*!
 * @brief Minimum idle threshold for PgCtrl
 * NOTE: Each PgCtrl has lower cap on minimum idle threshold. Idle threshold
 *       will set to lower cap, if default value is less than lower cap.
 */
#define LW2080_CTRL_LPWR_CTRL_IDLE_THRESHOLD_DEFAULT_MINIMUM_US                 (100)

/*!
 * Bitmask for supported PG_GR sub-features
 *
 * NOTE : These entries should match to RM_PMU_GR_FEATURE_MASK_*
 * If anything is changed here, corresponding change should be made there also
 */
#define LW2080_CTRL_LPWR_PG_CTRL_GR_FEATURE_SDIV_SLOWDOWN_SUPPORT                  0:0
#define LW2080_CTRL_LPWR_PG_CTRL_GR_FEATURE_SDIV_SLOWDOWN_SUPPORT_NO            (0x0000)
#define LW2080_CTRL_LPWR_PG_CTRL_GR_FEATURE_SDIV_SLOWDOWN_SUPPORT_YES           (0x0001)
#define LW2080_CTRL_LPWR_PG_CTRL_GR_FEATURE_RESTORE_QUADRO_SETTINGS_SUPPORT        1:1
#define LW2080_CTRL_LPWR_PG_CTRL_GR_FEATURE_RESTORE_QUADRO_SETTINGS_SUPPORT_NO  (0x0000)
#define LW2080_CTRL_LPWR_PG_CTRL_GR_FEATURE_RESTORE_QUADRO_SETTINGS_SUPPORT_YES (0x0001)
#define LW2080_CTRL_LPWR_PG_CTRL_GR_FEATURE_POWER_GATING_SUPPORT                   2:2
#define LW2080_CTRL_LPWR_PG_CTRL_GR_FEATURE_POWER_GATING_SUPPORT_NO             (0x0000)
#define LW2080_CTRL_LPWR_PG_CTRL_GR_FEATURE_POWER_GATING_SUPPORT_YES            (0x0001)
#define LW2080_CTRL_LPWR_PG_CTRL_GR_FEATURE_RPPG_SUPPORT                           3:3
#define LW2080_CTRL_LPWR_PG_CTRL_GR_FEATURE_RPPG_SUPPORT_NO                     (0x0000)
#define LW2080_CTRL_LPWR_PG_CTRL_GR_FEATURE_RPPG_SUPPORT_YES                    (0x0001)
#define LW2080_CTRL_LPWR_PG_CTRL_GR_FEATURE_PSI_SUPPORT                            4:4
#define LW2080_CTRL_LPWR_PG_CTRL_GR_FEATURE_PSI_SUPPORT_NO                      (0x0000)
#define LW2080_CTRL_LPWR_PG_CTRL_GR_FEATURE_PSI_SUPPORT_YES                     (0x0001)


/*!
 * Bitmask for supported PG_MS sub-features
 *
 * NOTE : These entries should match to RM_PMU_MS_FEATURE_MASK_*
 * If anything is changed here, corresponding change should be made there also
 */
#define LW2080_CTRL_LPWR_PG_CTRL_MS_FEATURE_CLOCKGATING_SUPPORT             0:0
#define LW2080_CTRL_LPWR_PG_CTRL_MS_FEATURE_CLOCKGATING_SUPPORT_NO              (0x0000)
#define LW2080_CTRL_LPWR_PG_CTRL_MS_FEATURE_CLOCKGATING_SUPPORT_YES             (0x0001)

#define LW2080_CTRL_LPWR_PG_CTRL_MS_FEATURE_SWASR_SUPPORT                   1:1
#define LW2080_CTRL_LPWR_PG_CTRL_MS_FEATURE_SWASR_SUPPORT_NO                    (0x0000)
#define LW2080_CTRL_LPWR_PG_CTRL_MS_FEATURE_SWASR_SUPPORT_YES                   (0x0001)

#define LW2080_CTRL_LPWR_PG_CTRL_MS_FEATURE_DISPLAY_SUPPORT                 2:2
#define LW2080_CTRL_LPWR_PG_CTRL_MS_FEATURE_DISPLAY_SUPPORT_NO                  (0x0000)
#define LW2080_CTRL_LPWR_PG_CTRL_MS_FEATURE_DISPLAY_SUPPORT_YES                 (0x0001)

#define LW2080_CTRL_LPWR_PG_CTRL_MS_FEATURE_ISO_STUTTER_SUPPORT             3:3
#define LW2080_CTRL_LPWR_PG_CTRL_MS_FEATURE_ISO_STUTTER_SUPPORT_NO              (0x0000)
#define LW2080_CTRL_LPWR_PG_CTRL_MS_FEATURE_ISO_STUTTER_SUPPORT_YES             (0x0001)

#define LW2080_CTRL_LPWR_PG_CTRL_MS_FEATURE_ELPG_DECOUPLED_MSCG_SUPPORT     4:4
#define LW2080_CTRL_LPWR_PG_CTRL_MS_FEATURE_ELPG_DECOUPLED_MSCG_SUPPORT_NO      (0x0000)
#define LW2080_CTRL_LPWR_PG_CTRL_MS_FEATURE_ELPG_DECOUPLED_MSCG_SUPPORT_YES     (0x0001)

#define LW2080_CTRL_LPWR_PG_CTRL_MS_FEATURE_FB_TRAINING_SUPPORT             5:5
#define LW2080_CTRL_LPWR_PG_CTRL_MS_FEATURE_FB_TRAINING_SUPPORT_NO              (0x0000)
#define LW2080_CTRL_LPWR_PG_CTRL_MS_FEATURE_FB_TRAINING_SUPPORT_YES             (0x0001)

#define LW2080_CTRL_LPWR_PG_CTRL_MS_FEATURE_HSHUB_SUPPORT                   6:6
#define LW2080_CTRL_LPWR_PG_CTRL_MS_FEATURE_HSHUB_SUPPORT_NO                    (0x0000)
#define LW2080_CTRL_LPWR_PG_CTRL_MS_FEATURE_HSHUB_SUPPORT_YES                   (0x0001)

#define LW2080_CTRL_LPWR_PG_CTRL_MS_FEATURE_SEC2_RTOS_SUPPORT               7:7
#define LW2080_CTRL_LPWR_PG_CTRL_MS_FEATURE_SEC2_RTOS_SUPPORT_NO                (0x0000)
#define LW2080_CTRL_LPWR_PG_CTRL_MS_FEATURE_SEC2_RTOS_SUPPORT_YES               (0x0001)

#define LW2080_CTRL_LPWR_PG_CTRL_MS_FEATURE_RPPG_SUPPORT                    8:8
#define LW2080_CTRL_LPWR_PG_CTRL_MS_FEATURE_RPPG_SUPPORT_NO                     (0x0000)
#define LW2080_CTRL_LPWR_PG_CTRL_MS_FEATURE_RPPG_SUPPORT_YES                    (0x0001)

#define LW2080_CTRL_LPWR_PG_CTRL_MS_FEATURE_IDLE_FLIPPED_RESET_SUPPORT      9:9
#define LW2080_CTRL_LPWR_PG_CTRL_MS_FEATURE_IDLE_FLIPPED_RESET_SUPPORT_NO       (0x0000)
#define LW2080_CTRL_LPWR_PG_CTRL_MS_FEATURE_IDLE_FLIPPED_RESET_SUPPORT_YES      (0x0001)

#define LW2080_CTRL_LPWR_PG_CTRL_MS_FEATURE_IMMEDIATE_FLIP_SUPPORT          10:10
#define LW2080_CTRL_LPWR_PG_CTRL_MS_FEATURE_IMMEDIATE_FLIP_SUPPORT_NO           (0x0000)
#define LW2080_CTRL_LPWR_PG_CTRL_MS_FEATURE_IMMEDIATE_FLIP_SUPPORT_YES          (0x0001)

#define LW2080_CTRL_LPWR_PG_CTRL_MS_FEATURE_LPWR_ONESHOT_SUPPORT            11:11
#define LW2080_CTRL_LPWR_PG_CTRL_MS_FEATURE_LPWR_ONESHOT_SUPPORT_NO             (0x0000)
#define LW2080_CTRL_LPWR_PG_CTRL_MS_FEATURE_LPWR_ONESHOT_SUPPORT_YES            (0x0001)

#define LW2080_CTRL_LPWR_PG_CTRL_MS_FEATURE_DWB_SUPPORT                     12:12
#define LW2080_CTRL_LPWR_PG_CTRL_MS_FEATURE_DWB_SUPPORT_NO                      (0x0000)
#define LW2080_CTRL_LPWR_PG_CTRL_MS_FEATURE_DWB_SUPPORT_YES                     (0x0001)

#define LW2080_CTRL_LPWR_PG_CTRL_MS_FEATURE_PSI_SUPPORT                     13:13
#define LW2080_CTRL_LPWR_PG_CTRL_MS_FEATURE_PSI_SUPPORT_NO                      (0x0000)
#define LW2080_CTRL_LPWR_PG_CTRL_MS_FEATURE_PSI_SUPPORT_YES                     (0x0001)

#define LW2080_CTRL_LPWR_PG_CTRL_MS_FEATURE_GDDR5_WR_TRAINING_SUPPORT       14:14
#define LW2080_CTRL_LPWR_PG_CTRL_MS_FEATURE_GDDR5_WR_TRAINING_SUPPORT_NO        (0x0000)
#define LW2080_CTRL_LPWR_PG_CTRL_MS_FEATURE_GDDR5_WR_TRAINING_SUPPORT_YES       (0x0001)

#define LW2080_CTRL_LPWR_PG_CTRL_MS_FEATURE_STOP_CLOCK_SUPPORT              15:15
#define LW2080_CTRL_LPWR_PG_CTRL_MS_FEATURE_STOP_CLOCK_SUPPORT_NO               (0x0000)
#define LW2080_CTRL_LPWR_PG_CTRL_MS_FEATURE_STOP_CLOCK_SUPPORT_YES              (0x0001)

#define LW2080_CTRL_LPWR_PG_CTRL_MS_FEATURE_GSP_SUPPORT                     16:16
#define LW2080_CTRL_LPWR_PG_CTRL_MS_FEATURE_GSP_SUPPORT_NO                      (0x0000)
#define LW2080_CTRL_LPWR_PG_CTRL_MS_FEATURE_GSP_SUPPORT_YES                     (0x0001)

#define LW2080_CTRL_LPWR_PG_CTRL_MS_FEATURE_PERF_SUPPORT                    17:17
#define LW2080_CTRL_LPWR_PG_CTRL_MS_FEATURE_PERF_SUPPORT_NO                     (0x0000)
#define LW2080_CTRL_LPWR_PG_CTRL_MS_FEATURE_PERF_SUPPORT_YES                    (0x0001)

#define LW2080_CTRL_LPWR_PG_CTRL_MS_FEATURE_IN_LOW_FPS_SUPPORT              18:18
#define LW2080_CTRL_LPWR_PG_CTRL_MS_FEATURE_IN_LOW_FPS_SUPPORT_NO               (0x0000)
#define LW2080_CTRL_LPWR_PG_CTRL_MS_FEATURE_IN_LOW_FPS_SUPPORT_YES              (0x0001)

/*!
 * @brief Bitmask for supported PG DI sub-features
 *
 * NOTE : These entries should be consistent with RM_PMU_DI_FEATURE_MASK_*
 * If anything is changed here, corresponding change should be made there also
 */
#define LW2080_CTRL_LPWR_PG_CTRL_DI_FEATURE_DFE_VAL                    0:0
#define LW2080_CTRL_LPWR_PG_CTRL_DI_FEATURE_DFE_VAL_DISABLED                    (0x0000)
#define LW2080_CTRL_LPWR_PG_CTRL_DI_FEATURE_DFE_VAL_ENABLED                     (0x0001)
#define LW2080_CTRL_LPWR_PG_CTRL_DI_FEATURE_MEM_ACTION_                1:1
#define LW2080_CTRL_LPWR_PG_CTRL_DI_FEATURE_MEM_ACTION_DISABLED                 (0x0000)
#define LW2080_CTRL_LPWR_PG_CTRL_DI_FEATURE_MEM_ACTION_ENABLED                  (0x0001)
#define LW2080_CTRL_LPWR_PG_CTRL_DI_FEATURE_REFMPLL                    2:2
#define LW2080_CTRL_LPWR_PG_CTRL_DI_FEATURE_REFMPLL_DISABLED                    (0x0000)
#define LW2080_CTRL_LPWR_PG_CTRL_DI_FEATURE_REFMPLL_ENABLED                     (0x0001)
#define LW2080_CTRL_LPWR_PG_CTRL_DI_FEATURE_DRAMPLL                    3:3
#define LW2080_CTRL_LPWR_PG_CTRL_DI_FEATURE_DRAMPLL_DISABLED                    (0x0000)
#define LW2080_CTRL_LPWR_PG_CTRL_DI_FEATURE_DRAMPLL_ENABLED                     (0x0001)
#define LW2080_CTRL_LPWR_PG_CTRL_DI_FEATURE_CORE_PLLS                  4:4
#define LW2080_CTRL_LPWR_PG_CTRL_DI_FEATURE_CORE_PLLS_DISABLED                  (0x0000)
#define LW2080_CTRL_LPWR_PG_CTRL_DI_FEATURE_CORE_PLLS_ENABLED                   (0x0001)
#define LW2080_CTRL_LPWR_PG_CTRL_DI_FEATURE_CORE_NAFLLS                5:5
#define LW2080_CTRL_LPWR_PG_CTRL_DI_FEATURE_CORE_NAFLLS_DISABLED                (0x0000)
#define LW2080_CTRL_LPWR_PG_CTRL_DI_FEATURE_CORE_NAFLLS_ENABLED                 (0x0001)
#define LW2080_CTRL_LPWR_PG_CTRL_DI_FEATURE_DISPL_PLLS                 6:6
#define LW2080_CTRL_LPWR_PG_CTRL_DI_FEATURE_DISPL_PLLS_DISABLED                 (0x0000)
#define LW2080_CTRL_LPWR_PG_CTRL_DI_FEATURE_DISPL_PLLS_ENABLED                  (0x0001)
#define LW2080_CTRL_LPWR_PG_CTRL_DI_FEATURE_OSM_CLKS                   7:7
#define LW2080_CTRL_LPWR_PG_CTRL_DI_FEATURE_OSM_CLKS_DISABLED                   (0x0000)
#define LW2080_CTRL_LPWR_PG_CTRL_DI_FEATURE_OSM_CLKS_ENABLED                    (0x0001)
#define LW2080_CTRL_LPWR_PG_CTRL_DI_FEATURE_PSI                        8:8
#define LW2080_CTRL_LPWR_PG_CTRL_DI_FEATURE_PSI_DISABLED                        (0x0000)
#define LW2080_CTRL_LPWR_PG_CTRL_DI_FEATURE_PSI_ENABLED                         (0x0001)
#define LW2080_CTRL_LPWR_PG_CTRL_DI_FEATURE_THERM_SENSOR               9:9
#define LW2080_CTRL_LPWR_PG_CTRL_DI_FEATURE_THERM_SENSOR_DISABLED               (0x0000)
#define LW2080_CTRL_LPWR_PG_CTRL_DI_FEATURE_THERM_SENSOR_ENABLED                (0x0001)
#define LW2080_CTRL_LPWR_PG_CTRL_DI_FEATURE_XTAL_SRC_SWITCH            10:10
#define LW2080_CTRL_LPWR_PG_CTRL_DI_FEATURE_XTAL_SRC_SWITCH_DISABLED            (0x0000)
#define LW2080_CTRL_LPWR_PG_CTRL_DI_FEATURE_XTAL_SRC_SWITCH_ENABLED             (0x0001)
#define LW2080_CTRL_LPWR_PG_CTRL_DI_FEATURE_SPPLLS                     11:11
#define LW2080_CTRL_LPWR_PG_CTRL_DI_FEATURE_SPPLLS_DISABLED                     (0x0000)
#define LW2080_CTRL_LPWR_PG_CTRL_DI_FEATURE_SPPLLS_ENABLED                      (0x0001)
#define LW2080_CTRL_LPWR_PG_CTRL_DI_FEATURE_XTAL4XPLL                  12:12
#define LW2080_CTRL_LPWR_PG_CTRL_DI_FEATURE_XTAL4XPLL_DISABLED                  (0x0000)
#define LW2080_CTRL_LPWR_PG_CTRL_DI_FEATURE_XTAL4XPLL_ENABLED                   (0x0001)
#define LW2080_CTRL_LPWR_PG_CTRL_DI_FEATURE_GPU_READY                  13:13
#define LW2080_CTRL_LPWR_PG_CTRL_DI_FEATURE_GPU_READY_DISABLED                  (0x0000)
#define LW2080_CTRL_LPWR_PG_CTRL_DI_FEATURE_GPU_READY_ENABLED                   (0x0001)
#define LW2080_CTRL_LPWR_PG_CTRL_DI_FEATURE_DEEP_L1                    14:14
#define LW2080_CTRL_LPWR_PG_CTRL_DI_FEATURE_DEEP_L1_DISABLED                    (0x0000)
#define LW2080_CTRL_LPWR_PG_CTRL_DI_FEATURE_DEEP_L1_ENABLED                     (0x0001)
#define LW2080_CTRL_LPWR_PG_CTRL_DI_FEATURE_L1SS                       15:15
#define LW2080_CTRL_LPWR_PG_CTRL_DI_FEATURE_L1SS_DISABLED                       (0x0000)
#define LW2080_CTRL_LPWR_PG_CTRL_DI_FEATURE_L1SS_ENABLED                        (0x0001)
#define LW2080_CTRL_LPWR_PG_CTRL_DI_FEATURE_RPPG                       16:16
#define LW2080_CTRL_LPWR_PG_CTRL_DI_FEATURE_RPPG_DISABLED                       (0x0000)
#define LW2080_CTRL_LPWR_PG_CTRL_DI_FEATURE_RPPG_ENABLED                        (0x0001)
#define LW2080_CTRL_LPWR_PG_CTRL_DI_FEATURE_PLL_IDDQ                   17:17
#define LW2080_CTRL_LPWR_PG_CTRL_DI_FEATURE_PLL_IDDQ_DISABLED                   (0x0000)
#define LW2080_CTRL_LPWR_PG_CTRL_DI_FEATURE_PLL_IDDQ_ENABLED                    (0x0001)

//
// Old Parameters/characteristics of AP
// TODO : Remove these params once LWAPI with changed param names gets checked in
//
#define LW2080_CTRL_LPWR_PARAMETER_ID_AP_GET_THRESHOLD_CNTR_0                   (0x0800)
#define LW2080_CTRL_LPWR_PARAMETER_ID_AP_GET_THRESHOLD_CNTR_1                   (0x0801)
#define LW2080_CTRL_LPWR_PARAMETER_ID_AP_GET_THRESHOLD_CNTR_2                   (0x0802)
#define LW2080_CTRL_LPWR_PARAMETER_ID_AP_GET_THRESHOLD_CNTR_3                   (0x0803)
#define LW2080_CTRL_LPWR_PARAMETER_ID_AP_GET_THRESHOLD_CNTR_4                   (0x0804)
#define LW2080_CTRL_LPWR_PARAMETER_ID_AP_GET_THRESHOLD_CNTR_5                   (0x0805)
#define LW2080_CTRL_LPWR_PARAMETER_ID_AP_GET_THRESHOLD_CNTR_6                   (0x0806)
#define LW2080_CTRL_LPWR_PARAMETER_ID_AP_GET_THRESHOLD_CNTR_7                   (0x0807)
#define LW2080_CTRL_LPWR_PARAMETER_ID_AP_GET_THRESHOLD_CNTR_8                   (0x0808)
#define LW2080_CTRL_LPWR_PARAMETER_ID_AP_GET_THRESHOLD_CNTR_9                   (0x0809)
#define LW2080_CTRL_LPWR_PARAMETER_ID_AP_GET_THRESHOLD_CNTR_10                  (0x080a)
#define LW2080_CTRL_LPWR_PARAMETER_ID_AP_GET_THRESHOLD_CNTR_11                  (0x080b)
#define LW2080_CTRL_LPWR_PARAMETER_ID_AP_GET_THRESHOLD_CNTR_12                  (0x080c)
#define LW2080_CTRL_LPWR_PARAMETER_ID_AP_GET_THRESHOLD_CNTR_13                  (0x080d)
#define LW2080_CTRL_LPWR_PARAMETER_ID_AP_GET_THRESHOLD_CNTR_14                  (0x080e)
#define LW2080_CTRL_LPWR_PARAMETER_ID_AP_GET_THRESHOLD_CNTR_15                  (0x080f)
// Param IDs 0x0810 - 0x081f reserved for future use.
#define LW2080_CTRL_LPWR_PARAMETER_ID_AP_GET_THRESHOLD_CNTR_DEFAULT             (0x0820)

// Parameters/characteristics of AP - BEGIN
#define LW2080_CTRL_LPWR_PARAMETER_ID_AP_BASE_INDEX                             (0x0001)
#define LW2080_CTRL_LPWR_PARAMETER_ID_AP_THRESHOLD_CNTR_DEFAULT                 (LW2080_CTRL_LPWR_PARAMETER_ID_AP_BASE_INDEX)
#define LW2080_CTRL_LPWR_PARAMETER_ID_AP_READ_PARAMS_FROM_PMU                   (0x2) /* finn: Evaluated from "(LW2080_CTRL_LPWR_PARAMETER_ID_AP_BASE_INDEX + 1)" */
#define LW2080_CTRL_LPWR_PARAMETER_ID_AP_LWRRENT_IDLE_THRESHOLD                 (0x3) /* finn: Evaluated from "(LW2080_CTRL_LPWR_PARAMETER_ID_AP_BASE_INDEX + 2)" */
#define LW2080_CTRL_LPWR_PARAMETER_ID_AP_ACTIVE                                 (0x4) /* finn: Evaluated from "(LW2080_CTRL_LPWR_PARAMETER_ID_AP_BASE_INDEX + 3)" */
#define LW2080_CTRL_LPWR_PARAMETER_ID_AP_IDLE_FILTER_X                          (0x5) /* finn: Evaluated from "(LW2080_CTRL_LPWR_PARAMETER_ID_AP_BASE_INDEX + 4)" */
#define LW2080_CTRL_LPWR_PARAMETER_ID_AP_POWER_SAVING_CYCLES                    (0x6) /* finn: Evaluated from "(LW2080_CTRL_LPWR_PARAMETER_ID_AP_BASE_INDEX + 5)" */
#define LW2080_CTRL_LPWR_PARAMETER_ID_AP_BAD_DECISION_COUNT                     (0x7) /* finn: Evaluated from "(LW2080_CTRL_LPWR_PARAMETER_ID_AP_BASE_INDEX + 6)" */
#define LW2080_CTRL_LPWR_PARAMETER_ID_AP_SKIP_COUNT                             (0x8) /* finn: Evaluated from "(LW2080_CTRL_LPWR_PARAMETER_ID_AP_BASE_INDEX + 7)" */
#define LW2080_CTRL_LPWR_PARAMETER_ID_AP_SUPPORT                                (0x9) /* finn: Evaluated from "(LW2080_CTRL_LPWR_PARAMETER_ID_AP_BASE_INDEX + 8)" */
#define LW2080_CTRL_LPWR_PARAMETER_ID_AP_ENABLE                                 (0xa) /* finn: Evaluated from "(LW2080_CTRL_LPWR_PARAMETER_ID_AP_BASE_INDEX + 9)" */
#define LW2080_CTRL_LPWR_PARAMETER_ID_AP_RESIDENCY                              (0xb) /* finn: Evaluated from "(LW2080_CTRL_LPWR_PARAMETER_ID_AP_BASE_INDEX + 10)" */
#define LW2080_CTRL_LPWR_PARAMETER_ID_AP_IDLE_THRESHOLD_MIN                     (0xc) /* finn: Evaluated from "(LW2080_CTRL_LPWR_PARAMETER_ID_AP_BASE_INDEX + 11)" */
#define LW2080_CTRL_LPWR_PARAMETER_ID_AP_IDLE_THRESHOLD_MAX                     (0xd) /* finn: Evaluated from "(LW2080_CTRL_LPWR_PARAMETER_ID_AP_BASE_INDEX + 12)" */
// Param IDs 0x000A - 0x000f reserved for future use.

#define LW2080_CTRL_LPWR_AP_THRESHOLD_COUNTER_BASE_INDEX                        (0x0800)
#define LW2080_CTRL_LPWR_PARAMETER_ID_AP_THRESHOLD_CNTR_0                       (LW2080_CTRL_LPWR_AP_THRESHOLD_COUNTER_BASE_INDEX)
#define LW2080_CTRL_LPWR_PARAMETER_ID_AP_THRESHOLD_CNTR_1                       (0x801) /* finn: Evaluated from "(LW2080_CTRL_LPWR_AP_THRESHOLD_COUNTER_BASE_INDEX + 1)" */
#define LW2080_CTRL_LPWR_PARAMETER_ID_AP_THRESHOLD_CNTR_2                       (0x802) /* finn: Evaluated from "(LW2080_CTRL_LPWR_AP_THRESHOLD_COUNTER_BASE_INDEX + 2)" */
#define LW2080_CTRL_LPWR_PARAMETER_ID_AP_THRESHOLD_CNTR_3                       (0x803) /* finn: Evaluated from "(LW2080_CTRL_LPWR_AP_THRESHOLD_COUNTER_BASE_INDEX + 3)" */
#define LW2080_CTRL_LPWR_PARAMETER_ID_AP_THRESHOLD_CNTR_4                       (0x804) /* finn: Evaluated from "(LW2080_CTRL_LPWR_AP_THRESHOLD_COUNTER_BASE_INDEX + 4)" */
#define LW2080_CTRL_LPWR_PARAMETER_ID_AP_THRESHOLD_CNTR_5                       (0x805) /* finn: Evaluated from "(LW2080_CTRL_LPWR_AP_THRESHOLD_COUNTER_BASE_INDEX + 5)" */
#define LW2080_CTRL_LPWR_PARAMETER_ID_AP_THRESHOLD_CNTR_6                       (0x806) /* finn: Evaluated from "(LW2080_CTRL_LPWR_AP_THRESHOLD_COUNTER_BASE_INDEX + 6)" */
#define LW2080_CTRL_LPWR_PARAMETER_ID_AP_THRESHOLD_CNTR_7                       (0x807) /* finn: Evaluated from "(LW2080_CTRL_LPWR_AP_THRESHOLD_COUNTER_BASE_INDEX + 7)" */
#define LW2080_CTRL_LPWR_PARAMETER_ID_AP_THRESHOLD_CNTR_8                       (0x808) /* finn: Evaluated from "(LW2080_CTRL_LPWR_AP_THRESHOLD_COUNTER_BASE_INDEX + 8)" */
#define LW2080_CTRL_LPWR_PARAMETER_ID_AP_THRESHOLD_CNTR_9                       (0x809) /* finn: Evaluated from "(LW2080_CTRL_LPWR_AP_THRESHOLD_COUNTER_BASE_INDEX + 9)" */
#define LW2080_CTRL_LPWR_PARAMETER_ID_AP_THRESHOLD_CNTR_10                      (0x80a) /* finn: Evaluated from "(LW2080_CTRL_LPWR_AP_THRESHOLD_COUNTER_BASE_INDEX + 10)" */
#define LW2080_CTRL_LPWR_PARAMETER_ID_AP_THRESHOLD_CNTR_11                      (0x80b) /* finn: Evaluated from "(LW2080_CTRL_LPWR_AP_THRESHOLD_COUNTER_BASE_INDEX + 11)" */
#define LW2080_CTRL_LPWR_PARAMETER_ID_AP_THRESHOLD_CNTR_12                      (0x80c) /* finn: Evaluated from "(LW2080_CTRL_LPWR_AP_THRESHOLD_COUNTER_BASE_INDEX + 12)" */
#define LW2080_CTRL_LPWR_PARAMETER_ID_AP_THRESHOLD_CNTR_13                      (0x80d) /* finn: Evaluated from "(LW2080_CTRL_LPWR_AP_THRESHOLD_COUNTER_BASE_INDEX + 13)" */
#define LW2080_CTRL_LPWR_PARAMETER_ID_AP_THRESHOLD_CNTR_14                      (0x80e) /* finn: Evaluated from "(LW2080_CTRL_LPWR_AP_THRESHOLD_COUNTER_BASE_INDEX + 14)" */
#define LW2080_CTRL_LPWR_PARAMETER_ID_AP_THRESHOLD_CNTR_15                      (0x80f) /* finn: Evaluated from "(LW2080_CTRL_LPWR_AP_THRESHOLD_COUNTER_BASE_INDEX + 15)" */
#define LW2080_CTRL_LPWR_PARAMETER_ID_AP_THRESHOLD_CNTR_MAX                     (LW2080_CTRL_LPWR_PARAMETER_ID_AP_THRESHOLD_CNTR_15)

#define LW2080_CTRL_LPWR_AP_BIN_BASE_INDEX                                      (0x810) /* finn: Evaluated from "(LW2080_CTRL_LPWR_PARAMETER_ID_AP_THRESHOLD_CNTR_MAX + 1)" */
#define LW2080_CTRL_LPWR_PARAMETER_ID_AP_BIN_0                                  (LW2080_CTRL_LPWR_AP_BIN_BASE_INDEX)
#define LW2080_CTRL_LPWR_PARAMETER_ID_AP_BIN_1                                  (0x811) /* finn: Evaluated from "(LW2080_CTRL_LPWR_AP_BIN_BASE_INDEX + 1)" */
#define LW2080_CTRL_LPWR_PARAMETER_ID_AP_BIN_2                                  (0x812) /* finn: Evaluated from "(LW2080_CTRL_LPWR_AP_BIN_BASE_INDEX + 2)" */
#define LW2080_CTRL_LPWR_PARAMETER_ID_AP_BIN_3                                  (0x813) /* finn: Evaluated from "(LW2080_CTRL_LPWR_AP_BIN_BASE_INDEX + 3)" */
#define LW2080_CTRL_LPWR_PARAMETER_ID_AP_BIN_4                                  (0x814) /* finn: Evaluated from "(LW2080_CTRL_LPWR_AP_BIN_BASE_INDEX + 4)" */
#define LW2080_CTRL_LPWR_PARAMETER_ID_AP_BIN_5                                  (0x815) /* finn: Evaluated from "(LW2080_CTRL_LPWR_AP_BIN_BASE_INDEX + 5)" */
#define LW2080_CTRL_LPWR_PARAMETER_ID_AP_BIN_6                                  (0x816) /* finn: Evaluated from "(LW2080_CTRL_LPWR_AP_BIN_BASE_INDEX + 6)" */
#define LW2080_CTRL_LPWR_PARAMETER_ID_AP_BIN_7                                  (0x817) /* finn: Evaluated from "(LW2080_CTRL_LPWR_AP_BIN_BASE_INDEX + 7)" */
#define LW2080_CTRL_LPWR_PARAMETER_ID_AP_BIN_8                                  (0x818) /* finn: Evaluated from "(LW2080_CTRL_LPWR_AP_BIN_BASE_INDEX + 8)" */
#define LW2080_CTRL_LPWR_PARAMETER_ID_AP_BIN_9                                  (0x819) /* finn: Evaluated from "(LW2080_CTRL_LPWR_AP_BIN_BASE_INDEX + 9)" */
#define LW2080_CTRL_LPWR_PARAMETER_ID_AP_BIN_10                                 (0x81a) /* finn: Evaluated from "(LW2080_CTRL_LPWR_AP_BIN_BASE_INDEX + 10)" */
#define LW2080_CTRL_LPWR_PARAMETER_ID_AP_BIN_11                                 (0x81b) /* finn: Evaluated from "(LW2080_CTRL_LPWR_AP_BIN_BASE_INDEX + 11)" */
#define LW2080_CTRL_LPWR_PARAMETER_ID_AP_BIN_12                                 (0x81c) /* finn: Evaluated from "(LW2080_CTRL_LPWR_AP_BIN_BASE_INDEX + 12)" */
#define LW2080_CTRL_LPWR_PARAMETER_ID_AP_BIN_13                                 (0x81d) /* finn: Evaluated from "(LW2080_CTRL_LPWR_AP_BIN_BASE_INDEX + 13)" */
#define LW2080_CTRL_LPWR_PARAMETER_ID_AP_BIN_14                                 (0x81e) /* finn: Evaluated from "(LW2080_CTRL_LPWR_AP_BIN_BASE_INDEX + 14)" */
#define LW2080_CTRL_LPWR_PARAMETER_ID_AP_BIN_15                                 (0x81f) /* finn: Evaluated from "(LW2080_CTRL_LPWR_AP_BIN_BASE_INDEX + 15)" */
#define LW2080_CTRL_LPWR_PARAMETER_ID_AP_BIN_MAX                                (LW2080_CTRL_LPWR_PARAMETER_ID_AP_BIN_15)
// Parameters/characteristics of AP - END

// Parameters/characteristics of Deep Idle - BEGIN
#define LW2080_CTRL_LPWR_PARAMETER_ID_DIDLE_BASE_INDEX                          (0x0001)
#define LW2080_CTRL_LPWR_PARAMETER_ID_DIDLE_PSTATE_SUPPORT_MASK                 (LW2080_CTRL_LPWR_PARAMETER_ID_DIDLE_BASE_INDEX)
#define LW2080_CTRL_LPWR_PARAMETER_ID_DIDLE_DIOS_ENABLED                        (0x2) /* finn: Evaluated from "(LW2080_CTRL_LPWR_PARAMETER_ID_DIDLE_BASE_INDEX + 1)" */
#define LW2080_CTRL_LPWR_PARAMETER_ID_DIDLE_SUPPORT                             (0x3) /* finn: Evaluated from "(LW2080_CTRL_LPWR_PARAMETER_ID_DIDLE_BASE_INDEX + 2)" */
// Parameters/characteristics of Deep Idle - END

// Parameters/characteristics of PGISLAND SCI_PMGR_GPIO_SYNC (Sub Features in PGISLAND)
#define LW2080_CTRL_LPWR_PARAMETER_ID_PGISLAND_SYNC_GPIO_PIN_SUPPORT            (0x0800)
#define LW2080_CTRL_LPWR_PARAMETER_ID_PGISLAND_SYNC_GPIO_PIN_MASK               (0x0801)

/*!
 * @brief Defines: Engine Idle Framework: Clients
 *
 * Following defines specifies unique IDs for clients of Engine Idle Framework(EI)
 *
 * @note These IDs are mapped to enum defining clientIds RM_PMU_LPWR_EI_CLIENT_ID_*
 * The below #defines need to be in sync with the enum values
 */
#define LW2080_CTRL_LPWR_EI_CLIENT_ID_RM                                        (0x0000)
#define LW2080_CTRL_LPWR_EI_CLIENT_ID_PMU_LPWR                                  (0x0001)
#define LW2080_CTRL_LPWR_EI_CLIENT_ID__COUNT                                    (0x0002)

#define LW2080_CTRL_LPWR_EI_CLIENT_ID__MAX                                      LW2080_CTRL_LPWR_EI_CLIENT_ID__COUNT
/*!
 * @brief Structure to identify power saving feature
 *
 * In general, Power saving feature is identify by featureId and SubFeature.
 * Add enum in this structure in case some specific power feature needs
 * additional fields. "Union" should follow XAPI standards.
 * https://wiki.lwpu.com/engwiki/index.php/Resman/RM_Foundations/Lwrrent_Projects/XAPI_and_XAPIGEN
 */
typedef struct LW2080_CTRL_LPWR_FEATURE {
    // Feature ID
    LwU16 id;

    // Sub Feature ID
    LwU16 subId;
} LW2080_CTRL_LPWR_FEATURE;

/*!
 * @brief Parameter structure
 *
 * Structure to get/set parameter/characteristic. Each parameter has 3 field
 * 1) ID    [In]     : Parameter Identifier
 * 2) Flag  [In/Out] : Flags
 * 3) Value [In/Out] : Value of parameter
 * 4) Info  [In/Out] : Additional Info for a parameter
 *
 * Add enum in this structure in case we need to additional fields for some
 * special parameters.
 */
typedef struct LW2080_CTRL_LPWR_PARAMETER {
    // Parameter ID
    LwU16 id;

    // Refer at "Flags for PARAMETER"
    LwU16 flag;

    // Parameter Value
    LwU32 val;

    //Additional info for a parameter
    LwU32 info;
} LW2080_CTRL_LPWR_PARAMETER;

/*!
 * @brief Flags for PARAMETER
 *
 * SUCCEED:
 * - Get/Set param call is succeed or not.
 * - Get Param call for given parameter succeed means RMCtrl retrieved valid
 *   value for this parameter.
 * - Set Param call for given parameter succeed means RMCtrl set value of this
 *   parameter.
 *
 * BLOCKING:
 * - Defines whether RM Ctrl call is blocking/non-blocking for given parameter.
 */
#define LW2080_CTRL_LPWR_FEATURE_PARAMETER_FLAG_SUCCEED                    0:0
#define LW2080_CTRL_LPWR_FEATURE_PARAMETER_FLAG_SUCCEED_NO        0x0
#define LW2080_CTRL_LPWR_FEATURE_PARAMETER_FLAG_SUCCEED_YES       0x1
#define LW2080_CTRL_LPWR_FEATURE_PARAMETER_FLAG_BLOCKING                   1:1
#define LW2080_CTRL_LPWR_FEATURE_PARAMETER_FLAG_BLOCKING_DISABLED 0x0
#define LW2080_CTRL_LPWR_FEATURE_PARAMETER_FLAG_BLOCKING_ENABLED  0x1

/*!
 * @brief Defines all information required to get/set the parameter for given
 *        power saving feature.
 */
typedef struct LW2080_CTRL_LPWR_FEATURE_PARAMETER {
    // Information to identity power saving feature
    LW2080_CTRL_LPWR_FEATURE   feature;

    // Parameter
    LW2080_CTRL_LPWR_PARAMETER param;
} LW2080_CTRL_LPWR_FEATURE_PARAMETER;


// Max size of FEATURE_PARAMETER structure for RMCtrl LW2080_CTRL_LPWR_GET/SET
#define LW2080_CTRL_LPWR_FEATURE_PARAMETER_LIST_MAX_SIZE (64)

/*
 * LW2080_CTRL_CMD_LPWR_FEATURE_PARAMETER_GET
 *
 * This command retrieves parameters/characteristics of power features. It can
 * query LW2080_CTRL_LPWR_FEATURE_PARAMETER_LIST_MAX_SIZE number of parameters
 * in one call. Command provides facility of collecting information on multiple
 * power saving features in one call.
 *
 * Commands returns SUCCESS only when it successfully retrieves value all
 * parameter in the list.
 *
 * listSize
 *      Number of valid entries in list.
 *
 * list
 *      List of parameters. Refer LW2080_CTRL_LPWR_FEATURE_PARAMETER to get
 *      details about each entry in the list.
 *
 * Possible status return values are:
 *  LW_OK
 *  LW_ERR_NOT_SUPPORTED
 *
 * Reference:
 * https://wiki.lwpu.com/engwiki/index.php/Resman/Resman_Components/LowPower/LPWR_Communication_Interfaces
 */
#define LW2080_CTRL_CMD_LPWR_FEATURE_PARAMETER_GET       (0x20802801) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_LPWR_INTERFACE_ID << 8) | LW2080_CTRL_LPWR_FEATURE_PARAMETER_GET_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_LPWR_FEATURE_PARAMETER_GET_PARAMS_MESSAGE_ID (0x1U)

typedef struct LW2080_CTRL_LPWR_FEATURE_PARAMETER_GET_PARAMS {
    LwU32                              listSize;
    LW2080_CTRL_LPWR_FEATURE_PARAMETER list[LW2080_CTRL_LPWR_FEATURE_PARAMETER_LIST_MAX_SIZE];
} LW2080_CTRL_LPWR_FEATURE_PARAMETER_GET_PARAMS;

/*
 * LW2080_CTRL_CMD_LPWR_FEATURE_PARAMETER_SET
 *
 * This command sets parameters/characteristics of power features. It can set
 * LW2080_CTRL_LPWR_FEATURE_PARAMETER_LIST_MAX_SIZE number of parameters in one
 * call. Command provides facility of setting parameters for multiple power
 * saving features in one call.
 *
 * Commands returns SUCCESS only when it successfully sets value of all
 * parameter in the list.
 *
 * listSize
 *      Number of valid entries in list.
 *
 * list
 *      List of parameters. Refer LW2080_CTRL_LPWR_FEATURE_PARAMETER to get
 *      details about each entry in the list.
 *
 * Possible status return values are:
 *  LW_OK
 *  LW_ERR_NOT_SUPPORTED
 *
 * Reference:
 * https://wiki.lwpu.com/engwiki/index.php/Resman/Resman_Components/LowPower/LPWR_Communication_Interfaces
 */
#define LW2080_CTRL_CMD_LPWR_FEATURE_PARAMETER_SET (0x20802802) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_LPWR_INTERFACE_ID << 8) | 0x2" */

typedef struct LW2080_CTRL_LPWR_FEATURE_PARAMETER_SET_PARAMS {
    LwU32                              listSize;
    LW2080_CTRL_LPWR_FEATURE_PARAMETER list[LW2080_CTRL_LPWR_FEATURE_PARAMETER_LIST_MAX_SIZE];
} LW2080_CTRL_LPWR_FEATURE_PARAMETER_SET_PARAMS;

/*!
 * LW2080_CTRL_CMD_LPWR_FAKE_I2CS_GET
 *
 * This command gets fake therm sensor support.
 * When the GPU is in GC6, we have the option to copy the therm I2CS address to
 * the SCI debug I2CS slave to fake the temp and prevent temp polling
 * from waking the GPU up.
 * This behaviour can be checked with this command.
 *
 */
#define LW2080_CTRL_CMD_LPWR_FAKE_I2CS_GET (0x20802803) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_LPWR_INTERFACE_ID << 8) | LW2080_CTRL_CMD_LPWR_FAKE_I2CS_GET_PARAMS_MESSAGE_ID" */

/*!
 * Structure containing fake therm sensor status
 */
#define LW2080_CTRL_CMD_LPWR_FAKE_I2CS_GET_PARAMS_MESSAGE_ID (0x3U)

typedef struct LW2080_CTRL_CMD_LPWR_FAKE_I2CS_GET_PARAMS {
    /*!
     * [out] - The current fake therm sensor enablement.
     */
    LwBool bFakeI2csEnabled;
    /*!
     * [out] - Flag specifying if fake sensor is supported on this GPU.
     */
    LwBool bFakeI2csSupported;
} LW2080_CTRL_CMD_LPWR_FAKE_I2CS_GET_PARAMS;

/*!
 * LW2080_CTRL_CMD_LPWR_FAKE_I2CS_SET
 *
 * This command sets fake therm sensor support.
 * When the GPU is in GC6, we have the option to copy the therm I2CS address to
 * the SCI debug I2CS slave to fake the temp and prevent temp polling
 * from waking the GPU up.
 * This behaviour can be modified with this command.
 *
 */
#define LW2080_CTRL_CMD_LPWR_FAKE_I2CS_SET (0x20802804) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_LPWR_INTERFACE_ID << 8) | LW2080_CTRL_CMD_LPWR_FAKE_I2CS_SET_PARAMS_MESSAGE_ID" */

/*!
 * Structure containing fake therm sensor status
 */
#define LW2080_CTRL_CMD_LPWR_FAKE_I2CS_SET_PARAMS_MESSAGE_ID (0x4U)

typedef struct LW2080_CTRL_CMD_LPWR_FAKE_I2CS_SET_PARAMS {
    /*!
     * [in] - The desired fake therm sensor support.
     */
    LwBool bFakeI2csEnabled;
} LW2080_CTRL_CMD_LPWR_FAKE_I2CS_SET_PARAMS;

/*!
 * LW2080_CTRL_CMD_LPWR_D3HOT_INFO
 *
 * This command retrieves D3Hot stats for GC6 and GOLD cycles,
 * and resets them if requested.
 * This behaviour can be modified with this command.
 *
 */
#define LW2080_CTRL_CMD_LPWR_D3HOT_INFO          (0x20802805) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_LPWR_INTERFACE_ID << 8) | LW2080_CTRL_CMD_LPWR_D3HOT_INFO_PARAMS_MESSAGE_ID" */

// Define for GC6 D3Hot stats variable interpretation
#define LW2080_CTRL_GCX_GC6_D3HOT_COUNT                             30: 0
#define LW2080_CTRL_GCX_GC6_D3HOT_LATEST                            31:31
#define LW2080_CTRL_GCX_GC6_D3HOT_LATEST_FALSE   (0x00000000)
#define LW2080_CTRL_GCX_GC6_D3HOT_LATEST_TRUE    (0x00000001)

// Define for GCOFF D3Hot stats variable interpretation
#define LW2080_CTRL_GCX_GCOFF_D3HOT_COUNT                           30: 0
#define LW2080_CTRL_GCX_GCOFF_D3HOT_LATEST                          31:31
#define LW2080_CTRL_GCX_GCOFF_D3HOT_LATEST_FALSE (0x00000000)
#define LW2080_CTRL_GCX_GCOFF_D3HOT_LATEST_TRUE  (0x00000001)

/*!
 * Structure containing D3Hot info
 */
#define LW2080_CTRL_CMD_LPWR_D3HOT_INFO_PARAMS_MESSAGE_ID (0x5U)

typedef struct LW2080_CTRL_CMD_LPWR_D3HOT_INFO_PARAMS {
    LwU32  goldD3HotCount;
    LwU32  gc6D3HotCount;
    LwBool bClearStats;
    LwBool bIsLastGC6D3Hot;
    LwBool bIsLastGCOffD3Hot;
} LW2080_CTRL_CMD_LPWR_D3HOT_INFO_PARAMS;

/*!
 * LW2080_CTRL_CMD_LPWR_DIFR_CTRL
 *
 * This command is used to control the DIFR
 * feature behavior.
 *
 */
#define LW2080_CTRL_CMD_LPWR_DIFR_CTRL            (0x20802806) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_LPWR_INTERFACE_ID << 8) | LW2080_CTRL_CMD_LPWR_DIFR_CTRL_PARAMS_MESSAGE_ID" */

/*!
 * @brief Various Values for control
 */
// Disable the DIFR
#define LW2080_CTRL_LPWR_DIFR_CTRL_DISABLE        (0x00000001)
// Enable the DIFR
#define LW2080_CTRL_LPWR_DIFR_CTRL_ENABLE         (0x00000002)

// Support status for DIFR
#define LW2080_CTRL_LPWR_DIFR_CTRL_SUPPORT_STATUS (0x00000003)

/*!
 * Structure containing DIFR control call Parameters
 */
#define LW2080_CTRL_CMD_LPWR_DIFR_CTRL_PARAMS_MESSAGE_ID (0x6U)

typedef struct LW2080_CTRL_CMD_LPWR_DIFR_CTRL_PARAMS {
    LwU32 ctrlParamVal;
} LW2080_CTRL_CMD_LPWR_DIFR_CTRL_PARAMS;

// Values for the SUPPORT Control Status
#define LW2080_CTRL_LPWR_DIFR_SUPPORTED                          (0x00000001)
#define LW2080_CTRL_LPWR_DIFR_NOT_SUPPORTED                      (0x00000002)

/*!
 * LW2080_CTRL_CMD_LPWR_DIFR_PREFETCH_RESPONSE
 *
 * This control call is used to send the prefetch response
 *
 */
#define LW2080_CTRL_CMD_LPWR_DIFR_PREFETCH_RESPONSE              (0x20802807) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_LPWR_INTERFACE_ID << 8) | LW2080_CTRL_CMD_LPWR_DIFR_PREFETCH_RESPONSE_PARAMS_MESSAGE_ID" */

/*!
 * @brief Various Values of Reponses for Prefetch Status
 */

// Prefetch is successfull.
#define LW2080_CTRL_LPWR_DIFR_PREFETCH_SUCCESS                   (0x00000001)
// OS Filps are enabled, so prefetch can not be done.
#define LW2080_CTRL_LPWR_DIFR_PREFETCH_FAIL_OS_FLIPS_ENABLED     (0x00000002)
// Current Display surface can not fit in L2
#define LW2080_CTRL_LPWR_DIFR_PREFETCH_FAIL_INSUFFICIENT_L2_SIZE (0x00000003)
// Fatal and un recoverable Error
#define LW2080_CTRL_LPWR_DIFR_PREFETCH_FAIL_CE_HW_ERROR          (0x00000004)

/*!
 * Structure containing DIFR prefetch response control call Parameters
 */
#define LW2080_CTRL_CMD_LPWR_DIFR_PREFETCH_RESPONSE_PARAMS_MESSAGE_ID (0x7U)

typedef struct LW2080_CTRL_CMD_LPWR_DIFR_PREFETCH_RESPONSE_PARAMS {
    LwU32 responseVal;
} LW2080_CTRL_CMD_LPWR_DIFR_PREFETCH_RESPONSE_PARAMS;
// _ctrl2080lpwr_h_

#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)

