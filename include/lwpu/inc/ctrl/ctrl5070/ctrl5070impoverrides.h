/*
 * SPDX-FileCopyrightText: Copyright (c) 2001-2021 LWPU CORPORATION & AFFILIATES. All rights reserved.
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

//
// This file was generated with FINN, an LWPU coding tool.
// Source file: ctrl/ctrl5070/ctrl5070impoverrides.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)




#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

/*
 * DISPIMP override inputs 
 */
#define DISP_INPUT_INDEX_BEGIN                                         0
#define DISP_INPUT_INDEX_END                                           (0x1) /* finn: Evaluated from "DISP_INPUT_INDEX_BEGIN + 0x1" */

/*
 *    FBIMP override inputs 
 */
#define FB_INPUT_INDEX_BEGIN                                           0x1000
#define FB_INPUT_INDEX_END                                             (0x1001) /* finn: Evaluated from "FB_INPUT_INDEX_BEGIN + 0x1" */

/*
 * IMP override outputs 
 */
#define IMP_OUTPUT_INDEX_BEGIN                                         0x8000
#define LW5070_CTRL_IMP_OVERRIDE_OUTPUT_RESULT                         (0x8001) /* finn: Evaluated from "IMP_OUTPUT_INDEX_BEGIN + 0x1" */
#define LW5070_CTRL_IMP_OVERRIDE_OUTPUT_MEMPOOL                        (0x8002) /* finn: Evaluated from "IMP_OUTPUT_INDEX_BEGIN + 0x2" */
#define LW5070_CTRL_IMP_OVERRIDE_OUTPUT_FB_FETCHRATE_BYTESPERSEC       (0x8003) /* finn: Evaluated from "IMP_OUTPUT_INDEX_BEGIN + 0x3" */
#define LW5070_CTRL_IMP_OVERRIDE_OUTPUT_MEMPOOL_BASE_CORE_BPP          (0x8004) /* finn: Evaluated from "IMP_OUTPUT_INDEX_BEGIN + 0x4" */
#define LW5070_CTRL_IMP_OVERRIDE_OUTPUT_MEMPOOL_OVLY_BPP               (0x8005) /* finn: Evaluated from "IMP_OUTPUT_INDEX_BEGIN + 0x5" */
#define LW5070_CTRL_IMP_OVERRIDE_OUTPUT_MIN_CLK_RAM                    (0x8006) /* finn: Evaluated from "IMP_OUTPUT_INDEX_BEGIN + 0x6" */
#define LW5070_CTRL_IMP_OVERRIDE_OUTPUT_MIN_CLK_L2                     (0x8007) /* finn: Evaluated from "IMP_OUTPUT_INDEX_BEGIN + 0x7" */
#define LW5070_CTRL_IMP_OVERRIDE_OUTPUT_MIN_CLK_XBAR                   (0x8008) /* finn: Evaluated from "IMP_OUTPUT_INDEX_BEGIN + 0x8" */
#define LW5070_CTRL_IMP_OVERRIDE_OUTPUT_MIN_CLK_HUB                    (0x8009) /* finn: Evaluated from "IMP_OUTPUT_INDEX_BEGIN + 0x9" */
#define LW5070_CTRL_IMP_OVERRIDE_OUTPUT_ASR_MIN_CLK_RAM                (0x800a) /* finn: Evaluated from "IMP_OUTPUT_INDEX_BEGIN + 0xA" */
#define LW5070_CTRL_IMP_OVERRIDE_OUTPUT_ASR_MIN_CLK_L2                 (0x800b) /* finn: Evaluated from "IMP_OUTPUT_INDEX_BEGIN + 0xB" */
#define LW5070_CTRL_IMP_OVERRIDE_OUTPUT_ASR_MIN_CLK_XBAR               (0x800c) /* finn: Evaluated from "IMP_OUTPUT_INDEX_BEGIN + 0xC" */
#define LW5070_CTRL_IMP_OVERRIDE_OUTPUT_ASR_MIN_CLK_HUB                (0x800d) /* finn: Evaluated from "IMP_OUTPUT_INDEX_BEGIN + 0xD" */
#define LW5070_CTRL_IMP_OVERRIDE_OUTPUT_ASR_tXSR                       (0x800e) /* finn: Evaluated from "IMP_OUTPUT_INDEX_BEGIN + 0xE" */
#define LW5070_CTRL_IMP_OVERRIDE_OUTPUT_ASR_ISVALID                    (0x800f) /* finn: Evaluated from "IMP_OUTPUT_INDEX_BEGIN + 0xF" */
#define LW5070_CTRL_IMP_OVERRIDE_OUTPUT_DISP_ISPOSSIBLE                (0x8010) /* finn: Evaluated from "IMP_OUTPUT_INDEX_BEGIN + 0x10" */
#define LW5070_CTRL_IMP_OVERRIDE_OUTPUT_DISP_REQD_DISPCLK              (0x8011) /* finn: Evaluated from "IMP_OUTPUT_INDEX_BEGIN + 0x11" */
#define LW5070_CTRL_IMP_OVERRIDE_OUTPUT_DISP_MAX_DISPCLK               (0x8012) /* finn: Evaluated from "IMP_OUTPUT_INDEX_BEGIN + 0x12" */
#define LW5070_CTRL_IMP_OVERRIDE_OUTPUT_DISP_REQD_DISPCLK_RATIO        (0x8013) /* finn: Evaluated from "IMP_OUTPUT_INDEX_BEGIN + 0x13" */
#define LW5070_CTRL_IMP_OVERRIDE_OUTPUT_DISP_FB_FETCHRATE_KHZ          (0x8014) /* finn: Evaluated from "IMP_OUTPUT_INDEX_BEGIN + 0x14" */
#define LW5070_CTRL_IMP_OVERRIDE_OUTPUT_DISP_METER                     (0x8015) /* finn: Evaluated from "IMP_OUTPUT_INDEX_BEGIN + 0x15" */
#define LW5070_CTRL_IMP_OVERRIDE_OUTPUT_DISP_CLK_ACTUAL_KHZ            (0x8016) /* finn: Evaluated from "IMP_OUTPUT_INDEX_BEGIN + 0x16" */
#define LW5070_CTRL_IMP_OVERRIDE_OUTPUT_DISP_MIN_FILLRATE_KHZ          (0x8017) /* finn: Evaluated from "IMP_OUTPUT_INDEX_BEGIN + 0x17" */
#define LW5070_CTRL_IMP_OVERRIDE_OUTPUT_DISP_AVG_FETCHRATE_KHZ         (0x8018) /* finn: Evaluated from "IMP_OUTPUT_INDEX_BEGIN + 0x18" */
#define LW5070_CTRL_IMP_OVERRIDE_OUTPUT_ISOHUB_ISPOSSIBLE              (0x8019) /* finn: Evaluated from "IMP_OUTPUT_INDEX_BEGIN + 0x19" */
#define LW5070_CTRL_IMP_OVERRIDE_OUTPUT_ISOHUB_MEMPOOL_BASEFULLY_LB    (0x801a) /* finn: Evaluated from "IMP_OUTPUT_INDEX_BEGIN + 0x1A" */
#define LW5070_CTRL_IMP_OVERRIDE_OUTPUT_ISOHUB_MEMPOOL_OVLYFULLY_LB    (0x801b) /* finn: Evaluated from "IMP_OUTPUT_INDEX_BEGIN + 0x1B" */
#define LW5070_CTRL_IMP_OVERRIDE_OUTPUT_ISOHUB_MEMPOOL_BASELIMIT       (0x801c) /* finn: Evaluated from "IMP_OUTPUT_INDEX_BEGIN + 0x1C" */
#define LW5070_CTRL_IMP_OVERRIDE_OUTPUT_ISOHUB_MEMPOOL_OVLYLIMIT       (0x801d) /* finn: Evaluated from "IMP_OUTPUT_INDEX_BEGIN + 0x1D" */
#define LW5070_CTRL_IMP_OVERRIDE_OUTPUT_ISOHUB_MEMPOOL_LAT_BUFBLCKS    (0x801e) /* finn: Evaluated from "IMP_OUTPUT_INDEX_BEGIN + 0x1E" */
#define LW5070_CTRL_IMP_OVERRIDE_OUTPUT_ISOHUB_MEMPOOL_MEMPOOLBLCKS    (0x801f) /* finn: Evaluated from "IMP_OUTPUT_INDEX_BEGIN + 0x1F" */
#define LW5070_CTRL_IMP_OVERRIDE_OUTPUT_ISOHUB_FREEMEMPOOLBLCKS        (0x8020) /* finn: Evaluated from "IMP_OUTPUT_INDEX_BEGIN + 0x20" */
#define LW5070_CTRL_IMP_OVERRIDE_OUTPUT_ISOHUB_REQDHUBCLK              (0x8021) /* finn: Evaluated from "IMP_OUTPUT_INDEX_BEGIN + 0x21" */
#define LW5070_CTRL_IMP_OVERRIDE_OUTPUT_ISOHUB_FETCHMETER              (0x8022) /* finn: Evaluated from "IMP_OUTPUT_INDEX_BEGIN + 0x22" */
#define LW5070_CTRL_IMP_OVERRIDE_OUTPUT_ISOHUB_FETCHLIMIT              (0x8023) /* finn: Evaluated from "IMP_OUTPUT_INDEX_BEGIN + 0x23" */
#define LW5070_CTRL_IMP_OVERRIDE_OUTPUT_ISOHUB_IS_ASR_POSSIBLE         (0x8024) /* finn: Evaluated from "IMP_OUTPUT_INDEX_BEGIN + 0x24" */
#define LW5070_CTRL_IMP_OVERRIDE_OUTPUT_ISOHUB_ASR_EFFICIENCY          (0x8025) /* finn: Evaluated from "IMP_OUTPUT_INDEX_BEGIN + 0x25" */
#define LW5070_CTRL_IMP_OVERRIDE_OUTPUT_ISOHUB_ASR_SETTING_MODE        (0x8026) /* finn: Evaluated from "IMP_OUTPUT_INDEX_BEGIN + 0x26" */
#define LW5070_CTRL_IMP_OVERRIDE_OUTPUT_ISOHUB_ASR_SETTING_SUBMODE     (0x8027) /* finn: Evaluated from "IMP_OUTPUT_INDEX_BEGIN + 0x27" */
#define LW5070_CTRL_IMP_OVERRIDE_OUTPUT_ISOHUB_ASR_SETTING_LWM         (0x8028) /* finn: Evaluated from "IMP_OUTPUT_INDEX_BEGIN + 0x28" */
#define LW5070_CTRL_IMP_OVERRIDE_OUTPUT_ISOHUB_ASR_SETTING_HWM         (0x8029) /* finn: Evaluated from "IMP_OUTPUT_INDEX_BEGIN + 0x29" */
#define LW5070_CTRL_IMP_OVERRIDE_OUTPUT_ISOHUB_IS_MCLK_SWITCH_POSSIBLE (0x802a) /* finn: Evaluated from "IMP_OUTPUT_INDEX_BEGIN + 0x2A" */
#define LW5070_CTRL_IMP_OVERRIDE_OUTPUT_ISOHUB_MCLK_OVERRIDE_MEMPOOL   (0x802b) /* finn: Evaluated from "IMP_OUTPUT_INDEX_BEGIN + 0x2B" */
#define LW5070_CTRL_IMP_OVERRIDE_OUTPUT_ISOHUB_MCLK_SETTING_MODE       (0x802c) /* finn: Evaluated from "IMP_OUTPUT_INDEX_BEGIN + 0x2C" */
#define LW5070_CTRL_IMP_OVERRIDE_OUTPUT_ISOHUB_MCLK_SETTING_SUBMODE    (0x802d) /* finn: Evaluated from "IMP_OUTPUT_INDEX_BEGIN + 0x2D" */
#define LW5070_CTRL_IMP_OVERRIDE_OUTPUT_ISOHUB_MCLK_SETTING_DWCFWM     (0x802e) /* finn: Evaluated from "IMP_OUTPUT_INDEX_BEGIN + 0x2E" */
#define LW5070_CTRL_IMP_OVERRIDE_OUTPUT_ISOHUB_MCLK_SETTING_MIDWM      (0x802f) /* finn: Evaluated from "IMP_OUTPUT_INDEX_BEGIN + 0x2F" */
#define LW5070_CTRL_IMP_OVERRIDE_OUTPUT_ISOHUB_MCLK_MEMPOOL            (0x8030) /* finn: Evaluated from "IMP_OUTPUT_INDEX_BEGIN + 0x30" */
#define LW5070_CTRL_IMP_OVERRIDE_OUTPUT_ISOHUB_DMI_DURATION            (0x8031) /* finn: Evaluated from "IMP_OUTPUT_INDEX_BEGIN + 0x31" */
#define LW5070_CTRL_IMP_OVERRIDE_OUTPUT_ISOHUB_BLANK_PIXELS            (0x8032) /* finn: Evaluated from "IMP_OUTPUT_INDEX_BEGIN + 0x32" */ // obsolete
#define LW5070_CTRL_IMP_OVERRIDE_OUTPUT_ISOHUB_IS_MSPG_POSSIBLE        (0x8033) /* finn: Evaluated from "IMP_OUTPUT_INDEX_BEGIN + 0x33" */
#define LW5070_CTRL_IMP_OVERRIDE_OUTPUT_ISOHUB_MSPG_EFFICIENCY         (0x8034) /* finn: Evaluated from "IMP_OUTPUT_INDEX_BEGIN + 0x34" */
#define LW5070_CTRL_IMP_OVERRIDE_OUTPUT_ISOHUB_MSPG_SETTING_MODE       (0x8035) /* finn: Evaluated from "IMP_OUTPUT_INDEX_BEGIN + 0x35" */
#define LW5070_CTRL_IMP_OVERRIDE_OUTPUT_ISOHUB_MSPG_SETTING_LWM        (0x8036) /* finn: Evaluated from "IMP_OUTPUT_INDEX_BEGIN + 0x36" */
#define LW5070_CTRL_IMP_OVERRIDE_OUTPUT_ISOHUB_MSPG_SETTING_HWM        (0x8037) /* finn: Evaluated from "IMP_OUTPUT_INDEX_BEGIN + 0x37" */
#define LW5070_CTRL_IMP_OVERRIDE_OUTPUT_ISOHUB_MSPG_WAKEUP_WATERMARK   (0x8038) /* finn: Evaluated from "IMP_OUTPUT_INDEX_BEGIN + 0x38" */
#define LW5070_CTRL_IMP_OVERRIDE_OUTPUT_ISOHUB_CRITICAL_WATERMARK      (0x8039) /* finn: Evaluated from "IMP_OUTPUT_INDEX_BEGIN + 0x39" */
#define LW5070_CTRL_IMP_OVERRIDE_OUTPUT_ISOHUB_CRITICAL_MODE           (0x8040) /* finn: Evaluated from "IMP_OUTPUT_INDEX_BEGIN + 0x40" */
#define LW5070_CTRL_IMP_OVERRIDE_OUTPUT_MIN_CLK_SYS                    (0x8041) /* finn: Evaluated from "IMP_OUTPUT_INDEX_BEGIN + 0x41" */
#define LW5070_CTRL_IMP_OVERRIDE_OUTPUT_ASR_MIN_CLK_SYS                (0x8042) /* finn: Evaluated from "IMP_OUTPUT_INDEX_BEGIN + 0x42" */
#define IMP_OUTPUT_INDEX_END                                           (0x8043) /* finn: Evaluated from "IMP_OUTPUT_INDEX_BEGIN + 0x43" */

/* 
 * CONTROL overrides
 */
#define OVERRIDES_CONTROL_INDEX_BEGIN                                  0xFF00
#define LW5070_CTRL_IMP_OVERRIDE_SET_HEAD_NUMBER                       (0xff01) /* finn: Evaluated from "OVERRIDES_CONTROL_INDEX_BEGIN + 0x1" */
#define LW5070_CTRL_IMP_OVERRIDE_SET_INDEX_LEVEL_1                     (0xff02) /* finn: Evaluated from "OVERRIDES_CONTROL_INDEX_BEGIN + 0x2" */
#define LW5070_CTRL_IMP_OVERRIDE_SET_INDEX_LEVEL_2                     (0xff03) /* finn: Evaluated from "OVERRIDES_CONTROL_INDEX_BEGIN + 0x3" */
#define LW5070_CTRL_IMP_OVERRIDE_SET_INDEX_LEVEL_3                     (0xff04) /* finn: Evaluated from "OVERRIDES_CONTROL_INDEX_BEGIN + 0x4" */
#define LW5070_CTRL_IMP_OVERRIDE_CLEAR_ALL_OVERRIDES                   (0xff05) /* finn: Evaluated from "OVERRIDES_CONTROL_INDEX_BEGIN + 0x5" */
#define OVERRIDES_CONTROL_INDEX_END                                    (0xff06) /* finn: Evaluated from "OVERRIDES_CONTROL_INDEX_BEGIN + 0x6" */
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

