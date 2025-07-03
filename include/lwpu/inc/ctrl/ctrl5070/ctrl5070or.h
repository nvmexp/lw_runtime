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

#include <lwtypes.h>
#if defined(_MSC_VER)
#pragma warning(disable:4324)
#endif

//
// This file was generated with FINN, an LWPU coding tool.
// Source file: ctrl/ctrl5070/ctrl5070or.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

#include "ctrl5070common.h"



#include "ctrl/ctrl5070/ctrl5070base.h"

#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

/*
 * LW5070_CTRL_CMD_SET_DAC_CONFIG
 *
 * This command configures the TV dac settings.
 *
 *      orNumber
 *          The dac for which the settings need to be programmed.
 *
 *      cpstDac0Src
 *          Type of input to dac0 channel when using a composite TV protocol.
 *
 *      cpstDac1Src
 *          Type of input to dac1 channel when using a composite TV protocol.
 *
 *      cpstDac2Src
 *          Type of input to dac2 channel when using a composite TV protocol.
 *
 *      cpstDac3Src
 *          Type of input to dac3 channel when using a composite TV protocol.
 *
 *      compDac0Src
 *          Type of input to dac0 channel when using a component TV protocol.
 *
 *      compDac1Src
 *          Type of input to dac1 channel when using a component TV protocol.
 *
 *      compDac2Src
 *          Type of input to dac2 channel when using a component TV protocol.
 *
 *      compDac3Src
 *          Type of input to dac3 channel when using a component TV protocol.
 *
 *      driveSync
 *          Should the h and v sync pins be driven or tristated.
 *
 * Possible status values returned are:
 *      LW_OK
 *      LW_ERR_ILWALID_ARGUMENT
 *      LW_ERR_GENERIC
 */
#define LW5070_CTRL_CMD_SET_DAC_CONFIG                              (0x50700402) /* finn: Evaluated from "(FINN_LW50_DISPLAY_OR_INTERFACE_ID << 8) | LW5070_CTRL_CMD_SET_DAC_CONFIG_PARAMS_MESSAGE_ID" */

#define LW5070_CTRL_CMD_SET_DAC_CONFIG_CPST_DAC0_SRC                        1:0
#define LW5070_CTRL_CMD_SET_DAC_CONFIG_CPST_DAC0_SRC_NONE           0x00000000
#define LW5070_CTRL_CMD_SET_DAC_CONFIG_CPST_DAC0_SRC_COMPOSITE      0x00000001
#define LW5070_CTRL_CMD_SET_DAC_CONFIG_CPST_DAC0_SRC_SVIDEO_Y       0x00000002
#define LW5070_CTRL_CMD_SET_DAC_CONFIG_CPST_DAC0_SRC_SVIDEO_C       0x00000003

#define LW5070_CTRL_CMD_SET_DAC_CONFIG_CPST_DAC1_SRC                        1:0
#define LW5070_CTRL_CMD_SET_DAC_CONFIG_CPST_DAC1_SRC_NONE           0x00000000
#define LW5070_CTRL_CMD_SET_DAC_CONFIG_CPST_DAC1_SRC_COMPOSITE      0x00000001
#define LW5070_CTRL_CMD_SET_DAC_CONFIG_CPST_DAC1_SRC_SVIDEO_Y       0x00000002
#define LW5070_CTRL_CMD_SET_DAC_CONFIG_CPST_DAC1_SRC_SVIDEO_C       0x00000003

#define LW5070_CTRL_CMD_SET_DAC_CONFIG_CPST_DAC2_SRC                        1:0
#define LW5070_CTRL_CMD_SET_DAC_CONFIG_CPST_DAC2_SRC_NONE           0x00000000
#define LW5070_CTRL_CMD_SET_DAC_CONFIG_CPST_DAC2_SRC_COMPOSITE      0x00000001
#define LW5070_CTRL_CMD_SET_DAC_CONFIG_CPST_DAC2_SRC_SVIDEO_Y       0x00000002
#define LW5070_CTRL_CMD_SET_DAC_CONFIG_CPST_DAC2_SRC_SVIDEO_C       0x00000003

#define LW5070_CTRL_CMD_SET_DAC_CONFIG_CPST_DAC3_SRC                        1:0
#define LW5070_CTRL_CMD_SET_DAC_CONFIG_CPST_DAC3_SRC_NONE           0x00000000
#define LW5070_CTRL_CMD_SET_DAC_CONFIG_CPST_DAC3_SRC_COMPOSITE      0x00000001
#define LW5070_CTRL_CMD_SET_DAC_CONFIG_CPST_DAC3_SRC_SVIDEO_Y       0x00000002
#define LW5070_CTRL_CMD_SET_DAC_CONFIG_CPST_DAC3_SRC_SVIDEO_C       0x00000003

#define LW5070_CTRL_CMD_SET_DAC_CONFIG_COMP_DAC0_SRC                        2:0
#define LW5070_CTRL_CMD_SET_DAC_CONFIG_COMP_DAC0_SRC_NONE           0x00000000
#define LW5070_CTRL_CMD_SET_DAC_CONFIG_COMP_DAC0_SRC_COMPONENT_G_Y  0x00000001
#define LW5070_CTRL_CMD_SET_DAC_CONFIG_COMP_DAC0_SRC_COMPONENT_R_PR 0x00000002
#define LW5070_CTRL_CMD_SET_DAC_CONFIG_COMP_DAC0_SRC_COMPONENT_B_PB 0x00000003
#define LW5070_CTRL_CMD_SET_DAC_CONFIG_COMP_DAC0_SRC_COMPOSITE      0x00000004
#define LW5070_CTRL_CMD_SET_DAC_CONFIG_COMP_DAC0_SRC_COMPONENT_Y    0x00000005


#define LW5070_CTRL_CMD_SET_DAC_CONFIG_COMP_DAC1_SRC                        2:0
#define LW5070_CTRL_CMD_SET_DAC_CONFIG_COMP_DAC1_SRC_NONE           0x00000000
#define LW5070_CTRL_CMD_SET_DAC_CONFIG_COMP_DAC1_SRC_COMPONENT_G_Y  0x00000001
#define LW5070_CTRL_CMD_SET_DAC_CONFIG_COMP_DAC1_SRC_COMPONENT_R_PR 0x00000002
#define LW5070_CTRL_CMD_SET_DAC_CONFIG_COMP_DAC1_SRC_COMPONENT_B_PB 0x00000003
#define LW5070_CTRL_CMD_SET_DAC_CONFIG_COMP_DAC1_SRC_COMPOSITE      0x00000004
#define LW5070_CTRL_CMD_SET_DAC_CONFIG_COMP_DAC1_SRC_COMPONENT_Y    0x00000005

#define LW5070_CTRL_CMD_SET_DAC_CONFIG_COMP_DAC2_SRC                        2:0
#define LW5070_CTRL_CMD_SET_DAC_CONFIG_COMP_DAC2_SRC_NONE           0x00000000
#define LW5070_CTRL_CMD_SET_DAC_CONFIG_COMP_DAC2_SRC_COMPONENT_G_Y  0x00000001
#define LW5070_CTRL_CMD_SET_DAC_CONFIG_COMP_DAC2_SRC_COMPONENT_R_PR 0x00000002
#define LW5070_CTRL_CMD_SET_DAC_CONFIG_COMP_DAC2_SRC_COMPONENT_B_PB 0x00000003
#define LW5070_CTRL_CMD_SET_DAC_CONFIG_COMP_DAC2_SRC_COMPOSITE      0x00000004
#define LW5070_CTRL_CMD_SET_DAC_CONFIG_COMP_DAC2_SRC_COMPONENT_Y    0x00000005

#define LW5070_CTRL_CMD_SET_DAC_CONFIG_COMP_DAC3_SRC                        2:0
#define LW5070_CTRL_CMD_SET_DAC_CONFIG_COMP_DAC3_SRC_NONE           0x00000000
#define LW5070_CTRL_CMD_SET_DAC_CONFIG_COMP_DAC3_SRC_COMPONENT_G_Y  0x00000001
#define LW5070_CTRL_CMD_SET_DAC_CONFIG_COMP_DAC3_SRC_COMPONENT_R_PR 0x00000002
#define LW5070_CTRL_CMD_SET_DAC_CONFIG_COMP_DAC3_SRC_COMPONENT_B_PB 0x00000003
#define LW5070_CTRL_CMD_SET_DAC_CONFIG_COMP_DAC3_SRC_COMPOSITE      0x00000004
#define LW5070_CTRL_CMD_SET_DAC_CONFIG_COMP_DAC3_SRC_COMPONENT_Y    0x00000005

#define LW5070_CTRL_CMD_SET_DAC_CONFIG_DRIVE_SYNC                           0:0
#define LW5070_CTRL_CMD_SET_DAC_CONFIG_DRIVE_SYNC_NO                0x00000000
#define LW5070_CTRL_CMD_SET_DAC_CONFIG_DRIVE_SYNC_YES               0x00000001

#define LW5070_CTRL_CMD_SET_DAC_CONFIG_PARAMS_MESSAGE_ID (0x2U)

typedef struct LW5070_CTRL_CMD_SET_DAC_CONFIG_PARAMS {
    LW5070_CTRL_CMD_BASE_PARAMS base;
    LwU32                       orNumber;

    LwU32                       cpstDac0Src;
    LwU32                       cpstDac1Src;
    LwU32                       cpstDac2Src;
    LwU32                       cpstDac3Src;
    LwU32                       compDac0Src;
    LwU32                       compDac1Src;
    LwU32                       compDac2Src;
    LwU32                       compDac3Src;
    LwU32                       driveSync;
} LW5070_CTRL_CMD_SET_DAC_CONFIG_PARAMS;

/*
 * LW5070_CTRL_CMD_GET_DAC_PWR
 *
 * This command gets the DAC power control register for specified orNumber.
 *
 *      orNumber
 *          The dac for which the settings need to be read.
 *
 *      normalHSync
 *          The normal operating state for the H sync signal.
 *
 *      normalVSync
 *          The normal operating state for the V sync signal.
 *
 *      normalData
 *          The normal video data input pin of the d/a colwerter.
 *
 *      normalPower
 *          The normal state of the dac macro power.
 *
 *      safeHSync
 *          The safe operating state for the H sync signal.
 *
 *      safeVSync
 *          The safe operating state for the V sync signal.
 *
 *      safeData
 *          The safe video data input pin of the d/a colwerter.
 *
 *      safePower
 *          The safe state of the dac macro power.
 *
 * Possible status values returned are:
 *      LW_OK
 *      LW_ERR_ILWALID_ARGUMENT
 *      LW_ERR_GENERIC
 *      LW_ERR_TIMEOUT
 */
#define LW5070_CTRL_CMD_GET_DAC_PWR                     (0x50700403) /* finn: Evaluated from "(FINN_LW50_DISPLAY_OR_INTERFACE_ID << 8) | LW5070_CTRL_CMD_GET_DAC_PWR_PARAMS_MESSAGE_ID" */

#define LW5070_CTRL_CMD_GET_DAC_PWR_NORMAL_HSYNC                            1:0
#define LW5070_CTRL_CMD_GET_DAC_PWR_NORMAL_HSYNC_ENABLE (0x00000000)
#define LW5070_CTRL_CMD_GET_DAC_PWR_NORMAL_HSYNC_LO     (0x00000001)
#define LW5070_CTRL_CMD_GET_DAC_PWR_NORMAL_HSYNC_HI     (0x00000002)

#define LW5070_CTRL_CMD_GET_DAC_PWR_NORMAL_VSYNC                            1:0
#define LW5070_CTRL_CMD_GET_DAC_PWR_NORMAL_VSYNC_ENABLE (0x00000000)
#define LW5070_CTRL_CMD_GET_DAC_PWR_NORMAL_VSYNC_LO     (0x00000001)
#define LW5070_CTRL_CMD_GET_DAC_PWR_NORMAL_VSYNC_HI     (0x00000002)

#define LW5070_CTRL_CMD_GET_DAC_PWR_NORMAL_DATA                             1:0
#define LW5070_CTRL_CMD_GET_DAC_PWR_NORMAL_DATA_ENABLE  (0x00000000)
#define LW5070_CTRL_CMD_GET_DAC_PWR_NORMAL_DATA_LO      (0x00000001)
#define LW5070_CTRL_CMD_GET_DAC_PWR_NORMAL_DATA_HI      (0x00000002)

#define LW5070_CTRL_CMD_GET_DAC_PWR_NORMAL_PWR                              0:0
#define LW5070_CTRL_CMD_GET_DAC_PWR_NORMAL_PWR_OFF      (0x00000000)
#define LW5070_CTRL_CMD_GET_DAC_PWR_NORMAL_PWR_ON       (0x00000001)

#define LW5070_CTRL_CMD_GET_DAC_PWR_SAFE_HSYNC                              1:0
#define LW5070_CTRL_CMD_GET_DAC_PWR_SAFE_HSYNC_ENABLE   (0x00000000)
#define LW5070_CTRL_CMD_GET_DAC_PWR_SAFE_HSYNC_LO       (0x00000001)
#define LW5070_CTRL_CMD_GET_DAC_PWR_SAFE_HSYNC_HI       (0x00000002)

#define LW5070_CTRL_CMD_GET_DAC_PWR_SAFE_VSYNC                              1:0
#define LW5070_CTRL_CMD_GET_DAC_PWR_SAFE_VSYNC_ENABLE   (0x00000000)
#define LW5070_CTRL_CMD_GET_DAC_PWR_SAFE_VSYNC_LO       (0x00000001)
#define LW5070_CTRL_CMD_GET_DAC_PWR_SAFE_VSYNC_HI       (0x00000002)

#define LW5070_CTRL_CMD_GET_DAC_PWR_SAFE_DATA                               1:0
#define LW5070_CTRL_CMD_GET_DAC_PWR_SAFE_DATA_ENABLE    (0x00000000)
#define LW5070_CTRL_CMD_GET_DAC_PWR_SAFE_DATA_LO        (0x00000001)
#define LW5070_CTRL_CMD_GET_DAC_PWR_SAFE_DATA_HI        (0x00000002)

#define LW5070_CTRL_CMD_GET_DAC_PWR_SAFE_PWR                                0:0
#define LW5070_CTRL_CMD_GET_DAC_PWR_SAFE_PWR_OFF        (0x00000000)
#define LW5070_CTRL_CMD_GET_DAC_PWR_SAFE_PWR_ON         (0x00000001)
#define LW5070_CTRL_CMD_GET_DAC_PWR_PARAMS_MESSAGE_ID (0x3U)

typedef struct LW5070_CTRL_CMD_GET_DAC_PWR_PARAMS {
    LW5070_CTRL_CMD_BASE_PARAMS base;
    LwU32                       orNumber;

    LwU32                       normalHSync;
    LwU32                       normalVSync;
    LwU32                       normalData;
    LwU32                       normalPower;
    LwU32                       safeHSync;
    LwU32                       safeVSync;
    LwU32                       safeData;
    LwU32                       safePower;
} LW5070_CTRL_CMD_GET_DAC_PWR_PARAMS;

/* LWRM_UNPUBLISHED */
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)



/*
 * LW5070_CTRL_CMD_SET_DAC_PWR
 *
 * This command sets the DAC power control register. orNumber, normalPower,
 * and safePower will always have to be specified. However, HSync, VSync,
 * and data for normal and/or safe mode can be empty, leaving the current
 * values intact.
 *
 *      orNumber
 *          The dac for which the settings need to be programmed.
 *
 *      normalHSync
 *          The normal operating state for the H sync signal.
 *
 *      normalVSync
 *          The normal operating state for the V sync signal.
 *
 *      normalData
 *          The normal video data input pin of the d/a colwerter.
 *
 *      normalPower
 *          The normal state of the dac macro power.
 *
 *      safeHSync
 *          The safe operating state for the H sync signal.
 *
 *      safeVSync
 *          The safe operating state for the V sync signal.
 *
 *      safeData
 *          The safe video data input pin of the d/a colwerter.
 *
 *      safePower
 *          The safe state of the dac macro power.
 *
 *      flags
 *          The following flags have been defined:
 *              (1) SPECIFIED_NORMAL: Indicates whether HSync, VSync, data,
 *                  for normal state have been specified in the parameters.
 *              (2) SPECIFIED_SAFE: Indicates whether HSync, VSync, data,
 *                  for safe state have been specified in the parameters.
 *              (3) SPECIFIED_FORCE_SWITCH: Indicates whether to force the
 *                  change immediately instead of waiting for VSync
 *
 * Possible status values returned are:
 *      LW_OK
 *      LW_ERR_ILWALID_ARGUMENT
 *      LW_ERR_GENERIC
 *      LW_ERR_TIMEOUT
 */
#define LW5070_CTRL_CMD_SET_DAC_PWR                            (0x50700404) /* finn: Evaluated from "(FINN_LW50_DISPLAY_OR_INTERFACE_ID << 8) | LW5070_CTRL_CMD_SET_DAC_PWR_PARAMS_MESSAGE_ID" */

#define LW5070_CTRL_CMD_SET_DAC_PWR_NORMAL_HSYNC                            1:0
#define LW5070_CTRL_CMD_SET_DAC_PWR_NORMAL_HSYNC_ENABLE        (0x00000000)
#define LW5070_CTRL_CMD_SET_DAC_PWR_NORMAL_HSYNC_LO            (0x00000001)
#define LW5070_CTRL_CMD_SET_DAC_PWR_NORMAL_HSYNC_HI            (0x00000002)

#define LW5070_CTRL_CMD_SET_DAC_PWR_NORMAL_VSYNC                            1:0
#define LW5070_CTRL_CMD_SET_DAC_PWR_NORMAL_VSYNC_ENABLE        (0x00000000)
#define LW5070_CTRL_CMD_SET_DAC_PWR_NORMAL_VSYNC_LO            (0x00000001)
#define LW5070_CTRL_CMD_SET_DAC_PWR_NORMAL_VSYNC_HI            (0x00000002)

#define LW5070_CTRL_CMD_SET_DAC_PWR_NORMAL_DATA                             1:0
#define LW5070_CTRL_CMD_SET_DAC_PWR_NORMAL_DATA_ENABLE         (0x00000000)
#define LW5070_CTRL_CMD_SET_DAC_PWR_NORMAL_DATA_LO             (0x00000001)
#define LW5070_CTRL_CMD_SET_DAC_PWR_NORMAL_DATA_HI             (0x00000002)

#define LW5070_CTRL_CMD_SET_DAC_PWR_NORMAL_PWR                              0:0
#define LW5070_CTRL_CMD_SET_DAC_PWR_NORMAL_PWR_OFF             (0x00000000)
#define LW5070_CTRL_CMD_SET_DAC_PWR_NORMAL_PWR_ON              (0x00000001)

#define LW5070_CTRL_CMD_SET_DAC_PWR_SAFE_HSYNC                              1:0
#define LW5070_CTRL_CMD_SET_DAC_PWR_SAFE_HSYNC_ENABLE          (0x00000000)
#define LW5070_CTRL_CMD_SET_DAC_PWR_SAFE_HSYNC_LO              (0x00000001)
#define LW5070_CTRL_CMD_SET_DAC_PWR_SAFE_HSYNC_HI              (0x00000002)

#define LW5070_CTRL_CMD_SET_DAC_PWR_SAFE_VSYNC                              1:0
#define LW5070_CTRL_CMD_SET_DAC_PWR_SAFE_VSYNC_ENABLE          (0x00000000)
#define LW5070_CTRL_CMD_SET_DAC_PWR_SAFE_VSYNC_LO              (0x00000001)
#define LW5070_CTRL_CMD_SET_DAC_PWR_SAFE_VSYNC_HI              (0x00000002)

#define LW5070_CTRL_CMD_SET_DAC_PWR_SAFE_DATA                               1:0
#define LW5070_CTRL_CMD_SET_DAC_PWR_SAFE_DATA_ENABLE           (0x00000000)
#define LW5070_CTRL_CMD_SET_DAC_PWR_SAFE_DATA_LO               (0x00000001)
#define LW5070_CTRL_CMD_SET_DAC_PWR_SAFE_DATA_HI               (0x00000002)

#define LW5070_CTRL_CMD_SET_DAC_PWR_SAFE_PWR                                0:0
#define LW5070_CTRL_CMD_SET_DAC_PWR_SAFE_PWR_OFF               (0x00000000)
#define LW5070_CTRL_CMD_SET_DAC_PWR_SAFE_PWR_ON                (0x00000001)

#define LW5070_CTRL_CMD_SET_DAC_PWR_FLAGS_SPECIFIED_NORMAL                  0:0
#define LW5070_CTRL_CMD_SET_DAC_PWR_FLAGS_SPECIFIED_NORMAL_NO  (0x00000000)
#define LW5070_CTRL_CMD_SET_DAC_PWR_FLAGS_SPECIFIED_NORMAL_YES (0x00000001)
#define LW5070_CTRL_CMD_SET_DAC_PWR_FLAGS_SPECIFIED_SAFE                    1:1
#define LW5070_CTRL_CMD_SET_DAC_PWR_FLAGS_SPECIFIED_SAFE_NO    (0x00000000)
#define LW5070_CTRL_CMD_SET_DAC_PWR_FLAGS_SPECIFIED_SAFE_YES   (0x00000001)
#define LW5070_CTRL_CMD_SET_DAC_PWR_FLAGS_FORCE_SWITCH                      2:2
#define LW5070_CTRL_CMD_SET_DAC_PWR_FLAGS_FORCE_SWITCH_NO      (0x00000000)
#define LW5070_CTRL_CMD_SET_DAC_PWR_FLAGS_FORCE_SWITCH_YES     (0x00000001)
#define LW5070_CTRL_CMD_SET_DAC_PWR_PARAMS_MESSAGE_ID (0x4U)

typedef struct LW5070_CTRL_CMD_SET_DAC_PWR_PARAMS {
    LW5070_CTRL_CMD_BASE_PARAMS base;
    LwU32                       orNumber;

    LwU32                       normalHSync;
    LwU32                       normalVSync;
    LwU32                       normalData;
    LwU32                       normalPower;
    LwU32                       safeHSync;
    LwU32                       safeVSync;
    LwU32                       safeData;
    LwU32                       safePower;
    LwU32                       flags;
} LW5070_CTRL_CMD_SET_DAC_PWR_PARAMS;

#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

/*
 * LW5070_CTRL_DAC_LOAD_PARAMS
 *
 * This structure contains DAC load control contents.
 *
 *      valDC
 *          The value used for DC load sensing.
 *
 *      valAC
 *          The value used for AC load sensing.
 *
 *      perDC
 *          The amount of VAL_DC to be driven out before VAL_AC is driven out.
 *
 *      perSample
 *          The amount of VAL_AC to be driven out.
 *
 *
 * Possible status values returned are:
 *      LW_OK
 *      LW_ERR_ILWALID_ARGUMENT
 *      LW_ERR_GENERIC
 */
typedef struct LW5070_CTRL_DAC_LOAD_PARAMS {
    LwU32 valDC;
    LwU32 valAC;
    LwU32 perDC;
    LwU32 perSample;
} LW5070_CTRL_DAC_LOAD_PARAMS;
typedef LW5070_CTRL_DAC_LOAD_PARAMS DACLOADSETUP;
typedef struct LW5070_CTRL_DAC_LOAD_PARAMS *PDACLOADSETUP;

#define LW5070_CTRL_DAC_LOAD_VAL_DC                                 9:0
#define LW5070_CTRL_DAC_LOAD_VAL_DC_INIT     (0x00000000)
#define LW5070_CTRL_DAC_LOAD_VAL_AC                                 9:0
#define LW5070_CTRL_DAC_LOAD_VAL_AC_INIT     (0x00000000)
#define LW5070_CTRL_DAC_LOAD_PER_DC                                15:0
#define LW5070_CTRL_DAC_LOAD_PER_DC_INIT     (0x00000000)
#define LW5070_CTRL_DAC_LOAD_PER_SAMPLE                            15:0
#define LW5070_CTRL_DAC_LOAD_PER_SAMPLE_INIT (0x00000000)

/*
 * LW5070_CTRL_CMD_GET_DAC_LOAD
 *
 * This command gets the DAC load control parameters.
 *
 *      orNumber
 *          The dac for which the settings need to be read.
 *
 *      mode
 *          The mode to use for load sensing.
 *
 *      crt
 *          The arguments of this structure and format are returned
 *          with CRT DAC loading info.  The format of this structure
 *          are describe in LW5070_CTRL_DAC_LOAD_PARAMS.
 *
 *      tv
 *          The arguments of this structure and format are returned
 *          with TV DAC loading values.  The format of this structure
 *          are describe in LW5070_CTRL_DAC_LOAD_PARAMS.
 *
 *      perAuto
 *          The repetition rate in auto mode.
 *
 *      load
 *          0 to 3 are are comparator outputs.
 *
 *      status
 *          Whether the comparator outputs are valid or not.
 *
 * Possible status values returned are:
 *      LW_OK
 *      LW_ERR_ILWALID_ARGUMENT
 *      LW_ERR_GENERIC
 *      LW_ERR_TIMEOUT
 */
#define LW5070_CTRL_CMD_GET_DAC_LOAD         (0x50700405) /* finn: Evaluated from "(FINN_LW50_DISPLAY_OR_INTERFACE_ID << 8) | LW5070_CTRL_CMD_GET_DAC_LOAD_PARAMS_MESSAGE_ID" */

#define LW5070_CTRL_CMD_GET_DAC_LOAD_PARAMS_MESSAGE_ID (0x5U)

typedef struct LW5070_CTRL_CMD_GET_DAC_LOAD_PARAMS {
    LW5070_CTRL_CMD_BASE_PARAMS base;
    LwU32                       orNumber;
    LwU32                       mode;
    LW5070_CTRL_DAC_LOAD_PARAMS crt;
    LW5070_CTRL_DAC_LOAD_PARAMS tv;
    LwU32                       perAuto;
    LwU32                       load;
    LwU32                       status;
} LW5070_CTRL_CMD_GET_DAC_LOAD_PARAMS;

#define LW5070_CTRL_CMD_GET_DAC_LOAD_MODE                                   2:0
#define LW5070_CTRL_CMD_GET_DAC_LOAD_MODE_INIT       (0x00000000)
#define LW5070_CTRL_CMD_GET_DAC_LOAD_MODE_OFF        (0x00000000)
#define LW5070_CTRL_CMD_GET_DAC_LOAD_MODE_CONT_DC    (0x00000001)
#define LW5070_CTRL_CMD_GET_DAC_LOAD_MODE_CONT_AC    (0x00000002)
#define LW5070_CTRL_CMD_GET_DAC_LOAD_MODE_AUTO_DC    (0x00000003)
#define LW5070_CTRL_CMD_GET_DAC_LOAD_MODE_AUTO_AC    (0x00000004)

#define LW5070_CTRL_CMD_GET_DAC_LOAD_PER_AUTO                               0:0
#define LW5070_CTRL_CMD_GET_DAC_LOAD_PER_AUTO_INIT   (0x00000000)
#define LW5070_CTRL_CMD_GET_DAC_LOAD_PER_AUTO_100MS  (0x00000000)
#define LW5070_CTRL_CMD_GET_DAC_LOAD_PER_AUTO_1000MS (0x00000001)

#define LW5070_CTRL_CMD_GET_DAC_LOAD_0                                      0:0
#define LW5070_CTRL_CMD_GET_DAC_LOAD_0_NO_LOAD       (0x00000000)
#define LW5070_CTRL_CMD_GET_DAC_LOAD_0_LOAD          (0x00000001)
#define LW5070_CTRL_CMD_GET_DAC_LOAD_1                                      1:1
#define LW5070_CTRL_CMD_GET_DAC_LOAD_1_NO_LOAD       (0x00000000)
#define LW5070_CTRL_CMD_GET_DAC_LOAD_1_LOAD          (0x00000001)
#define LW5070_CTRL_CMD_GET_DAC_LOAD_2                                      2:2
#define LW5070_CTRL_CMD_GET_DAC_LOAD_2_NO_LOAD       (0x00000000)
#define LW5070_CTRL_CMD_GET_DAC_LOAD_2_LOAD          (0x00000001)
#define LW5070_CTRL_CMD_GET_DAC_LOAD_3                                      3:3
#define LW5070_CTRL_CMD_GET_DAC_LOAD_3_NO_LOAD       (0x00000000)
#define LW5070_CTRL_CMD_GET_DAC_LOAD_3_LOAD          (0x00000001)

#define LW5070_CTRL_CMD_GET_DAC_LOAD_STATUS                                 0:0
#define LW5070_CTRL_CMD_GET_DAC_LOAD_STATUS_ILWALID  (0x00000000)
#define LW5070_CTRL_CMD_GET_DAC_LOAD_STATUS_VALID    (0x00000001)


/*
 * LW5070_CTRL_CMD_SET_DAC_LOAD
 *
 * This command sets the DAC load control parameters.
 *
 *      orNumber
 *          The dac for which the settings need to be programmed.
 *
 *      mode
 *          The mode to use for load sensing.
 *
 *      crt
 *          The arguments of this structure and format should be filled
 *          in with CRT DAC loading values.  The format of this structure
 *          are describe in LW5070_CTRL_DAC_LOAD_PARAMS.
 *
 *      tv
 *          The arguments of this structure and format should be filled
 *          in with TV DAC loading values.  The format of this structure
 *          are describe in LW5070_CTRL_DAC_LOAD_PARAMS.
 *
 *      perAuto
 *          The repetition rate in auto mode.
 *
 *      flags
 *          The following flags have been defined:
 *              (1) SKIP_WAIT_FOR_VALID: Indicates whether to skip wait for
 *                  status to become valid after writing non-OFF mode.
 *              (2) VALID_PARAMS: Indicates whether the users are intended to
 *                  set the crt type or the tv type in this structure.
 *
 * Possible status values returned are:
 *      LW_OK
 *      LW_ERR_ILWALID_ARGUMENT
 *      LW_ERR_GENERIC
 *      LW_ERR_TIMEOUT
 */
#define LW5070_CTRL_CMD_SET_DAC_LOAD                 (0x50700406) /* finn: Evaluated from "(FINN_LW50_DISPLAY_OR_INTERFACE_ID << 8) | LW5070_CTRL_CMD_SET_DAC_LOAD_PARAMS_MESSAGE_ID" */

#define LW5070_CTRL_CMD_SET_DAC_LOAD_PARAMS_MESSAGE_ID (0x6U)

typedef struct LW5070_CTRL_CMD_SET_DAC_LOAD_PARAMS {
    LW5070_CTRL_CMD_BASE_PARAMS base;
    LwU32                       orNumber;
    LwU32                       mode;
    LW5070_CTRL_DAC_LOAD_PARAMS crt;
    LW5070_CTRL_DAC_LOAD_PARAMS tv;
    LwU32                       perAuto;
    LwU32                       flags;
} LW5070_CTRL_CMD_SET_DAC_LOAD_PARAMS;

#define LW5070_CTRL_CMD_SET_DAC_LOAD_MODE                                   2:0
#define LW5070_CTRL_CMD_SET_DAC_LOAD_MODE_INIT                     (0x00000000)
#define LW5070_CTRL_CMD_SET_DAC_LOAD_MODE_OFF                      (0x00000000)
#define LW5070_CTRL_CMD_SET_DAC_LOAD_MODE_CONT_DC                  (0x00000001)
#define LW5070_CTRL_CMD_SET_DAC_LOAD_MODE_CONT_AC                  (0x00000002)
#define LW5070_CTRL_CMD_SET_DAC_LOAD_MODE_AUTO_DC                  (0x00000003)
#define LW5070_CTRL_CMD_SET_DAC_LOAD_MODE_AUTO_AC                  (0x00000004)

#define LW5070_CTRL_CMD_SET_DAC_LOAD_PER_AUTO                               0:0
#define LW5070_CTRL_CMD_SET_DAC_LOAD_PER_AUTO_INIT                 (0x00000000)
#define LW5070_CTRL_CMD_SET_DAC_LOAD_PER_AUTO_100MS                (0x00000000)
#define LW5070_CTRL_CMD_SET_DAC_LOAD_PER_AUTO_1000MS               (0x00000001)

#define LW5070_CTRL_CMD_SET_DAC_LOAD_FLAGS_SKIP_WAIT_FOR_VALID              0:0
#define LW5070_CTRL_CMD_SET_DAC_LOAD_FLAGS_SKIP_WAIT_FOR_VALID_NO  (0x00000000)
#define LW5070_CTRL_CMD_SET_DAC_LOAD_FLAGS_SKIP_WAIT_FOR_VALID_YES (0x00000001)

#define LW5070_CTRL_CMD_SET_DAC_LOAD_FLAGS_VALID_PARAMS                     2:1
#define LW5070_CTRL_CMD_SET_DAC_LOAD_FLAGS_VALID_PARAMS_INIT       (0x00000000)
#define LW5070_CTRL_CMD_SET_DAC_LOAD_FLAGS_VALID_PARAMS_CRT        (0x00000001)
#define LW5070_CTRL_CMD_SET_DAC_LOAD_FLAGS_VALID_PARAMS_TV         (0x00000002)

/* LWRM_UNPUBLISHED */
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)



/*
 * LW5070_CTRL_CMD_GET_SOR_PWM
 *
 * This command returns SOR's current PWM settings.
 *
 *      orNumber
 *          The OR number for which the seq ctrls are to be modified.
 *
 *      targetFreq
 *          The target PWM freq. This is the PWM frequency we planned on
 *          programming.
 *
 *      actualFreq
 *          Actual PWM freq programmed into PWM.
 *
 *      div
 *          The divider being used lwrrently for generating PWM clk.
 *          A valued of 0 means that PWM is disabled.
 *
 *      resolution
 *          The resolution of steps lwrrently programmed or the max number of
 *          clocks per cycle. The possible values for LW50 are 128, 256, 512
 *          and 1024. This field is irrelevant when div is 0.
 *
 *      dutyCycle
 *          Duty cycle in range 0-1024
 *
 *      sourcePCLK (OUT)
 *          The PWM source clock selector. This field is non-zero if the PCLK
 *          is selected as the PWM source clock. Otherwise, the PWM source
 *          clock is XTAL.
 *
 *      head (IN)
 *          The head for which the pixel clock is sourced from.
 *
 * Possible status values returned are:
 *      LW_OK
 *      LW_ERR_ILWALID_ARGUMENT
 *      LW_ERR_GENERIC
 */
#define LW5070_CTRL_CMD_GET_SOR_PWM                                (0x50700420) /* finn: Evaluated from "(FINN_LW50_DISPLAY_OR_INTERFACE_ID << 8) | LW5070_CTRL_CMD_GET_SOR_PWM_PARAMS_MESSAGE_ID" */

#define LW5070_CTRL_CMD_GET_SOR_PWM_PARAMS_MESSAGE_ID (0x20U)

typedef struct LW5070_CTRL_CMD_GET_SOR_PWM_PARAMS {
    LW5070_CTRL_CMD_BASE_PARAMS base;
    LwU32                       orNumber;
    LwU32                       targetFreq;
    LwU32                       actualFreq;
    LwU32                       div;
    LwU32                       resolution;
    LwU32                       dutyCycle;
    LwU32                       sourcePCLK;
    LwU32                       head;
} LW5070_CTRL_CMD_GET_SOR_PWM_PARAMS;


/*
 * LW5070_CTRL_CMD_SET_SOR_PWM
 *
 * This command returns SOR's current PWM settings.
 *
 *      orNumber
 *          The OR number for which the seq ctrls are to be modified.
 *
 *      targetFreq
 *          The target PWM freq to be programmed.
 *
 *      actualFreq
 *          Actual PWM freq programmed into PWM after all the specified
 *          settings have been applied.
 *
 *      div
 *          The divider to use for generating PWM clk.
 *          Set this to 0 to disable PWM. Note that only one of div
 *          or targetFreq can be specified at a time since specifying one
 *          automatically determines the value of the other. Selection is
 *          done via USE_SPECIFIED_DIV flag.
 *
 *      resolution
 *          The resolution or the max number of clocks per cycle desired.
 *          Note that if it's not possible to program the given resolution
 *          and frequency (or div) combination, RM would not attempt to
 *          smartly lower the resolution. The call would return failure.
 *          The possible values for LW50 are 128, 256, 512 and 1024. This
 *          field is irrelevant when div is 0.
 *
 *      dutyCycle
 *          Duty cycle in range 0-1024
 *
 *      flags
 *          The following flags have been defined:
 *              (1) USE_SPECIFIED_DIV: Indicates whether RM should use
 *                  specified div or targetFreq when determining the divider
 *                  for xtal clock.
 *              (2) PROG_DUTY_CYCLE: Indicates whether or not the caller
 *                  desires to program duty cycle. Normally whenever pwm freq
 *                  and range need to be programmed, it's expected that duty
 *                  cycle would be reprogrammed as well but this is not
 *                  enforced.
 *              (3) PROG_FREQ_AND_RANGE: Indicates whether or not the caller
 *                  desires to program a new PWM setting (div and resolution).
 *              (4) SOURCE_CLOCK: Indicates whether the PCLK or XTAL is used
 *                  as the PWM clock source. GT21x and better.
 *
 *      head (IN)
 *          The head for which the pixel clock is sourced from.
 *
 * Possible status values returned are:
 *      LW_OK
 *      LW_ERR_ILWALID_ARGUMENT
 *      LW_ERR_GENERIC
 */
#define LW5070_CTRL_CMD_SET_SOR_PWM                               (0x50700421) /* finn: Evaluated from "(FINN_LW50_DISPLAY_OR_INTERFACE_ID << 8) | LW5070_CTRL_CMD_SET_SOR_PWM_PARAMS_MESSAGE_ID" */

#define LW5070_CTRL_CMD_SET_SOR_PWM_FLAGS_USE_SPECIFIED_DIV                 0:0
#define LW5070_CTRL_CMD_SET_SOR_PWM_FLAGS_USE_SPECIFIED_DIV_NO    (0x00000000)
#define LW5070_CTRL_CMD_SET_SOR_PWM_FLAGS_USE_SPECIFIED_DIV_YES   (0x00000001)
#define LW5070_CTRL_CMD_SET_SOR_PWM_FLAGS_PROG_DUTY_CYCLE                   1:1
#define LW5070_CTRL_CMD_SET_SOR_PWM_FLAGS_PROG_DUTY_CYCLE_NO      (0x00000000)
#define LW5070_CTRL_CMD_SET_SOR_PWM_FLAGS_PROG_DUTY_CYCLE_YES     (0x00000001)
#define LW5070_CTRL_CMD_SET_SOR_PWM_FLAGS_PROG_FREQ_AND_RANGE               2:2
#define LW5070_CTRL_CMD_SET_SOR_PWM_FLAGS_PROG_FREQ_AND_RANGE_NO  (0x00000000)
#define LW5070_CTRL_CMD_SET_SOR_PWM_FLAGS_PROG_FREQ_AND_RANGE_YES (0x00000001)
#define LW5070_CTRL_CMD_SET_SOR_PWM_FLAGS_SOURCE_CLOCK                      3:3
#define LW5070_CTRL_CMD_SET_SOR_PWM_FLAGS_SOURCE_CLOCK_XTAL       (0x00000000)
#define LW5070_CTRL_CMD_SET_SOR_PWM_FLAGS_SOURCE_CLOCK_PCLK       (0x00000001)

#define LW5070_CTRL_CMD_SET_SOR_PWM_PARAMS_MESSAGE_ID (0x21U)

typedef struct LW5070_CTRL_CMD_SET_SOR_PWM_PARAMS {
    LW5070_CTRL_CMD_BASE_PARAMS base;
    LwU32                       orNumber;
    LwU32                       targetFreq;
    LwU32                       actualFreq;
    LwU32                       div;                  // equivalent of LW_PDISP_SOR_PWM_DIV_DIVIDE
    LwU32                       resolution;           // equivalent of LW_PDISP_SOR_PWM_DIV_RANGE
    LwU32                       dutyCycle;
    LwU32                       flags;
    LwU32                       head;
} LW5070_CTRL_CMD_SET_SOR_PWM_PARAMS;


/*
 * LW5070_CTRL_CMD_GET_SOR_OP_MODE
 *
 * This command returns current settings for the specified SOR.
 *
 *      orNumber
 *          The OR number for which the operating mode needs to be read.
 *
 *      category
 *          Whether LVDS or CSTM setting are desired.
 *
 *      puTxda
 *          Status of data pins of link A
 *
 *      puTxdb
 *          Status of data pins of link B
 *
 *      puTxca
 *          Status of link A clock
 *
 *      puTxcb
 *          Status of link B clock
 *
 *      upper
 *          Whether LVDS bank A is the upper, odd, or first pixel.
 *
 *      mode
 *          Current protocol.
 *
 *      linkActA
 *          Status of link B clock
 *
 *      linkActB
 *          Status of link B clock
 *
 *      lvdsEn
 *          Output driver configuration.
 *
 *      lvdsDual
 *          Whether LVDS dual-link mode is turned on or not.
 *
 *      dupSync
 *          Whether DE, HSYNC, and VSYNC are used for encoding instead of
 *          RES, CNTLE, and CNTLF.
 *
 *      newMode
 *          Whether new or old mode is being used.
 *
 *      balanced
 *          Whether balanced encoding is enabled.
 *
 *      plldiv
 *          Feedback divider for the hi-speed pll
 *
 *      rotClk
 *          Skew of TXC clock.
 *
 *      rotDat
 *          How much are the 8 bits of each color channel rotated by
 *
 * Possible status values returned are:
 *      LW_OK
 *      LW_ERR_ILWALID_ARGUMENT
 *      LW_ERR_GENERIC
 */
#define LW5070_CTRL_CMD_GET_SOR_OP_MODE                   (0x50700422) /* finn: Evaluated from "(FINN_LW50_DISPLAY_OR_INTERFACE_ID << 8) | LW5070_CTRL_CMD_GET_SOR_OP_MODE_PARAMS_MESSAGE_ID" */

#define LW5070_CTRL_CMD_GET_SOR_OP_MODE_CATEGORY                            0:0
#define LW5070_CTRL_CMD_GET_SOR_OP_MODE_CATEGORY_LVDS     0x00000000
#define LW5070_CTRL_CMD_GET_SOR_OP_MODE_CATEGORY_LWSTOM   0x00000001

#define LW5070_CTRL_CMD_GET_SOR_OP_MODE_PU_TXDA_0                           0:0
#define LW5070_CTRL_CMD_GET_SOR_OP_MODE_PU_TXDA_0_DISABLE 0x00000000
#define LW5070_CTRL_CMD_GET_SOR_OP_MODE_PU_TXDA_0_ENABLE  0x00000001
#define LW5070_CTRL_CMD_GET_SOR_OP_MODE_PU_TXDA_1                           1:1
#define LW5070_CTRL_CMD_GET_SOR_OP_MODE_PU_TXDA_1_DISABLE 0x00000000
#define LW5070_CTRL_CMD_GET_SOR_OP_MODE_PU_TXDA_1_ENABLE  0x00000001
#define LW5070_CTRL_CMD_GET_SOR_OP_MODE_PU_TXDA_2                           2:2
#define LW5070_CTRL_CMD_GET_SOR_OP_MODE_PU_TXDA_2_DISABLE 0x00000000
#define LW5070_CTRL_CMD_GET_SOR_OP_MODE_PU_TXDA_2_ENABLE  0x00000001
#define LW5070_CTRL_CMD_GET_SOR_OP_MODE_PU_TXDA_3                           3:3
#define LW5070_CTRL_CMD_GET_SOR_OP_MODE_PU_TXDA_3_DISABLE 0x00000000
#define LW5070_CTRL_CMD_GET_SOR_OP_MODE_PU_TXDA_3_ENABLE  0x00000001

#define LW5070_CTRL_CMD_GET_SOR_OP_MODE_PU_TXDB_0                           0:0
#define LW5070_CTRL_CMD_GET_SOR_OP_MODE_PU_TXDB_0_DISABLE 0x00000000
#define LW5070_CTRL_CMD_GET_SOR_OP_MODE_PU_TXDB_0_ENABLE  0x00000001
#define LW5070_CTRL_CMD_GET_SOR_OP_MODE_PU_TXDB_1                           1:1
#define LW5070_CTRL_CMD_GET_SOR_OP_MODE_PU_TXDB_1_DISABLE 0x00000000
#define LW5070_CTRL_CMD_GET_SOR_OP_MODE_PU_TXDB_1_ENABLE  0x00000001
#define LW5070_CTRL_CMD_GET_SOR_OP_MODE_PU_TXDB_2                           2:2
#define LW5070_CTRL_CMD_GET_SOR_OP_MODE_PU_TXDB_2_DISABLE 0x00000000
#define LW5070_CTRL_CMD_GET_SOR_OP_MODE_PU_TXDB_2_ENABLE  0x00000001
#define LW5070_CTRL_CMD_GET_SOR_OP_MODE_PU_TXDB_3                           3:3
#define LW5070_CTRL_CMD_GET_SOR_OP_MODE_PU_TXDB_3_DISABLE 0x00000000
#define LW5070_CTRL_CMD_GET_SOR_OP_MODE_PU_TXDB_3_ENABLE  0x00000001

#define LW5070_CTRL_CMD_GET_SOR_OP_MODE_PU_TXCA                             0:0
#define LW5070_CTRL_CMD_GET_SOR_OP_MODE_PU_TXCA_DISABLE   0x00000000
#define LW5070_CTRL_CMD_GET_SOR_OP_MODE_PU_TXCA_ENABLE    0x00000001

#define LW5070_CTRL_CMD_GET_SOR_OP_MODE_PU_TXCB                             0:0
#define LW5070_CTRL_CMD_GET_SOR_OP_MODE_PU_TXCB_DISABLE   0x00000000
#define LW5070_CTRL_CMD_GET_SOR_OP_MODE_PU_TXCB_ENABLE    0x00000001

#define LW5070_CTRL_CMD_GET_SOR_OP_MODE_UPPER                               0:0
#define LW5070_CTRL_CMD_GET_SOR_OP_MODE_UPPER_UPPER_RESET 0x00000001

#define LW5070_CTRL_CMD_GET_SOR_OP_MODE_MODE                                0:0
#define LW5070_CTRL_CMD_GET_SOR_OP_MODE_MODE_LVDS         0x00000000
#define LW5070_CTRL_CMD_GET_SOR_OP_MODE_MODE_TMDS         0x00000001

#define LW5070_CTRL_CMD_GET_SOR_OP_MODE_LINKACTA                            0:0
#define LW5070_CTRL_CMD_GET_SOR_OP_MODE_LINKACTA_DISABLE  0x00000000
#define LW5070_CTRL_CMD_GET_SOR_OP_MODE_LINKACTA_ENABLE   0x00000001

#define LW5070_CTRL_CMD_GET_SOR_OP_MODE_LINKACTB                            0:0
#define LW5070_CTRL_CMD_GET_SOR_OP_MODE_LINKACTB_DISABLE  0x00000000
#define LW5070_CTRL_CMD_GET_SOR_OP_MODE_LINKACTB_ENABLE   0x00000001

#define LW5070_CTRL_CMD_GET_SOR_OP_MODE_LVDS_EN                             0:0
#define LW5070_CTRL_CMD_GET_SOR_OP_MODE_LVDS_EN_DISABLE   0x00000000
#define LW5070_CTRL_CMD_GET_SOR_OP_MODE_LVDS_EN_ENABLE    0x00000001

#define LW5070_CTRL_CMD_GET_SOR_OP_MODE_LVDS_DUAL                           0:0
#define LW5070_CTRL_CMD_GET_SOR_OP_MODE_LVDS_DUAL_DISABLE 0x00000000
#define LW5070_CTRL_CMD_GET_SOR_OP_MODE_LVDS_DUAL_ENABLE  0x00000001

#define LW5070_CTRL_CMD_GET_SOR_OP_MODE_DUP_SYNC                            0:0
#define LW5070_CTRL_CMD_GET_SOR_OP_MODE_DUP_SYNC_DISABLE  0x00000000
#define LW5070_CTRL_CMD_GET_SOR_OP_MODE_DUP_SYNC_ENABLE   0x00000001

#define LW5070_CTRL_CMD_GET_SOR_OP_MODE_NEW_MODE                            0:0
#define LW5070_CTRL_CMD_GET_SOR_OP_MODE_NEW_MODE_DISABLE  0x00000000
#define LW5070_CTRL_CMD_GET_SOR_OP_MODE_NEW_MODE_ENABLE   0x00000001

#define LW5070_CTRL_CMD_GET_SOR_OP_MODE_BALANCED                            0:0
#define LW5070_CTRL_CMD_GET_SOR_OP_MODE_BALANCED_DISABLE  0x00000000
#define LW5070_CTRL_CMD_GET_SOR_OP_MODE_BALANCED_ENABLE   0x00000001

#define LW5070_CTRL_CMD_GET_SOR_OP_MODE_PLLDIV                              0:0
#define LW5070_CTRL_CMD_GET_SOR_OP_MODE_PLLDIV_BY_7       0x00000000
#define LW5070_CTRL_CMD_GET_SOR_OP_MODE_PLLDIV_BY_10      0x00000001

#define LW5070_CTRL_CMD_GET_SOR_OP_MODE_ROTCLK                              3:0

#define LW5070_CTRL_CMD_GET_SOR_OP_MODE_ROTDAT                              2:0

#define LW5070_CTRL_CMD_GET_SOR_OP_MODE_PARAMS_MESSAGE_ID (0x22U)

typedef struct LW5070_CTRL_CMD_GET_SOR_OP_MODE_PARAMS {
    LW5070_CTRL_CMD_BASE_PARAMS base;
    LwU32                       orNumber;

    LwU32                       category;
    LwU32                       puTxda;
    LwU32                       puTxdb;
    LwU32                       puTxca;
    LwU32                       puTxcb;
    LwU32                       upper;
    LwU32                       mode;
    LwU32                       linkActA;
    LwU32                       linkActB;
    LwU32                       lvdsEn;
    LwU32                       lvdsDual;
    LwU32                       dupSync;
    LwU32                       newMode;
    LwU32                       balanced;
    LwU32                       plldiv;
    LwU32                       rotClk;
    LwU32                       rotDat;
} LW5070_CTRL_CMD_GET_SOR_OP_MODE_PARAMS;


/*
 * LW5070_CTRL_CMD_SET_SOR_OP_MODE
 *
 * This command applies the specified settings to the specified SOR.
 *
 *      orNumber
 *          The OR number for which the operating mode needs to be read.
 *          Note that if DCB doesn't report LVDS for the specified orNumber,
 *          the call will return failure.
 *
 *      category
 *          Whether LVDS or CSTM settings are specified.
 *
 *      puTxda
 *          Used to enable or disable the data pins of link A.
 *
 *      puTxdb
 *          Used to enable or disable the data pins of link B.
 *
 *      puTxca
 *          Used to enable or disable link A clock.
 *
 *      puTxcb
 *          Used to enable or disable link B clock.
 *
 *      upper
 *          Whether LVDS bank A should be the upper, odd, or first pixel.
 *
 *      mode
 *          What protocol (LVDS/TMDS to use).
 *
 *      linkActA
 *          Used to enable or disable the digital logic of link A.
 *
 *      linkActB
 *          Used to enable or disable the digital logic of link B.
 *
 *      lvdsEn
 *          Output driver configuration.
 *
 *      lvdsDual
 *          Whether to turn on LVDS dual-link mode.
 *
 *      dupSync
 *          Whether to use DE, HSYNC, and VSYNC for encoding instead of
 *          RES, CNTLE, and CNTLF.
 *
 *      newMode
 *          Whether to use new or old mode.
 *
 *      balanced
 *          Whether or not to use balanced encoding.
 *
 *      plldiv
 *          Feedback divider to use for the hi-speed pll.
 *
 *      rotClk
 *          How much to skew TXC clock.
 *
 *      rotDat
 *          How much to rotate the 8 bits of each color channel by.
 *
 * Possible status values returned are:
 *      LW_OK
 *      LW_ERR_ILWALID_ARGUMENT
 *      LW_ERR_GENERIC
 */
#define LW5070_CTRL_CMD_SET_SOR_OP_MODE                   (0x50700423) /* finn: Evaluated from "(FINN_LW50_DISPLAY_OR_INTERFACE_ID << 8) | LW5070_CTRL_CMD_SET_SOR_OP_MODE_PARAMS_MESSAGE_ID" */

#define LW5070_CTRL_CMD_SET_SOR_OP_MODE_CATEGORY                            0:0
#define LW5070_CTRL_CMD_SET_SOR_OP_MODE_CATEGORY_LVDS     0x00000000
#define LW5070_CTRL_CMD_SET_SOR_OP_MODE_CATEGORY_LWSTOM   0x00000001

#define LW5070_CTRL_CMD_SET_SOR_OP_MODE_PU_TXDA_0                           0:0
#define LW5070_CTRL_CMD_SET_SOR_OP_MODE_PU_TXDA_0_DISABLE 0x00000000
#define LW5070_CTRL_CMD_SET_SOR_OP_MODE_PU_TXDA_0_ENABLE  0x00000001
#define LW5070_CTRL_CMD_SET_SOR_OP_MODE_PU_TXDA_1                           1:1
#define LW5070_CTRL_CMD_SET_SOR_OP_MODE_PU_TXDA_1_DISABLE 0x00000000
#define LW5070_CTRL_CMD_SET_SOR_OP_MODE_PU_TXDA_1_ENABLE  0x00000001
#define LW5070_CTRL_CMD_SET_SOR_OP_MODE_PU_TXDA_2                           2:2
#define LW5070_CTRL_CMD_SET_SOR_OP_MODE_PU_TXDA_2_DISABLE 0x00000000
#define LW5070_CTRL_CMD_SET_SOR_OP_MODE_PU_TXDA_2_ENABLE  0x00000001
#define LW5070_CTRL_CMD_SET_SOR_OP_MODE_PU_TXDA_3                           3:3
#define LW5070_CTRL_CMD_SET_SOR_OP_MODE_PU_TXDA_3_DISABLE 0x00000000
#define LW5070_CTRL_CMD_SET_SOR_OP_MODE_PU_TXDA_3_ENABLE  0x00000001

#define LW5070_CTRL_CMD_SET_SOR_OP_MODE_PU_TXDB_0                           0:0
#define LW5070_CTRL_CMD_SET_SOR_OP_MODE_PU_TXDB_0_DISABLE 0x00000000
#define LW5070_CTRL_CMD_SET_SOR_OP_MODE_PU_TXDB_0_ENABLE  0x00000001
#define LW5070_CTRL_CMD_SET_SOR_OP_MODE_PU_TXDB_1                           1:1
#define LW5070_CTRL_CMD_SET_SOR_OP_MODE_PU_TXDB_1_DISABLE 0x00000000
#define LW5070_CTRL_CMD_SET_SOR_OP_MODE_PU_TXDB_1_ENABLE  0x00000001
#define LW5070_CTRL_CMD_SET_SOR_OP_MODE_PU_TXDB_2                           2:2
#define LW5070_CTRL_CMD_SET_SOR_OP_MODE_PU_TXDB_2_DISABLE 0x00000000
#define LW5070_CTRL_CMD_SET_SOR_OP_MODE_PU_TXDB_2_ENABLE  0x00000001
#define LW5070_CTRL_CMD_SET_SOR_OP_MODE_PU_TXDB_3                           3:3
#define LW5070_CTRL_CMD_SET_SOR_OP_MODE_PU_TXDB_3_DISABLE 0x00000000
#define LW5070_CTRL_CMD_SET_SOR_OP_MODE_PU_TXDB_3_ENABLE  0x00000001

#define LW5070_CTRL_CMD_SET_SOR_OP_MODE_PU_TXCA                             0:0
#define LW5070_CTRL_CMD_SET_SOR_OP_MODE_PU_TXCA_DISABLE   0x00000000
#define LW5070_CTRL_CMD_SET_SOR_OP_MODE_PU_TXCA_ENABLE    0x00000001

#define LW5070_CTRL_CMD_SET_SOR_OP_MODE_PU_TXCB                             0:0
#define LW5070_CTRL_CMD_SET_SOR_OP_MODE_PU_TXCB_DISABLE   0x00000000
#define LW5070_CTRL_CMD_SET_SOR_OP_MODE_PU_TXCB_ENABLE    0x00000001

#define LW5070_CTRL_CMD_SET_SOR_OP_MODE_UPPER                               0:0
#define LW5070_CTRL_CMD_SET_SOR_OP_MODE_UPPER_UPPER_RESET 0x00000001

#define LW5070_CTRL_CMD_SET_SOR_OP_MODE_MODE                                0:0
#define LW5070_CTRL_CMD_SET_SOR_OP_MODE_MODE_LVDS         0x00000000
#define LW5070_CTRL_CMD_SET_SOR_OP_MODE_MODE_TMDS         0x00000001

#define LW5070_CTRL_CMD_SET_SOR_OP_MODE_LINKACTA                            0:0
#define LW5070_CTRL_CMD_SET_SOR_OP_MODE_LINKACTA_DISABLE  0x00000000
#define LW5070_CTRL_CMD_SET_SOR_OP_MODE_LINKACTA_ENABLE   0x00000001

#define LW5070_CTRL_CMD_SET_SOR_OP_MODE_LINKACTB                            0:0
#define LW5070_CTRL_CMD_SET_SOR_OP_MODE_LINKACTB_DISABLE  0x00000000
#define LW5070_CTRL_CMD_SET_SOR_OP_MODE_LINKACTB_ENABLE   0x00000001

#define LW5070_CTRL_CMD_SET_SOR_OP_MODE_LVDS_EN                             0:0
#define LW5070_CTRL_CMD_SET_SOR_OP_MODE_LVDS_EN_DISABLE   0x00000000
#define LW5070_CTRL_CMD_SET_SOR_OP_MODE_LVDS_EN_ENABLE    0x00000001

#define LW5070_CTRL_CMD_SET_SOR_OP_MODE_LVDS_DUAL                           0:0
#define LW5070_CTRL_CMD_SET_SOR_OP_MODE_LVDS_DUAL_DISABLE 0x00000000
#define LW5070_CTRL_CMD_SET_SOR_OP_MODE_LVDS_DUAL_ENABLE  0x00000001

#define LW5070_CTRL_CMD_SET_SOR_OP_MODE_DUP_SYNC                            0:0
#define LW5070_CTRL_CMD_SET_SOR_OP_MODE_DUP_SYNC_DISABLE  0x00000000
#define LW5070_CTRL_CMD_SET_SOR_OP_MODE_DUP_SYNC_ENABLE   0x00000001

#define LW5070_CTRL_CMD_SET_SOR_OP_MODE_NEW_MODE                            0:0
#define LW5070_CTRL_CMD_SET_SOR_OP_MODE_NEW_MODE_DISABLE  0x00000000
#define LW5070_CTRL_CMD_SET_SOR_OP_MODE_NEW_MODE_ENABLE   0x00000001

#define LW5070_CTRL_CMD_SET_SOR_OP_MODE_BALANCED                            0:0
#define LW5070_CTRL_CMD_SET_SOR_OP_MODE_BALANCED_DISABLE  0x00000000
#define LW5070_CTRL_CMD_SET_SOR_OP_MODE_BALANCED_ENABLE   0x00000001

#define LW5070_CTRL_CMD_SET_SOR_OP_MODE_PLLDIV                              0:0
#define LW5070_CTRL_CMD_SET_SOR_OP_MODE_PLLDIV_BY_7       0x00000000
#define LW5070_CTRL_CMD_SET_SOR_OP_MODE_PLLDIV_BY_10      0x00000001

#define LW5070_CTRL_CMD_SET_SOR_OP_MODE_ROTCLK                              3:0

#define LW5070_CTRL_CMD_SET_SOR_OP_MODE_ROTDAT                              2:0

#define LW5070_CTRL_CMD_SET_SOR_OP_MODE_PARAMS_MESSAGE_ID (0x23U)

typedef struct LW5070_CTRL_CMD_SET_SOR_OP_MODE_PARAMS {
    LW5070_CTRL_CMD_BASE_PARAMS base;
    LwU32                       orNumber;

    LwU32                       category;
    LwU32                       puTxda;
    LwU32                       puTxdb;
    LwU32                       puTxca;
    LwU32                       puTxcb;
    LwU32                       upper;
    LwU32                       mode;
    LwU32                       linkActA;
    LwU32                       linkActB;
    LwU32                       lvdsEn;
    LwU32                       lvdsDual;
    LwU32                       dupSync;
    LwU32                       newMode;
    LwU32                       balanced;
    LwU32                       plldiv;
    LwU32                       rotClk;
    LwU32                       rotDat;
} LW5070_CTRL_CMD_SET_SOR_OP_MODE_PARAMS;


#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)

/*
 * LW5070_CTRL_CMD_GET_PIOR_OP_MODE
 *
 * This command returns current settings for the specified PIOR.
 *
 *      orNumber
 *          The OR number for which the operating mode needs to be programmed.
 *
 *      category
 *          Whether ext TMDS, TV, DRO or DRI settings are desired.
 *          EXT TV is not supported at the moment.
 *          EXT DisplayPort is specified through EXT 10BPC 444.
 *
 *      clkPolarity
 *          Whether or not output clock is ilwerted relative to generated clock.
 *
 *      clkMode
 *          Whether data being transmitted is SDR or DDR.
 *
 *      clkPhs
 *          Position of the edge on which data is launched.
 *
 *      unusedPins
 *          Status of unused pins of this PIOR.
 *
 *      polarity
 *          Whether or not sync and DE pin polarities are ilwerted.
 *
 *      dataMuxing
 *          How are the bits are multiplexed together.
 *
 *      clkDelay
 *          Extra delay for the clock.
 *
 *      dataDelay
 *          Extra delay for the data.
 *
 * Possible status values returned are:
 *      LW_OK
 *      LW_ERR_ILWALID_ARGUMENT
 *      LW_ERR_GENERIC
 */
#define LW5070_CTRL_CMD_GET_PIOR_OP_MODE                        (0x50700430) /* finn: Evaluated from "(FINN_LW50_DISPLAY_OR_INTERFACE_ID << 8) | LW5070_CTRL_CMD_GET_PIOR_OP_MODE_PARAMS_MESSAGE_ID" */

#define LW5070_CTRL_CMD_GET_PIOR_OP_MODE_CATEGORY                           2:0
#define LW5070_CTRL_CMD_GET_PIOR_OP_MODE_CATEGORY_EXT_TMDS      0x00000000
#define LW5070_CTRL_CMD_GET_PIOR_OP_MODE_CATEGORY_EXT_TV        0x00000001
#define LW5070_CTRL_CMD_GET_PIOR_OP_MODE_CATEGORY_DRO           0x00000003
#define LW5070_CTRL_CMD_GET_PIOR_OP_MODE_CATEGORY_DRI           0x00000004
#define LW5070_CTRL_CMD_GET_PIOR_OP_MODE_CATEGORY_EXT_10BPC_444 0x00000005

#define LW5070_CTRL_CMD_GET_PIOR_OP_MODE_CLK_POLARITY                       0:0
#define LW5070_CTRL_CMD_GET_PIOR_OP_MODE_CLK_POLARITY_NORMAL    0x00000000
#define LW5070_CTRL_CMD_GET_PIOR_OP_MODE_CLK_POLARITY_ILW       0x00000001

#define LW5070_CTRL_CMD_GET_PIOR_OP_MODE_CLK_MODE                           0:0
#define LW5070_CTRL_CMD_GET_PIOR_OP_MODE_CLK_MODE_SDR           0x00000000
#define LW5070_CTRL_CMD_GET_PIOR_OP_MODE_CLK_MODE_DDR           0x00000001

#define LW5070_CTRL_CMD_GET_PIOR_OP_MODE_CLK_PHS                            1:0
#define LW5070_CTRL_CMD_GET_PIOR_OP_MODE_CLK_PHS_0              0x00000000
#define LW5070_CTRL_CMD_GET_PIOR_OP_MODE_CLK_PHS_1              0x00000001
#define LW5070_CTRL_CMD_GET_PIOR_OP_MODE_CLK_PHS_2              0x00000002
#define LW5070_CTRL_CMD_GET_PIOR_OP_MODE_CLK_PHS_3              0x00000003

#define LW5070_CTRL_CMD_GET_PIOR_OP_MODE_UNUSED_PINS                        0:0
#define LW5070_CTRL_CMD_GET_PIOR_OP_MODE_UNUSED_PINS_LO         0x00000000
#define LW5070_CTRL_CMD_GET_PIOR_OP_MODE_UNUSED_PINS_TS         0x00000001

#define LW5070_CTRL_CMD_GET_PIOR_OP_MODE_POLARITY_H                         0:0
#define LW5070_CTRL_CMD_GET_PIOR_OP_MODE_POLARITY_H_NORMAL      0x00000000
#define LW5070_CTRL_CMD_GET_PIOR_OP_MODE_POLARITY_H_ILW         0x00000001
#define LW5070_CTRL_CMD_GET_PIOR_OP_MODE_POLARITY_V                         1:1
#define LW5070_CTRL_CMD_GET_PIOR_OP_MODE_POLARITY_V_NORMAL      0x00000000
#define LW5070_CTRL_CMD_GET_PIOR_OP_MODE_POLARITY_V_ILW         0x00000001
#define LW5070_CTRL_CMD_GET_PIOR_OP_MODE_POLARITY_DE                        2:2
#define LW5070_CTRL_CMD_GET_PIOR_OP_MODE_POLARITY_DE_NORMAL     0x00000000
#define LW5070_CTRL_CMD_GET_PIOR_OP_MODE_POLARITY_DE_ILW        0x00000001

#define LW5070_CTRL_CMD_GET_PIOR_OP_MODE_DATA_MUXING                        3:0
#define LW5070_CTRL_CMD_GET_PIOR_OP_MODE_DATA_MUXING_RGB_0      0x00000000
#define LW5070_CTRL_CMD_GET_PIOR_OP_MODE_DATA_MUXING_RGB_1      0x00000001
#define LW5070_CTRL_CMD_GET_PIOR_OP_MODE_DATA_MUXING_DIST_RNDR  0x00000003
#define LW5070_CTRL_CMD_GET_PIOR_OP_MODE_DATA_MUXING_YUV_0      0x00000004
#define LW5070_CTRL_CMD_GET_PIOR_OP_MODE_DATA_MUXING_UYVY       0x00000005

#define LW5070_CTRL_CMD_GET_PIOR_OP_MODE_CLK_DLY                            2:0

#define LW5070_CTRL_CMD_GET_PIOR_OP_MODE_DATA_DLY                           2:0

#define LW5070_CTRL_CMD_GET_PIOR_OP_MODE_PARAMS_MESSAGE_ID (0x30U)

typedef struct LW5070_CTRL_CMD_GET_PIOR_OP_MODE_PARAMS {
    LW5070_CTRL_CMD_BASE_PARAMS base;
    LwU32                       orNumber;

    LwU32                       category;
    LwU32                       clkPolarity;
    LwU32                       clkMode;
    LwU32                       clkPhs;
    LwU32                       unusedPins;
    LwU32                       polarity;
    LwU32                       dataMuxing;
    LwU32                       clkDelay;
    LwU32                       dataDelay;
} LW5070_CTRL_CMD_GET_PIOR_OP_MODE_PARAMS;


/*
 * LW5070_CTRL_CMD_SET_PIOR_OP_MODE
 *
 * This command applies the specified settings to the specified PIOR.
 *
 *      orNumber
 *          The OR number for which the operating mode needs to be programmed.
 *
 *      category
 *          Whether ext TMDS, TV, DRO or DRI settings are to be programmed.
 *          EXT TV is not supported at the moment.
 *          EXT DisplayPort is specified through EXT 10BPC 444.
 *
 *      clkPolarity
 *          Whether or not to ilwert output clock relative to generated clock.
 *
 *      clkMode
 *          Whether data being transmitted should be SDR or DDR.
 *
 *      clkPhs
 *          Position of the edge on which data should be launched.
 *
 *      unusedPins
 *          What to do with unused pins of this PIOR.
 *
 *      polarity
 *          Whether or not to ilwert sync and DE pin polarities.
 *
 *      dataMuxing
 *          How to multiplex the bits together.
 *
 *      clkDelay
 *          Extra delay for the clock.
 *
 *      dataDelay
 *          Extra delay for the data.
 *
 * Possible status values returned are:
 *      LW_OK
 *      LW_ERR_ILWALID_ARGUMENT
 *      LW_ERR_GENERIC
 */
#define LW5070_CTRL_CMD_SET_PIOR_OP_MODE                           (0x50700431) /* finn: Evaluated from "(FINN_LW50_DISPLAY_OR_INTERFACE_ID << 8) | LW5070_CTRL_CMD_SET_PIOR_OP_MODE_PARAMS_MESSAGE_ID" */

#define LW5070_CTRL_CMD_SET_PIOR_OP_MODE_CATEGORY                           2:0
#define LW5070_CTRL_CMD_SET_PIOR_OP_MODE_CATEGORY_EXT_TMDS         0x00000000
#define LW5070_CTRL_CMD_SET_PIOR_OP_MODE_CATEGORY_EXT_TV           0x00000001
#define LW5070_CTRL_CMD_SET_PIOR_OP_MODE_CATEGORY_DRO              0x00000003
#define LW5070_CTRL_CMD_SET_PIOR_OP_MODE_CATEGORY_DRI              0x00000004
#define LW5070_CTRL_CMD_SET_PIOR_OP_MODE_CATEGORY_EXT_10BPC_444    0x00000005

#define LW5070_CTRL_CMD_SET_PIOR_OP_MODE_CLK_POLARITY                       0:0
#define LW5070_CTRL_CMD_SET_PIOR_OP_MODE_CLK_POLARITY_NORMAL       0x00000000
#define LW5070_CTRL_CMD_SET_PIOR_OP_MODE_CLK_POLARITY_ILW          0x00000001

#define LW5070_CTRL_CMD_SET_PIOR_OP_MODE_CLK_MODE                           0:0
#define LW5070_CTRL_CMD_SET_PIOR_OP_MODE_CLK_MODE_SDR              0x00000000
#define LW5070_CTRL_CMD_SET_PIOR_OP_MODE_CLK_MODE_DDR              0x00000001

#define LW5070_CTRL_CMD_SET_PIOR_OP_MODE_CLK_PHS                            1:0
#define LW5070_CTRL_CMD_SET_PIOR_OP_MODE_CLK_PHS_0                 0x00000000
#define LW5070_CTRL_CMD_SET_PIOR_OP_MODE_CLK_PHS_1                 0x00000001
#define LW5070_CTRL_CMD_SET_PIOR_OP_MODE_CLK_PHS_2                 0x00000002
#define LW5070_CTRL_CMD_SET_PIOR_OP_MODE_CLK_PHS_3                 0x00000003

#define LW5070_CTRL_CMD_SET_PIOR_OP_MODE_UNUSED_PINS                        0:0
#define LW5070_CTRL_CMD_SET_PIOR_OP_MODE_UNUSED_PINS_LO            0x00000000
#define LW5070_CTRL_CMD_SET_PIOR_OP_MODE_UNUSED_PINS_TS            0x00000001

#define LW5070_CTRL_CMD_SET_PIOR_OP_MODE_POLARITY_H                         0:0
#define LW5070_CTRL_CMD_SET_PIOR_OP_MODE_POLARITY_H_NORMAL         0x00000000
#define LW5070_CTRL_CMD_SET_PIOR_OP_MODE_POLARITY_H_ILW            0x00000001
#define LW5070_CTRL_CMD_SET_PIOR_OP_MODE_POLARITY_V                         1:1
#define LW5070_CTRL_CMD_SET_PIOR_OP_MODE_POLARITY_V_NORMAL         0x00000000
#define LW5070_CTRL_CMD_SET_PIOR_OP_MODE_POLARITY_V_ILW            0x00000001
#define LW5070_CTRL_CMD_SET_PIOR_OP_MODE_POLARITY_DE                        2:2
#define LW5070_CTRL_CMD_SET_PIOR_OP_MODE_POLARITY_DE_NORMAL        0x00000000
#define LW5070_CTRL_CMD_SET_PIOR_OP_MODE_POLARITY_DE_ILW           0x00000001

#define LW5070_CTRL_CMD_SET_PIOR_OP_MODE_DATA_MUXING                        3:0
#define LW5070_CTRL_CMD_SET_PIOR_OP_MODE_DATA_MUXING_RGB_0         0x00000000
#define LW5070_CTRL_CMD_SET_PIOR_OP_MODE_DATA_MUXING_RGB_1         0x00000001
#define LW5070_CTRL_CMD_SET_PIOR_OP_MODE_DATA_MUXING_DIST_RNDR     0x00000003
#define LW5070_CTRL_CMD_SET_PIOR_OP_MODE_DATA_MUXING_YUV_0         0x00000004
#define LW5070_CTRL_CMD_SET_PIOR_OP_MODE_DATA_MUXING_UYVY          0x00000005

#define LW5070_CTRL_CMD_SET_PIOR_OP_MODE_CLK_DLY                            2:0

#define LW5070_CTRL_CMD_SET_PIOR_OP_MODE_DATA_DLY                           2:0

#define LW5070_CTRL_CMD_SET_PIOR_OP_MODE_DRO_MASTER                         1:0

#define LW5070_CTRL_CMD_SET_PIOR_OP_MODE_DRO_DRIVE_PIN_SET                  2:0
#define LW5070_CTRL_CMD_SET_PIOR_OP_MODE_DRO_DRIVE_PIN_SET_NEITHER 0
#define LW5070_CTRL_CMD_SET_PIOR_OP_MODE_DRO_DRIVE_PIN_SET_A       1
#define LW5070_CTRL_CMD_SET_PIOR_OP_MODE_DRO_DRIVE_PIN_SET_B       2


#define LW5070_CTRL_CMD_SET_PIOR_OP_MODE_PARAMS_MESSAGE_ID (0x31U)

typedef struct LW5070_CTRL_CMD_SET_PIOR_OP_MODE_PARAMS {
    LW5070_CTRL_CMD_BASE_PARAMS base;
    LwU32                       orNumber;

    LwU32                       category;
    LwU32                       clkPolarity;
    LwU32                       clkMode;
    LwU32                       clkPhs;
    LwU32                       unusedPins;
    LwU32                       polarity;
    LwU32                       dataMuxing;
    LwU32                       clkDelay;
    LwU32                       dataDelay;
    LwU32                       dro_master;
    LwU32                       dro_drive_pin_set;
} LW5070_CTRL_CMD_SET_PIOR_OP_MODE_PARAMS;

/* LWRM_PUBLISHED_PENDING_IP_REVIEW */
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)



/*
 * LW5070_CTRL_CMD_SET_SOR_FLUSH_MODE
 *
 * Set the given SOR number into flush mode in preparation for DP link training.
 *
 *   orNumber [in]
 *     The SOR number to set into flush mode.
 *
 *   bEnable [in]
 *     Whether to enable or disable flush mode on this SOR.
 * 
 *   bImmediate [in]
 *     If set to true, will enable flush in immediate mode.
 *     If not, will enable flush in loadv mode.
 *     NOTE: We do not support exit flush in LoadV mode.
 * 
 *   headMask [in]
 *     Optional.  If set brings only the heads in the head mask out of flush
 *     OR will stay in flush mode until last head is out of flush mode.
 *     Caller can use _HEAD__ALL to specify all the heads are to be brought out.
 *     NOTE: headMask would be considered only while exiting from flush mode.
 * 
 *   bForceRgDiv [in]
 *      If set forces RgDiv. Should be used only for HW/SW testing
 * 
 *   bUseBFM [in]
 *      If Set then it mean we are using BFM else exelwting on non-BFM paltforms.
 * 
 *   bFireAndForget [in]
 *       Fire the flush mode & perform post-processing without waiting for it
 *       to be done. This is required for special cases like GC5 where we have
 *       ELV blocked, RG stall & we trigger flush for one shot mode & then do
 *       a modeset by disabling it without actually waiting for it to get
 *       disabled. We will not get any vblank interrupt in this case as we have
 *       stalled RG.
 * 
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 */

#define LW5070_CTRL_CMD_SET_SOR_FLUSH_MODE (0x50700457) /* finn: Evaluated from "(FINN_LW50_DISPLAY_OR_INTERFACE_ID << 8) | LW5070_CTRL_SET_SOR_FLUSH_MODE_PARAMS_MESSAGE_ID" */

#define LW5070_CTRL_SET_SOR_FLUSH_MODE_PARAMS_MESSAGE_ID (0x57U)

typedef struct LW5070_CTRL_SET_SOR_FLUSH_MODE_PARAMS {
    LW5070_CTRL_CMD_BASE_PARAMS base;
    LwU32                       sorNumber;
    LwBool                      bEnable;
    LwBool                      bImmediate;
    LwU32                       headMask;
    LwBool                      bForceRgDiv;
    LwBool                      bUseBFM;
    LwBool                      bFireAndForget;
} LW5070_CTRL_SET_SOR_FLUSH_MODE_PARAMS;

#define LW5070_CTRL_SET_SOR_FLUSH_MODE_PARAMS_HEADMASK_HEAD(i)          (i):(i)
#define LW5070_CTRL_SET_SOR_FLUSH_MODE_PARAMS_HEADMASK_HEAD__SIZE_1 LW5070_CTRL_CMD_MAX_HEADS
#define LW5070_CTRL_SET_SOR_FLUSH_MODE_PARAMS_HEADMASK_HEAD_ALL     0xFFFFFFFF

#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

/*
 * LW5070_CTRL_CMD_SET_PIOR_VIDEOBRIDGE_OP_MODE
 *
 * This command applies the specified settings to the video bridge.
 *
 *      orNumber
 *          The OR number for which the operating mode needs to be programmed.
 *
 *      operationMode
 *          Used to enable or disable a specific mode.
 *          Valid values are:
 *              LW5070_CTRL_CMD_SET_PIOR_VIDEOBRIDGE_OP_MODE_NONE:
 *                  Unsets any mode
 *              LW5070_CTRL_CMD_SET_PIOR_VIDEOBRIDGE_OP_MODE_DUALMIO:
 *                  Allows DualMIO to be engaged during modeset. The mode is persistent
 *                  and needs to be disabled explicitly.
 *                  Returns LW_ERR_ILWALID_STATE if the GPUs are not connected
 *                  with a dual video bridge (two video bridges betyween a pair of GPUs).
 *                  Returns LWOS_ERR_NOT_SUPPORTED if the GPU does not support DualMIO.
 *
 * Possible status values returned are:
 *      LW_OK
 *      LW_ERR_ILWALID_ARGUMENT
 *      LW_ERR_ILWALID_STATE
 *      LWOS_ERR_NOT_SUPPORTED
 *      LW_ERR_GENERIC
 */
#define LW5070_CTRL_CMD_SET_PIOR_VIDEOBRIDGE_OP_MODE                (0x50700458) /* finn: Evaluated from "(FINN_LW50_DISPLAY_OR_INTERFACE_ID << 8) | LW5070_CTRL_SET_PIOR_VIDEOBRIDGE_OP_MODE_PARAMS_MESSAGE_ID" */

#define LW5070_CTRL_CMD_SET_PIOR_VIDEOBRIDGE_OP_MODE_NONE           0
#define LW5070_CTRL_CMD_SET_PIOR_VIDEOBRIDGE_OP_MODE_DUALMIO        1
#define LW5070_CTRL_SET_PIOR_VIDEOBRIDGE_OP_MODE_PARAMS_MESSAGE_ID (0x58U)

typedef struct LW5070_CTRL_SET_PIOR_VIDEOBRIDGE_OP_MODE_PARAMS {
    LW5070_CTRL_CMD_BASE_PARAMS base;
    LwU32                       orNumber;

    LwU32                       operationMode;
} LW5070_CTRL_SET_PIOR_VIDEOBRIDGE_OP_MODE_PARAMS;

/* LWRM_UNPUBLISHED */
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)



/* _ctrl5070or_h_ */
