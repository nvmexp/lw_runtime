/*
 * SPDX-FileCopyrightText: Copyright (c) 2006-2015 LWPU CORPORATION & AFFILIATES. All rights reserved.
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
// Source file: ctrl/ctrl2080/ctrl2080fuse.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)



#include "ctrl/ctrl2080/ctrl2080base.h"

/* LW20_SUBDEVICE_XX fuse-related control commands and parameters */

/*
 * LW2080_CTRL_CMD_FUSE_EXELWTE_FUB_BINARY
 * 
 * This command exelwtes FUB binary on LWDEC falcon
 * 
 * returnStatus 
 *     RM function which actually loads FUB binary will pass on 
 *     FUB error/status code to MODS test as control call parameter
 */
#define LW2080_CTRL_CMD_FUSE_EXELWTE_FUB_BINARY (0x20800203) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_FUSE_INTERFACE_ID << 8) | LW2080_CTRL_CMD_FUSE_EXELWTE_FUB_BINARY_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_CMD_FUSE_EXELWTE_FUB_BINARY_PARAMS_MESSAGE_ID (0x3U)

typedef struct LW2080_CTRL_CMD_FUSE_EXELWTE_FUB_BINARY_PARAMS {
    LwU32 returnStatus;
} LW2080_CTRL_CMD_FUSE_EXELWTE_FUB_BINARY_PARAMS;

// Fuse Id
#define LW2080_CTRL_FUSE_ID_STRAP_SPEEDO                    0x00  // Speedo value
#define LW2080_CTRL_FUSE_ID_STRAP_SPEEDO_VERSION            0x01  // Speedo version
#define LW2080_CTRL_FUSE_ID_STRAP_IDDQ                      0x02  // IDDQ value for voltage rail 0
#define LW2080_CTRL_FUSE_ID_STRAP_IDDQ_VERSION              0x03  // IDDQ version
#define LW2080_CTRL_FUSE_ID_STRAP_IDDQ_1                    0x04  // IDDQ value for voltage rail 1
#define LW2080_CTRL_FUSE_ID_STRAP_BOARD_BINNING             0x05  // Speedo value for Board Binning
#define LW2080_CTRL_FUSE_ID_STRAP_BOARD_BINNING_VERSION     0x06  // Board Binning version
#define LW2080_CTRL_FUSE_ID_STRAP_SRAM_VMIN                 0x07  // SRAM Vmin
#define LW2080_CTRL_FUSE_ID_STRAP_SRAM_VMIN_VERSION         0x08  // SRAM Vmin version
#define LW2080_CTRL_FUSE_ID_STRAP_BOOT_VMIN_LWVDD           0x09  // LWVDD Boot Vmin
#define LW2080_CTRL_FUSE_ID_ISENSE_VCM_OFFSET               0x0A  // ISENSE VCM Offset
#define LW2080_CTRL_FUSE_ID_ISENSE_DIFF_GAIN                0x0B  // ISENSE Differential Gain
#define LW2080_CTRL_FUSE_ID_ISENSE_DIFF_OFFSET              0x0C  // ISENSE Differential Offset
#define LW2080_CTRL_FUSE_ID_ISENSE_CALIBRATION_VERSION      0x0D  // ISENSE Calibration version. This is a common version for the 3 fields above
#define LW2080_CTRL_FUSE_ID_KAPPA                           0x0E  // KAPPA fuse - Will link to fuse opt_kappa_info
#define LW2080_CTRL_FUSE_ID_KAPPA_VERSION                   0x0F  // KAPPA version.
#define LW2080_CTRL_FUSE_ID_STRAP_SPEEDO_1                  0x10  // SPEEDO_1
#define LW2080_CTRL_FUSE_ID_CPM_VERSION                     0x11  // Fuse OPT_CPM_REV
#define LW2080_CTRL_FUSE_ID_CPM_0                           0x12  // Fuse OPT_CPM0
#define LW2080_CTRL_FUSE_ID_CPM_1                           0x13  // Fuse OPT_CPM1
#define LW2080_CTRL_FUSE_ID_CPM_2                           0x14  // Fuse OPT_CPM2
#define LW2080_CTRL_FUSE_ID_ISENSE_VCM_COARSE_OFFSET        0x15  // ISENSE VCM Coarse Offset
#define LW2080_CTRL_FUSE_ID_STRAP_BOOT_VMIN_MSVDD           0x16  // MSVDD Boot Vmin
#define LW2080_CTRL_FUSE_ID_KAPPA_VALID                     0x17  // KAPPA fuse
#define LW2080_CTRL_FUSE_ID_IDDQ_LWVDD                      0x18  // LWVDD IDDQ
#define LW2080_CTRL_FUSE_ID_IDDQ_MSVDD                      0x19  // MSVDD IDDQ
#define LW2080_CTRL_FUSE_ID_STRAP_SPEEDO_2                  0x1A  // SPEEDO_2
#define LW2080_CTRL_FUSE_ID_OC_BIN                          0x1B  // OC_BIN
#define LW2080_CTRL_FUSE_ID_LV_FMAX_KNOB                    0x1C  // LV_FMAX_KNOB
#define LW2080_CTRL_FUSE_ID_MV_FMAX_KNOB                    0x1D  // MV_FMAX_KNOB
#define LW2080_CTRL_FUSE_ID_HV_FMAX_KNOB                    0x1E  // HV_FMAX_KNOB
#define LW2080_CTRL_FUSE_ID_PSTATE_VMIN_KNOB                0x1F  // PSTATE_VMIN_KNOB 
#define LW2080_CTRL_FUSE_ID_ISENSE_DIFFERENTIAL_COARSE_GAIN 0x20  // ISENSE DIFFERENTIAL Coarse Gain
#define LW2080_CTRL_FUSE_ID_ILWALID                         0xFF

/* _ctrl2080fuse_h_ */
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

