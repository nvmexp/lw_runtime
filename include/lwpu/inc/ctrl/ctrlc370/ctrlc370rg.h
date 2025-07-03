/*
 * SPDX-FileCopyrightText: Copyright (c) 2015-2021 LWPU CORPORATION & AFFILIATES. All rights reserved.
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
// Source file: ctrl/ctrlc370/ctrlc370rg.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)



#include "ctrl/ctrlc370/ctrlc370base.h"
/* C370 is partially derived from 5070 */
#include "ctrl/ctrl5070/ctrl5070rg.h"

#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

/*
 * The following control calls are defined in ctrl5070rg.h, but they are
 * still supported on LWC370. We redirect these control cmds to LW5070_CTRL_CMD,
 * and keep the _PARAMS unchanged for now.
 */

#define LWC370_CTRL_CMD_GET_RG_STATUS                      LW5070_CTRL_CMD_GET_RG_STATUS
#define LWC370_CTRL_CMD_GET_RG_UNDERFLOW_PROP              LW5070_CTRL_CMD_GET_RG_UNDERFLOW_PROP
#define LWC370_CTRL_CMD_SET_RG_UNDERFLOW_PROP              LW5070_CTRL_CMD_SET_RG_UNDERFLOW_PROP
#define LWC370_CTRL_CMD_GET_RG_FLIPLOCK_PROP               LW5070_CTRL_CMD_GET_RG_FLIPLOCK_PROP
#define LWC370_CTRL_CMD_SET_RG_FLIPLOCK_PROP               LW5070_CTRL_CMD_SET_RG_FLIPLOCK_PROP
#define LWC370_CTRL_CMD_GET_RG_CONNECTED_LOCKPIN           LW5070_CTRL_CMD_GET_RG_CONNECTED_LOCKPIN
#define LWC370_CTRL_CMD_SET_VIDEO_STATUS                   LW5070_CTRL_CMD_SET_VIDEO_STATUS
#define LWC370_CTRL_CMD_GET_RG_CONNECTED_LOCKPIN_STATELESS LW5070_CTRL_CMD_GET_RG_CONNECTED_LOCKPIN_STATELESS
#define LWC370_CTRL_CMD_GET_PINSET_LOCKPINS                LW5070_CTRL_CMD_GET_PINSET_LOCKPINS
#define LWC370_CTRL_CMD_GET_RG_SCAN_LINE                   LW5070_CTRL_CMD_GET_RG_SCAN_LINE
#define LWC370_CTRL_CMD_GET_FRAMELOCK_HEADER_LOCKPINS      LW5070_CTRL_CMD_GET_FRAMELOCK_HEADER_LOCKPINS
#define LWC370_CTRL_CMD_GET_STEREO_PHASE                   LW5070_CTRL_CMD_GET_STEREO_PHASE

/* LWRM_UNPUBLISHED */
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)




/*
 * LWC370_CTRL_CMD_GET_LOCKPINS_CAPS
 *
 * This command returns lockpins for the specified pinset,
 * as well as lockpins' HW capabilities.
 *
 *   pinset [in]
 *     This parameter takes the pinset whose corresponding
 *     lockpin numbers need to be determined. This only affects
 *     the return value for the RaterLock and FlipLock pins.
 *
 *   frameLockPin [out]
 *     This parameter returns the FrameLock pin index.
 *
 *   rasterLockPin [out]
 *     This parameter returns the RasterLock pin index.
 *
 *   flipLockPin [out]
 *     This parameter returns the FlipLock pin index.
 *
 *   stereoPin [out]
 *     This parameter returns the Stereo pin index.
 *
 *   numScanLockPins [out]
 *     This parameter returns the HW capability of ScanLock pins.
 *
 *   numFlipLockPins [out]
 *     This parameter returns the HW capability of FlipLock pins.
 *
 *   numStereoPins [out]
 *     This parameter returns the HW capability of Stereo pins.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_NOT_SUPPORTED
 */

#define LWC370_CTRL_CMD_GET_LOCKPINS_CAPS                  (0xc3700201) /* finn: Evaluated from "(FINN_LWC370_DISPLAY_RG_INTERFACE_ID << 8) | LWC370_CTRL_GET_LOCKPINS_CAPS_PARAMS_MESSAGE_ID" */

#define LWC370_CTRL_GET_LOCKPINS_CAPS_FRAME_LOCK_PIN_NONE  (0xffffffff)
#define LWC370_CTRL_GET_LOCKPINS_CAPS_RASTER_LOCK_PIN_NONE (0xffffffff)
#define LWC370_CTRL_GET_LOCKPINS_CAPS_FLIP_LOCK_PIN_NONE   (0xffffffff)
#define LWC370_CTRL_GET_LOCKPINS_CAPS_STEREO_PIN_NONE      (0xffffffff)
#define LWC370_CTRL_GET_LOCKPINS_CAPS_PARAMS_MESSAGE_ID (0x1U)

typedef struct LWC370_CTRL_GET_LOCKPINS_CAPS_PARAMS {
    LWC370_CTRL_CMD_BASE_PARAMS base;
    LwU32                       pinset;
    LwU32                       frameLockPin;
    LwU32                       rasterLockPin;
    LwU32                       flipLockPin;
    LwU32                       stereoPin;
    LwU32                       numScanLockPins;
    LwU32                       numFlipLockPins;
    LwU32                       numStereoPins;
} LWC370_CTRL_GET_LOCKPINS_CAPS_PARAMS;

/*
 * LWC370_CTRL_CMD_SET_SWAPRDY_GPIO_WAR
 *
 * This command switches SWAP_READY_OUT GPIO between SW
 * and HW control to WAR bug 200374184
 *
 *   bEnable [in]:
 *     This parameter indicates enable/disable external fliplock
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_GENERIC
 */

#define LWC370_CTRL_CMD_SET_SWAPRDY_GPIO_WAR (0xc3700202) /* finn: Evaluated from "(FINN_LWC370_DISPLAY_RG_INTERFACE_ID << 8) | LWC370_CTRL_SET_SWAPRDY_GPIO_WAR_PARAMS_MESSAGE_ID" */

#define LWC370_CTRL_SET_SWAPRDY_GPIO_WAR_PARAMS_MESSAGE_ID (0x2U)

typedef struct LWC370_CTRL_SET_SWAPRDY_GPIO_WAR_PARAMS {
    LWC370_CTRL_CMD_BASE_PARAMS base;
    LwBool                      bEnable;
} LWC370_CTRL_SET_SWAPRDY_GPIO_WAR_PARAMS;


