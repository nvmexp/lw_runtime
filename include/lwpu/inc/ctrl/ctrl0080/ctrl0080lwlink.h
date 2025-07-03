/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2021 LWPU CORPORATION & AFFILIATES. All rights reserved.
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
// Source file: ctrl/ctrl0080/ctrl0080lwlink.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)



#include "ctrl/ctrl0080/ctrl0080base.h"

/*
 * LW0080_CTRL_CMD_LWLINK_SYNC_LINK_MASKS_AND_VBIOS_INFO
 *
 * Syncs the different link masks and vbios defined values between CPU-RM and GSP-RM
 *
 * [in]  discoveredLinks
 *     Mask of links discovered from IOCTRLs
 *
 * [in]  connectedLinksMask
 *     Mask of links which are connected (remote present)
 *
 * [in]  bridgeSensableLinks
 *     Mask of links whose remote endpoint presence can be sensed
 *
 * [in]  bridgedLinks
 *    Mask of links which are connected (remote present)
 *    Same as connectedLinksMask, but also tracks the case where link
 *    is connected but marginal and could not initialize
 *
 * [out] initDisabledLinksMask
 *      Mask of links for which initialization is disabled
 *
 * [out] vbiosDisabledLinkMask
 *      Mask of links disabled in the VBIOS
 *
 * [out] initializedLinks
 *      Mask of initialized links
 *
 * [out] bEnableTrainingAtLoad
 *      Whether the links should be trained to active during driver load
 *
 * [out] bEnableSafeModeAtLoad
 *      Whether the links should be initialized to swcfg during driver load
 */
#define LW0080_CTRL_CMD_LWLINK_SYNC_LINK_MASKS_AND_VBIOS_INFO (0x802101) /* finn: Evaluated from "(FINN_LW01_DEVICE_0_LWLINK_INTERFACE_ID << 8) | LW0080_CTRL_LWLINK_SYNC_LINK_MASKS_AND_VBIOS_INFO_PARAMS_MESSAGE_ID" */

#define LW0080_CTRL_LWLINK_SYNC_LINK_MASKS_AND_VBIOS_INFO_PARAMS_MESSAGE_ID (0x1U)

typedef struct LW0080_CTRL_LWLINK_SYNC_LINK_MASKS_AND_VBIOS_INFO_PARAMS {
    LwU32  discoveredLinks;
    LwU32  connectedLinksMask;
    LwU32  bridgeSensableLinks;
    LwU32  bridgedLinks;
    LwU32  initDisabledLinksMask;
    LwU32  vbiosDisabledLinkMask;
    LwU32  initializedLinks;
    LwBool bEnableTrainingAtLoad;
    LwBool bEnableSafeModeAtLoad;
} LW0080_CTRL_LWLINK_SYNC_LINK_MASKS_AND_VBIOS_INFO_PARAMS;

/* _ctrl0080lwlink_h_ */
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

