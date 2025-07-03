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
// Source file: ctrl/ctrl5070/ctrl5070common.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)



#define LW5070_CTRL_CMD_CHANNEL_STATE_IDLE                              LWBIT(0)
#define LW5070_CTRL_CMD_CHANNEL_STATE_WRTIDLE                           LWBIT(1)
#define LW5070_CTRL_CMD_CHANNEL_STATE_QUIESCENT1                        LWBIT(2)
#define LW5070_CTRL_CMD_CHANNEL_STATE_QUIESCENT2                        LWBIT(3)
#define LW5070_CTRL_CMD_CHANNEL_STATE_EMPTY                             LWBIT(4)
#define LW5070_CTRL_CMD_CHANNEL_STATE_FLUSHED                           LWBIT(5)
#define LW5070_CTRL_CMD_CHANNEL_STATE_BUSY                              LWBIT(6)
#define LW5070_CTRL_CMD_CHANNEL_STATE_DEALLOC                           LWBIT(7)
#define LW5070_CTRL_CMD_CHANNEL_STATE_DEALLOC_LIMBO                     LWBIT(8)
#define LW5070_CTRL_CMD_CHANNEL_STATE_LIMBO1                            LWBIT(9)
#define LW5070_CTRL_CMD_CHANNEL_STATE_LIMBO2                            LWBIT(10)
#define LW5070_CTRL_CMD_CHANNEL_STATE_FCODEINIT                         LWBIT(11)
#define LW5070_CTRL_CMD_CHANNEL_STATE_FCODE                             LWBIT(12)
#define LW5070_CTRL_CMD_CHANNEL_STATE_VBIOSINIT                         LWBIT(13)
#define LW5070_CTRL_CMD_CHANNEL_STATE_VBIOSOPER                         LWBIT(14)
#define LW5070_CTRL_CMD_CHANNEL_STATE_UNCONNECTED                       LWBIT(15)
#define LW5070_CTRL_CMD_CHANNEL_STATE_INITIALIZE                        LWBIT(16)
#define LW5070_CTRL_CMD_CHANNEL_STATE_SHUTDOWN1                         LWBIT(17)
#define LW5070_CTRL_CMD_CHANNEL_STATE_SHUTDOWN2                         LWBIT(18)
#define LW5070_CTRL_CMD_CHANNEL_STATE_INIT                              LWBIT(19)

#define LW5070_CTRL_CMD_MAX_HEADS                       4U
#define LW5070_CTRL_CMD_MAX_DACS                        4U
#define LW5070_CTRL_CMD_MAX_SORS                        8U
#define LW5070_CTRL_CMD_MAX_PIORS                       4U
#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)

// TODO: Remove this define after MODS cleanup as WBOR and related code are removed from RM source.
#define LW5070_CTRL_CMD_MAX_WBORS                       4U
/* LWRM_PUBLISHED_PENDING_IP_REVIEW */
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)



#define LW5070_CTRL_CMD_OR_OWNER_NONE                   (0xFFFFFFFFU)
#define LW5070_CTRL_CMD_OR_OWNER_HEAD(i)                                (i)
#define LW5070_CTRL_CMD_OR_OWNER_HEAD__SIZE_1           LW5070_CTRL_CMD_MAX_HEADS

#define LW5070_CTRL_CMD_SOR_OWNER_MASK_NONE             (0x00000000U)
#define LW5070_CTRL_CMD_SOR_OWNER_MASK_HEAD(i)                          (1 << i)

#define LW5070_CTRL_CMD_DAC_PROTOCOL_RGB_CRT            (0x00000000U)

#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

#define LW5070_CTRL_CMD_DAC_PROTOCOL_CPST_NTSC_M        (0x00000001U)
#define LW5070_CTRL_CMD_DAC_PROTOCOL_CPST_NTSC_J        (0x00000002U)
#define LW5070_CTRL_CMD_DAC_PROTOCOL_CPST_PAL_BDGHI     (0x00000003U)
#define LW5070_CTRL_CMD_DAC_PROTOCOL_CPST_PAL_M         (0x00000004U)
#define LW5070_CTRL_CMD_DAC_PROTOCOL_CPST_PAL_N         (0x00000005U)
#define LW5070_CTRL_CMD_DAC_PROTOCOL_CPST_PAL_CN        (0x00000006U)
#define LW5070_CTRL_CMD_DAC_PROTOCOL_COMP_NTSC_M        (0x00000007U)
#define LW5070_CTRL_CMD_DAC_PROTOCOL_COMP_NTSC_J        (0x00000008U)
#define LW5070_CTRL_CMD_DAC_PROTOCOL_COMP_PAL_BDGHI     (0x00000009U)
#define LW5070_CTRL_CMD_DAC_PROTOCOL_COMP_PAL_M         (0x0000000AU)
#define LW5070_CTRL_CMD_DAC_PROTOCOL_COMP_PAL_N         (0x0000000BU)
#define LW5070_CTRL_CMD_DAC_PROTOCOL_COMP_PAL_CN        (0x0000000LW)
#define LW5070_CTRL_CMD_DAC_PROTOCOL_COMP_480P_60       (0x0000000DU)
#define LW5070_CTRL_CMD_DAC_PROTOCOL_COMP_576P_50       (0x0000000EU)
#define LW5070_CTRL_CMD_DAC_PROTOCOL_COMP_720P_50       (0x0000000FU)
#define LW5070_CTRL_CMD_DAC_PROTOCOL_COMP_720P_60       (0x00000010U)
#define LW5070_CTRL_CMD_DAC_PROTOCOL_COMP_1080I_50      (0x00000011U)
#define LW5070_CTRL_CMD_DAC_PROTOCOL_COMP_1080I_60      (0x00000012U)
#define LW5070_CTRL_CMD_DAC_PROTOCOL_LWSTOM             (0x0000003FU)

/* LWRM_UNPUBLISHED */
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)



#define LW5070_CTRL_CMD_SOR_PROTOCOL_SINGLE_TMDS_A      (0x00000000U)
#define LW5070_CTRL_CMD_SOR_PROTOCOL_SINGLE_TMDS_B      (0x00000001U)
#define LW5070_CTRL_CMD_SOR_PROTOCOL_DUAL_TMDS          (0x00000002U)
#define LW5070_CTRL_CMD_SOR_PROTOCOL_LVDS_LWSTOM        (0x00000003U)
#define LW5070_CTRL_CMD_SOR_PROTOCOL_DP_A               (0x00000004U)
#define LW5070_CTRL_CMD_SOR_PROTOCOL_DP_B               (0x00000005U)
#define LW5070_CTRL_CMD_SOR_PROTOCOL_SUPPORTED          (0xFFFFFFFFU)

#define LW5070_CTRL_CMD_PIOR_PROTOCOL_EXT_TMDS_ENC      (0x00000000U)

#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

#define LW5070_CTRL_CMD_PIOR_PROTOCOL_DIST_RENDER_OUT   (0x00000003U)
#define LW5070_CTRL_CMD_PIOR_PROTOCOL_DIST_RENDER_IN    (0x00000004U)
#define LW5070_CTRL_CMD_PIOR_PROTOCOL_DIST_RENDER_INOUT (0x00000005U)
#define LW5070_CTRL_CMD_PIOR_PROTOCOL_EXT_LVDS_ENC      (0x00000007U)
#define LW5070_CTRL_CMD_PIOR_PROTOCOL_EXT_CRT_ENC       (0x00000008U)
#define LW5070_CTRL_CMD_PIOR_PROTOCOL_EXT_DP_ENC        (0x00000009U)
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

