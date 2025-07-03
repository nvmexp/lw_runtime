/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2022 LWPU CORPORATION & AFFILIATES. All rights reserved.
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
// Source file: ctrl/ctrl2080/ctrl2080base.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)




#include "ctrl/ctrlxxxx.h"
/* LW20_SUBDEVICE_XX control commands and parameters */

#define LW2080_CTRL_CMD(cat,idx)                LWXXXX_CTRL_CMD(0x2080, LW2080_CTRL_##cat, idx)

#define LWxxxx_CTRL_OPAQUE_PRIVILEGED          (0xC0)
#define LWxxxx_CTRL_OPAQUE_NON_PRIVILEGED      (0x80)

/* Subdevice command categories (6bits) */
#define LW2080_CTRL_RESERVED                   (0x00)
#define LW2080_CTRL_GPU                        (0x01)
#define LW2080_CTRL_GPU_LEGACY_NON_PRIVILEGED  (0x81) /* finn: Evaluated from "(LW2080_CTRL_GPU | LWxxxx_CTRL_OPAQUE_NON_PRIVILEGED)" */
#define LW2080_CTRL_FUSE                       (0x02)
#define LW2080_CTRL_EVENT                      (0x03)
#define LW2080_CTRL_TIMER                      (0x04)
#define LW2080_CTRL_THERMAL                    (0x05)
#define LW2080_CTRL_I2C                        (0x06)
#define LW2080_CTRL_EXTI2C                     (0x07)
#define LW2080_CTRL_BIOS                       (0x08)
#define LW2080_CTRL_CIPHER                     (0x09)
#define LW2080_CTRL_INTERNAL                   (0x0A)
#define LW2080_CTRL_CLK_LEGACY_PRIVILEGED      (0xd0) /* finn: Evaluated from "(LW2080_CTRL_CLK | LWxxxx_CTRL_OPAQUE_PRIVILEGED)" */
#define LW2080_CTRL_CLK_LEGACY_NON_PRIVILEGED  (0x90) /* finn: Evaluated from "(LW2080_CTRL_CLK | LWxxxx_CTRL_OPAQUE_NON_PRIVILEGED)" */
#define LW2080_CTRL_CLK                        (0x10)
#define LW2080_CTRL_FIFO                       (0x11)
#define LW2080_CTRL_GR                         (0x12)
#define LW2080_CTRL_FB                         (0x13)
#define LW2080_CTRL_MC                         (0x17)
#define LW2080_CTRL_BUS                        (0x18)
#define LW2080_CTRL_PERF_LEGACY_PRIVILEGED     (0xe0) /* finn: Evaluated from "(LW2080_CTRL_PERF | LWxxxx_CTRL_OPAQUE_PRIVILEGED)" */
#define LW2080_CTRL_PERF_LEGACY_NON_PRIVILEGED (0xa0) /* finn: Evaluated from "(LW2080_CTRL_PERF | LWxxxx_CTRL_OPAQUE_NON_PRIVILEGED)" */
#define LW2080_CTRL_PERF                       (0x20)
#define LW2080_CTRL_LWIF                       (0x21)
#define LW2080_CTRL_RC                         (0x22)
#define LW2080_CTRL_GPIO                       (0x23)
#define LW2080_CTRL_LWD                        (0x24)
#define LW2080_CTRL_DMA                        (0x25)
#define LW2080_CTRL_PMGR                       (0x26)
#define LW2080_CTRL_PMGR_LEGACY_NON_PRIVILEGED (0xa6) /* finn: Evaluated from "(LW2080_CTRL_PMGR | LWxxxx_CTRL_OPAQUE_NON_PRIVILEGED)" */
#define LW2080_CTRL_POWER                      (0x27)
#define LW2080_CTRL_LPWR                       (0x28)
#define LW2080_CTRL_ACR                        (0x29)
#define LW2080_CTRL_CE                         (0x2A)
#define LW2080_CTRL_SPI                        (0x2B)
#define LW2080_CTRL_LWLINK                     (0x30)
#define LW2080_CTRL_FLCN                       (0x31)
#define LW2080_CTRL_VOLT                       (0x32)
#define LW2080_CTRL_VOLT_LEGACY_PRIVILEGED     (0xf2) /* finn: Evaluated from "(LW2080_CTRL_VOLT | LWxxxx_CTRL_OPAQUE_PRIVILEGED)" */
#define LW2080_CTRL_VOLT_LEGACY_NON_PRIVILEGED (0xb2) /* finn: Evaluated from "(LW2080_CTRL_VOLT | LWxxxx_CTRL_OPAQUE_NON_PRIVILEGED)" */
#define LW2080_CTRL_FAS                        (0x33)
#define LW2080_CTRL_ECC                        (0x34)
#define LW2080_CTRL_FLA                        (0x35)
#define LW2080_CTRL_GSP                        (0x36)
#define LW2080_CTRL_NNE                        (0x37)
#define LW2080_CTRL_GRMGR                      (0x38)
#define LW2080_CTRL_UCODE_FUZZER               (0x39)
#define LW2080_CTRL_DMABUF                     (0x3A)

// per-OS categories start at highest category and work backwards
#define LW2080_CTRL_OS_WINDOWS                 (0x3F)
#define LW2080_CTRL_OS_MACOS                   (0x3E)
#define LW2080_CTRL_OS_UNIX                    (0x3D)


/*
 * LW2080_CTRL_CMD_NULL
 *
 * This command does nothing.
 * This command does not take any parameters.
 *
 * Possible status values returned are:
 *     LW_OK
 */
#define LW2080_CTRL_CMD_NULL                   (0x20800000) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_RESERVED_INTERFACE_ID << 8) | 0x0" */

/* _ctrl2080base_h_ */
