/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2022 LWPU CORPORATION & AFFILIATES. All rights reserved.
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
// Source file: ctrl/ctrl2080/ctrl2080illum.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)



#include "ctrl/ctrl2080/ctrl2080base.h"
#include "ctrl/ctrl2080/ctrl2080boardobj.h"

/* --------------------------------- COMMON --------------------------------- */
/*!
 * List of control modes that can be applied to an Illumination Zone.
 */
#define LW2080_CTRL_ILLUM_CTRL_MODE_MANUAL_RGB                         0x00
#define LW2080_CTRL_ILLUM_CTRL_MODE_PIECEWISE_LINEAR_RGB               0x01
#define LW2080_CTRL_ILLUM_CTRL_MODE_ILWALID                            0xFF

/*!
 * Number of color points for the piecewise linear control mode.
 */
#define LW2080_CTRL_ILLUM_CTRL_MODE_PIECEWISE_LINEAR_COLOR_ENDPOINTS   2

#define LW2080_CTRL_ILLUM_CTRL_MODE_PIECEWISE_LINEAR_CYCLE_HALF_HALT   0
#define LW2080_CTRL_ILLUM_CTRL_MODE_PIECEWISE_LINEAR_CYCLE_FULL_HALT   1
#define LW2080_CTRL_ILLUM_CTRL_MODE_PIECEWISE_LINEAR_CYCLE_FULL_REPEAT 2
#define LW2080_CTRL_ILLUM_CTRL_MODE_PIECEWISE_LINEAR_CYCLE_ILWALID     0xFF

/*!
 * Enumeration of locations where an Illumination Zone might be present.
 * Encoding used -
 *   1:0 - Number specifier (0)
 *   4:2 - Location (TOP)
 *         0 - TOP
 *         1 - BOTTOM
 *         2 - FRONT
 *         3 - BACK
 *   7:5 - Type (GPU/SLI)
 *         0 - GPU
 *         1 - SLI
 */
typedef enum LW2080_CTRL_ILLUM_ZONE_LOCATION {
    LW2080_CTRL_ILLUM_ZONE_LOCATION_GPU_TOP_0 = 0,
    LW2080_CTRL_ILLUM_ZONE_LOCATION_GPU_FRONT_0 = 8,
    LW2080_CTRL_ILLUM_ZONE_LOCATION_GPU_BACK_0 = 12,
    LW2080_CTRL_ILLUM_ZONE_LOCATION_SLI_TOP_0 = 32,
} LW2080_CTRL_ILLUM_ZONE_LOCATION;


/* _ctrl2080illum_h_ */

#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)

#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1) && (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

#include "ctrl/ctrl2080/ctrl2080illum_opaque_non_privileged.h"
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1) && (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

