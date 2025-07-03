/*
 * SPDX-FileCopyrightText: Copyright (c) 2015-2023 LWPU CORPORATION & AFFILIATES. All rights reserved.
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

/*
 * This file should NEVER be published as it contains opaque privileged control
 * commands and parameters for Volt module. 
 */

#pragma once

//
// This file was generated with FINN, an LWPU coding tool.
// Source file: ctrl/ctrl2080/ctrl2080volt_opaque_privileged.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)


/* -- VOLT_RAIL's SET_CONTROL RMCTRL defines and structures -- */

/*!
 * LW2080_CTRL_CMD_VOLT_VOLT_RAILS_SET_CONTROL
 *
 * This command accepts client-specified control parameters for a set
 * of VOLT_RAIL entries and applies these new parameters to the set of
 * VOLT_RAIL entries
 *
 * See LW2080_CTRL_VOLT_VOLT_RAIL_CONTROL_PARAMS for documentation on the
 * parameters.
 *
 * Possible status values returned are
 *   LWOS_STATUS_SUCCESS
 *   LWOS_STATUS_ERROR_ILWALID_ARGUMENT
 */
#define LW2080_CTRL_CMD_VOLT_VOLT_RAILS_SET_CONTROL    (0x2080f214) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_VOLT_LEGACY_PRIVILEGED_INTERFACE_ID << 8) | 0x14" */

/* --------- VOLT_DEVICE's SET_CONTROLS defines and structures --------- */

/*!
 * LW2080_CTRL_CMD_VOLT_VOLT_DEVICES_SET_CONTROL
 *
 * This command accepts client-specified control parameters for a set
 * of VOLT_DEVICE entries and applies these new parameters to the set of
 * VOLT_DEVICE entries.
 *
 * See LW2080_CTRL_VOLT_VOLT_DEVICES_CONTROL_PARAMS for documentation on the parameters.
 *
 * Possible status values returned are
 *   LWOS_STATUS_SUCCESS
 *   LWOS_STATUS_ERROR_ILWALID_ARGUMENT
 */
#define LW2080_CTRL_CMD_VOLT_VOLT_DEVICES_SET_CONTROL  (0x2080f208) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_VOLT_LEGACY_PRIVILEGED_INTERFACE_ID << 8) | 0x8" */

/* -- VOLT_POLICY's SET_CONTROL RMCTRL defines and structures -- */

/*!
 * LW2080_CTRL_CMD_VOLT_VOLT_POLICIES_SET_CONTROL
 *
 * This command accepts client-specified control parameters for a set
 * of VOLT_POLICY entries in the Volt Policy Vbios table and applies these
 * new parameters to the set of VOLT_POLICY entries
 *
 * See LW2080_CTRL_VOLT_VOLT_POLICY_CONTROL_PARAMS for documentation on the parameters.
 *
 * Possible status values returned are
 *   LWOS_STATUS_SUCCESS
 *   LWOS_STATUS_ERROR_ILWALID_ARGUMENT
 */
#define LW2080_CTRL_CMD_VOLT_VOLT_POLICIES_SET_CONTROL (0x2080f212) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_VOLT_LEGACY_PRIVILEGED_INTERFACE_ID << 8) | 0x12" */

/* _ctrl2080volt_opaque_privileged_h_ */

#endif // (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

