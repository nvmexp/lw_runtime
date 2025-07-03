/*
 * _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2022 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */
//
// This file should NEVER be published.
//
#pragma once

//
// This file was generated with FINN, an LWPU coding tool.
// Source file: ctrl/ctrl2080/ctrl2080clkavfs_opaque_privileged.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

#include "lwfixedtypes.h"
#include "ctrl/ctrl2080/ctrl2080base.h"
#include "ctrl/ctrl2080/ctrl2080clk.h"
#include "lwmisc.h"

/*!
 * LW2080_CTRL_CMD_CLK_ADC_DEVICES_SET_CONTROL
 *
 * This command is used to set all the client-specified information for
 * all the ADC devices.
 *
 * Possible status values returned are:
 */
#define LW2080_CTRL_CMD_CLK_ADC_DEVICES_SET_CONTROL   (0x2080d0a3U) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_CLK_LEGACY_PRIVILEGED_INTERFACE_ID << 8) | 0xA3" */

/*!
 * LW2080_CTRL_CMD_CLK_NAFLL_DEVICES_SET_CONTROL
 *
 * This command accepts client-specified control parameters for a set
 * of NAFLL_DEVICES entries in the NAFLL Device Table, and applies
 * these new parameters to the set of NAFLL_DEVICES entries.
 *
 * See @ref LW2080_CTRL_CMD_CLK_NAFLL_DEVICES_CONTROL_PARAMS for
 * documentation on the parameters.
 *
 * Return values are specified per @ref BoardObjGrpSetControl.
 */
#define LW2080_CTRL_CMD_CLK_NAFLL_DEVICES_SET_CONTROL (0x2080d0b3U) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_CLK_LEGACY_PRIVILEGED_INTERFACE_ID << 8) | 0xB3" */


#endif // (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

