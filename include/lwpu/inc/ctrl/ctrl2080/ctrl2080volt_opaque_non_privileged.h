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
 * This file should NEVER be published as it contains opaque non privileged
 * control commands and parameters for Volt module. 
 */

#pragma once

#include <lwtypes.h>
#if defined(_MSC_VER)
#pragma warning(disable:4324)
#endif

//
// This file was generated with FINN, an LWPU coding tool.
// Source file: ctrl/ctrl2080/ctrl2080volt_opaque_non_privileged.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

#include "ctrl2080volt.h"
#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)


/* ---------- VOLT_RAIL's GET_INFO RMCTRL defines and structures ---------- */

/*!
 * Structure representing the VOLT RAIL RAM ASSIST INFO PARAMETERS
 */
typedef struct LW2080_CTRL_VOLT_VOLT_RAIL_RAM_ASSIST_INFO {
    /*!
     * Ram Assist Type of the rail.
     */
    LwU8                         type;

    /*!
     * VFE Equation Index of the entry that specifies the Vcrit Low voltage
     * for engaging ram assist cirlwitory.
     */
#endif // (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1) && (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
    LW2080_CTRL_PERF_VFE_EQU_IDX vCritLowVfeEquIdx;
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1) && (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

    /*!
     * VFE Equation Index of the entry that specifies the Vcrit High voltage
     * for disengaging ram assist cirlwitory.
     */
#endif // (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1) && (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
    LW2080_CTRL_PERF_VFE_EQU_IDX vCritHighVfeEquIdx;
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1) && (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
} LW2080_CTRL_VOLT_VOLT_RAIL_RAM_ASSIST_INFO;

/*!
 * Structure representing the static information associated with a VOLT_RAIL
 */
typedef struct LW2080_CTRL_VOLT_VOLT_RAIL_INFO {
    /*!
     * LW2080_CTRL_BOARDOBJ super class.  Must always be first object in
     * structure.
     */
#endif // (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1) && (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
    LW2080_CTRL_BOARDOBJ                       super;
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1) && (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

    /*!
     * HW Strap Logical Boot Voltage for the rail.
     */
    LwU32                                      bootVoltageuV;

    /*!
     * VFE Equation Index of the entry that specifies the default maximum
     * reliability limit of the silicon.
     */
#endif // (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1) && (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
    LW2080_CTRL_PERF_VFE_EQU_IDX               relLimitVfeEquIdx;
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1) && (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

    /*!
     * VFE Equation Index of the entry that specifies the alternate maximum
     * reliability limit of the silicon.
     */
#endif // (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1) && (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
    LW2080_CTRL_PERF_VFE_EQU_IDX               altRelLimitVfeEquIdx;
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1) && (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

    /*!
     * VFE Equation Index of the entry that specifies the maximum over-voltage
     * limit of the silicon.
     */
#endif // (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1) && (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
    LW2080_CTRL_PERF_VFE_EQU_IDX               ovLimitVfeEquIdx;
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1) && (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

    /*!
     * Power Equation table index for evaluating this rail's leakage power/current.
     */
    LwU8                                       leakagePwrEquIdx;

    /*!
     * Default VOLTAGE_DEVICE for the rail.
     */
    LwU8                                       voltDevIdxDefault;

    /*!
     * IPC VMIN VOLTAGE_DEVICE for the rail.
     */
    LwU8                                       voltDevIdxIPCVmin;

    /*!
     * VFE Equation Index of the entry that specifies the boot voltage.
     */
#endif // (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1) && (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
    LW2080_CTRL_PERF_VFE_EQU_IDX               bootVoltVfeEquIdx;
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1) && (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

    /*!
     * VFE Equation Index of the entry that specifies the Vmin voltage.
     */
#endif // (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1) && (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
    LW2080_CTRL_PERF_VFE_EQU_IDX               vminLimitVfeEquIdx;
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1) && (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

    /*!
     *  VFIELD ID @ref VFIELD_ID_<xyz> that specifies the boot voltage.
     */
    LwU8                                       bootVoltVfieldId;

    /*!
     * VFE Equation Index of the entry that specifies the worst case
     * voltage margin.
     */
#endif // (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1) && (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
    LW2080_CTRL_PERF_VFE_EQU_IDX               voltMarginLimitVfeEquIdx;
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1) && (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

    /*!
     * Power Equation table index for evaluating this rail's dynamic power/current.
     */
    LwU8                                       dynamicPwrEquIdx;

    /*!
     * Mask of ADC_DEVICEs for obtaining sensed voltage of the rail.
     */
#endif // (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1) && (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
    LW2080_CTRL_BOARDOBJGRP_MASK_E32           adcDevMask;
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1) && (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

    /*!
     * Mask of VOLTAGE_DEVICEs for the rail.
     */
#endif // (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1) && (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
    LW2080_CTRL_BOARDOBJGRP_MASK_E32           voltDevMask;
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1) && (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

    /*!
     * Mask of all CLK_DOMAINs which implement the CLK_DOMAIN_PROG
     * interface and have a Vmin on the VOLT_RAIL.
     */
#endif // (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1) && (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
    LW2080_CTRL_BOARDOBJGRP_MASK_E32           clkDomainsProgMask;
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1) && (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

    /*!
     *  Cached value of VBIOS boot voltage in uV.
     */
    LwU32                                      vbiosBootVoltageuV;

    /*!
     *  Volt Rail RAM Assist static info params.
     */
    LW2080_CTRL_VOLT_VOLT_RAIL_RAM_ASSIST_INFO ramAssist;


    /*!
     * Power Equation table index for BA Scaling.
     */
    LwU8                                       baScalingPwrEqnIdx;
} LW2080_CTRL_VOLT_VOLT_RAIL_INFO;
typedef struct LW2080_CTRL_VOLT_VOLT_RAIL_INFO *PLW2080_CTRL_VOLT_VOLT_RAIL_INFO;

/*!
 * Structure representing the static state information associated with the GPU's
 * VOLT_RAIL volt rail functionality.
 */
#define LW2080_CTRL_VOLT_VOLT_RAILS_INFO_PARAMS_MESSAGE_ID (0x1U)

typedef struct LW2080_CTRL_VOLT_VOLT_RAILS_INFO_PARAMS {
    /*!
     * LW2080_CTRL_BOARDOBJGRP super class.  Must always be first object in
     * structure.
     */
#endif // (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1) && (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
    LW2080_CTRL_BOARDOBJGRP_E32     super;
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1) && (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

    /*
     * The voltage domain HAL type specifies the list of enumerants to use when
     * interpreting the rail entries.
     */
    LwU8                            voltDomainHAL;

    /*!
     * [out] - Array of VOLT_RAIL entries. Has valid indexes corresponding to
     * the bits set in @ref railMask.
     */
    LW2080_CTRL_VOLT_VOLT_RAIL_INFO rails[LW2080_CTRL_VOLT_VOLT_RAIL_MAX_RAILS];
} LW2080_CTRL_VOLT_VOLT_RAILS_INFO_PARAMS;
typedef struct LW2080_CTRL_VOLT_VOLT_RAILS_INFO_PARAMS *PLW2080_CTRL_VOLT_VOLT_RAILS_INFO_PARAMS;

/*!
 * LW2080_CTRL_CMD_VOLT_VOLT_RAILS_GET_INFO
 *
 * This command returns the VOLT_RAIL static information as specified by the
 * Voltage Rail Table.
 *
 * See LW2080_CTRL_VOLT_VOLT_RAILS_INFO_PARAMS for documentation on the parameters.
 *
 * Possible status values returned are
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW2080_CTRL_CMD_VOLT_VOLT_RAILS_GET_INFO (0x2080b201) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_VOLT_LEGACY_NON_PRIVILEGED_INTERFACE_ID << 8) | LW2080_CTRL_VOLT_VOLT_RAILS_INFO_PARAMS_MESSAGE_ID" */

/* --------- VOLT_RAIL's GET_STATUS RMCTRL defines and structures --------- */

/*!
 * Structure representing the VOLT RAIL RAM ASSIST STATUS PARAMETERS
 */
typedef struct LW2080_CTRL_VOLT_VOLT_RAIL_RAM_ASSIST_STATUS {
    /*!
     * Vcrit High voltage for disengaging ram assist cirlwitory.
     */
    LwU32  vCritHighuV;

    /*!
     * Vcrit Low voltage for engaging ram assist cirlwitory.
     */
    LwU32  vCritLowuV;

    /*!
     * Mask indicating whether RAM assist is engaged. 
     */
    LwU32  engagedMask;

    /*!
     * Flag indicating whether RAM Assist control is enabled. 
     */
    LwBool bEnabled;
} LW2080_CTRL_VOLT_VOLT_RAIL_RAM_ASSIST_STATUS;

/*!
 * Structure representing the dynamic state associated with a VOLT_RAIL.
 */
typedef struct LW2080_CTRL_VOLT_VOLT_RAIL_STATUS {
    /*!
     * LW2080_CTRL_BOARDOBJ super class.  Must always be first object in
     * structure.
     */
#endif // (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1) && (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
    LW2080_CTRL_BOARDOBJ                         super;
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1) && (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

    /*
     * Current voltage of the default VOLT_DEVICE for the rail.
     */
    LwU32                                        lwrrVoltDefaultuV;

    /*
     * Cached value of default maximum reliability limit of the silicon.
     */
    LwU32                                        relLimituV;

    /*
     * Cached value of alternate maximum reliability limit of the silicon.
     */
    LwU32                                        altRelLimituV;

    /*
     * Cached value of maximum over-voltage limit of the silicon.
     */
    LwU32                                        ovLimituV;

    /*
     * Cached value of maximum voltage limit of the silicon.
     */
    LwU32                                        maxLimituV;

    /*
     * Cached value of minimum voltage limit of the silicon.
     */
    LwU32                                        vminLimituV;

    /*
     * Cached value of worst case voltage margin of the silicon.
     */
    LwS32                                        voltMarginLimituV;

    /*!
     * V_{min, noise-unaware} - The minimum voltage (uV) with respect to
     * noise-unaware constraints on this VOLT_RAIL.
     */
    LwU32                                        voltMinNoiseUnawareuV;

    /*!
     * Sensed voltage value corresponding to the ADC_DEVICE for the VOLT_RAIL.
     */
    LwU32                                        lwrrVoltSenseduV;

    /*!
     * This will denote which state the rail is lwrrently in as per LW2080_CTRL_VOLT_VOLT_RAIL_ACTION_<xyz>.
     */
    LwU8                                         railAction;

    /*!
     * Volt Rail RAM Assist Dynamic Status Params
     */
    LW2080_CTRL_VOLT_VOLT_RAIL_RAM_ASSIST_STATUS ramAssist;
} LW2080_CTRL_VOLT_VOLT_RAIL_STATUS;
typedef struct LW2080_CTRL_VOLT_VOLT_RAIL_STATUS *PLW2080_CTRL_VOLT_VOLT_RAIL_STATUS;

/*!
 * Structure representing the dynamic state associated with the GPU's
 * VOLT_RAIL functionality.
 */
#define LW2080_CTRL_VOLT_VOLT_RAILS_STATUS_PARAMS_MESSAGE_ID (0x2U)

typedef struct LW2080_CTRL_VOLT_VOLT_RAILS_STATUS_PARAMS {
    /*!
     * LW2080_CTRL_BOARDOBJGRP super class.  Must always be first object in
     * structure.
     */
#endif // (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1) && (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
    LW2080_CTRL_BOARDOBJGRP_E32       super;
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1) && (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

    /*!
     * External delta for reliability voltage limit. This value is
     * computed from the pctDelta parameter obtained from the user.
     */
    LwU32                             extRelDeltauV[LW2080_CTRL_VOLT_RAIL_VOLT_DELTA_MAX_ENTRIES];

    /*!
     * [out] - Array of VOLT_RAIL entries. Has valid indexes corresponding to
     * the bits set in @ref railMask.
     */
    LW2080_CTRL_VOLT_VOLT_RAIL_STATUS rails[LW2080_CTRL_VOLT_VOLT_RAIL_MAX_RAILS];
} LW2080_CTRL_VOLT_VOLT_RAILS_STATUS_PARAMS;
typedef struct LW2080_CTRL_VOLT_VOLT_RAILS_STATUS_PARAMS *PLW2080_CTRL_VOLT_VOLT_RAILS_STATUS_PARAMS;

/*!
 * LW2080_CTRL_CMD_VOLT_VOLT_RAILS_GET_STATUS
 *
 * This command returns the VOLT_RAIL dynamic state information associated by the
 * VOLT_RAIL functionality
 *
 * See LW2080_CTRL_VOLT_VOLT_RAILS_STATUS_PARAMS for documentation on the parameters.
 *
 * Possible status values returned are
 *   LWOS_STATUS_SUCCESS
 *   LWOS_STATUS_ERROR_ILWALID_ARGUMENT
 */
#define LW2080_CTRL_CMD_VOLT_VOLT_RAILS_GET_STATUS (0x2080b202) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_VOLT_LEGACY_NON_PRIVILEGED_INTERFACE_ID << 8) | LW2080_CTRL_VOLT_VOLT_RAILS_STATUS_PARAMS_MESSAGE_ID" */

/* -- VOLT_RAIL's GET_CONTROL/SET_CONTROL RMCTRL defines and structures -- */

/*!
 * Structure representing the control parameters associated with a VOLT_RAIL.
 */
typedef struct LW2080_CTRL_VOLT_VOLT_RAIL_CONTROL {
    /*!
     * LW2080_CTRL_BOARDOBJ super class.  Must always be first object in
     * structure.
     */
#endif // (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1) && (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
    LW2080_CTRL_BOARDOBJ super;
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1) && (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

    /*!
     * Array to store voltage delta for different voltage limits.
     */
    LwS32                voltDeltauV[LW2080_CTRL_VOLT_RAIL_VOLT_DELTA_MAX_ENTRIES];
} LW2080_CTRL_VOLT_VOLT_RAIL_CONTROL;
typedef struct LW2080_CTRL_VOLT_VOLT_RAIL_CONTROL *PLW2080_CTRL_VOLT_VOLT_RAIL_CONTROL;

/*!
 * Structure representing the control parameters associated with the GPU's
 * VOLT_RAIL functionality.
 */
typedef struct LW2080_CTRL_VOLT_VOLT_RAILS_CONTROL_PARAMS {
    /*!
     * LW2080_CTRL_BOARDOBJGRP_E32 super class. Must always be first object in
     * structure.
     */
#endif // (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1) && (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
    LW2080_CTRL_BOARDOBJGRP_E32        super;
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1) && (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

    /*!
     * Percentage voltage delta to be applied for over-volting purpose.
     */
    LwU8                               pctDelta;

    /*!
     * [out] - Array of VOLT_RAIL entries. Has valid indexes corresponding to
     * the bits set in @ref railMask.
     */
    LW2080_CTRL_VOLT_VOLT_RAIL_CONTROL rails[LW2080_CTRL_VOLT_VOLT_RAIL_MAX_RAILS];
} LW2080_CTRL_VOLT_VOLT_RAILS_CONTROL_PARAMS;
typedef struct LW2080_CTRL_VOLT_VOLT_RAILS_CONTROL_PARAMS *PLW2080_CTRL_VOLT_VOLT_RAILS_CONTROL_PARAMS;

/*!
 * LW2080_CTRL_CMD_VOLT_VOLT_RAILS_GET_CONTROL
 *
 * This command returns the VOLT_RAIL control parameters associated by the
 * VOLT_RAIL functionality
 *
 * See LW2080_CTRL_VOLT_VOLT_RAILS_CONTROL_PARAMS for documentation on the
 * parameters.
 *
 * Possible status values returned are
 *   LWOS_STATUS_SUCCESS
 *   LWOS_STATUS_ERROR_ILWALID_ARGUMENT
 */
#define LW2080_CTRL_CMD_VOLT_VOLT_RAILS_GET_CONTROL (0x2080b213) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_VOLT_LEGACY_NON_PRIVILEGED_INTERFACE_ID << 8) | 0x13" */

/* ---------- VOLT_DEVICE's GET_INFO RMCTRL defines and structures ---------- */

/*!
 *  Structure representing the static information associated with a
 *  VID VOLT_DEVICE
 */
typedef struct LW2080_CTRL_VOLT_VOLT_DEVICE_INFO_DATA_VID {
   /*!
     * VSEL mask.
     */
    LwU8  vselMask;

    /*!
     * Min VID.
     */
    LwU8  vidMin;

    /*!
     * Max VID.
     */
    LwU8  vidMax;

    /*!
     * Voltage base - in uV.
     */
    LwS32 voltageBaseuV;

    /*!
     * Voltage offset scale - in uV.
     */
    LwS32 voltageOffsetScaleuV;

    /*!
     * VID to VSEL mapping.
    */
    LwU8  vidToVselMapping[LW2080_CTRL_VOLT_VOLT_DEV_VID_VSEL_MAX_ENTRIES];

   /*!
    * GPIO pin array for describing the GPIO pin numbers for the VSEL GPIOs.
    */
    LwU8  gpioPin[LW2080_CTRL_VOLT_VOLT_DEV_VID_VSEL_MAX_ENTRIES];

    /*!
     * Vsel function table.
     */
    LwU8  vselFunctionTable[LW2080_CTRL_VOLT_VOLT_DEV_VID_VSEL_MAX_ENTRIES];
} LW2080_CTRL_VOLT_VOLT_DEVICE_INFO_DATA_VID;
typedef struct LW2080_CTRL_VOLT_VOLT_DEVICE_INFO_DATA_VID *PLW2080_CTRL_VOLT_VOLT_DEVICE_INFO_DATA_VID;

/*!
 * Structure representing the static information associated with a
 * PWM VOLT_DEVICE
 */
typedef struct LW2080_CTRL_VOLT_VOLT_DEVICE_INFO_DATA_PWM {
    /*!
     * Voltage base.
     */
    LwS32 voltageBaseuV;

    /*!
     * Voltage offset scale.
     */
    LwS32 voltageOffsetScaleuV;

    /*!
     * @ref LW2080_CTRL_PMGR_PMU_SOURCE_<xyz>.
     */
    LwU8  source;

    /*!
     * PWM period.
     */
    LwU32 rawPeriod;

    /*!
     * GPIO pin indicating voltage rail enablement.
     */
    LwU8  voltEnGpioPin;

    /*!
     * Gate state voltage in uV.
     */
    LwU32 gateVoltageuV;
} LW2080_CTRL_VOLT_VOLT_DEVICE_INFO_DATA_PWM;
typedef struct LW2080_CTRL_VOLT_VOLT_DEVICE_INFO_DATA_PWM *PLW2080_CTRL_VOLT_VOLT_DEVICE_INFO_DATA_PWM;

/*!
 * Union of type-specific static information data.
 */


/*!
 * Structure representing the static information associated with a VOLT_DEVICE
 */
typedef struct LW2080_CTRL_VOLT_VOLT_DEVICE_INFO {
    /*!
     * LW2080_CTRL_BOARDOBJ super class.  Must always be first object in
     * structure.
     */
#endif // (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1) && (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
    LW2080_CTRL_BOARDOBJ super;
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1) && (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

    /*!
     * XAPI does not support using an element within another structure as a
     * discriminant. Placing this redundant type value here until that design
     * constraint can be fixed.
     */
    LwU8                 type;

    /*!
     * Voltage Domain @ref LW2080_CTRL_VOLT_DOMAIN_<xyz> corresponding to the
     * VOLTAGE_DEVICE.
     */
    LwU8                 voltDomain;

    /*!
     * Index into the DCB I2C Devices table for device corresponding to the
     * VOLTAGE_DEVICE.  Invalid index (0xFF) means no corresponding I2C_DEVICE.
     */
    LwU8                 i2cDevIdx;

    /*!
     * Operation Type @ref LW2080_CTRL_VOLT_VOLT_DEVICE_OPERATION_TYPE_<xyz>
     * which indicates whether the VOLTAGE_DEVICE is operating in the default
     * or any of the LPWR modes.
     */
    LwU8                 operationType;

    /*!
     * Min voltage - in uV.
     */
    LwU32                voltageMinuV;

    /*!
     * Max voltage - in uV.
     */
    LwU32                voltageMaxuV;

    /*!
     * Voltage step value in uV.
     */
    LwU32                voltStepuV;

    /*!
     * Type-specific information.
     */
    union {
        LW2080_CTRL_VOLT_VOLT_DEVICE_INFO_DATA_VID vid;
        LW2080_CTRL_VOLT_VOLT_DEVICE_INFO_DATA_PWM pwm;
    } data;
} LW2080_CTRL_VOLT_VOLT_DEVICE_INFO;
typedef struct LW2080_CTRL_VOLT_VOLT_DEVICE_INFO *PLW2080_CTRL_VOLT_VOLT_DEVICE_INFO;

/*!
 * Structure representing the static state information associated with the GPU's
 * VOLT_DEVICE functionality.
 */
#define LW2080_CTRL_VOLT_VOLT_DEVICES_INFO_PARAMS_MESSAGE_ID (0x5U)

typedef struct LW2080_CTRL_VOLT_VOLT_DEVICES_INFO_PARAMS {
   /*!
     * LW2080_CTRL_BOARDOBJGRP super class.  Must always be first object in
     * structure.
     */
#endif // (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1) && (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
    LW2080_CTRL_BOARDOBJGRP_E32       super;
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1) && (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

    /*!
     * [out] - Array of VOLT_DEVICE entries. Has valid indexes corresponding to
     * the bits set in @ref deviceMask.
     */
    LW2080_CTRL_VOLT_VOLT_DEVICE_INFO devices[LW2080_CTRL_VOLT_VOLT_DEVICE_MAX_DEVICES];
} LW2080_CTRL_VOLT_VOLT_DEVICES_INFO_PARAMS;
typedef struct LW2080_CTRL_VOLT_VOLT_DEVICES_INFO_PARAMS *PLW2080_CTRL_VOLT_VOLT_DEVICES_INFO_PARAMS;

/*!
 * LW2080_CTRL_CMD_VOLT_VOLT_DEVICES_GET_INFO
 *
 * This command returns the VOLT_DEVICE static information as specified by the
 * Voltage Device Table.
 *
 * See LW2080_CTRL_VOLT_VOLT_DEVICES_INFO_PARAMS for documentation on the parameters.
 *
 * Possible status values returned are
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW2080_CTRL_CMD_VOLT_VOLT_DEVICES_GET_INFO (0x2080b205) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_VOLT_LEGACY_NON_PRIVILEGED_INTERFACE_ID << 8) | LW2080_CTRL_VOLT_VOLT_DEVICES_INFO_PARAMS_MESSAGE_ID" */

/* --------- VOLT_DEVICE's GET_CONTROL/SET_CONTROLS defines and structures --------- */

/*!
 *  Structure representing control parameters associated with a
 *  VID VOLT_DEVICE
 */
typedef struct LW2080_CTRL_VOLT_VOLT_DEVICE_CONTROL_DATA_VID {
    /*!
     * VOLT_DEVICE_VID does not contain any control parameters as of now.
     */
    LwU8 rsvd;
} LW2080_CTRL_VOLT_VOLT_DEVICE_CONTROL_DATA_VID;
typedef struct LW2080_CTRL_VOLT_VOLT_DEVICE_CONTROL_DATA_VID *PLW2080_CTRL_VOLT_VOLT_DEVICE_CONTROL_DATA_VID;

/*!
 * Structure representing control parameters associated with a
 * PWM VOLT_DEVICE
 */
typedef struct LW2080_CTRL_VOLT_VOLT_DEVICE_CONTROL_DATA_PWM {
    /*!
     * VOLT_DEVICE_PWM does not contain any control parameters as of now.
     */
    LwU8 rsvd;
} LW2080_CTRL_VOLT_VOLT_DEVICE_CONTROL_DATA_PWM;
typedef struct LW2080_CTRL_VOLT_VOLT_DEVICE_CONTROL_DATA_PWM *PLW2080_CTRL_VOLT_VOLT_DEVICE_CONTROL_DATA_PWM;

/*!
 * Union of type-specific control parameters.
 */


/*!
 * Structure representing control parameters associated with a VOLT_DEVICE
 */
typedef struct LW2080_CTRL_VOLT_VOLT_DEVICE_CONTROL {
    /*!
     * LW2080_CTRL_BOARDOBJ super class.  Must always be first object in
     * structure.
     */
#endif // (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1) && (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
    LW2080_CTRL_BOARDOBJ super;
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1) && (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

    /*!
     * XAPI does not support using an element within another structure as a
     * discriminant. Placing this redundant type value here until that design
     * constraint can be fixed.
     */
    LwU8                 type;

    /*!
     * Voltage Switch delay in us.
     */
    LwU32                switchDelayus;

    /*!
     * Type-specific information.
     */
    union {
        LW2080_CTRL_VOLT_VOLT_DEVICE_CONTROL_DATA_VID vid;
        LW2080_CTRL_VOLT_VOLT_DEVICE_CONTROL_DATA_PWM pwm;
    } data;
} LW2080_CTRL_VOLT_VOLT_DEVICE_CONTROL;
typedef struct LW2080_CTRL_VOLT_VOLT_DEVICE_CONTROL *PLW2080_CTRL_VOLT_VOLT_DEVICE_CONTROL;

/*!
 * Structure representing control parameters associated with the GPU's
 * VOLT_DEVICE functionality.
 */
typedef struct LW2080_CTRL_VOLT_VOLT_DEVICES_CONTROL_PARAMS {
   /*!
     * LW2080_CTRL_BOARDOBJGRP super class.  Must always be first object in
     * structure.
     */
#endif // (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1) && (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
    LW2080_CTRL_BOARDOBJGRP_E32          super;
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1) && (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

    /*!
     * [out] - Array of VOLT_DEVICE entries. Has valid indexes corresponding to
     * the bits set in @ref deviceMask.
     */
    LW2080_CTRL_VOLT_VOLT_DEVICE_CONTROL devices[LW2080_CTRL_VOLT_VOLT_DEVICE_MAX_DEVICES];
} LW2080_CTRL_VOLT_VOLT_DEVICES_CONTROL_PARAMS;
typedef struct LW2080_CTRL_VOLT_VOLT_DEVICES_CONTROL_PARAMS *PLW2080_CTRL_VOLT_VOLT_DEVICES_CONTROL_PARAMS;

/*!
 * LW2080_CTRL_CMD_VOLT_VOLT_DEVICES_GET_CONTROL
 *
 * This command returns the VOLT_DEVICE control parameters as specified by the
 * Voltage Device Table.
 *
 * See LW2080_CTRL_VOLT_VOLT_DEVICES_CONTROL_PARAMS for documentation on the parameters.
 *
 * Possible status values returned are
 *   LWOS_STATUS_SUCCESS
 *   LWOS_STATUS_ERROR_ILWALID_ARGUMENT
 */
#define LW2080_CTRL_CMD_VOLT_VOLT_DEVICES_GET_CONTROL (0x2080b207) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_VOLT_LEGACY_NON_PRIVILEGED_INTERFACE_ID << 8) | 0x7" */

/* ---------- VOLT_POLICY's GET_INFO RMCTRL defines and structures ---------- */

/*!
 *  Structure representing the static information associated with a
 *  SINGLE_RAIL VOLT_POLICY
 */
typedef struct LW2080_CTRL_VOLT_VOLT_POLICY_INFO_DATA_SINGLE_RAIL {
    /*!
     * Index into the Voltage Rail Table.
     */
    LwU32 railIdx;
} LW2080_CTRL_VOLT_VOLT_POLICY_INFO_DATA_SINGLE_RAIL;
typedef struct LW2080_CTRL_VOLT_VOLT_POLICY_INFO_DATA_SINGLE_RAIL *PLW2080_CTRL_VOLT_VOLT_POLICY_INFO_DATA_SINGLE_RAIL;

/*!
 *  Structure representing the static information associated with a
 *  SINGLE_RAIL_MULTI_STEP VOLT_POLICY
 */
typedef struct LW2080_CTRL_VOLT_VOLT_POLICY_INFO_DATA_SINGLE_RAIL_MULTI_STEP {
    /*!
     * VOLT_POLICY_SINGLE_RAIL super class. Must always be first element
     * in the structure.
     */
    LW2080_CTRL_VOLT_VOLT_POLICY_INFO_DATA_SINGLE_RAIL super;

    /*!
     * Ramp up step size in uV.
     * A value of 0 uV indicates that ramp isn't required.
     */
    LwU32                                              rampUpStepSizeuV;

    /*!
     * Ramp down step size in uV.
     * A value of 0 uV indicates that ramp isn't required.
     */
    LwU32                                              rampDownStepSizeuV;
} LW2080_CTRL_VOLT_VOLT_POLICY_INFO_DATA_SINGLE_RAIL_MULTI_STEP;
typedef struct LW2080_CTRL_VOLT_VOLT_POLICY_INFO_DATA_SINGLE_RAIL_MULTI_STEP *PLW2080_CTRL_VOLT_VOLT_POLICY_INFO_DATA_SINGLE_RAIL_MULTI_STEP;

/*!
 * Structure representing the static information associated with a
 * SPLIT_RAIL VOLT_POLICY
 */
typedef struct LW2080_CTRL_VOLT_VOLT_POLICY_INFO_DATA_SPLIT_RAIL {
    /*!
     * Index into the Voltage Rail Table for master rail.
     */
    LwU8                         railIdxMaster;

    /*!
     * Index into the Voltage Rail Table for slave rail.
     */
    LwU8                         railIdxSlave;

    /*!
     * VFE Equation Index for min voltage delta between slave and master rail.
     */
#endif // (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1) && (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
    LW2080_CTRL_PERF_VFE_EQU_IDX deltaMilwfeEquIdx;
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1) && (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

    /*!
     * VFE Equation Index for max voltage delta between slave and master rail.
     */
#endif // (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1) && (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
    LW2080_CTRL_PERF_VFE_EQU_IDX deltaMaxVfeEquIdx;
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1) && (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
} LW2080_CTRL_VOLT_VOLT_POLICY_INFO_DATA_SPLIT_RAIL;
typedef struct LW2080_CTRL_VOLT_VOLT_POLICY_INFO_DATA_SPLIT_RAIL *PLW2080_CTRL_VOLT_VOLT_POLICY_INFO_DATA_SPLIT_RAIL;

/*!
 * Structure representing the static information associated with a
 * SPLIT_RAIL_MULTI_STEP VOLT_POLICY
 */
typedef struct LW2080_CTRL_VOLT_VOLT_POLICY_INFO_DATA_SPLIT_RAIL_MULTI_STEP {
    /*!
     * VOLT_POLICY_SPLIT_RAIL super class. Must always be first element
     * in the structure.
     */
    LW2080_CTRL_VOLT_VOLT_POLICY_INFO_DATA_SPLIT_RAIL super;
} LW2080_CTRL_VOLT_VOLT_POLICY_INFO_DATA_SPLIT_RAIL_MULTI_STEP;
typedef struct LW2080_CTRL_VOLT_VOLT_POLICY_INFO_DATA_SPLIT_RAIL_MULTI_STEP *PLW2080_CTRL_VOLT_VOLT_POLICY_INFO_DATA_SPLIT_RAIL_MULTI_STEP;

/*!
 * Structure representing the static information associated with a
 * SPLIT_RAIL_SINGLE_STEP VOLT_POLICY
 */
typedef struct LW2080_CTRL_VOLT_VOLT_POLICY_INFO_DATA_SPLIT_RAIL_SINGLE_STEP {
    /*!
     * VOLT_POLICY_SPLIT_RAIL super class. Must always be first element
     * in the structure.
     */
    LW2080_CTRL_VOLT_VOLT_POLICY_INFO_DATA_SPLIT_RAIL super;
} LW2080_CTRL_VOLT_VOLT_POLICY_INFO_DATA_SPLIT_RAIL_SINGLE_STEP;
typedef struct LW2080_CTRL_VOLT_VOLT_POLICY_INFO_DATA_SPLIT_RAIL_SINGLE_STEP *PLW2080_CTRL_VOLT_VOLT_POLICY_INFO_DATA_SPLIT_RAIL_SINGLE_STEP;

/*!
 *  Structure representing the static information associated with a
 *  MULTI_RAIL VOLT_POLICY
 */
typedef struct LW2080_CTRL_VOLT_VOLT_POLICY_INFO_DATA_MULTI_RAIL {
    /*!
     * Mask of VOLT_RAILs for this policy.
     */
#endif // (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1) && (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
    LW2080_CTRL_BOARDOBJGRP_MASK_E32 voltRailMask;
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1) && (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
} LW2080_CTRL_VOLT_VOLT_POLICY_INFO_DATA_MULTI_RAIL;
typedef struct LW2080_CTRL_VOLT_VOLT_POLICY_INFO_DATA_MULTI_RAIL *PLW2080_CTRL_VOLT_VOLT_POLICY_INFO_DATA_MULTI_RAIL;

/*!
 * Union of type-specific static information data.
 */


/*!
 * Structure representing the static information associated with a VOLT_POLICY
 */
typedef struct LW2080_CTRL_VOLT_VOLT_POLICY_INFO {
    /*!
     * LW2080_CTRL_BOARDOBJ super class.  Must always be first object in
     * structure.
     */
#endif // (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1) && (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
    LW2080_CTRL_BOARDOBJ super;
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1) && (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

    /*!
     * XAPI does not support using an element within another structure as a
     * discriminant. Placing this redundant type value here until that design
     * constraint can be fixed.
     * @ref LW2080_CTRL_VOLT_VOLT_POLICY_TYPE_<xyz>.
     */
    LwU8                 type;

    /*!
     * Type-specific information.
     */
    union {
        LW2080_CTRL_VOLT_VOLT_POLICY_INFO_DATA_SINGLE_RAIL            singleRail;
        LW2080_CTRL_VOLT_VOLT_POLICY_INFO_DATA_SINGLE_RAIL_MULTI_STEP singleRailMS;
        LW2080_CTRL_VOLT_VOLT_POLICY_INFO_DATA_SPLIT_RAIL             splitRail;
        LW2080_CTRL_VOLT_VOLT_POLICY_INFO_DATA_SPLIT_RAIL_MULTI_STEP  splitRailMS;
        LW2080_CTRL_VOLT_VOLT_POLICY_INFO_DATA_SPLIT_RAIL_SINGLE_STEP splitRailSS;
        LW2080_CTRL_VOLT_VOLT_POLICY_INFO_DATA_MULTI_RAIL             multiRail;
    } data;
} LW2080_CTRL_VOLT_VOLT_POLICY_INFO;
typedef struct LW2080_CTRL_VOLT_VOLT_POLICY_INFO *PLW2080_CTRL_VOLT_VOLT_POLICY_INFO;

/*!
 * Structure representing the static state information associated with the GPU's
 * VOLT_POLICY volt policy functionality.
 */
#define LW2080_CTRL_VOLT_VOLT_POLICIES_INFO_PARAMS_MESSAGE_ID (0x9U)

typedef struct LW2080_CTRL_VOLT_VOLT_POLICIES_INFO_PARAMS {
    /*!
     * LW2080_CTRL_BOARDOBJGRP super class.  Must always be first object in
     * structure.
     */
#endif // (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1) && (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
    LW2080_CTRL_BOARDOBJGRP_E32       super;
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1) && (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

    /*!
     * Voltage Policy Table Index for Perf Core VF Sequence client.
     *
     * @note LW2080_CTRL_VOLT_VOLT_POLICY_INDEX_ILWALID indicates that
     * this policy is not present/specified.
     */
    LwU8                              perfCoreVFSeqPolicyIdx;

    /*!
     * [out] - Array of VOLT_POLICY entries. Has valid indexes corresponding to
     * the bits set in @ref policyMask.
     */
    LW2080_CTRL_VOLT_VOLT_POLICY_INFO policies[LW2080_CTRL_VOLT_VOLT_POLICY_MAX_POLICIES];
} LW2080_CTRL_VOLT_VOLT_POLICIES_INFO_PARAMS;
typedef struct LW2080_CTRL_VOLT_VOLT_POLICIES_INFO_PARAMS *PLW2080_CTRL_VOLT_VOLT_POLICIES_INFO_PARAMS;

/*!
 * LW2080_CTRL_CMD_VOLT_VOLT_POLICIES_GET_INFO
 *
 * This command returns the VOLT_POLICY static information as specified by the
 * Voltage Policy Table.
 *
 * See LW2080_CTRL_VOLT_VOLT_POLICIES_INFO_PARAMS for documentation on the parameters.
 *
 * Possible status values returned are
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW2080_CTRL_CMD_VOLT_VOLT_POLICIES_GET_INFO (0x2080b209) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_VOLT_LEGACY_NON_PRIVILEGED_INTERFACE_ID << 8) | LW2080_CTRL_VOLT_VOLT_POLICIES_INFO_PARAMS_MESSAGE_ID" */

/* --------- VOLT_POLICY's GET_STATUS RMCTRL defines and structures --------- */

/*!
 *  Structure representing the static state associated with a
 *  SINGLE_RAIL VOLT_POLICY
 */
typedef struct LW2080_CTRL_VOLT_VOLT_POLICY_STATUS_DATA_SINGLE_RAIL {
    /*!
     * Cached value of most recently requested voltage without applying
     * VOLT_POLICY::offsetVoltuV.
     */
    LwU32 lwrrVoltuV;
} LW2080_CTRL_VOLT_VOLT_POLICY_STATUS_DATA_SINGLE_RAIL;
typedef struct LW2080_CTRL_VOLT_VOLT_POLICY_STATUS_DATA_SINGLE_RAIL *PLW2080_CTRL_VOLT_VOLT_POLICY_STATUS_DATA_SINGLE_RAIL;

/*!
 * Structure representing the dynamic state associated with a
 * SINGLE_RAIL_MULTI_STEP VOLT_POLICY
 */
typedef struct LW2080_CTRL_VOLT_VOLT_POLICY_STATUS_DATA_SINGLE_RAIL_MULTI_STEP {
    /*!
     * VOLT_POLICY_SINGLE_RAIL super class. Must always be first element
     * in the structure.
     */
    LW2080_CTRL_VOLT_VOLT_POLICY_STATUS_DATA_SINGLE_RAIL super;
} LW2080_CTRL_VOLT_VOLT_POLICY_STATUS_DATA_SINGLE_RAIL_MULTI_STEP;
typedef struct LW2080_CTRL_VOLT_VOLT_POLICY_STATUS_DATA_SINGLE_RAIL_MULTI_STEP *PLW2080_CTRL_VOLT_VOLT_POLICY_STATUS_DATA_SINGLE_RAIL_MULTI_STEP;

/*!
 * Structure representing the dynamic state associated with a
 * SPLIT_RAIL VOLT_POLICY
 */
typedef struct LW2080_CTRL_VOLT_VOLT_POLICY_STATUS_DATA_SPLIT_RAIL {
    /*!
     * Min voltage delta between slave and master rail.
     */
    LwS32  deltaMinuV;

    /*!
     * Max voltage delta between slave and master rail.
     */
    LwS32  deltaMaxuV;

    /*!
     * Original min voltage delta between slave and master rail.
     */
    LwS32  origDeltaMinuV;

    /*!
     * Original max voltage delta between slave and master rail.
     */
    LwS32  origDeltaMaxuV;

    /*!
     * Cached value of most recently requested voltage for master rail without
     * applying VOLT_POLICY::offsetVoltuV.
     */
    LwU32  lwrrVoltMasteruV;

    /*!
     * Cached value of most recently requested voltage for slave rail without
     * applying VOLT_POLICY::offsetVoltuV.
     */
    LwU32  lwrrVoltSlaveuV;

    /*!
     * Boolean to indicate whether original delta constraints are violated.
     * The boolean is sticky and no code path clears it.
     */
    LwBool bViolation;
} LW2080_CTRL_VOLT_VOLT_POLICY_STATUS_DATA_SPLIT_RAIL;
typedef struct LW2080_CTRL_VOLT_VOLT_POLICY_STATUS_DATA_SPLIT_RAIL *PLW2080_CTRL_VOLT_VOLT_POLICY_STATUS_DATA_SPLIT_RAIL;

/*!
 * Structure representing the dynamic state associated with a
 * SPLIT_RAIL_MULTI_STEP VOLT_POLICY
 */
typedef struct LW2080_CTRL_VOLT_VOLT_POLICY_STATUS_DATA_SPLIT_RAIL_MULTI_STEP {
    /*!
     * VOLT_POLICY_SPLIT_RAIL super class. Must always be first element
     * in the structure.
     */
    LW2080_CTRL_VOLT_VOLT_POLICY_STATUS_DATA_SPLIT_RAIL super;
} LW2080_CTRL_VOLT_VOLT_POLICY_STATUS_DATA_SPLIT_RAIL_MULTI_STEP;
typedef struct LW2080_CTRL_VOLT_VOLT_POLICY_STATUS_DATA_SPLIT_RAIL_MULTI_STEP *PLW2080_CTRL_VOLT_VOLT_POLICY_STATUS_DATA_SPLIT_RAIL_MULTI_STEP;

/*!
 * Structure representing the dynamic state associated with a
 * SPLIT_RAIL_SINGLE_STEP VOLT_POLICY
 */
typedef struct LW2080_CTRL_VOLT_VOLT_POLICY_STATUS_DATA_SPLIT_RAIL_SINGLE_STEP {
    /*!
     * VOLT_POLICY_SPLIT_RAIL super class. Must always be first element
     * in the structure.
     */
    LW2080_CTRL_VOLT_VOLT_POLICY_STATUS_DATA_SPLIT_RAIL super;
} LW2080_CTRL_VOLT_VOLT_POLICY_STATUS_DATA_SPLIT_RAIL_SINGLE_STEP;
typedef struct LW2080_CTRL_VOLT_VOLT_POLICY_STATUS_DATA_SPLIT_RAIL_SINGLE_STEP *PLW2080_CTRL_VOLT_VOLT_POLICY_STATUS_DATA_SPLIT_RAIL_SINGLE_STEP;

/*!
 * Union of type-specific dynamic state data.
 */


/*!
 * Structure representing the dynamic state associated with a VOLT_POLICY.
 */
typedef struct LW2080_CTRL_VOLT_VOLT_POLICY_STATUS {
    /*!
     * LW2080_CTRL_BOARDOBJ super class.  Must always be first object in
     * structure.
     */
#endif // (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1) && (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
    LW2080_CTRL_BOARDOBJ super;
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1) && (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

    /*!
     * XAPI does not support using an element within another structure as a
     * discriminant. Placing this redundant type value here until that design
     * constraint can be fixed.
     */
    LwU8                 type;

    /*!
     * Cached value of requested voltage offset to be applied on one or more
     * VOLT_RAILs according to the appropriate VOLT_POLICY.
     */
    LwS32                offsetVoltRequV;

    /*!
     * Cached value of current voltage offset that is applied on one or more
     * VOLT_RAILs according to the appropriate VOLT_POLICY.
     */
    LwS32                offsetVoltLwrruV;

    /*!
     * Type-specific information.
     */
    union {
        LW2080_CTRL_VOLT_VOLT_POLICY_STATUS_DATA_SINGLE_RAIL            singleRail;
        LW2080_CTRL_VOLT_VOLT_POLICY_STATUS_DATA_SINGLE_RAIL_MULTI_STEP singleRailMS;
        LW2080_CTRL_VOLT_VOLT_POLICY_STATUS_DATA_SPLIT_RAIL             splitRail;
        LW2080_CTRL_VOLT_VOLT_POLICY_STATUS_DATA_SPLIT_RAIL_MULTI_STEP  splitRailMS;
        LW2080_CTRL_VOLT_VOLT_POLICY_STATUS_DATA_SPLIT_RAIL_SINGLE_STEP splitRailSS;
#endif // (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1) && (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
        LW2080_CTRL_VOLT_VOLT_POLICY_STATUS_DATA_MULTI_RAIL             multiRail;
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1) && (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
    } data;
} LW2080_CTRL_VOLT_VOLT_POLICY_STATUS;
typedef struct LW2080_CTRL_VOLT_VOLT_POLICY_STATUS *PLW2080_CTRL_VOLT_VOLT_POLICY_STATUS;

/*!
 * Structure representing the dynamic state associated with the GPU's
 * VOLT_POLICY functionality.
 */
#define LW2080_CTRL_VOLT_VOLT_POLICIES_STATUS_PARAMS_MESSAGE_ID (0x10U)

typedef struct LW2080_CTRL_VOLT_VOLT_POLICIES_STATUS_PARAMS {
    /*!
     * LW2080_CTRL_BOARDOBJGRP_E32 super class. Must always be first object in
     * structure.
     */
#endif // (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1) && (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
    LW2080_CTRL_BOARDOBJGRP_E32         super;
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1) && (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

    /*!
     * [out] - Array of VOLT_POLICY entries. Has valid indexes corresponding to
     * the bits set in @ref policyMask.
     */
    LW2080_CTRL_VOLT_VOLT_POLICY_STATUS policies[LW2080_CTRL_VOLT_VOLT_POLICY_MAX_POLICIES];
} LW2080_CTRL_VOLT_VOLT_POLICIES_STATUS_PARAMS;
typedef struct LW2080_CTRL_VOLT_VOLT_POLICIES_STATUS_PARAMS *PLW2080_CTRL_VOLT_VOLT_POLICIES_STATUS_PARAMS;

/*!
 * LW2080_CTRL_CMD_VOLT_VOLT_POLICIES_GET_STATUS
 *
 * This command returns the VOLT_POLICY dynamic state information associated by the
 * VOLT_POLICY functionality
 *
 * See LW2080_CTRL_VOLT_VOLT_POLICY_STATUS_PARAMS for documentation on the parameters.
 *
 * Possible status values returned are
 *   LWOS_STATUS_SUCCESS
 *   LWOS_STATUS_ERROR_ILWALID_ARGUMENT
 */
#define LW2080_CTRL_CMD_VOLT_VOLT_POLICIES_GET_STATUS (0x2080b210) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_VOLT_LEGACY_NON_PRIVILEGED_INTERFACE_ID << 8) | LW2080_CTRL_VOLT_VOLT_POLICIES_STATUS_PARAMS_MESSAGE_ID" */

/* -- VOLT_POLICY's GET_CONTROL/SET_CONTROL RMCTRL defines and structures -- */

/*!
 *  Structure representing the control parameters associated with a
 *  SINGLE_RAIL VOLT_POLICY
 */
typedef struct LW2080_CTRL_VOLT_VOLT_POLICY_CONTROL_DATA_SINGLE_RAIL {
    /*!
     * VOLT_POLICY_SINGLE_RAIL does not contain any control parameters as of now.
     */
    LwU8 rsvd;
} LW2080_CTRL_VOLT_VOLT_POLICY_CONTROL_DATA_SINGLE_RAIL;
typedef struct LW2080_CTRL_VOLT_VOLT_POLICY_CONTROL_DATA_SINGLE_RAIL *PLW2080_CTRL_VOLT_VOLT_POLICY_CONTROL_DATA_SINGLE_RAIL;

/*!
 *  Structure representing the control parameters associated with a
 *  SINGLE_RAIL_MULTI_STEP VOLT_POLICY
 */
typedef struct LW2080_CTRL_VOLT_VOLT_POLICY_CONTROL_DATA_SINGLE_RAIL_MULTI_STEP {
    /*!
     * VOLT_POLICY_SINGLE_RAIL super class. Must always be first element
     * in the structure.
     */
    LW2080_CTRL_VOLT_VOLT_POLICY_CONTROL_DATA_SINGLE_RAIL super;

    /*!
     * Settle time in microseconds for intermediate voltage switches.
     */
    LwU16                                                 interSwitchDelayus;
} LW2080_CTRL_VOLT_VOLT_POLICY_CONTROL_DATA_SINGLE_RAIL_MULTI_STEP;
typedef struct LW2080_CTRL_VOLT_VOLT_POLICY_CONTROL_DATA_SINGLE_RAIL_MULTI_STEP *PLW2080_CTRL_VOLT_VOLT_POLICY_CONTROL_DATA_SINGLE_RAIL_MULTI_STEP;

/*!
 * Structure representing the control parameters associated with a
 * SPLIT_RAIL VOLT_POLICY
 */
typedef struct LW2080_CTRL_VOLT_VOLT_POLICY_CONTROL_DATA_SPLIT_RAIL {
    /*!
     * Offset for min voltage delta between slave and master rail.
     */
    LwS32 offsetDeltaMinuV;

    /*!
     * Offset for max voltage delta between slave and master rail.
     */
    LwS32 offsetDeltaMaxuV;
} LW2080_CTRL_VOLT_VOLT_POLICY_CONTROL_DATA_SPLIT_RAIL;
typedef struct LW2080_CTRL_VOLT_VOLT_POLICY_CONTROL_DATA_SPLIT_RAIL *PLW2080_CTRL_VOLT_VOLT_POLICY_CONTROL_DATA_SPLIT_RAIL;

/*!
 * Structure representing the control parameters associated with a
 * SPLIT_RAIL_MULTI_STEP VOLT_POLICY
 */
typedef struct LW2080_CTRL_VOLT_VOLT_POLICY_CONTROL_DATA_SPLIT_RAIL_MULTI_STEP {
    /*!
     * VOLT_POLICY_SPLIT_RAIL super class. Must always be first element
     * in the structure.
     */
    LW2080_CTRL_VOLT_VOLT_POLICY_CONTROL_DATA_SPLIT_RAIL super;

    /*!
     * Settle time in microseconds for intermediate voltage switches.
     */
    LwU16                                                interSwitchDelayus;
} LW2080_CTRL_VOLT_VOLT_POLICY_CONTROL_DATA_SPLIT_RAIL_MULTI_STEP;
typedef struct LW2080_CTRL_VOLT_VOLT_POLICY_CONTROL_DATA_SPLIT_RAIL_MULTI_STEP *PLW2080_CTRL_VOLT_VOLT_POLICY_CONTROL_DATA_SPLIT_RAIL_MULTI_STEP;

/*!
 * Structure representing the control parameters associated with a
 * SPLIT_RAIL_SINGLE_STEP VOLT_POLICY
 */
typedef struct LW2080_CTRL_VOLT_VOLT_POLICY_CONTROL_DATA_SPLIT_RAIL_SINGLE_STEP {
    /*!
     * VOLT_POLICY_SPLIT_RAIL super class. Must always be first element
     * in the structure.
     */
    LW2080_CTRL_VOLT_VOLT_POLICY_CONTROL_DATA_SPLIT_RAIL super;
} LW2080_CTRL_VOLT_VOLT_POLICY_CONTROL_DATA_SPLIT_RAIL_SINGLE_STEP;
typedef struct LW2080_CTRL_VOLT_VOLT_POLICY_CONTROL_DATA_SPLIT_RAIL_SINGLE_STEP *PLW2080_CTRL_VOLT_VOLT_POLICY_CONTROL_DATA_SPLIT_RAIL_SINGLE_STEP;

/*!
 * Union of type-specific control parameters.
 */


/*!
 * Structure representing the control parameters associated with a VOLT_POLICY.
 */
typedef struct LW2080_CTRL_VOLT_VOLT_POLICY_CONTROL {
    /*!
     * LW2080_CTRL_BOARDOBJ super class.  Must always be first object in
     * structure.
     */
#endif // (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1) && (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
    LW2080_CTRL_BOARDOBJ super;
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1) && (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

    /*!
     * XAPI does not support using an element within another structure as a
     * discriminant. Placing this redundant type value here until that design
     * constraint can be fixed.
     */
    LwU8                 type;

    /*!
     * Type-specific information.
     */
    union {
        LW2080_CTRL_VOLT_VOLT_POLICY_CONTROL_DATA_SINGLE_RAIL            singleRail;
        LW2080_CTRL_VOLT_VOLT_POLICY_CONTROL_DATA_SINGLE_RAIL_MULTI_STEP singleRailMS;
        LW2080_CTRL_VOLT_VOLT_POLICY_CONTROL_DATA_SPLIT_RAIL             splitRail;
        LW2080_CTRL_VOLT_VOLT_POLICY_CONTROL_DATA_SPLIT_RAIL_MULTI_STEP  splitRailMS;
        LW2080_CTRL_VOLT_VOLT_POLICY_CONTROL_DATA_SPLIT_RAIL_SINGLE_STEP splitRailSS;
    } data;
} LW2080_CTRL_VOLT_VOLT_POLICY_CONTROL;
typedef struct LW2080_CTRL_VOLT_VOLT_POLICY_CONTROL *PLW2080_CTRL_VOLT_VOLT_POLICY_CONTROL;

/*!
 * Structure representing the control parameters associated with the GPU's
 * VOLT_POLICY functionality.
 */
typedef struct LW2080_CTRL_VOLT_VOLT_POLICIES_CONTROL_PARAMS {
    /*!
     * LW2080_CTRL_BOARDOBJGRP_E32 super class. Must always be first object in
     * structure.
     */
#endif // (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1) && (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
    LW2080_CTRL_BOARDOBJGRP_E32          super;
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1) && (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

    /*!
     * [out] - Array of VOLT_POLICY entries. Has valid indexes corresponding to
     * the bits set in @ref policyMask.
     */
    LW2080_CTRL_VOLT_VOLT_POLICY_CONTROL policies[LW2080_CTRL_VOLT_VOLT_POLICY_MAX_POLICIES];
} LW2080_CTRL_VOLT_VOLT_POLICIES_CONTROL_PARAMS;
typedef struct LW2080_CTRL_VOLT_VOLT_POLICIES_CONTROL_PARAMS *PLW2080_CTRL_VOLT_VOLT_POLICIES_CONTROL_PARAMS;

/*!
 * LW2080_CTRL_CMD_VOLT_VOLT_POLICIES_GET_CONTROL
 *
 * This command returns the VOLT_POLICY control parameters associated by the
 * VOLT_POLICY functionality
 *
 * See LW2080_CTRL_VOLT_VOLT_POLICY_CONTROL_PARAMS for documentation on the parameters.
 *
 * Possible status values returned are
 *   LWOS_STATUS_SUCCESS
 *   LWOS_STATUS_ERROR_ILWALID_ARGUMENT
 */
#define LW2080_CTRL_CMD_VOLT_VOLT_POLICIES_GET_CONTROL        (0x2080b211) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_VOLT_LEGACY_NON_PRIVILEGED_INTERFACE_ID << 8) | 0x11" */

/*
 * Volt Tests.
 */

/*
 ****** Important Notice ******
 * Please ensure that the test name identifiers below, match exactly with the
 * test name strings in rmt_volt.h file. These identifiers are used in
 * lw2080CtrlCmdVoltGenericTest() function, in file voltctrls.c
 */
#define LW2080_CTRL_VOLT_GENERIC_TEST_ID_VMIN_CAP             0x00000000
#define LW2080_CTRL_VOLT_GENERIC_TEST_ID_VMIN_CAP_NEGATIVE    0x00000001
#define LW2080_CTRL_VOLT_GENERIC_TEST_ID_DROOPY_ENGAGE        0x00000002
#define LW2080_CTRL_VOLT_GENERIC_TEST_ID_EDPP_THERM_EVENTS    0x00000003
#define LW2080_CTRL_VOLT_GENERIC_TEST_ID_THERM_MON_EDPP       0x00000004
#define LW2080_CTRL_VOLT_GENERIC_TEST_ID_FIXED_SLEW_RATE      0x00000005
#define LW2080_CTRL_VOLT_GENERIC_TEST_ID_FORCE_VMIN           0x00000006
#define LW2080_CTRL_VOLT_GENERIC_TEST_ID_VID_PWM_BOUND_FLOOR  0x00000007
#define LW2080_CTRL_VOLT_GENERIC_TEST_ID_VID_PWM_BOUND_CEIL   0x00000008
#define LW2080_CTRL_VOLT_GENERIC_TEST_ID_POSITIVE_CLVC_OFFSET 0x00000009
#define LW2080_CTRL_VOLT_GENERIC_TEST_ID_NEGATIVE_CLVC_OFFSET 0x0000000A

/*!
 * Macro to build a unique error code for each test ID.
 */
#define LW2080_CTRL_VOLT_TEST_BUILD_STATUS(_testid, _status) (((_testid) << 16) | (_status))

/*!
 * LW2080_CTRL_VOLT_GENERIC_TEST_ID_VMIN_CAP
 *
 * Possible reasons for LW2080_CTRL_VOLT_GENERIC_TEST_ID_VMIN_CAP to fail.
 *
 * _SUCCESS          : Test *_ID_VMIN_CAP is a success.
 * _VMIN_CAP_FAILURE : Test *_ID_VMIN_CAP failed because HW failed to Cap to Vmin.
 */
#define LW2080_CTRL_VOLT_GENERIC_TEST_ID_VMIN_CAP_STATUS_SUCCESS            LW2080_CTRL_VOLT_TEST_BUILD_STATUS(LW2080_CTRL_VOLT_GENERIC_TEST_ID_VMIN_CAP, 0x0)
#define LW2080_CTRL_VOLT_GENERIC_TEST_ID_VMIN_CAP_STATUS_VMIN_CAP_FAILURE   LW2080_CTRL_VOLT_TEST_BUILD_STATUS(LW2080_CTRL_VOLT_GENERIC_TEST_ID_VMIN_CAP, 0x1)

/*!
 * LW2080_CTRL_VOLT_GENERIC_TEST_ID_VMIN_CAP_NEGATIVE
 *
 * Possible reasons for LW2080_CTRL_VOLT_GENERIC_TEST_ID_VMIN_CAP_NEGATIVE to fail.
 *
 * _SUCCESS            : Test *_ID_VMIN_CAP_NEGATIVE is a success.
 * _UNEXPECTED_VOLTAGE : Test *_ID_VMIN_CAP_NEGATIVE failed because HW unexpectedtly capped to Vmin.
 */
#define LW2080_CTRL_VOLT_GENERIC_TEST_ID_VMIN_CAP_NEGATIVE_STATUS_SUCCESS               LW2080_CTRL_VOLT_TEST_BUILD_STATUS(LW2080_CTRL_VOLT_GENERIC_TEST_ID_VMIN_CAP_NEGATIVE, 0x0)
#define LW2080_CTRL_VOLT_GENERIC_TEST_ID_VMIN_CAP_NEGATIVE_STATUS_UNEXPECTED_VOLTAGE    LW2080_CTRL_VOLT_TEST_BUILD_STATUS(LW2080_CTRL_VOLT_GENERIC_TEST_ID_VMIN_CAP_NEGATIVE, 0x1)

/*!
 * LW2080_CTRL_VOLT_GENERIC_TEST_ID_DROOPY_ENGAGE
 *
 * Possible reasons for LW2080_CTRL_VOLT_GENERIC_TEST_ID_DROOPY_ENGAGE to fail.
 *
 * _SUCCESS                              : Test *_ID_DROOPY_ENGAGE is a success.
 * _UNEXPECTED_VOLTAGE_ON_ZERO_HI_OFFSET : Test *_ID_DROOPY_ENGAGE failed because of unexpected voltage on zero hi offset.
 * _UNEXPECTED_VOLTAGE_ON_HI_OFFSET      : Test *_ID_DROOPY_ENGAGE failed because of unexpected voltage some value of hi offset.
 * _VMIN_CAP_FAILURE                     : Test *_ID_DROOPY_ENGAGE failed because HW unexpectedtly capped to Vmin.
 */
#define LW2080_CTRL_VOLT_GENERIC_TEST_ID_DROOPY_ENGAGE_STATUS_SUCCESS                              LW2080_CTRL_VOLT_TEST_BUILD_STATUS(LW2080_CTRL_VOLT_GENERIC_TEST_ID_DROOPY_ENGAGE, 0x0)
#define LW2080_CTRL_VOLT_GENERIC_TEST_ID_DROOPY_ENGAGE_STATUS_UNEXPECTED_VOLTAGE_ON_ZERO_HI_OFFSET LW2080_CTRL_VOLT_TEST_BUILD_STATUS(LW2080_CTRL_VOLT_GENERIC_TEST_ID_DROOPY_ENGAGE, 0x1)
#define LW2080_CTRL_VOLT_GENERIC_TEST_ID_DROOPY_ENGAGE_STATUS_UNEXPECTED_VOLTAGE_ON_HI_OFFSET      LW2080_CTRL_VOLT_TEST_BUILD_STATUS(LW2080_CTRL_VOLT_GENERIC_TEST_ID_DROOPY_ENGAGE, 0x2)
#define LW2080_CTRL_VOLT_GENERIC_TEST_ID_DROOPY_ENGAGE_STATUS_VMIN_CAP_FAILURE                     LW2080_CTRL_VOLT_TEST_BUILD_STATUS(LW2080_CTRL_VOLT_GENERIC_TEST_ID_DROOPY_ENGAGE, 0x3)

/*!
 * LW2080_CTRL_VOLT_GENERIC_TEST_ID_EDPP_THERM_EVENTS
 *
 * Possible reasons for LW2080_CTRL_VOLT_GENERIC_TEST_ID_EDPP_THERM_EVENTS to fail.
 *
 * _SUCCESS                         : Test *_ID_EDPP_THERM_EVENTS is a success.
 * _UNEXPECTED_ASSERTION            : Test *_ID_EDPP_THERM_EVENTS failed because either of EDP events got unexpectedly asserted.
 * _EDPP_VMIN_UNEXPECTED_ASSERTION  : Test *_ID_EDPP_THERM_EVENTS failed because EDPP_VMIN event got unexpectedly asserted.
 * _EDPP_FONLY_UNEXPECTED_ASSERTION : Test *_ID_EDPP_THERM_EVENTS failed because EDPP_FONLY event got unexpectedly asserted.
 * _EDPP_VMIN_ASSERTION_FAILED      : Test *_ID_EDPP_THERM_EVENTS failed because EDPP_VMIN event did not assert.
 * _EDPP_FONLY_ASSERTION_FAILED     : Test *_ID_EDPP_THERM_EVENTS failed because EDPP_FONLY event did not assert.
 * _EDPP_VMIN_DEASSERTION_FAILED    : Test *_ID_EDPP_THERM_EVENTS failed because EDPP_VMIN event did not deassert.
  * _EDPP_FONLY_DEASSERTION_FAILED  : Test *_ID_EDPP_THERM_EVENTS failed because EDPP_FONLY event did not deassert.
 */
#define LW2080_CTRL_VOLT_GENERIC_TEST_ID_EDPP_THERM_EVENTS_STATUS_SUCCESS                         LW2080_CTRL_VOLT_TEST_BUILD_STATUS(LW2080_CTRL_VOLT_GENERIC_TEST_ID_EDPP_THERM_EVENTS, 0x0)
#define LW2080_CTRL_VOLT_GENERIC_TEST_ID_EDPP_THERM_EVENTS_STATUS_UNEXPECTED_ASSERTION            LW2080_CTRL_VOLT_TEST_BUILD_STATUS(LW2080_CTRL_VOLT_GENERIC_TEST_ID_EDPP_THERM_EVENTS, 0x1)
#define LW2080_CTRL_VOLT_GENERIC_TEST_ID_EDPP_THERM_EVENTS_STATUS_EDPP_VMIN_UNEXPECTED_ASSERTION  LW2080_CTRL_VOLT_TEST_BUILD_STATUS(LW2080_CTRL_VOLT_GENERIC_TEST_ID_EDPP_THERM_EVENTS, 0x2)
#define LW2080_CTRL_VOLT_GENERIC_TEST_ID_EDPP_THERM_EVENTS_STATUS_EDPP_FONLY_UNEXPECTED_ASSERTION LW2080_CTRL_VOLT_TEST_BUILD_STATUS(LW2080_CTRL_VOLT_GENERIC_TEST_ID_EDPP_THERM_EVENTS, 0x3)
#define LW2080_CTRL_VOLT_GENERIC_TEST_ID_EDPP_THERM_EVENTS_STATUS_EDPP_VMIN_ASSERTION_FAILED      LW2080_CTRL_VOLT_TEST_BUILD_STATUS(LW2080_CTRL_VOLT_GENERIC_TEST_ID_EDPP_THERM_EVENTS, 0x4)
#define LW2080_CTRL_VOLT_GENERIC_TEST_ID_EDPP_THERM_EVENTS_STATUS_EDPP_FONLY_ASSERTION_FAILED     LW2080_CTRL_VOLT_TEST_BUILD_STATUS(LW2080_CTRL_VOLT_GENERIC_TEST_ID_EDPP_THERM_EVENTS, 0x5)
#define LW2080_CTRL_VOLT_GENERIC_TEST_ID_EDPP_THERM_EVENTS_STATUS_EDPP_VMIN_DEASSERTION_FAILED    LW2080_CTRL_VOLT_TEST_BUILD_STATUS(LW2080_CTRL_VOLT_GENERIC_TEST_ID_EDPP_THERM_EVENTS, 0x6)
#define LW2080_CTRL_VOLT_GENERIC_TEST_ID_EDPP_THERM_EVENTS_STATUS_EDPP_FONLY_DEASSERTION_FAILED   LW2080_CTRL_VOLT_TEST_BUILD_STATUS(LW2080_CTRL_VOLT_GENERIC_TEST_ID_EDPP_THERM_EVENTS, 0x7)

/*!
 * LW2080_CTRL_VOLT_GENERIC_TEST_ID_THERM_MON_EDPP
 *
 * Possible reasons for LW2080_CTRL_VOLT_GENERIC_TEST_ID_THERM_MON_EDPP to fail.
 *
 * _SUCCESS            : Test *_ID_THERM_MON_EDPP is a success.
 * _INCREMENT_FAILURE  : Test *_ID_THERM_MON_EDPP failed because the monitor did not increment.
 * _UNKNOWN_EDPP_EVENT : Test *_ID_THERM_MON_EDPP failed because of unknown EDPP event.
 */
#define LW2080_CTRL_VOLT_GENERIC_TEST_ID_THERM_MON_EDPP_STATUS_SUCCESS               LW2080_CTRL_VOLT_TEST_BUILD_STATUS(LW2080_CTRL_VOLT_GENERIC_TEST_ID_THERM_MON_EDPP, 0x0)
#define LW2080_CTRL_VOLT_GENERIC_TEST_ID_THERM_MON_EDPP_STATUS_INCREMENT_FAILURE     LW2080_CTRL_VOLT_TEST_BUILD_STATUS(LW2080_CTRL_VOLT_GENERIC_TEST_ID_THERM_MON_EDPP, 0x1)
#define LW2080_CTRL_VOLT_GENERIC_TEST_ID_THERM_MON_EDPP_STATUS_UNKNOWN_EDPP_EVENT    LW2080_CTRL_VOLT_TEST_BUILD_STATUS(LW2080_CTRL_VOLT_GENERIC_TEST_ID_THERM_MON_EDPP, 0x2)

/*!
 * LW2080_CTRL_VOLT_GENERIC_TEST_ID_FIXED_SLEW_RATE
 *
 * Possible reasons for LW2080_CTRL_VOLT_GENERIC_TEST_ID_FIXED_SLEW_RATE to fail.
 *
 * _SUCCESS                                : Test *_ID_FIXED_SLEW_RATE is a success.
 * _UNEXPECTED_VOLTAGE_ON_FSR_DISABLED     : Test *_ID_FIXED_SLEW_RATE failed because of unexpected voltage when fixed slew rate is disabled.
 * _UNEXPECTED_VOLTAGE_ON_FSR_ENABLED      : Test *_ID_FIXED_SLEW_RATE failed because of unexpected voltage when fixed slew rate is enabled.
 * _UNEXPECTED_VOLTAGE_ON_FSR_IPC_ENABLED  : Test *_ID_FIXED_SLEW_RATE failed because of unexpected voltage when both fixed slew rate and IPC is enabled.
 */
#define LW2080_CTRL_VOLT_GENERIC_TEST_ID_FIXED_SLEW_RATE_STATUS_SUCCESS                               LW2080_CTRL_VOLT_TEST_BUILD_STATUS(LW2080_CTRL_VOLT_GENERIC_TEST_ID_FIXED_SLEW_RATE, 0x0)
#define LW2080_CTRL_VOLT_GENERIC_TEST_ID_FIXED_SLEW_RATE_STATUS_UNEXPECTED_VOLTAGE_ON_FSR_DISABLED    LW2080_CTRL_VOLT_TEST_BUILD_STATUS(LW2080_CTRL_VOLT_GENERIC_TEST_ID_FIXED_SLEW_RATE, 0x1)
#define LW2080_CTRL_VOLT_GENERIC_TEST_ID_FIXED_SLEW_RATE_STATUS_UNEXPECTED_VOLTAGE_ON_FSR_ENABLED     LW2080_CTRL_VOLT_TEST_BUILD_STATUS(LW2080_CTRL_VOLT_GENERIC_TEST_ID_FIXED_SLEW_RATE, 0x2)
#define LW2080_CTRL_VOLT_GENERIC_TEST_ID_FIXED_SLEW_RATE_STATUS_UNEXPECTED_VOLTAGE_ON_FSR_IPC_ENABLED LW2080_CTRL_VOLT_TEST_BUILD_STATUS(LW2080_CTRL_VOLT_GENERIC_TEST_ID_FIXED_SLEW_RATE, 0x3)

#define LW2080_CTRL_VOLT_TEST_STATUS(testid, status) (LW2080_CTRL_VOLT_GENERIC_TEST_ID_##testid##_STATUS_##status)

/*
 * LW2080_CTRL_CMD_VOLT_VOLT_GENERIC_TEST
 *
 *   Possible Volt Generic Test Result.
 *
 * _SUCCESS               : Test completed successfully.
 * _NOT_IMPLEMENTED       : Test is not implemented in RM/PMU.
 * _NOT_SUPPORTED         : Test is not supported on the GPU.
 * _UNSPECIFIED_PMU_ERROR : Test ran into an unspecified PMU error.
 * _ERROR_GENERIC         :  Otherwise.
 *
 */
#define LW2080_CTRL_VOLT_GENERIC_TEST_SUCCESS                 0x00000000
#define LW2080_CTRL_VOLT_GENERIC_TEST_NOT_IMPLEMENTED         0x00000001
#define LW2080_CTRL_VOLT_GENERIC_TEST_NOT_SUPPORTED           0x00000002
#define LW2080_CTRL_VOLT_GENERIC_TEST_INSUFFICIENT_PRIVILEDGE 0x00000003
#define LW2080_CTRL_VOLT_GENERIC_TEST_UNSPECIFIED_PMU_ERROR   0x00000004
#define LW2080_CTRL_VOLT_GENERIC_TEST_ERROR_GENERIC           0xFFFFFFFF

/*!
 * LW2080_CTRL_CMD_VOLT_VOLT_GENERIC_TEST
 *
 * This command runs one of the VOLT halified tests specified by
 * LW2080_CTRL_VOLT_GENERIC_TEST_ID_<xyz>.
 *
 * Possible status values returned are:
 *  LW_OK
 */
#define LW2080_CTRL_CMD_VOLT_VOLT_GENERIC_TEST                (0x2080b215) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_VOLT_LEGACY_NON_PRIVILEGED_INTERFACE_ID << 8) | LW2080_CTRL_VOLT_GENERIC_TEST_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_VOLT_GENERIC_TEST_PARAMS_MESSAGE_ID (0x15U)

typedef struct LW2080_CTRL_VOLT_GENERIC_TEST_PARAMS {
    /*!
     * [in] - Specifies the index of the test to execute as per
     * @ref volttest_name_a defined in rmt_volt.h
     */
    LwU32 index;

    /*!
     * [out] - Specifies the generic status of the test exelwted specified
     * by @index which is one of LW2080_CTRL_VOLT_GENERIC_TEST_<xyz>
     */
    LwU32 outStatus;

    /*!
     * [out] - Specifies the output status data given by the test exelwted
     * specified by @index.
     */
    LwU32 outData;
} LW2080_CTRL_VOLT_GENERIC_TEST_PARAMS;
typedef struct LW2080_CTRL_VOLT_GENERIC_TEST_PARAMS *PLW2080_CTRL_VOLT_GENERIC_TEST_PARAMS;

/*!
 * LW2080_CTRL_CMD_VOLT_PMUMON_VOLT_RAILS_GET_SAMPLES
 *
 * Control call to query the samples within the PWR_CHANNELS PMUMON queue.
 */
#define LW2080_CTRL_CMD_VOLT_PMUMON_VOLT_RAILS_GET_SAMPLES (0x2080b216) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_VOLT_LEGACY_NON_PRIVILEGED_INTERFACE_ID << 8) | LW2080_CTRL_VOLT_PMUMON_VOLT_RAILS_GET_SAMPLES_PARAMS_MESSAGE_ID" */

/*!
 * Input/Output parameters for @ref LW2080_CTRL_CMD_VOLT_PMUMON_VOLT_RAILS_GET_SAMPLES
 */
#define LW2080_CTRL_VOLT_PMUMON_VOLT_RAILS_GET_SAMPLES_PARAMS_MESSAGE_ID (0x16U)

typedef struct LW2080_CTRL_VOLT_PMUMON_VOLT_RAILS_GET_SAMPLES_PARAMS {
    /*!
     * [in/out] Meta-data for the samples[] below. Will be modified by the
     *          control call on caller's behalf and should be passed back in
     *          un-modified for subsequent calls.
     */
#endif // (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1) && (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
    LW2080_CTRL_PMUMON_GET_SAMPLES_SUPER super;
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1) && (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

    /*!
     * [out] Between the last call and current call, samples[0...super.numSamples-1]
     *       have been published to the pmumon queue. Samples are copied into
     *       this buffer in chronological order. Indexes within this buffer do
     *       not represent indexes of samples in the actual PMUMON queue.
     */
#endif // (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1) && (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
    LW_DECLARE_ALIGNED(LW2080_CTRL_VOLT_PMUMON_VOLT_RAILS_SAMPLE samples[LW2080_CTRL_VOLT_PMUMON_VOLT_RAILS_SAMPLE_COUNT], 8);
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1) && (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
} LW2080_CTRL_VOLT_PMUMON_VOLT_RAILS_GET_SAMPLES_PARAMS;
typedef struct LW2080_CTRL_VOLT_PMUMON_VOLT_RAILS_GET_SAMPLES_PARAMS *PLW2080_CTRL_VOLT_PMUMON_VOLT_RAILS_GET_SAMPLES_PARAMS;

/* _ctrl2080volt_opaque_non_privileged_h_ */

#endif // (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

