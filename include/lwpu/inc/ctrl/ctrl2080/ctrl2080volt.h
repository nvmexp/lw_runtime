/*
 * SPDX-FileCopyrightText: Copyright (c) 2015-2022 LWPU CORPORATION & AFFILIATES. All rights reserved.
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
// Source file: ctrl/ctrl2080/ctrl2080volt.finn
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
#include "ctrl/ctrl2080/ctrl2080pmumon.h"

/*!
 * Macros for defining the array indices for the 6 voltage limits -
 * relLimit, altRelLimit, ovLimit, vminLimit, vCritLow, VcritHigh.
 */
#define LW2080_CTRL_VOLT_RAIL_VOLT_DELTA_REL_LIM_IDX                   0x00U
#define LW2080_CTRL_VOLT_RAIL_VOLT_DELTA_ALT_REL_LIM_IDX               0x01U
#define LW2080_CTRL_VOLT_RAIL_VOLT_DELTA_OV_LIM_IDX                    0x02U
#define LW2080_CTRL_VOLT_RAIL_VOLT_DELTA_VMIN_LIM_IDX                  0x03U
#define LW2080_CTRL_VOLT_RAIL_VOLT_DELTA_VCRIT_LOW_IDX                 0x04U
#define LW2080_CTRL_VOLT_RAIL_VOLT_DELTA_VCRIT_HIGH_IDX                0x05U
#define LW2080_CTRL_VOLT_RAIL_VOLT_DELTA_MAX_ENTRIES                   0x06U

/*!
 * Maximum number of VOLT_RAILs which are exported to a RMCTRL client. This
 * number must be less than or equal to LW2080_CTRL_VOLT_VOLT_RAIL_MAX_RAILS.
 * Compile time sanity check is performed on this value.
 */
#define LW2080_CTRL_VOLT_VOLT_RAIL_CLIENT_MAX_RAILS                    0x04U

/*!
 * Maximum number of VOLT_RAILs which can be supported in the RM or PMU.
 */
#define LW2080_CTRL_VOLT_VOLT_RAIL_MAX_RAILS                           LW2080_CTRL_BOARDOBJGRP_E32_MAX_OBJECTS

/*!
 * Maximum number of VOLT_DEVICEs which can be supported in the RM or PMU.
 */
#define LW2080_CTRL_VOLT_VOLT_DEVICE_MAX_DEVICES                       LW2080_CTRL_BOARDOBJGRP_E32_MAX_OBJECTS

/*!
 * Maximum number of VOLT_POLICYs which can be supported in the RM or PMU.
 */
#define LW2080_CTRL_VOLT_VOLT_POLICY_MAX_POLICIES                      LW2080_CTRL_BOARDOBJGRP_E32_MAX_OBJECTS

/*!
 * Macros for Voltage Domain HAL.
 */
#define LW2080_CTRL_VOLT_VOLT_DOMAIN_HAL_ILWALID                       0xFFU
#define LW2080_CTRL_VOLT_VOLT_DOMAIN_HAL_GP10X_SINGLE_RAIL             0x00U
#define LW2080_CTRL_VOLT_VOLT_DOMAIN_HAL_GP10X_SPLIT_RAIL              0x01U
#define LW2080_CTRL_VOLT_VOLT_DOMAIN_HAL_GA10X_MULTI_RAIL              0x02U

/*!
 * Macros for Voltage Domains.
 * @defgroup LW2080_CTRL_VOLT_VOLT_DOMAIN_ENUM
 * @{
 */
#define LW2080_CTRL_VOLT_VOLT_DOMAIN_ILWALID                           0x00U
#define LW2080_CTRL_VOLT_VOLT_DOMAIN_LOGIC                             0x01U
#define LW2080_CTRL_VOLT_VOLT_DOMAIN_SRAM                              0x02U
#define LW2080_CTRL_VOLT_VOLT_DOMAIN_MSVDD                             0x03U
#define LW2080_CTRL_VOLT_VOLT_DOMAIN_MAX_ENTRIES                       0x03U
/*! @} */

/*!
 * Macros for Voltage Device Types.
 */
#define LW2080_CTRL_VOLT_VOLT_DEVICE_TYPE_BASE                         0x00U
#define LW2080_CTRL_VOLT_VOLT_DEVICE_TYPE_VID                          0x01U
#define LW2080_CTRL_VOLT_VOLT_DEVICE_TYPE_VID_REPROG                   0x02U
#define LW2080_CTRL_VOLT_VOLT_DEVICE_TYPE_PWM                          0x03U
#define LW2080_CTRL_VOLT_VOLT_DEVICE_TYPE_PWM_SCI                      0x04U
#define LW2080_CTRL_VOLT_VOLT_DEVICE_TYPE_SOC                          0x05U
// Insert new types here and increment _MAX
#define LW2080_CTRL_VOLT_VOLT_DEVICE_TYPE_MAX                          0x06U
#define LW2080_CTRL_VOLT_VOLT_DEVICE_TYPE_ILWALID                      0xFFU

/*!
 * Macros for Volt Policy types.
 */
#define LW2080_CTRL_VOLT_VOLT_POLICY_TYPE_BASE                         0x00U
#define LW2080_CTRL_VOLT_VOLT_POLICY_TYPE_SINGLE_RAIL                  0x01U
#define LW2080_CTRL_VOLT_VOLT_POLICY_TYPE_SPLIT_RAIL_MULTI_STEP        0x02U
#define LW2080_CTRL_VOLT_VOLT_POLICY_TYPE_SPLIT_RAIL_SINGLE_STEP       0x03U
#define LW2080_CTRL_VOLT_VOLT_POLICY_TYPE_SINGLE_RAIL_MULTI_STEP       0x04U
#define LW2080_CTRL_VOLT_VOLT_POLICY_TYPE_MULTI_RAIL                   0x05U
#define LW2080_CTRL_VOLT_VOLT_POLICY_TYPE_SPLIT_RAIL                   0x06U
// Insert new types here and increment _MAX
#define LW2080_CTRL_VOLT_VOLT_POLICY_TYPE_MAX                          0x07U
#define LW2080_CTRL_VOLT_VOLT_POLICY_TYPE_UNKNOWN                      0xFFU
#define LW2080_CTRL_VOLT_VOLT_POLICY_TYPE_ILWALID                      0xFFU

/*!
 * Macros for VOLT RAIL RAM ASSIST TYPES.
 */
#define LW2080_CTRL_VOLT_VOLT_RAIL_RAM_ASSIST_TYPE_DISABLED            0x00U
#define LW2080_CTRL_VOLT_VOLT_RAIL_RAM_ASSIST_TYPE_STATIC              0x01U
#define LW2080_CTRL_VOLT_VOLT_RAIL_RAM_ASSIST_TYPE_DYNAMIC_WITHOUT_LUT 0x02U
#define LW2080_CTRL_VOLT_VOLT_RAIL_RAM_ASSIST_TYPE_DYNAMIC_WITH_LUT    0x03U

/*!
 * Macros for Volt Policy Client types.
 */
#define LW2080_CTRL_VOLT_VOLT_POLICY_CLIENT_ILWALID                    0x00U
#define LW2080_CTRL_VOLT_VOLT_POLICY_CLIENT_PERF_CORE_VF_SEQ           0x01U

/*!
 * Maximum number of VSEL GPIOs for a VID VOLTAGE_DEVICE which can be supported
 * in the RM or PMU.
 */
#define LW2080_CTRL_VOLT_VOLT_DEV_VID_VSEL_MAX_ENTRIES                 (8U)

/*!
 * Macros for Volt Device Operation types.
 */
#define LW2080_CTRL_VOLT_VOLT_DEVICE_OPERATION_TYPE_ILWALID            0x00U
#define LW2080_CTRL_VOLT_VOLT_DEVICE_OPERATION_TYPE_DEFAULT            0x01U
#define LW2080_CTRL_VOLT_VOLT_DEVICE_OPERATION_TYPE_LPWR_STEADY_STATE  0x02U
#define LW2080_CTRL_VOLT_VOLT_DEVICE_OPERATION_TYPE_LPWR_SLEEP_STATE   0x03U
#define LW2080_CTRL_VOLT_VOLT_DEVICE_OPERATION_TYPE_IPC_VMIN           0x04U

#define LW2080_CTRL_VOLT_VOLT_DEVICE_PSV_GATE_VOLTAGE_UV_ILWALID       LW_U32_MAX

/*!
 * Special value corresponding to an invalid Voltage Rail Index.
 */
#define LW2080_CTRL_VOLT_VOLT_RAIL_INDEX_ILWALID                       LW2080_CTRL_BOARDOBJ_IDX_ILWALID_8BIT

/*!
 * Special value corresponding to an invalid Voltage Device Index.
 */
#define LW2080_CTRL_VOLT_VOLT_DEVICE_INDEX_ILWALID                     LW2080_CTRL_BOARDOBJ_IDX_ILWALID_8BIT

/*!
 * Special value corresponding to an invalid Voltage Policy Index.
 */
#define LW2080_CTRL_VOLT_VOLT_POLICY_INDEX_ILWALID                     LW2080_CTRL_BOARDOBJ_IDX_ILWALID_8BIT

/*!
 * Enumeration of different clients that can apply volt delta on target voltage.
 */
#define LW2080_CTRL_VOLT_VOLT_RAIL_OFFSET_CLFC                         0x00U
#define LW2080_CTRL_VOLT_VOLT_RAIL_OFFSET_CLVC                         0x01U
#define LW2080_CTRL_VOLT_VOLT_RAIL_OFFSET_VOLT_MARGIN                  0x02U
#define LW2080_CTRL_VOLT_VOLT_RAIL_OFFSET_MAX                          0x03U

/*!
 * @defgroup LW2080_CTRL_VOLT_VOLT_RAIL_SENSED_VOLTAGE_MODE_ENUM
 *
 * Macros for sensed voltage mode required to collate readings from multiple
 * ADCs for a VOLTAGE_RAIL.
 *
 * @{
 */
typedef LwU8 LW2080_CTRL_VOLT_VOLT_RAIL_SENSED_VOLTAGE_MODE_ENUM;
#define LW2080_CTRL_VOLT_VOLT_RAIL_SENSED_VOLTAGE_MODE_MIN 0x00U
#define LW2080_CTRL_VOLT_VOLT_RAIL_SENSED_VOLTAGE_MODE_MAX 0x01U
#define LW2080_CTRL_VOLT_VOLT_RAIL_SENSED_VOLTAGE_MODE_AVG 0x02U
#define LW2080_CTRL_VOLT_VOLT_RAIL_SENSED_VOLTAGE_MODE_NUM 0x03U
/*!@}*/

/*!
 * Rail Action parameter enumeration.
 * This will denote what control action is to be taken on a given voltage rail.
 */
#define LW2080_CTRL_VOLT_VOLT_RAIL_ACTION_VF_SWITCH        0x00U
#define LW2080_CTRL_VOLT_VOLT_RAIL_ACTION_GATE             0x01U
#define LW2080_CTRL_VOLT_VOLT_RAIL_ACTION_UNGATE           0x02U
#define LW2080_CTRL_VOLT_VOLT_RAIL_ACTION_ILWALID          0xFFU

/*!
 * BA SCALING PWR EQN and VOLT DEV TABLE Invalid Indices.
 */
#define LW2080_CTRL_BA_SCALING_PWR_EQN_IDX_ILWALID         0xFFU

/*!
 * Defines the structure that holds data used to execute the
 * VOLT_SET_VOLTAGE RPC.
 */
typedef struct LW2080_CTRL_VOLT_VOLT_RAIL_LIST_ITEM {
    /*!
     * Voltage Rail Index corresponding to a VOLT_RAIL.
     */
    LwU8  railIdx;

    /*!
     * Denotes the control action to be taken on the rail described via
     * LW2080_CTRL_VOLT_VOLT_RAIL_ACTION_<xyz>.
     */
    LwU8  railAction;

    /*!
     * Target voltage in uV.
     */
    LwU32 voltageuV;

    /*!
     * V_{min, noise-unaware} - The minimum voltage (uV) with respect to
     * noise-unaware constraints on this VOLT_RAIL.
     */
    LwU32 voltageMinNoiseUnawareuV;

    /*!
     * Array of Voltage offset (uV). Index in this array is statically mapped
     * to @ref LW2080_CTRL_VOLT_VOLT_RAIL_OFFSET_<XYZ>
     */
    LwS32 voltOffsetuV[LW2080_CTRL_VOLT_VOLT_RAIL_OFFSET_MAX];
} LW2080_CTRL_VOLT_VOLT_RAIL_LIST_ITEM;
typedef struct LW2080_CTRL_VOLT_VOLT_RAIL_LIST_ITEM *PLW2080_CTRL_VOLT_VOLT_RAIL_LIST_ITEM;

/*!
 * Structure containing the number of voltage rails and the list of rail items
 * @ref LW2080_CTRL_VOLT_VOLT_RAIL_LIST_ITEM.
 */
typedef struct LW2080_CTRL_VOLT_VOLT_RAIL_LIST {
    /*!
     * Number of VOLT_RAILs that require the voltage change.
     */
    LwU8                                 numRails;

    /*!
     * List of @ref LW2080_CTRL_VOLT_VOLT_RAIL_LIST_ITEM entries.
     */
    LW2080_CTRL_VOLT_VOLT_RAIL_LIST_ITEM rails[LW2080_CTRL_VOLT_VOLT_RAIL_CLIENT_MAX_RAILS];
} LW2080_CTRL_VOLT_VOLT_RAIL_LIST;
typedef struct LW2080_CTRL_VOLT_VOLT_RAIL_LIST *PLW2080_CTRL_VOLT_VOLT_RAIL_LIST;

/*!
 *  Structure representing the voltage data associated with a
 *  MULTI_RAIL VOLT_POLICY
 */
typedef struct LW2080_CTRL_VOLT_VOLT_POLICY_STATUS_DATA_MULTI_RAIL_ITEM {
    /*!
     * Voltage Rail Index corresponding to a VOLT_RAIL.
     */
    LwU8  railIdx;

    /*!
     * Cached value of most recently requested voltage without applying
     * VOLT_POLICY::offsetVoltuV.
     */
    LwU32 lwrrVoltuV;
} LW2080_CTRL_VOLT_VOLT_POLICY_STATUS_DATA_MULTI_RAIL_ITEM;
typedef struct LW2080_CTRL_VOLT_VOLT_POLICY_STATUS_DATA_MULTI_RAIL_ITEM *PLW2080_CTRL_VOLT_VOLT_POLICY_STATUS_DATA_MULTI_RAIL_ITEM;

/*!
 *  Structure representing the the number of rails and list of voltage data items
 *  @ref LW2080_CTRL_VOLT_VOLT_POLICY_STATUS_DATA_MULTI_RAIL_ITEM.
 */
typedef struct LW2080_CTRL_VOLT_VOLT_POLICY_STATUS_DATA_MULTI_RAIL {
    LwU8                                                     numRails;

    /*!
     * Cached value of most recently requested voltage without applying
     * VOLT_POLICY::offsetVoltuV.
     */
    LW2080_CTRL_VOLT_VOLT_POLICY_STATUS_DATA_MULTI_RAIL_ITEM railItems[LW2080_CTRL_VOLT_VOLT_RAIL_CLIENT_MAX_RAILS];
} LW2080_CTRL_VOLT_VOLT_POLICY_STATUS_DATA_MULTI_RAIL;
typedef struct LW2080_CTRL_VOLT_VOLT_POLICY_STATUS_DATA_MULTI_RAIL *PLW2080_CTRL_VOLT_VOLT_POLICY_STATUS_DATA_MULTI_RAIL;

/*!
 * @brief   With sample period being potentially as fast every 100ms, this gives
 *          us 5 seconds worth of data.
 */
#define LW2080_CTRL_VOLT_PMUMON_VOLT_RAILS_SAMPLE_COUNT   (50U)

/*!
 * Temporary until an INFO control call is stubbed out that exposes the supported
 * feature set of the sampling.
 */
#define LW2080_CTRL_VOLT_PMUMON_VOLT_RAILS_SAMPLE_ILWALID (LW_U32_MAX)

/*!
 * A single sample of the power channels at a particular point in time.
 */
typedef struct LW2080_CTRL_VOLT_PMUMON_VOLT_RAILS_SAMPLE {
    /*!
     * Ptimer timestamp of when this data was collected.
     */
    LW_DECLARE_ALIGNED(LW2080_CTRL_PMUMON_SAMPLE super, 8);

    /*!
     * Point sampled programmed LWVDD voltage in uV.
     *
     * LW2080_CTRL_VOLT_PMUMON_VOLT_RAILS_SAMPLE_ILWALID if not supported.
     */
    LwU32 lwvddVoltageuV;

    /*!
     * Point sampled programmed MSVDD voltage in uV.
     *
     * LW2080_CTRL_VOLT_PMUMON_VOLT_RAILS_SAMPLE_ILWALID if not supported.
     */
    LwU32 msvddVoltageuV;
} LW2080_CTRL_VOLT_PMUMON_VOLT_RAILS_SAMPLE;
typedef struct LW2080_CTRL_VOLT_PMUMON_VOLT_RAILS_SAMPLE *PLW2080_CTRL_VOLT_PMUMON_VOLT_RAILS_SAMPLE;


/* _ctrl2080volt_h_ */



#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)

#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1) && (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

#include "ctrl/ctrl2080/ctrl2080volt_opaque_non_privileged.h"
#include "ctrl/ctrl2080/ctrl2080volt_opaque_privileged.h"
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1) && (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

