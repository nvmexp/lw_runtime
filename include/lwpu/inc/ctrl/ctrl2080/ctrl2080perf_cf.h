/*
 * SPDX-FileCopyrightText: Copyright (c) 2017-2021 LWPU CORPORATION & AFFILIATES. All rights reserved.
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
// Source file: ctrl/ctrl2080/ctrl2080perf_cf.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)



#include "lwfixedtypes.h"
#include "ctrl/ctrl2080/ctrl2080base.h"
#include "ctrl/ctrl2080/ctrl2080boardobj.h"
#include "ctrl/ctrl2080/ctrl2080perf.h"
#include "ctrl/ctrl2080/ctrl2080pmgr.h"
#include "ctrl/ctrl2080/ctrl2080perf_cf_pwr_model.h"
#include "ctrl/ctrl2080/ctrl2080nne.h"

/* ---------------------------- PERF_CF Sensors ------------------------------ */

/*!
 * Macros for PERF_CF Sensor types
 */
#define LW2080_CTRL_PERF_PERF_CF_SENSOR_TYPE_PMU           0x00
#define LW2080_CTRL_PERF_PERF_CF_SENSOR_TYPE_PMU_FB        0x01
#define LW2080_CTRL_PERF_PERF_CF_SENSOR_TYPE_PEX           0x02
#define LW2080_CTRL_PERF_PERF_CF_SENSOR_TYPE_PTIMER        0x03
#define LW2080_CTRL_PERF_PERF_CF_SENSOR_TYPE_PTIMER_CLK    0x04
#define LW2080_CTRL_PERF_PERF_CF_SENSOR_TYPE_CLKCNTR       0x05
#define LW2080_CTRL_PERF_PERF_CF_SENSOR_TYPE_PGTIME        0x06
#define LW2080_CTRL_PERF_PERF_CF_SENSOR_TYPE_THERM_MONITOR 0x07

/*!
 * Maximum number of PERF_CF_SENSORs which can be supported in the RM or PMU.
 */
#define LW2080_CTRL_PERF_PERF_CF_SENSORS_MAX_OBJECTS       LW2080_CTRL_BOARDOBJGRP_E32_MAX_OBJECTS

/*!
 * Structure describing PERF_CF_SENSOR_PMU static information/POR.
 */
typedef struct LW2080_CTRL_PERF_PERF_CF_SENSOR_INFO_PMU {
    /*!
     * HW idle mask 0 value.
     */
    LwU32 idleMask0;
    /*!
     * HW idle mask 1 value.
     */
    LwU32 idleMask1;
    /*!
     * HW idle mask 2 value.
     */
    LwU32 idleMask2;
    /*!
     * HW counter index.
     */
    LwU8  counterIdx;
} LW2080_CTRL_PERF_PERF_CF_SENSOR_INFO_PMU;
typedef struct LW2080_CTRL_PERF_PERF_CF_SENSOR_INFO_PMU *PLW2080_CTRL_PERF_PERF_CF_SENSOR_INFO_PMU;

/*!
 * Structure describing PERF_CF_SENSOR_PMU_FB static information/POR.
 */
typedef struct LW2080_CTRL_PERF_PERF_CF_SENSOR_INFO_PMU_FB {
    /*!
     * PERF_CF_SENSOR_PMU super class.
     * Must always be first element in the structure.
     */
    LW2080_CTRL_PERF_PERF_CF_SENSOR_INFO_PMU super;
    /*!
     * Constant scaling factor.
     */
    LwUFXP20_12                              scale;
    /*!
     * HW idle counter for FB needs to be scaled by clocks ratio. This is the multiplier clock domain index.
     */
    LwU8                                     clkDomIdxMulti;
    /*!
     * HW idle counter for FB needs to be scaled by clocks ratio. This is the divisor clock domain index.
     */
    LwU8                                     clkDomIdxDiv;
} LW2080_CTRL_PERF_PERF_CF_SENSOR_INFO_PMU_FB;
typedef struct LW2080_CTRL_PERF_PERF_CF_SENSOR_INFO_PMU_FB *PLW2080_CTRL_PERF_PERF_CF_SENSOR_INFO_PMU_FB;

/*!
 * Structure describing PERF_CF_SENSOR_PEX static information/POR.
 */
typedef struct LW2080_CTRL_PERF_PERF_CF_SENSOR_INFO_PEX {
    /*!
     * LW_TRUE = RX, LW_FALSE = TX.
     */
    LwBool bRx;
} LW2080_CTRL_PERF_PERF_CF_SENSOR_INFO_PEX;
typedef struct LW2080_CTRL_PERF_PERF_CF_SENSOR_INFO_PEX *PLW2080_CTRL_PERF_PERF_CF_SENSOR_INFO_PEX;

/*!
 * Structure describing PERF_CF_SENSOR_PTIMER static information/POR.
 */
typedef struct LW2080_CTRL_PERF_PERF_CF_SENSOR_INFO_PTIMER {
    /*!
     * Lwrrently we do NOT have any static info parameter in this class.
     */
    LwU8 rsvd;
} LW2080_CTRL_PERF_PERF_CF_SENSOR_INFO_PTIMER;
typedef struct LW2080_CTRL_PERF_PERF_CF_SENSOR_INFO_PTIMER *PLW2080_CTRL_PERF_PERF_CF_SENSOR_INFO_PTIMER;

/*!
 * Structure describing PERF_CF_SENSOR_PTIMER_CLK static information/POR.
 */
typedef struct LW2080_CTRL_PERF_PERF_CF_SENSOR_INFO_PTIMER_CLK {
    /*!
     * PERF_CF_SENSOR_PTIMER super class.
     * Must always be first element in the structure.
     */
    LW2080_CTRL_PERF_PERF_CF_SENSOR_INFO_PTIMER super;
    /*!
     * Clock frequency (kHz) for PTIMER to HW base counter scaling.
     */
    LwU32                                       clkFreqKHz;
} LW2080_CTRL_PERF_PERF_CF_SENSOR_INFO_PTIMER_CLK;
typedef struct LW2080_CTRL_PERF_PERF_CF_SENSOR_INFO_PTIMER_CLK *PLW2080_CTRL_PERF_PERF_CF_SENSOR_INFO_PTIMER_CLK;

/*!
 * Structure describing PERF_CF_SENSOR_CLKCNTR static information/POR.
 */
typedef struct LW2080_CTRL_PERF_PERF_CF_SENSOR_INFO_CLKCNTR {
    /*!
     * Clock domain index to count.
     */
    LwU8 clkDomIdx;
} LW2080_CTRL_PERF_PERF_CF_SENSOR_INFO_CLKCNTR;
typedef struct LW2080_CTRL_PERF_PERF_CF_SENSOR_INFO_CLKCNTR *PLW2080_CTRL_PERF_PERF_CF_SENSOR_INFO_CLKCNTR;

/*!
 * Structure describing PERF_CF_SENSOR_PGTIME static information/POR.
 */
typedef struct LW2080_CTRL_PERF_PERF_CF_SENSOR_INFO_PGTIME {
    /*!
     * Power gating engine ID to sample.
     * @ref LW2080_CTRL_MC_POWERGATING_ENGINE_ID_<xyz>.
     */
    LwU8 pgEngineId;
} LW2080_CTRL_PERF_PERF_CF_SENSOR_INFO_PGTIME;
typedef struct LW2080_CTRL_PERF_PERF_CF_SENSOR_INFO_PGTIME *PLW2080_CTRL_PERF_PERF_CF_SENSOR_INFO_PGTIME;

/*!
 * Structure describing PERF_CF_SENSOR_THERM_MONITOR static information/POR.
 */
typedef struct LW2080_CTRL_PERF_PERF_CF_SENSOR_INFO_THERM_MONITOR {
    /*!
     * Thermal monitor object index to sample.
     */
    LwU8 thrmMonIdx;
} LW2080_CTRL_PERF_PERF_CF_SENSOR_INFO_THERM_MONITOR;
typedef struct LW2080_CTRL_PERF_PERF_CF_SENSOR_INFO_THERM_MONITOR *PLW2080_CTRL_PERF_PERF_CF_SENSOR_INFO_THERM_MONITOR;

/*!
 * PERF_CF_SENSOR type-specific data union. Discriminated by
 * PERF_CF_SENSOR::super.type.
 */


/*!
 * Structure describing PERF_CF_SENSOR static information/POR.
 * Implements the BOARDOBJ model/interface.
 */
typedef struct LW2080_CTRL_PERF_PERF_CF_SENSOR_INFO {
    /*!
     * LW2080_CTRL_BOARDOBJ super class.
     * Must always be first object in structure.
     */
    LW2080_CTRL_BOARDOBJ super;
    /*!
     * XAPI does not support using an element within another structure as a
     * discriminant. Placing this redundant type value here until that design
     * constraint can be fixed.
     */
    LwU8                 type;
    /*!
     * Type-specific data union.
     */
    union {
        LW2080_CTRL_PERF_PERF_CF_SENSOR_INFO_PMU           pmu;
        LW2080_CTRL_PERF_PERF_CF_SENSOR_INFO_PMU_FB        pmuFb;
        LW2080_CTRL_PERF_PERF_CF_SENSOR_INFO_PEX           pex;
        LW2080_CTRL_PERF_PERF_CF_SENSOR_INFO_PTIMER        ptimer;
        LW2080_CTRL_PERF_PERF_CF_SENSOR_INFO_PTIMER_CLK    ptimerClk;
        LW2080_CTRL_PERF_PERF_CF_SENSOR_INFO_CLKCNTR       clkcntr;
        LW2080_CTRL_PERF_PERF_CF_SENSOR_INFO_PGTIME        pgtime;
        LW2080_CTRL_PERF_PERF_CF_SENSOR_INFO_THERM_MONITOR thermMonitor;
    } data;
} LW2080_CTRL_PERF_PERF_CF_SENSOR_INFO;
typedef struct LW2080_CTRL_PERF_PERF_CF_SENSOR_INFO *PLW2080_CTRL_PERF_PERF_CF_SENSOR_INFO;

/*!
 * Structure describing PERF_CF_SENSOR static information/POR.
 * Implements the BOARDOBJGRP model/interface.
 */
#define LW2080_CTRL_PERF_PERF_CF_SENSORS_INFO_MESSAGE_ID (0xC0U)

typedef struct LW2080_CTRL_PERF_PERF_CF_SENSORS_INFO {
    /*!
     * LW2080_CTRL_BOARDOBJGRP_E32 super class.
     * Must always be first object in structure.
     */
    LW2080_CTRL_BOARDOBJGRP_E32          super;
    /*!
     * Array of PERF_CF_SENSOR structures. Has valid indexes corresponding to
     * the bits set in @ref super.objMask.
     */
    LW2080_CTRL_PERF_PERF_CF_SENSOR_INFO sensors[LW2080_CTRL_PERF_PERF_CF_SENSORS_MAX_OBJECTS];
} LW2080_CTRL_PERF_PERF_CF_SENSORS_INFO;
typedef struct LW2080_CTRL_PERF_PERF_CF_SENSORS_INFO *PLW2080_CTRL_PERF_PERF_CF_SENSORS_INFO;

/*!
 * LW2080_CTRL_CMD_PERF_PERF_CF_SENSORS_GET_INFO
 *
 * This command returns PERF_CF_SENSORS static object information/POR as
 * specified by the VBIOS in PERF_CF Table.
 *
 * See @ref LW2080_CTRL_PERF_PERF_CF_SENSORS_INFO for documentation on
 * the parameters.
 *
 * Return values are specified per @ref BoardObjGrpGetInfo.
 */
#define LW2080_CTRL_CMD_PERF_PERF_CF_SENSORS_GET_INFO (0x208020c0) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | LW2080_CTRL_PERF_PERF_CF_SENSORS_INFO_MESSAGE_ID" */


/*!
 * Structure representing the dynamic state of PERF_CF_SENSOR_PMU.
 */
typedef struct LW2080_CTRL_PERF_PERF_CF_SENSOR_STATUS_PMU {
    /*!
     * Last read HW counter value.
     */
    LwU32 last;
} LW2080_CTRL_PERF_PERF_CF_SENSOR_STATUS_PMU;
typedef struct LW2080_CTRL_PERF_PERF_CF_SENSOR_STATUS_PMU *PLW2080_CTRL_PERF_PERF_CF_SENSOR_STATUS_PMU;

/*!
 * Structure representing the dynamic state of PERF_CF_SENSOR_PMU_FB.
 */
typedef struct LW2080_CTRL_PERF_PERF_CF_SENSOR_STATUS_PMU_FB {
    /*!
     * PERF_CF_SENSOR_PMU super class.
     * Must always be first element in the structure.
     */
    LW2080_CTRL_PERF_PERF_CF_SENSOR_STATUS_PMU super;
    /*!
     * Last read HW clock counter value for multiplier.
     */
    LW_DECLARE_ALIGNED(LwU64 lastMulti, 8);
    /*!
     * Last read HW clock counter value for divisor.
     */
    LW_DECLARE_ALIGNED(LwU64 lastDiv, 8);
} LW2080_CTRL_PERF_PERF_CF_SENSOR_STATUS_PMU_FB;
typedef struct LW2080_CTRL_PERF_PERF_CF_SENSOR_STATUS_PMU_FB *PLW2080_CTRL_PERF_PERF_CF_SENSOR_STATUS_PMU_FB;

/*!
 * Structure representing the dynamic state of PERF_CF_SENSOR_PEX.
 */
typedef struct LW2080_CTRL_PERF_PERF_CF_SENSOR_STATUS_PEX {
    /*!
     * Last read HW counter value.
     */
    LwU32 last;
} LW2080_CTRL_PERF_PERF_CF_SENSOR_STATUS_PEX;
typedef struct LW2080_CTRL_PERF_PERF_CF_SENSOR_STATUS_PEX *PLW2080_CTRL_PERF_PERF_CF_SENSOR_STATUS_PEX;

/*!
 * Structure representing the dynamic state of PERF_CF_SENSOR_PTIMER.
 */
typedef struct LW2080_CTRL_PERF_PERF_CF_SENSOR_STATUS_PTIMER {
    /*!
     * Lwrrently we do NOT have any dynamic state of this class.
     */
    LwU8 rsvd;
} LW2080_CTRL_PERF_PERF_CF_SENSOR_STATUS_PTIMER;
typedef struct LW2080_CTRL_PERF_PERF_CF_SENSOR_STATUS_PTIMER *PLW2080_CTRL_PERF_PERF_CF_SENSOR_STATUS_PTIMER;

/*!
 * Structure representing the dynamic state of PERF_CF_SENSOR_PTIMER_CLK.
 */
typedef struct LW2080_CTRL_PERF_PERF_CF_SENSOR_STATUS_PTIMER_CLK {
    /*!
     * PERF_CF_SENSOR_PTIMER super class.
     * Must always be first element in the structure.
     */
    LW2080_CTRL_PERF_PERF_CF_SENSOR_STATUS_PTIMER super;
} LW2080_CTRL_PERF_PERF_CF_SENSOR_STATUS_PTIMER_CLK;
typedef struct LW2080_CTRL_PERF_PERF_CF_SENSOR_STATUS_PTIMER_CLK *PLW2080_CTRL_PERF_PERF_CF_SENSOR_STATUS_PTIMER_CLK;

/*!
 * Structure representing the dynamic state of PERF_CF_SENSOR_CLKCNTR.
 */
typedef struct LW2080_CTRL_PERF_PERF_CF_SENSOR_STATUS_CLKCNTR {
    /*!
     * Lwrrently we do NOT have any dynamic state of this class.
     */
    LwU8 rsvd;
} LW2080_CTRL_PERF_PERF_CF_SENSOR_STATUS_CLKCNTR;
typedef struct LW2080_CTRL_PERF_PERF_CF_SENSOR_STATUS_CLKCNTR *PLW2080_CTRL_PERF_PERF_CF_SENSOR_STATUS_CLKCNTR;

/*!
 * Structure representing the dynamic state of PERF_CF_SENSOR_PGTIME.
 */
typedef struct LW2080_CTRL_PERF_PERF_CF_SENSOR_STATUS_PGTIME {
    /*!
     * Lwrrently we do NOT have any dynamic state of this class.
     */
    LwU8 rsvd;
} LW2080_CTRL_PERF_PERF_CF_SENSOR_STATUS_PGTIME;
typedef struct LW2080_CTRL_PERF_PERF_CF_SENSOR_STATUS_PGTIME *PLW2080_CTRL_PERF_PERF_CF_SENSOR_STATUS_PGTIME;

/*!
 * Structure representing the dynamic state of PERF_CF_SENSOR_THERM_MONITOR.
 */
typedef struct LW2080_CTRL_PERF_PERF_CF_SENSOR_STATUS_THERM_MONITOR {
    /*!
     * Lwrrently we do NOT have any dynamic state of this class.
     */
    LwU8 rsvd;
} LW2080_CTRL_PERF_PERF_CF_SENSOR_STATUS_THERM_MONITOR;
typedef struct LW2080_CTRL_PERF_PERF_CF_SENSOR_STATUS_THERM_MONITOR *PLW2080_CTRL_PERF_PERF_CF_SENSOR_STATUS_THERM_MONITOR;

/*!
 * PERF_CF_SENSOR type-specific data union. Discriminated by
 * PERF_CF_SENSOR::super.type.
 */


/*!
 * Structure representing the dynamic state of each PERF_CF_SENSOR.
 * Implements the BOARDOBJ model/interface.
 */
typedef struct LW2080_CTRL_PERF_PERF_CF_SENSOR_STATUS {
    /*!
     * LW2080_CTRL_BOARDOBJ super class.
     * Must always be first object in structure.
     */
    LW2080_CTRL_BOARDOBJ super;
    /*!
     * XAPI does not support using an element within another structure as a
     * discriminant. Placing this redundant type value here until that design
     * constraint can be fixed.
     */
    LwU8                 type;
    /*!
     * Common reading for all sensor status.
     */
    LW_DECLARE_ALIGNED(LwU64 reading, 8);
    /*!
     * Type-specific data union.
     */
    union {
        LW2080_CTRL_PERF_PERF_CF_SENSOR_STATUS_PMU           pmu;
        LW_DECLARE_ALIGNED(LW2080_CTRL_PERF_PERF_CF_SENSOR_STATUS_PMU_FB pmuFb, 8);
        LW2080_CTRL_PERF_PERF_CF_SENSOR_STATUS_PEX           pex;
        LW2080_CTRL_PERF_PERF_CF_SENSOR_STATUS_PTIMER        ptimer;
        LW2080_CTRL_PERF_PERF_CF_SENSOR_STATUS_PTIMER_CLK    ptimerClk;
        LW2080_CTRL_PERF_PERF_CF_SENSOR_STATUS_CLKCNTR       clkcntr;
        LW2080_CTRL_PERF_PERF_CF_SENSOR_STATUS_PGTIME        pgtime;
        LW2080_CTRL_PERF_PERF_CF_SENSOR_STATUS_THERM_MONITOR thermMonitor;
    } data;
} LW2080_CTRL_PERF_PERF_CF_SENSOR_STATUS;
typedef struct LW2080_CTRL_PERF_PERF_CF_SENSOR_STATUS *PLW2080_CTRL_PERF_PERF_CF_SENSOR_STATUS;

/*!
 * Structure representing the dynamic state of PERF_CF_SENSORS.
 * Implements the BOARDOBJGRP model/interface.
 */
#define LW2080_CTRL_PERF_PERF_CF_SENSORS_STATUS_MESSAGE_ID (0xC1U)

typedef struct LW2080_CTRL_PERF_PERF_CF_SENSORS_STATUS {
    /*!
     * LW2080_CTRL_BOARDOBJGRP_E32 super class.
     * Must always be first object in structure.
     */
    LW2080_CTRL_BOARDOBJGRP_E32 super;
    /*!
     * Array of PERF_CF_SENSOR structures. Has valid indexes corresponding to
     * the bits set in @ref super.objMask.
     */
    LW_DECLARE_ALIGNED(LW2080_CTRL_PERF_PERF_CF_SENSOR_STATUS sensors[LW2080_CTRL_PERF_PERF_CF_SENSORS_MAX_OBJECTS], 8);
} LW2080_CTRL_PERF_PERF_CF_SENSORS_STATUS;
typedef struct LW2080_CTRL_PERF_PERF_CF_SENSORS_STATUS *PLW2080_CTRL_PERF_PERF_CF_SENSORS_STATUS;

/*!
 * LW2080_CTRL_CMD_PERF_PERF_CF_SENSORS_GET_STATUS
 *
 * This command returns PERF_CF_SENSORS dynamic state.
 *
 * See @ref LW2080_CTRL_PERF_PERF_CF_SENSORS_STATUS for documentation on
 * the parameters.
 *
 * Return values are specified per @ref BoardObjGrpGetInfo.
 */
#define LW2080_CTRL_CMD_PERF_PERF_CF_SENSORS_GET_STATUS (0x208020c1) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | LW2080_CTRL_PERF_PERF_CF_SENSORS_STATUS_MESSAGE_ID" */


/*!
 * Structure representing the control parameters of PERF_CF_SENSOR_PMU.
 */
typedef struct LW2080_CTRL_PERF_PERF_CF_SENSOR_CONTROL_PMU {
    /*!
     * Lwrrently we do NOT have any control parameter in this class.
     */
    LwU8 rsvd;
} LW2080_CTRL_PERF_PERF_CF_SENSOR_CONTROL_PMU;
typedef struct LW2080_CTRL_PERF_PERF_CF_SENSOR_CONTROL_PMU *PLW2080_CTRL_PERF_PERF_CF_SENSOR_CONTROL_PMU;

/*!
 * Structure representing the control parameters of PERF_CF_SENSOR_PMU_FB.
 */
typedef struct LW2080_CTRL_PERF_PERF_CF_SENSOR_CONTROL_PMU_FB {
    /*!
     * PERF_CF_SENSOR_PMU super class.
     * Must always be first element in the structure.
     */
    LW2080_CTRL_PERF_PERF_CF_SENSOR_CONTROL_PMU super;
} LW2080_CTRL_PERF_PERF_CF_SENSOR_CONTROL_PMU_FB;
typedef struct LW2080_CTRL_PERF_PERF_CF_SENSOR_CONTROL_PMU_FB *PLW2080_CTRL_PERF_PERF_CF_SENSOR_CONTROL_PMU_FB;

/*!
 * Structure representing the control parameters of PERF_CF_SENSOR_PEX.
 */
typedef struct LW2080_CTRL_PERF_PERF_CF_SENSOR_CONTROL_PEX {
    /*!
     * Lwrrently we do NOT have any control parameter in this class.
     */
    LwU8 rsvd;
} LW2080_CTRL_PERF_PERF_CF_SENSOR_CONTROL_PEX;
typedef struct LW2080_CTRL_PERF_PERF_CF_SENSOR_CONTROL_PEX *PLW2080_CTRL_PERF_PERF_CF_SENSOR_CONTROL_PEX;

/*!
 * Structure representing the control parameters of PERF_CF_SENSOR_PTIMER.
 */
typedef struct LW2080_CTRL_PERF_PERF_CF_SENSOR_CONTROL_PTIMER {
    /*!
     * Lwrrently we do NOT have any control parameter in this class.
     */
    LwU8 rsvd;
} LW2080_CTRL_PERF_PERF_CF_SENSOR_CONTROL_PTIMER;
typedef struct LW2080_CTRL_PERF_PERF_CF_SENSOR_CONTROL_PTIMER *PLW2080_CTRL_PERF_PERF_CF_SENSOR_CONTROL_PTIMER;

/*!
 * Structure representing the control parameters of PERF_CF_SENSOR_PTIMER_CLK.
 */
typedef struct LW2080_CTRL_PERF_PERF_CF_SENSOR_CONTROL_PTIMER_CLK {
    /*!
     * PERF_CF_SENSOR_PTIMER super class.
     * Must always be first element in the structure.
     */
    LW2080_CTRL_PERF_PERF_CF_SENSOR_CONTROL_PTIMER super;
} LW2080_CTRL_PERF_PERF_CF_SENSOR_CONTROL_PTIMER_CLK;
typedef struct LW2080_CTRL_PERF_PERF_CF_SENSOR_CONTROL_PTIMER_CLK *PLW2080_CTRL_PERF_PERF_CF_SENSOR_CONTROL_PTIMER_CLK;

/*!
 * Structure representing the control parameters of PERF_CF_SENSOR_CLKCNTR.
 */
typedef struct LW2080_CTRL_PERF_PERF_CF_SENSOR_CONTROL_CLKCNTR {
    /*!
     * Lwrrently we do NOT have any control parameter in this class.
     */
    LwU8 rsvd;
} LW2080_CTRL_PERF_PERF_CF_SENSOR_CONTROL_CLKCNTR;
typedef struct LW2080_CTRL_PERF_PERF_CF_SENSOR_CONTROL_CLKCNTR *PLW2080_CTRL_PERF_PERF_CF_SENSOR_CONTROL_CLKCNTR;

/*!
 * Structure representing the control parameters of PERF_CF_SENSOR_PGTIME.
 */
typedef struct LW2080_CTRL_PERF_PERF_CF_SENSOR_CONTROL_PGTIME {
    /*!
     * Lwrrently we do NOT have any control parameter in this class.
     */
    LwU8 rsvd;
} LW2080_CTRL_PERF_PERF_CF_SENSOR_CONTROL_PGTIME;
typedef struct LW2080_CTRL_PERF_PERF_CF_SENSOR_CONTROL_PGTIME *PLW2080_CTRL_PERF_PERF_CF_SENSOR_CONTROL_PGTIME;

/*!
 * Structure representing the control parameters of PERF_CF_SENSOR_THERM_MONITOR.
 */
typedef struct LW2080_CTRL_PERF_PERF_CF_SENSOR_CONTROL_THERM_MONITOR {
    /*!
     * Lwrrently we do NOT have any control parameter in this class.
     */
    LwU8 rsvd;
} LW2080_CTRL_PERF_PERF_CF_SENSOR_CONTROL_THERM_MONITOR;
typedef struct LW2080_CTRL_PERF_PERF_CF_SENSOR_CONTROL_THERM_MONITOR *PLW2080_CTRL_PERF_PERF_CF_SENSOR_CONTROL_THERM_MONITOR;

/*!
 * PERF_CF_SENSOR type-specific data union. Discriminated by
 * PERF_CF_SENSOR::super.type.
 */


/*!
 * Structure representing the control parameters of each PERF_CF_SENSOR.
 * Implements the BOARDOBJ model/interface.
 */
typedef struct LW2080_CTRL_PERF_PERF_CF_SENSOR_CONTROL {
    /*!
     * LW2080_CTRL_BOARDOBJ super class.
     * Must always be first object in structure.
     */
    LW2080_CTRL_BOARDOBJ super;
    /*!
     * XAPI does not support using an element within another structure as a
     * discriminant. Placing this redundant type value here until that design
     * constraint can be fixed.
     */
    LwU8                 type;
    /*!
     * Type-specific data union.
     */
    union {
        LW2080_CTRL_PERF_PERF_CF_SENSOR_CONTROL_PMU           pmu;
        LW2080_CTRL_PERF_PERF_CF_SENSOR_CONTROL_PMU_FB        pmuFb;
        LW2080_CTRL_PERF_PERF_CF_SENSOR_CONTROL_PEX           pex;
        LW2080_CTRL_PERF_PERF_CF_SENSOR_CONTROL_PTIMER        ptimer;
        LW2080_CTRL_PERF_PERF_CF_SENSOR_CONTROL_PTIMER_CLK    ptimerClk;
        LW2080_CTRL_PERF_PERF_CF_SENSOR_CONTROL_CLKCNTR       clkcntr;
        LW2080_CTRL_PERF_PERF_CF_SENSOR_CONTROL_PGTIME        pgtime;
        LW2080_CTRL_PERF_PERF_CF_SENSOR_CONTROL_THERM_MONITOR thermMonitor;
    } data;
} LW2080_CTRL_PERF_PERF_CF_SENSOR_CONTROL;
typedef struct LW2080_CTRL_PERF_PERF_CF_SENSOR_CONTROL *PLW2080_CTRL_PERF_PERF_CF_SENSOR_CONTROL;

/*!
 * Structure representing the control parameters of PERF_CF_SENSORS.
 * Implements the BOARDOBJ model/interface.
 */
typedef struct LW2080_CTRL_PERF_PERF_CF_SENSORS_CONTROL {
    /*!
     * LW2080_CTRL_BOARDOBJGRP_E32 super class.
     * Must always be first object in structure.
     */
    LW2080_CTRL_BOARDOBJGRP_E32             super;
    /*!
     * Array of PERF_CF_SENSOR structures. Has valid indexes corresponding to
     * the bits set in @ref super.objMask.
     */
    LW2080_CTRL_PERF_PERF_CF_SENSOR_CONTROL sensors[LW2080_CTRL_PERF_PERF_CF_SENSORS_MAX_OBJECTS];
} LW2080_CTRL_PERF_PERF_CF_SENSORS_CONTROL;
typedef struct LW2080_CTRL_PERF_PERF_CF_SENSORS_CONTROL *PLW2080_CTRL_PERF_PERF_CF_SENSORS_CONTROL;

/*!
 * LW2080_CTRL_CMD_PERF_PERF_CF_SENSORS_GET_CONTROL
 *
 * This command returns current PERF_CF_SENSORS control parameters.
 *
 * See @ref LW2080_CTRL_PERF_PERF_CF_SENSORS_CONTROL for documentation on
 * the parameters.
 *
 * Return values are specified per @ref BoardObjGrpGetControl.
 */
#define LW2080_CTRL_CMD_PERF_PERF_CF_SENSORS_GET_CONTROL           (0x208020c2) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | 0xC2" */


/*!
 * LW2080_CTRL_CMD_PERF_PERF_CF_SENSORS_SET_CONTROL
 *
 * This command accepts client-specified control parameters for a set of
 * PERF_CF_SENSORS and applies these new parameters.
 *
 * See @ref LW2080_CTRL_PERF_PERF_CF_SENSORS_CONTROL for documentation on
 * the parameters.
 *
 * Return values are specified per @ref BoardObjGrpSetControl.
 */
#define LW2080_CTRL_CMD_PERF_PERF_CF_SENSORS_SET_CONTROL           (0x208020c3) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | 0xC3" */

/* ---------------------------- PERF_CF PM Sensors --------------------------- */
/*!
 * Macros for PERF_CF_PM_SENSOR types
 */
#define LW2080_CTRL_PERF_PERF_CF_PM_SENSOR_V00                     0x00
#define LW2080_CTRL_PERF_PERF_CF_PM_SENSOR_V10                     0x01
#define LW2080_CTRL_PERF_PERF_CF_PM_SENSOR_CLW_DEV_V10             0x02
#define LW2080_CTRL_PERF_PERF_CF_PM_SENSOR_CLW_MIG_V10             0x03

/*!
 * Maximum number of GPCs in a GPU.
 */
#define LW2080_CTRL_PERF_PERF_CF_PM_SENSOR_CLW_V10_GPC_NUM         0x08

/*!
 * Maximum number of TPCs in a GPC.
 */
#define LW2080_CTRL_PERF_PERF_CF_PM_SENSOR_CLW_V10_TPC_NUM         0x09

/*!
 * Maximum number of FBPs in a GPU.
 */
#define LW2080_CTRL_PERF_PERF_CF_PM_SENSOR_CLW_V10_FBP_NUM         0x0C

/*!
 * Maximum number of LTSPs in a FBP.
 */
#define LW2080_CTRL_PERF_PERF_CF_PM_SENSOR_CLW_V10_FBP_LTSP_NUM    0x04

/*
 * Maximum number of SYS sections in device OOBRC.
 */
#define LW2080_CTRL_PERF_PERF_CF_PM_SENSOR_CLW_V10_SYS_NUM         0x5

/*!
 * Maximum number of MIGs in a GPU.
 */
#define LW2080_CTRL_PERF_PERF_CF_PM_SENSOR_CLW_V10_MIG_NUM         0x08

/*!
 * Invalid swizzId of MIG.
 */
#define LW2080_CTRL_PERF_PERF_CF_PM_SENSOR_CLW_MIG_ILWALID_SWIZZID 0xFF

/*!
 * Structure storing the configuration masks of TPC/FBP/SYS.
 */
typedef struct LW2080_CTRL_PERF_PERF_CF_PM_SENSOR_CLW_V10_CFG {
    /*!
     * GPC enabled masks that need to SNAP.
     */
    LwU8  gpcMask;

    /*!
     * TPC SNAP mask array.
     */
    LwU16 gpcTpcMask[LW2080_CTRL_PERF_PERF_CF_PM_SENSOR_CLW_V10_GPC_NUM];

    /*!
     * FPB enabled masks that need to SNAP.
     */
    LwU16 fbpMask;

    /*!
     * LTSP SNAP mask array.
     */
    LwU8  fbpLtspMask[LW2080_CTRL_PERF_PERF_CF_PM_SENSOR_CLW_V10_FBP_NUM];

    /*!
     * SYS enabled masks that need to SNAP.
     */
    LwU8  sysMask;
} LW2080_CTRL_PERF_PERF_CF_PM_SENSOR_CLW_V10_CFG;
typedef struct LW2080_CTRL_PERF_PERF_CF_PM_SENSOR_CLW_V10_CFG *PLW2080_CTRL_PERF_PERF_CF_PM_SENSOR_CLW_V10_CFG;

/*!
 * Maximum number of signals supported by any PERF_CF_PM_SIGNAL
 * implementation class.
 */
#define LW2080_CTRL_PERF_PERF_CF_PM_SENSOR_MAX_SIGNALS  LW2080_CTRL_BOARDOBJGRP_E1024_MAX_OBJECTS

 /*!
 * Maximum number of PERF_CF_PM_SENSORs which can be supported in RM or PMU.
 */
#define LW2080_CTRL_PERF_PERF_CF_PM_SENSORS_MAX_OBJECTS LW2080_CTRL_BOARDOBJGRP_E32_MAX_OBJECTS
/*!
 * Structure type for specifying a mask of PERF_CF_PM_SENSOR signals.
 * Used in various API to specify sets of individual PM signals.
 */
typedef LW2080_CTRL_BOARDOBJGRP_MASK_E1024 LW2080_CTRL_PERF_PERF_CF_PM_SENSOR_SIGNAL_MASK;

/*!
 * Structure describing PERF_CF_PM_SENSOR_V00 static information/POR.
 */
typedef struct LW2080_CTRL_PERF_PERF_CF_PM_SENSOR_INFO_V00 {
    /*!
     * Reserving space.  Actual parameters TBD.
     */
    LwU32 rsvd;
} LW2080_CTRL_PERF_PERF_CF_PM_SENSOR_INFO_V00;
typedef struct LW2080_CTRL_PERF_PERF_CF_PM_SENSOR_INFO_V00 *PLW2080_CTRL_PERF_PERF_CF_PM_SENSOR_INFO_V00;

/*!
 * Structure describing PERF_CF_PM_SENSOR_V10 static information/POR.
 */
typedef struct LW2080_CTRL_PERF_PERF_CF_PM_SENSOR_INFO_V10 {
    /*!
     * Reserving space.  Actual parameters TBD.
     */
    LwU32 rsvd;
} LW2080_CTRL_PERF_PERF_CF_PM_SENSOR_INFO_V10;
typedef struct LW2080_CTRL_PERF_PERF_CF_PM_SENSOR_INFO_V10 *PLW2080_CTRL_PERF_PERF_CF_PM_SENSOR_INFO_V10;

/*!
 * Structure describing PERF_CF_PM_SENSOR_CLW_DEV_V10 static information/POR.
 */
typedef struct LW2080_CTRL_PERF_PERF_CF_PM_SENSOR_INFO_CLW_DEV_V10 {
    /*!
     * Reserving space.  Actual parameters TBD.
     */
    LwU32 rsvd;
} LW2080_CTRL_PERF_PERF_CF_PM_SENSOR_INFO_CLW_DEV_V10;
typedef struct LW2080_CTRL_PERF_PERF_CF_PM_SENSOR_INFO_CLW_DEV_V10 *PLW2080_CTRL_PERF_PERF_CF_PM_SENSOR_INFO_CLW_DEV_V10;

/*!
 * Structure describing PERF_CF_PM_SENSOR_CLW_MIG_V10 static information/POR.
 */
typedef struct LW2080_CTRL_PERF_PERF_CF_PM_SENSOR_INFO_CLW_MIG_V10 {
    /*!
     * The ID of the MIG object, It is set
     * by RM with the value of MIG's swizzId during MIG creation
     */
    LwU8 swizzId;
} LW2080_CTRL_PERF_PERF_CF_PM_SENSOR_INFO_CLW_MIG_V10;
typedef struct LW2080_CTRL_PERF_PERF_CF_PM_SENSOR_INFO_CLW_MIG_V10 *PLW2080_CTRL_PERF_PERF_CF_PM_SENSOR_INFO_CLW_MIG_V10;

/*!
 * PERF_CF_PM_SENSOR type-specific data union. Discriminated by
 * PERF_CF_PM_SENSOR::super.type.
 */


/*!
 * Structure describing PERF_CF_PM_SENSOR static information/POR.
 * Implements the BOARDOBJ model/interface.
 */
typedef struct LW2080_CTRL_PERF_PERF_CF_PM_SENSOR_INFO {
    /*!
     * LW2080_CTRL_BOARDOBJ super class.
     * Must always be first object in structure.
     */
    LW2080_CTRL_BOARDOBJ super;
    /*!
     * XAPI does not support using an element within another structure as a
     * discriminant. Placing this redundant type value here until that design
     * constraint can be fixed.
     */
    LwU8                 type;
    /*!
     * Type-specific data union.
     */
    union {
        LW2080_CTRL_PERF_PERF_CF_PM_SENSOR_INFO_V00         pmV00;
        LW2080_CTRL_PERF_PERF_CF_PM_SENSOR_INFO_V10         pmV10;
        LW2080_CTRL_PERF_PERF_CF_PM_SENSOR_INFO_CLW_DEV_V10 clw_dev_V10;
        LW2080_CTRL_PERF_PERF_CF_PM_SENSOR_INFO_CLW_MIG_V10 clw_mig_V10;
    } data;
    /*!
     * Mask of signals which this PERF_CF_PM_SENSOR supports.
     */
    LW2080_CTRL_PERF_PERF_CF_PM_SENSOR_SIGNAL_MASK signalsSupportedMask;
} LW2080_CTRL_PERF_PERF_CF_PM_SENSOR_INFO;
typedef struct LW2080_CTRL_PERF_PERF_CF_PM_SENSOR_INFO *PLW2080_CTRL_PERF_PERF_CF_PM_SENSOR_INFO;

/*!
 * Structure describing PERF_CF_SENSOR static information/POR.
 * Implements the BOARDOBJGRP model/interface.
 */
#define LW2080_CTRL_PERF_PERF_CF_PM_SENSORS_INFO_MESSAGE_ID (0xE0U)

typedef struct LW2080_CTRL_PERF_PERF_CF_PM_SENSORS_INFO {
    /*!
     * LW2080_CTRL_BOARDOBJGRP_E32 super class.
     * Must always be first object in structure.
     */
    LW2080_CTRL_BOARDOBJGRP_E32             super;
    /*!
     * Array of PERF_CF_PM_SENSOR structures. Has valid indexes corresponding to
     * the bits set in @ref super.objMask.
     */
    LW2080_CTRL_PERF_PERF_CF_PM_SENSOR_INFO pmSensors[LW2080_CTRL_PERF_PERF_CF_PM_SENSORS_MAX_OBJECTS];
} LW2080_CTRL_PERF_PERF_CF_PM_SENSORS_INFO;
typedef struct LW2080_CTRL_PERF_PERF_CF_PM_SENSORS_INFO *PLW2080_CTRL_PERF_PERF_CF_PM_SENSORS_INFO;

/*!
 * LW2080_CTRL_CMD_PERF_PERF_CF_PM_SENSORS_GET_INFO
 *
 * This command returns PERF_CF_PM_SENSORS static object information/POR as
 * specified by the VBIOS in PERF_CF Table.
 *
 * See @ref LW2080_CTRL_PERF_PERF_CF_PM_SENSORS_INFO for documentation on
 * the parameters.
 *
 * Return values are specified per @ref BoardObjGrpGetInfo.
 */
#define LW2080_CTRL_CMD_PERF_PERF_CF_PM_SENSORS_GET_INFO (0x208020e0) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | LW2080_CTRL_PERF_PERF_CF_PM_SENSORS_INFO_MESSAGE_ID" */

/*!
 * Structure describing PERF_CF_PM_SENSOR_V00 dynamic status.
 */
typedef struct LW2080_CTRL_PERF_PERF_CF_PM_SENSOR_STATUS_V00 {
    /*!
     * Reserving space.  Actual parameters TBD.
     */
    LwU32 rsvd;
} LW2080_CTRL_PERF_PERF_CF_PM_SENSOR_STATUS_V00;
typedef struct LW2080_CTRL_PERF_PERF_CF_PM_SENSOR_STATUS_V00 *PLW2080_CTRL_PERF_PERF_CF_PM_SENSOR_STATUS_V00;

/*!
 * Structure describing PERF_CF_PM_SENSOR_V10 dynamic status.
 */
typedef struct LW2080_CTRL_PERF_PERF_CF_PM_SENSOR_STATUS_V10 {
    /*!
     * Reserving space.  Actual parameters TBD.
     */
    LwU32 rsvd;
} LW2080_CTRL_PERF_PERF_CF_PM_SENSOR_STATUS_V10;
typedef struct LW2080_CTRL_PERF_PERF_CF_PM_SENSOR_STATUS_V10 *PLW2080_CTRL_PERF_PERF_CF_PM_SENSOR_STATUS_V10;

/*!
 * Structure describing PERF_CF_PM_SENSOR_CLW_DEV_V10 dynamic status.
 */
typedef struct LW2080_CTRL_PERF_PERF_CF_PM_SENSOR_STATUS_CLW_DEV_V10 {
    /*!
     * Reserving space.  Actual parameters TBD.
     */
    LwU32 rsvd;
} LW2080_CTRL_PERF_PERF_CF_PM_SENSOR_STATUS_CLW_DEV_V10;
typedef struct LW2080_CTRL_PERF_PERF_CF_PM_SENSOR_STATUS_CLW_DEV_V10 *PLW2080_CTRL_PERF_PERF_CF_PM_SENSOR_STATUS_CLW_DEV_V10;

/*!
 * Structure describing PERF_CF_PM_SENSOR_CLW_MIG_V10 dynamic status.
 */
typedef struct LW2080_CTRL_PERF_PERF_CF_PM_SENSOR_STATUS_CLW_MIG_V10 {
    /*!
     * Reserving space.  Actual parameters TBD.
     */
    LwU32 rsvd;
} LW2080_CTRL_PERF_PERF_CF_PM_SENSOR_STATUS_CLW_MIG_V10;
typedef struct LW2080_CTRL_PERF_PERF_CF_PM_SENSOR_STATUS_CLW_MIG_V10 *PLW2080_CTRL_PERF_PERF_CF_PM_SENSOR_STATUS_CLW_MIG_V10;

/*!
 * PERF_CF_PM_SENSOR type-specific data union. Discriminated by
 * PERF_CF_PM_SENSOR::super.type.
 */


/*!
 * Structure representing the status a given PERF_CF_PM_SENSOR signal
 * - i.e. a single counted event in the PM sensor.
 */
typedef struct LW2080_CTRL_PERF_PERF_CF_PM_SENSOR_SIGNAL_STATUS {
  /*!
   * PERF_CF_PM_SENSOR signal's last count.  A 64-bit continuously
   * aclwmulating/incrementing counter, which will wrap-around on
   * overflow.
   *
   * On input, this is the current count for the last time this signal
   * was sampled by the given client via the GET_STATUS API.
   *
   * On output, this is the current count of the signal.
   */
    LwU64_ALIGN32 cntLast;
  /*!
   * PERF_CF_PM_SENSOR signal's count difference since the last call.
   *
   * On output, this is the difference between the current cntLast and
   * the last cntLast.  This provides an count of the given signal
   * aligned to the timescale of the caller.
   */
    LwU64_ALIGN32 cntDiff;
} LW2080_CTRL_PERF_PERF_CF_PM_SENSOR_SIGNAL_STATUS;
typedef struct LW2080_CTRL_PERF_PERF_CF_PM_SENSOR_SIGNAL_STATUS *PLW2080_CTRL_PERF_PERF_CF_PM_SENSOR_SIGNAL_STATUS;

/*!
 * Structure describing PERF_CF_PM_SENSOR static information/POR.
 * Implements the BOARDOBJ model/interface.
 */
typedef struct LW2080_CTRL_PERF_PERF_CF_PM_SENSOR_STATUS {
    /*!
     * LW2080_CTRL_BOARDOBJ super class.
     * Must always be first object in structure.
     */
    LW2080_CTRL_BOARDOBJ super;
    /*!
     * XAPI does not support using an element within another structure as a
     * discriminant. Placing this redundant type value here until that design
     * constraint can be fixed.
     */
    LwU8                 type;
    /*!
     * Type-specific data union.
     */
    union {
        LW2080_CTRL_PERF_PERF_CF_PM_SENSOR_STATUS_V00         pmV00;
        LW2080_CTRL_PERF_PERF_CF_PM_SENSOR_STATUS_V10         pmV10;
        LW2080_CTRL_PERF_PERF_CF_PM_SENSOR_STATUS_CLW_DEV_V10 clw_dev_V10;
        LW2080_CTRL_PERF_PERF_CF_PM_SENSOR_STATUS_CLW_MIG_V10 clw_mig_V10;
    } data;
    /*!
     * [in] Mask of signals which the client has requested
     * PERF_CF_PM_SENSOR status.  Must be a subset of @ref
     * LW2080_CTRL_PERF_PERF_CF_PM_SENSOR_INFO::signalsSupportedMask.
     */
    LW2080_CTRL_PERF_PERF_CF_PM_SENSOR_SIGNAL_MASK   signalsMask;
    /*!
     * [in/out] Array of individual PM signals status.  Indexed via
     * bits set in @ref signalsMask;
     */
    LW2080_CTRL_PERF_PERF_CF_PM_SENSOR_SIGNAL_STATUS signals[LW2080_CTRL_PERF_PERF_CF_PM_SENSOR_MAX_SIGNALS];
} LW2080_CTRL_PERF_PERF_CF_PM_SENSOR_STATUS;
typedef struct LW2080_CTRL_PERF_PERF_CF_PM_SENSOR_STATUS *PLW2080_CTRL_PERF_PERF_CF_PM_SENSOR_STATUS;

/*!
 * Structure representing the dynamic state of PERF_CF_PM_SENSORS.
 * Implements the BOARDOBJGRP model/interface.
 */
#define LW2080_CTRL_PERF_PERF_CF_PM_SENSORS_STATUS_MESSAGE_ID (0xE1U)

typedef struct LW2080_CTRL_PERF_PERF_CF_PM_SENSORS_STATUS {
    /*!
     * LW2080_CTRL_BOARDOBJGRP_E32 super class.
     * Must always be first object in structure.
     */
    LW2080_CTRL_BOARDOBJGRP_E32               super;
    /*!
     * Array of PERF_CF_SENSOR structures. Has valid indexes corresponding to
     * the bits set in @ref super.objMask.
     */
    LW2080_CTRL_PERF_PERF_CF_PM_SENSOR_STATUS pmSensors[LW2080_CTRL_PERF_PERF_CF_PM_SENSORS_MAX_OBJECTS];
} LW2080_CTRL_PERF_PERF_CF_PM_SENSORS_STATUS;
typedef struct LW2080_CTRL_PERF_PERF_CF_PM_SENSORS_STATUS *PLW2080_CTRL_PERF_PERF_CF_PM_SENSORS_STATUS;

/*!
 * LW2080_CTRL_CMD_PERF_PERF_CF_PM_SENSORS_GET_STATUS
 *
 * This command returns PERF_CF_PM_SENSORS dynamic state.
 *
 * See @ref LW2080_CTRL_PERF_PERF_CF_PM_SENSORS_STATUS for documentation on
 * the parameters.
 *
 * Return values are specified per @ref BoardObjGrpGetInfo.
 */
#define LW2080_CTRL_CMD_PERF_PERF_CF_PM_SENSORS_GET_STATUS (0x208020e1) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | LW2080_CTRL_PERF_PERF_CF_PM_SENSORS_STATUS_MESSAGE_ID" */

/*!
 * Structure representing the control parameters of PERF_CF_PM_SENSOR_V00.
 */
typedef struct LW2080_CTRL_PERF_PERF_CF_PM_SENSOR_CONTROL_V00 {
    /*!
     * Lwrrently we do NOT have any control parameter in this class.
     */
    LwU8 rsvd;
} LW2080_CTRL_PERF_PERF_CF_PM_SENSOR_CONTROL_V00;
typedef struct LW2080_CTRL_PERF_PERF_CF_PM_SENSOR_CONTROL_V00 *PLW2080_CTRL_PERF_PERF_CF_PM_SENSOR_CONTROL_V00;

/*!
 * Structure representing the control parameters of PERF_CF_PM_SENSOR_V10.
 */
typedef struct LW2080_CTRL_PERF_PERF_CF_PM_SENSOR_CONTROL_V10 {
    /*!
     * Lwrrently we do NOT have any control parameter in this class.
     */
    LwU8 rsvd;
} LW2080_CTRL_PERF_PERF_CF_PM_SENSOR_CONTROL_V10;
typedef struct LW2080_CTRL_PERF_PERF_CF_PM_SENSOR_CONTROL_V10 *PLW2080_CTRL_PERF_PERF_CF_PM_SENSOR_CONTROL_V10;

/*!
 * Structure representing the control parameters of PERF_CF_PM_SENSOR_CLW_DEV_V10.
 */
typedef struct LW2080_CTRL_PERF_PERF_CF_PM_SENSOR_CONTROL_CLW_DEV_V10 {
    /*!
     * Lwrrently we do NOT have any control parameter in this class.
     */
    LwU8 rsvd;
} LW2080_CTRL_PERF_PERF_CF_PM_SENSOR_CONTROL_CLW_DEV_V10;
typedef struct LW2080_CTRL_PERF_PERF_CF_PM_SENSOR_CONTROL_CLW_DEV_V10 *PLW2080_CTRL_PERF_PERF_CF_PM_SENSOR_CONTROL_CLW_DEV_V10;

/*!
 * Structure representing the control parameters of PERF_CF_PM_SENSOR_CLW_MIG_V10.
 */
typedef struct LW2080_CTRL_PERF_PERF_CF_PM_SENSOR_CONTROL_CLW_MIG_V10 {
    /*!
     * The swizzId of MIG to configure
     */
    LwU8                                           swizzId;
    /*!
     * Structure storing the MIG configuration (masks of TPC/FBP/SYS).
     */
    LW2080_CTRL_PERF_PERF_CF_PM_SENSOR_CLW_V10_CFG migCfg;
} LW2080_CTRL_PERF_PERF_CF_PM_SENSOR_CONTROL_CLW_MIG_V10;
typedef struct LW2080_CTRL_PERF_PERF_CF_PM_SENSOR_CONTROL_CLW_MIG_V10 *PLW2080_CTRL_PERF_PERF_CF_PM_SENSOR_CONTROL_CLW_MIG_V10;

/*!
 * PERF_CF_PM_SENSOR type-specific data union. Discriminated by
 * PERF_CF_PM_SENSOR::super.type.
 */


/*!
 * Structure representing the control parameters of each PERF_CF_PM_SENSOR.
 * Implements the BOARDOBJ model/interface.
 */
typedef struct LW2080_CTRL_PERF_PERF_CF_PM_SENSOR_CONTROL {
    /*!
     * LW2080_CTRL_BOARDOBJ super class.
     * Must always be first object in structure.
     */
    LW2080_CTRL_BOARDOBJ super;
    /*!
     * XAPI does not support using an element within another structure as a
     * discriminant. Placing this redundant type value here until that design
     * constraint can be fixed.
     */
    LwU8                 type;
    /*!
     * Type-specific data union.
     */
    union {
        LW2080_CTRL_PERF_PERF_CF_PM_SENSOR_CONTROL_V00         v00;
        LW2080_CTRL_PERF_PERF_CF_PM_SENSOR_CONTROL_V10         v10;
        LW2080_CTRL_PERF_PERF_CF_PM_SENSOR_CONTROL_CLW_DEV_V10 clw_dev_V10;
        LW2080_CTRL_PERF_PERF_CF_PM_SENSOR_CONTROL_CLW_MIG_V10 clw_mig_V10;
    } data;
} LW2080_CTRL_PERF_PERF_CF_PM_SENSOR_CONTROL;
typedef struct LW2080_CTRL_PERF_PERF_CF_PM_SENSOR_CONTROL *PLW2080_CTRL_PERF_PERF_CF_PM_SENSOR_CONTROL;

/*!
 * Structure representing the control parameters of PERF_CF_SENSORS.
 * Implements the BOARDOBJ model/interface.
 */
typedef struct LW2080_CTRL_PERF_PERF_CF_PM_SENSORS_CONTROL {
    /*!
     * LW2080_CTRL_BOARDOBJGRP_E32 super class.
     * Must always be first object in structure.
     */
    LW2080_CTRL_BOARDOBJGRP_E32                super;
    /*!
     * Array of PERF_CF_PM_SENSOR structures. Has valid indexes corresponding to
     * the bits set in @ref super.objMask.
     */
    LW2080_CTRL_PERF_PERF_CF_PM_SENSOR_CONTROL pmSensors[LW2080_CTRL_PERF_PERF_CF_PM_SENSORS_MAX_OBJECTS];
} LW2080_CTRL_PERF_PERF_CF_PM_SENSORS_CONTROL;
typedef struct LW2080_CTRL_PERF_PERF_CF_PM_SENSORS_CONTROL *PLW2080_CTRL_PERF_PERF_CF_PM_SENSORS_CONTROL;

/*!
 * LW2080_CTRL_CMD_PERF_PERF_CF_PM_SENSORS_GET_CONTROL
 *
 * This command returns current PERF_CF_PM_SENSORS control parameters.
 *
 * See @ref LW2080_CTRL_PERF_PERF_CF_PM_SENSORS_CONTROL for documentation on
 * the parameters.
 *
 * Return values are specified per @ref BoardObjGrpGetControl.
 */
#define LW2080_CTRL_CMD_PERF_PERF_CF_PM_SENSORS_GET_CONTROL   (0x208020e2) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | 0xE2" */


/*!
 * LW2080_CTRL_CMD_PERF_PERF_CF_PM_SENSORS_SET_CONTROL
 *
 * This command accepts client-specified control parameters for a set of
 * PERF_CF_PM_SENSORS and applies these new parameters.
 *
 * See @ref LW2080_CTRL_PERF_PERF_CF_PM_SENSORS_CONTROL for documentation on
 * the parameters.
 *
 * Return values are specified per @ref BoardObjGrpSetControl.
 */
#define LW2080_CTRL_CMD_PERF_PERF_CF_PM_SENSORS_SET_CONTROL   (0x208020e3) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | 0xE3" */

/* ---------------------------- PERF_CF Topologies --------------------------- */

/*!
 * Macros for PERF_CF Topology types
 */
#define LW2080_CTRL_PERF_PERF_CF_TOPOLOGY_TYPE_SENSED_BASE    0x00
#define LW2080_CTRL_PERF_PERF_CF_TOPOLOGY_TYPE_MIN_MAX        0x01
#define LW2080_CTRL_PERF_PERF_CF_TOPOLOGY_TYPE_SENSED         0x02

/*!
 * Macros for PERF_CF Topology index
 *
 * @note - There is a type mismatch between some indices which are of type
 * @ref LwBoardObjIdx
 */
#define LW2080_CTRL_PERF_PERF_CF_TOPOLOGY_INDEX_ILWALID       LW2080_CTRL_BOARDOBJ_IDX_ILWALID_8BIT

/*!
 * Macros for PERF_CF Topology units
 */
#define LW2080_CTRL_PERF_PERF_CF_TOPOLOGY_UNIT_PERCENTAGE     0x00
#define LW2080_CTRL_PERF_PERF_CF_TOPOLOGY_UNIT_BYTES_PER_NSEC 0x01
#define LW2080_CTRL_PERF_PERF_CF_TOPOLOGY_UNIT_GHZ            0x02
#define LW2080_CTRL_PERF_PERF_CF_TOPOLOGY_UNIT_NS             0x03

/*!
 * Macros for PERF_CF Topology labels
 */
#define LW2080_CTRL_PERF_PERF_CF_TOPOLOGY_LABEL_GR            0x00
#define LW2080_CTRL_PERF_PERF_CF_TOPOLOGY_LABEL_FB            0x01
#define LW2080_CTRL_PERF_PERF_CF_TOPOLOGY_LABEL_VID           0x02
#define LW2080_CTRL_PERF_PERF_CF_TOPOLOGY_LABEL_PCIE_TX       0x03
#define LW2080_CTRL_PERF_PERF_CF_TOPOLOGY_LABEL_PCIE_RX       0x04
#define LW2080_CTRL_PERF_PERF_CF_TOPOLOGY_LABEL_DEC0          0x05
#define LW2080_CTRL_PERF_PERF_CF_TOPOLOGY_LABEL_ENC0          0x06
#define LW2080_CTRL_PERF_PERF_CF_TOPOLOGY_LABEL_ENC1          0x07
#define LW2080_CTRL_PERF_PERF_CF_TOPOLOGY_LABEL_ENC2          0x08
#define LW2080_CTRL_PERF_PERF_CF_TOPOLOGY_LABEL_DEC1          0x09
#define LW2080_CTRL_PERF_PERF_CF_TOPOLOGY_LABEL_DEC2          0x0A
#define LW2080_CTRL_PERF_PERF_CF_TOPOLOGY_LABEL_DEC3          0x0B
#define LW2080_CTRL_PERF_PERF_CF_TOPOLOGY_LABEL_DEC4          0x0C
#define LW2080_CTRL_PERF_PERF_CF_TOPOLOGY_LABEL_JPG0          0x0D
#define LW2080_CTRL_PERF_PERF_CF_TOPOLOGY_LABEL_OFA           0x0E
#define LW2080_CTRL_PERF_PERF_CF_TOPOLOGY_LABEL_GSP           0x0F
#define LW2080_CTRL_PERF_PERF_CF_TOPOLOGY_LABEL_NONE          0xFF

/*!
 * Macros for PERF_CF Topology GPUMON tags
 */
#define LW2080_CTRL_PERF_PERF_CF_TOPOLOGY_GPUMON_TAG_FB       0x00
#define LW2080_CTRL_PERF_PERF_CF_TOPOLOGY_GPUMON_TAG_GR       0x01
#define LW2080_CTRL_PERF_PERF_CF_TOPOLOGY_GPUMON_TAG_LWENC    0x02
#define LW2080_CTRL_PERF_PERF_CF_TOPOLOGY_GPUMON_TAG_LWDEC    0x03
#define LW2080_CTRL_PERF_PERF_CF_TOPOLOGY_GPUMON_TAG_VID      0x04
#define LW2080_CTRL_PERF_PERF_CF_TOPOLOGY_GPUMON_TAG_NONE     0xFF

/*!
 * Maximum number of PERF_CF_TOPOLOGYs which can be supported in the RM or PMU.
 */
#define LW2080_CTRL_PERF_PERF_CF_TOPOLOGYS_MAX_OBJECTS        LW2080_CTRL_BOARDOBJGRP_E32_MAX_OBJECTS

/*!
 * Structure describing PERF_CF_TOPOLOGY_SENSED_BASE static information/POR.
 */
typedef struct LW2080_CTRL_PERF_PERF_CF_TOPOLOGY_INFO_SENSED_BASE {
    /*!
     * Index into the Performance Sensor Table for sensor.
     */
    LwU8 sensorIdx;
    /*!
     * Index into the Performance Sensor Table for base sensor.
     */
    LwU8 baseSensorIdx;
} LW2080_CTRL_PERF_PERF_CF_TOPOLOGY_INFO_SENSED_BASE;
typedef struct LW2080_CTRL_PERF_PERF_CF_TOPOLOGY_INFO_SENSED_BASE *PLW2080_CTRL_PERF_PERF_CF_TOPOLOGY_INFO_SENSED_BASE;

/*!
 * Structure describing PERF_CF_TOPOLOGY_MIN_MAX static information/POR.
 */
typedef struct LW2080_CTRL_PERF_PERF_CF_TOPOLOGY_INFO_MIN_MAX {
    /*!
     * Index into the Performance Topolgy Table for 1st topology.
     */
    LwU8   topologyIdx1;
    /*!
     * Index into the Performance Topolgy Table for 2nd topology.
     */
    LwU8   topologyIdx2;
    /*!
     * LW_TRUE = max. LW_FALSE = min.
     */
    LwBool bMax;
} LW2080_CTRL_PERF_PERF_CF_TOPOLOGY_INFO_MIN_MAX;
typedef struct LW2080_CTRL_PERF_PERF_CF_TOPOLOGY_INFO_MIN_MAX *PLW2080_CTRL_PERF_PERF_CF_TOPOLOGY_INFO_MIN_MAX;

/*!
 * Structure describing PERF_CF_TOPOLOGY_SENSED static information/POR.
 */
typedef struct LW2080_CTRL_PERF_PERF_CF_TOPOLOGY_INFO_SENSED {
    /*!
     * Index into the Performance Sensor Table for sensor.
     */
    LwBoardObjIdx sensorIdx;
} LW2080_CTRL_PERF_PERF_CF_TOPOLOGY_INFO_SENSED;

/*!
 * PERF_CF_TOPOLOGY type-specific data union. Discriminated by
 * PERF_CF_TOPOLOGY::super.type.
 */


/*!
 * Structure describing PERF_CF_TOPOLOGY static information/POR.
 * Implements the BOARDOBJ model/interface.
 */
typedef struct LW2080_CTRL_PERF_PERF_CF_TOPOLOGY_INFO {
    /*!
     * LW2080_CTRL_BOARDOBJ super class.
     * Must always be first object in structure.
     */
    LW2080_CTRL_BOARDOBJ super;
    /*!
     * XAPI does not support using an element within another structure as a
     * discriminant. Placing this redundant type value here until that design
     * constraint can be fixed.
     */
    LwU8                 type;
    /*!
     * Unit of topology reading. @ref LW2080_CTRL_PERF_PERF_CF_TOPOLOGY_UNIT_<xyz>.
     */
    LwU8                 unit;
    /*!
     * Label. @ref LW2080_CTRL_PERF_PERF_CF_TOPOLOGY_LABEL_<xyz>.
     */
    LwU8                 label;
    /*!
     * Tagged for GPUMON logging.
     * @ref LW2080_CTRL_PERF_PERF_CF_TOPOLOGY_GPUMON_TAG_<xyz>.
     */
    LwU8                 gpumonTag;
    /*!
     * This topology is not actually available (e.g. engine is floorsweept).
     */
    LwBool               bNotAvailable;
     /*!
     * Type-specific data union.
     */
    union {
        LW2080_CTRL_PERF_PERF_CF_TOPOLOGY_INFO_SENSED_BASE sensedBase;
        LW2080_CTRL_PERF_PERF_CF_TOPOLOGY_INFO_MIN_MAX     minMax;
        LW2080_CTRL_PERF_PERF_CF_TOPOLOGY_INFO_SENSED      sensed;
    } data;
} LW2080_CTRL_PERF_PERF_CF_TOPOLOGY_INFO;
typedef struct LW2080_CTRL_PERF_PERF_CF_TOPOLOGY_INFO *PLW2080_CTRL_PERF_PERF_CF_TOPOLOGY_INFO;

/*!
 * Structure describing PERF_CF_TOPOLOGY static information/POR.
 * Implements the BOARDOBJGRP model/interface.
 */
#define LW2080_CTRL_PERF_PERF_CF_TOPOLOGYS_INFO_MESSAGE_ID (0xC4U)

typedef struct LW2080_CTRL_PERF_PERF_CF_TOPOLOGYS_INFO {
    /*!
     * LW2080_CTRL_BOARDOBJGRP_E32 super class.
     * Must always be first object in structure.
     */
    LW2080_CTRL_BOARDOBJGRP_E32            super;
    /*!
     * RM advertized polling period that will be used by the PMU to poll the
     * corresponding performance sensors. In milliseconds.
     */
    LwU16                                  pollingPeriodms;
    /*!
     * HAL index specifies the list of enumerants to use when interpreting the
     * topology entries.
     */
    LwU8                                   halVal;
    /*!
     * Array of PERF_CF_TOPOLOGY structures. Has valid indexes corresponding to
     * the bits set in @ref super.objMask.
     */
    LW2080_CTRL_PERF_PERF_CF_TOPOLOGY_INFO topologys[LW2080_CTRL_PERF_PERF_CF_TOPOLOGYS_MAX_OBJECTS];
} LW2080_CTRL_PERF_PERF_CF_TOPOLOGYS_INFO;
typedef struct LW2080_CTRL_PERF_PERF_CF_TOPOLOGYS_INFO *PLW2080_CTRL_PERF_PERF_CF_TOPOLOGYS_INFO;

/*!
 * LW2080_CTRL_CMD_PERF_PERF_CF_TOPOLOGYS_GET_INFO
 *
 * This command returns PERF_CF_TOPOLOGYS static object information/POR as
 * specified by the VBIOS in PERF_CF Table.
 *
 * See @ref LW2080_CTRL_PERF_PERF_CF_TOPOLOGYS_INFO for documentation on
 * the parameters.
 *
 * Return values are specified per @ref BoardObjGrpGetInfo.
 */
#define LW2080_CTRL_CMD_PERF_PERF_CF_TOPOLOGYS_GET_INFO (0x208020c4) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | LW2080_CTRL_PERF_PERF_CF_TOPOLOGYS_INFO_MESSAGE_ID" */


/*!
 * Structure representing the dynamic state of PERF_CF_TOPOLOGY_SENSED_BASE.
 */
typedef struct LW2080_CTRL_PERF_PERF_CF_TOPOLOGY_STATUS_SENSED_BASE {
    /*!
     * Last sensor reading. Pass in values from previous call.
     */
    LW_DECLARE_ALIGNED(LwU64 lastSensorReading, 8);
    /*!
     * Last base sensor reading. Pass in values from previous call.
     */
    LW_DECLARE_ALIGNED(LwU64 lastBaseSensorReading, 8);
} LW2080_CTRL_PERF_PERF_CF_TOPOLOGY_STATUS_SENSED_BASE;
typedef struct LW2080_CTRL_PERF_PERF_CF_TOPOLOGY_STATUS_SENSED_BASE *PLW2080_CTRL_PERF_PERF_CF_TOPOLOGY_STATUS_SENSED_BASE;

/*!
 * Structure representing the dynamic state of PERF_CF_TOPOLOGY_MIN_MAX.
 */
typedef struct LW2080_CTRL_PERF_PERF_CF_TOPOLOGY_STATUS_MIN_MAX {
    /*!
     * Lwrrently we do NOT have any dynamic state of this class.
     */
    LwU8 rsvd;
} LW2080_CTRL_PERF_PERF_CF_TOPOLOGY_STATUS_MIN_MAX;
typedef struct LW2080_CTRL_PERF_PERF_CF_TOPOLOGY_STATUS_MIN_MAX *PLW2080_CTRL_PERF_PERF_CF_TOPOLOGY_STATUS_MIN_MAX;

/*!
 * Structure representing the dynamic state of PERF_CF_TOPOLOGY_SENSED.
 */
typedef struct LW2080_CTRL_PERF_PERF_CF_TOPOLOGY_STATUS_SENSED {
    /*!
     * Last sensor reading. Pass in values from previous call.
     */
    LW_DECLARE_ALIGNED(LwU64 lastSensorReading, 8);
} LW2080_CTRL_PERF_PERF_CF_TOPOLOGY_STATUS_SENSED;

/*!
 * PERF_CF_TOPOLOGY type-specific data union. Discriminated by
 * PERF_CF_TOPOLOGY::super.type.
 */


/*!
 * Structure representing the dynamic state of each PERF_CF_TOPOLOGY.
 * Implements the BOARDOBJ model/interface.
 */
typedef struct LW2080_CTRL_PERF_PERF_CF_TOPOLOGY_STATUS {
    /*!
     * LW2080_CTRL_BOARDOBJ super class.
     * Must always be first object in structure.
     */
    LW2080_CTRL_BOARDOBJ super;
    /*!
     * XAPI does not support using an element within another structure as a
     * discriminant. Placing this redundant type value here until that design
     * constraint can be fixed.
     */
    LwU8                 type;
    /*!
     * Common reading for all topologies. FXP40.24.
     * Note that XAPI doesn't like LwUFXP40_24.
     */
    LW_DECLARE_ALIGNED(LwU64 reading, 8);
    /*!
     * Last periodically polled reading. FXP40.24.
     * Note that XAPI doesn't like LwUFXP40_24.
     */
    LW_DECLARE_ALIGNED(LwU64 lastPolledReading, 8);
    /*!
     * Type-specific data union.
     */
    union {
        LW_DECLARE_ALIGNED(LW2080_CTRL_PERF_PERF_CF_TOPOLOGY_STATUS_SENSED_BASE sensedBase, 8);
        LW2080_CTRL_PERF_PERF_CF_TOPOLOGY_STATUS_MIN_MAX minMax;
        LW_DECLARE_ALIGNED(LW2080_CTRL_PERF_PERF_CF_TOPOLOGY_STATUS_SENSED sensed, 8);
    } data;
} LW2080_CTRL_PERF_PERF_CF_TOPOLOGY_STATUS;
typedef struct LW2080_CTRL_PERF_PERF_CF_TOPOLOGY_STATUS *PLW2080_CTRL_PERF_PERF_CF_TOPOLOGY_STATUS;

/*!
 * Structure representing the dynamic state of PERF_CF_TOPOLOGYS.
 * Implements the BOARDOBJGRP model/interface.
 */
#define LW2080_CTRL_PERF_PERF_CF_TOPOLOGYS_STATUS_MESSAGE_ID (0xC5U)

typedef struct LW2080_CTRL_PERF_PERF_CF_TOPOLOGYS_STATUS {
    /*!
     * LW2080_CTRL_BOARDOBJGRP_E32 super class.
     * Must always be first object in structure.
     */
    LW2080_CTRL_BOARDOBJGRP_E32 super;
    /*!
     * Array of PERF_CF_TOPOLOGY structures. Has valid indexes corresponding to
     * the bits set in @ref super.objMask.
     */
    LW_DECLARE_ALIGNED(LW2080_CTRL_PERF_PERF_CF_TOPOLOGY_STATUS topologys[LW2080_CTRL_PERF_PERF_CF_TOPOLOGYS_MAX_OBJECTS], 8);
} LW2080_CTRL_PERF_PERF_CF_TOPOLOGYS_STATUS;
typedef struct LW2080_CTRL_PERF_PERF_CF_TOPOLOGYS_STATUS *PLW2080_CTRL_PERF_PERF_CF_TOPOLOGYS_STATUS;

/*!
 * LW2080_CTRL_CMD_PERF_PERF_CF_TOPOLOGYS_GET_STATUS
 *
 * This command returns PERF_CF_TOPOLOGYS dynamic state.
 *
 * See @ref LW2080_CTRL_PERF_PERF_CF_TOPOLOGYS_STATUS for documentation on
 * the parameters.
 *
 * Return values are specified per @ref BoardObjGrpGetInfo.
 */
#define LW2080_CTRL_CMD_PERF_PERF_CF_TOPOLOGYS_GET_STATUS (0x208020c5) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | LW2080_CTRL_PERF_PERF_CF_TOPOLOGYS_STATUS_MESSAGE_ID" */


/*!
 * Structure representing the control parameters of PERF_CF_TOPOLOGY_SENSED_BASE.
 */
typedef struct LW2080_CTRL_PERF_PERF_CF_TOPOLOGY_CONTROL_SENSED_BASE {
    /*!
     * Lwrrently we do NOT have any control parameter in this class.
     */
    LwU8 rsvd;
} LW2080_CTRL_PERF_PERF_CF_TOPOLOGY_CONTROL_SENSED_BASE;
typedef struct LW2080_CTRL_PERF_PERF_CF_TOPOLOGY_CONTROL_SENSED_BASE *PLW2080_CTRL_PERF_PERF_CF_TOPOLOGY_CONTROL_SENSED_BASE;

/*!
 * Structure representing the control parameters of PERF_CF_TOPOLOGY_MIN_MAX.
 */
typedef struct LW2080_CTRL_PERF_PERF_CF_TOPOLOGY_CONTROL_MIN_MAX {
    /*!
     * Lwrrently we do NOT have any control parameter in this class.
     */
    LwU8 rsvd;
} LW2080_CTRL_PERF_PERF_CF_TOPOLOGY_CONTROL_MIN_MAX;
typedef struct LW2080_CTRL_PERF_PERF_CF_TOPOLOGY_CONTROL_MIN_MAX *PLW2080_CTRL_PERF_PERF_CF_TOPOLOGY_CONTROL_MIN_MAX;

/*!
 * Structure representing the control parameters of PERF_CF_TOPOLOGY_SENSED.
 */
typedef struct LW2080_CTRL_PERF_PERF_CF_TOPOLOGY_CONTROL_SENSED {
    /*!
     * Lwrrently we do NOT have any control parameter in this class.
     */
    LwU8 rsvd;
} LW2080_CTRL_PERF_PERF_CF_TOPOLOGY_CONTROL_SENSED;

/*!
 * PERF_CF_TOPOLOGY type-specific data union. Discriminated by
 * PERF_CF_TOPOLOGY::super.type.
 */


/*!
 * Structure representing the control parameters of each PERF_CF_TOPOLOGY.
 * Implements the BOARDOBJ model/interface.
 */
typedef struct LW2080_CTRL_PERF_PERF_CF_TOPOLOGY_CONTROL {
    /*!
     * LW2080_CTRL_BOARDOBJ super class.
     * Must always be first object in structure.
     */
    LW2080_CTRL_BOARDOBJ super;
    /*!
     * XAPI does not support using an element within another structure as a
     * discriminant. Placing this redundant type value here until that design
     * constraint can be fixed.
     */
    LwU8                 type;
    /*!
     * Type-specific data union.
     */
    union {
        LW2080_CTRL_PERF_PERF_CF_TOPOLOGY_CONTROL_SENSED_BASE sensedBase;
        LW2080_CTRL_PERF_PERF_CF_TOPOLOGY_CONTROL_MIN_MAX     minMax;
        LW2080_CTRL_PERF_PERF_CF_TOPOLOGY_CONTROL_SENSED      sensed;
    } data;
} LW2080_CTRL_PERF_PERF_CF_TOPOLOGY_CONTROL;
typedef struct LW2080_CTRL_PERF_PERF_CF_TOPOLOGY_CONTROL *PLW2080_CTRL_PERF_PERF_CF_TOPOLOGY_CONTROL;

/*!
 * Structure representing the control parameters of PERF_CF_TOPOLOGYS.
 * Implements the BOARDOBJ model/interface.
 */
typedef struct LW2080_CTRL_PERF_PERF_CF_TOPOLOGYS_CONTROL {
    /*!
     * LW2080_CTRL_BOARDOBJGRP_E32 super class.
     * Must always be first object in structure.
     */
    LW2080_CTRL_BOARDOBJGRP_E32               super;
    /*!
     * Array of PERF_CF_TOPOLOGY structures. Has valid indexes corresponding to
     * the bits set in @ref super.objMask.
     */
    LW2080_CTRL_PERF_PERF_CF_TOPOLOGY_CONTROL topologys[LW2080_CTRL_PERF_PERF_CF_TOPOLOGYS_MAX_OBJECTS];
} LW2080_CTRL_PERF_PERF_CF_TOPOLOGYS_CONTROL;
typedef struct LW2080_CTRL_PERF_PERF_CF_TOPOLOGYS_CONTROL *PLW2080_CTRL_PERF_PERF_CF_TOPOLOGYS_CONTROL;

/*!
 * LW2080_CTRL_CMD_PERF_PERF_CF_TOPOLOGYS_GET_CONTROL
 *
 * This command returns current PERF_CF_TOPOLOGYS control parameters.
 *
 * See @ref LW2080_CTRL_PERF_PERF_CF_TOPOLOGYS_CONTROL for documentation on
 * the parameters.
 *
 * Return values are specified per @ref BoardObjGrpGetControl.
 */
#define LW2080_CTRL_CMD_PERF_PERF_CF_TOPOLOGYS_GET_CONTROL (0x208020c6) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | 0xC6" */


/*!
 * LW2080_CTRL_CMD_PERF_PERF_CF_TOPOLOGYS_SET_CONTROL
 *
 * This command accepts client-specified control parameters for a set of
 * PERF_CF_TOPOLOGYS and applies these new parameters.
 *
 * See @ref LW2080_CTRL_PERF_PERF_CF_TOPOLOGYS_CONTROL for documentation on
 * the parameters.
 *
 * Return values are specified per @ref BoardObjGrpSetControl.
 */
#define LW2080_CTRL_CMD_PERF_PERF_CF_TOPOLOGYS_SET_CONTROL (0x208020c7) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | 0xC7" */

/* ---------------------------- PERF_CF Pwr Models ----------------------------- */

/*!
 * Macros for PERF_CF Pwr Model index
 */
#define LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_INDEX_ILWALID   LW2080_CTRL_BOARDOBJ_IDX_ILWALID

/*!
 * @brief   Type reserved for @ref LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_SEMANTIC_INDEX
 *          enumerations.
 */
typedef LwU8 LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_SEMANTIC_INDEX;

/*!
 * @defgroup LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_SEMANTIC_INDEX_ENUM
 *
 * Enumeration of different named PWR_MODEL indexes. Of type
 * @ref LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_SEMANTIC_INDEX.
 *
 * @{
 */
#define LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_SEMANTIC_INDEX_TGP  0x00
#define LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_SEMANTIC_INDEX__MAX 0x08
/*!@}*/

/* ---------------------------- PERF_CF Metrics ------------------------------ */
/*!
 * Maximum number of PERF_CF_PWR_MODELs which can be supported in the RM or PMU.
 */
#define LW2080_CTRL_PERF_PERF_CF_PWR_MODELS_MAX_OBJECTS        LW2080_CTRL_BOARDOBJGRP_E32_MAX_OBJECTS

/*!
 * Class providing all rail information that the DLPPM_1X power model uses.
 */
typedef struct LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_DLPPM_1X_RAIL {
    /*!
     * Mask of independent clock domains that reside on this rail.
     */
    LW2080_CTRL_BOARDOBJGRP_MASK_E32                    independentClkDomMask;

    /*!
     * Index to VOLT_RAIL table of the rail that the clock is for.
     */
    LwBoardObjIdx                                       voltRailIdx;

    /*!
     * Index into CLK_DOMAINS of the clock associated with this rail.
     */
    LwBoardObjIdx                                       clkDomIdx;

    /*!
     * Index into PERF_CF_TOPOLOGYS of the topology sensing the clock frequency
     * of @ref clkDomIdx.
     */
    LwBoardObjIdx                                       clkDomTopIdx;

    /*!
     * Index into PERF_CF_TOPOLOGY table of the topology providing the
     * utiliazation percentage of the clkDomIdx CLK_DOMAIN.
     */
    LwBoardObjIdx                                       clkDomUtilTopIdx;

    /*!
     * Index into PWR_CHANNEL table of the power channel measuring the input power
     * of the rail.
     */
    LwBoardObjIdx                                       inPwrChIdx;

    /*!
     * Index into PWR_CHANNEL table of the power channel measuring the output power
     * of the rail.
     */
    LwBoardObjIdx                                       outPwrChIdx;

    /*!
     * Index into PWR_CHRELATIONSHIP table of the VR efficiency equation for this rail
     * from input to output.
     */
    LwBoardObjIdx                                       vrEfficiencyChRelIdxInToOut;

    /*!
     * Index into PWR_CHRELATIONSHIP table of the VR efficiency equation for this rail.
     * from output to input.
     */
    LwBoardObjIdx                                       vrEfficiencyChRelIdxOutToIn;

    /*!
     * Deprecated -- replaced with vrEfficiencyChRelIdxOutToIn and vrEfficiencyChRelIdxInToOut
     * Index into PWR_CHRELATIONSHIP table of the VR efficiency equation for this rail.
     */
    LwBoardObjIdx                                       vrEfficiencyChRelIdx;

    /*!
     * Voltage sampling mode.
     */
    LW2080_CTRL_VOLT_VOLT_RAIL_SENSED_VOLTAGE_MODE_ENUM voltMode;
} LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_DLPPM_1X_RAIL;
typedef struct LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_DLPPM_1X_RAIL *PLW2080_CTRL_PERF_PERF_CF_PWR_MODEL_DLPPM_1X_RAIL;

/*!
 * Maximum number of core rails supported by DLPPM_1X
 */
#define LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_DLPPM_1X_MAX_CORE_RAILS (2)

/*!
 * Structure representing the rail(s) for the GPU core.
 */
typedef struct LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_DLPPM_1X_CORE_RAIL {
    /*!
     * Set of rails belonging to the core.
     */
    LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_DLPPM_1X_RAIL rails[LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_DLPPM_1X_MAX_CORE_RAILS];

    /*!
     * Number of rails in the core. The first @ref numRails are valid in
     * @ref coreRail.
     */
    LwU8                                             numRails;
} LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_DLPPM_1X_CORE_RAIL;
typedef struct LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_DLPPM_1X_CORE_RAIL *PLW2080_CTRL_PERF_PERF_CF_PWR_MODEL_DLPPM_1X_CORE_RAIL;

/*!
 * Structure representing the FBVDD rail.
 */
typedef struct LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_DLPPM_1X_FBVDD_RAIL {
    /*!
     * Super-class.
     */
    LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_DLPPM_1X_RAIL super;

    /*!
     * Name of the rail.
     */
    LwU8                                             voltRailName;
} LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_DLPPM_1X_FBVDD_RAIL;
typedef struct LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_DLPPM_1X_FBVDD_RAIL *PLW2080_CTRL_PERF_PERF_CF_PWR_MODEL_DLPPM_1X_FBVDD_RAIL;

/*!
 * Bounds on the range of values that can be used to correct DLPPM_1X metrics.
 */
typedef struct LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_DLPPM_1X_CORRECTION_BOUND {
    /*!
     * Minimum percentage that can be used for correction.
     */
    LwUFXP20_12 minPct;

    /*!
     * Maximum percentage that can be used for correction.
     */
    LwUFXP20_12 maxPct;
} LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_DLPPM_1X_CORRECTION_BOUND;

/*!
 * Bounds on the corrections for each DLPPM_1X metric.
 */
typedef struct LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_DLPPM_1X_CORRECTION_BOUNDS {
    /*!
     * Bounds on the base estimated performance metric.
     */
    LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_DLPPM_1X_CORRECTION_BOUND perf;

    /*!
     * Bounds on the base estimated core rail power metrics.
     */
    LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_DLPPM_1X_CORRECTION_BOUND coreRailPwr[LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_DLPPM_1X_MAX_CORE_RAILS];

    /*!
     * Bounds on the base estimated FB power metric.
     */
    LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_DLPPM_1X_CORRECTION_BOUND fbPwr;
} LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_DLPPM_1X_CORRECTION_BOUNDS;

/*!
 * Structure describing PERF_CF_PWR_MODEL_DLPPM_1X static information/POR.
 */
typedef struct LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_INFO_DLPPM_1X {
    /*!
     * Core rail state.
     */
    LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_DLPPM_1X_CORE_RAIL         coreRail;

    /*!
     * FBVDD rail state.
     */
    LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_DLPPM_1X_FBVDD_RAIL        fbRail;

    /*!
     * When doing guard railing on the core rail power versus performance, this
     * is the percent error allowed in the estimated power against the fit's
     * expected power. If the estimated power is outside of this percent
     * threshold, it is set to the threshold.
     */
    LwUFXP20_12                                                   coreRailPwrFitTolerance;

    /*!
     * Index into NNE_DESCS of the DL perf and power estimator.
     */
    LwBoardObjIdx                                                 nneDescIdx;

    /*!
     * Index into PERF_CF sensor table for PERF_CF_SENSOR representing
     * all PM's.
     */
    LwBoardObjIdx                                                 pmSensorIdx;

    /*!
     * Specifies which specific PM sensors the PERF_CF_PM_SENSOR should sample
     * for this PWR_MODEL.
     */
    LW2080_CTRL_PERF_PERF_CF_PM_SENSOR_SIGNAL_MASK                pmSensorSignalMask;

    /*!
     * Percent tolerance used to determine if two DRAMCLK values are equal
     */
    LwUFXP20_12                                                   dramclkMatchTolerance;

    /*!
     * Threshold value in between (0,1] that determines how smalll the callwlated
     * perfMs value from gpcclk count can be before the observed data is marked
     * as invalid
     */
    LwUFXP20_12                                                   observedPerfMsMinPct;

    /*!
     * Bounds on the correction of estimated DLPPM_1X metrics
     */
    LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_DLPPM_1X_CORRECTION_BOUNDS correctionBounds;

    /*!
     * Index into the PWR_CHANNEL table for the small rails.
     *
     * NOTE: If this is set to invalid, DLPPM_1X shall assume that small rails
     *       should be omitted from consideration.
     */
    LwBoardObjIdx                                                 smallRailPwrChIdx;

    /*!
     * Index into the PWR_CHANNEL table for the TGP.
     */
    LwBoardObjIdx                                                 tgpPwrChIdx;

    /*!
     * Index into the PERF_CF_TOPOLOGYS table for the topology for PTIMER
     */
    LwBoardObjIdx                                                 ptimerTopologyIdx;

    /*!
     * Number of VF points to infer in a batch, during efficiency lwrve search.
     * May be at most @ref LW2080_CTRL_NNE_NNE_DESC_INFERENCE_LOOPS_MAX.
     */
    LwU8                                                          vfInferenceBatchSize;

    /*
    * Boolean on whether to scale GPCCLK counts by the utilPct
    */
    LwBool                                                        bFakeGpcClkCounts;

    /*
     * Index into DLPPM_1X_OBSERVED::pmSensorDiff where the gpcclk counts
     * are located
     */
    LwBoardObjIdx                                                 gpcclkPmIndex;


    /*
     * Index into DLPPM_1X_OBSERVED::pmSensorDiff where the ltcclk counts
     * are located
     */
    LwBoardObjIdx                                                 xbarclkPmIndex;
} LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_INFO_DLPPM_1X;
typedef struct LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_INFO_DLPPM_1X *PLW2080_CTRL_PERF_PERF_CF_PWR_MODEL_INFO_DLPPM_1X;

/*!
 * Structure holding all device-info related data for DLPPM_1X
 */
typedef struct LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_DLPPM_1X_CHIP_CONFIG {
    /*!
     * Number of enabled TPCs on the chip.
     */
    LwU16 numTpc;

    /*!
     * Number of enabled LTC slices on the chip.
     */
    LwU16 numLtc;

    /*!
     * Number of enabked GPCs on the chip.
    */
    LwU16 numGpc;

    /*!
     * Number of enabled FBPs on the chip.
     */
    LwU8  numFbp;
} LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_DLPPM_1X_CHIP_CONFIG;
typedef struct LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_DLPPM_1X_CHIP_CONFIG *PLW2080_CTRL_PERF_PERF_CF_PWR_MODEL_DLPPM_1X_CHIP_CONFIG;

/*!
 * Class providing all rail metrics that DLPPM_1X uses.
 */
typedef struct LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X_RAIL {
    /*!
     * Estimated Input PWR_TUPLEwer of the current rail for a
     * given VF point. Found via callwlating a relationship from
     * pwrOutTuple --> pwrInTuple.
     * NOTE: power here is post perf correction for GPU idle cases
     */
    LW2080_CTRL_PMGR_PWR_TUPLE pwrInTuple;

    /*!
     * Estimated Output PWR_TUPLE of the current rail for a
     * given VF point. Found via scaling pwrOutDynamicNormalizedmWPostPerfCorrection
     * by a factor that is based on rail voltage.
     * NOTE: power here is post perf correction for GPU idle cases
     */
    LW2080_CTRL_PMGR_PWR_TUPLE pwrOutTuple;

    /*!
     * Frequency of the clock, in kHz.
     */
    LwU32                      freqkHz;

    /*!
     * Utilization percentage of the clock domain for this rail.
     *
     * @note    Actually an @ref LwUFXP40_24
     */
    LwU64_ALIGN32              utilPct;

    /*!
     * Output dynamic power of the rail, normalized to 1 Volt.
     * Direct output from NNE, primarily used for guard-railing.
     */
    LwU32                      pwrOutDynamicNormalizedmW;

    /*!
     * pwrOutDynamicNormalizedmW that has been scaled by a value
     * of (0,1] in order to correct power in the case where the GPU
     * is idle.
     * Primarily used as the estimated dynamic power at a VF point
     */
    LwU32                      pwrOutDynamicNormalizedmWPostPerfCorrection;

    /*!
     * Rail voltage in uV.
     */
    LwU32                      voltuV;

    /*!
     * Leakage power of the rail, in mW.
     */
    LwU32                      leakagePwrmW;

    /*!
     * Minimum voltage in [uV] needed support all clock domains on this rail,
     * incorporating the minimum allowable voltage for the rail.
     */
    LwU32                      voltMinuV;
} LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X_RAIL;
typedef struct LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X_RAIL *PLW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X_RAIL;

/*!
 * Collection of performance metrics used by DLPPM_1X
 */
typedef struct LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X_PERF {
    /*!
     * Performance, measured as the time required to produce a set of
     * metrics, in milliseconds.
     */
    LwUFXP20_12 perfms;

    /*!
     * Unitless metric for performance. Represents the relative increase in
     * performance for these metrics over the baseline.
     */
    LwUFXP20_12 perfRatio;
} LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X_PERF;

/*!
 * @brief   An invalid value of
 *          @ref LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X_PERF::perfms
 */
#define LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X_PERF_PERFMS_ILWALID     (0U)

/*!
 * @brief   An invalid value of
 *          @ref LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X_PERF::perfRatio
 */
#define LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X_PERF_PERF_RATIO_ILWALID (LW_U32_MAX)

/*!
 * Class providing all observed rail metrics that DLPPM_1X uses.
 */
typedef struct LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_OBSERVED_METRICS_DLPPM_1X_RAIL {
    /*!
     * super - must be first member of the structure.
     */
    LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X_RAIL super;

    /*!
     * @copydoc LW2080_CTLR_VOLT_RAIL_SENSED_VOLTAGE_DATA.
     */
    LW2080_CTRL_VOLT_RAIL_SENSED_VOLTAGE_DATA                voltData;

    /*!
     * Maximum Vmin observed for all clock domains on this rail.
     */
    LwU32                                                    maxIndependentClkDomVoltMinuV;

    /*!
     * Minimum voltage, in uV, that the rail can be set to.
     */
    LwU32                                                    vminLimituV;
} LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_OBSERVED_METRICS_DLPPM_1X_RAIL;
typedef struct LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_OBSERVED_METRICS_DLPPM_1X_RAIL *PLW2080_CTRL_PERF_PERF_CF_PWR_MODEL_OBSERVED_METRICS_DLPPM_1X_RAIL;

/*!
 * Class providing all rail metrics that DLPPM_1X uses.
 */
typedef struct LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X_CORE_RAIL {
    /*!
     * Set of rails belonging to the core.
     *
     * Number of valid rails intentionally omitted from this structure
     * to avoid redundancy. The number of valid entries in this array is available
     * everywhere (i.e. LWAPI, RM, PMU) through:
     * LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_DLPPM_1X_CORE_RAIL::numRails
     */
    LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X_RAIL rails[LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_DLPPM_1X_MAX_CORE_RAILS];

    /*!
     * Combined core-rails VR input PWR_TUPLE.
     */
    LW2080_CTRL_PMGR_PWR_TUPLE                               pwrInTuple;
} LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X_CORE_RAIL;
typedef struct LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X_CORE_RAIL *PLW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X_CORE_RAIL;

/*!
 * Class providing all observed rail metrics that DLPPM_1X uses.
 */
typedef struct LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_OBSERVED_METRICS_DLPPM_1X_CORE_RAIL {
    /*!
     * Set of rails belonging to the core.
     *
     * Number of valid rails intentionally omitted from this structure
     * to avoid redundancy. The number of valid entries in this array is available
     * everywhere (i.e. LWAPI, RM, PMU) through:
     * LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_DLPPM_1X_CORE_RAIL::numRails
     */
    LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_OBSERVED_METRICS_DLPPM_1X_RAIL rails[LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_DLPPM_1X_MAX_CORE_RAILS];

    /*!
     * Combined core-rails VR input PWR_TUPLE.
     */
    LW2080_CTRL_PMGR_PWR_TUPLE                                        pwrInTuple;
} LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_OBSERVED_METRICS_DLPPM_1X_CORE_RAIL;
typedef struct LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_OBSERVED_METRICS_DLPPM_1X_CORE_RAIL *PLW2080_CTRL_PERF_PERF_CF_PWR_MODEL_OBSERVED_METRICS_DLPPM_1X_CORE_RAIL;

/*!
 * @defgroup    LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X_METRIC
 *
 * @brief   Enumeration of metrics that can be guard-railed in DLPPM_1X.
 *
 * @details
 *
 *      LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X_METRIC_PERF
 *          Corresponds to the estimated performance ratio metric.
 *
 *      LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X_METRIC_CORE_RAIL_PWR
 *          Corresponds to the estimated dynamic normalized power on the primary
 *          core rail.
 *
 *      LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X_METRIC_FB_PWR
 *          Corresponds to the estimated output power on the FB rail.
 * @{
 */
typedef LwU32 LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X_METRIC;
#define LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X_METRIC_PERF          0U
#define LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X_METRIC_CORE_RAIL_PWR 1U
#define LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X_METRIC_FB_PWR        2U
#define LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X_METRIC_NUM           3U
/*!@}*/

/*!
 * @defgroup    LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X_GUARD_RAIL
 *
 * @brief   Enumeration of guard rails that can be applied to DLPPM_1X metrics
 *
 * @details
 *      LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X_GUARD_RAIL_ENDPOINT_PRIMARY_CLK
 *          Metric was non-monotonic with respect to primary clock against a
 *          fixed endpoint and so was set to the endpoint's value.
 *
 *      LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X_GUARD_RAIL_ADJACENT_PRIMARY_CLK
 *          Metric was non-monotonic with respect to primary clock against an
 *          adjacent metric and so was set to the adjacent value.
 *
 *      LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X_GUARD_RAIL_LINEAR_MINIMUM_PRIMARY_CLK
 *          Metric displayed sub-linear behavior and so was set to the minimum
 *          linear value.
 *
 *      LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X_GUARD_RAIL_LINEAR_MAXIMUM_PRIMARY_CLK
 *          Metric displayed super-linear behavior and so was set to the maximum
 *          linear value.
 *
 *      LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X_GUARD_RAIL_LINEAR_FIT
 *          Metric was outside the tolerated bounds of a generated linear fit
 *          and so was set to the tolerated value.
 *
 *
 *      LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X_GUARD_RAIL_ADJACENT_DRAMCLK
 *          Metric was non-monotonic with respect to DRAMCLK against an
 *          adjacent metric and so was set to the adjacent value.
 *
 *      LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X_GUARD_RAIL_LINEAR_DRAMCLK
 *          Metric displayed sub/super-linear behavior and so was set to the
 *          minimum/maximum linear value.
 * @{
 */
typedef LwU32 LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X_GUARD_RAIL;
#define LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X_GUARD_RAIL_ENDPOINT_PRIMARY_CLK       0U
#define LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X_GUARD_RAIL_ADJACENT_PRIMARY_CLK       1U
#define LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X_GUARD_RAIL_LINEAR_MINIMUM_PRIMARY_CLK 2U
#define LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X_GUARD_RAIL_LINEAR_MAXIMUM_PRIMARY_CLK 3U
#define LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X_GUARD_RAIL_LINEAR_FIT                 4U
#define LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X_GUARD_RAIL_ADJACENT_DRAMCLK           5U
#define LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X_GUARD_RAIL_LINEAR_DRAMCLK             6U
#define LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X_GUARD_RAIL_NUM                        7U
/*!@}*/

/*!
 * @brief   Bitvector of guard rails that were applied to an
 *          @ref LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X estimation.
 */
typedef LW2080_CTRL_BOARDOBJGRP_MASK_E32 LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X_GUARD_RAIL_FLAGS;

/*!
 * @brief   Total number of guard rails possible
 */
#define LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X_GUARD_RAIL_FLAGS_NUM (0x15U) /* finn: Evaluated from "(LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X_METRIC_NUM * LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X_GUARD_RAIL_NUM)" */

/*!
 * @brief   Initializes a
 *          @ref LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X_GUARD_RAIL_FLAGS
 *          structure.
 *
 * @param[out]      pFlags  @ref LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X_GUARD_RAIL_FLAGS
 *                          to initialize
 */
#define LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X_GUARD_RAIL_FLAGS_INIT(pFlags) \
    LW2080_CTRL_BOARDOBJGRP_MASK_E32_INIT(&(pFlags)->super)

/*!
 * @brief   Retrieves the index to be used in
 *          @ref LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X_GUARD_RAIL_FLAGS
 *          for a particular metric and guard rail on that metric.
 *
 * @param[in]   metric      @ref LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X_METRIC
 *                          for which to get index
 * @param[in]   guardRail   @ref LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X_GUARD_RAIL
 *                          for which to get index
 *
 * @return  Index in
 *          @ref LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X_GUARD_RAIL_FLAGS
 *          to use for metric and guardRail parameters.
 */
#define LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X_GUARD_RAIL_FLAGS_GUARD_RAIL_INDEX(metric, guardRail) \
    ((metric) * LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X_GUARD_RAIL_NUM + \
        (guardRail))

/*!
 * @brief   Gets a guard rail flag for a metric in an
 *          @ref LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X_GUARD_RAIL_FLAGS
 *
 * @param[out]  pFlags      @ref LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X_GUARD_RAIL_FLAGS
 *                          from which to get flag
 * @param[in]   metric      Metric for which to get guard rail flag
 * @param[in]   guardRail   Guard rail to get for metric
 */
#define LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X_GUARD_RAIL_FLAGS_GET(pFlags, metric, guardRail) \
    LW2080_CTRL_BOARDOBJGRP_MASK_BIT_GET( \
        &(pFlags)->super, \
        LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X_GUARD_RAIL_FLAGS_GUARD_RAIL_INDEX( \
            (metric), \
            (guardRail)))

/*!
 * @brief   Sets a guard rail flag for a metric in an
 *          @ref LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X_GUARD_RAIL_FLAGS
 *
 * @param[out]  pFlags      @ref LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X_GUARD_RAIL_FLAGS
 *                          in which to set flag
 * @param[in]   metric      Metric for which to set guard rail flag
 * @param[in]   guardRail   Guard rail to set for metric
 */
#define LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X_GUARD_RAIL_FLAGS_SET(pFlags, metric, guardRail) \
    LW2080_CTRL_BOARDOBJGRP_MASK_BIT_SET( \
        &(pFlags)->super, \
        LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X_GUARD_RAIL_FLAGS_GUARD_RAIL_INDEX( \
            (metric), \
            (guardRail)))

/*!
 * Structure describing the primary metrics for DLPPM_1X.
 */
typedef struct LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X {
    /*!
     * super - must be first member of the structure.
     */
    LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS                           super;

    /*!
     * Core rail metrics.
     */
    LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X_CORE_RAIL        coreRail;

    /*!
     * FB rail metrics.
     */
    LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X_RAIL             fbRail;

    /*!
     * Sensed PWR_TUPLE of the "small" rails.
     */
    LW2080_CTRL_PMGR_PWR_TUPLE                                           smallRailPwrTuple;

    /*!
     * Sensed PWR_TUPLE for Total GPU Power.
     */
    LW2080_CTRL_PMGR_PWR_TUPLE                                           tgpPwrTuple;

    /*!
     * Performance metrics.
     */
    LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X_PERF             perfMetrics;

    /*!
     * Guard rails applied to this set of metrics.
     */
    LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X_GUARD_RAIL_FLAGS guardRailFlags;
} LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X;
typedef struct LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X *PLW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X;

/*!
 * Structure representing the parameters for a linear fit
 */
typedef struct LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X_CORE_RAIL_POWER_FIT {
    /*!
     * Slope of the linear fit.
     */
    LwUFXP20_12 slopemWPerRatio;

    /*!
     * Intercept of the linear fit.
     */
    LwU32       interceptmW;

    /*!
     * Indicates whether this fit is valid or not.
     */
    LwBool      bValid;
} LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X_CORE_RAIL_POWER_FIT;

/*!
 * Structure to be used for estimates at a given DRAMCLK across the VF lwrve.
 */
typedef struct LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X_DRAMCLK_ESTIMATES {
    /*!
     * Set of points that give coarse grain image of the metrics across the
     * VF lwrve for this given DRAMCLK.
     */
    LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X                     estimatedMetrics[LW2080_CTRL_NNE_NNE_DESC_INFERENCE_LOOPS_MAX];

    /*!
     * Linear fit for core rail power against the render time.
     */
    LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X_CORE_RAIL_POWER_FIT coreRailPwrFit;

    /*!
     * Number of valid entries in
     * @ref LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X_DRAMCLK::estimatedMetrics
     */
    LwU8                                                                    numEstimatedMetrics;
} LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X_DRAMCLK_ESTIMATES;

typedef LwU8 LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X_CORRECTION_OUT_OF_BOUNDS_MASK;
#define LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X_CORRECTION_OUT_OF_BOUNDS_PERF      0x01
#define LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X_CORRECTION_OUT_OF_BOUNDS_CORE_RAIL 0x02
#define LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X_CORRECTION_OUT_OF_BOUNDS_FB_RAIL   0x03

#define LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X_CORRECTION_OUT_OF_BOUNDS_SET(_value,_bit) \
    do {                                                                                              \
        _value |= LWBIT(_bit);                                                                       \
    } while(LW_FALSE)
/*!
 * Corrections to be applied across DLPPM_1X estimations.
 */
typedef struct LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X_CORRECTION {
    /*!
     * Correction percentage to be multiplied against base estimated performance
     * metric.
     */
    LwUFXP20_12                                                                       perfPct;

    /*!
     * Correction percentage to be multiplied against base estimated core rail
     * power metrics.
     */
    LwUFXP20_12                                                                       coreRailPwrPct[LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_DLPPM_1X_MAX_CORE_RAILS];

    /*!
     * Correction percentage to be multiplied against base estimated FB power
     * metric.
     */
    LwUFXP20_12                                                                       fbPwrPct;

    /*!
     * Mask of corrections that went outside their bounds
     */
    LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X_CORRECTION_OUT_OF_BOUNDS_MASK outOfBoundsMask;
} LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X_CORRECTION;

/*!
 * The maximum of DRAMCLK initial estimates that can be observed.
 */
#define LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X_OBSERVED_INITIAL_DRAMCLK_ESTIMATES_MAX              3U

/*!
 * Ascending numerical list of reasons for ilwalidating the current set of observed metrics
 * _NONE: No reason for ilwalidating the metrics
 * _LOW_MEMCLK: Memclk frequency is lower then specified frequency in VPState DLPPM_1X index
 * _LOW_GPCCLK: Gpcclk frequency is lower then specified frequency in VPState DLPPM_1X index
 * _LOW_OBS_PERFMS: BAPM counting the "active" gpcclk counts reports low clock counts (i.e. under utilized)
 * _POISON_NORMALIZATION: Underlying model finds a problem with inputs during normlization
 * _OUT_OF_BOUNDS_CORRECTION: Underlying model gives bad estimations for the observed VF point
 * _BAD_INITIAL_ESTIMATES: Error where we do not find the correct number of initial estimates
 * _BAD_FREQUENCY_FOR_CORRECTION: The observed frequencies were not suitable for callwlating correction factors
 */
#define LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X_ILWALID_METRICS_REASON_NONE                         0x00
#define LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X_ILWALID_METRICS_REASON_LOW_MEMCLK                   0x01
#define LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X_ILWALID_METRICS_REASON_LOW_GPCCLK                   0x02
#define LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X_ILWALID_METRICS_REASON_LOW_OBS_PERFMS               0x03
#define LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X_ILWALID_METRICS_REASON_POISON_NORMALIZATION         0x04
#define LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X_ILWALID_METRICS_REASON_OUT_OF_BOUNDS_CORRECTION     0x05
#define LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X_ILWALID_METRICS_REASON_BAD_INITIAL_ESTIMATES        0x06
#define LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X_ILWALID_METRICS_REASON_BAD_FREQUENCY_FOR_CORRECTION 0x07
#define LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X_ILWALID_METRICS_REASON_FLCN_ERROR                   0xFF

#define LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X_ILWALID_METRICS_SET_REASON(_value, _reason)    \
    do {                                                                                                   \
        _value = LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X_ILWALID_METRICS_REASON##_reason;     \
    } while(LW_FALSE)

/*!
 * @defgroup LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_DLPPM_1X_SCALE_PROFILING_REGION_ENUM
 *
 * Enumeration of PERF_CF_PWR_MODEL_DLPPM_1X profiling regions for observation.
 * @{
 */
typedef LwU8 LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_DLPPM_1X_SCALE_PROFILING_REGION_ENUM;
#define LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_DLPPM_1X_SCALE_PROFILING_REGION_TOTAL             0x00
#define LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_DLPPM_1X_SCALE_PROFILING_REGION_METRICS_POPULATE  0x01
#define LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_DLPPM_1X_SCALE_PROFILING_REGION_NNE_INPUTS_SET    0x02
#define LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_DLPPM_1X_SCALE_PROFILING_REGION_NNE_EVALUATE      0x03
#define LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_DLPPM_1X_SCALE_PROFILING_REGION_NNE_OUTPUTS_PARSE 0x04
#define LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_DLPPM_1X_SCALE_PROFILING_REGION_GUARD_RAIL        0x05
#define LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_DLPPM_1X_SCALE_PROFILING_REGION__COUNT            0x06
/*!@}*/

typedef struct LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X_SCALE_PROFILING {
    /*!
     * Array of profiled "regions" in the code.
     */
    LwU64_ALIGN32                                profiledTimesns[LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_DLPPM_1X_SCALE_PROFILING_REGION__COUNT];

    /*!
     * The NNE profiling data for the NNE_DESC::inference call backing the scale
     * call.
     */
    LW2080_CTRL_NNE_NNE_DESC_INFERENCE_PROFILING nneDescInferenceProfiling;
} LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X_SCALE_PROFILING;

/*!
 * @defgroup LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X_OBSERVED_PROFILING_REGION_ENUM
 *
 * Enumeration of PERF_CF_PWR_MODEL_DLPPM_1X profiling regions for observation.
 * @{
 */
typedef LwU8 LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X_OBSERVED_PROFILING_REGION_ENUM;
#define LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X_OBSERVED_PROFILING_REGION_TOTAL                                                                        0x00U
#define LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X_OBSERVED_PROFILING_REGION_SAMPLE_DATA                                                                  0x01U
#define LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X_OBSERVED_PROFILING_REGION_VPSTATE_VALIDITY                                                             0x02U
#define LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X_OBSERVED_PROFILING_REGION_OBSERVE_PERF                                                                 0x03U
#define LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X_OBSERVED_PROFILING_REGION_NORMALIZE                                                                    0x04U
#define LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X_OBSERVED_PROFILING_REGION_OBSERVE_CORRECTION                                                           0x05U
#define LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X_OBSERVED_PROFILING_REGION_OBSERVE_INITIAL_DRAMCLK_ESTIMATES                                            0x06U
#define LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X_OBSERVED_PROFILING_REGION_OBSERVE_INITIAL_DRAMCLK_ESTIMATES_SCALE                                      0x07U
#define LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X_OBSERVED_PROFILING_REGION_OBSERVE_INITIAL_DRAMCLK_ESTIMATES_SCALE_ITERATION__START                     0x08U
#define LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X_OBSERVED_PROFILING_REGION_OBSERVE_INITIAL_DRAMCLK_ESTIMATES_SCALE_ITERATION_0                          0x08U
#define LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X_OBSERVED_PROFILING_REGION_OBSERVE_INITIAL_DRAMCLK_ESTIMATES_SCALE_ITERATION_1                          0x09U
#define LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X_OBSERVED_PROFILING_REGION_OBSERVE_INITIAL_DRAMCLK_ESTIMATES_SCALE_ITERATION_2                          0x0AU
#define LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X_OBSERVED_PROFILING_REGION_OBSERVE_INITIAL_DRAMCLK_ESTIMATES_SCALE_ITERATION__COUNT                     (0x3U) /* finn: Evaluated from "(LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X_OBSERVED_PROFILING_REGION_OBSERVE_INITIAL_DRAMCLK_ESTIMATES_SCALE_ITERATION_2 - LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X_OBSERVED_PROFILING_REGION_OBSERVE_INITIAL_DRAMCLK_ESTIMATES_SCALE_ITERATION__START + 1)" */
#define LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X_OBSERVED_PROFILING_REGION_OBSERVE_INITIAL_DRAMCLK_ESTIMATES_CROSS_DRAMCLK_GUARD_RAIL                   0x0BU
#define LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X_OBSERVED_PROFILING_REGION_OBSERVE_INITIAL_DRAMCLK_ESTIMATES_CALLWLATE_FIT_ITERATION__START             0x0LW
#define LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X_OBSERVED_PROFILING_REGION_OBSERVE_INITIAL_DRAMCLK_ESTIMATES_CALLWLATE_FIT_ITERATION_0                  0x0LW
#define LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X_OBSERVED_PROFILING_REGION_OBSERVE_INITIAL_DRAMCLK_ESTIMATES_CALLWLATE_FIT_ITERATION_1                  0x0DU
#define LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X_OBSERVED_PROFILING_REGION_OBSERVE_INITIAL_DRAMCLK_ESTIMATES_CALLWLATE_FIT_ITERATION_2                  0x0EU
#define LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X_OBSERVED_PROFILING_REGION_OBSERVE_INITIAL_DRAMCLK_ESTIMATES_CALLWLATE_FIT_ITERATION__COUNT             (0x3U) /* finn: Evaluated from "(LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X_OBSERVED_PROFILING_REGION_OBSERVE_INITIAL_DRAMCLK_ESTIMATES_CALLWLATE_FIT_ITERATION_2 - LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X_OBSERVED_PROFILING_REGION_OBSERVE_INITIAL_DRAMCLK_ESTIMATES_CALLWLATE_FIT_ITERATION__START + 1)" */
#define LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X_OBSERVED_PROFILING_REGION_OBSERVE_INITIAL_DRAMCLK_ESTIMATES_APPLY_FIT_AND_REPROPAGATE_ITERATION__START 0x0FU
#define LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X_OBSERVED_PROFILING_REGION_OBSERVE_INITIAL_DRAMCLK_ESTIMATES_APPLY_FIT_AND_REPROPAGATE_ITERATION_0      0x0FU
#define LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X_OBSERVED_PROFILING_REGION_OBSERVE_INITIAL_DRAMCLK_ESTIMATES_APPLY_FIT_AND_REPROPAGATE_ITERATION_1      0x10U
#define LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X_OBSERVED_PROFILING_REGION_OBSERVE_INITIAL_DRAMCLK_ESTIMATES_APPLY_FIT_AND_REPROPAGATE_ITERATION_2      0x11U
#define LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X_OBSERVED_PROFILING_REGION_OBSERVE_INITIAL_DRAMCLK_ESTIMATES_APPLY_FIT_AND_REPROPAGATE_ITERATION__COUNT (0x3U) /* finn: Evaluated from "(LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X_OBSERVED_PROFILING_REGION_OBSERVE_INITIAL_DRAMCLK_ESTIMATES_APPLY_FIT_AND_REPROPAGATE_ITERATION_2 - LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X_OBSERVED_PROFILING_REGION_OBSERVE_INITIAL_DRAMCLK_ESTIMATES_APPLY_FIT_AND_REPROPAGATE_ITERATION__START + 1)" */
#define LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X_OBSERVED_PROFILING_REGION__COUNT                                                                       0x12U
/*!@}*/

/*!
 * @defgroup LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X_OBSERVED_PROFILING_INFERENCE_ENUM
 *
 * Enumeration of PERF_CF_PWR_MODEL::scale calls done by
 * PERF_CF_PWR_MODEL_DLPPM_1X::observeMetrics, for identification for profiling.
 * @{
 */
typedef LwU8 LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X_OBSERVED_PROFILING_SCALE_ENUM;
#define LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X_OBSERVED_PROFILING_SCALE_CORRECTION               0U
#define LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X_OBSERVED_PROFILING_SCALE_INITIAL_ESTIMATES__START 1U
#define LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X_OBSERVED_PROFILING_SCALE_INITIAL_ESTIMATES_0      1U
#define LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X_OBSERVED_PROFILING_SCALE_INITIAL_ESTIMATES_1      2U
#define LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X_OBSERVED_PROFILING_SCALE_INITIAL_ESTIMATES_2      3U
#define LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X_OBSERVED_PROFILING_SCALE__COUNT                   4U
/*!@}*/

/*!
 * Structure containing profiling data for
 * PERF_CF_PWR_MODEL_DLPPM_1X::observeMetrics
 */
typedef struct LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X_OBSERVED_PROFILING {
    /*!
     * Array of profiled "regions" in the code.
     */
    LwU64_ALIGN32                                                       profiledTimesns[LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X_OBSERVED_PROFILING_REGION__COUNT];

    /*!
     * Array of profile structures for scale calls exelwted during
     * observeMetrics
     */
    LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X_SCALE_PROFILING scaleProfiling[LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X_OBSERVED_PROFILING_SCALE__COUNT];
} LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X_OBSERVED_PROFILING;

/*!
 * Structure describing all observed metrics for DLPPM_1X
 */
typedef struct LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X_OBSERVED {
    /*!
     * super - must be first member of the structure.
     */
    LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS                             super;

    /*!
     * Observed metrics for core rails.
     */
    LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_OBSERVED_METRICS_DLPPM_1X_CORE_RAIL coreRail;

    /*!
     * Observed metrics for FB rail.
     */
    LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_OBSERVED_METRICS_DLPPM_1X_RAIL      fbRail;

    /*!
     * Sensed PWR_TUPLE of the "small" rails.
     */
    LW2080_CTRL_PMGR_PWR_TUPLE                                             smallRailPwrTuple;

    /*!
     * Sensed PWR_TUPLE for Total GPU Power.
     */
    LW2080_CTRL_PMGR_PWR_TUPLE                                             tgpPwrTuple;

    /*!
     * Chip information used by DLPPM_1X.
     */
    LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_DLPPM_1X_CHIP_CONFIG                chipConfig;

    /*!
     * BA PM diff values used by DLPPM_1X.
     *
     * Number of valid sensor signals intentionally omitted from this structure
     * to avoid redundancy. The number of valid entries in this array is available
     * everywhere (i.e. LWAPI, RM, PMU) through:
     * LW2080_CTRL_PER_PER_CF_PM_SENSOR_STATUS::signalsMask
     */
    LwU64_ALIGN32                                                          pmSensorDiff[LW2080_CTRL_PERF_PERF_CF_PM_SENSOR_MAX_SIGNALS];

    /*!
     * Observed performance metrics.
     */
    LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X_PERF               perfMetrics;

    /*!
     * Ratio used as divisor to normalize the inputs to the neural network.
     *
     * @note    Actually an @ref LwF32.
     */
    LwU32                                                                  normRatio;

    /*!
     * Stores tracking data regarding the caching of the metrics for NNE
     */
    LW2080_CTRL_NNE_NNE_DESC_INFERENCE_STATIC_VAR_CACHE                    staticVarCache;

    /*!
     * An initial set of coarse-grained metrics covering the entire master clock
     * VF lwrve for each DRAMCLK to be considered.
     */
    LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X_DRAMCLK_ESTIMATES  initialDramclkEstimates[LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X_OBSERVED_INITIAL_DRAMCLK_ESTIMATES_MAX];

    /*!
     * Number of valid entries in
     * @ref LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X_OBSERVED::initialDramclkEstimates
     */
    LwU8                                                                   numInitialDramclkEstimates;

    /*!
     * Index for the current DRAMCLK inside of the
     * @ref LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X_OBSERVED::initialDramclkEstimates
     * array.
     */
    LwU8                                                                   lwrrDramclkInitialEstimatesIdx;

    /*!
     * Estimated metrics at the current observed operating point.
     */
    LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X                    observedEstimated;

    /*!
     * Corrections to apply across all estimated metrics.
     */
    LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X_CORRECTION         correction;

    /*!
     * @brief Structure holding all relevant data relating to input normalization for each
     *        normalization call
     */
    LW2080_CTRL_NNE_NNE_DESC_INPUT_NORM_STATUS                             inputNormStatus;

    /*!
     * Reason for ilwalidating the current observedMetrics structure
     */
    LwU8                                                                   ilwalidMetricsReason;

    /*!
     * Perf Target that is passed in by the controller
     */
    LwUFXP20_12                                                            perfTarget;

    /*!
     * Profiling data for collecting this set of observed metrics
     */
    LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X_OBSERVED_PROFILING profiling;
} LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X_OBSERVED;
typedef struct LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X_OBSERVED *PLW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X_OBSERVED;

/*!
 * Structure describing PERF_CF_PWR_MODEL_TGP_1X static information/POR.
 */
typedef struct LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_INFO_TGP_1X {
    /*!
     * Index into the PWR_POLICY BoardObjGrp of the workload policy used to
     * estimate core power.
     */
    LwBoardObjIdx workloadPwrPolicyIdx;
} LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_INFO_TGP_1X;
typedef struct LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_INFO_TGP_1X *PLW2080_CTRL_PERF_PERF_CF_PWR_MODEL_INFO_TGP_1X;

/*!
 * Structure describing the primary/estimated metrics for TGP_1X.
 */
typedef struct LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_TGP_1X {
    /*!
     * super - must be first member of the structure.
     */
    LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS                                           super;

    /*!
     * Estimated metrics from CWCC1X power policy.
     */
    LW2080_CTRL_PMGR_PWR_POLICY_PERF_CF_PWR_MODEL_ESTIMATED_METRICS_WORKLOAD_COMBINED_1X workloadCombined1x;

    /*!
     * Estimated frame buffer power in milliwatts.
     */
    LwU32                                                                                fbPwrmW;

    /*!
     * Other sources of power consumption in milliwatts.
     */
    LwU32                                                                                otherPwrmW;

    /*!
     * Total GPU Power in milliwatts.
     * estimatedTotalGpuPower = estimatedCorePower + fbPower + otherPower
     */
    LwU32                                                                                tgpPwrmW;
} LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_TGP_1X;
typedef struct LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_TGP_1X *PLW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_TGP_1X;

/*!
 * Parameters specific to the workload model that will be used internally
 */
typedef struct LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_TGP_1X_WORKLOAD_PARAMETERS {
    /*!
     * The current workload/active capacitance (w). Indexed by VOLT_RAIL.
     */
    LwUFXP20_12 workload[LW2080_CTRL_VOLT_VOLT_RAIL_CLIENT_MAX_RAILS];
} LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_TGP_1X_WORKLOAD_PARAMETERS;

/*!
 * Structure describing all observed metrics for TGP_1X
 */
typedef struct LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_TGP_1X_OBSERVED {
    /*!
     * super - must be first member of the structure.
     */
    LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS                            super;

    /*!
     * Parameters specific to the workload being modeled.
     */
    LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_TGP_1X_WORKLOAD_PARAMETERS workloadParameters;

    /*!
     * Observed frame buffer power in milliwatts.
     */
    LwU32                                                                 fbPwrmW;

    /*!
     * Other sources of power consumption in milliwatts.
     */
    LwU32                                                                 otherPwrmW;
} LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_TGP_1X_OBSERVED;
typedef struct LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_TGP_1X_OBSERVED *PLW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_TGP_1X_OBSERVED;

/*!
 * PERF_CF_PWR_MODEL type-specific data union. Discriminated by
 * PERF_CF_PWR_MODEL::super.type.
 */


/*!
 * Structure describing PERF_CF_PWR_MODEL static information/POR.
 * Implements the BOARDOBJ model/interface.
 */
typedef struct LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_INFO {
    /*!
     * LW2080_CTRL_BOARDOBJ super class.
     * Must always be first object in structure.
     */
    LW2080_CTRL_BOARDOBJ super;
    /*!
     * XAPI does not support using an element within another structure as a
     * discriminant. Placing this redundant type value here until that design
     * constraint can be fixed.
     */
    LwU8                 type;
    /*!
     * Type-specific data union.
     */
    union {
        LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_INFO_DLPPM_1X dlppm1x;
        LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_INFO_TGP_1X   tgp1x;
    } data;
} LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_INFO;
typedef struct LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_INFO *PLW2080_CTRL_PERF_PERF_CF_PWR_MODEL_INFO;

/*!
 * Structure describing PERF_CF_PWR_MODEL static information/POR.
 * Implements the BOARDOBJGRP model/interface.
 */
#define LW2080_CTRL_PERF_PERF_CF_PWR_MODELS_INFO_MESSAGE_ID (0xD0U)

typedef struct LW2080_CTRL_PERF_PERF_CF_PWR_MODELS_INFO {
    /*!
     * LW2080_CTRL_BOARDOBJGRP_E32 super class.
     * Must always be first object in structure.
     */
    LW2080_CTRL_BOARDOBJGRP_E32             super;
    /*!
     * Array of PERF_CF_PWR_MODEL structures. Has valid indexes corresponding to
     * the bits set in @ref super.objMask.
     */
    LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_INFO pwrModels[LW2080_CTRL_PERF_PERF_CF_PWR_MODELS_MAX_OBJECTS];
} LW2080_CTRL_PERF_PERF_CF_PWR_MODELS_INFO;
typedef struct LW2080_CTRL_PERF_PERF_CF_PWR_MODELS_INFO *PLW2080_CTRL_PERF_PERF_CF_PWR_MODELS_INFO;

/*!
 * LW2080_CTRL_CMD_PERF_PERF_CF_PWR_MODELS_GET_INFO
 *
 * This command returns PERF_CF_PWR_MODELS static object information/POR as
 * specified by the VBIOS in PERF_CF Table.
 *
 * See @ref LW2080_CTRL_PERF_PERF_CF_PWR_MODELS_INFO for documentation on
 * the parameters.
 *
 * Return values are specified per @ref BoardObjGrpGetInfo.
 */
#define LW2080_CTRL_CMD_PERF_PERF_CF_PWR_MODELS_GET_INFO (0x208020d0) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | LW2080_CTRL_PERF_PERF_CF_PWR_MODELS_INFO_MESSAGE_ID" */

/*!
 * Counts tracking how often DLPPM_1X guard railing has been applied.
 */
typedef struct LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_DLPPM_1X_GUARD_RAILS_STATUS {
    /*!
     * Count for each guard rail. Indexed via
     * @ref LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X_GUARD_RAIL_FLAGS_GUARD_RAIL_INDEX
     */
    LwU64_ALIGN32 guardRailCounts[LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X_GUARD_RAIL_FLAGS_NUM];

    /*!
     * Total number of times that iso-DRAMCLK/cross-primary-clock guard railing
     * has been run.
     */
    LwU64_ALIGN32 isoDramclkGuardRailCount;

    /*!
     * Total number of times that cross-DRAMCLK/iso-primary-clock guard railing
     * has been run.
     */
    LwU64_ALIGN32 isoPrimaryClkGuardRailCount;
} LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_DLPPM_1X_GUARD_RAILS_STATUS;

/*!
 * Structure representing the dynamic state of an
 * @ref LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_TYPE_DLPPM_1X object.
 */
typedef struct LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_STATUS_DLPPM_1X {
    /*!
     * Running state of violations of guard rails.
     */
    LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_DLPPM_1X_GUARD_RAILS_STATUS guardRailsStatus;
} LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_STATUS_DLPPM_1X;

/*!
 * Structure representing the dynamic state of an
 * @ref LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_TYPE_TGP_1X object.
 */
typedef struct LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_STATUS_TGP_1X {
    /*!
     * Temporary dummy data, actual status fields to be filled in at a later date.
     *
     * There is a possibility there will be no dynamic state for this class.
     */
    LwU32 rsvd;
} LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_STATUS_TGP_1X;

/*!
 * PERF_CF_PWR_MODEL type-specific data union. Discriminated by
 * PERF_CF_PWR_MODEL_DLPPM_1X::super.type.
 */


/*!
 * Structure representing the dynamic state of a PERF_CF_PWR_MODEL. Implements
 * the BOARDOBJ model/interface.
 */
typedef struct LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_STATUS {
    /*!
     * LW2080_CTRL_BOARDOBJ super class.
     * Must always be first object in structure.
     */
    LW2080_CTRL_BOARDOBJ super;

    /*!
     * XAPI does not support using an element within another structure as a
     * discriminant. Placing this redundant type value here until that design
     * constraint can be fixed.
     */
    LwU8                 type;

    /*!
     * Type-specific data union.
     */
    union {
        LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_STATUS_DLPPM_1X dlppm1x;
        LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_STATUS_TGP_1X   tgp1x;
    } data;
} LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_STATUS;

/*!
 * Structure representing the dynamic state of PERF_CF_PWR_MODELS.
 * Implements the BOARDOBJGRP model/interface.
 */
#define LW2080_CTRL_PERF_PERF_CF_PWR_MODELS_STATUS_MESSAGE_ID (0xD3U)

typedef struct LW2080_CTRL_PERF_PERF_CF_PWR_MODELS_STATUS {
    /*!
     * LW2080_CTRL_BOARDOBJGRP_E32 super class.
     * Must always be first object in structure.
     */
    LW2080_CTRL_BOARDOBJGRP_E32               super;

    /*!
     * Array of PERF_CF_PWR_MODEL structures. Has valid indexes corresponding to
     * the bits set in @ref super.objMask.
     */
    LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_STATUS pwrModels[LW2080_CTRL_PERF_PERF_CF_PWR_MODELS_MAX_OBJECTS];
} LW2080_CTRL_PERF_PERF_CF_PWR_MODELS_STATUS;

/*!
 * LW2080_CTRL_CMD_PERF_PERF_CF_PWR_MODELS_GET_STATUS
 *
 * This command returns PERF_CF_PWR_MODELS dynamic state.
 *
 * See @ref LW2080_CTRL_PERF_PERF_CF_PWR_MODELS_STATUS for documentation on
 * the parameters.
 *
 * Return values are specified per @ref BoardObjGrpGetStatus.
 */
#define LW2080_CTRL_CMD_PERF_PERF_CF_PWR_MODELS_GET_STATUS         (0x208020d3) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | LW2080_CTRL_PERF_PERF_CF_PWR_MODELS_STATUS_MESSAGE_ID" */

/*!
 * @brief   Maximum number of frequency inputs in
 *          @ref LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_SCALE_METRICS_INPUT
 */
#define LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_SCALE_METRICS_INPUT_MAX (LW2080_CTRL_CLK_CLK_DOMAIN_CLIENT_MAX_DOMAINS)

/*!
 * @brief   Represents input metrics at which to do a power model scale
 *          estimations.
 */
typedef struct LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_SCALE_METRICS_INPUT {
    /*!
     * @brief   Mask of CLK_DOMAIN indices to be selected from ::freqkHz as
     *          overridden input to PWR_MODEL_SCALE.
     */
    LW2080_CTRL_BOARDOBJGRP_MASK_E32 domainsMask;

    /*!
     * @brief   Frequencies (indexed by CLK_DOMAIN indices) to use as input to
     *          the estimation.
     */
    LwU32                            freqkHz[LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_SCALE_METRICS_INPUT_MAX];
} LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_SCALE_METRICS_INPUT;
typedef struct LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_SCALE_METRICS_INPUT *PLW2080_CTRL_PERF_PERF_CF_PWR_MODEL_SCALE_METRICS_INPUT;

/*!
 * @brief   Type-specific data for observed metrics.
 */


/*!
 * @brief   Structure representing the observed metrics for a PWR_MODEL.
 */
typedef struct LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_OBSERVED {
    /*!
     * XAPI does not support using an element within another structure as a
     * discriminant. Placing this redundant type value here until that design
     * constraint can be fixed.
     */
    LwU8 type;

    /*!
     * Type-specific data union.
     */
    union {
        LW2080_CTRL_PMGR_PWR_POLICY_PERF_CF_PWR_MODEL_OBSERVED_METRICS_WORKLOAD_SINGLE_1X   single1x;
        LW2080_CTRL_PMGR_PWR_POLICY_PERF_CF_PWR_MODEL_OBSERVED_METRICS_WORKLOAD_COMBINED_1X combined1x;
        LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X_OBSERVED                        dlppm1x;
        LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_TGP_1X_OBSERVED                          tgp1x;
    } data;
} LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_OBSERVED;

/*!
 * @brief   Type-specific data for estimated metrics.
 */


/*!
 * @brief   Structure representing the estimated metrics for a PWR_MODEL.
 */
typedef struct LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_ESTIMATED {
    /*!
     * XAPI does not support using an element within another structure as a
     * discriminant. Placing this redundant type value here until that design
     * constraint can be fixed.
     */
    LwU8 type;

    /*!
     * Type-specific data union.
     */
    union {
        LW2080_CTRL_PMGR_PWR_POLICY_PERF_CF_PWR_MODEL_ESTIMATED_METRICS_WORKLOAD_SINGLE_1X   single1x;
        LW2080_CTRL_PMGR_PWR_POLICY_PERF_CF_PWR_MODEL_ESTIMATED_METRICS_WORKLOAD_COMBINED_1X combined1x;
        LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X                                  dlppm1x;
        LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_TGP_1X                                    tgp1x;
    } data;
} LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_ESTIMATED;
typedef struct LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_ESTIMATED *PLW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_ESTIMATED;

/*!
 * Adds bounding parameters specific to @ref LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_TYPE_DLPPM_1X
 */
typedef struct LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_SCALE_BOUNDS_DLPPM_1X {
    /*!
     * The lower endpoint above which all primary clock inputs lie.
     */
    LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X endpointLo;

    /*!
     * The upper endpoint below which all primary clock inputs lie.
     */
    LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X endpointHi;

    /*!
     * Whether the guard rails should be applied or not.
     */
    LwBool                                              bGuardRailsApply;

    /*!
     * Whether the endpoint data is valid or not.
     */
    LwBool                                              bEndpointsValid;
} LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_SCALE_BOUNDS_DLPPM_1X;

/*!
 * @copydoc LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_SCALE_TYPE_PARAMS
 *
 * Adds parameters specific to @ref LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_TYPE_DLPPM_1X
 */
typedef struct LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_SCALE_TYPE_PARAMS_DLPPM_1X {
    /*!
     * Bounding parameters for the scale operation.
     */
    LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_SCALE_BOUNDS_DLPPM_1X            bounds;

    /*!
     * Profiling data for the scale exelwtion.
     */
    LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X_SCALE_PROFILING profiling;
} LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_SCALE_TYPE_PARAMS_DLPPM_1X;

/*!
 * @copydoc LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_SCALE_TYPE_PARAMS_TGP_1X
 *
 * Adds parameters specific to @ref LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_TYPE_TGP_1X
 */
typedef struct LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_SCALE_TYPE_PARAMS_TGP_1X {
    /*!
     * Temporary dummy data, actual bounding fields to be filled in at a later date.
     *
     * There is a possibility there will be no bounding inputs for this class.
     */
    LwU32 rsvd;
} LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_SCALE_TYPE_PARAMS_TGP_1X;

/*!
 * @brief   Structure for bounding information for PERF_CF_PWR_MODEL::scale
 *          interface.
 *
 * @deprecated  In favor of usage of
 *              @ref LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_SCALE_TYPE_PARAMS
 */
typedef struct LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_SCALE_BOUNDS {
    /*!
     * XAPI does not support using an element within another structure as a
     * discriminant. Placing this redundant type value here until that design
     * constraint can be fixed.
     *
     * @note    Maybe be set to
     *          @ref LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_TYPE_ILWALID
     *          to indicate that there are no bounding parameters for a
     *          particular call.
     */
    LwU8 type;

    /*!
     * Type-specific data union.
     */
    union {
        LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_SCALE_BOUNDS_DLPPM_1X dlppm1x;
    } data;
} LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_SCALE_BOUNDS;

/*!
 * @brief   Structure for type-specific parameters for PERF_CF_PWR_MODEL::scale
 *          interface.
 */
typedef struct LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_SCALE_TYPE_PARAMS {
    /*!
     * XAPI does not support using an element within another structure as a
     * discriminant. Placing this redundant type value here until that design
     * constraint can be fixed.
     *
     * @note    Maybe be set to
     *          @ref LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_TYPE_ILWALID
     *          to indicate that there are no bounding parameters for a
     *          particular call.
     */
    LwU8 type;

    /*!
     * Type-specific data union.
     */
    union {
        LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_SCALE_TYPE_PARAMS_DLPPM_1X dlppm1x;
        LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_SCALE_TYPE_PARAMS_TGP_1X   tgp1x;
    } data;
} LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_SCALE_TYPE_PARAMS;

/*!
 * @brief   Maximum number of metrics to estimated in one call to
 *          @ref LW2080_CTRL_CMD_PERF_PERF_CF_PWR_MODEL_SCALE
 */
#define LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_SCALE_ESTIMATED_METRICS_MAX 8U

/*!
 * @brief   Parameter structure to call into a PWR_MODEL's scale operation.
 */
#define LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_SCALE_PARAMS_MESSAGE_ID (0xD2U)

typedef struct LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_SCALE_PARAMS {
    /*!
     * @brief   Index of the PWR_MODEL to use to scale.
     */
    LwBoardObjIdx                                          pwrModelIdx;

    /*!
     * @brief   The observed metrics to use to scale.
     */
    LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_OBSERVED    observedMetrics;

    /*!
     * @brief   Number of metrics inputs to use to generate estimated metrics.
     */
    LwU8                                                   numEstimatedMetrics;

    /*!
     * @brief   Bounding parameters for this scale operation.
     *
     * @deprecated  In favor of usage of ::typeParams field.
     */
    LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_SCALE_BOUNDS        bounds;

    /*!
     * @brief   Bounding parameters for this scale operation.
     */
    LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_SCALE_TYPE_PARAMS   typeParams;

    /*!
     * @brief   The inputs to use to scale to.
     */
    LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_SCALE_METRICS_INPUT inputs[LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_SCALE_ESTIMATED_METRICS_MAX];

    /*!
     * @brief   The estimated metrics output by the scale.
     */
    LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_ESTIMATED   estimatedMetrics[LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_SCALE_ESTIMATED_METRICS_MAX];
} LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_SCALE_PARAMS;

#define LW2080_CTRL_CMD_PERF_PERF_CF_PWR_MODEL_SCALE                    (0x208020d2) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_SCALE_PARAMS_MESSAGE_ID" */

/* ---------------------------- Client PERF_CF Pwr Model Profiles ----------- */
/*!
 * Maximum number of CLIENT_PERF_CF_PWR_MODEL_PROFILEs which can be supported in
 * the RM or PMU.
 */
#define LW2080_CTRL_PERF_CLIENT_PERF_CF_PWR_MODEL_PROFILES_MAX_OBJECTS  LW2080_CTRL_BOARDOBJGRP_E32_MAX_OBJECTS

/*!
 * Macros for PERF_CF Pwr Model index
 */
#define LW2080_CTRL_PERF_CLIENT_PERF_CF_PWR_MODEL_PROFILE_INDEX_ILWALID LW2080_CTRL_BOARDOBJ_IDX_ILWALID

/*!
 * Macros for Client PERF_CF Power Model Profile types
 */
#define LW2080_CTRL_PERF_CLIENT_PERF_CF_PWR_MODEL_PROFILE_TYPE_TGP_1X   0x01

/*!
 * @brief   Type reserved for @ref LW2080_CTRL_PERF_CLIENT_PERF_CF_PWR_MODEL_PROFILE_ID
 *          enumerations.
 */
typedef LwU8 LW2080_CTRL_PERF_CLIENT_PERF_CF_PWR_MODEL_PROFILE_ID;

/*!
 * @defgroup LW2080_CTRL_PERF_CLIENT_PERF_CF_PWR_MODEL_PROFILE_ID_ENUM
 *
 * Enumeration of different named PWR_MODEL indexes. Of type
 * @ref LW2080_CTRL_PERF_CLIENT_PERF_CF_PWR_MODEL_PROFILE_ID.
 *
 * @{
 */
#define LW2080_CTRL_PERF_CLIENT_PERF_CF_PWR_MODEL_PROFILE_ID_DMMA_PERF         0x00
#define LW2080_CTRL_PERF_CLIENT_PERF_CF_PWR_MODEL_PROFILE_ID_DMMA_HIGH_K       0x01
#define LW2080_CTRL_PERF_CLIENT_PERF_CF_PWR_MODEL_PROFILE_ID_DMMA_LOW_K        0x02
#define LW2080_CTRL_PERF_CLIENT_PERF_CF_PWR_MODEL_PROFILE_ID_HMMA              0x03
#define LW2080_CTRL_PERF_CLIENT_PERF_CF_PWR_MODEL_PROFILE_ID_IMMA              0x04
#define LW2080_CTRL_PERF_CLIENT_PERF_CF_PWR_MODEL_PROFILE_ID_SGEMM             0x05
#define LW2080_CTRL_PERF_CLIENT_PERF_CF_PWR_MODEL_PROFILE_ID_TRANSFORMER       0x06
#define LW2080_CTRL_PERF_CLIENT_PERF_CF_PWR_MODEL_PROFILE_ID__RSVD_LOW         0x07
                                                                             // 0x07-0xF7 are reserved for expansion of enum
#define LW2080_CTRL_PERF_CLIENT_PERF_CF_PWR_MODEL_PROFILE_ID__RSVD_HIGH        0xF7
#define LW2080_CTRL_PERF_CLIENT_PERF_CF_PWR_MODEL_PROFILE_ID_LWSTOMER_LWSTOM_7 0xF8
#define LW2080_CTRL_PERF_CLIENT_PERF_CF_PWR_MODEL_PROFILE_ID_LWSTOMER_LWSTOM_6 0xF9
#define LW2080_CTRL_PERF_CLIENT_PERF_CF_PWR_MODEL_PROFILE_ID_LWSTOMER_LWSTOM_5 0xFA
#define LW2080_CTRL_PERF_CLIENT_PERF_CF_PWR_MODEL_PROFILE_ID_LWSTOMER_LWSTOM_4 0xFB
#define LW2080_CTRL_PERF_CLIENT_PERF_CF_PWR_MODEL_PROFILE_ID_LWSTOMER_LWSTOM_3 0xFC
#define LW2080_CTRL_PERF_CLIENT_PERF_CF_PWR_MODEL_PROFILE_ID_LWSTOMER_LWSTOM_2 0xFD
#define LW2080_CTRL_PERF_CLIENT_PERF_CF_PWR_MODEL_PROFILE_ID_LWSTOMER_LWSTOM_1 0xFE
#define LW2080_CTRL_PERF_CLIENT_PERF_CF_PWR_MODEL_PROFILE_ID_LWSTOMER_LWSTOM_0 0xFF
/*!@}*/

/*!
 * When used in a packed array, this is how many profile IDs exist.
 */
#define LW2080_CTRL_PERF_CLIENT_PERF_CF_PWR_MODEL_PROFILE_ID__NUM              (0xf) /* finn: Evaluated from "((0xFF - LW2080_CTRL_PERF_CLIENT_PERF_CF_PWR_MODEL_PROFILE_ID__RSVD_HIGH) + LW2080_CTRL_PERF_CLIENT_PERF_CF_PWR_MODEL_PROFILE_ID__RSVD_LOW)" */

/*!
 * Static POR info for CLIENT_PERF_CF_PWR_MODEL_PROFILE of type TGP_1X
 */
typedef struct LW2080_CTRL_PERF_CLIENT_PERF_CF_PWR_MODEL_PROFILE_INFO_TGP_1X {
    /*!
     * Parameters specific to the workload being modeled.
     */
    LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_TGP_1X_WORKLOAD_PARAMETERS workloadParameters;

    /*!
     * Characterized frame buffer power in milliwatts.
     */
    LwU32                                                                 fbPwrmW;

    /*!
     * Characterized sources of power consumption from other sources (i.e. not
     * core power or fb rail power) in milliwatts.
     */
    LwU32                                                                 otherPwrmW;
} LW2080_CTRL_PERF_CLIENT_PERF_CF_PWR_MODEL_PROFILE_INFO_TGP_1X;

/*!
 * CLIENT_PERF_CF_PWR_MODEL_PROFILE type-specific data union. Discriminated by
 * CLIENT_PERF_CF_PWR_MODEL_PROFILE::super.type.
 */


/*!
 * Structure describing CLIENT_PERF_CF_PWR_MODEL_PROFILE static information/POR.
 * Implements the BOARDOBJ model/interface.
 */
typedef struct LW2080_CTRL_PERF_CLIENT_PERF_CF_PWR_MODEL_PROFILE_INFO {
    /*!
     * LW2080_CTRL_BOARDOBJ super class.
     * Must always be first object in structure.
     */
    LW2080_CTRL_BOARDOBJ                                 super;

    /*!
     * XAPI does not support using an element within another structure as a
     * discriminant. Placing this redundant type value here until that design
     * constraint can be fixed.
     */
    LwU8                                                 type;

    /*!
     * Unique named identifier for the particular client profile. To be used as
     * input to client-wrapped scaling operations.
     */
    LW2080_CTRL_PERF_CLIENT_PERF_CF_PWR_MODEL_PROFILE_ID profileId;

    /*!
     * Semantic index of internal PERF_CF_PWR_MODEL used for scaling.
     * This semantic index is derived from the boardobj type of the profile.
     */
    LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_SEMANTIC_INDEX    pwrModelSemanticIndex;

    /*!
     * Type-specific data union.
     */
    union {
        LW2080_CTRL_PERF_CLIENT_PERF_CF_PWR_MODEL_PROFILE_INFO_TGP_1X tgp1x;
    } data;
} LW2080_CTRL_PERF_CLIENT_PERF_CF_PWR_MODEL_PROFILE_INFO;
typedef struct LW2080_CTRL_PERF_CLIENT_PERF_CF_PWR_MODEL_PROFILE_INFO *PLW2080_CTRL_PERF_CLIENT_PERF_CF_PWR_MODEL_PROFILE_INFO;

/*!
 * Structure describing CLIENT_PERF_CF_PWR_MODEL_PROFILE static information/POR.
 * Implements the BOARDOBJGRP model/interface.
 */
#define LW2080_CTRL_PERF_CLIENT_PERF_CF_PWR_MODEL_PROFILES_INFO_MESSAGE_ID (0xD4U)

typedef struct LW2080_CTRL_PERF_CLIENT_PERF_CF_PWR_MODEL_PROFILES_INFO {
    /*!
     * LW2080_CTRL_BOARDOBJGRP_E32 super class.
     * Must always be first object in structure.
     */
    LW2080_CTRL_BOARDOBJGRP_E32                            super;

    /*!
     * Array of CLIENT_PERF_CF_PWR_MODEL_PROFILE structures. Has valid indexes corresponding to
     * the bits set in @ref super.objMask.
     */
    LW2080_CTRL_PERF_CLIENT_PERF_CF_PWR_MODEL_PROFILE_INFO clientPwrModelProfiles[LW2080_CTRL_PERF_CLIENT_PERF_CF_PWR_MODEL_PROFILES_MAX_OBJECTS];
} LW2080_CTRL_PERF_CLIENT_PERF_CF_PWR_MODEL_PROFILES_INFO;
typedef struct LW2080_CTRL_PERF_CLIENT_PERF_CF_PWR_MODEL_PROFILES_INFO *PLW2080_CTRL_PERF_CLIENT_PERF_CF_PWR_MODEL_PROFILES_INFO;

/*!
 * LW2080_CTRL_CMD_PERF_CLIENT_PERF_CF_PWR_MODEL_PROFILES_GET_INFO
 *
 * This command returns CLIENT_PERF_CF_PWR_MODEL_PROFILES static object information/POR as
 * specified by the InfoRom in CLIENT_PERF_CF_PWR_MODEL_PROFILES Table.
 *
 * See @ref LW2080_CTRL_PERF_CLIENT_PERF_CF_PWR_MODEL_PROFILES_INFO for documentation on
 * the parameters.
 *
 * Return values are specified per @ref BoardObjGrpGetInfo.
 */
#define LW2080_CTRL_CMD_PERF_CLIENT_PERF_CF_PWR_MODEL_PROFILES_GET_INFO (0x208020d4) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | LW2080_CTRL_PERF_CLIENT_PERF_CF_PWR_MODEL_PROFILES_INFO_MESSAGE_ID" */

/*!
 * Inputs to CLIENT_PERF_CF_PWR_MODEL_SCALE.
 *
 * @ref LW2080_CTRL_CLK_CLIENT_CLK_DOMAINS_INFO for client clock domain specifics.
 */
typedef struct LW2080_CTRL_PERF_CLIENT_PERF_CF_PWR_MODEL_PROFILE_SCALE_INPUT {
    /*!
     * @brief   Mask of CLIENT_CLK_DOMAIN indices to be selected from ::freqkHz as
     *          overridden input to CLIENT_PERF_CF_PWR_MODEL_PROFILE_SCALE.
     */
    LW2080_CTRL_BOARDOBJGRP_MASK_E32 clientDomainsMask;

    /*!
     * @brief   Frequencies (indexed by CLIENT_CLK_DOMAIN indices) to use as input to
     *          the estimation.
     */
    LwU32                            freqkHz[LW2080_CTRL_CLK_CLK_DOMAIN_CLIENT_MAX_DOMAINS];
} LW2080_CTRL_PERF_CLIENT_PERF_CF_PWR_MODEL_PROFILE_SCALE_INPUT;


/*!
 * Outputs of CLIENT_PERF_CF_PWR_MODEL_SCALE.
 */
typedef struct LW2080_CTRL_PERF_CLIENT_PERF_CF_PWR_MODEL_PROFILE_SCALE_OUTPUT {
    /*!
     * @brief   The estimated Total GPU power.
     */
    LwU32 totalGpuPwrmW;
} LW2080_CTRL_PERF_CLIENT_PERF_CF_PWR_MODEL_PROFILE_SCALE_OUTPUT;

/*!
 * Structure describing CLIENT_PERF_CF_PWR_MODEL_PROFILE scaling inputs/outputs.
 */
#define LW2080_CTRL_PERF_CLIENT_PERF_CF_PWR_MODEL_PROFILE_SCALE_MESSAGE_ID (0xD5U)

typedef struct LW2080_CTRL_PERF_CLIENT_PERF_CF_PWR_MODEL_PROFILE_SCALE {
    /*!
     * [in] Unique named identifier for the client profile to be scaled.
     */
    LW2080_CTRL_PERF_CLIENT_PERF_CF_PWR_MODEL_PROFILE_ID           profileId;

    /*!
     * [in] Inputs to scaling operation.
     */
    LW2080_CTRL_PERF_CLIENT_PERF_CF_PWR_MODEL_PROFILE_SCALE_INPUT  input;

    /*!
     * [out] Outputs of scaling operation.
     */
    LW2080_CTRL_PERF_CLIENT_PERF_CF_PWR_MODEL_PROFILE_SCALE_OUTPUT output;
} LW2080_CTRL_PERF_CLIENT_PERF_CF_PWR_MODEL_PROFILE_SCALE;

/*!
 * LW2080_CTRL_CMD_PERF_CLIENT_PERF_CF_PWR_MODEL_PROFILE_SCALE
 *
 * This command scales a particular CLIENT_PERF_CF_PWR_MODEL_PROFILE using the
 * provided clock frequency inputs and provides an estimated power tuple as
 * output.
 *
 * See @ref LW2080_CTRL_PERF_CLIENT_PERF_CF_PWR_MODEL_PROFILES_INFO for information
 * on which profiles are available for scaling.
 *
 * See @ref LW2080_CTRL_PERF_CLIENT_PERF_CF_PWR_MODEL_PROFILE_SCALE for documentation on
 * the parameters.
 *
 * Return values are specified per @ref BoardObjGrpGetInfo.
 */
#define LW2080_CTRL_CMD_PERF_CLIENT_PERF_CF_PWR_MODEL_PROFILE_SCALE (0x208020d5) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | LW2080_CTRL_PERF_CLIENT_PERF_CF_PWR_MODEL_PROFILE_SCALE_MESSAGE_ID" */

/*!
 * @brief   Type reserved for @ref LW2080_CTRL_PERF_CLIENT_PERF_CF_PWR_MODEL_PROFILE_SCALE_SETUP_SETTING_TYPE
 *          enumerations.
 */
typedef LwU8 LW2080_CTRL_PERF_CLIENT_PERF_CF_PWR_MODEL_PROFILE_SCALE_SETUP_SETTING_TYPE;

/*!
 * @defgroup LW2080_CTRL_PERF_CLIENT_PERF_CF_PWR_MODEL_PROFILE_SCALE_SETUP_SETTING_TYPE_ENUM
 *
 * Enumeration of different LW2080_CTRL_PERF_CLIENT_PERF_CF_PWR_MODEL_PROFILE_SCALE_SETUP
 * setting types. Of type
 * @ref LW2080_CTRL_PERF_CLIENT_PERF_CF_PWR_MODEL_PROFILE_SCALE_SETUP_SETTING_TYPE.
 *
 * @{
 */
#define LW2080_CTRL_PERF_CLIENT_PERF_CF_PWR_MODEL_PROFILE_SCALE_SETUP_SETTING_TYPE_GPU_TEMPERATURE 0x00
/*!@}*/

/*!
 * LW2080_CTRL_PERF_CLIENT_PERF_CF_PWR_MODEL_PROFILE_SCALE_SETUP_SETTING for
 * overriding the sw state for gpu temperature (avg, max, etc).
 */
typedef struct LW2080_CTRL_PERF_CLIENT_PERF_CF_PWR_MODEL_PROFILE_SCALE_SETUP_SETTING_DATA_GPU_TEMPERATURE {
    /*!
     * GPU temp override
     */
    LwTemp temperature;
} LW2080_CTRL_PERF_CLIENT_PERF_CF_PWR_MODEL_PROFILE_SCALE_SETUP_SETTING_DATA_GPU_TEMPERATURE;
typedef struct LW2080_CTRL_PERF_CLIENT_PERF_CF_PWR_MODEL_PROFILE_SCALE_SETUP_SETTING_DATA_GPU_TEMPERATURE *PLW2080_CTRL_PERF_CLIENT_PERF_CF_PWR_MODEL_PROFILE_SCALE_SETUP_SETTING_DATA_GPU_TEMPERATURE;

/*!
 * Data union of possible
 * LW2080_CTRL_PERF_CLIENT_PERF_CF_PWR_MODEL_PROFILE_SCALE_SETUP_SETTING-s.
 */


/*!
 * Individual custom setting to be used within
 * LW2080_CTRL_PERF_CLIENT_PERF_CF_PWR_MODEL_PROFILE_SCALE_SETUP.
 */
typedef struct LW2080_CTRL_PERF_CLIENT_PERF_CF_PWR_MODEL_PROFILE_SCALE_SETUP_SETTING {
    /*!
     * XAPI does not support using an element within another structure as a
     * discriminant. Placing this redundant type value here until that design
     * constraint can be fixed.
     */
    LW2080_CTRL_PERF_CLIENT_PERF_CF_PWR_MODEL_PROFILE_SCALE_SETUP_SETTING_TYPE type;

    /*!
     * Type-specific data union.
     */
    union {
        LW2080_CTRL_PERF_CLIENT_PERF_CF_PWR_MODEL_PROFILE_SCALE_SETUP_SETTING_DATA_GPU_TEMPERATURE gpuTemperature;
    } data;
} LW2080_CTRL_PERF_CLIENT_PERF_CF_PWR_MODEL_PROFILE_SCALE_SETUP_SETTING;
typedef struct LW2080_CTRL_PERF_CLIENT_PERF_CF_PWR_MODEL_PROFILE_SCALE_SETUP_SETTING *PLW2080_CTRL_PERF_CLIENT_PERF_CF_PWR_MODEL_PROFILE_SCALE_SETUP_SETTING;

#define LW2080_CTRL_PERF_CLIENT_PERF_CF_PWR_MODEL_PROFILE_SCALE_SETUP_MAX_SETTINGS (8U)

/*!
 * Structure describing setup settings required before calling of
 * LW2080_CTRL_CMD_PERF_CLIENT_PERF_CF_PWR_MODEL_PROFILE_SCALE.
 *
 * In addition to any settings that are changed, the gpu clocks will be locked to IDLE
 * values until the setup is undone.
 */
#define LW2080_CTRL_PERF_CLIENT_PERF_CF_PWR_MODEL_PROFILE_SCALE_SETUP_MESSAGE_ID (0xD6U)

typedef struct LW2080_CTRL_PERF_CLIENT_PERF_CF_PWR_MODEL_PROFILE_SCALE_SETUP {
    /*!
     * [in] If LW_TRUE, this indicates that the list of setup settings described
     * in ::settings below will be applied.
     *
     * If LW_FALSE, any lwrrently applied changes will be reverted thus removing
     * any faked gpu state.
     */
    LwBool                                                                bSetup;

    /*!
     * Number of settings specified in ::settings below
     */
    LwU8                                                                  numSettings;

    /*!
     * List of settings to be applied when ::bSetup == LW_TRUE (or reverted if
     * ::bSetup == LW_FALSE).
     */
    LW2080_CTRL_PERF_CLIENT_PERF_CF_PWR_MODEL_PROFILE_SCALE_SETUP_SETTING settings[LW2080_CTRL_PERF_CLIENT_PERF_CF_PWR_MODEL_PROFILE_SCALE_SETUP_MAX_SETTINGS];
} LW2080_CTRL_PERF_CLIENT_PERF_CF_PWR_MODEL_PROFILE_SCALE_SETUP;
typedef struct LW2080_CTRL_PERF_CLIENT_PERF_CF_PWR_MODEL_PROFILE_SCALE_SETUP *PLW2080_CTRL_PERF_CLIENT_PERF_CF_PWR_MODEL_PROFILE_SCALE_SETUP;

/*!
 * LW2080_CTRL_CMD_PERF_CLIENT_PERF_CF_PWR_MODEL_PROFILE_SCALE_SETUP
 *
 * This command does any setup/teardown of select gpu SW state prior-to/after
 * any calls of LW2080_CTRL_CMD_PERF_CLIENT_PERF_CF_PWR_MODEL_PROFILE_SCALE.
 *
 * See @ref LW2080_CTRL_PERF_CLIENT_PERF_CF_PWR_MODEL_PROFILE_SCALE_SETUP for
 * documentation on the parameters.
 *
 * Return values are specified per @ref BoardObjGrpGetInfo.
 */
#define LW2080_CTRL_CMD_PERF_CLIENT_PERF_CF_PWR_MODEL_PROFILE_SCALE_SETUP     (0x208020d6) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | LW2080_CTRL_PERF_CLIENT_PERF_CF_PWR_MODEL_PROFILE_SCALE_SETUP_MESSAGE_ID" */

/* ---------------------------- PERF_CF Controllers -------------------------- */

/*!
 * Macros for PERF_CF Controller index
 */
#define LW2080_CTRL_PERF_PERF_CF_CONTROLLER_INDEX_ILWALID                     LW2080_CTRL_BOARDOBJ_IDX_ILWALID

/*!
 * Macros for PERF_CF Controller types
 */
#define LW2080_CTRL_PERF_PERF_CF_CONTROLLER_TYPE_UTIL                         0x00
#define LW2080_CTRL_PERF_PERF_CF_CONTROLLER_TYPE_RP_PC_1X                     0x01 // Deprecated
#define LW2080_CTRL_PERF_PERF_CF_CONTROLLER_TYPE_OPTP_2X                      0x02
#define LW2080_CTRL_PERF_PERF_CF_CONTROLLER_TYPE_DLPPC_1X                     0x03
#define LW2080_CTRL_PERF_PERF_CF_CONTROLLER_TYPE_MEM_TUNE_1X                  0x04
#define LW2080_CTRL_PERF_PERF_CF_CONTROLLER_TYPE_UTIL_2X                      0x05
#define LW2080_CTRL_PERF_PERF_CF_CONTROLLER_TYPE_MEM_TUNE_2X                  0x06

/*!
 * Macros for controlling clock domain test modes
 */
#define LW2080_CTRL_PERF_PERF_CF_CONTROLLER_MODE_DISABLED                     0x00
#define LW2080_CTRL_PERF_PERF_CF_CONTROLLER_MODE_DRAMCLK_ONLY                 0x01
#define LW2080_CTRL_PERF_PERF_CF_CONTROLLER_MODE_DRAMCLK_PWR_CEIL_GPCCLK      0x02
#define LW2080_CTRL_PERF_PERF_CF_CONTROLLER_MODE_MAX                          0x03

/*!
 * Maximum number of PERF_CF_CONTROLLERs which can be supported in the RM or PMU.
 */
#define LW2080_CTRL_PERF_PERF_CF_CONTROLLERS_MAX_OBJECTS                      LW2080_CTRL_BOARDOBJGRP_E32_MAX_OBJECTS

/*!
 * Power gating % sampling is not needed.
 */
#define LW2080_CTRL_PERF_PERF_CF_CONTROLLER_INFO_UTIL_PG_TOPOLOGY_IDX_NONE    0xFF
#define LW2080_CTRL_PERF_PERF_CF_CONTROLLER_INFO_UTIL_2X_PG_TOPOLOGY_IDX_NONE 0xFF
#define LW2080_CTRL_PERF_PERF_CF_CONTROLLER_INFO_OPTP_2X_PG_TOPOLOGY_IDX_NONE 0xFF

/*!
 * Structure describing PERF_CF_CONTROLLER_UTIL static information/POR.
 */
typedef struct LW2080_CTRL_PERF_PERF_CF_CONTROLLER_INFO_UTIL {
    /*!
     * Clock domain index this controller is controlling.
     */
    LwU8   clkDomIdx;
    /*!
     * Index of topology for clock frequency input to the controller.
     */
    LwU8   clkTopologyIdx;
    /*!
     * Index of topology for power gating % input to the controller. Can be
     * LW2080_CTRL_PERF_PERF_CF_CONTROLLER_INFO_UTIL_PG_TOPOLOGY_IDX_NONE.
     */
    LwU8   pgTopologyIdx;
    /*!
     * Do not program clocks if TRUE
     */
    LwBool bDoNotProgram;
    /*!
     * Support for feature of scaling the utilization threshold in proportion
     * with number of VM slots that are active in vGPU's scheduler:
     * LW_TRUE -> Feature is supported
     * LW_FALSE -> Feature is not supported
     */
    LwBool bUtilThresholdScaleByVMCountSupported;
} LW2080_CTRL_PERF_PERF_CF_CONTROLLER_INFO_UTIL;
typedef struct LW2080_CTRL_PERF_PERF_CF_CONTROLLER_INFO_UTIL *PLW2080_CTRL_PERF_PERF_CF_CONTROLLER_INFO_UTIL;

/*!
 * Structure describing PERF_CF_CONTROLLER_UTIL_2X static information/POR.
 */
typedef struct LW2080_CTRL_PERF_PERF_CF_CONTROLLER_INFO_UTIL_2X {
    /*!
     * super class.
     */
    LW2080_CTRL_PERF_PERF_CF_CONTROLLER_INFO_UTIL super;
    /*!
     * Number of IIR samples above target before increasing clocks.
     */
    LwU8                                          IIRCountInc;
    /*!
     * Number of IIR samples below target before decreasing clocks.
     */
    LwU8                                          IIRCountDec;
} LW2080_CTRL_PERF_PERF_CF_CONTROLLER_INFO_UTIL_2X;
typedef struct LW2080_CTRL_PERF_PERF_CF_CONTROLLER_INFO_UTIL_2X *PLW2080_CTRL_PERF_PERF_CF_CONTROLLER_INFO_UTIL_2X;

#define LW2080_CTRL_PERF_PERF_CF_CONTROLLER_MEM_TUNE_TARGET_TOTAL_ACTIVATES  0
#define LW2080_CTRL_PERF_PERF_CF_CONTROLLER_MEM_TUNE_TARGET_READ_ACCESS      1
#define LW2080_CTRL_PERF_PERF_CF_CONTROLLER_MEM_TUNE_TARGET_WRITE_ACCESS     2
#define LW2080_CTRL_PERF_PERF_CF_CONTROLLER_MEM_TUNE_TARGET_RD_SUB_WR_ACCESS 3

/*!
 * Max supported targets for memory tuning controller.
 */
#define LW2080_CTRL_PERF_PERF_CF_CONTROLLER_MEM_TUNE_TARGET_MAX              8

/*!
 * Max supported BAPM signals per target for memory tuning controller.
 */
#define LW2080_CTRL_PERF_PERF_CF_CONTROLLER_MEM_TUNE_TARGET_PM_SIGNAL_MAX    4

/*!
 * Structure holding the BA PM signals for given target.
 */
typedef struct LW2080_CTRL_PERF_PERF_CF_CONTROLLER_INFO_MEM_TUNE_TARGET {
    /*!
     * Total number of valid signals.
     */
    LwU8          numSignals;
    /*!
     * Array of BA PM signals being tracked by this controller.
     */
    LwBoardObjIdx signalIdx[LW2080_CTRL_PERF_PERF_CF_CONTROLLER_MEM_TUNE_TARGET_PM_SIGNAL_MAX];
} LW2080_CTRL_PERF_PERF_CF_CONTROLLER_INFO_MEM_TUNE_TARGET;

/*!
 * Structure holding POR information of all targets.
 */
typedef struct LW2080_CTRL_PERF_PERF_CF_CONTROLLER_INFO_MEM_TUNE_TARGETS {
    /*!
     * Total number of valid targets.
     */
    LwU8                                                     numTargets;
    /*!
     * Array of target's static POR information.
     * @ref LW2080_CTRL_PERF_PERF_CF_CONTROLLER_INFO_MEM_TUNE_TARGET
     */
    LW2080_CTRL_PERF_PERF_CF_CONTROLLER_INFO_MEM_TUNE_TARGET target[LW2080_CTRL_PERF_PERF_CF_CONTROLLER_MEM_TUNE_TARGET_MAX];
} LW2080_CTRL_PERF_PERF_CF_CONTROLLER_INFO_MEM_TUNE_TARGETS;

/*!
 * Predefined value of tFAW to program when PMU is going down due to
 * unforeseen exceptions events such as halt or breakpoint or ...
 *
 * This value is being used by PMU, FBFLCN and ACR unload ucode.
 * Please do not forget to update PERF_PERF_CF_MEM_TUNE_EXCEPTION_TFAW_VAL
 * when updating this value - FBFLCN and ACR
 * Also ACR unload ucode binary update will require HS signing.
 *
 */
#define LW2080_CTRL_PERF_PERF_CF_MEM_TUNE_EXCEPTION_TFAW_VAL  80U

/*!
 * Predefined value of tFAW to program when memory tuning controller is
 * engaged.
 */
#define LW2080_CTRL_PERF_PERF_CF_MEM_TUNE_ENGAGE_TFAW_VAL     56U

/*!
 * Predefined max value of tFAW to program when memory tuning controller is
 * engaged.
 */
#define LW2080_CTRL_PERF_PERF_CF_MEM_TUNE_ENGAGE_TFAW_VAL_MAX 80U

/*!
 * Predefined value of tFAW to indicate when memory tuning controller is
 * disengaged.
 */
#define LW2080_CTRL_PERF_PERF_CF_MEM_TUNE_DISENGAGE_TFAW_VAL  LW_U8_MAX

/*!
 * Predefined value of tFAW to program when memory tuning controller is
 * always on (sandbag driver).
 */
#define LW2080_CTRL_PERF_PERF_CF_MEM_TUNE_ALWAYS_ON_TFAW_VAL  48U

/*!
 * Structure describing PERF_CF_CONTROLLER_MEM_TUNE static information/POR.
 */
typedef struct LW2080_CTRL_PERF_PERF_CF_CONTROLLER_INFO_MEM_TUNE {
    /*!
     * Index into PERF_CF sensor table for PERF_CF_SENSOR representing
     * all PM's.
     */
    LwBoardObjIdx                                             pmSensorIdx;
    /*!
     * Index of topology for PTIMER input to the controller.
     */
    LwBoardObjIdx                                             timerTopologyIdx;
    /*!
     * Targets static POR information.
     * @ref LW2080_CTRL_PERF_PERF_CF_CONTROLLER_INFO_MEM_TUNE_TARGETS
     */
    LW2080_CTRL_PERF_PERF_CF_CONTROLLER_INFO_MEM_TUNE_TARGETS targets;
} LW2080_CTRL_PERF_PERF_CF_CONTROLLER_INFO_MEM_TUNE;

/*!
 * Structure describing PERF_CF_CONTROLLER_MEM_TUNE_2X static information/POR.
 */
typedef struct LW2080_CTRL_PERF_PERF_CF_CONTROLLER_INFO_MEM_TUNE_2X {
    /*!
     * TODO-Chandrashis: Introduce required fields.
     */
    LwU8 rsvd;
} LW2080_CTRL_PERF_PERF_CF_CONTROLLER_INFO_MEM_TUNE_2X;

/*!
 * Structure describing PERF_CF_CONTROLLER_OPTP_2X static information/POR.
 */
typedef struct LW2080_CTRL_PERF_PERF_CF_CONTROLLER_INFO_OPTP_2X {
    /*!
     * Do not cap clocks below this floor (in kHz) .
     */
    LwU32 freqFloorkHz;
    /*!
     * Index of topology for graphics clock frequency.
     */
    LwU8  grClkTopologyIdx;
    /*!
     * Index of topology for video clock frequency.
     */
    LwU8  vidClkTopologyIdx;
    /*!
     * Index of topology for power gating %. Can be
     * LW2080_CTRL_PERF_PERF_CF_CONTROLLER_INFO_OPTP_2X_PG_TOPOLOGY_IDX_NONE.
     */
    LwU8  pgTopologyIdx;
} LW2080_CTRL_PERF_PERF_CF_CONTROLLER_INFO_OPTP_2X;
typedef struct LW2080_CTRL_PERF_PERF_CF_CONTROLLER_INFO_OPTP_2X *PLW2080_CTRL_PERF_PERF_CF_CONTROLLER_INFO_OPTP_2X;

/*!
 * Structure describing PERF_CF_CONTROLLER_DLPPC_1X rail static information/POR.
 */
typedef struct LW2080_CTRL_PERF_PERF_CF_CONTROLLER_DLPPC_1X_RAIL {
    /*!
     * Clock domain that this rail description is for.
     */
    LwBoardObjIdx                                clkDomIdx;

    /*!
     * Index to VOLT_RAIL table for this rail.
     */
    LwBoardObjIdx                                voltRailIdx;

    /*!
     * Set of VR input power policy relationships for this rail.
     */
    LW2080_CTRL_PMGR_PWR_POLICY_RELATIONSHIP_SET pwrInRelSet;

    /*!
     * Set of VR input power policy relationships for this rail.
     */
    LW2080_CTRL_PMGR_PWR_POLICY_RELATIONSHIP_SET pwrOutRelSet;

    /*!
     * Max-limit index for this clock domain.
     */
    LwBoardObjIdx                                maxLimitIdx;
} LW2080_CTRL_PERF_PERF_CF_CONTROLLER_DLPPC_1X_RAIL;
typedef struct LW2080_CTRL_PERF_PERF_CF_CONTROLLER_DLPPC_1X_RAIL *PLW2080_CTRL_PERF_PERF_CF_CONTROLLER_DLPPC_1X_RAIL;

/*!
 * Maximum number of core rails supported by DLPPC_1X.
 */
#define LW2080_CTRL_PERF_PERF_CF_CONTROLLER_DLPPC_1X_MAX_CORE_RAILS (2)

/*!
 * Structure describing DLPPC_1X relevant information for the core rails.
 */
typedef struct LW2080_CTRL_PERF_PERF_CF_CONTROLLER_DLPPC_1X_CORE_RAIL {
    /*!
     * Set of rails belonging to the core.
     */
    LW2080_CTRL_PERF_PERF_CF_CONTROLLER_DLPPC_1X_RAIL rails[LW2080_CTRL_PERF_PERF_CF_CONTROLLER_DLPPC_1X_MAX_CORE_RAILS];

    /*!
     * Number of valid elements in @ref rails
     */
    LwU8                                              numRails;

    /*!
     * Set of all combined input power relationships.
     */
    LW2080_CTRL_PMGR_PWR_POLICY_RELATIONSHIP_SET      pwrInRelSet;
} LW2080_CTRL_PERF_PERF_CF_CONTROLLER_DLPPC_1X_CORE_RAIL;
typedef struct LW2080_CTRL_PERF_PERF_CF_CONTROLLER_DLPPC_1X_CORE_RAIL *PLW2080_CTRL_PERF_PERF_CF_CONTROLLER_DLPPC_1X_CORE_RAIL;

/*!
 * Structure describing PERF_CF_CONTROLLER_DLPPC_1X static information/POR.
 */
typedef struct LW2080_CTRL_PERF_PERF_CF_CONTROLLER_INFO_DLPPC_1X {
    /*!
     * Core rail state.
     */
    LW2080_CTRL_PERF_PERF_CF_CONTROLLER_DLPPC_1X_CORE_RAIL coreRail;

    /*!
     * FBVDD rail state.
     */
    LW2080_CTRL_PERF_PERF_CF_CONTROLLER_DLPPC_1X_RAIL      fbRail;

    /*!
     * TGP policy relationship set.
     */
    LW2080_CTRL_PMGR_PWR_POLICY_RELATIONSHIP_SET           tgpPolRelSet;

    /*!
     * Index into PERF_CF_PWR_MODELS for the DLPPM to use.
     */
    LwBoardObjIdx                                          dlppmIdx;

    /*!
     * Original Hysteresis values that were loaded for changing to lower dramclk freq
     */
    LwU8                                                   origDramHysteresisCountDec;

    /*!
     * Original Hysteresis values that were loaded for changing to higher dramclk freq
     */
    LwU8                                                   origDramHysteresisCountInc;

    /*!
     * Maximum percent error in dramclk frequency deviation. 1.12 FXP number
     */
    LwU32                                                  dramclkMatchTolerance;

    /*!
     * Whether the controller supports OPTP behavior
     */
    LwBool                                                 bOptpSupported;

    /*!
     * Whether the controller supports LWCA
     */
    LwBool                                                 bLwdaSupported;
} LW2080_CTRL_PERF_PERF_CF_CONTROLLER_INFO_DLPPC_1X;
typedef struct LW2080_CTRL_PERF_PERF_CF_CONTROLLER_INFO_DLPPC_1X *PLW2080_CTRL_PERF_PERF_CF_CONTROLLER_INFO_DLPPC_1X;

/*!
 * PERF_CF_CONTROLLER type-specific data union. Discriminated by
 * PERF_CF_CONTROLLER::super.type.
 */


/*!
 * Common topology index for controller is not needed.
 */
#define LW2080_CTRL_PERF_PERF_CF_CONTROLLER_INFO_TOPOLOGY_IDX_NONE 0xFF

/*!
 * Structure describing PERF_CF_CONTROLLER static information/POR.
 * Implements the BOARDOBJ model/interface.
 */
typedef struct LW2080_CTRL_PERF_PERF_CF_CONTROLLER_INFO {
    /*!
     * LW2080_CTRL_BOARDOBJ super class.
     * Must always be first object in structure.
     */
    LW2080_CTRL_BOARDOBJ super;
    /*!
     * XAPI does not support using an element within another structure as a
     * discriminant. Placing this redundant type value here until that design
     * constraint can be fixed.
     */
    LwU8                 type;
     /*!
     * This controller's sampling period is multiplier * baseSamplingPeriodms.
     */
    LwU8                 samplingMultiplier;
    /*!
     * Index of topology input to the controller. Can be
     * LW2080_CTRL_PERF_PERF_CF_CONTROLLER_INFO_TOPOLOGY_IDX_NONE.
     */
    LwU8                 topologyIdx;
    /*!
     * Type-specific data union.
     */
    union {
        LW2080_CTRL_PERF_PERF_CF_CONTROLLER_INFO_UTIL        util;
        LW2080_CTRL_PERF_PERF_CF_CONTROLLER_INFO_UTIL_2X     util2x;
        LW2080_CTRL_PERF_PERF_CF_CONTROLLER_INFO_OPTP_2X     optp2x;
        LW2080_CTRL_PERF_PERF_CF_CONTROLLER_INFO_DLPPC_1X    dlppc1x;
        LW2080_CTRL_PERF_PERF_CF_CONTROLLER_INFO_MEM_TUNE    memTune;
        LW2080_CTRL_PERF_PERF_CF_CONTROLLER_INFO_MEM_TUNE_2X memTune2x;
    } data;
} LW2080_CTRL_PERF_PERF_CF_CONTROLLER_INFO;
typedef struct LW2080_CTRL_PERF_PERF_CF_CONTROLLER_INFO *PLW2080_CTRL_PERF_PERF_CF_CONTROLLER_INFO;

/*!
 * Structure describing PERF_CF_CONTROLLER static information/POR.
 * Implements the BOARDOBJGRP model/interface.
 */
#define LW2080_CTRL_PERF_PERF_CF_CONTROLLERS_INFO_MESSAGE_ID (0xC8U)

typedef struct LW2080_CTRL_PERF_PERF_CF_CONTROLLERS_INFO {
    /*!
     * LW2080_CTRL_BOARDOBJGRP_E32 super class.
     * Must always be first object in structure.
     */
    LW2080_CTRL_BOARDOBJGRP_E32              super;
    /*!
     * Each PERF_CF controller run at a multiple of this base sampling period.
     */
    LwU16                                    baseSamplingPeriodms;
    /*!
     * Support for feature of scaling the utilization threshold in proportion
     * with number of VM slots that are active in vGPU's scheduler:
     * LW_TRUE -> Feature is supported by one or more utilization controllers.
     * LW_FALSE -> Feature is not supported on any of the utilization controller.
     */
    LwBool                                   bUtilThresholdScaleByVMCountSupported;
    /*!
     * Array of PERF_CF_CONTROLLER structures. Has valid indexes corresponding to
     * the bits set in @ref super.objMask.
     */
    LW2080_CTRL_PERF_PERF_CF_CONTROLLER_INFO controllers[LW2080_CTRL_PERF_PERF_CF_CONTROLLERS_MAX_OBJECTS];
} LW2080_CTRL_PERF_PERF_CF_CONTROLLERS_INFO;
typedef struct LW2080_CTRL_PERF_PERF_CF_CONTROLLERS_INFO *PLW2080_CTRL_PERF_PERF_CF_CONTROLLERS_INFO;

/*!
 * LW2080_CTRL_CMD_PERF_PERF_CF_CONTROLLERS_GET_INFO
 *
 * This command returns PERF_CF_CONTROLLERS static object information/POR as
 * specified by the VBIOS in PERF_CF Table.
 *
 * See @ref LW2080_CTRL_PERF_PERF_CF_CONTROLLERS_INFO for documentation on
 * the parameters.
 *
 * Return values are specified per @ref BoardObjGrpGetInfo.
 */
#define LW2080_CTRL_CMD_PERF_PERF_CF_CONTROLLERS_GET_INFO                     (0x208020c8) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | LW2080_CTRL_PERF_PERF_CF_CONTROLLERS_INFO_MESSAGE_ID" */

/*!
 * Index and size of PERF_CF Controller limits. The array entries are ordered
 * as follow: {DRAM_MIN, GPC_MIN, LWD_MIN, DRAM_MAX, GPC_MAX, LWD_MAX}.
 * Note that each clock domain always have a MIN/MAX pair of limits.
 * As an example, the index of GPC_MAX is IDX_MAX_START + CLKDOM_GPC.
 */

#define LW2080_CTRL_PERF_PERF_CF_CONTROLLERS_STATUS_LIMIT_CLKDOM_DRAM         0
#define LW2080_CTRL_PERF_PERF_CF_CONTROLLERS_STATUS_LIMIT_CLKDOM_GPC          1
#define LW2080_CTRL_PERF_PERF_CF_CONTROLLERS_STATUS_LIMIT_CLKDOM_LWD          2
#define LW2080_CTRL_PERF_PERF_CF_CONTROLLERS_STATUS_LIMIT_CLKDOM_XBAR         3
#define LW2080_CTRL_PERF_PERF_CF_CONTROLLERS_STATUS_LIMIT_MAX_CLKDOMS         4

#define LW2080_CTRL_PERF_PERF_CF_CONTROLLERS_STATUS_LIMIT_IDX_MIN_START       0
#define LW2080_CTRL_PERF_PERF_CF_CONTROLLERS_STATUS_LIMIT_IDX_MAX_START       (0x4) /* finn: Evaluated from "(LW2080_CTRL_PERF_PERF_CF_CONTROLLERS_STATUS_LIMIT_IDX_MIN_START + LW2080_CTRL_PERF_PERF_CF_CONTROLLERS_STATUS_LIMIT_MAX_CLKDOMS)" */
#define LW2080_CTRL_PERF_PERF_CF_CONTROLLERS_STATUS_LIMIT_MAX_LIMITS          (0x8) /* finn: Evaluated from "(LW2080_CTRL_PERF_PERF_CF_CONTROLLERS_STATUS_LIMIT_IDX_MAX_START + LW2080_CTRL_PERF_PERF_CF_CONTROLLERS_STATUS_LIMIT_MAX_CLKDOMS)" */

#define LW2080_CTRL_PERF_PERF_CF_CONTROLLERS_STATUS_LIMIT_IDX(_clkDom, _minMax) \
    (LW2080_CTRL_PERF_PERF_CF_CONTROLLERS_STATUS_LIMIT_IDX##_minMax##_START +   \
     LW2080_CTRL_PERF_PERF_CF_CONTROLLERS_STATUS_LIMIT_CLKDOM##_clkDom)

#define LW2080_CTRL_PERF_PERF_CF_CONTROLLERS_STATUS_LIMIT_IDX_IS_MIN(idx) \
    ((idx) < LW2080_CTRL_PERF_PERF_CF_CONTROLLERS_STATUS_LIMIT_IDX_MAX_START)
#define LW2080_CTRL_PERF_PERF_CF_CONTROLLERS_STATUS_LIMIT_IDX_IS_MAX(idx) \
    (((idx) >= LW2080_CTRL_PERF_PERF_CF_CONTROLLERS_STATUS_LIMIT_IDX_MAX_START) && \
     ((idx) <  LW2080_CTRL_PERF_PERF_CF_CONTROLLERS_STATUS_LIMIT_MAX_LIMITS))

/*!
 * Limit frequency has to be set to 0 when there is no request.
 */
#define LW2080_CTRL_PERF_PERF_CF_CONTROLLERS_STATUS_LIMIT_FREQ_KHZ_NO_REQUEST 0

/*!
 * Structure holding limit data.
 */
typedef struct LW2080_CTRL_PERF_PERF_CF_CONTROLLERS_STATUS_LIMIT {
    /*!
     * Current clock frequency request (in kHz).
     */
    LwU32                          freqkHz;
    /*!
     * Limit ID @ref LW2080_CTRL_PERF_PERF_LIMIT_ID_<xyz>.
     */
    LW2080_CTRL_PERF_PERF_LIMIT_ID id;
    /*!
     * Clock domain index of this limit.
     */
    LwU8                           clkDomIdx;
    /*!
     * Is the limit supported? (e.g. Fixed DRAMCLK is not supported.)
     */
    LwBool                         bSupported;
} LW2080_CTRL_PERF_PERF_CF_CONTROLLERS_STATUS_LIMIT;
typedef struct LW2080_CTRL_PERF_PERF_CF_CONTROLLERS_STATUS_LIMIT *PLW2080_CTRL_PERF_PERF_CF_CONTROLLERS_STATUS_LIMIT;

/*!
 * Structure representing the dynamic state of PERF_CF_CONTROLLER_UTIL.
 */
typedef struct LW2080_CTRL_PERF_PERF_CF_CONTROLLER_STATUS_UTIL {
    /*!
     * Latest utilization reading.
     */
    LwUFXP20_12 pct;
    /*!
     * Latest current frequency in kHz.
     */
    LwU32       lwrrentkHz;
    /*!
     * Latest ideal frequency in kHz.
     */
    LwU32       targetkHz;
    /*!
     * Average target frequency in kHz, with hysteresis consideration.
     */
    LwU32       avgTargetkHz;
    /*!
     * Current number of samples above/below target.
     */
    LwU8        hysteresisCountLwrr;
    /*!
     * Limit index to set.
     */
    LwU8        limitIdx;
    /*!
     * Is last sample above or below target? LW_TRUE = above, LW_FALSE = below.
     */
    LwBool      bIncLast;
} LW2080_CTRL_PERF_PERF_CF_CONTROLLER_STATUS_UTIL;
typedef struct LW2080_CTRL_PERF_PERF_CF_CONTROLLER_STATUS_UTIL *PLW2080_CTRL_PERF_PERF_CF_CONTROLLER_STATUS_UTIL;

/*!
 * Structure representing the dynamic state of PERF_CF_CONTROLLER_UTIL_2X.
 */
typedef struct LW2080_CTRL_PERF_PERF_CF_CONTROLLER_STATUS_UTIL_2X {
    /*!
     * Latest utilization reading.
     */
    LW2080_CTRL_PERF_PERF_CF_CONTROLLER_STATUS_UTIL super;
    /*!
     * Reserved.
     */
    LwU8                                            rsvd;
} LW2080_CTRL_PERF_PERF_CF_CONTROLLER_STATUS_UTIL_2X;
typedef struct LW2080_CTRL_PERF_PERF_CF_CONTROLLER_STATUS_UTIL_2X *PLW2080_CTRL_PERF_PERF_CF_CONTROLLER_STATUS_UTIL_2X;

/*!
 * Structure storing statuses of all PERF_CF_PM_SENSORS.
 */
typedef struct LW2080_CTRL_PERF_PERF_CF_CONTROLLER_STATUS_MEM_TUNE_PM_SIGNAL {
    /*!
     * Index of BAPM sensor.
     */
    LwBoardObjIdx index;

   /*!
    * PERF_CF_PM_SENSOR signal's count difference since the last call.
    *
    * On output, this is the difference between the current cntLast and
    * the last cntLast.  This provides an count of the given signal
    * aligned to the timescale of the caller.
    */
    LwU64_ALIGN32 cntDiff;
} LW2080_CTRL_PERF_PERF_CF_CONTROLLER_STATUS_MEM_TUNE_PM_SIGNAL;

/*!
 * Structure holding the BA PM signals for given target.
 */
typedef struct LW2080_CTRL_PERF_PERF_CF_CONTROLLER_STATUS_MEM_TUNE_TARGET {
    /*!
     * Observed value to compare against the target value to which
     * controller is driving.
     */
    LwUFXP20_12                                                   observed;
    /*!
     * Store aclwmulated sum of all signals counted value.
     */
    LW_DECLARE_ALIGNED(LwU64 cntDiffTotal, 8);
    /*!
     * Total number of valid signals.
     */
    LwU8                                                          numSignals;
    /*!
     * Array of BA PM signals being tracked by this controller.
     */
    LW2080_CTRL_PERF_PERF_CF_CONTROLLER_STATUS_MEM_TUNE_PM_SIGNAL signal[LW2080_CTRL_PERF_PERF_CF_CONTROLLER_MEM_TUNE_TARGET_PM_SIGNAL_MAX];
} LW2080_CTRL_PERF_PERF_CF_CONTROLLER_STATUS_MEM_TUNE_TARGET;

/*!
 * Structure holding POR information of all targets.
 */
typedef struct LW2080_CTRL_PERF_PERF_CF_CONTROLLER_STATUS_MEM_TUNE_TARGETS {
    /*!
     * Total number of valid targets.
     */
    LwU8 numTargets;
    /*!
     * Array of target's static POR information.
     * @ref LW2080_CTRL_PERF_PERF_CF_CONTROLLER_STATUS_MEM_TUNE_TARGET
     */
    LW_DECLARE_ALIGNED(LW2080_CTRL_PERF_PERF_CF_CONTROLLER_STATUS_MEM_TUNE_TARGET target[LW2080_CTRL_PERF_PERF_CF_CONTROLLER_MEM_TUNE_TARGET_MAX], 8);
} LW2080_CTRL_PERF_PERF_CF_CONTROLLER_STATUS_MEM_TUNE_TARGETS;

/*!
 * Structure representing the dynamic state of PERF_CF_CONTROLLER_MEM_TUNE.
 */
typedef struct LW2080_CTRL_PERF_PERF_CF_CONTROLLER_STATUS_MEM_TUNE {
    /*!
     * Targets static POR information.
     * @ref LW2080_CTRL_PERF_PERF_CF_CONTROLLER_STATUS_MEM_TUNE_TARGETS
     */
    LW_DECLARE_ALIGNED(LW2080_CTRL_PERF_PERF_CF_CONTROLLER_STATUS_MEM_TUNE_TARGETS targets, 8);
    /*!
     * Current number of samples above target.
     */
    LwU8   hysteresisCountIncLwrr;
    /*!
     * Current number of samples below target.
     */
    LwU8   hysteresisCountDecLwrr;
    /*!
     * Latest current memory clock frequency in kHz.
     * Note that XAPI doesn't like LwUFXP52_24.
     */
    LW_DECLARE_ALIGNED(LwU64 mclkkHz52_12, 8);
    /*!
     * PTIMER reading in msec.
     * Note that XAPI doesn't like LwUFXP52_24.
     */
    LW_DECLARE_ALIGNED(LwU64 perfms52_12, 8);
    /*!
     * Store act_rate_sol = <mem freq> * 2 * <# channels> * bytes/channel / bytes
     * Note that XAPI doesn't like LwUFXP52_24.
     */
    LW_DECLARE_ALIGNED(LwU64 activateRateSol52_12, 8);
    /*!
     * Store acc_rate_sol = <mem freq> * 2 * <# channels> * bytes/channel / bytes
     * Note that XAPI doesn't like LwUFXP52_24.
     */
    LW_DECLARE_ALIGNED(LwU64 accessRateSol52_12, 8);
    /*!
     * Boolean tracking whether PCIe override is enabled.
     */
    LwBool bPcieOverrideEn;
    /*!
     * Boolean tracking whether DISP override is enabled.
     */
    LwBool bDispOverrideEn;
    /*!
     * Boolean tracking whether to engage vs disengage tRRD WAR before
     * any overrides. This is basically tracking the controller decision.
     */
    LwBool bTrrdWarEngagePreOverride;
    /*!
     * Boolean tracking whether to engage vs disengage tRRD WAR.
     * This is a final decision post all overrides on top of the
     * controller decision.
     */
    LwBool bTrrdWarEngage;
    /*!
     * Counter tracking the samples where tRRD WAR was engaged.
     */
    LwU32  trrdWarEngageCounter;
} LW2080_CTRL_PERF_PERF_CF_CONTROLLER_STATUS_MEM_TUNE;

/*!
 * Structure representing the dynamic state of PERF_CF_CONTROLLER_MEM_TUNE_2X.
 */
typedef struct LW2080_CTRL_PERF_PERF_CF_CONTROLLER_STATUS_MEM_TUNE_2X {
    /*!
     * TODO-Chandrashis: Introduce required fields.
     */
    LwU8 rsvd;
} LW2080_CTRL_PERF_PERF_CF_CONTROLLER_STATUS_MEM_TUNE_2X;

#define LW2080_CTRL_PERF_PERF_CF_PERF_TARGET_AS_FAST_AS_POSSIBLE LW_U32_MAX

/*!
 * Structure representing the dynamic state of PERF_CF_CONTROLLER_OPTP_2X.
 */
typedef struct LW2080_CTRL_PERF_PERF_CF_CONTROLLER_STATUS_OPTP_2X {
    /*!
     * Latest observed perf target (aka perf ratio).
     */
    LwUFXP8_24 perfTarget;
    /*!
     * Latest observed graphics frequency in kHz.
     */
    LwU32      grkHz;
    /*!
     * Latest observed video frequency in kHz.
     */
    LwU32      vidkHz;
} LW2080_CTRL_PERF_PERF_CF_CONTROLLER_STATUS_OPTP_2X;
typedef struct LW2080_CTRL_PERF_PERF_CF_CONTROLLER_STATUS_OPTP_2X *PLW2080_CTRL_PERF_PERF_CF_CONTROLLER_STATUS_OPTP_2X;

/*!
 * @defgroup LW2080_CTRL_PERF_PERF_CF_CONTROLLER_STATUS_DLPPC_1X_DRAMCLK_METRICS_IDX_ENUM
 *
 * Enumeration of DRAMCLK metrics array indices.
 *
 * @{
 */
typedef LwU8 LW2080_CTRL_PERF_PERF_CF_CONTROLLER_STATUS_DLPPC_1X_DRAMCLK_METRICS_IDX_ENUM;
#define LW2080_CTRL_PERF_PERF_CF_CONTROLLER_STATUS_DLPPC_1X_DRAMCLK_METRICS_IDX_POWER_CEILING (0)
#define LW2080_CTRL_PERF_PERF_CF_CONTROLLER_STATUS_DLPPC_1X_DRAMCLK_METRICS_IDX_PERF_FLOOR    (1)
#define LW2080_CTRL_PERF_PERF_CF_CONTROLLER_STATUS_DLPPC_1X_DRAMCLK_METRICS_NUM               (2)
/*!@}*/

/*!
 * Structure representing VF data used within DLPPC_1X.
 */
typedef struct LW2080_CTRL_PERF_PERF_CF_CONTROLLER_STATUS_DLPPC_1X_DRAMCLK_METRICS {
    /*!
     * Latest power ceiling and perf floor metrics for this DRAMCLK.
     */
    LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X metrics;

    /*!
     * Boolean denoting if @ref metrics was snapped the master clock
     * range for the DRAMCLK
     */
    LwBool                                              bSaturated;
} LW2080_CTRL_PERF_PERF_CF_CONTROLLER_STATUS_DLPPC_1X_DRAMCLK_METRICS;
typedef struct LW2080_CTRL_PERF_PERF_CF_CONTROLLER_STATUS_DLPPC_1X_DRAMCLK_METRICS *PLW2080_CTRL_PERF_PERF_CF_CONTROLLER_STATUS_DLPPC_1X_DRAMCLK_METRICS;

/*!
 * Structure representing the dynamic state of each DLPPC_1X DRAMCLK search.
 */
typedef struct LW2080_CTRL_PERF_PERF_CF_CONTROLLER_STATUS_DLPPC_1X_DRAMCLK {
    /*!
     * Latest power ceiling and perf floor metrics for this DRAMCLK.
     */
    LW2080_CTRL_PERF_PERF_CF_CONTROLLER_STATUS_DLPPC_1X_DRAMCLK_METRICS dramclkMetrics[LW2080_CTRL_PERF_PERF_CF_CONTROLLER_STATUS_DLPPC_1X_DRAMCLK_METRICS_NUM];

    /*!
     * Estimated metrics at this DRAMCLK and exactly at the primary clock
     * minimum, if valid.
     */
    LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X                 primaryClkMinEstimated;

    /*!
     * Index into @ref metrics for the efficiency point that should be compared against
     * other DRAM clocks.
     */
    LwU8                                                                compMetricsIdx;
} LW2080_CTRL_PERF_PERF_CF_CONTROLLER_STATUS_DLPPC_1X_DRAMCLK;
typedef struct LW2080_CTRL_PERF_PERF_CF_CONTROLLER_STATUS_DLPPC_1X_DRAMCLK *PLW2080_CTRL_PERF_PERF_CF_CONTROLLER_STATUS_DLPPC_1X_DRAMCLK;

#define LW2080_CTRL_PERF_PERF_CF_CONTROLLER_STATUS_DLPPC_1X_FREQ_TUPLE_INIT(pFreqTuple)            \
LWMISC_MEMSET((pFreqTuple), LW2080_CTRL_PERF_PERF_CF_CONTROLLERS_STATUS_LIMIT_FREQ_KHZ_NO_REQUEST, \
    LW_SIZEOF32(LW2080_CTRL_PERF_PERF_CF_CONTROLLER_STATUS_DLPPC_1X_FREQ_TUPLE));

/*!
 * Structure representing a tuple of the rail frequencies used in DLPPC
 */
typedef struct LW2080_CTRL_PERF_PERF_CF_CONTROLLER_STATUS_DLPPC_1X_FREQ_TUPLE {
    /*!
     * Frequency values indexed by LW2080_CTRL_PERF_PERF_CF_CONTROLLERS_STATUS_LIMIT_CLKDOM_.
     */
    LwU32 freqkHz[LW2080_CTRL_PERF_PERF_CF_CONTROLLERS_STATUS_LIMIT_MAX_CLKDOMS];
} LW2080_CTRL_PERF_PERF_CF_CONTROLLER_STATUS_DLPPC_1X_FREQ_TUPLE;
typedef struct LW2080_CTRL_PERF_PERF_CF_CONTROLLER_STATUS_DLPPC_1X_FREQ_TUPLE *PLW2080_CTRL_PERF_PERF_CF_CONTROLLER_STATUS_DLPPC_1X_FREQ_TUPLE;

/*!
 * The maximum number of @ref PWR_POLICY_RELATIONSHIP objects supported in an
 * @ref LW2080_CTRL_PMGR_PWR_POLICY_RELATIONSHIP_SET by
 * @ref LW2080_CTRL_PERF_PERF_CF_CONTROLLER_DLPPC_1X_PWR_POLICY_RELATIONSHIP_SET_LIMITS
 */
#define LW2080_CTRL_PERF_PERF_CF_CONTROLLER_DLPPC_1X_PWR_POLICY_RELATIONSHIP_SET_LIMITS_MAX 4U

/*!
 * The limits imposed by a given
 * @ref LW2080_CTRL_PMGR_PWR_POLICY_RELATIONSHIP_SET
 */
typedef struct LW2080_CTRL_PERF_PERF_CF_CONTROLLER_DLPPC_1X_PWR_POLICY_RELATIONSHIP_SET_LIMITS {
    /*!
     * Collection of power limits for a given relationship set.
     *
     * @note    Each entry corresponds to a given
     *          @ref LW2080_CTRL_PMGR_PWR_POLICY_RELATIONSHIP_INFO
     *          in an @ref LW2080_CTRL_PMGR_PWR_POLICY_RELATIONSHIP_SET. The
     *          correspondence in indices is defined by
     *          @ref LW2080_CTRL_PERF_PERF_CF_CONTROLLER_DLPPC_1X_PWR_POLICY_RELATIONSHIP_SET_LIMITS_IDX_FROM_POLICY_REL_IDX
     */
    LwU32 policyLimits[LW2080_CTRL_PERF_PERF_CF_CONTROLLER_DLPPC_1X_PWR_POLICY_RELATIONSHIP_SET_LIMITS_MAX];
} LW2080_CTRL_PERF_PERF_CF_CONTROLLER_DLPPC_1X_PWR_POLICY_RELATIONSHIP_SET_LIMITS;

/*!
 * @brief   Retrieves the index in
 *          @ref LW2080_CTRL_PERF_PERF_CF_CONTROLLER_DLPPC_1X_PWR_POLICY_RELATIONSHIP_SET_LIMITS::policyLimits
 *          from the corresponding @ref LW2080_CTRL_PMGR_PWR_POLICY_RELATIONSHIP_SET
 *          and policy relationship index.
 *
 * @details The indices correspond as follows:
 *              @ref LW2080_CTRL_PERF_PERF_CF_CONTROLLER_DLPPC_1X_PWR_POLICY_RELATIONSHIP_SET_LIMITS_MAXp::policyLimits[i]
 *                  <->
 *              polcyRelIdx - @ref LW2080_CTRL_PMGR_PWR_POLICY_RELATIONSHIP_SET::policyRelStart
 *
 * @param[in]   pRelSet         @ref LW2080_CTRL_PMGR_PWR_POLICY_RELATIONSHIP_SET
 *                              structure for which to retrieve limit index
 * @param[in]   policyRelIdx    Policy relationship @ref LwBoardObjIdx for which
 *                              to retrieve limit index.
 *
 * @return  Index in
 *          @ref LW2080_CTRL_PERF_PERF_CF_CONTROLLER_DLPPC_1X_PWR_POLICY_RELATIONSHIP_SET_LIMITS::policyLimits
 *          for the given @ref LW2080_CTRL_PMGR_PWR_POLICY_RELATIONSHIP_SET and
 *          policy relationship index.
 */
#define LW2080_CTRL_PERF_PERF_CF_CONTROLLER_DLPPC_1X_PWR_POLICY_RELATIONSHIP_SET_LIMITS_IDX_FROM_POLICY_REL_IDX(pRelSet, policyRelIdx) \
    ((policyRelIdx) - (pRelSet)->policyRelStart)

/*!
 * Structure to hold the power limits imposed on a given
 * @ref LW2080_CTRL_PERF_PERF_CF_CONTROLLER_DLPPC_1X_RAIL
 */
typedef struct LW2080_CTRL_PERF_PERF_CF_CONTROLLER_DLPPC_1X_RAIL_LIMITS {
    /*!
     * Input power limits.
     */
    LW2080_CTRL_PERF_PERF_CF_CONTROLLER_DLPPC_1X_PWR_POLICY_RELATIONSHIP_SET_LIMITS pwrInLimits;

    /*!
     * Output power limits.
     */
    LW2080_CTRL_PERF_PERF_CF_CONTROLLER_DLPPC_1X_PWR_POLICY_RELATIONSHIP_SET_LIMITS pwrOutLimits;
} LW2080_CTRL_PERF_PERF_CF_CONTROLLER_DLPPC_1X_RAIL_LIMITS;

/*!
 * Dynamic status of a given
 * @ref LW2080_CTRL_PERF_PERF_CF_CONTROLLER_DLPPC_1X_RAIL
 */
typedef struct LW2080_CTRL_PERF_PERF_CF_CONTROLLER_DLPPC_1X_RAIL_STATUS {
    /*!
     * Power limits imposed on the rail.
     */
    LW2080_CTRL_PERF_PERF_CF_CONTROLLER_DLPPC_1X_RAIL_LIMITS limits;
} LW2080_CTRL_PERF_PERF_CF_CONTROLLER_DLPPC_1X_RAIL_STATUS;

/*!
 * Dynamic status of an
 * @ref LW2080_CTRL_PERF_PERF_CF_CONTROLLER_DLPPC_1X_CORE_RAIL
 */
typedef struct LW2080_CTRL_PERF_PERF_CF_CONTROLLER_DLPPC_1X_CORE_RAIL_STATUS {
    /*!
     * Status for each individual rail within the core rail.
     */
    LW2080_CTRL_PERF_PERF_CF_CONTROLLER_DLPPC_1X_RAIL_STATUS                        rails[LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_DLPPM_1X_MAX_CORE_RAILS];

    /*!
     * Input power limits on the core as a whole.
     */
    LW2080_CTRL_PERF_PERF_CF_CONTROLLER_DLPPC_1X_PWR_POLICY_RELATIONSHIP_SET_LIMITS pwrInLimits;
} LW2080_CTRL_PERF_PERF_CF_CONTROLLER_DLPPC_1X_CORE_RAIL_STATUS;

/*!
 * Profiling for a single iteration of a perf floor or power ceiling search
 * within a DRAMCLK in DLPPC_1X.
 */
typedef struct LW2080_CTRL_PERF_PERF_CF_CONTROLLER_DLPPC_1X_PROFILING_DRAMCLK_SEARCH_TYPE_ITERATION {
    /*!
     * Exelwtion time of the iteration.
     */
    LwU64_ALIGN32                                                       profiledTimens;

    /*!
     * Profile structure for the scale operation backing the iteration.
     */
    LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X_SCALE_PROFILING scaleProfiling;
} LW2080_CTRL_PERF_PERF_CF_CONTROLLER_DLPPC_1X_PROFILING_DRAMCLK_SEARCH_TYPE_ITERATION;

/*!
 * Maximum number of perf floor and power ceiling iterations to profile.
 */
#define LW2080_CTRL_PERF_PERF_CF_CONTROLLER_DLPPC_1X_PROFILING_DRAMCLK_SEARCH_TYPE_ITERATIONS_MAX 0x10

/*!
 * Profiling for the entirety of a given "type" of search within a DRAMCLK
 * (either the perf floor or the power ceiling).
 */
typedef struct LW2080_CTRL_PERF_PERF_CF_CONTROLLER_DLPPC_1X_PROFILING_DRAMCLK_SEARCH_TYPE {
    /*!
     * Total time for this search.
     */
    LwU64_ALIGN32                                                                        profiledTimens;

    /*!
     * Array of profiling structures for the iterations of the search.
     */
    LW2080_CTRL_PERF_PERF_CF_CONTROLLER_DLPPC_1X_PROFILING_DRAMCLK_SEARCH_TYPE_ITERATION iterations[LW2080_CTRL_PERF_PERF_CF_CONTROLLER_DLPPC_1X_PROFILING_DRAMCLK_SEARCH_TYPE_ITERATIONS_MAX];

    /*!
     * Number of iterations of the search
     *
     * @warning This can be greater than the size of
     *          @ref LW2080_CTRL_PERF_PERF_CF_CONTROLLER_DLPPC_1X_PROFILING_DRAMCLK_SEARCH_TYPE::iterations[],
     *          if more iterations were needed than structures are available;
     *          the value keeps incrementing to track the count of any iterations
     *          that were not profiled.
     */
    LwU8                                                                                 numIterations;
} LW2080_CTRL_PERF_PERF_CF_CONTROLLER_DLPPC_1X_PROFILING_DRAMCLK_SEARCH_TYPE;

/*!
 * Profiling for the search of a single DRAMCLK in DLPPC_1X
 */
typedef struct LW2080_CTRL_PERF_PERF_CF_CONTROLLER_DLPPC_1X_PROFILING_DRAMCLK_SEARCH {
    /*!
     * Total time for searching this DRAMCLK.
     */
    LwU64_ALIGN32                                                              profiledTimens;

    /*!
     * Profiling for the perf floor search.
     */
    LW2080_CTRL_PERF_PERF_CF_CONTROLLER_DLPPC_1X_PROFILING_DRAMCLK_SEARCH_TYPE perfFloorProfiling;

    /*!
     * Profiling for the power ceiling search.
     */
    LW2080_CTRL_PERF_PERF_CF_CONTROLLER_DLPPC_1X_PROFILING_DRAMCLK_SEARCH_TYPE powerCeilingProfiling;

    /*!
     * Profiling for the callwlation of
     * @ref LW2080_CTRL_PERF_PERF_CF_CONTROLLER_STATUS_DLPPC_1X_DRAMCLK::primaryClkMinEstimated
     * if necessary.
     */
    LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X_SCALE_PROFILING        primaryClkMinEstimatedProfiling;
} LW2080_CTRL_PERF_PERF_CF_CONTROLLER_DLPPC_1X_PROFILING_DRAMCLK_SEARCH;

#define LW2080_CTRL_PERF_PERF_CF_CONTROLLER_STATUS_DLPPC_1X_DRAMCLK_NUM LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X_OBSERVED_INITIAL_DRAMCLK_ESTIMATES_MAX

/*!
 * @defgroup LW2080_CTRL_PERF_PERF_CF_CONTROLLER_DLPPC_1X_PROFILING_REGION_ENUM
 *
 * Enumeration of PERF_CF_CONTROLLER_DLPPC_1X profiling regions for exelwtion.
 * @{
 */
typedef LwU8 LW2080_CTRL_PERF_PERF_CF_CONTROLLER_DLPPC_1X_PROFILING_REGION_ENUM;
#define LW2080_CTRL_PERF_PERF_CF_CONTROLLER_DLPPC_1X_PROFILING_REGION_TOTAL            0x00U
#define LW2080_CTRL_PERF_PERF_CF_CONTROLLER_DLPPC_1X_PROFILING_REGION_OBSERVE_METRICS  0x01U
#define LW2080_CTRL_PERF_PERF_CF_CONTROLLER_DLPPC_1X_PROFILING_REGION_PWR_LIMITS_CACHE 0x02U
#define LW2080_CTRL_PERF_PERF_CF_CONTROLLER_DLPPC_1X_PROFILING_REGION_VPSTATE_TO_FREQ  0x03U
#define LW2080_CTRL_PERF_PERF_CF_CONTROLLER_DLPPC_1X_PROFILING_REGION_INFLECTION_ZONE  0x04U
#define LW2080_CTRL_PERF_PERF_CF_CONTROLLER_DLPPC_1X_PROFILING_REGION_DRAMCLK_FILTER   0x05U
#define LW2080_CTRL_PERF_PERF_CF_CONTROLLER_DLPPC_1X_PROFILING_REGION_SEARCH           0x06U
#define LW2080_CTRL_PERF_PERF_CF_CONTROLLER_DLPPC_1X_PROFILING_REGION_HYSTERESIS       0x07U
#define LW2080_CTRL_PERF_PERF_CF_CONTROLLER_DLPPC_1X_PROFILING_REGION__COUNT           0x08U
/*!@}*/

/*!
 * Profiling structure for PERF_CF_CONTROLLER_DLPPC_1X exelwtion.
 */
typedef struct LW2080_CTRL_PERF_PERF_CF_CONTROLLER_DLPPC_1X_PROFILING {
    /*!
     * Array of profiled "regions" in the code.
     */
    LwU64_ALIGN32                                                         profiledTimesns[LW2080_CTRL_PERF_PERF_CF_CONTROLLER_DLPPC_1X_PROFILING_REGION__COUNT];

    /*!
     * Array of structures containing profiling data for each DRAMCLK's search.
     */
    LW2080_CTRL_PERF_PERF_CF_CONTROLLER_DLPPC_1X_PROFILING_DRAMCLK_SEARCH dramclkSearchProfiling[LW2080_CTRL_PERF_PERF_CF_CONTROLLER_STATUS_DLPPC_1X_DRAMCLK_NUM];
} LW2080_CTRL_PERF_PERF_CF_CONTROLLER_DLPPC_1X_PROFILING;

/*!
 * Structure representing the dynamic state of PERF_CF_CONTROLLER_DLPPC_1X.
 */
typedef struct LW2080_CTRL_PERF_PERF_CF_CONTROLLER_STATUS_DLPPC_1X {
    /*!
     * Latest observed metrics from the sampled data.
     */
    LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_DLPPM_1X_OBSERVED                    observedMetrics;

    /*!
     * DRAMCLK dynamic state.
     */
    LW2080_CTRL_PERF_PERF_CF_CONTROLLER_STATUS_DLPPC_1X_DRAMCLK                     dramclk[LW2080_CTRL_PERF_PERF_CF_CONTROLLER_STATUS_DLPPC_1X_DRAMCLK_NUM];

    /*!
     * Status of the core rail.
     */
    LW2080_CTRL_PERF_PERF_CF_CONTROLLER_DLPPC_1X_CORE_RAIL_STATUS                   coreRailStatus;

    /*!
     * Status of the FB rail.
     */
    LW2080_CTRL_PERF_PERF_CF_CONTROLLER_DLPPC_1X_RAIL_STATUS                        fbRailStatus;

    /*!
     * Latest TGP power limits seen.
     */
    LW2080_CTRL_PERF_PERF_CF_CONTROLLER_DLPPC_1X_PWR_POLICY_RELATIONSHIP_SET_LIMITS tgpLimits;

    /*!
     * Latest perf target that DLPPC_1X is trying to achieve.
     */
    LwUFXP20_12                                                                     perfTarget;

    /*!
     * Number of elements in @ref dramclk that are valid.
     */
    LwU8                                                                            numDramclk;

    /*!
     * Number of conselwtive cycles we have seen where a lower
     * dramclk frequency would provide better perf
     */
    LwU8                                                                            dramHysteresisLwrrCountDec;

    /*!
     * Number of conselwtive cycles we have seen where a higher
     * dramclk frequency would provide better perf
     */
    LwU8                                                                            dramHysteresisLwrrCountInc;

    /*!
     * Tuple of frequencies that hold the previous loop's operating frequencies
     */
    LW2080_CTRL_PERF_PERF_CF_CONTROLLER_STATUS_DLPPC_1X_FREQ_TUPLE                  tupleLastOperatingFrequencies;

    /*!
     * Structure holding the frequency recommendations that the controller chose in its
     * previous exelwtion for the arbirter.
     */
    LW2080_CTRL_PERF_PERF_CF_CONTROLLER_STATUS_DLPPC_1X_FREQ_TUPLE                  tupleLastChosenLimits;

    /*!
     * Structure holding the limit recommendations that the controller estimated
     * to be the best operating frequencies based on it's current exelwtion.
     */
    LW2080_CTRL_PERF_PERF_CF_CONTROLLER_STATUS_DLPPC_1X_FREQ_TUPLE                  tupleBestEstimatedLimits;

    /*!
     * Whether any of the @ref LW2080_CTRL_PMGR_PWR_POLICY_INFO objects with
     * which DLPPC_1X is concerned have an inflection limit below DLPPC_1X's
     * @ref LW2080_CTRL_PERF_VPSTATES_IDX_DLPPC_1X_SEARCH_MINIMUM corresponding
     * @ref LW2080_CTRL_PERF_PSTATE_INFO.
     *
     * If that is the case, DLPPC_1X defers back to enabling the inflection
     * points.
     */
    LwBool                                                                          bInInflectionZone;

    /*!
     * Profiling data for the last exelwtion of the controller.
     */
    LW2080_CTRL_PERF_PERF_CF_CONTROLLER_DLPPC_1X_PROFILING                          profiling;
} LW2080_CTRL_PERF_PERF_CF_CONTROLLER_STATUS_DLPPC_1X;
typedef struct LW2080_CTRL_PERF_PERF_CF_CONTROLLER_STATUS_DLPPC_1X *PLW2080_CTRL_PERF_PERF_CF_CONTROLLER_STATUS_DLPPC_1X;

/*!
 * PERF_CF_CONTROLLER type-specific data union. Discriminated by
 * PERF_CF_CONTROLLER::super.type.
 */


/*!
 * Structure representing the dynamic state of each PERF_CF_CONTROLLER.
 * Implements the BOARDOBJ model/interface.
 */
typedef struct LW2080_CTRL_PERF_PERF_CF_CONTROLLER_STATUS {
    /*!
     * LW2080_CTRL_BOARDOBJ super class.
     * Must always be first object in structure.
     */
    LW2080_CTRL_BOARDOBJ                                            super;
    /*!
     * XAPI does not support using an element within another structure as a
     * discriminant. Placing this redundant type value here until that design
     * constraint can be fixed.
     */
    LwU8                                                            type;
    /*!
     * Frequency limit requests in kHz. 0 == no request. Execute() stores the values here, to be arbitrated.
     */
    LwU32                                                           limitFreqkHz[LW2080_CTRL_PERF_PERF_CF_CONTROLLERS_STATUS_LIMIT_MAX_LIMITS];
    /*!
     * Counting the number of execute iterations.
     */
    LwU32                                                           iteration;
    /*!
     * Reset controller state on next cycle.
     */
    LwBool                                                          bReset;
    /*!
     * Current active/inactive state.
     */
    LwBool                                                          bActive;

    /*!
     * Inflection points disablement request from this controller.
     */
    LW2080_CTRL_PMGR_PWR_POLICIES_INFLECTION_POINTS_DISABLE_REQUEST inflectionPointsDisableRequest;

    /*!
     * Type-specific data union.
     */
    union {
        LW2080_CTRL_PERF_PERF_CF_CONTROLLER_STATUS_UTIL        util;
        LW2080_CTRL_PERF_PERF_CF_CONTROLLER_STATUS_UTIL_2X     util2x;
        LW2080_CTRL_PERF_PERF_CF_CONTROLLER_STATUS_OPTP_2X     optp2x;
        LW2080_CTRL_PERF_PERF_CF_CONTROLLER_STATUS_DLPPC_1X    dlppc1x;
        LW_DECLARE_ALIGNED(LW2080_CTRL_PERF_PERF_CF_CONTROLLER_STATUS_MEM_TUNE memTune, 8);
        LW2080_CTRL_PERF_PERF_CF_CONTROLLER_STATUS_MEM_TUNE_2X memTune2x;
    } data;
} LW2080_CTRL_PERF_PERF_CF_CONTROLLER_STATUS;
typedef struct LW2080_CTRL_PERF_PERF_CF_CONTROLLER_STATUS *PLW2080_CTRL_PERF_PERF_CF_CONTROLLER_STATUS;

/*!
 * OPTP is not active.
 */
#define LW2080_CTRL_PERF_PERF_CF_CONTROLLERS_STATUS_OPTP_PERF_RATIO_INACTIVE LW_U32_MAX


/*!
 * @defgroup LW2080_CTRL_PERF_PERF_CF_CONTROLLERS_PROFILING_MULT_DATA_SAMPLE_REGION_ENUM
 *
 * Enumeration of PERF_CF_CONTROLLERS mult data sampling regions.
 * @{
 */
typedef LwU8 LW2080_CTRL_PERF_PERF_CF_CONTROLLERS_PROFILING_MULT_DATA_SAMPLE_REGION_ENUM;
#define LW2080_CTRL_PERF_PERF_CF_CONTROLLERS_PROFILING_MULT_DATA_SAMPLE_REGION_TOTAL                         0U
#define LW2080_CTRL_PERF_PERF_CF_CONTROLLERS_PROFILING_MULT_DATA_SAMPLE_REGION_PERF_CF_TOPOLOGYS_STATUS_GET  1U
#define LW2080_CTRL_PERF_PERF_CF_CONTROLLERS_PROFILING_MULT_DATA_SAMPLE_REGION_PWR_POLICIES_STATUS_GET       2U
#define LW2080_CTRL_PERF_PERF_CF_CONTROLLERS_PROFILING_MULT_DATA_SAMPLE_REGION_PWR_CHANNELS_STATUS_GET       3U
#define LW2080_CTRL_PERF_PERF_CF_CONTROLLERS_PROFILING_MULT_DATA_SAMPLE_REGION_PERF_CF_PM_SENSORS_STATUS_GET 4U
#define LW2080_CTRL_PERF_PERF_CF_CONTROLLERS_PROFILING_MULT_DATA_SAMPLE_REGION__COUNT                        5U
/*!@}*/

/*!
 * Profiling data for the sampling in PERF_CF_CONTROLLERS
 */
typedef struct LW2080_CTRL_PERF_PERF_CF_CONTROLLERS_PROFILING_MULT_DATA_SAMPLE {
    /*!
     * Mask of controllesr for this set of data.
     */
    LW2080_CTRL_BOARDOBJGRP_MASK_E32 maskControllers;

    /*!
     * Array of profiled "regions"
     */
    LwU64_ALIGN32                    profiledTimesns[LW2080_CTRL_PERF_PERF_CF_CONTROLLERS_PROFILING_MULT_DATA_SAMPLE_REGION__COUNT];
} LW2080_CTRL_PERF_PERF_CF_CONTROLLERS_PROFILING_MULT_DATA_SAMPLE;

/*!
 * @defgroup LW2080_CTRL_PERF_PERF_CF_CONTROLLERS_PROFILING_REGION_ENUM
 *
 * Enumeration of PERF_CF_CONTROLLERS profiling regions.
 * @{
 */
typedef LwU8 LW2080_CTRL_PERF_PERF_CF_CONTROLLERS_PROFILING_REGION_ENUM;
#define LW2080_CTRL_PERF_PERF_CF_CONTROLLERS_PROFILING_REGION_TOTAL               0U
#define LW2080_CTRL_PERF_PERF_CF_CONTROLLERS_PROFILING_REGION_FILTER              1U
#define LW2080_CTRL_PERF_PERF_CF_CONTROLLERS_PROFILING_REGION_CONTROLLERS_EXELWTE 2U
#define LW2080_CTRL_PERF_PERF_CF_CONTROLLERS_PROFILING_REGION_ARBITRATE           3U
#define LW2080_CTRL_PERF_PERF_CF_CONTROLLERS_PROFILING_REGION_POST_PROCESS        4U
#define LW2080_CTRL_PERF_PERF_CF_CONTROLLERS_PROFILING_REGION_SET_LIMITS          5U
#define LW2080_CTRL_PERF_PERF_CF_CONTROLLERS_PROFILING_REGION__COUNT              6U
/*!@}*/

/*!
 * Maximum number of mult data samples to profile in a single
 * PERF_CF_CONTROLLERS exelwtion.
 */
#define LW2080_CTRL_PERF_PERF_CF_CONTROLLERS_PROFILING_MULT_DATA_SAMPLES_MAX      8U

/*!
 * Structure containing profiling dat for PERF_CF_CONTROLLERS exelwtion.
 */
typedef struct LW2080_CTRL_PERF_PERF_CF_CONTROLLERS_PROFILING {
    /*!
     * Elapsed times for various portions of PERF_CF_CONTROLLERS exelwtion
     */
    LwU64_ALIGN32                                                   profiledTimesns[LW2080_CTRL_PERF_PERF_CF_CONTROLLERS_PROFILING_REGION__COUNT];

    /*!
     * Array of profiled sampling steps for the exelwtion.
     */
    LW2080_CTRL_PERF_PERF_CF_CONTROLLERS_PROFILING_MULT_DATA_SAMPLE multDataSampleProfiling[LW2080_CTRL_PERF_PERF_CF_CONTROLLERS_PROFILING_MULT_DATA_SAMPLES_MAX];

    /*!
     * Number of mult data samples done.
     *
     * @warning This can be greater than the size of
     *          @ref LW2080_CTRL_PERF_PERF_CF_CONTROLLERS_PROFILING::multDataSampleProfiling[],
     *          if more mult datas were exelwted than structures are available;
     *          the value keeps incrementing to track the count of any mult
     *          datas that were not profiled.
     */
    LwU8                                                            numMultDataSamplesProfiled;
} LW2080_CTRL_PERF_PERF_CF_CONTROLLERS_PROFILING;

/*!
 * Structure representing the dynamic state of PERF_CF_CONTROLLERS.
 * Implements the BOARDOBJGRP model/interface.
 */
#define LW2080_CTRL_PERF_PERF_CF_CONTROLLERS_STATUS_MESSAGE_ID (0xC9U)

typedef struct LW2080_CTRL_PERF_PERF_CF_CONTROLLERS_STATUS {
    /*!
     * LW2080_CTRL_BOARDOBJGRP_E32 super class.
     * Must always be first object in structure.
     */
    LW2080_CTRL_BOARDOBJGRP_E32                       super;
    /*!
     * Current arbitrated limits across all Controllers.
     */
    LW2080_CTRL_PERF_PERF_CF_CONTROLLERS_STATUS_LIMIT limits[LW2080_CTRL_PERF_PERF_CF_CONTROLLERS_STATUS_LIMIT_MAX_LIMITS];
    /*!
     * Mask of the lwrrently active Controllers.
     */
    LW2080_CTRL_BOARDOBJGRP_MASK_E32                  maskActive;
    /*!
     * Latest performance ratio request from OPTP. Can be
     * LW2080_CTRL_PERF_PERF_CF_CONTROLLERS_STATUS_OPTP_PERF_RATIO_INACTIVE.
     */
    LwUFXP8_24                                        optpPerfRatio;
    /*!
     * Counting perfLimitsClientArbitrateAndApply() failures.
     */
    LwU32                                             limitsArbErrCount;
    /*!
     * Tracking the last perfLimitsClientArbitrateAndApply() FLCN_STATUS return code != OK.
     */
    LwU8                                              limitsArbErrLast;

    /*!
     * Profiling data for the last exelwtion of the controllers.
     */
    LW2080_CTRL_PERF_PERF_CF_CONTROLLERS_PROFILING    profiling;

    /*!
     * Array of PERF_CF_CONTROLLER structures. Has valid indexes corresponding to
     * the bits set in @ref super.objMask.
     */
    LW_DECLARE_ALIGNED(LW2080_CTRL_PERF_PERF_CF_CONTROLLER_STATUS controllers[LW2080_CTRL_PERF_PERF_CF_CONTROLLERS_MAX_OBJECTS], 8);
} LW2080_CTRL_PERF_PERF_CF_CONTROLLERS_STATUS;
typedef struct LW2080_CTRL_PERF_PERF_CF_CONTROLLERS_STATUS *PLW2080_CTRL_PERF_PERF_CF_CONTROLLERS_STATUS;

/*!
 * LW2080_CTRL_CMD_PERF_PERF_CF_CONTROLLERS_GET_STATUS
 *
 * This command returns PERF_CF_CONTROLLERS dynamic state.
 *
 * See @ref LW2080_CTRL_PERF_PERF_CF_CONTROLLERS_STATUS for documentation on
 * the parameters.
 *
 * Return values are specified per @ref BoardObjGrpGetInfo.
 */
#define LW2080_CTRL_CMD_PERF_PERF_CF_CONTROLLERS_GET_STATUS (0x208020c9) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | LW2080_CTRL_PERF_PERF_CF_CONTROLLERS_STATUS_MESSAGE_ID" */


/*!
 * Structure representing the control parameters of PERF_CF_CONTROLLER_UTIL.
 */
typedef struct LW2080_CTRL_PERF_PERF_CF_CONTROLLER_CONTROL_UTIL {
    /*!
     * Target utilization value to which controller is driving.
     */
    LwUFXP20_12 target;
    /*!
     * Threshold above which controller will jump to Base Clock.
     */
    LwUFXP20_12 jumpThreshold;
    /*!
     * Proportional error gain for increasing clocks.
     */
    LwUFXP20_12 gainInc;
    /*!
     * Proportional error gain for decreasing clocks.
     */
    LwUFXP20_12 gainDec;
    /*!
     * Number of samples above target before increasing clocks.
     */
    LwU8        hysteresisCountInc;
    /*!
     * Number of samples below target before decreasing clocks.
     */
    LwU8        hysteresisCountDec;
    /*!
     * Do not program clocks if TRUE
     */
    LwBool      bDoNotProgram;
} LW2080_CTRL_PERF_PERF_CF_CONTROLLER_CONTROL_UTIL;
typedef struct LW2080_CTRL_PERF_PERF_CF_CONTROLLER_CONTROL_UTIL *PLW2080_CTRL_PERF_PERF_CF_CONTROLLER_CONTROL_UTIL;

/*!
 * Structure representing the control parameters of PERF_CF_CONTROLLER_UTIL_2X.
 */
typedef struct LW2080_CTRL_PERF_PERF_CF_CONTROLLER_CONTROL_UTIL_2X {
    /*!
     * Target utilization value to which controller is driving.
     */
    LW2080_CTRL_PERF_PERF_CF_CONTROLLER_CONTROL_UTIL super;
    /*!
     * Number of IIR samples above target before increasing clocks.
     */
    LwU8                                             IIRCountInc;
    /*!
     * Number of IIR samples below target before decreasing clocks.
     */
    LwU8                                             IIRCountDec;
} LW2080_CTRL_PERF_PERF_CF_CONTROLLER_CONTROL_UTIL_2X;
typedef struct LW2080_CTRL_PERF_PERF_CF_CONTROLLER_CONTROL_UTIL_2X *PLW2080_CTRL_PERF_PERF_CF_CONTROLLER_CONTROL_UTIL_2X;

/*!
 * Structure representing the high and low target values.
 */
typedef struct LW2080_CTRL_PERF_PERF_CF_CONTROLLER_CONTROL_MEM_TUNE_TARGET {
    /*!
     * Array of target value above which controller will engage.
     */
    LwUFXP20_12 high;
    /*!
     * Array of target value below which controller will disengage.
     */
    LwUFXP20_12 low;
} LW2080_CTRL_PERF_PERF_CF_CONTROLLER_CONTROL_MEM_TUNE_TARGET;

/*!
 * Structure holding control params of all targets.
 */
typedef struct LW2080_CTRL_PERF_PERF_CF_CONTROLLER_CONTROL_MEM_TUNE_TARGETS {
    /*!
     * Total number of valid targets.
     * @note This param is not controllable.
     */
    LwU8                                                        numTargets;
    /*!
     * Array of target's control params.
     * @ref LW2080_CTRL_PERF_PERF_CF_CONTROLLER_CONTROL_MEM_TUNE_TARGET
     */
    LW2080_CTRL_PERF_PERF_CF_CONTROLLER_CONTROL_MEM_TUNE_TARGET target[LW2080_CTRL_PERF_PERF_CF_CONTROLLER_MEM_TUNE_TARGET_MAX];
} LW2080_CTRL_PERF_PERF_CF_CONTROLLER_CONTROL_MEM_TUNE_TARGETS;

/*!
 * Structure representing the control parameters of PERF_CF_CONTROLLER_MEM_TUNE.
 */
typedef struct LW2080_CTRL_PERF_PERF_CF_CONTROLLER_CONTROL_MEM_TUNE {
    /*!
     * Targets control params.
     * @ref LW2080_CTRL_PERF_PERF_CF_CONTROLLER_CONTROL_MEM_TUNE_TARGETS
     */
    LW2080_CTRL_PERF_PERF_CF_CONTROLLER_CONTROL_MEM_TUNE_TARGETS targets;
    /*!
     * Number of samples above target before increasing clocks.
     */
    LwU8                                                         hysteresisCountInc;
    /*!
     * Number of samples below target before decreasing clocks.
     */
    LwU8                                                         hysteresisCountDec;
} LW2080_CTRL_PERF_PERF_CF_CONTROLLER_CONTROL_MEM_TUNE;

/*!
 * Structure representing the control parameters of PERF_CF_CONTROLLER_MEM_TUNE_2X.
 */
typedef struct LW2080_CTRL_PERF_PERF_CF_CONTROLLER_CONTROL_MEM_TUNE_2X {
    /*!
     * TODO-Chandrashis: Introduce required fields.
     */
    LwU8 rsvd;
} LW2080_CTRL_PERF_PERF_CF_CONTROLLER_CONTROL_MEM_TUNE_2X;

/*!
 * Structure representing the control parameters of PERF_CF_CONTROLLER_OPTP_2X.
 */
typedef struct LW2080_CTRL_PERF_PERF_CF_CONTROLLER_CONTROL_OPTP_2X {
    /*!
     * Trigger limit change when perfTarget goes above this threshold.
     */
    LwUFXP8_24 highThreshold;
    /*!
     * Trigger limit change when perfTarget goes below this threshold.
     */
    LwUFXP8_24 lowThreshold;
} LW2080_CTRL_PERF_PERF_CF_CONTROLLER_CONTROL_OPTP_2X;
typedef struct LW2080_CTRL_PERF_PERF_CF_CONTROLLER_CONTROL_OPTP_2X *PLW2080_CTRL_PERF_PERF_CF_CONTROLLER_CONTROL_OPTP_2X;

/*!
 * Value that indicates that no fuzzy perf matching is needed
 */
#define LW2080_CTRL_PERF_PERF_CF_CONTROLLER_CONTROL_DLPPC_1X_NO_FUZZY_PERF_MATCH (0)

/*!
 * Structure representing the control parameters of PERF_CF_CONTROLLER_DLPPC_1X.
 */
typedef struct LW2080_CTRL_PERF_PERF_CF_CONTROLLER_CONTROL_DLPPC_1X {
    /*!
     * Number of cycles of seeing better perf at lower dramclk
     * before DLPPC will choose to switch to it
     */
    LwU8        dramHysteresisCountDec;

    /*!
     * Number of cycles of seeing better perf at higher dramclk
     * before DLPPC will choose to switch to it
     */
    LwU8        dramHysteresisCountInc;

    /*!
     * Signed Value between (-1,1) indicating the minimum percent
     * difference in two perfRatio values needed to be seen
     * for them to be consiered not equal to one another.
     * Note: this value is asymetric due to being signed.
     *
     * When positive, the threshold indicates the MIN percent diff GAIN in perf
     * a higher dramclk has to see compared to a lower dramclk in order for the
     * two dramclks to have "different" perf values
     *
     * When negative, the threshold indicates the amount of LOSS in perf
     * a higher dramclk can see compared to a lower dramclk in order for the
     * perf values of the two dramclks to be considered roughly equal.
     */
    LwSFXP20_12 approxPerfThreshold;

    /*!
     * Threshold, in percentage, that moves the "equality point" between TGP
     * values at different DRAMCLKs either up or down from the power value at
     * the lower DRAMCLK.
     *
     * That is, this is a signed value such that:
     *  1.) A positive value moves the "equality point" X% "up" from the lower
     *      DRAMCLK's power, i.e., the equality point becomes:
     *          (1 + approxPowerThresholdPct) * powerLomW
     *      and any power at the higher DRAMCLK less than this value is
     *      considered less than the power at the lower DRAMCLK. This has the
     *      effect of biasing towards the higher DRAMCLK, because the higher
     *      DRAMCLK can still be considered to use "less" power as long as it is
     *      in this range, even if in reality it is using more power than the
     *      lower DRAMCLK.
     *      
     *  2.) A negative value moves the "equality point" X% "down" from the lower
     *      DRAMCLK's power, i.e., the equality point becomes:
     *          (1 + approxPowerThresholdPct) * powerLomW
     *      and any power at the higher DRAMCLK must be less than this value to
     *      be considered less than the power at the lower DRAMCLK. This has the
     *      effect of biasing towards the lower DRAMCLK, because the higher 
     *      DRAMCLK can still be considered to use "more" power as long as it is
     *      in this range, even if in reality it is using less power than the
     *      lower DRAMCLK.
     */
    LwSFXP20_12 approxPowerThresholdPct;
} LW2080_CTRL_PERF_PERF_CF_CONTROLLER_CONTROL_DLPPC_1X;
typedef struct LW2080_CTRL_PERF_PERF_CF_CONTROLLER_CONTROL_DLPPC_1X *PLW2080_CTRL_PERF_PERF_CF_CONTROLLER_CONTROL_DLPPC_1X;

/*!
 * PERF_CF_CONTROLLER type-specific data union. Discriminated by
 * PERF_CF_CONTROLLER::super.type.
 */


/*!
 * Structure representing the control parameters of each PERF_CF_CONTROLLER.
 * Implements the BOARDOBJ model/interface.
 */
typedef struct LW2080_CTRL_PERF_PERF_CF_CONTROLLER_CONTROL {
    /*!
     * LW2080_CTRL_BOARDOBJ super class.
     * Must always be first object in structure.
     */
    LW2080_CTRL_BOARDOBJ super;
    /*!
     * XAPI does not support using an element within another structure as a
     * discriminant. Placing this redundant type value here until that design
     * constraint can be fixed.
     */
    LwU8                 type;
    /*!
     * Type-specific data union.
     */
    union {
        LW2080_CTRL_PERF_PERF_CF_CONTROLLER_CONTROL_UTIL        util;
        LW2080_CTRL_PERF_PERF_CF_CONTROLLER_CONTROL_UTIL_2X     util2x;
        LW2080_CTRL_PERF_PERF_CF_CONTROLLER_CONTROL_OPTP_2X     optp2x;
        LW2080_CTRL_PERF_PERF_CF_CONTROLLER_CONTROL_DLPPC_1X    dlppc1x;
        LW2080_CTRL_PERF_PERF_CF_CONTROLLER_CONTROL_MEM_TUNE    memTune;
        LW2080_CTRL_PERF_PERF_CF_CONTROLLER_CONTROL_MEM_TUNE_2X memTune2x;
    } data;
} LW2080_CTRL_PERF_PERF_CF_CONTROLLER_CONTROL;
typedef struct LW2080_CTRL_PERF_PERF_CF_CONTROLLER_CONTROL *PLW2080_CTRL_PERF_PERF_CF_CONTROLLER_CONTROL;

/*!
 * Structure representing the control parameters of PERF_CF_CONTROLLERS.
 * Implements the BOARDOBJ model/interface.
 */
typedef struct LW2080_CTRL_PERF_PERF_CF_CONTROLLERS_CONTROL {
    /*!
     * LW2080_CTRL_BOARDOBJGRP_E32 super class.
     * Must always be first object in structure.
     */
    LW2080_CTRL_BOARDOBJGRP_E32                 super;
    /*!
     * Number of VM slots that are active in vGPU's scheduler.
     * In PVMRL fair share mode, it is the number of VMs lwrrently running.
     * In PVMRL fixed share mode, it is the maximum number of VMs that could be run.
     */
    LwU8                                        maxVGpuVMCount;
    /*!
     * Array of PERF_CF_CONTROLLER structures. Has valid indexes corresponding to
     * the bits set in @ref super.objMask.
     */
    LW2080_CTRL_PERF_PERF_CF_CONTROLLER_CONTROL controllers[LW2080_CTRL_PERF_PERF_CF_CONTROLLERS_MAX_OBJECTS];
} LW2080_CTRL_PERF_PERF_CF_CONTROLLERS_CONTROL;
typedef struct LW2080_CTRL_PERF_PERF_CF_CONTROLLERS_CONTROL *PLW2080_CTRL_PERF_PERF_CF_CONTROLLERS_CONTROL;

/*!
 * LW2080_CTRL_CMD_PERF_PERF_CF_CONTROLLERS_GET_CONTROL
 *
 * This command returns current PERF_CF_CONTROLLERS control parameters.
 *
 * See @ref LW2080_CTRL_PERF_PERF_CF_CONTROLLERS_CONTROL for documentation on
 * the parameters.
 *
 * Return values are specified per @ref BoardObjGrpGetControl.
 */
#define LW2080_CTRL_CMD_PERF_PERF_CF_CONTROLLERS_GET_CONTROL        (0x208020ca) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | 0xCA" */


/*!
 * LW2080_CTRL_CMD_PERF_PERF_CF_CONTROLLERS_SET_CONTROL
 *
 * This command accepts client-specified control parameters for a set of
 * PERF_CF_CONTROLLERS and applies these new parameters.
 *
 * See @ref LW2080_CTRL_PERF_PERF_CF_CONTROLLERS_CONTROL for documentation on
 * the parameters.
 *
 * Return values are specified per @ref BoardObjGrpSetControl.
 */
#define LW2080_CTRL_CMD_PERF_PERF_CF_CONTROLLERS_SET_CONTROL        (0x208020cb) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | 0xCB" */

/* ---------------------- CLIENT_PERF_CF Controllers ----------------------- */

/*!
 * Macros for CLIENT_PERF_CF Controller index
 */
#define LW2080_CTRL_PERF_CLIENT_PERF_CF_CONTROLLER_INDEX_ILWALID    LW2080_CTRL_BOARDOBJ_IDX_ILWALID

/*!
 * Macros for CLIENT_PERF_CF Controller types
 */
#define LW2080_CTRL_PERF_CLIENT_PERF_CF_CONTROLLER_TYPE_MEM_TUNE_1X 0x00

/*!
 * Structure describing CLIENT_PERF_CF_CONTROLLER_MEM_TUNE_1X static information/POR.
 */
typedef struct LW2080_CTRL_PERF_CLIENT_PERF_CF_CONTROLLER_INFO_MEM_TUNE {
    /*!
     * reserved for future.
     */
    LwU8 rsvd;
} LW2080_CTRL_PERF_CLIENT_PERF_CF_CONTROLLER_INFO_MEM_TUNE;

/*!
 * CLIENT_PERF_CF_CONTROLLER type-specific data union. Discriminated by
 * CLIENT_PERF_CF_CONTROLLER::super.type.
 */


/*!
 * Structure describing CLIENT_PERF_CF_CONTROLLER static information/POR.
 * Implements the BOARDOBJ model/interface.
 */
typedef struct LW2080_CTRL_PERF_CLIENT_PERF_CF_CONTROLLER_INFO {
    /*!
     * LW2080_CTRL_BOARDOBJ super class.
     * Must always be first object in structure.
     */
    LW2080_CTRL_BOARDOBJ super;
    /*!
     * XAPI does not support using an element within another structure as a
     * discriminant. Placing this redundant type value here until that design
     * constraint can be fixed.
     */
    LwU8                 type;

    /*!
     * Type-specific data union.
     */
    union {
        LW2080_CTRL_PERF_CLIENT_PERF_CF_CONTROLLER_INFO_MEM_TUNE memTune;
    } data;
} LW2080_CTRL_PERF_CLIENT_PERF_CF_CONTROLLER_INFO;

/*!
 * Structure describing CLIENT_PERF_CF_CONTROLLER static information/POR.
 * Implements the BOARDOBJGRP model/interface.
 */
#define LW2080_CTRL_PERF_CLIENT_PERF_CF_CONTROLLERS_INFO_MESSAGE_ID (0xE5U)

typedef struct LW2080_CTRL_PERF_CLIENT_PERF_CF_CONTROLLERS_INFO {
    /*!
     * LW2080_CTRL_BOARDOBJGRP_E32 super class.
     * Must always be first object in structure.
     */
    LW2080_CTRL_BOARDOBJGRP_E32                     super;

    /*!
     * Array of CLIENT_PERF_CF_CONTROLLER structures. Has valid indexes corresponding to
     * the bits set in @ref super.objMask.
     */
    LW2080_CTRL_PERF_CLIENT_PERF_CF_CONTROLLER_INFO controllers[LW2080_CTRL_BOARDOBJGRP_E32_MAX_OBJECTS];
} LW2080_CTRL_PERF_CLIENT_PERF_CF_CONTROLLERS_INFO;

/*!
 * LW2080_CTRL_CMD_PERF_CLIENT_PERF_CF_CONTROLLERS_GET_INFO
 *
 * This command returns CLIENT_PERF_CF_CONTROLLERS static object information/POR as
 * specified by the VBIOS in PERF_CF Table.
 *
 * See @ref LW2080_CTRL_PERF_CLIENT_PERF_CF_CONTROLLERS_INFO for documentation on
 * the parameters.
 *
 * Return values are specified per @ref BoardObjGrpGetInfo.
 */
#define LW2080_CTRL_CMD_PERF_CLIENT_PERF_CF_CONTROLLERS_GET_INFO (0x208020e5) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | LW2080_CTRL_PERF_CLIENT_PERF_CF_CONTROLLERS_INFO_MESSAGE_ID" */

/*!
 * Structure representing the dynamic state of CLIENT_PERF_CF_CONTROLLER_MEM_TUNE_1X.
 */
typedef struct LW2080_CTRL_PERF_CLIENT_PERF_CF_CONTROLLER_STATUS_MEM_TUNE {
    /*!
     * Boolean tracking whether the controller is armed. Depending on
     * the system configuration the controller may be disarmed in which
     * case the controller will continue to monitor the workload but
     * will not actively engage to throttle performance.
     */
    LwBool bArmed;
    /*!
     * Boolean tracking whether the active workload met the engage
     * criteria for the controller.
     */
    LwBool bEngageCriteriaMet;
    /*!
     * Boolean tracking whether the controller is engaged to actively
     * throttle the performance. This boolean is basically set to true
     * if the controller is armed and the engage criteria are satisfied.
     */
    LwBool bEngaged;
    /*!
     * Counter tracking the samples where the controller decision was
     * to engage.
     */
    LwU32  engageCounter;
} LW2080_CTRL_PERF_CLIENT_PERF_CF_CONTROLLER_STATUS_MEM_TUNE;

/*!
 * CLIENT_PERF_CF_CONTROLLER type-specific data union. Discriminated by
 * CLIENT_PERF_CF_CONTROLLER::super.type.
 */


/*!
 * Structure representing the dynamic state of each CLIENT_PERF_CF_CONTROLLER.
 * Implements the BOARDOBJ model/interface.
 */
typedef struct LW2080_CTRL_PERF_CLIENT_PERF_CF_CONTROLLER_STATUS {
    /*!
     * LW2080_CTRL_BOARDOBJ super class.
     * Must always be first object in structure.
     */
    LW2080_CTRL_BOARDOBJ super;
    /*!
     * XAPI does not support using an element within another structure as a
     * discriminant. Placing this redundant type value here until that design
     * constraint can be fixed.
     */
    LwU8                 type;
    /*!
     * Type-specific data union.
     */
    union {
        LW2080_CTRL_PERF_CLIENT_PERF_CF_CONTROLLER_STATUS_MEM_TUNE memTune;
    } data;
} LW2080_CTRL_PERF_CLIENT_PERF_CF_CONTROLLER_STATUS;

/*!
 * Structure representing the dynamic state of CLIENT_PERF_CF_CONTROLLERS.
 * Implements the BOARDOBJGRP model/interface.
 */
#define LW2080_CTRL_PERF_CLIENT_PERF_CF_CONTROLLERS_STATUS_MESSAGE_ID (0xE6U)

typedef struct LW2080_CTRL_PERF_CLIENT_PERF_CF_CONTROLLERS_STATUS {
    /*!
     * LW2080_CTRL_BOARDOBJGRP_E32 super class.
     * Must always be first object in structure.
     */
    LW2080_CTRL_BOARDOBJGRP_E32                       super;

    /*!
     * Mask of the lwrrently active Controllers.
     */
    LW2080_CTRL_BOARDOBJGRP_MASK_E32                  maskActive;

    /*!
     * Array of CLIENT_PERF_CF_CONTROLLER structures. Has valid indexes corresponding to
     * the bits set in @ref super.objMask.
     */
    LW2080_CTRL_PERF_CLIENT_PERF_CF_CONTROLLER_STATUS controllers[LW2080_CTRL_BOARDOBJGRP_E32_MAX_OBJECTS];
} LW2080_CTRL_PERF_CLIENT_PERF_CF_CONTROLLERS_STATUS;

/*!
 * LW2080_CTRL_CMD_PERF_CLIENT_PERF_CF_CONTROLLERS_GET_STATUS
 *
 * This command returns CLIENT_PERF_CF_CONTROLLERS dynamic state.
 *
 * See @ref LW2080_CTRL_PERF_CLIENT_PERF_CF_CONTROLLERS_STATUS for documentation on
 * the parameters.
 *
 * Return values are specified per @ref BoardObjGrpGetInfo.
 */
#define LW2080_CTRL_CMD_PERF_CLIENT_PERF_CF_CONTROLLERS_GET_STATUS  (0x208020e6) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | LW2080_CTRL_PERF_CLIENT_PERF_CF_CONTROLLERS_STATUS_MESSAGE_ID" */

/* ---------------------------- PERF_CF Policies ----------------------------- */

/*!
 * Macros for PERF_CF Policy HALs
 */
#define LW2080_CTRL_PERF_PERF_CF_POLICY_TABLE_HEADER_HAL_TYPE_TU10X 0x00
/*!
 * Macros for PERF_CF Policy types
 */
#define LW2080_CTRL_PERF_PERF_CF_POLICY_TYPE_CTRL_MASK              0x00

/*!
 * Macros for PERF_CF Policy labels
 */
#define LW2080_CTRL_PERF_PERF_CF_POLICY_LABEL_UTIL_DEFAULT          0x00
#define LW2080_CTRL_PERF_PERF_CF_POLICY_LABEL_UTIL_VIDEO_PLAYBACK   0x01
#define LW2080_CTRL_PERF_PERF_CF_POLICY_LABEL_OPTP_2X               0x02
#define LW2080_CTRL_PERF_PERF_CF_POLICY_LABEL_RP_PC_1X              0x03 // Deprecated
#define LW2080_CTRL_PERF_PERF_CF_POLICY_LABEL_UTIL_SLI              0x04
#define LW2080_CTRL_PERF_PERF_CF_POLICY_LABEL_UTIL_GSP              0x05
#define LW2080_CTRL_PERF_PERF_CF_POLICY_LABEL_MEM_TUNE              0x06
#define LW2080_CTRL_PERF_PERF_CF_POLICY_LABEL_DLPPC_1X              0x07U
#define LW2080_CTRL_PERF_PERF_CF_POLICY_LABEL_NUM                   0x08U
#define LW2080_CTRL_PERF_PERF_CF_POLICY_LABEL_NONE                  0xFF

/*!
 * Maximum number of PERF_CF_POLICYs which can be supported in the RM or PMU.
 */
#define LW2080_CTRL_PERF_PERF_CF_POLICYS_MAX_OBJECTS                LW2080_CTRL_BOARDOBJGRP_E32_MAX_OBJECTS

/*!
 * Structure describing PERF_CF_POLICY_CTRL_MASK static information/POR.
 */
typedef struct LW2080_CTRL_PERF_PERF_CF_POLICY_INFO_CTRL_MASK {
    /*!
     * Mask of PERF_CF controllers to enable for this policy.
     */
    LW2080_CTRL_BOARDOBJGRP_MASK_E32 maskControllers;
} LW2080_CTRL_PERF_PERF_CF_POLICY_INFO_CTRL_MASK;
typedef struct LW2080_CTRL_PERF_PERF_CF_POLICY_INFO_CTRL_MASK *PLW2080_CTRL_PERF_PERF_CF_POLICY_INFO_CTRL_MASK;

/*!
 * PERF_CF_POLICY type-specific data union. Discriminated by
 * PERF_CF_POLICY::super.type.
 */


/*!
 * Structure describing PERF_CF_POLICY static information/POR.
 * Implements the BOARDOBJ model/interface.
 */
typedef struct LW2080_CTRL_PERF_PERF_CF_POLICY_INFO {
    /*!
     * LW2080_CTRL_BOARDOBJ super class.
     * Must always be first object in structure.
     */
    LW2080_CTRL_BOARDOBJ             super;
    /*!
     * XAPI does not support using an element within another structure as a
     * discriminant. Placing this redundant type value here until that design
     * constraint can be fixed.
     */
    LwU8                             type;
    /*!
     * When multiple active policys are in conflict with each other, higher priority wins.
     */
    LwU8                             priority;
    /*!
     * Label. @ref LW2080_CTRL_PERF_PERF_CF_POLICY_LABEL_<xyz>.
     */
    LwU8                             label;
    /*!
     * Mask of policies this policy is in conflict with, meaning only 1 can be active at a time.
     */
    LW2080_CTRL_BOARDOBJGRP_MASK_E32 conflictMask;
    /*!
     * Type-specific data union.
     */
    union {
        LW2080_CTRL_PERF_PERF_CF_POLICY_INFO_CTRL_MASK ctrlMask;
    } data;
} LW2080_CTRL_PERF_PERF_CF_POLICY_INFO;
typedef struct LW2080_CTRL_PERF_PERF_CF_POLICY_INFO *PLW2080_CTRL_PERF_PERF_CF_POLICY_INFO;

/*!
 * Structure describing PERF_CF_POLICY static information/POR.
 * Implements the BOARDOBJGRP model/interface.
 */
#define LW2080_CTRL_PERF_PERF_CF_POLICYS_INFO_MESSAGE_ID (0xCLW)

typedef struct LW2080_CTRL_PERF_PERF_CF_POLICYS_INFO {
    /*!
     * LW2080_CTRL_BOARDOBJGRP_E32 super class.
     * Must always be first object in structure.
     */
    LW2080_CTRL_BOARDOBJGRP_E32          super;
    /*!
     * HAL index specifies the list of enumerants to use when interpreting the
     * policy entries.
     */
    LwU8                                 halVal;

    /*!
     * Mapping from LW2080_CTRL_PERF_PERF_CF_POLICY_LABEL values to the
     * associated @ref LwBoardObjIdx within the group.
     */
    LwBoardObjIdx                        labelToIdxMap[LW2080_CTRL_PERF_PERF_CF_POLICY_LABEL_NUM];

    /*!
     * Array of PERF_CF_POLICY structures. Has valid indexes corresponding to
     * the bits set in @ref super.objMask.
     */
    LW2080_CTRL_PERF_PERF_CF_POLICY_INFO policys[LW2080_CTRL_PERF_PERF_CF_POLICYS_MAX_OBJECTS];
} LW2080_CTRL_PERF_PERF_CF_POLICYS_INFO;
typedef struct LW2080_CTRL_PERF_PERF_CF_POLICYS_INFO *PLW2080_CTRL_PERF_PERF_CF_POLICYS_INFO;

/*!
 * LW2080_CTRL_CMD_PERF_PERF_CF_POLICYS_GET_INFO
 *
 * This command returns PERF_CF_POLICYS static object information/POR as
 * specified by the VBIOS in PERF_CF Table.
 *
 * See @ref LW2080_CTRL_PERF_PERF_CF_POLICYS_INFO for documentation on
 * the parameters.
 *
 * Return values are specified per @ref BoardObjGrpGetInfo.
 */
#define LW2080_CTRL_CMD_PERF_PERF_CF_POLICYS_GET_INFO (0x208020cc) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | LW2080_CTRL_PERF_PERF_CF_POLICYS_INFO_MESSAGE_ID" */


/*!
 * Structure representing the dynamic state of PERF_CF_POLICY_CTRL_MASK.
 */
typedef struct LW2080_CTRL_PERF_PERF_CF_POLICY_STATUS_CTRL_MASK {
    /*!
     * Lwrrently we do NOT have any dynamic state of this class.
     */
    LwU8 rsvd;
} LW2080_CTRL_PERF_PERF_CF_POLICY_STATUS_CTRL_MASK;
typedef struct LW2080_CTRL_PERF_PERF_CF_POLICY_STATUS_CTRL_MASK *PLW2080_CTRL_PERF_PERF_CF_POLICY_STATUS_CTRL_MASK;

/*!
 * PERF_CF_POLICY type-specific data union. Discriminated by
 * PERF_CF_POLICY::super.type.
 */


/*!
 * Mask of requests that can be used to deactivate a @ref PERF_CF_POLICY
 */
typedef LW2080_CTRL_BOARDOBJGRP_MASK_E32 LW2080_CTRL_PERF_PERF_CF_POLICY_DEACTIVATE_REQUEST_ID_MASK;

/*!
 * @defgroup LW2080_CTRL_PERF_PERF_CF_POLICY_DEACTIVATE_REQUEST_ID
 *
 * @brief   Enumeration of requestable IDs for why @ref PERF_CF_POLICY
 *          objects can be deactivated.
 *
 * @details
 *      LW2080_CTRL_PERF_PERF_CF_POLICY_DEACTIVATE_REQUEST_ID_RM__START
 *          First numerical ID in a contiguous set that is controlled by RM.
 *
 *      LW2080_CTRL_PERF_PERF_CF_POLICY_DEACTIVATE_REQUEST_ID_RM_LWDA_CONFLICT
 *          Request from RM to disable a policy because it is conflicting with
 *          LWCA.
 *
 *      LW2080_CTRL_PERF_PERF_CF_POLICY_DEACTIVATE_REQUEST_ID_RM_4K_VIDEO_CONFLICT
 *          Request from RM to disable a policy because it is conflicting with
 *          4K video.
 *
 *      LW2080_CTRL_PERF_PERF_CF_POLICY_DEACTIVATE_REQUEST_ID_RM_BATTERY_CONFLICT
 *          Request from RM to disable a policy because it is conflicting with
 *          the battery mode.
 *
 *      LW2080_CTRL_PERF_PERF_CF_POLICY_DEACTIVATE_REQUEST_ID_RM__END
 *          Last numerical ID in the contiguous set that is controlled by RM.
 *
 *      LW2080_CTRL_PERF_PERF_CF_POLICY_DEACTIVATE_REQUEST_ID_PMU__START
 *          First numerical ID in a contiguous set that is controlled by the
 *          PMU.
 *
 *      LW2080_CTRL_PERF_PERF_CF_POLICY_DEACTIVATE_REQUEST_ID_PMU_OPTP_CONFLICT
 *          Request internally in the PMU to disable a policy because it is
 *          conflicting with OPTP.
 *
 *      LW2080_CTRL_PERF_PERF_CF_POLICY_DEACTIVATE_REQUEST_ID_PMU__END
 *          Last numerical ID in the contiguous set that is controlled by the
 *          PMU.
 *
 *      LW2080_CTRL_PERF_PERF_CF_POLICY_DEACTIVATE_REQUEST_ID__NUM
 *          Number of request IDs.
 * @{
 */
typedef LwBoardObjIdx LW2080_CTRL_PERF_PERF_CF_POLICY_DEACTIVATE_REQUEST_ID;
#define LW2080_CTRL_PERF_PERF_CF_POLICY_DEACTIVATE_REQUEST_ID_RM__START            0U
#define LW2080_CTRL_PERF_PERF_CF_POLICY_DEACTIVATE_REQUEST_ID_RM_LWDA_CONFLICT     (0x0U) /* finn: Evaluated from "(LW2080_CTRL_PERF_PERF_CF_POLICY_DEACTIVATE_REQUEST_ID_RM__START + 0)" */
#define LW2080_CTRL_PERF_PERF_CF_POLICY_DEACTIVATE_REQUEST_ID_RM_4K_VIDEO_CONFLICT (0x1U) /* finn: Evaluated from "(LW2080_CTRL_PERF_PERF_CF_POLICY_DEACTIVATE_REQUEST_ID_RM__START + 1)" */
#define LW2080_CTRL_PERF_PERF_CF_POLICY_DEACTIVATE_REQUEST_ID_RM_BATTERY_CONFLICT  (0x2U) /* finn: Evaluated from "(LW2080_CTRL_PERF_PERF_CF_POLICY_DEACTIVATE_REQUEST_ID_RM__START + 2)" */
#define LW2080_CTRL_PERF_PERF_CF_POLICY_DEACTIVATE_REQUEST_ID_RM__END              LW2080_CTRL_PERF_PERF_CF_POLICY_DEACTIVATE_REQUEST_ID_RM_BATTERY_CONFLICT
#define LW2080_CTRL_PERF_PERF_CF_POLICY_DEACTIVATE_REQUEST_ID_PMU__START           (0x3U) /* finn: Evaluated from "(LW2080_CTRL_PERF_PERF_CF_POLICY_DEACTIVATE_REQUEST_ID_RM__END + 1)" */
#define LW2080_CTRL_PERF_PERF_CF_POLICY_DEACTIVATE_REQUEST_ID_PMU_OPTP_CONFLICT    (0x3U) /* finn: Evaluated from "(LW2080_CTRL_PERF_PERF_CF_POLICY_DEACTIVATE_REQUEST_ID_PMU__START + 0)" */
#define LW2080_CTRL_PERF_PERF_CF_POLICY_DEACTIVATE_REQUEST_ID_PMU__END             (LW2080_CTRL_PERF_PERF_CF_POLICY_DEACTIVATE_REQUEST_ID_PMU_OPTP_CONFLICT)
#define LW2080_CTRL_PERF_PERF_CF_POLICY_DEACTIVATE_REQUEST_ID__NUM                 (0x4U) /* finn: Evaluated from "(LW2080_CTRL_PERF_PERF_CF_POLICY_DEACTIVATE_REQUEST_ID_PMU__END + 1)" */
/*!@}*/

/*!
 * Structure representing the dynamic state of each PERF_CF_POLICY.
 * Implements the BOARDOBJ model/interface.
 */
typedef struct LW2080_CTRL_PERF_PERF_CF_POLICY_STATUS {
    /*!
     * LW2080_CTRL_BOARDOBJ super class.
     * Must always be first object in structure.
     */
    LW2080_CTRL_BOARDOBJ                                       super;
    /*!
     * XAPI does not support using an element within another structure as a
     * discriminant. Placing this redundant type value here until that design
     * constraint can be fixed.
     */
    LwU8                                                       type;
    /*!
     * Current active/inactive state (after resolving conflict).
     */
    LwBool                                                     bActiveLwrr;

    /*!
     * Mask of requests to deactivate the policy.
     */
    LW2080_CTRL_PERF_PERF_CF_POLICY_DEACTIVATE_REQUEST_ID_MASK deactivateMask;

    /*!
     * Type-specific data union.
     */
    union {
        LW2080_CTRL_PERF_PERF_CF_POLICY_STATUS_CTRL_MASK ctrlMask;
    } data;
} LW2080_CTRL_PERF_PERF_CF_POLICY_STATUS;
typedef struct LW2080_CTRL_PERF_PERF_CF_POLICY_STATUS *PLW2080_CTRL_PERF_PERF_CF_POLICY_STATUS;

/*!
 * Structure representing the dynamic state of PERF_CF_POLICYS.
 * Implements the BOARDOBJGRP model/interface.
 */
#define LW2080_CTRL_PERF_PERF_CF_POLICYS_STATUS_MESSAGE_ID (0xCDU)

typedef struct LW2080_CTRL_PERF_PERF_CF_POLICYS_STATUS {
    /*!
     * LW2080_CTRL_BOARDOBJGRP_E32 super class.
     * Must always be first object in structure.
     */
    LW2080_CTRL_BOARDOBJGRP_E32            super;
    /*!
     * Mask of policies requested (disregarding potential conflicts).
     */
    LW2080_CTRL_BOARDOBJGRP_MASK_E32       activeMaskRequested;
    /*!
     * Mask of policies lwrrently active (after resolving conflict).
     */
    LW2080_CTRL_BOARDOBJGRP_MASK_E32       activeMaskArbitrated;
    /*!
     * Array of PERF_CF_POLICY structures. Has valid indexes corresponding to
     * the bits set in @ref super.objMask.
     */
    LW2080_CTRL_PERF_PERF_CF_POLICY_STATUS policys[LW2080_CTRL_PERF_PERF_CF_POLICYS_MAX_OBJECTS];
} LW2080_CTRL_PERF_PERF_CF_POLICYS_STATUS;
typedef struct LW2080_CTRL_PERF_PERF_CF_POLICYS_STATUS *PLW2080_CTRL_PERF_PERF_CF_POLICYS_STATUS;

/*!
 * LW2080_CTRL_CMD_PERF_PERF_CF_POLICYS_GET_STATUS
 *
 * This command returns PERF_CF_POLICYS dynamic state.
 *
 * See @ref LW2080_CTRL_PERF_PERF_CF_POLICYS_STATUS for documentation on
 * the parameters.
 *
 * Return values are specified per @ref BoardObjGrpGetInfo.
 */
#define LW2080_CTRL_CMD_PERF_PERF_CF_POLICYS_GET_STATUS (0x208020cd) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | LW2080_CTRL_PERF_PERF_CF_POLICYS_STATUS_MESSAGE_ID" */


/*!
 * Structure representing the control parameters of PERF_CF_POLICY_CTRL_MASK.
 */
typedef struct LW2080_CTRL_PERF_PERF_CF_POLICY_CONTROL_CTRL_MASK {
    /*!
     * Lwrrently we do NOT have any control parameter in this class.
     */
    LwU8 rsvd;
} LW2080_CTRL_PERF_PERF_CF_POLICY_CONTROL_CTRL_MASK;
typedef struct LW2080_CTRL_PERF_PERF_CF_POLICY_CONTROL_CTRL_MASK *PLW2080_CTRL_PERF_PERF_CF_POLICY_CONTROL_CTRL_MASK;

/*!
 * PERF_CF_POLICY type-specific data union. Discriminated by
 * PERF_CF_POLICY::super.type.
 */


/*!
 * Structure representing the control parameters of each PERF_CF_POLICY.
 * Implements the BOARDOBJ model/interface.
 */
typedef struct LW2080_CTRL_PERF_PERF_CF_POLICY_CONTROL {
    /*!
     * LW2080_CTRL_BOARDOBJ super class.
     * Must always be first object in structure.
     */
    LW2080_CTRL_BOARDOBJ super;
    /*!
     * XAPI does not support using an element within another structure as a
     * discriminant. Placing this redundant type value here until that design
     * constraint can be fixed.
     */
    LwU8                 type;
    /*!
     * LW_TRUE = activate/activated, LW_FALSE = deactivate/deactivated.
     */
    LwBool               bActivate;
    /*!
     * Type-specific data union.
     */
    union {
        LW2080_CTRL_PERF_PERF_CF_POLICY_CONTROL_CTRL_MASK ctrlMask;
    } data;
} LW2080_CTRL_PERF_PERF_CF_POLICY_CONTROL;
typedef struct LW2080_CTRL_PERF_PERF_CF_POLICY_CONTROL *PLW2080_CTRL_PERF_PERF_CF_POLICY_CONTROL;

/*!
 * Structure representing the control parameters of PERF_CF_POLICYS.
 * Implements the BOARDOBJ model/interface.
 */
typedef struct LW2080_CTRL_PERF_PERF_CF_POLICYS_CONTROL {
    /*!
     * LW2080_CTRL_BOARDOBJGRP_E32 super class.
     * Must always be first object in structure.
     */
    LW2080_CTRL_BOARDOBJGRP_E32             super;
    /*!
     * Array of PERF_CF_POLICY structures. Has valid indexes corresponding to
     * the bits set in @ref super.objMask.
     */
    LW2080_CTRL_PERF_PERF_CF_POLICY_CONTROL policys[LW2080_CTRL_PERF_PERF_CF_POLICYS_MAX_OBJECTS];
} LW2080_CTRL_PERF_PERF_CF_POLICYS_CONTROL;
typedef struct LW2080_CTRL_PERF_PERF_CF_POLICYS_CONTROL *PLW2080_CTRL_PERF_PERF_CF_POLICYS_CONTROL;

/*!
 * LW2080_CTRL_CMD_PERF_PERF_CF_POLICYS_GET_CONTROL
 *
 * This command returns current PERF_CF_POLICYS control parameters.
 *
 * See @ref LW2080_CTRL_PERF_PERF_CF_POLICYS_CONTROL for documentation on
 * the parameters.
 *
 * Return values are specified per @ref BoardObjGrpGetControl.
 */
#define LW2080_CTRL_CMD_PERF_PERF_CF_POLICYS_GET_CONTROL      (0x208020ce) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | 0xCE" */


/*!
 * LW2080_CTRL_CMD_PERF_PERF_CF_POLICYS_SET_CONTROL
 *
 * This command accepts client-specified control parameters for a set of
 * PERF_CF_POLICYS and applies these new parameters.
 *
 * See @ref LW2080_CTRL_PERF_PERF_CF_POLICYS_CONTROL for documentation on
 * the parameters.
 *
 * Return values are specified per @ref BoardObjGrpSetControl.
 */
#define LW2080_CTRL_CMD_PERF_PERF_CF_POLICYS_SET_CONTROL      (0x208020cf) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | 0xCF" */


/* ---------------------- CLIENT_PERF_CF Policies -------------------------- */

/*!
 * Macros for CLIENT_PERF_CF Policy types
 */
#define LW2080_CTRL_PERF_CLIENT_PERF_CF_POLICY_TYPE_CTRL_MASK 0x00

/*!
 * Macros for PERF_CF Policy labels
 */
#define LW2080_CTRL_PERF_CLIENT_PERF_CF_POLICY_LABEL_MEM_TUNE 0x00
#define LW2080_CTRL_PERF_CLIENT_PERF_CF_POLICY_LABEL_NUM      0x01
#define LW2080_CTRL_PERF_CLIENT_PERF_CF_POLICY_LABEL_NONE     0xFF

/*!
 * Structure describing CLIENT_PERF_CF_POLICY_CTRL_MASK static information/POR.
 */
typedef struct LW2080_CTRL_PERF_CLIENT_PERF_CF_POLICY_INFO_CTRL_MASK {
    /*!
     * Mask of CLIENT_PERF_CF controllers to enable for this policy.
     */
    LW2080_CTRL_BOARDOBJGRP_MASK_E32 maskControllers;
} LW2080_CTRL_PERF_CLIENT_PERF_CF_POLICY_INFO_CTRL_MASK;

/*!
 * CLIENT_PERF_CF_POLICY type-specific data union. Discriminated by
 * CLIENT_PERF_CF_POLICY::super.type.
 */


/*!
 * Structure describing CLIENT_PERF_CF_POLICY static information/POR.
 * Implements the BOARDOBJ model/interface.
 */
typedef struct LW2080_CTRL_PERF_CLIENT_PERF_CF_POLICY_INFO {
    /*!
     * LW2080_CTRL_BOARDOBJ super class.
     * Must always be first object in structure.
     */
    LW2080_CTRL_BOARDOBJ super;
    /*!
     * XAPI does not support using an element within another structure as a
     * discriminant. Placing this redundant type value here until that design
     * constraint can be fixed.
     */
    LwU8                 type;
    /*!
     * Label. @ref LW2080_CTRL_PERF_CLIENT_PERF_CF_POLICY_LABEL_<xyz>.
     */
    LwU8                 label;
    /*!
     * Type-specific data union.
     */
    union {
        LW2080_CTRL_PERF_CLIENT_PERF_CF_POLICY_INFO_CTRL_MASK ctrlMask;
    } data;
} LW2080_CTRL_PERF_CLIENT_PERF_CF_POLICY_INFO;

/*!
 * Structure describing CLIENT_PERF_CF_POLICY static information/POR.
 * Implements the BOARDOBJGRP model/interface.
 */
#define LW2080_CTRL_PERF_CLIENT_PERF_CF_POLICYS_INFO_MESSAGE_ID (0xE7U)

typedef struct LW2080_CTRL_PERF_CLIENT_PERF_CF_POLICYS_INFO {
    /*!
     * LW2080_CTRL_BOARDOBJGRP_E32 super class.
     * Must always be first object in structure.
     */
    LW2080_CTRL_BOARDOBJGRP_E32                 super;

    /*!
     * Array of CLIENT_PERF_CF_POLICY structures. Has valid indexes corresponding to
     * the bits set in @ref super.objMask.
     */
    LW2080_CTRL_PERF_CLIENT_PERF_CF_POLICY_INFO policys[LW2080_CTRL_BOARDOBJGRP_E32_MAX_OBJECTS];
} LW2080_CTRL_PERF_CLIENT_PERF_CF_POLICYS_INFO;

/*!
 * LW2080_CTRL_CMD_PERF_CLIENT_PERF_CF_POLICYS_GET_INFO
 *
 * This command returns CLIENT_PERF_CF_POLICYS static object information/POR as
 * specified by the VBIOS in PERF_CF Table.
 *
 * See @ref LW2080_CTRL_PERF_CLIENT_PERF_CF_POLICYS_INFO for documentation on
 * the parameters.
 *
 * Return values are specified per @ref BoardObjGrpGetInfo.
 */
#define LW2080_CTRL_CMD_PERF_CLIENT_PERF_CF_POLICYS_GET_INFO (0x208020e7) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | LW2080_CTRL_PERF_CLIENT_PERF_CF_POLICYS_INFO_MESSAGE_ID" */

/*!
 * Structure representing the dynamic state of each CLIENT_PERF_CF_POLICY.
 * Implements the BOARDOBJ model/interface.
 */
typedef struct LW2080_CTRL_PERF_CLIENT_PERF_CF_POLICY_STATUS {
    /*!
     * LW2080_CTRL_BOARDOBJ super class.
     * Must always be first object in structure.
     */
    LW2080_CTRL_BOARDOBJ super;
} LW2080_CTRL_PERF_CLIENT_PERF_CF_POLICY_STATUS;

/*!
 * Structure representing the dynamic state of CLIENT_PERF_CF_POLICYS.
 * Implements the BOARDOBJGRP model/interface.
 */
#define LW2080_CTRL_PERF_CLIENT_PERF_CF_POLICYS_STATUS_MESSAGE_ID (0xE8U)

typedef struct LW2080_CTRL_PERF_CLIENT_PERF_CF_POLICYS_STATUS {
    /*!
     * LW2080_CTRL_BOARDOBJGRP_E32 super class.
     * Must always be first object in structure.
     */
    LW2080_CTRL_BOARDOBJGRP_E32                   super;
    /*!
     * Mask of policies lwrrently active (after resolving conflict).
     */
    LW2080_CTRL_BOARDOBJGRP_MASK_E32              activeMask;
    /*!
     * Array of CLIENT_PERF_CF_POLICY structures. Has valid indexes
     * corresponding to the bits set in @ref super.objMask.
     */
    LW2080_CTRL_PERF_CLIENT_PERF_CF_POLICY_STATUS policys[LW2080_CTRL_BOARDOBJGRP_E32_MAX_OBJECTS];
} LW2080_CTRL_PERF_CLIENT_PERF_CF_POLICYS_STATUS;

/*!
 * LW2080_CTRL_CMD_PERF_CLIENT_PERF_CF_POLICYS_GET_STATUS
 *
 * This command returns CLIENT_PERF_CF_POLICYS dynamic state.
 *
 * See @ref LW2080_CTRL_PERF_CLIENT_PERF_CF_POLICYS_STATUS for documentation on
 * the parameters.
 *
 * Return values are specified per @ref BoardObjGrpGetInfo.
 */
#define LW2080_CTRL_CMD_PERF_CLIENT_PERF_CF_POLICYS_GET_STATUS     (0x208020e8) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | LW2080_CTRL_PERF_CLIENT_PERF_CF_POLICYS_STATUS_MESSAGE_ID" */


/* ---------------------------- PERF_CF Topologies ----------------------------- */
/*!
 * LW2080_CTRL_CMD_PERF_PMUMON_PERF_CF_TOPOLOGIES_GET_SAMPLES
 *
 * Control call to query the samples within the PWR_CHANNELS PMUMON queue.
 */
#define LW2080_CTRL_CMD_PERF_PMUMON_PERF_CF_TOPOLOGIES_GET_SAMPLES (0x208020d1) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | LW2080_CTRL_PERF_PMUMON_PERF_CF_TOPOLOGIES_GET_SAMPLES_PARAMS_MESSAGE_ID" */

/*!
 * @brief   With sample period being potentially as fast every 100ms, this gives
 *          us 5 seconds worth of data.
 */
#define LW2080_CTRL_PERF_PMUMON_PERF_CF_TOPOLOGIES_SAMPLE_COUNT    (50U)

/*!
 * Temporary until an INFO control call is stubbed out that exposes the supported
 * feature set of the sampling.
 */
#define LW2080_CTRL_PERF_PMUMON_PERF_CF_TOPOLOGIES_SAMPLE_ILWALID  (LW_U32_MAX)

/*!
 * A single sample of the power channels at a particular point in time.
 */
typedef struct LW2080_CTRL_PERF_PMUMON_PERF_CF_TOPOLOGIES_SAMPLE {
    /*!
     * Ptimer timestamp of when this data was collected.
     */
    LW_DECLARE_ALIGNED(LW2080_CTRL_PMUMON_SAMPLE super, 8);

    /*!
     * Percent of time GR was busy since last sample.
     * Units of percent*100; i.e. 5000 = 50%.
     *
     * LW2080_CTRL_PERF_PMUMON_PERF_CF_TOPOLOGIES_SAMPLE_ILWALID if not supported.
     */
    LwU32 grUtil;

    /*!
     * Percent of time FB was busy since last sample.
     * Units of percent*100; i.e. 5000 = 50%.
     *
     * LW2080_CTRL_PERF_PMUMON_PERF_CF_TOPOLOGIES_SAMPLE_ILWALID if not supported.
     */
    LwU32 fbUtil;

    /*!
     * Percent of time VID was busy since last sample.
     * Units of percent*100; i.e. 5000 = 50%.
     *
     * LW2080_CTRL_PERF_PMUMON_PERF_CF_TOPOLOGIES_SAMPLE_ILWALID if not supported.
     */
    LwU32 vidUtil;

    /*!
     * Percent of time LWENC was busy since last sample.
     * Units of percent*100; i.e. 5000 = 50%.
     *
     * This is the average across all LWENC if there are multiple.
     *
     * LW2080_CTRL_PERF_PMUMON_PERF_CF_TOPOLOGIES_SAMPLE_ILWALID if not supported.
     */
    LwU32 lwenlwtil;

    /*!
     * Percent of time LWDEC was busy since last sample.
     * Units of percent*100; i.e. 5000 = 50%.
     *
     * This is the average across all LWENC if there are multiple.
     *
     * LW2080_CTRL_PERF_PMUMON_PERF_CF_TOPOLOGIES_SAMPLE_ILWALID if not supported.
     */
    LwU32 lwdelwtil;
} LW2080_CTRL_PERF_PMUMON_PERF_CF_TOPOLOGIES_SAMPLE;
typedef struct LW2080_CTRL_PERF_PMUMON_PERF_CF_TOPOLOGIES_SAMPLE *PLW2080_CTRL_PERF_PMUMON_PERF_CF_TOPOLOGIES_SAMPLE;

/*!
 * Input/Output parameters for @ref LW2080_CTRL_CMD_PERF_PMUMON_PERF_CF_TOPOLOGIES_GET_SAMPLES
 */
#define LW2080_CTRL_PERF_PMUMON_PERF_CF_TOPOLOGIES_GET_SAMPLES_PARAMS_MESSAGE_ID (0xD1U)

typedef struct LW2080_CTRL_PERF_PMUMON_PERF_CF_TOPOLOGIES_GET_SAMPLES_PARAMS {
    /*!
     * [in/out] Meta-data for the samples[] below. Will be modified by the
     *          control call on caller's behalf and should be passed back in
     *          un-modified for subsequent calls.
     */
    LW2080_CTRL_PMUMON_GET_SAMPLES_SUPER super;

    /*!
     * [out] Between the last call and current call, samples[0...super.numSamples-1]
     *       have been published to the pmumon queue. Samples are copied into
     *       this buffer in chronological order. Indexes within this buffer do
     *       not represent indexes of samples in the actual PMUMON queue.
     */
    LW_DECLARE_ALIGNED(LW2080_CTRL_PERF_PMUMON_PERF_CF_TOPOLOGIES_SAMPLE samples[LW2080_CTRL_PERF_PMUMON_PERF_CF_TOPOLOGIES_SAMPLE_COUNT], 8);
} LW2080_CTRL_PERF_PMUMON_PERF_CF_TOPOLOGIES_GET_SAMPLES_PARAMS;
typedef struct LW2080_CTRL_PERF_PMUMON_PERF_CF_TOPOLOGIES_GET_SAMPLES_PARAMS *PLW2080_CTRL_PERF_PMUMON_PERF_CF_TOPOLOGIES_GET_SAMPLES_PARAMS;

/* ---------------------------- PERF_CF_PM Sensors ----------------------------- */
/*!
 * LW2080_CTRL_CMD_PERF_PMUMON_PERF_CF_PM_SENSORS_GET_SAMPLES
 *
 * Control call to query the samples within the PWR_CHANNELS PMUMON queue.
 */
#define LW2080_CTRL_CMD_PERF_PMUMON_PERF_CF_SENSORS_GET_SAMPLES                    (0x208020e4) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | LW2080_CTRL_PERF_PMUMON_PERF_CF_PM_SENSORS_GET_SAMPLES_PARAMS_ARRAY_MESSAGE_ID" */

/*!
 * @brief   With sample period being potentially as fast every 50ms, this gives
 *          us 1 second worth of data.
 * cannot be > 1 second worth of data as the accepted latency of < 5ms can only be obtained with buffer sizes less than 1 second
 * confluence: https://confluence.lwpu.com/display/RMDL/BA+counter+PMUMON+latency+issues
 */
#define LW2080_CTRL_PERF_PMUMON_PERF_CF_PM_SENSORS_SAMPLE_COUNT                    (20U)

#define LW2080_CTRL_PERF_PMUMON_PERF_CF_PM_SENSORS_SAMPLE_BOARDOBJ_SUPPORTED_COUNT (16U)

/*!
 * @brief  enabled board objects here
 */
#define LW2080_CTRL_PERF_PMUMON_PERF_CF_PM_SENSORS_SAMPLE_BOARDOBJ_ENABLED_COUNT   (10U)
/*!
 * Temporary until an INFO control call is stubbed out that exposes the supported
 * feature set of the sampling.
 */
#define LW2080_CTRL_PERF_PMUMON_PERF_CF_PM_SENSORS_SAMPLE_ILWALID                  (LW_U32_MAX)

/*!
 * A single sample of the perf cf sensors at a particular point in time.
 */
typedef struct LW2080_CTRL_PERF_PMUMON_PERF_CF_PM_SENSORS_SAMPLE {
    /*!
     * Ptimer timestamp of when this data was collected.
     */
    LW_DECLARE_ALIGNED(LW2080_CTRL_PMUMON_SAMPLE super, 8);

    /*!
     * Add the signals data here LW2080_CTRL_PERF_PERF_CF_PM_SENSOR_SIGNAL_STATUS
     */
    LW2080_CTRL_PERF_PERF_CF_PM_SENSOR_SIGNAL_STATUS signals[LW2080_CTRL_PERF_PERF_CF_PM_SENSOR_MAX_SIGNALS];
} LW2080_CTRL_PERF_PMUMON_PERF_CF_PM_SENSORS_SAMPLE;

/*!
 * Input/Output parameters for @ref LW2080_CTRL_CMD_PERF_PMUMON_PERF_CF_PM_SENSORS_GET_SAMPLES
 */
typedef struct LW2080_CTRL_PERF_PMUMON_PERF_CF_PM_SENSORS_GET_SAMPLES_PARAMS {
    /*!
     * [in/out] Meta-data for the samples[] below. Will be modified by the
     *          control call on caller's behalf and should be passed back in
     *          un-modified for subsequent calls.
     */
    LW2080_CTRL_PMUMON_GET_SAMPLES_SUPER super;

    /*!
     * [out] Between the last call and current call, samples[0...super.numSamples-1]
     *       have been published to the pmumon queue. Samples are copied into
     *       this buffer in chronological order. Indexes within this buffer do
     *       not represent indexes of samples in the actual PMUMON queue.
     */
    LW_DECLARE_ALIGNED(LW2080_CTRL_PERF_PMUMON_PERF_CF_PM_SENSORS_SAMPLE samples[LW2080_CTRL_PERF_PMUMON_PERF_CF_PM_SENSORS_SAMPLE_COUNT], 8);
} LW2080_CTRL_PERF_PMUMON_PERF_CF_PM_SENSORS_GET_SAMPLES_PARAMS;

/*!
 * Input/Output parameters for @ref LW2080_CTRL_CMD_PERF_PMUMON_PERF_CF_PM_SENSORS_GET_SAMPLES_PARAMS_ARRAY
 */
#define LW2080_CTRL_PERF_PMUMON_PERF_CF_PM_SENSORS_GET_SAMPLES_PARAMS_ARRAY_MESSAGE_ID (0xE4U)

typedef struct LW2080_CTRL_PERF_PMUMON_PERF_CF_PM_SENSORS_GET_SAMPLES_PARAMS_ARRAY {

    /*! [in/out]
     * LW2080_CTRL_BOARDOBJGRP_E32 super class.
     * set indexes in this mask dictate which per-object samplesArray entry is populated.
     */
    LW2080_CTRL_BOARDOBJGRP_E32 pmSensorsQueryMask;

    /*!
     * [out] Between the last call and current call, samples[0...super.numSamples-1]
     *       have been published to the pmumon queue. Samples are copied into
     *       this buffer in chronological order. Indexes within this buffer do
     *       not represent indexes of samples in the actual PMUMON queue.
     */
    LW_DECLARE_ALIGNED(LW2080_CTRL_PERF_PMUMON_PERF_CF_PM_SENSORS_GET_SAMPLES_PARAMS samplesArray[LW2080_CTRL_PERF_PMUMON_PERF_CF_PM_SENSORS_SAMPLE_BOARDOBJ_SUPPORTED_COUNT], 8);
} LW2080_CTRL_PERF_PMUMON_PERF_CF_PM_SENSORS_GET_SAMPLES_PARAMS_ARRAY;
/* _ctrl2080perf_cf_h_ */
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)

