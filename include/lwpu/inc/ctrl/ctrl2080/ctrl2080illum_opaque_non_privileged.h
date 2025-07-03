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

/*
 * This file should NEVER be published as it contains opaque privileged control
 * commands and parameters for Volt module. 
 */

#pragma once

#include <lwtypes.h>
#if defined(_MSC_VER)
#pragma warning(disable:4324)
#endif

//
// This file was generated with FINN, an LWPU coding tool.
// Source file: ctrl/ctrl2080/ctrl2080illum_opaque_non_privileged.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

#include "ctrl2080illum.h"
#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)


/* ---------------------------- ILLUM_DEVICE-s ------------------------------ */
/*!
 * Macros for ILLUM_DEVICE types
 */
#define LW2080_CTRL_PMGR_ILLUM_DEVICE_TYPE_ILWALID                   0x00
#define LW2080_CTRL_PMGR_ILLUM_DEVICE_TYPE_MLWV10                    0x01
#define LW2080_CTRL_PMGR_ILLUM_DEVICE_TYPE_GPIO_PWM_RGBW_V10         0x02
#define LW2080_CTRL_PMGR_ILLUM_DEVICE_TYPE_GPIO_PWM_SINGLE_COLOR_V10 0x03

/*!
 * Structure describing ILLUM_DEVICE_MLWV10 static information/POR.
 */
typedef struct LW2080_CTRL_PMGR_ILLUM_DEVICE_INFO_MLWV10 {
    /*!
     * I2C Device Index: Pointing to the illumination device in I2C Devices Table
     */
    LwU8 i2cDevIdx;
} LW2080_CTRL_PMGR_ILLUM_DEVICE_INFO_MLWV10;
typedef struct LW2080_CTRL_PMGR_ILLUM_DEVICE_INFO_MLWV10 *PLW2080_CTRL_PMGR_ILLUM_DEVICE_INFO_MLWV10;

/*!
 * Structure describing ILLUM_DEVICE_GPIO_PWM_SINGLE_COLOR_V10 static information/POR.
 */
typedef struct LW2080_CTRL_PMGR_ILLUM_DEVICE_INFO_GPIO_PWM_SINGLE_COLOR_V10 {
    /*!
     * Single Color GPIO Function.
     */
    LwU8  gpioFuncSingleColor;

    /*!
     * Single Color GPIO pin.
     */
    LwU8  gpioPinSingleColor;

    /*!
     * PWM source driving the GPIO pin.
     */
    LwU8  pwmSource;

    /*!
     * PWM period
     */
    LwU32 rawPeriod;
} LW2080_CTRL_PMGR_ILLUM_DEVICE_INFO_GPIO_PWM_SINGLE_COLOR_V10;
typedef struct LW2080_CTRL_PMGR_ILLUM_DEVICE_INFO_GPIO_PWM_SINGLE_COLOR_V10 *PLW2080_CTRL_PMGR_ILLUM_DEVICE_INFO_GPIO_PWM_SINGLE_COLOR_V10;

/*!
 * Structure describing ILLUM_DEVICE_GPIO_PWM_RGBW_V10 static information/POR.
 */
typedef struct LW2080_CTRL_PMGR_ILLUM_DEVICE_INFO_GPIO_PWM_RGBW_V10 {
    /*!
     * Red drive GPIO Function.
     */
    LwU8  gpioFuncRed;

    /*!
     * Green drive GPIO Function.
     */
    LwU8  gpioFuncGreen;

    /*!
     * Blue drive GPIO Function.
     */
    LwU8  gpioFuncBlue;

    /*!
     * White drive GPIO Function.
     */
    LwU8  gpioFuncWhite;

    /*!
     * Red drive GPIO pin.
     */
    LwU8  gpioPinRed;

    /*!
     * Green drive GPIO pin.
     */
    LwU8  gpioPinGreen;

    /*!
     * Blue drive GPIO pin.
     */
    LwU8  gpioPinBlue;

    /*!
     * White drive GPIO pin.
     */
    LwU8  gpioPinWhite;

    /*!
     * PWM source driving the red GPIO pin.
     */
    LwU8  pwmSourceRed;

    /*!
     * PWM source driving the green GPIO pin.
     */
    LwU8  pwmSourceGreen;

    /*!
     * PWM source driving the blue GPIO pin.
     */
    LwU8  pwmSourceBlue;

    /*!
     * PWM source driving the white GPIO pin.
     */
    LwU8  pwmSourceWhite;

    /*!
     * PWM period. This is common to all R,G,B and W.
     */
    LwU32 rawPeriod;
} LW2080_CTRL_PMGR_ILLUM_DEVICE_INFO_GPIO_PWM_RGBW_V10;
typedef struct LW2080_CTRL_PMGR_ILLUM_DEVICE_INFO_GPIO_PWM_RGBW_V10 *PLW2080_CTRL_PMGR_ILLUM_DEVICE_INFO_GPIO_PWM_RGBW_V10;

/*!
 * ILLUM_DEVICE type-specific data union. Discriminated by
 * ILLUM_DEVICE::super.type.
 */


/*!
 * Structure describing ILLUM_DEVICE static information/POR.
 * Implements the BOARDOBJ model/interface.
 */
typedef struct LW2080_CTRL_PMGR_ILLUM_DEVICE_INFO {
    /*!
     * LW2080_CTRL_BOARDOBJ super class.
     * Must always be first object in structure.
     */
#endif // (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1) && (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
    LW2080_CTRL_BOARDOBJ super;
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1) && (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

    /*!
     * Supported control modes for this illumination device
     */
    LwU32                ctrlModeMask;

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
        LW2080_CTRL_PMGR_ILLUM_DEVICE_INFO_MLWV10                    mlwv10;
        LW2080_CTRL_PMGR_ILLUM_DEVICE_INFO_GPIO_PWM_SINGLE_COLOR_V10 gpscv10;
        LW2080_CTRL_PMGR_ILLUM_DEVICE_INFO_GPIO_PWM_RGBW_V10         gprgbwv10;
    } data;
} LW2080_CTRL_PMGR_ILLUM_DEVICE_INFO;
typedef struct LW2080_CTRL_PMGR_ILLUM_DEVICE_INFO *PLW2080_CTRL_PMGR_ILLUM_DEVICE_INFO;

/*!
 * Structure describing ILLUM_DEVICE static information/POR.
 * Implements the BOARDOBJGRP model/interface.
 */
#define LW2080_CTRL_PMGR_ILLUM_DEVICES_INFO_MESSAGE_ID (0x40U)

typedef struct LW2080_CTRL_PMGR_ILLUM_DEVICES_INFO {
    /*!
     * LW2080_CTRL_BOARDOBJGRP_E32 super class.
     * Must always be first object in structure.
     */
#endif // (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1) && (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
    LW2080_CTRL_BOARDOBJGRP_E32        super;
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1) && (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
    /*!
     * Array of ILLUM_DEVICE structures. Has valid indexes corresponding to
     * the bits set in @ref super.objMask.
     */
    LW2080_CTRL_PMGR_ILLUM_DEVICE_INFO devices[LW2080_CTRL_BOARDOBJGRP_E32_MAX_OBJECTS];
} LW2080_CTRL_PMGR_ILLUM_DEVICES_INFO;
typedef struct LW2080_CTRL_PMGR_ILLUM_DEVICES_INFO *PLW2080_CTRL_PMGR_ILLUM_DEVICES_INFO;

/*!
 * LW2080_CTRL_CMD_PMGR_ILLUM_DEVICES_GET_INFO
 *
 * This command returns ILLUM_DEVICES static object information/POR as
 * specified by the VBIOS in ILLUM_DEVICE Table.
 *
 * See @ref LW2080_CTRL_PMGR_ILLUM_DEVICE_INFO for documentation on
 * the parameters.
 *
 * Return values are specified per @ref BoardObjGrpGetInfo.
 */
#define LW2080_CTRL_CMD_PMGR_ILLUM_DEVICES_GET_INFO (0x2080a640) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PMGR_LEGACY_NON_PRIVILEGED_INTERFACE_ID << 8) | LW2080_CTRL_PMGR_ILLUM_DEVICES_INFO_MESSAGE_ID" */

/*!
 * Structure representing the data required for synchronization.
 */
typedef struct LW2080_CTRL_PMGR_ILLUM_DEVICE_SYNC {
    /*!
     * Boolean representing the need for synchronization.
     */
    LwBool bSync;

    /*!
     * Time stamp value required for synchronization.
     */
    LW_DECLARE_ALIGNED(LwU64 timeStampms, 8);
} LW2080_CTRL_PMGR_ILLUM_DEVICE_SYNC;
typedef struct LW2080_CTRL_PMGR_ILLUM_DEVICE_SYNC *PLW2080_CTRL_PMGR_ILLUM_DEVICE_SYNC;

/*!
 * Structure representing the device control parameters of each ILLUM_DEVICE.
 * Implements the BOARDOBJ model/interface.
 */
typedef struct LW2080_CTRL_PMGR_ILLUM_DEVICE_CONTROL {
    /*!
     * LW2080_CTRL_BOARDOBJ super class.
     * Must always be first object in structure.
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
     * Structure containing the synchronization data for the illumination device.
     */
    LW_DECLARE_ALIGNED(LW2080_CTRL_PMGR_ILLUM_DEVICE_SYNC syncData, 8);
} LW2080_CTRL_PMGR_ILLUM_DEVICE_CONTROL;
typedef struct LW2080_CTRL_PMGR_ILLUM_DEVICE_CONTROL *PLW2080_CTRL_PMGR_ILLUM_DEVICE_CONTROL;

/*!
 * Structure representing the control parameters of ILLUM_DEVICE-s.
 * Implements the BOARDOBJ model/interface.
 */
typedef struct LW2080_CTRL_PMGR_ILLUM_DEVICES_CONTROL {
    /*!
     * LW2080_CTRL_BOARDOBJGRP_E32 super class.
     * Must always be first object in structure.
     */
#endif // (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1) && (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
    LW2080_CTRL_BOARDOBJGRP_E32 super;
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1) && (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

    /*!
     * Array of ILLUM_DEVICE structures. Has valid indexes corresponding to
     * the bits set in @ref super.objMask.
     */
    LW_DECLARE_ALIGNED(LW2080_CTRL_PMGR_ILLUM_DEVICE_CONTROL devices[LW2080_CTRL_BOARDOBJGRP_E32_MAX_OBJECTS], 8);
} LW2080_CTRL_PMGR_ILLUM_DEVICES_CONTROL;
typedef struct LW2080_CTRL_PMGR_ILLUM_DEVICES_CONTROL *PLW2080_CTRL_PMGR_ILLUM_DEVICES_CONTROL;

/*!
 * LW2080_CTRL_CMD_PMGR_ILLUM_DEVICES_GET_CONTROL
 *
 * This command returns current ILLUM_DEVICES control parameters.
 *
 * See @ref LW2080_CTRL_PMGR_ILLUM_DEVICES_CONTROL for documentation on
 * the parameters.
 *
 * Return values are specified per @ref BoardObjGrpGetControl.
 */
#define LW2080_CTRL_CMD_PMGR_ILLUM_DEVICES_GET_CONTROL (0x2080a644) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PMGR_LEGACY_NON_PRIVILEGED_INTERFACE_ID << 8) | 0x44" */
/*!
 * LW2080_CTRL_CMD_PMGR_ILLUM_DEVICES_GET_CONTROL
 *
 * This command returns current ILLUM_DEVICES control parameters.
 *
 * See @ref LW2080_CTRL_PMGR_ILLUM_DEVICES_CONTROL for documentation on
 * the parameters.
 *
 * Return values are specified per @ref BoardObjGrpSetControl.
 */
#define LW2080_CTRL_CMD_PMGR_ILLUM_DEVICES_SET_CONTROL (0x2080a645) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PMGR_LEGACY_NON_PRIVILEGED_INTERFACE_ID << 8) | 0x45" */

/* ------------------------------ ILLUM_ZONE-s ------------------------------ */
/*!
 * RGB parameters required to represent manual control mode.
 */
typedef struct LW2080_CTRL_PMGR_ILLUM_ZONE_CTRL_MODE_MANUAL_RGB_PARAMS {
    /*!
     * Red component of the color applied to the zone.
     */
    LwU8 colorR;

    /*!
     * Green component of the color applied to the zone.
     */
    LwU8 colorG;

    /*!
     * Blue component of the color applied to the zone.
     */
    LwU8 colorB;

    /*!
     * Brightness percentage value of the zone.
     */
    LwU8 brightnessPct;
} LW2080_CTRL_PMGR_ILLUM_ZONE_CTRL_MODE_MANUAL_RGB_PARAMS;
typedef struct LW2080_CTRL_PMGR_ILLUM_ZONE_CTRL_MODE_MANUAL_RGB_PARAMS *PLW2080_CTRL_PMGR_ILLUM_ZONE_CTRL_MODE_MANUAL_RGB_PARAMS;

/*!
 * RGBW parameters required to represent manual control mode.
 */
typedef struct LW2080_CTRL_PMGR_ILLUM_ZONE_CTRL_MODE_MANUAL_RGBW_PARAMS {
    /*!
     * Red component of the color applied to the zone.
     */
    LwU8 colorR;

    /*!
     * Green component of the color applied to the zone.
     */
    LwU8 colorG;

    /*!
     * Blue component of the color applied to the zone.
     */
    LwU8 colorB;

    /*!
     * White component of the color applied to the zone.
     */
    LwU8 colorW;


    /*!
     * Brightness percentage value of the zone.
     */
    LwU8 brightnessPct;
} LW2080_CTRL_PMGR_ILLUM_ZONE_CTRL_MODE_MANUAL_RGBW_PARAMS;
typedef struct LW2080_CTRL_PMGR_ILLUM_ZONE_CTRL_MODE_MANUAL_RGBW_PARAMS *PLW2080_CTRL_PMGR_ILLUM_ZONE_CTRL_MODE_MANUAL_RGBW_PARAMS;

/*!
 * SINGLE_COLOR parameters required to represent manual control mode.
 */
typedef struct LW2080_CTRL_PMGR_ILLUM_ZONE_CTRL_MODE_MANUAL_SINGLE_COLOR_PARAMS {
    /*!
     * Brightness percentage value of the zone.
     */
    LwU8 brightnessPct;
} LW2080_CTRL_PMGR_ILLUM_ZONE_CTRL_MODE_MANUAL_SINGLE_COLOR_PARAMS;
typedef struct LW2080_CTRL_PMGR_ILLUM_ZONE_CTRL_MODE_MANUAL_SINGLE_COLOR_PARAMS *PLW2080_CTRL_PMGR_ILLUM_ZONE_CTRL_MODE_MANUAL_SINGLE_COLOR_PARAMS;

/*!
 * Data required to represent manual control mode.
 */
typedef struct LW2080_CTRL_PMGR_ILLUM_ZONE_CTRL_MODE_MANUAL_RGB {
    /*!
     * RGB parameters required to represent manual control mode.
     */
    LW2080_CTRL_PMGR_ILLUM_ZONE_CTRL_MODE_MANUAL_RGB_PARAMS rgbParams;
} LW2080_CTRL_PMGR_ILLUM_ZONE_CTRL_MODE_MANUAL_RGB;
typedef struct LW2080_CTRL_PMGR_ILLUM_ZONE_CTRL_MODE_MANUAL_RGB *PLW2080_CTRL_PMGR_ILLUM_ZONE_CTRL_MODE_MANUAL_RGB;

/*!
 * Data required to represent manual control mode.
 */
typedef struct LW2080_CTRL_PMGR_ILLUM_ZONE_CTRL_MODE_MANUAL_RGBW {
    /*!
     * RGB parameters required to represent manual control mode.
     */
    LW2080_CTRL_PMGR_ILLUM_ZONE_CTRL_MODE_MANUAL_RGBW_PARAMS rgbwParams;
} LW2080_CTRL_PMGR_ILLUM_ZONE_CTRL_MODE_MANUAL_RGBW;
typedef struct LW2080_CTRL_PMGR_ILLUM_ZONE_CTRL_MODE_MANUAL_RGBW *PLW2080_CTRL_PMGR_ILLUM_ZONE_CTRL_MODE_MANUAL_RGBW;

/*!
 * Data required to represent manual control mode.
 */
typedef struct LW2080_CTRL_PMGR_ILLUM_ZONE_CTRL_MODE_MANUAL_SINGLE_COLOR {
    /*!
     * SINGLE_COLOR parameters required to represent manual control mode.
     */
    LW2080_CTRL_PMGR_ILLUM_ZONE_CTRL_MODE_MANUAL_SINGLE_COLOR_PARAMS singleColorParams;
} LW2080_CTRL_PMGR_ILLUM_ZONE_CTRL_MODE_MANUAL_SINGLE_COLOR;
typedef struct LW2080_CTRL_PMGR_ILLUM_ZONE_CTRL_MODE_MANUAL_SINGLE_COLOR *PLW2080_CTRL_PMGR_ILLUM_ZONE_CTRL_MODE_MANUAL_SINGLE_COLOR;

/*!
 * Data required to represent piecewise linear control mode.
 */
typedef struct LW2080_CTRL_PMGR_ILLUM_ZONE_CTRL_MODE_PIECEWISE_LINEAR_RGB {
    /*!
     * RGB parameters required to represent manual control mode.
     */
    LW2080_CTRL_PMGR_ILLUM_ZONE_CTRL_MODE_MANUAL_RGB_PARAMS rgbParams[LW2080_CTRL_ILLUM_CTRL_MODE_PIECEWISE_LINEAR_COLOR_ENDPOINTS];

    /*!
     * Type of cycle effect to apply among
     * @ref LW2080_CTRL_ILLUM_CTRL_MODE_PIECEWISE_LINEAR_CYCLE_<XYZ>
     */
    LwU8                                                    cycleType;

    /*!
     * Number of times to repeat function within group period.
     */
    LwU8                                                    grpCount;

    /*!
     * Time in ms to transition from color A to color B.
     */
    LwU16                                                   riseTimems;

    /*!
     * Time in ms to transition from color B to color A.
     */
    LwU16                                                   fallTimems;

    /*!
     * Time in ms to remain at color A before color A to color B transition.
     */
    LwU16                                                   ATimems;

    /*!
     * Time in ms to remain at color B before color B to color A transition.
     */
    LwU16                                                   BTimems;

    /*!
     * Time in ms to remain idle before next group of repeated function cycles.
     */
    LwU16                                                   grpIdleTimems;

    /*!
     * Time in ms to offset the cycle relative to other zones.
     */
    LwU16                                                   phaseOffsetms;
} LW2080_CTRL_PMGR_ILLUM_ZONE_CTRL_MODE_PIECEWISE_LINEAR_RGB;
typedef struct LW2080_CTRL_PMGR_ILLUM_ZONE_CTRL_MODE_PIECEWISE_LINEAR_RGB *PLW2080_CTRL_PMGR_ILLUM_ZONE_CTRL_MODE_PIECEWISE_LINEAR_RGB;

/*!
 * Data required to represent piecewise linear control mode.
 */
typedef struct LW2080_CTRL_PMGR_ILLUM_ZONE_CTRL_MODE_PIECEWISE_LINEAR_RGBW {
    /*!
     * RGB parameters required to represent manual control mode.
     */
    LW2080_CTRL_PMGR_ILLUM_ZONE_CTRL_MODE_MANUAL_RGBW_PARAMS rgbwParams[LW2080_CTRL_ILLUM_CTRL_MODE_PIECEWISE_LINEAR_COLOR_ENDPOINTS];

    /*!
     * Type of cycle effect to apply among
     * @ref LW2080_CTRL_ILLUM_CTRL_MODE_PIECEWISE_LINEAR_CYCLE_<XYZ>
     */
    LwU8                                                     cycleType;

    /*!
     * Number of times to repeat function within group period.
     */
    LwU8                                                     grpCount;

    /*!
     * Time in ms to transition from color A to color B.
     */
    LwU16                                                    riseTimems;

    /*!
     * Time in ms to transition from color B to color A.
     */
    LwU16                                                    fallTimems;

    /*!
     * Time in ms to remain at color A before color A to color B transition.
     */
    LwU16                                                    ATimems;

    /*!
     * Time in ms to remain at color B before color B to color A transition.
     */
    LwU16                                                    BTimems;

    /*!
     * Time in ms to remain idle before next group of repeated function cycles.
     */
    LwU16                                                    grpIdleTimems;

    /*!
     * Time in ms to offset the cycle relative to other zones.
     */
    LwU16                                                    phaseOffsetms;
} LW2080_CTRL_PMGR_ILLUM_ZONE_CTRL_MODE_PIECEWISE_LINEAR_RGBW;
typedef struct LW2080_CTRL_PMGR_ILLUM_ZONE_CTRL_MODE_PIECEWISE_LINEAR_RGBW *PLW2080_CTRL_PMGR_ILLUM_ZONE_CTRL_MODE_PIECEWISE_LINEAR_RGBW;

/*!
 * Data required to represent piecewise linear control mode.
 */
typedef struct LW2080_CTRL_PMGR_ILLUM_ZONE_CTRL_MODE_PIECEWISE_LINEAR_SINGLE_COLOR {
    /*!
     * SINGLE_COLOR parameters required to represent manual control mode.
     */
    LW2080_CTRL_PMGR_ILLUM_ZONE_CTRL_MODE_MANUAL_SINGLE_COLOR_PARAMS singleColorParams[LW2080_CTRL_ILLUM_CTRL_MODE_PIECEWISE_LINEAR_COLOR_ENDPOINTS];

    /*!
     * Type of cycle effect to apply among
     * @ref LW2080_CTRL_ILLUM_CTRL_MODE_PIECEWISE_LINEAR_CYCLE_<XYZ>
     */
    LwU8                                                             cycleType;

    /*!
     * Number of times to repeat function within group period.
     */
    LwU8                                                             grpCount;

    /*!
     * Time in ms to transition from color A to color B.
     */
    LwU16                                                            riseTimems;

    /*!
     * Time in ms to transition from color B to color A.
     */
    LwU16                                                            fallTimems;

    /*!
     * Time in ms to remain at color A before color A to color B transition.
     */
    LwU16                                                            ATimems;

    /*!
     * Time in ms to remain at color B before color B to color A transition.
     */
    LwU16                                                            BTimems;

    /*!
     * Time in ms to remain idle before next group of repeated function cycles.
     */
    LwU16                                                            grpIdleTimems;

    /*!
     * Time in ms to offset the cycle relative to other zones.
     */
    LwU16                                                            phaseOffsetms;
} LW2080_CTRL_PMGR_ILLUM_ZONE_CTRL_MODE_PIECEWISE_LINEAR_SINGLE_COLOR;
typedef struct LW2080_CTRL_PMGR_ILLUM_ZONE_CTRL_MODE_PIECEWISE_LINEAR_SINGLE_COLOR *PLW2080_CTRL_PMGR_ILLUM_ZONE_CTRL_MODE_PIECEWISE_LINEAR_SINGLE_COLOR;

/*!
 * Union containing all control modes.
 */
typedef union LW2080_CTRL_PMGR_ILLUM_ZONE_CTRL_MODE_RGB {
    LW2080_CTRL_PMGR_ILLUM_ZONE_CTRL_MODE_MANUAL_RGB           manualRGB;
    LW2080_CTRL_PMGR_ILLUM_ZONE_CTRL_MODE_PIECEWISE_LINEAR_RGB piecewiseLinearRGB;
} LW2080_CTRL_PMGR_ILLUM_ZONE_CTRL_MODE_RGB;

typedef union LW2080_CTRL_PMGR_ILLUM_ZONE_CTRL_MODE_RGB *PLW2080_CTRL_PMGR_ILLUM_ZONE_CTRL_MODE_RGB;

/*!
 * Union containing all control modes.
 */
typedef union LW2080_CTRL_PMGR_ILLUM_ZONE_CTRL_MODE_RGBW {
    LW2080_CTRL_PMGR_ILLUM_ZONE_CTRL_MODE_MANUAL_RGBW           manualRGBW;
    LW2080_CTRL_PMGR_ILLUM_ZONE_CTRL_MODE_PIECEWISE_LINEAR_RGBW piecewiseLinearRGBW;
} LW2080_CTRL_PMGR_ILLUM_ZONE_CTRL_MODE_RGBW;

typedef union LW2080_CTRL_PMGR_ILLUM_ZONE_CTRL_MODE_RGBW *PLW2080_CTRL_PMGR_ILLUM_ZONE_CTRL_MODE_RGBW;

/*!
 * Union containing all control modes.
 */
typedef union LW2080_CTRL_PMGR_ILLUM_ZONE_CTRL_MODE_SINGLE_COLOR {
    LW2080_CTRL_PMGR_ILLUM_ZONE_CTRL_MODE_MANUAL_SINGLE_COLOR           manualSingleColor;
    LW2080_CTRL_PMGR_ILLUM_ZONE_CTRL_MODE_PIECEWISE_LINEAR_SINGLE_COLOR piecewiseLinearSingleColor;
} LW2080_CTRL_PMGR_ILLUM_ZONE_CTRL_MODE_SINGLE_COLOR;

typedef union LW2080_CTRL_PMGR_ILLUM_ZONE_CTRL_MODE_SINGLE_COLOR *PLW2080_CTRL_PMGR_ILLUM_ZONE_CTRL_MODE_SINGLE_COLOR;

/*!
 * Macros for ILLUM_ZONE types
 */
#define LW2080_CTRL_PMGR_ILLUM_ZONE_TYPE_ILWALID      0x00
#define LW2080_CTRL_PMGR_ILLUM_ZONE_TYPE_RGB          0x01
#define LW2080_CTRL_PMGR_ILLUM_ZONE_TYPE_RGB_FIXED    0x02
#define LW2080_CTRL_PMGR_ILLUM_ZONE_TYPE_RGBW         0x03
#define LW2080_CTRL_PMGR_ILLUM_ZONE_TYPE_SINGLE_COLOR 0x04

/*!
 * Structure describing ILLUM_ZONE_RGB static information/POR.
 */
typedef struct LW2080_CTRL_PMGR_ILLUM_ZONE_INFO_RGB {
    /*!
     * Lwrrently we do NOT have any static info parameter in this class.
     */
    LwU8 rsvd;
} LW2080_CTRL_PMGR_ILLUM_ZONE_INFO_RGB;
typedef struct LW2080_CTRL_PMGR_ILLUM_ZONE_INFO_RGB *PLW2080_CTRL_PMGR_ILLUM_ZONE_INFO_RGB;

/*!
 * Structure describing ILLUM_ZONE_RGB static information/POR.
 */
typedef struct LW2080_CTRL_PMGR_ILLUM_ZONE_INFO_RGBW {
    /*!
     * Lwrrently we do NOT have any static info parameter in this class.
     */
    LwU8 rsvd;
} LW2080_CTRL_PMGR_ILLUM_ZONE_INFO_RGBW;
typedef struct LW2080_CTRL_PMGR_ILLUM_ZONE_INFO_RGBW *PLW2080_CTRL_PMGR_ILLUM_ZONE_INFO_RGBW;

/*!
 * Structure describing ILLUM_ZONE_RGB static information/POR.
 */
typedef struct LW2080_CTRL_PMGR_ILLUM_ZONE_INFO_SINGLE_COLOR {
    /*!
     * Lwrrently we do NOT have any static info parameter in this class.
     */
    LwU8 rsvd;
} LW2080_CTRL_PMGR_ILLUM_ZONE_INFO_SINGLE_COLOR;
typedef struct LW2080_CTRL_PMGR_ILLUM_ZONE_INFO_SINGLE_COLOR *PLW2080_CTRL_PMGR_ILLUM_ZONE_INFO_SINGLE_COLOR;
/*!
 * ILLUM_ZONE type-specific data union. Discriminated by
 * ILLUM_ZONE::super.type.
 */


/*!
 * Structure describing ILLUM_ZONE static information/POR.
 * Implements the BOARDOBJ model/interface.
 */
typedef struct LW2080_CTRL_PMGR_ILLUM_ZONE_INFO {
    /*!
     * LW2080_CTRL_BOARDOBJ super class.
     * Must always be first object in structure.
     */
#endif // (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1) && (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
    LW2080_CTRL_BOARDOBJ            super;
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1) && (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

    /*!
     * XAPI does not support using an element within another structure as a
     * discriminant. Placing this redundant type value here until that design
     * constraint can be fixed.
     */
    LwU8                            type;

    /*!
     * Index pointing to an Illumination Device in Illumination Device Table.
     */
    LwU8                            illumDeviceIdx;

    /*!
     * Provider index for representing logical to physical zone mapping.
     */
    LwU8                            provIdx;

    /*!
     * Location of the zone on the board.
     */
#endif // (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1) && (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
    LW2080_CTRL_ILLUM_ZONE_LOCATION zoneLocation;
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1) && (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

    /*!
     * Type-specific data union.
     */
    union {
        LW2080_CTRL_PMGR_ILLUM_ZONE_INFO_RGB          rgb;
        LW2080_CTRL_PMGR_ILLUM_ZONE_INFO_RGB          rgbw;
        LW2080_CTRL_PMGR_ILLUM_ZONE_INFO_SINGLE_COLOR singleColor;
    } data;
} LW2080_CTRL_PMGR_ILLUM_ZONE_INFO;
typedef struct LW2080_CTRL_PMGR_ILLUM_ZONE_INFO *PLW2080_CTRL_PMGR_ILLUM_ZONE_INFO;

/*!
 * Structure describing ILLUM_ZONE static information/POR.
 * Implements the BOARDOBJGRP model/interface.
 */
#define LW2080_CTRL_PMGR_ILLUM_ZONES_INFO_MESSAGE_ID (0x41U)

typedef struct LW2080_CTRL_PMGR_ILLUM_ZONES_INFO {
    /*!
     * LW2080_CTRL_BOARDOBJGRP_E32 super class.
     * Must always be first object in structure.
     */
#endif // (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1) && (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
    LW2080_CTRL_BOARDOBJGRP_E32      super;
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1) && (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

    /*!
     * Filter period in ms below which all requests
     * to program zones will be rejected.
     */
    LwU8                             filterPeriodMs;

    /*!
     * Array of ILLUM_ZONE structures. Has valid indexes corresponding to
     * the bits set in @ref super.objMask.
     */
    LW2080_CTRL_PMGR_ILLUM_ZONE_INFO zones[LW2080_CTRL_BOARDOBJGRP_E32_MAX_OBJECTS];
} LW2080_CTRL_PMGR_ILLUM_ZONES_INFO;
typedef struct LW2080_CTRL_PMGR_ILLUM_ZONES_INFO *PLW2080_CTRL_PMGR_ILLUM_ZONES_INFO;

/*!
 * LW2080_CTRL_CMD_PMGR_ILLUM_ZONES_GET_INFO
 *
 * This command returns ILLUM_ZONES static object information/POR as
 * specified by the VBIOS in ILLUM_ZONE Table.
 *
 * See @ref LW2080_CTRL_PMGR_ILLUM_ZONE_INFO for documentation on
 * the parameters.
 *
 * Return values are specified per @ref BoardObjGrpGetInfo.
 */
#define LW2080_CTRL_CMD_PMGR_ILLUM_ZONES_GET_INFO (0x2080a641) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PMGR_LEGACY_NON_PRIVILEGED_INTERFACE_ID << 8) | LW2080_CTRL_PMGR_ILLUM_ZONES_INFO_MESSAGE_ID" */

/*!
 * Structure representing the control parameters of ILLUM_ZONE_RGB.
 */
typedef struct LW2080_CTRL_PMGR_ILLUM_ZONE_CONTROL_RGB {
    /*!
     * XAPI does not support using an element within another structure as a
     * discriminant. Placing this redundant type value here until that design
     * constraint can be fixed.
     */
    LwU8                                      ctrlMode;

    /*!
     * Type-specific data union.
     */
    LW2080_CTRL_PMGR_ILLUM_ZONE_CTRL_MODE_RGB data;
} LW2080_CTRL_PMGR_ILLUM_ZONE_CONTROL_RGB;
typedef struct LW2080_CTRL_PMGR_ILLUM_ZONE_CONTROL_RGB *PLW2080_CTRL_PMGR_ILLUM_ZONE_CONTROL_RGB;

/*!
 * Structure representing the control parameters of ILLUM_ZONE_RGBW.
 */
typedef struct LW2080_CTRL_PMGR_ILLUM_ZONE_CONTROL_RGBW {
    /*!
     * XAPI does not support using an element within another structure as a
     * discriminant. Placing this redundant type value here until that design
     * constraint can be fixed.
     */
    LwU8                                       ctrlMode;

    /*!
     * Type-specific data union.
     */
    LW2080_CTRL_PMGR_ILLUM_ZONE_CTRL_MODE_RGBW data;
} LW2080_CTRL_PMGR_ILLUM_ZONE_CONTROL_RGBW;
typedef struct LW2080_CTRL_PMGR_ILLUM_ZONE_CONTROL_RGBW *PLW2080_CTRL_PMGR_ILLUM_ZONE_CONTROL_RGBW;

/*!
 * Structure representing the control parameters of ILLUM_ZONE_SINGLE_COLOR.
 */
typedef struct LW2080_CTRL_PMGR_ILLUM_ZONE_CONTROL_SINGLE_COLOR {
    /*!
     * XAPI does not support using an element within another structure as a
     * discriminant. Placing this redundant type value here until that design
     * constraint can be fixed.
     */
    LwU8                                               ctrlMode;

    /*!
     * Type-specific data union.
     */
    LW2080_CTRL_PMGR_ILLUM_ZONE_CTRL_MODE_SINGLE_COLOR data;
} LW2080_CTRL_PMGR_ILLUM_ZONE_CONTROL_SINGLE_COLOR;
typedef struct LW2080_CTRL_PMGR_ILLUM_ZONE_CONTROL_SINGLE_COLOR *PLW2080_CTRL_PMGR_ILLUM_ZONE_CONTROL_SINGLE_COLOR;

/*!
 * ILLUM_ZONE type-specific data union. Discriminated by
 * ILLUM_ZONE::super.type.
 */


/*!
 * Structure representing the control parameters of each ILLUM_ZONE.
 * Implements the BOARDOBJ model/interface.
 */
typedef struct LW2080_CTRL_PMGR_ILLUM_ZONE_CONTROL {
    /*!
     * LW2080_CTRL_BOARDOBJ super class.
     * Must always be first object in structure.
     */
#endif // (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1) && (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
    LW2080_CTRL_BOARDOBJ super;
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1) && (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

    /*!
     * Control Mode for this zone.
     */
    LwU8                 ctrlMode;

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
        LW2080_CTRL_PMGR_ILLUM_ZONE_CONTROL_RGB          rgb;
        LW2080_CTRL_PMGR_ILLUM_ZONE_CONTROL_RGBW         rgbw;
        LW2080_CTRL_PMGR_ILLUM_ZONE_CONTROL_SINGLE_COLOR singleColor;
    } data;
} LW2080_CTRL_PMGR_ILLUM_ZONE_CONTROL;
typedef struct LW2080_CTRL_PMGR_ILLUM_ZONE_CONTROL *PLW2080_CTRL_PMGR_ILLUM_ZONE_CONTROL;

/*!
 * Structure representing the control parameters of ILLUM_ZONE-s.
 * Implements the BOARDOBJ model/interface.
 */
typedef struct LW2080_CTRL_PMGR_ILLUM_ZONES_CONTROL {
    /*!
     * LW2080_CTRL_BOARDOBJGRP_E32 super class.
     * Must always be first object in structure.
     */
#endif // (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1) && (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
    LW2080_CTRL_BOARDOBJGRP_E32         super;
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1) && (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

    /*!
     * [in] Flag specifying the set of values to retrieve:
     * - VBIOS default (LW_TRUE)
     * - lwrrently active (LW_FALSE)
     */
    LwBool                              bDefault;

    /*!
     * Array of ILLUM_ZONE structures. Has valid indexes corresponding to
     * the bits set in @ref super.objMask.
     */
    LW2080_CTRL_PMGR_ILLUM_ZONE_CONTROL zones[LW2080_CTRL_BOARDOBJGRP_E32_MAX_OBJECTS];
} LW2080_CTRL_PMGR_ILLUM_ZONES_CONTROL;
typedef struct LW2080_CTRL_PMGR_ILLUM_ZONES_CONTROL *PLW2080_CTRL_PMGR_ILLUM_ZONES_CONTROL;

/*!
 * LW2080_CTRL_CMD_PMGR_ILLUM_ZONES_GET_CONTROL
 *
 * This command returns current ILLUM_ZONES control parameters.
 *
 * See @ref LW2080_CTRL_PMGR_ILLUM_ZONES_CONTROL for documentation on
 * the parameters.
 *
 * Return values are specified per @ref BoardObjGrpGetControl.
 */
#define LW2080_CTRL_CMD_PMGR_ILLUM_ZONES_GET_CONTROL (0x2080a642) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PMGR_LEGACY_NON_PRIVILEGED_INTERFACE_ID << 8) | 0x42" */

/*!
 * LW2080_CTRL_CMD_PMGR_ILLUM_ZONES_SET_CONTROL
 *
 * This command accepts client-specified control parameters for a set of
 * ILLUM_ZONES and applies these new parameters.
 *
 * See @ref LW2080_CTRL_PMGR_ILLUM_ZONES_CONTROL for documentation on
 * the parameters.
 *
 * Return values are specified per @ref BoardObjGrpSetControl.
 */
#define LW2080_CTRL_CMD_PMGR_ILLUM_ZONES_SET_CONTROL (0x2080a643) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PMGR_LEGACY_NON_PRIVILEGED_INTERFACE_ID << 8) | 0x43" */

/* _ctrl2080illum_opaque_non_privileged_h_ */
#endif // (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

