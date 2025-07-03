/*
 * SPDX-FileCopyrightText: Copyright (c) 2010-2022 LWPU CORPORATION & AFFILIATES. All rights reserved.
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
// Source file: ctrl/ctrl2080/ctrl2080pmgr.finn
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
#include "ctrl/ctrl2080/ctrl2080gpumon.h"
#include "ctrl/ctrl2080/ctrl2080volt.h"
#include "ctrl/ctrl2080/ctrl2080illum.h"
#include "ctrl/ctrl2080/ctrl2080perf_cf_pwr_model.h"
#include "ctrl/ctrl2080/ctrl2080pmumon.h"

/* LW20_SUBDEVICE_XX pmgr-related control commands and parameters */

/*
 * LW2080_CTRL_CMD_PMGR_GET_GPU_IDENTIFICATION_LED_STATUS
 *
 * This command is used to query the current status of the GPU identication LED.
 *
 * lwrrentLedStatus
 *   This parameter returns the current state of LED.
 *   Possible values are:
 *       LW2080_CTRL_PMGR_GPU_IDENTIFICATION_LED_STATUS_ON
 *       LW2080_CTRL_PMGR_GPU_IDENTIFICATION_LED_STATUS_OFF
 *
 * Possible return status values
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 *   LW_ERR_ILWALID_STATE
 *
 */

#define LW2080_CTRL_CMD_PMGR_GET_GPU_IDENTIFICATION_LED_STATUS (0x20802601U) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PMGR_INTERFACE_ID << 8) | LW2080_CTRL_PMGR_GET_GPU_IDENTIFICATION_LED_STATUS_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_PMGR_GET_GPU_IDENTIFICATION_LED_STATUS_PARAMS_MESSAGE_ID (0x1U)

typedef struct LW2080_CTRL_PMGR_GET_GPU_IDENTIFICATION_LED_STATUS_PARAMS {
    LwU32 lwrrentLedStatus;
} LW2080_CTRL_PMGR_GET_GPU_IDENTIFICATION_LED_STATUS_PARAMS;

#define LW2080_CTRL_PMGR_GPU_IDENTIFICATION_LED_STATUS_ON      (0x00000001U)
#define LW2080_CTRL_PMGR_GPU_IDENTIFICATION_LED_STATUS_OFF     (0x00000000U)

/*
 * LW2080_CTRL_CMD_PMGR_SET_GPU_IDENTIFICATION_LED_STATUS
 *
 * This command is used turn on or off GPU identication LED.
 *
 * ledStatus
 *   This parameter indicates whether to turn on or off GPU LED.
 *   Possible values are:
 *       LW2080_CTRL_PMGR_GPU_IDENTIFICATION_LED_STATUS_ON
 *       LW2080_CTRL_PMGR_GPU_IDENTIFICATION_LED_STATUS_OFF
 *
 * Possible return status values
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 *   LW_ERR_ILWALID_STATE
 *
 */

#define LW2080_CTRL_CMD_PMGR_SET_GPU_IDENTIFICATION_LED_STATUS (0x20802602U) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PMGR_INTERFACE_ID << 8) | LW2080_CTRL_PMGR_SET_GPU_IDENTIFICATION_LED_STATUS_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_PMGR_SET_GPU_IDENTIFICATION_LED_STATUS_PARAMS_MESSAGE_ID (0x2U)

typedef struct LW2080_CTRL_PMGR_SET_GPU_IDENTIFICATION_LED_STATUS_PARAMS {
    LwU32 ledStatus;
} LW2080_CTRL_PMGR_SET_GPU_IDENTIFICATION_LED_STATUS_PARAMS;

/*!
 * Macros for PWR_DEVICE types.
 * Enumerations 0x01 - 0x1F are intended for BA device types.
 * Enumerations 0x20 - 0x2F are intended for GPU ADC device types.
 * Enumerations from 0x30 onwards correspond to @ref LW_DCB4X_I2C_DEVICE_TYPE
 * which are I2C devices.
 */
#define LW2080_CTRL_PMGR_PWR_DEVICE_TYPE_DISABLED                        0x00U
#define LW2080_CTRL_PMGR_PWR_DEVICE_TYPE_BA00__DEPRECATED_DO_NOT_REUSE   0x01U
#define LW2080_CTRL_PMGR_PWR_DEVICE_TYPE_BA10HW                          0x02U
#define LW2080_CTRL_PMGR_PWR_DEVICE_TYPE_BA10SW__DEPRECATED_DO_NOT_REUSE 0x03U
#define LW2080_CTRL_PMGR_PWR_DEVICE_TYPE_BA11HW                          0x04U
#define LW2080_CTRL_PMGR_PWR_DEVICE_TYPE_BA11SW__DEPRECATED_DO_NOT_REUSE 0x05U
#define LW2080_CTRL_PMGR_PWR_DEVICE_TYPE_BA12HW                          0x06U
#define LW2080_CTRL_PMGR_PWR_DEVICE_TYPE_BA13HW                          0x07U
#define LW2080_CTRL_PMGR_PWR_DEVICE_TYPE_BA14HW                          0x08U
#define LW2080_CTRL_PMGR_PWR_DEVICE_TYPE_BA15HW                          0x09U
#define LW2080_CTRL_PMGR_PWR_DEVICE_TYPE_BA16HW                          0x0AU
#define LW2080_CTRL_PMGR_PWR_DEVICE_TYPE_BA20                            0x0BU
#define LW2080_CTRL_PMGR_PWR_DEVICE_TYPE_GPUADC1X                        0x20U
#define LW2080_CTRL_PMGR_PWR_DEVICE_TYPE_GPUADC10                        0x21U
#define LW2080_CTRL_PMGR_PWR_DEVICE_TYPE_GPUADC11                        0x22U
// 0x23 reserved for GPUADC12 which will be added in AD10x
#define LW2080_CTRL_PMGR_PWR_DEVICE_TYPE_GPUADC13                        0x24U
// NJ-TODO: Do not expose ADS1112 over LwAPI (RM in the process of its removal)
#define LW2080_CTRL_PMGR_PWR_DEVICE_TYPE_ADS1112                         0x30U
#define LW2080_CTRL_PMGR_PWR_DEVICE_TYPE_NCT3933U                        0x4BU
#define LW2080_CTRL_PMGR_PWR_DEVICE_TYPE_INA219                          0x4LW
#define LW2080_CTRL_PMGR_PWR_DEVICE_TYPE_INA209__DEPRECATED_DO_NOT_REUSE 0x4DU
#define LW2080_CTRL_PMGR_PWR_DEVICE_TYPE_INA3221                         0x4EU

/*!
 * Special value corresponding to an invalid Power Device index.  This value
 * means that device is not specified.
 */
#define LW2080_CTRL_PMGR_PWR_DEVICE_INDEX_ILWALID                        LW2080_CTRL_BOARDOBJ_IDX_ILWALID_8BIT

/*!
 * Macros for channel number of power device. This macro must sync with
 * @ref RM_PMU_PMGR_PWR_DEVICE_INA3221_CH_NUM
 */
#define LW2080_CTRL_PMGR_PWR_DEVICE_INA3221_CH_NUM                       0x03U

/*!
 * Macro for number of registers in the power device. Each register stores the
 * limit to be enforced by that DAC. Number of registers is same as the number
 * of DACs. This must sync with @ref RM_PMU_PMGR_PWR_DEVICE_NCT3933U_PROV_NUM
 */
#define LW2080_CTRL_PMGR_PWR_DEVICE_NCT3933U_REGISTERS_NUM               0x03U

/*!
 * Macro for number of threshold limits supported by power device. This macro
 * must sync with @ref RM_PMU_PMGR_PWR_DEVICE_NCT3933U_THRESHOLD_NUM
 */
#define LW2080_CTRL_PMGR_PWR_DEVICE_NCT3933U_THRESHOLD_NUM               0x01U

/*!
 * Macros for the PWR_DEVICE power rails.
 *
 * ~~~SPTODO~~~: Remove these enums, now found in PWR_CHANNEL.
 */
#define LW2080_CTRL_PMGR_PWR_DEVICE_POWER_RAIL_DISABLED                  (0x00000000U)
#define LW2080_CTRL_PMGR_PWR_DEVICE_POWER_RAIL_OUTPUT_LWVDD              (0x00000001U)
#define LW2080_CTRL_PMGR_PWR_DEVICE_POWER_RAIL_OUTPUT_FBVDD              (0x00000002U)
#define LW2080_CTRL_PMGR_PWR_DEVICE_POWER_RAIL_OUTPUT_FBVDDQ             (0x00000003U)
#define LW2080_CTRL_PMGR_PWR_DEVICE_POWER_RAIL_OUTPUT_FBVDD_Q            (0x00000004U)
#define LW2080_CTRL_PMGR_PWR_DEVICE_POWER_RAIL_OUTPUT_PEXVDD             (0x00000005U)
#define LW2080_CTRL_PMGR_PWR_DEVICE_POWER_RAIL_OUTPUT_A3V3               (0x00000006U)
#define LW2080_CTRL_PMGR_PWR_DEVICE_POWER_RAIL_INPUT_EXT12V_8PIN0        (0x000000FAU)
#define LW2080_CTRL_PMGR_PWR_DEVICE_POWER_RAIL_INPUT_EXT12V_8PIN1        (0x000000FBU)
#define LW2080_CTRL_PMGR_PWR_DEVICE_POWER_RAIL_INPUT_EXT12V_6PIN0        (0x000000FLW)
#define LW2080_CTRL_PMGR_PWR_DEVICE_POWER_RAIL_INPUT_EXT12V_6PIN1        (0x000000FDU)
#define LW2080_CTRL_PMGR_PWR_DEVICE_POWER_RAIL_INPUT_PEX3V3              (0x000000FEU)
#define LW2080_CTRL_PMGR_PWR_DEVICE_POWER_RAIL_INPUT_PEX12V              (0x000000FFU)

/*!
 * Structure containing a union {LwU16, LwUFXP8_8} for rShuntmOhm.
 */
typedef struct LW2080_CTRL_PMGR_PWR_DEVICE_INFO_RSHUNT {
    /*!
     * Boolean indicating if this structure is representing an unsigned
     * int16 value or an unsigned FXP8_8 value.
     */
    LwBool bUseFXP8_8;
    /*!
     * The rShunt union.
     */
    union {
        /*!
         * Interpreted as unsigned int16.
         */
        LwU16 u16;
        /*!
         * Interpreted as unsigned FXP8_8.
         * Should actually use LwUFXP8_8, however XAPI_GEN does not take
         * LwUFXP8_8. Since LwUFXP8_8 is actually LwU16, use LwU16 to WAR
         * XAPI's restriction here.
         */
        LwU16 ufxp8_8;
    } value;
} LW2080_CTRL_PMGR_PWR_DEVICE_INFO_RSHUNT;

/*!
 * Maximum possible value of the ADC raw output after correction
 */
#define LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC10_ADC_MAX_VALUE         0x3FU

/*!
 * Define number of physical providers for GPUADC10.
 * Physical providers have the capability to return power/current/voltage value
 */
#define LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC10_PHYSICAL_PROV_NUM     0x06U

/*!
 * Define total number of providers for GPUADC10. Includes both physical and
 * logical providers. Logical providers in GPUADC10 cannot return power/current
 * voltage values. These are used only as source for IPC.
 */
#define LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC10_PROV_NUM              0x09U

/*!
 * Define number of IPC instances supported by GPUADC10.
 */
#define LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC10_IPC_NUM               0x04U

/*!
 * Define the start value of provider index when using BA as input to IPC
 */
#define LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC10_BA_IPC_PROV_IDX_START LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC10_PHYSICAL_PROV_NUM

/*!
 * Enumeration of the IPC source types for GPUADC10. This is an 8-bit field.
 */
#define LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC10_IPC_SRC_TYPE_NONE     0x00U
#define LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC10_IPC_SRC_TYPE_ADC      0x01U
#define LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC10_IPC_SRC_TYPE_BA       0x02U

/*!
 * Enumeration of the GPUADC10 provider units. This is an 8-bit field.
 */
#define LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC10_PROV_UNIT_MV          0x00U
#define LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC10_PROV_UNIT_MA          0x01U
#define LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC10_PROV_UNIT_MW          0x02U
#define LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC10_PROV_UNIT_ILWALID     0xFFU

/*!
 * Structure holding the configuration parameters of each GPUADC10 IPC instance
 */
typedef struct LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC10_IPC_INSTANCE {
    /*!
     * IPC source @ref LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC10_IPC_SRC_TYPE_<XYZ>
     */
    LwU8 sourceType;
    /*!
     * Union of provider index and window index
     */
    union {
        /*!
         * Provider index if IPC source is ADC
         */
        LwU8 provider;
        /*!
         * Window index if IPC source is BA
         */
        LwU8 window;
    } index;
    /*!
     * Unit (mW, mV or mA)
     */
    LwU8 providerUnit;
    /*!
     * IIR_GAIN factor for IPC
     */
    LwU8 iirGain;
    /*!
     * IIR_LENGTH factor for IPC
     */
    LwU8 iirLength;
    /*!
     * IIR_DOWNSHIFT factor for IPC
     */
    LwU8 iirDownshift;
    /*!
     * PROPORTIONAL DOWNSHIFT factor for IPC
     */
    LwU8 propDownshift;
} LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC10_IPC_INSTANCE;

/*!
 * Structure holding the configuration parameters of all GPUADC10 IPC instances
 */
typedef struct LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC10_IPC {
    /*!
     * Per instance IPC configuration parameters
     */
    LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC10_IPC_INSTANCE instance[LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC10_IPC_NUM];
    /*!
     * Offset added to cap current IIR output, common to all IPC instances
     */
    LwU16                                             compMaxOffset;
} LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC10_IPC;

/*!
 * Structure holding the parameters for the GPUADC10 beacon feature
 */
typedef struct LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC10_BEACON {
    /*!
     * Boolean to indicate enabled/disabled for the GPUADC beacon feature
     */
    LwBool                           bEnable;
    /*!
     * Mask of GPUADC providers which will be used as beacon
     */
    LwU8                             provIdxMask;
    /*!
     * Threshold value (specified in ADC code) above which the GPUADC will be
     * considered working correctly
     */
    LwU8                             threshold;
    /*!
     * Max spread value (specified in ADC code) within which the min and max of
     * the GPUADC providers should lie for correct operation
     */
    LwU8                             maxSpread;
    /*!
     * Channel mask corresponding to the provIdxMask above. This is a mask of
     * channels pointing to the providers in the provIdxMask.
     */
    LW2080_CTRL_BOARDOBJGRP_MASK_E32 channelMask;
} LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC10_BEACON;

/*!
 * Maximum possible value of the ADC raw output after correction
 */
#define LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC11_ADC_MAX_VALUE          0x7FU

/*!
 * Define number of physical providers for GPUADC11.
 * Physical providers have the capability to return power/current/voltage value.
 * This is the maximum number of providers possible and not all may necessarily
 * be enabled. It will depend on OVR_M gen and other ADC configuration.
 */
#define LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC11_PHYSICAL_PROV_NUM      0x09U

/*!
 * Define total number of providers for GPUADC11. Includes both physical and
 * logical providers. Logical providers in GPUADC11 cannot return power/current
 * voltage values. These are used only as source for IPC.
 */
#define LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC11_PROV_NUM               0x0EU

/*!
 * Define number of thresholds for GPUADC11.
 * This is always equal to @ref LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC11_IPC_NUM
 */
#define LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC11_THRESHOLD_NUM          0x04U

/*!
 * Define number of OVRM devices supported by GPUADC11.
 */
#define LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC11_OVRM_NUM               0x02U

/*!
 * Define number of IPC instances supported by GPUADC11.
 */
#define LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC11_IPC_NUM                0x04U

/*!
 * Define number of beacon instances supported by GPUADC11.
 */
#define LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC11_BEACON_NUM             0x02U

/*!
 * Define number of offset instances supported by GPUADC11.
 */
#define LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC11_OFFSET_NUM             0x02U

/*!
 * Define number of sum instances supported by GPUADC11.
 */
#define LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC11_SUM_1                  0x00U
#define LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC11_SUM_2                  0x01U
#define LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC11_SUM_NUM                0x02U

/*!
 * Define number of inputs to each sum block supported by GPUADC11.
 */
#define LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC11_SUM_INPUT_NUM          0x04U

/*!
 * Define the start value of provider index when using BA as input to IPC
 */
#define LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC11_BA_IPC_PROV_IDX_START  (0xbU) /* finn: Evaluated from "(LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC11_PHYSICAL_PROV_NUM + LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC11_SUM_NUM)" */

/*!
 * Enumeration of the IPC source types for GPUADC11. This is an 8-bit field.
 */
#define LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC11_IPC_SRC_TYPE_NONE      0x00U
#define LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC11_IPC_SRC_TYPE_ADC       0x01U
#define LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC11_IPC_SRC_TYPE_BA        0x02U
#define LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC11_IPC_SRC_TYPE_ADC_SUM   0x03U

/*!
 * Enumeration of the GPUADC11 IPC provider units. This is an 8-bit field.
 */
#define LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC11_IPC_PROV_UNIT_MV       0x00U
#define LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC11_IPC_PROV_UNIT_MA       0x01U
#define LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC11_IPC_PROV_UNIT_MW       0x02U
#define LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC11_IPC_PROV_UNIT_ILWALID  0xFFU

/*!
 * Enumeration of the GPUADC11 SUM provider units. This is an 8-bit field.
 */
#define LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC11_SUM_PROV_UNIT_DISABLED 0x00U
#define LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC11_SUM_PROV_UNIT_MV       0x01U
#define LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC11_SUM_PROV_UNIT_MA       0x02U
#define LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC11_SUM_PROV_UNIT_MW       0x03U

/*!
 * Enumeration of the GPUADC11 operating modes. This is a 1-bit field.
 */
#define LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC11_OPERATING_MODE_NON_ALT 0x00U
#define LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC11_OPERATING_MODE_ALT     0x01U

/*!
 * Enumeration of the GPUADC11 beacon comparison modes. This is a 1-bit field.
 */
#define LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC11_BEACON_COMP_GT         0x00U
#define LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC11_BEACON_COMP_LT         0x01U

/*!
 * Enumeration of the OVR-M device generation. This is a 3-bit field.
 */
#define LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC11_OVRM_GEN_1             0x00U
#define LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC11_OVRM_GEN_2             0x01U
#define LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC11_OVRM_GEN_ILWALID       0xFFU

/*!
 * Define value for no provider index for GPUADC11
 */
#define LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC11_PROV_IDX_NONE          0xFFU

/*!
 * Structure holding the data specific to OVRM Gen2 per device
 */
typedef struct LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC11_OVRM_DATA_GEN2 {
    /*!
     * Boolean representing if ground reference is to be used for this device
     */
    LwBool bGround;
    /*!
     * Number of providers that will give power tuple values
     */
    LwU8   tupleProvNum;
    /*!
     * Number of IMON channels configured on this device
     */
    LwU8   imonNum;
} LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC11_OVRM_DATA_GEN2;

/*!
 * Union of OVRM gen-specific data.
 */


/*!
 * Structure holding the configuration parameters of OVRM per device
 */
typedef struct LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC11_OVRM_DEVICE {
    /*!
     * Boolean representing if this OVR-M device is enabled.
     */
    LwBool bEnabled;
    /*!
     * OVR-M device generation
     * @ref LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC11_OVRM_GEN_<XYZ>
     * When @ref bEnabled is LW_FALSE, this gen field is expected to be
     * @ref LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC11_OVRM_GEN_ILWALID.
     */
    LwU8   gen;
    /*!
     * Type-specific information.
     */
    union {
        LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC11_OVRM_DATA_GEN2 gen2;
    } data;
} LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC11_OVRM_DEVICE;

/*!
 * Structure holding the configuration parameters of all OVRM devices
 */
typedef struct LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC11_OVRM {
    /*!
     * Per device OVRM configuration parameters
     */
    LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC11_OVRM_DEVICE device[LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC11_OVRM_NUM];
} LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC11_OVRM;

/*!
 * Structure holding the configuration parameters of each GPUADC11 IPC instance
 */
typedef struct LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC11_IPC_INSTANCE {
    /*!
     * IPC source @ref LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC10_IPC_SRC_TYPE_<XYZ>
     */
    LwU8 sourceType;
    /*!
     * Union of provider index, window index and sum instance
     */
    union {
        /*!
         * Provider index if IPC source is ADC
         */
        LwU8 provider;
        /*!
         * Window index if IPC source is BA
         */
        LwU8 window;
        /*!
         * Sum instance if IPC source is ADC SUM
         */
        LwU8 sum;
    } index;
    /*!
     * Unit (mW, mV or mA)
     */
    LwU8 providerUnit;
} LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC11_IPC_INSTANCE;

/*!
 * Structure holding the configuration parameters of all GPUADC11 IPC instances
 */
typedef struct LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC11_IPC {
    /*!
     * Per instance IPC configuration parameters
     */
    LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC11_IPC_INSTANCE instance[LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC11_IPC_NUM];
} LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC11_IPC;

/*!
 * Structure holding the parameters for each GPUADC11 beacon instance
 */
typedef struct LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC11_BEACON_INSTANCE {
    /*!
     * Provider index which will be used as beacon
     * provIdx is @ref LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC11_PROV_IDX_NONE
     * for disabling a beacon.
     */
    LwU8 provIdx;
    /*!
     * Threshold value (specified in ADC code) above which the GPUADC will be
     * considered working correctly
     */
    LwU8 threshold;
    /*!
     * Comparison function (< or >) to be used for comparing against threshold
     */
    LwU8 compFunc;
} LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC11_BEACON_INSTANCE;

/*!
 * Structure holding the parameters for the GPUADC11 beacon feature
 */
typedef struct LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC11_BEACON {
    /*!
     * Per instance beacon configuration parameters
     */
    LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC11_BEACON_INSTANCE instance[LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC11_BEACON_NUM];
} LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC11_BEACON;

/*!
 * Structure holding the parameters for each GPUADC11 offset instance
 */
typedef struct LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC11_OFFSET_INSTANCE {
    /*!
     * Provider index which will be used as offset
     * provIdx is @ref LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC11_PROV_IDX_NONE
     * for disabling an offset.
     */
    LwU8  provIdx;
    /*!
     * Mask of providers which will use this offset
     */
    LwU16 provMask;
} LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC11_OFFSET_INSTANCE;

/*!
 * Structure holding the parameters for the GPUADC11 offset feature
 */
typedef struct LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC11_OFFSET {
    /*!
     * Per instance offset configuration parameters
     */
    LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC11_OFFSET_INSTANCE instance[LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC11_OFFSET_NUM];
} LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC11_OFFSET;

/*!
 * Structure containing parameters for inputs to sum block
 */
typedef struct LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC11_SUM_INPUT {
    /*!
     * Provider index of the input
     */
    LwU8 provIdx;
    /*!
     * Scaling factor or weight to be applied to this input
     */
    LwU8 scaleFactor;
} LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC11_SUM_INPUT;

/*!
 * Structure holding the parameters for each GPUADC11 sum instance
 */
typedef struct LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC11_SUM_INSTANCE {
    /*!
     * Unit (mW, mV or mA)
     * provUnit is @ref LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC11_SUM_PROV_UNIT_DISABLED
     * for disabling a SUM instance.
     */
    LwU8 provUnit;
    /*!
     * Union of lwrrColwFactor or pwrColwFactor based on provUnit
     */
    union {
        /*!
         * Colwersion factor which represents the current value in mA of a single
         * bit of SUM output (mA/code)
         * Should actually use LwUFXP28_4, however XAPI_GEN does not take
         * LwUFXP28_4. Since LwUFXP28_4 is actually LwU32, use LwU32 to WAR
         * XAPI's restriction.
         */
        LwU32 lwrr;
        /*!
         * Colwersion factor which represents the power value in mW of a single
         * bit of SUM output (mW/code)
         * Should actually use LwUFXP28_4, however XAPI_GEN does not take
         * LwUFXP28_4. Since LwUFXP28_4 is actually LwU32, use LwU32 to WAR
         * XAPI's restriction.
         */
        LwU32 pwr;
    } colwFactor;
    /*!
     * Per input parameters of sum block
     */
    LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC11_SUM_INPUT input[LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC11_SUM_INPUT_NUM];
} LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC11_SUM_INSTANCE;

/*!
 * Structure holding the parameters for the GPUADC11 sum blocks
 */
typedef struct LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC11_SUM {
    /*!
     * Per instance offset configuration parameters
     */
    LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC11_SUM_INSTANCE instance[LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC11_SUM_NUM];
} LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC11_SUM;

/*!
 * Maximum possible value of the ADC raw output after correction
 */
#define LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC13_ADC_MAX_VALUE          0x7FU

/*!
 * Define number of physical providers for GPUADC13.
 * Physical providers have the capability to return power/current/voltage value.
 * This is the maximum number of providers possible and not all may necessarily
 * be enabled. It will depend on OVR_M gen and other ADC configuration.
 */
#define LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC13_PHYSICAL_PROV_NUM      0x0EU

/*!
 * Define total number of providers for GPUADC13. Includes both physical and
 * logical providers. Logical providers in GPUADC13 cannot return power/current
 * voltage values. These are used only as source for IPC.
 */
#define LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC13_PROV_NUM               0x13U

/*!
 * Define number of thresholds for GPUADC13.
 * This is always equal to @ref LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC13_IPC_NUM
 */
#define LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC13_THRESHOLD_NUM          0x04U

/*!
 * Define number of OVRM devices supported by GPUADC13.
 */
#define LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC13_OVRM_NUM               0x02U

/*!
 * Define number of IPC instances supported by GPUADC13.
 */
#define LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC13_IPC_NUM                0x04U

/*!
 * Define number of beacon instances supported by GPUADC13.
 */
#define LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC13_BEACON_NUM             0x02U

/*!
 * Define number of offset instances supported by GPUADC13.
 */
#define LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC13_OFFSET_NUM             0x02U

/*!
 * Define number of sum instances supported by GPUADC13.
 */
#define LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC13_SUM_1                  0x00U
#define LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC13_SUM_2                  0x01U
#define LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC13_SUM_NUM                0x02U

/*!
 * Define number of inputs to each sum block supported by GPUADC13.
 */
#define LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC13_SUM_INPUT_NUM          0x04U

/*!
 * Define the start value of provider index when using BA as input to IPC
 */
#define LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC13_BA_IPC_PROV_IDX_START  (0x10U) /* finn: Evaluated from "(LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC13_PHYSICAL_PROV_NUM + LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC13_SUM_NUM)" */

/*!
 * Enumeration of the IPC source types for GPUADC13. This is an 8-bit field.
 */
#define LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC13_IPC_SRC_TYPE_NONE      0x00U
#define LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC13_IPC_SRC_TYPE_ADC       0x01U
#define LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC13_IPC_SRC_TYPE_BA        0x02U
#define LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC13_IPC_SRC_TYPE_ADC_SUM   0x03U

/*!
 * Enumeration of the GPUADC13 IPC provider units. This is an 8-bit field.
 */
#define LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC13_IPC_PROV_UNIT_MV       0x00U
#define LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC13_IPC_PROV_UNIT_MA       0x01U
#define LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC13_IPC_PROV_UNIT_MW       0x02U
#define LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC13_IPC_PROV_UNIT_ILWALID  0xFFU

/*!
 * Enumeration of the GPUADC13 SUM provider units. This is an 8-bit field.
 */
#define LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC13_SUM_PROV_UNIT_DISABLED 0x00U
#define LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC13_SUM_PROV_UNIT_MV       0x01U
#define LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC13_SUM_PROV_UNIT_MA       0x02U
#define LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC13_SUM_PROV_UNIT_MW       0x03U

/*!
 * Enumeration of the GPUADC13 operating modes. This is a 1-bit field.
 */
#define LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC13_OPERATING_MODE_NON_ALT 0x00U
#define LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC13_OPERATING_MODE_ALT     0x01U

/*!
 * Enumeration of the GPUADC13 beacon comparison modes. This is a 1-bit field.
 */
#define LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC13_BEACON_COMP_GT         0x00U
#define LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC13_BEACON_COMP_LT         0x01U

/*!
 * Enumeration of the OVR-M device generation. This is a 3-bit field.
 */
#define LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC13_OVRM_GEN_1             0x00U
#define LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC13_OVRM_GEN_2             0x01U
#define LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC13_OVRM_GEN_ILWALID       0xFFU

/*!
 * Enumeration of ADC PWM operating modes. This is a 1-bit field.
 */
#define LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC13_ADC_PWM_MODE_PRI       0x0U
#define LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC13_ADC_PWM_MODE_SEC       0x1U

/*!
 * Define value for no provider index for GPUADC13
 */
#define LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC13_PROV_IDX_NONE          0xFFU

/*!
 * Structure holding the data specific to IMON for an OVRM Gen2 device
 */
typedef struct LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC13_OVRM_GEN2_IMON {
    /*!
     * Boolean indicating if the IMON channel pair is enabled
     */
    LwBool bEnabled;
    /*!
     * Voltage (V) corresponding to ADC_MAX_MALUE for IMON (FXP8.8).
     * Should actually use LwUFXP8_8, however XAPI_GEN does not take LwUFXP8_8.
     * Since LwUFXP8_8 is actually LwU16, use LwU16 to WAR XAPI's restriction.
     */
    LwU16  fullRangeVoltage;
} LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC13_OVRM_GEN2_IMON;

/*!
 * Structure holding the data specific to OVRM Gen2 per device
 */
typedef struct LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC13_OVRM_DATA_GEN2 {
    /*!
     * Boolean representing if ground reference is to be used for this device
     */
    LwBool                                              bGround;
    /*!
     * Number of providers that will give power tuple values
     */
    LwU8                                                tupleProvNum;
    /*!
     * IMON specific data for this OVRM Gen2 device
     */
    LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC13_OVRM_GEN2_IMON imon;
} LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC13_OVRM_DATA_GEN2;

/*!
 * Union of OVRM gen-specific data.
 */


/*!
 * Structure holding the configuration parameters of OVRM per device
 */
typedef struct LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC13_OVRM_DEVICE {
    /*!
     * Boolean representing if this OVR-M device is enabled.
     */
    LwBool bEnabled;
    /*!
     * OVR-M device generation
     * @ref LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC13_OVRM_GEN_<XYZ>
     * When @ref bEnabled is LW_FALSE, this gen field is expected to be
     * @ref LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC13_OVRM_GEN_ILWALID.
     */
    LwU8   gen;
    /*!
     * Type-specific information.
     */
    union {
        LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC13_OVRM_DATA_GEN2 gen2;
    } data;
} LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC13_OVRM_DEVICE;

/*!
 * Structure holding the configuration parameters of all OVRM devices
 */
typedef struct LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC13_OVRM {
    /*!
     * Per device OVRM configuration parameters
     */
    LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC13_OVRM_DEVICE device[LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC13_OVRM_NUM];
} LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC13_OVRM;

/*!
 * Structure holding the configuration parameters of each GPUADC13 IPC instance
 */
typedef struct LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC13_IPC_INSTANCE {
    /*!
     * IPC source @ref LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC10_IPC_SRC_TYPE_<XYZ>
     */
    LwU8 sourceType;
    /*!
     * Union of provider index, window index and sum instance
     */
    union {
        /*!
         * Provider index if IPC source is ADC
         */
        LwU8 provider;
        /*!
         * Window index if IPC source is BA
         */
        LwU8 window;
        /*!
         * Sum instance if IPC source is ADC SUM
         */
        LwU8 sum;
    } index;
    /*!
     * Unit (mW, mV or mA)
     */
    LwU8          providerUnit;
    /*!
     * Index into Voltage Rail table for PWMVID
     */
    LwBoardObjIdx voltRailIdx;
    /*!
     * Floor value in A/W for IPC_REF
     */
    LwU16         floor;
    /*!
     * Ceiling value in A/W for IPC_REF
     */
    LwU16         ceil;
} LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC13_IPC_INSTANCE;

/*!
 * Structure holding the configuration parameters of all GPUADC13 IPC instances
 */
typedef struct LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC13_IPC {
    /*!
     * Per instance IPC configuration parameters
     */
    LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC13_IPC_INSTANCE instance[LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC13_IPC_NUM];
} LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC13_IPC;

/*!
 * Structure holding the parameters for each GPUADC13 beacon instance
 */
typedef struct LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC13_BEACON_INSTANCE {
    /*!
     * Provider index which will be used as beacon
     * provIdx is @ref LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC13_PROV_IDX_NONE
     * for disabling a beacon.
     */
    LwU8 provIdx;
    /*!
     * Threshold value (specified in ADC code) above which the GPUADC will be
     * considered working correctly
     */
    LwU8 threshold;
    /*!
     * Comparison function (< or >) to be used for comparing against threshold
     */
    LwU8 compFunc;
} LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC13_BEACON_INSTANCE;

/*!
 * Structure holding the parameters for the GPUADC13 beacon feature
 */
typedef struct LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC13_BEACON {
    /*!
     * Per instance beacon configuration parameters
     */
    LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC13_BEACON_INSTANCE instance[LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC13_BEACON_NUM];
} LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC13_BEACON;

/*!
 * Structure holding the parameters for each GPUADC13 offset instance
 */
typedef struct LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC13_OFFSET_INSTANCE {
    /*!
     * Provider index which will be used as offset
     * provIdx is @ref LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC13_PROV_IDX_NONE
     * for disabling an offset.
     */
    LwU8  provIdx;
    /*!
     * Mask of providers which will use this offset
     */
    LwU16 provMask;
} LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC13_OFFSET_INSTANCE;

/*!
 * Structure holding the parameters for the GPUADC13 offset feature
 */
typedef struct LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC13_OFFSET {
    /*!
     * Per instance offset configuration parameters
     */
    LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC13_OFFSET_INSTANCE instance[LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC13_OFFSET_NUM];
} LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC13_OFFSET;

/*!
 * Structure containing parameters for inputs to sum block
 */
typedef struct LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC13_SUM_INPUT {
    /*!
     * Provider index of the input
     */
    LwU8 provIdx;
    /*!
     * Scaling factor or weight to be applied to this input
     */
    LwU8 scaleFactor;
} LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC13_SUM_INPUT;

/*!
 * Structure holding the parameters for each GPUADC13 sum instance
 */
typedef struct LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC13_SUM_INSTANCE {
    /*!
     * Unit (mW, mV or mA)
     * provUnit is @ref LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC13_SUM_PROV_UNIT_DISABLED
     * for disabling a SUM instance.
     */
    LwU8 provUnit;
    /*!
     * Union of lwrrColwFactor or pwrColwFactor based on provUnit
     */
    union {
        /*!
         * Colwersion factor which represents the current value in mA of a single
         * bit of SUM output (mA/code)
         * Should actually use LwUFXP28_4, however XAPI_GEN does not take
         * LwUFXP28_4. Since LwUFXP28_4 is actually LwU32, use LwU32 to WAR
         * XAPI's restriction.
         */
        LwU32 lwrr;
        /*!
         * Colwersion factor which represents the power value in mW of a single
         * bit of SUM output (mW/code)
         * Should actually use LwUFXP28_4, however XAPI_GEN does not take
         * LwUFXP28_4. Since LwUFXP28_4 is actually LwU32, use LwU32 to WAR
         * XAPI's restriction.
         */
        LwU32 pwr;
    } colwFactor;
    /*!
     * Per input parameters of sum block
     */
    LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC13_SUM_INPUT input[LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC13_SUM_INPUT_NUM];
} LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC13_SUM_INSTANCE;

/*!
 * Structure holding the parameters for the GPUADC13 sum blocks
 */
typedef struct LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC13_SUM {
    /*!
     * Per instance offset configuration parameters
     */
    LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC13_SUM_INSTANCE instance[LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC13_SUM_NUM];
} LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC13_SUM;

/*!
 * Structure holding the data specific to ADC PWM in primary mode
 */
typedef struct LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC13_ADC_PWM_PRI {
    /*!
     * Positive portion of PWM for the last channel in units of utilsclk cycles
     * (used for synchronization)
     */
    LwU16 hiLast;
} LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC13_ADC_PWM_PRI;

/*!
 * Structure holding the data specific to ADC PWM in secondary mode
 */
typedef struct LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC13_ADC_PWM_SEC {
    /*!
     * Threshold for detection of the synchronization signal in utilsclk cycles
     */
    LwU16 syncThreshold;
    /*!
     * Threshold for detection of OVRM reset commands in microseconds
     */
    LwU16 resetThreshold;
} LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC13_ADC_PWM_SEC;

/*!
 * Union of ADC PWM mode-specific data.
 */


/*!
 * Structure holding the configuration parameters of ADC PWM
 */
typedef struct LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC13_ADC_PWM {
    /*!
     * Operating mode of ADC PWM, Primary vs Secondary
     * @ref LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC13_ADC_PWM_MODE_<XYZ>
     */
    LwU8 mode;
    /*!
     * Mode-specific data.
     */
    union {
        LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC13_ADC_PWM_PRI pri;
        LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC13_ADC_PWM_SEC sec;
    } data;
} LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC13_ADC_PWM;

/*!
 * Structure of static information specific to the BA1XHW power device.  This
 * Power Device is GPU's internal HW logic designed for current/power estimation
 * on GK11X (BA10HW) and GK20X (BA11HW).
 */
typedef struct LW2080_CTRL_PMGR_PWR_DEVICE_INFO_DATA_BA1XHW {
    /*!
     * If set device monitors/estimates GPU current [mA], otherwise power [mW]
     */
    LwBool bLwrrent;
    /*!
     * If set include BA figures from silicon powerd with LWVDD
     */
    LwBool bMonitorLWVDD;
    /*!
     * If set include BA figures from silicon powerd with FBVDDQ
     */
    LwBool bMonitorFBVDDQ;
    /*!
     * Index into Power Equation Table (PWR_EQUATION) for the scaling equation
     */
    LwU8   scalingEquIdx;
    /*!
     * Index into Power Equation Table (PWR_EQUATION) for the offset equation
     *
     * NJ-TODO: At present moment RM refers to this field as leakageEquIdx and
     *          that will change once we get better equation for "C".
     */
    LwU8   offsetEquIdx;
    /*!
     * Index of associated HW BA averaging window
     */
    LwU8   windowIdx;
    /*!
     * Number of bits to right-shift within Step Period Integrator HW
     */
    LwU8   sumShift;
    /*!
     * Size of BA averaging window expressed as log2(utils clocks)
     */
    LwU8   winPeriod;
} LW2080_CTRL_PMGR_PWR_DEVICE_INFO_DATA_BA1XHW;

/*!
 * Structure of static information specific to the BA15HW power device.
 */
typedef struct LW2080_CTRL_PMGR_PWR_DEVICE_INFO_DATA_BA15HW {
    LW2080_CTRL_PMGR_PWR_DEVICE_INFO_DATA_BA1XHW super;
    /*!
     * Index into Power Sensor Table pointing to GPUADC device
     */
    LwU8                                         gpuAdcDevIdx;
} LW2080_CTRL_PMGR_PWR_DEVICE_INFO_DATA_BA15HW;

/*!
 * VBIOS to RMCTRL defines for BA16
 */
#define LW2080_CTRL_PMGR_PWR_DEVICE_BA_16HW_LUT_ADC_SEL_LWVDD_SYS_ADC     0x0U
#define LW2080_CTRL_PMGR_PWR_DEVICE_BA_16HW_LUT_ADC_SEL_LWVDD_GPC_ADC_MIN 0x1U
#define LW2080_CTRL_PMGR_PWR_DEVICE_BA_16HW_LUT_ADC_SEL_LWVDD_GPC_ADC_AVG 0x2U
#define LW2080_CTRL_PMGR_PWR_DEVICE_BA_16HW_LUT_ADC_SEL_MSVDD_SYS_ADC     0x0U
#define LW2080_CTRL_PMGR_PWR_DEVICE_BA_16HW_LUT_ADC_SEL_MSVDD_GPC_ADC_MIN 0x1U
#define LW2080_CTRL_PMGR_PWR_DEVICE_BA_16HW_LUT_ADC_SEL_MSVDD_GPC_ADC_AVG 0x2U
#define LW2080_CTRL_PMGR_PWR_DEVICE_BA_16HW_POWER_DOMAIN_GPC_LWVDD        0x0U
#define LW2080_CTRL_PMGR_PWR_DEVICE_BA_16HW_POWER_DOMAIN_GPC_MSVDD        0x1U
#define LW2080_CTRL_PMGR_PWR_DEVICE_BA_16HW_POWER_DOMAIN_FBP_LWVDD        0x0U
#define LW2080_CTRL_PMGR_PWR_DEVICE_BA_16HW_POWER_DOMAIN_FBP_MSVDD        0x1U
#define LW2080_CTRL_PMGR_PWR_DEVICE_BA_16HW_POWER_DOMAIN_XBAR_LWVDD       0x0U
#define LW2080_CTRL_PMGR_PWR_DEVICE_BA_16HW_POWER_DOMAIN_XBAR_MSVDD       0x1U

/*!
 * Structure of static information specific to the BA16HW power device.
 */
typedef struct LW2080_CTRL_PMGR_PWR_DEVICE_INFO_DATA_BA16HW {
    LW2080_CTRL_PMGR_PWR_DEVICE_INFO_DATA_BA1XHW super;
    /*!
     * Index into PowerSensorTable pointing to GPUADC device.
     */
    LwU8                                         gpuAdcDevIdx;
     /*!
     * If set include BA figures from silicon powerd with LWVDD
     */
    LwBool                                       bMonitorLW;
    /*!
     * If set include BA figures from silicon powerd with FBVDDQ
     */
    LwBool                                       bMonitorMS;
    /*!
     * Booleans to enable HW realtime FACTOR_A/LEAKAGE_C control
     */
    LwBool                                       bFactorALutEnableLW;
    LwBool                                       bLeakageCLutEnableLW;
    /*!
     * Defines T1 (BA Threshold 1) mode as ADC or Normal
     */
    LwBool                                       bIsT1ModeADC;
    /*!
     * Defines T2 (BA Threshold 2) mode as ADC or Normal
     */
    LwBool                                       bIsT2ModeADC;
    /*!
     * Window_period = winSize * 2^stepSize (utilsclk)
     * winSize gives Window Size to callwlate window period.
     */
    LwU8                                         winSize;
    /*!
     * Window_period = winSize * 2^stepSize (utilsclk)
     * stepSize gives Step Size (log 2) to callwlate window period.
     */
    LwU8                                         stepSize;
    /*!
     * HW realtime FACTOR_A/LEAKAGE_C controls for MSVDD rail
     */
    LwBool                                       bFactorALutEnableMS;
    LwBool                                       bLeakageCLutEnableMS;
    /*!
     * Boolean to enable BA operation in FBVDD Mode.
     */
    LwBool                                       bFBVDDMode;
    /*!
     * Index into Power Equation Table (PWR_EQUATION) for the leakage equation for LWVDD
     */
    LwU8                                         leakageEquIdxMS;
    /*!
     * Index into Power Equation Table (PWR_EQUATION) for the scaling equation for MSVDD
     */
    LwU8                                         scalingEquIdxMS;
    /*!
     * Selects SYS/GPC ADC for LW/MS LUT A/C. When choose GPC_ADC_MIN and GPC_ADC_AVG,
     * HW will choose the minimum and average of all valid GPC ADC voltage respectively
     * Values can be @ref LW_THERM_PEAKPOWER_CONFIG1_LUT_ADC_SEL_LWVDD_XYZ
     */
    LwU8                                         adcSelLW;
    LwU8                                         adcSelMS;
    /*!
     * Specifies whether the BA from GPC is on LWVDD or MSVDD.
     */
    LwU8                                         pwrDomainGPC;
    /*!
     * Specifies whether the BA from XBAR is on LWVDD or MSVDD.
     */
    LwU8                                         pwrDomainXBAR;
    /*!
     * Specifies whether the BA from FBP is on LWVDD or MSVDD.
     */
    LwU8                                         pwrDomainFBP;
} LW2080_CTRL_PMGR_PWR_DEVICE_INFO_DATA_BA16HW;

/*!
 * VBIOS to RMCTRL defines for BA20
 */
#define LW2080_CTRL_PMGR_PWR_DEVICE_BA_20_LUT_ADC_SEL_SYS_ADC     0x0U
#define LW2080_CTRL_PMGR_PWR_DEVICE_BA_20_LUT_ADC_SEL_GPC_ADC_MIN 0x1U
#define LW2080_CTRL_PMGR_PWR_DEVICE_BA_20_LUT_ADC_SEL_GPC_ADC_AVG 0x2U
#define LW2080_CTRL_PMGR_PWR_DEVICE_BA_20_POWER_DOMAIN_GPC_LWVDD  0x0U
#define LW2080_CTRL_PMGR_PWR_DEVICE_BA_20_POWER_DOMAIN_GPC_MSVDD  0x1U
#define LW2080_CTRL_PMGR_PWR_DEVICE_BA_20_POWER_DOMAIN_FBP_LWVDD  0x0U
#define LW2080_CTRL_PMGR_PWR_DEVICE_BA_20_POWER_DOMAIN_FBP_MSVDD  0x1U
#define LW2080_CTRL_PMGR_PWR_DEVICE_BA_20_POWER_DOMAIN_XBAR_LWVDD 0x0U
#define LW2080_CTRL_PMGR_PWR_DEVICE_BA_20_POWER_DOMAIN_XBAR_MSVDD 0x1U
#define LW2080_CTRL_PMGR_PWR_DEVICE_BA_20_BA_SRC_DISABLED         0x0U
#define LW2080_CTRL_PMGR_PWR_DEVICE_BA_20_BA_SRC_ENABLED          0x1U
#define LW2080_CTRL_PMGR_PWR_DEVICE_BA_20_FACTOR_A_LUT_DISABLE    0x0U
#define LW2080_CTRL_PMGR_PWR_DEVICE_BA_20_FACTOR_A_LUT_ENABLE     0x1U
#define LW2080_CTRL_PMGR_PWR_DEVICE_BA_20_LEAKAGE_C_LUT_DISABLE   0x0U
#define LW2080_CTRL_PMGR_PWR_DEVICE_BA_20_LEAKAGE_C_LUT_ENABLE    0x1U

/*!
 * Define value for maximum number of voltage rails that may be supported by
 * each instance of BA20 power device.
 */
#define LW2080_CTRL_PMGR_PWR_DEVICE_BA_20_MAX_CONFIGS             0x2U

/*!
 * Structure of static information specific to the BA20 power device which
 * depends based on the voltage rail which the BA20 instance is monitoring.
 */
typedef struct LW2080_CTRL_PMGR_PWR_DEVICE_BA20_VOLT_RAIL_DATA {
    /*!
     * Index into Voltage Rail Table corresponding to the voltage rail which
     * the current Ba instance is monitoring.
     */
    LwU8   voltRailIdx;
    /*!
     * Booleans to enable HW realtime FACTOR_A/LEAKAGE_C control
     */
    LwBool bFactorALutEnable;
    LwBool bLeakageCLutEnable;
    /*!
     * Selects SYS/GPC ADC for LUT A/C. When choose GPC_ADC_MIN and GPC_ADC_AVG,
     * HW will choose the minimum and average of all valid GPC ADC voltage respectively
     * Values can be @ref LW_THERM_PEAKPOWER_CONFIG1_LUT_ADC_SEL_LWVDD_XYZ
     */
    LwU8   adcSel;
} LW2080_CTRL_PMGR_PWR_DEVICE_BA20_VOLT_RAIL_DATA;

/*!
 * Invalid Tuple Voltage value when the BA instance is monitoring multiple
 * rails. Lwrrently, we do not have visibility into the combined voltage value
 * when 1 BA instance is monitoring multiple rails (LWVDD + MSVDD). Hence, in
 * this scenario, the tuple voltage is set to this invalid value.
 */
#define LW2080_CTRL_PMGR_PWR_DEVICE_BA_20_TUPLE_VOLTAGE_ILWALID (LW_U32_MAX)

/*!
 * Invalid Tuple Current value when the BA instance is monitoring multiple
 * rails. If 1 BA instance is monitoring multiple rails (LWVDD + MSVDD), the
 * tuple voltage won't be meaningful and hence it will be set to an invalid
 * value as explained above. If BA is measuring power, the current reading
 * will not be meaningful in this scenario and hence will be set to this
 * invalid value.
 */
#define LW2080_CTRL_PMGR_PWR_DEVICE_BA_20_TUPLE_LWRRENT_ILWALID (LW_U32_MAX)

/*!
 * Invalid Tuple Power value when the BA instance is monitoring multiple
 * rails. If 1 BA instance is monitoring multiple rails (LWVDD + MSVDD), the
 * tuple voltage won't be meaningful and hence it will be set to an invalid
 * value as explained above. If BA is measuring current, the power reading
 * will not be meaningful in this scenario and hence will be set to this
 * invalid value.
 */
#define LW2080_CTRL_PMGR_PWR_DEVICE_BA_20_TUPLE_POWER_ILWALID   (LW_U32_MAX)

/*!
 * Number of thresholds supported by BA v2.x PWR_DEVICE.
 *
 * THRESHOLD_1H
 * THRESHOLD_2H
 * THRESHOLD_1L
 * THRESHOLD_2L
 */
#define LW2080_CTRL_PMGR_PWR_DEVICE_BA_2X_THRESHOLD_NUM         0x04

/*!
 * Providers supported by BA v2.x PWR_DEVICE:
 *
 * TOTAL          - Total Power / Total Current based on whether BA is
 *                  Power mode or Current mode.
 *                  Total Power = Dynamic Power + Leakage Power.
 *                  Total Current = Dynamic Current = Leakage Current.
 * DYNAMIC        - Dynamic Power / Dynamic Current Only
 * NUM_PROVS      - Maximum number of supported providers
 */
#define LW2080_CTRL_PMGR_PWR_DEVICE_BA_2X_PROV_TOTAL            0x00U
#define LW2080_CTRL_PMGR_PWR_DEVICE_BA_2X_PROV_DYNAMIC          0x01U
#define LW2080_CTRL_PMGR_PWR_DEVICE_BA_2X_PROV_NUM              (0x2U) /* finn: Evaluated from "(LW2080_CTRL_PMGR_PWR_DEVICE_BA_2X_PROV_DYNAMIC + 1)" */

/*!
 * Structure of static information specific to the BA20 power device.
 */
typedef struct LW2080_CTRL_PMGR_PWR_DEVICE_INFO_DATA_BA20 {
    /*!
     * If set device monitors/estimates GPU current [mA], otherwise power [mW]
     */
    LwBool                                          bLwrrent;
    /*!
     * Index of associated HW BA averaging window
     */
    LwU8                                            windowIdx;
    /*!
     * Index into PowerSensorTable pointing to GPUADC device.
     */
    LwU8                                            gpuAdcDevIdx;
    /*!
     * Defines T1 (BA Threshold 1) mode as ADC or Normal
     */
    LwBool                                          bIsT1ModeADC;
    /*!
     * Defines T2 (BA Threshold 2) mode as ADC or Normal
     */
    LwBool                                          bIsT2ModeADC;
    /*!
     * Boolean to enable BA operation in FBVDD Mode.
     */
    LwBool                                          bFBVDDMode;
    /*!
     * Window_period = winSize * 2^stepSize (utilsclk)
     * winSize gives Window Size to callwlate window period.
     */
    LwU8                                            winSize;
    /*!
     * Window_period = winSize * 2^stepSize (utilsclk)
     * stepSize gives Step Size (log 2) to callwlate window period.
     */
    LwU8                                            stepSize;
    /*!
     * Specifies whether the BA from GPC is on LWVDD or MSVDD.
     */
    LwU8                                            pwrDomainGPC;
    /*!
     * Specifies whether the BA from XBAR is on LWVDD or MSVDD.
     */
    LwU8                                            pwrDomainXBAR;
    /*!
     * Specifies whether the BA from FBP is on LWVDD or MSVDD.
     */
    LwU8                                            pwrDomainFBP;
    /*!
     * The scaling factor for Factor A/C (in both SW and HW modes), and also
     * for LW_CPWR_THERM_PEAKPOWER_CONFIG10_WIN_SUM_VALUE register output, but
     * in the reverse direction as compared to scaling of Factor A/C.
     */
    LwUFXP4_12                                      scaleFactor;
    /*!
     * Array of static information specific to the BA20 power device which
     * depends based on the voltage rail which the BA20 instance is monitoring.
     * Each entry in the array corresponds to a voltage rail that the BA20
     * instance may monitor.
     */
    LW2080_CTRL_PMGR_PWR_DEVICE_BA20_VOLT_RAIL_DATA voltRailData[LW2080_CTRL_PMGR_PWR_DEVICE_BA_20_MAX_CONFIGS];
} LW2080_CTRL_PMGR_PWR_DEVICE_INFO_DATA_BA20;

/*!
 * Structure of static information specific to the INA219 power device.  This
 * Power Device is TI's INA219 ADC which can monitor power, current, and voltage
 * of a single provider.
 */
typedef struct LW2080_CTRL_PMGR_PWR_DEVICE_INFO_DATA_INA219 {
    /*!
     * Shunt Resitor Resistance (mOhm)
     * ccs-TODO: Redirect LWAPI to use rShunt from old rShuntmOhm
     */
    LW2080_CTRL_PMGR_PWR_DEVICE_INFO_RSHUNT rShunt;
    LwU16                                   rShuntmOhm;
    /*!
     * Configuration register value
     */
    LwU16                                   calibration;
    /*!
     * Calibration register value
     */
    LwU16                                   configuration;
} LW2080_CTRL_PMGR_PWR_DEVICE_INFO_DATA_INA219;

/*!
 * Structure of static information specific to the INA3221 power device.  This
 * Power Device is TI's INA3221 ADC which can monitor power, current, and
 * voltage of 3 providers.
 */
typedef struct LW2080_CTRL_PMGR_PWR_DEVICE_INFO_DATA_INA3221 {
    /*!
     * Shunt Resitor Resistance (mOhm)
     * ccs-TODO: Redirect LWAPI to use rShunt from old rShuntmOhm
     */
    LW2080_CTRL_PMGR_PWR_DEVICE_INFO_RSHUNT rShunt[LW2080_CTRL_PMGR_PWR_DEVICE_INA3221_CH_NUM];
    LwU16                                   rShuntmOhm[LW2080_CTRL_PMGR_PWR_DEVICE_INA3221_CH_NUM];
    /*!
     * Configuration register value
     */
    LwU16                                   configuration;
    /*!
     * Mask/Enable register value
     */
    LwU16                                   maskEnable;
    /*!
     * The GPIO function this device could trigger.
     */
    LwU8                                    gpioFunction;
    /*!
     * Current value linear correction factor M. There is a bias offset on
     * INA3221's current measurement. We are using y=Mx+B to adjust
     * measured data.
     */
    LwUFXP4_12                              lwrrCorrectM;
    /*!
     * Current value linear correction factor B. Units of Amps.
     */
    LwSFXP4_12                              lwrrCorrectB;
    /*!
     * Power Sensor Device I2C index
     * This is used to Index into the DCB - I2C table
     */
    LwU8                                    i2cDevIdx;
} LW2080_CTRL_PMGR_PWR_DEVICE_INFO_DATA_INA3221;

/*!
 * Structure of static information specific to the NCT3933U power device.
 */
typedef struct LW2080_CTRL_PMGR_PWR_DEVICE_INFO_DATA_NCT3933U {
    /*!
     * Default limit of the sensor (mA)
     */
    LwU32 defaultLimitmA;
    /*!
     * Value of register CR04h in NCT3933U
     */
    LwU8  reg04val;
    /*!
     * Value of register CR05h in NCT3933U
     */
    LwU8  reg05val;
    /*!
     * Step size in mA of one single step of circuit
     */
    LwU16 stepSizemA;
} LW2080_CTRL_PMGR_PWR_DEVICE_INFO_DATA_NCT3933U;

/*!
 * Structure of static information specific to the GPU ADC 1.x power device.
 */
typedef struct LW2080_CTRL_PMGR_PWR_DEVICE_INFO_DATA_GPUADC1X {
    /*!
     * ADC PWM period in units of utilsclk cycles
     */
    LwU16 pwmPeriod;
    /*!
     * Length of IIR applied to ADC raw samples
     */
    LwU8  iirLength;
    /*!
     * Sampling delay in units of utilsclk cycles
     */
    LwU8  sampleDelay;
    /*!
     * Active GPUADC providers
     */
    LwU8  activeProvCount;
    /*!
     * Number of active channels. Represents the number channels that ADC will
     * cycle through
     */
    LwU8  activeChannelsNum;
    /*!
     * Maximum possible value of the ADC raw output after correction
     */
    LwU8  adcMaxValue;
    /*!
     * Length of reset cycle in us
     */
    LwU8  resetLength;
    /*!
     * VCM Offset VFE Variable Index
     */
    LwU8  vcmOffsetVfeVarIdx;
    /*!
     * Differential Offset VFE Variable Index
     */
    LwU8  diffOffsetVfeVarIdx;
    /*!
     * Differential Gain VFE Variable Index
     */
    LwU8  diffGailwfeVarIdx;
    /*!
     * Voltage (V) corresponding to ADC_MAX_VALUE (FXP8.8).
     * Should actually use LwUFXP8_8, however XAPI_GEN does not take LwUFXP8_8.
     * Since LwUFXP8_8 is actually LwU16, use LwU16 to WAR XAPI's restriction.
     */
    LwU16 fullRangeVoltage;
} LW2080_CTRL_PMGR_PWR_DEVICE_INFO_DATA_GPUADC1X;

/*!
 * Structure of static information specific to the GPU ADC 1.0 power device.
 */
typedef struct LW2080_CTRL_PMGR_PWR_DEVICE_INFO_DATA_GPUADC10 {
    LW2080_CTRL_PMGR_PWR_DEVICE_INFO_DATA_GPUADC1X super;

    /*!
     * Current (A) corresponding to ADC_MAX_VALUE (FXP8.8) for each provider.
     * Should actually use LwUFXP8_8, however XAPI_GEN does not take LwUFXP8_8.
     * Since LwUFXP8_8 is actually LwU16, use LwU16 to WAR XAPI's restriction.
     */
    LwU16                                          fullRangeLwrrent[LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC10_PHYSICAL_PROV_NUM];
    /*!
     * IPC configuration parameters
     */
    LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC10_IPC       ipc;
    /*!
     * Beacon configuration parameters
     */
    LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC10_BEACON    beacon;
} LW2080_CTRL_PMGR_PWR_DEVICE_INFO_DATA_GPUADC10;

/*!
 * Structure of static information specific to the GPUADC11 power device.
 */
typedef struct LW2080_CTRL_PMGR_PWR_DEVICE_INFO_DATA_GPUADC11 {
    LW2080_CTRL_PMGR_PWR_DEVICE_INFO_DATA_GPUADC1X super;

    /*!
     * Operating mode of ADC, Non-alternating vs Alternating
     * @ref LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC11_OPERATING_MODE_<XYZ>
     */
    LwU8                                           operatingMode;
    /*!
     * Number of samples that the HW multisampler block needs to accumulate
     * before sending to IIR filter. HW will accumulate (numSamples + 1).
     */
    LwU8                                           numSamples;
    /*!
     * VCM Coarse Offset VFE Variable Index
     */
    LwU8                                           vcmCoarseOffsetVfeVarIdx;
    /*!
     * Mask of providers configured to measure IMON
     */
    LwU16                                          imonProvMask;
    /*!
     * Current (A) corresponding to ADC_MAX_VALUE (FXP24.8) for each provider.
     * Should actually use LwUFXP24_8, however XAPI_GEN does not take LwUFXP24_8.
     * Since LwUFXP24_8 is actually LwU32, use LwU32 to WAR XAPI's restriction.
     */
    LwU32                                          fullRangeLwrrent[LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC11_PHYSICAL_PROV_NUM];
    /*!
     * OVRM configuration parameters
     */
    LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC11_OVRM      ovrm;
    /*!
     * IPC configuration parameters
     */
    LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC11_IPC       ipc;
    /*!
     * Beacon configuration parameters
     */
    LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC11_BEACON    beacon;
    /*!
     * Offset configuration parameters
     */
    LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC11_OFFSET    offset;
    /*!
     * Sum block configuration parameters
     */
    LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC11_SUM       sum;
} LW2080_CTRL_PMGR_PWR_DEVICE_INFO_DATA_GPUADC11;

/*!
 * Structure of static information specific to the GPUADC13 power device.
 */
typedef struct LW2080_CTRL_PMGR_PWR_DEVICE_INFO_DATA_GPUADC13 {
    LW2080_CTRL_PMGR_PWR_DEVICE_INFO_DATA_GPUADC1X super;

    /*!
     * Operating mode of ADC, Non-alternating vs Alternating
     * @ref LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC13_OPERATING_MODE_<XYZ>
     */
    LwU8                                           operatingMode;
    /*!
     * Number of samples that the HW multisampler block needs to accumulate
     * before sending to IIR filter. HW will accumulate (numSamples + 1).
     */
    LwU8                                           numSamples;
    /*!
     * VCM Coarse Offset VFE Variable Index
     */
    LwU8                                           vcmCoarseOffsetVfeVarIdx;
    /*!
     * Diff Coarse Gain VFE Variable Index
     */
    LwU8                                           diffCoarseGailwfeVarIdx;
    /*!
     * Current (A) corresponding to ADC_MAX_VALUE (FXP24.8) for each provider.
     * Should actually use LwUFXP16_16, however XAPI_GEN does not take LwUFXP16_16.
     * Since LwUFXP16_16 is actually LwU32, use LwU32 to WAR XAPI's restriction.
     */
    LwU32                                          fullRangeLwrrent[LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC13_PHYSICAL_PROV_NUM];
    /*!
     * OVRM configuration parameters
     */
    LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC13_OVRM      ovrm;
    /*!
     * IPC configuration parameters
     */
    LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC13_IPC       ipc;
    /*!
     * Beacon configuration parameters
     */
    LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC13_BEACON    beacon;
    /*!
     * Offset configuration parameters
     */
    LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC13_OFFSET    offset;
    /*!
     * Sum block configuration parameters
     */
    LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC13_SUM       sum;
    /*!
     * ADC PWM configuration parameters
     */
    LW2080_CTRL_PMGR_PWR_DEVICE_GPUADC13_ADC_PWM   adcPwm;
} LW2080_CTRL_PMGR_PWR_DEVICE_INFO_DATA_GPUADC13;

/*!
 * Union of PWR_DEVICE type-specific data.
 */


/*!
 * Structure of static information describing a PWR_DEVICE, which specifies per
 * the Power Sensors Table spec a power sensor on the GPU or board which is
 * capable of providing telemetry for some subset of power, current, and voltage
 * on a set number of providers.
 */
typedef struct LW2080_CTRL_PMGR_PWR_DEVICE_INFO {
    /*!
     * @ref LW2080_CTRL_PMGR_PWR_DEVICE_TYPE_<xyz>
     */
    LwU8 type;
    /*!
     * ~~~SPTODO~~~: Remove this value.  This is now found in PWR_CHANNEL.
     */
    LwU8 powerRail;

    /*!
     * Type-specific information.
     */
    union {
        LW2080_CTRL_PMGR_PWR_DEVICE_INFO_DATA_BA1XHW   ba10hw;
        LW2080_CTRL_PMGR_PWR_DEVICE_INFO_DATA_BA1XHW   ba11hw;
        LW2080_CTRL_PMGR_PWR_DEVICE_INFO_DATA_BA1XHW   ba12hw;
        LW2080_CTRL_PMGR_PWR_DEVICE_INFO_DATA_BA1XHW   ba13hw;
        LW2080_CTRL_PMGR_PWR_DEVICE_INFO_DATA_BA1XHW   ba14hw;
        LW2080_CTRL_PMGR_PWR_DEVICE_INFO_DATA_BA15HW   ba15hw;
        LW2080_CTRL_PMGR_PWR_DEVICE_INFO_DATA_BA16HW   ba16hw;
        LW2080_CTRL_PMGR_PWR_DEVICE_INFO_DATA_BA20     ba20;
        LW2080_CTRL_PMGR_PWR_DEVICE_INFO_DATA_INA219   ina219;
        LW2080_CTRL_PMGR_PWR_DEVICE_INFO_DATA_INA3221  ina3221;
        LW2080_CTRL_PMGR_PWR_DEVICE_INFO_DATA_NCT3933U nct3933u;
        LW2080_CTRL_PMGR_PWR_DEVICE_INFO_DATA_GPUADC1X gpuAdc1x;
        LW2080_CTRL_PMGR_PWR_DEVICE_INFO_DATA_GPUADC10 gpuAdc10;
        LW2080_CTRL_PMGR_PWR_DEVICE_INFO_DATA_GPUADC11 gpuAdc11;
        LW2080_CTRL_PMGR_PWR_DEVICE_INFO_DATA_GPUADC13 gpuAdc13;
    } data;
} LW2080_CTRL_PMGR_PWR_DEVICE_INFO;

/*!
 * LW2080_CTRL_CMD_PMGR_PWR_DEVICES_GET_INFO
 *
 * This command returns the static state describing the topology of PWR_DEVICEs
 * on the board.  This state primarily of the number of devices, their type, and
 * the power rail they are monitoring.
 *
 * This information is populated from the VBIOS Power Sensors Table (and
 * sometimes by RM overrides):
 * https://wiki.lwpu.com/engwiki/index.php/Resman/Silent_Running/Power_Sensors_And_Power_Capping_Tables_1.0_Specification#Power_Senors_Table_1.x_Structure
 * https://wiki.lwpu.com/engwiki/index.php/Resman/PState/Data_Tables/Power_Tables/Power_Sensors_Table_2.X
 *
 * See @ref LW2080_CTRL_PMGR_PWR_DEVICES_GET_INFO_PARAMS for documentation on
 * the parameters.
 *
 * Possible status values returned are
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_ILWALID_STATE
 *   LW_ERR_NOT_SUPPORTED
 */
#define LW2080_CTRL_CMD_PMGR_PWR_DEVICES_GET_INFO           (0x20802610U) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PMGR_INTERFACE_ID << 8) | LW2080_CTRL_PMGR_PWR_DEVICES_GET_INFO_PARAMS_MESSAGE_ID" */

/*!
 * Maximum number of PWR_DEVICES/sensors for power-monitoring/capping
 */
#define LW2080_CTRL_PMGR_PWR_DEVICES_MAX_DEVICES            (32U)

/*!
 * Maximum number of providers for a PWR_DEVICE
 */
#define LW2080_CTRL_PMGR_PWR_DEVICES_MAX_PROVIDERS          (0xEU)

/*!
 * Maximum number of thresholds for a PWR_DEVICE
 */
#define LW2080_CTRL_PMGR_PWR_DEVICE_PROVIDER_MAX_THRESHOLDS (4U)

#define LW2080_CTRL_PMGR_PWR_DEVICES_GET_INFO_PARAMS_MESSAGE_ID (0x10U)

typedef struct LW2080_CTRL_PMGR_PWR_DEVICES_GET_INFO_PARAMS {
    /*!
     * [out] - Returns the mask of valid entries in the Power Sensors Table.
     * The table may contain disabled entries, but in order for indexes to work
     * correctly, we need to reserve those entries.  The mask helps in this
     * regard.
     */
    LwU32                            devMask;

    /*!
     * [out] An array (of fixed size LW2080_CTRL_PMGR_PWR_DEVICES_MAX_DEVICES)
     * describing the individual PWR_DEVICES.  Has valid indexes corresponding to
     * bits set in @ref devMask.
     */
    LW2080_CTRL_PMGR_PWR_DEVICE_INFO devices[LW2080_CTRL_PMGR_PWR_DEVICES_MAX_DEVICES];
} LW2080_CTRL_PMGR_PWR_DEVICES_GET_INFO_PARAMS;

/*!
 * This structure is used to return power, current, voltage and energy as tuple.
 */
typedef struct LW2080_CTRL_PMGR_PWR_TUPLE {
    /*!
     * The power (mW).
     */
    LwU32         pwrmW;

    /*!
     * The current (mA).
     */
    LwU32         lwrrmA;

    /*!
     * The voltage (uV).
     */
    LwU32         voltuV;

    /*!
     * The energy (mJ).
     */
    LwU64_ALIGN32 energymJ;
} LW2080_CTRL_PMGR_PWR_TUPLE;
typedef struct LW2080_CTRL_PMGR_PWR_TUPLE *PLW2080_CTRL_PMGR_PWR_TUPLE;

/*!
 * Helper macro to initialize a @ref
 * LW2080_CTRL_PMGR_PWR_TUPLE structure to its default values.
 *
 * @param[in] pTuple  LW2080_CTRL_PMGR_PWR_TUPLE to init.
 */
#define LW2080_CTRL_PMGR_PWR_TUPLE_INIT(pTuple) \
    do {                                        \
        pTuple->pwrmW  = 0;                     \
        pTuple->lwrrmA = 0;                     \
        pTuple->voltuV = 0;                     \
    } while (LW_FALSE)

/*!
 * This structure is used to return power, current, voltage aclwmulations as tuple.
 */
typedef struct LW2080_CTRL_PMGR_PWR_TUPLE_ACC {
    /*!
     * The power aclwmulation (nJ = mW*us).
     */
    LwU64_ALIGN32 pwrAccnJ;

    /*!
     * The current aclwmulation (nC = mA*us).
     */
    LwU64_ALIGN32 lwrrAccnC;

    /*!
     * The voltage aclwmulation (mV*us).
     */
    LwU64_ALIGN32 voltAccmVus;

    /*!
     * Sequence id
     */
    LwU32         seqId;
} LW2080_CTRL_PMGR_PWR_TUPLE_ACC;

/*!
 * Structure representing the dynamic state of each provider to a PWR_DEVICE.
 */
typedef struct LW2080_CTRL_PMGR_PWR_DEVICE_PROVIDER_STATUS {
    /*!
     * Tuple to return the telemetry provided by the PWR_DEVICE's provider.
     */
    LW2080_CTRL_PMGR_PWR_TUPLE tuple;

    /*!
     * Number of thresholds supported by each provider.
     */
    LwU8                       numThresholds;

    /*!
     * Threshold values corresponding to each threshold index.
     */
    LwU32                      thresholds[LW2080_CTRL_PMGR_PWR_DEVICE_PROVIDER_MAX_THRESHOLDS];
} LW2080_CTRL_PMGR_PWR_DEVICE_PROVIDER_STATUS;
typedef struct LW2080_CTRL_PMGR_PWR_DEVICE_PROVIDER_STATUS *PLW2080_CTRL_PMGR_PWR_DEVICE_PROVIDER_STATUS;

/*!
 * Structure representing the dynamic state associated with a PWR_DEVICE
 * entry.
 */
typedef struct LW2080_CTRL_PMGR_PWR_DEVICE_STATUS {
    /*!
     * LW2080_CTRL_BOARDOBJ super class. Must always be first object in
     * structure.
     */
    LW2080_CTRL_BOARDOBJ                        super;

    /*!
     * @ref LW2080_CTRL_PMGR_PWR_DEVICE_TYPE_<xyz>.
     * XAPI does not support using an element within another structure as a
     * discriminant. Placing this redundant type value here until that design
     * constraint can be fixed.
     */
    LwU8                                        type;

    /*!
     * Number of providers supported by this PWR_DEVICE.
     */
    LwU8                                        numProviders;

    /*!
     * Array of providers.
     */
    LW2080_CTRL_PMGR_PWR_DEVICE_PROVIDER_STATUS providers[LW2080_CTRL_PMGR_PWR_DEVICES_MAX_PROVIDERS];
} LW2080_CTRL_PMGR_PWR_DEVICE_STATUS;
typedef struct LW2080_CTRL_PMGR_PWR_DEVICE_STATUS *PLW2080_CTRL_PMGR_PWR_DEVICE_STATUS;

/*!
 * LW2080_CTRL_CMD_PMGR_PWR_DEVICES_GET_STATUS
 *
 * This command returns the current state/readings for the requested set of
 * PWR_DEVICEs.
 *
 * See @ref LW2080_CTRL_PMGR_PWR_DEVICES_GET_STATUS_PARAMS for documentation on
 * the parameters.
 *
 * Possible status values returned are
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_ILWALID_STATE
 *   LW_ERR_NOT_SUPPORTED
 */
#define LW2080_CTRL_CMD_PMGR_PWR_DEVICES_GET_STATUS (0x20802611U) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PMGR_INTERFACE_ID << 8) | LW2080_CTRL_PMGR_PWR_DEVICES_GET_STATUS_PARAMS_MESSAGE_ID" */

/*!
 * Structure representing the dynamic status information associated with a set
 * of PWR_DEVICEs.
 */
#define LW2080_CTRL_PMGR_PWR_DEVICES_GET_STATUS_PARAMS_MESSAGE_ID (0x11U)

typedef struct LW2080_CTRL_PMGR_PWR_DEVICES_GET_STATUS_PARAMS {
    /*!
     * LW2080_CTRL_BOARDOBJGRP super class. Must always be first object in
     * structure.
     */
    LW2080_CTRL_BOARDOBJGRP_E32        super;

    /*!
     * [out] - Array of PWR_DEVICE entries. Has valid indexes corresponding to
     * the bits set in @ref devMask.
     */
    LW2080_CTRL_PMGR_PWR_DEVICE_STATUS devices[LW2080_CTRL_PMGR_PWR_DEVICES_MAX_DEVICES];
} LW2080_CTRL_PMGR_PWR_DEVICES_GET_STATUS_PARAMS;

/*!
 * Special value corresponding to an invalid Power Channel index.  This value
 * means that channel is not specified.
 */
#define LW2080_CTRL_PMGR_PWR_CHANNEL_INDEX_ILWALID                 LW2080_CTRL_BOARDOBJ_IDX_ILWALID_8BIT

/*!
 * Macros encoding the types of PWR_CHANNELs.
 */
#define LW2080_CTRL_PMGR_PWR_CHANNEL_TYPE_DEFAULT                  0x00U
#define LW2080_CTRL_PMGR_PWR_CHANNEL_TYPE_SUMMATION                0x01U
#define LW2080_CTRL_PMGR_PWR_CHANNEL_TYPE_ESTIMATION               0x02U
#define LW2080_CTRL_PMGR_PWR_CHANNEL_TYPE_SENSOR                   0x03U
#define LW2080_CTRL_PMGR_PWR_CHANNEL_TYPE_PSTATE_ESTIMATION_LUT    0x04U
#define LW2080_CTRL_PMGR_PWR_CHANNEL_TYPE_SENSOR_CLIENT_ALIGNED    0x05U

/*!
 * Macros for the PWR_CHANNEL power rails.
 */
#define LW2080_CTRL_PMGR_PWR_CHANNEL_POWER_RAIL_UNKNOWN            0x00U
#define LW2080_CTRL_PMGR_PWR_CHANNEL_POWER_RAIL_OUTPUT_LWVDD       0x01U
#define LW2080_CTRL_PMGR_PWR_CHANNEL_POWER_RAIL_OUTPUT_FBVDD       0x02U
#define LW2080_CTRL_PMGR_PWR_CHANNEL_POWER_RAIL_OUTPUT_FBVDDQ      0x03U
#define LW2080_CTRL_PMGR_PWR_CHANNEL_POWER_RAIL_OUTPUT_FBVDD_Q     0x04U
#define LW2080_CTRL_PMGR_PWR_CHANNEL_POWER_RAIL_OUTPUT_PEXVDD      0x05U
#define LW2080_CTRL_PMGR_PWR_CHANNEL_POWER_RAIL_OUTPUT_A3V3        0x06U
#define LW2080_CTRL_PMGR_PWR_CHANNEL_POWER_RAIL_OUTPUT_3V3LW       0x07U
#define LW2080_CTRL_PMGR_PWR_CHANNEL_POWER_RAIL_OUTPUT_TOTAL_GPU   0x08U
#define LW2080_CTRL_PMGR_PWR_CHANNEL_POWER_RAIL_OUTPUT_FBVDDQ_GPU  0x09U
#define LW2080_CTRL_PMGR_PWR_CHANNEL_POWER_RAIL_OUTPUT_FBVDDQ_MEM  0x0AU
#define LW2080_CTRL_PMGR_PWR_CHANNEL_POWER_RAIL_OUTPUT_SRAM        0x0BU
#define LW2080_CTRL_PMGR_PWR_CHANNEL_POWER_RAIL_INPUT_PEX12V1      0xDEU
#define LW2080_CTRL_PMGR_PWR_CHANNEL_POWER_RAIL_INPUT_TOTAL_BOARD2 0xDFU
#define LW2080_CTRL_PMGR_PWR_CHANNEL_POWER_RAIL_INPUT_HIGH_VOLT0   0xE0U
#define LW2080_CTRL_PMGR_PWR_CHANNEL_POWER_RAIL_INPUT_HIGH_VOLT1   0xE1U
#define LW2080_CTRL_PMGR_PWR_CHANNEL_POWER_RAIL_INPUT_LWVDD1       0xE2U
#define LW2080_CTRL_PMGR_PWR_CHANNEL_POWER_RAIL_INPUT_LWVDD2       0xE3U
#define LW2080_CTRL_PMGR_PWR_CHANNEL_POWER_RAIL_INPUT_EXT12V_8PIN2 0xE4U
#define LW2080_CTRL_PMGR_PWR_CHANNEL_POWER_RAIL_INPUT_EXT12V_8PIN3 0xE5U
#define LW2080_CTRL_PMGR_PWR_CHANNEL_POWER_RAIL_INPUT_EXT12V_8PIN4 0xE6U
#define LW2080_CTRL_PMGR_PWR_CHANNEL_POWER_RAIL_INPUT_EXT12V_8PIN5 0xE7U
#define LW2080_CTRL_PMGR_PWR_CHANNEL_POWER_RAIL_INPUT_MISC0        0xE8U
#define LW2080_CTRL_PMGR_PWR_CHANNEL_POWER_RAIL_INPUT_MISC1        0xE9U
#define LW2080_CTRL_PMGR_PWR_CHANNEL_POWER_RAIL_INPUT_MISC2        0xEAU
#define LW2080_CTRL_PMGR_PWR_CHANNEL_POWER_RAIL_INPUT_MISC3        0xEBU
#define LW2080_CTRL_PMGR_PWR_CHANNEL_POWER_RAIL_INPUT_USBC0        0xELW
#define LW2080_CTRL_PMGR_PWR_CHANNEL_POWER_RAIL_INPUT_USBC1        0xEDU
#define LW2080_CTRL_PMGR_PWR_CHANNEL_POWER_RAIL_INPUT_FAN0         0xEEU
#define LW2080_CTRL_PMGR_PWR_CHANNEL_POWER_RAIL_INPUT_FAN1         0xEFU
#define LW2080_CTRL_PMGR_PWR_CHANNEL_POWER_RAIL_INPUT_SRAM         0xF0U
#define LW2080_CTRL_PMGR_PWR_CHANNEL_POWER_RAIL_INPUT_PWR_SRC_PP   0xF1U
#define LW2080_CTRL_PMGR_PWR_CHANNEL_POWER_RAIL_INPUT_3V3_PP       0xF2U
#define LW2080_CTRL_PMGR_PWR_CHANNEL_POWER_RAIL_INPUT_3V3_MAIN     0xF3U
#define LW2080_CTRL_PMGR_PWR_CHANNEL_POWER_RAIL_INPUT_3V3_AON      0xF4U
#define LW2080_CTRL_PMGR_PWR_CHANNEL_POWER_RAIL_INPUT_TOTAL_BOARD  0xF5U
#define LW2080_CTRL_PMGR_PWR_CHANNEL_POWER_RAIL_INPUT_LWVDD        0xF6U
#define LW2080_CTRL_PMGR_PWR_CHANNEL_POWER_RAIL_INPUT_FBVDD        0xF7U
#define LW2080_CTRL_PMGR_PWR_CHANNEL_POWER_RAIL_INPUT_FBVDDQ       0xF8U
#define LW2080_CTRL_PMGR_PWR_CHANNEL_POWER_RAIL_INPUT_FBVDD_Q      0xF9U
#define LW2080_CTRL_PMGR_PWR_CHANNEL_POWER_RAIL_INPUT_EXT12V_8PIN0 0xFAU
#define LW2080_CTRL_PMGR_PWR_CHANNEL_POWER_RAIL_INPUT_EXT12V_8PIN1 0xFBU
#define LW2080_CTRL_PMGR_PWR_CHANNEL_POWER_RAIL_INPUT_EXT12V_6PIN0 0xFLW
#define LW2080_CTRL_PMGR_PWR_CHANNEL_POWER_RAIL_INPUT_EXT12V_6PIN1 0xFDU
#define LW2080_CTRL_PMGR_PWR_CHANNEL_POWER_RAIL_INPUT_PEX3V3       0xFEU
#define LW2080_CTRL_PMGR_PWR_CHANNEL_POWER_RAIL_INPUT_PEX12V       0xFFU

/*!
 * Structure for data specific to the legacy channels used in GF10X power
 * capping.
 */
typedef struct LW2080_CTRL_PMGR_PWR_CHANNEL_INFO_DATA_SENSOR {
    /*!
     * Index into the Power Sensors Table for the PWR_DEVICE from which this
     * PWR_CHANNEL should query power values.
     */
    LwU8 pwrDevIdx;

    /*!
     * Index of the PWR_DEVICE Provider to query.
     */
    LwU8 pwrDevProvIdx;
} LW2080_CTRL_PMGR_PWR_CHANNEL_INFO_DATA_SENSOR;

/*!
 * Structure for data specific to a SW channel which is the callwlated as the
 * sum of a set of other channels.  This set is specified as the array of
 * channel relationships between the first and last indexes (inclusive)
 * specified in this structure.
 */
typedef struct LW2080_CTRL_PMGR_PWR_CHANNEL_INFO_DATA_SUMMATION {
    /*!
     * Index of the first power channel relationship entry for this power
     * channel.
     */
    LwU8 relIdxFirst;
    /*!
     * Index of the last power channel relationship entry for this power
     * channel.
     */
    LwU8 relIdxLast;
} LW2080_CTRL_PMGR_PWR_CHANNEL_INFO_DATA_SUMMATION;

#define LW2080_CTRL_PMGR_PWR_CHANNEL_PSTATE_ESTIMATION_LUT_MAX_ENTRIES 2U

/*!
 * Union of PWR_CHANNEL_PSTATE_ESTIMATION_LUT_ENTRY data based on @ref bDynamicLut flag.
 */
typedef union LW2080_CTRL_PMGR_PWR_CHANNEL_PSTATE_ESTIMATION_LUT_ENTRY_DATA {

            /*!
             * Channel Relationship Index. Holds meaning only when @ref bDynamicLut
             * flag is enabled.
             */
    LwU8  dynChRelIdx;

            /*!
             * Estimated power offset value (in mW) to be used for pstates less than
             * or equal to the pState name for this entry. Holds meaning only when
             * @ref bDynamicLut flag is disabled.
             */
    LwU32 staticPowerOffsetmW;
} LW2080_CTRL_PMGR_PWR_CHANNEL_PSTATE_ESTIMATION_LUT_ENTRY_DATA;


/*!
 * Structure of the Pstate Estimation LUT Entry
 */
typedef struct LW2080_CTRL_PMGR_PWR_CHANNEL_PSTATE_ESTIMATION_LUT_ENTRY_INFO {
    /*!
     * LW2080 Pstate for LUT entry @ref LW2080_CTRL_PERF_PSTATES_XXX.
     */
    LwU32                                                         pstate;
    /*!
     * TODO-Chandrashis: Remove after LwAPI is fixed.
     */
    LwU32                                                         powerOffset;
    /*!
     * Boolean denoting whether Dynamic LUT is enabled.
     */
    LwBool                                                        bDynamicLut;
    /*!
     * Data based on @bDynamicLut flag.
     */
    LW2080_CTRL_PMGR_PWR_CHANNEL_PSTATE_ESTIMATION_LUT_ENTRY_DATA data;
} LW2080_CTRL_PMGR_PWR_CHANNEL_PSTATE_ESTIMATION_LUT_ENTRY_INFO;

/*!
 * Structure for data specific to the Pstate Estimation LUT
 */
typedef struct LW2080_CTRL_PMGR_PWR_CHANNEL_INFO_DATA_PSTATE_ESTIMATION_LUT {
    /*!
     * Array of LUT entry of size
     * LW2080_CTRL_PMGR_PWR_CHANNEL_PSTATE_ESTIMATION_LUT_MAX_ENTRIES.
     */
    LW2080_CTRL_PMGR_PWR_CHANNEL_PSTATE_ESTIMATION_LUT_ENTRY_INFO lutEntry[LW2080_CTRL_PMGR_PWR_CHANNEL_PSTATE_ESTIMATION_LUT_MAX_ENTRIES];
} LW2080_CTRL_PMGR_PWR_CHANNEL_INFO_DATA_PSTATE_ESTIMATION_LUT;

/*!
 * Union of PWR_CHANNEL type-specific data.
 */


/*!
 * This structure is used to contain static information related to a PWR_CHANNEL.
 */
typedef struct LW2080_CTRL_PMGR_PWR_CHANNEL_INFO {
    /*!
     * LW2080_CTRL_PMGR_PWR_CHANNEL_TYPE_<xyz>.
     */
    LwU8  type;
    /*!
     * LW2080_CTRL_PMGR_PWR_CHANNEL_POWER_RAIL_<xyz>.
     */
    LwU8  pwrRail;
    /*!
     * Fixed voltage (in uV) to assume for this PWR_CHANNEL.  Used to simplify
     * power <-> current colwersion.
     */
    LwU32 voltFixeduV;
    /*!
     * Correction slope (unitless FXP20.12) which is used to correct power
     * readings from this channel.
     */
    LwU32 pwrCorrSlope;
    /*!
     * Correction offset (mW) by which is used to correct power readings from
     * this channel.
     */
    LwS32 pwrCorrOffsetmW;
    /*!
     * Correction offset (mW) by which is used to correct power readings from
     * this channel.
     *
     * ~~~SPTODO~~~: Remove this value.  A duplicate value to prevent breaking
     * LWAPI.
     */
    LwS32 pwrOffsetmW;
    /*!
     * Correction slope (unitless FXP20.12) which is used to correct current
     * readings from this channel.
     */
    LwU32 lwrrCorrSlope;
    /*!
     * Correction offset (mA) by which is used to correct current readings from
     * this channel.
     */
    LwS32 lwrrCorrOffsetmA;

    /*!
     * Type-specific data.
     */
    union {
        LW2080_CTRL_PMGR_PWR_CHANNEL_INFO_DATA_SENSOR                sensor;
        LW2080_CTRL_PMGR_PWR_CHANNEL_INFO_DATA_SUMMATION             sum;
        LW2080_CTRL_PMGR_PWR_CHANNEL_INFO_DATA_PSTATE_ESTIMATION_LUT pstateEstLUT;
    } data;
} LW2080_CTRL_PMGR_PWR_CHANNEL_INFO;

/*!
 * Macros encoding types/classes of PWR_CHRELATIONSHIP entries.
 */

/*!
 * Enumeration of power channel relationship types the PMU supports.
 */
#define LW2080_CTRL_PMGR_PWR_CHRELATIONSHIP_TYPE_WEIGHT               0x00U
#define LW2080_CTRL_PMGR_PWR_CHRELATIONSHIP_TYPE_BALANCED_PHASE_EST   0x01U
#define LW2080_CTRL_PMGR_PWR_CHRELATIONSHIP_TYPE_BALANCING_PWM_WEIGHT 0x02U
#define LW2080_CTRL_PMGR_PWR_CHRELATIONSHIP_TYPE_REGULATOR_LOSS_EST   0x03U
#define LW2080_CTRL_PMGR_PWR_CHRELATIONSHIP_TYPE_REGULATOR_LOSS_DYN   0x04U
#define LW2080_CTRL_PMGR_PWR_CHRELATIONSHIP_TYPE_REGULATOR_EFF_EST_V1 0x05U

/*!
 * Enumeration for various types of voltage regulators.
 */
typedef enum LW2080_CTRL_PMGR_PWR_CHRELATIONSHIP_REGULATOR_TYPE {
    LW2080_CTRL_PMGR_PWR_CHRELATIONSHIP_REGULATOR_TYPE_LWVDD = 0,
    LW2080_CTRL_PMGR_PWR_CHRELATIONSHIP_REGULATOR_TYPE_FBVDD = 1,

    // Number of voltage regulators (always the last one).
    LW2080_CTRL_PMGR_PWR_CHRELATIONSHIP_REGULATOR_TYPE_COUNT = 2,
} LW2080_CTRL_PMGR_PWR_CHRELATIONSHIP_REGULATOR_TYPE;

/*!
 * Structure for data specific to a weighted power channel relationship type.
 * This is a relationship evaluates to a weight/proportion of power of the
 * specififed channel index.
 */
typedef struct LW2080_CTRL_PMGR_PWR_CHRELATIONSHIP_INFO_DATA_WEIGHT {
    /*!
     * Unitless signed FXP20.12 weight value to use to scale the power of the
     * specified channel.
     */
    LwS32 weight;
} LW2080_CTRL_PMGR_PWR_CHRELATIONSHIP_INFO_DATA_WEIGHT;

/*!
 * Structure for data specific to a Balanced PWM Weight Power Channel
 * Relationship type.  This is relationship evaluates to a weight/proportion of
 * power of the specified channel index, where the weight is the PWM percentage
 * associated with a Balancing Power Policy Relatioship.
 */
typedef struct LW2080_CTRL_PMGR_PWR_CHRELATIONSHIP_INFO_DATA_BALANCING_PWM_WEIGHT {
    /*!
     * Index to PWR_POLICY_RELATIONSHIP object of type
     * PWR_POLICY_RELATIONSHIP_BALANCE which will be queried for its current PWM
     * percentage.
     */
    LwU8   balancingRelIdx;

    /*!
     * Boolean indicating whether to use the primary/normal or
     * secondary/ilwerted PWM percentage as the weight.
     */
    LwBool bPrimary;
} LW2080_CTRL_PMGR_PWR_CHRELATIONSHIP_INFO_DATA_BALANCING_PWM_WEIGHT;

/*!
 * Structure defining a set of PWR_POLICY_RELATIONSHIPs
 */
typedef struct LW2080_CTRL_PMGR_PWR_POLICY_RELATIONSHIP_SET {
    /*!
     * Lower index of the PWR_POLICY_RELATIONSHIP set, inclusive.
     */
    LwBoardObjIdx policyRelStart;

    /*!
     * Upper index of the PWR_POLICY_RELATIONSHIP set, inclusive.
     */
    LwBoardObjIdx policyRelEnd;
} LW2080_CTRL_PMGR_PWR_POLICY_RELATIONSHIP_SET;
typedef struct LW2080_CTRL_PMGR_PWR_POLICY_RELATIONSHIP_SET *PLW2080_CTRL_PMGR_PWR_POLICY_RELATIONSHIP_SET;

/*!
 *
 * @defgroup    LW2080_CTRL_PMGR_PWR_POLICY_RELATIONSHIP_SET_FOR_EACH
 *
 * Macros for iterating over the @ref LwBoardObjIdx indices in a
 * @ref LW2080_CTRL_PMGR_PWR_POLICY_RELATIONSHIP_SET
 *
 * @{
 */
#define LW2080_CTRL_PMGR_PWR_POLICY_RELATIONSHIP_SET_FOR_EACH_BEGIN(_pRelSet, _polRelIdx) \
    for ((_polRelIdx) = (_pRelSet)->policyRelStart; \
         ((_polRelIdx) <= (_pRelSet)->policyRelEnd) && ((_polRelIdx) != LW2080_CTRL_BOARDOBJ_IDX_ILWALID); \
         (_polRelIdx)++) \
    {
#define  LW2080_CTRL_PMGR_PWR_POLICY_RELATIONSHIP_SET_FOR_EACH_END \
    }
/*!@*/

/*!
 * Structure for data specific to a Balanced Phase Estimate power channel
 * relationship type.  This relationship evaluates to a proportion of power
 * of the specififed channel index, where the proportion is a dynamic factor
 * depending on the value of the balanced phases.
 *
 * This Channel Relationship is intended to be used to compute the total input
 * power of a voltage regulator of which one or more phases are being
 * dynamically balanced via the _BALANCE Power Policy and Power Policy
 * Relationship Classes.
 */
typedef struct LW2080_CTRL_PMGR_PWR_CHRELATIONSHIP_INFO_DATA_BALANCED_PHASE_EST {
    /*!
     * Total number of phases by which to scale up the estimated power.
     */
    LwU8 numTotalPhases;
    /*!
     * Number of static phases which should be included in the evaluation of
     * this Channel Relationship.
     */
    LwU8 numStaticPhases;
    /*!
     * Index of first _BALANCED Power Policy Relationship representing a
     * balanced phase.  This relationship must be of type _BALANCED.
     *
     * TODO: move to use LW2080_CTRL_PMGR_PWR_POLICY_RELATIONSHIP_SET
     */
    LwU8 balancedPhasePolicyRelIdxFirst;
    /*!
     * Index of last _BALANCED Power Policy Relationship representing a
     * balanced phase.  This relationship must be of type _BALANCED.
     *
     * TODO: move to use LW2080_CTRL_PMGR_PWR_POLICY_RELATIONSHIP_SET
     */
    LwU8 balancedPhasePolicyRelIdxLast;
} LW2080_CTRL_PMGR_PWR_CHRELATIONSHIP_INFO_DATA_BALANCED_PHASE_EST;

/*!
 * Structure for data specific to a regulator loss estimation power channel
 * relationship type. This is a relationship class to estimate the output value
 * based on provided coefficients.
 */
typedef struct LW2080_CTRL_PMGR_PWR_CHRELATIONSHIP_INFO_DATA_REGULATOR_LOSS_EST {
   /*!
     * Regulator type.
     */
    LwU8        regulatorType;

    /*!
     * Constant Coefficient 0 (mW, FXP20.12 signed)
     */
    LwSFXP20_12 coefficient0;

    /*!
     * Constant Coefficient 1 (mV, FXP20.12 signed)
     */
    LwSFXP20_12 coefficient1;

    /*!
     * First-order Coefficient 2 (mW / mV, FXP20.12 signed)
     */
    LwSFXP20_12 coefficient2;

    /*!
     * First-order Coefficient 3 (mW / mV, FXP20.12 signed)
     */
    LwSFXP20_12 coefficient3;

    /*!
     * First-order Coefficient 4 (mV / mV, FXP20.12 signed)
     */
    LwSFXP20_12 coefficient4;

    /*!
     * First-order Coefficient 5 (mV / mV, FXP20.12 signed)
     */
    LwSFXP20_12 coefficient5;
} LW2080_CTRL_PMGR_PWR_CHRELATIONSHIP_INFO_DATA_REGULATOR_LOSS_EST;

/*!
 * Structure for data specific to a regulator loss dynamic power channel
 * relationship type. This is a relationship class to compute the output value
 * based on duty cycle of a power channel.
 */
typedef struct LW2080_CTRL_PMGR_PWR_CHRELATIONSHIP_INFO_DATA_REGULATOR_LOSS_DYN {
    /*!
     * Index into the Thermal Monitor Table entry
     */
    LwU8 thermMonIdx;

    /*!
     * Voltage Domain @ref LW2080_CTRL_VOLT_VOLT_DOMAIN_<xyz>
     */
    LwU8 voltDomain;
} LW2080_CTRL_PMGR_PWR_CHRELATIONSHIP_INFO_DATA_REGULATOR_LOSS_DYN;

/*!
 * Enumerations of primary channel unit for REGULATOR_EFF_EST_V1 class
 */
#define LW2080_CTRL_PMGR_PWR_CHRELATIONSHIP_REGULATOR_EFF_EST_V1_PRIMARY_UNIT_MW  0U
#define LW2080_CTRL_PMGR_PWR_CHRELATIONSHIP_REGULATOR_EFF_EST_V1_PRIMARY_UNIT_MA  1U

/*!
 * Enumerations of directions for REGULATOR_EFF_EST_V1 class
 */
#define LW2080_CTRL_PMGR_PWR_CHRELATIONSHIP_REGULATOR_EFF_EST_V1_DIRECTION_IN_OUT 0U
#define LW2080_CTRL_PMGR_PWR_CHRELATIONSHIP_REGULATOR_EFF_EST_V1_DIRECTION_OUT_IN 1U

/*!
 * Structure for data specific to a regulator efficiency estimation power channel
 * relationship type. It allows to callwlate the regulator efficiency using primary
 * and secondary topology channels for either input side or output side and based on
 * an equation model with known coefficients.
 */
typedef struct LW2080_CTRL_PMGR_PWR_CHRELATIONSHIP_INFO_DATA_REGULATOR_EFF_EST_V1 {
    /*!
     * Secondary channel index.
     */
    LwU8        chIdxSecondary;

    /*!
     * Primary channel unit.
     */
    LwU8        unitPrimary;

    /*!
     * Direction, input to output:efficiency needs to be multiplied
     * output to input: efficiency needs to be divided with primary channel
     */
    LwU8        direction;

    /*!
     * Second-order Coefficient 0 (Volt^2, FXP20.12 signed)
     */
    LwSFXP20_12 coefficient0;

    /*!
     * Second-order Coefficient 1 (Volt^2, FXP20.12 signed)
     */
    LwSFXP20_12 coefficient1;

    /*!
     * Second-order Coefficient 2 (mW^2 or mA^2 based on direction, FXP20.12 signed)
     */
    LwSFXP20_12 coefficient2;

    /*!
     * First-order Coefficient 3 (V, FXP20.12 signed)
     */
    LwSFXP20_12 coefficient3;

    /*!
     * First-order Coefficient 4 (V, FXP20.12 signed)
     */
    LwSFXP20_12 coefficient4;

    /*!
     * First-order Coefficient 5 (W or A based on direction, FXP20.12 signed)
     */
    LwSFXP20_12 coefficient5;

    /*!
     * Second-order Coefficient 6 (V^2, FXP20.12 signed)
     */
    LwSFXP20_12 coefficient6;

    /*!
     * Second-order Coefficient 7 (V*W or V*A based on direction, FXP20.12 signed)
     */
    LwSFXP20_12 coefficient7;

    /*!
     * Second-order Coefficient 8 (V*W or V*A based on direction, FXP20.12 signed)
     */
    LwSFXP20_12 coefficient8;

    /*!
     * Constant Coefficient 9 (unitless, FXP20.12 signed)
     */
    LwSFXP20_12 coefficient9;
} LW2080_CTRL_PMGR_PWR_CHRELATIONSHIP_INFO_DATA_REGULATOR_EFF_EST_V1;

/*!
 * Union of type-specific data.
 */


/*!
 * Structure representing a power channel relationship - specifying how another
 * channel can be used to callwlate the power of a given channel.  Power
 * channels whose power is callwlated based on the values of other channels use
 * indexes to power channel relationships to make these callwlations.
 */
typedef struct LW2080_CTRL_PMGR_PWR_CHRELATIONSHIP_INFO {
    /*!
     * LW2080_CTRL_PMGR_PWR_CHRELATIONSHIP_TYPE_<xyz>.
     */
    LwU8 type;

    /*!
     * PWR_CHANNEL index for this relationship.
     */
    LwU8 chIdx;

    /*!
     * Type-specific data.
     */
    union {
        LW2080_CTRL_PMGR_PWR_CHRELATIONSHIP_INFO_DATA_WEIGHT               weight;
        LW2080_CTRL_PMGR_PWR_CHRELATIONSHIP_INFO_DATA_BALANCED_PHASE_EST   balancedPhaseEst;
        LW2080_CTRL_PMGR_PWR_CHRELATIONSHIP_INFO_DATA_BALANCING_PWM_WEIGHT balancingPwmWeight;
        LW2080_CTRL_PMGR_PWR_CHRELATIONSHIP_INFO_DATA_REGULATOR_LOSS_EST   regulatorLossEst;
        LW2080_CTRL_PMGR_PWR_CHRELATIONSHIP_INFO_DATA_REGULATOR_LOSS_DYN   regulatorLossDyn;
        LW2080_CTRL_PMGR_PWR_CHRELATIONSHIP_INFO_DATA_REGULATOR_EFF_EST_V1 regulatorEffEstV1;
    } data;
} LW2080_CTRL_PMGR_PWR_CHRELATIONSHIP_INFO;

/*!
 * LW2080_CTRL_CMD_PMGR_PWR_MONITOR_GET_INFO
 *
 * This command returns the static state describing any power monitoring which
 * the PWR_MONITOR might be performing on this board.  This state is primarily
 * the sampling period, the number of samples, and the channels (a set of
 * PWR_DEVICES) which PWR_MONITOR is monitoring.  These channels are populated
 * by the VBIOS Power Capping Table (and possibly RM overrides):
 * https://wiki.lwpu.com/engwiki/index.php/Resman/Silent_Running/Power_Sensors_And_Power_Capping_Tables_1.0_Specification#Power_Capping_Table_1.x_Structure
 *
 * The PWR_MONITOR samples each channel at with the given sample period (ms) for
 * the given number of samples.  This sample set comprises an "iteration" for
 * which the PWR_MONITOR task collects various statistical data (max, avg, min,
 * possibly more to added).  For more information please see:
 * https://wiki.lwpu.com/engwiki/index.php/Resman/Silent_Running/Power_Sensors_And_Power_Capping_Tables_1.0_Specification#Power_Sensors_Monitoring
 *
 * See LW2080_CTRL_PMGR_PWR_MONITOR_GET_INFO_PARAMS for documentation on the
 * parameters.
 *
 * Possible status values returned are
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_ILWALID_STATE
 *   LW_ERR_NOT_SUPPORTED
 */
#define LW2080_CTRL_CMD_PMGR_PWR_MONITOR_GET_INFO        (0x20802612U) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PMGR_INTERFACE_ID << 8) | LW2080_CTRL_PMGR_PWR_MONITOR_GET_INFO_PARAMS_MESSAGE_ID" */

/*!
 * Maximum number of input channels for the power-capping algorithm.
 */
#define LW2080_CTRL_PMGR_PWR_CHANNEL_MAX_CHANNELS        (32U)

/*!
 * Maximum number of channel relationships which can be used for power monitoring.
 */
#define LW2080_CTRL_PMGR_PWR_CHANNEL_MAX_CHRELATIONSHIPS (32U)

#define LW2080_CTRL_PMGR_PWR_MONITOR_GET_INFO_PARAMS_MESSAGE_ID (0x12U)

typedef struct LW2080_CTRL_PMGR_PWR_MONITOR_GET_INFO_PARAMS {
    /*!
     * Boolean value representing whether PWR_MONITOR is enabled on this GPU.
     * If LW_FALSE, all other PWR_MONITOR RMCTRLs will return
     * LW_ERR_NOT_SUPPORTED.
     */
    LwBool                                   bSupported;

    /*!
     * Sampling period of the PWR_MONITOR task.  Only applicable if bSupported
     * == LW_TRUE.
     */
    LwU16                                    samplingPeriodms;
    /*!
     * Number of samples the PWR_MONITOR task collects for each "iteration."
     * Only applicable if bSupported == LW_TRUE.
     */
    LwU8                                     sampleCount;

    /*!
     * Returns the mask of valid channel entries in the Power Topology Table.
     * The table may contain disabled entries, but in order for indexes to work
     * correctly, we need to reserve those entries.  The mask helps in this
     * regard.  Only applicable if bSupported == LW_TRUE.
     */
    LwU32                                    channelMask;
    /*!
     * Set of channels, that, when added up, yield total GPU power.  Must be a
     * subset of @ref channelMask above.
     *
     * @note This is as legacy implementation for
     * PWR1.0/Power Capping Table, in which all rails are listed separately and
     * must be summed separately. For PWR2.0/Power Policy Table, this mask is
     * implemented but it is deprecated in favor of @ref totalGpuChannelIdx.
     */
    LwU32                                    totalGpuPowerChannelMask;
    /*!
     * Channel index corresponding to TOTAL_GPU power.  This value is to be
     * referenced in PWR2.0/Power Policy Table for the single channel which
     * represents total GPU power.
     *
     * @note LW2080_CTRL_PMGR_PWR_CHANNEL_INDEX_ILWALID denotes a single Power
     * Channel for total GPU power is not supported.
     */
    LwU8                                     totalGpuChannelIdx;

    /*!
     * An array (of fixed size LW2080_CTRL_PMGR_PWR_CHANNEL_MAX_CHANNELS)
     * describing individual power channels.  Has valid indexes corresponding to
     * bits set in channelMask.
     */
    LW2080_CTRL_PMGR_PWR_CHANNEL_INFO        channels[LW2080_CTRL_PMGR_PWR_CHANNEL_MAX_CHANNELS];

    /*!
     * Returns the mask of valid channel relationship entries in the Power
     * Topology Table.  The table may contain disabled entries, but in order for
     * indexes to work correctly, we need to reserve those entries.  The mask
     * helps in this regard.  Only applicable if bSupported == LW_TRUE.
     */
    LwU32                                    chRelMask;
    /*!
     * An array (of fixed size LW2080_CTRL_PMGR_PWR_CHANNEL_MAX_CHRELATIONSHIPS)
     * describing individual power channel relationships.  Has valid indexes
     * corresponding to bits set in chRelMask.
     */
    LW2080_CTRL_PMGR_PWR_CHRELATIONSHIP_INFO chRels[LW2080_CTRL_PMGR_PWR_CHANNEL_MAX_CHRELATIONSHIPS];
} LW2080_CTRL_PMGR_PWR_MONITOR_GET_INFO_PARAMS;

/*!
 * Structure representing the _SENSOR_CLIENT_ALIGNED-specific dynamic state
 * data
 */
typedef struct LW2080_CTRL_PMGR_PWR_CHANNEL_STATUS_SENSOR_CLIENT_ALIGNED {
    /*!
     * Last Aclwmulator readings. Pass in values from previous call.
     */
    LW2080_CTRL_PMGR_PWR_TUPLE_ACC lastTupleAcc;
    /*!
     * Last timer readings (ns). Pass in values from previous call.
     */
    LwU64_ALIGN32                  lastTimens;
} LW2080_CTRL_PMGR_PWR_CHANNEL_STATUS_SENSOR_CLIENT_ALIGNED;
typedef struct LW2080_CTRL_PMGR_PWR_CHANNEL_STATUS_SENSOR_CLIENT_ALIGNED *PLW2080_CTRL_PMGR_PWR_CHANNEL_STATUS_SENSOR_CLIENT_ALIGNED;

/*!
 * Union of type-specific data.
 */


typedef struct LW2080_CTRL_PMGR_PWR_CHANNEL_STATUS {
    /*!
     * LW2080_CTRL_BOARDOBJ super class. Must always be first object in
     * structure.
     */
    LW2080_CTRL_BOARDOBJ       super;

    /*!
     * @ref LW2080_CTRL_PMGR_PWR_CHANNEL_TYPE_<xyz>.
     */
    LwU8                       type;

    /*!
     * Structure describing the live power tuple for the individual channel.
     */
    LW2080_CTRL_PMGR_PWR_TUPLE tuple;

    /*!
     * Structure describing the polled power tuple for the individual channel.
     */
    LW2080_CTRL_PMGR_PWR_TUPLE tuplePolled;

    /*!
     * Type-specific dynamic state.
     */
    union {
        LW2080_CTRL_PMGR_PWR_CHANNEL_STATUS_SENSOR_CLIENT_ALIGNED sensorClientAligned;
    } data;
} LW2080_CTRL_PMGR_PWR_CHANNEL_STATUS;
typedef struct LW2080_CTRL_PMGR_PWR_CHANNEL_STATUS *PLW2080_CTRL_PMGR_PWR_CHANNEL_STATUS;

typedef struct LW2080_CTRL_PMGR_PWR_CHRELATIONSHIP_STATUS_DATA_REGULATOR_LOSS_DYN {
    /*!
     * Duty Cycle of VR callwlated by the relationship.
     */
    LwUFXP20_12 dutyCycle;
} LW2080_CTRL_PMGR_PWR_CHRELATIONSHIP_STATUS_DATA_REGULATOR_LOSS_DYN;

/*!
 * Union of type-specific data.
 */


typedef struct LW2080_CTRL_PMGR_PWR_CHRELATIONSHIP_STATUS {
    /*!
     * LW2080_CTRL_BOARDOBJ super class. Must always be first object in
     * structure.
     */
    LW2080_CTRL_BOARDOBJ super;

    /*!
     * @ref LW2080_CTRL_PMGR_PWR_CHRELATIONSHIP_TYPE_<xyz>.
     */
    LwU8                 type;

    /*!
     * Type-specific dynamic state.
     */
    union {
        LW2080_CTRL_PMGR_PWR_CHRELATIONSHIP_STATUS_DATA_REGULATOR_LOSS_DYN regulatorLossDyn;
    } data;
} LW2080_CTRL_PMGR_PWR_CHRELATIONSHIP_STATUS;

/*!
 * LW2080_CTRL_CMD_PMGR_PWR_MONITOR_GET_STATUS
 *
 * This command returns the PWR_MONITOR statistical information for the latest
 * iteration of sampling for the given subset of channels.  This command prompts
 * the RM to request this data from the PWR_MONITOR.
 *
 * Possible status values returned are
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_ILWALID_STATE
 *   LW_ERR_NOT_SUPPORTED
 */
#define LW2080_CTRL_CMD_PMGR_PWR_MONITOR_GET_STATUS (0x20802613U) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PMGR_INTERFACE_ID << 8) | LW2080_CTRL_PMGR_PWR_MONITOR_GET_STATUS_PARAMS_MESSAGE_ID" */

/*!
 * Structure representing the status of LW2080_CTRL_PMGR_PWR_CHRELATIONSHIPS_STATUS BOARDOBJGRP.
 */

typedef struct LW2080_CTRL_PMGR_PWR_CHRELATIONSHIPS_STATUS {
    /*!
     * LW2080_CTRL_BOARDOBJGRP_E32 super class. Must always be first object in
     * structure.
     */
    LW2080_CTRL_BOARDOBJGRP_E32                super;
    /*!
     * [out] - An array describing the sample data for the individual channel
     * relationship.
     * Data will be returned at the indexes corresponding to the bits set in the
     * chRelMask.
     */
    LW2080_CTRL_PMGR_PWR_CHRELATIONSHIP_STATUS chRels[LW2080_CTRL_PMGR_PWR_CHANNEL_MAX_CHRELATIONSHIPS];
} LW2080_CTRL_PMGR_PWR_CHRELATIONSHIPS_STATUS;

#define LW2080_CTRL_PMGR_PWR_MONITOR_GET_STATUS_PARAMS_MESSAGE_ID (0x13U)

typedef struct LW2080_CTRL_PMGR_PWR_MONITOR_GET_STATUS_PARAMS {
    /*!
     * LW2080_CTRL_BOARDOBJGRP_E32 super class. Must always be first object in
     * structure.
     */
    LW2080_CTRL_BOARDOBJGRP_E32                 super;

    /*!
     * [out] - Total GPU power corresponding to the last iteration of sampling.
     * This is the summation of the values corresponding to the Power Channels
     * indexes provided in @ref
     * LW2080_CTRL_PMGR_PWR_MONITOR_GET_INFO_PARAMS::totalGpuPowerChannelMask.
     */
    LwU32                                       totalGpuPowermW;
    /*!
     * [out] - A structure describing the sample data for the individual channel
     * Data will be returned at the indexes corresponding to the bits set in the
     * channelMask.
     */
    LW2080_CTRL_PMGR_PWR_CHANNEL_STATUS         channelStatus[LW2080_CTRL_PMGR_PWR_CHANNEL_MAX_CHANNELS];
    /*!
     * Nesting the PWR_CHRELATIONSHIPS_STATUS
     */
    LW2080_CTRL_PMGR_PWR_CHRELATIONSHIPS_STATUS chRelStatus;
} LW2080_CTRL_PMGR_PWR_MONITOR_GET_STATUS_PARAMS;

/*!
 * Maximum number of RM_PMU_PMGR_PWR_EQUATION entries which can be supported in
 * the RM or PMU.
 */
#define LW2080_CTRL_PMGR_PWR_EQUATION_MAX_EQUATIONS                          32U

/*!
 * Special value corresponding to an invalid Power Equation index.  This value
 * means that equation is not specified.
 */
#define LW2080_CTRL_PMGR_PWR_EQUATION_INDEX_ILWALID                          LW2080_CTRL_BOARDOBJ_IDX_ILWALID_8BIT

/*!
 * Macros encoding types/classes of PWR_EQUATION entries.
 */
#define LW2080_CTRL_PMGR_PWR_EQUATION_TYPE_LEAKAGE_DTCS10                    0x00U
#define LW2080_CTRL_PMGR_PWR_EQUATION_TYPE_LEAKAGE_DTCS11                    0x01U
#define LW2080_CTRL_PMGR_PWR_EQUATION_TYPE_BA1X_SCALE                        0x02U
#define LW2080_CTRL_PMGR_PWR_EQUATION_TYPE_BA00_FIT__DEPRECATED_DO_NOT_REUSE 0x03U
#define LW2080_CTRL_PMGR_PWR_EQUATION_TYPE_LEAKAGE_DTCS12                    0x04U
#define LW2080_CTRL_PMGR_PWR_EQUATION_TYPE_DYNAMIC_10                        0x05U
#define LW2080_CTRL_PMGR_PWR_EQUATION_TYPE_BA15_SCALE                        0x06U
#define LW2080_CTRL_PMGR_PWR_EQUATION_TYPE_LEAKAGE_DTCS13                    0x07U
#define LW2080_CTRL_PMGR_PWR_EQUATION_TYPE_DYNAMIC                           0xFDU
#define LW2080_CTRL_PMGR_PWR_EQUATION_TYPE_LEAKAGE                           0xFEU

/*!
 * Structure defining data common to all leakage power equations
 */
typedef struct LW2080_CTRL_PMGR_PWR_EQUATION_INFO_DATA_LEAKAGE {
    /*!
     * Floor-sweeping efficiency value by which to scale leakage power equations
     * for the amount of leakage power eliminated by floorswept area.
     *
     * Unsigned FXP 4.12 value.  Unitless.
     */
    LwUFXP4_12 fsEff;
    /*!
     * Power-gating efficiency value by which to scale leakage power equations
     * for the amount of leakage power eliminated while power-gating is engaged.
     *
     * Unsigned FXP 4.12 value.  Unitless.
     */
    LwUFXP4_12 pgEff;
} LW2080_CTRL_PMGR_PWR_EQUATION_INFO_DATA_LEAKAGE;

/*!
 * Structure of data specific to the the DTCS 1.1 leakage power equation.
 */
typedef struct LW2080_CTRL_PMGR_PWR_EQUATION_INFO_DATA_LEAKAGE_DTCS11 {
    /*!
     * Data common to all leakage power equations
     */
    LW2080_CTRL_PMGR_PWR_EQUATION_INFO_DATA_LEAKAGE leakage;
    /*!
     * Unsigned integer k0 coefficient.  Units of mV.
     */
    LwU32                                           k0;
    /*!
     * Unsigned FXP 20.12 value.  Unitless.
     */
    LwU32                                           k1;
    /*!
     * Unsigned FXP 20.12 value.  Unitless.
     */
    LwU32                                           k2;
    /*!
     * Signed FXP 24.8 value.  Units of C.
     */
    LwS32                                           k3;
} LW2080_CTRL_PMGR_PWR_EQUATION_INFO_DATA_LEAKAGE_DTCS11;

/*!
 * Structure of data specific to the the DTCS 1.2 leakage power equation.
 */
typedef struct LW2080_CTRL_PMGR_PWR_EQUATION_INFO_DATA_LEAKAGE_DTCS12 {
    /*!
     * DTCS11 super struct.
     */
    LW2080_CTRL_PMGR_PWR_EQUATION_INFO_DATA_LEAKAGE_DTCS11 super;
    /*!
     * VFE index representing IDDQ.
     */
    LwU8                                                   iddqVfeIdx;
    /*!
     * Therm Channel Index for Tj.
     */
    LwU8                                                   tjThermChIdx;
} LW2080_CTRL_PMGR_PWR_EQUATION_INFO_DATA_LEAKAGE_DTCS12;

/*!
 * Structure of data specific to the the DTCS 1.3 leakage power equation.
 */
typedef struct LW2080_CTRL_PMGR_PWR_EQUATION_INFO_DATA_LEAKAGE_DTCS13 {
    /*!
     * DTCS11 super struct.
     */
    LW2080_CTRL_PMGR_PWR_EQUATION_INFO_DATA_LEAKAGE_DTCS12 super;
    /*!
     * Leakage proportion in case GPC-RG is fully resident.
     * Value will be between 0 and 1
     */
    LwUFXP4_12                                             gpcrgEff;
     /*!
     * Leakage proportion in case GR-RPG is fully resident.
     * Value will be between 0 and 1
     */
    LwUFXP4_12                                             grrpgEff;
} LW2080_CTRL_PMGR_PWR_EQUATION_INFO_DATA_LEAKAGE_DTCS13;

/*!
 * Structure of data specific to the the BA v1.x Scale power equation.
 */
typedef struct LW2080_CTRL_PMGR_PWR_EQUATION_INFO_DATA_BA1X_SCALE {
    /*!
     * Reference voltage [uV].
     */
    LwU32       refVoltageuV;
    /*!
     * BA2mW scale factor [unitless unsigned FXP 20.12 value].
     */
    LwUFXP20_12 ba2mW;
    /*!
     * Reference GPCCLK [MHz].
     */
    LwU32       gpcClkMHz;
    /*!
     * Reference UTILSCLK [MHz].
     */
    LwU32       utilsClkMHz;
} LW2080_CTRL_PMGR_PWR_EQUATION_INFO_DATA_BA1X_SCALE;

/*!
 * Structure of data specific to the the BA v1.5 Scale power equation.
 */
typedef struct LW2080_CTRL_PMGR_PWR_EQUATION_INFO_DATA_BA15_SCALE {
    /*!
     * Reference voltage [uV].
     */
    LwU32       refVoltageuV;
    /*!
     * BA2mW scale factor [unitless unsigned FXP 20.12 value].
     */
    LwUFXP20_12 ba2mW;
    /*!
     * Reference GPCCLK [MHz].
     */
    LwU32       gpcClkMHz;
    /*!
     * Reference UTILSCLK [MHz].
     */
    LwU32       utilsClkMHz;
    /*!
     * Index into Power Equation Table (PWR_EQUATION) for the Dynamic equation.
     */
    LwU8        dynamicEquIdx;
    /*!
     * Max voltage [uV] that can be applied on given system. Used to ensure that
     * we do not overflow during the computation of the 8-bit scaling factor.
     */
    LwU32       maxVoltageuV;
} LW2080_CTRL_PMGR_PWR_EQUATION_INFO_DATA_BA15_SCALE;

/*!
 * Structure of data specific to the the DYNAMIC 1.x power equation.
 */
typedef struct LW2080_CTRL_PMGR_PWR_EQUATION_INFO_DATA_DYNAMIC {
    /*!
     * PWR_EQUATION_DYNAMIC does not have any type specific data.
     */
    LwU8 rsvd;
} LW2080_CTRL_PMGR_PWR_EQUATION_INFO_DATA_DYNAMIC;

/*!
 * Structure of data specific to the the DYNAMIC_10  power equation.
 */
typedef struct LW2080_CTRL_PMGR_PWR_EQUATION_INFO_DATA_DYNAMIC_10 {
    /*!
     * DYNAMIC 1.x super struct.
     */
    LW2080_CTRL_PMGR_PWR_EQUATION_INFO_DATA_DYNAMIC super;
    /*!
     * Voltage scaling exponent for dynamic current (mA) in UFXP20.12 unitless
     */
    LwUFXP20_12                                     dynLwrrentVoltExp;
} LW2080_CTRL_PMGR_PWR_EQUATION_INFO_DATA_DYNAMIC_10;

/*!
 * Type-specific data.
 */


/*!
 * Structure describing a PWR_EQUATION object, which specifies power equations.
 */
typedef struct LW2080_CTRL_PMGR_PWR_EQUATION_INFO {
    /*!
     * LW2080_CTRL_PMGR_PWR_EQUATION_TYPE_<xyz>
     */
    LwU8 type;

    /*!
     * Union of type-specific data.
     */
    union {
        LW2080_CTRL_PMGR_PWR_EQUATION_INFO_DATA_LEAKAGE        leakage;
        LW2080_CTRL_PMGR_PWR_EQUATION_INFO_DATA_LEAKAGE_DTCS11 dtcs11;
        LW2080_CTRL_PMGR_PWR_EQUATION_INFO_DATA_BA1X_SCALE     ba1xScale;
        LW2080_CTRL_PMGR_PWR_EQUATION_INFO_DATA_BA15_SCALE     ba15Scale;
        LW2080_CTRL_PMGR_PWR_EQUATION_INFO_DATA_LEAKAGE_DTCS12 dtcs12;
        LW2080_CTRL_PMGR_PWR_EQUATION_INFO_DATA_LEAKAGE_DTCS13 dtcs13;
        LW2080_CTRL_PMGR_PWR_EQUATION_INFO_DATA_DYNAMIC_10     dyn10;
    } data;
} LW2080_CTRL_PMGR_PWR_EQUATION_INFO;

/*!
 * LW2080_CTRL_CMD_PMGR_PWR_EQUATION_GET_INFO
 *
 * This command returns the PWR_EQUATION static information as specified by the
 * Power Equation Table.
 *
 * https://wiki.lwpu.com/engwiki/index.php/Resman/PState/Data_Tables/Power_Tables/Power_Equation_Table_1.X
 *
 * See LW2080_CTRL_PMGR_PWR_EQUATION_GET_INFO_PARAMS for documentation on
 * the parameters
 *
 * Possible status values returned are
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW2080_CTRL_CMD_PMGR_PWR_EQUATION_GET_INFO (0x20802616U) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PMGR_INTERFACE_ID << 8) | LW2080_CTRL_PMGR_PWR_EQUATION_GET_INFO_PARAMS_MESSAGE_ID" */

/*!
 * Structure representing the static state information associated with the GPU's
 * PWR_EQUATION power functionality.
 */
#define LW2080_CTRL_PMGR_PWR_EQUATION_GET_INFO_PARAMS_MESSAGE_ID (0x16U)

typedef struct LW2080_CTRL_PMGR_PWR_EQUATION_GET_INFO_PARAMS {
    /*!
     * [out] - Data describing IDDQ used for equation evaluation
     */
    struct {
        /*!
         * [out] - IDDQ_VERSION VFIELD/fuse value read from the GPU.  Must match the VBIOS
         * value @ref iddqVersion if reading the IDDQ VFIELD/fuse value.
         */
        LwU32 versionHw;
        /*!
         * [out] - IDDQ version from the VBIOS.  Must match the IDDQ_VERSION VFIELD/fuse
         * value @ref hwIddqVersion if reading IDDQ VFIELD/fuse.
         */
        LwU32 version;
        /*!
         * [out] - IDDQ value specified in [mA]
         */
        LwU32 valuemA;
    } iddq;
    /*!
     * [out] - Mask of PWR_EQUATION entries/equations specified on this GPU.
     */
    LwU32                              equationMask;
    /*!
     * [out] - Array of PWR_EQUATION entries/equations.  Has valid indexes corresponding
     * to the bits set in @ref equationMask.
     */
    LW2080_CTRL_PMGR_PWR_EQUATION_INFO equations[LW2080_CTRL_PMGR_PWR_EQUATION_MAX_EQUATIONS];
} LW2080_CTRL_PMGR_PWR_EQUATION_GET_INFO_PARAMS;

/*!
 * Enumeration of possible piecewise frequency floor sources for DOMGRP power policies.
 */
#define LW2080_CTRL_PMGR_PWR_POLICY_DOMGRP_PFF_SOURCE_NONE    0x00U
#define LW2080_CTRL_PMGR_PWR_POLICY_DOMGRP_PFF_SOURCE_THERMAL 0x01U
#define LW2080_CTRL_PMGR_PWR_POLICY_DOMGRP_PFF_SOURCE_POWER   0x02U

/*!
 * Enumeration of possible piecewise frequency floor tuple types.
 */
#define LW2080_CTRL_PMGR_PFF_TUPLE_DOMAIN_TYPE_GENERIC        0x0U
#define LW2080_CTRL_PMGR_PFF_TUPLE_DOMAIN_TYPE_TEMPERATURE    0x1U
#define LW2080_CTRL_PMGR_PFF_TUPLE_DOMAIN_TYPE_POWER          0x2U

/*!
 * Max number of uncapped cycles before we switch to RELAXED SLEEP MODE
 */
#define LW2080_CTRL_PMGR_PWR_POLICY_3X_MAX_UNCAPPED_CYCLES    0x3U

/*!
 * Number of modifiable frequency floor tuples (thermal/power) used to define piecewise linear frequency floor.
 */
#define LW2080_CTRL_PMGR_PFF_TUPLES_MAX                       3U

/*!
 * Union representing the different possible domains a pff tuple component may use.
 */


/*!
 * Structure representing a generic tuple which partially
 * defines a piecewise linear frequency flooring lwrve.
 * At the moment this is shared between INFO, STATUS, and CONTROL
 * interfaces.
 */
typedef struct LW2080_CTRL_PMGR_PFF_TUPLE {
    /*!
     * Type of the observed component of the tuple
     * @ref LW2080_CTRL_PMGR_PFF_TUPLE_DOMAIN_TYPE_*
     */
    LwU8  type;

    /*!
     * Range (frequency) component of the tuple (Y-Value)
     */
    LwU32 freqkHz;

    /*!
     * Domain component of the tuple (X-Value)
     */
    union {
        /*!
         * Generic domain component of the tuple
         * Assuming that all components in this union are sizeof(LwU32)
         */
        LwU32  value;

        /*!
         * Power component of the tuple (for power policies)
         */
        LwU32  powermW;

        /*!
         * Temperature component of the tuple (for thermal policies)
         */
        LwTemp temperature;
    } domain;
} LW2080_CTRL_PMGR_PFF_TUPLE;
typedef struct LW2080_CTRL_PMGR_PFF_TUPLE *PLW2080_CTRL_PMGR_PFF_TUPLE;

/*!
 * Structure representing the static POR info of a generic
 * tuple which partially defines a piecewise linear frequency
 * flooring lwrve.
 */
typedef struct LW2080_CTRL_PMGR_PFF_TUPLE_INFO {
    /*!
     * VBIOS POR for tuple frequency is defined as a vpstate index; this is the index
     * that will be translated during INIT to absolute frequencies for the `tuple`
     * below.
     */
    LwU8                       freqVpstateIdx;

    /*!
     * Structure representing a generic tuple which partially
     * defines a piecewise linear frequency flooring lwrve.
     */
    LW2080_CTRL_PMGR_PFF_TUPLE tuple;
} LW2080_CTRL_PMGR_PFF_TUPLE_INFO;
typedef struct LW2080_CTRL_PMGR_PFF_TUPLE_INFO *PLW2080_CTRL_PMGR_PFF_TUPLE_INFO;

/*!
 * Structure representing a POR VBIOS settings for the piecewise linear
 * frequency flooring lwrve.
 */
typedef struct LW2080_CTRL_PMGR_PFF_LWRVE_INFO {
    /*!
     * Array of tuples composing the modifiable section of the pff lwrve
     */
    LW2080_CTRL_PMGR_PFF_TUPLE_INFO tuples[LW2080_CTRL_PMGR_PFF_TUPLES_MAX];
} LW2080_CTRL_PMGR_PFF_LWRVE_INFO;
typedef struct LW2080_CTRL_PMGR_PFF_LWRVE_INFO *PLW2080_CTRL_PMGR_PFF_LWRVE_INFO;

/*!
 * Structure representing a POR VBIOS settings for the piecewise linear
 * frequency flooring lwrve.
 */
typedef struct LW2080_CTRL_PMGR_PFF_LWRVE_STATUS {
    /*!
     * Array of tuples composing the modifiable section of the pff lwrve.
     * These tuples will represent the runtime state of the lwrve which
     * is actually getting evaluated. Status = Control + Other runtime variables
     * such as OC.
     */
    LW2080_CTRL_PMGR_PFF_TUPLE tuples[LW2080_CTRL_PMGR_PFF_TUPLES_MAX];
} LW2080_CTRL_PMGR_PFF_LWRVE_STATUS;
typedef struct LW2080_CTRL_PMGR_PFF_LWRVE_STATUS *PLW2080_CTRL_PMGR_PFF_LWRVE_STATUS;

/*!
 * Structure representing controllable parameters of a piecewise linear
 * frequency flooring lwrve.
 */
typedef struct LW2080_CTRL_PMGR_PFF_LWRVE_CONTROL {
    /*!
     * Array of tuples composing the modifiable section of the pff lwrve
     */
    LW2080_CTRL_PMGR_PFF_TUPLE tuples[LW2080_CTRL_PMGR_PFF_TUPLES_MAX];
} LW2080_CTRL_PMGR_PFF_LWRVE_CONTROL;
typedef struct LW2080_CTRL_PMGR_PFF_LWRVE_CONTROL *PLW2080_CTRL_PMGR_PFF_LWRVE_CONTROL;

/*!
 * Structure representing INFO of a PFF interface. To be used in all
 * INFO RMCTRLs of implementing policies (THRM or PMGR).
 */
typedef struct LW2080_CTRL_PMGR_PFF_INFO {
    /*!
     * VBIOS setting for whether the PFF flooring functionality is enabled or not.
     */
    LwBool                          bFlooringEnabled;

    /*!
     * The piecewise frequency floor lwrve. This lwrve will be intersected at
     * a particular observed domain value to prolwce a frequency floor value
     */
    LW2080_CTRL_PMGR_PFF_LWRVE_INFO lwrve;
} LW2080_CTRL_PMGR_PFF_INFO;
typedef struct LW2080_CTRL_PMGR_PFF_INFO *PLW2080_CTRL_PMGR_PFF_INFO;

/*!
 * Structure representing the dynamic state data of a piecewise frequency floor
 * lwrve. To be used in all STATUS RMCTRLs of implementing policies (THRM or PMGR).
 */
typedef struct LW2080_CTRL_PMGR_PFF_STATUS {
    /*!
     * Last evaluated GPC frequency floor of the pff.
     *
     * This represents the output value of the lwrve at the intersection
     * of an observed domain value (power, temperature, etc.). VFMax if
     * Observed power/temperature is less than limitLwrr.
     */
    LwU32                             lastFloorkHz;

    /*!
     * The piecewise frequency floor lwrve with all runtime OC offsets applied.
     */
    LW2080_CTRL_PMGR_PFF_LWRVE_STATUS lwrve;
} LW2080_CTRL_PMGR_PFF_STATUS;
typedef struct LW2080_CTRL_PMGR_PFF_STATUS *PLW2080_CTRL_PMGR_PFF_STATUS;

/*!
 * Structure representing PMU specific CONTROL and construction parameters of a PFF interface.
 * To be used in all PMU implementing policies (THRM or PMGR).
 */
typedef struct LW2080_CTRL_PMGR_PFF_PMU_CONTROL {
    /*!
     * Control whether the PFF flooring functionality is enabled or not.
     * This should be initialized to LW2080_CTRL_PMGR_PFF_INFO::bFlooringEnabled
     * and not be allowed to change at runtime.
     */
    LwBool                             bFlooringEnabled;

    /*!
     * The piecewise frequency floor lwrve. This lwrve will be intersected at
     * a particular observed domain value to prolwce a frequency floor value
     */
    LW2080_CTRL_PMGR_PFF_LWRVE_CONTROL lwrve;
} LW2080_CTRL_PMGR_PFF_PMU_CONTROL;
typedef struct LW2080_CTRL_PMGR_PFF_PMU_CONTROL *PLW2080_CTRL_PMGR_PFF_PMU_CONTROL;

/*!
 * Structure representing CONTROL parameters of a PFF interface. To be  colwerted to
 * LW2080_CTRL_PMGR_PFF_PMU_CONTROL and passed to pmu as part of pmuDataInit for all
 * PFF enabled policies (THRM or PMGR).
 */
typedef struct LW2080_CTRL_PMGR_PFF_RM_CONTROL {
    /*!
     * [SET] Flag to denote that the pff lwrve has changed since the GET_CONTROL was
     * called, this will apply changes in `lwrve` and disable automatic scaling
     * with respect to limitLwrr. This is set to LW_FALSE on a GET_CONTROL
     */
    LwBool                             bPffLwrveChanged;

    /*!
     * Modifiable PFF lwrve and feature enable/disable-ment
     */
    LW2080_CTRL_PMGR_PFF_LWRVE_CONTROL lwrve;
} LW2080_CTRL_PMGR_PFF_RM_CONTROL;
typedef struct LW2080_CTRL_PMGR_PFF_RM_CONTROL *PLW2080_CTRL_PMGR_PFF_RM_CONTROL;

/*!
 * Maximum number of RM_PMU_PMGR_PWR_POLICY entries which can be supported in
 * the RM or PMU
 */
#define LW2080_CTRL_PMGR_PWR_POLICY_MAX_POLICIES                                32U

/*!
 * Special value corresponding to an invalid Power Policy index.  This value
 * means that policy is not specified.
 */
#define LW2080_CTRL_PMGR_PWR_POLICY_INDEX_ILWALID                               LW2080_CTRL_BOARDOBJ_IDX_ILWALID_8BIT

/*!
 * Special value corresponding to an invalid Power Policy Relationship index.
 * This value means the Policy has no child Power Policies.
 */
#define LW2080_CTRL_PMGR_PWR_POLICY_RELATIOSHIP_INDEX_ILWALID                   LW2080_CTRL_BOARDOBJ_IDX_ILWALID_8BIT

/*!
 * The maximum theoretical Policy limit value. This value means
 * the value ought to be skipped when applying various Policy limits
 */
#define LW2080_CTRL_PMGR_PWR_POLICY_LIMIT_MAX                                   (LW_U32_MAX)

/*!
 * Macros encoding types/classes of PWR_POLICY entries.
 *
 * Implementation PWR_POLICY classes are indexed starting from 0x00.  Virtual
 * PWR_POLICY classes are indexed starting from 0xFF.
 */
#define LW2080_CTRL_PMGR_PWR_POLICY_TYPE_TOTAL_GPU                              0x00U
#define LW2080_CTRL_PMGR_PWR_POLICY_TYPE_WORKLOAD                               0x01U
#define LW2080_CTRL_PMGR_PWR_POLICY_TYPE_BANG_BANG_VF                           0x02U
#define LW2080_CTRL_PMGR_PWR_POLICY_TYPE_PROP_LIMIT                             0x03U
#define LW2080_CTRL_PMGR_PWR_POLICY_TYPE_HW_THRESHOLD                           0x04U
#define LW2080_CTRL_PMGR_PWR_POLICY_TYPE_MARCH_VF                               0x07U
#define LW2080_CTRL_PMGR_PWR_POLICY_TYPE__DEPRECATED_DO_NOT_REUSE               0x08U
#define LW2080_CTRL_PMGR_PWR_POLICY_TYPE_BALANCE                                0x09U
#define LW2080_CTRL_PMGR_PWR_POLICY_TYPE_GEMINI                                 0x0AU
#define LW2080_CTRL_PMGR_PWR_POLICY_TYPE_WORKLOAD_MULTIRAIL                     0x0BU
#define LW2080_CTRL_PMGR_PWR_POLICY_TYPE_RP_PM_TGP_1X                           0x0LW // Deprecated
#define LW2080_CTRL_PMGR_PWR_POLICY_TYPE_RP_PM_CWC_1X                           0x0DU // Deprecated
#define LW2080_CTRL_PMGR_PWR_POLICY_TYPE_RP_PM_FB_1X                            0x0EU // Deprecated
#define LW2080_CTRL_PMGR_PWR_POLICY_TYPE_WORKLOAD_SINGLE_1X                     0x0FU
#define LW2080_CTRL_PMGR_PWR_POLICY_TYPE_WORKLOAD_COMBINED_1X                   0x10U
#define LW2080_CTRL_PMGR_PWR_POLICY_TYPE_MARCH                                  0xFBU
#define LW2080_CTRL_PMGR_PWR_POLICY_TYPE_DOMGRP                                 0xFDU
#define LW2080_CTRL_PMGR_PWR_POLICY_TYPE_LIMIT                                  0xFEU
#define LW2080_CTRL_PMGR_PWR_POLICY_TYPE_UNKNOWN                                0xFFU

/*!
 * Enumeration of PWR_POLICY's INTERFACE types.
 */
#define LW2080_CTRL_PMGR_PWR_POLICY_INTERFACE_TYPE_RP_PM_1X                     0x01U // Deprecated
#define LW2080_CTRL_PMGR_PWR_POLICY_INTERFACE_TYPE_WORKLOAD_MULTIRAIL_INTERFACE 0x02U
#define LW2080_CTRL_PMGR_PWR_POLICY_INTERFACE_TYPE_TOTAL_GPU_INTERFACE          0x03U
#define LW2080_CTRL_PMGR_PWR_POLICY_INTERFACE_TYPE_PERF_CF_PWR_MODEL            0x04U

/*!
 * Macros encoding units of limits in PWR_POLICY entries.
 */
#define LW2080_CTRL_PMGR_PWR_POLICY_LIMIT_UNIT_POWER_MW                         0x00000000U
#define LW2080_CTRL_PMGR_PWR_POLICY_LIMIT_UNIT_LWRRENT_MA                       0x00000001U

/*!
 * Special PWR_POLICY/client indexes for the @ref
 * LW2080_CTRL_PMGR_PWR_POLICY_LIMIT_INPUT::pwrPolicyIdx.
 * Following index ranges are allocated / reserved for:
 *
 * [ 0x00 - 0x1F ]:
 *      - directly map PWR_POLICY objects' indexes
 * [ 0x20 - 0x3F ]:
 *      - map PWR_VIOLATION objects' indexes using mapping macro
 *        LW2080_CTRL_PMGR_PWR_POLICY_LIMIT_INPUT_CLIENT_IDX_PWR_VIOLATION()
 * [ 0xA0 - 0xBF ]:
 *      - map THERM_POLICY objects' indexes using mapping macro
 *        LW2080_CTRL_PMGR_PWR_POLICY_LIMIT_INPUT_CLIENT_IDX_THERM_POLICY()
 *
 * The client indexes counting down from 0xFF are reserved for special purposes,
 * as defined below.
 *
 * _RM - The policy value specified/desired by the RM PWR code.
 *      In the RM layer, this entry holds the default/desired value from the
 *      Power Policy Table and any user/client-requested tweaks (via LWAPI /
 *      RMCTRL) to the PWR_POLICY value.  In the PMU layer, this is the
 *      arbitrated output from the RM layer, as sent to the PMU.
 * _PPO - The policy value specified by the Power Policy Override (PPO) table,
 *      as parsed by the RM.  This is used to allow the client the ability
 *      to tweak PWR_POLICY POR values without touching the Power Policy Table.
 * _BATT - The policy value specified as the Battery Limit in the Power Policy
 *      Table.  This client is engaged by the RM on AC->BATT transitions.
 * _MXM - The policy value specified by the MXM-SIS, used to set the AC and
 *      Battery limits at init time.
 * _DNOTIFIER - The policy value specified for the D2 - D5 power states,
 *      which allow the system to impose a GPU or board power limit.
 * _SMBUS - The policy exposed to the SMBus interface, which allows the SMBus
 *      Master to read a power channel referenced by the exposed GPU power
 *      controller and also impose a power limit to the exposed controller.
 * _KERNEL - The policy exposed to the Kernel mode client interface to query,
 *      get and  set limits.
 * _LWPCF - The policy exposed to LWPCF module interface to query,
 *      get limits and set soft limits from its controllers (PPAB_SOFT)
 * _EDPP - The EDPPeak power policy index exposed to the system to
 *      impose a peak current limit on the GPU
 * _TGP_TURBO - The TGP Turbo (aka TGP mode 2) client exposed to the system to
 *      allow a higher TGP operating limit.
 * _BATT_MAX - The policy value specified as the Battery Max Limit in the Power Policy
 *      Table.  This client is engaged by the RM on AC->BATT transitions.
 *
 * _SPECIAL_MIN - Must be defined to be equal the minimum of
 *     LW2080_CTRL_PMGR_PWR_POLICY_LIMIT_INPUT_CLIENT_IDX_<xyz>.
 *     It is used to verify validity of the presented index.
 */
#define LW2080_CTRL_PMGR_PWR_POLICY_LIMIT_INPUT_CLIENT_IDX_RM                   0xFEU
#define LW2080_CTRL_PMGR_PWR_POLICY_LIMIT_INPUT_CLIENT_IDX_PPO                  0xFDU
#define LW2080_CTRL_PMGR_PWR_POLICY_LIMIT_INPUT_CLIENT_IDX_BATT                 0xFLW
#define LW2080_CTRL_PMGR_PWR_POLICY_LIMIT_INPUT_CLIENT_IDX_MXM                  0xFBU
#define LW2080_CTRL_PMGR_PWR_POLICY_LIMIT_INPUT_CLIENT_IDX_DNOTIFIER            0xFAU
#define LW2080_CTRL_PMGR_PWR_POLICY_LIMIT_INPUT_CLIENT_IDX_SMBUS                0xF9U
#define LW2080_CTRL_PMGR_PWR_POLICY_LIMIT_INPUT_CLIENT_IDX_KERNEL               0xF8U
#define LW2080_CTRL_PMGR_PWR_POLICY_LIMIT_INPUT_CLIENT_IDX_LWPCF                0xF7U
#define LW2080_CTRL_PMGR_PWR_POLICY_LIMIT_INPUT_CLIENT_IDX_EDPP                 0xF6U
#define LW2080_CTRL_PMGR_PWR_POLICY_LIMIT_INPUT_CLIENT_IDX_TGP_TURBO            0xF5U
#define LW2080_CTRL_PMGR_PWR_POLICY_LIMIT_INPUT_CLIENT_IDX_LWPCF_HARD           0xF4U
#define LW2080_CTRL_PMGR_PWR_POLICY_LIMIT_INPUT_CLIENT_IDX_BATT_MAX             0xF3U
#define LW2080_CTRL_PMGR_PWR_POLICY_LIMIT_INPUT_CLIENT_IDX_SPECIAL_MIN          0xF3U

#define LW2080_CTRL_PMGR_PWR_POLICY_LIMIT_INPUT_CLIENT_IDX_THERM_POLICY_BASE    0xA0U
#define LW2080_CTRL_PMGR_PWR_POLICY_LIMIT_INPUT_CLIENT_IDX_THERM_POLICY_NUM     32U
#define LW2080_CTRL_PMGR_PWR_POLICY_LIMIT_INPUT_CLIENT_IDX_THERM_POLICY_MAX     (0xbfU) /* finn: Evaluated from "(LW2080_CTRL_PMGR_PWR_POLICY_LIMIT_INPUT_CLIENT_IDX_THERM_POLICY_BASE + LW2080_CTRL_PMGR_PWR_POLICY_LIMIT_INPUT_CLIENT_IDX_THERM_POLICY_NUM - 1)" */
#define LW2080_CTRL_PMGR_PWR_POLICY_LIMIT_INPUT_CLIENT_IDX_THERM_POLICY(i)       \
    (LW2080_CTRL_PMGR_PWR_POLICY_LIMIT_INPUT_CLIENT_IDX_THERM_POLICY_BASE + (i))

#define LW2080_CTRL_PMGR_PWR_POLICY_LIMIT_INPUT_CLIENT_IDX_PWR_VIOLATION_BASE   0x20U
#define LW2080_CTRL_PMGR_PWR_POLICY_LIMIT_INPUT_CLIENT_IDX_PWR_VIOLATION_NUM    32U
#define LW2080_CTRL_PMGR_PWR_POLICY_LIMIT_INPUT_CLIENT_IDX_PWR_VIOLATION_MAX    (0x3fU) /* finn: Evaluated from "(LW2080_CTRL_PMGR_PWR_POLICY_LIMIT_INPUT_CLIENT_IDX_PWR_VIOLATION_BASE + LW2080_CTRL_PMGR_PWR_POLICY_LIMIT_INPUT_CLIENT_IDX_PWR_VIOLATION_NUM - 1)" */
#define LW2080_CTRL_PMGR_PWR_POLICY_LIMIT_INPUT_CLIENT_IDX_PWR_VIOLATION(i)      \
    (LW2080_CTRL_PMGR_PWR_POLICY_LIMIT_INPUT_CLIENT_IDX_PWR_VIOLATION_BASE + (i))

/*!
 * Power Policy 3.x Only
 * Supported input filter functions in PWR_POLICY.
 */
#define LW2080_CTRL_PMGR_PWR_POLICY_3X_FILTER_TYPE_NONE                         0x00U
#define LW2080_CTRL_PMGR_PWR_POLICY_3X_FILTER_TYPE_BLOCK                        0x01U
#define LW2080_CTRL_PMGR_PWR_POLICY_3X_FILTER_TYPE_MOVING                       0x02U
#define LW2080_CTRL_PMGR_PWR_POLICY_3X_FILTER_TYPE_IIR                          0x03U

/*!
 * Power Policy 3.x Only
 * Block Average type filter info as an input filter structure
 * in LW2080_CTRL_PMGR_PWR_POLICY_3X_INFO.
 */
typedef struct LW2080_CTRL_PMGR_PWR_POLICY_3X_FILTER_INFO_BLOCK {
    /*!
     * Block size. This Policy should aggregate this num of data before
     * evaluation. After evaluation is done, data within block will be cleared
     * and will aggregate from 0 sample again.
     * Lwrrently, the function to be applied on this block is "Averaging"
     * i.e. output value will be average of all data within block.
     */
    LwU8 blockSize;
} LW2080_CTRL_PMGR_PWR_POLICY_3X_FILTER_INFO_BLOCK;

/*!
 * Power Policy 3.x Only
 * Moving Average type filter info as an input filter structure
 * in LW2080_CTRL_PMGR_PWR_POLICY_3X_INFO.
 */
typedef struct LW2080_CTRL_PMGR_PWR_POLICY_3X_FILTER_INFO_MOVING {
    /*!
     * Window size. This Policy will evaluate every time a new sample
     * comes in, however the new sample will replace oldest sample within
     * this window. Policy will be evaluated based on data within this
     * window.
     * Lwrrently, the function to be applied on this window is "Averaging"
     * i.e. output value will be average of all data within window.
     */
    LwU8 windowSize;
} LW2080_CTRL_PMGR_PWR_POLICY_3X_FILTER_INFO_MOVING;

/*!
 * Power Policy 3.x Only
 * Infinite Impulse Response (IIR) type filter info as an
 * input filter structure in LW2080_CTRL_PMGR_PWR_POLICY_3X_INFO.
 */
typedef struct LW2080_CTRL_PMGR_PWR_POLICY_3X_FILTER_INFO_IIR {
    /*!
     * Divisor. This Policy will be evaluated once a new sample comes in. The
     * value to be used in evaluation is:
     * new_Out = old_Out * (divisor - 1) / divisor + sample / divisor.
     * The divisor will control how responsive new_Out catches change in new
     * sample.
     */
    LwU8 divisor;
} LW2080_CTRL_PMGR_PWR_POLICY_3X_FILTER_INFO_IIR;

/*!
 * Power Policy 3.x Only
 * Union of all possible power policy filter info data structure.
 */


/*!
 * Power Policy 3.x Only
 * Structure represents the input filter info in PWR_POLICY
 */
typedef struct LW2080_CTRL_PMGR_PWR_POLICY_3X_FILTER_INFO {
    /*!
     * @ref LW2080_CTRL_PMGR_PWR_POLICY_3X_FILTER_TYPE_<xyz>
     */
    LwU8 type;
    /*!
     * @ref LW2080_CTRL_PMGR_PWR_POLICY_3X_FILTER_INFO_<xyz>
     */
    union {
        LW2080_CTRL_PMGR_PWR_POLICY_3X_FILTER_INFO_BLOCK  block;
        LW2080_CTRL_PMGR_PWR_POLICY_3X_FILTER_INFO_MOVING moving;
        LW2080_CTRL_PMGR_PWR_POLICY_3X_FILTER_INFO_IIR    iir;
    } params;
} LW2080_CTRL_PMGR_PWR_POLICY_3X_FILTER_INFO;

/*!
 * Structure representing an input/requested PWR_POLICY limit value from other
 * clients (other PWR_POLICYs or the RM).  Each PWR_POLICY object arbitrates
 * between its requested values to apply the lowest value.
 */
typedef struct LW2080_CTRL_PMGR_PWR_POLICY_LIMIT_INPUT {
    /*!
     * PWR_POLICY index for the client which requested this limit value.
     *
     * @note Special value LW2080_CTRL_PMGR_PWR_POLICY_LIMIT_INPUT_POLICY_IDX_RM
     * is used to denote the value requested by the RM.
     *
     * ~~~SPTODO~~~: Change this to "clientIdx".
     */
    LwU8  pwrPolicyIdx;
    /*!
     * Limit value requested by the client.  In units as specified in @ref
     * LW2080_CTRL_PMGR_PWR_POLICY_INFO::limitUnit.
     */
    LwU32 limitValue;
} LW2080_CTRL_PMGR_PWR_POLICY_LIMIT_INPUT;

/*!
 * Maximum number of LW2080_CTRL_PMGR_PWR_POLICY_LIMIT_INPUT structures a single
 * PWR_POLICY object can support.  This is the maximum number of clients which
 * can request a limit value for a given PWR_POLICY.
 */
#define LW2080_CTRL_PMGR_PWR_POLICY_MAX_LIMIT_INPUTS 0x8U

/*!
 * Structure representing the current set of active
 * LW2080_CTRL_PMGR_PWR_POLICY_LIMIT_INPUTs, by which clients can request limit
 * values for a given PWR_POLICY.  The PWR_POLICY will arbitrate between all
 * these entries (picking the lowest value) to determine the current limit
 * value.
 */
typedef struct LW2080_CTRL_PMGR_PWR_POLICY_LIMIT_ARBITRATION {
    /*!
     * Arbitration function - either max (LW_TRUE) or min (LW_FALSE).
     */
    LwBool                                  bArbMax;
    /*!
     * Current number of active limits.  Will always be <= @ref
     * LW2080_CTRL_PMGR_PWR_POLICY_MAX_LIMIT_INPUTS.
     */
    LwU8                                    numInputs;
    /*!
     * Arbitrated output value.
     */
    LwU32                                   output;
    /*!
     * Array of LW2080_CTRL_PMGR_PWR_POLICY_LIMIT_INPUT entries.  Has valid
     * indexes in the range of [0, @ref numInputs).
     */
    LW2080_CTRL_PMGR_PWR_POLICY_LIMIT_INPUT inputs[LW2080_CTRL_PMGR_PWR_POLICY_MAX_LIMIT_INPUTS];
} LW2080_CTRL_PMGR_PWR_POLICY_LIMIT_ARBITRATION;

/*!
 * Retrieves the arbitrated output of an
 * @ref LW2080_CTRL_PMGR_PWR_POLICY_LIMIT_ARBITRATION structure.
 *
 * @param[in]   pLimitArbitration   @ref LW2080_CTRL_PMGR_PWR_POLICY_LIMIT_ARBITRATION
 *                                  pointer from which to retrieve the arbirated
 *                                  output.
 *
 * @return  Arbitrated output value of pLimitArbitration
 */
#define LW2080_CTRL_PMGR_PWR_POLICY_LIMIT_ARBITRATION_OUTPUT_GET(pLimitArbitration) \
    ((pLimitArbitration)->output)

/*!
 * Structure of static information specific to the
 * Integral Control info in the power policy.
 */
typedef struct LW2080_CTRL_PMGR_PWR_POLICY_INFO_INTEGRAL {
    /*!
     * No of past samples, looking at which the control
     * algorithm does an adjustment to the future samples.
     */
    LwU8       pastSampleCount;
    /*!
     * No of future samples to apply the limit adjustment.
     */
    LwU8       nextSampleCount;
    /*!
     * The minimum value of the bounding box for the limit
     * adjustment, a ratio from the current policy limit.
     *
     * Unitless Unsigned FXP 4.12.
     */
    LwUFXP4_12 ratioLimitMin;
    /*!
     * The maximum value of the bounding box for the limit
     * adjustment, a ratio from the current policy limit.
     *
     * Unitless Unsigned FXP 4.12.
     */
    LwUFXP4_12 ratioLimitMax;
} LW2080_CTRL_PMGR_PWR_POLICY_INFO_INTEGRAL;

/*!
 * Structure defining the piecewise frequency floor source information
 * to be used by PWR_POLICY_DOMGRP derived classes.
 */
typedef struct LW2080_CTRL_PMGR_PWR_POLICY_INFO_DATA_DOMGRP_PFF_SOURCE {
    /*!
     * Refer to LW2080_CTRL_PMGR_PWR_POLICY_DOMGRP_PFF_SOURCE_*.
     * This is the source policy type (none/thermal/power) that can provide a
     * piecewise frequency floor.
     */
    LwU8 policyType;

    /*!
     * Index into either power policy table or thermal policy table of the policy
     * that can will supply the frequency floor. If policyType == NONE
     * this index should be @ref LW2080_CTRL_PMGR_PWR_POLICY_INDEX_ILWALID.
     */
    LwU8 policyIdx;
} LW2080_CTRL_PMGR_PWR_POLICY_INFO_DATA_DOMGRP_PFF_SOURCE;
typedef struct LW2080_CTRL_PMGR_PWR_POLICY_INFO_DATA_DOMGRP_PFF_SOURCE *PLW2080_CTRL_PMGR_PWR_POLICY_INFO_DATA_DOMGRP_PFF_SOURCE;

/*!
 * Structure of static information specific to the DOMGRP power policy.  This
 * Power Policy is a virtual/super class, which is extended/implemented by other
 * Power Policy clases (e.g. WORKLOAD and BANG_BANG_VF).
 */
typedef struct LW2080_CTRL_PMGR_PWR_POLICY_INFO_DATA_DOMGRP {
    /*!
     * A boolean flag to indicate that the output Domain Group limits computed
     * by this Power Policy (via @ref pwrPolicyDomGrpEvaluate) should be floored
     * to the 3D Boost VPstate (commonly referred to as "Base Clock" in the GPU
     * Boost/SmartPower/PWR 2.0 documentation).
     */
    LwBool                                                  b3DBoostVpstateFloor;

    /*!
     * Cap the system below the "Inflection vpstate index" when the current
     * limit is smaller than this "Inflection limit". This inflection limit can
     * help improve some pstate thrashing issue when the power limit is reduced
     * into the "battery" or certain lower pstate range.
     */
    LwU32                                                   limitInflection;

    /*!
     * Under some cirlwmstances, we want to cap the output of the Domain Group
     * limits computed by this power policy on the next cycle to the full
     * deflection point vpstate. This is to ensure that the power will be under
     * the target and will not overshoot. This bool indicates that this
     * behavior is enabled on this power policy.
     */
    LwBool                                                  bFullDeflectiolwpstate;

    /*!
     * Describes the source of the piecewise frequency floor lwrve that the
     * GPC Domain Group will use as a floor.
     */
    LW2080_CTRL_PMGR_PWR_POLICY_INFO_DATA_DOMGRP_PFF_SOURCE pffSource;
} LW2080_CTRL_PMGR_PWR_POLICY_INFO_DATA_DOMGRP;

/*!
 * Structure of static information specific to the BANG_BANG_VF power policy.
 * This Power Policy implements a bang-bang step controller along the VF lwrve.
 */
typedef struct LW2080_CTRL_PMGR_PWR_POLICY_INFO_DATA_BANG_BANG_VF {
    /*!
     * @copydoc LW2080_CTRL_PMGR_PWR_POLICY_INFO_DATA_DOMGRP
     *
     * Must always be first in structure!
     */
    LW2080_CTRL_PMGR_PWR_POLICY_INFO_DATA_DOMGRP domGrp;

    /*!
     * Ratio of the limit (PWR_POLICY::limitLwrr) below which the controlled
     * value (PWR_POLICY::valueLwrr) must fall in order for the Bang-Bang
     * algorithm to initiate the uncap action.
     *
     * Unitless Unsigned FXP 4.12.
     */
    LwU16                                        uncapLimitRatio;
} LW2080_CTRL_PMGR_PWR_POLICY_INFO_DATA_BANG_BANG_VF;

/*!
 * Enumerations/macros defining the different types of hysteresis values a
 * PWR_POLICY_MARCH may use.
 *
 * _RATIO - The hysteresis amount is a ratio of the limit value:
 *      uncapLimit = limitValue * ratio.
 *
 * _STATIC_VALUE - The hysteresis amount is a static value which is subtracted
 *      from the limit value:
 *      uncapLimit = limitValue - staticValue
 */
#define LW2080_CTRL_PMGR_PWR_POLICY_MARCH_HYSTERESIS_TYPE_RATIO        0x00U
#define LW2080_CTRL_PMGR_PWR_POLICY_MARCH_HYSTERESIS_TYPE_STATIC_VALUE 0x01U

/*!
 * Union of type-specific hysteresis values.
 */


/*!
 * Structure representing the hysteresis value for a PWR_POLICY_MARCH object.
 */
typedef struct LW2080_CTRL_PMGR_PWR_POLICY_MARCH_HYSTERESIS {
    /*!
     * @ref LW2080_CTRL_PMGR_PWR_POLICY_MARCH_HYSTERESIS_TYPE_<XYZ>
     */
    LwU8 type;
    /*!
     * Union of type specific data.  Interpreted by @ref type.
     */
    union {
        /*!
         * Ratio value corresponding to
         * LW2080_CTRL_PMGR_PWR_POLICY_MARCH_HYSTERESIS_TYPE_RATIO.
         */
        LwU16 ratio;
        /*!
         * Static value corresponding to
         * LW2080_CTRL_PMGR_PWR_POLICY_MARCH_HYSTERESIS_TYPE_STATIC_VALUE.
         */
        LwU16 staticValue;
    } data;
} LW2080_CTRL_PMGR_PWR_POLICY_MARCH_HYSTERESIS;

/*!
 * Union of type-specific static information data.
 */


/*!
 * Structure of static information specific to the MARCH power policy.  This is
 * a Power Policy Interface class which other Power Policy classes extend to
 * implement a full marching algorithm.
 */
typedef struct LW2080_CTRL_PMGR_PWR_POLICY_INFO_DATA_MARCH {
    /*!
     * @ref LW2080_CTRL_PMGR_PWR_POLICY_TYPE_MARCH
     */
    LwU8                                         type;
    /*!
     * Number of steps by which the implementing class should respond to various
     * actions.
     */
    LwU8                                         stepSize;
    /*!
     * Hysteresis amount for uncapping.
     */
    LW2080_CTRL_PMGR_PWR_POLICY_MARCH_HYSTERESIS hysteresis;

    /*!
     * Type-specific information.
     */
    union {
        LwU8 placeHolder;
    } data;
} LW2080_CTRL_PMGR_PWR_POLICY_INFO_DATA_MARCH;

/*!
 * Structure of static information specific to the MARCH_VF power policy.
 */
typedef struct LW2080_CTRL_PMGR_PWR_POLICY_INFO_DATA_MARCH_VF {
    /*!
     * @copydoc LW2080_CTRL_PMGR_PWR_POLICY_INFO_DATA_MARCH
     *
     * Must always be first in structure!
     */
    LW2080_CTRL_PMGR_PWR_POLICY_INFO_DATA_DOMGRP domGrp;
    /*!
     * @copydoc LW2080_CTRL_PMGR_PWR_POLICY_MARCH
     *
     * The common PWR_POLICY_MARCH initialization data.  This data does not need
     * to be at any fixed location, it is handled elsewhere.
     */
    LW2080_CTRL_PMGR_PWR_POLICY_INFO_DATA_MARCH  march;
} LW2080_CTRL_PMGR_PWR_POLICY_INFO_DATA_MARCH_VF;

/*!
 * Structure of static information specific to the PROP_LIMIT power policy.
 * This Power Policy will update the limits of all Power Policies corresponding
 * to the specified Power Policy Relationships by the proportion of this
 * policy's value and limit.
 */
typedef struct LW2080_CTRL_PMGR_PWR_POLICY_INFO_DATA_PROP_LIMIT {
    /*!
     * Index of first Power Policy Relationship in the Power Policy Table.  This
     * PWR_POLICY_PROP_LIMIT will adjust the limits of all PWR_POLICYs
     * corresponding to the PWR_POLICY_RELATIONSHIPs in the range
     * [policyRelIdxFirst, @ref policyRelIdxLast].
     *
     * TODO: move to use LW2080_CTRL_PMGR_PWR_POLICY_RELATIONSHIP_SET
     */
    LwU8   policyRelIdxFirst;
    /*!
     * Index of last Power Policy Relationship in the Power Policy Table.  This
     * PWR_POLICY_PROP_LIMIT will adjust the limits of all PWR_POLICYs
     * corresponding to the PWR_POLICY_RELATIONSHIPs in the range
     * [policyRelIdxFirst, @ref policyRelIdxLast].
     *
     * TODO: move to use LW2080_CTRL_PMGR_PWR_POLICY_RELATIONSHIP_SET
     */
    LwU8   policyRelIdxLast;
    /*!
     * Boolean flag indicating whether "dummy" operation is desired for this
     * PWR_POLICY_PROP_LIMIT object.  When "dummy" operation is engaged, the
     * PWR_POLICY_PROP_LIMIT object will compute the desired limit requests for
     * all referenced (via PWR_POLICY_RELATIONSHIPs) PWR_POLICY objects, but
     * will only store them internally and not issue the requests.
     *
     * This functionality is useful for IRB the PWR_POLICY_PROP_LIMIT is
     * controlling a single phase of LWVDD, such that any limit can be enforced
     * entirely via IRB by shifting the phase away -
     * i.e. even a limit of 0 can be satisfied.
     */
    LwBool bDummy;
} LW2080_CTRL_PMGR_PWR_POLICY_INFO_DATA_PROP_LIMIT;

/*!
 * Union representing how we should interpret static power number.
 * Fixed value (in specified units): The value to subtract out of the total
 * available limit before assigning the remainder to Core and FB.
 * Power Channel Index: The index to a power channel representing power
 * consumption for static rail.
 */


/*!
 * Structure containing info for applying adjustment to policy limits based on
 * TOTAL GPU limit
 */
typedef struct LW2080_CTRL_PMGR_PWR_POLICY_INFO_TOTAL_GPU_IFACE_ADJUSTMENT {
    /*!
     * Boolean indicating that the adjustment behavior is enabled. This allows
     * adjustment only when increasing limits.
     */
    LwBool                                       bAdjEnabled;
    /*!
     * Boolean indicating that the adjustment is enabled for both
     * increasing and decreasing limits.
     * This boolean can only be enabled if @ref bAdjEnabled is enabled.
     */
    LwBool                                       bAdjBidirectionalEnabled;
    /*!
     * Set of Power Policy Relationships in the Power Policy Table,
     * specifying the set of PWR_POLICY objects to adjust as this object's limit
     * is changed by the user. The set is specified as the range
     * [@ref adjRelSet.policyRelStart, @ref adjRelSet.policyRelEnd].
     */
    LW2080_CTRL_PMGR_PWR_POLICY_RELATIONSHIP_SET adjRelSet;
} LW2080_CTRL_PMGR_PWR_POLICY_INFO_TOTAL_GPU_IFACE_ADJUSTMENT;

/*!
 * Structure of static information specific to the TOTAL_GPU Interface.
 */
typedef struct LW2080_CTRL_PMGR_PWR_POLICY_INFO_DATA_TOTAL_GPU_INTERFACE {
    /*!
     * BOARDOBJ_INTERFACE super class.  Must always be first element in the structure.
     */
    LW2080_CTRL_BOARDOBJ_INTERFACE                              super;

    /*!
     * Power Policy Table Relationship index corresponding to the FB Power
     * Policy to update for this Total GPU Power Policy.
     */
    LwU8                                                        fbPolicyRelIdx;
    /*!
     * Power Policy Table Relationship index corresponding to the Core Power
     * Policy to update for this Total GPU Power Policy.
     */
    LwU8                                                        corePolicyRelIdx;
    /*!
     * Boolean indicating if we use a Power Channel Index to represent static
     * power, or use the fixed value to represent the static power.
     */
    LwBool                                                      bUseChannelIdxForStatic;
    /*!
     * Structure containing info for applying adjustment to policy limits based
     * on TOTAL GPU limit
     */
    LW2080_CTRL_PMGR_PWR_POLICY_INFO_TOTAL_GPU_IFACE_ADJUSTMENT adjustment;
    /*!
     * Union for how we should evaluate static rail's power value.
     */
    union {
        LwU32 fixed;
        LwU8  pwrChannelIdx;
    } staticVal;
    /*!
     * Inflection point 0
     * Cap the system below the "Inflection vpstate index" when the current
     * limit is smaller than this "Inflection limit". This inflection limit can
     * help improve some pstate thrashing issue when the power limit is reduced
     * into the "battery" or certain lower pstate range.
     */
    LwU32 limitInflection0;
    /*!
     * Inflection point 1
     */
    LwU32 limitInflection1;
} LW2080_CTRL_PMGR_PWR_POLICY_INFO_DATA_TOTAL_GPU_INTERFACE;

/*!
 * Structure of static information specific to the TOTAL_GPU power policy.
 */
typedef struct LW2080_CTRL_PMGR_PWR_POLICY_INFO_DATA_TOTAL_GPU {
    /*!
     * [Deprecated]
     * Static value (in specified units) to subtract out of the total available
     * limit before assigning the remainder to Core and FB.
     * We will remove this value once all reference to it is cleaned.
     */
    LwU32                                                     staticValue;
    /*!
     * Inflection point 2
     */
    LwU32                                                     limitInflection2;
    /*!
     * VBIOS info concerning the pff interface.
     */
    LW2080_CTRL_PMGR_PFF_INFO                                 pff;

    /*
     * Total GPU interface
     */
    LW2080_CTRL_PMGR_PWR_POLICY_INFO_DATA_TOTAL_GPU_INTERFACE tgpIface;
} LW2080_CTRL_PMGR_PWR_POLICY_INFO_DATA_TOTAL_GPU;

/*!
 * Structure of static information specific to the WORKLOAD power policy.
 */
typedef struct LW2080_CTRL_PMGR_PWR_POLICY_INFO_DATA_WORKLOAD {
    /*!
     * @copydoc LW2080_CTRL_PMGR_PWR_POLICY_INFO_DATA_DOMGRP
     *
     * Must always be first in structure!
     */
    LW2080_CTRL_PMGR_PWR_POLICY_INFO_DATA_DOMGRP domGrp;

    /*!
     * Index of leakage equation in Power Leakage Table.
     */
    LwU8                                         leakageIdx;

    /*!
     * Size of the workload median filter.
     */
    LwU8                                         medianFilterSize;
} LW2080_CTRL_PMGR_PWR_POLICY_INFO_DATA_WORKLOAD;

/*!
 * MACROs pointing to Workload Multirail Rail indexes
 */
#define LW2080_CTRL_PMGR_PWR_POLICY_WORKLOAD_MULTIRAIL_VOLT_RAIL_IDX0    0x00U
#define LW2080_CTRL_PMGR_PWR_POLICY_WORKLOAD_MULTIRAIL_VOLT_RAIL_IDX1    0x01U
#define LW2080_CTRL_PMGR_PWR_POLICY_WORKLOAD_MULTIRAIL_VOLT_RAIL_IDX_MAX 0x02U

/*!
 * Structure representing per rail specific parameters.
 */
typedef struct LW2080_CTRL_PMGR_PWR_POLICY_WORKLOAD_MULTIRAIL_VOLT_RAIL {
    /*!
     * Index into Power Topology Table (PWR_CHANNEL) for input channel
     * of given voltage rail
     */
    LwU8 chIdx;
    /*!
     * Index into Voltage Rail Table Index for given voltage rail
     */
    LwU8 voltRailIdx;
} LW2080_CTRL_PMGR_PWR_POLICY_WORKLOAD_MULTIRAIL_VOLT_RAIL;

/*!
 * Structure representing sensed voltage specific parameters.
 */
typedef struct LW2080_CTRL_PMGR_PWR_POLICY_WORKLOAD_MULTIRAIL_SENSED_VOLTAGE {
    /*!
     * Boolean indicating if sensed voltage is to be used instead of set
     * voltage which is used by default.
     */
    LwBool bEnabled;
    /*!
     * Sensed voltage mode. Specifies whether to use minimum, maximum or
     * average of the ADC samples for callwlating sensed voltage.
     * @ref LW2080_CTRL_VOLT_VOLT_RAIL_SENSED_VOLTAGE_<xyz>
     */
    LwU8   mode;
} LW2080_CTRL_PMGR_PWR_POLICY_WORKLOAD_MULTIRAIL_SENSED_VOLTAGE;

/*!
 * Structure of static information specific to the WORKLOAD MULTIRAIL interface.
 */
typedef struct LW2080_CTRL_PMGR_PWR_POLICY_INFO_DATA_WORKLOAD_MULTIRAIL_INTERFACE {
    /*!
     * BOARDOBJ_INTERFACE super class.  Must always be first element in the structure.
     */
    LW2080_CTRL_BOARDOBJ_INTERFACE                                super;

    /*!
     * copydoc @LW2080_CTRL_PMGR_PWR_POLICY_WORKLOAD_MULTIRAIL_VOLT_RAIL
     */
    LW2080_CTRL_PMGR_PWR_POLICY_WORKLOAD_MULTIRAIL_VOLT_RAIL      rail[LW2080_CTRL_PMGR_PWR_POLICY_WORKLOAD_MULTIRAIL_VOLT_RAIL_IDX_MAX];

    /*!
     * copydoc @LW2080_CTRL_PMGR_PWR_POLICY_WORKLOAD_MULTIRAIL_SENSED_VOLTAGE
     */
    LW2080_CTRL_PMGR_PWR_POLICY_WORKLOAD_MULTIRAIL_SENSED_VOLTAGE sensedVoltage;

    /*!
     * Voltage policy table index required for applying voltage delta.
     */
    LwU8                                                          voltPolicyIdx;
} LW2080_CTRL_PMGR_PWR_POLICY_INFO_DATA_WORKLOAD_MULTIRAIL_INTERFACE;

/*!
 * Structure of static information specific to the WORKLOAD power policy.
 */
typedef struct LW2080_CTRL_PMGR_PWR_POLICY_INFO_DATA_WORKLOAD_MULTIRAIL {
    /*!
     * @copydoc LW2080_CTRL_PMGR_PWR_POLICY_INFO_DATA_DOMGRP
     *
     * Must always be first in structure!
     */
    LW2080_CTRL_PMGR_PWR_POLICY_INFO_DATA_DOMGRP                       domGrp;
    /*!
     * Size of the workload median filter.
     */
    LwU8                                                               medianFilterSize;
    /*!
     * Workload multiRail interface
     */
    LW2080_CTRL_PMGR_PWR_POLICY_INFO_DATA_WORKLOAD_MULTIRAIL_INTERFACE workIface;
} LW2080_CTRL_PMGR_PWR_POLICY_INFO_DATA_WORKLOAD_MULTIRAIL;


/*!
 * Structure of static information specific to the soft floor behavior
 * for WORKLOAD_SINGLE_1X power policy.
 */
typedef struct LW2080_CTRL_PMGR_PWR_POLICY_WORKLOAD_SINGLE_1X_SOFT_FLOOR {
    /*!
     *  Boolean to control soft floor behavior on secondary rail
     */
    LwBool        bSoftFloor;
    /*!
     * Index to clock propagation topology entry
     */
    LwBoardObjIdx clkPropTopIdx;
    /*!
     * Index to clock controleed by @ref perfCfControllerIdx
     */
    LwBoardObjIdx perfCfControllerClkIdx;
    /*!
     * Index to utilization controller to provide lwdclk floor
     */
    LwBoardObjIdx perfCfControllerIdx;
} LW2080_CTRL_PMGR_PWR_POLICY_WORKLOAD_SINGLE_1X_SOFT_FLOOR;
typedef struct LW2080_CTRL_PMGR_PWR_POLICY_WORKLOAD_SINGLE_1X_SOFT_FLOOR *PLW2080_CTRL_PMGR_PWR_POLICY_WORKLOAD_SINGLE_1X_SOFT_FLOOR;

/*!
 * Structure of static information specific to the WORKLOAD_SINGLE_1X power policy.
 */
typedef struct LW2080_CTRL_PMGR_PWR_POLICY_INFO_DATA_WORKLOAD_SINGLE_1X {
    /*!
     * PERF_CF_PWR_MODEL interface
     */
    LW2080_CTRL_PMGR_PWR_POLICY_INFO_DATA_PERF_CF_PWR_MODEL   pwrModel;
    /*!
     * Index into Voltage Rail Table for this rail
     */
    LwBoardObjIdx                                             voltRailIdx;
    /*!
     * Clock domain that controls the rail specified by @ref voltRailIdx
     */
    LwBoardObjIdx                                             clkDomainIdx;
    /*!
     * Boolean representing whether the dynamic power callwlation accounts for
     * Clock Gating on this rail or not
     */
    LwBool                                                    bClkGatingAware;
    /*!
     * Size of the workload_single_1x median filter
     */
    LwU8                                                      medianFilterSize;
    /*!
     * Sensed voltage mode. Specifies whether to use minimum, maximum or
     * average of the ADC samples for callwlating sensed voltage.
     * @ref LW2080_CTRL_VOLT_VOLT_RAIL_SENSED_VOLTAGE_<xyz>
     */
    LwU8                                                      sensedVoltageMode;
    /*!
     * Boolean representing Dummy instance of SINGLE_1X (can be true in case
     * of merge rail design)
     */
    LwBool                                                    bDummy;
    /*!
     * Clock index to the Clock domain entry whose propagation logic needs
     * to be ignored.
     */
    LwBoardObjIdx                                             ignoreClkDomainIdx;
    /*!
     *  Structure to control soft floor behavior on secondary rail
     */
    LW2080_CTRL_PMGR_PWR_POLICY_WORKLOAD_SINGLE_1X_SOFT_FLOOR softFloor;
} LW2080_CTRL_PMGR_PWR_POLICY_INFO_DATA_WORKLOAD_SINGLE_1X;
typedef struct LW2080_CTRL_PMGR_PWR_POLICY_INFO_DATA_WORKLOAD_SINGLE_1X *PLW2080_CTRL_PMGR_PWR_POLICY_INFO_DATA_WORKLOAD_SINGLE_1X;

/*!
 * Structure of static information specific to the WORKLOAD_COMBINED_1X power policy.
 */
typedef struct LW2080_CTRL_PMGR_PWR_POLICY_INFO_DATA_WORKLOAD_COMBINED_1X {
    /*!
     * @copydoc LW2080_CTRL_PMGR_PWR_POLICY_INFO_DATA_DOMGRP
     *
     * Must always be first in structure!
     */
    LW2080_CTRL_PMGR_PWR_POLICY_INFO_DATA_DOMGRP            domGrp;
    /*!
     * PERF_CF_PWR_MODEL interface
     */
    LW2080_CTRL_PMGR_PWR_POLICY_INFO_DATA_PERF_CF_PWR_MODEL pwrModel;
    /*!
     * Power Policy Table Relationship index corresponding to first SINGLE_1X
     * Policy to be used by this combined policy. The first SINGLE_1X policy
     * always needs to be associated with the master clock domain.
     */
    LwBoardObjIdx                                           singleRelIdxFirst;
    /*!
     * Power Policy Table Relationship index corresponding to last SINGLE_1X
     * Policy to be used by this combined policy.
     */
    LwBoardObjIdx                                           singleRelIdxLast;
    /*!
     * Used by @interface PwrPolicyDomGrpIsCapped to check if we are actually
     * capping
     */
    LwUFXP4_12                                              capMultiplier;
} LW2080_CTRL_PMGR_PWR_POLICY_INFO_DATA_WORKLOAD_COMBINED_1X;
typedef struct LW2080_CTRL_PMGR_PWR_POLICY_INFO_DATA_WORKLOAD_COMBINED_1X *PLW2080_CTRL_PMGR_PWR_POLICY_INFO_DATA_WORKLOAD_COMBINED_1X;

/*!
 * Structure representing information for Power to Current colwersion.
 */
typedef struct LW2080_CTRL_PMGR_PWR_POLICY_INFO_DATA_HW_THRESHOLD_POWER_TO_LWRR_COLW {
    /*!
     * Specifies if current colwersion is needed or not.
     */
    LwBool bUseLwrrentColwersion;
    /*!
     * Voltage channel index number.
     */
    LwU8   lwrrentChannelIdx;
} LW2080_CTRL_PMGR_PWR_POLICY_INFO_DATA_HW_THRESHOLD_POWER_TO_LWRR_COLW;
typedef struct LW2080_CTRL_PMGR_PWR_POLICY_INFO_DATA_HW_THRESHOLD_POWER_TO_LWRR_COLW *PLW2080_CTRL_PMGR_PWR_POLICY_INFO_DATA_HW_THRESHOLD_POWER_TO_LWRR_COLW;

/*!
 * Structure of static information specific to the HW_THRESHOLD power policy.
 */
typedef struct LW2080_CTRL_PMGR_PWR_POLICY_INFO_DATA_HW_THRESHOLD {
    /*!
     * HW Threshold index. The Threshold Index is a PWR_DEVICE-specific enum for
     * different threshold features in a given PWR_DEVICE. The definition of
     * each PWR_DEVICE's supported threshold indexes can be found in Power
     * Sensors Table Spec.
     */
    LwU8                                                                  thresholdIdx;
    /*!
     * Low HW Threshold index. The Low Threshold Index is a PWR_DEVICE-specific
     * enum for threshold holding either low limit or hysteresis value. It is
     * used when LW2080_CTRL_PMGR_PWR_POLICY_INFO_DATA_HW_THRESHOLD
     * ::bUseLowThreshold is set to LW_TRUE. The definition of each PWR_DEVICE's
     * supported threshold indexes can be found in Power Sensors Table Spec.
     */
    LwU8                                                                  lowThresholdIdx;
    /*!
     * Specifies if low threshold data should be used or not. Supported only on
     * selected PWR_DEVICEs.
     */
    LwBool                                                                bUseLowThreshold;
    /*!
     * Value of low threshold relative to threshold limit. It is used when
     * LW2080_CTRL_PMGR_PWR_POLICY_INFO_DATA_HW_THRESHOLD::bUseLowThreshold is
     * set to LW_TRUE.
     */
    LwUFXP4_12                                                            lowThresholdValue;
    /*!
     * Information to do Power to Current colwersion.
     */
    LW2080_CTRL_PMGR_PWR_POLICY_INFO_DATA_HW_THRESHOLD_POWER_TO_LWRR_COLW pcc;
} LW2080_CTRL_PMGR_PWR_POLICY_INFO_DATA_HW_THRESHOLD;

/*!
 * Structure representing static information of @ref PWR_POLICY_BALANCE
 */
typedef struct LW2080_CTRL_PMGR_PWR_POLICY_INFO_DATA_BALANCE {
    /*!
     * Index of first Power Policy Relationship object for this class.
     * The classes which extend this Virtual class will run the power balancing
     * algorithm on all controllers specified by @ref PWR_POLICY_RELATIONSHIP
     * objects in the range (@ref policyRelIdxFirst, @ref policyRelIdxLast)
     *
     * TODO: move to use LW2080_CTRL_PMGR_PWR_POLICY_RELATIONSHIP_SET
     */
    LwU8 policyRelIdxFirst;

    /*!
     * Index of last Power Policy Relationship for this class.
     * The classes which extend this Virtual class will run the power balancing
     * algorithm on all controllers specified by @ref PWR_POLICY_RELATIONSHIP
     * objects in the range (@ref policyRelIdxFirst, @ref policyRelIdxLast)
     *
     * TODO: move to use LW2080_CTRL_PMGR_PWR_POLICY_RELATIONSHIP_SET
     */
    LwU8 policyRelIdxLast;
} LW2080_CTRL_PMGR_PWR_POLICY_INFO_DATA_BALANCE;

/*!
 * Enumeration/handle for which GPU a PWR_POLICY_GEMINI class represents in the
 * PWR_GEMINI synchronized balancing algorithm.
 */
typedef enum LW2080_CTRL_PMGR_PWR_POLICY_GEMINI_GPU {
    /*!
     * The SLAVE policy does not specify the control parameters in
     * the VBIOS.  In the PWR_GEMINI algorithm, this is policy which is
     * associated with the negative polarity of the limitDelta.
     *
     * Otherwise, the MASTER and SLAVE are functionally equivalent.
     */
    LW2080_CTRL_PMGR_PWR_POLICY_GEMINI_GPU_SLAVE = 0,
    /*!
     * The MASTER policy is the one which specifies the control parameters in
     * the VBIOS.  In the PWR_GEMINI algorithm, this is policy which is
     * associated with the positive polarity of the limitDelta.
     *
     * Otherwise, the MASTER and SLAVE are functionally equivalent.
     */
    LW2080_CTRL_PMGR_PWR_POLICY_GEMINI_GPU_MASTER = 1,
    /*!
     * Must always be last.
     */
    LW2080_CTRL_PMGR_PWR_POLICY_GEMINI_MAX_GPUS = 2,
} LW2080_CTRL_PMGR_PWR_POLICY_GEMINI_GPU;

/*!
 * Structure representing the Gemini SLI Balancing parameters, populated from
 * the VBIOS in a master GEMINI class.  However, they are common to both
 * master and slave.
 */
typedef struct LW2080_CTRL_PMGR_PWR_POLICY_INFO_DATA_GEMINI_CONTROLLER_PARAMS {
    /*!
     * Sampling polling period (ms).
     */
    LwU16 samplePeriodms;
    /*!
     * Step size to use when shifting the deltas from one policy to another.  In
     * limit units (i.e. mW or mA).
     */
    LwU32 deltaStepSize;
    /*!
     * Absolute value maximum possible delta.  The maximum amount of power which
     * can be shifted from one GPU to another.
     */
    LwU32 deltaMax;
    /*!
     * Maximum allowed difference of the respective limits from the respective
     * values of the balanced PWR_POLICYs associated with a PWR_POLICY_GEMINI
     * for the PWR_POLICY_GEMINI object to be considered for balancing.
     *
     * This feature prevents the PWR_GEMINI algorithm from making a balancing
     * action which would not have any impact on the clocks - i.e. the none of
     * the balanced PWR_POLICYs are constraining the clocks, so giving them more
     * power won't increase performance.
     */
    LwU32 diffMax;
} LW2080_CTRL_PMGR_PWR_POLICY_INFO_DATA_GEMINI_CONTROLLER_PARAMS;

/*!
 * Structure representing static information of @ref PWR_GEMINI.  This is the
 * master controller which is associated with a pair of @ref PWR_POLICY_GEMINI
 * objects and implements the actual Gemini synchronized power balancing
 * algorithm.
 */
typedef struct LW2080_CTRL_PMGR_PWR_POLICY_INFO_DATA_GEMINI_CONTROLLER {
    /*!
     * Index of this object within the OBJGPUMGR::pwrGemini PMGR_DEVICE_GROUP.
     */
    LwU8                                                           geminiIdx;

    /*!
     * The OBJGPU::boardId for this Gemini SLI balancing algorithm.  Unique
     * identifier for systems with multiple Gemini balancing algorithms.
     */
    LwU32                                                          boardId;

    /*!
     * Control parameters for this Gemini SLI balancing algorithm.
     */
    LW2080_CTRL_PMGR_PWR_POLICY_INFO_DATA_GEMINI_CONTROLLER_PARAMS params;
} LW2080_CTRL_PMGR_PWR_POLICY_INFO_DATA_GEMINI_CONTROLLER;

/*!
 * Structure representing static information of @ref PWR_POLICY_GEMINI
 */
typedef struct LW2080_CTRL_PMGR_PWR_POLICY_INFO_DATA_GEMINI {
    /*!
     * Handle for which GPU is this PWR_POLICY_GEMINI class in the
     * PWR_GEMINI synchronized balancing algorithm.
     */
    LW2080_CTRL_PMGR_PWR_POLICY_GEMINI_GPU                  gpu;

    /*!
     * Index of first Power Policy Relationship object for this class.
     * This class will apply the current delta value to all
     * PWR_POLICY_RELATIONSHIPs specified in the range [@ref policyRelIdxFirst,
     * @ref policyRelIdxLast].
     *
     * TODO: move to use LW2080_CTRL_PMGR_PWR_POLICY_RELATIONSHIP_SET
     */
    LwU8                                                    policyRelIdxFirst;

    /*!
     * Index of last Power Policy Relationship object for this class.
     * This class will apply the current delta value to all
     * PWR_POLICY_RELATIONSHIPs specified in the range [@ref policyRelIdxFirst,
     * @ref policyRelIdxLast].
     *
     * TODO: move to use LW2080_CTRL_PMGR_PWR_POLICY_RELATIONSHIP_SET
     */
    LwU8                                                    policyRelIdxLast;

    /*!
     * Master controller static state.
     */
    LW2080_CTRL_PMGR_PWR_POLICY_INFO_DATA_GEMINI_CONTROLLER controller;
} LW2080_CTRL_PMGR_PWR_POLICY_INFO_DATA_GEMINI;

/*!
 * Union of type-specific static information data.
 */


/*!
 * Structure of static information describing a PWR_POLICY, which specifies a
 * power policy/limit to enforce on the GPU.
 *
 * This structure contains shared info between 2X and 3X. Version specific
 * info data should be put in version specific structure.
 */
typedef struct LW2080_CTRL_PMGR_PWR_POLICY_SUPER_INFO {
    /*!
     * @ref LW2080_CTRL_PMGR_PWR_POLICY_TYPE_<xyz>.
     */
    LwU8                                      type;

    /*!
     * Index into Power Topology Table (PWR_CHANNEL) for input channel.
     */
    LwU8                                      chIdx;

    /*!
     * Units of limit values. @ref LW2080_CTRL_PMGR_PWR_POLICY_LIMIT_UNIT_<xyz>.
     */
    LwU8                                      limitUnit;

    /*!
     * Minimum allowed limit value.
     */
    LwU32                                     limitMin;
    /*!
     * Rated/default limit value.
     */
    LwU32                                     limitRated;
    /*!
     * Maximum allowed limit value.
     */
    LwU32                                     limitMax;
    /*!
     * Rated battery allowed limit value.
     * @deprecated - remove after lwapi switches to ::limitBattRated
     */
    LwU32                                     limitBatt;
    /*!
     * Rated battery allowed limit value.
     */
    LwU32                                     limitBattRated;
    /*!
     * Maximum battery allowed limit value.
     */
    LwU32                                     limitBattMax;
    /*!
     * Integral control info.
     */
    LW2080_CTRL_PMGR_PWR_POLICY_INFO_INTEGRAL integral;

    /*!
     * Power Policy 3.x Only
     * Sampling Multiplier for this policy. The polling period of this policy
     * will be sampleMult * baseSamplePeriod.
     */
    LwU8                                      sampleMult;

    /*!
     * Type-specific information.
     */
    union {
        LW2080_CTRL_PMGR_PWR_POLICY_INFO_DATA_DOMGRP               domGrp;
        LW2080_CTRL_PMGR_PWR_POLICY_INFO_DATA_BANG_BANG_VF         bangBangVf;
        LW2080_CTRL_PMGR_PWR_POLICY_INFO_DATA_MARCH_VF             marchVF;
        LW2080_CTRL_PMGR_PWR_POLICY_INFO_DATA_PROP_LIMIT           propLimit;
        LW2080_CTRL_PMGR_PWR_POLICY_INFO_DATA_TOTAL_GPU            totalGpu;
        LW2080_CTRL_PMGR_PWR_POLICY_INFO_DATA_WORKLOAD             workload;
        LW2080_CTRL_PMGR_PWR_POLICY_INFO_DATA_WORKLOAD_MULTIRAIL   workloadMulRail;
        LW2080_CTRL_PMGR_PWR_POLICY_INFO_DATA_WORKLOAD_SINGLE_1X   single1x;
        LW2080_CTRL_PMGR_PWR_POLICY_INFO_DATA_WORKLOAD_COMBINED_1X combined1x;
        LW2080_CTRL_PMGR_PWR_POLICY_INFO_DATA_HW_THRESHOLD         hwThreshold;
        LW2080_CTRL_PMGR_PWR_POLICY_INFO_DATA_BALANCE              balance;
        LW2080_CTRL_PMGR_PWR_POLICY_INFO_DATA_GEMINI               gemini;
    } data;
} LW2080_CTRL_PMGR_PWR_POLICY_SUPER_INFO;

/*!
 * 2X version of a PWR_POLICY's info data.
 * Lwrrently 2X has no specific PWR_POLICY field so doing a simple typedef
 * here. It should be extended to contain specific fields if needed in
 * future.
 */
typedef LW2080_CTRL_PMGR_PWR_POLICY_SUPER_INFO LW2080_CTRL_PMGR_PWR_POLICY_2X_INFO;

/*!
 * 3X version of a PWR_POLICY's info data. The super field
 */
typedef struct LW2080_CTRL_PMGR_PWR_POLICY_3X_INFO {
    /*!
     * Super class data info
     */
    LW2080_CTRL_PMGR_PWR_POLICY_SUPER_INFO     super;
    /*!
     * Power Policy 3.X only: sampling multiplier.
     */
    LwU8                                       sampleMult;
    /*!
     * Power Policy 3.X only: filter info parameter.
     */
    LW2080_CTRL_PMGR_PWR_POLICY_3X_FILTER_INFO filterInfo;
} LW2080_CTRL_PMGR_PWR_POLICY_3X_INFO;

/*!
 * Structure of static information describing a PWR_POLICY, which specifies a
 * power policy/limit to enforce on the GPU.
 */
typedef struct LW2080_CTRL_PMGR_PWR_POLICY_INFO {
   /*!
     * This PWR_POLICY structure's version. Determines how the union below should
     * be interpreted.
     */
    LwU8 version;

    /*!
     * Union containing per-policy data.
     */
    union {
        LW2080_CTRL_PMGR_PWR_POLICY_2X_INFO v2x;

        LW2080_CTRL_PMGR_PWR_POLICY_3X_INFO v3x;
    } spec;
} LW2080_CTRL_PMGR_PWR_POLICY_INFO;

/*!
 * Maximum number of LW2080_CTRL_PMGR_PWR_POLICY_RELATIONSHIP entries which can be
 * supported in the RM or PMU.
 */
#define LW2080_CTRL_PMGR_PWR_POLICY_MAX_POLICY_RELATIONSHIPS  32U

/*!
 * Enumeration of Power Policy Relationship types the PMU supports.
 *
 * This is an 8-bit field.
 */
#define LW2080_CTRL_PMGR_PWR_POLICY_RELATIONSHIP_TYPE_WEIGHT  0x00U
#define LW2080_CTRL_PMGR_PWR_POLICY_RELATIONSHIP_TYPE_BALANCE 0x01U

/*!
 * Structure representing a Power Policy Relationship which scales its applied
 * limit value by a specified coefficient/weight.
 */
typedef struct LW2080_CTRL_PMGR_PWR_POLICY_RELATIONSHIP_INFO_WEIGHT {
    /*!
     * Coefficient/weight by which to scale the limit value which the the
     * updating policy wants to apply to the Power Policy corresponding to @ref
     * LW2080_CTRL_PMGR_PWR_POLICY_RELATIONSHIP::policyIdx.
     *
     * Unsigned FXP4.12 value.  Unitless.
     */
    LwU16 weight;
} LW2080_CTRL_PMGR_PWR_POLICY_RELATIONSHIP_INFO_WEIGHT;

/*!
 * Structure representing static information for
 * Power Policy Relationship Balance type. @ref PWR_POLICY_RELATIONSHIP_BALANCE
 */
typedef struct LW2080_CTRL_PMGR_PWR_POLICY_RELATIONSHIP_INFO_BALANCE {
    /*!
     * Index into Power Policy Table. This should point to object of type
     * @ref PWR_POLICY_LIMIT (or class derived thereof)
     *
     * Another index used in conjunction with this is
     * @ref PWR_POLICY_RELATIONSHIP::policyIdx. (and hence this index is
     * secondary policy id)
     *
     * The policies pointed to by these two indexes will be used by a
     * @ref PWR_POLICY_BALANCE object which references instance of this class.
     */
    LwU8        secPolicyIdx;

    /*!
     * GPIO function for Power Balancing PWM signal.
     */
    LwU32       gpioFunc;

    /*!
     * PWM Source
     */
    LwU8        pwmSource;

    /*!
     * PWM frequency in Hz.
     */
    LwU32       pwmFreqHz;

    /*!
     * PWM period.
     */
    LwU32       pwmPeriod;

    /*!
     * PWM duty cycle at start. Unitless quantity in %.
     * Represented as unsigned FXP 16_16
     */
    LwUFXP16_16 pwmDutyCycleInitial;

    /*!
     * PWM duty cycle setp size. Unitless quantity in %.
     * Represented as unsigned FXP 16_16
     */
    LwUFXP16_16 pwmDutyCycleStepSize;

    /*!
     * Index into Power Topology Table @ref PWR_CHANNEL
     * The PWR_CHANNEL object at this index is used to estimate the power for the
     * corresponding phase. This estimate will be used by the
     * @ref PWR_POLICY_BALANCE object referenced by @ref PWR_POLICY_RELATIONSHIP::policyIdx.
     *
     * Value of @ref LW2080_CTRL_PMGR_PWR_CHANNEL_INDEX_ILWALID means the estimate
     * is not available and hence this value should not be used.
     */
    LwU8        phaseEstimateChIdx;

    /*!
     * The phase count multiplier coefficient, when needed to estimate
     * the total while observing from one phase.
     */
    LwU8        phaseCountMultiplier;
} LW2080_CTRL_PMGR_PWR_POLICY_RELATIONSHIP_INFO_BALANCE;

/*!
 * Union of type-specific data.
 */


/*!
 * Structure representing a Power Policy Relationship - specifying how one
 * policy can update the limit value of another.  Power Policies which take
 * policy actions by updating the limits of other channels (i.e. implementing
 * LW2080_CTRL_PMGR_PWR_POLICY_TYPE_LIMIT) use indexes to Power Policy Relationships
 * to determine which policies to update.
 */
typedef struct LW2080_CTRL_PMGR_PWR_POLICY_RELATIONSHIP_INFO {
    /*!
     * @ref LW2080_CTRL_PMGR_PWR_POLICY_RELATIONSHIP_TYPE_<xyz>
     */
    LwU8 type;
    /*!
     * Index of Power Policy to update.
     */
    LwU8 policyIdx;

    /*!
     * Type-specific information.
     */
    union {
        LW2080_CTRL_PMGR_PWR_POLICY_RELATIONSHIP_INFO_WEIGHT  weight;
        LW2080_CTRL_PMGR_PWR_POLICY_RELATIONSHIP_INFO_BALANCE balance;
    } data;
} LW2080_CTRL_PMGR_PWR_POLICY_RELATIONSHIP_INFO;

/*!
 * Maximum number of LW2080_CTRL_PMGR_PWR_VIOLATION entries which can be
 * supported in the RM or PMU.
 *
 * PRATIK TODO - use BOARDOBJ MAX
 */
#define LW2080_CTRL_PMGR_PWR_VIOLATION_MAX                      0x06U

/*!
 * Enumeration of supported Power Violation types.
 */
#define LW2080_CTRL_PMGR_PWR_VIOLATION_TYPE_PROPGAIN            0x00U
#define LW2080_CTRL_PMGR_PWR_VIOLATION_TYPE_UNKNOWN             0xFFU

/*!
 * Enumeration of therm index type for power violation.
 */
#define LW2080_CTRL_PMGR_PWR_VIOLATION_THERM_INDEX_TYPE_EVENT   0x00U
#define LW2080_CTRL_PMGR_PWR_VIOLATION_THERM_INDEX_TYPE_MONITOR 0x01U

/*!
 * Structure describing the static configuration/POR state of the _PROPGAIN class.
 */
typedef struct LW2080_CTRL_PMGR_PWR_VIOLATION_INFO_PROPGAIN {
    /*!
     * Index of first Power Policy Relationship in the Power Policy Table.
     *
     * TODO: move to use LW2080_CTRL_PMGR_PWR_POLICY_RELATIONSHIP_SET
     */
    LwU8 policyRelIdxFirst;
    /*!
     * Index of last Power Policy Relationship in the Power Policy Table.
     *
     * TODO: move to use LW2080_CTRL_PMGR_PWR_POLICY_RELATIONSHIP_SET
     */
    LwU8 policyRelIdxLast;
} LW2080_CTRL_PMGR_PWR_VIOLATION_INFO_PROPGAIN;
typedef struct LW2080_CTRL_PMGR_PWR_VIOLATION_INFO_PROPGAIN *PLW2080_CTRL_PMGR_PWR_VIOLATION_INFO_PROPGAIN;

/*!
 * PWR_VIOLATION type-specific data union.  Discriminated by
 * PWR_VIOLATION::super.type.
 */


/*!
 * Structure containing a union for thermIdx.
 */
typedef struct LW2080_CTRL_PMGR_PWR_VIOLATION_THERM_INDEX {
    /*!
     * Type of therm index @ref LW2080_CTRL_PMGR_PWR_VIOLATION_THERM_INDEX_TYPE_<xyz>
     */
    LwU8 thrmIdxType;
    /*!
     * Union of different types of indexes specified by thrmIdxType
     */
    union {
        /*!
         * Index of RM_PMU_THERM_EVENT_<xyz> providing violation information.
         */
        LwU8 thrmEvent;
        /*!
         * Index into the THERM_MONITOR table.
         */
        LwU8 thrmMon;
    } index;
} LW2080_CTRL_PMGR_PWR_VIOLATION_THERM_INDEX;
typedef struct LW2080_CTRL_PMGR_PWR_VIOLATION_THERM_INDEX *PLW2080_CTRL_PMGR_PWR_VIOLATION_THERM_INDEX;

/*!
 * Structure describing PWR_VIOLATION static information/POR.  Implements the
 * BOARDOBJ model/interface.
 */
typedef struct LW2080_CTRL_PMGR_PWR_VIOLATION_INFO {
    /*!
     * XAPI does not support using an element within another structure as a
     * discriminant.  Placing this redundant type value here until that design
     * constraint can be fixed.
     */
    LwU8                                       type;

    /*!
     * Structure containing info of type of therm index and union of therm
     * indexes based on type.
     */
    LW2080_CTRL_PMGR_PWR_VIOLATION_THERM_INDEX thrmIdx;

    /*!
     * Sample Multiplier.
     */
    LwU8                                       sampleMult;

    /*!
     * Type-specific data union.
     */
    union {
        LW2080_CTRL_PMGR_PWR_VIOLATION_INFO_PROPGAIN propGain;
    } data;
} LW2080_CTRL_PMGR_PWR_VIOLATION_INFO;
typedef struct LW2080_CTRL_PMGR_PWR_VIOLATION_INFO *PLW2080_CTRL_PMGR_PWR_VIOLATION_INFO;

/*!
 * Macros for semantic policy indexes/names - these indexes indicate which
 * policies are implementing certain special board functionality, per GPU/board
 * POR.  These indexes are used to abstract away implementation details from
 * client interfaces - e.g. a client can request the "TGP" policy without having
 * to know which policy index.
 *
 * _RTP - Power Policy controlling Room Temperature
 *     Power (RTP).  This is the soft limit for GPU power which will be enforced
 *     down to the 3D Boost VPstate (aka "Base Clock" in the PWR
 *     2.0/SmartPower/GPU Boost literature).  This policy is not required to be
 *     present on all GPUs.
 *
 * _TGP - Power Policy controlling Total GPU Power
 *    (TGP).  This is the hard limit for total GPU power which will always be
 *    enforced by the RM power policy functionality.
 *
 * _MXM - Power Policy controlling MXM input power.  Input power levels from the
 *    MXM-SIS will be treated as limits to this power policy.
 *
 * _DNOTIFIER - Power Policy for D-notifiers on non-MXM systems.  The system can
 *    limit GPU power by setting power state D2 - D5.
 *
 * _PWR_TGT - Power Target Policy
 *    Policy used by other driver/PMU components (e.g. thermal policies) to
 *    limit power of the GPU.
 *
 * _PWR_TGT_FLOOR - Power Target Policy with Floor
 *    Policy used by other driver/PMU components (e.g. thermal policies) to
 *    limit power of the GPU with vpstate base clocks or a piecewise frequency flooring lwrve.
 *
 * _SMBUS - Power Policy exposed to SMbus for both power reading from the power channel
 *    referenced by the power policy and get and set a power limit.
 *
 * _CORE - Power Policy is one that observes the core power topology index (channel).
 *
 * _KERNEL - Power Policy exposed to Kernel mode clients for both power queries
 *    from the power channel referenced by the power policy and get and set power limits.
 *
 * _EDPP - Power Policy exposed to system for EDPpeak control, allows to update EDPpeak limit
 *    to control peak current. This is similar to D-notifiers which allows control to the EDPcontinuous
 *    value. The limiting value is opaque to the system similar to the D-notifier limits.
 *
 */
#define LW2080_CTRL_PMGR_PWR_POLICY_IDX_RTP                0x00U
#define LW2080_CTRL_PMGR_PWR_POLICY_IDX_TGP                0x01U
#define LW2080_CTRL_PMGR_PWR_POLICY_IDX_MXM                0x02U
#define LW2080_CTRL_PMGR_PWR_POLICY_IDX_DNOTIFIER          0x03U
#define LW2080_CTRL_PMGR_PWR_POLICY_IDX_PWR_TGT            0x04U
#define LW2080_CTRL_PMGR_PWR_POLICY_IDX_PWR_TGT_FLOOR      0x05U
#define LW2080_CTRL_PMGR_PWR_POLICY_IDX_SMBUS              0x06U
#define LW2080_CTRL_PMGR_PWR_POLICY_IDX_CORE               0x07U
#define LW2080_CTRL_PMGR_PWR_POLICY_IDX_KERNEL             0x08U
#define LW2080_CTRL_PMGR_PWR_POLICY_IDX_EDPP               0x09U
#define LW2080_CTRL_PMGR_PWR_POLICY_IDX_NUM_INDEXES        0x0AU

#define LW2080_CTRL_PMGR_PWR_POLICY_IDX_PWR_TGT_RTDP_FLOOR LW2080_CTRL_PMGR_PWR_POLICY_IDX_PWR_TGT_FLOOR

/*!
 * Macros for Power Policy Table version.
 */
#define LW2080_CTRL_PMGR_PWR_POLICY_TABLE_VERSION_2X       0x20U
#define LW2080_CTRL_PMGR_PWR_POLICY_TABLE_VERSION_3X       0x30U

/*!
 * LW2080_CTRL_CMD_PMGR_PWR_POLICY_GET_INFO
 *
 * This command returns the PWR_POLICY static information as specified by the
 * Power Policy Table.
 *
 * https://wiki.lwpu.com/engwiki/index.php/Resman/PState/Data_Tables/Power_Tables/Power_Policy_Table_2.X
 *
 * See LW2080_CTRL_PMGR_PWR_POLICY_INFO_PARAMS for documentation on the
 * parameters.
 *
 * Possible status values returned are
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW2080_CTRL_CMD_PMGR_PWR_POLICY_GET_INFO           (0x20802618U) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PMGR_INTERFACE_ID << 8) | LW2080_CTRL_PMGR_PWR_POLICY_INFO_PARAMS_MESSAGE_ID" */

/*!
 * Enumeration of identifiers for external (system-imposed) power states.
 * We support up to four of these power states, which are requested by
 * the system at runtime via ACPI notification 0xD2 - 0xD5 in order to
 * impose power restrictions on GPU performance.  These power states
 * also correspond to auxiliary power states 1 to 4
 * (@ref LW2080_CTRL_PERF_AUX_POWER_STATE).  Only one state is active at any
 * moment, and there is a special state, D1 to represent no system-imposed
 * restrictions.
 *
 * Internally a power state provides a limit input value and client, along with
 * an index to a power policy object.  The limit inputs are additional inputs
 * to the arbiterer and do not override the limits set by VBIOS.
 */
typedef enum LW2080_CTRL_PMGR_PWR_POLICY_EXT_POWER_STATE_ID {
    LW2080_CTRL_PMGR_PWR_POLICY_EXT_POWER_STATE_ID_D2 = 0,
    LW2080_CTRL_PMGR_PWR_POLICY_EXT_POWER_STATE_ID_D3 = 1,
    LW2080_CTRL_PMGR_PWR_POLICY_EXT_POWER_STATE_ID_D4 = 2,
    LW2080_CTRL_PMGR_PWR_POLICY_EXT_POWER_STATE_ID_D5 = 3,
    LW2080_CTRL_PMGR_PWR_POLICY_EXT_POWER_STATE_ID_COUNT = 4,
    LW2080_CTRL_PMGR_PWR_POLICY_EXT_POWER_STATE_ID_D1 = 2147483647,
} LW2080_CTRL_PMGR_PWR_POLICY_EXT_POWER_STATE_ID;


/*!
 * Structure for a power limit corresponding to an externally set power state.
 */
typedef struct LW2080_CTRL_PMGR_PWR_POLICY_EXT_LIMIT {
    /*!
     * Power policy table index
     */
    LwU8  policyTableIdx;
    /*!
     * Power limit, in mW
     */
    LwU32 limit;
} LW2080_CTRL_PMGR_PWR_POLICY_EXT_LIMIT;

/*!
 * Structure representing 3X specific fields inside
 * @ref LW2080_CTRL_PMGR_PWR_POLICY_INFO_PARAMS
 */
typedef struct LW2080_CTRL_PMGR_PWR_POLICY_INFO_DATA_3X {
    /*!
     * [out] - Power Policy 3.X only: base sampling period.
     */
    LwU16  baseSamplePeriodms;

    /*!
     * [out] - Power Policy 3.X only: minimum client sampling period.
     */
    LwU16  minClientSamplePeriodms;

    /*!
     * [out] - Power Policy 3.X only: low sampling multiplier.
     */
    LwU8   lowSamplingMult;

    /*!
     * [out] - Power Policy 3.X only: Hide TGP reading.
     */
    LwBool bHideTgpReading;

    /*!
     * [out] - Power Policy 3.X only:
     * Boolean indicating if Power Policy Override (PPO) client will broadcast
     * its limitLwrr to RM client.
     */
    LwBool bPPOlimitLwrrBroadcast;

    /*!
     * [out] - Power Policy 3.X only: supported graphics clock domain ( i.e 1x/2x).
     */
    LwU32  graphicsClkDom;
} LW2080_CTRL_PMGR_PWR_POLICY_INFO_DATA_3X;

/*!
 * Union combining 2X and 3X (and potentially future versions) specific fields
 * together. Lwrrently there is no 2X specific field, so this union only
 * contains 3X data.
 */


/*!
 * Structure representing the static state information associated with the GPU's
 * PWR_POLICY power policy functionality.
 */
#define LW2080_CTRL_PMGR_PWR_POLICY_INFO_PARAMS_MESSAGE_ID (0x18U)

typedef struct LW2080_CTRL_PMGR_PWR_POLICY_INFO_PARAMS {
    /*!
     * [out] - Power Policy Table version. Represented by @ref
     * LW2080_CTRL_PMGR_PWR_POLICY_TABLE_VERSION_2X/3X macro.
     */
    LwU8                             version;

    /*!
     * Boolean indicating whether PWR_POLICY evaluation is enabled or not.  This
     * can disable of PWR_POLICY evaluation, while keeping the objects
     * constructed in the PMU such that they can be used for things like
     * PWR_CHANNEL evaluation.
     *
     * This feature is intended to be used for MODS and debug configurations
     * which need to disable power capping.
     */
    LwBool                           bEnabled;

    /*!
     * [out] - Mask of PWR_POLICY entries specified on this GPU.
     */
    LwU32                            policyMask;

    /*!
     * [out] - Mask of PWR_POLICY_DOMGRP entries which take policy actions
     * by setting Domain Group PERF_LIMITs.
     */
    LwU32                            domGrpPolicyMask;

    /*!
     * [out] - Mask of PWR_POLICY entries which implement the PERF_CF_PWR_MODEL
     * interface
     */
    LW2080_CTRL_BOARDOBJGRP_MASK_E32 perfCfPwrModelPolicyMask;

    /*!
     * [out] - Mask of PWR_POLICY_LIMIT entries which take policy actions by
     * requesting new limit values on other PWR_POLICY entries.
     */
    LwU32                            limitPolicyMask;

    /*!
     * [out] - Mask of PWR_POLICY_BALANCE entries
     */
    LwU32                            balancePolicyMask;

    /*!
     * [out] - Mask of PWR_POLICY entries which are contained entirely within the RM and
     * should not be sent down to the PMU.
     */
    LwU32                            nonPMUPolicyMask;

    /*!
     * Power Policy Table index for Power Policy controlling Total GPU Power
     * (TGP).  This is the hard limit for total GPU power which will always be
     * enforced by the RM power policy functionality.
     */
    LwU8                             tgpPolicyIdx;

    /*!
     * Power Policy Table index for Power Policy controlling Room Temperature
     * Power (RTP).  This is the soft limit for GPU power which will be
     * enforced down to the 3D Boost VPstate (aka "Base Clock" in the PWR
     * 2.0/SmartPower/GPU Boost literature).  This policy is not required to be
     * present on all GPUs.
     */
    LwU8                             rtpPolicyIdx;

    /*!
     * Power Policy 3.x Only
     * Power Policy Table index exposed to Kernel mode clients for both power
     * queries from the power channel referenced by the power policy and get and set
     * power limits.
     */
    LwU8                             kernelPolicyIdx;

    /*!
     * Power Policy 3.x Only
     * Base sampling period for all power policies in milli-seconds.
     */
    LwU16                            baseSamplePeriod;

    /*!
     * Indicates whether PWR_POLICIES supports disabling inflection points.
     */
    LwBool                           bInflectionPointsDisableSupported;

    /*!
     * Data union for version specific fields.
     */
    union {
        LW2080_CTRL_PMGR_PWR_POLICY_INFO_DATA_3X v3x;
    } dataUnion;

    /*!
     * Array of semantic policy indexes/names - these indexes indicate which
     * policies are implementing certain special board functionality, per GPU/board
     * POR.  These indexes are used to abstract away implementation details from
     * client interfaces - e.g. a client can request the "TGP" policy without
     * having to know which policy index.
     *
     * @note LW2080_CTRL_PMGR_PWR_POLICY_INDEX_ILWALID indicates that a semantic
     * policy is not present/specified on this GPU.
     */
    LwU8                                           policyIdxs[LW2080_CTRL_PMGR_PWR_POLICY_IDX_NUM_INDEXES];

    /*!
     * [out] - Array of PWR_POLICY entries.  Has valid indexes corresponding to
     * the bits set in @ref policyMask.
     */
    LW2080_CTRL_PMGR_PWR_POLICY_INFO               policies[LW2080_CTRL_PMGR_PWR_POLICY_MAX_POLICIES];

    /*!
     * [out] - Mask of PWR_POLICY_RELATIONSHIP entries specified on this GPU.
     */
    LwU32                                          policyRelMask;

    /*!
     * [out] - Array of PWR_POLICY_RELAIONSHIP entries.  Has valid indexes
     * corresponding to the bits set in @ref policyRelMask.
     */
    LW2080_CTRL_PMGR_PWR_POLICY_RELATIONSHIP_INFO  policyRels[LW2080_CTRL_PMGR_PWR_POLICY_MAX_POLICY_RELATIONSHIPS];

    /*!
     * [out] - Array of externally available power states
     */
    LW2080_CTRL_PMGR_PWR_POLICY_EXT_LIMIT          extLimits[LW2080_CTRL_PMGR_PWR_POLICY_EXT_POWER_STATE_ID_COUNT];

    /*!
     * [out] - Lwrrently set external power state
     */
    LW2080_CTRL_PMGR_PWR_POLICY_EXT_POWER_STATE_ID extPowerState;

    /*!
     * [out] - Mask of PWR_VIOLATION entries specified on this GPU.
     */
    LwU32                                          pwrViolMask;

    /*!
     * [out] - Array of PWR_VIOLATION structures. Has valid indexes corresponding to the
     * bits set in @ref pwrViolMask.
     */
    LW2080_CTRL_PMGR_PWR_VIOLATION_INFO            violations[LW2080_CTRL_PMGR_PWR_VIOLATION_MAX];

    /*!
     * [out] - Externally available EDPp power limit
     */
    LwU32                                          edppLimit;
} LW2080_CTRL_PMGR_PWR_POLICY_INFO_PARAMS;

/*!
 * The maximum number of domain groups the PWR_POLICY object supports.
 */
#define LW2080_CTRL_PMGR_PWR_POLICY_DOMAIN_GROUPS_NUM_GROUPS      3U

/*!
 * The maximum number of virtual Pstates we want to monitor violation behavior
 * for.
 * Note there is a "faked" vPstate i.e. MAX_CLOCK.
 * Must keep in sync with @ref RM_PMU_PERF_VPSTATE_IDX_MAX_IDXS.
 */
#define LW2080_CTRL_PMGR_PWR_POLICY_DOMAIN_GROUPS_NUM_VPSTATES    4U

/*!
 * The set of virtual Pstates that have violation counter. These vPstates are
 * shared between RM and PMU. Must keep in sync with
 * @ref RM_PMU_PERF_VPSTATE_IDX union.
 */
#define LW2080_CTRL_PMGR_PWR_POLICY_DOMAIN_GROUPS_VPS_RATED_TDP   0U
#define LW2080_CTRL_PMGR_PWR_POLICY_DOMAIN_GROUPS_VPS_3D_BOOST    1U
#define LW2080_CTRL_PMGR_PWR_POLICY_DOMAIN_GROUPS_VPS_TURBO_BOOST 2U
#define LW2080_CTRL_PMGR_PWR_POLICY_DOMAIN_GROUPS_VPS_MAX_CLOCK   3U

/*!
 * Structure of static information specific to the integral control
 * status in the power policy.
 */
typedef struct LW2080_CTRL_PMGR_PWR_POLICY_STATUS_INTEGRAL {
    /*!
     * The aclwmulated sum of differences between
     * the current power value and current policy limit.
     */
    LwS32 lwrrRunningDiff;

    /*!
     * The new callwlated power limit for all the future samples.
     */
    LwS32 lwrrIntegralLimit;
} LW2080_CTRL_PMGR_PWR_POLICY_STATUS_INTEGRAL;

/*!
 * The set of domain group limit values (lwrrently only maximums) that a
 * PWR_POLICY object wishes to apply to the GPU.
 */
typedef struct LW2080_CTRL_PMGR_PWR_POLICY_STATUS_DOMAIN_GROUP_LIMITS {
    /*!
     * Array of domain-group-specific limit values.  This array is indexed by
     * the domain group indexes.
     *
     * ~~~PS20TODO~~~: Add a full set of DOMAIN_GROUP index macros in @ref
     * ctrl2080perf.h.
     */
    LwU32 values[LW2080_CTRL_PMGR_PWR_POLICY_DOMAIN_GROUPS_NUM_GROUPS];
} LW2080_CTRL_PMGR_PWR_POLICY_STATUS_DOMAIN_GROUP_LIMITS;

typedef struct LW2080_CTRL_PMGR_PWR_POLICY_STATUS_DATA_DOMGRP {
    /*!
     * The domain group limit values this PWR_POLICY object wishes to apply to
     * GPU.
     */
    LW2080_CTRL_PMGR_PWR_POLICY_STATUS_DOMAIN_GROUP_LIMITS domGrpLimits;
    /*!
     * The current Domain Group ceiling for this PWR_POLICY_DOMGRP object.  Will
     * be used by evaluation of the PWR_POLICY_DOMGRP class to set the highest
     * limits the object may request.  This can enforce a lower ceiling than the
     * maximum (pstate, VF) point on the GPU under various operating conditions.
     */
    LW2080_CTRL_PMGR_PWR_POLICY_STATUS_DOMAIN_GROUP_LIMITS domGrpCeiling;
    /*!
     * A boolean flag to denote whether this DOMGRP POLICY is actively capping.
     */
    LwBool                                                 bIsCapped;
} LW2080_CTRL_PMGR_PWR_POLICY_STATUS_DATA_DOMGRP;

/*!
 * Macros encoding the actions/decisions the PWR_POLICY_BANG_BANG_VF controller
 * may make at each evaluation (@ref pwrPolicyDomGrpEvaluate_BANG_BANG_VF).
 *
 * _NONE - Current value is acceptable, so do not change the requested clock
 *     amount.
 * _CAP - Current value is above the limit, so VF values must be capped down one
 *     step (if available).
 * _UNCAP - Current value is <= the limit * uncapLimitRatio, so VF values may be
 *     uncaped one step (if available).
 */
#define LW2080_CTRL_PMGR_PWR_POLICY_BANG_BANG_VF_ACTION_NONE  0x00000000U
#define LW2080_CTRL_PMGR_PWR_POLICY_BANG_BANG_VF_ACTION_CAP   0x00000001U
#define LW2080_CTRL_PMGR_PWR_POLICY_BANG_BANG_VF_ACTION_UNCAP 0x00000002U

/*!
 * This structure represents (pstate index, VF entry index) set which represents
 * an operational VF point.  This structure is represent both the current/input
 * VF point and the requested/output VF point for the _BANG_BANG_VF Power
 * Policy.
 */
typedef struct LW2080_CTRL_PMGR_PWR_POLICY_STATUS_DATA_BANG_BANG_VF_INDEXES {
    /*!
     * Pstate index.  This is the index of the pstate within the VBIOS Perf
     * Table, not a pstate name/number - e.g. index 0 => P8, index 3 => P0.
     */
    LwU8 pstateIdx;
    /*!
     * VF Entry Index.  This is the index of the VF entry within the VBIOS VF
     * table.
     */
    LwU8 vfIdx;
} LW2080_CTRL_PMGR_PWR_POLICY_STATUS_DATA_BANG_BANG_VF_INDEXES;

/*!
 * Structure representing the _BANG_BANG_VF-specific algorithm dynamic state
 * data.  This is the current state of various algorithm input/output data for
 * the last evaluation of the algorithm.  This information is useful for
 * debugging algorithm behavior.
 */
typedef struct LW2080_CTRL_PMGR_PWR_POLICY_STATUS_DATA_BANG_BANG_VF {
    /*!
     * Must always be first in structure.
     */
    LW2080_CTRL_PMGR_PWR_POLICY_STATUS_DATA_DOMGRP               domGrp;

    /*!
     * Last action taken by the _BANG_BANG_VF algorithm.
     *
     * @ref LW2080_CTRL_PMGR_PWR_POLICY_BANG_BANG_VF_ACTION_<xyz>
     */
    LwU8                                                         action;
    /*!
     * Input VF point to the _BANG_BANG_VF algorithm for @ref action.
     */
    LW2080_CTRL_PMGR_PWR_POLICY_STATUS_DATA_BANG_BANG_VF_INDEXES input;
    /*!
     * Output VF point of the _BANG_BANG_VF algorithm as decided by the @ref
     * action.
     */
    LW2080_CTRL_PMGR_PWR_POLICY_STATUS_DATA_BANG_BANG_VF_INDEXES output;
} LW2080_CTRL_PMGR_PWR_POLICY_STATUS_DATA_BANG_BANG_VF;

/*!
 * Macros encoding the actions/decisions the PWR_POLICY_MARCH controller
 * may make at each evaluation.
 *
 * _NONE - Current value is acceptable, so do not change the requested state.
 * _CAP - Current value is above the limit, so algorithm must cap down one
 *     step size (if available).
 * _UNCAP - Current value is <= the uncap limit (determined by hysteresis), so
 *     algorithm may uncap by one step size (if available).
 */
#define LW2080_CTRL_PMGR_PWR_POLICY_MARCH_ACTION_NONE  0x00U
#define LW2080_CTRL_PMGR_PWR_POLICY_MARCH_ACTION_CAP   0x01U
#define LW2080_CTRL_PMGR_PWR_POLICY_MARCH_ACTION_UNCAP 0x02U

/*!
 * Union of type-specific dynamic state data.
 */


/*!
 * Structure representing the _MARCH-specific algorithm dynamic state
 * data.  This is the current state of various algorithm input/output data for
 * the last evaluation of the algorithm.  This information is useful for
 * debugging algorithm behavior.
 */
typedef struct LW2080_CTRL_PMGR_PWR_POLICY_STATUS_DATA_MARCH {
    /*!
     * @ref LW2080_CTRL_PMGR_PWR_POLICY_TYPE_MARCH_<xyz>
     */
    LwU8  type;
    /*!
     * Current uncap limit - as callwlated by output of PWR_POLICY::limitLwrr and
     * PWR_POLICY_MARCH::hysteresis.
     */
    LwU32 uncapLimit;
    /*!
     * Last action taken by the _MARCH algorithm.
     *
     * @ref LW2080_CTRL_PMGR_PWR_POLICY_MARCH_ACTION_<xyz>
     */
    LwU8  action;

    /*!
     * Type-specific data.
     */
    union {
        LwU8 placeHolder;
    } data;
} LW2080_CTRL_PMGR_PWR_POLICY_STATUS_DATA_MARCH;

/*!
 * Structure representing MARCH_VF-specific PWR_POLICY dynamic status.
 */
typedef struct LW2080_CTRL_PMGR_PWR_POLICY_STATUS_DATA_MARCH_VF {
    /*!
     * @copydoc LW2080_CTRL_PMGR_PWR_POLICY_STATUS_DATA_DOMGRP
     *
     * Must always be first in structure!
     */
    LW2080_CTRL_PMGR_PWR_POLICY_STATUS_DATA_DOMGRP domGrp;
    /*!
     * Common state of the PWR_POLICY_MARCH algorithm.  Does not need to be at
     * any specific location in the structure.
     */
    LW2080_CTRL_PMGR_PWR_POLICY_STATUS_DATA_MARCH  march;
} LW2080_CTRL_PMGR_PWR_POLICY_STATUS_DATA_MARCH_VF;

/*!
 * This represents the current GPU state which is passed as input to @ref
 * _pwrPolicyWorkloadComputeWorkload() to callwlate the current workload/active
 * capacitance (w).
 *
 *     w = (Ptotal - Pleakage) / (V^2 * f)
 *     w = (Itotal - Ileakage) / (V * f)
 */
typedef struct LW2080_CTRL_PMGR_PWR_POLICY_STATUS_DATA_WORKLOAD_WORK_INPUT {
    /*!
     * Estimated leakage power (mW).
     */
    LwU32 pwrLeakagemW;
    /*!
     * Estimated leakage - in units specified by @ref
     * RM_PMU_PMGR_PWR_POLICY::limitUnit - i.e. power (mW) or current (mA).
     */
    LwU32 leakagemX;
    /*!
     * Frequency (MHz)
     */
    LwU32 freqMHz;
    /*!
     * Voltage^2 (mV^2).
     */
    LwU32 voltmV2;
    /*!
     * Voltage in mVolt.
     */
    LwU32 voltmV;
} LW2080_CTRL_PMGR_PWR_POLICY_STATUS_DATA_WORKLOAD_WORK_INPUT;

/*!
 * This represents a leakage and voltage combination for which the WORKLOAD
 * algorithrm should compute the highest possible frequency.  This structure is
 * passed as input to @ref _pwrPolicyWorkloadVfEntryComputeClkMHz().
 *
 * The value returned from the PMU will represent the last callwlation for the
 * highest set of frequencies the WORKLOAD algorithm deems possible for the
 * given workload per @ref _pwrPolicyWorkloadComputeClkMHz().
 */
typedef struct LW2080_CTRL_PMGR_PWR_POLICY_STATUS_DATA_WORKLOAD_FREQ_INPUT {
    /*!
     * The filtered workload/active capacitance (w) which is used to determine
     * the target clocks.
     */
    LwUFXP20_12 workloadmWperMHzmV2;
    /*!
     * The filtered workload/active capacitance (w) which is used to determine
     * the target clocks.
     */
    LwUFXP20_12 workload;
    /*!
     * Estimated leakage power (mW).
     */
    LwU32       pwrLeakagemW;
    /*!
     * Estimated leakage - in units specified by @ref
     * RM_PMU_PMGR_PWR_POLICY::limitUnit - i.e. power (mW) or current (mA).
     */
    LwU32       leakagemX;
    /*!
     * Frequency (MHz)
     */
    LwU32       freqMaxMHz;
    /*!
     * Voltage^2 (mV^2).
     */
    LwU32       voltmV2;
    /*!
     * Voltage in mVolt
     */
    LwU32       voltmV;
    /*!
     * The current VF entry being evaluated.  This is purely for debugging, it
     * is not actually used in @ref _pwrPolicyWorkloadVfEntryComputeClkMHz().
     */
    LwU8        vfEntryIdx;
} LW2080_CTRL_PMGR_PWR_POLICY_STATUS_DATA_WORKLOAD_FREQ_INPUT;

/*!
 * Structure representing WORKLOAD-specific PWR_POLICY dynamic status.
 */
typedef struct LW2080_CTRL_PMGR_PWR_POLICY_STATUS_DATA_WORKLOAD {
    /*!
     * Must always be first in structure.
     */
    LW2080_CTRL_PMGR_PWR_POLICY_STATUS_DATA_DOMGRP              domGrp;
    LW2080_CTRL_PMGR_PWR_POLICY_STATUS_DATA_WORKLOAD_WORK_INPUT work;
    LW2080_CTRL_PMGR_PWR_POLICY_STATUS_DATA_WORKLOAD_FREQ_INPUT freq;
    LwU32                                                       workloadmWperMHzmV2;
    /*!
     * The current workload/active capacitance (w) callwlated by @ref
     * _pwrPolicyWorkloadComputeWorkload().  This value is unfiltered.
     */
    LwUFXP20_12                                                 workload;
} LW2080_CTRL_PMGR_PWR_POLICY_STATUS_DATA_WORKLOAD;

/*!
 * This represents the current GPU state which is passed as input to @ref
 * _pwrPolicyWorkloadMultiRailComputeWorkload() to callwlate the current
 * workload/active capacitance (w).
 */
typedef struct LW2080_CTRL_PMGR_PWR_POLICY_STATUS_DATA_WORKLOAD_MULTIRAIL_VOLT_RAIL_WORK_INPUT {
    /*!
     * Estimated leakage - in units specified by @ref
     * RM_PMU_PMGR_PWR_POLICY::limitUnit - i.e. power (mW) or current (mA).
     */
    LwU32 leakagemX;
    /*!
     * Voltage in mVolt.
     */
    LwU32 voltmV;
    /*!
     * Power in mW/ Current in mA
     */
    LwU32 observedVal;
} LW2080_CTRL_PMGR_PWR_POLICY_STATUS_DATA_WORKLOAD_MULTIRAIL_VOLT_RAIL_WORK_INPUT;
typedef struct LW2080_CTRL_PMGR_PWR_POLICY_STATUS_DATA_WORKLOAD_MULTIRAIL_VOLT_RAIL_WORK_INPUT *PLW2080_CTRL_PMGR_PWR_POLICY_STATUS_DATA_WORKLOAD_MULTIRAIL_VOLT_RAIL_WORK_INPUT;

/*!
 * This represents the current GPU state which is passed as input to @ref
 * _pwrPolicyWorkloadMultiRailComputeWorkload() to callwlate the current
 * workload/active capacitance (w).
 *
 *     w = (Ptotal - Pleakage) / (V^2 * f)
 *     w = (Itotal - Ileakage) / (V * f)
 */
typedef struct LW2080_CTRL_PMGR_PWR_POLICY_STATUS_DATA_WORKLOAD_MULTIRAIL_WORK_INPUT {
    /*!
     * Frequency (MHz)
     */
    LwU32                                                                           freqMHz;
    /*!
     * MSCG residency ratio [0, 1] callwlated based on sleep time. Updated on
     * every evaluation.
     */
    LwUFXP20_12                                                                     mscgResidency;
    /*!
     * PG residency ratio [0, 1] callwlated based on sleep time. Updated on
     * every evaluation.
     */
    LwUFXP20_12                                                                     pgResidency;
    /*!
     * copydoc@
     * LW2080_CTRL_PMGR_PWR_POLICY_STATUS_DATA_WORKLOAD_MULTIRAIL_VOLT_RAIL_WORK_INPUT
     */
    LW2080_CTRL_PMGR_PWR_POLICY_STATUS_DATA_WORKLOAD_MULTIRAIL_VOLT_RAIL_WORK_INPUT rail[LW2080_CTRL_PMGR_PWR_POLICY_WORKLOAD_MULTIRAIL_VOLT_RAIL_IDX_MAX];
} LW2080_CTRL_PMGR_PWR_POLICY_STATUS_DATA_WORKLOAD_MULTIRAIL_WORK_INPUT;
typedef struct LW2080_CTRL_PMGR_PWR_POLICY_STATUS_DATA_WORKLOAD_MULTIRAIL_WORK_INPUT *PLW2080_CTRL_PMGR_PWR_POLICY_STATUS_DATA_WORKLOAD_MULTIRAIL_WORK_INPUT;

/*!
 * This represents a leakage and voltage combination for which the WORKLOAD
 * algorithrm should compute the highest possible frequency.  This structure is
 * passed as input to @ref _pwrPolicyWorkloadMultiRailComputeClkMHz().
 *
 * The value returned from the PMU will represent the last callwlation for the
 * highest set of frequencies the WORKLOAD algorithm deems possible for the
 * given workload per @ref _pwrPolicyWorkloadMultiRailComputeClkMHz().
 */
typedef struct LW2080_CTRL_PMGR_PWR_POLICY_STATUS_DATA_WORKLOAD_MULTIRAIL_VOLT_RAIL_FREQ_INPUT {
    /*!
     * The filtered workload/active capacitance (w) which is used to determine
     * the target clocks.
     */
    LwUFXP20_12 workload;
    /*!
     * Estimated leakage - in units specified by @ref
     * RM_PMU_PMGR_PWR_POLICY::limitUnit - i.e. power (mW) or current (mA).
     */
    LwU32       leakagemX;
    /*!
     * Voltage in mVolt
     */
    LwU32       voltmV;
    /*!
     * Power in mW/ Current in mA
     */
    LwU32       estimatedVal;
    /*!
     * Voltage floor in uV.
     */
    LwU32       voltFlooruV;
} LW2080_CTRL_PMGR_PWR_POLICY_STATUS_DATA_WORKLOAD_MULTIRAIL_VOLT_RAIL_FREQ_INPUT;
typedef struct LW2080_CTRL_PMGR_PWR_POLICY_STATUS_DATA_WORKLOAD_MULTIRAIL_VOLT_RAIL_FREQ_INPUT *PLW2080_CTRL_PMGR_PWR_POLICY_STATUS_DATA_WORKLOAD_MULTIRAIL_VOLT_RAIL_FREQ_INPUT;

/*!
 * This represents a leakage and voltage combination for which the WORKLOAD
 * algorithrm should compute the highest possible frequency.  This structure is
 * passed as input to @ref _pwrPolicyWorkloadMultiRailComputeClkMHz().
 *
 * The value returned from the PMU will represent the last callwlation for the
 * highest set of frequencies the WORKLOAD algorithm deems possible for the
 * given workload per @ref _pwrPolicyWorkloadMultiRailComputeClkMHz().
 */
typedef struct LW2080_CTRL_PMGR_PWR_POLICY_STATUS_DATA_WORKLOAD_MULTIRAIL_FREQ_INPUT {
    /*!
     * Frequency (MHz)
     */
    LwU32                                                                           freqMaxMHz;
    /*!
     * MSCG residency ratio [0, 1] callwlated based on sleep time. Updated on
     * every evaluation.
     */
    LwUFXP20_12                                                                     mscgResidency;
    /*!
     * PG residency ratio [0, 1] callwlated based on sleep time. Updated on
     * every evaluation.
     */
    LwUFXP20_12                                                                     pgResidency;
    /*!
     * copydoc@
     * LW2080_CTRL_PMGR_PWR_POLICY_STATUS_DATA_WORKLOAD_MULTIRAIL_VOLT_RAIL_FREQ_INPUT
     */
    LW2080_CTRL_PMGR_PWR_POLICY_STATUS_DATA_WORKLOAD_MULTIRAIL_VOLT_RAIL_FREQ_INPUT rail[LW2080_CTRL_PMGR_PWR_POLICY_WORKLOAD_MULTIRAIL_VOLT_RAIL_IDX_MAX];
} LW2080_CTRL_PMGR_PWR_POLICY_STATUS_DATA_WORKLOAD_MULTIRAIL_FREQ_INPUT;
typedef struct LW2080_CTRL_PMGR_PWR_POLICY_STATUS_DATA_WORKLOAD_MULTIRAIL_FREQ_INPUT *PLW2080_CTRL_PMGR_PWR_POLICY_STATUS_DATA_WORKLOAD_MULTIRAIL_FREQ_INPUT;

/*!
 * Structure representing WORKLOAD_MULTIRAIL_INTERFACE-specific PWR_POLICY dynamic status.
 */
typedef struct LW2080_CTRL_PMGR_PWR_POLICY_STATUS_DATA_WORKLOAD_MULTIRAIL_INTERFACE {
    /*!
     * BOARDOBJ_INTERFACE super class.  Must always be first element in the structure.
     */
    LW2080_CTRL_BOARDOBJ_INTERFACE                                        super;

    /*!
     * @copydoc LW2080_CTRL_PMGR_PWR_POLICY_STATUS_DATA_WORKLOAD_MULTIRAIL_WORK_INPUT
     */
    LW2080_CTRL_PMGR_PWR_POLICY_STATUS_DATA_WORKLOAD_MULTIRAIL_WORK_INPUT work;

    /*!
     * @copydoc LW2080_CTRL_PMGR_PWR_POLICY_STATUS_DATA_WORKLOAD_MULTIRAIL_FREQ_INPUT
     */
    LW2080_CTRL_PMGR_PWR_POLICY_STATUS_DATA_WORKLOAD_MULTIRAIL_FREQ_INPUT freq;
    /*!
     * The current workload/active capacitance (w) callwlated by @ref
     * _pwrPolicyWorkloadMultiRailComputeWorkload().  This value is unfiltered.
     */
    LwUFXP20_12                                                           workload[LW2080_CTRL_PMGR_PWR_POLICY_WORKLOAD_MULTIRAIL_VOLT_RAIL_IDX_MAX];
} LW2080_CTRL_PMGR_PWR_POLICY_STATUS_DATA_WORKLOAD_MULTIRAIL_INTERFACE;
typedef struct LW2080_CTRL_PMGR_PWR_POLICY_STATUS_DATA_WORKLOAD_MULTIRAIL_INTERFACE *PLW2080_CTRL_PMGR_PWR_POLICY_STATUS_DATA_WORKLOAD_MULTIRAIL_INTERFACE;

/*!
 * Structure representing WORKLOAD-specific PWR_POLICY dynamic status.
 */
typedef struct LW2080_CTRL_PMGR_PWR_POLICY_STATUS_DATA_WORKLOAD_MULTIRAIL {
    /*!
     * Must always be first in structure.
     */
    LW2080_CTRL_PMGR_PWR_POLICY_STATUS_DATA_DOMGRP                       domGrp;
    /*!
     * Workload interface
     */
    LW2080_CTRL_PMGR_PWR_POLICY_STATUS_DATA_WORKLOAD_MULTIRAIL_INTERFACE workIface;
} LW2080_CTRL_PMGR_PWR_POLICY_STATUS_DATA_WORKLOAD_MULTIRAIL;
typedef struct LW2080_CTRL_PMGR_PWR_POLICY_STATUS_DATA_WORKLOAD_MULTIRAIL *PLW2080_CTRL_PMGR_PWR_POLICY_STATUS_DATA_WORKLOAD_MULTIRAIL;

/*!
 * Structure of dynamic information specific to the WORKLOAD SINGLE_1X power policy.
 */
typedef struct LW2080_CTRL_PMGR_PWR_POLICY_STATUS_DATA_WORKLOAD_SINGLE_1X {
    LwU32                                                     rsvd;
    /*!
     * PERF_CF_PWR_MODEL interface
     */
    LW2080_CTRL_PMGR_PWR_POLICY_STATUS_DATA_PERF_CF_PWR_MODEL pwrModel;
} LW2080_CTRL_PMGR_PWR_POLICY_STATUS_DATA_WORKLOAD_SINGLE_1X;
typedef struct LW2080_CTRL_PMGR_PWR_POLICY_STATUS_DATA_WORKLOAD_SINGLE_1X *PLW2080_CTRL_PMGR_PWR_POLICY_STATUS_DATA_WORKLOAD_SINGLE_1X;

/*!
 * Invalid search regime frequency
 */
#define LW2080_CTRL_PMGR_PWR_POLICY_WORKLOAD_COMBINED_1X_SEARCH_REGIME_FREQ_ILWALID      LW_U32_MAX

/*!
 * @defgroup    LW2080_CTRL_PMGR_PWR_POLICY_WORKLOAD_COMBINED_1X_SEARCH_REGIME_ID
 *
 * Macro defines for PWR_POLICY_WORKLOAD_COMBINED search regime IDs.
 * @{
 */
#define LW2080_CTRL_PMGR_PWR_POLICY_WORKLOAD_COMBINED_1X_SEARCH_REGIME_ID_SEC_VMIN_FLOOR 0x0U
#define LW2080_CTRL_PMGR_PWR_POLICY_WORKLOAD_COMBINED_1X_SEARCH_REGIME_ID_PRI_VMIN_FLOOR 0x1U
#define LW2080_CTRL_PMGR_PWR_POLICY_WORKLOAD_COMBINED_1X_SEARCH_REGIME_ID_MAX_RATIO      0x2U
#define LW2080_CTRL_PMGR_PWR_POLICY_WORKLOAD_COMBINED_1X_SEARCH_REGIME_ID_SEC_SOFT_FLOOR 0x3U
#define LW2080_CTRL_PMGR_PWR_POLICY_WORKLOAD_COMBINED_1X_SEARCH_REGIME_ID_DEFAULT_RATIO  0x4U
#define LW2080_CTRL_PMGR_PWR_POLICY_WORKLOAD_COMBINED_1X_SEARCH_REGIME_ID_MIN_RATIO      0x5U
#define LW2080_CTRL_PMGR_PWR_POLICY_WORKLOAD_COMBINED_1X_SEARCH_REGIME_ID_MAX            0x6U
#define LW2080_CTRL_PMGR_PWR_POLICY_WORKLOAD_COMBINED_1X_SEARCH_REGIME_ID_DUMMY_ROOT     (0x7U) /* finn: Evaluated from "(LW2080_CTRL_PMGR_PWR_POLICY_WORKLOAD_COMBINED_1X_SEARCH_REGIME_ID_MAX + 1)" */
#define LW2080_CTRL_PMGR_PWR_POLICY_WORKLOAD_COMBINED_1X_SEARCH_REGIME_ID_ILWALID        0xFFU
/*! @} */


// Legacy defines to be removed in follow-on CL once corresponding changes are submitted to LWAPI.
#define LW2080_CTRL_PMGR_PWR_POLICY_WORKLOAD_COMBINED_1X_SEARCH_REGIME_SEC_VMIN_FLOOR    LW2080_CTRL_PMGR_PWR_POLICY_WORKLOAD_COMBINED_1X_SEARCH_REGIME_ID_SEC_VMIN_FLOOR
#define LW2080_CTRL_PMGR_PWR_POLICY_WORKLOAD_COMBINED_1X_SEARCH_REGIME_PRI_VMIN_FLOOR    LW2080_CTRL_PMGR_PWR_POLICY_WORKLOAD_COMBINED_1X_SEARCH_REGIME_ID_PRI_VMIN_FLOOR
#define LW2080_CTRL_PMGR_PWR_POLICY_WORKLOAD_COMBINED_1X_SEARCH_REGIME_MAX_RATIO         LW2080_CTRL_PMGR_PWR_POLICY_WORKLOAD_COMBINED_1X_SEARCH_REGIME_ID_MAX_RATIO
#define LW2080_CTRL_PMGR_PWR_POLICY_WORKLOAD_COMBINED_1X_SEARCH_REGIME_SEC_SOFT_FLOOR    LW2080_CTRL_PMGR_PWR_POLICY_WORKLOAD_COMBINED_1X_SEARCH_REGIME_ID_SEC_SOFT_FLOOR
#define LW2080_CTRL_PMGR_PWR_POLICY_WORKLOAD_COMBINED_1X_SEARCH_REGIME_DEFAULT_RATIO     LW2080_CTRL_PMGR_PWR_POLICY_WORKLOAD_COMBINED_1X_SEARCH_REGIME_ID_DEFAULT_RATIO
#define LW2080_CTRL_PMGR_PWR_POLICY_WORKLOAD_COMBINED_1X_SEARCH_REGIME_MAX               LW2080_CTRL_PMGR_PWR_POLICY_WORKLOAD_COMBINED_1X_SEARCH_REGIME_ID_MAX
#define LW2080_CTRL_PMGR_PWR_POLICY_WORKLOAD_COMBINED_1X_SEARCH_REGIME_ILWALID           LW2080_CTRL_PMGR_PWR_POLICY_WORKLOAD_COMBINED_1X_SEARCH_REGIME_ID_ILWALID

/*!
 * @defgroup LW2080_CTRL_PMGR_PWR_POLICY_WORKLOAD_COMBINED_1X_SINGLE_1X_IDX
 *
 * Macros representing SINGLE_1X indices within the COMBINED_1X.
 *
 * @{
 */
#define LW2080_CTRL_PMGR_PWR_POLICY_WORKLOAD_COMBINED_1X_SINGLE_1X_IDX_PRI               0x0U
#define LW2080_CTRL_PMGR_PWR_POLICY_WORKLOAD_COMBINED_1X_SINGLE_1X_IDX_SEC               0x1U
#define LW2080_CTRL_PMGR_PWR_POLICY_WORKLOAD_COMBINED_1X_SINGLE_1X_IDX_MAX               0x2U
/*! @} */

/*!
 * Structure to hold the frequencies for a regime.
 */
typedef struct LW2080_CTRL_PMGR_PWR_POLICY_WORKLOAD_COMBINED_1X_REGIME_SINGLES_FREQ_TUPLE {
    /*!
     * frequencies corresponding to single1x instances
     */
    LwU32 freqMHz[LW2080_CTRL_PMGR_PWR_POLICY_WORKLOAD_COMBINED_1X_SINGLE_1X_IDX_MAX];
} LW2080_CTRL_PMGR_PWR_POLICY_WORKLOAD_COMBINED_1X_REGIME_SINGLES_FREQ_TUPLE;

/*!
 * @defgroup    LW2080_CTRL_PMGR_PWR_POLICY_WORKLOAD_COMBINED_1X_SEARCH_REGIME_TUPLE_IDX
 *
 * Macros representing indexes into @ref
 * LW2080_CTRL_PMGR_PWR_POLICY_WORKLOAD_COMBINED_1X_SEARCH_REGIME::freqTuples[].
 *
 * @ref
 * LW2080_CTRL_PMGR_PWR_POLICY_WORKLOAD_COMBINED_1X_SEARCH_REGIME_TUPLE_IDX_START
 * corresponds to @ref
 * LW2080_CTRL_PMGR_PWR_POLICY_WORKLOAD_COMBINED_1X_SEARCH_REGIME::tupleIdxStart.
 *
 * @ref
 * LW2080_CTRL_PMGR_PWR_POLICY_WORKLOAD_COMBINED_1X_SEARCH_REGIME_TUPLE_IDX_END
 * corresponds to @ref
 * LW2080_CTRL_PMGR_PWR_POLICY_WORKLOAD_COMBINED_1X_SEARCH_REGIME::tupleEndStart.
 *
 * @{
 */
#define LW2080_CTRL_PMGR_PWR_POLICY_WORKLOAD_COMBINED_1X_SEARCH_REGIME_TUPLE_IDX_START 0x0U
#define LW2080_CTRL_PMGR_PWR_POLICY_WORKLOAD_COMBINED_1X_SEARCH_REGIME_TUPLE_IDX_END   0x1U
#define LW2080_CTRL_PMGR_PWR_POLICY_WORKLOAD_COMBINED_1X_SEARCH_REGIME_TUPLE_IDX_MAX   0x2U
/*! @} */

/*!
 * Structure representing clock search regime specific data
 */
typedef struct LW2080_CTRL_PMGR_PWR_POLICY_WORKLOAD_COMBINED_1X_SEARCH_REGIME {
    /*!
     * Collapsed start index of the clock search regime
     */
    LwU16                                                                      tupleStartIdx;
    /*!
     * Collapsed end index of the clock search regime
     */
    LwU16                                                                      tupleEndIdx;
    /*!
     * Lower frequency index of the regime start point.
     * The frequency index corresponds to the clock domain pointed to
     * by @ref primaryClkDomainIdx
     */
    LwU16                                                                      clkFreqStartIdx;
    /*!
     * Lower frequency index of the regime start point.
     * The frequency index corresponds to the clock domain pointed to
     * by @ref primaryClkDomainIdx
     */
    LwU16                                                                      clkFreqEndIdx;
    /*!
     * regime ID LW2080_CTRL_PMGR_PWR_POLICY_WORKLOAD_COMBINED_1X_SEARCH_REGIME_ID
     */
    LwU8                                                                       regimeId;
    /*!
     * @ref
     * LW2080_CTRL_PMGR_PWR_POLICY_WORKLOAD_COMBINED_1X_REGIME_SINGLES_FREQ_TUPLE
     * structures corresponding to @ref tupleStartIdx and
     * @tupleEndIdx.  Taken together, these tuples define the
     * frequency range defined by this SEARCH_REGIME.
     */
    LW2080_CTRL_PMGR_PWR_POLICY_WORKLOAD_COMBINED_1X_REGIME_SINGLES_FREQ_TUPLE freqTuples[LW2080_CTRL_PMGR_PWR_POLICY_WORKLOAD_COMBINED_1X_SEARCH_REGIME_TUPLE_IDX_MAX];
} LW2080_CTRL_PMGR_PWR_POLICY_WORKLOAD_COMBINED_1X_SEARCH_REGIME;
typedef struct LW2080_CTRL_PMGR_PWR_POLICY_WORKLOAD_COMBINED_1X_SEARCH_REGIME *PLW2080_CTRL_PMGR_PWR_POLICY_WORKLOAD_COMBINED_1X_SEARCH_REGIME;

/*!
 * Structure representing inforamation on the chosen search regime
 * at different stages of regime search
 */
typedef struct LW2080_CTRL_PMGR_PWR_POLICY_WORKLOAD_COMBINED_1X_REGIMES_STATUS_TUPLE {
    /*!
     * Index in @ref searchRegimes[] array which contains the @ref
     * tupleIdx to which the search collapsed.
     */
    LwU8  searchRegimeIdx;
    /*!
     * Tuple index to which the search collapsed.  This is the tuple,
     * as specified by one of the regimes in the @ref searchRegimes[]
     * array, which specifies the input frequencies which generated
     * the @ref estMet data.
     */
    LwU16 tupleIdx;
} LW2080_CTRL_PMGR_PWR_POLICY_WORKLOAD_COMBINED_1X_REGIMES_STATUS_TUPLE;

/*!
 * Macros to indicate the stages of clock regime selection
 */
#define LW2080_CTRL_PMGR_PWR_POLICY_WORKLOAD_COMBINED_1X_REGIMES_STATUS_TUPLE_IDX_SEARCH  0x0U
#define LW2080_CTRL_PMGR_PWR_POLICY_WORKLOAD_COMBINED_1X_REGIMES_STATUS_TUPLE_IDX_SCALING 0x1U
#define LW2080_CTRL_PMGR_PWR_POLICY_WORKLOAD_COMBINED_1X_REGIMES_STATUS_TUPLE_IDX_MAX     0x2U

/*!
 * Structure representing the clock search regime status
 */
typedef struct LW2080_CTRL_PMGR_PWR_POLICY_WORKLOAD_COMBINED_1X_REGIMES_STATUS {
    /*!
     * Structure to store information regarding the search regime
     * chosen at different stages of clock search
     */
    LW2080_CTRL_PMGR_PWR_POLICY_WORKLOAD_COMBINED_1X_REGIMES_STATUS_TUPLE tuples[LW2080_CTRL_PMGR_PWR_POLICY_WORKLOAD_COMBINED_1X_REGIMES_STATUS_TUPLE_IDX_MAX];
} LW2080_CTRL_PMGR_PWR_POLICY_WORKLOAD_COMBINED_1X_REGIMES_STATUS;

/*!
 * Structure of dynamic information specific to the WORKLOAD COMBINED_1X power policy.
 */
typedef struct LW2080_CTRL_PMGR_PWR_POLICY_STATUS_DATA_WORKLOAD_COMBINED_1X {
    /*!
     * Must always be first in structure.
     */
    LW2080_CTRL_PMGR_PWR_POLICY_STATUS_DATA_DOMGRP                                       domGrp;
    /*!
     * PERF_CF_PWR_MODEL interface
     */
    LW2080_CTRL_PMGR_PWR_POLICY_STATUS_DATA_PERF_CF_PWR_MODEL                            pwrModel;
    /*!
     * @copydoc LW2080_CTRL_PMGR_PWR_POLICY_PWR_MODEL_OBSERVED_METRICS_WORKLOAD_COMBINED_1X
     */
    LW2080_CTRL_PMGR_PWR_POLICY_PERF_CF_PWR_MODEL_OBSERVED_METRICS_WORKLOAD_COMBINED_1X  obsMet;
    /*!
     * @copydoc LW2080_CTRL_PMGR_PWR_POLICY_PWR_MODEL_ESTIMATED_METRICS_WORKLOAD_COMBINED_1X
     */
    LW2080_CTRL_PMGR_PWR_POLICY_PERF_CF_PWR_MODEL_ESTIMATED_METRICS_WORKLOAD_COMBINED_1X estMet;
    /*!
     * Number of steps taken back to satisfy policy limit after binary search
     * colwerges to a single frequency point
     */
    LwU8                                                                                 numStepsBack;
    /*!
     * @copydoc LW2080_CTRL_PMGR_PWR_POLICY_WORKLOAD_COMBINED_1X_REGIMES_STATUS
     */
    LW2080_CTRL_PMGR_PWR_POLICY_WORKLOAD_COMBINED_1X_REGIMES_STATUS                      regimesStatus;
    /*!
     * Search space of clock regimes
     * TODO - A follow-up CL may consider moving this into @ref
     * LW2080_CTRL_PMGR_PWR_POLICY_WORKLOAD_COMBINED_1X_REGIMES_STATUS.
     */
    LW2080_CTRL_PMGR_PWR_POLICY_WORKLOAD_COMBINED_1X_SEARCH_REGIME                       searchRegimes[LW2080_CTRL_PMGR_PWR_POLICY_WORKLOAD_COMBINED_1X_SEARCH_REGIME_ID_MAX];
    /*!
     * Total number of valid entries in @ref searchRegimes per evaluation
     * cycle
     * TODO - A follow-up CL may consider moving this into @ref
     * LW2080_CTRL_PMGR_PWR_POLICY_WORKLOAD_COMBINED_1X_REGIMES_STATUS.
     */
    LwU8                                                                                 numRegimes;
} LW2080_CTRL_PMGR_PWR_POLICY_STATUS_DATA_WORKLOAD_COMBINED_1X;
typedef struct LW2080_CTRL_PMGR_PWR_POLICY_STATUS_DATA_WORKLOAD_COMBINED_1X *PLW2080_CTRL_PMGR_PWR_POLICY_STATUS_DATA_WORKLOAD_COMBINED_1X;

/*!
 * Maximum number of PWR_POLICY_RELATIONSHIP_BALANCE objects which a
 * PWR_POLICY_BALANCE object can reference.  This value is used specify the
 * maximum array size possible for sorting buffer.
 */
#define LW2080_CTRL_PMGR_PWR_POLICY_STATUS_DATA_BALANCE_MAX_RELATIONSHIP_ENTRIES 4

/*!
 * Structure representing a a PWR_POLICY_RELATIONSHIP_BALANCE object and its
 * current values when used to sort the set of objects belonging to a particular
 * PWR_POLICY_BALANCE object.
 */
typedef struct LW2080_CTRL_PMGR_PWR_POLICY_STATUS_DATA_BALANCE_RELATIONSHIP_ENTRY {
    /*!
     * Index of PWR_POLICY_RELATIONSHIP_BALANCE object corresponding to this entry.
     */
    LwU8  relIdx;
    /*!
     * Lower requested limit from either of the two PWR_POLICY objects to which
     * the corresponding PWR_POLICY_RELATIONSHIP_BALANCE object points.
     */
    LwU32 limitLower;
    /*!
     * Difference between lower and higher limits of the two PWR_POLICY objects
     * to which the corresponding PWR_POLICY_RELATIONSHIP_BALANCE object points.
     */
    LwU32 limitDiff;
} LW2080_CTRL_PMGR_PWR_POLICY_STATUS_DATA_BALANCE_RELATIONSHIP_ENTRY;

/*!
 * Structure representing all dynamic status corresponding to a
 * PWR_POLICY_BALANCE object.
 */
typedef struct LW2080_CTRL_PMGR_PWR_POLICY_STATUS_DATA_BALANCE {
    /*!
     * Sorted array of PWR_POLICY_RELATIONSHIP_BALANCE objects, reflecting the
     * sorting applied at the last iteration the PWR_POLICY_BALANCE object was
     * evaluated.
     */
    LW2080_CTRL_PMGR_PWR_POLICY_STATUS_DATA_BALANCE_RELATIONSHIP_ENTRY relEntries[LW2080_CTRL_PMGR_PWR_POLICY_STATUS_DATA_BALANCE_MAX_RELATIONSHIP_ENTRIES];
} LW2080_CTRL_PMGR_PWR_POLICY_STATUS_DATA_BALANCE;

/*!
 * Structure representing all dynamic status corresponding to a PWR_POLICY_LIMIT
 * object.
 */
typedef struct LW2080_CTRL_PMGR_PWR_POLICY_STATUS_DATA_PROP_LIMIT {
    /*!
     * Boolean indicating whether this PWR_POLICY_PROP_LIMIT object has been
     * dirtied by the @ref PWR_POLICY_BALANCE algorithm (via @ref
     * PWR_POLICY_RELATIONSHIP_BALANCE) and thus its capping request may be
     * inaclwrate.
     */
    LwBool bBalanceDirty;
} LW2080_CTRL_PMGR_PWR_POLICY_STATUS_DATA_PROP_LIMIT;

/*!
 * Enumeration of actions the PWR_GEMINI controller can take in an iteration.
 *
 * _TO_SLAVE - One delta step shifted to the SLAVE GPU
 * _NONE - No action
 * _MASTER - One delta step shifted to the MASTER GPU
 */
#define LW2080_CTRL_PMGR_PWR_POLICY_GEMINI_ACTION_TO_SLAVE  (-1)
#define LW2080_CTRL_PMGR_PWR_POLICY_GEMINI_ACTION_NONE      0U
#define LW2080_CTRL_PMGR_PWR_POLICY_GEMINI_ACTION_TO_MASTER 1U

/*!
 * Structure representing the dynamic state of a PWR_POLICY_GEMINI object/GPU
 * within a PWR_GEMINI controller.
 */
typedef struct LW2080_CTRL_PMGR_PWR_POLICY_STATUS_DATA_GEMINI_CONTROLLER_GPU {
    /*!
     * The average frequency (kHz) for the previous sampling iteration.
     */
    LwU32 freqAvgkHz;
} LW2080_CTRL_PMGR_PWR_POLICY_STATUS_DATA_GEMINI_CONTROLLER_GPU;

/*!
 * Structure representing the dynamic state of a PWR_GEMINI object.
 */
typedef struct LW2080_CTRL_PMGR_PWR_POLICY_STATUS_DATA_GEMINI_CONTROLLER {
    /*!
     * Balancing action taken during the previous iteration.  Stored here so it
     * can returned via @ref LW2080_CTRL_CMD_PMGR_PWR_POLICY_GET_STATUS for
     * debugging.
     *
     * @ref LW2080_CTRL_PMGR_PWR_POLICY_GEMINI_ACTION_<xyz>
     */
    LwS8                                                          action;

    /*!
     * Flag indicating whether the PWR_GEMINI algorithm is lwrrently enabled.
     */
    LwBool                                                        bEnable;

    /*!
     * The limit delta value lwrrently shifted to the MASTER gpu.  The SLAVE gpu
     * delta is just this value negated.
     */
    LwS32                                                         limitDelta;

    /*!
     * GPU-specific state.
     */
    LW2080_CTRL_PMGR_PWR_POLICY_STATUS_DATA_GEMINI_CONTROLLER_GPU gpus[LW2080_CTRL_PMGR_PWR_POLICY_GEMINI_MAX_GPUS];
} LW2080_CTRL_PMGR_PWR_POLICY_STATUS_DATA_GEMINI_CONTROLLER;

/*!
 * Structure represnting the dynamic state associated with a PWR_POLICY_GEMINI
 * policy.
 */
typedef struct LW2080_CTRL_PMGR_PWR_POLICY_STATUS_DATA_GEMINI {
    /*!
     * Status of the PWR_GEMINI object associated with this PWR_POLICY_GEMINI
     * object.
     */
    LW2080_CTRL_PMGR_PWR_POLICY_STATUS_DATA_GEMINI_CONTROLLER controller;
} LW2080_CTRL_PMGR_PWR_POLICY_STATUS_DATA_GEMINI;

/*!
 * PWR_POLICY_TOTAL_GPU Interface
 */
typedef struct LW2080_CTRL_PMGR_PWR_POLICY_STATUS_DATA_TOTAL_GPU_INTERFACE {
    /*!
     * BOARDOBJ_INTERFACE super class.  Must always be first element in the structure.
     */
    LW2080_CTRL_BOARDOBJ_INTERFACE super;
} LW2080_CTRL_PMGR_PWR_POLICY_STATUS_DATA_TOTAL_GPU_INTERFACE;

/*!
 * Structure represnting the dynamic state associated with a PWR_POLICY_TOTAL_GPU
 * policy.
 */
typedef struct LW2080_CTRL_PMGR_PWR_POLICY_STATUS_DATA_TOTAL_GPU {
    /*!
     * Dynamic state associated with this policy's pff interface.
     */
    LW2080_CTRL_PMGR_PFF_STATUS                                 pff;
    /*!
     * Dynamic state associated with this TOTAL GPU Interface
     */
    LW2080_CTRL_PMGR_PWR_POLICY_STATUS_DATA_TOTAL_GPU_INTERFACE tgpIface;
} LW2080_CTRL_PMGR_PWR_POLICY_STATUS_DATA_TOTAL_GPU;

/*!
 * Union of type-specific data.
 */
typedef union LW2080_CTRL_PMGR_PWR_POLICY_STATUS_DATA {
    LW2080_CTRL_PMGR_PWR_POLICY_STATUS_DATA_DOMGRP               domGrp;
    LW2080_CTRL_PMGR_PWR_POLICY_STATUS_DATA_BANG_BANG_VF         bangBangVf;
    LW2080_CTRL_PMGR_PWR_POLICY_STATUS_DATA_MARCH_VF             marchVF;
    LW2080_CTRL_PMGR_PWR_POLICY_STATUS_DATA_WORKLOAD             workload;
    LW2080_CTRL_PMGR_PWR_POLICY_STATUS_DATA_WORKLOAD_MULTIRAIL   workloadMulRail;
    LW2080_CTRL_PMGR_PWR_POLICY_STATUS_DATA_WORKLOAD_SINGLE_1X   single1x;
    LW2080_CTRL_PMGR_PWR_POLICY_STATUS_DATA_WORKLOAD_COMBINED_1X combined1x;
    LW2080_CTRL_PMGR_PWR_POLICY_STATUS_DATA_BALANCE              balance;
    LW2080_CTRL_PMGR_PWR_POLICY_STATUS_DATA_PROP_LIMIT           propLimit;
    LW2080_CTRL_PMGR_PWR_POLICY_STATUS_DATA_GEMINI               gemini;
    LW2080_CTRL_PMGR_PWR_POLICY_STATUS_DATA_TOTAL_GPU            totalGpu;
} LW2080_CTRL_PMGR_PWR_POLICY_STATUS_DATA;


/*!
 * Union of type-specific data. This is similar to
 * @ref LW2080_CTRL_PMGR_PWR_POLICY_STATUS_DATA except it does not have
 * @ref LW2080_CTRL_PMGR_PWR_POLICY_STATUS_DATA_WORKLOAD_COMBINED_1X and
 * @ref LW2080_CTRL_PMGR_PWR_POLICY_STATUS_DATA_WORKLOAD_SINGLE_1X. This is
 * used ONLY in @ref s_pwrPoliciesGpuMonSampleSizeGet to callwlate size of
 * LW2080_CTRL_PMGR_GPUMON_PWR_SAMPLE structure to ensure older chips donot
 * have a size increase impact.
 *
 * @note - Do not add new elements to this structure unless it is applicable
 * to pre PMU_CFG_PWR_POLICY_35
 */
typedef union LW2080_CTRL_PMGR_GPUMON_POLICY_30_PWR_POLICY_STATUS_DATA {
    LW2080_CTRL_PMGR_PWR_POLICY_STATUS_DATA_DOMGRP             domGrp;
    LW2080_CTRL_PMGR_PWR_POLICY_STATUS_DATA_BANG_BANG_VF       bangBangVf;
    LW2080_CTRL_PMGR_PWR_POLICY_STATUS_DATA_MARCH_VF           marchVF;
    LW2080_CTRL_PMGR_PWR_POLICY_STATUS_DATA_WORKLOAD           workload;
    LW2080_CTRL_PMGR_PWR_POLICY_STATUS_DATA_WORKLOAD_MULTIRAIL workloadMulRail;
    LW2080_CTRL_PMGR_PWR_POLICY_STATUS_DATA_BALANCE            balance;
    LW2080_CTRL_PMGR_PWR_POLICY_STATUS_DATA_PROP_LIMIT         propLimit;
    LW2080_CTRL_PMGR_PWR_POLICY_STATUS_DATA_GEMINI             gemini;
    LW2080_CTRL_PMGR_PWR_POLICY_STATUS_DATA_TOTAL_GPU          totalGpu;
} LW2080_CTRL_PMGR_GPUMON_POLICY_30_PWR_POLICY_STATUS_DATA;


/*!
 * Structure of representing the dynamic state associated with a PWR_POLICY
 * entry.
 */
typedef struct LW2080_CTRL_PMGR_PWR_POLICY_STATUS {
    /*!
     * @ref LW2080_CTRL_PMGR_PWR_POLICY_TYPE_<xyz>.
     */
    LwU8                                          type;

    /*!
     * Current limit value this PWR_POLICY object is enforcing.  This is the
     * effective of all the arbitrated output (lowest) of all the
     * LW2080_CTRL_PMGR_PWR_POLICY_LIMIT_INPUT entries specified in @ref
     * limitArb* below and any other policy specific constrain.
     * Will always be within range of [limitMin, limitMax].
     */
    LwU32                                         limitLwrr;
    /*!
     * Current value retrieved from the monitored PWR_CHANNEL.
     */
    LwU32                                         valueLwrr;

    /*!
     * Delta which should be applied to arbitrated output of all limit values.
     * This balancing delta is adjusted via inter-GPU balancing algorithms which
     * are trying to optimize the power distribution to achieve maximal perf for
     * the given power limits.  Those algorithms will be transferring power
     * from one GPU to another behind the back of the user, so we don't want to
     * actually change the specified limit values or report this value back to
     * the user.
     */
    LwS32                                         limitDelta;

    /*!
     * Integral control status.
     */
    LW2080_CTRL_PMGR_PWR_POLICY_STATUS_INTEGRAL   integral;

    /*!
     * Current state of RM PWR_POLICY object's limit arbitration structure.  The
     * output of this arbitration structure is passed to the PMU PWR_POLICY
     * object via clientIdx @ref
     * LW2080_CTRL_PMGR_PWR_POLICY_LIMIT_INPUT_CLIENT_IDX_RM in the @ref
     * limitInputs structure below.
     */
    LW2080_CTRL_PMGR_PWR_POLICY_LIMIT_ARBITRATION limitArbRmLwrr;

    /*!
     * Current state of RM PWR_POLICY object's limit arbitration structure. The
     * arbitrated output of which will assigned to _LIMIT_ID_LWRR on AC->BATT
     * transitions with LW2080_CTRL_PMGR_PWR_POLICY_LIMIT_INPUT_CLIENT_IDX_BATT.
     *
     * All inputs and the output must always be within range of arbitrated
     * output of [@ref limitMin, @ref limitBattMax].
     */
    LW2080_CTRL_PMGR_PWR_POLICY_LIMIT_ARBITRATION limitArbRmBattLwrr;

    /*!
     * Current state of PWR_POLICY object's limit arbitration structure.  The
     * output of this arbitration structure is the PMU PWR_POLICY
     * object's current limit to enforce.
     *
     * The arbitrated output of the RM is passed via clientIdx @ref
     * LW2080_CTRL_PMGR_PWR_POLICY_LIMIT_INPUT_CLIENT_IDX_RM.
     */
    LW2080_CTRL_PMGR_PWR_POLICY_LIMIT_ARBITRATION limitArbPmuLwrr;

    /*!
     * Type-specific dynamic state.
     */
    LW2080_CTRL_PMGR_PWR_POLICY_STATUS_DATA       data;
} LW2080_CTRL_PMGR_PWR_POLICY_STATUS;

/*!
 * Structure defining the dynamic state of a PWR_POLICY_PROP_LIMIT object when
 * the PWR_POLICY_RELATIONSHIP_BALANCE object tries to balance power between it
 * and its pair object.
 */
typedef struct LW2080_CTRL_PMGR_PWR_POLICY_RELATIONSHIP_STATUS_BALANCE_PROP_LIMIT {
    /*!
     * Boolean indicating whether this PWR_POLICY_PROP_LIMIT object has been
     * dirtied by the @ref PWR_POLICY_BALANCE algorithm (via @ref
     * PWR_POLICY_RELATIONSHIP_BALANCE) and thus its capping request may be
     * inaclwrate.
     */
    LwU8  bBalanceDirty;
    /*!
     * The input value (@ref PWR_POLICY::valueLwrr) of the
     * PWR_POLICY_PROP_LIMIT object when evaluation begins.
     */
    LwS32 valueOld;
    /*!
     * The updated input value (@ref PWR_POLICY::valueLwrr) of the
     * PWR_POLICY_PROP_LIMIT object after evaluation completes.
     */
    LwS32 valueNew;
    /*!
     * The requested limit (via @ref PWR_POLICY_RELATIONSHIP) of the
     * PWR_POLICY_PROP_LIMIT object when evaluation begins.
     */
    LwS32 limitRequestOld;
    /*!
     * The updated requested limit (via @ref PWR_POLICY_RELATIONSHIP) of the
     * PWR_POLICY_PROP_LIMIT object after evaluation completes.
     */
    LwS32 limitRequestNew;
    /*!
     * Count of the observed violation on the given power channel that is being
     * monitored and controlled by this prop limit policy.
     */
    LwU32 violCnt;
} LW2080_CTRL_PMGR_PWR_POLICY_RELATIONSHIP_STATUS_BALANCE_PROP_LIMIT;

/*!
 * Macros defining the actions the PWR_POLICY_RELATIONSHIP_BALANCE object took
 * for the most recent iteration.
 *
 * _NONE - No action was taken to shift power.
 * _TO_PRI - Power was shifted to the primary PWR_POLICY_PROP_LIMIT object.
 * _TO_SEC - Power was shifted to the secondary PWR_POLICY_PROP_LIMIT object.
 */
#define LW2080_CTRL_PMGR_PWR_POLICY_RELATIONSHIP_STATUS_BALANCE_ACTION_TO_PRI   1U
#define LW2080_CTRL_PMGR_PWR_POLICY_RELATIONSHIP_STATUS_BALANCE_ACTION_NONE     0U
#define LW2080_CTRL_PMGR_PWR_POLICY_RELATIONSHIP_STATUS_BALANCE_ACTION_TO_SEC   (-1)

/*!
 * Macros for indexes into @ref
 * LW2080_CTRL_PMGR_PWR_POLICY_RELATIONSHIP_STATUS_BALANCE::propLimits[] array.
 *
 * _PRI - The primary policy index - i.e. PWR_POLICY_RELATIONSHIP::policyIdx
 * _SEC - The secondary policy index - i.e.
 *      PWR_POLICY_RELATIONSHIP_BALANCE::secPolicyIdx.
 */
#define LW2080_CTRL_PMGR_PWR_POLICY_RELATIONSHIP_STATUS_BALANCE_PROP_LIMIT_PRI  0x0U
#define LW2080_CTRL_PMGR_PWR_POLICY_RELATIONSHIP_STATUS_BALANCE_PROP_LIMIT_SEC  0x1U
#define LW2080_CTRL_PMGR_PWR_POLICY_RELATIONSHIP_STATUS_BALANCE_MAX_PROP_LIMITS 0x2U

/*!
 * Structure representing all dynamic status corresponding to a
 * PWR_POLICY_RELATIONSHIP_BALANCE object.
 */
typedef struct LW2080_CTRL_PMGR_PWR_POLICY_RELATIONSHIP_STATUS_BALANCE {
    /*!
     * @ref LW2080_CTRL_PMGR_PWR_POLICY_RELATIONSHIP_STATUS_BALANCE_ACTION_<xyz>
     */
    LwS8                                                               action;
    /*!
     * Current PWM percent driven out on the balancing circuit's GPIO.
     */
    LwUFXP16_16                                                        pwmPctLwrr;
    /*!
     * Most recently evaluated violation rate (normalized to [0.0,1.0]).
     */
    LwUFXP4_12                                                         violLwrrent;
    /*!
     * Latest value retrieved from the monitored PWR_CHANNEL ref@ phaseEstimateChIdx.
     */
    LwU32                                                              phaseEstimateValueLwrr;
    /*!
     * Array of structures describing the dynamic state of both
     * PWR_POLICY_PROP_LIMIT objects.
     */
    LW2080_CTRL_PMGR_PWR_POLICY_RELATIONSHIP_STATUS_BALANCE_PROP_LIMIT propLimits[LW2080_CTRL_PMGR_PWR_POLICY_RELATIONSHIP_STATUS_BALANCE_MAX_PROP_LIMITS];
} LW2080_CTRL_PMGR_PWR_POLICY_RELATIONSHIP_STATUS_BALANCE;

/*!
 * Union of type-specific data.
 */


/*!
 * Structure of representing the dynamic state associated with a PWR_POLICY_RELATIONSHIP
 * entry.
 */
typedef struct LW2080_CTRL_PMGR_PWR_POLICY_RELATIONSHIP_STATUS {
    /*!
     * @ref LW2080_CTRL_PMGR_PWR_POLICY_RELATIONSHIP_TYPE_<xyz>.
     */
    LwU8 type;

    /*!
     * Type-specific dynamic state.
     */
    union {
        LW2080_CTRL_PMGR_PWR_POLICY_RELATIONSHIP_STATUS_BALANCE balance;
    } data;
} LW2080_CTRL_PMGR_PWR_POLICY_RELATIONSHIP_STATUS;

/*!
 * Structure describing the dynamic state associated with the _PROPGAIN class.
 */
typedef struct LW2080_CTRL_PMGR_PWR_VIOLATION_STATUS_PROPGAIN {
    /*!
     * Computed limit adjustment by scaling gain and normalizing units.
     */
    LwS32 pwrLimitAdj;
} LW2080_CTRL_PMGR_PWR_VIOLATION_STATUS_PROPGAIN;
typedef struct LW2080_CTRL_PMGR_PWR_VIOLATION_STATUS_PROPGAIN *PLW2080_CTRL_PMGR_PWR_VIOLATION_STATUS_PROPGAIN;

/*!
 * PWR_VIOLATION type-specific data union.  Discriminated by
 * PWR_VIOLATION::super.type.
 */


/*!
 * Structure describing PWR_VIOLATION dynamic state information. Implements the
 * BOARDOBJ model/interface.
 */
typedef struct LW2080_CTRL_PMGR_PWR_VIOLATION_STATUS {
    /*!
     * XAPI does not support using an element within another structure as a
     * discriminant.  Placing this redundant type value here until that design
     * constraint can be fixed.
     */
    LwU8       type;
    /*!
     * Most recently evaluated violation rate (normalized to [0.0,1.0]).
     */
    LwUFXP4_12 violLwrrent;
    /*!
     * Type-specific data union.
     */
    union {
        LW2080_CTRL_PMGR_PWR_VIOLATION_STATUS_PROPGAIN propGain;
    } data;
} LW2080_CTRL_PMGR_PWR_VIOLATION_STATUS;
typedef struct LW2080_CTRL_PMGR_PWR_VIOLATION_STATUS *PLW2080_CTRL_PMGR_PWR_VIOLATION_STATUS;

/*!
 * LW2080_CTRL_CMD_PMGR_PWR_POLICY_GET_STATUS
 *
 * This command returns the dynamic status of a set of client-specified
 * PWR_POLICY entries in the Power Policy Table.
 *
 * https://wiki.lwpu.com/engwiki/index.php/Resman/PState/Data_Tables/Power_Tables/Power_Policy_Table_2.X
 *
 * See LW2080_CTRL_PMGR_PWR_POLICY_STATUS_PARAMS for documentation on the
 * parameters
 *
 * Possible status values returned are
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW2080_CTRL_CMD_PMGR_PWR_POLICY_GET_STATUS (0x20802619U) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PMGR_INTERFACE_ID << 8) | LW2080_CTRL_PMGR_PWR_POLICY_STATUS_PARAMS_MESSAGE_ID" */

/*!
 * Structure representing the status of PWR_RELATIONSHIPS_STATUS BOARDOBJGRP.
 */
typedef struct LW2080_CTRL_PMGR_PWR_POLICY_RELATIONSHIPS_STATUS {
    /*!
     * LW2080_CTRL_BOARDOBJGRP_E32 super class. Must always be first object in
     * structure.
     */
    LW2080_CTRL_BOARDOBJGRP_E32                     super;
    /*!
     * [out] - Array of PWR_POLICY_RELATIONSHIP entries.  Has valid indexes
     * corresponding to the bits set in @ref policyRelMask.
     */
    LW2080_CTRL_PMGR_PWR_POLICY_RELATIONSHIP_STATUS policyRels[LW2080_CTRL_PMGR_PWR_POLICY_MAX_POLICY_RELATIONSHIPS];
} LW2080_CTRL_PMGR_PWR_POLICY_RELATIONSHIPS_STATUS;

/*!
 * Structure representing the status of PWR_VIOLATIONS_STATUS BOARDOBJGRP.
 */
typedef struct LW2080_CTRL_PMGR_PWR_VIOLATIONS_STATUS {
    /*!
     * LW2080_CTRL_BOARDOBJGRP_E32 super class. Must always be first object in
     * structure.
     */
    LW2080_CTRL_BOARDOBJGRP_E32           super;
    /*!
     * [out] - Array of PWR_VIOLATION structures.  Has valid indexes corresponding to the
     * bits set in @ref super.objMask.
     */
    LW2080_CTRL_PMGR_PWR_VIOLATION_STATUS violations[LW2080_CTRL_PMGR_PWR_VIOLATION_MAX];
} LW2080_CTRL_PMGR_PWR_VIOLATIONS_STATUS;

/*!
 * @brief   Mask of requestable IDs for disabling inflection points.
 */
typedef LW2080_CTRL_BOARDOBJGRP_MASK_E32 LW2080_CTRL_PMGR_PWR_POLICIES_INFLECTION_POINTS_DISABLE_REQUEST_ID_MASK;

/*!
 * @defgroup LW2080_CTRL_PMGR_PWR_POLICIES_INFLECTION_POINTS_DISABLE_REQUEST_ID
 *
 * @brief   Enumeration of requestable IDs for why PWR_POLICY inflection points
 *          can be disabled.
 *
 * @details
 *      LW2080_CTRL_PMGR_PWR_POLICIES_INFLECTION_POINTS_DISABLE_REQUEST_ID_API
 *          Client requested inflection points be disabled via the SET_CONTROL
 *          API.
 *
 *      LW2080_CTRL_PMGR_PWR_POLICIES_INFLECTION_POINTS_DISABLE_REQUEST_ID_PERF_CF_CONTROLLERS__START
 *          Number of first REQUEST_ID in a segment of REQUEST_IDs that are for
 *          individual @ref LW2080_CTRL_PERF_PERF_CF_CONTROLLER_INFO objects
 *
 *      LW2080_CTRL_PMGR_PWR_POLICIES_INFLECTION_POINTS_DISABLE_REQUEST_ID_PERF_CF_CONTROLLERS_CONTROLLER0
 *          Request from an individual
 *          @ref LW2080_CTRL_PERF_PERF_CF_CONTROLLER_INFO object.
 *
 *      LW2080_CTRL_PMGR_PWR_POLICIES_INFLECTION_POINTS_DISABLE_REQUEST_ID_PERF_CF_CONTROLLERS__END
 *          Number of last REQUEST_ID in a segment of REQUEST_IDs that are for
 *          individual @ref LW2080_CTRL_PERF_PERF_CF_CONTROLLER_INFO objects
 * @{
 */
typedef LwBoardObjIdx LW2080_CTRL_PMGR_PWR_POLICIES_INFLECTION_POINTS_DISABLE_REQUEST_ID;
#define LW2080_CTRL_PMGR_PWR_POLICIES_INFLECTION_POINTS_DISABLE_REQUEST_ID_API                             0U
#define LW2080_CTRL_PMGR_PWR_POLICIES_INFLECTION_POINTS_DISABLE_REQUEST_ID_PERF_CF_CONTROLLERS__START      1U
#define LW2080_CTRL_PMGR_PWR_POLICIES_INFLECTION_POINTS_DISABLE_REQUEST_ID_PERF_CF_CONTROLLERS_CONTROLLER0 (0x1U) /* finn: Evaluated from "(LW2080_CTRL_PMGR_PWR_POLICIES_INFLECTION_POINTS_DISABLE_REQUEST_ID_PERF_CF_CONTROLLERS__START + 0)" */
#define LW2080_CTRL_PMGR_PWR_POLICIES_INFLECTION_POINTS_DISABLE_REQUEST_ID_PERF_CF_CONTROLLERS_CONTROLLER1 (0x2U) /* finn: Evaluated from "(LW2080_CTRL_PMGR_PWR_POLICIES_INFLECTION_POINTS_DISABLE_REQUEST_ID_PERF_CF_CONTROLLERS__START + 1)" */
#define LW2080_CTRL_PMGR_PWR_POLICIES_INFLECTION_POINTS_DISABLE_REQUEST_ID_PERF_CF_CONTROLLERS_CONTROLLER2 (0x3U) /* finn: Evaluated from "(LW2080_CTRL_PMGR_PWR_POLICIES_INFLECTION_POINTS_DISABLE_REQUEST_ID_PERF_CF_CONTROLLERS__START + 2)" */
#define LW2080_CTRL_PMGR_PWR_POLICIES_INFLECTION_POINTS_DISABLE_REQUEST_ID_PERF_CF_CONTROLLERS_CONTROLLER3 (0x4U) /* finn: Evaluated from "(LW2080_CTRL_PMGR_PWR_POLICIES_INFLECTION_POINTS_DISABLE_REQUEST_ID_PERF_CF_CONTROLLERS__START + 3)" */
#define LW2080_CTRL_PMGR_PWR_POLICIES_INFLECTION_POINTS_DISABLE_REQUEST_ID_PERF_CF_CONTROLLERS__END        LW2080_CTRL_PMGR_PWR_POLICIES_INFLECTION_POINTS_DISABLE_REQUEST_ID_PERF_CF_CONTROLLERS_CONTROLLER3
#define LW2080_CTRL_PMGR_PWR_POLICIES_INFLECTION_POINTS_DISABLE_REQUEST_ID_NUM                             (0x5U) /* finn: Evaluated from "(LW2080_CTRL_PMGR_PWR_POLICIES_INFLECTION_POINTS_DISABLE_REQUEST_ID_PERF_CF_CONTROLLERS__END + 1)" */
/*!@}*/

/*!
 * @brief   Returns whether a particular inflection point disable request ID has
 *          been set.
 *
 * @param[in]   _pRequestIdMask @ref LW2080_CTRL_PMGR_PWR_POLICIES_INFLECTION_POINTS_DISABLE_REQUEST_ID_MASK
 *                              in which to check request ID
 * @param[in]   _requestId      @ref LW2080_CTRL_PMGR_PWR_POLICIES_INFLECTION_POINTS_DISABLE_REQUEST_ID
 *                              to check
 *
 * @return  @ref LW_TRUE    _requestId has been requested in _pRequestIdMask
 * @return  @ref LW_FALSE   Otherwise
 */
#define LW2080_CTRL_PMGR_PWR_POLICIES_INFLECTION_POINTS_DISABLE_REQUEST_ID_MASK_REQUEST_ID_REQUESTED(_pRequestIdMask, _requestId) \
    LW2080_CTRL_BOARDOBJGRP_MASK_BIT_GET(&(_pRequestIdMask)->super, (_requestId))

/*!
 * Represents a request to disable inflection points
 */
typedef struct LW2080_CTRL_PMGR_PWR_POLICIES_INFLECTION_POINTS_DISABLE_REQUEST {
    /*!
     * If request based on PWR_POLICY data, timestamp at which that data was
     * sampled.
     *
     * Should be set to
     * @ref LW2080_CTRL_PMGR_PWR_POLICIES_INFLECTION_POINTS_DISABLE_REQUEST_TIMESTAMP_INFINITELY_RECENT
     * if the request should never be ilwalidated.
     */
    LwU64_ALIGN32 timestamp;

    /*!
     * Client-specific data for this request, generally used for debugging. Not
     * read by the @ref PWR_POLICIES.
     */
    LwU32         clientData;

    /*!
     * The @ref LwBoardObjIdx of the lowset @ref LW2080_CTRL_PERF_PSTATE_INFO
     * for which this request would like to disable inflection.
     */
    LwBoardObjIdx pstateIdxLowest;
} LW2080_CTRL_PMGR_PWR_POLICIES_INFLECTION_POINTS_DISABLE_REQUEST;

/*!
 * Sentinel value for
 * @ref LW2080_CTRL_PMGR_PWR_POLICIES_INFLECTION_POINTS_DISABLE_REQUEST::timestamp
 * to indicate that the timestamp should never be ilwalidated.
 */
#define LW2080_CTRL_PMGR_PWR_POLICIES_INFLECTION_POINTS_DISABLE_REQUEST_TIMESTAMP_INFINITELY_RECENT LW_U64_MAX

/*!
 * @brief   Initializes an
 *          @ref LW2080_CTRL_PMGR_PWR_POLICIES_INFLECTION_POINTS_DISABLE_REQUEST
 *          structure.
 *
 * @param[out]  _pRequest   @ref LW2080_CTRL_PMGR_PWR_POLICIES_INFLECTION_POINTS_DISABLE_REQUEST
 *                          structure to initialize
 */
#define LW2080_CTRL_PMGR_PWR_POLICIES_INFLECTION_POINTS_DISABLE_REQUEST_INIT(_pRequest) \
    do \
    { \
        (_pRequest)->timestamp = (LwU64_ALIGN32){0U}; \
        (_pRequest)->clientData = 0U; \
        (_pRequest)->pstateIdxLowest = LW2080_CTRL_PERF_PSTATE_INDEX_ILWALID; \
    } while (LW_FALSE)

/*!
 * @brief   Determines if a
 *          @ref LW2080_CTRL_PMGR_PWR_POLICIES_INFLECTION_POINTS_DISABLE_REQUEST
 *          structure is valid.

 * @param[out]  _pRequest   @ref LW2080_CTRL_PMGR_PWR_POLICIES_INFLECTION_POINTS_DISABLE_REQUEST
 *                          structure check
 */
#define LW2080_CTRL_PMGR_PWR_POLICIES_INFLECTION_POINTS_DISABLE_REQUEST_VALID(_pRequest) \
    ((_pRequest)->pstateIdxLowest != LW2080_CTRL_PERF_PSTATE_INDEX_ILWALID)

/*!
 * @brief   Structure to control whether PWR_POLICY inflection points should be
 *          disabled
 */
typedef struct LW2080_CTRL_PMGR_PWR_POLICIES_INFLECTION_POINTS_DISABLE {
    /*!
     * Mask of IDs for which disablement has been requested.
     *
     * Set bits correspond to valid entries in
     * @ref LW2080_CTRL_PMGR_PWR_POLICIES_INFLECTION_POINTS_DISABLE::requests
     */
    LW2080_CTRL_PMGR_PWR_POLICIES_INFLECTION_POINTS_DISABLE_REQUEST_ID_MASK requestIdMask;

    /*!
     * Request data for each disablement request ID.
     *
     * @note    Indexed by
     *          @ref LW2080_CTRL_PMGR_PWR_POLICIES_INFLECTION_POINTS_DISABLE_REQUEST_ID
     */
    LW2080_CTRL_PMGR_PWR_POLICIES_INFLECTION_POINTS_DISABLE_REQUEST         requests[LW2080_CTRL_PMGR_PWR_POLICIES_INFLECTION_POINTS_DISABLE_REQUEST_ID_NUM];
} LW2080_CTRL_PMGR_PWR_POLICIES_INFLECTION_POINTS_DISABLE;

/*!
 * Structure representing the dynamic status information associated with a set
 * of PWR_POLICYs within the GPU's PWR_POLICY power policy functionality.
 */
#define LW2080_CTRL_PMGR_PWR_POLICY_STATUS_PARAMS_MESSAGE_ID (0x19U)

typedef struct LW2080_CTRL_PMGR_PWR_POLICY_STATUS_PARAMS {
    /*!
     * LW2080_CTRL_BOARDOBJGRP_E32 super class. Must always be first object in
     * structure.
     */
    LW2080_CTRL_BOARDOBJGRP_E32                             super;
    /*!
     * [out] - The current arbitrated output domain group limit values the PMU
     * PWR_POLICY functionality is applying to the GPU.  The current arbitration
     * algorithm is to apply the minimum values of all PWR_POLICYs.  The RM
     * feeds this value into the PERF PERF_LIMIT infrastructure for clock
     * arbitration.
     *
     * For more information see @ref LW2080_CTRL_CMD_PERF_LIMITS_GET_STATUS and
     * @ref ctrl2080perf.h
     */
    LW2080_CTRL_PMGR_PWR_POLICY_STATUS_DOMAIN_GROUP_LIMITS  domGrpLimits;

    /*!
     * Collection of current inflection point disablement status.
     */
    LW2080_CTRL_PMGR_PWR_POLICIES_INFLECTION_POINTS_DISABLE inflectionPointsDisable;

    /*!
     * [out] - Array of PWR_POLICY entries.  Has valid indexes corresponding to
     * the bits set in @ref policyMask.
     */
    LW2080_CTRL_PMGR_PWR_POLICY_STATUS                      policies[LW2080_CTRL_PMGR_PWR_POLICY_MAX_POLICIES];
    /*!
     * Nesting the PWR_RELATIONSHIPS_STATUS
     */
    LW2080_CTRL_PMGR_PWR_POLICY_RELATIONSHIPS_STATUS        policyRelStatus;
    /*!
     * Nesting the PWR_VIOLATIONS_STATUS
     */
    LW2080_CTRL_PMGR_PWR_VIOLATIONS_STATUS                  violStatus;
} LW2080_CTRL_PMGR_PWR_POLICY_STATUS_PARAMS;

/*!
 * Structure representing WORKLOAD-specifc PWR_POLICY control/policy parameters.
 */
typedef struct LW2080_CTRL_PMGR_PWR_POLICY_CONTROL_DATA_WORKLOAD {
    /*!
     * Ratio to by which scale clock changes up.
     */
    LwUFXP4_12 clkUpScale;
    /*!
     * Ratio to by which scale clock changes down.
     */
    LwUFXP4_12 clkDownScale;
} LW2080_CTRL_PMGR_PWR_POLICY_CONTROL_DATA_WORKLOAD;

/*!
 * Structure representing WORKLOAD_MULTIRAIL_INTERFACE-specifc PWR_POLICY control/policy
 * parameters.
 */
typedef struct LW2080_CTRL_PMGR_PWR_POLICY_CONTROL_DATA_WORKLOAD_MULTIRAIL_INTERFACE {
    /*!
     * BOARDOBJ_INTERFACE super class.  Must always be first element in the structure.
     */
    LW2080_CTRL_BOARDOBJ_INTERFACE super;
} LW2080_CTRL_PMGR_PWR_POLICY_CONTROL_DATA_WORKLOAD_MULTIRAIL_INTERFACE;

/*!
 * Structure representing WORKLOAD_MULTIRAIL-specifc PWR_POLICY control/policy
 * parameters.
 */
typedef struct LW2080_CTRL_PMGR_PWR_POLICY_CONTROL_DATA_WORKLOAD_MULTIRAIL {
    /*!
     * Ratio to by which scale clock changes up.
     */
    LwUFXP4_12                                                            clkUpScale;
    /*!
     * Ratio to by which scale clock changes down.
     */
    LwUFXP4_12                                                            clkDownScale;
    /*!
     * Workload interface.
     */
    LW2080_CTRL_PMGR_PWR_POLICY_CONTROL_DATA_WORKLOAD_MULTIRAIL_INTERFACE workIface;
} LW2080_CTRL_PMGR_PWR_POLICY_CONTROL_DATA_WORKLOAD_MULTIRAIL;

/*!
 * Structure representing WORKLOAD_SINGLE_1X-specifc PWR_POLICY control/policy
 * parameters.
 */
typedef struct LW2080_CTRL_PMGR_PWR_POLICY_CONTROL_DATA_WORKLOAD_SINGLE_1X {
    LwU32                                                      rsvd;
    /*!
     * PERF_CF_PWR_MODEL interface.
     */
    LW2080_CTRL_PMGR_PWR_POLICY_CONTROL_DATA_PERF_CF_PWR_MODEL pwrModel;
} LW2080_CTRL_PMGR_PWR_POLICY_CONTROL_DATA_WORKLOAD_SINGLE_1X;
typedef struct LW2080_CTRL_PMGR_PWR_POLICY_CONTROL_DATA_WORKLOAD_SINGLE_1X *PLW2080_CTRL_PMGR_PWR_POLICY_CONTROL_DATA_WORKLOAD_SINGLE_1X;

/*!
 * Structure representing WORKLOAD_COMBINED_1X-specifc PWR_POLICY control/policy
 * parameters.
 */
typedef struct LW2080_CTRL_PMGR_PWR_POLICY_CONTROL_DATA_WORKLOAD_COMBINED_1X {
    /*!
     * PERF_CF_PWR_MODEL interface.
     */
    LW2080_CTRL_PMGR_PWR_POLICY_CONTROL_DATA_PERF_CF_PWR_MODEL pwrModel;
    /*!
     * Ratio by which to scale master clock changes up.
     */
    LwUFXP4_12                                                 clkMasterUpScale;
    /*!
     * Ratio by which to scale master clock changes down.
     */
    LwUFXP4_12                                                 clkMasterDownScale;
} LW2080_CTRL_PMGR_PWR_POLICY_CONTROL_DATA_WORKLOAD_COMBINED_1X;
typedef struct LW2080_CTRL_PMGR_PWR_POLICY_CONTROL_DATA_WORKLOAD_COMBINED_1X *PLW2080_CTRL_PMGR_PWR_POLICY_CONTROL_DATA_WORKLOAD_COMBINED_1X;

/*!
 * TOTAL GPU interface
 */
typedef struct LW2080_CTRL_PMGR_PWR_POLICY_CONTROL_DATA_TOTAL_GPU_INTERFACE {
    /*!
     * BOARDOBJ_INTERFACE super class.  Must always be first element in the structure.
     */
    LW2080_CTRL_BOARDOBJ_INTERFACE super;
} LW2080_CTRL_PMGR_PWR_POLICY_CONTROL_DATA_TOTAL_GPU_INTERFACE;

/*!
 * Tuples and flags defining the optional piecewise linear frequency flooring
 * lwrve that this policy may enforce. Control data.
 */
typedef struct LW2080_CTRL_PMGR_PWR_POLICY_CONTROL_DATA_TOTAL_GPU {
    /*!
     * Control data concerning the pff interface.
     */
    LW2080_CTRL_PMGR_PFF_RM_CONTROL                              pff;
    /*!
     * TGP interface.
     */
    LW2080_CTRL_PMGR_PWR_POLICY_CONTROL_DATA_TOTAL_GPU_INTERFACE tgpIface;
} LW2080_CTRL_PMGR_PWR_POLICY_CONTROL_DATA_TOTAL_GPU;

/*!
 * Union of type-specific control/policy parameters.
 */


/*!
 * Structure repsenting the control/policy parameters of a PWR_POLICY entry.
 */
typedef struct LW2080_CTRL_PMGR_PWR_POLICY_CONTROL {
    /*!
     * @ref LW2080_CTRL_PMGR_PWR_POLICY_TYPE_<xyz>.
     */
    LwU8  type;

    /*!
     * Current limit value to enforce.  Must always be within range of
     * [limitMin, limitMax].
     */
    LwU32 limitLwrr;

    /*!
     * Type-specific control/policy parameters data.
     */
    union {
        LW2080_CTRL_PMGR_PWR_POLICY_CONTROL_DATA_WORKLOAD             workload;
        LW2080_CTRL_PMGR_PWR_POLICY_CONTROL_DATA_WORKLOAD_MULTIRAIL   workloadMulRail;
        LW2080_CTRL_PMGR_PWR_POLICY_CONTROL_DATA_WORKLOAD_SINGLE_1X   single1x;
        LW2080_CTRL_PMGR_PWR_POLICY_CONTROL_DATA_WORKLOAD_COMBINED_1X combined1x;
        LW2080_CTRL_PMGR_PWR_POLICY_CONTROL_DATA_TOTAL_GPU            totalGpu;
    } data;
} LW2080_CTRL_PMGR_PWR_POLICY_CONTROL;

/*!
 * Structure representing BALANCE-specifc PWR_POLICY_RELATIONSHIP control/policy
 * parameters.
 */
typedef struct LW2080_CTRL_PMGR_PWR_POLICY_RELATIONSHIP_CONTROL_DATA_BALANCE {
    /*!
     * Boolean indicating that the BALANCE relationship's PWM value should be
     * locked to a simulated value, overriding the behavior of the BALANCE
     * relationship controller.
     */
    LwBool      bPwmSim;
    /*!
     * The PWM value (specified as unsigned FXP 16.16) used to simulate.  This
     * value is only applied when @ref bPwmSim == LW_TRUE.
     */
    LwUFXP16_16 pwmPctSim;
} LW2080_CTRL_PMGR_PWR_POLICY_RELATIONSHIP_CONTROL_DATA_BALANCE;

/*!
 * Union of type-specific control/policy parameters.
 */


/*!
 * Structure repsenting the control/policy parameters of a
 * PWR_POLICY_RELATIONSHIP entry.
 */
typedef struct LW2080_CTRL_PMGR_PWR_POLICY_RELATIONSHIP_CONTROL {
    /*!
     * @ref LW2080_CTRL_PMGR_PWR_POLICY_TYPE_<xyz>.
     */
    LwU8 type;

    /*!
     * Type-specific control/policy parameters data.
     */
    union {
        LW2080_CTRL_PMGR_PWR_POLICY_RELATIONSHIP_CONTROL_DATA_BALANCE balance;
    } data;
} LW2080_CTRL_PMGR_PWR_POLICY_RELATIONSHIP_CONTROL;

/*!
 * Structure describing _PROPGAIN specific control parameters.
 */
typedef struct LW2080_CTRL_PMGR_PWR_VIOLATION_CONTROL_PROPGAIN {
    /*!
     * Signed FXP20.12 proportional gain value to use to adjust the power limit.
     */
    LwSFXP20_12 propGain;
} LW2080_CTRL_PMGR_PWR_VIOLATION_CONTROL_PROPGAIN;
typedef struct LW2080_CTRL_PMGR_PWR_VIOLATION_CONTROL_PROPGAIN *PLW2080_CTRL_PMGR_PWR_VIOLATION_CONTROL_PROPGAIN;

/*!
 * PWR_VIOLATION type-specific data union.  Discriminated by
 * PWR_VIOLATION::super.type.
 */


/*!
 * Structure representing the control parameters associated with a PWR_VIOLATION.
 */
typedef struct LW2080_CTRL_PMGR_PWR_VIOLATION_CONTROL {
    /*!
     * XAPI does not support using an element within another structure as a
     * discriminant.  Placing this redundant type value here until that design
     * constraint can be fixed.
     */
    LwU8       type;

    /*!
     * Target violation rate to maintain (normalized to [0.0,1.0]).
     */
    LwUFXP4_12 violTarget;

    /*!
     * Type-specific data union.
     */
    union {
        LW2080_CTRL_PMGR_PWR_VIOLATION_CONTROL_PROPGAIN propGain;
    } data;
} LW2080_CTRL_PMGR_PWR_VIOLATION_CONTROL;
typedef struct LW2080_CTRL_PMGR_PWR_VIOLATION_CONTROL *PLW2080_CTRL_PMGR_PWR_VIOLATION_CONTROL;

/*!
 * LW2080_CTRL_CMD_PMGR_PWR_POLICY_GET_CONTROL
 *
 * This command returns the control/policy parameters for a client-speicifed set
 * of PWR_POLICY entries in the Power Policy Table.
 *
 * https://wiki.lwpu.com/engwiki/index.php/Resman/PState/Data_Tables/Power_Tables/Power_Policy_Table_2.X
 *
 * See LW2080_CTRL_PMGR_PWR_POLICY_CONTROL_PARAMS for documentation on the
 * parameters
 *
 * Possible status values returned are
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW2080_CTRL_CMD_PMGR_PWR_POLICY_GET_CONTROL                                        (0x2080261aU) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PMGR_INTERFACE_ID << 8) | 0x1A" */

/*!
 * LW2080_CTRL_CMD_PMGR_PWR_POLICY_SET_CONTROL
 *
 * This command accepts client-specified control/policy parameters for a set of
 * PWR_POLICY entries in the Power Policy Table, and applies these new
 * parameters to the set of PWR_POLICY entries.
 *
 * https://wiki.lwpu.com/engwiki/index.php/Resman/PState/Data_Tables/Power_Tables/Power_Policy_Table_2.X
 *
 * See LW2080_CTRL_PMGR_PWR_POLICY_CONTROL_PARAMS for documentation on the
 * parameters
 *
 * Possible status values returned are
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW2080_CTRL_CMD_PMGR_PWR_POLICY_SET_CONTROL                                        (0x2080261bU) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PMGR_INTERFACE_ID << 8) | 0x1B" */

/*!
 * Sentinel value indicating that
 * @ref LW2080_CTRL_PMGR_PWR_POLICY_CONTROL_PARAMS::inflectionPointsDisableRequest
 * should be ignored in a control call.
 */
#define LW2080_CTRL_PMGR_PWR_POLICY_CONTROL_INFLECTION_POINT_DISABLE_REQUEST_IGNORE_PSTATE 0xFEU

/*!
 * Sets inflectionPointsDisableRequest in a
 * @ref LW2080_CTRL_PMGR_PWR_POLICY_CONTROL_PARAMS to an "ignore" state.
 *
 * @param[in]   _pControlParams Control parameters in which to ignore request
 */
#define LW2080_CTRL_PMGR_PWR_POLICY_CONTROL_INFLECTION_POINT_DISABLE_REQUEST_IGNORE_SET(_pControlParams) \
    do \
    { \
        (_pControlParams)->inflectionPointsDisableRequest.pstateIdxLowest = \
            LW2080_CTRL_PMGR_PWR_POLICY_CONTROL_INFLECTION_POINT_DISABLE_REQUEST_IGNORE_PSTATE; \
    } while (LW_FALSE)

/*!
 * Gets the "ignore" state of the inflectionPointsDisableRequest in a
 * @ref LW2080_CTRL_PMGR_PWR_POLICY_CONTROL_PARAMS
 *
 * @param[in]   _pControlParams Control parameters in which get ignore state
 *
 * @return  @ref LW_TRUE    Request is to be ignored
 * @return  @ref LW_FALSE   Request is not to be ignored
 */
#define LW2080_CTRL_PMGR_PWR_POLICY_CONTROL_INFLECTION_POINT_DISABLE_REQUEST_IGNORE_GET(_pControlParams) \
    ((_pControlParams)->inflectionPointsDisableRequest.pstateIdxLowest == \
        LW2080_CTRL_PMGR_PWR_POLICY_CONTROL_INFLECTION_POINT_DISABLE_REQUEST_IGNORE_PSTATE) \

/*!
 * Structure representing the control/policy parameters associated with a set of
 * PWR_POLICY entries in the GPU's PWR_POLICY power policy functionality.
 */
typedef struct LW2080_CTRL_PMGR_PWR_POLICY_CONTROL_PARAMS {
    /*!
     * Request to disable inflection points.
     *
     * @note    This disablement request corresponds to the
     *          @ref LW2080_CTRL_PMGR_PWR_POLICIES_INFLECTION_POINTS_DISABLE_REQUEST_ID
     *          of
     *          @ref LW2080_CTRL_PMGR_PWR_POLICIES_INFLECTION_POINTS_DISABLE_REQUEST_ID_API
     */
    LW2080_CTRL_PMGR_PWR_POLICIES_INFLECTION_POINTS_DISABLE_REQUEST inflectionPointsDisableRequest;

    /*!
     * [in] - Mask of PWR_POLICY entries requested by the client.
     */
    LwU32                                                           policyMask;
    /*!
     * [out] - Array of PWR_POLICY entries.  Has valid indexes corresponding to
     * the bits set in @ref policyMask.
     */
    LW2080_CTRL_PMGR_PWR_POLICY_CONTROL                             policies[LW2080_CTRL_PMGR_PWR_POLICY_MAX_POLICIES];
    /*!
     * [in] - Mask of PWR_POLICY_RELATIONSHIP entries requested by the client.
     */
    LwU32                                                           policyRelMask;
    /*!
     * [out] - Array of PWR_POLICY_RELATIONSHIP entries.  Has valid indexes
     * corresponding to the bits set in @ref policyRelMask.
     */
    LW2080_CTRL_PMGR_PWR_POLICY_RELATIONSHIP_CONTROL                policyRels[LW2080_CTRL_PMGR_PWR_POLICY_MAX_POLICY_RELATIONSHIPS];
    /*!
     * [in] - Mask of PWR_VIOLATION entries specified on this GPU.
     */
    LwU32                                                           pwrViolMask;

    /*!
     * [out] - Array of PWR_VIOLATION structures.  Has valid indexes corresponding to the
     * bits set in @ref super.objMask.
     */
    LW2080_CTRL_PMGR_PWR_VIOLATION_CONTROL                          violations[LW2080_CTRL_PMGR_PWR_VIOLATION_MAX];
} LW2080_CTRL_PMGR_PWR_POLICY_CONTROL_PARAMS;

/*!
 * LW2080_CTRL_CMD_PMGR_PWR_POLICY_IDX_INFO_GET
 *
 * This command returns PWR_POLICY static object information as specified by the
 * Power Policy Table, corresponding to PWR_POLICY index (@ref
 * LW2080_CTRL_PMGR_PWR_POLICY_IDX_<xyz>).
 *
 * https://wiki.lwpu.com/engwiki/index.php/Resman/PState/Data_Tables/Power_Tables/Power_Policy_Table_2.X
 *
 * This interface is effectivey a policy-index-based wrapper to
 * LW2080_CTRL_CMD_PMGR_PWR_POLICY_GET_INFO.
 *
 * See LW2080_CTRL_PMGR_PWR_POLICY_IDX_INFO_PARAMS for documentation on the
 * parameters.
 *
 * Possible status values returned are
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_NOT_SUPPORTED
 */
#define LW2080_CTRL_CMD_PMGR_PWR_POLICY_IDX_INFO_GET (0x2080261lw) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PMGR_INTERFACE_ID << 8) | LW2080_CTRL_PMGR_PWR_POLICY_IDX_INFO_PARAMS_MESSAGE_ID" */

/*!
 * Structure representing the static state information associated with the GPU's
 * PWR_POLICY index - an semantic index/name for a policy which is implementing
 * special board functionality per the GPU/board POR.
 */
#define LW2080_CTRL_PMGR_PWR_POLICY_IDX_INFO_PARAMS_MESSAGE_ID (0x1LW)

typedef struct LW2080_CTRL_PMGR_PWR_POLICY_IDX_INFO_PARAMS {
    /*!
     * [in] - Client's requested policy index (@ref
     * LW2080_CTRL_PMGR_PWR_POLICY_IDX_<xyz>).
     */
    LwU8                             policyIdx;
    /*!
     * [out] - PWR_POLICY entry corresponding to @ref policyIdx.
     */
    LW2080_CTRL_PMGR_PWR_POLICY_INFO policy;
} LW2080_CTRL_PMGR_PWR_POLICY_IDX_INFO_PARAMS;

/*!
 * LW2080_CTRL_CMD_PMGR_PWR_POLICY_IDX_LIMIT_ARB_INPUT_GET
 *
 * This command is a policy-index-based (@ref
 * LW2080_CTRL_PMGR_PWR_POLICY_IDX_<xyz>) interface to retrieve a client's (@ref
 * LW2080_CTRL_PMGR_PWR_POLICY_LIMIT_INPUT_CLIENT_IDX_<xyz>) limit arbitration
 * input to the RM's @ref PWR_POLICY::limitArbLwrr structure.
 *
 * This interface provides a simplified, client-friendly subset of @ref
 * LW2080_CTRL_CMD_PMGR_PWR_POLICY_GET_CONTROL by providing:
 * 1. Policy Indexes - Abstracts away implementation details for which actual
 *     policy objects/entries and classes are implementing common functionality.
 * 2. Limits Only - Only specifies limit values, abstracting away class specific
 *     details, so clients don't need to bother with them.
 * 3. Client Indexes - Support for different clients, so different client APIs
 *     (LWML/LWAPI, SMBPBI/PPO, etc.) can request simultaneous limits between
 *     which RM will arbitrate.
 *
 * See LW2080_CTRL_PMGR_PWR_POLICY_IDX_LIMIT_ARB_INPUT_PARAMS for documentation
 * on the parameters.
 *
 * Possible status values returned are
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_NOT_SUPPORTED
 */
#define LW2080_CTRL_CMD_PMGR_PWR_POLICY_IDX_LIMIT_ARB_INPUT_GET (0x2080261dU) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PMGR_INTERFACE_ID << 8) | 0x1D" */

/*!
 * LW2080_CTRL_CMD_PMGR_PWR_POLICY_IDX_LIMIT_ARB_INPUT_SET
 *
 * This command is a policy-index-based (@ref
 * LW2080_CTRL_PMGR_PWR_POLICY_IDX_<xyz>) interface to specify a client's (@ref
 * LW2080_CTRL_PMGR_PWR_POLICY_LIMIT_INPUT_CLIENT_IDX_<xyz>) limit arbitration
 * input to the RM's @ref PWR_POLICY::limitArbLwrr structure.
 *
 * This interface provides a simplified, client-friendly subset of @ref
 * LW2080_CTRL_CMD_PMGR_PWR_POLICY_SET_CONTROL by providing:
 * 1. Policy Indexes - Abstracts away implementation details for which actual
 *     policy objects/entries and classes are implementing common functionality.
 * 2. Limits Only - Only specifies limit values, abstracting away class specific
 *     details, so clients don't need to bother with them.
 * 3. Client Indexes - Support for different clients, so different client APIs
 *     (LWML/LWAPI, SMBPBI/PPO, etc.) can request simultaneous limits between
 *     which RM will arbitrate.
 *
 * See LW2080_CTRL_PMGR_PWR_POLICY_IDX_LIMIT_ARB_INPUT_PARAMS for documentation
 * on the parameters.
 *
 * Possible status values returned are
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_NOT_SUPPORTED
 */
#define LW2080_CTRL_CMD_PMGR_PWR_POLICY_IDX_LIMIT_ARB_INPUT_SET (0x2080261eU) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PMGR_INTERFACE_ID << 8) | 0x1E" */

/*!
 * Structure representing a client's limit arbitration input for a given policy
 * index's @ref PWR_POLICY::limitArbLwrr limit arbitration structure.  This
 * interface can be used to lwstomize limit arbitration.
 */
typedef struct LW2080_CTRL_PMGR_PWR_POLICY_IDX_LIMIT_ARB_INPUT_PARAMS {
    /*!
     * [in] - Client's requested policy index (@ref
     * LW2080_CTRL_PMGR_PWR_POLICY_IDX_<xyz>).
     */
    LwU8  policyIdx;
    /*!
     * [in] - Client's limit arbitration index (@ref
     * LW2080_CTRL_PMGR_PWR_POLICY_LIMIT_INPUT_CLIENT_IDX_<xyz>).
     */
    LwU8  clientIdx;
    /*!
     * [in/out] - Client's limit arbitration input value.
     */
    LwU32 limit;
    /*!
     * [in] - Flags.
     */
    LwU32 flags;
} LW2080_CTRL_PMGR_PWR_POLICY_IDX_LIMIT_ARB_INPUT_PARAMS;

/*
 * Arbitration input limit set by client is maintained only in RM.
 * To make limit persistent across boots, it can be stored on boards having
 * InfoROM. During RM init it is fetched from InfoROM and applied. To enable
 * this behaviour ensure
 *   clientIdx = PPO
 *   flag      = PERSIST_YES
 */
#define LW2080_CTRL_PMGR_PWR_POLICY_IDX_LIMIT_ARB_INPUT_PARAMS_FLAGS_PERSIST          1:0
#define LW2080_CTRL_PMGR_PWR_POLICY_IDX_LIMIT_ARB_INPUT_PARAMS_FLAGS_PERSIST_NO  0U
#define LW2080_CTRL_PMGR_PWR_POLICY_IDX_LIMIT_ARB_INPUT_PARAMS_FLAGS_PERSIST_YES 1U

/*!
 * LW2080_CTRL_CMD_PMGR_PWR_POLICY_IDX_LIMIT_ARB_OUTPUT_GET
 *
 * This command is a policy-index-based (@ref
 * LW2080_CTRL_PMGR_PWR_POLICY_IDX_<xyz>) interface to retrieve a limit
 * arbitration output of the RM's @ref PWR_POLICY::limitArbLwrr structure.
 *
 * See LW2080_CTRL_PMGR_PWR_POLICY_IDX_LIMIT_ARB_OUTPUT_PARAMS for documentation
 * on the parameters.
 *
 * Possible status values returned are
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_NOT_SUPPORTED
 */
#define LW2080_CTRL_CMD_PMGR_PWR_POLICY_IDX_LIMIT_ARB_OUTPUT_GET                 (0x2080261fU) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PMGR_INTERFACE_ID << 8) | LW2080_CTRL_PMGR_PWR_POLICY_IDX_LIMIT_ARB_OUTPUT_PARAMS_MESSAGE_ID" */

/*!
 * Structure representing a limit arbitration output for a given policy
 * index's @ref PWR_POLICY::limitArbLwrr limit arbitration structure.
 */
#define LW2080_CTRL_PMGR_PWR_POLICY_IDX_LIMIT_ARB_OUTPUT_PARAMS_MESSAGE_ID (0x1FU)

typedef struct LW2080_CTRL_PMGR_PWR_POLICY_IDX_LIMIT_ARB_OUTPUT_PARAMS {
    /*!
     * [in]  - Client's requested policy index (@ref
     * LW2080_CTRL_PMGR_PWR_POLICY_IDX_<xyz>).
     */
    LwU8  policyIdx;
    /*!
     * [out] - Client's limit arbitration output value.
     */
    LwU32 limit;
} LW2080_CTRL_PMGR_PWR_POLICY_IDX_LIMIT_ARB_OUTPUT_PARAMS;

/*
 * LW2080_CTRL_CMD_PMGR_READ_TACH
 *
 * This command can be used to setup and read the tachometer on a given pin.
 *
 * gpioPin
 *   This parameter specifies the mode of pwm we want to program the pin.
 *
 * period
 *   The timeperiod of the pwm to be programmed.
 *
 * dutyCycle
 *   The duty cycle in the unit of timeperiod.
 *
 * Possible status values returned are
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_ILWALID_STATE
 *   LW_ERR_NOT_SUPPORTED
 *
 */
#define LW2080_CTRL_CMD_PMGR_PROGRAM_PWM (0x20802603U) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PMGR_INTERFACE_ID << 8) | LW2080_CTRL_PMGR_PROGRAM_PWM_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_PMGR_PROGRAM_PWM_PARAMS_MESSAGE_ID (0x3U)

typedef struct LW2080_CTRL_PMGR_PROGRAM_PWM_PARAMS {
    LwU32 gpioPin;
    LwU32 flags;
    LwU32 period;
    LwU32 dutyCycle;
} LW2080_CTRL_PMGR_PROGRAM_PWM_PARAMS;

#define LW2080_CTRL_PMGR_PROGRAM_PWM_PARAMS_FLAGS_CONTROL                          15:0     // Control related
#define LW2080_CTRL_PMGR_PROGRAM_PWM_PARAMS_FLAGS_CONTROL_ENABLE                   0:0      // Enable flag
#define LW2080_CTRL_PMGR_PROGRAM_PWM_PARAMS_FLAGS_CONTROL_ENABLE_OFF   0U        // PWM OFF
#define LW2080_CTRL_PMGR_PROGRAM_PWM_PARAMS_FLAGS_CONTROL_ENABLE_ON    1U        // PWM ON
// Sense for PWM values
#define LW2080_CTRL_PMGR_PROGRAM_PWM_PARAMS_FLAGS_CONTROL_SENSE                    5:4      // SENSE field
#define LW2080_CTRL_PMGR_PROGRAM_PWM_PARAMS_FLAGS_CONTROL_SENSE_GPIO   0U        // Max period = GPIO's asserted state
#define LW2080_CTRL_PMGR_PROGRAM_PWM_PARAMS_FLAGS_CONTROL_SENSE_FUNC   1U        // Max period = FUNC's asserted state
#define LW2080_CTRL_PMGR_PROGRAM_PWM_PARAMS_FLAGS_CONTROL_SENSE_NATIVE 2U        // Max period = max period
#define LW2080_CTRL_PMGR_PROGRAM_PWM_PARAMS_FLAGS_CONTROL_SENSE_ILWERT 3U        // Max period = min period

/*
 * LW2080_CTRL_CMD_PMGR_READ_TACH
 *
 * This command can be used to setup and read the tachometer on a given pin.
 *
 * gpioPin
 *   This parameter specifies gpio pin # on which the tach has to be setup.
 *
 * setup
 *   This parameter specifies whether the tach needs to be setup on the pin.
 *   Typically the tach would be stup for a 1Hz base. So wait for at least a second after
 *   setting up to get an accurate reading. If this param is false; the tach reading would
 *   be returned.
 *
 * edgesPerSecond
 *   The number of rising edges latched per second.
 *
 * Possible status values returned are
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_ILWALID_STATE
 *   LW_ERR_NOT_SUPPORTED
 *
 */
#define LW2080_CTRL_CMD_PMGR_READ_TACH                                 (0x20802604U) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PMGR_INTERFACE_ID << 8) | LW2080_CTRL_PMGR_READ_TACH_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_PMGR_READ_TACH_PARAMS_MESSAGE_ID (0x4U)

typedef struct LW2080_CTRL_PMGR_READ_TACH_PARAMS {
    LwU32  gpioPin;
    LwBool bSetup;
    LwU32  edgesPerSecond;
} LW2080_CTRL_PMGR_READ_TACH_PARAMS;


#define LW2080_CTRL_PMGR_VOLT_VDT_ENTRY_INDEX_ILWALID 0xFFU

/*!
 * LW2080_CTRL_CMD_PMGR_VOLT_VDT_GET_INFO
 *
 * This command returns the Voltage Descriptor Table Header information.
 *
 * See LW2080_CTRL_PMGR_VOLT_VDT_GET_INFO_PARAMS for documentation on
 * the parameters
 *
 * Possible status values returned are
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW2080_CTRL_CMD_PMGR_VOLT_VDT_GET_INFO        (0x20802605U) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PMGR_INTERFACE_ID << 8) | LW2080_CTRL_PMGR_VOLT_VDT_GET_INFO_PARAMS_MESSAGE_ID" */

/*!
 * This struct is used to hold Voltage Descriptor Table Header information.
*/
#define LW2080_CTRL_PMGR_VOLT_VDT_GET_INFO_PARAMS_MESSAGE_ID (0x5U)

typedef struct LW2080_CTRL_PMGR_VOLT_VDT_GET_INFO_PARAMS {
    /*!
     * Cached HW SPEEDO fuse.  Used in VDT/CVB callwlations to callwlate target
     * voltage.
     */
    LwU32 hwSpeedo;
    /*!
     * Cached HW SPEEDO version fuse.  Compared against the VDT speedo version
     * value to confirm the units of the SPEEDO fuse match the expectation for
     * the VDT/CVB callwlations.
     */
    LwU32 hwSpeedoVersion;
    /*!
     * SPEEDO version to be used for this table.  Will be compared against a HW
     * fuse to confirm the units of the SPEEDO fuse match the expectation for
     * the VDT/CVB callwlations.
     */
    LwU32 speedoVersion;

    /*!
     * Polling period for re-evaluating any temperature-based requested
     * voltages.
     */
    LwU16 tempPollingPeriodms;

    /*!
     * Number of VDT entries in the table.
     */
    LwU8  numEntries;

    /*!
     * Original/nominal P0 voltage before voltage tuning.  This is a legacy
     * feature from GF100 voltage tuning to increase yields due to fabric
     * issues.
     */
    LwU8  nominalP0VdtEntry;
    /*!
     * Default Voltage Reliability Limit entry - index into table for entry
     * specifying the default maximum reliability limit of the silicon.
     * VOLTAGE_DESCRIPTOR_TABLE_ENTRY_ILWALID/0xFF = none.
     */
    LwU8  reliabilityLimitEntry;
    /*!
     * Alternate Voltage Reliability Limit entry - index into table for
     * entry specifying the alternate maximum reliability limit of the
     * silicon. VOLTAGE_DESCRIPTOR_TABLE_ENTRY_ILWALID/0xFF = none.
     */
    LwU8  altReliabilityLimitEntry;
    /*!
     * User Over-Voltage Limit entry - index into table for entry specifying
     * the maximum user over-voltage limit.
     * VOLTAGE_DESCRIPTOR_TABLE_ENTRY_ILWALID/0xFF = none.
     */
    LwU8  overVoltageLimitEntry;
    /*!
     * Voltage Tuning Entry - index into table for entry reserved for tuning
     * voltage with a delta value.
     * VOLTAGE_DESCRIPTOR_TABLE_ENTRY_ILWALID/0xFF = none.
     */
    LwU8  voltageTuningEntry;
} LW2080_CTRL_PMGR_VOLT_VDT_GET_INFO_PARAMS;

/*!
 * LW2080_CTRL_CMD_PMGR_VOLT_VDT_ENTRIES_GET_INFO
 *
 * This command returns a list of Voltage Descriptor Table Entries information.
 *
 * See LW2080_CTRL_PMGR_VOLT_VDT_ENTRY_INFO for documentation on
 * the parameters.
 *
 * Possible status values returned are
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW2080_CTRL_CMD_PMGR_VOLT_VDT_ENTRIES_GET_INFO (0x20802606U) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PMGR_INTERFACE_ID << 8) | 0x6" */

/*!
 * LW2080_CTRL_CMD_PMGR_VOLT_VDT_ENTRIES_SET_INFO
 *
 * This command mutates a set of Voltage Descriptor Table Entries.
 *
 * See LW2080_CTRL_PMGR_VOLT_VDT_ENTRY_INFO for documentation on
 * the parameters.
 *
 * Possible status values returned are
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW2080_CTRL_CMD_PMGR_VOLT_VDT_ENTRIES_SET_INFO (0x20802607U) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PMGR_INTERFACE_ID << 8) | 0x7" */

/*!
 * Possible values of the "type" member of LW2080_CTRL_PMGR_VOLT_VDT_ENTRY_INFO,
 * indicates if a VDT entry is of CVB1.0, CVB2.0, or CVB1.0 Double-Precision
 * Adjustment.
 */
#define LW2080_CTRL_PMGR_VOLT_VDT_ENTRY_TYPE_CVB10     0x00U
#define LW2080_CTRL_PMGR_VOLT_VDT_ENTRY_TYPE_CVB20     0x01U
#define LW2080_CTRL_PMGR_VOLT_VDT_ENTRY_TYPE_CVB10_DPA 0x02U
#define LW2080_CTRL_PMGR_VOLT_VDT_ENTRY_TYPE_MAX       0x03U

/*!
 * This struct contains the three coefficients that are used for CVB1.0.
 */
typedef struct LW2080_CTRL_PMGR_VOLT_VDT_ENTRY_DATA_CVB10 {
    /*!
     * Signed 32-bit coefficient0 (0.1 uV).  Stored as signed 64-bit to expedite
     * 64-bit callwlations.
     */
    LW_DECLARE_ALIGNED(LwS64 coefficient0, 8);
    /*!
     * Signed 32-bit coefficient1 (0.1 uV).  Stored as signed 64-bit to expedite
     * 64-bit callwlations.
     */
    LW_DECLARE_ALIGNED(LwS64 coefficient1, 8);
    /*!
     * Signed 32-bit coefficient2 (0.00001 uV).  Stored as signed 64-bit to
     * expedite 64-bit callwlations.
     */
    LW_DECLARE_ALIGNED(LwS64 coefficient2, 8);
} LW2080_CTRL_PMGR_VOLT_VDT_ENTRY_DATA_CVB10;

/*!
 * This struct contains the seven coefficients that are used for CVB2.0.
 */
typedef struct LW2080_CTRL_PMGR_VOLT_VDT_ENTRY_DATA_CVB20 {
    /*!
     * Second-order Coefficient 5 (V / C^2_adj, F8.24 signed)
     *
     * term5 = coefficient5 * tjSqAdj
     */
    LwS32 coefficient5;
    /*!
     * Second-order Coefficient 4 (V, F8.24 signed)
     *
     * term4 = coefficient4 * speedoSqAdj
     */
    LwS32 coefficient4;
    /*!
     * First-order Coefficient 3 (V / C, F16.16 signed)
     *
     * term3 = coefficient3 * speedoTjAdj
     */
    LwS32 coefficient3;
    /*!
     * First-order Coefficient 2 (V / C, F16.16 signed)
     *
     * term2 = coefficient2 * tj
     */
    LwS32 coefficient2;
    /*!
     * First-order Coefficient 1 (V, F8.24 signed)
     *
     * term1 = coefficient1 * speedo
     */
    LwS32 coefficient1;
    /*!
     * Constant Term Coefficient 0 (V, F8.24 signed)
     *
     * term0 = coefficient0
     */
    LwS32 coefficient0;
} LW2080_CTRL_PMGR_VOLT_VDT_ENTRY_DATA_CVB20;

/*!
 * This struct contains the three coefficients that are used for CVB1.0 Double-
 * Precision Adjustment.
 */
typedef struct LW2080_CTRL_PMGR_VOLT_VDT_ENTRY_DATA_CVB10_DPA {
    /*!
     * Signed 32-bit coefficient0 (0.1 uV).  Stored as signed 64-bit to expedite
     * 64-bit callwlations.
     */
    LW_DECLARE_ALIGNED(LwS64 coefficient0, 8);
    /*!
     * Signed 32-bit coefficient1 (0.1 uV).  Stored as signed 64-bit to expedite
     * 64-bit callwlations.
     */
    LW_DECLARE_ALIGNED(LwS64 coefficient1, 8);
    /*!
     * Signed 32-bit coefficient2 (0.00001 uV).  Stored as signed 64-bit to
     * expedite 64-bit callwlations.
     */
    LW_DECLARE_ALIGNED(LwS64 coefficient2, 8);
} LW2080_CTRL_PMGR_VOLT_VDT_ENTRY_DATA_CVB10_DPA;

/*!
 * This structure contains two indices used for CVB class MAX
 */
typedef struct LW2080_CTRL_PMGR_VOLT_VDT_ENTRY_DATA_MAX {
    /*
     * First VDT entry index evaluated by VDT MAX class
     */
    LwU8 index0;

    /*
     * Second VDT entry index evaluated by VDT MAX class
     */
    LwU8 index1;
} LW2080_CTRL_PMGR_VOLT_VDT_ENTRY_DATA_MAX;

/*!
 * This is a union of CVB1.0 struct and CVB2.0 struct, and will
 * take on corresponding type depending on the VDT entry type.
*/


/*!
 * This struct is used to hold parameters in a single VDT entry.
*/
typedef struct LW2080_CTRL_PMGR_VOLT_VDT_ENTRY_INFO {
    /*!
    * VDT table entry index
    */
    LwU8  index;
    /*!
    * The voltage callwlated based this VDT entry and all the ones
    * linked to it.
    * This value is bounded to what the voltage regulator can support,
    * and will be bound to 0 if it's negative.
    */
    LwU32 lwrrTargetVoltageuV;
    /*!
    * The voltage callwlated based only on the current VDT entry.
    * This value is unbounded, so it could be a negative value.
    */
    LwS32 localUnboundVoltageuV;
    /*!
    * LW2080_CTRL_PMGR_VOLT_VDT_ENTRY_TYPE_<xyz>
    */
    LwU8  type;
    /*!
     * Pointer to the next entry in the linked list to evaluate.
     */
    LwU8  nextEntry;
    /*!
     * Minimum voltage (uV) to bound the output of this entry
     */
    LwS32 voltageMinuV;
    /*!
     * Maximum voltage (uV) to bound the output of this entry
     */
    LwS32 voltageMaxuV;
    /*!
     * Union of the CVB10, CVB20, and CVB10 DPA coefficients
     */
    union {
        LW_DECLARE_ALIGNED(LW2080_CTRL_PMGR_VOLT_VDT_ENTRY_DATA_CVB10 cvb10, 8);

        LW2080_CTRL_PMGR_VOLT_VDT_ENTRY_DATA_CVB20 cvb20;

        LW_DECLARE_ALIGNED(LW2080_CTRL_PMGR_VOLT_VDT_ENTRY_DATA_CVB10_DPA cvb10Dpa, 8);

        LW2080_CTRL_PMGR_VOLT_VDT_ENTRY_DATA_MAX   vdtMax;
    } data;
} LW2080_CTRL_PMGR_VOLT_VDT_ENTRY_INFO;

/*!
 * This struct represents the set of VDT entries the client wants
 * have to access to.
*/
typedef struct LW2080_CTRL_PMGR_VOLT_VDT_ENTRIES_INFO_PARAMS {
    /*!
    * Number of VDT Entries to read out
    */
    LwU32 vdtEntryInfoListSize;
    /*!
    * Pointer to an array of LW2080_CTRL_PMGR_VOLT_VDT_ENTRY_INFO
    * structs with size >= vdtEntryInfoListSize * sizeof(LW2080_CTRL_PMGR_VOLT_VDT_ENTRY_INFO).
    */
    LW_DECLARE_ALIGNED(LwP64 vdtEntryInfoList, 8);
} LW2080_CTRL_PMGR_VOLT_VDT_ENTRIES_INFO_PARAMS;

/*!
 * Enumeration of voltage request clients who can request a minimum voltage from
 * the arbiter.
 *
 * Pstate client - the legacy client from pstates 1.0 as well as the pstates
 *     2.0 perf table component (primarily MCLK).
 *
 * GPC2CLK client - the minimum voltage required to support the set GPC2CLK.
 *
 * PEX client - the minimum voltage required to support the current PEX
 *     settings.
 *
 * DISPCLK client - the minimum voltage required to support the set DISPCLK.
 *
 * PIXELCLK client - the minimum voltage required to support the maximum set
 *     PIXELCLK (out of all heads).
 *
 * DP HBR2 client - the minimum voltage required to support DP HBR2 display.
 *     When the display is no longer active, the limit will be cleared.
 *
 * Number of recognized clients, must always be last entry!
 *
 * @note Clients must be grouped together such that all PERF-related entries
 * come first and all non-PERF entries follow.  The @ref
 * VOLTAGE_REQUEST_CLIENT_MASK_PERF and @ref VOLTAGE_REQUEST_CLIENT_MASK_NONPERF
 * macros below depend on this assumption!
 *
 * @note ~~~PS20TODO~~~ - This can be colwerted to a enum type - that way all the
 * bookkeeping can be handled automatically!
 *
 * 'Existing' client can be used only within VOLTAGE_REQUEST_INIT_<xyz> macros
 * to specify that we want to preserve an existing value of client field within
 * VOLTAGE_REQUEST structure.
 */
#define LW2080_CTRL_PMGR_VOLTAGE_REQUEST_CLIENT_PSTATE           0x00U
#define LW2080_CTRL_PMGR_VOLTAGE_REQUEST_CLIENT_GPC2CLK          0x01U
#define LW2080_CTRL_PMGR_VOLTAGE_REQUEST_CLIENT_PEX              0x02U
#define LW2080_CTRL_PMGR_VOLTAGE_REQUEST_CLIENT_DISPCLK          0x03U
#define LW2080_CTRL_PMGR_VOLTAGE_REQUEST_CLIENT_PIXELCLK         0x04U
#define LW2080_CTRL_PMGR_VOLTAGE_REQUEST_CLIENT_DP_HBR2_WAR_SOR0 0x05U
#define LW2080_CTRL_PMGR_VOLTAGE_REQUEST_CLIENT_DP_HBR2_WAR_SOR1 0x06U
#define LW2080_CTRL_PMGR_VOLTAGE_REQUEST_CLIENT_DP_HBR2_WAR_SOR2 0x07U
#define LW2080_CTRL_PMGR_VOLTAGE_REQUEST_CLIENT_DP_HBR2_WAR_SOR3 0x08U
#define LW2080_CTRL_PMGR_VOLTAGE_REQUEST_CLIENT_DP_HBR2_WAR_SOR(i)          (LwU8)(LW2080_CTRL_PMGR_VOLTAGE_REQUEST_CLIENT_DP_HBR2_WAR_SOR0 + i)
#define LW2080_CTRL_PMGR_VOLTAGE_REQUEST_CLIENT_DEBUG            0x09U
#define LW2080_CTRL_PMGR_VOLTAGE_REQUEST_CLIENT_NUM_CLIENTS      (0xaU) /* finn: Evaluated from "(LW2080_CTRL_PMGR_VOLTAGE_REQUEST_CLIENT_DEBUG + 1)" */
#define LW2080_CTRL_PMGR_VOLTAGE_REQUEST_CLIENT_EXISTING         0xFFU

/*!
 *  LW2080_CTRL_CMD_PMGR_VOLTAGE_REQUEST_ARBITER_CONTROL_GET
 *
 *  This command returns the control parameters which alter the behavior of
 *  VOLTAGE_REQUEST_ARBITER.
 *
 *  Possible status values returned are
 *      LW_OK
 */
#define LW2080_CTRL_CMD_PMGR_VOLTAGE_REQUEST_ARBITER_CONTROL_GET (0x20802620U) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PMGR_INTERFACE_ID << 8) | 0x20" */

/*!
 *  LW2080_CTRL_CMD_PMGR_VOLTAGE_REQUEST_ARBITER_CONTROL_SET
 *
 *  This command sets the control parameters which will alter the
 *  behavior of VOLTAGE_REQUEST_ARBITER.
 *
 *  Possible status values returned are
 *      LW_OK
 */
#define LW2080_CTRL_CMD_PMGR_VOLTAGE_REQUEST_ARBITER_CONTROL_SET (0x20802621U) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PMGR_INTERFACE_ID << 8) | 0x21" */

/*!
 * Enumeration of the supported client request types, which specify how/units
 * with which the client is actually requesting the minimum target voltage.
 *
 * DISABLE - Specified by the client to disable their current request.
 *
 * Logical - Client is specifying a voltage in uV.
 *
 * VDT - Client is specifying an index into the VDT (previously CVB)
 *     table.
 *
 * Number of recognized types, must always be last entry!
 */
#define LW2080_CTRL_CMD_PMGR_VOLTAGE_REQUEST_TYPE_DISABLE        0x00U
#define LW2080_CTRL_CMD_PMGR_VOLTAGE_REQUEST_TYPE_LOGICAL        0x01U
#define LW2080_CTRL_CMD_PMGR_VOLTAGE_REQUEST_TYPE_VDT            0x02U
#define LW2080_CTRL_CMD_PMGR_VOLTAGE_REQUEST_TYPE_NUM_TYPES      (0x3U) /* finn: Evaluated from "(LW2080_CTRL_CMD_PMGR_VOLTAGE_REQUEST_TYPE_VDT + 1)" */

/*!
 * Logical voltage client request data structure.
 */
typedef struct LW2080_CTRL_CMD_PMGR_VOLTAGE_REQUEST_DATA_LOGICAL {
    /*!
     * Type as specified by the
     * LW2080_CTRL_CMD_PMGR_VOLTAGE_REQUEST_TYPE_<xyz> macros.
     */
    LwU8  type;
    /*!
     * Logical voltage in uV.
     */
    LwU32 voltageuV;
} LW2080_CTRL_CMD_PMGR_VOLTAGE_REQUEST_DATA_LOGICAL;

/*!
 * VDT entry client request data structure.
 */
typedef struct LW2080_CTRL_CMD_PMGR_VOLTAGE_REQUEST_DATA_VDT {
    /*!
     * Type as specified by the
     * LW2080_CTRL_CMD_PMGR_VOLTAGE_REQUEST_TYPE_<xyz> macros.
     */
    LwU8 type;
    /*!
     * VDT entry index.
     */
    LwU8 entryIndex;
} LW2080_CTRL_CMD_PMGR_VOLTAGE_REQUEST_DATA_VDT;

/*!
 * Union of all client request data types.
 */


/*!
 * Structure representing a client minimum voltage request.
 */
typedef struct LW2080_CTRL_CMD_PMGR_VOLTAGE_REQUEST {
    /*!
     * Type as specified by the
     * LW2080_CTRL_CMD_PMGR_VOLTAGE_REQUEST_TYPE_<xyz> macros.
     */
    LwU8 type;

    /*!
     * Union of LOGICAL and VDT for voltage request.
     */
    union {
        LW2080_CTRL_CMD_PMGR_VOLTAGE_REQUEST_DATA_LOGICAL logical;

        LW2080_CTRL_CMD_PMGR_VOLTAGE_REQUEST_DATA_VDT     vdt;
    } data;
} LW2080_CTRL_CMD_PMGR_VOLTAGE_REQUEST;

/*!
 * LW2080_CTRL_CMD_PMGR_VOLTAGE_REQUEST_ARBITER_CONTROL_PARAMS
 *
 * This structure holds the values of control parameters related to VOLTAGE_REQUEST_ARBITER
 */
typedef struct LW2080_CTRL_CMD_PMGR_VOLTAGE_REQUEST_ARBITER_CONTROL_PARAMS {
    /*!
     * [in/out] Global voltage offset to be applied to final arbitrated target voltage.
     * This offset will be applied to the output of all clients' requests to the arbiter.
     * This alters the value of the final arbitrated voltage.
     */
    LwS32                                globalVoltageOffsetuV;

    /*!
     * [in/out] - Array of voltage offsets per client, to be applied per client
     * target voltage in uV.
     * Each client gets its offset added to the output of all of its requests to
     * the arbiter. This alters the value of the target voltage of only that
     * client for which it is specified.
     */
    LwS32                                clientVoltageOffsetsuV[LW2080_CTRL_PMGR_VOLTAGE_REQUEST_CLIENT_NUM_CLIENTS];

    /*!
     * [in/out] - Specifies the client requested floor value for voltage.
     */
    LW2080_CTRL_CMD_PMGR_VOLTAGE_REQUEST voltReq;
} LW2080_CTRL_CMD_PMGR_VOLTAGE_REQUEST_ARBITER_CONTROL_PARAMS;

/*!
 * LW2080_CTRL_CMD_PMGR_VOLT_DOMAINS_GET_INFO
 *
 * This command returns a list of voltage domain information.
 *
 * See LW2080_CTRL_PMGR_VOLT_DOMAINS_INFO for documentation on the parameters.
 *
 * Possible status values returned are
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW2080_CTRL_CMD_PMGR_VOLT_DOMAINS_GET_INFO (0x20802608U) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PMGR_INTERFACE_ID << 8) | LW2080_CTRL_PMGR_VOLT_DOMAINS_INFO_PARAMS_MESSAGE_ID" */

/*!
 * This struct is used to hold parameters of a single voltage domain.
 */
typedef struct LW2080_CTRL_PMGR_VOLT_DOMAINS_INFO {
    /*!
     * Specifies the voltage domain as defined in LW2080_CTRL_PERF_VOLTAGE_DOMAINS.
     */
    LwU32 domain;
    /*!
     * Voltage step size for the voltage domain in uV.
     */
    LwU32 stepSizeuV;
} LW2080_CTRL_PMGR_VOLT_DOMAINS_INFO;

/*!
 * This struct represents the set of voltage domain entries the client wants
 * to have the access to.
 */
#define LW2080_CTRL_PMGR_VOLT_DOMAINS_INFO_PARAMS_MESSAGE_ID (0x8U)

typedef struct LW2080_CTRL_PMGR_VOLT_DOMAINS_INFO_PARAMS {
    /*!
     * Number of voltage domain entries to read out.
     */
    LwU32 voltDomainsInfoListSize;
    /*!
     * Pointer to an array of LW2080_CTRL_PMGR_VOLT_DOMAINS_INFO
     * structs with size >= voltDomainsInfoListSize * sizeof(LW2080_CTRL_PMGR_VOLT_DOMAINS_INFO).
     */
    LW_DECLARE_ALIGNED(LwP64 voltDomainsInfoList, 8);
} LW2080_CTRL_PMGR_VOLT_DOMAINS_INFO_PARAMS;


/*!
 * This struct represents a power monitor status sample.
 */
typedef struct LW2080_CTRL_PMGR_GPUMON_PWR_MONITOR_STATUS_PARAMS {
    /*!
     * [out] - Total GPU power corresponding to the last iteration of sampling.
     * This is the summation of the values corresponding to the Power Channels
     * indexes provided in @ref
     * LW2080_CTRL_PMGR_PWR_MONITOR_GET_INFO_PARAMS::totalGpuPowerChannelMask.
     */
    LwU32 totalGpuPowermW;
} LW2080_CTRL_PMGR_GPUMON_PWR_MONITOR_STATUS_PARAMS;

/*!
 * Represents the GPU monitoring sample of power policy status value.
 */
typedef struct LW2080_CTRL_PMGR_GPUMON_PWR_POLICY_STATUS_PARAMS {
    /*!
     * [out] - Current value retrieved from the monitored PWR_CHANNEL.
     */
    LwU32 valueLwrr;
} LW2080_CTRL_PMGR_GPUMON_PWR_POLICY_STATUS_PARAMS;

/*!
 *
 */
#define LW2080_CTRL_PMGR_GPUMON_PWR_SAMPLE_TYPE_MONITOR_STATUS 0x00U
#define LW2080_CTRL_PMGR_GPUMON_PWR_SAMPLE_TYPE_POLICY_STATUS  0x01U



/*!
 * This struct represents the GPU monitoring sample of power policy status or
 * power monitor status value.
 */
typedef struct LW2080_CTRL_PMGR_GPUMON_PWR_SAMPLE {
    /*!
     * Base GPU monitoring sample.
     */
    LW_DECLARE_ALIGNED(LW2080_CTRL_GPUMON_SAMPLE base, 8);
    /*!
    * LW2080_CTRL_PMGR_GPUMON_PWR_SAMPLE_TYPE_<xyz>
    */
    LwU8 type;
    /*!
     * Union of policy status and monitor status.
     */
    union {
        LW2080_CTRL_PMGR_GPUMON_PWR_MONITOR_STATUS_PARAMS monStatus;

        LW2080_CTRL_PMGR_GPUMON_PWR_POLICY_STATUS_PARAMS  polStatus;
    } data;
} LW2080_CTRL_PMGR_GPUMON_PWR_SAMPLE;

/*!
 * Number of GPU monitoring sample in their respective buffers.
 */
#define LW2080_CTRL_PMGR_GPUMON_SAMPLE_COUNT_PWR       120U

#define LW2080_CTRL_PMGR_GPUMON_PWR_SAMPLE_BUFFER_SIZE         \
    LW_SIZEOF32(LW2080_CTRL_PMGR_GPUMON_PWR_SAMPLE) *          \
    LW2080_CTRL_PMGR_GPUMON_SAMPLE_COUNT_PWR

/*!
 * LW2080_CTRL_CMD_PMGR_GPUMON_PWR_GET_SAMPLES_V2
 *
 * This command returns PMGR gpu monitoring power sample.
 *
 * Possible status values returned are
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW2080_CTRL_CMD_PMGR_GPUMON_PWR_GET_SAMPLES_V2 (0x2080260bU) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PMGR_INTERFACE_ID << 8) | LW2080_CTRL_PMGR_GPUMON_PWR_GET_SAMPLES_V2_PARAMS_MESSAGE_ID" */

/*!
 * This struct represents the GPU monitoring samples of power values that
 * client wants the access to.
 */
#define LW2080_CTRL_PMGR_GPUMON_PWR_GET_SAMPLES_V2_PARAMS_MESSAGE_ID (0xBU)

typedef struct LW2080_CTRL_PMGR_GPUMON_PWR_GET_SAMPLES_V2_PARAMS {
    /*!
    * Type of the sample, see LW2080_CTRL_GPUMON_SAMPLE_TYPE_* for reference.
    */
    LwU8  type;
    /*!
    * tracks the offset of the tail in the cirlwlar queue array pSamples.
    */
    LwU32 tracker;
    /*!
    * A cirlwlar queue.
    *
    * @note This cirlwlar queue wraps around after 10 seconds of sampling,
    * and it is clients' responsibility to query within this time frame in
    * order to avoid losing samples.
    * @note With one exception, this queue contains last 10 seconds of samples
    * with tracker poiniting to oldest entry and entry before tracker as the
    * newest entry. Exception is when queue is not full (i.e. tracker is
    * pointing to a zeroed out entry), in that case valid entries are between 0
    * and tracker.
    * @note Clients can store tracker from previous query in order to provide
    * samples since last read.
    */
    LW_DECLARE_ALIGNED(LW2080_CTRL_PMGR_GPUMON_PWR_SAMPLE samples[LW2080_CTRL_PMGR_GPUMON_SAMPLE_COUNT_PWR], 8);
} LW2080_CTRL_PMGR_GPUMON_PWR_GET_SAMPLES_V2_PARAMS;


/*!
 * Enumeration of CLIENT_PWR_POLICY class types.
 */
#define LW2080_CTRL_PMGR_CLIENT_PWR_POLICY_TYPE_1X      0x00U
#define LW2080_CTRL_PMGR_CLIENT_PWR_POLICY_TYPE_UNKNOWN 0xFFU

/*!
 * Enumeration of CLIENT_PWR_POLICY_IDs.
 *
 * _KERNEL: This is the client power policy which is exposed to the kernel
 * driver.
 * _TGP: This is the client power policy which is exposed through LwAPI to all
 * end users.
 */
#define LW2080_CTRL_PMGR_CLIENT_PWR_POLICY_ID_KERNEL    0x00U
#define LW2080_CTRL_PMGR_CLIENT_PWR_POLICY_ID_TGP       0x01U

/*!
 * Structure describing CLIENT_PWR_POLICY_INFO static information/POR.  Implements the
 * BOARDOBJ model/interface.
 */
typedef struct LW2080_CTRL_PMGR_CLIENT_PWR_POLICY_INFO {
    /*!
     * LW2080_CTRL_BOARDOBJ super class.  Must always be first object in
     * structure.
     */
    LW2080_CTRL_BOARDOBJ super;
    /*!
     * XAPI does not support using an element within another structure as a
     * discriminant.  Placing this redundant type value here until that design
     * constraint can be fixed.
     */
    LwU8                 type;
    /*!
     * @ref LW2080_CTRL_PMGR_CLIENT_PWR_POLICY_ID_<xyz>
     */
    LwU8                 policyId;
    /*!
     * Units of limit values. @ref LW2080_CTRL_PMGR_PWR_POLICY_LIMIT_UNIT_<xyz>.
     */
    LwU8                 limitUnit;
    /*!
     * Minimum allowed limit value.
     */
    LwU32                limitMin;
    /*!
     * Rated/default limit value.
     */
    LwU32                limitRated;
    /*!
     * Maximum allowed limit value.
     */
    LwU32                limitMax;
    /*!
     * Rated battery allowed limit value.
     * @deprecated - remove after lwapi switches to ::limitBattRated
     */
    LwU32                limitBatt;
    /*!
     * Rated battery allowed limit value.
     */
    LwU32                limitBattRated;
    /*!
     * Max battery allowed limit value.
     */
    LwU32                limitBattMax;
} LW2080_CTRL_PMGR_CLIENT_PWR_POLICY_INFO;
typedef struct LW2080_CTRL_PMGR_CLIENT_PWR_POLICY_INFO *PLW2080_CTRL_PMGR_CLIENT_PWR_POLICY_INFO;

/*!
 * Structure describing CLIENT_PWR_POLICIES_INFO static information/POR. Implements
 * the BOARDOBJGRP model/interface.
 */
#define LW2080_CTRL_PMGR_CLIENT_PWR_POLICIES_INFO_MESSAGE_ID (0x30U)

typedef struct LW2080_CTRL_PMGR_CLIENT_PWR_POLICIES_INFO {
    /*!
     * LW2080_CTRL_BOARDOBJGRP_E32 super class. Must always be first object in
     * structure.
     */
    LW2080_CTRL_BOARDOBJGRP_E32             super;
    /*!
     * Array of CLIENT_PWR_POLICY_INFO structures. Has valid indexes corresponding
     * to the bits set in @ref super.objMask.
     */
    LW2080_CTRL_PMGR_CLIENT_PWR_POLICY_INFO clientPwrPolicies[LW2080_CTRL_BOARDOBJGRP_E32_MAX_OBJECTS];
} LW2080_CTRL_PMGR_CLIENT_PWR_POLICIES_INFO;
typedef struct LW2080_CTRL_PMGR_CLIENT_PWR_POLICIES_INFO *PLW2080_CTRL_PMGR_CLIENT_PWR_POLICIES_INFO;

/*!
 * LW2080_CTRL_CMD_PMGR_CLIENT_PWR_POLICIES_GET_INFO
 *
 * This command returns the PWR_POLICY subset of static data that can be exposed to a client
 *
 * See LW2080_CTRL_PMGR_CLIENT_PWR_POLICIES_INFO for documentation on the
 * parameters.
 *
 * Possible status values returned are
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW2080_CTRL_CMD_PMGR_CLIENT_PWR_POLICIES_GET_INFO (0x20802630U) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PMGR_INTERFACE_ID << 8) | LW2080_CTRL_PMGR_CLIENT_PWR_POLICIES_INFO_MESSAGE_ID" */

/*!
 * Structure describing CLIENT_PWR_POLICY_STATUS non static information.  Implements the
 * BOARDOBJ model/interface.
 */
typedef struct LW2080_CTRL_PMGR_CLIENT_PWR_POLICY_STATUS {
    /*!
     * LW2080_CTRL_BOARDOBJ super class. Must always be first object in
     * structure.
     */
    LW2080_CTRL_BOARDOBJ super;
    /*!
     * XAPI does not support using an element within another structure as a
     * discriminant.  Placing this redundant type value here until that design
     * constraint can be fixed.
     */
    LwU8                 type;
    /*!
     * Current reading value from this semantic power policy
     */
    LwU32                valueLwrr;
    /*!
     * Current resultant limit effective on this semantic power policy.
     */
    LwU32                limitLwrr;
} LW2080_CTRL_PMGR_CLIENT_PWR_POLICY_STATUS;
typedef struct LW2080_CTRL_PMGR_CLIENT_PWR_POLICY_STATUS *PLW2080_CTRL_PMGR_CLIENT_PWR_POLICY_STATUS;

/*!
 * Structure describing CLIENT_PWR_POLICIES_STATUS non static information. Implements
 * the BOARDOBJGRP model/interface.
 */
#define LW2080_CTRL_PMGR_CLIENT_PWR_POLICIES_STATUS_MESSAGE_ID (0x31U)

typedef struct LW2080_CTRL_PMGR_CLIENT_PWR_POLICIES_STATUS {
    /*!
     * LW2080_CTRL_BOARDOBJGRP_E32 super class. Must always be first object in
     * structure.
     */
    LW2080_CTRL_BOARDOBJGRP_E32               super;

    /*!
     * Array of CLIENT_PWR_POLICY_STATUS structures. Has valid indexes corresponding
     * to the bits set in @ref super.objMask.
     */
    LW2080_CTRL_PMGR_CLIENT_PWR_POLICY_STATUS clientPwrPolicies[LW2080_CTRL_BOARDOBJGRP_E32_MAX_OBJECTS];
} LW2080_CTRL_PMGR_CLIENT_PWR_POLICIES_STATUS;
typedef struct LW2080_CTRL_PMGR_CLIENT_PWR_POLICIES_STATUS *PLW2080_CTRL_PMGR_CLIENT_PWR_POLICIES_STATUS;

/*!
 * LW2080_CTRL_CMD_PMGR_CLIENT_PWR_POLICIES_GET_STATUS
 *
 * This command returns the PWR_POLICY subset of static data that can be exposed to a client
 *
 * See LW2080_CTRL_PMGR_CLIENT_PWR_POLICIES_STATUS for documentation on the
 * parameters.
 *
 * Possible status values returned are
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW2080_CTRL_CMD_PMGR_CLIENT_PWR_POLICIES_GET_STATUS (0x20802631U) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PMGR_INTERFACE_ID << 8) | LW2080_CTRL_PMGR_CLIENT_PWR_POLICIES_STATUS_MESSAGE_ID" */

/*!
 * Structure describing CLIENT_PWR_POLICY_CONTROL control information.  Implements the
 * BOARDOBJ model/interface.
 */
typedef struct LW2080_CTRL_PMGR_CLIENT_PWR_POLICY_CONTROL {
    /*!
     * LW2080_CTRL_BOARDOBJ super class.  Must always be first object in
     * structure.
     */
    LW2080_CTRL_BOARDOBJ super;
    /*!
     * XAPI does not support using an element within another structure as a
     * discriminant.  Placing this redundant type value here until that design
     * constraint can be fixed.
     */
    LwU8                 type;
    /*!
     * Current client limit, not necessarily the policy resultant limit
     * (applies to both get and set control).
     */
    LwU32                limitClientLwrr;
    /*!
     * Limit input client index.
     * @ref LW2080_CTRL_PMGR_PWR_POLICY_LIMIT_INPUT_CLIENT_IDX_<xyz>
     */
    LwU8                 limitClientIdx;
} LW2080_CTRL_PMGR_CLIENT_PWR_POLICY_CONTROL;
typedef struct LW2080_CTRL_PMGR_CLIENT_PWR_POLICY_CONTROL *PLW2080_CTRL_PMGR_CLIENT_PWR_POLICY_CONTROL;

/*!
 * Structure describing CLIENT_PWR_POLICIES_CONTROL control information. Implements
 * the BOARDOBJGRP model/interface.
 */
typedef struct LW2080_CTRL_PMGR_CLIENT_PWR_POLICIES_CONTROL {
    /*!
     * LW2080_CTRL_BOARDOBJGRP_E32 super class. Must always be first object in
     * structure.
     */
    LW2080_CTRL_BOARDOBJGRP_E32                super;
    /*!
     * Array of CLIENT_PWR_POLICY_CONTROL structures. Has valid indexes corresponding
     * to the bits set in @ref super.objMask.
     */
    LW2080_CTRL_PMGR_CLIENT_PWR_POLICY_CONTROL clientPwrPolicies[LW2080_CTRL_BOARDOBJGRP_E32_MAX_OBJECTS];
} LW2080_CTRL_PMGR_CLIENT_PWR_POLICIES_CONTROL;
typedef struct LW2080_CTRL_PMGR_CLIENT_PWR_POLICIES_CONTROL *PLW2080_CTRL_PMGR_CLIENT_PWR_POLICIES_CONTROL;

/*!
 * LW2080_CTRL_CMD_PMGR_CLIENT_PWR_POLICIES_GET_CONTROL
 *
 * This command returns the PWR_POLICY subset of static data that can be exposed to a client
 *
 * See LW2080_CTRL_PMGR_CLIENT_PWR_POLICIES_CONTROL for documentation on the
 * parameters.
 *
 * Possible status values returned are
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW2080_CTRL_CMD_PMGR_CLIENT_PWR_POLICIES_GET_CONTROL                                                         (0x20802632U) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PMGR_INTERFACE_ID << 8) | 0x32" */

/*!
 * LW2080_CTRL_CMD_PMGR_CLIENT_PWR_POLICIES_SET_CONTROL
 *
 * This command returns the PWR_POLICY subset of static data that can be exposed to a client
 *
 * See LW2080_CTRL_PMGR_CLIENT_PWR_POLICIES_CONTROL for documentation on the
 * parameters.
 *
 * Possible status values returned are
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW2080_CTRL_CMD_PMGR_CLIENT_PWR_POLICIES_SET_CONTROL                                                         (0x20802633U) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PMGR_INTERFACE_ID << 8) | 0x33" */

/*!
 * LW2080_CTRL_CMD_PMGR_CLIENT_PWR_POLICIES_INPUT_CLIENT_LWPCF_ARB_AND_SET_CONTROL
 *
 * This command is similar to LW2080_CTRL_CMD_PMGR_CLIENT_PWR_POLICIES_SET_CONTROL but a special
 * use case for input client LWPCF. It first arbitrates across various input client Ids
 * @LW2080_CTRL_PMGR_PWR_POLICY_LIMIT_INPUT_CLIENT_IDX_<XYZ> on Kernel client policy exposed externally
 * to LWPCF and set the resultant limits, before applying the resultant limits
 *
 * See LW2080_CTRL_PMGR_CLIENT_PWR_POLICIES_CONTROL for documentation on the
 * parameters.
 *
 * Possible status values returned are
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW2080_CTRL_CMD_PMGR_CLIENT_PWR_POLICIES_INPUT_CLIENT_LWPCF_ARB_AND_SET_CONTROL                              (0x20802634U) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PMGR_INTERFACE_ID << 8) | 0x34" */

/*
 * PMGR Tests.
 */

/*
 ****** Important Notice ******
 * Please ensure that the test name identifiers below, match exactly with the
 * test name strings in rmt_pmgr.h file. These identifiers are used in
 * lw2080CtrlCmdPmgrGenericTest() function, in file pmgrctrl.c
 */

#define LW2080_CTRL_PMGR_GENERIC_TEST_ID_ADC_INIT                                                                    0x00000000U
#define LW2080_CTRL_PMGR_GENERIC_TEST_ID_ADC_CHECK                                                                   0x00000001U
#define LW2080_CTRL_PMGR_GENERIC_TEST_ID_HI_OFFSET_MAX_CHECK                                                         0x00000002U
#define LW2080_CTRL_PMGR_GENERIC_TEST_ID_IPC_PARAMS_CHECK                                                            0x00000003U
#define LW2080_CTRL_PMGR_GENERIC_TEST_ID_BEACON_CHECK                                                                0x00000004U
#define LW2080_CTRL_PMGR_GENERIC_TEST_ID_OFFSET_CHECK                                                                0x00000005U

/*!
 * LW2080_CTRL_PMGR_GENERIC_TEST_ID_ADC_INIT
 *
 * Possible reasons for LW2080_CTRL_PMGR_GENERIC_TEST_ID_ADC_INIT to fail.
 *
 * _SUCCESS                     : Test *_ID_ADC_INIT is a success
 * _NON_ZERO_IIR_VALUE          : Test *_ID_ADC_INIT failed because on non-zero IIR value post init and reset
 * _NON_ZERO_ACC_LB_VALUE       : Test *_ID_ADC_INIT failed because on non-zero acc LB value post init and reset
 * _NON_ZERO_ACC_UB_VALUE       : Test *_ID_ADC_INIT failed because on non-zero acc UB value post init and reset
 * _NON_ZERO_ACC_SCNT_VALUE     : Test *_ID_ADC_INIT failed because on non-zero acc SMPCNT value post init and reset
 * _NON_ZERO_MUL_VALUE          : Test *_ID_ADC_INIT failed because on non-zero power value post init and reset
 * _NON_ZERO_MUL_ACC_LB_VALUE   : Test *_ID_ADC_INIT failed because on non-zero acc LB power value post init and reset
 * _NON_ZERO_MUL_ACC_UB_VALUE   : Test *_ID_ADC_INIT failed because on non-zero acc UB power value post init and reset
 * _NON_ZERO_MUL_ACC_SCNT_VALUE : Test *_ID_ADC_INIT failed because on non-zero acc SMPCNT power value post init and reset
 */
#define LW2080_CTRL_PMGR_GENERIC_TEST_ID_ADC_INIT_STATUS_SUCCESS                                                     0x00000000U
#define LW2080_CTRL_PMGR_GENERIC_TEST_ID_ADC_INIT_STATUS_NON_ZERO_IIR_VALUE                                          0x00000001U
#define LW2080_CTRL_PMGR_GENERIC_TEST_ID_ADC_INIT_STATUS_NON_ZERO_ACC_LB_VALUE                                       0x00000002U
#define LW2080_CTRL_PMGR_GENERIC_TEST_ID_ADC_INIT_STATUS_NON_ZERO_ACC_UB_VALUE                                       0x00000003U
#define LW2080_CTRL_PMGR_GENERIC_TEST_ID_ADC_INIT_STATUS_NON_ZERO_ACC_SCNT_VALUE                                     0x00000004U
#define LW2080_CTRL_PMGR_GENERIC_TEST_ID_ADC_INIT_STATUS_NON_ZERO_MUL_VALUE                                          0x00000005U
#define LW2080_CTRL_PMGR_GENERIC_TEST_ID_ADC_INIT_STATUS_NON_ZERO_MUL_ACC_LB_VALUE                                   0x00000006U
#define LW2080_CTRL_PMGR_GENERIC_TEST_ID_ADC_INIT_STATUS_NON_ZERO_MUL_ACC_UB_VALUE                                   0x00000007U
#define LW2080_CTRL_PMGR_GENERIC_TEST_ID_ADC_INIT_STATUS_NON_ZERO_MUL_ACC_SCNT_VALUE                                 0x00000008U

/*!
 * LW2080_CTRL_PMGR_GENERIC_TEST_ID_ADC_CHECK
 *
 * Possible reasons for LW2080_CTRL_PMGR_GENERIC_TEST_ID_ADC_CHECK to fail.
 *
 * _SUCCESS                : Test *_ID_ADC_CHECK is a success
 * _IIR_VALUE_MISMATCH     : Test *_ID_ADC_CHECK failed because HW and SW IIR values mismatched
 * _ACC_VALUE_MISMATCH     : Test *_ID_ADC_CHECK failed because HW and SW ACC values mismatched
 * _MUL_VALUE_MISMATCH     : Test *_ID_ADC_CHECK failed because HW and SW MUL values mismatched
 * _MUL_ACC_VALUE_MISMATCH : Test *_ID_ADC_CHECK failed because HW and SW MUL ACC values mismatched
 */
#define LW2080_CTRL_PMGR_GENERIC_TEST_ID_ADC_CHECK_STATUS_SUCCESS                                                    0x00000010U
#define LW2080_CTRL_PMGR_GENERIC_TEST_ID_ADC_CHECK_STATUS_IIR_VALUE_MISMATCH                                         0x00000011U
#define LW2080_CTRL_PMGR_GENERIC_TEST_ID_ADC_CHECK_STATUS_ACC_VALUE_MISMATCH                                         0x00000012U
#define LW2080_CTRL_PMGR_GENERIC_TEST_ID_ADC_CHECK_STATUS_MUL_VALUE_MISMATCH                                         0x00000013U
#define LW2080_CTRL_PMGR_GENERIC_TEST_ID_ADC_CHECK_STATUS_MUL_ACC_VALUE_MISMATCH                                     0x00000014U

/*!
 * LW2080_CTRL_PMGR_GENERIC_TEST_ID_HI_OFFSET_MAX_CHECK
 *
 * Possible reasons for LW2080_CTRL_PMGR_GENERIC_TEST_ID_HI_OFFSET_MAX_CHECK to fail.
 *
 * _SUCCESS                      : Test *_ID_HI_OFFSET_MAX_CHECK is a success
 * _HI_OFFSET_A_FAILURE          : Test *_ID_HI_OFFSET_MAX_CHECK failed because HW and SW values for HI_OFFSET_A mismatched
 * _HI_OFFSET_B_FAILURE          : Test *_ID_HI_OFFSET_MAX_CHECK failed because HW and SW values for HI_OFFSET_B mismatched
 * _HI_OFFSET_C_FAILURE          : Test *_ID_HI_OFFSET_MAX_CHECK failed because HW and SW values for HI_OFFSET_C mismatched
 * _SINGLE_IPC_CH_FAILURE        : Test *_ID_HI_OFFSET_MAX_CHECK failed because the SINGLE_IPC_CH test did not provide expected HI_OFFSET
 * _THREE_IPC_CH_FAILURE         : Test *_ID_HI_OFFSET_MAX_CHECK failed because the THREE_IPC_CH test did not provide expected HI_OFFSET
 * _SINGLE_IPC_CHP_FAILURE       : Test *_ID_HI_OFFSET_MAX_CHECK failed because the SINGLE_IPC_CHP test did not provide expected HI_OFFSET
 * _THREE_IPC_CH_FAILURE         : Test *_ID_HI_OFFSET_MAX_CHECK failed because the THREE_IPC_CHP test did not provide expected HI_OFFSET
 * _SINGLE_IPC_SUM_FAILURE       : Test *_ID_HI_OFFSET_MAX_CHECK failed because the SINGLE_IPC_SUM test did not provide expected HI_OFFSET
 * _THREE_IPC_SUM_SUM_CH_FAILURE : Test *_ID_HI_OFFSET_MAX_CHECK failed because the THREE_IPC_SUM_SUM_CH did not provide expected HI_OFFSET
 */
#define LW2080_CTRL_PMGR_GENERIC_TEST_ID_HI_OFFSET_MAX_CHECK_STATUS_SUCCESS                                          0x00000020U
#define LW2080_CTRL_PMGR_GENERIC_TEST_ID_HI_OFFSET_MAX_CHECK_STATUS_HI_OFFSET_A_FAILURE                              0x00000021U
#define LW2080_CTRL_PMGR_GENERIC_TEST_ID_HI_OFFSET_MAX_CHECK_STATUS_HI_OFFSET_B_FAILURE                              0x00000022U
#define LW2080_CTRL_PMGR_GENERIC_TEST_ID_HI_OFFSET_MAX_CHECK_STATUS_HI_OFFSET_C_FAILURE                              0x00000023U
#define LW2080_CTRL_PMGR_GENERIC_TEST_ID_HI_OFFSET_MAX_CHECK_STATUS_SINGLE_IPC_CH_FAILURE                            0x00000024U
#define LW2080_CTRL_PMGR_GENERIC_TEST_ID_HI_OFFSET_MAX_CHECK_STATUS_THREE_IPC_CH_FAILURE                             0x00000025U
#define LW2080_CTRL_PMGR_GENERIC_TEST_ID_HI_OFFSET_MAX_CHECK_STATUS_SINGLE_IPC_CHP_FAILURE                           0x00000026U
#define LW2080_CTRL_PMGR_GENERIC_TEST_ID_HI_OFFSET_MAX_CHECK_STATUS_THREE_IPC_CHP_FAILURE                            0x00000027U
#define LW2080_CTRL_PMGR_GENERIC_TEST_ID_HI_OFFSET_MAX_CHECK_STATUS_SINGLE_IPC_SUM_FAILURE                           0x00000028U
#define LW2080_CTRL_PMGR_GENERIC_TEST_ID_HI_OFFSET_MAX_CHECK_STATUS_THREE_IPC_SUM_SUM_CH_FAILURE                     0x00000029U

/*!
 * LW2080_CTRL_PMGR_GENERIC_TEST_ID_IPC_PARAMS_CHECK
 *
 * Possible reasons for LW2080_CTRL_PMGR_GENERIC_TEST_ID_IPC_PARAMS_CHECK to fail.
 *
 * _SUCCESS                        : Test *_ID_IPC_PARAMS_CHECK is a success
 * _CMOV_IIR_FAILURE               : Test *_ID_IPC_PARAMS_CHECK failed because IIR value didn't change in CMOV mode
 * _CMOV_MUL_FAILURE               : Test *_ID_IPC_PARAMS_CHECK failed because MUL value didn't change in CMOV mode
 * _CMOV_HI_OFFSET_FAILURE         : Test *_ID_IPC_PARAMS_CHECK failed because HI_OFFSET value didn't change in CMOV mode
 * _IPC_ALL_EN_FALIURE             : Test *_ID_IPC_PARAMS_CHECK failed because unexpected value of HI_OFFSET when all IPC instances are enabled
 * _IIR_DOWNSHIFT_FAILURE          : Test *_ID_IPC_PARAMS_CHECK failed because unexpected value of HI_OFFSET when IIR_DOWNSHIFT was programmed
 * _PROP_DOWNSHIFT_FAILURE         : Test *_ID_IPC_PARAMS_CHECK failed because unexpected value of HI_OFFSET when PROP_DOWNSHIFT was programmed
 * _IIR_AND_PROP_DOWNSHIFT_FAILURE : Test *_ID_IPC_PARAMS_CHECK failed because unexpected value of HI_OFFSET when IIR_DOWNSHIFT and PROP_DOWNSHIFT was programmed
 * _IIR_DOWNSHIFT_CHANGE_FAILURE   : Test *_ID_IPC_PARAMS_CHECK failed because unexpected value of HI_OFFSET when IIR_DOWNSHIFT was changed
 * _IIR_GAIN_DOWNSHIFT_FAILURE     : Test *_ID_IPC_PARAMS_CHECK failed because unexpected value of HI_OFFSET when IIR_GAIN and IIR_DOWNSHIFT was programmed
 */
#define LW2080_CTRL_PMGR_GENERIC_TEST_ID_IPC_PARAMS_CHECK_STATUS_SUCCESS                                             0x00000030U
#define LW2080_CTRL_PMGR_GENERIC_TEST_ID_IPC_PARAMS_CHECK_STATUS_CMOV_IIR_FAILURE                                    0x00000031U
#define LW2080_CTRL_PMGR_GENERIC_TEST_ID_IPC_PARAMS_CHECK_STATUS_CMOV_MUL_FAILURE                                    0x00000032U
#define LW2080_CTRL_PMGR_GENERIC_TEST_ID_IPC_PARAMS_CHECK_STATUS_CMOV_HI_OFFSET_FAILURE                              0x00000033U
#define LW2080_CTRL_PMGR_GENERIC_TEST_ID_IPC_PARAMS_CHECK_STATUS_IPC_ALL_EN_FALIURE                                  0x00000034U
#define LW2080_CTRL_PMGR_GENERIC_TEST_ID_IPC_PARAMS_CHECK_STATUS_IIR_DOWNSHIFT_FAILURE                               0x00000035U
#define LW2080_CTRL_PMGR_GENERIC_TEST_ID_IPC_PARAMS_CHECK_STATUS_PROP_DOWNSHIFT_FAILURE                              0x00000036U
#define LW2080_CTRL_PMGR_GENERIC_TEST_ID_IPC_PARAMS_CHECK_STATUS_IIR_AND_PROP_DOWNSHIFT_FAILURE                      0x00000037U
#define LW2080_CTRL_PMGR_GENERIC_TEST_ID_IPC_PARAMS_CHECK_STATUS_IIR_DOWNSHIFT_CHANGE_FAILURE                        0x00000038U
#define LW2080_CTRL_PMGR_GENERIC_TEST_ID_IPC_PARAMS_CHECK_STATUS_IIR_GAIN_DOWNSHIFT_FAILURE                          0x00000039U

/*!
 * LW2080_CTRL_PMGR_GENERIC_TEST_ID_BEACON_CHECK
 *
 * Possible reasons for LW2080_CTRL_PMGR_GENERIC_TEST_ID_BEACON_CHECK to fail.
 *
 * _SUCCESS                                                 : Test *_ID_BEACON_CHECK is a success
 * _BEACON1_INTERRUPT_NOT_PENDING                           : Test *_ID_BEACON_CHECK failed because BEACON1 did not raise interrupt when tested alone
 * _BEACON2_INTERRUPT_NOT_PENDING                           : Test *_ID_BEACON_CHECK failed because BEACON2 did not raise interrupt when tested alone
 * _BOTH_BEACON_INTERRUPTS_NOT_PENDING                      : Test *_ID_BEACON_CHECK failed because when tested simultaneously both BEACON1 and BEACON2 did not raise interrupt
 * _BEACON1_INTERRUPT_NOT_PENDING_BEACON2_INTERRUPT_PENDING : Test *_ID_BEACON_CHECK failed because when tested simultaneously BEACON1 did not raise interrupt even though BEACON2 did
 * _BEACON2_INTERRUPT_NOT_PENDING_BEACON1_INTERRUPT_PENDING : Test *_ID_BEACON_CHECK failed because when tested simultaneously BEACON2 did not raise interrupt even though BEACON1 did
 */
#define LW2080_CTRL_PMGR_GENERIC_TEST_ID_BEACON_CHECK_STATUS_SUCCESS                                                 0x00000040U
#define LW2080_CTRL_PMGR_GENERIC_TEST_ID_BEACON_CHECK_STATUS_BEACON1_INTERRUPT_NOT_PENDING                           0x00000041U
#define LW2080_CTRL_PMGR_GENERIC_TEST_ID_BEACON_CHECK_STATUS_BEACON2_INTERRUPT_NOT_PENDING                           0x00000042U
#define LW2080_CTRL_PMGR_GENERIC_TEST_ID_BEACON_CHECK_STATUS_BOTH_BEACON_INTERRUPTS_NOT_PENDING                      0x00000043U
#define LW2080_CTRL_PMGR_GENERIC_TEST_ID_BEACON_CHECK_STATUS_BEACON1_INTERRUPT_NOT_PENDING_BEACON2_INTERRUPT_PENDING 0x00000044U
#define LW2080_CTRL_PMGR_GENERIC_TEST_ID_BEACON_CHECK_STATUS_BEACON2_INTERRUPT_NOT_PENDING_BEACON1_INTERRUPT_PENDING 0x00000045U

/*!
 * LW2080_CTRL_PMGR_GENERIC_TEST_ID_OFFSET_CHECK
 *
 * Possible reasons for LW2080_CTRL_PMGR_GENERIC_TEST_ID_OFFSET_CHECK to fail.
 *
 * _SUCCESS                          : Test *_ID_OFFSET_CHECK is a success
 * _BOTH_OFFSETS_IIR_VALUE_ERROR     : Test *_ID_OFFSET_CHECK failed because both OFFSET1 and OFFSET2 action did not provide expected IIR value
 * _OFFSET1_IIR_VALUE_ERROR          : Test *_ID_OFFSET_CHECK failed because OFFSET1 did not provide expected IIR value even though OFFSET2 did
 * _OFFSET2_IIR_VALUE_ERROR          : Test *_ID_OFFSET_CHECK failed because OFFSET2 did not provide expected IIR value even though OFFSET1 did
 * _BOTH_OFFSETS_IIR_ACC_VALUE_ERROR : Test *_ID_OFFSET_CHECK failed because both OFFSET1 and OFFSET2 action did not provide expected IIR ACC value
 * _OFFSET1_IIR_ACC_VALUE_ERROR      : Test *_ID_OFFSET_CHECK failed because OFFSET1 did not provide expected IIR ACC value even though OFFSET2 did
 * _OFFSET2_IIR_ACC_VALUE_ERROR      : Test *_ID_OFFSET_CHECK failed because OFFSET2 did not provide expected IIR ACC value even though OFFSET1 did
 * _BOTH_OFFSETS_MUL_VALUE_ERROR     : Test *_ID_OFFSET_CHECK failed because both OFFSET1 and OFFSET2 action did not provide expected MUL value
 * _OFFSET1_MUL_VALUE_ERROR          : Test *_ID_OFFSET_CHECK failed because OFFSET1 did not provide expected MUL value even though OFFSET2 did
 * _OFFSET2_MUL_VALUE_ERROR          : Test *_ID_OFFSET_CHECK failed because OFFSET2 did not provide expected MUL value even though OFFSET1 did
 * _BOTH_OFFSETS_IIR_ACC_VALUE_ERROR : Test *_ID_OFFSET_CHECK failed because both OFFSET1 and OFFSET2 action did not provide expected MUL ACC value
 * _OFFSET1_MUL_ACC_VALUE_ERROR      : Test *_ID_OFFSET_CHECK failed because OFFSET1 did not provide expected MUL ACC value even though OFFSET2 did
 * _OFFSET2_MUL_ACC_VALUE_ERROR      : Test *_ID_OFFSET_CHECK failed because OFFSET2 did not provide expected MUL ACC value even though OFFSET1 did
 */
#define LW2080_CTRL_PMGR_GENERIC_TEST_ID_OFFSET_CHECK_STATUS_SUCCESS                                                 0x00000050U
#define LW2080_CTRL_PMGR_GENERIC_TEST_ID_OFFSET_CHECK_STATUS_BOTH_OFFSETS_IIR_VALUE_ERROR                            0x00000051U
#define LW2080_CTRL_PMGR_GENERIC_TEST_ID_OFFSET_CHECK_STATUS_OFFSET1_IIR_VALUE_ERROR                                 0x00000052U
#define LW2080_CTRL_PMGR_GENERIC_TEST_ID_OFFSET_CHECK_STATUS_OFFSET2_IIR_VALUE_ERROR                                 0x00000053U
#define LW2080_CTRL_PMGR_GENERIC_TEST_ID_OFFSET_CHECK_STATUS_BOTH_OFFSETS_IIR_ACC_VALUE_ERROR                        0x00000054U
#define LW2080_CTRL_PMGR_GENERIC_TEST_ID_OFFSET_CHECK_STATUS_OFFSET1_IIR_ACC_VALUE_ERROR                             0x00000055U
#define LW2080_CTRL_PMGR_GENERIC_TEST_ID_OFFSET_CHECK_STATUS_OFFSET2_IIR_ACC_VALUE_ERROR                             0x00000056U
#define LW2080_CTRL_PMGR_GENERIC_TEST_ID_OFFSET_CHECK_STATUS_BOTH_OFFSETS_MUL_VALUE_ERROR                            0x00000057U
#define LW2080_CTRL_PMGR_GENERIC_TEST_ID_OFFSET_CHECK_STATUS_OFFSET1_MUL_VALUE_ERROR                                 0x00000058U
#define LW2080_CTRL_PMGR_GENERIC_TEST_ID_OFFSET_CHECK_STATUS_OFFSET2_MUL_VALUE_ERROR                                 0x00000059U
#define LW2080_CTRL_PMGR_GENERIC_TEST_ID_OFFSET_CHECK_STATUS_BOTH_OFFSETS_MUL_ACC_VALUE_ERROR                        0x0000005AU
#define LW2080_CTRL_PMGR_GENERIC_TEST_ID_OFFSET_CHECK_STATUS_OFFSET1_MUL_ACC_VALUE_ERROR                             0x0000005BU
#define LW2080_CTRL_PMGR_GENERIC_TEST_ID_OFFSET_CHECK_STATUS_OFFSET2_MUL_ACC_VALUE_ERROR                             0x0000005LW

#define LW2080_CTRL_PMGR_TEST_STATUS(testid, status) (LW2080_CTRL_PMGR_GENERIC_TEST_ID_##testid##_STATUS_##status)

/*
 * LW2080_CTRL_CMD_PMGR_PMGR_GENERIC_TEST
 *
 *   Possible Pmgr Generic Test results.
 *
 * _SUCCESS               : Test completed successfully
 * _NOT_IMPLEMENTED       : Test is not implemented in RM/PMU
 * _NOT_SUPPORTED         : Test is not supported on the GPU
 * _UNSPECIFIED_PMU_ERROR : Test ran into an unspecified PMU error
 * _ERROR_GENERIC         : Otherwise
 *
 */
#define LW2080_CTRL_PMGR_GENERIC_TEST_SUCCESS                                                                        0x00000000U
#define LW2080_CTRL_PMGR_GENERIC_TEST_NOT_IMPLEMENTED                                                                0x00000001U
#define LW2080_CTRL_PMGR_GENERIC_TEST_NOT_SUPPORTED                                                                  0x00000002U
#define LW2080_CTRL_PMGR_GENERIC_TEST_INSUFFICIENT_PRIVILEGE                                                         0x00000003U
#define LW2080_CTRL_PMGR_GENERIC_TEST_UNSPECIFIED_PMU_ERROR                                                          0x00000004U
#define LW2080_CTRL_PMGR_GENERIC_TEST_ERROR_GENERIC                                                                  0xFFFFFFFFU

/*!
 * LW2080_CTRL_CMD_PMGR_PMGR_GENERIC_TEST
 *
 * This command runs one of the PMGR halified tests specified by
 * LW2080_CTRL_PMGR_GENERIC_TEST_ID_<xyz>
 *
 * Possible status values returned are:
 *  LW_OK
 */
#define LW2080_CTRL_CMD_PMGR_GENERIC_TEST                                                                            (0x20802635U) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PMGR_INTERFACE_ID << 8) | LW2080_CTRL_PMGR_GENERIC_TEST_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_PMGR_GENERIC_TEST_PARAMS_MESSAGE_ID (0x35U)

typedef struct LW2080_CTRL_PMGR_GENERIC_TEST_PARAMS {
    /*!
     * [in] - Specifies the index of the test to execute as per
     * @ref pmgrtest_name_a defined in rmt_pmgr.h
     */
    LwU32         index;

    /*!
     * [out] - Specifies the generic status of the test exelwted specified
     * by @index which is one of LW2080_CTRL_PMGR_GENERIC_TEST_<xyz>
     */
    LwU32         outStatus;

    /*!
     * [out] - Specifies the output status data given by the test exelwted
     * specified by @index.
     */
    LwU32         outData;

    /*!
     * [out] - Specifies the observed value for the test exelwted
     * specified by @index.
     */
    LwU64_ALIGN32 observedVal;

    /*!
     * [out] - Specifies the expected value for the test exelwted
     * specified by @index.
     */
    LwU64_ALIGN32 expectedVal;
} LW2080_CTRL_PMGR_GENERIC_TEST_PARAMS;
typedef struct LW2080_CTRL_PMGR_GENERIC_TEST_PARAMS *PLW2080_CTRL_PMGR_GENERIC_TEST_PARAMS;

/*!
 * LW2080_CTRL_CMD_PMGR_SETUP_BA_DMA_MODE
 *
 * Control call to setup RM-driven BA DMA mode.
 *
 * Possible status values returned are:
 *  LW_OK
 */
#define LW2080_CTRL_CMD_PMGR_SETUP_BA_DMA_MODE       (0x20802636U) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PMGR_INTERFACE_ID << 8) | LW2080_CTRL_CMD_PMGR_SETUP_BA_DMA_MODE_PARAMS_MESSAGE_ID" */

 /*!
  * Maximum number of temperature registers sampled in BA counter methods.
  */
#define LW2080_CTRL_PMGR_SETUP_BA_MAX_TEMP_REGISTERS (16)

#define LW2080_CTRL_CMD_PMGR_SETUP_BA_DMA_MODE_PARAMS_MESSAGE_ID (0x36U)

typedef struct LW2080_CTRL_CMD_PMGR_SETUP_BA_DMA_MODE_PARAMS {
     /*!
      * [in] - Specifies the handle to physical memory.  Used for DMA
      * destination memory.
      */
    LwU32  hPhysHandle;

     /*!
      * [in] - Hardware format of (TINT << 16) | TRUN in microseconds.
      */
    LwU32  timingParams;

     /*!
      * [in] - Flag used to indicate DMA is active.  1 means active;
      * 0 means inactive or completion.
      */
    LwU32  triggerStatus;

     /*!
      * [out] - GPC captured one statement before starting BA counter DMA.
      */
    LW_DECLARE_ALIGNED(LwU64 cycleCount, 8);

     /*!
      * [out] - Temperature registers from hardware.
      */
    LwU32  tempRegisters[LW2080_CTRL_PMGR_SETUP_BA_MAX_TEMP_REGISTERS];

     /*!
      * [in] - Flag used to indicate start DMA vs. check DMA status.
      */
    LwBool trigger;
} LW2080_CTRL_CMD_PMGR_SETUP_BA_DMA_MODE_PARAMS;
typedef struct LW2080_CTRL_CMD_PMGR_SETUP_BA_DMA_MODE_PARAMS *PLW2080_CTRL_CMD_PMGR_SETUP_BA_DMA_MODE_PARAMS;

/*!
 * LW2080_CTRL_CMD_PMGR_PMUMON_PWR_CHANNELS_GET_SAMPLES
 *
 * Control call to query the samples within the PWR_CHANNELS PMUMON queue.
 */
#define LW2080_CTRL_CMD_PMGR_PMUMON_PWR_CHANNELS_GET_SAMPLES (0x20802637U) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PMGR_INTERFACE_ID << 8) | LW2080_CTRL_PMGR_PMUMON_PWR_CHANNELS_GET_SAMPLES_PARAMS_MESSAGE_ID" */

/*!
 * @brief   With sample period being potentially as fast every 5ms, this gives
 *          us 5 seconds worth of data.
 */
#define LW2080_CTRL_PMGR_PMUMON_PWR_CHANNELS_SAMPLE_COUNT    (1000U)

/*!
 * Temporary until an INFO control call is stubbed out that exposes the supported
 * feature set of the sampling.
 */
#define LW2080_CTRL_PMGR_PMUMON_PWR_CHANNELS_SAMPLE_ILWALID  (LW_U32_MAX)

/*!
 * A single sample of the power channels at a particular point in time.
 */
typedef struct LW2080_CTRL_PMGR_PMUMON_PWR_CHANNELS_SAMPLE {
    /*!
     * Ptimer timestamp of when this data was collected.
     */
    LW_DECLARE_ALIGNED(LW2080_CTRL_PMUMON_SAMPLE super, 8);

    /*!
     * Total GPU power in milliwatts.
     *
     * LW2080_CTRL_PMGR_PMUMON_PWR_CHANNELS_SAMPLE_ILWALID if not supported.
     */
    LwU32 tgpmW;

    /*!
     * Core power in milliwatts.
     *
     * LW2080_CTRL_PMGR_PMUMON_PWR_CHANNELS_SAMPLE_ILWALID if not supported.
     */
    LwU32 coremW;

    /*!
     * Average total GPU power in milliwatts for the last second.
     *
     * LW2080_CTRL_PMGR_PMUMON_PWR_CHANNELS_SAMPLE_ILWALID if not supported.
     */
    LwU32 averageTgpmW;

    /*!
     * Average core power in milliwatts for the last second.
     *
     * LW2080_CTRL_PMGR_PMUMON_PWR_CHANNELS_SAMPLE_ILWALID if not supported.
     */
    LwU32 averageCoremW;
} LW2080_CTRL_PMGR_PMUMON_PWR_CHANNELS_SAMPLE;
typedef struct LW2080_CTRL_PMGR_PMUMON_PWR_CHANNELS_SAMPLE *PLW2080_CTRL_PMGR_PMUMON_PWR_CHANNELS_SAMPLE;

/*!
 * Input/Output parameters for @ref LW2080_CTRL_CMD_PMGR_PMUMON_PWR_CHANNEL_SAMPLES
 */
#define LW2080_CTRL_PMGR_PMUMON_PWR_CHANNELS_GET_SAMPLES_PARAMS_MESSAGE_ID (0x37U)

typedef struct LW2080_CTRL_PMGR_PMUMON_PWR_CHANNELS_GET_SAMPLES_PARAMS {
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
    LW_DECLARE_ALIGNED(LW2080_CTRL_PMGR_PMUMON_PWR_CHANNELS_SAMPLE samples[LW2080_CTRL_PMGR_PMUMON_PWR_CHANNELS_SAMPLE_COUNT], 8);
} LW2080_CTRL_PMGR_PMUMON_PWR_CHANNELS_GET_SAMPLES_PARAMS;
typedef struct LW2080_CTRL_PMGR_PMUMON_PWR_CHANNELS_GET_SAMPLES_PARAMS *PLW2080_CTRL_PMGR_PMUMON_PWR_CHANNELS_GET_SAMPLES_PARAMS;

/*!
 * LW2080_CTRL_CMD_PMGR_GET_MODULE_ID
 *
 * Control call to query the subdevice module ID.
 *
 * This is a static HW identifier that is unique for each module on a given baseboard.
 * For non-baseboard products this would always be 0.
 */
#define LW2080_CTRL_CMD_PMGR_GET_MODULE_ID (0x20802638U) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PMGR_INTERFACE_ID << 8) | LW2080_CTRL_PMGR_MODULE_ID_PARAMS_MESSAGE_ID" */

/*!
 */
#define LW2080_CTRL_PMGR_MODULE_ID_PARAMS_MESSAGE_ID (0x38U)

typedef struct LW2080_CTRL_PMGR_MODULE_ID_PARAMS {
    /*!
     * [out] Module ID for the given GPU.
     */
    LwU32 moduleId;
} LW2080_CTRL_PMGR_MODULE_ID_PARAMS;

/* _ctrl2080pmgr_h_ */
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)

