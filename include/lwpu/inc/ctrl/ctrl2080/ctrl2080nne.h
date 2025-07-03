/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2020 LWPU CORPORATION & AFFILIATES. All rights reserved.
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
// Source file: ctrl/ctrl2080/ctrl2080nne.finn
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
#include "lwstatus.h"

/*!
 * Note: Structures in this file may be shared between RM<->PMU, ergo, 64-bit
 * datatypes must use LwU64_ALIGN32.
 */

/* --------------------------- NNE Variable --------------------------------- */
/*!
 * @defgroup LW2080_CTRL_NNE_NNE_VAR_TYPE_ENUM
 *
 * Enumeration of NNE variable types.
 *
 * @{
 */
typedef LwU8 LW2080_CTRL_NNE_NNE_VAR_TYPE_ENUM;
#define LW2080_CTRL_NNE_NNE_VAR_TYPE_FREQ        0x00U
#define LW2080_CTRL_NNE_NNE_VAR_TYPE_PM          0x01U
#define LW2080_CTRL_NNE_NNE_VAR_TYPE_CHIP_CONFIG 0x02U
#define LW2080_CTRL_NNE_NNE_VAR_TYPE_POWER_DN    0x03U
#define LW2080_CTRL_NNE_NNE_VAR_TYPE_POWER_TOTAL 0x04U
/*!@}*/

/*!
 * @brief Structure for tuple of uniquely identifying an NNE_VAR_FREQ NNE variable.
 */
typedef struct LW2080_CTRL_NNE_VAR_ID_FREQ {
    /*!
     * @brief Clock domain index.
     */
    LwBoardObjIdx clkDomainIdx;

    /*!
     * @brief Clock specification mode (i.e. absolute vs relative).
     *
     * LW_TRUE    Value of the clock freq should be interpreted literally as clock frequency
     * LW_FALSE   Value of the clock freq should be interpreted as a
     *            frequency delta relative to the current clock frequency.
     */
    LwBool        bAbsolute;
} LW2080_CTRL_NNE_VAR_ID_FREQ;
typedef struct LW2080_CTRL_NNE_VAR_ID_FREQ *PLW2080_CTRL_NNE_VAR_ID_FREQ;

/*!
 * @brief   Structure describing @ref LW2080_CTRL_NNE_NNE_VAR_TYPE_FREQ static
 *          information/POR.
 */
typedef struct LW2080_CTRL_NNE_NNE_VAR_INFO_FREQ {
    /*!
     * @brief   Identifying information for this frequency variable
     */
    LW2080_CTRL_NNE_VAR_ID_FREQ freqId;
} LW2080_CTRL_NNE_NNE_VAR_INFO_FREQ;
typedef struct LW2080_CTRL_NNE_NNE_VAR_INFO_FREQ *PLW2080_CTRL_NNE_NNE_VAR_INFO_FREQ;

/*!
 * @brief Structure to specify a clock-frequency value to use for inference.
 */
typedef struct LW2080_CTRL_NNE_NNE_VAR_INPUT_FREQ {
    /*!
     * @brief Tuple uniquely identifying a clock-frequency variable.
     */
    LW2080_CTRL_NNE_VAR_ID_FREQ freqId;

    /*!
     * @brief Input clock-frequency value, in MHz.
     */
    LwS32                       freqMhz;
} LW2080_CTRL_NNE_NNE_VAR_INPUT_FREQ;
typedef struct LW2080_CTRL_NNE_NNE_VAR_INPUT_FREQ *PLW2080_CTRL_NNE_NNE_VAR_INPUT_FREQ;

/*!
 * @brief Structure for tuple of uniquely identifying an NNE_VAR_PM NNE variable
 */
typedef struct LW2080_CTRL_NNE_VAR_ID_PM {
    /*!
     * @brief BA PM index
     */
    LwU16 baIdx;
} LW2080_CTRL_NNE_VAR_ID_PM;
typedef struct LW2080_CTRL_NNE_VAR_ID_PM *PLW2080_CTRL_NNE_VAR_ID_PM;

/*!
 * @brief   Structure describing @ref LW2080_CTRL_NNE_NNE_VAR_TYPE_PM static
 *          information/POR.
 */
typedef struct LW2080_CTRL_NNE_NNE_VAR_INFO_PM {
    /*!
     * @brief   Identifying information for this PM variable
     */
    LW2080_CTRL_NNE_VAR_ID_PM pmId;
    /*!
     * Secondary normalization value for this PM signal (as IEEE-754
     * 32-bit floating point).  A value of zero (0.0) indicates
     * normalization is disabled for this PM.
     */
    LwU32                     secNorm;
} LW2080_CTRL_NNE_NNE_VAR_INFO_PM;
typedef struct LW2080_CTRL_NNE_NNE_VAR_INFO_PM *PLW2080_CTRL_NNE_NNE_VAR_INFO_PM;

/*!
 * @brief Structure to specify a Block Activity (BA) Performance Monitor (PM)
 *        value to use for inference.
 */
typedef struct LW2080_CTRL_NNE_NNE_VAR_INPUT_PM {
    /*!
     * @brief Tuple uniquely identifying a BA PM variable.
     */
    LW2080_CTRL_NNE_VAR_ID_PM pmId;

    /*!
     * @brief Input BA PM count
     *
     * Note: this structure is used in both RM(64-bit) and PMU(32-bit), necessitating
     * the need to 32-bit align everything
     */
    LwU64_ALIGN32             pmCount;
} LW2080_CTRL_NNE_NNE_VAR_INPUT_PM;
typedef struct LW2080_CTRL_NNE_NNE_VAR_INPUT_PM *PLW2080_CTRL_NNE_NNE_VAR_INPUT_PM;

/*!
 * @defgroup LW2080_CTRL_NNE_NNE_VAR_CHIP_CONFIG_CONFIG_TYPE_ENUM
 *
 * Enumeration of all CHIP_CONFIG input variable types supported by NNE.
 *
 * @{
 */
typedef LwU8 LW2080_CTRL_NNE_NNE_VAR_CHIP_CONFIG_CONFIG_TYPE_ENUM;
#define LW2080_CTRL_NNE_NNE_VAR_CHIP_CONFIG_CONFIG_TYPE_TPC       0x00U
#define LW2080_CTRL_NNE_NNE_VAR_CHIP_CONFIG_CONFIG_TYPE_FBP       0x01U
#define LW2080_CTRL_NNE_NNE_VAR_CHIP_CONFIG_CONFIG_TYPE_LTC_SLICE 0x02U
#define LW2080_CTRL_NNE_NNE_VAR_CHIP_CONFIG_CONFIG_TYPE_NUM       0x03U
/*!@}*/

/*!
 * @brief Structure for tuple of uniquely identifying an NNE_VAR_CHIP_CONFIG NNE variable
 */
typedef struct LW2080_CTRL_NNE_VAR_ID_CHIP_CONFIG {
    /*!
     * @brief chip-config information type
     *
     * @ref LW2080_CTRL_NNE_NNE_VAR_CHIP_CONFIG_CONFIG_TYPE_ENUM
     */
    LW2080_CTRL_NNE_NNE_VAR_CHIP_CONFIG_CONFIG_TYPE_ENUM configType;
} LW2080_CTRL_NNE_VAR_ID_CHIP_CONFIG;
typedef struct LW2080_CTRL_NNE_VAR_ID_CHIP_CONFIG *PLW2080_CTRL_NNE_VAR_ID_CHIP_CONFIG;

/*!
 * @brief   Structure describing @ref LW2080_CTRL_NNE_NNE_VAR_TYPE_CHIP_CONFIG static
 *          information/POR.
 */
typedef struct LW2080_CTRL_NNE_NNE_VAR_INFO_CHIP_CONFIG {
    /*!
     * @brief   Identifying information for this PM variable
     */
    LW2080_CTRL_NNE_VAR_ID_CHIP_CONFIG configId;
} LW2080_CTRL_NNE_NNE_VAR_INFO_CHIP_CONFIG;
typedef struct LW2080_CTRL_NNE_NNE_VAR_INFO_CHIP_CONFIG *PLW2080_CTRL_NNE_NNE_VAR_INFO_CHIP_CONFIG;

/*!
 * @brief Structure to specify a chip-configuration value for use
 *        for inference.
 */
typedef struct LW2080_CTRL_NNE_NNE_VAR_INPUT_CHIP_CONFIG {
    /*!
     * @brief Tuple uniquely identifying a clock-frequency variable.
     */
    LW2080_CTRL_NNE_VAR_ID_CHIP_CONFIG configId;

    /*!
     * @brief Input clock-frequency value, in MHz.
     */
    LwU32                              config;
} LW2080_CTRL_NNE_NNE_VAR_INPUT_CHIP_CONFIG;
typedef struct LW2080_CTRL_NNE_NNE_VAR_INPUT_CHIP_CONFIG *PLW2080_CTRL_NNE_NNE_VAR_INPUT_CHIP_CONFIG;

/*!
 * @brief Structure for tuple of uniquely identifying an NNE_VAR_POWER_DN NNE variable
 */
typedef struct LW2080_CTRL_NNE_VAR_ID_POWER_DN {
    /*!
     * @brief VOLT_RAIL index for the power rail.
     */
    LwBoardObjIdx voltRailIdx;
} LW2080_CTRL_NNE_VAR_ID_POWER_DN;
typedef struct LW2080_CTRL_NNE_VAR_ID_POWER_DN *PLW2080_CTRL_NNE_VAR_ID_POWER_DN;

/*!
 * @brief   Structure describing @ref LW2080_CTRL_NNE_NNE_VAR_TYPE_POWER_DN static
 *          information/POR.
 */
typedef struct LW2080_CTRL_NNE_NNE_VAR_INFO_POWER_DN {
    /*!
     * @brief   Identifying information for this PM variable
     */
    LW2080_CTRL_NNE_VAR_ID_POWER_DN id;

    /*!
     * @brief Voltage normalization value for the dynamic "normalized"
     * power.  NNE will normalize the power from 1.0 V client input to
     * this voltage value on inference.
     */
    LwU32                           voltageuV;
} LW2080_CTRL_NNE_NNE_VAR_INFO_POWER_DN;
typedef struct LW2080_CTRL_NNE_NNE_VAR_INFO_POWER_DN *PLW2080_CTRL_NNE_NNE_VAR_INFO_POWER_DN;

/*!
 * @brief Structure to specify a chip-configuration value for use
 *        for inference.
 */
typedef struct LW2080_CTRL_NNE_NNE_VAR_INPUT_POWER_DN {
    /*!
     * @brief   Identifying information for this PM variable
     */
    LW2080_CTRL_NNE_VAR_ID_POWER_DN id;

    /*!
     * @brief Power value in mW.
     */
    LwU32                           powermW;
} LW2080_CTRL_NNE_NNE_VAR_INPUT_POWER_DN;
typedef struct LW2080_CTRL_NNE_NNE_VAR_INPUT_POWER_DN *PLW2080_CTRL_NNE_NNE_VAR_INPUT_POWER_DN;

/*!
 * @defgroup LW2080_CTRL_NNE_VAR_ID_POWER_TOTAL_VOLT_RAIL_NAME
 *
 * Voltage/power rail "names" for the NNE_VAR_POWER_TOTAL class.
 * These are used to describe rails which aren't properly
 * defined/described within RM/PMU PERF/PWR infrastructure.  For now,
 * this is really just FBVDD.
 *
 * @{
 */
#define LW2080_CTRL_NNE_VAR_ID_POWER_TOTAL_VOLT_RAIL_NAME_FBVDD 0x00
#define LW2080_CTRL_NNE_VAR_ID_POWER_TOTAL_VOLT_RAIL_NAME_MAX   0x01
/*! @} */

/*!
 * @brief Structure for tuple of uniquely identifying an NNE_VAR_POWER_TOTAL NNE variable
 */
typedef struct LW2080_CTRL_NNE_VAR_ID_POWER_TOTAL {
    /*!
     * @brief "NAME" index for the power rail used here.  @ref
     * LW2080_CTRL_NNE_VAR_ID_POWER_TOTAL_VOLT_RAIL_NAME.
     */
    LwU8 voltRailName;
} LW2080_CTRL_NNE_VAR_ID_POWER_TOTAL;
typedef struct LW2080_CTRL_NNE_VAR_ID_POWER_TOTAL *PLW2080_CTRL_NNE_VAR_ID_POWER_TOTAL;

/*!
 * @brief   Structure describing @ref LW2080_CTRL_NNE_NNE_VAR_TYPE_POWER_TOTAL static
 *          information/POR.
 */
typedef struct LW2080_CTRL_NNE_NNE_VAR_INFO_POWER_TOTAL {
    /*!
     * @brief   Identifying information for this PM variable
     */
    LW2080_CTRL_NNE_VAR_ID_POWER_TOTAL id;
} LW2080_CTRL_NNE_NNE_VAR_INFO_POWER_TOTAL;
typedef struct LW2080_CTRL_NNE_NNE_VAR_INFO_POWER_TOTAL *PLW2080_CTRL_NNE_NNE_VAR_INFO_POWER_TOTAL;

/*!
 * @brief Structure to specify a chip-configuration value for use
 *        for inference.
 */
typedef struct LW2080_CTRL_NNE_NNE_VAR_INPUT_POWER_TOTAL {
    /*!
     * @brief   Identifying information for this PM variable
     */
    LW2080_CTRL_NNE_VAR_ID_POWER_TOTAL id;

    /*!
     * @brief Power value in mW.
     */
    LwU32                              powermW;
} LW2080_CTRL_NNE_NNE_VAR_INPUT_POWER_TOTAL;
typedef struct LW2080_CTRL_NNE_NNE_VAR_INPUT_POWER_TOTAL *PLW2080_CTRL_NNE_NNE_VAR_INPUT_POWER_TOTAL;

/*!
 * @brief   NNE_VAR type-specific data union. Discriminated by NNE_VAR::super.type.
 */


/*!
 * @brief   Structure describing NNE_VAR static information/POR. Implements the
 *          BOARDOBJ model/interface.
 */
typedef struct LW2080_CTRL_NNE_NNE_VAR_INFO {
    /*!
     * @brief   LW2080_CTRL_BOARDOBJ super class. Must always be first object in
     *          structure.
     */
    LW2080_CTRL_BOARDOBJ super;

    /*!
     * @brief   XAPI does not support using an element within another structure
     *          as a discriminant.  Placing this redundant type value here until
     *          that design constraint can be fixed.
     */
    LwU8                 type;

    /*!
     * @brief   Type-specific data union.
     */
    union {
        LW2080_CTRL_NNE_NNE_VAR_INFO_PM          pm;
        LW2080_CTRL_NNE_NNE_VAR_INFO_FREQ        freq;
        LW2080_CTRL_NNE_NNE_VAR_INFO_CHIP_CONFIG config;
        LW2080_CTRL_NNE_NNE_VAR_INFO_POWER_DN    powerDn;
        LW2080_CTRL_NNE_NNE_VAR_INFO_POWER_TOTAL powerTotal;
    } data;
} LW2080_CTRL_NNE_NNE_VAR_INFO;
typedef struct LW2080_CTRL_NNE_NNE_VAR_INFO *PLW2080_CTRL_NNE_NNE_VAR_INFO;

/*!
 * @brief   Structure describing NNE_VARS static information/POR.  Implements the
 *          BOARDOBJGRP model/interface.
 */
#define LW2080_CTRL_NNE_NNE_VARS_INFO_MESSAGE_ID (0x0U)

typedef struct LW2080_CTRL_NNE_NNE_VARS_INFO {
    /*!
     * @brief   LW2080_CTRL_BOARDOBJGRP_E512 super class.  Must always be first
     *          object in structure.
     */
    LW2080_CTRL_BOARDOBJGRP_E512 super;

    /*!
     * @brief   Array of NNE_VAR structures. Has valid indexes corresponding to
     *          the bits set in @ref super.objMask.
     */
    LW2080_CTRL_NNE_NNE_VAR_INFO vars[LW2080_CTRL_BOARDOBJGRP_E512_MAX_OBJECTS];
} LW2080_CTRL_NNE_NNE_VARS_INFO;
typedef struct LW2080_CTRL_NNE_NNE_VARS_INFO *PLW2080_CTRL_NNE_NNE_VARS_INFO;

/*!
 * LW2080_CTRL_CMD_NNE_VARS_GET_INFO
 *
 * This command returns NNE_VARS static object information/POR as specified
 * by the VBIOS in NNE Variables Table.
 *
 * See @ref LW2080_CTRL_NNE_NNE_VARS_INFO for documentation on the
 * parameters.
 *
 * Return values are specified per @ref BoardObjGrpGetInfo.
 */
#define LW2080_CTRL_CMD_NNE_NNE_VARS_GET_INFO (0x20803700) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_NNE_INTERFACE_ID << 8) | LW2080_CTRL_NNE_NNE_VARS_INFO_MESSAGE_ID" */

/*!
 * @brief   NNE_VAR input type-specific data union. Discriminated by NNE_VAR::super.type.
 */


/*!
 * Structure describing the input value of an NNE_VAR in the context
 * of a client's requested NNE_DESC inference evaluation.
 */
typedef struct LW2080_CTRL_NNE_NNE_VAR_INPUT {
    /*!
     * @brief   XAPI does not support using an element within another structure
     *          as a discriminant.  Placing this redundant type value here until
     *          that design constraint can be fixed.
     */
    LwU8 type;

    /*!
     * @brief   Type-specific data union.
     */
    union {
        LW2080_CTRL_NNE_NNE_VAR_INPUT_PM          pm;
        LW2080_CTRL_NNE_NNE_VAR_INPUT_FREQ        freq;
        LW2080_CTRL_NNE_NNE_VAR_INPUT_CHIP_CONFIG config;
        LW2080_CTRL_NNE_NNE_VAR_INPUT_POWER_DN    powerDn;
        LW2080_CTRL_NNE_NNE_VAR_INPUT_POWER_TOTAL powerTotal;
    } data;
} LW2080_CTRL_NNE_NNE_VAR_INPUT;
typedef struct LW2080_CTRL_NNE_NNE_VAR_INPUT *PLW2080_CTRL_NNE_NNE_VAR_INPUT;

/* --------------------------- NNE Layer ------------------------------------ */

/*!
 * @defgroup LW2080_CTRL_NNE_NNE_LAYER_TYPE_ENUM
 *
 * Enumeration of NNE layer types.
 *
 * @{
 */
typedef LwU8 LW2080_CTRL_NNE_NNE_LAYER_TYPE_ENUM;
#define LW2080_CTRL_NNE_NNE_LAYER_TYPE_FC_10 0x00U
/*!@}*/

/*!
 * @defgroup LW2080_CTRL_NNE_NNE_LAYER_FC_10_ACTIVATION_FUNCTION_TYPE_ENUM
 *
 * Fully-connected layer version 1.0 supported activation functions.
 *
 * @{
 */
typedef LwU8 LW2080_CTRL_NNE_NNE_LAYER_FC_10_ACTIVATION_FUNCTION_TYPE_ENUM;
typedef LwU8 *PLW2080_CTRL_NNE_NNE_LAYER_FC_10_ACTIVATION_FUNCTION_TYPE_ENUM;
#define LW2080_CTRL_NNE_NNE_LAYER_FC_10_ACTIVATION_FUNCTION_TYPE_IDENTITY   0x00U
#define LW2080_CTRL_NNE_NNE_LAYER_FC_10_ACTIVATION_FUNCTION_TYPE_RELU       0x01U
#define LW2080_CTRL_NNE_NNE_LAYER_FC_10_ACTIVATION_FUNCTION_TYPE_LEAKY_RELU 0x02U
/*!@}*/

/*!
 * @brief   Structure describing @ref LW2080_CTRL_NNE_NNE_LAYER_TYPE_FC_10
 *          static information/POR.
 */
typedef struct LW2080_CTRL_NNE_NNE_LAYER_INFO_FC_10 {
    /*!
     * @brief   Number of inputs into this layer
     */
    LwU16                                                         numInputs;

    /*!
     * @brief   Number of neurons in this layer
     */
    LwU8                                                          numNeurons;

    /*!
     * @brief   Whether this layer uses a bias
     */
    LwBool                                                        bHasBias;

    /*!
     * @brief   The activation function for this layer
     */
    LW2080_CTRL_NNE_NNE_LAYER_FC_10_ACTIVATION_FUNCTION_TYPE_ENUM activationFunction;

    /*!
     * @brief   If @ref LW2080_CTRL_NNE_NNE_LAYER_INFO_FC_10::activationFunction
     *          is @ref LW2080_CTRL_NNE_NNE_LAYER_FC_10_ACTIVATION_FUNCTION_TYPE_LEAKY_RELU,
     *          provides the function's slope.
     *
     * @note    This is actually a 32-bit floating point number, not an
     *          unsigned integer.
     */
    LwU32                                                         leakyReLUSlope;
} LW2080_CTRL_NNE_NNE_LAYER_INFO_FC_10;

/*!
 * @brief   NNE_LAYER type-specific data union.  Discriminated by
 *          NNE_LAYER::super.type.
 */


/*!
 * @defgroup LW2080_CTRL_NNE_NNE_LAYER_WEIGHT_TYPE_ENUM
 *
 * Layer weight data type.
 *
 * @{
 */
typedef LwU8 LW2080_CTRL_NNE_NNE_LAYER_WEIGHT_TYPE_ENUM;
typedef LwU8 *PLW2080_CTRL_NNE_NNE_LAYER_WEIGHT_TYPE_ENUM;
#define LW2080_CTRL_NNE_NNE_LAYER_WEIGHT_TYPE_FP16 0x00U
#define LW2080_CTRL_NNE_NNE_LAYER_WEIGHT_TYPE_FP32 0x01U
/*!@}*/

/*!
 * @brief   Macro for indicating that an NNE Layer index is invalid.
 */
#define LW2080_CTRL_NNE_NNE_LAYER_INDEX_ILWALID    LW2080_CTRL_BOARDOBJ_IDX_ILWALID_8BIT

/*!
 * @brief   Structure describing NNE_LAYER static information/POR. Implements the
 *          BOARDOBJ model/interface.
 */
typedef struct LW2080_CTRL_NNE_NNE_LAYER_INFO {
    /*!
     * @brief   LW2080_CTRL_BOARDOBJ super class. Must always be first object in
     *          structure.
     */
    LW2080_CTRL_BOARDOBJ                       super;

    /*!
     * @brief   XAPI does not support using an element within another structure
     *          as a discriminant.  Placing this redundant type value here until
     *          that design constraint can be fixed.
     */
    LwU8                                       type;

    /*!
     * @brief   Index of the next layer in the network.
     */
    LwBoardObjIdx                              nextLayerIdx;

    /*!
     * @brief   Index of the previous layer in the network
     */
    LwBoardObjIdx                              prevLayerIdx;

    /*!
     * @brief   Type of the weights in this NNE_LAYER
     */
    LW2080_CTRL_NNE_NNE_LAYER_WEIGHT_TYPE_ENUM weightType;

    /*!
     * @brief   Index of the first weight for this layer in
     *          @ref LW2080_CTRL_NNE_NNE_LAYERS_INFO::fp16Weights or
     *          @ref LW2080_CTRL_NNE_NNE_LAYERS_INFO::fp32Weights,
     *          depending on weightType.
     */
    LwU16                                      weightIdx;

    /*!
     * Type-specific data union.
     */
    union {
        LW2080_CTRL_NNE_NNE_LAYER_INFO_FC_10 fc10;
    } data;
} LW2080_CTRL_NNE_NNE_LAYER_INFO;

/*!
 * @brief   The maximum size, in bytes, lwrrently supported for all NNE_LAYER
 *          weights, in terms of bytes.
 */
#define LW2080_CTRL_NNE_NNE_LAYER_WEIGHTS_MAX_BYTES (0x10000U)

/*!
 * @brief   The maximum lwrrently supported number of NNE_LAYER weights, if they
 *          are all FP16s.
 */
#define LW2080_CTRL_NNE_NNE_LAYER_WEIGHTS_MAX_FP16  (0x8000U) /* finn: Evaluated from "(LW2080_CTRL_NNE_NNE_LAYER_WEIGHTS_MAX_BYTES / 2)" */

/*!
 * @brief   The maximum lwrrently supported number of NNE_LAYER weights, if they
 *          are all FP32s.
 */
#define LW2080_CTRL_NNE_NNE_LAYER_WEIGHTS_MAX_FP32  (0x4000U) /* finn: Evaluated from "(LW2080_CTRL_NNE_NNE_LAYER_WEIGHTS_MAX_BYTES / 4)" */

/*!
 * @brief   Structure describing NNE_LAYERS static information/POR.  Implements
 *          the BOARDOBJGRP model/interface.
 */
#define LW2080_CTRL_NNE_NNE_LAYERS_INFO_MESSAGE_ID (0x1U)

typedef struct LW2080_CTRL_NNE_NNE_LAYERS_INFO {
    /*!
     * @brief   LW2080_CTRL_BOARDOBJGRP_E255 super class.  Must always be first
     *          object in structure.
     */
    LW2080_CTRL_BOARDOBJGRP_E255   super;

    /*!
     * @brief   Number of valid fp16Weights
     */
    LwU16                          numFp16Weights;

    /*!
     * @brief   The FP16 weights for these NNE_LAYERs
     *
     * @note    These are pure binary representations of the 16-bit floating
     *          point numbers.
     */
    LwU16                          fp16Weights[LW2080_CTRL_NNE_NNE_LAYER_WEIGHTS_MAX_FP16];

    /*!
     * @brief   Number of valid fp32Weights
     */
    LwU16                          numFp32Weights;

    /*!
     * @brief   The FP32 weights for these NNE_LAYERs
     *
     * @note    These are pure binary representations of the 32-bit floating
     *          point numbers.
     */
    LwU32                          fp32Weights[LW2080_CTRL_NNE_NNE_LAYER_WEIGHTS_MAX_FP32];

    /*!
     * @brief   Array of NNE_LAYER structures.  Has valid indexes corresponding
     *          to the bits set in @ref super.objMask.
     */
    LW2080_CTRL_NNE_NNE_LAYER_INFO layers[LW2080_CTRL_BOARDOBJGRP_E255_MAX_OBJECTS];
} LW2080_CTRL_NNE_NNE_LAYERS_INFO;

/*!
 * LW2080_CTRL_CMD_NNE_NNE_LAYERS_GET_INFO
 *
 * This command returns NNE_LAYERS static object information/POR as specified
 * by the VBIOS in NNE Layers Table.
 *
 * See @ref LW2080_CTRL_NNE_NNE_LAYERS_INFO for documentation on the
 * parameters.
 *
 * Return values are specified per @ref BoardObjGrpGetInfo.
 */
#define LW2080_CTRL_CMD_NNE_NNE_LAYERS_GET_INFO (0x20803701) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_NNE_INTERFACE_ID << 8) | LW2080_CTRL_NNE_NNE_LAYERS_INFO_MESSAGE_ID" */

/* --------------------------- NNE Descriptor ------------------------------- */
/*!
 * @defgroup LW2080_CTRL_NNE_NNE_DESC_TYPE_ENUM
 *
 * Enumeration of NNE descriptor types.
 *
 * @{
 */
typedef LwU8 LW2080_CTRL_NNE_NNE_DESC_TYPE_ENUM;
#define LW2080_CTRL_NNE_NNE_DESC_TYPE_FC_10 0x00U
/*!@}*/

/*!
 * @brief   Structure describing @ref LW2080_CTRL_NNE_NNE_DESC_TYPE_FC_10
 *          static information/POR.
 */
typedef struct LW2080_CTRL_NNE_NNE_DESC_INFO_FC_10 {
    /*!
     * @brief   Reserved field because the FC_10 type has no type-specific
     *          information
     */
    LwU32 rsvd;
} LW2080_CTRL_NNE_NNE_DESC_INFO_FC_10;

/*!
 * @brief   NNE_DESC type-specific data union. Discriminated by NNE_DESC::super.type.
 */


/*!
 * @brief Maximum number of variable indices in the variable index array
 *
 * Determined by maximum param RAM size of the Deep Learning Coprocessor
 *
 * TODO: We temporarily reduce this value to 512 so that the variable index
 *       array will fill in the FBQ. Remove once super-surface chunk isn't
 *       entirely copied into the FBQ.
 */
#define LW2080_CTRL_NNE_NNE_DESC_NUM_VAR_IDX_MAX (512U)

/*
 * @defgroup LW2080_CTRL_NNE_NNE_DESC_OUTPUT_TYPE_ENUM
 *
 * Enumeration of NNE neural net output types.
* @{
 */
typedef LwU8 LW2080_CTRL_NNE_NNE_DESC_OUTPUT_TYPE_ENUM;
#define LW2080_CTRL_NNE_NNE_DESC_OUTPUT_TYPE_DISABLED     0x00U
#define LW2080_CTRL_NNE_NNE_DESC_OUTPUT_TYPE_POWER_DN     0x01U
#define LW2080_CTRL_NNE_NNE_DESC_OUTPUT_TYPE_POWER_TOTAL  0x02U
#define LW2080_CTRL_NNE_NNE_DESC_OUTPUT_TYPE_ENERGY_DN    0x03U
#define LW2080_CTRL_NNE_NNE_DESC_OUTPUT_TYPE_ENERGY_TOTAL 0x04U
#define LW2080_CTRL_NNE_NNE_DESC_OUTPUT_TYPE_PERF         0x05U
#define LW2080_CTRL_NNE_NNE_DESC_OUTPUT_TYPE_POWER        0x06U  // Deprecated will be removed in follow-up, aliasing to _POWER_DN for now
/*!@}*/

/*!
 * @brief Class holding tuple of data that uniquely identifies a NNE_DEESC_OUTPUT_POWER_DN
 */
typedef struct LW2080_CTRL_NNE_NNE_DESC_OUTPUT_ID_POWER_DN {
    /*!
     * @brief Index of the VOLT_RAIL index that this output is estimating.
     */
    LwBoardObjIdx voltRailIdx;

    /*!
     * @brief Voltage normalization value for the dynamic "normalized"
     * power.  NNE will normalize the power from this voltage value to 1.0V
     * output for the client to consume.
     */
    LwU32         voltageuV;
} LW2080_CTRL_NNE_NNE_DESC_OUTPUT_ID_POWER_DN;

/*!
 * @brief Structure holding data for inferred dynamic-normalized power of a rail.
 */
typedef struct LW2080_CTRL_NNE_NNE_DESC_OUTPUT_POWER_DN {
    /*!
     * @brief Tuple that uniquely identifies an inferred power output of a neural-net.
     */
    LW2080_CTRL_NNE_NNE_DESC_OUTPUT_ID_POWER_DN id;

    /*!
     * @brief Power inferred by a neural-net, in mW.
     */
    LwU32                                       powermW;
} LW2080_CTRL_NNE_NNE_DESC_OUTPUT_POWER_DN;
typedef struct LW2080_CTRL_NNE_NNE_DESC_OUTPUT_POWER_DN *PLW2080_CTRL_NNE_NNE_DESC_OUTPUT_POWER_DN;

/*!
 * @defgroup LW2080_CTRL_NNE_DESC_OUTPUT_ID_VOLT_RAIL_NAME
 *
 * Voltage/power rail "names" for the NNE_DESC_OUTPUT classes
 * (_POWER_TOTAL and _ENERGY_TOTAL).  These are used to describe rails
 * which aren't properly defined/described within RM/PMU PERF/PWR
 * infrastructure.  For now, this is really just FBVDD.
 *
 * @{
 */
typedef LwU8 LW2080_CTRL_NNE_NNE_DESC_OUTPUT_ID_VOLT_RAIL_NAME_ENUM;
typedef LwU8 *PLW2080_CTRL_NNE_NNE_DESC_OUTPUT_ID_VOLT_RAIL_NAME_ENUM;
#define LW2080_CTRL_NNE_NNE_DESC_OUTPUT_ID_VOLT_RAIL_NAME_FBVDD 0x00U
/*! @} */

/*!
 * @brief Class holding tuple of data that uniquely identifies a NNE_DEESC_OUTPUT_POWER_TOTAL
 */
typedef struct LW2080_CTRL_NNE_NNE_DESC_OUTPUT_ID_POWER_TOTAL {
    /*!
     * @brief "Name" of the VOLT_RAIL that this output is estimating.
     * Specified as @ref
     * LW2080_CTRL_NNE_NNE_DESC_OUTPUT_ID_VOLT_RAIL_NAME.
     */
    LwU8 voltRailName;
} LW2080_CTRL_NNE_NNE_DESC_OUTPUT_ID_POWER_TOTAL;

/*!
 * @brief Structure holding data for total inferred power of a rail.
 */
typedef struct LW2080_CTRL_NNE_NNE_DESC_OUTPUT_POWER_TOTAL {
    /*!
     * @brief Tuple that uniquely identifies an inferred power output of a neural-net.
     */
    LW2080_CTRL_NNE_NNE_DESC_OUTPUT_ID_POWER_TOTAL id;

    /*!
     * @brief Power inferred by a neural-net, in mW.
     */
    LwU32                                          powermW;
} LW2080_CTRL_NNE_NNE_DESC_OUTPUT_POWER_TOTAL;
typedef struct LW2080_CTRL_NNE_NNE_DESC_OUTPUT_POWER_TOTAL *PLW2080_CTRL_NNE_NNE_DESC_OUTPUT_POWER_TOTAL;

/*!
 * @brief Class holding tuple of data that uniquely identifies a NNE_DEESC_OUTPUT_ENERGY_DN
 */
typedef struct LW2080_CTRL_NNE_NNE_DESC_OUTPUT_ID_ENERGY_DN {
    /*!
     * @brief Index of the VOLT_RAIL index that this output is estimating.
     */
    LwBoardObjIdx voltRailIdx;

    /*!
     * @brief Voltage normalization value for the dynamic "normalized"
     * energy.  NNE will normalize the energy this voltage value to 1.0V
     * output for the client to consume.
     */
    LwU32         voltageuV;
} LW2080_CTRL_NNE_NNE_DESC_OUTPUT_ID_ENERGY_DN;

/*!
 * @brief Structure holding data for the inferred dynamic-normalized energy of a rail.
 */
typedef struct LW2080_CTRL_NNE_NNE_DESC_OUTPUT_ENERGY_DN {
    /*!
     * @brief Tuple that uniquely identifies an inferred energy output of a neural-net.
     */
    LW2080_CTRL_NNE_NNE_DESC_OUTPUT_ID_ENERGY_DN id;

    /*!
     * @brief Energy inferred by a neural-net, in mJ.
     */
    LwU32                                        energymJ;
} LW2080_CTRL_NNE_NNE_DESC_OUTPUT_ENERGY_DN;
typedef struct LW2080_CTRL_NNE_NNE_DESC_OUTPUT_ENERGY_DN *PLW2080_CTRL_NNE_NNE_DESC_OUTPUT_ENERGY_DN;

/*!
 * @brief Class holding tuple of data that uniquely identifies a NNE_DEESC_OUTPUT_ENERGY_TOTAL
 */
typedef struct LW2080_CTRL_NNE_NNE_DESC_OUTPUT_ID_ENERGY_TOTAL {
    /*!
     * @brief "Name" of the VOLT_RAIL that this output is estimating.
     * Specified as @ref
     * LW2080_CTRL_NNE_NNE_DESC_OUTPUT_ID_VOLT_RAIL_NAME.
     */
    LwU8 voltRailName;
} LW2080_CTRL_NNE_NNE_DESC_OUTPUT_ID_ENERGY_TOTAL;

/*!
 * @brief Structure holding data for inferred total energy.
 */
typedef struct LW2080_CTRL_NNE_NNE_DESC_OUTPUT_ENERGY_TOTAL {
    /*!
     * @brief Tuple that uniquely identifies an inferred energy output of a neural-net.
     */
    LW2080_CTRL_NNE_NNE_DESC_OUTPUT_ID_ENERGY_TOTAL id;

    /*!
     * @brief Energy inferred by a neural-net, in mJ.
     */
    LwU32                                           energymJ;
} LW2080_CTRL_NNE_NNE_DESC_OUTPUT_ENERGY_TOTAL;
typedef struct LW2080_CTRL_NNE_NNE_DESC_OUTPUT_ENERGY_TOTAL *PLW2080_CTRL_NNE_NNE_DESC_OUTPUT_ENERGY_TOTAL;

/*!
 * @brief Class holding tuple of data that uniquely identifies a power output
 *
 * @note Deprecated.   Will be removed in a followup CL.
 */
typedef struct LW2080_CTRL_NNE_NNE_DESC_OUTPUT_ID_POWER {
    /*!
     * @brief Index of the PWR_CHANNEL that this output is estimating
     */
    LwBoardObjIdx pwrChIdx;
} LW2080_CTRL_NNE_NNE_DESC_OUTPUT_ID_POWER;

/*!
 * @brief Structure holding the power inferred by a neural-net for a particular rail.
 *
 * @note Deprecated.   Will be removed in a followup CL.
 */
typedef struct LW2080_CTRL_NNE_NNE_DESC_OUTPUT_POWER {
    /*!
     * @brief Tuple that uniquely identifies an inferred power output of a neural-net.
     */
    LW2080_CTRL_NNE_NNE_DESC_OUTPUT_ID_POWER powerId;

    /*!
     * @brief Power inferred by a neural-net, in mW.
     */
    LwU32                                    powerMw;
} LW2080_CTRL_NNE_NNE_DESC_OUTPUT_POWER;
typedef struct LW2080_CTRL_NNE_NNE_DESC_OUTPUT_POWER *PLW2080_CTRL_NNE_NNE_DESC_OUTPUT_POWER;

/*!
 * @brief Class holding tuple of data that uniquely identifies a perf output
 */
typedef struct LW2080_CTRL_NNE_NNE_DESC_OUTPUT_ID_PERF {
    /*!
     * @brief Reserved byte for suppressing compiler warnings
     *
     * No additional data to uniquely identify a perf output. Add a rerserved byte to
     * suppress compiler warning about having empty struct.
     */
    LwU8 rsvd;
} LW2080_CTRL_NNE_NNE_DESC_OUTPUT_ID_PERF;

/*!
 * Structure specifying the output value of a NNE_DESC_OUTPUT of type
 */
typedef struct LW2080_CTRL_NNE_NNE_DESC_OUTPUT_PERF {
    /*!
     * @brief Tuple that uniquely identifies an inferred perf output of a neural-net.
     */
    LW2080_CTRL_NNE_NNE_DESC_OUTPUT_ID_PERF perfId;

    /*!
     * @brief Unitless value representing relative perf, inferred by a neural-net.
     */
    LwUFXP20_12                             perf;
} LW2080_CTRL_NNE_NNE_DESC_OUTPUT_PERF;
typedef struct LW2080_CTRL_NNE_NNE_DESC_OUTPUT_PERF *PLW2080_CTRL_NNE_NNE_DESC_OUTPUT_PERF;

/*!
 * @brief Union describing NNE_DESC_OUTPUT_ID's
 */
typedef struct LW2080_CTRL_NNE_NNE_DESC_OUTPUT_ID {
    /*!
     * @brief   Enum for what kind of output this is.
     *
     * @ref LW2080_CTRL_NNE_NNE_DESC_OUTPUT_TYPE_ENUM
     */
    LW2080_CTRL_NNE_NNE_DESC_OUTPUT_TYPE_ENUM type;

    /*!
     * @brief   Output-type specific unique identifier data.
     */
    union {
        LW2080_CTRL_NNE_NNE_DESC_OUTPUT_ID_POWER_DN     powerDN;
        LW2080_CTRL_NNE_NNE_DESC_OUTPUT_ID_POWER_TOTAL  powerTotal;
        LW2080_CTRL_NNE_NNE_DESC_OUTPUT_ID_ENERGY_DN    energyDN;
        LW2080_CTRL_NNE_NNE_DESC_OUTPUT_ID_ENERGY_TOTAL energyTotal;
        LW2080_CTRL_NNE_NNE_DESC_OUTPUT_ID_POWER        power;
        LW2080_CTRL_NNE_NNE_DESC_OUTPUT_ID_PERF         perf;
    } typeSpecificId;
} LW2080_CTRL_NNE_NNE_DESC_OUTPUT_ID;
typedef struct LW2080_CTRL_NNE_NNE_DESC_OUTPUT_ID *PLW2080_CTRL_NNE_NNE_DESC_OUTPUT_ID;

/*!
 * Union of all NNE_DESC_OUTPUT type-specific values.
 */


/*!
 * Structure specifying NNE_DESC output from a single loop of NNE_DESC inference.
 */
typedef struct LW2080_CTRL_NNE_NNE_DESC_OUTPUT {
    /*!
     * @brief   XAPI does not support using an element within another structure
     *          as a discriminant.  Placing this redundant type value here until
     *          that design constraint can be fixed.
     */
    LwU8 type;

    /*!
     * @brief   Type-specific data union.
     */
    union {
        LW2080_CTRL_NNE_NNE_DESC_OUTPUT_POWER_DN     powerDN;
        LW2080_CTRL_NNE_NNE_DESC_OUTPUT_POWER_TOTAL  powerTotal;
        LW2080_CTRL_NNE_NNE_DESC_OUTPUT_ENERGY_DN    energyDN;
        LW2080_CTRL_NNE_NNE_DESC_OUTPUT_ENERGY_TOTAL energyTotal;
        LW2080_CTRL_NNE_NNE_DESC_OUTPUT_POWER        power;
        LW2080_CTRL_NNE_NNE_DESC_OUTPUT_PERF         perf;
    } data;
} LW2080_CTRL_NNE_NNE_DESC_OUTPUT;
typedef struct LW2080_CTRL_NNE_NNE_DESC_OUTPUT *PLW2080_CTRL_NNE_NNE_DESC_OUTPUT;

#define LW2080_CTRL_NNE_NNE_DESC_OUTPUTS_MAX        (8U)

// Temporary alias to unbreak LWAPI
#define LW2080_CTRL_NNE_NNE_DESC_OUTPUT_MAX_OUTPUTS LW2080_CTRL_NNE_NNE_DESC_OUTPUTS_MAX

/*
 * @brief   Enum type that defines the normalization applied on the input
 */
typedef LwU8 LW2080_CTRL_NNE_NNE_DESC_INPUT_NORM_MODE_ENUM;
#define LW2080_CTRL_NNE_NNE_DESC_INPUT_NORM_MODE_SECONDARY          0U
#define LW2080_CTRL_NNE_NNE_DESC_INPUT_NORM_MODE_SATURATION         1U
#define LW2080_CTRL_NNE_NNE_DESC_INPUT_NORM_MODE_POISON             2U
#define LW2080_CTRL_NNE_NNE_DESC_INPUT_NORM_MODE_NUM_MODES          3U
#define LW2080_CTRL_NNE_NNE_DESC_INPUT_NORM_MODE_DEFAULT            (LW_U8_MAX)


#define LW2080_CTRL_NNE_NNE_DESC_INPUT_NORM_MODE_THRESHOLD_DISABLED (LW_U16_MAX)

/*
 * @brief   Structure holding all input normalization per call
 */
typedef struct LW2080_CTRL_NNE_NNE_DESC_INPUT_NORM_STATUS {
    /*!
     * @brief   Mask of inputs that violated their maximum norm value. Indexed by
     *          varIdx
     */
    LW2080_CTRL_BOARDOBJGRP_MASK_E512             violatiolwarMask;

    /*!
     * @brief   Number of inputs that violated their maximum norm value.
     */
    LwU16                                         violationCount;

    /*!
     * @brief   Field holding the _NNE_DESC_INPUT_NORM_MODE used when applying
     *          normalization to the inputs
     */
    LW2080_CTRL_NNE_NNE_DESC_INPUT_NORM_MODE_ENUM appliedMode;
} LW2080_CTRL_NNE_NNE_DESC_INPUT_NORM_STATUS;

/*!
 * @brief   Macro to reset the fields that get updated every time input
 *          normalization is requested.
 */
#define LW2080_CTRL_NNE_NNE_DESC_INPUT_NORM_STATUS_RESET(pInputNormStatus)                    \
    do {                                                                                      \
        LW2080_CTRL_BOARDOBJGRP_MASK_E512_INIT(&(pInputNormStatus->violatiolwarMask.super));  \
        pInputNormStatus->violationCount = 0;                                                 \
        pInputNormStatus->appliedMode = LW2080_CTRL_NNE_NNE_DESC_INPUT_NORM_MODE_DEFAULT;     \
    } while (LW_FALSE)

/*
 * @brief   Structure holding all static input normalization information
 */
typedef struct LW2080_CTRL_NNE_NNE_DESC_INPUT_NORM_INFO {

    /*!
     * @brief   Array of LwU16 that holds the minimum threshold for triggering
     *          each of the _INPUT_NORM_MODES. Array is indexed by the _INPUT_NORM_MODE
     *          enum values
     */
    LwU16 modeThresholds[LW2080_CTRL_NNE_NNE_DESC_INPUT_NORM_MODE_NUM_MODES];
} LW2080_CTRL_NNE_NNE_DESC_INPUT_NORM_INFO;

/*
 * @brief   Structure describing NNE_DESC static information/POR. Implements the
 *          BOARDOBJ model/interface.
 */
typedef struct LW2080_CTRL_NNE_NNE_DESC_INFO {
    /*!
     * @brief   LW2080_CTRL_BOARDOBJ super class. Must always be first object in
     *          structure.
     */
    LW2080_CTRL_BOARDOBJ                     super;

    /*!
     * @brief   XAPI does not support using an element within another structure
     *          as a discriminant.  Placing this redundant type value here until
     *          that design constraint can be fixed.
     */
    LwU8                                     type;

    /*!
     * @brief   The first output layer in the neural net
     */
    LwBoardObjIdx                            firstLayerIdx;

    /*!
     * @brief   The last output layer in the neural net
     */
    LwBoardObjIdx                            lastLayerIdx;

    /*!
     * @brief   Index of the first variable in a continuous range in
     *          @ref LW2080_CTRL_NNE_NNE_DESCS_INFO::varIdxArray
     */
    LwU16                                    firstVarIdx;

    /*!
     * @brief   Index of the last variable in a continuous range in
     *          @ref LW2080_CTRL_NNE_NNE_DESCS_INFO::varIdxArray
     */
    LwU16                                    lastVarIdx;

    /*!
     * @brief   Number of layers in this NNE_DESC.
     */
    LwU16                                    numLayers;

    /*!
     * @brief   Maximum number of inputs from any hidden layer, or the output layer.
     *
     * @deprecated  To be removed once removed from LwAPI.
     *
     * @protected
     */
    LwU16                                    maxLayerNumInputs;

    /*!
     * @brief   Maximum number of outputs from any intermediate (input or
     *          hidden) layer.
     *
     * @protected
     */
    LwU16                                    maxInterLayerNumOutputs;

    /*!
     * Version of network which this NNE_DESC object represents.
     *
     * This is a purely diagnostic value to track across potential
     * multiple versions of a network.  This value can be populated in
     * whatever manner the network authors prefer.
     */
    LwU8                                     networkVersion;

    /*!
     * @brief   Number of outputs this neural-net produces and number of valid
     *          entries in outputs array.
     */
    LwU8                                     numOutputs;

    /*!
     *  @brief  Array of types for outputs for this NNE_DESC
     */
    LW2080_CTRL_NNE_NNE_DESC_OUTPUT_ID       outputs[LW2080_CTRL_NNE_NNE_DESC_OUTPUTS_MAX];

    /*!
     * @brief Structure holding all relevant data relating to input normalization
     */
    LW2080_CTRL_NNE_NNE_DESC_INPUT_NORM_INFO inputNormInfo;

    union {
        LW2080_CTRL_NNE_NNE_DESC_INFO_FC_10 fc10;
    } data;
} LW2080_CTRL_NNE_NNE_DESC_INFO;

/*!
 * @brief   Structure describing NNE_DESCS static information/POR.  Implements
 *          the BOARDOBJGRP model/interface.
 */
#define LW2080_CTRL_NNE_NNE_DESCS_INFO_MESSAGE_ID (0x2U)

typedef struct LW2080_CTRL_NNE_NNE_DESCS_INFO {
    /*!
     * @brief   LW2080_CTRL_BOARDOBJGRP_E32 super class.  Must always be first
     *          object in structure.
     */
    LW2080_CTRL_BOARDOBJGRP_E32   super;

    /*!
     * @brief   Number of valid elements in varIdxArray
     */
    LwU16                         numVarIdx;

    /*!
     * @brief   Array containing indices pointing at NNE_VARs.
     */
    LwBoardObjIdx                 varIdxArray[LW2080_CTRL_NNE_NNE_DESC_NUM_VAR_IDX_MAX];

    /*!
     * @brief   Array of NNE_DESC structures.  Has valid indexes corresponding
     *          to the bits set in @ref super.objMask.
     */
    LW2080_CTRL_NNE_NNE_DESC_INFO descs[LW2080_CTRL_BOARDOBJGRP_E32_MAX_OBJECTS];
} LW2080_CTRL_NNE_NNE_DESCS_INFO;

/*!
 * LW2080_CTRL_CMD_NNE_DESCS_GET_INFO
 *
 * This command returns NNE_DESCS static object information/POR as specified
 * by the VBIOS in NNE Descriptors Table.
 *
 * See @ref LW2080_CTRL_NNE_NNE_DESCS_INFO for documentation on the
 * parameters.
 *
 * Return values are specified per @ref BoardObjGrpGetInfo.
 */
#define LW2080_CTRL_CMD_NNE_NNE_DESCS_GET_INFO (0x20803702) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_NNE_INTERFACE_ID << 8) | LW2080_CTRL_NNE_NNE_DESCS_INFO_MESSAGE_ID" */

/*!
 * @brief   The maximum number of loop-independent input variables allowed.
 */
#define LW2080_CTRL_NNE_NNE_VAR_MAX            (512)

/*!
 * @defgroup LW2080_CTRL_NNE_NNE_DESC_INFERENCE_PROFILING_REGION_ENUM
 *
 * Enumeration of NNE inference profiling regions.
 * @{
 */
typedef LwU8 LW2080_CTRL_NNE_NNE_DESC_INFERENCE_PROFILING_REGION_ENUM;
#define LW2080_CTRL_NNE_NNE_DESC_INFERENCE_PROFILING_REGION_TOTAL_TIME                    (0)
#define LW2080_CTRL_NNE_NNE_DESC_INFERENCE_PROFILING_REGION_INPUT_LOAD                    (1)
#define LW2080_CTRL_NNE_NNE_DESC_INFERENCE_PROFILING_REGION_SWZL_GENERATION               (2)
#define LW2080_CTRL_NNE_NNE_DESC_INFERENCE_PROFILING_REGION_DESC_LOAD                     (3)
#define LW2080_CTRL_NNE_NNE_DESC_INFERENCE_PROFILING_REGION_DLC_EVAL                      (4)
#define LW2080_CTRL_NNE_NNE_DESC_INFERENCE_PROFILING_REGION_PARM_RAM_MULTI_INFERENCE_LOAD (5)
#define LW2080_CTRL_NNE_NNE_DESC_INFERENCE_PROFILING_REGION_PARM_RAM_INFERENCE_LOAD       (6)
#define LW2080_CTRL_NNE_NNE_DESC_INFERENCE_PROFILING_REGION_DESC_INFERENCE_CONFIG         (7)
#define LW2080_CTRL_NNE_NNE_DESC_INFERENCE_PROFILING_REGION_MAX                           (8)
/*!@}*/

typedef struct LW2080_CTRL_NNE_NNE_DESC_INFERENCE_PROFILING {
    /*!
     * Array of of all NNE profiled elapsed times.
     *
     * @note    Indexed using
     *          @ref LW2080_CTRL_NNE_NNE_DESC_INFERENCE_PROFILING_REGION_ENUM.
     */
    LwU64_ALIGN32 elapsedTimes[LW2080_CTRL_NNE_NNE_DESC_INFERENCE_PROFILING_REGION_MAX];

    /*!
     * Whether the NNE_DESC for this inference was already loaded or not.
     */
    LwBool        bDescriptorLoaded;

    /*!
     * Whether the multi-inference parameter RAM allocations remained valid from
     * a prior inference.
     */
    LwBool        bMultiInferenceAllocValid;
} LW2080_CTRL_NNE_NNE_DESC_INFERENCE_PROFILING;

/*!
 * @defgroup LW2080_CTRL_NNE_NNE_DESC_INFERENCE_ROUNDING_MODE_ENUM
 *
 * These enumerated types should match the rounding modes provided by HW specified at
 * LW_CPWR_THERM_DLPPE_CFG_ROUNDING_MODE. See refmans for definition of each mode.
 *
 * @{
 */
typedef LwU8 LW2080_CTRL_NNE_NNE_DESC_INFERENCE_ROUNDING_MODE_ENUM;
#define LW2080_CTRL_NNE_NNE_DESC_INFERENCE_ROUNDING_MODE_TO_NEAREST_EVEN           (0)
#define LW2080_CTRL_NNE_NNE_DESC_INFERENCE_ROUNDING_MODE_TOWARDS_ZERO              (1)
#define LW2080_CTRL_NNE_NNE_DESC_INFERENCE_ROUNDING_MODE_TOWARDS_POSITIVE_INFINITY (2)
#define LW2080_CTRL_NNE_NNE_DESC_INFERENCE_ROUNDING_MODE_TOWARDS_NEGATIVE_INFINITY (3)
#define LW2080_CTRL_NNE_NNE_DESC_INFERENCE_ROUNDING_MODE_TO_NEAREST_UP             (4)
#define LW2080_CTRL_NNE_NNE_DESC_INFERENCE_ROUNDING_MODE_AWAY_FROM_ZERO            (5)
/*!@}*/

/*!
 * @brief Structure holding inference invocation settings.
 */
typedef struct LW2080_CTRL_NNE_NNE_DESC_INFERENCE_CONFIG {
    /*!
     * @brief [IN]   Rounding mode for the requested inference.
     *
     * @ref LW2080_CTRL_NNE_NNE_DESC_INFERENCE_ROUNDING_MODE_ENUM
     */
    LW2080_CTRL_NNE_NNE_DESC_INFERENCE_ROUNDING_MODE_ENUM roundingMode;

    /*!
     * @brief [IN]   Boolean to enable clamping of initity values.
     */
    LwBool                                                bClampInfinityToNormal;
} LW2080_CTRL_NNE_NNE_DESC_INFERENCE_CONFIG;
typedef struct LW2080_CTRL_NNE_NNE_DESC_INFERENCE_CONFIG *PLW2080_CTRL_NNE_NNE_DESC_INFERENCE_CONFIG;

/*!
 * @defgroup    LW2080_CTRL_NNE_NNE_DESC_INFERENCE_FP_EXCEPTION_FLAGS
 *
 * @brief   Bitvector of floating point exceptions that oclwrred during
 *          inference
 *
 * @details
 *
 *      LW2080_CTRL_NNE_NNE_DESC_INFERENCE_FP_EXCEPTION_NONE
 *          No exceptions.
 *
 *      LW2080_CTRL_NNE_NNE_DESC_INFERENCE_FP_EXCEPTION_ILWALID
 *          Detected when multiplying zero times infinity or when subtracting
 *          infinity from infinity
 *
 *      LW2080_CTRL_NNE_NNE_DESC_INFERENCE_FP_EXCEPTION_UNDERFLOW
 *          Detected when a result is subnormal
 *
 *      LW2080_CTRL_NNE_NNE_DESC_INFERENCE_FP_EXCEPTION_OVERFLOW
 *          Detected when a result is +/- infinity (note that overflow detection
 *          is prior to any clamping by
 *          @ref LW2080_CTRL_NNE_NNE_DESC_INFERENCE_CONFIG::bClampInfinityToNormal)
 *
 *      LW2080_CTRL_NNE_NNE_DESC_INFERENCE_FP_EXCEPTION_INEXACT
 *          Detected when a rounded result is not exactly equal to the infinite
 *          precise value before rounding
 * @{
 */
typedef LwU32 LW2080_CTRL_NNE_NNE_DESC_INFERENCE_FP_EXCEPTION_FLAGS;
#define LW2080_CTRL_NNE_NNE_DESC_INFERENCE_FP_EXCEPTION_NONE                  0x00000000U
#define LW2080_CTRL_NNE_NNE_DESC_INFERENCE_FP_EXCEPTION_ILWALID                 0U:0U
#define LW2080_CTRL_NNE_NNE_DESC_INFERENCE_FP_EXCEPTION_ILWALID_NOT_PENDING   0x00000000U
#define LW2080_CTRL_NNE_NNE_DESC_INFERENCE_FP_EXCEPTION_ILWALID_PENDING       0x00000001U
#define LW2080_CTRL_NNE_NNE_DESC_INFERENCE_FP_EXCEPTION_UNDERFLOW               1U:1U
#define LW2080_CTRL_NNE_NNE_DESC_INFERENCE_FP_EXCEPTION_UNDERFLOW_NOT_PENDING 0x00000000U
#define LW2080_CTRL_NNE_NNE_DESC_INFERENCE_FP_EXCEPTION_UNDERFLOW_PENDING     0x00000001U
#define LW2080_CTRL_NNE_NNE_DESC_INFERENCE_FP_EXCEPTION_OVERFLOW                2U:2U
#define LW2080_CTRL_NNE_NNE_DESC_INFERENCE_FP_EXCEPTION_OVERFLOW_NOT_PENDING  0x00000000U
#define LW2080_CTRL_NNE_NNE_DESC_INFERENCE_FP_EXCEPTION_OVERFLOW_PENDING      0x00000001U
#define LW2080_CTRL_NNE_NNE_DESC_INFERENCE_FP_EXCEPTION_INEXACT                 3U:3U
#define LW2080_CTRL_NNE_NNE_DESC_INFERENCE_FP_EXCEPTION_INEXACT_NOT_PENDING   0x00000000U
#define LW2080_CTRL_NNE_NNE_DESC_INFERENCE_FP_EXCEPTION_INEXACT_PENDING       0x00000001U
/*!@}*/

/*!
 * @brief Structure specifying all the loop-specific inputs and outputs for a
 *        NNE_DESC inference invocation.
 */
typedef struct LW2080_CTRL_NNE_NNE_DESC_INFERENCE_LOOP {
    /*!
     * @brief [IN] Array of loop-specific input values.
     *
     * These input values are for a single iteration of a NNE_DESC inference loop.
     */
    LW2080_CTRL_NNE_NNE_VAR_INPUT                         varInputs[LW2080_CTRL_NNE_NNE_VAR_MAX];

    /*!
     * @brief [OUT] Exceptions that oclwrred during exelwtion of the loop.
     */
    LW2080_CTRL_NNE_NNE_DESC_INFERENCE_FP_EXCEPTION_FLAGS exceptions;

    /*!
     * @brief [OUT] Array of outputs for a single iteration of the inference loop.
     */
    LW2080_CTRL_NNE_NNE_DESC_OUTPUT                       descOutputs[LW2080_CTRL_NNE_NNE_DESC_OUTPUTS_MAX];
} LW2080_CTRL_NNE_NNE_DESC_INFERENCE_LOOP;
typedef struct LW2080_CTRL_NNE_NNE_DESC_INFERENCE_LOOP *PLW2080_CTRL_NNE_NNE_DESC_INFERENCE_LOOP;

/*!
 * @brief   Type to be used for sequence numbers in parameter RAM allocations.
 */
typedef LwU32 LW2080_CTRL_NNE_NNE_DESC_INFERENCE_PARM_RAM_SEQ_NUM;

/*!
 * @brief   Initial value to be used for the parameter RAM sequence number.
 */
#define LW2080_CTRL_NNE_NNE_DESC_INFERENCE_PARM_RAM_SEQ_NUM_INIT    (0U)

/*!
 * @brief   Indicates that the sequence number is no longer for a valid
 *          allocation.
 */
#define LW2080_CTRL_NNE_NNE_DESC_INFERENCE_PARM_RAM_SEQ_NUM_ILWALID (LW_U32_MAX)

/*!
 * @brief   Implements increment for
 *          @ref LW2080_CTRL_NNE_NNE_DESC_INFERENCE_PARM_RAM_SEQ_NUM
 *
 * @details Necessary to avoid a
 *          @ref LW2080_CTRL_NNE_NNE_DESC_INFERENCE_PARM_RAM_SEQ_NUM variable
 *          from taking on
 *          @ref LW2080_CTRL_NNE_NNE_DESC_INFERENCE_PARM_RAM_SEQ_NUM_ILWALID
 *
 * @param[in,out]   pSeqNum Sequence number to increment.
 */
#define LW2080_CTRL_NNE_NNE_DESC_INFERENCE_PARM_RAM_SEQ_NUM_INCREMENT(pSeqNum) \
    do                                                                         \
    {                                                                          \
        (*(pSeqNum))++;                                                        \
        if (*(pSeqNum) ==                                                      \
                LW2080_CTRL_NNE_NNE_DESC_INFERENCE_PARM_RAM_SEQ_NUM_ILWALID)   \
        {                                                                      \
            (*(pSeqNum))++;                                                    \
        }                                                                      \
    } while (LW_FALSE)

/*!
 * @brief   Contains information about static inference inputs used for caching.
 */
typedef struct LW2080_CTRL_NNE_NNE_DESC_INFERENCE_STATIC_VAR_CACHE {
    /*!
     * @brief   Sequence number assigned by NNE to track which data are
     *          lwrrently cached.
     */
    LW2080_CTRL_NNE_NNE_DESC_INFERENCE_PARM_RAM_SEQ_NUM parmRamSeqNum;

    /*!
     * @brief   The offset into parameter RAM for the allocation for which
     *          @ref LW2080_CTRL_NNE_NNE_DESC_INFERENCE_STATIC_VAR_CACHE::parmRamSeqNum
     *          was made.
     */
    LwU32                                               parmRamOffset;
} LW2080_CTRL_NNE_NNE_DESC_INFERENCE_STATIC_VAR_CACHE;

/*!
 * @brief   Ilwalidates the cache information in a
 *          @ref LW2080_CTRL_NNE_NNE_DESC_INFERENCE_STATIC_VAR_CACHE structure.
 *
 * @param[out]  pCache  The cache structure to ilwalidate.
 */
#define LW2080_CTRL_NNE_NNE_DESC_INFERENCE_STATIC_VAR_CACHE_ILWALIDATE(pCache) \
    do                                                                         \
    {                                                                          \
        (pCache)->parmRamSeqNum =                                              \
            LW2080_CTRL_NNE_NNE_DESC_INFERENCE_PARM_RAM_SEQ_NUM_ILWALID;       \
    } while (LW_FALSE)

/*!
 * @brief Structure holding all metadata describing the inputs/outputs of
 *        for an inference.
 */
typedef struct LW2080_CTRL_NNE_NNE_DESC_INFERENCE_HEADER {
    /*!
     * @brief [IN] Inference invocation specific configuration settings.
     */
    LW2080_CTRL_NNE_NNE_DESC_INFERENCE_CONFIG           inferenceCfg;

    /*!
     * @brief [IN] Index of NNE_DESC for which to run given inference operation.
     */
    LwBoardObjIdx                                       descIdx;

    /*!
     * @brief [OUT] Number of outputs per iteration of the inference loop of the NNE_DESC.
     */
    LwU8                                                descOutputCnt;

    /*!
     * @brief [IN] Number of loops for which to run evaluation.
     * Must be in range [1, @ref LW2080_CTRL_NNE_NNE_DESC_INFERENCE_LOOPS_MAX).
     */
    LwU8                                                loopCnt;

    /*!
     * @brief [IN] Number of inputs that are constant across inference loops.
     */
    LwU16                                               varInputCntStatic;

    /*!
     * @brief [IN] Number of inputs that change every inference iteration.
     *
     * If this value is >0, then @ref loopCnt must be > 1.
     */
    LwU16                                               varInputCntLoop;

    /*!
     * @brief [IN/OUT] Used to help with caching the static inputs to the
     *                 inference to improve performance.
     *
     * @note    This is used only for PMU-internal clients and ignored for any
     *          other clients.
     */
    LW2080_CTRL_NNE_NNE_DESC_INFERENCE_STATIC_VAR_CACHE staticVarCache;

    /*!
     * @brief [OUT] Array of of all NNE profiled elapsed times.
     *
     * Indexed using @ref LW2080_CTRL_NNE_NNE_DESC_INFERENCE_PROFILING_REGION_ENUM.
     *
     * @deprecated  In favor of
     *              @ref LW2080_CTRL_NNE_NNE_DESC_INFERENCE_HEADER::profiling
     */
    LwU64_ALIGN32                                       elapsedTimes[LW2080_CTRL_NNE_NNE_DESC_INFERENCE_PROFILING_REGION_MAX];

    /*!
     * [OUT] Profiling data for this inference.
     */
    LW2080_CTRL_NNE_NNE_DESC_INFERENCE_PROFILING        profiling;

    /*!
     * @brief [OUT] Structure holding all relevant data relating to input normalization for each
     *        normalization call
     */
    LW2080_CTRL_NNE_NNE_DESC_INPUT_NORM_STATUS          inputNormStatus;
} LW2080_CTRL_NNE_NNE_DESC_INFERENCE_HEADER;
typedef struct LW2080_CTRL_NNE_NNE_DESC_INFERENCE_HEADER *PLW2080_CTRL_NNE_NNE_DESC_INFERENCE_HEADER;

/*!
 * @brief Max number of shmoo loops supported in a single invocation of NNE
 */
#define LW2080_CTRL_NNE_NNE_DESC_INFERENCE_LOOPS_MAX (8)

/*!
 * @brief Structure specifying all inputs/outputs of an inference of an NNE_DESC.
 *
 * A single inference invocation on a NNE_DESC object may specify several "loops"
 * of inference evaluation substituting a subset of the inputs on a per loop basis.
 */
#define LW2080_CTRL_NNE_NNE_DESC_INFERENCE_PARAMS_MESSAGE_ID (0x3U)

typedef struct LW2080_CTRL_NNE_NNE_DESC_INFERENCE_PARAMS {
    /*!
     * @brief [IN] Header containing all metadata.
     *
     * @copydoc LW2080_CTRL_NNE_NNE_DESC_INFERENCE_HEADER
     */
    LW2080_CTRL_NNE_NNE_DESC_INFERENCE_HEADER hdr;

    /*!
     * @brief [IN] Array of input values that are constant across all inference loops.
     *
     * Values are valid between [0, @ref nneVarCntStatic).
     */
    LW2080_CTRL_NNE_NNE_VAR_INPUT             varInputsStatic[LW2080_CTRL_NNE_NNE_VAR_MAX];

    /*!
     * @brief [IN/OUT] Array of inference loop-specific input/output data.
     *
     * Values are valid between [0, @ref loopCnt).
     */
    LW2080_CTRL_NNE_NNE_DESC_INFERENCE_LOOP   loops[LW2080_CTRL_NNE_NNE_DESC_INFERENCE_LOOPS_MAX];

    /*!
     * @brief [OUT] Global inference status passed back to the client of this RMCTRL.
    */
    LW_STATUS                                 inferenceStatus;
} LW2080_CTRL_NNE_NNE_DESC_INFERENCE_PARAMS;
typedef struct LW2080_CTRL_NNE_NNE_DESC_INFERENCE_PARAMS *PLW2080_CTRL_NNE_NNE_DESC_INFERENCE_PARAMS;

#define LW2080_CTRL_CMD_NNE_NNE_DESC_INFERENCE (0x20803703) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_NNE_INTERFACE_ID << 8) | LW2080_CTRL_NNE_NNE_DESC_INFERENCE_PARAMS_MESSAGE_ID" */

/* _ctrl2080nne_h_ */

#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)

