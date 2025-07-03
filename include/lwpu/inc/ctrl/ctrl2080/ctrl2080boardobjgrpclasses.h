/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2021 LWPU CORPORATION & AFFILIATES. All rights reserved.
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
// Source file: ctrl/ctrl2080/ctrl2080boardobjgrpclasses.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)



#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)

#include "lwtypes.h"

/*!
 * @file    ctrl2080boardobjgrpclasses.h
 *
 * @brief   Enumeration of all unique BOARDOBJGRP class identifiers.
 */

/*!
 * @brief   Macro to provide unique Class ID number for a given PMU unit and
 *          one of its BOARODOBJGRP classes.
 *
 * @param[in]   _unit   The unit
 * @param[in]   _class  The class
 *
 * @return  Unique BOARDOBJGRP class identifier
 */
#define LW2080_CTRL_BOARDOBJGRP_CLASS_ID(_unit, _class)                        \
    LW2080_CTRL_BOARDOBJGRP_CLASS_ID_##_unit##_##_class

/*!
 * @brief   Type reserved for @ref LW2080_CTRL_BOARDOBJGRP_CLASS_ID_ENUM
 *          enumerations.
 */
typedef LwU8 LW2080_CTRL_BOARDOBJGRP_CLASS_ID;

/*!
 * @defgroup LW2080_CTRL_BOARDOBJGRP_CLASS_ID_ENUM
 *
 * Enumeration of BOARDOBJGRP class identifiers. Of type
 * @ref LW2080_CTRL_BOARDOBJGRP_CLASS_ID.
 *
 * @{
 */

// UNIT SPI
#define LW2080_CTRL_BOARDOBJGRP_CLASS_ID_SPI__FIRST                        (0x00U)
#define LW2080_CTRL_BOARDOBJGRP_CLASS_ID_SPI_SPI_DEVICE                    (0x00U)
// Insert new SPI class ID here...
#define LW2080_CTRL_BOARDOBJGRP_CLASS_ID_SPI__LAST                         (0x00U)


// UNIT CLK
#define LW2080_CTRL_BOARDOBJGRP_CLASS_ID_CLK__FIRST                        (0x1U) /* finn: Evaluated from "(LW2080_CTRL_BOARDOBJGRP_CLASS_ID_SPI__LAST + 0x01)" */
#define LW2080_CTRL_BOARDOBJGRP_CLASS_ID_CLK_CLK_DOMAIN                    (0x1U) /* finn: Evaluated from "(LW2080_CTRL_BOARDOBJGRP_CLASS_ID_CLK__FIRST + 0x00)" */
#define LW2080_CTRL_BOARDOBJGRP_CLASS_ID_CLK_CLK_PROG                      (0x2U) /* finn: Evaluated from "(LW2080_CTRL_BOARDOBJGRP_CLASS_ID_CLK__FIRST + 0x01)" */
#define LW2080_CTRL_BOARDOBJGRP_CLASS_ID_CLK_ADC_DEVICE                    (0x3U) /* finn: Evaluated from "(LW2080_CTRL_BOARDOBJGRP_CLASS_ID_CLK__FIRST + 0x02)" */
#define LW2080_CTRL_BOARDOBJGRP_CLASS_ID_CLK_NAFLL_DEVICE                  (0x4U) /* finn: Evaluated from "(LW2080_CTRL_BOARDOBJGRP_CLASS_ID_CLK__FIRST + 0x03)" */
#define LW2080_CTRL_BOARDOBJGRP_CLASS_ID_CLK_CLK_VF_POINT                  (0x5U) /* finn: Evaluated from "(LW2080_CTRL_BOARDOBJGRP_CLASS_ID_CLK__FIRST + 0x04)" */
#define LW2080_CTRL_BOARDOBJGRP_CLASS_ID_CLK_CLK_FREQ_CONTROLLER           (0x6U) /* finn: Evaluated from "(LW2080_CTRL_BOARDOBJGRP_CLASS_ID_CLK__FIRST + 0x05)" */
#define LW2080_CTRL_BOARDOBJGRP_CLASS_ID_CLK_PLL_DEVICE                    (0x7U) /* finn: Evaluated from "(LW2080_CTRL_BOARDOBJGRP_CLASS_ID_CLK__FIRST + 0x06)" */
#define LW2080_CTRL_BOARDOBJGRP_CLASS_ID_CLK_FREQ_DOMAIN                   (0x8U) /* finn: Evaluated from "(LW2080_CTRL_BOARDOBJGRP_CLASS_ID_CLK__FIRST + 0x07)" */
#define LW2080_CTRL_BOARDOBJGRP_CLASS_ID_CLK_CLK_VOLT_CONTROLLER           (0x9U) /* finn: Evaluated from "(LW2080_CTRL_BOARDOBJGRP_CLASS_ID_CLK__FIRST + 0x08)" */
#define LW2080_CTRL_BOARDOBJGRP_CLASS_ID_CLK_CLK_ENUM                      (0xaU) /* finn: Evaluated from "(LW2080_CTRL_BOARDOBJGRP_CLASS_ID_CLK__FIRST + 0x09)" */
#define LW2080_CTRL_BOARDOBJGRP_CLASS_ID_CLK_CLK_VF_REL                    (0xbU) /* finn: Evaluated from "(LW2080_CTRL_BOARDOBJGRP_CLASS_ID_CLK__FIRST + 0x0A)" */
#define LW2080_CTRL_BOARDOBJGRP_CLASS_ID_CLK_CLK_PROP_REGIME               (0xlw) /* finn: Evaluated from "(LW2080_CTRL_BOARDOBJGRP_CLASS_ID_CLK__FIRST + 0x0B)" */
#define LW2080_CTRL_BOARDOBJGRP_CLASS_ID_CLK_CLK_PROP_TOP                  (0xdU) /* finn: Evaluated from "(LW2080_CTRL_BOARDOBJGRP_CLASS_ID_CLK__FIRST + 0x0C)" */
#define LW2080_CTRL_BOARDOBJGRP_CLASS_ID_CLK_CLK_PROP_TOP_REL              (0xeU) /* finn: Evaluated from "(LW2080_CTRL_BOARDOBJGRP_CLASS_ID_CLK__FIRST + 0x0D)" */
#define LW2080_CTRL_BOARDOBJGRP_CLASS_ID_CLK_CLIENT_CLK_VF_POINT           (0xfU) /* finn: Evaluated from "(LW2080_CTRL_BOARDOBJGRP_CLASS_ID_CLK__FIRST + 0x0E)" */
#define LW2080_CTRL_BOARDOBJGRP_CLASS_ID_CLK_CLIENT_CLK_PROP_TOP_POL       (0x10U) /* finn: Evaluated from "(LW2080_CTRL_BOARDOBJGRP_CLASS_ID_CLK__FIRST + 0x0F)" */
// Insert new class ID here...
#define LW2080_CTRL_BOARDOBJGRP_CLASS_ID_CLK__LAST                         (0x10U) /* finn: Evaluated from "(LW2080_CTRL_BOARDOBJGRP_CLASS_ID_CLK__FIRST + 0x0F)" */


// UNIT VOLT
#define LW2080_CTRL_BOARDOBJGRP_CLASS_ID_VOLT__FIRST                       (0x11U) /* finn: Evaluated from "(LW2080_CTRL_BOARDOBJGRP_CLASS_ID_CLK__LAST + 0x01)" */
#define LW2080_CTRL_BOARDOBJGRP_CLASS_ID_VOLT_VOLT_RAIL                    (0x11U) /* finn: Evaluated from "(LW2080_CTRL_BOARDOBJGRP_CLASS_ID_VOLT__FIRST + 0x00)" */
#define LW2080_CTRL_BOARDOBJGRP_CLASS_ID_VOLT_VOLT_DEVICE                  (0x12U) /* finn: Evaluated from "(LW2080_CTRL_BOARDOBJGRP_CLASS_ID_VOLT__FIRST + 0x01)" */
#define LW2080_CTRL_BOARDOBJGRP_CLASS_ID_VOLT_VOLT_POLICY                  (0x13U) /* finn: Evaluated from "(LW2080_CTRL_BOARDOBJGRP_CLASS_ID_VOLT__FIRST + 0x02)" */
// Insert new class ID here...
#define LW2080_CTRL_BOARDOBJGRP_CLASS_ID_VOLT__LAST                        (0x13U) /* finn: Evaluated from "(LW2080_CTRL_BOARDOBJGRP_CLASS_ID_VOLT__FIRST + 0x02)" */


// UNIT FAN
#define LW2080_CTRL_BOARDOBJGRP_CLASS_ID_FAN__FIRST                        (0x14U) /* finn: Evaluated from "(LW2080_CTRL_BOARDOBJGRP_CLASS_ID_VOLT__LAST + 0x01)" */
#define LW2080_CTRL_BOARDOBJGRP_CLASS_ID_FAN_FAN_COOLER                    (0x14U) /* finn: Evaluated from "(LW2080_CTRL_BOARDOBJGRP_CLASS_ID_FAN__FIRST + 0x00)" */
#define LW2080_CTRL_BOARDOBJGRP_CLASS_ID_FAN_FAN_POLICY                    (0x15U) /* finn: Evaluated from "(LW2080_CTRL_BOARDOBJGRP_CLASS_ID_FAN__FIRST + 0x01)" */
#define LW2080_CTRL_BOARDOBJGRP_CLASS_ID_FAN_FAN_ARBITER                   (0x16U) /* finn: Evaluated from "(LW2080_CTRL_BOARDOBJGRP_CLASS_ID_FAN__FIRST + 0x02)" */
// Insert new class ID here...
#define LW2080_CTRL_BOARDOBJGRP_CLASS_ID_FAN__LAST                         (0x16U) /* finn: Evaluated from "(LW2080_CTRL_BOARDOBJGRP_CLASS_ID_FAN__FIRST + 0x02)" */


// UNIT PERF
#define LW2080_CTRL_BOARDOBJGRP_CLASS_ID_PERF__FIRST                       (0x17U) /* finn: Evaluated from "(LW2080_CTRL_BOARDOBJGRP_CLASS_ID_FAN__LAST + 0x01)" */
#define LW2080_CTRL_BOARDOBJGRP_CLASS_ID_PERF_VFE_VAR                      (0x17U) /* finn: Evaluated from "(LW2080_CTRL_BOARDOBJGRP_CLASS_ID_PERF__FIRST + 0x00)" */
#define LW2080_CTRL_BOARDOBJGRP_CLASS_ID_PERF_VFE_EQU                      (0x18U) /* finn: Evaluated from "(LW2080_CTRL_BOARDOBJGRP_CLASS_ID_PERF__FIRST + 0x01)" */
#define LW2080_CTRL_BOARDOBJGRP_CLASS_ID_PERF_VPSTATE                      (0x19U) /* finn: Evaluated from "(LW2080_CTRL_BOARDOBJGRP_CLASS_ID_PERF__FIRST + 0x02)" */
#define LW2080_CTRL_BOARDOBJGRP_CLASS_ID_PERF_PSTATE                       (0x1aU) /* finn: Evaluated from "(LW2080_CTRL_BOARDOBJGRP_CLASS_ID_PERF__FIRST + 0x03)" */
#define LW2080_CTRL_BOARDOBJGRP_CLASS_ID_PERF_PERF_LIMIT                   (0x1bU) /* finn: Evaluated from "(LW2080_CTRL_BOARDOBJGRP_CLASS_ID_PERF__FIRST + 0x04)" */
#define LW2080_CTRL_BOARDOBJGRP_CLASS_ID_PERF_POLICY                       (0x1lw) /* finn: Evaluated from "(LW2080_CTRL_BOARDOBJGRP_CLASS_ID_PERF__FIRST + 0x05)" */
// Insert new class ID here...
#define LW2080_CTRL_BOARDOBJGRP_CLASS_ID_PERF__LAST                        (0x1lw) /* finn: Evaluated from "(LW2080_CTRL_BOARDOBJGRP_CLASS_ID_PERF__FIRST + 0x05)" */


// UNIT THERM
#define LW2080_CTRL_BOARDOBJGRP_CLASS_ID_THERM__FIRST                      (0x1dU) /* finn: Evaluated from "(LW2080_CTRL_BOARDOBJGRP_CLASS_ID_PERF__LAST + 0x01)" */
#define LW2080_CTRL_BOARDOBJGRP_CLASS_ID_THERM_THERM_DEVICE                (0x1dU) /* finn: Evaluated from "(LW2080_CTRL_BOARDOBJGRP_CLASS_ID_THERM__FIRST + 0x00)" */
#define LW2080_CTRL_BOARDOBJGRP_CLASS_ID_THERM_THERM_CHANNEL               (0x1eU) /* finn: Evaluated from "(LW2080_CTRL_BOARDOBJGRP_CLASS_ID_THERM__FIRST + 0x01)" */
#define LW2080_CTRL_BOARDOBJGRP_CLASS_ID_THERM_THERM_POLICY                (0x1fU) /* finn: Evaluated from "(LW2080_CTRL_BOARDOBJGRP_CLASS_ID_THERM__FIRST + 0x02)" */
#define LW2080_CTRL_BOARDOBJGRP_CLASS_ID_THERM_THERM_MONITOR               (0x20U) /* finn: Evaluated from "(LW2080_CTRL_BOARDOBJGRP_CLASS_ID_THERM__FIRST + 0x03)" */
// Insert new class ID here...
#define LW2080_CTRL_BOARDOBJGRP_CLASS_ID_THERM__LAST                       (0x20U) /* finn: Evaluated from "(LW2080_CTRL_BOARDOBJGRP_CLASS_ID_THERM__FIRST + 0x03)" */


// UNIT PMGR
#define LW2080_CTRL_BOARDOBJGRP_CLASS_ID_PMGR__FIRST                       (0x21U) /* finn: Evaluated from "(LW2080_CTRL_BOARDOBJGRP_CLASS_ID_THERM__LAST + 0x01)" */
#define LW2080_CTRL_BOARDOBJGRP_CLASS_ID_PMGR_PWR_EQUATION                 (0x21U) /* finn: Evaluated from "(LW2080_CTRL_BOARDOBJGRP_CLASS_ID_PMGR__FIRST + 0x00)" */
#define LW2080_CTRL_BOARDOBJGRP_CLASS_ID_PMGR_PWR_DEVICE                   (0x22U) /* finn: Evaluated from "(LW2080_CTRL_BOARDOBJGRP_CLASS_ID_PMGR__FIRST + 0x01)" */
#define LW2080_CTRL_BOARDOBJGRP_CLASS_ID_PMGR_PWR_CHANNEL                  (0x23U) /* finn: Evaluated from "(LW2080_CTRL_BOARDOBJGRP_CLASS_ID_PMGR__FIRST + 0x02)" */
#define LW2080_CTRL_BOARDOBJGRP_CLASS_ID_PMGR_PWR_POLICY                   (0x24U) /* finn: Evaluated from "(LW2080_CTRL_BOARDOBJGRP_CLASS_ID_PMGR__FIRST + 0x03)" */
#define LW2080_CTRL_BOARDOBJGRP_CLASS_ID_PMGR_I2C_DEVICE                   (0x25U) /* finn: Evaluated from "(LW2080_CTRL_BOARDOBJGRP_CLASS_ID_PMGR__FIRST + 0x04)" */
#define LW2080_CTRL_BOARDOBJGRP_CLASS_ID_PMGR_ILLUM_DEVICE                 (0x26U) /* finn: Evaluated from "(LW2080_CTRL_BOARDOBJGRP_CLASS_ID_PMGR__FIRST + 0x05)" */
#define LW2080_CTRL_BOARDOBJGRP_CLASS_ID_PMGR_ILLUM_ZONE                   (0x27U) /* finn: Evaluated from "(LW2080_CTRL_BOARDOBJGRP_CLASS_ID_PMGR__FIRST + 0x06)" */
// Insert new class ID here...
#define LW2080_CTRL_BOARDOBJGRP_CLASS_ID_PMGR__LAST                        (0x27U) /* finn: Evaluated from "(LW2080_CTRL_BOARDOBJGRP_CLASS_ID_PMGR__FIRST + 0x06)" */


// UNIT PERF_CF
#define LW2080_CTRL_BOARDOBJGRP_CLASS_ID_PERF_CF__FIRST                    (0x28U) /* finn: Evaluated from "(LW2080_CTRL_BOARDOBJGRP_CLASS_ID_PMGR__LAST + 0x01)" */
#define LW2080_CTRL_BOARDOBJGRP_CLASS_ID_PERF_CF_SENSOR                    (0x28U) /* finn: Evaluated from "(LW2080_CTRL_BOARDOBJGRP_CLASS_ID_PERF_CF__FIRST + 0x00)" */
#define LW2080_CTRL_BOARDOBJGRP_CLASS_ID_PERF_CF_PM_SENSOR                 (0x29U) /* finn: Evaluated from "(LW2080_CTRL_BOARDOBJGRP_CLASS_ID_PERF_CF__FIRST + 0x01)" */
#define LW2080_CTRL_BOARDOBJGRP_CLASS_ID_PERF_CF_TOPOLOGY                  (0x2aU) /* finn: Evaluated from "(LW2080_CTRL_BOARDOBJGRP_CLASS_ID_PERF_CF__FIRST + 0x02)" */
#define LW2080_CTRL_BOARDOBJGRP_CLASS_ID_PERF_CF_CONTROLLER                (0x2bU) /* finn: Evaluated from "(LW2080_CTRL_BOARDOBJGRP_CLASS_ID_PERF_CF__FIRST + 0x03)" */
#define LW2080_CTRL_BOARDOBJGRP_CLASS_ID_PERF_CF_POLICY                    (0x2lw) /* finn: Evaluated from "(LW2080_CTRL_BOARDOBJGRP_CLASS_ID_PERF_CF__FIRST + 0x04)" */
#define LW2080_CTRL_BOARDOBJGRP_CLASS_ID_PERF_CF_PWR_MODEL                 (0x2dU) /* finn: Evaluated from "(LW2080_CTRL_BOARDOBJGRP_CLASS_ID_PERF_CF__FIRST + 0x05)" */
#define LW2080_CTRL_BOARDOBJGRP_CLASS_ID_PERF_CF_CLIENT_PWR_MODEL_PROFILE  (0x2eU) /* finn: Evaluated from "(LW2080_CTRL_BOARDOBJGRP_CLASS_ID_PERF_CF__FIRST + 0x06)" */
#define LW2080_CTRL_BOARDOBJGRP_CLASS_ID_PERF_CF_CLIENT_PERF_CF_POLICY     (0x2fU) /* finn: Evaluated from "(LW2080_CTRL_BOARDOBJGRP_CLASS_ID_PERF_CF__FIRST + 0x07)" */
#define LW2080_CTRL_BOARDOBJGRP_CLASS_ID_PERF_CF_CLIENT_PERF_CF_CONTROLLER (0x30U) /* finn: Evaluated from "(LW2080_CTRL_BOARDOBJGRP_CLASS_ID_PERF_CF__FIRST + 0x08)" */
// Insert new class ID here...
#define LW2080_CTRL_BOARDOBJGRP_CLASS_ID_PERF_CF__LAST                     (0x30U) /* finn: Evaluated from "(LW2080_CTRL_BOARDOBJGRP_CLASS_ID_PERF_CF__FIRST + 0x08)" */


// UNIT NNE
#define LW2080_CTRL_BOARDOBJGRP_CLASS_ID_NNE__FIRST                        (0x31U) /* finn: Evaluated from "(LW2080_CTRL_BOARDOBJGRP_CLASS_ID_PERF_CF__LAST + 0x01)" */
#define LW2080_CTRL_BOARDOBJGRP_CLASS_ID_NNE_NNE_VAR                       (0x31U) /* finn: Evaluated from "(LW2080_CTRL_BOARDOBJGRP_CLASS_ID_NNE__FIRST + 0x00)" */
#define LW2080_CTRL_BOARDOBJGRP_CLASS_ID_NNE_NNE_LAYER                     (0x32U) /* finn: Evaluated from "(LW2080_CTRL_BOARDOBJGRP_CLASS_ID_NNE__FIRST + 0x01)" */
#define LW2080_CTRL_BOARDOBJGRP_CLASS_ID_NNE_NNE_DESC                      (0x33U) /* finn: Evaluated from "(LW2080_CTRL_BOARDOBJGRP_CLASS_ID_NNE__FIRST + 0x02)" */
#define LW2080_CTRL_BOARDOBJGRP_CLASS_ID_NNE__LAST                         (0x33U) /* finn: Evaluated from "(LW2080_CTRL_BOARDOBJGRP_CLASS_ID_NNE__FIRST + 0x02)" */

// Add a new UNIT's BOARDOBJGRP Class IDs here...

// Update this when a new unit is added
#define LW2080_CTRL_BOARDOBJGRP_CLASS_ID__MAX                              (LW2080_CTRL_BOARDOBJGRP_CLASS_ID_NNE__LAST)
#define LW2080_CTRL_BOARODBJGRP_CLASS_ID_ILWALID                           (LW_U8_MAX) // Should match max value of LW2080_CTRL_BOARDOBJGRP_CLASS_ID
/*!@}*/
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)

