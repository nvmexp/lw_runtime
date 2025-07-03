/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2021 LWPU CORPORATION & AFFILIATES. All rights reserved.
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
// Source file: ctrl/ctrl2080/ctrl2080perf_cf_pwr_model.finn
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
#include "ctrl/ctrl2080/ctrl2080clk.h"
#include "ctrl/ctrl2080/ctrl2080mc.h"

/*!
 * Macros for PERF_CF Power Model types
 */
#define LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_TYPE_RP_PM_1X                     0x01 // Deprecated
#define LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_TYPE_DLPPM_1X                     0x02
#define LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_TYPE_TGP_1X                       0x03

/*!
 * Unused type for perf/pwr metrics
 */
#define LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_TYPE_UNUSED               LW_U32_MAX

/*!
 * Max value for a metric type
 */
#define LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_VALUE_MAX                 LW_U32_MAX

/*!
 * Type of the PWR_MODEL metrics contained in the PWR_MODEL_METRICS structure.
 */
#define LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_TYPE_ILWALID              0x00
#define LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_TYPE_RP_PM_1X             0x01 // Deprecated
#define LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_TYPE_WORKLOAD_SINGLE_1X   0x02
#define LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_TYPE_WORKLOAD_COMBINED_1X 0x03
#define LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_TYPE_DLPPM_1X             0x04
#define LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_TYPE_TGP_1X               0x05

/*!
 * Max number of ADC devices
 */
#define LW2080_CTRL_VOLT_CLK_ADC_ACC_SAMPLE_MAX                              13

/*!
 * Structure describes attributes common to all types of Pwr Models
 */
typedef struct LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS {
    /*!
     * LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS_TYPE_<xyz>
     */
    LwU8   type;
    /*!
     * Has the metrics structure been filled completely?
     */
    LwBool bValid;
} LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS;
typedef struct LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS *PLW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS;

/*!
 * @brief Defines the structure that holds data used to execute the
 * @ref voltRailGetVoltageSensed() API.
 */
typedef struct LW2080_CTRL_VOLT_RAIL_SENSED_VOLTAGE_DATA {
    /*!
     * @brief [in/out] Client provided output buffer to store
     * @ref CLK_ADC_ACC_SAMPLE data. Client should provide buffer size that can
     * store at-least (boardObjGrpMaskBitIdxHighest(VOLT_RAIL::adcDevMask) + 1)
     * samples as the buffer is used to directly index into the ADC Device Table
     * without packing/compressing for unused ADC Device Table entries.
     */
    LW2080_CTRL_CLK_ADC_ACC_SAMPLE clkAdcAccSample[LW2080_CTRL_VOLT_CLK_ADC_ACC_SAMPLE_MAX];
    /*!
     * @brief [in] Number of samples that can be stored in client provided
     * output buffer.
     */
    LwU8                           numSamples;
    /*!
     * @brief [in] Sensed voltage mode
     * @ref LW2080_CTRL_VOLT_VOLT_RAIL_SENSED_VOLTAGE_MODE_<xyz>.
     */
    LwU8                           mode;
    /*!
     * @brief [out] Actual sensed voltage reading based on
     * @ref LW2080_CTRL_VOLT_VOLT_RAIL_SENSED_VOLTAGE_MODE_<xyz>.
     */
    LwU32                          voltageuV;
} LW2080_CTRL_VOLT_RAIL_SENSED_VOLTAGE_DATA;
typedef struct LW2080_CTRL_VOLT_RAIL_SENSED_VOLTAGE_DATA *PLW2080_CTRL_VOLT_RAIL_SENSED_VOLTAGE_DATA;

/*!
 * A frequency and its minimum required voltage for a particular clock domain
 */
typedef struct LW2080_CTRL_PMGR_PWR_POLICY_PERF_CF_PWR_MODEL_WORKLOAD_SINGLE_1X_INDEPENDENT_DOMAIN_VMIN {
    /*!
     * Current frequency in kHz
     */
    LwU32 freqkHz;
    /*!
     * Required voltage to sustain freqkHz.
     */
    LwU32 voltuV;
} LW2080_CTRL_PMGR_PWR_POLICY_PERF_CF_PWR_MODEL_WORKLOAD_SINGLE_1X_INDEPENDENT_DOMAIN_VMIN;

/*!
 * Structure describing frequencies across independent clock domains and their
 * respective vmin requirements.
 */
typedef struct LW2080_CTRL_PMGR_PWR_POLICY_PERF_CF_PWR_MODEL_WORKLOAD_SINGLE_1X_INDEPENDENT_DOMAINS_VMIN {
    /*!
     * Voltage floor in uV.
     *
     * This is the maximum of the rail vmin and all the clock vmins
     */
    LwU32                                                                                    voltFlooruV;
    /*!
     * Vmin for the rail itself (irrespective of clock domains), 0 uV if not
     * supported.
     */
    LwU32                                                                                    railVminuV;
    /*!
     * Mask of clock domains where voltage stops scaling
     */
    LW2080_CTRL_BOARDOBJGRP_MASK_E32                                                         independentClkDomainMask;
    /*!
     * Independent domain vmins considered when callwlating ::voltFlooruV
     */
    LW2080_CTRL_PMGR_PWR_POLICY_PERF_CF_PWR_MODEL_WORKLOAD_SINGLE_1X_INDEPENDENT_DOMAIN_VMIN domains[LW2080_CTRL_CLK_CLK_DOMAIN_CLIENT_MAX_DOMAINS];
} LW2080_CTRL_PMGR_PWR_POLICY_PERF_CF_PWR_MODEL_WORKLOAD_SINGLE_1X_INDEPENDENT_DOMAINS_VMIN;

/*!
 * Structure representing observed metrics for WORKLOAD_SINGLE_1X
 */
typedef struct LW2080_CTRL_PMGR_PWR_POLICY_PERF_CF_PWR_MODEL_OBSERVED_METRICS_WORKLOAD_SINGLE_1X {
    /*!
     * super - must be the first member of the structure
     */
    LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS                                                super;
    /*!
     * Frequency (MHz)
     */
    LwU32                                                                                     freqMHz;
    /*!
     * Estimated leakage - in units specified by @ref
     * RM_PMU_PMGR_PWR_POLICY::limitUnit - i.e. power (mW) or current (mA).
     */
    LwU32                                                                                     leakagemX;
    /*!
     * Voltage in mVolt.
     */
    LwU32                                                                                     voltmV;
    /*!
     * Power in mW/ Current in mA
     */
    LwU32                                                                                     observedVal;
    /*!
     * The current workload/active capacitance (w) callwlated by @ref
     * s_pwrPolicyWorkloadComputeWorkload().  This value is filtered.
     */
    LwUFXP20_12                                                                               workload;
    /*!
     * Last counter and timestamp values to be used for callwlating average
     * frequency over the sampled power period.
     */
    LW2080_CTRL_CLK_CNTR_SAMPLE_ALIGNED                                                       clkCntrStart;
    /*!
     * @copydoc LW2080_CTRL_VOLT_RAIL_SENSED_VOLTAGE_DATA.
     */
    LW2080_CTRL_VOLT_RAIL_SENSED_VOLTAGE_DATA                                                 sensed;
    /*!
     * Vmin that must be considered in model for independent clock domains.
     */
    LW2080_CTRL_PMGR_PWR_POLICY_PERF_CF_PWR_MODEL_WORKLOAD_SINGLE_1X_INDEPENDENT_DOMAINS_VMIN independentDomainsVmin;
    /*!
     * Residency of lowpower features
     */
    LwUFXP20_12                                                                               lpwrResidency[LW2080_CTRL_MC_POWERGATING_ENGINE_ID_MAX];
    /*!
     * Soft Floor value  (MHz)
     */
    LwU32                                                                                     freqSoftFloorMHz;
    /*!
     * Fmax @ Vmin (MHz) as computed from @ref independentDomainsVmin.
     */
    LwU32                                                                                     freqFmaxVminMHz;
} LW2080_CTRL_PMGR_PWR_POLICY_PERF_CF_PWR_MODEL_OBSERVED_METRICS_WORKLOAD_SINGLE_1X;
typedef struct LW2080_CTRL_PMGR_PWR_POLICY_PERF_CF_PWR_MODEL_OBSERVED_METRICS_WORKLOAD_SINGLE_1X *PLW2080_CTRL_PMGR_PWR_POLICY_PERF_CF_PWR_MODEL_OBSERVED_METRICS_WORKLOAD_SINGLE_1X;

/*!
 * Structure representing estimated metrics for WORKLOAD_SINGLE_1X
 */
typedef struct LW2080_CTRL_PMGR_PWR_POLICY_PERF_CF_PWR_MODEL_ESTIMATED_METRICS_WORKLOAD_SINGLE_1X {
    /*!
     * super - must be the first member of the structure
     */
    LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS                                                super;
    /*!
     * The filtered workload/active capacitance (w) which is used to determine
     * the target clocks.
     */
    LwUFXP20_12                                                                               workload;
    /*!
     * Estimated leakage - in units specified by @ref
     * RM_PMU_PMGR_PWR_POLICY::limitUnit - i.e. power (mW) or current (mA).
     */
    LwU32                                                                                     leakagemX;
    /*!
     * Voltage in mVolt
     */
    LwU32                                                                                     voltmV;
    /*!
     * In/Out Frequency (MHz).
     */
    LwU32                                                                                     freqMHz;
    /*!
     * Effective frequency after scaling it by lpwr/clk gating residency
     */
    LwU32                                                                                     effectiveFreqMHz;
    /*!
     * Power in mW/ Current in mA
     */
    LwU32                                                                                     estimatedVal;
    /*!
     * Frequency in MHz at the voltage floor.
     */
    LwU32                                                                                     freqFloorMHz;
    /*!
     * Power in mW/ Current in mA at the voltage floor.
     */
    LwU32                                                                                     estimatedValFloor;
    /*!
     * Vmin that must be considered in model for independent clock domains.
     */
    LW2080_CTRL_PMGR_PWR_POLICY_PERF_CF_PWR_MODEL_WORKLOAD_SINGLE_1X_INDEPENDENT_DOMAINS_VMIN independentDomainsVmin;
    /*!
     * Estimated residency of lowpower features
     */
    LwUFXP20_12                                                                               lpwrResidency[LW2080_CTRL_MC_POWERGATING_ENGINE_ID_MAX];
} LW2080_CTRL_PMGR_PWR_POLICY_PERF_CF_PWR_MODEL_ESTIMATED_METRICS_WORKLOAD_SINGLE_1X;
typedef struct LW2080_CTRL_PMGR_PWR_POLICY_PERF_CF_PWR_MODEL_ESTIMATED_METRICS_WORKLOAD_SINGLE_1X *PLW2080_CTRL_PMGR_PWR_POLICY_PERF_CF_PWR_MODEL_ESTIMATED_METRICS_WORKLOAD_SINGLE_1X;

/*!
 * Maximum number of WORKLOAD_SINGLE policies
 */
#define LW2080_CTRL_PMGR_PWR_POLICY_WORKLOAD_SINGLE_1X_MAX 0x02

/*!
 * Structure representing observed metrics for WORKLOAD_COMBINED
 */
typedef struct LW2080_CTRL_PMGR_PWR_POLICY_PERF_CF_PWR_MODEL_OBSERVED_METRICS_WORKLOAD_COMBINED_1X {
    /*!
     * super - must be the first member of the structure
     */
    LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS                                        super;
    /*!
     * Power in mW/Current in mA which is sum of the observedVal of all
     * WORKLOAD_SINGLE_1X rails referenced by this policy.
     */
    LwU32                                                                             observedVal;
    /*!
     * @copydoc LW2080_CTRL_PMGR_PWR_POLICY_PERF_CF_PWR_MODEL_OBSERVED_METRICS_WORKLOAD_SINGLE_1X
     */
    LW2080_CTRL_PMGR_PWR_POLICY_PERF_CF_PWR_MODEL_OBSERVED_METRICS_WORKLOAD_SINGLE_1X singles[LW2080_CTRL_PMGR_PWR_POLICY_WORKLOAD_SINGLE_1X_MAX];
} LW2080_CTRL_PMGR_PWR_POLICY_PERF_CF_PWR_MODEL_OBSERVED_METRICS_WORKLOAD_COMBINED_1X;
typedef struct LW2080_CTRL_PMGR_PWR_POLICY_PERF_CF_PWR_MODEL_OBSERVED_METRICS_WORKLOAD_COMBINED_1X *PLW2080_CTRL_PMGR_PWR_POLICY_PERF_CF_PWR_MODEL_OBSERVED_METRICS_WORKLOAD_COMBINED_1X;

/*!
 * Structure representing estimated metrics for WORKLOAD_COMBINED_1X
 */
typedef struct LW2080_CTRL_PMGR_PWR_POLICY_PERF_CF_PWR_MODEL_ESTIMATED_METRICS_WORKLOAD_COMBINED_1X {
    /*!
     * super - must be the first member of the structure
     */
    LW2080_CTRL_PERF_PERF_CF_PWR_MODEL_METRICS                                         super;
    /*!
     * Power in mW/Current in mA which is sum of the estimatedVal of all
     * WORKLOAD_SINGLE_1X rails referenced by this policy.
     */
    LwU32                                                                              estimatedVal;
    /*!
     * @copydoc LW2080_CTRL_PMGR_PWR_POLICY_PERF_CF_PWR_MODEL_ESTIMATED_METRICS_WORKLOAD_SINGLE_1X
     */
    LW2080_CTRL_PMGR_PWR_POLICY_PERF_CF_PWR_MODEL_ESTIMATED_METRICS_WORKLOAD_SINGLE_1X singles[LW2080_CTRL_PMGR_PWR_POLICY_WORKLOAD_SINGLE_1X_MAX];
} LW2080_CTRL_PMGR_PWR_POLICY_PERF_CF_PWR_MODEL_ESTIMATED_METRICS_WORKLOAD_COMBINED_1X;
typedef struct LW2080_CTRL_PMGR_PWR_POLICY_PERF_CF_PWR_MODEL_ESTIMATED_METRICS_WORKLOAD_COMBINED_1X *PLW2080_CTRL_PMGR_PWR_POLICY_PERF_CF_PWR_MODEL_ESTIMATED_METRICS_WORKLOAD_COMBINED_1X;

/*!
 * Structure of static information specific to the PERF_CF_PWR_MODEL interface.
 */
typedef struct LW2080_CTRL_PMGR_PWR_POLICY_INFO_DATA_PERF_CF_PWR_MODEL {
    /*!
     * BOARDOBJ_INTERFACE super class.  Must always be first element in the structure.
     */
    LW2080_CTRL_BOARDOBJ_INTERFACE super;
} LW2080_CTRL_PMGR_PWR_POLICY_INFO_DATA_PERF_CF_PWR_MODEL;
typedef struct LW2080_CTRL_PMGR_PWR_POLICY_INFO_DATA_PERF_CF_PWR_MODEL *PLW2080_CTRL_PMGR_PWR_POLICY_INFO_DATA_PERF_CF_PWR_MODEL;

/*!
 * Structure representing PERF_CF_PWR_MODEL-specific PWR_POLICY dynamic status.
 */
typedef struct LW2080_CTRL_PMGR_PWR_POLICY_STATUS_DATA_PERF_CF_PWR_MODEL {
    /*!
     * BOARDOBJ_INTERFACE super class.  Must always be first element in the structure.
     */
    LW2080_CTRL_BOARDOBJ_INTERFACE super;
} LW2080_CTRL_PMGR_PWR_POLICY_STATUS_DATA_PERF_CF_PWR_MODEL;
typedef struct LW2080_CTRL_PMGR_PWR_POLICY_STATUS_DATA_PERF_CF_PWR_MODEL *PLW2080_CTRL_PMGR_PWR_POLICY_STATUS_DATA_PERF_CF_PWR_MODEL;

/*!
 * Structure representing PERF_CF_PWR_MODEL-specifc PWR_POLICY control/policy parameters.
 */
typedef struct LW2080_CTRL_PMGR_PWR_POLICY_CONTROL_DATA_PERF_CF_PWR_MODEL {
    /*!
     * BOARDOBJ_INTERFACE super class.  Must always be first element in the structure.
     */
    LW2080_CTRL_BOARDOBJ_INTERFACE super;
} LW2080_CTRL_PMGR_PWR_POLICY_CONTROL_DATA_PERF_CF_PWR_MODEL;
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)

