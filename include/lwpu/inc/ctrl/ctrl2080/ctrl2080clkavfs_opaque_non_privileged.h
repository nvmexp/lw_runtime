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

#include <lwtypes.h>
#if defined(_MSC_VER)
#pragma warning(disable:4324)
#endif

//
// This file was generated with FINN, an LWPU coding tool.
// Source file: ctrl/ctrl2080/ctrl2080clkavfs_opaque_non_privileged.finn
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
 * Define the maximum number of devices that may be included in the ADC devices
 * descriptor table (@ref LW2080_CTRL_CLK_ADC_MAX_DEVICES).
 */
#define LW2080_CTRL_CLK_ADC_MAX_DEVICES                             (32U)

/*!
 * Valid global ADC ID values
 */
#define LW2080_CTRL_CLK_ADC_ID_SYS                                  (0x00000000U)
#define LW2080_CTRL_CLK_ADC_ID_LTC                                  (0x00000001U) // GP100 only
#define LW2080_CTRL_CLK_ADC_ID_XBAR                                 (0x00000002U)
#define LW2080_CTRL_CLK_ADC_ID_GPC0                                 (0x00000003U)
#define LW2080_CTRL_CLK_ADC_ID_GPC1                                 (0x00000004U)
#define LW2080_CTRL_CLK_ADC_ID_GPC2                                 (0x00000005U)
#define LW2080_CTRL_CLK_ADC_ID_GPC3                                 (0x00000006U)
#define LW2080_CTRL_CLK_ADC_ID_GPC4                                 (0x00000007U)
#define LW2080_CTRL_CLK_ADC_ID_GPC5                                 (0x00000008U)
#define LW2080_CTRL_CLK_ADC_ID_GPCS                                 (0x00000009U)
#define LW2080_CTRL_CLK_ADC_ID_SRAM                                 (0x0000000AU) // GP100 and GP10X only
#define LW2080_CTRL_CLK_ADC_ID_LWD                                  (0x0000000BU) // Volta and later
#define LW2080_CTRL_CLK_ADC_ID_HOST                                 (0x0000000LW) // Volta and later
#define LW2080_CTRL_CLK_ADC_ID_GPC6                                 (0x0000000DU) // Ampere and later
#define LW2080_CTRL_CLK_ADC_ID_GPC7                                 (0x0000000EU) // Ampere and later
#define LW2080_CTRL_CLK_ADC_ID_GPC8                                 (0x0000000FU) // AD10X and later
#define LW2080_CTRL_CLK_ADC_ID_GPC9                                 (0x00000010U) // AD10X and later
#define LW2080_CTRL_CLK_ADC_ID_GPC10                                (0x00000011U) // AD10X and later
#define LW2080_CTRL_CLK_ADC_ID_GPC11                                (0x00000012U) // AD10X and later
#define LW2080_CTRL_CLK_ADC_ID_SYS_ISINK                            (0x00000013U) // AD10X and later
#define LW2080_CTRL_CLK_ADC_ID_MAX                                  (0x00000014U) // MUST BE LAST
#define LW2080_CTRL_CLK_ADC_ID_UNDEFINED                            (0x000000FFU)

/*!
 * Mask of all GPC ADC IDs supported by RM
 */
#define    LW2080_CTRL_CLK_ADC_MASK_UNICAST_GPC    (LWBIT(LW2080_CTRL_CLK_ADC_ID_GPC0)  | \
                                                    LWBIT(LW2080_CTRL_CLK_ADC_ID_GPC1)  | \
                                                    LWBIT(LW2080_CTRL_CLK_ADC_ID_GPC2)  | \
                                                    LWBIT(LW2080_CTRL_CLK_ADC_ID_GPC3)  | \
                                                    LWBIT(LW2080_CTRL_CLK_ADC_ID_GPC4)  | \
                                                    LWBIT(LW2080_CTRL_CLK_ADC_ID_GPC5)  | \
                                                    LWBIT(LW2080_CTRL_CLK_ADC_ID_GPC6)  | \
                                                    LWBIT(LW2080_CTRL_CLK_ADC_ID_GPC7)  | \
                                                    LWBIT(LW2080_CTRL_CLK_ADC_ID_GPC8)  | \
                                                    LWBIT(LW2080_CTRL_CLK_ADC_ID_GPC9)  | \
                                                    LWBIT(LW2080_CTRL_CLK_ADC_ID_GPC10) | \
                                                    LWBIT(LW2080_CTRL_CLK_ADC_ID_GPC11))

/*!
 * Various ADC device table versions that are supported
 */
#define LW2080_CTRL_CLK_ADC_DEVICES_DISABLED                        (0x00000000U)
#define LW2080_CTRL_CLK_ADC_DEVICES_V10                             (0x00000001U)
#define LW2080_CTRL_CLK_ADC_DEVICES_V20                             (0x00000002U)

/*!
 * Various types of ADC device that the GPU can support
 */
#define LW2080_CTRL_CLK_ADC_DEVICE_TYPE_BASE                        (0x00000000U)
#define LW2080_CTRL_CLK_ADC_DEVICE_TYPE_V10                         (0x00000001U)
#define LW2080_CTRL_CLK_ADC_DEVICE_TYPE_V20                         (0x00000002U)
#define LW2080_CTRL_CLK_ADC_DEVICE_TYPE_V30                         (0x00000003U)
#define LW2080_CTRL_CLK_ADC_DEVICE_TYPE_V30_ISINK_V10               (0x00000004U)
// Insert new types here and increment _MAX
#define LW2080_CTRL_CLK_ADC_DEVICE_TYPE_MAX                         (0x00000005U)
#define LW2080_CTRL_CLK_ADC_DEVICE_TYPE_DISABLED                    (0x000000FFU)

/*!
 * Special define to represent an invalid CLK_ADC index.
 */
#define LW2080_CTRL_CLK_ADC_INDEX_ILWALID                           LW2080_CTRL_BOARDOBJ_IDX_ILWALID_8BIT

/*!
 * Various types of ADC calibration that the GPU can support
 * Valid/Used only through Turing - Ampere onwards we support offset/gain based
 * calibration by default.
 */
#define LW2080_CTRL_CLK_ADC_CAL_TYPE_V10                            (0x00000000U)
#define LW2080_CTRL_CLK_ADC_CAL_TYPE_V20                            (0x00000001U)

/*!
 * Enumeration of clients which can disable/enable ADC device.
 *
 * TODO: kwadhwa - Bug 200605415 - merge with LW2080_CTRL_CLK_CLIENT_ID_
 */

#define LW2080_CTRL_CLK_ADC_CLIENT_ID_RM                            (0x00U)
#define LW2080_CTRL_CLK_ADC_CLIENT_ID_INIT                          (0x01U)
#define LW2080_CTRL_CLK_ADC_CLIENT_ID_PMU                           (0x02U)
#define LW2080_CTRL_CLK_ADC_CLIENT_ID_LPWR_DI                       (0x03U)
#define LW2080_CTRL_CLK_ADC_CLIENT_ID_LPWR_CG                       (0x04U)
#define LW2080_CTRL_CLK_ADC_CLIENT_ID_LPWR_GRRG                     (0x05U)
#define LW2080_CTRL_CLK_ADC_CLIENT_ID_MAX_NUM                       (0x06U)

// Mask of all valid clients that are listed
#define LW2080_CTRL_CLK_ADC_CLIENT_ID_VALID_MASK                 \
            (LWBIT(LW2080_CTRL_CLK_ADC_CLIENT_ID_MAX_NUM) - 1)

/*!
 * ADC output MUX's variout modes of operations.
 */
#define LW2080_CTRL_CLK_ADC_SW_OVERRIDE_ADC_USE_ILWALID             (0x00000000U)    // Invalid Mode to capture uninitialized input.
#define LW2080_CTRL_CLK_ADC_SW_OVERRIDE_ADC_USE_HW_REQ              (0x00000001U)    // HW mode
#define LW2080_CTRL_CLK_ADC_SW_OVERRIDE_ADC_USE_MIN                 (0x00000002U)    // MIN(HW, SW)
#define LW2080_CTRL_CLK_ADC_SW_OVERRIDE_ADC_USE_SW_REQ              (0x00000003U)    // SW Mode

/*!
 * Define the maximum number of devices that may be included in the NAFLL devices
 * descriptor table (@ref LW2080_CTRL_CLK_NAFLL_MAX_DEVICES).
 */
#define LW2080_CTRL_CLK_NAFLL_MAX_DEVICES                           (32U)

/*!
 * Valid global NAFLL ID values
 */
#define LW2080_CTRL_CLK_NAFLL_ID_SYS                                (0x00000000U)
#define LW2080_CTRL_CLK_NAFLL_ID_LTC                                (0x00000001U)  // GP100 only
#define LW2080_CTRL_CLK_NAFLL_ID_XBAR                               (0x00000002U)
#define LW2080_CTRL_CLK_NAFLL_ID_GPC0                               (0x00000003U)
#define LW2080_CTRL_CLK_NAFLL_ID_GPC1                               (0x00000004U)
#define LW2080_CTRL_CLK_NAFLL_ID_GPC2                               (0x00000005U)
#define LW2080_CTRL_CLK_NAFLL_ID_GPC3                               (0x00000006U)
#define LW2080_CTRL_CLK_NAFLL_ID_GPC4                               (0x00000007U)
#define LW2080_CTRL_CLK_NAFLL_ID_GPC5                               (0x00000008U)
#define LW2080_CTRL_CLK_NAFLL_ID_GPCS                               (0x00000009U)
#define LW2080_CTRL_CLK_NAFLL_ID_LWD                                (0x0000000AU)  // Volta and later
#define LW2080_CTRL_CLK_NAFLL_ID_HOST                               (0x0000000BU)  // Volta and later
#define LW2080_CTRL_CLK_NAFLL_ID_GPC6                               (0x0000000LW)  // Ampere and later
#define LW2080_CTRL_CLK_NAFLL_ID_GPC7                               (0x0000000DU)  // Ampere and later
#define LW2080_CTRL_CLK_NAFLL_ID_GPC8                               (0x0000000EU)  // AD10X and later
#define LW2080_CTRL_CLK_NAFLL_ID_GPC9                               (0x0000000FU)  // AD10X and later
#define LW2080_CTRL_CLK_NAFLL_ID_GPC10                              (0x00000010U)  // AD10X and later
#define LW2080_CTRL_CLK_NAFLL_ID_GPC11                              (0x00000011U)  // AD10X and later
#define LW2080_CTRL_CLK_NAFLL_ID_MAX                                (0x00000012U)  // MUST BE LAST
#define LW2080_CTRL_CLK_NAFLL_ID_UNDEFINED                          (0x000000FFU)
#define LW2080_CTRL_CLK_NAFLL_MASK_UNDEFINED                        (0x00000000U)

// Mask of all valid NAFLL IDs that are listed
#define LW2080_CTRL_CLK_NAFLL_ID_VALID_MASK (LWBIT(LW2080_CTRL_CLK_NAFLL_ID_MAX) - 1)

#define LW2080_CTRL_CLK_IS_NAFLL_ID_VALID(_id)                                      \
        (((LWBIT_TYPE(_id, LwU32) & LW2080_CTRL_CLK_NAFLL_ID_VALID_MASK) != 0U)?    \
            LW_TRUE : LW_FALSE)

/*!
 * Mask of all GPC NAFLL IDs supported by RM
 */
#define    LW2080_CTRL_CLK_NAFLL_MASK_UNICAST_GPC    (LWBIT(LW2080_CTRL_CLK_NAFLL_ID_GPC0)  | \
                                                      LWBIT(LW2080_CTRL_CLK_NAFLL_ID_GPC1)  | \
                                                      LWBIT(LW2080_CTRL_CLK_NAFLL_ID_GPC2)  | \
                                                      LWBIT(LW2080_CTRL_CLK_NAFLL_ID_GPC3)  | \
                                                      LWBIT(LW2080_CTRL_CLK_NAFLL_ID_GPC4)  | \
                                                      LWBIT(LW2080_CTRL_CLK_NAFLL_ID_GPC5)  | \
                                                      LWBIT(LW2080_CTRL_CLK_NAFLL_ID_GPC6)  | \
                                                      LWBIT(LW2080_CTRL_CLK_NAFLL_ID_GPC7)  | \
                                                      LWBIT(LW2080_CTRL_CLK_NAFLL_ID_GPC8)  | \
                                                      LWBIT(LW2080_CTRL_CLK_NAFLL_ID_GPC9)  | \
                                                      LWBIT(LW2080_CTRL_CLK_NAFLL_ID_GPC10) | \
                                                      LWBIT(LW2080_CTRL_CLK_NAFLL_ID_GPC11))

/*!
 * Various types of NAFLL device that the GPU can support
 */
#define LW2080_CTRL_CLK_NAFLL_DEVICE_TYPE_BASE                      (0x00000000U)
#define LW2080_CTRL_CLK_NAFLL_DEVICE_TYPE_V10                       (0x00000001U)
#define LW2080_CTRL_CLK_NAFLL_DEVICE_TYPE_V20                       (0x00000002U)
#define LW2080_CTRL_CLK_NAFLL_DEVICE_TYPE_V30                       (0x00000003U)
// Insert new types here and increment _MAX
#define LW2080_CTRL_CLK_NAFLL_DEVICE_TYPE_MAX                       (0x00000004U)
#define LW2080_CTRL_CLK_NAFLL_DEVICE_TYPE_DISABLED                  (0x000000FFU)

/*!
 * Various secondary VF lwrves
 */
#define LW2080_CTRL_CLK_NAFLL_LUT_VF_LWRVE_SEC_0                    (0x00000000U)
#define LW2080_CTRL_CLK_NAFLL_LUT_VF_LWRVE_SEC_1                    (0x00000001U)
// Insert new types here and increment _MAX
#define LW2080_CTRL_CLK_NAFLL_LUT_VF_LWRVE_SEC_MAX                  (0x00000002U)

/*!
 * Various modes for the LUT_VSELECT Mux
 */
#define LW2080_CTRL_CLK_NAFLL_LUT_VSELECT_LOGIC                     (0x00000000U)
#define LW2080_CTRL_CLK_NAFLL_LUT_VSELECT_MIN                       (0x00000001U)
#define LW2080_CTRL_CLK_NAFLL_LUT_VSELECT_SRAM                      (0x00000003U)

/*!
 * Various modes for the SW_OVERRIDE_LUT Mux
 */
#define LW2080_CTRL_CLK_NAFLL_SW_OVERRIDE_LUT_USE_HW_REQ            (0x00000000U)
#define LW2080_CTRL_CLK_NAFLL_SW_OVERRIDE_LUT_USE_MIN               (0x00000001U)
#define LW2080_CTRL_CLK_NAFLL_SW_OVERRIDE_LUT_USE_SW_REQ            (0x00000003U)

/*!
 * Various regime IDs for the NAFLL devices
 * FFR = Fixed Frequency regime
 * FR  = Frequency regime
 * VR  = Voltage Regime
 * VR_ABOVE_NOISE_UNAWARE_VMIN = Voltage Regime to be set only when V > Vmin
 * FFR_BELOW_DVCO_MIN = Fixed Frequency Regime to be set only when F < FDvcoMin
 * VR_WITH_CPM = Voltage Regime to be set with Critical Path Monitor
 */
#define LW2080_CTRL_CLK_NAFLL_REGIME_ID_ILWALID                     (0x00000000U)
#define LW2080_CTRL_CLK_NAFLL_REGIME_ID_FFR                         (0x00000001U)
#define LW2080_CTRL_CLK_NAFLL_REGIME_ID_FR                          (0x00000002U)
#define LW2080_CTRL_CLK_NAFLL_REGIME_ID_VR                          (0x00000003U)
#define LW2080_CTRL_CLK_NAFLL_REGIME_ID_VR_ABOVE_NOISE_UNAWARE_VMIN (0x00000004U)
#define LW2080_CTRL_CLK_NAFLL_REGIME_ID_FFR_BELOW_DVCO_MIN          (0x00000005U)
#define LW2080_CTRL_CLK_NAFLL_REGIME_ID_VR_WITH_CPM                 (0x00000006U)
/*!
 * The last value of regime IDs, used for bounds checking.
 * Should be updated if the list above changes.
 */
#define LW2080_CTRL_CLK_NAFLL_REGIME_ID_MAX                         (0x00000007U)

/*!
 * Helper macro to check whether a given regime ID is valid.
 */
#define LW2080_CTRL_CLK_NAFLL_REGIME_IS_VALID(x)                \
    ((x) > LW2080_CTRL_CLK_NAFLL_REGIME_ID_ILWALID &&           \
     (x) < LW2080_CTRL_CLK_NAFLL_REGIME_ID_MAX)

/*!
 * Macros for PLDIV values
 */
#define LW2080_CTRL_CLK_NAFLL_CLK_NAFLL_PLDIV_DIV1                  (0x1U)
#define LW2080_CTRL_CLK_NAFLL_CLK_NAFLL_PLDIV_DIV2                  (0x2U)
#define LW2080_CTRL_CLK_NAFLL_CLK_NAFLL_PLDIV_DIV3                  (0x3U)
#define LW2080_CTRL_CLK_NAFLL_CLK_NAFLL_PLDIV_DIV4                  (0x4U)

/*!
 * LW2080_CTRL_CMD_CLK_ADC_DEVICES_GET_INFO
 *
 * This command is used to get all the static information of all the ADC
 * present on the GPU
 */
#define LW2080_CTRL_CMD_CLK_ADC_DEVICES_GET_INFO                    (0x208090a0U) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_CLK_LEGACY_NON_PRIVILEGED_INTERFACE_ID << 8) | LW2080_CTRL_CLK_ADC_DEVICES_INFO_PARAMS_MESSAGE_ID" */

typedef struct LW2080_CTRL_CLK_ADC_CAL_V10 {
    /*!
     * Slope value in uV/code for the ADC calibration
     */
    LwU32 slope;

    /*!
     * Intercept value in uV for the ADC calibration
     */
    LwU32 intercept;
} LW2080_CTRL_CLK_ADC_CAL_V10;

typedef struct LW2080_CTRL_CLK_ADC_CAL_V20 {
    /*!
     * Offset value for the ADC calibration version 2
     */
    LwS8                         offset;

    /*!
     * Gain value for the ADC calibration version 2
     */
    LwS8                         gain;

    /*!
     * Coarse common control value for the ADC calibration version 2
     */
    LwU8                         coarseControl;

    /*!
     * VFE Equation Index to provide Offset for individual ADC
     */
#endif // (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1) && (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
    LW2080_CTRL_PERF_VFE_EQU_IDX offsetVfeIdx;
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1) && (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
} LW2080_CTRL_CLK_ADC_CAL_V20;

/*!
 * Structure of static information specific to the V10 ADC device.
 */
typedef struct LW2080_CTRL_CLK_ADC_DEVICE_INFO_DATA_V10 {
    LW2080_CTRL_CLK_ADC_CAL_V10 adcCal;
} LW2080_CTRL_CLK_ADC_DEVICE_INFO_DATA_V10;

/*!
 * Union of ADC calibration data - type-specific data.
 */


/*!
 * Structure of static information specific to the V20 ADC device.
 */
typedef struct LW2080_CTRL_CLK_ADC_DEVICE_INFO_DATA_V20 {
    /*!
     * XAPI does not support using an element within another structure as a
     * discriminant.  Placing this redundant type value here until that design
     * constraint can be fixed. @ref LW2080_CTRL_CLK_ADC_CAL_TYPE_<xyz>
     */
    LwU8 calType;


    /*!
     * Type-specific information.
     */
    union {
        LW2080_CTRL_CLK_ADC_CAL_V10 calV10;

        LW2080_CTRL_CLK_ADC_CAL_V20 calV20;
    } adcCal;
} LW2080_CTRL_CLK_ADC_DEVICE_INFO_DATA_V20;

/*!
 * Structure of static information specific to the V30 ADC device.
 */
typedef struct LW2080_CTRL_CLK_ADC_DEVICE_INFO_DATA_V30 {
    /*!
     * Offset value for ADC calibration obtained either from VBIOS/fuse
     * (sign-magnitude).
     */
    LwS8 offset;

    /*!
     * Gain value for ADC calibration obtained either from VBIOS/fuse
     * (sign-magnitude).
     */
    LwS8 gain;

    /*!
     * Coarse calibration offset for ADC calibration obtained either from
     * VBIOS/fuse (sign-magnitude).
     */
    LwS8 coarseOffset;

    /*!
     * Coarse calibration gain for ADC calibration obtained either from
     * VBIOS/fuse (sign-magnitude).
     */
    LwS8 coarseGain;

    /*!
     * Low temperature error delta - precal ADC code error @ 850mV obtained
     * from fuse (2's complement).
     */
    LwS8 lowTempLowVoltErr;

    /*!
     * Low temperature error delta - precal ADC code error @ 1050 mV obtained
     * from fuse (2's complement).
     */
    LwS8 lowTempHighVoltErr;

    /*!
     * High temperature error delta - precal ADC code error @ 850mV obtained
     * from fuse (2's complement).
     */
    LwS8 highTempLowVoltErr;

    /*!
     * High temperature error delta - precal ADC code error @ 1050 mV obtained
     * from fuse (2's complement).
     */
    LwS8 highTempHighVoltErr;

    /*!
     * ADC code correction offset value for ADC calibration
     * This is callwlated/programmed by SW on change in temperature/voltage.
     */
    LwS8 adcCodeCorrectionOffset;
} LW2080_CTRL_CLK_ADC_DEVICE_INFO_DATA_V30;

/*!
 * Structure of static information specific to the ISINK V10 ADC device.
 */
typedef struct LW2080_CTRL_CLK_ADC_DEVICE_INFO_DATA_ISINK_V10 {
    /*!
     * Minimum cap on ADC code correction offset - 2's complement
     */
    LwS8  adcCodeCorrectionOffsetMin;

    /*!
     * Maximum cap on ADC code correction offset - 2's complement
     */
    LwS8  adcCodeCorrectionOffsetMax;

    /*!
     * Mask of invalid fuse revisions for the fuse from which we obtain gain, 
     * offset and coarse offset (opt_adc_cal).
     */
    LwU8  calIlwalidFuseRevMask;

    /*!
     * Mask of invalid fuse revisions for the fuse from which we obtain low
     * temperature error deltas (opt_adc_code_err_lt).
     */
    LwU8  lowTempErrIlwalidFuseRevMask;

    /*!
     * Mask of invalid fuse revisions for the fuse from which we obtain high
     * temperature error deltas (opt_adc_code_err_ht).
     */
    LwU8  highTempErrIlwalidFuseRevMask;

    /*!
     * Fused ADC calibration revision - opt_adc_cal_fuse_rev
     */
    LwU8  adcCalRevFused;

    /*!
     * Fused ADC code high temperature error fuse revision
     * opt_adc_code_err_ht_fuse_rev
     */
    LwU8  adcCodeErrHtRevFused;

    /*!
     * Fused ADC code low temperature error fuse revision
     * opt_adc_code_err_lt_fuse_rev
     */
    LwU8  adcCodeErrLtRevFused;

    /*!
     * Temperature at which low temperature error deltas are callwlated
     * (25C or -40C) - (degree celsius, 2's complement). The numbers here are
     * likely to change - check the VBIOS for the actual value being used.
     */
    LwS8  lowTemp;

    /*!
     * Temperature at which high temperature error deltas are callwlated
     * (105C or 125C) - (degree celsius, 2's complement). The numbers here are
     * likely to change - check the VBIOS for the actual value being used.
     */
    LwS8  highTemp;

    /*!
     * Reference temperature at which we do our base ADC calibration at ATE and
     * populate opt_adc_cal (gain, offset and coarse offset) -
     * (degree celsius, 2's complement).
     */
    LwS8  refTemp;

    /*!
     * Low voltage at which the low/high temperature error deltas are callwlated
     * (950 mV). The number is likely to change - check the VBIOS for the actual
     * value being used.
     */
    LwU32 lowVoltuV;

    /*!
     * High voltage at which the low/high temperature error deltas are callwlated
     * (1050 mV). The number is likely to change - check the VBIOS for the actual
     * value being used.
     */
    LwU32 highVoltuV;

    /*!
     * Reference voltage at which we do our base ADC calibration at ATE and
     * populate opt_adc_cal (gain, offset and coarse offset).
     */
    LwU32 refVoltuV;
} LW2080_CTRL_CLK_ADC_DEVICE_INFO_DATA_ISINK_V10;

/*!
 * Structure of static information specific to the ISINK V10 ADC V30 device.
 */
typedef struct LW2080_CTRL_CLK_ADC_DEVICE_INFO_DATA_V30_ISINK_V10 {
    /*!
     * LW2080_CTRL_CLK_ADC_DEVICE_INFO_DATA_V30 super class.
     * This should always be the first member!
     */
    LW2080_CTRL_CLK_ADC_DEVICE_INFO_DATA_V30       super;

    /*!
     * LW2080_CTRL_CLK_ADC_DEVICE_INFO_DATA_ISINK_V10 class.
     */
    LW2080_CTRL_CLK_ADC_DEVICE_INFO_DATA_ISINK_V10 data;
} LW2080_CTRL_CLK_ADC_DEVICE_INFO_DATA_V30_ISINK_V10;

/*!
 * Union of ADC_DEVICE type-specific data.
 */


/*!
 * Structure of static information describing a ADC_DEVICE, which specifies per
 * the ADC Descriptor Table spec a ADC device on the GPU.
 */
typedef struct LW2080_CTRL_CLK_ADC_DEVICE_INFO {
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
     * discriminant.  Placing this redundant type value here until that design
     * constraint can be fixed. @ref LW2080_CTRL_CLK_ADC_TYPE_<xyz>
     */
    LwU8                 type;

    /*!
     * Global ID of device @ref LW2080_CTRL_CLK_ADC_ID_<XYZ>
     */
    LwU8                 id;

    /*!
     * Voltage Domain @ref LW2080_CTRL_VOLT_DOMAIN_<xyz> that the ADC samples
     */
    LwU8                 voltDomain;

    /*!
     * ADC MUX SW override mode of operation set by POR.
     * @ref LW2080_CTRL_CLK_ADC_SW_OVERRIDE_ADC_USE_<abc>
     */
    LwU8                 porOverrideMode;

    /*!
     * Mask of NAFLL devices sharing this ADC device
     */
    LwU32                nafllsSharedMask;

    /*!
     * Set if dynamic calibration needs to be enabled
     */
    LwBool               bDynCal;

    /*!
     * Logical GPC index based API ADC ID of this device.
     * For non-GPC ADC devices, logical and physical IDs are same
     * @ref LW2080_CTRL_CLK_ADC_ID_<XYZ>
     */
    LwU8                 logicalApiId;

    /*!
     * Type-specific information.
     */
    union {
        LW2080_CTRL_CLK_ADC_DEVICE_INFO_DATA_V10           adcV10;

        LW2080_CTRL_CLK_ADC_DEVICE_INFO_DATA_V20           adcV20;

        LW2080_CTRL_CLK_ADC_DEVICE_INFO_DATA_V30           adcV30;

        LW2080_CTRL_CLK_ADC_DEVICE_INFO_DATA_V30_ISINK_V10 adcV30IsinkV10;
    } data;
} LW2080_CTRL_CLK_ADC_DEVICE_INFO;

/*!
 * Structure of static information specific to the V10 ADC devices
 */
typedef struct LW2080_CTRL_CLK_ADC_DEVICES_INFO_DATA_V10 {
    /*!
     * ADC calibration revision contained in the header of the VBIOS ADC
     * Descriptor Table.
     */
    LwU8 calibrationRevVbios;

    /*!
     * Fused ADC calibration revision
     */
    LwU8 calibrationRevFused;
} LW2080_CTRL_CLK_ADC_DEVICES_INFO_DATA_V10;

/*!
 * Structure of static information specific to the V20 ADC devices
 */
typedef struct LW2080_CTRL_CLK_ADC_DEVICES_INFO_DATA_V20 {
    /*!
     * Mask of invalid fuse revisions for the fuse from which we obtain gain, 
     * offset and coarse offset (opt_adc_cal).
     */
    LwU8  calIlwalidFuseRevMask;

    /*!
     * Mask of invalid fuse revisions for the fuse from which we obtain low
     * temperature error deltas (opt_adc_code_err_lt).
     */
    LwU8  lowTempErrIlwalidFuseRevMask;

    /*!
     * Mask of invalid fuse revisions for the fuse from which we obtain high
     * temperature error deltas (opt_adc_code_err_ht).
     */
    LwU8  highTempErrIlwalidFuseRevMask;

    /*!
     * Fused ADC calibration revision - opt_adc_cal_fuse_rev
     */
    LwU8  adcCalRevFused;

    /*!
     * Fused ADC code high temperature error fuse revision
     * opt_adc_code_err_ht_fuse_rev
     */
    LwU8  adcCodeErrHtRevFused;

    /*!
     * Fused ADC code low temperature error fuse revision
     * opt_adc_code_err_lt_fuse_rev
     */
    LwU8  adcCodeErrLtRevFused;

    /*!
     * Index of VFE variable that needs to be used to get the current temperature
     * needed for callwlating ADC code correction offset.
     */
    LwU8  tempVfeVarIdx;

    /*!
     * Temperature at which low temperature error deltas are callwlated
     * (25C or -40C) - (degree celsius, 2's complement). The numbers here are
     * likely to change - check the VBIOS for the actual values being used.
     */
    LwS8  lowTemp;

    /*!
     * Temperature at which high temperature error deltas are callwlated
     * (105C or 125C) - (degree celsius, 2's complement). The numbers here are
     * likely to change - check the VBIOS for the actual values being used.
     */
    LwS8  highTemp;

    /*!
     * Reference temperature at which we do our base ADC calibration at ATE and
     * populate opt_adc_cal (gain, offset and coarse offset) -
     * (degree celsius, 2's complement).
     */
    LwS8  refTemp;

    /*!
     * Minimum cap on ADC code correction offset - 2's complement
     */
    LwS8  adcCodeCorrectionOffsetMin;

    /*!
     * Maximum cap on ADC code correction offset - 2's complement
     */
    LwS8  adcCodeCorrectionOffsetMax;

    /*!
     * Low voltage at which the low/high temperature error deltas are callwlated
     * We expect this to be a constant for all ADCs / calibration schemes
     * with low volt = 850mV. The numbers here are likely to change - check the
     * VBIOS for the actual values being used.
     */
    LwU32 lowVoltuV;

    /*!
     * High voltage at which the low/high temperature error deltas are callwlated
     * We expect this to be a constant for all ADCs / calibration schemes
     * with low volt = 1050mV. The numbers here are likely to change - check the
     * VBIOS for the actual values being used.
     */
    LwU32 highVoltuV;

    /*!
     * Reference voltage at which we do our base ADC calibration at ATE and
     * populate opt_adc_cal (gain, offset and coarse offset).
     */
    LwU32 refVoltuV;
} LW2080_CTRL_CLK_ADC_DEVICES_INFO_DATA_V20;

/*!
 * Union of ADC DEVICES type-specific data.
 */


#define LW2080_CTRL_CLK_ADC_DEVICES_INFO_PARAMS_MESSAGE_ID (0xA0U)

typedef struct LW2080_CTRL_CLK_ADC_DEVICES_INFO_PARAMS {
    /*!
     * [out] LW2080_CTRL_BOARDOBJGRP super class.  Must always be first object
     * in the structure.
     * Returns the mask of valid entries in the ADC Device Descriptor
     * Table. The table may contain disabled entries, but in order for indexes
     * to work correctly, we need to reserve those entries.  The mask helps
     * in this regard.
     */
#endif // (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1) && (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
    LW2080_CTRL_BOARDOBJGRP_E32 super;
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1) && (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

    /*!
     * XAPI does not support using an element within another structure as a
     * discriminant.  Placing this redundant version value here until that
     * design constraint can be fixed. @ref LW2080_CTRL_CLK_ADC_DEVICES_<xyz>
     */
    LwU8                        version;

    /*!
     * [out] Global disable control for all ADCs.
     */
    LwBool                      bAdcIsDisableAllowed;

    /*!
     * Version specific data
     */
    union {
        LW2080_CTRL_CLK_ADC_DEVICES_INFO_DATA_V10 adcsV10;

        LW2080_CTRL_CLK_ADC_DEVICES_INFO_DATA_V20 adcsV20;
    } data;

    /*!
     * [out] An array (of fixed size LW2080_CTRL_CLK_ADC_MAX_DEVICES)
     * describing the individual ADC_DEVICES.  Has valid indexes corresponding
     * to bits set in @ref devMask.
     */
    LW2080_CTRL_CLK_ADC_DEVICE_INFO devices[LW2080_CTRL_CLK_ADC_MAX_DEVICES];
} LW2080_CTRL_CLK_ADC_DEVICES_INFO_PARAMS;

/*!
 * LW2080_CTRL_CMD_CLK_ADC_DEVICES_GET_STATUS
 *
 * This command is used to get all the dynamic status information of all the
 * ADC devices from the PMU
 *
 * Possible status values returned are:
 *   'status' from perfPmuQuery
 */
#define LW2080_CTRL_CMD_CLK_ADC_DEVICES_GET_STATUS (0x208090a1U) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_CLK_LEGACY_NON_PRIVILEGED_INTERFACE_ID << 8) | LW2080_CTRL_CLK_ADC_DEVICES_STATUS_PARAMS_MESSAGE_ID" */

/*!
 * Structure representing ADC SW override voltage request.
 */
typedef struct LW2080_CTRL_CLK_ADC_SW_OVERRIDE_VOLT {
    /*!
     * ADC MUX SW override mode of operation.
     * @ref LW2080_CTRL_CLK_ADC_SW_OVERRIDE_ADC_USE_<abc>
     *
     * ADC will drop the requests if the requested override mode
     * does not match the @ref CLK_ADC_DEVICE::overrideMode.
     * This is implementation to ensure the sanity of ADC/NAFLL
     * HW in different modes of operations.
     */
    LwU8  overrideMode;
    /*!
     * VOLT_RAIL voltage values (uV).
     */
    LwU32 voltageuV;
} LW2080_CTRL_CLK_ADC_SW_OVERRIDE_VOLT;
typedef struct LW2080_CTRL_CLK_ADC_SW_OVERRIDE_VOLT *PLW2080_CTRL_CLK_ADC_SW_OVERRIDE_VOLT;

/*!
 * Structure representing ADC override SW request.
 */
typedef struct LW2080_CTRL_CLK_ADC_SW_OVERRIDE_LIST {
    /*!
     * Mask of VOLT_RAILs to be programmed (1-1 mapped with VOLT_RAIL index).
     * For each volt rail to be programmed, client will set the corresponding
     * bit in this mask and fill-in @ref LW2080_CTRL_CLK_ADC_SW_OVERRIDE_VOLT
     * entry mapped to that index.
     */
#endif // (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1) && (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
    LW2080_CTRL_BOARDOBJGRP_MASK_E32     voltRailsMask;
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1) && (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

    /*!
     * VOLT_RAIL change information corresponding to the mask bit set in
     * @ref voltRailsMask
     */
    LW2080_CTRL_CLK_ADC_SW_OVERRIDE_VOLT volt[LW2080_CTRL_VOLT_VOLT_RAIL_CLIENT_MAX_RAILS];
} LW2080_CTRL_CLK_ADC_SW_OVERRIDE_LIST;
typedef struct LW2080_CTRL_CLK_ADC_SW_OVERRIDE_LIST *PLW2080_CTRL_CLK_ADC_SW_OVERRIDE_LIST;

/*!
 * Structure of status information specific to the V20 ADC device.
 */
typedef struct LW2080_CTRL_CLK_ADC_DEVICE_STATUS_DATA_V20 {
    /*!
     * Offset value for the ADC calibration
     */
    LwS32 offset;
} LW2080_CTRL_CLK_ADC_DEVICE_STATUS_DATA_V20;
typedef struct LW2080_CTRL_CLK_ADC_DEVICE_STATUS_DATA_V20 *PLW2080_CTRL_CLK_ADC_DEVICE_STATUS_DATA_V20;

/*!
 * Structure of status information specific to the V30 ADC device.
 */
typedef struct LW2080_CTRL_CLK_ADC_DEVICE_STATUS_DATA_V30 {
    /*!
     * ADC code correction offset value for ADC runtime calibration
     * This is callwlated/programmed by SW on change in temperature/voltage.
     */
    LwS8   adcCodeCorrectionOffset;

    /*!
     * Temperature coefficient for the ADC device callwlated during init time for
     * temperature based offset correction.
     */
    LwF32  tempCoefficient;

    /*!
     * Volt-temperature coefficient for the ADC device callwlated during init time
     * for (V,T) based gain correction.
     */
    LwF32  voltTempCoefficient;

    /*!
     * Boolean indicating whether RAM Assist lineked to 
     * the given ADC device is engaged or not
     */
    LwBool bRamAssistEngaged;
} LW2080_CTRL_CLK_ADC_DEVICE_STATUS_DATA_V30;
typedef struct LW2080_CTRL_CLK_ADC_DEVICE_STATUS_DATA_V30 *PLW2080_CTRL_CLK_ADC_DEVICE_STATUS_DATA_V30;

/*!
 * Structure of status information specific to the ISINK V10 ADC V30 device.
 */
typedef struct LW2080_CTRL_CLK_ADC_DEVICE_STATUS_DATA_V30_ISINK_V10 {
    /*!
     * LW2080_CTRL_CLK_ADC_DEVICE_STATUS_DATA_V30 super class.
     * This should always be the first member!
     */
    LW2080_CTRL_CLK_ADC_DEVICE_STATUS_DATA_V30 super;
} LW2080_CTRL_CLK_ADC_DEVICE_STATUS_DATA_V30_ISINK_V10;
typedef struct LW2080_CTRL_CLK_ADC_DEVICE_STATUS_DATA_V30_ISINK_V10 *PLW2080_CTRL_CLK_ADC_DEVICE_STATUS_DATA_V30_ISINK_V10;

/*!
 * Union of ADC type-specific status data.
 */


/*!
 * Structure representing the dynamic state associated with a ADC Device.
 */
typedef struct LW2080_CTRL_CLK_ADC_DEVICE_STATUS {
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
     * discriminant.  Placing this redundant type value here until that design
     * constraint can be fixed. @ref LW2080_CTRL_CLK_ADC_TYPE_<xyz>
     */
    LwU8                 type;

    /*!
     * [out] - Current value of the ADC voltage in uV
     */
    LwU32                actualVoltageuV;

    /*!
     * [out] - Current value of the corrected ADC voltage in uV
     */
    LwU32                correctedVoltageuV;

    /*!
     * SW cached value of the actual sampled code (0-79)
     */
    LwU8                 sampledCode;

    /*!
     * SW cached value of the override code (0-79)
     */
    LwU8                 overrideCode;

    /*!
     * Instantaneous ADC code value read from the ADC_MONITOR
     */
    LwU8                 instCode;

    /*!
     * Type-specific information.
     */
    union {
        LW2080_CTRL_CLK_ADC_DEVICE_STATUS_DATA_V20           adcV20;

        LW2080_CTRL_CLK_ADC_DEVICE_STATUS_DATA_V30           adcV30;

        LW2080_CTRL_CLK_ADC_DEVICE_STATUS_DATA_V30_ISINK_V10 adcV30IsinkV10;
    } statusData;
} LW2080_CTRL_CLK_ADC_DEVICE_STATUS;
typedef struct LW2080_CTRL_CLK_ADC_DEVICE_STATUS *PLW2080_CTRL_CLK_ADC_DEVICE_STATUS;

/*!
 * Structure representing the dynamic state associated with the GPU's
 * ADC device functionality.
 */
#define LW2080_CTRL_CLK_ADC_DEVICES_STATUS_PARAMS_MESSAGE_ID (0xA1U)

typedef struct LW2080_CTRL_CLK_ADC_DEVICES_STATUS_PARAMS {
    /*!
     * [in] LW2080_CTRL_BOARDOBJGRP super class.  Must always be first object
     * in the structure.
     * Mask of ADC device entries requested by the client.
     */
#endif // (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1) && (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
    LW2080_CTRL_BOARDOBJGRP_E32       super;
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1) && (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

    /*!
     * [out] - Array of ADC entries. Has valid indexes corresponding to
     * the bits set in @ref devMask.
     */
    LW2080_CTRL_CLK_ADC_DEVICE_STATUS devices[LW2080_CTRL_CLK_ADC_MAX_DEVICES];
} LW2080_CTRL_CLK_ADC_DEVICES_STATUS_PARAMS;

/*!
 * LW2080_CTRL_CMD_CLK_ADC_DEVICES_GET_CONTROL
 *
 * This command is used to get all the client-specified information for
 * all the ADC devices.
 *
 * Possible status values returned are:
 */
#define LW2080_CTRL_CMD_CLK_ADC_DEVICES_GET_CONTROL (0x208090a2U) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_CLK_LEGACY_NON_PRIVILEGED_INTERFACE_ID << 8) | 0xA2" */

/*!
 * Structure representing control parameters specific to the V10 ADC device.
 */
typedef struct LW2080_CTRL_CLK_ADC_DEVICE_CONTROL_DATA_V10 {
    LW2080_CTRL_CLK_ADC_CAL_V10 adcCal;
} LW2080_CTRL_CLK_ADC_DEVICE_CONTROL_DATA_V10;

/*!
 * Union of ADC calibration data - type-specific data.
 */


/*!
 * Structure representing control parameters specific to the V20 ADC device.
 */
typedef struct LW2080_CTRL_CLK_ADC_DEVICE_CONTROL_DATA_V20 {
    /*!
     * XAPI does not support using an element within another structure as a
     * discriminant.  Placing this redundant type value here until that design
     * constraint can be fixed. @ref LW2080_CTRL_CLK_ADC_CAL_TYPE_<xyz>
     */
    LwU8 calType;

    /*!
     * Type-specific information.
     */
    union {
        LW2080_CTRL_CLK_ADC_CAL_V10 calV10;

        LW2080_CTRL_CLK_ADC_CAL_V20 calV20;
    } adcCal;
} LW2080_CTRL_CLK_ADC_DEVICE_CONTROL_DATA_V20;

/*!
 * Structure representing control parameters specific to the V30 ADC device
 */
typedef struct LW2080_CTRL_CLK_ADC_DEVICE_CONTROL_DATA_V30 {
    /*!
     * Offset value for ADC calibration obtained either from VBIOS/fuse
     * (sign-magnitude).
     */
    LwS8 offset;

    /*!
     * Gain value for ADC calibration obtained either from VBIOS/fuse
     * (sign-magnitude).
     */
    LwS8 gain;

    /*!
     * Coarse calibration offset for ADC calibration obtained either from
     * VBIOS/fuse (sign-magnitude).
     */
    LwS8 coarseOffset;

    /*!
     * Coarse calibration gain for ADC calibration obtained either from
     * VBIOS/fuse (sign-magnitude).
     */
    LwS8 coarseGain;

    /*!
     * Low temperature error delta - precal ADC code error @ 850mV obtained
     * from fuse (2's complement).
     */
    LwS8 lowTempLowVoltErr;

    /*!
     * Low temperature error delta - precal ADC code error @ 1050 mV obtained
     * from fuse (2's complement).
     */
    LwS8 lowTempHighVoltErr;

    /*!
     * High temperature error delta - precal ADC code error @ 850mV obtained
     * from fuse (2's complement).
     */
    LwS8 highTempLowVoltErr;

    /*!
     * High temperature error delta - precal ADC code error @ 1050 mV obtained
     * from fuse (2's complement).
     */
    LwS8 highTempHighVoltErr;

    /*!
     * ADC code correction offset value for ADC calibration
     * This is callwlated/programmed by SW on change in temperature/voltage.
     */
    LwS8 adcCodeCorrectionOffset;
} LW2080_CTRL_CLK_ADC_DEVICE_CONTROL_DATA_V30;

/*!
 * Structure representing control parameters specific to the ISINK V10 ADC V30 device.
 */
typedef struct LW2080_CTRL_CLK_ADC_DEVICE_CONTROL_DATA_V30_ISINK_V10 {
    /*!
     * LW2080_CTRL_CLK_ADC_DEVICE_CONTROL_DATA_V30 super class.
     * This should always be the first member!
     */
    LW2080_CTRL_CLK_ADC_DEVICE_CONTROL_DATA_V30 super;
} LW2080_CTRL_CLK_ADC_DEVICE_CONTROL_DATA_V30_ISINK_V10;

/*!
 * Union of type-specific control parameters.
 */


/*!
 * Structure representing the control parameters associated with a ADC_DEVICE.
 */
typedef struct LW2080_CTRL_CLK_ADC_DEVICE_CONTROL {
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
     * discriminant.  Placing this redundant type value here until that design
     * constraint can be fixed. @ref LW2080_CTRL_CLK_ADC_TYPE_<xyz>
     */
    LwU8                 type;

    /*!
     * ADC MUX SW override mode of operation init to POR value
     * during first construct but can be updated by run time code.
     * @ref LW2080_CTRL_CLK_ADC_SW_OVERRIDE_ADC_USE_<abc>
     *
     * Client MUST trigger VF change to program ADC with new
     * override mode and voltage values.
     */
    LwU8                 overrideMode;

    /*!
     * Type-specific information.
     */
    union {
        LW2080_CTRL_CLK_ADC_DEVICE_CONTROL_DATA_V10           adcV10;

        LW2080_CTRL_CLK_ADC_DEVICE_CONTROL_DATA_V20           adcV20;

        LW2080_CTRL_CLK_ADC_DEVICE_CONTROL_DATA_V30           adcV30;

        LW2080_CTRL_CLK_ADC_DEVICE_CONTROL_DATA_V30_ISINK_V10 adcV30IsinkV10;
    } data;
} LW2080_CTRL_CLK_ADC_DEVICE_CONTROL;

/*!
 * Structure representing the control parameters associated with the GPU's
 * ADC Devices functionality.
 */
typedef struct LW2080_CTRL_CLK_ADC_DEVICES_CONTROL_PARAMS {
    /*!
     * LW2080_CTRL_BOARDOBJ super class.  Must always be first object in
     * structure.
     */
#endif // (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1) && (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
    LW2080_CTRL_BOARDOBJGRP_E32        super;
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1) && (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

    /*!
     * [in] Flag specifying the set of values to retrieve:
     * - VBIOS default (LW_TRUE)
     * - lwrrently active (LW_FALSE)
     */
    LwBool                             bDefault;

    /*!
     * [in out] - Array of ADC_DEVICE entries. Has valid indexes corresponding to
     * the bits set in @ref devMask.
     */
    LW2080_CTRL_CLK_ADC_DEVICE_CONTROL devices[LW2080_CTRL_CLK_ADC_MAX_DEVICES];
} LW2080_CTRL_CLK_ADC_DEVICES_CONTROL_PARAMS;

/*!
 * LW2080_CTRL_CMD_CLK_NAFLL_DEVICES_GET_INFO
 *
 * This command is used to get all the static information of all the NAFLL
 * present on the GPU
 */
#define LW2080_CTRL_CMD_CLK_NAFLL_DEVICES_GET_INFO (0x208090b0U) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_CLK_LEGACY_NON_PRIVILEGED_INTERFACE_ID << 8) | LW2080_CTRL_CLK_NAFLL_DEVICES_INFO_PARAMS_MESSAGE_ID" */

/*!
 * Structure of static information specific to the V10 NAFLL device.
 */
typedef struct LW2080_CTRL_CLK_NAFLL_DEVICE_INFO_DATA_V10 {
    LwU32 placeholder;
} LW2080_CTRL_CLK_NAFLL_DEVICE_INFO_DATA_V10;

/*!
 * Structure of static information specific to the V20 NAFLL device.
 */
typedef struct LW2080_CTRL_CLK_NAFLL_DEVICE_INFO_DATA_V20 {
    LwU32 placeholder;
} LW2080_CTRL_CLK_NAFLL_DEVICE_INFO_DATA_V20;

/*!
 * Structure of static information specific to the V30 NAFLL device.
 */
typedef struct LW2080_CTRL_CLK_NAFLL_DEVICE_INFO_DATA_V30 {
    LwU32 placeholder;
} LW2080_CTRL_CLK_NAFLL_DEVICE_INFO_DATA_V30;

/*!
 * Union of NAFLL_DEVICE type-specific data.
 */


/*!
 * Structure of static information describing a NAFLL_DEVICE, which specifies per
 * the NAFLL Descriptor Table spec a NAFLL device on the GPU.
 */
typedef struct LW2080_CTRL_CLK_NAFLL_DEVICE_INFO {
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
     * discriminant.  Placing this redundant type value here until that design
     * constraint can be fixed. @ref LW2080_CTRL_CLK_NAFLL_DEVICE_TYPE_<xyz>
     */
    LwU8                 type;

    /*!
     * Global ID of device @ref LW2080_CTRL_CLK_NAFLL_ID_<XYZ>
     */
    LwU8                 id;

    /*!
     * M-Divider value for this NAFLL device
     */
    LwU8                 mdiv;

    /*!
     * Flag to indicate whether to skip setting Pldiv below DVCO min or not.
     */
    LwBool               bSkipPldivBelowDvcoMin;

    /*!
     * The mode value determined by the LUT Vselect Mux
     * @ref LW2080_CTRL_CLK_NAFLL_LUT_VSELECT_<XYZ>
     */
    LwU8                 vselectMode;

    /*!
     * Input pre-divider reference clock frequency for this NAFLL in MHz.
     * Input frequency for this NAFLL device = @ref inputRefClkMHz / @ref inputRefClkDivVal.
     *
     * To-do akshatam: Today we get NAFLL input frequency from the VBIOS. This is
     * an overhead and not scalable when it comes to input frequency changes such
     * as Bug 3014997. We should look at having PMU being self-sufficient for
     * reading NAFLL input clock just like what we do for other clocks.
     */
    LwU16                inputRefClkFreqMHz;

    /*!
     * REF_CLK divider value for this NAFLL device.
     */
    LwU16                inputRefClkDivVal;

    /*!
     * The clock domain @ref LW2080_CTRL_CLK_DOMAIN_<xyz>. associated with
     * this NAFLL device
     */
    LwU32                clkDomain;

    /*!
     * Index of Logic ADC device that feeds this NAFLL
     */
    LwU8                 adcIdxLogic;

    /*!
     * Index of SRAM ADC device that feeds this NAFLL
     */
    LwU8                 adcIdxSram;

    /*!
     * VFE Equation Index to provide min freq supported by DVCO
     */
    LwU8                 dvcoMinFreqVFEIdx;

    /*!
     * Index into the frequency controller table
     */
    LwU8                 freqCtrlIdx;

    /*!
     * The ADC code threshold for hysteresis
     */
    LwU16                hysteresisThreshold;

    /*!
     * The fixed frequency regime limit in MHz
     */
    LwU16                fixedFreqRegimeLimitMHz;

    /*!
     * Flag to indicate whether 1x DVCO is in use. TRUE means 1x DVCO,
     * otherwise 2x DVCO.
     */
    LwBool               bDvco1x;

    /*!
     * Type-specific information.
     */
    union {
        LW2080_CTRL_CLK_NAFLL_DEVICE_INFO_DATA_V10 V10;
        LW2080_CTRL_CLK_NAFLL_DEVICE_INFO_DATA_V20 V20;
        LW2080_CTRL_CLK_NAFLL_DEVICE_INFO_DATA_V30 V30;
    } data;
} LW2080_CTRL_CLK_NAFLL_DEVICE_INFO;

#define LW2080_CTRL_CLK_NAFLL_DEVICES_INFO_PARAMS_MESSAGE_ID (0xB0U)

typedef struct LW2080_CTRL_CLK_NAFLL_DEVICES_INFO_PARAMS {
    /*!
     * [out] LW2080_CTRL_BOARDOBJGRP super class.  Must always be first object in
     * structure.
     * Returns the mask of valid entries in the NAFLL Device Descriptor
     * Table. The table may contain disabled entries, but in order for indexes
     * to work correctly, we need to reserve those entries.  The mask helps
     * in this regard.
     */
#endif // (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1) && (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
    LW2080_CTRL_BOARDOBJGRP_E32       super;
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1) && (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

   /*!
    * The worst case DVCO min frequency across pstate Vmin lwrve for all
    * NAFLL devices.
    */
    LwU16                             maxDvcoMinFreqMHz;

    /*!
     * [out] - Reserved for future use
     */
    LwU8                              rsvd;

   /*!
    * Total number of entries that each table in the LUT can hold. This is
    * common to all NAFLL devices
    */
    LwU8                              lutNumEntries;

    /*!
    * The step size for the LUT entries. This is common to all NAFLL devices
    */
    LwU32                             lutStepSizeuV;

    /*!
    * The minimum voltage in uV that's present in the LUT. i.e. the base
    * "uncalibrated voltage" corresponding to code 0 in the LUT
    */
    LwU32                             lutMilwoltageuV;

    /*!
     * [out] An array (of fixed size LW2080_CTRL_CLK_NAFLL_MAX_DEVICES)
     * describing the individual NAFLL_DEVICES.  Has valid indexes corresponding
     * to bits set in @ref devMask.
     */
    LW2080_CTRL_CLK_NAFLL_DEVICE_INFO devices[LW2080_CTRL_CLK_NAFLL_MAX_DEVICES];
} LW2080_CTRL_CLK_NAFLL_DEVICES_INFO_PARAMS;

/*!
 * LW2080_CTRL_CMD_CLK_NAFLL_DEVICES_GET_STATUS
 *
 * This command is used to get all the dynamic status information of all the
 * NAFLL devices from the PMU
 *
 * Possible status values returned are:
 *   'status' from perfPmuQuery
 */
#define LW2080_CTRL_CMD_CLK_NAFLL_DEVICES_GET_STATUS    (0x208090b1U) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_CLK_LEGACY_NON_PRIVILEGED_INTERFACE_ID << 8) | LW2080_CTRL_CLK_NAFLL_DEVICES_STATUS_PARAMS_MESSAGE_ID" */

/*!
 * The minimum voltage in uV that's present in the LUT. This value came from
 * the safe V/F lwrve given by the CS team.
 */
#define LW2080_CTRL_CLK_LUT_MIN_VOLTAGE_UV              (450000U)

/*!
 * The maximum voltage in uV that's present in the LUT. This value came from
 * the safe V/F lwrve given by the CS team.
 */
#define LW2080_CTRL_CLK_LUT_MAX_VOLTAGE_UV              (1250000U)

/*!
 * Macros listing valid ADC step sizes.
 */
#define LW2080_CTRL_CLK_ADC_STEP_SIZE_6250UV            (6250U)
#define LW2080_CTRL_CLK_ADC_STEP_SIZE_10000UV           (10000U)

/*!
 * Max number of entries for a LUT
 */
#define LW2080_CTRL_CLK_LUT_NUM_ENTRIES_MAX             (128U)

/*!
 * The minimum voltage in uV that is modelled on EMU.
 */
#define LW2080_CTRL_CLK_EMU_SUPPORTED_MIN_VOLTAGE_UV    (560000U)

/*!
 * The minimum frequency in MHz that is supported on EMU.
 */
#define LW2080_CTRL_CLK_EMU_SUPPORTED_MIN_FREQUENCY_MHZ (900U)

/*!
 * The maximum frequency in MHz that is supported on EMU.
 */
#define LW2080_CTRL_CLK_EMU_SUPPORTED_MAX_FREQUENCY_MHZ (3200U)

/*!
 * Struct describing each V/F entry value as programmed in the LUT V10
 */
typedef struct LW2080_CTRL_CLK_LUT_10_VF_ENTRY {
    /*!
     * NDIV.
     */
    LwU16 ndiv;

    /*!
     * VFGAIN.
     */
    LwU16 vfgain;
} LW2080_CTRL_CLK_LUT_10_VF_ENTRY;
typedef struct LW2080_CTRL_CLK_LUT_10_VF_ENTRY *PLW2080_CTRL_CLK_LUT_10_VF_ENTRY;

/*!
 * Struct describing each V/F entry value as programmed in the LUT V20
 */
typedef struct LW2080_CTRL_CLK_LUT_20_VF_ENTRY {
    /*!
     * NDIV.
     */
    LwU16 ndiv;

    /*!
     * NDIV Offset.
     */
    LwU8  ndivOffset;

    /*!
     * DVCO Offset.
     */
    LwU8  dvcoOffset;
} LW2080_CTRL_CLK_LUT_20_VF_ENTRY;
typedef struct LW2080_CTRL_CLK_LUT_20_VF_ENTRY *PLW2080_CTRL_CLK_LUT_20_VF_ENTRY;

/*!
 * Struct describing each V/F entry value as programmed in the LUT V30
 */
typedef struct LW2080_CTRL_CLK_LUT_30_VF_ENTRY {
    /*!
     * NDIV.
     */
    LwU16 ndiv;

    /*!
     * NDIV Offset.
     */
    LwU8  ndivOffset[LW2080_CTRL_CLK_NAFLL_LUT_VF_LWRVE_SEC_MAX];

    /*!
     * DVCO Offset.
     */
    LwU8  dvcoOffset[LW2080_CTRL_CLK_NAFLL_LUT_VF_LWRVE_SEC_MAX];

    /*!
     * CPM Max NDIV offset
     */
    LwU16 cpmMaxNdivOffset;
} LW2080_CTRL_CLK_LUT_30_VF_ENTRY;
typedef struct LW2080_CTRL_CLK_LUT_30_VF_ENTRY *PLW2080_CTRL_CLK_LUT_30_VF_ENTRY;

/*!
 * Union describing each V/F entry value as programmed in the LUT
 */
typedef union LW2080_CTRL_CLK_LUT_VF_ENTRY_DATA {
        /*!
         * DWORD to hold all LUT params.
         */
    LwU32                           value;
    LW2080_CTRL_CLK_LUT_10_VF_ENTRY lut10;
    LW2080_CTRL_CLK_LUT_20_VF_ENTRY lut20;
    LW2080_CTRL_CLK_LUT_30_VF_ENTRY lut30;
} LW2080_CTRL_CLK_LUT_VF_ENTRY_DATA;

typedef union LW2080_CTRL_CLK_LUT_VF_ENTRY_DATA *PLW2080_CTRL_CLK_LUT_VF_ENTRY_DATA;

/*!
 * Struct describing each V/F entry value as programmed in the LUT
 */
typedef struct LW2080_CTRL_CLK_LUT_VF_ENTRY {
    /*!
     * NAFLL Device type. @ref LW2080_CTRL_CLK_NAFLL_DEVICE_TYPE_<xyz>
     */
    LwU8                              type;

    /*!
     * version-specific data union.
     */
    LW2080_CTRL_CLK_LUT_VF_ENTRY_DATA data;
} LW2080_CTRL_CLK_LUT_VF_ENTRY;
typedef struct LW2080_CTRL_CLK_LUT_VF_ENTRY *PLW2080_CTRL_CLK_LUT_VF_ENTRY;

/*!
 * Structure to describe the per-LUT VF surface/lwrve
 */
typedef struct LW2080_CTRL_CLK_LUT_VF_LWRVE {
    /*!
     * Array of LUT VF entries.
     */
    LW2080_CTRL_CLK_LUT_VF_ENTRY lutVfEntries[LW2080_CTRL_CLK_LUT_NUM_ENTRIES_MAX];
} LW2080_CTRL_CLK_LUT_VF_LWRVE;
typedef struct LW2080_CTRL_CLK_LUT_VF_LWRVE *PLW2080_CTRL_CLK_LUT_VF_LWRVE;

/*!
 * Structure representing the dynamic state associated with a NAFLL Device.
 */
typedef struct LW2080_CTRL_CLK_NAFLL_DEVICE_STATUS {
    /*!
     * LW2080_CTRL_BOARDOBJ super class.  Must always be first object in
     * structure.
     */
#endif // (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1) && (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
    LW2080_CTRL_BOARDOBJ         super;
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1) && (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

    /*!
     * XAPI does not support using an element within another structure as a
     * discriminant.  Placing this redundant type value here until that design
     * constraint can be fixed. @ref LW2080_CTRL_CLK_NAFLL_DEVICE_TYPE_<xyz>
     */
    LwU8                         type;

    /*!
     * Lwrrently set regime id for the NAFLL device
     */
    LwU8                         lwrrentRegimeId;

    /*!
     * Current DVCO min frequency (in MHz) for the NAFLL device
     */
    LwU16                        dvcoMinFreqMHz;

    /*!
     * Flag indicating if the DVCO min was reached for the NAFLL device
     */
    LwBool                       bDvcoMinReached;

    /*!
     * Reserved padding for future use
     */
    LwU8                         reserved[2];

    /*!
     * The mode value determined by the SW Override Mux.
     */
    LwU8                         swOverrideMode;

    /*!
     * Cache SW override LUT entry value.
     */
    LW2080_CTRL_CLK_LUT_VF_ENTRY swOverride;

    /*!
     * Lwrrently programmed LUT VF lwrve for the NAFLL device
     */
    LW2080_CTRL_CLK_LUT_VF_LWRVE lutVfLwrve;
} LW2080_CTRL_CLK_NAFLL_DEVICE_STATUS;
typedef struct LW2080_CTRL_CLK_NAFLL_DEVICE_STATUS *PLW2080_CTRL_CLK_NAFLL_DEVICE_STATUS;

/*!
 * Structure representing the dynamic state associated with the GPU's
 * NAFLL device functionality.
 */
#define LW2080_CTRL_CLK_NAFLL_DEVICES_STATUS_PARAMS_MESSAGE_ID (0xB1U)

typedef struct LW2080_CTRL_CLK_NAFLL_DEVICES_STATUS_PARAMS {
    /*!
     * [in] LW2080_CTRL_BOARDOBJGRP super class.  Must always be first object
     * in the structure.
     * Mask of NAFLL device entries requested by the client.
     */
#endif // (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1) && (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
    LW2080_CTRL_BOARDOBJGRP_E32         super;
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1) && (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

    /*!
     * [out] - Array of NAFLL entries. Has valid indexes corresponding to
     * the bits set in @ref devMask.
     */
    LW2080_CTRL_CLK_NAFLL_DEVICE_STATUS devices[LW2080_CTRL_CLK_NAFLL_MAX_DEVICES];
} LW2080_CTRL_CLK_NAFLL_DEVICES_STATUS_PARAMS;


/*!
 * Structure describing NAFLL_DEVICES_V10 specific control parameters.
 */
typedef struct LW2080_CTRL_CLK_NAFLL_DEVICES_CONTROL_DATA_V10 {
    LwU32 placeholder;
} LW2080_CTRL_CLK_NAFLL_DEVICES_CONTROL_DATA_V10;
typedef struct LW2080_CTRL_CLK_NAFLL_DEVICES_CONTROL_DATA_V10 *PLW2080_CTRL_CLK_NAFLL_DEVICES_CONTROL_DATA_V10;

/*!
 * Structure describing NAFLL_DEVICES_V20 specific control parameters.
 */
typedef struct LW2080_CTRL_CLK_NAFLL_DEVICES_CONTROL_DATA_V20 {
    LwU32 placeholder;
} LW2080_CTRL_CLK_NAFLL_DEVICES_CONTROL_DATA_V20;
typedef struct LW2080_CTRL_CLK_NAFLL_DEVICES_CONTROL_DATA_V20 *PLW2080_CTRL_CLK_NAFLL_DEVICES_CONTROL_DATA_V20;

/*!
 * Structure describing NAFLL_DEVICES_V30 specific control parameters.
 */
typedef struct LW2080_CTRL_CLK_NAFLL_DEVICES_CONTROL_DATA_V30 {
    /*!
     * Array of booleans per VF lwrve to request force engage/disengage of the
     * quick slowdown feature
     */
    LwBool bQuickSlowdownForceEngage[LW2080_CTRL_CLK_NAFLL_LUT_VF_LWRVE_SEC_MAX];
} LW2080_CTRL_CLK_NAFLL_DEVICES_CONTROL_DATA_V30;
typedef struct LW2080_CTRL_CLK_NAFLL_DEVICES_CONTROL_DATA_V30 *PLW2080_CTRL_CLK_NAFLL_DEVICES_CONTROL_DATA_V30;

/*!
 * CLK_FREQ_CONTROLLER type-specific data union.  Discriminated by
 * CLK_FREQ_CONTROLLER::super.type.
 */


/*!
 * Structure representing the control parameters associated with a CLK_VF_POINT.
 */
typedef struct LW2080_CTRL_CLK_NAFLL_DEVICES_CONTROL {
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
     * discriminant.  Placing this redundant type value here until that design
     * constraint can be fixed.
     */
    LwU8                 type;

    /*!
     * Override value for the target regime Id
     */
    LwU8                 targetRegimeIdOverride;

    /*!
     * Type-specific data union.
     */
    union {
        LW2080_CTRL_CLK_NAFLL_DEVICES_CONTROL_DATA_V10 v10;
        LW2080_CTRL_CLK_NAFLL_DEVICES_CONTROL_DATA_V20 v20;
        LW2080_CTRL_CLK_NAFLL_DEVICES_CONTROL_DATA_V30 v30;
    } data;
} LW2080_CTRL_CLK_NAFLL_DEVICES_CONTROL;
typedef struct LW2080_CTRL_CLK_NAFLL_DEVICES_CONTROL *PLW2080_CTRL_CLK_NAFLL_DEVICES_CONTROL;

/*!
 * Structure representing the control parameters associated with a NAFLL_DEVICES.
 */
typedef struct LW2080_CTRL_CLK_NAFLL_DEVICES_CONTROL_PARAMS {
    /*!
     * LW2080_CTRL_BOARDOBJGRP super class.  Must always be first object in
     * structure.
     */
#endif // (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1) && (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
    LW2080_CTRL_BOARDOBJGRP_E32           super;
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1) && (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

    /*!
     * Array of NAFLL_DEVICES structures.  Has valid indexes corresponding to the
     * bits set in @ref super.objMask.
     */
    LW2080_CTRL_CLK_NAFLL_DEVICES_CONTROL devices[LW2080_CTRL_CLK_NAFLL_MAX_DEVICES];
} LW2080_CTRL_CLK_NAFLL_DEVICES_CONTROL_PARAMS;
typedef struct LW2080_CTRL_CLK_NAFLL_DEVICES_CONTROL_PARAMS *PLW2080_CTRL_CLK_NAFLL_DEVICES_CONTROL_PARAMS;

/*!
 * LW2080_CTRL_CMD_CLK_NAFLL_DEVICES_GET_CONTROL
 *
 * This command returns NAFLL_DEVICES control parameters as specified by the
 * VBIOS in the NAFLL Device Table
 *
 * See @ref LW2080_CTRL_CMD_CLK_NAFLL_DEVICES_CONTROL_PARAMS for
 * documentation on the parameters.
 *
 * Return values are specified per @ref BoardObjGrpGetControl.
 */
#define LW2080_CTRL_CMD_CLK_NAFLL_DEVICES_GET_CONTROL (0x208090b2U) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_CLK_LEGACY_NON_PRIVILEGED_INTERFACE_ID << 8) | 0xB2" */

/* _ctrl2080clk_h_ */

#endif // (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

