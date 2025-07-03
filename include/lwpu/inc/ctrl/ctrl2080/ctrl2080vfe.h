/*
 * SPDX-FileCopyrightText: Copyright (c) 2015-2020 LWPU CORPORATION & AFFILIATES. All rights reserved.
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
// Source file: ctrl/ctrl2080/ctrl2080vfe.finn
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
#include "ctrl/ctrl2080/ctrl2080bios.h"

/* --------------------------- Forward Defines ----------------------------- */
/*!
 * Type to be used for all VFE equation indices.
 * Intended for use in RMCTRL structures, RM code and data structures,
 * and RM-PMU shared structures only.
 *
 * @ref LW2080_CTRL_PERF_VFE_EQU_IDX
 *
 * PMU will have its own VFE index type @ref LwVfeEquIdx
 *
 * @note    This type is lwrrently typedef-ed to LwU8, which is the same
 *          as current VFE equation indices. However, once VFE internals are
 *          moved to 16-bit indices and >255 VFE entries, this will be
 *          typedefed to LwU16 (through LwBoardObjIdx)
 *
 * @note    This is lwrrently placed in ctrl2080boardobj.finn/h due to
 *          cirlwlar dependency in preprocessor header files issues.
 *          If and when that is resolved, the define will be brought here.
 *
 * */

/* --------------------------- VFE Variable -------------------------------- */

/*!
 * @defgroup LW2080_CTRL_PERF_VFE_VAR_TYPE_ENUM
 *
 * Enumeration of VFE variable types.
 *
 * @{
 */
#define LW2080_CTRL_PERF_VFE_VAR_TYPE_BASE                      0x00U
#define LW2080_CTRL_PERF_VFE_VAR_TYPE_DERIVED                   0x01U
#define LW2080_CTRL_PERF_VFE_VAR_TYPE_DERIVED_PRODUCT           0x02U
#define LW2080_CTRL_PERF_VFE_VAR_TYPE_DERIVED_SUM               0x03U
#define LW2080_CTRL_PERF_VFE_VAR_TYPE_SINGLE                    0x04U
#define LW2080_CTRL_PERF_VFE_VAR_TYPE_SINGLE_FREQUENCY          0x05U
#define LW2080_CTRL_PERF_VFE_VAR_TYPE_SINGLE_SENSED             0x06U
#define LW2080_CTRL_PERF_VFE_VAR_TYPE_SINGLE_SENSED_FUSE        0x07U
#define LW2080_CTRL_PERF_VFE_VAR_TYPE_SINGLE_SENSED_TEMP        0x08U
#define LW2080_CTRL_PERF_VFE_VAR_TYPE_SINGLE_VOLTAGE            0x09U
#define LW2080_CTRL_PERF_VFE_VAR_TYPE_SINGLE_CALLER_SPECIFIED   0x0AU
#define LW2080_CTRL_PERF_VFE_VAR_TYPE_SINGLE_GLOBALLY_SPECIFIED 0x0BU
#define LW2080_CTRL_PERF_VFE_VAR_TYPE_SINGLE_SENSED_FUSE_BASE   0x0LW
#define LW2080_CTRL_PERF_VFE_VAR_TYPE_SINGLE_SENSED_FUSE_20     0x0DU

// Insert new types here and increment _MAX
#define LW2080_CTRL_PERF_VFE_VAR_TYPE_MAX                       0x0DU
#define LW2080_CTRL_PERF_VFE_VAR_TYPE_ILWALID                   0xFFU
/*!@}*/

/*!
 * @defgroup LW2080_CTRL_PERF_VFE_VAR_SINGLE_OVERRIDE_TYPE_ENUM
 *
 * Enumeration of RmCtrl VFE variable overrides. This enumeration specifies the
 * behavior of the @ref RM_PMU_VFE_VAR_SINGLE::override "override" value found
 * in @ref RM_PMU_VFE_VAR_SINGLE.
 *
 * @{
 */
/*!
 * The override value does nothing and is ignored.
 */
#define LW2080_CTRL_PERF_VFE_VAR_SINGLE_OVERRIDE_TYPE_NONE      0x00U
/*!
 * The override value replaces the actual value of the variable.
 */
#define LW2080_CTRL_PERF_VFE_VAR_SINGLE_OVERRIDE_TYPE_VALUE     0x01U
/*!
 * The override value supplies an offset to the actual value. The override value
 * is added to the actual value to produce a new result.
 */
#define LW2080_CTRL_PERF_VFE_VAR_SINGLE_OVERRIDE_TYPE_OFFSET    0x02U
/*!
 * The override value supplies a scalar to the actual value. The override value
 * is multiplied to the actual value to produce a new result.
 */
#define LW2080_CTRL_PERF_VFE_VAR_SINGLE_OVERRIDE_TYPE_SCALE     0x03U
/*!@}*/

/*!
 * Macro for indicating that a VFE Variable index is invalid.
 */
#define LW2080_CTRL_PERF_VFE_VAR_INDEX_ILWALID                  LW2080_CTRL_BOARDOBJ_IDX_ILWALID_8BIT

/*!
 * Structure describing VFE_VAR_DERIVED static information/POR.
 */
typedef struct LW2080_CTRL_PERF_VFE_VAR_INFO_DERIVED {
    /*!
     * Lwrrently we do NOT have any static info parameter in this class.
     */
    LwU8 rsvd;
} LW2080_CTRL_PERF_VFE_VAR_INFO_DERIVED;

/*!
 * Structure describing VFE_VAR_DERIVED_PRODUCT static information/POR.
 */
typedef struct LW2080_CTRL_PERF_VFE_VAR_INFO_DERIVED_PRODUCT {
    /*!
     * VFE_VAR_DERIVED super class. Must always be first element in the structure.
     */
    LW2080_CTRL_PERF_VFE_VAR_INFO_DERIVED super;
    /*!
     * Index of the first variable used by product().
     */
    LwU8                                  varIdx0;
    /*!
     * Index of the second variable used by product().
     */
    LwU8                                  varIdx1;
} LW2080_CTRL_PERF_VFE_VAR_INFO_DERIVED_PRODUCT;

/*!
 * Structure describing VFE_VAR_DERIVED_SUM static information/POR.
 */
typedef struct LW2080_CTRL_PERF_VFE_VAR_INFO_DERIVED_SUM {
    /*!
     * VFE_VAR_DERIVED super class. Must always be first element in the structure.
     */
    LW2080_CTRL_PERF_VFE_VAR_INFO_DERIVED super;
    /*!
     * Index of the first variable used by sum().
     */
    LwU8                                  varIdx0;
    /*!
     * Index of the second variable used by sum().
     */
    LwU8                                  varIdx1;
} LW2080_CTRL_PERF_VFE_VAR_INFO_DERIVED_SUM;

/*!
 * Structure describing VFE_VAR_SINGLE static information/POR.
 */
typedef struct LW2080_CTRL_PERF_VFE_VAR_INFO_SINGLE {
    /*!
     * Lwrrently we do NOT have any static info parameter in this class.
     */
    LwU8 rsvd;
} LW2080_CTRL_PERF_VFE_VAR_INFO_SINGLE;

/*!
 * Structure describing VFE_VAR_SINGLE_FREQUENCY static information/POR.
 */
typedef struct LW2080_CTRL_PERF_VFE_VAR_INFO_SINGLE_FREQUENCY {
    /*!
     * VFE_VAR_SINGLE super class. Must always be first element in the structure.
     */
    LW2080_CTRL_PERF_VFE_VAR_INFO_SINGLE super;

    /*!
     * Clock domain index into the CLK_DOMAINS BOARDOBJGRP.
     * The special value of @ref LW2080_CTRL_CLK_CLK_DOMAIN_INDEX_ILWALID
     * indicates the domain index is not available for this frequency.
     */
    LwU8                                 clkDomainIdx;
} LW2080_CTRL_PERF_VFE_VAR_INFO_SINGLE_FREQUENCY;

/*!
 * Macros for VFE_VAR_SINGLE_CALLER_SPECIFIED Unique ID defines.
 */
#define LW2080_CTRL_PERF_VFE_VAR_CALLER_SPECIFIED_UID_WORK_TYPE  0x0U
#define LW2080_CTRL_PERF_VFE_VAR_CALLER_SPECIFIED_UID_UTIL_RATIO 0x1U

// Range for the unique ID, always the last ID
#define LW2080_CTRL_PERF_VFE_VAR_CALLER_SPECIFIED_UID_MAX        0x1U

/*!
 * Structure describing VFE_VAR_SINGLE_CALLER_SPECIFIED static information/POR.
 */
typedef struct LW2080_CTRL_PERF_VFE_VAR_INFO_SINGLE_CALLER_SPECIFIED {
    /*!
     * VFE_VAR_SINGLE super class. Must always be first element in the structure.
     */
    LW2080_CTRL_PERF_VFE_VAR_INFO_SINGLE super;

    /*!
     * Unique Identification for the generic caller specified class.
     */
    LwU8                                 uniqueId;
} LW2080_CTRL_PERF_VFE_VAR_INFO_SINGLE_CALLER_SPECIFIED;

/*!
 * Macros for VFE_VAR_SINGLE_GLOBALLY_SPECIFIED Unique ID defines.
 * @defgroup LW2080_CTRL_VOLT_VOLT_DOMAIN_ENUM
 * @{
 */
#define LW2080_CTRL_PERF_VFE_VAR_SINGLE_GLOBALLY_SPECIFIED_UID_PERFORMANCE_MODE 0x0U

// Range for the unique ID, always the last ID
#define LW2080_CTRL_PERF_VFE_VAR_SINGLE_GLOBALLY_SPECIFIED_UID_MAX              0x1U
/*! @} */

/*!
 * Structure describing VFE_VAR_SINGLE_GLOBALLY_SPECIFIED static information/POR.
 */
typedef struct LW2080_CTRL_PERF_VFE_VAR_INFO_SINGLE_GLOBALLY_SPECIFIED {
    /*!
     * VFE_VAR_SINGLE super class. Must always be first element in the structure.
     */
    LW2080_CTRL_PERF_VFE_VAR_INFO_SINGLE super;

    /*!
     * Unique Identification for the generic caller specified class.
     * @ref LW2080_CTRL_PERF_VFE_VAR_SINGLE_GLOBALLY_SPECIFIED_UID_<xyz>
     */
    LwU8                                 uniqueId;

    /*!
     * Number of fractional bits in @ref valDefault.
     */
    LwU8                                 numFracBits;
    /*!
     * Default value specified by POR.
     */
    LwS32                                valDefault;
} LW2080_CTRL_PERF_VFE_VAR_INFO_SINGLE_GLOBALLY_SPECIFIED;

/*!
 * Structure describing VFE_VAR_SINGLE_SENSED static information/POR.
 */
typedef struct LW2080_CTRL_PERF_VFE_VAR_INFO_SINGLE_SENSED {
    /*!
     * VFE_VAR_SINGLE super class. Must always be first element in the structure.
     */
    LW2080_CTRL_PERF_VFE_VAR_INFO_SINGLE super;
} LW2080_CTRL_PERF_VFE_VAR_INFO_SINGLE_SENSED;

/*!
 * Max number of registers needed to address/define a single fuse from the VF Table.
 */
#define LW2080_CTRL_PERF_VFE_VAR_SINGLE_SENSED_FUSE_SEGMENTS_MAX 3

/*!
 * struct with all the information needed to read a fuse
 */
typedef struct LW2080_CTRL_PERF_VFE_VAR_SINGLE_SENSED_FUSE_INFO {
    /*!
     * fuse segment count
     */
    LwU8                                     segmentCount;
    /*!
     * List of segments that make up the fuse, lwrrently we support just one
     * segment per fuse register
     */
    LW2080_CTRL_BIOS_VFIELD_REGISTER_SEGMENT segments[LW2080_CTRL_PERF_VFE_VAR_SINGLE_SENSED_FUSE_SEGMENTS_MAX];
} LW2080_CTRL_PERF_VFE_VAR_SINGLE_SENSED_FUSE_INFO;
typedef struct LW2080_CTRL_PERF_VFE_VAR_SINGLE_SENSED_FUSE_INFO *PLW2080_CTRL_PERF_VFE_VAR_SINGLE_SENSED_FUSE_INFO;

/*!
 * struct with Fuse value override info
 */
typedef struct LW2080_CTRL_PERF_VFE_VAR_SINGLE_SENSED_FUSE_OVERRIDE_INFO {
    /*!
     * Fuse value to be set when overridden by regkey.
     */
    LwU32  fuseValOverride;
    /*!
     * Set to true if the fuse value is overridden by regkey. Override value is in fuseValOverride
     */
    LwBool bFuseRegkeyOverride;
} LW2080_CTRL_PERF_VFE_VAR_SINGLE_SENSED_FUSE_OVERRIDE_INFO;

/*!
 * Union for Fuse Value
 */


typedef struct LW2080_CTRL_PERF_VFE_VAR_SINGLE_SENSED_FUSE_VALUE {
    /*!
     * Boolean flag indicating the specific type in
     * LW2080_CTRL_PERF_VFE_VAR_SINGLE_SENSED_FUSE_VALUE (singedFuseValue or
     * unsignedFuseValue)
     */
    LwBool bSigned;
    /*!
     * Type-specific data union.
     */
    union {
        /*!
         * Fuse value when its signed integer
         */
        LwS32 signedValue;
        /*!
         * Fuse value when its unsigned integer
         */
        LwU32 unsignedValue;
    } data;
} LW2080_CTRL_PERF_VFE_VAR_SINGLE_SENSED_FUSE_VALUE;
typedef struct LW2080_CTRL_PERF_VFE_VAR_SINGLE_SENSED_FUSE_VALUE *PLW2080_CTRL_PERF_VFE_VAR_SINGLE_SENSED_FUSE_VALUE;

/*!
 * Helper macros to init fuse value
 *
 * @param[in] pVar
 *     PLW2080_CTRL_PERF_VFE_VAR_SINGLE_SENSED_FUSE_VALUE object in which the
 *     fuse value will be assigned.
 * @param[in] value
 *     Fuse value to be saved in pVar
 * @param[in] bFuseValSigned
 *     Boolean flag indicating if the fuse value is signed on unsigned
 */
#define LW2080_CTRL_PERF_VFE_VAR_SINGLE_SENSED_FUSE_VALUE_INIT(pVar, value, bFuseValSigned) \
    do {                                                                        \
        if (bFuseValSigned)                                                     \
        {                                                                       \
            ((pVar)->data.signedValue) = ((LwS32)(value));                      \
        }                                                                       \
        else                                                                    \
        {                                                                       \
            ((pVar)->data.unsignedValue) = ((LwU32)(value));                    \
        }                                                                       \
        ((pVar)->bSigned) = (bFuseValSigned);                                   \
    } while (LW_FALSE)

/*!
 * Helper macros to set fuse value
 *
 * @param[in] pVar
 *     PLW2080_CTRL_PERF_VFE_VAR_SINGLE_SENSED_FUSE_VALUE object in which the
 *     fuse value will be assigned.
 * @param[in] value
 *     Fuse value to be saved in pVar
 */
#define LW2080_CTRL_PERF_VFE_VAR_SINGLE_SENSED_FUSE_VALUE_SIGNED_SET(pVar, value) \
    LW2080_CTRL_PERF_VFE_VAR_SINGLE_SENSED_FUSE_VALUE_INIT((pVar), ((LwS32)(value)), (LW_TRUE))
#define LW2080_CTRL_PERF_VFE_VAR_SINGLE_SENSED_FUSE_VALUE_UNSIGNED_SET(pVar, value) \
    LW2080_CTRL_PERF_VFE_VAR_SINGLE_SENSED_FUSE_VALUE_INIT((pVar), ((LwU32)(value)), (LW_FALSE))

/*!
 * Accessor macro for the fuse values
 *
 * @param[in] pFuseValue
 *     PLW2080_CTRL_PERF_VFE_VAR_SINGLE_SENSED_FUSE_VALUE object used to retrieve
 *     the fuse value
 */
#define LW2080_CTRL_PERF_VFE_VAR_SINGLE_SENSED_FUSE_VALUE_SIGNED_GET(pFuseValue) \
    ((pFuseValue)->data.signedValue)
#define LW2080_CTRL_PERF_VFE_VAR_SINGLE_SENSED_FUSE_VALUE_UNSIGNED_GET(pFuseValue) \
    ((pFuseValue)->data.unsignedValue)

/*!
 * Struct with fuse version check information
 */
typedef struct LW2080_CTRL_PERF_VFE_VAR_SINGLE_SENSED_FUSE_VER_INFO {

    /*!
     * Expected version.
     */
    LwU8   verExpected;
    /*!
     * Set to true if verExpected is to be interpreted as a bit mask with bits
     * corresponding to expected versions set. Else interpret it as single value
     */
    LwBool bVerExpectedIsMask;
    /*!
     * Set to true if fuseVersion check needs to done.
     * Default is to do a fuseVersion check. Can be overridden through regkey.
     */
    LwBool bVerCheck;
    /*!
     * Set to true if fuseVersion check needs to be done but does not generate error.
     * Default is to do a check and break. Can be overridden through regkey.
     */
    LwBool bVerCheckIgnore;
    /*!
     * When set to true use default fuse value on version check fail even if the
     * fuse value read from HW is NOT zero(i.e. fuse value is not corrupted)
     */
    LwBool bUseDefaultOlwerCheckFail;
} LW2080_CTRL_PERF_VFE_VAR_SINGLE_SENSED_FUSE_VER_INFO;

/*!
 * Invalid fuse value. Note - LW_U32_MAX can be a valid signed fuse value
 */
#define LW2080_CTRL_PERF_VFE_VAR_SINGLE_SENSED_FUSE_ILWALID 0x80000000U

/*!
 * Structure describing VFE_VAR_SINGLE_SENSED_FUSE_BASE static information/POR.
 */
typedef struct LW2080_CTRL_PERF_VFE_VAR_INFO_SINGLE_SENSED_FUSE_BASE {
    /*!
     * VFE_VAR_SINGLE_SENSED  super class. Must always be first element in the structure.
     */
    LW2080_CTRL_PERF_VFE_VAR_INFO_SINGLE_SENSED               super;

    /*!
     * fuse version as read from HW.
     */
    LwU8                                                      fuseVersion;

    /*!
     * Current Fuse value in integer format. Reflects fuse value overridden
     * through regkey and default value override.
     */
    LW2080_CTRL_PERF_VFE_VAR_SINGLE_SENSED_FUSE_VALUE         fuseValue;

    /*!
     * Fuse value as read from HW.
     */
    LW2080_CTRL_PERF_VFE_VAR_SINGLE_SENSED_FUSE_VALUE         fuseValueHw;

    /*!
     * Default fuse value.
     */
    LW2080_CTRL_PERF_VFE_VAR_SINGLE_SENSED_FUSE_VALUE         fuseValDefault;

    /*!
     * Information about fuse override through regkey
     */
    LW2080_CTRL_PERF_VFE_VAR_SINGLE_SENSED_FUSE_OVERRIDE_INFO overrideInfo;

    /*!
     * Set to true if version check is done. Fuse version check needs to be done
     * just once per driver load
     */
    LwBool                                                    bVersionCheckDone;

    /*!
     * Current Fuse value in integer format. Reflects fuse value overridden
     * through regkey and default value override.
     */
    LwU32                                                     fuseValueInteger;

    /*!
     * Fuse value as read from HW.
     */
    LwU32                                                     fuseValueHwInteger;

    /*!
     * All the information needed to read fuse version
     */
    LW2080_CTRL_PERF_VFE_VAR_SINGLE_SENSED_FUSE_VER_INFO      verInfo;

    /*!
     * HW Correction Scale factor - used to multiply the HW read fuse value to
     * correct the units of the fuse on the GPU. Default value = 1.0
     */
    LwUFXP20_12                                               hwCorrectionScale;

    /*!
     * HW Correction offset factor - used to offset the HW read fuse value to
     * correct the units of the fuse on the GPU. Default value = 0
     */
    LwS32                                                     hwCorrectionOffset;
} LW2080_CTRL_PERF_VFE_VAR_INFO_SINGLE_SENSED_FUSE_BASE;

/*!
 * Structure describing VFE_VAR_SINGLE_SENSED_FUSE static information/POR.
 */
typedef struct LW2080_CTRL_PERF_VFE_VAR_INFO_SINGLE_SENSED_FUSE {
    /*!
     * VFE_VAR_SINGLE_SENSED_BASE  super class. Must always be first element in the structure.
     */
    LW2080_CTRL_PERF_VFE_VAR_INFO_SINGLE_SENSED_FUSE_BASE super;
    /*!
     * Fuse VFIELD ID
     */
    LwU8                                                  vFieldId;
    /*!
     * Fuse version VFIELD ID
     */
    LwU8                                                  vFieldIdVer;
    /*!
     * Register information to read the fuse
     */
    LW2080_CTRL_PERF_VFE_VAR_SINGLE_SENSED_FUSE_INFO      fuse;
    /*!
     * Register information to read the fuse version
     */
    LW2080_CTRL_PERF_VFE_VAR_SINGLE_SENSED_FUSE_INFO      fuseVer;
} LW2080_CTRL_PERF_VFE_VAR_INFO_SINGLE_SENSED_FUSE;

/*!
 * Structure describing VFE_VAR_SINGLE_SENSED_FUSE_20 static information/POR.
 */
typedef struct LW2080_CTRL_PERF_VFE_VAR_INFO_SINGLE_SENSED_FUSE_20 {
    /*!
     * VFE_VAR_SINGLE_SENSED_BASE  super class. Must always be first element in the structure.
     */
    LW2080_CTRL_PERF_VFE_VAR_INFO_SINGLE_SENSED_FUSE_BASE super;

    /*!
     * Fuse Id
     */
    LwU8                                                  fuseId;

    /*!
     * Fuse version Id
     */
    LwU8                                                  fuseIdVer;
} LW2080_CTRL_PERF_VFE_VAR_INFO_SINGLE_SENSED_FUSE_20;

/*!
 * Structure describing VFE_VAR_SINGLE_SENSED_TEMP static information/POR.
 */
typedef struct LW2080_CTRL_PERF_VFE_VAR_INFO_SINGLE_SENSED_TEMP {
    /*!
     * VFE_VAR_SINGLE_SENSED  super class. Must always be first element in the structure.
     */
    LW2080_CTRL_PERF_VFE_VAR_INFO_SINGLE_SENSED super;
    /*!
     * Index of the Thermal Channel that is temperature source of this variable.
     */
    LwU8                                        thermChannelIndex;
} LW2080_CTRL_PERF_VFE_VAR_INFO_SINGLE_SENSED_TEMP;

/*!
 * Structure describing VFE_VAR_SINGLE_SENSED_VOLTAGE static information/POR.
 */
typedef struct LW2080_CTRL_PERF_VFE_VAR_INFO_SINGLE_VOLTAGE {
    /*!
     * VFE_VAR_SINGLE_SENSED  super class. Must always be first element in the structure.
     */
    LW2080_CTRL_PERF_VFE_VAR_INFO_SINGLE super;
} LW2080_CTRL_PERF_VFE_VAR_INFO_SINGLE_VOLTAGE;

/*!
 * VFE_VAR type-specific data union.  Discriminated by
 * VFE_VAR::super.type.
 */


/*!
 * Structure describing VFE_VAR static information/POR. Implements the
 * BOARDOBJ model/interface.
 */
typedef struct LW2080_CTRL_PERF_VFE_VAR_INFO {
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
     * Type-specific data union.
     */
    union {
        LW2080_CTRL_PERF_VFE_VAR_INFO_DERIVED                   derived;
        LW2080_CTRL_PERF_VFE_VAR_INFO_DERIVED_PRODUCT           derivedProd;
        LW2080_CTRL_PERF_VFE_VAR_INFO_DERIVED_SUM               derivedSum;
        LW2080_CTRL_PERF_VFE_VAR_INFO_SINGLE                    single;
        LW2080_CTRL_PERF_VFE_VAR_INFO_SINGLE_FREQUENCY          singleFreq;
        LW2080_CTRL_PERF_VFE_VAR_INFO_SINGLE_SENSED             sensed;
        LW2080_CTRL_PERF_VFE_VAR_INFO_SINGLE_SENSED_FUSE_BASE   sensedFuseBase;
        LW2080_CTRL_PERF_VFE_VAR_INFO_SINGLE_SENSED_FUSE        sensedFuse;
        LW2080_CTRL_PERF_VFE_VAR_INFO_SINGLE_SENSED_FUSE_20     sensedFuse20;
        LW2080_CTRL_PERF_VFE_VAR_INFO_SINGLE_SENSED_TEMP        sensedTemp;
        LW2080_CTRL_PERF_VFE_VAR_INFO_SINGLE_VOLTAGE            singleVolt;
        LW2080_CTRL_PERF_VFE_VAR_INFO_SINGLE_CALLER_SPECIFIED   callerSpecified;
        LW2080_CTRL_PERF_VFE_VAR_INFO_SINGLE_GLOBALLY_SPECIFIED globallySpecified;
    } data;
} LW2080_CTRL_PERF_VFE_VAR_INFO;

/*!
 * Structure describing VFE_VARS static information/POR.  Implements the
 * BOARDOBJGRP model/interface.
 */
#define LW2080_CTRL_PERF_VFE_VARS_INFO_MESSAGE_ID (0xB1U)

typedef struct LW2080_CTRL_PERF_VFE_VARS_INFO {
    /*!
     * LW2080_CTRL_BOARDOBJGRP_E255 super class. Must always be first object in
     * structure.
     */
    LW2080_CTRL_BOARDOBJGRP_E255  super;
    /*!
     * Array of VFE_VAR structures.  Has valid indexes corresponding to the
     * bits set in @ref super.objMask.
     */
    LW2080_CTRL_PERF_VFE_VAR_INFO vars[LW2080_CTRL_BOARDOBJGRP_E255_MAX_OBJECTS];
} LW2080_CTRL_PERF_VFE_VARS_INFO;

/*!
 * LW2080_CTRL_CMD_PERF_VFE_VARS_GET_INFO
 *
 * This command returns VFE_VARS static object information/POR as specified
 * by the VBIOS in VFE Table.
 *
 * See @ref LW2080_CTRL_PERF_VFE_VARS_INFO for documentation on the
 * parameters.
 *
 * Return values are specified per @ref BoardObjGrpGetInfo.
 */
#define LW2080_CTRL_CMD_PERF_VFE_VARS_GET_INFO (0x208020b1) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | LW2080_CTRL_PERF_VFE_VARS_INFO_MESSAGE_ID" */

/*!
 * Structure representing the dynamic state associated with VFE_VAR_SINGLE.
 */
typedef struct LW2080_CTRL_PERF_VFE_VAR_STATUS_SINGLE {
    /*!
     * Lwrrently we do NOT have any static info parameter in this class.
     */
    LwU8 rsvd;
} LW2080_CTRL_PERF_VFE_VAR_STATUS_SINGLE;

/*!
 * Structure representing the dynamic state associated with VFE_VAR_SINGLE.
 */
typedef struct LW2080_CTRL_PERF_VFE_VAR_STATUS_SINGLE_GLOBALLY_SPECIFIED {
    /*!
     * Current active value of globally specified single variable.
     */
    LwS32 valLwrr;
} LW2080_CTRL_PERF_VFE_VAR_STATUS_SINGLE_GLOBALLY_SPECIFIED;

/*!
 * Structure representing the dynamic state associated with VFE_VAR_SINGLE_SENSED.
 */
typedef struct LW2080_CTRL_PERF_VFE_VAR_STATUS_SINGLE_SENSED {
    /*!
     * VFE_VAR_SINGLE super class. Must always be first element in the structure.
     */
    LW2080_CTRL_PERF_VFE_VAR_STATUS_SINGLE super;
} LW2080_CTRL_PERF_VFE_VAR_STATUS_SINGLE_SENSED;

/*!
 * Structure representing the dynamic state associated with VFE_VAR_SINGLE_SENSED_TEMP.
 */
typedef struct LW2080_CTRL_PERF_VFE_VAR_STATUS_SINGLE_SENSED_TEMP {
    /*!
     * VFE_VAR_SINGLE_SENSED  super class. Must always be first element in the structure.
     */
    LW2080_CTRL_PERF_VFE_VAR_STATUS_SINGLE_SENSED super;
    /*!
     * Most recently sampled temperature that exceeded hysteresis range.
     */
    LwTemp                                        lwTemp;
} LW2080_CTRL_PERF_VFE_VAR_STATUS_SINGLE_SENSED_TEMP;

/*!
 * VFE_VAR type-specific data union.  Discriminated by
 * VFE_VAR::super.type.
 */


/*!
 * Structure representing the dynamic state associated with each VFE_VAR.
 */
typedef struct LW2080_CTRL_PERF_VFE_VAR_STATUS {
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
     * Type-specific data union.
     */
    union {
        LW2080_CTRL_PERF_VFE_VAR_STATUS_SINGLE                    single;
        LW2080_CTRL_PERF_VFE_VAR_STATUS_SINGLE_SENSED             sensed;
        LW2080_CTRL_PERF_VFE_VAR_STATUS_SINGLE_SENSED_TEMP        sensedTemp;
        LW2080_CTRL_PERF_VFE_VAR_STATUS_SINGLE_GLOBALLY_SPECIFIED globallySpecified;
    } data;
} LW2080_CTRL_PERF_VFE_VAR_STATUS;

/*!
 * Structure representing the dynamic state associated with VFE_VARS.
 */
#define LW2080_CTRL_PERF_VFE_VARS_STATUS_MESSAGE_ID (0xB2U)

typedef struct LW2080_CTRL_PERF_VFE_VARS_STATUS {
    /*!
     * LW2080_CTRL_BOARDOBJGRP_E255 super class.  Must always be first object in
     * structure.
     */
    LW2080_CTRL_BOARDOBJGRP_E255    super;
    /*!
     * Array of VFE_VAR structures.  Has valid indexes corresponding to the
     * bits set in @ref super.objMask.
     */
    LW2080_CTRL_PERF_VFE_VAR_STATUS vars[LW2080_CTRL_BOARDOBJGRP_E255_MAX_OBJECTS];
} LW2080_CTRL_PERF_VFE_VARS_STATUS;

/*!
 * LW2080_CTRL_CMD_PERF_VFE_VARS_GET_STATUS
 *
 * This command returns VFE_VARS dynamic state as specified by the VBIOS in
 * VFE Table.
 *
 * See @ref LW2080_CTRL_PERF_VFE_VARS_STATUS for documentation on the
 * parameters.
 *
 * Return values are specified per @ref BoardObjGrpGetStatus.
 */
#define LW2080_CTRL_CMD_PERF_VFE_VARS_GET_STATUS (0x208020b2) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | LW2080_CTRL_PERF_VFE_VARS_STATUS_MESSAGE_ID" */

/*!
 * Structure representing the control parameters associated with VFE_VAR_DERIVED.
 */
typedef struct LW2080_CTRL_PERF_VFE_VAR_CONTROL_DERIVED {
    /*!
     * Lwrrently we do NOT have any static info parameter in this class.
     */
    LwU8 rsvd;
} LW2080_CTRL_PERF_VFE_VAR_CONTROL_DERIVED;

/*!
 * Structure representing the control parameters associated with VFE_VAR_DERIVED_PRODUCT.
 */
typedef struct LW2080_CTRL_PERF_VFE_VAR_CONTROL_DERIVED_PRODUCT {
    /*!
     * VFE_VAR_DERIVED super class. Must always be first element in the structure.
     */
    LW2080_CTRL_PERF_VFE_VAR_CONTROL_DERIVED super;
} LW2080_CTRL_PERF_VFE_VAR_CONTROL_DERIVED_PRODUCT;

/*!
 * Structure representing the control parameters associated with VFE_VAR_DERIVED_SUM.
 */
typedef struct LW2080_CTRL_PERF_VFE_VAR_CONTROL_DERIVED_SUM {
    /*!
     * VFE_VAR_DERIVED super class. Must always be first element in the structure.
     */
    LW2080_CTRL_PERF_VFE_VAR_CONTROL_DERIVED super;
} LW2080_CTRL_PERF_VFE_VAR_CONTROL_DERIVED_SUM;

/*!
 * Structure representing the control parameters associated with VFE_VAR_SINGLE.
 */
typedef struct LW2080_CTRL_PERF_VFE_VAR_CONTROL_SINGLE {
    /*!
     * LW2080_CTRL_PERF_VFE_VAR_SINGLE_OVERRIDE_TYPE_<xyz> ID of a VFE variable
     * override type (by default set to _NONE).
     */
    LwU8  overrideType;
    /*!
     * Value of a VFE variable override (as IEEE-754 32-bit floating point).
     */
    LwU32 overrideValue;
} LW2080_CTRL_PERF_VFE_VAR_CONTROL_SINGLE;

/*!
 * Structure representing the control parameters associated with VFE_VAR_SINGLE_FREQUENCY.
 */
typedef struct LW2080_CTRL_PERF_VFE_VAR_CONTROL_SINGLE_FREQUENCY {
    /*!
     * VFE_VAR_SINGLE super class. Must always be first element in the structure.
     */
    LW2080_CTRL_PERF_VFE_VAR_CONTROL_SINGLE super;
} LW2080_CTRL_PERF_VFE_VAR_CONTROL_SINGLE_FREQUENCY;

/*!
 * Structure describing VFE_VAR_SINGLE_SENSED static CONTROLrmation/POR.
 */
typedef struct LW2080_CTRL_PERF_VFE_VAR_CONTROL_SINGLE_SENSED {
    /*!
     * VFE_VAR_SINGLE super class. Must always be first element in the structure.
     */
    LW2080_CTRL_PERF_VFE_VAR_CONTROL_SINGLE super;
} LW2080_CTRL_PERF_VFE_VAR_CONTROL_SINGLE_SENSED;

/*!
 * Structure representing the control parameters associated with VFE_VAR_SINGLE_SENSED_FUSE.
 */
typedef struct LW2080_CTRL_PERF_VFE_VAR_CONTROL_SINGLE_SENSED_FUSE {
    /*!
     * VFE_VAR_SINGLE_SENSED  super class. Must always be first element in the structure.
     */
    LW2080_CTRL_PERF_VFE_VAR_CONTROL_SINGLE_SENSED super;
} LW2080_CTRL_PERF_VFE_VAR_CONTROL_SINGLE_SENSED_FUSE;

/*!
 * Structure representing the control parameters associated with VFE_VAR_SINGLE_SENSED_TEMP.
 */
typedef struct LW2080_CTRL_PERF_VFE_VAR_CONTROL_SINGLE_SENSED_TEMP {
    /*!
     * VFE_VAR_SINGLE_SENSED  super class. Must always be first element in the structure.
     */
    LW2080_CTRL_PERF_VFE_VAR_CONTROL_SINGLE_SENSED super;
    /*!
     * Absolute value of Positive Temperature Hysteresis (0 => no hysteresis).
     * (hysteresis to apply when temperature has positive delta)
     */
    LwTemp                                         tempHysteresisPositive;
    /*!
     * Absolute value of Negative Temperature Hysteresis (0 => no hysteresis).
     * (hysteresis to apply when temperature has negative delta)
     */
    LwTemp                                         tempHysteresisNegative;
} LW2080_CTRL_PERF_VFE_VAR_CONTROL_SINGLE_SENSED_TEMP;

/*!
 * Structure representing the control parameters associated with VFE_VAR_SINGLE_SENSED_VOLTAGE.
 */
typedef struct LW2080_CTRL_PERF_VFE_VAR_CONTROL_SINGLE_VOLTAGE {
    /*!
     * VFE_VAR_SINGLE_SENSED  super class. Must always be first element in the structure.
     */
    LW2080_CTRL_PERF_VFE_VAR_CONTROL_SINGLE super;
} LW2080_CTRL_PERF_VFE_VAR_CONTROL_SINGLE_VOLTAGE;

/*!
 * Structure representing the control parameters associated with VFE_VAR_SINGLE_CALLER_SPECIFIED.
 */
typedef struct LW2080_CTRL_PERF_VFE_VAR_CONTROL_SINGLE_CALLER_SPECIFIED {
    /*!
     * VFE_VAR_SINGLE  super class. Must always be first element in the structure.
     */
    LW2080_CTRL_PERF_VFE_VAR_CONTROL_SINGLE super;
} LW2080_CTRL_PERF_VFE_VAR_CONTROL_SINGLE_CALLER_SPECIFIED;

/*!
 * Special macro to define invalid value for @ref valOverride
 */
#define LW2080_CTRL_PERF_VFE_VAR_SINGLE_GLOBALLY_SPECIFIED_VAL_OVERRIDE_ILWALID LW_S32_MIN

/*!
 * Structure representing the control parameters associated with VFE_VAR_SINGLE_GLOBALLY_SPECIFIED.
 */
typedef struct LW2080_CTRL_PERF_VFE_VAR_CONTROL_SINGLE_GLOBALLY_SPECIFIED {
    /*!
     * VFE_VAR_SINGLE  super class. Must always be first element in the structure.
     */
    LW2080_CTRL_PERF_VFE_VAR_CONTROL_SINGLE super;

    /*!
     * Client override value of globally specified single variable.
     * RM / PMU will use this when this is not equal to the _ILWALID
     * @protected
     */
    LwS32                                   valOverride;
} LW2080_CTRL_PERF_VFE_VAR_CONTROL_SINGLE_GLOBALLY_SPECIFIED;

/*!
 * VFE_VAR type-specific data union.  Discriminated by
 * VFE_VAR::super.type.
 */


/*!
 * Structure representing the control parameters associated with each VFE_VAR.
 */
typedef struct LW2080_CTRL_PERF_VFE_VAR_CONTROL {
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
     * Lower limit for the variable value (as IEEE-754 32-bit floating point).
     */
    LwU32                outRangeMin;
    /*!
     * Upper limit for the variable value (as IEEE-754 32-bit floating point).
     */
    LwU32                outRangeMax;
    /*!
     * Type-specific data union.
     */
    union {
        LW2080_CTRL_PERF_VFE_VAR_CONTROL_DERIVED                   derived;
        LW2080_CTRL_PERF_VFE_VAR_CONTROL_DERIVED_PRODUCT           derivedProd;
        LW2080_CTRL_PERF_VFE_VAR_CONTROL_DERIVED_SUM               derivedSum;
        LW2080_CTRL_PERF_VFE_VAR_CONTROL_SINGLE                    single;
        LW2080_CTRL_PERF_VFE_VAR_CONTROL_SINGLE_FREQUENCY          singleFreq;
        LW2080_CTRL_PERF_VFE_VAR_CONTROL_SINGLE_SENSED             sensed;
        LW2080_CTRL_PERF_VFE_VAR_CONTROL_SINGLE_SENSED_FUSE        sensedFuse;
        LW2080_CTRL_PERF_VFE_VAR_CONTROL_SINGLE_SENSED_TEMP        sensedTemp;
        LW2080_CTRL_PERF_VFE_VAR_CONTROL_SINGLE_VOLTAGE            singleVolt;
        LW2080_CTRL_PERF_VFE_VAR_CONTROL_SINGLE_CALLER_SPECIFIED   callerSpecified;
        LW2080_CTRL_PERF_VFE_VAR_CONTROL_SINGLE_GLOBALLY_SPECIFIED globallySpecified;
    } data;
} LW2080_CTRL_PERF_VFE_VAR_CONTROL;

/*!
 * Structure representing the control parameters associated with VFE_VARS.
 */
typedef struct LW2080_CTRL_PERF_VFE_VARS_CONTROL {
    /*!
     * LW2080_CTRL_BOARDOBJGRP_E255 super class. Must always be first object in
     * structure.
     */
    LW2080_CTRL_BOARDOBJGRP_E255     super;
    /*!
     * Polling period [ms] at which the VFE logic will evaluate for dynamic
     * observed changes in the independent variables (e.g. temperature).
     */
    LwU8                             pollingPeriodms;
    /*!
     * Array of VFE_VAR structures.  Has valid indexes corresponding to the
     * bits set in @ref super.objMask.
     */
    LW2080_CTRL_PERF_VFE_VAR_CONTROL vars[LW2080_CTRL_BOARDOBJGRP_E255_MAX_OBJECTS];
} LW2080_CTRL_PERF_VFE_VARS_CONTROL;

/*!
 * LW2080_CTRL_CMD_PERF_VFE_VARS_GET_CONTROL
 *
 * This command returns VFE_VARS control parameters as specified by the VFE_VARS
 * entries in the VFE Table.
 *
 * See @ref LW2080_CTRL_PERF_VFE_VARS_CONTROL for documentation on the
 * parameters.
 *
 * Return values are specified per @ref BoardObjGrpGetControl.
 */
#define LW2080_CTRL_CMD_PERF_VFE_VARS_GET_CONTROL                (0x208020b3) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | 0xB3" */

/*!
 * LW2080_CTRL_CMD_PERF_VFE_VARS_SET_CONTROL
 *
 * This command accepts client-specified control parameters for a set of
 * VFE_VARS entries in the VFE Table and applies these new parameters to
 * the set of VFE_VARS entries.
 *
 * See @ref LW2080_CTRL_PERF_VFE_VARS_CONTROL for documentation on the
 * parameters.
 *
 * Return values are specified per @ref BoardObjGrpSetControl.
 */
#define LW2080_CTRL_CMD_PERF_VFE_VARS_SET_CONTROL                (0x208020b4) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | 0xB4" */

/* --------------------------- VFE Equation -------------------------------- */

/*!
 * @defgroup LW2080_CTRL_PERF_VFE_EQU_TYPE_ENUM
 *
 * Enumeration of VFE_EQU types.
 *
 * @{
 */
#define LW2080_CTRL_PERF_VFE_EQU_TYPE_BASE                       0x00U
#define LW2080_CTRL_PERF_VFE_EQU_TYPE_COMPARE                    0x01U
#define LW2080_CTRL_PERF_VFE_EQU_TYPE_MINMAX                     0x02U
#define LW2080_CTRL_PERF_VFE_EQU_TYPE_QUADRATIC                  0x03U
#define LW2080_CTRL_PERF_VFE_EQU_TYPE_EQUATION_SCALAR            0x04U
// Insert new types here and increment _MAX
#define LW2080_CTRL_PERF_VFE_EQU_TYPE_MAX                        0x05U
#define LW2080_CTRL_PERF_VFE_EQU_TYPE_ILWALID                    0xFFU
/*!@}*/

/*!
 * @defgroup LW2080_CTRL_PERF_VFE_EQU_OUTPUT_TYPE_ENUM
 *
 * @brief Enumeration of VFE_EQU output types.
 *
 * Result type for _FREQ_MHZ, _VOLT_UV and _VF_GAIN is LwU32.
 * Result type for _VOLT_DELTA_UV is LwS32.
 *
 * @{
 */
/*!
 * The output is unitless.
 */
#define LW2080_CTRL_PERF_VFE_EQU_OUTPUT_TYPE_UNITLESS            0x00U
/*!
 * The output is a frequency specified in megahertz.
 */
#define LW2080_CTRL_PERF_VFE_EQU_OUTPUT_TYPE_FREQ_MHZ            0x01U
/*!
 * The output is a voltage specified in microvolts.
 */
#define LW2080_CTRL_PERF_VFE_EQU_OUTPUT_TYPE_VOLT_UV             0x02U
/*!
 * The output is the VF gain.
 */
#define LW2080_CTRL_PERF_VFE_EQU_OUTPUT_TYPE_VF_GAIN             0x03U
/*!
 * The output is a voltage specified in microvolts.
 */
#define LW2080_CTRL_PERF_VFE_EQU_OUTPUT_TYPE_VOLT_DELTA_UV       0x04U
/*!
 * The output is work type, a unitless FXP 20.12.
 */
#define LW2080_CTRL_PERF_VFE_EQU_OUTPUT_TYPE_WORK_TYPE           0x06U
/*!
 * The output is a utilization ratio, a unitless FXP 20.12.
 */
#define LW2080_CTRL_PERF_VFE_EQU_OUTPUT_TYPE_UTIL_RATIO          0x07U
/*!
 * The output is the work FB norm, a unitless FXP 20.12.
 */
#define LW2080_CTRL_PERF_VFE_EQU_OUTPUT_TYPE_WORK_FB_NORM        0x08U
/*!
 * The output is a power value specified in milliwatts.
 */
#define LW2080_CTRL_PERF_VFE_EQU_OUTPUT_TYPE_POWER_MW            0x09U
/*!
 * The output is a power/utilization slope in mW/utilization (FXP 20.12).
 */
#define LW2080_CTRL_PERF_VFE_EQU_OUTPUT_TYPE_PWR_OVER_UTIL_SLOPE 0x0AU
/*!
 * The output is the ADC code, a LwU32.
 */
#define LW2080_CTRL_PERF_VFE_EQU_OUTPUT_TYPE_ADC_CODE            0x0BU
/*!
 * The output is the threshold percentage, a unitless FXP 20.12.
 */
#define LW2080_CTRL_PERF_VFE_EQU_OUTPUT_TYPE_THRESH_PERCENT      0x0LW
/*!@}*/

/*!
 * Number of coeffs in the Quadratic Equation.
 */
#define LW2080_CTRL_PERF_VFE_EQU_QUADRATIC_COEFF_COUNT           0x03U

/*!
 * Special macro for invalid VFE equation index - case where feature not supported
 * @note    This will be changed to 16-bit
 *          invalid value @ref LW2080_CTRL_BOARDOBJ_IDX_ILWALID
 */
#define LW2080_CTRL_PERF_VFE_EQU_INDEX_ILWALID                   LW2080_CTRL_BOARDOBJ_IDX_ILWALID

/*!
 * Special macro for invalid VFE equation index - case where feature not supported
 * Meant for modules that will remain in 8-bit forever
 * even with rest of VFE and other VFE client modules
 * using 16-bit VFE index.
 */
#define LW2080_CTRL_PERF_VFE_EQU_INDEX_ILWALID_8BIT              LW2080_CTRL_BOARDOBJ_IDX_ILWALID_8BIT

/*!
 * Defines the max number of objects in the VFE EQU
 * BOARDOBJGRP.
 *
 * Lwrrently this value is 255, but after VFE transitions
 * to 1024 objects this value will be _1024.
 * */
#define LW2080_CTRL_BOARDOBJGRP_MAX_OBJECTS_PERF_VFE_EQU         LW2080_CTRL_BOARDOBJGRP_E1024_MAX_OBJECTS

/*!
 * @defgroup LW2080_CTRL_PERF_VFE_EQU_COMPARE_FUNCTION_ENUM
 *
 * Enumeration of VFE_EQU_COMPARE function types.
 *
 * @{
 */
/*!
 * Comparison to determine if the two values are equal.
 */
#define LW2080_CTRL_PERF_VFE_EQU_COMPARE_FUNCTION_EQUAL          0x00U
/*!
 * Comparison to determine if the first value is greater than or equal to the
 * second value.
 */
#define LW2080_CTRL_PERF_VFE_EQU_COMPARE_FUNCTION_GREATER_EQ     0x01U
/*!
 * Comparison to determine if the first value is greater than the second value.
 */
#define LW2080_CTRL_PERF_VFE_EQU_COMPARE_FUNCTION_GREATER        0x02U
/*!@}*/

/*!
 * Structure describing VFE_EQU_TYPE_COMPARE static information/POR.
 */
typedef struct LW2080_CTRL_PERF_VFE_EQU_INFO_COMPARE {
    /*!
     * Index of an equation to evaluate when comparison resulted in LW_TRUE.
     */
    LW2080_CTRL_PERF_VFE_EQU_IDX equIdxTrue;
    /*!
     * Index of an equation to evaluate when comparison resulted in LW_FALSE.
     */
    LW2080_CTRL_PERF_VFE_EQU_IDX equIdxFalse;
} LW2080_CTRL_PERF_VFE_EQU_INFO_COMPARE;

/*!
 * Structure describing VFE_EQU_TYPE_MINMAX static information/POR.
 */
typedef struct LW2080_CTRL_PERF_VFE_EQU_INFO_MINMAX {
    /*!
     * Index of the first equation used by min()/max().
     */
    LW2080_CTRL_PERF_VFE_EQU_IDX equIdx0;
    /*!
     * Index of the second equation used by min()/max().
     */
    LW2080_CTRL_PERF_VFE_EQU_IDX equIdx1;
} LW2080_CTRL_PERF_VFE_EQU_INFO_MINMAX;

/*!
 * Structure describing VFE_EQU_TYPE_QUADRATIC static information/POR.
 */
typedef struct LW2080_CTRL_PERF_VFE_EQU_INFO_QUADRATIC {
    /*!
     * Lwrrently we do not have any static info parameters in this class.
     */
    LwU8 rsvd;
} LW2080_CTRL_PERF_VFE_EQU_INFO_QUADRATIC;

/*!
 * Structure describing VFE_EQU_TYPE_EQUATION_SCALAR static information/POR.
 */
typedef struct LW2080_CTRL_PERF_VFE_EQU_INFO_EQUATION_SCALAR {
    /*!
     * The Index of The Equation to Scale.
     */
    LW2080_CTRL_PERF_VFE_EQU_IDX equIdxToScale;
} LW2080_CTRL_PERF_VFE_EQU_INFO_EQUATION_SCALAR;

/*!
 * VFE_EQU type-specific data union.  Discriminated by
 * VFE_EQU::super.type.
 */


/*!
 * Structure describing VFE_EQU static information/POR. Implements the
 * BOARDOBJ model/interface.
 */
typedef struct LW2080_CTRL_PERF_VFE_EQU_INFO {
    /*!
     * LW2080_CTRL_BOARDOBJ super class. Must always be first object in
     * structure.
     */
    LW2080_CTRL_BOARDOBJ         super;
    /*!
     * XAPI does not support using an element within another structure as a
     * discriminant.  Placing this redundant type value here until that design
     * constraint can be fixed.
     */
    LwU8                         type;
    /*!
     * Index of an Independent Variable used by child object (where required).
     */
    LwU8                         varIdx;
    /*!
     * Index of the next VFE Equation in the Equation Chain (linked list).
     * List is terminated by @ref LW2080_CTRL_PERF_VFE_EQU_INDEX_ILWALID value.
     */
    LW2080_CTRL_PERF_VFE_EQU_IDX equIdxNext;
    /*!
     * Equation's result type as LW2080_CTRL_PERF_VFE_EQU_OUTPUT_TYPE_<xyz>.
     */
    LwU8                         outputType;
    /*!
     * Type-specific data union.
     */
    union {
        LW2080_CTRL_PERF_VFE_EQU_INFO_COMPARE         compare;
        LW2080_CTRL_PERF_VFE_EQU_INFO_MINMAX          minmax;
        LW2080_CTRL_PERF_VFE_EQU_INFO_QUADRATIC       quadratic;
        LW2080_CTRL_PERF_VFE_EQU_INFO_EQUATION_SCALAR equScalar;
    } data;
} LW2080_CTRL_PERF_VFE_EQU_INFO;

/*!
 * Structure describing VFE_EQUS static information/POR.  Implements the
 * BOARDOBJGRP model/interface.
 *
 * @note To be changed to 1024 soon.
 */
#define LW2080_CTRL_PERF_VFE_EQUS_INFO_MESSAGE_ID (0xB5U)

typedef struct LW2080_CTRL_PERF_VFE_EQUS_INFO {
    /*!
     * LW2080_CTRL_BOARDOBJGRP_E1024 super class.  Must always be first object in
     * structure.
     */
    LW2080_CTRL_BOARDOBJGRP_E1024 super;
    /*!
     * Array of VFE_EQU structures.  Has valid indexes corresponding to the
     * bits set in @ref super.objMask.
     */
    LW2080_CTRL_PERF_VFE_EQU_INFO equs[LW2080_CTRL_BOARDOBJGRP_MAX_OBJECTS_PERF_VFE_EQU];
} LW2080_CTRL_PERF_VFE_EQUS_INFO;

/*!
 * LW2080_CTRL_CMD_PERF_VFE_EQUS_GET_INFO
 *
 * This command returns VFE_EQUS static object information/POR as specified
 * by the VBIOS in VFE Table.
 *
 * See @ref LW2080_CTRL_PERF_VFE_EQUS_INFO for documentation on the
 * parameters.
 *
 * Return values are specified per @ref BoardObjGrpGetInfo.
 */
#define LW2080_CTRL_CMD_PERF_VFE_EQUS_GET_INFO (0x208020b5) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | LW2080_CTRL_PERF_VFE_EQUS_INFO_MESSAGE_ID" */

/*!
 * Structure representing the control parameters associated with VFE_EQU_TYPE_COMPARE.
 */
typedef struct LW2080_CTRL_PERF_VFE_EQU_CONTROL_COMPARE {
    /*!
     * Comparison function ID as LW2080_CTRL_PERF_VFE_EQU_COMPARE_FUNCTION_<xyz>.
     */
    LwU8  funcId;
    /*!
     * Numeric value used when evaluating comparison function @ref funcId
     * (as IEEE-754 32-bit floating point).
     */
    LwU32 criteria;
} LW2080_CTRL_PERF_VFE_EQU_CONTROL_COMPARE;

/*!
 * Structure representing the control parameters associated with VFE_EQU_TYPE_MINMAX.
 */
typedef struct LW2080_CTRL_PERF_VFE_EQU_CONTROL_MINMAX {
    /*!
     * When set class evaluates max(), otherwise min().
     */
    LwBool bMax;
} LW2080_CTRL_PERF_VFE_EQU_CONTROL_MINMAX;

/*!
 * Structure representing the control parameters associated with VFE_EQU_TYPE_QUADRATIC.
 */
typedef struct LW2080_CTRL_PERF_VFE_EQU_CONTROL_QUADRATIC {
    /*!
     * Quadratic Equation's coefficients (as IEEE-754 32-bit floating point).
     */
    LwU32 coeffs[LW2080_CTRL_PERF_VFE_EQU_QUADRATIC_COEFF_COUNT];
} LW2080_CTRL_PERF_VFE_EQU_CONTROL_QUADRATIC;

/*!
 * VFE_EQU type-specific data union.  Discriminated by
 * VFE_EQU::super.type.
 */


/*!
 * Structure representing the control parameters associated with VFE_EQU. Implements the
 * BOARDOBJ model/interface.
 */
typedef struct LW2080_CTRL_PERF_VFE_EQU_CONTROL {
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
     * Lower limit for the equation value (as IEEE-754 32-bit floating point).
     */
    LwU32                outRangeMin;
    /*!
     * Upper limit for the equation value (as IEEE-754 32-bit floating point).
     */
    LwU32                outRangeMax;
    /*!
     * Type-specific data union.
     */
    union {
        LW2080_CTRL_PERF_VFE_EQU_CONTROL_COMPARE   compare;
        LW2080_CTRL_PERF_VFE_EQU_CONTROL_MINMAX    minmax;
        LW2080_CTRL_PERF_VFE_EQU_CONTROL_QUADRATIC quadratic;
    } data;
} LW2080_CTRL_PERF_VFE_EQU_CONTROL;

/*!
 * Structure representing the control parameters associated with VFE_EQUS.  Implements the
 * BOARDOBJGRP model/interface.
 *
 * @note To be changed to 1024 soon.
 */
typedef struct LW2080_CTRL_PERF_VFE_EQUS_CONTROL {
    /*!
     * LW2080_CTRL_BOARDOBJGRP_E1024 super class.  Must always be first object in
     * structure.
     */
    LW2080_CTRL_BOARDOBJGRP_E1024    super;
    /*!
     * Array of VFE_EQU structures.  Has valid indexes corresponding to the
     * bits set in @ref super.objMask.
     */
    LW2080_CTRL_PERF_VFE_EQU_CONTROL equs[LW2080_CTRL_BOARDOBJGRP_MAX_OBJECTS_PERF_VFE_EQU];
} LW2080_CTRL_PERF_VFE_EQUS_CONTROL;

/*!
 * LW2080_CTRL_CMD_PERF_VFE_EQUS_GET_CONTROL
 *
 * This command returns VFE_EQUS control parameters as specified by the VFE_EQUS
 * entries in the VFE Table.
 *
 * See @ref LW2080_CTRL_PERF_VFE_EQUS_CONTROL for documentation on the
 * parameters.
 *
 * Return values are specified per @ref BoardObjGrpGetControl.
 */
#define LW2080_CTRL_CMD_PERF_VFE_EQUS_GET_CONTROL (0x208020b6) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | 0xB6" */

/*!
 * LW2080_CTRL_CMD_PERF_VFE_EQUS_SET_CONTROL
 *
 * This command accepts client-specified control parameters for a set of
 * VFE_EQUS entries in the VFE Table and applies these new parameters to
 * the set of VFE_EQUS entries.
 *
 * See @ref LW2080_CTRL_PERF_VFE_EQUS_CONTROL for documentation on the
 * parameters.
 *
 * Return values are specified per @ref BoardObjGrpSetControl.
 */
#define LW2080_CTRL_CMD_PERF_VFE_EQUS_SET_CONTROL (0x208020b7) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | 0xB7" */

/* _ctrl2080vfe_h_ */
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)

