/*
 * Copyright (c) 2019-2021, LWPU CORPORATION. All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#ifndef INCLUDED_LWSCIBUF_UTILS_H
#define INCLUDED_LWSCIBUF_UTILS_H

#include "lwsciipc_internal.h"
#include "lwscicommon_utils.h"
#include "lwscibuf_colorcolwersion.h"

/**
 * \brief This macro compares two numbers, returns the greater number.
 *
 * Thread-safe: Yes
 *
 * This is a pure function with no side-effects
 *
 * \param a refers to number a
 * \param b refers to number b
 *
 * \return greater of a and b
 *
 * \implements{18842259}
 */
#define LW_SCI_BUF_MAX_NUM(a, b)  (((a) > (b)) ? (a) : (b))

/**
 * \brief This macro compares two numbers, returns the smaller number.
 *
 * Thread-safe: Yes
 *
 * This is a pure function with no side-effects
 *
 * \param a refers to number a.
 * \param b refers to number b.
 *
 * \return smaller of a and b
 *
 * \implements{18842262}
 */
#define LW_SCI_BUF_MIN_NUM(a, b)  (((a) < (b)) ? (a) : (b))

/**
 * \brief This macro callwlates the greater of dest and src, and assigns the
 * dest with the callwlated value.
 *
 * Thread-safe: Yes
 *
 * This is a pure function with no side-effects
 *
 * \param dest is a number
 * \param src is a number
 *
 * \return dest is assigned the greater of dest and src.
 *
 * \implements{18842265}
 */
#define LW_SCI_BUF_RECONCILE_MAX(dest, src) \
            ((dest) = LW_SCI_BUF_MAX_NUM((dest), (src)))

/**
 * \brief Derive LwSciBufAttrValDataType corresponding to
 *  LwSciBufAttrValColorFmt from LwColorDataType corresponding to
 *  LwColorFormat.
 *
 * Conlwrrency:
 *  - Thread-safe: Yes
 *  - Synchronization
 *      - This is a pure function with no side-effects
 *
 * \param[in] colorDataType LwColorDataType that needs to be transformed. The
 *            valid value range is LwColorDataType_Integer <= colorDataType
 *            < LwColorDataType_Force32. This can be derived using
 *            LwColorGetDataType() for the corresponding LwColorFormat.
 * \param[in] channelCount refer to the number of individual color component
 *            channels. The valid value is one returned by
 *            LwSciColorGetComponentCount() for the corresponding
 *            LwSciBufAttrValColorFmt.
 * \param[in] colorBPP refer to color bits per pixel. The valid value is one
 *            returned by LwColorGetBPP() for the corresponding LwColorFormat.
 *
 * \return LwSciBufAttrValDataType. Returned LwSciBufAttrValDataType is valid if it is <
 * LwSciDataType_UpperBound.
 *
 * \implements{18842988}
 */
LwSciBufAttrValDataType LwColorDataTypeToLwSciBufDataType(
    LwColorDataType colorDataType,
    uint8_t channelCount,
    uint32_t colorBPP);

/**
 * \brief Compare if data in the src1 is greater than the
 *  data in the src2.
 *
 * Conlwrrency:
 *  - Thread-safe: No
 *  - Synchronization
 *      - The user must ensure that the input parameter src1 is not modified
 *      - The user must ensure that the input parameter src2 is not modified
 *
 * \param[in] src1 pointer to the data
 * \param[in] src2 pointer to another data
 * \param[in] len length of the data in bytes. valid value should be greater
 *            than 0.
 * \param[out] isBigger
 *                 - @a true refer to src1 is greater than or equal to src2
 *                 - @a false refer to src1 is smaller than src2
 *
 * \return LwSciError
 * - LwSciError_Success if successful
 * - LwSciError_BadParameter if any of the following oclwrs:
 *                           - @a src1 is NULL
 *                           - @a src2 is NULL
 *                           - @a isBigger is NULL
 *                           - @a len is 0
 *
 * \implements{18842991}
 */
LwSciError LwSciBufIsMaxValue(
    const void* src1,
    const void* src2,
    size_t len,
    bool* isBigger);

/**
 * \brief aligns 64-bit @a value to given 64-bit @a alignment.
 *
 * Conlwrrency:
 *  - Thread-safe: Yes
 *  - Synchronization
 *      - The function only operates using immutable data provided via input
 *        parameters (so there is no data-dependency)
 *
 * \param[in] value input value to be aligned
 * \param[in] alignment value to which input value to be aligned
 * \param[out] alignedValue aligned value
 *
 * \return LwSciError
 * - LwSciError_Success if successful
 * - LwSciError_Overflow if arithmetic overflow oclwrs
 * - Panics if @a alignedValue is NULL
 *
 * \implements{20488785}
 */
LwSciError LwSciBufAliglwalue64(
    uint64_t value,
    uint64_t alignment,
    uint64_t* alignedValue);

/**
 * \brief aligns 32-bit @a value to given 32-bit @a alignment.
 *
 * Conlwrrency:
 *  - Thread-safe: Yes
 *  - Synchronization
 *      - The function only operates using immutable data provided via input
 *        parameters (so there is no data-dependency)
 *
 * \param[in] value input value to be aligned
 * \param[in] alignment value to which input value to be aligned
 * \param[out] alignedValue aligned value
 *
 * \return LwSciError
 * - LwSciError_Success if successful
 * - LwSciError_Overflow if arithmetic overflow oclwrs
 * - Panics if @a alignedValue is NULL
 *
 * \implements{20488791}
 */
LwSciError LwSciBufAliglwalue32(
    uint32_t value,
    uint32_t alignment,
    uint32_t* alignedValue);

/**
 * \brief Identifies whether @a ipcEndpoint crosses SoC boundary.
 *
 * \param[in] ipcEndpoint LwSciIpcEndpoint.
 * Valid value: Valid LwSciIpcEndpoint.
 * \param[out] isSocBoundary boolean flag indicating if @a ipcEndpoint crosses
 * SoC boudary. True indicates that SoC boundary is crossed, false otherwise.
 *
 * \return LwSciError
 * - LwSciError_Success if successful.
 * - LwSciError_BadParameter if @a ipcEndpoint is invalid.
 * - Panics if @a isSocBoundary is NULL.
 *
 * \implements{}
 */
LwSciError LwSciBufIsSocBoundary(
    LwSciIpcEndpoint ipcEndpoint,
    bool* isSocBoundary);

#endif  /* INCLUDED_LWSCIBUF_UTILS_H */
