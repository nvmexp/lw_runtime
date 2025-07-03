/*
 * Copyright (c) 2019-2021, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 */

/**
 * \file
 * \brief <b>LwSci Utils Interface</b>
 *
 * @b Description: This file contains LwSci Utils interfaces
 */

#ifndef INCLUDED_LWSCIUTILS_H
#define INCLUDED_LWSCIUTILS_H

#include <inttypes.h>
#include "lwscicommon_covanalysis.h"
#include "lwscicommon_os.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup lwscicommon_platformutils_api LwSciCommon APIs for platform utils.
 * List of APIs exposed at Inter-Element level.
 * @{
 */

/**
 * Macro to indicate success status of arithmetic operation
 *
 * \implements{21755931}
 * \implements{21751553}
 * \implements{18850782}
 */
#define OP_SUCCESS 1U
/**
 * Macro to indicate failure status of arithmetic operation
 *
 * \implements{21751554}
 * \implements{21755932}
 * \implements{18850785}
 */
#define OP_FAIL 0U

#define ADD_FUNC(funcname, type) \
    static inline void funcname(type var1, type var2, \
                                type *res, uint8_t *res_status) \
    { \
        type tmpResult; \
        if ((NULL == res_status) || (NULL == res)) { \
            LwSciCommonPanic();\
        } \
        tmpResult = var1 + var2; \
        if (tmpResult < var1) { \
            *res_status = OP_FAIL; \
        } else { \
            *res_status = OP_SUCCESS; \
            *res = tmpResult; \
        } \
    }

#define SUB_FUNC(funcname, type) \
    static inline void funcname(type var1, type var2, \
                                type *res, uint8_t *res_status) \
    { \
        if ((NULL == res_status) || (NULL == res)) { \
            LwSciCommonPanic();\
        } \
        if (var1 < var2) { \
            *res_status = OP_FAIL; \
        } else { \
            *res_status = OP_SUCCESS; \
            *res = var1 - var2; \
        } \
    }

#define MUL_FUNC(funcname, type) \
    static inline void (funcname)(type var1, type var2, \
                                  type *res, uint8_t *res_status) \
    { \
        if ((NULL == res_status) || (NULL == res)) { \
            LwSciCommonPanic();\
        } \
        if ((var2 != 0U) && (var1 > ((~((typeof(var2))0U)) / var2))) { \
            *res_status = OP_FAIL; \
        } else { \
            *res_status = OP_SUCCESS; \
            *res = var1 * var2; \
        } \
    }

/**
 * \brief Performs addition operation on 32-bit unsigned integer.
 *  Indicates overflow status if the summation exceeds the maximum representable
 *  value by 32-bit unsigned integer. Output status flag will have value OP_FAIL
 *  if there was an overflow detected else OP_SUCCESS.
 *
 * \param[in] var1 first operand
 *  Valid value: [0, UINT32_MAX]
 * \param[in] var2 second operand
 *  Valid value: [0, UINT32_MAX]
 * \param[out] res address of output variable
 * \param[out] res_status status flag to indicate if there was an overflow.
 * @a res is valid only if @a res_status is equal to OP_SUCCESS.
 *
 * \return void
 * - Panics if any of the following oclwrs:
 *      - @a res is NULL
 *      - @a res_status is NULL
 *
 * \conlwrrency:
 *  - Thread-safe: Yes
 *  - Synchronization
 *      - No access to any shared data.
 *
 * \implements{18851262}
 * \implements{21749978}
 * \implements{21755810}
 * \implements{18850680}
 */
ADD_FUNC(u32Add, uint32_t)

/**
 * \brief Performs addition operation on 64-bit unsigned integer.
 *  Indicates overflow status if the summation exceeds the maximum representable
 *  value by 64-bit unsigned integer. Output status flag will have value OP_FAIL
 *  if there was an overflow detected else OP_SUCCESS.
 *
 * \param[in] var1 first operand
 *  Valid value: [0, UINT64_MAX]
 * \param[in] var2 second operand
 *  Valid value: [0, UINT64_MAX]
 * \param[out] res address of the output variable
 * \param[out] res_status status flag to indicate if there was an overflow.
 * @a res is valid only if @a res_status is equal to OP_SUCCESS.
 *
 * \return void
 * - Panics if any of the following oclwrs:
 *      - @a res is NULL
 *      - @a res_status is NULL
 *
 * \conlwrrency:
 *  - Thread-safe: Yes
 *  - Synchronization
 *      - No access to any shared data.
 *
 * \implements{18851265}
 * \implements{21749980}
 * \implements{21755811}
 * \implements{18850683}
 */
ADD_FUNC(u64Add, uint64_t)

/**
 * \brief Performs addition operation on size_t integer.
 *  Indicates overflow status if the summation exceeds the maximum representable
 *  value by size_t integer. Output status flag will have value OP_FAIL if
 *  there was an overflow detected else OP_SUCCESS.
 *
 * \param[in] var1 first operand
 *  Valid value: [0, SIZE_MAX]
 * \param[in] var2 second operand
 *  Valid value: [0, SIZE_MAX]
 * \param[out] res address of the output variable
 * \param[out] res_status status flag to indicate if there was an overflow.
 * @a res is valid only if @a res_status is equal to OP_SUCCESS.
 *
 * \return void
 * - Panics if any of the following oclwrs:
 *      - @a res is NULL
 *      - @a res_status is NULL
 *
 * \conlwrrency:
 *  - Thread-safe: Yes
 *  - Synchronization
 *      - No access to any shared data.
 *
 * \implements{21755812}
 * \implements{21607449}
 * \implements{21607447}
 */
ADD_FUNC(sizeAdd, size_t)

/**
 * \brief Performs subtraction operation on 32-bit unsigned integer.
 *  Indicates underflow status if the subtraction is smaller than the minimum
 *  representable value by 32-bit unsigned integer. Output status flag will have
 *  value OP_FAIL if there was an underflow detected else OP_SUCCESS.
 *
 * \param[in] var1 first operand
 *  Valid value: [0, UINT32_MAX]
 * \param[in] var2 second operand
 *  Valid value: [0, UINT32_MAX]
 * \param[out] res address of the output variable
 * \param[out] res_status status flag to indicate if there was an underflow.
 * @a res is valid only if @a res_status is equal to OP_SUCCESS.
 *
 * \return void
 * - Panics if any of the following oclwrs:
 *      - @a res is NULL
 *      - @a res_status is NULL
 *
 * \conlwrrency:
 *  - Thread-safe: Yes
 *  - Synchronization
 *      - No access to any shared data.
 *
 * \implements{18851268}
 * \implements{21749983}
 * \implements{18850686}
 */
SUB_FUNC(u32Sub, uint32_t)

/**
 * \brief Performs subtraction operation on 64-bit unsigned integer.
 *  Indicates underflow status if the subtraction is smaller than the minimum
 *  representable value by 64-bit unsigned integer. Output status flag will have
 *  value OP_FAIL if there was an underflow detected else OP_SUCCESS.
 *
 * \param[in] var1 first operand
 *  Valid value: [0, UINT64_MAX]
 * \param[in] var2 second operand
 *  Valid value: [0, UINT64_MAX]
 * \param[out] res address of the output variable
 * \param[out] res_status status flag to indicate if there was an underflow.
 * @a res is valid only if @a res_status is equal to OP_SUCCESS.
 *
 * \return void
 * - Panics if any of the following oclwrs:
 *      - @a res is NULL
 *      - @a res_status is NULL
 *
 * \conlwrrency:
 *  - Thread-safe: Yes
 *  - Synchronization
 *      - No access to any shared data.
 *
 * \implements{18851271}
 * \implements{21749984}
 * \implements{21755816}
 * \implements{18850689}
 */
SUB_FUNC(u64Sub, uint64_t)

/**
 * \brief Performs multiplication operation on 32-bit unsigned integer.
 *  Indicates overflow status if the product exceeds maximum representable
 *  value by 32-bit unsigned integer. Output status flag will have value OP_FAIL
 *  if there was an overflow detected else OP_SUCCESS.
 *
 * \param[in] var1 first operand
 *  Valid value: [0, UINT32_MAX]
 * \param[in] var2 second operand
 *  Valid value: [0, UINT32_MAX]
 * \param[out] res address of the output variable
 * \param[out] res_status status flag to indicate if there was an overflow.
 * @a res is valid only if @a res_status is equal to OP_SUCCESS.
 *
 * \return void
 * - Panics if any of the following oclwrs:
 *      - @a res is NULL
 *      - @a res_status is NULL
 *
 * \conlwrrency:
 *  - Thread-safe: Yes
 *  - Synchronization
 *      - No access to any shared data.
 *
 * \implements{18851274}
 * \implements{21749985}
 * \implements{18850692}
 */
LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 1_2), "LwSciCommon-ADV-MISRAC2012-001")
MUL_FUNC(u32Mul, uint32_t)

/**
 * \brief Performs multiplication operation on 64-bit unsigned integer.
 *  Indicates overflow status if the product exceeds the maximum representable
 *  value by 64-bit unsigned integer. Output status flag will have value OP_FAIL
 *  if there was an overflow detected else OP_SUCCESS.
 *
 * \param[in] var1 first operand
 *  Valid value: [0, UINT64_MAX]
 * \param[in] var2 second operand
 *  Valid value: [0, UINT64_MAX]
 * \param[out] res address of the output variable
 * \param[out] res_status status flag to indicate if there was an overflow.
 * @a res is valid only if @a res_status is equal to OP_SUCCESS.
 *
 * \return void
 * - Panics if any of the following oclwrs:
 *      - @a res is NULL
 *      - @a res_status is NULL
 *
 * \conlwrrency:
 *  - Thread-safe: Yes
 *  - Synchronization
 *      - No access to any shared data.
 *
 * \implements{18851277}
 * \implements{21749986}
 * \implements{21755817}
 * \implements{18850695}
 */
LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 1_2), "LwSciCommon-ADV-MISRAC2012-001")
MUL_FUNC(u64Mul, uint64_t)


/**
 * \brief Performs multiplication operation on size_t integer.
 *  Indicates overflow status if the product exceeds the maximum representable
 *  value by size_t integer. Output status flag will have value OP_FAIL if
 *  there was an overflow detected else OP_SUCCESS.
 *
 * \param[in] var1 first operand
 *  Valid value: [0, SIZE_MAX]
 * \param[in] var2 second operand
 *  Valid value: [0, SIZE_MAX]
 * \param[out] res address of the output variable
 * \param[out] res_status status flag to indicate if there was an overflow.
 * @a res is valid only if @a res_status is equal to OP_SUCCESS.
 *
 * \return void
 * - Panics if any of the following oclwrs:
 *      - @a res is NULL
 *      - @a res_status is NULL
 *
 * \conlwrrency:
 *  - Thread-safe: Yes
 *  - Synchronization
 *      - No access to any shared data.
 *
 * \implements{18851280}
 * \implements{21749987}
 * \implements{21755823}
 * \implements{18850698}
 *
 */
LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 1_2), "LwSciCommon-ADV-MISRAC2012-001")
MUL_FUNC(sizeMul, size_t)

/**
 * @}
 */

#ifdef __cplusplus
}
#endif

#endif /* INCLUDED_LWSCIUTILS_H */
