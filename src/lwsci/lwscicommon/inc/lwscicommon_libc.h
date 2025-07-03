/*
 * Copyright (c) 2019-2021, LWPU CORPORATION. All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

/**
 * \file
 * \brief <b>LwSciCommon libc Interface</b>
 *
 * @b Description: This file contains LwSciCommon libc APIs
 */

#ifndef INCLUDED_LWSCICOMMON_LIBC_H
#define INCLUDED_LWSCICOMMON_LIBC_H

#include <stddef.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdbool.h>
#include "lwscicommon_covanalysis.h"
#include "lwscierror.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup lwscicommon_blanket_statements LwSciCommon blanket statements.
 * Generic statements applicable for LwSciCommon interfaces.
 * @{
 */

/**
 * \page page_blanket_statements LwSciCommon blanket statements
 *
 * \section element_dependency Dependency on other elements
 * LwSciCommon calls below libc interfaces:
 * - calloc() to allocate memory for an array of a given number of elements,
 *  each of them input size bytes long, and initializes all bytes to 0.
 * - free() to deallocate memory.
 * - memcmp() to compare two memory blocks.
 * - memcpy() to copy data from one memory block to another.
 *
 */

/**
 * @}
 */


/**
 * @defgroup lwscicommon_platformutils_api LwSciCommon APIs for platform utils.
 * List of APIs exposed at Inter-Element level.
 * @{
 */


/**
 * \brief Allocates memory for an array of a given number of elements, each of
 *  them input size bytes long, and initializes all bytes to 0.
 *
 * \param[in] numItems number of elements to allocate.
 *  Valid value: [0, SIZE_MAX]
 * \param[in] size of each element.
 *  Valid value: [0, SIZE_MAX]
 *
 * \return following values based on condition,
 *  - pointer to the allocated chunk of memory if successful.
 *  - NULL in case of failure to allocate memory.
 *  - NULL if input @a numItems is 0
 *  - NULL if input @a size is 0
 *
 * \conlwrrency:
 *  - Thread-safe: Yes
 *  - Synchronization
 *      - No access to any shared data.
 *
 * \implements{18851223}
 */
void* LwSciCommonCalloc(
    size_t numItems,
    size_t size);

/**
 * \brief Overwrites input block of memory with zeros and deallocates the block
 *  of memory previously allocated by a call to LwSciCommonCalloc.
 *
 * \param[in] ptr pointer to the memory to be freed
 *
 * \return void
 *  - Panics if this memory was not allocated via call to LwSciCommonCalloc
 *
 * \conlwrrency:
 *  - Thread-safe: No
 *  - Synchronization
 *      - The user must ensure that the same object is freed only once.
 *
 * \implements{18851226}
 */
void LwSciCommonFree(
    void* ptr);

 /**
 * \brief Copy memory of specified size in bytes from source memory to
 *  destination memory.
 *
 * Prior to copying, checks the following:
 *  - the destination size is greater than or equal to size of data to be copied
 *  - the source and destination memory do not overlap
 *
 *
 * \param[out] dest pointer to the destination memory that is overwritten.
 * \param[in] destSize size of the destination memory.
 *  Valid value: [0, SIZE_MAX]
 * \param[in] src pointer to the source memory that is copied.
 *  Valid value: @a src is not NULL.
 * \param[in] n size to be copied in bytes.
 *  Valid value: [0, destSize]
 *
 * \return void
 * - Panics if any of the following oclwrs:
 *      - @a dest is NULL
 *      - @a src is NULL
 *      - @a destSize is less than @a n
 *      - @a dest and @a src memory overlap
 *
 * \conlwrrency:
 *  - Thread-safe: No
 *  - Synchronization
 *      - The user must ensure that the same output @a dest parameter is not
 *        used by multiple threads at the same time.
 *      - The user must ensure that the data referenced by the input @a src
 *        parameter is not modified by other threads at the same time.
 *
 * \implements{18851229}
 */
void LwSciCommonMemcpyS(
    void* dest,
    size_t destSize,
    const void* src,
    size_t n);

/**
 * \brief Compares the input size bytes of memory beginning at first input
 * memory block against the input size bytes of memory beginning at second
 * input memory block.
 *
 *
 * \param[in] ptr1 first pointer for memory comparison
 *  Valid value: @a ptr1 is not NULL.
 * \param[in] ptr2 second pointer for memory comparison
 *  Valid value: @a ptr2 is not NULL.
 * \param[in] size size of bytes to be compared
 *  Valid value: [0, SIZE_MAX]
 *
 * \return int, integer value indicating the following relationship between the
 *  content of the memory blocks:
 *  - < 0 if the first byte that does not match in both memory blocks has a
 *  lower value in @a ptr1 than in @a ptr2
 *  - 0 the contents of both memory blocks are equal
 *  - > 0 if the first byte that does not match in both memory blocks has a
 *  greater value in @a ptr1 than in @a ptr2
 *  - Panics if any of the following oclwrs:
 *      - @a ptr1 is NULL
 *      - @a ptr2 is NULL
 *
 * \conlwrrency:
 *  - Thread-safe: No
 *  - Synchronization
 *      - The user must ensure that the data referenced by the input @a ptr1
 *        and @a ptr2 parameters is not modified by other threads at the same
 *        time.
 *
 * \implements{18851232}
 */
LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Directive, 4_6), "LwSciCommon-ADV-MISRAC2012-003")
int LwSciCommonMemcmp(
    const void* ptr1,
    const void* ptr2,
    size_t size);
LWCOV_ALLOWLIST_END(LWCOV_MISRA(Directive, 4_6))

/**
 * \brief A helper function pointer data type to a comparison function.
 * It should be used in conjunction with the LwSciCommonSort API.
 * The function is expected to return 1 if arg2 is to be placed
 * before arg1 in sorted list by LwSciCommonSort API.
 * For any other value except 1, the relative order between arg1 and arg2
 * is maintained by LwSciCommonSort API in the sorted list as per the input
 * list to LwSciCommonSort API.
 *
 * \implements{21964271}
 * \implements{21964265}
 * \implements{21964275}
 */
typedef int32_t (*LwSciCommonSortComparator)(const void* arg1, const void* arg2);

/**
 * @brief Sorts element in list using in-place, non-relwrsive, stable algorithm.
 * The relative order of elements in sorted list is decided by the input comparator function.
 *
 * @param[in,out] base Start of target array.
 *  Valid value: if @a base is not NULL
 * @param[in] nmemb number of members in the array
 *  Valid value: [1, SIZE_MAX]
 * @param[in] size size of each array element
 *  Valid value: [1, SIZE_MAX]
 * @param[in] compare The comparator function which is called with two
 *  void* arguments that point to the elements being compared.
 *  Valid value: if @a compare is not NULL
 *
 * \return void
 * - Panics if any of the following oclwrs:
 *      - @a base is NULL
 *      - @a compare is NULL
 *      - @a nmemb is 0
 *      - @a size is 0
 *
 * \conlwrrency:
 *  - Thread-safe: No
 *  - Synchronization
 *      - The user must ensure that the same output @a base parameter is not
 *        used by multiple threads at the same time.
 *
 * \implements{18851283}
 */
void LwSciCommonSort(
    void* base,
    size_t nmemb,
    size_t size,
    LwSciCommonSortComparator compare);

/**
 * @}
 */


#ifdef __cplusplus
}
#endif

#endif
