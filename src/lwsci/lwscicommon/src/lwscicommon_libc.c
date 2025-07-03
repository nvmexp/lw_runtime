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
 * \brief <b>LwSciCommon libc Implementation</b>
 *
 * @b Description: The APIs in this file use libc functions
 */

#include "lwscicommon_libc.h"

#include <stdint.h>
#include <errno.h>
#include <stdlib.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include "lwscicommon_covanalysis.h"
#include "lwscicommon_libc_priv.h"
#include "lwscicommon_os.h"
#include "lwscicommon_utils.h"
#include "lwscilog.h"

/**
 * @brief Perform a bytewise swap of @a size bytes starting at @a a and @a b.
 *
 * @param[in] a The start address of the first element that we wish to swap.
 * @param[in] b The start address of the second element that we wish to swap.
 * @param[in] size The number of bytes that we wish to swap, starting from the
 * start address.
 */
#define LW_SCI_SWAP_BYTES(a, b, size) \
    do { \
        size_t tempSize = (size); \
        char* tempA = (a); \
        char* tempB = (b); \
        do { \
            char tempTmp = *tempA; \
            *tempA = *tempB; \
            *tempB = tempTmp; \
            tempA++; \
            tempB++; \
        } while (--tempSize > 0UL); \
    } while (1 == 0)

void* LwSciCommonCalloc(
    size_t numItems,
    size_t size)
{
    void* returnPtr = NULL;
    uint8_t addResult = 0;
    uint8_t mulResult = 0;
    size_t allocSize = 0UL;
    size_t totalSize = 0UL;
    LwSciCommonAllocHeader* hdr = NULL;

    if ((0U == numItems) || (0U == size)) {
        LWSCI_ERR_STR("Calling calloc with size 0 is implementation defined");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciCommon-ADV-MISRAC2012-006")
        goto ret;
    }

    u64Mul(size, numItems, &totalSize, &mulResult);
    u64Add(totalSize, sizeof(LwSciCommonAllocHeader), &allocSize, &addResult);
    if (OP_FAIL == (mulResult & addResult)) {
        LWSCI_ERR_STR("Cannot allocate memory of size\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciCommon-ADV-MISRAC2012-006")
        goto ret;
    }

    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 21_3), "Approved TID-609");
    returnPtr = calloc(1, allocSize);
    if (NULL == returnPtr) {
        LWSCI_ERR_INT("calloc failed with error: ", errno);
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciCommon-ADV-MISRAC2012-006")
        goto ret;
    }

    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 11_5), "LwSciCommon-ADV-MISRAC2012-005")
    hdr = (LwSciCommonAllocHeader*)returnPtr;
    hdr->magic = LWSCICOMMON_ALLOC_MAGIC;
    hdr->allocSize = totalSize;

    LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 11_5), "LwSciCommon-ADV-MISRAC2012-005")
    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 18_4), "LwSciCommon-ADV-MISRAC2012-007")
    returnPtr = (void*)((char*)returnPtr + sizeof(LwSciCommonAllocHeader));
    LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 11_5))

ret:
    return returnPtr;
}

void LwSciCommonFree(
    void* ptr)
{
    LwSciCommonAllocHeader* hdr = NULL;

    if (NULL == ptr) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciCommon-ADV-MISRAC2012-006")
        goto ret;
    }

    LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 11_5), "LwSciCommon-ADV-MISRAC2012-005")
    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 18_4), "LwSciCommon-ADV-MISRAC2012-007")
    hdr = (void*)((char*)ptr - sizeof(LwSciCommonAllocHeader));
    LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 11_5))

    if (LWSCICOMMON_ALLOC_MAGIC != hdr->magic) {
        LwSciCommonPanic();
    }

    (void)memset(ptr, 0x00, hdr->allocSize);
    hdr->allocSize = 0x00;
    hdr->magic = 0x00;

    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 21_3), "Approved TID-609");
    free(hdr);

ret:
    return;
}

void LwSciCommonMemcpyS(
    void* dest,
    size_t destSize,
    const void* src,
    size_t n)
{
    LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 11_5), "LwSciCommon-ADV-MISRAC2012-005")
    LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 18_4), "LwSciCommon-ADV-MISRAC2012-007")
    const uint8_t * dBuf = (uint8_t*) dest;
    const uint8_t* sBuf = (const uint8_t*) src;
    LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 11_5))
    LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 18_4))

    if ((NULL == dest) || (NULL == src)) {
        LWSCI_ERR_STR("NULL input to MemcpyS\n");
        LwSciCommonPanic();
    }

    if (destSize < n) {
        LWSCI_ERR_STR("dest buffer in MemcpyS too small\n");
        LwSciCommonPanic();
    }

    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 18_4), "LwSciCommon-ADV-MISRAC2012-007")
     if ((dBuf <= sBuf) && ((dBuf + n) > sBuf)) {
        LWSCI_ERR_STR("dest and src buffers overlap\n");
        LwSciCommonPanic();
    }

    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 18_4), "LwSciCommon-ADV-MISRAC2012-007")
    if ((sBuf <= dBuf) && ((sBuf + n) > dBuf)) {
        LWSCI_ERR_STR("dest and src buffers overlap\n");
        LwSciCommonPanic();
    }

    (void)memcpy(dest, src, n);
}

LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Directive, 4_6), "LwSciCommon-ADV-MISRAC2012-003")
LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 8_7), "LwSciCommon-ADV-MISRAC2012-009")
int LwSciCommonMemcmp(
    const void* ptr1,
    const void* ptr2,
    size_t size)
LWCOV_ALLOWLIST_END(LWCOV_MISRA(Directive, 4_6))
{
    LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 11_5), "LwSciCommon-ADV-MISRAC2012-005")
    const uint8_t* uint8Ptr1 = (const uint8_t*)ptr1;
    const uint8_t* uint8Ptr2 = (const uint8_t*)ptr2;
    LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 11_5))

    if ((NULL == ptr1) || (NULL == ptr2)) {
        LwSciCommonPanic();
    }
    return memcmp(uint8Ptr1, uint8Ptr2, size);
}

LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 8_7), "LwSciCommon-ADV-MISRAC2012-009")
void LwSciCommonSort(
    void* base,
    size_t nmemb,
    size_t size,
    LwSciCommonSortComparator compare)
{
    size_t i = 0U;
    size_t j = 0U;
    void* aStartAddr = NULL;
    void* bStartAddr = NULL;
    bool swapped = false;

    if ((NULL == compare) || (NULL == base) || (0UL == nmemb) || (0UL == size)) {
        LwSciCommonPanic();
    }
    /*
     * We need to meet MISRA requirements that the sorting algorithm must be:
     *
     *   1. Stable: It must "behave consistently when comparing elements" per
     *      Rule 21.9 (ie. the sort must be a stable sorting algorithm)
     *   2. Non-relwrsive: MISRA Rule 17.2
     *   3. In-place: Without implementing an object pool or using a custom
     *      allocator via LwSciCommonCalloc, we meet MISRA Rule 21.3 by sorting
     *      in-place.
     *
     * As such, we implement bubble sort since we don't expect to be sorting
     * large arrays at the moment.
     */
    for (i = 0U; i < (nmemb - 1U); ++i) {
        swapped = false;
        for (j = 0U; j < (nmemb - i - 1U); ++j) {
            LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 11_5), "LwSciCommon-ADV-MISRAC2012-005")
            LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 18_4), "LwSciCommon-ADV-MISRAC2012-007")
            aStartAddr = (char*)base + (j * size);
            bStartAddr = (char*)base + ((j + 1U) * size);
            LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 11_5))
            LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 18_4))

            /*
             * If the current value is greater than the adjacent value, then
             * we need to perform a swap.
             */
            LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 11_5), "LwSciCommon-ADV-MISRAC2012-005")
            LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 18_4), "LwSciCommon-ADV-MISRAC2012-007")
            if (1 == compare((char *)aStartAddr, (char *)bStartAddr)) {
                LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Directive, 4_9), "LwSciCommon-ADV-MISRAC2012-004")
                LW_SCI_SWAP_BYTES(aStartAddr, bStartAddr, size);
                swapped = true;
            }
            LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 11_5))
            LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 18_4))
        }
        /*
         * If no swaps were performed, then break so we don't need to continue
         * with more passes.
         */
        if (false == swapped) {
            break;
        }
    }
}
