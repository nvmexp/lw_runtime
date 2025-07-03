/*
 * Copyright (c) 2019-2022, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 */

#include "lwscibuf_attr_constraint_priv.h"

#include <stdbool.h>
#include <string.h>

#include "lwscibuf_colorcolwersion.h"
#include "lwscibuf_constraint_lib.h"
#include "lwscibuf_utils.h"
#include "lwscicommon_os.h"
#include "lwscicommon_covanalysis.h"

#define INPUT_ALIGN_KEYVAL_PAIR_COUNT 2
#define OUTPUT_ALIGN_KEYVAL_PAIR_COUNT 1

#define INPUT_PITCH_KEYVAL_PAIR_COUNT 4
#define OUTPUT_PITCH_KEYVAL_PAIR_COUNT 2

#define INPUT_ALIGN_HEIGHT_KEYVAL_PAIR_COUNT 4
#define OUTPUT_ALIGN_HEIGHT_KEYVAL_PAIR_COUNT 1

#define INPUT_SIZE_KEYVAL_PAIR_COUNT 8
#define OUTPUT_SIZE_KEYVAL_PAIR_COUNT 3

#define INPUT_IMAGESIZE_KEYVAL_PAIR_COUNT 5
#define OUTPUT_IMAGESIZE_KEYVAL_PAIR_COUNT 2

#define INPUT_KEYVAL_PAIR_COUNT 2
#define OUTPUT_KEYVAL_PAIR_COUNT 2
#define OUTPUT_INT_KEYVAL_PAIR_COUNT 4

#define CONSTRAINT_PUBLIC_KEYVAL_PAIR_COUNT 3
#define CONSTRAINT_PRIVATE_KEYVAL_PAIR_COUNT 2

#define RAW_PUBLIC_KEYVAL_PAIR_COUNT 2
#define RAW_PRIVATE_KEYVAL_PAIR_COUNT 2

#define STRIDE_KEYVAL_PAIR_COUNT 2

#define INPUT_PYRAMID_KEYVAL_PAIR_COUNT 7
#define OUTPUT_PYRAMID_KEYVAL_PAIR_COUNT 3
#define PYRAMID_PRIVATE_KEYVAL_PAIR_COUNT 2

#define INPUT_ARR_PUBLIC_KEYVAL_PAIR_COUNT 3
#define OUTPUT_ARR_PUBLIC_KEYVAL_PAIR_COUNT 2
#define ARR_PRIVATE_KEYVAL_PAIR_COUNT 2

static void LwSciBufConstraintMatchBufType(
    const LwSciBufType* bufTypePtr,
    size_t numBufTypes,
    LwSciBufType bufType,
    bool* match)
{
    size_t index = 0U;

    LWSCI_FNENTRY("");

    *match = false;

    LWSCI_INFO("Input: bufTypePtr %p, numBufTypes %zu, bufType: %u, match ptr: %p\n",
        bufTypePtr, numBufTypes, bufType, match);

    for (index = 0; index < numBufTypes; index++) {
        if (bufTypePtr[index] == bufType) {
            *match = true;
            break;
        }
    }

    LWSCI_INFO("Output: match: %s\n", *match ? "true" : "false");

    LWSCI_FNEXIT("");
}

static LwSciError LwSciBufConstrainComputeAlignment(
    LwSciBufAttrList attrList,
    const LwSciBufHwConstraints* constraints,
    uint32_t level)
{
    LwSciError err = LwSciError_Success;
    LwSciBufAttrKeyValuePair ipImageKeyValPair[INPUT_ALIGN_KEYVAL_PAIR_COUNT];
    LwSciBufAttrKeyValuePair opImageKeyValPair[OUTPUT_ALIGN_KEYVAL_PAIR_COUNT];
    uint32_t planeAlignment[LW_SCI_BUF_PYRAMID_MAX_PLANES] = {0};
    uint32_t planeCount = 0U;
    uint32_t planeNum = 0U;
    uint32_t index = 0U;
    uint32_t tmpMul = 0U;
    uint64_t tmpMul2 = 0U;
    uint32_t tmpAdd = 0U;
    uint8_t addStatus = OP_FAIL;
    uint8_t mulStatus = OP_FAIL;

    LWSCI_FNENTRY("");

    LWSCI_INFO("Input: attrList %p, constraints %p\n", attrList,
        constraints);

    (void)memset(ipImageKeyValPair, 0x0, sizeof(ipImageKeyValPair));

    ipImageKeyValPair[0].key = LwSciBufImageAttrKey_PlaneCount;
    ipImageKeyValPair[1].key = LwSciBufImageAttrKey_PlaneBaseAddrAlign;

    err = LwSciBufAttrListCommonGetAttrs(attrList, 0, &ipImageKeyValPair,
            INPUT_ALIGN_KEYVAL_PAIR_COUNT, LwSciBufAttrKeyType_Public, true);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("Failed to get key \n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
    planeCount = *(const uint32_t*)ipImageKeyValPair[0].value;

    LwSciCommonMemcpyS(&planeAlignment[0], sizeof(planeAlignment),
                        ipImageKeyValPair[1].value, ipImageKeyValPair[1].len);

    for (planeNum = 0U; planeNum < planeCount; planeNum++) {
        u32Mul(level, planeCount, &tmpMul, &mulStatus);
        u32Add(planeNum, tmpMul, &index, &addStatus);

        if (OP_SUCCESS != (addStatus & mulStatus)) {
            LWSCI_ERR_STR("Buffer overflow\n");
            err = LwSciError_Overflow;
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
            goto ret;
        }

        if (planeAlignment[index] == 0U) {
            /* Default value */
            planeAlignment[index] = 1U;
        }

        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Directive, 4_9), "LwSciBuf-ADV-MISRAC2012-008")
        planeAlignment[index] = LW_SCI_BUF_MAX_NUM(constraints->startAddrAlign,
                                                    planeAlignment[index]);

        LWSCI_INFO("index %" PRIu32 " align %" PRIu32 "\n", index,
            planeAlignment[index]);
    }

    /* planeCount*(level+1U) is used to write computed values of this level
     *  and all previous levels
     */

    u64Mul(sizeof(planeAlignment[0]), planeCount, &tmpMul2, &mulStatus);
    u32Add(level, 1U, &tmpAdd, &addStatus);

    if (OP_SUCCESS != (mulStatus & addStatus)) {
        LWSCI_ERR_STR("Buffer Overflow\n");
        err = LwSciError_Overflow;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    (void)memset(opImageKeyValPair, 0x0, sizeof(opImageKeyValPair));

    opImageKeyValPair[0].len = tmpMul2 * tmpAdd;
    opImageKeyValPair[0].key = LwSciBufImageAttrKey_PlaneBaseAddrAlign;
    opImageKeyValPair[0].value = &planeAlignment[0];

    err = LwSciBufAttrListCommonSetAttrs(attrList, 0, &opImageKeyValPair,
            OUTPUT_ALIGN_KEYVAL_PAIR_COUNT, LwSciBufAttrKeyType_Public, true,
            false);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("Failed to set key \n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

ret:
    LWSCI_FNEXIT("");
    return err;
}

static LwSciError LwSciBufConstrainComputePitch(
    LwSciBufAttrList attrList,
    const LwSciBufHwConstraints* constraints,
    uint32_t level)
{
    LwSciError err = LwSciError_Success;
    LwSciBufAttrKeyValuePair ipImageKeyValPair[INPUT_PITCH_KEYVAL_PAIR_COUNT];
    LwSciBufAttrKeyValuePair opImageKeyValPair[OUTPUT_PITCH_KEYVAL_PAIR_COUNT];
    uint32_t planeNum = 0U;
    uint32_t planeWidth[LW_SCI_BUF_PYRAMID_MAX_PLANES] = {0};
    uint32_t planeWidthInBits = 0U;
    uint32_t planeWidthInBytes = 0U;
    uint32_t planeCount = 0U;
    LwColorFormat lwColorFmt = LwColorFormat_Unspecified;
    uint32_t planePitch[LW_SCI_BUF_PYRAMID_MAX_PLANES] = {0};
    uint32_t colorBPP[LW_SCI_BUF_IMAGE_MAX_PLANES] = {0};
    LwSciBufAttrValColorFmt planeColorFmt[LW_SCI_BUF_IMAGE_MAX_PLANES];
    uint32_t index = 0U;
    uint32_t tmpMul = 0U;
    uint8_t addStatus = OP_FAIL;
    uint8_t mulStatus = OP_FAIL;

    LWSCI_FNENTRY("");

    LWSCI_INFO("Input: attrList %p, constraints %p\n", attrList,
        constraints);

    (void)memset(ipImageKeyValPair, 0x0, sizeof(ipImageKeyValPair));

    ipImageKeyValPair[0].key = LwSciBufImageAttrKey_PlaneCount;
    ipImageKeyValPair[1].key = LwSciBufImageAttrKey_PlaneWidth;
    ipImageKeyValPair[2].key = LwSciBufImageAttrKey_PlanePitch;
    ipImageKeyValPair[3].key = LwSciBufImageAttrKey_PlaneColorFormat;

    err = LwSciBufAttrListCommonGetAttrs(attrList, 0, &ipImageKeyValPair,
            INPUT_PITCH_KEYVAL_PAIR_COUNT, LwSciBufAttrKeyType_Public, true);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("Failed to get key\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    /* LwSciBufImageAttrKey_PlaneCount
     * Get Plane count
     */
    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
    planeCount = *(const uint32_t*)ipImageKeyValPair[0].value;

    /* LwSciBufImageAttrKey_PlaneWidth
     * Get Width for all planes
     */
    LwSciCommonMemcpyS(&planeWidth[0], sizeof(planeWidth),
                        ipImageKeyValPair[1].value,
                        (uint64_t)ipImageKeyValPair[1].len);

    /* LwSciBufImageAttrKey_PlanePitch
     * Get Pitch for all planes
     */
    LwSciCommonMemcpyS(&planePitch[0], sizeof(planePitch),
                        ipImageKeyValPair[2].value,
                        (uint64_t)ipImageKeyValPair[2].len);

    /* LwSciBufImageAttrKey_PlaneColorFormat
     * Get Color Format for all planes
     */
    LwSciCommonMemcpyS(&planeColorFmt[0], sizeof(planeColorFmt),
                        ipImageKeyValPair[3].value,
                        (uint64_t)ipImageKeyValPair[3].len);

    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_4), "LwSciBuf-ADV-MISRAC2012-016")
    for (planeNum = 0U; planeNum < planeCount; planeNum++) {
        //index = planeNum + planeCount * level
        u32Mul(planeCount, level, &tmpMul, &mulStatus);
        u32Add(planeNum, tmpMul, &index, &addStatus);

        if (OP_SUCCESS != (addStatus & mulStatus)) {
            LWSCI_ERR_STR("Buffer overflow\n");
            err = LwSciError_Overflow;
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
            goto ret;
        }
        err = LwSciColorToLwColor(planeColorFmt[planeNum], &lwColorFmt);
        if (LwSciError_Success != err) {
            LWSCI_ERR_STR("Failed to colwert color format\n");
            LWSCI_ERR_HEXUINT("planeColorFmt[planeNum]: ", (uint64_t)planeColorFmt[planeNum]);
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
            goto ret;
        }

        colorBPP[planeNum] = LwColorGetBPP(lwColorFmt);
        planeWidthInBits = planeWidth[index]*colorBPP[planeNum];
        planeWidthInBytes = (planeWidthInBits + 7U) >> 3U;

        err = LwSciBufAliglwalue32(planeWidthInBytes, constraints->pitchAlign,
            &planePitch[index]);
        if (LwSciError_Success != err) {
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
            goto ret;
        }

        LWSCI_INFO("colorBPP %" PRIu32 "\n", colorBPP[planeNum]);
        LWSCI_INFO("plane %" PRIu32 " planepitch %" PRIu32 "\n",
            (uint32_t)index, (uint32_t)planePitch[index]);
    }

    (void)memset(opImageKeyValPair, 0x0, sizeof(opImageKeyValPair));

    /* LwSciBufImageAttrKey_PlanePitch
     * Set Pitch for all planes
     */
    opImageKeyValPair[0].value = &planePitch[0];
    opImageKeyValPair[0].key = LwSciBufImageAttrKey_PlanePitch;
    opImageKeyValPair[0].len =
                            sizeof(planePitch[0]) * planeCount * (level + 1U);

    /* LwSciBufImageAttrKey_PlaneBitsPerPixel
     * Set Bits per pixel for all planes
     */
    opImageKeyValPair[1].value = &colorBPP[0];
    opImageKeyValPair[1].key = LwSciBufImageAttrKey_PlaneBitsPerPixel;
    opImageKeyValPair[1].len = sizeof(colorBPP[0])*planeCount;

    err = LwSciBufAttrListCommonSetAttrs(attrList, 0, &opImageKeyValPair,
            OUTPUT_PITCH_KEYVAL_PAIR_COUNT, LwSciBufAttrKeyType_Public, true,
            false);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("Failed to set key \n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

ret:
    LWSCI_FNEXIT("");
    return err;
}

static LwSciError LwSciBufConstrainComputeAlignedHeight(
    LwSciBufAttrList attrList,
    const LwSciBufHwConstraints* constraints,
    uint32_t level)
{
    LwSciError err = LwSciError_Success;
    LwSciBufAttrKeyValuePair
                    ipImageKeyValPair[INPUT_ALIGN_HEIGHT_KEYVAL_PAIR_COUNT];
    LwSciBufAttrKeyValuePair
                    opImageKeyValPair[OUTPUT_ALIGN_HEIGHT_KEYVAL_PAIR_COUNT];
    uint32_t planeNum = 0U;
    uint32_t planeCount = 0U;
    uint32_t halfPlaneHeight = 0U;
    uint32_t planeAlignedHeight[LW_SCI_BUF_PYRAMID_MAX_PLANES] = {0};
    uint32_t planeHeight[LW_SCI_BUF_PYRAMID_MAX_PLANES] = {0};
    uint32_t aliglwalue = 0U;
    LwSciBufAttrValImageScanType scanType;
    uint32_t index = 0U;
    uint32_t tmpMul = 0U;
    uint8_t mulStatus = OP_FAIL;
    uint8_t addStatus = OP_FAIL;

    LWSCI_FNENTRY("");

    LWSCI_INFO("Input: attrList %p, constraints %p\n", attrList,
        constraints);

    (void)memset(ipImageKeyValPair, 0x0, sizeof(ipImageKeyValPair));

    ipImageKeyValPair[0].key = LwSciBufImageAttrKey_PlaneCount;
    ipImageKeyValPair[1].key = LwSciBufImageAttrKey_ScanType;
    ipImageKeyValPair[2].key = LwSciBufImageAttrKey_PlaneHeight;
    ipImageKeyValPair[3].key = LwSciBufImageAttrKey_PlaneAlignedHeight;

    err = LwSciBufAttrListCommonGetAttrs(attrList, 0, &ipImageKeyValPair,
            INPUT_ALIGN_HEIGHT_KEYVAL_PAIR_COUNT, LwSciBufAttrKeyType_Public,
            true);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("Failed to get key \n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    /* LwSciBufImageAttrKey_PlaneCount
     * Get Plane Count
     */
    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
    planeCount = *(const uint32_t*)ipImageKeyValPair[0].value;

    /* LwSciBufImageAttrKey_ScanType
     * Get Scan type
     */
    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
    scanType = *(const LwSciBufAttrValImageScanType*)ipImageKeyValPair[1].value;

    /* LwSciBufImageAttrKey_PlaneHeight
     * Get Height for all planes
     */
    LwSciCommonMemcpyS(&planeHeight[0], sizeof(planeHeight),
                        ipImageKeyValPair[2].value,
                        (uint64_t)ipImageKeyValPair[2].len);

    /* LwSciBufImageAttrKey_PlaneAlignedHeight
     * Get Height alignment for all planes
     */
    LwSciCommonMemcpyS(&planeAlignedHeight[0], sizeof(planeAlignedHeight),
                        ipImageKeyValPair[3].value,
                        (uint64_t)ipImageKeyValPair[3].len);

    aliglwalue = (uint32_t)constraints->heightAlign;

    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_4), "LwSciBuf-ADV-MISRAC2012-016")
    for (planeNum = 0U; planeNum < planeCount; planeNum++) {
        //index = planeNum + planeCount * level
        u32Mul(planeCount, level, &tmpMul, &mulStatus);

        u32Add(planeNum, tmpMul, &index, &addStatus);

        if (OP_SUCCESS != (mulStatus & addStatus)) {
            LWSCI_ERR_STR("Buffer Overflow\n");
            err = LwSciError_Overflow;
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
            goto ret;
        }

        if (LwSciBufScan_InterlaceType == scanType) {
            halfPlaneHeight = planeHeight[index] >> 1U;
            err = LwSciBufAliglwalue32(halfPlaneHeight, aliglwalue,
                &planeAlignedHeight[index]);
            if (LwSciError_Success != err) {
                LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
                goto ret;
            }
        } else {
            err = LwSciBufAliglwalue32(planeHeight[index], aliglwalue,
                &planeAlignedHeight[index]);
            if (LwSciError_Success != err) {
                LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
                goto ret;
            }
        }

        LWSCI_INFO("plane %" PRIu32 " alignHeight %" PRIu32 "\n",
            (uint32_t)index, (uint32_t)planeAlignedHeight[index]);
    }

    (void)memset(opImageKeyValPair, 0x0, sizeof(opImageKeyValPair));
    /* LwSciBufImageAttrKey_PlaneAlignedHeight
     * Set Height alignment for all planes
     */
    opImageKeyValPair[0].value = &planeAlignedHeight[0];
    opImageKeyValPair[0].key = LwSciBufImageAttrKey_PlaneAlignedHeight;
    opImageKeyValPair[0].len =
                        sizeof(planeAlignedHeight[0])*planeCount*(level + 1U);
    err = LwSciBufAttrListCommonSetAttrs(attrList, 0, &opImageKeyValPair,
            OUTPUT_ALIGN_HEIGHT_KEYVAL_PAIR_COUNT, LwSciBufAttrKeyType_Public,
            true, false);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("Failed to set key \n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

ret:
    LWSCI_FNEXIT("");
    return err;
}

static LwSciError LwSciBufConstrainComputeSize(
    LwSciBufAttrList attrList,
    uint32_t level)
{
    LwSciError err = LwSciError_Success;
    LwSciBufAttrKeyValuePair ipImageKeyValPair[INPUT_SIZE_KEYVAL_PAIR_COUNT];
    LwSciBufAttrKeyValuePair opImageKeyValPair[OUTPUT_SIZE_KEYVAL_PAIR_COUNT];
    uint32_t planeCount = 0U;
    uint32_t planeNum = 0U;
    uint32_t planeAlignedHeight[LW_SCI_BUF_PYRAMID_MAX_PLANES] = {0};
    uint32_t planeAlignment[LW_SCI_BUF_PYRAMID_MAX_PLANES] = {0};
    uint64_t planeAlignmentSize[LW_SCI_BUF_PYRAMID_MAX_PLANES] = {0};
    uint32_t planePitch[LW_SCI_BUF_PYRAMID_MAX_PLANES] = {0};
    uint64_t planeOffset[LW_SCI_BUF_PYRAMID_MAX_PLANES] = {0};
    uint64_t secondFieldOffset[LW_SCI_BUF_PYRAMID_MAX_PLANES] = {0};
    uint64_t firstHalfPlaneSize = 0U;
    uint64_t secondHalfPlaneSize = 0U;
    uint64_t planeSize = 0U;
    uint64_t offset = 0U;
    LwSciBufAttrValImageScanType scanType = 0U;
    uint32_t index = 0U;
    uint32_t tmpMul = 0U;
    uint8_t mulStatus = OP_FAIL;
    uint8_t addStatus = OP_FAIL;

    LWSCI_FNENTRY("");

    LWSCI_INFO("Input: attrList %p\n", attrList);

    (void)memset(ipImageKeyValPair, 0x0, sizeof(ipImageKeyValPair));

    ipImageKeyValPair[0].key = LwSciBufImageAttrKey_PlaneCount;
    ipImageKeyValPair[1].key = LwSciBufImageAttrKey_PlanePitch;
    ipImageKeyValPair[2].key = LwSciBufImageAttrKey_PlaneAlignedHeight;
    ipImageKeyValPair[3].key = LwSciBufImageAttrKey_ScanType;
    ipImageKeyValPair[4].key = LwSciBufImageAttrKey_PlaneBaseAddrAlign;
    ipImageKeyValPair[5].key = LwSciBufImageAttrKey_PlaneAlignedSize;
    ipImageKeyValPair[6].key = LwSciBufImageAttrKey_PlaneOffset;
    ipImageKeyValPair[7].key = LwSciBufImageAttrKey_PlaneSecondFieldOffset;

    err = LwSciBufAttrListCommonGetAttrs(attrList, 0, &ipImageKeyValPair,
            INPUT_SIZE_KEYVAL_PAIR_COUNT, LwSciBufAttrKeyType_Public, true);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("Failed to get key \n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    /* LwSciBufImageAttrKey_PlaneCount
     * Get Plane count
     */
    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
    planeCount = *(const uint32_t*)ipImageKeyValPair[0].value;

    /* LwSciBufImageAttrKey_PlanePitch
     * Get Pitch for all planes
     */
    LwSciCommonMemcpyS(&planePitch[0], sizeof(planePitch),
                        ipImageKeyValPair[1].value,
                        (uint64_t)ipImageKeyValPair[1].len);

    /* LwSciBufImageAttrKey_PlaneAlignedHeight
     * Get Aligned height for all planes
     */
    LwSciCommonMemcpyS(&planeAlignedHeight[0], sizeof(planeAlignedHeight),
                        ipImageKeyValPair[2].value,
                        (uint64_t)ipImageKeyValPair[2].len);

    /* LwSciBufImageAttrKey_ScanType
     * Get Scan type
     */
    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
    scanType = *(const LwSciBufAttrValImageScanType*)ipImageKeyValPair[3].value;

    /* LwSciBufImageAttrKey_PlaneBaseAddrAlign
     * Get Base address alignment for all planes
     */
    LwSciCommonMemcpyS(&planeAlignment[0], sizeof(planeAlignment),
                        ipImageKeyValPair[4].value,
                        (uint64_t)ipImageKeyValPair[4].len);

    /* LwSciBufImageAttrKey_PlaneAlignedSize
     * Get Size alignment for all planes
     */
    LwSciCommonMemcpyS(&planeAlignmentSize[0], sizeof(planeAlignmentSize),
                        ipImageKeyValPair[5].value,
                        (uint64_t)ipImageKeyValPair[5].len);

    /* LwSciBufImageAttrKey_PlaneOffset
     * Get offset for all planes
     */
    LwSciCommonMemcpyS(&planeOffset[0], sizeof(planeOffset),
                        ipImageKeyValPair[6].value,
                        (uint64_t)ipImageKeyValPair[6].len);

    /* LwSciBufImageAttrKey_PlaneSecondFieldOffset
     * Get Second field offset for all planes
     */
    LwSciCommonMemcpyS(&secondFieldOffset[0], sizeof(secondFieldOffset),
                        ipImageKeyValPair[7].value,
                        (uint64_t)ipImageKeyValPair[7].len);

    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_4), "LwSciBuf-ADV-MISRAC2012-016")
    for (planeNum = 0U; planeNum < planeCount; planeNum++) {

        // index = planeNum + planeCount*level
        u32Mul(planeCount, level, &tmpMul, &mulStatus);
        u32Add(planeNum, tmpMul, &index, &addStatus);

        if (OP_SUCCESS != (addStatus & mulStatus)) {
            LWSCI_ERR_STR("Buffer overflow\n");
            err = LwSciError_Overflow;
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
            goto ret;
        }

        planeOffset[index] = offset;

        firstHalfPlaneSize = (uint64_t)planePitch[index]*
                (uint64_t)planeAlignedHeight[index];
        err = LwSciBufAliglwalue64(firstHalfPlaneSize, planeAlignment[index],
            &firstHalfPlaneSize);
        if (LwSciError_Success != err) {
            LWSCI_ERR_STR("Failed to align first half plane size");
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
            goto ret;
        }

        if (LwSciBufScan_InterlaceType == scanType) {
            secondHalfPlaneSize = firstHalfPlaneSize;
            secondFieldOffset[index] = offset + firstHalfPlaneSize;
        } else {
            secondHalfPlaneSize = 0U;
            secondFieldOffset[index] = 0;
        }

        planeSize = firstHalfPlaneSize + secondHalfPlaneSize;

        err = LwSciBufAliglwalue64(planeSize,
                                LWSCIBUF_PREFERRED_PLANE_ALIGNMENT_CONSTRAINT,
                                &planeAlignmentSize[index]);
        if (LwSciError_Success != err) {
            LWSCI_ERR_STR("Failed to align plane size");
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
            goto ret;
        }

        offset += planeAlignmentSize[index];

        LWSCI_INFO("index %" PRIu32 " planeAlignmentSize %" PRIu32 "\n",
            (uint32_t)index, (uint32_t)planeAlignmentSize[index]);
        LWSCI_INFO("index %" PRIu32 " planeOffset %" PRIu32 "\n",
            (uint32_t)index, (uint32_t)planeOffset[index]);
        LWSCI_INFO("index %" PRIu32 " secondFieldOffset %" PRIu32 "\n",
            (uint32_t)index, (uint32_t)secondFieldOffset[index]);
    }

    (void)memset(opImageKeyValPair, 0x0, sizeof(opImageKeyValPair));

    /* LwSciBufImageAttrKey_PlaneAlignedSize
     * Set Size alignment for all planes
     */
    opImageKeyValPair[0].value = &planeAlignmentSize[0];
    opImageKeyValPair[0].key = LwSciBufImageAttrKey_PlaneAlignedSize;
    opImageKeyValPair[0].len =
                        sizeof(planeAlignmentSize[0])*planeCount*(level + 1U);

    /* LwSciBufImageAttrKey_PlaneOffset
     * Set offset for all planes
     */
    opImageKeyValPair[1].value = &planeOffset[0];
    opImageKeyValPair[1].key = LwSciBufImageAttrKey_PlaneOffset;
    opImageKeyValPair[1].len = sizeof(planeOffset[0])*planeCount*(level + 1U);

    /* LwSciBufImageAttrKey_PlaneSecondFieldOffset
     * Set Second field offset for all planes
     */
    opImageKeyValPair[2].value = &secondFieldOffset[0];
    opImageKeyValPair[2].key = LwSciBufImageAttrKey_PlaneSecondFieldOffset;
    opImageKeyValPair[2].len =
                        sizeof(secondFieldOffset[0])*planeCount*(level + 1U);

    err = LwSciBufAttrListCommonSetAttrs(attrList, 0, &opImageKeyValPair,
            OUTPUT_SIZE_KEYVAL_PAIR_COUNT, LwSciBufAttrKeyType_Public, true,
            false);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("Failed to set key \n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

ret:
    LWSCI_FNEXIT("");
    return err;
}

static LwSciError LwSciBufConstrainComputeImageSizeAlign(
    LwSciBufAttrList attrList,
    uint32_t level)
{
    LwSciError err = LwSciError_Success;
    LwSciBufAttrKeyValuePair
                    ipImageKeyValPair[INPUT_IMAGESIZE_KEYVAL_PAIR_COUNT];
    LwSciBufAttrKeyValuePair
                    opImageKeyValPair[OUTPUT_IMAGESIZE_KEYVAL_PAIR_COUNT];
    uint32_t planeAlignment[LW_SCI_BUF_PYRAMID_MAX_PLANES] = {0};
    uint64_t planeAlignedSize[LW_SCI_BUF_PYRAMID_MAX_PLANES] = {0};
    uint64_t imageSize = 0U;
    uint64_t imageAlignment = 0U;
    uint32_t planeCount = 0U;
    uint32_t planeNum = 0U;
    uint32_t index = 0U;
    uint32_t tmpMul = 0U;
    uint8_t mulStatus = OP_FAIL;
    uint8_t addStatus = OP_FAIL;
    uint8_t addStatus2 = OP_FAIL;

    LWSCI_FNENTRY("");

    LWSCI_INFO("Input: attrList %p\n", attrList);

    (void)memset(ipImageKeyValPair, 0x0, sizeof(ipImageKeyValPair));

    ipImageKeyValPair[0].key = LwSciBufImageAttrKey_PlaneCount;
    ipImageKeyValPair[1].key = LwSciBufImageAttrKey_PlaneAlignedSize;
    ipImageKeyValPair[2].key = LwSciBufImageAttrKey_PlaneBaseAddrAlign;
    ipImageKeyValPair[3].key = LwSciBufImageAttrKey_Size;
    ipImageKeyValPair[4].key = LwSciBufImageAttrKey_Alignment;

    err = LwSciBufAttrListCommonGetAttrs(attrList, 0, &ipImageKeyValPair,
            INPUT_IMAGESIZE_KEYVAL_PAIR_COUNT, LwSciBufAttrKeyType_Public,
            true);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("Failed to get key\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    /* LwSciBufImageAttrKey_PlaneCount
     * Get Plane count
     */
    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
    planeCount = *(const uint32_t*)ipImageKeyValPair[0].value;

    /* LwSciBufImageAttrKey_PlaneAlignedSize
     * Get Aligned size for all planes
     */
    LwSciCommonMemcpyS(&planeAlignedSize[0], sizeof(planeAlignedSize),
                        ipImageKeyValPair[1].value,
                        (uint64_t)ipImageKeyValPair[1].len);

    /* LwSciBufImageAttrKey_PlaneBaseAddrAlign
     * Get Base Address alignment for all planes
     */
    LwSciCommonMemcpyS(&planeAlignment[0], sizeof(planeAlignment),
                        ipImageKeyValPair[2].value,
                        (uint64_t)ipImageKeyValPair[2].len);

    if (level != 0U) {
        /* Read the imageSize from previous level here. For image-pyramid,
         * the total imageSize is sum of the sizes of all levels.
         * Also read imageAlignment of previous level here. For level 0, the
         * imageAlignment is set as alignment of 0th plane of 0th level. For
         * all other levels, the imageAlignment is set for level 0 is reused.
         */
        imageSize = *(const uint64_t*)ipImageKeyValPair[3].value;
        imageAlignment = *(const uint64_t*)ipImageKeyValPair[4].value;
    }

    for (planeNum = 0U; planeNum < planeCount; planeNum++) {
        //index = planeNum + planeCount*level
        u32Mul(planeCount, level, &tmpMul, &mulStatus);
        u32Add(planeNum, tmpMul, &index, &addStatus);

        //imageSize += planeAlignedSize[index]
        u64Add(imageSize, planeAlignedSize[index], &imageSize, &addStatus2);

        if (OP_SUCCESS != (mulStatus & addStatus & addStatus2)) {
            LWSCI_ERR_STR("Buffer overflow\n");
            err = LwSciError_Overflow;
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
            goto ret;
        }

        if (0U == index) {
            imageAlignment = planeAlignment[0];
        }
    }

    LWSCI_INFO("imageSize %" PRIu32 "\n", (uint32_t)imageSize);
    LWSCI_INFO("imageAlignment %" PRIu32 "\n", (uint32_t)imageAlignment);

    (void)memset(opImageKeyValPair, 0x0, sizeof(opImageKeyValPair));

    /* LwSciBufImageAttrKey_Size
     * Set Image size
     */
    opImageKeyValPair[0].value = &imageSize;
    opImageKeyValPair[0].key = LwSciBufImageAttrKey_Size;
    opImageKeyValPair[0].len = sizeof(imageSize);

    /* LwSciBufImageAttrKey_Alignment
     * Set Image alignment
     */
    opImageKeyValPair[1].value = &imageAlignment;
    opImageKeyValPair[1].key = LwSciBufImageAttrKey_Alignment;
    opImageKeyValPair[1].len = sizeof(imageAlignment);

    err = LwSciBufAttrListCommonSetAttrs(attrList, 0, &opImageKeyValPair,
            OUTPUT_IMAGESIZE_KEYVAL_PAIR_COUNT, LwSciBufAttrKeyType_Public,
            true, false);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("Failed to set key\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

ret:
    LWSCI_FNEXIT("");
    return err;
}

static LwSciError LwSciBufComputeImageOutputAttr(
    LwSciBufAttrList attrList)
{
    LwSciError err = LwSciError_Success;
    LwSciBufAttrKeyValuePair ipImagePublicKeyValPair[INPUT_KEYVAL_PAIR_COUNT];
    LwSciBufAttrKeyValuePair opImagePublicKeyValPair[OUTPUT_KEYVAL_PAIR_COUNT];
    LwSciBufInternalAttrKeyValuePair
                            opImageIntKeyValPair[OUTPUT_INT_KEYVAL_PAIR_COUNT];
    uint32_t planeCount = 0U;
    uint32_t planeNum = 0U;
    uint32_t gobSize[LW_SCI_BUF_IMAGE_MAX_PLANES] = {0};
    uint32_t gobPerBlockX[LW_SCI_BUF_IMAGE_MAX_PLANES] = {0};
    uint32_t gobPerBlockY[LW_SCI_BUF_IMAGE_MAX_PLANES] = {0};
    uint32_t gobPerBlockZ[LW_SCI_BUF_IMAGE_MAX_PLANES] = {0};
    uint8_t channelCount[LW_SCI_BUF_IMAGE_MAX_PLANES] = {0};
    uint32_t colorBPP[LW_SCI_BUF_IMAGE_MAX_PLANES] = {0};

    LwColorFormat colorFmt = 0U;
    LwColorDataType colorDataType = 0U;

    LwSciBufAttrValDataType planeDataType[LW_SCI_BUF_IMAGE_MAX_PLANES] = {0};
    LwSciBufAttrValColorFmt planeColorFmt[LW_SCI_BUF_IMAGE_MAX_PLANES] = {0};

    LWSCI_FNENTRY("");

    LWSCI_INFO("Input: attrList %p\n", attrList);

    (void)memset(ipImagePublicKeyValPair, 0x0, sizeof(ipImagePublicKeyValPair));

    ipImagePublicKeyValPair[0].key = LwSciBufImageAttrKey_PlaneCount;
    ipImagePublicKeyValPair[1].key = LwSciBufImageAttrKey_PlaneColorFormat;

    err = LwSciBufAttrListCommonGetAttrs(attrList, 0, &ipImagePublicKeyValPair,
            INPUT_KEYVAL_PAIR_COUNT, LwSciBufAttrKeyType_Public, true);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("Failed to get key\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    /* LwSciBufImageAttrKey_PlaneCount
     * Get Plane count
     */
    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
    planeCount = *(const uint32_t*)ipImagePublicKeyValPair[0].value;

    /* LwSciBufImageAttrKey_PlaneColorFormat
     * Get Color format for all planes
     */
    LwSciCommonMemcpyS(&planeColorFmt[0], sizeof(planeColorFmt),
                    ipImagePublicKeyValPair[1].value,
                    (uint64_t)ipImagePublicKeyValPair[1].len);

    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_4), "LwSciBuf-ADV-MISRAC2012-016")
    for (planeNum = 0U; planeNum < planeCount; planeNum++) {
        /* TODO need to fix this hardcoding */
        /* lwbugs/200671636 [LwSciBuf] Fix default image constraints in constraint library */
        gobSize[planeNum] = 0U;
        gobPerBlockX[planeNum] = 0U;
        gobPerBlockY[planeNum] = 1U;
        gobPerBlockZ[planeNum] = 0U;

        err = LwSciColorToLwColor(planeColorFmt[planeNum], &colorFmt);
        if (LwSciError_Success != err) {
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
            goto ret;
        }

        err = LwSciColorGetComponentCount(planeColorFmt[planeNum],
                &channelCount[planeNum]);
        if (LwSciError_Success != err) {
            LWSCI_ERR_ULONG("Failed to get channel count from LwSciColor\n", (uint64_t)planeColorFmt[planeNum]);
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
            goto ret;
        }

        colorDataType = LwColorGetDataType(colorFmt);
        colorBPP[planeNum] = LwColorGetBPP(colorFmt);

        planeDataType[planeNum] = LwColorDataTypeToLwSciBufDataType(
                colorDataType, channelCount[planeNum], colorBPP[planeNum]);

        LWSCI_INFO("plane %" PRIu32 " gobSize %" PRIu32 "\n",
            (uint32_t)planeNum, (uint32_t)gobSize[planeNum]);
        LWSCI_INFO("plane %" PRIu32 " gobPerBlockX %" PRIu32 "\n",
            (uint32_t)planeNum, (uint32_t)gobPerBlockX[planeNum]);
        LWSCI_INFO("plane %" PRIu32 " gobPerBlockY %" PRIu32 "\n",
            (uint32_t)planeNum, (uint32_t)gobPerBlockY[planeNum]);
        LWSCI_INFO("plane %" PRIu32 " gobPerBlockZ %" PRIu32 "\n",
            (uint32_t)planeNum, (uint32_t)gobPerBlockZ[planeNum]);
        LWSCI_INFO("plane %" PRIu32 " channelCount %" PRIu32 "\n",
            (uint32_t)planeNum, (uint32_t)channelCount[planeNum]);
        LWSCI_INFO("plane %" PRIu32 " planeDataType %" PRIu32 "\n",
            (uint32_t)planeNum, (uint32_t)planeDataType[planeNum]);
    }

    (void)memset(opImageIntKeyValPair, 0x0, sizeof(opImageIntKeyValPair));

    /* LwSciBufInternalImageAttrKey_PlaneGobSize
     * Set GOB size for all planes
     */
    opImageIntKeyValPair[0].value = &gobSize[0];
    opImageIntKeyValPair[0].key = LwSciBufInternalImageAttrKey_PlaneGobSize;
    opImageIntKeyValPair[0].len = sizeof(gobSize[0])*planeNum;

    /* LwSciBufInternalImageAttrKey_PlaneLog2GobsPerBlockX
     * Set Log2GOB per Block X for all planes
     */
    opImageIntKeyValPair[1].value = &gobPerBlockX[0];
    opImageIntKeyValPair[1].key =
                            LwSciBufInternalImageAttrKey_PlaneLog2GobsPerBlockX;
    opImageIntKeyValPair[1].len = sizeof(gobPerBlockX[0])*planeNum;

    /* LwSciBufInternalImageAttrKey_PlaneLog2GobsPerBlockY
     * Set Log2GOB per Block Y for all planes
     */
    opImageIntKeyValPair[2].value = &gobPerBlockY[0];
    opImageIntKeyValPair[2].key =
                            LwSciBufInternalImageAttrKey_PlaneLog2GobsPerBlockY;
    opImageIntKeyValPair[2].len = sizeof(gobPerBlockY[0])*planeNum;

    /* LwSciBufInternalImageAttrKey_PlaneLog2GobsPerBlockZ
     * Set Log2GOB per Block Z for all planes
     */
    opImageIntKeyValPair[3].value = &gobPerBlockZ[0];
    opImageIntKeyValPair[3].key =
                            LwSciBufInternalImageAttrKey_PlaneLog2GobsPerBlockZ;
    opImageIntKeyValPair[3].len = sizeof(gobPerBlockZ[0])*planeNum;

    err = LwSciBufAttrListCommonSetAttrs(attrList, 0, &opImageIntKeyValPair,
            OUTPUT_INT_KEYVAL_PAIR_COUNT, LwSciBufAttrKeyType_Internal, true,
            false);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("Failed to set internal keys\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    (void)memset(opImagePublicKeyValPair, 0x0, sizeof(opImagePublicKeyValPair));
    /* LwSciBufImageAttrKey_PlaneChannelCount
     * Set Channel count for all planes
     */
    opImagePublicKeyValPair[0].value = &channelCount[0];
    opImagePublicKeyValPair[0].key = LwSciBufImageAttrKey_PlaneChannelCount;
    opImagePublicKeyValPair[0].len = sizeof(channelCount[0])*planeNum;

    /* LwSciBufImageAttrKey_PlaneDatatype
     * Set Data type for all planes
     */
    opImagePublicKeyValPair[1].value = &planeDataType[0];
    opImagePublicKeyValPair[1].key = LwSciBufImageAttrKey_PlaneDatatype;
    opImagePublicKeyValPair[1].len = sizeof(planeDataType[0])*planeNum;

    err = LwSciBufAttrListCommonSetAttrs(attrList, 0, &opImagePublicKeyValPair,
            OUTPUT_KEYVAL_PAIR_COUNT, LwSciBufAttrKeyType_Public, true,
            false);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("Failed to set key\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

ret:
    LWSCI_FNEXIT("");
    return err;
}

static LwSciError LwSciBufCommonImageConstraint(
    LwSciBufAttrList attrList,
    const LwSciBufHwConstraints* constraints,
    uint32_t level)
{
    LwSciError err = LwSciError_Success;

    LWSCI_FNENTRY("");

    err = LwSciBufConstrainComputeAlignment(attrList, constraints, level);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("Failed to compute alignment constraints.\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    err = LwSciBufConstrainComputePitch(attrList, constraints, level);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("Failed to compute pitch constraint.\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    err = LwSciBufConstrainComputeAlignedHeight(attrList, constraints, level);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("Failed to compute height constraint.\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    err = LwSciBufConstrainComputeSize(attrList, level);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("Failed to compute size constraint.\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    err = LwSciBufComputeImageOutputAttr(attrList);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("Failed to compute image output constraint.\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    err = LwSciBufConstrainComputeImageSizeAlign(attrList, level);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("Failed to compute size and alignment.\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

ret:
    LWSCI_FNEXIT("");
    return err;
}

static LwSciError LwSciBufImageConstraint(
    LwSciBufAttrList attrList,
    const LwSciBufHwConstraints* constraints)
{
    LwSciError err = LwSciError_Success;
    LwSciBufAttrKeyValuePair
                ipImagePublicKeyValPair[CONSTRAINT_PUBLIC_KEYVAL_PAIR_COUNT];
    LwSciBufPrivateAttrKeyValuePair
                opImagePrivateKeyValPair[CONSTRAINT_PRIVATE_KEYVAL_PAIR_COUNT];
    uint64_t len = 0U, imageCount = 0U, imageSize = 0U, nImageSize = 0U;
    uint8_t mulStatus = OP_FAIL;

    LWSCI_FNENTRY("");

    LWSCI_INFO("Input: attrList ptr %p, constraints ptr %p\n", attrList,
        constraints);

    /* For only image datatype we have only one level */
    err = LwSciBufCommonImageConstraint(attrList, constraints, 0U);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("Failed to compute common image attributes\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    (void)memset(ipImagePublicKeyValPair, 0x0, sizeof(ipImagePublicKeyValPair));

    ipImagePublicKeyValPair[0].key = LwSciBufImageAttrKey_Size;
    ipImagePublicKeyValPair[1].key = LwSciBufImageAttrKey_ImageCount;
    ipImagePublicKeyValPair[2].key = LwSciBufImageAttrKey_Alignment;

    err = LwSciBufAttrListCommonGetAttrs(attrList, 0, &ipImagePublicKeyValPair,
            CONSTRAINT_PUBLIC_KEYVAL_PAIR_COUNT, LwSciBufAttrKeyType_Public,
            true);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("Failed to get image size.\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }
    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
    imageSize = *(const uint64_t*)ipImagePublicKeyValPair[0].value;


    len = (uint64_t)ipImagePublicKeyValPair[1].len;
    if (0U == len) {
        /* image count will be set via key LwSciBufImageAttrKey_ImageCount only
         * for image/tensor interop use-case. For other use-cases, set the
         * default value to 1
         */
        imageCount = 1UL;
    } else {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
        imageCount = *(const uint64_t*)ipImagePublicKeyValPair[1].value;
    }

    (void)memset(opImagePrivateKeyValPair, 0x0,
                sizeof(opImagePrivateKeyValPair));

    u64Mul(imageSize, imageCount, &nImageSize, &mulStatus);
    if (OP_SUCCESS != mulStatus) {
        LWSCI_ERR_STR("Buffer overflow\n");
        err = LwSciError_Overflow;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    opImagePrivateKeyValPair[0].key = LwSciBufPrivateAttrKey_Size;
    opImagePrivateKeyValPair[0].value = &nImageSize;
    opImagePrivateKeyValPair[0].len = sizeof(uint64_t);

    opImagePrivateKeyValPair[1].key = LwSciBufPrivateAttrKey_Alignment;
    opImagePrivateKeyValPair[1].value = ipImagePublicKeyValPair[2].value;
    opImagePrivateKeyValPair[1].len = sizeof(uint64_t);

    err = LwSciBufAttrListCommonSetAttrs(attrList, 0, &opImagePrivateKeyValPair,
            CONSTRAINT_PRIVATE_KEYVAL_PAIR_COUNT, LwSciBufAttrKeyType_Private,
            true, false);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("Failed to set private keys.\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

ret:
    LWSCI_FNEXIT("");
    return err;
}

static LwSciError LwSciBufPyramidConstraint(
    LwSciBufAttrList attrList,
    const LwSciBufHwConstraints* constraints)
{
    LwSciError err = LwSciError_Success;
    uint64_t totalSize = 0U;
    uint32_t level = 0U;
    uint32_t index = 0U;
    uint32_t levelCount = 0U;
    uint32_t planeCount = 0U;
    uint32_t planeNum = 0U;
    uint32_t width[LW_SCI_BUF_PYRAMID_MAX_PLANES] = {0};
    uint32_t height[LW_SCI_BUF_PYRAMID_MAX_PLANES] = {0};
    uint64_t levelSize[LW_SCI_BUF_PYRAMID_MAX_PLANES] = {0};
    uint64_t levelOffset[LW_SCI_BUF_PYRAMID_MAX_PLANES] = {0};
    const uint64_t* imageSize = NULL;
    const uint64_t* planeAlignment = NULL;
    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Directive, 4_6), "LwSciBuf-ADV-MISRAC2012-006")
    float scale = 0.0F;

    LwSciBufAttrKeyValuePair
                        ipPyramidKeyValPair[INPUT_PYRAMID_KEYVAL_PAIR_COUNT];
    LwSciBufAttrKeyValuePair
                        opPyramidKeyValPair[OUTPUT_PYRAMID_KEYVAL_PAIR_COUNT];
    LwSciBufPrivateAttrKeyValuePair
                    privatePyramidKeyValPair[PYRAMID_PRIVATE_KEYVAL_PAIR_COUNT];

    LWSCI_FNENTRY("");

    if ((NULL == attrList) || (NULL == constraints)) {
        err = LwSciError_BadParameter;
        LWSCI_ERR_STR("Invalid argument to LwSciBufPyramidConstraint\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    LWSCI_INFO("Input: attrList ptr %p, constraints ptr %p\n", attrList,
        constraints);

    (void)memset(ipPyramidKeyValPair, 0x0, sizeof(ipPyramidKeyValPair));

    ipPyramidKeyValPair[0].key = LwSciBufPyramidAttrKey_NumLevels;
    ipPyramidKeyValPair[1].key = LwSciBufPyramidAttrKey_Scale;
    ipPyramidKeyValPair[2].key = LwSciBufImageAttrKey_PlaneWidth;
    ipPyramidKeyValPair[3].key = LwSciBufImageAttrKey_PlaneHeight;
    ipPyramidKeyValPair[4].key = LwSciBufImageAttrKey_Size;
    ipPyramidKeyValPair[5].key = LwSciBufImageAttrKey_PlaneCount;
    ipPyramidKeyValPair[6].key = LwSciBufImageAttrKey_Alignment;

    err = LwSciBufAttrListCommonGetAttrs(attrList, 0, &ipPyramidKeyValPair,
            INPUT_PYRAMID_KEYVAL_PAIR_COUNT, LwSciBufAttrKeyType_Public, true);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("Failed to get Attributes\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
    levelCount = *(const uint32_t*)ipPyramidKeyValPair[0].value;

    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Directive, 4_6), "LwSciBuf-ADV-MISRAC2012-006")
    scale = *(const float*)ipPyramidKeyValPair[1].value;
    LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 11_5))

    LwSciCommonMemcpyS(&width[0], sizeof(width), ipPyramidKeyValPair[2].value,
                        (uint64_t)ipPyramidKeyValPair[2].len);

    LwSciCommonMemcpyS(&height[0], sizeof(height), ipPyramidKeyValPair[3].value,
                        (uint64_t)ipPyramidKeyValPair[3].len);

    LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
    imageSize = (const uint64_t*)ipPyramidKeyValPair[4].value;

    planeCount = *(const uint32_t*)ipPyramidKeyValPair[5].value;

    planeAlignment = (const uint64_t*)ipPyramidKeyValPair[6].value;
    LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 11_5))

    /* Check if inputs are according to PyramidHwConstraints */
    if ((levelCount > constraints->levelCount) ||
        (scale < constraints->scaleFactor)) {
        err = LwSciError_BadParameter;
        LWSCI_ERR_UINT("Input level count ,", levelCount);
        LWSCI_ERR_UINT("Hw constraint level count \n", constraints->levelCount);
        LWSCI_ERR_ULONG("Input scale factor \n", scale);
        LWSCI_ERR_ULONG("Hw constraint scale factor \n", constraints->scaleFactor);
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_4), "LwSciBuf-ADV-MISRAC2012-016")
    for (level = 0U; level < levelCount; level++) {
        /* For level 0 *Key_PlaneWidth & *Key_PlaneHeight is already provided
         *  by the user. Hence we don't have to calwlate it based on scale and
         *  previous level information.
         */
        if (0U != level) {
            uint32_t tmp2 = 0U;

            for (planeNum = 0U; planeNum < planeCount; planeNum++) {
                LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Directive, 4_6), "LwSciBuf-ADV-MISRAC2012-006")
                float tmp1 = 0.0f;
                index = planeNum + (planeCount*level);
                tmp1 = (float)height[index - planeCount] * scale;
                height[index] = (uint32_t)tmp1;
                tmp1 = (float)width[index - planeCount] * scale;
                LWCOV_ALLOWLIST_END(LWCOV_MISRA(Directive, 4_6))
                width[index] = (uint32_t)tmp1;
                LWSCI_INFO("index %" PRIu32 " height %f width %f\n",
                           index, height[index], width[index]);
            }

            tmp2 = (planeCount + (planeCount * level));

            (void)memset(opPyramidKeyValPair, 0x0, sizeof(opPyramidKeyValPair));

            opPyramidKeyValPair[0].key = LwSciBufImageAttrKey_PlaneWidth;
            opPyramidKeyValPair[0].len = sizeof(uint32_t) * tmp2;
            opPyramidKeyValPair[0].value = width;
            opPyramidKeyValPair[1].key = LwSciBufImageAttrKey_PlaneHeight;
            opPyramidKeyValPair[1].len = sizeof(uint32_t) * tmp2;
            opPyramidKeyValPair[1].value = height;

            err = LwSciBufAttrListCommonSetAttrs(attrList, 0,
                        &opPyramidKeyValPair, 2, LwSciBufAttrKeyType_Public,
                        true, false);
            if (LwSciError_Success != err) {
                LWSCI_ERR_STR("Failed to set key.\n");
                LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
                goto ret;
            }

        }

        /* call compute image constraints for level */
        err = LwSciBufCommonImageConstraint(attrList, constraints, level);
        if (LwSciError_Success != err) {
            LWSCI_ERR_STR("Failed to compute image constraints.\n");
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
            goto ret;
        }

        levelSize[level] = *imageSize;

        if (0U != level) {
            levelOffset[level] = levelOffset[level-1U] + levelSize[level-1U];
        } else {
            levelOffset[level] = 0U;
        }

        totalSize += levelSize[level];
        LWSCI_INFO("index %" PRIu32 " levelsize %" PRIu64 " leveloffset %"
                PRIu64 "\n", index, levelSize[level], levelOffset[level]);
    }

    (void)memset(opPyramidKeyValPair, 0x0, sizeof(opPyramidKeyValPair));

    opPyramidKeyValPair[0].key = LwSciBufPyramidAttrKey_LevelOffset;
    opPyramidKeyValPair[0].len = sizeof(uint64_t)*levelCount;
    opPyramidKeyValPair[0].value = levelOffset;

    opPyramidKeyValPair[1].key = LwSciBufPyramidAttrKey_LevelSize;
    opPyramidKeyValPair[1].len = sizeof(uint64_t)*levelCount;
    opPyramidKeyValPair[1].value = &levelSize;

    opPyramidKeyValPair[2].key = LwSciBufPyramidAttrKey_Alignment;
    opPyramidKeyValPair[2].len = sizeof(uint64_t);
    opPyramidKeyValPair[2].value = planeAlignment;

    err = LwSciBufAttrListCommonSetAttrs(attrList, 0, &opPyramidKeyValPair,
            OUTPUT_PYRAMID_KEYVAL_PAIR_COUNT, LwSciBufAttrKeyType_Public, true,
            false);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("Failed to set key.\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    (void)memset(privatePyramidKeyValPair, 0x0,
                    sizeof(privatePyramidKeyValPair));

    privatePyramidKeyValPair[0].key = LwSciBufPrivateAttrKey_Size;
    privatePyramidKeyValPair[0].len = sizeof(uint64_t);
    privatePyramidKeyValPair[0].value = &totalSize;

    privatePyramidKeyValPair[1].key = LwSciBufPrivateAttrKey_Alignment;
    privatePyramidKeyValPair[1].len = sizeof(uint64_t);
    privatePyramidKeyValPair[1].value = planeAlignment;

    err = LwSciBufAttrListCommonSetAttrs(attrList, 0, &privatePyramidKeyValPair,
            PYRAMID_PRIVATE_KEYVAL_PAIR_COUNT, LwSciBufAttrKeyType_Private,
            true, false);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("Failed to set key.\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

ret:
    LWSCI_FNEXIT("");
    return err;
}

static LwSciError LwSciBufRawConstraint(
    LwSciBufAttrList attrList,
    const LwSciBufHwConstraints* constraints)
{
    LwSciError err = LwSciError_Success;
    LwSciBufAttrKeyValuePair
                        rawBufPublicKeyValPair[RAW_PUBLIC_KEYVAL_PAIR_COUNT];
    LwSciBufPrivateAttrKeyValuePair
                        rawBufPrivateKeyValPair[RAW_PRIVATE_KEYVAL_PAIR_COUNT];
    uint64_t rawBufferSize = 0U;
    uint64_t rawBufferAlign = 0U;

    (void)constraints;

    LWSCI_FNENTRY("");

    LWSCI_INFO("Input: attrList %p, constraints ptr %p\n", attrList,
        constraints);

    (void)memset(rawBufPublicKeyValPair, 0x0, sizeof(rawBufPublicKeyValPair));

    /* Get reconciled size and alignment */
    rawBufPublicKeyValPair[0].key = LwSciBufRawBufferAttrKey_Size;
    rawBufPublicKeyValPair[1].key = LwSciBufRawBufferAttrKey_Align;

    err = LwSciBufAttrListCommonGetAttrs(attrList, 0, &rawBufPublicKeyValPair,
            RAW_PUBLIC_KEYVAL_PAIR_COUNT, LwSciBufAttrKeyType_Public, true);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("Failed to get key \n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    rawBufferSize = *(const uint64_t*)rawBufPublicKeyValPair[0].value;

    if (rawBufPublicKeyValPair[1].len != 0U) {
        rawBufferAlign = *(const uint64_t*)rawBufPublicKeyValPair[1].value;
    } else {
        /* Default value */
        rawBufferAlign = 1U;
    }

    rawBufferAlign = LW_SCI_BUF_MAX_NUM(constraints->startAddrAlign,
                        rawBufferAlign);

    rawBufPublicKeyValPair[0].key = LwSciBufRawBufferAttrKey_Align;
    rawBufPublicKeyValPair[0].len = sizeof(rawBufferAlign);
    rawBufPublicKeyValPair[0].value = &rawBufferAlign;

    err = LwSciBufAttrListCommonSetAttrs(attrList, 0,
            &rawBufPublicKeyValPair, 1, LwSciBufAttrKeyType_Public, true,
            false);
    if (err != LwSciError_Success) {
        LWSCI_ERR_STR("Setting LwSciBufRawBufferAttrKey_Align failed.");
        goto ret;
    }

    (void)memset(rawBufPrivateKeyValPair, 0x0, sizeof(rawBufPrivateKeyValPair));

    /* set size in private key so that alloc can read it */
    rawBufPrivateKeyValPair[0].key = LwSciBufPrivateAttrKey_Size;
    rawBufPrivateKeyValPair[0].value = &rawBufferSize;
    rawBufPrivateKeyValPair[0].len = sizeof(uint64_t);

    /* set alignment in private key so that alloc can read it */
    rawBufPrivateKeyValPair[1].key = LwSciBufPrivateAttrKey_Alignment;
    rawBufPrivateKeyValPair[1].value = &rawBufferAlign;
    rawBufPrivateKeyValPair[1].len = sizeof(uint64_t);

    err = LwSciBufAttrListCommonSetAttrs(attrList, 0, &rawBufPrivateKeyValPair,
            RAW_PRIVATE_KEYVAL_PAIR_COUNT, LwSciBufAttrKeyType_Private, true,
            false);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("Failed to set key \n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

ret:
    LWSCI_FNEXIT("");
    return err;
}

static void LwSciBufGetDataTypeSize(
    LwSciBufAttrValDataType dataType,
    uint32_t* size)
{
    uint32_t index = (uint32_t)dataType;
    const uint32_t dataSizeMap[LwSciDataType_UpperBound] = {
        [LwSciDataType_Int4] =      4U,
        [LwSciDataType_Uint4] =     4U,
        [LwSciDataType_Int8] =      8U,
        [LwSciDataType_Uint8] =     8U,
        [LwSciDataType_Int16] =     16U,
        [LwSciDataType_Uint16] =    16U,
        [LwSciDataType_Int32] =     32U,
        [LwSciDataType_Uint32] =    32U,
        [LwSciDataType_Float16] =   16U,
        [LwSciDataType_Float32] =   32U,
        [LwSciDataType_FloatISP] =  32U,
        [LwSciDataType_Bool] =      1U,
    };

    LWSCI_FNENTRY("");

    LWSCI_INFO("Input: dataType: %" PRIu32 ", size: %p\n", index, size);

    if ((uint32_t)LwSciDataType_UpperBound <= index) {
        LWSCI_ERR_STR("Input dataType is invalid.\n");
        LwSciCommonPanic();
    }

    *size = dataSizeMap[index];

    LWSCI_INFO("Output: size: %" PRIu32 "\n", *size);

    LWSCI_FNEXIT("");
}

static LwSciError LwSciBufArrayConstraint(
    LwSciBufAttrList attrList,
    const LwSciBufHwConstraints* constraints)
{
    LwSciError err = LwSciError_Success;
    LwSciBufAttrKeyValuePair
                    ipArrPublicKeyValPair[INPUT_ARR_PUBLIC_KEYVAL_PAIR_COUNT];
    LwSciBufAttrKeyValuePair
                    opArrPublicKeyValPair[OUTPUT_ARR_PUBLIC_KEYVAL_PAIR_COUNT];
    LwSciBufPrivateAttrKeyValuePair
                    arrPrivateKeyValPair[ARR_PRIVATE_KEYVAL_PAIR_COUNT];
    uint64_t size = 0U;
    uint32_t dataTypeSize = 0U;
    uint64_t startAddrAlign = 0U;
    const uint64_t* strideVal = NULL;
    const uint64_t* capacity = NULL;
    uint64_t tmpMul = 0U;
    uint8_t mulStatus = OP_FAIL;
    const LwSciBufAttrValDataType* valDataType = NULL;

    LWSCI_FNENTRY("");

    if ((NULL == attrList) || (NULL == constraints)) {
        err = LwSciError_BadParameter;
        LWSCI_ERR_STR("Bad parametere supplied to LwSciBufArrayConstraint\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    LWSCI_INFO("Input: attrList ptr %p, constraints %p\n", attrList,
        constraints);

    (void)memset(ipArrPublicKeyValPair, 0x0, sizeof(ipArrPublicKeyValPair));

    ipArrPublicKeyValPair[0].key = LwSciBufArrayAttrKey_DataType;
    ipArrPublicKeyValPair[1].key = LwSciBufArrayAttrKey_Stride;
    ipArrPublicKeyValPair[2].key = LwSciBufArrayAttrKey_Capacity;

    err = LwSciBufAttrListCommonGetAttrs(attrList, 0, &ipArrPublicKeyValPair,
            INPUT_ARR_PUBLIC_KEYVAL_PAIR_COUNT, LwSciBufAttrKeyType_Public,
            true);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("Failed to get keys\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
    valDataType =
        (const LwSciBufAttrValDataType*)ipArrPublicKeyValPair[0].value;

    strideVal = (const uint64_t*)ipArrPublicKeyValPair[1].value;
    LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 11_5))

    LwSciBufGetDataTypeSize(*valDataType, &dataTypeSize);

    if (*strideVal < dataTypeSize) {
        LWSCI_ERR_STR("stride cannot be less than size of datatype\n");
        err = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
    capacity = (const uint64_t*)ipArrPublicKeyValPair[2].value;
    startAddrAlign = constraints->startAddrAlign;

    u64Mul(*capacity, *strideVal, &tmpMul, &mulStatus);

    if (OP_SUCCESS != mulStatus) {
       LWSCI_ERR_STR("Buffer overflow");
       err = LwSciError_Overflow;
       LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
       goto ret;
    }
    err = LwSciBufAliglwalue64(sizeof(uint64_t), constraints->dataAlign, &size);
    if (LwSciError_Success != err) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }
    size = size + tmpMul;

    (void)memset(arrPrivateKeyValPair, 0x0, sizeof(arrPrivateKeyValPair));

    arrPrivateKeyValPair[0].key = LwSciBufPrivateAttrKey_Size;
    arrPrivateKeyValPair[0].value = &size;
    arrPrivateKeyValPair[0].len = sizeof(uint64_t);

    arrPrivateKeyValPair[1].key = LwSciBufPrivateAttrKey_Alignment;
    arrPrivateKeyValPair[1].value = &startAddrAlign;
    arrPrivateKeyValPair[1].len = sizeof(uint64_t);

    err = LwSciBufAttrListCommonSetAttrs(attrList, 0, &arrPrivateKeyValPair,
            ARR_PRIVATE_KEYVAL_PAIR_COUNT, LwSciBufAttrKeyType_Private, true,
            false);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("Failed to set array keys\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    (void)memset(opArrPublicKeyValPair, 0x0, sizeof(opArrPublicKeyValPair));

    opArrPublicKeyValPair[0].key = LwSciBufArrayAttrKey_Size;
    opArrPublicKeyValPair[0].value = &size;
    opArrPublicKeyValPair[0].len = sizeof(uint64_t);

    opArrPublicKeyValPair[1].key = LwSciBufArrayAttrKey_Alignment;
    opArrPublicKeyValPair[1].value = &startAddrAlign;
    opArrPublicKeyValPair[1].len = sizeof(uint64_t);

    err = LwSciBufAttrListCommonSetAttrs(attrList, 0, &opArrPublicKeyValPair,
            OUTPUT_ARR_PUBLIC_KEYVAL_PAIR_COUNT, LwSciBufAttrKeyType_Public,
            true, false);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("Failed to set public key");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

ret:
    LWSCI_FNEXIT("");
    return err;
}

static LwSciError LwSciBufTensorGetAttrDataType(
    LwSciBufAttrList attrList,
    uint32_t *bitsPerElement)
{
    LwSciError err = LwSciError_Success;
    LwSciBufAttrKeyValuePair tensorKeyValPair;
    const LwSciBufAttrValDataType* dataType = NULL;

    LWSCI_FNENTRY("");

    /* Get tensor datatype */
    tensorKeyValPair.key = LwSciBufTensorAttrKey_DataType;
    err = LwSciBufAttrListCommonGetAttrs(attrList, 0, &tensorKeyValPair, 1,
            LwSciBufAttrKeyType_Public, true);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("Failed to get tensor key\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
    dataType = (const LwSciBufAttrValDataType*)tensorKeyValPair.value;

    LwSciBufGetDataTypeSize(*dataType, bitsPerElement);

ret:
    LWSCI_FNEXIT("");
    return err;
}

static LwSciError LwSciBufTensorGetAttrNumDims(
    LwSciBufAttrList attrList,
    uint32_t *numDims)
{
    LwSciError err = LwSciError_Success;
    LwSciBufAttrKeyValuePair tensorKeyValPair;

    LWSCI_FNENTRY("");

    (void)memset(&tensorKeyValPair, 0x0, sizeof(tensorKeyValPair));

    /* Get Number of Dimensions */
    tensorKeyValPair.key = LwSciBufTensorAttrKey_NumDims;
    err = LwSciBufAttrListCommonGetAttrs(attrList, 0, &tensorKeyValPair, 1,
            LwSciBufAttrKeyType_Public, true);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("Failed to get tensor key\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
    *numDims = *(const uint32_t*)tensorKeyValPair.value;
    if ((0U == *numDims) || ((uint32_t)LW_SCI_BUF_TENSOR_MAX_DIMS < *numDims)) {
        LWSCI_ERR_STR("Invalid number of dimensions provided via LwSciBufTensorAttrKey_NumDims key\n");
        LWSCI_ERR_UINT("Number of dimensions: \n", *numDims);
        LwSciCommonPanic();
    }

ret:
    LWSCI_FNEXIT("");
    return err;
}

static LwSciError LwSciBufTensorGetAttrSizePerDim(
    LwSciBufAttrList attrList,
    const uint64_t** sizePerDims)
{
    LwSciError err = LwSciError_Success;
    LwSciBufAttrKeyValuePair tensorKeyValPair;

    LWSCI_FNENTRY("");

    (void)memset(&tensorKeyValPair, 0x0, sizeof(tensorKeyValPair));

    /* Get Size per dimension */
    tensorKeyValPair.key = LwSciBufTensorAttrKey_SizePerDim;
    err = LwSciBufAttrListCommonGetAttrs(attrList, 0, &tensorKeyValPair, 1,
            LwSciBufAttrKeyType_Public, true);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("Failed to get tensor key\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
    *sizePerDims = (const uint64_t*)tensorKeyValPair.value;

ret:
    LWSCI_FNEXIT("");
    return err;
}

static LwSciError LwSciBufTensorGetAttrStridesPerDim(
    LwSciBufAttrList attrList,
    uint64_t* stridePerDims)
{
    LwSciError err = LwSciError_Success;
    LwSciBufAttrKeyValuePair tensorKeyValPair;

    LWSCI_FNENTRY("");

    (void)memset(&tensorKeyValPair, 0x0, sizeof(tensorKeyValPair));

    /* Get Strides per dimension */
    tensorKeyValPair.key = LwSciBufTensorAttrKey_StridesPerDim;
    err = LwSciBufAttrListCommonGetAttrs(attrList, 0, &tensorKeyValPair, 1,
            LwSciBufAttrKeyType_Public, true);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("Failed to get tensor key\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
    *stridePerDims = *(const uint64_t*)tensorKeyValPair.value;

ret:
    LWSCI_FNEXIT("");
    return err;
}

static LwSciError LwSciBufTensorComputeAlignmentPerDim(
    LwSciBufAttrList attrList,
    const LwSciBufHwConstraints* constraints,
    const uint32_t** alignPerDimsPtr,
    uint32_t* alignPerDims)
{
    LwSciError err = LwSciError_Success;
    LwSciBufAttrKeyValuePair tensorKeyValPair;
    uint64_t len = 0U;
    bool isImageType = false;
    size_t numBufTypes = 0U;
    const LwSciBufType* bufTypePtr = NULL;
    uint32_t tmpMul = 0U;
    uint8_t mulStatus = OP_FAIL;

    LWSCI_FNENTRY("");

    (void)memset(&tensorKeyValPair, 0x0, sizeof(tensorKeyValPair));

    /* Get alignment per dimension of tensor */
    tensorKeyValPair.key = LwSciBufTensorAttrKey_AlignmentPerDim;
    err = LwSciBufAttrListCommonGetAttrs(attrList, 0, &tensorKeyValPair, 1,
            LwSciBufAttrKeyType_Public, true);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("Failed to get raw buffer size key\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }
    len = tensorKeyValPair.len;

    err = LwSciBufAttrListGetDataTypes(attrList, &bufTypePtr, &numBufTypes);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("LwSciBufAttrListGetDataTypes failed\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    LwSciBufConstraintMatchBufType(bufTypePtr, numBufTypes,
            LwSciBufType_Image, &isImageType);

    // alignPerDims[0] = LW_SCI_BUF_MAX_NUM(constraints->pitchAlign*constraints->heightAlign,
    //                                      constraints->sizeAlign)
    // The first parameter me wrap, we need to add a check.
    u32Mul(constraints->pitchAlign, constraints->heightAlign, &tmpMul, &mulStatus);

    if (OP_SUCCESS != mulStatus) {
        LWSCI_ERR_STR("Buffer overflow\n");
        err = LwSciError_Overflow;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }
    if ((0U == len) && (true == isImageType)) {
            /* This is image/tensor interop and user has not set
             * LwSciBufTensorAttrKey_AlignmentPerDim key. Use HW constraints
             * for alignment. (Assume NHWC tensor)
             */
            LWSCI_INFO("LwSciBufTensorAttrKey_AlignmentPerDim key not set. Using hardware constraints\n");
            /* alignment for 'C' dimension */
            alignPerDims[3] = 1;
            /* alignment for 'W' dimension */
            alignPerDims[2] = 1;
            /* pitch alignment for 'H' dimension */
            alignPerDims[1] = (uint32_t)constraints->pitchAlign;
            /* plane alignment for 'N' dimension */
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Directive, 4_9), "LwSciBuf-ADV-MISRAC2012-008")
            alignPerDims[0] = LW_SCI_BUF_MAX_NUM(tmpMul, constraints->sizeAlign);
            /* TODO: alignPerDims[0] = 128*1024 is a WAR since image datatype
             * uses same hard-coded value to align image size. size alignment
             * for image should be moved to constraints library and hard-coding
             * should be removed.
             * alignPerDims[0] = MAX(constraints->sizeAlign,
                            constraints->pitchAlign * constraints->heightAlign)
             * once hard-coding is removed
             */
             LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Directive, 4_9), "LwSciBuf-ADV-MISRAC2012-008")
            alignPerDims[0] = LW_SCI_BUF_MAX_NUM((128U * 1024U),
                                alignPerDims[0]);

            *alignPerDimsPtr = alignPerDims;
    } else if (true == isImageType) {
        LWSCI_INFO("Reconciling HW constraints with alignment constraints from LwSciBufTensorAttrKey_AlignmentPerDim key\n");
        /* This is image/tensor interop and user has also set
         * LwSciBufTensorAttrKey_AlignmentPerDim key. Reconcile alignments from
         * key and HW alignment constraints. (Assume NHWC tensor)
         */
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
        *alignPerDimsPtr = (const uint32_t*)tensorKeyValPair.value;

        LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Directive, 4_9), "LwSciBuf-ADV-MISRAC2012-008")
        /* reconcile alignment for 'C' dimension */
        alignPerDims[3] = LW_SCI_BUF_MAX_NUM((*alignPerDimsPtr)[3], 1U);

        /* reconcile alignment for 'W' dimension */
        alignPerDims[2] = LW_SCI_BUF_MAX_NUM((*alignPerDimsPtr)[2], 1U);

        /* reconcile alignment for 'H' dimension */
        alignPerDims[1] = LW_SCI_BUF_MAX_NUM((*alignPerDimsPtr)[1],
                            constraints->pitchAlign);

        /* reconcile alignment for 'N' dimension */
        alignPerDims[0] = LW_SCI_BUF_MAX_NUM((*alignPerDimsPtr)[0],
                                LW_SCI_BUF_MAX_NUM(tmpMul, constraints->sizeAlign));
        /* TODO: alignPerDims[0] = 128*1024 is a WAR since image datatype
         * uses same hard-coded value to align image size. size alignment
         * for image should be moved to constraints library and hard-coding
         * should be removed.
         * alignPerDims[0] = MAX(alignPerDimsPtr[0], constraints->sizeAlign,
                            constraints->pitchAlign * constraints->heightAlign)
         * once hard-coding is removed
         */
        alignPerDims[0] = LW_SCI_BUF_MAX_NUM((128U * 1024U),
                            alignPerDims[0]);
        LWCOV_ALLOWLIST_END(LWCOV_MISRA(Directive, 4_9))

        *alignPerDimsPtr = alignPerDims;
    } else if (0U != len) {
        LWSCI_INFO("Using alignment constraints from LwSciBufTensorAttrKey_AlignmentPerDim key\n");
        /* this is NOT image/tensor interop. Use alignment constraints specified
         * via LwSciBufTensorAttrKey_AlignmentPerDim key
         */
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
        *alignPerDimsPtr = (const uint32_t*)tensorKeyValPair.value;
    } else {
        LWSCI_ERR_STR("LwSciBufTensorAttrKey_AlignmentPerDim key not set\n");
        err = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

ret:
    LWSCI_FNEXIT("");
    return err;
}

static LwSciError LwSciBufTensorSetAttrSizeStride(
    LwSciBufAttrList attrList,
    const uint64_t* size,
    const uint64_t* stridePerDims,
    uint32_t numDims)
{
    LwSciError err = LwSciError_Success;
    LwSciBufAttrKeyValuePair tensorPublicKeyValPair[STRIDE_KEYVAL_PAIR_COUNT];
    LwSciBufPrivateAttrKeyValuePair tensorPrivateKeyValPair;

    LWSCI_FNENTRY("");

    /* set size in private key so that alloc can read it */
    tensorPrivateKeyValPair.key = LwSciBufPrivateAttrKey_Size;
    tensorPrivateKeyValPair.len = sizeof(*size);
    tensorPrivateKeyValPair.value = size;
    err = LwSciBufAttrListCommonSetAttrs(attrList, 0, &tensorPrivateKeyValPair,
            1, LwSciBufAttrKeyType_Private, true, false);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("Failed to set LwSciBufPrivateAttrKey_Sizelen key\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    (void)memset(tensorPublicKeyValPair, 0x0, sizeof(tensorPublicKeyValPair));

    /* set size in output key */
    tensorPublicKeyValPair[0].key = LwSciBufTensorAttrKey_Size;
    tensorPublicKeyValPair[0].len = sizeof(*size);
    tensorPublicKeyValPair[0].value = size;

    /* set stride per dimension */
    tensorPublicKeyValPair[1].key = LwSciBufTensorAttrKey_StridesPerDim;
    tensorPublicKeyValPair[1].len = sizeof(*stridePerDims)*(numDims);
    tensorPublicKeyValPair[1].value = stridePerDims;
    err = LwSciBufAttrListCommonSetAttrs(attrList, 0, &tensorPublicKeyValPair,
            STRIDE_KEYVAL_PAIR_COUNT, LwSciBufAttrKeyType_Public, true, false);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("Failed to set LwSciBufTensorAttrKey_StridesPerDim key\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

ret:
    LWSCI_FNEXIT("");
    return err;
}

static LwSciError LwSciBufTensorSetAttrAlignment(
    LwSciBufAttrList attrList,
    const LwSciBufHwConstraints* constraints)
{
    LwSciError err = LwSciError_Success;
    LwSciBufAttrKeyValuePair tensorKeyValPair;
    LwSciBufPrivateAttrKeyValuePair pvtKeyValPair;
    uint64_t baseAddrAlignment = 0U;

    LWSCI_FNENTRY("");

    (void)memset(&tensorKeyValPair, 0x0, sizeof(tensorKeyValPair));
    (void)memset(&pvtKeyValPair, 0x0, sizeof(pvtKeyValPair));

    /* set base addr alignment */
    tensorKeyValPair.key = LwSciBufTensorAttrKey_BaseAddrAlign;
    err = LwSciBufAttrListCommonGetAttrs(attrList, 0, &tensorKeyValPair, 1,
            LwSciBufAttrKeyType_Public, true);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("Failed to get LwSciBufTensorAttrKey_BaseAddrAlign key\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    if (tensorKeyValPair.len != 0U) {
        baseAddrAlignment = *(const uint64_t*)tensorKeyValPair.value;
    } else {
        /* Default value */
        baseAddrAlignment = 1U;
    }

    if (((uint64_t)constraints->startAddrAlign) > baseAddrAlignment) {
        /* use base addr alignment coming from HW constraints if it is greater
         * than value provided by user via LwSciBufTensorAttrKey_BaseAddrAlign
         * key
         */
        baseAddrAlignment = constraints->startAddrAlign;
    }

    tensorKeyValPair.key = LwSciBufTensorAttrKey_BaseAddrAlign;
    tensorKeyValPair.value = &baseAddrAlignment;
    tensorKeyValPair.len = sizeof(baseAddrAlignment);
    err = LwSciBufAttrListCommonSetAttrs(attrList, 0, &tensorKeyValPair, 1,
            LwSciBufAttrKeyType_Public, true, false);
    if (err != LwSciError_Success) {
        LWSCI_ERR_STR("Failed to set LwSciBufTensorAttrKey_BaseAddrAlign key.");
        goto ret;
    }

    pvtKeyValPair.key = LwSciBufPrivateAttrKey_Alignment;
    pvtKeyValPair.len = sizeof(uint64_t);
    pvtKeyValPair.value = &baseAddrAlignment;
    err = LwSciBufAttrListCommonSetAttrs(attrList, 0, &pvtKeyValPair, 1,
            LwSciBufAttrKeyType_Private, true, false);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("Failed to set LwSciBufPrivateAttrKey_Alignment key\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

ret:
    LWSCI_FNEXIT("");
    return err;
}

static LwSciError LwSciBufTensorConstraint(
    LwSciBufAttrList attrList,
    const LwSciBufHwConstraints* constraints)
{
    LwSciError err = LwSciError_Success;
    uint32_t bitsPerElement = 0U;
    uint32_t alignPerDims[LW_SCI_BUF_TENSOR_MAX_DIMS] = {0};
    const uint32_t* alignPerDimsPtr = NULL;
    const uint64_t* sizePerDims = NULL;
    uint64_t stridePerDims[LW_SCI_BUF_TENSOR_MAX_DIMS] = {0};
    uint64_t size = 0U;
    uint32_t numDims = 0U;
    int64_t dim = 0;
    uint8_t mulStatus = OP_FAIL;
    uint32_t numDims1 = 0U;

    LWSCI_FNENTRY("");

    LWSCI_INFO("Input: attrList ptr %p, constraints ptr %p\n", attrList,
        constraints);

    err =  LwSciBufTensorGetAttrDataType(attrList, &bitsPerElement);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("Failed to get tensor key DataType\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    err = LwSciBufTensorGetAttrNumDims(attrList, &numDims);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("Failed to get tensor key NumDims\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    err = LwSciBufTensorGetAttrSizePerDim(attrList, &sizePerDims);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("Failed to get tensor key SizePerDim\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    err = LwSciBufTensorGetAttrStridesPerDim(attrList, &stridePerDims[0]);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("Failed to get tensor key StridesPerDim\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    err = LwSciBufTensorComputeAlignmentPerDim(attrList,
            constraints, &alignPerDimsPtr, alignPerDims);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("Failed to get tensor key AlignmentPerDim\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    if (1U > numDims) {
        LWSCI_ERR_STR("Failed to get tensor key NumDims\n");
        err = LwSciError_Unknown;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }
    numDims1 = numDims - 1U;

    stridePerDims[numDims1] = ((uint64_t)bitsPerElement) >> 3U;

    err = LwSciBufAliglwalue64(stridePerDims[numDims1],
        alignPerDimsPtr[numDims1], &stridePerDims[numDims1]);
    if (LwSciError_Success != err) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    LWSCI_INFO("stride %u %lu\n", numDims1, stridePerDims[numDims1]);

    for (dim = (int64_t)numDims-2; dim >= 0; dim--) {
        stridePerDims[dim] = sizePerDims[dim+1] * stridePerDims[dim+1];

        err = LwSciBufAliglwalue64(stridePerDims[dim], alignPerDimsPtr[dim],
            &stridePerDims[dim]);
        if (LwSciError_Success != err) {
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
            goto ret;
        }
        LWSCI_INFO("stride %ld %lu\n", dim, stridePerDims[dim]);
    }

    //size = sizePerDims[0] * stridePerDims[0]
    u64Mul(sizePerDims[0], stridePerDims[0], &size, &mulStatus);
    if (OP_SUCCESS != mulStatus) {
        LWSCI_ERR_STR("Buffer overflow\n");
        err = LwSciError_Overflow;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }
    LWSCI_INFO("size %lu\n", size);

    err = LwSciBufTensorSetAttrSizeStride(attrList, &size, stridePerDims,
            numDims);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("Failed to set keys size and StridePerDims\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    err = LwSciBufTensorSetAttrAlignment(attrList, constraints);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("Failed to modify alignment key\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }


ret:
    LWSCI_FNEXIT("");
    return err;
}

LwSciError LwSciBufApplyConstraints(
    LwSciBufAttrList reconcileList)
{
    LwSciError err = LwSciError_Success;
    LwSciBufPrivateAttrKeyValuePair pvtKeyValPair;
    size_t numBufTypes = 0U, index = 0U, engineCount = 0U;
    LwSciBufType tmpBufType = LwSciBufType_MaxValid;
    LwSciBufType bufType = LwSciBufType_MaxValid;
    const LwSciBufType* bufTypePtr = NULL;
    LwSciBufApplyTypeConstraint typeConstraint = NULL;
    LwSciBufHwConstraints constraints = {0};
    LwSciBufAttrKeyValuePair keyValPair = {0};
    LwSciBufInternalAttrKeyValuePair internalKeyValPair = {0};
    const LwSciBufHwEngine* engineArray = NULL;
    LwSciBufAttrValImageLayoutType imageLayout = LwSciBufImage_PitchLinearType;
    uint64_t tmpBufSize = 0U, bufSize = 0;

    static const LwSciBufApplyTypeConstraint
        typeConstraintMap[LwSciBufType_MaxValid] = {
        [LwSciBufType_General] = NULL,
        [LwSciBufType_RawBuffer] = LwSciBufRawConstraint,
        [LwSciBufType_Image] = LwSciBufImageConstraint,
        [LwSciBufType_Tensor] = LwSciBufTensorConstraint,
        [LwSciBufType_Array] = LwSciBufArrayConstraint,
        [LwSciBufType_Pyramid] = LwSciBufPyramidConstraint,
    };


    LWSCI_FNENTRY("");

    /* Validate attribute list */
    err = LwSciBufAttrListValidate(reconcileList);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("Failed to validate reconcileList.\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    LWSCI_INFO("Input: reconcileList %p\n", reconcileList);

    /* get buffer type */
    err = LwSciBufAttrListGetDataTypes(reconcileList, &bufTypePtr,
            &numBufTypes);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("LwSciBufAttrListGetDataTypes failed\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    /* get HW engines operating on the buffer */
    internalKeyValPair.key = LwSciBufInternalGeneralAttrKey_EngineArray;
    err = LwSciBufAttrListCommonGetAttrs(reconcileList, 0, &internalKeyValPair,
            1, LwSciBufAttrKeyType_Internal, true);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("LwSciBufAttrListGetInternalAttrs failed\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
    engineArray = (const LwSciBufHwEngine*)internalKeyValPair.value;
    engineCount = internalKeyValPair.len / sizeof(LwSciBufHwEngine);

    /* get consolidated HW constraints for engines corresponding to all the
     * specified datatypes
     */
    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_4), "LwSciBuf-ADV-MISRAC2012-016")
    for (index = 0; index < numBufTypes; index++) {
        if ((LwSciBufType_Image == bufTypePtr[index])
                || (LwSciBufType_Pyramid == bufTypePtr[index])
                ) {
            /* we need to pass image layout to LwSciBufGetConstraints function
             * in order to get either pitchlinear of blocklinear constraints
             */
            keyValPair.key = LwSciBufImageAttrKey_Layout;
            err = LwSciBufAttrListCommonGetAttrs(reconcileList, 0, &keyValPair, 1,
                    LwSciBufAttrKeyType_Public, true);
            if (LwSciError_Success != err) {
                LWSCI_ERR_STR("Could not get value of key LwSciBufImageAttrKey_Layout from attrlist\n");
                LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
                goto ret;
            }

            if (0U != keyValPair.len) {
                LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
                imageLayout =
                    *(const LwSciBufAttrValImageLayoutType*)keyValPair.value;
                LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 11_5))
            } else {
                err = LwSciError_BadParameter;
                LWSCI_ERR_STR("LwSciBufImageAttrKey_Layout key not set for image datatype\n");
                LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
                goto ret;
            }
        }

        err = LwSciBufGetConstraints(bufTypePtr[index], LWRM_T194_ID,
                engineArray, (uint32_t)engineCount,
                &constraints, (void *)&imageLayout);
        if (LwSciError_Success != err) {
            LWSCI_ERR_STR("LwSciBufGetConstraints failed\n");
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
            goto ret;
        }
    }

    /* apply buffer constraints one by one for all buffer types and verify that
     * size callwlated for every buffer type matches
     */
    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_4), "LwSciBuf-ADV-MISRAC2012-016")
    for (index = 0; index < numBufTypes; index++) {
        bufType = bufTypePtr[index];

        if ((bufType < LwSciBufType_General) || (bufType >= LwSciBufType_MaxValid)) {
            // This is impossible error
            LwSciCommonPanic();
        }
        typeConstraint = typeConstraintMap[bufType];
        if (NULL == typeConstraint) {
            LWSCI_ERR_UINT("No contraint mapping for buffer type \n", (uint32_t)bufType);
            LwSciCommonPanic();
        }

        err = typeConstraint(reconcileList, &constraints);
        if (LwSciError_Success != err) {
            LWSCI_ERR_UINT("Failed to apply constraints for buffer type \n",
            (uint32_t)bufType);
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
            goto ret;
        }

        pvtKeyValPair.key = LwSciBufPrivateAttrKey_Size;
        err = LwSciBufAttrListCommonGetAttrs(reconcileList, 0, &pvtKeyValPair, 1,
                LwSciBufAttrKeyType_Private, true);
        if (LwSciError_Success != err) {
            LWSCI_ERR_STR("Failed to get LwSciBufPrivateAttrKey_Size value.\n");
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
            goto ret;
        }
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
        bufSize = *(const uint64_t*)pvtKeyValPair.value;

        if (0U != index) {
            if (tmpBufSize != bufSize) {
                err = LwSciError_IlwalidState;
                LWSCI_ERR_UINT("Callwlated size for bufType  ", (uint32_t)tmpBufType);
                LWSCI_ERR_UINT("and \n", (uint32_t)bufType);
                LWSCI_ERR_UINT("mismatch Size for bufType  ", (uint32_t)tmpBufType);
                LWSCI_ERR_ULONG("is: \n", tmpBufSize);
                LWSCI_ERR_UINT("Size for bufType  ", (uint32_t)bufType);
                LWSCI_ERR_ULONG("is: \n", bufSize);
                LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
                goto ret;
            }
        }

        /* assign current values of buftype and bufsize to temp variables to be
         * used in next iteration
         */
        tmpBufType = bufType;
        tmpBufSize = bufSize;
    }

ret:
    LWSCI_FNEXIT("");
    return err;
}
