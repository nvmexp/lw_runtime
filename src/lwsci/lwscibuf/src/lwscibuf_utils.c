/*
 * Copyright (c) 2019-2022, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 */

#include "lwscibuf.h"
#include "lwscilog.h"
#include "lwscibuf_utils_priv.h"

LwSciBufAttrValDataType LwColorDataTypeToLwSciBufDataType(
    LwColorDataType colorDataType,
    uint8_t channelCount,
    uint32_t colorBPP)
{
    LwSciBufAttrValDataType varDataType = LwSciDataType_Float32;
    uint32_t bitsPerChannel = 0U;
    int32_t powOf2BPC = -1;
    uint32_t powOf2BPCIndex = 0U;

    /* 2nd dimension is power of 2, since we can have 2^5=32 max type.
     * max limit of 2nd dimension is 6 [0-5]
     * The type of datatype is callwlated using log2(colorBPP/channelCount)
     */
    const LwSciBufAttrValDataType dataTypeMap[4][6] = {
        [LwColorDataType_Integer] = {
            LwSciDataType_UpperBound,
            LwSciDataType_UpperBound,
            LwSciDataType_Uint4,
            LwSciDataType_Uint8,
            LwSciDataType_Uint16,
            LwSciDataType_Uint32,
            },
        [LwColorDataType_Float] = {
            LwSciDataType_UpperBound,
            LwSciDataType_UpperBound,
            LwSciDataType_UpperBound,
            LwSciDataType_UpperBound,
            LwSciDataType_Float16,
            LwSciDataType_Float32,
            },
        [LwColorDataType_Signed] = {
            LwSciDataType_UpperBound,
            LwSciDataType_UpperBound,
            LwSciDataType_Int4,
            LwSciDataType_Int8,
            LwSciDataType_Int16,
            LwSciDataType_Int32
            },
        [LwColorDataType_FloatISP] = {
            LwSciDataType_UpperBound,
            LwSciDataType_UpperBound,
            LwSciDataType_UpperBound,
            LwSciDataType_UpperBound,
            LwSciDataType_FloatISP,
            LwSciDataType_UpperBound
            },
    };

    LWSCI_FNENTRY("");

    LWSCI_INFO("Input: LwColorDataType %" PRIu32 " channelCount %" PRIu8
                " colorBPP %" PRIu32 "\n", colorDataType, channelCount,
                colorBPP);

    bitsPerChannel = colorBPP/(uint32_t)channelCount;

    while (0U < bitsPerChannel) {
        bitsPerChannel >>= 1U;
        powOf2BPC++;
        if (5 < powOf2BPC) {
            break;
        }
    }

    if ((LwColorDataType_FloatISP < colorDataType) ||
        (5 < powOf2BPC) || (0 > powOf2BPC)) {
        varDataType = LwSciDataType_UpperBound;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    } else {
        powOf2BPCIndex = (uint32_t)powOf2BPC;
    }
    varDataType = dataTypeMap[(uint32_t)colorDataType][powOf2BPCIndex];

ret:
    LWSCI_INFO("Output: LwSciBufAttrValDataType %" PRIu32 "\n", varDataType);
    LWSCI_FNEXIT("");
    return varDataType;
}

LwSciError LwSciBufIsMaxValue(
    const void* src1,
    const void* src2,
    size_t len,
    bool* isBigger)
{
    size_t i = 0;
    LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
    const uint8_t* cmp1 = src1;
    const uint8_t* cmp2 = src2;
    LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 11_5))
    LwSciError err = LwSciError_Success;

    LWSCI_FNENTRY("");

    if ((NULL == src1) || (NULL == src2) || (NULL == isBigger) || (0U == len)) {
        LWSCI_ERR_STR("Bad arguments to compare.");
        err = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    LWSCI_INFO("Inputs - S1 %p S2 %p isBigger %p\n", src1, src2, isBigger);

    LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 18_4), "LwSciBuf-ADV-MISRAC2012-017")
    cmp1 += len - 1U;
    cmp2 += len - 1U;
    LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 18_4))

    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_4), "LwSciBuf-ADV-MISRAC2012-016")
    for (i = 0; i < len; i++) {
        if (*cmp1 > *cmp2) {
            *isBigger = true;
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
            goto ret;
        }
        if (*cmp1 < *cmp2) {
            *isBigger = false;
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
            goto ret;
        }
        cmp1--;
        cmp2--;
    }
    /* Return true if both are same */
    *isBigger = true;

ret:
    LWSCI_INFO("Output: Is Bigger: %s\n" ,(*isBigger == true)?"True":"False");
    LWSCI_FNEXIT("");
    return (err);
}

LwSciError LwSciBufAliglwalue64(
    uint64_t value,
    uint64_t alignment,
    uint64_t* alignedValue)
{
    LwSciError err = LwSciError_Success;
    uint64_t tmpAdd = 0U;
    uint64_t alignmentMask = 0U;
    uint8_t addStatus = OP_FAIL;
    uint8_t subStatus = OP_FAIL;

    LWSCI_FNENTRY("");

    if (NULL == alignedValue) {
        LwSciCommonPanic();
    }

    u64Sub(alignment, 1U, &alignmentMask, &subStatus);
    u64Add(value, alignmentMask, &tmpAdd, &addStatus);
    if (OP_SUCCESS != (subStatus & addStatus)) {
        LWSCI_ERR_STR("Arithmetic overflow\n");
        err = LwSciError_Overflow;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    *alignedValue = tmpAdd & ~alignmentMask;

ret:
    LWSCI_FNEXIT("");
    return err;
}

LwSciError LwSciBufAliglwalue32(
    uint32_t value,
    uint32_t alignment,
    uint32_t* alignedValue)
{
    LwSciError err = LwSciError_Success;
    uint32_t tmpAdd = 0U;
    uint32_t alignmentMask = 0U;
    uint8_t addStatus = OP_FAIL;
    uint8_t subStatus = OP_FAIL;

    LWSCI_FNENTRY("");

    if (NULL == alignedValue) {
        LwSciCommonPanic();
    }

    u32Sub(alignment, 1U, &alignmentMask, &subStatus);
    u32Add(value, alignmentMask, &tmpAdd, &addStatus);
    if (OP_SUCCESS != (subStatus & addStatus)) {
        LWSCI_ERR_STR("Arithmetic overflow\n");
        err = LwSciError_Overflow;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    *alignedValue = tmpAdd & ~alignmentMask;

ret:
    LWSCI_FNEXIT("");
    return err;
}

LwSciError LwSciBufIsSocBoundary(
    LwSciIpcEndpoint ipcEndpoint,
    bool* isSocBoundary)
{
    LwSciError err = LwSciError_Success;
    LwSciIpcTopoId tmpTopoId = {};

    LWSCI_FNENTRY("");
    if (0UL == ipcEndpoint) {
        LWSCI_ERR_STR("invalid ipcEndpoint supplied to LwSciBufIsSocBoundary().");
        err = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
            "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    if (NULL == isSocBoundary) {
        LWSCI_ERR_STR("isSocBoundary is NULL.");
        LwSciCommonPanic();
    }
    *isSocBoundary = false;

#if (LW_IS_SAFETY == 0)
    err = LwSciIpcEndpointGetTopoId(ipcEndpoint, &tmpTopoId);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("LwSciIpcEndpointGetTopoId failed.");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
            "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }
#else
    tmpTopoId.SocId = LWSCIIPC_SELF_SOCID;
    tmpTopoId.VmId = LWSCIIPC_SELF_VMID;
#endif

    if (LWSCIIPC_SELF_SOCID != tmpTopoId.SocId) {
        *isSocBoundary = true;
        goto ret;
    }

ret:
    LWSCI_FNEXIT("");
    return err;
}
