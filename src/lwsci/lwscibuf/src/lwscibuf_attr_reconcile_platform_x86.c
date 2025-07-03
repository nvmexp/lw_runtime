/*
 * Copyright (c) 2020-2021, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 */

#include <stdbool.h>
#include "lwscibuf_internal.h"
#include "lwscibuf_attr_key_dep.h"
#include "lwscibuf_attr_constraint_x86.h"
#include "lwscibuf_attr_reconcile_platform.h"

static void GetOutputAttrCountForKeyType(
    const LwSciBufOutputAttrKeyValuePair* outputAttributes,
    LwSciBufAttrKeyType attrKeyType,
    size_t* totalAttrCount)
{
    if (LwSciBufAttrKeyType_Public == attrKeyType) {
        *totalAttrCount = outputAttributes->publicAttrsCount;
    } else if (LwSciBufAttrKeyType_Internal == attrKeyType) {
        *totalAttrCount = outputAttributes->internalAttrsCount;
    } else {
        LwSciCommonPanic();
    }
}

static LwSciError VerifyOutputAttrSanity(
    const LwSciBufOutputAttrKeyValuePair* outputAttributes,
    LwSciBufAttrKeyType attrKeyType)
{
    LwSciError err = LwSciError_Success;

    if (LwSciBufAttrKeyType_Public == attrKeyType) {
        if (NULL == outputAttributes->publicAttrs ||
            0UL == outputAttributes->publicAttrsCount) {
            err = LwSciError_BadParameter;
        }
    } else if (LwSciBufAttrKeyType_Internal == attrKeyType) {
        if (NULL == outputAttributes->internalAttrs ||
            0UL == outputAttributes->internalAttrsCount) {
            err = LwSciError_BadParameter;
        }
    } else {
        LwSciCommonPanic();
    }

    return err;
}

static void GetOutputAttrKeyIndexedData(
    const LwSciBufOutputAttrKeyValuePair* outputAttributes,
    LwSciBufAttrKeyType attrKeyType,
    size_t index,
    uint32_t* key,
    const void** value,
    size_t* len)
{
    if (attrKeyType == LwSciBufAttrKeyType_Public) {
        *key = (uint32_t)outputAttributes->publicAttrs[index].key;
        *value = outputAttributes->publicAttrs[index].value;
        *len = outputAttributes->publicAttrs[index].len;
    } else if (LwSciBufAttrKeyType_Internal == attrKeyType) {
        *key = (uint32_t)outputAttributes->internalAttrs[index].key;
        *value = outputAttributes->internalAttrs[index].value;
        *len = outputAttributes->internalAttrs[index].len;
    } else {
        LwSciCommonPanic();
    }
}

/* TODO
 * This is a work around for http://lwbugs/200668495 [LwSciBuf] Change Image GoB keys Qualifier to Conditional
 */
static LwSciError LwSciBufReconciledKeyCheckNeeded(
    LwSciBufAttrList reconciledList,
    uint32_t key,
    bool* needKeyValidation)
{
    LwSciError err = LwSciError_Success;

    LWSCI_FNENTRY("");

    *needKeyValidation = true;

    err = LwSciBufImportCheckingNeeded(reconciledList, key, needKeyValidation);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("Failed to get key details from reconciled attribute list");
        LwSciCommonPanic();
    }

    switch (key) {
        case ((uint32_t)LwSciBufInternalImageAttrKey_PlaneGobSize):
        case ((uint32_t)LwSciBufInternalImageAttrKey_PlaneLog2GobsPerBlockX):
        case ((uint32_t)LwSciBufInternalImageAttrKey_PlaneLog2GobsPerBlockY):
        case ((uint32_t)LwSciBufInternalImageAttrKey_PlaneLog2GobsPerBlockZ):
        {
            LwSciBufAttrKeyValuePair keyValPair;
            LwSciBufAttrValImageLayoutType layoutType;

            keyValPair.key = LwSciBufImageAttrKey_Layout;
            err = LwSciBufAttrListCommonGetAttrs(reconciledList, 0,
                            &keyValPair, 1, LwSciBufAttrKeyType_Public, true);
            if (LwSciError_Success != err) {
                LWSCI_ERR_STR("Could not get value of key LwSciBufImageAttrKey_Layout from attrlist");
                // LwSciBufImageAttrKey_Layout should always be set for
                // reconciled attribute list with LwSciBufType as
                // LwSciBufType_Image.
                LwSciCommonPanic();
            }

            layoutType =
                *(const LwSciBufAttrValImageLayoutType*)keyValPair.value;

            *needKeyValidation = ((keyValPair.len > 0UL) &&
                (LwSciBufImage_BlockLinearType == layoutType));
            break;
        }
        default:
        {
            break;
        }
    }

    LWSCI_FNEXIT("");
    return err;
}

static LwSciError ValidateOutputAttrKeysForKeyTypeHelper(
    LwSciBufAttrList reconciledList,
    const LwSciBufOutputAttrKeyValuePair* outputAttributes,
    LwSciBufType bufType,
    LwSciBufAttrKeyType attrKeyType,
    size_t* verifiedAttrCount)
{
    LwSciError err = LwSciError_Success;
    LwSciBufAttrKeyIterator iter;
    bool keyEnd = false;
    bool datatypeEnd = false;
    bool keytypeEnd = false;
    uint32_t key = 0U;
    bool isOutputAttrValid = true;
    size_t totalAttrCount =0UL;
    size_t i = 0UL;
    LwSciBufKeyAccess keyAccess;
    uint32_t opAttrKey = 0U;
    const void* opAttrValue = NULL;
    size_t opAttrLen = 0UL;
    size_t dataSize = 0U;
    uint32_t dataMaxInstance = 0U;
    bool needKeyValidation = false;

    LWSCI_FNENTRY("");

    *verifiedAttrCount = 0UL;
    LwSciBufAttrKeyIterInit((uint32_t)attrKeyType, bufType, 0U, &iter);

    GetOutputAttrCountForKeyType(outputAttributes, attrKeyType,
                                &totalAttrCount);

    /* for all keys in a particular keyType & datatype */
    while (true) {
        LwSciBufAttrKeyIterNext(&iter, &keytypeEnd, &datatypeEnd,
                                &keyEnd, &key);

        if (true == keyEnd) {
            break;
        }

        /* we are only interested in keys which are mandatory in reconciled
         *  attribute list. */
        needKeyValidation = true;

        err = LwSciBufReconciledKeyCheckNeeded(reconciledList, key,
                &needKeyValidation);
        if (LwSciError_Success != err) {
            LWSCI_ERR_STR("Failed to get key details from reconciled attribute list");
            LwSciCommonPanic();
        }

        /* we are only interested in LwSciBufKeyAccess_Output keys*/
        LwSciBufAttrGetKeyAccessDetails(key, &keyAccess);
        needKeyValidation &= (LwSciBufKeyAccess_Output == keyAccess);

        if (false == needKeyValidation) {
            continue;
        }

        /* there is atleast 1 output only key in this keyType & dataType combination
         * validate if user provided output Attribute is valid
         * validate if user provided output Attribute len is valid
         * accumulate count of all validated keys */
        err = VerifyOutputAttrSanity(outputAttributes, attrKeyType);
        if (LwSciError_Success != err) {
            LWSCI_ERR_STR("Failed to validate output Attributes sanity");
            goto ret;
        }

        LwSciBufAttrGetDataDetails(key, &dataSize, &dataMaxInstance);
        /* search for this output key in user outputAttributes */
        for(i = 0UL; i < totalAttrCount; i++) {
            GetOutputAttrKeyIndexedData(outputAttributes, attrKeyType, i,
                    &opAttrKey, &opAttrValue, &opAttrLen);

            isOutputAttrValid = (opAttrKey == key);
            isOutputAttrValid &= (0U == opAttrLen % dataSize) &&
                                (opAttrLen <= (dataSize * dataMaxInstance));
            isOutputAttrValid &= (NULL != opAttrValue);
            if (true == isOutputAttrValid) {
                *verifiedAttrCount += 1;
                break;
            }
        }

        /* key was not found in the outputAttribute */
        if (i == totalAttrCount) {
            LWSCI_ERR_HEXUINT("Necessary output key not supplied by user: ", key);
            err = LwSciError_BadParameter;
            goto ret;
        }
    }

ret:
    LWSCI_FNEXIT("");
    return err;
}

static LwSciError ValidateOutputAttrKeysForKeyType(
    LwSciBufAttrList reconciledList,
    const LwSciBufOutputAttrKeyValuePair* outputAttributes,
    LwSciBufAttrKeyType attrKeyType)
{
    LwSciError err = LwSciError_Success;
    size_t numBufTypes = 0UL;
    size_t idx = 0UL;
    const LwSciBufType* bufTypePtr = NULL;
    LwSciBufType bufType;
    size_t ulwerifiedAttrCount = 0UL;
    size_t tempVerifiedAttrCount = 0UL;

    LWSCI_FNENTRY("");

    GetOutputAttrCountForKeyType(outputAttributes, attrKeyType,
                                &ulwerifiedAttrCount);

    err = LwSciBufAttrListGetDataTypes(reconciledList, &bufTypePtr,
                                        &numBufTypes);
    if (LwSciError_Success != err) {
        err = LwSciError_IlwalidState;
        LWSCI_ERR_STR("LwSciBufAttrListGetDataTypes failed");
        goto ret;
    }

    /* for all data types */
    for (idx = 0UL; idx < numBufTypes; idx++) {
        bufType = bufTypePtr[idx];

        tempVerifiedAttrCount = 0UL;
        err = ValidateOutputAttrKeysForKeyTypeHelper(reconciledList,
                outputAttributes, bufType, attrKeyType, &tempVerifiedAttrCount);
        if (LwSciError_Success != err) {
            LWSCI_ERR_STR("Failed to validate output Attributes");
            err = LwSciError_BadParameter;
            goto ret;
        }

        /* if we reached here, means we have verified tempVerifiedAttrCount
         * number of Attrs for bufType dataType */
        ulwerifiedAttrCount -= tempVerifiedAttrCount;
    }

    if (0UL != ulwerifiedAttrCount) {
        LWSCI_ERR_STR("Unable to verify all attributes in outputAttributes");
        err = LwSciError_BadParameter;
        goto ret;
    }

ret:
    LWSCI_FNEXIT("");
    return err;
}

static LwSciError ValidateOutputAttrKeys(
    LwSciBufAttrList reconciledList,
    const LwSciBufOutputAttrKeyValuePair* outputAttributes)
{
    LwSciError err = LwSciError_Success;

    LWSCI_FNENTRY("");

    err = ValidateOutputAttrKeysForKeyType(reconciledList, outputAttributes,
              LwSciBufAttrKeyType_Public);
    if (err != LwSciError_Success) {
        LWSCI_ERR_STR("Failed to validate public outputAttributes");
        goto ret;
    }

    err = ValidateOutputAttrKeysForKeyType(reconciledList, outputAttributes,
             LwSciBufAttrKeyType_Internal);
    if (err != LwSciError_Success) {
        LWSCI_ERR_STR("Failed to validate internal outputAttributes");
        goto ret;
    }

ret:
    LWSCI_FNEXIT("");
    return err;
}

LwSciError LwSciBufAttrListReconcileWithOutputAttrs(
    LwSciBufAttrList inputUnreconciledAttrLists[],
    size_t attrListCount,
    LwSciBufOutputAttrKeyValuePair outputAttributes,
    LwSciBufAttrList* outputReconciledAttrList,
    LwSciBufAttrList* conflictAttrList)
{
    LwSciError err = LwSciError_Success;

    LWSCI_FNENTRY("");

    if (NULL == inputUnreconciledAttrLists ||
        0UL == attrListCount ||
        NULL == outputReconciledAttrList ||
        NULL == conflictAttrList) {
        LWSCI_ERR_STR("Bad parameters to LwSciBufAttrListReconcileWithOutputAttrs");
        err = LwSciError_BadParameter;
        goto ret;
    }

    *outputReconciledAttrList = NULL;
    *conflictAttrList = NULL;

    err = LwSciBufAttrListReconcile(inputUnreconciledAttrLists, attrListCount,
            outputReconciledAttrList, conflictAttrList);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("Failed to reconcile input attr lists");
        // Failed to reconcile return outputs from AttrListReconcile to user
        goto ret;
    }

    err = ValidateOutputAttrKeys(*outputReconciledAttrList, &outputAttributes);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("Failed to validate output attributes keys");
        goto free_reconciled_attrlist;
    }

    err = LwSciBufValidateOutputAttrData(*outputReconciledAttrList,
                                            &outputAttributes);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("Failed to validate output attributes data");
        goto free_reconciled_attrlist;
    }

    goto ret;

free_reconciled_attrlist:
    LwSciBufAttrListFree(*outputReconciledAttrList);
    *outputReconciledAttrList = NULL;

ret:
    LWSCI_FNEXIT("");
    return err;
}

LwSciError LwSciBufValidateGpuType(
    LwSciBufAttrList attrList,
    LwSciRmGpuId gpuId,
    LwSciBufGpuType gpuType,
    bool* match
)
{
    LwSciError err = LwSciError_Success;

    LWSCI_FNENTRY("");
    (void)gpuId;

    err = LwSciBufAttrListValidate(attrList);
    if (err != LwSciError_Success) {
        LWSCI_ERR_STR("LwSciBufAttrListValidate failed.");
        goto ret;
    }

    if (match == NULL) {
        LwSciCommonPanic();
    }
    *match = false;

    /* There is no iGPU on x86 system. Thus, directly check if GPU type
     * being matched against is dGPU
     */
    if (gpuType == LwSciBufGpuType_dGPU) {
        *match = true;
    }

ret:
    LWSCI_FNEXIT("");
    return err;
}
