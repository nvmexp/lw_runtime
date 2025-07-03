/*
 * Copyright (c) 2018-2022, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 */

#include "lwscibuf_attr_reconcile_priv.h"
#include "lwscibuf_attr_reconcile_platform.h"
#include "lwscibuf_attr_key_dep.h"
#include "lwscibuf_attr_reconcile_image_tensor.h"

/* map of functions to figure out interdependency of keys in same datatypes
 * as well as interdependency of keys in different datatypes when interop is
 * requested via LwSciBufType[] array
 */
LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 9_3), "LwSciBuf-REQ-MISRAC2012-002")
LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 11_9), "LwSciBuf-REQ-MISRAC2012-004")
static const LwSciBufAttrListKeyDependency
        keyDependency[LwSciBufType_MaxValid][LwSciBufType_MaxValid] = {
        [LwSciBufType_RawBuffer][LwSciBufType_RawBuffer]    = NULL,

        [LwSciBufType_Image][LwSciBufType_Image]            = NULL,

        [LwSciBufType_Tensor][LwSciBufType_Tensor]          =
                                    LwSciBufAttrListTensorKeyDependency,

        [LwSciBufType_Array][LwSciBufType_Array]            = NULL,

        [LwSciBufType_Pyramid][LwSciBufType_Pyramid]        = NULL,

        [LwSciBufType_Image][LwSciBufType_Tensor]           =
                                    LwSciBufAttrListImageTensorKeyDependency,

        [LwSciBufType_Tensor][LwSciBufType_Image]           =
                                    LwSciBufAttrListImageTensorKeyDependency,
};
LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 11_9))
LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 9_3))

static const LwSciBufPolicySameKey
        recPolicyMapSameKey[LwSciBuf_PolicyUpperBound] = {
    [LwSciBuf_MatchPolicy]             = MatchPolicySameKey,
    [LwSciBuf_OrPolicy]                = OrPolicySameKey,
    [LwSciBuf_MaxPolicy]               = MaxPolicySameKey,
    [LwSciBuf_ArrayUnionPolicy]        = ArrayUnionPolicySameKey,
    [LwSciBuf_ListUnionPolicy]         = ListUnionPolicySameKey,
    [LwSciBuf_ArrayIntersectionPolicy] = ArrayIntersectionPolicySameKey,
    [LwSciBuf_GpuCacheAndPolicy]       = GpuCacheAndPolicySameKey,
    [LwSciBuf_GpuCompressionMatchPolicy]    = GpuCompressionMatchPolicySameKey,
    [LwSciBuf_IlwalidPolicy]           = NULL,
};

/** @brief Static function definition */
static LwSciError GpuCompressionMatchPolicySameKey(
    uint32_t key,
    const void* ipAddr,
    const LwSciBufAttrStatus* ipStatus,
    const uint64_t* ipSetSize,
    void* recAddr,
    LwSciBufAttrStatus* recStatus,
    uint64_t* recSetSize,
    const LwSciBufReconcilePolicyCookie* cookie)
{
    LwSciError err = LwSciError_Success;
    uint64_t setLen1 = 0U;
    uint64_t setLen2 = 0U;
    size_t dataSize = 0U;
    uint32_t dataMaxInstance = 0U;
    uint32_t index1 = 0U;
    uint32_t index2 = 0U;
    const LwSciBufAttrValGpuCompression* ipGpuCompression = NULL;
    LwSciBufAttrValGpuCompression* recGpuCompression = NULL;

    LWSCI_FNENTRY("");

    (void)ipStatus;
    (void)recStatus;
    (void)cookie;

    LwSciBufAttrGetDataDetails(key, &dataSize, &dataMaxInstance);

    if (NULL == ipSetSize) {
        /* When reconciling lists of two different datatypes,
         * input attrlist of one datatype will not have memory allocated
         * for holding values of keys of other datatype. In that case do nothing
         */
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    setLen1 = *ipSetSize / dataSize;
    ipGpuCompression = (const LwSciBufAttrValGpuCompression *)ipAddr;
    recGpuCompression = (LwSciBufAttrValGpuCompression *)recAddr;

    for (index1 = 0U; index1 < setLen1; index1++) {
        setLen2 = *recSetSize / dataSize;

        for (index2 = 0U; index2 < setLen2; index2++) {
            if (LwSciCommonMemcmp(&ipGpuCompression[index1].gpuId,
                &recGpuCompression[index2].gpuId,
                sizeof(recGpuCompression[index2].gpuId)) == 0U) {
                if (LwSciCommonMemcmp(
                    &ipGpuCompression[index1].compressionType,
                    &recGpuCompression[index2].compressionType,
                    sizeof(recGpuCompression[index2].compressionType))
                    != 0U) {
                    goto reconcile_fail;
                } else {
                    break;
                }
            }
        }

        if ((index2 == setLen2) && (setLen2 >= dataMaxInstance)) {
            /* Trying to set more entries than allowed by LwSciBuf */
            LWSCI_ERR_UINT("Maximum entries allowed to be stored by LwSciBuf: ",
                dataMaxInstance);
            goto reconcile_fail;
        } else if (index2 == setLen2){
            LwSciCommonMemcpyS(&recGpuCompression[index2], dataSize,
                &ipGpuCompression[index1], dataSize);
            *recSetSize += dataSize;
        } else {
            /* do nothing */
        }
    }

    LWSCI_INFO("Output: recSetSize %lu", *recSetSize);

    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
    goto ret;

reconcile_fail:
    err = LwSciError_ReconciliationFailed;

ret:
    LWSCI_FNEXIT("");
    return err;
}

static LwSciError GpuCacheAndPolicySameKey(
    uint32_t key,
    const void* ipAddr,
    const LwSciBufAttrStatus* ipStatus,
    const uint64_t* ipSetSize,
    void* recAddr,
    LwSciBufAttrStatus* recStatus,
    uint64_t* recSetSize,
    const LwSciBufReconcilePolicyCookie* cookie)
{
    LwSciError err = LwSciError_Success;
    uint64_t setLen1 = 0U;
    size_t dataSize = 0U;
    uint32_t dataMaxInstance = 0U;
    const LwSciBufAttrValGpuCache* ipGpuCache = NULL;
    LwSciBufAttrValGpuCache* recGpuCache = NULL;
    LwSciBufAttrList ipAttrList = NULL;
    LwSciBufAttrList recAttrList = NULL;
    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")

    LWSCI_FNENTRY("");

    (void)recStatus;

    ipAttrList = cookie->appendList;
    recAttrList = cookie->reconciledList;

    LwSciBufAttrGetDataDetails(key, &dataSize, &dataMaxInstance);

    if (NULL == ipSetSize) {
        /* When reconciling lists of two different datatypes,
         * input attrlist of one datatype will not have memory allocated
         * for holding values of keys of other datatype. In that case do nothing
         */
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    setLen1 = *ipSetSize / dataSize;

    /* LwSciBuf only supports GPU cacheability control for single iGPU.
     */
    if (setLen1 > 1U) {
        LWSCI_ERR_STR("GPU cacheability control is supported for single GPU Only as of now.");
        goto reconcile_fail;
    }

    ipGpuCache = (const LwSciBufAttrValGpuCache *)ipAddr;
    recGpuCache = (LwSciBufAttrValGpuCache *)recAddr;

    if (*ipSetSize != 0U) {
        bool match = false;
        err = LwSciBufValidateGpuType(ipAttrList, (*ipGpuCache).gpuId,
                LwSciBufGpuType_iGPU, &match);
        if (err != LwSciError_Success) {
            LWSCI_ERR_STR("LwSciBufValidateGpuType failed.");
            goto reconcile_fail;
        }

        /* Lwrrently, LwSciBuf allows specifying GPU cacheability control for
         * iGPU only.
         */
        if (match == false) {
            LWSCI_ERR_STR("GPU cacheability control is supported for iGPU Only as of now.");
            goto reconcile_fail;
        }
    }

    if ((*ipStatus != LwSciBufAttrStatus_Empty) &&
        (*recSetSize != 0U)) {
        /* lwrrently, GPU cacheability control is supported for single iGPU
         * only and thus, all unreconciled attribute lists that have
         * specified this value should have this value specified for the
         * same GPU for reconciliation.
         */
        if (LwSciCommonMemcmp(&ipGpuCache->gpuId, &recGpuCache->gpuId,
            sizeof(ipGpuCache->gpuId)) != 0) {
            LWSCI_ERR_STR("Unreconciled list specified cacheability value for the GPU ID which is different than GPU ID for which this value is specified by other unreconciled attribute list(s).");
            goto reconcile_fail;
        }

        recGpuCache->cacheability = (recGpuCache->cacheability &&
                                        ipGpuCache->cacheability);
    } else if ((*ipStatus != LwSciBufAttrStatus_Empty) &&
        (*recSetSize == 0U)) {
        *recGpuCache = *ipGpuCache;
        *recSetSize = *ipSetSize;
    } else if ((*ipStatus == LwSciBufAttrStatus_Empty) &&
        (*recSetSize != 0U)) {
        bool defaultCacheValue = false;
        /* Get the default value for the GPU type/platform/Memory domain */
        err = LwSciBufAttrListPlatformGetDefaultGpuCacheability(recAttrList,
            recGpuCache->gpuId, &defaultCacheValue);
        if (err != LwSciError_Success) {
            LWSCI_ERR_STR("LwSciBufAttrListPlatformGetDefaultGpuCacheability() failed.");
            goto reconcile_fail;
        }

        recGpuCache->cacheability = (recGpuCache->cacheability &&
                                        defaultCacheValue);
    } else {
        /* do nothing */
    }

    if ((0UL != setLen1) && (0UL == *recSetSize)) {
        goto reconcile_fail;
    }

    LWSCI_INFO("Output: recSetSize %lu", *recSetSize);

    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
    goto ret;

reconcile_fail:
    err = LwSciError_ReconciliationFailed;

ret:
    LWSCI_FNEXIT("");
    return err;
}

static LwSciError MatchPolicySameKey(
    uint32_t key,
    const void* ipAddr,
    const LwSciBufAttrStatus* ipStatus,
    const uint64_t* ipSetSize,
    void* recAddr,
    LwSciBufAttrStatus* recStatus,
    uint64_t* recSetSize,
    const LwSciBufReconcilePolicyCookie* cookie)
{
    LwSciError err = LwSciError_Success;
    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Directive, 4_6), "LwSciBuf-ADV-MISRAC2012-006")
    int mergeErr = 0;
    const uint32_t* compare1 = NULL;
    const uint32_t* compare2 = NULL;
    size_t dataSize = 0U;
    uint32_t dataMaxInstance = 0U;

    LWSCI_FNENTRY("");

    (void)recStatus;
    (void)cookie;

    LwSciBufAttrGetDataDetails(key, &dataSize, &dataMaxInstance);

    if (NULL == ipSetSize) {
        /* When reconciling lists of two different datatypes,
         * input attrlist of one datatype will not have memory allocated
         * for holding values of keys of other datatype. In that case do nothing
         */
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    if ((LwSciBufAttrStatus_Empty != *ipStatus) &&
        /* if both values are present we need to reconcile */
        (0UL != *recSetSize)) {
        if (*ipSetSize != *recSetSize) {
            /* size mismatch. */
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
                "LwSciBuf-ADV-MISRAC2012-015")
            goto reconcile_fail;
        }

        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
        compare1 = ipAddr; compare2 = recAddr;
        mergeErr = LwSciCommonMemcmp(compare1, compare2, *ipSetSize);
        LWSCI_INFO("app != empty, rec != empty, comparing %d\n", mergeErr);
        if (0 != mergeErr) {
            LWSCI_ERR_STR("Failed to compare\n");
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
            goto reconcile_fail;
        }
    } else if ((LwSciBufAttrStatus_Empty != *ipStatus) &&
                (0UL == *recSetSize)) {
        /* if reconcile value is missing, copy from input attr list*/
        LWSCI_INFO("app != empty, rec == empty, copying\n");
        LwSciCommonMemcpyS(recAddr, *ipSetSize, ipAddr, *ipSetSize);
        *recSetSize = *ipSetSize;
    } else {
        /* do nothing */
    }

    LWSCI_INFO("Output: recSetSize %lu", *recSetSize);

    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
    goto ret;

reconcile_fail:
    err = LwSciError_ReconciliationFailed;

ret:
    LWSCI_FNEXIT("");
    return err;
}

static LwSciError OrPolicySameKey(
    uint32_t key,
    const void* ipAddr,
    const LwSciBufAttrStatus* ipStatus,
    const uint64_t* ipSetSize,
    void* recAddr,
    LwSciBufAttrStatus* recStatus,
    uint64_t* recSetSize,
    const LwSciBufReconcilePolicyCookie* cookie)
{
    LwSciError err = LwSciError_Success;
    uint64_t instance = 0U;
    uint64_t setLen1 = 0U;
    uint64_t setLen2 = 0U;
    uint64_t setLen = 0U;
    size_t dataSize = 0U;
    uint32_t dataMaxInstance = 0U;
    const bool* tmpIpAddr = NULL;
    bool* tmpRecAddr = NULL;

    LWSCI_FNENTRY("");

    (void)ipStatus;
    (void)recStatus;
    (void)cookie;

    LwSciBufAttrGetDataDetails(key, &dataSize, &dataMaxInstance);

    if (NULL == ipSetSize) {
        /* When reconciling lists of two different datatypes,
         * input attrlist of one datatype will not have memory allocated
         * for holding values of keys of other datatype. In that case do nothing
         */
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    setLen1 = *ipSetSize / dataSize;
    setLen2 = *recSetSize / dataSize;

    if (setLen1 > setLen2) {
        setLen = setLen2;
    } else {
        setLen = setLen1;
    }

    LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 11_5),
        "LwSciBuf-ADV-MISRAC2012-014")
    tmpIpAddr = (const bool *)ipAddr;
    tmpRecAddr = (bool *)recAddr;
    LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 11_5))

    for (instance = 0U; instance < setLen; instance++) {
        tmpRecAddr[instance] = tmpIpAddr[instance] || tmpRecAddr[instance];
    }

    if (setLen1 > setLen2) {
        uint64_t index = 0UL;

        for (index = instance; index < setLen1; index++) {
            tmpRecAddr[index] = tmpIpAddr[index];
            *recSetSize += dataSize;
        }
    }

    LWSCI_INFO("Output: recSetSize %lu", *recSetSize);

ret:
    LWSCI_FNEXIT("");
    return err;
}

static LwSciError MaxPolicySameKey(
    uint32_t key,
    const void* ipAddr,
    const LwSciBufAttrStatus* ipStatus,
    const uint64_t* ipSetSize,
    void* recAddr,
    LwSciBufAttrStatus* recStatus,
    uint64_t* recSetSize,
    const LwSciBufReconcilePolicyCookie* cookie)
{
    LwSciError err = LwSciError_Success;
    uint64_t instance = 0U;
    uint64_t setLen1 = 0U;
    uint64_t setLen2 = 0U;
    uint64_t setLen = 0U;
    size_t dataSize = 0U;
    uint32_t dataMaxInstance = 0U;
    const uint8_t* tmpIpAddr = NULL;
    uint8_t* tmpRecAddr = NULL;

    LWSCI_FNENTRY("");

    (void)ipStatus;
    (void)recStatus;
    (void)cookie;

    LwSciBufAttrGetDataDetails(key, &dataSize, &dataMaxInstance);

    if (NULL == ipSetSize) {
        /* When reconciling lists of two different datatypes,
         * input attrlist of one datatype will not have memory allocated
         * for holding values of keys of other datatype. In that case do nothing
         */
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    setLen1 = *ipSetSize / dataSize;
    setLen2 = *recSetSize / dataSize;

    if (setLen1 > setLen2) {
        setLen = setLen2;
    } else {
        setLen = setLen1;
    }

    LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 11_5),
        "LwSciBuf-ADV-MISRAC2012-014")
    tmpIpAddr = (const uint8_t *)ipAddr;
    tmpRecAddr = (uint8_t *)recAddr;
    LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 11_5))

    for (instance = 0U; instance < setLen; instance++) {
        if (0 > LwSciCommonMemcmp(tmpRecAddr, tmpIpAddr, dataSize)) {
            LwSciCommonMemcpyS(tmpRecAddr, dataSize, tmpIpAddr, dataSize);
        }

        LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 18_4),
            "LwSciBuf-ADV-MISRAC2012-017")
        tmpRecAddr += dataSize;
        tmpIpAddr += dataSize;
        LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 18_4))
    }

    if (setLen1 > setLen2) {
        uint64_t index = 0UL;

        for (index = instance; index < setLen1; index++) {
            LwSciCommonMemcpyS(tmpRecAddr, dataSize, tmpIpAddr, dataSize);

            LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 18_4),
                "LwSciBuf-ADV-MISRAC2012-017")
            tmpRecAddr += dataSize;
            tmpIpAddr += dataSize;
            LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 18_4))
            *recSetSize += dataSize;
        }
    }

    LWSCI_INFO("Output: recSetSize %lu", *recSetSize);

ret:
    LWSCI_FNEXIT("");
    return err;
}

static LwSciError PushKeyValToIpcTable(
    LwSciBufAttrList recAttrList,
    uint64_t slotIndex,
    uint32_t key,
    uint64_t len,
    const void* value)
{
    LwSciError err = LwSciError_Success;
    LwSciBufPrivateAttrKeyValuePair pvtKeyValPair = {0};
    LwSciBufIpcTable* const * ipcTablePtr = NULL;

    LWSCI_FNENTRY("");

    pvtKeyValPair.key = LwSciBufPrivateAttrKey_IPCTable;
    err = LwSciBufAttrListCommonGetAttrs(recAttrList, 0, &pvtKeyValPair, 1,
            LwSciBufAttrKeyType_Private, true);
    if (err != LwSciError_Success) {
        LWSCI_ERR_UINT("Failed to get IPC Table information: ", err);
        goto ret;
    }

    if (pvtKeyValPair.len == 0U) {
        /* This should not happen */
        LwSciCommonPanic();
    }

    ipcTablePtr = (LwSciBufIpcTable* const *)pvtKeyValPair.value;

    err = LwSciBufIpcAddAttributeToTable(*ipcTablePtr, slotIndex, key, len,
            value);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("LwSciBufIpcAddAttributeToTable() failed.");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
            "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

#if (LW_IS_SAFETY == 0)
    LWSCI_INFO("Outputs: \n");
    LwSciBufPrintIpcTable(*ipcTablePtr);
#endif

ret:
    LWSCI_FNEXIT("");
    return err;
}

static LwSciError ArrayUnionPolicySameKey(
    uint32_t key,
    const void* ipAddr,
    const LwSciBufAttrStatus* ipStatus,
    const uint64_t* ipSetSize,
    void* recAddr,
    LwSciBufAttrStatus* recStatus,
    uint64_t* recSetSize,
    const LwSciBufReconcilePolicyCookie* cookie)
{
    LwSciError err = LwSciError_Success;
    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 8_13), "LwSciBuf-ADV-MISRAC2012-012")
    const void* iterAddr1 = NULL;
    void* iterAddr2 = NULL;
    const uint32_t* compare1 = NULL;
    const uint32_t* compare2 = NULL;
    uint64_t i = 0U, j = 0U;
    uint64_t setLen1 = 0UL, setLen2 = 0UL;
    size_t dataSize = 0U;
    uint32_t dataMaxInstance = 0U;

    LWSCI_FNENTRY("");

    (void)ipStatus;
    (void)recStatus;
    (void)cookie;

    LwSciBufAttrGetDataDetails(key, &dataSize, &dataMaxInstance);

    if (NULL == ipSetSize) {
        /* When reconciling lists of two different datatypes,
         * input attrlist of one datatype will not have memory allocated
         * for holding values of keys of other datatype. In that case do nothing
         */
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    setLen1 = *ipSetSize / dataSize;

    for (i=0; i<setLen1; i++) {
        LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 18_4), "LwSciBuf-ADV-MISRAC2012-017")
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
        iterAddr1 = (const char*)ipAddr + (dataSize * i);
        LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 18_4))
        setLen2 = *recSetSize / dataSize;
        for (j=0; j<setLen2; j++) {
            LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 18_4), "LwSciBuf-ADV-MISRAC2012-017")
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
            iterAddr2 = (char*)recAddr + (dataSize * j);
            LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 18_4))
            LWSCI_INFO("comparing i = %lu, j = %lu \n", i, j);
            LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
            compare2 = iterAddr2;
            compare1 = iterAddr1;
            LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 11_5))
            if (LwSciCommonMemcmp(compare1, compare2, dataSize) == 0) {
                LWSCI_INFO(" found skipping\n");
                break;
            }
        }

        if (j >= dataMaxInstance) {
            /*
             * This means we have previously reconciled all maxInstances.
             * Still we got one more unique entry, this means
             *  i) either user has given more unique entries than needed
             *  ii) LwSciBuf internal container size is less than max possible
             *      unique entries.
             * in either senario mark failed to reconcile.
             */
            LWSCI_ERR_ULONG("Found ", j + 1U);
            LWSCI_ERR_UINT("the unique entry which is greater than", dataMaxInstance);
            LWSCI_ERR_STR("max container size\n");
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
            goto reconcile_fail;
        }

        if (j == setLen2) {
            LWSCI_INFO(" not found adding at %lu\n", j);
            LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 18_4), "LwSciBuf-ADV-MISRAC2012-017")
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
            iterAddr2 = (char*)recAddr + (dataSize * j);
            LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 18_4))
            LwSciCommonMemcpyS(iterAddr2, dataSize, iterAddr1, dataSize);
            *recSetSize = ((uint64_t)(j + 1U) * dataSize);
        }
    }

    LWSCI_INFO("Output: recSetSize %lu", *recSetSize);

    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
    goto ret;

reconcile_fail:
    err = LwSciError_ReconciliationFailed;

ret:
    LWSCI_FNEXIT("");
    return err;
}

static void SwapBuf(
    void* a,
    void* b,
    size_t n)
{
    size_t tempSize = n;
    LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
    char* tempA = a;
    char* tempB = b;
    LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 11_5))

    if (0U == n) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    do {
        char tempTmp = *tempA;
        *tempA = *tempB;
        *tempB = tempTmp;
        tempA++;
        tempB++;
    } while (0UL < --tempSize);

ret:
    return;
}

static LwSciError ArrayIntersectionPolicySameKey(
    uint32_t key,
    const void* ipAddr,
    const LwSciBufAttrStatus* ipStatus,
    const uint64_t* ipSetSize,
    void* recAddr,
    LwSciBufAttrStatus* recStatus,
    uint64_t* recSetSize,
    const LwSciBufReconcilePolicyCookie* cookie)
{
    LwSciError err = LwSciError_Success;

    uint64_t setLen1 = 0UL;
    size_t dataSize = 0U;
    uint32_t dataMaxInstance = 0U;

    LWSCI_FNENTRY("");

    (void)ipStatus;
    (void)recStatus;
    (void)cookie;

    LwSciBufAttrGetDataDetails(key, &dataSize, &dataMaxInstance);

    if (NULL == ipSetSize) {
        /* When reconciling lists of two different datatypes,
         * input attrlist of one datatype will not have memory allocated
         * for holding values of keys of other datatype. In that case do nothing
         */
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    setLen1 = *ipSetSize / dataSize;

    if (0U == *recSetSize) {
        if (NULL != ipAddr) {
            /* If the reconciled list is empty, then this is the first slot and
             * so we need to copy everything into the reconciled list (since
             * the intersection of a single array is itself). */
            const void* iterAddr1 = ipAddr;
            void* iterAddr2 = recAddr;

            LwSciCommonMemcpyS(iterAddr2, *ipSetSize, iterAddr1, *ipSetSize);
            *recSetSize = *ipSetSize;
        }
    } else {
        uint64_t i = 0U;
        uint64_t lwrrentIndex = 0U;
        uint64_t reconciledAttrLen = *recSetSize / dataSize;
        uint8_t arithmeticStatus = OP_SUCCESS;
        uint64_t tmpArithmetic = 0U;

        /* Otherwise, for each element:
         *  1. Check if it exists in the reconciled list.
         *  2. If not, then remove it from the reconciled list
         */
        for (i = 0U; i < setLen1; i++) {
            const void* iterAddr1 = NULL;
            uint64_t j = 0U;

            u64Mul(dataSize, i, &tmpArithmetic, &arithmeticStatus);
            if (arithmeticStatus != OP_SUCCESS) {
                LwSciCommonPanic();
            }

            iterAddr1 = (const char*)ipAddr + tmpArithmetic;

            for (j = lwrrentIndex; j < reconciledAttrLen; j++) {
                LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 18_4), "LwSciBuf-ADV-MISRAC2012-017")
                LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
                void* iterAddr2 = (char*)recAddr + (dataSize * j);
                LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 18_4))

                if (LwSciCommonMemcmp(iterAddr1, iterAddr2, dataSize) == 0) {
                    void* iterAddr3 = NULL;
                    LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 18_4), "LwSciBuf-ADV-MISRAC2012-017")
                    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
                    iterAddr3 = (char*)recAddr + (dataSize * lwrrentIndex);
                    LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 18_4))

                    /* Swap the two values at recAddr[lwrrentIndex] and recAddr[j] */
                    SwapBuf(iterAddr2, iterAddr3, dataSize);
                    u64Add(lwrrentIndex, 1, &lwrrentIndex, &arithmeticStatus);
                    if (arithmeticStatus != OP_SUCCESS) {
                        LwSciCommonPanic();
                    }
                    break;
                }
            }
        }

        if (0U != setLen1) {
            /* Update the length only when this slot is non-empty. */
            size_t maxDataSize = dataMaxInstance * dataSize;
            void* startEndPtr = NULL;

            *recSetSize = (uint64_t)(lwrrentIndex * dataSize);

            /* Zero out any memory after the end of the reconciled array for
             * ease of debugging */
            LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 18_4), "LwSciBuf-ADV-MISRAC2012-017")
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
            startEndPtr = (void*)((char*)recAddr + *recSetSize);
            LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 18_4))
            (void)memset(startEndPtr, 0x0, maxDataSize - *recSetSize);

            if (0U == *recSetSize) {
                LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
                goto reconcile_fail;
            }
        }
    }

    LWSCI_INFO("Output: recSetSize %lu", *recSetSize);

    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
    goto ret;

reconcile_fail:
        err = LwSciError_ReconciliationFailed;

ret:
    LWSCI_FNEXIT("");
    return err;
}

static LwSciError ListUnionPolicySameKey(
    uint32_t key,
    const void* ipAddr,
    const LwSciBufAttrStatus* ipStatus,
    const uint64_t* ipSetSize,
    void* recAddr,
    LwSciBufAttrStatus* recStatus,
    uint64_t* recSetSize,
    const LwSciBufReconcilePolicyCookie* cookie)
{
    LwSciError err = LwSciError_Success;
    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Directive, 4_6), "LwSciBuf-ADV-MISRAC2012-006")
    int error = 0;
    LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 8_13), "LwSciBuf-ADV-MISRAC2012-012")
    const uint8_t* compare1 = NULL;
    const uint8_t* compare2 = NULL;
    const LwSciBufUmdAttrValData* iterNode = NULL;
    LWListRec* iterNodeHead = NULL;
    LwSciBufUmdAttrValData* slotNode = NULL;
    const LWListRec* slotNodeHead = NULL;
    LwSciBufUmdAttrValData* privateKeyNode = NULL;
    LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 8_13))
    size_t dataSize = 0U;
    uint32_t dataMaxInstance = 0U;

    LWSCI_FNENTRY("");

    (void)ipStatus;
    (void)recStatus;
    (void)recSetSize;
    (void)cookie;

    LwSciBufAttrGetDataDetails(key, &dataSize, &dataMaxInstance);

    if (NULL == ipSetSize) {
        /* When reconciling lists of two different datatypes,
         * input attrlist of one datatype will not have memory allocated
         * for holding values of keys of other datatype. In that case do nothing
         */
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
    slotNodeHead = ipAddr;
    iterNodeHead = recAddr;
    LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 11_5))

    LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 15_4), "LwSciBuf-ADV-MISRAC2012-016")
    LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 18_4), "LwSciBuf-ADV-MISRAC2012-017")
    LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
    LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Directive, 4_9), "LwSciBuf-ADV-MISRAC2012-008")
    LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 1_2), "LwSciBuf-ADV-MISRAC2012-001")
    lwListForEachEntry(slotNode, slotNodeHead, listEntry) {
        /* for all nodes in reconciled list */
        lwListForEachEntry(iterNode, iterNodeHead, listEntry) {
            LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 1_2))
            LWCOV_ALLOWLIST_END(LWCOV_MISRA(Directive, 4_9))
            LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 11_5))
            LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 18_4))
            LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 15_4))
            LWSCI_INFO("For iter key %d \n", iterNode->key);

            if(slotNode->key == iterNode->key) {
                /* key is present try reconcile */
                LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
                compare1 = slotNode->value;
                compare2 = iterNode->value;
                LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 11_5))
                error = LwSciCommonMemcmp(compare1, compare2, slotNode->len);
                if (0 != error) {
                    LWSCI_INFO("Private key not matching\n");
                    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
                    goto reconcile_fail;
                }

                LWSCI_INFO("Found Key in reconciled list\n");
                break;
            } else if (iterNode->key > slotNode->key) {
                /* optimization since this is sorted list */
                break;
            } else {
                continue;
            }
        }

        /* key is not present in reconciled list, add it */
        if ((&iterNode->listEntry == recAddr) ||
            (iterNode->key > slotNode->key)) {

            err = LwSciBufCreatePrivateKeyNode(slotNode->key,
                    slotNode->len, slotNode->value, &privateKeyNode);
            if (LwSciError_Success != err) {
                LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
                goto reconcile_fail;
            }
            LWSCI_INFO("adding key %d to reconciled list\n", slotNode->key);

            lwListAppend(&privateKeyNode->listEntry, iterNodeHead);
        }
    }

    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
    goto ret;

reconcile_fail:
#if (LW_IS_SAFETY == 0)
    slotNode->privateAttrStatus = LwSciBufAttrStatus_Conflict;
#endif
        err = LwSciError_ReconciliationFailed;

ret:
    LWSCI_FNEXIT("");
    return err;
}

static LwSciError VerifyMatchPolicySameKey(
    uint32_t key,
    const void* ipAddr,
    const LwSciBufAttrStatus* ipStatus,
    const uint64_t* ipSetSize,
    void* recAddr,
    LwSciBufAttrStatus* recStatus,
    uint64_t* recSetSize,
    const LwSciBufReconcilePolicyCookie* cookie)
{
    LwSciError err = LwSciError_Success;
    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Directive, 4_6), "LwSciBuf-ADV-MISRAC2012-006")
    int mergeErr = 0;
    const uint32_t* compare1 = NULL;
    const uint32_t* compare2 = NULL;
    uint64_t instance = 0U;
    uint64_t setLen1 = 0U;
    size_t dataSize = 0U;
    uint32_t dataMaxInstance = 0U;

    LWSCI_FNENTRY("");

    (void)recSetSize;
    (void)cookie;

    LwSciBufAttrGetDataDetails(key, &dataSize, &dataMaxInstance);

    setLen1 = *ipSetSize / dataSize;

    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_4), "LwSciBuf-ADV-MISRAC2012-016")
    for (instance = 0; instance < setLen1; instance++) {
        LWSCI_INFO("Merging for instance %u\n", instance);

        if ((LwSciBufAttrStatus_Empty != *ipStatus ) &&
            (LwSciBufAttrStatus_Empty != *recStatus)) {
            LWSCI_INFO("app != empty, rec != empty, comparing %d\n", mergeErr);
            LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
            compare1 = ipAddr;
            compare2 = recAddr;
            LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 11_5))
            mergeErr = LwSciCommonMemcmp(compare1, compare2, dataSize);
            if (0 != mergeErr) {
                LWSCI_ERR_STR("Failed to compare\n");
                LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
                goto reconcile_fail;
            }
        } else if ((LwSciBufAttrStatus_Empty != *ipStatus ) &&
                    (LwSciBufAttrStatus_Empty == *recStatus)) {
            LWSCI_INFO("app != empty, rec == empty, error\n");
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
            goto reconcile_fail;
        } else {
            /* do nothing */
        }
        LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 18_4), "LwSciBuf-ADV-MISRAC2012-017")
        LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
        ipAddr = (const char*)ipAddr + dataSize;
        recAddr = (char*)recAddr + dataSize;
        LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 11_5))
        LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 18_4))
    }

    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
    goto ret;

reconcile_fail:
    err = LwSciError_ReconciliationFailed;
ret:
    LWSCI_FNEXIT("");
    return err;
}

static LwSciError VerifyOrPolicySameKey(
    uint32_t key,
    const void* ipAddr,
    const LwSciBufAttrStatus* ipStatus,
    const uint64_t* ipSetSize,
    void* recAddr,
    LwSciBufAttrStatus* recStatus,
    uint64_t* recSetSize,
    const LwSciBufReconcilePolicyCookie* cookie)
{
    LwSciError err = LwSciError_Success;
    uint64_t instance = 0U;
    uint64_t setLen1 = 0U;
    size_t dataSize = 0U;
    uint32_t dataMaxInstance = 0U;

    LWSCI_FNENTRY("");

    (void)recSetSize;
    (void)cookie;

    LwSciBufAttrGetDataDetails(key, &dataSize, &dataMaxInstance);

    setLen1 = *ipSetSize / dataSize;

    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_4), "LwSciBuf-ADV-MISRAC2012-016")
    for (instance = 0UL; instance < setLen1; instance++) {
        LWSCI_INFO("Merging for instance %u\n", instance);

        if ((LwSciBufAttrStatus_Empty != *ipStatus ) &&
            (LwSciBufAttrStatus_Empty != *recStatus)) {
            LWSCI_INFO("app != empty, rec != empty, comparing\n");
            LWSCI_INFO("before slot: %d, rec: %d\n", *(const bool*)ipAddr,
                *(bool*)recAddr);
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
            if (*(const bool*)ipAddr && !*(bool*)recAddr) {
                LWSCI_INFO("slot = 1, rec = 0, failed\n");
                LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
                goto reconcile_fail;
            }
        } else if ((LwSciBufAttrStatus_Empty != *ipStatus ) &&
                    (LwSciBufAttrStatus_Empty == *recStatus)) {
            LWSCI_INFO("app != empty, rec == empty\n");
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
            goto reconcile_fail;
        } else {
            /* Do nothing */
        }
        LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 18_4), "LwSciBuf-ADV-MISRAC2012-017")
        LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
        ipAddr = (const char*)ipAddr + dataSize;
        recAddr = (char*)recAddr + dataSize;
        LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 11_5))
        LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 18_4))
    }

    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
    goto ret;

reconcile_fail:
    err = LwSciError_ReconciliationFailed;

ret:
    LWSCI_FNEXIT("");
    return err;
}

static LwSciError VerifyMaxPolicySameKey(
    uint32_t key,
    const void* ipAddr,
    const LwSciBufAttrStatus* ipStatus,
    const uint64_t* ipSetSize,
    void* recAddr,
    LwSciBufAttrStatus* recStatus,
    uint64_t* recSetSize,
    const LwSciBufReconcilePolicyCookie* cookie)
{
    LwSciError err = LwSciError_Success;
    uint64_t val1 = 0U, val2 = 0U;
    const uint64_t* compare1 = NULL;
    const uint64_t* compare2 = NULL;
    uint64_t instance = 0U, setLen1 = 0U;
    size_t dataSize = 0U;
    uint32_t dataMaxInstance = 0U;

    LWSCI_FNENTRY("");

    (void)recSetSize;
    (void)cookie;

    LwSciBufAttrGetDataDetails(key, &dataSize, &dataMaxInstance);

    setLen1 = *ipSetSize / dataSize;

    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_4), "LwSciBuf-ADV-MISRAC2012-016")
    for (instance = 0U; instance < setLen1; instance++) {
        LWSCI_INFO("Merging for instance %u\n", instance);
        LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
        compare1 = ipAddr;
        compare2 = recAddr;
        LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 11_5))
        LwSciCommonMemcpyS(&val1, dataSize, compare1, dataSize);
        LwSciCommonMemcpyS(&val2, dataSize, compare2, dataSize);

        if ((LwSciBufAttrStatus_Empty != *ipStatus ) &&
            (LwSciBufAttrStatus_Empty != *recStatus)) {
            LWSCI_INFO("app != empty, rec != empty, comparing\n");
            LWSCI_INFO("before slot: %lu, rec: %lu\n", val1, val2);
            if (val1 > val2) {
                LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
                goto reconcile_fail;
            }
        } else if ((LwSciBufAttrStatus_Empty != *ipStatus ) &&
                    (LwSciBufAttrStatus_Empty == *recStatus)) {
            LWSCI_INFO("app != empty, rec == empty\n");
            LWSCI_INFO("before slot: %lu, rec: %lu\n", val1, val2);
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
            goto reconcile_fail;
        } else {
            /* Do nothing */
        }
        LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 18_4), "LwSciBuf-ADV-MISRAC2012-017")
        LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
        ipAddr = (const char*)ipAddr + dataSize;
        recAddr = (char*)recAddr + dataSize;
        LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 11_5))
        LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 18_4))
    }

    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
    goto ret;

reconcile_fail:
    err = LwSciError_ReconciliationFailed;

ret:
    LWSCI_FNEXIT("");
    return err;
}

static LwSciError VerifyArrayIntersectionPolicySameKey(
    uint32_t key,
    const void* ipAddr,
    const LwSciBufAttrStatus* ipStatus,
    const uint64_t* ipSetSize,
    void* recAddr,
    LwSciBufAttrStatus* recStatus,
    uint64_t* recSetSize,
    const LwSciBufReconcilePolicyCookie* cookie)
{
    LwSciError err = LwSciError_Success;
    uint64_t i = 0U;
    uint64_t j = 0U;
    uint64_t setLen1 = 0U;
    uint64_t setLen2 = 0U;
    size_t dataSize = 0U;
    uint32_t dataMaxInstance = 0U;

    LWSCI_FNENTRY("");

    (void)ipStatus;
    (void)recStatus;
    (void)cookie;

    LwSciBufAttrGetDataDetails(key, &dataSize, &dataMaxInstance);

    setLen1 = *ipSetSize / dataSize;
    setLen2 = *recSetSize / dataSize;

    /* Verification succeeds if the entire array in the Reconciled Attribute
     * List is a subset of the Unreconciled Attribute List */
    for (i = 0; i < setLen2; ++i) {
        if (0U != setLen1) {
            /* Skip any empty attribute lists */
            for (j = 0U; j < setLen1; j++) {
                LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 18_4), "LwSciBuf-ADV-MISRAC2012-017")
                LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
                const void* iterAddr1 = (const char*)ipAddr + (dataSize * j);
                const void* iterAddr2 = (char*)recAddr + (dataSize * i);
                LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 11_5))
                LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 18_4))

                if (LwSciCommonMemcmp(iterAddr1, iterAddr2, dataSize) == 0) {
                    break;
                }
            }
            if (j == setLen1) {
                LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
                goto reconcile_fail;
            }
        }
    }

    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
    goto ret;

reconcile_fail:
    err = LwSciError_ReconciliationFailed;

ret:
    LWSCI_FNEXIT("");
    return err;
}

static LwSciError VerifyArrayUnionPolicySameKey(
    uint32_t key,
    const void* ipAddr,
    const LwSciBufAttrStatus* ipStatus,
    const uint64_t* ipSetSize,
    void* recAddr,
    LwSciBufAttrStatus* recStatus,
    uint64_t* recSetSize,
    const LwSciBufReconcilePolicyCookie* cookie)
{
    LwSciError err = LwSciError_Success;
    const void* iterAddr1 = NULL;
    const void* iterAddr2 = NULL;
    const uint32_t* compare1 = NULL;
    const uint32_t* compare2 = NULL;
    uint64_t i = 0U, j = 0U;
    uint64_t setLen1 = 0U, setLen2 = 0U;
    size_t dataSize = 0U;
    uint32_t dataMaxInstance = 0U;

    LWSCI_FNENTRY("");

    (void)ipStatus;
    (void)recStatus;
    (void)cookie;

    LwSciBufAttrGetDataDetails(key, &dataSize, &dataMaxInstance);

    setLen1 = *ipSetSize / dataSize;

    for (i = 0U; i < setLen1; i++) {
        LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 18_4), "LwSciBuf-ADV-MISRAC2012-017")
        LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
        iterAddr1 = (const char*)ipAddr + (dataSize * i);
        setLen2 = *recSetSize / dataSize;
        for (j = 0U; j < setLen2; j++) {
            iterAddr2 = (char*)recAddr + (dataSize * j);
            LWSCI_INFO("comparing i = %lu, j = %lu \n", i, j);
            compare2 = iterAddr2;
            compare1 = iterAddr1;
            LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 11_5))
            LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 18_4))
            if (LwSciCommonMemcmp(compare1, compare2, dataSize) == 0) {
                LWSCI_INFO(" found skipping\n");
                break;
            }
        }

        /*
         * Unlike ArrayUnionPolicy, in this function we are not adding any entry
         * to reconciled list, hence there is no possible array overflow.
         * Here, we are only checking if the entry is present in reconciled list
         * Hence, no need for (j >= dataMaxInstance) condition
         */

        if (j == setLen2) {
            LWSCI_INFO(" not found failing\n");
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
            goto reconcile_fail;
        }
    }

    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
    goto ret;

reconcile_fail:
    err = LwSciError_ReconciliationFailed;

ret:
    LWSCI_FNEXIT("");
    return err;
}

static LwSciError VerifyListUnionPolicySameKey(
    uint32_t key,
    const void* ipAddr,
    const LwSciBufAttrStatus* ipStatus,
    const uint64_t* ipSetSize,
    void* recAddr,
    LwSciBufAttrStatus* recStatus,
    uint64_t* recSetSize,
    const LwSciBufReconcilePolicyCookie* cookie)
{
    LwSciError err = LwSciError_Success;
    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Directive, 4_6), "LwSciBuf-ADV-MISRAC2012-006")
    int error = 0;
    const uint8_t* compare1 = NULL;
    const uint8_t* compare2 = NULL;
    const LwSciBufUmdAttrValData* recNode = NULL;
    LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 8_13), "LwSciBuf-ADV-MISRAC2012-012")
    LWListRec* recNodeHead = NULL;
    LwSciBufUmdAttrValData* ipAttrNode = NULL;
    const LWListRec* ipAttrNodeHead = NULL;
    LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 8_13))
    size_t dataSize = 0U;
    uint32_t dataMaxInstance = 0U;

    LWSCI_FNENTRY("");

    (void)ipStatus;
    (void)ipSetSize;
    (void)recStatus;
    (void)recSetSize;
    (void)cookie;

    LwSciBufAttrGetDataDetails(key, &dataSize, &dataMaxInstance);

    LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
    ipAttrNodeHead = ipAddr;
    recNodeHead = recAddr;
    LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 11_5))

    LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 15_4), "LwSciBuf-ADV-MISRAC2012-016")
    LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 18_4), "LwSciBuf-ADV-MISRAC2012-017")
    LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Directive, 4_9), "LwSciBuf-ADV-MISRAC2012-008")
    LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
    LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 1_2), "LwSciBuf-ADV-MISRAC2012-001")
    lwListForEachEntry(ipAttrNode, ipAttrNodeHead, listEntry) {
        LWSCI_INFO("Searching slot key %d\n", ipAttrNode->key);

        /* for all nodes in reconciled list */
        lwListForEachEntry(recNode, recNodeHead, listEntry) {
            LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 1_2))
            LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 11_5))
            LWCOV_ALLOWLIST_END(LWCOV_MISRA(Directive, 4_9))
            LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 18_4))
            LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 15_4))
            LWSCI_INFO("For iter key %d\n", recNode->key);

            /* key is present try reconcile */
            if(ipAttrNode->key == recNode->key) {
                LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
                compare1 = ipAttrNode->value;
                compare2 = recNode->value;
                LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 11_5))
                LWSCI_INFO("Found Key in reconciled list\n");
                error = LwSciCommonMemcmp(compare1, compare2, ipAttrNode->len);
                if (0 != error) {
                    LWSCI_INFO("Private key not matching\n");
                    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
                    goto reconcile_fail;
                }

                LWSCI_INFO("Data matched\n");
                break;
            } else if (recNode->key > ipAttrNode->key) {
                break;
            } else {
                continue;
            }
        }

        /* key is not present in reconciled list, fail validation */
        if (&recNode->listEntry == recAddr) {
            LWSCI_INFO("Reached end, key not found\n");
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
            goto reconcile_fail;
        }
    }

    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
    goto ret;

reconcile_fail:
#if (LW_IS_SAFETY == 0)
    ipAttrNode->privateAttrStatus = LwSciBufAttrStatus_Conflict;
#endif
    err = LwSciError_ReconciliationFailed;
ret:
    LWSCI_FNEXIT("");
    return err;
}

static LwSciError VerifyGpuCacheAndPolicySameKey(
    uint32_t key,
    const void* ipAddr,
    const LwSciBufAttrStatus* ipStatus,
    const uint64_t* ipSetSize,
    void* recAddr,
    LwSciBufAttrStatus* recStatus,
    uint64_t* recSetSize,
    const LwSciBufReconcilePolicyCookie* cookie)
{
    LwSciError err = LwSciError_Success;
    (void)key;
    (void)ipAddr;
    (void)ipStatus;
    (void)ipSetSize;
    (void)recAddr;
    (void)recStatus;
    (void)recSetSize;
    (void)cookie;

    LWSCI_FNENTRY("");

    //TODO: Adding stub for now. Lets revisit this in subsequent patch.
    LWSCI_FNEXIT("");
    return err;
}

static LwSciError VerifyGpuCompressionMatchPolicySameKey(
    uint32_t key,
    const void* ipAddr,
    const LwSciBufAttrStatus* ipStatus,
    const uint64_t* ipSetSize,
    void* recAddr,
    LwSciBufAttrStatus* recStatus,
    uint64_t* recSetSize,
    const LwSciBufReconcilePolicyCookie* cookie)
{
    (void)key;
    (void)ipAddr;
    (void)ipStatus;
    (void)ipSetSize;
    (void)recAddr;
    (void)recStatus;
    (void)recSetSize;
    (void)cookie;

    return LwSciError_Success;
}

static LwSciError CompareDifferentKeysForMatch(
    uint32_t key1,
    uint32_t key2,
    LwSciBufAttrList reconciledList)
{
    LwSciError err = LwSciError_Success;
    uint32_t decodedKey = 0U, decodedDataType = 0U, decodedKeyType = 0U;
    const void* value1 = NULL, *value2 = NULL;
    size_t len1 = 0, len2 = 0, len = 0U;

    LWSCI_FNENTRY("");

    LWSCI_INFO("Input: reconciledList: %p, key1 %u, key2 %u\n", reconciledList,
        key1, key2);

    err = LwSciBufAttrKeyDecode(key1, &decodedKey, &decodedDataType,
            &decodedKeyType);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("LwSciBufAttrKeyDecode failed\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    if ((uint32_t)LwSciBufAttrKeyType_Public == decodedKeyType) {
        LwSciBufAttrKeyValuePair keyValPair = {0};

        LwSciCommonMemcpyS(&keyValPair.key, sizeof(keyValPair.key),
                            &key1, sizeof(key1));
        err = LwSciBufAttrListCommonGetAttrs(reconciledList, 0, &keyValPair, 1,
                LwSciBufAttrKeyType_Public, true);
        if (LwSciError_Success != err) {
            LWSCI_ERR_STR("LwSciBufAttrListCommonGetAttrs failed\n");
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
            goto ret;
        }

        value1 = keyValPair.value;
        len1 = keyValPair.len;
    } else if ((uint32_t)LwSciBufAttrKeyType_Internal == decodedKeyType) {
        LwSciBufInternalAttrKeyValuePair internalKeyValPair = {0};

        LwSciCommonMemcpyS(&internalKeyValPair.key, sizeof(internalKeyValPair.key),
                            &key1, sizeof(key1));
        err = LwSciBufAttrListCommonGetAttrs(reconciledList, 0,
                &internalKeyValPair, 1, LwSciBufAttrKeyType_Internal, true);
        if (LwSciError_Success != err) {
            LWSCI_ERR_STR("LwSciBufAttrListCommonGetAttrs failed\n");
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
            goto ret;
        }
        value1 = internalKeyValPair.value;
        len1 = internalKeyValPair.len;
    } else {
        /* no else needed */
    }

    err = LwSciBufAttrKeyDecode(key2, &decodedKey, &decodedDataType,
            &decodedKeyType);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("LwSciBufAttrKeyDecode failed\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    if ((uint32_t)LwSciBufAttrKeyType_Public == decodedKeyType) {
        LwSciBufAttrKeyValuePair keyValPair = {0};

        LwSciCommonMemcpyS(&keyValPair.key, sizeof(keyValPair.key),
                            &key2, sizeof(key2));
        err = LwSciBufAttrListCommonGetAttrs(reconciledList, 0, &keyValPair, 1,
                LwSciBufAttrKeyType_Public, true);
        if (LwSciError_Success != err) {
            LWSCI_ERR_STR("LwSciBufAttrListCommonGetAttrs failed\n");
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
            goto ret;
        }

        value2 = keyValPair.value;
        len2 = keyValPair.len;
    } else if((uint32_t)LwSciBufAttrKeyType_Internal == decodedKeyType) {
        LwSciBufInternalAttrKeyValuePair internalKeyValPair = {0};
        LwSciCommonMemcpyS(&internalKeyValPair.key, sizeof(internalKeyValPair.key),
            &key2, sizeof(key2));
        err = LwSciBufAttrListCommonGetAttrs(reconciledList, 0,
                &internalKeyValPair, 1, LwSciBufAttrKeyType_Internal, true);
        if (LwSciError_Success != err) {
            LWSCI_ERR_STR("LwSciBufAttrListCommonGetAttrs failed\n");
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
            goto ret;
        }

        value2 = internalKeyValPair.value;
        len2 = internalKeyValPair.len;
    } else {
         /* no else needed */
    }

    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 1_2), "LwSciBuf-ADV-MISRAC2012-001")
    LWSCI_INFO("Applying policy %s on keys %u and %u\n", __FUNCTION__, key1,
        key2);

    if ((0U == len1) || (0U == len2)) {
        err = LwSciError_ReconciliationFailed;
        LWSCI_ERR_STR("One or both of the keys to be reconciled are not set\n");
        LWSCI_ERR_UINT("key1: . ", key1);
        LWSCI_ERR_ULONG("Len of key1: \n", len1);
        LWSCI_ERR_UINT("key2: . ", key2);
        LWSCI_ERR_ULONG("Len of key2: \n", len2);
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Directive, 4_9), "LwSciBuf-ADV-MISRAC2012-008")
    len = LW_SCI_BUF_MIN_NUM(len1, len2);

    if (LwSciCommonMemcmp(value1, value2, len) != 0) {
        err = LwSciError_ReconciliationFailed;
        LWSCI_ERR_UINT("Value mismatch for values of keys ", key1);
        LWSCI_ERR_UINT("& \n", key2);
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 1_2), "LwSciBuf-ADV-MISRAC2012-001")
    LWSCI_INFO("Output: Policy: %s: Reconciliation successful for keys %u & %u\n",
        __FUNCTION__, key1, key2);
    LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 1_2))

ret:
    LWSCI_FNEXIT("");
    return err;
}

static LwSciError LwSciBufAttrListMergeAttrKey(
    uint32_t key,
    LwSciBufAttrList appendList,
    LwSciBufAttrList reconciledList,
    const LwSciBufPolicySameKey policyMap[])
{
    LwSciError err = LwSciError_Success;

    size_t slotCount = LwSciBufAttrListGetSlotCount(appendList);
    LwSciBuf_ReconcilePolicy policy = LwSciBuf_IlwalidPolicy;

    LwSciBufIpcRouteAffinity localPeerAffinity = LwSciBufIpcRoute_Max;
    LwSciBufIpcRouteAffinity remotePeerAffinity = LwSciBufIpcRoute_Max;

    /* Identify if the function is called from reconciliation or
     * reconciliation validation.
     */
    const bool isReconciliation = (recPolicyMapSameKey == policyMap);

    LwSciBufAttrKeyGetPolicy(key, &policy);

    LwSciBufAttrKeyGetIpcRouteAffinity(key, true, &localPeerAffinity);
    LwSciBufAttrKeyGetIpcRouteAffinity(key, false, &remotePeerAffinity);

    for (size_t slotNum = 0; slotNum < slotCount; slotNum++) {
        LwSciBufPrivateAttrKeyValuePair pvtKeyValPair = {};
        const LwSciBufIpcRoute* ipcRoute = NULL;
        bool match = false;

        void* ipAddr = NULL;
        void* recAddr = NULL;
        LwSciBufAttrStatus* ipStatus = NULL;
        LwSciBufAttrStatus* recStatus = NULL;
        uint64_t* ipSetSize = NULL;
        uint64_t* recSetSize = NULL;

        LwSciBufAttrGetKeyDetail(appendList, slotNum, key, (void**)&ipAddr,
            &ipStatus, &ipSetSize);

        LwSciBufAttrGetKeyDetail(reconciledList, 0, key, (void**)&recAddr,
            &recStatus, &recSetSize);

        if (NULL == ipSetSize) {
            /* When reconciling lists of two different datatypes,
             * input attrlist of one datatype will not have memory allocated
             * for holding values of keys of other datatype. In that case do
             * nothing.
             */
            continue;
        }

        if ((LwSciBufIpcRoute_AffinityNone != remotePeerAffinity) &&
            (true == isReconciliation)) {

            if (0UL != *ipSetSize) {
                err = PushKeyValToIpcTable(reconciledList, slotNum, key,
                        *ipSetSize, ipAddr);
                if (LwSciError_Success != err) {
                    LWSCI_ERR_STR("PushKeyValToIpcTable() failed.");
                    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
                        "LwSciBuf-ADV-MISRAC2012-015")
                    goto ret;
                }
            }
        }

        pvtKeyValPair.key = LwSciBufPrivateAttrKey_SciIpcRoute;
        err = LwSciBufAttrListCommonGetAttrs(appendList, slotNum,
                &pvtKeyValPair, 1, LwSciBufAttrKeyType_Private, true);
        if (LwSciError_Success != err) {
            LwSciCommonPanic();
        }

        if (0UL == pvtKeyValPair.len) {
            LwSciCommonPanic();
        }

        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 11_5),
            "LwSciBuf-ADV-MISRAC2012-014")
        ipcRoute = *(const LwSciBufIpcRoute* const *)pvtKeyValPair.value;
        LwSciBufIpcRouteMatchAffinity(ipcRoute, localPeerAffinity, 0UL,
            true, &match);

        if (true == match) {
            LwSciBufReconcilePolicyCookie cookie = {};
            const LwSciBufReconcilePolicyCookie* tmpCookie = NULL;

            LWSCI_INFO("Applying Key 0x%x slot %lu, policy %d\n", key,
                slotNum, policy);
            if (LwSciBuf_IlwalidPolicy != policy) {
                if (GpuCacheAndPolicySameKey ==
                    policyMap[(uint32_t)policy]) {
                    cookie.appendList = appendList;
                    cookie.reconciledList = reconciledList;
                    tmpCookie = &cookie;
                } else {
                    tmpCookie = NULL;
                }

                err = policyMap[(uint32_t)policy](key, ipAddr, ipStatus,
                        ipSetSize, recAddr, recStatus, recSetSize,
                        tmpCookie);
                if (LwSciError_Success != err) {
#if (LW_IS_SAFETY == 0)
                    *ipStatus = LwSciBufAttrStatus_Conflict;
                    if (true == isReconciliation) {
                        LwSciError errFail = LwSciError_Success;
                        LwSciBufPrivateAttrKeyValuePair pvtKeyValPair;

                        pvtKeyValPair.key =
                            LwSciBufPrivateAttrKey_ConflictKey;
                        pvtKeyValPair.len = sizeof(uint32_t);
                        pvtKeyValPair.value = &key;

                        errFail = LwSciBufAttrListCommonSetAttrs(appendList,
                                    slotNum, &pvtKeyValPair, 1,
                                    LwSciBufAttrKeyType_Private, true,
                                    false);
                        if (LwSciError_Success != errFail) {
                            LWSCI_ERR_STR("Failed to set conflict status");
                        }
                    }
#endif
                    LWSCI_ERR_UINT("Failed to merge attr key: ", key);
                    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
                        "LwSciBuf-ADV-MISRAC2012-015")
                    goto ret;
                }
            }
        }

        if (0UL != *recSetSize) {
            *recStatus = LwSciBufAttrStatus_Reconciled;
        }
    }

ret:
    return err;
}

static LwSciError LwSciBufAttrListMergeAttrHelper(
    LwSciBufAttrKeyIterator* iter,
    LwSciBufAttrList appendList,
    LwSciBufAttrList reconciledList,
    const LwSciBufPolicySameKey policyMap[])
{
    LwSciError err = LwSciError_Success;

    LWSCI_FNENTRY("");

    LWSCI_INFO("Input: iter: %p, appendList: %p, reconciledList: %p, policyMap: %p",
        iter, appendList, reconciledList, policyMap);

    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_4), "LwSciBuf-ADV-MISRAC2012-016")
    while (true) {
        uint32_t key = 0U;
        bool keyEnd = false;
        bool datatypeEnd = false;
        bool keytypeEnd = false;

        LwSciBufAttrKeyIterNext(iter, &keytypeEnd, &datatypeEnd,
            &keyEnd, &key);

        if (true == keyEnd) {
            break;
        }

        err = LwSciBufAttrListMergeAttrKey(key, appendList, reconciledList, policyMap);
        if (LwSciError_Success != err) {
            LWSCI_ERR_UINT("Failed to merge decode", key);
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
            goto ret;
        }
    }

ret:
    LWSCI_FNEXIT("");
    return err;
}

static LwSciError LwSciBufAttrListCheckAttrHelper(
    LwSciBufAttrKeyIterator* iter,
    LwSciBufAttrList reconciledList)
{
    LwSciError err = LwSciError_Success;

    LWSCI_FNENTRY("");

    while (true) {
        uint32_t key = 0U;
        bool keyEnd = false;
        bool datatypeEnd = false;
        bool keytypeEnd = false;
        bool needsCheck = false;

        LwSciBufAttrKeyIterNext(iter, &keytypeEnd, &datatypeEnd,
            &keyEnd, &key);

        if (true == keyEnd) {
            break;
        }

        err = LwSciBufReconcileCheckingNeeded(reconciledList, key, &needsCheck);
        if (LwSciError_Success != err) {
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
            goto ret;
        }

        if (needsCheck) {
            uint32_t decodedKey = 0U;
            uint32_t decodedDataType = 0U;
            uint32_t decodedKeyType = 0U;

            // Check that the key is set
            LwSciBufAttrKeyValuePair keyValPair;
            LwSciBufAttrKeyType keyType = LwSciBufAttrKeyType_MaxValid;

            err = LwSciBufAttrKeyDecode(key, &decodedKey, &decodedDataType, &decodedKeyType);
            if (LwSciError_Success != err) {
                LWSCI_ERR_UINT("Failed to get decode key", key);
                LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
                goto ret;
            }

            (void)memset(&keyValPair, 0x0, sizeof(keyValPair));
            LwSciCommonMemcpyS(&keyValPair.key, sizeof(keyValPair.key),
                                &key, sizeof(key));
            LwSciCommonMemcpyS(&keyType, sizeof(keyType),
                                &decodedKeyType, sizeof(decodedKeyType));
            err = LwSciBufAttrListCommonGetAttrs(reconciledList, 0, &keyValPair, 1,
                    keyType, true);
            if (LwSciError_Success != err) {
                LWSCI_ERR_HEXUINT("Failed to get set key: ", key);
                LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
                goto ret;
            }
            if (0U == keyValPair.len) {
                LWSCI_ERR_HEXUINT("Required key was not set: ", key);
                err = LwSciError_ReconciliationFailed;
                LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
                goto ret;
            }
        }
    }

ret:
    LWSCI_FNEXIT("");
    return err;
}

static LwSciError LwSciBufAttrListMergeGenAttr(
    LwSciBufAttrList appendList,
    LwSciBufAttrList reconciledList,
    const LwSciBufPolicySameKey policyMap[])
{
    LwSciError err = LwSciError_Success;
    LwSciBufAttrKeyIterator iter;

    LWSCI_FNENTRY("");

    LwSciBufAttrKeyIterInit((uint32_t)LwSciBufAttrKeyType_Public,
        (uint32_t)LwSciBufType_General, 1U, &iter);

    err = LwSciBufAttrListMergeAttrHelper(&iter, appendList, reconciledList,
            policyMap);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("LwSciBufAttrListMergeAttrHelper Failed\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    LwSciBufAttrKeyIterInit((uint32_t)LwSciBufAttrKeyType_Internal,
        (uint32_t)LwSciBufType_General, 1U, &iter);

    err = LwSciBufAttrListMergeAttrHelper(&iter, appendList, reconciledList,
            policyMap);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("LwSciBufAttrListMergeAttrHelper Failed\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    LwSciBufAttrKeyIterInit((uint32_t)LwSciBufAttrKeyType_Private,
        (uint32_t)LwSciBufType_General, 1, &iter);

    err = LwSciBufAttrListMergeAttrHelper(&iter, appendList, reconciledList,
            policyMap);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("LwSciBufAttrListMergeAttrHelper Failed\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    /* if this function is called via Reconcile API we need to allocate memory
     *  for datatype attribute keys.
     * If we are called via VerifyReconciled API we dont need to allocate memory
     */
    if (recPolicyMapSameKey == policyMap) {
        err = LwSciBufAttrListMallocBufferType(reconciledList, 0);
        if (LwSciError_Success != err) {
            LWSCI_ERR_STR("Failed to allocate buffer type in reconciled list\n");
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
            goto ret;
        }
    }

ret:
    LWSCI_FNEXIT("");
    return err;
}

static LwSciError LwSciBufAttrListMergeUmdAttr(
    LwSciBufAttrList appendList,
    LwSciBufAttrList reconciledList,
    const LwSciBufPolicySameKey policyMap[])
{
    LwSciError err = LwSciError_Success;
    uint32_t key = 0U;
    uint64_t slotNum = 0U;
    LwSciBuf_ReconcilePolicy policy = LwSciBuf_IlwalidPolicy;
    size_t slotCount = 0U;
    void* ipAddr = NULL;
    void* recAddr = NULL;
    LwSciBufAttrStatus* ipStatus = NULL;
    LwSciBufAttrStatus* recStatus = NULL;
    uint64_t* ipSetSize = NULL;
    uint64_t* recSetSize = NULL;

    LWSCI_FNENTRY("");

    key = (uint32_t)LwSciBufInternalAttrKey_LwMediaPrivateFirst;
    LwSciBufAttrKeyGetPolicy(key, &policy);

    slotCount = LwSciBufAttrListGetSlotCount(appendList);

    for (slotNum = 0; slotNum < slotCount; slotNum++) {
        LwSciBufAttrGetKeyDetail(appendList, slotNum, key, (void**)&ipAddr,
        &ipStatus, &ipSetSize);

        LwSciBufAttrGetKeyDetail(reconciledList, 0, key, (void**)&recAddr,
        &recStatus, &recSetSize);

        LWSCI_INFO("Applying Key %u slot %lu, policy %d\n", key, slotNum,
            policy);
        if (LwSciBuf_IlwalidPolicy != policy) {
            err = policyMap[(uint32_t)policy](key, ipAddr, ipStatus,
                    ipSetSize, recAddr, recStatus, recSetSize, NULL);
            if (LwSciError_Success != err) {
#if (LW_IS_SAFETY == 0)
                if (recPolicyMapSameKey == policyMap) {
                    LwSciError errFail = LwSciError_Success;
                    LwSciBufPrivateAttrKeyValuePair pvtKeyValPair;

                    pvtKeyValPair.key = LwSciBufPrivateAttrKey_ConflictKey;
                    pvtKeyValPair.len = sizeof(uint32_t);
                    pvtKeyValPair.value = &key;

                    errFail = LwSciBufAttrListCommonSetAttrs(appendList,
                                slotNum, &pvtKeyValPair, 1,
                                LwSciBufAttrKeyType_Private, true, false);
                    if (LwSciError_Success != errFail) {
                        LWSCI_ERR_STR("Failed to set conflict status\n");
                    }
                }
#endif
                LWSCI_ERR_STR("Failed to merge Umd Attr Keys\n");
                LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
                goto ret;
            }
        }
    }

    *recStatus = LwSciBufAttrStatus_Reconciled;

ret:
    LWSCI_FNEXIT("");
    return err;
}

static LwSciError LwSciBufAttrListMergeRawAttr(
    LwSciBufAttrList appendList,
    LwSciBufAttrList reconciledList,
    const LwSciBufPolicySameKey policyMap[])
{
    LwSciError err = LwSciError_Success;
    LwSciBufAttrKeyIterator iter;

    LWSCI_FNENTRY("");

    LwSciBufAttrKeyIterInit((uint32_t)LwSciBufAttrKeyType_Public,
        (uint32_t)LwSciBufType_RawBuffer, 0, &iter);

    err = LwSciBufAttrListMergeAttrHelper(&iter, appendList, reconciledList,
            policyMap);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("LwSciBufAttrListMergeAttrHelper Failed\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

ret:
    LWSCI_FNEXIT("");
    return err;
}

static LwSciError LwSciBufAttrListMergeTensorAttr(
    LwSciBufAttrList appendList,
    LwSciBufAttrList reconciledList,
    const LwSciBufPolicySameKey policyMap[])
{
    LwSciError err = LwSciError_Success;
    LwSciBufAttrKeyIterator iter;

    LWSCI_FNENTRY("");

    LwSciBufAttrKeyIterInit((uint32_t)LwSciBufAttrKeyType_Public,
        (uint32_t)LwSciBufType_Tensor, 0U, &iter);

    err = LwSciBufAttrListMergeAttrHelper(&iter, appendList, reconciledList,
            policyMap);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("LwSciBufAttrListMergeAttrHelper Failed\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

ret:
    LWSCI_FNEXIT("");
    return err;
}

static LwSciError LwSciBufAttrListMergeArrAttr(
    LwSciBufAttrList appendList,
    LwSciBufAttrList reconciledList,
    const LwSciBufPolicySameKey policyMap[])
{
    LwSciError err = LwSciError_Success;
    LwSciBufAttrKeyIterator iter;

    LWSCI_FNENTRY("");

    LwSciBufAttrKeyIterInit((uint32_t)LwSciBufAttrKeyType_Public,
        (uint32_t)LwSciBufType_Array, 0U, &iter);

    err = LwSciBufAttrListMergeAttrHelper(&iter, appendList, reconciledList,
            policyMap);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("LwSciBufAttrListMergeAttrHelper Failed\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

ret:
    LWSCI_FNEXIT("");
    return err;
}

static LwSciError LwSciBufAttrListMergeImgAttr(
    LwSciBufAttrList appendList,
    LwSciBufAttrList reconciledList,
    const LwSciBufPolicySameKey policyMap[])
{
    LwSciError err = LwSciError_Success;
    LwSciBufAttrKeyIterator iter;

    LWSCI_FNENTRY("");

    LwSciBufAttrKeyIterInit((uint32_t)LwSciBufAttrKeyType_Public,
        (uint32_t)LwSciBufType_Image, 0, &iter);

    err = LwSciBufAttrListMergeAttrHelper(&iter, appendList, reconciledList,
            policyMap);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("LwSciBufAttrListMergeAttrHelper Failed\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    LwSciBufAttrKeyIterInit((uint32_t)LwSciBufAttrKeyType_Internal,
        (uint32_t)LwSciBufType_Image, 0, &iter);

    err = LwSciBufAttrListMergeAttrHelper(&iter, appendList, reconciledList,
            policyMap);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("LwSciBufAttrListMergeAttrHelper Failed\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

ret:
    LWSCI_FNEXIT("");
    return err;
}

static LwSciError LwSciBufAttrListMergePyrAttr(
    LwSciBufAttrList appendList,
    LwSciBufAttrList reconciledList,
    const LwSciBufPolicySameKey policyMap[])
{
    LwSciError err = LwSciError_Success;
    LwSciBufAttrKeyIterator iter;

    LWSCI_FNENTRY("");

    /* Merge all Image attributes */
    err = LwSciBufAttrListMergeImgAttr(appendList, reconciledList, policyMap);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("Failed to get object from LwSciBufAttrList reference\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    /* Merge all Pyramid attributes */

    LwSciBufAttrKeyIterInit((uint32_t)LwSciBufAttrKeyType_Public,
        (uint32_t)LwSciBufType_Pyramid, 0, &iter);

    err = LwSciBufAttrListMergeAttrHelper(&iter, appendList, reconciledList,
            policyMap);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("LwSciBufAttrListMergeAttrHelper Failed\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

ret:
    LWSCI_FNEXIT("");
    return err;
}

static LwSciError LwSciBufAttrListCheckRawAttr(
    LwSciBufAttrList reconciledList)
{
    LwSciError err = LwSciError_Success;
    LwSciBufAttrKeyIterator iter;

    LWSCI_FNENTRY("");

    LwSciBufAttrKeyIterInit((uint32_t)LwSciBufAttrKeyType_Public,
        (uint32_t)LwSciBufType_RawBuffer, 0, &iter);

    err = LwSciBufAttrListCheckAttrHelper(&iter, reconciledList);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("LwSciBufAttrListCheckAttrHelper Failed");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

ret:
    LWSCI_FNEXIT("");
    return err;
}

static LwSciError LwSciBufAttrListCheckTensorAttr(
    LwSciBufAttrList reconciledList)
{
    LwSciError err = LwSciError_Success;
    LwSciBufAttrKeyIterator iter;

    LWSCI_FNENTRY("");

    LwSciBufAttrKeyIterInit((uint32_t)LwSciBufAttrKeyType_Public,
        (uint32_t)LwSciBufType_Tensor, 0U, &iter);

    err = LwSciBufAttrListCheckAttrHelper(&iter, reconciledList);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("LwSciBufAttrListCheckAttrHelper Failed");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

ret:
    LWSCI_FNEXIT("");
    return err;
}

static LwSciError LwSciBufAttrListCheckArrAttr(
    LwSciBufAttrList reconciledList)
{
    LwSciError err = LwSciError_Success;
    LwSciBufAttrKeyIterator iter;

    LWSCI_FNENTRY("");

    LwSciBufAttrKeyIterInit((uint32_t)LwSciBufAttrKeyType_Public,
        (uint32_t)LwSciBufType_Array, 0U, &iter);

    err = LwSciBufAttrListCheckAttrHelper(&iter, reconciledList);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("LwSciBufAttrListCheckAttrHelper Failed");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

ret:
    LWSCI_FNEXIT("");
    return err;
}

static LwSciError LwSciBufAttrListCheckImgAttr(
    LwSciBufAttrList reconciledList)
{
    LwSciError err = LwSciError_Success;
    LwSciBufAttrKeyIterator iter;

    LWSCI_FNENTRY("");

    LwSciBufAttrKeyIterInit((uint32_t)LwSciBufAttrKeyType_Public,
        (uint32_t)LwSciBufType_Image, 0, &iter);

    err = LwSciBufAttrListCheckAttrHelper(&iter, reconciledList);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("LwSciBufAttrListCheckAttrHelper Failed");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    LwSciBufAttrKeyIterInit((uint32_t)LwSciBufAttrKeyType_Internal,
        (uint32_t)LwSciBufType_Image, 0, &iter);

    err = LwSciBufAttrListCheckAttrHelper(&iter, reconciledList);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("LwSciBufAttrListCheckAttrHelper Failed");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

ret:
    LWSCI_FNEXIT("");
    return err;
}

static LwSciError LwSciBufAttrListCheckPyrAttr(
    LwSciBufAttrList reconciledList)
{
    LwSciError err = LwSciError_Success;
    LwSciBufAttrKeyIterator iter;

    LWSCI_FNENTRY("");

    /* Check all Image attributes */
    err = LwSciBufAttrListCheckImgAttr(reconciledList);
    if (LwSciError_Success != err) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    /* Check all Pyramid attributes */

    LwSciBufAttrKeyIterInit((uint32_t)LwSciBufAttrKeyType_Public,
        (uint32_t)LwSciBufType_Pyramid, 0, &iter);

    err = LwSciBufAttrListCheckAttrHelper(&iter, reconciledList);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("LwSciBufAttrListCheckAttrHelper Failed");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

ret:
    LWSCI_FNEXIT("");
    return err;
}

static LwSciError LwSciBufAttrListMergeDataType(
    LwSciBufAttrList appendList,
    LwSciBufAttrList reconciledList,
    const LwSciBufType* dataType,
    size_t len,
    const LwSciBufPolicySameKey policyMap[])
{
    LwSciError err = LwSciError_Success;
    uint64_t count = 0UL, index = 0UL;
    LwSciBufMergeBufferType mergeBufferTypeFuncPtr = NULL;

    /* map of Merget*Attr() function is used to merge different buffer type keys
     * according to the merge policies specified in attrdesc table.
     * NOTE: This mapping cannot be spare and all indexes must have
     * corresponding Merge*Attr() function associated.
     */
    static const LwSciBufMergeBufferType
        mergeBufferTypeMap[LwSciBufType_MaxValid] = {
        [LwSciBufType_RawBuffer] = LwSciBufAttrListMergeRawAttr,
        [LwSciBufType_Image] = LwSciBufAttrListMergeImgAttr,
        [LwSciBufType_Tensor] = LwSciBufAttrListMergeTensorAttr,
        [LwSciBufType_Array] = LwSciBufAttrListMergeArrAttr,
        [LwSciBufType_Pyramid] = LwSciBufAttrListMergePyrAttr,
    };

    LWSCI_FNENTRY("");

    if (0UL == len) {
        LWSCI_ERR_STR("Bad parameter supplied to LwSciBufAttrListMergeDataType\n");
        LWSCI_ERR_ULONG("len: \n", len);
        LwSciCommonPanic();
    }

    count = len / sizeof(LwSciBufType);

    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_4), "LwSciBuf-ADV-MISRAC2012-016")
    for (index = 0; index < count; index++) {
        if (LwSciBufType_MaxValid <= dataType[index]) {
            LwSciCommonPanic();
        }

        mergeBufferTypeFuncPtr = mergeBufferTypeMap[dataType[index]];
        if (NULL == mergeBufferTypeFuncPtr) {
            err = LwSciError_IlwalidState;
            LWSCI_ERR_UINT("Empty merge mapping function for buffer type \n",
                (uint32_t)dataType[index]);
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
            goto ret;
        }

        err = mergeBufferTypeFuncPtr(appendList, reconciledList, policyMap);
        if (LwSciError_Success != err) {
            LWSCI_ERR_STR("Failed to merge data type attributes\n");
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
            goto ret;
        }
    }

ret:
    LWSCI_FNEXIT("");
    return err;
}

static LwSciError LwSciBufAttrListCheckDataType(
    LwSciBufAttrList reconciledList,
    const LwSciBufType* dataType,
    size_t len)
{
    LwSciError err = LwSciError_Success;
    uint64_t count = 0UL;

    /* map of Check*Attr() function is used to merge different buffer type keys
     * according to the merge policies specified in attrdesc table.
     * NOTE: This mapping cannot be spare and all indexes must have
     * corresponding Check*Attr() function associated.
     */
    static const LwSciBufCheckBufferType
        checkBufferTypeMap[LwSciBufType_MaxValid] = {
        [LwSciBufType_RawBuffer] = LwSciBufAttrListCheckRawAttr,
        [LwSciBufType_Image] = LwSciBufAttrListCheckImgAttr,
        [LwSciBufType_Tensor] = LwSciBufAttrListCheckTensorAttr,
        [LwSciBufType_Array] = LwSciBufAttrListCheckArrAttr,
        [LwSciBufType_Pyramid] = LwSciBufAttrListCheckPyrAttr,
    };

    LWSCI_FNENTRY("");

    if (0UL == len) {
        LWSCI_ERR_STR("Bad parameter supplied to LwSciBufAttrListCheckDataType");
        LWSCI_ERR_ULONG("len: ", len);
        LwSciCommonPanic();
    }

    count = len / sizeof(LwSciBufType);

    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_4), "LwSciBuf-ADV-MISRAC2012-016")
    for (uint64_t index = 0; index < count; index++) {
        LwSciBufCheckBufferType checkBufferTypeFuncPtr = NULL;
        if (LwSciBufType_MaxValid <= dataType[index]) {
            LwSciCommonPanic();
        }

        checkBufferTypeFuncPtr = checkBufferTypeMap[dataType[index]];
        if (NULL == checkBufferTypeFuncPtr) {
            LWSCI_ERR_UINT("Empty check mapping function for buffer type ",
                    (uint32_t)dataType[index]);
            LwSciCommonPanic();
        }

        err = checkBufferTypeFuncPtr(reconciledList);
        if (LwSciError_Success != err) {
            LWSCI_ERR_STR("Failed to check data type attributes");
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
            goto ret;
        }
    }

ret:
    LWSCI_FNEXIT("");
    return err;
}

static LwSciError LwSciBufAttrListMergeSameKeys(
    LwSciBufAttrList appendList,
    LwSciBufAttrList reconciledList,
    const LwSciBufPolicySameKey policyMap[])
{
    LwSciError err = LwSciError_Success;
    LwSciBufAttrKeyValuePair dataTypePair;
    const LwSciBufType* bufType = NULL;
    size_t bufLen = 0;
    LWSCI_FNENTRY("");

    (void)memset(&dataTypePair, 0x0, sizeof(dataTypePair));

    err = LwSciBufAttrListMergeGenAttr(appendList, reconciledList, policyMap);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("Failed to merge Gen Attr Keys\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    err = LwSciBufAttrListMergeUmdAttr(appendList, reconciledList, policyMap);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("Failed to merge Umd Attr Keys\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    dataTypePair.key = LwSciBufGeneralAttrKey_Types;
    err = LwSciBufAttrListCommonGetAttrs(reconciledList, 0, &dataTypePair, 1,
            LwSciBufAttrKeyType_Public, true);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("Failed to get buffer data type\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }
    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
    bufType = dataTypePair.value;
    bufLen = dataTypePair.len;

    LWSCI_INFO("Reconciled type is %d\n", *bufType);
    err = LwSciBufAttrListMergeDataType(appendList, reconciledList, bufType,
            bufLen, policyMap);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("Failed to merge data type\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

ret:
    LWSCI_FNEXIT("");
    return err;
}

static LwSciError LwSciBufAttrListCheckRequiredSameKeys(
    LwSciBufAttrList reconciledList)
{
    LwSciError err = LwSciError_Success;

    LwSciBufAttrKeyIterator iter = { 0 };

    LwSciBufAttrKeyValuePair dataTypePair = { 0 };
    const LwSciBufType* bufType = NULL;
    size_t bufLen = 0U;

    LWSCI_FNENTRY("");

    LwSciBufAttrKeyIterInit((uint32_t)LwSciBufAttrKeyType_Public,
        (uint32_t)LwSciBufType_General, 1U, &iter);
    err = LwSciBufAttrListCheckAttrHelper(&iter, reconciledList);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("LwSciBufAttrListCheckAttrHelper failed");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    LwSciBufAttrKeyIterInit((uint32_t)LwSciBufAttrKeyType_Internal,
        (uint32_t)LwSciBufType_General, 1U, &iter);
    err = LwSciBufAttrListCheckAttrHelper(&iter, reconciledList);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("LwSciBufAttrListCheckAttrHelper failed");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    LwSciBufAttrKeyIterInit((uint32_t)LwSciBufAttrKeyType_Private,
        (uint32_t)LwSciBufType_General, 1, &iter);
    err = LwSciBufAttrListCheckAttrHelper(&iter, reconciledList);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("LwSciBufAttrListCheckAttrHelper failed");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    dataTypePair.key = LwSciBufGeneralAttrKey_Types;
    err = LwSciBufAttrListCommonGetAttrs(reconciledList, 0, &dataTypePair, 1,
            LwSciBufAttrKeyType_Public, true);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("Failed to get buffer data type");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }
    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
    bufType = dataTypePair.value;
    bufLen = dataTypePair.len;

    LWSCI_INFO("Reconciled type is %d\n", *bufType);
    err = LwSciBufAttrListCheckDataType(reconciledList, bufType, bufLen);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("Failed to merge data type");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

ret:
    LWSCI_FNEXIT("");
    return err;
}

static LwSciError LwSciBufAttrListAreDataTypeCompatible(
    LwSciBufAttrList attrList,
    bool* isCompatible)
{
    LwSciError err = LwSciError_Success;
    size_t numArray = 0, index1 = 0, index2 = 0;
    const LwSciBufType* bufType = NULL;
    LwSciBufType bufType1 = LwSciBufType_General,
                 bufType2 = LwSciBufType_General;

    /* Map of compatible buffer types for reconciliation
     * NOTE: Only add heterogenous datatypes here. Same datatype is always
     * considered to be reconciliable with itself
     */
    LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 9_3), "LwSciBuf-REQ-MISRAC2012-002")
    static const bool
        dataTypeCompatible[LwSciBufType_MaxValid][LwSciBufType_MaxValid] = {
        [LwSciBufType_Image][LwSciBufType_Tensor] = (bool)true,
        [LwSciBufType_Tensor][LwSciBufType_Image] = (bool)true,
    };
    LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 9_3))

    LWSCI_FNENTRY("");

    LWSCI_INFO("Input: attrList ptr: %p, isCompatible ptr: %p\n", attrList,
            isCompatible);

    err = LwSciBufAttrListGetDataTypes(attrList, &bufType, &numArray);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("LwSciBufAttrListGetDataTypes failed\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    if (1UL == numArray) {
        /* there is only one buffer type */
        *isCompatible = true;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto out;
    }

    /* More than one buffer are ilwolved. Make sure that buffer types used for
     * interop are compatible
     */
    for (index1 = 0UL; index1 < numArray; index1++) {
        for (index2 = index1 + 1UL; index2 < numArray; index2++) {
            bufType1 = bufType[index1];
            bufType2 = bufType[index2];

            if ((LwSciBufType_MaxValid <= bufType1) ||
                (LwSciBufType_MaxValid <= bufType2)) {
                LWSCI_ERR_STR("Invalid state\n");
                LwSciCommonPanic();
            }

            if (false == dataTypeCompatible[(uint32_t)bufType1][(uint32_t)bufType2]) {
                LWSCI_ERR_UINT("bufTypes ", (uint32_t)bufType1);
                LWSCI_ERR_UINT("and ", (uint32_t)bufType2);
                LWSCI_ERR_STR("cannot be reconciled\n");
                break;
            }
        }
    }

    *isCompatible = dataTypeCompatible[(uint32_t)bufType1][(uint32_t)bufType2];

out:
    LWSCI_INFO("Output: isCompatible: %s\n", *isCompatible ? "true": "false");

ret:
    LWSCI_FNEXIT("");
    return err;
}

static LwSciError LwSciBufAttrListValidateDiffKeyDependency(
    LwSciBufAttrList attrList)
{
    /*
     * Validate interdependency of keys set for different datatypes
     */
    LwSciError err = LwSciError_Success;
    size_t numArray = 0, index1 = 0, index2 = 0;
    const LwSciBufType* bufType = NULL;
    LwSciBufType bufType1 = LwSciBufType_MaxValid,
        bufType2 = LwSciBufType_MaxValid;
    LwSciBufAttrListKeyDependency keyDependencyFnPtr = NULL;

    LWSCI_FNENTRY("");

    LWSCI_INFO("Input: attrList ptr: %p\n", attrList);

    err = LwSciBufAttrListGetDataTypes(attrList, &bufType, &numArray);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("LwSciBufAttrListGetDataTypes failed\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    /* Figure out interdependency of keys for different datatypes */
    for (index1 = 0; index1 < numArray; index1++) {
        for (index2 = index1 + 1U; index2 < numArray; index2++) {
            bufType1 = bufType[index1];
            bufType2 = bufType[index2];

            if ((LwSciBufType_MaxValid <= bufType1) || (LwSciBufType_MaxValid <= bufType2)) {
                LWSCI_ERR_UINT("bufType1: , ", (uint32_t)bufType1);
                LWSCI_ERR_UINT("bufType2: \n", (uint32_t)bufType2);
                LwSciCommonPanic();
            }

            keyDependencyFnPtr = keyDependency[(uint32_t)bufType1][(uint32_t)bufType2];
            if (NULL != keyDependencyFnPtr) {
                err = keyDependencyFnPtr(attrList);
                if (LwSciError_Success != err) {
                    LWSCI_ERR_UINT("key dependencies failed for bufType1  ",bufType1);
                    LWSCI_ERR_UINT("and bufType2 \n", (uint32_t)bufType2);
                    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
                    goto ret;
                }
            }
        }
    }

ret:
    LWSCI_FNEXIT("");
    return err;
}

static LwSciError LwSciBufAttrListValidateKeyDependency(
    LwSciBufAttrList attrList)
{
    /* Validate interdependency of keys set for same datatype as well as
     * interdependency of different keys of different datatypes
     */
    LwSciError err = LwSciError_Success;
    size_t numArray = 0, index1 = 0;
    const LwSciBufType* bufType = NULL;
    LwSciBufType bufType1 = LwSciBufType_MaxValid;
    LwSciBufAttrListKeyDependency keyDependencyFnPtr = NULL;

    LWSCI_FNENTRY("");

    LWSCI_INFO("Input: attrList ptr: %p\n", attrList);

    err = LwSciBufAttrListGetDataTypes(attrList, &bufType, &numArray);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("LwSciBufAttrListGetDataTypes failed\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    /* Figure out interdependency of keys for same datatype */
    for (index1 = 0; index1 < numArray; index1++) {
        bufType1 = bufType[index1];

        if (LwSciBufType_MaxValid <= bufType1) {
            LWSCI_ERR_UINT("bufType is invalid: \n", bufType1);
            LWSCI_ERR_UINT("bufType obtained: \n", bufType1);
            LwSciCommonPanic();
        }

        keyDependencyFnPtr = keyDependency[(uint32_t)bufType1][(uint32_t)bufType1];
        if (NULL != keyDependencyFnPtr) {
            err = keyDependencyFnPtr(attrList);
            if (LwSciError_Success != err) {
                LWSCI_ERR_UINT("key dependency failed for bufType \n",
                    bufType1);
                LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
                goto ret;
            }
        }
    }

    err = LwSciBufAttrListValidateDiffKeyDependency(attrList);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("Failed to validate different key dependencies \n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

ret:
    LWSCI_FNEXIT("");
    return err;
}

static LwSciError LwSciBufAttrListCompareDifferentKeys(
    LwSciBufAttrList reconciledList,
    const LwSciBufPolicyDifferentKey policyMap[])
{
    LwSciError err = LwSciError_Success;
    const LwSciBufType* bufType = NULL;
    LwSciBufType bufType1 = LwSciBufType_MaxValid;
    LwSciBufType bufType2 = LwSciBufType_MaxValid;
    const LwSciBufAttrListRecKeyPair* recKeyPair = NULL;
    LwSciBuf_ReconcilePolicy policy =  LwSciBuf_IlwalidPolicy;
    size_t numDataTypes = 0, numRecKeyPair = 0;
    uint32_t index1 = 0U, index2 = 0U, index3 = 0U;
    uint32_t key1 = 0U, key2 = 0U;

    /* map of function pointers returning pair of keys belonging to different
     * that need to be reconciled.
     * NOTE: Only function pointers for heteorgenous datatype pairs should be listed
     * here. This mapping is not needed for reconciliation of same datatypes
     */
    LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 9_3), "LwSciBuf-REQ-MISRAC2012-002")
    static const LwSciBufAttrListGetRecKeyPair
            recKeyPairFnPtr[LwSciBufType_MaxValid][LwSciBufType_MaxValid] = {
        [LwSciBufType_Image][LwSciBufType_Tensor] =
            LwSciBufAttrListGetImageTensorRecKeyPair,

        [LwSciBufType_Tensor][LwSciBufType_Image] =
            LwSciBufAttrListGetImageTensorRecKeyPair,
    };

    /* map of function pointers used to reconcile different datatypes for which
     * custom checks are required which cannot be satisified using one of the
     * key reconciliation policies
     * NOTE: Only function pointers for heterogenous datatype pairs should be listed
     * here. This mapping is not needed for reconciliation of same datatypes since
     * same datatypes can be reconciled using reconciliation policies
     */
    static const LwSciBufAttrListLwstomCompare
        lwstomComparisonFnPtr[LwSciBufType_MaxValid][LwSciBufType_MaxValid] = {
        [LwSciBufType_Image][LwSciBufType_Tensor] =
            LwSciBufAttrListLwstomCompareImageTensor,

        [LwSciBufType_Tensor][LwSciBufType_Image] =
            LwSciBufAttrListLwstomCompareImageTensor,
    };
    LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 9_3))


    LWSCI_FNENTRY("");

    LWSCI_INFO("Input: reconciledList %p, policyMap %p\n", reconciledList,
        policyMap);

    /* obtain datatypes which need to be reconciled  */
    err = LwSciBufAttrListGetDataTypes(reconciledList, &bufType, &numDataTypes);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("LwSciBufAttrListGetDataTypes failed\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    if (1UL == numDataTypes) {
        /* only one lwscibuf datatype ilwolved in interop. Do nothing */
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_4), "LwSciBuf-ADV-MISRAC2012-016")
    for (index1 = 0U; index1 < numDataTypes; index1++) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_4), "LwSciBuf-ADV-MISRAC2012-016")
        for (index2 = index1 + 1U; index2 < numDataTypes; index2++) {
            bufType1 = bufType[index1];
            bufType2 = bufType[index2];

            if ((LwSciBufType_MaxValid <= bufType1) || (LwSciBufType_MaxValid <= bufType2)) {
                LWSCI_ERR_UINT("bufType1: , ", bufType1);
                LWSCI_ERR_UINT("bufType2: \n", bufType2);
                LwSciCommonPanic();
            }

            /* for every combination of datatype pair, get recKeyPairs */
            if (NULL != recKeyPairFnPtr[(uint32_t)bufType1][(uint32_t)bufType2]) {
                err = recKeyPairFnPtr[(uint32_t)bufType1][(uint32_t)bufType2](&recKeyPair,
                        &numRecKeyPair);
                if (LwSciError_Success != err) {
                    LWSCI_ERR_UINT("Could not obtain key-pairs for reconciliation between bufTypes  ", bufType1);
                    LWSCI_ERR_UINT("and \n", bufType2);
                    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
                    goto ret;
                }

                /* for every key-pair in recKeyPair array, reconcile keys
                 * according to policy defined in the array
                 */
                for (index3 = 0; index3 < numRecKeyPair; index3++) {
                    policy = recKeyPair[index3].policy;
                    key1 = recKeyPair[index3].key1;
                    key2 = recKeyPair[index3].key2;
                    err = policyMap[(uint32_t)policy](key1, key2,
                            reconciledList);
                    if (LwSciError_Success != err) {
                        LWSCI_ERR_UINT("Could not reconcile key  ", key1);
                        LWSCI_ERR_UINT("of bufType  ", bufType1);
                        LWSCI_ERR_UINT("and key  ", key2);
                        LWSCI_ERR_UINT("of bufType \n", bufType2);
                        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
                        goto ret;
                    }
                }
            }

            /* For every combination of datatype pair, check if custom
             * comparison is needed
             */
            if (NULL != lwstomComparisonFnPtr[(uint32_t)bufType1][(uint32_t)bufType2]) {
                err = lwstomComparisonFnPtr[(uint32_t)bufType1][(uint32_t)bufType2](reconciledList);
                if (LwSciError_Success != err) {
                    LWSCI_ERR_UINT("Could not custom reconcile datatypes  ", (uint32_t)bufType1);
                    LWSCI_ERR_UINT("and \n", (uint32_t)bufType2);
                    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
                    goto ret;
                }
            }
        }
    }

ret:
    LWSCI_FNEXIT("");
    return err;
}

static LwSciError LwSciBufCreateReconciledIpcTable(
    LwSciBufAttrList inputAppendAttrList,
    LwSciBufAttrList newAttrList)
{
    LwSciError err = LwSciError_Success;
    size_t slotCount = 0UL;
    uint64_t index = 0UL;
    LwSciBufIpcTable* newIpcTable = NULL;
    const LwSciBufIpcRoute* ipcRoute = NULL;
    LwSciBufPrivateAttrKeyValuePair keyValPair = {0};

    LWSCI_FNENTRY("");

    slotCount = LwSciBufAttrListGetSlotCount(inputAppendAttrList);

    err = LwSciBufCreateIpcTable(slotCount, &newIpcTable);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("Failed to Create Ipc Table during reconciliation.\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    keyValPair.key = LwSciBufPrivateAttrKey_IPCTable;
    keyValPair.value = &newIpcTable;
    keyValPair.len = sizeof(newIpcTable);

    err = LwSciBufAttrListCommonSetAttrs(newAttrList, 0, &keyValPair, 1,
            LwSciBufAttrKeyType_Private, true, false);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("Failed to set IPC Table key.\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    for (index = 0UL; index < slotCount; index++) {
        keyValPair.key = LwSciBufPrivateAttrKey_SciIpcRoute;

        err = LwSciBufAttrListCommonGetAttrs(inputAppendAttrList, index,
                &keyValPair, 1, LwSciBufAttrKeyType_Private, true);
        if (LwSciError_Success != err) {
            LWSCI_ERR_HEXUINT("Failed to get IPC route for slot index: ",
                index);
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
            "LwSciBuf-ADV-MISRAC2012-015")
            goto ret;
        }

        if (0U == keyValPair.len) {
            /* This should not happen since we at least set a NULL IPC route
             * in the LwSciBufPrivateAttrKey_SciIpcRoute attribute when
             * unreconciled attribute list is created.
             */
            LwSciCommonPanic();
        }

        ipcRoute = *(const LwSciBufIpcRoute* const *)keyValPair.value;

        err = LwSciBufIpcAddRouteToTable(newIpcTable, ipcRoute, index);
        if (LwSciError_Success != err) {
            LWSCI_ERR_STR("LwSciBufIpcAddRouteToTable failed.");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
            "LwSciBuf-ADV-MISRAC2012-015")
            goto ret;
        }
    }

ret:
    LWSCI_FNEXIT("");
    return err;
}

static LwSciError LwSciBufValidateMergedAttrList(
    LwSciBufAttrList mergedAttrList)
{
    bool isValid = false;
    LwSciError err = LwSciError_Success;

    LWSCI_FNENTRY("");

    /* Figure out if different LwSciBuf datatypes ilwolved can be reconciled
     * or not
     */
    err = LwSciBufAttrListAreDataTypeCompatible(mergedAttrList, &isValid);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("LwSciBufAttrListAreDataTypeCompatible failed\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    if (false == isValid) {
        err = LwSciError_ReconciliationFailed;
        LWSCI_ERR_STR("LwSciBuf datatypes cannot be reconciled\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    err = LwSciBufAttrListValidateKeyDependency(mergedAttrList);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("LwSciBufAttrListValidateKeyDependency failed\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

ret:
    LWSCI_FNEXIT("");
    return err;
}

LwSciError LwSciBufAttrListReconcileFromIpcTable(
    LwSciBufAttrList reconciledList,
    uint32_t key,
    LwSciIpcEndpoint ipcEndPoint,
    bool localPeer,
    bool overrideKeyAffinity,
    LwSciBufIpcRouteAffinity routeAffinity)
{
    LwSciError err = LwSciError_Success;
    bool isReconciled = false;
    LwSciBuf_ReconcilePolicy policy = LwSciBuf_IlwalidPolicy;
    struct LwSciIpcEndpointInfo info = {};
    LwSciBufIpcTableIter* ipcTableIter = NULL;
    void* recAddr = NULL;
    LwSciBufAttrStatus* recStatus = NULL;
    uint64_t* recLen = NULL;
    LwSciBufAttrList dummyAppendList = NULL;

    LWSCI_FNENTRY("");

    if (((false == localPeer) && (0UL == ipcEndPoint)) ||
        ((true == overrideKeyAffinity) &&
        (LwSciBufIpcRoute_Max <= routeAffinity))) {
        LWSCI_ERR_STR("Bad parameter supplied to LwSciBufAttrListReconcileFromIpcTable().");
        LwSciCommonPanic();
    }

    if (0UL != ipcEndPoint) {
        err = LwSciIpcGetEndpointInfo(ipcEndPoint, &info);
        if (LwSciError_Success != err) {
            LWSCI_ERR_STR("Failed to validate LwSciIpcEndpoint.");
            err = LwSciError_BadParameter;
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
                "LwSciBuf-ADV-MISRAC2012-015")
            goto ret;
        }
    }

    if (NULL == reconciledList) {
        LWSCI_ERR_STR("NULL reconciledList supplied to LwSciBufAttrListReconcileFromIpcTable().");
        err = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
            "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    err = LwSciBufAttrListValidate(reconciledList);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("LwSciBufAttrListValidate failed.");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
            "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    err = LwSciBufAttrListIsReconciled(reconciledList, &isReconciled);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("LwSciBufAttrListIsReconciled failed.");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
            "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    if (false == isReconciled) {
        LWSCI_ERR_STR("LwSciBufAttrListReconcileFromIpcTable() must be called with reconciled LwSciBufAttrList.");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
            "LwSciBuf-ADV-MISRAC2012-015")
        err = LwSciError_BadParameter;
        goto ret;
    }

    if (false == overrideKeyAffinity) {
        LwSciBufAttrKeyGetIpcRouteAffinity(key, localPeer, &routeAffinity);
    }

    LwSciBufAttrKeyGetPolicy(key, &policy);

    LwSciBufAttrGetKeyDetail(reconciledList, 0, key, &recAddr, &recStatus,
        &recLen);

    if ((false == localPeer) &&
    (LwSciBufIpcRoute_SocAffinity == routeAffinity)) {
        bool isSocBoundary = false;

        err = LwSciBufIsSocBoundary(ipcEndPoint, &isSocBoundary);
        if (LwSciError_Success != err) {
            LWSCI_ERR_STR("LwSciBufIsSocBoundary failed.");
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
                "LwSciBuf-ADV-MISRAC2012-015")
            goto ret;
        }

        if (false == isSocBoundary) {
            /* We are supposed to callwlate the key according to
             * LwSciBufIpcRoute_SocAffinity but we are not crossing the Soc
             * boundary. Thus, there wont be any values in IPC table
             * corresponding to remote peer. If we are not crossing the Soc
             * boundary then reconciled value for local peer and remote
             * peer is the same. As such, skip reconciliation and reuse the
             * value already present in the reconciled list.
             */
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
                "LwSciBuf-ADV-MISRAC2012-015")
            goto ret;
        }

        if ((true == isSocBoundary) &&
            ((LwSciBufGeneralAttrKey_GpuId == key) ||
            (LwSciBufGeneralAttrKey_VidMem_GpuId == key) ||
            (LwSciBufGeneralAttrKey_EnableGpuCache == key) ||
            (LwSciBufGeneralAttrKey_GpuSwNeedCacheCoherency == key) ||
            (LwSciBufGeneralAttrKey_EnableGpuCompression == key) ||
            (LwSciBufInternalGeneralAttrKey_MemDomainArray == key) ||
            (LwSciBufPrivateAttrKey_HeapType == key))) {
            /* All the attributes in codition above are dependent on the
             * platform for their callwlation. As such, in Inter-Soc case,
             * the exporting peer might not be capable of callwlating these keys
             * during export. For ex: in cheetah <-> X86 case, if cheetah Soc is
             * exporter then peer in cheetah cannot query underlying GPUs in the
             * system. As such, callwlate these keys during import. For export,
             * just reuse the already present value in the reconciled list.
             * Also, skip callwlation of any attributes that are dependent on
             * platform dependent attributes. For ex:
             * LwSciBufInternalGeneralAttrKey_MemDomainArray is dependent on
             * LwSciBufGeneralAttrKey_VidMem_GpuId and thus, we need to
             * recallwlate MemDomain during import after we recallwlate
             * Vidmem.
             */
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
                "LwSciBuf-ADV-MISRAC2012-015")
            *recStatus = LwSciBufAttrStatus_Empty;
            *recLen = 0UL;
            goto ret;
        }
    }

    *recStatus = LwSciBufAttrStatus_Empty;
    *recLen = 0UL;

    if (LwSciBuf_IlwalidPolicy != policy) {
        LwSciBufPrivateAttrKeyValuePair pvtKeyValPair = {};
        const LwSciBufIpcTable* ipcTable = NULL;

        pvtKeyValPair.key = LwSciBufPrivateAttrKey_IPCTable;
        err = LwSciBufAttrListCommonGetAttrs(reconciledList, 0, &pvtKeyValPair,
                1, LwSciBufAttrKeyType_Private, true);
        if (LwSciError_Success != err) {
            LWSCI_ERR_STR("Failed to get IPC table attribute.");
            LwSciCommonPanic();
        }

        if (0U == pvtKeyValPair.len) {
            LWSCI_ERR_STR("IPC table attribute not set in reconciled LwSciBufAttrList.");
            LwSciCommonPanic();
        }

        ipcTable = *(LwSciBufIpcTable* const *)pvtKeyValPair.value;

        err = LwSciBufInitIpcTableIter(ipcTable, &ipcTableIter);
        if (LwSciError_Success != err) {
            LWSCI_ERR_STR("Failed to init ipcTable iterator.");
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
                "LwSciBuf-ADV-MISRAC2012-015")
            goto ret;
        }

        while (LwSciBufIpcTableIterNext(ipcTableIter)) {
            const void* ipAddr = NULL;
            uint64_t ipSetSize = 0UL;
            LwSciBufAttrStatus ipStatus = LwSciBufAttrStatus_Empty;
            const LwSciBufIpcRoute* ipcRoute = NULL;
            bool isMatch = false;
            LwSciBufReconcilePolicyCookie cookie = {};
            const LwSciBufReconcilePolicyCookie* tmpCookie = NULL;

            LwSciBufIpcGetRouteFromIter(ipcTableIter, &ipcRoute);

            LwSciBufIpcRouteMatchAffinity(ipcRoute, routeAffinity, ipcEndPoint,
                localPeer, &isMatch);
            if (false == isMatch) {
                continue;
            }

            err = LwSciBufIpcIterLwrrGetAttrKey(ipcTableIter, key, &ipSetSize,
                    &ipAddr);
            if (LwSciError_Success != err) {
                LWSCI_ERR_STR("Failed to Get AttrKey from IPC Table.");
                LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
                    "LwSciBuf-ADV-MISRAC2012-015")
                goto free_ret;
            }

            if (0UL == ipSetSize) {
                continue;
            }

            ipStatus = LwSciBufAttrStatus_SetLocked;

            if (GpuCacheAndPolicySameKey ==
                recPolicyMapSameKey[(uint32_t)policy]) {
                LwSciBufModule module = NULL;
                /* Create a dummy unreconciled list since it needs to be passed
                 * as cookie to GpuCacheAndPolicySameKey.
                 */
                err = LwSciBufAttrListGetModule(reconciledList, &module);
                if (LwSciError_Success != err) {
                    LWSCI_ERR_STR("LwSciBufAttrListGetModule failed.");
                    goto free_ret;
                }

                err = LwSciBufAttrListCreate(module, &dummyAppendList);
                if (LwSciError_Success != err) {
                    LWSCI_ERR_STR("LwSciBufAttrListCreate failed.");
                    goto free_ret;
                }

                cookie.appendList = dummyAppendList;
                cookie.reconciledList = reconciledList;
                tmpCookie = &cookie;
            } else {
                tmpCookie = NULL;
            }

            err = recPolicyMapSameKey[(uint32_t)policy](key, ipAddr, &ipStatus,
                    &ipSetSize, recAddr, recStatus, recLen, tmpCookie);
            if (LwSciError_Success != err) {
                LWSCI_ERR_HEXUINT("Reconciliation from IPC table failed for key: ",
                    key);
                LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
                    "LwSciBuf-ADV-MISRAC2012-015")
                goto free_attrlist;
            }

            *recStatus = LwSciBufAttrStatus_Reconciled;
        }

        if (LwSciBufInternalGeneralAttrKey_MemDomainArray == key) {
            /* mem domain is special because it is callwlated using
             * reconciliation policy and then its dependency on other keys
             * is callwlated.
             */
            err = LwSciBufAttrListMemDomainDependency(reconciledList);
            if (LwSciError_Success != err) {
                LWSCI_ERR_STR("LwSciBufAttrListMemDomainDependency failed.");
                LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
                    "LwSciBuf-ADV-MISRAC2012-015")
                goto free_attrlist;
            }

        } else if (LwSciBufGeneralAttrKey_GpuId == key) {
            /* GpuId key is special because it is reconciled using
             * reconciliation policy and then its dependency on other keys
             * is callwlated.
             */

            /* GpuId depends on VidMemGpuId and MemDomain. However, when we are
             * here in code, it is not guaranteed that these keys are callwlated
             * because it comes later in iteration sequence than GpuId. Thus,
             * callwlate these keys first.
             */
            /* TODO: We are calling LwSciBufAttrListReconcileFromIpcTable()
             * here which MISRA will flag as relwrsion. This can be solved if
             * we rewrite LwSciBufAttrListReconcileFromIpcTable() such that
             * it iterates over all the keys, callwlates the keys using
             * reconciliation policies and then callwalate depedency.
             * Essentially, this function will work the same way as
             * LwSciBufAttrListReconcile() but only for keys that need to be
             * callwlated during export. This approach however needs overhaul
             * of transport unit where transport unit lwrrently iterates over
             * keys. Transport unit needs to change such that it passes
             * reconciled list to this function and it gets the keys which
             * need to be recallwlated during export outputted in reconciled
             * list. Also, we will have to write new function which would give
             * reconciliation of keys for localPeer. (This function handles
             * both cases today using localPeer flag).
             *
             * The other option is to move to new reconciliation framework
             * which statically sorts keys based on their dependency but
             * this is a long shot for now.
             */
            err = LwSciBufAttrListReconcileFromIpcTable(reconciledList,
                    LwSciBufGeneralAttrKey_VidMem_GpuId, ipcEndPoint, localPeer,
                    true, routeAffinity);
            if (LwSciError_Success != err) {
                LWSCI_ERR_STR("LwSciBufAttrListReconcileFromIpcTable failed for LwSciBufGeneralAttrKey_VidMem_GpuId.");
                LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
                    "LwSciBuf-ADV-MISRAC2012-015")
                goto free_attrlist;
            }

            err = LwSciBufAttrListReconcileFromIpcTable(reconciledList,
                    LwSciBufInternalGeneralAttrKey_MemDomainArray, ipcEndPoint,
                    localPeer, true, routeAffinity);
            if (LwSciError_Success != err) {
                LWSCI_ERR_STR("LwSciBufAttrListReconcileFromIpcTable failed for LwSciBufInternalGeneralAttrKey_MemDomainArray.");
                LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
                    "LwSciBuf-ADV-MISRAC2012-015")
                goto free_attrlist;
            }

            err = LwSciBufAttrListGpuIdDependency(reconciledList);
            if (LwSciError_Success != err) {
                LWSCI_ERR_STR("LwSciBufAttrListGpuIdDependency failed.");
                LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
                    "LwSciBuf-ADV-MISRAC2012-015")
                goto free_attrlist;
            }
        } else if (LwSciBufGeneralAttrKey_EnableGpuCache == key) {
            /* EnableGpuCache key is special because it is reconciled using
             * reconciliation policy and then its dependency on other keys
             * is callwlated.
             */
            err = LwSciBufAttrListGpuCacheEnableDependency(reconciledList);

            if (LwSciError_Success != err) {
                LWSCI_ERR_STR("LwSciBufAttrListGpuCacheEnableDependency failed.");
                LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
                    "LwSciBuf-ADV-MISRAC2012-015")
                goto free_attrlist;
            }
        } else if (LwSciBufGeneralAttrKey_EnableGpuCompression == key) {
            /* GpuCacheCompression depends on EngineArray and MemDomain (and
             * other keys). However, when we are here in code, it is not
             * guaranteed that these keys are callwlated because they come
             * later in iteration sequence than GpuCompression. Thus, callwlate
             * these keys first.
             */
            /* TODO: Check the comment about MISRA violation for relwrsion
             * in the GpuId section above.
             */
            err = LwSciBufAttrListReconcileFromIpcTable(reconciledList,
                    LwSciBufInternalGeneralAttrKey_EngineArray, ipcEndPoint,
                    localPeer, true, routeAffinity);
            if (LwSciError_Success != err) {
                LWSCI_ERR_STR("LwSciBufAttrListReconcileFromIpcTable failed for LwSciBufInternalGeneralAttrKey_EngineArray.");
                LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
                    "LwSciBuf-ADV-MISRAC2012-015")
                goto free_attrlist;
            }

            err = LwSciBufAttrListReconcileFromIpcTable(reconciledList,
                    LwSciBufInternalGeneralAttrKey_MemDomainArray, ipcEndPoint,
                    localPeer, true, routeAffinity);
            if (LwSciError_Success != err) {
                LWSCI_ERR_STR("LwSciBufAttrListReconcileFromIpcTable failed for LwSciBufInternalGeneralAttrKey_MemDomainArray.");
                LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
                    "LwSciBuf-ADV-MISRAC2012-015")
                goto free_attrlist;
            }

            err = LwSciBufAttrListGpuCompressionDependency(reconciledList,
                    localPeer, ipcEndPoint);
            if (LwSciError_Success != err) {
                LWSCI_ERR_STR("LwSciBufAttrListGpuCompressionDependency() failed.");
                LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
                    "LwSciBuf-ADV-MISRAC2012-015")
                goto free_attrlist;
            }
        } else if (LwSciBufGeneralAttrKey_RequiredPerm == key) {
            err = LwSciBufAttrListSetDefaultRequiredPerm(reconciledList);
            if (LwSciError_Success != err) {
                LWSCI_ERR_STR("LwSciBufAttrListSetDefaultRequiredPerm() failed.");
                LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
                    "LwSciBuf-ADV-MISRAC2012-015")
                goto free_attrlist;
            }
        } else if (LwSciBufGeneralAttrKey_NeedCpuAccess == key) {
            err = LwSciBufAttrListSetDefaultCpuAccess(reconciledList);
            if (LwSciError_Success != err) {
                LWSCI_ERR_STR("LwSciBufAttrListSetDefaultCpuAccess() failed.");
                LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
                    "LwSciBuf-ADV-MISRAC2012-015")
                goto free_attrlist;
            }
        } else if (LwSciBufGeneralAttrKey_VidMem_GpuId == key) {
            /* Vidmem_GpuId is special because it depends on EngineArray
             * However, when we are here in code, it is not guaranteed that
             * EngineArray is callwlated because it comes later in iteration
             * sequence than Vidmem_GpuId. Thus, callwlate EngineArray first.
             * TODO (Bug 3468578): Check the comment about MISRA violation for
             * relwrsion in the GpuId section above.
             */
            err = LwSciBufAttrListReconcileFromIpcTable(reconciledList,
                    LwSciBufInternalGeneralAttrKey_EngineArray, ipcEndPoint,
                    localPeer, true, routeAffinity);
            if (LwSciError_Success != err) {
                LWSCI_ERR_STR("LwSciBufAttrListReconcileFromIpcTable failed for LwSciBufInternalGeneralAttrKey_EngineArray.");
                LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
                    "LwSciBuf-ADV-MISRAC2012-015")
                goto free_attrlist;
            }

            err = LwSciBufAttrListVidmemDependency(reconciledList);
            if (LwSciError_Success != err) {
                LWSCI_ERR_STR("LwSciBufAttrListVidmemDependency() failed.");
                LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
                    "LwSciBuf-ADV-MISRAC2012-015")
                goto free_attrlist;
            }
        } else {
            /*  Add keys as and when we support them. */
        }
    } else {
        /* output only attribute. This assumes that the attributes that output
         * only attribute is dependent on are supplied in the
         * reconciledList provided as input to this function.
         */
        if (LwSciBufGeneralAttrKey_ActualPerm == key) {
            err = LwSciBufAttrListActualPermDependency(reconciledList);
            if (LwSciError_Success != err) {
                LWSCI_ERR_STR("LwSciBufAttrListPermDependency failed.");
                LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
                    "LwSciBuf-ADV-MISRAC2012-015")
                goto ret;
            }
        } else if (LwSciBufGeneralAttrKey_CpuNeedSwCacheCoherency == key) {
            err = LwSciBufAttrListCpuKeysDependency(reconciledList, localPeer,
                    ipcEndPoint);
            if (LwSciError_Success != err) {
                LWSCI_ERR_STR("LwSciBufAttrListCpuKeysDependency failed.");
                LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
                    "LwSciBuf-ADV-MISRAC2012-015")
                goto ret;
            }
        } else if (LwSciBufGeneralAttrKey_GpuSwNeedCacheCoherency == key) {
            /* GpuCacheCoherency depends on EngineArray and MemDomain (and other
             * keys). However, when we are here in code, it is not guaranteed
             * that these keys are callwlated because they comes later in
             * iteration sequence than GpuCacheCoherency. Thus, callwlate these
             * keys first.
             */
            /* TODO: Check the comment about MISRA violation for relwrsion
             * in the GpuId section above.
             */
            err = LwSciBufAttrListReconcileFromIpcTable(reconciledList,
                    LwSciBufInternalGeneralAttrKey_EngineArray, ipcEndPoint,
                    localPeer, true, routeAffinity);
            if (LwSciError_Success != err) {
                LWSCI_ERR_STR("LwSciBufAttrListReconcileFromIpcTable failed for LwSciBufInternalGeneralAttrKey_EngineArray.");
                LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
                    "LwSciBuf-ADV-MISRAC2012-015")
                goto ret;
            }

            err = LwSciBufAttrListReconcileFromIpcTable(reconciledList,
                    LwSciBufInternalGeneralAttrKey_MemDomainArray, ipcEndPoint,
                    localPeer, true, routeAffinity);
            if (LwSciError_Success != err) {
                LWSCI_ERR_STR("LwSciBufAttrListReconcileFromIpcTable failed for LwSciBufInternalGeneralAttrKey_MemDomainArray.");
                LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
                    "LwSciBuf-ADV-MISRAC2012-015")
                goto ret;
            }

            err = LwSciBufAttrListGpuSwCacheCoherDependency(reconciledList,
                    localPeer, ipcEndPoint);
            if (LwSciError_Success != err) {
                LWSCI_ERR_STR("LwSciBufAttrListGpuSwCacheCoherDependency failed.");
                LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
                    "LwSciBuf-ADV-MISRAC2012-015")
                goto ret;
            }
        } else if (LwSciBufPrivateAttrKey_HeapType == key) {
            err = LwSciBufAttrListHeapTypeDependency(reconciledList);
            if (LwSciError_Success != err) {
                LWSCI_ERR_STR("LwSciBufAttrListHeapTypeDependency failed.");
                LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
                    "LwSciBuf-ADV-MISRAC2012-015")
                goto ret;
            }
        } else {
            /* support more keys as and when required. */
        }
    }

free_attrlist:
    if (NULL != dummyAppendList) {
        LwSciBufAttrListFree(dummyAppendList);
    }

free_ret:
    if (NULL != ipcTableIter) {
        LwSciBufFreeIpcIter(ipcTableIter);
    }

ret:
    LWSCI_FNEXIT("");
    return err;
}

LwSciError LwSciBufAttrListReconcileInternal(
    const LwSciBufAttrList inputArray[],
    size_t inputCount,
    LwSciBufAttrList newAttrList,
    LwSciBufAttrList* newConflictList,
    bool ignoreUnreconciledLists)
{
    LwSciError err = LwSciError_Success;
    LwSciBufAttrList inputAppendAttrList = NULL;

    static const LwSciBufPolicyDifferentKey
            recPolicyMapDifferentKey[LwSciBuf_PolicyUpperBound] = {
        [LwSciBuf_MatchPolicy] = CompareDifferentKeysForMatch,
    };

#if (LW_IS_SAFETY != 0)
    (void)newConflictList;
#endif

    LWSCI_FNENTRY("");

#if (LW_IS_SAFETY == 0)
    *newConflictList = NULL;
#endif

    if (((false == ignoreUnreconciledLists) &&
        ((0U == inputCount) || (NULL == inputArray))) || (NULL == newAttrList)
#if (LW_IS_SAFETY == 0)
        || (NULL == newConflictList)
#endif
    ) {
        err = LwSciError_BadParameter;
        LWSCI_ERR_STR("Bad parameters to LwSciBufAttrListReconcileInternal().");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

#if (LW_IS_SAFETY == 0)
    *newConflictList = NULL;
#endif

    err = LwSciBufAttrListValidate(newAttrList);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("Invalid parameter 'newAttrList' supplied to LwSciBufAttrListReconcileInternal().");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    err = LwSciBufAttrListSetState(newAttrList,
            LwSciBufAttrListState_Reconciled);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("Failed to set reconciliation flag.");
        LwSciCommonPanic();
    }

    if (false == ignoreUnreconciledLists) {
        err = LwSciBufValidateAttrListArray(inputArray, inputCount);
        if (LwSciError_Success != err) {
            LWSCI_ERR_STR("Invalid attrList to LwSciBufAttrListReconcileInternal().");
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
            goto ret;
        }

        err = LwSciBufAttrListAppendWithLocksUnreconciled(inputArray,
                inputCount, false, &inputAppendAttrList);
        if (LwSciError_Success != err) {
            LWSCI_ERR_STR("Failed to append all input lists.");
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
            goto ret_conflictlist;
        }

        err = LwSciBufCreateReconciledIpcTable(inputAppendAttrList,
                newAttrList);
        if (LwSciError_Success != err) {
            LWSCI_ERR_STR("Failed to Create Ipc Table during reconciliation.");
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
            goto ret_conflictlist;
        }

        /* Reconcile appended attribute list. Same keys from appended attribute
         * list are merged according to reconciliation policy set for the keys.
         * In case of interop between different datatypes, different keys
         * belonging to different datatypes need to be reconciled as well. This
         * is taken care of later.
         */
        err = LwSciBufAttrListMergeSameKeys(inputAppendAttrList, newAttrList,
                recPolicyMapSameKey);
        if (LwSciError_Success != err) {
            LWSCI_ERR_STR("Failed to reconcile list.");
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
            goto ret_conflictlist;
        }

        /* Check for required keys. */
        err = LwSciBufAttrListCheckRequiredSameKeys(newAttrList);
        if (LwSciError_Success != err) {
            LWSCI_ERR_STR("Failed to reconcile list.");
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
            goto ret_conflictlist;
        }
    }

    err = LwSciBufAttrListSetGeneralKeyDependency(newAttrList, true, 0U);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("LwSciBufAttrListSetGeneralKeyDependency() failed.");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret_conflictlist;
    }

    err = LwSciBufValidateMergedAttrList(newAttrList);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("LwSciBufValidateMergedAttrList failed.");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret_conflictlist;
    }

    err = LwSciBufAttrListCompareDifferentKeys(newAttrList,
            recPolicyMapDifferentKey);
    if (LwSciError_Success != err) {
        err = LwSciError_ReconciliationFailed;
        LWSCI_ERR_STR("Failure when reconciling keys from different datatypes.");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret_conflictlist;
    }

    err = LwSciBufApplyConstraints(newAttrList);
    if (LwSciError_Success != err) {
        err = LwSciError_ReconciliationFailed;
        LWSCI_ERR_STR("Failed to apply hardware constraints.");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret_conflictlist;
    }

    if (NULL != inputAppendAttrList) {
        LwSciBufAttrListFree(inputAppendAttrList);
    }

    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
    goto ret;

ret_conflictlist:
#if (LW_IS_SAFETY == 0)
    *newConflictList = inputAppendAttrList;
#else
    LwSciBufAttrListFree(inputAppendAttrList);
#endif

ret:
    LWSCI_FNEXIT("");
    return err;
}

LwSciError LwSciBufAttrListReconcile(
    const LwSciBufAttrList inputArray[],
    size_t inputCount,
    LwSciBufAttrList* newReconciledAttrList,
    LwSciBufAttrList* newConflictList)
{
    LwSciError err = LwSciError_Success;
    LwSciBufAttrList newAttrList = NULL;
    LwSciBufModule module = NULL;

#if (LW_IS_SAFETY != 0)
    (void)newConflictList;
#endif

    LWSCI_FNENTRY("");

    if ((0U == inputCount) || (NULL == newReconciledAttrList)
#if (LW_IS_SAFETY == 0)
        || (NULL == newConflictList)
#endif
        || (NULL == inputArray)) {
        err = LwSciError_BadParameter;
        LWSCI_ERR_STR("Bad parameters to LwSciBufAttrListReconcile\n");
        LWSCI_ERR_ULONG("inputCount: \n", inputCount);
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }
    *newReconciledAttrList = NULL;
#if (LW_IS_SAFETY == 0)
    *newConflictList = NULL;
#endif

    err = LwSciBufValidateAttrListArray(inputArray, inputCount);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("Invalid attrList to LwSciBufAttrListReconcile.");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    err = LwSciBufAttrListsLock(inputArray, inputCount);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("Failed to lock all Attribute Lists\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    err = LwSciBufAttrListGetModule(inputArray[0], &module);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("Failed to get module from input attribute list.");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto unlock_attr_lists;
    }

    err = LwSciBufAttrListCreateMultiSlot(module, 1, &newAttrList, false);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("Failed to create empty reconciled list.");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto unlock_attr_lists;
    }

    err = LwSciBufAttrListReconcileInternal(inputArray, inputCount, newAttrList,
            newConflictList, false);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("LwSciBufAttrListReconcileInternal() failed.");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto free_newAttrList;
    }

    *newReconciledAttrList = newAttrList;
    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
    goto unlock_attr_lists;

free_newAttrList:
    LwSciBufAttrListFree(newAttrList);
    *newReconciledAttrList = NULL;

unlock_attr_lists:
    {
        LwSciError error = LwSciError_Success;

        error = LwSciBufAttrListsUnlock(inputArray, inputCount);
        if (LwSciError_Success != error) {
            LWSCI_ERR_STR("Could not unlock Attribute Lists\n");
            /** Only update return error code if no error lwrrently exists. */
            if (LwSciError_Success == err) {
                err = error;
            }
        }
    }

ret:
    LWSCI_FNEXIT("");
    return err;
}

LwSciError LwSciBufAttrListValidateReconciled(
    LwSciBufAttrList reconciledAttrList,
    const LwSciBufAttrList unreconciledAttrListArray[],
    size_t unreconciledAttrListCount,
    bool* isReconcileListValid)
{
    LwSciError err = LwSciError_Success;
    LwSciBufAttrList inputAppendAttrList = NULL;

    static const LwSciBufPolicySameKey
            verPolicyMapSameKey[LwSciBuf_PolicyUpperBound] = {
        [LwSciBuf_MatchPolicy]             = VerifyMatchPolicySameKey,
        [LwSciBuf_OrPolicy]                = VerifyOrPolicySameKey,
        [LwSciBuf_MaxPolicy]               = VerifyMaxPolicySameKey,
        [LwSciBuf_ArrayUnionPolicy]        = VerifyArrayUnionPolicySameKey,
        [LwSciBuf_ListUnionPolicy]         = VerifyListUnionPolicySameKey,
        [LwSciBuf_ArrayIntersectionPolicy] = VerifyArrayIntersectionPolicySameKey,
        [LwSciBuf_GpuCacheAndPolicy]       = VerifyGpuCacheAndPolicySameKey,
        [LwSciBuf_GpuCompressionMatchPolicy] =
                                            VerifyGpuCompressionMatchPolicySameKey,
        [LwSciBuf_IlwalidPolicy]        = NULL,
    };

    LWSCI_FNENTRY("");
    if ((0U == unreconciledAttrListCount) || (NULL == reconciledAttrList)
        || (NULL == unreconciledAttrListArray) || (NULL == isReconcileListValid)) {
        err = LwSciError_BadParameter;
        LWSCI_ERR_STR("Bad parameters to LwSciBufAttrListVerifyReconcile\n");
        LWSCI_ERR_ULONG("inputcount: \n", unreconciledAttrListCount);
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }
    *isReconcileListValid = false;

    err = LwSciBufValidateAttrListArray(unreconciledAttrListArray,
            unreconciledAttrListCount);
    if (LwSciError_Success != err) {
        err = LwSciError_BadParameter;
        LWSCI_ERR_STR("Invalid attrList to LwSciBufAttrListVerifyReconcile\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    err = LwSciBufAttrListAppendUnreconciled(
            unreconciledAttrListArray, unreconciledAttrListCount,
            &inputAppendAttrList);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("Failed to append all input lists\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    err = LwSciBufAttrListMergeSameKeys(inputAppendAttrList, reconciledAttrList,
            verPolicyMapSameKey);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("Failed to verify list\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto free_appened_list;
    }

    *isReconcileListValid = true;

free_appened_list:
    LwSciBufAttrListFree(inputAppendAttrList);

ret:
    LWSCI_FNEXIT("");
    return err;
}

static LwSciError AddEngineBeforeExport(
    LwSciBufAttrList attrList,
    const LwSciBufHwEngName engineName)
{
    LwSciError err = LwSciError_Success;
    LwSciBufHwEngine engineList[LW_SCI_BUF_HW_ENGINE_MAX_NUMBER] = {0};
    LwSciBufInternalAttrKeyValuePair intKeyValPair = {0};
    size_t slotCount = LwSciBufAttrListGetSlotCount(attrList);
    size_t engineCount = 0UL;
    int64_t engineID = 0LL;
    uint64_t len = 0UL;
    bool hasEngine = false;

    intKeyValPair.key = LwSciBufInternalGeneralAttrKey_EngineArray;

    /* If the input list is appended attribute list we set this engine in all slots */
    for (size_t slotNum=0U; slotNum<slotCount; slotNum++) {
        err = LwSciBufAttrListCommonGetAttrs(attrList, slotNum, &intKeyValPair,
                            1U, LwSciBufAttrKeyType_Internal, true);
        if (LwSciError_Success != err) {
            LWSCI_ERR_STR("LwSciBufAttrListCommonGetAttrs failed.");
            LwSciCommonPanic();
        }

        len = (uint64_t)intKeyValPair.len;
        LwSciCommonMemcpyS(engineList,
                        sizeof(LwSciBufHwEngine)*LW_SCI_BUF_HW_ENGINE_MAX_NUMBER,
                        intKeyValPair.value,
                        len);

        engineCount = len / sizeof(LwSciBufHwEngine);

        err = LwSciBufHasEngine(engineList, engineCount, engineName, &hasEngine);

        if (false == hasEngine) {
            if (LW_SCI_BUF_HW_ENGINE_MAX_NUMBER == engineCount) {
                err = LwSciError_InsufficientResource;
                goto ret;
            }
            err = LwSciBufHwEngCreateIdWithoutInstance(engineName, &engineID);
            if (LwSciError_Success != err) {
                LWSCI_ERR_STR("LwSciBufHwEngCreateIdWithoutInstance failed.");
                LwSciCommonPanic();
            }

            engineList[engineCount].engNamespace = LwSciBufHwEngine_TegraNamespaceId;
            engineList[engineCount].rmModuleID = engineID;
            engineList[engineCount].rev.engine_rev = 0U;
            engineCount++;

            intKeyValPair.key = LwSciBufInternalGeneralAttrKey_EngineArray;
            intKeyValPair.value = engineList;
            intKeyValPair.len = sizeof(LwSciBufHwEngine)*engineCount;
            err = LwSciBufAttrListCommonSetAttrs(attrList, slotNum, &intKeyValPair,
                1U, LwSciBufAttrKeyType_Internal, true, false);
            if (LwSciError_Success != err) {
                LWSCI_ERR_STR("LwSciBufAttrListCommonSetAttrs failed.");
                LwSciCommonPanic();
            }
        }
    }

ret:
    return err;
}

static LwSciError UnreconciledAttrListUpdateEngineListBeforeExport(
    LwSciBufAttrList attrList,
    const LwSciIpcEndpoint ipcEndPoint)
{
    LwSciError err = LwSciError_Success;
    bool isSocBoundary = false;

    err = LwSciBufIsSocBoundary(ipcEndPoint, &isSocBoundary);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("LwSciBufIsSocBoundary failed.");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    /* Lwrrently we do not have any update anything if the attribute is exported within SoC */
    if (false == isSocBoundary) {
        goto ret;
    }

    /* We eed PCIe engine in engine list for inter SoC case so that we can set cache coherency correctly. */
    err = AddEngineBeforeExport(attrList, LwSciBufHwEngName_PCIe);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("Failed to add PCIe engine to engineList for inter SoC use case.");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

ret:
    LWSCI_FNEXIT("");
    return err;
}

static LwSciError UnreconciledAttrListUpdateBeforeExport(
    LwSciBufAttrList attrList,
    const LwSciIpcEndpoint ipcEndPoint)
{
    LwSciError err = LwSciError_Success;
    err = UnreconciledAttrListUpdateEngineListBeforeExport(attrList, ipcEndPoint);
    return err;
}

static LwSciError ReconciledAttrListUpdateBeforeExport(
    LwSciBufAttrList attrList,
    const LwSciIpcEndpoint ipcEndPoint)
{
    (void)attrList;
    (void)ipcEndPoint;
    /* Lwrrently we do not have any update anything if the attribute is reconciled */
    return LwSciError_Success;
}

LwSciError LwSciBufAttrListUpdateBeforeExport(
    LwSciBufAttrList attrList,
    const LwSciIpcEndpoint ipcEndPoint)
{
    LwSciError err = LwSciError_Success;
    bool isReconciled = false;
    struct LwSciIpcEndpointInfo info = {0};

    err = LwSciBufAttrListValidate(attrList);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("LwSciBufAttrListValidate failed.");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    err = LwSciIpcGetEndpointInfo(ipcEndPoint, &info);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("Failed to validate LwSciIpcEndpoint.");
        err = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    err = LwSciBufAttrListIsReconciled(attrList, &isReconciled);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("LwSciBufAttrListIsReconciled failed.");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    if (true == isReconciled) {
        err = ReconciledAttrListUpdateBeforeExport(attrList, ipcEndPoint);
    } else {
        err = UnreconciledAttrListUpdateBeforeExport(attrList, ipcEndPoint);
    }

ret:
    LWSCI_FNEXIT("");
    return err;
}
