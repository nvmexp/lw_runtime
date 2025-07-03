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
 * \brief <b>LwSciBuf CheetAh Common Memory Operation Implementations</b>
 *
 * @b Description: This file implements LwSciBuf CheetAh common allocation APIs
 *
 * The code in this file is organised as below:
 * -Public interfaces definition.
 */

#include "lwscibuf_alloc_common_tegra_priv.h"
#include "lwscicommon_os.h"

static void LwSciBufAllocCommonTegraColwertAccPerm(
    LwSciBufAttrValAccessPerm accPerm,
    uint32_t* lwRmAccPerm)
{
    static const LwSciBufAllocCommonTegraAccPermMap
        allocCommonAccPermTable[LwSciBufAccessPerm_Ilwalid] = {
        [LwSciBufAccessPerm_Readonly] =
            {LWOS_MEM_READ},

        [LwSciBufAccessPerm_ReadWrite] =
            {LWOS_MEM_READ | LWOS_MEM_WRITE},
    };

    LWSCI_FNENTRY("");

    /* verify input paramters */
    if (!((accPerm == LwSciBufAccessPerm_Readonly)
        || (accPerm == LwSciBufAccessPerm_ReadWrite))) {
        LWSCI_ERR_STR("Bad parameter supplied to LwSciBufAllocCommonColwertAccPerm\n");
        LWSCI_ERR_UINT("accPerm: \n", (uint32_t)accPerm);
        LwSciCommonPanic();
    }

    /* print input parameters */
    LWSCI_INFO("Input: accPerm: " PRIu32 "lwRmAccPerm: %p\n", accPerm,
        lwRmAccPerm);

    *lwRmAccPerm =
        allocCommonAccPermTable[accPerm].lwRmAccessPerm;

    /* print output parameters */
    LWSCI_INFO("Output: *lwRmAccPerm:" PRIu32 "\n", *lwRmAccPerm);

    LWSCI_FNEXIT("");
}

void LwSciBufAllocCommonTegraColwertCoherency(
    bool lwSciBufCoherency,
    LwOsMemAttribute* lwRmCoherency)
{
    LWSCI_FNENTRY("");

    /* verify input parameters */
    if (lwRmCoherency == NULL) {
        LWSCI_ERR_STR("Bad parameter supplied to LwSciBufAllocSysMemToLwRmCoherency\n");
        LwSciCommonPanic();
    }

    /* print input parameters */
    LWSCI_INFO("Input: lwSciBufCoherency: %s, lwRmCoherency ptr: %p\n",
        true ? "true": "false", lwRmCoherency);

    if (lwSciBufCoherency == true) {
        *lwRmCoherency = LwOsMemAttribute_WriteBack;
    } else {
        *lwRmCoherency = LwOsMemAttribute_WriteCombined;
    }

    /* print output parameters */
    LWSCI_INFO("Output: lwRmCoherency: " PRIu32 "\n", *lwRmCoherency);

    LWSCI_FNEXIT("");
}

void LwSciBufAllocCommonTegraMemFree(
    LwRmMemHandle memHandle)
{
    LWSCI_FNENTRY("");

    if (memHandle == 0U) {
        LWSCI_ERR_STR("memHandle is 0U.\n");
        LwSciCommonPanic();
    }

    LwRmMemHandleFree(memHandle);

    LWSCI_FNEXIT("");
}

LwSciError LwSciBufAllocCommonTegraDupHandle(
    LwRmMemHandle memHandle,
    LwSciBufAttrValAccessPerm newPerm,
    LwRmMemHandle* dupHandle)
{
    LwSciError err = LwSciError_Success;
    uint32_t lwRmAccFlags = 0U;
    LwError lwErr = LwError_Success;

    LWSCI_FNENTRY("");

    if ((memHandle == 0U) || (dupHandle == NULL)) {
        LWSCI_ERR_STR("Bad parameter supplied to LwSciBufAllocCommonDupHandle\n");
        LWSCI_ERR_UINT("memHandle: ", memHandle);
        LwSciCommonPanic();
    }

    LWSCI_INFO("Input: memHandle: " PRIu32 ", dupHandle: %p\n", memHandle,
        dupHandle);

    LwSciBufAllocCommonTegraColwertAccPerm(newPerm, &lwRmAccFlags);

    lwErr = LwRmMemHandleDuplicate(memHandle, lwRmAccFlags, dupHandle);
    if (lwErr != LwError_Success) {
        err = LwSciError_ResourceError;
        LWSCI_ERR_INT("LwRmMemHandleDuplicate failed. LwError: \n", (int32_t)lwErr);
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    LWSCI_INFO("Output: dupHandle: " PRIu32 "\n", *dupHandle);

ret:
    LWSCI_FNEXIT("");
    return err;
}

LwSciError LwSciBufAllocCommonTegraMemMap(
    LwRmMemHandle memHandle,
    uint64_t offset,
    uint64_t len,
    LwSciBufAttrValAccessPerm accPerm,
    void** ptr)
{
    LwSciError err = LwSciError_Success;
    uint32_t lwRmAccFlags = 0U;
    LwError lwErr = LwError_Success;

    LWSCI_FNENTRY("");

    if ((memHandle == 0U) || (len == 0U) || (ptr == NULL)) {
        LWSCI_ERR_STR("Bad parameter supplied to LwSciBufAllocCommonTegraMemMap\n");
        LWSCI_ERR_UINT("memHandle: ", memHandle);
        LWSCI_ERR_ULONG("offset: ", offset);
        LWSCI_ERR_ULONG("len: ", len);
        LWSCI_ERR_UINT("accPerm: ", (uint32_t)accPerm);
        LwSciCommonPanic();
    }

    LWSCI_INFO("Input: memHandle: " PRIu32 ", offset: " PRIu64 ", len: " PRIu64 ", accPerm: " PRIu32 ", ptr: %p\n",
        memHandle, offset, len, accPerm, ptr);

    LwSciBufAllocCommonTegraColwertAccPerm(accPerm, &lwRmAccFlags);

    lwErr = LwRmMemMap(memHandle, offset, len, lwRmAccFlags, ptr);
    if (lwErr != LwError_Success) {
        err = LwSciError_ResourceError;
        LWSCI_ERR_INT("LwRmMemMap failed. LwError: \n", (int32_t)lwErr);
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    LWSCI_INFO("Output: *ptr: %p\n", *ptr);

ret:
    LWSCI_FNEXIT("");
    return err;
}

LwSciError LwSciBufAllocCommonTegraMemUnmap(
    LwRmMemHandle memHandle,
    void* ptr,
    uint64_t size)
{
    LwSciError err = LwSciError_Success;
    LwError lwErr = LwError_Success;

    LWSCI_FNENTRY("");

    if ((memHandle == 0U) || (ptr == NULL) || (size == 0U)) {
        LWSCI_ERR_STR("Bad parameter supplied to LwSciBufAllocCommonTegraMemUnmap\n");
        LWSCI_ERR_UINT("memHandle: ", memHandle);
        LWSCI_ERR_ULONG("size: ", size);
        LwSciCommonPanic();
    }

    LWSCI_INFO("Input: memHandle: " PRIu32 ", ptr: %p, size: " PRIu64 "\n",
        memHandle, ptr, size);

    lwErr = LwRmMemUnmap(memHandle, ptr, size);
    if (lwErr != LwError_Success) {
        err = LwSciError_ResourceError;
        LWSCI_ERR_INT("LwRmMemUnmap failed. LwError: \n", (int32_t)lwErr);
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

ret:
    LWSCI_FNEXIT("");
    return err;
}

LwSciError LwSciBufAllocCommonTegraGetMemSize(
    LwRmMemHandle memHandle,
    uint64_t* size)
{
    LwSciError err = LwSciError_Success;
    LwError lwErr = LwError_Success;
    LwRmMemHandleParams params = {0};

    LWSCI_FNENTRY("");

    if ((memHandle == 0U) || (size == NULL)) {
        LWSCI_ERR_STR("Bad parameter supplied to LwSciBufAllocCommonTegraGetMemSize\n");
        LWSCI_ERR_UINT("memHandle: ", memHandle);
        LwSciCommonPanic();
    }

    LWSCI_INFO("Input: memHandle: " PRIu32 ", size: %p\n", memHandle, size);

    if (sizeof(params) > UINT32_MAX) {
        LWSCI_ERR_STR("Size of LwRmMemHandleParams too big to be stored "
                      "in 32-bit storage.\n");
        LwSciCommonPanic();
    }
    lwErr = LwRmMemQueryHandleParams(memHandle, 0U, &params, (uint32_t)sizeof(params));
    if (lwErr != LwError_Success) {
        err = LwSciError_ResourceError;
        LWSCI_ERR_INT("LwRmMemQueryHandleParams failed. LwError: \n", (int32_t)lwErr);
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    *size = params.Size;

ret:
    LWSCI_FNEXIT("");
    return err;
}

LwSciError LwSciBufAllocCommonTegraCpuCacheFlush(
    LwRmMemHandle memHandle,
    void* ptr,
    uint64_t len)
{
    LwSciError err = LwSciError_Success;
    LwError lwErr = LwError_Success;

    LWSCI_FNENTRY("");

    if ((memHandle == 0U) || (ptr == NULL) || (len == 0U)) {
        LWSCI_ERR_STR("Bad parameter supplied to LwSciBufAllocCommonTegraCpuCacheFlush\n");
        LWSCI_ERR_UINT("memHandle: ", memHandle);
        LWSCI_ERR_ULONG("len: ", len);
        LwSciCommonPanic();
    }

    LWSCI_INFO("Input: memHandle: " PRIu32 ", ptr: %p, len: " PRIu64 "\n",
        memHandle, ptr, len);

    /*
     * This function is supposed to be implemented using an API that
     * provides flush and ilwalidate functions. But LwRm provides two
     * functions SyncForCPU and SyncForDevice that does the same.
     * But as per API spec of LwRm, SyncForDevice does both flush and
     * ilwalidate and hence we will use SyncForDevice API.
     */
    lwErr = LwRmMemCacheSyncForDevice(memHandle, ptr, len);
    if (lwErr != LwError_Success) {
        err = LwSciError_ResourceError;
        LWSCI_ERR_INT("LwRmMemCacheSyncForDevice failed. LwError: \n", (int32_t)lwErr);
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

ret:
    LWSCI_FNEXIT("");
    return err;
}
