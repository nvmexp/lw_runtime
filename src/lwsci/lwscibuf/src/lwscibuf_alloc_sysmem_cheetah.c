/*
 * Copyright (c) 2019-2021, LWPU CORPORATION. All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#include "lwscibuf_alloc_sysmem_tegra_priv.h"
#include "lwscibuf_alloc_common_tegra.h"
#include "lwscicommon_os.h"

LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 8_9), "LwSciBuf-ADV-MISRAC2012-011")
static const LwSciBufAllocSysMemToLwRmGpuAccessMap
    allocSysMemToLwRmGpuAccessMap[LwSciBufAllocSysGpuAccess_Ilwalid] = {
    [LwSciBufAllocSysGpuAccess_None] =
        {"LwSciBufAllocSysGpuAccess_None",
            (LwRmHeap)LwRmMemGpuAccess_NO_GPU},

    [LwSciBufAllocSysGpuAccess_iGPU] =
        {"LwSciBufAllocSysGpuAccess_iGPU",
            (LwRmHeap)LwRmMemGpuAccess_IGPU},

    [LwSciBufAllocSysGpuAccess_dGPU] =
        {"LwSciBufAllocSysGpuAccess_dGPU",
            (LwRmHeap)LwRmMemGpuAccess_DGPU},

    [LwSciBufAllocSysGpuAccess_GPU] =
        {"LwSciBufAllocSysGpuAccess_GPU",
            (LwRmHeap)LwRmMemGpuAccess_GPU},
};
LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 8_9))

/**
 * @brief static function definitions
 */
static LwSciError LwSciBufAllocSysMemToLwRmGpuAccessType(
    LwSciBufAllocSysGpuAccessType gpuAccessType,
    LwRmMemGpuAccess* lwRmGpuAccessType)
{
    LwSciError sciErr = LwSciError_Success;

    LWSCI_FNENTRY("");

    /* print input parameters */
    LWSCI_INFO("Input: gpuAccessType %u, lwRmGpuAccessType %p\n",
        gpuAccessType, lwRmGpuAccessType);

    if ((LwSciBufAllocSysGpuAccess_Ilwalid <= gpuAccessType) ||
        (LwSciBufAllocSysGpuAccess_None > gpuAccessType)) {
        LWSCI_ERR_STR("Invalid GPU access type supplied\n");
        LWSCI_ERR_ULONG("enum value for GPU access type: \n",
            (uint64_t)gpuAccessType);
        sciErr = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    LwSciCommonMemcpyS(lwRmGpuAccessType, sizeof(*lwRmGpuAccessType),
        &allocSysMemToLwRmGpuAccessMap[gpuAccessType].allocLwRmGpuAccess,
        sizeof(allocSysMemToLwRmGpuAccessMap[gpuAccessType].allocLwRmGpuAccess));

ret:
    LWSCI_FNEXIT("");
    return sciErr;
}

static LwSciBufAllocSysGpuAccessType ColwertGpuAccessBitsetToGpuAccesTypeEnum(
    uint32_t gpuAccessBitset)
{
    const LwSciBufAllocSysGpuAccessType lookup[] = {
        LwSciBufAllocSysGpuAccess_None,
        LwSciBufAllocSysGpuAccess_iGPU,
        LwSciBufAllocSysGpuAccess_dGPU,
        LwSciBufAllocSysGpuAccess_GPU,
    };

    return lookup[gpuAccessBitset];
}

LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 8_9), "LwSciBuf-ADV-MISRAC2012-011")
static const LwSciBufAllocSysMemToLwRmHeapMap
    allocSysMemToLwRmHeapMap[LwSciBufAllocSysMemHeapType_Ilwalid] = {
    [LwSciBufAllocSysMemHeapType_IOMMU] =
        {"LwSciBufAllocSysMemHeapType_IOMMU",
            LwRmHeap_IOMMU},

    [LwSciBufAllocSysMemHeapType_ExternalCarveout] =
        {"LwSciBufAllocSysMemHeapType_ExternalCarveout",
            LwRmHeap_ExternalCarveOut},

    [LwSciBufAllocSysMemHeapType_IVC] =
        {"LwSciBufAllocSysMemHeapType_IVC",
            LwRmHeap_IVC},

    [LwSciBufAllocSysMemHeapType_VidMem] =
        {"LwSciBufAllocSysMemHeapType_VidMem",
            LwRmHeap_VidMem},

    [LwSciBufAllocSysMemHeapType_CvsRam] =
        {"LwSciBufAllocSysMemHeapType_CvsRam",
            LwRmHeap_CvsRam},
};
LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 8_9))

/**
 * @brief static function definitions
 */
static LwSciError LwSciBufAllocSysMemHeapTypeToLwRmHeapType(
    const LwSciBufAllocSysMemHeapType* sysMemAllocHeapType,
    LwRmHeap* lwRmHeapType,
    uint32_t sysMemAllocNumHeaps)
{
    LwSciError sciErr = LwSciError_Success;
    uint32_t i = 0;

    LWSCI_FNENTRY("");

    /* print input parameters */
    LWSCI_INFO("Input: sysMemAllocHeapType: %p\tlwRmHeapType: %p\t"
        "sysMemAllocNumHeaps: %u\n", sysMemAllocHeapType,
        lwRmHeapType, sysMemAllocNumHeaps);

    for (i = 0; i < sysMemAllocNumHeaps; i++) {
        if (LwSciBufAllocSysMemHeapType_Ilwalid <= sysMemAllocHeapType[i]) {
            LWSCI_ERR_STR("Invalid heap type supplied\n");
            LWSCI_ERR_UINT("enum value for heap type: \n", (uint32_t)sysMemAllocHeapType[i]);
            sciErr = LwSciError_BadParameter;
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
            goto ret;
        }

        lwRmHeapType[i] =
            allocSysMemToLwRmHeapMap[sysMemAllocHeapType[i]].allocLwRmHeap;
    }

ret:
    LWSCI_FNEXIT("");
    return sciErr;
}

static LwSciError LwSciBufAllocSysMemToLwRmAllocVal(
    LwSciBufAllocSysMemVal* sysMemAllocVal,
    LwRmMemHandleAttr* lwRmAllocVal,
    LwRmHeap* lwRmHeaps)
{
    LwSciError sciErr = LwSciError_Success;

    LWSCI_FNENTRY("");

    if (UINT32_MAX < sysMemAllocVal->alignment) {
        sciErr = LwSciError_BadParameter;
        LWSCI_ERR_STR("sysMemAllocVal->alignment is greater than UINT32_MAX\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    /* print input parameters */
    LWSCI_INFO("Input: sysMemAllocVal:%p\tsysMemAllocVal->size: %u\t"
        "sysMemAllocVal->heap: %p\tsysMemAllocVal->numHeaps: %u\t"
        "lwRmAllocVal: %p\tlwRmHeaps: %p\n", sysMemAllocVal,
        sysMemAllocVal->size, sysMemAllocVal->heap, sysMemAllocVal->numHeaps,
        lwRmAllocVal, lwRmHeaps);

#if (LW_IS_SAFETY == 0)
    LwSciBufAllocSysMemPrintHeapTypes(sysMemAllocVal->heap,
                sysMemAllocVal->numHeaps);
#endif

    lwRmAllocVal->Size = sysMemAllocVal->size;
    lwRmAllocVal->Alignment = (uint32_t)sysMemAllocVal->alignment;
    LwSciBufAllocCommonTegraColwertCoherency(sysMemAllocVal->coherency,
        &lwRmAllocVal->Coherency);
    lwRmAllocVal->NumHeaps = sysMemAllocVal->numHeaps;

    sciErr = LwSciBufAllocSysMemHeapTypeToLwRmHeapType(sysMemAllocVal->heap,
                lwRmHeaps, sysMemAllocVal->numHeaps);
    if (LwSciError_Success != sciErr) {
        LWSCI_ERR_STR("LwSciBufAllocSysMemHeapTypeToLwRmHeapType failed\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    lwRmAllocVal->Heaps = lwRmHeaps;
    sysMemAllocVal->heapNumber = 0U;
    lwRmAllocVal->HeapNumber = &sysMemAllocVal->heapNumber;
    lwRmAllocVal->Tags = LwRmMemTags_LwSci;

    sciErr = LwSciBufAllocSysMemToLwRmGpuAccessType(sysMemAllocVal->gpuAccess,
                &lwRmAllocVal->GpuAccess);
    if (LwSciError_Success != sciErr) {
        LWSCI_ERR_STR("LwSciBufAllocSysMemToLwRmGpuAccessType failed\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

#if (LW_IS_SAFETY == 0)
    sciErr = LwSciBufAllocSysMemPrintGpuAccessTypes((LwSciBufAllocSysGpuAccessType)(lwRmAllocVal->GpuAccess));
    if (LwSciError_Success != sciErr) {
        LWSCI_ERR_STR("LwSciBufAllocSysMemPrintGpuAccessTypes failed\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }
#endif

ret:
    LWSCI_FNEXIT("");
    return sciErr;
}

static LwSciError LwSciBufRmAlloc(
    LwSciBufAllocSysMemVal* sysMemAllocVal,
    LwSciBufRmHandle *rmHandle)
{
    LwSciError sciErr = LwSciError_Success;
    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Directive, 4_9), "LwSciBuf-ADV-MISRAC2012-008")
    LWRM_DEFINE_MEM_HANDLE_ATTR(lwRmAllocVal);
    LwRmHeap* lwRmHeaps = NULL;
    LwError err = LwError_Success;

    LWSCI_FNENTRY("");

    LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
    lwRmHeaps = LwSciCommonCalloc(sysMemAllocVal->numHeaps,
                            sizeof(LwRmMemHandleAttr));
    LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 11_5))

    if (NULL == lwRmHeaps) {
        sciErr = LwSciError_InsufficientMemory;
        LWSCI_ERR_STR("Could not allocate memory for LwRmHeap\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    sciErr = LwSciBufAllocSysMemToLwRmAllocVal(sysMemAllocVal, &lwRmAllocVal,
                lwRmHeaps);
    if (LwSciError_Success != sciErr) {
        LWSCI_ERR_STR("LwSciBufAllocSysMemToLwRmAllocVal failed\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto free_lwRmHeap;
    }

    err = LwRmMemHandleAllocAttr(NULL, &lwRmAllocVal,
            &rmHandle->memHandle);
    if (LwError_Success != err) {
        sciErr = LwSciError_ResourceError;
        LWSCI_ERR_INT("LwRmMemHandleAllocAttr failed. LwError: \n", (int32_t)err);
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto free_lwRmHeap;
    }

free_lwRmHeap:
    LwSciCommonFree(lwRmHeaps);
ret:
    LWSCI_FNEXIT("");
    return sciErr;
}

/**
 * @brief public function definitions
 */
#if (LW_IS_SAFETY == 0)
void LwSciBufAllocSysMemPrintHeapTypes(
    LwSciBufAllocSysMemHeapType* heaps,
    uint32_t numHeaps)
{
    uint32_t i;

    LWSCI_FNENTRY("");

    /* verify input parameters */
    if (NULL == heaps || 0U == numHeaps) {
        LWSCI_ERR_STR("Bad parameter supplied to LwSciBufAllocSysMemPrintHeapTypes\n");
        LWSCI_ERR_UINT("numHeaps: \n", numHeaps);
        LwSciCommonPanic();
    }

    /* print input parameters */
    LWSCI_INFO("Input: heaps: %p\tnumHeaps: %u\n", heaps, numHeaps);

    LWSCI_INFO("allocSysMemHeapTypes:\t");

    for (i = 0; i < numHeaps; i++) {
        if (LwSciBufAllocSysMemHeapType_Ilwalid <= heaps[i]) {
            LWSCI_ERR_STR("Invalid heap type supplied\n");
            LWSCI_ERR_UINT("enum value for heap type: \n", heaps[i]);
            LwSciCommonPanic();
        }

        LWSCI_INFO("%s: \t", allocSysMemToLwRmHeapMap[heaps[i]].heapName);
    }
    LWSCI_INFO("\n");

    LWSCI_FNEXIT("");
}

LwSciError LwSciBufAllocSysMemPrintGpuAccessTypes(
    LwSciBufAllocSysGpuAccessType gpuAccess)
{
    LwSciError sciErr = LwSciError_Success;

    LWSCI_FNENTRY("");

    /* print input parameters */
    LWSCI_INFO("Input: gpuAccess: %u\n", gpuAccess);

    if (LwSciBufAllocSysGpuAccess_Ilwalid <= gpuAccess) {
        LWSCI_ERR_STR("Invalid GPU access type supplied\n");
        LWSCI_ERR_UINT("enum value for GPU access type: \n", (uint32_t)gpuAccess);
        LwSciCommonPanic();
    }

    LWSCI_INFO("%s: \n",
        allocSysMemToLwRmGpuAccessMap[gpuAccess].gpuAccessTypeName);

    LWSCI_FNEXIT("");
    return sciErr;
}
#endif

LwSciError LwSciBufSysMemOpen(
    LwSciBufDev devHandle,
    void** context)
{
    LwSciError sciErr = LwSciError_Success;
    LwSciBufAllocSysMemContext* sysMemContext = NULL;

    LWSCI_FNENTRY("");

    /* verify input parameters */
    if ((NULL == context) || (NULL == devHandle)) {
        LWSCI_ERR_STR("Bad parameter supplied to LwSciBufSysMemOpen\n");
        LwSciCommonPanic();
    }
    if (NULL == devHandle) {
        LWSCI_ERR_STR("Bad parameter supplied to LwSciBufSysMemOpen\n");
        LwSciCommonPanic();
    }

    /* print input parameters */
    LWSCI_INFO("Input: context: %p\n", context);

    *context = NULL;

    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
    sysMemContext = LwSciCommonCalloc(1, sizeof(LwSciBufAllocSysMemContext));
    if (NULL == sysMemContext) {
        LWSCI_ERR_STR("Could not allocate memory for LwSciBufAllocSysMemContext\n");
        sciErr = LwSciError_InsufficientMemory;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    sysMemContext->magic = LWSCIBUF_SYSMEM_CONTEXT_MAGIC;

    LwSciBufDevGetAllGpuContext(devHandle, &sysMemContext->allGpuContext);

    *context = (void*)sysMemContext;

    /* print output parameters */
    LWSCI_INFO("Output: *context: %p\n", *context);

    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
    goto ret;

ret:
    LWSCI_FNEXIT("");
    return sciErr;
}

LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 8_13), "LwSciBuf-ADV-MISRAC2012-012")
LwSciError LwSciBufSysMemAlloc(
    const void* context,
    void* allocVal,
    const LwSciBufDev devHandle,
    LwSciBufRmHandle* rmHandle)
{
    LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 8_13))
    LwSciError sciErr = LwSciError_Success;
    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
    LwSciBufAllocSysMemVal* sysMemAllocVal = (LwSciBufAllocSysMemVal*)allocVal;
    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 8_13), "LwSciBuf-ADV-MISRAC2012-012")
    const LwSciBufAllocSysMemContext* sysMemContext = NULL;
    LWSCI_FNENTRY("");

    /* verify input parameters */
    if ((NULL == context) || (NULL == sysMemAllocVal)) {
        LWSCI_ERR_STR("Bad parameter supplied to LwSciBufSysMemAlloc\n");
        LwSciCommonPanic();
    }

    if ((0U == sysMemAllocVal->size) || (NULL == sysMemAllocVal->heap) ||
        (0U == sysMemAllocVal->numHeaps) || (NULL == devHandle) || (NULL == rmHandle)) {
        LWSCI_ERR_STR("Bad parameter supplied to LwSciBufSysMemAlloc\n");
        LWSCI_ERR_UINT("sysMemAllocVal->size: \t", sysMemAllocVal->size);
        LWSCI_ERR_UINT("sysMemAllocVal->numHeaps: \n", sysMemAllocVal->numHeaps);
        LwSciCommonPanic();
    }

    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
    sysMemContext = (const LwSciBufAllocSysMemContext*)context;
    if (LWSCIBUF_SYSMEM_CONTEXT_MAGIC != sysMemContext->magic) {
        LWSCI_ERR_STR("Invalid context parameter provided to LwSciBufSysMemAlloc\n");
        LwSciCommonPanic();
    }

    sysMemAllocVal->gpuAccess = sysMemContext->gpuAccessParam;

    /* print input parameters */
    LWSCI_INFO("Input: context: %p\tsysMemAllocVal: %p\tsysMemAllocVal->size: %u\t"
        "sysMemAllocVal->heap: %p\tsysMemAllocVal->numHeaps: %u\t"
        "devHandle: %p\trmHandle: %p\n", context, sysMemAllocVal,
        sysMemAllocVal->size, sysMemAllocVal->heap, sysMemAllocVal->numHeaps,
        devHandle, rmHandle);


    sciErr = LwSciBufRmAlloc(sysMemAllocVal, rmHandle);
    if (LwSciError_Success != sciErr) {
        LWSCI_ERR_STR("LwSciBufRmAlloc failed\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    /* print output parameters */
    LWSCI_INFO("Output: memHandle: %u\n", rmHandle->memHandle);

ret:
    LWSCI_FNEXIT("");
    return sciErr;
}

LwSciError LwSciBufSysMemDealloc(
    void* context,
    LwSciBufRmHandle rmHandle)
{
    LwSciError sciErr = LwSciError_Success;
    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 8_13), "LwSciBuf-ADV-MISRAC2012-012")
    const LwSciBufAllocSysMemContext* sysMemContext = NULL;

    LWSCI_FNENTRY("");

    /* verify input parameters */
    if ((NULL == context) || (0U == rmHandle.memHandle)) {
        LWSCI_ERR_STR("Bad parameter supplied to LwSciBufSysMemDealloc\n");
        LWSCI_ERR_UINT("rmHandle.memHandle: \n", rmHandle.memHandle);
        LwSciCommonPanic();
    }

    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
    sysMemContext = (const LwSciBufAllocSysMemContext*)context;
    if (LWSCIBUF_SYSMEM_CONTEXT_MAGIC != sysMemContext->magic) {
        LWSCI_ERR_STR("Invalid context parameter provided to LwSciBufSysMemDealloc\n");
        LwSciCommonPanic();
    }

    /* print input parameters */
    LWSCI_INFO("Input: context: %p\trmHandle.memHandle: %u\n",
        context, rmHandle.memHandle);

    LwSciBufAllocCommonTegraMemFree(rmHandle.memHandle);

    LWSCI_FNEXIT("");
    return sciErr;
}

LwSciError LwSciBufSysMemDupHandle(
    const void* context,
    LwSciBufAttrValAccessPerm newPerm,
    LwSciBufRmHandle rmHandle,
    LwSciBufRmHandle* dupRmHandle)
{
    LwSciError sciErr = LwSciError_Success;
    const LwSciBufAllocSysMemContext* sysMemContext = NULL;

    LWSCI_FNENTRY("");

    /* verify input parameters */
    if ((NULL == context) || (0U == rmHandle.memHandle) || (NULL == dupRmHandle)) {
        LWSCI_ERR_STR("Bad parameter supplied to LwSciBufSysMemDupHandle\n");
        LWSCI_ERR_UINT("rmHandle.memHandle: \n", rmHandle.memHandle);
        LwSciCommonPanic();
    }

    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
    sysMemContext = (const LwSciBufAllocSysMemContext*)context;
    if (LWSCIBUF_SYSMEM_CONTEXT_MAGIC != sysMemContext->magic) {
        LWSCI_ERR_STR("Invalid context parameter provided to LwSciBufSysMemDupHandle\n");
        LwSciCommonPanic();
    }

    /* print input parameters */
    LWSCI_INFO("Input: context: %p\trmHandle.memHandle: %u\t"
        "dupRmHandle: %p\n", context, rmHandle.memHandle,
        dupRmHandle);

    sciErr = LwSciBufAllocCommonTegraDupHandle(rmHandle.memHandle,
                newPerm, &dupRmHandle->memHandle);
    if (LwSciError_Success != sciErr) {
        LWSCI_ERR_STR("LwSciBufAllocCommonTegraDupHandle failed\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    /* print output parameters */
    LWSCI_INFO("Output: dupRmHandle->memHandle: %u\n", dupRmHandle->memHandle);

ret:
    LWSCI_FNEXIT("");
    return sciErr;
}

LwSciError LwSciBufSysMemMemMap(
    const void* context,
    LwSciBufRmHandle rmHandle,
    uint64_t offset,
    uint64_t len,
    LwSciBufAttrValAccessPerm accPerm,
    void **ptr)
{
    LwSciError sciErr = LwSciError_Success;
    const LwSciBufAllocSysMemContext* sysMemContext = NULL;

    LWSCI_FNENTRY("");

    /* verify input parameters */
    if ((NULL == context) || (0U == rmHandle.memHandle) || (0U == len) ||
        (LwSciBufAccessPerm_Ilwalid <= accPerm) || (NULL == ptr)) {
        LWSCI_ERR_STR("Bad parameter supplied to LwSciBufSysMemMemMap\n");
        LWSCI_ERR_UINT("rmHandle.memHandle: \t", rmHandle.memHandle);
        LWSCI_ERR_ULONG("offset: \t", offset);
        LWSCI_ERR_ULONG("len: \t", len);
        LWSCI_ERR_UINT("accPerm: \n", (uint32_t)accPerm);
        LwSciCommonPanic();
    }

    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
    sysMemContext = (const LwSciBufAllocSysMemContext*)context;
    if (LWSCIBUF_SYSMEM_CONTEXT_MAGIC != sysMemContext->magic) {
        LWSCI_ERR_STR("Invalid context parameter provided to LwSciBufSysMemMemMap\n");
        LwSciCommonPanic();
    }

    /* print input parameters */
    LWSCI_INFO("Input: context: %p\trmHandle.memHandle: %u\t"
        "offset: %lu\tlen: %lu\taccPerm: %u\tptr: %p\n", context,
            rmHandle.memHandle, offset, len, accPerm, ptr);

    sciErr = LwSciBufAllocCommonTegraMemMap(rmHandle.memHandle, offset, len,
            accPerm, ptr);
    if (LwSciError_Success != sciErr) {
        LWSCI_ERR_STR("LwRmMemMap failed\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    /* print output parameters */
    LWSCI_INFO("Output: *ptr: %p\n", *ptr);

ret:
    LWSCI_FNEXIT("");
    return sciErr;
}

LwSciError LwSciBufSysMemMemUnMap(
    const void* context,
    LwSciBufRmHandle rmHandle,
    void* ptr,
    uint64_t size)
{
    LwSciError sciErr = LwSciError_Success;
    const LwSciBufAllocSysMemContext* sysMemContext = NULL;

    LWSCI_FNENTRY("");

    /* verify input parameters */
    if ((NULL == context) || (0U == rmHandle.memHandle) || (NULL == ptr)
        || (0U == size)) {
        LWSCI_ERR_STR("Bad parameter supplied to LwSciBufSysMemMemUnMap\n");
        LWSCI_ERR_UINT("rmHandle.memHandle: \t", rmHandle.memHandle);
        LWSCI_ERR_ULONG("size: \n", size);
        LwSciCommonPanic();
    }

    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
    sysMemContext = (const LwSciBufAllocSysMemContext*)context;
    if (LWSCIBUF_SYSMEM_CONTEXT_MAGIC != sysMemContext->magic) {
        LWSCI_ERR_STR("Invalid context parameter provided to LwSciBufSysMemMemUnMap\n");
        LwSciCommonPanic();
    }

    /* print input parameters */
    LWSCI_INFO("Input: context: %p\trmHandle.memHandle: %u\t"
        "ptr: %p\tsize: %lu\n", context, rmHandle.memHandle, ptr,
        size);

    sciErr = LwSciBufAllocCommonTegraMemUnmap(rmHandle.memHandle, ptr, size);
    if (LwSciError_Success != sciErr) {
        LWSCI_ERR_STR("LwSciBufAllocCommonTegraMemUnmap failed\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

ret:
    LWSCI_FNEXIT("");
    return sciErr;
}

LwSciError LwSciBufSysMemGetSize(
    const void* context,
    LwSciBufRmHandle rmHandle,
    uint64_t* size)
{
    LwSciError sciErr = LwSciError_Success;
    const LwSciBufAllocSysMemContext* sysMemContext = NULL;

    LWSCI_FNENTRY("");

    /* verify input parameters */
    if ((NULL == context) || (NULL == size) || (0U == rmHandle.memHandle)) {
        LWSCI_ERR_STR("Bad parameter supplied to LwSciBufSysMemGetSize\n");
        LWSCI_ERR_UINT("rmHandle.memHandle: \n", rmHandle.memHandle);
        LwSciCommonPanic();
    }

    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
    sysMemContext = (const LwSciBufAllocSysMemContext*)context;
    if (LWSCIBUF_SYSMEM_CONTEXT_MAGIC != sysMemContext->magic) {
        LWSCI_ERR_STR("Invalid context parameter provided to LwSciBufSysMemGetSize\n");
        LwSciCommonPanic();
    }

    /* print input parameters */
    LWSCI_INFO("Input: context: %p\trmHandle.memHandle: %u\tsize: %p\n",
        context, rmHandle.memHandle, size);

    sciErr = LwSciBufAllocCommonTegraGetMemSize(rmHandle.memHandle, size);
    if (LwSciError_Success != sciErr) {
        LWSCI_ERR_STR("LwSciBufAllocCommonTegraGetMemSize failed\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    /* print output parameters */
    LWSCI_INFO("Output: buf size: %lu\n", *size);

ret:
    LWSCI_FNEXIT("");
    return sciErr;
}

#if (LW_IS_SAFETY == 0)
LwSciError LwSciBufSysMemGetAlignment(
    const void* context,
    LwSciBufRmHandle rmHandle,
    uint64_t* alignment)
{
    LwSciError err = LwSciError_Success;

    (void)context;
    (void)rmHandle;
    (void)alignment;

    LWSCI_FNENTRY("");

    LWSCI_FNEXIT("");
    return err;
}

LwSciError LwSciBufSysMemGetHeapType(
    const void* context,
    LwSciBufRmHandle rmHandle,
    void* heapType)
{
    LwSciError err = LwSciError_Success;

    (void)context;
    (void)rmHandle;
    (void)heapType;

    LWSCI_FNENTRY("");

    LWSCI_FNEXIT("");
    return err;
}
#endif

LwSciError LwSciBufSysMemCpuCacheFlush(
    const void* context,
    LwSciBufRmHandle rmHandle,
    void* cpuPtr,
    uint64_t len)
{
    LwSciError sciErr = LwSciError_Success;
    const LwSciBufAllocSysMemContext* sysMemContext = NULL;

    LWSCI_FNENTRY("");

    /* verify input parameters */
    if ((NULL == context) || (0U == rmHandle.memHandle) ||
        (NULL == cpuPtr) || (0U == len)) {
        LWSCI_ERR_STR("Bad parameter supplied to LwSciBufSysMemGetSize\n");
        LWSCI_ERR_UINT("rmHandle.memHandle: \t", rmHandle.memHandle);
        LWSCI_ERR_ULONG("len: \n", len);
        LwSciCommonPanic();
    }

    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
    sysMemContext = (const LwSciBufAllocSysMemContext*)context;
    if (LWSCIBUF_SYSMEM_CONTEXT_MAGIC != sysMemContext->magic) {
        LWSCI_ERR_STR("Invalid context parameter provided to LwSciBufSysMemCpuCacheFlush\n");
        LwSciCommonPanic();
    }

    /* print input parameters */
    LWSCI_INFO("Input: context: %p\trmHandle.memHandle: %u\t"
               "cpuPtr: %p\tlen:%lu\n",
            context, rmHandle.memHandle, cpuPtr, len);

    sciErr = LwSciBufAllocCommonTegraCpuCacheFlush(rmHandle.memHandle, cpuPtr,
            len);
    if (LwSciError_Success != sciErr) {
        LWSCI_ERR_STR("Failed to do CPU Cache Flush.\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    /* print output parameters */
    LWSCI_INFO("Successfully completed LwRm CPU Cache flush.\n");

ret:
    LWSCI_FNEXIT("");
    return sciErr;
}

void LwSciBufSysMemClose(
    void* context)
{
    LWSCI_FNENTRY("");

    LwSciCommonFree(context);

    LWSCI_FNEXIT("");
}

static void LwSciBufTestGpuPresence(
    const LwSciBufPerGpuContext* perGpuContext,
    size_t gpuListSize,
    const LwSciRmGpuId* gpuIdArray,
    uint64_t numGpuIds,
    uint32_t* iGpuPresent,
    uint32_t* dGpuPresent)
{
    size_t devIdx = 0U;
    uint64_t gpuIdx = 0U;
    const LwRmGpuDeviceInfo* gpuDeviceInfo = NULL;
    LWSCI_FNENTRY("");

    *iGpuPresent = 0U;
    *dGpuPresent = 0U;

    // For all initialized GPUs
    for (devIdx = 0U; devIdx < gpuListSize; devIdx++) {
        gpuDeviceInfo = perGpuContext[devIdx].gpuDeviceInfo;

        // For all GPU ids provided by application
        for (gpuIdx = 0U; gpuIdx < numGpuIds; gpuIdx++) {

            if (LwSciCommonMemcmp(&gpuDeviceInfo->deviceId.gid,
                &gpuIdArray[gpuIdx], sizeof(LwRmGpuDeviceGID)) == 0) {

                if (LwRmGpuType_iGPU == gpuDeviceInfo->gpuType) {
                    *iGpuPresent |= 0x1U;
                } else {
                    *dGpuPresent |= 0x1U;
                }
            }
        }
    }
    LWSCI_FNEXIT("");
}

static void LwSciBufHasXGpu(
    const LwSciBufAllGpuContext* allGpuContext,
    const LwSciRmGpuId* gpuIdArray,
    uint64_t numGpuIds,
    uint32_t* iGpuPresent,
    uint32_t* dGpuPresent)
{
    LWSCI_FNENTRY("");

    LWSCI_INFO("Input: gpuIdArray: %p numGpuIds: %u iGpuPresent: %p"
                " dGpuPresent: %p\n", gpuIdArray, numGpuIds, iGpuPresent,
                dGpuPresent);

    LwSciBufTestGpuPresence(allGpuContext->perGpuContext,
                                    allGpuContext->gpuListSize, gpuIdArray,
                                    numGpuIds, iGpuPresent, dGpuPresent);

    LWSCI_INFO("Output: iGpuPresent: %"PRIu32" dGpuPresent: %"PRIu32"\n",
        *iGpuPresent, *dGpuPresent);

    LWSCI_FNEXIT("");
}

LwSciError LwSciBufSysMemGetAllocContext(
    const void* allocContextParam,
    void* openContext,
    void** allocContext)
{
    LwSciError err = LwSciError_Success;
    LwSciBufAllocSysMemContext* sysMemContext = NULL;
    const LwSciBufAllocSysMemAllocContextParam* sysMemAllocContextParams = NULL;
    uint32_t iGpuPresent = 0U, dGpuPresent = 0U;
    uint32_t gpuAccessBitset = 0U;

    LWSCI_FNENTRY("");

    if ((NULL == allocContextParam) || (NULL == openContext) ||
        (NULL == allocContext)) {
        LWSCI_ERR_STR("Bad parameter supplied to LwSciBufSysMemGetAllocContext\n");
        LwSciCommonPanic();
    }

    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
    sysMemContext = (LwSciBufAllocSysMemContext*)openContext;
    if (LWSCIBUF_SYSMEM_CONTEXT_MAGIC != sysMemContext->magic) {
        LWSCI_ERR_STR("Invalid openContext parameter provided to"
                  " LwSciBufSysMemGetAllocContext\n");
        LwSciCommonPanic();
    }

    LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
    sysMemAllocContextParams =
        (const LwSciBufAllocSysMemAllocContextParam*)allocContextParam;
    LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 11_5))

    LwSciBufHasXGpu(sysMemContext->allGpuContext,
        sysMemAllocContextParams->gpuIds,
        sysMemAllocContextParams->gpuIdsCount, &iGpuPresent, &dGpuPresent);

    if (1U == iGpuPresent) {
        gpuAccessBitset |= (uint32_t)LwSciBufAllocSysGpuAccess_iGPU;
    }

    if (1U == dGpuPresent) {
        gpuAccessBitset |= (uint32_t)LwSciBufAllocSysGpuAccess_dGPU;
    }

    sysMemContext->gpuAccessParam = ColwertGpuAccessBitsetToGpuAccesTypeEnum(gpuAccessBitset);

    LWSCI_INFO("Input: allocContextParam: %p, openContext: %p,"
        " allocContext: %p\n", allocContextParam, openContext, allocContext);

    *allocContext = (void*)sysMemContext;

    LWSCI_FNEXIT("");
    return err;
}
