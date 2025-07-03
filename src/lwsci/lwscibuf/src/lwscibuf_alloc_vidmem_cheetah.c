/*
 * Copyright (c) 2018-2021, LWPU CORPORATION. All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#include "lwscibuf_utils.h"
#include "lwscibuf_alloc_vidmem_tegra_priv.h"
#include "lwscibuf_alloc_common_tegra.h"

static LwSciError LwSciBufAllocVidMemToVidMemAllocVal(
    LwSciBufAllocVidMemVal* lwSciBufVidMemAllocVal,
    LwRmGpuDeviceAllocateMemoryAttr* vidMemAllocVal,
    const LwSciBufPerGpuContext* perGpuContext)
{
    LwSciError err = LwSciError_Success;

    LWSCI_FNENTRY("");

    LWSCI_INFO("Input: vSciBufVidMemAllocVal: %p, vidMemAllocVal: %p\n",
        lwSciBufVidMemAllocVal, vidMemAllocVal);

    /**
     * LwGpu supports allocation in multiple of small page size. Align the size
     * retrieved from LwSciBuf to LwGpu small page size.
     */
    err = LwSciBufAliglwalue64(lwSciBufVidMemAllocVal->size,
            perGpuContext->gpuDeviceInfo->smallPageSize,
            &lwSciBufVidMemAllocVal->size);
    if (err != LwSciError_Success) {
        LWSCI_ERR_STR("Could not align buffer size obtained from LwSciBuf to LwGpu's small page size\n");
        goto ret;
    }

    vidMemAllocVal->alignment = lwSciBufVidMemAllocVal->alignment;
    LwSciBufAllocCommonTegraColwertCoherency(
        lwSciBufVidMemAllocVal->coherency, &vidMemAllocVal->cpuCoherency);

    vidMemAllocVal->cpuMappable = lwSciBufVidMemAllocVal->cpuMapping;

ret:
    LWSCI_FNEXIT("");
    return err;
}

LwSciError LwSciBufVidMemOpen(
    LwSciBufDev devHandle,
    void** context)
{
    LwSciError err = LwSciError_Success;
    LwSciBufAllocVidMemContext* gpuContext = NULL;

    LWSCI_FNENTRY("");

    if (context == NULL) {
        LWSCI_ERR_STR("Bad parameter supplied to LwSciBufVidMemOpen\n");
        err = LwSciError_BadParameter;
        goto ret;
    }

    LWSCI_INFO("Input: context ptr: %p\n", context);

    /* initialize output parameter */
    *context = NULL;

    gpuContext = LwSciCommonCalloc(1, sizeof(LwSciBufAllocVidMemContext));
    if (gpuContext == NULL) {
        LWSCI_ERR_STR("Could not allocate memory for LwSciBufAllocVidMemContext\n");
        err = LwSciError_InsufficientMemory;
        goto ret;
    }

    gpuContext->magic = LW_SCI_BUF_GPU_CONTEXT_MAGIC;

    LwSciBufDevGetAllGpuContext(devHandle, &gpuContext->allGpuContext);

    *context = (void*)gpuContext;

    goto ret;

ret:
    LWSCI_FNEXIT("");
    return err;
}

LwSciError LwSciBufVidMemAlloc(
    const void* context,
    void* allocVal,
    LwSciBufDev devHandle,
    LwSciBufRmHandle* rmHandle)
{
    LwSciError err = LwSciError_Success;
    LwSciBufAllocVidMemVal* lwSciBufVidMemAllocVal = NULL;
    const LwSciBufPerGpuContext* perGpuContext = NULL;
    LWRM_GPU_DEFINE_DEVICE_ALLOCATE_MEMORY_ATTR(vidMemAllocVal);
    LwError lwErr = LwError_Success;

    (void)devHandle;

    LWSCI_FNENTRY("");

    if (context == NULL || allocVal == NULL || rmHandle == NULL) {
        LWSCI_ERR_STR("Bad parameter supplied to LwSciBufVidMemAlloc\n");
        err = LwSciError_BadParameter;
        goto ret;
    }

    LWSCI_INFO("Input: context: %p, allocVal: %p, rmHandle: %p\n", context,
        allocVal, rmHandle);

    perGpuContext = (const LwSciBufPerGpuContext*)context;

    lwSciBufVidMemAllocVal = (LwSciBufAllocVidMemVal*)allocVal;
    if (lwSciBufVidMemAllocVal->size == 0U) {
        LWSCI_ERR_STR("size provided to LwSciBufVidMemAlloc for allocation is 0!\n");
        err = LwSciError_BadParameter;
        goto ret;
    }

    err = LwSciBufAllocVidMemToVidMemAllocVal(lwSciBufVidMemAllocVal,
            &vidMemAllocVal, perGpuContext);
    if (err != LwSciError_Success) {
        LWSCI_ERR_STR("LwSciBufAllocVidMemToVidMemAllocVal failed\n");
        goto ret;
    }

    lwErr = LwRmGpuDeviceAllocateMemory(perGpuContext->gpuDevice,
                LwRmGpuDeviceMemoryLocation_Vidmem, lwSciBufVidMemAllocVal->size,
                LwRmMemTags_LwSci, &vidMemAllocVal, &rmHandle->memHandle);
    if (lwErr != LwError_Success) {
        err = LwSciError_ResourceError;
        LWSCI_ERR_INT("LwRmGpuDeviceAllocateMemory failed. LwError: \n", lwErr);
        goto ret;
    }

    LWSCI_INFO("Output: memHandle: %u\n", rmHandle->memHandle);

ret:
    LWSCI_FNEXIT("");
    return err;
}

LwSciError LwSciBufVidMemDealloc(
    void* context,
    LwSciBufRmHandle rmHandle)
{
    LwSciError err = LwSciError_Success;

    LWSCI_FNENTRY("");

    if (context == NULL || rmHandle.memHandle == 0U) {
        LWSCI_ERR_STR("Bad parameter supplied to LwSciBufVidMemDealloc\n");
        LWSCI_ERR_UINT("rmHandle.memHandle: \n", rmHandle.memHandle);
        err = LwSciError_BadParameter;
        goto ret;
    }

    LWSCI_INFO("Input: context: %p, rmHandle.memHandle: %u\n", context,
        rmHandle.memHandle);

    LwSciBufAllocCommonTegraMemFree(rmHandle.memHandle);

ret:
    LWSCI_FNEXIT("");
    return err;
}

LwSciError LwSciBufVidMemDupHandle(
    const void* context,
    LwSciBufAttrValAccessPerm newPerm,
    LwSciBufRmHandle rmHandle,
    LwSciBufRmHandle *dupRmHandle)
{
    LwSciError err = LwSciError_Success;

    LWSCI_FNENTRY("");

    if (context == NULL || rmHandle.memHandle == 0U || dupRmHandle == NULL) {
        LWSCI_ERR_STR("Bad parameter supplied to LwSciBufVidMemDupHandle\n");
        LWSCI_ERR_UINT("rmHandle.memHandle: \n", rmHandle.memHandle);
        err = LwSciError_BadParameter;
        goto ret;
    }

    LWSCI_INFO("Input: context: %p, rmHandle.memHandle: %u, dupRmHandle: %p\n",
        context, rmHandle.memHandle, dupRmHandle);

    err = LwSciBufAllocCommonTegraDupHandle(rmHandle.memHandle, newPerm,
            &dupRmHandle->memHandle);
    if (err != LwSciError_Success) {
        LWSCI_ERR_STR("LwSciBufAllocCommonTegraDupHandle failed\n");
        goto ret;
    }

    LWSCI_INFO("Output: dupRmHandle->memHandle: %u\n", dupRmHandle->memHandle);

ret:
    LWSCI_FNEXIT("");
    return err;
}

LwSciError LwSciBufVidMemMemMap(
    const void* context,
    LwSciBufRmHandle rmHandle,
    uint64_t offset,
    uint64_t len,
    LwSciBufAttrValAccessPerm accPerm,
    void **ptr)
{
    LwSciError err = LwSciError_Success;

    LWSCI_FNENTRY("");

    if (context == NULL || rmHandle.memHandle == 0U || len == 0U ||
        accPerm >= LwSciBufAccessPerm_Ilwalid || ptr == NULL) {
        LWSCI_ERR_STR("Bad parameter supplied to LwSciBufVidMemMemMap\n");
        LWSCI_ERR_UINT("rmHandle.memHandle: \t", rmHandle.memHandle);
        LWSCI_ERR_ULONG("offset: \t", offset);
        LWSCI_ERR_ULONG("len: \t", len);
        LWSCI_ERR_UINT("accPerm: \n", accPerm);
        err = LwSciError_BadParameter;
        goto ret;
    }

    LWSCI_INFO("Input: context: %p, rmHandle.memHandle: %u, offset: %lu, len: %lu, accPerm: %u, ptr: %p\n",
        context, rmHandle.memHandle, offset, len, accPerm, ptr);

    err = LwSciBufAllocCommonTegraMemMap(rmHandle.memHandle, offset, len,
            accPerm, ptr);
    if (err != LwSciError_Success) {
        LWSCI_ERR_STR("LwRmMemMap failed\n");
        goto ret;
    }

    LWSCI_INFO("Output: *ptr: %p\n", *ptr);

ret:
    LWSCI_FNEXIT("");
    return err;
}

LwSciError LwSciBufVidMemMemUnMap(
    const void* context,
    LwSciBufRmHandle rmHandle,
    void* ptr,
    uint64_t size)
{
    LwSciError err = LwSciError_Success;

    LWSCI_FNENTRY("");

    if (context == NULL || rmHandle.memHandle == 0U || ptr == NULL
        || size == 0U) {
        LWSCI_ERR_STR("Bad parameter supplied to LwSciBufVidMemMemUnMap\n");
        LWSCI_ERR_UINT("rmHandle.memHandle: \t", rmHandle.memHandle);
        LWSCI_ERR_ULONG("size: \n", size);
        err = LwSciError_BadParameter;
        goto ret;
    }

    LWSCI_INFO("Input: context: %p, rmHandle.memHandle: %u, ptr: %p, size: %lu\n",
        context, rmHandle.memHandle, ptr, size);

    err = LwSciBufAllocCommonTegraMemUnmap(rmHandle.memHandle, ptr, size);
    if (err != LwSciError_Success) {
        LWSCI_ERR_STR("LwSciBufAllocCommonTegraMemUnmap failed\n");
        goto ret;
    }

ret:
    LWSCI_FNEXIT("");
        return err;
}

LwSciError LwSciBufVidMemGetSize(
    const void* context,
    LwSciBufRmHandle rmHandle,
    uint64_t* size)
{
    LwSciError err = LwSciError_Success;

    LWSCI_FNENTRY("");

    if (context == NULL || size == NULL || rmHandle.memHandle == 0U) {
        err = LwSciError_BadParameter;
        LWSCI_ERR_STR("Bad parameter supplied to LwSciBufVidMemGetSize\n");
        LWSCI_ERR_UINT("rmHandle.memHandle: \n", rmHandle.memHandle);
        goto ret;
    }

    LWSCI_INFO("Input: context: %p, rmHandle.memHandle: %u, size: %p\n",
        context, rmHandle.memHandle, size);

    err = LwSciBufAllocCommonTegraGetMemSize(rmHandle.memHandle, size);
    if (err != LwSciError_Success) {
        LWSCI_ERR_STR("LwSciBufAllocCommonTegraGetMemSize failed\n");
        goto ret;
    }

    LWSCI_INFO("Output: buf size: %lu\n", *size);

ret:
    LWSCI_FNEXIT("");
    return err;
}

LwSciError LwSciBufVidMemGetAlignment(
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

LwSciError LwSciBufVidMemGetHeapType(
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

LwSciError LwSciBufVidMemCpuCacheFlush(
    const void* context,
    LwSciBufRmHandle rmHandle,
    void* cpuPtr,
    uint64_t len)
{
    LwSciError err = LwSciError_Success;

    LWSCI_FNENTRY("");

    if (context == NULL || rmHandle.memHandle == 0U ||
        cpuPtr == NULL || len == 0U) {
        err = LwSciError_BadParameter;
        LWSCI_ERR_STR("Bad parameter supplied to LwSciBufVidMemCpuCacheFlush\n");
        LWSCI_ERR_UINT("rmHandle.memHandle: \t", rmHandle.memHandle);
        LWSCI_ERR_ULONG("len: \n", len);
        goto ret;
    }

    LWSCI_INFO("Input: context: %p, rmHandle.memHandle: %u, cpuPtr: %p, len:%lu\n",
        context, rmHandle.memHandle, cpuPtr, len);

    err = LwSciBufAllocCommonTegraCpuCacheFlush(rmHandle.memHandle, cpuPtr,
            len);
    if (err != LwSciError_Success) {
        LWSCI_ERR_STR("Failed to do CPU Cache Flush.\n");
        goto ret;
    }

    LWSCI_INFO("Successfully completed LwRm CPU Cache flush.\n");

ret:
    LWSCI_FNEXIT("");
    return err;
}

void LwSciBufVidMemClose(
    void* context)
{
    LwSciBufAllocVidMemContext* gpuContext = NULL;

    LWSCI_FNENTRY("");

    if (context == NULL) {
        LWSCI_ERR_STR("Bad parameter supplied to LwSciBufVidMemClose\n");
        goto ret;
    }

    LWSCI_INFO("Input: context ptr: %p\n", context);

    gpuContext = (LwSciBufAllocVidMemContext*)context;

    LwSciCommonFree(gpuContext);

ret:
    LWSCI_FNEXIT("");
}

LwSciError LwSciBufVidMemGetAllocContext(
    const void* allocContextParam,
    void* openContext,
    void** allocContext)
{
    LwSciError err = LwSciError_Success;
    LwSciBufAllocVidMemContext* vidMemContext = NULL;
    const LwSciBufAllGpuContext* allGpuContext = NULL;
    const LwSciBufAllocVidMemAllocContextParam* vidMemAllocContextParam = NULL;
    const LwRmGpuDeviceInfo* gpuDeviceInfo = NULL;
    size_t i = 0;

    LWSCI_FNENTRY("");

    if (allocContextParam == NULL || openContext == NULL ||
        allocContext == NULL) {
        LWSCI_ERR_STR("Bad parameter supplied to LwSciBufSysMemGetAllocContext\n");
        err = LwSciError_Success;
        goto ret;
    }

    vidMemContext = (LwSciBufAllocVidMemContext*)openContext;
    if (vidMemContext->magic != (uint32_t)LW_SCI_BUF_GPU_CONTEXT_MAGIC) {
        LWSCI_ERR_STR("Invalid openContext parameter supplied to LwSciBufVidMemGetAllocContext\n");
        err = LwSciError_BadParameter;
        goto ret;
    }

    allGpuContext = vidMemContext->allGpuContext;

    *allocContext = NULL;

    vidMemAllocContextParam =
        (const LwSciBufAllocVidMemAllocContextParam*)allocContextParam;

    for (i = 0U; i < allGpuContext->gpuListSize; i++) {
        gpuDeviceInfo = allGpuContext->perGpuContext[i].gpuDeviceInfo;
        if (gpuDeviceInfo == NULL) {
            continue;
        }

        if (LwSciCommonMemcmp(&gpuDeviceInfo->deviceId.gid,
            &vidMemAllocContextParam->gpuId, sizeof(LwRmGpuDeviceGID)) == 0) {
            *allocContext = &allGpuContext->perGpuContext[i];
            goto ret;
        }
    }

    /* we are here implies that we could not find matching GPU UUID passed via
     * allocContext with all the GPUs that are present in GPU context
     */
    err = LwSciError_ResourceError;
    LWSCI_ERR_STR("GPU with below UUID not initialized.\n");
    LWSCI_ERR_STR("Error might have oclwrred during opening of devices in LwSciBufVidMemOpen call\n");
    for (i = 0; i < sizeof(LwRmGpuDeviceGID)/sizeof(uint8_t); i++) {
        LWSCI_ERR_ULONG("byte[]: ",i);
        LWSCI_ERR_UINT("\n", vidMemAllocContextParam->gpuId.bytes[i]);
    }

ret:
    LWSCI_FNEXIT("");
    return err;
}
