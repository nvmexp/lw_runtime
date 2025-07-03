/*
 * Copyright (c) 2019-2021, LWPU CORPORATION. All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#include "lwscibuf_alloc_sysmem_x86_priv.h"

void LwSciBufAllocSysMemPrintHeapTypes(
    LwSciBufAllocSysMemHeapType* heaps,
    uint32_t numHeaps)
{
    (void)heaps;
    (void)numHeaps;

    LWSCI_FNENTRY("");

    LWSCI_FNEXIT("");
}

static LwSciError LwSciBufSetAllocParams(
    LwSciBufAllocSysMemVal* sysMemAllocVal,
    LwRmSysMemAllocVal* lwAllocParams)
{
    LwSciError sciErr = LwSciError_Success;

    LWSCI_FNENTRY("");

    /* verify input parameters */
    if (sysMemAllocVal == NULL || lwAllocParams == NULL) {
        sciErr = LwSciError_BadParameter;
        LWSCI_ERR("Bad parameter supplied to getOS32AttrFlags\n");
        LWSCI_ERR("sysMemAllocVal: %p\tlwAllocParams: %p\n",
                    sysMemAllocVal, lwAllocParams);
        goto ret;
    }

    /* print input parameters */
    LWSCI_INFO("Input: sysMemAllocVal: %p\tlwAllocParams: %p\n",
                sysMemAllocVal, lwAllocParams);

    lwAllocParams->location = LwRmShimMemLocation_PCI;

    if (sysMemAllocVal->coherency == true) {
        lwAllocParams->cacheCoherency = LwRmShimCacheCoherency_WriteBack;
    } else {
        lwAllocParams->cacheCoherency = LwRmShimCacheCoherency_WriteCombine;
    }

    LWSCI_INFO("Output: location: %u, cache: %u\n",
                lwAllocParams->location, lwAllocParams->cacheCoherency);

ret:
    LWSCI_FNEXIT("");
    return sciErr;
}

static LwSciError LwSciBufSetMapParams(
    LwSciBufResmanCpuMappingParam* resmanCpuMappingParam,
    LwResmanCpuMappingParam* lwResmanCpuMappingParam)
{
    LwSciError sciErr = LwSciError_Success;

    LWSCI_FNENTRY("");

    /* verify input parameters */
    if (resmanCpuMappingParam == NULL || lwResmanCpuMappingParam == NULL) {
        sciErr = LwSciError_BadParameter;
        LWSCI_ERR("Bad parameter supplied\n");
        LWSCI_ERR("resmanCpuMappingParam: %p\tlwResmanCpuMappingParam: %p\n",
                    resmanCpuMappingParam, lwResmanCpuMappingParam);
        goto ret;
    }

    /* print input parameters */
    LWSCI_INFO("Input: resmanCpuMappingParam: %p\tlwResmanCpuMappingParam: %p\n",
                resmanCpuMappingParam, lwResmanCpuMappingParam);

    switch(resmanCpuMappingParam->redirection)
    {
        case LwSciBufAllocResmanRedirection_Default:
            lwResmanCpuMappingParam->redirection = LwRmShimMemMapping_Default;
            break;
        case LwSciBufAllocResmanRedirection_Direct:
            lwResmanCpuMappingParam->redirection = LwRmShimMemMapping_Direct;
            break;
        case LwSciBufAllocResmanRedirection_Reflected:
            lwResmanCpuMappingParam->redirection = LwRmShimMemMapping_Reflected;
            break;
        case LwSciBufAllocResmanRedirection_Ilwalid:
            break;
    }

    switch(resmanCpuMappingParam->access)
    {
        case LwSciBufAccessPerm_ReadWrite:
            lwResmanCpuMappingParam->access = LwRmShimMemAccess_ReadWrite;
            break;
        case LwSciBufAccessPerm_Readonly:
            lwResmanCpuMappingParam->access = LwRmShimMemAccess_ReadOnly;
            break;
        case LwSciBufAccessPerm_Auto:
        case LwSciBufAccessPerm_Ilwalid:
            break;
    }

    LWSCI_INFO("Output: redirection: %u, access: %u\n",
                lwResmanCpuMappingParam->redirection,
                lwResmanCpuMappingParam->access);

ret:
    LWSCI_FNEXIT("");
    return sciErr;
}

static LwSciError LwSciBufFillRmShimFvt(
    LwSciBufDev devHandle,
    LwSciBufAllocSysMemContext* sysMemContext)
{
    LwSciError sciErr = LwSciError_Success;
    allocMemFunc rmAllocMem = NULL;
    mapMemFunc rmMemMap = NULL;
    dupMemFunc rmDupMem = NULL;
    unmapMemFunc rmUnMapMem = NULL;
    freeMemFunc rmFreeMem = NULL;
    flushCpuCacheFunc rmFlushCpuCache = NULL;
    void* rmLib = NULL;

    LWSCI_FNENTRY("");

    /* verify input parameters */
    if (devHandle == NULL || sysMemContext == NULL) {
        LWSCI_ERR("Bad parameter supplied\n");
        LWSCI_ERR("devHandle: %p\tsysMemContext: %p\n",
                    devHandle, sysMemContext);
        sciErr = LwSciError_BadParameter;
        goto ret;
    }

    sciErr = LwSciBufDevGetRmLibrary(devHandle, &rmLib);
    if (sciErr != LwSciError_Success) {
        LWSCI_ERR("Failed to get RmShim Lib\n");
        goto ret;
    }

    rmAllocMem = dlsym(rmLib, "LwRmShimAllocMem");
    if (rmAllocMem == NULL) {
        sciErr = LwSciError_IlwalidState;
        LWSCI_ERR("Failed to get function \n");
        goto ret;
    }

    rmMemMap = dlsym(rmLib, "LwRmShimMapMemory");
    if (rmMemMap == NULL) {
        sciErr = LwSciError_IlwalidState;
        LWSCI_ERR("Failed to get function \n");
        goto ret;
    }

    rmDupMem = dlsym(rmLib, "LwRmShimDupMemContext");
    if (rmDupMem == NULL) {
        sciErr = LwSciError_IlwalidState;
        LWSCI_ERR("Failed to get function \n");
        goto ret;
    }

    rmUnMapMem = dlsym(rmLib, "LwRmShimUnMapMemory");
    if (rmUnMapMem == NULL) {
        sciErr = LwSciError_IlwalidState;
        LWSCI_ERR("Failed to get function \n");
        goto ret;
    }

    rmFlushCpuCache = dlsym(rmLib, "LwRmShimFlushCpuCache");
    if (rmFlushCpuCache == NULL) {
        sciErr = LwSciError_IlwalidState;
        LWSCI_ERR("Failed to get function \n");
        goto ret;
    }

    rmFreeMem = dlsym(rmLib, "LwRmShimFreeMem");
    if (rmFreeMem == NULL) {
        sciErr = LwSciError_IlwalidState;
        LWSCI_ERR("Failed to get function \n");
        goto ret;
    }

    sysMemContext->rmShimAllocFvt.rmAllocMem = rmAllocMem;
    sysMemContext->rmShimAllocFvt.rmMemMap = rmMemMap;
    sysMemContext->rmShimAllocFvt.rmDupMem = rmDupMem;
    sysMemContext->rmShimAllocFvt.rmUnMapMem = rmUnMapMem;
    sysMemContext->rmShimAllocFvt.rmFreeMem = rmFreeMem;
    sysMemContext->rmShimAllocFvt.rmFlushCpuCache = rmFlushCpuCache;

ret:
    LWSCI_FNEXIT("");
    return sciErr;
}

static LwSciError LwSciBufColwertRmShimContextToRmHandle(
    LwRmShimSessionContext* rmSession,
    LwRmShimDeviceContext* rmDevice,
    LwRmShimMemoryContext* rmMemory,
    LwSciBufRmHandle* rmHandle)
{
    LwSciError sciErr = LwSciError_Success;
    LWSCI_FNENTRY("");

    /* verify input parameters */
    if (rmSession == NULL || rmDevice == NULL || rmMemory == NULL ||
        rmHandle == NULL) {
        LWSCI_ERR("Bad parameter supplied\n");
        LWSCI_ERR("rmSession: %p\trmDevice: %p\trmMemory: %p\trmHandle: %p\n",
                rmSession, rmDevice, rmMemory, rmHandle);
        sciErr = LwSciError_BadParameter;
        goto ret;
    }

    /* print input parameters */
    LWSCI_INFO("Input: rmSession: %p\trmDevice: %p\trmMemory: %p"
        "\trmHandle: %p\n", rmSession, rmDevice, rmMemory, rmHandle);

    rmHandle->hClient = rmDevice->hClient;
    rmHandle->hDevice = rmDevice->hDevice;
    rmHandle->hMemory = rmMemory->pHandle;

    LWSCI_INFO("Input: hClient: %u\thDevice: %u\thMemory: %u\n",
        rmHandle->hClient, rmHandle->hDevice, rmHandle->hMemory);

ret:
    LWSCI_FNEXIT("");
    return sciErr;
}

static LwSciError LwSciBufColwertRmHandleToRmShimContext(
    LwSciBufRmHandle* rmHandle,
    LwRmShimSessionContext* rmSession,
    LwRmShimDeviceContext* rmDevice,
    LwRmShimMemoryContext* rmMemory)
{
    LwSciError sciErr = LwSciError_Success;
    LWSCI_FNENTRY("");

    /* verify input parameters */
    if (rmSession == NULL || rmDevice == NULL || rmMemory == NULL ||
        rmHandle == NULL) {
        LWSCI_ERR("Bad parameter supplied\n");
        LWSCI_ERR("rmSession: %p\trmDevice: %p\trmMemory: %p\trmHandle: %p\n",
                rmSession, rmDevice, rmMemory, rmHandle);
        sciErr = LwSciError_BadParameter;
        goto ret;
    }

    /* print input parameters */
    LWSCI_INFO("Input: rmSession: %p\trmDevice: %p\trmMemory: %p"
        "\trmHandle: %p\n", rmSession, rmDevice, rmMemory, rmHandle);

    rmDevice->hClient = rmHandle->hClient;
    rmDevice->hDevice = rmHandle->hDevice;
    rmMemory->pHandle = rmHandle->hMemory;

    LWSCI_INFO("Input: hClient: %u\thDevice: %u\thMemory: %u\n",
        rmSession->pHandle, rmDevice->pHandle, rmMemory->pHandle);

ret:
    LWSCI_FNEXIT("");
    return sciErr;
}

static LwSciError LwSciBufColwertRmHandleToRmShimMemoryContext(
    LwSciBufRmHandle* rmHandle,
    LwRmShimMemoryContext* rmMemory)
{
    LwSciError sciErr = LwSciError_Success;
    LWSCI_FNENTRY("");

    /* verify input parameters */
    if (rmMemory == NULL || rmHandle == NULL) {
        LWSCI_ERR("Bad parameter supplied\n");
        LWSCI_ERR("rmMemory: %p\trmHandle: %p\n", rmMemory, rmHandle);
        sciErr = LwSciError_BadParameter;
        goto ret;
    }

    /* print input parameters */
    LWSCI_INFO("Input: rmMemory: %p\trmHandle: %p\n", rmMemory, rmHandle);

    rmMemory->pHandle = rmHandle->hMemory;

    LWSCI_INFO("Input: hMemory: %u\n", rmMemory->pHandle);

ret:
    LWSCI_FNEXIT("");
    return sciErr;
}

LwSciError LwSciBufSysMemOpen(
    LwSciBufDev devHandle,
    void** context)
{
    LwSciError sciErr = LwSciError_Success;
    LwSciBufAllocSysMemContext* sysMemContext = NULL;
    uint32_t i = 0U;

    LWSCI_FNENTRY("");

    /* verify input parameters */
    if (devHandle == NULL || context == NULL) {
        LWSCI_ERR("Bad parameter supplied to LwSciBufSysMemOpen\n");
        LWSCI_ERR("devHandle: %p\tcontext: %p\n", devHandle, context);
        sciErr = LwSciError_BadParameter;
        goto ret;
    }

    /* print input parameters */
    LWSCI_INFO("Input: devHandle: %p\tcontext: %p\n", devHandle, context);

    *context = LwSciCommonCalloc(1, sizeof(LwSciBufAllocSysMemContext));
    if (*context == NULL) {
        LWSCI_ERR("Could not allocate memory for LwSciBufAllocSysMemContext\n");
        sciErr = LwSciError_InsufficientMemory;
        goto ret;
    }
    sysMemContext = (LwSciBufAllocSysMemContext*)(*context);

    sciErr = LwSciBufFillRmShimFvt(devHandle, sysMemContext);
    if (sciErr != LwSciError_Success) {
        LWSCI_ERR("Failed to get RmShim Fvt\n");
        goto free_context;
    }

    sciErr = LwSciBufDevGetRmSessionDevice(devHandle,
                        &sysMemContext->rmSessionPtr,
                        &sysMemContext->rmDevicePtr);
    if (sciErr != LwSciError_Success) {
        LWSCI_ERR("LwSciBufGetRmSessionDevice failed\n");
        goto free_context;
    }

    sysMemContext->perGpuContext = LwSciCommonCalloc(
                                    sysMemContext->rmSessionPtr->numGpus,
                                    sizeof(LwSciBufSysMemPerGpuContext));
    if (sysMemContext->perGpuContext == NULL) {
        LWSCI_ERR("Could not allocate memory for LwSciBufSysMemPerGpuContext"
               " struct\n");
        sciErr = LwSciError_InsufficientMemory;
        goto free_context;
    }

    for (i = 0U; i < sysMemContext->rmSessionPtr->numGpus; i++) {
        sysMemContext->perGpuContext[i].sysMemContextPtr = sysMemContext;
        sysMemContext->perGpuContext[i].rmDevicePtr = &sysMemContext->rmDevicePtr[i];
    }

    /* print output parameters */
    LWSCI_INFO("Output: sysMemContext: %p\thClient: %u\thDevice: %u\n",
        *context, sysMemContext->rmSessionPtr->pHandle,
        sysMemContext->rmDevicePtr->pHandle);

    /* All opeartions are successful. Directly jump to 'ret' from here */
    goto ret;

free_context:
    LwSciCommonFree(*context);
ret:
    LWSCI_FNEXIT("");
    return sciErr;
}

void LwSciBufSysMemClose(
    void* context)
{
    LwSciBufAllocSysMemContext* sysMemContext =
                (LwSciBufAllocSysMemContext*)context;

    LWSCI_FNENTRY("");

    if (sysMemContext == NULL) {
        LWSCI_ERR("Context to close is NULL.\n");
        goto ret;
    }

    /* print input parameters */
    LWSCI_INFO("Input: context: %p\n", sysMemContext);

    LwSciCommonFree(sysMemContext->perGpuContext);
    LwSciCommonFree(context);
ret:
    LWSCI_FNEXIT("");
}

LwSciError LwSciBufSysMemAlloc(
    const void* context,
    void* allocVal,
    const LwSciBufDev devHandle,
    LwSciBufRmHandle* rmHandle)
{
    LwSciError sciErr = LwSciError_Success;
    LwSciBufAllocSysMemVal* sysMemAllocVal = (LwSciBufAllocSysMemVal*)allocVal;
    LwSciBufAllocSysMemContext* sysMemContext = NULL;
    const LwSciBufSysMemPerGpuContext* perGpuContext = (const LwSciBufSysMemPerGpuContext*)context;
    LwRmSysMemAllocVal lwRmSysMemAllocVal = {0};
    LwRmShimMemoryContext rmMemory = {0};
    LwRmShimError errShim = LWRMSHIM_OK;
    LwRmShimAllocMemParams memParams;
    uint64_t attr[2] = {0, 0};

    LWSCI_FNENTRY("");

    /* verify input parameters */
    if (perGpuContext == NULL || sysMemAllocVal == NULL) {
        sciErr = LwSciError_BadParameter;
        LWSCI_ERR("Bad parameter supplied to LwSciBufSysMemAlloc\n");
        LWSCI_ERR("context: %p\tsysMemAllocVal: %p\n", context, sysMemAllocVal);
        goto ret;
    }

    if (sysMemAllocVal->size == 0U || devHandle == NULL || rmHandle == NULL) {
        sciErr = LwSciError_BadParameter;
        LWSCI_ERR("Bad parameter supplied to LwSciBufSysMemAlloc\n");
        LWSCI_ERR("sysMemAllocVal->size %u\t"
            "devHandle: %p\trmHandle: %p\n",
            sysMemAllocVal->size, devHandle, rmHandle);
        goto ret;
    }

    /* print input parameters */
    LWSCI_INFO("Input: context: %p\tsysMemAllocVal: %p\tsysMemAllocVal->size: %u\t"
        "sysMemAllocVal->alignment: %u\tsysMemAllocVal->coherency: %u\t"
        "devHandle: %p\trmHandle: %p\n",
        context, sysMemAllocVal, sysMemAllocVal->size, sysMemAllocVal->alignment,
        sysMemAllocVal->coherency, devHandle, rmHandle);

    sysMemContext =  perGpuContext->sysMemContextPtr;

    sciErr = LwSciBufSetAllocParams(sysMemAllocVal,
                &lwRmSysMemAllocVal);
    if (sciErr != LwSciError_Success) {
        LWSCI_ERR("LwSciBufSetAllocParams failed\n");
        goto ret;
    }

    memset(&memParams, 0x00, sizeof(memParams));
    memParams.size = sysMemAllocVal->size;
    memParams.alignment = sysMemAllocVal->alignment;
    memParams.offset = 0;
    attr[0] = (uint64_t)lwRmSysMemAllocVal.location;
    attr[1] = (uint64_t)lwRmSysMemAllocVal.cacheCoherency;
    memParams.attr = &attr[0];
    memParams.numAttr = 2U;

    errShim = sysMemContext->rmShimAllocFvt.rmAllocMem(
                sysMemContext->rmSessionPtr,
                perGpuContext->rmDevicePtr,
                &rmMemory,
                &memParams);
    if (errShim != LWRMSHIM_OK) {
        sciErr = LwRmShimErrorToLwSciError(errShim);
        LWSCI_ERR("Allocation failed\n");
        goto ret;
    }

    sciErr = LwSciBufColwertRmShimContextToRmHandle(sysMemContext->rmSessionPtr,
                perGpuContext->rmDevicePtr, &rmMemory,
                rmHandle);
    if (sciErr != LwSciError_Success) {
        LWSCI_ERR("Failed to colwert RmShim to RmHandle\n");
        goto ret;
    }

    /* print output parameters */
    LWSCI_INFO("Output: resman hMem Handle: %u\n", rmHandle->hMemory);

ret:
    LWSCI_FNEXIT("");
    return sciErr;
}

LwSciError LwSciBufSysMemDealloc(
    void* context,
    LwSciBufRmHandle rmHandle)
{
    LwSciError sciErr = LwSciError_Success;
    LwRmShimError errShim = LWRMSHIM_OK;
    LwSciBufAllocSysMemContext* sysMemContext = NULL;
    LwSciBufSysMemPerGpuContext* perGpuContext = (LwSciBufSysMemPerGpuContext*)context;
    LwRmShimMemoryContext rmMemory = {0};

    LWSCI_FNENTRY("");

    /* verify input parameters */
    if (perGpuContext == NULL || rmHandle.hClient == 0U ||
        rmHandle.hDevice == 0U || rmHandle.hMemory == 0U) {
        LWSCI_ERR("Bad parameter supplied to LwSciBufSysMemDealloc\n");
        LWSCI_ERR("context: %p\thclient: %u\thDevice: %u\thMemory: %u\n",
            perGpuContext, rmHandle.hClient,
            rmHandle.hDevice, rmHandle.hMemory);
        sciErr = LwSciError_BadParameter;
        goto ret;
    }

    sysMemContext =  perGpuContext->sysMemContextPtr;

    sciErr = LwSciBufColwertRmHandleToRmShimMemoryContext(&rmHandle, &rmMemory);
    if (sciErr != LwSciError_Success) {
        LWSCI_ERR("Failed to colwert RmHandle to memcontext\n");
        goto ret;
    }

    errShim = sysMemContext->rmShimAllocFvt.rmFreeMem(
                sysMemContext->rmSessionPtr,
                perGpuContext->rmDevicePtr,
                &rmMemory);
    if (errShim != LWRMSHIM_OK) {
        sciErr = LwRmShimErrorToLwSciError(errShim);
        LWSCI_ERR("Deallocation failed\n");
        goto ret;
    }

ret:
    LWSCI_FNEXIT("");
    return sciErr;
}

LwSciError LwSciBufSysMemDupHandle(
    const void* context,
    LwSciBufAttrValAccessPerm newPerm,
    LwSciBufRmHandle rmHandle,
    LwSciBufRmHandle *dupRmHandle)
{
    LwSciError sciErr = LwSciError_Success;

    const LwSciBufAllocSysMemContext* sysMemContext = NULL;
    const LwSciBufSysMemPerGpuContext* perGpuContext =
            (const LwSciBufSysMemPerGpuContext*)context;
    LwRmShimError errShim = LWRMSHIM_OK;
    LwRmShimDupMemContextParams dupMemParams;
    LwRmShimSessionContext ipSession = {0};
    LwRmShimDeviceContext ipDevice = {0};
    LwRmShimMemoryContext ipMemory = {0};
    LwRmShimMemoryContext rmMemory = {0};

    (void)newPerm;

    LWSCI_FNENTRY("");

    /* verify input parameters */
    if (perGpuContext == NULL || rmHandle.hClient == 0U || rmHandle.hDevice == 0U ||
        rmHandle.hMemory == 0U || dupRmHandle == NULL) {
        LWSCI_ERR("Bad parameter supplied to LwSciBufSysMemDupHandle\n");
        LWSCI_ERR("context: %p\thClient: %u\thDevice: %u\t"
            "hMemory: %u\tdupRmHandle: %p\n", perGpuContext, rmHandle.hClient,
            rmHandle.hDevice, rmHandle.hMemory, dupRmHandle);
        sciErr = LwSciError_Success;
        goto ret;
    }

    sysMemContext =  perGpuContext->sysMemContextPtr;

    sciErr = LwSciBufColwertRmHandleToRmShimContext(&rmHandle, &ipSession,
                &ipDevice, &ipMemory);
    if (sciErr != LwSciError_Success) {
        LWSCI_ERR("Failed to colwert RmShim to RmHandle\n");
        goto ret;
    }

    memset(&dupMemParams, 0x00, sizeof(dupMemParams));
    dupMemParams.dupMemory = &rmMemory;
    dupMemParams.dupSession = sysMemContext->rmSessionPtr;
    dupMemParams.dupDevice = perGpuContext->rmDevicePtr;
    uint64_t attr[1] = {0};
    dupMemParams.attr = &attr[0];
    dupMemParams.numAttr = 0U;

    errShim = sysMemContext->rmShimAllocFvt.rmDupMem(
                &ipSession,
                &ipDevice,
                &ipMemory,
                &dupMemParams);
    if (errShim != LWRMSHIM_OK) {
        sciErr = LwRmShimErrorToLwSciError(errShim);
        LWSCI_ERR("Duplication failed\n");
        goto ret;
    }

    sciErr = LwSciBufColwertRmShimContextToRmHandle(sysMemContext->rmSessionPtr,
                perGpuContext->rmDevicePtr, &rmMemory,
                dupRmHandle);
    if (sciErr != LwSciError_Success) {
        LWSCI_ERR("Failed to colwert RmShim to RmHandle\n");
        goto ret;
    }

    /* print output parameters */
    LWSCI_INFO("Output: dupRmHandle->hMemory: %u\n", dupRmHandle->hMemory);

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
    void** ptr)
{
    LwSciError sciErr = LwSciError_Success;
    LwSciBufResmanCpuMappingParam resmanMappingParam = {0};
    LwResmanCpuMappingParam lwResmanCPUMappingParam = {0};
    const LwSciBufAllocSysMemContext* sysMemContext = NULL;
    const LwSciBufSysMemPerGpuContext* perGpuContext =
            (const LwSciBufSysMemPerGpuContext*)context;
    LwRmShimError errShim = LWRMSHIM_OK;
    LwRmShimMemMapParams mapParam = {0};
    LwRmShimMemoryContext rmMemory = {0};
    uint64_t attr[2] = {0, 0};

    LWSCI_FNENTRY("");

    /* verify input parameters */
    if (perGpuContext == NULL || rmHandle.hClient == 0U || rmHandle.hDevice == 0U ||
        rmHandle.hMemory == 0U || len == 0U ||
        accPerm >= LwSciBufAccessPerm_Ilwalid || ptr == NULL) {
        LWSCI_ERR("Bad parameter supplied to LwSciBufSysMemMemMap\n");
        LWSCI_ERR("context: %p\trmHandle.hClient: %u\trmHandle.hDevice: %u\t"
            "rmHandle.hMemory: %u\toffset: %lu\tlen: %lu\taccPerm: %u\tptr: %p\n",
            perGpuContext, rmHandle.hClient, rmHandle.hDevice, rmHandle.hMemory,
            offset, len, accPerm, ptr);
        sciErr = LwSciError_BadParameter;
        goto ret;
    }

    /* print input parameters */
    LWSCI_INFO("Input: context: %p\trmHandle.hMemory: %u\t"
        "offset: %lu\tlen: %lu\taccPerm: %u\tptr: %p\n", perGpuContext,
            rmHandle.hMemory, offset, len, accPerm, ptr);

    sysMemContext = perGpuContext->sysMemContextPtr;

    sciErr = LwSciBufColwertRmHandleToRmShimMemoryContext(&rmHandle, &rmMemory);
    if (sciErr != LwSciError_Success) {
        LWSCI_ERR("Failed to colwert RmHandle to memcontext\n");
        goto ret;
    }

    resmanMappingParam.redirection = LwSciBufAllocResmanRedirection_Default;
    resmanMappingParam.access = accPerm;

    sciErr = LwSciBufSetMapParams(&resmanMappingParam,
                &lwResmanCPUMappingParam);
    if (sciErr != LwSciError_Success) {
        LWSCI_ERR("LwSciBufSetMapParams failed\n");
        goto ret;
    }

    memset(&mapParam, 0x00, sizeof(LwRmShimMemMapParams));
    mapParam.offset = offset;
    mapParam.len = len;
    attr[0] = lwResmanCPUMappingParam.redirection;
    attr[1] = lwResmanCPUMappingParam.access;
    mapParam.attr = &attr[0];
    mapParam.numAttr = 2U;

    errShim = sysMemContext->rmShimAllocFvt.rmMemMap(
                sysMemContext->rmSessionPtr,
                perGpuContext->rmDevicePtr,
                &rmMemory,
                &mapParam);
    if (errShim != LWRMSHIM_OK) {
        sciErr = LwRmShimErrorToLwSciError(errShim);
        LWSCI_ERR("Mapping failed\n");
        goto ret;
    }

    *ptr = mapParam.cpuPtr;

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
    const LwSciBufSysMemPerGpuContext* perGpuContext =
            (const LwSciBufSysMemPerGpuContext*)context;
    LwRmShimMemUnMapParams unMapParams;
    LwRmShimError errShim = LWRMSHIM_OK;
    LwRmShimMemoryContext rmMemory = {0};

    LWSCI_FNENTRY("");

    /* verify input parameters */
    if (perGpuContext == NULL || rmHandle.hClient == 0U || rmHandle.hDevice == 0U ||
        rmHandle.hMemory == 0U || ptr == NULL || size == 0U) {
        LWSCI_ERR("Bad parameter supplied to LwSciBufSysMemMemUnMap\n");
        LWSCI_ERR("context: %p\trmHandle.hClient: %u\trmHandle.hDevice: %u\t"
            "rmHandle.hMemory: %u\tptr: %p\tsize: %lu\n", perGpuContext,
            rmHandle.hClient, rmHandle.hDevice, rmHandle.hMemory,
            ptr, size);
        sciErr = LwSciError_BadParameter;
        goto ret;
    }

    /* print input parameters */
    LWSCI_INFO("Input: context: %p\tptr: %p\tsize: %lu\n",
        perGpuContext, ptr, size);

    sysMemContext =  perGpuContext->sysMemContextPtr;

    sciErr = LwSciBufColwertRmHandleToRmShimMemoryContext(&rmHandle, &rmMemory);
    if (sciErr != LwSciError_Success) {
        LWSCI_ERR("Failed to colwert RmHandle to memcontext\n");
        goto ret;
    }

    memset(&unMapParams, 0x00, sizeof(LwRmShimMemUnMapParams));
    unMapParams.cpuPtr = ptr;
    uint64_t attr[1] = {0};
    unMapParams.attr = &attr[0];
    unMapParams.numAttr = 0U;

    errShim = sysMemContext->rmShimAllocFvt.rmUnMapMem(
                sysMemContext->rmSessionPtr,
                perGpuContext->rmDevicePtr,
                &rmMemory,
                &unMapParams);
    if (errShim != LWRMSHIM_OK) {
        sciErr = LwRmShimErrorToLwSciError(errShim);
        LWSCI_ERR("Deallocation failed\n");
        goto ret;
    }

ret:
    LWSCI_FNEXIT("");
    return sciErr;
}

LwSciError LwSciBufSysMemGetAlignment(
    const void* context,
    LwSciBufRmHandle rmHandle,
    uint64_t* alignment)
{
    LwSciError err = LwSciError_NotSupported;

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
    LwSciError err = LwSciError_NotSupported;

    (void)context;
    (void)rmHandle;
    (void)heapType;

    LWSCI_FNENTRY("");

    LWSCI_FNEXIT("");
    return err;
}

LwSciError LwSciBufSysMemGetSize(
    const void* context,
    LwSciBufRmHandle rmHandle,
    uint64_t* size)
{
    LwSciError sciErr = LwSciError_Success;
    const LwSciBufAllocSysMemContext* sysMemContext = (const LwSciBufAllocSysMemContext*)context;

    LWSCI_FNENTRY("");

    /* verify input parameters */
    if (sysMemContext == NULL || size == NULL || rmHandle.hClient == 0U ||
        rmHandle.hDevice == 0U || rmHandle.hMemory == 0U) {
        sciErr = LwSciError_BadParameter;
        LWSCI_ERR("Bad parameter supplied to LwSciBufSysMemGetSize\n");
        LWSCI_ERR("context: %p\trmHandle.hClient: %u\trmHandle.hDevice: %u\t"
            "rmHandle.hMemory: %u\tsize: %p\n", sysMemContext,
            rmHandle.hClient, rmHandle.hDevice, rmHandle.hMemory, size);
        goto ret;
    }

    /* print input parameters */
    LWSCI_INFO("Input: context: %p\trmHandle.hClient: %u\trmHandle.hDevice: %u\t"
            "rmHandle.hMemory: %u\tsize: %p\n", sysMemContext,
            rmHandle.hClient, rmHandle.hDevice, rmHandle.hMemory, size);

    /* this value is used by objmgmt unit to verify if the allocated size is
     *  greater then equal to requested size.
     * Since ResMan doesn't have API to give size from rmHandle,
     *  we return fake highest value so that objmgmt unit doesnt fail.
     * TODO investigate this more with ResMan team.
     */
    *size = UINTMAX_MAX;

    /* print output parameters */
    LWSCI_INFO("Output: buf size: %lu\n", *size);

ret:
    LWSCI_FNEXIT("");
    return sciErr;
}

LwSciError LwSciBufSysMemCpuCacheFlush(
    const void* context,
    LwSciBufRmHandle rmHandle,
    void* cpuPtr,
    uint64_t len)
{
    LwSciError sciErr = LwSciError_Success;
    const LwSciBufAllocSysMemContext* sysMemContext = (const LwSciBufAllocSysMemContext*)context;

    LWSCI_FNENTRY("");

    /* verify input parameters */
    if (sysMemContext == NULL || rmHandle.hClient == 0U || rmHandle.hDevice == 0U ||
        rmHandle.hMemory == 0U || cpuPtr == NULL || len ==0U) {
        sciErr = LwSciError_BadParameter;
        LWSCI_ERR("Bad parameter supplied to LwSciBufSysMemCpuCacheFlush\n");
        LWSCI_ERR("context: %p\trmHandle.hClient: %u\trmHandle.hDevice: %u\t"
            "rmHandle.hMemory: %u\tcpuPtr: %p\tlen:%lu\n",
            sysMemContext, rmHandle.hClient, rmHandle.hDevice,
            rmHandle.hMemory, cpuPtr, len);
        goto ret;
    }

    /* print input parameters */
    LWSCI_INFO("Input: context: %p\trmHandle.hClient: %u\trmHandle.hDevice: %u\t"
            "rmHandle.hMemory: %u\tcpuPtr: %p\tlen:%lu\n",
            sysMemContext, rmHandle.hClient, rmHandle.hDevice,
            rmHandle.hMemory, cpuPtr, len);

    LWSCI_INFO("No need to flush any CPU caches as x86 PCIe is IO coherent\n");

   /* print output parameters */
    LWSCI_INFO("Successfully completed LwRm CPU Cache flush.\n");

ret:
    LWSCI_FNEXIT("");
    return sciErr;
}

LwSciError LwSciBufSysMemGetAllocContext(
    const void* allocContextParam,
    void* openContext,
    void** allocContext)
{
    LwSciError err = LwSciError_Success;
    LwSciBufAllocSysMemContext* sysMemContext = NULL;
    const LwSciBufAllocSysMemAllocContextParam* sysMemAllocContextParam = NULL;
    LwRmShimSessionContext* session = NULL;
    uint32_t i = 0U;

    LWSCI_FNENTRY("");

    if (allocContextParam == NULL || openContext == NULL ||
        allocContext == NULL) {
        LWSCI_ERR("Bad parameter supplied to LwSciBufSysMemGetAllocContext\n");
        LWSCI_ERR("allocContextParam: %p, openContext: %p, allocContext: %p\n",
            allocContextParam, openContext, allocContext);
        err = LwSciError_Success;
        goto ret;
    }

    LWSCI_INFO("Input: allocContextParam: %p, openContext: %p, allocContext: %p\n",
        allocContextParam, openContext, allocContext);

    sysMemAllocContextParam =
        (const LwSciBufAllocSysMemAllocContextParam*)allocContextParam;
    sysMemContext = (LwSciBufAllocSysMemContext*)openContext;
    session = sysMemContext->rmSessionPtr;

    /* Use device 0 if GpuId is not provided */
    if (sysMemAllocContextParam->gpuIdsCount == 0) {
        *allocContext = (void*)&sysMemContext->perGpuContext[0];
        goto ret;
    }

    for (i = 0U; i< session->numGpus; i++) {
        LWSCI_INFO("Comparing 0x%lx %lx 0x%lx %lx\n",
            *(uint64_t*)&session->gpuUUID[i].bytes[0],
            *(uint64_t*)&session->gpuUUID[i].bytes[8],
            *(uint64_t*)&sysMemAllocContextParam->gpuIds[0].bytes[0],
            *(uint64_t*)&sysMemAllocContextParam->gpuIds[0].bytes[8]);
        if (LwSciCommonMemcmp(session->gpuUUID[i].bytes,
                &sysMemAllocContextParam->gpuIds[0].bytes,
                sizeof(LwSciRmGpuId)) == 0) {
            *allocContext = (void*)&sysMemContext->perGpuContext[i];
            goto ret;
        }
    }

    /* we are here implies that we could not find matching GPU UUID passed via
     * allocContext with all the GPUs that are present in GPU context
     */
    err = LwSciError_ResourceError;
    LWSCI_ERR("GPU with below UUID not initialized.\n");
    LWSCI_ERR("Error might have oclwrred during opening of devices in LwSciBufSysMemOpen call\n");
    for (i = 0U; i < sizeof(LwSciRmGpuId)/sizeof(uint8_t); i++) {
        LWSCI_ERR("byte[%zu]: %02" PRIx8 "\n", i,
        sysMemAllocContextParam->gpuIds[0].bytes[i]);
    }

ret:
    LWSCI_FNEXIT("");
    return err;
}
