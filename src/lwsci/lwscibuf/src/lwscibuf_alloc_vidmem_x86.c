/*
 * Copyright (c) 2019-2021, LWPU CORPORATION. All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#include "lwscibuf_alloc_vidmem_x86_priv.h"

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

static LwSciError LwSciBufSetAllocParams(
    LwSciBufAllocVidMemVal* vidMemAllocVal,
    LwRmSysMemAllocVal* lwAllocParams)
{
    LwSciError sciErr = LwSciError_Success;

    LWSCI_FNENTRY("");

    /* verify input parameters */
    if (vidMemAllocVal == NULL || lwAllocParams == NULL) {
        sciErr = LwSciError_BadParameter;
        LWSCI_ERR("Bad parameter supplied to getOS32AttrFlags\n");
        LWSCI_ERR("vidMemAllocVal: %p\tlwAllocParams: %p\n",
                    vidMemAllocVal, lwAllocParams);
        goto ret;
    }

    /* print input parameters */
    LWSCI_INFO("Input: vidMemAllocVal: %p\tlwAllocParams: %p\n",
                vidMemAllocVal, lwAllocParams);

    lwAllocParams->location = LwRmShimMemLocation_VidMem;

    if (vidMemAllocVal->coherency == true) {
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

static LwSciError LwSciBufFillRmShimFvt(
    LwSciBufDev devHandle,
    LwSciBufAllocVidMemContext* vidMemContext)
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
    if (devHandle == NULL || vidMemContext == NULL) {
        LWSCI_ERR("Bad parameter supplied\n");
        LWSCI_ERR("devHandle: %p\tvidMemContext: %p\n",
                    devHandle, vidMemContext);
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

    vidMemContext->rmShimAllocFvt.rmAllocMem = rmAllocMem;
    vidMemContext->rmShimAllocFvt.rmMemMap = rmMemMap;
    vidMemContext->rmShimAllocFvt.rmDupMem = rmDupMem;
    vidMemContext->rmShimAllocFvt.rmUnMapMem = rmUnMapMem;
    vidMemContext->rmShimAllocFvt.rmFreeMem = rmFreeMem;
    vidMemContext->rmShimAllocFvt.rmFlushCpuCache = rmFlushCpuCache;

ret:
    LWSCI_FNEXIT("");
    return sciErr;
}

LwSciError LwSciBufVidMemOpen(
    LwSciBufDev devHandle,
    void** context)
{
    LwSciError sciErr = LwSciError_Success;
    LwSciBufAllocVidMemContext* vidMemContext = NULL;
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

    *context = LwSciCommonCalloc(1, sizeof(LwSciBufAllocVidMemContext));
    if (*context == NULL) {
        LWSCI_ERR("Could not allocate memory for LwSciBufAllocVidMemContext\n");
        sciErr = LwSciError_InsufficientMemory;
        goto ret;
    }
    vidMemContext = (LwSciBufAllocVidMemContext*)(*context);

    sciErr = LwSciBufFillRmShimFvt(devHandle, vidMemContext);
    if (sciErr != LwSciError_Success) {
        LWSCI_ERR("Failed to get RmShim Fvt\n");
        goto free_context;
    }


    sciErr = LwSciBufDevGetRmSessionDevice(devHandle,
                        &vidMemContext->rmSessionPtr,
                        &vidMemContext->rmDevicePtr);
    if (sciErr != LwSciError_Success) {
        LWSCI_ERR("LwSciBufGetRmSessionDevice failed\n");
        goto free_context;
    }

    vidMemContext->perGpuContext = LwSciCommonCalloc(
                                    vidMemContext->rmSessionPtr->numGpus,
                                    sizeof(LwSciBufPerGpuContext));
    if (vidMemContext->perGpuContext == NULL) {
        LWSCI_ERR("Could not allocate memory for LwSciBufPerGpuContext struct\n");
        sciErr = LwSciError_InsufficientMemory;
        goto free_context;
    }

    for (i = 0U; i < vidMemContext->rmSessionPtr->numGpus; i++) {
        vidMemContext->perGpuContext[i].vidMemContextPtr = vidMemContext;
        vidMemContext->perGpuContext[i].rmDevicePtr = &vidMemContext->rmDevicePtr[i];
    }

    /* print output parameters */
    LWSCI_INFO("Output: vidMemContext: %p\thClient: %u\thDevice: %u\n",
        *context, vidMemContext->rmSessionPtr->pHandle,
        vidMemContext->rmDevicePtr->pHandle);

    /* All opeartions are successful. Directly jump to 'ret' from here */
    goto ret;

free_context:
    LwSciCommonFree(*context);
ret:
    LWSCI_FNEXIT("");
    return sciErr;
}

LwSciError LwSciBufVidMemAlloc(
    const void* context,
    void* allocVal,
    LwSciBufDev devHandle,
    LwSciBufRmHandle* rmHandle)
{
    LwSciError sciErr = LwSciError_Success;
    LwSciBufAllocVidMemVal* vidMemAllocVal = (LwSciBufAllocVidMemVal*)allocVal;
    LwSciBufAllocVidMemContext* vidMemContext = NULL;
    const LwSciBufPerGpuContext* perGpuContext = (const LwSciBufPerGpuContext*)context;
    LwRmSysMemAllocVal lwRmSysMemAllocVal = {0};
    LwRmShimMemoryContext rmMemory = {0};
    LwRmShimError errShim = LWRMSHIM_OK;
    LwRmShimAllocMemParams memParams;
    uint64_t attr[2] = {0, 0};

    LWSCI_FNENTRY("");

    /* verify input parameters */
    if (context == NULL || vidMemAllocVal == NULL) {
        sciErr = LwSciError_BadParameter;
        LWSCI_ERR("Bad parameter supplied to LwSciBufSysMemAlloc\n");
        LWSCI_ERR("context: %p\tvidMemAllocVal: %p\n", context, vidMemAllocVal);
        goto ret;
    }

    if (vidMemAllocVal->size == 0U || devHandle == NULL || rmHandle == NULL) {
        sciErr = LwSciError_BadParameter;
        LWSCI_ERR("Bad parameter supplied to LwSciBufSysMemAlloc\n");
        LWSCI_ERR("vidMemAllocVal->size %u\t"
            "devHandle: %p\trmHandle: %p\n",
            vidMemAllocVal->size, devHandle, rmHandle);
        goto ret;
    }

    /* print input parameters */
    LWSCI_INFO("Input: context: %p\tvidMemAllocVal: %p\tvidMemAllocVal->size: %u\t"
        "vidMemAllocVal->alignment: %u\tvidMemAllocVal->coherency: %u\t"
        "devHandle: %p\trmHandle: %p\n",
        context, vidMemAllocVal, vidMemAllocVal->size, vidMemAllocVal->alignment,
        vidMemAllocVal->coherency, devHandle, rmHandle);

    vidMemContext =  perGpuContext->vidMemContextPtr;

    sciErr = LwSciBufSetAllocParams(vidMemAllocVal,
                &lwRmSysMemAllocVal);
    if (sciErr != LwSciError_Success) {
        LWSCI_ERR("LwSciBufSetAllocParams failed\n");
        goto ret;
    }

    memset(&memParams, 0x00, sizeof(memParams));
    memParams.size = vidMemAllocVal->size;
    memParams.alignment = vidMemAllocVal->alignment;
    memParams.offset = 0;
    attr[0] = (uint64_t)lwRmSysMemAllocVal.location;
    attr[1] = (uint64_t)lwRmSysMemAllocVal.cacheCoherency;
    memParams.attr = &attr[0];
    memParams.numAttr = 2U;

    errShim = vidMemContext->rmShimAllocFvt.rmAllocMem(
                vidMemContext->rmSessionPtr,
                perGpuContext->rmDevicePtr,
                &rmMemory,
                &memParams);
    if (errShim != LWRMSHIM_OK) {
        sciErr = LwRmShimErrorToLwSciError(errShim);
        LWSCI_ERR("Allocation failed\n");
        goto ret;
    }

    sciErr = LwSciBufColwertRmShimContextToRmHandle(vidMemContext->rmSessionPtr,
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

LwSciError LwSciBufVidMemDealloc(
    void* context,
    LwSciBufRmHandle rmHandle)
{
   LwSciError sciErr = LwSciError_Success;
    LwRmShimError errShim = LWRMSHIM_OK;
    LwSciBufAllocVidMemContext* vidMemContext = NULL;
    LwSciBufPerGpuContext* perGpuContext = (LwSciBufPerGpuContext*)context;
    LwRmShimMemoryContext rmMemory = {0};

    LWSCI_FNENTRY("");

    /* verify input parameters */
    if (context == NULL || rmHandle.hClient == 0U ||
        rmHandle.hDevice == 0U || rmHandle.hMemory == 0U) {
        LWSCI_ERR("Bad parameter supplied to LwSciBufSysMemDealloc\n");
        LWSCI_ERR("context: %p\thclient: %u\thDevice: %u\thMemory: %u\n",
            context, rmHandle.hClient,
            rmHandle.hDevice, rmHandle.hMemory);
        sciErr = LwSciError_BadParameter;
        goto ret;
    }

    vidMemContext =  perGpuContext->vidMemContextPtr;

    sciErr = LwSciBufColwertRmHandleToRmShimMemoryContext(&rmHandle, &rmMemory);
    if (sciErr != LwSciError_Success) {
        LWSCI_ERR("Failed to colwert RmHandle to memcontext\n");
        goto ret;
    }

    errShim = vidMemContext->rmShimAllocFvt.rmFreeMem(
                vidMemContext->rmSessionPtr,
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

LwSciError LwSciBufVidMemDupHandle(
    const void* context,
    LwSciBufAttrValAccessPerm newPerm,
    LwSciBufRmHandle rmHandle,
    LwSciBufRmHandle *dupRmHandle)
{
    LwSciError sciErr = LwSciError_Success;

    LwSciBufAllocVidMemContext* vidMemContext = NULL;
    const LwSciBufPerGpuContext* perGpuContext = (const LwSciBufPerGpuContext*)context;
    LwRmShimError errShim = LWRMSHIM_OK;
    LwRmShimDupMemContextParams dupMemParams;
    LwRmShimSessionContext ipSession = {0};
    LwRmShimDeviceContext ipDevice = {0};
    LwRmShimMemoryContext ipMemory = {0};
    LwRmShimMemoryContext rmMemory = {0};

    (void)newPerm;

    LWSCI_FNENTRY("");

    /* verify input parameters */
    if (context == NULL || rmHandle.hClient == 0U || rmHandle.hDevice == 0U ||
        rmHandle.hMemory == 0U || dupRmHandle == NULL) {
        LWSCI_ERR("Bad parameter supplied to LwSciBufSysMemDupHandle\n");
        LWSCI_ERR("context: %p\thClient: %u\thDevice: %u\t"
            "hMemory: %u\tdupRmHandle: %p\n", context, rmHandle.hClient,
            rmHandle.hDevice, rmHandle.hMemory, dupRmHandle);
        sciErr = LwSciError_BadParameter;
        goto ret;
    }

    vidMemContext =  perGpuContext->vidMemContextPtr;

    sciErr = LwSciBufColwertRmHandleToRmShimContext(&rmHandle, &ipSession,
                &ipDevice, &ipMemory);
    if (sciErr != LwSciError_Success) {
        LWSCI_ERR("Failed to colwert RmShim to RmHandle\n");
        goto ret;
    }

    memset(&dupMemParams, 0x00, sizeof(dupMemParams));
    dupMemParams.dupMemory = &rmMemory;
    dupMemParams.dupSession = vidMemContext->rmSessionPtr;
    dupMemParams.dupDevice = perGpuContext->rmDevicePtr;
    uint64_t attr[1] = {0};
    dupMemParams.attr = &attr[0];
    dupMemParams.numAttr = 0U;

    errShim = vidMemContext->rmShimAllocFvt.rmDupMem(
                &ipSession,
                &ipDevice,
                &ipMemory,
                &dupMemParams);
    if (errShim != LWRMSHIM_OK) {
        sciErr = LwRmShimErrorToLwSciError(errShim);
        LWSCI_ERR("Duplication failed\n");
        goto ret;
    }

    sciErr = LwSciBufColwertRmShimContextToRmHandle(vidMemContext->rmSessionPtr,
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

LwSciError LwSciBufVidMemMemMap(
    const void* context,
    LwSciBufRmHandle rmHandle,
    uint64_t offset,
    uint64_t len,
    LwSciBufAttrValAccessPerm accPerm,
    void **ptr)
{
    LwSciError sciErr = LwSciError_Success;
    LwSciBufResmanCpuMappingParam resmanMappingParam = {0};
    LwResmanCpuMappingParam lwResmanCPUMappingParam = {0};
    LwSciBufAllocVidMemContext* vidMemContext = NULL;
    const LwSciBufPerGpuContext* perGpuContext = (const LwSciBufPerGpuContext*)context;
    LwRmShimError errShim = LWRMSHIM_OK;
    LwRmShimMemMapParams mapParam = {0};
    LwRmShimMemoryContext rmMemory = {0};
    uint64_t attr[2] = {0, 0};

    LWSCI_FNENTRY("");

    /* verify input parameters */
    if (context == NULL || rmHandle.hClient == 0U || rmHandle.hDevice == 0U ||
        rmHandle.hMemory == 0U || len == 0U ||
        accPerm >= LwSciBufAccessPerm_Ilwalid || ptr == NULL) {
        LWSCI_ERR("Bad parameter supplied to LwSciBufSysMemMemMap\n");
        LWSCI_ERR("context: %p\trmHandle.hClient: %u\trmHandle.hDevice: %u\t"
            "rmHandle.hMemory: %u\toffset: %lu\tlen: %lu\taccPerm: %u\tptr: %p\n",
            context, rmHandle.hClient, rmHandle.hDevice, rmHandle.hMemory,
            offset, len, accPerm, ptr);
        sciErr = LwSciError_BadParameter;
        goto ret;
    }

    /* print input parameters */
    LWSCI_INFO("Input: context: %p\trmHandle.hMemory: %u\t"
        "offset: %lu\tlen: %lu\taccPerm: %u\tptr: %p\n", context,
            rmHandle.hMemory, offset, len, accPerm, ptr);

    vidMemContext = perGpuContext->vidMemContextPtr;

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

    errShim = vidMemContext->rmShimAllocFvt.rmMemMap(
                vidMemContext->rmSessionPtr,
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

LwSciError LwSciBufVidMemMemUnMap(
    const void* context,
    LwSciBufRmHandle rmHandle,
    void* ptr,
    uint64_t size)
{
    LwSciError sciErr = LwSciError_Success;
    LwSciBufAllocVidMemContext* vidMemContext = NULL;
    const LwSciBufPerGpuContext* perGpuContext = (const LwSciBufPerGpuContext*)context;
    LwRmShimMemUnMapParams unMapParams;
    LwRmShimError errShim = LWRMSHIM_OK;
    LwRmShimMemoryContext rmMemory = {0};

    LWSCI_FNENTRY("");

    /* verify input parameters */
    if (context == NULL || rmHandle.hClient == 0U || rmHandle.hDevice == 0U ||
        rmHandle.hMemory == 0U || ptr == NULL || size == 0U) {
        LWSCI_ERR("Bad parameter supplied to LwSciBufSysMemMemUnMap\n");
        LWSCI_ERR("context: %p\trmHandle.hClient: %u\trmHandle.hDevice: %u\t"
            "rmHandle.hMemory: %u\tptr: %p\tsize: %lu\n", context,
            rmHandle.hClient, rmHandle.hDevice, rmHandle.hMemory,
            ptr, size);
        sciErr = LwSciError_BadParameter;
        goto ret;
    }

    /* print input parameters */
    LWSCI_INFO("Input: context: %p\tptr: %p\tsize: %lu\n",
        context, ptr, size);

    vidMemContext =  perGpuContext->vidMemContextPtr;

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

    errShim = vidMemContext->rmShimAllocFvt.rmUnMapMem(
                vidMemContext->rmSessionPtr,
                perGpuContext->rmDevicePtr,
                &rmMemory,
                &unMapParams);
    if (errShim != LWRMSHIM_OK) {
        sciErr = LwSciError_IlwalidState;
        LWSCI_ERR("Deallocation failed\n");
        goto ret;
    }

ret:
    LWSCI_FNEXIT("");
    return sciErr;
}

LwSciError LwSciBufVidMemGetSize(
    const void* context,
    LwSciBufRmHandle rmHandle,
    uint64_t* size)
{
    LwSciError err = LwSciError_NotSupported;

    (void)context;
    (void)rmHandle;
    (void)size;

    LWSCI_FNENTRY("");

    LWSCI_FNEXIT("");
    return err;
}

LwSciError LwSciBufVidMemGetAlignment(
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

LwSciError LwSciBufVidMemGetHeapType(
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

LwSciError LwSciBufVidMemCpuCacheFlush(
    const void* context,
    LwSciBufRmHandle rmHandle,
    void* cpuPtr,
    uint64_t len)
{
    LwSciError sciErr = LwSciError_Success;

    LWSCI_FNENTRY("");

    /* verify input parameters */
    if (context == NULL || rmHandle.hClient == 0U || rmHandle.hDevice == 0U ||
        rmHandle.hMemory == 0U || cpuPtr == NULL || len ==0U) {
        sciErr = LwSciError_BadParameter;
        LWSCI_ERR("Bad parameter supplied to LwSciBufSysMemCpuCacheFlush\n");
        LWSCI_ERR("context: %p\trmHandle.hClient: %u\trmHandle.hDevice: %u\t"
            "rmHandle.hMemory: %u\tcpuPtr: %p\tlen:%lu\n",
            context, rmHandle.hClient, rmHandle.hDevice,
            rmHandle.hMemory, cpuPtr, len);
        goto ret;
    }

    /* print input parameters */
    LWSCI_INFO("Input: context: %p\trmHandle.hClient: %u\trmHandle.hDevice: %u\t"
            "rmHandle.hMemory: %u\tcpuPtr: %p\tlen:%lu\n",
            context, rmHandle.hClient, rmHandle.hDevice,
            rmHandle.hMemory, cpuPtr, len);

    LWSCI_INFO("No need to flush any CPU caches as x86 PCIe is IO coherent\n");

   /* print output parameters */
    LWSCI_INFO("Successfully completed LwRm CPU Cache flush.\n");

ret:
    LWSCI_FNEXIT("");
    return sciErr;
}

void LwSciBufVidMemClose(
    void* context)
{
    LwSciBufAllocVidMemContext* vidMemContext = (LwSciBufAllocVidMemContext*)context;

    LWSCI_FNENTRY("");

    if (context == NULL) {
        LWSCI_ERR("Context to close is NULL.\n");
        goto ret;
    }

    /* print input parameters */
    LWSCI_INFO("Input: context: %p\n", context);

    LwSciCommonFree(vidMemContext->perGpuContext);
    LwSciCommonFree(context);
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
    const LwSciBufAllocVidMemAllocContextParam* vidMemAllocContextParam = NULL;
    LwRmShimSessionContext* session = NULL;
    uint32_t i = 0U;

    LWSCI_FNENTRY("");

    if (allocContextParam == NULL || openContext == NULL ||
        allocContext == NULL) {
        LWSCI_ERR("Bad parameter supplied to LwSciBufVidMemGetAllocContext\n");
        LWSCI_ERR("allocContextParam: %p, openContext: %p, allocContext: %p\n",
            allocContextParam, openContext, allocContext);
        err = LwSciError_BadParameter;
        goto ret;
    }

    LWSCI_INFO("Input: allocContextParam: %p, openContext: %p, allocContext: %p\n",
        allocContextParam, openContext, allocContext);

    vidMemAllocContextParam =
        (const LwSciBufAllocVidMemAllocContextParam*)allocContextParam;
    vidMemContext = (LwSciBufAllocVidMemContext*)openContext;
    session = vidMemContext->rmSessionPtr;

    for (i = 0U; i< session->numGpus; i++) {
        LWSCI_INFO("Comparing 0x%lx %lx 0x%lx %lx\n",
            *(uint64_t*)&session->gpuUUID[i].bytes[0],
            *(uint64_t*)&session->gpuUUID[i].bytes[8],
            *(uint64_t*)&vidMemAllocContextParam->gpuId.bytes[0],
            *(uint64_t*)&vidMemAllocContextParam->gpuId.bytes[8]);
        if (LwSciCommonMemcmp(session->gpuUUID[i].bytes,
                &vidMemAllocContextParam->gpuId.bytes,
                sizeof(LwSciRmGpuId)) == 0) {
            *allocContext = (void*)&vidMemContext->perGpuContext[i];
            goto ret;
        }
    }

    /* we are here implies that we could not find matching GPU UUID passed via
     * allocContext with all the GPUs that are present in GPU context
     */
    err = LwSciError_ResourceError;
    LWSCI_ERR("GPU with below UUID not initialized.\n");
    LWSCI_ERR("Error might have oclwrred during opening of devices in LwSciBufVidMemOpen call\n");
    for (i = 0U; i < sizeof(LwSciRmGpuId)/sizeof(uint8_t); i++) {
        LWSCI_ERR("byte[%zu]: %02" PRIx8 "\n", i,
        vidMemAllocContextParam->gpuId.bytes[i]);
    }

ret:
    LWSCI_FNEXIT("");
    return err;
}
