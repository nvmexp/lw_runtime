/*
 * Copyright (c) 2019, LWPU CORPORATION. All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#include <dlfcn.h>

#include "lwscibuf_dev_platform_x86_priv.h"

static LwSciError LwSciBufOpenRmShimLibrary(
    LwSciBufDevPriv* dev)
{
    LwSciError sciErr = LwSciError_Success;
    sessionCreateFunc sessionCreate = NULL;
    sessionDestroyFunc sessionDestroy = NULL;
    openGpuInstanceFunc rmOpenGpuInstance = NULL;
    closeGpuInstanceFunc rmCloseGpuInstance = NULL;
    validateUUIDFunc rmValidateUUID = NULL;

    LWSCI_FNENTRY("");

    /* verify input parameters */
    if (dev == NULL) {
        LWSCI_ERR("Bad parameter supplied\n");
        LWSCI_ERR("LwSciBufDev* dev: %p\n", dev);
        sciErr = LwSciError_BadParameter;
        goto ret;
    }

    /* print input parameters */
    LWSCI_INFO("Input: LwSciBufDev* dev: %p\n", dev);

    dev->lwRmShimLib = dlopen(LWRMSHIM_LIBRARY_NAME, RTLD_LAZY);
    if (dev->lwRmShimLib == NULL) {
        sciErr = LwSciError_IlwalidState;
        LWSCI_ERR("Failed to open %s library\n", LWRMSHIM_LIBRARY_NAME);
        goto ret;
    }

    sessionCreate = dlsym(dev->lwRmShimLib, "LwRmShimSessionCreate");
    if (sessionCreate == NULL) {
        sciErr = LwSciError_IlwalidState;
        LWSCI_ERR("Failed to get function \n");
        goto ret;
    }

    sessionDestroy = dlsym(dev->lwRmShimLib, "LwRmShimSessionDestroy");
    if (sessionDestroy == NULL) {
        sciErr = LwSciError_IlwalidState;
        LWSCI_ERR("Failed to get function \n");
        goto ret;
    }

    rmOpenGpuInstance = dlsym(dev->lwRmShimLib,
                                "LwRmShimOpenGpuInstance");
    if (rmOpenGpuInstance == NULL) {
        sciErr = LwSciError_IlwalidState;
        LWSCI_ERR("Failed to get function \n");
        goto ret;
    }

    rmCloseGpuInstance = dlsym(dev->lwRmShimLib,
                                "LwRmShimCloseGpuInstance");
    if (rmCloseGpuInstance == NULL) {
        sciErr = LwSciError_IlwalidState;
        LWSCI_ERR("Failed to get function \n");
        goto ret;
    }

    rmValidateUUID = dlsym(dev->lwRmShimLib,"LwRmShimValidateUUID");
    if (rmValidateUUID == NULL) {
        sciErr = LwSciError_IlwalidState;
        LWSCI_ERR("Failed to get function \n");
        goto ret;
    }

    dev->rmShimDevFvt.sessionCreate = sessionCreate;
    dev->rmShimDevFvt.sessionDestroy = sessionDestroy;
    dev->rmShimDevFvt.rmOpenGpuInstance = rmOpenGpuInstance;
    dev->rmShimDevFvt.rmCloseGpuInstance = rmCloseGpuInstance;
    dev->rmShimDevFvt.rmValidateUUID = rmValidateUUID;

ret:
    LWSCI_FNEXIT("");
    return sciErr;
}

static void LwSciBufCloseRmShimLibrary(
    LwSciBufDevPriv* dev)
{
    LWSCI_FNENTRY("");

    /* verify input parameters */
    if (dev == NULL) {
        LWSCI_ERR("Bad parameter supplied\n");
        LWSCI_ERR("LwSciBufDev* dev: %p\n", dev);
        goto ret;
    }

    /* print input parameters */
    LWSCI_INFO("Input: LwSciBufDev* dev: %p\n", dev);

    if (dlclose(dev->lwRmShimLib) != 0) {
        LWSCI_ERR("Failed to close library %s\n", LWRMSHIM_LIBRARY_NAME);
        goto ret;
    }

    dev->lwRmShimLib = NULL;

ret:
    LWSCI_FNEXIT("");
    return;
}


LwSciError LwSciBufDevOpen(
    LwSciBufDev* newDev)
{
    LwSciError sciErr = LwSciError_Success;
    LwSciBufDev dev = NULL;
    LwRmShimError errShim = LWRMSHIM_OK;
    LwRmShimGpuOpenParams gpuOpenParam;
    uint32_t i = 0U;

    LWSCI_FNENTRY("");

    /* verify input parameters */
    if (newDev == NULL) {
        LWSCI_ERR("Bad parameter supplied to LwSciBufDevOpen\n");
        LWSCI_ERR("LwSciBufDev* newDev: %p\n", newDev);
        sciErr = LwSciError_BadParameter;
        goto ret;
    }

    *newDev = NULL;
    /* print input parameters */
    LWSCI_INFO("Input: LwSciBufDev* newDev: %p\n", newDev);

    dev = LwSciCommonCalloc(1, sizeof(LwSciBufDevPriv));
    if (dev == NULL) {
        LWSCI_ERR("Could not allocate memory for LwSciBufDevPriv struct\n");
        sciErr = LwSciError_InsufficientMemory;
        goto ret;
    }

    sciErr = LwSciBufOpenRmShimLibrary(dev);
    if (sciErr != LwSciError_Success) {
        LWSCI_ERR("Failed to open supporting library\n");
        goto free_dev;
    }

    errShim = dev->rmShimDevFvt.sessionCreate(&dev->rmSession);
    if (errShim != LWRMSHIM_OK) {
        sciErr = LwRmShimErrorToLwSciError(errShim);
        LWSCI_ERR("Failed to open resman session\n");
        goto close_lib;
    }


   dev->rmDeviceList = LwSciCommonCalloc(dev->rmSession.numGpus, sizeof(*dev->rmDeviceList));
    if (dev->rmDeviceList == NULL) {
        sciErr = LwRmShimErrorToLwSciError(errShim);
        LWSCI_ERR("Could not allocate memory for rmDeviceList struct\n");
        goto close_lib;
    }

    for (i = 0; i < dev->rmSession.numGpus; i++) {
        memset(&gpuOpenParam, 0x00, sizeof(gpuOpenParam));
        gpuOpenParam.gpuId = dev->rmSession.gpuId[i];
        uint64_t attr[1] = {(uint64_t)&dev->rmSession.gpuUUID[i]};
        gpuOpenParam.attr = &attr[0];
        gpuOpenParam.numAttr = 1;
        errShim = dev->rmShimDevFvt.rmOpenGpuInstance(
                        &dev->rmSession, &dev->rmDeviceList[i],
                        &gpuOpenParam);
        if (errShim != LWRMSHIM_OK) {
            sciErr = LwRmShimErrorToLwSciError(errShim);
            LWSCI_ERR("Failed to open GPU instance %u\n", i);
            goto free_device;
        }
    }

    *newDev = dev;

    /* print output parameters */
    LWSCI_INFO("Output: *newDev: %p\n",
        newDev ? *newDev : 0);

    /* Everything succeeds, ret*/
    goto ret;

free_device:
    LwSciCommonFree(dev->rmDeviceList);
close_lib:
    LwSciBufCloseRmShimLibrary(dev);
free_dev:
    LwSciCommonFree(dev);
    *newDev = NULL;
ret:
    LWSCI_FNEXIT("");
    return sciErr;
}

void LwSciBufDevClose(
    LwSciBufDev dev)
{
    LWSCI_FNENTRY("");
    LwRmShimError errShim = LWRMSHIM_OK;
    uint32_t i = 0U;

    /* verify input parameters */
    if (dev == NULL) {
        LWSCI_ERR("Bad parameter supplied to LwSciBufDevClose\n");
        LWSCI_ERR("dev: %p\n", dev);
        LwSciCommonPanic();
    }

    /* print input parameters */
    LWSCI_INFO("Input: dev: %p\n", dev);

    for (i = 0; i < dev->rmSession.numGpus; i++) {
        errShim = dev->rmShimDevFvt.rmCloseGpuInstance(
                        &dev->rmSession, &dev->rmDeviceList[i]);
        if (errShim != LWRMSHIM_OK) {
            LWSCI_ERR("Failed to close GPU instance %u\n", i);
            goto ret;
        }
    }

    errShim = dev->rmShimDevFvt.sessionDestroy(&dev->rmSession);
    if (errShim != LWRMSHIM_OK) {
        LWSCI_ERR("Failed to close resman session\n");
        goto ret;
    }


ret:
    LwSciBufCloseRmShimLibrary(dev);
    LwSciCommonFree(dev->rmDeviceList);
    LwSciCommonFree(dev);
    LWSCI_FNEXIT("");
}

LwSciError LwSciBufDevGetRmLibrary(
    LwSciBufDev dev,
    void** rmLib)
{
    LwSciError sciErr = LwSciError_Success;

    LWSCI_FNENTRY("");

    /* verify input parameters */
    if (dev == NULL || rmLib == NULL) {
        LWSCI_ERR("Bad parameter supplied\n");
        LWSCI_ERR("dev: %p\trmLib: %p\n", dev, rmLib);
        sciErr = LwSciError_BadParameter;
        goto ret;
    }

    /* print input parameters */
    LWSCI_INFO("Input: dev: %p\trmLib: %p\n", dev, rmLib);

    *rmLib = dev->lwRmShimLib;

    /* print output parameters */
    LWSCI_INFO("Output: resman Library: %p\n", *rmLib);

ret:
    LWSCI_FNEXIT("");
    return sciErr;
}

LwSciError LwSciBufDevGetRmSessionDevice(
    LwSciBufDev dev,
    LwRmShimSessionContext** rmSession,
    LwRmShimDeviceContext** rmDevice)
{
    LwSciError sciErr = LwSciError_Success;

    LWSCI_FNENTRY("");

    /* verify input parameters */
    if (dev == NULL || rmSession == NULL || rmDevice == NULL) {
        LWSCI_ERR("Bad parameter supplied\n");
        LWSCI_ERR("dev: %p\trmSession: %p\trmDevice: %p\n",
                    dev, rmSession, rmDevice);
        sciErr = LwSciError_BadParameter;
        goto ret;
    }

    /* print input parameters */
    LWSCI_INFO("Input: dev: %p\trmSession: %p\t rmDevice: %p\n",
                    dev, rmSession, rmDevice);

    *rmSession = &dev->rmSession;
    *rmDevice = &dev->rmDeviceList[0];

    /* print output parameters */
    LWSCI_INFO("Output: resman Session: %p\t Device: %p\n",
                *rmSession, *rmDevice);

ret:
    LWSCI_FNEXIT("");
    return sciErr;
}

LwSciError LwSciBufCheckPlatformVersionCompatibility(
    bool* platformCompatibility)
{
    LwSciError sciErr = LwSciError_Success;
    void* lwRmShimLib = NULL;
    getVersionFunc getVersion = NULL;
    LwRmShimError errShim = LWRMSHIM_OK;
    LwRmShimVersion ver = {0};

    LWSCI_FNENTRY("");

    /* verify input parameters */
    if (platformCompatibility == NULL) {
        LWSCI_ERR("Bad parameter supplied to %s\n", __FUNCTION__);
        LWSCI_ERR("platformCompatibility %p\n", platformCompatibility);
        sciErr = LwSciError_BadParameter;
        goto ret;
    }

    /* print input parameters */
    LWSCI_INFO("Input: platformCompatibility %p\n", platformCompatibility);

    *platformCompatibility = false;

    lwRmShimLib = dlopen(LWRMSHIM_LIBRARY_NAME, RTLD_LAZY);
    if (lwRmShimLib == NULL) {
        sciErr = LwSciError_IlwalidState;
        LWSCI_ERR("Failed to open %s library\n", LWRMSHIM_LIBRARY_NAME);
        goto ret;
    }

    getVersion = dlsym(lwRmShimLib, "LwRmShimGetVersion");
    if (getVersion == NULL) {
        sciErr = LwSciError_IlwalidState;
        LWSCI_ERR("Failed to get %s symbol\n", "LwRmShimGetVersion");
        goto close_lib;
    }

    ver.major = LwRmShimMajorVersion;
    ver.minor = LwRmShimMinorVersion;
    errShim = getVersion(&ver);
    if (errShim != LWRMSHIM_OK) {
        sciErr = LwRmShimErrorToLwSciError(errShim);
        LWSCI_ERR("Failed to RmShim version\n");
        goto close_lib;
    }

    *platformCompatibility = (ver.major == LwRmShimMajorVersion);

    /* print output parameters */
    LWSCI_INFO("Output: platformCompatibility: %u\n", *platformCompatibility);

close_lib:
    if (dlclose(lwRmShimLib) != 0) {
        LWSCI_ERR("Failed to close library %s\n", LWRMSHIM_LIBRARY_NAME);
    }

    lwRmShimLib = NULL;

ret:
    LWSCI_FNEXIT("");
    return sciErr;
}

LwSciError LwSciBufDevValidateUUID(
    LwSciBufDev dev,
    LwSciBufMemDomain memDomain,
    const LwSciRmGpuId* gpuId,
    uint64_t numGpus,
    bool* isValid)
{
    LwSciError sciErr = LwSciError_Success;
    uint64_t attr[3];
    LwRmShimUuidValidationParams validationParams = {0};
    LwRmShimError errShim = LWRMSHIM_OK;

    LWSCI_FNENTRY("");

    /* verify input parameters */
    if (dev == NULL || gpuId == NULL || isValid == NULL ||
            memDomain >= LwSciBufMemDomain_UpperBound || numGpus == 0U) {
        LWSCI_ERR("Bad parameter supplied, dev %p, gpuId %p, isValid %p, memDomain %d, numGpus %d\n", dev, gpuId, isValid, memDomain, numGpus);
        sciErr = LwSciError_BadParameter;
        goto ret;
    }

    attr[0] = (uint64_t)(uintptr_t)gpuId;
    attr[1] = (uint64_t)numGpus;
    attr[2] = (uint64_t)memDomain;
    validationParams.attr = &attr[0];
    validationParams.numAttr = 3;

    errShim = dev->rmShimDevFvt.rmValidateUUID(&dev->rmSession,
            &validationParams, isValid);
    if (errShim != LWRMSHIM_OK) {
        sciErr = LwRmShimErrorToLwSciError(errShim);
        LWSCI_ERR("Failed to open resman session\n");
        goto ret;
    }

ret:
    LWSCI_FNEXIT("");
    return sciErr;
}
