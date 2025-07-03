/*
 * Copyright (c) 2019-2022, LWPU CORPORATION. All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#include "lwscibuf_test_integration.h"

#include "lwsci_test_gpu_platform_specific.h"

static struct LwStatusCodeString
{
    LW_STATUS   statusCode;
    const char *statusString;
} g_StatusCodeList[] = {
   #include "lwstatuscodes.h"
   { 0xffffffff, "Unknown error code!" } // Some compilers don't like the trailing ','
};

static DeviceInfo deviceInfo[LW0000_CTRL_GPU_MAX_PROBED_GPUS];

const char *lwstatusToString(LW_STATUS lwStatusIn)
{
    LwU32 i;
    LwU32 n = ((LwU32)(sizeof(g_StatusCodeList))/(LwU32)(sizeof(g_StatusCodeList[0])));
    for (i = 0; i < n; i++)
    {
        if (g_StatusCodeList[i].statusCode == lwStatusIn)
        {
            return g_StatusCodeList[i].statusString;
        }
    }

    return "Unknown error code!";
}

static void attachGpu(LwU32 hClient)
{
    LwU32 status;
    int i = 0;
    LwU32 gpuId = 0;
    int numProbedGpus = 0;
    LW0000_CTRL_GPU_GET_PROBED_IDS_PARAMS probedParams;
    LW0000_CTRL_GPU_ATTACH_IDS_PARAMS attachParams;
    LW0000_CTRL_GPU_GET_ID_INFO_PARAMS idInfoParams;

    // Probe GPUs present on the system
    memset(&probedParams, 0, sizeof(probedParams));
    status = LwRmControl(hClient, hClient,
                         LW0000_CTRL_CMD_GPU_GET_PROBED_IDS,
                         &probedParams,
                         sizeof(probedParams));
    if (status != LW_OK) {
        printf("LwRmControl probe gpu failed, err code: %u\t, %s\n", status, lwstatusToString(status));
        return;
    }

    for (i = 0; i < LW0000_CTRL_GPU_MAX_PROBED_GPUS; i++)
    {
        gpuId = probedParams.gpuIds[i];
        if (gpuId == LW0000_CTRL_GPU_ILWALID_ID)
        {
            numProbedGpus = i;
            while(i < LW0000_CTRL_GPU_MAX_PROBED_GPUS)
            {
                deviceInfo[i].probed = 0U;
                deviceInfo[i].attached = 0U;
                i++;
            }
        }
        else
        {
            deviceInfo[i].probed = 1U;
            deviceInfo[i].deviceId = gpuId;
            deviceInfo[i].attached = 0U;
        }
    }

    // attach gpu
    memset(&attachParams, 0, sizeof(attachParams));
    attachParams.gpuIds[0] = deviceInfo[0].deviceId;
    attachParams.gpuIds[1] = LW0000_CTRL_GPU_ILWALID_ID;

    status = LwRmControl(hClient,
                         hClient,
                         LW0000_CTRL_CMD_GPU_ATTACH_IDS,
                         &attachParams,
                         sizeof(attachParams));
    if (status != LW_OK) {
        printf("LwRmControl attach gpu failed, err code: %u\t, %s\n", status, lwstatusToString(status));
        return;
    }

    // Get device and subdevice instance
    memset(&idInfoParams, 0, sizeof(idInfoParams));
    idInfoParams.gpuId = deviceInfo[0].deviceId;

    status = LwRmControl(hClient,
                         hClient,
                         LW0000_CTRL_CMD_GPU_GET_ID_INFO,
                         &idInfoParams,
                         sizeof(idInfoParams));
    if (status != LW_OK) {
        printf("LwRmControl get gpu info failed, err code: %u\t, %s\n", status, lwstatusToString(status));
        return;
    }

    deviceInfo[0].attached = 1U;
    deviceInfo[0].deviceInstance = idInfoParams.deviceInstance;
    deviceInfo[0].subdeviceInstance = idInfoParams.subDeviceInstance;
}

static void detachGpu(LwU32 hClient)
{
    LwU32 status;
    LW0000_CTRL_GPU_DETACH_IDS_PARAMS detachParams;

    deviceInfo[0].attached = 0U;

    memset(&detachParams, 0, sizeof(detachParams));
    detachParams.gpuIds[0] = deviceInfo[0].deviceId;
    detachParams.gpuIds[1] = LW0000_CTRL_GPU_ILWALID_ID;

    status = LwRmControl(hClient,
                         hClient,
                         LW0000_CTRL_CMD_GPU_DETACH_IDS,
                         &detachParams,
                         sizeof(detachParams));
    if (status != LW_OK) {
        printf("LwRmControl detachGPU failed, err code: %u\t, %s\n", status, lwstatusToString(status));
        return;
    }
}

/*
 * For resman cases, we use the raw rmAPI to open the hClient and hDevice
 * And use that context to dup the passed in handle to do sanity tests.
 * The reason is unit test and lwscibuf are both tatically linking the RMAPI,
 * therefore we need to open the handle in unit test to have the corresponding
 * context value in the rmapi.o. This is smiliar to the LWCA use case.
 * @return For resman case, we only return the page size which should be 4096
 */
LwU64 GetMemorySize(LwSciBufRmHandle rmhandle)
{
    LwU32 hClient = 0U;
    LwU32 hDevice = 0xaa7056e1U;
    LwU32 hMemDup = 0U;
    LwU32 status;
    LW0080_ALLOC_PARAMETERS param;
    LW0041_CTRL_GET_MEM_PAGE_SIZE_PARAMS sizeParams;

    // open hClient
    LwRmAllocRoot(&hClient);

    // open hDevice
    attachGpu(hClient);
    memset(&param, 0, sizeof(LW0080_ALLOC_PARAMETERS));
    param.deviceId = deviceInfo[0].deviceInstance;
    param.hClientShare = hClient;
    param.vaMode = LW_DEVICE_ALLOCATION_VAMODE_OPTIONAL_MULTIPLE_VASPACES;
    status = LwRmAlloc(hClient, hClient, hDevice, LW01_DEVICE_0, &param);
    if (status != LW_OK) {
        printf("LwRmAlloc hDevice failed, err code: %u\t, %s\n", status, lwstatusToString(status));
        return 0;
    }

    // dup
    status = LwRmDupObject2(hClient,
                            hDevice,
                            &hMemDup,
                            rmhandle.hClient,
                            rmhandle.hMemory, 0);
    if (status != LW_OK) {
        printf("LwRmDupObject2 failed, err code: %u\t, %s\n", status, lwstatusToString(status));
        return 0;
    }

    // test
    memset(&sizeParams, 0, sizeof(sizeParams));
    status = LwRmControl(hClient,
                         hMemDup,
                         LW0041_CTRL_CMD_GET_MEM_PAGE_SIZE,
                         &sizeParams,
                         sizeof(sizeParams));
    if (status != LW_OK) {
        printf("LwRmControl get size failed, err code: %u\t, %s\n", status, lwstatusToString(status));
        return 0;
    }

    // release handle
    LwRmFree(hClient, hDevice, hMemDup);
    LwRmFree(hClient, hClient, hDevice);
    detachGpu(hClient);
    LwRmFree(hClient, hClient, hClient);

    return sizeParams.pageSize;
}

/*
 * For resman usecases, skip the access flags verification.
 */
bool CheckBufferAccessFlags(
    LwSciBufObj bufObj,
    LwSciBufRmHandle rmHandle)
{
    return true;
}

bool CompareRmHandlesAccessPermissions(
    LwSciBufRmHandle rmHandle1,
    LwSciBufRmHandle rmHandle2)
{
    return true;
}

// TODO Fill this function
bool isRMHandleFree(LwSciBufRmHandle rmHandle)
{
    return true;
}

LwU32 GetLwRmAccessFlags(
    LwSciBufAttrValAccessPerm perm)
{
    /* TODO: Implement this for X86 */
    return 0;
}
