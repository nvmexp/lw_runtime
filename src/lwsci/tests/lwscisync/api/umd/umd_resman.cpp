/*
 * Copyright (c) 2019-2021 LWPU Corporation.  All Rights Reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property and
 * proprietary rights in and to this software and related documentation.  Any
 * use, reproduction, disclosure or distribution of this software and related
 * documentation without an express license agreement from LWPU Corporation
 * is strictly prohibited.
 */

/*
 * This file illustrates the sample UMD APIs using LwSciSync internal APIs.
 */

#include <umd.h>
#include <lwscicommon_libc.h>
#include <lwstatus.h>
#include <lwscibuf_internal.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <pthread.h>
#include <lwRmApi.h>
#include <ctrl/ctrl0000.h>
#include <cl90f1.h> // FERMI_VASPACE_A
#include <string.h>
#include <ctrl/ctrl0080/ctrl0080fifo.h>
#include <ctrl/ctrl0080.h>
#include <ctrl/ctrl906f.h>
#include <ctrl/ctrla06f.h>
#include <ctrl/ctrlc36f.h>
#include <cla0b5.h> // KEPLER_DMA_COPY_A
#include <lwmisc.h>
#include <cla06fsubch.h>
#include <hwref/fermi/gf100/dev_ram.h>
#include <cl506f.h>
#include <cl906f.h>
#include <unistd.h>
#include <lwscicommon_errcolwersion.h>
#include <class/cla06f.h> // KEPLER_CHANNEL_GPFIFO_A
#include <class/cla16f.h> // KEPLER_CHANNEL_GPFIFO_B
#include <class/cla26f.h> // KEPLER_CHANNEL_GPFIFO_C
#include <class/clb06f.h> // MAXWELL_CHANNEL_GPFIFO_A
#include <class/clc06f.h> // PASCAL_CHANNEL_GPFIFO_A
#include <class/clc36f.h> // VOLTA_CHANNEL_GPFIFO_A
#include <class/clc46f.h> // TURING_CHANNEL_GPFIFO_A
#include <class/clc56f.h> // AMPERE_CHANNEL_GPFIFO_A
#include <class/clc361.h> // VOLTA_USERMODE_A
#include <class/cl0080.h> // LW01_DEVICE_0
#include <class/cl2080.h> // LW20_SUBDEVICE_0
#include <class/clc637.h> // LWC637_ALLOCATION_PARAMETERS

pthread_mutex_t attachGpuLock = PTHREAD_MUTEX_INITIALIZER;

struct TestResourcesRec
{
    // resman handles
    uint32_t hClient;
    uint32_t deviceId;
    uint32_t deviceInstance;
    uint32_t subdeviceInstance;
    uint32_t hDevice;
    uint32_t hSubdevice;
    uint32_t hVASpace;
    uint32_t hGpuInstance;

    // GPFIFO
    uint32_t hPhyMemoryGpFifo;
    void* phyAddrGpFifo;
    uint32_t hVirtMemoryGpFifo;
    LwU64 virtAddrGpFifo;

    // Channel info
    uint32_t hChannel;
    void* hChannelCtrl;
    uint32_t gpFifoClass;
    uint32_t objClass;

    // USERD
    uint32_t hPhyMemoryUserd;
    void* phyAddrUserd;

    // push buffer
    uint32_t hPhyMemoryPusher;
    void* phyAddrPusher;
    uint32_t hVirtMemoryPusher;
    LwU64 virtAddrPusher;
    uint32_t gpPut;

    // doorbell memory
    uint32_t hMemDoorBell;
    uint32_t* phyAddrDoorbell;

    // marker - job tracking semaphore
    uint32_t hPhyMemoryMarker;
    void* phyAddrMarker;
    uint32_t hVirtMemoryMarker;
    LwU64 virtAddrMarker;

    // semaphore memory
    uint32_t hPhyMemorySema;
    uint32_t hVirtMemorySema;
    void* phyAddrSema;
    LwU64 virtAddrSema;

    // external semaphore memory
    uint32_t hPhyMemorySemaExternal;
    uint64_t sizeSemaExternal;

    // Fence
    uint32_t fenceId;
    uint32_t fenceVal;
    uint32_t maxFenceId;

    LwSciSyncObj syncObj;
    uint32_t semaphoreSize;
};

#define DEVICE_HANDLE 0xaa000000
#define SUBDEVICE_HANDLE 0xab000000
#define VASPACE_HANDLE 0xbb000000
#define CHANNEL_HANDLE 0xcc000000

static uint32_t GpFifo[] =
{
    AMPERE_CHANNEL_GPFIFO_A,
    TURING_CHANNEL_GPFIFO_A,
    VOLTA_CHANNEL_GPFIFO_A,
    PASCAL_CHANNEL_GPFIFO_A,
    MAXWELL_CHANNEL_GPFIFO_A,
    KEPLER_CHANNEL_GPFIFO_C,
    KEPLER_CHANNEL_GPFIFO_B,
    KEPLER_CHANNEL_GPFIFO_A,
};

LwSciError waitOlwalue(
    uint32_t* pSemaCpu,
    uint32_t value)
{
    uint32_t count;
    count = 0;
    while (*pSemaCpu != value) {
        if (count < 2000) {
            usleep(1);
            count++;
        } else {
            return LwSciError_Timeout;
        }
    }
    return LwSciError_Success;
}

static LW_STATUS clientAlloc(
    TestResources res)
{
    LW_STATUS status = LW_OK;
    uint32_t hClient = 0;
    uint32_t i = 0;

    status = LwRmAllocRoot(&res->hClient);
    if (status != LW_OK) {
        return status;
    }

    // Probe GPUs present on the system
    LW0000_CTRL_GPU_GET_PROBED_IDS_PARAMS probedParams;
    memset(&probedParams, 0, sizeof(probedParams));
    status = LwRmControl(res->hClient, res->hClient,
            LW0000_CTRL_CMD_GPU_GET_PROBED_IDS, &probedParams,
            sizeof(probedParams));
    if (status != LW_OK) {
        return status;
    }

    // Attach GPU
    /* x86 ResMan Atttach API is not thread safe, accessing Attach API from
     *  two threads simultaniously causes FD leak.
     * Hence adding this application side global lock.
     */
    pthread_mutex_lock(&attachGpuLock);

    for (i = 0U; i < LW0000_CTRL_GPU_MAX_PROBED_GPUS; i++) {
        res->deviceId = probedParams.gpuIds[i];
        if (res->deviceId != LW0000_CTRL_GPU_ILWALID_ID) {
            LW0000_CTRL_GPU_ATTACH_IDS_PARAMS attachParams;
            memset(&attachParams, 0, sizeof(attachParams));
            attachParams.gpuIds[0] = res->deviceId;
            attachParams.gpuIds[1] = LW0000_CTRL_GPU_ILWALID_ID;
            status = LwRmControl(res->hClient, res->hClient,
                    LW0000_CTRL_CMD_GPU_ATTACH_IDS, &attachParams, sizeof(attachParams));
            if (status == LW_OK) {
                break;
            } else if ((status != LW_ERR_OPERATING_SYSTEM) &&
                    (status != LW_ERR_INSUFFICIENT_PERMISSIONS)) {
                return status;
            }
        }
    }

    pthread_mutex_unlock(&attachGpuLock);

    // Get device and subdevice instance
    LW0000_CTRL_GPU_GET_ID_INFO_PARAMS idInfoParams;
    memset(&idInfoParams, 0, sizeof(idInfoParams));
    idInfoParams.gpuId = res->deviceId;
    status = LwRmControl(res->hClient, res->hClient,
            LW0000_CTRL_CMD_GPU_GET_ID_INFO, &idInfoParams, sizeof(idInfoParams));
    if (status != LW_OK) {
        return status;
    }
    res->deviceInstance = idInfoParams.deviceInstance;
    res->subdeviceInstance = idInfoParams.subDeviceInstance;

    return status;
}

static LW_STATUS deviceAlloc(
    TestResources res)
{
    res->hDevice = DEVICE_HANDLE;
    LW0080_ALLOC_PARAMETERS params;
    memset(&params, 0, sizeof(LW0080_ALLOC_PARAMETERS));
    params.deviceId = res->deviceInstance;
    params.hClientShare = res->hClient;
    params.vaMode = LW_DEVICE_ALLOCATION_VAMODE_OPTIONAL_MULTIPLE_VASPACES;
    return LwRmAlloc(res->hClient, res->hClient, res->hDevice, LW01_DEVICE_0,
            &params);
}

static LW_STATUS subdeviceAlloc(
    TestResources res)
{
    res->hSubdevice = SUBDEVICE_HANDLE;
    LW2080_ALLOC_PARAMETERS params;
    memset(&params, 0, sizeof(params));
    params.subDeviceId = res->subdeviceInstance;
    return LwRmAlloc(res->hClient, res->hDevice, res->hSubdevice,
            LW20_SUBDEVICE_0, &params);
}

static LW_STATUS allocVASpace(
    TestResources res)
{
    res->hVASpace = VASPACE_HANDLE;
    LW_VASPACE_ALLOCATION_PARAMETERS params;
    memset(&params, 0, sizeof(LW_VASPACE_ALLOCATION_PARAMETERS));
    params.index = LW_VASPACE_ALLOCATION_INDEX_GPU_NEW;
    return LwRmAlloc(res->hClient, res->hDevice, res->hVASpace,
            FERMI_VASPACE_A, &params);
}

static LW_STATUS physMemAlloc(
    uint32_t hClient,
    uint32_t hParent,
    uint32_t *hMemory,
    uint64_t *size)
{
    LW_STATUS status = LW_OK;
    uint32_t allocAttrFlags = 0;

    allocAttrFlags = FLD_SET_DRF(OS32, _ATTR, _LOCATION, _PCI, allocAttrFlags);
    allocAttrFlags = FLD_SET_DRF(OS32, _ATTR, _PHYSICALITY, _CONTIGUOUS,
                                 allocAttrFlags);
    allocAttrFlags = FLD_SET_DRF(OS32, _ATTR, _DEPTH, _8, allocAttrFlags);
    allocAttrFlags = FLD_SET_DRF(OS32, _ATTR, _COHERENCY, _UNCACHED,
                                 allocAttrFlags);
    allocAttrFlags = FLD_SET_DRF(OS32, _ATTR, _PAGE_SIZE, _4KB,
                                 allocAttrFlags);

    LWOS32_PARAMETERS allocParams;
    memset(&allocParams, 0, sizeof(allocParams));

    allocParams.hRoot = hClient;
    allocParams.hObjectParent = hParent;

    allocParams.function = LWOS32_FUNCTION_ALLOC_SIZE;
    allocParams.data.AllocSize.hMemory = *hMemory;
    allocParams.data.AllocSize.owner = hParent;
    allocParams.data.AllocSize.type = LWOS32_TYPE_IMAGE;
    allocParams.data.AllocSize.flags = 0;
    allocParams.data.AllocSize.attr = allocAttrFlags;
    allocParams.data.AllocSize.attr2 = 0;
    allocParams.data.AllocSize.offset = 0;
    allocParams.data.AllocSize.size = *size;

    status = LwRmVidHeapControl((void*)(&allocParams));
    if (status != LW_OK) {
        return status;
    }

    *hMemory = allocParams.data.AllocSize.hMemory;
    *size = allocParams.data.AllocSize.size;

    return status;
}

static LW_STATUS gpuVirtMemAlloc(
    uint32_t hClient,
    uint32_t hParent,
    uint32_t hVASpace,
    uint64_t size,
    uint32_t *hMemory)
{
    LW_STATUS status = LW_OK;

    uint32_t gpuVirtMemFlags = LWOS32_ALLOC_FLAGS_VIRTUAL;

    LWOS32_PARAMETERS params;
    memset(&params, 0, sizeof(params));

    params.hRoot = hClient;
    params.hObjectParent = hParent;
    params.hVASpace = hVASpace;
    params.function = LWOS32_FUNCTION_ALLOC_SIZE;
    params.data.AllocSize.hMemory = *hMemory;
    params.data.AllocSize.owner = hParent;
    params.data.AllocSize.type = LWOS32_TYPE_IMAGE;
    params.data.AllocSize.flags = gpuVirtMemFlags;
    params.data.AllocSize.attr = 0;
    params.data.AllocSize.attr2 = 0;
    params.data.AllocSize.size = size;
    status = LwRmVidHeapControl((void*)(&params));
    if (status != LW_OK) {
        return status;
    }
    *hMemory = params.data.AllocSize.hMemory;

    return status;
}

static LwU64 GetUserDSize(
    uint32_t hClass)
{
    switch (hClass) {
        case KEPLER_CHANNEL_GPFIFO_A:
            return sizeof(KeplerAControlGPFifo);
        case KEPLER_CHANNEL_GPFIFO_B:
            return sizeof(KeplerBControlGPFifo);
        case KEPLER_CHANNEL_GPFIFO_C:
            return sizeof(KeplerCControlGPFifo);
        case MAXWELL_CHANNEL_GPFIFO_A:
            return sizeof(MaxwellAControlGPFifo);
        case PASCAL_CHANNEL_GPFIFO_A:
            return sizeof(PascalAControlGPFifo);
        case VOLTA_CHANNEL_GPFIFO_A:
            return sizeof(VoltaAControlGPFifo);
        case TURING_CHANNEL_GPFIFO_A:
            return sizeof(TuringAControlGPFifo);
        case AMPERE_CHANNEL_GPFIFO_A:
            return sizeof(AmpereAControlGPFifo);
        default:
            printf("Unrecognized channel class 0x%x", hClass);
            return 0U;
    }
}

static LW_STATUS setSupportedClass(TestResources res,
    uint32_t* supportedList,
    uint32_t size,
    uint32_t engineType,
    uint32_t* hClass)
{
    LW_STATUS status = LW_OK;
    uint32_t* pSupportedClassList = NULL;
    uint32_t i = 0U;
    uint32_t j = 0U;
    // Figure out how many classes we need to allocate memory for
    LW2080_CTRL_GPU_GET_ENGINE_CLASSLIST_PARAMS params;
    memset(&params, 0, sizeof(params));
    params.engineType = engineType;
    status = LwRmControl(res->hClient, res->hSubdevice,
            LW2080_CTRL_CMD_GPU_GET_ENGINE_CLASSLIST,
              &params, sizeof(params));
    if (status != LW_OK) {
        return status;
    }
    // Call again to get the list of classes
    pSupportedClassList = new uint32_t[params.numClasses];
    params.classList = reinterpret_cast<LwP64>(pSupportedClassList);
    status = LwRmControl(res->hClient, res->hSubdevice,
            LW2080_CTRL_CMD_GPU_GET_ENGINE_CLASSLIST,
              &params, sizeof(params));
    if (status != LW_OK) {
        delete[] pSupportedClassList;
        return status;
    }
    for (i = 0; i < size; i++) {
        for (j = 0; j < params.numClasses; j++) {
            if (supportedList[i] == pSupportedClassList[j]) {
                *hClass = supportedList[i];
                goto done;
            }
        }
    }
done:
    delete[] pSupportedClassList;
    return status;
}

static LW_STATUS setupMigHandles(
    TestResources res)
{
    LW_STATUS err = LW_OK;
    LwU32 i;
    /* Get MIG mode info */
    LW2080_CTRL_GPU_GET_INFO_V2_PARAMS infoparams;
    memset(&infoparams, 0x0, sizeof(infoparams));
    infoparams.gpuInfoList[0].index = LW2080_CTRL_GPU_INFO_INDEX_GPU_SMC_MODE;
    infoparams.gpuInfoListSize = 1;
    err = LwRmControl(res->hClient,
                      res->hSubdevice,
                      LW2080_CTRL_CMD_GPU_GET_INFO_V2,
                      &infoparams,
                      sizeof(infoparams));
    if (err != LW_OK) {
        goto ret;
    }
    if (infoparams.gpuInfoList[0].data == LW2080_CTRL_GPU_INFO_GPU_SMC_MODE_ENABLED) {
        LW2080_CTRL_GPU_GET_ACTIVE_PARTITION_IDS_PARAMS activeParams;
        memset(&activeParams, 0x0, sizeof(activeParams));
        err = LwRmControl(res->hClient,
                          res->hSubdevice,
                          LW2080_CTRL_CMD_GPU_GET_ACTIVE_PARTITION_IDS,
                          &activeParams,
                          sizeof(activeParams));
        if (err != LW_OK) {
            goto ret;
        }
        for (i = 0; i < activeParams.partitionCount; ++i) {
            if (activeParams.swizzId[i] != LWC637_DEVICE_LEVEL_SWIZZID) {
                res->hGpuInstance = 0xad000000;
                LWC637_ALLOCATION_PARAMETERS smcAllocParams;
                memset(&smcAllocParams, 0x0, sizeof(smcAllocParams));
                smcAllocParams.swizzId = activeParams.swizzId[i];
                err = LwRmAlloc(res->hClient,
                                res->hSubdevice,
                                res->hGpuInstance,
                                AMPERE_SMC_PARTITION_REF,
                                &smcAllocParams);
                if ((err != LW_ERR_OPERATING_SYSTEM) &&
                        (err != LW_ERR_INSUFFICIENT_PERMISSIONS)) {
                    goto ret;
                }
            }
        }
    }
ret:
    return err;
}

static LW_STATUS createChannel(
    TestResources res)
{
    LW_STATUS status = LW_OK;

    uint64_t mSize;
    uint32_t NUM_GPFIFOS = 128;
    uint32_t GPFIFO_SIZE = NUM_GPFIFOS * 8;

    // GPFIFO
    status = setSupportedClass(res, GpFifo, sizeof(GpFifo),
            LW2080_ENGINE_TYPE_HOST, &res->gpFifoClass);
    if (status != LW_OK) {
        return status;
    }
    mSize = GPFIFO_SIZE;
    status = physMemAlloc(res->hClient, res->hDevice,
        &res->hPhyMemoryGpFifo, &mSize);
    if (status != LW_OK) {
        return status;
    }

    status = gpuVirtMemAlloc(res->hClient, res->hDevice, res->hVASpace,
        mSize, &res->hVirtMemoryGpFifo);
    if (status != LW_OK) {
        return status;
    }

    status = LwRmMapMemory(res->hClient, res->hDevice, res->hPhyMemoryGpFifo,
            0, mSize,
            &res->phyAddrGpFifo, 0);
    if (status != LW_OK) {
        return status;
    }

    uint32_t flags = 0;
    flags = FLD_SET_DRF(OS46, _FLAGS, _ACCESS, _READ_WRITE, flags);
    flags = FLD_SET_DRF(OS46, _FLAGS, _CACHE_SNOOP, _ENABLE, flags);

    status = LwRmMapMemoryDma(res->hClient, res->hDevice,
            res->hVirtMemoryGpFifo, res->hPhyMemoryGpFifo, 0,
            mSize, flags,
            static_cast<LwU64*>(&res->virtAddrGpFifo));
    if (status != LW_OK) {
        return status;
    }

    res->hChannel = CHANNEL_HANDLE+1;

    LW_CHANNELGPFIFO_ALLOCATION_PARAMETERS gpfifoParams;
    memset(&gpfifoParams, 0, sizeof(LW_CHANNELGPFIFO_ALLOCATION_PARAMETERS));

    // For Volta+, USERD is no longer allocated by the RM, hence has to be
    // managed in test app. When allocating the channel, USERD memory handle
    // will be passed through LW_CHANNELGPFIFO_ALLOCATION_PARAMETERS.
    if (res->gpFifoClass >= AMPERE_CHANNEL_GPFIFO_A) {
        mSize = GetUserDSize(res->gpFifoClass);
        status = physMemAlloc(res->hClient, res->hDevice,
            &res->hPhyMemoryUserd, &mSize);
        if (status != LW_OK) {
            return status;
        }
        gpfifoParams.hUserdMemory[0] = res->hPhyMemoryUserd;
        gpfifoParams.userdOffset[0]  = 0;
    }

    gpfifoParams.hObjectBuffer = res->hVirtMemoryGpFifo;
    gpfifoParams.gpFifoOffset  = res->virtAddrGpFifo;
    gpfifoParams.gpFifoEntries = NUM_GPFIFOS;
    gpfifoParams.engineType    = LW2080_ENGINE_TYPE_GRAPHICS;
    gpfifoParams.flags         = 0;
    gpfifoParams.hVASpace      = res->hVASpace;

    status = LwRmAlloc(res->hClient, res->hDevice, res->hChannel, res->gpFifoClass,
            &gpfifoParams);
    if (status != LW_OK) {
        return status;
    }

    if (res->gpFifoClass < AMPERE_CHANNEL_GPFIFO_A) {
        flags = 0;
        flags = FLD_SET_DRF(OS33, _FLAGS, _MAPPING, _DEFAULT, flags);
        flags = FLD_SET_DRF(OS33, _FLAGS, _ACCESS, _READ_WRITE, flags);
        status = LwRmMapMemory(res->hClient, res->hDevice,
                res->hChannel, 0, GetUserDSize(res->gpFifoClass),
                &res->hChannelCtrl, flags);
        if (status != LW_OK) {
            return status;
        }
    } else {
        status = LwRmMapMemory(res->hClient, res->hDevice,
                res->hPhyMemoryUserd, 0, GetUserDSize(res->gpFifoClass),
                &res->phyAddrUserd, 0);
        if (status != LW_OK) {
            return status;
        }
        res->hChannelCtrl = (char*)res->phyAddrUserd + gpfifoParams.userdOffset[0];
    }

    LWA06F_CTRL_GPFIFO_SCHEDULE_PARAMS gpFifoSchedulParams;
    memset(&gpFifoSchedulParams, 0, sizeof(gpFifoSchedulParams));
    gpFifoSchedulParams.bEnable = true;
    status = LwRmControl(res->hClient, res->hChannel,
              LWA06F_CTRL_CMD_GPFIFO_SCHEDULE, &gpFifoSchedulParams,
              sizeof(gpFifoSchedulParams));
    if (status != LW_OK) {
        return status;
    }

    return status;
}

static LW_STATUS createPusher(
    TestResources res,
    uint32_t pushbufferSize)
{
    LW_STATUS status = LW_OK;
    uint64_t mSize = pushbufferSize;

    status = physMemAlloc(res->hClient, res->hDevice,
        &res->hPhyMemoryPusher, &mSize);
    if (status != LW_OK) {
        return status;
    }

    status = gpuVirtMemAlloc(res->hClient, res->hDevice, res->hVASpace,
        mSize, &res->hVirtMemoryPusher);
    if (status != LW_OK) {
        return status;
    }

    status = LwRmMapMemory(res->hClient, res->hDevice, res->hPhyMemoryPusher,
            0, mSize,
            &res->phyAddrPusher, 0);
    if (status != LW_OK) {
        return status;
    }

    uint32_t flags = 0;
    flags = FLD_SET_DRF(OS46, _FLAGS, _ACCESS, _READ_WRITE, flags);
    flags = FLD_SET_DRF(OS46, _FLAGS, _CACHE_SNOOP, _ENABLE, flags);

    status = LwRmMapMemoryDma(res->hClient, res->hDevice, res->hVirtMemoryPusher,
            res->hPhyMemoryPusher,
            0, mSize,
            flags, static_cast<LwU64*>(&res->virtAddrPusher));
    if (status != LW_OK) {
        return status;
    }

    return status;
}

static LW_STATUS createMarkerSema(
    TestResources res)
{
    LW_STATUS status = LW_OK;
    uint64_t mSize = 4U;

    status = physMemAlloc(res->hClient, res->hDevice,
        &res->hPhyMemoryMarker, &mSize);
    if (status != LW_OK) {
        return status;
    }

    status = gpuVirtMemAlloc(res->hClient, res->hDevice, res->hVASpace,
        mSize, &res->hVirtMemoryMarker);
    if (status != LW_OK) {
        return status;
    }

    status = LwRmMapMemory(res->hClient, res->hDevice, res->hPhyMemoryMarker,
            0, mSize, &res->phyAddrMarker, 0);
    if (status != LW_OK) {
        return status;
    }

    uint32_t flags = 0;
    flags = FLD_SET_DRF(OS46, _FLAGS, _ACCESS, _READ_WRITE, flags);
    flags = FLD_SET_DRF(OS46, _FLAGS, _CACHE_SNOOP, _ENABLE, flags);

    status = LwRmMapMemoryDma(res->hClient, res->hDevice, res->hVirtMemoryMarker,
            res->hPhyMemoryMarker, 0, mSize, flags,
            static_cast<LwU64*>(&res->virtAddrMarker));
    if (status != LW_OK) {
        return status;
    }

    return status;
}

static LW_STATUS createExternalSema(
    TestResources res)
{
    LW_STATUS status = LW_OK;
    uint64_t semaCnt = 2U;
    uint64_t semaSize = 16U;
    res->sizeSemaExternal = (semaSize * semaCnt);
    uint64_t mSize = res->sizeSemaExternal;

    status = physMemAlloc(res->hClient, res->hDevice,
        &res->hPhyMemorySemaExternal, &mSize);
    if (status != LW_OK) {
        return status;
    }

    return status;
}

static uint32_t Command(
    uint32_t subCh,
    uint32_t method,
    uint32_t count)
{
    return (DRF_DEF(_FIFO, _DMA, _INCR_OPCODE, _VALUE) |
            DRF_NUM(_FIFO, _DMA, _INCR_COUNT, count) |
            DRF_NUM(_FIFO, _DMA, _INCR_SUBCHANNEL, subCh) |
            DRF_NUM(_FIFO, _DMA, _INCR_ADDRESS, method>>2));
}

static LwSciError mapSemaphore(
    TestResources res,
    LwSciSyncObj syncObj)
{
    LW_STATUS status = LW_OK;
    LwSciError sciErr;
    LwSciSyncSemaphoreInfo semaphoreInfo;
    LwSciBufRmHandle memHandle;
    uint64_t offset;
    uint64_t len;
    uint32_t flags;

    sciErr = LwSciSyncObjGetSemaphoreInfo(syncObj, 0, &semaphoreInfo);
    if (sciErr != LwSciError_Success) {
        return sciErr;
    }

    res->semaphoreSize = semaphoreInfo.semaphoreSize;
    res->syncObj = syncObj;

    sciErr = LwSciBufObjGetMemHandle(semaphoreInfo.bufObj, &memHandle, &offset,
            &len);
    if (sciErr != LwSciError_Success) {
        return sciErr;
    }

    status = LwRmDupObject2(res->hClient, res->hDevice, &res->hPhyMemorySema,
            memHandle.hClient, memHandle.hMemory, 0);
    if (status != LW_OK) {
        goto fn_exit;
    }

    status = LwRmMapMemory(res->hClient, res->hDevice, res->hPhyMemorySema,
            0, len, &res->phyAddrSema, 0);
    if (status != LW_OK) {
        goto fn_exit;
    }
    status = gpuVirtMemAlloc(res->hClient, res->hDevice, res->hVASpace,
        len, &res->hVirtMemorySema);
    if (status != LW_OK) {
        goto fn_exit;
    }

    flags = 0;
    flags = FLD_SET_DRF(OS46, _FLAGS, _ACCESS, _READ_WRITE, flags);
    flags = FLD_SET_DRF(OS46, _FLAGS, _CACHE_SNOOP, _ENABLE, flags);

    status = LwRmMapMemoryDma(res->hClient, res->hDevice, res->hVirtMemorySema,
            res->hPhyMemorySema, 0, len,
            flags, static_cast<LwU64*>(&res->virtAddrSema));
    if (status != LW_OK) {
        goto fn_exit;
    }

    sciErr = LwSciSyncObjGetNumPrimitives(syncObj, &res->maxFenceId);
    if (sciErr != LwSciError_Success) {
        return sciErr;
    }

fn_exit:
    return sciErr;
}

static void ringDoorBell(
    TestResources res)
{
    if (res->gpFifoClass >= VOLTA_CHANNEL_GPFIFO_A) {
        // VOLTA and above
        LW_STATUS status = LW_OK;
        uint32_t token;
        LWC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN_PARAMS params;

        memset(&params, 0, sizeof(params));
        status = LwRmControl(res->hClient, res->hChannel,
                    LWC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN, &params,
                    sizeof(params));
        if (status != LW_OK) {
            printf("failed to get work token error = 0x%x - %s\n", status,
                LwStatusToString(status));
            return;
        }

        token = params.workSubmitToken;

        *res->phyAddrDoorbell = token;
    }
}

static LW_STATUS mapDoorBell(
    TestResources res)
{
    LW_STATUS status = LW_OK;
    if (res->gpFifoClass >= VOLTA_CHANNEL_GPFIFO_A) {
        // map doorbell
        res->hMemDoorBell = 0xaabbccdd;
        status = LwRmAlloc(res->hClient,
                          res->hSubdevice,
                          res->hMemDoorBell,
                          VOLTA_USERMODE_A,
                          NULL);

        if (status != LW_OK) {
            printf("failed to alloc door bell memory error = 0x%x - %s\n", status,
                LwStatusToString(status));
            goto fn_exit;
        }

        status = LwRmMapMemory(res->hClient,
                            res->hSubdevice,
                            res->hMemDoorBell,
                            LWC361_NOTIFY_CHANNEL_PENDING,
                            sizeof(*res->phyAddrDoorbell),
                            (void**)&res->phyAddrDoorbell,
                            0);

        if (status != LW_OK) {
            printf("failed to map doorbell mem error = 0x%x - %s\n", status,
                LwStatusToString(status));
            goto fn_exit;
        }
    }

fn_exit:
    return status;
}

static void kickOffPushBuffer(
    TestResources res,
    uint32_t length)
{
    uint64_t pbGpuVA;
    pbGpuVA = res->virtAddrPusher;
    uint32_t gpEntry0;
    uint32_t gpEntry1;
    gpEntry0 = DRF_NUM(506F, _GP_ENTRY0, _GET, LwU64_LO32(pbGpuVA) >> 2);
    gpEntry1 = DRF_NUM(506F, _GP_ENTRY1, _GET_HI, LwU64_HI32(pbGpuVA)) |
               DRF_NUM(506F, _GP_ENTRY1, _LENGTH, length);

    uint32_t* pGpFifoBase;
    pGpFifoBase = static_cast<uint32_t*>(res->phyAddrGpFifo);
    pGpFifoBase[res->gpPut*2+0] = gpEntry0;
    pGpFifoBase[res->gpPut*2+1] = gpEntry1;
    res->gpPut = (res->gpPut+1) % 128;
    LwA16FControl* pChannelCtrl;
    pChannelCtrl = static_cast<LwA16FControl*>(res->hChannelCtrl);
    pChannelCtrl->GPPut = res->gpPut;

    ringDoorBell(res);
}

static void fillCmdBufHostSemaphoreRelease(
    TestResources res,
    uint64_t semaphoreVa,
    uint32_t payload)
{
    uint32_t* pPbLwrrent;
    pPbLwrrent = static_cast<uint32_t*>(res->phyAddrPusher);
    uint32_t semaphoreD;
    uint32_t method;

    if (res->gpFifoClass < 0xB000) {
        // KEPLER
        semaphoreD = DRF_DEF(A06F, _SEMAPHORED, _OPERATION, _RELEASE) |
                     DRF_DEF(A06F, _SEMAPHORED, _RELEASE_SIZE, _4BYTE);
        method = LWA06F_SEMAPHOREA;
    } else {
        // MAXWELL and above
        semaphoreD = DRF_DEF(B06F, _SEMAPHORED, _OPERATION, _RELEASE) |
                     DRF_DEF(B06F, _SEMAPHORED, _RELEASE_SIZE, _4BYTE);
        method = LWB06F_SEMAPHOREA;
    }
    uint32_t d0 = (uint32_t)(semaphoreVa >> 32);
    uint32_t d1 = (uint32_t)(semaphoreVa & 0xFFFFFFFF);
    uint32_t d2 = payload;
    uint32_t d3 = semaphoreD;
    pPbLwrrent[0] = Command(LWA06F_SUBCHANNEL_3D, method, 4);
    pPbLwrrent[1] = d0;
    pPbLwrrent[2] = d1;
    pPbLwrrent[3] = d2;
    pPbLwrrent[4] = d3;
}

static void fillCmdBufSemaphoreWait(
    TestResources res,
    uint64_t semaphoreVa,
    uint32_t payload)
{
    uint32_t* pPbLwrrent;
    uint32_t semaphoreD;
    uint32_t method;
    pPbLwrrent = static_cast<uint32_t*>(res->phyAddrPusher);
    if (res->gpFifoClass < 0xB000) {
        // KEPLER
        semaphoreD = DRF_DEF(A06F, _SEMAPHORED, _OPERATION, _ACQ_GEQ) |
                     DRF_DEF(A06F, _SEMAPHORED, _ACQUIRE_SWITCH, _ENABLED);
        method = LWA06F_SEMAPHOREA;
    } else {
        // MAXWELL and above
        semaphoreD = DRF_DEF(B06F, _SEMAPHORED, _OPERATION, _ACQ_GEQ) |
                     DRF_DEF(B06F, _SEMAPHORED, _ACQUIRE_SWITCH, _ENABLED);
        method = LWB06F_SEMAPHOREA;
    }
    uint32_t d0 = (uint32_t)(semaphoreVa >> 32);
    uint32_t d1 = (uint32_t)(semaphoreVa & 0xFFFFFFFF);
    uint32_t d2 = payload;
    uint32_t d3 = semaphoreD;
    pPbLwrrent[0] = Command(LWA06F_SUBCHANNEL_3D, method, 4);
    pPbLwrrent[1] = d0;
    pPbLwrrent[2] = d1;
    pPbLwrrent[3] = d2;
    pPbLwrrent[4] = d3;
}

static LwSciError submitNoopSignaler(
    TestResources res,
    uint32_t* id,
    uint32_t* value)
{
    LwSciError err;
    LwSciError sciErr;
    LwSciSyncSemaphoreInfo semaphoreInfo;
    uint32_t* pSemaCpu = (uint32_t*)(res->phyAddrSema) +
            (res->semaphoreSize/sizeof(uint32_t))*res->fenceId;

    // Wait for submitted job to complete
    err = waitOlwalue(pSemaCpu, res->fenceVal);
    if (err != LwSciError_Success) {
        return err;
    }

    res->fenceId = (res->fenceId+1)%(res->maxFenceId);

    // Using LwSciSyncObjGetSemaphoreInfo() only in submitNoopSignaler()
    // and not in submitNoopWaiter() to validate offset callwlation.
    sciErr = LwSciSyncObjGetSemaphoreInfo(res->syncObj, res->fenceId,
            &semaphoreInfo);
    if (sciErr != LwSciError_Success) {
        return sciErr;
    }

    // Push semaphore release methods
    uint64_t gpuAddr = res->virtAddrSema + semaphoreInfo.offset;

    fillCmdBufHostSemaphoreRelease(res, gpuAddr, ++(res->fenceVal));

    kickOffPushBuffer(res, 5);

    *id = res->fenceId;
    *value = res->fenceVal;


    return LwSciError_Success;
}

LwSciError umdGetPostLwSciSyncFence(
    LwSciSyncObj syncObj,
    TestResources resource,
    LwSciSyncFence* syncFence)
{
    LwSciError err;
    uint32_t id;
    uint32_t value;

    // Submit no-op to GPU
    err = submitNoopSignaler(resource, &id, &value);
    if (err != LwSciError_Success) {
        return err;
    }

    err = LwSciSyncFenceUpdateFence(syncObj, id, value, syncFence);
    if (err != LwSciError_Success) {
        printf("err = %d\n", err);
        return err;
    }

    return LwSciError_Success;
}

static LwSciError submitNoopWaiter(
    TestResources res,
    uint32_t id,
    uint32_t value)
{
    LwSciError err;
    // Push semaphore acquire methods
    uint32_t* pMarkerCpu = (uint32_t*)(res->phyAddrMarker);
    uint64_t semaGpuAddr = res->virtAddrSema + res->semaphoreSize*id;
    uint64_t markerGpuAddr = res->virtAddrMarker;

    fillCmdBufSemaphoreWait(res, semaGpuAddr, value);

    fillCmdBufHostSemaphoreRelease(res, markerGpuAddr, ++(res->fenceVal));

    kickOffPushBuffer(res, 10);

    // Wait for submitted job to complete
    err = waitOlwalue(pMarkerCpu, res->fenceVal);
    if (err != LwSciError_Success) {
        return err;
    }

    return LwSciError_Success;
}

LwSciError umdWaitOnPreLwSciSyncFence(
    TestResources resource,
    LwSciSyncFence* syncFence)
{
    LwSciError err = LwSciError_Success;
    uint64_t id;
    uint64_t value;

    err = LwSciSyncFenceExtractFence(syncFence, &id, &value);
    if (err != LwSciError_Success) {
        printf("err = %d\n", err);
        return err;
    }
    // Submit no-op to GPU with pre-fence
    err = submitNoopWaiter(resource, id, value);
    if (err != LwSciError_Success) {
        return err;
    }

    return LwSciError_Success;
}

LwSciError LwRmGpu_TestSetup(
    TestResources* resources)
{
    LW_STATUS status = LW_OK;

    TestResources res = (TestResources)
            calloc(1, sizeof(struct TestResourcesRec));
    if (res == NULL) {
        printf("Failed to allocate memory\n");
        return LwSciError_InsufficientMemory;
    }

    // Allocate a new client
    status = clientAlloc(res);
    if (status != LW_OK) {
        goto fn_exit;
    }

    // Allocate a new device
    status = deviceAlloc(res);
    if (status != LW_OK) {
        goto fn_exit;
    }

    // Create sub-device
    status = subdeviceAlloc(res);
    if (status != LW_OK) {
        goto fn_exit;
    }

    // Subscibe to MIG GI if MIG mode enabled
    status = setupMigHandles(res);
    if (status != LW_OK) {
        goto fn_exit;
    }

    // Create VA Space
    status = allocVASpace(res);
    if (status != LW_OK) {
        goto fn_exit;
    }

    // Create channel
    status = createChannel(res);
    if (status != LW_OK) {
        goto fn_exit;
    }

    // map doorbell
    status = mapDoorBell(res);
    if (status != LW_OK) {
        goto fn_exit;
    }

    // Create pushbuffer
    status = createPusher(res, 0x1000);
    if (status != LW_OK) {
        goto fn_exit;
    }

    // Create marker
    status = createMarkerSema(res);
    if (status != LW_OK) {
        goto fn_exit;
    }

    // Create external semaphore
    status = createExternalSema(res);
    if (status != LW_OK) {
        goto fn_exit;
    }

    *resources = res;

fn_exit:
    return LwStatusToLwSciErr(status);
}

LwSciError LwRmGpu_TestMapSemaphore(
    TestResources res,
    LwSciSyncObj syncObj)
{
    LW_STATUS status = LW_OK;

    // Map semaphore
    status = mapSemaphore(res, syncObj);
    if (status != LW_OK) {
        goto fn_exit;
    }

fn_exit:
    return LwStatusToLwSciErr(status);
}

LW_STATUS MemoryTeardown(
    TestResources res,
    uint32_t hPhyMemory,
    void* cpuAddr,
    uint32_t hVirtMemory,
    uint64_t virtMemOffset)
{
    LW_STATUS status = LW_OK;

    status = LwRmUnmapMemory(res->hClient, res->hDevice, hPhyMemory,
            cpuAddr, 0);
    if (status != LW_OK) {
        goto fn_exit;
    }
    status = LwRmUnmapMemoryDma(res->hClient, res->hDevice, hVirtMemory,
            hPhyMemory, 0, virtMemOffset);
    if (status != LW_OK) {
        goto fn_exit;
    }
    status = LwRmFree(res->hClient, res->hDevice, hPhyMemory);
    if (status != LW_OK) {
        goto fn_exit;
    }
    status = LwRmFree(res->hClient, res->hDevice, hVirtMemory);
    if (status != LW_OK) {
        goto fn_exit;
    }

fn_exit:
    return status;
}

void LwRmGpu_TestTeardown(TestResources res)
{
    LW_STATUS status = LW_OK;

    if (res->gpFifoClass >= VOLTA_CHANNEL_GPFIFO_A) {
        // unmap doorbell memory
        status = LwRmUnmapMemory(res->hClient, res->hSubdevice,
                    res->hMemDoorBell, res->phyAddrDoorbell, 0U);
        if (status != LW_OK) {
            goto fn_exit;
        }

        // free doorbell memory
        status = LwRmFree(res->hClient, res->hSubdevice, res->hMemDoorBell);
        if (status != LW_OK) {
            goto fn_exit;
        }
    }

    if (res->gpFifoClass >= AMPERE_CHANNEL_GPFIFO_A) {
        // unmap userd memory
        status = LwRmUnmapMemory(res->hClient, res->hDevice,
                    res->hPhyMemoryUserd, res->phyAddrUserd, 0U);
        if (status != LW_OK) {
            goto fn_exit;
        }

        // free userd memory
        status = LwRmFree(res->hClient, res->hDevice, res->hPhyMemoryUserd);
        if (status != LW_OK) {
            goto fn_exit;
        }
    }

    if (res->gpFifoClass < AMPERE_CHANNEL_GPFIFO_A) {
        status = LwRmUnmapMemory(res->hClient, res->hDevice, res->hChannel,
                    res->hChannelCtrl, 0U);
        if (status != LW_OK) {
            goto fn_exit;
        }
    }

    // free semaphore memory
    status = MemoryTeardown(res, res->hPhyMemorySema, res->phyAddrSema,
            res->hVirtMemorySema, res->virtAddrSema);
    if (status != LW_OK) {
        goto fn_exit;
    }

    // free marker memory
    status = MemoryTeardown(res, res->hPhyMemoryMarker, res->phyAddrMarker,
            res->hVirtMemoryMarker, res->virtAddrMarker);
    if (status != LW_OK) {
        goto fn_exit;
    }

    // free pusher memory
    status = MemoryTeardown(res, res->hPhyMemoryPusher, res->phyAddrPusher,
            res->hVirtMemoryPusher, res->virtAddrPusher);
    if (status != LW_OK) {
        goto fn_exit;
    }

    // free GpFifo memory
    status = MemoryTeardown(res, res->hPhyMemoryGpFifo, res->phyAddrGpFifo,
            res->hVirtMemoryGpFifo, res->virtAddrGpFifo);
    if (status != LW_OK) {
        goto fn_exit;
    }

    // free external semaphore memory
    status = LwRmFree(res->hClient, res->hDevice, res->hPhyMemorySemaExternal);
    if (status != LW_OK) {
        goto fn_exit;
    }

    // Free resman handles
    (void)LwRmFree(res->hClient, res->hSubdevice, res->hGpuInstance);
    status = LwRmFree(res->hClient, res->hDevice, res->hChannel);
    if (status != LW_OK) {
        goto fn_exit;
    }
    status = LwRmFree(res->hClient, res->hDevice, res->hSubdevice);
    if (status != LW_OK) {
        goto fn_exit;
    }
    status = LwRmFree(res->hClient, res->hDevice, res->hVASpace);
    if (status != LW_OK) {
        goto fn_exit;
    }
    status = LwRmFree(res->hClient, res->hClient, res->hDevice);
    if (status != LW_OK) {
        goto fn_exit;
    }

    // detach GPU
    LW0000_CTRL_GPU_DETACH_IDS_PARAMS detachParams;
    detachParams.gpuIds[0] = res->deviceId;
    detachParams.gpuIds[1] = LW0000_CTRL_GPU_ILWALID_ID;
    status = LwRmControl(res->hClient, res->hClient, LW0000_CTRL_CMD_GPU_DETACH_IDS,
            &detachParams, sizeof(detachParams));
    if (status != LW_OK) {
        goto fn_exit;
    }

    status = LwRmFree(res->hClient, res->hClient, res->hClient);
    if (status != LW_OK) {
        goto fn_exit;
    }

    free(res);

fn_exit:
    if (status != LW_OK) {
        printf("Error: %d\n", LwStatusToLwSciErr(status));
    }
}

LwSciError MockUmdSignalStreamFrame(
    LwSciSyncObj syncObj,
    TestResources resource,
    LwSciSyncFence* syncFence,
    uint32_t slotIndex)
{
    LwSciError err;
    uint32_t id;
    uint32_t value;

    /* Submit no-op to GPU */
    err = submitNoopSignaler(resource, &id, &value);
    if (err != LwSciError_Success) {
        return err;
    }

    err = LwSciSyncFenceUpdateFenceWithTimestamp(syncObj, id,
            value, slotIndex, syncFence);
    if (err != LwSciError_Success) {
        printf("err = %d\n", err);
        return err;
    }

    return LwSciError_Success;
}

#ifdef LWSCISYNC_EMU_SUPPORT
LwSciError umdAddExternalPrimitiveInfo(
    LwSciSyncAttrList list,
    TestResources res)
{
    LwSciError err = LwSciError_Success;
    LwSciSyncInternalAttrKeyValuePair internalKeyValue;
    LwSciSyncSemaphorePrimitiveInfo semaPrimitiveInfo;
    LwSciBufRmHandle memHandle;

    memHandle.hClient = res->hClient;
    memHandle.hDevice = res->hDevice;
    memHandle.hMemory = res->hPhyMemorySemaExternal;
    semaPrimitiveInfo = {
        .primitiveType = LwSciSyncInternalAttrValPrimitiveType_SysmemSemaphore,
        .memHandle = memHandle,
        .offset = 0U,
        .len = res->sizeSemaExternal,
    };

    void* primitiveInfo[] = {&semaPrimitiveInfo};
    internalKeyValue.attrKey =
        LwSciSyncInternalAttrKey_SignalerExternalPrimitiveInfo;
    internalKeyValue.value = (const void*) &primitiveInfo[0];
    internalKeyValue.len = sizeof(primitiveInfo);
    err = LwSciSyncAttrListSetInternalAttrs(list, &internalKeyValue, 1U);
    if (err != LwSciError_Success) {
        goto fail;
    }

fail:
    return err;
}
#endif
