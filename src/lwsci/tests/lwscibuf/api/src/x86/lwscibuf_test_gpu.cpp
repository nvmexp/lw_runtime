/*
 * Copyright (c) 2022, LWPU CORPORATION. All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#include "lwscibuf_test_gpu_platform_helper.h"

#include <dlfcn.h>
#include <unistd.h>
#include "lwscicommon_errcolwersion.h"
#include "lwstatus.h"
#include "lwRmApi.h"
#include "ctrl/ctrl0000.h"
#include "cl90f1.h" // FERMI_VASPACE_A
#include "cla06f.h" // KEPLER_CHANNEL_GPFIFO_A
#include "clb06f.h" // MAXWELL_CHANNEL_GPFIFO_A
#include "ctrl/ctrl2080/ctrl2080mc.h"
#include "class/cl0080.h"
#include "ctrl/ctrl0080/ctrl0080fifo.h"
#include "ctrl/ctrl0080.h"
#include "ctrl/ctrl906f.h"
#include "ctrl/ctrla06f.h"
#include "ctrl/ctrlc36f.h"
#include "cla0b5.h" // KEPLER_DMA_COPY_A
#include "cla0b5sw.h" // KEPLER_DMA_COPY_A
#include "lwmisc.h"
#include "cla06fsubch.h"
#include "hwref/fermi/gf100/dev_ram.h"
#include "cl506f.h"
#include "cl906f.h"

#include "class/cl0080.h"
#include "class/cl2080.h"
#include "class/cla06f.h" // KEPLER_CHANNEL_GPFIFO_A
#include "class/cla16f.h" // KEPLER_CHANNEL_GPFIFO_B
#include "class/cla26f.h" // KEPLER_CHANNEL_GPFIFO_C
#include "class/clb06f.h" // MAXWELL_CHANNEL_GPFIFO_A
#include "class/clc06f.h" // PASCAL_CHANNEL_GPFIFO_A
#include "class/clc36f.h" // VOLTA_CHANNEL_GPFIFO_A
#include "class/clc46f.h" // TURING_CHANNEL_GPFIFO_A
#include "class/clc56f.h" // AMPERE_CHANNEL_GPFIFO_A
#include "class/clc637.h" // LWC637_ALLOCATION_PARAMETERS

#include "cla0b5.h" // KEPLER_DMA_COPY_A
#include "clb0b5.h" // MAXWELL_DMA_COPY_A
#include "clc0b5.h" // PASCAL_DMA_COPY_A
#include "clc1b5.h" // PASCAL_DMA_COPY_B
#include "clc3b5.h" // VOLTA_DMA_COPY_A
#include "clc5b5.h" // TURING_DMA_COPY_A
#include "clc6b5.h" // AMPERE_DMA_COPY_A
#include "clc7b5.h" // AMPERE_DMA_COPY_B

#include "clc361.h"
#include "lwRmShim/lwRmShim.h"

#define DEVICE_HANDLE 0xaa000000

#define PUSHBUFFER_SIZE 0x1000

static LW_STATUS deviceAlloc(
    GpuTestResourceHandle res)
{
    LW_STATUS status = LW_OK;
    res->hDevice = 0U;
    LW0080_ALLOC_PARAMETERS params;

    memset(&params, 0U, sizeof(LW0080_ALLOC_PARAMETERS));
    res->hDevice = DEVICE_HANDLE + res->m_gpuId;
    params.deviceId = res->deviceInfo[res->m_gpuId].deviceInstance;
    params.hClientShare = res->hClient;
    params.vaMode = LW_DEVICE_ALLOCATION_VAMODE_OPTIONAL_MULTIPLE_VASPACES;

    status = LwRmAlloc(res->hClient, res->hClient, res->hDevice,
                LW01_DEVICE_0, &params);
    if (status != LW_OK) {
        printf("%s: Allocate device failed err - %u : %s\n",
            __FUNCTION__, status, LwStatusToString(status));
    }

    return status;
}

#define SUBDEVICE_HANDLE 0xab000000

static LW_STATUS subdeviceAlloc(
    GpuTestResourceHandle res)
{
    LW_STATUS status = LW_OK;
    res->hSubdevice = SUBDEVICE_HANDLE;

    LW2080_ALLOC_PARAMETERS params;
    memset(&params, 0, sizeof(params));
    params.subDeviceId = res->deviceInfo[res->m_gpuId].subdeviceInstance;

    status = LwRmAlloc(res->hClient, res->hDevice, res->hSubdevice,
                LW20_SUBDEVICE_0, &params);
    if (status != LW_OK) {
        printf("%s: Allocate subdevice failed err - %u : %s\n", __FUNCTION__,
            status, LwStatusToString(status));
        return status;
    }

    return status;
}

static LW_STATUS setupMigHandles(
    GpuTestResourceHandle res)
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

#define VASPACE_HANDLE 0xbb000000

static LW_STATUS allocVASpace(
    GpuTestResourceHandle res)
{
    LW_STATUS status = LW_OK;
    res->hVASpace = VASPACE_HANDLE;

    LW_VASPACE_ALLOCATION_PARAMETERS params;
    memset(&params, 0, sizeof(LW_VASPACE_ALLOCATION_PARAMETERS));
    params.index = LW_VASPACE_ALLOCATION_INDEX_GPU_NEW;
    // FERMI is only VASPACE macro available for x86
    status = LwRmAlloc(res->hClient, res->hDevice, res->hVASpace,
                FERMI_VASPACE_A, &params);
    if (status != LW_OK) {
        printf("%s: Allocate VASpace failed err - %u : %s\n", __FUNCTION__,
            status, LwStatusToString(status));
        return status;
    }

    return status;
}

// this is for sysmem allocation
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
        printf("Failed to allocate system memory error = 0x%x - %s\n", status,
            LwStatusToString(status));
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
    params.data.AllocSize.attr = 0U;
    params.data.AllocSize.attr2 = 0U;
    params.data.AllocSize.size = size;

    status = LwRmVidHeapControl((void*)(&params));
    if (status != LW_OK) {
        printf("Failed to allocate vid memory error = 0x%x - %s\n", status,
            LwStatusToString(status));
        return status;
    }

    *hMemory = params.data.AllocSize.hMemory;

    return status;
}

#define CHANNEL_HANDLE 0xcc000000

static LW_STATUS setSupportedClass(
    GpuTestResourceHandle res,
    uint32_t* hClass)
{
    LW_STATUS status = LW_OK;
    uint32_t engineType = LW2080_ENGINE_TYPE_HOST;
    uint32_t* pSupportedClassList = NULL;
    uint32_t i = 0U, j = 0U;
    uint32_t supportedList[] =
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
    uint32_t size = sizeof(supportedList)/sizeof(supportedList[0]);

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
        printf("get engine classlist error code = 0x%x - %s\n", status,
            LwStatusToString(status));
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
            printf("Unrecognized channel class 0x%x\n", hClass);
            return 0U;
    }
}

static LW_STATUS createChannel(
    GpuTestResourceHandle res)
{
    LW_STATUS status = LW_OK;

    uint32_t flags = 0;
    uint64_t m_size;
    uint32_t NUM_GPFIFOS = 128;
    uint32_t GPFIFO_SIZE = NUM_GPFIFOS * LWA06F_GP_ENTRY__SIZE;

    //Get Supported class list for selected GPU
    status = setSupportedClass(res, &res->gpFifoClass);
    if (status != LW_OK) {
        printf("get engine classlist error = 0x%x - %s\n", status,
            LwStatusToString(status));
        goto ret;
    }

    /* GPFIFO */
    m_size = GPFIFO_SIZE;
    status = physMemAlloc(res->hClient, res->hDevice,
        &res->gpfifoPhyMem_hMemory, &m_size);
    if (status != LW_OK) {
        printf("GPFIFO physMemoc failed error = 0x%x - %s\n", status,
            LwStatusToString(status));
        goto ret;
    }
    res->gpfifoPhyMem_length = m_size;

    status = gpuVirtMemAlloc(res->hClient, res->hDevice, res->hVASpace,
        res->gpfifoPhyMem_length, &res->gpfifoVirtMem_hMemory);
    if (status != LW_OK) {
        printf("GPFIFO gpuVirtMemAlloc error = 0x%x - %s\n", status,
            LwStatusToString(status));
        goto ret;
    }

    status = LwRmMapMemory(res->hClient, res->hDevice,
                res->gpfifoPhyMem_hMemory, 0U, res->gpfifoPhyMem_length,
                &res->gpfifo_phyAddress, 0U);
    if (status != LW_OK) {
        printf("Phy memory mem map error = 0x%x - %s\n", status,
            LwStatusToString(status));
        goto ret;
    }

    flags = FLD_SET_DRF(OS46, _FLAGS, _ACCESS, _READ_WRITE, flags);
    flags = FLD_SET_DRF(OS46, _FLAGS, _CACHE_SNOOP, _ENABLE, flags);

    status = LwRmMapMemoryDma(res->hClient, res->hDevice,
            res->gpfifoVirtMem_hMemory, res->gpfifoPhyMem_hMemory,
            0U, res->gpfifoPhyMem_length, flags,
            static_cast<LwU64*>(&res->gpfifo_virtAddress));
    if (status != LW_OK) {
        printf("LwRmMapMemoryDma failed for GPFIFO error = 0x%x - %s\n", status,
            LwStatusToString(status));
        goto ret;
    }

    res->hChannel = CHANNEL_HANDLE;

    LW_CHANNELGPFIFO_ALLOCATION_PARAMETERS gpfifoParams;
    memset(&gpfifoParams, 0, sizeof(LW_CHANNELGPFIFO_ALLOCATION_PARAMETERS));

    // For Volta+, USERD is no longer allocated by the RM, hence has to be
    // managed in test app. When allocating the channel, USERD memory handle
    // will be passed through LW_CHANNELGPFIFO_ALLOCATION_PARAMETERS.
    if (res->gpFifoClass >= AMPERE_CHANNEL_GPFIFO_A) {
        m_size = GetUserDSize(res->gpFifoClass);
        status = physMemAlloc(res->hClient, res->hDevice,
            &res->hPhyMemoryUserd, &m_size);
        if (status != LW_OK) {
            printf("USERD alloc failed code = 0x%x - %s\n", status,
                LwStatusToString(status));
            goto ret;
        }
        gpfifoParams.hUserdMemory[0] = res->hPhyMemoryUserd;
        gpfifoParams.userdOffset[0]  = 0;
    }

    gpfifoParams.hObjectBuffer = res->gpfifoVirtMem_hMemory;
    gpfifoParams.gpFifoOffset  = res->gpfifo_virtAddress;
    gpfifoParams.gpFifoEntries = NUM_GPFIFOS;
    gpfifoParams.engineType    = LW2080_ENGINE_TYPE_COPY0;
    gpfifoParams.flags         = 0U;
    gpfifoParams.hVASpace      = res->hVASpace;

    status = LwRmAlloc(res->hClient, res->hDevice, res->hChannel,
            res->gpFifoClass, &gpfifoParams);
    if (status != LW_OK) {
        printf("Channel alloc failed code = 0x%x - %s\n", status,
            LwStatusToString(status));
        goto ret;
    }

    if (res->gpFifoClass < AMPERE_CHANNEL_GPFIFO_A) {
        flags = 0;
        flags = FLD_SET_DRF(OS33, _FLAGS, _MAPPING, _DEFAULT, flags);
        flags = FLD_SET_DRF(OS33, _FLAGS, _ACCESS, _READ_WRITE, flags);
        status = LwRmMapMemory(res->hClient, res->hDevice,
                res->hChannel, 0, GetUserDSize(res->gpFifoClass),
                &res->m_pChannelCtrl, flags);
        if (status != LW_OK) {
            printf("Failed to map channel allocation error = 0x%x - %s\n", status,
                LwStatusToString(status));
            goto ret;
        }
    } else {
        status = LwRmMapMemory(res->hClient, res->hDevice,
                res->hPhyMemoryUserd, 0, GetUserDSize(res->gpFifoClass),
                &res->phyAddrUserd, 0);
        if (status != LW_OK) {
            printf("Failed to map userd, error = 0x%x - %s\n", status,
                LwStatusToString(status));
            goto ret;
        }
        res->m_pChannelCtrl = (char*)res->phyAddrUserd + gpfifoParams.userdOffset[0];
    }

ret:
    return status;
}

static LW_STATUS scheduleChannel(
    GpuTestResourceHandle res)
{
    LW_STATUS status = LW_OK;

    LWA06F_CTRL_GPFIFO_SCHEDULE_PARAMS gpFifoSchedulParams;
    memset(&gpFifoSchedulParams, 0, sizeof(gpFifoSchedulParams));
    gpFifoSchedulParams.bEnable = true;

    status = LwRmControl(res->hClient, res->hChannel,
                LWA06F_CTRL_CMD_GPFIFO_SCHEDULE, &gpFifoSchedulParams,
                sizeof(gpFifoSchedulParams));
    if (status != LW_OK) {
        printf("GPFIFO schedule error = 0x%x - %s\n", status,
            LwStatusToString(status));
        goto ret;
    }

ret:
    return status;
}

static LW_STATUS createPusher(
    GpuTestResourceHandle res,
    uint32_t pushbufferSize)
{
    LW_STATUS status = LW_OK;
    uint64_t m_size = pushbufferSize;
    uint32_t flags = 0;

    status = physMemAlloc(res->hClient, res->hDevice,
                &res->pusherPhyMem_hMemory, &m_size);
    if (status != LW_OK) {
        printf("sys mem for pusher failed error = 0x%x - %s\n", status,
            LwStatusToString(status));
        goto ret;
    }
    res->pusherPhyMem_length = m_size;

    status = gpuVirtMemAlloc(res->hClient, res->hDevice, res->hVASpace, m_size,
                &res->pusherVirtMem_hMemory);
    if (status != LW_OK) {
        printf("vid mem for pusher failed error = 0x%x - %s\n", status,
            LwStatusToString(status));
        goto ret;
    }

    status = LwRmMapMemory(res->hClient, res->hDevice,
            res->pusherPhyMem_hMemory, 0U, res->pusherPhyMem_length,
            &res->pusher_phyAddress, 0U);
    if (status != LW_OK) {
        printf("sys mem map for pusher failed error = 0x%x - %s\n", status,
            LwStatusToString(status));
        goto ret;
    }

    flags = FLD_SET_DRF(OS46, _FLAGS, _ACCESS, _READ_WRITE, flags);
    flags = FLD_SET_DRF(OS46, _FLAGS, _CACHE_SNOOP, _ENABLE, flags);

    status = LwRmMapMemoryDma(res->hClient, res->hDevice,
            res->pusherVirtMem_hMemory, res->pusherPhyMem_hMemory,
            0U, res->pusherPhyMem_length, flags,
            static_cast<LwU64*>(&res->pusher_virtAddress));
    if (status != LW_OK) {
        printf("pusher mem map dma failed error = 0x%x - %s\n", status,
            LwStatusToString(status));
        goto ret;
    }

ret:
    return status;
}

static LW_STATUS ringDoorBell(GpuTestResourceHandle res)
{
    LW_STATUS status = LW_OK;
    LwU32 token;
    LWC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN_PARAMS params;

    switch(res->gpFifoClass) {
        case PASCAL_CHANNEL_GPFIFO_A:
        case MAXWELL_CHANNEL_GPFIFO_A:
        case KEPLER_CHANNEL_GPFIFO_C:
        case KEPLER_CHANNEL_GPFIFO_B:
        case KEPLER_CHANNEL_GPFIFO_A:
            goto ret;
            break;
        default:
            break;
    }

    memset(&params, 0, sizeof(params));
    status = LwRmControl(res->hClient, res->hChannel,
                LWC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN, &params,
                sizeof(params));
    if (status != LW_OK) {
        printf("failed to get work token error = 0x%x - %s\n", status,
            LwStatusToString(status));
        goto ret;
    }

    token = params.workSubmitToken;
    res->hDoorBell_hMemory = 0xaabbccdd;
    status = LwRmAlloc(res->hClient,
                      res->hSubdevice,
                      res->hDoorBell_hMemory,
                      VOLTA_USERMODE_A,
                      NULL);

    if (status != LW_OK) {
        printf("failed to alloc door bell memory error = 0x%x - %s\n", status,
            LwStatusToString(status));
        goto ret;
    }

    status = LwRmMapMemory(res->hClient,
                        res->hSubdevice,
                        res->hDoorBell_hMemory,
                        LWC361_NOTIFY_CHANNEL_PENDING,
                        sizeof(*res->pDoorbell_phyAddress),
                        (void**)&res->pDoorbell_phyAddress,
                        0);

    if (status != LW_OK) {
        printf("failed to map doorbell mem error = 0x%x - %s\n", status,
            LwStatusToString(status));
        goto ret;
    }

    *res->pDoorbell_phyAddress = token;

ret:
    return status;
}

// create new vidmem for testing
static LW_STATUS createTestMem(
    GpuTestResourceHandle res,
    uint32_t size,
    uint32_t* phy_hMemory,
    uint32_t* virt_hMemory,
    void** phyAddress,
    LwU64* virtAddress)
{
    LW_STATUS status = LW_OK;
    uint64_t length;
    uint32_t flags = 0;

    // Allocate physical memory
    length = size;
    status = physMemAlloc(res->hClient, res->hDevice, phy_hMemory, &length);
    if (status != LW_OK) {
        printf("test mem phy alloc failed failed error = 0x%x - %s\n", status,
            LwStatusToString(status));
        goto ret;
    }

    // Allocate virtual memory
    status = gpuVirtMemAlloc(res->hClient, res->hDevice, res->hVASpace, size,
                virt_hMemory);
    if (status != LW_OK) {
        printf("test mem vir alloc failed failed error = 0x%x - %s\n", status,
            LwStatusToString(status));
        goto ret;
    }

    // Map memory to CPU
    status = LwRmMapMemory(res->hClient, res->hDevice, *phy_hMemory,
            0, length, phyAddress, 0);
    if (status != LW_OK) {
        printf("test phy mem map failed failed error = 0x%x - %s\n", status,
            LwStatusToString(status));
        goto ret;
    }

    flags = 0U;
    // Map memory to GPU
    flags = FLD_SET_DRF(OS46, _FLAGS, _ACCESS, _READ_WRITE, flags);
    flags = FLD_SET_DRF(OS46, _FLAGS, _CACHE_SNOOP, _ENABLE, flags);

    status = LwRmMapMemoryDma(res->hClient, res->hDevice, *virt_hMemory,
            *phy_hMemory, 0, length, flags, static_cast<LwU64*>(virtAddress));
    if (status != LW_OK) {
        printf("test vir mem map failed failed error = 0x%x - %s\n", status,
            LwStatusToString(status));
        goto ret;
    }

ret:
    return status;
}

// create new vidmem for testing
static LW_STATUS createTestMemFromSciRm(
    GpuTestResourceHandle res,
    LwSciBufRmHandle* scibufRmHandle,
    uint32_t size,
    uint32_t* phy_hMemory,
    uint32_t* virt_hMemory,
    void** phyAddress,
    LwU64* virtAddress)
{
    LW_STATUS status = LW_OK;
    uint32_t flags = 0;

    status = LwRmDupObject2(res->hClient, res->hDevice, phy_hMemory,
                        scibufRmHandle->hClient,
                        scibufRmHandle->hMemory, 0);
    if (status != LW_OK) {
        printf("dup handle failed error = 0x%x - %s\n", status,
            LwStatusToString(status));
        goto ret;
    }

    // Allocate virtual memory
    status = gpuVirtMemAlloc(res->hClient, res->hDevice, res->hVASpace, size,
                virt_hMemory);
    if (status != LW_OK) {
        printf("failed to allocate virtual memory error = 0x%x - %s\n", status,
            LwStatusToString(status));
        goto ret;
    }

    // Map memory to CPU
    status = LwRmMapMemory(res->hClient, res->hDevice, *phy_hMemory,
            0, size, phyAddress, 0);
    if (status != LW_OK) {
        printf("failed map memory error = 0x%x - %s\n", status,
            LwStatusToString(status));
        goto ret;
    }

    // Map memory to GPU
    flags = FLD_SET_DRF(OS46, _FLAGS, _ACCESS, _READ_WRITE, flags);
    flags = FLD_SET_DRF(OS46, _FLAGS, _CACHE_SNOOP, _ENABLE, flags);

    status = LwRmMapMemoryDma(res->hClient, res->hDevice, *virt_hMemory,
            *phy_hMemory, 0, size, flags, static_cast<LwU64*>(virtAddress));
    if (status != LW_OK) {
        printf("failed map memory DMA error = 0x%x - %s\n", status,
            LwStatusToString(status));
        goto ret;
    }

ret:
    return status;
}

static uint32_t Command(uint32_t subCh, uint32_t method, uint32_t count)
{
    return  (DRF_DEF(_FIFO, _DMA, _INCR_OPCODE, _VALUE) |
             DRF_NUM(_FIFO, _DMA, _INCR_COUNT, count) |
             DRF_NUM(_FIFO, _DMA, _INCR_SUBCHANNEL, subCh) |
             DRF_NUM(_FIFO, _DMA, _INCR_ADDRESS, method>>2));
}

static void kickOff(
    GpuTestResourceHandle res)
{
    // push DMA copy methods
    uint32_t* m_pPbLwrrent;
    uint32_t* m_pPbStart;
    m_pPbLwrrent = static_cast<uint32_t*>(res->pusher_phyAddress);
    m_pPbStart = m_pPbLwrrent;
    uint32_t method;
    uint32_t d0;
    uint32_t d1;
    uint32_t d2;
    uint32_t d3;

    method = LWA06F_SET_OBJECT;
    m_pPbLwrrent[0] = Command(LWA06F_SUBCHANNEL_COPY_ENGINE, method, 1);
    m_pPbLwrrent[1] = TURING_DMA_COPY_A;
    m_pPbLwrrent += 2;

    method = LWA0B5_OFFSET_IN_UPPER;
    d0 = (uint32_t)(res->src_virtAddress >> 32);
    d1 = (uint32_t)(res->src_virtAddress & 0xFFFFFFFF);
    d2 = (uint32_t)(res->dest_virtAddress >> 32);
    d3 = (uint32_t)(res->dest_virtAddress & 0xFFFFFFFF);
    m_pPbLwrrent[0] = Command(LWA06F_SUBCHANNEL_COPY_ENGINE, method, 4);
    m_pPbLwrrent[1] = d0;
    m_pPbLwrrent[2] = d1;
    m_pPbLwrrent[3] = d2;
    m_pPbLwrrent[4] = d3;
    m_pPbLwrrent += 5;

    method = LWA0B5_LINE_LENGTH_IN;
    m_pPbLwrrent[0] = Command(LWA06F_SUBCHANNEL_COPY_ENGINE, method, 2);
    m_pPbLwrrent[1] = res->memSize;
    m_pPbLwrrent[2] = 0x1;
    m_pPbLwrrent += 3;

    method = LWA0B5_SET_SRC_WIDTH;
    m_pPbLwrrent[0] = Command(LWA06F_SUBCHANNEL_COPY_ENGINE, method, 4);
    m_pPbLwrrent[1] = res->memSize;
    m_pPbLwrrent[2] = 0x1;
    m_pPbLwrrent[3] = 0x1;
    m_pPbLwrrent[4] = 0x0;
    m_pPbLwrrent += 5;

    method = LWA0B5_SET_DST_WIDTH;
    m_pPbLwrrent[0] = Command(LWA06F_SUBCHANNEL_COPY_ENGINE, method, 4);
    m_pPbLwrrent[1] = res->memSize;
    m_pPbLwrrent[2] = 0x1;
    m_pPbLwrrent[3] = 0x1;
    m_pPbLwrrent[4] = 0x0;
    m_pPbLwrrent += 5;

    uint32_t launch_dma;
    launch_dma = DRF_DEF(A0B5, _LAUNCH_DMA, _FLUSH_ENABLE, _FALSE) |
                 DRF_DEF(A0B5, _LAUNCH_DMA, _SRC_MEMORY_LAYOUT, _PITCH) |
                 DRF_DEF(A0B5, _LAUNCH_DMA, _DST_MEMORY_LAYOUT, _PITCH) |
                 DRF_DEF(A0B5, _LAUNCH_DMA, _DATA_TRANSFER_TYPE, _PIPELINED);
    method = LWA0B5_LAUNCH_DMA;
    m_pPbLwrrent[0] = Command(LWA06F_SUBCHANNEL_COPY_ENGINE, method, 1);
    m_pPbLwrrent[1] = launch_dma;
    m_pPbLwrrent += 2;

    uint32_t length;
    length = (uint32_t)(m_pPbLwrrent - m_pPbStart);
    uint64_t pbGpuVA;
    pbGpuVA = res->pusher_virtAddress;
    LwU32 gpEntry0;
    LwU32 gpEntry1;
    gpEntry0 = DRF_NUM(506F, _GP_ENTRY0, _GET, LwU64_LO32(pbGpuVA) >> 2);
    gpEntry1 = DRF_NUM(506F, _GP_ENTRY1, _GET_HI, LwU64_HI32(pbGpuVA)) |
               DRF_NUM(506F, _GP_ENTRY1, _LENGTH, length);

    uint32_t* m_pGpFifoBase;
    uint32_t m_gpPut;
    m_pPbLwrrent = static_cast<uint32_t*>(res->pusher_phyAddress);
    m_pGpFifoBase = static_cast<uint32_t*>(res->gpfifo_phyAddress);
    m_gpPut = 0;
    m_pGpFifoBase[m_gpPut*2+0] = gpEntry0;
    m_pGpFifoBase[m_gpPut*2+1] = gpEntry1;
    m_gpPut = (m_gpPut+1);
    Lw906fControl* pChannelCtrl;
    pChannelCtrl = static_cast<Lw906fControl*>(res->m_pChannelCtrl);

    printf("src: %x, dst: %x\n", *(uint32_t*)res->src_phyAddress,
            *(uint32_t*)res->dest_phyAddress);

    // This kickoff's pushbuffer commands
    pChannelCtrl->GPPut = m_gpPut;
}

static void freeAllResources(
    GpuTestResourceHandle res)
{
    LW_STATUS status = LW_OK;
    uint32_t i = 0U;
    LwRmShimError errShim = LWRMSHIM_OK;

    switch(res->gpFifoClass) {
        case PASCAL_CHANNEL_GPFIFO_A:
        case MAXWELL_CHANNEL_GPFIFO_A:
        case KEPLER_CHANNEL_GPFIFO_C:
        case KEPLER_CHANNEL_GPFIFO_B:
        case KEPLER_CHANNEL_GPFIFO_A:
            goto free_common_resources;
            break;
        case TURING_CHANNEL_GPFIFO_A:
        case VOLTA_CHANNEL_GPFIFO_A:
            goto free_volta_resources;
            break;
        default:
            break;
    }

    status = LwRmUnmapMemory(res->hClient, res->hDevice,
                res->hPhyMemoryUserd, res->phyAddrUserd, 0U);
    if (status != LW_OK) {
        printf("%s: free userd phy mem map failed, err - %u : %s\n",
        __FUNCTION__, status, LwStatusToString(status));
        goto ret;
    }

    status = LwRmFree(res->hClient, res->hDevice, res->hPhyMemoryUserd);
    if (status != LW_OK) {
        printf("%s: free userd phy failed, err - %u : %s\n",
        __FUNCTION__, status, LwStatusToString(status));
        goto ret;
    }

free_volta_resources:
    status = LwRmUnmapMemory(res->hClient, res->hSubdevice,
                res->hDoorBell_hMemory, res->pDoorbell_phyAddress, 0U);
    if (status != LW_OK) {
        printf("%s: free doorbell phy mem map failed, err - %u : %s\n",
        __FUNCTION__, status, LwStatusToString(status));
        goto ret;
    }

    status = LwRmFree(res->hClient, res->hSubdevice, res->hDoorBell_hMemory);
    if (status != LW_OK) {
        printf("%s: free doorbell phy failed, err - %u : %s\n",
        __FUNCTION__, status, LwStatusToString(status));
        goto ret;
    }

free_common_resources:
    status = LwRmUnmapMemoryDma(res->hClient, res->hDevice,
                res->destVirtMem_hMemory, res->destPhyMem_hMemory, 0U,
                res->dest_virtAddress);
    if (status != LW_OK) {
        printf("%s: free dup vir mem map failed, err - %u : %s\n",
        __FUNCTION__, status, LwStatusToString(status));
        goto ret;
    }

    status = LwRmUnmapMemory(res->hClient, res->hDevice,
                res->destPhyMem_hMemory, res->dest_phyAddress, 0U);
    if (status != LW_OK) {
        printf("%s: free dup phy mem map failed, err - %u : %s\n",
        __FUNCTION__, status, LwStatusToString(status));
        goto ret;
    }

    status = LwRmFree(res->hClient, res->hDevice, res->destPhyMem_hMemory);
    if (status != LW_OK) {
        printf("%s: free dup phy failed, err - %u : %s\n",
        __FUNCTION__, status, LwStatusToString(status));
        goto ret;
    }

    status = LwRmFree(res->hClient, res->hDevice, res->destVirtMem_hMemory);
    if (status != LW_OK) {
        printf("%s: free dup vir mem failed, err - %u : %s\n",
        __FUNCTION__, status, LwStatusToString(status));
        goto ret;
    }

    status = LwRmUnmapMemoryDma(res->hClient, res->hDevice,
                res->srcVirtMem_hMemory, res->srcPhyMem_hMemory, 0U,
                res->src_virtAddress);
    if (status != LW_OK) {
        printf("%s: free dup vir mem map failed, err - %u : %s\n",
        __FUNCTION__, status, LwStatusToString(status));
        goto ret;
    }

    status = LwRmUnmapMemory(res->hClient, res->hDevice,
                res->srcPhyMem_hMemory, res->src_phyAddress, 0U);
    if (status != LW_OK) {
        printf("%s: free dup phy mem map failed, err - %u : %s\n",
        __FUNCTION__, status, LwStatusToString(status));
        goto ret;
    }

    status = LwRmFree(res->hClient, res->hDevice, res->srcPhyMem_hMemory);
    if (status != LW_OK) {
        printf("%s: free dup phy failed, err - %u : %s\n",
        __FUNCTION__, status, LwStatusToString(status));
        goto ret;
    }

    status = LwRmFree(res->hClient, res->hDevice, res->srcVirtMem_hMemory);
    if (status != LW_OK) {
        printf("%s: free dup vir mem failed, err - %u : %s\n",
        __FUNCTION__, status, LwStatusToString(status));
        goto ret;
    }

    status = LwRmUnmapMemoryDma(res->hClient, res->hDevice,
            res->pusherVirtMem_hMemory, res->pusherPhyMem_hMemory,
            0U, res->pusher_virtAddress);
    if (status != LW_OK) {
        printf("%s: free pusher vir mem map failed, err - %u : %s\n",
        __FUNCTION__, status, LwStatusToString(status));
        goto ret;
    }

    status = LwRmUnmapMemory(res->hClient, res->hDevice,
                res->pusherPhyMem_hMemory, res->pusher_phyAddress, 0U);
    if (status != LW_OK) {
        printf("%s: free pusher phy mem map failed, err - %u : %s\n",
        __FUNCTION__, status, LwStatusToString(status));
        goto ret;
    }

    status = LwRmFree(res->hClient, res->hDevice, res->pusherVirtMem_hMemory);
    if (status != LW_OK) {
        printf("%s: free pusher vir mem failed, err - %u : %s\n",
        __FUNCTION__, status, LwStatusToString(status));
        goto ret;
    }

    status = LwRmFree(res->hClient, res->hDevice, res->pusherPhyMem_hMemory);
    if (status != LW_OK) {
        printf("%s: free pusher phy failed, err - %u : %s\n",
        __FUNCTION__, status, LwStatusToString(status));
        goto ret;
    }

    if (res->gpFifoClass < AMPERE_CHANNEL_GPFIFO_A) {
        status = LwRmUnmapMemory(res->hClient, res->hDevice, res->hChannel,
                    res->m_pChannelCtrl, 0U);
        if (status != LW_OK) {
            printf("%s: free channel phy mem map failed, err - %u : %s\n",
            __FUNCTION__, status, LwStatusToString(status));
            goto ret;
        }
    }

    status = LwRmFree(res->hClient, res->hDevice, res->hChannel);
    if (status != LW_OK) {
        printf("%s: free channel failed, err - %u : %s\n", __FUNCTION__,
        status, LwStatusToString(status));
        goto ret;
    }

    status = LwRmUnmapMemoryDma(res->hClient, res->hDevice,
            res->gpfifoVirtMem_hMemory, res->gpfifoPhyMem_hMemory,
            0U, res->gpfifo_virtAddress);
    if (status != LW_OK) {
        printf("%s: free gpfifo vir mem map failed, err - %u : %s\n",
        __FUNCTION__, status, LwStatusToString(status));
        goto ret;
    }

    status = LwRmUnmapMemory(res->hClient, res->hDevice,
                res->gpfifoPhyMem_hMemory, res->gpfifo_phyAddress, 0U);
    if (status != LW_OK) {
        printf("%s: free gpfifo phy mem map failed, err - %u : %s\n",
        __FUNCTION__, status, LwStatusToString(status));
        goto ret;
    }

    status = LwRmFree(res->hClient, res->hDevice, res->gpfifoVirtMem_hMemory);
    if (status != LW_OK) {
        printf("%s: free gpfifo vir mem failed, err - %u : %s\n", __FUNCTION__,
        status, LwStatusToString(status));
        goto ret;
    }

    status = LwRmFree(res->hClient, res->hDevice, res->gpfifoPhyMem_hMemory);
    if (status != LW_OK) {
        printf("%s: free gpfifo phy mem failed, err - %u : %s\n", __FUNCTION__,
        status, LwStatusToString(status));
        goto ret;
    }

    status = LwRmFree(res->hClient, res->hDevice, res->hVASpace);
    if (status != LW_OK) {
        printf("%s: free VASpace failed, err - %u : %s\n", __FUNCTION__,
        status, LwStatusToString(status));
        goto ret;
    }

    (void)LwRmFree(res->hClient, res->hSubdevice, res->hGpuInstance);

    status = LwRmFree(res->hClient, res->hDevice, res->hSubdevice);
    if (status != LW_OK) {
        printf("%s: free subdevice failed, err - %u : %s\n", __FUNCTION__,
        status, LwStatusToString(status));
        goto ret;
    }

    status = LwRmFree(res->hClient, res->hClient, res->hDevice);
    if (status != LW_OK) {
        printf("%s: free device failed, err - %u : %s\n", __FUNCTION__,
        status, LwStatusToString(status));
        goto ret;
    }

    /* Detach GPU */
    LW0000_CTRL_GPU_DETACH_IDS_PARAMS detachParams;
    memset(&detachParams, 0, sizeof(detachParams));

    for (i = 0;
        i < res->deviceInfo.size(); i++) {
        detachParams.gpuIds[i] = res->deviceInfo[i].deviceId;
    }

    if (i != LW0000_CTRL_GPU_MAX_PROBED_GPUS) {
        detachParams.gpuIds[i] = LW0000_CTRL_GPU_ILWALID_ID;
    }

    status = LwRmControl(res->hClient, res->hClient,
                LW0000_CTRL_CMD_GPU_DETACH_IDS,
                &detachParams, sizeof(detachParams));
    if (status != LW_OK) {
        printf("%s: Detach GPU failed, err - %u : %s\n", __FUNCTION__, status,
            LwStatusToString(status));
        goto ret;
    }

    status = LwRmFree(res->hClient, res->hClient, res->hClient);
    if (status != LW_OK) {
        printf("%s: free client failed, err - %u : %s\n", __FUNCTION__, status,
            LwStatusToString(status));
        goto ret;
    }

   LwRmShimError (*sessionDestroy)(LwRmShimSessionContext*);
    sessionDestroy = (LwRmShimError (*)(LwRmShimSessionContext*))
                     dlsym(res->lib_h, "LwRmShimSessionDestroy");
    if (sessionDestroy == NULL) {
        status = LW_ERR_ILWALID_STATE;
        printf("[ERR] sessionDestroy null\n");
        goto ret;
    }
    errShim = sessionDestroy(&res->session);
    if (errShim != LWRMSHIM_OK) {
        status = LW_ERR_ILWALID_STATE;
        goto ret;
    }

ret:
    return;
}

LwSciError testGpuMapping(
    GpuTestResourceHandle res,
    LwSciBufRmHandle rmHandle,
    uint64_t memSize)
{
    LwSciError sciErr = LwSciError_Success;
    LW_STATUS status = LW_OK;
    uint32_t count;

    res->memSize = memSize;

    // Allocate all available devices
    status = deviceAlloc(res);
    if (status != LW_OK) {
        printf("deviceAlloc failed error code = 0x%x - %s\n", status,
            LwStatusToString(status));
        goto ret;
    }

    // Create sub-device based on device
    status = subdeviceAlloc(res);
    if (status != LW_OK) {
        printf("subdeviceAlloc failed error code = 0x%x - %s\n", status,
            LwStatusToString(status));
        goto ret;
    }

    // Subscibe to MIG GI if MIG mode enabled
    status = setupMigHandles(res);
    if (status != LW_OK) {
        printf("setupMigHandles failed error code = 0x%x - %s\n", status,
            LwStatusToString(status));
        goto ret;
    }

    // Create VA Space
    status = allocVASpace(res);
    if (status != LW_OK) {
        printf("allocVASpace failed error code = 0x%x - %s\n", status,
            LwStatusToString(status));
        goto ret;
    }

    // Create channel
    status = createChannel(res);
    if (status != LW_OK) {
        printf("createChannel failed error code = 0x%x - %s\n", status,
            LwStatusToString(status));
        goto ret;
    }

    // Create pushbuffer
    status = createPusher(res, PUSHBUFFER_SIZE);
    if (status != LW_OK) {
        printf("createPusher failed error code = 0x%x - %s\n", status,
            LwStatusToString(status));
        goto ret;
    }

    status = scheduleChannel(res);
    if (status != LW_OK) {
        printf("scheduleChannel failed error code = 0x%x - %s\n", status,
            LwStatusToString(status));
        goto ret;
    }

    // Allocate source test memory. (not actually allocating, we will dup the
    //  phys mem handle returned from scibuf obj)
    status = createTestMemFromSciRm(res, &rmHandle, memSize,
                &res->srcPhyMem_hMemory, &res->srcVirtMem_hMemory,
                &res->src_phyAddress, &res->src_virtAddress);
    if (status != LW_OK) {
        printf("createTestMemFromSciRm failed error code = 0x%x - %s\n", status,
            LwStatusToString(status));
        goto ret;
    }

    // Write data to source memory
    *(uint32_t*)res->src_phyAddress = 0xdeadbeef;

    // Allocate destination test memory
    status = createTestMem(res, memSize, &res->destPhyMem_hMemory,
            &res->destVirtMem_hMemory, &res->dest_phyAddress,
            &res->dest_virtAddress);
    if (status != LW_OK) {
        printf("createTestMem failed error code = 0x%x - %s\n", status,
            LwStatusToString(status));
        goto ret;
    }

    kickOff(res);

    status = ringDoorBell(res);
    if (status != LW_OK) {
        printf("ringDoorBell failed err - %u : %s\n", status,
            LwStatusToString(status));
        goto ret;
    }

    // Wait for some time, else can return error
    count = 0;
    while (*(uint32_t*)res->src_phyAddress !=
            *(uint32_t*)res->dest_phyAddress) {
        if (count < 1000) {
            usleep(500);
            count++;
        } else {
            status = LW_ERR_GENERIC;
            printf("DMA failed\n");
            break;
        }
    }

    printf("src: %x, dst: %x\n", *(uint32_t*)res->src_phyAddress,
            *(uint32_t*)res->dest_phyAddress);

    freeAllResources(res);

ret:
    sciErr = LwStatusToLwSciErr(status);
    return sciErr;
}
