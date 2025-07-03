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

#include <unistd.h>
#include <string.h>
#include <stdio.h>

#if defined(__x86_64__)
#include <lwstatus.h>
#else
#ifdef LW_TEGRA_MIRROR_INCLUDES
#include <mobile_common.h>
#else
#include <lwerror.h>
#include <lwrm_channel.h>
#endif //LW_TEGRA_MIRROR_INCLUDES
#endif //(__x86_64__)

#include <umd.h>

#include "lwscisync_test_common.h"
#include <lwscicommon_libc.h>
#include <cla26f.h>
#include <cla0c0.h>
#include <clb06f.h>
#include <lwrm_gpu.h>
#include <lwrm_host1x_safe.h>

#define METHOD_INCR(subch, cnt, addr) \
    DRF_NUM(A26F, _DMA, _INCR_ADDRESS, (addr) >> 2) | \
    DRF_NUM(A26F, _DMA, _INCR_SUBCHANNEL, subch) | \
    DRF_NUM(A26F, _DMA, _INCR_COUNT, cnt) | \
    DRF_DEF(A26F, _DMA, _INCR_OPCODE, _VALUE)

#define HWCONST(d, r, f, c)  DRF_DEF( d, _ ## r, _ ## f, _ ## c)
#define HWVALUE(d, r, f, v)  DRF_NUM( d, _ ## r, _ ## f, (uint32_t)(v) )

#define TEST_PB_SIZE_WORDS 4096

#define TEST_MAX_CHANNEL 1

typedef struct
{
    LwRmGpuChannel* gpuChannel;
    LwRmGpuChannelInfo gpuChannelInfo;
    LwRmGpuAddressSpaceInfo gpuAsInfo;
    LwRmGpuAddressSpaceAllocation* asAlloc;
    LwRmGpuAddressSpaceAllocationInfo asAllocInfo;
    LwRmMemHandle pbMem;
    LwRmGpuMapping* pbMapping;
    size_t pbLwr;
    LwRmGpuMappingInfo pbMappingInfo;
    uint32_t* pbPtr; // CPU cached: must use cache maintenance
    size_t memAttrSize;
    uint32_t fenceId;
    uint32_t fenceVal;
    uint64_t fenceGpuva;
} TestChannel;

struct TestResourcesRec
{
    LwRmHost1xHandle host1x;
    LwRmHost1xWaiterHandle waiterHandle;
    LwRmGpuLib* gpuLib;
    LwRmGpuDevice* gpuDevice;
    const LwRmGpuDeviceInfo* gpuDeviceInfo;
    LwRmGpuAddressSpace* gpuAs;
    TestChannel channels[TEST_MAX_CHANNEL];
    LwRmGpuTaskSchedulingGroup* tsg;
};

static inline void channelPbAppend(TestChannel* cmdBuf, uint32_t word)
{
    if (cmdBuf->pbLwr >= TEST_PB_SIZE_WORDS) {
        printf("%s %d: writing beyond the buffer\n",
               __func__, __LINE__);
        return;
    }
    cmdBuf->pbPtr[cmdBuf->pbLwr++] = word;
}

static inline void cacheSyncChannelPbForDevice(TestResources res,
        size_t channel)
{
    LwRmMemCacheSyncForDevice(res->channels[channel].pbMem,
            res->channels[channel].pbPtr, TEST_PB_SIZE_WORDS *
            sizeof(uint32_t));
}

static void fillCmdBufSemaphoreRelease(
    TestChannel* cmd,
    uint64_t semaphoreVa,
    uint32_t payload,
    bool awakenEnable)
{
    channelPbAppend(cmd, 0);
    channelPbAppend(cmd, METHOD_INCR(0, 4, LWA0C0_SET_REPORT_SEMAPHORE_A));
    channelPbAppend(cmd, semaphoreVa >> 32);
    channelPbAppend(cmd, semaphoreVa);
    channelPbAppend(cmd, HWVALUE(A0C0, SET_REPORT_SEMAPHORE_C, PAYLOAD, payload));
    channelPbAppend(cmd, HWCONST(A0C0, SET_REPORT_SEMAPHORE_D, OPERATION, RELEASE) |
                         HWVALUE(A0C0, SET_REPORT_SEMAPHORE_D, AWAKEN_ENABLE, awakenEnable) |
                         HWCONST(A0C0, SET_REPORT_SEMAPHORE_D, STRUCTURE_SIZE, ONE_WORD));
}

static void fillCmdBufSemaphoreWait(
    TestChannel *cmd,
    uint64_t semaphoreVa,
    uint32_t payload)
{
    channelPbAppend(cmd, METHOD_INCR(6, 4, LWB06F_SEMAPHOREA));
    channelPbAppend(cmd, semaphoreVa >> 32);
    channelPbAppend(cmd, semaphoreVa);
    channelPbAppend(cmd, payload);
    channelPbAppend(cmd, DRF_DEF(B06F, _SEMAPHORED, _OPERATION, _ACQ_GEQ) |
                         DRF_DEF(B06F, _SEMAPHORED, _ACQUIRE_SWITCH, _ENABLED));
}

static LwSciError submitNoopSignaler(
    TestResources res,
    int chId,
    uint32_t* id,
    uint32_t* value)
{
    LwError err;
    TestChannel* testChannel = &res->channels[chId];
    const size_t pbStart = testChannel->pbLwr;

    /* Submit job to increment syncpoint via syncpoint-semaphore shim. */
    /* The shim will just increment the syncpoint by 1 on each write access */
    /* irrespective of payload */
    fillCmdBufSemaphoreRelease(testChannel, testChannel->fenceGpuva,
            ++testChannel->fenceVal, LW_FALSE);

    cacheSyncChannelPbForDevice(res, chId);

    LWRM_GPU_DEFINE_GPFIFOENTRY(pbEntry);
    pbEntry.gpuVa = testChannel->pbMappingInfo.gpuVa +
            pbStart * sizeof(uint32_t);
    pbEntry.numWords = testChannel->pbLwr - pbStart;

    /* deterministic channels don't do automatic flushing, hence before
     * kickoff: flush mappings */
    err = LwRmGpuAddressSpaceFlushDeferredMappings(res->gpuAs);
    if (err != LwSuccess) {
        return LwSciError_IlwalidState;
    }

    err = LwRmGpuChannelKickoffPb(testChannel->gpuChannel,
                LWRM_GPU_GENERIC_GPFIFO_FORMAT_CLASS,
                &pbEntry, 1, NULL, NULL);
    if (err != LwSuccess) {
        return LwSciError_IlwalidState;
    }

    /* Wait for submitted job to complete */
    err = LwRmHost1xSyncpointWait(res->waiterHandle, testChannel->fenceId,
            testChannel->fenceVal, LWRMHOST1X_MAX_WAIT, NULL);
    if (err != LwSuccess) {
        return LwSciError_IlwalidState;
    }

    /* Reset push buffer index */
    testChannel->pbLwr = pbStart;

    *id = testChannel->fenceId;
    *value = testChannel->fenceVal;

    return LwSciError_Success;
}

static LwSciError submitNoopWaiter(
    TestResources res,
    int chId,
    uint32_t id,
    uint32_t value)
{
    LwError err;
    TestChannel* testChannel = &res->channels[chId];
    const size_t pbStart = testChannel->pbLwr;
    uint64_t syncpointGpuVa;

    /* When waiting for a syncpoint ID/value via the shim, use a GPU */
    /* semaphore wait method */
    err = LwRmGpuAddressSpaceGetSyncpointShimRoAddress(res->gpuAs, id,
            &syncpointGpuVa);
    if (err != LwSuccess) {
        return LwSciError_IlwalidState;
    }
    fillCmdBufSemaphoreWait(testChannel, syncpointGpuVa, value);

    /* Submit job to increment this channel's syncpoint */
    fillCmdBufSemaphoreRelease(testChannel, testChannel->fenceGpuva,
            ++testChannel->fenceVal, LW_FALSE);

    cacheSyncChannelPbForDevice(res, chId);

    LWRM_GPU_DEFINE_GPFIFOENTRY(pbEntry);
    pbEntry.gpuVa = testChannel->pbMappingInfo.gpuVa +
            pbStart * sizeof(uint32_t);
    pbEntry.numWords = testChannel->pbLwr - pbStart;

    /* deterministic channels don't do automatic flushing, hence before
     * kickoff: flush mappings */
    err = LwRmGpuAddressSpaceFlushDeferredMappings(res->gpuAs);
    if (err != LwSuccess) {
        return LwSciError_IlwalidState;
    }

    err = LwRmGpuChannelKickoffPb(testChannel->gpuChannel,
                LWRM_GPU_GENERIC_GPFIFO_FORMAT_CLASS,
                &pbEntry, 1, NULL, NULL);
    if (err != LwSuccess) {
        return LwSciError_IlwalidState;
    }

    /* Wait for submitted job to complete */
    err = LwRmHost1xSyncpointWait(res->waiterHandle, testChannel->fenceId,
            testChannel->fenceVal, LWRMHOST1X_MAX_WAIT, NULL);
    if (err != LwSuccess) {
        return LwSciError_IlwalidState;
    }

    /* Reset push buffer index */
    testChannel->pbLwr = pbStart;

    return LwSciError_Success;
}

LwSciError LwRmGpu_TestSetup(TestResources* resources)
{
    size_t i;
    LwError err;
    int gpuDeviceIndex = LWRM_GPU_DEVICE_INDEX_DEFAULT;
    LwRmHost1xOpenAttrs host1xOpenAttrs;
    LWRM_GPU_DEFINE_TSG_ATTR(tsgAttr);
    LWRM_GPU_DEFINE_DEVICE_OPEN_ATTR(deviceAttr);
    LWRM_GPU_DEFINE_AS_ALLOC_ATTR(lwRmGpuAsAllocAttr);

    TestResources res = (TestResources)
            LwSciCommonCalloc(1, sizeof(struct TestResourcesRec));
    if (res == NULL) {
        printf("Failed to allocate memory\n");
        return LwSciError_InsufficientMemory;
    }

    host1xOpenAttrs = LwRmHost1xGetDefaultOpenAttrs();
    err = LwRmHost1xOpen(&res->host1x, host1xOpenAttrs);
    if (err != LwSuccess) {
        goto fail;
    }

    err = LwRmHost1xWaiterAllocate(&res->waiterHandle, res->host1x);
    if (err != LwSuccess) {
        goto fail;
    }

    res->gpuLib = LwRmGpuLibOpen(NULL);
    if (!res->gpuLib)
        printf("Failed to open the GPU lib: %d.\n",
                LwError_InsufficientMemory);

    err = LwRmGpuDeviceOpen(res->gpuLib, gpuDeviceIndex, &deviceAttr,
            &res->gpuDevice);
    if (err)
    {
        printf("Failed to open the GPU device.\n");
    }
    if (err != LwSuccess) {
        goto fail;
    }

    res->gpuDeviceInfo = LwRmGpuDeviceGetInfo(res->gpuDevice);

    err = LwRmGpuAddressSpaceCreate(res->gpuDevice, NULL, &res->gpuAs);
    if (err != LwSuccess) {
        goto fail;
    }

    err = LwRmGpuTaskSchedulingGroupCreate(res->gpuDevice, &tsgAttr,
            &res->tsg);
    if (err != LwSuccess) {
        goto fail;
    }

    for (i = 0; i < TEST_MAX_CHANNEL; ++i)
    {
        TestChannel* testChannel = &res->channels[i];
        LWRM_GPU_DEFINE_CHANNEL_ATTR(channelAttr);
        channelAttr.hAddressSpace = res->gpuAs;
        channelAttr.syncType = LwRmGpuSyncType_Syncpoint;
        channelAttr.numGpFifoEntries = 128;
        channelAttr.deterministic = true;
        channelAttr.userModeSubmit = true;
        channelAttr.suppressWfi = true;
        channelAttr.disableWatchdog = true;
        channelAttr.hTSG = res->tsg;
        channelAttr.channelObjectClass = res->gpuDeviceInfo->gpuComputeClass;
        err = LwRmGpuChannelCreate(res->gpuDevice, &channelAttr,
                &testChannel->gpuChannel);
        if (err != LwSuccess) {
            goto fail;
        }
        LwRmGpuChannelGetInfo(testChannel->gpuChannel,
                &testChannel->gpuChannelInfo);
        LwRmGpuAddressSpaceGetInfo(testChannel->gpuChannelInfo.hAddressSpace,
                &testChannel->gpuAsInfo);

        /* read the channel syncpoint info */
        testChannel->fenceId = testChannel->gpuChannelInfo.syncpointID;
        testChannel->fenceVal = testChannel->gpuChannelInfo.syncpointMax;
        testChannel->fenceGpuva = testChannel->gpuChannelInfo.syncpointGpuva;

        // create and map pushbuffer
        LWRM_DEFINE_MEM_HANDLE_ATTR(memAttr);
        LWRM_MEM_HANDLE_SET_GPU_ACCESS(memAttr, LwRmMemGpuAccess_GPU);
        LWRM_MEM_HANDLE_SET_ATTR(memAttr, testChannel->gpuAsInfo.smallPageSize,
                LwOsMemAttribute_WriteBack, TEST_PB_SIZE_WORDS * sizeof(uint32_t),
                LwRmMemTags_Tests);

        err = LwRmMemHandleAllocAttr(NULL, &memAttr, &testChannel->pbMem);
        if (err != LwSuccess) {
            goto fail;
        }

        LWRM_GPU_DEFINE_MAPPING_ATTR(pbMapAttr);
        pbMapAttr.pageSize = testChannel->gpuAsInfo.smallPageSize;

        lwRmGpuAsAllocAttr.fixed = LW_TRUE;
        lwRmGpuAsAllocAttr.gpuVa = testChannel->gpuAsInfo.fixedAsAllocHeaps->gpuVa;

        err = LwRmGpuAddressSpaceAllocationCreate(testChannel->gpuChannelInfo.hAddressSpace,
            memAttr.Size / pbMapAttr.pageSize + 1,
            pbMapAttr.pageSize, &lwRmGpuAsAllocAttr,
            &testChannel->asAlloc, &testChannel->asAllocInfo);
        if (err != LwSuccess) {
            goto fail;
        }
        err = LwRmGpuMappingCreateFixed(testChannel->asAlloc,
                testChannel->pbMem,
                testChannel->asAllocInfo.gpuVa,
                memAttr.Size,
                &pbMapAttr, &testChannel->pbMapping, &testChannel->pbMappingInfo);
        if (err != LwSuccess) {
            goto fail;
        }
        err = LwRmMemMap(testChannel->pbMem, 0, memAttr.Size,
                LWOS_MEM_READ_WRITE, (void**) &testChannel->pbPtr);
        if (err != LwSuccess) {
            goto fail;
        }
        testChannel->memAttrSize = memAttr.Size;
    }

    *resources = res;

    return LwSciError_Success;
fail:
    return LwSciError_IlwalidState;
}

void LwRmGpu_TestTeardown(TestResources res)
{
    size_t i;

    for (i = 0U; i < TEST_MAX_CHANNEL; ++i) {
        TestChannel* testChannel = &res->channels[i];

        LwRmMemUnmap(testChannel->pbMem, testChannel->pbPtr, testChannel->memAttrSize);
        LwRmGpuMappingClose(testChannel->pbMapping);
        LwRmMemHandleFree(testChannel->pbMem);
        LwRmGpuAddressSpaceAllocationClose(testChannel->asAlloc);
        LwRmGpuChannelClose(testChannel->gpuChannel);
    }

    LwRmGpuTaskSchedulingGroupClose(res->tsg);
    LwRmGpuAddressSpaceClose(res->gpuAs);
    LwRmGpuDeviceClose(res->gpuDevice);
    LwRmGpuLibClose(res->gpuLib);
    LwRmHost1xWaiterFree(res->waiterHandle);
    LwRmHost1xClose(res->host1x);
    LwSciCommonFree(res);
}

LwSciError umdGetPostLwSciSyncFence(
    LwSciSyncObj syncObj,
    TestResources resource,
    LwSciSyncFence* syncFence)
{
    LwSciError err;
    uint32_t id;
    uint32_t value;

    /* Submit no-op to GPU */
    err = submitNoopSignaler(resource, 0, &id, &value);
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
    /* Submit no-op to GPU with pre-fence */
    err = submitNoopWaiter(resource, 0, id, value);
    if (err != LwSciError_Success) {
        return err;
    }

    return LwSciError_Success;
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
    err = submitNoopSignaler(resource, 0, &id, &value);
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
/* TODO Add semaphore primitive info while adding semaphore tests on CheetAh */
LwSciError umdAddExternalPrimitiveInfo(
    LwSciSyncAttrList list,
    TestResources res)
{
    LwSciError err = LwSciError_Success;
    LwSciSyncInternalAttrKeyValuePair internalKeyValue;
    uint64_t dummyId = 2048U;
    uint64_t ids[] = {res->channels[0].fenceId, dummyId};
    LwSciSyncSimplePrimitiveInfo syncpntPrimitiveInfo;
    size_t i = 0U;
    const uintptr_t* primitiveInfoOffset;
    LwSciSyncInternalAttrKey key;
    const void* value;
    size_t len;

    syncpntPrimitiveInfo = {
        .primitiveType = LwSciSyncInternalAttrValPrimitiveType_Syncpoint,
        .ids = &ids[0],
        .numIds = sizeof(ids)/sizeof(ids[0]),
    };

    void* primitiveInfo[] = {&syncpntPrimitiveInfo};
    size_t primitiveInfoSize[] = {sizeof(syncpntPrimitiveInfo)};
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

/* TODO Implement this function while adding semaphore tests on CheetAh */
LwSciError LwRmGpu_TestMapSemaphore(
    TestResources res,
    LwSciSyncObj syncObj)
{
    return LwSciError_Success;
}
