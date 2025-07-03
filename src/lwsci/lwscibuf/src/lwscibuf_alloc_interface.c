/*
 * Copyright (c) 2019-2021, LWPU CORPORATION. All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#include "lwscibuf_alloc_interface_priv.h"
#include "lwscicommon_os.h"

/**
 * @brief Array of structures of type LwSciBufAllocIfaceFvt, and the size
 * depends on the number of LwSciBufAllocIfaceType(s) supported. The function
 * pointer values of structure LwSciBufAllocIfaceFvt, is initialized with
 * the functions respectively for the supported LwSciBufAllocIfaceType(s).
 */
LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 11_1), "LwSciBuf-REQ-MISRAC2012-003")
static const LwSciBufAllocIfaceFvt allocIfaceFvt[LwSciBufAllocIfaceType_Max] = {
    [LwSciBufAllocIfaceType_SysMem] = {
        .iFaceOpen              = LwSciBufSysMemOpen,
        .iFaceAlloc             = LwSciBufSysMemAlloc,
        .iFaceDeAlloc           = LwSciBufSysMemDealloc,
        .iFaceDupHandle         = LwSciBufSysMemDupHandle,
        .iFaceMemMap            = LwSciBufSysMemMemMap,
        .iFaceMemUnMap          = LwSciBufSysMemMemUnMap,
        .iFaceGetSize           = LwSciBufSysMemGetSize,
#if (LW_IS_SAFETY == 0)
        .iFaceGetAlignment      = LwSciBufSysMemGetAlignment,
        .iFaceGetHeapType       = LwSciBufSysMemGetHeapType,
#endif
        .iFaceCpuCacheFlush     = LwSciBufSysMemCpuCacheFlush,
        .iFaceClose             = LwSciBufSysMemClose,
        .iFaceGetAllocContext   = LwSciBufSysMemGetAllocContext,
    },

#if (LW_IS_SAFETY == 0)
    [LwSciBufAllocIfaceType_VidMem] = {
        .iFaceOpen              = LwSciBufVidMemOpen,
        .iFaceAlloc             = LwSciBufVidMemAlloc,
        .iFaceDeAlloc           = LwSciBufVidMemDealloc,
        .iFaceDupHandle         = LwSciBufVidMemDupHandle,
        .iFaceMemMap            = LwSciBufVidMemMemMap,
        .iFaceMemUnMap          = LwSciBufVidMemMemUnMap,
        .iFaceGetSize           = LwSciBufVidMemGetSize,
        .iFaceGetAlignment      = LwSciBufVidMemGetAlignment,
        .iFaceGetHeapType       = LwSciBufVidMemGetHeapType,
        .iFaceCpuCacheFlush     = LwSciBufVidMemCpuCacheFlush,
        .iFaceClose             = LwSciBufVidMemClose,
        .iFaceGetAllocContext   = LwSciBufVidMemGetAllocContext,
    },
#endif
};
LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 11_1))

/**
 * @brief Array of structures of type LwSciBufAllocIfaceHelperFvt, and the size
 * depends on the number of LwSciBufAllocIfaceType(s) supported. The function
 * pointer values of structure LwSciBufAllocIfaceHelperFvt, is initialized with
 * the functions respectively for the supported LwSciBufAllocIfaceType(s).
 */
static const LwSciBufAllocIfaceHelperFvt
                allocIfaceHelperFvt[LwSciBufAllocIfaceType_Max] = {
    [LwSciBufAllocIfaceType_SysMem] = {
        .iFaceCreateAllocVal    = LwSciBufAllocIfaceCreateSysMemAllocVal,
        .iFaceDestroyAllocVal   = LwSciBufAllocIfaceDestroySysMemAllocVal,
        .iFaceCreateAllocContextParams =
                                LwSciBufAllocIfaceCreateSysMemAllocContextParams,
        .iFaceDestroyAllocContextParams =
                            LwSciBufAllocIfaceDestroySysMemAllocContextParams,
    },

#if (LW_IS_SAFETY == 0)
    [LwSciBufAllocIfaceType_VidMem] = {
        .iFaceCreateAllocVal    = LwSciBufAllocIfaceCreateVidMemAllocVal,
        .iFaceDestroyAllocVal   = LwSciBufAllocIfaceDestroyVidMemAllocVal,
        .iFaceCreateAllocContextParams =
                                LwSciBufAllocIfaceCreateVidMemAllocContextParams,
        .iFaceDestroyAllocContextParams =
                            LwSciBufAllocIfaceDestroyVidMemAllocContextParams,
    },
#endif
};

LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 8_9), "LwSciBuf-ADV-MISRAC2012-011")
static const LwSciBufAllocIfaceToSysMemHeapMap
    allocIfaceToSysMemHeapMap[LwSciBufAllocIfaceHeapType_Ilwalid] = {
    [LwSciBufAllocIfaceHeapType_IOMMU] =
        {"LwSciBufAllocIfaceHeapType_IOMMU",
            LwSciBufAllocSysMemHeapType_IOMMU},

    [LwSciBufAllocIfaceHeapType_ExternalCarveout] =
        {"LwSciBufAllocIfaceHeapType_ExternalCarveout",
            LwSciBufAllocSysMemHeapType_ExternalCarveout},

    [LwSciBufAllocIfaceHeapType_IVC] =
        {"LwSciBufAllocIfaceHeapType_IVC",
            LwSciBufAllocSysMemHeapType_IVC},

    [LwSciBufAllocIfaceHeapType_VidMem] =
        {"LwSciBufAllocIfaceHeapType_VidMem",
            LwSciBufAllocSysMemHeapType_VidMem},

    [LwSciBufAllocIfaceHeapType_CvsRam] =
        {"LwSciBufAllocIfaceHeapType_CvsRam",
            LwSciBufAllocSysMemHeapType_CvsRam},
};
LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 8_9))

/**
 * @brief static function definitions
 */
static void LwSciBufAllocIfaceHeapTypeToSysMemHeapType(
    const LwSciBufAllocIfaceHeapType* allocIfaceHeapType,
    LwSciBufAllocSysMemHeapType* allocSysMemHeapType,
    uint32_t allocIfaceNumHeaps,
    uint32_t allocSysMemNumHeaps)
{
    uint32_t i;

    LWSCI_FNENTRY("");

#if (LW_IS_SAFETY)
    (void)allocSysMemNumHeaps;

    /* print input parameters */
    LWSCI_INFO("Input: allocIfaceHeapType: %p\tallocSysMemHeapType :%p\t"
        "allocIfaceNumHeaps: %u",
        allocIfaceHeapType, allocSysMemHeapType, allocIfaceNumHeaps);
#else
    /* print input parameters */
    LWSCI_INFO("Input: allocIfaceHeapType: %p\tallocSysMemHeapType :%p\t"
        "allocIfaceNumHeaps: %u\tallocSysMemNumHeaps: %u",
        allocIfaceHeapType, allocSysMemHeapType, allocIfaceNumHeaps,
        allocSysMemNumHeaps);
#endif


#if (LW_IS_SAFETY == 0)
    LwSciBufAllocIfacePrintHeapTypes(allocIfaceHeapType, allocIfaceNumHeaps);
#endif

    for (i = 0; i < allocIfaceNumHeaps; i++) {
        if (LwSciBufAllocIfaceHeapType_Ilwalid <= allocIfaceHeapType[i]) {
            LWSCI_ERR_STR("Invalid heap type supplied\n");
            LWSCI_ERR_UINT("enum value for heap type: \n", (uint32_t)allocIfaceHeapType[i]);
            LwSciCommonPanic();
        }

        allocSysMemHeapType[i] =
            allocIfaceToSysMemHeapMap[allocIfaceHeapType[i]].allocSysMemHeap;
    }

#if (LW_IS_SAFETY == 0)
    /* print output parameters */
    LwSciBufAllocSysMemPrintHeapTypes(allocSysMemHeapType, allocSysMemNumHeaps);
#endif

    LWSCI_FNEXIT("");
}

static void LwSciBufAllocIfaceValToSysMemAllocVal(
    LwSciBufAllocIfaceVal iFaceAllocVal,
    LwSciBufAllocSysMemVal* sysMemAllocVal)
{
    LWSCI_FNENTRY("");

    /* print input parameters */
    LWSCI_INFO("Input: sysMemAllocVal: %p\tiFaceAllocVal.size: %u\t"
        "iFaceAllocVal.heap: %p\tiFaceAllocVal.numHeaps: %u\n",
        sysMemAllocVal, iFaceAllocVal.size, iFaceAllocVal.heap,
        iFaceAllocVal.numHeaps);

#if (LW_IS_SAFETY == 0)
    LwSciBufAllocIfacePrintHeapTypes(iFaceAllocVal.heap,
        iFaceAllocVal.numHeaps);
#endif

    sysMemAllocVal->size = iFaceAllocVal.size;
    sysMemAllocVal->alignment = iFaceAllocVal.alignment;
    sysMemAllocVal->coherency = iFaceAllocVal.coherency;
    sysMemAllocVal->numHeaps = iFaceAllocVal.numHeaps;

    LwSciBufAllocIfaceHeapTypeToSysMemHeapType(iFaceAllocVal.heap,
        sysMemAllocVal->heap, iFaceAllocVal.numHeaps, sysMemAllocVal->numHeaps);

    LWSCI_FNEXIT("");
}

static LwSciError LwSciBufAllocIfaceCreateSysMemAllocVal(
    LwSciBufAllocIfaceVal iFaceAllocVal,
    void** allocVal)
{
    LwSciError sciErr = LwSciError_Success;
    LwSciBufAllocSysMemVal* sysMemAllocVal = NULL;

    LWSCI_FNENTRY("");

    /* print input parameters */
    LWSCI_INFO("Input: iFaceAllocVal.size: %u\tiFaceAllocVal.heap: %p\t"
        "iFaceAllocVal.numHeaps: %u\tallocVal: %p\n", iFaceAllocVal.size,
        iFaceAllocVal.heap, iFaceAllocVal.numHeaps, allocVal);

    /* allocate memory for storing alloc values of sysmem alloc interface */
    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
    sysMemAllocVal = LwSciCommonCalloc(1, sizeof(LwSciBufAllocSysMemVal));
    if (NULL == sysMemAllocVal) {
        sciErr = LwSciError_InsufficientMemory;
        LWSCI_ERR_STR("Could not allocate memory for LwSciBufAllocSysMemVal struct\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    /* allocate memory for storing sysmem alloc interface heap types */
    LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
    sysMemAllocVal->heap = LwSciCommonCalloc(iFaceAllocVal.numHeaps,
                            sizeof(LwSciBufAllocSysMemHeapType));
    LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 11_5))
    if (NULL == sysMemAllocVal->heap) {
        sciErr = LwSciError_InsufficientMemory;
        LWSCI_ERR_STR("Could not allocate memory for LwSciBufAllocSysMemHeapType\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto free_sysMemAllocVal;
    }

    LwSciBufAllocIfaceValToSysMemAllocVal(iFaceAllocVal, sysMemAllocVal);

    *allocVal = sysMemAllocVal;

    /* print output parameters */
    LWSCI_INFO("Output: *allocVal: %p\n", *allocVal);

    /* All opeartions are successful. Directly jump to 'ret' from here */
    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
    goto ret;

free_sysMemAllocVal:
    LwSciCommonFree(sysMemAllocVal);
ret:
    LWSCI_FNEXIT("");
    return sciErr;
}

static void LwSciBufAllocIfaceDestroySysMemAllocVal(
    void* allocVal)
{
    LwSciBufAllocSysMemVal* sysMemAllocVal = NULL;

    LWSCI_FNENTRY("");

    /* print input parameters */
    LWSCI_INFO("Input: allocVal: %p\n", allocVal);

    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
    sysMemAllocVal = (LwSciBufAllocSysMemVal*)allocVal;

    LwSciCommonFree(sysMemAllocVal->heap);
    LwSciCommonFree(sysMemAllocVal);

    LWSCI_FNEXIT("");
}

static LwSciError LwSciBufAllocIfaceCreateSysMemAllocContextParams(
    LwSciBufAllocIfaceAllocContextParams iFaceAllocContextParams,
    void** allocContextParams)
{
    LwSciError err = LwSciError_Success;
    LwSciBufAllocSysMemAllocContextParam* sysMemAllocContextParams = NULL;

    LWSCI_FNENTRY("");

    LWSCI_INFO("Input: allocContextParams %p\n", allocContextParams);

    LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
    sysMemAllocContextParams = LwSciCommonCalloc(1,
                                sizeof(LwSciBufAllocSysMemAllocContextParam));
    LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 11_5))
    if (NULL == sysMemAllocContextParams) {
        LWSCI_ERR_STR("Could not allocate memory for LwSciBufAllocSysMemAllocContextParam struct\n");
        err = LwSciError_InsufficientMemory;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    sysMemAllocContextParams->gpuIds = iFaceAllocContextParams.gpuIds;
    sysMemAllocContextParams->gpuIdsCount = iFaceAllocContextParams.gpuIdsCount;

    *allocContextParams = sysMemAllocContextParams;

ret:
    LWSCI_FNEXIT("");
    return err;
}

static void LwSciBufAllocIfaceDestroySysMemAllocContextParams(
    void* allocContextParams)
{
    LWSCI_FNENTRY("");

    LwSciCommonFree(allocContextParams);

    LWSCI_FNEXIT("");
}

#if (LW_IS_SAFETY == 0)
static void LwSciBufAllocIfaceValToVidMemAllocVal(
    LwSciBufAllocIfaceVal iFaceAllocVal,
    LwSciBufAllocVidMemVal* vidMemAllocVal)
{
    LWSCI_FNENTRY("");

    /* print input parameters */
    LWSCI_INFO("Input: iFaceAllocVal.size: %u\n", iFaceAllocVal.size);

    vidMemAllocVal->size = iFaceAllocVal.size;
    vidMemAllocVal->alignment = iFaceAllocVal.alignment;
    vidMemAllocVal->coherency = iFaceAllocVal.coherency;
    vidMemAllocVal->cpuMapping = iFaceAllocVal.cpuMapping;

    LWSCI_FNEXIT("");
}

static LwSciError LwSciBufAllocIfaceCreateVidMemAllocVal(
    LwSciBufAllocIfaceVal iFaceAllocVal,
    void** allocVal)
{
    LwSciError sciErr = LwSciError_Success;
    LwSciBufAllocVidMemVal* vidMemAllocVal = NULL;

    LWSCI_FNENTRY("");

    /* print input parameters */
    LWSCI_INFO("Input: iFaceAllocVal.size: %u, allocVal: %p\n",
        iFaceAllocVal.size, allocVal);

    /* allocate memory for storing alloc values of sysmem alloc interface */
    vidMemAllocVal = LwSciCommonCalloc(1, sizeof(LwSciBufAllocVidMemVal));
    if (NULL == vidMemAllocVal) {
        sciErr = LwSciError_InsufficientMemory;
        LWSCI_ERR_STR("Could not allocate memory for LwSciBufAllocVidMemVal struct\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    LwSciBufAllocIfaceValToVidMemAllocVal(iFaceAllocVal, vidMemAllocVal);

    *allocVal = vidMemAllocVal;

    /* print output parameters */
    LWSCI_INFO("Output: *allocVal: %p\n", *allocVal);

ret:
    LWSCI_FNEXIT("");
    return sciErr;
}

static void LwSciBufAllocIfaceDestroyVidMemAllocVal(
    void* allocVal)
{
    LwSciBufAllocVidMemVal* vidMemAllocVal = NULL;

    LWSCI_FNENTRY("");

    /* print input parameters */
    LWSCI_INFO("Input: allocVal: %p\n", allocVal);

    vidMemAllocVal = (LwSciBufAllocVidMemVal*)allocVal;

    LwSciCommonFree(vidMemAllocVal);

    LWSCI_FNEXIT("");
}

static LwSciError LwSciBufAllocIfaceCreateVidMemAllocContextParams(
    LwSciBufAllocIfaceAllocContextParams iFaceAllocContextParams,
    void** allocContextParams)
{
    LwSciError err = LwSciError_Success;
    LwSciBufAllocVidMemAllocContextParam *vidMemAllocContextParams = NULL;

    LWSCI_FNENTRY("");

    LWSCI_INFO("Input: allocContextParams %p\n", allocContextParams);

    vidMemAllocContextParams = LwSciCommonCalloc(1,
                                sizeof(LwSciBufAllocVidMemAllocContextParam));
    if (NULL == vidMemAllocContextParams) {
        LWSCI_ERR_STR("Could not allocate memory for LwSciBufAllocVidMemAllocContextParam struct\n");
        err = LwSciError_InsufficientMemory;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    vidMemAllocContextParams->gpuId = iFaceAllocContextParams.gpuId;

    *allocContextParams = vidMemAllocContextParams;

ret:
    LWSCI_FNEXIT("");
    return err;
}

static void LwSciBufAllocIfaceDestroyVidMemAllocContextParams(
    void* allocContextParams)
{
    LWSCI_FNENTRY("");

    LwSciCommonFree(allocContextParams);

    LWSCI_FNEXIT("");
}

#endif

/**
 * @brief public function definitions
 */
#if (LW_IS_SAFETY == 0)
void LwSciBufAllocIfacePrintHeapTypes(
    const LwSciBufAllocIfaceHeapType* heaps,
    uint32_t numHeaps)
{
    uint32_t i;

    LWSCI_FNENTRY("");

    /* verify input parameters */
    if ((NULL == heaps) || (0U == numHeaps)) {
        LWSCI_ERR_STR("Bad parameter supplied to LwSciBufAllocIfacePrintHeapTypes\n");
        LWSCI_ERR_UINT("numHeaps: \n", numHeaps);
        LwSciCommonPanic();
    }

    /* print input parameters */
    LWSCI_INFO("Input: heaps: %p\tnumHeaps: %u\n", heaps, numHeaps);

    LWSCI_INFO("allocIfaceHeapTypes:\t");

    for (i = 0; i < numHeaps; i++) {
        if (LwSciBufAllocIfaceHeapType_Ilwalid <= heaps[i]) {
            LWSCI_ERR_STR("Invalid heap type supplied\n");
            LWSCI_ERR_UINT("enum value for heap type: \n", heaps[i]);
            LwSciCommonPanic();
        }

        LWSCI_INFO("%s: \t", allocIfaceToSysMemHeapMap[heaps[i]].heapName);
    }
    LWSCI_INFO("\n");

    LWSCI_FNEXIT("");
}
#endif

LwSciError LwSciBufAllocIfaceOpen(
    LwSciBufAllocIfaceType allocType,
    LwSciBufDev devHandle,
    void** context)
{
    LwSciError sciErr = LwSciError_Success;
    LwSciBufAllocIfaceOpenFnPtr iFaceOpen;

    LWSCI_FNENTRY("");

    /* verify input parameters */
    if ((0 > (int32_t) allocType) || (LwSciBufAllocIfaceType_Max <= allocType)
        || (NULL == devHandle) || (NULL == context)) {
        LWSCI_ERR_STR("Bad parameter supplied to LwSciBufAllocIfaceOpen\n");
        LWSCI_ERR_UINT("allocType: \n", allocType);
        LwSciCommonPanic();
    }

    /* print input parameters */
    LWSCI_INFO("Input: allocType: %u\tdevHandle: %p\tcontext: %p\n",
        allocType, devHandle, context);

    iFaceOpen = allocIfaceFvt[allocType].iFaceOpen;
    if (NULL != iFaceOpen) {
        sciErr = iFaceOpen(devHandle, context);
        if (LwSciError_Success != sciErr) {
            if (LwSciError_NotSupported == sciErr) {
                LWSCI_WARN("open call to interface type %u not supported\n",
                    allocType);
            } else {
                LWSCI_ERR_UINT("open call to interface type failed: \n", (uint32_t)allocType);
            }

            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
            goto ret;
        }
    } else {
        LWSCI_WARN("open call for interface type %u not supported\n", allocType);
        sciErr = LwSciError_NotSupported;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    /* print output parameters */
    LWSCI_INFO("Output: *context: %p\n", *context);

ret:
    LWSCI_FNEXIT("");
    return sciErr;
}

LwSciError LwSciBufAllocIfaceAlloc(
    LwSciBufAllocIfaceType allocType,
    void* context,
    LwSciBufAllocIfaceVal iFaceAllocVal,
    LwSciBufDev devHandle,
    LwSciBufRmHandle *rmHandle)
{
    LwSciError sciErr = LwSciError_Success;
    void* allocVal = NULL;
    LwSciBufAllocIfaceAllocFnPtr iFaceAlloc = NULL;
    LwSciBufAllocIfaceCreateAllocVal iFaceCreateAllocVal = NULL;
    LwSciBufAllocIfaceDestroyAllocVal iFaceDestroyAllocVal = NULL;

    LWSCI_FNENTRY("");

    /* verify input parameters */
    if ((0 > (int32_t)allocType) || (LwSciBufAllocIfaceType_Max <= allocType)
        || (NULL == context) || (0U == iFaceAllocVal.size) || (NULL == iFaceAllocVal.heap)
        || (0U == iFaceAllocVal.numHeaps) || (NULL == devHandle) || (NULL == rmHandle))
    {
        LWSCI_ERR_STR("Bad parameter supplied to LwSciBufAllocIfaceAlloc\n");
        LWSCI_ERR_UINT("allocType: \t", (uint32_t)allocType);
        LWSCI_ERR_ULONG("iFaceAllocVal.size: \t", iFaceAllocVal.size);
        LWSCI_ERR_UINT("iFaceAllocVal.numHeaps: \n", iFaceAllocVal.numHeaps);
        LwSciCommonPanic();
    }

    iFaceCreateAllocVal = allocIfaceHelperFvt[allocType].iFaceCreateAllocVal;
    if (NULL != iFaceCreateAllocVal) {
        sciErr = iFaceCreateAllocVal(iFaceAllocVal, &allocVal);
        if (LwSciError_Success != sciErr) {
            LWSCI_ERR_UINT("LwSciBufAllocIfaceCreateAllocVal interface failed for allocType: \n",
                allocType);
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
            goto ret;
        }
    } else {
        sciErr = LwSciError_NotSupported;
        LWSCI_WARN("allocType: %u not supported yet!\n", allocType);
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    iFaceAlloc = allocIfaceFvt[allocType].iFaceAlloc;
    if (NULL != iFaceAlloc) {
        sciErr = iFaceAlloc(context, allocVal, devHandle, rmHandle);
        if (LwSciError_Success != sciErr) {
            LWSCI_ERR_UINT("allocation failed for allocation type: \n", allocType);
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
            goto free_allocVal;
        }
    } else {
        LWSCI_WARN("alloc call for interface type %u not supported\n",
            allocType);
        sciErr = LwSciError_NotSupported;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto free_allocVal;
    }

free_allocVal:
    iFaceDestroyAllocVal = allocIfaceHelperFvt[allocType].iFaceDestroyAllocVal;
    if (NULL != iFaceDestroyAllocVal) {
        iFaceDestroyAllocVal(allocVal);
    } else {
        LWSCI_WARN("allocType: %u not supported yet!\n", allocType);
        LWSCI_WARN("Potential memory leak! Please check if allocVal was allocated through LwSciBufAllocIfaceCreateAllocVal interface thay may not have been freed\n");
    }

ret:
   LWSCI_FNEXIT("");
    return sciErr;
}

LwSciError LwSciBufAllocIfaceDeAlloc(
    LwSciBufAllocIfaceType allocType,
    void* context,
    LwSciBufRmHandle rmHandle)
{
    LwSciError sciErr = LwSciError_Success;
    LwSciBufAllocIfaceDeAllocFnPtr iFaceDeAlloc = NULL;

    LWSCI_FNENTRY("");

    /* verify input parameters */
    if ((0 > (int32_t) allocType)
        || (LwSciBufAllocIfaceType_Max <= allocType)
        || (NULL == context))
    {
        LWSCI_ERR_STR("Bad parameter supplied to LwSciBufAllocIfaceDeAlloc\n");
        LWSCI_ERR_UINT("allocType: \n", allocType);
        LwSciCommonPanic();
    }

    /* print input parameters */
    LWSCI_INFO("Input: allocType: %u\tcontext: %p\n", allocType, context);

    iFaceDeAlloc = allocIfaceFvt[allocType].iFaceDeAlloc;
    if (NULL != iFaceDeAlloc) {
        sciErr = iFaceDeAlloc(context, rmHandle);
        if (LwSciError_Success != sciErr) {
            LWSCI_ERR_UINT("DeAllocation failed for allocation type \n",
                (uint32_t)allocType);
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
            goto ret;
        }
    } else {
        LWSCI_WARN("dealloc call for interface type %u not supported\n",
            allocType);
        sciErr = LwSciError_NotSupported;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

ret:
    LWSCI_FNEXIT("");
    return sciErr;
}

LwSciError LwSciBufAllocIfaceDupHandle(
    LwSciBufAllocIfaceType allocType,
    const void* context,
    LwSciBufAttrValAccessPerm newPerm,
    LwSciBufRmHandle rmHandle,
    LwSciBufRmHandle* dupHandle)
{
    LwSciError sciErr = LwSciError_Success;
    LwSciBufAllocIfaceDupHandleFnPtr iFaceDupHandle = NULL;

    LWSCI_FNENTRY("");

    /* verify input parameters */
    if ((0 > (int32_t) allocType) || (LwSciBufAllocIfaceType_Max <= allocType)
        || (NULL == context) || (NULL == dupHandle)
        || (LwSciBufAccessPerm_Ilwalid <= newPerm))
    {
        LWSCI_ERR_STR("Bad parameter supplied to LwSciBufAllocIfaceDupHandle\n");
        LWSCI_ERR_UINT("allocType: \n", (uint32_t)allocType);
        LwSciCommonPanic();
    }

    /* print input parameters */
    LWSCI_INFO("Input: allocType: %u\tcontext: %p\tdupHandle: %p\n",
        allocType, context, dupHandle);

    iFaceDupHandle = allocIfaceFvt[allocType].iFaceDupHandle;
    if (NULL != iFaceDupHandle) {
        sciErr = iFaceDupHandle(context, newPerm, rmHandle, dupHandle);
        if (LwSciError_Success != sciErr) {
            LWSCI_ERR_UINT("DupHandle failed for allocation type \n",
                allocType);
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
            goto ret;
        }
    } else {
        LWSCI_WARN("duphandle call for interface type %u not supported\n",
            allocType);
        sciErr = LwSciError_NotSupported;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

ret:
    LWSCI_FNEXIT("");
    return sciErr;
}

LwSciError LwSciBufAllocIfaceMemMap(
    LwSciBufAllocIfaceType allocType,
    const void* context,
    LwSciBufRmHandle rmHandle,
    uint64_t offset,
    uint64_t len,
    LwSciBufAttrValAccessPerm iFaceAccPerm,
    void** ptr)
{
    LwSciError sciErr = LwSciError_Success;
    LwSciBufAllocIfaceMemMapFnPtr iFaceMemMap = NULL;

    LWSCI_FNENTRY("");

    /* verify input parameters */
    if ((0 > (int32_t) allocType) || (LwSciBufAllocIfaceType_Max <= allocType)
        || (NULL == context) || (0U == len)
        || (LwSciBufAccessPerm_Ilwalid <= iFaceAccPerm) || (NULL == ptr))
    {
        LWSCI_ERR_STR("Bad parameter supplied to LwSciBufAllocIfaceMemMap\n");
        LWSCI_ERR_UINT("allocType: \t", allocType);
        LWSCI_ERR_ULONG("offset: \t", offset);
        LWSCI_ERR_ULONG("len: \t", len);
        LWSCI_ERR_UINT("iFaceAccPerm: \n", (uint32_t)iFaceAccPerm);
        LwSciCommonPanic();
    }

    iFaceMemMap = allocIfaceFvt[allocType].iFaceMemMap;
    if (NULL != iFaceMemMap) {
        sciErr = iFaceMemMap(context, rmHandle, offset, len, iFaceAccPerm, ptr);
        if (LwSciError_Success != sciErr) {
            LWSCI_ERR_UINT("MemMap failed for allocation type \n", (uint32_t)allocType);
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
            goto ret;
        }
    } else {
        LWSCI_WARN("memmap call for interface type %u not supported\n",
            allocType);
        sciErr = LwSciError_NotSupported;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    /* print output parameter */
    LWSCI_INFO("Output: *ptr: %p\n", *ptr);

ret:
    LWSCI_FNEXIT("");
    return sciErr;
}

LwSciError LwSciBufAllocIfaceMemUnMap(
    LwSciBufAllocIfaceType allocType,
    const void* context,
    LwSciBufRmHandle rmHandle,
    void* ptr,
    uint64_t size)
{
    LwSciError sciErr = LwSciError_Success;
    LwSciBufAllocIfaceMemUnMapFnPtr iFaceMemUnMap = NULL;

    LWSCI_FNENTRY("");

    /* verify input parameters */
    if ((0 > (int32_t) allocType) || (LwSciBufAllocIfaceType_Max <= allocType)
        || (NULL == context) || (NULL == ptr) || (0U == size))
    {
        LWSCI_ERR_STR("Bad parameter supplied to LwSciBufAllocIfaceMemUnMap\n");
        LWSCI_ERR_UINT("allocType: \n", (uint32_t)allocType);
        LWSCI_ERR_ULONG("size: \n", size);
        LwSciCommonPanic();
    }

    /* print input parameters */
    LWSCI_INFO("Input: allocType: %u\tcontext: %p\tptr: %p\t"
        "size: %lu\n", allocType, context, ptr, size);

    iFaceMemUnMap = allocIfaceFvt[allocType].iFaceMemUnMap;
    if (NULL != iFaceMemUnMap) {
        sciErr = iFaceMemUnMap(context, rmHandle, ptr, size);
        if (LwSciError_Success != sciErr) {
            LWSCI_ERR_UINT("MemUnMap failed for allocation type \n",
                allocType);
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
            goto ret;
        }
    } else {
        LWSCI_WARN("memunmap call for interface type %u not supported\n",
            allocType);
        sciErr = LwSciError_NotSupported;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

ret:
    LWSCI_FNEXIT("");
    return sciErr;
}

LwSciError LwSciBufAllocIfaceGetSize(
    LwSciBufAllocIfaceType allocType,
    const void* context,
    LwSciBufRmHandle rmHandle,
    uint64_t* size)
{
    LwSciError sciErr = LwSciError_Success;
    LwSciBufAllocIfaceGetSizeFnPtr iFaceGetSize = NULL;

    LWSCI_FNENTRY("");

    /*  verify input parameters */
    if ((0 > (int32_t) allocType) || (LwSciBufAllocIfaceType_Max <= allocType)
        || (NULL == context) || (size == NULL))
    {
        LWSCI_ERR_STR("Bad parameter supplied to LwSciBufAllocIfaceGetSize\n");
        LWSCI_ERR_UINT("allocType: \n", allocType);
        LwSciCommonPanic();
    }

    /* print input parameters */
    LWSCI_INFO("Input: allocType: %u\tcontext: %p\tsize: %p\n", allocType,
        context, size);

    iFaceGetSize = allocIfaceFvt[allocType].iFaceGetSize;

    if (NULL != iFaceGetSize) {
        sciErr = iFaceGetSize(context, rmHandle, size);
        if (LwSciError_Success != sciErr) {
            LWSCI_ERR_UINT("GetSize failed for allocType: \n", (uint32_t)allocType);
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
            goto ret;
        }
    } else {
        sciErr = LwSciError_NotSupported;
        LWSCI_WARN("Getting buffer size from memhandle is not supported for allocType: %u\n",
            allocType);
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    /* print output parameters */
    LWSCI_INFO("Output: buffer size: %lu\n", *size);

ret:
    LWSCI_FNEXIT("");
    return sciErr;
}

#if (LW_IS_SAFETY == 0)
LwSciError LwSciBufAllocIfaceGetAlignment(
    LwSciBufAllocIfaceType allocType,
    void* context,
    LwSciBufRmHandle rmHandle,
    uint64_t* alignment)
{
    LwSciError err = LwSciError_Success;

    (void)allocType;
    (void)rmHandle;
    (void)alignment;
    (void)context;

    LWSCI_FNENTRY("");

    LWSCI_FNEXIT("");
    return err;
}

LwSciError LwSciBufAllocIfaceGetHeapType(
    LwSciBufAllocIfaceType allocType,
    void* context,
    LwSciBufRmHandle rmHandle,
    LwSciBufAllocIfaceHeapType* heap)
{
    LwSciError err = LwSciError_Success;

    (void)allocType;
    (void)context;
    (void)rmHandle;
    (void)heap;

    LWSCI_FNENTRY("");

    LWSCI_FNEXIT("");
    return err;
}
#endif

LwSciError LwSciBufAllocIfaceCpuCacheFlush(
    LwSciBufAllocIfaceType allocType,
    const void* context,
    LwSciBufRmHandle rmHandle,
    void* cpuPtr,
    uint64_t len)
{
    LwSciError sciErr = LwSciError_Success;
    LwSciBufAllocIfaceCpuCacheFlushFnPtr iFaceCpuCacheFlush;

    LWSCI_FNENTRY("");

    /* verify input parameters */
    if ((0 > (int32_t) allocType) || (LwSciBufAllocIfaceType_Max <= allocType)
        || (NULL == context) || (NULL == cpuPtr) || (0U == len)) {
        LWSCI_ERR_STR("Bad parameter supplied to LwSciBufAllocIfaceCpuCacheFlush\n");
        LWSCI_ERR_UINT("allocType: \n", (uint32_t)allocType);
        LWSCI_ERR_ULONG("len: \n", len);
        LwSciCommonPanic();
    }

    /* print input parameters */
    LWSCI_INFO("Inputs - allocType: %u\tcontext: %p\t"
               "cpuPtr: %p\tlen: %lu \n",
               allocType, context, cpuPtr, len);

    iFaceCpuCacheFlush = allocIfaceFvt[allocType].iFaceCpuCacheFlush;
    if (NULL != iFaceCpuCacheFlush) {
        sciErr = iFaceCpuCacheFlush(context, rmHandle, cpuPtr, len);
    } else {
        LWSCI_WARN("CpuCacheFlush call for interface type %u not supported\n",
            allocType);
        sciErr = LwSciError_NotSupported;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

ret:
    LWSCI_FNEXIT("");
    return (sciErr);
}

void LwSciBufAllocIfaceClose(
    LwSciBufAllocIfaceType allocType,
    void* context)
{
    LwSciBufAllocIfaceCloseFnPtr iFaceClose;

    LWSCI_FNENTRY("");

    /* verify input parameters */
    if ((0 > (int32_t) allocType)
        || (LwSciBufAllocIfaceType_Max <= allocType)
        || (NULL == context))
    {
        LWSCI_ERR_STR("Bad parameter supplied to LwSciBufAllocIfaceClose\n");
        LWSCI_ERR_UINT("allocType: \n", (uint32_t)allocType);
        LwSciCommonPanic();
    }

    /* print input parameters */
    LWSCI_INFO("Input: allocType: %u\tcontext: %p\n", allocType, context);

    iFaceClose = allocIfaceFvt[allocType].iFaceClose;
    if (NULL != iFaceClose) {
        iFaceClose(context);
    } else {
        LWSCI_WARN("close call for interface type %u not supported\n",
            allocType);
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

ret:
    LWSCI_FNEXIT("");
}

LwSciError LwSciBufAllocIfaceGetAllocContext(
    LwSciBufAllocIfaceType allocType,
    LwSciBufAllocIfaceAllocContextParams iFaceAllocContextParams,
    void* openContext,
    void** allocContext)
{
    LwSciError err = LwSciError_Success;
    void* allocContextParams = NULL;
    LwSciBufAllocIfaceCreateAllocContextParam iFaceCreateAllocContextParam =
                                                                        NULL;
    LwSciBufAllocIfaceDestroyAllocContextParam iFaceDestroyAllocContextParam =
                                                                        NULL;
    LwSciBufAllocIfaceGetAllocContextFnPtr iFaceGetAllocContext = NULL;

    LWSCI_FNENTRY("");

    if ((0 > (int32_t) allocType) || (LwSciBufAllocIfaceType_Max <= allocType)
        || (NULL == openContext) || (NULL == allocContext)) {
        LWSCI_ERR_STR("Bad parameter supplied to LwSciBufAllocIfaceGetAllocContext\n");
        LWSCI_ERR_UINT("allocType: \n", allocType);
        LwSciCommonPanic();
    }

    LWSCI_INFO("Input: allocType: %u, openContext: %p, allocContext: %p\n",
        allocType, openContext, allocContext);

    /* colwert alloc context params */
    iFaceCreateAllocContextParam =
        allocIfaceHelperFvt[allocType].iFaceCreateAllocContextParams;
    if (NULL != iFaceCreateAllocContextParam) {
        err = iFaceCreateAllocContextParam(iFaceAllocContextParams,
                &allocContextParams);
        if (LwSciError_Success != err) {
            LWSCI_ERR_UINT("Failed to colwert alloc context params for alloc Type: \n",
                allocType);
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
            goto ret;
        }
    } else {
        LWSCI_WARN("creating alloc context for alloc type %u not supported\n",
            allocType);
        err = LwSciError_NotSupported;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    iFaceGetAllocContext = allocIfaceFvt[allocType].iFaceGetAllocContext;
    if (NULL != iFaceGetAllocContext) {
    err = iFaceGetAllocContext(allocContextParams, openContext, allocContext);
        if (LwSciError_Success != err) {
            LWSCI_ERR_UINT("Could not get alloc context for alloc type \n", (uint32_t)allocType);
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
            goto free_allocContext;
        }
    } else {
        LWSCI_WARN("getting alloc context for alloc type %u not supported\n",
            allocType);
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto free_allocContext;
    }

free_allocContext:
    iFaceDestroyAllocContextParam =
        allocIfaceHelperFvt[allocType].iFaceDestroyAllocContextParams;
    if (NULL != iFaceDestroyAllocContextParam) {
        iFaceDestroyAllocContextParam(allocContextParams);
    } else {
        LWSCI_WARN("destroying alloc context for alloc type %u not supported\n");
        LWSCI_WARN("Potential memory leak! Please check if alloc context was created via LwSciBufAllocIfaceCreateAllocContextParam call that may not have been freed\n");
    }

ret:
    LWSCI_FNEXIT("");
    return err;
}
