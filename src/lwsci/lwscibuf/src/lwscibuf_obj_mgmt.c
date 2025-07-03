/*
 * Copyright (c) 2018-2021, LWPU CORPORATION. All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#include <string.h>

#include "lwscibuf_obj_mgmt_priv.h"
#include "lwscicommon_os.h"

/**
 * @brief static variables
 */
LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 8_9), "LwSciBuf-ADV-MISRAC2012-011")
static const LwSciBufObjToallocIfaceHeapMap
    allocObjToallocIfaceHeapMap[LwSciBufHeapType_Ilwalid] = {
    [LwSciBufHeapType_IOMMU] =
        {"LwSciBufHeapType_IOMMU",
            LwSciBufAllocIfaceHeapType_IOMMU},

    [LwSciBufHeapType_ExternalCarveout] =
        {"LwSciBufHeapType_ExternalCarveout",
            LwSciBufAllocIfaceHeapType_ExternalCarveout},

    [LwSciBufHeapType_IVC] =
        {"LwSciBufHeapType_IVC",
            LwSciBufAllocIfaceHeapType_IVC},

    [LwSciBufHeapType_VidMem] =
        {"LwSciBufHeapType_VidMem",
            LwSciBufAllocIfaceHeapType_VidMem},

    [LwSciBufHeapType_CvsRam] =
        {"LwSciBufHeapType_CvsRam",
            LwSciBufAllocIfaceHeapType_CvsRam},
};
LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 8_9))

/**
 * @brief static functions
 */
static LwSciError ValidateAccessPerm(
    LwSciBufAttrValAccessPerm perm)
{
    LwSciError error = LwSciError_Success;

    LWSCI_FNENTRY("");

    switch (perm) {
        case LwSciBufAccessPerm_Readonly:
        case LwSciBufAccessPerm_ReadWrite:
        {
            error = LwSciError_Success;
            break;
        }
        default:
        {
            error = LwSciError_BadParameter;
            break;
        }
    }

    LWSCI_FNEXIT("");
    return error;
}

static void LwSciBufObjValidate(
    const LwSciBufObjPriv* objPriv)
{
    LWSCI_FNENTRY("");

    /* validate LwSciBufObjPriv object */
    if (LW_SCI_BUF_OBJ_MAGIC != objPriv->magic) {
        LWSCI_ERR_STR("Validattion of LwSciBufObjPriv object failed\n");
        LWSCI_ERR_HEXUINT("objPriv->magic: 0x\n", objPriv->magic);
        LwSciCommonPanic();
    }

    /* print input parameters */
    LWSCI_INFO("objPriv: %p, objPriv->magic: 0x%x\n", objPriv,
        objPriv != NULL ? objPriv->magic : 0U);

    LWSCI_FNEXIT("");
}

static void LwSciBufObjGetAndValidate(
    LwSciBufObj bufObj,
    LwSciBufObjPriv** objPriv)
{
    LwSciObj* objPrivParam = NULL;

    LWSCI_FNENTRY("");

    /* print input parameters */
    LWSCI_INFO("Input: bufObj: %p, objPriv: %p\n", bufObj, objPriv);

    LwSciCommonGetObjFromRef(&bufObj->refHeader, &objPrivParam);

    *objPriv = LwSciCastObjToBufObjPriv(objPrivParam);

    LwSciBufObjValidate(*objPriv);

    /* print output variables */
    LWSCI_INFO("Output: *objPriv: %p\n", *objPriv);

    LWSCI_FNEXIT("");
}

static void LwSciBufObjPrintHeapTypes(
    const LwSciBufHeapType* heaps,
    uint32_t numHeaps)
{
    uint32_t i;

    LWSCI_FNENTRY("");

    /* print input parameters */
    LWSCI_INFO("Input: heaps: %p, numHeaps: %u\n", heaps, numHeaps);

    LWSCI_INFO("objAllocHeapTypes:, ");

    for (i = 0; i < numHeaps; i++) {
        if (LwSciBufHeapType_Ilwalid <= heaps[i]) {
            LWSCI_ERR_STR("Invalid heap type supplied\n");
            LWSCI_ERR_UINT("enum value for heap type: \n", (uint32_t)heaps[i]);
            LwSciCommonPanic();
        }

        LWSCI_INFO("%s: , ", allocObjToallocIfaceHeapMap[heaps[i]].heapName);
    }
    LWSCI_INFO("\n");

    LWSCI_FNEXIT("");
}

static void LwSciBufObjHeapTypeToallocIfaceHeapType(
    const LwSciBufHeapType* allocObjHeapType,
    LwSciBufAllocIfaceHeapType* allocIfaceHeapType,
    uint32_t allocObjNumHeaps,
    uint32_t allocIfaceNumHeaps)
{
    uint32_t i;
    (void)allocIfaceNumHeaps;

    LWSCI_FNENTRY("");

    /* print input parameters */
    LWSCI_INFO("Input: allocObjHeapType: %p, allocIfaceHeapType :%p, "
        "allocObjNumHeaps: %u, allocIfaceNumHeaps: %u\n",
        allocObjHeapType, allocIfaceHeapType, allocObjNumHeaps,
        allocIfaceNumHeaps);
    LwSciBufObjPrintHeapTypes(allocObjHeapType, allocObjNumHeaps);

    for (i = 0; i < allocObjNumHeaps; i++) {
        if (LwSciBufHeapType_Ilwalid <= allocObjHeapType[i]) {
            LWSCI_ERR_STR("Invalid heap type supplied\n");
            LWSCI_ERR_UINT("enum value for heap type: \n", (uint32_t)allocObjHeapType[i]);
            LwSciCommonPanic();
        }

        allocIfaceHeapType[i] =
            allocObjToallocIfaceHeapMap[allocObjHeapType[i]].allocIfaceHeap;
    }

#if (LW_IS_SAFETY == 0)
    /* print output parameters */
    LwSciBufAllocIfacePrintHeapTypes(allocIfaceHeapType, allocIfaceNumHeaps);
#endif

}

static void LwSciBufObjAllocValToAllocIfaceAllocVal(
    LwSciBufObjAllocVal allocVal,
    LwSciBufAllocIfaceVal* allocIfaceVal)
{
    LWSCI_FNENTRY("");

    /* print input parameters */
    LWSCI_INFO("Input: LwSciBufObjAllocVal:\n");
    LWSCI_INFO("size: %u\nalignment: %u\ncoherency: %u\nnumHeaps: %u\n",
        allocVal.size, allocVal.alignment, allocVal.coherency,
        allocVal.numHeaps);
    LwSciBufObjPrintHeapTypes(allocVal.heap, allocVal.numHeaps);

    allocIfaceVal->size         = allocVal.size;
    allocIfaceVal->alignment    = allocVal.alignment;
    allocIfaceVal->coherency    = allocVal.coherency;
    allocIfaceVal->numHeaps     = allocVal.numHeaps;
    allocIfaceVal->cpuMapping   = allocVal.cpuMapping;

    LwSciBufObjHeapTypeToallocIfaceHeapType(allocVal.heap,
                allocIfaceVal->heap, allocVal.numHeaps,
                allocIfaceVal->numHeaps);
}

static LwSciError LwSciBufObjIsCpuAccessNeeded(
    LwSciBufAttrList reconciledAttrList,
    bool* needCpuAccess)
{
    LwSciError sciErr = LwSciError_Success;
    LwSciBufAttrKeyValuePair pairArray[1];

    LWSCI_FNENTRY("");

    sciErr = LwSciBufAttrListCompareReconcileStatus(reconciledAttrList, true);
    if (LwSciError_Success != sciErr) {
        LWSCI_ERR_STR("LwSciBufAttrListCompareReconcileStatus failed\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    /* print input parameters */
    LWSCI_INFO("Input: reconciledAttrList: %p, needCpuAccess: %p\n",
        reconciledAttrList, needCpuAccess);

    pairArray[0].key = LwSciBufGeneralAttrKey_NeedCpuAccess;

    sciErr = LwSciBufAttrListCommonGetAttrs(reconciledAttrList, 0, pairArray,
                1, LwSciBufAttrKeyType_Public, true);
    if (LwSciError_Success != sciErr) {
        LWSCI_ERR_STR("LwSciBufAttrListCommonGetAttrs failed\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    LwSciCommonMemcpyS(needCpuAccess, sizeof(*needCpuAccess),
                        pairArray[0].value, pairArray[0].len);

    /* print output parameters */
    LWSCI_INFO("Output: needCpuAccess: %s\n",
                    (*needCpuAccess == true) ? "Yes" : "No");

ret:
    LWSCI_FNEXIT("");
    return sciErr;
}

static LwSciError LwSciBufObjGetAllocVal(
    LwSciBufAttrList reconciledAttrList,
    LwSciBufObjAllocVal* allocVal)
{
    LwSciError sciErr = LwSciError_Success;
    LwSciBufAttrKeyValuePair genKeyPair[2];
    LwSciBufPrivateAttrKeyValuePair pairArray[3];
    /* number of elements in LwSciBufObjAllocVal struct */

    LWSCI_FNENTRY("");

    sciErr = LwSciBufAttrListCompareReconcileStatus(reconciledAttrList, true);
    if (LwSciError_Success != sciErr) {
        LWSCI_ERR_STR("LwSciBufAttrListCompareReconcileStatus failed\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    /* print input parameters */
    LWSCI_INFO("Input: allocVal: %p, reconciledAttrList: %p\n", allocVal,
        reconciledAttrList);

    /* Read Size, Alignment, HeapType and Enable CPU Cache flag attrs */
    pairArray[0].key = LwSciBufPrivateAttrKey_Size;
    pairArray[1].key = LwSciBufPrivateAttrKey_Alignment;
    pairArray[2].key = LwSciBufPrivateAttrKey_HeapType;
    /* TODO: assert if len returned by LwSciBufAttrListGetPrivateAttrs !=
     *  len(attrs)
     */
    sciErr = LwSciBufAttrListCommonGetAttrs(reconciledAttrList, 0, pairArray,
                sizeof(pairArray)/sizeof(pairArray[0]),
                LwSciBufAttrKeyType_Private, true);
    if (LwSciError_Success != sciErr) {
        LWSCI_ERR_STR("LwSciBufAttrListCommonGetAttrs failed\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    LwSciCommonMemcpyS(&allocVal->size, sizeof(allocVal->size),
                        pairArray[0].value, pairArray[0].len);

    LwSciCommonMemcpyS(&allocVal->alignment, sizeof(allocVal->alignment),
                        pairArray[1].value, pairArray[1].len);

    LwSciCommonMemcpyS(&allocVal->heap[0], sizeof(allocVal->heap),
                        pairArray[2].value, pairArray[2].len);

    allocVal->numHeaps = 1;

    /* TODO: assert if len returned by LwSciBufAttrListGetAttrs != len(attrs)
     */
    genKeyPair[0].key = LwSciBufGeneralAttrKey_EnableCpuCache;
    genKeyPair[1].key = LwSciBufGeneralAttrKey_NeedCpuAccess;
    sciErr = LwSciBufAttrListCommonGetAttrs(reconciledAttrList, 0, genKeyPair,
                2, LwSciBufAttrKeyType_Public, true);
    if (LwSciError_Success != sciErr) {
        LWSCI_ERR_STR("LwSciBufAttrListCommonGetAttrs failed\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    LwSciCommonMemcpyS(&allocVal->coherency, sizeof(allocVal->coherency),
                        genKeyPair[0].value, genKeyPair[0].len);

    LwSciCommonMemcpyS(&allocVal->cpuMapping, sizeof(allocVal->cpuMapping),
                        genKeyPair[1].value, genKeyPair[1].len);

    /* sanitize obtained alloc parameters */
    if (0U == allocVal->size) {
        LWSCI_ERR_STR("size obtained through reconciled attrlist is 0\n");
        LwSciCommonPanic();
    }

    /* print output parameters */
    LWSCI_INFO("Output:\n");
    LWSCI_INFO("allocVal->size: %u\n", allocVal->size);
    LWSCI_INFO("allocVal->alignment: %u\n", allocVal->alignment);
    LWSCI_INFO("allocVal->coherency: %s\n", allocVal->coherency ?
                                                        "true" : "false");
    LWSCI_INFO("allocVal->cpuMapping: %s\n", allocVal->cpuMapping ?
                                                        "true" : "false");

ret:
    LWSCI_FNEXIT("");
    return sciErr;
}

#if (LW_IS_SAFETY == 0)
/* LwSciBufObjCompareAllocVal() function is only used by
   LwSciBufObjCreateSubBuf() API. Since, LwSciBufObjCreateSubBuf() API is out
   of safety scope, this function needs to be out of safety scope too.
   We can re-introduce this function in safety build once any other functions
   which are part of safety start using it.
*/
static bool LwSciBufObjCompareAllocVal(
    const LwSciBufObjAllocVal* parentAllocVal,
    const LwSciBufObjAllocVal* childAllocVal)
{
    bool result = false;

    LWSCI_FNENTRY("");

    /* print input parameters */
    LWSCI_INFO("Input: parentAllocVal: %p, childAllocVal: %p\n",
        parentAllocVal, childAllocVal);

    /* compare alloc values */
    if (parentAllocVal->size <= childAllocVal->size) {
        /* size requested for child buf is more than parent buf size */
        LWSCI_ERR_STR("size of child buf cannot be more than parent buf\n");
        LWSCI_ERR_UINT("parent buf size: ", parentAllocVal->size);
        LWSCI_ERR_UINT("  child buf size: \n", childAllocVal->size);
        LwSciCommonPanic();
    }

    if (parentAllocVal->alignment < childAllocVal->alignment) {
        /* child buffer alignment should be equal or greater than parent buf */
        LWSCI_ERR_STR("child buf alignment should be equal or greater than parent buf\n");
        LWSCI_ERR_UINT("child buf alignment requested: \n",
            childAllocVal->alignment);
        LwSciCommonPanic();
    }

    if (parentAllocVal->coherency != childAllocVal->coherency) {
        LWSCI_ERR_STR("parent and child buf coherency do not match\n");
        LwSciCommonPanic();
    }

    /* TODO: compare heap types */
    result = true;

    LWSCI_FNEXIT("");

    return result;
}
#endif

static LwSciError LwSciBufObjGetAccessPerm(
    LwSciBufAttrList reconciledAttrList,
    LwSciBufAttrValAccessPerm* accessPerm)
{
    LwSciError sciErr = LwSciError_Success;
    LwSciBufAttrKeyValuePair pairArray[1];

    LWSCI_FNENTRY("");

    sciErr = LwSciBufAttrListCompareReconcileStatus(reconciledAttrList, true);
    if (LwSciError_Success != sciErr) {
        LWSCI_ERR_STR("LwSciBufAttrListCompareReconcileStatus failed\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    /* print input parameters */
    LWSCI_INFO("Input: reconciledAttrList: %p, accessPerm: %p\n",
        reconciledAttrList, accessPerm);

    pairArray[0].key = LwSciBufGeneralAttrKey_ActualPerm;

    sciErr = LwSciBufAttrListCommonGetAttrs(reconciledAttrList, 0, pairArray,
                1, LwSciBufAttrKeyType_Public, true);
    if (LwSciError_Success != sciErr) {
        LWSCI_ERR_STR("LwSciBufAttrListCommonGetAttrs failed\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    LwSciCommonMemcpyS(accessPerm, sizeof(*accessPerm),
                        pairArray[0].value, pairArray[0].len);

    /* print output parameters */
    LWSCI_INFO("Output: accessPerm: %u\n", *accessPerm);

ret:
    LWSCI_FNEXIT("");
    return sciErr;
}

static LwSciError LwSciBufObjGetAllocType(
    LwSciBufObjAllocTypeParams allocTypeParams,
    LwSciBufAllocIfaceType* allocType)
{
    LwSciError sciErr = LwSciError_Success;

    LWSCI_FNENTRY("");

    /* print input parameters */
    LWSCI_INFO("Input: allocType:%p\n", allocType);
    LWSCI_INFO("allocTypeParams:\n");
    LWSCI_INFO("memory domain: %u\n", allocTypeParams.memDomain);

    if (LwSciBufMemDomain_Sysmem == allocTypeParams.memDomain
#if (LW_IS_SAFETY == 0)
        || LwSciBufMemDomain_Cvsram == allocTypeParams.memDomain
#endif
        ) {
        *allocType = LwSciBufAllocIfaceType_SysMem;
#if (LW_IS_SAFETY == 0)
    } else if (LwSciBufMemDomain_Vidmem == allocTypeParams.memDomain){
        *allocType = LwSciBufAllocIfaceType_VidMem;
#endif
    } else {
        LWSCI_ERR_STR("Could not set allocation type based on allocTypeParams\n");
        *allocType = LwSciBufAllocIfaceType_Max;
        sciErr = LwSciError_NotSupported;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    /* print output params */
    LWSCI_INFO("Output: allocType: %u\n", *allocType);

ret:
    LWSCI_FNEXIT("");
    return sciErr;
}

static LwSciError LwSciBufObjGetAllocContextParams(
    LwSciBufAttrList reconciledList,
    LwSciBufAllocIfaceAllocContextParams* allocContextParams)
{
    LwSciError sciErr = LwSciError_Success;
    LwSciBufAttrKeyValuePair keyValPair[2];

    LWSCI_FNENTRY("");

    (void)memset(&keyValPair, 0x0, sizeof(keyValPair));

    sciErr = LwSciBufAttrListCompareReconcileStatus(reconciledList, true);
    if (LwSciError_Success != sciErr) {
        LWSCI_ERR_STR("LwSciBufAttrListCompareReconcileStatus failed\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    LWSCI_INFO("Input: reconciledList %p, allocContextParams %p\n", reconciledList,
                    allocContextParams);

    /* get GPU ID */
    keyValPair[0].key = LwSciBufGeneralAttrKey_VidMem_GpuId;
    keyValPair[1].key = LwSciBufGeneralAttrKey_GpuId;

    sciErr = LwSciBufAttrListCommonGetAttrs(reconciledList, 0, &keyValPair[0],
        sizeof(keyValPair)/sizeof(keyValPair[0]),
        LwSciBufAttrKeyType_Public, true);
    if (LwSciError_Success != sciErr) {
        LWSCI_ERR_STR("LwSciBufAttrListCommonGetAttrs failed\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

#if (LW_IS_SAFETY == 0)
    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
    allocContextParams->gpuId = *(const LwSciRmGpuId*)keyValPair[0].value;
#endif
    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
    allocContextParams->gpuIds = (const LwSciRmGpuId*)keyValPair[1].value;
    allocContextParams->gpuIdsCount = keyValPair[1].len / sizeof(LwSciRmGpuId);

ret:
    LWSCI_FNEXIT("");
    return sciErr;
}

static LwSciError LwSciBufObjGetAllocTypeParams(
    LwSciBufAttrList reconciledAttrList,
    LwSciBufObjAllocTypeParams* allocTypeParams)
{
    LwSciError sciErr = LwSciError_Success;
    LwSciBufPrivateAttrKeyValuePair pairArray[1] = {
        {
            .key = LwSciBufPrivateAttrKey_LowerBound,
            .value = NULL,
            .len = 0U
        }
    };

    LWSCI_FNENTRY("");

    sciErr = LwSciBufAttrListCompareReconcileStatus(reconciledAttrList, true);
    if (LwSciError_Success != sciErr) {
        LWSCI_ERR_STR("LwSciBufAttrListCompareReconcileStatus failed\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    /* print input parameters */
    LWSCI_INFO("Input: reconciledAttrList: %p, allocTypeParams: %p\n",
        reconciledAttrList, allocTypeParams);

    pairArray[0].key = LwSciBufPrivateAttrKey_MemDomain;

    sciErr = LwSciBufAttrListCommonGetAttrs(reconciledAttrList, 0, pairArray,
                sizeof(pairArray)/sizeof(pairArray[0]),
                LwSciBufAttrKeyType_Private, true);
    if (LwSciError_Success != sciErr) {
        LWSCI_ERR_STR("LwSciBufAttrListCommonGetAttrs failed\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    /* verify that length of value returned by LwSciBufPrivateAttrKey_MemDomain
     * matches with sizeof LwSciBufMemDomain
     */
    if ((sizeof(LwSciBufMemDomain) != pairArray[0].len)) {
        LWSCI_ERR_STR("Error oclwred in LwSciBufAttrListGetPrivateAttrs() call while reading LwSciBufPrivateAttrKey_MemDomain key");
        LWSCI_ERR_ULONG("length of value returned by LwSciBufAttrListGetPrivateAttrs(): , ", pairArray[0].len);
        LWSCI_ERR_ULONG("size of LwSciBufMemDomain datatype: ", sizeof(LwSciBufMemDomain));
        LwSciCommonPanic();
    }

    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
    allocTypeParams->memDomain = *(const LwSciBufMemDomain *)pairArray[0].value;

    /* print output parameters */
    LWSCI_INFO("Output: LwSciBufObjAllocTypeParams:\n");
    LWSCI_INFO("memory domain: %u\n", allocTypeParams->memDomain);

ret:
    LWSCI_FNEXIT("");
    return sciErr;
}

LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 8_13), "LwSciBuf-ADV-MISRAC2012-012")
static LwSciError LwSciBufObjGetAllocAttrs(
    LwSciBufAttrList reconciledAttrList,
    LwSciBufObjPriv** objPriv)
{
    LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 8_13))
    LwSciError sciErr = LwSciError_Success;
    LwSciBufObjAllocTypeParams allocTypeparam = {0};

    LWSCI_FNENTRY("");

    /* Get parameters needed for finding out allocation interface to be used
     * for allocating buffer
     */
    sciErr = LwSciBufObjGetAllocTypeParams(reconciledAttrList, &allocTypeparam);
    if (LwSciError_Success != sciErr) {
        LWSCI_ERR_STR("LwSciBufObjGetAllocTypeParams failed\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    /* get allocation interface (type) */
    sciErr = LwSciBufObjGetAllocType(allocTypeparam, &(*objPriv)->allocType);
    if (LwSciError_Success != sciErr) {
        LWSCI_ERR_STR("LwSciBufObjGetAllocType failed\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    /* Get attrs needed for buffer allocation */
    sciErr = LwSciBufObjGetAllocVal(reconciledAttrList, &(*objPriv)->allocVal);
    if (LwSciError_Success != sciErr) {
        LWSCI_ERR_STR("LwSciBufObjGetAllocVal failed\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }
ret:
    LWSCI_FNEXIT("");
    return sciErr;
}

LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 8_13), "LwSciBuf-ADV-MISRAC2012-012")
static LwSciError LwSciBufGetModuleAndAllocAttrs(
    LwSciBufAttrList reconciledAttrList,
    LwSciBufModule* module,
    LwSciBufObjPriv** objPtr)
{
    LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 8_13))
    LwSciError sciErr = LwSciError_Success;
    LwSciBufObjPriv* objPriv = NULL;
    LwSciBufAllocIfaceAllocContextParams allocContextParams;

    LWSCI_FNENTRY("");

    (void)memset(&allocContextParams, 0, sizeof(LwSciBufAllocIfaceAllocContextParams));

    objPriv = *objPtr;

    /* Get device handle */
    sciErr = LwSciBufAttrListGetModule(objPriv->attrList, module);
    if (LwSciError_Success != sciErr) {
        LWSCI_ERR_STR("LwSciBufAttrListGetModule failed\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    sciErr = LwSciBufModuleGetAllocIfaceOpenContext(*module, objPriv->allocType,
                &objPriv->openContext);
    if (LwSciError_Success != sciErr) {
        LWSCI_ERR_STR("LwSciBufModuleGetAllocIfaceOpenContext failed\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    sciErr = LwSciBufObjGetAllocContextParams(reconciledAttrList,
                &allocContextParams);
    if (LwSciError_Success != sciErr) {
        LWSCI_ERR_STR("LwSciBufObjGetAllocContextParams failed\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    /* get allocation context for alloc interface*/
    sciErr = LwSciBufAllocIfaceGetAllocContext(objPriv->allocType,
                allocContextParams, objPriv->openContext,
                &objPriv->allocContext);
    if (LwSciError_Success != sciErr) {
        LWSCI_ERR_STR("LwSciBufAllocIfaceGetAllocContext failed\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

ret:
    LWSCI_FNEXIT("");
    return sciErr;
}

LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 8_13), "LwSciBuf-ADV-MISRAC2012-012")
static LwSciError LwSciBufAllocIfaceValAndGetDevHandle(
    LwSciBufAttrList reconciledAttrList,
    LwSciBufAllocIfaceVal* allocIfaceVal,
    LwSciBufObjPriv** objPtr)
{
    LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 8_13))
    LwSciError sciErr = LwSciError_Success;
    LwSciBufObjPriv* objPriv = NULL;
    LwSciBufDev devHandle = NULL;
    LwSciBufModule module = NULL;

    LWSCI_FNENTRY("");

    objPriv = *objPtr;
    LwSciBufObjAllocValToAllocIfaceAllocVal(objPriv->allocVal,
                allocIfaceVal);

    sciErr = LwSciBufGetModuleAndAllocAttrs(reconciledAttrList, &module, objPtr);
    if (LwSciError_Success != sciErr) {
        LWSCI_ERR_STR("LwSciBufGetModuleAndAllocAttrs failed\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    sciErr = LwSciBufModuleGetDevHandle(module, &devHandle);
    if (LwSciError_Success != sciErr) {
        LWSCI_ERR_STR("LwSciModuleGetDevHandle failed\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    /* allocate buffer */
    sciErr = LwSciBufAllocIfaceAlloc(objPriv->allocType,
                objPriv->allocContext, *allocIfaceVal, devHandle,
                &objPriv->rmHandle);
    if (LwSciError_Success != sciErr) {
        LWSCI_ERR_STR("Could not allocate buffer from allocation interface\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }
ret:
    LWSCI_FNEXIT("");
    return sciErr;
}

LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 8_13), "LwSciBuf-ADV-MISRAC2012-012")
static LwSciError LwSciBufMapBuffer(
    LwSciBufAttrList reconciledAttrList,
    LwSciBufObjPriv** objPtr)
{
    LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 8_13))
    LwSciError sciErr = LwSciError_Success;
    LwSciBufObjPriv* objPriv = NULL;

    LWSCI_FNENTRY("");

    objPriv = *objPtr;

    /* check if CPU access is needed for buffer */
    sciErr = LwSciBufObjIsCpuAccessNeeded(reconciledAttrList,
                &objPriv->needCpuAccess);
    if (LwSciError_Success != sciErr) {
        LWSCI_ERR_STR("LwSciBufObjIsCpuAccessNeeded failed\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    /* map buffer into CPU space */
    if (true == objPriv->needCpuAccess ) {
        sciErr = LwSciBufAllocIfaceMemMap(objPriv->allocType,
                    objPriv->allocContext, objPriv->rmHandle, objPriv->offset,
                    objPriv->allocVal.size, objPriv->accessPerm, &objPriv->ptr);
        if (LwSciError_Success != sciErr) {
            LWSCI_ERR_STR("Could not map buffer into CPU space\n");
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
            goto ret;
        }
    }

ret:
    LWSCI_FNEXIT("");
    return sciErr;
}

LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 8_13), "LwSciBuf-ADV-MISRAC2012-012")
static LwSciError LwSciBufAllocDupHandleAndMapBuffer(
    LwSciBufAttrList reconciledAttrList,
    const LwSciBufRmHandle memHandle,
    bool dupHandle,
#if (LW_IS_SAFETY == 0) && (LW_L4T == 0)
    bool isRemoteObject,
#endif
    LwSciBufObjPriv** objPtr)
{
    LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 8_13))
    LwSciError sciErr = LwSciError_Success;
    LwSciBufObjPriv* objPriv = NULL;

    LWSCI_FNENTRY("");

    objPriv = *objPtr;

#if (LW_IS_SAFETY == 0) && (LW_L4T == 0)
    if (false == isRemoteObject) {
#endif
        if (true == dupHandle) {
            /* take reference to downstream memhandle */
            sciErr = LwSciBufAllocIfaceDupHandle(objPriv->allocType,
                        objPriv->allocContext, objPriv->accessPerm,
                        memHandle, &objPriv->rmHandle);
            if (LwSciError_Success != sciErr) {
                LWSCI_ERR_STR("LwSciBufAllocIfaceDupHandle failed\n");
                LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
                goto ret;
            }
        } else {
            objPriv->rmHandle = memHandle;
        }

        /* check if CPU access is needed for buffer */
        sciErr = LwSciBufObjIsCpuAccessNeeded(reconciledAttrList,
                    &objPriv->needCpuAccess);
        if (LwSciError_Success != sciErr) {
            LWSCI_ERR_STR("LwSciBufObjIsCpuAccessNeeded failed\n");
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
                "LwSciBuf-ADV-MISRAC2012-015")
            goto free_handle;
        }

        /* map buffer into CPU space */
        if (true == objPriv->needCpuAccess ) {
            sciErr = LwSciBufAllocIfaceMemMap(objPriv->allocType,
                        objPriv->allocContext, objPriv->rmHandle,
                        objPriv->offset, objPriv->allocVal.size,
                        objPriv->accessPerm, &objPriv->ptr);
            if (LwSciError_Success != sciErr) {
                LWSCI_ERR_STR("Could not map buffer into CPU space\n");
                LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
                    "LwSciBuf-ADV-MISRAC2012-015")
                goto free_handle;
            }
        }
#if (LW_IS_SAFETY == 0) && (LW_L4T == 0)
    }
#endif

    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
    goto ret;

free_handle:
    if (dupHandle) {
        LwSciError sciErr1 = LwSciError_Success;

        sciErr1 = LwSciBufAllocIfaceDeAlloc(objPriv->allocType,
                    objPriv->allocContext, objPriv->rmHandle);

        if (LwSciError_Success != sciErr1) {
            /* If we are in this path, this means we already encountered an error,
             * so there is nothing we can do about this new failure without
             * clobbering the existing error code. */
            LWSCI_ERR_ULONG("Could not deallocate the interface with error: ", (uint64_t)sciErr1);
        }
    }

ret:
    LWSCI_FNEXIT("");
    return sciErr;
}

static LwSciError LwSciBufObjGetAttrsAndModule(
    LwSciBufAttrList reconciledAttrList,
    const LwSciBufRmHandle memHandle,
    uint64_t len,
    uint64_t* bufSize,
#if (LW_IS_SAFETY == 0) && (LW_L4T == 0)
    bool isRemoteObject,
#endif
    LwSciBufObjPriv** objPtr)
{
    LwSciError sciErr = LwSciError_Success;
    LwSciBufObjPriv* objPriv = NULL;
    LwSciBufModule module = NULL;

    LWSCI_FNENTRY("");

    *bufSize = 0UL;
    objPriv = *objPtr;

    sciErr = LwSciBufObjGetAllocAttrs(reconciledAttrList, &objPriv);
    if (LwSciError_Success != sciErr) {
        LWSCI_ERR_STR("LwSciBufObjGetAllocAttrs failed\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    /* get access permissions for the buffer */
    sciErr = LwSciBufObjGetAccessPerm(reconciledAttrList, &objPriv->accessPerm);
    if (LwSciError_Success != sciErr) {
        LWSCI_ERR_STR("LwSciBufObjGetAccessPerm failed\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    /* check that size obtained through reconciled attrlist for child buffer is
     * equal to or greater than that requested through
     * LwSciBufObjCreateFromMemHandle API
     */
    if (len > objPriv->allocVal.size) {
        LWSCI_ERR_STR("len requested for buffer through"
            "LwSciBufObjCreateFromMemHandle API is more than that"
            "obtained through reconciled attrlist\n");
        sciErr = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    sciErr = LwSciBufGetModuleAndAllocAttrs(reconciledAttrList, &module, objPtr);
    if (LwSciError_Success != sciErr) {
        LWSCI_ERR_STR("LwSciBufGetModuleAndAllocAttrs failed\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    /* TODO: Lwrrently lwrm layer ONLY provides ability to retrieve size but
     * does not provide ability to retrieve other allocation parameters such as
     * alignment/coherency/heap type.
     * We need to add an ability to compare allocation values requested through
     * reconciled attrlist with that obtained through downstream
     * lwrm (or equivalent) layer. For now, we are only comparing size
     */
#if (LW_IS_SAFETY == 0) && (LW_L4T == 0)
    if (false == isRemoteObject) {
#endif
        sciErr = LwSciBufAllocIfaceGetSize(objPriv->allocType,
                    objPriv->allocContext, memHandle, bufSize);
        if (LwSciError_Success != sciErr) {
            LWSCI_ERR_STR("LwSciBufAllocIfaceGetSize failed\n");
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
            goto ret;
        }
#if (LW_IS_SAFETY == 0) && (LW_L4T == 0)
    }
#endif

ret:
    LWSCI_FNEXIT("");
    return sciErr;
}

LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 8_7), "LwSciBuf-ADV-MISRAC2012-010")
void LwSciBufObjCleanup(
    LwSciObj* obj)
{
    LwSciError sciErr = LwSciError_Success;
    const LwSciBufObjPriv* objPriv = NULL;

    LWSCI_FNENTRY("");

    if (NULL == obj) {
        LwSciCommonPanic();
    }

    objPriv = LwSciCastObjToBufObjPriv(obj);

    LwSciBufObjValidate(objPriv);

    if (NULL != objPriv->parentObj) {
        /* remove reference to parent object */
        LwSciBufObjFree(objPriv->parentObj);
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto free_attrlist;
    }

#if (LW_IS_SAFETY == 0) && (LW_L4T == 0)
    if (NULL != objPriv->c2cInterfaceTargetHandle.pcieTargetHandle) {
        if (NULL != objPriv->c2cCopyFuncs.bufFreeTargetHandle) {
            int c2cErrorCode = -1;
            c2cErrorCode = objPriv->c2cCopyFuncs.bufFreeTargetHandle(
                            objPriv->c2cInterfaceTargetHandle.pcieTargetHandle);
            if (0 != c2cErrorCode) {
                LWSCI_ERR_STR("Could not free C2c target handle.");
                LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
                    "LwSciBuf-ADV-MISRAC2012-015")
                goto ret;
            }
        } else {
            LwSciCommonPanic();
        }
    }
#endif

#if (LW_IS_SAFETY == 0) && (LW_L4T == 0)
    if (false == objPriv->isRemoteObject) {
#endif
        if (true == objPriv->needCpuAccess ) {
            /* unmap allocation */
            sciErr = LwSciBufAllocIfaceMemUnMap(objPriv->allocType,
                         objPriv->allocContext, objPriv->rmHandle, objPriv->ptr,
                         objPriv->allocVal.size);
            if (LwSciError_Success != sciErr) {
                LWSCI_ERR_STR("buffer unmapping failed\n");
                LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
                goto ret;
            }
        }

        /* de-allocate buffer */
        sciErr = LwSciBufAllocIfaceDeAlloc(objPriv->allocType,
                    objPriv->allocContext, objPriv->rmHandle);
        if (LwSciError_Success != sciErr) {
            LWSCI_ERR_STR("Could not free buffer\n");
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
            goto ret;
        }
#if (LW_IS_SAFETY == 0) && (LW_L4T == 0)
    }
#endif

free_attrlist:
    /* Free reference to reconciled attrList associated with LwSciBufObj */
    LwSciBufAttrListFree(objPriv->attrList);

ret:
    LWSCI_FNEXIT("");
}

/**
 * @brief public functions
 */
LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 8_7), "LwSciBuf-ADV-MISRAC2012-010")
LwSciError LwSciBufObjDup(
    LwSciBufObj bufObj,
    LwSciBufObj* dupObj)
{
    LwSciError sciErr = LwSciError_Success;
    LwSciBufObjPriv* objPriv = NULL;
    LwSciRef* dupObjParam = NULL;

    LWSCI_FNENTRY("");

    /* verify input parameters */
    if ((NULL == bufObj) || (NULL == dupObj)) {
        sciErr = LwSciError_BadParameter;
        LWSCI_ERR_STR("Bad parameter supplied to LwSciBufObjDup\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    /* print input variables */
    LWSCI_INFO("Input: bufObj: %p, dupObj: %p\n", bufObj, dupObj);

    LwSciBufObjGetAndValidate(bufObj, &objPriv);

    sciErr = LwSciCommonDuplicateRef(&bufObj->refHeader, &dupObjParam);
    if (LwSciError_Success != sciErr) {
        LWSCI_ERR_STR("Failed to duplicate LwSciBufObj\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }
    *dupObj = LwSciCastRefToBufObjRefPriv(dupObjParam);

    /* print output variables */
    LWSCI_INFO("Output: *dupObj: %p\n", *dupObj);

ret:
    LWSCI_FNEXIT("");
    return sciErr;
}

#if (LW_IS_SAFETY == 0)
LwSciError LwSciBufObjCreateSubBuf(
    LwSciBufObj parentObj,
    size_t offset,
    size_t len,
    LwSciBufAttrList reconciledAttrList,
    LwSciBufObj* childObj)
{
    LwSciError sciErr = LwSciError_Success;

    LwSciBufObjRefPriv* objRefPriv = NULL;
    LwSciRef* objRefPrivParam = NULL;

    LwSciBufObjPriv* objPriv = NULL;
    LwSciBufObjPriv* parentObjPriv = NULL;
    LwSciObj* objPrivParam = NULL;
    LwSciBufObj dupParentObj = NULL;
    LwSciBufAttrList newReconciledAttrList = NULL;

    /* Parameters used for validation */
    LwSciBufObjAllocTypeParams allocTypeParam = {0};

    size_t tmpAdd = 0U;
    uint8_t addStatus = OP_FAIL;

    LWSCI_FNENTRY("");

    /* verify input parameters */
    if (NULL == parentObj || 0U == len || NULL == reconciledAttrList ||
        NULL == childObj) {
        sciErr = LwSciError_BadParameter;
        LWSCI_ERR_STR("Bad parameter supplied to LwSciBufObjCreateSubBuf\n");
        LWSCI_ERR_ULONG("offset: , ", offset);
        LWSCI_ERR_ULONG("len: \n", len);
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    /* print input parameters */
    LWSCI_INFO("Input: parentObj: %p, offset: %lu, len: %lu, "
        "reconciledAttrList: %p, childObj: %p\n", parentObj, offset, len,
        reconciledAttrList, childObj);

    /* initialize output parameter  */
    *childObj = NULL;

    /* make sure that attrlist list is reconciled */
    sciErr = LwSciBufAttrListCompareReconcileStatus(reconciledAttrList, true);
    if (LwSciError_Success != sciErr) {
        LWSCI_ERR_STR("LwSciBufAttrListCompareReconcileStatus failed\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    /* retrieve parent object from ref */
    LwSciBufObjGetAndValidate(parentObj, &parentObjPriv);

    sciErr = LwSciCommonAllocObjWithRef(sizeof(LwSciBufObjPriv),
                sizeof(LwSciBufObjRefPriv), &objPrivParam,
                &objRefPrivParam);
    if (LwSciError_Success != sciErr) {
        LWSCI_ERR_STR("LwSciCommonAllocObjWithRef failed\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }
    objPriv = LwSciCastObjToBufObjPriv(objPrivParam);
    objRefPriv = LwSciCastRefToBufObjRefPriv(objRefPrivParam);
    objPriv->magic = LW_SCI_BUF_OBJ_MAGIC;

    /* Retrieve allocation interface requested through reconciled attrlist */
    sciErr = LwSciBufObjGetAllocTypeParams(reconciledAttrList, &allocTypeParam);
    if (sciErr!= LwSciError_Success) {
        LWSCI_ERR_STR("LwSciBufObjGetAllocTypeParams failed\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto free_objAndRef;
    }

    /* get allocation interface (type) */
    sciErr = LwSciBufObjGetAllocType(allocTypeParam, &objPriv->allocType);
    if (LwSciError_Success != sciErr) {
        LWSCI_ERR_STR("LwSciBufObjGetAllocType failed\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto free_objAndRef;
    }

    /* compare allocation types of parent and child object */
    if (objPriv->allocType != parentObjPriv->allocType) {
        LWSCI_ERR_STR("allocation interface requested by child object is different than parent object\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto free_objAndRef;
    }

    /* Get attrs needed for buffer allocation from reconciled attrlist */
    sciErr = LwSciBufObjGetAllocVal(reconciledAttrList, &objPriv->allocVal);
    if (LwSciError_Success != sciErr) {
        LWSCI_ERR_STR("LwSciBufObjGetAllocVal failed\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto free_objAndRef;
    }

    /* compare allocation values of parent and child object */
    if (false == LwSciBufObjCompareAllocVal(&parentObjPriv->allocVal, &objPriv->allocVal)) {
        sciErr = LwSciError_BadParameter;
        LWSCI_ERR_STR("allocation values for parent and child buffer do not match\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto free_objAndRef;
    }

    /* check if CPU access is needed for buffer */
    sciErr = LwSciBufObjIsCpuAccessNeeded(reconciledAttrList, &objPriv->needCpuAccess);
    if (LwSciError_Success != sciErr) {
        LWSCI_ERR_STR("LwSciBufObjIsCpuAccessNeeded failed\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto free_objAndRef;
    }

    /* get access permission for the buffer */
    sciErr = LwSciBufObjGetAccessPerm(reconciledAttrList, &objPriv->accessPerm);
    if (LwSciError_Success != sciErr) {
        LWSCI_ERR_STR("LwSciBufObjGetAccessPerm failed\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto free_objAndRef;
    }

    /* check that size and offset specified in LwSciBufObjCreateSubBuf API  are
     * in valid range
     */
    u64Add(len, offset, &tmpAdd, &addStatus);
    if (OP_SUCCESS != addStatus) {
        LWSCI_ERR_STR("Buffer overflow\n");
        sciErr = LwSciError_Overflow;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto free_objAndRef;
    }
    if (tmpAdd > parentObjPriv->allocVal.size) {
        sciErr = LwSciError_BadParameter;
        LWSCI_ERR_ULONG("len  ", len);
        LWSCI_ERR_ULONG("+ offset requested for child buffer: ", offset);
        LWSCI_ERR_UINT("is greater than size of parent buffer: \n", parentObjPriv->allocVal.size);
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto free_objAndRef;
    }

    /* check that size obtained through reconciled attrlist for child buffer is
     * equal to or greater than that requested through LwSciBufObjCreateSubBuf
     * API
     */
    if (len > objPriv->allocVal.size) {
        sciErr = LwSciError_BadParameter;
        LWSCI_ERR_STR("length of child buffer requested through LwSciBufObjCreateSubBuf is more than that obtained through reconciled attrlist\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto free_objAndRef;
    }

    sciErr = LwSciBufAttrListDupRef(reconciledAttrList, &newReconciledAttrList);
    if (LwSciError_Success != sciErr) {
        LWSCI_ERR_STR("LwSciBufAttrListDupRef failed\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto free_objAndRef;
    }
    objPriv->attrList = newReconciledAttrList;
    /* take a reference to parent object */
    sciErr = LwSciBufObjDup(parentObj, &dupParentObj);
    if (LwSciError_Success != sciErr) {
        LWSCI_ERR_STR("LwSciBufObjDup failed\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto unref_attrList;
    }
    objPriv->parentObj = dupParentObj;
    objPriv->rmHandle = parentObjPriv->rmHandle;
#if (LW_L4T == 0)
    objPriv->c2cCopyFuncs = parentObjPriv->c2cCopyFuncs;
    objPriv->c2cInterfaceTargetHandle = parentObjPriv->c2cInterfaceTargetHandle;
    objPriv->isRemoteObject = parentObjPriv->isRemoteObject;
#endif
    objPriv->allocContext = parentObjPriv->allocContext;
    /* LwSciBufObjCreateSubBuf API takes offset of child buffer from parent
     * buffer. Callwlate total offset of child buffer in the rmHandle by adding
     * parent buffer's offset to it
     */
    objPriv->offset = offset + parentObjPriv->offset;

    if (true == objPriv->needCpuAccess ) {
        sciErr = LwSciBufAllocIfaceMemMap(objPriv->allocType,
                    objPriv->allocContext, objPriv->rmHandle, objPriv->offset,
                    objPriv->allocVal.size, objPriv->accessPerm, &objPriv->ptr);
        if (LwSciError_Success != sciErr) {
            LWSCI_ERR_STR("Could not map buffer into CPU space\n");
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
            goto unref_parentObj;
        }
    }

    *childObj = objRefPriv;

    /* print output parameters */
    LWSCI_INFO("Output: *childObj: %p\n", *childObj);
    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
    goto ret;

unref_parentObj:
    LwSciBufObjFree(dupParentObj);

unref_attrList:
    LwSciBufAttrListFree(objPriv->attrList);

free_objAndRef:
    LwSciCommonFreeObjAndRef(&objRefPriv->refHeader, NULL, NULL);
ret:
    LWSCI_FNEXIT("");
    return sciErr;
}
#endif

LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 8_7), "LwSciBuf-ADV-MISRAC2012-010")
LwSciError LwSciBufObjRef(
    LwSciBufObj bufObj)
{
    LwSciError sciErr = LwSciError_Success;
    LwSciBufObjPriv* objPriv = NULL;

    LWSCI_FNENTRY("");

    /* validate input parameters */
    if (NULL == bufObj) {
        sciErr = LwSciError_BadParameter;
        LWSCI_ERR_STR("Bad parameter supplied to LwSciBufObjRef\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    /* print input variables */
    LWSCI_INFO("Input: bufObj: %p\n", bufObj);

    LwSciBufObjGetAndValidate(bufObj, &objPriv);

    sciErr = LwSciCommonIncrAllRefCounts(&bufObj->refHeader);
    if (LwSciError_Success != sciErr) {
        LWSCI_ERR_STR("LwSciCommonIncrAllRefCounts failed\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

ret:
    LWSCI_FNEXIT("");
    return sciErr;
}

LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 8_7), "LwSciBuf-ADV-MISRAC2012-010")
LwSciError LwSciBufAttrListReconcileAndObjAlloc(
    const LwSciBufAttrList attrListArray[],
    size_t attrListCount,
    LwSciBufObj* bufObj,
    LwSciBufAttrList* newConflictList)
{
    LwSciError sciErr = LwSciError_Success;
    LwSciBufAttrList reconciledAttrList = NULL;

    LWSCI_FNENTRY("");

    /* verify input parameters */
    if ((NULL == attrListArray) || (NULL == bufObj)
#if (LW_IS_SAFETY == 0)
        || (NULL == newConflictList)
#endif
        || (0U == attrListCount)) {
        sciErr = LwSciError_BadParameter;
        LWSCI_ERR_STR("Bad parameter supplied to LwSciBufAttrListReconcileAndObjAlloc\n");
        LWSCI_ERR_ULONG("attrLitCount: \n", attrListCount);
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    /* print input parameters */
    LWSCI_INFO("Input: attrListArray: %p, attrListCount:%zu, bufObj: %p, "
        "newConflictList: %p\n", attrListArray, attrListCount, bufObj,
        newConflictList);

    sciErr = LwSciBufAttrListReconcile(attrListArray, attrListCount,
                &reconciledAttrList, newConflictList);
    if (LwSciError_Success != sciErr) {
        LWSCI_ERR_STR("LwSciBufAttrListReconcile failed. Returning conflict list\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    sciErr = LwSciBufObjAlloc(reconciledAttrList, bufObj);
    if (LwSciError_Success != sciErr) {
        LWSCI_ERR_STR("LwSciBufObjAlloc failed\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto free_reconciledAttrList;
    }

    /* print output parameters */
    LWSCI_INFO("Output: *bufObj: %p\n", *bufObj);

free_reconciledAttrList:
    /*
     * Buffer object holds reference to reconciledAttrList.
     * Hence, reconciledAttrList which is local reference can be safely freed.
     */
    LwSciBufAttrListFree(reconciledAttrList);

ret:
    LWSCI_FNEXIT("");
    return sciErr;
}

void LwSciBufObjFree(
    LwSciBufObj bufObj)
{
    LwSciBufObjPriv* objPriv = NULL;

    LWSCI_FNENTRY("");

    /* verify input parameters */
    if (NULL == bufObj) {
        LWSCI_ERR_STR("Bad parameter supplied to LwSciBufObjFree\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    /* print input parameters */
    LWSCI_INFO("Input: bufObj: %p\n", bufObj);

    LwSciBufObjGetAndValidate(bufObj, &objPriv);

    LwSciCommonFreeObjAndRef(&bufObj->refHeader, LwSciBufObjCleanup, NULL);

ret:
    LWSCI_FNEXIT("");
}

LwSciError LwSciBufObjGetAttrList(
    LwSciBufObj bufObj,
    LwSciBufAttrList* bufAttrList)
{
    LwSciError sciErr = LwSciError_Success;
    LwSciBufObjPriv* objPriv = NULL;

    LWSCI_FNENTRY("");

    /* verify input paramters */
    if ((NULL == bufObj) || (NULL == bufAttrList)) {
        sciErr = LwSciError_BadParameter;
        LWSCI_ERR_STR("Bad parameter supplied to LwSciBufObjGetAttrList\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    /* print input parameters */
    LWSCI_INFO("Input: bufObj: %p, bufAttrList: %p\n", bufObj, bufAttrList);

    LwSciBufObjGetAndValidate(bufObj, &objPriv);

    *bufAttrList = objPriv->attrList;

    /* print output parameters */
    LWSCI_INFO("Output: *bufAttrList: %p\n", *bufAttrList);

ret:
    LWSCI_FNEXIT("");
    return sciErr;
}

LwSciError LwSciBufObjGetMemHandle(
    LwSciBufObj bufObj,
    LwSciBufRmHandle* memHandle,
    uint64_t* offset,
    uint64_t* len)
{
    LwSciError sciErr = LwSciError_Success;
    LwSciBufObjPriv* objPriv = NULL;

    LWSCI_FNENTRY("");

    /* verify input parameters */
    if ((NULL == bufObj) || (NULL == memHandle) || (NULL == offset) || (NULL == len)) {
        sciErr = LwSciError_BadParameter;
        LWSCI_ERR_STR("Bad parameter supplied to LwSciBufObjGetMemHandle\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    /* print input parameters */
    LWSCI_INFO("Input: bufObj: %p, memHandle: %p, offset: %p, len: %p\n",
        bufObj, memHandle, offset, len);

    LwSciBufObjGetAndValidate(bufObj, &objPriv);

    *memHandle = objPriv->rmHandle;
    *offset = objPriv->offset;
    *len = objPriv->allocVal.size;

    /* print output parameters */
    LWSCI_INFO("Output: memHandle: %p, offset: %lu, len: %lu\n", memHandle,
        *offset, *len);

ret:
    LWSCI_FNEXIT("");
    return sciErr;
}

LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 8_7), "LwSciBuf-ADV-MISRAC2012-010")
LwSciError LwSciBufObjGetCpuPtr(
    LwSciBufObj bufObj,
    void** ptr)
{
    LwSciError sciErr = LwSciError_Success;
    LwSciBufObjPriv* objPriv = NULL;

    LWSCI_FNENTRY("");

    /* verify input parameters */
    if ((NULL == bufObj) || (NULL == ptr)) {
        sciErr = LwSciError_BadParameter;
        LWSCI_ERR_STR("Bad parameter supplied to LwSciBufObjGetCpuPtr\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    /* print input parameters */
    LWSCI_INFO("Input: buObj: %p, ptr: %p\n", bufObj, ptr);

    LwSciBufObjGetAndValidate(bufObj, &objPriv);

    if ((false == objPriv->needCpuAccess) ||
        (LwSciBufAccessPerm_ReadWrite != objPriv->accessPerm)) {
        sciErr = LwSciError_BadParameter;
        LWSCI_ERR_STR("LwSciBufObj either did not request for CPU access OR does not have read-write permissions to the buffer\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    *ptr = objPriv->ptr;

    /* print output parameters */
    LWSCI_INFO("Output: *ptr: %p\n", *ptr);

ret:
    LWSCI_FNEXIT("");
    return sciErr;
}

LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 8_7), "LwSciBuf-ADV-MISRAC2012-010")
LwSciError LwSciBufObjGetConstCpuPtr(LwSciBufObj bufObj, const void** ptr)
{
    LwSciError sciErr = LwSciError_Success;
    LwSciBufObjPriv* objPriv = NULL;

    LWSCI_FNENTRY("");

    /* verify input parameters */
    if ((NULL == bufObj) || (NULL == ptr)) {
        sciErr = LwSciError_BadParameter;
        LWSCI_ERR_STR("Bad parameter supplied to LwSciBufObjGetConstCpuPtr\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    /* print input parameters */
    LWSCI_INFO("Input: buObj: %p, ptr: %p\n", bufObj, ptr);

    LwSciBufObjGetAndValidate(bufObj, &objPriv);

    if ((false == objPriv->needCpuAccess) ||
        ((LwSciBufAccessPerm_Readonly != objPriv->accessPerm) &&
        (LwSciBufAccessPerm_ReadWrite != objPriv->accessPerm))) {
        sciErr = LwSciError_BadParameter;
        LWSCI_ERR_STR("LwSciBufObj either did not request for CPU access OR does not have atleast read permissions to the buffer\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    *ptr = objPriv->ptr;

    /* print output parameters */
    LWSCI_INFO("Output: *ptr: %p\n", *ptr);

ret:
    LWSCI_FNEXIT("");
    return sciErr;
}

LwSciError LwSciBufObjCreateFromMemHandlePriv(
    const LwSciBufRmHandle memHandle,
    uint64_t offset,
    uint64_t len,
    LwSciBufAttrList reconciledAttrList,
    bool dupHandle,
#if (LW_IS_SAFETY == 0) && (LW_L4T == 0)
    bool isRemoteObject,
    bool dupC2cTargetHandle,
    LwSciC2cCopyFuncs copyFuncs,
    LwSciC2cInterfaceTargetHandle c2cTargetHandle,
#endif
    LwSciBufObj* bufObj)
{
    LwSciError sciErr = LwSciError_Success;
    LwSciBufObjRefPriv* objRefPriv = NULL;
    LwSciRef* objRefPrivParam = NULL;
    LwSciBufObjPriv* objPriv = NULL;
    LwSciObj* objPrivParam = NULL;
    uint64_t bufSize = 0U;
    LwSciBufAttrList newReconciledAttrList = NULL;
    uint64_t tmpAdd = 0U;
    uint8_t status = OP_FAIL;

    LWSCI_FNENTRY("");

    /* verify input parameters */
    if ((0U == len) || (NULL == reconciledAttrList) || (NULL == bufObj)
#if (LW_IS_SAFETY == 0) && (LW_L4T == 0)
        || ((true == dupHandle) && (true == isRemoteObject))
#endif
        ) {
        LWSCI_ERR_STR("Bad parameter supplied to LwSciBufObjCreateFromMemHandlePriv\n");
        LWSCI_ERR_ULONG("offset: ,", offset);
        LWSCI_ERR_ULONG("len: ", len);
        sciErr = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    /* print input parameters */
    LWSCI_INFO("Input: offset: %lu, len: %lu, reconciledAttrList: %p, bufObj: %p\n",
        offset, len, reconciledAttrList, bufObj);

    /* initialize output parameters */
    *bufObj = NULL;

    /* make sure that attrlist is reconciled */
    sciErr = LwSciBufAttrListCompareReconcileStatus(reconciledAttrList, true);
    if (LwSciError_Success != sciErr) {
        LWSCI_ERR_STR("LwSciBufAttrListCompareReconcileStatus failed\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    sciErr = LwSciCommonAllocObjWithRef(sizeof(LwSciBufObjPriv),
                sizeof(LwSciBufObjRefPriv), &objPrivParam,
                &objRefPrivParam);
    if (LwSciError_Success != sciErr) {
        LWSCI_ERR_STR("LwSciCommonAllocObjWithRef failed\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }
    objPriv = LwSciCastObjToBufObjPriv(objPrivParam);
    objRefPriv = LwSciCastRefToBufObjRefPriv(objRefPrivParam);

    objPriv->magic = LW_SCI_BUF_OBJ_MAGIC;
    sciErr = LwSciBufAttrListDupRef(reconciledAttrList, &newReconciledAttrList);
    if (LwSciError_Success != sciErr) {
        LWSCI_ERR_STR("LwSciBufAttrListDupRef failed\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto free_objAndRef;
    }
    objPriv->attrList = newReconciledAttrList;
    objPriv->parentObj = NULL;
#if (LW_IS_SAFETY == 0) && (LW_L4T == 0)
    objPriv->isRemoteObject = isRemoteObject;
    objPriv->c2cCopyFuncs = copyFuncs;

    if (NULL != c2cTargetHandle.pcieTargetHandle) {
        if (true == dupC2cTargetHandle) {
            int c2cErrorCode = -1;
            if (NULL != objPriv->c2cCopyFuncs.bufDupTargetHandle) {
                c2cErrorCode = objPriv->c2cCopyFuncs.bufDupTargetHandle(
                        c2cTargetHandle.pcieTargetHandle,
                        &objPriv->c2cInterfaceTargetHandle.pcieTargetHandle);
                if (0 != c2cErrorCode) {
                    LWSCI_ERR_STR("C2c target handle duplication failed.");
                    sciErr = LwSciError_ResourceError;
                    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
                        "LwSciBuf-ADV-MISRAC2012-015")
                    goto free_objAndRef;
                }
            } else {
                /* We are trying to duplicate the valid target handle but there
                 * is no function pointer to duplicate it? This should not
                 * happen.
                 */
                LwSciCommonPanic();
            }
        } else {
            objPriv->c2cInterfaceTargetHandle = c2cTargetHandle;
        }
    }
#endif

    sciErr = LwSciBufObjGetAttrsAndModule(reconciledAttrList,
                memHandle, len, &bufSize,
#if (LW_IS_SAFETY == 0) && (LW_L4T == 0)
                isRemoteObject,
#endif
                &objPriv);
    if (LwSciError_Success != sciErr) {
        LWSCI_ERR_STR("LwSciBufObjGetAttrsAndModule failed\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto free_objAndRef;
    }

    /* verify that size of the buffer requested through reconciled attrlist is
     * less than actual buffer size
     */
    if ((objPriv->allocVal.size > bufSize)
#if (LW_IS_SAFETY == 0) && (LW_L4T == 0)
        && (false == isRemoteObject)
#endif
        ) {
        sciErr = LwSciError_BadParameter;
        LWSCI_ERR_STR("buffer size obtained after reconciliation is greater than actual size of buffer from memory handle\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto free_objAndRef;
    }

#if (LW_IS_SAFETY == 0) && (LW_L4T == 0)
    if (false == isRemoteObject) {
#endif
        /* verify that len + offset of buffer requested through API is less than
         * buffer size from memhandle
         */
        u64Add(len, offset, &tmpAdd, &status);
        if (OP_SUCCESS != status) {
            LWSCI_ERR_STR("Buffer overflow\n");
            sciErr = LwSciError_Overflow;
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
                "LwSciBuf-ADV-MISRAC2012-015")
            goto free_objAndRef;
        }
        if (tmpAdd > bufSize) {
            sciErr = LwSciError_BadParameter;
            LWSCI_ERR_ULONG("len ", len);
            LWSCI_ERR_ULONG("+ offset requested for buffer: ", offset);
            LWSCI_ERR_ULONG("is greater than size of buffer from memhandle: ",
                bufSize);
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
                "LwSciBuf-ADV-MISRAC2012-015")
            goto free_objAndRef;
        }
#if (LW_IS_SAFETY == 0) && (LW_L4T == 0)
    }
#endif

    objPriv->offset = offset;

    sciErr = LwSciBufAllocDupHandleAndMapBuffer(reconciledAttrList, memHandle,
                dupHandle,
#if (LW_IS_SAFETY == 0) && (LW_L4T == 0)
                isRemoteObject,
#endif
                &objPriv);
    if (LwSciError_Success != sciErr) {
        LWSCI_ERR_STR("LwSciBufAllocDupHandleAndMapBuffer failed\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto free_objAndRef;
    }

    *bufObj = objRefPriv;

    /* print output parameters */
    LWSCI_INFO("Output: *bufObj: %p\n", *bufObj);
    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
    goto ret;

free_objAndRef:
    LwSciCommonFreeObjAndRef(&objRefPriv->refHeader, NULL, NULL);
ret:
    LWSCI_FNEXIT("");
    return sciErr;
}

LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 8_7), "LwSciBuf-ADV-MISRAC2012-010")
LwSciError LwSciBufObjCreateFromMemHandle(
    const LwSciBufRmHandle memHandle,
    uint64_t offset,
    uint64_t len,
    LwSciBufAttrList reconciledAttrList,
    LwSciBufObj* bufObj)
{
    LwSciError err = LwSciError_Success;
#if (LW_IS_SAFETY == 0) && (LW_L4T == 0)
    LwSciC2cCopyFuncs copyFuncs = {};
    LwSciC2cInterfaceTargetHandle c2cTargetHandle = {};
#endif

    LWSCI_FNENTRY("");

#if (LW_IS_SAFETY == 0) && (LW_L4T == 0)
    (void)memset(&copyFuncs, 0, sizeof(copyFuncs));
    (void)memset(&c2cTargetHandle, 0, sizeof(c2cTargetHandle));
#endif

    /* verify input parameters */
    if ((0U == len) || (NULL == reconciledAttrList) || (NULL == bufObj)) {
        LWSCI_ERR_STR("Bad parameter supplied to LwSciBufObjCreateFromMemHandle.");
        LWSCI_ERR_ULONG("offset: ,", offset);
        LWSCI_ERR_ULONG("len: \n", len);
        err = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    err = LwSciBufObjCreateFromMemHandlePriv(memHandle, offset, len,
            reconciledAttrList, true,
#if (LW_IS_SAFETY == 0) && (LW_L4T == 0)
            false, true, copyFuncs, c2cTargetHandle,
#endif
            bufObj);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("LwSciBufObjCreateFromMemHandlePriv failed.");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
            "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

ret:
    LWSCI_FNEXIT("");
    return err;
}

LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 8_7), "LwSciBuf-ADV-MISRAC2012-010")
LwSciError LwSciBufObjAlloc(
    LwSciBufAttrList reconciledAttrList,
    LwSciBufObj* bufObj)
{
    LwSciError sciErr = LwSciError_Success;
    LwSciBufObjRefPriv* objRefPriv = NULL;
    LwSciRef* objRefPrivParam = NULL;
    LwSciBufObjPriv* objPriv = NULL;
    LwSciObj* objPrivParam = NULL;
    LwSciBufAllocIfaceVal allocIfaceVal = {0};
    LwSciBufAttrList newReconciledAttrList = NULL;
#if (LW_IS_SAFETY == 0) && (LW_L4T == 0)
    LwSciC2cCopyFuncs copyFuncs = {};
    LwSciC2cInterfaceTargetHandle c2cTargetHandle = {};
#endif

    LwSciBufAttrValAccessPerm actualPerm = LwSciBufAccessPerm_Ilwalid;
    LwSciBufAttrKeyValuePair keyValPair = {};

    LWSCI_FNENTRY("");

#if (LW_IS_SAFETY == 0) && (LW_L4T == 0)
    (void)memset(&copyFuncs, 0, sizeof(copyFuncs));
    (void)memset(&c2cTargetHandle, 0, sizeof(c2cTargetHandle));
#endif
    (void)memset(&keyValPair, 0, sizeof(keyValPair));

    /* verify input parameters */
    if ((NULL == bufObj) || (NULL == reconciledAttrList)) {
        sciErr = LwSciError_BadParameter;
        LWSCI_ERR_STR("Bad parameter supplied to LwSciBufObjAlloc");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    /* print input parameters */
    LWSCI_INFO("Input: reconciledAttrList: %p, bufObj: %p",
        reconciledAttrList, bufObj);

    /* make sure that attrlist is reconciled */
    sciErr = LwSciBufAttrListCompareReconcileStatus(reconciledAttrList, true);
    if (LwSciError_Success != sciErr) {
        LWSCI_ERR_STR("LwSciBufAttrListCompareReconcileStatus failed");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    sciErr = LwSciCommonAllocObjWithRef(sizeof(LwSciBufObjPriv),
                sizeof(LwSciBufObjRefPriv), &objPrivParam,
                &objRefPrivParam);
    if (LwSciError_Success != sciErr) {
        LWSCI_ERR_STR("LwSciCommonAllocObjWithRef failed");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }
    objPriv = LwSciCastObjToBufObjPriv(objPrivParam);
    objRefPriv = LwSciCastRefToBufObjRefPriv(objRefPrivParam);

    /* initialize out parameter bufObj to NULL */
    *bufObj = NULL;

    objPriv->magic = LW_SCI_BUF_OBJ_MAGIC;


    /* Access permissions for the allocating process will always be RW. */
    objPriv->accessPerm = LwSciBufAccessPerm_ReadWrite;

    /* Check whether this attribute list satisfies the read/write permissions */
    keyValPair.key = LwSciBufGeneralAttrKey_ActualPerm;
    sciErr = LwSciBufAttrListCommonGetAttrs(reconciledAttrList, 0, &keyValPair,
                1, LwSciBufAttrKeyType_Public, true);
    if (LwSciError_Success != sciErr) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto free_objAndRef;
    }
    if (0U == keyValPair.len) {
        // This should always be set since this is a reconciled list
        LwSciCommonPanic();
    }
    actualPerm = *(LwSciBufAttrValAccessPerm const*)keyValPair.value;
    if (LwSciBufAccessPerm_ReadWrite != actualPerm) {
        /* Clone the list such that modification of this list doesn't affect
         * LwSciBufAttrList(s) associated with existing LwSciBufObj(s). */
        sciErr = LwSciBufAttrListClone(reconciledAttrList, &newReconciledAttrList);
        if (LwSciError_Success != sciErr) {
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
            goto free_objAndRef;
        }

        objPriv->attrList = newReconciledAttrList;

        /* set access permission for buffer to RW */
        keyValPair.key = LwSciBufGeneralAttrKey_ActualPerm;
        keyValPair.len = sizeof(LwSciBufAttrValAccessPerm);
        keyValPair.value = &objPriv->accessPerm;
        sciErr = LwSciBufAttrListCommonSetAttrs(newReconciledAttrList, 0,
                &keyValPair, 1, LwSciBufAttrKeyType_Public, true, false);
        if (LwSciError_Success != sciErr) {
            LWSCI_ERR_STR("LwSciBufObj Setting ActualPerm to RW failed");
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
            goto unref_attrList;
        }
    } else {
        // Otherwise we can just dupe the existing reference.
        sciErr = LwSciBufAttrListDupRef(reconciledAttrList, &newReconciledAttrList);
        if (LwSciError_Success != sciErr) {
            LWSCI_ERR_STR("LwSciBufAttrListDupRef failed");
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
            goto free_objAndRef;
        }

        objPriv->attrList = newReconciledAttrList;
    }

    objPriv->parentObj = NULL;
    objPriv->offset = 0;
#if (LW_IS_SAFETY == 0) && (LW_L4T == 0)
    objPriv->c2cCopyFuncs = copyFuncs;
    objPriv->c2cInterfaceTargetHandle = c2cTargetHandle;
    objPriv->isRemoteObject = false;
#endif

    sciErr = LwSciBufObjGetAllocAttrs(newReconciledAttrList, &objPriv);
    if (LwSciError_Success != sciErr) {
        LWSCI_ERR_STR("LwSciBufObjGetAllocAttrs failed");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto unref_attrList;
    }

    /* colwert alloc values from object management layer to
     * alloc abstraction layer
     */
    /* allocate an array for storing allocIface heaps types */
    LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
    allocIfaceVal.heap = LwSciCommonCalloc(objPriv->allocVal.numHeaps,
                            sizeof(LwSciBufAllocIfaceHeapType));
    LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 11_5))
    if (NULL == allocIfaceVal.heap) {
        sciErr = LwSciError_InsufficientMemory;
        LWSCI_ERR_STR("Could not allocate memory for LwSciBufAllocIfaceHeapType");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto unref_attrList;
    }


    sciErr = LwSciBufAllocIfaceValAndGetDevHandle(newReconciledAttrList,
        &allocIfaceVal, &objPriv);
    if (LwSciError_Success != sciErr) {
        LWSCI_ERR_STR("LwSciBufObj Getting attr failed");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto free_allocIfaceValHeap;
    }

    sciErr = LwSciBufMapBuffer(newReconciledAttrList, &objPriv);
    if (LwSciError_Success != sciErr) {
        LWSCI_ERR_STR("LwSciBufObj MapBuffer failed");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto free_rmHandle;
    }

    *bufObj = objRefPriv;

    /* print output parameters */
    LWSCI_INFO("Output: *bufObj: %p", *bufObj);

    /* we are here implies that alloction/buffer mapping was successful.
     * Now, go to free_allocIfaceValHeap and free the memory allocated for
     * colwerting heap types from obj management layer to alloc interface
     * layer
     */
    LwSciCommonFree(allocIfaceVal.heap);
    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
    goto ret;

free_rmHandle:
    if (LwSciError_Success != LwSciBufAllocIfaceDeAlloc(objPriv->allocType, objPriv->allocContext,
                                  objPriv->rmHandle)) {
        LWSCI_ERR_STR("LwSciBufAllocIfaceDeAlloc failed");
    }
free_allocIfaceValHeap:
    LwSciCommonFree(allocIfaceVal.heap);
unref_attrList:
    LwSciBufAttrListFree(objPriv->attrList);
free_objAndRef:
    LwSciCommonFreeObjAndRef(&objRefPriv->refHeader, NULL, NULL);
ret:
    LWSCI_FNEXIT("");
    return sciErr;
}

LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 8_7), "LwSciBuf-ADV-MISRAC2012-010")
LwSciError LwSciBufObjFlushCpuCacheRange(
    LwSciBufObj bufObj,
    uint64_t offset,
    uint64_t len)
{
    LwSciError sciErr = LwSciError_Success;
    LwSciBufObjPriv* objPriv = NULL;
    LwSciBufAttrKeyValuePair genKeyPair;
    bool cpuNeedSwCoherency = false;
    uint8_t* cpuPtr = NULL;
    uint64_t tmpAdd = 0U;
    uint8_t addStatus = OP_FAIL;

    LWSCI_FNENTRY("");

    if ((NULL == bufObj) || (0U == len)) {
        sciErr = LwSciError_BadParameter;
        LWSCI_ERR_STR("Bad parameter supplied to LwSciBufObjFlushCpuCacheRange\n");
        LWSCI_ERR_ULONG("len: \n", len);
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    LwSciBufObjGetAndValidate(bufObj, &objPriv);

    u64Add(len, offset, &tmpAdd, &addStatus);
    if (OP_SUCCESS != addStatus) {
        LWSCI_ERR_STR("Buffer overflow\n");
        sciErr = LwSciError_Overflow;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    if (tmpAdd > objPriv->allocVal.size) {
        sciErr = LwSciError_BadParameter;
        LWSCI_ERR_STR("offset + len is greater thant the size of the buffer\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
    cpuPtr = objPriv->ptr;
    if (NULL == cpuPtr) {
        LWSCI_ERR_STR("Object is not mapped to CPU. Operation not permitted.\n");
        sciErr = LwSciError_NotPermitted;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    genKeyPair.key = LwSciBufGeneralAttrKey_CpuNeedSwCacheCoherency;
    sciErr = LwSciBufAttrListCommonGetAttrs(objPriv->attrList, 0, &genKeyPair,
                1, LwSciBufAttrKeyType_Public, true);
    if (LwSciError_Success != sciErr) {
        LWSCI_ERR_STR("Failed to get Sw Cache coherency flag for the object.\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }
    LwSciCommonMemcpyS(&cpuNeedSwCoherency, sizeof(cpuNeedSwCoherency),
                        genKeyPair.value, genKeyPair.len);

    if (true == cpuNeedSwCoherency) {
        LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 18_4), "LwSciBuf-ADV-MISRAC2012-017")
        sciErr = LwSciBufAllocIfaceCpuCacheFlush(objPriv->allocType,
                        objPriv->allocContext, objPriv->rmHandle,
                        (cpuPtr + offset), len);
        LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 18_4))
        if (LwSciError_Success != sciErr) {
            LWSCI_ERR_STR("Unable to CPU Cache flush");
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
            goto ret;
        }
    } else {
        LWSCI_INFO("SW Cache coherency is not needed.");
        LWSCI_INFO("Skipping CPU Cache flush.\n");
    }

    LWSCI_INFO("Successfully able to flush CPU Cache.\n");

ret:
    LWSCI_FNEXIT("");
    return sciErr;
}

LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 8_7), "LwSciBuf-ADV-MISRAC2012-010")
bool LwSciBufObjAtomicGetAndSetLwMediaFlag(
    LwSciBufObj bufObj,
    uint32_t flagIndex,
    bool newValue)
{
     /* Note: This API is supposed to be implemented with lock
     * free Atomic functions. But atomic functions are supported
     * only in C11 standard. Since LwSciBuf claims support with
     * C99, this API is implemented with mutex locks.
     */
    bool prevValue = false;
    uint32_t mask = 0;
    uint64_t maxIndex = LW_SCI_BUF_LWMEDIA_FLAG_COUNT;

    LWSCI_FNENTRY("");

    /* verify input parameters */
    if ((NULL == bufObj) || (flagIndex >= maxIndex)) {
        LWSCI_ERR_STR("Bad parameter supplied to LwSciBufObjAtomicGetAndSetLwMediaFlag\n");
        LWSCI_ERR_UINT("flagIndex: ", flagIndex);
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    /* print input parameters */
    LWSCI_INFO("Input: bufObj: %p, flagIndex: %u, newValue: %s\n", bufObj,
        flagIndex, newValue ? "true" : "false");

    LwSciCommonRefLock(&bufObj->refHeader);

    mask = (uint32_t)1 << (uint32_t)flagIndex;
    prevValue = ((bufObj->lwMediaFlag & mask) == mask);

    if (true == newValue) {
        bufObj->lwMediaFlag |= mask;
    } else {
        bufObj->lwMediaFlag &= ~mask;
    }

    LwSciCommonRefUnlock(&bufObj->refHeader);

    /* print output values */
    LWSCI_INFO("Output: prevValue: %s\n", prevValue ? "true" : "false");

ret:
    LWSCI_FNEXIT("");
    return prevValue;
}

LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 8_7), "LwSciBuf-ADV-MISRAC2012-010")
LwSciError LwSciBufObjDupWithReducePerm(
    LwSciBufObj bufObj,
    LwSciBufAttrValAccessPerm reducedPerm,
    LwSciBufObj* newBufObj)
{
    LwSciError sciErr = LwSciError_Success;
    LwSciBufObjPriv* objPriv = NULL;
    LwSciBufAttrList reconciledAttrList = NULL;
    LwSciBufAttrList clonedAttrList = NULL;
    LwSciBufAttrValAccessPerm actualPerm = LwSciBufAccessPerm_Ilwalid;

    LWSCI_FNENTRY("");

    if ((NULL == bufObj) || (NULL == newBufObj) ) {
        LWSCI_ERR_STR("Invalid params to dup object with reduced permissions.\n");
        sciErr = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    sciErr = ValidateAccessPerm(reducedPerm);
    if (LwSciError_Success != sciErr) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    /* print input variables */
    LWSCI_INFO("Input: bufObj: %p, newBufObj: %p\n", bufObj, newBufObj);

    LwSciBufObjGetAndValidate(bufObj, &objPriv);

    reconciledAttrList = objPriv->attrList;

    sciErr = LwSciBufObjGetAccessPerm(reconciledAttrList, &actualPerm);
    if (LwSciError_Success != sciErr) {
        LWSCI_ERR_STR("Failed to get actual access permissions from attrlist.\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    if (actualPerm < reducedPerm) {
        sciErr = LwSciError_BadParameter;
        LWSCI_ERR_STR("Object doesn't have enough permissions to duplicate.\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    if (actualPerm != reducedPerm) {
        LwSciBufAttrKeyValuePair keyValPair = {0};

#if (LW_IS_SAFETY == 0) && (LW_L4T == 0)
        if (true == objPriv->isRemoteObject) {
            /* How to reduce permissions of remote LwSciBufObj since there is
             * no backing memory?.
             */
            sciErr = LwSciError_NotSupported;
            LWSCI_ERR_STR("Reducing permissions for remote LwSciBufObj in C2c case is not yet supported.");
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
                "LwSciBuf-ADV-MISRAC2012-015")
            goto ret;
        }
#endif

        sciErr = LwSciBufAttrListClone(reconciledAttrList, &clonedAttrList);
        if (LwSciError_Success != sciErr) {
            LWSCI_ERR_STR("Unable to clone lwscibuf attrlist.\n");
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
            goto ret;
        }

        /* set access permission in the clonedlist to reduced perms */
        keyValPair.key = LwSciBufGeneralAttrKey_ActualPerm;
        keyValPair.len = sizeof(LwSciBufAttrValAccessPerm);
        keyValPair.value = &reducedPerm;

        sciErr = LwSciBufAttrListCommonSetAttrs(clonedAttrList, 0, &keyValPair, 1,
                    LwSciBufAttrKeyType_Public, true, false);
        if (LwSciError_Success != sciErr) {
            LWSCI_ERR_STR("Setting ActualPerm to Cloned Attrlist failed\n");
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
            goto free_clonedlist;
        }

        sciErr = LwSciBufObjCreateFromMemHandlePriv(objPriv->rmHandle,
                    objPriv->offset, objPriv->allocVal.size, clonedAttrList,
                    /* This flag will change to !(objPriv->isRemoteObject)
                     * if/when we start supporting permission reduction for
                     * remote LwSciBufObj.
                     */
                    true,
#if (LW_IS_SAFETY == 0) && (LW_L4T == 0)
                    /* This will be always false for now since we are not
                     * supporting permission reduction for remote LwSciBufObj.
                     */
                    objPriv->isRemoteObject,
                    true,
                    objPriv->c2cCopyFuncs, objPriv->c2cInterfaceTargetHandle,
#endif
                    newBufObj);
        if (LwSciError_Success != sciErr) {
            LWSCI_ERR_STR("Unable to duplicate object with reduced perm.\n");
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
            goto free_clonedlist;
        }
    } else {
        sciErr = LwSciBufObjDup(bufObj, newBufObj);
        if (LwSciError_Success != sciErr) {
            LWSCI_ERR_STR("Not able to duplicate object with reduced perm.\n");
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
            goto ret;
        }
    }

    /* print output variables */
    LWSCI_INFO("Output: *newBufObj: %p\n", *newBufObj);

free_clonedlist:
    /* Note that we fall through and free the cloned attribute list in success
     * case as well because LwSciBufObjCreateFromMemHandlePriv above will take
     * reference to cloned attribute list by default and thus we free the
     * additional reference here.
     */
    if (NULL != clonedAttrList) {
        LwSciBufAttrListFree(clonedAttrList);
    }

ret:
    LWSCI_FNEXIT("");
    return(sciErr);
}

#if (LW_IS_SAFETY == 0) && (LW_L4T == 0)
LwSciError LwSciBufObjSetC2cTargetHandle(
    LwSciBufObj bufObj,
    LwSciC2cInterfaceTargetHandle targetHandle)
{
    LwSciError err = LwSciError_Success;
    LwSciObj* objPrivParam = NULL;
    LwSciBufObjPriv* objPriv = NULL;

    LWSCI_FNENTRY("");

    if (NULL == bufObj) {
        LWSCI_ERR_STR("NULL bufObj supplied to LwSciBufObjSetTargetHandle.");
        err = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
            "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    if (NULL == targetHandle.pcieTargetHandle) {
        LWSCI_ERR_STR("NULL targetHandle supplied to LwSciBufObjSetTargetHandle.");
        LwSciCommonPanic();
    }

    LwSciCommonGetObjFromRef(&bufObj->refHeader, &objPrivParam);
    objPriv = LwSciCastObjToBufObjPriv(objPrivParam);

    LwSciBufObjValidate(objPriv);

    objPriv->c2cInterfaceTargetHandle.pcieTargetHandle =
        targetHandle.pcieTargetHandle;
ret:
    LWSCI_FNEXIT("");
    return err;
}

LwSciError LwSciBufObjSetC2cCopyFunctions(
    LwSciBufObj bufObj,
    LwSciC2cCopyFuncs c2cCopyFuncs)
{
    LwSciError err = LwSciError_Success;
    LwSciObj* objPrivParam = NULL;
    LwSciBufObjPriv* objPriv = NULL;

    LWSCI_FNENTRY("");

    if (NULL == bufObj) {
        LWSCI_ERR_STR("NULL bufObj supplied to LwSciBufObjSetTargetHandle.");
        err = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
            "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    LwSciCommonGetObjFromRef(&bufObj->refHeader, &objPrivParam);
    objPriv = LwSciCastObjToBufObjPriv(objPrivParam);

    LwSciBufObjValidate(objPriv);

    objPriv->c2cCopyFuncs = c2cCopyFuncs;
ret:
    LWSCI_FNEXIT("");
    return err;
}

LwSciError LwSciBufObjGetC2cInterfaceTargetHandle(
    LwSciBufObj bufObj,
    LwSciC2cInterfaceTargetHandle* c2cInterfaceTargetHandle)
{
    LwSciError err = LwSciError_Success;
    LwSciObj* objPrivParam = NULL;
    LwSciBufObjPriv* objPriv = NULL;

    LWSCI_FNENTRY("");

    if ((NULL == bufObj) || (NULL == c2cInterfaceTargetHandle)) {
        LWSCI_ERR_STR("NULL obj passed to LwSciBufObjGetC2cInterfaceTargetHandle.");
        err = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
            "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    LwSciCommonGetObjFromRef(&bufObj->refHeader, &objPrivParam);
    objPriv = LwSciCastObjToBufObjPriv(objPrivParam);

    LwSciBufObjValidate(objPriv);

    *c2cInterfaceTargetHandle = objPriv->c2cInterfaceTargetHandle;

ret:
    LWSCI_FNEXIT("");
    return err;
}
#endif
