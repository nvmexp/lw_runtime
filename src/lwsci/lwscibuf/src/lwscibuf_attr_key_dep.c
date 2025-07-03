/*
 * Copyright (c) 2019-2021, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 */

#include <string.h>
#include "lwscibuf_attr_key_dep.h"
/* NOTE: lwscibuf_attr_reconcile.c includes lwscibuf_attr_key_dep.h so this is
 * cirlwlar dependency but we claim that lwscibuf_attr_key_dep.c/.h is part
 * of 'Attribute Reconcile' unit and thus logically, this is not cirlwlar
 * dependency.
 */
#include "lwscibuf_attr_reconcile.h"

#define LW_SCI_BUF_VALIDATE_TENSOR_PAIRCOUNT 3
#define LW_SCI_BUF_VALIDATE_IMG_PAIRCOUNT 6
#define LW_SCI_BUF_DEP_TENSOR_PAIRCOUNT 3
#define LW_SCI_BUF_CPU_DEP_PAIRCOUNT 1
#define LW_SCI_BUF_GPU_ID_DEP_PUBLIC_PAIRCOUNT 2
#define LW_SCI_BUF_GPU_CACHE_DEP_PUBLIC_PAIRCOUNT 2
#define LW_SCI_BUF_GPU_SW_COHER_DEP_PUBLIC_PAIRCOUNT 2
#define LW_SCI_BUF_GPU_SW_COHER_DEP_INTERNAL_PAIRCOUNT 2
#define LW_SCI_BUF_GPU_COMP_DEP_PUBLIC_PAIRCOUNT 4
#define LW_SCI_BUF_GPU_COMP_DEP_INTERNAL_PAIRCOUNT 1
#define LW_SCI_BUF_VIDMEM_GPU_DEP_PUBLIC_PAIRCOUNT 2

static bool LwSciBufMemDomainContains(
    const LwSciBufMemDomain* memDomainArr,
    size_t memDomainArrLen,
    LwSciBufMemDomain memDomain)
{
    bool result = false;
    size_t i = 0U;

    for (i = 0U; i < memDomainArrLen; ++i) {
        if (memDomainArr[i] == memDomain) {
            result = true;
            break;
        }
    }

    return result;
}

static LwSciError LwSciBufSetMemDomainPrivKey(
    LwSciBufAttrList attrList)
{
    LwSciError err = LwSciError_Success;

    LwSciBufAttrKeyValuePair keyValPair;
    LwSciBufInternalAttrKeyValuePair intKeyValPair;
    LwSciBufPrivateAttrKeyValuePair pvtKeyValPair;
    LwSciBufMemDomain memDomailwal = LwSciBufMemDomain_UpperBound;
    uint64_t vidMemIdLen = 0U;

    const LwSciBufMemDomain* memDomainArr = NULL;
    size_t memDomainArrLen = 0U;

    LWSCI_FNENTRY("");

    LWSCI_INFO("Input : attrList %p", attrList);

    (void)memset(&keyValPair, 0x0, sizeof(keyValPair));
    (void)memset(&intKeyValPair, 0x0, sizeof(intKeyValPair));

    intKeyValPair.key = LwSciBufInternalGeneralAttrKey_MemDomainArray;
    err = LwSciBufAttrListCommonGetAttrs(attrList, 0, &intKeyValPair, 1,
            LwSciBufAttrKeyType_Internal, true);
    if (err != LwSciError_Success) {
        LWSCI_ERR_STR("Failed to get LwSciBufInternalGeneralAttrKey_MemDomainArray key");
        goto ret;
    }

    /* if LwSciBufGeneralAttrKey_VidMem_GpuId key is set, override memdomain
     * to vidmem
     */
    keyValPair.key = LwSciBufGeneralAttrKey_VidMem_GpuId;
    err = LwSciBufAttrListCommonGetAttrs(attrList, 0, &keyValPair, 1,
            LwSciBufAttrKeyType_Public, true);
    if (err != LwSciError_Success) {
        LWSCI_ERR_STR("Failed to get VidMem_GpuId key");
        goto ret;
    }
    vidMemIdLen = (uint64_t)keyValPair.len;

    if (vidMemIdLen != 0U) {
#if (LW_IS_SAFETY == 0)
        LWSCI_INFO("LwSciBufGeneralAttrKey_VidMem_GpuId is set");
#else
        err = LwSciError_NotSupported;
        LWSCI_ERR_STR("LwSciBufGeneralAttrKey_VidMem_GpuId is set, which indicates user wants to allocate from VidMem which is not supported in this build type.");
        goto ret;
#endif
    }

    memDomainArr = intKeyValPair.value;
    memDomainArrLen = intKeyValPair.len / sizeof(LwSciBufMemDomain);

    // TODO: Account for architecture-specifics like whether the memory is
    // accessible via the given engines (or if the memory domain even exists).
#if (LW_IS_SAFETY == 0)
    if ((vidMemIdLen != 0U) &&
        (LwSciBufMemDomainContains(memDomainArr, memDomainArrLen,
            LwSciBufMemDomain_Vidmem) || (memDomainArrLen == 0U))) {
        /* VidMem is chosen if:
         * - LwSciBufGeneralAttrKey_VidMem_GpuId is set, AND
         * - either:
         *   - LwSciBufMemDomain_Vidmem is set, OR
         *   - LwSciBufInternalGeneralAttrKey_MemDomainArray was never set
         */
        memDomailwal = LwSciBufMemDomain_Vidmem;

        LWSCI_INFO("LwSciBufMemDomain_Vidmem selected");
    } else if (LwSciBufMemDomainContains(memDomainArr, memDomainArrLen,
        LwSciBufMemDomain_Cvsram)) {
        memDomailwal = LwSciBufMemDomain_Cvsram;

        LWSCI_INFO("LwSciBufMemDomain_Cvsram selected");
    } else
#endif
    if ((memDomainArrLen == 0U) ||
        LwSciBufMemDomainContains(memDomainArr, memDomainArrLen,
            LwSciBufMemDomain_Sysmem)) {
        /* SysMem is chosen if:
         * - LwSciBufMemDomain_Sysmem is set, OR
         * - LwSciBufInternalGeneralAttrKey_MemDomainArray was never set
         */
        memDomailwal = LwSciBufMemDomain_Sysmem;

        LWSCI_INFO("LwSciBufMemDomain_Sysmem selected");
    } else {
        /* Impossible to satisfy the constraints. */
        err = LwSciError_ReconciliationFailed;

        LWSCI_ERR_STR("Could not reconcile LwSciBufInternalGeneralAttrKey_MemDomainArray");
        goto ret;
    }

    // Set Internal Key
    intKeyValPair.key = LwSciBufInternalGeneralAttrKey_MemDomainArray;
    intKeyValPair.value = &memDomailwal;
    intKeyValPair.len = sizeof(memDomailwal);
    err = LwSciBufAttrListCommonSetAttrs(attrList, 0U, &intKeyValPair, 1U,
            LwSciBufAttrKeyType_Internal, true, false);
    if (err != LwSciError_Success) {
        LWSCI_ERR_STR("Failed to set Internal Memory Domain key");
        goto ret;
    }

    // Set Private Key
    pvtKeyValPair.key = LwSciBufPrivateAttrKey_MemDomain;
    pvtKeyValPair.value = &memDomailwal;
    pvtKeyValPair.len = sizeof(LwSciBufMemDomain);
    err = LwSciBufAttrListCommonSetAttrs(attrList, 0U, &pvtKeyValPair, 1U,
            LwSciBufAttrKeyType_Private, true, false);
    if (err != LwSciError_Success) {
        LWSCI_ERR_STR("Failed to set Private Memory Domain key");
        goto ret;
    }

ret:
    LWSCI_FNEXIT("");
    return err;
}

static LwSciError LwSciBufAttrListHasDlaEngine(
    LwSciBufAttrList attrList,
    bool* hasDlaEngine)
{
    LwSciError err = LwSciError_Success;
    size_t arrayCount = 0U;
    const LwSciBufHwEngine* engineArray = NULL;
    LwSciBufInternalAttrKeyValuePair keyValPair = {0};

    LWSCI_FNENTRY("");

    LWSCI_INFO("Input: attrList ptr: %p, hasDlaEngine ptr: %p\n", attrList,
        hasDlaEngine);

    /* initialize output*/
    *hasDlaEngine = false;

    keyValPair.key = LwSciBufInternalGeneralAttrKey_EngineArray;
    err = LwSciBufAttrListGetInternalAttrs(attrList, &keyValPair, 1);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("LwSciBufAttrListGetInternalAttrs failed\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    arrayCount = keyValPair.len/sizeof(LwSciBufHwEngine);
    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
    engineArray = (const LwSciBufHwEngine*)keyValPair.value;

    err = LwSciBufHasDlaEngine(engineArray, arrayCount, hasDlaEngine);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("LwSciBufHasDlaEngine failed\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    LWSCI_INFO("Output: hasDlaEngine: %s\n", *hasDlaEngine ? "true" : "false");

ret:
    LWSCI_FNEXIT("");
    return err;
}

static LwSciError LwSciBufValidateImageAttr(
    LwSciBufAttrList attrList)
{
    LwSciError err = LwSciError_Success;
    LwSciBufAttrKeyValuePair imageKeyValPair[LW_SCI_BUF_VALIDATE_IMG_PAIRCOUNT];
    LwSciBufAttrValImageLayoutType imageLayout = LwSciBufImage_BlockLinearType;
    uint32_t imagePlaneCount = 0U;
    uint64_t padding = 0U;
    uint32_t i = 0U;

    LWSCI_FNENTRY("");

    (void)memset(imageKeyValPair, 0x0, sizeof(imageKeyValPair));

    imageKeyValPair[0].key = LwSciBufImageAttrKey_PlaneCount;
    imageKeyValPair[1].key = LwSciBufImageAttrKey_Layout;
    imageKeyValPair[2].key = LwSciBufImageAttrKey_TopPadding;
    imageKeyValPair[3].key = LwSciBufImageAttrKey_BottomPadding;
    imageKeyValPair[4].key = LwSciBufImageAttrKey_LeftPadding;
    imageKeyValPair[5].key = LwSciBufImageAttrKey_RightPadding;
    err = LwSciBufAttrListCommonGetAttrs(attrList, 0, imageKeyValPair,
            LW_SCI_BUF_VALIDATE_IMG_PAIRCOUNT, LwSciBufAttrKeyType_Public, true);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("LwSciBufAttrListCommonGetAttrs failed\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    /* For image/tensor reconciliation, we only support RGB images. Check the
     * planecount supplied via key LwSciBufImageAttrKey_PlaneCount to be 1
     */
    if (0U != imageKeyValPair[i].len) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
        imagePlaneCount = *(const uint32_t*)imageKeyValPair[i].value;
        ++i;
        if (1U != imagePlaneCount) {
            LWSCI_ERR_STR("Image/Tensor reconciliation for YUV images is not supported yet\n");
            err = LwSciError_ReconciliationFailed;
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
            goto ret;
        }
    }

    /* For image/tensor reconciliation, only pitchlinear images are supported
     * for now
     */
    if (0U != imageKeyValPair[i].len) {
        LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
        imageLayout =
            *(const LwSciBufAttrValImageLayoutType*)imageKeyValPair[i].value;
        LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 11_5))
        ++i;
        if (LwSciBufImage_PitchLinearType != imageLayout) {
            LWSCI_ERR_STR("Only pitchlinear layout is supported for image/tensor reconciliation\n");
            err = LwSciError_ReconciliationFailed;
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
            goto ret;
        }
    }

    /* Check that padding is not specified for image */
    for (; i < (uint32_t)LW_SCI_BUF_VALIDATE_IMG_PAIRCOUNT; i++) {
        if (0U != imageKeyValPair[i].len) {
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
            padding = *(const uint64_t*)imageKeyValPair[i].value;
            if (0U != padding) {
                LWSCI_ERR_STR("Image padding is not supported for buf allocation ilwolving image and tensor datatype\n");
                err = LwSciError_ReconciliationFailed;
                LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
                goto ret;
            }
        }
    }

ret:
    LWSCI_FNEXIT("");
    return err;
}

static inline LwSciError LwSciBufValidateTensorDataTypeWithColorFmt(
    LwSciBufAttrValColorFmt tensorColorFmt,
    LwSciBufAttrValDataType tensorDataType)
{
    LwSciError err = LwSciError_Success;

    LWSCI_FNENTRY("");

    if (((LwSciColor_A8B8G8R8 == tensorColorFmt) &&
            (LwSciDataType_Uint8 != tensorDataType)) ||
            ((LwSciColor_Float_A16B16G16R16 == tensorColorFmt) &&
            (LwSciDataType_Float16 != tensorDataType))) {
        LWSCI_ERR_STR("Tensor datatype and colorformat mismatch\n");
        LWSCI_ERR_UINT("Tensor datatype: \n", (uint32_t)tensorDataType);
        LWSCI_ERR_UINT("Tensor color format: \n", (uint32_t)tensorColorFmt);
        err = LwSciError_ReconciliationFailed;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

ret:
    LWSCI_FNEXIT("");
    return err;
}

static LwSciError LwSciBufValidateTensorAttr(
    LwSciBufAttrList attrList)
{
    LwSciError err = LwSciError_Success;
    LwSciBufAttrKeyValuePair tensorKeyValPair[LW_SCI_BUF_VALIDATE_TENSOR_PAIRCOUNT];
    LwSciBufAttrValColorFmt tensorColorFmt = LwSciColor_UpperBound;
    LwSciBufAttrValDataType tensorDataType = LwSciDataType_UpperBound;
    int32_t tensorDims = 0;

    LWSCI_FNENTRY("");
    (void)memset(tensorKeyValPair, 0x0, sizeof(tensorKeyValPair));

    tensorKeyValPair[0].key = LwSciBufTensorAttrKey_PixelFormat;
    tensorKeyValPair[1].key = LwSciBufTensorAttrKey_DataType;
    tensorKeyValPair[2].key = LwSciBufTensorAttrKey_NumDims;
    err = LwSciBufAttrListCommonGetAttrs(attrList, 0, tensorKeyValPair,
            LW_SCI_BUF_VALIDATE_TENSOR_PAIRCOUNT, LwSciBufAttrKeyType_Public, true);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("LwSciBufAttrListCommonGetAttrs failed\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    /* For image/tensor reonciliation, only LwSciColor_Float_A16B16G16R16 and
     * LwSciColor_A8B8G8R8 color formats are supported as of now.
     */
    LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
    tensorColorFmt =
                *(const LwSciBufAttrValColorFmt*)tensorKeyValPair[0].value;
    LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 11_5))

    if ((LwSciColor_Float_A16B16G16R16 != tensorColorFmt) &&
        (LwSciColor_A8B8G8R8 != tensorColorFmt)) {
        LWSCI_ERR_STR("Tensor datatype only supports LwSciColor_Float_A16B16G16R16 and LwSciColor_A8B8G8R8 color formats\n");
        err = LwSciError_ReconciliationFailed;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    /* verify that correct tensor datatype is set for given tensor colorfmt */
    LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
    tensorDataType =
                *(const LwSciBufAttrValDataType*)tensorKeyValPair[1].value;
    LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 11_5))

    err = LwSciBufValidateTensorDataTypeWithColorFmt(tensorColorFmt,
            tensorDataType);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("Failed to validate tensor data type and color format\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    /* Image/Tensor interop only supports 4 dimensions as of now */
    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
    tensorDims = *(const int32_t*)tensorKeyValPair[2].value;
    if (NUM_TENSOR_DIMS != tensorDims) {
        LWSCI_ERR_STR("Image/Tensor reconciliation only supports 4 dimensions yet\n");
        err = LwSciError_ReconciliationFailed;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

ret:
    LWSCI_FNEXIT("");
    return err;
}

LwSciError LwSciBufAttrListImageTensorKeyDependency(
    LwSciBufAttrList attrList)
{
    LwSciError err = LwSciError_Success;
    bool hasDlaEngine = false;

    LWSCI_FNENTRY("");

    err = LwSciBufAttrListValidate(attrList);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("attrList is invalid.\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    LWSCI_INFO("Input: attrList ptr: %p\n", attrList);

    err = LwSciBufValidateImageAttr(attrList);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("LwSciBufValidateImageAttr failed\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    err = LwSciBufValidateTensorAttr(attrList);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("LwSciBufValidateTensorAttr failed\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    /* We are only supporting tensor/image reconciliation for DLA engine as of
     * now since none of the formats used by TRT on GPU path map to image
     * directly. We should remove this check when TRT starts supporting formats
     * that are directly mapped to images
     */
    err = LwSciBufAttrListHasDlaEngine(attrList, &hasDlaEngine);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("LwSciBufAttrListHasDlaEngine failed\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    if (false == hasDlaEngine) {
        LWSCI_ERR_STR("Image/Tensor reconciliation is only supported for DLA path\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

ret:
    LWSCI_FNEXIT("");
    return err;
}

LwSciError LwSciBufAttrListTensorKeyDependency(
    LwSciBufAttrList attrList)
{
    LwSciError err = LwSciError_Success;
    LwSciBufAttrKeyValuePair tensorKeyValPair[LW_SCI_BUF_DEP_TENSOR_PAIRCOUNT];
    size_t sizePerDimValueArrayLen = 0UL;
    uint32_t numDims= 0U;
    bool isAlignmentKeyMandatory = false;

    LWSCI_FNENTRY("");

    err = LwSciBufAttrListValidate(attrList);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("attrList is invalid.\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    (void)memset(tensorKeyValPair, 0x0, sizeof(tensorKeyValPair));

    tensorKeyValPair[0].key = LwSciBufTensorAttrKey_NumDims;
    tensorKeyValPair[1].key = LwSciBufTensorAttrKey_SizePerDim;
    tensorKeyValPair[2].key = LwSciBufTensorAttrKey_AlignmentPerDim;

    err = LwSciBufAttrListCommonGetAttrs(attrList, 0, tensorKeyValPair,
            LW_SCI_BUF_DEP_TENSOR_PAIRCOUNT, LwSciBufAttrKeyType_Public, true);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("LwSciBufAttrListCommonGetAttrs failed\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
    numDims =
        *(const LW_SCI_BUF_TENSKEYTYPE_NUMDIMS *)tensorKeyValPair[0].value;
    LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 11_5))
    sizePerDimValueArrayLen =
        tensorKeyValPair[1].len/sizeof(uint64_t);

    /* LwSciBufTensorAttrKey_SizePerDim key takes value in terms of array.
     * Check that number of elements in value array are NOT less than value
     * provided for LwSciBufTensorAttrKey_NumDims key. It is okay if elements
     * are more than value provided for LwSciBufTensorAttrKey_NumDims key. In
     * that case, we just ignore the additional values.
     */
    if (sizePerDimValueArrayLen < (size_t)numDims) {
        LWSCI_ERR_STR("number of elements in value array of LwSciBufTensorAttrKey_SizePerDim key are less than value for LwSciBufTensorAttrKey_NumDims key\n");
        LWSCI_ERR_UINT("LwSciBufTensorAttrKey_NumDims: \n", numDims);
        LWSCI_ERR_ULONG("number of elements in value array in LwSciBufTensorAttrKey_SizePerDim: \n",
            sizePerDimValueArrayLen);
        err = LwSciError_ReconciliationFailed;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    err = LwSciBufReconcileCheckingNeeded(attrList,
            (uint32_t)LwSciBufTensorAttrKey_AlignmentPerDim, &isAlignmentKeyMandatory);

    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("LwSciBufReconcileCheckingNeeded failed for LwSciBufTensorAttrKey_AlignmentPerDim key\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    if (true == isAlignmentKeyMandatory) {
        size_t alignPerDimValueArrayLen = 0UL;

        alignPerDimValueArrayLen =
            tensorKeyValPair[2].len/sizeof(uint32_t);

        /* LwSciBufTensorAttrKey_AlignmentPerDim key takes value in terms of
         * array. Check that number of elements in value array are NOT less than
         * value provided for LwSciBufTensorAttrKey_NumDims key. It is okay if
         * elements are more than value provided for
         * LwSciBufTensorAttrKey_NumDims key. In that case, we just ignore the
         * additional values.
         */
        if (alignPerDimValueArrayLen < (size_t)numDims) {
            LWSCI_ERR_STR("number of elements in value array of LwSciBufTensorAttrKey_AlignmentPerDim key are less than value for LwSciBufTensorAttrKey_NumDims key\n");
            LWSCI_ERR_UINT("LwSciBufTensorAttrKey_NumDims: \n", numDims);
            LWSCI_ERR_ULONG("number of elements in value array in LwSciBufTensorAttrKey_AlignmentPerDim: \n",
                alignPerDimValueArrayLen);
            err = LwSciError_ReconciliationFailed;
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
            goto ret;
        }
    }

ret:
    LWSCI_FNEXIT("");
    return err;
}

LwSciError LwSciBufAttrListGpuCompressionDependency(
    LwSciBufAttrList attrList,
    bool localPeer,
    LwSciIpcEndpoint ipcEndpoint)
{
    LwSciError err = LwSciError_Success;
    LwSciBufAttrKeyValuePair keyValPair = {};
    LwSciBufAttrKeyValuePair
        publicKeyValPair[LW_SCI_BUF_GPU_COMP_DEP_PUBLIC_PAIRCOUNT] = {};
    LwSciBufAttrKeyValuePair opKeyValPair = {};
    LwSciBufInternalAttrKeyValuePair
        intKeyValPair[LW_SCI_BUF_GPU_COMP_DEP_INTERNAL_PAIRCOUNT] = {};
    const LwSciRmGpuId* gpuIdPtr = NULL;
    const LwSciBufAttrValGpuCompression* gpuCompPtr = NULL;
    LwSciBufAttrValGpuCompression gpuCompArray[LW_SCI_BUF_MAX_GPUS] = {};
    size_t numGpuIds = 0U;
    size_t numGpuComp = 0U;
    size_t numEngines = 0U;
    size_t index1 = 0U;
    size_t index2 = 0U;
    LwSciBufAttrValImageLayoutType layout = LwSciBufImage_PitchLinearType;
    bool isAggregateCompressionAllowed = false;
    bool needCpuAccessPeer = false;
    bool needCpuAccessReconciler = false;
    bool needCpuAccessFinal = false;
    LwSciBufAttrList clonedList = NULL;

    LWSCI_FNENTRY("");

    (void)memset(gpuCompArray, 0x0, sizeof(gpuCompArray));

    publicKeyValPair[0].key = LwSciBufGeneralAttrKey_GpuId;
    publicKeyValPair[1].key = LwSciBufGeneralAttrKey_EnableGpuCompression;
    publicKeyValPair[2].key = LwSciBufGeneralAttrKey_NeedCpuAccess;
    publicKeyValPair[3].key = LwSciBufImageAttrKey_Layout;

    err = LwSciBufAttrListCommonGetAttrs(attrList, 0, publicKeyValPair,
            LW_SCI_BUF_GPU_COMP_DEP_PUBLIC_PAIRCOUNT,
            LwSciBufAttrKeyType_Public, true);
    if (err != LwSciError_Success) {
        LWSCI_ERR_STR("LwSciBufAttrListCommonGetAttrs() failed.");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
            "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    numGpuIds = publicKeyValPair[0].len / sizeof(LwSciRmGpuId);
    gpuIdPtr = (const LwSciRmGpuId*)publicKeyValPair[0].value;

    numGpuComp = publicKeyValPair[1].len /
                    sizeof(LwSciBufAttrValGpuCompression);
    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
    gpuCompPtr =
        (const LwSciBufAttrValGpuCompression*)publicKeyValPair[1].value;

    if ((0U == numGpuIds) && (0U != numGpuComp)) {
        err = LwSciError_ReconciliationFailed;
        LWSCI_ERR_STR("GPU compression can only be requested for GPU ID set in LwSciBufGeneralAttrKey_GpuId.");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
            "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    } else if (numGpuIds == 0U) {
        /* GPU IDs not specified in both LwSciBufGeneralAttrKey_GpuId and
         * LwSciBufGeneralAttrKey_EnableGpuCompression.
         */
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
            "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    } else {
        /* fall through */
    }

    /* Verify that every GPU ID specified in the
     * LwSciBufGeneralAttrKey_EnableGpuCompression is also specified in
     * LwSciBufGeneralAttrKey_GpuId.
     */
    for (index1 = 0U; index1 < numGpuComp; index1++) {
        for (index2 = 0U; index2 < numGpuIds; index2++) {
            if (0 == LwSciCommonMemcmp(&gpuCompPtr[index1].gpuId,
                &gpuIdPtr[index2], sizeof(gpuIdPtr[index2]))) {
                break;
            }
        }

        if (index2 == numGpuIds) {
            err = LwSciError_ReconciliationFailed;
            LWSCI_ERR_STR("GPU ID specified in LwSciBufGeneralAttrKey_EnableGpuCompression does not match with any of the GPU IDs specified in LwSciBufGeneralAttrKey_GpuId.");
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
                "LwSciBuf-ADV-MISRAC2012-015")
            goto ret;
        }
    }

    /* Now, fill the default values for the GPU IDs which are present in
     * LwSciBufGeneralAttrKey_GpuId but for which compression was not requested
     * via LwSciBufGeneralAttrKey_EnableGpuCompression.
     */
    for (index1 = 0U; index1 < numGpuIds; index1++) {
        gpuCompArray[index1].gpuId = gpuIdPtr[index1];

        for (index2 = 0U; index2 < numGpuComp; index2++) {
            if (0 == LwSciCommonMemcmp(&gpuIdPtr[index1],
                &gpuCompPtr[index2].gpuId, sizeof(gpuIdPtr[index1]))) {
                /* GPU IDs matched. Take the value from
                 * LwSciBufGeneralAttrKey_EnableGpuCompression.
                 */
                gpuCompArray[index1].compressionType =
                    gpuCompPtr[index2].compressionType;
                break;
            }
        }

        if (index2 == numGpuComp) {
            /* GPU ID not specified in
             * LwSciBufGeneralAttrKey_EnableGpuCompression. Let's take default
             * value.
             */
            gpuCompArray[index1].compressionType = LwSciBufCompressionType_None;
        }
    }

    /* Query LWPU driver stack to check if the GPU HW allows the requested
     * compression.
     */
    if (0U != publicKeyValPair[3].len) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 11_5),
            "LwSciBuf-ADV-MISRAC2012-014")
        layout =
            *(const LwSciBufAttrValImageLayoutType*)publicKeyValPair[3].value;
    } else {
        /* if LwSciBufImageAttrKey_Layout is not set, it implies that
         * LwSciBufType is not image. Only image datatype can set blocklinear
         * layout. In all other cases, consider the layout as pitchlinear.
         */
        layout = LwSciBufImage_PitchLinearType;
    }

    for (index1 = 0U; index1 < numGpuIds; index1++) {
        bool isBlockLinear = (layout == LwSciBufImage_BlockLinearType);
        bool isCompressible = false;

        /* If user requested compression, check if underlying GPU HW supports
         * it.
         */
        if (LwSciBufCompressionType_None !=
            gpuCompArray[index1].compressionType) {
            err = LwSciBufAttrListPlatformGpuCompressionDependency(attrList,
                    gpuCompArray[index1].gpuId, isBlockLinear, &isCompressible);

            if (LwSciError_Success != err) {
                err = LwSciError_ReconciliationFailed;
                LWSCI_ERR_STR("LwSciBufAttrListPlatformGpuCompDependency failed.");
                LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
                    "LwSciBuf-ADV-MISRAC2012-015")
                goto ret;
            }

            if (false == isCompressible) {
                /* If GPU HW does not support compression then fall back to
                 * incompressible type.
                 */
                gpuCompArray[index1].compressionType =
                    LwSciBufCompressionType_None;
            }
        }
    }

    /* get the engine array */
    intKeyValPair[0].key = LwSciBufInternalGeneralAttrKey_EngineArray;
    err = LwSciBufAttrListCommonGetAttrs(attrList, 0, intKeyValPair,
            LW_SCI_BUF_GPU_COMP_DEP_INTERNAL_PAIRCOUNT,
            LwSciBufAttrKeyType_Internal, true);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("LwSciBufAttrListCommonGetAttrs() failed.");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
            "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    numEngines = intKeyValPair[0].len / sizeof(LwSciBufHwEngine);

    /* check if CPU access is needed. */
    if (publicKeyValPair[2].len != 0U) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 11_5),
            "LwSciBuf-ADV-MISRAC2012-014")
        needCpuAccessReconciler = *(const bool*)publicKeyValPair[2].value;
    }

    /* Note that reading the LwSciBufGeneralAttrKey_NeedCpuAccess from
     * reconciled attribute list will give reconciled value for the reconciler
     * only. We also need to get the value of this attribute for ALL other peers
     * from IPC table and set 'needCpuAccessFinal' to true if any of the peer
     * requested CPU access.
     */
    err = LwSciBufAttrListClone(attrList, &clonedList);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("LwSciBufAttrListClone failed.");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
            "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    /* This gets the reconciled value for all the peers except the reconciler */
    err = LwSciBufAttrListReconcileFromIpcTable(clonedList,
            LwSciBufGeneralAttrKey_NeedCpuAccess, ipcEndpoint, localPeer, true,
            LwSciBufIpcRoute_SocAffinity);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("LwSciBufAttrListReconcileFromIpcTable() failed.");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
            "LwSciBuf-ADV-MISRAC2012-015")
        goto free_clonedList;
    }

    keyValPair.key = LwSciBufGeneralAttrKey_NeedCpuAccess;
    err = LwSciBufAttrListCommonGetAttrs(clonedList, 0, &keyValPair, 1,
            LwSciBufAttrKeyType_Public, true);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("LwSciBufAttrListCommonGetAttrs failed.");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
            "LwSciBuf-ADV-MISRAC2012-015")
        goto free_clonedList;
    }

    if (0U != keyValPair.len) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 11_5),
            "LwSciBuf-ADV-MISRAC2012-014")
        needCpuAccessPeer = *(const bool*)keyValPair.value;
    }

    /* Now, do logical OR with the value of this key reconciled for the
     * reconciler to get the final value.
     */
    needCpuAccessFinal = (needCpuAccessPeer || needCpuAccessReconciler);

    /* LwSciBuf does not allow compression for any of the GPUs in following
     * cases.
     * 1. There are multiple GPUs requesting buffer access. Multiple GPUs can't
     * share comptags and thus, app/UMDs may have to do in-place compression
     * decompression at GPU hand-off boundary. in-place compression
     * decompression does not allow any improvements in perf compared to
     * incompressible and thus, we rather take simple approach of falling back
     * to incompressible kind.
     * 2. If any of non-GPU engines requested to access the buffer. (non-GPU
     * engines can't read compressible buffer).
     * 3. If CPU requested to access the buffer. (CPU can't read compressible
     * buffer).
     */
    isAggregateCompressionAllowed = !((1U < numGpuIds) || (0U < numEngines) ||
                                        (true == needCpuAccessFinal));
    if (false == isAggregateCompressionAllowed) {
        for (index1 = 0U; index1 < numGpuIds; index1++) {
            gpuCompArray[index1].compressionType = LwSciBufCompressionType_None;
        }
    }

    opKeyValPair.key = LwSciBufGeneralAttrKey_EnableGpuCompression;
    opKeyValPair.len = numGpuIds * sizeof(LwSciBufAttrValGpuCompression);
    opKeyValPair.value = gpuCompArray;

    err = LwSciBufAttrListCommonSetAttrs(attrList, 0, &opKeyValPair, 1,
            LwSciBufAttrKeyType_Public, true, false);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("LwSciBufAttrListCommonSetAttrs() failed.");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
            "LwSciBuf-ADV-MISRAC2012-015")
        goto free_clonedList;
    }

free_clonedList:
    LwSciBufAttrListFree(clonedList);

ret:
    LWSCI_FNEXIT("");
    return err;
}

LwSciError LwSciBufAttrListGpuSwCacheCoherDependency(
    LwSciBufAttrList attrList,
    bool localPeer,
    LwSciIpcEndpoint ipcEndpoint)
{
    LwSciError err = LwSciError_Success;
    bool combinedCoherencyCond = false;
    LwSciBufAttrKeyValuePair
        keyValPair[LW_SCI_BUF_GPU_SW_COHER_DEP_PUBLIC_PAIRCOUNT] = {};
    LwSciBufAttrKeyValuePair opKeyValPair = {};
    LwSciBufInternalAttrKeyValuePair
        intKeyValPair[LW_SCI_BUF_GPU_SW_COHER_DEP_INTERNAL_PAIRCOUNT] = {};
    size_t numGpus = 0U;
    const LwSciBufAttrValGpuCache* gpuCache = NULL;
    LwSciBufAttrValGpuCache gpuSwCacheCoherency[LW_SCI_BUF_MAX_GPUS] = {};
    bool needCpuAccessPeer = false;
    bool needCpuAccessReconciler = false;
    bool needCpuAccessFinal = false;
    bool isGpuCacheDisabled = false;
    LwSciBufMemDomain memDomain = LwSciBufMemDomain_UpperBound;
    size_t numEngines = 0U;
    size_t index = 0U;
    LwSciBufAttrList clonedList = NULL;

    LWSCI_FNENTRY("");

    (void)memset(gpuSwCacheCoherency, 0x0, sizeof(gpuSwCacheCoherency));

    /* SW cache coherency is needed for a particular GPU if
     * GPU caching is enabled for that particular GPU and the memory domain is
     * sysmem and any of the following conditions are true:
     * (Note that, we dont worry about vidmem as of now because LwSciBuf does
     * not support peer mem access and thus, LwSciBuf can do successful
     * allocations only if there is single dGPU and memory is allocated from
     * vidmem of that dGPU and there are no other engines requesting access and
     * thus, we dont worry about SW cache coherency at all in vidmem case. This
     * will need to be supported when peer mem (aka multiple dGPUs) start
     * accessing vidmem).
     * a) GPU caching is disabled for at least one GPU.
     * b) CPU access is requested for the buffer.
     * c) LwSciBufInternalGeneralAttrKey_EngineArray contains at least one
     *    engine.
     */
    keyValPair[0].key = LwSciBufGeneralAttrKey_EnableGpuCache;
    keyValPair[1].key = LwSciBufGeneralAttrKey_NeedCpuAccess;

    err = LwSciBufAttrListCommonGetAttrs(attrList, 0, keyValPair,
            LW_SCI_BUF_GPU_SW_COHER_DEP_PUBLIC_PAIRCOUNT,
            LwSciBufAttrKeyType_Public, true);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("LwSciBufAttrListCommonGetAttrs() failed.");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
            "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    numGpus = keyValPair[0].len / sizeof(LwSciBufAttrValGpuCache);
    if (0U == numGpus) {
        /* LwSciBufGeneralAttrKey_EnableGpuCache not set during reconciliation.
         * Thus, no need to set LwSciBufGeneralAttrKey_GpuSwNeedCacheCoherency.
         */
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
            "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
    gpuCache = (const LwSciBufAttrValGpuCache*)keyValPair[0].value;

    if (0U != keyValPair[1].len) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 11_5),
            "LwSciBuf-ADV-MISRAC2012-014")
        needCpuAccessReconciler = *(const bool*)keyValPair[1].value;
    }

    /* Note that reading the LwSciBufGeneralAttrKey_NeedCpuAccess from
     * reconciled attribute list will give reconciled value for the reconciler
     * only. We also need to get the value of this attribute for ALL other peers
     * from IPC table and set 'needCpuAccessFinal' to true if any of the peer
     * requested CPU access.
     */
    err = LwSciBufAttrListClone(attrList, &clonedList);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("LwSciBufAttrListClone failed.");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
            "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    /* This gets the reconciled value for all the peers except the reconciler */
    err = LwSciBufAttrListReconcileFromIpcTable(clonedList,
            LwSciBufGeneralAttrKey_NeedCpuAccess, ipcEndpoint, localPeer, true,
            LwSciBufIpcRoute_SocAffinity);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("LwSciBufAttrListReconcileFromIpcTable() failed.");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
            "LwSciBuf-ADV-MISRAC2012-015")
        goto free_clonedList;
    }

    keyValPair[0].key = LwSciBufGeneralAttrKey_NeedCpuAccess;
    err = LwSciBufAttrListCommonGetAttrs(clonedList, 0, keyValPair, 1,
            LwSciBufAttrKeyType_Public, true);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("LwSciBufAttrListCommonGetAttrs failed.");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
            "LwSciBuf-ADV-MISRAC2012-015")
        goto free_clonedList;
    }

    if (0U != keyValPair[0].len) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 11_5),
            "LwSciBuf-ADV-MISRAC2012-014")
        needCpuAccessPeer = *(const bool*)keyValPair[0].value;
    }

    /* Now, do logical OR with the value of this key reconciled for the
     * reconciler to get the final value.
     */
    needCpuAccessFinal = (needCpuAccessPeer || needCpuAccessReconciler);

    intKeyValPair[0].key = LwSciBufInternalGeneralAttrKey_MemDomainArray;
    intKeyValPair[1].key = LwSciBufInternalGeneralAttrKey_EngineArray;
    err = LwSciBufAttrListCommonGetAttrs(attrList, 0, intKeyValPair,
            LW_SCI_BUF_GPU_SW_COHER_DEP_INTERNAL_PAIRCOUNT,
            LwSciBufAttrKeyType_Internal, true);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("LwSciBufAttrListCommonGetAttrs() failed.");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
            "LwSciBuf-ADV-MISRAC2012-015")
        goto free_clonedList;
    }

    memDomain = *(const LwSciBufMemDomain*)intKeyValPair[0].value;
    numEngines = intKeyValPair[1].len / sizeof(LwSciBufHwEngine);

    /* find out if at least one GPU requested cache to be disabled */
    for (index = 0U; index < numGpus; index++) {
        if (false == gpuCache[index].cacheability) {
            isGpuCacheDisabled = true;
            break;
        }
    }

    combinedCoherencyCond = (memDomain == LwSciBufMemDomain_Sysmem) &&
        ((true == isGpuCacheDisabled) || (true == needCpuAccessFinal) ||
        (0U < numEngines));

#if (LW_IS_SAFETY == 0)
    /* Lwrrently no hardware engine is coherent with VidMem, so we can probably
     * get away with simply checking the number of engines (since the dGPU will
     * be specified in the GPU attribute key). But if in the future there are
     * other engines that are coherent, then we will need to specify each
     * engine's coherency properties. */
    combinedCoherencyCond =
        (combinedCoherencyCond ||
         ((memDomain == LwSciBufMemDomain_Vidmem) && (0U < numEngines)));
#endif

    for (index = 0U; index < numGpus; index++) {
        gpuSwCacheCoherency[index].gpuId = gpuCache[index].gpuId;
        gpuSwCacheCoherency[index].cacheability =
            (gpuCache[index].cacheability && combinedCoherencyCond);
    }

    opKeyValPair.key = LwSciBufGeneralAttrKey_GpuSwNeedCacheCoherency;
    opKeyValPair.len = numGpus * sizeof(LwSciBufAttrValGpuCache);
    opKeyValPair.value = gpuSwCacheCoherency;

    err = LwSciBufAttrListCommonSetAttrs(attrList, 0, &opKeyValPair, 1,
            LwSciBufAttrKeyType_Public, true, false);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("LwSciBufAttrListCommonSetAttrs() failed.");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
            "LwSciBuf-ADV-MISRAC2012-015")
        goto free_clonedList;
    }

free_clonedList:
    LwSciBufAttrListFree(clonedList);

ret:
    LWSCI_FNEXIT("");
    return err;
}

LwSciError LwSciBufAttrListGpuCacheEnableDependency(
    LwSciBufAttrList attrList)
{
    LwSciError err = LwSciError_Success;
    LwSciBufAttrKeyValuePair
        ipKeyValPair[LW_SCI_BUF_GPU_CACHE_DEP_PUBLIC_PAIRCOUNT] = {};
    LwSciBufAttrKeyValuePair opKeyValPair = {};
    const LwSciRmGpuId* gpuIdPtr = NULL;
    LwSciBufAttrValGpuCache gpuCacheVal = {};
    LwSciBufAttrValGpuCache gpuCacheValArray[LW_SCI_BUF_MAX_GPUS] = {};
    size_t numGpuIds = 0U;
    size_t numGpuCacheIds = 0U;
    size_t index = 0U;

    LWSCI_FNENTRY("");

    (void)memset(gpuCacheValArray, 0x0, sizeof(gpuCacheValArray));

    ipKeyValPair[0].key = LwSciBufGeneralAttrKey_GpuId;
    ipKeyValPair[1].key = LwSciBufGeneralAttrKey_EnableGpuCache;
    err = LwSciBufAttrListCommonGetAttrs(attrList, 0, ipKeyValPair,
            LW_SCI_BUF_GPU_CACHE_DEP_PUBLIC_PAIRCOUNT,
            LwSciBufAttrKeyType_Public, true);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("LwSciBufAttrListCommonGetAttrs failed.");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
            "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    numGpuIds = ipKeyValPair[0].len / sizeof(LwSciRmGpuId);
    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 11_5),
        "LwSciBuf-ADV-MISRAC2012-014")
    gpuIdPtr = (const LwSciRmGpuId*)ipKeyValPair[0].value;

    numGpuCacheIds = ipKeyValPair[1].len / sizeof(LwSciBufAttrValGpuCache);

    if ((0U == numGpuIds) && (0U != numGpuCacheIds)) {
        err = LwSciError_ReconciliationFailed;
        LWSCI_ERR_STR("Cacheability control can only be requested for GPU ID set in LwSciBufGeneralAttrKey_GpuId.");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
            "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    } else if (0U == numGpuIds) {
        /* GPU IDs not specified in both LwSciBufGeneralAttrKey_GpuId and
         * LwSciBufGeneralAttrKey_EnableGpuCache. Do nothing.
         */
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
            "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    } else {
        /* fall through */
    }

    if (0U != numGpuCacheIds) {
        /* Validate that the GPU for which cacheability control is requested via
         * LwSciBufGeneralAttrKey_EnableGpuCache is also present as part of
         * LwSciBufGeneralAttrKey_GpuId. Otherwise, it would imply that user is
         * requesting cacheability control for the GPU which is not going to
         * access the buffer.
         */
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 11_5),
            "LwSciBuf-ADV-MISRAC2012-014")
        gpuCacheVal = *(const LwSciBufAttrValGpuCache*)ipKeyValPair[1].value;

        for (index = 0U; index < numGpuIds; index++) {
            if (0 == LwSciCommonMemcmp(&gpuIdPtr[index], &gpuCacheVal.gpuId,
                sizeof(LwSciRmGpuId))) {
                /* GPU IDs match. break from the loop */
                break;
            }
        }

        if (index == numGpuIds) {
            /* The GPU IDs did not match. Fail reconciliation */
            err = LwSciError_ReconciliationFailed;
            LWSCI_ERR_STR("GPU ID specified in LwSciBufGeneralAttrKey_EnableGpuCache does not match with any of the GPU IDs specified in LwSciBufGeneralAttrKey_GpuId.");
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
                "LwSciBuf-ADV-MISRAC2012-015")
            goto ret;
        }
    }

    /* Now, lets plug in default values in LwSciBufGeneralAttrKey_EnableGpuCache
     * attribute for the GPU IDs for which the cacheability control was not
     * specified by user.
     */
    for (index = 0U; index < numGpuIds; index++) {
        /* Fill in the GPU ID */
        gpuCacheValArray[index].gpuId = gpuIdPtr[index];

        if ((0U != numGpuCacheIds) && (0 == LwSciCommonMemcmp(&gpuIdPtr[index],
            &gpuCacheVal.gpuId, sizeof(LwSciRmGpuId)))) {
            /* The reconciliation policy has callwlated the cacheability value.
             * Lets plug in reconciled value for this GPU ID.
             */
            gpuCacheValArray[index].cacheability = gpuCacheVal.cacheability;
            continue;
        }

        err = LwSciBufAttrListPlatformGetDefaultGpuCacheability(attrList,
                gpuIdPtr[index], &gpuCacheValArray[index].cacheability);
        if (LwSciError_Success != err) {
            LWSCI_ERR_STR("LwSciBufAttrListPlatformGetDefaultGpuCacheability failed.");
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
                "LwSciBuf-ADV-MISRAC2012-015")
            goto ret;
        }
    }

    /* set the LwSciBufGeneralAttrKey_EnableGpuCache attribute */
    opKeyValPair.key = LwSciBufGeneralAttrKey_EnableGpuCache;
    opKeyValPair.value = gpuCacheValArray;
    opKeyValPair.len = numGpuIds * sizeof(LwSciBufAttrValGpuCache);
    err = LwSciBufAttrListCommonSetAttrs(attrList, 0, &opKeyValPair, 1,
            LwSciBufAttrKeyType_Public, true, false);

ret:
    LWSCI_FNEXIT("");
    return err;
}

LwSciError LwSciBufAttrListGpuIdDependency(
    LwSciBufAttrList attrList)
{
    LwSciError err = LwSciError_Success;

#if (LW_IS_SAFETY == 0)
    LwSciBufAttrKeyValuePair keyValPair[LW_SCI_BUF_GPU_ID_DEP_PUBLIC_PAIRCOUNT];
    LwSciBufInternalAttrKeyValuePair intKeyValPair;
    LwSciBufMemDomain memDomain = LwSciBufMemDomain_UpperBound;
    const LwSciRmGpuId* gpuIdPtr = NULL;
    const LwSciRmGpuId* vidmemGpuIdPtr = NULL;
    size_t numGpus = 0U;
    size_t vidmemGpuLen = 0U;

    LWSCI_FNENTRY("");

    /* Get the GPU IDs */
    keyValPair[0].key = LwSciBufGeneralAttrKey_GpuId;
    keyValPair[1].key = LwSciBufGeneralAttrKey_VidMem_GpuId;
    err = LwSciBufAttrListCommonGetAttrs(attrList, 0, keyValPair,
            LW_SCI_BUF_GPU_ID_DEP_PUBLIC_PAIRCOUNT, LwSciBufAttrKeyType_Public,
            true);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("LwSciBufAttrListCommonGetAttrs failed.");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
            "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }
    numGpus = keyValPair[0].len / sizeof(LwSciRmGpuId);
    vidmemGpuLen = keyValPair[1].len;

    if ((0U == numGpus) && (0U != vidmemGpuLen)) {
        LWSCI_ERR_STR("No GPUs set in LwSciBufGeneralAttrKey_GpuId to access the vidmem of GPU specified in LwSciBufGeneralAttrKey_VidMem_GpuId.");
        err = LwSciError_ReconciliationFailed;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
            "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    } else if (0U == numGpus) {
        LWSCI_INFO("No GPUs accessing sysmem.");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
            "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    } else {
        /* do nothing, just fall through from here. */
    }

    LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 11_5),
        "LwSciBuf-ADV-MISRAC2012-014")
    gpuIdPtr = (const LwSciRmGpuId*)keyValPair[0].value;
    vidmemGpuIdPtr = (const LwSciRmGpuId*)keyValPair[1].value;
    LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 11_5))

    /* Get the memory domain */
    intKeyValPair.key = LwSciBufInternalGeneralAttrKey_MemDomainArray;
    err = LwSciBufAttrListCommonGetAttrs(attrList, 0, &intKeyValPair, 1,
            LwSciBufAttrKeyType_Internal, true);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("LwSciBufAttrListCommonGetAttrs failed.");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
            "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }
    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 11_5),
        "LwSciBuf-ADV-MISRAC2012-014")
    memDomain = *(const LwSciBufMemDomain*)intKeyValPair.value;

    /* if memory domain = cvsram then reconciliation fails since GPU cannot
     * access cvsram.
     */
    if (LwSciBufMemDomain_Cvsram == memDomain) {
        err = LwSciError_ReconciliationFailed;
        LWSCI_ERR_STR("GPUs cannot access CVSRAM memory.");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
            "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    /* if memory domain = vidmem and there are multiple GPU specified via
     * LwSciBufGeneralAttrKey_GpuId then fail reconciliation since we dont
     * support peer mem case.
     */
    if ((LwSciBufMemDomain_Vidmem == memDomain) && (1U < numGpus)) {
        LWSCI_ERR_STR("Peer mem GPU access not supported.");
        err = LwSciError_ReconciliationFailed;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
            "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    /* if memory domain = vidmem and GPU ID specified in
     * LwSciBufGeneralAttrKey_GpuId != GPU ID specified in
     * LwSciBufGeneralAttrKey_VidMem_GpuId then this constitues a peer mem case
     * which is not supported yet.
     */
    if ((LwSciBufMemDomain_Vidmem == memDomain) &&
        (0 != LwSciCommonMemcmp(gpuIdPtr, vidmemGpuIdPtr,
            sizeof(LwSciRmGpuId)))) {
        LWSCI_ERR_STR("Peer mem GPU access not supported.");
        err = LwSciError_ReconciliationFailed;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
            "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }
#endif //(LW_IS_SAFETY == 0)

    err = LwSciBufAttrListPlatformGpuIdDependency(attrList);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("LwSciBufAttrListPlatformGpuIdDependency() failed.");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
            "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

ret:
    LWSCI_FNEXIT("");
    return err;
}

LwSciError LwSciBufAttrListMemDomainDependency(
    LwSciBufAttrList attrList)
{
    LwSciError err = LwSciError_Success;

    LWSCI_FNENTRY("");

    /* Memory Domain */
    err = LwSciBufSetMemDomainPrivKey(attrList);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("Failed to set Memory Domain key.");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
            "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

ret:
    LWSCI_FNEXIT("");
    return err;
}

LwSciError LwSciBufAttrListHeapTypeDependency(
    LwSciBufAttrList attrList)
{
    LwSciError err = LwSciError_Success;
    LwSciBufHeapType heapType = LwSciBufHeapType_Ilwalid;
    LwSciBufPrivateAttrKeyValuePair pvtKeyValPair;

    LWSCI_FNENTRY("");

    err = LwSciBufAttrListPlatformHeapDependency(attrList, &heapType);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("LwSciBufAttrListPlatformHeapDependency() failed.");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
            "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    pvtKeyValPair.key = LwSciBufPrivateAttrKey_HeapType;
    pvtKeyValPair.len = sizeof(LwSciBufHeapType);
    pvtKeyValPair.value = &heapType;
    err = LwSciBufAttrListCommonSetAttrs(attrList, 0, &pvtKeyValPair, 1,
            LwSciBufAttrKeyType_Private, true, false);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("Failed to set actual permission key");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
            "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

ret:
    LWSCI_FNEXIT("");
    return err;
}

LwSciError LwSciBufAttrListCpuKeysDependency(
    LwSciBufAttrList attrList,
    bool localPeer,
    LwSciIpcEndpoint ipcEndpoint)
{
    LwSciError err = LwSciError_Success;
    LwSciBufAttrKeyValuePair keyValPair[LW_SCI_BUF_CPU_DEP_PAIRCOUNT];
    bool needCpuAccess = false;
    bool cpuCacheEnabled = false;
    bool isIsoEngine = false;
    bool needSwCacheCoherency = false;
    size_t needCpuAccessLen = 0U;
    size_t cpuCacheEnabledLen = 0U;
    LwSciBufAttrList clonedList = NULL;

    LWSCI_FNENTRY("");

    LWSCI_INFO("Input : attrList %p", attrList);

    /* get EnableCpuCache from reconciled list */
    keyValPair[0].key = LwSciBufGeneralAttrKey_EnableCpuCache;
    err = LwSciBufAttrListCommonGetAttrs(attrList, 0, keyValPair,
            LW_SCI_BUF_CPU_DEP_PAIRCOUNT, LwSciBufAttrKeyType_Public, true);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("Failed to get LwSciBufGeneralAttrKey_EnableCpuCache.");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
            "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    cpuCacheEnabledLen = keyValPair[0].len;

    if (0U != cpuCacheEnabledLen) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 11_5),
            "LwSciBuf-ADV-MISRAC2012-014")
        cpuCacheEnabled = *(const bool*)keyValPair[0].value;
    }

    /* With C2c support, the reconciliation of EngineArray is changed such
     * that during reconciliation, LwSciBuf considers all unreconciled lists
     * to reconcile EngineArray since it needs to consider HW constraints
     * of all engines. However, during export, it recallwlates EngineArray such
     * that engines belonging to the remote SoC (assuming C2c case) are
     * returned. Thus, when callwlating the CpuNeedSwCacheCoherency attribute
     * during reconciliation (localPeer = true), we need to recallwlate the
     * EngineArray such that engines from local SoC only are considered,
     * otherwise CpuNeedSwCacheCoherency will be callwlated wrong if we
     * directly consider the EngineArray value from @a attrList (aka reconciled
     * list passed to this function).
     * Also during export (localPeer = false), since CpuNeedSwCacheCoherency
     * comes before EngineArray in the iterator, we need to callwlate
     * EngineArray here to get correct value for CpuNeedSwCacheCoherency. Note
     * that this would go away if we move to new reconciliation framework
     * since we are planning to statically sort the keys based on their
     * dependencies in that framework.
     */
    err = LwSciBufAttrListClone(attrList, &clonedList);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("LwSciBufAttrListClone failed.");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
            "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    err = LwSciBufAttrListReconcileFromIpcTable(clonedList,
            LwSciBufInternalGeneralAttrKey_EngineArray, ipcEndpoint, localPeer,
            true, LwSciBufIpcRoute_SocAffinity);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("LwSciBufAttrListReconcileFromIpcTable() failed.");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
            "LwSciBuf-ADV-MISRAC2012-015")
        goto free_clonedList;
    }

    /* check if any iso engines are present in cloned attrlist */
    err = LwSciBufAttrListIsIsoEngine(clonedList, &isIsoEngine);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("LwSciBufAttrListIsIsoEngine failed.");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
            "LwSciBuf-ADV-MISRAC2012-015")
        goto free_clonedList;
    }

    /* NeedCpuAccess is callwlated with LwSciBufIpcRoute_OwnerAffinity such
     * that during reconciliation, the unreconciled lists of reconciler are
     * considered. During export, unreconciled lists of the peer to which
     * reconciled list is being exported are considered. However, to callwlate
     * CpuNeedSwCacheCoherency, we need to know if any of the peers within
     * Soc have requested for NeedCpuAccess. Reading the @a attrList won't
     * give us that information because NeedCpuAccess is callwlated with
     * LwSciBufIpcRoute_OwnerAffinity by default. Thus, we need to get
     * NeedCpuAccess with LwSciBufIpcRoute_SocAffinity here.
     */
    err = LwSciBufAttrListReconcileFromIpcTable(clonedList,
            LwSciBufGeneralAttrKey_NeedCpuAccess, ipcEndpoint, localPeer,
            true, LwSciBufIpcRoute_SocAffinity);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("LwSciBufAttrListReconcileFromIpcTable() failed.");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
            "LwSciBuf-ADV-MISRAC2012-015")
        goto free_clonedList;
    }

    keyValPair[0].key = LwSciBufGeneralAttrKey_NeedCpuAccess;
    err = LwSciBufAttrListCommonGetAttrs(clonedList, 0, keyValPair,
            LW_SCI_BUF_CPU_DEP_PAIRCOUNT, LwSciBufAttrKeyType_Public, true);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("Failed to get LwSciBufGeneralAttrKey_NeedCpuAccess.");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
            "LwSciBuf-ADV-MISRAC2012-015")
        goto free_clonedList;
    }

    needCpuAccessLen = keyValPair[0].len;
    if (0UL != needCpuAccessLen) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 11_5),
            "LwSciBuf-ADV-MISRAC2012-014")
        needCpuAccess = *(const bool*)keyValPair[0].value;
    }

    /* ISO engines cannot snoop CPU cache and thus if CPU access is needed and
     * CPU caching is enabled and ISO engine is accessing the buffer then user
     * needs to perform SW cache coherence operations.
     */
    needSwCacheCoherency = ((true == needCpuAccess) && (true == cpuCacheEnabled)
                                                    && (true == isIsoEngine));

    /* set CpuNeedSwCacheCoherency */
    keyValPair[0].key = LwSciBufGeneralAttrKey_CpuNeedSwCacheCoherency;
    keyValPair[0].len = sizeof(bool);
    keyValPair[0].value = &needSwCacheCoherency;
    err = LwSciBufAttrListCommonSetAttrs(attrList, 0, keyValPair, 1,
            LwSciBufAttrKeyType_Public, true, false);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("Failed to set CpuNeedSwCacheCoherency key.");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
            "LwSciBuf-ADV-MISRAC2012-015")
        goto free_clonedList;
    }

free_clonedList:
    if (NULL != clonedList) {
        LwSciBufAttrListFree(clonedList);
    }

ret:
    LWSCI_FNEXIT("");
    return err;
}

LwSciError LwSciBufAttrListSetDefaultCpuAccess(
    LwSciBufAttrList attrList)
{
    LwSciError err = LwSciError_Success;
    bool defaultAccess = false;
    LwSciBufAttrKeyValuePair keyValPair = {};
    void* recAddr = NULL;
    LwSciBufAttrStatus* recStatus = NULL;
    uint64_t* recLen = NULL;

    LWSCI_FNENTRY("");

    keyValPair.key = LwSciBufGeneralAttrKey_NeedCpuAccess;
    err = LwSciBufAttrListCommonGetAttrs(attrList, 0, &keyValPair, 1,
            LwSciBufAttrKeyType_Public, true);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("Failed to get LwSciBufGeneralAttrKey_NeedCpuAccess key.");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
            "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    if (0U == keyValPair.len) {
        /* Assign default value. */
        keyValPair.key = LwSciBufGeneralAttrKey_NeedCpuAccess;
        keyValPair.len = sizeof(defaultAccess);
        keyValPair.value = &defaultAccess;
        err = LwSciBufAttrListCommonSetAttrs(attrList, 0, &keyValPair, 1,
                LwSciBufAttrKeyType_Public, true, false);
        if (LwSciError_Success != err) {
            LWSCI_ERR_STR("Failed to set LwSciBufGeneralAttrKey_NeedCpuAccess key.");
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
                "LwSciBuf-ADV-MISRAC2012-015")
            goto ret;
        }

        LwSciBufAttrGetKeyDetail(attrList, 0,
            LwSciBufGeneralAttrKey_NeedCpuAccess, &recAddr, &recStatus,
            &recLen);
        *recStatus = LwSciBufAttrStatus_Reconciled;
    }

ret:
    LWSCI_FNEXIT("");
    return err;
}

LwSciError LwSciBufAttrListSetDefaultRequiredPerm(
    LwSciBufAttrList attrList)
{
    LwSciError err = LwSciError_Success;
    LwSciBufAttrValAccessPerm defaultPerm = LwSciBufAccessPerm_Readonly;
    LwSciBufAttrKeyValuePair keyValPair = {};
    void* recAddr = NULL;
    LwSciBufAttrStatus* recStatus = NULL;
    uint64_t* recLen = NULL;

    LWSCI_FNENTRY("");

    keyValPair.key = LwSciBufGeneralAttrKey_RequiredPerm;
    err = LwSciBufAttrListCommonGetAttrs(attrList, 0, &keyValPair, 1,
            LwSciBufAttrKeyType_Public, true);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("Failed to get LwSciBufGeneralAttrKey_RequiredPerm key.");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
            "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    if (0U == keyValPair.len) {
        /* Assign default value. */
        keyValPair.key = LwSciBufGeneralAttrKey_RequiredPerm;
        keyValPair.len = sizeof(defaultPerm);
        keyValPair.value = &defaultPerm;
        err = LwSciBufAttrListCommonSetAttrs(attrList, 0, &keyValPair, 1,
                LwSciBufAttrKeyType_Public, true, false);
        if (LwSciError_Success != err) {
            LWSCI_ERR_STR("Failed to set LwSciBufGeneralAttrKey_RequiredPerm key.");
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
                "LwSciBuf-ADV-MISRAC2012-015")
            goto ret;
        }

        LwSciBufAttrGetKeyDetail(attrList, 0,
            LwSciBufGeneralAttrKey_RequiredPerm, &recAddr, &recStatus, &recLen);
        *recStatus = LwSciBufAttrStatus_Reconciled;
    }

ret:
    LWSCI_FNEXIT("");
    return err;
}

LwSciError LwSciBufAttrListActualPermDependency(
    LwSciBufAttrList attrList)
{
    LwSciError err = LwSciError_Success;
    LwSciBufAttrKeyValuePair keyValPair = {};
    LwSciBufAttrValAccessPerm accPerm = LwSciBufAccessPerm_Readonly;
    void* recAddr = NULL;
    LwSciBufAttrStatus* recStatus = NULL;
    uint64_t* recLen = NULL;

    LWSCI_FNENTRY("");

    keyValPair.key = LwSciBufGeneralAttrKey_RequiredPerm;
    err = LwSciBufAttrListCommonGetAttrs(attrList, 0, &keyValPair, 1,
            LwSciBufAttrKeyType_Public, true);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("Failed to get LwSciBufGeneralAttrKey_RequiredPerm key.");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
            "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    /* Use reconciled value of requiredPerm attribute set in reconciled
     * attribute list as actualPerm value.
     * If the value is not present, use default value.
     */
    if (0U != keyValPair.len) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 11_5),
            "LwSciBuf-ADV-MISRAC2012-014")
        accPerm = *(const LwSciBufAttrValAccessPerm*)keyValPair.value;
    } else {
        /* This should never happen because we always set requiredPerm (either
         * reconciled or default value).
         */
        LwSciCommonPanic();
    }

    keyValPair.key = LwSciBufGeneralAttrKey_ActualPerm;
    keyValPair.len = sizeof(LwSciBufAttrValAccessPerm);
    keyValPair.value = &accPerm;
    err = LwSciBufAttrListCommonSetAttrs(attrList, 0, &keyValPair, 1,
            LwSciBufAttrKeyType_Public, true, false);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("Failed to set LwSciBufGeneralAttrKey_ActualPerm key.");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
            "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    LwSciBufAttrGetKeyDetail(attrList, 0, LwSciBufGeneralAttrKey_ActualPerm,
        &recAddr, &recStatus, &recLen);
    *recStatus = LwSciBufAttrStatus_Reconciled;

ret:
    LWSCI_FNEXIT("");
    return err;
}

LwSciError LwSciBufAttrListVidmemDependency(
    LwSciBufAttrList attrList)
{
    LwSciError err = LwSciError_Success;
    LwSciBufAttrKeyValuePair
        keyValPair[LW_SCI_BUF_VIDMEM_GPU_DEP_PUBLIC_PAIRCOUNT] = {0};
    bool needCpuAccess = false;

    LWSCI_FNENTRY("");

    keyValPair[0].key = LwSciBufGeneralAttrKey_VidMem_GpuId;
    keyValPair[1].key = LwSciBufGeneralAttrKey_NeedCpuAccess;
    err = LwSciBufAttrListCommonGetAttrs(attrList, 0, keyValPair,
            LW_SCI_BUF_VIDMEM_GPU_DEP_PUBLIC_PAIRCOUNT,
            LwSciBufAttrKeyType_Public, true);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("LwSciBufAttrListCommonGetAttrs failed to get LwSciBufGeneralAttrKey_VidMem_GpuId.");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
            "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    if (0UL == keyValPair[0].len) {
        /* LwSciBufGeneralAttrKey_VidMem_GpuId not set. No need to check for
         * dependency.
         */
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
            "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    /* Get NeedCpuAccess. */
    if (0UL == keyValPair[1].len) {
        /* This should not happen. LwSciBufGeneralAttrKey_NeedCpuAccess is
         * set during reconciliation either by reconciling values from
         * unreconciled lists or by setting default value if none of the
         * unreconciled list has this key set.
         */
        LWSCI_ERR_STR("LwSciBufGeneralAttrKey_NeedCpuAccess not set in reconciled list at this point. This should not happen!!");
        LwSciCommonPanic();
    }

    /* Fail reconciliation if it is set to TRUE since vidmem cannot be CPU
     * mapped.
     */
    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
    needCpuAccess = *(const bool *)keyValPair[1].value;
    if (true == needCpuAccess) {
        LWSCI_ERR_STR("CPU access requested for Vidmem. Vidmem is not CPU mappable.");
        err = LwSciError_ReconciliationFailed;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
            "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    err = LwSciBufAttrListPlatformVidmemDependency(attrList);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("LwSciBufAttrListPlatformVidmemDependency failed.");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
            "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

ret:
    LWSCI_FNEXIT("");
    return err;
}

LwSciError LwSciBufAttrListSetGeneralKeyDependency(
    LwSciBufAttrList attrList,
    bool localPeer,
    LwSciIpcEndpoint ipcEndpoint)
{
    LwSciError err = LwSciError_Success;

    LWSCI_FNENTRY("");

    err = LwSciBufAttrListValidate(attrList);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("attrList is invalid.");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
            "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    err = LwSciBufAttrListSetDefaultRequiredPerm(attrList);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("LwSciBufAttrListSetDefaultRequiredPerm() failed.");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
            "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    err = LwSciBufAttrListSetDefaultCpuAccess(attrList);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("LwSciBufAttrListSetDefaultCpuAccess() failed.");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
            "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    err = LwSciBufAttrListActualPermDependency(attrList);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("LwSciBufAttrListActualPermDependency() failed.");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
            "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    err = LwSciBufAttrListCpuKeysDependency(attrList, localPeer, ipcEndpoint);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("LwSciBufAttrListCpuKeysDependency() failed.");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
            "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    err = LwSciBufAttrListVidmemDependency(attrList);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("LwSciBufAttrListVidmemDependency() failed.");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
            "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    err = LwSciBufAttrListMemDomainDependency(attrList);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("LwSciBufAttrListMemDomainDependency() failed.");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
            "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    /* heap type has dependency on memory domain and hence this function must
     * be called after LwSciBufAttrListMemDomainDependency()
     */
    err = LwSciBufAttrListHeapTypeDependency(attrList);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("LwSciBufAttrListHeapTypeDependency() failed.");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
            "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    /* LwSciBufGeneralAttrKey_GpuId depends on types of GPUs accessing the
     * buffer as well as combination of GPUs and memory domain. Thus, this
     * function should be called after LwSciBufAttrListMemDomainDependency().
     */
    err = LwSciBufAttrListGpuIdDependency(attrList);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("LwSciBufAttrListGpuIdDependency() failed.");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
            "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    /* LwSciBufGeneralAttrKey_EnableGpuCache depends on
     * LwSciBufGeneralAttrKey_GpuId. Thus, this function should be called
     * after LwSciBufAttrListGpuIdDependency().
     */
    err = LwSciBufAttrListGpuCacheEnableDependency(attrList);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("LwSciBufAttrListGpuCacheEnableDependency() failed.");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
            "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    /* LwSciBufGeneralAttrKey_GpuSwNeedCacheCoherency depends on
     * a) LwSciBufGeneralAttrKey_EnableGpuCache
     * b) LwSciBufInternalGeneralAttrKey_MemDomainArray
     * c) LwSciBufGeneralAttrKey_NeedCpuAccess
     * d) LwSciBufInternalGeneralAttrKey_EngineArray
     */
    err = LwSciBufAttrListGpuSwCacheCoherDependency(attrList, localPeer,
            ipcEndpoint);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("LwSciBufAttrListGpuSwCacheCoherDependency() failed.");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
            "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    /* LwSciBufGeneralAttrKey_EnableGpuCompression depends on
     * a) LwSciBufGeneralAttrKey_GpuId
     * b) LwSciBufGeneralAttrKey_NeedCpuAccess
     * c) LwSciBufInternalGeneralAttrKey_EngineArray
     * d) LwSciBufImageAttrKey_Layout (to figure out if compression is requested
     * for blocklinear or pitchlinear surface).
     */
    err = LwSciBufAttrListGpuCompressionDependency(attrList, localPeer,
            ipcEndpoint);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("LwSciBufAttrListGpuCompressionDependency() failed.");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
            "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

ret:
    LWSCI_FNEXIT("");
    return err;
}
