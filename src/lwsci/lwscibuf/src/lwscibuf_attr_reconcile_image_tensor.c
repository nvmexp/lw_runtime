/*
 * Copyright (c) 2018-2021, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 */

#include "lwscibuf_attr_reconcile_image_tensor_priv.h"
#include "lwscibuf_attr_mgmt.h"
#include "lwscibuf_utils.h"

LwSciError LwSciBufAttrListGetImageTensorRecKeyPair(
    const LwSciBufAttrListRecKeyPair** recKeyPair,
    size_t* numElements) {
    LwSciError err = LwSciError_Success;
    static const LwSciBufAttrListRecKeyPair imageTensorRecKeyPair[] = {
       {(uint32_t)LwSciBufImageAttrKey_PlaneColorFormat,
        (uint32_t)LwSciBufTensorAttrKey_PixelFormat,
            LwSciBuf_MatchPolicy},
    };

    LWSCI_FNENTRY("");

    if ((NULL == recKeyPair) || (NULL == numElements)) {
        LWSCI_ERR_STR("Bad parameter supplied to LwSciBufAttrListGetImageTensorRecKeyPair\n");
        LwSciCommonPanic();
    }

    LWSCI_INFO("Input: recKeyPair %p, numElements %p\n", recKeyPair,
        numElements);

    *recKeyPair = imageTensorRecKeyPair;
    *numElements =
        sizeof(imageTensorRecKeyPair)/sizeof(imageTensorRecKeyPair[0]);

    LWSCI_FNEXIT("");
    return err;
}

LwSciError LwSciBufAttrListLwstomCompareImageTensor(
    LwSciBufAttrList attrList)
{
    #define imagePairCount  4U
    #define tensorPairCount 2U
    LwSciError err = LwSciError_Success;
    LwSciBufAttrValColorFmt imageColorFmt = LwSciColor_UpperBound;
    LwSciBufAttrKeyValuePair imageKeyValPair[imagePairCount] = {{LwSciBufAttrKey_LowerBound, NULL, 0UL}};
    LwSciBufAttrKeyValuePair tensorKeyValPair[tensorPairCount] = {{LwSciBufAttrKey_LowerBound, NULL, 0UL}};
    LwColorFormat colorFormat;
    LwColorDataType colorDataType = 0U;
    LwSciBufAttrValDataType imageDataType = LwSciDataType_UpperBound;
    LwSciBufAttrValDataType tensorDataType = LwSciDataType_UpperBound;
    uint8_t channelCount = 0U;
    const uint64_t *tensorSizePerDim = NULL;
    uint64_t tensorDimN = 0U, tensorDimH = 0U, tensorDimW = 0U, tensorDimC = 0U;
    uint64_t imageCount = 0U;
    uint32_t imageHeight = 0U, imageWidth = 0U;
    size_t numElements = 0U;
    uint32_t colorBPP = 0U;

    LWSCI_FNENTRY("");

    if (NULL == attrList) {
        LWSCI_ERR_STR("Bad parameter supplied to LwSciBufAttrListLwstomReconcileImageTensor\n");
        LwSciCommonPanic();
    }

    LWSCI_INFO("Input: attrList: %p\n", attrList);

    /* Obtain key values for image datatype */
    imageKeyValPair[0].key = LwSciBufImageAttrKey_PlaneColorFormat;
    imageKeyValPair[1].key = LwSciBufImageAttrKey_ImageCount;
    imageKeyValPair[2].key = LwSciBufImageAttrKey_PlaneHeight;
    imageKeyValPair[3].key = LwSciBufImageAttrKey_PlaneWidth;

    err = LwSciBufAttrListCommonGetAttrs(attrList, 0, imageKeyValPair, imagePairCount,
            LwSciBufAttrKeyType_Public, true);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("LwSciBufAttrListCommonGetAttrs failed\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
    imageColorFmt = *(const LwSciBufAttrValColorFmt*)imageKeyValPair[0].value;
    imageCount = *(const uint64_t*)imageKeyValPair[1].value;
    imageHeight = *(const uint32_t*)imageKeyValPair[2].value;
    imageWidth = *(const uint32_t*)imageKeyValPair[3].value;
    LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 11_5))

    /* get channel count for image colorformat and image datatype */
    err = LwSciColorToLwColor(imageColorFmt, &colorFormat);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("LwSciColorToLwColor failed\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    err = LwSciColorGetComponentCount(imageColorFmt, &channelCount);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("LwSciColorGetComponentCount failed\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    colorDataType = LwColorGetDataType(colorFormat);
    colorBPP = LwColorGetBPP(colorFormat);
    imageDataType = LwColorDataTypeToLwSciBufDataType(colorDataType,
                        channelCount, colorBPP);

    /* Obtain key values for tensor datatype */
    tensorKeyValPair[0].key = LwSciBufTensorAttrKey_DataType;
    tensorKeyValPair[1].key = LwSciBufTensorAttrKey_SizePerDim;

    err = LwSciBufAttrListCommonGetAttrs(attrList, 0, tensorKeyValPair,
            tensorPairCount, LwSciBufAttrKeyType_Public, true);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("LwSciBufAttrListCommonGetAttrs failed\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
    tensorDataType = *(const LwSciBufAttrValDataType*)tensorKeyValPair[0].value;
    tensorSizePerDim = (const uint64_t*)tensorKeyValPair[1].value;
    LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 11_5))
    numElements = tensorKeyValPair[1].len / sizeof(uint64_t);
    /* number of elements in tensorSizePerDim array should be 4 since
     * tensor dimensions supported for image/tensor reconciliation = 4
     */
    if (4UL != numElements) {
        err = LwSciError_ReconciliationFailed;
        LWSCI_ERR_STR("Image tensor reconciliation supports 4 dimensions only\n");
        LWSCI_ERR_ULONG("Number of elements provided as part of key value LwSciBufTensorAttrKey_SizePerDim are: \n",
            numElements);
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    /* compare image/tensor datatypes */
    if (imageDataType != tensorDataType) {
        err = LwSciError_ReconciliationFailed;
        LWSCI_ERR_STR("Datatypes of image and tensor mismatch\n");
        LWSCI_ERR_UINT("Image datatype: \n", (uint32_t)imageDataType);
        LWSCI_ERR_UINT("Tensor datatype: \n", (uint32_t)tensorDataType);
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    /* DLA sw (and lwmedia tensor) only supports following color formats
     * - LwSciColor_A8B8G8R8
     * - LwSciColor_Float_A16B16G16R16
     * these image color formats map to tensor of order NHWC.
     * Thus, we can find out NHWC parameters of tensor supplied by user via
     * LwSciBufTensorAttrKey_SizePerDim array
     */

    /* get NHWC dimensions for tensor */
    tensorDimN = tensorSizePerDim[0];
    tensorDimH = tensorSizePerDim[1];
    tensorDimW = tensorSizePerDim[2];
    tensorDimC = tensorSizePerDim[3];

    if (imageCount != tensorDimN) {
        err = LwSciError_ReconciliationFailed;
        LWSCI_ERR_STR("number of images should match 'N' dimension of tensor\n");
        LWSCI_ERR_ULONG("number of images: , ", imageCount);
        LWSCI_ERR_ULONG("'N' dimension of tensor: \n", tensorDimN);
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    if ((uint64_t)imageHeight != tensorDimH) {
        err = LwSciError_ReconciliationFailed;
        LWSCI_ERR_STR("Height of image should match 'H' dimension of tensor\n");
        LWSCI_ERR_UINT("Image height: , ", imageHeight);
        LWSCI_ERR_ULONG("'H' dimension of tensor: \n", tensorDimH);
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    if ((uint64_t)imageWidth != tensorDimW) {
        err = LwSciError_ReconciliationFailed;
        LWSCI_ERR_STR("Width of image should match 'W' dimension of tensor\n");
        LWSCI_ERR_UINT("Image width: , ", imageWidth);
        LWSCI_ERR_ULONG("'W' dimension of tensor: \n", tensorDimW);
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    if (channelCount != tensorDimC) {
        err = LwSciError_ReconciliationFailed;
        LWSCI_ERR_STR("Channel count for image colorformat should match 'C' component of tensor\n");
        LWSCI_ERR_UINT("Channel count for colorformat  ", (uint32_t)imageColorFmt);
        LWSCI_ERR_UINT("is: \n", (uint32_t)channelCount);
        LWSCI_ERR_ULONG("'C' component of tensor is: \n", tensorDimC);
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

ret:
    LWSCI_FNEXIT("");
    return err;
}
