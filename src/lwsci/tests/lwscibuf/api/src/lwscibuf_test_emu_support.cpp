/*
 * Copyright (c) 2020, LWPU CORPORATION. All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#include "lwscibuf_test_integration.h"
#include "lwscibuf_internal_x86.h"

static LwSciError umd1Setup(
    LwSciBufModule bufModule,
    LwSciBufAttrList* umd1AttrList)
{
    LwSciError err = LwSciError_Success;
    LwSciBufType bufType = LwSciBufType_Image;
    LwSciBufAttrValImageLayoutType layout = LwSciBufImage_BlockLinearType;
    uint64_t lrPad = 0U, tbPad = 100U;
    bool cpuAccessFlag = true;
    bool vpr = false;
    int32_t planeCount = 2U;

    LwSciBufAttrValColorFmt planeColorFmts[] =
            { LwSciColor_Y16, LwSciColor_U8V8 };
    LwSciBufAttrValColorStd planeColorStds[] =
            { LwSciColorStd_YcCbcCrc_SR, LwSciColorStd_YcCbcCrc_SR };
    LwSciBufAttrValImageScanType planeScanType[] =
            { LwSciBufScan_InterlaceType};
    int32_t planeWidths[] = { 640U, 320U };
    int32_t planeHeights[] = { 480U, 240U };
    LwSciBufAttrKeyValuePair imgBufAttrs[] = {
        {
            LwSciBufGeneralAttrKey_Types,
            &bufType,
            sizeof(bufType)
        },
        {
            LwSciBufGeneralAttrKey_NeedCpuAccess,
            &cpuAccessFlag,
            sizeof(cpuAccessFlag)
        },
        {
            LwSciBufImageAttrKey_Layout,
            &layout,
            sizeof(layout)
        },
        {
            LwSciBufImageAttrKey_TopPadding,
            &tbPad, sizeof(tbPad)
        },
        {
            LwSciBufImageAttrKey_BottomPadding,
            &tbPad,
            sizeof(tbPad)
        },
        {
            LwSciBufImageAttrKey_LeftPadding,
            &lrPad,
            sizeof(lrPad)
        },
        {
            LwSciBufImageAttrKey_RightPadding,
            &lrPad,
            sizeof(lrPad)
        },
        {
            LwSciBufImageAttrKey_VprFlag,
            &vpr,
            sizeof(vpr)
        },
        {
            LwSciBufImageAttrKey_PlaneCount,
            &planeCount,
            sizeof(planeCount)
        },
        {
            LwSciBufImageAttrKey_PlaneColorFormat,
            planeColorFmts,
            sizeof(planeColorFmts)
        },
        {
            LwSciBufImageAttrKey_PlaneColorStd,
            planeColorStds,
            sizeof(planeColorStds)
        },
        {
            LwSciBufImageAttrKey_PlaneWidth,
            planeWidths,
            sizeof(planeWidths)
        },
        {
            LwSciBufImageAttrKey_PlaneHeight,
            planeHeights,
            sizeof(planeHeights)
        },
        {
            LwSciBufImageAttrKey_ScanType,
            planeScanType,
            sizeof(planeScanType)
        },
    };

    err = LwSciBufAttrListCreate(bufModule, umd1AttrList);
    TESTERR_CHECK(err, "Failed to create UMD1 attribute list", err);

    err = LwSciBufAttrListSetAttrs(*umd1AttrList, imgBufAttrs,
            sizeof(imgBufAttrs)/sizeof(LwSciBufAttrKeyValuePair));
    TESTERR_CHECK(err, "Failed to set UMD1 attribute list", err);

    return err;

}

static LwSciError umd2Setup(
    LwSciBufModule bufModule,
    LwSciBufAttrList* umd2AttrList)
{
    LwSciError err = LwSciError_Success;
    LwSciBufType bufType = LwSciBufType_Image;
    LwSciBufMemDomain memDomain = LwSciBufMemDomain_Sysmem;

    LwSciBufAttrKeyValuePair imageAttrs[] = {
        {
            LwSciBufGeneralAttrKey_Types,
            &bufType,
            sizeof(bufType)
        },
    };

    LwSciBufInternalAttrKeyValuePair imageIntAttrs[] = {
        {
            LwSciBufInternalGeneralAttrKey_MemDomainArray,
            &memDomain,
            sizeof(memDomain),
        },
    };

    err = LwSciBufAttrListCreate(bufModule, umd2AttrList);
    TESTERR_CHECK(err, "Failed to create UMD2 attribute list", err);

    err = LwSciBufAttrListSetAttrs(*umd2AttrList, imageAttrs,
            sizeof(imageAttrs)/sizeof(LwSciBufAttrKeyValuePair));
    TESTERR_CHECK(err, "Failed to set UMD2 attribute list", err);

    err = LwSciBufAttrListSetInternalAttrs(*umd2AttrList, imageIntAttrs,
            sizeof(imageIntAttrs)/sizeof(LwSciBufInternalAttrKeyValuePair));
    TESTERR_CHECK(err, "Failed to set internal attribute list", err);

    return err;
}

TEST(TestLwSciBufEMUSupport, NegativeBadParameters)
{
    LwSciError err = LwSciError_Success;
    LwSciBufAttrList inputUnreconciledAttrLists[1] = {};
    LwSciBufAttrList attrListTemp = NULL;
    size_t attrListCount = 1;
    LwSciBufAttrKeyValuePair publicAttrs;
    size_t publicAttrsCount = 1;
    LwSciBufInternalAttrKeyValuePair internalAttrs;
    size_t internalAttrsCount = 1;
    LwSciBufOutputAttrKeyValuePair outputAttributes = {
        .publicAttrs = &publicAttrs,
        .publicAttrsCount = publicAttrsCount,
        .internalAttrs = &internalAttrs,
        .internalAttrsCount = internalAttrsCount};
    LwSciBufAttrList outputReconciledAttrList;
    LwSciBufAttrList conflictAttrList;

    NEGATIVE_TEST();

    err = LwSciBufAttrListReconcileWithOutputAttrs(
            NULL, attrListCount, outputAttributes,
            &outputReconciledAttrList, &conflictAttrList);

    ASSERT_EQ(err, LwSciError_BadParameter)
        << "Failed to validate inputUnreconciledAttrLists as NULL";

    err = LwSciBufAttrListReconcileWithOutputAttrs(
            inputUnreconciledAttrLists, 0, outputAttributes,
            &outputReconciledAttrList, &conflictAttrList);

    ASSERT_EQ(err, LwSciError_BadParameter)
        << "Failed to validate attrListCount as 0";

    err = LwSciBufAttrListReconcileWithOutputAttrs(
            inputUnreconciledAttrLists, attrListCount, outputAttributes,
            NULL, &conflictAttrList);

    ASSERT_EQ(err, LwSciError_BadParameter)
        << "Failed to validate outputReconciledAttrList as NULL";

    err = LwSciBufAttrListReconcileWithOutputAttrs(
            inputUnreconciledAttrLists, attrListCount, outputAttributes,
            &outputReconciledAttrList, NULL);

    ASSERT_EQ(err, LwSciError_BadParameter)
        << "Failed to validate conflictAttrList as NULL";

    outputAttributes.publicAttrs = NULL;
    err = LwSciBufAttrListReconcileWithOutputAttrs(
            inputUnreconciledAttrLists, attrListCount, outputAttributes,
            &outputReconciledAttrList, &conflictAttrList);

    ASSERT_EQ(err, LwSciError_BadParameter)
        << "Failed to validate outputAttributes.publicAttrs as NULL";
    outputAttributes.publicAttrs = &publicAttrs;

    outputAttributes.publicAttrsCount = 0;
    err = LwSciBufAttrListReconcileWithOutputAttrs(
            inputUnreconciledAttrLists, attrListCount, outputAttributes,
            &outputReconciledAttrList, &conflictAttrList);

    ASSERT_EQ(err, LwSciError_BadParameter)
        << "Failed to validate outputAttributes.publicAttrsCount as 0";
    outputAttributes.publicAttrsCount = publicAttrsCount;

    inputUnreconciledAttrLists[0] = attrListTemp;
    err = LwSciBufAttrListReconcileWithOutputAttrs(
            inputUnreconciledAttrLists, attrListCount, outputAttributes,
            &outputReconciledAttrList, &conflictAttrList);

    ASSERT_EQ(err, LwSciError_BadParameter)
        << "Failed to validate uninitialized attrList";

}

TEST(TestLwSciBufEMUSupport, NegativeImageBadOutputAttr)
{
    LwSciError err = LwSciError_Success;
    LwSciBufModule bufModule = NULL;
    LwSciBufAttrList umd1AttrList = NULL, umd2AttrList = NULL;
    LwSciBufObj bufObj = NULL;
    LwSciBufAttrList conflictList = NULL;
    LwSciBufRmHandle rmHandle = {0};
    uint64_t offset = 0U, len = 0U, size = 0U;


    LwSciBufAttrList inputUnreconciledAttrLists[1] = {};
    size_t attrListCount = 1;
    LwSciBufAttrKeyValuePair publicAttrs;
    size_t publicAttrsCount = 1;
    LwSciBufInternalAttrKeyValuePair internalAttrs;
    size_t internalAttrsCount = 0;
    LwSciBufOutputAttrKeyValuePair outputAttributes = {
        .publicAttrs = &publicAttrs,
        .publicAttrsCount = publicAttrsCount,
        .internalAttrs = &internalAttrs,
        .internalAttrsCount = internalAttrsCount};
    LwSciBufAttrList outputReconciledAttrList;
    LwSciBufAttrList conflictAttrList;

    NEGATIVE_TEST();

    ASSERT_EQ(LwSciBufModuleOpen(&bufModule), LwSciError_Success)
        << "Failed to open LwSciBuf Module";

    ASSERT_EQ(umd1Setup(bufModule, &umd1AttrList), LwSciError_Success)
        << "Failed to set UMD1 attributes";

    inputUnreconciledAttrLists[0] = umd1AttrList;

    err = LwSciBufAttrListReconcileWithOutputAttrs(
            inputUnreconciledAttrLists, attrListCount, outputAttributes,
            &outputReconciledAttrList, &conflictAttrList);

    ASSERT_EQ(err, LwSciError_BadParameter)
        << "Failed to verify output keys in outputAttr";

    LwSciBufAttrListFree(umd1AttrList);
    LwSciBufModuleClose(bufModule);
}

TEST(TestLwSciBufEMUSupport, NegativeImageIncompleteInternalOutputAttr)
{
    LwSciError err = LwSciError_Success;
    LwSciBufModule bufModule = NULL;
    LwSciBufAttrList umd1AttrList = NULL, umd2AttrList = NULL;
    LwSciBufObj bufObj = NULL;
    LwSciBufAttrList conflictList = NULL;
    LwSciBufRmHandle rmHandle = {0};
    uint64_t offset = 0U, len = 0U, size = 0U;

    uint64_t outputSize = 1;
    uint64_t outputAlign = 1;
    uint32_t outputBPP[2] = {1,1};
    uint64_t outputOffset[2] = {1,1};
    LwSciBufAttrValDataType outputDataType[2] = {LwSciDataType_Float32, LwSciDataType_Float32};
    uint8_t outputChannelCount[2] = {1,1};
    uint64_t outputSecondOffset[2] = {1,1};
    uint32_t outputPitch[2] = {1,1};
    uint32_t outputHeight[2] = {1,1};
    uint64_t outputPlaneSize[2] = {1,1};

    LwSciBufAttrKeyValuePair imagePublicOutputAttrs[] = {
        {
            LwSciBufImageAttrKey_Size,
            &outputSize,
            sizeof(outputSize)
        },
        {
            LwSciBufImageAttrKey_Alignment,
            &outputAlign,
            sizeof(outputAlign)
        },
        {
            LwSciBufImageAttrKey_PlaneBitsPerPixel,
            &outputBPP,
            sizeof(outputBPP)
        },
        {
            LwSciBufImageAttrKey_PlaneOffset,
            &outputOffset,
            sizeof(outputOffset)
        },
        {
            LwSciBufImageAttrKey_PlaneDatatype,
            &outputDataType,
            sizeof(outputDataType)
        },
        {
            LwSciBufImageAttrKey_PlaneChannelCount,
            &outputChannelCount,
            sizeof(outputChannelCount)
        },
        {
            LwSciBufImageAttrKey_PlaneSecondFieldOffset,
            &outputSecondOffset,
            sizeof(outputSecondOffset)
        },
        {
            LwSciBufImageAttrKey_PlanePitch,
            &outputPitch,
            sizeof(outputPitch)
        },
        {
            LwSciBufImageAttrKey_PlaneAlignedHeight,
            &outputHeight,
            sizeof(outputHeight)
        },
        {
            LwSciBufImageAttrKey_PlaneAlignedSize,
            &outputPlaneSize,
            sizeof(outputPlaneSize)
        },
    };

    uint32_t outputGobSize[2] = {1,1};
    uint32_t outputGobX[2] = {1,1};
    uint32_t outputGobY[2] = {1,1};
    uint32_t outputGobZ[2] = {1,1};
    LwSciBufInternalAttrKeyValuePair imageOutputIntAttrs[] = {
        {
            LwSciBufInternalImageAttrKey_PlaneGobSize,
            &outputGobSize,
            sizeof(outputGobSize),
        },
        {
            LwSciBufInternalImageAttrKey_PlaneLog2GobsPerBlockX,
            &outputGobX,
            sizeof(outputGobX),
        },
        {
            LwSciBufInternalImageAttrKey_PlaneLog2GobsPerBlockY,
            &outputGobY,
            sizeof(outputGobY),
        },
    };


    LwSciBufAttrList inputUnreconciledAttrLists[2] = {};
    size_t attrListCount = 2;
    size_t publicAttrsCount = 10;
    LwSciBufInternalAttrKeyValuePair internalAttrs;
    size_t internalAttrsCount = 3;
    LwSciBufOutputAttrKeyValuePair outputAttributes = {
        .publicAttrs = imagePublicOutputAttrs,
        .publicAttrsCount = publicAttrsCount,
        .internalAttrs = imageOutputIntAttrs,
        .internalAttrsCount = internalAttrsCount};
    LwSciBufAttrList outputReconciledAttrList;
    LwSciBufAttrList conflictAttrList;

    NEGATIVE_TEST();

    ASSERT_EQ(LwSciBufModuleOpen(&bufModule), LwSciError_Success)
        << "Failed to open LwSciBuf Module";

    ASSERT_EQ(umd1Setup(bufModule, &umd1AttrList), LwSciError_Success)
        << "Failed to set UMD1 attributes";

    ASSERT_EQ(umd2Setup(bufModule, &umd2AttrList), LwSciError_Success)
        << "Failed to set UMD2 attributes";

    inputUnreconciledAttrLists[0] = umd1AttrList;
    inputUnreconciledAttrLists[1] = umd2AttrList;

    err = LwSciBufAttrListReconcileWithOutputAttrs(
            inputUnreconciledAttrLists, attrListCount, outputAttributes,
            &outputReconciledAttrList, &conflictAttrList);

    ASSERT_EQ(err, LwSciError_BadParameter)
        << "Failed to validate missing internal output keys";

    LwSciBufAttrListFree(umd1AttrList);
    LwSciBufAttrListFree(umd2AttrList);
    LwSciBufModuleClose(bufModule);
}

TEST(TestLwSciBufEMUSupport, NegativeImageIncompletePublicOutputAttr)
{
    LwSciError err = LwSciError_Success;
    LwSciBufModule bufModule = NULL;
    LwSciBufAttrList umd1AttrList = NULL, umd2AttrList = NULL;
    LwSciBufObj bufObj = NULL;
    LwSciBufAttrList conflictList = NULL;
    LwSciBufRmHandle rmHandle = {0};
    uint64_t offset = 0U, len = 0U, size = 0U;

    uint64_t outputSize = 1;
    uint64_t outputAlign = 1;
    uint32_t outputBPP[2] = {1,1};
    uint64_t outputOffset[2] = {1,1};
    LwSciBufAttrValDataType outputDataType[2] = {LwSciDataType_Float32, LwSciDataType_Float32};
    uint8_t outputChannelCount[2] = {1,1};
    uint64_t outputSecondOffset[2] = {1,1};
    uint32_t outputPitch[2] = {1,1};
    uint32_t outputHeight[2] = {1,1};
    uint64_t outputPlaneSize[2] = {1,1};

    LwSciBufAttrKeyValuePair imagePublicOutputAttrs[] = {
        {
            LwSciBufImageAttrKey_Size,
            &outputSize,
            sizeof(outputSize)
        },
        {
            LwSciBufImageAttrKey_Alignment,
            &outputAlign,
            sizeof(outputAlign)
        },
        {
            LwSciBufImageAttrKey_PlaneBitsPerPixel,
            &outputBPP,
            sizeof(outputBPP)
        },
        {
            LwSciBufImageAttrKey_PlaneOffset,
            &outputOffset,
            sizeof(outputOffset)
        },
        {
            LwSciBufImageAttrKey_PlaneDatatype,
            &outputDataType,
            sizeof(outputDataType)
        },
        {
            LwSciBufImageAttrKey_PlaneChannelCount,
            &outputChannelCount,
            sizeof(outputChannelCount)
        },
        {
            LwSciBufImageAttrKey_PlaneSecondFieldOffset,
            &outputSecondOffset,
            sizeof(outputSecondOffset)
        },
        {
            LwSciBufImageAttrKey_PlanePitch,
            &outputPitch,
            sizeof(outputPitch)
        },
        {
            LwSciBufImageAttrKey_PlaneAlignedSize,
            &outputPlaneSize,
            sizeof(outputPlaneSize)
        },
    };

    uint32_t outputGobSize[2] = {1,1};
    uint32_t outputGobX[2] = {1,1};
    uint32_t outputGobY[2] = {1,1};
    uint32_t outputGobZ[2] = {1,1};
    LwSciBufInternalAttrKeyValuePair imageOutputIntAttrs[] = {
        {
            LwSciBufInternalImageAttrKey_PlaneGobSize,
            &outputGobSize,
            sizeof(outputGobSize),
        },
        {
            LwSciBufInternalImageAttrKey_PlaneLog2GobsPerBlockX,
            &outputGobX,
            sizeof(outputGobX),
        },
        {
            LwSciBufInternalImageAttrKey_PlaneLog2GobsPerBlockY,
            &outputGobY,
            sizeof(outputGobY),
        },
        {
            LwSciBufInternalImageAttrKey_PlaneLog2GobsPerBlockZ,
            &outputGobZ,
            sizeof(outputGobZ),
        },
    };


    LwSciBufAttrList inputUnreconciledAttrLists[2] = {};
    size_t attrListCount = 2;
    size_t publicAttrsCount = 9;
    LwSciBufInternalAttrKeyValuePair internalAttrs;
    size_t internalAttrsCount = 4;
    LwSciBufOutputAttrKeyValuePair outputAttributes = {
        .publicAttrs = imagePublicOutputAttrs,
        .publicAttrsCount = publicAttrsCount,
        .internalAttrs = imageOutputIntAttrs,
        .internalAttrsCount = internalAttrsCount};
    LwSciBufAttrList outputReconciledAttrList;
    LwSciBufAttrList conflictAttrList;

    NEGATIVE_TEST();

    ASSERT_EQ(LwSciBufModuleOpen(&bufModule), LwSciError_Success)
        << "Failed to open LwSciBuf Module";

    ASSERT_EQ(umd1Setup(bufModule, &umd1AttrList), LwSciError_Success)
        << "Failed to set UMD1 attributes";

    ASSERT_EQ(umd2Setup(bufModule, &umd2AttrList), LwSciError_Success)
        << "Failed to set UMD2 attributes";

    inputUnreconciledAttrLists[0] = umd1AttrList;
    inputUnreconciledAttrLists[1] = umd2AttrList;

    err = LwSciBufAttrListReconcileWithOutputAttrs(
            inputUnreconciledAttrLists, attrListCount, outputAttributes,
            &outputReconciledAttrList, &conflictAttrList);

    ASSERT_EQ(err, LwSciError_BadParameter)
        << "Failed to validate missing public output keys";

    LwSciBufAttrListFree(umd1AttrList);
    LwSciBufAttrListFree(umd2AttrList);
    LwSciBufModuleClose(bufModule);
}

TEST(TestLwSciBufEMUSupport, NegativeImageDuplicatePublicKey)
{
    LwSciError err = LwSciError_Success;
    LwSciBufModule bufModule = NULL;
    LwSciBufAttrList umd1AttrList = NULL, umd2AttrList = NULL;
    LwSciBufObj bufObj = NULL;
    LwSciBufAttrList conflictList = NULL;
    LwSciBufRmHandle rmHandle = {0};
    uint64_t offset = 0U, len = 0U, size = 0U;

    uint64_t outputSize = 1;
    uint64_t outputAlign = 1;
    uint32_t outputBPP[2] = {1,1};
    uint64_t outputOffset[2] = {1,1};
    LwSciBufAttrValDataType outputDataType[2] = {LwSciDataType_Float32, LwSciDataType_Float32};
    uint8_t outputChannelCount[2] = {1,1};
    uint64_t outputSecondOffset[2] = {1,1};
    uint32_t outputPitch[2] = {1,1};
    uint32_t outputHeight[2] = {1,1};
    uint64_t outputPlaneSize[2] = {1,1};

    LwSciBufAttrKeyValuePair imagePublicOutputAttrs[] = {
        {
            LwSciBufImageAttrKey_Size,
            &outputSize,
            sizeof(outputSize)
        },
        {
            LwSciBufImageAttrKey_Alignment,
            &outputAlign,
            sizeof(outputAlign)
        },
        {
            LwSciBufImageAttrKey_PlaneBitsPerPixel,
            &outputBPP,
            sizeof(outputBPP)
        },
        {
            LwSciBufImageAttrKey_PlaneOffset,
            &outputOffset,
            sizeof(outputOffset)
        },
        {
            LwSciBufImageAttrKey_PlaneDatatype,
            &outputDataType,
            sizeof(outputDataType)
        },
        {
            LwSciBufImageAttrKey_PlaneChannelCount,
            &outputChannelCount,
            sizeof(outputChannelCount)
        },
        {
            LwSciBufImageAttrKey_PlaneSecondFieldOffset,
            &outputSecondOffset,
            sizeof(outputSecondOffset)
        },
        {
            LwSciBufImageAttrKey_PlanePitch,
            &outputPitch,
            sizeof(outputPitch)
        },
        {
            LwSciBufImageAttrKey_PlaneAlignedHeight,
            &outputHeight,
            sizeof(outputHeight)
        },
        {
            LwSciBufImageAttrKey_PlaneAlignedSize,
            &outputPlaneSize,
            sizeof(outputPlaneSize)
        },
        {
            LwSciBufImageAttrKey_PlaneSecondFieldOffset,
            &outputSecondOffset,
            sizeof(outputSecondOffset)
        },
    };

    uint32_t outputGobSize[2] = {1,1};
    uint32_t outputGobX[2] = {1,1};
    uint32_t outputGobY[2] = {1,1};
    uint32_t outputGobZ[2] = {1,1};
    LwSciBufInternalAttrKeyValuePair imageOutputIntAttrs[] = {
        {
            LwSciBufInternalImageAttrKey_PlaneGobSize,
            &outputGobSize,
            sizeof(outputGobSize),
        },
        {
            LwSciBufInternalImageAttrKey_PlaneLog2GobsPerBlockX,
            &outputGobX,
            sizeof(outputGobX),
        },
        {
            LwSciBufInternalImageAttrKey_PlaneLog2GobsPerBlockY,
            &outputGobY,
            sizeof(outputGobY),
        },
        {
            LwSciBufInternalImageAttrKey_PlaneLog2GobsPerBlockZ,
            &outputGobZ,
            sizeof(outputGobZ),
        },
    };


    LwSciBufAttrList inputUnreconciledAttrLists[2] = {};
    size_t attrListCount = 2;
    size_t publicAttrsCount = 11;
    LwSciBufInternalAttrKeyValuePair internalAttrs;
    size_t internalAttrsCount = 4;
    LwSciBufOutputAttrKeyValuePair outputAttributes = {
        .publicAttrs = imagePublicOutputAttrs,
        .publicAttrsCount = publicAttrsCount,
        .internalAttrs = imageOutputIntAttrs,
        .internalAttrsCount = internalAttrsCount};
    LwSciBufAttrList outputReconciledAttrList;
    LwSciBufAttrList conflictAttrList;

    NEGATIVE_TEST();

    ASSERT_EQ(LwSciBufModuleOpen(&bufModule), LwSciError_Success)
        << "Failed to open LwSciBuf Module";

    ASSERT_EQ(umd1Setup(bufModule, &umd1AttrList), LwSciError_Success)
        << "Failed to set UMD1 attributes";

    ASSERT_EQ(umd2Setup(bufModule, &umd2AttrList), LwSciError_Success)
        << "Failed to set UMD2 attributes";

    inputUnreconciledAttrLists[0] = umd1AttrList;
    inputUnreconciledAttrLists[1] = umd2AttrList;

    err = LwSciBufAttrListReconcileWithOutputAttrs(
            inputUnreconciledAttrLists, attrListCount, outputAttributes,
            &outputReconciledAttrList, &conflictAttrList);

    ASSERT_EQ(err, LwSciError_BadParameter)
        << "Failed to validate duplicate public output attr";

    LwSciBufAttrListFree(umd1AttrList);
    LwSciBufAttrListFree(umd2AttrList);
    LwSciBufModuleClose(bufModule);
}

TEST(TestLwSciBufEMUSupport, NegativeImageDuplicateInternalKey)
{
    LwSciError err = LwSciError_Success;
    LwSciBufModule bufModule = NULL;
    LwSciBufAttrList umd1AttrList = NULL, umd2AttrList = NULL;
    LwSciBufObj bufObj = NULL;
    LwSciBufAttrList conflictList = NULL;
    LwSciBufRmHandle rmHandle = {0};
    uint64_t offset = 0U, len = 0U, size = 0U;

    uint64_t outputSize = 1;
    uint64_t outputAlign = 1;
    uint32_t outputBPP[2] = {1,1};
    uint64_t outputOffset[2] = {1,1};
    LwSciBufAttrValDataType outputDataType[2] = {LwSciDataType_Float32, LwSciDataType_Float32};
    uint8_t outputChannelCount[2] = {1,1};
    uint64_t outputSecondOffset[2] = {1,1};
    uint32_t outputPitch[2] = {1,1};
    uint32_t outputHeight[2] = {1,1};
    uint64_t outputPlaneSize[2] = {1,1};

    LwSciBufAttrKeyValuePair imagePublicOutputAttrs[] = {
        {
            LwSciBufImageAttrKey_Size,
            &outputSize,
            sizeof(outputSize)
        },
        {
            LwSciBufImageAttrKey_Alignment,
            &outputAlign,
            sizeof(outputAlign)
        },
        {
            LwSciBufImageAttrKey_PlaneBitsPerPixel,
            &outputBPP,
            sizeof(outputBPP)
        },
        {
            LwSciBufImageAttrKey_PlaneOffset,
            &outputOffset,
            sizeof(outputOffset)
        },
        {
            LwSciBufImageAttrKey_PlaneDatatype,
            &outputDataType,
            sizeof(outputDataType)
        },
        {
            LwSciBufImageAttrKey_PlaneChannelCount,
            &outputChannelCount,
            sizeof(outputChannelCount)
        },
        {
            LwSciBufImageAttrKey_PlaneSecondFieldOffset,
            &outputSecondOffset,
            sizeof(outputSecondOffset)
        },
        {
            LwSciBufImageAttrKey_PlanePitch,
            &outputPitch,
            sizeof(outputPitch)
        },
        {
            LwSciBufImageAttrKey_PlaneAlignedHeight,
            &outputHeight,
            sizeof(outputHeight)
        },
        {
            LwSciBufImageAttrKey_PlaneAlignedSize,
            &outputPlaneSize,
            sizeof(outputPlaneSize)
        },
    };

    uint32_t outputGobSize[2] = {1,1};
    uint32_t outputGobX[2] = {1,1};
    uint32_t outputGobY[2] = {1,1};
    uint32_t outputGobZ[2] = {1,1};
    LwSciBufInternalAttrKeyValuePair imageOutputIntAttrs[] = {
        {
            LwSciBufInternalImageAttrKey_PlaneGobSize,
            &outputGobSize,
            sizeof(outputGobSize),
        },
        {
            LwSciBufInternalImageAttrKey_PlaneLog2GobsPerBlockX,
            &outputGobX,
            sizeof(outputGobX),
        },
        {
            LwSciBufInternalImageAttrKey_PlaneLog2GobsPerBlockY,
            &outputGobY,
            sizeof(outputGobY),
        },
        {
            LwSciBufInternalImageAttrKey_PlaneLog2GobsPerBlockZ,
            &outputGobZ,
            sizeof(outputGobZ),
        },
        {
            LwSciBufInternalImageAttrKey_PlaneLog2GobsPerBlockZ,
            &outputGobZ,
            sizeof(outputGobZ),
        },
    };


    LwSciBufAttrList inputUnreconciledAttrLists[2] = {};
    size_t attrListCount = 2;
    size_t publicAttrsCount = 11;
    LwSciBufInternalAttrKeyValuePair internalAttrs;
    size_t internalAttrsCount = 5;
    LwSciBufOutputAttrKeyValuePair outputAttributes = {
        .publicAttrs = imagePublicOutputAttrs,
        .publicAttrsCount = publicAttrsCount,
        .internalAttrs = imageOutputIntAttrs,
        .internalAttrsCount = internalAttrsCount};
    LwSciBufAttrList outputReconciledAttrList;
    LwSciBufAttrList conflictAttrList;

    NEGATIVE_TEST();

    ASSERT_EQ(LwSciBufModuleOpen(&bufModule), LwSciError_Success)
        << "Failed to open LwSciBuf Module";

    ASSERT_EQ(umd1Setup(bufModule, &umd1AttrList), LwSciError_Success)
        << "Failed to set UMD1 attributes";

    ASSERT_EQ(umd2Setup(bufModule, &umd2AttrList), LwSciError_Success)
        << "Failed to set UMD2 attributes";

    inputUnreconciledAttrLists[0] = umd1AttrList;
    inputUnreconciledAttrLists[1] = umd2AttrList;

    err = LwSciBufAttrListReconcileWithOutputAttrs(
            inputUnreconciledAttrLists, attrListCount, outputAttributes,
            &outputReconciledAttrList, &conflictAttrList);

    ASSERT_EQ(err, LwSciError_BadParameter)
        << "Failed to validate duplicate internal output attr";

    LwSciBufAttrListFree(umd1AttrList);
    LwSciBufAttrListFree(umd2AttrList);
    LwSciBufModuleClose(bufModule);
}

TEST(TestLwSciBufEMUSupport, NegativeImageUnknownPublicKey)
{
    LwSciError err = LwSciError_Success;
    LwSciBufModule bufModule = NULL;
    LwSciBufAttrList umd1AttrList = NULL, umd2AttrList = NULL;
    LwSciBufObj bufObj = NULL;
    LwSciBufAttrList conflictList = NULL;
    LwSciBufRmHandle rmHandle = {0};
    uint64_t offset = 0U, len = 0U, size = 0U;

    uint64_t outputSize = 1;
    uint64_t outputAlign = 1;
    uint32_t outputBPP[2] = {1,1};
    uint64_t outputOffset[2] = {1,1};
    LwSciBufAttrValDataType outputDataType[2] = {LwSciDataType_Float32, LwSciDataType_Float32};
    uint8_t outputChannelCount[2] = {1,1};
    uint64_t outputSecondOffset[2] = {1,1};
    uint32_t outputPitch[2] = {1,1};
    uint32_t outputHeight[2] = {1,1};
    uint64_t outputPlaneSize[2] = {1,1};
    uint32_t inputPlaneCount = 1;

    LwSciBufAttrKeyValuePair imagePublicOutputAttrs[] = {
        {
            LwSciBufImageAttrKey_Size,
            &outputSize,
            sizeof(outputSize)
        },
        {
            LwSciBufImageAttrKey_Alignment,
            &outputAlign,
            sizeof(outputAlign)
        },
        {
            LwSciBufImageAttrKey_PlaneBitsPerPixel,
            &outputBPP,
            sizeof(outputBPP)
        },
        {
            LwSciBufImageAttrKey_PlaneOffset,
            &outputOffset,
            sizeof(outputOffset)
        },
        {
            LwSciBufImageAttrKey_PlaneDatatype,
            &outputDataType,
            sizeof(outputDataType)
        },
        {
            LwSciBufImageAttrKey_PlaneChannelCount,
            &outputChannelCount,
            sizeof(outputChannelCount)
        },
        {
            LwSciBufImageAttrKey_PlaneSecondFieldOffset,
            &outputSecondOffset,
            sizeof(outputSecondOffset)
        },
        {
            LwSciBufImageAttrKey_PlanePitch,
            &outputPitch,
            sizeof(outputPitch)
        },
        {
            LwSciBufImageAttrKey_PlaneAlignedHeight,
            &outputHeight,
            sizeof(outputHeight)
        },
        {
            LwSciBufImageAttrKey_PlaneAlignedSize,
            &outputPlaneSize,
            sizeof(outputPlaneSize)
        },
        {
            LwSciBufImageAttrKey_PlaneCount,
            &inputPlaneCount,
            sizeof(inputPlaneCount)
        },
    };

    uint32_t outputGobSize[2] = {1,1};
    uint32_t outputGobX[2] = {1,1};
    uint32_t outputGobY[2] = {1,1};
    uint32_t outputGobZ[2] = {1,1};
    LwSciBufInternalAttrKeyValuePair imageOutputIntAttrs[] = {
        {
            LwSciBufInternalImageAttrKey_PlaneGobSize,
            &outputGobSize,
            sizeof(outputGobSize),
        },
        {
            LwSciBufInternalImageAttrKey_PlaneLog2GobsPerBlockX,
            &outputGobX,
            sizeof(outputGobX),
        },
        {
            LwSciBufInternalImageAttrKey_PlaneLog2GobsPerBlockY,
            &outputGobY,
            sizeof(outputGobY),
        },
        {
            LwSciBufInternalImageAttrKey_PlaneLog2GobsPerBlockZ,
            &outputGobZ,
            sizeof(outputGobZ),
        },
        {
            LwSciBufInternalImageAttrKey_PlaneLog2GobsPerBlockZ,
            &outputGobZ,
            sizeof(outputGobZ),
        },
    };


    LwSciBufAttrList inputUnreconciledAttrLists[2] = {};
    size_t attrListCount = 2;
    size_t publicAttrsCount = 11;
    LwSciBufInternalAttrKeyValuePair internalAttrs;
    size_t internalAttrsCount = 5;
    LwSciBufOutputAttrKeyValuePair outputAttributes = {
        .publicAttrs = imagePublicOutputAttrs,
        .publicAttrsCount = publicAttrsCount,
        .internalAttrs = imageOutputIntAttrs,
        .internalAttrsCount = internalAttrsCount};
    LwSciBufAttrList outputReconciledAttrList;
    LwSciBufAttrList conflictAttrList;

    NEGATIVE_TEST();

    ASSERT_EQ(LwSciBufModuleOpen(&bufModule), LwSciError_Success)
        << "Failed to open LwSciBuf Module";

    ASSERT_EQ(umd1Setup(bufModule, &umd1AttrList), LwSciError_Success)
        << "Failed to set UMD1 attributes";

    ASSERT_EQ(umd2Setup(bufModule, &umd2AttrList), LwSciError_Success)
        << "Failed to set UMD2 attributes";

    inputUnreconciledAttrLists[0] = umd1AttrList;
    inputUnreconciledAttrLists[1] = umd2AttrList;

    err = LwSciBufAttrListReconcileWithOutputAttrs(
            inputUnreconciledAttrLists, attrListCount, outputAttributes,
            &outputReconciledAttrList, &conflictAttrList);

    ASSERT_EQ(err, LwSciError_BadParameter)
        << "Failed to validate unknown public output attr";

    LwSciBufAttrListFree(umd1AttrList);
    LwSciBufAttrListFree(umd2AttrList);
    LwSciBufModuleClose(bufModule);
}

TEST(TestLwSciBufEMUSupport, NegativeBadLen)
{
    LwSciError err = LwSciError_Success;
    LwSciBufModule bufModule = NULL;
    LwSciBufAttrList umd1AttrList = NULL, umd2AttrList = NULL;
    LwSciBufObj bufObj = NULL;
    LwSciBufAttrList conflictList = NULL;
    LwSciBufRmHandle rmHandle = {0};
    uint64_t offset = 0U, len = 0U, size = 0U;

    uint64_t outputSize = 1;
    uint64_t outputAlign = 1;
    uint32_t outputBPP[2] = {1,1};
    uint64_t outputOffset[2] = {1,1};
    LwSciBufAttrValDataType outputDataType[2] = {LwSciDataType_Float32, LwSciDataType_Float32};
    uint8_t outputChannelCount[2] = {1,1};
    uint64_t outputSecondOffset[2] = {1,1};
    uint32_t outputPitch[2] = {1,1};
    uint32_t outputHeight[2] = {1,1};
    uint64_t outputPlaneSize[2] = {1,1};

    LwSciBufAttrKeyValuePair imagePublicOutputAttrs[] = {
        {
            LwSciBufImageAttrKey_Size,
            &outputSize,
            sizeof(outputSize)
        },
        {
            LwSciBufImageAttrKey_Alignment,
            &outputAlign,
            sizeof(outputAlign)
        },
        {
            LwSciBufImageAttrKey_PlaneBitsPerPixel,
            &outputBPP,
            sizeof(outputBPP)
        },
        {
            LwSciBufImageAttrKey_PlaneOffset,
            &outputOffset,
            sizeof(outputOffset)
        },
        {
            LwSciBufImageAttrKey_PlaneDatatype,
            &outputDataType,
            sizeof(outputDataType)
        },
        {
            LwSciBufImageAttrKey_PlaneChannelCount,
            &outputChannelCount,
            sizeof(outputChannelCount)
        },
        {
            LwSciBufImageAttrKey_PlaneSecondFieldOffset,
            &outputSecondOffset,
            sizeof(outputSecondOffset)
        },
        {
            LwSciBufImageAttrKey_PlanePitch,
            &outputPitch,
            sizeof(outputPitch)
        },
        {
            LwSciBufImageAttrKey_PlaneAlignedHeight,
            &outputHeight,
            sizeof(outputHeight)
        },
        {
            LwSciBufImageAttrKey_PlaneAlignedSize,
            &outputPlaneSize,
            sizeof(outputPlaneSize)
        },
    };

    uint32_t outputGobSize[2] = {1,1};
    uint32_t outputGobX[2] = {1,1};
    uint32_t outputGobY[2] = {1,1};
    uint32_t outputGobZ[2] = {1,1};
    LwSciBufInternalAttrKeyValuePair imageOutputIntAttrs[] = {
        {
            LwSciBufInternalImageAttrKey_PlaneGobSize,
            &outputGobSize,
            sizeof(outputGobSize),
        },
        {
            LwSciBufInternalImageAttrKey_PlaneLog2GobsPerBlockX,
            &outputGobX,
            sizeof(outputGobX),
        },
        {
            LwSciBufInternalImageAttrKey_PlaneLog2GobsPerBlockY,
            &outputGobY,
            sizeof(outputGobY),
        },
        {
            LwSciBufInternalImageAttrKey_PlaneLog2GobsPerBlockZ,
            &outputGobZ,
            sizeof(outputGobZ)*2,
        },
    };


    LwSciBufAttrList inputUnreconciledAttrLists[2] = {};
    size_t attrListCount = 2;
    size_t publicAttrsCount = 10;
    LwSciBufInternalAttrKeyValuePair internalAttrs;
    size_t internalAttrsCount = 4;
    LwSciBufOutputAttrKeyValuePair outputAttributes = {
        .publicAttrs = imagePublicOutputAttrs,
        .publicAttrsCount = publicAttrsCount,
        .internalAttrs = imageOutputIntAttrs,
        .internalAttrsCount = internalAttrsCount};
    LwSciBufAttrList outputReconciledAttrList;
    LwSciBufAttrList conflictAttrList;


    ASSERT_EQ(LwSciBufModuleOpen(&bufModule), LwSciError_Success)
        << "Failed to open LwSciBuf Module";

    ASSERT_EQ(umd1Setup(bufModule, &umd1AttrList), LwSciError_Success)
        << "Failed to set UMD1 attributes";

    ASSERT_EQ(umd2Setup(bufModule, &umd2AttrList), LwSciError_Success)
        << "Failed to set UMD2 attributes";

    inputUnreconciledAttrLists[0] = umd1AttrList;
    inputUnreconciledAttrLists[1] = umd2AttrList;

    err = LwSciBufAttrListReconcileWithOutputAttrs(
            inputUnreconciledAttrLists, attrListCount, outputAttributes,
            &outputReconciledAttrList, &conflictAttrList);

    ASSERT_EQ(err, LwSciError_BadParameter)
        << "Failed to validate wrong len attribute";

    LwSciBufAttrListFree(umd1AttrList);
    LwSciBufAttrListFree(umd2AttrList);
    LwSciBufModuleClose(bufModule);
}

TEST(TestLwSciBufEMUSupport, ImageSuccess)
{
    LwSciError err = LwSciError_Success;
    LwSciBufModule bufModule = NULL;
    LwSciBufAttrList umd1AttrList = NULL, umd2AttrList = NULL;
    LwSciBufObj bufObj = NULL;
    LwSciBufAttrList conflictList = NULL;
    LwSciBufRmHandle rmHandle = {0};

    uint64_t outputSize = 927504U;
    uint64_t outputAlign = 8192U;
    uint32_t outputBPP[2] = {16, 16};
    uint64_t outputOffset[2] = {0, 655360};
    LwSciBufAttrValDataType outputDataType[2] = {LwSciDataType_Uint16, LwSciDataType_Uint8};
    uint8_t outputChannelCount[2] = {1, 2};
    uint64_t outputSecondOffset[2] = {327680, 737280};
    uint32_t outputPitch[2] = {1280, 640};
    uint32_t outputHeight[2] = {256, 128};
    uint64_t outputPlaneSize[2] = {655360, 262144};

    LwSciBufAttrKeyValuePair imagePublicOutputAttrs[] = {
        {
            LwSciBufImageAttrKey_Size,
            &outputSize,
            sizeof(outputSize)
        },
        {
            LwSciBufImageAttrKey_Alignment,
            &outputAlign,
            sizeof(outputAlign)
        },
        {
            LwSciBufImageAttrKey_PlaneBitsPerPixel,
            &outputBPP,
            sizeof(outputBPP)
        },
        {
            LwSciBufImageAttrKey_PlaneOffset,
            &outputOffset,
            sizeof(outputOffset)
        },
        {
            LwSciBufImageAttrKey_PlaneDatatype,
            &outputDataType,
            sizeof(outputDataType)
        },
        {
            LwSciBufImageAttrKey_PlaneChannelCount,
            &outputChannelCount,
            sizeof(outputChannelCount)
        },
        {
            LwSciBufImageAttrKey_PlaneSecondFieldOffset,
            &outputSecondOffset,
            sizeof(outputSecondOffset)
        },
        {
            LwSciBufImageAttrKey_PlanePitch,
            &outputPitch,
            sizeof(outputPitch)
        },
        {
            LwSciBufImageAttrKey_PlaneAlignedHeight,
            &outputHeight,
            sizeof(outputHeight)
        },
        {
            LwSciBufImageAttrKey_PlaneAlignedSize,
            &outputPlaneSize,
            sizeof(outputPlaneSize)
        },
    };

    uint32_t outputGobSize[2] = {1, 1};
    uint32_t outputGobX[2] = {0, 0};
    uint32_t outputGobY[2] = {1,1};
    uint32_t outputGobZ[2] = {0, 0};
    LwSciBufInternalAttrKeyValuePair imageOutputIntAttrs[] = {
        {
            LwSciBufInternalImageAttrKey_PlaneGobSize,
            &outputGobSize,
            sizeof(outputGobSize),
        },
        {
            LwSciBufInternalImageAttrKey_PlaneLog2GobsPerBlockX,
            &outputGobX,
            sizeof(outputGobX),
        },
        {
            LwSciBufInternalImageAttrKey_PlaneLog2GobsPerBlockY,
            &outputGobY,
            sizeof(outputGobY),
        },
        {
            LwSciBufInternalImageAttrKey_PlaneLog2GobsPerBlockZ,
            &outputGobZ,
            sizeof(outputGobZ),
        },
    };


    LwSciBufAttrList inputUnreconciledAttrLists[2] = {};
    size_t attrListCount = 2;
    size_t publicAttrsCount = 10;
    LwSciBufInternalAttrKeyValuePair internalAttrs;
    size_t internalAttrsCount = 4;
    LwSciBufOutputAttrKeyValuePair outputAttributes = {
        .publicAttrs = imagePublicOutputAttrs,
        .publicAttrsCount = publicAttrsCount,
        .internalAttrs = imageOutputIntAttrs,
        .internalAttrsCount = internalAttrsCount};
    LwSciBufAttrList outputReconciledAttrList;
    LwSciBufAttrList conflictAttrList;


    ASSERT_EQ(LwSciBufModuleOpen(&bufModule), LwSciError_Success)
        << "Failed to open LwSciBuf Module";

    ASSERT_EQ(umd1Setup(bufModule, &umd1AttrList), LwSciError_Success)
        << "Failed to set UMD1 attributes";

    ASSERT_EQ(umd2Setup(bufModule, &umd2AttrList), LwSciError_Success)
        << "Failed to set UMD2 attributes";

    inputUnreconciledAttrLists[0] = umd1AttrList;
    inputUnreconciledAttrLists[1] = umd2AttrList;

    err = LwSciBufAttrListReconcileWithOutputAttrs(
            inputUnreconciledAttrLists, attrListCount, outputAttributes,
            &outputReconciledAttrList, &conflictAttrList);

    ASSERT_EQ(err, LwSciError_Success)
        << "Failed to override";

    /* allocate RAW buffer */
    LwSciBufAttrList rawAttrList;

    err = LwSciBufAttrListCreate(bufModule, &rawAttrList);
    ASSERT_EQ(err, LwSciError_Success) << "Failed to allocate Raw buffer AttrList";

    LwSciBufType rawBufType = LwSciBufType_RawBuffer;
    uint64_t rawSize = 927504U;
    uint64_t rawAlignment = 8192U;
    bool rawCpuAccessFlag = true;

    LwSciBufAttrKeyValuePair rawBufAttrs[] = {
        {
            LwSciBufGeneralAttrKey_Types,
            &rawBufType,
            sizeof(rawBufType)
        },
        {
            LwSciBufRawBufferAttrKey_Size,
            &rawSize,
            sizeof(rawSize)
        },
        {
            LwSciBufRawBufferAttrKey_Align,
            &rawAlignment,
            sizeof(rawAlignment)
        },
        {
            LwSciBufGeneralAttrKey_NeedCpuAccess,
            &rawCpuAccessFlag,
            sizeof(rawCpuAccessFlag)
        },
    };

    err = LwSciBufAttrListSetAttrs(rawAttrList, rawBufAttrs,
            sizeof(rawBufAttrs)/sizeof(LwSciBufAttrKeyValuePair));
    ASSERT_EQ(err, LwSciError_Success) << "Failed to set raw buffer attrs";

    LwSciBufObj rawBufObj = NULL;
    LwSciBufAttrList rawConflictList = NULL;

    err = LwSciBufAttrListReconcileAndObjAlloc(&rawAttrList, 1U,
                &rawBufObj, &rawConflictList);
    ASSERT_EQ(err, LwSciError_Success) << "Failed to reconcile and alloc attributes";

    LwSciBufRmHandle rawRmHandle = {0};
    uint64_t offset = 0U, len = 0U, size = 0U;
    err = LwSciBufObjGetMemHandle(rawBufObj, &rawRmHandle, &offset, &len);
    ASSERT_EQ(err, LwSciError_Success) << "Failed to Get Lwrm Memhandle for the object";

    LwSciBufObj createdObfObj;
    err = LwSciBufObjCreateFromMemHandle(rawRmHandle, 0, len,
            outputReconciledAttrList, &createdObfObj);
    ASSERT_EQ(err, LwSciError_Success) << "Failed to create mem handle";

    LwSciBufObjFree(createdObfObj);
    LwSciBufObjFree(rawBufObj);
    LwSciBufAttrListFree(rawAttrList);
    LwSciBufAttrListFree(outputReconciledAttrList);
    LwSciBufAttrListFree(umd1AttrList);
    LwSciBufAttrListFree(umd2AttrList);
    LwSciBufModuleClose(bufModule);
}

