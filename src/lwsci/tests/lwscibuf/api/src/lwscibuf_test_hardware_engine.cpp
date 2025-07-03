/*
 * Copyright (c) 2020-2022, LWPU CORPORATION. All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#include "lwscibuf_basic_test.h"
#include <string.h>
#include <vector>

//This is to avoid rtti compilation error in x86 build
#define GTEST_HAS_RTTI 0

class EngineInfo
{
public:
    LwSciBufHwEngName engineName;
    int64_t engineID;

    EngineInfo(LwSciBufHwEngName name, int64_t id) {
        engineName = name;
        engineID = id;
    }

    bool verifyEngineId(int64_t id) const {
        return engineID == id;
    }

    bool verifyEngineName(LwSciBufHwEngName name) const {
        return engineName == name;
    }
};

class HardwareEngine : public LwSciBufBasicTest
{
protected :
    std::shared_ptr<LwSciBufAttrListRec> list;
    std::vector<EngineInfo> engineList;

    void setEngineList() {
        engineList.push_back(EngineInfo(LwSciBufHwEngName_Display, 4));
        engineList.push_back(EngineInfo(LwSciBufHwEngName_Isp,     11));
        engineList.push_back(EngineInfo(LwSciBufHwEngName_Vi,      12));
        engineList.push_back(EngineInfo(LwSciBufHwEngName_Csi,     30));
        engineList.push_back(EngineInfo(LwSciBufHwEngName_Vic,     106));
        engineList.push_back(EngineInfo(LwSciBufHwEngName_Gpu,     107));
        engineList.push_back(EngineInfo(LwSciBufHwEngName_MSENC,   109));
        engineList.push_back(EngineInfo(LwSciBufHwEngName_LWDEC,   117));
        engineList.push_back(EngineInfo(LwSciBufHwEngName_LWJPG,   118));
        engineList.push_back(EngineInfo(LwSciBufHwEngName_PVA,     121));
        engineList.push_back(EngineInfo(LwSciBufHwEngName_DLA,     122));
        engineList.push_back(EngineInfo(LwSciBufHwEngName_PCIe,    123));
        engineList.push_back(EngineInfo(LwSciBufHwEngName_OFA,     124));
    }

    void SetUp() override
    {
        LwSciError error = LwSciError_Success;

        LwSciBufBasicTest::SetUp();

        list = peer.createAttrList(&error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_NE(list.get(), nullptr);
    }

    void TearDown() override
    {
        LwSciBufBasicTest::TearDown();

        list.reset();
    }
};

/**
* Test case:  Various engines can be set as values of EngineList attribute
*/
TEST_F(HardwareEngine, EngineListTest)
{
    uint64_t lPad = 0U, tPad = 100U, bPad = 50U, rPad = 25U;
    uint32_t planeCount = 2U;
    uint64_t imageCount = 1U;

    bool vprFlag = true;
    bool needCpuAccess = true;

    LwSciBufAttrValColorFmt planeColorFmts[2] = { LwSciColor_Y16,
                                                 LwSciColor_U8V8 };
    LwSciBufAttrValColorStd planeColorStds[2] = { LwSciColorStd_YcCbcCrc_SR,
                                                 LwSciColorStd_YcCbcCrc_SR };
    LwSciBufAttrValImageScanType planeScanType[1] = { LwSciBufScan_ProgressiveType };
    LwSciBufAttrValImageLayoutType layout = LwSciBufImage_BlockLinearType;
    LwSciBufType bufType = LwSciBufType_Image;
    LwSciBufAttrValAccessPerm perm = LwSciBufAccessPerm_ReadWrite;

    uint32_t planeWidths[2] = { 640U, 320U };
    uint32_t planeHeights[2] = { 480U, 240U };

    SET_ATTR(list.get(), LwSciBufGeneralAttrKey_Types, bufType);
    SET_ATTR(list.get(), LwSciBufGeneralAttrKey_NeedCpuAccess, needCpuAccess);
    SET_ATTR(list.get(), LwSciBufImageAttrKey_Layout, layout);
    SET_ATTR(list.get(), LwSciBufImageAttrKey_TopPadding, tPad);
    SET_ATTR(list.get(), LwSciBufImageAttrKey_BottomPadding, bPad);
    SET_ATTR(list.get(), LwSciBufImageAttrKey_LeftPadding, lPad);
    SET_ATTR(list.get(), LwSciBufImageAttrKey_RightPadding, rPad);
    SET_ATTR(list.get(), LwSciBufImageAttrKey_VprFlag, vprFlag);
    SET_ATTR(list.get(), LwSciBufImageAttrKey_PlaneCount, planeCount);
    SET_ATTR(list.get(), LwSciBufImageAttrKey_ImageCount, imageCount);
    SET_ATTR(list.get(), LwSciBufImageAttrKey_PlaneColorFormat, planeColorFmts);
    SET_ATTR(list.get(), LwSciBufImageAttrKey_PlaneColorStd, planeColorStds);
    SET_ATTR(list.get(), LwSciBufImageAttrKey_PlaneWidth, planeWidths);
    SET_ATTR(list.get(), LwSciBufImageAttrKey_PlaneHeight, planeHeights);
    SET_ATTR(list.get(), LwSciBufImageAttrKey_ScanType, planeScanType);

#if !defined(__x86_64__)
    LwSciBufHwEngine engine1{};
    engine1.engNamespace = LwSciBufHwEngine_TegraNamespaceId;
    LwSciBufHwEngCreateIdWithoutInstance(LwSciBufHwEngName_Vi,
                                         &engine1.rmModuleID);

    LwSciBufHwEngine engine2{};
    engine2.engNamespace = LwSciBufHwEngine_TegraNamespaceId;
    LwSciBufHwEngCreateIdWithoutInstance(LwSciBufHwEngName_Csi,
                                         &engine2.rmModuleID);

    LwSciBufHwEngine engine3{};
    engine3.engNamespace = LwSciBufHwEngine_TegraNamespaceId;
    LwSciBufHwEngCreateIdWithoutInstance(LwSciBufHwEngName_Vic,
                                         &engine3.rmModuleID);

    LwSciBufHwEngine engine4{};
    engine4.engNamespace = LwSciBufHwEngine_TegraNamespaceId;
    LwSciBufHwEngCreateIdWithoutInstance(LwSciBufHwEngName_Gpu,
                                         &engine4.rmModuleID);

    LwSciBufHwEngine engine5{};
    engine5.engNamespace = LwSciBufHwEngine_TegraNamespaceId;
    LwSciBufHwEngCreateIdWithoutInstance(LwSciBufHwEngName_MSENC,
                                         &engine5.rmModuleID);

    LwSciBufHwEngine engine6{};
    engine6.engNamespace = LwSciBufHwEngine_TegraNamespaceId;
    LwSciBufHwEngCreateIdWithoutInstance(LwSciBufHwEngName_LWDEC,
                                         &engine6.rmModuleID);

    LwSciBufHwEngine engine7{};
    engine7.engNamespace = LwSciBufHwEngine_TegraNamespaceId;
    LwSciBufHwEngCreateIdWithoutInstance(LwSciBufHwEngName_LWJPG,
                                         &engine7.rmModuleID);

    LwSciBufHwEngine engine8{};
    engine8.engNamespace = LwSciBufHwEngine_TegraNamespaceId;
    LwSciBufHwEngCreateIdWithoutInstance(LwSciBufHwEngName_PVA,
                                         &engine8.rmModuleID);

    LwSciBufHwEngine engine9{};
    engine9.engNamespace = LwSciBufHwEngine_TegraNamespaceId;
    LwSciBufHwEngCreateIdWithoutInstance(LwSciBufHwEngName_DLA,
                                         &engine9.rmModuleID);

    LwSciBufHwEngine engineArray[] = { engine1, engine2, engine3, engine4,
                    engine5, engine6, engine7, engine8, engine9 };
#else
    LwSciBufHwEngine engine{};
    engine.engNamespace = LwSciBufHwEngine_ResmanNamespaceId;
    engine.subEngineID = LW2080_ENGINE_TYPE_GRAPHICS;
    engine.rev.gpu.arch = LW2080_CTRL_MC_ARCH_INFO_ARCHITECTURE_GV100;

    LwSciBufHwEngine engineArray[] = { engine };
#endif

    SET_INTERNAL_ATTR(list.get(), LwSciBufInternalGeneralAttrKey_EngineArray,
                      engineArray);
}

/**
* Test case: Test that a unique identifier generated for each supported hardware
* engine
*/
TEST_F(HardwareEngine, GetEngineIdTest)
{
    int64_t id = 0;
    uint32_t instance = 0;

    setEngineList();

    for (auto const& engine: engineList) {
        ASSERT_EQ(LwSciBufHwEngCreateIdWithoutInstance(engine.engineName, &id),
                LwSciError_Success);
        ASSERT_EQ(engine.verifyEngineId(id), true);
    }

    {
        NEGATIVE_TEST();
        std::vector<EngineInfo> ilwalidEngineList;
        ilwalidEngineList.push_back(EngineInfo(LwSciBufHwEngName_Ilwalid, 0));

        for (auto const& engine: ilwalidEngineList) {
            ASSERT_EQ(LwSciBufHwEngCreateIdWithoutInstance(engine.engineName, &id),
                    LwSciError_BadParameter);
        }
    }

    for (auto const& engine: engineList) {
        ASSERT_EQ(LwSciBufHwEngCreateIdWithInstance(engine.engineName, instance,
                &id), LwSciError_Success);
        ASSERT_EQ(engine.verifyEngineId(id), true);
    }
}

/**
* Test case: Test to retrieve a hardware engine instance from a valid unique
* engine identifier
*/
TEST_F(HardwareEngine, GetEngineInstanceTest)
{
    int64_t engId = 4;
    uint32_t instance = 4;
    uint32_t desiredInstance = 0;

    ASSERT_EQ(LwSciBufHwEngGetInstanceFromId(engId, &instance),
            LwSciError_Success);
    ASSERT_EQ(instance, desiredInstance);
}

TEST_F(HardwareEngine, GetIlwalidEngineInstanceTest)
{
    int64_t engId = 0xFFFFFFFFFFFFFFFF;
    uint32_t instance = 4;

    {
        NEGATIVE_TEST();
        ASSERT_EQ(LwSciBufHwEngGetInstanceFromId(engId, &instance),
                LwSciError_BadParameter);
    }
}

/**
* Test case: Test to retrieve hardware engine name from a valid unique engine
* identifier
*/
TEST_F(HardwareEngine, GetEngineNameTest)
{
    LwSciBufHwEngName name;

    setEngineList();

    for (auto const& engine: engineList) {
        ASSERT_EQ(LwSciBufHwEngGetNameFromId(engine.engineID, &name),
                  LwSciError_Success);
        ASSERT_EQ(engine.verifyEngineName(name), true)
            << "Failed to verify engine name";
    }
}
