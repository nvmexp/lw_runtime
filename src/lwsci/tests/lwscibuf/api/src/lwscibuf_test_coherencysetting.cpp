/*
* Copyright (c) 2020-2021, LWPU CORPORATION. All rights reserved.
*
* LWPU Corporation and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from LWPU Corporation is strictly prohibited.
*/

#include "lwscibuf_basic_test.h"
#include "gtest/gtest.h"
#include <unordered_set>

// This is to avoid rtti compilation error in x86 build
#define GTEST_HAS_RTTI 0

class CacheCoherency : public LwSciBufBasicTest
{
public:
    virtual void SetUp() override
    {
        LwSciError error = LwSciError_Success;

        LwSciBufBasicTest::SetUp();

        listA = peer.createAttrList(&error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_NE(listA.get(), nullptr);

        listB = peer.createAttrList(&error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_NE(listB.get(), nullptr);
    }

    virtual void TearDown() override
    {
        LwSciBufBasicTest::TearDown();

        listA.reset();
        listB.reset();
    }

    bool verifyIsoEngine(LwSciBufAttrList reconciledList,
                         LwSciBufHwEngine desiredEngArray[],
                         int numberOfElements)
    {
        std::unordered_set<decltype(desiredEngArray[0].rmModuleID)>
            requestedEngineIds;
        for (int i = 0; i < numberOfElements; ++i) {
            requestedEngineIds.insert(desiredEngArray[i].rmModuleID);
        }

        LwSciBufInternalAttrKeyValuePair keyValuePair = {
            LwSciBufInternalGeneralAttrKey_EngineArray,
        };

        if (LwSciBufAttrListGetInternalAttrs(reconciledList, &keyValuePair,
            1) != LwSciError_Success) {
            return false;
        }

        std::unordered_set<decltype(desiredEngArray[0].rmModuleID)>
            reconciledEngineIds;
        const LwSciBufHwEngine* engineArray =
            (const LwSciBufHwEngine*)keyValuePair.value;
        for (int i = 0; i < keyValuePair.len / sizeof(LwSciBufHwEngine); ++i) {
            reconciledEngineIds.insert(engineArray[i].rmModuleID);
        }

        return (requestedEngineIds == reconciledEngineIds);
    }

    std::shared_ptr<LwSciBufAttrListRec> listA;
    std::shared_ptr<LwSciBufAttrListRec> listB;
};

/**
* Test case: Negative Test to validate NeedSWCacheCoherency attribute value
*/
TEST_F(CacheCoherency, CoherencySettingNegative)
{
    LwSciError error = LwSciError_Success;
    bool isReconciledListValid = false;

    LwSciBufType bufType = LwSciBufType_RawBuffer;
    uint64_t size = 128 * 1024;
    uint64_t align = 4 * 1024;
    bool needCpuAccess = true;
    bool enableCpuCache = true;
    bool cacheCoherencyFlag = false;

    SET_ATTR(listA.get(), LwSciBufGeneralAttrKey_Types, bufType);
    SET_ATTR(listA.get(), LwSciBufRawBufferAttrKey_Size, size);
    SET_ATTR(listA.get(), LwSciBufRawBufferAttrKey_Align, align);
    SET_ATTR(listA.get(), LwSciBufGeneralAttrKey_NeedCpuAccess, needCpuAccess);
    SET_ATTR(listA.get(), LwSciBufGeneralAttrKey_EnableCpuCache,
             enableCpuCache);

    SET_ATTR(listB.get(), LwSciBufGeneralAttrKey_Types, bufType);
    SET_ATTR(listB.get(), LwSciBufRawBufferAttrKey_Size, size);
    SET_ATTR(listB.get(), LwSciBufRawBufferAttrKey_Align, align);

    // Reconcile listA and listB
    auto reconciledList =
        LwSciBufPeer::attrListReconcile({listA.get(), listB.get()}, &error);
    ASSERT_EQ(error, LwSciError_Success);
    ASSERT_NE(reconciledList, nullptr);

    // Validate reconciled list
    ASSERT_EQ(LwSciBufPeer::validateReconciled({listA.get(), listB.get()},
                                               reconciledList.get(),
                                               &isReconciledListValid),
              LwSciError_Success);
    ASSERT_TRUE(isReconciledListValid);

    ASSERT_TRUE(LwSciBufPeer::verifyAttr(
        reconciledList.get(), LwSciBufGeneralAttrKey_Types, bufType));
    ASSERT_TRUE(LwSciBufPeer::verifyAttr(reconciledList.get(),
                                         LwSciBufRawBufferAttrKey_Size, size));
    ASSERT_TRUE(LwSciBufPeer::verifyAttr(
        reconciledList.get(), LwSciBufRawBufferAttrKey_Align, align));
    ASSERT_TRUE(LwSciBufPeer::verifyAttr(reconciledList.get(),
                                         LwSciBufGeneralAttrKey_NeedCpuAccess,
                                         needCpuAccess));
    ASSERT_TRUE(LwSciBufPeer::verifyAttr(reconciledList.get(),
                                         LwSciBufGeneralAttrKey_EnableCpuCache,
                                         enableCpuCache));
    ASSERT_TRUE(LwSciBufPeer::verifyAttr(
        reconciledList.get(), LwSciBufGeneralAttrKey_CpuNeedSwCacheCoherency,
        cacheCoherencyFlag));
}

/**
* Test case: Positive Test to validate NeedSWCacheCoherency attribute value
*/
#if BACKEND_RESMAN == 0
TEST_F(CacheCoherency, CoherencySettingPositive)
{
    LwSciError error = LwSciError_Success;
    bool isReconciledListValid = false;

    uint64_t size = 128 * 1024;
    uint64_t align = 4 * 1024;
    LwSciBufType bufType = LwSciBufType_RawBuffer;

    bool needCpuAccess = true;
    bool enableCpuCache = true;
    bool cacheCoherencyFlag = true;

    LwSciBufHwEngine engine1{};
    LwSciBufHwEngCreateIdWithoutInstance(LwSciBufHwEngName_Display,
                                         &engine1.rmModuleID);
    LwSciBufHwEngine engine2{};
    LwSciBufHwEngCreateIdWithoutInstance(LwSciBufHwEngName_Vi,
                                         &engine2.rmModuleID);
    LwSciBufHwEngine engineArray[] = {engine1, engine2};

    SET_INTERNAL_ATTR(listA.get(), LwSciBufInternalGeneralAttrKey_EngineArray,
                      engineArray);

    SET_ATTR(listA.get(), LwSciBufGeneralAttrKey_Types, bufType);
    SET_ATTR(listA.get(), LwSciBufRawBufferAttrKey_Size, size);
    SET_ATTR(listA.get(), LwSciBufRawBufferAttrKey_Align, align);
    SET_ATTR(listA.get(), LwSciBufGeneralAttrKey_NeedCpuAccess, needCpuAccess);
    SET_ATTR(listA.get(), LwSciBufGeneralAttrKey_EnableCpuCache,
             enableCpuCache);

    SET_ATTR(listB.get(), LwSciBufGeneralAttrKey_Types, bufType);
    SET_ATTR(listB.get(), LwSciBufRawBufferAttrKey_Size, size);
    SET_ATTR(listB.get(), LwSciBufRawBufferAttrKey_Align, align);

    // Reconcile listA and listB
    auto reconciledList =
        LwSciBufPeer::attrListReconcile({listA.get(), listB.get()}, &error);
    ASSERT_EQ(error, LwSciError_Success);
    ASSERT_NE(reconciledList, nullptr);

    // Validate reconciled list
    ASSERT_EQ(LwSciBufPeer::validateReconciled({listA.get(), listB.get()},
                                               reconciledList.get(),
                                               &isReconciledListValid),
              LwSciError_Success);
    ASSERT_TRUE(isReconciledListValid);

    ASSERT_TRUE(LwSciBufPeer::verifyAttr(reconciledList.get(),
                                         LwSciBufRawBufferAttrKey_Size, size));
    ASSERT_TRUE(LwSciBufPeer::verifyAttr(
        reconciledList.get(), LwSciBufRawBufferAttrKey_Align, align));
    ASSERT_TRUE(LwSciBufPeer::verifyAttr(
        reconciledList.get(), LwSciBufGeneralAttrKey_Types, bufType));
    ASSERT_TRUE(LwSciBufPeer::verifyAttr(reconciledList.get(),
                                         LwSciBufGeneralAttrKey_NeedCpuAccess,
                                         needCpuAccess));
    ASSERT_TRUE(LwSciBufPeer::verifyAttr(reconciledList.get(),
                                         LwSciBufGeneralAttrKey_EnableCpuCache,
                                         enableCpuCache));
    ASSERT_TRUE(LwSciBufPeer::verifyAttr(
        reconciledList.get(), LwSciBufGeneralAttrKey_CpuNeedSwCacheCoherency,
        cacheCoherencyFlag));

    ASSERT_TRUE(verifyIsoEngine(reconciledList.get(), engineArray, 2))
        << "Reconciled list verification failed";
}
#endif
