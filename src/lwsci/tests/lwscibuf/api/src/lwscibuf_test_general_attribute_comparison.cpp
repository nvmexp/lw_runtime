/*
 * Copyright (c) 2020-2022, LWPU CORPORATION. All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#include <memory>
#include <stdint.h>
#include <stdio.h>

#include "lwscibuf_basic_test.h"
#include "lwsci_igpu_or_dgpu_test.h"
#include "gtest/gtest.h"

// This is to avoid rtti compilation error in x86 build
#define GTEST_HAS_RTTI 0

class GeneralAttributes : public LwSciiGpuOrdGpuTest,
                            public LwSciBufBasicTest
{
protected:
    std::shared_ptr<LwSciBufAttrListRec> listA;
    std::shared_ptr<LwSciBufAttrListRec> listB;

    void setupList(LwSciBufType bufType1, LwSciBufType bufType2)
    {
        uint64_t size = (128U * 1024U);
        uint64_t align = (4U * 1024U);
        LwSciBufMemDomain memDomain = LwSciBufMemDomain_Sysmem;

#if !defined(__x86_64__)
        LwSciBufHwEngine engine1{};
        LwSciBufHwEngCreateIdWithoutInstance(LwSciBufHwEngName_PVA,
                                             &engine1.rmModuleID);
        LwSciBufHwEngine engine2{};
        LwSciBufHwEngCreateIdWithoutInstance(LwSciBufHwEngName_DLA,
                                             &engine2.rmModuleID);
        LwSciBufHwEngine engineArray[] = { engine1, engine2 };
#else
        LwSciBufHwEngine engine{};

        engine.engNamespace = LwSciBufHwEngine_ResmanNamespaceId;
        engine.subEngineID = LW2080_ENGINE_TYPE_GRAPHICS;
        engine.rev.gpu.arch = LW2080_CTRL_MC_ARCH_INFO_ARCHITECTURE_GV100;

        LwSciBufHwEngine engineArray[] = { engine };
#endif

        SET_ATTR(listA.get(), LwSciBufGeneralAttrKey_Types, bufType1);
        SET_ATTR(listA.get(), LwSciBufRawBufferAttrKey_Size, size);
        SET_ATTR(listA.get(), LwSciBufRawBufferAttrKey_Align, align);
        SET_INTERNAL_ATTR(listA.get(),
                          LwSciBufInternalGeneralAttrKey_MemDomainArray,
                          memDomain);
        SET_INTERNAL_ATTR(listA.get(),
                          LwSciBufInternalGeneralAttrKey_EngineArray,
                          engineArray);

        SET_ATTR(listB.get(), LwSciBufGeneralAttrKey_Types, bufType2);
        SET_ATTR(listB.get(), LwSciBufRawBufferAttrKey_Size, size);
        SET_ATTR(listB.get(), LwSciBufRawBufferAttrKey_Align, align);
        SET_ATTR(listB.get(), LwSciBufGeneralAttrKey_NeedCpuAccess, true);
        SET_ATTR(listB.get(), LwSciBufGeneralAttrKey_EnableCpuCache, true);

        uuId = testGpuId;

        SET_ATTR(listB.get(), LwSciBufGeneralAttrKey_GpuId, uuId);
        SET_INTERNAL_ATTR(listB.get(),
                          LwSciBufInternalGeneralAttrKey_MemDomainArray,
                          memDomain);
    }

    void SetUp() override
    {
        LwSciError error = LwSciError_Success;

        LwSciBufBasicTest::SetUp();
        LwSciiGpuOrdGpuTest::SetUp();

        listA = peer.createAttrList(&error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_NE(listA.get(), nullptr);

        listB = peer.createAttrList(&error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_NE(listB.get(), nullptr);
    }

    void TearDown() override
    {
        LwSciBufBasicTest::TearDown();
        LwSciiGpuOrdGpuTest::TearDown();

        listA.reset();
        listB.reset();
    }

    void reconcileAndValidate(void)
    {
        LwSciError error = LwSciError_Success;

        bool isReconciledListValid = false;
        // Reconcile listA and listB
        auto reconciledList =
            LwSciBufPeer::attrListReconcile({listA.get(), listB.get()}, &error);
        ASSERT_NE(reconciledList, nullptr);

        // Validate Reconciled
        ASSERT_EQ(LwSciBufPeer::validateReconciled({listA.get(), listB.get()},
                                                   reconciledList.get(),
                                                   &isReconciledListValid),
                  LwSciError_Success);
        ASSERT_TRUE(isReconciledListValid);
    }

    bool verifyGpuId(LwSciBufAttrList reconciledList, LwSciRmGpuId desiredgpuId,
                     int numberOfElements)
    {
        bool gpuIdSet = false;
        LwSciRmGpuId gpuId;
        LwSciBufAttrKeyValuePair keyValuePair = {
            LwSciBufGeneralAttrKey_GpuId,
        };
        if (LwSciBufAttrListGetAttrs(reconciledList, &keyValuePair, 1) !=
            LwSciError_Success) {
            return false;
        }

        gpuId = *(const LwSciRmGpuId*)keyValuePair.value;
        for (int i = 0; i < numberOfElements; i++) {
            if (gpuId.bytes[i] == desiredgpuId.bytes[i])
                gpuIdSet = true;
        }

        return gpuIdSet;
    }

    bool verifyEngineList(LwSciBufAttrList reconciledList,
                          LwSciBufHwEngine desiredEngArray[],
                          int numberOfElements)
    {
        bool engineSet = true;
        LwSciBufHwEngine* engArray;
        int numberOfGetElements = 0;
        LwSciBufInternalAttrKeyValuePair keyValuePair = {
            LwSciBufInternalGeneralAttrKey_EngineArray,
        };
        if (LwSciBufAttrListGetInternalAttrs(reconciledList, &keyValuePair,
                                             1) != LwSciError_Success) {
            return false;
        }

        numberOfGetElements = keyValuePair.len / sizeof(desiredEngArray[0]);
        if (numberOfGetElements != numberOfElements) {
            return false;
        }

        engArray = (LwSciBufHwEngine*)keyValuePair.value;
        for (int i = 0; i < numberOfElements; i++) {
#if !defined(__x86_64__)
            engineSet = ((engineSet) &&
                (engArray[i].rmModuleID == desiredEngArray[i].rmModuleID));
#else
            engineSet = ((engineSet) &&
                (engArray[i].engNamespace == desiredEngArray[i].engNamespace) &&
                (engArray[i].subEngineID == desiredEngArray[i].subEngineID) &&
                (engArray[i].rev.gpu.arch == desiredEngArray[i].rev.gpu.arch));
#endif
        }

        return engineSet;
    }

    LwSciRmGpuId uuId;
};

/**
* Test case: Test to verify that reconciliation of buffer type attribute is
*            performed according to same buffer type reconciliation policy.
**/
TEST_F(GeneralAttributes, SameBufferTypeReconciliation)
{
    setupList(LwSciBufType_RawBuffer, LwSciBufType_RawBuffer);

    LwSciError error = LwSciError_Success;
    bool isReconciledListValid = false;
    // Reconcile listA and listB
    auto reconciledList =
        LwSciBufPeer::attrListReconcile({listA.get(), listB.get()}, &error);
    ASSERT_NE(reconciledList, nullptr);

    // Validate Reconciled
    ASSERT_EQ(LwSciBufPeer::validateReconciled({listA.get(), listB.get()},
                                               reconciledList.get(),
                                               &isReconciledListValid),
              LwSciError_Success);
    ASSERT_TRUE(isReconciledListValid);

    ASSERT_TRUE(LwSciBufPeer::verifyAttr(reconciledList.get(),
                                         LwSciBufGeneralAttrKey_Types,
                                         LwSciBufType_RawBuffer))
        << "Same buffer type attribute verification failed";
}

/**
* Test case: Test to validate attribute MemoryDomain with EqualvaluePolicy
**/
TEST_F(GeneralAttributes, MemoryDomain)
{
    setupList(LwSciBufType_RawBuffer, LwSciBufType_RawBuffer);

    LwSciError error = LwSciError_Success;
    bool isReconciledListValid = false;
    // Reconcile listA and listB
    auto reconciledList =
        LwSciBufPeer::attrListReconcile({listA.get(), listB.get()}, &error);
    ASSERT_NE(reconciledList, nullptr);

    // Validate Reconciled
    ASSERT_EQ(LwSciBufPeer::validateReconciled({listA.get(), listB.get()},
                                               reconciledList.get(),
                                               &isReconciledListValid),
              LwSciError_Success);
    ASSERT_TRUE(isReconciledListValid);

    LwSciBufPeer::verifyInternalAttr(
        reconciledList.get(), LwSciBufInternalGeneralAttrKey_MemDomainArray,
        LwSciBufMemDomain_Sysmem);
}

/**
* Test case: Test to validate attribute NeedCpuAccess if any of the slots of all
*            the attribute lists that are owned by the reconciler in the
*            provided unreconciled attribute lists is set to a value equivalent of true
**/
TEST_F(GeneralAttributes, CpuAccessEnabled)
{
    setupList(LwSciBufType_RawBuffer, LwSciBufType_RawBuffer);

    LwSciError error = LwSciError_Success;
    bool isReconciledListValid = false;
    // Reconcile listA and listB
    auto reconciledList =
        LwSciBufPeer::attrListReconcile({listA.get(), listB.get()}, &error);
    ASSERT_NE(reconciledList, nullptr);

    // Validate Reconciled
    ASSERT_EQ(LwSciBufPeer::validateReconciled({listA.get(), listB.get()},
                                               reconciledList.get(),
                                               &isReconciledListValid),
              LwSciError_Success);
    ASSERT_TRUE(isReconciledListValid);

    ASSERT_TRUE(LwSciBufPeer::verifyAttr(
        reconciledList.get(), LwSciBufGeneralAttrKey_NeedCpuAccess, true))
        << "Memory domain attribute verification failed";
}

/**
* Test case: Test to validate attribute NeedCpuAccess if none of the slots of
*            all the attribute lists that are owned by the reconciler in the provided
*            unreconciled attribute lists is set to a value equivalent of true.
**/
TEST_F(GeneralAttributes, CpuAccessDisabled)
{
    setupList(LwSciBufType_RawBuffer, LwSciBufType_RawBuffer);

    LwSciError error = LwSciError_Success;
    bool isReconciledListValid = false;
    // Reconcile listA and listB
    auto reconciledList =
        LwSciBufPeer::attrListReconcile({listA.get(), listB.get()}, &error);
    ASSERT_NE(reconciledList, nullptr);

    // Validate Reconciled
    ASSERT_EQ(LwSciBufPeer::validateReconciled({listA.get(), listB.get()},
                                               reconciledList.get(),
                                               &isReconciledListValid),
              LwSciError_Success);
    ASSERT_TRUE(isReconciledListValid);

    ASSERT_FALSE(LwSciBufPeer::verifyAttr(
        reconciledList.get(), LwSciBufGeneralAttrKey_NeedCpuAccess, false))
        << "Memory domain attribute verification failed";
}

/**
* Test case: Test to validate attribute NeedCpuCaching if any of the slots of
*            the provided unreconciled attribute lists has the attribute value set to
*            equivalent of true.
**/
TEST_F(GeneralAttributes, CpuCacheEnabled)
{
    setupList(LwSciBufType_RawBuffer, LwSciBufType_RawBuffer);

    LwSciError error = LwSciError_Success;
    bool isReconciledListValid = false;
    // Reconcile listA and listB
    auto reconciledList =
        LwSciBufPeer::attrListReconcile({listA.get(), listB.get()}, &error);
    ASSERT_NE(reconciledList, nullptr);

    // Validate Reconciled
    ASSERT_EQ(LwSciBufPeer::validateReconciled({listA.get(), listB.get()},
                                               reconciledList.get(),
                                               &isReconciledListValid),
              LwSciError_Success);
    ASSERT_TRUE(isReconciledListValid);

    ASSERT_TRUE(LwSciBufPeer::verifyAttr(
        reconciledList.get(), LwSciBufGeneralAttrKey_EnableCpuCache, true))
        << "Reconciled list verification failed";
}

/**
* Test case: Test to validate attribute NeedCpuCaching if none of the slots of
*            the provided unreconciled attribute lists has the attribute value set to
*            equivalent of true.
**/
TEST_F(GeneralAttributes, CpuCacheDisabled)
{
    setupList(LwSciBufType_RawBuffer, LwSciBufType_RawBuffer);

    LwSciError error = LwSciError_Success;
    bool isReconciledListValid = false;
    // Reconcile listA and listB
    auto reconciledList =
        LwSciBufPeer::attrListReconcile({listA.get(), listB.get()}, &error);
    ASSERT_NE(reconciledList, nullptr);

    // Validate Reconciled
    ASSERT_EQ(LwSciBufPeer::validateReconciled({listA.get(), listB.get()},
                                               reconciledList.get(),
                                               &isReconciledListValid),
              LwSciError_Success);
    ASSERT_TRUE(isReconciledListValid);

    ASSERT_FALSE(LwSciBufPeer::verifyAttr(
        reconciledList.get(), LwSciBufGeneralAttrKey_EnableCpuCache, false))
        << "Reconciled list verification failed";
}

/**
* Test case: Test to validate attribute GpuId with ValueSetPolicy
**/
TEST_F(GeneralAttributes, GpuId)
{
    setupList(LwSciBufType_RawBuffer, LwSciBufType_RawBuffer);

    LwSciError error = LwSciError_Success;
    bool isReconciledListValid = false;
    // Reconcile listA and listB
    auto reconciledList =
        LwSciBufPeer::attrListReconcile({listA.get(), listB.get()}, &error);
    ASSERT_NE(reconciledList, nullptr);

    // Validate Reconciled
    ASSERT_EQ(LwSciBufPeer::validateReconciled({listA.get(), listB.get()},
                                               reconciledList.get(),
                                               &isReconciledListValid),
              LwSciError_Success);
    ASSERT_TRUE(isReconciledListValid);

    ASSERT_TRUE(verifyGpuId(reconciledList.get(), uuId, 16))
        << "Reconciled list verification failed";
}

/**
* Test case: Test to validate attribute EngineList with ValueSetPolicy
**/
TEST_F(GeneralAttributes, EngineList)
{
    setupList(LwSciBufType_RawBuffer, LwSciBufType_RawBuffer);

#if !defined(__x86_64__)
        LwSciBufHwEngine engine1{};
        LwSciBufHwEngCreateIdWithoutInstance(LwSciBufHwEngName_PVA,
                                             &engine1.rmModuleID);
        LwSciBufHwEngine engine2{};
        LwSciBufHwEngCreateIdWithoutInstance(LwSciBufHwEngName_DLA,
                                             &engine2.rmModuleID);
        LwSciBufHwEngine engineArray[] = { engine1, engine2 };
        int numEngines = 2;
#else
        LwSciBufHwEngine engine{};

        engine.engNamespace = LwSciBufHwEngine_ResmanNamespaceId;
        engine.subEngineID = LW2080_ENGINE_TYPE_GRAPHICS;
        engine.rev.gpu.arch = LW2080_CTRL_MC_ARCH_INFO_ARCHITECTURE_GV100;

        LwSciBufHwEngine engineArray[] = { engine };
        int numEngines = 1;
#endif

    LwSciError error = LwSciError_Success;
    bool isReconciledListValid = false;
    // Reconcile listA and listB
    auto reconciledList =
        LwSciBufPeer::attrListReconcile({listA.get(), listB.get()}, &error);
    ASSERT_NE(reconciledList, nullptr);

    // Validate Reconciled
    ASSERT_EQ(LwSciBufPeer::validateReconciled({listA.get(), listB.get()},
                                               reconciledList.get(),
                                               &isReconciledListValid),
              LwSciError_Success);
    ASSERT_TRUE(isReconciledListValid);

    ASSERT_TRUE(verifyEngineList(reconciledList.get(), engineArray, numEngines))
        << "Reconciled list verification failed";
}

/**
* Test case: Test to verify that reconciliation of buffer type attribute is
*            performed according to different buffer type reconciliation policy.
*/
TEST_F(GeneralAttributes, DISABLED_DifferentBufferTypeReconciliation)
{
    setupList(LwSciBufType_Tensor, LwSciBufType_Image);

    LwSciError error = LwSciError_Success;
    bool isReconciledListValid = false;

    // Reconcile listA and listB
    auto reconciledList =
        LwSciBufPeer::attrListReconcile({listA.get(), listB.get()}, &error);
    ASSERT_NE(reconciledList, nullptr);

    // Validate Reconciled
    ASSERT_EQ(LwSciBufPeer::validateReconciled({listA.get(), listB.get()},
                                               reconciledList.get(),
                                               &isReconciledListValid),
              LwSciError_Success);
    ASSERT_TRUE(isReconciledListValid);

    ASSERT_TRUE(LwSciBufPeer::verifyAttr(reconciledList.get(),
                                         LwSciBufGeneralAttrKey_Types,
                                         LwSciBufType_RawBuffer))
        << "Same buffer type attribute verification failed";
}
