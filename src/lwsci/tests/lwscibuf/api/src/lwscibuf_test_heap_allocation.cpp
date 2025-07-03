/*
 * Copyright (c) 2020-2021, LWPU CORPORATION. All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#include "lwtegrahv.h"
#include "lwrm_memmgr_safe.h"
#include "lwscibuf_basic_test.h"

class HeapAllocation : public LwSciBufBasicTest
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

        for (auto const& list : {listA, listB}) {
            LwSciBufType bufType = LwSciBufType_Image;
            LwSciBufAttrValImageLayoutType layout =
                LwSciBufImage_PitchLinearType;
            uint32_t planeCount = 1U;
            LwSciBufAttrValColorFmt colorFmts[] = {LwSciColor_A8R8G8B8};
            uint32_t planeWidths[] = {640U};
            uint32_t planeHeights[] = {480U};
            bool cpuAccessFlag = false;

            SET_ATTR(list.get(), LwSciBufGeneralAttrKey_Types, bufType);
            SET_ATTR(list.get(), LwSciBufImageAttrKey_Layout, layout);
            SET_ATTR(list.get(), LwSciBufImageAttrKey_PlaneCount, planeCount);
            SET_ATTR(list.get(), LwSciBufImageAttrKey_PlaneColorFormat,
                     colorFmts);
            SET_ATTR(list.get(), LwSciBufImageAttrKey_PlaneWidth, planeWidths);
            SET_ATTR(list.get(), LwSciBufImageAttrKey_PlaneHeight,
                     planeHeights);
            SET_ATTR(list.get(), LwSciBufImageAttrKey_ScanType,
                     LwSciBufScan_ProgressiveType);
            SET_ATTR(list.get(), LwSciBufGeneralAttrKey_NeedCpuAccess,
                     cpuAccessFlag);
        }
    }

    virtual void TearDown() override
    {
        LwSciBufBasicTest::TearDown();

        listA.reset();
        listB.reset();
    }

    void checkHeapLocation(std::shared_ptr<LwSciBufObjRefRec> bufObj,
                           LwRmHeap expectedHeap)
    {
        LwSciBufRmHandle rmHandle = {0};
        uint64_t offset = 0U;
        uint64_t len = 0U;
        ASSERT_EQ(
            LwSciBufObjGetMemHandle(bufObj.get(), &rmHandle, &offset, &len),
            LwSciError_Success);

        LwRmMemHandleParams params = {};
        ASSERT_EQ(LwRmMemQueryHandleParams(rmHandle.memHandle,
                                           rmHandle.memHandle, &params,
                                           sizeof(params)),
                  LwError_Success);

        ASSERT_EQ(params.Heap, expectedHeap);
    }

    std::shared_ptr<LwSciBufAttrListRec> listA;
    std::shared_ptr<LwSciBufAttrListRec> listB;
};

TEST_F(HeapAllocation, ViAndDisplay)
{
    {
        LwSciBufHwEngine engine{};
        LwSciBufHwEngCreateIdWithoutInstance(LwSciBufHwEngName_Vi,
                                             &engine.rmModuleID);
        SET_INTERNAL_ATTR(listA.get(),
                          LwSciBufInternalGeneralAttrKey_EngineArray, engine);
    }

    {
        LwSciBufHwEngine engine{};
        LwSciBufHwEngCreateIdWithoutInstance(LwSciBufHwEngName_Display,
                                             &engine.rmModuleID);
        SET_INTERNAL_ATTR(listB.get(),
                          LwSciBufInternalGeneralAttrKey_EngineArray, engine);
    }

    LwSciError error = LwSciError_Success;
    auto bufObj =
        LwSciBufPeer::reconcileAndAllocate({listA.get(), listB.get()}, &error);
    ASSERT_EQ(error, LwSciError_Success);

    {
        // The heap allocation should occur from the IVC Carveout, which is
        // only supported on Safety
#if (LW_IS_SAFETY == 0)
        if (LwHvCheckOsNative() == 0) {
            checkHeapLocation(bufObj, LwRmHeap_ExternalCarveOut);
        } else {
            checkHeapLocation(bufObj, LwRmHeap_IOMMU);
        }
#else
        checkHeapLocation(bufObj, LwRmHeap_IVC);
#endif
    }
}

TEST_F(HeapAllocation, OnlyDisplay)
{
    {
        LwSciBufHwEngine engine{};
        LwSciBufHwEngCreateIdWithoutInstance(LwSciBufHwEngName_Display,
                                             &engine.rmModuleID);
        SET_INTERNAL_ATTR(listA.get(),
                          LwSciBufInternalGeneralAttrKey_EngineArray, engine);
    }

    {
        LwSciBufHwEngine engine{};
        LwSciBufHwEngCreateIdWithoutInstance(LwSciBufHwEngName_Display,
                                             &engine.rmModuleID);
        SET_INTERNAL_ATTR(listB.get(),
                          LwSciBufInternalGeneralAttrKey_EngineArray, engine);
    }

    LwSciError error = LwSciError_Success;
    auto bufObj =
        LwSciBufPeer::reconcileAndAllocate({listA.get(), listB.get()}, &error);
    ASSERT_EQ(error, LwSciError_Success);

    {
        // On HV, the heap allocation should occur from the External Carveout.
        // On native, it is from IOMMU.
        if (LwHvCheckOsNative() == 0) {
            checkHeapLocation(bufObj, LwRmHeap_ExternalCarveOut);
        } else {
            checkHeapLocation(bufObj, LwRmHeap_IOMMU);
        }
    }
}

TEST_F(HeapAllocation, OnlyVi)
{
    {
        LwSciBufHwEngine engine{};
        LwSciBufHwEngCreateIdWithoutInstance(LwSciBufHwEngName_Vi,
                                             &engine.rmModuleID);
        SET_INTERNAL_ATTR(listA.get(),
                          LwSciBufInternalGeneralAttrKey_EngineArray, engine);
    }

    {
        LwSciBufHwEngine engine{};
        LwSciBufHwEngCreateIdWithoutInstance(LwSciBufHwEngName_Vi,
                                             &engine.rmModuleID);
        SET_INTERNAL_ATTR(listB.get(),
                          LwSciBufInternalGeneralAttrKey_EngineArray, engine);
    }

    LwSciError error = LwSciError_Success;
    auto bufObj =
        LwSciBufPeer::reconcileAndAllocate({listA.get(), listB.get()}, &error);
    ASSERT_EQ(error, LwSciError_Success);

    {
        // The heap allocation should occur from the IVC Carveout, which is
        // only supported on Safety
#if (LW_IS_SAFETY == 0)
        if (LwHvCheckOsNative() == 0) {
            checkHeapLocation(bufObj, LwRmHeap_ExternalCarveOut);
        } else {
            checkHeapLocation(bufObj, LwRmHeap_IOMMU);
        }
#else
        checkHeapLocation(bufObj, LwRmHeap_IVC);
#endif
    }
}
