/*
 * Copyright (c) 2020-2022 LWPU Corporation.  All Rights Reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property and
 * proprietary rights in and to this software and related documentation.  Any
 * use, reproduction, disclosure or distribution of this software and related
 * documentation without an express license agreement from LWPU Corporation
 * is strictly prohibited.
 */

#include "lwscisync_basic_test.h"
#include "lwscisync_test_attribute_list.h"
#include "lwscisync_interprocess_test.h"

#include "gmock/gmock-matchers.h"

/* Declare new tests with this macro to make sure each test case has Jama ID */
#define LWSCISYNC_ATTRIBUTE_LIST_RECONCILE_TEST(testSuite, testName, JamaID)   \
    class _##testSuite##JamaID : public LwSciSyncAttrListTest<JamaID>          \
    {                                                                          \
    };                                                                         \
    TEST_F(_##testSuite##JamaID, testName)

/**
 * @jama{15643568} Successful reconciliation
 *
 * @verifies @jama{13561817}
 *
 * This test checks that we successfully return a new reconciled attribute
 * list.
 */
LWSCISYNC_ATTRIBUTE_LIST_RECONCILE_TEST(
    AttributeListReconciliation,
    SuccessfulReconciliation,
    15643568)
{
    LwSciError error = LwSciError_Success;
    auto signalerAttrList = peer.createAttrList();
    auto waiterAttrList = peer.createAttrList();

    // Set up Signaler Attribute List
    {
        ASSERT_EQ(LwSciSyncAttrListSetAttrs(
                  signalerAttrList.get(),
                  LwSciSyncPeer::attrs.cpuSignaler.data(),
                  LwSciSyncPeer::attrs.cpuSignaler.size()),
              LwSciError_Success);

        ASSERT_EQ(LwSciSyncAttrListSetInternalAttrs(
                      signalerAttrList.get(),
                      LwSciSyncPeer::attrs.signalerPlatformDefaultAttrs.data(),
                      LwSciSyncPeer::attrs.signalerPlatformDefaultAttrs.size()),
                  LwSciError_Success);
    }

    // Set up Waiter Attribute List
    {
        ASSERT_EQ(LwSciSyncAttrListSetAttrs(
                  waiterAttrList.get(),
                  LwSciSyncPeer::attrs.cpuWaiter.data(),
                  LwSciSyncPeer::attrs.cpuWaiter.size()),
              LwSciError_Success);

        ASSERT_EQ(LwSciSyncAttrListSetInternalAttrs(
                      waiterAttrList.get(),
                      LwSciSyncPeer::attrs.waiterPlatformDefaultAttrs.data(),
                      LwSciSyncPeer::attrs.waiterPlatformDefaultAttrs.size()),
                  LwSciError_Success);
    }

    // Assert on LwSciSyncAttrListReconcile
    {
        auto attrLists = {
            signalerAttrList.get(),
            waiterAttrList.get()
        };
        auto newReconciledList =
            LwSciSyncPeer::reconcileLists(attrLists, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_NE(newReconciledList.get(), nullptr);

        LwSciSyncPeer::verifyAttr(newReconciledList.get(),
                                  LwSciSyncAttrKey_NeedCpuAccess, true);
        LwSciSyncPeer::verifyAttr(newReconciledList.get(),
                                  LwSciSyncAttrKey_ActualPerm,
                                  LwSciSyncAccessPerm_WaitSignal);
        LwSciSyncPeer::verifyInternalAttr(
            newReconciledList.get(),
            LwSciSyncInternalAttrKey_SignalerPrimitiveInfo,
            LwSciSyncPeer::attrs.defaultPlatformPrimitive);
        LwSciSyncPeer::verifyInternalAttr(
            newReconciledList.get(),
            LwSciSyncInternalAttrKey_WaiterPrimitiveInfo,
            LwSciSyncPeer::attrs.defaultPlatformPrimitive);
        LwSciSyncPeer::verifyInternalAttr(
            newReconciledList.get(),
            LwSciSyncInternalAttrKey_SignalerPrimitiveCount, 1);

        // Then assert that the attribute list is reconciled
        bool isReconciled = true;
        ASSERT_EQ(LwSciSyncPeer::validateReconciled(
                      {signalerAttrList.get(), waiterAttrList.get()},
                      newReconciledList.get(), &isReconciled),
                  LwSciError_Success);
        ASSERT_TRUE(isReconciled);
    }

    // Assert on LwSciSyncAttrListReconcileAndObjAlloc
    {
        // Need not free
        LwSciSyncAttrList newReconciledList = nullptr;
        auto attrLists = {
            signalerAttrList.get(),
            waiterAttrList.get()
        };

        auto syncObj = LwSciSyncPeer::reconcileAndAllocate(attrLists, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_NE(syncObj.get(), nullptr);

        ASSERT_EQ(LwSciSyncObjGetAttrList(syncObj.get(), &newReconciledList),
                  LwSciError_Success);

        LwSciSyncPeer::verifyAttr(newReconciledList,
                LwSciSyncAttrKey_NeedCpuAccess, true);
        LwSciSyncPeer::verifyAttr(newReconciledList,
                LwSciSyncAttrKey_ActualPerm,
                LwSciSyncAccessPerm_WaitSignal);
        LwSciSyncPeer::verifyInternalAttr(newReconciledList,
                LwSciSyncInternalAttrKey_SignalerPrimitiveInfo,
                LwSciSyncPeer::attrs.defaultPlatformPrimitive);
        LwSciSyncPeer::verifyInternalAttr(newReconciledList,
                LwSciSyncInternalAttrKey_WaiterPrimitiveInfo,
                LwSciSyncPeer::attrs.defaultPlatformPrimitive);
        LwSciSyncPeer::verifyInternalAttr(newReconciledList,
                LwSciSyncInternalAttrKey_SignalerPrimitiveCount,
                1);

        // Then assert that the attribute list is reconciled
        bool isReconciled = true;
        ASSERT_EQ(
                LwSciSyncPeer::validateReconciled(
                    {signalerAttrList.get(), waiterAttrList.get()},
                    newReconciledList, &isReconciled),
                LwSciError_Success);
        ASSERT_TRUE(isReconciled);
    }
}

/**
 * @jama{15644470} NeedCpuAccess reconciliation: True
 *
 * @verifies @jama{13561819}
 *
 * This test checks that we successfully reconcile NeedCpuAccess to True if
 * any of the input attribute lists has their NeedCpuAccess attribute set to
 * True.
 */
LWSCISYNC_ATTRIBUTE_LIST_RECONCILE_TEST(
    AttributeListReconciliation,
    SuccessNeedCpuAccess_True,
    15644470)
{
    LwSciError error = LwSciError_Success;
    auto signalerAttrList = peer.createAttrList();
    auto waiterAttrList = peer.createAttrList();

    // Set up Signaler Attribute List
    {
        ASSERT_EQ(LwSciSyncAttrListSetAttrs(
                  signalerAttrList.get(),
                  LwSciSyncPeer::attrs.cpuSignaler.data(),
                  LwSciSyncPeer::attrs.cpuSignaler.size()),
              LwSciError_Success);

        ASSERT_EQ(LwSciSyncAttrListSetInternalAttrs(
                      signalerAttrList.get(),
                      LwSciSyncPeer::attrs.signalerPlatformDefaultAttrs.data(),
                      LwSciSyncPeer::attrs.signalerPlatformDefaultAttrs.size()),
                  LwSciError_Success);
    }

    // Set up Waiter Attribute List
    {
        ASSERT_EQ(LwSciSyncAttrListSetAttrs(
                  waiterAttrList.get(),
                  LwSciSyncPeer::attrs.waiter.data(),
                  LwSciSyncPeer::attrs.waiter.size()),
              LwSciError_Success);

        ASSERT_EQ(LwSciSyncAttrListSetInternalAttrs(
                      waiterAttrList.get(),
                      LwSciSyncPeer::attrs.waiterPlatformDefaultAttrs.data(),
                      LwSciSyncPeer::attrs.waiterPlatformDefaultAttrs.size()),
                  LwSciError_Success);
    }

    // Assert on LwSciSyncAttrListReconcile
    {
        auto attrLists = {
            signalerAttrList.get(),
            waiterAttrList.get()
        };
        auto newReconciledList =
            LwSciSyncPeer::reconcileLists(attrLists, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_NE(newReconciledList.get(), nullptr);

        LwSciSyncPeer::verifyAttr(newReconciledList.get(),
                                  LwSciSyncAttrKey_NeedCpuAccess, true);
    }

    // Assert on LwSciSyncAttrListReconcileAndObjAlloc
    {
        // Need not free
        LwSciSyncAttrList newReconciledList = nullptr;
        auto attrLists = {
            signalerAttrList.get(),
            waiterAttrList.get()
        };

        auto syncObj = LwSciSyncPeer::reconcileAndAllocate(attrLists, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_NE(syncObj.get(), nullptr);

        ASSERT_EQ(LwSciSyncObjGetAttrList(syncObj.get(), &newReconciledList),
                  LwSciError_Success);

        LwSciSyncPeer::verifyAttr(newReconciledList,
                LwSciSyncAttrKey_NeedCpuAccess, true);
    }
}

/**
 * @jama{15644492} NeedCpuAccess reconciliation: False
 *
 * @verifies @jama{13561819}
 *
 * This test checks that we successfully reconcile NeedCpuAccess to False if
 * none of the input attribute lists has their NeedCpuAccess attribute set to
 * True.
 */
LWSCISYNC_ATTRIBUTE_LIST_RECONCILE_TEST(
    AttributeListReconciliation,
    SuccessNeedCpuAccess_False,
    15644492)
{
    LwSciError error = LwSciError_Success;
    auto signalerAttrList = peer.createAttrList();
    auto waiterAttrList = peer.createAttrList();

    // Set up Signaler Attribute List
    {
        ASSERT_EQ(LwSciSyncAttrListSetAttrs(
                  signalerAttrList.get(),
                  LwSciSyncPeer::attrs.signaler.data(),
                  LwSciSyncPeer::attrs.signaler.size()),
              LwSciError_Success);

        ASSERT_EQ(LwSciSyncAttrListSetInternalAttrs(
                      signalerAttrList.get(),
                      LwSciSyncPeer::attrs.signalerPlatformDefaultAttrs.data(),
                      LwSciSyncPeer::attrs.signalerPlatformDefaultAttrs.size()),
                  LwSciError_Success);
    }

    // Set up Waiter Attribute List
    {
        ASSERT_EQ(LwSciSyncAttrListSetAttrs(
                  waiterAttrList.get(),
                  LwSciSyncPeer::attrs.waiter.data(),
                  LwSciSyncPeer::attrs.waiter.size()),
              LwSciError_Success);

        ASSERT_EQ(LwSciSyncAttrListSetInternalAttrs(
                      waiterAttrList.get(),
                      LwSciSyncPeer::attrs.waiterPlatformDefaultAttrs.data(),
                      LwSciSyncPeer::attrs.waiterPlatformDefaultAttrs.size()),
                  LwSciError_Success);
    }

    // Assert on LwSciSyncAttrListReconcile
    {
        auto newReconciledList = LwSciSyncPeer::reconcileLists(
            {signalerAttrList.get(), waiterAttrList.get()}, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_NE(newReconciledList.get(), nullptr);

        // Assert on result values
        LwSciSyncPeer::verifyAttr(newReconciledList.get(),
                                  LwSciSyncAttrKey_NeedCpuAccess, false);
    }

    // Assert on LwSciSyncAttrListReconcileAndObjAlloc
    {
        // Need not free
        LwSciSyncAttrList newReconciledList = nullptr;
        auto attrLists = {
            signalerAttrList.get(),
            waiterAttrList.get()
        };

        auto syncObj = LwSciSyncPeer::reconcileAndAllocate(attrLists, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_NE(syncObj.get(), nullptr);
        ASSERT_EQ(LwSciSyncObjGetAttrList(syncObj.get(), &newReconciledList),
                  LwSciError_Success);

        LwSciSyncPeer::verifyAttr(newReconciledList,
                LwSciSyncAttrKey_NeedCpuAccess, false);
    }
}

/**
 * @jama{15644504} ActualPerm reconciliation
 *
 * @verifies @jama{13561821}
 *
 * This test checks that we successfully reconcile the ActualPerm attribute
 * to the supremum of permissions in RequiredPerm attributes of the input
 * attribute lists.
 *
 */
LWSCISYNC_ATTRIBUTE_LIST_RECONCILE_TEST(
    AttributeListReconciliation,
    SuccessActualPerm,
    15644504)
{
    LwSciError error = LwSciError_Success;
    auto signalerAttrList = peer.createAttrList();
    auto waiterAttrList = peer.createAttrList();

    // Set up Signaler Attribute List
    {
        ASSERT_EQ(LwSciSyncAttrListSetAttrs(
                  signalerAttrList.get(),
                  LwSciSyncPeer::attrs.cpuSignaler.data(),
                  LwSciSyncPeer::attrs.cpuSignaler.size()),
              LwSciError_Success);
    }

    // Set up Waiter Attribute List
    {
        ASSERT_EQ(LwSciSyncAttrListSetAttrs(
                  waiterAttrList.get(),
                  LwSciSyncPeer::attrs.cpuWaiter.data(),
                  LwSciSyncPeer::attrs.cpuWaiter.size()),
              LwSciError_Success);
    }

    // Assert on LwSciSyncAttrListReconcile
    {
        auto attrLists = {signalerAttrList.get(), waiterAttrList.get()};
        auto newReconciledList =
            LwSciSyncPeer::reconcileLists(attrLists, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_NE(newReconciledList.get(), nullptr);

        LwSciSyncPeer::verifyAttr(newReconciledList.get(),
                                  LwSciSyncAttrKey_NeedCpuAccess, true);
        LwSciSyncPeer::verifyAttr(newReconciledList.get(),
                                  LwSciSyncAttrKey_ActualPerm,
                                  LwSciSyncAccessPerm_WaitSignal);
    }
    // Assert on LwSciSyncAttrListReconcileAndObjAlloc
    {
        // Need not free
        LwSciSyncAttrList newReconciledList = nullptr;
        auto attrLists = {
            signalerAttrList.get(),
            waiterAttrList.get(),
        };

        auto syncObj = LwSciSyncPeer::reconcileAndAllocate(attrLists, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_NE(syncObj.get(), nullptr);

        ASSERT_EQ(LwSciSyncObjGetAttrList(syncObj.get(), &newReconciledList),
                  LwSciError_Success);

        LwSciSyncPeer::verifyAttr(newReconciledList,
                LwSciSyncAttrKey_NeedCpuAccess, true);
        LwSciSyncPeer::verifyAttr(newReconciledList,
                LwSciSyncAttrKey_ActualPerm,
                LwSciSyncAccessPerm_WaitSignal);
    }
}

/**
 * @jama{15644506} WaiterContextInsensitiveFenceExports reconciliation: True
 *
 * @verifies @jama{13561823}
 *
 * This test checks that we successfully reconcile the
 * WaiterContextInsensitiveFenceExports to True if any of the input attribute
 * lists has WaiterContextInsensitiveFenceExports set to True.
 *
 */
LWSCISYNC_ATTRIBUTE_LIST_RECONCILE_TEST(
    AttributeListReconciliation,
    ReconciliationWaiterContextInsensitiveFenceExports_True,
    15644506)
{
    LwSciError error = LwSciError_Success;
    auto signalerAttrList = peer.createAttrList();
    auto waiterAttrListA = peer.createAttrList();
    auto waiterAttrListB = peer.createAttrList();

    // Set up Signaler Attribute List
    {
        ASSERT_EQ(LwSciSyncAttrListSetAttrs(
                  signalerAttrList.get(),
                  LwSciSyncPeer::attrs.cpuSignaler.data(),
                  LwSciSyncPeer::attrs.cpuSignaler.size()),
              LwSciError_Success);
    }

    // Set up Waiter A Attribute List
    {
        ASSERT_EQ(LwSciSyncAttrListSetAttrs(
                  waiterAttrListA.get(),
                  LwSciSyncPeer::attrs.cpuWaiter.data(),
                  LwSciSyncPeer::attrs.cpuWaiter.size()),
              LwSciError_Success);

        ASSERT_EQ(
                LwSciSyncPeer::setAttr(waiterAttrListA.get(),
                    LwSciSyncAttrKey_WaiterContextInsensitiveFenceExports,
                    true),
                LwSciError_Success);
    }

    // Set up Waiter B Attribute List
    {
        ASSERT_EQ(LwSciSyncAttrListSetAttrs(
                  waiterAttrListB.get(),
                  LwSciSyncPeer::attrs.cpuWaiter.data(),
                  LwSciSyncPeer::attrs.cpuWaiter.size()),
              LwSciError_Success);

        ASSERT_EQ(
                LwSciSyncPeer::setAttr(waiterAttrListB.get(),
                    LwSciSyncAttrKey_WaiterContextInsensitiveFenceExports,
                    false),
                LwSciError_Success);
    }

    // Assert on LwSciSyncAttrListReconcile
    {
        auto attrLists = {
            signalerAttrList.get(),
            waiterAttrListA.get(),
            waiterAttrListB.get()
        };
        auto newReconciledList =
            LwSciSyncPeer::reconcileLists(attrLists, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_NE(newReconciledList.get(), nullptr);

        LwSciSyncPeer::verifyAttr(newReconciledList.get(),
                                  LwSciSyncAttrKey_NeedCpuAccess, true);
        LwSciSyncPeer::verifyAttr(
            newReconciledList.get(),
            LwSciSyncAttrKey_WaiterContextInsensitiveFenceExports, true);
    }
    // Assert on LwSciSyncAttrListReconcileAndObjAlloc
    {
        // Need not free
        LwSciSyncAttrList newReconciledList = nullptr;
        auto attrLists = {
            signalerAttrList.get(),
            waiterAttrListA.get(),
            waiterAttrListB.get()
        };

        auto syncObj = LwSciSyncPeer::reconcileAndAllocate(attrLists, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_NE(syncObj.get(), nullptr);

        ASSERT_EQ(LwSciSyncObjGetAttrList(syncObj.get(), &newReconciledList),
                  LwSciError_Success);

        LwSciSyncPeer::verifyAttr(newReconciledList,
                LwSciSyncAttrKey_NeedCpuAccess, true);
        LwSciSyncPeer::verifyAttr(newReconciledList,
                LwSciSyncAttrKey_WaiterContextInsensitiveFenceExports,
                true);
    }
}

/**
 * @jama{15644632} WaiterContextInsensitiveFenceExports reconciliation: False
 *
 * @verifies @jama{13561823}
 *
 * This test checks that we successfully reconcile the
 * WaiterContextInsensitiveFenceExports to False if none of the input attribute
 * lists has WaiterContextInsensitiveFenceExports set to True.
 *
 */
LWSCISYNC_ATTRIBUTE_LIST_RECONCILE_TEST(
    AttributeListReconciliation,
    ReconciliationWaiterContextInsensitiveFenceExports_False,
    15644632)
{
    LwSciError error = LwSciError_Success;
    auto signalerAttrList = peer.createAttrList();
    auto waiterAttrListA = peer.createAttrList();
    auto waiterAttrListB = peer.createAttrList();

    // Set up Signaler Attribute List
    {
        ASSERT_EQ(
                LwSciSyncAttrListSetAttrs(
                    signalerAttrList.get(),
                    LwSciSyncPeer::attrs.cpuSignaler.data(),
                    LwSciSyncPeer::attrs.cpuSignaler.size()),
                LwSciError_Success);
    }

    // Set up Waiter A Attribute List
    {
        ASSERT_EQ(LwSciSyncAttrListSetAttrs(
                  waiterAttrListA.get(),
                  LwSciSyncPeer::attrs.cpuWaiter.data(),
                  LwSciSyncPeer::attrs.cpuWaiter.size()),
              LwSciError_Success);

        ASSERT_EQ(
                LwSciSyncPeer::setAttr(
                    waiterAttrListA.get(),
                    LwSciSyncAttrKey_WaiterContextInsensitiveFenceExports,
                    false),
                LwSciError_Success);
    }

    // Set up Waiter B Attribute List
    {
        ASSERT_EQ(LwSciSyncAttrListSetAttrs(
                  waiterAttrListB.get(),
                  LwSciSyncPeer::attrs.cpuWaiter.data(),
                  LwSciSyncPeer::attrs.cpuWaiter.size()),
              LwSciError_Success);

        ASSERT_EQ(
                LwSciSyncPeer::setAttr(
                    waiterAttrListB.get(),
                    LwSciSyncAttrKey_WaiterContextInsensitiveFenceExports,
                    false),
                LwSciError_Success);
    }

    // Assert on LwSciSyncAttrListReconcile
    {
        auto attrLists = {signalerAttrList.get(), waiterAttrListA.get(),
                          waiterAttrListB.get()};

        auto newReconciledList =
            LwSciSyncPeer::reconcileLists(attrLists, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_NE(newReconciledList.get(), nullptr);

        // Assert on results
        LwSciSyncPeer::verifyAttr(newReconciledList.get(),
                                  LwSciSyncAttrKey_NeedCpuAccess, true);
        LwSciSyncPeer::verifyAttr(
            newReconciledList.get(),
            LwSciSyncAttrKey_WaiterContextInsensitiveFenceExports, false);
    }

    // Assert on LwSciSyncAttrListReconcileAndObjAlloc
    {
        // Need not free
        LwSciSyncAttrList newReconciledList = nullptr;
        auto attrLists = {signalerAttrList.get(), waiterAttrListA.get(),
                          waiterAttrListB.get()};

        auto syncObj = LwSciSyncPeer::reconcileAndAllocate(attrLists, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_NE(syncObj.get(), nullptr);

        ASSERT_EQ(LwSciSyncObjGetAttrList(syncObj.get(), &newReconciledList),
                  LwSciError_Success);

        // Assert on results
        LwSciSyncPeer::verifyAttr(newReconciledList,
                LwSciSyncAttrKey_NeedCpuAccess, true);
        LwSciSyncPeer::verifyAttr(newReconciledList,
                LwSciSyncAttrKey_WaiterContextInsensitiveFenceExports,
                false);
    }
}

/**
 * @jama{15644652} PrimitiveInfo reconciliation: NeedCpuAccess False
 *
 * @verifies @jama{13561825}
 *
 * This test checks that we successfully reconcile the
 * {Signaler,Waiter}PrimitiveInfo key to a primitive present in all of the
 * input attribute lists, when NeedCpuAccess is set to False.
 *
 */
LWSCISYNC_ATTRIBUTE_LIST_RECONCILE_TEST(
    AttributeListReconciliation,
    ReconciliationPrimitiveInfo_NeedCpuAccess_False,
    15644652)
{
    LwSciError error = LwSciError_Success;
    auto signalerAttrList = peer.createAttrList();
    auto waiterAttrList = peer.createAttrList();

    // Set up Signaler Attribute List
    {
        ASSERT_EQ(LwSciSyncAttrListSetAttrs(
                  signalerAttrList.get(), LwSciSyncPeer::attrs.signaler.data(),
                  LwSciSyncPeer::attrs.signaler.size()),
              LwSciError_Success);

        ASSERT_EQ(LwSciSyncAttrListSetInternalAttrs(
                      signalerAttrList.get(),
                      LwSciSyncPeer::attrs.umdInternalAttrs.data(),
                      LwSciSyncPeer::attrs.umdInternalAttrs.size()),
                  LwSciError_Success);
    }

    // Set up Waiter Attribute List
    {
        ASSERT_EQ(LwSciSyncAttrListSetAttrs(
                  waiterAttrList.get(), LwSciSyncPeer::attrs.waiter.data(),
                  LwSciSyncPeer::attrs.waiter.size()),
              LwSciError_Success);

        ASSERT_EQ(LwSciSyncAttrListSetInternalAttrs(
                      waiterAttrList.get(),
                      LwSciSyncPeer::attrs.waiterPlatformDefaultAttrs.data(),
                      LwSciSyncPeer::attrs.waiterPlatformDefaultAttrs.size()),
                  LwSciError_Success);
    }

    // Assert on LwSciSyncAttrListReconcile
    {
        auto attrLists = {signalerAttrList.get(), waiterAttrList.get()};
        auto newReconciledList =
            LwSciSyncPeer::reconcileLists(attrLists, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_NE(newReconciledList.get(), nullptr);

        // Assert on result values
        LwSciSyncPeer::verifyAttr(newReconciledList.get(),
                                  LwSciSyncAttrKey_NeedCpuAccess, false);
        LwSciSyncPeer::verifyInternalAttr(
            newReconciledList.get(),
            LwSciSyncInternalAttrKey_SignalerPrimitiveInfo,
            LwSciSyncPeer::attrs.defaultPlatformPrimitive);
        LwSciSyncPeer::verifyInternalAttr(
            newReconciledList.get(),
            LwSciSyncInternalAttrKey_WaiterPrimitiveInfo,
            LwSciSyncPeer::attrs.defaultPlatformPrimitive);
    }

    // Assert on LwSciSyncAttrListReconcileAndObjAlloc
    {
        // Need not free
        LwSciSyncAttrList newReconciledList = nullptr;
        auto attrLists = {
            signalerAttrList.get(),
            waiterAttrList.get(),
        };

        auto syncObj = LwSciSyncPeer::reconcileAndAllocate(attrLists, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_NE(syncObj.get(), nullptr);

        ASSERT_EQ(LwSciSyncObjGetAttrList(syncObj.get(), &newReconciledList),
                  LwSciError_Success);

        // Assert on results
        LwSciSyncPeer::verifyAttr(newReconciledList,
                LwSciSyncAttrKey_NeedCpuAccess,
                false);
        LwSciSyncInternalAttrValPrimitiveType primitiveType =
                LwSciSyncPeer::attrs.defaultPlatformPrimitive;
        LwSciSyncPeer::verifyInternalAttr(newReconciledList,
                LwSciSyncInternalAttrKey_SignalerPrimitiveInfo,
                primitiveType);
        LwSciSyncPeer::verifyInternalAttr(newReconciledList,
                LwSciSyncInternalAttrKey_WaiterPrimitiveInfo,
                primitiveType);
    }
}

/**
 * @jama{15644664} PrimitiveInfo reconciliation: NeedCpuAccess True
 *
 * @verifies @jama{13561825}
 *
 * This test checks that we successfully reconcile the
 * {Signaler,Waiter}PrimitiveInfo key to a CPU primitive when NeedCpuAccess
 * is set to true.
 *
 */
LWSCISYNC_ATTRIBUTE_LIST_RECONCILE_TEST(
    AttributeListReconciliation,
    ReconciliationPrimitiveInfo_NeedCpuAccess_True,
    15644664)
{
    LwSciError error = LwSciError_Success;
    auto signalerAttrList = peer.createAttrList();
    auto waiterAttrList = peer.createAttrList();

    // Set up Signaler Attribute List
    {
        ASSERT_EQ(LwSciSyncAttrListSetAttrs(
                  signalerAttrList.get(),
                  LwSciSyncPeer::attrs.cpuSignaler.data(),
                  LwSciSyncPeer::attrs.cpuSignaler.size()),
              LwSciError_Success);

        ASSERT_EQ(LwSciSyncAttrListSetInternalAttrs(
                      signalerAttrList.get(),
                      LwSciSyncPeer::attrs.umdInternalAttrs.data(),
                      LwSciSyncPeer::attrs.umdInternalAttrs.size()),
                  LwSciError_Success);
    }

    // Set up Waiter Attribute List
    {
        ASSERT_EQ(LwSciSyncAttrListSetAttrs(
                  waiterAttrList.get(), LwSciSyncPeer::attrs.waiter.data(),
                  LwSciSyncPeer::attrs.waiter.size()),
              LwSciError_Success);

        ASSERT_EQ(LwSciSyncAttrListSetInternalAttrs(
                      waiterAttrList.get(),
                      LwSciSyncPeer::attrs.waiterPlatformDefaultAttrs.data(),
                      LwSciSyncPeer::attrs.waiterPlatformDefaultAttrs.size()),
                  LwSciError_Success);
    }

    // Assert on LwSciSyncAttrListReconcile
    {
        // Reconcile Attribute Lists
        auto attrLists = {signalerAttrList.get(), waiterAttrList.get()};
        auto newReconciledList =
            LwSciSyncPeer::reconcileLists(attrLists, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_NE(newReconciledList.get(), nullptr);

        // Assert on result values
        LwSciSyncPeer::verifyAttr(newReconciledList.get(),
                                  LwSciSyncAttrKey_NeedCpuAccess, true);
        LwSciSyncPeer::verifyInternalAttr(
            newReconciledList.get(),
            LwSciSyncInternalAttrKey_SignalerPrimitiveInfo,
            LwSciSyncPeer::attrs.defaultPlatformPrimitive);
        LwSciSyncPeer::verifyInternalAttr(
            newReconciledList.get(),
            LwSciSyncInternalAttrKey_WaiterPrimitiveInfo,
            LwSciSyncPeer::attrs.defaultPlatformPrimitive);
    }

    // Assert on LwSciSyncAttrListReconcileAndObjAlloc
    {
        // Need not free
        LwSciSyncAttrList newReconciledList = nullptr;
        auto attrLists = {
            signalerAttrList.get(),
            waiterAttrList.get(),
        };

        auto syncObj = LwSciSyncPeer::reconcileAndAllocate(attrLists, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_NE(syncObj.get(), nullptr);

        ASSERT_EQ(LwSciSyncObjGetAttrList(syncObj.get(), &newReconciledList),
                  LwSciError_Success);

        // Assert on results
        LwSciSyncPeer::verifyAttr(newReconciledList,
                LwSciSyncAttrKey_NeedCpuAccess,
                true);
        LwSciSyncInternalAttrValPrimitiveType primitiveType =
                LwSciSyncPeer::attrs.defaultPlatformPrimitive;
        LwSciSyncPeer::verifyInternalAttr(newReconciledList,
                LwSciSyncInternalAttrKey_SignalerPrimitiveInfo,
                primitiveType);
        LwSciSyncPeer::verifyInternalAttr(newReconciledList,
                LwSciSyncInternalAttrKey_WaiterPrimitiveInfo,
                primitiveType);
    }
}

/**
 * @jama{15644674} Signaler's requested primitive types: CPU Primitives
 *
 * @verifies @jama{13839339}
 *
 * This test checks that we successfully reconcile SignalerPrimitiveInfo to
 * a CPU Primitive if the Signaler's attribute list has NeedCpuAccess set to
 * True and SignalerPrimitiveInfo is unset.
 */
LWSCISYNC_ATTRIBUTE_LIST_RECONCILE_TEST(
    AttributeListReconciliation,
    ReconciliationSignalerPrimitiveInfo_CpuPrimitives,
    15644674)
{
    LwSciError error = LwSciError_Success;
    auto signalerAttrList = peer.createAttrList();
    auto waiterAttrList = peer.createAttrList();

    // Set up Signaler Attribute List
    {
        ASSERT_EQ(LwSciSyncAttrListSetAttrs(
                  signalerAttrList.get(),
                  LwSciSyncPeer::attrs.cpuSignaler.data(),
                  LwSciSyncPeer::attrs.cpuSignaler.size()),
              LwSciError_Success);
    }

    // Set up Waiter Attribute List
    {
        ASSERT_EQ(LwSciSyncAttrListSetAttrs(
                  waiterAttrList.get(), LwSciSyncPeer::attrs.waiter.data(),
                  LwSciSyncPeer::attrs.waiter.size()),
              LwSciError_Success);

        ASSERT_EQ(LwSciSyncAttrListSetInternalAttrs(
                      waiterAttrList.get(),
                      LwSciSyncPeer::attrs.waiterPlatformDefaultAttrs.data(),
                      LwSciSyncPeer::attrs.waiterPlatformDefaultAttrs.size()),
                  LwSciError_Success);
    }

    // Assert on LwSciSyncAttrListReconcile
    {
        auto attrLists = {signalerAttrList.get(), waiterAttrList.get()};
        auto newReconciledList =
            LwSciSyncPeer::reconcileLists(attrLists, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_NE(newReconciledList.get(), nullptr);

        // Assert on result values
        LwSciSyncPeer::verifyAttr(newReconciledList.get(),
                                  LwSciSyncAttrKey_NeedCpuAccess, true);

        LwSciSyncPeer::verifyInternalAttr(
            newReconciledList.get(),
            LwSciSyncInternalAttrKey_SignalerPrimitiveInfo,
            LwSciSyncPeer::attrs.defaultPlatformPrimitive);
    }

    // Assert on LwSciSyncAttrListReconcile
    {
        // Need not free
        LwSciSyncAttrList newReconciledList = nullptr;

        auto attrLists = {
            signalerAttrList.get(),
            waiterAttrList.get(),
        };

        auto syncObj = LwSciSyncPeer::reconcileAndAllocate(attrLists, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_NE(syncObj.get(), nullptr);

        ASSERT_EQ(LwSciSyncObjGetAttrList(syncObj.get(), &newReconciledList),
                  LwSciError_Success);

        // Assert on result values
        LwSciSyncPeer::verifyAttr(newReconciledList,
                LwSciSyncAttrKey_NeedCpuAccess, true);

        LwSciSyncPeer::verifyInternalAttr(newReconciledList,
                LwSciSyncInternalAttrKey_SignalerPrimitiveInfo,
                LwSciSyncPeer::attrs.defaultPlatformPrimitive);
    }
}

/**
 * @jama{15644686} Signaler's requested primitive types: SignalPrimitiveInfo value
 *
 * @verifies @jama{13839339}
 *
 * This test checks that we successfully reconcile SignalerPrimitiveInfo to
 * take the value of SignalerPrimitiveInfo if NeedCpuAccess is set to False.
 */
LWSCISYNC_ATTRIBUTE_LIST_RECONCILE_TEST(
    AttributeListReconciliation,
    ReconciliationSignalerPrimitiveInfo_SignalerPrimitiveInfo,
    15644686)
{
    LwSciError error = LwSciError_Success;
    auto signalerAttrList = peer.createAttrList();
    auto waiterAttrList = peer.createAttrList();

    // Set up Signaler Attribute List
    {
        ASSERT_EQ(LwSciSyncAttrListSetAttrs(
                      signalerAttrList.get(),
                      LwSciSyncPeer::attrs.signaler.data(),
                      LwSciSyncPeer::attrs.signaler.size()),
                  LwSciError_Success);

        ASSERT_EQ(LwSciSyncAttrListSetInternalAttrs(
                      signalerAttrList.get(),
                      LwSciSyncPeer::attrs.signalerPlatformDefaultAttrs.data(),
                      LwSciSyncPeer::attrs.signalerPlatformDefaultAttrs.size()),
                  LwSciError_Success);
    }

    // Set up Waiter Attribute List
    {
        ASSERT_EQ(LwSciSyncAttrListSetAttrs(
                  waiterAttrList.get(),
                  LwSciSyncPeer::attrs.waiter.data(),
                  LwSciSyncPeer::attrs.waiter.size()),
              LwSciError_Success);

        ASSERT_EQ(LwSciSyncAttrListSetInternalAttrs(
                      waiterAttrList.get(),
                      LwSciSyncPeer::attrs.waiterPlatformDefaultAttrs.data(),
                      LwSciSyncPeer::attrs.waiterPlatformDefaultAttrs.size()),
                  LwSciError_Success);
    }

    // Assert on LwSciSyncAttrListReconcile
    {
        auto attrLists = {signalerAttrList.get(), waiterAttrList.get()};
        auto newReconciledList =
            LwSciSyncPeer::reconcileLists(attrLists, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_NE(newReconciledList.get(), nullptr);

        // Assert on result values
        LwSciSyncPeer::verifyAttr(newReconciledList.get(),
                                  LwSciSyncAttrKey_NeedCpuAccess, false);

        LwSciSyncPeer::verifyInternalAttr(
            newReconciledList.get(),
            LwSciSyncInternalAttrKey_SignalerPrimitiveInfo,
            LwSciSyncPeer::attrs.defaultPlatformPrimitive);
    }

    // Assert on LwSciSyncAttrListReconcileAndObjAlloc
    {
        // Need not free
        LwSciSyncAttrList newReconciledList = nullptr;

        auto attrLists = {
            signalerAttrList.get(),
            waiterAttrList.get(),
        };

        auto syncObj = LwSciSyncPeer::reconcileAndAllocate(attrLists, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_NE(syncObj.get(), nullptr);

        ASSERT_EQ(LwSciSyncObjGetAttrList(syncObj.get(), &newReconciledList),
                  LwSciError_Success);

        // Assert on result values
        LwSciSyncPeer::verifyAttr(newReconciledList,
                LwSciSyncAttrKey_NeedCpuAccess, false);

        LwSciSyncPeer::verifyInternalAttr(newReconciledList,
                LwSciSyncInternalAttrKey_SignalerPrimitiveInfo,
                LwSciSyncPeer::attrs.defaultPlatformPrimitive);
    }
}

/**
 * @jama{15644692} Waiter's requested primitive types: NeedCpuAccess True
 *
 * @verifies @jama{13839342}
 *
 * This test checks that we successfully reconcile WaiterPrimitiveInfo to
 * a CPU Primitive if the Waiter's attribute list has NeedCpuAccess set to
 * True and WaiterPrimitiveInfo is unset.
 */
LWSCISYNC_ATTRIBUTE_LIST_RECONCILE_TEST(
    AttributeListReconciliation,
    WaiterPrimitiveInfo_NeedCpuAccess_True,
    15644692)
{
    LwSciError error = LwSciError_Success;
    auto signalerAttrList = peer.createAttrList();
    auto waiterAttrList = peer.createAttrList();

    // Set up Signaler Attribute List
    {
        ASSERT_EQ(LwSciSyncAttrListSetAttrs(
                      signalerAttrList.get(),
                      LwSciSyncPeer::attrs.signaler.data(),
                      LwSciSyncPeer::attrs.signaler.size()),
                  LwSciError_Success);

        ASSERT_EQ(LwSciSyncAttrListSetInternalAttrs(
                      signalerAttrList.get(),
                      LwSciSyncPeer::attrs.signalerPlatformDefaultAttrs.data(),
                      LwSciSyncPeer::attrs.signalerPlatformDefaultAttrs.size()),
                  LwSciError_Success);
    }

    // Set up Waiter Attribute List
    {
        ASSERT_EQ(LwSciSyncAttrListSetAttrs(
                      waiterAttrList.get(),
                      LwSciSyncPeer::attrs.cpuWaiter.data(),
                      LwSciSyncPeer::attrs.cpuWaiter.size()),
                  LwSciError_Success);
    }

    // Assert on LwSciSyncAttrListReconcile
    {
        auto newReconciledList = LwSciSyncPeer::reconcileLists(
            {signalerAttrList.get(), waiterAttrList.get()}, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_NE(newReconciledList.get(), nullptr);

        // Assert on result values
        LwSciSyncPeer::verifyAttr(newReconciledList.get(),
                                  LwSciSyncAttrKey_NeedCpuAccess, true);

        LwSciSyncPeer::verifyInternalAttr(
            newReconciledList.get(),
            LwSciSyncInternalAttrKey_WaiterPrimitiveInfo,
            LwSciSyncPeer::attrs.defaultPlatformPrimitive);
    }

    // Assert on LwSciSyncAttrListReconcileAndObjAlloc
    {
        // Need not free
        LwSciSyncAttrList newReconciledList = nullptr;

        auto attrLists = {
            signalerAttrList.get(),
            waiterAttrList.get(),
        };

        auto syncObj = LwSciSyncPeer::reconcileAndAllocate(attrLists, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_NE(syncObj.get(), nullptr);

        ASSERT_EQ(LwSciSyncObjGetAttrList(syncObj.get(), &newReconciledList),
                  LwSciError_Success);

        // Assert on result values
        LwSciSyncPeer::verifyAttr(newReconciledList,
                LwSciSyncAttrKey_NeedCpuAccess, true);

        LwSciSyncPeer::verifyInternalAttr(newReconciledList,
                LwSciSyncInternalAttrKey_WaiterPrimitiveInfo,
                LwSciSyncPeer::attrs.defaultPlatformPrimitive);
    }
}

/**
 * @jama{15644696} Waiter's requested primitive types: NeedCpuAccess False
 *
 * @verifies @jama{13839342}
 *
 * This test checks that we successfully reconcile WaiterPrimitiveInfo to
 * take the value of WaiterPrimitiveInfo if the Waiter's attribute list has
 * NeedCpuAccess set to False.
 */
LWSCISYNC_ATTRIBUTE_LIST_RECONCILE_TEST(
    AttributeListReconciliation,
    WaiterPrimitiveInfo_NeedCpuAccess_False,
    15644696)
{
    LwSciError error = LwSciError_Success;
    auto signalerAttrList = peer.createAttrList();
    auto waiterAttrList = peer.createAttrList();

    // Set up Signaler Attribute List
    {
        ASSERT_EQ(LwSciSyncAttrListSetAttrs(
                      signalerAttrList.get(),
                      LwSciSyncPeer::attrs.signaler.data(),
                      LwSciSyncPeer::attrs.signaler.size()),
                  LwSciError_Success);

        ASSERT_EQ(LwSciSyncAttrListSetInternalAttrs(
                      signalerAttrList.get(),
                      LwSciSyncPeer::attrs.signalerPlatformDefaultAttrs.data(),
                      LwSciSyncPeer::attrs.signalerPlatformDefaultAttrs.size()),
                  LwSciError_Success);
    }

    // Set up Waiter Attribute List
    {
        ASSERT_EQ(LwSciSyncAttrListSetAttrs(
                      waiterAttrList.get(),
                      LwSciSyncPeer::attrs.waiter.data(),
                      LwSciSyncPeer::attrs.waiter.size()),
                  LwSciError_Success);

        ASSERT_EQ(LwSciSyncAttrListSetInternalAttrs(
                      waiterAttrList.get(),
                      LwSciSyncPeer::attrs.waiterPlatformDefaultAttrs.data(),
                      LwSciSyncPeer::attrs.waiterPlatformDefaultAttrs.size()),
                  LwSciError_Success);
    }

    // Assert on LwSciSyncAttrListReconcile
    {
        auto newReconciledList = LwSciSyncPeer::reconcileLists(
            {signalerAttrList.get(), waiterAttrList.get()}, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_NE(newReconciledList.get(), nullptr);

        // Assert on result values
        LwSciSyncPeer::verifyAttr(newReconciledList.get(),
                                  LwSciSyncAttrKey_NeedCpuAccess, false);

        LwSciSyncPeer::verifyInternalAttr(
            newReconciledList.get(),
            LwSciSyncInternalAttrKey_WaiterPrimitiveInfo,
            LwSciSyncPeer::attrs.defaultPlatformPrimitive);
    }

    // Assert on LwSciSyncAttrListReconcileAndObjAlloc
    {
        // Need not free
        LwSciSyncAttrList newReconciledList = nullptr;

        auto attrLists = {
            signalerAttrList.get(),
            waiterAttrList.get(),
        };

        auto syncObj = LwSciSyncPeer::reconcileAndAllocate(attrLists, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_NE(syncObj.get(), nullptr);

        ASSERT_EQ(LwSciSyncObjGetAttrList(syncObj.get(), &newReconciledList),
                  LwSciError_Success);

        // Assert on result values
        LwSciSyncPeer::verifyAttr(newReconciledList,
                LwSciSyncAttrKey_NeedCpuAccess, false);

        LwSciSyncPeer::verifyInternalAttr(newReconciledList,
                LwSciSyncInternalAttrKey_WaiterPrimitiveInfo,
                LwSciSyncPeer::attrs.defaultPlatformPrimitive);
    }
}

/**
 * @jama{15644700} SignalerPrimitiveCount reconciliation: NeedCpuAccess True
 *
 * @verifies @jama{13839345}
 *
 * This test checks that we successfully reconcile SignalerPrimitiveCount to
 * 1 if NeedCpuAccess is set to True in the signaler's unreconciled
 * attribute list.
 */
LWSCISYNC_ATTRIBUTE_LIST_RECONCILE_TEST(
    AttributeListReconciliation,
    SignalerPrimitiveCount_NeedCpuAccess_True,
    15644700)
{
    LwSciError error = LwSciError_Success;
    auto signalerAttrList = peer.createAttrList();
    auto waiterAttrList = peer.createAttrList();

    // Set up Signaler Attribute List
    {
        ASSERT_EQ(LwSciSyncAttrListSetAttrs(
                      signalerAttrList.get(),
                      LwSciSyncPeer::attrs.cpuSignaler.data(),
                      LwSciSyncPeer::attrs.cpuSignaler.size()),
                  LwSciError_Success);

        ASSERT_EQ(LwSciSyncAttrListSetInternalAttrs(
                      signalerAttrList.get(),
                      LwSciSyncPeer::attrs.signalerPlatformDefaultAttrs.data(),
                      LwSciSyncPeer::attrs.signalerPlatformDefaultAttrs.size()),
                  LwSciError_Success);
    }

    // Set up Waiter Attribute List
    {
        ASSERT_EQ(LwSciSyncAttrListSetAttrs(
                      waiterAttrList.get(),
                      LwSciSyncPeer::attrs.waiter.data(),
                      LwSciSyncPeer::attrs.waiter.size()),
                  LwSciError_Success);

        ASSERT_EQ(LwSciSyncAttrListSetInternalAttrs(
                      waiterAttrList.get(),
                      LwSciSyncPeer::attrs.waiterPlatformDefaultAttrs.data(),
                      LwSciSyncPeer::attrs.waiterPlatformDefaultAttrs.size()),
                  LwSciError_Success);
    }

    // Assert on LwSciSyncAttrListReconcile
    {
        auto newReconciledList = LwSciSyncPeer::reconcileLists(
            {signalerAttrList.get(), waiterAttrList.get()}, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_NE(newReconciledList.get(), nullptr);

        // Assert on result values
        LwSciSyncPeer::verifyAttr(newReconciledList.get(),
                                  LwSciSyncAttrKey_NeedCpuAccess, true);

        LwSciSyncPeer::verifyInternalAttr(
            newReconciledList.get(),
            LwSciSyncInternalAttrKey_SignalerPrimitiveCount, 1);
    }

    // Assert on LwSciSyncAttrListReconcileAndObjAlloc
    {
        // Need not free
        LwSciSyncAttrList newReconciledList = nullptr;

        auto attrLists = {
            signalerAttrList.get(),
            waiterAttrList.get(),
        };

        auto syncObj = LwSciSyncPeer::reconcileAndAllocate(attrLists, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_NE(syncObj.get(), nullptr);

        ASSERT_EQ(LwSciSyncObjGetAttrList(syncObj.get(), &newReconciledList),
                  LwSciError_Success);

        // Assert on result values
        LwSciSyncPeer::verifyAttr(newReconciledList,
                LwSciSyncAttrKey_NeedCpuAccess, true);

        LwSciSyncPeer::verifyInternalAttr(newReconciledList,
                LwSciSyncInternalAttrKey_SignalerPrimitiveCount,
                1);
    }
}

/**
 * @jama{15644702} SignalerPrimitiveCount reconciliation: NeedCpuAccess False
 *
 * @verifies @jama{13839345}
 *
 * This test checks that we successfully reconcile SignalerPrimitiveCount to
 * SignalerPrimitiveCount attribute in the signaler's attribute list if
 * NeedCpuAccess is set to False.
 */
LWSCISYNC_ATTRIBUTE_LIST_RECONCILE_TEST(
    AttributeListReconciliation,
    SignalerPrimitiveCount_NeedCpuAccess_False,
    15644702)
{
    LwSciError error = LwSciError_Success;
    auto signalerAttrList = peer.createAttrList();
    auto waiterAttrList = peer.createAttrList();

    const uint32_t signalerPrimitiveCount = 24601;

    // Set up Signaler Attribute List
    {
        ASSERT_EQ(LwSciSyncAttrListSetAttrs(
                      signalerAttrList.get(),
                      LwSciSyncPeer::attrs.signaler.data(),
                      LwSciSyncPeer::attrs.signaler.size()),
                  LwSciError_Success);

        ASSERT_EQ(
                LwSciSyncPeer::setInternalAttr(signalerAttrList.get(),
                    LwSciSyncInternalAttrKey_SignalerPrimitiveInfo,
                    LwSciSyncPeer::attrs.defaultPlatformPrimitive),
                LwSciError_Success);

        ASSERT_EQ(
                LwSciSyncPeer::setInternalAttr(signalerAttrList.get(),
                    LwSciSyncInternalAttrKey_SignalerPrimitiveCount,
                    signalerPrimitiveCount),
                LwSciError_Success);
    }

    // Set up Waiter Attribute List
    {
        ASSERT_EQ(LwSciSyncAttrListSetAttrs(
                      waiterAttrList.get(),
                      LwSciSyncPeer::attrs.waiter.data(),
                      LwSciSyncPeer::attrs.waiter.size()),
                  LwSciError_Success);

        ASSERT_EQ(
                LwSciSyncPeer::setInternalAttr(waiterAttrList.get(),
                    LwSciSyncInternalAttrKey_WaiterPrimitiveInfo,
                    LwSciSyncPeer::attrs.defaultPlatformPrimitive),
                LwSciError_Success);
    }

    // Assert on LwSciSyncAttrListReconcile
    {
        auto newReconciledList = LwSciSyncPeer::reconcileLists(
            {signalerAttrList.get(), waiterAttrList.get()}, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_NE(newReconciledList.get(), nullptr);

        // Assert on result values
        LwSciSyncPeer::verifyAttr(newReconciledList.get(),
                                  LwSciSyncAttrKey_NeedCpuAccess, false);

        LwSciSyncPeer::verifyInternalAttr(
            newReconciledList.get(),
            LwSciSyncInternalAttrKey_SignalerPrimitiveCount,
            signalerPrimitiveCount);
    }

    // Assert on LwSciSyncAttrListReconcileAndObjAlloc
    {
        // Need not free
        LwSciSyncAttrList newReconciledList = nullptr;

        auto attrLists = {
            signalerAttrList.get(),
            waiterAttrList.get(),
        };

        auto syncObj = LwSciSyncPeer::reconcileAndAllocate(attrLists, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_NE(syncObj.get(), nullptr);

        ASSERT_EQ(LwSciSyncObjGetAttrList(syncObj.get(), &newReconciledList),
                  LwSciError_Success);

        // Assert on result values
        LwSciSyncPeer::verifyAttr(newReconciledList,
                LwSciSyncAttrKey_NeedCpuAccess, false);

        LwSciSyncPeer::verifyInternalAttr(newReconciledList,
                LwSciSyncInternalAttrKey_SignalerPrimitiveCount,
                signalerPrimitiveCount);
    }
}

/**
 * @jama{} RequireDeterministicFences reconciliation: Signaler true
 *
 * @verifies @jama{}
 *
 * This test checks that we successfully reconcile RequireDeterministicFences
 * to true when the signaler sets RequireDeterministicFences.
 */
LWSCISYNC_ATTRIBUTE_LIST_RECONCILE_TEST(
    AttributeListReconciliation,
    RequireDeterministicFences_SignalerTrue,
    1)
{
    LwSciError error = LwSciError_Success;
    auto signalerAttrList = peer.createAttrList();
    auto waiterAttrList = peer.createAttrList();

    // Set up Signaler Attribute List
    {
        ASSERT_EQ(LwSciSyncAttrListSetAttrs(
                  signalerAttrList.get(),
                  LwSciSyncPeer::attrs.cpuSignaler.data(),
                  LwSciSyncPeer::attrs.cpuSignaler.size()),
              LwSciError_Success);

        ASSERT_EQ(LwSciSyncAttrListSetInternalAttrs(
                      signalerAttrList.get(),
                      LwSciSyncPeer::attrs.signalerSemaphoreAttrs.data(),
                      LwSciSyncPeer::attrs.signalerSemaphoreAttrs.size()),
                  LwSciError_Success);

        // Set RequireDeterministicFences
        ASSERT_EQ(
                LwSciSyncPeer::setAttr(signalerAttrList.get(),
                    LwSciSyncAttrKey_RequireDeterministicFences,
                    true),
                LwSciError_Success);
    }

    // Set up Waiter Attribute List
    {
        ASSERT_EQ(LwSciSyncAttrListSetAttrs(
                  waiterAttrList.get(),
                  LwSciSyncPeer::attrs.cpuWaiter.data(),
                  LwSciSyncPeer::attrs.cpuWaiter.size()),
              LwSciError_Success);

        ASSERT_EQ(LwSciSyncAttrListSetInternalAttrs(
                      waiterAttrList.get(),
                      LwSciSyncPeer::attrs.waiterSemaphoreAttrs.data(),
                      LwSciSyncPeer::attrs.waiterSemaphoreAttrs.size()),
                  LwSciError_Success);
    }

    // Assert on LwSciSyncAttrListReconcile
    {
        auto newReconciledList = LwSciSyncPeer::reconcileLists(
            {signalerAttrList.get(), waiterAttrList.get()}, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_NE(newReconciledList.get(), nullptr);

        // Assert on result values
        LwSciSyncPeer::verifyAttr(newReconciledList.get(),
                                  LwSciSyncAttrKey_RequireDeterministicFences, true);
        LwSciSyncPeer::verifyInternalAttr(
            newReconciledList.get(),
            LwSciSyncInternalAttrKey_SignalerPrimitiveInfo,
            LwSciSyncInternalAttrValPrimitiveType_SysmemSemaphore);
        LwSciSyncPeer::verifyInternalAttr(
            newReconciledList.get(),
            LwSciSyncInternalAttrKey_WaiterPrimitiveInfo,
            LwSciSyncInternalAttrValPrimitiveType_SysmemSemaphore);
    }

    // Assert on LwSciSyncAttrListReconcileAndObjAlloc
    {
        // Need not free
        LwSciSyncAttrList newReconciledList = nullptr;

        auto attrLists = {
            signalerAttrList.get(),
            waiterAttrList.get(),
        };

        auto syncObj = LwSciSyncPeer::reconcileAndAllocate(attrLists, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_NE(syncObj.get(), nullptr);

        ASSERT_EQ(LwSciSyncObjGetAttrList(syncObj.get(), &newReconciledList),
                  LwSciError_Success);

        // Assert on result values
        LwSciSyncPeer::verifyAttr(newReconciledList,
                LwSciSyncAttrKey_RequireDeterministicFences, true);
        LwSciSyncPeer::verifyInternalAttr(
            newReconciledList,
            LwSciSyncInternalAttrKey_SignalerPrimitiveInfo,
            LwSciSyncInternalAttrValPrimitiveType_SysmemSemaphore);
        LwSciSyncPeer::verifyInternalAttr(
            newReconciledList,
            LwSciSyncInternalAttrKey_WaiterPrimitiveInfo,
            LwSciSyncInternalAttrValPrimitiveType_SysmemSemaphore);
    }
}

/**
 * @jama{} RequireDeterministicFences reconciliation: Waiter true
 *
 * @verifies @jama{}
 *
 * This test checks that we successfully reconcile RequireDeterministicFences
 * to true when the waiter sets RequireDeterministicFences.
 */
LWSCISYNC_ATTRIBUTE_LIST_RECONCILE_TEST(
    AttributeListReconciliation,
    RequireDeterministicFences_WaiterTrue,
    2)
{
    LwSciError error = LwSciError_Success;
    auto signalerAttrList = peer.createAttrList();
    auto waiterAttrList = peer.createAttrList();

    // Set up Signaler Attribute List
    {
        ASSERT_EQ(LwSciSyncAttrListSetAttrs(
                  signalerAttrList.get(),
                  LwSciSyncPeer::attrs.cpuSignaler.data(),
                  LwSciSyncPeer::attrs.cpuSignaler.size()),
              LwSciError_Success);

        ASSERT_EQ(LwSciSyncAttrListSetInternalAttrs(
                      signalerAttrList.get(),
                      LwSciSyncPeer::attrs.signalerSemaphoreAttrs.data(),
                      LwSciSyncPeer::attrs.signalerSemaphoreAttrs.size()),
                  LwSciError_Success);
    }

    // Set up Waiter Attribute List
    {
        ASSERT_EQ(LwSciSyncAttrListSetAttrs(
                  waiterAttrList.get(),
                  LwSciSyncPeer::attrs.cpuWaiter.data(),
                  LwSciSyncPeer::attrs.cpuWaiter.size()),
              LwSciError_Success);

        ASSERT_EQ(LwSciSyncAttrListSetInternalAttrs(
                      waiterAttrList.get(),
                      LwSciSyncPeer::attrs.waiterSemaphoreAttrs.data(),
                      LwSciSyncPeer::attrs.waiterSemaphoreAttrs.size()),
                  LwSciError_Success);

        // Set RequireDeterministicFences
        ASSERT_EQ(
                LwSciSyncPeer::setAttr(waiterAttrList.get(),
                    LwSciSyncAttrKey_RequireDeterministicFences,
                    true),
                LwSciError_Success);
    }

    // Assert on LwSciSyncAttrListReconcile
    {
        auto newReconciledList = LwSciSyncPeer::reconcileLists(
            {signalerAttrList.get(), waiterAttrList.get()}, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_NE(newReconciledList.get(), nullptr);

        // Assert on result values
        LwSciSyncPeer::verifyAttr(newReconciledList.get(),
                                  LwSciSyncAttrKey_RequireDeterministicFences, true);
        LwSciSyncPeer::verifyInternalAttr(
            newReconciledList.get(),
            LwSciSyncInternalAttrKey_SignalerPrimitiveInfo,
            LwSciSyncInternalAttrValPrimitiveType_SysmemSemaphore);
        LwSciSyncPeer::verifyInternalAttr(
            newReconciledList.get(),
            LwSciSyncInternalAttrKey_WaiterPrimitiveInfo,
            LwSciSyncInternalAttrValPrimitiveType_SysmemSemaphore);
    }

    // Assert on LwSciSyncAttrListReconcileAndObjAlloc
    {
        // Need not free
        LwSciSyncAttrList newReconciledList = nullptr;

        auto attrLists = {
            signalerAttrList.get(),
            waiterAttrList.get(),
        };

        auto syncObj = LwSciSyncPeer::reconcileAndAllocate(attrLists, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_NE(syncObj.get(), nullptr);

        ASSERT_EQ(LwSciSyncObjGetAttrList(syncObj.get(), &newReconciledList),
                  LwSciError_Success);

        // Assert on result values
        LwSciSyncPeer::verifyAttr(newReconciledList,
                LwSciSyncAttrKey_RequireDeterministicFences, true);
        LwSciSyncPeer::verifyInternalAttr(
            newReconciledList,
            LwSciSyncInternalAttrKey_SignalerPrimitiveInfo,
            LwSciSyncInternalAttrValPrimitiveType_SysmemSemaphore);
        LwSciSyncPeer::verifyInternalAttr(
            newReconciledList,
            LwSciSyncInternalAttrKey_WaiterPrimitiveInfo,
            LwSciSyncInternalAttrValPrimitiveType_SysmemSemaphore);
    }
}

/**
 * @jama{} RequireDeterministicFences reconciliation: False
 *
 * @verifies @jama{}
 *
 * This test checks that we successfully reconcile RequireDeterministicFences
 * to false when both signaler and waiter don't set RequireDeterministicFences.
 */
LWSCISYNC_ATTRIBUTE_LIST_RECONCILE_TEST(
    AttributeListReconciliation,
    RequireDeterministicFences_False,
    4)
{
    LwSciError error = LwSciError_Success;
    auto signalerAttrList = peer.createAttrList();
    auto waiterAttrList = peer.createAttrList();

    // Set up Signaler Attribute List
    {
        ASSERT_EQ(LwSciSyncAttrListSetAttrs(
                  signalerAttrList.get(),
                  LwSciSyncPeer::attrs.cpuSignaler.data(),
                  LwSciSyncPeer::attrs.cpuSignaler.size()),
              LwSciError_Success);

        ASSERT_EQ(LwSciSyncAttrListSetInternalAttrs(
                      signalerAttrList.get(),
                      LwSciSyncPeer::attrs.signalerSemaphoreAttrs.data(),
                      LwSciSyncPeer::attrs.signalerSemaphoreAttrs.size()),
                  LwSciError_Success);
    }

    // Set up Waiter Attribute List
    {
        ASSERT_EQ(LwSciSyncAttrListSetAttrs(
                  waiterAttrList.get(),
                  LwSciSyncPeer::attrs.cpuWaiter.data(),
                  LwSciSyncPeer::attrs.cpuWaiter.size()),
              LwSciError_Success);

        ASSERT_EQ(LwSciSyncAttrListSetInternalAttrs(
                      waiterAttrList.get(),
                      LwSciSyncPeer::attrs.waiterSemaphoreAttrs.data(),
                      LwSciSyncPeer::attrs.waiterSemaphoreAttrs.size()),
                  LwSciError_Success);
    }

    // Assert on LwSciSyncAttrListReconcile
    {
        auto newReconciledList = LwSciSyncPeer::reconcileLists(
            {signalerAttrList.get(), waiterAttrList.get()}, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_NE(newReconciledList.get(), nullptr);

        // Assert on result values
        LwSciSyncPeer::verifyAttr(newReconciledList.get(),
                                  LwSciSyncAttrKey_RequireDeterministicFences, false);
        LwSciSyncPeer::verifyInternalAttr(
            newReconciledList.get(),
            LwSciSyncInternalAttrKey_SignalerPrimitiveInfo,
            LwSciSyncInternalAttrValPrimitiveType_SysmemSemaphore);
        LwSciSyncPeer::verifyInternalAttr(
            newReconciledList.get(),
            LwSciSyncInternalAttrKey_WaiterPrimitiveInfo,
            LwSciSyncInternalAttrValPrimitiveType_SysmemSemaphore);
    }

    // Assert on LwSciSyncAttrListReconcileAndObjAlloc
    {
        // Need not free
        LwSciSyncAttrList newReconciledList = nullptr;

        auto attrLists = {
            signalerAttrList.get(),
            waiterAttrList.get(),
        };

        auto syncObj = LwSciSyncPeer::reconcileAndAllocate(attrLists, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_NE(syncObj.get(), nullptr);

        ASSERT_EQ(LwSciSyncObjGetAttrList(syncObj.get(), &newReconciledList),
                  LwSciError_Success);

        // Assert on result values
        LwSciSyncPeer::verifyAttr(newReconciledList,
                LwSciSyncAttrKey_RequireDeterministicFences, false);
        LwSciSyncPeer::verifyInternalAttr(
            newReconciledList,
            LwSciSyncInternalAttrKey_SignalerPrimitiveInfo,
            LwSciSyncInternalAttrValPrimitiveType_SysmemSemaphore);
        LwSciSyncPeer::verifyInternalAttr(
            newReconciledList,
            LwSciSyncInternalAttrKey_WaiterPrimitiveInfo,
            LwSciSyncInternalAttrValPrimitiveType_SysmemSemaphore);
    }
}

#if !defined(__x86_64__)
/**
 * @jama{} RequireDeterministicFences reconciliation: Invalid Primitives
 *
 * @verifies @jama{}
 *
 * This test checks that we fail to reconcile RequireDeterministicFences when
 * the only supported primitives are not deterministic.
 */
LWSCISYNC_ATTRIBUTE_LIST_RECONCILE_TEST(
    AttributeListReconciliation,
    RequireDeterministicFences_IlwalidPrimitives,
    5)
{
    // Note: This test can only run on CheetAh, since on x86 syncpoints are not
    // supported. Thus, CheetAh is the only platform that supports more than 1
    // primitive.
    LwSciError error = LwSciError_Success;
    auto signalerAttrList = peer.createAttrList();
    auto waiterAttrList = peer.createAttrList();

    // Set up Signaler Attribute List
    {
        ASSERT_EQ(LwSciSyncAttrListSetAttrs(
                  signalerAttrList.get(),
                  LwSciSyncPeer::attrs.cpuSignaler.data(),
                  LwSciSyncPeer::attrs.cpuSignaler.size()),
              LwSciError_Success);

        ASSERT_EQ(LwSciSyncAttrListSetInternalAttrs(
                      signalerAttrList.get(),
                      LwSciSyncPeer::attrs.signalerSyncpointAttrs.data(),
                      LwSciSyncPeer::attrs.signalerSyncpointAttrs.size()),
                  LwSciError_Success);
    }

    // Set up Waiter Attribute List
    {
        ASSERT_EQ(LwSciSyncAttrListSetAttrs(
                  waiterAttrList.get(),
                  LwSciSyncPeer::attrs.cpuWaiter.data(),
                  LwSciSyncPeer::attrs.cpuWaiter.size()),
              LwSciError_Success);

        ASSERT_EQ(LwSciSyncAttrListSetInternalAttrs(
                      waiterAttrList.get(),
                      LwSciSyncPeer::attrs.waiterSyncpointAttrs.data(),
                      LwSciSyncPeer::attrs.waiterSyncpointAttrs.size()),
                  LwSciError_Success);

        // Set RequireDeterministicFences
        ASSERT_EQ(
                LwSciSyncPeer::setAttr(waiterAttrList.get(),
                    LwSciSyncAttrKey_RequireDeterministicFences,
                    true),
                LwSciError_Success);
    }

    // Assert on LwSciSyncAttrListReconcile
    {
        NegativeTestPrint();
        auto newReconciledList = LwSciSyncPeer::reconcileLists(
            {signalerAttrList.get(), waiterAttrList.get()}, &error);
        ASSERT_NE(error, LwSciError_Success);
    }

    // Assert on LwSciSyncAttrListReconcileAndObjAlloc
    {
        // Need not free
        LwSciSyncAttrList newReconciledList = nullptr;

        auto attrLists = {
            signalerAttrList.get(),
            waiterAttrList.get(),
        };

        NegativeTestPrint();
        auto syncObj = LwSciSyncPeer::reconcileAndAllocate(attrLists, &error);
        ASSERT_NE(error, LwSciError_Success);
    }
}

/**
 * @jama{} RequireDeterministicFences reconciliation: Multiple Primitives
 *
 * @verifies @jama{}
 *
 * This test checks that we successfully reconcile and select a deterministic
 * primitive when RequireDeterministicFences is set.
 */
LWSCISYNC_ATTRIBUTE_LIST_RECONCILE_TEST(
    AttributeListReconciliation,
    RequireDeterministicFences_MultiplePrimitives,
    6)
{
    // Note: This test can only run on CheetAh, since on x86 syncpoints are not
    // supported. Thus, CheetAh is the only platform that supports more than 1
    // primitive.
    LwSciError error = LwSciError_Success;
    auto signalerAttrList = peer.createAttrList();
    auto waiterAttrList = peer.createAttrList();

    // Set up Signaler Attribute List
    {
        ASSERT_EQ(LwSciSyncAttrListSetAttrs(
                  signalerAttrList.get(),
                  LwSciSyncPeer::attrs.cpuSignaler.data(),
                  LwSciSyncPeer::attrs.cpuSignaler.size()),
              LwSciError_Success);

        ASSERT_EQ(LwSciSyncAttrListSetInternalAttrs(
                      signalerAttrList.get(),
                      LwSciSyncPeer::attrs.umdInternalAttrs.data(),
                      LwSciSyncPeer::attrs.umdInternalAttrs.size()),
                  LwSciError_Success);
    }

    // Set up Waiter Attribute List
    {
        ASSERT_EQ(LwSciSyncAttrListSetAttrs(
                  waiterAttrList.get(),
                  LwSciSyncPeer::attrs.cpuWaiter.data(),
                  LwSciSyncPeer::attrs.cpuWaiter.size()),
              LwSciError_Success);

        ASSERT_EQ(LwSciSyncAttrListSetInternalAttrs(
                      waiterAttrList.get(),
                      LwSciSyncPeer::attrs.waiterMultiPrimitiveAttrs.data(),
                      LwSciSyncPeer::attrs.waiterMultiPrimitiveAttrs.size()),
                  LwSciError_Success);

        // Set RequireDeterministicFences
        ASSERT_EQ(
                LwSciSyncPeer::setAttr(waiterAttrList.get(),
                    LwSciSyncAttrKey_RequireDeterministicFences,
                    true),
                LwSciError_Success);
    }

    // Assert on LwSciSyncAttrListReconcile
    {
        auto newReconciledList = LwSciSyncPeer::reconcileLists(
            {signalerAttrList.get(), waiterAttrList.get()}, &error);
        ASSERT_EQ(error, LwSciError_Success);

        LwSciSyncPeer::verifyAttr(newReconciledList.get(),
                LwSciSyncAttrKey_RequireDeterministicFences, true);
        LwSciSyncPeer::verifyInternalAttr(
            newReconciledList.get(),
            LwSciSyncInternalAttrKey_SignalerPrimitiveInfo,
            LwSciSyncInternalAttrValPrimitiveType_SysmemSemaphore);
        LwSciSyncPeer::verifyInternalAttr(
            newReconciledList.get(),
            LwSciSyncInternalAttrKey_WaiterPrimitiveInfo,
            LwSciSyncInternalAttrValPrimitiveType_SysmemSemaphore);
    }

    // Assert on LwSciSyncAttrListReconcileAndObjAlloc
    {
        // Need not free
        LwSciSyncAttrList newReconciledList = nullptr;

        auto attrLists = {
            signalerAttrList.get(),
            waiterAttrList.get(),
        };

        auto syncObj = LwSciSyncPeer::reconcileAndAllocate(attrLists, &error);
        ASSERT_EQ(error, LwSciError_Success);

        ASSERT_EQ(LwSciSyncObjGetAttrList(syncObj.get(), &newReconciledList),
                                  LwSciError_Success);

        LwSciSyncPeer::verifyAttr(newReconciledList,
                LwSciSyncAttrKey_RequireDeterministicFences, true);
        LwSciSyncPeer::verifyInternalAttr(
            newReconciledList,
            LwSciSyncInternalAttrKey_SignalerPrimitiveInfo,
            LwSciSyncInternalAttrValPrimitiveType_SysmemSemaphore);
        LwSciSyncPeer::verifyInternalAttr(
            newReconciledList,
            LwSciSyncInternalAttrKey_WaiterPrimitiveInfo,
            LwSciSyncInternalAttrValPrimitiveType_SysmemSemaphore);
    }
}
#endif

/**
 * @jama{} RequireDeterministicFences reconciliation: Multiple Deterministic
 * Primitives
 *
 * @verifies @jama{}
 *
 * This test checks that we successfully reconcile RequireDeterministicFences
 * to true when the waiter sets RequireDeterministicFences and that a
 * deterministic primitive is selected when multiple potential deterministic
 * primitives were requested.
 */
LWSCISYNC_ATTRIBUTE_LIST_RECONCILE_TEST(
    AttributeListReconciliation,
    RequireDeterministicFences_MultipleDeterministicPrimitives,
    7)
{
    LwSciError error = LwSciError_Success;
    auto signalerAttrList = peer.createAttrList();
    auto waiterAttrList = peer.createAttrList();

    // Lwrrently we prefer Sysmem Semaphores over the 64-bit semaphore payload
    // primitive. This is an implementation detail, so we will just assert
    // within this test that the reconciled primitive is deterministic.
    std::array<LwSciSyncInternalAttrValPrimitiveType, 2> potentialDeterministicPrimitives = {
        LwSciSyncInternalAttrValPrimitiveType_SysmemSemaphore,
        LwSciSyncInternalAttrValPrimitiveType_SysmemSemaphorePayload64b,
    };

    // Set up Signaler Attribute List
    {
        ASSERT_EQ(LwSciSyncAttrListSetAttrs(
                  signalerAttrList.get(),
                  LwSciSyncPeer::attrs.cpuSignaler.data(),
                  LwSciSyncPeer::attrs.cpuSignaler.size()),
              LwSciError_Success);

        ASSERT_EQ(LwSciSyncAttrListSetInternalAttrs(
                      signalerAttrList.get(),
                      LwSciSyncPeer::attrs.signalerMultiDeterministicPrimitiveAttrs.data(),
                      LwSciSyncPeer::attrs.signalerMultiDeterministicPrimitiveAttrs.size()),
                  LwSciError_Success);
    }

    // Set up Waiter Attribute List
    {
        ASSERT_EQ(LwSciSyncAttrListSetAttrs(
                  waiterAttrList.get(),
                  LwSciSyncPeer::attrs.cpuWaiter.data(),
                  LwSciSyncPeer::attrs.cpuWaiter.size()),
              LwSciError_Success);

        ASSERT_EQ(LwSciSyncAttrListSetInternalAttrs(
                      waiterAttrList.get(),
                      LwSciSyncPeer::attrs.waiterMultiDeterministicPrimitiveAttrs.data(),
                      LwSciSyncPeer::attrs.waiterMultiDeterministicPrimitiveAttrs.size()),
                  LwSciError_Success);

        // Set RequireDeterministicFences
        ASSERT_EQ(
                LwSciSyncPeer::setAttr(waiterAttrList.get(),
                    LwSciSyncAttrKey_RequireDeterministicFences,
                    true),
                LwSciError_Success);
    }

    // Assert on LwSciSyncAttrListReconcile
    {
        auto newReconciledList = LwSciSyncPeer::reconcileLists(
            {signalerAttrList.get(), waiterAttrList.get()}, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_NE(newReconciledList.get(), nullptr);

        // Assert on result values
        LwSciSyncPeer::verifyAttr(newReconciledList.get(),
                                  LwSciSyncAttrKey_RequireDeterministicFences, true);

        LwSciSyncInternalAttrKeyValuePair attr = {};
        // Assert on signaler values
        attr.attrKey = LwSciSyncInternalAttrKey_SignalerPrimitiveInfo;
        attr.value = nullptr;
        attr.len = 0;
        ASSERT_EQ(
            LwSciSyncAttrListGetInternalAttrs(newReconciledList.get(), &attr, 1),
            LwSciError_Success);

        ASSERT_NE(attr.value, nullptr);
        ASSERT_EQ(attr.len, sizeof(LwSciSyncInternalAttrValPrimitiveType));
        LwSciSyncInternalAttrValPrimitiveType* signalerPrimitive =
            (LwSciSyncInternalAttrValPrimitiveType*)attr.value;

        ASSERT_THAT(potentialDeterministicPrimitives, testing::Contains(*signalerPrimitive));

        // Assert on waiter values
        attr.attrKey = LwSciSyncInternalAttrKey_WaiterPrimitiveInfo;
        attr.value = nullptr;
        attr.len = 0;
        ASSERT_EQ(
            LwSciSyncAttrListGetInternalAttrs(newReconciledList.get(), &attr, 1),
            LwSciError_Success);

        ASSERT_NE(attr.value, nullptr);
        ASSERT_EQ(attr.len, sizeof(LwSciSyncInternalAttrValPrimitiveType));
        LwSciSyncInternalAttrValPrimitiveType* waiterPrimitive =
            (LwSciSyncInternalAttrValPrimitiveType*)attr.value;

        ASSERT_EQ(*signalerPrimitive, *waiterPrimitive);
    }

    // Assert on LwSciSyncAttrListReconcileAndObjAlloc
    {
        // Need not free
        LwSciSyncAttrList newReconciledList = nullptr;

        auto attrLists = {
            signalerAttrList.get(),
            waiterAttrList.get(),
        };

        auto syncObj = LwSciSyncPeer::reconcileAndAllocate(attrLists, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_NE(syncObj.get(), nullptr);

        ASSERT_EQ(LwSciSyncObjGetAttrList(syncObj.get(), &newReconciledList),
                  LwSciError_Success);

        // Assert on result values
        LwSciSyncPeer::verifyAttr(newReconciledList,
                LwSciSyncAttrKey_RequireDeterministicFences, true);

        LwSciSyncInternalAttrKeyValuePair attr = {};
        // Assert on signaler values
        attr.attrKey = LwSciSyncInternalAttrKey_SignalerPrimitiveInfo;
        attr.value = nullptr;
        attr.len = 0;
        ASSERT_EQ(
            LwSciSyncAttrListGetInternalAttrs(newReconciledList, &attr, 1),
            LwSciError_Success);

        ASSERT_NE(attr.value, nullptr);
        ASSERT_EQ(attr.len, sizeof(LwSciSyncInternalAttrValPrimitiveType));
        LwSciSyncInternalAttrValPrimitiveType* signalerPrimitive =
            (LwSciSyncInternalAttrValPrimitiveType*)attr.value;

        ASSERT_THAT(potentialDeterministicPrimitives, testing::Contains(*signalerPrimitive));

        // Assert on waiter values
        attr.attrKey = LwSciSyncInternalAttrKey_WaiterPrimitiveInfo;
        attr.value = nullptr;
        attr.len = 0;
        ASSERT_EQ(
            LwSciSyncAttrListGetInternalAttrs(newReconciledList, &attr, 1),
            LwSciError_Success);

        ASSERT_NE(attr.value, nullptr);
        ASSERT_EQ(attr.len, sizeof(LwSciSyncInternalAttrValPrimitiveType));
        LwSciSyncInternalAttrValPrimitiveType* waiterPrimitive =
            (LwSciSyncInternalAttrValPrimitiveType*)attr.value;

        ASSERT_EQ(*signalerPrimitive, *waiterPrimitive);
    }
}

/**
 * @jama{} RequireDeterministicFences reconciliation: 64-bit semaphore payload
 *
 * @verifies @jama{}
 *
 * This test checks that 64-bit semaphore payload primitives are considered
 * deterministic.
 */
LWSCISYNC_ATTRIBUTE_LIST_RECONCILE_TEST(
    AttributeListReconciliation,
    RequireDeterministicFences_64bPayloadSemaphoreIsDeterministic,
    8)
{
    // Lwrrently we prefer Sysmem Semaphores over the 64-bit semaphore payload
    // primitive.
    LwSciError error = LwSciError_Success;
    auto signalerAttrList = peer.createAttrList();
    auto waiterAttrList = peer.createAttrList();

    // Set up Signaler Attribute List
    {
        ASSERT_EQ(LwSciSyncAttrListSetAttrs(
                  signalerAttrList.get(),
                  LwSciSyncPeer::attrs.cpuSignaler.data(),
                  LwSciSyncPeer::attrs.cpuSignaler.size()),
              LwSciError_Success);

        ASSERT_EQ(LwSciSyncAttrListSetInternalAttrs(
                      signalerAttrList.get(),
                      LwSciSyncPeer::attrs.signalerSemaphore64bPayloadAttrs.data(),
                      LwSciSyncPeer::attrs.signalerSemaphore64bPayloadAttrs.size()),
                  LwSciError_Success);
    }

    // Set up Waiter Attribute List
    {
        ASSERT_EQ(LwSciSyncAttrListSetAttrs(
                  waiterAttrList.get(),
                  LwSciSyncPeer::attrs.cpuWaiter.data(),
                  LwSciSyncPeer::attrs.cpuWaiter.size()),
              LwSciError_Success);

        ASSERT_EQ(LwSciSyncAttrListSetInternalAttrs(
                      waiterAttrList.get(),
                      LwSciSyncPeer::attrs.waiterSemaphore64bPayloadAttrs.data(),
                      LwSciSyncPeer::attrs.waiterSemaphore64bPayloadAttrs.size()),
                  LwSciError_Success);

        // Set RequireDeterministicFences
        ASSERT_EQ(
                LwSciSyncPeer::setAttr(waiterAttrList.get(),
                    LwSciSyncAttrKey_RequireDeterministicFences,
                    true),
                LwSciError_Success);
    }

    // Assert on LwSciSyncAttrListReconcile
    {
        auto newReconciledList = LwSciSyncPeer::reconcileLists(
            {signalerAttrList.get(), waiterAttrList.get()}, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_NE(newReconciledList.get(), nullptr);

        // Assert on result values
        LwSciSyncPeer::verifyAttr(newReconciledList.get(),
                                  LwSciSyncAttrKey_RequireDeterministicFences, true);
        LwSciSyncPeer::verifyInternalAttr(
            newReconciledList.get(),
            LwSciSyncInternalAttrKey_SignalerPrimitiveInfo,
            LwSciSyncInternalAttrValPrimitiveType_SysmemSemaphorePayload64b);
        LwSciSyncPeer::verifyInternalAttr(
            newReconciledList.get(),
            LwSciSyncInternalAttrKey_WaiterPrimitiveInfo,
            LwSciSyncInternalAttrValPrimitiveType_SysmemSemaphorePayload64b);
    }

    // Assert on LwSciSyncAttrListReconcileAndObjAlloc
    {
        // Need not free
        LwSciSyncAttrList newReconciledList = nullptr;

        auto attrLists = {
            signalerAttrList.get(),
            waiterAttrList.get(),
        };

        auto syncObj = LwSciSyncPeer::reconcileAndAllocate(attrLists, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_NE(syncObj.get(), nullptr);

        ASSERT_EQ(LwSciSyncObjGetAttrList(syncObj.get(), &newReconciledList),
                  LwSciError_Success);

        // Assert on result values
        LwSciSyncPeer::verifyAttr(newReconciledList,
                LwSciSyncAttrKey_RequireDeterministicFences, true);
        LwSciSyncPeer::verifyInternalAttr(
            newReconciledList,
            LwSciSyncInternalAttrKey_SignalerPrimitiveInfo,
            LwSciSyncInternalAttrValPrimitiveType_SysmemSemaphorePayload64b);
        LwSciSyncPeer::verifyInternalAttr(
            newReconciledList,
            LwSciSyncInternalAttrKey_WaiterPrimitiveInfo,
            LwSciSyncInternalAttrValPrimitiveType_SysmemSemaphorePayload64b);
    }
}

class IlwalidLwSciSyncTimestampFormat : public LwSciSyncBasicTest,
    public ::testing::WithParamInterface<std::tuple<LwSciSyncInternalAttrKey>>
{
};
TEST_P(IlwalidLwSciSyncTimestampFormat, IlwalidFormats)
{
    auto params = GetParam();
    LwSciSyncInternalAttrKey signalerTimestampInfoKey = std::get<0>(params);

    LwSciError error = LwSciError_Success;
    auto signalerAttrList = peer.createAttrList();
    auto waiterAttrList = peer.createAttrList();

    // Set up Signaler Attribute List
    {
        ASSERT_EQ(LwSciSyncAttrListSetAttrs(
                  signalerAttrList.get(),
                  LwSciSyncPeer::attrs.cpuSignaler.data(),
                  LwSciSyncPeer::attrs.cpuSignaler.size()),
              LwSciError_Success);

        LwSciSyncTimestampFormat ilwalidFormat =
            (LwSciSyncTimestampFormat)(LwSciSyncTimestampFormat_EmbeddedInPrimitive + 1);

        LwSciSyncAttrValTimestampInfo ilwalidInfo = {
            .format = ilwalidFormat,
            .scaling = {
                .scalingFactorNumerator = 1U,
                .scalingFactorDenominator = 1U,
                .sourceOffset = 0U,
            },
        };
        SET_INTERNAL_ATTR(signalerAttrList.get(), signalerTimestampInfoKey,
                ilwalidInfo);
        uint32_t primitiveCount = 1U;
        SET_INTERNAL_ATTR(signalerAttrList.get(),
                          LwSciSyncInternalAttrKey_SignalerPrimitiveCount,
                          primitiveCount);
    }

    // Set up Waiter Attribute List
    {
        ASSERT_EQ(LwSciSyncAttrListSetAttrs(
                  waiterAttrList.get(),
                  LwSciSyncPeer::attrs.cpuWaiter.data(),
                  LwSciSyncPeer::attrs.cpuWaiter.size()),
              LwSciError_Success);

        SET_ATTR(waiterAttrList.get(), LwSciSyncAttrKey_WaiterRequireTimestamps,
                 true);
    }

    // Assert on LwSciSyncAttrListReconcile
    {
        NegativeTestPrint();
        auto newReconciledList = LwSciSyncPeer::reconcileLists(
            {signalerAttrList.get(), waiterAttrList.get()}, &error);
        ASSERT_EQ(error, LwSciError_BadParameter);
    }

    // Assert on LwSciSyncAttrListReconcileAndObjAlloc
    {
        // Need not free
        LwSciSyncAttrList newReconciledList = nullptr;

        auto attrLists = {
            signalerAttrList.get(),
            waiterAttrList.get(),
        };

        NegativeTestPrint();
        auto syncObj = LwSciSyncPeer::reconcileAndAllocate(attrLists, &error);
        ASSERT_EQ(error, LwSciError_BadParameter);
    }
}
INSTANTIATE_TEST_CASE_P(
    IlwalidLwSciSyncTimestampFormat,
    IlwalidLwSciSyncTimestampFormat,
    ::testing::Values(
    /**
     * @jama{} SignalerTimestampInfo: Invalid LwSciSyncTimestampFormat
     *
     * @verifies @jama{}
     *
     * This test checks that we fail to successfully reconcile
     * SignalerTimestampInfo when LwSciSyncTimestampFormat is invalid.
     */
    std::make_tuple(
        LwSciSyncInternalAttrKey_SignalerTimestampInfo),
    std::make_tuple(
        LwSciSyncInternalAttrKey_SignalerTimestampInfoMulti)
    ));

/**
 * @jama{} SignalerTimestampInfoMulti: Invalid Mutually Exclusive
 * SignalerTimestampInfo
 *
 * @verifies @jama{}
 *
 * This test checks that we fail to successfully reconcile
 * SignalerTimestampInfoMulti when both SignalerTimestampInfo and
 * SignalerTimestampInfoMulti are set.
 */
LWSCISYNC_ATTRIBUTE_LIST_RECONCILE_TEST(
    AttributeListReconciliation,
    SignalerTimestampInfoMulti_Ilwalid_MutuallyExclusiveAttrKeys,
    101)
{
    LwSciError error = LwSciError_Success;
    auto signalerAttrList = peer.createAttrList();
    auto waiterAttrList = peer.createAttrList();

    // Set up Signaler Attribute List
    {
        ASSERT_EQ(LwSciSyncAttrListSetAttrs(
                  signalerAttrList.get(),
                  LwSciSyncPeer::attrs.cpuSignaler.data(),
                  LwSciSyncPeer::attrs.cpuSignaler.size()),
              LwSciError_Success);

        ASSERT_EQ(LwSciSyncAttrListSetInternalAttrs(
                      signalerAttrList.get(),
                      LwSciSyncPeer::attrs.signalerSemaphoreAttrs.data(),
                      LwSciSyncPeer::attrs.signalerSemaphoreAttrs.size()),
                  LwSciError_Success);

        // Sets SignalerTimestampInfo
        LwSciSyncAttrValTimestampInfo timestampInfo = {
            .format = LwSciSyncTimestampFormat_8Byte,
            .scaling = {
                .scalingFactorNumerator = 1U,
                .scalingFactorDenominator = 1U,
                .sourceOffset = 0U,
            },
        };
        SET_INTERNAL_ATTR(signalerAttrList.get(),
            LwSciSyncInternalAttrKey_SignalerTimestampInfo, timestampInfo);
        SET_INTERNAL_ATTR(signalerAttrList.get(),
            LwSciSyncInternalAttrKey_SignalerTimestampInfoMulti, timestampInfo);
    }

    // Set up Waiter Attribute List
    {
        ASSERT_EQ(LwSciSyncAttrListSetAttrs(
                  waiterAttrList.get(),
                  LwSciSyncPeer::attrs.cpuWaiter.data(),
                  LwSciSyncPeer::attrs.cpuWaiter.size()),
              LwSciError_Success);

        ASSERT_EQ(LwSciSyncAttrListSetInternalAttrs(
                      waiterAttrList.get(),
                      LwSciSyncPeer::attrs.waiterSemaphoreAttrs.data(),
                      LwSciSyncPeer::attrs.waiterSemaphoreAttrs.size()),
                  LwSciError_Success);

        SET_ATTR(waiterAttrList.get(), LwSciSyncAttrKey_WaiterRequireTimestamps,
                 true);
    }

    // Assert on LwSciSyncAttrListReconcile
    {
        NegativeTestPrint();
        auto newReconciledList = LwSciSyncPeer::reconcileLists(
            {signalerAttrList.get(), waiterAttrList.get()}, &error);
        ASSERT_EQ(error, LwSciError_BadParameter);
    }

    // Assert on LwSciSyncAttrListReconcileAndObjAlloc
    {
        // Need not free
        LwSciSyncAttrList newReconciledList = nullptr;
        auto attrLists = {
            signalerAttrList.get(),
            waiterAttrList.get()
        };

        NegativeTestPrint();
        auto syncObj = LwSciSyncPeer::reconcileAndAllocate(attrLists, &error);
        ASSERT_EQ(error, LwSciError_BadParameter);
    }
}

class MorePrimitivesThanSignalerTimestampInfo : public LwSciSyncBasicTest,
    public ::testing::WithParamInterface<std::tuple<LwSciSyncInternalAttrKey, LwSciError>>
{
};

TEST_P(MorePrimitivesThanSignalerTimestampInfo,
    MorePrimitivesThanSignalerTimestampInfo)
{
    auto params = GetParam();
    LwSciSyncInternalAttrKey signalerTimestampInfoKey = std::get<0>(params);
    LwSciError expectedErr = std::get<1>(params);

    LwSciError error = LwSciError_Success;
    auto signalerAttrList = peer.createAttrList();
    auto waiterAttrList = peer.createAttrList();

    // Set up Signaler Attribute List
    {
        ASSERT_EQ(LwSciSyncAttrListSetAttrs(
                  signalerAttrList.get(),
                  LwSciSyncPeer::attrs.cpuSignaler.data(),
                  LwSciSyncPeer::attrs.cpuSignaler.size()),
              LwSciError_Success);

        ASSERT_EQ(LwSciSyncAttrListSetInternalAttrs(
                      signalerAttrList.get(),
                      LwSciSyncPeer::attrs.signalerMultiSemaphoreAttrs.data(),
                      LwSciSyncPeer::attrs.signalerMultiSemaphoreAttrs.size()),
                  LwSciError_Success);

        // when used with SignalerPrimitiveInfo, this is correct
        // when used with SignalerPrimitiveInfoMulti, this is too short
        LwSciSyncAttrValTimestampInfo timestampInfo = {
            .format = LwSciSyncTimestampFormat_8Byte,
            .scaling = {
                .scalingFactorNumerator = 1U,
                .scalingFactorDenominator = 1U,
                .sourceOffset = 0U,
            },
        };
        SET_INTERNAL_ATTR(signalerAttrList.get(), signalerTimestampInfoKey,
                timestampInfo);
    }

    // Set up Waiter Attribute List
    {
        ASSERT_EQ(LwSciSyncAttrListSetAttrs(
                  waiterAttrList.get(),
                  LwSciSyncPeer::attrs.cpuWaiter.data(),
                  LwSciSyncPeer::attrs.cpuWaiter.size()),
              LwSciError_Success);

        ASSERT_EQ(LwSciSyncAttrListSetInternalAttrs(
                      waiterAttrList.get(),
                      LwSciSyncPeer::attrs.waiterSemaphoreAttrs.data(),
                      LwSciSyncPeer::attrs.waiterSemaphoreAttrs.size()),
                  LwSciError_Success);

        SET_ATTR(waiterAttrList.get(), LwSciSyncAttrKey_WaiterRequireTimestamps,
                 true);
    }

    // Assert on LwSciSyncAttrListReconcile
    {
        if (expectedErr != LwSciError_Success) {
            NegativeTestPrint();
        }
        auto newReconciledList = LwSciSyncPeer::reconcileLists(
            {signalerAttrList.get(), waiterAttrList.get()}, &error);
        ASSERT_EQ(error, expectedErr);

        if (expectedErr == LwSciError_Success) {
            LwSciSyncPeer::verifyInternalAttr(
                newReconciledList.get(),
                LwSciSyncInternalAttrKey_SignalerPrimitiveInfo,
                LwSciSyncInternalAttrValPrimitiveType_SysmemSemaphore);
            LwSciSyncPeer::verifyInternalAttr(
                newReconciledList.get(),
                LwSciSyncInternalAttrKey_WaiterPrimitiveInfo,
                LwSciSyncInternalAttrValPrimitiveType_SysmemSemaphore);
        }
    }

    // Assert on LwSciSyncAttrListReconcileAndObjAlloc
    {
        // Need not free
        LwSciSyncAttrList newReconciledList = nullptr;
        auto attrLists = {
            signalerAttrList.get(),
            waiterAttrList.get()
        };

        if (expectedErr != LwSciError_Success) {
            NegativeTestPrint();
        }
        auto syncObj = LwSciSyncPeer::reconcileAndAllocate(attrLists, &error);
        ASSERT_EQ(error, expectedErr);

        if (expectedErr == LwSciError_Success) {
            ASSERT_EQ(LwSciSyncObjGetAttrList(syncObj.get(), &newReconciledList),
                  LwSciError_Success);

            LwSciSyncPeer::verifyInternalAttr(
                newReconciledList,
                LwSciSyncInternalAttrKey_SignalerPrimitiveInfo,
                LwSciSyncInternalAttrValPrimitiveType_SysmemSemaphore);
            LwSciSyncPeer::verifyInternalAttr(
                newReconciledList,
                LwSciSyncInternalAttrKey_WaiterPrimitiveInfo,
                LwSciSyncInternalAttrValPrimitiveType_SysmemSemaphore);
        }
    }
};
INSTANTIATE_TEST_CASE_P(
    MorePrimitivesThanSignalerTimestampInfo,
    MorePrimitivesThanSignalerTimestampInfo,
    ::testing::Values(
    /**
     * @jama{} SignalerTimestampInfoMulti: Invalid More SignalerPrimitiveInfo
     * than SignalerTimestampInfoMulti entries.
     *
     * @verifies @jama{}
     *
     * This test checks that we fail to successfully reconcile
     * SignalerTimestampInfo/SignalerTimestampInfoMulti when more
     * SignalerPrimitiveInfo entries are present.
     */
    std::make_tuple(
        LwSciSyncInternalAttrKey_SignalerTimestampInfoMulti,
        LwSciError_BadParameter),
    /**
     * @jama{} SignalerTimestampInfo: More SignalerPrimitiveInfo than
     * SignalerTimestampInfo entries.
     *
     * @verifies @jama{}
     *
     * This test checks that we successfully reconcile SignalerTimestampInfo,
     * when more SignalerPrimitiveInfo entries are present, since when using
     * this key there is a maximum of 1 SignalerTimestampInfo.
     */
    std::make_tuple(
        LwSciSyncInternalAttrKey_SignalerTimestampInfo,
        LwSciError_Success)
    ));


class MoreSignalerTimestampInfoThanPrimitives : public LwSciSyncBasicTest
{
};

/**
 * @jama{} SignalerTimestampInfoMulti: Invalid More
 * SignalerTimestampInfoMulti than SignalerPrimitiveInfo entries.
 *
 * @verifies @jama{}
 *
 * This test checks that we fail to successfully reconcile
 * SignalerTimestampInfoMulti when more SignalerTimestampInfoMulti entries
 * are present compared to SignalerPrimitiveInfo entries.
 */
TEST_F(MoreSignalerTimestampInfoThanPrimitives, Invalid)
{
    LwSciError error = LwSciError_Success;
    auto signalerAttrList = peer.createAttrList();
    auto waiterAttrList = peer.createAttrList();

    // Set up Signaler Attribute List
    {
        ASSERT_EQ(LwSciSyncAttrListSetAttrs(
                  signalerAttrList.get(),
                  LwSciSyncPeer::attrs.cpuSignaler.data(),
                  LwSciSyncPeer::attrs.cpuSignaler.size()),
              LwSciError_Success);

        ASSERT_EQ(LwSciSyncAttrListSetInternalAttrs(
                      signalerAttrList.get(),
                      LwSciSyncPeer::attrs.signalerSemaphoreAttrs.data(),
                      LwSciSyncPeer::attrs.signalerSemaphoreAttrs.size()),
                  LwSciError_Success);

        // We specify multiple timestamp formats corresponding to multiple
        // primitives via SignalerTimestampInfoMulti, but only 1 primitive via
        // SignalerPrimitiveInfo.
        LwSciSyncAttrValTimestampInfo timestampInfo[] = {
            {
                .format = LwSciSyncTimestampFormat_8Byte,
                .scaling = {
                    .scalingFactorNumerator = 1U,
                    .scalingFactorDenominator = 1U,
                    .sourceOffset = 0U,
                },
            },
            {
                .format = LwSciSyncTimestampFormat_8Byte,
                .scaling = {
                    .scalingFactorNumerator = 1U,
                    .scalingFactorDenominator = 1U,
                    .sourceOffset = 0U,
                },
            }
        };
        SET_INTERNAL_ATTR(signalerAttrList.get(),
            LwSciSyncInternalAttrKey_SignalerTimestampInfoMulti,
            timestampInfo);
    }

    // Set up Waiter Attribute List
    {
        ASSERT_EQ(LwSciSyncAttrListSetAttrs(
                  waiterAttrList.get(),
                  LwSciSyncPeer::attrs.cpuWaiter.data(),
                  LwSciSyncPeer::attrs.cpuWaiter.size()),
              LwSciError_Success);

        ASSERT_EQ(LwSciSyncAttrListSetInternalAttrs(
                      waiterAttrList.get(),
                      LwSciSyncPeer::attrs.waiterSemaphoreAttrs.data(),
                      LwSciSyncPeer::attrs.waiterSemaphoreAttrs.size()),
                  LwSciError_Success);

        SET_ATTR(waiterAttrList.get(), LwSciSyncAttrKey_WaiterRequireTimestamps,
                 true);
    }

    // Assert on LwSciSyncAttrListReconcile
    {
        NegativeTestPrint();
        auto newReconciledList = LwSciSyncPeer::reconcileLists(
            {signalerAttrList.get(), waiterAttrList.get()}, &error);
        ASSERT_EQ(error, LwSciError_BadParameter);
    }

    // Assert on LwSciSyncAttrListReconcileAndObjAlloc
    {
        // Need not free
        LwSciSyncAttrList newReconciledList = nullptr;
        auto attrLists = {
            signalerAttrList.get(),
            waiterAttrList.get()
        };

        NegativeTestPrint();
        auto syncObj = LwSciSyncPeer::reconcileAndAllocate(attrLists, &error);
        ASSERT_EQ(error, LwSciError_BadParameter);
    }
}

class TestLwSciSyncReconcileTimestamps : public LwSciSyncInterProcessTest,
    public ::testing::WithParamInterface<std::tuple<LwSciSyncInternalAttrKey>>
{
};

TEST_P(TestLwSciSyncReconcileTimestamps, Success)
{
    auto params = GetParam();
    LwSciSyncInternalAttrKey signalerTimestampInfoKey = std::get<0>(params);

    LwSciError error = LwSciError_Success;
    int peerNumber = 2;
    pid_t pids[peerNumber];
    pid = 0;

    if ((pids[0] = fork()) == 0) {
        pid = 1;
        auto peer = std::make_shared<LwSciSyncIpcPeer>();
        peers.push_back(peer);
        peer->SetUp("lwscisync_a_0");

        auto signalerAttrList = peer->createAttrList(&error);
        ASSERT_EQ(error, LwSciError_Success);

        SET_ATTR(signalerAttrList.get(), LwSciSyncAttrKey_NeedCpuAccess, true);
        SET_ATTR(signalerAttrList.get(), LwSciSyncAttrKey_RequiredPerm,
                 LwSciSyncAccessPerm_SignalOnly);

        LwSciSyncInternalAttrValPrimitiveType primitiveInfo[] = {
            LwSciSyncInternalAttrValPrimitiveType_SysmemSemaphore
        };
        uint32_t primitiveCount = 1U;
        SET_INTERNAL_ATTR(signalerAttrList.get(),
                          LwSciSyncInternalAttrKey_SignalerPrimitiveInfo,
                          primitiveInfo);
        SET_INTERNAL_ATTR(signalerAttrList.get(),
                          LwSciSyncInternalAttrKey_SignalerPrimitiveCount,
                          primitiveCount);
        // Set SignalerTimestampInfo such that only
        // SysmemSemaphore specifies a timestamp format.
        LwSciSyncAttrValTimestampInfo timestampInfo[] = {
            {
                .format = LwSciSyncTimestampFormat_8Byte,
                .scaling = {
                    .scalingFactorNumerator = 1U,
                    .scalingFactorDenominator = 1U,
                    .sourceOffset = 0U,
                },
            },
        };
        SET_INTERNAL_ATTR(signalerAttrList.get(), signalerTimestampInfoKey,
                          timestampInfo);

        // Import Unreconciled Waiter Attribute List
        auto waiterListDescBuf = peer->recvBuf(&error);
        ASSERT_EQ(error, LwSciError_Success);
        auto waiterAttrList =
            peer->importUnreconciledList(waiterListDescBuf, &error);
        ASSERT_EQ(error, LwSciError_Success);

        {
            // There was no primitive that specified a supported timestamp format.
            auto reconciledList = LwSciSyncPeer::attrListReconcile(
                {signalerAttrList.get(), waiterAttrList.get()}, &error);
            ASSERT_EQ(error, LwSciError_Success);
        }

        {
            // There was no primitive that specified a supported timestamp format.
            auto bufObj = LwSciSyncPeer::reconcileAndAllocate(
                {signalerAttrList.get(), waiterAttrList.get()}, &error);
            ASSERT_EQ(error, LwSciError_Success);
        }
    } else if ((pids[1] = fork()) == 0) {
        pid = 2;
        auto peer = std::make_shared<LwSciSyncIpcPeer>();
        peers.push_back(peer);
        peer->SetUp("lwscisync_a_1");

        auto waiterAttrList = peer->createAttrList(&error);
        ASSERT_EQ(error, LwSciError_Success);

        SET_ATTR(waiterAttrList.get(), LwSciSyncAttrKey_NeedCpuAccess, true);
        SET_ATTR(waiterAttrList.get(), LwSciSyncAttrKey_RequiredPerm,
                 LwSciSyncAccessPerm_WaitOnly);
        SET_ATTR(waiterAttrList.get(), LwSciSyncAttrKey_WaiterRequireTimestamps,
                 true);

        LwSciSyncInternalAttrValPrimitiveType primitiveInfo[] = {
            LwSciSyncInternalAttrValPrimitiveType_SysmemSemaphore
        };
        SET_INTERNAL_ATTR(waiterAttrList.get(),
                          LwSciSyncInternalAttrKey_WaiterPrimitiveInfo,
                          primitiveInfo);

        // Export unreconciled waiter list to Peer A
        auto listDescBuf = peer->exportUnreconciledList(
                {waiterAttrList.get()}, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_EQ(peer->sendBuf(listDescBuf), LwSciError_Success);
    } else {
        int status = EXIT_SUCCESS;
        for (auto const& peerPid : pids) {
            TEST_COUT << "Wait for PID " << peerPid << " to exit";
            status |= LwSciSyncInterProcessTest::wait_for_child_fork(peerPid);
        }
        ASSERT_EQ(status, EXIT_SUCCESS);
    }
}
// These two attribute keys should have the identical behaviour
INSTANTIATE_TEST_CASE_P(
    TestLwSciSyncReconcileTimestamps,
    TestLwSciSyncReconcileTimestamps,
    ::testing::Values(
    std::make_tuple(
        LwSciSyncInternalAttrKey_SignalerTimestampInfo),
    std::make_tuple(
        LwSciSyncInternalAttrKey_SignalerTimestampInfoMulti)
    ));

class TestLwSciSyncReconcileTimestampPrimitive
    : public LwSciSyncInterProcessTest,
      public ::testing::WithParamInterface<
          std::tuple<std::array<LwSciSyncInternalAttrValPrimitiveType, 2>,
                     std::array<LwSciSyncAttrValTimestampInfo, 2>>>
{
};

TEST_P(TestLwSciSyncReconcileTimestampPrimitive, Success)
{
    auto params = GetParam();
    std::array<LwSciSyncInternalAttrValPrimitiveType, 2> primitiveInfoParam =
        std::get<0>(params);
    std::array<LwSciSyncAttrValTimestampInfo, 2> timestampInfoParam =
        std::get<1>(params);

    // Colwert std::array to C-style arrays
    LwSciSyncInternalAttrValPrimitiveType primitiveInfo[2]{};
    std::copy(std::begin(primitiveInfoParam), std::end(primitiveInfoParam),
              std::begin(primitiveInfo));
    LwSciSyncAttrValTimestampInfo timestampInfo[2]{};
    std::copy(std::begin(timestampInfoParam), std::end(timestampInfoParam),
              std::begin(timestampInfo));

    LwSciError error = LwSciError_Success;
    int peerNumber = 2;
    pid_t pids[peerNumber];
    pid = 0;

    if ((pids[0] = fork()) == 0) {
        pid = 1;
        auto peer = std::make_shared<LwSciSyncIpcPeer>();
        peers.push_back(peer);
        peer->SetUp("lwscisync_a_0");

        auto signalerAttrList = peer->createAttrList(&error);
        ASSERT_EQ(error, LwSciError_Success);

        SET_ATTR(signalerAttrList.get(), LwSciSyncAttrKey_NeedCpuAccess, true);
        SET_ATTR(signalerAttrList.get(), LwSciSyncAttrKey_RequiredPerm,
                 LwSciSyncAccessPerm_SignalOnly);

        uint32_t primitiveCount = 1U;
        SET_INTERNAL_ATTR(signalerAttrList.get(),
                          LwSciSyncInternalAttrKey_SignalerPrimitiveInfo,
                          primitiveInfo);
        SET_INTERNAL_ATTR(signalerAttrList.get(),
                          LwSciSyncInternalAttrKey_SignalerPrimitiveCount,
                          primitiveCount);
        SET_INTERNAL_ATTR(signalerAttrList.get(),
                          LwSciSyncInternalAttrKey_SignalerTimestampInfoMulti,
                          timestampInfo);

        // Import Unreconciled Waiter Attribute List
        auto waiterListDescBuf = peer->recvBuf(&error);
        ASSERT_EQ(error, LwSciError_Success);
        auto waiterAttrList =
            peer->importUnreconciledList(waiterListDescBuf, &error);
        ASSERT_EQ(error, LwSciError_Success);

        {
            auto reconciledList = LwSciSyncPeer::attrListReconcile(
                {signalerAttrList.get(), waiterAttrList.get()}, &error);
            ASSERT_EQ(error, LwSciError_Success);

            // This should reconcile to a Sysmem semaphore regardless of the
            // order in which they're presented.
            ASSERT_TRUE(LwSciSyncPeer::verifyInternalAttrNew(
                reconciledList.get(),
                LwSciSyncInternalAttrKey_SignalerPrimitiveInfo,
                LwSciSyncInternalAttrValPrimitiveType_SysmemSemaphore));
        }

        {
            // Need not free
            LwSciSyncAttrList newReconciledList = nullptr;

            auto syncObj = LwSciSyncPeer::reconcileAndAllocate(
                {signalerAttrList.get(), waiterAttrList.get()}, &error);
            ASSERT_EQ(error, LwSciError_Success);

            ASSERT_EQ(
                LwSciSyncObjGetAttrList(syncObj.get(), &newReconciledList),
                LwSciError_Success);

            // This should reconcile to a Sysmem semaphore regardless of the
            // order in which they're presented.
            ASSERT_TRUE(LwSciSyncPeer::verifyInternalAttrNew(
                newReconciledList,
                LwSciSyncInternalAttrKey_SignalerPrimitiveInfo,
                LwSciSyncInternalAttrValPrimitiveType_SysmemSemaphore));
        }
    } else if ((pids[1] = fork()) == 0) {
        pid = 2;
        auto peer = std::make_shared<LwSciSyncIpcPeer>();
        peers.push_back(peer);
        peer->SetUp("lwscisync_a_1");

        auto waiterAttrList = peer->createAttrList(&error);
        ASSERT_EQ(error, LwSciError_Success);

        SET_ATTR(waiterAttrList.get(), LwSciSyncAttrKey_NeedCpuAccess, true);
        SET_ATTR(waiterAttrList.get(), LwSciSyncAttrKey_RequiredPerm,
                 LwSciSyncAccessPerm_WaitOnly);
        SET_ATTR(waiterAttrList.get(), LwSciSyncAttrKey_WaiterRequireTimestamps,
                 true);

        SET_INTERNAL_ATTR(waiterAttrList.get(),
                          LwSciSyncInternalAttrKey_WaiterPrimitiveInfo,
                          primitiveInfo);

        // Export unreconciled waiter list to Peer A
        auto listDescBuf =
            peer->exportUnreconciledList({waiterAttrList.get()}, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_EQ(peer->sendBuf(listDescBuf), LwSciError_Success);
    } else {
        int status = EXIT_SUCCESS;
        for (auto const& peerPid : pids) {
            TEST_COUT << "Wait for PID " << peerPid << " to exit";
            status |= LwSciSyncInterProcessTest::wait_for_child_fork(peerPid);
        }
        ASSERT_EQ(status, EXIT_SUCCESS);
    }
}
INSTANTIATE_TEST_CASE_P(
    TestLwSciSyncReconcileTimestampPrimitive,
    TestLwSciSyncReconcileTimestampPrimitive,
    ::testing::Values(
        /**
         * @jama{} ReconcileTimestampPrimitive: Ensure that
         * SignalerTimestampInfoMulti successfully applies the constraints
         * supplied via the Timestamp Info when the constraint is the first
         * index.
         *
         * @verifies @jama{}
         *
         * This test checks that we successfully reconcile to a backing
         * primitive based on the constraints specified by the timestamp format.
         */
        std::make_tuple(
            std::array<LwSciSyncInternalAttrValPrimitiveType, 2>{
                LwSciSyncInternalAttrValPrimitiveType_SysmemSemaphorePayload64b,
                LwSciSyncInternalAttrValPrimitiveType_SysmemSemaphore},
            std::array<LwSciSyncAttrValTimestampInfo, 2>{
                {{
                     .format = LwSciSyncTimestampFormat_Unsupported,
                     .scaling =
                         {
                             .scalingFactorNumerator = 1U,
                             .scalingFactorDenominator = 1U,
                             .sourceOffset = 0U,
                         },
                 },
                 {
                     .format = LwSciSyncTimestampFormat_8Byte,
                     .scaling =
                         {
                             .scalingFactorNumerator = 1U,
                             .scalingFactorDenominator = 1U,
                             .sourceOffset = 0U,
                         },
                 }}}),
        /**
         * @jama{} ReconcileTimestampPrimitive: Ensure that
         * SignalerTimestampInfoMulti successfully applies the constraints
         * supplied via the Timestamp Info when the constraint is the second
         * index.
         *
         * @verifies @jama{}
         *
         * This test checks that we successfully reconcile to a backing
         * primitive based on the constraints specified by the timestamp format
         * and that the order in which the timestamp formats and corresponding
         * primitives are presented doesn't matter.
         */
        std::make_tuple(
            std::array<LwSciSyncInternalAttrValPrimitiveType, 2>{
                LwSciSyncInternalAttrValPrimitiveType_SysmemSemaphore,
                LwSciSyncInternalAttrValPrimitiveType_SysmemSemaphorePayload64b},
            std::array<LwSciSyncAttrValTimestampInfo, 2>{
                {{
                     .format = LwSciSyncTimestampFormat_8Byte,
                     .scaling =
                         {
                             .scalingFactorNumerator = 1U,
                             .scalingFactorDenominator = 1U,
                             .sourceOffset = 0U,
                         },
                 },
                 {
                     .format = LwSciSyncTimestampFormat_Unsupported,
                     .scaling =
                         {
                             .scalingFactorNumerator = 1U,
                             .scalingFactorDenominator = 1U,
                             .sourceOffset = 0U,
                         },
                 }}})));

class TestLwSciSyncReconcileEngineArray
    : public LwSciSyncInterProcessTest,
      public ::testing::WithParamInterface<std::tuple<bool, bool>>
{
};
TEST_P(TestLwSciSyncReconcileEngineArray, Reconciliation)
{
    // Maybe pass std::functions instead? Or figure out generators + GTest
    auto params = GetParam();
    bool setEngineSignaler = std::get<0>(params);
    bool setEngineWaiter = std::get<1>(params);

    LwSciError error = LwSciError_Success;
    int peerNumber = 2;
    pid_t pids[peerNumber];
    pid = 0;

    LwSciSyncHwEngine engine{};
    LwSciError err = LwSciError_Success;
#if !defined(__x86_64__)
    engine.engNamespace = LwSciSyncHwEngine_TegraNamespaceId;
#else
    engine.engNamespace = LwSciSyncHwEngine_ResmanNamespaceId;
#endif
    err = LwSciSyncHwEngCreateIdWithoutInstance(LwSciSyncHwEngName_PCIe,
                                                &engine.rmModuleID);
    ASSERT_EQ(err, LwSciError_Success);

    if ((pids[0] = fork()) == 0) {
        pid = 1;
        auto peer = std::make_shared<LwSciSyncIpcPeer>();
        peers.push_back(peer);
        peer->SetUp("lwscisync_a_0");

        auto signalerAttrList = peer->createAttrList(&error);
        ASSERT_EQ(error, LwSciError_Success);

        ASSERT_EQ(LwSciSyncAttrListSetInternalAttrs(
                      signalerAttrList.get(),
                      LwSciSyncPeer::attrs.signalerPlatformDefaultAttrs.data(),
                      LwSciSyncPeer::attrs.signalerPlatformDefaultAttrs.size()),
                  LwSciError_Success);
        SET_ATTR(signalerAttrList.get(), LwSciSyncAttrKey_RequiredPerm,
                 LwSciSyncAccessPerm_SignalOnly);

        if (setEngineSignaler) {
            SET_INTERNAL_ATTR(signalerAttrList.get(),
                              LwSciSyncInternalAttrKey_EngineArray, engine);
        }

        // Import Unreconciled Waiter Attribute List
        auto waiterListDescBuf = peer->recvBuf(&error);
        ASSERT_EQ(error, LwSciError_Success);
        auto waiterAttrList =
            peer->importUnreconciledList(waiterListDescBuf, &error);
        ASSERT_EQ(error, LwSciError_Success);

        // Reconcile
        auto reconciledList = LwSciSyncPeer::attrListReconcile(
            {signalerAttrList.get(), waiterAttrList.get()}, &error);
        ASSERT_EQ(error, LwSciError_Success);

        if (setEngineSignaler) {
            ASSERT_TRUE(LwSciSyncPeer::verifyInternalAttrNew(
                reconciledList.get(), LwSciSyncInternalAttrKey_EngineArray,
                engine));
        } else {
            ASSERT_TRUE(LwSciSyncPeer::verifyInternalAttrUnset(
                reconciledList.get(), LwSciSyncInternalAttrKey_EngineArray));
        }

        auto reconciledListDesc =
            peer->exportReconciledList(reconciledList.get(), &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_EQ(peer->sendBuf(reconciledListDesc), LwSciError_Success);
    } else if ((pids[1] = fork()) == 0) {
        pid = 2;
        auto peer = std::make_shared<LwSciSyncIpcPeer>();
        peers.push_back(peer);
        peer->SetUp("lwscisync_a_1");

        auto waiterAttrList = peer->createAttrList(&error);
        ASSERT_EQ(error, LwSciError_Success);

        ASSERT_EQ(LwSciSyncAttrListSetInternalAttrs(
                      waiterAttrList.get(),
                      LwSciSyncPeer::attrs.waiterPlatformDefaultAttrs.data(),
                      LwSciSyncPeer::attrs.waiterPlatformDefaultAttrs.size()),
                  LwSciError_Success);
        SET_ATTR(waiterAttrList.get(), LwSciSyncAttrKey_RequiredPerm,
                 LwSciSyncAccessPerm_WaitOnly);

        if (setEngineWaiter) {
            SET_INTERNAL_ATTR(waiterAttrList.get(),
                              LwSciSyncInternalAttrKey_EngineArray, engine);
        }

        auto listDescBuf =
            peer->exportUnreconciledList({waiterAttrList.get()}, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_EQ(peer->sendBuf(listDescBuf), LwSciError_Success);

        auto reconciledListDescBuf = peer->recvBuf(&error);
        ASSERT_EQ(error, LwSciError_Success);
        auto reconciledList = peer->importReconciledList(
            reconciledListDescBuf, {waiterAttrList.get()}, &error);
        ASSERT_EQ(error, LwSciError_Success);

        if (setEngineWaiter) {
            ASSERT_TRUE(LwSciSyncPeer::verifyInternalAttrNew(
                reconciledList.get(), LwSciSyncInternalAttrKey_EngineArray,
                engine));
        } else {
            ASSERT_TRUE(LwSciSyncPeer::verifyInternalAttrUnset(
                reconciledList.get(), LwSciSyncInternalAttrKey_EngineArray));
        }
    } else {
        int status = EXIT_SUCCESS;
        for (auto const& peerPid : pids) {
            TEST_COUT << "Wait for PID " << peerPid << " to exit";
            status |= LwSciSyncInterProcessTest::wait_for_child_fork(peerPid);
        }
        ASSERT_EQ(status, EXIT_SUCCESS);
    }
}
INSTANTIATE_TEST_CASE_P(TestLwSciSyncReconcileEngineArray,
                        TestLwSciSyncReconcileEngineArray,
                        ::testing::Values(
                            // signaler: PCIe, waiter: none
                            std::make_tuple(true, false),
                            // signaler: none, waiter: PCIe
                            std::make_tuple(false, true)));
