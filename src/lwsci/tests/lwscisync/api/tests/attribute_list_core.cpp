/*
 * Copyright (c) 2020-2022 LWPU Corporation.  All Rights Reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property and
 * proprietary rights in and to this software and related documentation.  Any
 * use, reproduction, disclosure or distribution of this software and related
 * documentation without an express license agreement from LWPU Corporation
 * is strictly prohibited.
 */

#include "lwscisync_test_attribute_list.h"

/**
 * @jama{10507587} Attribute List set and get attributes
 * @verifies @jama{12976942} Getting attributes
 * @verifies @jama{12977760} Setting attributes
 * @verifies @jama{13750473} LwSciSync attributes
 * @verifies @jama{13750491} PrimitiveInfo values
 * @verifies @jama{13750638} Permissions values
 */
LWSCISYNC_ATTRIBUTE_LIST_TEST(AttributeListCoreSetAndGetAttributes, 14676431)

/**
 * @jama{10507587} Attribute List set and get attributes
 * Test Case 1: Unreconciled attribute list
 * @setup Allocate empty attribute list
 * @behaviour Try to set each attribute, read it back. The values should be
 * equal. Try to overwrite the attribute - operation should return
 * BadParameter
 */
LWSCISYNC_ATTRIBUTE_LIST_TEST_CASE(AttributeListCoreSetAndGetAttributes,
                                   UnreconciledList)
{
    VERIFY_SET_ATTR(LwSciSyncAttrKey_NeedCpuAccess, true);
    VERIFY_SET_ATTR(LwSciSyncAttrKey_NeedCpuAccess, false);

    VERIFY_SET_ATTR(LwSciSyncAttrKey_RequiredPerm,
                    LwSciSyncAccessPerm_SignalOnly);
    VERIFY_SET_ATTR(LwSciSyncAttrKey_RequiredPerm,
                    LwSciSyncAccessPerm_WaitOnly);
    VERIFY_SET_ATTR(LwSciSyncAttrKey_RequiredPerm,
                    LwSciSyncAccessPerm_WaitSignal);

    VERIFY_SET_ATTR(LwSciSyncAttrKey_WaiterContextInsensitiveFenceExports,
                    true);
    VERIFY_SET_ATTR(LwSciSyncAttrKey_WaiterContextInsensitiveFenceExports,
                    false);

    VERIFY_SET_INTERNAL_ATTR(LwSciSyncInternalAttrKey_SignalerPrimitiveInfo,
                             LwSciSyncInternalAttrValPrimitiveType_Syncpoint);
    VERIFY_SET_INTERNAL_ATTR(
        LwSciSyncInternalAttrKey_SignalerPrimitiveInfo,
        LwSciSyncInternalAttrValPrimitiveType_SysmemSemaphore);
    VERIFY_SET_INTERNAL_ATTR(
        LwSciSyncInternalAttrKey_SignalerPrimitiveInfo,
        LwSciSyncInternalAttrValPrimitiveType_VidmemSemaphore);

    VERIFY_SET_INTERNAL_ATTR(LwSciSyncInternalAttrKey_WaiterPrimitiveInfo,
                             LwSciSyncInternalAttrValPrimitiveType_Syncpoint);
    VERIFY_SET_INTERNAL_ATTR(
        LwSciSyncInternalAttrKey_WaiterPrimitiveInfo,
        LwSciSyncInternalAttrValPrimitiveType_SysmemSemaphore);
    VERIFY_SET_INTERNAL_ATTR(
        LwSciSyncInternalAttrKey_WaiterPrimitiveInfo,
        LwSciSyncInternalAttrValPrimitiveType_VidmemSemaphore);

    LwSciSyncInternalAttrValPrimitiveType primitiveArray[3] = {
        LwSciSyncInternalAttrValPrimitiveType_VidmemSemaphore,
        LwSciSyncInternalAttrValPrimitiveType_Syncpoint,
        LwSciSyncInternalAttrValPrimitiveType_SysmemSemaphore};

    VERIFY_SET_INTERNAL_ATTR(LwSciSyncInternalAttrKey_SignalerPrimitiveInfo,
                             primitiveArray);

    VERIFY_SET_INTERNAL_ATTR(LwSciSyncInternalAttrKey_SignalerPrimitiveCount,
                             0);
    VERIFY_SET_INTERNAL_ATTR(LwSciSyncInternalAttrKey_SignalerPrimitiveCount,
                             1);

    VERIFY_SET_ATTR_OUTPUT_ONLY(LwSciSyncAttrKey_ActualPerm,
                                LwSciSyncAccessPerm_SignalOnly);

    LwSciSyncHwEngine engine{};
    LwSciError err = LwSciError_Success;
#if !defined(__x86_64__)
    engine.engNamespace = LwSciSyncHwEngine_TegraNamespaceId;
#else
    engine.engNamespace = LwSciSyncHwEngine_ResmanNamespaceId;
#endif
    err = LwSciSyncHwEngCreateIdWithoutInstance(LwSciSyncHwEngName_PCIe, &engine.rmModuleID);
    ASSERT_EQ(err, LwSciError_Success);

    VERIFY_SET_INTERNAL_ATTR(LwSciSyncInternalAttrKey_EngineArray, engine);
}

/**
 * @jama{10507587} Attribute List set and get attributes
 * Test Case 2: Reconciled attribute list
 * @setup Allocate CPU signaler and CPU waiter attribute lists, reconcile
 * @behaviour Read each attribute of reconciled list and compare with
 * expected value. Attempt to write to any of the attributes of reconciled list
 * should return BadParameter. Do not check for RequiredPerm, since it is not
 * set in reconciled list.
 */
LWSCISYNC_ATTRIBUTE_LIST_TEST_CASE(AttributeListCoreSetAndGetAttributes,
                                   ReconciledList)
{
    LwSciError error = LwSciError_Success;
    auto listA = peer.createAttrList();
    auto listB = peer.createAttrList();

    ASSERT_EQ(LwSciSyncAttrListSetAttrs(
                  listA.get(), LwSciSyncPeer::attrs.cpuSignaler.data(),
                  LwSciSyncPeer::attrs.cpuSignaler.size()),
              LwSciError_Success);

    ASSERT_EQ(LwSciSyncAttrListSetAttrs(listB.get(),
                                        LwSciSyncPeer::attrs.cpuWaiter.data(),
                                        LwSciSyncPeer::attrs.cpuWaiter.size()),
              LwSciError_Success);

    auto newReconciledList =
        LwSciSyncPeer::reconcileLists({listA.get(), listB.get()}, &error);
    ASSERT_EQ(error, LwSciError_Success);
    ASSERT_NE(newReconciledList.get(), nullptr);

    // Check that attributes are correct and read-only
    VERIFY_ATTR_LIST_IS_RECONCILED(newReconciledList.get());
}

/**
 * @jama{10507587} Attribute List set and get attributes
 * Test Case 3: RepeatedKeys
 * @setup Create 2 lists and set key-value pairs with repeated keys.
 * @behaviour Should return BadParameter.
 */
LWSCISYNC_ATTRIBUTE_LIST_TEST_CASE(AttributeListCoreSetAndGetAttributes,
                                   RepeatedKeys)
{
    LwSciError error = LwSciError_Success;
    auto listA = peer.createAttrList();
    auto listB = peer.createAttrList();

    NegativeTestPrint();
    ASSERT_EQ(LwSciSyncAttrListSetAttrs(
                  listA.get(), LwSciSyncPeer::attrs.repeatedPublicKeys.data(),
                  LwSciSyncPeer::attrs.repeatedPublicKeys.size()),
              LwSciError_BadParameter);

    NegativeTestPrint();
    ASSERT_EQ(LwSciSyncAttrListSetInternalAttrs(
                  listB.get(), LwSciSyncPeer::attrs.repeatedInternalKeys.data(),
                  LwSciSyncPeer::attrs.repeatedInternalKeys.size()),
              LwSciError_BadParameter);
}

/**
 * @jama{14685849} Check if attribute list is reconciled
 * @verifies @jama{12977752} Checking if the list is reconciled
 * @verifies @jama{13750638} Permissions values
 */
LWSCISYNC_ATTRIBUTE_LIST_TEST(AttributeListCoreAttrListIsReconciled, 14685849)

/**
 * @jama{14685849} Check if attribute list is reconciled
 * Test Case 1: Reconcile signaler and waiter lists
 * @setup Create and reconcile signaler and waiter lists
 * @behaviour LwSciSyncAttrListReconcile() should return LwSciError_Success,
 * resulting list should have proper attribute values
 */
LWSCISYNC_ATTRIBUTE_LIST_TEST_CASE(AttributeListCoreAttrListIsReconciled,
                                   SignalerAndWaiter)
{
    LwSciError error = LwSciError_Success;
    auto listA = peer.createAttrList();
    auto listB = peer.createAttrList();

    ASSERT_EQ(LwSciSyncAttrListSetAttrs(
                  listA.get(), LwSciSyncPeer::attrs.cpuSignaler.data(),
                  LwSciSyncPeer::attrs.cpuSignaler.size()),
              LwSciError_Success);

    ASSERT_EQ(LwSciSyncAttrListSetAttrs(listB.get(),
                                        LwSciSyncPeer::attrs.cpuWaiter.data(),
                                        LwSciSyncPeer::attrs.cpuWaiter.size()),
              LwSciError_Success);

    LwSciSyncPeer::checkAttrListIsReconciled(listA.get(), false);
    LwSciSyncPeer::checkAttrListIsReconciled(listB.get(), false);

    auto newReconciledList =
        LwSciSyncPeer::reconcileLists({listA.get(), listB.get()}, &error);
    ASSERT_EQ(error, LwSciError_Success);
    ASSERT_NE(newReconciledList.get(), nullptr);

    // Check that attributes are correct and read-only
    VERIFY_ATTR_LIST_IS_RECONCILED(newReconciledList.get());
}

/**
 * @jama{14685849} Check if attribute list is reconciled
 * Test Case 2: Reconcile waiter and waiterSignaler lists
 * @setup Create and reconcile waiterSignaler and waiter lists
 * @behaviour LwSciSyncAttrListReconcile() should return LwSciError_Success,
 * resulting list should have proper attribute values
 */
LWSCISYNC_ATTRIBUTE_LIST_TEST_CASE(AttributeListCoreAttrListIsReconciled,
                                   WaiterAndWaiterSignaler)
{
    LwSciError error = LwSciError_Success;
    auto listA = peer.createAttrList();
    auto listB = peer.createAttrList();

    ASSERT_EQ(LwSciSyncAttrListSetAttrs(listA.get(),
                                        LwSciSyncPeer::attrs.cpuWaiter.data(),
                                        LwSciSyncPeer::attrs.cpuWaiter.size()),
              LwSciError_Success);

    ASSERT_EQ(LwSciSyncAttrListSetAttrs(
                  listB.get(), LwSciSyncPeer::attrs.cpuWaiterSignaler.data(),
                  LwSciSyncPeer::attrs.cpuWaiterSignaler.size()),
              LwSciError_Success);

    LwSciSyncPeer::checkAttrListIsReconciled(listA.get(), false);
    LwSciSyncPeer::checkAttrListIsReconciled(listB.get(), false);

    auto newReconciledList =
        LwSciSyncPeer::reconcileLists({listA.get(), listB.get()}, &error);
    ASSERT_EQ(error, LwSciError_Success);
    ASSERT_NE(newReconciledList.get(), nullptr);

    // Check that attributes are correct and read-only
    VERIFY_ATTR_LIST_IS_RECONCILED(newReconciledList.get());
}

/**
 * @jama{14676787} Check LwSciSyncAttrListClone() behaviour
 * @verifies @jama{13561775} Attribute list cloning
 */
LWSCISYNC_ATTRIBUTE_LIST_TEST(AttributeListClone, 14676787)

/**
 * @jama{14676787} Check LwSciSyncAttrListClone() behaviour
 * Test Case 1: clone unreconciled attribute list
 * @setup Create unreconciled attrbite list, set all possible attribtes
 * @behaviour LwSciSyncAttrListClone should return LwSciError_Success,
 * resulting list should be writable
 */
LWSCISYNC_ATTRIBUTE_LIST_TEST_CASE(AttributeListClone, UnreconciledList)
{
    auto listA = peer.createAttrList();

    // Set all possible attributes
    VERIFY_ATTR_LIST_IS_WRITABLE(listA.get());

    // Clone list
    LwSciSyncAttrList clonedList = nullptr;
    ASSERT_EQ(LwSciSyncAttrListClone(listA.get(), &clonedList),
              LwSciError_Success);
    auto clonedListPtr = std::shared_ptr<LwSciSyncAttrListRec>(
        clonedList, LwSciSyncAttrListFree);
    LwSciSyncPeer::checkAttrListIsReconciled(clonedList, false);
    LwSciSyncPeer::checkAttrListsEqual(listA.get(), clonedList);

    // Verify attributes in cloned list are writable
    VERIFY_ATTR_LIST_IS_WRITABLE(clonedList);
}

/**
 * @jama{14676787} Check LwSciSyncAttrListClone() behaviour
 * Test Case 2: clone reconciled attribute list
 * @setup Create signaler and waiter attribute lists, reconcile them.
 * @beaviour LwSciSyncAttrListClone on reconciled list should return
 * LwSciError_Success, cloned list is equal to reconciled list and read-only
 */
LWSCISYNC_ATTRIBUTE_LIST_TEST_CASE(AttributeListClone, ReconciledList)
{
    LwSciError error = LwSciError_Success;
    auto listA = peer.createAttrList();
    auto listB = peer.createAttrList();

    ASSERT_EQ(LwSciSyncAttrListSetAttrs(
                  listA.get(), LwSciSyncPeer::attrs.cpuSignaler.data(),
                  LwSciSyncPeer::attrs.cpuSignaler.size()),
              LwSciError_Success);

    ASSERT_EQ(LwSciSyncAttrListSetAttrs(listB.get(),
                                        LwSciSyncPeer::attrs.cpuWaiter.data(),
                                        LwSciSyncPeer::attrs.cpuWaiter.size()),
              LwSciError_Success);

    LwSciSyncPeer::checkAttrListIsReconciled(listA.get(), false);
    LwSciSyncPeer::checkAttrListIsReconciled(listB.get(), false);

    // Reconcile
    auto newReconciledList =
        LwSciSyncPeer::reconcileLists({listA.get(), listB.get()}, &error);
    ASSERT_EQ(error, LwSciError_Success);
    ASSERT_NE(newReconciledList.get(), nullptr);

    // Check that attributes are correct and read-only
    VERIFY_ATTR_LIST_IS_RECONCILED(newReconciledList.get());

    // Clone
    LwSciSyncAttrList clonedList = nullptr;
    ASSERT_EQ(LwSciSyncAttrListClone(newReconciledList.get(), &clonedList),
              LwSciError_Success);
    auto clonedListPtr = std::shared_ptr<LwSciSyncAttrListRec>(
        clonedList, LwSciSyncAttrListFree);

    // Check that attributes are correct and read-only
    // Attributes should be equal to original reconciled list
    VERIFY_ATTR_LIST_IS_RECONCILED(clonedList);
}

/**
 * @jama{14679089} Check LwSciSyncAttrListAppendUnreconciled() behaviour
 * @verifies @jama{13561777} Attribute list appending
 */
LWSCISYNC_ATTRIBUTE_LIST_TEST(AttributeListAppending, 14679089)

/**
 * @jama{14679089} Check LwSciSyncAttrListAppendUnreconciled() behaviour
 * Test Case 1: Unreconciled List
 * @setup Create 2 lists, set attributes
 * @behaviour LwSciSyncAttrListAppendUnreconciled() should return
 * LwSciError_Success, slot count on appended list is equal to 2, appended list
 * is read-only. Attribute values in corresponding slots are equal to input
 * lists.
 */
LWSCISYNC_ATTRIBUTE_LIST_TEST_CASE(AttributeListAppending, UnreconciledList)
{
    auto listA = peer.createAttrList();
    auto listB = peer.createAttrList();

    ASSERT_EQ(LwSciSyncAttrListSetAttrs(
                  listA.get(), LwSciSyncPeer::attrs.cpuSignaler.data(),
                  LwSciSyncPeer::attrs.cpuSignaler.size()),
              LwSciError_Success);

    ASSERT_EQ(LwSciSyncAttrListSetAttrs(listB.get(),
                                        LwSciSyncPeer::attrs.cpuWaiter.data(),
                                        LwSciSyncPeer::attrs.cpuWaiter.size()),
              LwSciError_Success);

    LwSciSyncPeer::checkAttrListIsReconciled(listA.get(), false);
    LwSciSyncPeer::checkAttrListIsReconciled(listB.get(), false);

    std::array<LwSciSyncAttrList, 2> lists = {
        listA.get(),
        listB.get(),
    };

    LwSciSyncAttrList appendedList = nullptr;
    ASSERT_EQ(LwSciSyncAttrListAppendUnreconciled(lists.data(), lists.size(),
                                                  &appendedList),
              LwSciError_Success);
    auto appendedListPtr = std::shared_ptr<LwSciSyncAttrListRec>(
        appendedList, LwSciSyncAttrListFree);

    LwSciSyncPeer::checkAttrListIsReconciled(appendedList, false);

    size_t slot_count_sum = 0;
    for (auto const& list : lists) {
        slot_count_sum += LwSciSyncAttrListGetSlotCount(list);
    }

    ASSERT_EQ(slot_count_sum, LwSciSyncAttrListGetSlotCount(appendedList));

    {
        LwSciSyncAttrKeyValuePair slotAttrs[2];
        memset(slotAttrs, 0, sizeof(slotAttrs));
        slotAttrs[0].attrKey = LwSciSyncAttrKey_NeedCpuAccess;
        slotAttrs[1].attrKey = LwSciSyncAttrKey_RequiredPerm;
        LwSciSyncAttrListSlotGetAttrs(appendedList, 0, &slotAttrs[0], 2);
        ASSERT_EQ(*((const bool*)slotAttrs[0].value), true);
        ASSERT_EQ(*((const uint32_t*)slotAttrs[1].value),
                  LwSciSyncAccessPerm_SignalOnly);
    }
    {
        LwSciSyncAttrKeyValuePair slotAttrs2[2];
        memset(slotAttrs2, 0, sizeof(slotAttrs2));
        slotAttrs2[0].attrKey = LwSciSyncAttrKey_NeedCpuAccess;
        slotAttrs2[1].attrKey = LwSciSyncAttrKey_RequiredPerm;
        LwSciSyncAttrListSlotGetAttrs(appendedList, 1, &slotAttrs2[0], 2);
        ASSERT_EQ(*((const bool*)slotAttrs2[0].value), true);
        ASSERT_EQ(*((const uint32_t*)slotAttrs2[1].value),
                  LwSciSyncAccessPerm_WaitOnly);
    }

    VERIFY_ATTR_LIST_IS_READONLY(appendedList);

    // Reconcile lists and check that reconciled lists are equal
    LwSciError error = LwSciError_Success;
    auto reconciledListA =
        LwSciSyncPeer::reconcileLists({listA.get(), listB.get()}, &error);
    ASSERT_NE(reconciledListA.get(), nullptr)
        << "Failed to reconcile: " << error;

    auto reconciledListB =
        LwSciSyncPeer::reconcileLists({appendedList}, &error);
    ASSERT_NE(reconciledListB.get(), nullptr)
        << "Failed to reconcile: " << error;

    LwSciSyncPeer::checkAttrListsEqual(reconciledListA.get(),
                                       reconciledListB.get());
}

/**
 * @jama{14685751} Check LwSciSyncAttrListGetSlotCount() behaviour
 * @verifies @jama{13561779} Getting number of attribute list slots
 */
LWSCISYNC_ATTRIBUTE_LIST_TEST(AttributeListGetNumberOfSlots, 14685751)

/**
 * @jama{14685751} Check LwSciSyncAttrListGetSlotCount() behaviour
 * Test Case 1: Unreconciled list
 * @setup Create and append 3 lists
 * @behaviour LwSciSyncAttrListGetSlotCount() should return 3
 */
LWSCISYNC_ATTRIBUTE_LIST_TEST_CASE(AttributeListGetNumberOfSlots,
                                   UnreconciledList)
{
    auto listA = peer.createAttrList();
    auto listB = peer.createAttrList();
    auto listC = peer.createAttrList();

    std::array<LwSciSyncAttrList, 3> lists = {
        listA.get(),
        listB.get(),
        listC.get(),
    };

    LwSciSyncAttrList newUnreconciledList = nullptr;
    ASSERT_EQ(LwSciSyncAttrListAppendUnreconciled(lists.data(), lists.size(),
                                                  &newUnreconciledList),
              LwSciError_Success);
    auto newUnreconciledListPtr = std::shared_ptr<LwSciSyncAttrListRec>(
        newUnreconciledList, LwSciSyncAttrListFree);

    size_t slot_count_sum = 0;
    for (auto const& list : lists) {
        slot_count_sum += LwSciSyncAttrListGetSlotCount(list);
    }

    ASSERT_EQ(slot_count_sum,
              LwSciSyncAttrListGetSlotCount(newUnreconciledList));
}

/**
 * @jama{0} Check LwSciSyncAttrListCreate() behaviour
 * @verifies @jama{13561769} Attribute list creation
 */
LWSCISYNC_ATTRIBUTE_LIST_TEST(AttributeListCreate, 0)

/**
 * @jama{0} Check LwSciSyncAttrListCreate() behaviour
 * Test Case 1: Basic scenario
 * @setup Allocate 2 attribute lists from different modules
 * @behaviour LwSciSyncAttrListGetSlotCount() should return 1,
 * LwSciSyncAttrListIsReconciled() should return false.
 * LwSciSyncAttrListReconcile() should return LwSciError_BadParameter.
 */
LWSCISYNC_ATTRIBUTE_LIST_TEST_CASE(AttributeListCreate, Basic)
{
    LwSciError error = LwSciError_Success;
    auto listA = peer.createAttrList();

    ASSERT_EQ(LwSciSyncAttrListGetSlotCount(listA.get()), 1);
    LwSciSyncPeer::checkAttrListIsReconciled(listA.get(), false);

    // Allocate list in another module, then try to reconcile.
    // Reconciliation should fail, since lists belong to different modules.
    auto listB = otherPeer.createAttrList();

    ASSERT_EQ(LwSciSyncAttrListSetAttrs(
                  listA.get(), LwSciSyncPeer::attrs.cpuSignaler.data(),
                  LwSciSyncPeer::attrs.cpuSignaler.size()),
              LwSciError_Success);

    ASSERT_EQ(LwSciSyncAttrListSetAttrs(listB.get(),
                                        LwSciSyncPeer::attrs.cpuWaiter.data(),
                                        LwSciSyncPeer::attrs.cpuWaiter.size()),
              LwSciError_Success);

    {
        NegativeTestPrint();
        auto newReconciledList =
            LwSciSyncPeer::reconcileLists({listA.get(), listB.get()}, &error);

        ASSERT_EQ(error, LwSciError_BadParameter);
        ASSERT_EQ(newReconciledList.get(), nullptr);
    }
}
