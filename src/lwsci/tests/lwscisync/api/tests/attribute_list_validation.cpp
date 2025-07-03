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
#include "lwscisync_interprocess_test.h"
#include "lwscisync_test_attribute_list.h"

/* Declare new tests with this macro to make sure each test case has Jama ID */
#define LWSCISYNC_ATTRIBUTE_LIST_VALIDATION_TEST(testSuite, testName, JamaID)             \
    class _##testSuite##JamaID : public LwSciSyncAttrListTest<JamaID>          \
    {                                                                          \
    };                                                                         \
    TEST_F(_##testSuite##JamaID, testName)
/**
 * @jama{15644706} Successful Validation
 *
 * @verifies @jama{13561835}
 *
 * This test checks that when the Attribute List is successfully reconciled,
 * the output parameter when validating the reconciled attribute list is set
 * to True.
 */
LWSCISYNC_ATTRIBUTE_LIST_VALIDATION_TEST(
    AttributeListValidation,
    SuccessfulValidation,
    15644706)
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

    // Reconcile Attribute Lists
    auto attrLists = {signalerAttrList.get(), waiterAttrList.get()};
    auto newReconciledList = LwSciSyncPeer::reconcileLists(attrLists, &error);
    ASSERT_EQ(error, LwSciError_Success);
    ASSERT_NE(newReconciledList.get(), nullptr);

    // Assert on result values
    // First assert that all attributes are satisfied
    // Note: We have no requirement to reconcile LwSciSyncAttrKey_RequiredPerm,
    // so we don't verify it
    LwSciSyncPeer::verifyAttr(newReconciledList.get(),
                              LwSciSyncAttrKey_NeedCpuAccess, true);
    LwSciSyncPeer::verifyAttr(newReconciledList.get(),
                              LwSciSyncAttrKey_ActualPerm,
                              LwSciSyncAccessPerm_WaitSignal);

    LwSciSyncPeer::verifyInternalAttr(
        newReconciledList.get(), LwSciSyncInternalAttrKey_SignalerPrimitiveInfo,
        LwSciSyncPeer::attrs.defaultPlatformPrimitive);
    LwSciSyncPeer::verifyInternalAttr(
        newReconciledList.get(), LwSciSyncInternalAttrKey_WaiterPrimitiveInfo,
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
    ASSERT_EQ(isReconciled, true);
}

class LwSciSyncAttributeListValidateTest :
    public LwSciSyncTransportTest<15644742>
{
};

/**
 * @jama{15644742} Unsatisfied permissions
 *
 * @verifies @jama{13561839}
 *
 * This test checks that validation returns False if any of the input
 * unreconciled lists has bigger permission in RequiredPerm attribute than
 * the input reconciled attribute list's ActualPerm attribute.
 */
TEST_F(LwSciSyncAttributeListValidateTest, UnsatisfiedPermissions)
{
    LwSciError error = LwSciError_Success;
    pid = fork();
    ASSERT_GE(pid, 0) << "fork failed";

    if (pid == 0) {
        initIpc();
        peer.SetUp("lwscisync_a_0");

        auto signalerAttrList = peer.createAttrList();
        ASSERT_EQ(LwSciSyncAttrListSetAttrs(
                      signalerAttrList.get(),
                      LwSciSyncPeer::attrs.cpuSignaler.data(),
                      LwSciSyncPeer::attrs.cpuSignaler.size()),
                  LwSciError_Success);

        // Import unreconciled attr lists
        auto importedBuf = peer.recvBuf(&error);
        ASSERT_EQ(error, LwSciError_Success);
        auto importedWaiterAttrList =
            peer.importUnreconciledList(importedBuf, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_NE(importedWaiterAttrList.get(), nullptr);

        // Reconcile Attribute Lists
        auto newReconciledList = LwSciSyncPeer::reconcileLists(
            {signalerAttrList.get(), importedWaiterAttrList.get()}, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_NE(newReconciledList.get(), nullptr);

        // Now export the reconciled attribute list
        auto listDescBuf =
            peer.exportReconciledList(newReconciledList.get(), &error);
        ASSERT_EQ(peer.sendBuf(listDescBuf), LwSciError_Success);
        ASSERT_EQ(peer.waitComplete(), LwSciError_Success);
    } else {
        initIpc();
        peer.SetUp("lwscisync_a_1");

        auto waiterAttrList = peer.createAttrList();
        ASSERT_EQ(LwSciSyncAttrListSetAttrs(
                      waiterAttrList.get(),
                      LwSciSyncPeer::attrs.cpuWaiter.data(),
                      LwSciSyncPeer::attrs.cpuWaiter.size()),
                  LwSciError_Success);

        // Export unreconciled attr lists
        auto listDescBuf =
            peer.exportUnreconciledList({waiterAttrList.get()}, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_EQ(peer.sendBuf(listDescBuf), LwSciError_Success);

        // Import Reconciled list
        auto importedBuf = peer.recvBuf(&error);
        ASSERT_EQ(error, LwSciError_Success);
        auto reconciledAttrList = peer.importReconciledList(
            importedBuf, {waiterAttrList.get()}, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_NE(reconciledAttrList.get(), nullptr);

        // Create invalid signaler attribute list
        auto ilwalidSignalerAttrList = peer.createAttrList();
        {
            ASSERT_EQ(
                    LwSciSyncPeer::setAttr(ilwalidSignalerAttrList.get(),
                        LwSciSyncAttrKey_NeedCpuAccess,
                        true),
                    LwSciError_Success);
            ASSERT_EQ(
                    LwSciSyncPeer::setAttr(ilwalidSignalerAttrList.get(),
                        LwSciSyncAttrKey_RequiredPerm,
                        LwSciSyncAccessPerm_WaitSignal),
                    LwSciError_Success);
        }

        // Verify the permissions
        LwSciSyncPeer::verifyAttr(reconciledAttrList.get(),
                                  LwSciSyncAttrKey_ActualPerm,
                                  LwSciSyncAccessPerm_WaitOnly);

        // Now use the invalid attribute lists to validate
        {
            bool isReconciledListValid = true;

            // TODO: Align this with LwSciBuf, which returns
            // LwSciError_ReconciliationFailed
            NegativeTestPrint();
            ASSERT_EQ(LwSciSyncPeer::validateReconciled(
                          {ilwalidSignalerAttrList.get()},
                          reconciledAttrList.get(), &isReconciledListValid),
                      LwSciError_BadParameter);

            ASSERT_EQ(isReconciledListValid, false);
        }

        // wait for other peer to exit
        ASSERT_EQ(peer.signalComplete(), LwSciError_Success);
        ASSERT_EQ(0, wait_for_child_fork(pid));
    }
}

/**
 * @jama{15644770} Unsatisfied Attributes: Unsatisfied primitive
 *
 * @verifies @jama{13561841}
 *
 * This test checks that validation returns False if the primitive in
 * SignalerPrimitiveInfo attribute of the input reconciled attribute list is
 * not present in any of the SignalerPrimitiveInfo, WaiterPrimitiveInfo
 * attributes of the input unreconciled attribute lists.
 */
LWSCISYNC_ATTRIBUTE_LIST_VALIDATION_TEST(
    AttributeListReconciliation,
    ValidationUnsatisfiedAttributes_Primitive,
    15644770)
{
    LwSciError error = LwSciError_Success;
    auto signalerAttrList = peer.createAttrList();
    auto waiterAttrList = peer.createAttrList();
    auto ilwalidSignalerAttrList = peer.createAttrList();
    auto ilwalidWaiterAttrList = peer.createAttrList();

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
                      waiterAttrList.get(), LwSciSyncPeer::attrs.waiter.data(),
                      LwSciSyncPeer::attrs.waiter.size()),
                  LwSciError_Success);

        ASSERT_EQ(LwSciSyncAttrListSetInternalAttrs(
                      waiterAttrList.get(),
                      LwSciSyncPeer::attrs.waiterPlatformDefaultAttrs.data(),
                      LwSciSyncPeer::attrs.waiterPlatformDefaultAttrs.size()),
                  LwSciError_Success);
    }
    // Now create an invalid Signaler attribute list to validate against
    {
        ASSERT_EQ(LwSciSyncAttrListSetAttrs(
                      ilwalidSignalerAttrList.get(),
                      LwSciSyncPeer::attrs.signaler.data(),
                      LwSciSyncPeer::attrs.signaler.size()),
                  LwSciError_Success);

        ASSERT_EQ(LwSciSyncAttrListSetInternalAttrs(
                      ilwalidSignalerAttrList.get(),
                      LwSciSyncPeer::attrs.signalerPlatformIlwalidAttrs.data(),
                      LwSciSyncPeer::attrs.signalerPlatformIlwalidAttrs.size()),
                  LwSciError_Success);
    }
    // Now create an invalid Waiter attribute list to validate against
    {
        ASSERT_EQ(LwSciSyncAttrListSetAttrs(
                      ilwalidWaiterAttrList.get(),
                      LwSciSyncPeer::attrs.waiter.data(),
                      LwSciSyncPeer::attrs.waiter.size()),
                  LwSciError_Success);

        ASSERT_EQ(LwSciSyncAttrListSetInternalAttrs(
                      ilwalidWaiterAttrList.get(),
                      LwSciSyncPeer::attrs.waiterPlatformIlwalidAttrs.data(),
                      LwSciSyncPeer::attrs.waiterPlatformIlwalidAttrs.size()),
                  LwSciError_Success);
    }

    // Reconcile Attribute Lists
    auto newReconciledList = LwSciSyncPeer::reconcileLists(
        {signalerAttrList.get(), waiterAttrList.get()}, &error);
    ASSERT_EQ(error, LwSciError_Success);
    ASSERT_NE(newReconciledList.get(), nullptr);

    // Assert on result values
    // First validate that this is a correct reconciled attribute list
    LwSciSyncPeer::verifyAttr(newReconciledList.get(),
                              LwSciSyncAttrKey_NeedCpuAccess, false);
    LwSciSyncPeer::verifyAttr(newReconciledList.get(),
                              LwSciSyncAttrKey_ActualPerm,
                              LwSciSyncAccessPerm_WaitSignal);

    LwSciSyncPeer::verifyInternalAttr(
        newReconciledList.get(), LwSciSyncInternalAttrKey_SignalerPrimitiveInfo,
        LwSciSyncPeer::attrs.defaultPlatformPrimitive);
    LwSciSyncPeer::verifyInternalAttr(
        newReconciledList.get(), LwSciSyncInternalAttrKey_WaiterPrimitiveInfo,
        LwSciSyncPeer::attrs.defaultPlatformPrimitive);

    // Now use the invalid attribute lists to validate
    {
        bool isReconciledListValid = true;

        // TODO: Align this with LwSciBuf, which returns
        // LwSciError_ReconciliationFailed
        NegativeTestPrint();
        ASSERT_EQ(
            LwSciSyncPeer::validateReconciled(
                {ilwalidSignalerAttrList.get(), ilwalidWaiterAttrList.get()},
                newReconciledList.get(), &isReconciledListValid),
            LwSciError_BadParameter);

        ASSERT_EQ(isReconciledListValid, false);
    }
}

/**
 * @jama{15644822} Unsatisfied Attributes: Unsatisfied SignalerPrimitiveCount CPU Signaler not equal to 1
 *
 * @verifies @jama{13561845}
 *
 * This test checks that validation returns False if there is an input
 * attribute list with signaling permissions and cpuNeedAccess set to True
 * and SignalerPrimitiveCount in the input reconciled list does not equal 1
 */
LWSCISYNC_ATTRIBUTE_LIST_VALIDATION_TEST(
    AttributeListReconciliation,
    ValidationUnsatisfiedAttributes_SignalerPrimitiveCount_CpuNotOne,
    15644822)
{
    LwSciError error = LwSciError_Success;
    auto signalerAttrList = peer.createAttrList();
    auto waiterAttrList = peer.createAttrList();
    auto ilwalidSignalerAttrList = peer.createAttrList();

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
                    9001),
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

    // Set up invalid Signaler Attribute List
    {
        ASSERT_EQ(LwSciSyncAttrListSetAttrs(
                      ilwalidSignalerAttrList.get(),
                      LwSciSyncPeer::attrs.cpuSignaler.data(),
                      LwSciSyncPeer::attrs.cpuSignaler.size()),
                  LwSciError_Success);

        ASSERT_EQ(LwSciSyncAttrListSetInternalAttrs(
                      ilwalidSignalerAttrList.get(),
                      LwSciSyncPeer::attrs.signalerPlatformDefaultAttrs.data(),
                      LwSciSyncPeer::attrs.signalerPlatformDefaultAttrs.size()),
                  LwSciError_Success);
    }

    // Reconcile Attribute Lists
    auto newReconciledList = LwSciSyncPeer::reconcileLists(
        {signalerAttrList.get(), waiterAttrList.get()}, &error);
    ASSERT_EQ(error, LwSciError_Success);
    ASSERT_NE(newReconciledList.get(), nullptr);

    // Assert on result values
    // First validate that this is a correct reconciled attribute list
    LwSciSyncPeer::verifyAttr(newReconciledList.get(),
                              LwSciSyncAttrKey_NeedCpuAccess, false);
    LwSciSyncPeer::verifyAttr(newReconciledList.get(),
                              LwSciSyncAttrKey_ActualPerm,
                              LwSciSyncAccessPerm_WaitSignal);

    LwSciSyncPeer::verifyInternalAttr(
        newReconciledList.get(), LwSciSyncInternalAttrKey_SignalerPrimitiveInfo,
        LwSciSyncPeer::attrs.defaultPlatformPrimitive);
    LwSciSyncPeer::verifyInternalAttr(
        newReconciledList.get(), LwSciSyncInternalAttrKey_WaiterPrimitiveInfo,
        LwSciSyncPeer::attrs.defaultPlatformPrimitive);
    LwSciSyncPeer::verifyInternalAttr(
        newReconciledList.get(),
        LwSciSyncInternalAttrKey_SignalerPrimitiveCount, 9001);

    // Now use the invalid attribute lists to validate
    {
        bool isReconciledListValid = true;

        // TODO: Align this with LwSciBuf, which returns
        // LwSciError_ReconciliationFailed
        NegativeTestPrint();
        ASSERT_EQ(LwSciSyncPeer::validateReconciled(
                      {ilwalidSignalerAttrList.get(), waiterAttrList.get()},
                      newReconciledList.get(), &isReconciledListValid),
                  LwSciError_BadParameter);

        ASSERT_EQ(isReconciledListValid, false);
    }
}

/**
 * @jama{15644838} Unsatisfied Attributes: Unsatisfied SignalerPrimitiveCount mismatch
 *
 * @verifies @jama{13561845}
 *
 * This test checks that validation returns False if there is an input
 * attribute list with signaling permissions and cpuNeedAccess set to False
 * and SignalerPrimitveCount in that attribute list does not match the
 * SignalerPrimitiveCount in the input reconciled attribute list.
 */
LWSCISYNC_ATTRIBUTE_LIST_VALIDATION_TEST(
    AttributeListReconciliation,
    ValidationUnsatisfiedAttributes_SignalerPrimitiveCount_Mismatch,
    15644838)
{
    LwSciError error = LwSciError_Success;
    auto signalerAttrList = peer.createAttrList();
    auto waiterAttrList = peer.createAttrList();
    auto ilwalidSignalerAttrList = peer.createAttrList();

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
                    9001),
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

    // Set up invalid Signaler Attribute List
    {
        ASSERT_EQ(LwSciSyncAttrListSetAttrs(
                      ilwalidSignalerAttrList.get(),
                      LwSciSyncPeer::attrs.signaler.data(),
                      LwSciSyncPeer::attrs.signaler.size()),
                  LwSciError_Success);

        ASSERT_EQ(LwSciSyncAttrListSetInternalAttrs(
                      ilwalidSignalerAttrList.get(),
                      LwSciSyncPeer::attrs.signalerPlatformDefaultAttrs.data(),
                      LwSciSyncPeer::attrs.signalerPlatformDefaultAttrs.size()),
                  LwSciError_Success);
    }

    // Reconcile Attribute Lists
    auto newReconciledList = LwSciSyncPeer::reconcileLists(
        {signalerAttrList.get(), waiterAttrList.get()}, &error);
    ASSERT_EQ(error, LwSciError_Success);
    ASSERT_NE(newReconciledList.get(), nullptr);

    // Assert on result values
    // First validate that this is a correct reconciled attribute list
    LwSciSyncPeer::verifyAttr(newReconciledList.get(),
                              LwSciSyncAttrKey_NeedCpuAccess, false);
    LwSciSyncPeer::verifyAttr(newReconciledList.get(),
                              LwSciSyncAttrKey_ActualPerm,
                              LwSciSyncAccessPerm_WaitSignal);

    LwSciSyncPeer::verifyInternalAttr(
        newReconciledList.get(), LwSciSyncInternalAttrKey_SignalerPrimitiveInfo,
        LwSciSyncPeer::attrs.defaultPlatformPrimitive);
    LwSciSyncPeer::verifyInternalAttr(
        newReconciledList.get(), LwSciSyncInternalAttrKey_WaiterPrimitiveInfo,
        LwSciSyncPeer::attrs.defaultPlatformPrimitive);
    LwSciSyncPeer::verifyInternalAttr(
        newReconciledList.get(),
        LwSciSyncInternalAttrKey_SignalerPrimitiveCount, 9001);

    // Now use the invalid attribute lists to validate
    {
        bool isReconciledListValid = true;

        // TODO: Align this with LwSciBuf, which returns
        // LwSciError_ReconciliationFailed
        NegativeTestPrint();
        ASSERT_EQ(LwSciSyncPeer::validateReconciled(
                      {ilwalidSignalerAttrList.get(), waiterAttrList.get()},
                      newReconciledList.get(), &isReconciledListValid),
                  LwSciError_BadParameter);

        ASSERT_EQ(isReconciledListValid, false);
    }
}

/**
 * @jama{22781534} Unsatisfied needCpuAccess
 *
 * @verifies @jama{21749789}
 *
 * This test checks that validation returns False if any of the input
 * unreconciled list has needCpuAccess set to True and the input reconciled
 * list has needCpuAccess set to False.
 */
LWSCISYNC_ATTRIBUTE_LIST_VALIDATION_TEST(
    AttributeListReconciliation,
    ValidationUnsatisfiedAttributes_Unsatisfied_NeedCpuAccess,
    22781534)
{
    LwSciError error = LwSciError_Success;
    auto signalerAttrList = peer.createAttrList();
    auto waiterAttrList = peer.createAttrList();
    auto cpuSignalerAttrList = peer.createAttrList();

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

    // Set up cpu Signaler Attribute List
    {
        ASSERT_EQ(LwSciSyncAttrListSetAttrs(
                      cpuSignalerAttrList.get(),
                      LwSciSyncPeer::attrs.cpuSignaler.data(),
                      LwSciSyncPeer::attrs.cpuSignaler.size()),
                  LwSciError_Success);

        ASSERT_EQ(LwSciSyncAttrListSetInternalAttrs(
                      cpuSignalerAttrList.get(),
                      LwSciSyncPeer::attrs.signalerPlatformDefaultAttrs.data(),
                      LwSciSyncPeer::attrs.signalerPlatformDefaultAttrs.size()),
                  LwSciError_Success);
    }

    // Reconcile Attribute Lists
    auto newReconciledList = LwSciSyncPeer::reconcileLists(
        { signalerAttrList.get(), waiterAttrList.get() }, &error);
    ASSERT_EQ(error, LwSciError_Success);
    ASSERT_NE(newReconciledList.get(), nullptr);

    // Assert on result values
    // First validate that this is a correct reconciled attribute list
    LwSciSyncPeer::verifyAttr(newReconciledList.get(),
                              LwSciSyncAttrKey_NeedCpuAccess, false);
    LwSciSyncPeer::verifyAttr(newReconciledList.get(),
                              LwSciSyncAttrKey_ActualPerm,
                              LwSciSyncAccessPerm_WaitSignal);

    // Now use the cpu signaler attribute lists to validate
    {
        bool isReconciledListValid = true;

        NegativeTestPrint();
        ASSERT_EQ(LwSciSyncPeer::validateReconciled(
                      { cpuSignalerAttrList.get(), waiterAttrList.get() },
                      newReconciledList.get(), &isReconciledListValid),
                  LwSciError_BadParameter);

        ASSERT_EQ(isReconciledListValid, false);
    }
}

/**
 * @jama{} Unsatisfied Attributes: RequireDeterministicFences mismatch
 *
 * @verifies @jama{}
 *
 * This test checks that validation returns False if there is an input
 * attribute list which requires deterministic primitives and
 * RequireDeterministicFences in that attribute list does not match the
 * RequireDeterministicFences attribute key in the input reconciled attribute
 * list.
 */
LWSCISYNC_ATTRIBUTE_LIST_VALIDATION_TEST(
    AttributeListReconciliation,
    ValidationUnsatisfiedAttributes_RequireDeterministicFences_Mismatch,
    1)
{
    // Note: Since Semaphores are the only deterministic primitive supported on
    // all platforms that LwSciSync supports, we need to force the usage of the
    // Semaphore primitive.
    LwSciError error = LwSciError_Success;
    auto signalerAttrList = peer.createAttrList();
    auto waiterAttrList = peer.createAttrList();
    auto deterministicWaiterAttrList = peer.createAttrList();

    // Set up Signaler Attribute List
    {
        ASSERT_EQ(LwSciSyncAttrListSetAttrs(
                      signalerAttrList.get(),
                      LwSciSyncPeer::attrs.signaler.data(),
                      LwSciSyncPeer::attrs.signaler.size()),
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
                      LwSciSyncPeer::attrs.waiter.data(),
                      LwSciSyncPeer::attrs.waiter.size()),
                  LwSciError_Success);

        ASSERT_EQ(LwSciSyncAttrListSetInternalAttrs(
                      waiterAttrList.get(),
                      LwSciSyncPeer::attrs.waiterSemaphoreAttrs.data(),
                      LwSciSyncPeer::attrs.waiterSemaphoreAttrs.size()),
                  LwSciError_Success);
    }

    // Set up deterministic Waiter Attribute List
    {
        ASSERT_EQ(LwSciSyncAttrListSetAttrs(
                      deterministicWaiterAttrList.get(),
                      LwSciSyncPeer::attrs.waiter.data(),
                      LwSciSyncPeer::attrs.waiter.size()),
                  LwSciError_Success);

        ASSERT_EQ(LwSciSyncAttrListSetInternalAttrs(
                      deterministicWaiterAttrList.get(),
                      LwSciSyncPeer::attrs.waiterSemaphoreAttrs.data(),
                      LwSciSyncPeer::attrs.waiterSemaphoreAttrs.size()),
                  LwSciError_Success);

        // Set RequireDeterministicFences
        ASSERT_EQ(
                LwSciSyncPeer::setAttr(deterministicWaiterAttrList.get(),
                    LwSciSyncAttrKey_RequireDeterministicFences,
                    true),
                LwSciError_Success);
    }

    // Reconcile Attribute Lists
    auto newReconciledList = LwSciSyncPeer::reconcileLists(
        {signalerAttrList.get(), waiterAttrList.get()}, &error);
    ASSERT_EQ(error, LwSciError_Success);
    ASSERT_NE(newReconciledList.get(), nullptr);

    // Assert on result values
    // First validate that this is a correct reconciled attribute list
    // We expect that the reconciled attribute list should not have
    // RequireDeterministicFences set to true.
    LwSciSyncPeer::verifyAttr(newReconciledList.get(),
                              LwSciSyncAttrKey_RequireDeterministicFences,
                              false);

    LwSciSyncPeer::verifyInternalAttr(
        newReconciledList.get(), LwSciSyncInternalAttrKey_SignalerPrimitiveInfo,
        LwSciSyncInternalAttrValPrimitiveType_SysmemSemaphore);
    LwSciSyncPeer::verifyInternalAttr(
        newReconciledList.get(), LwSciSyncInternalAttrKey_WaiterPrimitiveInfo,
        LwSciSyncInternalAttrValPrimitiveType_SysmemSemaphore);

    // Now use the invalid attribute lists to validate
    {
        bool isReconciledListValid = true;

        // TODO: Align this with LwSciBuf, which returns
        // LwSciError_ReconciliationFailed
        NegativeTestPrint();
        ASSERT_EQ(LwSciSyncPeer::validateReconciled(
                      {signalerAttrList.get(), deterministicWaiterAttrList.get()},
                      newReconciledList.get(), &isReconciledListValid),
                  LwSciError_BadParameter);

        ASSERT_EQ(isReconciledListValid, false);
    }
}

//
class TimestampFormatValidation : public LwSciSyncBasicTest,
    public ::testing::WithParamInterface<std::tuple<LwSciSyncInternalAttrKey>>
{
};

/**
 * @jama{} Unsatisfied Attributes: Different timestamp formats
 *
 * @verifies @jama{}
 *
 * This test checks that validation returns False if there is an input
 * attribute list which sets the Timestamp Info key but the reconciled
 * attribute list does not for each of SignalerTimestampInfo,
 * SignalerTimestampInfoMulti.
 */
TEST_P(TimestampFormatValidation, DifferentTimestampFormats)
{
    auto params = GetParam();
    LwSciSyncInternalAttrKey timestampInfoKey = std::get<0>(params);

    LwSciError error = LwSciError_Success;
    auto signalerAttrList = peer.createAttrList();
    auto waiterAttrList = peer.createAttrList();
    auto differentTimestampFormatList = peer.createAttrList();

    const LwSciSyncAttrValTimestampInfo timestampInfo = {
        .format = LwSciSyncTimestampFormat_8Byte,
        .scaling = {
            .scalingFactorNumerator = 1U,
            .scalingFactorDenominator = 1U,
            .sourceOffset = 0U,
        },
    };

    // Set up Signaler Attribute List
    {
        ASSERT_EQ(LwSciSyncAttrListSetAttrs(
                      signalerAttrList.get(),
                      LwSciSyncPeer::attrs.signaler.data(),
                      LwSciSyncPeer::attrs.signaler.size()),
                  LwSciError_Success);

        ASSERT_EQ(LwSciSyncAttrListSetInternalAttrs(
                      signalerAttrList.get(),
                      LwSciSyncPeer::attrs.signalerSemaphoreAttrs.data(),
                      LwSciSyncPeer::attrs.signalerSemaphoreAttrs.size()),
                  LwSciError_Success);

        SET_INTERNAL_ATTR(signalerAttrList.get(), timestampInfoKey,
                          timestampInfo);
    }

    // Set up Waiter Attribute List
    {
        ASSERT_EQ(LwSciSyncAttrListSetAttrs(
                      waiterAttrList.get(),
                      LwSciSyncPeer::attrs.waiter.data(),
                      LwSciSyncPeer::attrs.waiter.size()),
                  LwSciError_Success);
        SET_ATTR(waiterAttrList.get(), LwSciSyncAttrKey_WaiterRequireTimestamps,
                 true);

        ASSERT_EQ(LwSciSyncAttrListSetInternalAttrs(
                      waiterAttrList.get(),
                      LwSciSyncPeer::attrs.waiterSemaphoreAttrs.data(),
                      LwSciSyncPeer::attrs.waiterSemaphoreAttrs.size()),
                  LwSciError_Success);
    }

    // Set up invalid signaler timestamp list containing the other timestamp
    // format.
    {
        ASSERT_EQ(LwSciSyncAttrListSetAttrs(
                      differentTimestampFormatList.get(),
                      LwSciSyncPeer::attrs.signaler.data(),
                      LwSciSyncPeer::attrs.signaler.size()),
                  LwSciError_Success);

        ASSERT_EQ(LwSciSyncAttrListSetInternalAttrs(
                      differentTimestampFormatList.get(),
                      LwSciSyncPeer::attrs.signalerSemaphoreAttrs.data(),
                      LwSciSyncPeer::attrs.signalerSemaphoreAttrs.size()),
                  LwSciError_Success);

        LwSciSyncAttrValTimestampInfo differentTimestampInfo = {
            .format = LwSciSyncTimestampFormat_16Byte,
            .scaling = {
                .scalingFactorNumerator = 1U,
                .scalingFactorDenominator = 1U,
                .sourceOffset = 0U,
            },
        };
        SET_INTERNAL_ATTR(differentTimestampFormatList.get(), timestampInfoKey,
                          differentTimestampInfo);
    }

    // Reconcile Attribute Lists
    auto newReconciledList = LwSciSyncPeer::reconcileLists(
        {signalerAttrList.get(), waiterAttrList.get()}, &error);
    ASSERT_EQ(error, LwSciError_Success);
    ASSERT_NE(newReconciledList.get(), nullptr);

    // Assert on result values
    LwSciSyncPeer::verifyAttrNew(newReconciledList.get(),
            LwSciSyncAttrKey_WaiterRequireTimestamps, true);
    bool isReconciled = true;
    ASSERT_EQ(LwSciSyncPeer::validateReconciled(
                  {signalerAttrList.get(), waiterAttrList.get()},
                  newReconciledList.get(), &isReconciled),
              LwSciError_Success);
    ASSERT_TRUE(isReconciled);

    {
        isReconciled = false;
        NegativeTestPrint();

        LwSciSyncPeer::verifyInternalAttr(newReconciledList.get(),
            timestampInfoKey, timestampInfo);
        // Should fail, since the other attribute key was set.
        ASSERT_EQ(LwSciSyncPeer::validateReconciled(
                      {differentTimestampFormatList.get(), waiterAttrList.get()},
                      newReconciledList.get(), &isReconciled),
                  LwSciError_BadParameter);
        ASSERT_FALSE(isReconciled);
    }
}
// These two attribute keys should have the identical behaviour
INSTANTIATE_TEST_CASE_P(
    TimestampFormatValidation,
    TimestampFormatValidation,
    ::testing::Values(
        /**
         * @jama{} Unsatisfied Attributes: SignalerTimestampInfo timestamp format
         *
         * @verifies @jama{}
         *
         * This test checks that validation returns False if there is an input
         * attribute list which sets the SignalerTimestampInfo key which is
         * different than that set on the reconciled LwSciSyncAttrList.
         */
        std::make_tuple(
            LwSciSyncInternalAttrKey_SignalerTimestampInfo),
        /**
         * @jama{} Unsatisfied Attributes: SignalerTimestampInfoMulti Timestamp
         * format
         *
         * @verifies @jama{}
         *
         * This test checks that validation returns False if there is an input
         * attribute list which sets the TimestampInfoMulti key which is
         * different than that set on the reconciled LwSciSyncAttrList.
         */
        std::make_tuple(
            LwSciSyncInternalAttrKey_SignalerTimestampInfoMulti)
    )
);

class AttributeListValidation : public LwSciSyncBasicTest,
    public ::testing::WithParamInterface<std::tuple<LwSciSyncInternalAttrKey, LwSciSyncInternalAttrKey> >
{
};

/**
 * @jama{} Unsatisfied Attributes: SignalerTimestampInfo mutually exclusive
 *
 * @verifies @jama{}
 *
 * This test checks that validation returns False if there is an input
 * attribute list which sets the Timestamp Info key but the reconciled
 * attribute list does not for each of SignalerTimestampInfo,
 * SignalerTimestampInfoMulti when timestamps are requested by the waiter.
 */
TEST_P(AttributeListValidation,
    ValidationUnsatisfiedAttributes_SignalerTimestampInfo_Exclusive)
{
    auto params = GetParam();
    LwSciSyncInternalAttrKey validKey = std::get<0>(params);
    LwSciSyncInternalAttrKey ilwalidKey = std::get<1>(params);

    LwSciError error = LwSciError_Success;
    auto signalerAttrList = peer.createAttrList();
    auto waiterAttrList = peer.createAttrList();
    auto multiTimestampList = peer.createAttrList();

    // Set up Signaler Attribute List
    {
        ASSERT_EQ(LwSciSyncAttrListSetAttrs(
                      signalerAttrList.get(),
                      LwSciSyncPeer::attrs.signaler.data(),
                      LwSciSyncPeer::attrs.signaler.size()),
                  LwSciError_Success);

        ASSERT_EQ(LwSciSyncAttrListSetInternalAttrs(
                      signalerAttrList.get(),
                      LwSciSyncPeer::attrs.signalerSemaphoreAttrs.data(),
                      LwSciSyncPeer::attrs.signalerSemaphoreAttrs.size()),
                  LwSciError_Success);

        LwSciSyncAttrValTimestampInfo timestampInfo = {
            .format = LwSciSyncTimestampFormat_8Byte,
            .scaling = {
                .scalingFactorNumerator = 1U,
                .scalingFactorDenominator = 1U,
                .sourceOffset = 0U,
            },
        };
        SET_INTERNAL_ATTR(signalerAttrList.get(), validKey, timestampInfo);
    }

    // Set up Waiter Attribute List
    {
        ASSERT_EQ(LwSciSyncAttrListSetAttrs(
                      waiterAttrList.get(),
                      LwSciSyncPeer::attrs.waiter.data(),
                      LwSciSyncPeer::attrs.waiter.size()),
                  LwSciError_Success);
        SET_ATTR(waiterAttrList.get(), LwSciSyncAttrKey_WaiterRequireTimestamps,
                 true);

        ASSERT_EQ(LwSciSyncAttrListSetInternalAttrs(
                      waiterAttrList.get(),
                      LwSciSyncPeer::attrs.waiterSemaphoreAttrs.data(),
                      LwSciSyncPeer::attrs.waiterSemaphoreAttrs.size()),
                  LwSciError_Success);
    }

    // Set up invalid signaler timestamp list containing the other key
    {
        ASSERT_EQ(LwSciSyncAttrListSetAttrs(
                      multiTimestampList.get(),
                      LwSciSyncPeer::attrs.signaler.data(),
                      LwSciSyncPeer::attrs.signaler.size()),
                  LwSciError_Success);

        ASSERT_EQ(LwSciSyncAttrListSetInternalAttrs(
                      multiTimestampList.get(),
                      LwSciSyncPeer::attrs.signalerSemaphoreAttrs.data(),
                      LwSciSyncPeer::attrs.signalerSemaphoreAttrs.size()),
                  LwSciError_Success);

        LwSciSyncAttrValTimestampInfo timestampInfo = {
            .format = LwSciSyncTimestampFormat_8Byte,
            .scaling = {
                .scalingFactorNumerator = 1U,
                .scalingFactorDenominator = 1U,
                .sourceOffset = 0U,
            },
        };
        SET_INTERNAL_ATTR(multiTimestampList.get(), ilwalidKey, timestampInfo);
    }

    // Reconcile Attribute Lists
    auto newReconciledList = LwSciSyncPeer::reconcileLists(
        {signalerAttrList.get(), waiterAttrList.get()}, &error);
    ASSERT_EQ(error, LwSciError_Success);
    ASSERT_NE(newReconciledList.get(), nullptr);

    // Assert on result values
    // First validate that this is a correct reconciled attribute list
    // We expect that the reconciled attribute list should not have
    // RequireDeterministicFences set to true.
    ASSERT_TRUE(
        LwSciSyncPeer::verifyInternalAttrUnset(newReconciledList.get(),
            ilwalidKey));
    ASSERT_TRUE(LwSciSyncPeer::verifyAttrNew(newReconciledList.get(),
            LwSciSyncAttrKey_WaiterRequireTimestamps, true));
    bool isReconciledValid = false;
    ASSERT_EQ(LwSciSyncPeer::validateReconciled(
                  {signalerAttrList.get(), waiterAttrList.get()},
                  newReconciledList.get(), &isReconciledValid),
              LwSciError_Success);
    ASSERT_TRUE(isReconciledValid);

    {
        isReconciledValid = false;
        NegativeTestPrint();

        LwSciSyncAttrValTimestampInfo timestampInfo = {
            .format = LwSciSyncTimestampFormat_8Byte,
            .scaling = {
                .scalingFactorNumerator = 1U,
                .scalingFactorDenominator = 1U,
                .sourceOffset = 0U,
            },
        };

        ASSERT_TRUE(
            LwSciSyncPeer::verifyInternalAttrNew(multiTimestampList.get(),
                ilwalidKey, timestampInfo));
        // Should fail, since the other attribute key was set.
        ASSERT_EQ(LwSciSyncPeer::validateReconciled(
                      {multiTimestampList.get(), waiterAttrList.get()},
                      newReconciledList.get(), &isReconciledValid),
                  LwSciError_BadParameter);
        ASSERT_FALSE(isReconciledValid);
    }
}
// These two attribute keys should have the identical behaviour
INSTANTIATE_TEST_CASE_P(
    AttributeListValidation,
    AttributeListValidation,
    ::testing::Values(
    std::make_tuple(
        LwSciSyncInternalAttrKey_SignalerTimestampInfo,
        LwSciSyncInternalAttrKey_SignalerTimestampInfoMulti),
    std::make_tuple(
        LwSciSyncInternalAttrKey_SignalerTimestampInfoMulti,
        LwSciSyncInternalAttrKey_SignalerTimestampInfo)
    ));

class LwSciSyncValidateWaiterRequireTimestamps : public LwSciSyncBasicTest
{
};

/**
 * @jama{} Unsatisfied Attributes: WaiterRequireTimestamps mismatch
 *
 * @verifies @jama{}
 *
 * This test checks that validation returns False if there is an input
 * attribute list which sets the WaiterRequireTimestamps key but the reconciled
 * attribute list does not.
 */
TEST_F(LwSciSyncValidateWaiterRequireTimestamps, Mismatch)
{
    LwSciError error = LwSciError_Success;

    auto signalerAttrList = peer.createAttrList();
    auto waiterAttrList = peer.createAttrList();
    auto timestampWaiterAttrList = peer.createAttrList();

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

    // Set up Timestamp Waiter Attribute List
    {
        ASSERT_EQ(LwSciSyncAttrListSetAttrs(
                      timestampWaiterAttrList.get(),
                      LwSciSyncPeer::attrs.cpuWaiter.data(),
                      LwSciSyncPeer::attrs.cpuWaiter.size()),
                  LwSciError_Success);
        SET_ATTR(timestampWaiterAttrList.get(),
                 LwSciSyncAttrKey_WaiterRequireTimestamps, true);

        ASSERT_EQ(LwSciSyncAttrListSetInternalAttrs(
                      timestampWaiterAttrList.get(),
                      LwSciSyncPeer::attrs.waiterPlatformDefaultAttrs.data(),
                      LwSciSyncPeer::attrs.waiterPlatformDefaultAttrs.size()),
                  LwSciError_Success);
    }

    // Reconcile Attribute Lists
    auto newReconciledList = LwSciSyncPeer::reconcileLists(
        {signalerAttrList.get(), waiterAttrList.get()}, &error);
    ASSERT_EQ(error, LwSciError_Success);
    ASSERT_NE(newReconciledList.get(), nullptr);

    // Assert on result values
    // First validate that this is a correct reconciled attribute list
    // We expect that the reconciled attribute list should not have
    // WaiterRequireTimestamps set to true.
    ASSERT_TRUE(LwSciSyncPeer::verifyAttrNew(newReconciledList.get(),
        LwSciSyncAttrKey_WaiterRequireTimestamps, false));
    bool isReconciled = true;
    ASSERT_EQ(LwSciSyncPeer::validateReconciled(
                  {signalerAttrList.get(), waiterAttrList.get()},
                  newReconciledList.get(), &isReconciled),
              LwSciError_Success);
    ASSERT_TRUE(isReconciled);

    {
        NegativeTestPrint();

        isReconciled = false;
        ASSERT_EQ(LwSciSyncPeer::validateReconciled(
                      {signalerAttrList.get(), timestampWaiterAttrList.get()},
                      newReconciledList.get(), &isReconciled),
                  LwSciError_BadParameter);
        ASSERT_FALSE(isReconciled);
    }
}

class LwSciSyncValidateTimestampFormats : public LwSciSyncBasicTest,
    public ::testing::WithParamInterface<std::tuple<LwSciSyncInternalAttrKey>>
{
};

/**
 * @jama{} Timestamp Formats: Invalid Timestamp Format: Scaling
 *
 * @verifies @jama{}
 *
 * This test checks that validation returns False if there is an input
 * attribute list which uses an invalid Timestamp Format for each of
 * LwSciSyncInternalAttrKey_SignalerTimestampInfo and
 * LwSciSyncInternalAttrKey_SignalerTimestampInfoMulti.
 */
TEST_P(LwSciSyncValidateTimestampFormats, Invalid)
{
    auto params = GetParam();
    LwSciSyncInternalAttrKey timestampInfoKey = std::get<0>(params);

    LwSciError error = LwSciError_Success;

    auto signalerAttrList = peer.createAttrList();
    auto waiterAttrList = peer.createAttrList();
    auto ilwalidFormatSignalerAttrList = peer.createAttrList();

    LwSciSyncAttrValTimestampInfo timestampInfo = {
        .format = LwSciSyncTimestampFormat_8Byte,
        .scaling = {
            .scalingFactorNumerator = 0U,
            .scalingFactorDenominator = 1U,
            .sourceOffset = 0U,
        },
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
                      LwSciSyncPeer::attrs.signalerPlatformDefaultAttrs.data(),
                      LwSciSyncPeer::attrs.signalerPlatformDefaultAttrs.size()),
                  LwSciError_Success);

        SET_INTERNAL_ATTR(signalerAttrList.get(), timestampInfoKey,
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
                      LwSciSyncPeer::attrs.waiterPlatformDefaultAttrs.data(),
                      LwSciSyncPeer::attrs.waiterPlatformDefaultAttrs.size()),
                  LwSciError_Success);

        SET_ATTR(waiterAttrList.get(), LwSciSyncAttrKey_WaiterRequireTimestamps,
                 true);
    }

    // Set up Invalid Format Timestamp Signaler Attribute List
    {
        ASSERT_EQ(LwSciSyncAttrListSetAttrs(
                      ilwalidFormatSignalerAttrList.get(),
                      LwSciSyncPeer::attrs.cpuSignaler.data(),
                      LwSciSyncPeer::attrs.cpuSignaler.size()),
                  LwSciError_Success);

        ASSERT_EQ(LwSciSyncAttrListSetInternalAttrs(
                      ilwalidFormatSignalerAttrList.get(),
                      LwSciSyncPeer::attrs.signalerPlatformDefaultAttrs.data(),
                      LwSciSyncPeer::attrs.signalerPlatformDefaultAttrs.size()),
                  LwSciError_Success);

        LwSciSyncAttrValTimestampInfo ilwalidFormat = {
            .format = LwSciSyncTimestampFormat_8Byte,
            .scaling = {
                .scalingFactorNumerator = 0U,
                .scalingFactorDenominator = 0U,
                .sourceOffset = 0U,
            },
        };
        SET_INTERNAL_ATTR(ilwalidFormatSignalerAttrList.get(),
                          timestampInfoKey,
                          ilwalidFormat);
    }

    // Reconcile Attribute Lists
    auto newReconciledList = LwSciSyncPeer::reconcileLists(
        {signalerAttrList.get(), waiterAttrList.get()}, &error);
    ASSERT_EQ(error, LwSciError_Success);
    ASSERT_NE(newReconciledList.get(), nullptr);

    // Assert on result values
    // First validate that this is a correct reconciled attribute list
    ASSERT_TRUE(LwSciSyncPeer::verifyInternalAttrNew(newReconciledList.get(),
        timestampInfoKey, timestampInfo));
    bool isReconciled = true;
    ASSERT_EQ(LwSciSyncPeer::validateReconciled(
                  {signalerAttrList.get(), waiterAttrList.get()},
                  newReconciledList.get(), &isReconciled),
              LwSciError_Success);
    ASSERT_TRUE(isReconciled);

    {
        NegativeTestPrint();

        isReconciled = false;
        ASSERT_EQ(LwSciSyncPeer::validateReconciled(
                      {ilwalidFormatSignalerAttrList.get(), waiterAttrList.get()},
                      newReconciledList.get(), &isReconciled),
                  LwSciError_BadParameter);
        ASSERT_FALSE(isReconciled);
    }
}
// These two attribute keys should have the identical behaviour
INSTANTIATE_TEST_CASE_P(
    LwSciSyncValidateTimestampFormats,
    LwSciSyncValidateTimestampFormats,
    ::testing::Values(
    std::make_tuple(
        LwSciSyncInternalAttrKey_SignalerTimestampInfoMulti),
    std::make_tuple(
        LwSciSyncInternalAttrKey_SignalerTimestampInfo)
    ));

class LwSciSyncValidateTimestampFormatSame : public LwSciSyncBasicTest,
    public ::testing::WithParamInterface<std::tuple<LwSciSyncInternalAttrKey>>
{
};

/**
 * @jama{} Timestamp Formats: Validate different timestamp formats
 *
 * @verifies @jama{}
 *
 * This test checks that validation returns False if there is an input
 * attribute list which requests a different timestamp format than what was
 * reconciled for each of LwSciSyncInternalAttrKey_SignalerTimestampInfo and
 * LwSciSyncInternalAttrKey_SignalerTimestampInfoMulti.
 */
TEST_P(LwSciSyncValidateTimestampFormatSame, Invalid)
{
    auto params = GetParam();
    LwSciSyncInternalAttrKey timestampInfoKey = std::get<0>(params);

    LwSciError error = LwSciError_Success;

    auto signalerAttrList = peer.createAttrList();
    auto waiterAttrList = peer.createAttrList();
    auto differentFormatSignalerAttrList = peer.createAttrList();

    LwSciSyncAttrValTimestampInfo timestampInfo = {
        .format = LwSciSyncTimestampFormat_8Byte,
        .scaling = {
            .scalingFactorNumerator = 0U,
            .scalingFactorDenominator = 1U,
            .sourceOffset = 0U,
        },
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
                      LwSciSyncPeer::attrs.signalerPlatformDefaultAttrs.data(),
                      LwSciSyncPeer::attrs.signalerPlatformDefaultAttrs.size()),
                  LwSciError_Success);

        SET_INTERNAL_ATTR(signalerAttrList.get(), timestampInfoKey,
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
                      LwSciSyncPeer::attrs.waiterPlatformDefaultAttrs.data(),
                      LwSciSyncPeer::attrs.waiterPlatformDefaultAttrs.size()),
                  LwSciError_Success);

        SET_ATTR(waiterAttrList.get(), LwSciSyncAttrKey_WaiterRequireTimestamps,
                 true);
    }

    // Set up Different Format Timestamp Signaler Attribute List
    {
        ASSERT_EQ(LwSciSyncAttrListSetAttrs(
                      differentFormatSignalerAttrList.get(),
                      LwSciSyncPeer::attrs.cpuSignaler.data(),
                      LwSciSyncPeer::attrs.cpuSignaler.size()),
                  LwSciError_Success);

        ASSERT_EQ(LwSciSyncAttrListSetInternalAttrs(
                      differentFormatSignalerAttrList.get(),
                      LwSciSyncPeer::attrs.signalerPlatformDefaultAttrs.data(),
                      LwSciSyncPeer::attrs.signalerPlatformDefaultAttrs.size()),
                  LwSciError_Success);

        LwSciSyncAttrValTimestampInfo ilwalidFormat = {
            .format = LwSciSyncTimestampFormat_16Byte,
            .scaling = {
                .scalingFactorNumerator = 0U,
                .scalingFactorDenominator = 1U,
                .sourceOffset = 0U,
            },
        };
        SET_INTERNAL_ATTR(differentFormatSignalerAttrList.get(),
                          timestampInfoKey,
                          ilwalidFormat);
    }

    // Reconcile Attribute Lists
    auto newReconciledList = LwSciSyncPeer::reconcileLists(
        {signalerAttrList.get(), waiterAttrList.get()}, &error);
    ASSERT_EQ(error, LwSciError_Success);
    ASSERT_NE(newReconciledList.get(), nullptr);

    // Assert on result values
    // First validate that this is a correct reconciled attribute list
    ASSERT_TRUE(LwSciSyncPeer::verifyInternalAttrNew(newReconciledList.get(),
        timestampInfoKey, timestampInfo));
    bool isReconciled = true;
    ASSERT_EQ(LwSciSyncPeer::validateReconciled(
                  {signalerAttrList.get(), waiterAttrList.get()},
                  newReconciledList.get(), &isReconciled),
              LwSciError_Success);
    ASSERT_TRUE(isReconciled);

    {
        NegativeTestPrint();

        isReconciled = false;
        ASSERT_EQ(LwSciSyncPeer::validateReconciled(
                      {differentFormatSignalerAttrList.get(), waiterAttrList.get()},
                      newReconciledList.get(), &isReconciled),
                  LwSciError_BadParameter);
        ASSERT_FALSE(isReconciled);
    }
}
INSTANTIATE_TEST_CASE_P(
    LwSciSyncValidateTimestampFormatSame,
    LwSciSyncValidateTimestampFormatSame,
    ::testing::Values(
    std::make_tuple(
        LwSciSyncInternalAttrKey_SignalerTimestampInfoMulti),
    std::make_tuple(
        LwSciSyncInternalAttrKey_SignalerTimestampInfo)
    ));

/* LwSciSyncInternalAttrKey_EngineArray depends on the IPC table so validation
 * needs to exercise IPC unlike other tests in this file. */
class LwSciSyncValidateEngineArray
    : public LwSciSyncInterProcessTest,
      public ::testing::WithParamInterface<std::tuple<bool, bool>>
{
};
TEST_P(LwSciSyncValidateEngineArray, Values)
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

        // Import dupe Unreconciled Waiter Attribute List
        auto dupeWaiterListDescBuf = peer->recvBuf(&error);
        ASSERT_EQ(error, LwSciError_Success);
        auto dupeWaiterAttrList =
            peer->importUnreconciledList(dupeWaiterListDescBuf, &error);
        ASSERT_EQ(error, LwSciError_Success);

        // Reconcile
        auto reconciledList = LwSciSyncPeer::attrListReconcile(
            {signalerAttrList.get(), waiterAttrList.get(), dupeWaiterAttrList.get()}, &error);
        ASSERT_EQ(error, LwSciError_Success);

        // Test Validation
        // should be successful with any subset of the lists
        std::vector<std::vector<LwSciSyncAttrList>> lists = {
            {signalerAttrList.get()},
            {waiterAttrList.get()},
            {dupeWaiterAttrList.get()},
            {signalerAttrList.get(), waiterAttrList.get()},
            {signalerAttrList.get(), dupeWaiterAttrList.get()},
            {waiterAttrList.get(), dupeWaiterAttrList.get()},
            {signalerAttrList.get(), waiterAttrList.get(), dupeWaiterAttrList.get()}
        };
        bool isReconciledListValid = false;
        for (auto const& list : lists) {
            error = LwSciSyncPeer::validateReconciled(
                list, reconciledList.get(), &isReconciledListValid);
            ASSERT_EQ(error, LwSciError_Success);
            ASSERT_TRUE(isReconciledListValid);
        }

        if (setEngineSignaler == false) {
            // Invalid list
            auto ilwalidSignalerAttrList =
                peer->cloneAttrList(signalerAttrList.get(), &error);
            ASSERT_EQ(error, LwSciError_Success);

            SET_INTERNAL_ATTR(ilwalidSignalerAttrList.get(),
                              LwSciSyncInternalAttrKey_EngineArray, engine);

            NegativeTestPrint();
            error = LwSciSyncPeer::validateReconciled(
                {ilwalidSignalerAttrList.get()}, reconciledList.get(),
                &isReconciledListValid);
            ASSERT_EQ(error, LwSciError_BadParameter);
            ASSERT_FALSE(isReconciledListValid);
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

        // Used to ensure that validation still succeeds when multiple attribute
        // lists are provided by a peer
        auto dupeWaiterAttrList = peer->cloneAttrList(waiterAttrList.get(), &error);
        ASSERT_EQ(error, LwSciError_Success);

        if (setEngineWaiter) {
            SET_INTERNAL_ATTR(waiterAttrList.get(),
                              LwSciSyncInternalAttrKey_EngineArray, engine);
        }

        auto listDescBuf =
            peer->exportUnreconciledList({waiterAttrList.get()}, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_EQ(peer->sendBuf(listDescBuf), LwSciError_Success);

        auto dupeListDescBuf =
            peer->exportUnreconciledList({dupeWaiterAttrList.get()}, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_EQ(peer->sendBuf(dupeListDescBuf), LwSciError_Success);

        auto reconciledListDescBuf = peer->recvBuf(&error);
        ASSERT_EQ(error, LwSciError_Success);
        auto reconciledList = peer->importReconciledList(
            reconciledListDescBuf, {waiterAttrList.get()}, &error);
        ASSERT_EQ(error, LwSciError_Success);

        bool isReconciledListValid = false;
        // should be successful with any subset of the lists
        std::vector<std::vector<LwSciSyncAttrList>> lists = {
            {waiterAttrList.get()},
            {dupeWaiterAttrList.get()},
            {waiterAttrList.get(), dupeWaiterAttrList.get()}
        };
        for (auto const& list : lists) {
            error = LwSciSyncPeer::validateReconciled(
                list, reconciledList.get(), &isReconciledListValid);
            ASSERT_EQ(error, LwSciError_Success);
            ASSERT_TRUE(isReconciledListValid);
        }

        if (setEngineWaiter == false) {
            // Invalid list
            auto ilwalidWaiterAttrList =
                peer->cloneAttrList(waiterAttrList.get(), &error);
            ASSERT_EQ(error, LwSciError_Success);

            SET_INTERNAL_ATTR(ilwalidWaiterAttrList.get(),
                              LwSciSyncInternalAttrKey_EngineArray, engine);

            NegativeTestPrint();
            error = LwSciSyncPeer::validateReconciled(
                {ilwalidWaiterAttrList.get()}, reconciledList.get(),
                &isReconciledListValid);
            ASSERT_EQ(error, LwSciError_BadParameter);
            ASSERT_FALSE(isReconciledListValid);
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
INSTANTIATE_TEST_CASE_P(LwSciSyncValidateEngineArray,
                        LwSciSyncValidateEngineArray,
                        ::testing::Values(
                            // signaler: PCIe, waiter: none
                            std::make_tuple(true, false),
                            // signaler: none, waiter: PCIe
                            std::make_tuple(false, true)));
