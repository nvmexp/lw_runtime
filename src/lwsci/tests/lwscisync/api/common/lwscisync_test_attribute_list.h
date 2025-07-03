/*
 * Copyright (c) 2020-2022 LWPU Corporation.  All Rights Reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property and
 * proprietary rights in and to this software and related documentation.  Any
 * use, reproduction, disclosure or distribution of this software and related
 * documentation without an express license agreement from LWPU Corporation
 * is strictly prohibited.
 */
#ifndef LWSCISYNC_TEST_ATTRIBUTE_LIST_H
#define LWSCISYNC_TEST_ATTRIBUTE_LIST_H

#include "lwscisync_ipc_peer_old.h"
#include "lwscisync_test_common.h"

#define VERIFY_SET_ATTR(key, value)                                            \
    do {                                                                       \
        auto list = peer.createAttrList();                                     \
        ASSERT_EQ(LwSciSyncPeer::setAttr(list.get(), (key), (value)),          \
                  LwSciError_Success);                                         \
        LwSciSyncPeer::verifyAttr(list.get(), (key), (value));                 \
        NegativeTestPrint();                                                   \
        ASSERT_EQ(LwSciSyncPeer::setAttr(list.get(), (key), (value)),          \
                  LwSciError_BadParameter);                                    \
    } while (0)

#define VERIFY_SET_INTERNAL_ATTR(key, value)                                   \
    do {                                                                       \
        auto list = peer.createAttrList();                                     \
        ASSERT_EQ(LwSciSyncPeer::setInternalAttr(list.get(), (key), (value)),  \
                  LwSciError_Success);                                         \
        LwSciSyncPeer::verifyInternalAttr(list.get(), (key), (value));         \
        NegativeTestPrint();                                                   \
        ASSERT_EQ(LwSciSyncPeer::setInternalAttr(list.get(), (key), (value)),  \
                  LwSciError_BadParameter);                                    \
    } while (0)

#define VERIFY_ATTR_READONLY(list, key, value)                                 \
    do {                                                                       \
        LwSciSyncPeer::verifyAttr(list, key, value);                           \
        NegativeTestPrint();                                                   \
        ASSERT_EQ(LwSciSyncPeer::setAttr(list, (key), (value)),                \
                  LwSciError_BadParameter);                                    \
    } while (0)

#define VERIFY_INTERNAL_ATTR_READONLY(list, key, value)                        \
    do {                                                                       \
        LwSciSyncPeer::verifyInternalAttr(list, key, value);                   \
        NegativeTestPrint();                                                   \
        ASSERT_EQ(LwSciSyncPeer::setInternalAttr(list, (key), (value)),        \
                  LwSciError_BadParameter);                                    \
    } while (0)

#define VERIFY_SET_ATTR_OUTPUT_ONLY(key, value)                                \
    do {                                                                       \
        auto list = peer.createAttrList();                                     \
        NegativeTestPrint();                                                   \
        ASSERT_EQ(LwSciSyncPeer::setAttr(list.get(), (key), (value)),          \
                  LwSciError_BadParameter);                                    \
    } while (0)

#define VERIFY_ATTR_LIST_IS_READONLY(list)                                     \
    do {                                                                       \
        NegativeTestPrint();                                                   \
        ASSERT_EQ(LwSciSyncPeer::setAttr(                                      \
                      (list), LwSciSyncAttrKey_NeedCpuAccess, true),           \
                  LwSciError_BadParameter);                                    \
        NegativeTestPrint();                                                   \
        ASSERT_EQ(LwSciSyncPeer::setAttr((list),                               \
                                         LwSciSyncAttrKey_RequiredPerm,        \
                                         LwSciSyncAccessPerm_SignalOnly),      \
                  LwSciError_BadParameter);                                    \
        NegativeTestPrint();                                                   \
        ASSERT_EQ(LwSciSyncPeer::setAttr(                                      \
                      (list),                                                  \
                      LwSciSyncAttrKey_WaiterContextInsensitiveFenceExports,   \
                      true),                                                   \
                  LwSciError_BadParameter);                                    \
        NegativeTestPrint();                                                   \
        ASSERT_EQ(LwSciSyncPeer::setInternalAttr(                              \
                      (list), LwSciSyncInternalAttrKey_SignalerPrimitiveInfo,  \
                      LwSciSyncInternalAttrValPrimitiveType_SysmemSemaphore),  \
                  LwSciError_BadParameter);                                    \
        NegativeTestPrint();                                                   \
        ASSERT_EQ(LwSciSyncPeer::setInternalAttr(                              \
                      (list), LwSciSyncInternalAttrKey_WaiterPrimitiveInfo,    \
                      LwSciSyncInternalAttrValPrimitiveType_SysmemSemaphore),  \
                  LwSciError_BadParameter);                                    \
        NegativeTestPrint();                                                   \
        ASSERT_EQ(                                                             \
            LwSciSyncPeer::setInternalAttr(                                    \
                (list), LwSciSyncInternalAttrKey_SignalerPrimitiveCount, 1),   \
            LwSciError_BadParameter);                                          \
    } while (0)

#define VERIFY_ATTR_LIST_IS_WRITABLE(list)                                     \
    do {                                                                       \
        SET_ATTR((list), LwSciSyncAttrKey_NeedCpuAccess, true);                \
        SET_ATTR((list), LwSciSyncAttrKey_RequiredPerm,                        \
                 LwSciSyncAccessPerm_WaitSignal);                              \
        SET_ATTR((list),                                                       \
                 LwSciSyncAttrKey_WaiterContextInsensitiveFenceExports,        \
                 false);                                                       \
        SET_INTERNAL_ATTR((list),                                              \
                          LwSciSyncInternalAttrKey_WaiterPrimitiveInfo,        \
                          LwSciSyncInternalAttrValPrimitiveType_Syncpoint);    \
        SET_INTERNAL_ATTR((list),                                              \
                          LwSciSyncInternalAttrKey_SignalerPrimitiveInfo,      \
                          LwSciSyncInternalAttrValPrimitiveType_Syncpoint);    \
        SET_INTERNAL_ATTR((list),                                              \
                          LwSciSyncInternalAttrKey_SignalerPrimitiveCount, 1); \
    } while (0)

#if defined(__x86_64__)
const LwSciSyncInternalAttrValPrimitiveType
ALL_CPU_PRIMITIVES[] = {
    LwSciSyncInternalAttrValPrimitiveType_SysmemSemaphore,
    LwSciSyncInternalAttrValPrimitiveType_SysmemSemaphorePayload64b,
};
#else
const LwSciSyncInternalAttrValPrimitiveType
ALL_CPU_PRIMITIVES[] = {
    LwSciSyncInternalAttrValPrimitiveType_Syncpoint,
    LwSciSyncInternalAttrValPrimitiveType_SysmemSemaphore,
    LwSciSyncInternalAttrValPrimitiveType_SysmemSemaphorePayload64b,
};
#endif

#if defined(__x86_64__)
#define DEFAULT_RECONCILED_PRIMITIVE LwSciSyncInternalAttrValPrimitiveType_SysmemSemaphore
#else
#define DEFAULT_RECONCILED_PRIMITIVE LwSciSyncInternalAttrValPrimitiveType_Syncpoint
#endif

/**
 * @brief Verify reconciled attributes for default CPU Signaler / CPU Waiter
 * scenario
 */
#define VERIFY_ATTR_LIST_IS_RECONCILED(list)                                   \
    do {                                                                       \
        LwSciSyncPeer::checkAttrListIsReconciled((list), true);                \
                                                                               \
        VERIFY_ATTR_READONLY((list), LwSciSyncAttrKey_NeedCpuAccess, true);    \
        VERIFY_ATTR_READONLY((list), LwSciSyncAttrKey_ActualPerm,              \
                             LwSciSyncAccessPerm_WaitSignal);                  \
        VERIFY_ATTR_READONLY(                                                  \
            (list), LwSciSyncAttrKey_WaiterContextInsensitiveFenceExports,     \
            false);                                                            \
        VERIFY_INTERNAL_ATTR_READONLY(                                         \
            (list), LwSciSyncInternalAttrKey_WaiterPrimitiveInfo,              \
            DEFAULT_RECONCILED_PRIMITIVE);                                                \
        VERIFY_INTERNAL_ATTR_READONLY(                                         \
            (list), LwSciSyncInternalAttrKey_SignalerPrimitiveInfo,            \
            DEFAULT_RECONCILED_PRIMITIVE);                                                \
        VERIFY_INTERNAL_ATTR_READONLY(                                         \
            (list), LwSciSyncInternalAttrKey_SignalerPrimitiveCount, 1);       \
    } while (0)

template <int64_t JamaID>
class LwSciSyncAttrListTest : public LwSciSyncBaseTest<JamaID>
{
public:
    LwSciSyncPeer peer;
    LwSciSyncPeer otherPeer;

    void SetUp() override
    {
        peer.SetUp();
        otherPeer.SetUp();
        LwSciSyncBaseTest<JamaID>::SetUp();
    }

    void TearDown() override
    {
        peer.TearDown();
        otherPeer.TearDown();
        LwSciSyncBaseTest<JamaID>::TearDown();
    }
};

/* Declare new tests with this macro to make sure each test case has Jama ID */
#define LWSCISYNC_ATTRIBUTE_LIST_TEST(testSuite, JamaID)                       \
    class testSuite : public LwSciSyncAttrListTest<JamaID>                     \
    {                                                                          \
    };

/* Declare additional test case for a test */
#define LWSCISYNC_ATTRIBUTE_LIST_TEST_CASE(testSuite, testName)                \
    TEST_F(testSuite, testName)

#endif //LWSCISYNC_TEST_ATTRIBUTE_LIST_H
