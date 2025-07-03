/*
 * Copyright (c) 2020-2022 LWPU Corporation.  All Rights Reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property and
 * proprietary rights in and to this software and related documentation.  Any
 * use, reproduction, disclosure or distribution of this software and related
 * documentation without an express license agreement from LWPU Corporation
 * is strictly prohibited.
 */

#ifndef LWSCISYNC_PEER_H
#define LWSCISYNC_PEER_H

#include "lwscisync_test_common.h"
#include <array>
#include <memory>

// This is to avoid rtti compilation error in x86 build
#define GTEST_HAS_RTTI 0
#include <gtest/gtest.h>

#define SET_ATTR(list, key, value)                                             \
    do {                                                                       \
        ASSERT_EQ(LwSciSyncPeer::setAttr((list), (key), (value)),              \
                  LwSciError_Success);                                         \
        ASSERT_EQ(LwSciSyncPeer::verifyAttrNew((list), (key), (value)), true); \
    } while (0)

#define SET_INTERNAL_ATTR(list, key, value)                                    \
    do {                                                                       \
        ASSERT_EQ(LwSciSyncPeer::setInternalAttr((list), (key), (value)),      \
                  LwSciError_Success);                                         \
        LwSciSyncPeer::verifyInternalAttr((list), (key), (value));             \
    } while (0)

static constexpr uint32_t
    TEST_RECONCILIATION_CONFLICTS_ILWALID_PRIMITIVE_COUNT = 0U;

class LwSciSyncTestAttrs
{
public:
    std::array<LwSciSyncAttrKeyValuePair, 2> cpuSignaler;
    std::array<LwSciSyncAttrKeyValuePair, 2> cpuWaiter;
    std::array<LwSciSyncAttrKeyValuePair, 2> cpuWaiterSignaler;
    std::array<LwSciSyncAttrKeyValuePair, 2> signaler;
    std::array<LwSciSyncAttrKeyValuePair, 2> waiter;
    std::array<LwSciSyncAttrKeyValuePair, 2> repeatedPublicKeys;
    std::array<LwSciSyncInternalAttrKeyValuePair, 2> umdInternalAttrs;
    std::array<LwSciSyncInternalAttrKeyValuePair, 2>
        signalerPlatformDefaultAttrs;
    std::array<LwSciSyncInternalAttrKeyValuePair, 2>
        signalerPlatformIlwalidAttrs;
    std::array<LwSciSyncInternalAttrKeyValuePair, 2> signalerSemaphoreAttrs;
    std::array<LwSciSyncInternalAttrKeyValuePair, 2> signalerMultiSemaphoreAttrs;
    std::array<LwSciSyncInternalAttrKeyValuePair, 2> signalerSyncpointAttrs;
    std::array<LwSciSyncInternalAttrKeyValuePair, 2> signalerSemaphore64bPayloadAttrs;
    std::array<LwSciSyncInternalAttrKeyValuePair, 2> signalerMultiDeterministicPrimitiveAttrs;
    std::array<LwSciSyncInternalAttrKeyValuePair, 1> waiterPlatformDefaultAttrs;
    std::array<LwSciSyncInternalAttrKeyValuePair, 1> waiterPlatformIlwalidAttrs;
    std::array<LwSciSyncInternalAttrKeyValuePair, 1> waiterSemaphoreAttrs;
    std::array<LwSciSyncInternalAttrKeyValuePair, 1> waiterSyncpointAttrs;
    std::array<LwSciSyncInternalAttrKeyValuePair, 1> waiterSemaphore64bPayloadAttrs;
    std::array<LwSciSyncInternalAttrKeyValuePair, 1> waiterMultiDeterministicPrimitiveAttrs;
    std::array<LwSciSyncInternalAttrKeyValuePair, 1> waiterMultiPrimitiveAttrs;
    std::array<LwSciSyncInternalAttrKeyValuePair, 2> ilwalidPrimitiveCount;
    std::array<LwSciSyncInternalAttrKeyValuePair, 2> repeatedInternalKeys;
#if (__x86_64__)
    const LwSciSyncInternalAttrValPrimitiveType defaultPlatformPrimitive =
        LwSciSyncInternalAttrValPrimitiveType_SysmemSemaphore;
    const LwSciSyncInternalAttrValPrimitiveType ilwalidPlatformPrimitive =
        LwSciSyncInternalAttrValPrimitiveType_Syncpoint;
#else
    const LwSciSyncInternalAttrValPrimitiveType defaultPlatformPrimitive =
        LwSciSyncInternalAttrValPrimitiveType_Syncpoint;
    const LwSciSyncInternalAttrValPrimitiveType ilwalidPlatformPrimitive =
        LwSciSyncInternalAttrValPrimitiveType_SysmemSemaphore;
#endif

    LwSciSyncTestAttrs()
    {
        static bool cpuTrue = true;
        static bool cpuFalse = false;
        static LwSciSyncAccessPerm permSignalOnly =
            LwSciSyncAccessPerm_SignalOnly;
        static LwSciSyncAccessPerm permWaitOnly = LwSciSyncAccessPerm_WaitOnly;
        static LwSciSyncAccessPerm cpuPermWaitSignal =
            LwSciSyncAccessPerm_WaitSignal;
        static LwSciSyncInternalAttrValPrimitiveType umdPrimitiveInfo[2] = {
            LwSciSyncInternalAttrValPrimitiveType_SysmemSemaphore,
            LwSciSyncInternalAttrValPrimitiveType_Syncpoint};
        static LwSciSyncInternalAttrValPrimitiveType
            multiDeterministicPrimitiveInfo[2] = {
            LwSciSyncInternalAttrValPrimitiveType_SysmemSemaphore,
            LwSciSyncInternalAttrValPrimitiveType_SysmemSemaphorePayload64b
        };
        static uint32_t primitiveCount = 1;
        static uint32_t umdPrimitiveCount = 2;
        static LwSciSyncInternalAttrValPrimitiveType semaphore =
            LwSciSyncInternalAttrValPrimitiveType_SysmemSemaphore;
        static LwSciSyncInternalAttrValPrimitiveType semaphore64bPayload =
            LwSciSyncInternalAttrValPrimitiveType_SysmemSemaphorePayload64b;
        static LwSciSyncInternalAttrValPrimitiveType semaphoreMulti[2] = {
            LwSciSyncInternalAttrValPrimitiveType_SysmemSemaphore,
            LwSciSyncInternalAttrValPrimitiveType_SysmemSemaphorePayload64b,
        };
        static LwSciSyncInternalAttrValPrimitiveType syncpoint =
            LwSciSyncInternalAttrValPrimitiveType_Syncpoint;
        static uint32_t ilwalidPrimitiveCountVal =
            TEST_RECONCILIATION_CONFLICTS_ILWALID_PRIMITIVE_COUNT;

        cpuSignaler[0].attrKey = LwSciSyncAttrKey_NeedCpuAccess;
        cpuSignaler[0].value = (void*)&cpuTrue;
        cpuSignaler[0].len = sizeof(cpuTrue);
        cpuSignaler[1].attrKey = LwSciSyncAttrKey_RequiredPerm;
        cpuSignaler[1].value = (void*)&permSignalOnly;
        cpuSignaler[1].len = sizeof(permSignalOnly);

        cpuWaiter[0].attrKey = LwSciSyncAttrKey_NeedCpuAccess;
        cpuWaiter[0].value = (void*)&cpuTrue;
        cpuWaiter[0].len = sizeof(cpuTrue);
        cpuWaiter[1].attrKey = LwSciSyncAttrKey_RequiredPerm;
        cpuWaiter[1].value = (void*)&permWaitOnly;
        cpuWaiter[1].len = sizeof(permWaitOnly);

        signaler[0].attrKey = LwSciSyncAttrKey_NeedCpuAccess;
        signaler[0].value = (void*)&cpuFalse;
        signaler[0].len = sizeof(cpuFalse);
        signaler[1].attrKey = LwSciSyncAttrKey_RequiredPerm;
        signaler[1].value = (void*)&permSignalOnly;
        signaler[1].len = sizeof(permSignalOnly);

        waiter[0].attrKey = LwSciSyncAttrKey_NeedCpuAccess;
        waiter[0].value = (void*)&cpuFalse;
        waiter[0].len = sizeof(cpuFalse);
        waiter[1].attrKey = LwSciSyncAttrKey_RequiredPerm;
        waiter[1].value = (void*)&permWaitOnly;
        waiter[1].len = sizeof(permWaitOnly);

        cpuWaiterSignaler[0].attrKey = LwSciSyncAttrKey_NeedCpuAccess;
        cpuWaiterSignaler[0].value = (void*)&cpuTrue;
        cpuWaiterSignaler[0].len = sizeof(cpuTrue);
        cpuWaiterSignaler[1].attrKey = LwSciSyncAttrKey_RequiredPerm;
        cpuWaiterSignaler[1].value = (void*)&cpuPermWaitSignal;
        cpuWaiterSignaler[1].len = sizeof(cpuPermWaitSignal);

        repeatedPublicKeys[0].attrKey = LwSciSyncAttrKey_NeedCpuAccess;
        repeatedPublicKeys[0].value = (void*)&cpuFalse;
        repeatedPublicKeys[0].len = sizeof(cpuFalse);
        repeatedPublicKeys[1].attrKey = LwSciSyncAttrKey_NeedCpuAccess;
        repeatedPublicKeys[1].value = (void*)&cpuFalse;
        repeatedPublicKeys[1].len = sizeof(cpuFalse);

        repeatedInternalKeys[0].attrKey =
            LwSciSyncInternalAttrKey_SignalerPrimitiveInfo;
        repeatedInternalKeys[0].value = (const void*)umdPrimitiveInfo;
        repeatedInternalKeys[0].len = sizeof(umdPrimitiveInfo);
        repeatedInternalKeys[1].attrKey =
            LwSciSyncInternalAttrKey_SignalerPrimitiveInfo;
        repeatedInternalKeys[1].value = (const void*)umdPrimitiveInfo;
        repeatedInternalKeys[1].len = sizeof(umdPrimitiveInfo);

        umdInternalAttrs[0].attrKey =
            LwSciSyncInternalAttrKey_SignalerPrimitiveInfo;
        umdInternalAttrs[0].value = (const void*)umdPrimitiveInfo;
        umdInternalAttrs[0].len = sizeof(umdPrimitiveInfo);
        umdInternalAttrs[1].attrKey =
            LwSciSyncInternalAttrKey_SignalerPrimitiveCount;
        umdInternalAttrs[1].value = (void*)&umdPrimitiveCount;
        umdInternalAttrs[1].len = sizeof(umdPrimitiveCount);

        signalerSemaphoreAttrs[0].attrKey =
            LwSciSyncInternalAttrKey_SignalerPrimitiveInfo;
        signalerSemaphoreAttrs[0].value = (const void*)&semaphore;
        signalerSemaphoreAttrs[0].len = sizeof(semaphore);
        signalerSemaphoreAttrs[1].attrKey =
            LwSciSyncInternalAttrKey_SignalerPrimitiveCount;
        signalerSemaphoreAttrs[1].value = (void*)&primitiveCount;
        signalerSemaphoreAttrs[1].len = sizeof(primitiveCount);

        signalerSemaphore64bPayloadAttrs[0].attrKey =
            LwSciSyncInternalAttrKey_SignalerPrimitiveInfo;
        signalerSemaphore64bPayloadAttrs[0].value =
            (const void*)&semaphore64bPayload;
        signalerSemaphore64bPayloadAttrs[0].len =
            sizeof(semaphore64bPayload);
        signalerSemaphore64bPayloadAttrs[1].attrKey =
            LwSciSyncInternalAttrKey_SignalerPrimitiveCount;
        signalerSemaphore64bPayloadAttrs[1].value = (void*)&primitiveCount;
        signalerSemaphore64bPayloadAttrs[1].len = sizeof(primitiveCount);

        signalerMultiDeterministicPrimitiveAttrs[0].attrKey =
            LwSciSyncInternalAttrKey_SignalerPrimitiveInfo;
        signalerMultiDeterministicPrimitiveAttrs[0].value =
            (const void*)multiDeterministicPrimitiveInfo;
        signalerMultiDeterministicPrimitiveAttrs[0].len =
            sizeof(multiDeterministicPrimitiveInfo);
        signalerMultiDeterministicPrimitiveAttrs[1].attrKey =
            LwSciSyncInternalAttrKey_SignalerPrimitiveCount;
        signalerMultiDeterministicPrimitiveAttrs[1].value = (void*)&primitiveCount;
        signalerMultiDeterministicPrimitiveAttrs[1].len = sizeof(primitiveCount);

        signalerMultiSemaphoreAttrs[0].attrKey =
            LwSciSyncInternalAttrKey_SignalerPrimitiveInfo;
        signalerMultiSemaphoreAttrs[0].value = (const void*)semaphoreMulti;
        signalerMultiSemaphoreAttrs[0].len = sizeof(semaphoreMulti);
        signalerMultiSemaphoreAttrs[1].attrKey =
            LwSciSyncInternalAttrKey_SignalerPrimitiveCount;
        signalerMultiSemaphoreAttrs[1].value = (void*)&primitiveCount;
        signalerMultiSemaphoreAttrs[1].len = sizeof(primitiveCount);

        signalerSyncpointAttrs[0].attrKey =
            LwSciSyncInternalAttrKey_SignalerPrimitiveInfo;
        signalerSyncpointAttrs[0].value = (const void*)&syncpoint;
        signalerSyncpointAttrs[0].len = sizeof(syncpoint);
        signalerSyncpointAttrs[1].attrKey =
            LwSciSyncInternalAttrKey_SignalerPrimitiveCount;
        signalerSyncpointAttrs[1].value = (void*)&primitiveCount;
        signalerSyncpointAttrs[1].len = sizeof(primitiveCount);

#if (__x86_64__)
        signalerPlatformDefaultAttrs = signalerSemaphoreAttrs;
#else
        signalerPlatformDefaultAttrs = signalerSyncpointAttrs;
#endif

        signalerPlatformIlwalidAttrs[0].attrKey =
            LwSciSyncInternalAttrKey_SignalerPrimitiveInfo;
        signalerPlatformIlwalidAttrs[0].value =
            (const void*)&ilwalidPlatformPrimitive;
        signalerPlatformIlwalidAttrs[0].len = sizeof(ilwalidPlatformPrimitive);
        signalerPlatformIlwalidAttrs[1].attrKey =
            LwSciSyncInternalAttrKey_SignalerPrimitiveCount;
        signalerPlatformIlwalidAttrs[1].value = (void*)&primitiveCount;
        signalerPlatformIlwalidAttrs[1].len = sizeof(primitiveCount);

        waiterSemaphoreAttrs[0].attrKey =
            LwSciSyncInternalAttrKey_WaiterPrimitiveInfo;
        waiterSemaphoreAttrs[0].value = (const void*)&semaphore;
        waiterSemaphoreAttrs[0].len = sizeof(semaphore);

        waiterSyncpointAttrs[0].attrKey =
            LwSciSyncInternalAttrKey_WaiterPrimitiveInfo;
        waiterSyncpointAttrs[0].value = (const void*)&syncpoint;
        waiterSyncpointAttrs[0].len = sizeof(syncpoint);

        waiterSemaphore64bPayloadAttrs[0].attrKey =
            LwSciSyncInternalAttrKey_WaiterPrimitiveInfo;
        waiterSemaphore64bPayloadAttrs[0].value = (const void*)&semaphore64bPayload;
        waiterSemaphore64bPayloadAttrs[0].len = sizeof(semaphore64bPayload);

        waiterMultiPrimitiveAttrs[0].attrKey =
            LwSciSyncInternalAttrKey_WaiterPrimitiveInfo;
        waiterMultiPrimitiveAttrs[0].value = (const void*)umdPrimitiveInfo;
        waiterMultiPrimitiveAttrs[0].len = sizeof(umdPrimitiveInfo);

        waiterMultiDeterministicPrimitiveAttrs[0].attrKey =
            LwSciSyncInternalAttrKey_WaiterPrimitiveInfo;
        waiterMultiDeterministicPrimitiveAttrs[0].value =
            (const void*)multiDeterministicPrimitiveInfo;
        waiterMultiDeterministicPrimitiveAttrs[0].len =
            sizeof(multiDeterministicPrimitiveInfo);

#if (__x86_64__)
        waiterPlatformDefaultAttrs = waiterSemaphoreAttrs;
#else
        waiterPlatformDefaultAttrs = waiterSyncpointAttrs;
#endif

        waiterPlatformIlwalidAttrs[0].attrKey =
            LwSciSyncInternalAttrKey_WaiterPrimitiveInfo;
        waiterPlatformIlwalidAttrs[0].value =
            (const void*)&ilwalidPlatformPrimitive;
        waiterPlatformIlwalidAttrs[0].len = sizeof(ilwalidPlatformPrimitive);

        ilwalidPrimitiveCount[0].attrKey =
            LwSciSyncInternalAttrKey_SignalerPrimitiveInfo;
        ilwalidPrimitiveCount[0].value = (const void*)umdPrimitiveInfo;
        ilwalidPrimitiveCount[0].len = sizeof(umdPrimitiveInfo);
        ilwalidPrimitiveCount[1].attrKey =
            LwSciSyncInternalAttrKey_SignalerPrimitiveCount;
        ilwalidPrimitiveCount[1].value = (const void*)&ilwalidPrimitiveCountVal;
        ilwalidPrimitiveCount[1].len = sizeof(ilwalidPrimitiveCountVal);
    }
};

class LwSciSyncPeer
{
public:
    LwSciSyncPeer() : modulePtr(nullptr)
    {
    }

    virtual void SetUp()
    {
        LwSciSyncModule module = nullptr;
        ASSERT_EQ(LwSciSyncModuleOpen(&module), LwSciError_Success);
        ASSERT_NE(module, nullptr);
        modulePtr =
            std::shared_ptr<LwSciSyncModuleRec>(module, LwSciSyncModuleClose);
    }

    virtual void SetUp(const LwSciSyncPeer& other)
    {
        modulePtr = other.modulePtr;
    }

    virtual void TearDown()
    {
        if (modulePtr) {
            modulePtr.reset();
        }
    }

    LwSciSyncModule module() const
    {
        return modulePtr.get();
    }

    virtual ~LwSciSyncPeer()
    {
    }

    void createAttrList(LwSciSyncAttrList* newList)
    {
        ASSERT_EQ(LwSciSyncAttrListCreate(module(), newList),
                  LwSciError_Success);
    }

    std::shared_ptr<LwSciSyncAttrListRec> createAttrList()
    {
        LwSciSyncAttrList list = nullptr;
        LwSciSyncAttrListCreate(module(), &list);
        return std::shared_ptr<LwSciSyncAttrListRec>(list,
                                                     LwSciSyncAttrListFree);
    }

    std::shared_ptr<LwSciSyncAttrListRec> createAttrList(LwSciError* error)
    {
        LwSciSyncAttrList attrList = nullptr;
        *error = LwSciSyncAttrListCreate(module(), &attrList);
        return std::shared_ptr<LwSciSyncAttrListRec>(attrList, LwSciSyncAttrListFree);
    }

    std::shared_ptr<LwSciSyncAttrListRec> cloneAttrList(LwSciSyncAttrList list,
                                                        LwSciError* error)
    {
        LwSciSyncAttrList attrList = nullptr;
        *error = LwSciSyncAttrListClone(list, &attrList);
        return std::shared_ptr<LwSciSyncAttrListRec>(attrList,
                                                     LwSciSyncAttrListFree);
    }

    template <typename T>
    static LwSciError setAttr(LwSciSyncAttrList list, LwSciSyncAttrKey key,
                              T&& value)
    {
        LwSciSyncAttrKeyValuePair listAttrs[] = {key, (const void*)&value,
                                                 sizeof(T)};
        return LwSciSyncAttrListSetAttrs(list, listAttrs, 1);
    }

    template <typename T>
    static LwSciError setInternalAttr(LwSciSyncAttrList list,
                                      LwSciSyncInternalAttrKey key, T&& value)
    {
        LwSciSyncInternalAttrKeyValuePair listAttrs[] = {
            key, (const void*)&value, sizeof(T)};
        return LwSciSyncAttrListSetInternalAttrs(list, listAttrs, 1);
    }

    template <typename T>
    static void verifyAttr(LwSciSyncAttrList attrList, LwSciSyncAttrKey key,
                           T&& expected)
    {
        LwSciSyncAttrKeyValuePair attr = {key, nullptr, 0};

        ASSERT_EQ(LwSciError_Success,
                  LwSciSyncAttrListSlotGetAttrs(attrList, 0, &attr, 1));

        ASSERT_EQ(attr.len, sizeof(T)) << ATTR_NAME(key);
        ASSERT_NE(nullptr, attr.value) << ATTR_NAME(key);
        ASSERT_EQ(memcmp(&expected, attr.value, sizeof(T)), 0)
            << ATTR_NAME(key);
    }

    static void dumpHex(const void* data, size_t size)
    {
        char ascii[17];
        size_t i, j;
        ascii[16] = '\0';
        for (i = 0; i < size; ++i) {
            printf("%02X ", ((unsigned char*)data)[i]);
            if (((unsigned char*)data)[i] >= ' ' &&
                ((unsigned char*)data)[i] <= '~') {
                ascii[i % 16] = ((unsigned char*)data)[i];
            } else {
                ascii[i % 16] = '.';
            }
            if ((i + 1) % 8 == 0 || i + 1 == size) {
                printf(" ");
                if ((i + 1) % 16 == 0) {
                    printf("|  %s \n", ascii);
                } else if (i + 1 == size) {
                    ascii[(i + 1) % 16] = '\0';
                    if ((i + 1) % 16 <= 8) {
                        printf(" ");
                    }
                    for (j = (i + 1) % 16; j < 16; ++j) {
                        printf("   ");
                    }
                    printf("|  %s \n", ascii);
                }
            }
        }
    }

    template <typename T>
    static bool verifyAttrNew(LwSciSyncAttrList attrList,
                           LwSciSyncAttrKey key,
                           T&& expected,
                           size_t slotIndex = 0)
    {
        LwSciError error = LwSciError_Success;
        LwSciSyncAttrKeyValuePair attr = {key, nullptr, 0};

        error = LwSciSyncAttrListSlotGetAttrs(attrList, slotIndex, &attr, 1);
        if (error != LwSciError_Success) {
            TEST_COUT << "verifyAttrNew: LwSciSyncAttrListSlotGetAttrs failed: "
                      << error;
            return false;
        }

        if (attr.len != sizeof(T)) {
            TEST_COUT << "verifyAttrNew: attr.len != " << sizeof(T);
            return false;
        }

        if (attr.value == nullptr) {
            TEST_COUT << "verifyAttrNew: attr.value == nullptr";
            return false;
        }

        if (memcmp(&expected, attr.value, sizeof(T)) != 0) {
            TEST_COUT
                << "verifyAttrNew: memcmp(&expected, attr.value, sizeof(T)) != 0";
            TEST_COUT << "Expected:";
            dumpHex(&expected, sizeof(T));
            TEST_COUT << "attr.value:";
            dumpHex(attr.value, sizeof(T));
            return false;
        }

        return true;
    }

    static bool verifyAttrUnset(LwSciSyncAttrList attrList,
                           LwSciSyncAttrKey key,
                           size_t slotIndex = 0)
    {
        LwSciError error = LwSciError_Success;
        LwSciSyncAttrKeyValuePair attr = {key, nullptr, 0};

        error = LwSciSyncAttrListSlotGetAttrs(attrList, slotIndex, &attr, 1);
        if (error != LwSciError_Success) {
            TEST_COUT << "verifyAttrNew: LwSciSyncAttrListSlotGetAttrs failed: "
                      << error;
            return false;
        }

        if (attr.len != 0U) {
            TEST_COUT << "verifyAttrNew: attr.len != 0";
            return false;
        }

        return true;
    }

    static bool verifyInternalAttrUnset(LwSciSyncAttrList attrList,
                           LwSciSyncInternalAttrKey key,
                           size_t slotIndex = 0)
    {
        LwSciError error = LwSciError_Success;
        LwSciSyncInternalAttrKeyValuePair attr = {key, nullptr, 0};

        error = LwSciSyncAttrListGetInternalAttrs(attrList, &attr, 1);
        if (error != LwSciError_Success) {
            TEST_COUT << "verifyInternalAttrUnset: LwSciSyncAttrListGetInternalAttrs failed: "
                      << error;
            return false;
        }

        if (attr.len != 0U) {
            TEST_COUT << "verifyInternalAttrUnset: attr.len != 0";
            return false;
        }

        return true;
    }

    template <typename T>
    static bool verifyInternalAttrNew(LwSciSyncAttrList attrList,
                                   LwSciSyncInternalAttrKey key, T&& expected)
    {
        LwSciSyncInternalAttrKeyValuePair attr = {key, nullptr, 0};

        LwSciError error = LwSciSyncAttrListGetInternalAttrs(attrList, &attr, 1);
        if (error != LwSciError_Success) {
            TEST_COUT << "verifyAttrNew: LwSciSyncAttrListSlotGetAttrs failed: "
                      << error;
            return false;
        }

        if (attr.len != sizeof(T)) {
            TEST_COUT << "verifyAttrNew: attr.len != " << sizeof(T);
            return false;
        }

        if (attr.value == nullptr) {
            TEST_COUT << "verifyAttrNew: attr.value == nullptr";
            return false;
        }

        if (memcmp(&expected, attr.value, sizeof(T)) != 0) {
            TEST_COUT
                << "verifyAttrNew: memcmp(&expected, attr.value, sizeof(T)) != 0";
            TEST_COUT << "Expected:";
            dumpHex(&expected, sizeof(T));
            TEST_COUT << "attr.value:";
            dumpHex(attr.value, sizeof(T));
            return false;
        }
        return true;
    }

    template <typename T>
    static void verifyInternalAttr(LwSciSyncAttrList attrList,
                                   LwSciSyncInternalAttrKey key, T&& expected)
    {
        LwSciSyncInternalAttrKeyValuePair attr = {key, nullptr, 0};

        ASSERT_EQ(LwSciError_Success,
                  LwSciSyncAttrListGetInternalAttrs(attrList, &attr, 1))
            << INTERNAL_ATTR_NAME(key);

        ASSERT_EQ(attr.len, sizeof(T)) << INTERNAL_ATTR_NAME(key);
        ASSERT_NE(nullptr, attr.value) << INTERNAL_ATTR_NAME(key);
        ASSERT_EQ(memcmp(&expected, attr.value, sizeof(T)), 0)
            << INTERNAL_ATTR_NAME(key);
    }

    // Helper to ignore conflict list
    // TODO: remove this API once everyone migrates to attrListReconcile
    static std::shared_ptr<LwSciSyncAttrListRec>
    reconcileLists(std::vector<LwSciSyncAttrList> inputLists, LwSciError* error)
    {
        return LwSciSyncPeer::attrListReconcile(inputLists, error);
    }

    static std::shared_ptr<LwSciSyncAttrListRec>
    attrListReconcile(const std::vector<LwSciSyncAttrList>& lists,
                                    LwSciError* error)
    {
        LwSciSyncAttrList newList = nullptr;
        LwSciSyncAttrList newConflictList = nullptr;
        *error = LwSciSyncAttrListReconcile(lists.data(), lists.size(), &newList,
                                           &newConflictList);
        if (*error != LwSciError_Success) {
            if (newConflictList) {
                LwSciSyncAttrListFree(newConflictList);
            }
            return std::shared_ptr<LwSciSyncAttrListRec>(nullptr);
        } else {
            return std::shared_ptr<LwSciSyncAttrListRec>(newList,
                                                        LwSciSyncAttrListFree);
        }
    }

    static LwSciError
    validateReconciled(std::vector<LwSciSyncAttrList> inputLists,
                       LwSciSyncAttrList newReconciledList,
                       bool* isReconciledListValid)
    {
        return LwSciSyncAttrListValidateReconciled(
            newReconciledList,
            const_cast<LwSciSyncAttrList*>(inputLists.data()),
            inputLists.size(), isReconciledListValid);
    }

    static void checkAttrListIsReconciled(LwSciSyncAttrList list, bool expected)
    {
        bool isReconciled = false;
        ASSERT_EQ(LwSciSyncAttrListIsReconciled(list, &isReconciled),
                  LwSciError_Success);
        ASSERT_EQ(isReconciled, expected);
    }

    static void checkAttrEqual(LwSciSyncAttrList listA, LwSciSyncAttrList listB,
                               LwSciSyncAttrKey key, size_t slotIndex)
    {
        const void* valueA;
        const void* valueB;
        size_t lenA;
        size_t lenB;
        LwSciSyncAttrKeyValuePair pairA[1] = {key, nullptr, 0};
        LwSciSyncAttrKeyValuePair pairB[1] = {key, nullptr, 0};
        ASSERT_EQ(LwSciSyncAttrListSlotGetAttrs(listA, slotIndex, pairA, 1),
                  LwSciError_Success)
            << ATTR_NAME(key) << " slotIndex " << slotIndex;
        ASSERT_EQ(LwSciSyncAttrListSlotGetAttrs(listB, slotIndex, pairB, 1),
                  LwSciError_Success)
            << ATTR_NAME(key) << " slotIndex " << slotIndex;
        ASSERT_EQ(pairA->len, pairB->len)
            << ATTR_NAME(key) << " slotIndex " << slotIndex;
        if (pairA->len > 0) {
            ASSERT_EQ(memcmp(pairA->value, pairB->value, pairA->len), 0)
                << ATTR_NAME(key) << " slotIndex " << slotIndex;
        }
    }

    static void checkInternalAttrEqual(LwSciSyncAttrList listA,
                                       LwSciSyncAttrList listB,
                                       LwSciSyncInternalAttrKey key)
    {
        LwSciSyncInternalAttrKeyValuePair pairA[1] = {key, nullptr, 0};
        LwSciSyncInternalAttrKeyValuePair pairB[1] = {key, nullptr, 0};
        ASSERT_EQ(LwSciSyncAttrListGetInternalAttrs(listA, pairA, 1),
                  LwSciError_Success)
            << INTERNAL_ATTR_NAME(key);
        ASSERT_EQ(LwSciSyncAttrListGetInternalAttrs(listB, pairB, 1),
                  LwSciError_Success)
            << INTERNAL_ATTR_NAME(key);
        ASSERT_EQ(pairA->len, pairB->len) << INTERNAL_ATTR_NAME(key);
        if (pairA->len > 0) {
            ASSERT_EQ(memcmp(pairA->value, pairB->value, pairA->len), 0)
                << INTERNAL_ATTR_NAME(key);
        }
    }

    static void checkAttrListsEqual(LwSciSyncAttrList listA,
                                    LwSciSyncAttrList listB)
    {
        std::array<LwSciSyncAttrKey, 5> attrKeys = {
            LwSciSyncAttrKey_NeedCpuAccess, LwSciSyncAttrKey_RequiredPerm,
            LwSciSyncAttrKey_ActualPerm,
            LwSciSyncAttrKey_WaiterContextInsensitiveFenceExports,
            LwSciSyncAttrKey_WaiterRequireTimestamps};

        std::array<LwSciSyncInternalAttrKey, 5> internalAttrKeys = {
            LwSciSyncInternalAttrKey_SignalerPrimitiveInfo,
            LwSciSyncInternalAttrKey_WaiterPrimitiveInfo,
            LwSciSyncInternalAttrKey_SignalerPrimitiveCount,
            LwSciSyncInternalAttrKey_GpuId,
            LwSciSyncInternalAttrKey_SignalerTimestampInfo};

        ASSERT_EQ(LwSciSyncAttrListGetSlotCount(listA),
                  LwSciSyncAttrListGetSlotCount(listB));

        for (size_t i = 0; i < LwSciSyncAttrListGetSlotCount(listA); i++) {
            for (auto const& attrKey : attrKeys) {
                checkAttrEqual(listA, listB, attrKey, i);
            }
        }

        for (auto const& internalAttrKey : internalAttrKeys) {
            checkInternalAttrEqual(listA, listB, internalAttrKey);
        }
    }

    static void fenceCleanup(LwSciSyncFence* syncFence)
    {
        if (syncFence != nullptr) {
            LwSciSyncFenceClear(syncFence);
            delete syncFence;
        }
    }

    static std::shared_ptr<LwSciSyncFence> initFence()
    {
        auto syncFence =
            std::shared_ptr<LwSciSyncFence>(new LwSciSyncFence, fenceCleanup);
        *syncFence = LwSciSyncFenceInitializer;

        return syncFence;
    }

    static std::shared_ptr<LwSciSyncFence> generateFence(LwSciSyncObj syncObj,
                                                         LwSciError* error)
    {
        auto syncFence =
            std::shared_ptr<LwSciSyncFence>(new LwSciSyncFence, fenceCleanup);
        *syncFence = LwSciSyncFenceInitializer;
        *error = LwSciSyncObjGenerateFence(syncObj, syncFence.get());
        return syncFence;
    }

    static std::shared_ptr<LwSciSyncObjRec>
    allocateSyncObj(LwSciSyncAttrList reconciledList, LwSciError* error)
    {
        LwSciSyncObj syncObj = nullptr;
        *error = LwSciSyncObjAlloc(reconciledList, &syncObj);
        if (*error != LwSciError_Success) {
            return std::shared_ptr<LwSciSyncObjRec>(nullptr);
        } else {
            return std::shared_ptr<LwSciSyncObjRec>(syncObj, LwSciSyncObjFree);
        }
    }

    static std::shared_ptr<LwSciSyncObjRec>
    duplicateSyncObj(LwSciSyncObj syncObj, LwSciError* error)
    {
        LwSciSyncObj syncObjDup = nullptr;
        *error = LwSciSyncObjDup(syncObj, &syncObjDup);
        if (*error != LwSciError_Success) {
            return std::shared_ptr<LwSciSyncObjRec>(nullptr);
        } else {
            return std::shared_ptr<LwSciSyncObjRec>(syncObjDup,
                                                    LwSciSyncObjFree);
        }
    }

    std::shared_ptr<LwSciSyncCpuWaitContextRec>
    allocateCpuWaitContext(LwSciError* error)
    {
        LwSciSyncCpuWaitContext waitContext = nullptr;
        *error = LwSciSyncCpuWaitContextAlloc(module(), &waitContext);
        if (*error != LwSciError_Success) {
            return std::shared_ptr<LwSciSyncCpuWaitContextRec>(nullptr);
        } else {
            return std::shared_ptr<LwSciSyncCpuWaitContextRec>(
                waitContext, LwSciSyncCpuWaitContextFree);
        }
    }

    static std::shared_ptr<LwSciSyncObjRec>
    reconcileAndAllocate(std::vector<LwSciSyncAttrList> inputLists,
                         LwSciError* error)
    {
        LwSciSyncObj syncObj = nullptr;
        LwSciSyncAttrList newConflictList = nullptr;
        *error = LwSciSyncAttrListReconcileAndObjAlloc(
            const_cast<LwSciSyncAttrList*>(inputLists.data()),
            inputLists.size(), &syncObj, &newConflictList);

        if (*error != LwSciError_Success) {
            if (newConflictList) {
#if (LW_IS_SAFETY == 1)
                TEST_COUT
                    << "Reconciliation failed and conflictList is not NULL";
#endif
                LwSciSyncAttrListFree(newConflictList);
            }
            return std::shared_ptr<LwSciSyncObjRec>(nullptr);
        } else {
            return std::shared_ptr<LwSciSyncObjRec>(syncObj, LwSciSyncObjFree);
        }
    }

    std::shared_ptr<LwSciSyncModuleRec> modulePtr;
    static LwSciSyncTestAttrs attrs;
};

#endif // LWSCISYNC_PEER_H
