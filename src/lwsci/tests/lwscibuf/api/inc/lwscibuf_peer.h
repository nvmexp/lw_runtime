/*
 * lwscibuf_test_peer.h
 *
 * Copyright (c) 2020-2022, LWPU CORPORATION. All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */
#ifndef INCLUDED_LWSCIBUF_PEER_H
#define INCLUDED_LWSCIBUF_PEER_H

#include "lwscibuf_test_attributes.h"
#include "lwscibuf_test_integration.h"

#include <memory>
#include <vector>
#include <array>

#define SET_ATTR(list, key, value)                                             \
    do {                                                                       \
        ASSERT_EQ(LwSciBufPeer::setAttr((list), (key), (value)),               \
                  LwSciError_Success);                                         \
        ASSERT_EQ(LwSciBufPeer::verifyAttr((list), (key), (value)), true);     \
    } while (0)

#define SET_INTERNAL_ATTR(list, key, value)                                    \
    do {                                                                       \
        ASSERT_EQ(LwSciBufPeer::setInternalAttr((list), (key), (value)),       \
                  LwSciError_Success);                                         \
        LwSciBufPeer::verifyInternalAttr((list), (key), (value));              \
    } while (0)

class LwSciBufPeer
{
public:
    virtual void SetUp()
    {
        LwSciBufModule module = nullptr;
        ASSERT_EQ(LwSciBufModuleOpen(&module), LwSciError_Success);
        ASSERT_NE(module, nullptr);
        m_module =
            std::shared_ptr<LwSciBufModuleRec>(module, LwSciBufModuleClose);
    }

    virtual void SetUp(const LwSciBufPeer& otherPeer)
    {
        m_module = otherPeer.m_module;
    }

    virtual void TearDown()
    {
        m_module.reset();
    }

    LwSciBufModule module() const
    {
        return m_module.get();
    }

    std::shared_ptr<LwSciBufAttrListRec> createAttrList(LwSciError* error);

    template <typename T>
    static LwSciError
    setAttr(LwSciBufAttrList list, LwSciBufAttrKey key, T&& value)
    {
        LwSciBufAttrKeyValuePair listAttrs[] = {key, (const void*)&value,
                                                sizeof(T)};
        return LwSciBufAttrListSetAttrs(list, listAttrs, 1);
    }

    template <typename T>
    static LwSciError setInternalAttr(LwSciBufAttrList list,
                                      LwSciBufInternalAttrKey key,
                                      T&& value)
    {
        LwSciBufInternalAttrKeyValuePair listAttrs[] = {
            key, (const void*)&value, sizeof(T)};
        return LwSciBufAttrListSetInternalAttrs(list, listAttrs, 1);
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
    static bool verifyAttr(LwSciBufAttrList attrList,
                           LwSciBufAttrKey key,
                           T&& expected,
                           size_t slotIndex = 0)
    {
        LwSciError error = LwSciError_Success;
        LwSciBufAttrKeyValuePair attr = {key, nullptr, 0};

        error = LwSciBufAttrListSlotGetAttrs(attrList, slotIndex, &attr, 1);
        if (error != LwSciError_Success) {
            TEST_COUT << "verifyAttr: LwSciBufAttrListSlotGetAttrs failed: "
                      << error;
            return false;
        }

        if (attr.len != sizeof(T)) {
            TEST_COUT << "verifyAttr: attr.len != " << sizeof(T);
            return false;
        }

        if (attr.value == nullptr) {
            TEST_COUT << "verifyAttr: attr.value == nullptr";
            return false;
        }

        if (memcmp(&expected, attr.value, sizeof(T)) != 0) {
            TEST_COUT
                << "verifyAttr: memcmp(&expected, attr.value, sizeof(T)) != 0";
            TEST_COUT << "Expected:";
            dumpHex(&expected, sizeof(T));
            TEST_COUT << "attr.value:";
            dumpHex(attr.value, sizeof(T));
            return false;
        }

        return true;
    }

    template <typename T>
    static void verifyInternalAttr(LwSciBufAttrList attrList,
                                   LwSciBufInternalAttrKey key,
                                   T&& expected)
    {
        LwSciBufInternalAttrKeyValuePair attr = {key, nullptr, 0};

        ASSERT_EQ(LwSciError_Success,
                  LwSciBufAttrListGetInternalAttrs(attrList, &attr, 1))
            << INTERNAL_ATTR_NAME(key);

        ASSERT_EQ(attr.len, sizeof(T)) << INTERNAL_ATTR_NAME(key);
        ASSERT_NE(nullptr, attr.value) << INTERNAL_ATTR_NAME(key);
        ASSERT_EQ(memcmp(&expected, attr.value, sizeof(T)), 0)
            << INTERNAL_ATTR_NAME(key);
    }

    static std::shared_ptr<LwSciBufAttrListRec>
    attrListReconcile(const std::vector<LwSciBufAttrList>& lists,
                      LwSciError* error);

    static std::shared_ptr<LwSciBufObjRefRec>
    reconcileAndAllocate(std::vector<LwSciBufAttrList> inputLists,
                         LwSciError* error)
    {
        LwSciBufObj bufObj = nullptr;
        LwSciBufAttrList newConflictList = nullptr;
        *error = LwSciBufAttrListReconcileAndObjAlloc(
            const_cast<LwSciBufAttrList*>(inputLists.data()),
            inputLists.size(), &bufObj, &newConflictList);

        if (*error != LwSciError_Success) {
            if (newConflictList) {
#if (LW_IS_SAFETY == 1)
                TEST_COUT
                    << "Reconciliation failed and conflictList is not NULL";
#endif
                LwSciBufAttrListFree(newConflictList);
            }
            return std::shared_ptr<LwSciBufObjRefRec>(nullptr);
        } else {
            return std::shared_ptr<LwSciBufObjRefRec>(bufObj, LwSciBufObjFree);
        }
    }

    static std::shared_ptr<LwSciBufObjRefRec>
    allocateBufObj(LwSciBufAttrList reconciledList, LwSciError* error)
    {
        LwSciBufObj bufObj = nullptr;
        *error = LwSciBufObjAlloc(reconciledList, &bufObj);
        if (*error != LwSciError_Success) {
            return std::shared_ptr<LwSciBufObjRefRec>(nullptr);
        } else {
            return std::shared_ptr<LwSciBufObjRefRec>(bufObj, LwSciBufObjFree);
        }
    }

    static std::shared_ptr<LwSciBufObjRefRec>
    duplicateBufObj(LwSciBufObj bufObj, LwSciError* error)
    {
        LwSciBufObj bufObjDup = nullptr;
        *error = LwSciBufObjDup(bufObj, &bufObjDup);
        if (*error != LwSciError_Success) {
            return std::shared_ptr<LwSciBufObjRefRec>(nullptr);
        } else {
            return std::shared_ptr<LwSciBufObjRefRec>(bufObjDup,
                                                      LwSciBufObjFree);
        }
    }

    static std::shared_ptr<LwSciBufObjRefRec> duplicateBufObjWithReducedPerm(
        LwSciBufObj bufObj, LwSciBufAttrValAccessPerm perm, LwSciError* error)
    {
        LwSciBufObj bufObjDupReducedPerm = nullptr;
        *error =
            LwSciBufObjDupWithReducePerm(bufObj, perm, &bufObjDupReducedPerm);
        if (*error != LwSciError_Success) {
            return std::shared_ptr<LwSciBufObjRefRec>(nullptr);
        } else {
            return std::shared_ptr<LwSciBufObjRefRec>(bufObjDupReducedPerm,
                                                      LwSciBufObjFree);
        }
    }

    static std::shared_ptr<LwSciBufObjRefRec>
    createFromMemHandle(LwSciBufRmHandle memHandle,
                        LwSciBufAttrList reconciledList, uint64_t offset,
                        uint64_t len, LwSciError* error)
    {
        LwSciBufObj bufObjFromMemHandle = nullptr;
        *error = LwSciBufObjCreateFromMemHandle(
            memHandle, offset, len, reconciledList, &bufObjFromMemHandle);
        if (*error != LwSciError_Success) {
            return std::shared_ptr<LwSciBufObjRefRec>(nullptr);
        } else {
            return std::shared_ptr<LwSciBufObjRefRec>(bufObjFromMemHandle,
                                                      LwSciBufObjFree);
        }
    }

    static void testObject(LwSciBufObj bufferObj, size_t alignment);

    static LwSciError
    validateReconciled(std::vector<LwSciBufAttrList> inputLists,
                       LwSciBufAttrList newReconciledList,
                       bool* isReconciledListValid);

    static std::shared_ptr<LwSciBufAttrListRec>
    attrListAppend(const std::vector<LwSciBufAttrList>& lists,
                   LwSciError* error);

    static std::shared_ptr<LwSciBufAttrListRec>
    attrListClone(LwSciBufAttrList list, LwSciError* error);

    static void checkAttrEqual(LwSciBufAttrList listA,
                               LwSciBufAttrList listB,
                               LwSciBufAttrKey key,
                               size_t slotIndex);

    static void checkInternalAttrEqual(LwSciBufAttrList listA,
                                       LwSciBufAttrList listB,
                                       LwSciBufInternalAttrKey key);

    static void checkAttrListsEqual(LwSciBufAttrList listA,
                                    LwSciBufAttrList listB);

    static std::array<LwSciBufAttrKey, 1> inputAttrKeys;
    static std::array<LwSciBufAttrKey, 20> outputAttrKeys;
    static std::array<LwSciBufAttrKey, 42> attrKeys;
    static std::array<LwSciBufInternalAttrKey, 4> outputInternalAttrKeys;
    static std::array<LwSciBufInternalAttrKey, 2> internalAttrKeys;

    std::shared_ptr<LwSciBufModuleRec> m_module;
};

#endif // INCLUDED_LWSCIBUF_PEER_H
