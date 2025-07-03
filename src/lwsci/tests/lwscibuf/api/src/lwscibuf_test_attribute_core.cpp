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
#include <array>
#include <limits>
#include <type_traits>

class LwSciBufTestAttrlistAppend : public LwSciBufBasicTest
{
public:
    void SetUp() override
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

    void TearDown() override
    {
        LwSciBufBasicTest::TearDown();
        listA.reset();
        listB.reset();
    }

    std::shared_ptr<LwSciBufAttrListRec> listA;
    std::shared_ptr<LwSciBufAttrListRec> listB;
};

TEST_F(LwSciBufTestAttrlistAppend, PositiveDifferentAlignment)
{
    LwSciError error = LwSciError_Success;
    LwSciBufType bufType = LwSciBufType_RawBuffer;
    uint64_t size = 128 * 1024;
    uint64_t align1 = 4 * 1024;
    uint64_t align2 = 8 * 1024;

    SET_ATTR(listA.get(), LwSciBufGeneralAttrKey_Types, bufType);
    SET_ATTR(listA.get(), LwSciBufRawBufferAttrKey_Size, size);
    SET_ATTR(listA.get(), LwSciBufRawBufferAttrKey_Align, align1);

    SET_ATTR(listB.get(), LwSciBufGeneralAttrKey_Types, bufType);
    SET_ATTR(listB.get(), LwSciBufRawBufferAttrKey_Size, size);
    SET_ATTR(listB.get(), LwSciBufRawBufferAttrKey_Align, align2);

    auto appendedList =
        LwSciBufPeer::attrListAppend({listA.get(), listB.get()}, &error);
    ASSERT_EQ(error, LwSciError_Success);

    ASSERT_EQ(LwSciBufAttrListGetSlotCount(appendedList.get()), 2);

    // Verify attributes in slot 0 are equal to listA
    LwSciBufPeer::verifyAttr(appendedList.get(), LwSciBufGeneralAttrKey_Types,
                             bufType, 0);
    LwSciBufPeer::verifyAttr(appendedList.get(), LwSciBufRawBufferAttrKey_Size,
                             size, 0);
    LwSciBufPeer::verifyAttr(appendedList.get(), LwSciBufRawBufferAttrKey_Align,
                             align1, 0);
    // Verify attributes in slot 1 are equal to listB
    LwSciBufPeer::verifyAttr(appendedList.get(), LwSciBufGeneralAttrKey_Types,
                             bufType, 1);
    LwSciBufPeer::verifyAttr(appendedList.get(), LwSciBufRawBufferAttrKey_Size,
                             size, 1);
    LwSciBufPeer::verifyAttr(appendedList.get(), LwSciBufRawBufferAttrKey_Align,
                             align2, 1);

    auto reconciledA =
        LwSciBufPeer::attrListReconcile({listA.get(), listB.get()}, &error);
    ASSERT_EQ(error, LwSciError_Success);

    auto reconciledB =
        LwSciBufPeer::attrListReconcile({appendedList.get()}, &error);
    ASSERT_EQ(error, LwSciError_Success);

    LwSciBufPeer::checkAttrListsEqual(reconciledA.get(), reconciledB.get());
}

TEST_F(LwSciBufTestAttrlistAppend, NegativeDifferentSize)
{
    LwSciError error = LwSciError_Success;
    LwSciBufType bufType = LwSciBufType_RawBuffer;
    uint64_t size1 = 128 * 1024;
    uint64_t size2 = 256 * 1024;
    uint64_t align = 4 * 1024;

    SET_ATTR(listA.get(), LwSciBufGeneralAttrKey_Types, bufType);
    SET_ATTR(listA.get(), LwSciBufRawBufferAttrKey_Size, size1);
    SET_ATTR(listA.get(), LwSciBufRawBufferAttrKey_Align, align);

    SET_ATTR(listB.get(), LwSciBufGeneralAttrKey_Types, bufType);
    SET_ATTR(listB.get(), LwSciBufRawBufferAttrKey_Size, size2);
    SET_ATTR(listB.get(), LwSciBufRawBufferAttrKey_Align, align);

    auto appendedList =
        LwSciBufPeer::attrListAppend({listA.get(), listB.get()}, &error);
    ASSERT_EQ(error, LwSciError_Success);

    ASSERT_EQ(LwSciBufAttrListGetSlotCount(appendedList.get()), 2);

    // Verify attributes in slot 0 are equal to listA
    LwSciBufPeer::verifyAttr(appendedList.get(), LwSciBufGeneralAttrKey_Types,
                             bufType, 0);
    LwSciBufPeer::verifyAttr(appendedList.get(), LwSciBufRawBufferAttrKey_Size,
                             size1, 0);
    LwSciBufPeer::verifyAttr(appendedList.get(), LwSciBufRawBufferAttrKey_Align,
                             align, 0);
    // Verify attributes in slot 1 are equal to listB
    LwSciBufPeer::verifyAttr(appendedList.get(), LwSciBufGeneralAttrKey_Types,
                             bufType, 1);
    LwSciBufPeer::verifyAttr(appendedList.get(), LwSciBufRawBufferAttrKey_Size,
                             size2, 1);
    LwSciBufPeer::verifyAttr(appendedList.get(), LwSciBufRawBufferAttrKey_Align,
                             align, 1);

    {
        NEGATIVE_TEST();
        auto reconciledA =
            LwSciBufPeer::attrListReconcile({listA.get(), listB.get()}, &error);
        ASSERT_EQ(error, LwSciError_ReconciliationFailed);

        auto reconciledB =
            LwSciBufPeer::attrListReconcile({appendedList.get()}, &error);
        ASSERT_EQ(error, LwSciError_ReconciliationFailed);
    }
}

TEST_F(LwSciBufTestAttrlistAppend, NegativeReconciled)
{
    LwSciError error = LwSciError_Success;
    LwSciBufType bufType = LwSciBufType_RawBuffer;
    uint64_t size = 128 * 1024;
    uint64_t align1 = 4 * 1024;
    uint64_t align2 = 8 * 1024;

    SET_ATTR(listA.get(), LwSciBufGeneralAttrKey_Types, bufType);
    SET_ATTR(listA.get(), LwSciBufRawBufferAttrKey_Size, size);
    SET_ATTR(listA.get(), LwSciBufRawBufferAttrKey_Align, align1);

    SET_ATTR(listB.get(), LwSciBufGeneralAttrKey_Types, bufType);
    SET_ATTR(listB.get(), LwSciBufRawBufferAttrKey_Size, size);
    SET_ATTR(listB.get(), LwSciBufRawBufferAttrKey_Align, align2);

    auto appendedList =
        LwSciBufPeer::attrListAppend({listA.get(), listB.get()}, &error);
    ASSERT_EQ(error, LwSciError_Success);

    ASSERT_EQ(LwSciBufAttrListGetSlotCount(appendedList.get()), 2);

    // Verify attributes in slot 0 are equal to listA
    LwSciBufPeer::verifyAttr(appendedList.get(), LwSciBufGeneralAttrKey_Types,
                             bufType, 0);
    LwSciBufPeer::verifyAttr(appendedList.get(), LwSciBufRawBufferAttrKey_Size,
                             size, 0);
    LwSciBufPeer::verifyAttr(appendedList.get(), LwSciBufRawBufferAttrKey_Align,
                             align1, 0);
    // Verify attributes in slot 1 are equal to listB
    LwSciBufPeer::verifyAttr(appendedList.get(), LwSciBufGeneralAttrKey_Types,
                             bufType, 1);
    LwSciBufPeer::verifyAttr(appendedList.get(), LwSciBufRawBufferAttrKey_Size,
                             size, 1);
    LwSciBufPeer::verifyAttr(appendedList.get(), LwSciBufRawBufferAttrKey_Align,
                             align2, 1);

    auto reconciledA =
        LwSciBufPeer::attrListReconcile({listA.get(), listB.get()}, &error);
    ASSERT_EQ(error, LwSciError_Success);

    auto reconciledB =
        LwSciBufPeer::attrListReconcile({appendedList.get()}, &error);
    ASSERT_EQ(error, LwSciError_Success);

    {
        NEGATIVE_TEST();
        LwSciBufPeer::attrListAppend({reconciledA.get(), reconciledB.get()},
                                     &error);
        ASSERT_EQ(error, LwSciError_BadParameter);

        LwSciBufPeer::attrListAppend({listA.get(), reconciledB.get()}, &error);
        ASSERT_EQ(error, LwSciError_BadParameter);
    }
}

TEST_F(LwSciBufTestAttrlistAppend, NegativeWriteAppendedList)
{
    LwSciError error = LwSciError_Success;
    LwSciBufType bufType = LwSciBufType_Image;
    uint32_t planeCount = 1U;
    LwSciBufAttrValColorFmt colorFmt = LwSciColor_A8B8G8R8;
    uint32_t width = 1080U;
    uint32_t height = 1920U;

    SET_ATTR(listA.get(), LwSciBufGeneralAttrKey_Types, bufType);
    SET_ATTR(listA.get(), LwSciBufImageAttrKey_PlaneCount, planeCount);
    SET_ATTR(listA.get(), LwSciBufImageAttrKey_PlaneColorFormat, colorFmt);
    SET_ATTR(listA.get(), LwSciBufImageAttrKey_PlaneWidth, width);
    SET_ATTR(listA.get(), LwSciBufImageAttrKey_PlaneHeight, height);

    SET_ATTR(listB.get(), LwSciBufGeneralAttrKey_Types, bufType);
    SET_ATTR(listB.get(), LwSciBufImageAttrKey_PlaneCount, planeCount);
    SET_ATTR(listB.get(), LwSciBufImageAttrKey_PlaneColorFormat, colorFmt);
    SET_ATTR(listB.get(), LwSciBufImageAttrKey_PlaneWidth, width);
    SET_ATTR(listB.get(), LwSciBufImageAttrKey_PlaneHeight, height);

    auto appendedList =
        LwSciBufPeer::attrListAppend({listA.get(), listB.get()}, &error);
    ASSERT_EQ(error, LwSciError_Success);

    {
        LwSciBufAttrValImageLayoutType layout = LwSciBufImage_BlockLinearType;
        LwSciBufMemDomain memDomain[] = {LwSciBufMemDomain_Sysmem};

        NEGATIVE_TEST();

        ASSERT_EQ(LwSciBufPeer::setAttr(appendedList.get(),
            LwSciBufImageAttrKey_Layout, layout), LwSciError_BadParameter);

        ASSERT_EQ(LwSciBufPeer::setInternalAttr(appendedList.get(),
            LwSciBufInternalGeneralAttrKey_MemDomainArray, memDomain),
            LwSciError_BadParameter);
    }
}

class LwSciBufTestAttrlistClone : public LwSciBufBasicTest
{
};

TEST_F(LwSciBufTestAttrlistClone, Positive)
{
    LwSciError error = LwSciError_Success;
    auto listA = peer.createAttrList(&error);
    ASSERT_EQ(error, LwSciError_Success);
    auto listB = peer.createAttrList(&error);
    ASSERT_EQ(error, LwSciError_Success);

    LwSciBufType bufType = LwSciBufType_RawBuffer;
    uint64_t size = 128 * 1024;
    uint64_t align1 = 4 * 1024;
    uint64_t align2 = 8 * 1024;

    SET_ATTR(listA.get(), LwSciBufGeneralAttrKey_Types, bufType);
    SET_ATTR(listA.get(), LwSciBufRawBufferAttrKey_Size, size);
    SET_ATTR(listA.get(), LwSciBufRawBufferAttrKey_Align, align1);

    SET_ATTR(listB.get(), LwSciBufGeneralAttrKey_Types, bufType);
    SET_ATTR(listB.get(), LwSciBufRawBufferAttrKey_Size, size);
    SET_ATTR(listB.get(), LwSciBufRawBufferAttrKey_Align, align2);

    auto clonedListA = LwSciBufPeer::attrListClone(listA.get(), &error);
    ASSERT_EQ(error, LwSciError_Success);
    LwSciBufPeer::checkAttrListsEqual(listA.get(), clonedListA.get());

    // Verify cloned list is writable
    SET_ATTR(clonedListA.get(), LwSciBufGeneralAttrKey_Types, bufType);
    SET_ATTR(clonedListA.get(), LwSciBufRawBufferAttrKey_Size, size);
    SET_ATTR(clonedListA.get(), LwSciBufRawBufferAttrKey_Align, align1);

    auto clonedListB = LwSciBufPeer::attrListClone(listB.get(), &error);
    ASSERT_EQ(error, LwSciError_Success);
    LwSciBufPeer::checkAttrListsEqual(listB.get(), clonedListB.get());

    // Verify cloned list is writable
    SET_ATTR(clonedListB.get(), LwSciBufGeneralAttrKey_Types, bufType);
    SET_ATTR(clonedListB.get(), LwSciBufRawBufferAttrKey_Size, size);
    SET_ATTR(clonedListB.get(), LwSciBufRawBufferAttrKey_Align, align2);

    auto reconciledListA =
        LwSciBufPeer::attrListReconcile({listA.get(), listB.get()}, &error);
    ASSERT_EQ(error, LwSciError_Success);

    auto reconciledListB = LwSciBufPeer::attrListReconcile(
        {clonedListA.get(), clonedListB.get()}, &error);
    ASSERT_EQ(error, LwSciError_Success);

    LwSciBufPeer::checkAttrListsEqual(reconciledListA.get(),
                                      reconciledListB.get());

    auto clonedReconciled =
        LwSciBufPeer::attrListClone(reconciledListB.get(), &error);
    ASSERT_EQ(error, LwSciError_Success);

    LwSciBufPeer::checkAttrListsEqual(clonedReconciled.get(),
                                      reconciledListB.get());
}

class LwSciBufTestAttrListGetAttrs : public LwSciBufBasicTest
{
};

TEST_F(LwSciBufTestAttrListGetAttrs, Positive)
{
    LwSciError error = LwSciError_Success;
    auto list = peer.createAttrList(&error);
    ASSERT_EQ(error, LwSciError_Success);

    LwSciBufType bufType = LwSciBufType_RawBuffer;
    uint64_t size = 128 * 1024;
    uint64_t align = 4 * 1024;

    LwSciBufAttrKeyValuePair setAttrs[] = {
        {.key = LwSciBufGeneralAttrKey_Types,
         .value = &bufType,
         .len = sizeof(bufType)},
        {.key = LwSciBufRawBufferAttrKey_Size,
         .value = &size,
         .len = sizeof(size)},
        {.key = LwSciBufRawBufferAttrKey_Align,
         .value = &align,
         .len = sizeof(align)}};
    size_t length = sizeof(setAttrs) / sizeof(LwSciBufAttrKeyValuePair);

    error = LwSciBufAttrListSetAttrs(list.get(), setAttrs, length);
    ASSERT_EQ(error, LwSciError_Success);

    LwSciBufAttrKeyValuePair getAttrs[] = {
        {.key = LwSciBufGeneralAttrKey_Types, .value = nullptr, .len = 0},
        {.key = LwSciBufRawBufferAttrKey_Size, .value = nullptr, .len = 0},
        {.key = LwSciBufRawBufferAttrKey_Align, .value = nullptr, .len = 0}};

    error = LwSciBufAttrListGetAttrs(list.get(), getAttrs, length);
    ASSERT_EQ(error, LwSciError_Success);

    for (int i = 0; i < length; i++) {
        ASSERT_EQ(setAttrs[i].len, getAttrs[i].len);
        ASSERT_EQ(memcmp(setAttrs[i].value, getAttrs[i].value, setAttrs[i].len),
                  0);
    }
}

#if (LW_IS_SAFETY == 0)
class LwSciBufTestAttrListDebugDump : public LwSciBufBasicTest
{
};

// TODO: Disabled due to a bug in LwSciBufAttrListDebugDump(), zero IPC endpoint
// is passed to LwSciBufAttrListIpcExportUnreconciled() which is forbidden
TEST_F(LwSciBufTestAttrListDebugDump, DISABLED_Positive)
{
    LwSciError error = LwSciError_Success;
    auto list = peer.createAttrList(&error);
    ASSERT_EQ(error, LwSciError_Success);

    LwSciBufType bufType = LwSciBufType_RawBuffer;
    uint64_t size = 128 * 1024;
    uint64_t align = 4 * 1024;

    LwSciBufAttrKeyValuePair setAttrs[] = {
        {.key = LwSciBufGeneralAttrKey_Types,
         .value = &bufType,
         .len = sizeof(bufType)},
        {.key = LwSciBufRawBufferAttrKey_Size,
         .value = &size,
         .len = sizeof(size)},
        {.key = LwSciBufRawBufferAttrKey_Align,
         .value = &align,
         .len = sizeof(align)}};
    size_t length = sizeof(setAttrs) / sizeof(LwSciBufAttrKeyValuePair);

    ASSERT_EQ(LwSciBufAttrListSetAttrs(list.get(), setAttrs, length),
              LwSciError_Success);

    void* buf = nullptr;
    size_t bufSize = 0;
    ASSERT_EQ(LwSciBufAttrListDebugDump(list.get(), &buf, &bufSize),
              LwSciError_Success);
    ASSERT_NE(buf, nullptr);
    ASSERT_NE(bufSize, 0);
    LwSciBufPeer::dumpHex(buf, bufSize);
}
#endif

class LwSciBufTestAttributeCore : public LwSciBufBasicTest
{
public:
    void SetUp() override
    {
        LwSciError error = LwSciError_Success;

        LwSciBufBasicTest::SetUp();

        list = peer.createAttrList(&error);
        ASSERT_EQ(error, LwSciError_Success);
    }

    std::shared_ptr<LwSciBufAttrListRec> list;
};

TEST_F(LwSciBufTestAttributeCore, RevertSetAttrs)
{
    LwSciError error = LwSciError_Success;

    uint64_t sizes[] = {2, 8, 64, 32, 32};
    uint32_t alignment[] = {1, 1, 32, 1, 1};
    uint32_t dimcount = 5;
    uint32_t dataType = LwSciDataType_Int16;
    LwSciBufType type = LwSciBufType_Tensor;

    // Set some general attributes and buffer type
    std::vector<LwSciBufAttrKeyValuePair> attrs = {
        {LwSciBufTensorAttrKey_DataType, &dataType, sizeof(dataType)},
        {LwSciBufTensorAttrKey_NumDims, &dimcount, sizeof(dimcount)},
        {LwSciBufTensorAttrKey_SizePerDim, &sizes, sizeof(sizes)},
        {LwSciBufGeneralAttrKey_Types, &type, sizeof(type)},
    };
    ASSERT_EQ(LwSciBufAttrListSetAttrs(list.get(), attrs.data(), attrs.size()),
              LwSciError_Success);
    // Verify the attributes
    ASSERT_EQ(LwSciBufPeer::verifyAttr(
                  list.get(), LwSciBufTensorAttrKey_DataType, dataType),
              true);
    ASSERT_EQ(LwSciBufPeer::verifyAttr(list.get(),
                                       LwSciBufTensorAttrKey_NumDims, dimcount),
              true);
    ASSERT_EQ(LwSciBufPeer::verifyAttr(list.get(),
                                       LwSciBufTensorAttrKey_SizePerDim, sizes),
              true);
    ASSERT_EQ(LwSciBufPeer::verifyAttr(list.get(), LwSciBufGeneralAttrKey_Types,
                                       type),
              true);

    // Make a clone for the reference
    auto listClone = LwSciBufPeer::attrListClone(list.get(), &error);

    // Perform single invalid setAttr operation
    {
        NEGATIVE_TEST();
        ASSERT_EQ(LwSciBufPeer::setAttr(
                      list.get(), LwSciBufTensorAttrKey_DataType, dataType),
                  LwSciError_BadParameter);
    }
    // Check that list is unchanged
    LwSciBufPeer::checkAttrListsEqual(list.get(), listClone.get());

    // Perform single invalid setAttr operation
    {
        NEGATIVE_TEST();
        LwSciBufAttrValAccessPerm perm = LwSciBufAccessPerm_Readonly;
        bool needCpuAccess = true;
        std::vector<LwSciBufAttrKeyValuePair> ilwalidAttrs = {
            {LwSciBufGeneralAttrKey_RequiredPerm, &perm, sizeof(perm)},
            {LwSciBufGeneralAttrKey_NeedCpuAccess, &needCpuAccess,
             sizeof(needCpuAccess)},
            {LwSciBufTensorAttrKey_AlignmentPerDim, &alignment,
             sizeof(alignment)},
            {LwSciBufTensorAttrKey_SizePerDim, &sizes,
             sizeof(sizes)}, // SizePerDim has already been set
        };
        ASSERT_EQ(LwSciBufAttrListSetAttrs(list.get(), ilwalidAttrs.data(),
                                           ilwalidAttrs.size()),
                  LwSciError_BadParameter);
    }
    // Check that list is unchanged
    LwSciBufPeer::checkAttrListsEqual(list.get(), listClone.get());
}

TEST_F(LwSciBufTestAttributeCore, RevertSetAttrs2)
{
    LwSciError error = LwSciError_Success;

    LwSciBufAttrValAccessPerm perm = LwSciBufAccessPerm_Readonly;
    bool needCpuAccess = true;
    LwSciBufType type = LwSciBufType_Tensor;
    std::vector<LwSciBufAttrKeyValuePair> attrs = {
        {LwSciBufGeneralAttrKey_RequiredPerm, &perm, sizeof(perm)},
        {LwSciBufGeneralAttrKey_NeedCpuAccess, &needCpuAccess,
         sizeof(needCpuAccess)},
    };
    std::vector<LwSciBufAttrKeyValuePair> ilwalidAttrs = {
        {LwSciBufGeneralAttrKey_RequiredPerm, &perm,
         sizeof(perm)},                                      // Already set
        {LwSciBufGeneralAttrKey_Types, &type, sizeof(type)}, // New key
    };

    // Set some general attributes
    ASSERT_EQ(LwSciBufAttrListSetAttrs(list.get(), attrs.data(), attrs.size()),
              LwSciError_Success);

    {
        // Perform single invalid setAttr operation
        NEGATIVE_TEST();
        ASSERT_EQ(LwSciBufPeer::verifyAttr(list.get(),
                                           LwSciBufGeneralAttrKey_Types, type),
                  false);

        ASSERT_EQ(LwSciBufAttrListSetAttrs(list.get(), ilwalidAttrs.data(),
                                           ilwalidAttrs.size()),
                  LwSciError_BadParameter);

        ASSERT_EQ(LwSciBufPeer::verifyAttr(list.get(),
                                           LwSciBufGeneralAttrKey_Types, type),
                  false);
    }
}

TEST_F(LwSciBufTestAttributeCore, RevertSetAttrs3)
{
    LwSciError error = LwSciError_Success;

    uint32_t dataType = LwSciDataType_Int16;
    LwSciBufType type = LwSciBufType_Image;
    std::vector<LwSciBufAttrKeyValuePair> ilwalidAttrs = {
        {LwSciBufTensorAttrKey_DataType, &dataType,
         sizeof(dataType)}, // Key type is invalid for LwSciBufType_Image
        {LwSciBufGeneralAttrKey_Types, &type, sizeof(type)},
    };

    {
        // Perform single invalid setAttr operation:
        // Set incompatible attribute key types
        NEGATIVE_TEST();
        ASSERT_EQ(LwSciBufAttrListSetAttrs(list.get(), ilwalidAttrs.data(),
                                           ilwalidAttrs.size()),
                  LwSciError_BadParameter);
        ASSERT_EQ(LwSciBufPeer::verifyAttr(list.get(),
                                           LwSciBufGeneralAttrKey_Types, type),
                  false);
        ASSERT_EQ(LwSciBufPeer::verifyAttr(
                      list.get(), LwSciBufTensorAttrKey_DataType, dataType),
                  false);
    }

    // Set LwSciBufGeneralAttrKey_Types, should be successfull
    ASSERT_EQ(
        LwSciBufPeer::setAttr(list.get(), LwSciBufGeneralAttrKey_Types, type),
        LwSciError_Success);
    ASSERT_EQ(LwSciBufPeer::verifyAttr(list.get(), LwSciBufGeneralAttrKey_Types,
                                       type),
              true);
}

TEST_F(LwSciBufTestAttributeCore, RevertSetAttrs4)
{
    LwSciError error = LwSciError_Success;

    LwSciBufType tensorType = LwSciBufType_Tensor;
    uint32_t dataType = LwSciDataType_Int16;
    std::vector<LwSciBufAttrKeyValuePair> tensorAttrs = {
        {LwSciBufGeneralAttrKey_Types, &tensorType, sizeof(tensorType)},
        {LwSciBufTensorAttrKey_DataType, &dataType, sizeof(dataType)},
    };

    LwSciBufType rawType = LwSciBufType_RawBuffer;
    uint64_t bufSize = 1024;
    std::vector<LwSciBufAttrKeyValuePair> rawAttrs = {
        {LwSciBufGeneralAttrKey_Types, &rawType, sizeof(rawType)},
        {LwSciBufRawBufferAttrKey_Size, &bufSize, sizeof(bufSize)},
    };

    LwSciBufType bufTypes[2] = {LwSciBufType_RawBuffer, LwSciBufType_Tensor};
    std::vector<LwSciBufAttrKeyValuePair> rawAndTensorAttrs = {
        {LwSciBufGeneralAttrKey_Types, &bufTypes, sizeof(bufTypes)},
        {LwSciBufRawBufferAttrKey_Size, &bufSize, sizeof(bufSize)},
    };

    uint32_t planeCount = 1;
    std::vector<LwSciBufAttrKeyValuePair> ilwalidAttrs = {
        {LwSciBufGeneralAttrKey_Types, &rawType, sizeof(rawType)},
        // invalid combination of attributes -- LwSciBufImageAttrKey_PlaneCount
        // is not of LwSciBufType_RawBuffer
        {LwSciBufImageAttrKey_PlaneCount, &planeCount, sizeof(planeCount)},
    };

    {
        // Set Tensor buffer type and a tensor-specific attribute
        ASSERT_EQ(LwSciBufAttrListSetAttrs(list.get(), tensorAttrs.data(),
                                           tensorAttrs.size()),
                  LwSciError_Success);

        // Verify that attributes has been actually set
        ASSERT_EQ(LwSciBufPeer::verifyAttr(
                      list.get(), LwSciBufGeneralAttrKey_Types, tensorType),
                  true);
        ASSERT_EQ(LwSciBufPeer::verifyAttr(
                      list.get(), LwSciBufTensorAttrKey_DataType, dataType),
                  true);
    }

    {
        // Now buffer type now is locked, try to overwrite with a new type --
        // should fail
        NEGATIVE_TEST();
        ASSERT_EQ(LwSciBufAttrListSetAttrs(list.get(), rawAttrs.data(),
                                           rawAttrs.size()),
                  LwSciError_BadParameter);

        // Verify that attributes has not been set
        ASSERT_EQ(LwSciBufPeer::verifyAttr(list.get(),
                                           LwSciBufGeneralAttrKey_Types, rawType),
                  false);
        ASSERT_EQ(LwSciBufPeer::verifyAttr(
                      list.get(), LwSciBufRawBufferAttrKey_Size, bufSize),
                  false);

        // Verify that old values have been preserved
        ASSERT_EQ(LwSciBufPeer::verifyAttr(
                      list.get(), LwSciBufGeneralAttrKey_Types, tensorType),
                  true);
        ASSERT_EQ(LwSciBufPeer::verifyAttr(
                      list.get(), LwSciBufTensorAttrKey_DataType, dataType),
                  true);
    }

    {
        // Clone attribute list, buffer type now is unlocked
        // -- set different new type, previous type attributes should be cleared
        auto listClone = LwSciBufPeer::attrListClone(list.get(), &error);
        ASSERT_EQ(LwSciBufAttrListSetAttrs(listClone.get(), rawAttrs.data(),
                                           rawAttrs.size()),
                  LwSciError_Success);

        // New attributes have been set
        ASSERT_EQ(LwSciBufPeer::verifyAttr(
                      listClone.get(), LwSciBufGeneralAttrKey_Types, rawType),
                  true);
        ASSERT_EQ(LwSciBufPeer::verifyAttr(
                      listClone.get(), LwSciBufRawBufferAttrKey_Size, bufSize),
                  true);
        // buffer type is different
        ASSERT_EQ(LwSciBufPeer::verifyAttr(listClone.get(),
                                           LwSciBufGeneralAttrKey_Types,
                                           tensorType),
                  false);
        // unused datatypes are cleared
        ASSERT_EQ(LwSciBufPeer::verifyAttr(listClone.get(),
                                           LwSciBufTensorAttrKey_DataType,
                                           dataType),
                  false);
    }

    {
        // Clone attribute list, buffer type now is unlocked
        // -- set different new type AND the previous one, all attributes
        // should be preserved
        auto listClone = LwSciBufPeer::attrListClone(list.get(), &error);
        ASSERT_EQ(LwSciBufAttrListSetAttrs(listClone.get(),
                                           rawAndTensorAttrs.data(),
                                           rawAndTensorAttrs.size()),
                  LwSciError_Success);

        // New attributes have been set
        ASSERT_EQ(LwSciBufPeer::verifyAttr(
                      listClone.get(), LwSciBufGeneralAttrKey_Types, bufTypes),
                  true);
        ASSERT_EQ(LwSciBufPeer::verifyAttr(
                      listClone.get(), LwSciBufRawBufferAttrKey_Size, bufSize),
                  true);
        // LwSciBufType_Tensor attributes should be preserved
        ASSERT_EQ(LwSciBufPeer::verifyAttr(listClone.get(),
                                           LwSciBufTensorAttrKey_DataType,
                                           dataType),
                  true);
    }

    {
        // Clone attribute list, buffer type now is unlocked -- setting new
        // type should pass.
        // Set invalid attributes this time to check that
        // old data type attributes are not freed.
        NEGATIVE_TEST();
        auto listClone = LwSciBufPeer::attrListClone(list.get(), &error);
        ASSERT_EQ(LwSciBufAttrListSetAttrs(listClone.get(), ilwalidAttrs.data(),
                                           ilwalidAttrs.size()),
                  LwSciError_BadParameter);

        // Verify that invalid attributes have not been set
        ASSERT_EQ(LwSciBufPeer::verifyAttr(
                      listClone.get(), LwSciBufGeneralAttrKey_Types, rawType),
                  false);
        ASSERT_EQ(LwSciBufPeer::verifyAttr(listClone.get(),
                                           LwSciBufImageAttrKey_PlaneCount,
                                           planeCount),
                  false);

        // Verify that old values have been preserved
        ASSERT_EQ(LwSciBufPeer::verifyAttr(listClone.get(),
                                           LwSciBufGeneralAttrKey_Types,
                                           tensorType),
                  true);
        ASSERT_EQ(LwSciBufPeer::verifyAttr(listClone.get(),
                                           LwSciBufTensorAttrKey_DataType,
                                           dataType),
                  true);
    }
}

TEST_F(LwSciBufTestAttributeCore,
       RevertMultipleKeys_LwSciBufGeneralAttrKey_Types)
{
    LwSciError error = LwSciError_Success;
    LwSciBufAttrKey duplicateKey = LwSciBufGeneralAttrKey_Types;

    LwSciBufType bufType = LwSciBufType_RawBuffer;
    LwSciBufAttrKeyValuePair setAttrs[] = {
        {.key = duplicateKey, .value = &bufType, .len = sizeof(bufType)},
        {.key = duplicateKey, .value = &bufType, .len = sizeof(bufType)}};
    size_t length = sizeof(setAttrs) / sizeof(LwSciBufAttrKeyValuePair);

    {
        NEGATIVE_TEST();
        error = LwSciBufAttrListSetAttrs(list.get(), setAttrs, length);
        ASSERT_EQ(error, LwSciError_BadParameter);
    }

    LwSciBufAttrKeyValuePair attr = {duplicateKey, nullptr, 0};
    error = LwSciBufAttrListSlotGetAttrs(list.get(), 0, &attr, 1);
    ASSERT_EQ(error, LwSciError_Success);

    // Assert that the value hasn't been changed
    // Note: We can't use a clone here since cloning requires that the
    // LwSciBufGeneralAttrKey_Types key be set
    ASSERT_EQ(attr.len, 0U);
}

TEST_F(LwSciBufTestAttributeCore,
       RevertMultipleKeys_LwSciBufGeneralAttrKey_NeedCpuAccess)
{
    LwSciError error = LwSciError_Success;
    LwSciBufAttrKey duplicateKey = LwSciBufGeneralAttrKey_NeedCpuAccess;

    SET_ATTR(list.get(), LwSciBufGeneralAttrKey_Types, LwSciBufType_RawBuffer);
    // Make a clone for the reference
    auto listClone = LwSciBufPeer::attrListClone(list.get(), &error);

    bool needCpuAccess = true;
    bool enableCpuCache = false;
    LwSciBufAttrKeyValuePair setAttrs[] = {
        {.key = duplicateKey,
         .value = &needCpuAccess,
         .len = sizeof(needCpuAccess)},
        {.key = LwSciBufGeneralAttrKey_EnableCpuCache,
         .value = &enableCpuCache,
         .len = sizeof(enableCpuCache)},
        {.key = duplicateKey,
         .value = &needCpuAccess,
         .len = sizeof(needCpuAccess)},
    };
    size_t length = sizeof(setAttrs) / sizeof(LwSciBufAttrKeyValuePair);

    {
        NEGATIVE_TEST();
        error = LwSciBufAttrListSetAttrs(list.get(), setAttrs, length);
        ASSERT_EQ(error, LwSciError_BadParameter);
    }

    // Check that list is unchanged
    LwSciBufPeer::checkAttrListsEqual(list.get(), listClone.get());
}

TEST_F(LwSciBufTestAttributeCore,
       RevertMultipleKeys_LwSciBufInternalGeneralAttrKey_MemDomainArray)
{
    LwSciError error = LwSciError_Success;
    LwSciBufInternalAttrKey duplicateKey =
        LwSciBufInternalGeneralAttrKey_MemDomainArray;

    SET_ATTR(list.get(), LwSciBufGeneralAttrKey_Types, LwSciBufType_RawBuffer);
    // Make a clone for the reference
    auto listClone = LwSciBufPeer::attrListClone(list.get(), &error);

    LwSciBufMemDomain memDomain = LwSciBufMemDomain_Sysmem;
    LwSciBufInternalAttrKeyValuePair setAttrs[] = {
        {.key = duplicateKey, .value = &memDomain, .len = sizeof(memDomain)},
        {.key = duplicateKey, .value = &memDomain, .len = sizeof(memDomain)},
    };
    size_t length = sizeof(setAttrs) / sizeof(LwSciBufInternalAttrKeyValuePair);

    {
        NEGATIVE_TEST();
        error = LwSciBufAttrListSetInternalAttrs(list.get(), setAttrs, length);
        ASSERT_EQ(error, LwSciError_BadParameter);
    }

    // Check that list is unchanged
    LwSciBufPeer::checkAttrListsEqual(list.get(), listClone.get());
}

TEST_F(LwSciBufTestAttributeCore, RevertMultipleKeys_UmdKey)
{
    LwSciError error = LwSciError_Success;
    LwSciBufInternalAttrKey lwMediaPrivKey;
    ASSERT_EQ(
        LwSciBufGetUMDPrivateKeyWithOffset(
            LwSciBufInternalAttrKey_LwMediaPrivateFirst, 1, &lwMediaPrivKey),
        LwSciError_Success);

    LwSciBufInternalAttrKey duplicateKey = lwMediaPrivKey;

    uint32_t lwMediaPrivKeyVal = 10;
    LwSciBufInternalAttrKeyValuePair setAttrs[] = {
        {.key = lwMediaPrivKey,
         .value = &lwMediaPrivKeyVal,
         .len = sizeof(lwMediaPrivKeyVal)},
        {.key = lwMediaPrivKey,
         .value = &lwMediaPrivKeyVal,
         .len = sizeof(lwMediaPrivKeyVal)},
    };
    size_t length = sizeof(setAttrs) / sizeof(LwSciBufInternalAttrKeyValuePair);

    {
        NEGATIVE_TEST();
        error = LwSciBufAttrListSetInternalAttrs(list.get(), setAttrs, length);
        ASSERT_EQ(error, LwSciError_BadParameter);
    }

    LwSciBufInternalAttrKeyValuePair attr = {duplicateKey, nullptr, 0};
    error = LwSciBufAttrListGetInternalAttrs(list.get(), &attr, 1);
    ASSERT_EQ(error, LwSciError_Success);

    // Assert that the value hasn't been changed
    // Note: We can't use a cloned list here since we can't enumerate all UMD
    // keys for an equality comparison
    ASSERT_EQ(attr.len, 0U);
}

TEST_F(LwSciBufTestAttributeCore, LwSciBufAttrListSlotGetAttrsNullAttrList)
{
    LwSciError error = LwSciError_Success;

    {
        NEGATIVE_TEST();

        LwSciBufAttrKeyValuePair pairArray{ };
        pairArray.key = LwSciBufGeneralAttrKey_Types;

        // Pass NULL LwSciBufAttrList
        error = LwSciBufAttrListSlotGetAttrs(nullptr, 0, &pairArray, 1);
        ASSERT_EQ(error, LwSciError_BadParameter);
    }
}

TEST_F(LwSciBufTestAttributeCore, LwSciBufAttrListSlotGetAttrsIlwalidSlotIndex)
{
    LwSciError error = LwSciError_Success;

    SET_ATTR(list.get(), LwSciBufGeneralAttrKey_Types, LwSciBufType_RawBuffer);

    size_t slotCount = LwSciBufAttrListGetSlotCount(list.get());

    {
        NEGATIVE_TEST();

        LwSciBufAttrKeyValuePair pairArray{ };
        pairArray.key = LwSciBufGeneralAttrKey_Types;

        // Pass invalid slot index
        error = LwSciBufAttrListSlotGetAttrs(list.get(), slotCount + 1, &pairArray, 1);
        ASSERT_EQ(error, LwSciError_BadParameter);
    }
}

TEST_F(LwSciBufTestAttributeCore, LwSciBufAttrListSlotGetAttrsNullPairArray)
{
    LwSciError error = LwSciError_Success;

    SET_ATTR(list.get(), LwSciBufGeneralAttrKey_Types, LwSciBufType_RawBuffer);

    {
        NEGATIVE_TEST();

        // Pass NULL pairArray
        error = LwSciBufAttrListSlotGetAttrs(list.get(), 0, nullptr, 1);
        ASSERT_EQ(error, LwSciError_BadParameter);
    }
}

TEST_F(LwSciBufTestAttributeCore, LwSciBufAttrListSlotGetAttrsEmptyPairCount)
{
    LwSciError error = LwSciError_Success;

    SET_ATTR(list.get(), LwSciBufGeneralAttrKey_Types, LwSciBufType_RawBuffer);

    {
        NEGATIVE_TEST();

        LwSciBufAttrKeyValuePair pairArray{ };
        pairArray.key = LwSciBufGeneralAttrKey_Types;

        // Pass empty pairCount
        error = LwSciBufAttrListSlotGetAttrs(list.get(), 0, &pairArray, 0);
        ASSERT_EQ(error, LwSciError_BadParameter);
    }
}

class TestLwSciBufAttrValColorFmt
    : public LwSciBufBasicTest,
      public ::testing::WithParamInterface<std::tuple<LwSciBufAttrValColorFmt>>
{
public:
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

    std::shared_ptr<LwSciBufAttrListRec> list;
};

TEST_P(TestLwSciBufAttrValColorFmt, TensorPixelFormats)
{
    auto params = GetParam();
    LwSciBufAttrValColorFmt pixelFormat = std::get<0>(params);

    LwSciBufType bufType = LwSciBufType_Tensor;

    SET_ATTR(list.get(), LwSciBufGeneralAttrKey_Types, bufType);
    SET_ATTR(list.get(), LwSciBufTensorAttrKey_PixelFormat, pixelFormat);
}
INSTANTIATE_TEST_CASE_P(
    TestLwSciBufAttrValColorFmt, TestLwSciBufAttrValColorFmt,
    /* Note: T3 Requirements follows the format used in T2, which specifies
     * each channel in reverse order specified in LwSciBuf in SWAD/SWUD. */
    testing::Values(
        /* RAW PACKED */
        /* T3 Requirements: Raw8BitPackedBGGR */
        std::make_tuple(LwSciColor_Bayer8RGGB),
        /* T3 Requirements: Raw8BitPackedCCCC */
        std::make_tuple(LwSciColor_Bayer8CCCC),
        /* T3 Requirements: Raw8BitPackedRGGB */
        std::make_tuple(LwSciColor_Bayer8BGGR),
        /* T3 Requirements: Raw8BitPackedGRBG */
        std::make_tuple(LwSciColor_Bayer8GBRG),
        /* T3 Requirements: Raw8BitPackedGBRG */
        std::make_tuple(LwSciColor_Bayer8GRBG),
        /* T3 Requirements: Raw16BitPackedRGGB */
        std::make_tuple(LwSciColor_Bayer16BGGR),
        /* T3 Requirements: Raw16BitPackedCCCC */
        std::make_tuple(LwSciColor_Bayer16CCCC),
        /* T3 Requirements: Raw16BitPackedGRBG */
        std::make_tuple(LwSciColor_Bayer16GBRG),
        /* T3 Requirements: Raw16BitPackedGBRG */
        std::make_tuple(LwSciColor_Bayer16GRBG),
        /* T3 Requirements: Raw16BitPackedBGGR */
        std::make_tuple(LwSciColor_Bayer16RGGB),
        /* T3 Requirements: Raw16BitPackedBCCR */
        std::make_tuple(LwSciColor_Bayer16RCCB),
        /* T3 Requirements: Raw16BitPackedRCCB */
        std::make_tuple(LwSciColor_Bayer16BCCR),
        /* T3 Requirements: Raw16BitPackedCBRC */
        std::make_tuple(LwSciColor_Bayer16CRBC),
        /* T3 Requirements: Raw16BitPackedCRBC */
        std::make_tuple(LwSciColor_Bayer16CBRC),
        /* T3 Requirements: Raw16BitPackedCCCR */
        std::make_tuple(LwSciColor_Bayer16RCCC),
        /* T3 Requirements: Raw16BitPackedRCCC */
        std::make_tuple(LwSciColor_Bayer16CCCR),
        /* T3 Requirements: Raw16BitPackedCCRC */
        std::make_tuple(LwSciColor_Bayer16CRCC),
        /* T3 Requirements: Raw16BitPackedCRCC */
        std::make_tuple(LwSciColor_Bayer16CCRC),
        /* T3 Requirements: Raw14BitPackedGRBG */
        std::make_tuple(LwSciColor_X2Bayer14GBRG),
        /* T3 Requirements: Raw12BitPackedGRBG */
        std::make_tuple(LwSciColor_X4Bayer12GBRG),
        /* T3 Requirements: Raw10BitPackedGRBG */
        std::make_tuple(LwSciColor_X6Bayer10GBRG),
        /* T3 Requirements: Raw14BitPackedGBRG */
        std::make_tuple(LwSciColor_X2Bayer14GRBG),
        /* T3 Requirements: Raw12BitPackedGBRG */
        std::make_tuple(LwSciColor_X4Bayer12GRBG),
        /* T3 Requirements: Raw10BitPackedGBRG */
        std::make_tuple(LwSciColor_X6Bayer10GRBG),
        /* T3 Requirements: Raw14BitPackedRGGB */
        std::make_tuple(LwSciColor_X2Bayer14BGGR),
        /* T3 Requirements: Raw12BitPackedRGGB */
        std::make_tuple(LwSciColor_X4Bayer12BGGR),
        /* T3 Requirements: Raw10BitPackedRGGB */
        std::make_tuple(LwSciColor_X6Bayer10BGGR),
        /* T3 Requirements: Raw14BitPackedBGGR */
        std::make_tuple(LwSciColor_X2Bayer14RGGB),
        /* T3 Requirements: Raw12BitPackedBGGR */
        std::make_tuple(LwSciColor_X4Bayer12RGGB),
        /* T3 Requirements: Raw10BitPackedBGGR */
        std::make_tuple(LwSciColor_X6Bayer10RGGB),
        /* T3 Requirements: Raw14BitPackedCCCC */
        std::make_tuple(LwSciColor_X2Bayer14CCCC),
        /* T3 Requirements: Raw12BitPackedCCCC */
        std::make_tuple(LwSciColor_X4Bayer12CCCC),
        /* T3 Requirements: Raw10BitPackedCCCC */
        std::make_tuple(LwSciColor_X6Bayer10CCCC),
        /* T3 Requirements: Raw12BitPackedBCCR */
        std::make_tuple(LwSciColor_X4Bayer12RCCB),
        /* T3 Requirements: Raw12BitPackedRCCB */
        std::make_tuple(LwSciColor_X4Bayer12BCCR),
        /* T3 Requirements: Raw12BitPackedCBRC*/
        std::make_tuple(LwSciColor_X4Bayer12CRBC),
        /* T3 Requirements: Raw12BitPackedCRBC */
        std::make_tuple(LwSciColor_X4Bayer12CBRC),
        /* T3 Requirements: Raw12BitPackedCCCR */
        std::make_tuple(LwSciColor_X4Bayer12RCCC),
        /* T3 Requirements: Raw12BitPackedRCCC */
        std::make_tuple(LwSciColor_X4Bayer12CCCR),
        /* T3 Requirements: Raw12BitPackedCCRC */
        std::make_tuple(LwSciColor_X4Bayer12CRCC),
        /* T3 Requirements: Raw12BitPackedCRCC */
        std::make_tuple(LwSciColor_X4Bayer12CCRC),
        /* T3 Requirements: RawSigned14BitPackedCCCC */
        std::make_tuple(LwSciColor_Signed_X2Bayer14CCCC),
        /* T3 Requirements: RawSigned12BitPackedCCCC */
        std::make_tuple(LwSciColor_Signed_X4Bayer12CCCC),
        /* T3 Requirements: RawSigned10BitPackedCCCC */
        std::make_tuple(LwSciColor_Signed_X6Bayer10CCCC),
        /* T3 Requirements: RawSigned16BitPackedCCCC */
        std::make_tuple(LwSciColor_Signed_Bayer16CCCC),
        /* T3 Requirements: RawFloatISP16BitPackedCCCC */
        std::make_tuple(LwSciColor_FloatISP_Bayer16CCCC),
        /* T3 Requirements: RawFloatISP16BitPackedBGGR */
        std::make_tuple(LwSciColor_FloatISP_Bayer16RGGB),
        /* T3 Requirements: RawFloatISP16BitPackedRGGB */
        std::make_tuple(LwSciColor_FloatISP_Bayer16BGGR),
        /* T3 Requirements: RawFloatISP16BitPackedGBRG */
        std::make_tuple(LwSciColor_FloatISP_Bayer16GRBG),
        /* T3 Requirements: RawFloatISP16BitPackedGRBG */
        std::make_tuple(LwSciColor_FloatISP_Bayer16GBRG),
        /* T3 Requirements: RawFloatISP16BitPackedBCCR */
        std::make_tuple(LwSciColor_FloatISP_Bayer16RCCB),
        /* T3 Requirements: RawFloatISP16BitPackedRCCB */
        std::make_tuple(LwSciColor_FloatISP_Bayer16BCCR),
        /* T3 Requirements: RawFloatISP16BitPackedCBRC */
        std::make_tuple(LwSciColor_FloatISP_Bayer16CRBC),
        /* T3 Requirements: RawFloatISP16BitPackedCRBC */
        std::make_tuple(LwSciColor_FloatISP_Bayer16CBRC),
        /* T3 Requirements: RawFloatISP16BitPackedCCCR */
        std::make_tuple(LwSciColor_FloatISP_Bayer16RCCC),
        /* T3 Requirements: RawFloatISP16BitPackedRCCC */
        std::make_tuple(LwSciColor_FloatISP_Bayer16CCCR),
        /* T3 Requirements: RawFloatISP16BitPackedCCRC */
        std::make_tuple(LwSciColor_FloatISP_Bayer16CRCC),
        /* T3 Requirements: RawFloatISP16BitPackedCRCC */
        std::make_tuple(LwSciColor_FloatISP_Bayer16CCRC),
        /* T3 Requirements: Raw20BitPackedCCCC */
        std::make_tuple(LwSciColor_X12Bayer20CCCC),
        /* T3 Requirements: Raw20BitPackedRGGB */
        std::make_tuple(LwSciColor_X12Bayer20BGGR),
        /* T3 Requirements: Raw20BitPackedBGGR */
        std::make_tuple(LwSciColor_X12Bayer20RGGB),
        /* T3 Requirements: Raw20BitPackedGBRG */
        std::make_tuple(LwSciColor_X12Bayer20GRBG),
        /* T3 Requirements: Raw20BitPackedGRBG (unsigned) */
        std::make_tuple(LwSciColor_X12Bayer20GBRG),
        /* T3 Requirements: Raw20BitPackedBCCR */
        std::make_tuple(LwSciColor_X12Bayer20RCCB),
        /* T3 Requirements: Raw20BitPackedRCCB */
        std::make_tuple(LwSciColor_X12Bayer20BCCR),
        /* T3 Requirements: Raw20BitPackedCBRC */
        std::make_tuple(LwSciColor_X12Bayer20CRBC),
        /* T3 Requirements: Raw20BitPackedCRBC */
        std::make_tuple(LwSciColor_X12Bayer20CBRC),
        /* T3 Requirements: Raw20BitPackedCCCR */
        std::make_tuple(LwSciColor_X12Bayer20RCCC),
        /* T3 Requirements: Raw20BitPackedRCCC */
        std::make_tuple(LwSciColor_X12Bayer20CCCR),
        /* T3 Requirements: Raw20BitPackedCCRC */
        std::make_tuple(LwSciColor_X12Bayer20CRCC),
        /* T3 Requirements: Raw20BitPackedCRCC */
        std::make_tuple(LwSciColor_X12Bayer20CCRC),
        /* T3 Requirements: RawSigned20BitPackedCCCC */
        std::make_tuple(LwSciColor_Signed_X12Bayer20CCCC),
        /* T3 Requirements: Raw20BitPackedGRBG (signed) */
        std::make_tuple(LwSciColor_Signed_X12Bayer20GBRG),

        /* Semiplanar formats */
        /* T3 Requirements: SemiUV8Bit (as single 16-bit word) */
        std::make_tuple(LwSciColor_U8V8),
        /* T3 Requirements: SemiUV8Bit (as 2 8-bit words) */
        std::make_tuple(LwSciColor_U8_V8),
        /* T3 Requirements: SemiVU8Bit (as single 16-bit word */
        std::make_tuple(LwSciColor_V8U8),
        /* T3 Requirements: SemiVU8Bit (as 2 8-bit words */
        std::make_tuple(LwSciColor_V8_U8),
        /* T3 Requirements: SemiUV10Bit */
        std::make_tuple(LwSciColor_U10V10),
        /* T3 Requirements: SemiVU10Bit */
        std::make_tuple(LwSciColor_V10U10),
        /* T3 Requirements: SemiUV12Bit */
        std::make_tuple(LwSciColor_U12V12),
        /* T3 Requirements: SemiVU12Bit */
        std::make_tuple(LwSciColor_V12U12),
        /* T3 Requirements: SemiVU16Bit */
        std::make_tuple(LwSciColor_U16V16),
        /* T3 Requirements: SemiUV16Bit */
        std::make_tuple(LwSciColor_V16U16),

        /* PLANAR formats */
        /* T3 Requirements: PlanarY8Bit */
        std::make_tuple(LwSciColor_Y8),
        /* T3 Requirements: PlanarY10Bit */
        std::make_tuple(LwSciColor_Y10),
        /* T3 Requirements: PlanarY12Bit */
        std::make_tuple(LwSciColor_Y12),
        /* T3 Requirements: PlanarY16Bit */
        std::make_tuple(LwSciColor_Y16),
        /* T3 Requirements: PlanarU8Bit */
        std::make_tuple(LwSciColor_U8),
        /* T3 Requirements: PlanarV8Bit */
        std::make_tuple(LwSciColor_V8),
        /* T3 Requirements: PlanarU10Bit */
        std::make_tuple(LwSciColor_U10),
        /* T3 Requirements: PlanarV10Bit */
        std::make_tuple(LwSciColor_V10),
        /* T3 Requirements: PlanarU12Bit */
        std::make_tuple(LwSciColor_U12),
        /* T3 Requirements: PlanarV12Bit */
        std::make_tuple(LwSciColor_V12),
        /* T3 Requirements: PlanarU16Bit */
        std::make_tuple(LwSciColor_U16),
        /* T3 Requirements: PlanarV16Bit */
        std::make_tuple(LwSciColor_V16),

        /* Packed YUV formats */
        /* T3 Requirements: PackedAYUV8Bit */
        std::make_tuple(LwSciColor_A8Y8U8V8),
        /* T3 Requirements: PackedYUYV8Bit */
        std::make_tuple(LwSciColor_Y8U8Y8V8),
        /* T3 Requirements: PackedYVYU8Bit */
        std::make_tuple(LwSciColor_Y8V8Y8U8),
        /* T3 Requirements: PackedUYVY8Bit */
        std::make_tuple(LwSciColor_U8Y8V8Y8),
        /* T3 Requirements: PackedVYUY8Bit */
        std::make_tuple(LwSciColor_V8Y8U8Y8),
        /* T3 Requirements: PackedAYUV16Bit */
        std::make_tuple(LwSciColor_A16Y16U16V16),

        /* RGBA PACKED */
        /* T3 Requirements: PackedAlpha8Bit */
        std::make_tuple(LwSciColor_A8),
        /* T3 Requirements: PackedAlphaSigned8Bit */
        std::make_tuple(LwSciColor_Signed_A8),
        /* T3 Requirements: PackedARGB8Bit */
        std::make_tuple(LwSciColor_B8G8R8A8),
        /* T3 Requirements: PackedBGRA8Bit */
        std::make_tuple(LwSciColor_A8R8G8B8),
        /* T3 Requirements: PackedRGBA8Bit */
        std::make_tuple(LwSciColor_A8B8G8R8),
        /* T3 Requirements: PackedA2R10G10B10 */
        std::make_tuple(LwSciColor_A2R10G10B10),
        /* T3 Requirements: PackedAlpha16Bit */
        std::make_tuple(LwSciColor_A16),
        /* T3 Requirements: PackedAlphaSigned16Bit */
        std::make_tuple(LwSciColor_Signed_A16),
        /* T3 Requirements: PackedRGSigned16Bit */
        std::make_tuple(LwSciColor_Signed_R16G16),
        /* T3 Requirements: PackedRGBA16Bit */
        std::make_tuple(LwSciColor_A16B16G16R16),
        /* T3 Requirements: PackedRGBASigned16Bit */
        std::make_tuple(LwSciColor_Signed_A16B16G16R16),
        /* T3 Requirements: PackedRGBAFP16Bit */
        std::make_tuple(LwSciColor_Float_A16B16G16R16),
        /* T3 Requirements: PackedAlpha32Bit */
        std::make_tuple(LwSciColor_A32),
        /* T3 Requirements: PackedAlphaSigned32Bit */
        std::make_tuple(LwSciColor_Signed_A32),
        /* T3 Requirements: PackedAlphaFP16Bit */
        std::make_tuple(LwSciColor_Float_A16)));

class TestLwSciBufSurfType : public LwSciBufBasicTest,
                             public ::testing::WithParamInterface<
                                 std::tuple<LwSciBufSurfType, LwSciError>>
{
};

TEST_P(TestLwSciBufSurfType, LwSciBufSurfTypes)
{
    auto params = GetParam();

    LwSciBufSurfType surfaceType = std::get<0>(params);
    LwSciError expectedErr = std::get<1>(params);

    LwSciError error = LwSciError_Success;
    auto list = peer.createAttrList(&error);
    ASSERT_EQ(error, LwSciError_Success);

    SET_ATTR(list.get(), LwSciBufGeneralAttrKey_Types, LwSciBufType_Image);
    if (expectedErr == LwSciError_Success) {
        SET_ATTR(list.get(), LwSciBufImageAttrKey_SurfType, surfaceType);
    } else {
        NEGATIVE_TEST();
        LwSciBufAttrKeyValuePair pairArray[] = {
            {
                LwSciBufImageAttrKey_SurfType,
                &surfaceType,
                sizeof(surfaceType),
            },
        };
        error = LwSciBufAttrListSetAttrs(
            list.get(), pairArray, sizeof(pairArray) / sizeof(pairArray[0]));
        ASSERT_EQ(error, expectedErr);
    }
}
INSTANTIATE_TEST_CASE_P(
    TestLwSciBufSurfType, TestLwSciBufSurfType,
    testing::Values(
        std::make_tuple(LwSciSurfType_YUV, LwSciError_Success),
        std::make_tuple(LwSciSurfType_RGBA, LwSciError_BadParameter),
        std::make_tuple(LwSciSurfType_RAW, LwSciError_BadParameter),
        std::make_tuple(LwSciSurfType_MaxValid, LwSciError_BadParameter),
        std::make_tuple(
            static_cast<LwSciBufSurfType>(
                std::numeric_limits<
                    std::underlying_type<LwSciBufSurfType>::type>::max()),
            LwSciError_BadParameter)));

class TestLwSciBufSurfMemLayout
    : public LwSciBufBasicTest,
      public ::testing::WithParamInterface<
          std::tuple<LwSciBufSurfMemLayout, LwSciError>>
{
};

TEST_P(TestLwSciBufSurfMemLayout, LwSciBufSurfMemLayouts)
{
    auto params = GetParam();

    LwSciBufSurfMemLayout surfaceMemLayout = std::get<0>(params);
    LwSciError expectedErr = std::get<1>(params);

    LwSciError error = LwSciError_Success;
    auto list = peer.createAttrList(&error);
    ASSERT_EQ(error, LwSciError_Success);

    SET_ATTR(list.get(), LwSciBufGeneralAttrKey_Types, LwSciBufType_Image);
    if (expectedErr == LwSciError_Success) {
        SET_ATTR(list.get(), LwSciBufImageAttrKey_SurfMemLayout,
                 surfaceMemLayout);
    } else {
        NEGATIVE_TEST();
        LwSciBufAttrKeyValuePair pairArray[] = {
            {
                LwSciBufImageAttrKey_SurfMemLayout,
                &surfaceMemLayout,
                sizeof(surfaceMemLayout),
            },
        };
        error = LwSciBufAttrListSetAttrs(
            list.get(), pairArray, sizeof(pairArray) / sizeof(pairArray[0]));
        ASSERT_EQ(error, expectedErr);
    }
}
INSTANTIATE_TEST_CASE_P(
    TestLwSciBufSurfMemLayout, TestLwSciBufSurfMemLayout,
    testing::Values(
        std::make_tuple(LwSciSurfMemLayout_Planar, LwSciError_Success),
        std::make_tuple(LwSciSurfMemLayout_SemiPlanar, LwSciError_Success),
        std::make_tuple(LwSciSurfMemLayout_Packed, LwSciError_BadParameter),
        std::make_tuple(LwSciSurfMemLayout_MaxValid, LwSciError_BadParameter),
        std::make_tuple(
            static_cast<LwSciBufSurfMemLayout>(
                std::numeric_limits<
                    std::underlying_type<LwSciBufSurfMemLayout>::type>::max()),
            LwSciError_BadParameter)));

class TestLwSciBufSurfSampleType
    : public LwSciBufBasicTest,
      public ::testing::WithParamInterface<
          std::tuple<LwSciBufSurfSampleType, LwSciError>>
{
};

TEST_P(TestLwSciBufSurfSampleType, LwSciBufSurfSampleTypes)
{
    auto params = GetParam();

    LwSciBufSurfSampleType surfaceSampleType = std::get<0>(params);
    LwSciError expectedErr = std::get<1>(params);

    LwSciError error = LwSciError_Success;
    auto list = peer.createAttrList(&error);
    ASSERT_EQ(error, LwSciError_Success);

    SET_ATTR(list.get(), LwSciBufGeneralAttrKey_Types, LwSciBufType_Image);
    if (expectedErr == LwSciError_Success) {
        SET_ATTR(list.get(), LwSciBufImageAttrKey_SurfSampleType,
                 surfaceSampleType);
    } else {
        NEGATIVE_TEST();
        LwSciBufAttrKeyValuePair pairArray[] = {
            {
                LwSciBufImageAttrKey_SurfSampleType,
                &surfaceSampleType,
                sizeof(surfaceSampleType),
            },
        };
        error = LwSciBufAttrListSetAttrs(
            list.get(), pairArray, sizeof(pairArray) / sizeof(pairArray[0]));
        ASSERT_EQ(error, expectedErr);
    }
}
INSTANTIATE_TEST_CASE_P(
    TestLwSciBufSurfSampleType, TestLwSciBufSurfSampleType,
    testing::Values(
        std::make_tuple(LwSciSurfSampleType_420, LwSciError_Success),
        std::make_tuple(LwSciSurfSampleType_422, LwSciError_Success),
        std::make_tuple(LwSciSurfSampleType_444, LwSciError_Success),
        std::make_tuple(LwSciSurfSampleType_422R, LwSciError_Success),
        std::make_tuple(LwSciSurfSampleType_MaxValid, LwSciError_BadParameter),
        std::make_tuple(
            static_cast<LwSciBufSurfSampleType>(
                std::numeric_limits<
                    std::underlying_type<LwSciBufSurfSampleType>::type>::max()),
            LwSciError_BadParameter)));

class TestLwSciBufSurfBPC : public LwSciBufBasicTest,
                            public ::testing::WithParamInterface<
                                std::tuple<LwSciBufSurfBPC, LwSciError>>
{
};

TEST_P(TestLwSciBufSurfBPC, LwSciBufSurfBPCs)
{
    auto params = GetParam();

    LwSciBufSurfBPC surfaceBPC = std::get<0>(params);
    LwSciError expectedErr = std::get<1>(params);

    LwSciError error = LwSciError_Success;
    auto list = peer.createAttrList(&error);
    ASSERT_EQ(error, LwSciError_Success);

    SET_ATTR(list.get(), LwSciBufGeneralAttrKey_Types, LwSciBufType_Image);
    if (expectedErr == LwSciError_Success) {
        SET_ATTR(list.get(), LwSciBufImageAttrKey_SurfBPC, surfaceBPC);
    } else {
        NEGATIVE_TEST();
        LwSciBufAttrKeyValuePair pairArray[] = {
            {
                LwSciBufImageAttrKey_SurfBPC,
                &surfaceBPC,
                sizeof(surfaceBPC),
            },
        };
        error = LwSciBufAttrListSetAttrs(
            list.get(), pairArray, sizeof(pairArray) / sizeof(pairArray[0]));
        ASSERT_EQ(error, expectedErr);
    }
}
INSTANTIATE_TEST_CASE_P(
    TestLwSciBufSurfBPC, TestLwSciBufSurfBPC,
    testing::Values(
        std::make_tuple(LwSciSurfBPC_Layout_16_8_8, LwSciError_Success),
        std::make_tuple(LwSciSurfBPC_Layout_10_8_8, LwSciError_Success),
        std::make_tuple(LwSciSurfBPC_MaxValid, LwSciError_BadParameter),
        std::make_tuple(
            static_cast<LwSciBufSurfBPC>(
                std::numeric_limits<
                    std::underlying_type<LwSciBufSurfBPC>::type>::max()),
            LwSciError_BadParameter)));

class TestLwSciBufSurfComponentOrder
    : public LwSciBufBasicTest,
      public ::testing::WithParamInterface<
          std::tuple<LwSciBufSurfComponentOrder, LwSciError>>
{
};

TEST_P(TestLwSciBufSurfComponentOrder, LwSciBufSurfComponentOrders)
{
    auto params = GetParam();

    LwSciBufSurfComponentOrder surfaceComponentOrder = std::get<0>(params);
    LwSciError expectedErr = std::get<1>(params);

    LwSciError error = LwSciError_Success;
    auto list = peer.createAttrList(&error);
    ASSERT_EQ(error, LwSciError_Success);

    SET_ATTR(list.get(), LwSciBufGeneralAttrKey_Types, LwSciBufType_Image);
    if (expectedErr == LwSciError_Success) {
        SET_ATTR(list.get(), LwSciBufImageAttrKey_SurfComponentOrder,
                 surfaceComponentOrder);
    } else {
        NEGATIVE_TEST();
        LwSciBufAttrKeyValuePair pairArray[] = {
            {
                LwSciBufImageAttrKey_SurfComponentOrder,
                &surfaceComponentOrder,
                sizeof(surfaceComponentOrder),
            },
        };
        error = LwSciBufAttrListSetAttrs(
            list.get(), pairArray, sizeof(pairArray) / sizeof(pairArray[0]));
        ASSERT_EQ(error, expectedErr);
    }
}
INSTANTIATE_TEST_CASE_P(
    TestLwSciBufSurfComponentOrder, TestLwSciBufSurfComponentOrder,
    testing::Values(
        std::make_tuple(LwSciSurfComponentOrder_YUV, LwSciError_Success),
        std::make_tuple(LwSciSurfComponentOrder_YVU, LwSciError_Success),
        std::make_tuple(LwSciSurfComponentOrder_MaxValid,
                        LwSciError_BadParameter),
        std::make_tuple(static_cast<LwSciBufSurfComponentOrder>(
                            std::numeric_limits<std::underlying_type<
                                LwSciBufSurfComponentOrder>::type>::max()),
                        LwSciError_BadParameter)));
