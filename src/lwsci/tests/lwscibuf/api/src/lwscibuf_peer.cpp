/*
 * Copyright (c) 2020-2022, LWPU CORPORATION. All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#include "lwscibuf_peer.h"
#include <array>

std::array<LwSciBufAttrKey, 1> LwSciBufPeer::inputAttrKeys = {
    LwSciBufGeneralAttrKey_RequiredPerm,
};

std::array<LwSciBufAttrKey, 20> LwSciBufPeer::outputAttrKeys = {
    LwSciBufGeneralAttrKey_CpuNeedSwCacheCoherency,
    LwSciBufGeneralAttrKey_ActualPerm,
    LwSciBufGeneralAttrKey_GpuSwNeedCacheCoherency,
    LwSciBufImageAttrKey_Size,
    LwSciBufImageAttrKey_Alignment,
    LwSciBufImageAttrKey_PlaneBitsPerPixel,
    LwSciBufImageAttrKey_PlaneOffset,
    LwSciBufImageAttrKey_PlaneDatatype,
    LwSciBufImageAttrKey_PlaneChannelCount,
    LwSciBufImageAttrKey_PlaneSecondFieldOffset,
    LwSciBufImageAttrKey_PlanePitch,
    LwSciBufImageAttrKey_PlaneAlignedHeight,
    LwSciBufImageAttrKey_PlaneAlignedSize,
    LwSciBufTensorAttrKey_StridesPerDim,
    LwSciBufTensorAttrKey_Size,
    LwSciBufArrayAttrKey_Size,
    LwSciBufArrayAttrKey_Alignment,
    LwSciBufPyramidAttrKey_LevelOffset,
    LwSciBufPyramidAttrKey_LevelSize,
    LwSciBufPyramidAttrKey_Alignment,
};

std::array<LwSciBufAttrKey, 42> LwSciBufPeer::attrKeys = {
    LwSciBufGeneralAttrKey_Types,
    LwSciBufGeneralAttrKey_NeedCpuAccess,
    LwSciBufGeneralAttrKey_EnableCpuCache,
    LwSciBufGeneralAttrKey_GpuId,
    LwSciBufGeneralAttrKey_VidMem_GpuId,
    LwSciBufGeneralAttrKey_EnableGpuCache,
    LwSciBufGeneralAttrKey_EnableGpuCompression,
    LwSciBufRawBufferAttrKey_Size,
    LwSciBufRawBufferAttrKey_Align,
    LwSciBufImageAttrKey_Layout,
    LwSciBufImageAttrKey_TopPadding,
    LwSciBufImageAttrKey_BottomPadding,
    LwSciBufImageAttrKey_LeftPadding,
    LwSciBufImageAttrKey_RightPadding,
    LwSciBufImageAttrKey_VprFlag,
    LwSciBufImageAttrKey_PlaneCount,
    LwSciBufImageAttrKey_PlaneColorFormat,
    LwSciBufImageAttrKey_PlaneColorStd,
    LwSciBufImageAttrKey_PlaneBaseAddrAlign,
    LwSciBufImageAttrKey_PlaneWidth,
    LwSciBufImageAttrKey_PlaneHeight,
    LwSciBufImageAttrKey_PlaneScanType,
    LwSciBufImageAttrKey_ScanType,
    LwSciBufImageAttrKey_ImageCount,
    LwSciBufImageAttrKey_SurfType,
    LwSciBufImageAttrKey_SurfMemLayout,
    LwSciBufImageAttrKey_SurfSampleType,
    LwSciBufImageAttrKey_SurfBPC,
    LwSciBufImageAttrKey_SurfComponentOrder,
    LwSciBufImageAttrKey_SurfWidthBase,
    LwSciBufImageAttrKey_SurfHeightBase,
    LwSciBufTensorAttrKey_DataType,
    LwSciBufTensorAttrKey_NumDims,
    LwSciBufTensorAttrKey_SizePerDim,
    LwSciBufTensorAttrKey_AlignmentPerDim,
    LwSciBufTensorAttrKey_PixelFormat,
    LwSciBufTensorAttrKey_BaseAddrAlign,
    LwSciBufArrayAttrKey_DataType,
    LwSciBufArrayAttrKey_Stride,
    LwSciBufArrayAttrKey_Capacity,
    LwSciBufPyramidAttrKey_NumLevels,
    LwSciBufPyramidAttrKey_Scale,
};

std::array<LwSciBufInternalAttrKey, 4> LwSciBufPeer::outputInternalAttrKeys = {
    LwSciBufInternalImageAttrKey_PlaneGobSize,
    LwSciBufInternalImageAttrKey_PlaneLog2GobsPerBlockX,
    LwSciBufInternalImageAttrKey_PlaneLog2GobsPerBlockY,
    LwSciBufInternalImageAttrKey_PlaneLog2GobsPerBlockZ,
};

std::array<LwSciBufInternalAttrKey, 2> LwSciBufPeer::internalAttrKeys = {
    LwSciBufInternalGeneralAttrKey_EngineArray,
    LwSciBufInternalGeneralAttrKey_MemDomainArray,
};

std::shared_ptr<LwSciBufAttrListRec>
LwSciBufPeer::createAttrList(LwSciError* error)
{
    LwSciBufAttrList attrList = nullptr;
    *error = LwSciBufAttrListCreate(m_module.get(), &attrList);
    return std::shared_ptr<LwSciBufAttrListRec>(attrList, LwSciBufAttrListFree);
}

std::shared_ptr<LwSciBufAttrListRec>
LwSciBufPeer::attrListReconcile(const std::vector<LwSciBufAttrList>& lists,
                                LwSciError* error)
{
    LwSciBufAttrList newList = nullptr;
    LwSciBufAttrList newConflictList = nullptr;
    *error = LwSciBufAttrListReconcile(lists.data(), lists.size(), &newList,
                                       &newConflictList);
    if (*error != LwSciError_Success) {
        if (newConflictList) {
            LwSciBufAttrListFree(newConflictList);
        }
        return std::shared_ptr<LwSciBufAttrListRec>(nullptr);
    } else {
        return std::shared_ptr<LwSciBufAttrListRec>(newList,
                                                    LwSciBufAttrListFree);
    }
}

LwSciError
LwSciBufPeer::validateReconciled(std::vector<LwSciBufAttrList> inputLists,
                                 LwSciBufAttrList newReconciledList,
                                 bool* isReconciledListValid)
{
    return LwSciBufAttrListValidateReconciled(
        newReconciledList, const_cast<LwSciBufAttrList*>(inputLists.data()),
        inputLists.size(), isReconciledListValid);
}

std::shared_ptr<LwSciBufAttrListRec>
LwSciBufPeer::attrListAppend(const std::vector<LwSciBufAttrList>& lists,
                             LwSciError* error)
{
    LwSciBufAttrList newList = nullptr;
    *error = LwSciBufAttrListAppendUnreconciled(lists.data(), lists.size(),
                                                &newList);
    return std::shared_ptr<LwSciBufAttrListRec>(newList, LwSciBufAttrListFree);
}

std::shared_ptr<LwSciBufAttrListRec>
LwSciBufPeer::attrListClone(LwSciBufAttrList list, LwSciError* error)
{
    LwSciBufAttrList newList = nullptr;
    *error = LwSciBufAttrListClone(list, &newList);
    return std::shared_ptr<LwSciBufAttrListRec>(newList, LwSciBufAttrListFree);
}

void LwSciBufPeer::testObject(LwSciBufObj bufferObj, size_t alignment)
{
    /* get mem handle */
    LwSciBufRmHandle rmhandle;
    uint64_t offset, len, size;
    ASSERT_EQ(LwSciBufObjGetMemHandle(bufferObj, &rmhandle, &offset, &len),
              LwSciError_Success);

    /* get CPU */
    void* va_ptr;
    ASSERT_EQ(LwSciBufObjGetCpuPtr(bufferObj, &va_ptr), LwSciError_Success);

    /* Verify CPU access */
    uint64_t testval = 0U;
    *(uint32_t*)va_ptr = (uint32_t)0xC0DEC0DE;
    testval = *(uint32_t*)va_ptr;

    ASSERT_EQ(testval, *(uint32_t*)va_ptr);
    ASSERT_EQ(testval, 0xC0DEC0DE);

    /* verify size */
    ASSERT_EQ(GetMemorySize(rmhandle), CEIL_TO_LEVEL(len, alignment));
}

void LwSciBufPeer::checkAttrEqual(LwSciBufAttrList listA,
                                  LwSciBufAttrList listB,
                                  LwSciBufAttrKey key,
                                  size_t slotIndex)
{
    const void* valueA;
    const void* valueB;
    size_t lenA;
    size_t lenB;
    LwSciBufAttrKeyValuePair pairA[1] = {key, nullptr, 0};
    LwSciBufAttrKeyValuePair pairB[1] = {key, nullptr, 0};
    ASSERT_EQ(LwSciBufAttrListSlotGetAttrs(listA, slotIndex, pairA, 1),
              LwSciError_Success)
        << "key: " << key << " [" << ATTR_NAME(key)
        << "], slotIndex: " << slotIndex;
    ASSERT_EQ(LwSciBufAttrListSlotGetAttrs(listB, slotIndex, pairB, 1),
              LwSciError_Success)
        << "key: " << key << " [" << ATTR_NAME(key)
        << "], slotIndex: " << slotIndex;
    ASSERT_EQ(pairA->len, pairB->len)
        << "key: " << key << " [" << ATTR_NAME(key)
        << "], slotIndex: " << slotIndex;
    if (pairA->len > 0) {
        ASSERT_EQ(memcmp(pairA->value, pairB->value, pairA->len), 0)
            << ATTR_NAME(key) << " slotIndex " << slotIndex;
    }
}

void LwSciBufPeer::checkInternalAttrEqual(LwSciBufAttrList listA,
                                          LwSciBufAttrList listB,
                                          LwSciBufInternalAttrKey key)
{
    LwSciBufInternalAttrKeyValuePair pairA[1] = {key, nullptr, 0};
    LwSciBufInternalAttrKeyValuePair pairB[1] = {key, nullptr, 0};
    ASSERT_EQ(LwSciBufAttrListGetInternalAttrs(listA, pairA, 1),
              LwSciError_Success)
        << "key: " << key << " [" << INTERNAL_ATTR_NAME(key) << "]";
    ASSERT_EQ(LwSciBufAttrListGetInternalAttrs(listB, pairB, 1),
              LwSciError_Success)
        << "key: " << key << " [" << INTERNAL_ATTR_NAME(key) << "]";
    ASSERT_EQ(pairA->len, pairB->len) << INTERNAL_ATTR_NAME(key);
    if (pairA->len > 0) {
        ASSERT_EQ(memcmp(pairA->value, pairB->value, pairA->len), 0)
            << "key: " << key << " [" << INTERNAL_ATTR_NAME(key) << "]";
    }
}

void LwSciBufPeer::checkAttrListsEqual(LwSciBufAttrList listA,
                                       LwSciBufAttrList listB)
{
    ASSERT_EQ(LwSciBufAttrListGetSlotCount(listA),
              LwSciBufAttrListGetSlotCount(listB));

    bool listAReconciled = false;
    bool listBReconciled = false;
    ASSERT_EQ(LwSciBufAttrListIsReconciled(listA, &listAReconciled),
              LwSciError_Success);
    ASSERT_EQ(LwSciBufAttrListIsReconciled(listB, &listBReconciled),
              LwSciError_Success);

    ASSERT_EQ(listAReconciled, listBReconciled);

    if (listAReconciled) {
        // Check for output-only attributes in reconciled list
        for (auto const& attrKey : LwSciBufPeer::outputAttrKeys) {
            checkAttrEqual(listA, listB, attrKey, 0);
        }
        for (auto const& internalAttrKey :
             LwSciBufPeer::outputInternalAttrKeys) {
            checkInternalAttrEqual(listA, listB, internalAttrKey);
        }
    } else {
        // Check for input-only attributes in unreconciled list
        for (size_t i = 0; i < LwSciBufAttrListGetSlotCount(listA); i++) {
            for (auto const& attrKey : LwSciBufPeer::inputAttrKeys) {
                checkAttrEqual(listA, listB, attrKey, i);
            }
        }
    }

    for (size_t i = 0; i < LwSciBufAttrListGetSlotCount(listA); i++) {
        for (auto const& attrKey : LwSciBufPeer::attrKeys) {
            checkAttrEqual(listA, listB, attrKey, i);
        }
    }

    for (auto const& internalAttrKey : LwSciBufPeer::internalAttrKeys) {
        checkInternalAttrEqual(listA, listB, internalAttrKey);
    }
}
