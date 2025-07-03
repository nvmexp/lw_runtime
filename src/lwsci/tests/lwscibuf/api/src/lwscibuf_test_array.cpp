/*
 * Copyright (c) 2019-2022, LWPU CORPORATION. All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#include "lwscibuf_basic_test.h"

// This is to avoid rtti compilation error in x86 build
#define GTEST_HAS_RTTI 0

class TestLwSciBufArray : public LwSciBufBasicTest
{
public:
    void SetUp() override
    {
        LwSciError error = LwSciError_Success;

        LwSciBufBasicTest::SetUp();

        umd1AttrList = peer.createAttrList(&error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_NE(umd1AttrList.get(), nullptr);

        umd2AttrList = peer.createAttrList(&error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_NE(umd2AttrList.get(), nullptr);
    }

    void TearDown() override
    {
        LwSciBufBasicTest::TearDown();

        umd1AttrList.reset();
        umd2AttrList.reset();
    }

    std::shared_ptr<LwSciBufAttrListRec> umd1AttrList;
    std::shared_ptr<LwSciBufAttrListRec> umd2AttrList;
};

TEST_F(TestLwSciBufArray, IntraThreadArray)
{
    {
        LwSciBufType bufType = LwSciBufType_Array;
        LwSciBufAttrValDataType dataType = LwSciDataType_Uint32;
        uint64_t capacity = 100U;
        uint64_t stride = 64U;
        bool cpuAccessFlag = true;

        SET_ATTR(umd1AttrList.get(), LwSciBufGeneralAttrKey_Types, bufType);
        SET_ATTR(umd1AttrList.get(), LwSciBufArrayAttrKey_DataType, dataType);
        SET_ATTR(umd1AttrList.get(), LwSciBufArrayAttrKey_Stride, stride);
        SET_ATTR(umd1AttrList.get(), LwSciBufArrayAttrKey_Capacity, capacity);
        SET_ATTR(umd1AttrList.get(), LwSciBufGeneralAttrKey_NeedCpuAccess,
                 cpuAccessFlag);
    }

    {
        LwSciBufType bufType = LwSciBufType_Array;
        LwSciBufAttrValDataType dataType = LwSciDataType_Uint32;
        bool cpuAccessFlag = true;
        uint64_t capacity = 100;
        uint64_t stride = 64;
        LwSciBufAttrValAccessPerm perm = LwSciBufAccessPerm_ReadWrite;
        LwSciBufMemDomain memDomain = LwSciBufMemDomain_Sysmem;
        LwSciBufHwEngine engine{};

#if !defined(__x86_64__)
        LwSciBufHwEngCreateIdWithoutInstance(LwSciBufHwEngName_Vic,
                                             &engine.rmModuleID);
#else
        // the following field should be queried first by UMD
        engine.engNamespace = LwSciBufHwEngine_ResmanNamespaceId;
        engine.subEngineID = LW2080_ENGINE_TYPE_GRAPHICS;
        engine.rev.gpu.arch = LW2080_CTRL_MC_ARCH_INFO_ARCHITECTURE_GV100;
#endif

        SET_ATTR(umd2AttrList.get(), LwSciBufGeneralAttrKey_Types, bufType);
        SET_ATTR(umd2AttrList.get(), LwSciBufArrayAttrKey_DataType, dataType);
        SET_ATTR(umd2AttrList.get(), LwSciBufArrayAttrKey_Stride, stride);
        SET_ATTR(umd2AttrList.get(), LwSciBufArrayAttrKey_Capacity, capacity);
        SET_ATTR(umd2AttrList.get(), LwSciBufGeneralAttrKey_RequiredPerm, perm);

        SET_INTERNAL_ATTR(umd2AttrList.get(),
                          LwSciBufInternalGeneralAttrKey_EngineArray, engine);
        SET_INTERNAL_ATTR(umd2AttrList.get(),
                          LwSciBufInternalGeneralAttrKey_MemDomainArray,
                          memDomain);
    }

    LwSciError error = LwSciError_Success;
    auto bufObj = LwSciBufPeer::reconcileAndAllocate(
        {umd1AttrList.get(), umd2AttrList.get()}, &error);
    ASSERT_EQ(error, LwSciError_Success);

    LwSciBufRmHandle rmHandle = {0};
    uint64_t offset = 0U, len = 0U, testval = 0U, size = 0U;
    ASSERT_EQ(LwSciBufObjGetMemHandle(bufObj.get(), &rmHandle, &offset, &len),
              LwSciError_Success)
        << "Failed to Get Lwrm Memhandle for the object";

    /* Get CPU address */
    void* vaPtr = NULL;
    ASSERT_EQ(LwSciBufObjGetCpuPtr(bufObj.get(), &vaPtr), LwSciError_Success)
        << "Failed to Get CPU ptr for the object";

    /* Verify CPU access */
    *(uint32_t *)vaPtr = (uint32_t)0xC0DEC0DEU;
    testval = *(uint32_t *)vaPtr;
    ASSERT_EQ(testval, *(uint32_t *)vaPtr) << "Write failed";
    ASSERT_EQ(testval, 0xC0DEC0DEU) << "Read failed";

    /* Get size from RM */
    size = GetMemorySize(rmHandle);
    ASSERT_EQ(size, CEIL_TO_LEVEL(len, GetPageSize()))
        << "size verification failed"
        << " Expected " << size << " Got " << CEIL_TO_LEVEL(len, GetPageSize());
}

class TestLwSciBufArrayMandatoryOptionalAttrs :
                          public TestLwSciBufArray,
                          public ::testing::WithParamInterface<
                            std::tuple<LwSciBufAttrKey, LwSciError>>
{
};

/*
 * This test verifies that if mandatory attributes are not set in the
 * unreconciled list then the reconciliation fails.
 */
TEST_P(TestLwSciBufArrayMandatoryOptionalAttrs, MandatoryOptionalAttrs)
{
    LwSciBufType bufType = LwSciBufType_Array;
    LwSciBufAttrValDataType dataType = LwSciDataType_Uint32;
    uint64_t stride = 64U;
    uint64_t capacity = 100U;
    LwSciError error = LwSciError_Success;

    auto param = GetParam();
    auto key = std::get<0>(param);
    auto expectedError = std::get<1>(param);

    SET_ATTR(umd1AttrList.get(), LwSciBufGeneralAttrKey_Types, bufType);

    if (key != LwSciBufArrayAttrKey_DataType) {
        SET_ATTR(umd1AttrList.get(), LwSciBufArrayAttrKey_DataType, dataType);
    }

    if (key != LwSciBufArrayAttrKey_Stride) {
        SET_ATTR(umd1AttrList.get(), LwSciBufArrayAttrKey_Stride, stride);
    }

    if (key != LwSciBufArrayAttrKey_Capacity) {
        SET_ATTR(umd1AttrList.get(), LwSciBufArrayAttrKey_Capacity, capacity);
    }

    if (expectedError != LwSciError_Success) {
        NEGATIVE_TEST();
        auto reconciledList =
            LwSciBufPeer::attrListReconcile({umd1AttrList.get()}, &error);
    } else {
        auto reconciledList =
            LwSciBufPeer::attrListReconcile({umd1AttrList.get()}, &error);
    }
    ASSERT_EQ(error, expectedError);
}
INSTANTIATE_TEST_CASE_P(
    TestLwSciBufArrayMandatoryOptionalAttrs,
    TestLwSciBufArrayMandatoryOptionalAttrs,
    ::testing::Values(
        /* First value in tuple represents an attribute to be skipped from
         * setting in the unreconciled list.
         * Second value in tuple represents an expected error code during
         * reconciliation based on whether the attribute skipped from being
         * set in unreconciled list is mandatory or optional.
         */
        std::make_tuple(LwSciBufArrayAttrKey_DataType,
                LwSciError_ReconciliationFailed),
        std::make_tuple(LwSciBufArrayAttrKey_Stride,
                LwSciError_ReconciliationFailed),
        std::make_tuple(LwSciBufArrayAttrKey_Capacity,
                LwSciError_ReconciliationFailed),
        std::make_tuple(LwSciBufAttrKey_LowerBound,
                LwSciError_Success)));

/*
 * This test verifies the input/output accessibility of the array attributes.
 * If the attribute is input attribute (input only or input/output) then
 * attribute can be set in unreconciled list. It can also be read from
 * unreconciled list.
 * If the attribute is output attribute (output only or input/output) then
 * attribute can be read from reconciled list.
 */
TEST_F(TestLwSciBufArray, InOutAttrs)
{
    LwSciError error = LwSciError_Success;
    LwSciBufType bufType = LwSciBufType_Array;
    LwSciBufAttrValDataType dataType = LwSciDataType_Uint32;
    uint64_t stride = 64U;
    uint64_t capacity = 100U;
    uint64_t size = 4096U;
    uint64_t alignment = 4096U;

    std::vector<LwSciBufAttrKeyValuePair> arrayAttrKeySet = {
        {
            .key = LwSciBufArrayAttrKey_DataType,
            .value = &dataType,
            .len = sizeof(dataType)
        },

        {
            .key = LwSciBufArrayAttrKey_Stride,
            .value = &stride,
            .len = sizeof(stride)
        },

        {
            .key = LwSciBufArrayAttrKey_Capacity,
            .value = &capacity,
            .len = sizeof(capacity)
        },

        {
            .key = LwSciBufArrayAttrKey_Size,
            .value = &size,
            .len = sizeof(size)
        },

        {
            .key = LwSciBufArrayAttrKey_Alignment,
            .value = &alignment,
            .len = sizeof(alignment)
        },
    };

    SET_ATTR(umd1AttrList.get(), LwSciBufGeneralAttrKey_Types, bufType);

    for (auto arrayAttrKey : arrayAttrKeySet) {
        auto it1 = std::find(LwSciBufPeer::outputAttrKeys.begin(),
                            LwSciBufPeer::outputAttrKeys.end(),
                            arrayAttrKey.key);
        if (it1 == LwSciBufPeer::outputAttrKeys.end()) {
            /* attribute is input only or input/output attribute. We should be
             * able to set it in the unreconciled list. We should also be able
             * to get it from unreconciled list.
             */
            error = LwSciBufAttrListSetAttrs(umd1AttrList.get(),
                        &arrayAttrKey, 1);
            ASSERT_EQ(error, LwSciError_Success);

            error = LwSciBufAttrListGetAttrs(umd1AttrList.get(),
                        &arrayAttrKey, 1);
            ASSERT_EQ(error, LwSciError_Success);
        } else {
            /* attribute is output only. Trying to set it in unreconciled list
             * should throw an error. Similarly, trying to get it from
             * unreconciled list should throw an error.
             */
            NEGATIVE_TEST();
            error = LwSciBufAttrListSetAttrs(umd1AttrList.get(),
                        &arrayAttrKey, 1);
            ASSERT_EQ(error, LwSciError_BadParameter);

            error = LwSciBufAttrListGetAttrs(umd1AttrList.get(),
                        &arrayAttrKey, 1);
            ASSERT_EQ(error, LwSciError_BadParameter);
        }
    }

    /* Now, reconcile the list. */
    auto reconciledList =
        LwSciBufPeer::attrListReconcile({umd1AttrList.get()}, &error);
    ASSERT_EQ(error, LwSciError_Success);

    for (auto arrayAttrKey : arrayAttrKeySet) {
        auto it1 = std::find(LwSciBufPeer::inputAttrKeys.begin(),
                            LwSciBufPeer::inputAttrKeys.end(),
                            arrayAttrKey.key);
        if (it1 == LwSciBufPeer::inputAttrKeys.end()) {
            /* Attribute is output only or input/output. We should be able to
             * read it from reconciled list.
             */
            error = LwSciBufAttrListGetAttrs(reconciledList.get(),
                        &arrayAttrKey, 1);
            ASSERT_EQ(error, LwSciError_Success);
        } else {
            /* Attribute is input only attribute. Trying to read it from
             * reconciled list should throw an error.
             */
            NEGATIVE_TEST();
            error = LwSciBufAttrListGetAttrs(reconciledList.get(),
                        &arrayAttrKey, 1);
            ASSERT_EQ(error, LwSciError_BadParameter);
        }
    }
}

/*
 * This test verifies the reconciliation validation functionality for array
 * attributes.
 */
TEST_F(TestLwSciBufArray, Reconciliatiolwalidation)
{
    LwSciError error = LwSciError_Success;
    LwSciBufType bufType = LwSciBufType_Array;
    LwSciBufAttrValDataType dataType = LwSciDataType_Uint32;
    uint64_t stride = 64U;
    uint64_t capacity = 100U;

    SET_ATTR(umd1AttrList.get(), LwSciBufGeneralAttrKey_Types, bufType);
    SET_ATTR(umd1AttrList.get(), LwSciBufArrayAttrKey_DataType, dataType);
    SET_ATTR(umd1AttrList.get(), LwSciBufArrayAttrKey_Stride, stride);
    SET_ATTR(umd1AttrList.get(), LwSciBufArrayAttrKey_Capacity, capacity);

    auto reconciledList =
        LwSciBufPeer::attrListReconcile({umd1AttrList.get()}, &error);
    ASSERT_EQ(error, LwSciError_Success);

    {
        /* Setup another unreconciled list to be verified against reconciled
         * list.
         */
        bool isReconciledListValid = false;

        /* LwSciBufArrayAttrKey_DataType uses equal value validation policy
         * Thus set the value in unreconciled list which is not equal to that
         * set in umd1AttrList and verify that validation fails.
         */
        LwSciBufAttrValDataType validateDataType = LwSciDataType_Int4;
        /* LwSciBufArrayAttrKey_Stride uses equal value validation policy
         * Thus set the value in unreconciled list which is not equal to that
         * set in umd1AttrList and verify that validation fails.
         */
        uint64_t validateStride = 32U;
        /* LwSciBufArrayAttrKey_Capacity uses equal value validation policy
         * Thus set the value in unreconciled list which is not equal to that
         * set in umd1AttrList and verify that validation fails.
         */
        uint64_t validateCapacity = 200U;
        std::vector<LwSciBufAttrKeyValuePair> keyValPair = {
            {
                .key = LwSciBufArrayAttrKey_DataType,
                .value = &validateDataType,
                .len = sizeof(validateDataType),
            },

            {
                .key = LwSciBufArrayAttrKey_Stride,
                .value = &validateStride,
                .len = sizeof(validateStride),
            },

            {
                .key = LwSciBufArrayAttrKey_Capacity,
                .value = &validateCapacity,
                .len = sizeof(validateCapacity),
            }
        };

        for (auto keyVal : keyValPair) {
            auto validateList = peer.createAttrList(&error);
            ASSERT_EQ(error, LwSciError_Success);

            SET_ATTR(validateList.get(), LwSciBufGeneralAttrKey_Types, bufType);

            error = LwSciBufAttrListSetAttrs(validateList.get(), &keyVal, 1);
            ASSERT_EQ(error, LwSciError_Success);

            NEGATIVE_TEST();

            ASSERT_EQ(LwSciBufPeer::validateReconciled({validateList.get()},
                reconciledList.get(), &isReconciledListValid),
                LwSciError_ReconciliationFailed);

            ASSERT_FALSE(isReconciledListValid);
        }
    }
}
