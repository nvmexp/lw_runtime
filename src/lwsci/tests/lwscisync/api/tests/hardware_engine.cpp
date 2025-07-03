/*
 * Copyright (c) 2021-2022, LWPU CORPORATION. All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */
#include "lwscisync_basic_test.h"

#include "lwscisync_internal.h"
#include <utility>

class LwSciSyncHardwareEngineTest : public LwSciSyncBasicTest
{
};

class TestLwSciSyncHwEngCreateIdWithoutInstance
    : public LwSciSyncBasicTest,
      public ::testing::WithParamInterface<
          std::tuple<LwSciSyncHwEngName, LwSciError, int64_t>>
{
};

TEST_P(TestLwSciSyncHwEngCreateIdWithoutInstance, EngineIds)
{
    auto params = GetParam();
    LwSciSyncHwEngName engineName = std::get<0>(params);
    LwSciError expectedError = std::get<1>(params);
    int64_t expectedEngId = std::get<2>(params);

    int64_t engId = 0;
    LwSciError err = LwSciError_Success;
    if (err != LwSciError_Success) {
        NegativeTestPrint();
        err = LwSciSyncHwEngCreateIdWithoutInstance(engineName, &engId);
    } else {
        err = LwSciSyncHwEngCreateIdWithoutInstance(engineName, &engId);
    }
    ASSERT_EQ(err, expectedError);

    // Output parameters are only valid if successful
    if (expectedError == LwSciError_Success) {
        ASSERT_EQ(engId, expectedEngId);
    }
}
INSTANTIATE_TEST_CASE_P(
    TestLwSciSyncHwEngCreateIdWithoutInstance,
    TestLwSciSyncHwEngCreateIdWithoutInstance,
    ::testing::Values(std::make_tuple(LwSciSyncHwEngName_LowerBound,
                                      LwSciError_BadParameter, (int64_t)0),
                      std::make_tuple(LwSciSyncHwEngName_UpperBound,
                                      LwSciError_BadParameter, (int64_t)124),
                      std::make_tuple(LwSciSyncHwEngName_PCIe,
                                      LwSciError_Success, (int64_t)123)));

TEST_F(LwSciSyncHardwareEngineTest,
       TestLwSciSyncHwEngCreateIdWithoutInstanceNull)
{
    NEGATIVE_TEST();
    LwSciError err = LwSciSyncHwEngCreateIdWithoutInstance(
        LwSciSyncHwEngName_PCIe, nullptr);
    ASSERT_EQ(err, LwSciError_BadParameter);
}

class TestLwSciSyncHwEngGetNameFromId
    : public LwSciSyncBasicTest,
      public ::testing::WithParamInterface<
          std::tuple<int64_t, LwSciError, LwSciSyncHwEngName>>
{
};
TEST_P(TestLwSciSyncHwEngGetNameFromId, EngineNames)
{
    auto params = GetParam();
    int64_t engId = std::get<0>(params);
    LwSciError expectedError = std::get<1>(params);
    LwSciSyncHwEngName expectedEngineName = std::get<2>(params);

    LwSciSyncHwEngName engineName = LwSciSyncHwEngName_LowerBound;
    LwSciError err = LwSciSyncHwEngGetNameFromId(engId, &engineName);
    ASSERT_EQ(err, expectedError);

    // Output parameters are only valid if successful
    if (expectedError == LwSciError_Success) {
        ASSERT_EQ(engineName, expectedEngineName);
    }
}
INSTANTIATE_TEST_CASE_P(
    TestLwSciSyncHwEngGetNameFromId, TestLwSciSyncHwEngGetNameFromId,
    ::testing::Values(std::make_tuple(
                          // Valid engine names
                          (int64_t)123, LwSciError_Success,
                          LwSciSyncHwEngName_PCIe),
                      std::make_tuple((int64_t)65659, LwSciError_Success,
                                      LwSciSyncHwEngName_PCIe),
                      // Invalid engine names
                      std::make_tuple((int64_t)65536, LwSciError_BadParameter,
                                      LwSciSyncHwEngName_LowerBound),
                      std::make_tuple((int64_t)65538, LwSciError_BadParameter,
                                      LwSciSyncHwEngName_LowerBound)));

TEST_F(LwSciSyncHardwareEngineTest, TestLwSciSyncHwEngGetNameFromIdNull)
{
    NegativeTestPrint();
    LwSciError err = LwSciSyncHwEngGetNameFromId((int64_t)1, nullptr);
    ASSERT_EQ(err, LwSciError_BadParameter);
}

class TestLwSciSyncHwEngGetInstanceFromId
    : public LwSciSyncBasicTest,
      public ::testing::WithParamInterface<
          std::tuple<int64_t, LwSciError, uint32_t>>
{
};
TEST_P(TestLwSciSyncHwEngGetInstanceFromId, EngineIds)
{
    auto params = GetParam();
    int64_t engId = std::get<0>(params);
    LwSciError expectedError = std::get<1>(params);
    uint32_t expectedEngineInstance = std::get<2>(params);

    uint32_t engineInstance = 0U;
    LwSciError err = LwSciError_Success;
    if (expectedError != LwSciError_Success) {
        NegativeTestPrint();
        err = LwSciSyncHwEngGetInstanceFromId(engId, &engineInstance);
    } else {
        err = LwSciSyncHwEngGetInstanceFromId(engId, &engineInstance);
    }
    ASSERT_EQ(err, expectedError);

    if (expectedError == LwSciError_Success) {
        ASSERT_EQ(engineInstance, expectedEngineInstance);
    }
}
INSTANTIATE_TEST_CASE_P(
    TestLwSciSyncHwEngGetInstanceFromId, TestLwSciSyncHwEngGetInstanceFromId,
    ::testing::Values(
        // Valid engine IDs
        std::make_tuple((int64_t)123, LwSciError_Success, (uint32_t)0U),
        std::make_tuple((int64_t)65659, LwSciError_Success, (uint32_t)1U),
        // Invalid engine names
        std::make_tuple((int64_t)65536, LwSciError_BadParameter, (uint32_t)0U),
        std::make_tuple((int64_t)65538, LwSciError_BadParameter, (uint32_t)0U),
        // Extra bits
        std::make_tuple(int64_t(8590000129LL), LwSciError_BadParameter,
                        (uint32_t)0U)));

TEST_F(LwSciSyncHardwareEngineTest, TestLwSciSyncHwEngGetInstanceFromIdNull)
{
    NegativeTestPrint();
    LwSciError err = LwSciSyncHwEngGetInstanceFromId((int64_t)1, nullptr);
    ASSERT_EQ(err, LwSciError_BadParameter);
}
