/*
 * Copyright (c) 2019, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 */

#ifndef LWTENSOR_TEST_PUBLICAPIMASTER_H
#define LWTENSOR_TEST_PUBLICAPIMASTER_H
/**
 * @file
 * @brief This file defines the public API tests of lwTensor library.
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <memory>
#include <ostream>
#include <limits>
#include <cstdint>
#include "apiTest.h"
#include "gtest/gtest.h"
#include "lwtensor.h"
#include "lwtensor/types.h"

namespace APITESTING
{
    // Note: the following APIs have been tested in other places, such as apiMaster.h
    // lwtensorInitTensorDescriptor
    // lwtensorElementwiseTrinary
    // lwtensorElementwiseBinary
    // lwtensorPermutation

    // normal test case
    TEST_F(PublicApiTestDefault, lwtensorInitContractionDescriptor_0)
    {
        lwtensorContractionDescriptor_t desc;
        EXPECT_EQ(lwtensorInitContractionDescriptor(&handle,
                &desc,
                &descA, opts.modeA.data(), alignmentRequirementA,
                &descB, opts.modeB.data(), alignmentRequirementB,
                &descC, opts.modeC.data(), alignmentRequirementC,
                &descC, opts.modeC.data(), alignmentRequirementC,
                LWTENSOR_R_MIN_32F), LWTENSOR_STATUS_SUCCESS);

        lwtensorContractionFind_t find;
        lwtensorInitContractionFind(&handle, &find, LWTENSOR_ALGO_DEFAULT);
        size_t worksize = 0;
        lwtensorContractionGetWorkspace(&handle, &desc, &find, LWTENSOR_WORKSPACE_RECOMMENDED, &worksize);
        void *work = nullptr;
        if(worksize > 0)
        {
            if(lwdaSuccess != lwdaMalloc(&work, worksize))
            {
                work = nullptr;
                worksize = 0;
            }
        }
        lwtensorContractionPlan_t plan;
        lwtensorInitContractionPlan(&handle, &plan, &desc, &find, worksize);
        EXPECT_EQ(lwtensorContraction(&handle, &plan, opts.alpha, A_d, B_d,
                opts.beta, C_d, C_d, work, worksize, pStream), LWTENSOR_STATUS_SUCCESS);

        if(work) lwdaFree(work);
    }

    // uninitialized handle
    TEST_F(PublicApiTestDefault, lwtensorInitContractionDescriptor_1)
    {
        lwtensorContractionDescriptor_t desc;
        lwtensorHandle_t handle;
        memset(&handle, 0, sizeof(lwtensorHandle_t));
        EXPECT_EQ(lwtensorInitContractionDescriptor(&handle,
                &desc,
                &descA, opts.modeA.data(), alignmentRequirementA,
                &descB, opts.modeB.data(), alignmentRequirementB,
                &descC, opts.modeC.data(), alignmentRequirementC,
                &descC, opts.modeC.data(), alignmentRequirementC,
                LWTENSOR_R_MIN_32F), LWTENSOR_STATUS_NOT_INITIALIZED);
    }

    // null Contraction descriptor
    TEST_F(PublicApiTestDefault, lwtensorInitContractionDescriptor_2)
    {
        EXPECT_EQ(lwtensorInitContractionDescriptor(&handle,
                nullptr,
                &descA, opts.modeA.data(), alignmentRequirementA,
                &descB, opts.modeB.data(), alignmentRequirementB,
                &descC, opts.modeC.data(), alignmentRequirementC,
                &descC, opts.modeC.data(), alignmentRequirementC,
                LWTENSOR_R_MIN_32F), LWTENSOR_STATUS_ILWALID_VALUE);
    }

    // null Tensor descriptors
    TEST_F(PublicApiTestDefault, lwtensorInitContractionDescriptor_3)
    {
        lwtensorContractionDescriptor_t desc;
        EXPECT_EQ(lwtensorInitContractionDescriptor(&handle,
                &desc,
                nullptr, opts.modeA.data(), alignmentRequirementA,
                &descB, opts.modeB.data(), alignmentRequirementB,
                &descC, opts.modeC.data(), alignmentRequirementC,
                &descC, opts.modeC.data(), alignmentRequirementC,
                LWTENSOR_R_MIN_32F), LWTENSOR_STATUS_ILWALID_VALUE);
    }

    // null mode data
    TEST_F(PublicApiTestDefault, lwtensorInitContractionDescriptor_4)
    {
        lwtensorContractionDescriptor_t desc;
        EXPECT_EQ(lwtensorInitContractionDescriptor(&handle,
                &desc,
                &descA, nullptr, alignmentRequirementA,
                &descB, opts.modeB.data(), alignmentRequirementB,
                &descC, opts.modeC.data(), alignmentRequirementC,
                &descC, opts.modeC.data(), alignmentRequirementC,
                LWTENSOR_R_MIN_32F), LWTENSOR_STATUS_ILWALID_VALUE);
    }

    // invalid alignmentRequirementA
    TEST_F(PublicApiTestDefault, lwtensorInitContractionDescriptor_5)
    {
        lwtensorContractionDescriptor_t desc;
        EXPECT_EQ(lwtensorInitContractionDescriptor(&handle,
                &desc,
                &descA, opts.modeA.data(), 33U,
                &descB, opts.modeB.data(), alignmentRequirementB,
                &descC, opts.modeC.data(), alignmentRequirementC,
                &descC, opts.modeC.data(), alignmentRequirementC,
                LWTENSOR_R_MIN_32F), LWTENSOR_STATUS_ILWALID_VALUE);
    }

    // normal test case
    TEST(PUBLIC_APIS, lwtensorInit_0)
    {
        lwtensorHandle_t handle;
        EXPECT_EQ(lwtensorInit(&handle), LWTENSOR_STATUS_SUCCESS);
    }

    // null handle
    TEST(PUBLIC_APIS, lwtensorInit_1)
    {
        EXPECT_EQ(lwtensorInit(nullptr), LWTENSOR_STATUS_ILWALID_VALUE);
    }

    // normal test case
    TEST(PUBLIC_APIS, lwtensorInitContractionFind_0)
    {
        lwtensorHandle_t handle;
        lwtensorInit(&handle);
        lwtensorContractionFind_t find;
        EXPECT_EQ(lwtensorInitContractionFind(&handle, &find, LWTENSOR_ALGO_DEFAULT), LWTENSOR_STATUS_SUCCESS);
    }

    // null handle
    TEST(PUBLIC_APIS, lwtensorInitContractionFind_1)
    {
        lwtensorContractionFind_t find;
        EXPECT_EQ(lwtensorInitContractionFind(nullptr, &find, LWTENSOR_ALGO_DEFAULT), LWTENSOR_STATUS_NOT_INITIALIZED);
    }

    // null find
    TEST(PUBLIC_APIS, lwtensorInitContractionFind_2)
    {
        lwtensorHandle_t handle;
        lwtensorInit(&handle);
        EXPECT_EQ(lwtensorInitContractionFind(&handle, nullptr, LWTENSOR_ALGO_DEFAULT), LWTENSOR_STATUS_ILWALID_VALUE);
    }

    // invalid algo, need to  check whether the given algo is correct
    // Commentet because invalid algos are detected and handled during planning, the find does not validate it
    // TEST(PUBLIC_APIS, lwtensorInitContractionFind_3)
    // {
    //     lwtensorHandle_t handle;
    //     lwtensorInit(&handle);
    //     lwtensorContractionFind_t find;
    //     EXPECT_EQ(lwtensorInitContractionFind(&handle, &find, lwtensorAlgo_t(70)), LWTENSOR_STATUS_NOT_SUPPORTED);
    // }

    // normal test case is included in lwtensorContractionGetWorkspace_0
    // null handle
    TEST_F(PublicApiTestDefault, lwtensorContractionGetWorkspace_0)
    {
        lwtensorContractionDescriptor_t desc;
        EXPECT_EQ(lwtensorInitContractionDescriptor(&handle,
                &desc,
                &descA, opts.modeA.data(), alignmentRequirementA,
                &descB, opts.modeB.data(), alignmentRequirementB,
                &descC, opts.modeC.data(), alignmentRequirementC,
                &descC, opts.modeC.data(), alignmentRequirementC,
                LWTENSOR_R_MIN_32F), LWTENSOR_STATUS_SUCCESS);

        lwtensorContractionFind_t find;
        lwtensorInitContractionFind(&handle, &find, LWTENSOR_ALGO_DEFAULT);
        size_t worksize = 0;
        EXPECT_EQ(lwtensorContractionGetWorkspace(nullptr, &desc, &find, LWTENSOR_WORKSPACE_RECOMMENDED, &worksize), LWTENSOR_STATUS_NOT_INITIALIZED);
    }

    // null desc
    TEST_F(PublicApiTestDefault, lwtensorContractionGetWorkspace_1)
    {
        lwtensorContractionDescriptor_t desc;
        EXPECT_EQ(lwtensorInitContractionDescriptor(&handle,
                &desc,
                &descA, opts.modeA.data(), alignmentRequirementA,
                &descB, opts.modeB.data(), alignmentRequirementB,
                &descC, opts.modeC.data(), alignmentRequirementC,
                &descC, opts.modeC.data(), alignmentRequirementC,
                LWTENSOR_R_MIN_32F), LWTENSOR_STATUS_SUCCESS);

        lwtensorContractionFind_t find;
        lwtensorInitContractionFind(&handle, &find, LWTENSOR_ALGO_DEFAULT);
        size_t worksize = 0;
        EXPECT_EQ(lwtensorContractionGetWorkspace(&handle, nullptr, &find, LWTENSOR_WORKSPACE_RECOMMENDED, &worksize), LWTENSOR_STATUS_ILWALID_VALUE);
    }

    // null find
    TEST_F(PublicApiTestDefault, lwtensorContractionGetWorkspace_2)
    {
        lwtensorContractionDescriptor_t desc;
        EXPECT_EQ(lwtensorInitContractionDescriptor(&handle,
                &desc,
                &descA, opts.modeA.data(), alignmentRequirementA,
                &descB, opts.modeB.data(), alignmentRequirementB,
                &descC, opts.modeC.data(), alignmentRequirementC,
                &descC, opts.modeC.data(), alignmentRequirementC,
                LWTENSOR_R_MIN_32F), LWTENSOR_STATUS_SUCCESS);

        lwtensorContractionFind_t find;
        lwtensorInitContractionFind(&handle, &find, LWTENSOR_ALGO_DEFAULT);
        size_t worksize = 0;
        EXPECT_EQ(lwtensorContractionGetWorkspace(&handle, &desc, nullptr, LWTENSOR_WORKSPACE_RECOMMENDED, &worksize), LWTENSOR_STATUS_ILWALID_VALUE);
    }

    // null workspace
    TEST_F(PublicApiTestDefault, lwtensorContractionGetWorkspace_3)
    {
        lwtensorContractionDescriptor_t desc;
        EXPECT_EQ(lwtensorInitContractionDescriptor(&handle,
                &desc,
                &descA, opts.modeA.data(), alignmentRequirementA,
                &descB, opts.modeB.data(), alignmentRequirementB,
                &descC, opts.modeC.data(), alignmentRequirementC,
                &descC, opts.modeC.data(), alignmentRequirementC,
                LWTENSOR_R_MIN_32F), LWTENSOR_STATUS_SUCCESS);

        lwtensorContractionFind_t find;
        lwtensorInitContractionFind(&handle, &find, LWTENSOR_ALGO_DEFAULT);
        EXPECT_EQ(lwtensorContractionGetWorkspace(&handle, &desc, &find, LWTENSOR_WORKSPACE_RECOMMENDED, nullptr), LWTENSOR_STATUS_ILWALID_VALUE);
    }

    // invalid algo
    TEST_F(PublicApiTestDefault, lwtensorContractionGetWorkspace_4)
    {
        lwtensorContractionDescriptor_t desc;
        EXPECT_EQ(lwtensorInitContractionDescriptor(&handle,
                &desc,
                &descA, opts.modeA.data(), alignmentRequirementA,
                &descB, opts.modeB.data(), alignmentRequirementB,
                &descC, opts.modeC.data(), alignmentRequirementC,
                &descC, opts.modeC.data(), alignmentRequirementC,
                LWTENSOR_R_MIN_32F), LWTENSOR_STATUS_SUCCESS);

        lwtensorContractionFind_t find;
        lwtensorInitContractionFind(&handle, &find, lwtensorAlgo_t(-100));
        size_t worksize = 0;
        EXPECT_EQ(lwtensorContractionGetWorkspace(&handle, &desc, &find, LWTENSOR_WORKSPACE_RECOMMENDED, &worksize), LWTENSOR_STATUS_ILWALID_VALUE);
    }

    // null handle, null desc, null find, and null plan
    TEST_F(PublicApiTestDefault, lwtensorInitContractionPlan_0)
    {
        lwtensorContractionDescriptor_t desc;
        EXPECT_EQ(lwtensorInitContractionDescriptor(&handle,
                &desc,
                &descA, opts.modeA.data(), alignmentRequirementA,
                &descB, opts.modeB.data(), alignmentRequirementB,
                &descC, opts.modeC.data(), alignmentRequirementC,
                &descC, opts.modeC.data(), alignmentRequirementC,
                LWTENSOR_R_MIN_32F), LWTENSOR_STATUS_SUCCESS);

        lwtensorContractionFind_t find;
        lwtensorInitContractionFind(&handle, &find, LWTENSOR_ALGO_DEFAULT);
        size_t worksize = 0;
        lwtensorContractionGetWorkspace(&handle, &desc, &find, LWTENSOR_WORKSPACE_RECOMMENDED, &worksize);
        lwtensorContractionPlan_t plan;
        EXPECT_EQ(lwtensorInitContractionPlan(nullptr, &plan, &desc, &find, worksize), LWTENSOR_STATUS_NOT_INITIALIZED);
        EXPECT_EQ(lwtensorInitContractionPlan(&handle, nullptr, &desc, &find, worksize), LWTENSOR_STATUS_ILWALID_VALUE);
        EXPECT_EQ(lwtensorInitContractionPlan(&handle, &plan, nullptr, &find, worksize), LWTENSOR_STATUS_ILWALID_VALUE);
        EXPECT_EQ(lwtensorInitContractionPlan(&handle, &plan, &desc, nullptr, worksize), LWTENSOR_STATUS_ILWALID_VALUE);
    }

    //  invalid handle, invalid plan
    TEST_F(PublicApiTestDefault, lwtensorContraction_0)
    {
        lwtensorContractionDescriptor_t desc;
        EXPECT_EQ(lwtensorInitContractionDescriptor(&handle,
                &desc,
                &descA, opts.modeA.data(), alignmentRequirementA,
                &descB, opts.modeB.data(), alignmentRequirementB,
                &descC, opts.modeC.data(), alignmentRequirementC,
                &descC, opts.modeC.data(), alignmentRequirementC,
                LWTENSOR_R_MIN_32F), LWTENSOR_STATUS_SUCCESS);

        lwtensorContractionFind_t find;
        lwtensorInitContractionFind(&handle, &find, LWTENSOR_ALGO_DEFAULT);
        size_t worksize = 0;
        lwtensorContractionGetWorkspace(&handle, &desc, &find, LWTENSOR_WORKSPACE_RECOMMENDED, &worksize);
        void *work = nullptr;
        if(worksize > 0)
        {
            if(lwdaSuccess != lwdaMalloc(&work, worksize))
            {
                work = nullptr;
                worksize = 0;
            }
        }
        lwtensorContractionPlan_t plan;
        lwtensorInitContractionPlan(&handle, &plan, &desc, &find, worksize);
        EXPECT_EQ(lwtensorContraction(nullptr, &plan, opts.alpha, A_d, B_d,
                opts.beta, C_d, C_d, work, worksize, pStream), LWTENSOR_STATUS_NOT_INITIALIZED);
        EXPECT_EQ(lwtensorContraction(&handle, nullptr, opts.alpha, A_d, B_d,
                opts.beta, C_d, C_d, work, worksize, pStream), LWTENSOR_STATUS_ILWALID_VALUE);

        if(work) lwdaFree(work);
    }

    //  invalid alpha, beta, A, B, C, D
    TEST_F(PublicApiTestDefault, lwtensorContraction_1)
    {
        lwtensorContractionDescriptor_t desc;
        EXPECT_EQ(lwtensorInitContractionDescriptor(&handle,
                &desc,
                &descA, opts.modeA.data(), alignmentRequirementA,
                &descB, opts.modeB.data(), alignmentRequirementB,
                &descC, opts.modeC.data(), alignmentRequirementC,
                &descC, opts.modeC.data(), alignmentRequirementC,
                LWTENSOR_R_MIN_32F), LWTENSOR_STATUS_SUCCESS);

        lwtensorContractionFind_t find;
        lwtensorInitContractionFind(&handle, &find, LWTENSOR_ALGO_DEFAULT);
        size_t worksize = 0;
        lwtensorContractionGetWorkspace(&handle, &desc, &find, LWTENSOR_WORKSPACE_RECOMMENDED, &worksize);
        void *work = nullptr;
        if(worksize > 0)
        {
            if(lwdaSuccess != lwdaMalloc(&work, worksize))
            {
                work = nullptr;
                worksize = 0;
            }
        }
        lwtensorContractionPlan_t plan;
        lwtensorInitContractionPlan(&handle, &plan, &desc, &find, worksize);
        EXPECT_EQ(lwtensorContraction(&handle, &plan, nullptr, A_d, B_d,
                opts.beta, C_d, C_d, work, worksize, pStream), LWTENSOR_STATUS_ILWALID_VALUE);
        EXPECT_EQ(lwtensorContraction(&handle, &plan, opts.alpha, A_d, B_d,
                nullptr, C_d, C_d, work, worksize, pStream), LWTENSOR_STATUS_ILWALID_VALUE);
        EXPECT_EQ(lwtensorContraction(&handle, &plan, opts.alpha, nullptr, B_d,
                opts.beta, C_d, C_d, work, worksize, pStream), LWTENSOR_STATUS_ILWALID_VALUE);
        EXPECT_EQ(lwtensorContraction(&handle, &plan, opts.alpha, A_d, nullptr,
                opts.beta, C_d, C_d, work, worksize, pStream), LWTENSOR_STATUS_ILWALID_VALUE);
        EXPECT_EQ(lwtensorContraction(&handle, &plan, opts.alpha, A_d, B_d,
                opts.beta, nullptr, C_d, work, worksize, pStream), LWTENSOR_STATUS_ILWALID_VALUE);
        EXPECT_EQ(lwtensorContraction(&handle, &plan, opts.alpha, A_d, B_d,
                opts.beta, C_d, nullptr, work, worksize, pStream), LWTENSOR_STATUS_ILWALID_VALUE);

        if(work) lwdaFree(work);
    }

    //  invalid worksize and workspace
    TEST_F(PublicApiTestDefault, lwtensorContraction_2)
    {
        lwtensorContractionDescriptor_t desc;
        EXPECT_EQ(lwtensorInitContractionDescriptor(&handle,
                &desc,
                &descA, opts.modeA.data(), alignmentRequirementA,
                &descB, opts.modeB.data(), alignmentRequirementB,
                &descC, opts.modeC.data(), alignmentRequirementC,
                &descC, opts.modeC.data(), alignmentRequirementC,
                LWTENSOR_R_MIN_32F), LWTENSOR_STATUS_SUCCESS);

        lwtensorContractionFind_t find;
        lwtensorInitContractionFind(&handle, &find, LWTENSOR_ALGO_DEFAULT);
        size_t worksize = 0;
        lwtensorContractionGetWorkspace(&handle, &desc, &find, LWTENSOR_WORKSPACE_RECOMMENDED, &worksize);
        printf("worksize  = %d\n", (int)worksize);
        lwtensorContractionPlan_t plan;
        lwtensorInitContractionPlan(&handle, &plan, &desc, &find, worksize);
        EXPECT_EQ(lwtensorContraction(&handle, &plan, opts.alpha, A_d, B_d,
                opts.beta, C_d, C_d, nullptr, worksize, pStream), LWTENSOR_STATUS_INSUFFICIENT_WORKSPACE);
    }

    // normal case
    TEST(PUBLIC_APIS, lwtensorContractionMaxAlgos_0)
    {
        int32_t maxNumAlgos;
        EXPECT_EQ(lwtensorContractionMaxAlgos(&maxNumAlgos), LWTENSOR_STATUS_SUCCESS);
    }

    // null input, TODO: should add defence for nullptr
    TEST(PUBLIC_APIS, lwtensorContractionMaxAlgos_1)
    {
        EXPECT_EQ(lwtensorContractionMaxAlgos(nullptr), LWTENSOR_STATUS_ILWALID_VALUE);
    }

    TEST(PUBLIC_APIS, lwtensorHandleWriteCacheToFile_0)
    {
        lwtensorHandle_t handle;
        EXPECT_EQ(lwtensorInit(&handle), LWTENSOR_STATUS_SUCCESS);
        // no cache attached
        EXPECT_EQ(lwtensorHandleWriteCacheToFile(&handle,""), LWTENSOR_STATUS_ILWALID_VALUE);
    }

    TEST(PUBLIC_APIS, lwtensorHandleWriteCacheToFile_1)
    {
        lwtensorHandle_t handle;
        EXPECT_EQ(lwtensorInit(&handle), LWTENSOR_STATUS_SUCCESS);

        constexpr int32_t numCachelines = 1024;
        std::unique_ptr<lwtensorPlanCacheline_t[]> cachelines(new lwtensorPlanCacheline_t[numCachelines]);
        EXPECT_EQ( lwtensorHandleAttachPlanCachelines(&handle, cachelines.get(), numCachelines), LWTENSOR_STATUS_SUCCESS);
        EXPECT_EQ(lwtensorHandleWriteCacheToFile(&handle,""), LWTENSOR_STATUS_IO_ERROR);
    }

} //namespace

#endif /* define LWTENSOR_TEST_PUBLICAPIMASTER_H */

