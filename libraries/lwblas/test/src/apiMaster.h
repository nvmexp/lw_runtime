/*
 * Copyright (c) 2019, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 */

#ifndef LWTENSOR_TEST_APIMASTER_H
#define LWTENSOR_TEST_APIMASTER_H
/**
 * @file
 * @brief This file defines the API tests of lwTensor library.
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <ostream>
#include <limits>
#include <cstdint>
#include "apiTest.h"
#include "gtest/gtest.h"
extern "C"
{
#include "lwtensor.h"
#include "lwtensor/types.h"
}
#include "lwtensor/internal/lwtensor.h"
#include "lwtensor/internal/types.h"
#include "lwtensor/internal/util.h"
#include "lwtensor/internal/operatorsPLC3.h"
#include "lwtensor/internal/defines.h"
#include "lwtensor/internal/exceptions.h"
#include "lwtensor/internal/elementwise.h"
#include "lwtensor/internal/elementwisePrototype.h"

namespace APITESTING
{
    using ::testing::TestWithParam;
    using ::testing::Bool;
    using ::testing::Values;
    using ::testing::Combine;

    // DATATYPE COMBINATIONS

    //Supported dataType combinations, cccc, dddd, ddss, hhhh, hhss, ssdd, sshh, ssss, zzzz
    std::vector<lwdaDataType_t> type1 = {LWDA_R_16F, LWDA_R_16F, LWDA_R_16F, LWDA_R_16F};
    std::vector<lwdaDataType_t> type2 = {LWDA_R_32F, LWDA_R_32F, LWDA_R_32F, LWDA_R_32F};
    std::vector<lwdaDataType_t> type3 = {LWDA_R_64F, LWDA_R_64F, LWDA_R_64F, LWDA_R_64F};
    std::vector<lwdaDataType_t> type4 = {LWDA_C_32F, LWDA_C_32F, LWDA_C_32F, LWDA_C_32F};
    std::vector<lwdaDataType_t> type5 = {LWDA_C_64F, LWDA_C_64F, LWDA_C_64F, LWDA_C_64F};
    std::vector<lwdaDataType_t> type6 = {LWDA_R_16F, LWDA_R_16F, LWDA_R_16F, LWDA_R_32F};
    std::vector<lwdaDataType_t> type7 = {LWDA_R_32F, LWDA_R_32F, LWDA_R_16F, LWDA_R_32F};
    std::vector<lwdaDataType_t> type8 = {LWDA_R_64F, LWDA_R_64F, LWDA_R_32F, LWDA_R_64F};   
    std::vector<lwdaDataType_t> type9 = {LWDA_C_64F, LWDA_C_64F, LWDA_C_32F, LWDA_C_64F};

    // Unsupported dataType combinations
    std::vector<lwdaDataType_t> type10 = {LWDA_R_64F, LWDA_R_64F, LWDA_R_32F, LWDA_R_32F};
    std::vector<lwdaDataType_t> type11 = {LWDA_R_16F, LWDA_R_16F, LWDA_R_32F, LWDA_R_32F};
    std::vector<lwdaDataType_t> type12 = {LWDA_R_32F, LWDA_R_32F, LWDA_R_64F, LWDA_R_64F};
    std::vector<lwdaDataType_t> type13 = {LWDA_R_32F, LWDA_R_32F, LWDA_R_16F, LWDA_R_16F};
    std::vector<lwdaDataType_t> type14 = {LWDA_R_8I , LWDA_R_32F, LWDA_R_8I , LWDA_R_32F};
    std::vector<lwdaDataType_t> type15 = {LWDA_R_32I, LWDA_R_32I, LWDA_R_32I, LWDA_R_32I};

    std::vector<lwdaDataType_t> ApiTestTypeNegative1 = {LWDA_R_16F, LWDA_R_16F, LWDA_R_16F, LWDA_R_32F}; // Will be colwerted to {16F, 32F, 16F, 32F}, which is lwrrently suported.
    std::vector<lwdaDataType_t> ApiTestTypeNegative2 = {LWDA_R_16F, LWDA_R_16F, LWDA_R_64F, LWDA_R_64F};
    std::vector<lwdaDataType_t> ApiTestTypeNegative3 = {LWDA_R_64F, LWDA_R_64F, LWDA_R_16F, LWDA_R_16F};
    std::vector<lwdaDataType_t> ApiTestTypeNegative4 = {LWDA_R_32F, LWDA_R_64F, LWDA_R_32F, LWDA_R_32F}; //also supported, expected???
    std::vector<lwdaDataType_t> ApiTestTypeNegative5 = {LWDA_C_64F, LWDA_C_32F, LWDA_C_32F, LWDA_C_32F};
    std::vector<lwdaDataType_t> ApiTestTypeNegative6 = {LWDA_C_32F, LWDA_C_64F, LWDA_C_64F, LWDA_C_64F};

    // MODE COMBINATIONS
    std::vector<int> widthMode1 = {1, 1, 1, -1, -1, -1}; //more combinations should be added, but most will fail
    std::vector<int> widthMode2 = {1, 8, 1, -1,  1, -1};
    //std::vector<int> widthMode3 = {2, 1, 1, 0, -1, -1}; //TODO: Relative Error between CPU and GPU is bigger than the threshold
    //std::vector<int> widthMode4 = {4, 1, 1, 0, -1, -1};
    //std::vector<int> widthMode5 = {8, 1, 1, 0, -1, -1};


    TEST_F(ApiTestDefault, lwtensorhandleAttachPlanCachelinesPos) {
        /*
         * Positive tests
         */
        const int32_t numCachelines = 18;
        lwtensorPlanCacheline_t cachelines[numCachelines];
        auto ret = lwtensorHandleAttachPlanCachelines(&handle,
                cachelines,
                numCachelines);
        EXPECT_EQ( ret, LWTENSOR_STATUS_SUCCESS);
    }

    TEST_F(ApiTestDefault, lwtensorhandleAttachPlanCachelinesNeg) {
        const int32_t numCachelines = 18;
        lwtensorPlanCacheline_t cachelines[numCachelines];
        /*
         * Negative tests
         */
        EXPECT_EQ( lwtensorHandleAttachPlanCachelines(&handle,
                nullptr,
                numCachelines), LWTENSOR_STATUS_ILWALID_VALUE);

        EXPECT_EQ( lwtensorHandleAttachPlanCachelines(&handle,
                cachelines,
                0), LWTENSOR_STATUS_ILWALID_VALUE);

        EXPECT_EQ( lwtensorHandleAttachPlanCachelines(nullptr,
                cachelines,
                numCachelines), LWTENSOR_STATUS_ILWALID_VALUE);
    }

    TEST_F(ApiTestDefault, lwtensorhandleDetachPlanCachelinesPos)
    {
        /*
         * Positive tests
         */
        const int32_t numCachelines = 18;
        lwtensorPlanCacheline_t cachelines[numCachelines];
        EXPECT_EQ( lwtensorHandleAttachPlanCachelines(&handle,
                cachelines,
                numCachelines), LWTENSOR_STATUS_SUCCESS);
        EXPECT_EQ(lwtensorHandleDetachPlanCachelines(&handle), LWTENSOR_STATUS_SUCCESS);
    }

    TEST_F(ApiTestDefault, lwtensorhandleDetachPlanCachelinesNeg)
    {
        /*
         * Negative tests
         */
        EXPECT_EQ(lwtensorHandleDetachPlanCachelines(&handle), LWTENSOR_STATUS_NOT_SUPPORTED);
        EXPECT_EQ(lwtensorHandleDetachPlanCachelines(nullptr), LWTENSOR_STATUS_ILWALID_VALUE);
    }

    TEST_F(publicApiTestContraction, lwtensorContractionFindSetAttributePos) {
        /*
         * Positive tests
         */
        lwtensorAutotuneMode_t autotuneMode = LWTENSOR_AUTOTUNE_INCREMENTAL;
        EXPECT_EQ( lwtensorContractionFindSetAttribute(
                &handle,
                &find,
                LWTENSOR_CONTRACTION_FIND_AUTOTUNE_MODE,
                &autotuneMode,
                sizeof(autotuneMode)), LWTENSOR_STATUS_SUCCESS);
    }

    TEST_F(publicApiTestContraction, lwtensorContractionFindSetAttributeNeg) {
        /*
         * Negative tests
         */
        lwtensorAutotuneMode_t autotuneMode = LWTENSOR_AUTOTUNE_INCREMENTAL;
        EXPECT_EQ( lwtensorContractionFindSetAttribute(
                nullptr,
                &find,
                LWTENSOR_CONTRACTION_FIND_AUTOTUNE_MODE,
                &autotuneMode,
                sizeof(autotuneMode)), LWTENSOR_STATUS_ILWALID_VALUE);
        EXPECT_EQ( lwtensorContractionFindSetAttribute(
                &handle,
                nullptr,
                LWTENSOR_CONTRACTION_FIND_AUTOTUNE_MODE,
                &autotuneMode,
                sizeof(autotuneMode)), LWTENSOR_STATUS_ILWALID_VALUE);
        EXPECT_EQ( lwtensorContractionFindSetAttribute(
                &handle,
                &find,
                LWTENSOR_CONTRACTION_FIND_AUTOTUNE_MODE,
                nullptr,
                sizeof(autotuneMode)), LWTENSOR_STATUS_ILWALID_VALUE);
        EXPECT_EQ( lwtensorContractionFindSetAttribute(
                &handle,
                &find,
                LWTENSOR_CONTRACTION_FIND_AUTOTUNE_MODE,
                &autotuneMode,
                sizeof(autotuneMode)-1), LWTENSOR_STATUS_ILWALID_VALUE);
        int dummy = 123014;
        EXPECT_EQ( lwtensorContractionFindSetAttribute(
                &handle,
                &find,
                LWTENSOR_CONTRACTION_FIND_AUTOTUNE_MODE,
                (void*)&dummy, // invalid autotune mode
                sizeof(autotuneMode)-1), LWTENSOR_STATUS_ILWALID_VALUE);
    }

    /**
     * \id lwtensorElementwiseDefault
     * \brief the functionality of API "lwtensorElementwiseTrinary" and check the correctness by comparing the results with that of CPU
     * \depends the required inputs should be defined in class ApiTestDefault
     * \setup parameters and context initialization, memory allocation, et al. (implemented in class ApiTestDefault)
     * \testprocedure ./apiTest --gtest_filter=ApiTestDefault.lwtensorElementwiseDefault
     * \teardown context destroy, memory free, et al. (implemented in class ApiTestDefault)
     * \designid LWTENSOR_DES_001, LWTENSOR_DES_002, LWTENSOR_DES_004, LWTENSOR_DES_005, LWTENSOR_DES_007, LWTENSOR_DES_008, LWTENSOR_DES_009
     * \testgroup ApiTestDefault
     * \inputs the parameters of lwtensorElementwiseTrinary, defined in class ApiTestDefault
     * \outputs None
     * \expected States: no failure of API lwtensorElementwiseTrinary, same reult with CPU
     */
    TEST_F(ApiTestDefault, lwtensorElementwiseDefault) {
        //lwTensor computation
        callingInfo("lwtensorElementwiseTrinary");
        EXPECT_EQ(lwtensorElementwiseTrinary(&handle,
                    opts.alpha, A_d, &descA, &opts.modeA[0],
                    opts.beta,  B_d, &descB, &opts.modeB[0],
                    opts.gamma, C_d, &descC, &opts.modeC[0],
                    C_d, &descC, &opts.modeC[0],
                    opts.opAB, opts.opABC, opts.typeCompute, 0 /* stream */),
                LWTENSOR_STATUS_SUCCESS);

        // Run CPU Reference
        callingInfo("lwtensorElementwiseReference");
        lwtensorStatus_t err = lwtensorElementwiseReference(&handle,
                opts.alpha, A, &descA, &opts.modeA[0],
                opts.beta,  B, &descB, &opts.modeB[0],
                opts.gamma, C, &descC, &opts.modeC[0],
                opts.opAB, opts.opUnaryAfterBinary, opts.opABC, C_ref, opts.typeCompute);
        if(err != LWTENSOR_STATUS_SUCCESS) {
            printf("ERROR: REFERENCE FAILED!\n");
            exit(-1);
        }

        //compare the results between the CPU and GPU
        if(err == LWTENSOR_STATUS_SUCCESS) {
            size_t elementsC = 1;
            for (int i = 0; i < opts.modeC.size(); ++i)
                elementsC *= opts.extent[opts.modeC[i]];
            lwdaMemcpy2DAsync(C, sizeC, C_d, sizeC, sizeC, 1, lwdaMemcpyDefault, pStream);
            lwdaStreamSynchronize(pStream);
            double relError = opts.disableVerification ? 0 : verify(C, C_ref, elementsC, opts.typeC);
            /*std::cout << "Comparing the results between the CPU and GPU ..."<<std::endl;*/
            EXPECT_FALSE(relError > 0);
            if(relError > 0)
                std::cout << "Relative Error between CPU and GPU is bigger than the threshold: " << relError << "." <<std::endl;
        }
    }


    /**
     * \id lwtensorElementwiseDifferentOperators
     * \brief test the functionality of API "lwtensorElementwiseTrinary" with different operators
     * \depends the required inputs should be defined in class ApiTestDefault
     * \setup parameters and context initialization, memory allocation, et al. (implemented in class ApiTestDefault)
     * \testprocedure ./apiTest --gtest_filter=ApiTestDefault.lwtensorElementwiseDifferentOperators
     * \teardown context destroy, memory free, et al. (implemented in class ApiTestDefault)
     * \designid LWTENSOR_DES_001, LWTENSOR_DES_002, LWTENSOR_DES_004, LWTENSOR_DES_005, LWTENSOR_DES_007, LWTENSOR_DES_008, LWTENSOR_DES_009
     * \testgroup ApiTestDefault
     * \inputs the parameters of lwtensorElementwiseTrinary, defined in class ApiTestDefault
     * \outputs None
     * \expected States: no failure of API lwtensorElementwiseTrinary, same reult with CPU
     */
    TEST_P(ApiTestOperator, lwtensorElementwiseDifferentOperators) {
        callingInfo("lwtensorElementwiseTrinary with different operators combinations");
        EXPECT_EQ(lwtensorElementwiseTrinary(&handle,
                    opts.alpha, A_d, &descA, &opts.modeA[0],
                    opts.beta,  B_d, &descB, &opts.modeB[0],
                    opts.gamma, C_d, &descC, &opts.modeC[0],
                    C_d, &descC, &opts.modeC[0],
                    opts.opAB, opts.opABC, opts.typeCompute, 0 /* stream */),
                LWTENSOR_STATUS_SUCCESS);

        // Run CPU Reference
        callingInfo("lwtensorElementwiseReference");
        lwtensorStatus_t err = lwtensorElementwiseReference(&handle,
                opts.alpha, A, &descA, &opts.modeA[0],
                opts.beta,  B, &descB, &opts.modeB[0],
                opts.gamma, C, &descC, &opts.modeC[0],
                opts.opAB, opts.opUnaryAfterBinary, opts.opABC, C_ref, opts.typeCompute);
        if(err != LWTENSOR_STATUS_SUCCESS) {
            printf("ERROR: REFERENCE FAILED!\n");
            exit(-1);
        }

        //compare the results between the CPU and GPU
        if(err == LWTENSOR_STATUS_SUCCESS) {
            size_t elementsC = 1;
            for (int i = 0; i < opts.modeC.size(); ++i)
                elementsC *= opts.extent[opts.modeC[i]];
            lwdaStreamSynchronize(0);
            lwdaMemcpy2DAsync(C, sizeC, C_d, sizeC, sizeC, 1, lwdaMemcpyDefault, pStream);
            lwdaStreamSynchronize(pStream);
            double relError = opts.disableVerification ? 0 : verify(C, C_ref, elementsC, opts.typeC);
            /*std::cout << "Comparing the results between the CPU and GPU ..."<<std::endl;*/
            EXPECT_FALSE(relError > 0);
            if(relError > 0)
                std::cout << "Relative Error between CPU and GPU is bigger than the threshold: " << relError << "." <<std::endl;
        }
    }

    INSTANTIATE_TEST_CASE_P(LWTENSOR_TEST, ApiTestOperator,
            Combine(
                Values(LWTENSOR_OP_IDENTITY),
                Values(LWTENSOR_OP_IDENTITY),
                Values(LWTENSOR_OP_IDENTITY),
                Values(LWTENSOR_OP_ADD),
                Values(LWTENSOR_OP_ADD)
                ));

    /**
     * \id lwtensorElementwiseDifferentTypes
     * \brief test the functionality of API "lwtensorElementwiseTrinary" with different types
     * \depends the required inputs should be defined in class ApiTestDefault
     * \setup parameters and context initialization, memory allocation, et al. (implemented in class ApiTestDefault)
     * \testprocedure ./apiTest --gtest_filter=ApiTestDefault.lwtensorElementwiseDifferentTypes
     * \teardown context destroy, memory free, et al. (implemented in class ApiTestDefault)
     * \designid LWTENSOR_DES_001, LWTENSOR_DES_002, LWTENSOR_DES_004, LWTENSOR_DES_005, LWTENSOR_DES_007, LWTENSOR_DES_008, LWTENSOR_DES_009
     * \testgroup ApiTestDefault
     * \inputs the parameters of lwtensorElementwiseTrinary, defined in class ApiTestDefault
     * \outputs None
     * \expected States: no failure of API lwtensorElementwiseTrinary, same reult with CPU
     */
    TEST_P(ApiTestTypes, lwtensorElementwiseDifferentTypes) {
        callingInfo("lwtensorElementwiseReference with different types combinations");
        EXPECT_EQ(lwtensorElementwiseTrinary(&handle,
                    opts.alpha, A_d, &descA, &opts.modeA[0],
                    opts.beta,  B_d, &descB, &opts.modeB[0],
                    opts.gamma, C_d, &descC, &opts.modeC[0],
                    C_d, &descC, &opts.modeC[0],
                    opts.opAB, opts.opABC, opts.typeCompute, 0 /* stream */),
                LWTENSOR_STATUS_SUCCESS);

        // Run CPU Reference
        callingInfo("lwtensorElementwiseReference");
        lwtensorStatus_t err = lwtensorElementwiseReference(&handle,
                opts.alpha, A, &descA, &opts.modeA[0],
                opts.beta,  B, &descB, &opts.modeB[0],
                opts.gamma, C, &descC, &opts.modeC[0],
                opts.opAB, opts.opUnaryAfterBinary, opts.opABC, C_ref, opts.typeCompute);
        if(err != LWTENSOR_STATUS_SUCCESS) {
            printf("ERROR: REFERENCE FAILED!\n");
            exit(-1);
        }

        //compare the results between the CPU and GPU
        if(err == LWTENSOR_STATUS_SUCCESS) {
            size_t elementsC = 1;
            for (int i = 0; i < opts.modeC.size(); ++i)
                elementsC *= opts.extent[opts.modeC[i]];
            lwdaStreamSynchronize(0);
            lwdaMemcpy2DAsync(C, sizeC, C_d, sizeC, sizeC, 1, lwdaMemcpyDefault, pStream);
            lwdaStreamSynchronize(pStream);
            double relError = opts.disableVerification ? 0 : verify(C, C_ref, elementsC, opts.typeC);
            /*std::cout << "Comparing the results between the CPU and GPU ..."<<std::endl;*/
            EXPECT_FALSE(relError > 0);
            if(relError > 0)
                std::cout << "Relative Error between CPU and GPU is bigger than the threshold: " << relError << "." <<std::endl;
        }
    }

    TEST_P(ApiTestTypes, lwtensorElementwiseIlwalidateOpts) {
        opts.opAB = LWTENSOR_OP_IDENTITY;
        EXPECT_EQ(lwtensorElementwiseTrinary(&handle,
                    opts.alpha, A_d, &descA, &opts.modeA[0],
                    opts.beta,  B_d, &descB, &opts.modeB[0],
                    opts.gamma, C_d, &descC, &opts.modeC[0],
                    C_d, &descC, &opts.modeC[0],
                    opts.opAB, opts.opABC, opts.typeCompute, 0 /* stream */),
                //LWTENSOR_STATUS_NOT_SUPPORTED);
            LWTENSOR_STATUS_ILWALID_VALUE);
    }

    INSTANTIATE_TEST_CASE_P(LWTENSOR_TEST, ApiTestTypes,
            Values(
               type1,
               type2,
               type3,
               type4,
               type5,
               type6,
               type7
                ));

    /**
     * \id lwtensorElementwiseDifferentCoefficients
     * \brief test the functionality of API "lwtensorElementwiseTrinary" with different coefficients
     * \depends the required inputs should be defined in class ApiTestCoefficient
     * \setup parameters(including revisable alpha/beta/gamma) and context initialization, memory allocation, et al.(implemented in class ApiTestCoefficient)
     * \testprocedure ./apiTest --gtest_filter=ApiTestCoefficient.lwtensorElementwiseDifferentCoefficients
     * \context destroy, memory free, et al.(implemented in class ApiTestCoefficient)
     * \designid LWTENSOR_DES_001, LWTENSOR_DES_002, LWTENSOR_DES_004, LWTENSOR_DES_005, LWTENSOR_DES_007, LWTENSOR_DES_008, LWTENSOR_DES_009
     * \testgroup ApiTestCoefficient
     * \inputs the parameters of lwtensorElementwiseTrinary, defined in class ApiTestCoefficient
     * \outputs None
     * \expected States: no failure of API lwtensorElementwiseTrinary, same reult with CPU
     */
    TEST_P(ApiTestCoefficient, lwtensorElementwiseDifferentCoefficients) {
        callingInfo("lwtensorElementwiseReference with different coefficients combinations");
        EXPECT_EQ(lwtensorElementwiseTrinary(&handle,
                    opts.alpha, A_d, &descA, &opts.modeA[0],
                    opts.beta,  B_d, &descB, &opts.modeB[0],
                    opts.gamma, C_d, &descC, &opts.modeC[0],
                    C_d, &descC, &opts.modeC[0],
                    opts.opAB, opts.opABC, opts.typeCompute, 0 /* stream */),
                LWTENSOR_STATUS_SUCCESS);

        // Run CPU Reference
        callingInfo("lwtensorElementwiseReference");
        lwtensorStatus_t err = lwtensorElementwiseReference(&handle,
                opts.alpha, A, &descA, &opts.modeA[0],
                opts.beta,  B, &descB, &opts.modeB[0],
                opts.gamma, C, &descC, &opts.modeC[0],
                opts.opAB, opts.opUnaryAfterBinary, opts.opABC, C_ref, opts.typeCompute);
        if(err != LWTENSOR_STATUS_SUCCESS) {
            printf("ERROR: REFERENCE FAILED!\n");
            exit(-1);
        }

        //compare the results between the CPU and GPU
        if(err == LWTENSOR_STATUS_SUCCESS) {
            size_t elementsC = 1;
            for (int i = 0; i < opts.modeC.size(); ++i)
                elementsC *= opts.extent[opts.modeC[i]];
            lwdaStreamSynchronize(0);
            lwdaMemcpy2DAsync(C, sizeC, C_d, sizeC, sizeC, 1, lwdaMemcpyDefault, pStream);
            lwdaStreamSynchronize(pStream);
            double relError = opts.disableVerification ? 0 : verify(C, C_ref, elementsC, opts.typeC);
            /*std::cout << "Comparing the results between the CPU and GPU ..."<<std::endl;*/
            EXPECT_FALSE(relError > 0);
            if(relError > 0)
                std::cout << "Relative Error between CPU and GPU is bigger than the threshold: " << relError << "." <<std::endl;
        }
    }

    INSTANTIATE_TEST_CASE_P(LWTENSOR_TEST, ApiTestCoefficient,
            Combine(
                Values(1.5f), //alpha must not be 0
                Values(0.0f), //should be 0, otherwise --> LWTENSOR ERROR: WIP: the support for B has been temporarily deactivated.
                Values(0.0f, 1.5f)
                ));

    /**
     * \id lwtensorElementwiseOperatorNegative
     * \brief test the functionality of API "lwtensorElementwiseTrinary" with negative inputs
     * \depends the required inputs should be defined in class ApiTestOperatorNegative
     * \setup parameters(including revisable operations) and context initialization, memory allocation, et al.(implemented in class ApiTestOperatorNegative)
     * \testprocedure ./apiTest --gtest_filter=ApiTestOperatorNegative.lwtensorElementwiseOperatorNegative
     * \teardown context destroy, memory free, et al.(implemented in class ApiTestOperatorNegative)
     * \testgroup ApiTestOperatorNegative
     * \inputs the parameters of lwtensorElementwiseTrinary, defined in class ApiTestOperatorNegative
     * \outputs None
     * \expected States: LWTENSOR_STATUS_ILWALID_VALUE
     */
    TEST_P(ApiTestOperatorNegative, lwtensorElementwiseOperatorNegative) {
        std::cout << "opA: " << opts.opA << ", "
            << "opB: " << opts.opB << ", "
            << "opC: " << opts.opC << ", "
            << "opAB: " << opts.opAB << ", "
            << "opABC: " << opts.opABC <<std::endl;

        if(opts.opA == LWTENSOR_OP_UNKNOWN ||
                opts.opB == LWTENSOR_OP_UNKNOWN ||
                opts.opC == LWTENSOR_OP_UNKNOWN ||
                opts.opAB == LWTENSOR_OP_UNKNOWN ||
                opts.opABC == LWTENSOR_OP_UNKNOWN) {
            //TODO
            callingInfo("lwtensorElementwiseTrinary with negative inputs");
            EXPECT_EQ(lwtensorElementwiseTrinary(&handle,
                        opts.alpha, A_d, &descA, &opts.modeA[0],
                        opts.beta,  B_d, &descB, &opts.modeB[0],
                        opts.gamma, C_d, &descC, &opts.modeC[0],
                        C_d, &descC, &opts.modeC[0],
                        opts.opAB, opts.opABC, opts.typeCompute, 0 /* stream */),
                    LWTENSOR_STATUS_ILWALID_VALUE);
        }
        else
        {
            callingInfo("lwtensorElementwiseTrinary with correct inputs");
            EXPECT_EQ(lwtensorElementwiseTrinary(&handle,
                        opts.alpha, A_d, &descA, &opts.modeA[0],
                        opts.beta,  B_d, &descB, &opts.modeB[0],
                        opts.gamma, C_d, &descC, &opts.modeC[0],
                        C_d, &descC, &opts.modeC[0],
                        opts.opAB, opts.opABC, opts.typeCompute, 0),
                    LWTENSOR_STATUS_SUCCESS);
        }

    }

    INSTANTIATE_TEST_CASE_P(LWTENSOR_TEST, ApiTestOperatorNegative,
            Combine(
                Values(LWTENSOR_OP_IDENTITY, LWTENSOR_OP_NEG), //not all the combinations are listed here
                Values(LWTENSOR_OP_RELU, LWTENSOR_OP_SQRT),
                Values(LWTENSOR_OP_RCP, LWTENSOR_OP_RELU),
                Values(LWTENSOR_OP_ADD, LWTENSOR_OP_MUL, LWTENSOR_OP_MIN),
                Values(LWTENSOR_OP_MAX, LWTENSOR_OP_UNKNOWN)
                ));


    /**
     * \id lwtensorElementwiseOperatorNegative
     * \brief negative test cases for data-type for API lwtensorElementwiseTrinary
     * \depends the required inputs should be defined in class ApiTestTypeNegative
     * \setup parameters(including revisable data types) and context initialization, memory allocation, et al.(implemented in class ApiTestTypeNegative)
     * \testprocedure ./apiTest --gtest_filter=ApiTestTypeNegative.lwtensorElementwiseTypeNegative
     * \teardown context destroy, memory free, et al.(implemented in class ApiTestTypeNegative)
     * \designid LWTENSOR_DES_001, LWTENSOR_DES_002, LWTENSOR_DES_004, LWTENSOR_DES_005, LWTENSOR_DES_007, LWTENSOR_DES_008, LWTENSOR_DES_009
     * \testgroup ApiTestTypeNegative
     * \inputs the parameters of lwtensorElementwiseTrinary, defined in class ApiTestTypeNegative
     * \outputs None
     * \expected States: output errors should be as expected with incorrect inputs
     */
    TEST_P(ApiTestTypeNegative, lwtensorElementwiseTypeNegative) {
        callingInfo("lwtensorElementwiseTrinary with negative data-type");
        EXPECT_EQ(lwtensorElementwiseTrinary(&handle,
                    opts.alpha, A_d, &descA, &opts.modeA[0],
                    opts.beta,  B_d, &descB, &opts.modeB[0],
                    opts.gamma, C_d, &descC, &opts.modeC[0],
                    C_d, &descC, &opts.modeC[0],
                    opts.opAB, opts.opABC, opts.typeCompute, 0 /* stream */),
                LWTENSOR_STATUS_NOT_SUPPORTED);
    }

    INSTANTIATE_TEST_CASE_P(LWTENSOR_TEST, ApiTestTypeNegative,
            Values(
                ApiTestTypeNegative2,
                ApiTestTypeNegative3,
                ApiTestTypeNegative5,
                ApiTestTypeNegative6
                ));


    /**
     * \id lwtensorElementwiseBetaNegative
     * \brief Test the functionality of API "lwtensorElementwiseTrinary" with negative beta
     * \depends the required inputs should be defined in class ApiTestBetaNegative
     * \setup parameters and context initialization, memory allocation, et al. (implemented in class ApiTestBetaNegative)
     * \testprocedure ./apiTest --gtest_filter=ApiTestBetaNegative.lwtensorElementwiseBetaNegative
     * \teardown context destroy, memory free, et al.
     * \testgroup ApiTestBetaNegative
     * \inputs the parameters of lwtensorElementwiseTrinary, defined in class ApiTestBetaNegative
     * \outputs None
     * \expected States: output errors should be as expected with incorrect inputs
     */
    TEST_F(ApiTestBetaNegative, lwtensorElementwiseBetaNegative)
    {
        callingInfo("lwtensorElementwiseTrinary with negative beta");
        EXPECT_EQ(lwtensorElementwiseTrinary(&handle,
                    opts.alpha, A_d, &descA, &opts.modeA[0],
                    opts.beta,  B_d, &descB, &opts.modeB[0],
                    opts.gamma, C_d, &descC, &opts.modeC[0],
                    C_d, &descC, &opts.modeC[0],
                    opts.opAB, opts.opABC, opts.typeCompute, 0 /* stream */),
                LWTENSOR_STATUS_SUCCESS);
    }

    /**
     * \id lwtensorElementwiseHandleNegative
     * \brief Test the functionality of API "lwtensorElementwiseTrinary" with negative handle
     * \depends the required inputs should be defined in class ApiTestNegative
     * \setup parameters and context initialization, memory allocation, et al.
     * \testprocedure ./apiTest --gtest_filter=ApiTestNegative.lwtensorElementwiseHandleNegative
     * \teardown context destroy, memory free, et al.
     * \testgroup ApiTestNegative
     * \inputs the parameters of lwtensorElementwiseTrinary, defined in class ApiTestNegative
     * \outputs None
     * \expected States: output errors should be as expected with incorrect inputs
     */
    TEST_F(ApiTestNegative, lwtensorElementwiseHandleNegative) {
        callingInfo("lwtensorElementwiseTrinary with negative handle");
        EXPECT_EQ(lwtensorElementwiseTrinary(nullptr,
                    opts.alpha, A_d, &descA, &opts.modeA[0],
                    opts.beta,  B_d, &descB, &opts.modeB[0],
                    opts.gamma, C_d, &descC, &opts.modeC[0],
                    C_d, &descC, &opts.modeC[0],
                    opts.opAB, opts.opABC, opts.typeCompute, 0 /* stream */),
                LWTENSOR_STATUS_NOT_INITIALIZED);
    }

    /**
     * \id lwtensorElementwiseABCNegative
     * \brief Test the functionality of API "lwtensorElementwiseTrinary" with negative A/B/C
     * \depends the required inputs should be defined in class ApiTestNegative
     * \setup parameters and context initialization, memory allocation, et al.
     * \testprocedure ./apiTest --gtest_filter=ApiTestNegative.lwtensorElementwiseABCNegative
     * \teardown context destroy, memory free, et al.
     * \testgroup ApiTestNegative
     * \inputs the parameters of lwtensorElementwiseTrinary, defined in class ApiTestNegative
     * \outputs None
     * \expected States: output errors should be as expected with incorrect inputs
     */
    TEST_F(ApiTestNegative, lwtensorElementwiseABCNegative) {
        callingInfo("lwtensorElementwiseTrinary");
        EXPECT_EQ(lwtensorElementwiseTrinary(&handle,
                    opts.alpha, nullptr, &descA, &opts.modeA[0], // at least valid input (either A, B, or C)
                    opts.beta,  nullptr, &descB, &opts.modeB[0],
                    opts.gamma, nullptr, &descC, &opts.modeC[0],
                    nullptr, &descC, &opts.modeC[0],
                    opts.opAB, opts.opABC, opts.typeCompute, 0 /* stream */),
                LWTENSOR_STATUS_ILWALID_VALUE);
    }

#ifdef DEVELOP
    // bug 200473273
    /**
     * \id lwtensorElementwiseModeNegative
     * \brief Test the functionality of API "lwtensorElementwiseTrinary" with negative mode
     * \depends the required inputs should be defined in class ApiTestNegative
     * \parameters and context initialization, memory allocation, et al.(implemented in class ApiTestNegative)
     * \testprocedure ./apiTest --gtest_filter=ApiTestNegative.lwtensorElementwiseModeNegative
     * \context destroy, memory free, et al.(implemented in class ApiTestNegative)
     * \testgroup ApiTestNegative
     * \inputs the parameters of lwtensorElementwiseTrinary, defined in class ApiTestNegative
     * \outputs None
     * \expected States: output errors should be as expected with incorrect inputs
     */
    TEST_F(ApiTestNegative, lwtensorElementwiseModeNegative) {  // TODO move to positive
        ((TensorDescriptor*)&descA)->setNumModes(0) ;
        callingInfo("lwtensorElementwiseTrinary");
        EXPECT_EQ(lwtensorElementwiseTrinary(&handle,
                    opts.alpha, A_d, &descA, nullptr,
                    opts.beta,  B_d, &descB, &opts.modeB[0],
                    opts.gamma, C_d, &descC, &opts.modeC[0],
                    C_d, &descC, &opts.modeC[0],
                    opts.opAB, opts.opABC, opts.typeCompute, 0 /* stream */),
                LWTENSOR_STATUS_SUCCESS);
    }
#endif

    /**
     * \id lwtensorElementwiseDescNegative
     * \brief Test the functionality of API "lwtensorElementwiseTrinary" with negative descriptions
     * \depends the required inputs should be defined in class ApiTestNegative
     * \parameters and context initialization, memory allocation, et al.(implemented in class ApiTestNegative)
     * \testprocedure ./apiTest --gtest_filter=ApiTestNegative.lwtensorElementwiseDescNegative
     * \context destroy, memory free, et al.(implemented in class ApiTestNegative)
     * \testgroup ApiTestNegative
     * \inputs the parameters of lwtensorElementwiseTrinary, defined in class ApiTestNegative
     * \outputs None
     * \expected States: output errors should be as expected with incorrect inputs
     */
    TEST_F(ApiTestNegative, lwtensorElementwiseDescNegative) {
        callingInfo("lwtensorElementwiseTrinary");
        EXPECT_EQ(lwtensorElementwiseTrinary(&handle,
                    opts.alpha, A_d, &descA, &opts.modeA[0],
                    opts.beta,  B_d, &descB, &opts.modeB[0],
                    opts.gamma, C_d, nullptr, &opts.modeC[0],
                    C_d, nullptr, &opts.modeC[0],
                    opts.opAB, opts.opABC, opts.typeCompute, 0 /* stream */),
                LWTENSOR_STATUS_ILWALID_VALUE);
    }

    /**
     * \id lwtensorElementwiseTrinaryModeNegative
     * \brief Test the functionality of API "lwtensorElementwiseTrinary" with negative mode
     * \depends the required inputs should be defined in class ApiTestNegative
     * \parameters and context initialization, memory allocation, et al.(implemented in class ApiTestNegative)
     * \testprocedure ./apiTest --gtest_filter=ApiTestNegative.lwtensorElementwiseTrinaryModeNegative
     * \context destroy, memory free, et al.(implemented in class ApiTestNegative)
     * \testgroup ApiTestNegative
     * \inputs the parameters of lwtensorElementwiseTrinary, defined in class ApiTestNegative
     * \outputs None
     * \expected States: output errors should be as expected with incorrect inputs
     */
    TEST_F(ApiTestNegative, lwtensorElementwiseTrinaryModeNegative) {
        callingInfo("lwtensorElementwiseTrinary");
        EXPECT_EQ(lwtensorElementwiseTrinary(&handle,
                    opts.alpha, A_d, &descA, NULL,
                    opts.beta,  B_d, &descB, &opts.modeB[0],
                    opts.gamma, C_d, &descC, &opts.modeC[0],
                    C_d, &descC, &opts.modeC[0],
                    opts.opAB, opts.opABC, opts.typeCompute, 0 /* stream */),
                LWTENSOR_STATUS_ILWALID_VALUE);
    }

#ifdef DEVELOP
    /**
     * \id lwtensorElementwiseTrinaryZeroMode
     * \brief Test the functionality of API "lwtensorElementwiseTrinary" with zero mode
     * \depends the required inputs should be defined in class ApiTestNegative
     * \parameters and context initialization, memory allocation, et al.(implemented in class ApiTestNegative)
     * \testprocedure ./apiTest --gtest_filter=ApiTestNegative.lwtensorElementwiseTrinaryZeroMode
     * \context destroy, memory free, et al.(implemented in class ApiTestNegative)
     * \testgroup ApiTestNegative
     * \inputs the parameters of lwtensorElementwiseTrinary, defined in class ApiTestNegative
     * \outputs None
     * \expected States: output errors should be as expected with incorrect inputs
     */
    TEST_F(ApiTestNegative, lwtensorElementwiseTrinaryZeroMode) { // TODO move to positive test
        ((TensorDescriptor*)&descA)->setNumModes(0);
        ((TensorDescriptor*)&descB)->setNumModes(0);
        ((TensorDescriptor*)&descC)->setNumModes(0);
        callingInfo("lwtensorElementwiseTrinary");
        EXPECT_EQ(lwtensorElementwiseTrinary(&handle,
                    opts.alpha, A_d, &descA, opts.modeA.data(),
                    opts.beta,  B_d, &descB, opts.modeB.data(),
                    opts.gamma, C_d, &descC, opts.modeC.data(),
                    C_d, &descC, opts.modeC.data(),
                    opts.opAB, opts.opABC, opts.typeCompute, 0 /* stream */),
                LWTENSOR_STATUS_SUCCESS);
    }
#endif

    /**
     * \id lwtensorElementwiseTrinaryZeroAlpha
     * \brief Test the functionality of API "lwtensorElementwiseTrinary" with zero alpha
     * \depends the required inputs should be defined in class ApiTestNegative
     * \parameters and context initialization, memory allocation, et al.(implemented in class ApiTestNegative)
     * \testprocedure ./apiTest --gtest_filter=ApiTestNegative.lwtensorElementwiseTrinaryZeroAlpha
     * \context destroy, memory free, et al.(implemented in class ApiTestNegative)
     * \designid LWTENSOR_DES_001, LWTENSOR_DES_002, LWTENSOR_DES_004, LWTENSOR_DES_005, LWTENSOR_DES_007, LWTENSOR_DES_008, LWTENSOR_DES_009
     * \testgroup ApiTestNegative
     * \inputs the parameters of lwtensorElementwiseTrinary, defined in class ApiTestNegative
     * \outputs None
     * \expected States: output errors should be as expected with incorrect inputs
     */
    TEST_F(ApiTestNegative, lwtensorElementwiseTrinaryZeroAlpha) {
        initialize(opts.alpha, opts.typeCompute, 1, 0.0f);
        initialize(opts.beta, opts.typeCompute, 1);
        callingInfo("lwtensorElementwiseTrinary");
        EXPECT_EQ(lwtensorElementwiseTrinary(&handle,
                    opts.alpha, A_d, &descA, &opts.modeA[0],
                    opts.beta,  B_d, &descB, &opts.modeB[0],
                    opts.gamma, C_d, &descC, &opts.modeC[0],
                    C_d, &descC, &opts.modeC[0],
                    opts.opAB, opts.opABC, opts.typeCompute, 0 /* stream */),
                LWTENSOR_STATUS_SUCCESS);
    }

    /**
     * \id lwtensorElementwiseTrinaryZeroAB
     * \brief Test the functionality of API "lwtensorElementwiseTrinary" with zero alpha/beta
     * \depends the required inputs should be defined in class ApiTestNegative
     * \parameters and context initialization, memory allocation, et al.(implemented in class ApiTestNegative)
     * \testprocedure ./apiTest --gtest_filter=ApiTestNegative.lwtensorElementwiseTrinaryZeroAB
     * \context destroy, memory free, et al.(implemented in class ApiTestNegative)
     * \designid LWTENSOR_DES_001, LWTENSOR_DES_002, LWTENSOR_DES_004, LWTENSOR_DES_005, LWTENSOR_DES_007, LWTENSOR_DES_008, LWTENSOR_DES_009
     * \testgroup ApiTestNegative
     * \inputs the parameters of lwtensorElementwiseTrinary, defined in class ApiTestNegative
     * \outputs None
     * \expected States: output errors should be as expected with incorrect inputs
     */
    TEST_F(ApiTestNegative, lwtensorElementwiseTrinaryZeroAB) {
        initialize(opts.alpha, opts.typeCompute, 1, 0.0f);
        initialize(opts.beta, opts.typeCompute, 1, 0.0f);
        callingInfo("lwtensorElementwiseTrinary");
        EXPECT_EQ(lwtensorElementwiseTrinary(&handle,
                    opts.alpha, A_d, &descA, &opts.modeA[0],
                    opts.beta,  B_d, &descB, &opts.modeB[0],
                    opts.gamma, C_d, &descC, &opts.modeC[0],
                    C_d, &descC, &opts.modeC[0],
                    opts.opAB, opts.opABC, opts.typeCompute, 0 /* stream */),
                LWTENSOR_STATUS_SUCCESS);
    }

    /**
     * \id lwtensorElementwiseTrinaryZeroABC
     * \brief Test the functionality of API "lwtensorElementwiseTrinary" with zero alpha/beta/gamma
     * \depends the required inputs should be defined in class ApiTestNegative
     * \parameters and context initialization, memory allocation, et al.(implemented in class ApiTestNegative)
     * \testprocedure ./apiTest --gtest_filter=ApiTestNegative.lwtensorElementwiseTrinaryZeroABC
     * \context destroy, memory free, et al.(implemented in class ApiTestNegative)
     * \designid LWTENSOR_DES_001, LWTENSOR_DES_002, LWTENSOR_DES_004, LWTENSOR_DES_005, LWTENSOR_DES_007, LWTENSOR_DES_008, LWTENSOR_DES_009
     * \testgroup ApiTestNegative
     * \inputs the parameters of lwtensorElementwiseTrinary, defined in class ApiTestNegative
     * \outputs None
     * \expected States: output errors should be as expected with incorrect inputs
     */
    TEST_F(ApiTestNegative, lwtensorElementwiseTrinaryZeroABC) {
        initialize(opts.alpha, opts.typeCompute, 1, 0.0f);
        initialize(opts.beta, opts.typeCompute, 1, 0.0f);
        initialize(opts.gamma, opts.typeCompute, 1, 0.0f);
        callingInfo("lwtensorElementwiseTrinary");
        EXPECT_EQ(lwtensorElementwiseTrinary(&handle,
                    opts.alpha, A_d, &descA, &opts.modeA[0],
                    opts.beta,  B_d, &descB, &opts.modeB[0],
                    opts.gamma, C_d, &descC, &opts.modeC[0],
                    C_d, &descC, &opts.modeC[0],
                    opts.opAB, opts.opABC, opts.typeCompute, 0 /* stream */),
                LWTENSOR_STATUS_SUCCESS);
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //              API: lwtensorElementwiseBinary
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    /**
     * \id lwtensorElementwiseBinaryNormal
     * \brief Test the functionality of API "lwtensorElementwiseBinary"
     * \depends the required inputs should be defined in class ApiTestDefault
     * \setup parameters and context initialization, memory allocation, et al. (implemented in class ApiTestDefault)
     * \testprocedure ./apiTest --gtest_filter=ApiTestDefault.lwtensorElementwiseBinaryNormal
     * \teardown context destroy, memory free, et al. (implemented in class ApiTestDefault)
     * \designid LWTENSOR_DES_001, LWTENSOR_DES_002, LWTENSOR_DES_004, LWTENSOR_DES_005, LWTENSOR_DES_007, LWTENSOR_DES_008, LWTENSOR_DES_009
     * \testgroup ApiTestDefault
     * \inputs the parameters of lwtensorElementwiseTrinary, defined in class ApiTestDefault
     * \outputs None
     * \expected States: output errors should be as expected with incorrect inputs
     */
    TEST_F(ApiTestDefault, lwtensorElementwiseBinaryNormal) {
        callingInfo("lwtensorElementwiseBinary");
        EXPECT_EQ(lwtensorElementwiseBinary(&handle,
                    opts.alpha, A_d, &descA, &opts.modeA[0],
                    opts.gamma, C_d, &descC, &opts.modeC[0],
                    C_d, &descC, &opts.modeC[0],
                    opts.opABC, opts.typeCompute, 0 /* stream */),
                LWTENSOR_STATUS_SUCCESS);
        // Run CPU Reference
        callingInfo("lwtensorElementwiseReference");
        lwtensorStatus_t err = lwtensorElementwiseReference(&handle,
                opts.alpha, A, &descA, &opts.modeA[0],
                NULL, NULL, NULL, NULL,
                opts.gamma, C, &descC, &opts.modeC[0],
                opts.opAB, opts.opUnaryAfterBinary, opts.opABC, C_ref, opts.typeCompute);
        //compare the results between the CPU and GPU
        if(err == LWTENSOR_STATUS_SUCCESS) {
            size_t elementsC = 1;
            for(int i = 0; i < opts.modeC.size(); ++i)
                elementsC *= opts.extent[opts.modeC[i]];
            lwdaMemcpy2DAsync(C, sizeC, C_d, sizeC, sizeC, 1, lwdaMemcpyDefault, pStream);
            lwdaStreamSynchronize(pStream);
            double relError = opts.disableVerification ? 0 : verify(C, C_ref, elementsC, opts.typeC);
            EXPECT_FALSE(relError > 0);
            if(relError > 0)
                std::cout << "Relative Error between CPU and GPU is bigger than the threshold: " << relError << "." <<std::endl;
        }
    }

    /**
     * \id lwtensorElementwiseBinaryModeNegative
     * \brief Test the functionality of API "lwtensorElementwiseBinary" with negative mode
     * \depends the required inputs should be defined in class ApiTestNegative
     * \parameters and context initialization, memory allocation, et al.(implemented in class ApiTestNegative)
     * \testprocedure ./apiTest --gtest_filter=ApiTestNegative.lwtensorElementwiseBinaryModeNegative
     * \context destroy, memory free, et al.(implemented in class ApiTestNegative)
     * \designid LWTENSOR_DES_001, LWTENSOR_DES_002, LWTENSOR_DES_004, LWTENSOR_DES_005, LWTENSOR_DES_007, LWTENSOR_DES_008, LWTENSOR_DES_009
     * \testgroup ApiTestNegative
     * \inputs the parameters of lwtensorElementwiseTrinary, defined in class ApiTestNegative
     * \outputs None
     * \expected States: output errors should be as expected with incorrect inputs
     */
    TEST_F(ApiTestNegative, lwtensorElementwiseBinaryModeNegative) {
        callingInfo("lwtensorElementwiseBinary");
        EXPECT_EQ(lwtensorElementwiseBinary(&handle,
                    opts.alpha, A_d, &descA, NULL,
                    opts.gamma, C_d, &descC, &opts.modeC[0],
                    C_d, &descC, &opts.modeC[0],
                    opts.opABC, opts.typeCompute, 0 /* stream */),
                LWTENSOR_STATUS_ILWALID_VALUE);
    }

#ifdef DEVELOP
    /**
     * \id lwtensorElementwiseBinaryZeroMode
     * \brief Test the functionality of API "lwtensorElementwiseBinary" with zero mode
     * \depends the required inputs should be defined in class ApiTestNegative
     * \parameters and context initialization, memory allocation, et al.(implemented in class ApiTestNegative)
     * \testprocedure ./apiTest --gtest_filter=ApiTestNegative.lwtensorElementwiseBinaryZeroMode
     * \context destroy, memory free, et al.(implemented in class ApiTestNegative)
     * \designid LWTENSOR_DES_001, LWTENSOR_DES_002, LWTENSOR_DES_004, LWTENSOR_DES_005, LWTENSOR_DES_007, LWTENSOR_DES_008, LWTENSOR_DES_009
     * \testgroup ApiTestNegative
     * \inputs the parameters of lwtensorElementwiseTrinary, defined in class ApiTestNegative
     * \outputs None
     * \expected States: output errors should be as expected with incorrect inputs
     */
    TEST_F(ApiTestNegative, lwtensorElementwiseBinaryZeroMode) { //TODO move to positive test
        ((TensorDescriptor*)&descA)->setNumModes(0);
        ((TensorDescriptor*)&descC)->setNumModes(0);
        callingInfo("lwtensorElementwiseBinary");
        EXPECT_EQ(lwtensorElementwiseBinary(&handle,
                    opts.alpha, A_d, &descA, &opts.modeA[0],
                    opts.gamma, C_d, &descC, &opts.modeC[0],
                    C_d, &descC, &opts.modeC[0],
                    opts.opABC, opts.typeCompute, 0 /* stream */),
                LWTENSOR_STATUS_SUCCESS);
    }
#endif


    /**
     * \id lwtensorElementwiseTrinaryOneModeC
     * \brief Test the functionality of API "lwtensorElementwiseTrinary" with one mode
     * \depends the required inputs should be defined in class ApiTestOneModeC
     * \setup parameters(including modeA/B/C) and context initialization, memory allocation, et al.
     * \testprocedure ./apiTest --gtest_filter=ApiTestOneModeC.lwtensorElementwiseTrinaryOneModeC
     * \teardown context destroy, memory free, et al.
     * \designid LWTENSOR_DES_001, LWTENSOR_DES_002, LWTENSOR_DES_004, LWTENSOR_DES_005, LWTENSOR_DES_007, LWTENSOR_DES_008, LWTENSOR_DES_009
     * \testgroup ApiTestOneModeC
     * \inputs the parameters of lwtensorElementwiseTrinary, defined in class ApiTestOneModeC
     * \outputs None
     * \expected States: output errors should be as expected with incorrect inputs
     */
    TEST_P(ApiTestOneModeC, lwtensorElementwiseTrinaryOneModeC) {
        callingInfo("lwtensorElementwiseTinary");
        EXPECT_EQ(lwtensorElementwiseTrinary(&handle,
                    opts.alpha, A_d, &descA, &opts.modeA[0],
                    opts.beta,  B_d, &descB, &opts.modeB[0],
                    opts.gamma, C_d, &descC, &opts.modeC[0],
                    C_d, &descC, &opts.modeC[0],
                    opts.opAB, opts.opABC, opts.typeCompute, 0 /* stream */),
                LWTENSOR_STATUS_SUCCESS);

    }

    /**
     * \id lwtensorElementwiseTrinaryOneModeCIlwalidOpts
     * \brief Test the functionality of API "lwtensorElementwiseTrinary" with invalid opts
     * \depends the required inputs should be defined in class ApiTestOneModeC
     * \setup parameters(including modeA/B/C) and context initialization, memory allocation, et al.
     * \testprocedure ./apiTest --gtest_filter=ApiTestOneModeC.lwtensorElementwiseTrinaryOneModeCIlwalidOpts
     * \teardown context destroy, memory free, et al.
     * \designid LWTENSOR_DES_001, LWTENSOR_DES_002, LWTENSOR_DES_004, LWTENSOR_DES_005, LWTENSOR_DES_007, LWTENSOR_DES_008, LWTENSOR_DES_009
     * \testgroup ApiTestOneModeC
     * \inputs the parameters of lwtensorElementwiseTrinary, defined in class ApiTestOneModeC
     * \outputs None
     * \expected States: output errors should be as expected with incorrect inputs
     */
    TEST_P(ApiTestOneModeC, lwtensorElementwiseTrinaryOneModeCIlwalidOpts) {
        opts.opAB = LWTENSOR_OP_IDENTITY;
        callingInfo("lwtensorElementwiseTrinary");
        EXPECT_EQ(lwtensorElementwiseTrinary(&handle,
                    opts.alpha, A_d, &descA, &opts.modeA[0],
                    opts.beta,  B_d, &descB, &opts.modeB[0],
                    opts.gamma, C_d, &descC, &opts.modeC[0],
                    C_d, &descC, &opts.modeC[0],
                    opts.opAB, opts.opABC, opts.typeCompute, 0 /* stream */),
                //CHANGE: LWTENSOR_STATUS_NOT_SUPPORTED);
            LWTENSOR_STATUS_ILWALID_VALUE);
    }

    INSTANTIATE_TEST_CASE_P(LWTENSOR_TEST, ApiTestOneModeC,
            Values(
                type1,
                type2,
                type3,
                type4,
                type5,
                type6,
                type7,
                type8,
                type9
                ));

    /**
     * \id lwtensorElementwiseTrinarySmallExtent
     * \brief Test the functionality of API "lwtensorElementwiseTrinary" with small extent
     * \depends the required inputs should be defined in class ApiTestSmallExtent
     * \setup parameters(small extent) and context initialization, memory allocation, et al.
     * \testprocedure ./apiTest --gtest_filter=ApiTestSmallExtent.lwtensorElementwiseTrinarySmallExtent
     * \teardown context destroy, memory free, et al.
     * \designid LWTENSOR_DES_001, LWTENSOR_DES_002, LWTENSOR_DES_004, LWTENSOR_DES_005, LWTENSOR_DES_007, LWTENSOR_DES_008, LWTENSOR_DES_009
     * \testgroup ApiTestSmallExtent
     * \inputs the parameters of lwtensorElementwiseTrinary, defined in class ApiTestSmallExtent
     * \outputs None
     * \expected States: output errors should be as expected with incorrect inputs
     */
    TEST_P(ApiTestSmallExtent, lwtensorElementwiseTrinarySmallExtent) {
        callingInfo("lwtensorElementwiseTrinary");
        EXPECT_EQ(lwtensorElementwiseTrinary(&handle,
                    opts.alpha, A_d, &descA, &opts.modeA[0],
                    opts.beta,  B_d, &descB, &opts.modeB[0],
                    opts.gamma, C_d, &descC, &opts.modeC[0],
                    C_d, &descC, &opts.modeC[0],
                    opts.opAB, opts.opABC, opts.typeCompute, 0 /* stream */),
                LWTENSOR_STATUS_SUCCESS);

    }

    /**
     * \id lwtensorElementwiseTrinarySmallExtentIlwalidOpts
     * \brief Test the functionality of API "lwtensorElementwiseTrinary" with invalid opts
     * \depends the required inputs should be defined in class ApiTestSmallExtent
     * \setup parameters(small extent) and context initialization, memory allocation, et al.
     * \testprocedure ./apiTest --gtest_filter=ApiTestSmallExtent.lwtensorElementwiseTrinarySmallExtentIlwalidOpts
     * \context destroy, memory free, et al.
     * \designid LWTENSOR_DES_001, LWTENSOR_DES_002, LWTENSOR_DES_004, LWTENSOR_DES_005, LWTENSOR_DES_007, LWTENSOR_DES_008, LWTENSOR_DES_009
     * \testgroup ApiTestSmallExtent
     * \inputs the parameters of lwtensorElementwiseTrinary, defined in class ApiTestSmallExtent
     * \outputs None
     * \expected States: output errors should be as expected with incorrect inputs
     */
    TEST_P(ApiTestSmallExtent, lwtensorElementwiseTrinarySmallExtentIlwalidOpts) {
        opts.opAB = LWTENSOR_OP_IDENTITY;
        callingInfo("lwtensorElementwiseTrinary");
        EXPECT_EQ(lwtensorElementwiseTrinary(&handle,
                    opts.alpha, A_d, &descA, &opts.modeA[0],
                    opts.beta,  B_d, &descB, &opts.modeB[0],
                    opts.gamma, C_d, &descC, &opts.modeC[0],
                    C_d, &descC, &opts.modeC[0],
                    opts.opAB, opts.opABC, opts.typeCompute, 0 /* stream */),
                //LWTENSOR_STATUS_NOT_SUPPORTED);
            LWTENSOR_STATUS_ILWALID_VALUE);
    }
    INSTANTIATE_TEST_CASE_P(LWTENSOR_TEST, ApiTestSmallExtent,
            Values(
                type1,
                type2,
                type3,
                type4,
                type5,
                type6,
                type7,
                type8,
                type9
                ));

    /**
     * \id lwtensorPermutationDefault
     * \brief Test the functionality of API "lwtensorElementwiseTrinary" with invalid opts
     * \depends the required inputs should be defined in class PermutationTestDefault
     * \setup parameters and context initialization, memory allocation, et al.
     * \testprocedure ./apiTest --gtest_filter=PermutationTestDefault.lwtensorPermutationDefault
     * \teardown context destroy, memory free, et al.
     * \designid LWTENSOR_DES_001, LWTENSOR_DES_002, LWTENSOR_DES_004, LWTENSOR_DES_005, LWTENSOR_DES_007, LWTENSOR_DES_008, LWTENSOR_DES_009
     * \testgroup PermutationTestDefault
     * \inputs the parameters of lwtensorElementwiseTrinary, defined in class PermutationTestDefault
     * \outputs None
     * \expected States: output errors should be as expected with incorrect inputs
     */
    TEST_F(PermutationTestDefault, lwtensorPermutationDefault) {
        callingInfo("lwtensorPermutation");
        EXPECT_EQ(lwtensorPermutation(&handle, opts.alpha, A_d, &descA, &opts.modeA[0],
                    B_d, &descB, &opts.modeB[0], opts.typeCompute, 0 /* stream */), LWTENSOR_STATUS_SUCCESS);

        //compute the CPU reference result
        callingInfo("lwtensorElementwiseReference");
        lwtensorStatus_t err = lwtensorElementwiseReference(&handle,
                opts.alpha, A, &descA, &opts.modeA[0],
                NULL, NULL, NULL, NULL,
                NULL, B,    &descB,  &opts.modeB[0],
                opts.opAB, opts.opUnaryAfterBinary, opts.opABC, B_ref, opts.typeCompute);

        if(err != LWTENSOR_STATUS_SUCCESS) {
            printf("ERROR: REFERENCE FAILED!\n");
            exit(-1);
        }

        //compare the results between the CPU and GPU
        if(err == LWTENSOR_STATUS_SUCCESS) {
            size_t elementsB = 1;
            for (int i = 0; i < opts.modeB.size(); ++i) {
                elementsB *= opts.extent[opts.modeB[i]];
            }
            lwdaStreamSynchronize(0);
            lwdaMemcpy2DAsync(B, sizeB, B_d, sizeB, sizeB, 1, lwdaMemcpyDefault, pStream);
            lwdaStreamSynchronize(pStream);
            double relError = opts.disableVerification ? 0 : verify(B, B_ref, elementsB, opts.typeA);
            EXPECT_FALSE(relError > 0);
            if(relError > 0)
                std::cout << "Relative Error between CPU and GPU is bigger than the threshold: " << relError << "." <<std::endl;
        }
    }

    /**
     * \id lwtensorPermutationNullHandle
     * \brief Test the functionality of API "lwtensorElementwiseTrinary" with invalid descriptor
     * \depends the required inputs should be defined in class PermutationTestDefault
     * \setup parameters and context initialization, memory allocation, et al.
     * \testprocedure ./apiTest --gtest_filter=PermutationTestDefault.lwtensorPermutationNullHandle
     * \context destroy, memory free, et al.
     * \designid LWTENSOR_DES_001, LWTENSOR_DES_002, LWTENSOR_DES_004, LWTENSOR_DES_005, LWTENSOR_DES_007, LWTENSOR_DES_008, LWTENSOR_DES_009
     * \testgroup PermutationTestDefault
     * \inputs the parameters of lwtensorElementwiseTrinary, defined in class PermutationTestDefault
     * \outputs None
     * \expected States: output errors should be as expected with incorrect inputs
     */
    TEST_F(PermutationTestDefault, lwtensorPermutationNullHandle) {
        callingInfo("lwtensorPermutation");
        EXPECT_EQ(lwtensorPermutation(NULL, opts.alpha, A_d, &descA, &opts.modeA[0],
                    B_d, &descB, &opts.modeB[0], opts.typeCompute, 0 /* stream */), LWTENSOR_STATUS_NOT_INITIALIZED);
    }

    /**
     * \id lwtensorPermutationNullAlpha
     * \brief Test the functionality of API "lwtensorElementwiseTrinary" with null alpha
     * \depends the required inputs should be defined in class PermutationTestDefault
     * \setup parameters and context initialization, memory allocation, et al.
     * \testprocedure ./apiTest --gtest_filter=PermutationTestDefault.lwtensorPermutationNullAlpha
     * \context destroy, memory free, et al.
     * \designid LWTENSOR_DES_001, LWTENSOR_DES_002, LWTENSOR_DES_004, LWTENSOR_DES_005, LWTENSOR_DES_007, LWTENSOR_DES_008, LWTENSOR_DES_009
     * \testgroup PermutationTestDefault
     * \inputs the parameters of lwtensorElementwiseTrinary, defined in class PermutationTestDefault
     * \outputs None
     * \expected States: output errors should be as expected with incorrect inputs
     */
    TEST_F(PermutationTestDefault, lwtensorPermutationNullAlpha) {
        callingInfo("lwtensorPermutation");
        EXPECT_EQ(lwtensorPermutation(&handle, NULL, A_d, &descA, &opts.modeA[0],
                    B_d, &descB, &opts.modeB[0], opts.typeCompute, 0 /* stream */), LWTENSOR_STATUS_ILWALID_VALUE);
    }

    /**
     * \id lwtensorPermutationZeroAlpha
     * \brief Test the functionality of API "lwtensorElementwiseTrinary" with zero alpha
     * \depends the required inputs should be defined in class PermutationTestDefault
     * \setup parameters and context initialization, memory allocation, et al.
     * \testprocedure ./apiTest --gtest_filter=PermutationTestDefault.lwtensorPermutationZeroAlpha
     * \context destroy, memory free, et al.
     * \designid LWTENSOR_DES_001, LWTENSOR_DES_002, LWTENSOR_DES_004, LWTENSOR_DES_005, LWTENSOR_DES_007, LWTENSOR_DES_008, LWTENSOR_DES_009
     * \testgroup PermutationTestDefault
     * \inputs the parameters of lwtensorElementwiseTrinary, defined in class PermutationTestDefault
     * \outputs None
     * \expected States: output errors should be as expected with incorrect inputs
     */
    TEST_F(PermutationTestDefault, lwtensorPermutationZeroAlpha) {
        initialize(opts.alpha, opts.typeCompute, 1, 0);
        callingInfo("lwtensorPermutation");
        EXPECT_EQ(lwtensorPermutation(&handle, opts.alpha, A_d, &descA, &opts.modeA[0],
                    B_d, &descB, &opts.modeB[0], opts.typeCompute, 0 /* stream */),
                LWTENSOR_STATUS_SUCCESS);
    }

    /**
     * \id lwtensorPermutationNullDescA
     * \brief Test the functionality of API "lwtensorElementwiseTrinary" with null descA
     * \depends the required inputs should be defined in class PermutationTestDefault
     * \setup parameters and context initialization, memory allocation, et al.
     * \testprocedure ./apiTest --gtest_filter=PermutationTestDefault.lwtensorPermutationNullDescA
     * \context destroy, memory free, et al.
     * \designid LWTENSOR_DES_001, LWTENSOR_DES_002, LWTENSOR_DES_004, LWTENSOR_DES_005, LWTENSOR_DES_007, LWTENSOR_DES_008, LWTENSOR_DES_009
     * \testgroup PermutationTestDefault
     * \inputs the parameters of lwtensorElementwiseTrinary, defined in class PermutationTestDefault
     * \outputs None
     * \expected States: output errors should be as expected with incorrect inputs
     */
    TEST_F(PermutationTestDefault, lwtensorPermutationNullDescA) {
        callingInfo("lwtensorPermutation");
        EXPECT_EQ(lwtensorPermutation(&handle, opts.alpha, A_d, NULL, &opts.modeA[0],
                    B_d, &descB, &opts.modeB[0], opts.typeCompute, 0 /* stream */), LWTENSOR_STATUS_ILWALID_VALUE);
    }

    /**
     * \id lwtensorPermutationNullModeA
     * \brief Test the functionality of API "lwtensorElementwiseTrinary" with null modeA
     * \depends the required inputs should be defined in class PermutationTestDefault
     * \setup parameters and context initialization, memory allocation, et al.
     * \testprocedure ./apiTest --gtest_filter=PermutationTestDefault.lwtensorPermutationNullModeA
     * \context destroy, memory free, et al.
     * \designid LWTENSOR_DES_001, LWTENSOR_DES_002, LWTENSOR_DES_004, LWTENSOR_DES_005, LWTENSOR_DES_007, LWTENSOR_DES_008, LWTENSOR_DES_009
     * \testgroup PermutationTestDefault
     * \inputs the parameters of lwtensorElementwiseTrinary, defined in class PermutationTestDefault
     * \outputs None
     * \expected States: output errors should be as expected with incorrect inputs
     */
    TEST_F(PermutationTestDefault, lwtensorPermutationNullModeA) {
        callingInfo("lwtensorPermutation");
        EXPECT_EQ(lwtensorPermutation(&handle, opts.alpha, A_d, &descA, NULL,
                    B_d, &descB, &opts.modeB[0], opts.typeCompute, 0 /* stream */), LWTENSOR_STATUS_ILWALID_VALUE);
    }

    /**
     * \id lwtensorPermutationOpts
     * \brief Test the functionality of API "lwtensorElementwiseTrinary" with a different opB
     * \depends the required inputs should be defined in class PermutationTestDefault
     * \setup parameters and context initialization, memory allocation, et al.
     * \testprocedure ./apiTest --gtest_filter=PermutationTestDefault.lwtensorPermutationOpts
     * \context destroy, memory free, et al.
     * \designid LWTENSOR_DES_001, LWTENSOR_DES_002, LWTENSOR_DES_004, LWTENSOR_DES_005, LWTENSOR_DES_007, LWTENSOR_DES_008, LWTENSOR_DES_009
     * \testgroup PermutationTestDefault
     * \inputs the parameters of lwtensorElementwiseTrinary, defined in class PermutationTestDefault
     * \outputs None
     * \expected States: output errors should be as expected with incorrect inputs
     */
    TEST_F(PermutationTestDefault, lwtensorPermutationOpts) {
        opts.opB = LWTENSOR_OP_SQRT;
        callingInfo("lwtensorPermutation");
        EXPECT_EQ(lwtensorPermutation(&handle, opts.alpha, A_d, &descA, NULL,
                    B_d, &descB, &opts.modeB[0], opts.typeCompute, 0 /* stream */), LWTENSOR_STATUS_ILWALID_VALUE);
    }

#ifdef DEVELOP
    /**
     * \id lwtensorPermutationTypes
     * \brief Test the functionality of API "lwtensorElementwiseTrinary" with type LWDA_R_8I
     * \depends the required inputs should be defined in class PermutationTestDefault
     * \setup parameters and context initialization, memory allocation, et al.
     * \testprocedure ./apiTest --gtest_filter=PermutationTestDefault.lwtensorPermutationTypes
     * \context destroy, memory free, et al.
     * \designid LWTENSOR_DES_001, LWTENSOR_DES_002, LWTENSOR_DES_004, LWTENSOR_DES_005, LWTENSOR_DES_007, LWTENSOR_DES_008, LWTENSOR_DES_009
     * \testgroup PermutationTestDefault
     * \inputs the parameters of lwtensorElementwiseTrinary, defined in class PermutationTestDefault
     * \outputs None
     * \expected States: output errors should be as expected with incorrect inputs
     */
    TEST_F(PermutationTestDefault, lwtensorPermutationTypes) {
        ((TensorDescriptor*)&descA)->setDataType(LWDA_R_8I);
        callingInfo("lwtensorPermutation");
        EXPECT_EQ(lwtensorPermutation(&handle, opts.alpha, A_d, &descA, NULL,
                    B_d, &descB, &opts.modeB[0], opts.typeCompute, 0 /* stream */), LWTENSOR_STATUS_ILWALID_VALUE);
    }
#endif

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //              API: functions in operators.h
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    /**
     * \id lwGet_signedChar0
     * \brief validate the functionality of operator lwGet, colwert from signed char to signed char
     * \depends None
     * \setup None
     * \testprocedure ./apiTest --gtest_filter=OPERATORS_H.lwGet_signedChar0
     * \teardown None
     * \testgroup OPERATORS_H
     * \inputs None
     * \outputs None
     * \expected States: the returned value is as expected
     */
    TEST(OPERATORS_H, lwGet_signedChar0)
    {
        signed char a;
        signed char b;
        a = 'a';
        callingInfo("lwGet<signed char>");
        b = lwGet<signed char>(a);
        EXPECT_EQ(a, b);
    }

    /**
     * \id lwGet_signedChar1
     * \brief validate the functionality of operator lwGet, colwert from signed char to unsigned char
     * \depends None
     * \setup None
     * \testprocedure ./apiTest --gtest_filter=OPERATORS_H.lwGet_signedChar1
     * \teardown None
     * \testgroup OPERATORS_H
     * \inputs None
     * \outputs None
     * \expected States: the returned value is as expected
     */
    TEST(OPERATORS_H, lwGet_signedChar1)
    {
        signed char a;
        unsigned char b;
        a = 'a';
        callingInfo("lwGet<unsigned char>");
        b = lwGet<unsigned char>(a);
        EXPECT_EQ((unsigned char)a, b);
    }

    /**
     * \id lwGet_signedChar2
     * \brief validate the functionality of operator lwGet, colwert from signed char to int
     * \depends None
     * \setup None
     * \testprocedure ./apiTest --gtest_filter=OPERATORS_H.lwGet_signedChar2
     * \teardown None
     * \testgroup OPERATORS_H
     * \inputs None
     * \outputs None
     * \expected States: the returned value is as expected
     */
    TEST(OPERATORS_H, lwGet_signedChar2)
    {
        signed char a;
        int b;
        a = 'a';
        callingInfo("lwGet<int>");
        b = lwGet<int>(a);
        EXPECT_EQ((int)a, b);
    }

    /**
     * \id lwGet_signedChar3
     * \brief validate the functionality of operator lwGet, colwert from signed char to half
     * \depends None
     * \setup None
     * \testprocedure ./apiTest --gtest_filter=OPERATORS_H.lwGet_signedChar3
     * \teardown None
     * \testgroup OPERATORS_H
     * \inputs None
     * \outputs None
     * \expected States: the returned value is as expected
     */
    TEST(OPERATORS_H, lwGet_signedChar3)
    {
        signed char a;
        half b;
        a = 'a';
        callingInfo("lwGet<half>");
        b = lwGet<half>(a);
        EXPECT_TRUE(lwIsEqual(__float2half_rn(float(a)), b));
    }

    /**
     * \id lwGet_signedChar4
     * \brief validate the functionality of operator lwGet, colwert from signed char to float
     * \depends None
     * \setup None
     * \testprocedure ./apiTest --gtest_filter=OPERATORS_H.lwGet_signedChar4
     * \teardown None
     * \testgroup OPERATORS_H
     * \inputs None
     * \outputs None
     * \expected States: the returned value is as expected
     */
    TEST(OPERATORS_H, lwGet_signedChar4)
    {
        signed char a;
        float b;
        a = 'a';
        callingInfo("lwGet<float>");
        b = lwGet<float>(a);
        EXPECT_EQ(float(a), b);
    }

    /**
     * \id lwGet_signedChar5
     * \brief validate the functionality of operator lwGet, colwert from signed char to unsigned
     * \depends None
     * \setup None
     * \testprocedure ./apiTest --gtest_filter=OPERATORS_H.lwGet_signedChar5
     * \teardown None
     * \testgroup OPERATORS_H
     * \inputs None
     * \outputs None
     * \expected States: the returned value is as expected
     */
    TEST(OPERATORS_H, lwGet_signedChar5)
    {
        signed char a;
        unsigned b;
        a = 'a';
        callingInfo("lwGet<unsigned>");
        b = lwGet<unsigned>(a);
        EXPECT_EQ(unsigned(a), b);
    }

    /**
     * \id lwGet_signedChar6
     * \brief validate the functionality of operator lwGet, colwert from signed char to double
     * \depends None
     * \setup None
     * \testprocedure ./apiTest --gtest_filter=OPERATORS_H.lwGet_signedChar6
     * \teardown None
     * \testgroup OPERATORS_H
     * \inputs None
     * \outputs None
     * \expected States: the returned value is as expected
     */
    TEST(OPERATORS_H, lwGet_signedChar6)
    {
        signed char a;
        double b;
        a = 'a';
        callingInfo("lwGet<double>");
        b = lwGet<double>(a);
        EXPECT_EQ(double(a), b);
    }

    // ----------------------------------------------------------------------------
    // Functions to initialize T_ELEM from int
    // ----------------------------------------------------------------------------

    /**
     * \id lwGet_int0
     * \brief validate the functionality of operator lwGet, colwert from int(negative) to signed char
     * \depends None
     * \setup None
     * \testprocedure ./apiTest --gtest_filter=OPERATORS_H.lwGet_int0
     * \teardown None
     * \testgroup OPERATORS_H
     * \inputs None
     * \outputs None
     * \expected States: the returned value is as expected
     */
    TEST(OPERATORS_H, lwGet_int0)
    {
        int a;
        signed char b;
        a = -129;
        callingInfo("lwGet<signed char>");
        b = lwGet<signed char>(a);
        EXPECT_EQ((signed char)(-128), b);
    }

    /**
     * \id lwGet_int1
     * \brief validate the functionality of operator lwGet, colwert from int(positive) to signed char
     * \depends None
     * \setup None
     * \testprocedure ./apiTest --gtest_filter=OPERATORS_H.lwGet_int1
     * \teardown None
     * \testgroup OPERATORS_H
     * \inputs None
     * \outputs None
     * \expected States: the returned value is as expected
     */
    TEST(OPERATORS_H, lwGet_int1)
    {
        int a;
        signed char b;
        a = 128;
        callingInfo("lwGet<signed char>");
        b = lwGet<signed char>(a);
        EXPECT_EQ((signed char)(127), b);
    }

    /**
     * \id lwGet_int2
     * \brief validate the functionality of operator lwGet, colwert from int to signed char
     * \depends None
     * \setup None
     * \testprocedure ./apiTest --gtest_filter=OPERATORS_H.lwGet_int2
     * \teardown None
     * \testgroup OPERATORS_H
     * \inputs None
     * \outputs None
     * \expected States: the returned value is as expected
     */
    TEST(OPERATORS_H, lwGet_int2)
    {
        int a;
        signed char b;
        a = 1;
        callingInfo("lwGet<signed char>");
        b = lwGet<signed char>(a);
        EXPECT_EQ((signed char)a, b);
    }

    /**
     * \id lwGet_int3
     * \brief validate the functionality of operator lwGet, colwert from int(negative) to unsigned char
     * \depends None
     * \setup None
     * \testprocedure ./apiTest --gtest_filter=OPERATORS_H.lwGet_int3
     * \teardown None
     * \testgroup OPERATORS_H
     * \inputs None
     * \outputs None
     * \expected States: the returned value is as expected
     */
    TEST(OPERATORS_H, lwGet_int3)
    {
        int a;
        unsigned char b;
        a = -1;
        callingInfo("lwGet<unsigned char>");
        b = lwGet<unsigned char>(a);
        EXPECT_EQ((unsigned char)0, b);
    }

    /**
     * \id lwGet_int4
     * \brief validate the functionality of operator lwGet, colwert from int(a large value) to unsigned char
     * \depends None
     * \setup None
     * \testprocedure ./apiTest --gtest_filter=OPERATORS_H.lwGet_int4
     * \teardown None
     * \testgroup OPERATORS_H
     * \inputs None
     * \outputs None
     * \expected States: the returned value is as expected
     */
    TEST(OPERATORS_H, lwGet_int4)
    {
        int a;
        unsigned char b;
        a = 256;
        callingInfo("lwGet<unsigned char>");
        b = lwGet<unsigned char>(a);
        EXPECT_EQ((unsigned char)255, b);
    }

    /**
     * \id lwGet_int5
     * \brief validate the functionality of operator lwGet, colwert from int to unsigned char
     * \depends None
     * \setup None
     * \testprocedure ./apiTest --gtest_filter=OPERATORS_H.lwGet_int5
     * \teardown None
     * \testgroup OPERATORS_H
     * \inputs None
     * \outputs None
     * \expected States: the returned value is as expected
     */
    TEST(OPERATORS_H, lwGet_int5)
    {
        int a;
        unsigned char b;
        a = 1;
        callingInfo("lwGet<unsigned char>");
        b = lwGet<unsigned char>(a);
        EXPECT_EQ((unsigned char)a, b);
    }

    /**
     * \id lwGet_int6
     * \brief validate the functionality of operator lwGet, colwert from int to unsigned
     * \depends None
     * \setup None
     * \testprocedure ./apiTest --gtest_filter=OPERATORS_H.lwGet_int6
     * \teardown None
     * \testgroup OPERATORS_H
     * \inputs None
     * \outputs None
     * \expected States: the returned value is as expected
     */
    TEST(OPERATORS_H, lwGet_int6)
    {
        int a;
        unsigned b;
        a = 1;
        callingInfo("lwGet<unsigned>");
        b = lwGet<unsigned>(a);
        EXPECT_EQ((unsigned)a, b);
    }

    /**
     * \id lwGet_int7
     * \brief validate the functionality of operator lwGet, colwert from int to int
     * \depends None
     * \setup None
     * \testprocedure ./apiTest --gtest_filter=OPERATORS_H.lwGet_int7
     * \teardown None
     * \testgroup OPERATORS_H
     * \inputs None
     * \outputs None
     * \expected States: the returned value is as expected
     */
    TEST(OPERATORS_H, lwGet_int7)
    {
        int a;
        int b;
        a = -1;
        callingInfo("lwGet<int>");
        b = lwGet<int>(a);
        EXPECT_EQ(a, b);
    }

    /**
     * \id lwGet_int8
     * \brief validate the functionality of operator lwGet, colwert from int to half
     * \depends None
     * \setup None
     * \testprocedure ./apiTest --gtest_filter=OPERATORS_H.lwGet_int8
     * \teardown None
     * \testgroup OPERATORS_H
     * \inputs None
     * \outputs None
     * \expected States: the returned value is as expected
     */
    TEST(OPERATORS_H, lwGet_int8)
    {
        int a;
        half b;
        a = -1;
        callingInfo("lwGet<half>");
        b = lwGet<half>(a);
        callingInfo("lwIsEqual");
        EXPECT_TRUE(lwIsEqual(__float2half_rn(float(a)), b));
    }

    /**
     * \id lwGet_int9
     * \brief validate the functionality of operator lwGet, colwert from int to float
     * \depends None
     * \setup None
     * \testprocedure ./apiTest --gtest_filter=OPERATORS_H.lwGet_int9
     * \teardown None
     * \testgroup OPERATORS_H
     * \inputs None
     * \outputs None
     * \expected States: the returned value is as expected
     */
    TEST(OPERATORS_H, lwGet_int9)
    {
        int a;
        float b;
        a = -1;
        callingInfo("lwGet<float>");
        b = lwGet<float>(a);
        EXPECT_EQ(float(a), b);
    }

    /**
     * \id lwGet_int10
     * \brief validate the functionality of operator lwGet, colwert from int to double
     * \depends None
     * \setup None
     * \testprocedure ./apiTest --gtest_filter=OPERATORS_H.lwGet_int10
     * \teardown None
     * \testgroup OPERATORS_H
     * \inputs None
     * \outputs None
     * \expected States: the returned value is as expected
     */
    TEST(OPERATORS_H, lwGet_int10)
    {
        int a;
        double b;
        a = -1;
        callingInfo("lwGet<double>");
        b = lwGet<double>(a);
        EXPECT_EQ(double(a), b);
    }

    // ----------------------------------------------------------------------------
    // Functions to initialize T_ELEM from unsigned
    // ----------------------------------------------------------------------------
    /**
     * \id lwGet_unsigned0
     * \brief validate the functionality of operator lwGet, colwert from unsigned to uint32_t
     * \depends None
     * \setup None
     * \testprocedure ./apiTest --gtest_filter=OPERATORS_H.lwGet_unsigned0
     * \teardown None
     * \testgroup OPERATORS_H
     * \inputs None
     * \outputs None
     * \expected States: the returned value is as expected
     */
    TEST(OPERATORS_H, lwGet_unsigned0)
    {
        unsigned a;
        uint32_t b;
        a = 1;
        callingInfo("lwGet<uint32_t>");
        b = lwGet<uint32_t>(a);
        EXPECT_EQ(uint32_t(a), b);
    }

    /**
     * \id lwGet_unsigned1
     * \brief validate the functionality of operator lwGet, colwert from unsigned to unsigned char
     * \depends None
     * \setup None
     * \testprocedure ./apiTest --gtest_filter=OPERATORS_H.lwGet_unsigned1
     * \teardown None
     * \testgroup OPERATORS_H
     * \inputs None
     * \outputs None
     * \expected States: the returned value is as expected
     */
    TEST(OPERATORS_H, lwGet_unsigned1)
    {
        unsigned a;
        unsigned char b;
        a = 1;
        callingInfo("lwGet<unsigned char>");
        b = lwGet<unsigned char>(a);
        EXPECT_EQ((unsigned char)a, b);
    }

    /**
     * \id lwGet_unsigned2
     * \brief validate the functionality of operator lwGet, colwert from unsigned to signed char
     * \depends None
     * \setup None
     * \testprocedure ./apiTest --gtest_filter=OPERATORS_H.lwGet_unsigned2
     * \teardown None
     * \testgroup OPERATORS_H
     * \inputs None
     * \outputs None
     * \expected States: the returned value is as expected
     */
    TEST(OPERATORS_H, lwGet_unsigned2)
    {
        unsigned a;
        signed char b;
        a = 1;
        callingInfo("lwGet<signed char>");
        b = lwGet<signed char>(a);
        EXPECT_EQ((signed char)a, b);
    }

    /**
     * \id lwGet_unsigned3
     * \brief validate the functionality of operator lwGet, colwert from unsigned to int
     * \depends None
     * \setup None
     * \testprocedure ./apiTest --gtest_filter=OPERATORS_H.lwGet_unsigned3
     * \teardown None
     * \testgroup OPERATORS_H
     * \inputs None
     * \outputs None
     * \expected States: the returned value is as expected
     */
    TEST(OPERATORS_H, lwGet_unsigned3)
    {
        unsigned a;
        int b;
        a = 1;
        callingInfo("lwGet<int>");
        b = lwGet<int>(a);
        EXPECT_EQ((int)a, b);
    }

    /**
     * \id lwGet_unsigned4
     * \brief validate the functionality of operator lwGet, colwert from unsigned to half
     * \depends None
     * \setup None
     * \testprocedure ./apiTest --gtest_filter=OPERATORS_H.lwGet_unsigned4
     * \teardown None
     * \testgroup OPERATORS_H
     * \inputs None
     * \outputs None
     * \expected States: the returned value is as expected
     */
    TEST(OPERATORS_H, lwGet_unsigned4)
    {
        unsigned a;
        half b;
        a = 1;
        callingInfo("lwGet<half>");
        b = lwGet<half>(a);
        EXPECT_TRUE(lwIsEqual(__float2half_rn(float(a)), b));
    }

    /**
     * \id lwGet_unsigned5
     * \brief validate the functionality of operator lwGet, colwert from unsigned to float
     * \depends None
     * \setup None
     * \testprocedure ./apiTest --gtest_filter=OPERATORS_H.lwGet_unsigned5
     * \teardown None
     * \testgroup OPERATORS_H
     * \inputs None
     * \outputs None
     * \expected States: the returned value is as expected
     */
    TEST(OPERATORS_H, lwGet_unsigned5)
    {
        unsigned a;
        float b;
        a = 1;
        callingInfo("lwGet<float>");
        b = lwGet<float>(a);
        EXPECT_EQ(float(a), b);
    }

    /**
     * \id lwGet_unsigned6
     * \brief validate the functionality of operator lwGet, colwert from unsigned to double
     * \depends None
     * \setup None
     * \testprocedure ./apiTest --gtest_filter=OPERATORS_H.lwGet_unsigned6
     * \teardown None
     * \testgroup OPERATORS_H
     * \inputs None
     * \outputs None
     * \expected States: the returned value is as expected
     */
    TEST(OPERATORS_H, lwGet_unsigned6)
    {
        unsigned a;
        double b;
        a = 1;
        callingInfo("lwGet<double>");
        b = lwGet<double>(a);
        EXPECT_EQ(double(a), b);
    }

    /**
     * \id lwGet_unsigned_char
     * \brief validate the functionality of operator lwGet, colwert from float to unsigned char
     * \depends None
     * \setup None
     * \testprocedure ./apiTest --gtest_filter=OPERATORS_H.lwGet_unsigned_char
     * \teardown None
     * \testgroup OPERATORS_H
     * \inputs None
     * \outputs None
     * \expected States: the returned value is as expected
     */
    TEST(OPERATORS_H, lwGet_unsigned_char)
    {
        float a = 1.1f;
        callingInfo("lwGet<unsigned char>");
        unsigned char b = lwGet<unsigned char>(a);
        EXPECT_EQ(b, (unsigned char)a);
    }

    // ----------------------------------------------------------------------------
    // Functions to initialize T_ELEM from float
    // ----------------------------------------------------------------------------
    /**
     * \id lwGet_float0
     * \brief validate the functionality of operator lwGet, colwert from float(negative) to signed char
     * \depends None
     * \setup None
     * \testprocedure ./apiTest --gtest_filter=OPERATORS_H.lwGet_float0
     * \teardown None
     * \testgroup OPERATORS_H
     * \inputs None
     * \outputs None
     * \expected States: the returned value is as expected
     */
    TEST(OPERATORS_H, lwGet_float0)
    {
        float a;
        signed char b;
        a = -129.1f;
        callingInfo("lwGet<signed char>");
        b = lwGet<signed char>(a);
        EXPECT_EQ((signed char)(-128), b);
    }

    /**
     * \id lwGet_float1
     * \brief validate the functionality of operator lwGet, colwert from float(positive) to signed char
     * \depends None
     * \setup None
     * \testprocedure ./apiTest --gtest_filter=OPERATORS_H.lwGet_float1
     * \teardown None
     * \testgroup OPERATORS_H
     * \inputs None
     * \outputs None
     * \expected States: the returned value is as expected
     */
    TEST(OPERATORS_H, lwGet_float1)
    {
        float a;
        signed char b;
        a = 128.1f;
        callingInfo("lwGet<signed char>");
        b = lwGet<signed char>(a);
        EXPECT_EQ((signed char)(127), b);
    }

    /**
     * \id lwGet_float2_0
     * \brief validate the functionality of operator lwGet, colwert from float to signed char
     * \depends None
     * \setup None
     * \testprocedure ./apiTest --gtest_filter=OPERATORS_H.lwGet_float2_0
     * \teardown None
     * \testgroup OPERATORS_H
     * \inputs None
     * \outputs None
     * \expected States: the returned value is as expected
     */
    TEST(OPERATORS_H, lwGet_float2_0)
    {
        float a;
        signed char b;
        a = 1.0f;
        callingInfo("lwGet<signed char>");
        b = lwGet<signed char>(a);
        EXPECT_EQ((signed char)1, b);
    }

    /**
     * \id lwGet_float2_1
     * \brief validate the functionality of operator lwGet, colwert from float to signed char
     * \depends None
     * \setup None
     * \testprocedure ./apiTest --gtest_filter=OPERATORS_H.lwGet_float2_1
     * \teardown None
     * \testgroup OPERATORS_H
     * \inputs None
     * \outputs None
     * \expected States: the returned value is as expected
     */
    TEST(OPERATORS_H, lwGet_float2_1)
    {
        float a;
        signed char b;
        a = 1.5f;
        callingInfo("lwGet<signed char>");
        b = lwGet<signed char>(a);
        EXPECT_EQ((signed char)(std::round( a / 2 ) * 2), b);
    }

    /**
     * \id lwGet_float3
     * \brief validate the functionality of operator lwGet, colwert from float to int
     * \depends None
     * \setup None
     * \testprocedure ./apiTest --gtest_filter=OPERATORS_H.lwGet_float3
     * \teardown None
     * \testgroup OPERATORS_H
     * \inputs None
     * \outputs None
     * \expected States: the returned value is as expected
     */
    TEST(OPERATORS_H, lwGet_float3)
    {
        float a;
        int b;
        a = -1.2f;
        callingInfo("lwGet<int>");
        b = lwGet<int>(a);
        EXPECT_EQ(int(a), b);
    }

    /**
     * \id lwGet_float4
     * \brief validate the functionality of operator lwGet, colwert from float to unsigned
     * \depends None
     * \setup None
     * \testprocedure ./apiTest --gtest_filter=OPERATORS_H.lwGet_float4
     * \teardown None
     * \testgroup OPERATORS_H
     * \inputs None
     * \outputs None
     * \expected States: the returned value is as expected
     */
    TEST(OPERATORS_H, lwGet_float4)
    {
        float a;
        unsigned b;
        a = -1.2f;
        callingInfo("lwGet<unsigned>");
        b = lwGet<unsigned>(a);
        EXPECT_EQ(unsigned(a), b);
    }

    /**
     * \id lwGet_float5
     * \brief validate the functionality of operator lwGet, colwert from float to half
     * \depends None
     * \setup None
     * \testprocedure ./apiTest --gtest_filter=OPERATORS_H.lwGet_float5
     * \teardown None
     * \testgroup OPERATORS_H
     * \designid LWTENSOR_DES_013
     * \testgroup OPERATORS_H
     * \inputs None
     * \outputs None
     * \expected States: the returned value is as expected
     */
    TEST(OPERATORS_H, lwGet_float5)
    {
        float a;
        half b;
        a = -1.2f;
        callingInfo("lwGet<half>");
        b = lwGet<half>(a);
        EXPECT_TRUE(lwIsEqual(__float2half_rn(float(a)), b));
    }

    /**
     * \id lwGet_float6
     * \brief validate the functionality of operator lwGet, colwert from float to float
     * \depends None
     * \setup None
     * \testprocedure ./apiTest --gtest_filter=OPERATORS_H.lwGet_float6
     * \teardown None
     * \designid LWTENSOR_DES_013
     * \testgroup OPERATORS_H
     * \inputs None
     * \outputs None
     * \expected States: the returned value is as expected
     */
    TEST(OPERATORS_H, lwGet_float6)
    {
        float a;
        float b;
        a = -1.2f;
        callingInfo("lwGet<float>");
        b = lwGet<float>(a);
        EXPECT_EQ(a, b);
    }

    /**
     * \id lwGet_float7
     * \brief validate the functionality of operator lwGet, colwert from float to double
     * \depends None
     * \setup None
     * \testprocedure ./apiTest --gtest_filter=OPERATORS_H.lwGet_float7
     * \teardown None
     * \testgroup OPERATORS_H
     * \inputs None
     * \outputs None
     * \expected States: the returned value is as expected
     */
    TEST(OPERATORS_H, lwGet_float7)
    {
        float a;
        double b;
        a = -1.2f;
        callingInfo("lwGet<double>");
        b = lwGet<double>(a);
        EXPECT_EQ(double(a), b);
    }

    // ----------------------------------------------------------------------------
    // Functions to initialize T_ELEM from half
    // ----------------------------------------------------------------------------
    /**
     * \id lwGet_half0
     * \brief validate the functionality of operator lwGet, colwert from half to int
     * \depends None
     * \setup None
     * \testprocedure ./apiTest --gtest_filter=OPERATORS_H.lwGet_half0
     * \teardown None
     * \designid LWTENSOR_DES_013
     * \testgroup OPERATORS_H
     * \inputs None
     * \outputs None
     * \expected States: the returned value is as expected
     */
    TEST(OPERATORS_H, lwGet_half0)
    {
        half a;
        int  b;
        a = __float2half(1);
        callingInfo("lwGet<int>");
        b = lwGet<int>(a);
        EXPECT_EQ(1, b);
    }

    /**
     * \id lwGet_half1
     * \brief validate the functionality of operator lwGet, colwert from half to uint32_t
     * \depends None
     * \setup None
     * \testprocedure ./apiTest --gtest_filter=OPERATORS_H.lwGet_half1
     * \teardown None
     * \testgroup OPERATORS_H
     * \inputs None
     * \outputs None
     * \expected States: the returned value is as expected
     */
    TEST(OPERATORS_H, lwGet_half1)
    {
        half a;
        uint32_t b;
        a = __float2half(1);
        callingInfo("lwGet<uint32_t>");
        b = lwGet<uint32_t>(a);
        EXPECT_EQ(int(__half2float(a)), b);
    }

    /**
     * \id lwGet_half2
     * \brief validate the functionality of operator lwGet, colwert from half to half
     * \depends None
     * \setup None
     * \testprocedure ./apiTest --gtest_filter=OPERATORS_H.lwGet_half2
     * \teardown None
     * \designid LWTENSOR_DES_013
     * \testgroup OPERATORS_H
     * \inputs None
     * \outputs None
     * \expected States: the returned value is as expected
     */
    TEST(OPERATORS_H, lwGet_half2)
    {
        half a;
        half b;
        a = __float2half(1);
        callingInfo("lwGet<half>");
        b = lwGet<half>(a);
        EXPECT_TRUE(lwIsEqual(a, b));
    }

    /**
     * \id lwGet_half3
     * \brief validate the functionality of operator lwGet, colwert from half to float
     * \depends None
     * \setup None
     * \testprocedure ./apiTest --gtest_filter=OPERATORS_H.lwGet_half3
     * \teardown None
     * \designid LWTENSOR_DES_013
     * \testgroup OPERATORS_H
     * \inputs None
     * \outputs None
     * \expected States: the returned value is as expected
     */
    TEST(OPERATORS_H, lwGet_half3)
    {
        half a;
        float b;
        a = __float2half(1);
        callingInfo("lwGet<float>");
        b = lwGet<float>(a);
        EXPECT_EQ(__half2float(a), b);
    }

    /**
     * \id lwGet_half4
     * \brief validate the functionality of operator lwGet, colwert from half to signed char
     * \depends None
     * \setup None
     * \testprocedure ./apiTest --gtest_filter=OPERATORS_H.lwGet_half4
     * \teardown None
     * \testgroup OPERATORS_H
     * \inputs None
     * \outputs None
     * \expected States: the returned value is as expected
     */
    TEST(OPERATORS_H, lwGet_half4)
    {
        half a;
        signed char b;
        a = __float2half(1);
        callingInfo("lwGet<siged char>");
        b = lwGet<signed char>(a);
        EXPECT_EQ(lwGet<signed char>(lwGet<float>(a)), b);
    }

    //TODO: seems there is no function to colwert float to unsigned char
    /**
     * \id lwGet_half5
     * \brief validate the functionality of operator lwGet, colwert from half to unsigned char
     * \depends None
     * \setup None
     * \testprocedure ./apiTest --gtest_filter=OPERATORS_H.lwGet_half5
     * \teardown None
     * \testgroup OPERATORS_H
     * \inputs None
     * \outputs None
     * \expected States: the returned value is as expected
     */
    TEST(OPERATORS_H, lwGet_half5)
    {
        half a;
        unsigned char b;
        unsigned char b_ref;
        a = __float2half(513);
        callingInfo("lwGet<unsiged char>");
        b = lwGet<unsigned char>(a);
        callingInfo("lwGet<float>");
        b_ref = (unsigned char)(lwGet<float>(a));
        printf("unsigned char: %d\n", b);
        EXPECT_EQ(memcmp(&b_ref, &b, sizeof(b)), 0);
    }

    /**
     * \id lwGet_half6
     * \brief validate the functionality of operator lwGet, colwert from half to double
     * \depends None
     * \setup None
     * \testprocedure ./apiTest --gtest_filter=OPERATORS_H.lwGet_half6
     * \teardown None
     * \testgroup OPERATORS_H
     * \inputs None
     * \outputs None
     * \expected States: the returned value is as expected
     */
    TEST(OPERATORS_H, lwGet_half6)
    {
        half a;
        double b;
        a = __float2half(1);
        callingInfo("lwGet<double>");
        b = lwGet<double>(a);
        EXPECT_EQ(double(lwGet<float>(a)), b);
    }

    // ----------------------------------------------------------------------------
    // Functions to initialize T_ELEM from double
    // ----------------------------------------------------------------------------
    /**
     * \id lwGet_double0
     * \brief validate the functionality of operator lwGet, colwert from double(negative) to signed char
     * \depends None
     * \setup None
     * \testprocedure ./apiTest --gtest_filter=OPERATORS_H.lwGet_double0
     * \teardown None
     * \testgroup OPERATORS_H
     * \inputs None
     * \outputs None
     * \expected States: the returned value is as expected
     */
    TEST(OPERATORS_H, lwGet_double0)
    {
        double a;
        signed char b;
        a = -129.1;
        callingInfo("lwGet<signed char>");
        b = lwGet<signed char>(a);
        EXPECT_EQ((signed char)(-128), b);
    }

    /**
     * \id lwGet_double1
     * \brief validate the functionality of operator lwGet, colwert from double(postive) to signed char
     * \depends None
     * \setup None
     * \testprocedure ./apiTest --gtest_filter=OPERATORS_H.lwGet_double1
     * \teardown None
     * \testgroup OPERATORS_H
     * \inputs None
     * \outputs None
     * \expected States: the returned value is as expected
     */
    TEST(OPERATORS_H, lwGet_double1)
    {
        double a;
        signed char b;
        a = 128.1;
        callingInfo("lwGet<signed char>");
        b = lwGet<signed char>(a);
        EXPECT_EQ((signed char)(127), b);
    }

    /**
     * \id lwGet_double2
     * \brief validate the functionality of operator lwGet, colwert from double to signed char
     * \depends None
     * \setup None
     * \testprocedure ./apiTest --gtest_filter=OPERATORS_H.lwGet_double2
     * \teardown None
     * \testgroup OPERATORS_H
     * \inputs None
     * \outputs None
     * \expected States: the returned value is as expected
     */
    TEST(OPERATORS_H, lwGet_double2)
    {
        double a;
        signed char b;
        a = 1.0;
        callingInfo("lwGet<signed char>");
        b = lwGet<signed char>(a);
        EXPECT_EQ((signed char)1, b);
    }

    /**
     * \id lwGet_double3
     * \brief validate the functionality of operator lwGet, colwert from double to unsigned char
     * \depends None
     * \setup None
     * \testprocedure ./apiTest --gtest_filter=OPERATORS_H.lwGet_double3
     * \teardown None
     * \testgroup OPERATORS_H
     * \inputs None
     * \outputs None
     * \expected States: the returned value is as expected
     */
    TEST(OPERATORS_H, lwGet_double3)
    {
        double a;
        unsigned char b;
        a = -1.0;
        callingInfo("lwGet<unsigned char>");
        b = lwGet<unsigned char>(a);
        EXPECT_EQ((unsigned char)0, b);
    }

    /**
     * \id lwGet_double4
     * \brief validate the functionality of operator lwGet, colwert from double to unsigned char
     * \depends None
     * \setup None
     * \testprocedure ./apiTest --gtest_filter=OPERATORS_H.lwGet_double4
     * \teardown None
     * \testgroup OPERATORS_H
     * \inputs None
     * \outputs None
     * \expected States: the returned value is as expected
     */
    TEST(OPERATORS_H, lwGet_double4)
    {
        double a;
        unsigned char b;
        a = 256.5;
        callingInfo("lwGet<unsigned char>");
        b = lwGet<unsigned char>(a);
        EXPECT_EQ((unsigned char)UCHAR_MAX, b);
    }

    /**
     * \id lwGet_double5
     * \brief validate the functionality of operator lwGet, colwert from double to unsigned char
     * \depends None
     * \setup None
     * \testprocedure ./apiTest --gtest_filter=OPERATORS_H.lwGet_double5
     * \teardown None
     * \testgroup OPERATORS_H
     * \inputs None
     * \outputs None
     * \expected States: the returned value is as expected
     */
    TEST(OPERATORS_H, lwGet_double5)
    {
        double a;
        unsigned char b;
        a = 1.1;
        callingInfo("lwGet<unsigned char>");
        b = lwGet<unsigned char>(a);
        EXPECT_EQ((unsigned char)(a), b);
    }

    /**
     * \id lwGet_double6
     * \brief validate the functionality of operator lwGet, colwert from double to int
     * \depends None
     * \setup None
     * \testprocedure ./apiTest --gtest_filter=OPERATORS_H.lwGet_double6
     * \teardown None
     * \testgroup OPERATORS_H
     * \inputs None
     * \outputs None
     * \expected States: the returned value is as expected
     */
    TEST(OPERATORS_H, lwGet_double6)
    {
        double a;
        int b;
        a = -1.2;
        callingInfo("lwGet<int>");
        b = lwGet<int>(a);
        EXPECT_EQ(int(a), b);
    }

    /**
     * \id lwGet_double7
     * \brief validate the functionality of operator lwGet, colwert from double to unsigned
     * \depends None
     * \setup None
     * \testprocedure ./apiTest --gtest_filter=OPERATORS_H.lwGet_double7
     * \teardown None
     * \testgroup OPERATORS_H
     * \inputs None
     * \outputs None
     * \expected States: the returned value is as expected
     */
    TEST(OPERATORS_H, lwGet_double7)
    {
        double a;
        unsigned b;
        a = 1.2;
        callingInfo("lwGet<unsigned>");
        b = lwGet<unsigned>(a);
        EXPECT_EQ(unsigned(a), b);
    }

    /**
     * \id lwGet_double8
     * \brief validate the functionality of operator lwGet, colwert from double to half
     * \depends None
     * \setup None
     * \testprocedure ./apiTest --gtest_filter=OPERATORS_H.lwGet_double8
     * \teardown None
     * \testgroup OPERATORS_H
     * \inputs None
     * \outputs None
     * \expected States: the returned value is as expected
     */
    TEST(OPERATORS_H, lwGet_double8)
    {
        double a;
        half b;
        a = -1.2;
        callingInfo("lwGet<half>");
        b = lwGet<half>(a);
        EXPECT_TRUE(lwIsEqual(__float2half_rn(float(a)), b));
    }

    /**
     * \id lwGet_double9
     * \brief validate the functionality of operator lwGet, colwert from double to float
     * \depends None
     * \setup None
     * \testprocedure ./apiTest --gtest_filter=OPERATORS_H.lwGet_double9
     * \teardown None
     * \testgroup OPERATORS_H
     * \inputs None
     * \outputs None
     * \expected States: the returned value is as expected
     */
    TEST(OPERATORS_H, lwGet_double9)
    {
        double a;
        float b;
        a = -1.2;
        callingInfo("lwGet<float>");
        b = lwGet<float>(a);
        EXPECT_EQ(float(a), b);
    }

    /**
     * \id lwGet_double10
     * \brief validate the functionality of operator lwGet, colwert from double to double
     * \depends None
     * \setup None
     * \testprocedure ./apiTest --gtest_filter=OPERATORS_H.lwGet_double10
     * \teardown None
     * \testgroup OPERATORS_H
     * \inputs None
     * \outputs None
     * \expected States: the returned value is as expected
     */
    TEST(OPERATORS_H, lwGet_double10)
    {
        double a;
        double b;
        a = -1.2;
        callingInfo("lwGet<double>");
        b = lwGet<double>(a);
        EXPECT_EQ(a, b);
    }

    /**
     * \id lwGet_lwComplex
     * \brief validate the functionality of operator lwGet
     * \depends None
     * \setup None
     * \testprocedure ./apiTest --gtest_filter=OPERATORS_H.lwGet_lwComplex
     * \teardown None
     * \testgroup OPERATORS_H
     * \inputs None
     * \outputs None
     * \expected States: the returned value is as expected
     */
    TEST(OPERATORS_H, lwGet_lwComplex)
    {
        lwComplex x0 = make_lwComplex(1.0f, 2.0f);
        callingInfo("lwGet<lwComplex>");
        lwComplex y0 = lwGet<lwComplex>(x0);
        EXPECT_TRUE(lwIsEqual(x0, y0));

        lwComplex x1 = make_lwComplex(1.0f, 2.0f);
        callingInfo("lwGet<int8_t>");
        int8_t y1 = lwGet<int8_t>(x1);
        int8_t r1= lwGet<int8_t>(x1.x);
        EXPECT_TRUE(lwIsEqual(y1, r1));

        int8_t x2 = lwGet<int8_t>(1.5f);
        callingInfo("lwGet<int8_t>");
        lwComplex y2 = lwGet<lwComplex>(x2);
        lwComplex r2 = make_lwComplex(lwGet<float>(x2), 0.0);
        EXPECT_TRUE(lwIsEqual(y2, r2));

        lwComplex x3 = make_lwComplex(1.0f, 2.0f);
        callingInfo("lwGet<int8_t>");
        float y3 = lwGet<float>(x3);
        float r3 = x3.x;
        EXPECT_FLOAT_EQ(y3, r3);

        float x4 = 1.5f;
        callingInfo("lwGet<int8_t>");
        lwComplex y4 = lwGet<lwComplex>(x4);
        lwComplex r4 = make_lwComplex(x4, 0.0);
        EXPECT_TRUE(lwIsEqual(y4, r4));

        int x5 = 3;
        callingInfo("lwGet<int8_t>");
        lwComplex y5 = lwGet<lwComplex>(x5);
        lwComplex r5 = make_lwComplex(lwGet<float>(x5), 0.0);
        EXPECT_TRUE(lwIsEqual(y5, r5));

        lwComplex a0 = make_lwComplex(1.0f, 2.0f);
        lwComplex b0 = make_lwComplex(1.0f, 2.0f);
        callingInfo("lwIsEqual<lwComplex>");
        EXPECT_TRUE(lwIsEqual(a0, b0));

        lwComplex b1 = make_lwComplex(2.0f, 1.0f);
        callingInfo("lwIsEqual<lwComplex>");
        EXPECT_FALSE(lwIsEqual(a0, b1));

        callingInfo("lwMul<lwComplex>");
        lwComplex c = lwMul(a0, b0);
        lwComplex cr = lwCmulf(a0, b0);
        EXPECT_TRUE(lwIsEqual(c, cr));

        callingInfo("lwAdd<lwComplex>");
        lwComplex e = lwAdd(a0, b0);
        lwComplex er = make_lwComplex(a0.x + b0.x, a0.y + b0.y);
        EXPECT_TRUE(lwIsEqual(e, er));

        callingInfo("lwSub<lwComplex>");
        lwComplex f = lwSub(a0, b0);
        lwComplex fr = make_lwComplex(a0.x - b0.x, a0.y - b0.y);
        EXPECT_TRUE(lwIsEqual(f, fr));
    }

    /**
     * \id lwSquare2Norm
     * \brief validate the functionality of operator lwSquare2Norm
     * \depends None
     * \setup None
     * \testprocedure ./apiTest --gtest_filter=OPERATORS_H.lwSquare2Norm
     * \teardown None
     * \testgroup OPERATORS_H
     * \inputs None
     * \outputs None
     * \expected States: the returned value is as expected
     */
    TEST(OPERATORS_H, lwSquare2Norm)
    {
        half a = lwGet<half>(1.2f);
        callingInfo("lwSquare2Norm<half>");
        double b = lwSquare2Norm(a);
        double br = lwMul(lwGet<double>(a), lwGet<double>(a));
        EXPECT_DOUBLE_EQ(b, br);

        float c = 1.2f;
        callingInfo("lwSquare2Norm<float>");
        double d = lwSquare2Norm(c);
        double dr = (double)(c * c);
        EXPECT_DOUBLE_EQ(d, dr);

        double e = 1.2;
        callingInfo("lwSquare2Norm<double>");
        double f = lwSquare2Norm(e);
        double fr = (double)(e * e);
        EXPECT_DOUBLE_EQ(f, fr);
    }

    /**
     * \id math_double
     * \brief validate the functionality of operators
     * \depends None
     * \setup None
     * \testprocedure ./apiTest --gtest_filter=OPERATORS_H.math_double
     * \teardown None
     * \testgroup OPERATORS_H
     * \inputs None
     * \outputs None
     * \expected States: the returned value is as expected
     */
    TEST(OPERATORS_H, math_double)
    {
        double x = 0.2; //should be [-1, +1] for some cases
        callingInfo("lwSin<double>");
        double y0 = lwSin(x);
        double r0 = sin(x);
        EXPECT_DOUBLE_EQ(y0, r0);

        callingInfo("lwCos<double>");
        double y1 = lwCos(x);
        double r1 = cos(x);
        EXPECT_DOUBLE_EQ(y1, r1);

        callingInfo("lwTan<double>");
        double y2 = lwTan(x);
        double r2 = tan(x);
        EXPECT_DOUBLE_EQ(y2, r2);

        callingInfo("lwSinh<double>");
        double y3 = lwSinh(x);
        double r3 = sinh(x);
        EXPECT_DOUBLE_EQ(y3, r3);

        callingInfo("lwCosh<double>");
        double y4 = lwCosh(x);
        double r4 = cosh(x);
        EXPECT_DOUBLE_EQ(y4, r4);

        callingInfo("lwTanh<double>");
        double y5 = lwTanh(x);
        double r5 = tanh(x);
        EXPECT_DOUBLE_EQ(y5, r5);

        callingInfo("lwAsin<double>");
        double y6 = lwAsin(x);
        double r6 = asin(x);
        EXPECT_DOUBLE_EQ(y6, r6);

        callingInfo("lwAcos<double>");
        double y7 = lwAcos(x);
        double r7 = acos(x);
        EXPECT_DOUBLE_EQ(y7, r7);

        callingInfo("lwAtan<double>");
        double y8 = lwAtan(x);
        double r8 = atan(x);
        EXPECT_DOUBLE_EQ(y8, r8);

        callingInfo("lwAsinh<double>");
        double y9 = lwAsinh(x);
        double r9 = asinh(x);
        EXPECT_DOUBLE_EQ(y9, r9);

        double x10 = 2.3;
        callingInfo("lwAcosh<double>");
        double y10 = lwAcosh(x10);
        double r10 = acosh(x10);
        EXPECT_DOUBLE_EQ(y10, r10);

        callingInfo("lwAtanh<double>");
        double y11 = lwAtanh(x);
        double r11 = atanh(x);
        EXPECT_DOUBLE_EQ(y11, r11);
    }

    /**
     * \id lwAbs_unsigned
     * \brief validate the functionality of operator lwAbs
     * \depends None
     * \setup None
     * \testprocedure ./apiTest --gtest_filter=OPERATORS_H.lwAbs_unsigned
     * \teardown None
     * \testgroup OPERATORS_H
     * \inputs None
     * \outputs None
     * \expected States: TRUE
     */
    TEST(OPERATORS_H, lwAbs_unsigned)
    {
        unsigned int x = 1U;
        callingInfo("lwAbs<unsigned int>");
        EXPECT_EQ(lwAbs(x), x);
    }

    /**
     * \id lwIsEqualTrue
     * \brief validate the functionality of operator lwIsEqual with type int
     * \depends None
     * \setup None
     * \testprocedure ./apiTest --gtest_filter=OPERATORS_H.lwIsEqualTrue
     * \teardown None
     * \testgroup OPERATORS_H
     * \inputs None
     * \outputs None
     * \expected States: TRUE
     */
    TEST(OPERATORS_H, lwIsEqualTrue)
    {
        callingInfo("lwIsEqual<int>");
        EXPECT_TRUE(lwIsEqual<int>(1, 1));
    }

    /**
     * \id lwIsEqualTrue
     * \brief validate the functionality of operator lwIsEqual with type int
     * \depends None
     * \setup None
     * \testprocedure ./apiTest --gtest_filter=OPERATORS_H.lwIsEqualTrue
     * \teardown None
     * \testgroup OPERATORS_H
     * \inputs None
     * \outputs None
     * \expected States: FALSE
     */
    TEST(OPERATORS_H, lwIsEqualFalse)
    {
        callingInfo("lwIsEqual<int>");
        EXPECT_EQ(lwIsEqual<int>(1, 2), false);
    }

    /**
     * \id lwAddInt
     * \brief validate the functionality of operator lwAdd with type int
     * \depends None
     * \setup None
     * \testprocedure ./apiTest --gtest_filter=OPERATORS_H.lwAddInt
     * \teardown None
     * \designid LWTENSOR_DES_016
     * \testgroup OPERATORS_H
     * \inputs None
     * \outputs None
     * \expected States: the expected value
     */
    TEST(OPERATORS_H, lwAddInt)
    {
        callingInfo("lwAdd<int>");
        EXPECT_EQ(lwAdd<int>(1, 2), 3);
    }

    /**
     * \id operators_double
     * \brief validate the functionality of operators lwRcp, lwExp, lwLn
     * \depends None
     * \setup None
     * \testprocedure ./apiTest --gtest_filter=OPERATORS_H.operators_double
     * \teardown None
     * \designid LWTENSOR_DES_015
     * \testgroup OPERATORS_H
     * \inputs None
     * \outputs None
     * \expected States: the expected value
     */
    TEST(OPERATORS_H, operators_double)
    {
        double ti = 1.23456;
        callingInfo("lwLn<double>");
        EXPECT_FLOAT_EQ(lwLn(ti), log(ti));

        callingInfo("lwExp<double>");
        EXPECT_FLOAT_EQ(lwExp(ti), exp(ti));

        callingInfo("lwRcp<double>");
        EXPECT_FLOAT_EQ(lwRcp(ti), 1.0/ti);
    }

    /**
     * \id lwtensorUnaryOpfloat
     * \brief validate the functionality of operator lwtensorUnaryOp with type float
     * \depends None
     * \setup None
     * \testprocedure ./apiTest --gtest_filter=OPERATORS_H.lwtensorUnaryOpfloat
     * \teardown None
     * \designid LWTENSOR_DES_015, LWTENSOR_DES_018
     * \testgroup OPERATORS_H
     * \inputs None
     * \outputs None
     * \expected States: the expected value
     */
    TEST(OPERATORS_H, lwtensorUnaryOpfloat)
    {
        ElementwiseParameters::ActivationContext ctx;

        float ti = UniformRandomNumber<float>(0.5, 10.0);
        callingInfo("lwtensorUnaryOp<float>");
        float t1 = lwtensorUnaryOp<float>(ti, LWTENSOR_OP_IDENTITY, &ctx);
        EXPECT_TRUE(lwIsEqual(t1, ti));

        callingInfo("lwtensorUnaryOp<float>");
        float t2 = lwtensorUnaryOp<float>(ti, LWTENSOR_OP_SQRT, &ctx);
        // EXPECT_TRUE(lwIsEqual(t2, ti * ti));
        EXPECT_TRUE(lwIsEqual(t2, std::sqrt(ti)));

        callingInfo("lwtensorUnaryOp<float>");
        float t3 = lwtensorUnaryOp<float>(ti, LWTENSOR_OP_RELU, &ctx);
        EXPECT_TRUE(lwIsEqual(t3, ti > 0.f ? ti : 0.f));

        callingInfo("lwtensorUnaryOp<float>");
        float t4 = lwtensorUnaryOp<float>(ti, LWTENSOR_OP_CONJ, &ctx);
        EXPECT_TRUE(lwIsEqual(t4, ti));

        callingInfo("lwtensorUnaryOp<float>");
        float t7 = lwtensorUnaryOp<float>(ti, LWTENSOR_OP_SIGMOID, &ctx);
        EXPECT_TRUE(lwIsEqual(t7, 0.5f * (tanhf(0.5f * ti) + 1.f)));

        callingInfo("lwtensorUnaryOp<float>");
        float t8 = lwtensorUnaryOp<float>(ti, LWTENSOR_OP_TANH, &ctx);
        EXPECT_TRUE(lwIsEqual(t8, tanhf(ti)));

        // callingInfo("lwtensorUnaryOp<float>");
        // ctx.elu.alpha = 0.1f;
        // float t9 = lwtensorUnaryOp<float>(ti, LWTENSOR_OP_ELU, &ctx);
        // EXPECT_TRUE(lwIsEqual(t9, ti >= 0.f ? ti : ctx.elu.alpha * (expf(ti) - 1.f)));

        // callingInfo("lwtensorUnaryOp<float>");
        // ctx.leakyRelu.k = 0.1f;
        // float t10 = lwtensorUnaryOp<float>(ti, LWTENSOR_OP_LEAKY_RELU, &ctx);
        // EXPECT_TRUE(lwIsEqual(t10, ti >= 0.f ? ti : ctx.leakyRelu.k * ti));

        // callingInfo("lwtensorUnaryOp<float>");
        // ctx.softPlus.inScale = 2.f;
        // ctx.softPlus.outScale = 0.8f;
        // ctx.softPlus.approximateThreshold = 20.f;
        // float t12 = lwtensorUnaryOp<float>(ti, LWTENSOR_OP_SOFT_PLUS, &ctx);
        // EXPECT_TRUE(lwIsEqual(t12, ctx.softPlus.outScale * (logf(1.f + expf(ctx.softPlus.inScale * ti)))));

        // callingInfo("lwtensorUnaryOp<float>");
        // float t13 = lwtensorUnaryOp<float>(ti, LWTENSOR_OP_SOFT_SIGN, &ctx);
        // EXPECT_TRUE(lwIsEqual(t13, ti / (1.f + abs(ti))));

        // callingInfo("lwtensorUnaryOp<float>");
        // ctx.selu.eluAlpha = 0.1f;
        // ctx.selu.outScale = 0.5f;
        // float t14 = lwtensorUnaryOp<float>(ti, LWTENSOR_OP_SELU, &ctx);
        // EXPECT_TRUE(lwIsEqual(t14, ctx.selu.outScale * (ti >= 0.f ? ti : ctx.selu.eluAlpha * (expf(ti) - 1))));

        // callingInfo("lwtensorUnaryOp<float>");
        // ctx.hardSigmoid.slope = 0.5f;
        // ctx.hardSigmoid.shift = 0.5f;
        // float t15 = lwtensorUnaryOp<float>(ti, LWTENSOR_OP_HARD_SIGMOID, &ctx);
        // EXPECT_TRUE(lwIsEqual(t15, std::min(1.f, std::max(0.f, ctx.hardSigmoid.slope * ti + 0.5f))));

        // callingInfo("lwtensorUnaryOp<float>");
        // ctx.scaledTanh.inScale = 0.5f;
        // ctx.scaledTanh.outScale = 1.5f;
        // float t16 = lwtensorUnaryOp<float>(ti, LWTENSOR_OP_SCALED_TANH, &ctx);
        // EXPECT_TRUE(lwIsEqual(t16, ctx.scaledTanh.outScale * (tanhf(ctx.scaledTanh.inScale * ti))));

        // callingInfo("lwtensorUnaryOp<float>");
        // ctx.thresholdedRelu.threshold = 1.5f;
        // float t17 = lwtensorUnaryOp<float>(ti, LWTENSOR_OP_THRESHOLDED_RELU, &ctx);
        // EXPECT_TRUE(lwIsEqual(t17, ti > ctx.thresholdedRelu.threshold ? ti : 0.f));

        // callingInfo("lwtensorUnaryOp<float>");
        // ctx.clip.lower = 0.5f;
        // ctx.clip.upper = 1.5f;
        // float t18 = lwtensorUnaryOp<float>(ti, LWTENSOR_OP_CLIP, &ctx);
        // EXPECT_TRUE(lwIsEqual(t18, std::max(std::min(ti, ctx.clip.upper), ctx.clip.lower)));

        callingInfo("lwtensorUnaryOp<float>");
        ctx.clip.lower = 0.5f;
        ctx.clip.upper = 1.5f;
        float t19 = lwtensorUnaryOp<float>(ti, LWTENSOR_OP_UNKNOWN, &ctx);
        EXPECT_TRUE(lwIsEqual(t19, ti));

        // New operators for TRT unary runner.
        callingInfo("lwtensorUnaryOp<float>");
        float t20 = lwtensorUnaryOp<float>(ti, LWTENSOR_OP_EXP, &ctx);
        EXPECT_TRUE(lwIsEqual(t20, std::exp(ti)));

        callingInfo("lwtensorUnaryOp<float>");
        float t21 = lwtensorUnaryOp<float>(ti, LWTENSOR_OP_LOG, &ctx);
        EXPECT_TRUE(lwIsEqual(t21, std::log(ti)));

        callingInfo("lwtensorUnaryOp<float>");
        float t22 = lwtensorUnaryOp<float>(ti, LWTENSOR_OP_ABS, &ctx);
        EXPECT_TRUE(lwIsEqual(t22, std::abs(ti)));

        callingInfo("lwtensorUnaryOp<float>");
        float t23 = lwtensorUnaryOp<float>(ti, LWTENSOR_OP_NEG, &ctx);
        EXPECT_TRUE(lwIsEqual(t23, -ti));

        callingInfo("lwtensorUnaryOp<float>");
        float t24 = lwtensorUnaryOp<float>(ti, LWTENSOR_OP_SIN, &ctx);
        EXPECT_TRUE(lwIsEqual(t24, std::sin(ti)));

        callingInfo("lwtensorUnaryOp<float>");
        float t25 = lwtensorUnaryOp<float>(ti, LWTENSOR_OP_COS, &ctx);
        EXPECT_TRUE(lwIsEqual(t25, std::cos(ti)));

        callingInfo("lwtensorUnaryOp<float>");
        float t26 = lwtensorUnaryOp<float>(ti, LWTENSOR_OP_TAN, &ctx);
        EXPECT_TRUE(lwIsEqual(t26, std::tan(ti)));

        callingInfo("lwtensorUnaryOp<float>");
        float t27 = lwtensorUnaryOp<float>(ti, LWTENSOR_OP_SINH, &ctx);
        EXPECT_TRUE(lwIsEqual(t27, std::sinh(ti)));

        callingInfo("lwtensorUnaryOp<float>");
        float t28 = lwtensorUnaryOp<float>(ti, LWTENSOR_OP_COSH, &ctx);
        EXPECT_TRUE(lwIsEqual(t28, std::cosh(ti)));

        callingInfo("lwtensorUnaryOp<float>");
        float t29 = lwtensorUnaryOp<float>(ti, LWTENSOR_OP_TANH, &ctx);
        EXPECT_TRUE(lwIsEqual(t29, std::tanh(ti)));

        auto clippedTi = std::min(std::max(ti, -1.f), 1.f);

        callingInfo("lwtensorUnaryOp<float>");
        float t30 = lwtensorUnaryOp<float>(clippedTi, LWTENSOR_OP_ASIN, &ctx);
        EXPECT_TRUE(lwIsEqual(t30, std::asin(clippedTi)));

        callingInfo("lwtensorUnaryOp<float>");
        float t31 = lwtensorUnaryOp<float>(clippedTi, LWTENSOR_OP_ACOS, &ctx);
        EXPECT_TRUE(lwIsEqual(t31, std::acos(clippedTi)));

        callingInfo("lwtensorUnaryOp<float>");
        float t32 = lwtensorUnaryOp<float>(clippedTi, LWTENSOR_OP_ATAN, &ctx);
        EXPECT_TRUE(lwIsEqual(t32, std::atan(clippedTi)));

        callingInfo("lwtensorUnaryOp<float>");
        float t33 = lwtensorUnaryOp<float>(clippedTi, LWTENSOR_OP_ASINH, &ctx);
        EXPECT_TRUE(lwIsEqual(t33, std::asinh(clippedTi)));

        callingInfo("lwtensorUnaryOp<float>");
        auto clippedAcoshTi = std::max(ti, 2.3f);
        float t34 = lwtensorUnaryOp<float>(clippedAcoshTi, LWTENSOR_OP_ACOSH, &ctx);
        EXPECT_TRUE(lwIsEqual(t34, std::acosh(clippedAcoshTi)));

        callingInfo("lwtensorUnaryOp<float>");
        float t35 = lwtensorUnaryOp<float>(clippedTi, LWTENSOR_OP_ATANH, &ctx);
        EXPECT_TRUE(lwIsEqual(t35, std::atanh(clippedTi)));

        callingInfo("lwtensorUnaryOp<float>");
        float t36 = lwtensorUnaryOp<float>(ti, LWTENSOR_OP_CEIL, &ctx);
        EXPECT_FLOAT_EQ(t36, std::ceil(ti));

        callingInfo("lwtensorUnaryOp<float>");
        float t37 = lwtensorUnaryOp<float>(ti, LWTENSOR_OP_FLOOR, &ctx);
        EXPECT_FLOAT_EQ(t37, std::floor(ti));

        callingInfo("lwtensorUnaryOp<float>");
        float t38 = lwtensorUnaryOp<float>(ti, LWTENSOR_OP_ADD, &ctx); // LWTENSOR_OP_ADD is unsupported
        EXPECT_FLOAT_EQ(t38, ti);
    }

    /**
     * \id lwtensorUnaryOp_lwComplex
     * \brief validate the functionality of operator lwtensorUnaryOp with type lwComplex
     * \depends None
     * \setup None
     * \testprocedure ./apiTest --gtest_filter=OPERATORS_H.lwtensorUnaryOp_lwComplex
     * \teardown None
     * \designid LWTENSOR_DES_015, LWTENSOR_DES_018
     * \testgroup OPERATORS_H
     * \inputs None
     * \outputs None
     * \expected States: the expected value
     */
    TEST(OPERATORS_H, lwtensorUnaryOp_lwComplex)
    {
        ElementwiseParameters::ActivationContext ctx;
        lwComplex x = make_lwComplex(1.2f, 3.4f);
        lwComplex y0 = lwtensorUnaryOp(x, LWTENSOR_OP_ATANH, &ctx);
        EXPECT_TRUE(lwIsEqual(y0, x));

        lwComplex y1 = lwtensorUnaryOp(x, LWTENSOR_OP_SQRT, &ctx);
        EXPECT_TRUE(lwIsEqual(y1, x));

        lwComplex y2 = lwtensorUnaryOp(x, LWTENSOR_OP_CONJ, &ctx);
        EXPECT_TRUE(lwIsEqual(y2, make_lwComplex(1.2f, -3.4f)));
    }

    /**
     * \id activationWithQuantization_types
     * \brief validate the functionality of wactivationWithQuantization
     * \depends None
     * \setup None
     * \testprocedure ./apiTest --gtest_filter=OPERATORS_H.activationWithQuantization_types
     * \teardown None
     * \testgroup OPERATORS_H
     * \inputs None
     * \outputs None
     * \expected States: the expected value
     */
    TEST(OPERATORS_H, activationWithQuantization_types)
    {
        ElementwiseParameters::ActivationContext ctx;
        lwComplex x = make_lwComplex(1.2f, 3.4f);
        lwComplex scalar = make_lwComplex(1.2f, 3.4f);
        lwComplex y = activationWithQuantization(x, scalar, &ctx, LWTENSOR_OP_IDENTITY);
        lwComplex yr = make_lwComplex(lwtensorUnaryOp(x.x * scalar.x, LWTENSOR_OP_CONJ, &ctx) * scalar.y, 0.0);
        EXPECT_TRUE(lwIsEqual(y, yr));
    }

    /**
     * \id lwtensorUnaryOpHalf
     * \brief validate the functionality of operator lwtensorUnaryOp with type half
     * \depends None
     * \setup None
     * \testprocedure ./apiTest --gtest_filter=OPERATORS_H.lwtensorUnaryOpHalf
     * \teardown None
     * \designid LWTENSOR_DES_015, LWTENSOR_DES_018
     * \testgroup OPERATORS_H
     * \inputs None
     * \outputs None
     * \expected States: the expected value
     */
    TEST(OPERATORS_H, lwtensorUnaryOpHalf)
    {
        ElementwiseParameters::ActivationContext ctx;

        half ti = UniformRandomNumber<half>(0.5, 10.0);
        callingInfo("lwtensorUnaryOp<half>");
        half t1 = lwtensorUnaryOp<half>(ti, LWTENSOR_OP_IDENTITY, &ctx);
        EXPECT_TRUE(lwIsEqual(t1, ti));

        callingInfo("lwtensorUnaryOp<half>");
        half t2 = lwtensorUnaryOp<half>(ti, LWTENSOR_OP_SQRT, &ctx);
        // EXPECT_TRUE(lwIsEqual(t2, lwMul(ti, ti)));
        EXPECT_TRUE(lwIsEqual(t2, lwGet<half>(std::sqrt(lwGet<float>(ti)))));

        callingInfo("lwtensorUnaryOp<half>");
        half t3 = lwtensorUnaryOp<half>(ti, LWTENSOR_OP_RELU, &ctx);
        EXPECT_TRUE(lwIsEqual(t3, lwGet<float>(ti) > 0.f ? ti : lwGet<half>(0.f)));

        callingInfo("lwtensorUnaryOp<half>");
        half t4 = lwtensorUnaryOp<half>(ti, LWTENSOR_OP_CONJ, &ctx);
        EXPECT_TRUE(lwIsEqual(t4, ti));

        callingInfo("lwtensorUnaryOp<half>");
        half t5 = lwtensorUnaryOp<half>(ti, LWTENSOR_OP_RCP, &ctx);
        EXPECT_TRUE(lwIsEqual(t5, lwGet<half>(1.0f / lwGet<float>(ti))));

        callingInfo("lwtensorUnaryOp<half>");
        half tn = UniformRandomNumber<half>(-10.0, -1.0);
        half t6 = lwtensorUnaryOp<half>(tn, LWTENSOR_OP_RELU, &ctx);
        EXPECT_TRUE(lwIsEqual(t6, lwGet<half>(0.F)));

        float tolerance = 0.003;
        callingInfo("lwtensorUnaryOp<half>");
        half t7 = lwtensorUnaryOp<half>(ti, LWTENSOR_OP_SIGMOID, &ctx);
        //EXPECT_FLOAT_EQ(t7, lwGet<half>(0.5f * (tanhf(0.5f * lwGet<float>(ti)) + 1.f)));
        /* fix: LWTS-144 */
        EXPECT_NEAR(t7, lwGet<float>(lwGet<half>(0.5f * (tanhf(0.5f * lwGet<float>(ti)) + 1.f))), tolerance *lwGet<float>(t7));

        callingInfo("lwtensorUnaryOp<half>");
        half t8 = lwtensorUnaryOp<half>(ti, LWTENSOR_OP_TANH, &ctx);
        EXPECT_FLOAT_EQ(t8, lwGet<half>(tanhf(lwGet<float>(ti))));

        // callingInfo("lwtensorUnaryOp<half>");
        // ctx.elu.alpha = 0.1f;
        // half t9 = lwtensorUnaryOp<half>(ti, LWTENSOR_OP_ELU, &ctx);
        // EXPECT_TRUE(lwIsEqual(
        //     t9,
        //     lwGet<half>(lwGet<float>(ti) >= 0.f ? lwGet<float>(ti) : ctx.elu.alpha * (expf(lwGet<float>(ti)) - 1))));

        // callingInfo("lwtensorUnaryOp<half>");
        // ctx.leakyRelu.k = 0.1f;
        // half t10 = lwtensorUnaryOp<half>(ti, LWTENSOR_OP_LEAKY_RELU, &ctx);
        // EXPECT_TRUE(lwIsEqual(
        //     t10, lwGet<half>(lwGet<float>(ti) >= 0.f ? lwGet<float>(ti) : ctx.leakyRelu.k * lwGet<float>(ti))));

        // callingInfo("lwtensorUnaryOp<half>");
        // ctx.softPlus.inScale = 0.8f;
        // ctx.softPlus.outScale = 2.f;
        // ctx.softPlus.approximateThreshold = 20.f;
        // half t12 = lwtensorUnaryOp<half>(ti, LWTENSOR_OP_SOFT_PLUS, &ctx);
        // float expected = ctx.softPlus.outScale * (logf(1.f + expf(ctx.softPlus.inScale * lwGet<float>(ti))));
        // EXPECT_TRUE(std::abs(lwGet<float>(t12) - expected) < expected * 0.005f);

        // callingInfo("lwtensorUnaryOp<half>");
        // half t13 = lwtensorUnaryOp<half>(ti, LWTENSOR_OP_SOFT_SIGN, &ctx);
        // EXPECT_FLOAT_EQ(t13, lwGet<half>(lwGet<float>(ti) / lwGet<float>(lwAdd(lwGet<half>(1.f), lwAbs(ti)))));

        // callingInfo("lwtensorUnaryOp<half>");
        // ctx.selu.eluAlpha = 0.1f;
        // ctx.selu.outScale = 0.5f;
        // half t14 = lwtensorUnaryOp<half>(ti, LWTENSOR_OP_SELU, &ctx);
        // EXPECT_FLOAT_EQ( t14,
        //               lwGet<half>(ctx.selu.outScale
        //                           * (lwGet<float>(ti) >= 0.f ? lwGet<float>(ti)
        //                                                      : ctx.selu.eluAlpha * (expf(lwGet<float>(ti)) - 1.f))));

        // callingInfo("lwtensorUnaryOp<half>");
        // ctx.hardSigmoid.slope = 0.5f;
        // ctx.hardSigmoid.shift = 0.5f;
        // half t15 = lwtensorUnaryOp<half>(ti, LWTENSOR_OP_HARD_SIGMOID, &ctx);
        // EXPECT_TRUE(
        //     lwIsEqual(t15, lwGet<half>(std::min(1.f, std::max(0.f, ctx.hardSigmoid.slope * lwGet<float>(ti) + 0.5f)))));

        // callingInfo("lwtensorUnaryOp<half>");
        // ctx.scaledTanh.inScale = 0.5f;
        // ctx.scaledTanh.outScale = 1.5f;
        // half t16 = lwtensorUnaryOp<half>(ti, LWTENSOR_OP_SCALED_TANH, &ctx);
        // EXPECT_NEAR( t16, lwGet<half>(ctx.scaledTanh.outScale * (tanhf(ctx.scaledTanh.inScale * lwGet<float>(ti)))), tolerance * lwGet<float>(t16));

        // callingInfo("lwtensorUnaryOp<half>");
        // ctx.thresholdedRelu.threshold = 1.5f;
        // half t17 = lwtensorUnaryOp<half>(ti, LWTENSOR_OP_THRESHOLDED_RELU, &ctx);
        // EXPECT_TRUE(
        //     lwIsEqual(t17, lwGet<half>(lwGet<float>(ti) > ctx.thresholdedRelu.threshold ? lwGet<float>(ti) : 0.f)));

        // callingInfo("lwtensorUnaryOp<half>");
        // ctx.clip.lower = 0.5f;
        // ctx.clip.upper = 1.5f;
        // half t18 = lwtensorUnaryOp<half>(ti, LWTENSOR_OP_CLIP, &ctx);
        // EXPECT_TRUE(lwIsEqual(t18, lwGet<half>(std::max(std::min(lwGet<float>(ti), ctx.clip.upper), ctx.clip.lower))));

        // New operators for TRT unary runner.
        callingInfo("lwtensorUnaryOp<half>");
        half t20 = lwtensorUnaryOp<half>(ti, LWTENSOR_OP_EXP, &ctx);
        EXPECT_TRUE(lwIsEqual(t20, lwGet<half>(std::exp(lwGet<float>(ti)))));

        callingInfo("lwtensorUnaryOp<half>");
        half t21 = lwtensorUnaryOp<half>(ti, LWTENSOR_OP_LOG, &ctx);
        EXPECT_TRUE(lwIsEqual(t21, lwGet<half>(std::log(lwGet<float>(ti)))));

        callingInfo("lwtensorUnaryOp<half>");
        half t22 = lwtensorUnaryOp<half>(ti, LWTENSOR_OP_ABS, &ctx);
        EXPECT_TRUE(lwIsEqual(t22, lwGet<half>(std::abs(lwGet<float>(ti)))));

        callingInfo("lwtensorUnaryOp<half>");
        half t23 = lwtensorUnaryOp<half>(ti, LWTENSOR_OP_NEG, &ctx);
        EXPECT_TRUE(lwIsEqual(t23, lwSub(lwGet<half>(0.f), ti)));

        callingInfo("lwtensorUnaryOp<half>");
        half t24 = lwtensorUnaryOp<half>(ti, LWTENSOR_OP_SIN, &ctx);
        EXPECT_TRUE(lwIsEqual(t24, lwGet<half>(std::sin(lwGet<float>(ti)))));

        callingInfo("lwtensorUnaryOp<half>");
        half t25 = lwtensorUnaryOp<half>(ti, LWTENSOR_OP_COS, &ctx);
        EXPECT_TRUE(lwIsEqual(t25, lwGet<half>(std::cos(lwGet<float>(ti)))));

        callingInfo("lwtensorUnaryOp<half>");
        half t26 = lwtensorUnaryOp<half>(ti, LWTENSOR_OP_TAN, &ctx);
        EXPECT_TRUE(lwIsEqual(t26, lwGet<half>(std::tan(lwGet<float>(ti)))));

        callingInfo("lwtensorUnaryOp<half>");
        half t27 = lwtensorUnaryOp<half>(ti, LWTENSOR_OP_SINH, &ctx);
        EXPECT_TRUE(lwIsEqual(t27, lwGet<half>(std::sinh(lwGet<float>(ti)))));

        callingInfo("lwtensorUnaryOp<half>");
        half t28 = lwtensorUnaryOp<half>(ti, LWTENSOR_OP_COSH, &ctx);
        EXPECT_TRUE(lwIsEqual(t28, lwGet<half>(std::cosh(lwGet<float>(ti)))));

        callingInfo("lwtensorUnaryOp<half>");
        half t29 = lwtensorUnaryOp<half>(ti, LWTENSOR_OP_TANH, &ctx);
        EXPECT_TRUE(lwIsEqual(t29, lwGet<half>(std::tanh(lwGet<float>(ti)))));

        auto clippedTi = lwMin(lwMax(ti, lwGet<half>(-1.f)), lwGet<half>(1.f));

        callingInfo("lwtensorUnaryOp<half>");
        half t30 = lwtensorUnaryOp<half>(clippedTi, LWTENSOR_OP_ASIN, &ctx);
        EXPECT_TRUE(lwIsEqual(t30, lwGet<half>(std::asin(lwGet<float>(clippedTi)))));

        callingInfo("lwtensorUnaryOp<half>");
        half t31 = lwtensorUnaryOp<half>(clippedTi, LWTENSOR_OP_ACOS, &ctx);
        EXPECT_TRUE(lwIsEqual(t31, lwGet<half>(std::acos(lwGet<float>(clippedTi)))));

        callingInfo("lwtensorUnaryOp<half>");
        half t32 = lwtensorUnaryOp<half>(clippedTi, LWTENSOR_OP_ATAN, &ctx);
        EXPECT_TRUE(lwIsEqual(t32, lwGet<half>(std::atan(lwGet<float>(clippedTi)))));

        callingInfo("lwtensorUnaryOp<half>");
        half t33 = lwtensorUnaryOp<half>(clippedTi, LWTENSOR_OP_ASINH, &ctx);
        EXPECT_TRUE(lwIsEqual(t33, lwGet<half>(std::asinh(lwGet<float>(clippedTi)))));

        callingInfo("lwtensorUnaryOp<half>");
        auto clippedAcoshTi = lwMax(ti, lwGet<half>(2.3f));
        half t34 = lwtensorUnaryOp<half>(clippedAcoshTi, LWTENSOR_OP_ACOSH, &ctx);
        EXPECT_TRUE(lwIsEqual(t34, lwGet<half>(std::acosh(lwGet<float>(clippedAcoshTi)))));

        callingInfo("lwtensorUnaryOp<half>");
        half t35 = lwtensorUnaryOp<half>(clippedTi, LWTENSOR_OP_ATANH, &ctx);
        EXPECT_TRUE(lwIsEqual(t35, lwGet<half>(std::atanh(lwGet<float>(clippedTi)))));

        callingInfo("lwtensorUnaryOp<half>");
        half t36 = lwtensorUnaryOp<half>(ti, LWTENSOR_OP_CEIL, &ctx);
        EXPECT_TRUE(lwIsEqual(t36, lwGet<half>(std::ceil(lwGet<float>(ti)))));

        callingInfo("lwtensorUnaryOp<half>");
        half t37 = lwtensorUnaryOp<half>(ti, LWTENSOR_OP_FLOOR, &ctx);
        EXPECT_TRUE(lwIsEqual(t37, lwGet<half>(std::floor(lwGet<float>(ti)))));

        callingInfo("lwtensorUnaryOp<half>");
        half t38 = lwtensorUnaryOp<half>(ti, LWTENSOR_OP_ADD, &ctx); // LWTENSOR_OP_ADD is unspported
        EXPECT_TRUE(lwIsEqual(t38, ti));
    }

    /**
     * \id lwtensorUnaryOpInt
     * \brief validate the functionality of operator lwtensorUnaryOp with type int
     * \depends None
     * \setup None
     * \testprocedure ./apiTest --gtest_filter=lwtensorUnaryOpIntTest.lwtensorUnaryOpInt
     * \teardown None
     * \designid LWTENSOR_DES_015, LWTENSOR_DES_018
     * \testgroup OPERATORS_H
     * \inputs None
     * \outputs None
     * \expected States: the expected value
     */
    TEST_P(lwtensorUnaryOpIntTest, lwtensorUnaryOpInt)
    {
        ElementwiseParameters::ActivationContext ctx;
        switch (opt)
        {
            case LWTENSOR_OP_IDENTITY:
                callingInfo("lwtensorUnaryOp<int>");
                EXPECT_EQ(lwtensorUnaryOp<int>(x, opt, &ctx), x);
                break;
            case LWTENSOR_OP_SQRT:
                callingInfo("lwtensorUnaryOp<int>");
                EXPECT_EQ(lwtensorUnaryOp<int>(x, opt, &ctx), lwSqrt(x));
                break;
            case LWTENSOR_OP_RCP:
                callingInfo("lwtensorUnaryOp<int>");
                EXPECT_EQ(lwtensorUnaryOp<int>(x, opt, &ctx), 1 / x);
                break;
            case LWTENSOR_OP_CONJ:
                callingInfo("lwtensorUnaryOp<int>");
                EXPECT_EQ(lwtensorUnaryOp<int>(x, opt, &ctx), x);
                break;
            case LWTENSOR_OP_RELU:
                callingInfo("lwtensorUnaryOp<int>");
                EXPECT_EQ(lwtensorUnaryOp<int>(x, opt, &ctx), x > 0 ? x : 0);
                break;
            // case LWTENSOR_OP_CLIP:
            //     ctx.clip.upper = 2.0f;
            //     ctx.clip.lower = 1.0f;
            //     callingInfo("lwtensorUnaryOp<int>");
            //     EXPECT_EQ(lwtensorUnaryOp<int>(x, opt, &ctx), lwClip(x, ctx.clip.lower, ctx.clip.upper));
            //     break;
            // case LWTENSOR_OP_THRESHOLDED_RELU:
            //     ctx.thresholdedRelu.threshold = 1.f;
            //     callingInfo("lwtensorUnaryOp<int>");
            //     EXPECT_EQ(lwtensorUnaryOp<int>(x, opt, &ctx), lwThresholdedRelu(x, ctx.thresholdedRelu.threshold));
            //     break;
            // case LWTENSOR_OP_LEAKY_RELU:
            //     ctx.leakyRelu.k = 1.0f;
            //     callingInfo("lwtensorUnaryOp<int>");
            //     EXPECT_EQ(lwtensorUnaryOp<int>(x, opt, &ctx), lwLeakyRelu(x, ctx.leakyRelu.k));
            //     break;

            default:
                callingInfo("lwtensorUnaryOp<int>");
                EXPECT_EQ(lwtensorUnaryOp<int>(x, opt, &ctx), x);
                printf("\n############### opt=%d should not appear in lwtensorUnaryOpIntTest. ###############\n", opt);
        }
    }

    INSTANTIATE_TEST_CASE_P(LWTENSOR_TEST, lwtensorUnaryOpIntTest,
                            Combine(Values(1, -1),
                                    Values(LWTENSOR_OP_IDENTITY, LWTENSOR_OP_RELU)));

    /**
     * \id lwtensorBinaryOpInt
     * \brief validate the functionality of operator lwtensorBinaryOp with type int
     * \depends None
     * \setup None
     * \testprocedure ./apiTest --gtest_filter=lwtensorBinaryOpIntTest.lwtensorBinaryOpInt
     * \teardown None
     * \designid LWTENSOR_DES_016
     * \testgroup OPERATORS_H
     * \inputs None
     * \outputs None
     * \expected States: the expected value
     */
    TEST_P(lwtensorBinaryOpIntTest, lwtensorBinaryOpInt)
    {
        ElementwiseParameters::ActivationContext ctx;
        switch(opt)
        {
            case LWTENSOR_OP_ADD:
                callingInfo("lwtensorBinaryOp<int> and lwAdd");
                EXPECT_EQ(lwtensorBinaryOp<int>(x, y, opt, &ctx), lwAdd(x,y));
                break;
            case LWTENSOR_OP_MUL:
                callingInfo("lwtensorBinaryOp<int> and lwMul");
                EXPECT_EQ(lwtensorBinaryOp<int>(x, y, opt, &ctx), lwMul(x,y));
                break;
            case LWTENSOR_OP_MAX:
                callingInfo("lwtensorBinaryOp<int> and lwMax");
                EXPECT_EQ(lwtensorBinaryOp<int>(x, y, opt, &ctx), lwMax(x,y));
                break;
            default: //case LWTENSOR_OP_MIN:
                callingInfo("lwtensorBinaryOp<int> and lwMin");
                EXPECT_EQ(lwtensorBinaryOp<int>(x, y, opt, &ctx), lwMin(x,y));
                break;
                //            default:
                //                EXPECT_EQ(lwtensorBinaryOp<int>(x, y, opt, NULL), 0);
                //                break;
        }
    }

    INSTANTIATE_TEST_CASE_P(LWTENSOR_TEST, lwtensorBinaryOpIntTest,
            Combine(
                Values(1, -1),
                Values(1, -1),
                Values(LWTENSOR_OP_IDENTITY, LWTENSOR_OP_SQRT, LWTENSOR_OP_RELU, LWTENSOR_OP_CONJ,
                    LWTENSOR_OP_RCP, LWTENSOR_OP_ADD, LWTENSOR_OP_MUL, LWTENSOR_OP_MAX, LWTENSOR_OP_MIN,
                    LWTENSOR_OP_UNKNOWN)
                ));

    /**
     * \id lwtensorBinaryOpfloat
     * \brief validate the functionality of operator lwtensorBinaryOp with type float
     * \depends None
     * \setup None
     * \testprocedure ./apiTest --gtest_filter=OPERATORS_H.lwtensorBinaryOpfloat
     * \teardown None
     * \designid LWTENSOR_DES_016
     * \testgroup OPERATORS_H
     * \inputs None
     * \outputs None
     * \expected States: the expected value
     */
    TEST(OPERATORS_H, lwtensorBinaryOpfloat)
    {
        ElementwiseParameters::ActivationContext ctx;

        float ti1 = UniformRandomNumber<float>(0.0, 10.0);
        float ti2 = UniformRandomNumber<float>(0.0, 10.0);
        callingInfo("lwtensorBinaryOp<float>");
        float to1 = lwtensorBinaryOp<float>(ti1, ti2, LWTENSOR_OP_ADD, &ctx);
        EXPECT_TRUE(lwIsEqual(to1, ti1 + ti2));

        callingInfo("lwtensorBinaryOp<float>");
        float to2 = lwtensorBinaryOp<float>(ti1, ti2, LWTENSOR_OP_MUL, &ctx);
        EXPECT_TRUE(lwIsEqual(to2, ti1 * ti2));

        callingInfo("lwtensorBinaryOp<float>");
        float to3 = lwtensorBinaryOp<float>(ti1, ti2, LWTENSOR_OP_MAX, &ctx);
        EXPECT_TRUE(lwIsEqual(to3, (ti1>ti2)?ti1:ti2));

        callingInfo("lwtensorBinaryOp<float>");
        float to4 = lwtensorBinaryOp<float>(ti1, ti2, LWTENSOR_OP_MIN, &ctx);
        EXPECT_TRUE(lwIsEqual(to4, (ti1<ti2)?ti1:ti2));

        //        float to5 = lwtensorBinaryOp<float>(ti1, ti2, LWTENSOR_OP_IDENTITY, NULL, &ctx);
        //        EXPECT_TRUE(lwIsEqual(to5, 0.f));
    }

    /**
     * \id lwtensorBinaryOphalf
     * \brief validate the functionality of operator lwtensorBinaryOp with type half
     * \depends None
     * \setup None
     * \testprocedure ./apiTest --gtest_filter=OPERATORS_H.lwtensorBinaryOphalf
     * \teardown None
     * \designid LWTENSOR_DES_016
     * \testgroup OPERATORS_H
     * \inputs None
     * \outputs None
     * \expected States: the expected value
     */
    TEST(OPERATORS_H, lwtensorBinaryOphalf )
    {
        ElementwiseParameters::ActivationContext ctx;

        half  ti1 = UniformRandomNumber<half >(0.0, 10.0);
        half  ti2 = UniformRandomNumber<half >(0.0, 10.0);
        callingInfo("lwtensorBinaryOp<half>");
        half  to1 = lwtensorBinaryOp<half >(ti1, ti2, LWTENSOR_OP_ADD, &ctx);
        EXPECT_TRUE(lwIsEqual(to1, lwAdd(ti1, ti2)));

        callingInfo("lwtensorBinaryOp<half>");
        half  to2 = lwtensorBinaryOp<half >(ti1, ti2, LWTENSOR_OP_MUL, &ctx);
        EXPECT_TRUE(lwIsEqual(to2, lwMul(ti1, ti2)));

        //        half  to3 = lwtensorBinaryOp<half >(ti1, ti2, LWTENSOR_OP_IDENTITY, NULL, &ctx);
        //        EXPECT_TRUE(lwIsEqual(to3, lwGet<half>(0.0f)));
    }

    /**
     * \id lwtensorBinaryOplwComplex
     * \brief validate the functionality of operator lwtensorBinaryOp with type lwComplex
     * \depends None
     * \setup None
     * \testprocedure ./apiTest --gtest_filter=OPERATORS_H.lwtensorBinaryOplwComplex
     * \teardown None
     * \designid LWTENSOR_DES_016
     * \testgroup OPERATORS_H
     * \inputs None
     * \outputs None
     * \expected States: the expected value
     */
    TEST(OPERATORS_H, lwtensorBinaryOplwComplex)
    {
        ElementwiseParameters::ActivationContext ctx;

        float ti1 = UniformRandomNumber<float>(0.0, 10.0);
        float ti2 = UniformRandomNumber<float>(0.0, 10.0);
        lwComplex ci1 = make_lwComplex(ti1, ti2);
        lwComplex ci2 = make_lwComplex(ti2, ti1);

        callingInfo("lwtensorBinaryOp<lwComplex>");
        lwComplex to1 = lwtensorBinaryOp<lwComplex>(ci1, ci2, LWTENSOR_OP_ADD, &ctx, LWTENSOR_OP_IDENTITY);
        EXPECT_TRUE(lwIsEqual(to1, lwAdd(ci1, ci2)));

        callingInfo("lwtensorBinaryOp<lwComplex>");
        lwComplex to2 = lwtensorBinaryOp<lwComplex>(ci1, ci2, LWTENSOR_OP_MUL, &ctx, LWTENSOR_OP_IDENTITY);
        EXPECT_TRUE(lwIsEqual(to2, lwMul(ci1, ci2)));

        // callingInfo("lwtensorBinaryOp<lwComplex>");
        // lwComplex to4 = lwtensorBinaryOp<lwComplex>(ci1, ci2, LWTENSOR_OP_ACTIVATION_WITH_QUANTIZATION, &ctx, LWTENSOR_OP_IDENTITY);
        // EXPECT_TRUE(lwIsEqual(to4, activationWithQuantization(ci1, ci2, &ctx, LWTENSOR_OP_IDENTITY)));
    }

    /**
     * \id lwSqrt
     * \brief validate the functionality of operator lwSqrt with type double
     * \depends None
     * \setup None
     * \testprocedure ./apiTest --gtest_filter=OPERATORS_H.lwSqrt
     * \teardown None
     * \testgroup OPERATORS_H
     * \inputs None
     * \outputs None
     * \expected States: the expected value
     */
    TEST(OPERATORS_H, lwSqrt)
    {
        double a = 100.0;
        double b;
        callingInfo("lwSqrt");
        b = lwSqrt(a);
        EXPECT_EQ(b, sqrt(a));
    }

    /**
     * \id lw2Norm_double
     * \brief validate the functionality of operator lw2Norm with type double
     * \depends None
     * \setup None
     * \testprocedure ./apiTest --gtest_filter=OPERATORS_H.lw2Norm_double
     * \teardown None
     * \testgroup OPERATORS_H
     * \inputs None
     * \outputs None
     * \expected States: the expected value
     */
    TEST(OPERATORS_H, lw2Norm_double)
    {
        double a = 1.1;
        callingInfo("lw2Norm<double>");
        double b = lw2Norm<double>(a);
        EXPECT_EQ(b, a);

        a = -1.1;
        callingInfo("lw2Norm<double>");
        b = lw2Norm<double>(a);
        EXPECT_EQ(b, -a);
    }

    /**
     * \id lw2Norm_types
     * \brief validate the functionality of operator lw2Norm
     * \depends None
     * \setup None
     * \testprocedure ./apiTest --gtest_filter=OPERATORS_H.lw2Norm_types
     * \teardown None
     * \testgroup OPERATORS_H
     * \inputs None
     * \outputs None
     * \expected States: the expected value
     */
    TEST(OPERATORS_H, lw2Norm_types)
    {
        half a = __float2half(1.1f);
        callingInfo("lw2Norm<half>");
        double b = lw2Norm(lwGet<float>(a));
        double c = lw2Norm(a);
        EXPECT_EQ(b, c);

        uint32_t a1 = 3U;
        callingInfo("lw2Norm<uint32_t>");
        double b1 = lw2Norm(a1);
        double c1 = static_cast<double>(a1);
        EXPECT_EQ(b1, c1);

        uint32_t a2 = 3U;
        callingInfo("lw2Norm<uint8_t>");
        double b2 = lw2Norm(a2);
        double c2 = static_cast<double>(a2);
        EXPECT_EQ(b2, c2);
    }

    /**
     * \id Branch_lwMax
     * \brief validate the functionality of operator lwMax
     * \depends None
     * \setup None
     * \testprocedure ./apiTest --gtest_filter=OPERATORS_H.Branch_lwMax
     * \teardown None
     * \testgroup OPERATORS_H
     * \inputs None
     * \outputs None
     * \expected States: the expected value
     */
    TEST(OPERATORS_H, Branch_lwMax)
    {
        callingInfo("lwMax<int>");
        EXPECT_EQ(lwMax<int>(1, 2), 2);
        EXPECT_EQ(lwMax<int>(2, 1), 2);

        callingInfo("lwMax<float>");
        EXPECT_EQ(lwMax<float>(1.0f, 2.0f), 2.0f);
        EXPECT_EQ(lwMax<float>(2.0f, 1.0f), 2.0f);

        callingInfo("lwMax<double>");
        EXPECT_EQ(lwMax<double>(1.0, 2.0), 2.0);
        EXPECT_EQ(lwMax<double>(2.0, 1.0), 2.0);

        callingInfo("lwMax<half>");
        half a = lwGet<half>(3.0f);
        half b = lwGet<half>(2.0f);
        half ma = lwMax(lwGet<float>(a), lwGet<float>(b));
        EXPECT_TRUE(lwIsEqual(lwMax(a, b), ma));
    }

    /**
     * \id Branch_lwMin
     * \brief validate the functionality of operator lwMin
     * \depends None
     * \setup None
     * \testprocedure ./apiTest --gtest_filter=OPERATORS_H.Branch_lwMin
     * \teardown None
     * \testgroup OPERATORS_H
     * \inputs None
     * \outputs None
     * \expected States: the expected value
     */
    TEST(OPERATORS_H, Branch_lwMin)
    {
        callingInfo("lwMin<int>");
        EXPECT_EQ(lwMin<int>(1, 2), 1);
        EXPECT_EQ(lwMin<int>(2, 1), 1);

        callingInfo("lwMin<float>");
        EXPECT_EQ(lwMin<float>(1.0f, 2.0f), 1.0f);
        EXPECT_EQ(lwMin<float>(2.0f, 1.0f), 1.0f);

        callingInfo("lwMin<double>");
        EXPECT_EQ(lwMin<double>(1.0, 2.0), 1.0);
        EXPECT_EQ(lwMin<double>(2.0, 1.0), 1.0);

        callingInfo("lwMax<half>");
        half a = lwGet<half>(3.0f);
        half b = lwGet<half>(2.0f);
        half mi = lwMin(lwGet<float>(a), lwGet<float>(b));
        EXPECT_TRUE(lwIsEqual(lwMin(a, b), mi));
    }

    /**
     * \id Branch_lwSub
     * \brief Add test case to cover more branches in function lwSub
     * \depends None
     * \setup None
     * \testprocedure ./apiTest --gtest_filter=OPERATORS_H.Branch_lwSub
     * \teardown None
     * \testgroup OPERATORS_H
     * \inputs None
     * \outputs None
     * \expected States: the expected value
     */
    TEST(OPERATORS_H, Branch_lwSub)
    {
        half a = lwGet<half>(2);
        half b = lwGet<half>(1);
        callingInfo("lwSub<half>");
        EXPECT_TRUE(lwIsEqual(lwSub<half>(a, b), lwGet<half>(1)));

        int c = 2;
        int d = 1;
        callingInfo("lwSub<int>");
        EXPECT_EQ(lwSub(c, d), c-d);
    }

    /**
     * \id Branch_lwSoftPlus
     * \brief Add test case to cover more branches in function lwSoftPlus
     * \depends None
     * \setup None
     * \testprocedure ./apiTest --gtest_filter=OPERATORS_H.Branch_lwSoftPlus
     * \teardown None
     * \testgroup OPERATORS_H
     * \inputs None
     * \outputs None
     * \expected States: the expected value
     */
    TEST(OPERATORS_H, Branch_lwSoftPlus)
    {
        callingInfo("lwSoftPlus<float>");
        EXPECT_EQ(lwSoftPlus<float>(100.0f, 1.0f, 1.0f, 2.0f), 100);
    }

    /**
     * \id floorCeil
     * \brief validate the functionality of operator lwFloor and lwCeil
     * \depends None
     * \setup None
     * \testprocedure ./apiTest --gtest_filter=OPERATORS_H.floorCeil
     * \teardown None
     * \testgroup OPERATORS_H
     * \inputs None
     * \outputs None
     * \expected States: the expected value
     */
    TEST(OPERATORS_H, floorCeil)
    {
        float x0 = 1.2f;
        callingInfo("lwFloor<float>");
        float y0 = lwFloor(x0);
        float z0 = std::floor(x0);
        EXPECT_FLOAT_EQ(y0, z0);

        float x1 = 1.2f;
        callingInfo("lwCeil<float>");
        float y1 = lwCeil(x1);
        float z1 = std::ceil(x1);
        EXPECT_FLOAT_EQ(y1, z1);

        double x2 = 1.2;
        callingInfo("lwFloor<double>");
        double y2 = lwFloor(x2);
        double z2 = std::floor(x2);
        EXPECT_FLOAT_EQ(y2, z2);

        double x3 = 1.2;
        callingInfo("lwCeil<double>");
        double y3 = lwCeil(x3);
        double z3 = std::ceil(x3);
        EXPECT_FLOAT_EQ(y3, z3);
    }

    /**
     * \id Branch_operators
     * \brief Add test case to cover more branches in several operators
     * \depends None
     * \setup None
     * \testprocedure ./apiTest --gtest_filter=OPERATORS_H.Branch_operators
     * \teardown None
     * \testgroup OPERATORS_H
     * \inputs None
     * \outputs None
     * \expected States: the expected value
     */
    TEST(OPERATORS_H, Branch_operators)
    {
        float x0 = -1.2F;
        float alpha0 = 1.0F;
        callingInfo("lwElu<float>");
        float y0 = lwElu(x0, alpha0);
        float z0 = (std::exp(x0) - 1.F) * alpha0;
        EXPECT_FLOAT_EQ(y0, z0);

        float x1 = -1.2F;
        callingInfo("lwAbs<float>");
        float y1 = lwAbs(x1);
        float z1 = 0.0F - x1;
        EXPECT_FLOAT_EQ(y1, z1);

        int x2 = -2;
        callingInfo("lwAbs<int>");
        int y2 = lwAbs(x2);
        int z2 = -x2;
        EXPECT_FLOAT_EQ(y2, z2);

        half x3 = lwGet<half>(-1.2F);
        callingInfo("lwAbs<half>");
        half y3 = lwAbs(x3);
        half z3 = lwNeg(x3);
        EXPECT_FLOAT_EQ(y3, z3);
    }

    /**
     * \id lw2Norm_uint8_t
     * \brief Add test case to cover lw2Norm (input type uint8_t, output type double)
     * \depends None
     * \setup None
     * \testprocedure ./apiTest --gtest_filter=OPERATORS_H.lw2Norm_uint8_t
     * \teardown None
     * \testgroup OPERATORS_H
     * \inputs None
     * \outputs None
     * \expected States: the expected value
     */
    TEST(OPERATORS_H, lw2Norm_uint8_t)
    {
        uint8_t x0 = 2U;
        callingInfo("lw2Norm<double>");
        double y0 = lw2Norm(x0);
        double z0 = static_cast<double>(x0);
        EXPECT_DOUBLE_EQ(y0, z0);
    }


    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //              tests for src/lwtensor.lw, these tests were added to enhance the code coverage
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////


    /**
     * \id lwtensorInitNullHandle
     * \brief test the function lwtensorInit with negative null handle
     * \depends None
     * \setup None
     * \testprocedure ./apiTest --gtest_filter=LWTENSOR_CPP.lwtensorInitNullHandle
     * \teardown None
     * \testgroup LWTENSOR_CPP
     * \inputs null handle
     * \outputs None
     * \expected States: return LWTENSOR_STATUS_ILWALID_VALUE
     */
    TEST(LWTENSOR_CPP, lwtensorInitNullHandle)
    {
        callingInfo("lwtensorInit");
        EXPECT_EQ(lwtensorInit(NULL), LWTENSOR_STATUS_ILWALID_VALUE);
    }

    /**
     * \id lwtensorInitTensorLargeStride
     * \brief test the function lwtensorInitTensorDescriptor with extremely large stride
     * \depends None
     * \setup extent and stride
     * \testprocedure ./apiTest --gtest_filter=LWTENSOR_CPP.lwtensorInitTensorLargeStride
     * \teardown None
     * \designid LWTENSOR_DES_009
     * \testgroup LWTENSOR_CPP
     * \inputs the parameters of function lwtensorInitTensorDescriptor
     * \outputs desc
     * \expected States: return LWTENSOR_STATUS_ILWALID_VALUE
     */
    TEST(LWTENSOR_CPP, lwtensorInitTensorLargeStride) //JIRA/CUT-51
    {
        std::vector<int64_t> extent = {4, 8, 12};
        std::vector<int64_t> stride = {static_cast<int64_t>(std::numeric_limits<stride_type>::max()), 12, 1};
        stride[0] += 1;
        lwtensorHandle_t handle;
        EXPECT_EQ(lwtensorInit(&handle), LWTENSOR_STATUS_SUCCESS);
        lwtensorTensorDescriptor_t desc;
        EXPECT_EQ(lwtensorInit(&handle), LWTENSOR_STATUS_SUCCESS);
        callingInfo("lwtensorInitTensorDescriptor");
        EXPECT_EQ(lwtensorInitTensorDescriptor(&handle, &desc, 3, &extent[0], &stride[0], LWDA_R_32F, LWTENSOR_OP_IDENTITY, 1, 0), LWTENSOR_STATUS_NOT_SUPPORTED);
    }

    /**
     * \id lwtensorInitTensorDescriptorSimpleNull
     * \brief test the function lwtensorInitTensorDescriptor with null descriptor
     * \depends None
     * \setup None
     * \testprocedure ./apiTest --gtest_filter=LWTENSOR_CPP.lwtensorInitTensorDescriptorSimpleNull
     * \teardown None
     * \testgroup LWTENSOR_CPP
     * \inputs all the parameters
     * \outputs None
     * \expected States: return LWTENSOR_STATUS_ILWALID_VALUE
     */
    TEST(LWTENSOR_CPP, lwtensorInitTensorDescriptorSimpleNull)
    {
        std::vector<int64_t> extent = {4, 8, 12};
        std::vector<int64_t> stride = {96, 12, 1};
        lwtensorHandle_t handle;
        EXPECT_EQ(lwtensorInit(&handle), LWTENSOR_STATUS_SUCCESS);
        callingInfo("lwtensorInitTensorDescriptor");
        EXPECT_EQ(lwtensorInitTensorDescriptor(&handle, NULL, 3, &extent[0], &stride[0], LWDA_R_32F, LWTENSOR_OP_IDENTITY, 1, 0), LWTENSOR_STATUS_ILWALID_VALUE);
    }

    /**
     * \id lwtensorInitTensorDescriptorSimpleExtent
     * \brief test the function lwtensorInitTensorDescriptor with negative extent
     * \depends None
     * \setup extent and stride
     * \testprocedure ./apiTest --gtest_filter=LWTE.lwtensorInitTensorDescriptorSimpleExtent
     * \teardown None
     * \testgroup LWTENSOR_CPP
     * \inputs the parameters of function lwtensorInitTensorDescriptor
     * \outputs desc
     * \expected States: LWTENSOR_STATUS_NOT_SUPPORTED
     */
    TEST(LWTENSOR_CPP, lwtensorInitTensorDescriptorSimpleExtent)
    {
        std::vector<int64_t> extent = {-4, 8, 12};
        std::vector<int64_t> stride = {96, 12, 1};
        lwtensorTensorDescriptor_t desc;
        lwtensorHandle_t handle;
        EXPECT_EQ(lwtensorInit(&handle), LWTENSOR_STATUS_SUCCESS);
        callingInfo("lwtensorInitTensorDescriptor");
        EXPECT_EQ(lwtensorInitTensorDescriptor(&handle, &desc, 3, &extent[0], &stride[0], LWDA_R_32F, LWTENSOR_OP_IDENTITY, 1, 0), LWTENSOR_STATUS_NOT_SUPPORTED);

    }

    /**
     * \id lwtensorInitTensorDescriptorSimpleStride
     * \brief test the function lwtensorInitTensorDescriptor with negative stride.
     * \depends None
     * \setup extent and stride
     * \testprocedure ./apiTest --gtest_filter=LWTENSOR_CPP.lwtensorInitTensorDescriptorSimpleStride
     * \teardown None
     * \designid LWTENSOR_DES_010
     * \testgroup LWTENSOR_CPP
     * \inputs the parameters of function lwtensorInitTensorDescriptor
     * \outputs desc
     * \expected States: return value LWTENSOR_STATUS_NOT_SUPPORTED
     */
    TEST(LWTENSOR_CPP, lwtensorInitTensorDescriptorSimpleStride)
    {
        std::vector<int64_t> extent = {4, 8, 12};
        std::vector<int64_t> stride = {-96, 12, 1};
        lwtensorTensorDescriptor_t desc;
        lwtensorHandle_t handle;
        EXPECT_EQ(lwtensorInit(&handle), LWTENSOR_STATUS_SUCCESS);
        callingInfo("lwtensorInitTensorDescriptor");
        EXPECT_EQ(lwtensorInitTensorDescriptor(&handle, &desc, 3, &extent[0], &stride[0], LWDA_R_32F, LWTENSOR_OP_IDENTITY, 1, 0), LWTENSOR_STATUS_NOT_SUPPORTED);
    }

    /**
     * \id lwtensorInitTensorDescriptorIlwalidWidth
     * \brief test the function lwtensorInitTensorDescriptor with invalid stride.
     * \depends None
     * \setup extent and stride
     * \testprocedure ./apiTest --gtest_filter=LWTENSOR_CPP.lwtensorInitTensorDescriptorIlwalidWidth
     * \teardown None
     * \testgroup LWTENSOR_CPP
     * \inputs the parameters for function lwtensorInitTensorDescriptor
     * \outputs desc
     * \expected States: return LWTENSOR_STATUS_ILWALID_VALUE
     */
    TEST(LWTENSOR_CPP, lwtensorInitTensorDescriptorIlwalidWidth)
    {
        std::vector<int64_t> extent = {4, 8, 12};
        std::vector<int64_t> stride = {96, 12, 1};
        lwtensorTensorDescriptor_t desc;
        lwtensorHandle_t handle;
        EXPECT_EQ(lwtensorInit(&handle), LWTENSOR_STATUS_SUCCESS);
        callingInfo("lwtensorInitTensorDescriptor");
        EXPECT_EQ(lwtensorInitTensorDescriptor(&handle, &desc, 3, &extent[0], &stride[0], LWDA_R_32F, LWTENSOR_OP_IDENTITY, 133, 0), LWTENSOR_STATUS_ILWALID_VALUE);
    }

    /**
     * \id lwtensorInitTensorDescriptorIlwalidModeIndex
     * \brief test the function lwtensorInitTensorDescriptor with invalid vectorModeIndex
     * \depends None
     * \setup extent and stride
     * \testprocedure ./apiTest --gtest_filter=LWTENSOR_CPP.lwtensorInitTensorDescriptorIlwalidModeIndex
     * \teardown None
     * \testgroup LWTENSOR_CPP
     * \inputs the parameters for function lwtensorInitTensorDescriptor
     * \outputs desc
     * \expected States: return LWTENSOR_STATUS_ILWALID_VALUE
     */
    TEST(LWTENSOR_CPP, lwtensorInitTensorDescriptorIlwalidModeIndex)
    {
        std::vector<int64_t> extent = {4, 8, 12};
        std::vector<int64_t> stride = {96, 12, 1};
        lwtensorTensorDescriptor_t desc;
        lwtensorHandle_t handle;
        EXPECT_EQ(lwtensorInit(&handle), LWTENSOR_STATUS_SUCCESS);
        callingInfo("lwtensorInitTensorDescriptor");
        EXPECT_EQ(lwtensorInitTensorDescriptor(&handle, &desc, 3, &extent[0], &stride[0], LWDA_R_32F, LWTENSOR_OP_IDENTITY, 8, extent.size() + 1), LWTENSOR_STATUS_ILWALID_VALUE);
    }

    /**
     * \id lwtensorInitTensorDescriptorVectorized
     * \brief test the function lwtensorInitTensorDescriptor with invalid vectorwidth (not all strides are multiple of the vector width).
     * \depends None
     * \setup extent and stride
     * \testprocedure ./apiTest --gtest_filter=LWTENSOR_CPP.lwtensorInitTensorDescriptorVectorized
     * \teardown None
     * \testgroup LWTENSOR_CPP
     * \designid LWTENSOR_DES_010
     * \testgroup LWTENSOR_CPP
     * \inputs the parameters for function lwtensorInitTensorDescriptor
     * \outputs desc
     * \expected States: return LWTENSOR_STATUS_ILWALID_VALUE
     */
    TEST(LWTENSOR_CPP, lwtensorInitTensorDescriptorVectorized)
    {
        std::vector<int64_t> extent = {4, 8, 12};
        std::vector<int64_t> stride = {96, 12, 1};
        lwtensorTensorDescriptor_t desc;
        lwtensorHandle_t handle;
        EXPECT_EQ(lwtensorInit(&handle), LWTENSOR_STATUS_SUCCESS);
        callingInfo("lwtensorInitTensorDescriptor");
        EXPECT_EQ(lwtensorInitTensorDescriptor(&handle, &desc, 3, &extent[0], &stride[0], LWDA_R_32F, LWTENSOR_OP_IDENTITY, 2, 0), LWTENSOR_STATUS_ILWALID_VALUE);
    }

    /**
     * \id lwtensorInitTensorDescriptorSimple_NotSupported
     * \brief test the function lwtensorInitTensorDescriptor with invalid extent and stride.
     * \depends None
     * \setup extent and stride
     * \testprocedure ./apiTest --gtest_filter=LWTENSOR_CPP.lwtensorInitTensorDescriptorSimple_NotSupported
     * \teardown None
     * \designid LWTENSOR_DES_009
     * \testgroup LWTENSOR_CPP
     * \inputs the parameters of function lwtensorInitTensorDescriptor
     * \outputs desc
     * \expected States: LWTENSOR_STATUS_NOT_SUPPORTED
     */
    TEST(LWTENSOR_CPP, lwtensorInitTensorDescriptorSimple_NotSupported)
    {
        std::vector<int64_t> extent = {static_cast<int64_t>(std::numeric_limits<stride_type>::max() - 1), 1};
        std::vector<int64_t> stride = {static_cast<int64_t>(std::numeric_limits<stride_type>::max() - 1), 1};
        lwtensorTensorDescriptor_t desc;
        lwtensorHandle_t handle;
        EXPECT_EQ(lwtensorInit(&handle), LWTENSOR_STATUS_SUCCESS);
        callingInfo("lwtensorInitTensorDescriptor");
        EXPECT_EQ(lwtensorInitTensorDescriptor(&handle, &desc, extent.size(), &extent[0], &stride[0], LWDA_R_32F, LWTENSOR_OP_IDENTITY, 1, 0), LWTENSOR_STATUS_ILWALID_VALUE); // extent exceeds int32_t => negative
    }

    TEST(LWTENSOR_CPP, lwtensorInitTensorDescriptorSimple_Supported64)
    {
        int64_t extent = 1;
        std::vector<int64_t> stride = {static_cast<int64_t>(std::numeric_limits<stride_type>::max() - 1), 1};
        lwtensorTensorDescriptor_t desc;
        lwtensorHandle_t handle;
        EXPECT_EQ(lwtensorInit(&handle), LWTENSOR_STATUS_SUCCESS);
        callingInfo("lwtensorInitTensorDescriptor");
        EXPECT_EQ(lwtensorInitTensorDescriptor(&handle, &desc, 1, &extent, &stride[0], LWDA_R_32F, LWTENSOR_OP_IDENTITY, 1, 0), LWTENSOR_STATUS_SUCCESS);
    }

    /**
     * \id lwtensorInitTensorDescriptor
     * \brief test the function user-level lwtensorInitTensorDescriptor.
     * \depends None
     * \setup extent and stride
     * \testprocedure ./apiTest --gtest_filter=LWTENSOR_CPP.lwtensorInitTensorDescriptor
     * \teardown None
     * \testgroup LWTENSOR_CPP
     * \inputs the parameters of function lwtensorInitTensorDescriptor
     * \outputs desc
     * \expected States: LWTENSOR_STATUS_SUCCESS
     */
    TEST(LWTENSOR_CPP, lwtensorInitTensorDescriptor)
    {
        std::vector<int64_t> extent = {4, 8, 12};
        std::vector<int64_t> stride = {96, 12, 1};
        lwtensorTensorDescriptor_t desc;
        lwtensorHandle_t handle;
        EXPECT_EQ(lwtensorInit(&handle), LWTENSOR_STATUS_SUCCESS);
        callingInfo("lwtensorInitTensorDescriptor");
        EXPECT_EQ(lwtensorInitTensorDescriptor(&handle, &desc, 3, &extent[0], &stride[0], LWDA_R_32F, LWTENSOR_OP_IDENTITY), LWTENSOR_STATUS_SUCCESS);
    }

    /**
     * \id lwtensorInitTensorDescriptorException
     * \brief test the class lwtensorInitTensorDescriptor with exceptions.
     * \depends None
     * \setup construct desc in advance
     * \testprocedure ./apiTest --gtest_filter=LWTENSOR_CPP.lwtensorInitTensorDescriptorException
     * \teardown None
     * \designid LWTENSOR_DES_001
     * \testgroup LWTENSOR_CPP
     * \inputs the parameters of function lwtensorInitTensorDescriptor
     * \outputs None
     * \expected States: return LWTENSOR_STATUS_INTERNAL_ERROR
     */
    TEST(LWTENSOR_CPP, lwtensorInitTensorDescriptorException)
    {
        std::vector<int64_t> extent = {1, 2, 1, 2, 1, 2, 1, 1, 1, 1, 2, 2, 3}; //number is bigger than LWTENSOR_MAX_MODES_EXTERNAL
        lwtensorTensorDescriptor_t desc;
        lwtensorHandle_t handle;
        EXPECT_EQ(lwtensorInit(&handle), LWTENSOR_STATUS_SUCCESS);
        callingInfo("lwtensorInitTensorDescriptor");
        EXPECT_EQ(lwtensorInitTensorDescriptor(&handle, &desc, extent.size(), &extent[0], NULL, LWDA_R_32F, LWTENSOR_OP_IDENTITY, 1, 0), LWTENSOR_STATUS_SUCCESS);
    }

    TEST(LWTENSOR_CPP, test_LWTENSOR_STATUS_INTERNAL_ERROR)
    {
        std::vector<int64_t> extent = {1, 2, 3, 4, 5, 6, 7, 8, 9}; //number is bigger than LWTENSOR_MAX_MODES_EXTERNAL
        lwtensorTensorDescriptor_t desc;
        lwtensorHandle_t handle;
        EXPECT_EQ(lwtensorInit(&handle), LWTENSOR_STATUS_SUCCESS);
        callingInfo("lwtensorInitTensorDescriptor");
        EXPECT_EQ(lwtensorInitTensorDescriptor(&handle, &desc, extent.size(), extent.data(), NULL, LWDA_R_32F, LWTENSOR_OP_IDENTITY, 1, 0), LWTENSOR_STATUS_SUCCESS);
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //              tests for src/types.cpp, these tests were added to enhance the code coverage
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    /**
     * \id TensorDescriptor_setVectorization1
     * \brief test the method setVectorization in struct TensorDescriptor
     * \depends None
     * \setup extent
     * \testprocedure ./apiTest --gtest_filter=TYPESPLC3_CPP.TensorDescriptor_setVectorization1
     * \teardown None
     * \testgroup TYPESPLC3_CPP
     * \inputs the parameters of TensorDescriptor
     * \outputs None
     * \expected States: return LWTENSOR_STATUS_ILWALID_VALUE
     */
    TEST(TYPESPLC3_CPP, TensorDescriptor_setVectorization1)
    {
        std::vector<extent_type> extent = {4, 8, 12};
        TensorDescriptor desc(extent.size(), &extent[0], NULL, LWDA_R_32F, LWTENSOR_OP_IDENTITY, 1, 0);
        callingInfo("setVectorization");
        EXPECT_EQ(desc.setVectorization(10, 2), LWTENSOR_STATUS_ILWALID_VALUE);
    }

    /**
     * \id TensorDescriptor0
     * \brief test struct with invalid numMode
     * \depends None
     * \setup extent
     \testprocedure ./apiTest --gtest_filter=TYPESPLC3_CPP.TensorDescriptor0
     * \teardown None
     * \testgroup TYPESPLC3_CPP
     * \inputs the parameters of TensorDescriptor
     * \outputs None
     * \expected States: catch the exception: "Number of modes is invalid.\n"
     */
    TEST(TYPESPLC3_CPP, TensorDescriptor0)
    {
        std::vector<extent_type> extent = {4, 8, 12};
        try {
            callingInfo("TensorDescriptor");
            TensorDescriptor desc(100, &extent[0], NULL, LWDA_R_32F, LWTENSOR_OP_IDENTITY, 1, 0);
        }
        catch(const std::exception& e) {
            EXPECT_TRUE(strcmp(e.what(), "Number of modes is invalid.\n")==0);
        }
    }

    /**
     * \id TensorDescriptor2
     * \brief test the copy constructor of TensorDescriptor
     * \depends None
     * \setup extent and instantiate of TensorDescriptor
     * \testprocedure ./apiTest --gtest_filter=TYPESPLC3_CPP.TensorDescriptor2
     * \teardown None
     * \testgroup TYPESPLC3_CPP
     * \inputs the parameters of TensorDescriptor
     * \outputs None
     * \expected States: None
     */
    TEST(TYPESPLC3_CPP, TensorDescriptor2)
    {
        std::vector<extent_type> extent = {4, 8, 12};
        callingInfo("TensorDescriptor");
        TensorDescriptor descA(extent.size(), &extent[0], NULL, LWDA_R_32F, LWTENSOR_OP_IDENTITY, 1, 0);
        TensorDescriptor descB(descA);
    }

    /**
     * \id TensorDescriptor5
     * \brief test srtuct TensorDescriptor with non-default vectorWidth and vectorMode
     * \depends None
     * \setup vectorWidth and vectorMode
     * \testprocedure ./apiTest --gtest_filter=TYPESPLC3_CPP.TensorDescriptor5
     * \teardown None
     * \testgroup TYPESPLC3_CPP
     * \inputs the parameters of TensorDescriptor
     * \outputs None
     * \expected States: None
     */
    TEST(TYPESPLC3_CPP, TensorDescriptor5)
    {
        std::vector<extent_type> extent = {4, 8, 12};
        int vectorWidth = 1;
        int vectorMode = 1;
        callingInfo("TensorDescriptor");
        TensorDescriptor desc(extent.size(), &extent[0], NULL, LWDA_R_32F, LWTENSOR_OP_IDENTITY, vectorWidth, vectorMode);
        /* CHANGE: print() has been removed from the safety project. */
        //desc.print();
    }

    /**
     * \id TensorDescriptor6
     * \brief test stuct TensorDescriptor with invalid stride.
     * \depends None
     * \setup extent, stride, vectorWidth and vectorMode
     * \testprocedure ./apiTest --gtest_filter=TYPESPLC3_CPP.TensorDescriptor6
     * \teardown None
     * \designid LWTENSOR_DES_003, LWTENSOR_DES_009, LWTENSOR_DES_010
     * \testgroup TYPESPLC3_CPP
     * \inputs parameters of TensorDescriptor
     * \outputs None
     * \expected States: catch the exception: "all strides must be the multiple of the vector width.\n"
     */
    TEST(TYPESPLC3_CPP, TensorDescriptor6)
    {
        try{
            std::vector<extent_type> extent = {4, 8, 12};
            std::vector<stride_type> stride = {1, 2, 4};
            int vectorWidth = 2;
            int vectorMode = 1;
            callingInfo("TensorDescriptor");
            TensorDescriptor desc(extent.size(), &extent[0], &stride[0], LWDA_R_32F, LWTENSOR_OP_IDENTITY, vectorWidth, vectorMode);
        }
        catch(const std::exception& e) {
            EXPECT_TRUE(strcmp(e.what(), "all strides must be the multiple of the vector width.\n")==0);
        }
    }

    /**
     * \id TensorDescriptor7
     * \brief test stuct TensorDescriptor with invalid lwdaDataType_t
     * \depends None
     * \setup extent, stride, vectorWidth and vectorMode
     * \testprocedure ./apiTest --gtest_filter=TYPESPLC3_CPP.TensorDescriptor7
     * \teardown None
     * \testgroup TYPESPLC3_CPP
     * \inputs parameters of TensorDescriptor
     * \outputs None
     * \expected States: catch the exception: "all strides must be the multiple of the vector width.\n"
     */
    TEST(TYPESPLC3_CPP, TensorDescriptor7)
    {
        try{
            std::vector<extent_type> extent = {4, 8, 12};
            callingInfo("TensorDescriptor");
            TensorDescriptor desc(extent.size(), &extent[0], NULL, (lwdaDataType_t)(-1), LWTENSOR_OP_IDENTITY, 1, 0);
        }
        catch(const std::exception& e) {
            EXPECT_TRUE(strcmp(e.what(), "Datatype is not yet supported.\n")==0);
        }
    }

    /**
     * \id TensorDescriptor8
     * \brief test stuct TensorDescriptor with zeroPadding.
     * \depends None
     * \setup extent, stride, vectorWidth and vectorMode
     * \testprocedure ./apiTest --gtest_filter=TYPESPLC3_CPP.TensorDescriptor8
     * \teardown None
     * \designid LWTENSOR_DES_011
     * \testgroup TYPESPLC3_CPP
     * \inputs parameters of TensorDescriptor
     * \outputs None
     * \expected States: catch the exception: "all strides must be the multiple of the vector width.\n"
     */
    TEST(TYPESPLC3_CPP, TensorDescriptor8)
    {
        std::vector<extent_type> extent = {4, 8, 12};
        callingInfo("TensorDescriptor");
        TensorDescriptor desc(extent.size(), &extent[0], NULL, LWDA_R_32F, LWTENSOR_OP_IDENTITY, 2, 1, 0, true);
    }

    /**
     * \id TensorDescriptor9
     * \brief test stuct TensorDescriptor with invalid vector offset.
     * \depends None
     * \setup extent, stride, vectorWidth and vectorMode
     * \testprocedure ./apiTest --gtest_filter=TYPESPLC3_CPP.TensorDescriptor9
     * \teardown None
     * \designid LWTENSOR_DES_003, LWTENSOR_DES_012
     * \testgroup TYPESPLC3_CPP
     * \inputs parameters of TensorDescriptor
     * \outputs None
     * \expected States: catch the exception: "Vector offset is invalid.\n"
     */
    TEST(TYPESPLC3_CPP, TensorDescriptor9)
    {
        try{
            std::vector<extent_type> extent = {4, 8, 12};
            callingInfo("TensorDescriptor");
            TensorDescriptor desc(extent.size(), &extent[0], NULL, LWDA_R_32F, LWTENSOR_OP_IDENTITY, 2, 1, 2, true);
        }
        catch(const std::exception& e) {
            EXPECT_TRUE(strcmp(e.what(), "Vector offset is invalid.\n")==0);
        }
    }

    /**
     * \id toLwdaDataType
     * \brief test function toLwdaDataType
     * \depends None
     * \setup None
     * \testprocedure ./apiTest --gtest_filter=TYPESPLC3_H.toLwdaDataType
     * \teardown None
     * \testgroup TYPESPLC3_H
     * \inputs None
     * \outputs None
     * \expected States: return expected lwca data type
     */
    TEST(TYPESPLC3_H, toLwdaDataType)
    {
        EXPECT_EQ(toLwdaDataType<double>(), LWDA_R_64F);
        EXPECT_EQ(toLwdaDataType<lwComplex>(), LWDA_C_32F);
        EXPECT_EQ(toLwdaDataType<lwDoubleComplex>(), LWDA_C_64F);
    }

    // TODO: failed when running with other test cases
    /**
     * \id elementwiseDispatchAlgorithmTest
     * \brief test function elementwiseDispatchAlgorithm with null alpha
     * \depends None
     * \setup parameters and context initialization, memory allocation, et al. (implemented in class ApiTestDefault)
     * \testprocedure ./apiTest --gtest_filter=ApiTestDefault.elementwiseDispatchAlgorithmTest
     * \teardown context destroy, memory free, et al. (implemented in class ApiTestDefault)
     * \testgroup TYPESPLC3_H
     * \inputs None
     * \outputs None
     * \expected States LWTENSOR_STATUS_SUCCESS
     */
//    TEST_F(ApiTestDefault, elementwiseDispatchAlgorithmTest0)
//    {
//        using typePack =  LWTENSOR_NAMESPACE::ElementwiseStaticTypePack<float, float, float, float>;
//        ElementwiseOpPack opPack;
//        using LWTENSOR_NAMESPACE::Context;
//        using LWTENSOR_NAMESPACE::TensorDescriptor;
//
//        const TensorDescriptor *descA_ = static_cast<const TensorDescriptor *>(descA);
//        const TensorDescriptor *descB_ = static_cast<const TensorDescriptor *>(descB);
//        const TensorDescriptor *descC_ = static_cast<const TensorDescriptor *>(descC);
//        const TensorDescriptor *descD_ = static_cast<const TensorDescriptor *>(descC);
//        LWTENSOR_NAMESPACE::ElementwisePlan plan;
//        lwtensorStatus_t status;
//        status = LWTENSOR_NAMESPACE::elementwiseTrinaryCreate(
//                opts.alpha, *descA_, opts.modeA.data(),
//                opts.beta, *descB_, opts.modeB.data(),
//                opts.gamma, *descC_, opts.modeC.data(),
//                *descD_, opts.modeC.data(),
//                opts.opAB, opts.opUnaryAfterBinary, opts.opABC, LWDA_R_32F, plan);
//        ElementwiseParameters params = plan.params_;
//        opPack.opA = LWTENSOR_OP_IDENTITY;
//        opPack.opB = LWTENSOR_OP_IDENTITY;
//        opPack.opC = LWTENSOR_OP_IDENTITY;
//        opPack.opAB = LWTENSOR_OP_ADD;
//        opPack.opUnaryAfterBinary = LWTENSOR_OP_IDENTITY;
//        opPack.opABC = LWTENSOR_OP_ADD;
//
//        lwdaError_t err = lwdaSuccess;
//        err = lwdaGetLastError();
//        if(err != lwdaSuccess)
//        {
//            std::cout << "something faild with device: errid(" << int(err) << ") in line: " << __LINE__ << std::endl;
//        }
//        using typeA = typename typePack::typeA;
//        using typeB = typename typePack::typeB;
//        using typeC = typename typePack::typeC;
//        using typeCompute = typename typePack::typeCompute;
//        status = elementwiseDispatchAlgorithm<3U,
//               ElementwiseConfig<1, 256,  64, 1>,
//               ElementwiseConfig<2,  16, 256, 1>,
//               ElementwiseConfig<2,  32, 256, 1>,
//               typeA, typeB, typeC, typeCompute> (
//                       params, opPack,
//                       static_cast<const typeCompute *>(nullptr),
//                       static_cast<const typeA *>(A_d),
//                       static_cast<const typeCompute *>(opts.beta),
//                       static_cast<const typeB *>(B_d),
//                       static_cast<const typeCompute *>(opts.gamma),
//                       static_cast<const typeC *>(C_d),
//                       static_cast<typeC *>(C_d),
//                       0);
//        EXPECT_EQ(status, LWTENSOR_STATUS_SUCCESS);
//        err = lwdaGetLastError();
//        if(err != lwdaSuccess)
//        {
//            std::cout << "something faild with device: errid(" << int(err) << ") in line: " << __LINE__ << std::endl;
//        }
//    }


    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //              tests for src/util.cpp
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /**
     * \id isZero0
     * \brief validate the functionality of isZero with type LWDA_R_8I
     * \depends None
     * \setup None
     * \testprocedure ./apiTest --gtest_filter=UTIL_CPP.isZero0
     * \teardown None
     * \testgroup UTIL_CPP
     * \inputs None
     * \outputs None
     * \expected States: the expected value
     */
    TEST(UTIL_CPP, isZero0)
    {
        void * ptr;
        ptr = (void *)malloc(8);
        memset(ptr, 0, 8);
        callingInfo("isZero");
        EXPECT_TRUE(isZero(ptr, LWDA_R_8I));
        free(ptr);
    }

    /**
     * \id isZero1
     * \brief validate the functionality of isZero with type LWDA_R_8U
     * \depends None
     * \setup None
     * \testprocedure ./apiTest --gtest_filter=UTIL_CPP.isZero1
     * \teardown None
     * \testgroup UTIL_CPP
     * \inputs None
     * \outputs None
     * \expected States: the expected value
     */
    TEST(UTIL_CPP, isZero1)
    {
        void * ptr;
        ptr = (void *)malloc(8);
        memset(ptr, 0, 8);
        callingInfo("isZero");
        EXPECT_TRUE(isZero(ptr, LWDA_R_8U));
        free(ptr);
    }

    /**
     * \id isZero2
     * \brief validate the functionality of isZero with type LWDA_R_32U
     * \depends None
     * \setup None
     * \testprocedure ./apiTest --gtest_filter=UTIL_CPP.isZero2
     * \teardown None
     * \testgroup UTIL_CPP
     * \inputs None
     * \outputs None
     * \expected States: the expected value
     */
    TEST(UTIL_CPP, isZero2)
    {
        void * ptr;
        ptr = (void *)malloc(32);
        memset(ptr, 0, 32);
        callingInfo("isZero");
        EXPECT_TRUE(isZero(ptr, LWDA_R_32U));
        free(ptr);
    }

    /**
     * \id isZero3
     * \brief validate the functionality of isZero with unsupported data types
     * \depends None
     * \setup None
     * \testprocedure ./apiTest --gtest_filter=UTIL_CPP.isZero3
     * \teardown None
     * \testgroup UTIL_CPP
     * \inputs None
     * \outputs None
     * \expected States: the expected value
     */
    TEST(UTIL_CPP, isZero3)
    {
        try{
            callingInfo("isZero");
            isZero(NULL, (lwdaDataType_t)1999);
        }
        catch(const std::exception& e) {
            EXPECT_TRUE(strcmp(e.what(), "Datatype is not yet supported (isZero).\n")==0);
        }
    }

    /**
     * \id isZero4
     * \brief validate the functionality of isZero with type LWDA_R_32I
     * \depends None
     * \setup None
     * \testprocedure ./apiTest --gtest_filter=UTIL_CPP.isZero4
     * \teardown None
     * \testgroup UTIL_CPP
     * \inputs None
     * \outputs None
     * \expected States: the expected value
     */
    TEST(UTIL_CPP, isZero4)
    {
        void * ptr;
        ptr = (void *)malloc(32);
        memset(ptr, 0, 32);
        callingInfo("isZero");
        EXPECT_TRUE(isZero(ptr, LWDA_R_32I));
        free(ptr);
    }

    /**
     * \id isZero5
     * \brief validate the functionality of isZero with type LWDA_R_16F
     * \depends None
     * \setup None
     * \testprocedure ./apiTest --gtest_filter=UTIL_CPP.isZero5
     * \teardown None
     * \testgroup UTIL_CPP
     * \inputs None
     * \outputs None
     * \expected States: the expected value
     */
    TEST(UTIL_CPP, isZero5)
    {
        void * ptr;
        ptr = (void *)malloc(16);
        memset(ptr, 0, 16);
        callingInfo("isZero");
        EXPECT_TRUE(isZero(ptr, LWDA_R_16F));
        free(ptr);
    }

    /**
     * \id isZero6
     * \brief validate the functionality of isZero with type LWDA_R_32F
     * \depends None
     * \setup None
     * \testprocedure ./apiTest --gtest_filter=UTIL_CPP.isZero6
     * \teardown None
     * \testgroup UTIL_CPP
     * \inputs None
     * \outputs None
     * \expected States: the expected value
     */
    TEST(UTIL_CPP, isZero6)
    {
        void * ptr;
        ptr = (void *)malloc(32);
        memset(ptr, 0, 32);
        callingInfo("isZero");
        EXPECT_TRUE(isZero(ptr, LWDA_R_32F));
        free(ptr);
    }

    /**
     * \id isZero7
     * \brief validate the functionality of isZero with type LWDA_R_64F
     * \depends None
     * \setup None
     * \testprocedure ./apiTest --gtest_filter=UTIL_CPP.isZero7
     * \teardown None
     * \testgroup UTIL_CPP
     * \inputs None
     * \outputs None
     * \expected States: the expected value
     */
    TEST(UTIL_CPP, isZero7)
    {
        void * ptr;
        ptr = (void *)malloc(64);
        memset(ptr, 0, 64);
        callingInfo("isZero");
        EXPECT_TRUE(isZero(ptr, LWDA_R_64F));
        free(ptr);
    }

    /**
     * \id isZero8
     * \brief validate the functionality of isZero with type LWDA_C_32F
     * \depends None
     * \setup None
     * \testprocedure ./apiTest --gtest_filter=UTIL_CPP.isZero8
     * \teardown None
     * \testgroup UTIL_CPP
     * \inputs None
     * \outputs None
     * \expected States: the expected value
     */
    TEST(UTIL_CPP, isZero8)
    {
        void * ptr;
        ptr = (void *)malloc(32);
        memset(ptr, 0, 32);
        callingInfo("isZero");
        EXPECT_TRUE(isZero(ptr, LWDA_C_32F));
        free(ptr);
    }

    TEST(UTIL_CPP, isZero9)
    {
        void * ptr;
        ptr = (void *)malloc(32);
        memset(ptr, 0, 32);
        callingInfo("isZero");
        EXPECT_TRUE(isZero(ptr, LWDA_C_32F));
        free(ptr);
    }

    TEST(UTIL_CPP, isZero10)
    {
        void * ptr;
        ptr = (void *)malloc(64);
        memset(ptr, 0, 64);
        callingInfo("isZero");
        EXPECT_TRUE(isZero(ptr, LWDA_C_64F));
        free(ptr);
    }

    /**
     * \id isZero11
     * \brief validate the functionality of isZero with type LWDA_R_8I
     * \depends None
     * \setup None
     * \testprocedure ./apiTest --gtest_filter=UTIL_CPP.isZero11
     * \teardown None
     * \testgroup UTIL_CPP
     * \inputs None
     * \outputs None
     * \expected States: the expected value
     */
    TEST(UTIL_CPP, isZero11)
    {
        void * ptr;
        ptr = (void *)malloc(8);
        memset(ptr, 1, 8);
        callingInfo("isZero");
        EXPECT_FALSE(isZero(ptr, LWDA_R_8I));
        free(ptr);
    }

    /**
     * \id hasDuplicates
     * \brief validate the functionality of hasDuplicates
     * \depends None
     * \setup None
     * \testprocedure ./apiTest --gtest_filter=UTIL_CPP.hasDuplicates
     * \teardown None
     * \testgroup UTIL_CPP
     * \inputs None
     * \outputs None
     * \expected States: the expected value
     */
    TEST(UTIL_CPP, hasDuplicates)
    {
        std::vector<mode_type> mode = {'a', 'b', 'b'};
        callingInfo("hasDuplicates");
        EXPECT_TRUE(hasDuplicates(&mode[0], mode.size()));
    }

    /**
     * \id lwtensorGetErrorString
     * \brief validate the functionality of lwtensorGetErrorString
     * \depends None
     * \setup None
     * \testprocedure ./apiTest --gtest_filter=UTIL_CPP.lwtensorGetErrorString
     * \teardown None
     * \testgroup UTIL_CPP
     * \inputs None
     * \outputs None
     * \expected States: the expected value
     */
    TEST(UTIL_CPP, lwtensorGetErrorString)
    {
        callingInfo("lwtensorGetErrorString");
        EXPECT_TRUE(strcmp(lwtensorGetErrorString(LWTENSOR_STATUS_SUCCESS), "LWTENSOR_STATUS_SUCCESS") == 0);
        EXPECT_TRUE(strcmp(lwtensorGetErrorString(LWTENSOR_STATUS_NOT_INITIALIZED), "LWTENSOR_STATUS_NOT_INITIALIZED") == 0);
        EXPECT_TRUE(strcmp(lwtensorGetErrorString(LWTENSOR_STATUS_ALLOC_FAILED), "LWTENSOR_STATUS_ALLOC_FAILED") == 0);
        EXPECT_TRUE(strcmp(lwtensorGetErrorString(LWTENSOR_STATUS_ILWALID_VALUE), "LWTENSOR_STATUS_ILWALID_VALUE") == 0);
        EXPECT_TRUE(strcmp(lwtensorGetErrorString(LWTENSOR_STATUS_ARCH_MISMATCH), "LWTENSOR_STATUS_ARCH_MISMATCH") == 0);
        EXPECT_TRUE(strcmp(lwtensorGetErrorString(LWTENSOR_STATUS_MAPPING_ERROR), "LWTENSOR_STATUS_MAPPING_ERROR") == 0);
        EXPECT_TRUE(strcmp(lwtensorGetErrorString(LWTENSOR_STATUS_EXELWTION_FAILED), "LWTENSOR_STATUS_EXELWTION_FAILED") == 0);
        EXPECT_TRUE(strcmp(lwtensorGetErrorString(LWTENSOR_STATUS_NOT_SUPPORTED), "LWTENSOR_STATUS_NOT_SUPPORTED") == 0);
        EXPECT_TRUE(strcmp(lwtensorGetErrorString(LWTENSOR_STATUS_LICENSE_ERROR), "LWTENSOR_STATUS_LICENSE_ERROR") == 0);
        EXPECT_TRUE(strcmp(lwtensorGetErrorString(LWTENSOR_STATUS_LWBLAS_ERROR), "LWTENSOR_STATUS_LWBLAS_ERROR") == 0);
        EXPECT_TRUE(strcmp(lwtensorGetErrorString(LWTENSOR_STATUS_LWDA_ERROR), "LWTENSOR_STATUS_LWDA_ERROR") == 0);
        EXPECT_TRUE(strcmp(lwtensorGetErrorString(LWTENSOR_STATUS_INTERNAL_ERROR), "LWTENSOR_STATUS_INTERNAL_ERROR") == 0);
        EXPECT_TRUE(strcmp(lwtensorGetErrorString(LWTENSOR_STATUS_INSUFFICIENT_WORKSPACE), "LWTENSOR_STATUS_INSUFFICIENT_WORKSPACE") == 0);
        EXPECT_TRUE(strcmp(lwtensorGetErrorString(LWTENSOR_STATUS_INSUFFICIENT_DRIVER), "LWTENSOR_STATUS_INSUFFICIENT_DRIVER") == 0);
        EXPECT_TRUE(strcmp(lwtensorGetErrorString(lwtensorStatus_t(999)), "<unknown>") == 0);
    }

    /**
     * \id HadnleError0
     * \brief validate the functionality of HandleError with lwdaErrorNoDevice
     * \depends None
     * \setup None
     * \testprocedure ./apiTest --gtest_filter=UTIL_CPP.HandleError0
     * \teardown None
     * \testgroup UTIL_CPP
     * \inputs None
     * \outputs None
     * \expected States: LWTENSOR_STATUS_ARCH_MISMATCH
     */
    TEST(UTIL_CPP, HandleError0)
    {
        callingInfo("handleError");
        lwdaError_t err = lwdaErrorNoDevice;
        EXPECT_EQ(handleError(err), LWTENSOR_STATUS_ARCH_MISMATCH);
    }

    /**
     * \id HadnleError1
     * \brief validate the functionality of HandleError with lwdaErrorIlwalidValue
     * \depends None
     * \setup None
     * \testprocedure ./apiTest --gtest_filter=UTIL_CPP.HandleError1
     * \teardown None
     * \testgroup UTIL_CPP
     * \inputs None
     * \outputs None
     * \expected States: LWTENSOR_STATUS_LWDA_ERROR
     */
    TEST(UTIL_CPP, HandleError1)
    {
        callingInfo("handleError");
        lwdaError_t err = lwdaErrorIlwalidValue;
        EXPECT_EQ(handleError(err), LWTENSOR_STATUS_LWDA_ERROR);
    }

    /**
     * \id HadnleError2
     * \brief validate the functionality of HandleError with  lwdaErrorInsufficientDriver
     * \depends None
     * \setup None
     * \testprocedure ./apiTest --gtest_filter=UTIL_CPP.HandleError2
     * \teardown None
     * \testgroup UTIL_CPP
     * \inputs None
     * \outputs None
     * \expected States: LWTENSOR_STATUS_INSUFFICIENT_DRIVER
     */
    TEST(UTIL_CPP, HandleError2)
    {
        callingInfo("handleError");
        lwdaError_t err = lwdaErrorInsufficientDriver;
        EXPECT_EQ(handleError(err), LWTENSOR_STATUS_INSUFFICIENT_DRIVER);
    }

    /**
     * \id validateStride0
     * \brief validate the functionality of validateStride with invalid stride
     * \depends None
     * \setup None
     * \testprocedure ./apiTest --gtest_filter=UTIL_CPP.validateStride0
     * \teardown None
     * \testgroup UTIL_CPP
     * \inputs None
     * \outputs None
     * \expected States:  LWTENSOR_STATUS_INTERNAL_ERROR
     */
    TEST(UTIL_CPP, validateStride0)
    {
        ModeList modes = {0xa, 0xb, 0xc};
        StrideMap strides;
        strides[0xc] = 2;
        strides[0xb] = 1;
        callingInfo("validateStride");
        EXPECT_EQ(validateStride(strides, modes), LWTENSOR_STATUS_INTERNAL_ERROR);
    }

    /**
     * \id validateStride1
     * \brief validate the functionality of validateStride with empty stride
     * \depends None
     * \setup None
     * \testprocedure ./apiTest --gtest_filter=UTIL_CPP.validateStride1
     * \teardown None
     * \testgroup UTIL_CPP
     * \inputs None
     * \outputs None
     * \expected States:  LWTENSOR_STATUS_INTERNAL_ERROR
     */
    TEST(UTIL_CPP, validateStride1)
    {
        ModeList modes = {0xa, 0xb, 0xc};
        StrideMap strides;
        callingInfo("validateStride");
        EXPECT_EQ(validateStride(strides, modes), LWTENSOR_STATUS_INTERNAL_ERROR);
    }

    /**
     * \id validateStride3
     * \brief validate the functionality of validateStride with invalid stride
     * \depends None
     * \setup None
     * \testprocedure ./apiTest --gtest_filter=UTIL_CPP.validateStride3
     * \teardown None
     * \testgroup UTIL_CPP
     * \inputs None
     * \outputs None
     * \expected States:  LWTENSOR_STATUS_INTERNAL_ERROR
     */
    TEST(UTIL_CPP, validateStride3)
    {
        ModeList modes = {0xa, 0xb, 0xc};
        StrideMap strides;
        strides[0xa] = 8; //error
        strides[0xb] = 2;
        strides[0xc] = 4;
        callingInfo("validateStride");
        EXPECT_EQ(validateStride(strides, modes), LWTENSOR_STATUS_INTERNAL_ERROR);
    }

    /**
     * \id validateStride4
     * \brief validate the functionality of validateStride with invalid stride
     * \depends None
     * \setup None
     * \testprocedure ./apiTest --gtest_filter=UTIL_CPP.validateStride4
     * \teardown None
     * \testgroup UTIL_CPP
     * \inputs None
     * \outputs None
     * \expected States:  LWTENSOR_STATUS_INTERNAL_ERROR
     */
    TEST(UTIL_CPP, validateStride4)
    {
        ModeList modes = {0xa, 0xb, 0xc};
        StrideMap strides;
        strides[0xd] = 8;
        strides[0xb] = 2;
        strides[0xc] = 4;
        callingInfo("validateStride");
        EXPECT_EQ(validateStride(strides, modes ), LWTENSOR_STATUS_INTERNAL_ERROR);
    }

    /**
     * \id validateStride5
     * \brief validate the functionality of validateStride with invalid stride
     * \depends None
     * \setup None
     * \testprocedure ./apiTest --gtest_filter=UTIL_CPP.validateStride5
     * \teardown None
     * \testgroup UTIL_CPP
     * \inputs None
     * \outputs None
     * \expected States:  LWTENSOR_STATUS_INTERNAL_ERROR
     */
    TEST(UTIL_CPP, validateStride5)
    {
        ModeList modes = {0xa, 0xb, 0xc};
        StrideMap strides;
        strides[0xa] = 8;
        strides[0xc] = 2;
        strides[0xd] = 4;
        callingInfo("validateStride");
        EXPECT_EQ(validateStride(strides, modes ), LWTENSOR_STATUS_INTERNAL_ERROR);
    }

    /**
     * \id initStride0
     * \brief validate the functionality of initStride with normal inputs
     * \depends None
     * \setup None
     * \testprocedure ./apiTest --gtest_filter=UTIL_CPP.initStride0
     * \teardown None
     * \testgroup UTIL_CPP
     * \inputs None
     * \outputs None
     * \expected States:  LWTENSOR_STATUS_SUCCESS
     */
    TEST(UTIL_CPP, initStride0)
    {
        ModeList mode = {0xa, 0xb, 0xc};
        ExtentMap extent;
        extent[0xa] = 1;
        extent[0xb] = 2;
        extent[0xc] = 3;
        StrideMap stride;
        callingInfo("validateStride");
        EXPECT_EQ(initStride(extent, mode, stride), LWTENSOR_STATUS_SUCCESS);
    }

    /**
     * \id initStride1
     * \brief validate the functionality of initStride with invalid extent and stride
     * \depends None
     * \setup None
     * \testprocedure ./apiTest --gtest_filter=UTIL_CPP.initStride1
     * \teardown None
     * \testgroup UTIL_CPP
     * \inputs None
     * \outputs None
     * \expected States:  LWTENSOR_STATUS_SUCCESS
     */
    TEST(UTIL_CPP, initStride1)
    {
        ModeList  mode = {0xa, 0xb, 0xc};
        ExtentMap extent;
        extent[0xa] = 1;
        StrideMap stride;
        callingInfo("validateStride");
        EXPECT_EQ(initStride(extent, mode, stride), LWTENSOR_STATUS_INTERNAL_ERROR);
    }

    /**
     * \id initStrideExtentModesSorted1
     * \brief validate the functionality of initStrideExtentModesSorted with invalid inputs
     * \depends None
     * \setup None
     * \testprocedure ./apiTest --gtest_filter=UTIL_CPP.initStrideExtentModesSorted1
     * \teardown None
     * \testgroup UTIL_CPP
     * \inputs None
     * \outputs None
     * \expected States: LWTENSOR_STATUS_ILWALID_VALUE
     */
    TEST(UTIL_CPP, initStrideExtentModesSorted1)
    {
        //       constexpr int MAX_DIM = TensorDescriptor::LWTENSOR_MAX_MODES;
        ExtentMap extent;
        StrideMap strides;
        ModeList modeA_sorted;

        uint32_t nmodeA = 3;
        //       mode_type modeA[] = {0xa,0xb,0xc};
        extent_type extentA[] = {20,10,30};
        TensorDescriptor descA(nmodeA, extentA, NULL, LWDA_R_32F, LWTENSOR_OP_IDENTITY, 1, 0);
        mode_type modeB[] = {0xc,0xb,0xa};
        extent[0xc] = 30;
        //EXPECT_EQ(initStrideExtentModesSorted(&descA, modeA, strides, modeA_sorted, extent), LWTENSOR_STATUS_ILWALID_VALUE);
        callingInfo("initStrideExtentModesSorted");
        EXPECT_EQ(initStrideExtentModesSorted(&descA, modeB, strides, modeA_sorted, extent), LWTENSOR_STATUS_ILWALID_VALUE);
    }

    /**
     * \id getDataTypeSize0
     * \brief validate the functionality of getDataTypeSize with unsupported dataType
     * \depends None
     * \setup None
     * \testprocedure ./apiTest --gtest_filter=UTIL_CPP.getDataTypeSize0
     * \teardown None
     * \testgroup UTIL_CPP
     * \inputs None
     * \outputs None
     * \expected States: expected error message
     */
    TEST(UTIL_CPP, getDataTypeSize0)
    {
        try{
            callingInfo("getDataTypeSize");
            getDataTypeSize(lwdaDataType_t(999));
        } catch(const std::exception& e) {
            EXPECT_TRUE(strcmp(e.what(), "Datatype is not yet supported.\n")==0);
        }
    }

    TEST(UTIL_CPP, isValidUnaryOperator0)
    {
        callingInfo("isValidUnaryOperator");
        EXPECT_TRUE(isValidUnaryOperator(LWTENSOR_OP_RCP, LWDA_R_32F));
    }

    TEST(UTIL_CPP, isValidUnaryOperator1)
    {
        callingInfo("isValidUnaryOperator");
        EXPECT_FALSE(isValidUnaryOperator(LWTENSOR_OP_RCP, LWDA_C_32F));
    }

    TEST(UTIL_CPP, isValidUnaryOperator2)
    {
        callingInfo("isValidUnaryOperator");
        EXPECT_FALSE(isValidUnaryOperator(LWTENSOR_OP_CONJ, LWDA_R_8I));
    }

    TEST(UTIL_CPP, isValidUnaryOperator3)
    {
        callingInfo("isValidUnaryOperator");
        EXPECT_FALSE(isValidUnaryOperator(LWTENSOR_OP_CONJ, lwdaDataType_t(999)));
    }

    TEST(UTIL_CPP, handleException_InternalError)
    {
        std::exception err;
        EXPECT_EQ(handleException(err), LWTENSOR_STATUS_INTERNAL_ERROR);
    }

    TEST_F(ApiTestDefault, pwValidateInput0)
    {
        callingInfo("lwtensorElementwiseTrinary");
        EXPECT_EQ(lwtensorElementwiseTrinary(&handle,
                    opts.alpha, A_d, &descA, opts.modeA.data(),
                    opts.beta,  B_d, &descB, opts.modeB.data(),
                    opts.gamma, C_d, &descC, opts.modeC.data(),
                    C_d, &descA, opts.modeC.data(), //descD != &descC
                    opts.opAB, opts.opABC, opts.typeCompute, 0 /* stream */
                    ), LWTENSOR_STATUS_NOT_SUPPORTED);

    }

    TEST_F(ApiTestDefault, pwValidateInput1)
    {
        std::vector<int64_t> extent;
        for (auto mode : opts.modeA)
            extent.push_back(opts.extent[mode]);
        lwtensorTensorDescriptor_t desc;
        // conj not allowed for float
        EXPECT_EQ(lwtensorInitTensorDescriptor(&handle, &desc, opts.modeA.size(), extent.data(),
                opts.strideA, opts.typeA, LWTENSOR_OP_CONJ, 1, 0), LWTENSOR_STATUS_ILWALID_VALUE);
        callingInfo("lwtensorElementwiseTrinary");
        // desc not initialized
        EXPECT_EQ(lwtensorElementwiseTrinary(&handle,
                    opts.alpha, A_d, &desc,  opts.modeA.data(),
                    opts.beta,  B_d, &descB, opts.modeB.data(),
                    opts.gamma, C_d, &descC, opts.modeC.data(),
                    C_d, &descC, opts.modeC.data(),
                    opts.opAB, opts.opABC, opts.typeCompute, 0 /* stream */
                    ), LWTENSOR_STATUS_ILWALID_VALUE);
    }

    TEST_F(ApiTestDefault, pwValidateInput2)
    {
        std::vector<int64_t> extent;
        for (auto mode : opts.modeA)
            extent.push_back(opts.extent[mode]);
        lwtensorTensorDescriptor_t desc;
        // conj not allowed for float
        EXPECT_EQ(lwtensorInitTensorDescriptor(&handle, &desc, opts.modeA.size(), extent.data(),
                opts.strideA, opts.typeA, LWTENSOR_OP_CONJ, 1, 0), LWTENSOR_STATUS_ILWALID_VALUE);

        callingInfo("lwtensorElementwiseTrinary");
        // desc not initialized
        EXPECT_EQ(lwtensorElementwiseTrinary(&handle,
                    opts.alpha, A_d, &descA,  opts.modeA.data(),
                    opts.alpha,  B_d, &desc, opts.modeB.data(), /* let beta = alpha */
                    opts.gamma, C_d, &descC, opts.modeC.data(),
                    C_d, &descC, opts.modeC.data(),
                    opts.opAB, opts.opABC, opts.typeCompute, 0 /* stream */
                    ), LWTENSOR_STATUS_ILWALID_VALUE);
    }

    TEST_F(ApiTestDefault, pwValidateInput3)
    {
        std::vector<int64_t> extent;
        for (auto mode : opts.modeA)
            extent.push_back(opts.extent[mode]);
        lwtensorTensorDescriptor_t desc;
        // conj not allowed for float
        EXPECT_EQ(lwtensorInitTensorDescriptor(&handle, &desc, opts.modeA.size(), extent.data(),
                opts.strideA, opts.typeA, LWTENSOR_OP_CONJ, 1, 0), LWTENSOR_STATUS_ILWALID_VALUE);

        callingInfo("lwtensorElementwiseTrinary");
        // desc not initialized
        EXPECT_EQ(lwtensorElementwiseTrinary(&handle,
                    opts.alpha, A_d, &descA,  opts.modeA.data(),
                    opts.beta,  B_d, &descB, opts.modeB.data(),
                    opts.gamma, C_d, &desc, opts.modeC.data(),
                    C_d, &desc, opts.modeC.data(),
                    opts.opAB, opts.opABC, opts.typeCompute, 0 /* stream */
                    ), LWTENSOR_STATUS_ILWALID_VALUE);
    }

    TEST_F(ApiTestDefault, pwValidateInput4)
    {
        std::vector<int64_t> extent;
        for (auto mode : opts.modeA)
            extent.push_back(opts.extent[mode]);
        lwtensorTensorDescriptor_t desc;
        // conj not allowed for float
        EXPECT_EQ(lwtensorInitTensorDescriptor(&handle, &desc, opts.modeA.size(), extent.data(),
                opts.strideA, opts.typeA, LWTENSOR_OP_CONJ, 1, 0), LWTENSOR_STATUS_ILWALID_VALUE);
        callingInfo("lwtensorElementwiseTrinary");
        // desc not initialized
        EXPECT_EQ(lwtensorElementwiseTrinary(&handle,
                    opts.alpha, A_d, &descA,  opts.modeA.data(),
                    opts.beta,  B_d, &descB, opts.modeB.data(),
                    opts.gamma, C_d, &descC, opts.modeC.data(),
                    C_d, &descC, opts.modeC.data(),
                    opts.opAB, LWTENSOR_OP_MAX, LWDA_C_32F, 0 /* stream */
                    ), LWTENSOR_STATUS_ILWALID_VALUE);
    }

    TEST_F(ApiTestDefault, pwValidateInput5) //TODO: line 128
    {
        callingInfo("lwtensorElementwiseTrinary");
        EXPECT_EQ(lwtensorElementwiseTrinary(&handle,
                    opts.alpha, A_d, &descA, nullptr,
                    opts.beta,  B_d, &descB, opts.modeB.data(),
                    opts.gamma, C_d, &descC, opts.modeC.data(),
                    C_d, &descC, opts.modeC.data(),
                    opts.opAB, opts.opABC, opts.typeCompute, 0 /* stream */
                    ), LWTENSOR_STATUS_ILWALID_VALUE);
    }

    /* TODO: elementwise no longer needs the handle. */
    //TEST_F(ApiTestDefault, pwValidateInput6)
    //{
    //    struct thandle{
    //        thandle() : isInitialized(false) {}
    //        uint32_t numSMs;
    //        bool isInitialized;
    //        uint32_t blocking[6];
    //    };
    //    struct thandle tmp;
    //    lwtensorHandle_t handle0 = static_cast<lwtensorHandle_t>(&tmp);
    //    callingInfo("lwtensorElementwiseTrinary");
    //    EXPECT_EQ(lwtensorElementwiseTrinary(handle0,
    //                opts.alpha, A_d, &descA, opts.modeA.data(),
    //                opts.beta,  B_d, &descB, opts.modeB.data(),
    //                opts.gamma, C_d, &descC, opts.modeC.data(),
    //                            C_d, &descC, opts.modeC.data(),
    //                opts.opAB, opts.opABC, opts.typeCompute, 0 /* stream */
    //                ), LWTENSOR_STATUS_NOT_INITIALIZED);
    //}

    TEST_F(ApiTestDefault, pwValidateInput7)
    {
        opts.modeA[0] = 'a';
        opts.modeA[1] = 'a';
        lwtensorStatus_t exp_status;
        exp_status =LWTENSOR_STATUS_ILWALID_VALUE;

        callingInfo("lwtensorElementwiseTrinary");
        EXPECT_EQ(lwtensorElementwiseTrinary(&handle,
                    opts.alpha, A_d, &descA, opts.modeA.data(),
                    opts.beta,  B_d, &descB, opts.modeB.data(),
                    opts.gamma, C_d, &descC, opts.modeC.data(),
                    C_d, &descC, opts.modeC.data(),
                    opts.opAB, opts.opABC, opts.typeCompute, 0 /* stream */
                    ), exp_status);
    }

    TEST_F(ApiTestDefault, pwValidateInput8)
    {
        opts.modeA[0] = -2;
        callingInfo("lwtensorElementwiseTrinary");
        EXPECT_EQ(lwtensorElementwiseTrinary(&handle,
                    opts.alpha, A_d, &descA, opts.modeA.data(),
                    opts.beta,  B_d, &descB, opts.modeB.data(),
                    opts.gamma, C_d, &descC, opts.modeC.data(),
                    C_d, &descC, opts.modeC.data(),
                    opts.opAB, opts.opABC, opts.typeCompute, 0 /* stream */
                    ), LWTENSOR_STATUS_ILWALID_VALUE);
    }

    TEST_F(ApiTestDefault, pwValidateInput9)
    {
        opts.modeA[0] = 'e'; // mode 'e' not in the output tensor
        callingInfo("lwtensorElementwiseTrinary");
        EXPECT_EQ(lwtensorElementwiseTrinary(&handle,
                    opts.alpha, A_d, &descA, opts.modeA.data(),
                    opts.beta,  B_d, &descB, opts.modeB.data(),
                    opts.gamma, C_d, &descC, opts.modeC.data(),
                    C_d, &descC, opts.modeC.data(),
                    opts.opAB, opts.opABC, opts.typeCompute, 0 /* stream */
                    ), LWTENSOR_STATUS_ILWALID_VALUE);
    }

    // Since we internally re-number, negative modes are allowed again
    // TEST_F(ApiTestDefault, pwValidateInput10)
    // {
    //     opts.modeC[3] = -2;
    //     callingInfo("lwtensorElementwiseTrinary");
    //     EXPECT_EQ(lwtensorElementwiseTrinary(&handle,
    //                 opts.alpha, A_d, &descA, opts.modeA.data(),
    //                 opts.beta,  B_d, &descB, opts.modeB.data(),
    //                 opts.gamma, C_d, &descC, opts.modeC.data(),
    //                 C_d, &descC, opts.modeC.data(),
    //                 opts.opAB, opts.opABC, opts.typeCompute, 0 /* stream */
    //                 ), LWTENSOR_STATUS_ILWALID_VALUE);
    // }

    TEST_F(ApiTestDefault, pwValidateInput11)
    {
        opts.modeB[0] = -2;
        callingInfo("lwtensorElementwiseTrinary");
        EXPECT_EQ(lwtensorElementwiseTrinary(&handle,
                    opts.alpha, A_d, &descA, opts.modeA.data(),
                    opts.alpha,  B_d, &descB, opts.modeB.data(),
                    opts.gamma, C_d, &descC, opts.modeC.data(),
                    C_d, &descC, opts.modeC.data(),
                    opts.opAB, opts.opABC, opts.typeCompute, 0 /* stream */
                    ), LWTENSOR_STATUS_ILWALID_VALUE);
    }

    TEST_F(ApiTestDefault, pwValidateInput12)
    {
        opts.modeB[0] = 'e'; // mode 'e' not in the output tensor
        callingInfo("lwtensorElementwiseTrinary");
        EXPECT_EQ(lwtensorElementwiseTrinary(&handle,
                    opts.alpha, A_d, &descA, opts.modeA.data(),
                    opts.alpha,  B_d, &descB, opts.modeB.data(),
                    opts.gamma, C_d, &descC, opts.modeC.data(),
                    C_d, &descC, opts.modeC.data(),
                    opts.opAB, opts.opABC, opts.typeCompute, 0 /* stream */
                    ), LWTENSOR_STATUS_ILWALID_VALUE);
    }

    TEST_F(ApiTestDefault, pwValidateInput13)
    {
        callingInfo("lwtensorElementwiseTrinary");
        EXPECT_EQ(lwtensorElementwiseTrinary(&handle,
                    opts.alpha, A_d, &descA, opts.modeA.data(),
                    opts.alpha,  B_d, &descB, opts.modeB.data(),
                    opts.gamma, C_d, &descC, opts.modeC.data(),
                    NULL, &descC, opts.modeC.data(),
                    opts.opAB, opts.opABC, opts.typeCompute, 0 /* stream */
                    ), LWTENSOR_STATUS_ILWALID_VALUE);
    }

    TEST_F(ApiTestDefault, pwValidateInputDirect0)
    {
        TensorDescriptor * desca = reinterpret_cast<TensorDescriptor *>(&descA);
        TensorDescriptor * descb = reinterpret_cast<TensorDescriptor *>(&descB);
        TensorDescriptor * descc = reinterpret_cast<TensorDescriptor *>(&descC);
        try {
            callingInfo("pwValidateInput");
            EXPECT_EQ(pwValidateInput(
                        (lwtensor_internal_namespace::Context *)&handle,
                        opts.alpha, desca, opts.modeA.data(),
                        opts.alpha, descb, opts.modeB.data(),
                        opts.gamma, descc, opts.modeC.data(),
                        descc,
                        opts.opA, LWTENSOR_OP_CONJ, opts.opC,
                        opts.opAB, opts.opABC, LWDA_R_32F,
                        true, true, true /* useA, useB, useC */
                        ), LWTENSOR_STATUS_ILWALID_VALUE);

        }
        catch (std::exception &e) {
            EXPECT_TRUE(strcmp(e.what(), "Operator opB invalid.\n")==0);
        }
    }


    TEST_F(ApiTestDefault, pwValidateInputDirect1)
    {
        TensorDescriptor * desca = reinterpret_cast<TensorDescriptor *>(&descA);
        TensorDescriptor * descb = reinterpret_cast<TensorDescriptor *>(&descB);
        TensorDescriptor * descc = reinterpret_cast<TensorDescriptor *>(&descC);
        try {
            callingInfo("pwValidateInput");
            EXPECT_EQ(pwValidateInput(
                        (lwtensor_internal_namespace::Context *)&handle,
                        opts.alpha, desca, opts.modeA.data(),
                        opts.alpha, descb, opts.modeB.data(),
                        opts.gamma, descc, opts.modeC.data(),
                        descc,
                        opts.opA, opts.opB, LWTENSOR_OP_CONJ,
                        opts.opAB, opts.opABC, LWDA_R_32F,
                        true, true, true /* useA, useB, useC */
                        ), LWTENSOR_STATUS_ILWALID_VALUE);

        }
        catch (std::exception &e) {
            EXPECT_TRUE(strcmp(e.what(), "Operator opC invalid.\n")==0);
        }
    }

    TEST_F(ApiTestDefault, pwValidateInputDirect2)
    {
        TensorDescriptor * desca = reinterpret_cast<TensorDescriptor *>(&descA);
        TensorDescriptor * descb = reinterpret_cast<TensorDescriptor *>(&descB);
        TensorDescriptor * descc = reinterpret_cast<TensorDescriptor *>(&descC);
        try {
            callingInfo("pwValidateInput");
            EXPECT_EQ(pwValidateInput((lwtensor_internal_namespace::Context *)&handle,
                        opts.alpha, desca, nullptr,
                        opts.alpha, descb, opts.modeB.data(),
                        opts.gamma, descc, opts.modeC.data(),
                        descc,
                        opts.opA, opts.opB, opts.opC,
                        opts.opAB, opts.opABC, LWDA_R_32F,
                        true, true, true /* useA, useB, useC */
                        ), LWTENSOR_STATUS_ILWALID_VALUE);

        }
        catch (std::exception &e) {
            EXPECT_TRUE(strcmp(e.what(), "modeA may not be null.\n")==0);
        }
    }

    TEST_F(ApiTestDefault, pwValidateInputDirect3)
    {
        TensorDescriptor * desca = reinterpret_cast<TensorDescriptor *>(&descA);
        TensorDescriptor * descb = reinterpret_cast<TensorDescriptor *>(&descB);
        TensorDescriptor * descc = reinterpret_cast<TensorDescriptor *>(&descC);
        try {
            callingInfo("pwValidateInput");
            EXPECT_EQ(pwValidateInput((lwtensor_internal_namespace::Context *)&handle,
                        opts.alpha, desca, opts.modeA.data(),
                        opts.alpha, descb, nullptr,
                        opts.gamma, descc, opts.modeC.data(),
                        descc,
                        opts.opA, opts.opB, opts.opC,
                        opts.opAB, opts.opABC, LWDA_R_32F,
                        true, true, true /* useA, useB, useC */
                        ), LWTENSOR_STATUS_ILWALID_VALUE);

        }
        catch (std::exception &e) {
            EXPECT_TRUE(strcmp(e.what(), "modeB may not be null.\n")==0);
        }
    }

    TEST_F(ApiTestDefault, pwValidateInputDirect4)
    {
        TensorDescriptor * desca = reinterpret_cast<TensorDescriptor *>(&descA);
        TensorDescriptor * descb = reinterpret_cast<TensorDescriptor *>(&descB);
        TensorDescriptor * descc = reinterpret_cast<TensorDescriptor *>(&descC);
        try {
            callingInfo("pwValidateInput");
            EXPECT_EQ(pwValidateInput((lwtensor_internal_namespace::Context *)&handle,
                        opts.alpha, desca, opts.modeA.data(),
                        opts.alpha, descb, opts.modeB.data(),
                        opts.gamma, descc, nullptr,
                        descc,
                        opts.opA, opts.opB, opts.opC,
                        opts.opAB, opts.opABC, LWDA_R_32F,
                        true, true, true /* useA, useB, useC */
                        ), LWTENSOR_STATUS_ILWALID_VALUE);

        }
        catch (std::exception &e) {
            EXPECT_TRUE(strcmp(e.what(), "modeC may not be null.\n")==0);
        }
    }

    TEST_F(ApiTestDefault, pwValidateInputDirect5)
    {
        TensorDescriptor * desca = reinterpret_cast<TensorDescriptor *>(&descA);
        TensorDescriptor * descb = reinterpret_cast<TensorDescriptor *>(&descB);
        try {
            callingInfo("pwValidateInput");
            EXPECT_EQ(pwValidateInput((lwtensor_internal_namespace::Context *)&handle,
                        opts.alpha, desca, opts.modeA.data(),
                        opts.alpha, descb, opts.modeB.data(),
                        opts.gamma, nullptr, opts.modeC.data(),
                        nullptr,
                        opts.opA, opts.opB, opts.opC,
                        opts.opAB, opts.opABC, LWDA_R_32F,
                        true, true, false /* useA, useB, useC */
                        ), LWTENSOR_STATUS_ILWALID_VALUE);

        }
        catch (std::exception &e) {
            EXPECT_TRUE(strcmp(e.what(), "Descriptor for C may not be null.\n")==0);
        }
    }

    TEST_F(ApiTestDefault, pwValidateInputDirect6)
    {
        TensorDescriptor * desca = reinterpret_cast<TensorDescriptor *>(&descA);
        TensorDescriptor * descb = reinterpret_cast<TensorDescriptor *>(&descB);
        TensorDescriptor * descc = reinterpret_cast<TensorDescriptor *>(&descC);
        try {
            callingInfo("pwValidateInput");
            EXPECT_EQ(pwValidateInput((lwtensor_internal_namespace::Context *)&handle,
                        opts.alpha, desca, opts.modeA.data(),
                        opts.alpha, descb, opts.modeB.data(),
                        opts.gamma, descc, opts.modeC.data(),
                        descc,
                        opts.opA, opts.opB, opts.opC,
                        opts.opAB, opts.opABC, LWDA_R_32F,
                        false, true, true /* useA, useB, useC */
                        ), LWTENSOR_STATUS_ILWALID_VALUE);

        }
        catch (std::exception &e) {
            EXPECT_TRUE(strcmp(e.what(), "alpha, *alpha, and A must not be zero.\n")==0);
        }
    }

    TEST_F(ApiTestDefault, pwValidateInputDirect7)
    {
        opts.modeA[0] = 'a';
        opts.modeA[1] = 'a';
        TensorDescriptor * desca = reinterpret_cast<TensorDescriptor *>(&descA);
        TensorDescriptor * descb = reinterpret_cast<TensorDescriptor *>(&descB);
        TensorDescriptor * descc = reinterpret_cast<TensorDescriptor *>(&descC);
        try {
            callingInfo("pwValidateInput");
            EXPECT_EQ(pwValidateInput((lwtensor_internal_namespace::Context *)&handle,
                        opts.alpha, desca, opts.modeA.data(),
                        opts.alpha, descb, opts.modeB.data(),
                        opts.gamma, descc, opts.modeC.data(),
                        descc,
                        opts.opA, opts.opB, opts.opC,
                        opts.opAB, opts.opABC, LWDA_R_32F,
                        true, true, true /* useA, useB, useC */
                        ), LWTENSOR_STATUS_ILWALID_VALUE);

        }
        catch (std::exception &e) {
            EXPECT_TRUE(strcmp(e.what(), "Each mode may only appear up to once per tensor.\n")==0);
        }
    }


    TEST_F(ApiTestDefault, pwValidateInputDirect_NuseABC)
    {
        opts.modeA[0] = 'a';
        opts.modeA[1] = 'a';
        TensorDescriptor * desca = reinterpret_cast<TensorDescriptor *>(&descA);
        TensorDescriptor * descb = reinterpret_cast<TensorDescriptor *>(&descB);
        TensorDescriptor * descc = reinterpret_cast<TensorDescriptor *>(&descC);
        callingInfo("pwValidateInput");
        EXPECT_EQ(pwValidateInput((lwtensor_internal_namespace::Context *)&handle,
                    opts.alpha, desca, opts.modeA.data(),
                    opts.alpha, descb, opts.modeB.data(),
                    opts.gamma, descc, opts.modeC.data(),
                    descc,
                    opts.opA, opts.opB, opts.opC,
                    opts.opAB, opts.opABC, LWDA_C_8U, //LWDA_C_8U is not supported
                    false, false, false /* useA, useB, useC */
                    ), LWTENSOR_STATUS_ILWALID_VALUE);
    }

    /**
     * \designid LWTENSOR_DES_001, LWTENSOR_DES_002, LWTENSOR_DES_004, LWTENSOR_DES_005, LWTENSOR_DES_007, LWTENSOR_DES_008, LWTENSOR_DES_009
     */
    TEST_F(ApiTestDefault, lwtensorElementwiseInternal_L1_0)
    {
        callingInfo("lwtensorElementwiseTrinary");
        EXPECT_EQ(lwtensorElementwiseTrinary(&handle,
                    opts.alpha, A_d, &descA, opts.modeA.data(),
                    opts.beta,  B_d, &descB, opts.modeB.data(),
                    opts.gamma, C_d, &descC, opts.modeC.data(),
                    C_d, &descC, opts.modeC.data(),
                    opts.opAB, opts.opABC, opts.typeCompute, 0 /* stream */
                    ), LWTENSOR_STATUS_SUCCESS);
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //              tests for ./include/lwtensor/internal/types.h, these tests were added to enhance the code coverage
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////


    /**
     * \id TensorDescriptor_getExtent
     * \brief test method TensorDescriptor::getExtent
     * \depends None
     * \setup need to construct descriptor(s) in advance
     * \testprocedure ./apiTest --gtest_filter=INTER_TYPES_H.TensorDescriptor_getExtent
     * \teardown None
     * \testgroup INTER_TYPES_H
     * \inputs TensorDescriptor object and the corresponding parameters
     * \outputs None
     * \expected States: catch exception: "Out of bounds!.\n"
     */
    TEST(INTER_TYPES_H, TensorDescriptor_getExtent)
    {
        int numModes = 3;
        std::vector<extent_type> extent = {2, 3, 4};
        std::vector<stride_type> stride = {1, 2, 6};
        struct TensorDescriptor desc(numModes, extent.data(), stride.data());
        try {
            callingInfo("desc.getExtent");
            desc.getExtent(numModes + 1);
        }
        catch(const std::exception& e) {
            EXPECT_TRUE(strcmp(e.what(), "Out of bounds!.\n")==0);
        }
    }

    /**
     * \id TensorDescriptor_getStride
     * \brief test method TensorDescriptor::getStride
     * \depends None
     * \setup need to construct descriptor(s) in advance
     * \testprocedure ./apiTest --gtest_filter=INTER_TYPES_H.TensorDescriptor_getStride
     * \teardown None
     * \testgroup INTER_TYPES_H
     * \inputs TensorDescriptor object and the corresponding parameters
     * \outputs None
     * \expected States: catch exception: "Out of bounds!.\n"
     */
    TEST(INTER_TYPES_H, TensorDescriptor_getStride)
    {
        int numModes = 3;
        std::vector<extent_type> extent = {2, 3, 4};
        std::vector<stride_type> stride = {1, 2, 6};
        struct TensorDescriptor desc(numModes, extent.data(), stride.data());
        try {
            callingInfo("desc.getStride");
            desc.getStride(numModes + 1);
        }
        catch(const std::exception& e) {
            EXPECT_TRUE(strcmp(e.what(), "Out of bounds!.\n")==0);
        }
    }

    /**
     * \id TensorDescriptor_hasSameVectorization0
     * \brief test method TensorDescriptor::hasSameVectorization
     * \depends None
     * \setup need to construct descriptor(s) in advance
     * \testprocedure ./apiTest --gtest_filter=INTER_TYPES_H.TensorDescriptor_hasSameVectorization0
     * \teardown None
     * \testgroup INTER_TYPES_H
     * \inputs TensorDescriptor object and the corresponding parameters
     * \outputs None
     * \expected States: return True
     */
    TEST(INTER_TYPES_H, TensorDescriptor_hasSameVectorization0)
    {
        int numModes = 3;
        std::vector<extent_type> extent = {2, 3, 4};
        std::vector<stride_type> stride = {1, 2, 6};
        struct TensorDescriptor descA(numModes, extent.data(), stride.data());
        struct TensorDescriptor descB(numModes, extent.data(), stride.data());
        callingInfo("desc.hasSameVectorization");
        EXPECT_TRUE(descA.hasSameVectorization(descB));
    }

    /**
     * \id TensorDescriptor_hasSameVectorization1
     * \brief test method TensorDescriptor::hasSameVectorization
     * \depends None
     * \setup need to construct descriptor(s) in advance
     * \testprocedure ./apiTest --gtest_filter=INTER_TYPES_H.TensorDescriptor_hasSameVectorization1
     * \teardown None
     * \testgroup INTER_TYPES_H
     * \inputs TensorDescriptor object and the corresponding parameters
     * \outputs None
     * \expected States: return False
     */
    TEST(INTER_TYPES_H, TensorDescriptor_hasSameVectorization1)
    {
        int numModes = 3;
        std::vector<extent_type> extent = {2, 3, 4};
        std::vector<stride_type> strideA = {1, 2, 6};
        std::vector<stride_type> strideB = {2, 4, 12};
        struct TensorDescriptor descA(numModes,
                extent.data(), strideA.data(), LWDA_R_32F, LWTENSOR_OP_IDENTITY, 1, 0);
        struct TensorDescriptor descB(numModes,
                extent.data(), strideB.data(), LWDA_R_32F, LWTENSOR_OP_IDENTITY, 2, 1);
        callingInfo("desc.hasSameVectorization");
        EXPECT_FALSE(descA.hasSameVectorization(descB));
    }

    /**
     * \id TensorDescriptor_isSimilar0
     * \brief test method TensorDescriptor::isSimilar
     * \depends None
     * \setup need to construct descriptor(s) in advance
     * \testprocedure ./apiTest --gtest_filter=INTER_TYPES_H.TensorDescriptor_isSimilar0
     * \teardown None
     * \testgroup INTER_TYPES_H
     * \inputs TensorDescriptor object and the corresponding parameters
     * \outputs None
     * \expected States: return True
     */
    TEST(INTER_TYPES_H, TensorDescriptor_isSimilar0)
    {
        int numModes = 3;
        std::vector<extent_type> extent = {2, 3, 4};
        std::vector<stride_type> stride = {1, 2, 6};
        struct TensorDescriptor descA(numModes,
                extent.data(), stride.data(), LWDA_R_32F, LWTENSOR_OP_IDENTITY, 1, 0);
        struct TensorDescriptor descB(numModes,
                extent.data(), stride.data(), LWDA_R_32F, LWTENSOR_OP_IDENTITY, 1, 0);
        callingInfo("desc.isSimilar");
        EXPECT_TRUE(descA.isSimilar(descB));
    }

    /**
     * \id TensorDescriptor_isSimilar1
     * \brief test method TensorDescriptor::isSimilar
     * \depends None
     * \setup need to construct descriptor(s) in advance
     * \testprocedure ./apiTest --gtest_filter=INTER_TYPES_H.TensorDescriptor_isSimilar1
     * \teardown None
     * \testgroup INTER_TYPES_H
     * \inputs TensorDescriptor object and the corresponding parameters
     * \outputs None
     * \expected States: return False
     */
    TEST(INTER_TYPES_H, TensorDescriptor_isSimilar1)
    {
        int numModes = 3;
        std::vector<extent_type> extent = {2, 3, 4};
        std::vector<stride_type> stride = {1, 2, 6};
        struct TensorDescriptor descA(numModes,
                extent.data(), stride.data(), LWDA_R_32F, LWTENSOR_OP_IDENTITY, 1, 0);
        struct TensorDescriptor descB(numModes - 1,
                extent.data(), stride.data(), LWDA_R_32F, LWTENSOR_OP_IDENTITY, 1, 0);
        callingInfo("desc.isSimilar");
        EXPECT_FALSE(descA.isSimilar(descB));
    }

    /**
     * \id TensorDescriptor_isSimilar2
     * \brief test method TensorDescriptor::isSimilar
     * \depends None
     * \setup need to construct descriptor(s) in advance
     * \testprocedure ./apiTest --gtest_filter=INTER_TYPES_H.TensorDescriptor_isSimilar2
     * \teardown None
     * \testgroup INTER_TYPES_H
     * \inputs TensorDescriptor object and the corresponding parameters
     * \outputs None
     * \expected States: return False
     */
    TEST(INTER_TYPES_H, TensorDescriptor_isSimilar2)
    {
        int numModes = 3;
        std::vector<extent_type> extentA = {2, 3, 4};
        std::vector<stride_type> strideA = {1, 2, 6};
        std::vector<extent_type> extentB = {4, 3, 4};
        std::vector<stride_type> strideB = {1, 4, 12};
        struct TensorDescriptor descA(numModes,
                extentA.data(), strideA.data(), LWDA_R_32F, LWTENSOR_OP_IDENTITY, 1, 0);
        struct TensorDescriptor descB(numModes - 1,
                extentB.data(), strideB.data(), LWDA_R_32F, LWTENSOR_OP_IDENTITY, 1, 0);
        callingInfo("desc.isSimilar");
        EXPECT_FALSE(descA.isSimilar(descB));
    }

    /**
     * \id TensorDescriptor_isSimilar3
     * \brief test method TensorDescriptor::isSimilar with different extent
     * \depends None
     * \setup need to construct descriptor(s) in advance
     * \testprocedure ./apiTest --gtest_filter=INTER_TYPES_H.TensorDescriptor_isSimilar3
     * \teardown None
     * \testgroup INTER_TYPES_H
     * \inputs TensorDescriptor object and the corresponding parameters
     * \outputs None
     * \expected States: return False
     */
    TEST(INTER_TYPES_H, TensorDescriptor_isSimilar3)
    {
        int numModes = 3;
        std::vector<extent_type> extentA = {2, 3, 4};
        std::vector<stride_type> strideA = {1, 2, 6};
        std::vector<extent_type> extentB = {4, 3, 4};
        std::vector<stride_type> strideB = {1, 4, 12};
        struct TensorDescriptor descA(numModes,
                extentA.data(), strideA.data(), LWDA_R_32F, LWTENSOR_OP_IDENTITY, 1, 0);
        struct TensorDescriptor descB(numModes,
                extentB.data(), strideB.data(), LWDA_R_32F, LWTENSOR_OP_IDENTITY, 1, 0);
        callingInfo("desc.isSimilar");
        EXPECT_FALSE(descA.isSimilar(descB));
    }

    /**
     * \id TensorDescriptor_operator0
     * \brief test whether two TensorDescriptor objects are equal
     * \depends None
     * \setup need to construct descriptor(s) in advance
     * \testprocedure ./apiTest --gtest_filter=INTER_TYPES_H.TensorDescriptor_operator0
     * \teardown None
     * \testgroup INTER_TYPES_H
     * \inputs TensorDescriptor object and the corresponding parameters
     * \outputs None
     * \expected States: Returns False
     */
    TEST(INTER_TYPES_H, TensorDescriptor_operator0)
    {
        int numModes = 3;
        std::vector<extent_type> extent = {2, 3, 4};
        std::vector<stride_type> stride = {1, 2, 6};
        struct TensorDescriptor descA(numModes,
                extent.data(), stride.data(), LWDA_R_32F, LWTENSOR_OP_IDENTITY, 1, 0);
        struct TensorDescriptor descB(numModes - 1,
                extent.data(), stride.data(), LWDA_R_32F, LWTENSOR_OP_IDENTITY, 1, 0);
        EXPECT_FALSE(descA == descB);
    }

    /**
     * \id TensorDescriptor_operator1
     * \brief test whether two TensorDescriptor objects are equal
     * \depends None
     * \setup need to construct descriptor(s) in advance
     * \testprocedure ./apiTest --gtest_filter=INTER_TYPES_H.TensorDescriptor_operator1
     * \teardown None
     * \testgroup INTER_TYPES_H
     * \inputs TensorDescriptor object and the corresponding parameters
     * \outputs None
     * \expected States: return False
     */
    TEST(INTER_TYPES_H, TensorDescriptor_operator1)
    {
        int numModes = 3;
        std::vector<extent_type> extent = {2, 3, 4};
        std::vector<stride_type> strideA = {1, 2, 6};
        std::vector<stride_type> strideB = {1, 2, 8};
        struct TensorDescriptor descA(numModes,
                extent.data(), strideA.data(), LWDA_R_32F, LWTENSOR_OP_IDENTITY, 1, 0);
        struct TensorDescriptor descB(numModes,
                extent.data(), strideB.data(), LWDA_R_32F, LWTENSOR_OP_IDENTITY, 1, 0);
        EXPECT_FALSE(descA == descB);
    }

    /**
     * \id TensorDescriptor_operator2
     * \brief test whether two TensorDescriptor objects are equal
     * \depends None
     * \setup need to construct descriptor(s) in advance
     * \testprocedure ./apiTest --gtest_filter=INTER_TYPES_H.TensorDescriptor_operator2
     * \teardown None
     * \testgroup INTER_TYPES_H
     * \inputs TensorDescriptor object and the corresponding parameters
     * \outputs None
     * \expected States: return True
     */
    TEST(INTER_TYPES_H, TensorDescriptor_operator2)
    {
        int numModes = 3;
        std::vector<extent_type> extent = {2, 3, 4};
        std::vector<stride_type> stride = {1, 2, 6};
        struct TensorDescriptor descA(numModes,
                extent.data(), stride.data());
        struct TensorDescriptor descB(numModes,
                extent.data(), stride.data());
        EXPECT_TRUE(descA == descB);
    }

    /**
     * \id TensorDescriptor_setOp
     * \brief test method TensorDescriptor::getOp
     * \depends None
     * \setup need to construct descriptor(s) in advance
     * \testprocedure ./apiTest --gtest_filter=INTER_TYPES_H.TensorDescriptor_setOp
     * \teardown None
     * \testgroup INTER_TYPES_H
     * \inputs TensorDescriptor object and the corresponding parameters
     * \outputs None
     * \expected States: check whether the returned op is as expected
     */
    TEST(INTER_TYPES_H, TensorDescriptor_setOp)
    {
        int numModes = 3;
        std::vector<extent_type> extent = {2, 3, 4};
        std::vector<stride_type> stride = {1, 2, 6};
        struct TensorDescriptor desc(numModes,
                extent.data(), stride.data());
        desc.setOp(LWTENSOR_OP_RCP);
        callingInfo("desc.getOp");
        lwtensorOperator_t op = desc.getOp();

        EXPECT_EQ(op, LWTENSOR_OP_RCP);
    }

    /**
     * \id IlwalidArgumentTest
     * \brief test the constructor and method of class IlwalidArgument
     * \depends None
     * \setup None
     * \testprocedure ./apiTest --gtest_filter=EXCEPTIONS_H.IlwalidArgumentTest
     * \teardown None
     * \testgroup EXCEPTIONS_H
     * \inputs a string
     * \outputs  a description string
     * \expected States: return the description string
     */
    TEST(EXCEPTIONS_H, IlwalidArgumentTest)
    {
        std::string args = "stride";
        LWTENSOR_NAMESPACE::IlwalidArgument e(args.c_str());
        std::cout << e.what() << std::endl;
        EXPECT_TRUE(strcmp(e.what(), "Invalid Argument: stride") == 0);
    }

    /**
     * \id IlwalidArgument0
     * \brief test the constructor and method of class IlwalidArgument with NULL string
     * \depends None
     * \setup None
     * \testprocedure ./apiTest --gtest_filter=EXCEPTIONS_H.IlwalidArgument0
     * \teardown None
     * \testgroup EXCEPTIONS_H
     * \inputs a string
     * \outputs  a description string
     * \expected States: return the default description string
     */
    TEST(EXCEPTIONS_H, IlwalidArgument0)
    {
        LWTENSOR_NAMESPACE::IlwalidArgument ilwalidArg(NULL, 1);
        EXPECT_TRUE(strcmp(ilwalidArg.what(), "Invalid Argument1") == 0);
    }

    /**
     * \id IlwalidArgument1
     * \brief test the constructor and method of class IlwalidArgument with a string
     * \depends None
     * \setup None
     * \testprocedure ./apiTest --gtest_filter=EXCEPTIONS_H.IlwalidArgument1
     * \teardown None
     * \testgroup EXCEPTIONS_H
     * \inputs a string
     * \outputs  a description string
     * \expected States: return the description string
     */
    TEST(EXCEPTIONS_H, IlwalidArgument1)
    {
        std::string tmp = "testing ";
        LWTENSOR_NAMESPACE::IlwalidArgument ilwalidArg(tmp.c_str(), 1);
        EXPECT_TRUE(strcmp(ilwalidArg.what(), "Invalid Argument: testing 1") == 0);
    }

    /**
     * \id NotInitializedTest
     * \brief test the constructor and method of class NotInitialized
     * \depends None
     * \setup None
     * \testprocedure ./apiTest --gtest_filter=EXCEPTIONS_H.NotInitializedTest
     * \teardown None
     * \testgroup EXCEPTIONS_H
     * \inputs None
     * \outputs  a description string
     * \expected States: return the default description string
     */
    TEST(EXCEPTIONS_H, NotInitializedTest)
    {
        LWTENSOR_NAMESPACE::NotInitialized e;
        EXPECT_TRUE(strcmp(e.what(), "Not Initialized.\n") == 0);
    }

    /**
     * \id NotSupproted
     * \brief test the constructor and method of class NotSupproted
     * \depends None
     * \setup None
     * \testprocedure ./apiTest --gtest_filter=EXCEPTIONS_H.NotSupproted
     * \teardown None
     * \testgroup EXCEPTIONS_H
     * \inputs None
     * \outputs  a null string
     * \expected States: return a default null string
     */
    TEST(EXCEPTIONS_H, NotSupproted)
    {
        LWTENSOR_NAMESPACE::NotSupported e; // to call the default constructor
        EXPECT_TRUE(strcmp(e.what(), "") == 0);
    }

    //TODO: double check why it will lead to error
    //TEST(ELEMENTWISE_CPP, lwtensorPermutationCreate_nullPlan)
    //{
    //    std::vector<int64_t> extent = {4, 8, 12};
    //    lwtensorTensorDescriptor_t desc;
    //    lwtensorInitTensorDescriptor(&desc, 3, &extent[0], NULL, LWDA_R_32F, LWTENSOR_OP_IDENTITY, 1, 0);
    //    TensorDescriptor * descp = static_cast<TensorDescriptor*>(&desc);
    //    std::vector<mode_type_external> mode = {'a', 'b', 'c'};
    //    float alpha = 1.0f;
    //    LWTENSOR_NAMESPACE::ElementwisePlan *p = NULL;
    //
    //    EXPECT_EQ(lwtensorPermutationCreate(static_cast<void *>(&alpha), *descp, &mode[0], *descp, &mode[0], LWDA_R_32F, *p), LWTENSOR_STATUS_SUCCESS);
    //
    //}

    /**
     * \id elementwiseTrinaryCreate0
     * \brief test the function elementwiseTrinaryCreate with invalid modeA
     * \depends None
     * \setup None
     * \testprocedure ./apiTest --gtest_filter=ELEMENTWISE_CPP.lwtensorElementwiseTrinaryCreate0
     * \teardown None
     * \testgroup ELEMENTWISE_CPP
     * \inputs null alpha
     * \outputs None
     * \expected States: return LWTENSOR_STATUS_ILWALID_VALUE
     */
    TEST(ELEMENTWISE_CPP, elementwiseTrinaryCreate0)
    {
        std::vector<int64_t> extent = {4, 8, 12};
        lwtensorHandle_t handle;
        lwtensorTensorDescriptor_t desc;
        std::vector<mode_type_external> mode = {'a', 'b', 'c'};
        lwtensorInit(&handle);
        lwtensorInitTensorDescriptor(&handle, &desc, mode.size(), &extent[0], NULL, LWDA_R_32F, LWTENSOR_OP_IDENTITY, 1, 0);
        const TensorDescriptor * descp = reinterpret_cast<const TensorDescriptor*>(&desc);
        float alpha = 1.0f;
        LWTENSOR_NAMESPACE::ElementwisePlan p;

        auto ctx = reinterpret_cast<const Context*>(&handle);
        EXPECT_EQ(LWTENSOR_NAMESPACE::elementwiseTrinaryCreate(
                    ctx,
                    static_cast<void *>(&alpha), *descp, NULL, 256,
                    static_cast<void *>(&alpha), *descp, &mode[0], 256,
                    static_cast<void *>(&alpha), *descp, &mode[0], 256,
                    *descp, &mode[0], 256, LWTENSOR_OP_ADD, LWTENSOR_OP_ADD,
                    LWDA_R_32F, p), LWTENSOR_STATUS_ILWALID_VALUE);
    }

    /**
     * \id elementwiseBinaryCreate0
     * \brief test the function elementwiseBinaryCreate with invalid modeA
     * \depends None
     * \setup extent, mode and descriptor
     * \testprocedure ./apiTest --gtest_filter=ELEMENTWISE_CPP.lwtensorElementwiseBinaryCreate0
     * \teardown None
     * \testgroup ELEMENTWISE_CPP
     * \inputs null alpha
     * \outputs None
     * \expected States: return LWTENSOR_STATUS_ILWALID_VALUE
     */
    TEST(ELEMENTWISE_CPP, elementwiseBinaryCreate0)
    {
        std::vector<int64_t> extent = {4, 8, 12};
        lwtensorHandle_t handle;
        lwtensorTensorDescriptor_t desc;
        std::vector<mode_type_external> mode = {'a', 'b', 'c'};
        lwtensorInit(&handle);
        lwtensorInitTensorDescriptor(&handle, &desc, mode.size(), &extent[0], NULL, LWDA_R_32F, LWTENSOR_OP_IDENTITY, 1, 0);
        const TensorDescriptor * descp = reinterpret_cast<const TensorDescriptor*>(&desc);
        float alpha = 1.0f;
        LWTENSOR_NAMESPACE::ElementwisePlan p;

        auto ctx = reinterpret_cast<const Context*>(&handle);
        EXPECT_EQ(LWTENSOR_NAMESPACE::elementwiseBinaryCreate(
                    ctx,
                    static_cast<void *>(&alpha), *descp, NULL, 256,
                    static_cast<void *>(&alpha), *descp, &mode[0], 256,
                    *descp, &mode[0], 256, LWTENSOR_OP_ADD, LWDA_R_32F, p
                    ), LWTENSOR_STATUS_ILWALID_VALUE);
    }

    /**
     * \id elementwisePermutationCreate0
     * \brief test the function lwtensorPermutationCreate with invalid modeA
     * \depends None
     * \setup extent, mode and descriptor
     * \testprocedure ./apiTest --gtest_filter=ELEMENTWISE_CPP.lwtensorElementwisePermutationCreate0
     * \teardown None
     * \testgroup ELEMENTWISE_CPP
     * \inputs null alpha
     * \outputs None
     * \expected States: return LWTENSOR_STATUS_ILWALID_VALUE
     */
    TEST(ELEMENTWISE_CPP, elementwisePermutationCreate0)
    {
        std::vector<int64_t> extent = {4, 8, 12};
        lwtensorHandle_t handle;
        lwtensorTensorDescriptor_t desc;
        std::vector<mode_type_external> mode = {'a', 'b', 'c'};
        lwtensorInit(&handle);
        lwtensorInitTensorDescriptor(&handle, &desc, mode.size(), &extent[0], NULL, LWDA_R_32F, LWTENSOR_OP_IDENTITY, 1, 0);
        const TensorDescriptor * descp = reinterpret_cast<const TensorDescriptor*>(&desc);
        float alpha = 1.0f;
        LWTENSOR_NAMESPACE::ElementwisePlan p;

        auto ctx = reinterpret_cast<const Context*>(&handle);
        EXPECT_EQ(LWTENSOR_NAMESPACE::permutationCreate(
                    ctx,
                    static_cast<void *>(&alpha), *descp, NULL, 256,
                    *descp, &mode[0], 256, LWDA_R_32F, p
                    ), LWTENSOR_STATUS_ILWALID_VALUE);
    }

    /**
     * \id lwtensorElementwiseInternal_L0_0
     * \brief test the function lwtensorElementwiseInternal_L0 with null descD
     * \depends None
     * \setup extent, mode and descriptor
     * \testprocedure ./apiTest --gtest_filter=ELEMENTWISE_CPP.lwtensorElementwiseInternal_L0_0
     * \teardown None
     * \testgroup ELEMENTWISE_CPP
     * \inputs null alpha
     * \outputs None
     * \expected States: return LWTENSOR_STATUS_NOT_SUPPORTED
     */
    TEST(ELEMENTWISE_CPP, lwtensorElementwiseInternal_L0_0)
    {
        std::vector<int64_t> extent = {4, 8, 12};
        lwtensorHandle_t handle;
        lwtensorTensorDescriptor_t desc;
        std::vector<mode_type_external> mode = {'a', 'b', 'c'};
        lwtensorInit(&handle);
        lwtensorInitTensorDescriptor(&handle, &desc, mode.size(), &extent[0], NULL, LWDA_R_32F, LWTENSOR_OP_IDENTITY, 1, 0);
        const TensorDescriptor * descp = reinterpret_cast<const TensorDescriptor*>(&desc);
        float alpha = 1.0f;

        auto ctx = reinterpret_cast<const Context*>(&handle);
        EXPECT_EQ(lwtensorElementwiseInternal_L0(
                    ctx,
                    NULL, descp, &mode[0], 256,
                    static_cast<void *>(&alpha), descp, &mode[0], 256,
                    static_cast<void *>(&alpha), descp, &mode[0], 256,
                                                 descp, NULL, 256, LWTENSOR_OP_IDENTITY, // raise LWTENSOR_STATUS_NOT_SUPPORTED
                    LWTENSOR_OP_IDENTITY, LWTENSOR_OP_IDENTITY,
                    LWTENSOR_OP_ADD, LWTENSOR_OP_ADD,
                    LWDA_R_32F, NULL), LWTENSOR_STATUS_NOT_SUPPORTED);
    }

    /**
     * \id lwtensorElementwiseInternal_L0_0
     * \brief test the function lwtensorElementwiseInternal_L0 with false useA and useB
     * \depends None
     * \setup extent, mode and descriptor
     * \testprocedure ./apiTest --gtest_filter=ELEMENTWISE_CPP.lwtensorElementwiseInternal_L0_1
     * \teardown None
     * \testgroup ELEMENTWISE_CPP
     * \inputs null alpha
     * \outputs None
     * \expected States: return LWTENSOR_STATUS_NOT_SUPPORTED
     */
    TEST(ELEMENTWISE_CPP, lwtensorElementwiseInternal_L0_1)
    {
        std::vector<int64_t> extent = {4, 8, 12};
        lwtensorHandle_t handle;
        lwtensorTensorDescriptor_t desc;
        std::vector<mode_type_external> mode = {'a', 'b', 'c'};
        lwtensorInit(&handle);
        lwtensorInitTensorDescriptor(&handle, &desc, mode.size(), &extent[0], NULL, LWDA_R_32F, LWTENSOR_OP_IDENTITY, 1, 0);
        const TensorDescriptor * descp = reinterpret_cast<const TensorDescriptor*>(&desc);
        float alpha = 1.0f;
        LWTENSOR_NAMESPACE::ElementwisePlan p;

        auto ctx = reinterpret_cast<const Context*>(&handle);
        EXPECT_EQ(lwtensorElementwiseInternal_L0(
                    ctx,
                    NULL, descp, &mode[0], 256,
                    NULL, descp, &mode[0], 256,
                    static_cast<void *>(&alpha), descp, &mode[0], 256,
                          descp, NULL, 256, LWTENSOR_OP_IDENTITY, //raise LWTENSOR_STATUS_NOT_SUPPORTED
                    LWTENSOR_OP_IDENTITY, LWTENSOR_OP_IDENTITY,
                    LWTENSOR_OP_ADD, LWTENSOR_OP_ADD,
                    LWDA_R_32F, &p), LWTENSOR_STATUS_NOT_SUPPORTED);
    }

    /**
     * \id pwValidateInputFalseUseABC
     * \brief test the function pwValidateInput, invalid useA, useB, useC
     * \depends None
     * \setup None
     * \testprocedure ./apiTest --gtest_filter=ELEMENTWISE_CPP.pwValidateInputFalseUseABC
     * \teardown None
     * \testgroup ELEMENTWISE_CPP
     * \inputs null alpha
     * \outputs None
     * \expected States: LWTENSOR_STATUS_ILWALID_VALUE
     */
    TEST(ELEMENTWISE_CPP, pwValidateInputFalseUseABC)
    {
        EXPECT_EQ(pwValidateInput(nullptr,NULL, NULL, NULL,
                                  NULL, NULL, NULL,
                                  NULL, NULL, NULL,
                                        NULL,
                LWTENSOR_OP_RELU, LWTENSOR_OP_RELU, LWTENSOR_OP_RELU, // invalid operator
                LWTENSOR_OP_IDENTITY/*invalid UnaryOperator*/, LWTENSOR_OP_ADD,
                LWDA_C_32F, false, false, false), LWTENSOR_STATUS_ILWALID_VALUE);

    }

    /**
     * \id validateStrideAError
     * \brief test the function lwtensorElementwiseInternal_L1, invalid strideA
     * \depends None
     * \setup None
     * \testprocedure ./apiTest --gtest_filter=ELEMENTWISE_CPP.validateStrideAError
     * \teardown None
     * \testgroup ELEMENTWISE_CPP
     * \inputs null alpha
     * \outputs None
     * \expected States: LWTENSOR_STATUS_INTERNAL_ERROR
     */
    TEST(ELEMENTWISE_CPP, validateStrideAError)
    {
        StrideMap stride;
        StrideMap strideNULL;
        ExtentMap extents;
        ModeList mode;
        mode.push_back(0xa);
        mode.push_back(0xb);
        extents[0xa] = 40;
        extents[0xb] = 20;
        stride[0xa] = 1;
        stride[0xb] = 40;
        ElementwisePlan dummyPlan; 

        lwtensorHandle_t handle;
        EXPECT_EQ(lwtensorInit(&handle), LWTENSOR_STATUS_SUCCESS);
        EXPECT_EQ(lwtensorElementwiseInternal_L1((Context*)&handle, NULL, LWDA_R_32F, strideNULL, mode, 256,
                                                 NULL, LWDA_R_32F, stride,     mode, 256,
                                                 NULL, LWDA_R_32F, stride,     mode, 256, 256,
                                                 extents, LWTENSOR_OP_IDENTITY, LWTENSOR_OP_IDENTITY, LWTENSOR_OP_IDENTITY,
                                                 LWTENSOR_OP_ADD, LWTENSOR_OP_ADD, LWDA_R_32F, &dummyPlan),
                                                 LWTENSOR_STATUS_INTERNAL_ERROR);
    }

    /**
     * \id validateStrideBError
     * \brief test the function lwtensorElementwiseInternal_L1, invalid strideB
     * \depends None
     * \setup stride, mode and extent initialization
     * \testprocedure ./apiTest --gtest_filter=ELEMENTWISE_CPP.validateStrideBError
     * \teardown None
     * \testgroup ELEMENTWISE_CPP
     * \inputs null alpha
     * \outputs None
     * \expected States: LWTENSOR_STATUS_INTERNAL_ERROR
     */
    TEST(ELEMENTWISE_CPP, validateStrideBError)
    {
        StrideMap stride;
        StrideMap strideNULL;
        ExtentMap extents;
        ModeList mode;
        mode.push_back(0xa);
        mode.push_back(0xb);
        extents[0xa] = 40;
        extents[0xb] = 20;
        stride[0xa] = 1;
        stride[0xb] = 40;
        ElementwisePlan dummyPlan; 

        lwtensorHandle_t handle;
        EXPECT_EQ(lwtensorInit(&handle), LWTENSOR_STATUS_SUCCESS);
        EXPECT_EQ(lwtensorElementwiseInternal_L1((Context*)&handle, NULL, LWDA_R_32F, stride, mode, 256,
                                                 NULL, LWDA_R_32F, strideNULL, mode, 256,
                                                 NULL, LWDA_R_32F, stride,     mode, 256, 256,
                                                 extents, LWTENSOR_OP_IDENTITY, LWTENSOR_OP_IDENTITY, LWTENSOR_OP_IDENTITY,
                                                 LWTENSOR_OP_ADD, LWTENSOR_OP_ADD, LWDA_R_32F, &dummyPlan),
                                                 LWTENSOR_STATUS_INTERNAL_ERROR);
    }

    /**
     * \id validateStrideCError
     * \brief test the function lwtensorElementwiseInternal_L1, invalid strideC
     * \depends None
     * \setup stride, mode and extent initialization
     * \testprocedure ./apiTest --gtest_filter=ELEMENTWISE_CPP.validateStrideCError
     * \teardown None
     * \testgroup ELEMENTWISE_CPP
     * \inputs null alpha
     * \outputs None
     * \expected States: LWTENSOR_STATUS_INTERNAL_ERROR
     */
    TEST(ELEMENTWISE_CPP, validateStrideCError)
    {
        StrideMap stride;
        StrideMap strideNULL;
        ExtentMap extents;
        ModeList mode;
        mode.push_back(0xa);
        mode.push_back(0xb);
        extents[0xa] = 40;
        extents[0xb] = 20;
        stride[0xa] = 1;
        stride[0xb] = 40;
        ElementwisePlan dummyPlan; 

        lwtensorHandle_t handle;
        EXPECT_EQ(lwtensorInit(&handle), LWTENSOR_STATUS_SUCCESS);
        EXPECT_EQ(lwtensorElementwiseInternal_L1((Context*)&handle, NULL, LWDA_R_32F, stride, mode, 256,
                                                 NULL, LWDA_R_32F, stride, mode, 256,
                                                 NULL, LWDA_R_32F, strideNULL, mode, 256, 256,
                                                 extents, LWTENSOR_OP_IDENTITY, LWTENSOR_OP_IDENTITY, LWTENSOR_OP_IDENTITY,
                                                 LWTENSOR_OP_ADD, LWTENSOR_OP_ADD, LWDA_R_32F, &dummyPlan),
                                                 LWTENSOR_STATUS_INTERNAL_ERROR);
    }

    /**
     * \id falsePlan
     * \brief test the function lwtensorElementwiseInternal_L1, null plan
     * \depends None
     * \setup stride, mode and extent initialization
     * \testprocedure ./apiTest --gtest_filter=ELEMENTWISE_CPP.falsePlan
     * \teardown None
     * \testgroup ELEMENTWISE_CPP
     * \inputs null alpha
     * \outputs None
     * \expected States: LWTENSOR_STATUS_SUCCESS
     */
    TEST(ELEMENTWISE_CPP, falsePlan)
    {
        StrideMap stride;
        ExtentMap extents;
        ModeList mode;
        mode.push_back(0xa);
        mode.push_back(0xb);
        extents[0xa] = 40;
        extents[0xb] = 20;
        stride[0xa] = 1;
        stride[0xb] = 40;
        float scaler = 10.f;
        size_t elementsizeA = extents[0xa] * extents[0xb] * sizeof(LWDA_R_32F);
        size_t elementsizeC = extents[0xa] * extents[0xb] * sizeof(LWDA_R_32F);
        ElementwisePlan dummyPlan; 

        lwtensorHandle_t handle;
        EXPECT_EQ(lwtensorInit(&handle), LWTENSOR_STATUS_SUCCESS);
        EXPECT_EQ(lwtensorElementwiseInternal_L1((Context*)&handle, &scaler, LWDA_R_32F, stride, mode, 256,
                                                 NULL, LWDA_R_32F, stride, mode, 256,
                                                 &scaler, LWDA_R_32F, stride, mode, 256, 256,
                                                 extents, LWTENSOR_OP_IDENTITY, LWTENSOR_OP_IDENTITY, LWTENSOR_OP_IDENTITY,
                                                 LWTENSOR_OP_ADD, LWTENSOR_OP_ADD, LWDA_R_32F, &dummyPlan),
                                                 LWTENSOR_STATUS_SUCCESS);
    }

    TEST_F(ApiTestDefault, lwtensorElementwiseBinary_falseOperator)
    {
        callingInfo("lwtensorElementwiseBinary");
        EXPECT_EQ(lwtensorElementwiseBinary(&handle,
                    opts.alpha, A_d, &descA, &opts.modeA[0],
                    opts.gamma, C_d, &descC, &opts.modeC[0],
                    C_d, &descC, &opts.modeC[0],
                    LWTENSOR_OP_UNKNOWN, LWDA_R_32F, 0 /* stream */),
                    LWTENSOR_STATUS_ILWALID_VALUE);
    }

    TEST_F(ApiTestDefault, lwtensorElementwiseBinaryException)
    {
        callingInfo("lwtensorElementwiseBinary");
        // invalid data type
        EXPECT_EQ(lwtensorElementwiseBinary(&handle,
                    opts.alpha, A_d, &descA, &opts.modeA[0],
                    opts.gamma, C_d, &descC, &opts.modeC[0],
                    C_d, &descC, &opts.modeC[0],
                    opts.opABC, lwdaDataType_t(100), 0 /* stream */),
                LWTENSOR_STATUS_ILWALID_VALUE);
    }

    TEST_F(ApiTestDefault, lwtensorElementwiseTrinaryException)
    {
        callingInfo("lwtensorElementwiseTrinary");
        // invalid data type
        EXPECT_EQ(lwtensorElementwiseTrinary(&handle,
                    opts.alpha, A_d, &descA, &opts.modeA[0],
                    opts.beta,  B_d, &descB, &opts.modeB[0],
                    opts.gamma, C_d, &descC, &opts.modeC[0],
                    C_d, &descC, &opts.modeC[0],
                    opts.opAB, opts.opABC, lwdaDataType_t(100), 0 /* stream */),
                LWTENSOR_STATUS_ILWALID_VALUE);
    }

    TEST_F(ApiTestDefault, lwtensorPermutationException)
    {
        callingInfo("lwtensorPermutation");
        // invalid data type
        EXPECT_EQ(lwtensorPermutation(&handle,
                    opts.alpha, A_d, &descA, &opts.modeA[0],
                    C_d, &descC, &opts.modeC[0],
                    lwdaDataType_t(100), 0 /* stream */),
                LWTENSOR_STATUS_ILWALID_VALUE);
    }
} //namespace

#endif /* define LWTENSOR_TEST_APIMASTER_H */
