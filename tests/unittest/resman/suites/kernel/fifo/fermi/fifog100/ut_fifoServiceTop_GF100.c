/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2010-2010 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

/* ******************Unit test cases for fifoServiceTop_GF100 *****************
 *                                                                            *
 * TC1 : Lwrrently it is to demonstrate how to configure the functionality    *
 *       of hal interface, object interface and register access operations.   *
 *       Along with that also specify how to Assert on specifc values in order*
 *       to verify the outcome                                                *
 *                                                                            *
 * TC2 : This demonstrate how to configure PDB properties and also trying to  *
 *       cover 'true' side of condition                                       *
 *       API_GPU_IN_RESET_SANITY_CHECK(pGpu)                                  *
 *****************************************************************************/

#include "rmunit.h" // RmUnitInfra Header
#include "fermi/gf100/dev_fifo.h"

// testcase setu function
void setup_fifoServiceTop_GF100_TC1(LwTest *pTc);
// testcase teardown function
void teardown_fifoServiceTop_GF100_TC1(LwTest *pTc);
// test cases
void fifoServiceTop_GF100_TC1(LwTest *pTc);
void fifoServiceTop_GF100_TC2(LwTest *pTc);

// static objects used in the test cases
static POBJGPU pGpu = NULL;
static POBJFIFO pFifo = NULL;
static POBJRC pRc = NULL;

/*
 * @brief Test Suite where to add all test cases for fifoServiceTop_GF100
 */
LwSuite* suite_fifoServiceTop_GF100()
{
    LwSuite* suite = UTAPI_NEW_SUITE(NULL, NULL);

    UTAPI_ADD_TC(suite,
                 setup_fifoServiceTop_GF100_TC1,
                 fifoServiceTop_GF100_TC1,
                 teardown_fifoServiceTop_GF100_TC1);

    UTAPI_ADD_TC(suite,
                 setup_fifoServiceTop_GF100_TC1,
                 fifoServiceTop_GF100_TC2,
                 teardown_fifoServiceTop_GF100_TC1);

    return suite;
}

/******************************************************************************
 ************************ Setup Function for All test cases *******************
 ******************************************************************************/

/*
 * @brief Setup function for test case 1 and 2
 *
 * @param[in] tc  Test Case pointer
 *
 */

void setup_fifoServiceTop_GF100_TC1(LwTest *tc)
{
    UTAPI_USE_CHIP(GF100);

    pGpu  =  UTAPI_GET_GPU();
    pFifo = GPU_GET_FIFO(pGpu);
    pRc   = GPU_GET_RC(pGpu);
}

/*******************************************************************************
 *********************** TearDown Function for All test cases ******************
 ******************************************************************************/

/*
 * @brief Teardown function for test case 1 and 2
 *
 * @param[in] tc  Test Case pointer
 *
 */
void teardown_fifoServiceTop_GF100_TC1(LwTest *tc)
{
    pGpu = NULL;
    pFifo = NULL;
    pRc = NULL;
    return;
}

/******************************************************************************
 ******************** Actual test cases for fifoServiceTop_GF100 **************
 *****************************************************************************/

/*
 * @brief Test Case 1 : It is to demonstrate how to apply mock for hal,
 *                      object interafce, and register access operations
 *                      and to get user configured values from them.
 *                      Further asserting on expected values in order to
 *                      complete verification part of test.
 *
 * @param[in] tc  Test Case pointer
 *
 */
void fifoServiceTop_GF100_TC1(LwTest *tc)
{
    LwV32 intr;

    //
    // Configure fifoGetNumFifos_GF100 (which is a hal interface)
    // to return value 2 from it
    //
    UTAPI_MockWillReturn("fifoGetNumFifos", 2);

    //
    // Configure register LW_PFIFO_INTR_0 to return value '0xff' from
    // its first read
    //
    UTAPI_INSTALL_READ_RETURN_UNTIL_COUNT(LW_PFIFO_INTR_0, 0xff, 1);

    //
    // Configure register LW_PFIFO_INTR_0 to return value '0xff' from
    // its next 1 read
    //
    UTAPI_INSTALL_READ_RETURN_UNTIL_COUNT(LW_PFIFO_INTR_0, 0xf, 1);

    //
    // Configure gpuSetTimeout0 (which is a object interface)
    // to return value RM_OK from it
    //
    UTAPI_MockWillReturn("gpuSetTimeout", RM_OK);

    //
    // Configure fifoServiceHal_GF100 (which is a hal interface)
    // to return value 'RM_OK' from it
    //
    UTAPI_MockWillReturn("fifoServiceHal", RM_OK);

    // Call SUT : fifoServiceTop_GF100
    intr = fifoServiceTop_GF100(pGpu, pFifo);

    // Assert on pRc->ErrorInfo.fifo_Intr
    UTAPI_ASSERT_INT_EQUALS(tc, 0xff, pRc->ErrorInfo.fifo_Intr);

    //Assert on intr
    UTAPI_ASSERT_INT_EQUALS(tc, 0xf, intr);
}

/*
 * @brief Test Case 1 : It is to demonstrate how to apply set
 *                      PDB property for SUT.
 *
 * @param[in] tc  Test Case pointer
 *
 */
void fifoServiceTop_GF100_TC2(LwTest *tc)
{
    RM_STATUS status;

    // Set PDB property PDB_PROP_GPU_IN_FULLCHIP_RESET
    pGpu->setProperty(pGpu, PDB_PROP_GPU_IN_FULLCHIP_RESET, TRUE);

    // Call SUT : fifoServiceTop_GF100
    status = fifoServiceTop_GF100(pGpu, pFifo);

    // ASSERT on return value
    UTAPI_ASSERT_INT_EQUALS(tc, RM_OK, status);
}
