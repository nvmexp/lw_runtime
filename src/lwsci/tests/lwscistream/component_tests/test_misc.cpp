//! \file
//! \brief LwSciStream APIs unit testing - Miscellaneous.
//!
//! \copyright
//! Copyright (c) 2019-2022 LWPU Corporation. All rights reserved.
//!
//! LWPU Corporation and its licensors retain all intellectual property and
//! proprietary rights in and to this software, related documentation and any
//! modifications thereto. Any use, reproduction, disclosure or distribution
//! of this software and related documentation without an express license
//! agreement from LWPU Corporation is strictly prohibited.

#include "lwscistreamtest.h"
#include "util.h"

//==========================================================================
// Define Miscellaneous test suite.
//==========================================================================
class Miscellaneous :
    public LwSciStreamTest
{
protected:
    LwSciStreamBlock ipcsrc = 0U;
    LwSciStreamBlock ipcdst = 0U;

    Endpoint ipcProd;
    Endpoint ipcCons;

    bool isLwSciIpcInitDone = false;

    virtual void SetUp()
    {
        //TODO: use "threadsafe" setting. "fast" setting spews warning
        // when running death tests.
        // (Lwrrently, with threadsafe setting, on safety QNX gtest fails
        //  to intercept the abort() call, as a result the process will abort
        //  in death tests).
        ::testing::FLAGS_gtest_death_test_style = "fast";
    };

    virtual void setupModulesAndEndpoints()
    {
        strncpy(ipcProd.chname, "itc_test_0", sizeof("itc_test_0"));
        strncpy(ipcCons.chname, "itc_test_1", sizeof("itc_test_1"));

        ASSERT_EQ(LwSciError_Success, LwSciBufModuleOpen(&bufModule));
        ASSERT_EQ(LwSciError_Success, LwSciSyncModuleOpen(&syncModule));
        ASSERT_EQ(LwSciError_Success, LwSciIpcInit());
        isLwSciIpcInitDone = true;

        ASSERT_EQ(LwSciError_Success, LwSciIpcOpenEndpoint(ipcProd.chname,
                    &ipcProd.endpoint));
        LwSciIpcResetEndpoint(ipcProd.endpoint);
        ASSERT_EQ(LwSciError_Success, LwSciIpcOpenEndpoint(ipcCons.chname,
                    &ipcCons.endpoint));
        LwSciIpcResetEndpoint(ipcCons.endpoint);

        ASSERT_EQ(LwSciError_Success, LwSciStreamIpcSrcCreate(
                    ipcProd.endpoint, syncModule, bufModule, &ipcsrc));
        ASSERT_EQ(LwSciError_Success, LwSciStreamIpcDstCreate(
                    ipcCons.endpoint, syncModule, bufModule, &ipcdst));
    }

    virtual void TearDown()
    {
        if (ipcsrc != 0U) {
            LwSciStreamBlockDelete(ipcsrc);
            ipcsrc = 0U;
        }

        if (ipcdst != 0U) {
            LwSciStreamBlockDelete(ipcdst);
            ipcdst = 0U;
        }

        if (isLwSciIpcInitDone) {
            LwSciIpcDeinit();
            isLwSciIpcInitDone = false;
        }

        if (bufModule != nullptr) {
            LwSciBufModuleClose(bufModule);
            bufModule = nullptr;
        }

        if (syncModule != nullptr) {
            LwSciSyncModuleClose(syncModule);
            syncModule = nullptr;
        }
    }
};

//==========================================================================
// Test Case : Query all queryable attributes.
//==========================================================================

TEST_F(Miscellaneous, QueryAttributes)
{
    // Query max sync objects
    int32_t maxSyncObjs = 0U;
    ASSERT_EQ(LwSciError_Success, LwSciStreamAttributeQuery(
                LwSciStreamQueryableAttrib_MaxSyncObj, &maxSyncObjs));
    ASSERT_LE(MAX_SYNC_COUNT, maxSyncObjs);

    // Query max packet elements
    int32_t maxPacketElements = 0U;
    ASSERT_EQ(LwSciError_Success, LwSciStreamAttributeQuery(
                LwSciStreamQueryableAttrib_MaxElements, &maxPacketElements));
    ASSERT_LE(MAX_ELEMENT_PER_PACKET, maxPacketElements);

    // Query max dst connections
    int32_t maxDstConnections = 0U;
    ASSERT_EQ(LwSciError_Success, LwSciStreamAttributeQuery(
                LwSciStreamQueryableAttrib_MaxMulticastOutputs, &maxDstConnections));
    ASSERT_LE(MAX_DST_CONNECTIONS , maxDstConnections);
}

//==========================================================================
// Test Case : double block free
// LwSciStream returns error if a block is deleted twice.
//==========================================================================

TEST_F(Miscellaneous, DoubleBlockFree)
{
    // Create stream blocks
    createBlocks(QueueType::Mailbox, NUM_CONSUMERS);

    // Free all blocks
    ASSERT_EQ(LwSciError_Success, LwSciStreamBlockDelete(producer));
    ASSERT_EQ(LwSciError_Success, LwSciStreamBlockDelete(pool));

    if (NUM_CONSUMERS > 1) {
        ASSERT_EQ(LwSciError_Success, LwSciStreamBlockDelete(multicast));
    }

    for (uint32_t n = 0U; n < NUM_CONSUMERS; ++n) {
        ASSERT_EQ(LwSciError_Success, LwSciStreamBlockDelete(queue[n]));
        ASSERT_EQ(LwSciError_Success, LwSciStreamBlockDelete(consumer[n]));
    }

    // Free all blocks for second time
    CHECK_ERR_OR_PANIC(LwSciError_BadParameter,
        LwSciStreamBlockDelete(producer));
    CHECK_ERR_OR_PANIC(LwSciError_BadParameter,
        LwSciStreamBlockDelete(pool));

    for (uint32_t n = 0U; n < NUM_CONSUMERS; ++n) {
        CHECK_ERR_OR_PANIC(LwSciError_BadParameter,
            LwSciStreamBlockDelete(queue[n]));
        CHECK_ERR_OR_PANIC(LwSciError_BadParameter,
            LwSciStreamBlockDelete(consumer[n]));
    }

    if (NUM_CONSUMERS > 1) {
        CHECK_ERR_OR_PANIC(LwSciError_BadParameter,
            LwSciStreamBlockDelete(multicast));
    }

    // Ilwalidate blocks to prevent freeing again in destructor
    producer = 0U;
    pool = 0U;
    multicast = 0U;
    for (uint32_t n = 0U; n < MAX_CONSUMERS; ++n) {
        queue[n] = 0U;
        consumer[n] = 0U;
    }
}

//==========================================================================
// Test Case : (multicast, ipcsrc) block connection
//==========================================================================

TEST_F(Miscellaneous, MulticastIpcSrcConnection)
{
    setupModulesAndEndpoints();

    ASSERT_EQ(LwSciError_Success, LwSciStreamMulticastCreate(2U, &multicast));

    ASSERT_EQ(LwSciError_Success, LwSciStreamStaticPoolCreate(NUM_PACKETS, &pool));

    ASSERT_EQ(LwSciError_Success, LwSciStreamBlockConnect(multicast, ipcsrc));
}

//==========================================================================
// Test Case : (ipcdst, pool) block connection
// This connection is not part Drive-OS use case
//==========================================================================

TEST_F(Miscellaneous, IpcDstPoolConnection_Negative)
{
    setupModulesAndEndpoints();

    ASSERT_EQ(LwSciError_Success, LwSciStreamStaticPoolCreate(NUM_PACKETS, &pool));

    ASSERT_EQ(LwSciError_AccessDenied, LwSciStreamBlockConnect(ipcdst, pool));
}

//==========================================================================
// Test Case : Retrieve the block internal errors
//==========================================================================

TEST_F(Miscellaneous, BlockInternalErrors)
{
    LwSciError status = LwSciError_AccessDenied;

    createBlocks(QueueType::Mailbox, 2, 2);

    ASSERT_EQ(LwSciError_Success, LwSciSyncModuleOpen(&syncModule));

    ASSERT_EQ(LwSciError_Success,
        LwSciStreamPresentSyncCreate(syncModule, &presentsync));
    ASSERT_NE(0, presentsync);

    ASSERT_EQ(LwSciError_Success,
        LwSciStreamReturnSyncCreate(syncModule, &returnsync));
    ASSERT_NE(0, returnsync);

    /* Retrieve internal error on pool block */
    ASSERT_EQ(LwSciError_Success,
        LwSciStreamBlockErrorGet(pool, &status));
    ASSERT_EQ(LwSciError_Success, status);

    /* reset before next call */
    status = LwSciError_AccessDenied;

    /* Retrieve internal error on queue block */
    ASSERT_EQ(LwSciError_Success,
        LwSciStreamBlockErrorGet(queue[0], &status));
    ASSERT_EQ(LwSciError_Success, status);

    /* reset before next call */
    status = LwSciError_AccessDenied;

    /* Retrieve internal error on multicast block */
    ASSERT_EQ(LwSciError_Success,
        LwSciStreamBlockErrorGet(multicast, &status));
    ASSERT_EQ(LwSciError_Success, status);

    /* reset before next call */
    status = LwSciError_AccessDenied;

    /* Retrieve internal error on producer block */
    ASSERT_EQ(LwSciError_Success,
        LwSciStreamBlockErrorGet(producer, &status));
    ASSERT_EQ(LwSciError_Success, status);

    /* reset before next call */
    status = LwSciError_AccessDenied;

    /* Retrieve internal error on consumer block */
    ASSERT_EQ(LwSciError_Success,
        LwSciStreamBlockErrorGet(consumer[0], &status));
    ASSERT_EQ(LwSciError_Success, status);

    /* reset before next call */
    status = LwSciError_AccessDenied;

    /* Retrieve internal error on presentSync block */
    ASSERT_EQ(LwSciError_Success,
        LwSciStreamBlockErrorGet(presentsync, &status));
    ASSERT_EQ(LwSciError_Success, status);

    /* reset before next call */
    status = LwSciError_AccessDenied;

    /* Retrieve internal error on returnSync block */
    ASSERT_EQ(LwSciError_Success,
        LwSciStreamBlockErrorGet(returnsync, &status));
    ASSERT_EQ(LwSciError_Success, status);

}