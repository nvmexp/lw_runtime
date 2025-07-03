//! \file
//! \brief LwSciStream APIs unit testing - Sync Object Setup.
//!
//! \copyright
//! Copyright (c) 2019-2020 LWPU Corporation. All rights reserved.
//!
//! LWPU Corporation and its licensors retain all intellectual property and
//! proprietary rights in and to this software, related documentation and any
//! modifications thereto. Any use, reproduction, disclosure or distribution
//! of this software and related documentation without an express license
//! agreement from LWPU Corporation is strictly prohibited.

#include "lwscistreamtest.h"
#include <vector>
#include <future>

//==========================================================================
// Define SyncObjSetup test suite.
//==========================================================================

class SyncObjSetup :
    public LwSciStreamTest
{
protected:
    uint32_t prodSyncCount = NUM_SYNCOBJS;
    uint32_t conSyncCount = NUM_SYNCOBJS;

    virtual void SetUp()
    {
        ASSERT_EQ(LwSciError_Success, LwSciBufModuleOpen(&bufModule));
        makeRawBufferAttrList(bufModule, rawBufAttrList);

        ASSERT_EQ(LwSciError_Success, LwSciSyncModuleOpen(&syncModule));

        prodSynchronousOnly = false;
        cpuWaiterAttrList(syncModule, prodSyncAttrList);

        consSynchronousOnly = false;
        cpuWaiterAttrList(syncModule, consSyncAttrList);
    };

    inline void queryMaxSyncCount()
    {
        int32_t maxNumSyncObjs = 0;
        ASSERT_EQ(LwSciError_Success, LwSciStreamAttributeQuery(
            LwSciStreamQueryableAttrib_MaxSyncObj, &maxNumSyncObjs));

        // Make sure the MAX_SYNC_COUNT is less that limit imposed by library
        ASSERT_TRUE(MAX_SYNC_COUNT <= maxNumSyncObjs);

        ASSERT_TRUE(prodSyncCount <= MAX_SYNC_COUNT);
        ASSERT_TRUE((conSyncCount * numConsumers) <= MAX_SYNC_COUNT);
    };

    // Producer sends its sync object requirement to the consumer.
    inline void prodSendSyncAttr()
    {
        for (uint32_t i = 0U; i < prodSyncCount; ++i) {
            ASSERT_EQ(LwSciError_Success,
                LwSciStreamBlockElementWaiterAttrSet(
                    producer, i,
                    prodSynchronousOnly ? nullptr : prodSyncAttrList));
        }
        ASSERT_EQ(LwSciError_Success,
            LwSciStreamBlockSetupStatusSet(
                producer, LwSciStreamSetup_WaiterAttrExport, true));
    };

    // Consumer sends its sync object requirement to the producer.
    inline void consSendSyncAttr()
    {
        for (uint32_t n = 0U; n < numConsumers; n++) {
            for (uint32_t i = 0U; i < conSyncCount; ++i) {
                ASSERT_EQ(LwSciError_Success,
                    LwSciStreamBlockElementWaiterAttrSet(
                        consumer[n], i,
                        consSynchronousOnly ? nullptr : consSyncAttrList));
            }
            ASSERT_EQ(LwSciError_Success,
                LwSciStreamBlockSetupStatusSet(
                    consumer[n], LwSciStreamSetup_WaiterAttrExport, true));
        }
    };

    // Producer receives consumer's sync object requirement.
    inline void prodRecvSyncAttr()
    {
        LwSciStreamEventType event;
        ASSERT_EQ(LwSciError_Success,
            LwSciStreamBlockEventQuery(producer, EVENT_QUERY_TIMEOUT, &event));
        EXPECT_EQ(LwSciStreamEventType_WaiterAttr, event);
        for (uint32_t i = 0U; i < elementCount; ++i) {
            LwSciSyncAttrList attr;
            ASSERT_EQ(LwSciError_Success,
                LwSciStreamBlockElementWaiterAttrGet(producer, i, &attr));
            EXPECT_EQ((attr != nullptr),
                (!consSynchronousOnly && (i < conSyncCount)));
            LwSciSyncAttrListFree(attr);
        }
        ASSERT_EQ(LwSciError_Success,
            LwSciStreamBlockSetupStatusSet(
                producer, LwSciStreamSetup_WaiterAttrImport, true));
    };

    // Consumer receives producer's sync object requirement.
    inline void consRecvSyncAttr()
    {
        for (uint32_t n = 0U; n < numConsumers; n++) {
            LwSciStreamEventType event;
            ASSERT_EQ(LwSciError_Success,
                LwSciStreamBlockEventQuery(consumer[n], EVENT_QUERY_TIMEOUT, &event));
            EXPECT_EQ(LwSciStreamEventType_WaiterAttr, event);
            for (uint32_t i = 0U; i < elementCount; ++i) {
                LwSciSyncAttrList attr;
                ASSERT_EQ(LwSciError_Success,
                    LwSciStreamBlockElementWaiterAttrGet(
                        consumer[n], i, &attr));
                EXPECT_EQ((attr != nullptr),
                   (!prodSynchronousOnly && (i < prodSyncCount)));
                LwSciSyncAttrListFree(attr);
            }
            ASSERT_EQ(LwSciError_Success,
                LwSciStreamBlockSetupStatusSet(
                    consumer[n], LwSciStreamSetup_WaiterAttrImport, true));
        }
    };

    inline void prodSendSyncObj()
    {
        // Producer creates sync objects based on consumer's requirement and
        // sends the sync object to the consumer.
        for (uint32_t i = 0U; i < prodSyncCount; ++i) {
            getSyncObj(syncModule, prodSyncObjs[i]);
            EXPECT_EQ(((!consSynchronousOnly && (i < conSyncCount))
                       ? LwSciError_Success
                       : LwSciError_InconsistentData),
                LwSciStreamBlockElementSignalObjSet(
                    producer, i, prodSyncObjs[i]));
        }
        ASSERT_EQ(LwSciError_Success,
            LwSciStreamBlockSetupStatusSet(
                producer, LwSciStreamSetup_SignalObjExport, true));
    };

    inline void consSendSyncObj()
    {
        for (uint32_t n = 0U; n < numConsumers; ++n) {
            consSyncCount[n] = conSyncCount;
        }
        totalConsSync = conSyncCount * numConsumers;

        for (uint32_t n = 0U; n < numConsumers; n++) {
            // Consumer creates sync objects based on producer's requirement
            // and sends the sync object to the producer.
            for (uint32_t i = 0U; i < consSyncCount[n]; ++i) {
                getSyncObj(syncModule, consSyncObjs[n][i]);
                EXPECT_EQ(((!prodSynchronousOnly && (i < prodSyncCount))
                           ? LwSciError_Success
                           : LwSciError_InconsistentData),
                    LwSciStreamBlockElementSignalObjSet(
                        consumer[n], i, consSyncObjs[n][i]));
            }
            ASSERT_EQ(LwSciError_Success,
                LwSciStreamBlockSetupStatusSet(
                    consumer[n], LwSciStreamSetup_SignalObjExport, true));
        }
    };

    inline void prodRecvSyncObj()
    {
        LwSciStreamEventType event;

        // Producer receives consumer's sync objects.
        ASSERT_EQ(LwSciError_Success,
            LwSciStreamBlockEventQuery(producer, EVENT_QUERY_TIMEOUT, &event));
        EXPECT_EQ(LwSciStreamEventType_SignalObj, event);
        for (uint32_t n = 0U; n < numConsumers; ++n) {
            for (uint32_t i = 0U; i < elementCount; ++i) {
                LwSciSyncObj syncObj;
                ASSERT_EQ(LwSciError_Success,
                    LwSciStreamBlockElementSignalObjGet(
                        producer, n, i, &syncObj));
                EXPECT_EQ((syncObj != nullptr),
                          (i < consSyncCount[n]) && !prodSynchronousOnly);
                LwSciSyncObjFree(syncObj);
            }
        }
        ASSERT_EQ(LwSciError_Success,
            LwSciStreamBlockSetupStatusSet(
                producer, LwSciStreamSetup_SignalObjImport, true));
    };

    inline void consRecvSyncObj()
    {
        LwSciStreamEventType event;

        for (uint32_t n = 0U; n < numConsumers; n++) {
            // Consumer receives producer's sync count.
            ASSERT_EQ(LwSciError_Success,
                LwSciStreamBlockEventQuery(
                    consumer[n], EVENT_QUERY_TIMEOUT, &event));
            EXPECT_EQ(LwSciStreamEventType_SignalObj, event);
            for (uint32_t i = 0U; i < elementCount; ++i) {
                LwSciSyncObj syncObj;
                ASSERT_EQ(LwSciError_Success,
                    LwSciStreamBlockElementSignalObjGet(
                        consumer[n], 0U, i, &syncObj));
                EXPECT_EQ((syncObj != nullptr),
                          (i < prodSyncCount) && !consSynchronousOnly);
                LwSciSyncObjFree(syncObj);
            }
            ASSERT_EQ(LwSciError_Success,
                LwSciStreamBlockSetupStatusSet(
                    consumer[n], LwSciStreamSetup_SignalObjImport, true));
        }
    }
};

//==========================================================================
// Set up producer's sync object for a mailbox stream.
// * Consumer sends sync object requirement to the producer.
// * Producer creates a sync object based on consumer's requirement and
//   sends it to the consumer.
//==========================================================================
TEST_F(SyncObjSetup, CreateProducerSync_Mailbox)
{
    // Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    packetAttrSetup();

    queryMaxSyncCount();

    consSendSyncAttr();

    prodRecvSyncAttr();
    prodSendSyncObj();

    consRecvSyncObj();
}

//==========================================================================
// Set up producer's sync object for a mailbox stream. (multiple consumer)
//==========================================================================
TEST_F(SyncObjSetup, CreateProducerSync_Mailbox_Multicast)
{
    // Create a mailbox stream.
    createBlocks(QueueType::Mailbox, NUM_CONSUMERS);
    connectStream();

    packetAttrSetup();

    queryMaxSyncCount();

    consSendSyncAttr();

    prodRecvSyncAttr();
    prodSendSyncObj();

    consRecvSyncObj();
}

//==========================================================================
// Set up consumer's sync object for a mailbox stream.
// * Producer sends sync object requirement to the consumer.
// * Consumer creates a sync object based on producer's requirement and
//   sends it to the producer.
//==========================================================================
TEST_F(SyncObjSetup, CreateConsumerSync_Mailbox)
{
    // Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    packetAttrSetup();

    queryMaxSyncCount();

    prodSendSyncAttr();

    consRecvSyncAttr();
    consSendSyncObj();

    prodRecvSyncObj();
}

//==========================================================================
// Set up consumer's sync object for a mailbox stream. (multiple consumer)
//==========================================================================
TEST_F(SyncObjSetup, CreateConsumerSync_Mailbox_Multicast)
{
    // Create a mailbox stream.
    createBlocks(QueueType::Mailbox, NUM_CONSUMERS);
    connectStream();

    packetAttrSetup();

    queryMaxSyncCount();

    prodSendSyncAttr();

    consRecvSyncAttr();
    consSendSyncObj();

    prodRecvSyncObj();
}

//==========================================================================
// Set up producer's sync object for a fifo stream.
// * Consumer sends sync object requirement to the producer.
// * Producer creates a sync object based on consumer's requirement and
//   sends it to the consumer.
//==========================================================================
TEST_F(SyncObjSetup, CreateProducerSync_Fifo)
{
    // Create a fifo stream.
    createBlocks(QueueType::Fifo);
    connectStream();

    packetAttrSetup();

    queryMaxSyncCount();

    consSendSyncAttr();

    prodRecvSyncAttr();
    prodSendSyncObj();

    consRecvSyncObj();
}

//==========================================================================
// Set up producer's sync object for a fifo stream. (multiple consumers)
//==========================================================================
TEST_F(SyncObjSetup, CreateProducerSync_Fifo_Multicast)
{
    // Create a fifo stream.
    createBlocks(QueueType::Fifo, NUM_CONSUMERS);
    connectStream();

    packetAttrSetup();

    queryMaxSyncCount();

    consSendSyncAttr();

    prodRecvSyncAttr();
    prodSendSyncObj();

    consRecvSyncObj();
}

//==========================================================================
// Set up consumer's sync object for a fifo stream.
// * Producer sends sync object requirement to the consumer.
// * Consumer creates a sync object based on producer's requirement and
//   sends it to the producer.
//==========================================================================
TEST_F(SyncObjSetup, CreateConsumerSync_Fifo)
{
    // Create a fifo stream.
    createBlocks(QueueType::Fifo);
    connectStream();

    packetAttrSetup();

    queryMaxSyncCount();

    prodSendSyncAttr();

    consRecvSyncAttr();
    consSendSyncObj();

    prodRecvSyncObj();
}
//==========================================================================
// Set up consumer's sync object for a fifo stream. (multiple consumer)
//==========================================================================
TEST_F(SyncObjSetup, CreateConsumerSync_Fifo_Multicast)
{
    // Create a fifo stream.
    createBlocks(QueueType::Fifo, NUM_CONSUMERS);
    connectStream();

    packetAttrSetup();

    queryMaxSyncCount();

    prodSendSyncAttr();

    consRecvSyncAttr();
    consSendSyncObj();

    prodRecvSyncObj();
}

//==========================================================================
// Producer fails to send sync obj requirement before a complete downstream
// connection.
//==========================================================================
TEST_F(SyncObjSetup, SendProducerSyncAttr_Failure)
{
    // Create all blocks.
    createBlocks(QueueType::Mailbox);

    // Failure
    // TODO: return error may need to be updated. Same error code should
    // be used for incorrect event order.
    ASSERT_EQ(LwSciError_StreamNotConnected,
        LwSciStreamBlockElementWaiterAttrSet(producer, 0, nullptr));

    // Connect the stream.
    connectStream();

    packetAttrSetup();

    // Success
    prodSendSyncAttr();
    consRecvSyncAttr();
}

//==========================================================================
// Consumer fails to send sync obj requirement before a complete upstream
// connection.
//==========================================================================
TEST_F(SyncObjSetup, SendConsumerSyncAttr_Failure)
{
    // Create all blocks.
    createBlocks(QueueType::Mailbox);

    // Failure
    // TODO: return error may need to be updated. Same error code should
    // be used for incorrect event order.
    ASSERT_EQ(LwSciError_StreamNotConnected,
        LwSciStreamBlockElementWaiterAttrSet(consumer[0], 0, nullptr));

    // Connect the stream.
    connectStream();

    packetAttrSetup();

    // Success
    consSendSyncAttr();
    prodRecvSyncAttr();
}

//==========================================================================
// Producer fails to send sync obj before receiving consumer's sync object
// requirement.
//==========================================================================
TEST_F(SyncObjSetup, SendProducerSync_Failure)
{
    LwSciSyncObj prodSyncObj;
    getSyncObj(syncModule, prodSyncObj);

    // Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();
    packetAttrSetup();


    // Failure
    EXPECT_EQ(LwSciError_NotYetAvailable,
        LwSciStreamBlockElementSignalObjSet(producer, 0U, prodSyncObj));

    queryMaxSyncCount();

    consSendSyncAttr();

    prodRecvSyncAttr();

    // Success
    prodSendSyncObj();

    consRecvSyncObj();
    LwSciSyncObjFree(prodSyncObj);
}

//==========================================================================
// Consumer fails to send sync obj before receiving producer's sync object
// requirement.
//==========================================================================
TEST_F(SyncObjSetup, SendConsumerSync_Failure)
{
    LwSciSyncObj consSyncObj;
    getSyncObj(syncModule, consSyncObj);

    // Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();
    packetAttrSetup();


    // Failure
    EXPECT_EQ(LwSciError_NotYetAvailable,
        LwSciStreamBlockElementSignalObjSet(consumer[0], 0U, consSyncObj));

    queryMaxSyncCount();

    prodSendSyncAttr();

    consRecvSyncAttr();

    // Success
    consSendSyncObj();
    prodRecvSyncObj();
    LwSciSyncObjFree(consSyncObj);
}

TEST_F(SyncObjSetup, CreateSync_Fifo)
{
    createBlocks(QueueType::Fifo);
    connectStream();

    packetAttrSetup();

    createSync();
}

TEST_F(SyncObjSetup, CreateSync_Mailbox)
{
    createBlocks(QueueType::Mailbox);
    connectStream();

    packetAttrSetup();

    createSync();
}

TEST_F(SyncObjSetup, CreateSync_Fifo_Multicast)
{
    createBlocks(QueueType::Fifo, NUM_CONSUMERS);
    connectStream();

    packetAttrSetup();

    createSync();
}

TEST_F(SyncObjSetup, CreateSync_Mailbox_Multicast)
{
    createBlocks(QueueType::Mailbox, NUM_CONSUMERS);
    connectStream();

    packetAttrSetup();

    createSync();
}

//==========================================================================
// Ensure conlwrrency of LwSciStreamBlockElementSignalObjSet()
//==========================================================================
TEST_F(SyncObjSetup, ConlwrrentLwSciStreamBlockElementSignalObjSet)
{
    int consSuccessfullOps = 0;
    const int maxIterations = 10;
    std::vector<std::future<LwSciError>> connectTasks;
    conSyncCount = 1;

    // Create a mailbox stream.
    createBlocks(QueueType::Mailbox, 1);
    connectStream();
    packetAttrSetup();


    queryMaxSyncCount();
    prodSendSyncAttr();
    consRecvSyncAttr();

    getSyncObj(syncModule, consSyncObjs[0][0]);
    for (int i = 0; i < maxIterations; i++) {
        auto task = std::async(
            std::launch::async,
            [](LwSciStreamBlock const block,
                uint32_t const index,
                LwSciSyncObj const syncObj) -> LwSciError {
                return LwSciStreamBlockElementSignalObjSet(block, index, syncObj);
            },
            consumer[0], 0, consSyncObjs[0][0]);
        connectTasks.push_back(std::move(task));
    }

    for (auto& task : connectTasks) {
        LwSciError err = task.get();
        if (err == LwSciError_Success) {
            consSuccessfullOps++;
        }
    }

    ASSERT_EQ(maxIterations, consSuccessfullOps);
}
