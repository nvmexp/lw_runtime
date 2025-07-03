//! \file
//! \brief LwSciStream APIs unit testing - Blocks Creation.
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

//==========================================================================
// Define BlocksCreate test suite.
//==========================================================================
class BlocksCreate :
    public LwSciStreamTest
{
protected:
    virtual void SetUp()
    {
        ASSERT_EQ(LwSciError_Success,
            LwSciEventLoopServiceCreate(1U, &eventLoopService));
    };


    inline void SetupEventService(SignalMode signalMode, LwSciStreamBlock block)
    {
        LwSciStreamEventType event;
        LwSciEventNotifier* eventNotifier;

        if (signalMode == SignalMode::EventService) {
            ASSERT_EQ(LwSciError_Success,
                LwSciStreamBlockEventServiceSetup(block,
                    &eventLoopService->EventService, &eventNotifier));
        }

        ASSERT_EQ(LwSciError_Timeout,
            LwSciStreamBlockEventQuery(block, 0, &event));

        switch (signalMode) {
        case SignalMode::EventService:
            // Non-zero timeout value causes BadParameter error.
            ASSERT_EQ(LwSciError_BadParameter,
            LwSciStreamBlockEventQuery(block, EVENT_QUERY_TIMEOUT, &event));
            eventNotifier->Delete(eventNotifier);
            break;
        case SignalMode::Internal: // Fallthrough
        default:
            ASSERT_EQ(LwSciError_Timeout,
            LwSciStreamBlockEventQuery(block, EVENT_QUERY_TIMEOUT, &event));
            break;
        }
    };

    inline void CreatePool(SignalMode signalMode) {
        ASSERT_EQ(LwSciError_Success,
            LwSciStreamStaticPoolCreate(NUM_PACKETS, &pool));
        ASSERT_NE(0, pool);

        SetupEventService(signalMode, pool);
    }

    inline void CreateQueue(QueueType type, SignalMode signalMode) {
        switch (type) {
        case QueueType::Mailbox:
            ASSERT_EQ(LwSciError_Success,
                LwSciStreamMailboxQueueCreate(&queue[0]));
            break;
        case QueueType::Fifo:
            ASSERT_EQ(LwSciError_Success,
                LwSciStreamFifoQueueCreate(&queue[0]));
            break;
        default:
            ASSERT_TRUE(false) << "Invalid queue type\n";
            break;
        }
        ASSERT_NE(0, queue[0]);

        SetupEventService(signalMode, queue[0]);
    }

    inline void CreateMulticast(SignalMode signalMode) {
        ASSERT_EQ(LwSciError_Success,
            LwSciStreamMulticastCreate(NUM_CONSUMERS, &multicast));
        ASSERT_NE(0, multicast);

        SetupEventService(signalMode, multicast);
    }

    inline void CreateProducer(SignalMode signalMode) {
        LwSciStreamEventType event;
        LwSciEventNotifier* eventNotifier;
        ASSERT_EQ(LwSciError_Success,
            LwSciStreamStaticPoolCreate(NUM_PACKETS, &pool));
        ASSERT_NE(0, pool);

        ASSERT_EQ(LwSciError_Success,
            LwSciStreamProducerCreate(pool, &producer));
        ASSERT_NE(0, producer);

        SetupEventService(signalMode, producer);
    }

    inline void CreateConsumer(QueueType type, SignalMode signalMode)
    {
        switch (type) {
        case QueueType::Mailbox:
            ASSERT_EQ(LwSciError_Success,
                LwSciStreamMailboxQueueCreate(&queue[0]));
            break;
        case QueueType::Fifo:
            ASSERT_EQ(LwSciError_Success,
                LwSciStreamFifoQueueCreate(&queue[0]));
            break;
        default:
            ASSERT_TRUE(false) << "Invalid queue type\n";
            break;
        }
        ASSERT_NE(0, queue[0]);

        ASSERT_EQ(LwSciError_Success,
            LwSciStreamConsumerCreate(queue[0], &consumer[0]));
        ASSERT_NE(0, consumer[0]);

        SetupEventService(signalMode, consumer[0]);
    };

    inline void CreatePresentSync(SignalMode signalMode) {
        ASSERT_EQ(LwSciError_Success, LwSciSyncModuleOpen(&syncModule));

        ASSERT_EQ(LwSciError_Success,
            LwSciStreamPresentSyncCreate(syncModule, &presentsync));
        ASSERT_NE(0, presentsync);

        SetupEventService(signalMode, presentsync);
    };

    inline void CreateReturnSync(SignalMode signalMode) {

        ASSERT_EQ(LwSciError_Success, LwSciSyncModuleOpen(&syncModule));

        ASSERT_EQ(LwSciError_Success,
            LwSciStreamReturnSyncCreate(syncModule, &returnsync));
        ASSERT_NE(0, returnsync);

        SetupEventService(signalMode, returnsync);
    };
};

//==========================================================================
// Create blocks.
//==========================================================================

TEST_F(BlocksCreate, CreateProducerWithPool)
{
    CreateProducer(SignalMode::Internal);
}

TEST_F(BlocksCreate, CreateConsumerWithFifoQ)
{
    CreateConsumer(QueueType::Fifo, SignalMode::Internal);
}

TEST_F(BlocksCreate, CreateConsumerWithMailboxQ)
{
    CreateConsumer(QueueType::Mailbox, SignalMode::Internal);
}

TEST_F(BlocksCreate, CreatePool)
{
    CreatePool(SignalMode::Internal);
}

TEST_F(BlocksCreate, CreateMailboxQ)
{
    CreateQueue(QueueType::Mailbox, SignalMode::Internal);
}

TEST_F(BlocksCreate, CreateFifoQ)
{
    CreateQueue(QueueType::Fifo, SignalMode::Internal);
}

TEST_F(BlocksCreate, CreateMulticast)
{
    CreateMulticast(SignalMode::Internal);
}

TEST_F(BlocksCreate, CreatePresentSync)
{
    CreatePresentSync(SignalMode::Internal);
}

TEST_F(BlocksCreate, CreateReturnSync)
{
    CreateReturnSync(SignalMode::Internal);
}
//==========================================================================
// Create blocks with event service setup
//==========================================================================

TEST_F(BlocksCreate, CreateProducerWithPoolAndEventSetup)
{
    CreateProducer(SignalMode::EventService);
}

TEST_F(BlocksCreate, CreateConsumerWithFifoQAndEventSetup)
{
    CreateConsumer(QueueType::Fifo, SignalMode::EventService);
}

TEST_F(BlocksCreate, CreateConsumerWithMailboxQAndEventSetup)
{
    CreateConsumer(QueueType::Mailbox, SignalMode::EventService);
}

TEST_F(BlocksCreate, CreatePoolAndEventSetup)
{
    CreatePool(SignalMode::EventService);
}

TEST_F(BlocksCreate, CreateMailboxQAndEventSetup)
{
    CreateQueue(QueueType::Mailbox, SignalMode::EventService);
}

TEST_F(BlocksCreate, CreateFifoQAndEventSetup)
{
    CreateQueue(QueueType::Fifo, SignalMode::EventService);
}

TEST_F(BlocksCreate, CreateMulticastAndEventSetup)
{
    CreateMulticast(SignalMode::EventService);
}

TEST_F(BlocksCreate, CreatePresentSyncAndEventSetup)
{
    CreatePresentSync(SignalMode::EventService);
}

TEST_F(BlocksCreate, CreateReturnSyncAndEventSetup)
{
    CreateReturnSync(SignalMode::EventService);
}

//==========================================================================
// Ensure only maximum number of outputs per Multicast block are allowed.
//==========================================================================
TEST_F(BlocksCreate, QueryMaxMulticastOutputs)
{
    int32_t value;

    // Query maximum number of outputs allowed per Multicast block
    ASSERT_EQ(LwSciError_Success, LwSciStreamAttributeQuery(
        LwSciStreamQueryableAttrib_MaxMulticastOutputs, &value));

    // Create as Multicast block with invalid number of outputs
    ASSERT_EQ(LwSciError_BadParameter,
        LwSciStreamMulticastCreate(value + 1U, &multicast));

    // Create as Multicast block with valid number of outputs
    ASSERT_EQ(LwSciError_Success,
        LwSciStreamMulticastCreate(value, &multicast));
}
