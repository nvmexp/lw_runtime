//! \file
//! \brief LwSciStream APIs unit testing - Blocks Connection.
//!
//! \copyright
//! Copyright (c) 2019 LWPU Corporation. All rights reserved.
//!
//! LWPU Corporation and its licensors retain all intellectual property and
//! proprietary rights in and to this software, related documentation and any
//! modifications thereto. Any use, reproduction, disclosure or distribution
//! of this software and related documentation without an express license
//! agreement from LWPU Corporation is strictly prohibited.

#include "lwscistreamtest.h"
#include <future>

//==========================================================================
// Define BlocksConnect test suite.
//==========================================================================
class BlocksConnect :
    public LwSciStreamTest
{
};

//==========================================================================
// Connect producer and consumer blocks.
//==========================================================================
TEST_F(BlocksConnect, FifoStream)
{
    createBlocks(QueueType::Fifo);
    connectStream();
}

TEST_F(BlocksConnect, MailboxStream)
{
    createBlocks(QueueType::Mailbox);
    connectStream();
}

//==========================================================================
// Connect producer, multicast, and consumers
//==========================================================================
TEST_F(BlocksConnect, FifoMulticastStream)
{
    createBlocks(QueueType::Fifo, NUM_CONSUMERS);
    connectStream();
}

TEST_F(BlocksConnect, MailboxMulticastStream)
{
    createBlocks(QueueType::Mailbox, NUM_CONSUMERS);
    connectStream();
}

//==========================================================================
// Test conlwrrent block connection
//==========================================================================

TEST_F(BlocksConnect, ConlwrrentMulticastStream)
{
    const uint32_t allowedConnections = 4;
    uint32_t successfullConnections = 0;
    createBlocks(QueueType::Mailbox, allowedConnections);

    ASSERT_EQ(LwSciError_Success, LwSciStreamBlockConnect(producer, multicast));

    std::vector<std::future<LwSciError>> connectTasks;
    // Launch twice as much connection tasks than allowed
    for (int i = 0; i < allowedConnections * 2; i++) {
        auto task = std::async(
            std::launch::async,
            [](LwSciStreamBlock upstream,
               LwSciStreamBlock downstream) -> LwSciError {
                return LwSciStreamBlockConnect(upstream, downstream);
            },
            multicast, consumer[i % allowedConnections]);
        connectTasks.push_back(std::move(task));
    }

    for (auto& task : connectTasks) {
        LwSciError err = task.get();
        if (err == LwSciError_Success) {
            successfullConnections++;
        } else {
            // Make sure returned error indicates already connected state
            // SWUD is outdated, states that error should be
            ASSERT_EQ(err, LwSciError_InsufficientResource);
        }
    }

    ASSERT_EQ(allowedConnections, successfullConnections);
}

TEST_F(BlocksConnect, ConlwrrentStream)
{
    const uint32_t allowedConnections = 1;
    uint32_t successfullConnections = 0;
    createBlocks(QueueType::Mailbox, allowedConnections);

    std::vector<std::future<LwSciError>> connectTasks;
    // Launch twice as much connection tasks than allowed
    for (int i = 0; i < allowedConnections * 2; i++) {
        auto task = std::async(
            std::launch::async,
            [](LwSciStreamBlock upstream,
               LwSciStreamBlock downstream) -> LwSciError {
                return LwSciStreamBlockConnect(upstream, downstream);
            },
            producer, consumer[0]);
        connectTasks.push_back(std::move(task));
    }

    for (auto& task : connectTasks) {
        LwSciError err = task.get();
        if (err == LwSciError_Success) {
            successfullConnections++;
        } else {
            // Make sure returned error indicates already connected state
            // SWUD is outdated, states that error should be
            ASSERT_EQ(err, LwSciError_InsufficientResource);
        }
    }

    ASSERT_EQ(allowedConnections, successfullConnections);
}