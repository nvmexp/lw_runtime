//! \file
//! \brief LwSciStream API testing.
//!
//! \copyright
//! Copyright (c) 2020-2021 LWPU Corporation. All rights reserved.
//!
//! LWPU Corporation and its licensors retain all intellectual property and
//! proprietary rights in and to this software, related documentation and any
//! modifications thereto. Any use, reproduction, disclosure or distribution
//! of this software and related documentation without an express license
//! agreement from LWPU Corporation is strictly prohibited.

#ifndef INTERPROCESSCONSUMER_H
#define INTERPROCESSCONSUMER_H

#include <array>
#include <algorithm>

#include "gtest/gtest.h"
#include "interprocess.h"
#include "lwscistreamtest.h"

static const char *consChannelsList[MAX_CHANNELS] = {"lwscistream_1",
                                                     "lwscistream_3",
                                                     "lwscistream_5",
                                                     "lwscistream_7"};

class InterProcessConsumer :
    public InterProcess
{
public:
    InterProcessConsumer(const char *chname);
    ~InterProcessConsumer(void);

    void createBlocks(QueueType type, uint32_t numConsumers);
    void createStream(void);
    void disconnectStream(void);

private:
    LwSciStreamBlock ipcdst = 0U;
    LwSciStreamBlock multicast = 0U;
    LwSciStreamBlock queue[MAX_CONSUMERS] { };
    LwSciStreamBlock consumer[MAX_CONSUMERS] { };

    uint32_t consumerCount = 1U;
    QueueType qType = QueueType::Mailbox;
};

InterProcessConsumer::InterProcessConsumer(const char *chname)
{
    validateIpcChannel(chname, consChannelsList);
    strcpy(ipc[0].chname, chname);
}

InterProcessConsumer::~InterProcessConsumer(void)
{
    // The blocks deletion happens in disconnectStream().
    // If some failure oclwrs, then block deletion happens here
    for (uint32_t i = 0U; i < consumerCount; ++i) {
        if (queue[i] != 0U) {
            LwSciStreamBlockDelete(queue[i]);
            queue[i] = 0U;
        }

        if (consumer[i] != 0) {
            LwSciStreamBlockDelete(consumer[i]);
            consumer[i] = 0U;
        }
    }

    if (multicast != 0U) {
        LwSciStreamBlockDelete(multicast);
        multicast = 0U;
    }

    if (ipcdst != 0U) {
        LwSciStreamBlockDelete(ipcdst);
        ipcdst = 0U;
    }
}

void InterProcessConsumer::createBlocks(QueueType type,
                                        uint32_t numConsumers = 1)
{
    ASSERT_LT(numConsumers, MAX_CONSUMERS)
        << "Number of consumers is more than the limit";
    consumerCount = numConsumers;

    ASSERT_EQ(LwSciError_Success, LwSciStreamIpcDstCreate(ipc[0].endpoint,
                                            syncModule, bufModule, &ipcdst));
    ASSERT_NE(0U, ipcdst);

    if (consumerCount > 1) {
        ASSERT_EQ(LwSciError_Success,
                  LwSciStreamMulticastCreate(consumerCount, &multicast));
        ASSERT_NE(0U, multicast);
    }

    for (uint32_t i = 0U; i < consumerCount; ++i) {
        switch(qType) {
            case QueueType::Mailbox:
                ASSERT_EQ(LwSciError_Success,
                          LwSciStreamMailboxQueueCreate(&queue[i]));
                break;
            case QueueType::Fifo:

                ASSERT_EQ(LwSciError_Success,
                          LwSciStreamFifoQueueCreate(&queue[i]));
                break;
            default:
                ASSERT_TRUE(false);
                break;
        }
        ASSERT_NE(0U, queue[i]);

        ASSERT_EQ(LwSciError_Success,
                  LwSciStreamConsumerCreate(queue[i], &consumer[i]));
        ASSERT_NE(0U, consumer[i]);
    }
}

void InterProcessConsumer::createStream(void)
{
    LwSciStreamEventType event;

    if (consumerCount == 1U) {
        ASSERT_EQ(LwSciError_Success,
                  LwSciStreamBlockConnect(ipcdst, consumer[0]));
    } else {
        ASSERT_EQ(LwSciError_Success,
                  LwSciStreamBlockConnect(ipcdst, multicast));

        for (uint32_t i = 0; i < consumerCount; ++i) {
            ASSERT_EQ(LwSciError_Success,
                      LwSciStreamBlockConnect(multicast, consumer[i]));
        }
    }

    std::array<bool, MAX_CONSUMERS> consumerConnected {};
    std::array<bool, MAX_CONSUMERS> queueConnected {};
    bool multicastConnected = false;
    bool ipcdstConnected = false;

    uint32_t consEvents;
    uint32_t queueEvents;
    LwSciError err = LwSciError_Success;

    for (uint32_t i = 0; i < MAX_WAIT_ITERS; ++i) {
        for (uint32_t k = 0U; k < consumerCount; ++k) {
            if (!consumerConnected[k]) {
                err = LwSciStreamBlockEventQuery(consumer[k],
                                                 EVENT_QUERY_TIMEOUT, &event);
                if (err != LwSciError_Timeout) {
                    ASSERT_EQ(LwSciError_Success, err);
                    consumerConnected[k] = (event
                                            == LwSciStreamEventType_Connected);
                }
            }
        }

        if (!ipcdstConnected) {
            err = LwSciStreamBlockEventQuery(ipcdst,
                                             EVENT_QUERY_TIMEOUT, &event);
            if (err != LwSciError_Timeout) {
                ASSERT_EQ(LwSciError_Success, err);
                ipcdstConnected = (event == LwSciStreamEventType_Connected);
            }
        }

        if (consumerCount > 1U && !multicastConnected) {
            err = LwSciStreamBlockEventQuery(multicast,
                                             EVENT_QUERY_TIMEOUT, &event);
            if (err != LwSciError_Timeout) {
                ASSERT_EQ(LwSciError_Success, err);
                multicastConnected = (event == LwSciStreamEventType_Connected);
            }
        } else {
            // only one consumer block, no functional multicast block in stream
            multicastConnected = true;
        }

        for (uint32_t k = 0U; k < consumerCount; k++) {
            if (!queueConnected[k]) {
                err = LwSciStreamBlockEventQuery(queue[k],
                                                 EVENT_QUERY_TIMEOUT, &event);
                if (err != LwSciError_Timeout) {
                    ASSERT_EQ(LwSciError_Success, err);
                    queueConnected[k] =
                        (event == LwSciStreamEventType_Connected);
                }
            }
        }

        consEvents = std::count_if(consumerConnected.begin(),
                                    consumerConnected.end(), isTrue);
        queueEvents = std::count_if(queueConnected.begin(),
                                    queueConnected.end(), isTrue);

        if (ipcdstConnected && multicastConnected &&
            (consEvents == consumerCount) &&
            (queueEvents == consumerCount)) {
            break;
        }

        usleep(100000);
    }

    ASSERT_TRUE(consEvents == consumerCount)
        << "[LwSciStream] consumer(s) failed to receive Connected event\n";
    ASSERT_TRUE(queueEvents == consumerCount)
        << "[LwSciStream] queue(s) failed to receive Connected event\n";
    ASSERT_TRUE(ipcdstConnected)
        << "[LwSciStream] ipcDst failed to receive Connected event\n";
    ASSERT_TRUE(multicastConnected)
        << "[LwSciStream] multicast failed to receive Connected event\n";
}

void InterProcessConsumer::disconnectStream(void)
{
    if (ipcdst != 0U) {
        LwSciStreamBlockDelete(ipcdst);
        ipcdst = 0U;
    }

    if (multicast != 0U) {
        LwSciStreamBlockDelete(multicast);
        multicast = 0U;
    }

    for (uint32_t i = 0U; i < consumerCount; ++i) {
        if (queue[i] != 0U) {
            LwSciStreamBlockDelete(queue[i]);
            queue[i] = 0U;
        }

        if (consumer[i] != 0) {
            LwSciStreamBlockDelete(consumer[i]);
            consumer[i] = 0U;
        }
    }
}

#endif // !INTERPROCESSCONSUMER_H
