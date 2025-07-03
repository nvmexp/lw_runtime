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

#ifndef INTERPROCESSPRODUCER_H
#define INTERPROCESSPRODUCER_H

#include <array>
#include <algorithm>

#include "gtest/gtest.h"
#include "interprocess.h"
#include "lwscistreamtest.h"

static const char *prodChannelsList[MAX_CHANNELS] = {"lwscistream_0",
                                                     "lwscistream_2",
                                                     "lwscistream_4",
                                                     "lwscistream_6"};

class InterProcessProducer :
    public InterProcess
{
public:
    InterProcessProducer(const char **chnames, uint32_t numChannels);
    ~InterProcessProducer(void);

    void createBlocks(uint32_t numPackets);
    void createStream(void);
    void disconnectStream(void);

private:
    LwSciStreamBlock producer = 0U;
    LwSciStreamBlock pool = 0U;
    LwSciStreamBlock multicast = 0U;
    LwSciStreamBlock ipcsrc[MAX_CHANNELS] { };
};


InterProcessProducer::InterProcessProducer(const char **chname, uint32_t numChannels = 1U)
{
    validateNumChannels(numChannels);
    channelCount = numChannels;

    for (uint32_t i = 0U; i < numChannels; ++i) {
        validateIpcChannel(chname[i], prodChannelsList);
        strcpy(ipc[i].chname, chname[i]);
    }
}

InterProcessProducer::~InterProcessProducer(void)
{
    // The blocks deletion happens in disconnectStream().
    // If some failure oclwrs then block deletion happens here
    if (producer != 0U) {
        LwSciStreamBlockDelete(producer);
        producer = 0U;
    }

    if (pool != 0U) {
        LwSciStreamBlockDelete(pool);
        pool = 0U;
    }

    if (multicast != 0U) {
        LwSciStreamBlockDelete(multicast);
        multicast = 0U;
    }

    for (uint32_t i = 0U; i < channelCount; ++i) {
        if (ipcsrc[i] != 0U) {
            LwSciStreamBlockDelete(ipcsrc[i]);
            ipcsrc[i] = 0U;
        }
    }
}

void InterProcessProducer::createBlocks(uint32_t numPackets = 1U)
{
    ASSERT_EQ(LwSciError_Success,
                    LwSciStreamStaticPoolCreate(numPackets, &pool));
    ASSERT_NE(0U, pool);

    ASSERT_EQ(LwSciError_Success,
                    LwSciStreamProducerCreate(pool, &producer));
    ASSERT_NE(0U, producer);

    if (channelCount > 1U) {
        ASSERT_EQ(LwSciError_Success,
                  LwSciStreamMulticastCreate(channelCount, &multicast));
        ASSERT_NE(0U, multicast);
    }

    for (uint32_t i = 0U; i < channelCount; ++i) {
        ASSERT_EQ(LwSciError_Success, LwSciStreamIpcSrcCreate(ipc[i].endpoint,
                                            syncModule, bufModule, &ipcsrc[i]));
        ASSERT_NE(0U, ipcsrc[i]);
    }
}

void InterProcessProducer::createStream(void)
{
    LwSciStreamEventType event;

    if (channelCount == 1U) {
        ASSERT_EQ(LwSciError_Success, LwSciStreamBlockConnect(producer,
                            ipcsrc[0])) << "Fail to connect producer and ipcsrc";
    } else {
        ASSERT_EQ(LwSciError_Success,
                  LwSciStreamBlockConnect(producer, multicast));

        for (uint32_t i = 0; i < channelCount; ++i) {
            ASSERT_EQ(LwSciError_Success,
                      LwSciStreamBlockConnect(multicast, ipcsrc[i]));
        }
    }

    // The ordering of events are non-determiistic so poll them all
    bool producerConnected = false;
    bool poolConnected = false;
    bool multicastConnected = false;
    std::array<bool, MAX_CHANNELS> ipcsrcConnected {};
    uint32_t ipcsrcEvents;
    LwSciError err = LwSciError_Success;

    // Poll events
    for (uint32_t i = 0U; i < MAX_WAIT_ITERS; ++i) {
        if (!producerConnected) {
            err = LwSciStreamBlockEventQuery(producer,
                                            EVENT_QUERY_TIMEOUT, &event);
            if (err != LwSciError_Timeout) {
                ASSERT_EQ(LwSciError_Success, err);
                producerConnected = (event ==
                                     LwSciStreamEventType_Connected);
            }
        }

        if (!poolConnected) {
            err = LwSciStreamBlockEventQuery(pool, EVENT_QUERY_TIMEOUT, &event);
            if (err != LwSciError_Timeout) {
                ASSERT_EQ(LwSciError_Success, err);
                poolConnected = (event == LwSciStreamEventType_Connected);
            }
        }

        if (channelCount > 1U && !multicastConnected) {
            err = LwSciStreamBlockEventQuery(multicast,
                                             EVENT_QUERY_TIMEOUT, &event);
            if (err != LwSciError_Timeout) {
                ASSERT_EQ(LwSciError_Success, err);
                multicastConnected = (event == LwSciStreamEventType_Connected);
            }
        } else {
            // only one channel, no functional multicast block in stream
            multicastConnected = true;
        }

        for (uint32_t k = 0U; k < channelCount; ++k) {
            if (!ipcsrcConnected[k]) {
                err = LwSciStreamBlockEventQuery(ipcsrc[k],
                                                 EVENT_QUERY_TIMEOUT, &event);
                if (err != LwSciError_Timeout) {
                    ASSERT_EQ(LwSciError_Success, err);
                    ipcsrcConnected[k] = (event == LwSciStreamEventType_Connected);
                }
            }
        }

        ipcsrcEvents = std::count_if(ipcsrcConnected.begin(),
                                     ipcsrcConnected.end(), isTrue);

        if (producerConnected && poolConnected && multicastConnected &&
            (ipcsrcEvents == channelCount)) {
            break;
        }

        usleep(100000);
    }

    ASSERT_TRUE(producerConnected)
        << "[LwSciStream] Producer failed to receive Connected event\n";
    ASSERT_TRUE(poolConnected)
        << "[LwSciStream] Pool failed to receive Connected event\n";

    if(channelCount > 1U) {
        ASSERT_TRUE(multicastConnected)
            << "[LwSciStream] multicast failed to receive Connected event\n";
    }

    ASSERT_TRUE(ipcsrcEvents == channelCount)
        << "[LwSciStream] IpcSrc(s) failed to receive Connected event\n";
}

void InterProcessProducer::disconnectStream(void)
{
    for (uint32_t i = 0U; i < channelCount; ++i) {
        if (ipcsrc[i] != 0U) {
            LwSciStreamBlockDelete(ipcsrc[i]);
            ipcsrc[i] = 0U;
        }
    }

    if (multicast != 0U) {
        LwSciStreamBlockDelete(multicast);
        multicast = 0U;
    }

    if (pool != 0U) {
        LwSciStreamBlockDelete(pool);
        pool = 0U;
    }

    if (producer != 0U) {
        LwSciStreamBlockDelete(producer);
        producer = 0U;
    }
}

#endif // !INTERPROCESSPRODUCER_H
