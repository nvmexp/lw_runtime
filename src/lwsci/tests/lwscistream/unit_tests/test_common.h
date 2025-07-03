//! \file
//! \brief LwSciStream unit testing.
//!
//! \copyright
//! Copyright (c) 2019-2021 LWPU Corporation. All rights reserved.
//!
//! LWPU Corporation and its licensors retain all intellectual property and
//! proprietary rights in and to this software, related documentation and any
//! modifications thereto. Any use, reproduction, disclosure or distribution
//! of this software and related documentation without an express license
//! agreement from LWPU Corporation is strictly prohibited.

#ifndef LWSCISTREAMTEST_H
#define LWSCISTREAMTEST_H

#include <unordered_map>
#include "common_includes.h"
#include "glob_test_vars.h"
#include "pool.h"
#include "producer.h"
#include "consumer.h"
#include "queue.h"
#include "limiter.h"
#include "ipcsrc.h"
#include "ipcdst.h"
#include "multicast.h"

using std::unordered_map;
using CookiePacketMap = unordered_map<LwSciStreamCookie, LwSciStreamPacket>;

constexpr int64_t EVENT_QUERY_TIMEOUT = 5000000;
constexpr uint32_t NUM_PACKETS = 5U;
constexpr uint32_t MAX_SYNC_COUNT = 4U;
constexpr uint32_t NUM_SYNCOBJS = 2U;
constexpr uint32_t MAX_ELEMENT_PER_PACKET = 4U;
constexpr uint32_t NUM_PACKET_ELEMENTS = 2U;
constexpr uint32_t NUM_FRAMES = 10U;
constexpr uint32_t MAX_CONSUMERS = 4U;
constexpr uint32_t NUM_CONSUMERS = 2U;
constexpr LwSciStreamCookie COOKIE_BASE = 100U;
constexpr uint32_t MAX_WAIT_ITERS = 100U;
constexpr uint32_t ILWALID_PACKET_HANDLE = 123456U;
constexpr uint32_t MAX_DST_CONNECTIONS = 4U;
constexpr uint32_t ALLOWED_MAX_ELEMENTS = 17U;
constexpr uint32_t MAX_LIMITER_COUNT = 4;

typedef enum {
    Mailbox,
    Fifo
} QueueType;

typedef enum {
    EventService,
    Internal
} SignalMode;

// LwSciBuf util functions

void makeRawBufferAttrList(
    LwSciBufModule bufModule,
    LwSciBufAttrList &attrList)
{
    LwSciBufType bufType = LwSciBufType_RawBuffer;
    uint64_t rawsize = (128 * 1024);
    uint64_t align = (4 * 1024);
    bool cpuaccess_flag = true;
    LwSciBufAttrValAccessPerm perm = LwSciBufAccessPerm_ReadWrite;

    LwSciBufAttrKeyValuePair rawbuffattrs[] = {
        { LwSciBufGeneralAttrKey_Types, &bufType, sizeof(bufType) },
        { LwSciBufRawBufferAttrKey_Size, &rawsize, sizeof(rawsize) },
        { LwSciBufRawBufferAttrKey_Align, &align, sizeof(align) },
        { LwSciBufGeneralAttrKey_NeedCpuAccess, &cpuaccess_flag,
            sizeof(cpuaccess_flag) },
        { LwSciBufGeneralAttrKey_RequiredPerm, &perm, sizeof(perm) }
    };

    ASSERT_EQ(LwSciError_Success,
        LwSciBufAttrListCreate(bufModule, &attrList));

    ASSERT_EQ(LwSciError_Success,
        LwSciBufAttrListSetAttrs(attrList, rawbuffattrs,
            sizeof(rawbuffattrs) / sizeof(LwSciBufAttrKeyValuePair)));
}

void makeRawBuffer(
    const LwSciBufAttrList& attrList,
    LwSciBufObj &bufObj)
{
    LwSciBufAttrList unreconciledAttrList[1] = { attrList };
    LwSciBufAttrList reconciledAttrList = nullptr;
    LwSciBufAttrList conflictAttrList = nullptr;

    ASSERT_EQ(LwSciError_Success,
        LwSciBufAttrListReconcile(
            unreconciledAttrList,
            1U,
            &reconciledAttrList,
            &conflictAttrList));

    ASSERT_EQ(LwSciError_Success,
        LwSciBufObjAlloc(reconciledAttrList, &bufObj));

    if (reconciledAttrList != nullptr) {
        LwSciBufAttrListFree(reconciledAttrList);
    }
    if (conflictAttrList != nullptr) {
        LwSciBufAttrListFree(conflictAttrList);
    }
}

// LwSciSync util functions

void cpuSignalerAttrList(
    LwSciSyncModule syncModule,
    LwSciSyncAttrList& list)
{
    EXPECT_EQ(LwSciError_Success,
        LwSciSyncAttrListCreate(syncModule, &list));

    LwSciSyncAttrKeyValuePair keyValue[2];
    bool cpuSignaler = true;
    keyValue[0].attrKey = LwSciSyncAttrKey_NeedCpuAccess;
    keyValue[0].value = (void*)&cpuSignaler;
    keyValue[0].len = sizeof(cpuSignaler);
    LwSciSyncAccessPerm cpuPerm = LwSciSyncAccessPerm_SignalOnly;
    keyValue[1].attrKey = LwSciSyncAttrKey_RequiredPerm;
    keyValue[1].value = (void*)&cpuPerm;
    keyValue[1].len = sizeof(cpuPerm);

    EXPECT_EQ(LwSciError_Success,
        LwSciSyncAttrListSetAttrs(list, keyValue, 2));
}

void cpuWaiterAttrList(
    LwSciSyncModule syncModule,
    LwSciSyncAttrList& list)
{
    EXPECT_EQ(LwSciError_Success,
        LwSciSyncAttrListCreate(syncModule, &list));

    LwSciSyncAttrKeyValuePair keyValue[2];
    bool cpuWaiter = true;
    keyValue[0].attrKey = LwSciSyncAttrKey_NeedCpuAccess;
    keyValue[0].value = (void*)&cpuWaiter;
    keyValue[0].len = sizeof(cpuWaiter);
    LwSciSyncAccessPerm cpuPerm = LwSciSyncAccessPerm_WaitOnly;
    keyValue[1].attrKey = LwSciSyncAttrKey_RequiredPerm;
    keyValue[1].value = (void*)&cpuPerm;
    keyValue[1].len = sizeof(cpuPerm);

    EXPECT_EQ(LwSciError_Success,
        LwSciSyncAttrListSetAttrs(list, keyValue, 2));
}

void getSyncObj(
    LwSciSyncModule syncModule,
    LwSciSyncObj& syncObj)
{
    LwSciSyncAttrList unreconciledList[2];
    LwSciSyncAttrList reconciledList;
    LwSciSyncAttrList conflictList;
    cpuSignalerAttrList(syncModule, unreconciledList[0]);
    cpuWaiterAttrList(syncModule, unreconciledList[1]);
    EXPECT_EQ(LwSciError_Success,
    LwSciSyncAttrListReconcile(unreconciledList, 2,
                               &reconciledList, &conflictList));

    EXPECT_EQ(LwSciError_Success,
              LwSciSyncObjAlloc(reconciledList, &syncObj));

    LwSciSyncAttrListFree(unreconciledList[0]);
    LwSciSyncAttrListFree(unreconciledList[1]);
    LwSciSyncAttrListFree(reconciledList);

}

//==========================================================================
// Define LwSciStream test suite.
//==========================================================================
class LwSciStreamTest :
    public ::testing::Test
{
protected:
    LwSciStreamBlock producer = 0U;
    LwSciStreamBlock pool = 0U;
    LwSciStreamBlock consumer[MAX_CONSUMERS] = {};
    LwSciStreamBlock queue[MAX_CONSUMERS] = {};
    LwSciStreamBlock multicast = 0U;
    LwSciStreamBlock limiter[MAX_CONSUMERS] = {};
    LwSciStreamBlock ipcsrc = 0U;
    LwSciStreamBlock ipcdst = 0U;
    std::shared_ptr<LwSciStream::Pool> poolPtr;
    std::shared_ptr<LwSciStream::Producer> producerPtr;
    std::shared_ptr<LwSciStream::Consumer> consumerPtr[MAX_CONSUMERS];
    std::shared_ptr<LwSciStream::Queue> queuePtr[MAX_CONSUMERS];
    std::shared_ptr<LwSciStream::Limiter> limiterPtr[MAX_CONSUMERS];
    std::shared_ptr<LwSciStream::IpcSrc> ipcsrcPtr;
    std::shared_ptr<LwSciStream::IpcDst> ipcdstPtr;
    std::shared_ptr<LwSciStream::MultiCast> multicastPtr;

    // Consumer count
    uint32_t numConsumers = 0U;

    // To enable the limiter
    bool limiterEnable = false;

    // To enable IPC streaming
    bool ipcStreamEnable = false;

    // LwSciSync member data
    uint32_t prodSyncCount = 0U;
    uint32_t consSyncCount[MAX_CONSUMERS] = {};
    uint32_t totalConsSync = 0U;

    LwSciSyncModule syncModule = nullptr;

    bool prodSynchronousOnly;
    LwSciSyncAttrList prodSyncAttrList;
    // All consumers use the same sync attributes.
    bool consSynchronousOnly;
    LwSciSyncAttrList consSyncAttrList;

    LwSciSyncObj prodSyncObjs[MAX_SYNC_COUNT];
    LwSciSyncObj consSyncObjs[MAX_CONSUMERS][MAX_SYNC_COUNT];

    // LwSciBuf member data
    uint32_t elementCount = 0U;
    LwSciBufModule bufModule = nullptr;
    LwSciBufAttrList rawBufAttrList = 0xABCDEF;

    // Packet member data
    uint32_t numPackets = 0U;

    CookiePacketMap poolCPMap;
    CookiePacketMap prodCPMap;
    CookiePacketMap consCPMap[MAX_CONSUMERS];

    // Wait time of LwSciStreamBlockEventQuery
    // The default value is zero, which is used
    // in single-thread test cases.
    int64_t eventQueryTimeout = 0L;

    // EventService object
    LwSciEventLoopService* eventLoopService = nullptr;

    LwSciStreamTest()
    {
        prodSyncAttrList = 0xABCDEF;
        consSyncAttrList = 0xABCDEF;

        for (uint32_t i = 0U; i < MAX_SYNC_COUNT; i++) {
            prodSyncObjs[i] = 0xABCDEF;
            for (uint32_t n = 0U; n < MAX_CONSUMERS; n++) {
                consSyncObjs[n][i] = 0xABCDEF;
            }
        }

        // Initialise all glob test vars to false
        LwSciStream::init_glob_test_vars();
    };

    virtual ~LwSciStreamTest()
    {
        if(ipcStreamEnable) {
            // The blocks getting destroyed before the IPC messages are read,
            // can result in missed message reads and failures in the test conditions.
            // Hence, adding a delay here to hold off the block delete.
            usleep(1000000);
        }

        if (producer != 0U) {
            LwSciStreamBlockDelete(producer);
        }

        if (pool != 0U) {
            LwSciStreamBlockDelete(pool);
        }

        if (multicast != 0U) {
            LwSciStreamBlockDelete(multicast);
        }

        if (ipcsrc != 0U) {
            LwSciStreamBlockDelete(ipcsrc);
        }

        if (ipcdst != 0U) {
            LwSciStreamBlockDelete(ipcdst);
        }

        for (uint32_t n = 0U; n < MAX_CONSUMERS; n++) {
            if (limiter[n] != 0U) {
                LwSciStreamBlockDelete(limiter[n]);
            }

            if (queue[n] != 0U) {
                LwSciStreamBlockDelete(queue[n]);
            }

            if (consumer[n] != 0U) {
                LwSciStreamBlockDelete(consumer[n]);
            }
        }

        // Free buffer resources
        if (rawBufAttrList != nullptr) {
            LwSciBufAttrListFree(rawBufAttrList);
        }

        // Free sync resources
        if (prodSyncAttrList != nullptr) {
            LwSciSyncAttrListFree(prodSyncAttrList);
        }
        if (consSyncAttrList != nullptr) {
            LwSciSyncAttrListFree(consSyncAttrList);
        }

        for (uint32_t i = 0U; i < MAX_SYNC_COUNT; i++) {
            if (prodSyncObjs[i] != nullptr) {
                LwSciSyncObjFree(prodSyncObjs[i]);
            }
            for (uint32_t n = 0U; n < MAX_CONSUMERS; n++) {
                if (consSyncObjs[n][i] != nullptr) {
                    LwSciSyncObjFree(consSyncObjs[n][i]);
                }
            }
        }

        if(ipcStreamEnable) {
            if (syncModule != nullptr) {
                LwSciSyncModuleClose(syncModule);
                syncModule = nullptr;
            }
            if (bufModule != nullptr) {
                LwSciBufModuleClose(bufModule);
                bufModule = nullptr;
            }

            deinitIpcChannel();
        }
    };

    void setLimiterEnable()
    {
        limiterEnable = true;
    }

    void initIpcChannel()
    {
        ipcStreamEnable = true;

        ASSERT_EQ(LwSciError_Success, LwSciBufModuleOpen(&bufModule));
        ASSERT_EQ(LwSciError_Success, LwSciSyncModuleOpen(&syncModule));

        ASSERT_EQ(LwSciError_Success, LwSciIpcInit()) << "LwSciIpcInit() Failed";

        strncpy(ipcSrc.chname, "InterProcessProducer", sizeof("InterProcessProducer"));
        ASSERT_EQ(LwSciError_Success, LwSciIpcOpenEndpoint(ipcSrc.chname, &ipcSrc.endpoint));

        strncpy(ipcDst.chname, "InterProcessConsumer", sizeof("InterProcessConsumer"));
        ASSERT_EQ(LwSciError_Success, LwSciIpcOpenEndpoint(ipcDst.chname, &ipcDst.endpoint));
    }

    void deinitIpcChannel()
    {
        if (ipcSrc.endpoint != 0U) {
            LwSciIpcCloseEndpoint(ipcSrc.endpoint);
            ipcSrc.endpoint = 0U;
        }
        if (ipcDst.endpoint != 0U) {
            LwSciIpcCloseEndpoint(ipcDst.endpoint);
            ipcDst.endpoint = 0U;
        }

        LwSciIpcDeinit();
    }

    inline void createBlocks(
        QueueType type,
        uint32_t numConsumersParam = 1U,
        uint32_t numPacketsParam = NUM_PACKETS,
        uint32_t numPacketsLimit = MAX_LIMITER_COUNT )
    {
        numConsumers = numConsumersParam;
        numPackets = numPacketsParam;

        // Create Pool block
        poolPtr = std::make_shared<LwSciStream::Pool>(numPackets);
        ASSERT_EQ(true, LwSciStream::Block::registerBlock(poolPtr));
        pool = poolPtr->getHandle();
        ASSERT_NE(0, pool);

        // Create Producer block
        producerPtr = std::make_shared<LwSciStream::Producer>();
        ASSERT_EQ(true, LwSciStream::Block::registerBlock(producerPtr));
        ASSERT_EQ(LwSciError_Success, producerPtr->BindPool(poolPtr));
        poolPtr->eventDefaultSetup();
        producer = producerPtr->getHandle();
        ASSERT_NE(0, producer);

        if (ipcStreamEnable) {
            // Create IpcSrc block
            ASSERT_EQ(limiterEnable , 0);
            ASSERT_EQ(LwSciError_Success,
                LwSciStreamIpcSrcCreate(ipcSrc.endpoint, syncModule, bufModule, &ipcsrc));
            ASSERT_NE(0, ipcsrc);

            // Create IpcDst block
            ASSERT_EQ(LwSciError_Success,
                LwSciStreamIpcDstCreate(ipcDst.endpoint, syncModule, bufModule, &ipcdst));
            ASSERT_NE(0, ipcdst);
        }

        for (uint32_t n = 0U; n < numConsumers; n++) {
            // Create Queue block(s)
            switch (type) {
            case QueueType::Mailbox:
                queuePtr[n] = std::make_shared<LwSciStream::Mailbox>();
                ASSERT_EQ(true, LwSciStream::Block::registerBlock(queuePtr[n]));
                queue[n] = queuePtr[n]->getHandle();
                break;
            case QueueType::Fifo:
                queuePtr[n] = std::make_shared<LwSciStream::Fifo>();
                ASSERT_EQ(true, LwSciStream::Block::registerBlock(queuePtr[n]));
                queue[n] = queuePtr[n]->getHandle();
                break;
            default:
                ASSERT_TRUE(false) << "Invalid queue type\n";
                break;
            }
            ASSERT_NE(0, queue[n]);

            // Create Consumer block(s)
            consumerPtr[n] = std::make_shared<LwSciStream::Consumer>();
            ASSERT_EQ(true, LwSciStream::Block::registerBlock(consumerPtr[n]));
            ASSERT_EQ(LwSciError_Success, consumerPtr[n]->BindQueue(queuePtr[n]));
            queuePtr[n]->eventDefaultSetup();
            consumer[n] = consumerPtr[n]->getHandle();
            ASSERT_NE(0, consumer[n]);

        }
        if(limiterEnable) {
            // Create limiter block
            for (uint32_t n = 0U; n < numConsumers; n++) {
                ASSERT_EQ(LwSciError_Success,
                LwSciStreamLimiterCreate(numPacketsLimit, &limiter[n]));
                ASSERT_NE(0, limiter[n]);
            }
        }
        if (numConsumers > 1U) {
            // Create Multicast block
            multicastPtr = std::make_shared<LwSciStream::MultiCast>(numConsumers);
            ASSERT_EQ(true, LwSciStream::Block::registerBlock(multicastPtr));
            multicast = multicastPtr->getHandle();
            ASSERT_NE(0, multicast);
        }
    };


    inline void connectStream()
    {
        // Connect blocks to create a complete stream.
        if (numConsumers == 1U) {
            if (ipcStreamEnable) {
                // Connect Upstream blocks
                ASSERT_EQ(LwSciError_Success,
                LwSciStreamBlockConnect(producer, ipcsrc));

                // Connect Downstream blocks
                ASSERT_EQ(LwSciError_Success,
                LwSciStreamBlockConnect(ipcdst, consumer[0]));
            } else {
                    if (limiterEnable){
                        ASSERT_EQ(LwSciError_Success,
                        LwSciStreamBlockConnect(producer, limiter[0]));
                        ASSERT_EQ(LwSciError_Success,
                        LwSciStreamBlockConnect(limiter[0], consumer[0]));
                    } else {
                        ASSERT_EQ(LwSciError_Success,
                        LwSciStreamBlockConnect(producer, consumer[0]));
                      }
              }
        } else {
                ASSERT_EQ(LwSciError_Success,
                LwSciStreamBlockConnect(producer, multicast));
                    for (uint32_t n = 0U; n < numConsumers; n++) {
                        if (limiterEnable) {
                            ASSERT_EQ(LwSciError_Success,
                            LwSciStreamBlockConnect(multicast, limiter[n]));
                            ASSERT_EQ(LwSciError_Success,
                            LwSciStreamBlockConnect(limiter[n], consumer[n]));
                        } else {
                            ASSERT_EQ(LwSciError_Success,
                            LwSciStreamBlockConnect(multicast, consumer[n]));
                          }
                    }

          }

        // Check Connect events
        LwSciStreamEventType event;

        if (ipcStreamEnable) {
            // The ordering of events are non-determiistic so poll them all
            bool producerConnected = false;
            bool ipcsrcConnected = false;
            bool poolConnected = false;
            bool ipcdstConnected = false;
            bool queueConnected = false;
            bool consumerConnected = false;

            LwSciError err = LwSciError_Success;

            // Poll events
            do {
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

                if (!ipcsrcConnected) {
                    err = LwSciStreamBlockEventQuery(ipcsrc,
                                                    EVENT_QUERY_TIMEOUT, &event);
                    if (err != LwSciError_Timeout) {
                        ASSERT_EQ(LwSciError_Success, err);
                        ipcsrcConnected = (event == LwSciStreamEventType_Connected);
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

                if (!queueConnected) {
                    err = LwSciStreamBlockEventQuery(queue[0],
                                                    EVENT_QUERY_TIMEOUT, &event);
                    if (err != LwSciError_Timeout) {
                        ASSERT_EQ(LwSciError_Success, err);
                        queueConnected =
                            (event == LwSciStreamEventType_Connected);
                    }
                }

                if (!consumerConnected) {
                    err = LwSciStreamBlockEventQuery(consumer[0],
                                                    EVENT_QUERY_TIMEOUT, &event);
                    if (err != LwSciError_Timeout) {
                        ASSERT_EQ(LwSciError_Success, err);
                        consumerConnected = (event
                                             == LwSciStreamEventType_Connected);
                    }
                }

                usleep(100000);

            } while (!(producerConnected && poolConnected && ipcsrcConnected &&
                     ipcdstConnected && queueConnected && consumerConnected));

            ASSERT_EQ(true, producerConnected);
            ASSERT_EQ(true, poolConnected);
            ASSERT_EQ(true, ipcsrcConnected);
            ASSERT_EQ(true, ipcdstConnected);
            ASSERT_EQ(true, queueConnected);
            ASSERT_EQ(true, consumerConnected);
        }
        else {
            EXPECT_EQ(LwSciError_Success,
                LwSciStreamBlockEventQuery(producer, EVENT_QUERY_TIMEOUT, &event));
            EXPECT_EQ(LwSciStreamEventType_Connected, event);

            EXPECT_EQ(LwSciError_Success,
                LwSciStreamBlockEventQuery(pool, EVENT_QUERY_TIMEOUT, &event));
            EXPECT_EQ(LwSciStreamEventType_Connected, event);

            for (uint32_t n = 0U; n < numConsumers; n++) {
                EXPECT_EQ(LwSciError_Success,
                    LwSciStreamBlockEventQuery(consumer[n], EVENT_QUERY_TIMEOUT, &event));
                EXPECT_EQ(LwSciStreamEventType_Connected, event);

                EXPECT_EQ(LwSciError_Success,
                    LwSciStreamBlockEventQuery(queue[n], EVENT_QUERY_TIMEOUT, &event));
                EXPECT_EQ(LwSciStreamEventType_Connected, event);
            }

            if (numConsumers > 1U) {
                EXPECT_EQ(LwSciError_Success,
                    LwSciStreamBlockEventQuery(multicast, EVENT_QUERY_TIMEOUT, &event));
                EXPECT_EQ(LwSciStreamEventType_Connected, event);

                if (limiterEnable) {
                    for (uint32_t n = 0U; n < numConsumers; n++) {
                        EXPECT_EQ(LwSciError_Success,
                            LwSciStreamBlockEventQuery(limiter[n], EVENT_QUERY_TIMEOUT, &event));
                        EXPECT_EQ(LwSciStreamEventType_Connected, event);
                    }
                }
            }
        }
        
        // Make sure producer and pool see correct number of consumers
        uint32_t queryVal;
        EXPECT_EQ(LwSciError_Success,
                  LwSciStreamBlockConsumerCountGet(producer, &queryVal));
        EXPECT_EQ(queryVal, numConsumers);
        EXPECT_EQ(LwSciError_Success,
                  LwSciStreamBlockConsumerCountGet(pool, &queryVal));
        EXPECT_EQ(queryVal, numConsumers);
    };

    inline virtual void createSync(uint32_t prodSyncCountParam = NUM_SYNCOBJS,
                                   uint32_t consSyncCountParam = NUM_SYNCOBJS)
    {
        LwSciStreamEventType event;

        ASSERT_TRUE(prodSyncCountParam <= MAX_SYNC_COUNT);
        ASSERT_TRUE(consSyncCountParam <= MAX_SYNC_COUNT);

        // Clamp number of sync objects to number of elements.
        //   Passing in a sync count less than the element count would allow
        //   testing of a case where some elements were asynchronous and
        //   some were synchronous.
        if (prodSyncCountParam > elementCount) {
            prodSyncCountParam = elementCount;
        }
        if (consSyncCountParam > elementCount) {
            consSyncCountParam = elementCount;
        }

        // Initialize prodSyncCount and consSyncCount
        //   If synchronous only flag is set for either, the count is zero
        uint32_t perConsSync = 0U;
        if (!prodSynchronousOnly) {
            prodSyncCount = prodSyncCountParam;
        }
        if (!consSynchronousOnly) {
            for (uint32_t n = 0U; n < numConsumers; n++) {
                consSyncCount[n] = consSyncCountParam;
            }
            perConsSync = consSyncCountParam;
            totalConsSync = consSyncCountParam * numConsumers;
        }

        // Producer sends its sync waiter requirements to the consumer.
        for (uint32_t i = 0; i < prodSyncCount; i++) {
            ASSERT_EQ(LwSciError_Success,
                LwSciStreamBlockElementWaiterAttrSet(
                    producer, i, prodSyncAttrList));
        }
        ASSERT_EQ(LwSciError_Success,
            LwSciStreamBlockSetupStatusSet(
                producer, LwSciStreamSetup_WaiterAttrExport, true));

        // Consumer sends its sync object requirement to the producer
        for (uint32_t n = 0U; n < numConsumers; n++) {
            for (uint32_t i = 0; i < consSyncCount[n]; i++) {
                ASSERT_EQ(LwSciError_Success,
                    LwSciStreamBlockElementWaiterAttrSet(
                        consumer[n], i, consSyncAttrList));
            }
            ASSERT_EQ(LwSciError_Success,
                LwSciStreamBlockSetupStatusSet(
                    consumer[n], LwSciStreamSetup_WaiterAttrExport, true));
        }

        // Query max number of sync objects
        int32_t maxNumSyncObjs = 0;
        ASSERT_EQ(LwSciError_Success, LwSciStreamAttributeQuery(
            LwSciStreamQueryableAttrib_MaxSyncObj, &maxNumSyncObjs));

        // Producer receives consumer's sync object requirement
        EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
            producer, eventQueryTimeout, &event));
        EXPECT_EQ(LwSciStreamEventType_WaiterAttr, event);
        for (uint32_t i = 0; i < elementCount; i++) {
            LwSciSyncAttrList eventSyncAttrList;
            ASSERT_EQ(LwSciError_Success,
                LwSciStreamBlockElementWaiterAttrGet(
                    producer, i, &eventSyncAttrList));
            if (i < perConsSync) {
                ASSERT_NE(nullptr, eventSyncAttrList);
                LwSciSyncAttrListFree(eventSyncAttrList);
            } else {
                ASSERT_EQ(nullptr, eventSyncAttrList);
            }
        }
        ASSERT_EQ(LwSciError_Success,
            LwSciStreamBlockSetupStatusSet(
                producer, LwSciStreamSetup_WaiterAttrImport, true));

        // Producer creates sync objects based on consumer's requirement and
        // sends the sync object to the consumer.
        // TODO: This doesn't take into account whether the consumer is
        //   synchronous
        for (uint32_t i = 0; i < prodSyncCount; i++) {
            getSyncObj(syncModule, prodSyncObjs[i]);
            ASSERT_EQ(LwSciError_Success,
                LwSciStreamBlockElementSignalObjSet(
                    producer, i, prodSyncObjs[i]));
        }
        ASSERT_EQ(LwSciError_Success,
            LwSciStreamBlockSetupStatusSet(
                producer, LwSciStreamSetup_SignalObjExport, true));

        for (uint32_t n = 0U; n < numConsumers; ++n) {

            // Consumer receives producer's sync object requirement
            EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                consumer[n], eventQueryTimeout, &event));
            EXPECT_EQ(LwSciStreamEventType_WaiterAttr, event);
            for (uint32_t i = 0; i < elementCount; i++) {
                LwSciSyncAttrList eventSyncAttrList;
                ASSERT_EQ(LwSciError_Success,
                    LwSciStreamBlockElementWaiterAttrGet(
                        consumer[n], i, &eventSyncAttrList));
                if (i < prodSyncCount) {
                    ASSERT_NE(nullptr, eventSyncAttrList);
                    LwSciSyncAttrListFree(eventSyncAttrList);
                } else {
                    ASSERT_EQ(nullptr, eventSyncAttrList);
                }
            }
            ASSERT_EQ(LwSciError_Success,
                LwSciStreamBlockSetupStatusSet(
                    consumer[n], LwSciStreamSetup_WaiterAttrImport, true));

            // Consumer creates sync objects based on producer's requirement
            //   and sends the sync object to the producer.
            // TODO: This doesn't take into account whether the producer
            //   is synchronous for the element
            for (uint32_t i = 0U; i <  consSyncCount[n]; i++) {
                getSyncObj(syncModule, consSyncObjs[n][i]);
                ASSERT_EQ(LwSciError_Success,
                    LwSciStreamBlockElementSignalObjSet(
                        consumer[n], i, consSyncObjs[n][i]));
            }
            ASSERT_EQ(LwSciError_Success,
                LwSciStreamBlockSetupStatusSet(
                    consumer[n], LwSciStreamSetup_SignalObjExport, true));
        }

        // Producer receives consumer's sync objects
        ASSERT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
            producer, eventQueryTimeout, &event));
        ASSERT_EQ(LwSciStreamEventType_SignalObj, event);
        ASSERT_EQ(LwSciError_Success,
            LwSciStreamBlockSetupStatusSet(
                producer, LwSciStreamSetup_SignalObjImport, true));

        for (uint32_t n = 0U; n < numConsumers; n++) {
            // Consumer receives producer's sync objects
            ASSERT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                consumer[n], eventQueryTimeout, &event));
            ASSERT_EQ(LwSciStreamEventType_SignalObj, event);
            ASSERT_EQ(LwSciError_Success,
                LwSciStreamBlockSetupStatusSet(
                    consumer[n], LwSciStreamSetup_SignalObjImport, true));
        }
    };

    inline virtual void packetAttrSetup(uint32_t numElements = NUM_PACKET_ELEMENTS)
    {
        LwSciStreamEventType event;
        uint32_t receivedCount;

        ASSERT_TRUE(numElements <= MAX_ELEMENT_PER_PACKET);
        elementCount = numElements;

        // Query maximum num of packet elements allowed
        int32_t value;
        ASSERT_EQ(LwSciError_Success, LwSciStreamAttributeQuery(
            LwSciStreamQueryableAttrib_MaxElements, &value));
        ASSERT_TRUE(elementCount <= value);

        // Set producer packet requirements
        uint32_t producerElementType[MAX_ELEMENT_PER_PACKET];
        LwSciBufAttrList producerElementAttr[MAX_ELEMENT_PER_PACKET];
        for (uint32_t i = 0U; i < elementCount; i++) {
            // Use index so that type is unique for each element.
            producerElementType[i] = i;
            producerElementAttr[i] = rawBufAttrList;
            ASSERT_EQ(LwSciError_Success,
                      LwSciStreamBlockElementAttrSet(producer, i,
                                                     rawBufAttrList));
        }

        // Indicate producer element specification is done
        ASSERT_EQ(LwSciError_Success,
                  LwSciStreamBlockSetupStatusSet(
                      producer, LwSciStreamSetup_ElementExport, true));

        // Set consumer packet requirements
        uint32_t consumerElementType[MAX_ELEMENT_PER_PACKET];
        LwSciBufAttrList consumerElementAttr[MAX_ELEMENT_PER_PACKET];
        for (uint32_t i = 0U; i < elementCount; i++) {
            // Use index so that type is unique for each element.
            consumerElementType[i] = i;
            consumerElementAttr[i] = rawBufAttrList;
        }
        for (uint32_t n = 0U; n < numConsumers; n++) {
            for (uint32_t i = 0U; i < elementCount; i++) {
                ASSERT_EQ(LwSciError_Success,
                          LwSciStreamBlockElementAttrSet(consumer[n], i,
                                                         rawBufAttrList));
            }

            // Indicate consumer element specification is done
            ASSERT_EQ(LwSciError_Success,
                      LwSciStreamBlockSetupStatusSet(
                          consumer[n], LwSciStreamSetup_ElementExport, true));
        }

        // Pool receives Elements event
        EXPECT_EQ(LwSciError_Success,
                  LwSciStreamBlockEventQuery(pool, eventQueryTimeout, &event));
        EXPECT_EQ(LwSciStreamEventType_Elements, event);

        // Check element count from producer
        EXPECT_EQ(LwSciError_Success,
                  LwSciStreamBlockElementCountGet(
                      pool, LwSciStreamBlockType_Producer, &receivedCount));
        EXPECT_EQ(elementCount, receivedCount);

        // Check elements from producer
        for (uint32_t i = 0; i < elementCount; i++) {
            uint32_t elemType;
            EXPECT_EQ(LwSciError_Success,
                      LwSciStreamBlockElementAttrGet(
                          pool, LwSciStreamBlockType_Producer, i,
                          &elemType, nullptr));
            EXPECT_EQ(producerElementType[i], elemType);
            // TODO: Check attrs
        }

        // Check element count from consumer
        EXPECT_EQ(LwSciError_Success,
                  LwSciStreamBlockElementCountGet(
                      pool, LwSciStreamBlockType_Consumer, &receivedCount));
        EXPECT_EQ(elementCount, receivedCount);

        // Check elements from consumer
        for (uint32_t i = 0; i < elementCount; i++) {
            uint32_t elemType;
            EXPECT_EQ(LwSciError_Success,
                      LwSciStreamBlockElementAttrGet(
                          pool, LwSciStreamBlockType_Consumer, i,
                          &elemType, nullptr));
            EXPECT_EQ(consumerElementType[i], elemType);
            // TODO: Check attrs
        }

        // Indicate pool element import is done
        ASSERT_EQ(LwSciError_Success,
                  LwSciStreamBlockSetupStatusSet(
                      pool, LwSciStreamSetup_ElementImport, true));

        // Pool record packet attr
        for (uint32_t i = 0U; i < elementCount; i++) {
            // Use index so that type is unique for each element.
            ASSERT_EQ(LwSciError_Success,
                      LwSciStreamBlockElementAttrSet(pool, i, rawBufAttrList));
        }

        // Indicate pool element specification is done
        ASSERT_EQ(LwSciError_Success,
                  LwSciStreamBlockSetupStatusSet(
                      pool, LwSciStreamSetup_ElementExport, true));

        // Producer receives element specification from pool
        EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
            producer, eventQueryTimeout, &event));
        EXPECT_EQ(LwSciStreamEventType_Elements, event);

        // Check element count at producer
        EXPECT_EQ(LwSciError_Success,
                  LwSciStreamBlockElementCountGet(
                      producer, LwSciStreamBlockType_Pool, &receivedCount));
        EXPECT_LE(elementCount, receivedCount);

        // Check elements at producer
        for (uint32_t i = 0; i < receivedCount; i++) {
            uint32_t elemType;
            EXPECT_EQ(LwSciError_Success,
                      LwSciStreamBlockElementAttrGet(
                          producer, LwSciStreamBlockType_Pool, i,
                          &elemType, nullptr));
            EXPECT_EQ(i, elemType);
            // TODO: Check attrs
        }

        // Indicate producer element import is done
        ASSERT_EQ(LwSciError_Success,
                  LwSciStreamBlockSetupStatusSet(
                      producer, LwSciStreamSetup_ElementImport, true));

        // Check all consumers
        for (uint32_t n = 0U; n < numConsumers; n++) {
            // Consumer receives element specification from pool
            EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                consumer[n], eventQueryTimeout, &event));
            EXPECT_EQ(LwSciStreamEventType_Elements, event);

            // Check element count at consumer
            EXPECT_EQ(LwSciError_Success,
                      LwSciStreamBlockElementCountGet(
                          consumer[n], LwSciStreamBlockType_Pool,
                          &receivedCount));
            EXPECT_LE(elementCount, receivedCount);

            // Check elements at consumer and indicate used
            for (uint32_t i = 0; i < receivedCount; i++) {
                uint32_t elemType;
                EXPECT_EQ(LwSciError_Success,
                          LwSciStreamBlockElementAttrGet(
                              consumer[n], LwSciStreamBlockType_Pool, i,
                              &elemType, nullptr));
                EXPECT_EQ(i, elemType);
                // TODO: Check attrs
                EXPECT_EQ(LwSciError_Success,
                          LwSciStreamBlockElementUsageSet(
                              consumer[n], i, true));
            }

            // Indicate consumer element import is done
            ASSERT_EQ(LwSciError_Success,
                      LwSciStreamBlockSetupStatusSet(
                          consumer[n], LwSciStreamSetup_ElementImport, true));
        }
    };

    inline virtual void createPacket()
    {
        LwSciStreamEventType event;

        for (uint32_t i = 0U; i < numPackets; ++i) {
            // Choose pool's cookie and for new packet
            LwSciStreamPacket packetHandle;
            LwSciStreamCookie poolCookie
                = static_cast<LwSciStreamCookie>(i + COOKIE_BASE);
            ASSERT_EQ(LwSciError_Success,
                LwSciStreamPoolPacketCreate(pool, poolCookie, &packetHandle));

            // Save the cookie-to-handle mapping
            poolCPMap.emplace(poolCookie, packetHandle);

            // Register buffer to packet handle
            LwSciBufObj poolElementBuf[MAX_ELEMENT_PER_PACKET];
            for (uint32_t k = 0; k < elementCount; ++k) {
                makeRawBuffer(rawBufAttrList, poolElementBuf[k]);
                ASSERT_EQ(LwSciError_Success,
                    LwSciStreamPoolPacketInsertBuffer(pool,
                                                      packetHandle, k,
                                                      poolElementBuf[k]));
            }

            // Indicate packet complete
            ASSERT_EQ(LwSciError_Success,
                LwSciStreamPoolPacketComplete(pool, packetHandle));

            // Producer receives PacketCreate event
            EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                producer, eventQueryTimeout, &event));
            EXPECT_EQ(LwSciStreamEventType_PacketCreate, event);

            // Retrieve packet handle
            LwSciStreamPacket producerPacket;
            EXPECT_EQ(LwSciError_Success,
                LwSciStreamBlockPacketNewHandleGet(producer,
                                                   &producerPacket));

            // Assign cookie to producer packet handle
            LwSciStreamCookie producerCookie
                = static_cast<LwSciStreamCookie>(i + COOKIE_BASE);
            LwSciError producerError = LwSciError_Success;

            // Producer accepts a packet provided by the pool
            ASSERT_EQ(LwSciError_Success, LwSciStreamBlockPacketStatusSet(
                producer, producerPacket, producerCookie, producerError));

            // Save the cookie-to-handle mapping
            prodCPMap.emplace(producerCookie, producerPacket);

            // Receive the packet at the consumers
            for (uint32_t n = 0U; n < numConsumers; n++) {

                // Consumer receives PacketCreate event
                EXPECT_EQ(LwSciError_Success,
                          LwSciStreamBlockEventQuery(consumer[n],
                                                     eventQueryTimeout,
                                                     &event));
                EXPECT_EQ(LwSciStreamEventType_PacketCreate, event);

                // Retrieve packet handle
                LwSciStreamPacket consumerPacket;
                EXPECT_EQ(LwSciError_Success,
                    LwSciStreamBlockPacketNewHandleGet(consumer[n],
                                                       &consumerPacket));

                // Assign cookie to consumer packet handle
                LwSciStreamCookie consumerCookie
                    = static_cast<LwSciStreamCookie>(i + COOKIE_BASE);
                LwSciError consumerError = LwSciError_Success;

                // Consumer accepts packet provided by the pool
                ASSERT_EQ(LwSciError_Success,
                          LwSciStreamBlockPacketStatusSet(consumer[n],
                                                          consumerPacket,
                                                          consumerCookie,
                                                          consumerError));

                // Save the cookie-to-handle mapping
                consCPMap[n].emplace(consumerCookie, consumerPacket);
            }

            // Free the buffer objs in the pool
            for (uint32_t k = 0; k < elementCount; ++k) {
                LwSciBufObjFree(poolElementBuf[k]);
            }
        }
    };

    inline virtual void checkPacketStatus()
    {
        LwSciStreamEventType event;
        uint32_t numPacketComplete = 0U;

        // Pool must receive all PacketStatus events
        while (numPacketComplete < numPackets) {

            if (LwSciError_Success !=
                LwSciStreamBlockEventQuery(pool, eventQueryTimeout, &event)) {
                ASSERT_TRUE(false);
                break;
            }
            ASSERT_EQ(LwSciStreamEventType_PacketStatus, event);

            numPacketComplete++;
        }

        // Pool indicates packet list completed
        ASSERT_EQ(LwSciError_Success,
            LwSciStreamBlockSetupStatusSet(
               pool, LwSciStreamSetup_PacketExport, true));
        ASSERT_EQ(LwSciError_Success,
            LwSciStreamBlockSetupStatusSet(
               pool, LwSciStreamSetup_PacketImport, true));

        // Producer receives PacketsComplete event
        ASSERT_EQ(LwSciError_Success,
            LwSciStreamBlockEventQuery(producer, eventQueryTimeout, &event));
        ASSERT_EQ(LwSciStreamEventType_PacketsComplete, event);

        // Producer indicates packet setup completed
        ASSERT_EQ(LwSciError_Success,
            LwSciStreamBlockSetupStatusSet(
               producer, LwSciStreamSetup_PacketImport, true));

        // Complete setup on consumers
        for (uint32_t n = 0U; n < numConsumers; n++) {

            // Consumer receives PacketsComplete event
            ASSERT_EQ(LwSciError_Success,
                LwSciStreamBlockEventQuery(consumer[n], eventQueryTimeout,
                                           &event));
            ASSERT_EQ(LwSciStreamEventType_PacketsComplete, event);

            // Consumer indicates packet setup completed
            ASSERT_EQ(LwSciError_Success,
                LwSciStreamBlockSetupStatusSet(
                    consumer[n], LwSciStreamSetup_PacketImport, true));
        }
    };

    inline void setupComplete()
    {
        LwSciStreamEventType event;
        EXPECT_EQ(LwSciError_Success,
                  LwSciStreamBlockEventQuery(producer,
                                             eventQueryTimeout,
                                             &event));
        EXPECT_EQ(LwSciStreamEventType_SetupComplete, event);
        EXPECT_EQ(LwSciError_Success,
                  LwSciStreamBlockEventQuery(pool,
                                             eventQueryTimeout,
                                             &event));
        EXPECT_EQ(LwSciStreamEventType_SetupComplete, event);
        if (1U < numConsumers) {
            EXPECT_EQ(LwSciError_Success,
                      LwSciStreamBlockEventQuery(multicast,
                                                 eventQueryTimeout,
                                                 &event));
            EXPECT_EQ(LwSciStreamEventType_SetupComplete, event);
        }
        for (uint32_t n = 0U; n < numConsumers; n++) {
            EXPECT_EQ(LwSciError_Success,
                      LwSciStreamBlockEventQuery(queue[n],
                                                 eventQueryTimeout,
                                                 &event));
            EXPECT_EQ(LwSciStreamEventType_SetupComplete, event);
            EXPECT_EQ(LwSciError_Success,
                      LwSciStreamBlockEventQuery(consumer[n],
                                                 eventQueryTimeout,
                                                 &event));
            EXPECT_EQ(LwSciStreamEventType_SetupComplete, event);
        }
    };

    inline virtual void streaming(uint32_t numFrames = NUM_FRAMES)
    {
        LwSciStreamEventType event;
        LwSciSyncFence fence;

        for (uint32_t i = 0; i < numFrames; ++i) {
            LwSciStreamCookie cookie;

            // Pool sends packet ready event to producer
            EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                producer, eventQueryTimeout, &event));
            EXPECT_EQ(LwSciStreamEventType_PacketReady, event);

            // Producer get a packet from the pool
            ASSERT_EQ(LwSciError_Success,
                LwSciStreamProducerPacketGet(producer, &cookie));
            LwSciStreamPacket handle = prodCPMap[cookie];
            for (uint32_t n = 0U; n < numConsumers; n++) {
                for (uint32_t j = 0U; j < consSyncCount[n]; j++) {
                    ASSERT_EQ(LwSciError_Success,
                        LwSciStreamBlockPacketFenceGet(
                            producer, handle, n, j, &fence));
                    LwSciSyncFenceClear(&fence);
                }
            }

            // Producer inserts a data packet into the stream
            fence = LwSciSyncFenceInitializer;
            for (uint32_t j = 0U; j < prodSyncCount; j++) {
                ASSERT_EQ(LwSciError_Success,
                    LwSciStreamBlockPacketFenceSet(
                        producer, handle, j, &fence));
            }
            ASSERT_EQ(LwSciError_Success,
                LwSciStreamProducerPacketPresent(producer, handle));

            for (uint32_t n = 0U; n < numConsumers; n++) {
                // Pool sends packet ready event to consumer
                EXPECT_EQ(LwSciError_Success,
                    LwSciStreamBlockEventQuery(
                        consumer[n], eventQueryTimeout, &event));
                EXPECT_EQ(LwSciStreamEventType_PacketReady, event);

                ASSERT_EQ(LwSciError_Success,
                    LwSciStreamConsumerPacketAcquire(consumer[n], &cookie));
                handle = (consCPMap[n])[cookie];
                for (uint32_t j = 0U; j < prodSyncCount; j++) {
                    ASSERT_EQ(LwSciError_Success,
                        LwSciStreamBlockPacketFenceGet(
                            consumer[n], handle, 0U, j, &fence));
                    LwSciSyncFenceClear(&fence);
                }

                // Consumer returns a data packet to the stream
                fence = LwSciSyncFenceInitializer;
                for (uint32_t j = 0U; j < consSyncCount[n]; j++) {
                    ASSERT_EQ(LwSciError_Success,
                        LwSciStreamBlockPacketFenceSet(
                            consumer[n], handle, j, &fence));
                }
                ASSERT_EQ(LwSciError_Success,
                    LwSciStreamConsumerPacketRelease(consumer[n], handle));
            }
        } // End of sending frames
    };

    inline virtual void disconnectStream()
    {
        LwSciStreamEventType event;

        // Delete producer block
        ASSERT_EQ(LwSciError_Success, LwSciStreamBlockDelete(producer));
        producer = 0U;

        // Pool receives both DisconnectUpstream and DisconnectDownstream events.
        EXPECT_EQ(LwSciError_Success,
            LwSciStreamBlockEventQuery(pool, eventQueryTimeout, &event));
        EXPECT_EQ(LwSciStreamEventType_Disconnected, event);

        // Delete pool block
        ASSERT_EQ(LwSciError_Success, LwSciStreamBlockDelete(pool));
        pool = 0U;

        if (numConsumers > 1U) {
            // Multicast receives DisconnectUpstream and DisconnectDownstream events.
            EXPECT_EQ(LwSciError_Success,
                LwSciStreamBlockEventQuery(multicast, eventQueryTimeout, &event));
            EXPECT_EQ(LwSciStreamEventType_Disconnected, event);

            // Delete queue block
            ASSERT_EQ(LwSciError_Success, LwSciStreamBlockDelete(multicast));
            multicast = 0U;
        }

        for (uint32_t n = 0U; n < numConsumers; n++) {
            // Queue receives DisconnectUpstream and DisconnectDownstream events.
            EXPECT_EQ(LwSciError_Success,
                LwSciStreamBlockEventQuery(queue[n], eventQueryTimeout, &event));
            EXPECT_EQ(LwSciStreamEventType_Disconnected, event);

            // Delete queue block
            ASSERT_EQ(LwSciError_Success, LwSciStreamBlockDelete(queue[n]));
            queue[n] = 0U;

            // Consumer receives DisconnectUpstream event
            EXPECT_EQ(LwSciError_Success,
                LwSciStreamBlockEventQuery(consumer[n], eventQueryTimeout, &event));
            EXPECT_EQ(LwSciStreamEventType_Disconnected, event);

            // Delete consumer block
            ASSERT_EQ(LwSciError_Success, LwSciStreamBlockDelete(consumer[n]));
            consumer[n] = 0U;
        }
    };
};

#endif // !LWSCISTREAMTEST_H
