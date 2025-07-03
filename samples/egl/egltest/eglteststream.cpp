/*
 * Copyright (c) 2016-2017, LWPU CORPORATION. All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "egltest.h"
#include "eglstream.h"
#include "tests.h"

typedef enum {
    ClientTypeStream1 = 1,
    ClientTypeStream2,
} ClientType;

LwError LwEglTestProducerReturnFrame(
    LwEglApiStreamInfo info,
    LwEglApiStreamFrame *frame)
{
    LOG_INFO("returnFrame producerImage %p consumerImage %p\n",
        frame->producerImage, frame->consumerImage);

    return LwError_Success;
}

LwError LwEglTestConsumerUpdateFrameAttr(
    LwEglApiContext     *context,
    LwEglApiStreamInfo   info,
    LwEglApiStreamFrame *acquireFrame,
    LwEglApiStreamFrame *releaseFrame,
    const LwEglApiStreamUpdateAttrs *attrs)
{
    if (acquireFrame) {
        LOG_INFO("updateFrameAttr acquireFrame: producerImage %p consumerImage %p\n",
            acquireFrame->producerImage, acquireFrame->consumerImage);
    }

    if (releaseFrame) {
        LOG_INFO("updateFrameAttr releaseFrame: producerImage %p consumerImage %p\n",
            releaseFrame->producerImage, releaseFrame->consumerImage);
    }

    return LwError_Success;
}

LwError LwEglTestConsumerRegisterBuffer(
    LwEglApiContext      *context,
    LwEglApiStreamInfo    info,
    LwEglApiStreamFrame  *newBuffer)
{
    LOG_INFO("registerBuffer producerImage %p consumerImage %p\n",
        newBuffer->producerImage, newBuffer->consumerImage);

    return LwError_Success;
}

LwError LwEglTestConsumerUnregisterBuffer(
    LwEglApiContext      *context,
    LwEglApiStreamInfo    info,
    LwEglApiStreamFrame  *oldBuffer)
{
    LOG_INFO("unregisterBuffer producerImage %p consumerImage %p\n",
        oldBuffer->producerImage, oldBuffer->consumerImage);

    return LwError_Success;
}

template<class CL>
static void LwEglTestClientInit(CL *client)
{
    client->init(gEglState.display, gEglState.eglStream);
}

template<class CL>
static bool LwEglTestClientCreate(CL *client)
{
    if (!client) {
        LOG_ERR("Invalid client.\n");
        return false;
    }

    if (!client->connect()) {
        LOG_ERR("Client connect failed.\n");
        return false;
    }

    if (!client->getCaps()) {
        return false;
    }

    client->create();

    return true;
}

static bool LwEglTestAllocStreamBuffers(ClientType type)
{
    EglTestStreamBuffer *buffers = NULL;
    EGLint fifoLength = 0, numBuffers = 0;
    int i;
    int maxAcquiredConsumerFrames = 0;

    if (!EGLStreamQuery(gEglState.display, gEglState.eglStream,
                        EGL_STREAM_FIFO_LENGTH_KHR, &fifoLength)) {
        goto failed;
    }

    // Initialize the buffers. Count needed to support worst case is
    //   1 +                       // 1 frame lwrrently being rendered
    //   max(1, fifoLength) +      // Full FIFO pending, or 1 for mailbox mode
    //   max(1, maxAcquiredFrames) // Limit of frames held by consumer
    if (type == ClientTypeStream1) {
        LwEglApiConsumerCaps consumerCaps;
        memset(&consumerCaps, 0, sizeof(LwEglApiConsumerCaps));
        if (!TestExpStream1ConsumerGetCaps(gEglState.display, gEglState.eglStream,
                                          &consumerCaps)) {
            LOG_ERR("streamConsumerGetCaps failed\n");
            return false;
        }
        maxAcquiredConsumerFrames = consumerCaps.maxAcquiredFrames;
    }

    numBuffers = 1 +
                 LW_MAX(1, fifoLength) +
                 LW_MAX(1, maxAcquiredConsumerFrames);
    LOG_INFO("EGL Stream Buffer Count: %d\n", numBuffers);

    buffers = (EglTestStreamBuffer*)malloc(numBuffers * sizeof(EglTestStreamBuffer));

    if (buffers == NULL) {
        goto failed;
    }

    for (i = 0; i< numBuffers; ++i) {
        buffers[i].sync = NULL;
        buffers[i].eglIndex = -1;
    }

    gEglState.streamBuffers = buffers;
    gEglState.streamBufferCount = numBuffers;
    // mark as available
    if (sem_post(&(gEglState.streamBufferReady)) != 0) {
        LOG_ERR("Stream Buffer Sem Post failed.\n");
        goto failed;
    }

    return true;

failed:
    if (buffers) {
        free(buffers);
    }
    gEglState.streamBufferCount = 0;
    gEglState.streamBuffers = NULL;

    LOG_ERR("StreamBuffers could not be allocated.\n");

    return false;
}

static bool LwEglTestInitStreamBuffers(void)
{
    // Init stream buffers
    gEglState.streamBufferCount = 0;
    gEglState.streamBuffers = NULL;
    if (sem_init(&gEglState.streamBufferReady, 0, 0) != 0) {
        LOG_ERR("Stream Buffer Sem Init failed.\n");
        return false;
    }

    return true;
}

static void LwEglTestFreeStreamBuffers(void)
{
    if (gEglState.streamBuffers != NULL) {
        free(gEglState.streamBuffers);
        gEglState.streamBufferCount = 0;
        gEglState.streamBuffers = NULL;
    }

    sem_destroy(&(gEglState.streamBufferReady));
}

static bool LwEglTestStreamSingleProcess(ClientType type, Producer *producer, Consumer *consumer)
{
    bool ret = false;

    // Init stream buffers
    if (!LwEglTestInitStreamBuffers()) {
        goto done;
    }

    // Init consumer + producer
    LwEglTestClientInit<Consumer>(consumer);
    LwEglTestClientInit<Producer>(producer);


    // Create connection
    if (!LwEglTestClientCreate<Consumer>(consumer)) {
        goto done;
    }

    if (!LwEglTestClientCreate<Producer>(producer)) {
        goto done;
    }

    if (!LwEglTestAllocStreamBuffers(type)) {
        goto done;
    }

    producer->complete();
    consumer->complete();

    if (!consumer->disconnect() || !producer->disconnect()) {
        goto done;
    }

    ret = true;

done:
    producer->join();
    consumer->join();

    ret &= (producer->getRunStatus() && consumer->getRunStatus());

    LwEglTestFreeStreamBuffers();

    EGLStreamFini();

    return ret;
}

template<class P>
static bool LwEglTestStreamProducerCrossP(ClientType type, P *producer)
{
    bool ret = false;

    // Init stream buffers
    if (!LwEglTestInitStreamBuffers()) {
        goto done;
    }

    // Init producer
    LwEglTestClientInit<P>(producer);

    if (!LwEglTestClientCreate<P>(producer)) {
        goto done;
    }

    if (!LwEglTestAllocStreamBuffers(type)) {
        goto done;
    }

    producer->complete();

    if (!producer->disconnect()) {
        goto done;
    }

    ret = true;

done:
    producer->join();

    LwEglTestFreeStreamBuffers();

    EGLStreamFini();

    return ret;
}

template<class C>
static bool LwEglTestStreamConsumerCrossP(C *consumer)
{
    bool ret = false;

    // Init consumer
    LwEglTestClientInit<C>(consumer);

    if (!LwEglTestClientCreate<C>(consumer)) {
        goto done;
    }

    consumer->complete();

    if (!consumer->disconnect()) {
        goto done;
    }

    ret = true;

done:
    consumer->join();

    EGLStreamFini();

    return ret;
}

static Producer *LwEglTestGetProducer(ClientType type, int testIndex)
{
    Producer *producer = NULL;

    if (type == ClientTypeStream1) {
        switch (testIndex) {
            case 1: // Test Producer1Stream1 and Consumer1Stream1
            case 3: // Test Producer1Stream1 and Consumer2Stream1
                producer = new Producer1Stream1;
                break;
            case 2: // Test Producer2Stream1 and Consumer1Stream1
            case 4: // Test Producer2Stream1 and Consumer2Stream1
                producer = new Producer2Stream1;
                break;
            case 5: // Test Producer3Stream1 and Consumer3Stream1
                producer = new Producer3Stream1;
                break;
            default:
                assert(0);
                break;
        }
    } else if (type == ClientTypeStream2) {
        switch (testIndex) {
            case 1: // Test Producer1 and Consumer1Stream2
            case 3: // Test Producer1Stream2 and Consumer2Stream2
                producer = new Producer1Stream2;
                break;
            case 2: // Test Producer2Stream2 and Consumer1Stream2
            case 4: // Test Producer2Stream2 and Consumer2Stream2
                producer = new Producer2Stream2;
                break;
            case 5: // Test Producer3Stream2 and Consumer3Stream2
                producer = new Producer3Stream2;
                break;
            default:
                assert(0);
                break;
        }
    } else {
        assert(0);
    }

    return producer;
}

static Consumer *LwEglTestGetConsumer(ClientType type, int testIndex)
{
    Consumer *consumer = NULL;

    if (type == ClientTypeStream1) {
        switch (testIndex) {
            case 1: // Test Producer1Stream1 and Consumer1Stream1
            case 2: // Test Producer2Stream1 and Consumer1Stream1
                consumer = new Consumer1Stream1;
                break;
            case 3: // Test Producer1Stream1 and Consumer2Stream1
            case 4: // Test Producer2Stream1 and Consumer2Stream1
                consumer = new Consumer2Stream1;
                break;
            case 5: // Test Producer3Stream1 and Consumer3Stream1
                consumer = new Consumer3Stream1;
                break;
            default:
                assert(0);
                break;
        }
    } else if (type == ClientTypeStream2) {
        switch (testIndex) {
            case 1: // Test Producer1Stream2 and Consumer1Stream2
            case 2: // Test Producer2Stream2 and Consumer1Stream2
                consumer = new Consumer1Stream2;
                break;
            case 3: // Test Producer1Stream2 and Consumer2Stream2
            case 4: // Test Producer2Stream2 and Consumer2Stream2
                consumer = new Consumer2Stream2;
                break;
            case 5: // Test Producer3Stream2 and Consumer3Stream2
                consumer = new Consumer3Stream2;
                break;
            default:
                assert(0);
                break;
        }
    }

    return consumer;
}

EGLBoolean LwEglTestEglStream()
{
    EGLBoolean ret = EGL_TRUE;
    EGLBoolean expected = EGL_TRUE;
    int testIndex, testEnd;
    const ClientType clType = (gTestArgs.testName != NULL && !strcmp(gTestArgs.testName, "stream2")) ? ClientTypeStream2 : ClientTypeStream1;
    const char* clLabel = (clType == ClientTypeStream2) ? "EGLStream2" : "EGLStream1";
    Producer *producer = NULL;
    Consumer *consumer = NULL;
    char processMode[20], streamMode[20];

    if (gTestArgs.testNo) {
        testIndex = gTestArgs.testNo;
        testEnd = testIndex + 1;
    } else {
        testIndex = 1;
        if (gTestArgs.processMode == SINGLE_PROCESS) {
            testEnd = MAX_EGLSTREAM_TESTS + 1;
        } else {
            testEnd = testIndex + 1;
        }
    }

    while (testIndex < testEnd)
    {
        strcpy(streamMode, (gTestArgs.fifoLength > 0) ? "fifo mode" : "mailbox mode");

        switch (gTestArgs.processMode)
        {
            case SINGLE_PROCESS:
            {
                strcpy(processMode, "single-process");
                if (!EGLStreamInitSingleProcess()) {
                    LOG_ERR("Failed EGLStreamInitSingleProcess.\n");
                    ret = EGL_FALSE;
                    break;
                }

                producer = LwEglTestGetProducer(clType, testIndex);
                consumer = LwEglTestGetConsumer(clType, testIndex);

                if (LwEglTestStreamSingleProcess(clType, producer, consumer) != expected) {
                    LOG_ERR("%s test %d (%s, %s) failed.\n",
                            clLabel, testIndex, processMode, streamMode);
                    ret = EGL_FALSE;
                } else {
                    LOG_MSG("%s test %d (%s, %s) passed.\n",
                            clLabel, testIndex, processMode, streamMode);
                }
                break;
            }
            case CROSS_PROCESS:
            case CROSS_PARTITION:
            {
                strcpy(processMode, (gTestArgs.processMode == CROSS_PROCESS) ?
                                        "cross-process" : "cross-partition");
                if (!EGLStreamInitCrossP()) {
                    LOG_ERR("%s test %d (%s, %s) failed.\n",
                            clLabel, testIndex, processMode, streamMode);
                    ret = EGL_FALSE;
                    break;
                }

                switch (gTestArgs.procType) {
                case PRODUCER:
                    producer = LwEglTestGetProducer(clType, testIndex);
                    if (!producer || !LwEglTestStreamProducerCrossP(clType, producer)) {
                        LOG_ERR("%s test %d (%s, producer, %s) failed.\n",
                                clLabel, testIndex, processMode, streamMode);
                        ret = EGL_FALSE;
                    } else {
                        LOG_MSG("%s test %d (%s, producer, %s) passed.\n",
                                clLabel, testIndex, processMode, streamMode);
                    }
                    break;
                case CONSUMER:
                    consumer = LwEglTestGetConsumer(clType, testIndex);
                    if (!consumer || !LwEglTestStreamConsumerCrossP(consumer)) {
                        LOG_ERR("%s test %d (%s, consumer, %s) failed.\n",
                                clLabel, testIndex, processMode, streamMode);
                        ret = EGL_FALSE;
                        break;
                    } else {
                        LOG_MSG("%s test %d (%s, consumer, %s) passed.\n",
                                clLabel, testIndex, processMode, streamMode);
                    }
                    break;
                default:
                    assert(0);
                    break;
                }
                break;
            }
            default:
                assert(0);
                break;
        }

        if (producer) {
            delete producer;
            producer = NULL;
        }
        if (consumer) {
            delete consumer;
            consumer = NULL;
        }
        testIndex++;
    }

    return ret;
}
