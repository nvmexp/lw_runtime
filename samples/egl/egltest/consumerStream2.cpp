/*
 * Copyright (c) 2016-2017, LWPU CORPORATION. All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 */

#include "consumer.h"

static void* runConsumerStream2(void *arg)
{
    ConsumerStream2 *consumer = (ConsumerStream2*) arg;

    LOG_INFO(("Running consumer.\n"));

    bool status = consumer->run();
    if (!status) {
        LOG_ERR("Consumer: Run failed.\n");
    }

    consumer->setRunStatus(status);
    return NULL;
}

void ConsumerStream2::init(EGLDisplay dsp, EGLStreamKHR stream)
{
    Consumer::init(dsp, stream);

    // set consumer info.
    memset(&caps, 0, sizeof(caps));
    caps.image.useGlsi = 1;
    caps.sync.types = LwCommonSyncTypeBit_Reg | LwCommonSyncTypeBit_Sem;
    caps.sync.useLwRm = 1;

    // timeouts
    stateWaitTimeout = 3000;
    foreverTimeout = -1;

    // # buffers init: assumes mailbox
    this->numBuffers = 3;

    // Reserve
    pvtEglStream = NULL;
    if (!reserve()) {
        LOG_ERR("Consumer: reserve failed.\n");
    }
}

bool ConsumerStream2::reserve(void)
{
    LwU64 streamState = 0;
    pvtEglStream = NULL;

    if (!TestExpStream2ConsumerReserve(display, eglStream, &pvtEglStream)) {
        LOG_ERR("Consumer: streamConsumerReserve failed.\n");
        return false;
    }

    if (!TestExpStream2StatePoll(pvtEglStream, &streamState)) {
        LOG_ERR("Consumer: stream state poll failed.\n");
        return false;
    }

    // consumer state should be reserved
    if ((streamState & LW_EGL_API_STREAM2_CONS_RSVD) == 0) {
        LOG_ERR("Consumer: Stream consumer reserved bit not set.\n");
        return false;
    }

    return true;
}

bool ConsumerStream2::connect(void)
{
    LwU64 streamState;

    if (!TestExpStream2ConsumerConnect(pvtEglStream, &caps)) {
        LOG_ERR("Consumer: streamConsumerConnect failed.\n");
        return false;
    }

    // consumer state should be connected
    if (!TestExpStream2StatePoll(pvtEglStream, &streamState)) {
        LOG_ERR("Consumer: stream state poll for connection failed.\n");
        return false;
    }

    if ((streamState & LW_EGL_API_STREAM2_CONS_CONN) == 0) {
        LOG_ERR("Consumer: stream state consumer connected bit not set.\n");
        return false;
    }

    LOG_INFO("Consumer: Consumer connected successfully!.\n");

    return true;
}

bool ConsumerStream2::getCaps(void)
{
    memset(&caps, 0, sizeof(LwEglApiStream2ConsumerCaps));

    if (!TestExpStream2ConsumerGetCaps(pvtEglStream, &caps)) {
        LOG_ERR("Consumer: streamConsumerGetCaps failed.\n");
        return false;
    }

    return true;
}

bool ConsumerStream2::producerCapsMatch(void)
{
    LwU64 streamState = 0;
    LwU64 stateToMatch = LW_EGL_API_STREAM2_PROD_CONN;

    // Fetch consumer caps
    if (!getCaps()) {
        return false;
    }

    // Make sure producer has connected
    if (!TestExpStream2StateWait(pvtEglStream, foreverTimeout, stateToMatch, &streamState)) {
        LOG_ERR("Consumer: StateWatch on connection waiting for producer connection .\n");
        return false;
    }

    // Fetch producer caps
    memset(&prodCaps, 0, sizeof(LwEglApiStream2ProducerCaps));
    if(!TestExpStream2ProducerGetCaps(pvtEglStream, &prodCaps)) {
        LOG_ERR("Consumer: Failed to fetch producer caps.\n");
        return false;
    }

    // Verify caps match
    if(caps.image.useGlsi != prodCaps.image.useGlsi ||
       caps.sync.useLwRm != prodCaps.sync.useLwRm) {
            LOG_ERR("Consumer: Failed to match producer capabilities.\n");
            return false;
    }

    return true;
}

bool ConsumerStream2::getConstAttribs(void)
{
    LwU64 streamState = 0;
    LwU64 stateToMatch = LW_EGL_API_STREAM2_INIT_ATTR;

    // Constant attributes
    if (!TestExpStream2StateWait(pvtEglStream, stateWaitTimeout, stateToMatch, &streamState)) {
        LOG_ERR("Consumer: StateWatch on constant attributes failed.\n");
        return false;
    }

    if (streamState & stateToMatch) {
        // Get Fifo length
        LwEglApiStream2ConstantAttr attr = {0};
        if (TestExpStream2GetConstAttr(pvtEglStream, &attr)) {
            // update buffer count
            numBuffers = LW_MAX(1, attr.fifoSize) + 2;
            LOG_INFO("Consumer: Num. Buffers: %d\n", numBuffers);
        } else {
            LOG_ERR("Consumer: Could not get constant attributes.\n");
            return false;
        }
    }

    return true;
}

bool ConsumerStream2::getElwironment(void)
{
    LwU64 streamState = 0;
    LwU64 stateToMatch = LW_EGL_API_STREAM2_INIT_ELW;

    // Environment
    if (!TestExpStream2StateWait(pvtEglStream, stateWaitTimeout, LW_EGL_API_STREAM2_INIT_ELW, &streamState)) {
        LOG_ERR("Consumer: StateWatch on environment failed.\n");
        return false;
    }

    if ((streamState & LW_EGL_API_STREAM2_INIT_ELW) == 0) {
        LOG_ERR("Consumer: Stream environment not available.\n");
        return false;
    }

    // TODO: examine elw. attributes
    // validate GPU / HV, crossProcess is true / false
    return true;
}

void ConsumerStream2::create(void)
{
    pthread_create(&thread, NULL, runConsumerStream2, this);
}

bool ConsumerStream2::registerBuffer(LwEglApiClientBuffer* clientBuffer)
{
    // Obtain buffer
    LwEglApiBufferHandle image;
    if (!TestExpStream2ConsumerBufferObtain(pvtEglStream, &image)) {
        LOG_ERR("Consumer: ConsumerBufferObtain failed!\n");
        return false;
    }

    // TODO: Grab references to underlying image surfaces
    // use these saved handles to verify image contents during frame cycle
    // LwGlsiEglImageHandle glsiImage = LwGlsiEglImage::FromHandle(image)->unref();
    // LwGlsiEglImage::FromHandle(image)->unref();

    // Notify producer of buffer registration
    if(!TestExpStream2ConsumerBufferNotify(pvtEglStream,
                                           image,
                                           (LwError) 0,
                                           *clientBuffer)) {
        LOG_ERR("Consumer: Buffer Notify failed!\n");
        return false;
    }

    return true;
}

bool ConsumerStream2::verifyFrame(LwEglApiStream2Frame* acquiredFrame)
{
    // TODO: Read contents of acquire frame image and validate
    // int bufIdx = ((EglTestStreamBuffer *)(acquiredFrame.buffer))->eglIndex;
    // use this index to fetch underlying image surface
    return true;
}

bool ConsumerStream2::processFrame(int acquireTimeout)
{
    // acquire
    LwEglApiStream2Frame acquiredFrame = {0};
    if(!TestExpStream2ConsumerFrameAcquire(pvtEglStream, acquireTimeout, &acquiredFrame)) {
        LOG_ERR("Consumer: frame acquire failed!!\n");
        return false;
    }

    // get and verify processed frame number
    if (!EGLStreamQuery(display, eglStream, EGL_CONSUMER_FRAME_KHR, &frameNumber)) {
        LOG_ERR("Consumer: EGL_CONSUMER_FRAME_KHR query failed\n");
        return false;
    }
    assert (frameNumber == (int) acquiredFrame.number);

    // verify frame content
    if(!verifyFrame(&acquiredFrame)) {
        LOG_ERR("Consumer: verifyFrame failed\n");
        return false;
    }

    // verify metadata
    if(!queryMetadata(&acquiredFrame)) {
        LOG_ERR("Consumer: queryMetadata failed\n");
        return false;
    }

    LOG_INFO("Consumer: acquired and processed frameNumber: %d\n", frameNumber);

    // release buffer associated with recently acquired frame
    LwEglApiStream2Frame releaseFrame = {0};
    releaseFrame.buffer = acquiredFrame.buffer;
    LW_EGL_API_FRAME_SYNC_INIT(releaseFrame.lwrmSync.sync);
    if (!TestExpStream2ConsumerFrameRelease(pvtEglStream, &releaseFrame)) {
        LOG_ERR("Consumer frame release failed!!\n");
        return false;
    }

    LOG_INFO("Consumer: released frameNumber: %d\n", frameNumber);

    return true;
}

bool ConsumerStream2::run(void)
{
    EglTestStreamBuffer* buffers = NULL;
    LwU64 streamState = 0;
    LwU64 stateToMatch = 0;
    bool ret = false;
    int acquireTimeout;
    const struct timespec reg = {0, 1000};
    int regIdx = 0;

    // const attributes
    if (!getConstAttribs()) {
        goto done;
    }

    // validate environment
    if (!getElwironment()) {
        goto done;
    }

    // Wait for connection to succeed
    stateToMatch = LW_EGL_API_STREAM2_INIT_COMM | LW_EGL_API_STREAM2_CONS_CONN;
    if (!TestExpStream2StateWait(pvtEglStream, foreverTimeout, stateToMatch, &streamState)) {
        LOG_ERR("Consumer: StateWatch on connection wait .\n");
        goto done;
    }

    if (!(streamState & stateToMatch)) {
        LOG_ERR("Consumer: Connection failed!!.\n");
        goto done;
    }

    LOG_INFO("Consumer: Connection ready!!!\n");

    if (!producerCapsMatch()) {
        LOG_ERR("Consumer: Could not find matching producer capabilities.\n");
        goto done;
    }

    // get acquire timeout
    if (!EGLStreamQuery(display, eglStream, EGL_CONSUMER_ACQUIRE_TIMEOUT_USEC_KHR, &acquireTimeout)) {
        acquireTimeout = 16000;
    }
    LOG_INFO("Consumer: acquire timeout: %d usec.\n", acquireTimeout);

    // client buffers
    buffers = (EglTestStreamBuffer*)malloc(numBuffers * sizeof(EglTestStreamBuffer));
    if (buffers == NULL) {
        LOG_ERR("StreamBuffer allocaion failed.\n");
        goto done;
    }

    // Register all available buffers
    regIdx = 0;
    stateToMatch = LW_EGL_API_STREAM2_REG_NEW;
    while (regIdx < numBuffers) {
        // wait on producer buffer registration
        // TODO: Change forever wait to a reasonable amount and handle failure
        if (!TestExpStream2StateWait(pvtEglStream, foreverTimeout, stateToMatch, &streamState)) {
            LOG_ERR("Consumer: StateWatch on new registration available failed!, index: %d\n", regIdx);
            goto done;
        }

        if ((streamState & stateToMatch) == 0) {
            LOG_ERR("Consumer: No Buffer registration received!, index: %d\n", regIdx);
            goto done;
        }

        // init and register
        buffers[regIdx].sync = NULL;
        buffers[regIdx].eglIndex = regIdx;

        LwEglApiClientBuffer clientBuffer = &(buffers[regIdx]);
        if (!registerBuffer(&clientBuffer)) {
            LOG_ERR("Consumer: Failed buffer registration, index: %d\n", regIdx);
            goto done;
        }

        LOG_INFO("Consumer: Buffer %d registered successfully!\n", regIdx);

        regIdx++;
    }

    // Enter Acquire - Release cycle
    LOG_INFO("Consumer: Entering acquire release cycle ...\n");
    while (!gSignalStop && (frameNumber < gTestArgs.maxFrames)) {
        // Acquire and process frames
        // Wait for new frame to be available and while producer is connected
        stateToMatch = LW_EGL_API_STREAM2_FRAME_AVAIL | LW_EGL_API_STREAM2_PROD_DISC;
        if (!TestExpStream2StateWait(pvtEglStream, foreverTimeout, stateToMatch, &streamState)) {
            LOG_ERR("Consumer: StateWatch on Frame available + producer disconnect failed. frameNumber: %d ...\n", frameNumber);
            break;
        }

        if ((streamState & LW_EGL_API_STREAM2_PROD_DISC)) {
            LOG_ERR("Producer disconnected preemptively!\n");
            break;
        }

        // Process available frame
        if (streamState & LW_EGL_API_STREAM2_FRAME_AVAIL) {
            if (!processFrame(acquireTimeout)) {
                LOG_ERR("Consumer: Failed to process frame. Last frame successfully seen: %d\n", frameNumber);
                break;
            }
        }
    }

done:
    if (frameNumber < gTestArgs.maxFrames) {
        LOG_ERR("Consumer: failed to get through all frames.\n");
    } else {
        LOG_INFO("Consumer: Received all frames successfully.\n");
        ret = true;
    }

    // cleanup
    if (buffers) {
        free(buffers);
        buffers = NULL;
    }

    isCompleted = true;
    return ret;
}

bool ConsumerStream2::disconnect(void)
{
    if (!TestExpStream2ConsumerDisconnect(pvtEglStream)) {
        LOG_ERR("streamConsumerDisconnect failed.\n");
        return false;
    }

    return true;
}

bool ConsumerStream2::queryMetadata(const LwEglApiStream2Frame* acquiredFrame)
{
    // noop by default
    return true;
}


