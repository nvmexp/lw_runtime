/*
 * Copyright (c) 2016-2018, LWPU CORPORATION. All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 */

#include "producer.h"
#include "surface.h"

static void* runProducer(void *arg)
{
    ProducerStream2 *producer = (ProducerStream2*) arg;

    LOG_INFO("running Producer.\n");

    bool status = producer->run();
    if (!status) {
        LOG_ERR("Producer: run failed.\n");
        gSignalStop = true;
    }

    producer->setRunStatus(status);
    return NULL;
}

void ProducerStream2::init(EGLDisplay dsp, EGLStreamKHR stream)
{
    Producer::init(dsp, stream);

    // set producer info.
    memset(&caps, 0, sizeof(caps));
    caps.image.useGlsi = 1;
    caps.sync.types = LwCommonSyncTypeBit_Reg | LwCommonSyncTypeBit_Sem;
    caps.sync.useLwRm = 1;

    // Init LwEglApiStream2Frame struct to pass to EGL.
    memset(&apiFrame, 0, sizeof(apiFrame));

    // timeouts
    stateWaitTimeout = 3000;
    foreverTimeout = -1;

    // Reserve
    pvtEglStream = NULL;
    if (!reserve()) {
        LOG_ERR("Producer: Reserve failed.\n");
    }
}

bool ProducerStream2::reserve(void)
{
    LwU64 streamState = 0;

    if (!TestExpStream2ProducerReserve(display, eglStream, &pvtEglStream)) {
        LOG_ERR("Producer: ProducerReserve failed.\n");
        return false;
    }

    if (!TestExpStream2StatePoll(pvtEglStream, &streamState)) {
        LOG_ERR("Producer: StatePoll failed.\n");
        return false;
    }

    // producer state should be reserved
    if ((streamState & LW_EGL_API_STREAM2_PROD_RSVD) == 0) {
        LOG_ERR("Producer: PROD_RSVD not set.\n");
        return false;
    }

    return true;
}

bool ProducerStream2::connect()
{
    LwU64 streamState;

    if (!TestExpStream2ProducerConnectStream(pvtEglStream, &caps)) {
        LOG_ERR("Producer: ProducerConnectStream failed.\n");
        return false;
    }

    if (!TestExpStream2StatePoll(pvtEglStream, &streamState)) {
        LOG_ERR("Producer: StatePoll failed.\n");
        return false;
    }

    // consumer state should be connected
    if ((streamState & LW_EGL_API_STREAM2_PROD_CONN) == 0) {
        LOG_ERR("Producer: PROD_CONN not set.\n");
        return false;
    }

    return true;
}

bool ProducerStream2::getCaps(void)
{
    memset(&caps, 0, sizeof(LwEglApiStream2ProducerCaps));

    if (!TestExpStream2ProducerGetCaps(pvtEglStream, &caps)) {
        LOG_ERR("Producer: ProducerGetCaps failed\n");
        return false;
    }

    return true;
}

void ProducerStream2::create(void)
{
    pthread_create(&thread, NULL, runProducer, this);
}

bool ProducerStream2::createSurface(LwGlsiEglImageHandle *glsiImageHandle)
{
    EglTestSurface *testSurf = NULL;

    testSurf = LwEglTestSurfaceCreate(type,
                                      gEglState.width,
                                      gEglState.height,
                                      (gTestArgs.processMode == CROSS_PARTITION) ?
                                          true : false,
                                      gTestArgs.vmId);
    if (testSurf == NULL) {
        LOG_ERR("Producer: Could not create a EglTestSurface.\n");
        return false;
    }

    LwGlsiEglImageHandle glsiImage = (LwGlsiEglImageHandle)NULL;
    // Create Glsi Image
    // Note: This takes a ref count
    if (!TestExpGlsiImageFromLwRmSurface(&glsiImage,
                                         display,
                                         (const LwRmSurface*)&testSurf->rmSurface,
                                         testSurf->numSurfaces)) {
        LOG_ERR("Producer: Could not create LwGlsiEglImage from LwRmSurface.\n");
        return false;
    }

    // Destroy EglTestSurface
    LwEglTestSurfaceDestroy(testSurf);

    *glsiImageHandle = glsiImage;

    return true;
}

bool ProducerStream2::registerBuffers()
{
    LwU32 index = 0;

    // preallocate frame image handles
    glsiImages = (LwGlsiEglImageHandle*)malloc(gEglState.streamBufferCount * sizeof(LwGlsiEglImageHandle));

    // now go through each slot and register the buffer
    // also allocate the backing surface
    for (index = 0; index < gEglState.streamBufferCount; index++) {
        // allocate image surface
        createSurface(&(glsiImages[index]));
        // TODO: flood Glsi image with some pattern that the consumer can verify
        LwGlsiEglImage::FromHandle(glsiImages[index]);

        // get the client buffer object
        LwEglApiClientBuffer streambuff = (LwEglApiClientBuffer)&gEglState.streamBuffers[index];

        // unregister buffers if already registered
        if (gEglState.streamBuffers[index].eglIndex != -1) {
            if (!TestExpStream2ProducerUnregisterBuffer(pvtEglStream,
                                                        streambuff)) {
                LOG_ERR("Producer: UnregisterBuffer failed, frame %d.\n", index);
                return false;
            }

            gEglState.streamBuffers[index].eglIndex != -1;
            // TODO: handle gEglState.streamBuffers[index]. sync if it is not NULL
        }

        // Register Buffers
        gEglState.streamBuffers[index].eglIndex = index;
        gEglState.streamBuffers[index].sync = NULL;
        LwEglApiClientBuffer clientBuffer = &(gEglState.streamBuffers[index]);
        if (!TestExpStream2ProducerRegisterBuffer(pvtEglStream,
                                                  (LwEglApiBufferHandle) glsiImages[index],
                                                  clientBuffer)) {
            LOG_ERR("Producer: RegisterBuffer failed, frame %d.\n", index);
            return false;
        }
    }

    return true;
}

bool ProducerStream2::updateFrame()
{
    const struct timespec reg = {0, 1000};
    LwU32 index = gEglState.streamBufferLwrr;
    // Simulate that Producer spends some time on producing a frame
    nanosleep(&reg, NULL);

    // update frame object
    memset(&apiFrame, 0, sizeof(apiFrame));
    apiFrame.buffer    = (LwEglApiClientBuffer)&gEglState.streamBuffers[index];
    LW_EGL_API_FRAME_SYNC_INIT(apiFrame.lwrmSync.sync);

    // wrap increment buffer count
    gEglState.streamBufferLwrr++;
    if (gEglState.streamBufferLwrr >= gEglState.streamBufferCount) {
        gEglState.streamBufferLwrr = 0;
    }

    return true;
}

bool ProducerStream2::run(void)
{
    LwU64 streamState = 0;
    const struct timespec reg = {0, 1000};
    bool ret = false;

    // Wait for consumer connection
    LwU64 stateToMatch = LW_EGL_API_STREAM2_INIT_COMM | LW_EGL_API_STREAM2_CONS_CONN;
    if (!TestExpStream2StateWait(pvtEglStream, stateWaitTimeout, stateToMatch, &streamState)) {
        LOG_ERR("Producer: StateWait failed on INIT_COMM | CONS_CONN.\n");
        goto done;
    }

    if ((streamState & stateToMatch) != 0) {
        LOG_INFO("Producer: Connection ready.\n");
    }

    // wait for all stream buffers to be allocated
    if (sem_wait(&(gEglState.streamBufferReady)) != 0) {
        LOG_ERR("Producer: Stream Buffer Sem wait failed.\n");
        goto done;
    }

    assert(gEglState.streamBuffers != NULL);

    // pre register client buffers and allocate images
    if (!registerBuffers()) {
        LOG_ERR("Producer: Buffer registration failed");
        goto done;
    }

    for (int index = 0; (index < gTestArgs.maxFrames) && !gSignalStop; ++index) {
        // after producing first streambufferCount frames, we need to wait
        // for a buffer from previous frame to be available
        if (index > gEglState.streamBufferCount - 1) {
            stateToMatch = LW_EGL_API_STREAM2_FRAME_RETURN;
            if (!TestExpStream2StateWait(pvtEglStream, foreverTimeout, stateToMatch, &streamState)) {
                LOG_ERR("Producer: StateWait failed on FRAME_RETURN\n");
                break;
            }

            if (streamState & stateToMatch) {
                LOG_INFO("Producer: Processing frame, index: %d.\n", index);

                LwEglApiStream2Frame returnFrame = {0};
                LwEglApiClientBuffer waitOnBuffer = (LwEglApiClientBuffer)&gEglState.streamBuffers[gEglState.streamBufferLwrr];
                if (!TestExpStream2ProducerReturnFrame(pvtEglStream,
                                                       stateWaitTimeout,
                                                       waitOnBuffer,
                                                       &returnFrame)) {
                    LOG_ERR("Producer: ReturnFrame failed.\n");
                    break;
                }

                LOG_INFO("Producer: Returned frame %d, buffer %p.\n", (int) returnFrame.number, returnFrame.buffer);

                assert(waitOnBuffer == returnFrame.buffer);
            }
        }

        if (!setMetadata(index)) {
            LOG_ERR("Producer: setMetadata failed for frame %d\n", index);
            return false;
        }

        if (!updateFrame()) {
            LOG_ERR("Producer: updateFrame failed for frame %d\n", index);
            break;
        }

        if (!TestExpStream2ProducerPresentFrame(pvtEglStream, &apiFrame)) {
            LOG_ERR("Producer: PresentFrame failed, frame %d.\n", index);
            break;
        }

        if (!EGLStreamQuery(display, eglStream, EGL_PRODUCER_FRAME_KHR, &frameNumber)) {
            break;
        }

        // TODO: Enable this once Bug 2404192 is fixed
        // assert(frameNumber == (int) apiFrame.number);
        LOG_INFO("Producer: Presented frameNumber = %d.\n", frameNumber);
    }

done:
    if (frameNumber < gTestArgs.maxFrames) {
        LOG_ERR("Producer: failed to get through all frames\n");
        ret = false;
    } else {
        LOG_INFO("Producer: presented all frames successfully\n");
        ret = true;
    }

    isCompleted = true;
    return ret;
}

bool ProducerStream2::disconnect(void)
{
    // cleanup
    for (int index = 0; index < gEglState.streamBufferCount; index++) {
        LwGlsiEglImage::FromHandle(glsiImages[index])->unref();
    }
    if (glsiImages) {
        free(glsiImages);
        glsiImages = NULL;
    }

    // disconnect
    if (!TestExpStream2ProducerDisconnect(pvtEglStream)) {
        LOG_ERR("Producer: Disconnect failed.\n");
        return false;
    }

    return true;
}

