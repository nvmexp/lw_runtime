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
    Producer *producer = (Producer*) arg;

    LOG_INFO("running producer.\n");

    bool status = producer->run();
    if (!status) {
        LOG_ERR("Producer failed to present all frames.\n");
        gSignalStop = true;
    }

    producer->setRunStatus(status);
    return NULL;
}

void ProducerStream1::init(EGLDisplay dsp, EGLStreamKHR stream)
{
    Producer::init(dsp, stream);

    // set producer info.
    memset(&producerInfo, 0, sizeof(producerInfo));
    producerInfo.apiIdx = LW_EGL_API_TEST;
    producerInfo.handle = NULL;

    producerInfo.funcs.returnFrame = &LwEglTestProducerReturnFrame;

    // Init LwEglApiStreamFrame struct to pass to EGL.
    memset(&apiFrame, 0, sizeof(apiFrame));
}

bool ProducerStream1::connect()
{
    EGLint streamState;

    if (!TestExpStream1ProducerConnect(display, eglStream, &producerInfo)) {
        LOG_ERR("streamProducerConnect failed.\n");
        return false;
    }

    if (!EGLStreamQuery(display, eglStream,
                        EGL_STREAM_STATE_KHR, &streamState)) {
        return false;
    }

    return true;
}

bool ProducerStream1::getCaps(void)
{
    memset(&caps, 0, sizeof(LwEglApiProducerCaps));

    if (!TestExpStream1ProducerGetCaps(display, eglStream, &caps)) {
        LOG_ERR("streamProducerGetCaps failed\n");
        return false;
    }

    return true;
}

void ProducerStream1::create(void)
{
    pthread_create(&thread, NULL, runProducer, this);
}

bool ProducerStream1::run(void)
{
    const struct timespec reg = {0, 1000};
    bool ret = true;

    while (!gEglState.streamBuffers) {
        // Wait for streamBuffers allocation done
        sleep(1);
    }

    for (int index = 0; (index < gTestArgs.maxFrames) && !gSignalStop; ++index) {
        // Simulate that Producer spends some time on producing a frame
        nanosleep(&reg, NULL);

        if (!setMetadata(index)) {
            LOG_ERR("setMetadata failed for frame %d\n", index);
            return false;
        }

        if (!updateFrame()) {
            LOG_ERR("updateFrame failed for frame %d\n", index);
            break;
        }

        if (!TestExpStream1ProducerPresentFrame(display, eglStream, &apiFrame)) {
            LOG_ERR("streamProducerPresentFrame failed for frame %d.\n", index);
            break;
        }

        // Unrefcount GLSI image after RegisterBuffer/PresentFrame
        // (EGL still holds a reference)
        LwGlsiEglImage::FromHandle(apiFrame.producerImage)->unref();

        if (!EGLStreamQuery(display, eglStream,
                            EGL_PRODUCER_FRAME_KHR, &frameNumber)) {
            break;
        }
        LOG_INFO("Present frameNumber = %d.\n", frameNumber);
    }

    if (frameNumber < gTestArgs.maxFrames) {
        ret = false;
    }

    isCompleted = true;
    return ret;
}

bool ProducerStream1::disconnect(void)
{
    if (!TestExpStream1ProducerDisconnect(display, eglStream)) {
        LOG_ERR("streamProducerDisconnect failed.\n");
        return false;
    }

    return true;
}

bool ProducerStream1::updateFrame(void)
{
    LwGlsiEglImageHandle glsiImage = (LwGlsiEglImageHandle)NULL;
    LwU32 index = gEglState.streamBufferLwrr;
    EglTestSurface *testSurf = NULL;

    testSurf = LwEglTestSurfaceCreate(type,
                                      gEglState.width,
                                      gEglState.height,
                                      (gTestArgs.processMode == CROSS_PARTITION) ?
                                          true : false,
                                      gTestArgs.vmId);
    if (testSurf == NULL) {
        LOG_ERR("Could not create a EglTestSurface.\n");
        return false;
    }

    if (!TestExpGlsiImageFromLwRmSurface(&glsiImage,
                                         display,
                                         (const LwRmSurface*)&testSurf->rmSurface,
                                         testSurf->numSurfaces)) {
        LOG_ERR("Could not create LwGlsiEglImage from LwRmSurface.\n");
        return false;
    }

    // Destroy EglTestSurface
    LwEglTestSurfaceDestroy(testSurf);

    assert(gEglState.streamBuffers != NULL);

    if (caps.bufferCache) {
        if (gEglState.streamBuffers[index].eglIndex != -1) {
            memset(&apiFrame, 0, sizeof(apiFrame));
            apiFrame.index = gEglState.streamBuffers[index].eglIndex;

            if (!TestExpStream1ProducerUnregisterBuffer(display,
                                                       eglStream,
                                                       &apiFrame)) {
                LOG_ERR("streamProducerUnregisterBuffer failed for frame %d.\n", index);
                return false;
            }

            gEglState.streamBuffers[index].eglIndex = -1;

            // TODO: handle gEglState.streamBuffers[index].sync if it is not NULL
        }
    }

    memset(&apiFrame, 0, sizeof(apiFrame));
    apiFrame.producerHandle    = (LwEglApiClientBuffer)&gEglState.streamBuffers[index];
    apiFrame.producerImage     = glsiImage;
    LW_EGL_API_FRAME_SYNC_INIT(apiFrame.sync);

    if (caps.bufferCache) {
        if (!TestExpStream1ProducerRegisterBuffer(display, eglStream, &apiFrame)) {
            LOG_ERR("streamProducerPresentFrame failed for frame %d.\n", index);
            return false;
        }
    }

    gEglState.streamBufferLwrr++;
    if (gEglState.streamBufferLwrr >= gEglState.streamBufferCount) {
        gEglState.streamBufferLwrr = 0;
    }

    return true;
}
