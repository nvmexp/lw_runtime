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

static void* runConsumer(void *arg)
{
    Consumer *consumer = (Consumer*) arg;

    LOG_INFO(("running consumer.\n"));

    bool status = consumer->run();
    if (!status) {
        LOG_ERR("Consumer failed.\n");
    }

    consumer->setRunStatus(status);
    return NULL;
}

void ConsumerStream1::init(EGLDisplay dsp, EGLStreamKHR stream)
{
    Consumer::init(dsp, stream);

    // set consumer info.
    memset(&consumerInfo, 0, sizeof(consumerInfo));
    consumerInfo.apiIdx = LW_EGL_API_TEST;
    consumerInfo.handle = NULL;

    consumerInfo.caps.supportClientUpdate = true;

    consumerInfo.funcs.updateFrameAttr = &LwEglTestConsumerUpdateFrameAttr;
    consumerInfo.funcs.registerBuffer = &LwEglTestConsumerRegisterBuffer;
    consumerInfo.funcs.unregisterBuffer = &LwEglTestConsumerUnregisterBuffer;
}

bool ConsumerStream1::connect(void)
{
    EGLint streamState;

    // Test assertion in EGL code path if [-t assert] is specified
    if (gTestArgs.testName && !strcmp(gTestArgs.testName, "assert")) {
        if (!TestExpStream1ConsumerConnect(display, eglStream, NULL)) {
            LOG_ERR("streamConsumerConnect failed.\n");
            return false;
        }
    }

    if (!TestExpStream1ConsumerConnect(display, eglStream, &consumerInfo)) {
        LOG_ERR("streamConsumerConnect failed.\n");
        return false;
    }

    if (!EGLStreamQuery(display, eglStream,
                        EGL_STREAM_STATE_KHR, &streamState)) {
        return false;
    }

    return true;
}

bool ConsumerStream1::getCaps(void)
{
    memset(&caps, 0, sizeof(LwEglApiConsumerCaps));

    if (!TestExpStream1ConsumerGetCaps(display, eglStream, &caps)) {
        LOG_ERR("streamConsumerGetCaps failed.\n");
        return false;
    }

    return true;
}

void ConsumerStream1::create(void)
{
    pthread_create(&thread, NULL, runConsumer, this);
}

bool ConsumerStream1::run(void)
{
    const struct timespec reg = {0, 2000};
    EGLint streamState = 0;
    bool ret = true;
    int timeout;

    if (!EGLStreamQuery(display, eglStream,
                        EGL_CONSUMER_ACQUIRE_TIMEOUT_USEC_KHR, &timeout)) {
        timeout = 16000;
    }
    LOG_INFO("Consumer acquire timeout: %d usec.\n", timeout);

    while (!gSignalStop && (frameNumber < gTestArgs.maxFrames)) {
        if (!EGLStreamQuery(display, eglStream,
                            EGL_STREAM_STATE_KHR, &streamState)) {
            break;
        }

        if ((streamState != EGL_STREAM_STATE_NEW_FRAME_AVAILABLE_KHR) &&
            (streamState != EGL_STREAM_STATE_OLD_FRAME_AVAILABLE_KHR)) {
            LOG_INFO("Producer has not generated any frames yet!\n");
            continue;
        }

        if (!acquireFrame(timeout)) {
            break;
        }

        nanosleep(&reg, NULL);

        if (!EGLStreamQuery(display, eglStream,
                            EGL_CONSUMER_FRAME_KHR, &frameNumber)) {
            break;
        }

        if (!queryMetadata(frameNumber)) {
            break;
        }

        LOG_INFO("Consume frameNumber = %d.\n", frameNumber);
    }

    if (frameNumber < gTestArgs.maxFrames) {
        ret = false;
    }

    isCompleted = true;
    return ret;
}

bool ConsumerStream1::disconnect(void)
{
    if (!TestExpStream1ConsumerDisconnect(display, eglStream)) {
        LOG_ERR("streamConsumerDisconnect failed.\n");
        return false;
    }

    return true;
}

bool ConsumerStream1::acquireFrame(int timeout)
{
    LwEglApiStreamUpdateAttrs updateAttrs;

    // Fill in EGL attribute structure with timeout,
    // which will be passed through to the Import function.
    memset(&updateAttrs, 0, sizeof(updateAttrs));
    updateAttrs.timeout = timeout;

    if (!TestExpStream1ConsumerUpdateFrameAttr(display,
                                              eglStream,
                                              true,
                                              false,
                                              &updateAttrs)) {
        LOG_ERR("updateFrameAttr failed.\n");
        return false;
    }

    return true;
}

bool ConsumerStream1::queryMetadata(int frameIdx)
{
    // noop by default
    return true;
}