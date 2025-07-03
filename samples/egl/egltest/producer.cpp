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

void Producer::init(EGLDisplay dsp, EGLStreamKHR stream)
{
    display = dsp;
    eglStream = stream;
    frameNumber = 0;
    isCompleted = false;
    runStatus = false;
    type = EGLTEST_DATA_FORMAT_TYPE_YUV422;
}

bool Producer::complete(void)
{
    while (!isCompleted) {
        sleep(1);
    }

    return true;
}

void Producer::join(void)
{
    pthread_join(thread, NULL);
}

bool Producer::setMetadata(int frameIdx)
{
    // default noop
    return true;
}
