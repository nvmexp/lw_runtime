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

void Consumer::init(EGLDisplay dsp, EGLStreamKHR stream)
{
    display = dsp;
    eglStream = stream;
    frameNumber = 0;
    isCompleted = false;
    runStatus = false;
}

bool Consumer::complete(void)
{
    while (!isCompleted) {
       sleep(1);
    }

    return true;
}

void Consumer::join(void)
{
    pthread_join(thread, NULL);
}



