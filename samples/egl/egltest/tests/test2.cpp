/*
 * Copyright (c) 2016-2017, LWPU CORPORATION. All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 */

#include "tests.h"

/*
 * producerInfo.caps.bufferCache needs to be set to true
 * in order to test the export functions:
 *   streamProducerRegisterBuffer
 *   streamProducerUnregisterBuffer
 */
void Producer2Stream1::init(EGLDisplay dsp, EGLStreamKHR stream)
{
    ProducerStream1::init(dsp, stream);

    producerInfo.caps.bufferCache = LW_TRUE;
}

bool Producer2Stream1::run(void)
{
    bool ret = ProducerStream1::run();

    LOG_INFO("Producer2Stream1 run\n");

    return ret;
}

void Consumer2Stream1::init(EGLDisplay dsp, EGLStreamKHR stream)
{
    ConsumerStream1::init(dsp, stream);

    consumerInfo.caps.bufferCache = LW_TRUE;
}

bool Consumer2Stream1::run(void)
{
    bool ret = ConsumerStream1::run();

    LOG_INFO("Consumer2Stream1 run\n");

    return ret;
}

void Producer2Stream2::init(EGLDisplay dsp, EGLStreamKHR stream)
{
    // TODO: Enable this once test is implemented
    // ProducerStream2::init(dsp, stream);
    isCompleted = true;
}

bool Producer2Stream2::run(void)
{
    // TODO: Enable this once test is implemented
    // bool ret = ProducerStream2::run();
    return true;
}

void Consumer2Stream2::init(EGLDisplay dsp, EGLStreamKHR stream)
{
    // TODO: Enable this once test is implemented
    // ConsumerStream2::init(dsp, stream);
    isCompleted = true;
}

bool Consumer2Stream2::run(void)
{
    // TODO: Enable this once test is implemented
    // bool ret = ConsumerStream2::run();
    return true;
}
