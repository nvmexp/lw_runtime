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

void Producer1Stream1::init(EGLDisplay dsp, EGLStreamKHR stream)
{
    ProducerStream1::init(dsp, stream);
}

bool Producer1Stream1::run(void)
{
    bool ret = ProducerStream1::run();

    LOG_INFO("ProducerStream1 run\n");

    return ret;
}

void Consumer1Stream1::init(EGLDisplay dsp, EGLStreamKHR stream)
{
    ConsumerStream1::init(dsp, stream);
}

bool Consumer1Stream1::run(void)
{
    bool ret = ConsumerStream1::run();

    LOG_INFO("Consumer1Stream1 run\n");

    return ret;
}

void Producer1Stream2::init(EGLDisplay dsp, EGLStreamKHR stream)
{
    ProducerStream2::init(dsp, stream);
}

bool Producer1Stream2::run(void)
{
    bool ret = ProducerStream2::run();

    LOG_INFO("Producer1Stream2 run\n");

    return ret;
}

void Consumer1Stream2::init(EGLDisplay dsp, EGLStreamKHR stream)
{
    ConsumerStream2::init(dsp, stream);
}

bool Consumer1Stream2::run(void)
{
    bool ret = ConsumerStream2::run();

    LOG_INFO("Consumer1Stream2 run\n");

    return ret;
}
