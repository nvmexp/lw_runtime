/*
 * Copyright (c) 2016 - 2017 LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#ifndef _EGLTESTS_H
#define _EGLTESTS_H

#include "producer.h"
#include "consumer.h"

class Producer1Stream1 : public ProducerStream1 {
public:
    virtual void init(EGLDisplay dsp, EGLStreamKHR stream);
    virtual bool run(void);
};

class Consumer1Stream1 : public ConsumerStream1 {
public:
    virtual void init(EGLDisplay dsp, EGLStreamKHR stream);
    virtual bool run(void);
};

class Producer2Stream1 : public ProducerStream1 {
public:
    virtual void init(EGLDisplay dsp, EGLStreamKHR stream);
    virtual bool run(void);
};

class Consumer2Stream1 : public ConsumerStream1 {
public:
    virtual void init(EGLDisplay dsp, EGLStreamKHR stream);
    virtual bool run(void);
 };

class Producer3Stream1 : public ProducerStream1 {
public:
    ~Producer3Stream1();

    virtual void init(EGLDisplay dsp, EGLStreamKHR stream);
    virtual bool run(void);

protected:
    EglTestMetadata *metadata[LW_EGL_STREAM_METADATA_INTERNAL_BLOCKS];

    virtual bool setMetadata(int frameIdx);
};

class Consumer3Stream1 : public ConsumerStream1 {
public:
    virtual void init(EGLDisplay dsp, EGLStreamKHR stream);
    virtual bool run(void);

protected:
    virtual bool queryMetadata(int frameIdx);
 };

class Producer1Stream2 : public ProducerStream2 {
public:
    virtual void init(EGLDisplay dsp, EGLStreamKHR stream);
    virtual bool run(void);
};

class Consumer1Stream2 : public ConsumerStream2 {
public:
    virtual void init(EGLDisplay dsp, EGLStreamKHR stream);
    virtual bool run(void);
};

class Producer2Stream2 : public ProducerStream2 {
public:
    virtual void init(EGLDisplay dsp, EGLStreamKHR stream);
    virtual bool run(void);
};

class Consumer2Stream2 : public ConsumerStream2 {
public:
    virtual void init(EGLDisplay dsp, EGLStreamKHR stream);
    virtual bool run(void);
 };

class Producer3Stream2 : public ProducerStream2 {
public:
    ~Producer3Stream2();

    virtual void init(EGLDisplay dsp, EGLStreamKHR stream);
    virtual bool run(void);

protected:
    EglTestMetadata *metadata[LW_EGL_STREAM_METADATA_INTERNAL_BLOCKS];

    virtual bool setMetadata(int frameIdx);
};

class Consumer3Stream2 : public ConsumerStream2 {
public:
    virtual void init(EGLDisplay dsp, EGLStreamKHR stream);
    virtual bool run(void);

protected:
    virtual bool producerCapsMatch(void);
    virtual bool queryMetadata(const LwEglApiStream2Frame* acquiredFrame);
 };

#endif //_EGLTESTS_H
