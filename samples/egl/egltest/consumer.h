/*
 * Copyright (c) 2016 - 2017 LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#ifndef _EGLTEST_CONSUMER_H
#define _EGLTEST_CONSUMER_H

#include <pthread.h>
#include "eglstream.h"

class Consumer {
public:
    Consumer() {};
    ~Consumer() {};

    virtual void init(EGLDisplay dsp, EGLStreamKHR stream);
    virtual bool run(void) = 0;
    virtual bool connect(void) = 0;
    virtual bool getCaps(void) = 0;
    virtual void create(void) = 0;
    virtual bool disconnect(void) = 0;

    bool complete(void);
    void join(void);
    inline void setRunStatus(bool status) { runStatus = status; }
    inline bool getRunStatus() const { return runStatus; }

protected:
    EGLDisplay display;
    // public api handle
    EGLStreamKHR eglStream;
    pthread_t thread;
    int frameNumber;
    // track if dispatched run completed
    bool isCompleted;
    // track if dispatched run was successfull
    bool runStatus;
};

class ConsumerStream1 : public Consumer {
public:
    ConsumerStream1() {};
    ~ConsumerStream1() {};

    virtual void init(EGLDisplay dsp, EGLStreamKHR stream);
    virtual bool run(void);
    bool connect(void);
    bool getCaps(void);
    void create(void);
    bool disconnect(void);
    const LwEglApiConsumerCaps &retCaps(void) { return caps; }

protected:
    LwEglApiConsumerInfo consumerInfo;
    LwEglApiConsumerCaps caps;

    bool acquireFrame(int timeout);
    virtual bool queryMetadata(int frameIdx);
};

class ConsumerStream2 : public Consumer {
public:
    ConsumerStream2() {};
    ~ConsumerStream2() {};

    virtual void init(EGLDisplay dsp, EGLStreamKHR stream);
    virtual bool run(void);
    bool connect(void);
    bool getCaps(void);
    void create(void);
    bool disconnect(void);
    const LwEglApiStream2ConsumerCaps &retCaps(void) { return caps; }

protected:
    EGLStreamKHR pvtEglStream; // This is the private internal client handle
    LwEglApiStream2ConsumerCaps caps;
    LwEglApiStream2ProducerCaps prodCaps;
    LwS64 stateWaitTimeout;
    LwS64 foreverTimeout;
    int numBuffers;

    bool reserve(void);
    bool getConstAttribs(void);
    bool getElwironment(void);
    bool registerBuffer(LwEglApiClientBuffer* clientBuffer);
    bool processFrame(int timeout);
    bool verifyFrame(LwEglApiStream2Frame* acquiredFrame);
    virtual bool producerCapsMatch(void);
    virtual bool queryMetadata(const LwEglApiStream2Frame* acquiredFrame);
};

#endif //_EGLTEST_CONSUMER_H
