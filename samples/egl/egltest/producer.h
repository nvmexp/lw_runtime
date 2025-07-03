/*
 * Copyright (c) 2016 - 2017 LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#ifndef _EGLTEST_PRODUCER_H
#define _EGLTEST_PRODUCER_H

#include <pthread.h>
#include "eglstream.h"

class Producer {
public:
    Producer() {};
    ~Producer() {};

    virtual void init(EGLDisplay dsp, EGLStreamKHR stream);
    virtual bool run(void) = 0;
    virtual bool connect(void) = 0;
    virtual bool getCaps(void) = 0;
    virtual void create(void) = 0;
    virtual bool disconnect(void) = 0;
    virtual bool updateFrame(void) = 0;

    bool complete(void);
    void join(void);
    inline void setRunStatus(bool status) { runStatus = status; }
    inline bool getRunStatus() const { return runStatus; }

protected:
    EGLDisplay display;
    EGLStreamKHR eglStream;
    pthread_t thread;
    int frameNumber;
    // track if dispatched run completed
    bool isCompleted;
    // track if dispatched run was successfull
    bool runStatus;
    EglTestDataFormatType type;

    virtual bool setMetadata(int frameIdx);
};

class ProducerStream1 : public Producer {
public:
    ProducerStream1() {};
    ~ProducerStream1() {};

    virtual void init(EGLDisplay dsp, EGLStreamKHR stream);
    virtual bool run(void);
    bool connect(void);
    bool getCaps(void);
    void create(void);
    bool disconnect(void);
    bool updateFrame(void);

protected:
    LwEglApiProducerInfo producerInfo;
    LwEglApiProducerCaps caps;
    LwEglApiStreamFrame apiFrame;
};

class ProducerStream2 : public Producer {
public:
    ProducerStream2() {};
    ~ProducerStream2() {};

    virtual void init(EGLDisplay dsp, EGLStreamKHR stream);
    virtual bool run(void);
    bool connect(void);
    bool getCaps(void);
    void create(void);
    bool disconnect(void);
    bool updateFrame(void);

protected:
    EGLStreamKHR pvtEglStream; // This is the private internal client handle
    LwEglApiStream2Elw elw;
    LwEglApiStream2ConstantAttr attr;
    LwEglApiStream2ProducerCaps caps;
    LwEglApiStream2Frame apiFrame;
    LwGlsiEglImageHandle* glsiImages;
    LwS64 stateWaitTimeout;
    LwS64 foreverTimeout;

    bool reserve(void);
    bool createSurface(LwGlsiEglImageHandle *glsiImageHandle);
    bool registerBuffers();
};

#endif //_EGLTEST_PRODUCER_H
