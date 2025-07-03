/*
 *  Copyright 2018-2021 LWPU Corporation.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 *
 */

#pragma once

#include <signal.h>
#include <time.h>

#if defined(__linux__) && (!defined(LW_MODS) || defined(LW_MODS_GDM_BUILD))
#include <pthread.h>
#endif

#if defined(_WINDOWS)
#define NOMINMAX
#include <windows.h>
#endif

typedef void (*TimerCB) (void*);

class FMTimer
{

public:
    FMTimer(TimerCB callBackFunc, void* callbackCtx);

    ~FMTimer();
#if defined(__linux__)
    static void timeoutHandler(sigval_t signum);
#else
    static void timeoutHandler(void *instancePointer, bool TimerOrWaitFired);
#endif

    void start(unsigned int interval);
    void stop(void);
    void restart(void);
    void onTimerExpiry(void);

    unsigned int getInterval() { return mInterval; }
#ifdef LW_MODS
    unsigned int getRunning() { return mRunning; }
    unsigned int getStart() { return mStart; }
    unsigned int getTimeoutMs() { return mTimeoutMs; }
    void increaseTimeout(unsigned int ms) { mTimeoutMs += ms; }
#endif
private:

    void createTimer(void);
    void deleteTimer(void);
    void startTimer(void);
    void stopTimer(void);
    void createTimerOS();
    void deleteTimerOS();
    void startTimerOS();
    void stopTimerOS();

    unsigned int mInterval;
    TimerCB mCallBackFunc;
    void* mCallbackCtx;

#ifdef LW_MODS
    int mThreadId;
    unsigned int mRunning;
    unsigned int mStart;
    unsigned int mTimeoutMs;
#endif
#if defined(__linux__)
    timer_t mTimerId;
    struct itimerspec mTimerSpec;
#else
    void *mTimerQueue;
    void *mTimerHandle;
#endif
};
