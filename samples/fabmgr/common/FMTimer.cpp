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

#include <iostream>
#include <stdexcept>
#include <stdlib.h>
#include <string.h>

#include "FMTimer.h"
#include "fm_log.h"

#if defined(LW_MODS) && !defined(LW_MODS_GDM_BUILD)
#include "modsdrv.h"
#endif

#if defined(LW_MODS) && !defined(LW_MODS_GDM_BUILD)
#define HANDLER_INTERVAL_MS 10
static void ModsHandler(void* args)
{
    FMTimer* pObj = (FMTimer*)args;

    while (pObj->getRunning())
    {
        if (pObj->getStart())
        {
            if (pObj->getTimeoutMs() >= (pObj->getInterval() * 1000))
            {
                pObj->onTimerExpiry();
                pObj->stop();
            }
            else
            {
                pObj->increaseTimeout(HANDLER_INTERVAL_MS);
            }
        }

        ModsDrvSleep(HANDLER_INTERVAL_MS);
    }
}
#endif

#if defined(__linux__)
void
FMTimer::timeoutHandler(sigval_t signum)
{
    FMTimer* pObj = (FMTimer*) signum.sival_ptr;
    pObj->onTimerExpiry();
}
#else
void CALLBACK
FMTimer::timeoutHandler(void *instancePointer, bool TimerOrWaitFired)
{
    FMTimer* pObj = (FMTimer*) instancePointer;
    pObj->onTimerExpiry();
}
#endif

FMTimer::FMTimer(TimerCB callBackFunc, void* callbackCtx)
{
    mCallBackFunc = callBackFunc;
    mCallbackCtx = callbackCtx;
    mInterval = 0;
#if defined(LW_MODS) && !defined(LW_MODS_GDM_BUILD)
    mThreadId = 0;
    mRunning = 0;
	mStart = 0;
    mTimeoutMs = 0;
#elif __linux__
    mTimerSpec = {{0}};
#else
    mTimerQueue = NULL;
    mTimerHandle = NULL;
#endif
    createTimer();
}

FMTimer::~FMTimer()
{
    deleteTimer();
}

void
FMTimer::start(unsigned int interval)
{
    mInterval = interval;
    startTimer();
}

void
FMTimer::stop(void)
{
    stopTimer();
}

void
FMTimer::restart(void)
{
    stopTimer();
    startTimer();
}

void
FMTimer::onTimerExpiry(void)
{
    mCallBackFunc( mCallbackCtx );
}

void
FMTimer::createTimer(void)
{
#if defined(LW_MODS) && !defined(LW_MODS_GDM_BUILD)
    mRunning = 1;
    mThreadId = ModsDrvCreateThread(ModsHandler, this, 0, "FM timer");
#else
    createTimerOS();
#endif
}

void
FMTimer::createTimerOS()
{
#if defined(__linux__)
    struct sigevent sigEvent = {{0}};
    pthread_attr_t threadAttr;
    sched_param schedParam;

    pthread_attr_init( &threadAttr );
    schedParam.sched_priority = 255;
    pthread_attr_setschedparam( &threadAttr, &schedParam );

    // this will create a new timer and the sigev_notify_function will be called from
    // a new thread context
    sigEvent.sigev_notify_attributes = &threadAttr;
    sigEvent.sigev_notify = SIGEV_THREAD;
    sigEvent.sigev_notify_function = timeoutHandler;
    sigEvent.sigev_value.sival_ptr = this;

    timer_create( CLOCK_REALTIME, &sigEvent, &mTimerId );
#else
    mTimerQueue = CreateTimerQueue();
    if (mTimerQueue == NULL) {
        FM_LOG_ERROR("Request to create a timer queue has failed with error: %d", GetLastError());
    }
#endif
}

void
FMTimer::deleteTimer(void)
{
#if defined(LW_MODS) && !defined(LW_MODS_GDM_BUILD)
    mRunning = 0;
    ModsDrvJoinThread(mThreadId);
    mThreadId = 0;
#else
    deleteTimerOS();
#endif
}

void
FMTimer::deleteTimerOS()
{
#if defined(__linux__)
    timer_delete( mTimerId );
#else
    if (!DeleteTimerQueueEx(mTimerQueue, NULL)) {
        FM_LOG_ERROR("Request to delete the timer queue has failed with error: %d", GetLastError());
    }
#endif
}

void
FMTimer::startTimer(void)
{
#if defined(LW_MODS) && !defined(LW_MODS_GDM_BUILD)
    mStart = 1;
    mTimeoutMs = 0;
#else
    startTimerOS();
#endif
}

void
FMTimer::startTimerOS()
{
#if defined(__linux__)
    memset( &mTimerSpec, 0, sizeof(mTimerSpec) );
    // Start the timer
    mTimerSpec.it_value.tv_sec = mInterval;
    mTimerSpec.it_value.tv_nsec = 0;
    timer_settime( mTimerId, 0, &mTimerSpec, NULL );
#else
    bool timer = CreateTimerQueueTimer(&mTimerHandle, mTimerQueue, (WAITORTIMERCALLBACK)timeoutHandler,
                                          this, (mInterval ) * 1000, 0, WT_EXELWTEINTIMERTHREAD);
    if (timer == 0) {
        //log error
        FM_LOG_ERROR("Request to start timer failed with error: %d", GetLastError());
    }
#endif
}

void
FMTimer::stopTimer(void)
{
#if defined(LW_MODS) && !defined(LW_MODS_GDM_BUILD)
    mStart = 0;
    mTimeoutMs = 0;
#else
    stopTimerOS();
#endif
}

void
FMTimer::stopTimerOS()
{
#if defined(__linux__)
    memset( &mTimerSpec, 0, sizeof(mTimerSpec) );
    mTimerSpec.it_value.tv_sec = 0;
    mTimerSpec.it_value.tv_nsec = 0;
    timer_settime( mTimerId, 0, &mTimerSpec, NULL );
#else
    if (!DeleteTimerQueueTimer(mTimerQueue, mTimerHandle, NULL)) {
        //log error
        DWORD gle = GetLastError();
        if (gle == ERROR_IO_PENDING)
            return;
        else
            FM_LOG_ERROR("Request to stop timer failed with error: %d", GetLastError());
    }
#endif
}
