
#include <iostream>
#include <stdexcept>
#include <stdlib.h>
#include <string.h>

#include "DcgmFMTimer.h"

void
DcgmFMTimer::timeoutHandler(sigval_t signum)
{
    DcgmFMTimer* pObj = (DcgmFMTimer*) signum.sival_ptr;
    pObj->onTimerExpiry();
}

DcgmFMTimer::DcgmFMTimer(TimerCB callBackFunc, void* callbackCtx)
{
    mCallBackFunc = callBackFunc;
    mCallbackCtx = callbackCtx;
    mInterval = 0;
    createTimer();
}

DcgmFMTimer::~DcgmFMTimer()
{
    deleteTimer();
}

void
DcgmFMTimer::start(unsigned int interval)
{
    mInterval = interval;
    startTimer();
}

void
DcgmFMTimer::stop(void)
{
    stopTimer();
}

void
DcgmFMTimer::restart(void)
{
    stopTimer();
    startTimer();
}

void
DcgmFMTimer::onTimerExpiry(void)
{
    mCallBackFunc( mCallbackCtx );
}

void
DcgmFMTimer::createTimer(void)
{
    struct sigevent sigEvent;
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
}

void
DcgmFMTimer::deleteTimer(void)
{
    timer_delete( mTimerId );
}

void
DcgmFMTimer::startTimer(void)
{
    memset( &mTimerSpec, 0, sizeof(mTimerSpec) );
    // Start the timer
    mTimerSpec.it_value.tv_sec = mInterval;
    mTimerSpec.it_value.tv_nsec = 0;
    timer_settime( mTimerId, 0, &mTimerSpec, NULL );    
}

void
DcgmFMTimer::stopTimer(void)
{
    memset( &mTimerSpec, 0, sizeof(mTimerSpec) );
    mTimerSpec.it_value.tv_sec = 0;
    mTimerSpec.it_value.tv_nsec = 0;
    timer_settime( mTimerId, 0, &mTimerSpec, NULL );
}
