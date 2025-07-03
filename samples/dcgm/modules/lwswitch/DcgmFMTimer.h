#pragma once

#include <signal.h>
#include <time.h>
#include <pthread.h>

typedef void (*TimerCB) (void*); 

class DcgmFMTimer
{

public:
    DcgmFMTimer(TimerCB callBackFunc, void* callbackCtx);

    ~DcgmFMTimer();

    static void timeoutHandler(sigval_t signum);

    void start(unsigned int interval);
    void stop(void);
    void restart(void);
    void onTimerExpiry(void);
private:

    void createTimer(void);
    void deleteTimer(void);
    void startTimer(void);
    void stopTimer(void);
    
    unsigned int mInterval;
    TimerCB mCallBackFunc;
    void* mCallbackCtx;
    timer_t mTimerId;
    struct itimerspec mTimerSpec;
};

