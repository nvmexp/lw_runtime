/*
 * Copyright (c) 2020-2021, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 */

#ifndef LWSCIEVENT_LINUX_H
#define LWSCIEVENT_LINUX_H

#include <lwscievent_internal.h>

/**
 * define CONSOLE_DEBUG to enable log on console
 */

/**
 * debug message
 */
#ifdef CONSOLE_DEBUG
    #define lwscievent_dbg(fmt, args...) \
        do { printf("scievt_dbg[L:%d]:%s: " fmt "\n",    \
        __LINE__, __func__, ## args); } while (false)
#else
    #define lwscievent_dbg(fmt, args...)
#endif /* CONSOLE_DEBUG */

#ifdef CONSOLE_DEBUG
    #define lwscievent_err(fmt, val)  \
           printf("%s: " fmt ": %d, %d\n", \
                __func__, __LINE__, val)
#else
    #define lwscievent_err(str, val)
#endif /* CONSOLE_DEBUG */

#define CheckZero(val) \
        do { \
            err = (val); \
            if(0 != (err)) { \
                lwscievent_err("scievt_err: error: line, ret", err); \
                ret = LwSciError_IlwalidState; \
                goto fail; \
            } \
        } while (false)

#define CheckZeroWithLock(val) \
        do { \
            err = (val); \
            if(0 != (err)) { \
                lwscievent_err("scievt_err: error: line, ret", err); \
                ret = LwSciError_IlwalidState; \
                CheckZero(pthread_mutex_lock(&linuxLoopService->mutex)); \
                goto fail; \
            } \
        } while (false)

#define CheckZeroLogPrint(val) \
        do { \
            err = (val); \
            if(0 != (err)) { \
                lwscievent_err("scievt_err: error: line, ret", err); \
                ret = LwSciError_IlwalidState; \
            } \
        } while (false)

#define CheckNull(val, scierr) \
        do { \
            if (NULL == (void const *)(val)) { \
                ret = (scierr); \
                lwscievent_err("scievt_err: null pointer: line, ret", (int32_t)ret);\
                goto fail; \
            } \
        } while (false)

#define CheckBothNull(val1, val2, scierr) \
        do { \
            if ((NULL == (void const *)(val1)) && \
                (NULL == (void const *)(val2))) { \
                ret = (scierr); \
                lwscievent_err("scievt_err: null pointer: line, ret", (int32_t)ret);\
                goto fail; \
            } \
        } while (false)

#define CheckTimeout(timeout, scierr) \
        do { \
            if ((((timeout) < 0) && \
                ((timeout) != LW_SCI_EVENT_INFINITE_WAIT)) || \
                ((timeout) > MAX_TIMEOUT_US)) { \
                ret = (scierr); \
                lwscievent_err("error: invalid timeout: ret", ret); \
                goto fail; \
            } \
        } while (false)


/*
 * Max event notifier count per process
 */
#define LWSCIEVENT_MAX_EVENTNOTIFIER 100U

typedef struct LwSciLinuxUserLocalEvent LwSciLinuxUserLocalEvent;
typedef struct LwSciLinuxUserNativeEvent LwSciLinuxUserNativeEvent;
typedef struct LwSciLinuxUserTimerEvent LwSciLinuxUserTimerEvent;
typedef struct LwSciLinuxUserEventNotifier LwSciLinuxUserEventNotifier;
typedef struct LwSciLinuxUserEventService LwSciLinuxUserEventService;
typedef struct LwSciLinuxUserEventLoop LwSciLinuxUserEventLoop;
typedef struct LwSciLinuxUserEventLoopService LwSciLinuxUserEventLoopService;

struct LwSciLinuxUserNativeEvent {
    LwSciNativeEvent nativeEvent;

    /* add Linux OS specific items */
};

struct LwSciLinuxUserLocalEvent {
    LwSciLocalEvent localEvent;

    /* Linux OS specific */
    int pipeFd[2]; /* pipe fd used for local events */
};

struct LwSciLinuxUserTimerEvent {
    LwSciTimerEvent timerEvent;

    /* Linux OS specific */
    int timerEventFd; /* timer fd used for timer events */
};

struct LwSciLinuxUserEventNotifier {
    LwSciEventNotifier eventNotifier;

    /* Linux OS specific */
    LwSciLinuxUserEventLoopService *linuxLoopService;

    /* EventNotifier associated with either of native/local/Timer event */
    LwSciLinuxUserNativeEvent *linuxNativeEvent;
    LwSciLinuxUserLocalEvent *linuxLocalEvent;
    LwSciLinuxUserTimerEvent *linuxTimerEvent;

    /* For Notifier call back */
    void (*callback)(void *cookie);
    void *cookie;
};

struct LwSciLinuxUserEventService {
    LwSciEventService eventService;
    LwSciLinuxUserEventNotifier *linuxNotifier[LWSCIEVENT_MAX_EVENTNOTIFIER];
    int numNotifier;
};

struct LwSciLinuxUserEventLoop {
    LwSciEventLoop eventLoop;
    LwSciLinuxUserEventLoopService *linuxEventLoopService;
};

struct LwSciLinuxUserEventLoopService {
    LwSciEventLoopService eventLoopService;
    LwSciLinuxUserEventService *linuxEventService;

    int epollFd; /* used when waiting on all notifiers in the EventService */
    int epollFdMin; /* used when waiting on few notifiers in the EventService */
    /** \brief epollEventArray is passed to each epoll_wait() call. */
    //struct epoll_event epollEventArray;
    /** \brief epollEventCount is the number of entries in epollEventArray. It
     * is always greater than or equal to the number of native notifiers so that
     * all pending events can be retrieved with a single epoll_wait() call. */
    size_t epollEventCount;

    int timerFd;

     /** \brief Used to ensure only one thread is in Run() at a time. */
    bool isRunning;

    /* To ensure only one wait is in progress at a time */
    bool isWaitInProgress;
    pthread_mutex_t mutex;
};
/******************************************************************************/


#endif


