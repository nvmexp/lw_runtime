/*
 * Copyright (c) 2019-2021, LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <errno.h>
#include <sys/types.h>
#include <unistd.h>
#include <ctype.h>
#include <stdint.h>
#include <signal.h>
#include <pthread.h>
#include <sys/epoll.h>
#include <sys/timerfd.h>

#include <lwsciipc_internal.h>
#include <lwscievent_linux.h>

#define MAX_EPOLL_EVENTS 10
#define MAX_TIMEOUT_US 315360000000000

static LwSciError LwSciEventLoopService_CreateEventLoop(
            LwSciEventLoopService* eventLoopService,
            LwSciEventLoop **newEventLoop);
static LwSciError LwSciEventLoopService_WaitForEvent(
            LwSciEventNotifier *eventNotifier,
            int64_t microseconds);

static LwSciError LwSciEventLoopService_WaitForMultipleEvents(
            LwSciEventNotifier* const * eventNotifierArray,
            size_t eventNotifierCount,
            int64_t microseconds,
            bool *newEventArray);

static LwSciError LwSciEventLoopService_WaitForMultipleEventsExt(
            LwSciEventService* eventService,
            LwSciEventNotifier* const * eventNotifierArray,
            size_t eventNotifierCount,
            int64_t microseconds,
            bool *newEventArray);

static LwSciError LwSciLinuxUserEvent_WaitMultipleEvents(
            LwSciLinuxUserEventLoopService *linuxLoopService,
            LwSciEventNotifier* const * eventNotifierArray,
            size_t eventNotifierCount, int64_t microseconds,
            bool execTillTimeout, bool *newEventArray);


static LwSciError LwSciEventLoopRun(LwSciEventLoop *eventLoop);

static LwSciError LwSciEventService_CreateNativeEventNotifier(
            LwSciEventService *thisEventService,
            LwSciNativeEvent *nativeEvent,
            LwSciEventNotifier **newEventNotifier);
static LwSciError LwSciEventService_CreateLocalEvent(
            LwSciEventService* thisEventService,
            LwSciLocalEvent** newLocalEvent);
static LwSciError LwSciEventService_CreateTimerEvent(
            LwSciEventService* thisEventService,
            LwSciTimerEvent** newTimerEvent);
static void LwSciEventService_Delete(LwSciEventService* thisEventService);

static LwSciError LwSciEventNotifier_SetHandler(
            LwSciEventNotifier *thisEventNotifier,
            void (*callback)(void* cookie),
            void* cookie,
            uint32_t priority);

static void LwSciEventNotifier_Delete(LwSciEventNotifier *thisEventNotifier);

static LwSciError LwSciLocalEvent_Signal(LwSciLocalEvent *thisLocalEvent);
static void LwSciLocalEvent_Delete(LwSciLocalEvent *thisLocalEvent);

static LwSciError LwSciTimerEvent_SetTimer(LwSciTimerEvent *thisTimerEvent,
            int64_t microSeconds);
static LwSciError LwSciTimerEvent_ClearTimer(LwSciTimerEvent *thisTimerEvent);
static void LwSciTimerEvent_Delete(LwSciTimerEvent *thisTimerEvent);

LwSciError LwSciEventLoopServiceCreate(
        size_t maxEventLoops,
        LwSciEventLoopService **newEventLoopService)
{
    LwSciEventLoopService *eventLoopService = NULL;
    LwSciLinuxUserEventLoopService *linuxEventLoopService = NULL;
    LwSciEventService *eventService = NULL;
    LwSciLinuxUserEventService *linuxEventService = NULL;
    LwSciError ret = LwSciError_Success;
    int epollFd = 0, epollFdMin = 0, timerFd = 0, epollRet = 0;
    int err;
    struct epoll_event epollEvent;

    lwscievent_dbg("scievt_dbg:enter");

    if (maxEventLoops != 1) {
        ret = LwSciError_NotSupported;
        lwscievent_err("can't create more than 1 EventLoop: line, ret", ret);
        goto fail;
    }

    if (newEventLoopService == NULL) {
        ret = LwSciError_BadParameter;
        lwscievent_err(
            "scievt_err: newEventLoopService is invalid: line, ret", ret);
        goto fail;
    }

    linuxEventLoopService = calloc(1, sizeof(LwSciLinuxUserEventLoopService));
    CheckNull(linuxEventLoopService, LwSciError_InsufficientMemory);

    linuxEventService = calloc(1, sizeof(LwSciLinuxUserEventService));
    CheckNull(linuxEventService, LwSciError_InsufficientMemory);

    eventService = &linuxEventService->eventService;
    eventLoopService = &linuxEventLoopService->eventLoopService;
    eventLoopService->EventService = *eventService;
    eventLoopService->CreateEventLoop = LwSciEventLoopService_CreateEventLoop;
    eventLoopService->WaitForEvent = LwSciEventLoopService_WaitForEvent;
    eventLoopService->WaitForMultipleEvents =
        LwSciEventLoopService_WaitForMultipleEvents;
    eventLoopService->WaitForMultipleEventsExt =
        LwSciEventLoopService_WaitForMultipleEventsExt;
    *newEventLoopService = eventLoopService;

    eventService->CreateNativeEventNotifier =
        LwSciEventService_CreateNativeEventNotifier;
    eventService->CreateLocalEvent =
        LwSciEventService_CreateLocalEvent;
    eventService->CreateTimerEvent =
        LwSciEventService_CreateTimerEvent;
    eventService->Delete = LwSciEventService_Delete;
    eventLoopService->EventService = *eventService;
    linuxEventLoopService->linuxEventService = linuxEventService;
    linuxEventService->numNotifier = 0;
    linuxEventLoopService->isWaitInProgress = false;
    CheckZero(pthread_mutex_init(&linuxEventLoopService->mutex, NULL));

    timerFd = timerfd_create(CLOCK_MONOTONIC, 0);
    if (timerFd < 0) {
        ret = LwSciError_ResourceError;
        goto fail;
    }
    linuxEventLoopService->timerFd = timerFd;
    lwscievent_dbg("timer create successful. fd=%d\n", timerFd);

    epollFd = epoll_create(LWSCIEVENT_MAX_EVENTNOTIFIER);
    if (epollFd < 0) {
        ret = LwSciError_ResourceError;
        goto fail;
    }
    linuxEventLoopService->epollFd = epollFd;
    lwscievent_dbg("epool create successful. fd=%d\n", epollFd);
    epollFdMin = epoll_create(LWSCIEVENT_MAX_EVENTNOTIFIER);
    if (epollFdMin < 0) {
        ret = LwSciError_ResourceError;
        goto fail;
    }
    linuxEventLoopService->epollFdMin = epollFdMin;
    lwscievent_dbg("epool Min create successful. fd=%d\n", epollFdMin);

    epollEvent.events = EPOLLIN;
    epollEvent.data.ptr = &linuxEventLoopService->timerFd;
    epollRet = epoll_ctl(epollFd, EPOLL_CTL_ADD, timerFd, &epollEvent);
    if (epollRet == -1) {
        ret = LwSciError_ResourceError;
        lwscievent_err(
            "scievt_err: epoll_ctl failed: line, ret", ret);
    }
    epollRet = epoll_ctl(epollFdMin, EPOLL_CTL_ADD, timerFd, &epollEvent);
    if (epollRet == -1) {
        ret = LwSciError_ResourceError;
        lwscievent_err(
            "scievt_err: epoll_ctl failed: line, ret", ret);
    }

fail:
    if (ret != LwSciError_Success) {
        if (epollFd > 0) {
            close(epollFd);
            linuxEventLoopService->epollFd = 0;
        }

        if (epollFdMin > 0) {
            close(epollFdMin);
            linuxEventLoopService->epollFdMin = 0;
        }

        if (timerFd > 0) {
            close(timerFd);
            linuxEventLoopService->timerFd = 0;
        }

        if(linuxEventLoopService != NULL) {
            free(linuxEventLoopService);
        }
    }

    lwscievent_dbg("scievt_dbg:exit");
    return ret;
}

LwSciError LwSciEventLoopService_CreateEventLoop(
            LwSciEventLoopService* eventLoopService,
            LwSciEventLoop **newEventLoop)
{
    LwSciLinuxUserEventLoop *linuxEventLoop = NULL;
    LwSciEventLoop *eventLoop = NULL;
    LwSciError ret = LwSciError_Success;

    lwscievent_dbg("scievt_dbg:enter");
    linuxEventLoop = calloc(1, sizeof(LwSciLinuxUserEventLoop));

    if(linuxEventLoop == NULL) {
        ret = LwSciError_InsufficientMemory;
        goto fail;
    }

    eventLoop = &linuxEventLoop->eventLoop;

    eventLoop->Run = LwSciEventLoopRun;
    eventLoop->Stop = NULL;
    eventLoop->Delete = NULL;
    linuxEventLoop->linuxEventLoopService =
        (LwSciLinuxUserEventLoopService *)eventLoopService;

    *newEventLoop = eventLoop;

fail:
    lwscievent_dbg("scievt_dbg:exit");
    return ret;
}

/* Check if notifier is valid */
static inline LwSciError LwSciLinuxUserEvent_ValidateNotifier(
        LwSciLinuxUserEventNotifier *linuxNotifier)
{
    LwSciError ret = LwSciError_Success;

    lwscievent_dbg("scievt_dbg:enter");

    if (linuxNotifier == NULL) {
        ret = LwSciError_BadParameter;
    }
    else if (linuxNotifier->linuxLoopService == NULL) {
        ret = LwSciError_IlwalidState;
    } else {
        /* validated */
    }

    lwscievent_dbg("scievt_dbg:exit");
    return ret;
}

LwSciError LwSciEventLoopService_WaitForEvent(LwSciEventNotifier *eventNotifier,
            int64_t microseconds)
{
    LwSciLinuxUserEventNotifier *linuxEventNotifier =
        (LwSciLinuxUserEventNotifier *)eventNotifier;
    LwSciNativeEvent *nativeEvent;
    LwSciLinuxUserLocalEvent *linuxLocalEvent;
    LwSciLinuxUserTimerEvent *linuxTimerEvent;
    LwSciLinuxUserEventLoopService *linuxLoopService = NULL;
    fd_set rfds, flushRfds;
    int retVal, selectFd;
    struct timeval selectTimeout;
    LwSciError ret = LwSciError_Success;
    int err;
    uint8_t readData;
    uint64_t readTimerData;

    lwscievent_dbg("scievt_dbg:enter");

    CheckTimeout(microseconds, LwSciError_BadParameter);

    ret = LwSciLinuxUserEvent_ValidateNotifier(linuxEventNotifier);

    if (ret != LwSciError_Success) {
        goto fail;
    }

    nativeEvent = (LwSciNativeEvent* )linuxEventNotifier->linuxNativeEvent;
    linuxLocalEvent =  linuxEventNotifier->linuxLocalEvent;
    linuxTimerEvent =  linuxEventNotifier->linuxTimerEvent;

    if ((nativeEvent == NULL) && (linuxLocalEvent == NULL) &&
        (linuxTimerEvent == NULL)) {
        ret = LwSciError_BadParameter;
        goto fail;
    }

    linuxLoopService = linuxEventNotifier->linuxLoopService;
    CheckZero(pthread_mutex_lock(&linuxLoopService->mutex));

    if (linuxLoopService->isWaitInProgress) {
        CheckZero(pthread_mutex_unlock(&linuxLoopService->mutex));
        ret = LwSciError_NotSupported;
        goto fail;
    }
    linuxLoopService->isWaitInProgress = true;

    CheckZero(pthread_mutex_unlock(&linuxLoopService->mutex));

    if (nativeEvent != NULL) {
        selectFd = nativeEvent->fd;
    } else if (linuxLocalEvent != NULL) {
        selectFd = linuxLocalEvent->pipeFd[0];
    } else
        selectFd = linuxTimerEvent->timerEventFd;

    FD_ZERO(&rfds);
    FD_SET(selectFd, &rfds);
    selectTimeout.tv_sec = microseconds / 1000000;
    selectTimeout.tv_usec = microseconds % 1000000;

    retVal = select(selectFd + 1, &rfds, NULL, NULL, &selectTimeout);
    if (retVal < 0) {
        ret = LwSciIpcErrnoToLwSciErr(errno);
        lwscievent_err(
                "scievt_err: error in select: ", ret);
        if ( ret != LwSciError_InterruptedCall) {
            ret = LwSciError_ResourceError;
        }
        goto fail_update_wip;
    } else if (retVal == 0) {
        ret = LwSciError_Timeout;
        goto fail_update_wip;
    }

    if ((linuxLocalEvent != NULL) || (linuxTimerEvent != NULL)) {
        /* Flush out any data in Local/Timer Event Fd */
        FD_ZERO(&flushRfds);
        FD_SET(selectFd, &flushRfds);
        memset(&selectTimeout, 0, sizeof(struct timeval));

        while(true) {
            retVal = select(selectFd + 1, &flushRfds, NULL, NULL,
                &selectTimeout);

            if(retVal < 1) {
                break;
            }
            if (linuxLocalEvent != NULL) {
                read(selectFd, &readData, sizeof(readData));
            } else {
                read(selectFd, &readTimerData, sizeof(readTimerData));
            }
        }
    }

    if (FD_ISSET(selectFd, &rfds)) {
        if (linuxEventNotifier->callback != NULL)
            linuxEventNotifier->callback(linuxEventNotifier->cookie);
    }

fail_update_wip:
    CheckZero(pthread_mutex_lock(&linuxLoopService->mutex));
    linuxLoopService->isWaitInProgress = false;
    CheckZero(pthread_mutex_unlock(&linuxLoopService->mutex));

fail:
    lwscievent_dbg("scievt_dbg:exit");
    return ret;
}

/* Check input parameters */
static inline LwSciError LwSciEvent_CheckMultipleEventsParams(
    LwSciEventNotifier* const * eventNotifierArray,
    size_t eventNotifierCount,
    int64_t microseconds,
    bool *newEventArray,
    LwSciLinuxUserEventLoopService **loopService)
{
    LwSciLinuxUserEventLoopService *linuxLoopService = NULL;
    LwSciLinuxUserEventNotifier *linuxNotifier;
    LwSciError ret = LwSciError_BadParameter;
    uint32_t inx;

    lwscievent_dbg("scievt_dbg:enter");

    linuxLoopService = *loopService;

    /* check if microseconds is negative execpt infinite wait */
    CheckTimeout(microseconds, LwSciError_BadParameter);
    CheckNull(eventNotifierArray, LwSciError_BadParameter);
    CheckNull(newEventArray, LwSciError_BadParameter);

    /* check eventNotifierCount is in valid range */
    if ((0UL == eventNotifierCount) ||
    (eventNotifierCount > LWSCIEVENT_MAX_EVENTNOTIFIER)) {
        ret = LwSciError_BadParameter;
        goto fail;
    }

    /* check if eventNotifier is invalid */
    for (inx = 0U; inx < eventNotifierCount; inx++) {
        linuxNotifier =
            (LwSciLinuxUserEventNotifier*)(void*)eventNotifierArray[inx];

        /* check if input param is invalid */
        ret = LwSciLinuxUserEvent_ValidateNotifier(linuxNotifier);

        if (ret != LwSciError_Success) {
            goto fail;
        }

       if ((linuxNotifier->linuxNativeEvent == NULL) &&
           (linuxNotifier->linuxLocalEvent == NULL) &&
           (linuxNotifier->linuxTimerEvent == NULL)) {
           ret = LwSciError_BadParameter;
           goto fail;
        }

        if (NULL == linuxLoopService) {
            linuxLoopService = linuxNotifier->linuxLoopService;
        } else {
            if (linuxLoopService != linuxNotifier->linuxLoopService) {
                ret = LwSciError_BadParameter;
                lwscievent_err("error: linuxLoopService is invalid", inx);
                goto fail;
            }
        }
    }

    if ((linuxLoopService != NULL) && (*loopService != linuxLoopService)) {
        *loopService = linuxLoopService;
    }

    ret = LwSciError_Success;

fail:
    lwscievent_dbg("scievt_dbg:exit");
    return ret;
}

LwSciError LwSciEventLoopService_WaitForMultipleEventsExt(
            LwSciEventService* eventService,
            LwSciEventNotifier* const * eventNotifierArray,
            size_t eventNotifierCount,
            int64_t microseconds,
            bool *newEventArray)
{
    LwSciError ret = LwSciError_Success;
    LwSciLinuxUserEventLoopService *linuxLoopService = NULL;
    bool execTillTimeout = false;
    int32_t err;
    uint32_t indx;

    lwscievent_dbg("scievt_dbg:enter");

    if (eventService == NULL) {
        ret = LwSciError_BadParameter;
        goto fail;
    }

    linuxLoopService = (LwSciLinuxUserEventLoopService *)eventService;

    if (eventNotifierArray == NULL) {
        if ((eventNotifierCount != 0) || (newEventArray != NULL)) {
            ret = LwSciError_BadParameter;
            goto fail;
        }
        execTillTimeout = true;
        eventNotifierCount = linuxLoopService->linuxEventService->numNotifier;
        /* check if microseconds is negative execpt infinite wait */
        CheckTimeout(microseconds, LwSciError_BadParameter);
    } else {

        ret = LwSciEvent_CheckMultipleEventsParams(eventNotifierArray,
            eventNotifierCount, microseconds, newEventArray,
            &linuxLoopService);

        if (ret != LwSciError_Success) {
            lwscievent_err(
            "scievt_err: CheckMultipleEventsParams failed: line, ret", ret);
            goto fail;
        }
    }

    if (newEventArray != NULL) {

        /* initialize newEventArray with false */
        for (indx = 0; indx < eventNotifierCount; indx++) {
            newEventArray[indx] = false;
        }
    }

    CheckZero(pthread_mutex_lock(&linuxLoopService->mutex));

    if (linuxLoopService->isWaitInProgress) {
        ret = LwSciError_NotSupported;
        goto fail_locked;
    }

    linuxLoopService->isWaitInProgress = true;

    CheckZero(pthread_mutex_unlock(&linuxLoopService->mutex));

    ret = LwSciLinuxUserEvent_WaitMultipleEvents(linuxLoopService,
            eventNotifierArray, eventNotifierCount, microseconds,
            execTillTimeout, newEventArray);

    CheckZero(pthread_mutex_lock(&linuxLoopService->mutex));
    linuxLoopService->isWaitInProgress = false;
    CheckZero(pthread_mutex_unlock(&linuxLoopService->mutex));

fail_locked:
    err = pthread_mutex_unlock(&linuxLoopService->mutex);
    if (err != 0 && ret == LwSciError_Success) {
        ret = LwSciError_IlwalidState;
        lwscievent_err(
        "scievt_err: pthread_mutex_unlock failed: line, ret", ret);
    }

fail:
    lwscievent_dbg("scievt_dbg:exit");
    return ret;
}

static LwSciError LwSciEvent_AddNotifierToWaitList(
    LwSciLinuxUserEventLoopService *linuxLoopService,
    LwSciEventNotifier* const * eventNotifierArray,
    size_t eventNotifierCount)
{
    uint32_t i = 0;
    struct epoll_event epollEvent;
    struct LwSciLinuxUserEventNotifier *linuxNotifier;
    int notiFd = 0;
    LwSciError ret = LwSciError_Success;
    struct LwSciLinuxUserLocalEvent *linuxLocalEvent;
    struct LwSciLinuxUserTimerEvent *linuxTimerEvent;
    struct LwSciNativeEvent *nativeEvent;
    int64_t epollRet;

    epollEvent.events = EPOLLIN;
    for (i = 0; i < eventNotifierCount; i++) {

        epollEvent.data.ptr = eventNotifierArray[i];
        linuxNotifier = (struct LwSciLinuxUserEventNotifier *)
                        eventNotifierArray[i];
        linuxLocalEvent = linuxNotifier->linuxLocalEvent;
        linuxTimerEvent = linuxNotifier->linuxTimerEvent;
        nativeEvent = (LwSciNativeEvent *)linuxNotifier->linuxNativeEvent;
        if (linuxLocalEvent != NULL) {
            /* Local Event use pipe Fd */
            notiFd = linuxLocalEvent->pipeFd[0];
        } else if (linuxTimerEvent != NULL) {
            notiFd = linuxTimerEvent->timerEventFd;
        } else {
            notiFd = nativeEvent->fd;
        }

        epollRet = epoll_ctl(linuxLoopService->epollFdMin, EPOLL_CTL_ADD,
            notiFd, &epollEvent);
        if (epollRet < 0) {
            ret = LwSciError_ResourceError;
            goto fail;
        }
    }

fail:
    return ret;
}

static void LwSciEvent_RemoveNotifierFromWaitList(
    LwSciLinuxUserEventLoopService *linuxLoopService,
    LwSciEventNotifier* const * eventNotifierArray,
    size_t eventNotifierCount)
{
    uint32_t i = 0;
    struct epoll_event epollEvent;
    struct LwSciLinuxUserEventNotifier *linuxNotifier;
    int notiFd = 0;
    struct LwSciLinuxUserLocalEvent *linuxLocalEvent;
    struct LwSciLinuxUserTimerEvent *linuxTimerEvent;
    struct LwSciNativeEvent *nativeEvent;
    int64_t epollRet;

    epollEvent.events = EPOLLIN;

    for (i = 0; i < eventNotifierCount; i++) {
        epollEvent.data.ptr = eventNotifierArray[i];
        linuxNotifier = (struct LwSciLinuxUserEventNotifier *)
                        eventNotifierArray[i];
        linuxLocalEvent = linuxNotifier->linuxLocalEvent;
        linuxTimerEvent = linuxNotifier->linuxTimerEvent;
        nativeEvent = (LwSciNativeEvent *)linuxNotifier->linuxNativeEvent;
        if (linuxLocalEvent != NULL) {
            /* Local Event use pipe Fd */
            notiFd = linuxLocalEvent->pipeFd[0];
        } else if (linuxTimerEvent != NULL) {
            notiFd = linuxTimerEvent->timerEventFd;
        } else {
            notiFd = nativeEvent->fd;
        }

        epollRet = epoll_ctl(linuxLoopService->epollFdMin, EPOLL_CTL_DEL,
            notiFd, &epollEvent);
        if (epollRet < 0) {
            lwscievent_dbg("Error in removing notifier = %p\n",
                epollEvent.data.ptr);
        }
    }

    return;
}

LwSciError LwSciEventLoopService_WaitForMultipleEvents(
            LwSciEventNotifier* const * eventNotifierArray,
            size_t eventNotifierCount,
            int64_t microseconds,
            bool *newEventArray)
{
    uint32_t indx;
    LwSciError ret = LwSciError_Success;
    LwSciLinuxUserEventLoopService *linuxLoopService = NULL;
    int32_t err;

    lwscievent_dbg("scievt_dbg:enter");

    ret = LwSciEvent_CheckMultipleEventsParams(eventNotifierArray,
        eventNotifierCount, microseconds, newEventArray,
        &linuxLoopService);

    if (ret != LwSciError_Success) {
        lwscievent_err(
        "scievt_err: CheckMultipleEventsParams failed: line, ret", ret);
        goto fail;
    }

    /* initialize newEventArray with false */
    for (indx = 0; indx < eventNotifierCount; indx++) {
        newEventArray[indx] = false;
    }

    CheckZero(pthread_mutex_lock(&linuxLoopService->mutex));

    if (linuxLoopService->isWaitInProgress) {
        ret = LwSciError_NotSupported;
        goto fail_locked;
    }

    linuxLoopService->isWaitInProgress = true;

    CheckZero(pthread_mutex_unlock(&linuxLoopService->mutex));


    ret = LwSciLinuxUserEvent_WaitMultipleEvents(linuxLoopService,
            eventNotifierArray, eventNotifierCount, microseconds,
            false, newEventArray);

    CheckZero(pthread_mutex_lock(&linuxLoopService->mutex));
    linuxLoopService->isWaitInProgress = false;
    CheckZero(pthread_mutex_unlock(&linuxLoopService->mutex));

fail_locked:
    err = pthread_mutex_unlock(&linuxLoopService->mutex);
    if (err != 0 && ret == LwSciError_Success) {
        ret = LwSciError_IlwalidState;
        lwscievent_err(
        "scievt_err: pthread_mutex_unlock failed: line, ret", ret);
    }

fail:
    lwscievent_dbg("scievt_dbg:exit");
    return ret;
}

static LwSciError LwSciLinuxUserEvent_WaitMultipleEvents(
        LwSciLinuxUserEventLoopService *linuxLoopService,
        LwSciEventNotifier* const * eventNotifierArray,
        size_t eventNotifierCount,
        int64_t microseconds,
        bool execTillTimeout,
        bool *newEventArray)
{
    struct itimerspec ts;
    LwSciError ret = LwSciError_Success;
    struct epoll_event events[MAX_EPOLL_EVENTS];
    int64_t epollRet, timeout = -1;
    uint32_t inx, einx, ninx;
    bool eventReceived = false, addNotifierToWait = false;;
    uint64_t timerData;
    uint8_t flushData;
    fd_set rfds;
    struct timeval selectTimeout;
    int retVal = 0;
    int *timerFd = &linuxLoopService->timerFd;
    int epollFd = linuxLoopService->epollFd;
    int flushFd;
    struct LwSciLinuxUserEventNotifier *linuxNotifier;
    struct LwSciLinuxUserEventNotifier **linuxNotifierArray;
    struct LwSciLinuxUserLocalEvent *linuxLocalEvent;
    struct LwSciLinuxUserTimerEvent *linuxTimerEvent;
    LwSciEventNotifier* eventNotifier = NULL;

    lwscievent_dbg("scievt_dbg:enter");

    if (eventNotifierCount <
        (size_t)linuxLoopService->linuxEventService->numNotifier) {
        addNotifierToWait = true;
        ret = LwSciEvent_AddNotifierToWaitList(linuxLoopService, eventNotifierArray,
            eventNotifierCount);
        if ( ret != LwSciError_Success) {
            lwscievent_err("AddNotifierToWaitList Failed\n", ret);
            goto fail;
        }
        lwscievent_dbg("AddNotifierToWaitList Passed\n");
        epollFd = linuxLoopService->epollFdMin;
    }

    /* Flush out any data in timerFd*/
    FD_ZERO(&rfds);
    FD_SET(*timerFd, &rfds);
    memset(&selectTimeout, 0, sizeof(struct timeval));

    while(true) {
        retVal = select(*timerFd + 1, &rfds, NULL, NULL, &selectTimeout);

        if(retVal < 1) {
            break;
        }
        read(*timerFd, &timerData, sizeof(timerData));
    }

    if (microseconds > 0) {
        ts.it_interval.tv_sec = 0;
        ts.it_interval.tv_nsec = 0;
        ts.it_value.tv_sec = microseconds / 1000000;
        ts.it_value.tv_nsec = (microseconds % 1000000) * 1000;
        retVal = timerfd_settime(linuxLoopService->timerFd, 0, &ts, NULL);
        if (retVal < 0) {
            ret = LwSciError_ResourceError;
            goto fail;
        }
    } else if (microseconds == 0) {
        timeout = 0;
    }

    linuxNotifierArray =
        linuxLoopService->linuxEventService->linuxNotifier;

    while (true) {
        memset(events, 0, MAX_EPOLL_EVENTS * sizeof(struct epoll_event));
        epollRet = epoll_wait(epollFd, events, MAX_EPOLL_EVENTS, timeout);

        if (epollRet < 0) {
            /* epoll wait failed */
            if (eventReceived == 0) {
                ret = LwSciIpcErrnoToLwSciErr(errno);
                if (ret != LwSciError_InterruptedCall) {
                    ret = LwSciError_ResourceError;
                }
            }
            break;
        } else if ((epollRet == 0) && (execTillTimeout == false))  {
            /* no event oclwred in the last epoll_wait */
            if (eventReceived == 0) {
                /* no event ever oclwred */
                ret = LwSciError_Timeout;
            }
            break;
        }

        if ((epollRet == 1) && (events[0].data.ptr == (void *)timerFd)) {
            /* timeout oclwred while waiting for event */
            if (eventReceived == 0) {
                ret = LwSciError_Timeout;
            }
            break;
        }

        ninx = 0;

        /* scan through the events received and update newEventArray */
        for (inx = 0; inx < eventNotifierCount; inx++) {
            if (eventNotifierArray == NULL) {

                eventNotifier = NULL;
                while(ninx < LWSCIEVENT_MAX_EVENTNOTIFIER) {
                    if (linuxNotifierArray[ninx] != NULL) {
                        eventNotifier = &linuxNotifierArray[ninx]->eventNotifier;
                        ninx++;
                        break;
                    }
                    ninx++;
                }

                if (eventNotifier == NULL) {
                    break;
                }

            } else {
                eventNotifier = eventNotifierArray[inx];
            }

            for (einx = 0; einx < epollRet; einx++) {
                if(events[einx].data.ptr == eventNotifier) {
                    if(newEventArray != NULL) {
                        newEventArray[inx] = true;
                    }

                    eventReceived = true;
                    linuxNotifier = (struct LwSciLinuxUserEventNotifier *)
                        eventNotifier;
                    linuxLocalEvent = linuxNotifier->linuxLocalEvent;
                    linuxTimerEvent = linuxNotifier->linuxTimerEvent;
                    if (linuxLocalEvent != NULL) {
                        /* Flush out any data in Local Event Fd */
                        flushFd = linuxLocalEvent->pipeFd[0];
                        FD_ZERO(&rfds);
                        FD_SET(flushFd, &rfds);
                        memset(&selectTimeout, 0, sizeof(struct timeval));

                        while(true) {
                            retVal = select(flushFd + 1, &rfds, NULL, NULL,
                                &selectTimeout);

                            if(retVal < 1) {
                                break;
                            }

                            read(flushFd, &flushData, sizeof(flushData));
                        }
                    } /* if (linuxLocalEvent != NULL) */
                    if (linuxTimerEvent != NULL) {
                        /* Flush out any data in Timer Event Fd */
                        flushFd = linuxTimerEvent->timerEventFd;
                        FD_ZERO(&rfds);
                        FD_SET(flushFd, &rfds);
                        memset(&selectTimeout, 0, sizeof(struct timeval));

                        while(true) {
                            retVal = select(flushFd + 1, &rfds, NULL, NULL,
                                &selectTimeout);

                            if(retVal < 1) {
                                break;
                            }

                            read(flushFd, &timerData, sizeof(timerData));
                        }
                    } /* if (linuxTimerEvent != NULL) */

                    if (linuxNotifier->callback != NULL) {
                        linuxNotifier->callback(linuxNotifier->cookie);
                    }

                }
            }
        }

        if ((eventReceived == true) && (epollRet < MAX_EPOLL_EVENTS)) {
            /* valid event(s) received and no more event pending.
             * break if execTillTimeout is not set */
            if (execTillTimeout == false)
                break;
        } else if ((timeout == 0) && (epollRet < MAX_EPOLL_EVENTS)) {
                break;
        } else if (eventReceived == true) {
            /* event received, do epoll wait with 0 timeout */
            timeout = 0;
        } else {
            /* check if timerfd event oclwred. if oclwred, then set timeout
               to 0 and do epoll_wait to fetch events ready without blocking.
               If timerfd event didn't occur then continue with blocking
               epoll_wait
             */
            for (einx = 0; einx < epollRet; einx++) {
                if(events[einx].data.ptr == (void *)timerFd) {
                    timeout = 0;
                    execTillTimeout = false;
                    /* Flush out any data in timerFd*/
                    FD_ZERO(&rfds);
                    FD_SET(*timerFd, &rfds);
                    memset(&selectTimeout, 0, sizeof(struct timeval));

                    while(true) {
                        retVal = select(*timerFd + 1, &rfds, NULL, NULL, &selectTimeout);

                        if(retVal < 1) {
                            break;
                        }
                        read(*timerFd, &timerData, sizeof(timerData));
                    }
                }
            }
        }
    }

fail:
    if (addNotifierToWait == true) {
         LwSciEvent_RemoveNotifierFromWaitList(linuxLoopService,
             eventNotifierArray, eventNotifierCount);
    }
    lwscievent_dbg("scievt_dbg:exit");
    return ret;
}

LwSciError LwSciEventLoopRun(LwSciEventLoop *eventLoop)
{
    LwSciLinuxUserEventLoop *linuxEventLoop =
        (LwSciLinuxUserEventLoop *)eventLoop;
    LwSciLinuxUserEventLoopService *linuxEventLoopService =
        linuxEventLoop->linuxEventLoopService;
    LwSciEventLoopService *eventLoopService =
        &linuxEventLoopService->eventLoopService;
    LwSciLinuxUserEventService *linuxEventService =
        linuxEventLoopService->linuxEventService;
    int numNotifier, err;
    LwSciError ret = LwSciError_Success;

    lwscievent_dbg("scievt_dbg:enter");

    CheckZero(pthread_mutex_lock(&linuxEventLoopService->mutex));
    numNotifier = linuxEventService->numNotifier;
    CheckZero(pthread_mutex_unlock(&linuxEventLoopService->mutex));

    if (numNotifier == 1) {
        LwSciEventNotifier *eventNotifier =
            &(linuxEventService->linuxNotifier[numNotifier - 1]->eventNotifier);
        while(1) {
            eventLoopService->WaitForEvent(eventNotifier, 0);
        }

    } else {
    // do MultipleWaitForEvent
    }

fail:
    lwscievent_dbg("scievt_dbg:exit");
    return ret;
}


static LwSciError AddNotifierToEventServiceList(
    LwSciLinuxUserEventService *linuxEventService,
    LwSciLinuxUserEventNotifier *linuxEventNotifier)
{
    int numNotifier, i;
    LwSciError ret = LwSciError_InsufficientMemory;

    numNotifier = linuxEventService->numNotifier;
    if (numNotifier >= (int)LWSCIEVENT_MAX_EVENTNOTIFIER)  {
        goto fail;
    }

    for (i = 0; i < (int)LWSCIEVENT_MAX_EVENTNOTIFIER; i++) {
        if (linuxEventService->linuxNotifier[numNotifier] == NULL) {
            linuxEventService->linuxNotifier[numNotifier] = linuxEventNotifier;
            linuxEventService->numNotifier++;
            ret = LwSciError_Success;
            break;
        }
    }

fail:
    return ret;
}


static void RemoveNotifierFromEventServiceList(
    LwSciLinuxUserEventService *linuxEventService,
    LwSciLinuxUserEventNotifier *linuxEventNotifier)
{
    int numNotifier, i;

    numNotifier = linuxEventService->numNotifier;
    if (numNotifier <= 0)  {
        return;
    }

    for (i = 0; i < (int)LWSCIEVENT_MAX_EVENTNOTIFIER; i++) {
        if (linuxEventService->linuxNotifier[numNotifier] == linuxEventNotifier) {
            linuxEventService->linuxNotifier[numNotifier] = NULL;
            linuxEventService->numNotifier--;
            break;
        }
    }

    return;
}


LwSciError LwSciEventService_CreateNativeEventNotifier(
            LwSciEventService *thisEventService,
            LwSciNativeEvent *nativeEvent,
            LwSciEventNotifier **newEventNotifier)
{
    LwSciLinuxUserEventLoopService *linuxEventLoopService =
        (LwSciLinuxUserEventLoopService *)thisEventService;
    LwSciLinuxUserEventService *linuxEventService =
        linuxEventLoopService->linuxEventService;
    LwSciLinuxUserEventNotifier *linuxEventNotifier = NULL;
    LwSciEventNotifier *eventNotifier = NULL;
    int epollRet, err;
    struct epoll_event epollEvent;
    LwSciError ret = LwSciError_Success;

    lwscievent_dbg("scievt_dbg:enter");

    /* check if input param is invalid */
    CheckNull(thisEventService, LwSciError_BadParameter);
    CheckNull(nativeEvent, LwSciError_BadParameter);
    CheckNull(newEventNotifier, LwSciError_BadParameter);

    linuxEventNotifier = calloc(1, sizeof(LwSciLinuxUserEventNotifier));
    CheckNull(linuxEventNotifier, LwSciError_InsufficientMemory);
    eventNotifier = &linuxEventNotifier->eventNotifier;
    eventNotifier->SetHandler = LwSciEventNotifier_SetHandler;
    eventNotifier->Delete = LwSciEventNotifier_Delete;
    linuxEventNotifier->linuxNativeEvent =
        (LwSciLinuxUserNativeEvent *)nativeEvent;
    linuxEventNotifier->linuxLoopService = linuxEventLoopService;
    linuxEventNotifier->callback = NULL;

    epollEvent.events = EPOLLIN;
    epollEvent.data.ptr = linuxEventNotifier;
    epollRet = epoll_ctl(linuxEventLoopService->epollFd, EPOLL_CTL_ADD,
        nativeEvent->fd, &epollEvent);
    if (epollRet < 0) {
        ret = LwSciError_ResourceError;
        goto fail;
    }

    CheckZero(pthread_mutex_lock(&linuxEventLoopService->mutex));
    ret = AddNotifierToEventServiceList(linuxEventService, linuxEventNotifier);
    if (ret != LwSciError_Success) {
        CheckZeroLogPrint(pthread_mutex_unlock(&linuxEventLoopService->mutex));
        ret = LwSciError_InsufficientMemory;
        lwscievent_err(
                "scievt_err: max notifier count: line, ret", ret);
        goto fail;
    }

    CheckZero(pthread_mutex_unlock(&linuxEventLoopService->mutex));

    *newEventNotifier = eventNotifier;

fail:
    if(ret != LwSciError_Success) {
        if (linuxEventNotifier != NULL) {
            free(linuxEventNotifier);
        }
    }

    lwscievent_dbg("scievt_dbg:exit");
    return ret;
}

LwSciError LwSciEventService_CreateLocalEvent(
        LwSciEventService *thisEventService,
        LwSciLocalEvent **newLocalEvent)
{
    struct LwSciLinuxUserEventLoopService *linuxLoopService =
        (struct LwSciLinuxUserEventLoopService *)(void *)thisEventService;
    struct LwSciLinuxUserEventNotifier *linuxNotifier = NULL;
    struct LwSciLinuxUserLocalEvent *linuxLocalEvent = NULL;
    struct LwSciLocalEvent *localEvent = NULL;
    struct LwSciLinuxUserEventService *linuxEventService = NULL;
    int retVal = 0;
    LwSciError ret = LwSciError_BadParameter;
    int32_t err; /* used in CheckZero */
    struct epoll_event epollEvent;

    lwscievent_dbg("scievt_dbg:enter");

    /* check if input param is invalid */
    CheckNull(thisEventService, LwSciError_BadParameter);
    CheckNull(newLocalEvent, LwSciError_BadParameter);

    linuxLocalEvent = calloc(1UL, sizeof(struct LwSciLinuxUserLocalEvent));
    CheckNull(linuxLocalEvent, LwSciError_InsufficientMemory);

    linuxNotifier = calloc(1UL, sizeof(struct LwSciLinuxUserEventNotifier));
    CheckNull(linuxNotifier, LwSciError_InsufficientMemory);

    linuxNotifier->eventNotifier.SetHandler = &LwSciEventNotifier_SetHandler;
    linuxNotifier->eventNotifier.Delete = &LwSciEventNotifier_Delete;
    linuxNotifier->linuxLoopService = linuxLoopService;
    linuxNotifier->linuxLocalEvent = linuxLocalEvent;

    linuxLocalEvent->pipeFd[0] = 0;
    linuxLocalEvent->pipeFd[1] = 0;
    retVal = pipe(linuxLocalEvent->pipeFd);
    if(retVal != 0 ) {
        ret = LwSciError_IlwalidState;
        goto fail;
    }

    epollEvent.events = EPOLLIN;
    epollEvent.data.ptr = linuxNotifier;
    retVal = epoll_ctl(linuxLoopService->epollFd, EPOLL_CTL_ADD,
        linuxLocalEvent->pipeFd[0], &epollEvent);
    if (retVal < 0) {
        ret = LwSciError_IlwalidState;
        goto fail;
    }

    localEvent = (LwSciLocalEvent *)(void *)linuxLocalEvent;
    localEvent->eventNotifier = (LwSciEventNotifier *)(void *)linuxNotifier;
    localEvent->Signal = &LwSciLocalEvent_Signal;
    localEvent->Delete = &LwSciLocalEvent_Delete;

    CheckZero(pthread_mutex_lock(&linuxLoopService->mutex));

    linuxEventService = linuxLoopService->linuxEventService;

    ret = AddNotifierToEventServiceList(linuxEventService, linuxNotifier);
    if (ret != LwSciError_Success) {
        CheckZeroLogPrint(pthread_mutex_unlock(&linuxLoopService->mutex));
        ret = LwSciError_InsufficientMemory;
        lwscievent_err(
                "scievt_err: max notifier count: line, ret", ret);
        goto fail;
    }

    CheckZero(pthread_mutex_unlock(&linuxLoopService->mutex));

    ret = LwSciError_Success;

fail:
    /* free memory allocated here if ret is not a success */
    if (ret == LwSciError_Success) {
        *newLocalEvent = localEvent;
    } else {
        if (NULL != linuxNotifier) {
            free(linuxNotifier);
        }

        if (NULL != linuxLocalEvent) {
            if(linuxLocalEvent->pipeFd[0] != 0) {
                close(linuxLocalEvent->pipeFd[0]);
            }

            if(linuxLocalEvent->pipeFd[1] != 0) {
                close(linuxLocalEvent->pipeFd[1]);
            }

            free(linuxLocalEvent);
        }
    }

    lwscievent_dbg("scievt_dbg:exit");
    return ret;
}

LwSciError LwSciEventService_CreateTimerEvent(
        LwSciEventService *thisEventService,
        LwSciTimerEvent **newTimerEvent)
{
    struct LwSciLinuxUserEventLoopService *linuxLoopService =
        (struct LwSciLinuxUserEventLoopService *)(void *)thisEventService;
    struct LwSciLinuxUserEventNotifier *linuxNotifier = NULL;
    struct LwSciLinuxUserTimerEvent *linuxTimerEvent = NULL;
    struct LwSciTimerEvent *timerEvent = NULL;
    struct LwSciLinuxUserEventService *linuxEventService = NULL;
    int retVal = 0;
    LwSciError ret = LwSciError_BadParameter;
    int32_t err; /* used in CheckZero */
    struct epoll_event epollEvent;

    lwscievent_dbg("scievt_dbg:enter");

    /* check if input param is invalid */
    CheckNull(thisEventService, LwSciError_BadParameter);
    CheckNull(newTimerEvent, LwSciError_BadParameter);

    linuxTimerEvent = calloc(1UL, sizeof(struct LwSciLinuxUserTimerEvent));
    CheckNull(linuxTimerEvent, LwSciError_InsufficientMemory);

    linuxNotifier = calloc(1UL, sizeof(struct LwSciLinuxUserEventNotifier));
    CheckNull(linuxNotifier, LwSciError_InsufficientMemory);

    linuxNotifier->eventNotifier.SetHandler = &LwSciEventNotifier_SetHandler;
    linuxNotifier->eventNotifier.Delete = &LwSciEventNotifier_Delete;
    linuxNotifier->linuxLoopService = linuxLoopService;
    linuxNotifier->linuxTimerEvent = linuxTimerEvent;

    linuxTimerEvent->timerEventFd = timerfd_create(CLOCK_MONOTONIC, 0);
    if (linuxTimerEvent->timerEventFd < 0) {
        ret = LwSciError_IlwalidState;
        goto fail;
    }

    epollEvent.events = EPOLLIN;
    epollEvent.data.ptr = linuxNotifier;
    retVal = epoll_ctl(linuxLoopService->epollFd, EPOLL_CTL_ADD,
        linuxTimerEvent->timerEventFd, &epollEvent);
    if (retVal < 0) {
        ret = LwSciError_IlwalidState;
        goto fail;
    }

    timerEvent = (LwSciTimerEvent *)(void *)linuxTimerEvent;
    timerEvent->eventNotifier = (LwSciEventNotifier *)(void *)linuxNotifier;
    timerEvent->SetTimer = &LwSciTimerEvent_SetTimer;
    timerEvent->ClearTimer = &LwSciTimerEvent_ClearTimer;
    timerEvent->Delete = &LwSciTimerEvent_Delete;

    CheckZero(pthread_mutex_lock(&linuxLoopService->mutex));

    linuxEventService = linuxLoopService->linuxEventService;

    ret = AddNotifierToEventServiceList(linuxEventService, linuxNotifier);
    if (ret != LwSciError_Success) {
        CheckZeroLogPrint(pthread_mutex_unlock(&linuxLoopService->mutex));
        ret = LwSciError_InsufficientMemory;
        lwscievent_err(
                "scievt_err: max notifier count: line, ret", ret);
        goto fail;
    }

    CheckZero(pthread_mutex_unlock(&linuxLoopService->mutex));

    ret = LwSciError_Success;

fail:
    /* free memory allocated here if ret is not a success */
    if (ret == LwSciError_Success) {
        *newTimerEvent = timerEvent;
    } else {
        if (NULL != linuxNotifier) {
            free(linuxNotifier);
        }

        if (NULL != linuxTimerEvent) {
            if(linuxTimerEvent->timerEventFd != 0) {
                close(linuxTimerEvent->timerEventFd);
            }

            free(linuxTimerEvent);
        }
    }

    lwscievent_dbg("scievt_dbg:exit");
    return ret;
}

void LwSciEventService_Delete(LwSciEventService* thisEventService)
{
    LwSciLinuxUserEventLoopService *linuxLoopService =
        (LwSciLinuxUserEventLoopService *)(void *)thisEventService;
    uint32_t refCount;
    int retVal;

    lwscievent_dbg("scievt_dbg:enter");

    /* check if input param is invalid */
    if(linuxLoopService == NULL) {
        lwscievent_err("error: line, ret", LwSciError_BadParameter);
        goto fail;
    }

    retVal = pthread_mutex_lock(&linuxLoopService->mutex);
    if (retVal != 0) {
        goto fail;
    }

    refCount = linuxLoopService->linuxEventService->numNotifier;

    retVal = pthread_mutex_unlock(&linuxLoopService->mutex);
    if (retVal != 0) {
        goto fail;
    }

    if (refCount == 0U) {
        close(linuxLoopService->epollFd);
        close(linuxLoopService->epollFdMin);
        close(linuxLoopService->timerFd);

        free(linuxLoopService);
    }

fail:
    lwscievent_dbg("scievt_dbg:exit");
    return;
}

LwSciError LwSciEventNotifier_SetHandler(LwSciEventNotifier *thisEventNotifier,
            void (*callback)(void* cookie),
            void* cookie,
            uint32_t priority)
{
    LwSciLinuxUserEventNotifier *linuxEventNotifier =
        (LwSciLinuxUserEventNotifier *)thisEventNotifier;

    lwscievent_dbg("scievt_dbg:enter");

    if ((thisEventNotifier == NULL) || (callback == NULL) || (cookie == NULL)) {
        return LwSciError_BadParameter;
    }

    linuxEventNotifier->callback = callback;
    linuxEventNotifier->cookie = cookie;
    lwscievent_dbg("scievt_dbg:exit");

    return LwSciError_Success;
}

void LwSciEventNotifier_Delete(LwSciEventNotifier *thisEventNotifier)
{
    LwSciLinuxUserEventNotifier *linuxEventNotifier =
        (LwSciLinuxUserEventNotifier *)(void *)thisEventNotifier;
    LwSciLinuxUserEventLoopService *linuxEventLoopService = NULL;
    LwSciLinuxUserEventService *linuxEventService = NULL;
    int epollRet = 0;
    struct epoll_event epollEvent;
    LwSciNativeEvent *nativeEvent = NULL;
    LwSciLinuxUserLocalEvent *linuxLocalEvent = NULL;
    LwSciError ret;
    int retVal = 0;

    lwscievent_dbg("scievt_dbg:enter");

    ret = LwSciLinuxUserEvent_ValidateNotifier(linuxEventNotifier);

    if (ret != LwSciError_Success) {
        goto fail;
    }

    CheckBothNull(linuxEventNotifier->linuxNativeEvent,
        linuxEventNotifier->linuxLocalEvent, LwSciError_BadParameter);

    nativeEvent = (LwSciNativeEvent* )linuxEventNotifier->linuxNativeEvent;
    linuxLocalEvent = linuxEventNotifier->linuxLocalEvent;
    linuxEventLoopService = linuxEventNotifier->linuxLoopService;

    if (nativeEvent != NULL) {
        epollEvent.events = EPOLLIN;
        epollEvent.data.ptr = &nativeEvent->fd; /* dummy pointer */
        epollRet = epoll_ctl(linuxEventLoopService->epollFd, EPOLL_CTL_DEL,
            nativeEvent->fd, &epollEvent);
        if (epollRet < 0) {
            lwscievent_dbg("scievt_dbg:epoll_ctl failed");
        }
    } else if (linuxLocalEvent != NULL) {
        epollEvent.events = EPOLLIN;
        epollEvent.data.ptr = &linuxLocalEvent->pipeFd[0]; /* dummy pointer */
        epollRet = epoll_ctl(linuxEventLoopService->epollFd, EPOLL_CTL_DEL,
            linuxLocalEvent->pipeFd[0], &epollEvent);
        if (epollRet < 0) {
            lwscievent_dbg("scievt_dbg:epoll_ctl failed");
        }
    }

    retVal = pthread_mutex_lock(&linuxEventLoopService->mutex);
    if (retVal != 0) {
        goto fail;
    }

    linuxEventService = linuxEventLoopService->linuxEventService;
    RemoveNotifierFromEventServiceList(linuxEventService, linuxEventNotifier);
    free(linuxEventNotifier);
    pthread_mutex_unlock(&linuxEventLoopService->mutex);

fail:
    lwscievent_dbg("scievt_dbg:exit");
    return;
}

LwSciError LwSciLocalEvent_Signal(LwSciLocalEvent *thisLocalEvent)
{
    LwSciEventNotifier *eventNotifier;
    LwSciLinuxUserLocalEvent *linuxLocalEvent;
    int32_t err;
    LwSciError ret = LwSciError_BadParameter;
    uint8_t data = 0;

    lwscievent_dbg("scievt_dbg:enter");

    /* check if input param is invalid */
    CheckNull(thisLocalEvent, LwSciError_BadParameter);
    eventNotifier = thisLocalEvent->eventNotifier;
    CheckNull(eventNotifier, LwSciError_BadParameter);

    linuxLocalEvent = (LwSciLinuxUserLocalEvent *)(void *)thisLocalEvent;

    err = write(linuxLocalEvent->pipeFd[1], &data, 1);

    if (err < 1) {
        lwscievent_err(
                "scievt_err: signal local event: ret", ret);
        if (err < 0) {
            ret = LwSciError_IlwalidState;
        } else {
            ret = LwSciError_TryItAgain;
        }
    } else {
        ret = LwSciError_Success;
    }

fail:
    lwscievent_dbg("scievt_dbg:exit");
    return ret;
}

void LwSciLocalEvent_Delete(LwSciLocalEvent *thisLocalEvent)
{
    LwSciLinuxUserLocalEvent *linuxLocalEvent;

    lwscievent_dbg("scievt_dbg:enter");

    /* check if input param is invalid */
    if(thisLocalEvent== NULL) {
        goto fail;
    };

    linuxLocalEvent = (LwSciLinuxUserLocalEvent *)thisLocalEvent;

    if (linuxLocalEvent->pipeFd[0] != 0) {
        close(linuxLocalEvent->pipeFd[0]);
    }

    if (linuxLocalEvent->pipeFd[1] != 0) {
        close(linuxLocalEvent->pipeFd[1]);
    }

    free(thisLocalEvent);

fail:
    lwscievent_dbg("scievt_dbg:exit");
    return;
}

LwSciError LwSciTimerEvent_SetTimer(LwSciTimerEvent *thisTimerEvent,
    int64_t microSeconds)
{
    LwSciEventNotifier *eventNotifier;
    LwSciLinuxUserTimerEvent *linuxTimerEvent;
    LwSciError ret = LwSciError_BadParameter;
    struct itimerspec ts;
    int retVal = 0;

    lwscievent_dbg("scievt_dbg:enter");

    /* check if input param is invalid */
    CheckNull(thisTimerEvent, LwSciError_BadParameter);
    eventNotifier = thisTimerEvent->eventNotifier;
    CheckNull(eventNotifier, LwSciError_BadParameter);

    linuxTimerEvent = (LwSciLinuxUserTimerEvent *)(void *)thisTimerEvent;

    if (microSeconds >= 0) {
        ts.it_interval.tv_sec = 0;
        ts.it_interval.tv_nsec = 0;
        ts.it_value.tv_sec = microSeconds / 1000000;
        ts.it_value.tv_nsec = (microSeconds % 1000000) * 1000;
    } else {
        ret = LwSciError_BadParameter;
        goto fail;
    }

    retVal = timerfd_settime(linuxTimerEvent->timerEventFd, 0, &ts, NULL);
    if (retVal < 0) {
        lwscievent_err(
                "scievt_err: timer event set timer: ret", retVal);
        ret = LwSciError_IlwalidState;
    } else {
        ret = LwSciError_Success;
    }

fail:
    lwscievent_dbg("scievt_dbg:exit");
    return ret;
}

LwSciError LwSciTimerEvent_ClearTimer(LwSciTimerEvent *thisTimerEvent)
{
    LwSciEventNotifier *eventNotifier;
    LwSciLinuxUserTimerEvent *linuxTimerEvent;
    LwSciError ret = LwSciError_BadParameter;
    struct itimerspec ts;
    int retVal = 0;

    lwscievent_dbg("scievt_dbg:enter");

    /* check if input param is invalid */
    CheckNull(thisTimerEvent, LwSciError_BadParameter);
    eventNotifier = thisTimerEvent->eventNotifier;
    CheckNull(eventNotifier, LwSciError_BadParameter);

    linuxTimerEvent = (LwSciLinuxUserTimerEvent *)(void *)thisTimerEvent;

    ts.it_interval.tv_sec = 0;
    ts.it_interval.tv_nsec = 0;
    ts.it_value.tv_sec = 0;
    ts.it_value.tv_nsec = 0;

    retVal = timerfd_settime(linuxTimerEvent->timerEventFd, 0, &ts, NULL);
    if (retVal < 0) {
        lwscievent_err(
                "scievt_err: timer event cancel timer: ret", retVal);
        ret = LwSciError_IlwalidState;
    } else {
        ret = LwSciError_Success;
    }

fail:
    lwscievent_dbg("scievt_dbg:exit");
    return ret;
}

void LwSciTimerEvent_Delete(LwSciTimerEvent *thisTimerEvent)
{
    LwSciLinuxUserTimerEvent *linuxTimerEvent;

    lwscievent_dbg("scievt_dbg:enter");

    /* check if input param is invalid */
    if(thisTimerEvent == NULL) {
        goto fail;
    };

    linuxTimerEvent = (LwSciLinuxUserTimerEvent *)thisTimerEvent;

    if (linuxTimerEvent->timerEventFd != 0) {
        close(linuxTimerEvent->timerEventFd);
    }

    free(thisTimerEvent);

fail:
    lwscievent_dbg("scievt_dbg:exit");
    return;
}

