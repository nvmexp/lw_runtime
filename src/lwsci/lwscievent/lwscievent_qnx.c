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
#include <pthread.h>
#include <sys/neutrino.h>
#include <sys/syspage.h>

#include <lwsciipc_internal.h>
#include <lwscievent_qnx.h>
#include <lwscievent_internal.h>
#if defined(LW_IS_TRACER_ENABLED)
#include <tracer.h>
#endif

#define THOUSAND 1000U
#define MILLION 1000000U
/* (10*365*24*60*60*MILLION) - 10 years in microseconds*/
#define MAX_TIMEOUT_US 315360000000000

/*
 * member funtions of LwSciEventService
 */

static LwSciError LwSciQnxEventService_CreateNativeEventNotifier(
        LwSciEventService *thisEventService,
        LwSciNativeEvent *nativeEvent,
        LwSciEventNotifier **newEventNotifier);

static LwSciError LwSciQnxEventService_CreateLocalEvent(
        LwSciEventService* thisEventService,
        LwSciLocalEvent** newLocalEvent);

static LwSciError LwSciQnxEventService_CreateTimerEvent(
        LwSciEventService* thisEventService,
        LwSciTimerEvent** newTimerEvent);

static void LwSciQnxEventService_Delete(LwSciEventService* thisEventService);

/*
 * member funtions of LwSciEventLoopService
 */

static LwSciError LwSciQnxEventLoopService_CreateEventLoop(
        LwSciEventLoopService* eventLoopService,
        LwSciEventLoop** eventLoop);

static LwSciError LwSciQnxEventLoopService_WaitForEvent(
        LwSciEventNotifier* eventNotifier,
        int64_t microseconds);

static LwSciError LwSciQnxEventLoopService_WaitForMultipleEvents(
        LwSciEventNotifier* const * eventNotifierArray,
        size_t eventNotifierCount,
        int64_t microseconds,
        bool* newEventArray);

/*
 * member functions of LwSciEventNotifier
 */

static LwSciError LwSciQnxEventNotifier_SetHandler(
        LwSciEventNotifier *thisEventNotifier,
        void (*callback)(void* cookie),
        void* cookie,
        uint32_t priority);

static void LwSciQnxEventNotifier_Delete(
        LwSciEventNotifier *thisEventNotifier);

/*
 * member functions of LwSciLocalEvent
 */

static void LwSciQnxLocalEvent_Delete(
        LwSciLocalEvent *thisLocalEvent);

static LwSciError LwSciQnxLocalEvent_Signal(
        LwSciLocalEvent *thisLocalEvent);

/*
 * Helper functions to handle timeout
 */

/* Colwert clock cycles into usec */
static inline uint64_t LwSciQnxEvent_Colwert2Usec(uint64_t counter)
{
    /* Static so that the value is cached */
    static uint64_t cycles_per_usec = 0ULL;
    static uint64_t usec;

    if (cycles_per_usec == 0U) {
        cycles_per_usec = (SYSPAGE_ENTRY(qtime)->cycles_per_sec) / MILLION;
    }

    usec = counter / cycles_per_usec;
    return usec;
}

/* Callwlate remaining time before timeout */
static inline void LwSciQnxEvent_CalTimeout(int64_t microseconds,
        uint64_t waitCycles, uint64_t *waitTime)
{
    bool flag;
    uint64_t elapsed = 0ULL;
    uint64_t cycles = 0ULL;
    uint64_t wcycles = 0ULL;

    if (LW_SCI_EVENT_INFINITE_WAIT != microseconds) {
        cycles = ClockCycles();
        if (waitCycles < cycles) {
            wcycles = cycles - waitCycles;
        } else {
            wcycles = cycles + (UINT64_MAX - waitCycles);
        }

        elapsed = LwSciQnxEvent_Colwert2Usec(wcycles);
        flag = SubU64(*waitTime, elapsed, waitTime);
        if (false == flag) {
            *waitTime = 0ULL; /* set to '0' if it already passed by */
        }
    }
}

/* Update eventArray */
static inline void LwSciEvent_UpdateEventNotified(
        LwSciQnxEventNotifier *rcvdNotifier,
        size_t count,
        LwSciQnxEventNotifier* const * eventNotifierArray,
        bool *eventArray, bool *validEvent, uint64_t *wTime)
{
    uint32_t inx;

    if (rcvdNotifier != NULL) {
        rcvdNotifier->isPending = true;
        for (inx = 0; inx < count; inx++) {
            if (eventNotifierArray[inx] == rcvdNotifier) {
                /* Event has been notified at this index */
                eventArray[inx] = true;
                *validEvent = true;
                /*
                 * Any event being waited is notified
                 * Time to switch to checking pending event
                 */
                *wTime = 0ULL;
            }
        }
    }
    else {
        lwscievent_err(
            "scievt_err: pulse with NULL notifier received: ",
            0);
    }
}

/* Clean up pending flag of notifier */
static inline void LwSciEvent_ClearEventNotified(
        LwSciQnxEventNotifier* const * eventNotifierArray,
        size_t count)
{
    uint32_t inx;

    for (inx = 0U; inx < count; inx++) {
        if (eventNotifierArray[inx]->isPending == true) {
            eventNotifierArray[inx]->isPending = false;
        }
    }
}

/* Check if notifier is pending and update eventArray */
static inline bool LwSciEvent_CheckEventPending(
        LwSciQnxEventNotifier* const * eventNotifierArray,
        size_t count, bool *eventArray)
{
    bool pending = false;
    uint32_t inx;

    for (inx = 0U; inx < count; inx++) {
        if (eventNotifierArray[inx]->isPending == true) {
            eventArray[inx] = true;
            pending = true;
        }
    }
    return pending;
}

/* Check if notifier is valid */
static inline LwSciError LwSciQnxEvent_ValidateNotifier(
        LwSciQnxEventNotifier *qnxNotifier)
{
    LwSciError ret = LwSciError_BadParameter;

    if (qnxNotifier == NULL) {
        ret = LwSciError_BadParameter;
    }
    else if ((qnxNotifier->qnxLoopService == NULL) ||
        (qnxNotifier->magic != LWSCIEVENT_MAGIC)) {
        ret = LwSciError_BadParameter;
    } else {
        /* validated */
        ret = LwSciError_Success;
    }

    return ret;
}

/* Check input parameters */
static inline LwSciError LwSciEvent_CheckMultipleEventsParams(
            LwSciEventNotifier* const * eventNotifierArray,
            size_t eventNotifierCount,
            int64_t microseconds,
            bool *newEventArray,
            LwSciQnxEventLoopService **loopService)
{
    LwSciQnxEventLoopService *qnxLoopService = NULL;
    LwSciQnxEventNotifier *qnxNotifier;
    LwSciError ret = LwSciError_BadParameter;
    uint32_t inx;

    /* check if microseconds is negative execpt infinite wait */
    CheckTimeoutTo(microseconds, LwSciError_BadParameter, fail);
    CheckNull(eventNotifierArray, LwSciError_BadParameter);
    CheckNull(newEventArray, LwSciError_BadParameter);

    /* check eventNotifierCount is in valid range */
    if ((0UL == eventNotifierCount) ||
        (eventNotifierCount > LWSCIEVENT_MAX_EVENTNOTIFIER)) {
        ret = LwSciError_BadParameter;
        goto fail;
    }

    /* check if eventNotifier is invalid */
    LWOS_COV_WHITELIST(deviate, LWOS_MISRA(Rule, 15_4), "<QNXBSP>:<lwpu>:<1>:<Bug 200736827>")
    for (inx = 0U; inx < eventNotifierCount; inx++) {
        qnxNotifier = (LwSciQnxEventNotifier*)(void*)eventNotifierArray[inx];
        /* check if input param is invalid */
        ret = LwSciQnxEvent_ValidateNotifier(qnxNotifier);
        if (ret != LwSciError_Success) {
            lwscievent_err("error: invalid notifier line, ret", ret);
            goto fail;
        }

        CheckBothNull(qnxNotifier->qnxEvent,
            qnxNotifier->qnxLocalEvent, LwSciError_BadParameter);

        if (NULL == qnxLoopService) {
            qnxLoopService = qnxNotifier->qnxLoopService;
        }
        else {
            if (qnxLoopService != qnxNotifier->qnxLoopService) {
                ret = LwSciError_BadParameter;
                lwscievent_err("error: qnxLoopService is invalid", inx);
                goto fail;
            }
        }
    }
    *loopService = qnxLoopService;
    ret = LwSciError_Success;

fail:
    return ret;
}

/* Unmask interrupt if notifier is associated with native event for inter-VM */
static inline LwSciError LwSciQnxEvent_UnmaskInterrupt(
    LwSciQnxEventNotifier *qnxNotifier)
{
    LwSciNativeEvent *nativeEvent;
    /*
     * Treat the case that does not run UnmaskInterrupt as a success
     * because the case falls into either local event or intra-VM and both have
     * nothing to do with interrupt.
     * Having LwSciError_Success as initial retrun value is more code efficient
     * in this case.
     */
    LwSciError ret = LwSciError_Success;

    if ((NULL != qnxNotifier) &&
        (NULL != qnxNotifier->qnxEvent)) { /* Native event */
        nativeEvent = &qnxNotifier->qnxEvent->nativeEvent;
        if ((NULL != nativeEvent) &&
            (NULL != nativeEvent->UnmaskInterrupt)) {
            /* Inter-VM backend interrupt */
            ret = nativeEvent->UnmaskInterrupt(nativeEvent);
        }
    }
    return ret;
}

/**
 * This function waits for any event in evnetNotifierArray to be notified with
 * given microseconds. When any event is notified, it updates newEventArray if
 * the event is in eventNotifierArray and drains out pending events in queue
 * without waiting further. It also updates notifier's IsPending flag indicating
 * the new event will be handled in next call if the notifier is not in
 * eventNotifierArray.
 *
 * @param[in]  qnxLoopService     Pointer to LwSciQnxEventLoopService object
 * @param[in]  eventNotifierArray Array of LwSciQnxEventNotifier object pointer
 * @param[in]  eventNotifierCount Array size of eventNotifierArray
 * @param[in]  microseconds       64-bit integer timeout in microseconds.
 *                                set 0 for infinite timeout
 * @param[out] newEventArray      Array of boolean indicating the corresponding
 *                                EventNotifier in evnetNotifierArray gets event
 *
 * @return ::LwSciError, the completion code of operations:
 * - ::LwSciError_Success         Indicates a successful operation.
 * - ::LwSciError_Timeout         Indicates timeout oclwrence.
 * - ::LwSciError_NoSuchProcess   Indicates channel not exist.
 * - ::LwSciError_InterruptedCall Indicates an interrupt oclwrred.
 * - ::LwSciError_IlwalidState    Indicates an invalid operation state.
 */
static LwSciError LwSciQnxEvent_WaitMultipleEvents(
        LwSciQnxEventLoopService *qnxLoopService,
        LwSciQnxEventNotifier* const * eventNotifierArray,
        size_t eventNotifierCount,
        int64_t microseconds,
        bool *newEventArray)
{
    LwSciQnxEventNotifier *rcvdNotifier = NULL;
    bool firstValidEvent = false;
    int32_t err;
    LwSciError ret = LwSciError_IlwalidState;
    struct _pulse pulse;
    int32_t result;
    uint64_t waitTime = 0ULL; /* usec */
    uint64_t timeout = 0ULL; /* nsec */
    uint64_t waitCycles = 0ULL;

    lwscievent_dbg("WaitMultipleEvents: enter");

    if (microseconds > 0) {
        waitTime = (uint64_t)microseconds;
    }

    LWOS_COV_WHITELIST(deviate, LWOS_MISRA(Rule, 15_4), "<QNXBSP>:<lwpu>:<1>:<Bug 200736827>")
    while(true) {
        /* arm timer */
        if (firstValidEvent ||
            (LW_SCI_EVENT_INFINITE_WAIT != microseconds)) {
            /* assert "waitTime <= MAX_TIMEOUT_US" in CheckTimeoutTo */
            MultU64WithExit(waitTime, THOUSAND, &timeout);
            waitCycles = ClockCycles();

            LWOS_COV_WHITELIST(deviate, LWOS_MISRA(Rule, 10_1), "<QNXBSP>:<qnx_asil_header>:<2>:<TID-257>") LWOS_COV_WHITELIST(deviate, LWOS_MISRA(Rule, 15_4), "<QNXBSP>:<lwpu>:<1>:<Bug 200736827>")
            CheckZero(TimerTimeout_r(CLOCK_MONOTONIC, _NTO_TIMEOUT_RECEIVE, NULL, &timeout, NULL));
        }
        result = MsgReceivePulse_r(
            qnxLoopService->waitChid, &pulse, sizeof(pulse), NULL);

        if (0 == result) { /* got pulse */
            if (_PULSE_CODE_MINAVAIL == pulse.code) {
                /* First pulse event is received */
                rcvdNotifier = (LwSciQnxEventNotifier *)pulse.value.sival_ptr;
                /* Unmask interrupt if required */
                CheckSuccess(LwSciQnxEvent_UnmaskInterrupt(rcvdNotifier));
                /* Update newEventArray[] with true */
                LwSciEvent_UpdateEventNotified(rcvdNotifier,
                    eventNotifierCount, eventNotifierArray,
                    newEventArray, &firstValidEvent, &waitTime);
            }  else if (_PULSE_CODE_MINAVAIL < pulse.code) {
                lwscievent_err(
                    "scievt_err: Unexpected pulse received: pulse.code",
                    (uint8_t)pulse.code);
            } else {
                /* System pulse, ignore it as it is not error */
            }
        } else if (-ETIMEDOUT == result ) {
            if (!firstValidEvent) {
                ret = LwSciError_Timeout; /* timeout without valid event */
            } else {
                ret = LwSciError_Success;
            }
            break; /* There is no more pending pulse */
        } else {
            /*
             * EFAULT not happen with "struct pulse" buffer
             * EINTR, ESRCH could happen
             */
            ret = LwSciIpcErrnoToLwSciErr(result);
            lwscievent_err(
                "scievt_err: Signal received: ", ret);
            goto fail;
        }

        /* callwlate remainig timeout */
        LwSciQnxEvent_CalTimeout(microseconds, waitCycles, &waitTime);
    }

fail:
    LwSciEvent_ClearEventNotified(eventNotifierArray, eventNotifierCount);
    lwscievent_dbg("scievt_dbg:exit");

    LWOS_COV_WHITELIST(deviate, LWOS_MISRA(Directive, 4_13), "<QNXBSP>:<lwpu>:<1>:<Bug 200736827>")
    return ret;
}

/**
 * This function creates event notifier which reports the oclwrrence of an event
 * from the OS environment to the event service.
 * To config the event pulse it calls function in nativeEvent with the notifier
 * pointer, which is support function in LwSciIpc library.
 *
 * @param[in]  thisEventService LwSciEventService object pointer created by
 *                              LwSciEventLoopServiceCreate()
 * @param[in]  nativeEvent LwSciNativeEvent object pointer
 * @param[out]  newEventNotifier LwSciEventNotifier object double pointer
 *
 * @return ::LwSciError, the completion code of operations:
 * - ::LwSciError_Success         Indicates a successful operation.
 * - ::LwSciError_InsufficientMemory  Indicates memory is not sufficient.
 * - ::LwSciError_BadParameter    Indicates an invalid input parameters.
 * - ::LwSciError_ResourceError   Indicates not enough system resources.
 * - ::LwSciError_IlwalidState    Indicates an invalid operation state.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt: No
 *   - Signal handler(QNX): No
 *   - Thread: Yes
 * - Is thread safe: Yes
 * - Required Privileges(QNX):: None
 * - API Group
 *   - Initialization: Yes
 *   - Runtime: No
 *   - De-initialization: No
 */
static LwSciError LwSciQnxEventService_CreateNativeEventNotifier(
        LwSciEventService *thisEventService,
        LwSciNativeEvent *nativeEvent,
        LwSciEventNotifier **newEventNotifier)
{
    struct LwSciQnxEventLoopService *qnxLoopService =
        (struct LwSciQnxEventLoopService *)(void *)thisEventService;
    struct LwSciQnxEventNotifier *qnxNotifier = NULL;
    LwSciError ret;
    int32_t err;

    lwscievent_dbg("scievt_dbg:enter");

    /* check if input param is invalid */
    CheckNull(thisEventService, LwSciError_BadParameter);
    CheckNull(nativeEvent, LwSciError_BadParameter);
    CheckNull(newEventNotifier, LwSciError_BadParameter);
    CheckNull(nativeEvent->ConfigurePulseParams, LwSciError_BadParameter);
    if (qnxLoopService->magic != LWSCIEVENT_MAGIC) {
        ret = LwSciError_BadParameter;
        goto fail;
    }

    /* check if memory allocation is success */
    LWOS_COV_WHITELIST(deviate, LWOS_MISRA(Rule, 21_3), "<QNXBSP>:<lwpu>:<1>:<TID-385>")
    qnxNotifier = calloc(1UL, sizeof(*qnxNotifier));
    CheckNull(qnxNotifier, LwSciError_InsufficientMemory);

    qnxNotifier->eventNotifier.SetHandler = &LwSciQnxEventNotifier_SetHandler;
    qnxNotifier->eventNotifier.Delete = &LwSciQnxEventNotifier_Delete;
    qnxNotifier->qnxLoopService = qnxLoopService;
    qnxNotifier->qnxEvent = (LwSciQnxNativeEvent *)(void *)nativeEvent;
    qnxNotifier->isPending = false;

    /*
     * It can return these error codes
     * LwSciError_Success, LwSciError_ResourceError, LwSciError_IlwalidState
     */
    ret = nativeEvent->ConfigurePulseParams(nativeEvent,
        qnxLoopService->waitCoid, SIGEV_PULSE_PRIO_INHERIT,
        _PULSE_CODE_MINAVAIL, qnxNotifier);
    if (ret != LwSciError_Success) {
        /*
         * Error from configurePulseParams
         * - LwSciError_NotInitialized
         * - LwSciError_Busy
         */
        LWOS_COV_WHITELIST(deviate, LWOS_MISRA(Rule, 21_3), "<QNXBSP>:<lwpu>:<1>:<TID-385>")
        goto fail;
    }

    CheckZero(pthread_mutex_lock(&qnxLoopService->mutex));
    /* maximum event notifier count per process: 200U */
    if (qnxLoopService->refCount >= LWSCIEVENT_MAX_EVENTNOTIFIER) {
        CheckZeroLogPrint(pthread_mutex_unlock(&qnxLoopService->mutex));
        ret = LwSciError_InsufficientMemory;
        LWOS_COV_WHITELIST(deviate, LWOS_MISRA(Rule, 21_3), "<QNXBSP>:<lwpu>:<1>:<TID-385>")
        lwscievent_err(
                "scievt_err: max notifier count: line, ret", ret);
        goto fail;
    }
    qnxLoopService->refCount = qnxLoopService->refCount + 1U;
    CheckZero(pthread_mutex_unlock(&qnxLoopService->mutex));

    qnxNotifier->magic = LWSCIEVENT_MAGIC;
    *newEventNotifier = &qnxNotifier->eventNotifier;

    ret = LwSciError_Success;

fail:
    /* free memory allocated here if ret is not a success */
    if (ret != LwSciError_Success) {
        if (NULL != qnxNotifier) {
            LWOS_COV_WHITELIST(deviate, LWOS_MISRA(Rule, 21_3), "<QNXBSP>:<lwpu>:<1>:<TID-385>")
            free(qnxNotifier);
        }
    }

    lwscievent_dbg("scievt_dbg:exit");
    /* qnxNotifier will be freed in LwSciQnxEventNotifier_Delete() */
    return ret;
}


/**
 * This function creates a timer event with an event notifier that reports each
 * that reports each event signaled through it.
 *
 * NOT SUPPORTED
 */
static LwSciError LwSciQnxEventService_CreateTimerEvent(
        LwSciEventService* thisEventService,
        LwSciTimerEvent** newTimerEvent)
{
    (void)thisEventService;
    (void)newTimerEvent;

    lwscievent_dbg("scievt_dbg:enter");
    lwscievent_dbg("scievt_dbg:exit");
    return LwSciError_NotSupported;
}

/**
 * release resources associated with LwSciEventService and LwSciEventService
 * which is created by LwSciEventLoopServiceCreate().
 * note it must be called after releasing notifier and LwSciEventService is
 * no longer required.
 *
 * @param[in]  thisEventService LwSciEventService object pointer created by
 *                              LwSciEventLoopServiceCreate()
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt: No
 *   - Signal handler(QNX): No
 *   - Thread: Yes
 * - Is thread safe: Yes
 * - Required Privileges(QNX):: None
 * - API Group
 *   - Initialization: No
 *   - Runtime: No
 *   - De-initialization: Yes
 */
static void LwSciQnxEventService_Delete(LwSciEventService* thisEventService)
{
    LwSciQnxEventLoopService *qnxLoopService =
        (LwSciQnxEventLoopService *)(void *)thisEventService;
    int32_t err;
    uint32_t newRefCount;
    LwSciError ret;

    lwscievent_dbg("scievt_dbg:enter");

    /* check if input param is invalid */
    CheckNull(qnxLoopService, LwSciError_BadParameter);
    if (qnxLoopService->magic != LWSCIEVENT_MAGIC) {
        lwscievent_err("error: magic number is invalid",
            qnxLoopService->magic);
        goto fail;
    }

    CheckZero(pthread_mutex_lock(&qnxLoopService->mutex));
    if (qnxLoopService->refCount > 0U) {
        qnxLoopService->refCount--;
    }
    newRefCount = qnxLoopService->refCount;
    CheckZero(pthread_mutex_unlock(&qnxLoopService->mutex));

#ifdef LWSCIIPC_DEBUG
    LwOsDebugPrintStrUInt(LWOS_SLOG_CODE_IPC, SLOG2_INFO,
            "scievt_dbg: refcount", newRefCount);
#endif /* LWSCIIPC_DEBUG */

    if (newRefCount == 0U) {
        err = ConnectDetach_r(qnxLoopService->waitCoid);
        if (err != EOK) {
            ret = LwSciIpcErrnoToLwSciErr(err);
            lwscievent_err("error: line, ret", ret);
        }

        err = ChannelDestroy_r(qnxLoopService->waitChid);
        if (err != EOK) {
            ret = LwSciIpcErrnoToLwSciErr(err);
            lwscievent_err("error: line, ret", ret);
        }

        /* free qnxLoopService which is created by
         * LwSciEventLoopServiceCreate()
         */
        LWOS_COV_WHITELIST(deviate, LWOS_MISRA(Rule, 21_3), "<QNXBSP>:<lwpu>:<1>:<TID-385>")
        free(qnxLoopService);
    }

fail:
    lwscievent_dbg("scievt_dbg:exit");
    return;
}

/**
 * This function creates an event loop that can handle events for
 * eventLoopService.
 *
 * NOT SUPPORTED
 */
static LwSciError LwSciQnxEventLoopService_CreateEventLoop(
    LwSciEventLoopService* eventLoopService,
    LwSciEventLoop** eventLoop)
{
    (void)eventLoopService;
    (void)eventLoop;

    lwscievent_dbg("scievt_dbg:enter");
    lwscievent_dbg("scievt_dbg:exit");
    return LwSciError_NotSupported;
}

/**
 * this function waits up to a configurable timeout to receive pulse event
 * which is configured on LwSciQnxEventService_CreateNativeEventNotifier()
 * or LwSciQnxEventService_CreateLocalEvent().
 * eventNotifer must have been created through EventService before calling.
 *
 * @param[in]  eventNotifier LwSciEventNotifier object pointer
 * @param[in]  microseconds 64-bit integer timeout in microsecond,
 *                          set -1 for infinite timeout.
 *
 * @return ::LwSciError, the completion code of operations:
 * - ::LwSciError_Success         Indicates a successful operation.
 * - ::LwSciError_BadParameter    Indicates an invalid input parameters
 * - ::LwSciError_IlwalidState    Indicates an invalid operation state.
 * - ::LwSciError_NotSupported    Indicates not support condition
 * - ::LwSciError_Timeout         Indicates timeout oclwrrence
 * - ::LwSciError_ResourceError   Indicates not enough system resources.
 * - ::LwSciError_InterruptedCall Indicates interrupt oclwrred
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt: No
 *   - Signal handler(QNX): No
 *   - Thread: Yes
 * - Is thread safe: Yes
 * - Required Privileges(QNX):: None
 * - API Group
 *   - Initialization: No
 *   - Runtime: Yes
 *   - De-initialization: No
 */
static LwSciError LwSciQnxEventLoopService_WaitForEvent(
        LwSciEventNotifier* eventNotifier,
        int64_t microseconds)
{
#define SINGLE_EVENT_CNT 1U
    LwSciQnxEventNotifier *qnxNotifier =
        (LwSciQnxEventNotifier *)(void *)eventNotifier;
    LwSciQnxEventLoopService *qnxLoopService = NULL;
    LwSciQnxEventNotifier* qnxNotifierArray[SINGLE_EVENT_CNT] = {qnxNotifier};
    bool eventArray[SINGLE_EVENT_CNT] = {false};
    bool hasPendingEvent = false;
    int64_t msec = microseconds;
    int32_t err;
    LwSciError ret = LwSciError_IlwalidState;

    lwscievent_dbg("scievt_dbg:enter");

    /*
     * check if microseconds is negative execpt infinite wait
     * Or if microseconds is greater than 10 years (MAX TIMEOUT)
     */
    CheckTimeoutTo(msec, LwSciError_BadParameter, fail2);

    /* check if input param and notifier member are valid */
    ret = LwSciQnxEvent_ValidateNotifier(qnxNotifier);
    if (ret != LwSciError_Success) {
        lwscievent_err("error: invalid notifier: line, ret", ret);
        goto fail2;
    }
    qnxLoopService = qnxNotifier->qnxLoopService;

    CheckZeroTo(pthread_mutex_lock(&qnxLoopService->mutex), fail2);
    if (qnxLoopService->isWaitInProgress) {
        ret = LwSciError_NotSupported;
        err = pthread_mutex_unlock(&qnxLoopService->mutex);
        if (err != EOK) {
            lwscievent_err("error: pthread_mutex_unlock:", err);
        }
        goto fail2;
    }

    hasPendingEvent = LwSciEvent_CheckEventPending(qnxNotifierArray,
            SINGLE_EVENT_CNT, eventArray);
    if (hasPendingEvent) {
        /*
         * Any event notified in previous call is pending in notifier array
         * Drain pending other events without waiting for given time.
         */
        msec = 0LL;
    }

    qnxLoopService->isWaitInProgress = true;
    CheckZero(pthread_mutex_unlock(&qnxLoopService->mutex));

    ret = LwSciQnxEvent_WaitMultipleEvents(qnxLoopService,
            qnxNotifierArray, SINGLE_EVENT_CNT, msec, &eventArray[0]);

    if (hasPendingEvent && (ret == LwSciError_Timeout)) {
        /*
         * At least there is one pending event then timeout for other events
         * is treated as a success case
         */
        ret = LwSciError_Success;
    }
    else if (ret == LwSciError_NoSuchProcess) {
        /*
         * Colwert to resource error for clarity (unified error with linux)
         */
        ret = LwSciError_ResourceError;
    }

fail:
    qnxLoopService->isWaitInProgress = false;

fail2:
    lwscievent_dbg("scievt_dbg:exit");
    return ret;
}

/**
 * this function registers or unregisters a handler for a particular
 * event notifier
 *
 * NOT SUPPORTED
 */
static LwSciError LwSciQnxEventNotifier_SetHandler(
        LwSciEventNotifier *thisEventNotifier,
LWOS_COV_WHITELIST(deviate, LWOS_MISRA(Rule, 2_7), "<QNXBSP>:<lwpu>:<1>:<Bug 200736827>")
        void (*callback)(void* cookie),
        void* cookie,
        uint32_t priority)
{
    (void)thisEventNotifier;
    (void)cookie;
    (void)priority;

    lwscievent_dbg("scievt_dbg:enter");
    lwscievent_dbg("scievt_dbg:exit");
    return LwSciError_NotSupported;
}

/**
 * this function releases the EventNotifier and Unregister event handler.
 * it should be called when the EventNotifier is no longer required.
 *
 * @param[in]  thisEventNotifier The event handler to unregister and delete
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt: No
 *   - Signal handler(QNX): No
 *   - Thread: Yes
 * - Is thread safe: Yes
 * - Required Privileges(QNX):: None
 * - API Group
 *   - Initialization: No
 *   - Runtime: No
 *   - De-initialization: Yes
 */
static void LwSciQnxEventNotifier_Delete(LwSciEventNotifier *thisEventNotifier)
{
    LwSciQnxEventNotifier *qnxNotifier =
        (LwSciQnxEventNotifier *)(void *)thisEventNotifier;
    LwSciQnxEventLoopService *qnxLoopService = NULL;
    int32_t err;
    LwSciError ret;
    void *pFunc;

    lwscievent_dbg("scievt_dbg:enter");

    /* TODO: remove event handler */

    /* check if input param is invalid */
    ret = LwSciQnxEvent_ValidateNotifier(qnxNotifier);
    if (ret != LwSciError_Success) {
        lwscievent_err("error: invalid notifier line, ret", ret);
        goto fail;
    }

    CheckBothNull(qnxNotifier->qnxEvent,
        qnxNotifier->qnxLocalEvent, LwSciError_BadParameter);

    qnxLoopService = qnxNotifier->qnxLoopService;

    /* Native event has pulse param to unconfigure */
    if (NULL != qnxNotifier->qnxEvent) {
        pFunc =
            (void *)qnxNotifier->qnxEvent->nativeEvent.UnconfigurePulseParams;
        CheckNull(pFunc, LwSciError_BadParameter);

        qnxNotifier->qnxEvent->nativeEvent.UnconfigurePulseParams(
            &qnxNotifier->qnxEvent->nativeEvent);
    }
    /* free qnxNotifier which is created by
    * LwSciQnxEventService_CreateNativeEventNotifier
    */
    LWOS_COV_WHITELIST(deviate, LWOS_MISRA(Rule, 21_3), "<QNXBSP>:<lwpu>:<1>:<TID-385>")
    free(qnxNotifier);

    /* decrease reference count */
    CheckZero(pthread_mutex_lock(&qnxLoopService->mutex));
    if (qnxLoopService->refCount > 0U) {
        qnxLoopService->refCount--;
    }

    CheckZero(pthread_mutex_unlock(&qnxLoopService->mutex));

fail:
    lwscievent_dbg("scievt_dbg:exit");
}

/**
 * This function creates a new event loop service EventLoopService which is
 * primary instance of event service. Application must call event service
 * functions along with EventLoopService.
 * The number of event loops that can be created in the new event loop service
 * will be limited to at most maxEventLoops.
 *
 * @param[in]  maxEventLoops The number of event loops, it must be 1.
 * @param[out]  newEventLoopService LwSciEventNotifier object double pointer
 *
 * @return ::LwSciError, the completion code of operations:
 * - ::LwSciError_Success         Indicates a successful operation.
 * - ::LwSciError_InsufficientMemory  Indicates memory is not sufficient
 * - ::LwSciError_NotSupported    Indicates not support condition
 * - ::LwSciError_IlwalidState    Indicates an invalid operation state.
 * - ::LwSciError_BadParameter    Indicates an invalid or NULL argument.
 * - ::LwSciError_ResourceError   Indicates not enough system resources.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt: No
 *   - Signal handler(QNX): No
 *   - Thread: Yes
 * - Is thread safe: Yes
 * - Required Privileges(QNX):: None
 * - API Group
 *   - Initialization: Yes
 *   - Runtime: No
 *   - De-initialization: No
 */
LwSciError LwSciEventLoopServiceCreate(
        size_t maxEventLoops,
        LwSciEventLoopService **newEventLoopService)
{
    LwSciQnxEventLoopService *qnxLoopService = NULL;
    int32_t chid;
    int32_t coid;
    int32_t err;
    LwSciError ret = LwSciError_IlwalidState;

    lwscievent_dbg("scievt_dbg:enter");

    if (maxEventLoops != 1U) {
        ret = LwSciError_NotSupported;
        lwscievent_err(
            "scievt_err: Can NOT create more than 1 EventLoop: line, ret", ret);
        goto fail;
    }

    if (newEventLoopService == NULL) {
        ret = LwSciError_BadParameter;
        lwscievent_err(
            "scievt_err: newEventLoopService is invalid: line, ret", ret);
        goto fail;
    }

    /* check memory allocation is success */
    LWOS_COV_WHITELIST(deviate, LWOS_MISRA(Rule, 21_3), "<QNXBSP>:<lwpu>:<1>:<TID-385>")
    qnxLoopService = calloc(1, sizeof(LwSciQnxEventLoopService));
    CheckNull(qnxLoopService, LwSciError_InsufficientMemory);

    qnxLoopService->eventLoopService.EventService.CreateNativeEventNotifier =
        &LwSciQnxEventService_CreateNativeEventNotifier;
    qnxLoopService->eventLoopService.EventService.CreateLocalEvent =
        &LwSciQnxEventService_CreateLocalEvent;
    qnxLoopService->eventLoopService.EventService.CreateTimerEvent =
        &LwSciQnxEventService_CreateTimerEvent;
    qnxLoopService->eventLoopService.EventService.Delete =
        &LwSciQnxEventService_Delete;

    qnxLoopService->eventLoopService.CreateEventLoop =
        &LwSciQnxEventLoopService_CreateEventLoop;
    qnxLoopService->eventLoopService.WaitForEvent =
        &LwSciQnxEventLoopService_WaitForEvent;
    qnxLoopService->eventLoopService.WaitForMultipleEvents =
        &LwSciQnxEventLoopService_WaitForMultipleEvents;

    chid = ChannelCreate_r(_NTO_CHF_FIXED_PRIORITY | _NTO_CHF_PRIVATE);
    if (chid < 0) {
        lwscievent_err("scievt_err: ChannelCreate failed: line, ret", chid);
        ret = LwSciError_ResourceError;
        goto fail;
    }
    qnxLoopService->waitChid = chid;

    coid = ConnectAttach_r(0, 0, qnxLoopService->waitChid,
        _NTO_SIDE_CHANNEL, _NTO_COF_CLOEXEC);
    if (coid < 0) {
        lwscievent_err("scievt_err: ConnectAttach failed: line, ret", coid);
        ret = LwSciError_ResourceError;
        goto fail;
    }
    qnxLoopService->waitCoid = coid;

#ifdef LWSCIIPC_DEBUG
    LwOsDebugPrintStrWith2Int(LWOS_SLOG_CODE_IPC, SLOG2_INFO,
            "scievt_dbg: pid, tid", getpid(), gettid());
    LwOsDebugPrintStrWith2Int(LWOS_SLOG_CODE_IPC, SLOG2_INFO,
            "scievt_dbg: chid, coid",
            qnxLoopService->waitChid, qnxLoopService->waitCoid);
#endif /* LWSCIIPC_DEBUG */

    CheckZero(pthread_mutex_init(&qnxLoopService->mutex, NULL));
    qnxLoopService->isWaitInProgress = false;
    qnxLoopService->refCount = 1U;

    qnxLoopService->magic = LWSCIEVENT_MAGIC;

    *newEventLoopService = &qnxLoopService->eventLoopService;

    ret = LwSciError_Success;

fail:
    if ((ret != LwSciError_Success) && (qnxLoopService != NULL)) {
        LwSciQnxEventService_Delete(
                &qnxLoopService->eventLoopService.EventService);
    }
    lwscievent_dbg("scievt_dbg:exit");

    /* qnxLoopService will be freed in LwSciQnxEventService_Delete() */
    return ret;
}

/**
 * This function waits up to a configurable timeout for any of a set of
 * particular pulse event which is configured on
 * LwSciQnxEventService_CreateNativeEventNotifier() or
 * LwSciQnxEventService_CreateLocalEvent().
 *
 * Each event notifier in eventNotifierArray must have been created through
 * EventService before calling this function.
 *
 * On a successful return, for each integer i in the range
 * [0, eventNotifierCount), newEventArray[i] will be true if and only if
 * eventNotifierArray[i] had a new event.
 *
 * @param[in]  eventNotifierArray Array of LwSciEventNotifier object pointer
 * @param[in]  eventNotifierCount Array size of eventNotifierArray
 * @param[in]  microseconds 64-bit integer timeout in microseconds,
                            set -1 for infinite timeout
 * @param[out]  newEventArray Array of boolean indicating the corresponding
                            EventNotifier in eventNotifierArray gets event
 *
 * @return ::LwSciError, the completion code of operations:
 * - ::LwSciError_Success         Indicates a successful operation.
 * - ::LwSciError_BadParameter    Indicates an invalid input parameters
 * - ::LwSciError_IlwalidState    Indicates an invalid operation state.
 * - ::LwSciError_NotSupported    Indicates not support condition
 * - ::LwSciError_Timeout         Indicates timeout oclwrrence
 * - ::LwSciError_ResourceError   Indicates not enough system resources.
 * - ::LwSciError_InterruptedCall Indicates interrupt oclwrred
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt: No
 *   - Signal handler(QNX): No
 *   - Thread: Yes
 * - Is thread safe: Yes
 * - Required Privileges(QNX):: None
 * - API Group
 *   - Initialization: No
 *   - Runtime: Yes
 *   - De-initialization: No
 */
static LwSciError LwSciQnxEventLoopService_WaitForMultipleEvents(
            LwSciEventNotifier* const * eventNotifierArray,
            size_t eventNotifierCount,
            int64_t microseconds,
            bool *newEventArray)
{
    LwSciQnxEventNotifier* const * qnxNotifierArray =
        (LwSciQnxEventNotifier* const *)(void* const *)eventNotifierArray;
    LwSciQnxEventLoopService *qnxLoopService = NULL;
    /* ret is overwritten before it can be used */
    LwSciError ret;
    bool hasPendingEvent = false;
    int64_t msec = microseconds;
    int32_t err;
    uint32_t inx;

    lwscievent_dbg("scievt_dbg:enter");

    ret = LwSciEvent_CheckMultipleEventsParams(eventNotifierArray,
        eventNotifierCount, msec, newEventArray, &qnxLoopService);
    if (ret != LwSciError_Success) {
        goto fail2;
    }

    /* initialize newEventArray with false */
    for (inx = 0; inx < eventNotifierCount; inx++) {
        newEventArray[inx] = false;
    }

    CheckZeroTo(pthread_mutex_lock(&qnxLoopService->mutex), fail2);
    if (qnxLoopService->isWaitInProgress) {
        ret = LwSciError_NotSupported;
        err = pthread_mutex_unlock(&qnxLoopService->mutex);
        if (err != EOK) {
            lwscievent_err("error: pthread_mutex_unlock:", err);
        }
        goto fail2;
    }

    hasPendingEvent = LwSciEvent_CheckEventPending(qnxNotifierArray,
                        eventNotifierCount, newEventArray);
    if (hasPendingEvent) {
        /*
         * Any event notified in previous call is pending in notifier array
         * Drain pending other events without waiting for given time.
         */
        msec = 0LL;
    }

    qnxLoopService->isWaitInProgress = true;
    CheckZero(pthread_mutex_unlock(&qnxLoopService->mutex));

    ret = LwSciQnxEvent_WaitMultipleEvents(qnxLoopService,
            qnxNotifierArray, eventNotifierCount, msec, newEventArray);
    if (hasPendingEvent && (ret == LwSciError_Timeout)) {
        /*
         * At least there is one pending event then timeout for other events
         * is treated as a success case
         */
        ret = LwSciError_Success;
    }
    else if (ret == LwSciError_NoSuchProcess) {
        /*
         * Colwert to resource error for clarity (unified error with linux)
         */
        ret = LwSciError_ResourceError;
    }

fail:
    qnxLoopService->isWaitInProgress = false;

fail2:
    lwscievent_dbg("scievt_dbg:exit");
    return ret;
}

/**
 * This function creates a new intra-process local event with which one thread
 * sends a pulse event and the other waits for the event. It registers a signal
 * event with pulse. Event service must be created before this function is
 * called.
 *
 * @param[in]  thisEventService LwSciEventService object pointer created by
 *                              LwSciEventLoopServiceCreate()
 * @param[out]  newLocalEvent LwSciLocaEvent object double pointer
 *
 * @return ::LwSciError, the completion code of operations:
 * - ::LwSciError_Success           Indicates a successful operation.
 * - ::LwSciError_InsufficientMemory  Indicates memory is not sufficient
 * - ::LwSciError_BadParameter      Indicates an invalid input parameters
 * - ::LwSciError_IlwalidState      Indicates an invalid operation state.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt: No
 *   - Signal handler(QNX): No
 *   - Thread: Yes
 * - Is thread safe: Yes
 * - Required Privileges(QNX):: None
 * - API Group
 *   - Initialization: Yes
 *   - Runtime: No
 *   - De-initialization: No
 */
static LwSciError LwSciQnxEventService_CreateLocalEvent(
        LwSciEventService *thisEventService,
        LwSciLocalEvent **newLocalEvent)
{
    struct LwSciQnxEventLoopService *qnxLoopService =
        (struct LwSciQnxEventLoopService *)(void *)thisEventService;
    struct LwSciQnxEventNotifier *qnxNotifier = NULL;
    struct LwSciQnxLocalEvent *qnxLocalEvent = NULL;
    struct LwSciLocalEvent *localEvent = NULL;
    LwSciError ret = LwSciError_IlwalidState;
    int32_t err; /* used in CheckZero */

    lwscievent_dbg("scievt_dbg:enter");

    /* check if input param is invalid */
    CheckNull(thisEventService, LwSciError_BadParameter);
    CheckNull(newLocalEvent, LwSciError_BadParameter);

    /* allocate memory to local event */
    LWOS_COV_WHITELIST(deviate, LWOS_MISRA(Rule, 21_3), "<QNXBSP>:<lwpu>:<1>:<TID-385>")
    qnxLocalEvent = calloc(1UL, sizeof(*qnxLocalEvent));
    CheckNull(qnxLocalEvent, LwSciError_InsufficientMemory);

    /* allocate memory to notifier */
    LWOS_COV_WHITELIST(deviate, LWOS_MISRA(Rule, 21_3), "<QNXBSP>:<lwpu>:<1>:<TID-385>")
    qnxNotifier = calloc(1UL, sizeof(*qnxNotifier));
    CheckNull(qnxNotifier, LwSciError_InsufficientMemory);

    qnxNotifier->eventNotifier.SetHandler = &LwSciQnxEventNotifier_SetHandler;
    qnxNotifier->eventNotifier.Delete = &LwSciQnxEventNotifier_Delete;
    qnxNotifier->qnxLoopService = qnxLoopService;
    qnxNotifier->qnxLocalEvent = qnxLocalEvent;
    qnxNotifier->isPending = false;

    qnxLocalEvent->coid = qnxLoopService->waitCoid;
    localEvent = (LwSciLocalEvent *)(void *)qnxLocalEvent;
    localEvent->eventNotifier = (LwSciEventNotifier *)(void *)qnxNotifier;
    localEvent->Signal = &LwSciQnxLocalEvent_Signal;
    localEvent->Delete = &LwSciQnxLocalEvent_Delete;

    CheckZero(pthread_mutex_lock(&qnxLoopService->mutex));
    /* maximum event notifier count per process: 200U */
    if (qnxLoopService->refCount >= LWSCIEVENT_MAX_EVENTNOTIFIER) {
        CheckZeroLogPrint(pthread_mutex_unlock(&qnxLoopService->mutex));
        ret = LwSciError_InsufficientMemory;
        LWOS_COV_WHITELIST(deviate, LWOS_MISRA(Rule, 21_3), "<QNXBSP>:<lwpu>:<1>:<TID-385>")
        lwscievent_err(
                "scievt_err: max notifier count: line, ret", ret);
        goto fail;
    }
    qnxLoopService->refCount = qnxLoopService->refCount + 1U;
    CheckZero(pthread_mutex_unlock(&qnxLoopService->mutex));

    ret = LwSciError_Success;

fail:
    /* free memory allocated here if ret is not a success */
    if (ret == LwSciError_Success) {
        qnxLocalEvent->magic = LWSCIEVENT_MAGIC;
        qnxNotifier->magic = LWSCIEVENT_MAGIC;
        *newLocalEvent = localEvent;
    } else {
        if (NULL != qnxNotifier) {
            LWOS_COV_WHITELIST(deviate, LWOS_MISRA(Rule, 21_3), "<QNXBSP>:<lwpu>:<1>:<TID-385>")
            free(qnxNotifier);
        }
        if (NULL != qnxLocalEvent) {
            LWOS_COV_WHITELIST(deviate, LWOS_MISRA(Rule, 21_3), "<QNXBSP>:<lwpu>:<1>:<TID-385>")
            free(qnxLocalEvent);
        }
    }

    lwscievent_dbg("scievt_dbg:exit");
    return ret;
}

/**
 * This function sends a pulse event to a peer thread waiting for the event.
 *
 * @param[in]  thisLocalEvent LwSciLocaEvent object double pointer
 *
 * @return ::LwSciError, the completion code of operations:
 * - ::LwSciError_Success         Indicates a successful operation.
 * - ::LwSciError_BadParameter    Indicates an invalid input parameters
 * - ::LwSciError_TryItAgain      Indicates an kernel pulse queue shortage
 * - ::LwSciError_IlwalidState    Indicates an invalid operation state.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt: No
 *   - Signal handler(QNX): No
 *   - Thread: Yes
 * - Is thread safe: Yes
 * - Required Privileges(QNX):: None
 * - API Group
 *   - Initialization: No
 *   - Runtime: Yes
 *   - De-initialization: No
 */
static LwSciError LwSciQnxLocalEvent_Signal(LwSciLocalEvent *thisLocalEvent)
{
    LwSciEventNotifier *eventNotifier;
    LwSciQnxLocalEvent *qnxLocalEvent;
    int32_t err;
    LwSciError ret = LwSciError_IlwalidState;

    /* check if input param is invalid */
    CheckNull(thisLocalEvent, LwSciError_BadParameter);
    eventNotifier = thisLocalEvent->eventNotifier;
    CheckNull(eventNotifier, LwSciError_BadParameter);

    qnxLocalEvent = (LwSciQnxLocalEvent *)(void *)thisLocalEvent;
    if (qnxLocalEvent->magic != LWSCIEVENT_MAGIC) {
        ret = LwSciError_BadParameter;
        goto fail;
    }

    err = MsgSendPulsePtr_r(qnxLocalEvent->coid, SIGEV_PULSE_PRIO_INHERIT,
        _PULSE_CODE_MINAVAIL, eventNotifier);
    if (EOK != err) {
        lwscievent_err(
                "scievt_err: signal local event: ret", ret);
        if (EAGAIN == err) {
            ret = LwSciError_TryItAgain;
        }
        else {
            ret = LwSciError_IlwalidState;
        }
    } else {
        ret = LwSciError_Success;
    }

fail:
    return ret;
}

/**
 * This function releases the LocalEvent and unregister signal event with pulse.
 *
 * @param[in]  thisLocalEent LwSciLocaEvent object double pointer.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt: No
 *   - Signal handler(QNX): No
 *   - Thread: Yes
 * - Is thread safe: Yes
 * - Required Privileges(QNX):: None
 * - API Group
 *   - Initialization: No
 *   - Runtime: No
 *   - De-initialization: Yes
 */
 static void LwSciQnxLocalEvent_Delete(LwSciLocalEvent *thisLocalEvent)
 {
    LwSciQnxLocalEvent *qnxLocalEvent =
        (LwSciQnxLocalEvent *)(void *)thisLocalEvent;
    LwSciError ret; /* used in CheckNull */

    /* check if input param is invalid */
    CheckNull(thisLocalEvent, LwSciError_BadParameter);
    if (qnxLocalEvent->magic != LWSCIEVENT_MAGIC) {
        lwscievent_err("error: magic number is invalid",
            qnxLocalEvent->magic);
        goto fail;
    }
    LWOS_COV_WHITELIST(deviate, LWOS_MISRA(Rule, 21_3), "<QNXBSP>:<lwpu>:<1>:<TID-385>")
    free(thisLocalEvent);
fail:
    return;
 }
