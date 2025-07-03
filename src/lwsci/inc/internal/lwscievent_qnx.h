/*
 * Copyright (c) 2019-2021, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 */

#ifndef INCLUDED_LWSCIEVENT_QNX_H
#define INCLUDED_LWSCIEVENT_QNX_H

#include <lwtypes_tegra_safety.h>
#include <lwscievent_internal.h>
#include <lwos_static_analysis.h>

#include "lwos_s3_tegra_log.h"
/**
 * define LWSCIIPC_DEBUG to enable debug log on LwOsDebugPrint
 * define CONSOLE_DEBUG to enable log on console instead of LwOsDebugPrint
 * or you can define these debug flags on each source code
 */

/**
 * debug message
 */
#ifdef LWSCIIPC_DEBUG2
    #ifdef CONSOLE_DEBUG
        #define lwscievent_dbg(fmt, args...) \
            do { printf("scievt_dbg[L:%d]:%s: " fmt "\n",    \
            __LINE__, __func__, ## args); } while (LW_FALSE)
    #else
        #define lwscievent_dbg(str)  \
            do { (void)LwOsDebugPrintStr(LWOS_SLOG_CODE_IPC, SLOG2_INFO, str); \
            } while (LW_FALSE)
    #endif /* CONSOLE_DEBUG */
#else
    #define lwscievent_dbg(fmt, args...)
#endif /* LWSCIIPC_DEBUG */

#ifdef CONSOLE_DEBUG
    #define lwscievent_err(fmt, val)  \
           printf("%s: " fmt ": %d, %d\n", \
                __func__, __LINE__, val)
#else
    #define lwscievent_err(str, val)  \
        (void)LwOsDebugPrintStrInt(LWOS_SLOG_CODE_IPC, SLOG2_ERROR, \
                str, (int32_t)(val))
#endif /* CONSOLE_DEBUG */

/*
 * APIs whose return value is checked with CheckZero
 * - pthread_mutex_init
 * - phtread_mutex_lock
 * - pthread_mutex_unlock
 */
#define CheckZero(val) \
        do { \
            err = (val); \
            if(0 != (err)) { \
                lwscievent_err("scievt_err: error: line, ret", err); \
                ret = LwSciError_IlwalidState; \
                goto fail; \
            } \
        } while (LW_FALSE)

#define CheckZeroTo(val, label) \
        do { \
            err = (val); \
            if(0 != (err)) { \
                lwscievent_err("scievt_err: error: line, ret", err); \
                ret = LwSciError_IlwalidState; \
                goto label; \
            } \
        } while (LW_FALSE)

/*
 * APIs whose return value is checked with CheckZero
 * - pthread_mutex_unlock
 */
#define CheckZeroLogPrint(val) \
        do { \
            err = (val); \
            if(0 != (err)) { \
                lwscievent_err("scievt_err: error: line, ret", err); \
                ret = LwSciError_IlwalidState; \
            } \
        } while (LW_FALSE)

#define CheckNull(val, scierr) \
        do { \
            if (NULL == (void const *)(val)) { \
                ret = (scierr); \
                lwscievent_err("scievt_err: null pointer: line, ret", (int32_t)ret);\
                goto fail; \
            } \
        } while (LW_FALSE)

#define CheckBothNull(val1, val2, scierr) \
        do { \
            if ((NULL == (void const *)(val1)) && \
                (NULL == (void const *)(val2))) { \
                ret = (scierr); \
                lwscievent_err("scievt_err: null pointer: line, ret", (int32_t)ret);\
                goto fail; \
            } \
        } while (LW_FALSE)

#define CheckTimeoutTo(timeout, scierr, label) \
        do { \
            if ((((timeout) < 0) && \
                ((timeout) != LW_SCI_EVENT_INFINITE_WAIT)) || \
                ((timeout) > MAX_TIMEOUT_US)) { \
                ret = (scierr); \
                lwscievent_err("error: invalid timeout: ret", ret); \
                goto label; \
            } \
        } while (LW_FALSE)

/*
 * APIs whose return value is checked with CheckSuccess
 * - LwSciQnxEvent_UnmaskInterrupt
 */
#define CheckSuccess(val) \
        do { \
            if(LwSciError_Success != (val)) { \
                goto fail; \
            } \
        } while (LW_FALSE)

/*
 * Max event notifier count per process
 * Number of native event is bound for connection limit,
 * which is 100 per process in QNX.
 * Assumed same number of event notifier for local event are allowed
 * if maximum native events are created or
 * event notifier for local event can be created up to 200 if only local event
 * is used.
 */
#define LWSCIEVENT_MAX_EVENTNOTIFIER 200U

#define LWSCIEVENT_MAGIC 0x5a655674U /* "ZeVt" */

typedef struct LwSciQnxNativeEvent      LwSciQnxNativeEvent;
typedef struct LwSciQnxLocalEvent       LwSciQnxLocalEvent;
typedef struct LwSciQnxEventNotifier    LwSciQnxEventNotifier;
typedef struct LwSciQnxEventService     LwSciQnxEventService;
typedef struct LwSciQnxEventLoopService LwSciQnxEventLoopService;
typedef struct LwSciQnxEventLoop        LwSciQnxEventLoop;

struct LwSciQnxNativeEvent {
    LwSciNativeEvent nativeEvent;

    /* QNX OS specific */
    /* client specific internal handle
     * hide it due to security issue.
     */
    void *handle;
};

struct LwSciQnxLocalEvent {
    LwSciLocalEvent localEvent;

    /* QNX Os specific */
    int32_t coid; /* coid which is used to send pulse event */
    uint32_t magic;
};

struct LwSciQnxEventNotifier {
    LwSciEventNotifier eventNotifier;

    /* QNX OS specific */
    LwSciQnxEventLoopService *qnxLoopService;

    /* EventNotifier associated with either of native event or local event */
    LwSciQnxNativeEvent *qnxEvent; /* native event */
    LwSciQnxLocalEvent *qnxLocalEvent; /* local event */

    bool isPending;
    uint32_t magic;
};

struct LwSciQnxEventService {
    LwSciEventService eventService;

    /* add QNX OS specific */
};

struct LwSciQnxEventLoopService {
    LwSciEventLoopService eventLoopService;

    /* QNX OS specific */
    bool isWaitInProgress;
    pthread_mutex_t mutex;
    int32_t waitChid;
    int32_t waitCoid;
    uint32_t refCount;

    uint32_t magic;
};

struct LwSciQnxEventLoop {
    LwSciEventLoop eventLoop;

    /* add QNX OS specific */
};
/******************************************************************************/

#endif /* INCLUDED_LWSCIEVENT_QNX_H */

