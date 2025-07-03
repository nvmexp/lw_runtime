/*
 * Copyright (c) 2018-2021, LWPU Corporation.  All rights reserved.
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
#include <libgen.h>

#ifdef __QNX__
#include <sys/neutrino.h>
#include <sys/procmgr.h>
#include <sys/syspage.h>
#include <sys/stat.h>
#include <inttypes.h>
#include <sys/stat.h>
#include <sys/slog2.h>
#include <fcntl.h>
#endif /* __QNX__ */

#ifdef LINUX
#include <stdint.h>
#include <signal.h>
#include <errno.h>
#include <pthread.h>
#endif

#include <lwsciipc_internal.h>

#define LWIPC_DEBUG 0

#define PRIO_INTR 0x15
#define TEST_EVENT_CODE 32U /* 1 byte */
#define DEFAULT_BACKOFF_NS 100U /* 100ns */

/*
 * This test tool supports inter-VM and inter-Process usecase
 */

#define MAX_CHANNEL_NAME_SIZE    128
#define LWIPC_MSG       "LWIPC VM1"
#define DEFAULT_ITERATIONS       128
#define DEFAULT_LOC_ITERATIONS   32
#define MAX_ENDPOINT    16
#define MAX_LOCALEVENT  16
#define MAX_EVENTNOTIFIER 32
#define MAX_GID_NUM 255

#ifdef LINUX
    #define EOK 0
#endif

struct endpoint
{
    char chname[LWSCIIPC_MAX_ENDPOINT_NAME];    /* endpoint name */
    LwSciIpcEndpoint h;  /* LwSciIpc handle */
    struct LwSciIpcEndpointInfo info; /* endpoint info */
    void *buf;  /* test buffer */
    LwSciEventNotifier *eventNotifier;
    bool established; /* connection is established ? */
    uint32_t idx; /* index of evtNotiArray */
    volatile uint32_t rxEcnt; /* event receive count */
    volatile uint32_t rxCnt; /* frame receive count */
    bool openFlag; /* endpoint is opened successfully */
};

struct localevent
{
    LwSciLocalEvent *evt;
    volatile uint32_t txEcnt; /* signal count */
    volatile uint32_t rxEcnt; /* event receive count */
    uint32_t idx; /* index of tp->loc[] array */
    pthread_t tid; /* thread id */
    bool received; /* test purpose: WaitForMultipleEvents() received flag */
};

typedef struct {
    LwSciEventLoopService *eventLoopServiceP;
    struct endpoint ep[MAX_ENDPOINT];
    struct localevent loc[MAX_LOCALEVENT];
    LwSciEventNotifier *evtNotiArray[MAX_EVENTNOTIFIER];
    uint32_t epCnt; /* endpoint count */
    uint32_t locEvtCnt; /* local event object count */
    uint32_t evtNotiCnt; /* event notifier count */
    uint32_t iterations;        /* iterations for endpoint data transaction */
    uint32_t loc_iterations;    /* iterations for local event */
    bool evtSvcFlag; /* use LwSciEventService */
    int32_t timeout; /* msec timeout for LwSciEventService, (-1: infinite) */
    bool allEstablished; /* all endpoints have been established */
#ifdef __QNX__
    int32_t chid;
    int32_t coid;
    int32_t range_lo; /* lower limit on range of ability */
    int32_t range_hi; /* upper limit on range of ability */
    uint32_t prio; /* pulse priority */

    bool outputLog; /* print result to output file */
    uint32_t testid;    /* testid to be printed in outputLog */
    char testinfo[255]; /* test info string to be printed in outputLog */
    bool pause; /* no exit */
#endif /* __QNX__ */
    volatile uint32_t TotalEvtCnt;   /* total event count */
    volatile uint32_t TotTxLocEvtCnt;   /* total Tx local event count */
    volatile uint32_t TotRxLocEvtCnt;   /* total Rx local event count */
    volatile uint32_t TotalFrmCnt;   /* total frame count */
    uint32_t backoff_delay; /* ns: for perf */
    uint32_t perftest; /* performacne test flag */
    uint32_t bidirtest; /* bi-directional test flag */
    uint32_t opentest; /* opening endpoint test flag. skip data transaction */
    uid_t uid;
    gid_t gid[MAX_GID_NUM]; /* gid[0]=GID, gid[1..255]=SGIDs */
    uint32_t num_gid;
    char *prgname;
    uint32_t errcnt;
    bool initFlag; /* library is initialized successfully */
} test_params;

static test_params s_params;    /* initialized */

static bool s_verboseFlag = false;
static uint32_t s_Stop;

#define dprintf(fmt, args...) \
    if (s_verboseFlag) { \
        printf(fmt, ## args); \
    }

void print_usage(char *argv[]);
#ifdef __QNX__
int32_t drop_privileges(void);
#endif /* __QNX__ */
LwSciError wait_event(int32_t value, bool *newEvent);
void read_test(void);
void read_perf_test(void);
static void setup_termination_handlers(void);
static LwSciError init_resources(void);
static void release_resources(void);


void print_usage(char *argv[])
{
    fprintf(stderr, "Usage: %s [OPTION]...\n", argv[0]);
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "\t -h               : "
            "Print this help screen\n");
    fprintf(stderr, "\t -c <epname1:epname2:...>: "
            "names of LwSciIpc endpoint (max endpoint: %d)\n", MAX_ENDPOINT);
    fprintf(stderr, "\t -l <cnt1>{,cnt2} : "
            "cnt1 - loop count for Native Event\n"
            "\t cnt2 - loop count for Local Event\n");
    fprintf(stderr, "\t -v               : verbose mode\n");
#ifdef __QNX__
    fprintf(stderr, "\t -a <lo:hi>       : "
            "set lo/hi limit on range of ability (hexadecimal)\n");
    fprintf(stderr, "\t -d backoff delay : "
            "delay(def:%dns) btw each perf iterations\n", DEFAULT_BACKOFF_NS);
    fprintf(stderr, "\t -i priority      : "
            "pulse event priority (default:-1(INHERIT), 21(INTR)\n");
    fprintf(stderr, "\t -o open endpoint test\n");
    fprintf(stderr, "\t -p performance test\n");
    fprintf(stderr, "\t -u <uid:gid>     : "
            "UID and GID setting for test\n");
    fprintf(stderr, "\t -E <msec>        : "
            "use LwSciEventService with msec timeout\n"
            "\t wait infinitely if msec is U (-E U)\n");
    fprintf(stderr, "\t -J [id:INFO_STR] : "
            "generate test log file with test id, info string and result\n");
    fprintf(stderr, "\t -@               : "
            "no exit\n");
#endif /* __QNX__ */
#ifdef LINUX
    fprintf(stderr, "\t -E <msec>        : "
            "use LwSciEventService with msec timeout\n"
            "\t wait infinitely if msec is U (-E U)\n"
            "\t test supports only EventService based wait, "
            "so this parmeter is mandatory\n");
#endif /* LINUX */
    fprintf(stderr, "\t -L <count>       : "
            "use {count} LocalEvents\n");
}

/*
 * 0 : pass
 * 1 : fail
 *
 * tester can define and pass other result value from API
 */
static void writeTestLog(uint32_t result)
{
#ifdef __QNX__
    test_params *tp = &s_params;
    FILE *fp;

    if (tp->outputLog) {
        fp = fopen("/tmp/LWSCIIPC_TEST", "a+");
        if (fp != NULL) {
            fprintf(fp, "%d %s %d\n", tp->testid, tp->testinfo, result);
            fclose(fp);
        }
    }
#endif /* __QNX__ */
}

#ifdef __QNX__
/*
 * can be use for dropping root privilege
 * 0000:0000 : root
 * 1000:1000 : lwpu
 * 2000:2000 : lwsciipc
 */
int32_t drop_privileges(void)
{
    test_params *tp = &s_params;
    int32_t abilityId;
    int32_t ret;

    if ((tp->gid[0] != 0) && (tp->uid != 0)) {
        abilityId = procmgr_ability_lookup("LwSciIpcEndpoint");
        /* MEM_PHYS and INTERRUPTEVENT are inter-VM specific */
        ret = procmgr_ability (0,
            PROCMGR_ADN_NONROOT | PROCMGR_AOP_ALLOW | PROCMGR_AID_MEM_PHYS,
            PROCMGR_ADN_NONROOT | PROCMGR_AOP_ALLOW | PROCMGR_AID_INTERRUPTEVENT,
            PROCMGR_ADN_NONROOT | PROCMGR_AOP_ALLOW | PROCMGR_AID_PUBLIC_CHANNEL,
            PROCMGR_ADN_NONROOT | PROCMGR_AOP_ALLOW | PROCMGR_AID_CONNECTION,
            PROCMGR_ADN_NONROOT | PROCMGR_AOP_ALLOW | PROCMGR_AOP_SUBRANGE | PROCMGR_AOP_LOCK | abilityId,
            (uint64_t)tp->range_lo, (uint64_t)tp->range_hi,
            PROCMGR_ADN_NONROOT | PROCMGR_AID_EOL);
        if (ret != EOK) {
            printf("Unable to reserve procmgr abilities: %d\n", ret);
            return ret;
        }
    }

    /* set sub group ids */
    if (tp->num_gid > 1) {
        ret = setgroups(tp->num_gid-1, &tp->gid[1]);
        if (ret == -1) {
            return ret;
        }
    }

    /* if gid is not root */
    if (tp->gid[0] != 0) {
        ret = setregid(tp->gid[0], tp->gid[0]);
        if (ret != EOK) {
            printf("Failed to set GID to %d: %d\n", tp->gid[0], ret);
            return ret;
        }
    }

    /* if uid is not root */
    if (tp->uid != 0) {
        ret = setreuid(tp->uid, tp->uid);
        if (ret != EOK) {
            printf("Failed to set UID to %d: %d\n", tp->uid, ret);
            return ret;
        }
    }

    return EOK;
}

/*
 * Returns:
 * LwSciError_Success : success
 * other              : failure
 */
static LwSciError LwMsgReceivePulse(int32_t chid,
    struct _pulse *pulse, size_t bytes, struct _msg_info *info)
{
    test_params *tp = &s_params;
    LwSciError err;
    int32_t ret;

    do {
        ret = MsgReceivePulse_r(chid, pulse, bytes, info);
        if (ret == EOK) {
            if (pulse->code < 0) {
                /* system pulse */
                continue;
            }
            else if (((uint32_t)pulse->code < TEST_EVENT_CODE) ||
            ((uint32_t)pulse->code >= (TEST_EVENT_CODE + tp->epCnt))) {
                printf("%s: invalid pulse: %d\n", __func__, pulse->code);
                ret = -EILWAL;
            }
        }
        err = LwSciIpcErrnoToLwSciErr(ret);
        break;
    } while(true);

    return err;
}

#endif /* #ifdef __QNX__ */

static uint32_t event2value(bool *newEvent, uint32_t count)
{
    uint32_t i;
    uint32_t value = 0;

    for(i=0;i<count;i++) {
        if (newEvent[i]) {
            value |= (1 << i);
        }
    }

    return value;
}

static LwSciError wait_for_connection(void)
{
    test_params *tp = &s_params;
    struct endpoint *ep;
#ifdef __QNX__
    struct _pulse pulse;
#endif
    uint32_t event = 0;
    uint32_t i;
    uint32_t conn_cnt;
    int32_t timeout;
    LwSciError err;

    while(!s_Stop) {
        conn_cnt = 0;
        for (i = 0; i < tp->epCnt; i++) {
            ep = &tp->ep[i];
            if (ep->established) {
                conn_cnt++;
                continue;
            }

            /*0* LwSciIpcGetEvent() */
            err = LwSciIpcGetEvent(ep->h, &event);
            if (err != LwSciError_Success) {
                printf("[%20s] %s: get event: %d\n",
                    tp->prgname, __func__, err);
                return err;
            }
            if (event & LW_SCI_IPC_EVENT_CONN_EST_ALL) {
                ep->established = true;
                conn_cnt++;
                dprintf("[%20s][EP%02d] %s: established\n",
                    tp->prgname, i, __func__);
            }
        }
        /* all endpoint connections are now established */
        if (conn_cnt == tp->epCnt) {
            tp->allEstablished = true;
            return LwSciError_Success;
        }

        /* LwSciEventService */
        if (tp->evtSvcFlag) {
            /* msec to usec */
            timeout = (tp->timeout >= 0)?(tp->timeout*1000):tp->timeout;

            /*2* WaitForEvent() - single eventNotifier */
            if (tp->evtNotiCnt == 1) {
                err = tp->eventLoopServiceP->WaitForEvent(
                    tp->evtNotiArray[0], timeout);
                if (err != LwSciError_Success) {
                    printf("[%20s] %s: WaitForEvent err: %d\n",
                        tp->prgname, __func__, err);
                    return err;
                }
                /* store statistics */
                tp->TotalEvtCnt++;  /* count event */
                if (tp->epCnt == 1) {
                    tp->ep[0].rxEcnt++; // native event
                    dprintf("[%20s] %s: WaitForEvent (got Native)\n",
                        tp->prgname, __func__);
                }
            }
            /*2* WaitForMultipleEvents() */
            else {
                uint32_t bitEvent;
                volatile uint32_t nativeCnt;
                /* localEvents + nativeEvents */
                bool newEvent[MAX_EVENTNOTIFIER];

                /* wait for native events only */
                memset(newEvent, 0, sizeof(newEvent));
                err = tp->eventLoopServiceP->WaitForMultipleEvents(
                    &tp->evtNotiArray[tp->locEvtCnt],
                    tp->epCnt, timeout, newEvent);
                if (err != LwSciError_Success) {
                    printf("[%20s] %s: WaitForMultipleEvents err: %d\n",
                        tp->prgname, __func__, err);
                    return err;
                }

                /* store Native Event statistics */
                nativeCnt = 0;
                for(i=0;i<tp->epCnt;i++) {
                    if (newEvent[i] == false)
                        continue;
                    tp->ep[i].rxEcnt++; /* count event */
                    tp->TotalEvtCnt++;
                    nativeCnt++;
                }

                if (s_verboseFlag) {
                    bitEvent = event2value(newEvent, tp->evtNotiCnt);
                }

                if (nativeCnt == 0) {
                    printf("[%20s] %s: WaitForMultipleEvents "
                        "returns no event\n",
                        tp->prgname, __func__);
                    return LwSciError_BadParameter;
                }
                else {
                    dprintf("[%20s] %s: WaitForMultipleEvents"
                        "(got Native): 0x%08x\n",
                        tp->prgname, __func__, bitEvent);
                }
            }
        }
#ifdef __QNX__
        /* legacy LwSciIpc */
        else {
            uint64_t to;
            int32_t ret;

            if (tp->timeout >= 0) {
                to = tp->timeout * 1000000U; /* msec to nsec */
                if ((ret = TimerTimeout_r(CLOCK_MONOTONIC, _NTO_TIMEOUT_RECEIVE, NULL,
                    &to, NULL)) < 0) {
                    err = LwSciIpcErrnoToLwSciErr(ret);
                    printf("[%20s] %s: TimerTimeout: %d\n",
                        tp->prgname, __func__, err);
                    return err;
                }
            }
            /*1* MsgReceivePulse() */
            err = LwMsgReceivePulse(tp->chid, &pulse, sizeof(pulse), NULL);
            if (err != LwSciError_Success) {
                printf("[%20s] %s: MsgReceivePulse err: %d\n",
                    tp->prgname, __func__, err);
                return err;
            }
            i = pulse.code - TEST_EVENT_CODE;
            if ((i < 0) || (i >= MAX_ENDPOINT)) {
                printf("[%20s] %s: pulse code err: %d\n",
                    tp->prgname, __func__, i);
                return EILWAL;
            }
            tp->ep[i].rxEcnt++;
            tp->TotalEvtCnt++;  /* count event */
#if LWIPC_DEBUG
            printf("[%20s] %s: code:%x, val:%x\n",
                tp->prgname, __func__, pulse.code, pulse.value.sival_int);
#endif
        }
#endif /* #ifdef __QNX__ */

    }

    return LwSciError_InterruptedCall;
}

LwSciError wait_event(int32_t value, bool *newEvent)
{
    test_params *tp = &s_params;
    struct endpoint *ep;
#ifdef __QNX__
    struct _pulse pulse;
#endif
    uint32_t event = 0;
    int32_t timeout;
    uint32_t i;
    bool gotEvent;
    bool bothEventSet = false;
    LwSciError err;

    while(!s_Stop) {
        gotEvent = false;
        /* only for Native Event */
        for (i = 0; i < tp->epCnt; i++) {
            ep = &tp->ep[i];
            /*0* LwSciIpcGetEvent() */
            err = LwSciIpcGetEvent(ep->h, &event);
            if (err != LwSciError_Success) {
                printf("[%20s] %s: get event: %d\n",
                    tp->prgname, __func__, err);
                return err;
            }
            if (event & value) {
                newEvent[tp->locEvtCnt+i] = true;
                gotEvent = true;
            }
            else {
                /* checked notification, but it's not desired event */
                newEvent[tp->locEvtCnt+i] = false;
            }
        }
        /* desired event is set by LwSciIpcGetEvent() ? otherwise,
         * previous WaitForMultipleEvents() call returns both event type ?
         */
        if (gotEvent || bothEventSet) {
            return LwSciError_Success;
        }

        /* LwSciEventService */
        if (tp->evtSvcFlag) {
            /* msec to usec */
            timeout = (tp->timeout >= 0)?(tp->timeout*1000):tp->timeout;

            /*2* WaitForEvent() - single eventNotifier */
            if (tp->evtNotiCnt == 1) {
                err = tp->eventLoopServiceP->WaitForEvent(
                    tp->evtNotiArray[0], timeout);
                if (err != LwSciError_Success) {
                    printf("[%20s] %s: WaitForEvent err: %d\n",
                        tp->prgname, __func__, err);
                    return err;
                }
                /* store statistics */
                tp->TotalEvtCnt++;  /* count event */
                if (tp->epCnt == 1) {
                    tp->ep[0].rxEcnt++; // native event
                    dprintf("[%20s] %s: WaitForEvent (got Native)\n",
                        tp->prgname, __func__);
                }
                else {
                    tp->loc[0].rxEcnt++; // local event
                    tp->TotRxLocEvtCnt++;  /* count local event */
                    dprintf("[%20s] %s: WaitForEvent (got Local)\n",
                        tp->prgname, __func__);
                    tp->loc[0].received = true;
                    return LwSciError_Success; /* local event only */
                }
            }
            /*2* WaitForMultipleEvents() */
            else {
                uint32_t start;
                uint32_t end;
                uint32_t bitEvent;
                volatile uint32_t localCnt;
                volatile uint32_t nativeCnt;

                err = tp->eventLoopServiceP->WaitForMultipleEvents(
                    tp->evtNotiArray, tp->evtNotiCnt, timeout, newEvent);
                if (err != LwSciError_Success) {
                    printf("[%20s] %s: WaitForMultipleEvents err: %d\n",
                        tp->prgname, __func__, err);
                    return err;
                }

                bothEventSet = false;
                localCnt = 0;
                if (tp->locEvtCnt > 0) { /* has Local Event ? */
                    for(i=0;i<tp->locEvtCnt;i++) {
                        if (newEvent[i] == false)
                            continue;
                        tp->loc[i].rxEcnt++; /* count event */
                        tp->TotRxLocEvtCnt++;  /* count local event */
                        tp->TotalEvtCnt++;
                        localCnt++;
                        tp->loc[i].received = true;
                    }
                }

                nativeCnt = 0;
                if (tp->epCnt > 0) {    /* has Native Event ? */
                    /* store Native Event statistics */
                    if (tp->locEvtCnt > 0) {
                        start = tp->locEvtCnt;
                        end = tp->evtNotiCnt;
                    }
                    else {
                        start = 0;
                        end = tp->epCnt;
                    }

                    for(i=start;i<end;i++) {
                        if (newEvent[i] == false)
                            continue;
                        tp->ep[i-start].rxEcnt++; /* count event */
                        tp->TotalEvtCnt++;
                        nativeCnt++;
                    }
                }

                if (s_verboseFlag) {
                    bitEvent = event2value(newEvent, tp->evtNotiCnt);
                }

                /* no endpoint or no RX native event */
                if ((localCnt > 0) && (nativeCnt == 0)) {
                    dprintf("[%20s] %s: WaitForMultipleEvents"
                        "(local only ): 0x%08x\n",
                        tp->prgname, __func__, bitEvent);
                    return LwSciError_Success; /* local events only */
                }
                else if ((localCnt > 0) && (nativeCnt > 0)) {
                    bothEventSet = true;
                    dprintf("[%20s] %s: WaitForMultipleEvents"
                        "(mixed event): 0x%08x\n",
                        tp->prgname, __func__, bitEvent);
                }
                else if ((localCnt == 0) && (nativeCnt > 0)) {
                    dprintf("[%20s] %s: WaitForMultipleEvents"
                        "(native only): 0x%08x\n",
                        tp->prgname, __func__, bitEvent);
                }
            }
        }
#ifdef __QNX__
        else {
            uint64_t to;
            int32_t ret;

            if (tp->timeout >= 0) {
                to = tp->timeout * 1000000U; /* msec to nsec */
                if ((ret = TimerTimeout_r(CLOCK_MONOTONIC,
                    _NTO_TIMEOUT_RECEIVE, NULL, &to, NULL)) < 0) {
                    err = LwSciIpcErrnoToLwSciErr(ret);
                    printf("[%20s] %s: TimerTimeout: %d\n",
                        tp->prgname, __func__, err);
                    return err;
                }
            }
            /*1* MsgReceivePulse() */
            err = LwMsgReceivePulse(tp->chid, &pulse, sizeof(pulse), NULL);
            if (err != LwSciError_Success) {
                printf("[%20s] %s: MsgReceivePulse err: %d\n",
                    tp->prgname, __func__, err);
                return err;
            }
            i = pulse.code - TEST_EVENT_CODE;
            if ((i < 0) || (i >= MAX_ENDPOINT)) {
                printf("[%20s] %s: pulse code err: %d\n",
                    tp->prgname, __func__, i);
                return EILWAL;
            }
            tp->ep[i].rxEcnt++;
            tp->TotalEvtCnt++;  /* count event */
#if LWIPC_DEBUG
            printf("[%20s] %s: code:%x, val:%x\n",
                tp->prgname, __func__, pulse.code, pulse.value.sival_int);
#endif
        }
#endif /* #ifdef __QNX__ */
    }

    return LwSciError_InterruptedCall;
}

#ifdef __QNX__
static void print_uid_gid(char *str)
{
    dprintf("[%20s] UID: %d, GID: %d, PID: %d\n",
        str, getuid(), getgid(), getpid());
}
#endif /* __QNX__ */

static void release_resources(void)
{
    test_params *tp = &s_params;
    struct endpoint *ep;
    LwSciLocalEvent *local;
    uint32_t i;

    for (i = 0; i < tp->locEvtCnt; i++) {
        local = tp->loc[i].evt;
        local->eventNotifier->Delete(local->eventNotifier);
        local->Delete(local);
    }

    for (i = 0; i < tp->epCnt; i++) {
        ep = &tp->ep[i];

        if (ep->buf) {
            free(ep->buf);
            ep->buf = NULL;
        }

        /*2* EventNotifier::Delete() */
        if (tp->evtSvcFlag) {
            ep->eventNotifier->Delete(ep->eventNotifier);
        }

        dprintf("[%20s] closing LwSciIpc endpoint (%s)\n", tp->prgname, ep->chname);
        /*0* LwSciIpcCloseEndpoint() */
        if (ep->openFlag) {
            LwSciIpcCloseEndpoint(ep->h);
            ep->openFlag = false;
        }
    }

#ifdef __QNX__
    if (tp->coid != 0) {
        (void)ConnectDetach_r(tp->coid);
        tp->coid = 0;
    }
    if (tp->chid != 0) {
        (void)ChannelDestroy_r(tp->chid);
        tp->chid = 0;
    }
#endif
    /*0* LwSciIpcDeinit() */
    if (tp->initFlag) {
        LwSciIpcDeinit();
        tp->initFlag = false;
    }

    /*2* LwSciEventService::Delete() */
    if (tp->evtSvcFlag) {
        tp->eventLoopServiceP->EventService.Delete(&tp->eventLoopServiceP->EventService);
    }

    if (s_Stop == 2) {
        exit(1);
    }
}

static void sig_handler(int sig_num)
{
    s_Stop = 2;

    release_resources();
}

static void setup_termination_handlers(void)
{
    signal(SIGINT, sig_handler);
    signal(SIGTERM, sig_handler);
    signal(SIGHUP, sig_handler);
    signal(SIGQUIT, sig_handler);
    signal(SIGABRT, sig_handler);
}

static LwSciError init_resources(void)
{
    test_params *tp = &s_params;
    struct endpoint *ep;
    uint32_t i;
    LwSciError err = LwSciError_Success;

#ifdef __QNX__
    uint32_t chid;
    uint32_t coid;
    int32_t ret;

    ret = drop_privileges();
    if (ret != EOK) {
        err = LwSciIpcErrnoToLwSciErr(ret);
        goto fail;
    }
    print_uid_gid(tp->prgname);
#endif

    /*2* LwSciEventLoopServiceCreate() */
    if (tp->evtSvcFlag) {
        dprintf("[%20s] %s: LwSciEventLoopServiceCreate\n",
            tp->prgname, __func__);
        err = LwSciEventLoopServiceCreate(1, &tp->eventLoopServiceP);
        if (err != LwSciError_Success) {
            printf("[%20s] %s: LwSciEventLoopServiceCreate: fail (%d)\n",
                tp->prgname, __func__, err);
            goto fail;
        }

        /*2* EventService::CreateLocalEvent() */
        for (i = 0; i < tp->locEvtCnt; i++) {
            LwSciLocalEvent *local;
            err = tp->eventLoopServiceP->EventService.CreateLocalEvent(
                    &tp->eventLoopServiceP->EventService,
                    &tp->loc[i].evt);
            if (err != LwSciError_Success) {
                printf("[%20s] %s: CreatLocalEvent: fail (%d)\n",
                    tp->prgname, __func__, err);
                goto fail;
            }
            local = tp->loc[i].evt;
            tp->loc[i].idx = i; /* numbering local event id */
            /* fill event notifier array for local event */
            tp->evtNotiArray[tp->evtNotiCnt++] = local->eventNotifier;
        }
    }

    // any endpoint ?
    if (tp->epCnt > 0) {
        /*0* LwSciIpcInit */
        err = LwSciIpcInit();
        if (err != LwSciError_Success) {
            printf("[%20s] %s: LwSciIpcInit: fail (%d)\n",
                tp->prgname, __func__, err);
            goto fail;
        }
        tp->initFlag = true;

#ifdef __QNX__
        /* without LwSciEventService --> */
        if (!tp->evtSvcFlag) {
            /* create single channel to get multiple endpoint events */
            chid = ChannelCreate_r(0U);
            if (chid < 0) {
                err = LwSciIpcErrnoToLwSciErr(chid);
                printf("[%20s] %s: ChannelCreate_r: fail (%d)\n",
                    tp->prgname,  __func__, err);
                goto fail;
            }
            tp->chid = chid;
            coid = ConnectAttach_r(0, 0, chid, _NTO_SIDE_CHANNEL, 0);
            if (coid < 0) {
                err = LwSciIpcErrnoToLwSciErr(coid);
                printf("[%20s] %s: ConnectAttach_r: fail (%d)\n",
                    tp->prgname, __func__, err);
                goto fail;
            }
            tp->coid = coid;
            dprintf("[%20s] %s: chid:%d, coid:%d\n",
                tp->prgname, __func__, chid, coid);
        }
#endif
        /* <-- without LwSciEventService */
    }

    for (i = 0; i < tp->epCnt; i++) {
        ep = &tp->ep[i];

        printf("[%20s] opening LwSciIpc endpoint: %s\n",
            tp->prgname, ep->chname);
        /*2* LwSciIpcOpenEndpointWithEventService() */
        if (tp->evtSvcFlag) {
            dprintf("[%20s] %s: LwSciIpcOpenEndpointWithEventService\n",
                tp->prgname, __func__);
            err = LwSciIpcOpenEndpointWithEventService(ep->chname, &ep->h,
                &tp->eventLoopServiceP->EventService);
        }
        /*1* LwSciIpcOpenEndpoint() */
        else {
            err = LwSciIpcOpenEndpoint(ep->chname, &ep->h);
        }
        if (err != LwSciError_Success) {
            printf("[%20s] %s: LwSciIpcOpenEndpoint: fail (%d)\n",
                tp->prgname, __func__, err);
            goto fail;
        }
        ep->openFlag = true;
        dprintf("[%20s] %s: Endpoint Handle: 0x%lx\n",
            tp->prgname, __func__, ep->h);

        /*2* LwSciIpcGetEventNotifier() */
        if (tp->evtSvcFlag) {
            dprintf("[%20s] %s: LwSciIpcGetEventNotifier\n",
                tp->prgname, __func__);
            err = LwSciIpcGetEventNotifier(ep->h, &ep->eventNotifier);
            if (err != LwSciError_Success) {
                printf("[%20s] %s: LwSciIpcGetEventNotifier: fail (%d)\n",
                    tp->prgname, __func__, err);
                goto fail;
            }
            /* set event notifier array index using count */
            ep->idx = tp->evtNotiCnt;
            /* fill event notifier array for native event */
            tp->evtNotiArray[tp->evtNotiCnt++] = ep->eventNotifier;
        }
#ifdef __QNX__
        /*1* LwSciIpcSetQnxPulseParam() */
        else {
            err = LwSciIpcSetQnxPulseParam(ep->h, tp->coid, tp->prio,
                (TEST_EVENT_CODE+i), (void *)NULL);
            if (err != LwSciError_Success) {
                printf("[%20s] %s: LwSciIpcSetQnxPulseParam: fail (%d)\n",
                    tp->prgname, __func__, err);
                goto fail;
            }
            ep->idx = i;
        }
#endif
        /*0* LwSciIpcGetEndpointInfo() */
        err = LwSciIpcGetEndpointInfo(ep->h, &ep->info);
        if (err != LwSciError_Success) {
            printf("[%20s] %s: LwSciIpcGetEndpointInfo: fail (%d)\n",
                tp->prgname, __func__, err);
            goto fail;
        }
        printf("[%20s] endpoint_info: nframes = %d, frame_size = %d\n",
            tp->prgname, ep->info.nframes, ep->info.frame_size);

        ep->buf = calloc(1, ep->info.frame_size);
        if (ep->buf == NULL) {
            err = LwSciIpcErrnoToLwSciErr(errno);
            printf("[%20s] %s: Failed to allocate buffer of size %u\n",
                tp->prgname, __func__, ep->info.frame_size);
            goto fail;
        }

        /*0* LwSciIpcResetEndpoint() */
        LwSciIpcResetEndpoint(ep->h);
    }

fail:
    return err;
}

static void display_statistics(void)
{
    test_params *tp = &s_params;
    uint32_t i;

    if (tp->epCnt > 0) {
        for (i=0;i<tp->epCnt;i++) {
            printf("[%20s][EP%02d] read frame count        : %u\n",
                tp->prgname, i, tp->ep[i].rxCnt);
        }
        printf("[%20s]       total read frame count  : %d\n",
            tp->prgname, tp->TotalFrmCnt);
    }

    if (tp->evtSvcFlag && (tp->locEvtCnt > 0)) {
        for (i=0;i<tp->locEvtCnt;i++) {
            printf("[%20s][LO%02d] local event count       : (Tx:%u, Rx:%u)\n",
                tp->prgname, i, tp->loc[i].txEcnt, tp->loc[i].rxEcnt);
        }
        printf("[%20s]       total local event count : (Tx:%u, Rx:%u)\n",
            tp->prgname, tp->TotTxLocEvtCnt, tp->TotRxLocEvtCnt);
    }
    for (i=0;i<tp->epCnt;i++) {
        printf("[%20s][EP%02d] native event count      : %u\n",
            tp->prgname, i, tp->ep[i].rxEcnt);
    }

    printf("[%20s]       total event count       : %u\n",
        tp->prgname, tp->TotalEvtCnt);
}

void read_test(void)
{
    test_params *tp = &s_params;
    struct endpoint *ep;
    uint32_t *ptr;
    int32_t bytes;
    uint32_t start;
    uint32_t end;
    uint32_t i;
    uint32_t total_iterations;
    bool newEvent[MAX_EVENTNOTIFIER]; /* localEvents + nativeEvents */
    LwSciError err;

    total_iterations = tp->iterations + (tp->loc_iterations * tp->locEvtCnt);

    printf("[%20s] Ping Test mode (loop: %d)\n",
        tp->prgname, total_iterations);

    if (tp->epCnt > 0) {
        /* if there is any endpoint,
         * need to wait for connection establishment
         * before starting any data transaction.
         */
        err = wait_for_connection();
        if (err != LwSciError_Success) {
            tp->errcnt++;
            goto fail;
        }
    }

    /* native event count + local iterations * thread count */
    while (((tp->TotalFrmCnt + tp->TotRxLocEvtCnt) <
        total_iterations) && !s_Stop) {

        memset(newEvent, 0, sizeof(newEvent));
        err = wait_event(LW_SCI_IPC_EVENT_READ, newEvent);

        if (err == LwSciError_Success) {
            if (tp->epCnt > 0) { /* Native Event ? */
                if (tp->evtSvcFlag) {
                    /* use LwSciEventService */
                    start = tp->locEvtCnt;
                    end = tp->evtNotiCnt;
                }
                else {
                    /* legacy LwSciIpc */
                    start = 0;
                    end = tp->epCnt;
                }

                for(i=start;i<end;i++) {
                    if (newEvent[i] == false)
                        continue;

                    ep = &tp->ep[i-start];
                    ptr = (uint32_t *)ep->buf;

                    /*0* LwSciIpcRead() */
                    err = LwSciIpcRead(ep->h, ep->buf,
                        ep->info.frame_size, &bytes);
                    if(err != LwSciError_Success) {
                        printf("[%20s] %s: error in reading: %d\n",
                            tp->prgname, __func__, err);
                        tp->errcnt++;
                        break;
                    }
                    ep->rxCnt++;
                    dprintf("[%20s] %10sRD#%02d: %d %d (endpoint: %s)\n",
                        tp->prgname, "", ep->rxCnt, ptr[0], ptr[1], ep->chname);
                    tp->TotalFrmCnt++;
                }
            }
        }
        else {
            printf("[%20s] %s: wait_event() error: %d\n",
                tp->prgname, __func__, err);
            tp->errcnt++;
            break;
        }
    }

fail:
    /* callwlate Total Tx Local Event */
    for (i=0;i<tp->locEvtCnt;i++) {
        tp->TotTxLocEvtCnt += tp->loc[i].txEcnt;
    }

    if ((tp->TotalFrmCnt != tp->iterations) ||
    (tp->TotTxLocEvtCnt != tp->TotRxLocEvtCnt)) {
        printf("[%20s] %s: count mismatch\n", tp->prgname, __func__);
        dprintf("[%20s] TotalFrmCnt:%d, iteration:%d\n", tp->prgname,
            tp->TotalFrmCnt, tp->iterations);
        dprintf("[%20s] TotTxLocEvtCnt:%d, TotRxLocEvtCnt:%d\n", tp->prgname,
            tp->TotTxLocEvtCnt, tp->TotRxLocEvtCnt);
        tp->errcnt++;
    }

    /* Show statistics */
    display_statistics();

    /* terminate thread */
    if (tp->errcnt > 0) {
        s_Stop = 1;
    }
}

void read_perf_test(void) {}


static void *signal_localevent_thread(void *arg)
{
    test_params *tp = &s_params;
    struct localevent *loc = (struct localevent *)arg;
    LwSciLocalEvent *local = loc->evt;
    uint32_t i;
    LwSciError err;

    if (tp->epCnt > 0) {
        while (!tp->allEstablished && !s_Stop) {
            /* 1) sleep to get tid.
             * 2) wait till all connections are established.
             */
            usleep(1000);
        }
    }
    else {
        usleep(1000);   /* sleep to get tid */
    }

    dprintf("[%20s][T:%u][LO%d] %s is created\n",
        tp->prgname, (unsigned int)loc->tid, loc->idx,  __func__);

    /*2* LwSciLocalEvent::Signal() */
    for(i=0;i < tp->loc_iterations;i++) {
        loc->received = false;
        err = local->Signal(local);
        if (err != LwSciError_Success) {
            printf("[%20s][T:%u][LO%d]: signal local event error: %d\n",
                tp->prgname, (unsigned int)loc->tid, loc->idx, err);
            tp->errcnt++;
        }
        loc->txEcnt++;

        dprintf("[%20s][T:%d][LO%d]: Signal#%d\n", tp->prgname,
            (unsigned int)loc->tid, loc->idx, i);

        /* waiting time to prevent signal overlapping.
         * [NOTE] this is not actual usecase, just for event counting
         */
        while(!loc->received && !s_Stop && (err == LwSciError_Success)) {
            usleep(1000);   /* 1ms */
        }
    }

    dprintf("[%20s][T:%u][LO%d]: completed\n",
        tp->prgname, (unsigned int)loc->tid, loc->idx);

    return NULL;
}

#ifdef __QNX__
static void change_slogbuf_size(void)
{
    int ret;
    slog2_buffer_t handle;
    slog2_buffer_set_config_t config;
    extern char *__progname;

    if (!slog2_reset()) {
        config.buffer_set_name = (const char *)__progname;
        config.num_buffers = 1;
        config.verbosity_level = SLOG2_INFO;
        config.buffer_config[0].buffer_name = (const char *)__progname;
        config.buffer_config[0].num_pages = 1024; /* 4MB buffer */
        config.max_retries = 3U;
        ret = slog2_register(&config, &handle, SLOG2_LIMIT_RETRIES);
        if (-1 == ret) {
            printf("Error in slog2_register\n");
        } else {
            slog2_set_default_buffer(handle);
        }
    } else {
        printf("Error in slog2_reset\n");
    }
}
#endif /* __QNX__ */

int main(int argc, char *argv[])
{
    test_params *tp = &s_params;
    int32_t opt;
#ifdef __QNX__
    int32_t tokerr = -1;
#endif
    uint32_t exitcode;
    uint32_t i;
    LwSciError err;

#ifdef __QNX__
    change_slogbuf_size();

    tp->prio = SIGEV_PULSE_PRIO_INHERIT; /* default: -1, INTR: 0x15(21) */
    tp->range_lo = 0;
    tp->range_hi = INT_MAX;
    tp->pause = false;
#endif
    tp->timeout = LW_SCI_EVENT_INFINITE_WAIT;

    tp->iterations = DEFAULT_ITERATIONS;
    tp->loc_iterations = DEFAULT_LOC_ITERATIONS;
    tp->backoff_delay = DEFAULT_BACKOFF_NS;    /* 200ns */

    printf("[%20s] enter LwSciIpc test\n", argv[0]);

    while ((opt = getopt(argc, argv, "a:c:hi:l:opt:u:vw:E:J:L:@")) != -1)
    {
        switch (opt)
        {
#ifdef __QNX__
            case 'a':
                /* Set ability subrange */
                {
                    char* token = strtok(optarg, ":");

                    tokerr = -1;
                    if (token != NULL) {
                        tp->range_lo = (int32_t)strtoul(token, NULL, 16);
                        token = strtok(NULL, ":");
                        if (token != NULL) {
                            tp->range_hi = (int32_t)strtoul(token, NULL, 16);
                            tokerr = 0;
                        }
                    }
                    if (tokerr != 0) {
                        printf("%s: range_lo/range_hi not specified correctly\n", argv[0]);
                        tp->errcnt++;
                        goto done;
                    }
                    break;
                }
            case 'd':
                tp->backoff_delay = strtoul(optarg, NULL, 0);
                break;
#endif /* __QNX__ */
            case 'c':
                {
                    char *token = strtok(optarg, ":");
                    int32_t cnt = 0;

                    while (token != NULL || cnt >= MAX_ENDPOINT) {
                        memcpy((char *)tp->ep[cnt++].chname, token,
                            strlen(token));
                        token = strtok(NULL, ":");
                    }
                    tp->epCnt = cnt;

                    if (tp->epCnt > MAX_ENDPOINT) {
                        printf("%s: endpoint count is out of range\n", argv[0]);
                        tp->errcnt++;
                        goto done;
                    }
                }
                break;
            case 'h': /* HELP */
                print_usage(argv);
                goto done;
#ifdef __QNX__
            case 'i':
                tp->prio = strtoul(optarg, NULL, 0);
                break;
#endif /* __QNX__ */
            case 'l':
                {
                    char* token = strtok(optarg, ",");
                    int32_t val;
                    if (token != NULL) {
                        errno = EOK;
                        val = (int32_t)strtoul(token, NULL, 0);
                        if (val < 0 || errno != EOK) {
                            printf("%s: invalid main iterations: %d\n",
                                argv[0], val);
                            tp->errcnt++;
                            goto done;
                        }
                        tp->iterations = val;
                        token = strtok(NULL, ",");
                        if (token != NULL) {
                            errno = EOK;
                            val = (int32_t)strtoul(token, NULL, 0);
                            if (val < 0 || errno != EOK) {
                                printf("%s: invalid local iterations: %d\n",
                                    argv[0], val);
                                tp->errcnt++;
                                goto done;
                            }
                            tp->loc_iterations = val;
                        }

                        if ((tp->iterations == 0) &&
                            (tp->loc_iterations == 0)) {
                            printf("%s: invalid iterations: %d\n",
                                argv[0], val);
                            tp->errcnt++;
                            goto done;
                        }
                    }
                }
                break;
#ifdef __QNX__
            case 'o':
                tp->opentest = 1;
                break;
            case 'p':
                tp->perftest = 1;
                break;
            case 'u':
                /* Set UID:GID */
                {
                    char* token = strtok(optarg, ":");

                    tokerr = -1;
                    if (token != NULL) {
                        tp->uid = (uid_t)strtoul(token, NULL, 10);
                        while ((token = strtok(NULL, ":,")) != NULL) {
                            tp->gid[tp->num_gid] = (gid_t)strtoul(token, NULL, 10);
                            tp->num_gid++;
                            tokerr = 0;
                        }
                    }
                    if (tokerr != 0) {
                        printf("%s: UID/GID not specified correctly\n", argv[0]);
                        tp->errcnt++;
                        goto done;
                    }
                    break;
                }
            case 'J':
                {
                    /* -J TEST_ID:TEST_INFO_STR
                     * ex) -J 100:API_TEST1
                     */
                    char* token = strtok(optarg, ":");

                    if (token != NULL) {
                        tp->testid = (int32_t)strtoul(token, NULL, 10);
                    }
                    token = strtok(NULL, ":");
                    if (token != NULL) {
                        strncpy(tp->testinfo, token, sizeof(tp->testinfo));
                    }
                    tp->outputLog = true;
                }
                break;
            case '@':
                tp->pause = true;
                break;
#endif /* __QNX__ */
            case 'v':
                s_verboseFlag = true;
                break;
            case 'E':   /* use LwSciEventService */
                {
                    uint32_t val;
                    errno = EOK;
                    val = strtoul(optarg, NULL, 0);
                    if (val >= 0 && errno == 0) { // decimal including 0
                        tp->timeout = val;
                    }
                    tp->evtSvcFlag = true;
                }
                break;
            case 'L':   /* Local Event */
                tp->locEvtCnt = strtoul(optarg, NULL, 0);
                if (tp->locEvtCnt > MAX_LOCALEVENT) {
                    printf("%s: local event count is out of range\n", argv[0]);
                    tp->errcnt++;
                    goto done;
                }
                break;
#ifdef LINUX
            case ':':
                fprintf(stderr, "Option `-%c` requires an argument.\n", optopt);
                return EILWAL;
            case '?':
                if (isprint(optopt))
                {
                    fprintf(stderr, "Unknown option `-%c`.\n", optopt);
                }
                else
                {
                    fprintf(stderr, "Unknown option ``\\x%x`.\n", optopt);
                }
                return EILWAL;
#endif /* LINUX */
            default:
                print_usage(argv);
                tp->errcnt++;
                goto done;
        }
    }

#ifdef LINUX
    if (tp->evtSvcFlag != true) {
        fprintf(stderr, "-E parameter is mandatory and not provided\n");
        print_usage(argv);
        return -1;
    }
#endif /* LINUX */

    tp->prgname = basename(argv[0]);
    if (tp->epCnt == 0 && tp->locEvtCnt == 0) {
        fprintf(stderr, "need one endpoint or localevent at least\n");
        print_usage(argv);
        tp->errcnt++;
        goto done;
    }

    /* display configuration */
    for(i=0;i < tp->epCnt; i++) {
        printf("[%20s][EP%02d] endpoint name: %s\n", tp->prgname, i,
            tp->ep[i].chname);
    }
    if (tp->epCnt == 0) {
        tp->iterations = 0;
    }
    else {
        printf("[%20s] iteration for Native Event : %d\n", tp->prgname,
            tp->iterations);
    }

    if (tp->evtSvcFlag) {
        if (tp->locEvtCnt == 0) {
            tp->loc_iterations = 0;
        }
        printf("[%20s] Use LwSciEventService (timeout:%dms)\n",
            tp->prgname, tp->timeout);
        printf("[%20s] Local Event count: %d\n", tp->prgname, tp->locEvtCnt);
        printf("[%20s] iteration for Local Event : %d\n", tp->prgname,
            tp->loc_iterations);
    }

    /* setup sig handler */
    setup_termination_handlers();

    /* init LwSciIpc resources */
    err = init_resources();
    if (err != LwSciError_Success) {
        tp->errcnt++;
        goto done;
    }
    dprintf("[%20s] Total EventNotifiers: %d\n", tp->prgname, tp->evtNotiCnt);

    /* create threads for local event */
    if (tp->locEvtCnt > 0) {
        for (i=0;i<tp->locEvtCnt;i++) {
            pthread_create(&tp->loc[i].tid, NULL,
                &signal_localevent_thread, &tp->loc[i]);
            printf("[%20s] signal_localevent_thread (tid:%u)\n",
                tp->prgname, (unsigned int)tp->loc[i].tid);
        }
    }

    /* skip data transfer test in opening endpoint test */
    if (tp->opentest == 0) {
        if (tp->perftest) {
            read_perf_test();
        }
        else {
            read_test();
        }
    }

    /* join threads */
    if (tp->locEvtCnt > 0) {
        for (i=0;i<tp->locEvtCnt;i++) {
            pthread_join(tp->loc[i].tid, NULL);
            printf("[%20s] local event thread is terminated (tid:%u)\n",
                tp->prgname, (unsigned int)tp->loc[i].tid);
        }
    }

done:
    /* release LwSciIpc resources */
    release_resources();

    if (tp->errcnt != 0) {
        printf("[%20s] : test FAILED\n", tp->prgname);
        exitcode = EXIT_FAILURE;
    }
    else {
        printf("[%20s] : test PASSED\n", tp->prgname);
        exitcode = EXIT_SUCCESS;
    }

    writeTestLog(exitcode);

#ifdef __QNX__
    while (tp->pause) {
        pause();
    }
#endif /* __QNX__ */

    return exitcode;
}

