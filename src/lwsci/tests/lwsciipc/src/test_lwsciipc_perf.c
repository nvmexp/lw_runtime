/*
 * Copyright (c) 2018-2021, LWPU CORPORATION. All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 */

/*
 * LwSciIpc loopback test on QNX.
 *
 * This test tool support inter-Thread and inter-VM usecase
 */

#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <inttypes.h>
#include <string.h>
#include <errno.h>
#include <pthread.h>
#include <semaphore.h>
#include <libgen.h>

#ifdef __QNX__
#include <sys/neutrino.h>
#include <sys/syspage.h>
#include <sys/procmgr.h>
#include <sys/stat.h>
#include <fcntl.h>
#define LWIPC_DEBUG 0
#endif /* __QNX__ */

#ifdef LINUX
#include <stdint.h>
#include <signal.h>
#endif

#include <lwsciipc_internal.h>

#define MAX_ENDPOINT    100
#define MAX_EVENTNOTIFIER (MAX_ENDPOINT)

#define MEGABYTE 1048576
#define TEST_EVENT_CODE 0x1 /* 1 byte */
#define TIMEOUT_CODE 44
#define DEFAULT_TIMEOUT_MS 500

#define RX_ROLE 1
#define TX_ROLE 2

#define DEFAULT_R_BACKOFF_DELAY 100UL
#define DEFAULT_W_BACKOFF_DELAY 100UL
#define DEFAULT_ITERATIONS    128UL
#define MAX_GID_NUM 255

/* Test type */
#define TT_NORMAL  0
#define TT_PERF    1
#define TT_STRESS  2

struct endpoint
{
    /* this is inter-thread communication.
     * Each endpoint should have own EventLoopService object
     */
    LwSciEventLoopService *eventLoopServiceP;

    char chname[LWSCIIPC_MAX_ENDPOINT_NAME];    /* endpoint name */
    LwSciIpcEndpoint h;    /* LwSciIpc handle */
    struct LwSciIpcEndpointInfo info; /* endpoint info */
#ifdef __QNX__
    int32_t chid;   /* channel id to get event */
    int32_t coid;   /* connection id to send event in library */
#endif /* __QNX__ */
#ifdef LINUX
    int32_t fd;   /* fd to get event */
#endif
    int32_t role; /* main role */
    pthread_t tid;    /* thread id */
    pthread_t peertid;    /* peer thread id */
    void *buf;    /* test buffer */
    void *rbuf;    /* test buffer */
    void *sbuf;    /* test buffer */
    uint32_t evt_cnt;
    LwSciEventNotifier *eventNotifier; /* LwSciEventService */
    uint64_t iterations; /* for recovery test */
    bool openFlag;  /* endpoint is opened successfully */

    uint32_t prevData; /* for stress test (comparing data) */
    volatile uint32_t rtxCnt; /* frame Rx/Tx count */
};

typedef struct {
    struct endpoint tx[MAX_ENDPOINT];
    struct endpoint rx[MAX_ENDPOINT];
    LwSciEventNotifier *evtNotiArray[MAX_EVENTNOTIFIER];
    uint32_t epCnt; /* endpoint count */
    uint32_t evtNotiCnt; /* event notifier count */
    uint64_t iterations;
    bool evtSvcFlag; /* API variation test flag: LwSciEventService */
    bool bindEvtSvcFlag; /* API variation test flag: LwSciIpcBindEventService */
#ifdef __QNX__
    uint32_t start_delay;    /* ms */
    uint32_t repeat_delay;    /* ms */

    uint64_t r_backoff_delay; /* ns for read */
    uint64_t w_backoff_delay; /* ns for write */
    uint32_t bidirtest; /* bidirectional test */
    uint32_t connrcvtest; /* connection recovery test */
    uint32_t priority; /* pulse priority */
    int32_t range_lo; /* lower limit on range of ability */
    int32_t range_hi; /* upper limit on range of ability */

    char txprefix[16];  /* Tx endpoint prefix for stress test */
    char rxprefix[16];  /* Rx endpoint prefix for stress test */
    uint32_t epstart;   /* start endpoint number for stress test */
    uint32_t epend;     /* end endpoint number for stress test */
    uint32_t logpersec; /* logging period in sec */

    int32_t pid;    /* process id */
    int32_t rtid;   /* rx thread id */
    int32_t ttid;   /* tx thread id */
    /* channel/connection for stress test */
    int32_t tchid;   /* Tx channel id to get event */
    int32_t tcoid;   /* Tx connection id to send event in library */
    int32_t rchid;   /* Rx channel id to get event */
    int32_t rcoid;   /* Rx connection id to send event in library */
    /* statistics for stress test */
    volatile uint64_t TotalRxEvtCnt; /* total event count (Rx) */
    volatile uint64_t TotalTxEvtCnt; /* total event count (Tx) */
    volatile uint64_t TotalRxFrmCnt; /* total frame count (Rx) */
    volatile uint64_t TotalTxFrmCnt; /* total frame count (Tx) */
    uint32_t timestampS;    /* print start timestamp */
    uint32_t timestampE;    /* print end timestamp */

    bool outputLog; /* print result to output file */
    uint32_t testid;    /* testid to be printed in outputLog */
    char testinfo[255]; /* test info string to be printed in outputLog */
#endif /* __QNX__ */
    uint32_t testtype; /* TT_NORMAL, TT_PERF, TT_STRESS */
    int32_t timeout; /* msec timeout for WaitForEvent/MsgReceivePulse */
    uid_t uid;
    gid_t gid[MAX_GID_NUM]; /* gid[0]=GID, gid[1..255]=SGIDs */
    uint32_t num_gid;
    int32_t errcnt;
    uint32_t txerrcnt;   /* error count for tx (stress test) */
    uint32_t rxerrcnt;   /* error count for rx (stress test) */
    char *prgname;
    bool initFlag; /* library is initialized successfully */
} test_params;

static bool s_verboseFlag = false;
static bool s_quietFlag = false;
static uint32_t s_Stop;
static test_params s_params;

#define mprintf(fmt, args...) \
    if (!s_quietFlag) { \
        printf(fmt, ## args); \
    }

#define dprintf(fmt, args...) \
    if (s_verboseFlag && !s_quietFlag) { \
        printf(fmt, ## args); \
    }

#ifdef __QNX__
static int32_t drop_privileges(void);
static void print_uid_gid(char *str);
#endif /* __QNX__ */
static void setup_termination_handlers(void);
static LwSciError init_resources(struct endpoint *ep);
static void release_resources(void);

#ifdef LINUX
static LwSciError wait_event(struct endpoint *ep, uint32_t mask, uint32_t *events)
{
        fd_set rfds;
        uint32_t event = 0;
        int32_t ret;

        FD_ZERO(&rfds);
        FD_SET(ep->fd, &rfds);

        while(!s_Stop) {
                ret = LwSciIpcGetEvent(ep->h, &event);
                if (ret != LwSciError_Success) {
                        printf("%s: get event: %d\n", __func__, ret);
                        return ret;
                }

                if (event & mask) {
                        *events = (event & mask);
                        return LwSciError_Success;
                }

                ret = select(ep->fd + 1, &rfds, NULL, NULL, NULL);
                if ((ret < 0) & (ret != EINTR)) {
                        printf("error in select\n");
                        exit(1);
                }
                ep->evt_cnt++;
        }

        return LwSciError_Success;
}
#endif

#ifdef __QNX__
/*
 * can be use for dropping root privilege
 * 0000:0000 : root
 * 1000:1000 : lwpu
 * 2000:2000 : lwsciipc
 */
static int32_t drop_privileges(void)
{
    test_params *tp = (test_params *)&s_params;
    int32_t abilityId;
    int32_t ret;

    if ((tp->gid[0] != 0) && (tp->uid != 0)) {
        abilityId = procmgr_ability_lookup("LwSciIpcEndpoint");
        ret = procmgr_ability (0,
            PROCMGR_ADN_NONROOT | PROCMGR_AOP_ALLOW | PROCMGR_AID_MEM_PHYS,
            PROCMGR_ADN_NONROOT | PROCMGR_AOP_ALLOW | PROCMGR_AID_INTERRUPTEVENT,
            PROCMGR_ADN_NONROOT | PROCMGR_AOP_ALLOW | PROCMGR_AID_PUBLIC_CHANNEL,
            PROCMGR_ADN_NONROOT | PROCMGR_AOP_ALLOW | PROCMGR_AID_CONNECTION,
            PROCMGR_ADN_NONROOT | PROCMGR_AOP_ALLOW | PROCMGR_AOP_SUBRANGE | PROCMGR_AOP_LOCK | abilityId,
            (uint64_t)tp->range_lo, (uint64_t)tp->range_hi,
            PROCMGR_ADN_NONROOT | PROCMGR_AID_EOL);
        if (ret != EOK) {
            printf("%s: Unable to reserve procmgr abilities: %d\n",
                __func__, ret);
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
 * others             : failure
 */
static LwSciError LwMsgReceivePulse(struct endpoint *ep, int32_t code,
    struct _pulse *pulse, size_t bytes, struct _msg_info *info,
    uint32_t *cookie)
{
    test_params *tp = &s_params;
    LwSciError ret;
    int32_t err;

    do {
        err = MsgReceivePulse_r(ep->chid, pulse, bytes, info);
        if (err == EOK) {
            if (pulse->code < 0) {
                /* system pulse */
                continue;
            }
            else if (((uint32_t)pulse->code < TEST_EVENT_CODE) ||
                ((uint32_t)pulse->code >= (TEST_EVENT_CODE + tp->epCnt))) {
                printf("%s: invalid pulse: %d\n", __func__, pulse->code);
                err = -EILWAL;
            }
            else if (cookie != NULL) {
                *cookie = pulse->value.sival_int;
            }
        }
        ret = LwSciIpcErrnoToLwSciErr(err);
        break;
    } while(true);

    return ret;
}

/*
 * blocking call to wait specific event
 */
static LwSciError wait_event(struct endpoint *ep, uint32_t mask,
    uint32_t *events)
{
    test_params *tp = &s_params;
    struct _pulse pulse;
    uint32_t event = 0;
    uint64_t to;
    int32_t timeout;
    int32_t ret;
    LwSciError err;

    while(!s_Stop) {
        err = LwSciIpcGetEvent(ep->h, &event);
        if (err != LwSciError_Success) {
            printf("%s: get event: %d\n", __func__, err);
            return err;
        }
        if (event & mask) {
            *events = (event & mask);
            return LwSciError_Success;
        }

#if LWIPC_DEBUG
        printf("[tid:%d] %s: event: 0x%x\n", gettid(),  __func__, event);
#endif
        if (tp->evtSvcFlag) {
            /* msec to usec */
            timeout = (tp->timeout >= 0)?(tp->timeout*1000):tp->timeout;

            err = ep->eventLoopServiceP->WaitForEvent(ep->eventNotifier,
                timeout);
            if (err != LwSciError_Success) {
                printf("[%20s] %s: WaitForEvent err: %d\n",
                    tp->prgname, __func__, err);
                return err;
            }
        }
        else {
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
            err = LwMsgReceivePulse(ep, TEST_EVENT_CODE,
                &pulse, sizeof(pulse), NULL, NULL);
            if (err != LwSciError_Success) {
                printf("[%20s] %s: MsgReceivePulse err: %d\n",
                    tp->prgname, __func__, err);
                return 0;
            }
            ep->evt_cnt++;
#if LWIPC_DEBUG
            printf("[tid:%d] %s: code:%x, val:%x\n",
                gettid(), __func__, pulse.code, pulse.value.sival_int);
#endif
        }
    }
    return 0;
}

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

static LwSciError wait_multi_events(int32_t dir, int32_t value,
    LwSciEventLoopService *eventLoopServiceP, bool *newEvent)
{
    test_params *tp = &s_params;
    struct endpoint *ep;
    struct _pulse pulse;
    uint32_t event = 0;
    int32_t timeout;
    uint32_t i;
    bool gotEvent;
    LwSciError err;

    while(!s_Stop) {
        gotEvent = false;
        /* only for Native Event */
        for (i = 0; i < tp->epCnt; i++) {
            if (dir == RX_ROLE)
                ep = &tp->rx[i];
            else
                ep = &tp->tx[i];

            /*0* LwSciIpcGetEvent() */
            err = LwSciIpcGetEvent(ep->h, &event);
            if (err != LwSciError_Success) {
                printf("[%20s] %s: get event: %d\n",
                    tp->prgname, __func__, err);
                return err;
            }
            if (event & value) {
                newEvent[i] = true;
                gotEvent = true;
            }
            else {
                /* checked notification, but it's not desired event */
                newEvent[i] = false;
            }
        }
        /* desired event is set by LwSciIpcGetEvent() ? otherwise,
         * previous WaitForMultipleEvents() call returns both event type ?
         */
        if (gotEvent) {
            return LwSciError_Success;
        }
        /* LwSciEventService */
        if (tp->evtSvcFlag) {
            uint32_t end;
            uint32_t bitEvent;

            /* msec to usec */
            timeout = (tp->timeout >= 0)?(tp->timeout*1000):tp->timeout;

            err = eventLoopServiceP->WaitForMultipleEvents(
                tp->evtNotiArray, tp->evtNotiCnt, timeout, newEvent);
            if (err != LwSciError_Success) {
                printf("[%20s] %s: WaitForMultipleEvents err: %d\n",
                    tp->prgname, __func__, err);
                return err;
            }

            if (tp->epCnt > 0) {    /* has Native Event ? */
                /* store Native Event statistics */
                end = tp->epCnt;

                for(i=0;i<end;i++) {
                    if (newEvent[i] == false)
                        continue;
                    if (dir == RX_ROLE)
                        tp->TotalRxEvtCnt++;
                    else
                        tp->TotalTxEvtCnt++;
                }
            }

            if (s_verboseFlag) {
                bitEvent = event2value(newEvent, tp->evtNotiCnt);
            }

            /* no endpoint or no RX native event */
            dprintf("[%20s] %s: WaitForMultipleEvents"
                "(native only): 0x%08x\n",
                tp->prgname, __func__, bitEvent);
        }
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
            if (dir == RX_ROLE) {
                err = LwMsgReceivePulse(&tp->rx[0], TEST_EVENT_CODE,
                    &pulse, sizeof(pulse), NULL, NULL);
            }
            else {
                err = LwMsgReceivePulse(&tp->tx[0], TEST_EVENT_CODE,
                    &pulse, sizeof(pulse), NULL, NULL);
            }
            if (err != LwSciError_Success) {
                printf("[%20s] %s: MsgReceivePulse err: %d\n",
                    tp->prgname, __func__, err);
                return err;
            }
            /* pulse code offset is endpoint index */
            i = pulse.code - TEST_EVENT_CODE;
            if ((i < 0) || (i >= MAX_ENDPOINT)) {
                printf("[%20s] %s: pulse code err: %d\n",
                    tp->prgname, __func__, i);
                return EILWAL;
            }

            if (dir == RX_ROLE)
                tp->TotalRxEvtCnt++;  /* count event */
            else
                tp->TotalTxEvtCnt++;  /* count event */

#if LWIPC_DEBUG
            printf("[%20s] %s: code:%x, val:%x\n",
                tp->prgname, __func__, pulse.code, pulse.value.sival_int);
#endif
        }
    }

    return LwSciError_InterruptedCall;
}

static char *get_time_str(uint32_t sec)
{
    static char buf[32];
    uint32_t dd,hh,mm,ss;

    dd = sec / 3600 / 24;
    hh = (sec / 3600) % 24;
    mm = (sec / 60) % 60;
    ss = sec % 60;
    sprintf(buf, "%d-%02d:%02d:%02d", dd, hh, mm, ss);

    return buf;
}

static uint32_t get_sec_from_cycles(uint64_t cycles)
{
    return (cycles / ((double)SYSPAGE_ENTRY(qtime)->cycles_per_sec));
}

/* logging is enough for rx side only */
static void display_stress_progress(int32_t dir, uint64_t cycles)
{
    test_params *tp = &s_params;
    uint32_t sec = get_sec_from_cycles(cycles);
    static uint32_t oldsec = 0xDEADBEEF;

    if (sec != oldsec) {
        if ((tp->logpersec != 0) && (sec % tp->logpersec) == 0) {
            if (dir == RX_ROLE) {
                printf("%20s:%s:%s:RD%5ld/%5ld:time(%s)\n",
                    tp->prgname,
                    tp->rx[0].chname,
                    tp->rx[tp->epCnt-1].chname,
                    tp->TotalRxFrmCnt, (tp->iterations * tp->epCnt),
                    get_time_str(sec));
            }
            else {
                printf("%20s:%s:%s:WR%5ld/%5ld:time(%s)\n",
                    tp->prgname,
                    tp->tx[0].chname,
                    tp->tx[tp->epCnt-1].chname,
                    tp->TotalTxFrmCnt, (tp->iterations * tp->epCnt),
                    get_time_str(sec));
            }
        }
        oldsec = sec;
    }
}

/* statistics is enough for rx side only */
static void display_stress_statistics(int32_t dir, uint64_t cycles)
{
    test_params *tp = &s_params;
    uint32_t sec = get_sec_from_cycles(cycles);

    if (dir == RX_ROLE) {
        printf("[%4s]%20s:%s:%s:nf%d:fs%d:%s:RD%5ld/%5ld:ERR%d:"
            "rate%.2lfMB/s:time(%s)\n",
            (tp->rxerrcnt > 0)?"FAIL":"PASS",
            tp->prgname,
            tp->rx[0].chname,
            tp->rx[tp->epCnt-1].chname,
            tp->rx[0].info.nframes,
            tp->rx[0].info.frame_size,
            (tp->evtSvcFlag)?"LWSCIEVT":"NATIVE",
            tp->TotalRxFrmCnt, (tp->iterations * tp->epCnt),
            tp->rxerrcnt,
            (double)(tp->TotalRxFrmCnt * tp->rx[0].info.frame_size / sec) / MEGABYTE,
            get_time_str(sec));
    }
    else {
        printf("[%4s]%20s:%s:%s:nf%d:fs%d:%s:WR%5ld/%5ld:ERR%d:"
            "rate%.2lfMB/s:time(%s)\n",
            (tp->txerrcnt > 0)?"FAIL":"PASS",
            tp->prgname,
            tp->tx[0].chname,
            tp->tx[tp->epCnt-1].chname,
            tp->tx[0].info.nframes,
            tp->tx[0].info.frame_size,
            (tp->evtSvcFlag)?"LWSCIEVT":"NATIVE",
            tp->TotalTxFrmCnt, (tp->iterations * tp->epCnt),
            tp->txerrcnt,
            (double)(tp->TotalTxFrmCnt * tp->tx[0].info.frame_size / sec) / MEGABYTE,
            get_time_str(sec));
    }
}

static void print_uid_gid(char *str)
{
    dprintf("[%s] UID: %d, GID: %d\n", str, getuid(), getgid());
}
#endif /* QNX */

static void release_endpoint_resource(struct endpoint *ep)
{
    test_params *tp = &s_params;

    if (ep->buf) {
        free(ep->buf);
        ep->buf = NULL;
    }
    if (ep->rbuf) {
        free(ep->rbuf);
        ep->rbuf = NULL;
    }
    if (ep->sbuf) {
        free(ep->sbuf);
        ep->sbuf = NULL;
    }

    if (tp->evtSvcFlag && ep->eventNotifier != NULL) {
        ep->eventNotifier->Delete(ep->eventNotifier);
    }

    if (ep->openFlag) {
        LwSciIpcCloseEndpoint(ep->h);
        ep->openFlag = false;
    }

#ifdef __QNX__
    if (ep->coid != 0) {
        (void)ConnectDetach_r(ep->coid);
        ep->coid = 0;
    }
    if (ep->chid != 0) {
        (void)ChannelDestroy_r(ep->chid);
        ep->chid = 0;
    }

    if (tp->evtSvcFlag && ep->eventLoopServiceP != NULL) {
        ep->eventLoopServiceP->EventService.Delete(
            &ep->eventLoopServiceP->EventService);
    }
#endif /* __QNX__ */
}

static void release_resources(void)
{
    test_params *tp = (test_params *)&s_params;

    if (tp->initFlag) {
        LwSciIpcDeinit();
        tp->initFlag = false;
    }

    if (s_Stop) {
        exit(1);
    }
}

static void sig_handler(int sig_num)
{
    s_Stop = 1;

    pthread_cancel(s_params.tx[0].tid);
    pthread_cancel(s_params.rx[0].tid);
    pthread_join(s_params.tx[0].tid, NULL);
    pthread_join(s_params.rx[0].tid, NULL);

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

static LwSciError init_resources(struct endpoint *ep)
{
#ifdef __QNX__
    test_params *tp = (test_params *)&s_params;
    int32_t chid;
    int32_t coid;
#endif /* __QNX__ */
#ifdef LINUX
    int32_t fd;
#endif
    LwSciError ret;

    printf("[T:%d] opening LwSciIpc endpoint: %s\n", (int)ep->tid, ep->chname);
#ifdef __QNX__
    if (tp->evtSvcFlag) {
        dprintf("[T:%d]%s:LwSciEventLoopServiceCreate\n",
            (int)ep->tid, __func__);
        ret = LwSciEventLoopServiceCreate(1, &ep->eventLoopServiceP);
        if (ret != LwSciError_Success) {
            printf("[T:%d]%s: LwSciEventLoopServiceCreate fail (%d)\n",
                (int)ep->tid, __func__, ret);
            return ret;
        }
    }

    if (tp->evtSvcFlag && !tp->bindEvtSvcFlag) {
            dprintf("[T:%d]%s:LwSciIpcOpenEndpointWithEventService\n", ep->tid, __func__);
            ret = LwSciIpcOpenEndpointWithEventService(ep->chname, &ep->h,
                    &ep->eventLoopServiceP->EventService);
    }
    else
#endif /* __QNX__ */
    {
        ret = LwSciIpcOpenEndpoint(ep->chname, &ep->h);
    }
    if (ret != LwSciError_Success) {
        printf("[T:%d]%s:LwSciIpcOpenEndpoint: fail (%d)\n", (int)ep->tid, __func__, ret);
        return ret;
    }
    ep->openFlag = true;
    dprintf("[T:%d]%s: endpoint handle: 0x%lx\n", (int)ep->tid, __func__, ep->h);

#ifdef __QNX__
    if (tp->evtSvcFlag && tp->bindEvtSvcFlag) {
        ret = LwSciIpcBindEventService(ep->h,
            &ep->eventLoopServiceP->EventService);
        if (ret != LwSciError_Success) {
            printf("[T:%d]%s:LwSciIpcBindEventService: fail (%d)\n", (int)ep->tid, __func__, ret);
            return ret;
        }
    }
#endif /* __QNX__ */

#ifdef LINUX
    ret = LwSciIpcGetLinuxEventFd(ep->h, &fd);
    if (ret != LwSciError_Success) {
        printf("%s:LwSciIpcGetLinuxEventFd: fail (%d)\n", __func__, ret);
        return ret;
    }
    printf("%s:LwSciIpcGetLinuxEventFd: fd(%d)\n", __func__, fd);

    ep->fd = fd;
#endif

#ifdef __QNX__
    if (tp->evtSvcFlag) {
        dprintf("[T:%d]%s:LwSciIpcGetEventNotifier\n", ep->tid, __func__);
        ret = LwSciIpcGetEventNotifier(ep->h, &ep->eventNotifier);
        if (ret != LwSciError_Success) {
            printf("[T:%d]%s: LwSciIpcGetEventNotifier: fail (%d)\n",
                ep->tid, __func__, ret);
            return ret;
        }
        /* fill event notifier array for native event */
        tp->evtNotiArray[tp->evtNotiCnt++] = ep->eventNotifier;
    }
    else {
        chid = ChannelCreate_r(0U);
        if (chid < 0) {
            ret = LwSciIpcErrnoToLwSciErr(chid);
            printf("[T:%d]%s:ChannelCreate_r: fail (%d)\n", ep->tid, __func__, ret);
            return ret;
        }
        ep->chid = chid;
        coid = ConnectAttach_r(0, 0, chid, _NTO_SIDE_CHANNEL, 0);
        if (coid < 0) {
            ret = LwSciIpcErrnoToLwSciErr(coid);
            printf("[T:%d]%s:ConnectAttach_r: fail (%d)\n", ep->tid, __func__, ret);
            return ret;
        }
        ep->coid = coid;
        dprintf("[T:%d]%s: chid:%d,coid:%d\n", ep->tid, __func__, chid, coid);

        ret = LwSciIpcSetQnxPulseParam(ep->h, ep->coid, tp->priority,
            TEST_EVENT_CODE, (void *)NULL);
        if (ret != LwSciError_Success) {
            printf("[T:%d]%s:LwSciIpcSetQnxPulseParam: fail (%d)\n", ep->tid, __func__, ret);
            return ret;
        }
    }
#endif /* __QNX__ */

    ret = LwSciIpcGetEndpointInfo(ep->h, &ep->info);
    if (ret != LwSciError_Success) {
        printf("[T:%d]%s:LwSciIpcGetEndpointInfo: fail (%d)\n",
                (int)ep->tid, __func__, ret);
        return ret;
    }
    printf("[T:%d] endpoint_info: nframes = %d, frame_size = %d\n",
        (int)ep->tid, ep->info.nframes, ep->info.frame_size);

    ep->buf = calloc(1, ep->info.frame_size);
    if (ep->buf == NULL) {
        printf("[T:%d]%s: Failed to allocate buffer of size %u\n",
                (int)ep->tid, __func__, ep->info.frame_size);
        ret = LwSciIpcErrnoToLwSciErr(errno);
        return ret;
    }

    ep->sbuf = calloc(1, ep->info.frame_size);
    if (ep->sbuf == NULL) {
        printf("[T:%d]%s: Failed to allocate buffer of size %u\n",
                (int)ep->tid, __func__, ep->info.frame_size);
        ret = LwSciIpcErrnoToLwSciErr(errno);
        return ret;
    }

    ep->rbuf = calloc(1, ep->info.frame_size);
    if (ep->rbuf == NULL) {
        printf("[T:%d]%s: Failed to allocate buffer of size %u\n",
                (int)ep->tid, __func__, ep->info.frame_size);
        ret = LwSciIpcErrnoToLwSciErr(errno);
        return ret;
    }
    return LwSciError_Success;
}

#ifdef __QNX__
static LwSciError init_stress_resources(struct endpoint *ep,
    int32_t idx, int32_t tid, int32_t coid, LwSciEventService *eventService)
{
    test_params *tp = (test_params *)&s_params;
    LwSciError ret;

    dprintf("[%d:%d]%s:opening LwSciIpc endpoint: %s\n",
        tp->pid, tp->rtid, __func__, ep->chname);
    if (tp->evtSvcFlag && !tp->bindEvtSvcFlag) {
        dprintf("[%d:%d]%s:LwSciIpcOpenEndpointWithEventService\n",
            tp->pid, tid, __func__);
        ret = LwSciIpcOpenEndpointWithEventService(ep->chname, &ep->h,
                eventService);
    }
    else {
        ret = LwSciIpcOpenEndpoint(ep->chname, &ep->h);
    }
    if (ret != LwSciError_Success) {
        printf("[%d:%d]%s:LwSciIpcOpenEndpoint: fail (%d)\n",
            tp->pid, tid, __func__, ret);
        return ret;
    }
    ep->prevData = 0xDEADBEEF;
    ep->openFlag = true;
    dprintf("[%d:%d]%s:handle:0x%lx\n", tp->pid, tid, __func__, ep->h);

    if (tp->evtSvcFlag && tp->bindEvtSvcFlag) {
        ret = LwSciIpcBindEventService(ep->h,
            &ep->eventLoopServiceP->EventService);
        if (ret != LwSciError_Success) {
            printf("[T:%d]%s:LwSciIpcBindEventService: fail (%d)\n", (int)ep->tid, __func__, ret);
            return ret;
        }
    }

    if (tp->evtSvcFlag) {
        dprintf("[%d:%d]%s:LwSciIpcGetEventNotifier\n",
            tp->pid, tid, __func__);
        ret = LwSciIpcGetEventNotifier(ep->h, &ep->eventNotifier);
        if (ret != LwSciError_Success) {
            printf("[%d:%d]%s: LwSciIpcGetEventNotifier: fail (%d)\n",
                tp->pid, tid, __func__, ret);
            return ret;
        }
    }
    else {
        ret = LwSciIpcSetQnxPulseParam(ep->h, coid, tp->priority,
            (TEST_EVENT_CODE + idx), (void *)NULL);
        if (ret != LwSciError_Success) {
            printf("[%d:%d]%s:LwSciIpcSetQnxPulseParam: fail (%d)\n",
                tp->pid, tid, __func__, ret);
            return ret;
        }
    }

    ret = LwSciIpcGetEndpointInfo(ep->h, &ep->info);
    if (ret != LwSciError_Success) {
        printf("[%d:%d]%s:LwSciIpcGetEndpointInfo: fail (%d)\n",
            tp->pid, tid, __func__, ret);
        return ret;
    }
    dprintf("[%d:%d] endpoint_info: nframes = %d, frame_size = %d\n",
        tp->pid, tid, ep->info.nframes, ep->info.frame_size);

    ep->buf = calloc(1, ep->info.frame_size);
    if (ep->buf == NULL) {
        printf("[%d:%d]%s:Failed to allocate buffer of size %u\n",
            tp->pid, tid, __func__, ep->info.frame_size);
        ret = LwSciIpcErrnoToLwSciErr(errno);
        return ret;
    }

    LwSciIpcResetEndpoint(ep->h);

    return LwSciError_Success;
}
#endif /* __QNX__ */

#ifdef LINUX
/* Send thread */
static void *send_msg(void *arg)
{
    test_params *tp = (test_params *)arg;
    struct endpoint *ep = (struct endpoint *)&tp->tx;
    uint64_t iterations = tp->iterations;
    uint32_t events;
    uint32_t *ptr;
    int32_t bytes;
    uint64_t count;
    LwSciError ret;

    ret = init_resources(ep);
    if (ret != LwSciError_Success) {
        tp->errcnt++;
        return NULL;
    }

    LwSciIpcResetEndpoint(ep->h);

    count = 0;
    ptr = (uint32_t *)ep->buf;
    while ((count < iterations) && !s_Stop) {
        ret = wait_event(ep, LW_SCI_IPC_EVENT_WRITE, &events);
        if (ret != LwSciError_Success) {
            printf("%s: write: wait_event: %d\n", __func__, ret);
            break;
        }

        ptr[0] = count;
        ret = LwSciIpcWrite(ep->h, ep->buf, ep->info.frame_size, &bytes);
        if (ret != LwSciError_Success) {
            printf("%s: write: %d\n", __func__, ret);
            continue;
        }
        count++;
    }

    printf("%s: send complete\n", __func__);

    release_endpoint_resource(ep);

    return NULL;
}

/* Recv thread */
static void *recv_msg(void *arg)
{
    test_params *tp = (test_params *)arg;
    struct endpoint *ep = (struct endpoint *)&tp->rx;
    uint64_t iterations = tp->iterations;
    uint32_t events;
    uint64_t count = 0;
    uint32_t *ptr;
    int32_t bytes;
    LwSciError ret;

    ret = init_resources(ep);
    if (ret != LwSciError_Success) {
        tp->errcnt++;
        return NULL;
    }

    LwSciIpcResetEndpoint(ep->h);

    /* Start reading */
    count = 0;
    ptr = (uint32_t *)ep->buf;
    while ((count < iterations) && !s_Stop) {
        ret = wait_event(ep, LW_SCI_IPC_EVENT_READ, &events);
        if (ret != LwSciError_Success) {
            printf("%s: read: wait_event: %d\n", __func__, ret);
            break;
        }

        ret = LwSciIpcRead(ep->h, ep->buf, ep->info.frame_size, &bytes);
        if (ret != LwSciError_Success) {
            printf("%s: read: %d\n", __func__, ret);
            continue;
        }
        if (ptr[0] != count)
            printf("mismatch buf %x != exp %x\n", ptr[0], (uint32_t)count);
        count++;
    }
    printf("%s: recv complete\n", __func__);

    release_endpoint_resource(ep);

    return NULL;
}
#endif /* LINUX */

#ifdef __QNX__
static char *get_dir(struct endpoint *ep)
{
    switch(ep->role) {
        case RX_ROLE:
            return "RX";
        case TX_ROLE:
            return "TX";
    }

    return "--";
}

static void *proc_msg(void *arg)
{
    struct endpoint *ep = (struct endpoint *)arg;
    test_params *tp = (test_params *)&s_params;
    uint64_t iterations;
    uint64_t count = 0UL;
    uint32_t *rptr = NULL;
    uint32_t *wptr = NULL;
    uint64_t rd_count = 0UL;
    uint64_t wr_count = 0UL;
    uint64_t rd_err_cnt = 0UL;
    uint64_t wr_err_cnt = 0UL;
    uint64_t start = 0UL;
    uint64_t end;
    uint32_t events = 0U;
    int32_t bytes;
    uint32_t event_mask = 0U;
    int32_t weight;
    char str[80];
    LwSciError ret;
    double rate;

    ep->tid = gettid();
    sprintf(str, "%s-%s", __func__, get_dir(ep));
    print_uid_gid(str);

    ret = init_resources(ep);
    if (ret != LwSciError_Success) {
        tp->errcnt++;
        pthread_cancel(ep->peertid);
        return NULL;
    }

    LwSciIpcResetEndpoint(ep->h);

    /* set event mask */
    if (tp->bidirtest) {
        /* bi-directional test */
        event_mask = (LW_SCI_IPC_EVENT_READ|LW_SCI_IPC_EVENT_WRITE);
        rptr = (uint32_t *)ep->rbuf;
        wptr = (uint32_t *)ep->sbuf;
        weight = 2;
    }
    else {
        /* uni-directional test */
        if (ep->role == RX_ROLE) {
            event_mask = LW_SCI_IPC_EVENT_READ;
            rptr = (uint32_t *)ep->rbuf;
        }
        else {
            event_mask = LW_SCI_IPC_EVENT_WRITE;
            wptr = (uint32_t *)ep->sbuf;
        }
        weight = 1;
    }
    if (tp->connrcvtest) {
        /* use different iterations for connection recovery test */
        iterations = ep->iterations;
    } else {
        iterations = tp->iterations * weight;
    }

    count = 0;
    rd_count = 0;
    wr_count = 0;
    while ((count < iterations) && !s_Stop) {
        ret = wait_event(ep, event_mask, &events);
        if (ret != LwSciError_Success) {
            rd_err_cnt++;
            wr_err_cnt++;
            break;
        }

        if (start == 0UL) {
            start = ClockCycles();
        }

        if (events & LW_SCI_IPC_EVENT_READ) {
            ret = LwSciIpcRead(ep->h, ep->rbuf, ep->info.frame_size, &bytes);
            if (ret != LwSciError_Success) {
                rd_err_cnt++;
                break;
            }
            if ((rptr[0] != rd_count) && (tp->connrcvtest == 0)) {
                dprintf("[T:%d]%s: mismatch (rx: %d, expected: %ld)\n",
                    ep->tid, get_dir(ep), rptr[0], rd_count);
                rd_err_cnt++;
            }
            rd_count++;
            nanospin_ns(tp->r_backoff_delay);
        }
        if (events & LW_SCI_IPC_EVENT_WRITE) {
            wptr[0] = wr_count;
            wptr[1] = 0;
            if (wr_count < tp->iterations) {
                ret = LwSciIpcWrite(ep->h, ep->sbuf, ep->info.frame_size, &bytes);
                if (ret != LwSciError_Success) {
                    wr_err_cnt++;
                    break;
                }
                wr_count++;
                nanospin_ns(tp->w_backoff_delay);
            }
        }
        count = rd_count + wr_count;
    }
    end = ClockCycles();

    printf("[T:%d]%s: count: %"PRId64", rd_cnt: %"PRId64", wr_cnt: %"PRId64
            ", rd_err_cnt: %"PRId64", wr_err_cnt: %"PRId64"\n",
            ep->tid, get_dir(ep), count, rd_count, wr_count, rd_err_cnt, wr_err_cnt);
    dprintf("[T:%d]%s: Start = %lu, End = %lu\n", ep->tid, get_dir(ep), start, end);
    rate = (((double)(count * ep->info.frame_size)) *
            ((double)SYSPAGE_ENTRY(qtime)->cycles_per_sec / MEGABYTE)) /
            (end - start);
    printf("[T:%d]%s: Rate %lf MB/sec (%lf Mfps; %lf us/f)\n", ep->tid, get_dir(ep),
            rate, (rate * MEGABYTE)/(double)(ep->info.frame_size * 1000000),
            (double)(ep->info.frame_size * 1000000)/(rate * MEGABYTE));
    dprintf("[T:%d]%s: event_count: %u\n", ep->tid, get_dir(ep), ep->evt_cnt);

    /* set fail condition */
    if (rd_err_cnt != 0UL || wr_err_cnt != 0UL) {
        tp->errcnt++;
    }

    release_endpoint_resource(ep);

    return NULL;
}

static void *stress_recv_msg(void *arg)
{
    test_params *tp = (test_params *)&s_params;
    struct endpoint *ep;
    uint32_t *ptr;
    uint64_t iterations;
    uint64_t cstart = 0UL;
    uint64_t cend = 0UL;
    uint64_t end;
    uint32_t i;
    int32_t bytes;
    int32_t pid, tid;
    int32_t chid, coid;
    bool newEvent[MAX_EVENTNOTIFIER];
    LwSciError err;

    usleep(3000);
    pid = tp->pid;
    tid = gettid();

    if (tp->evtSvcFlag) {   /* LwSciEventService */
        dprintf("[%d:%d]%s:LwSciEventLoopServiceCreate\n",
            pid, tid, __func__);
        err = LwSciEventLoopServiceCreate(1, &tp->rx[0].eventLoopServiceP);
        if (err != LwSciError_Success) {
            printf("[%d:%d]%s:LwSciEventLoopServiceCreate fail (%d)\n",
                pid, tid, __func__, err);
            goto rfail;
        }
    }
    else {  /* native event handling */
        /* create chid/coid */
        chid = ChannelCreate_r(0U);
        if (chid < 0) {
            printf("[%d:%d]%s:ChannelCreate_r(err:%d)\n",
                pid, tid, __func__, chid);
            goto rfail;
        }
        tp->rx[0].chid = chid;
        coid = ConnectAttach_r(0, 0, chid, _NTO_SIDE_CHANNEL, 0);
        if (coid < 0) {
            printf("[%d:%d]%s:ConnectAttach_r(err:%d)\n",
                pid, tid, __func__, coid);
            goto rfail;
        }
        tp->rx[0].coid = coid;
        dprintf("[%d:%d]%s:chid:%d,coid:%d\n", pid, tid, __func__, chid, coid);
    }

    /* open and reset endpoint */
    for (i=0;i<tp->epCnt;i++) {
        /* make endpoint name */
        sprintf(tp->rx[i].chname, "%s%04d", tp->rxprefix, (tp->epstart + i));
        err = init_stress_resources(&tp->rx[i], i, tp->rtid, tp->rx[0].coid,
            &tp->rx[0].eventLoopServiceP->EventService);
        if (err != LwSciError_Success) {
            tp->rxerrcnt++;
            break;
        }
    }

    if (tp->rxerrcnt > 0) {
        tp->rxerrcnt++;
        pthread_cancel(tp->ttid);
        goto rfail2;
    }

    iterations = (tp->epCnt * tp->iterations);

    while ((tp->TotalRxFrmCnt < iterations) && !s_Stop) {
        err = wait_multi_events(RX_ROLE, LW_SCI_IPC_EVENT_READ,
            tp->rx[0].eventLoopServiceP, newEvent);

        if (cstart == 0ULL) {
            cstart = ClockCycles();
        }

        /* uni-directional test */
        if (err == LwSciError_Success) {
            if (tp->evtSvcFlag) {
                end = tp->evtNotiCnt;
            }
            else {
                end = tp->epCnt;
            }

            for (i=0;i<end;i++) {
                if (newEvent[i] == false)
                    continue;
                ep = &tp->rx[i];
                ptr = (uint32_t *)ep->buf;

                err = LwSciIpcRead(ep->h, ep->buf,
                    ep->info.frame_size, &bytes);
                if (err != LwSciError_Success) {
                    tp->rxerrcnt++;
                    break;
                }
                if ((ep->prevData != 0xDEADBEEF) &&
                    (ptr[0] != (ep->prevData+1))) {
                    tp->rxerrcnt++;   /* data mismatch error */
                    printf("[%d:%d]%s:RD#%d:%d,%d(%s)\n",
                        pid, tid, __func__, ep->rtxCnt, ptr[0],
                        ep->prevData, ep->chname);
                }
                ep->rtxCnt++;
                dprintf("[%d:%d]%s:RD#%d:%d,%d(%s)\n",
                    pid, tid, __func__, ep->rtxCnt, ptr[0],
                    ep->prevData, ep->chname);
                ep->prevData = ptr[0];
                tp->TotalRxFrmCnt++;
            }
        }
        else {
            printf("[%d:%d]%s:wait_multi_events() error: %d\n",
                pid, tid, __func__, err);
            tp->rxerrcnt++;
            break;
        }

        cend = ClockCycles();
        display_stress_progress(RX_ROLE, cend-cstart);
    }

    /* Show statistics */
    display_stress_statistics(RX_ROLE, cend-cstart);

rfail2:
    for (i=0;i<tp->epCnt;i++) {
        release_endpoint_resource(&tp->rx[i]);
    }
    if (tp->rcoid != 0) {
        (void)ConnectDetach_r(tp->rcoid);
        tp->rcoid = 0;
    }
    if (tp->rchid != 0) {
        (void)ChannelDestroy_r(tp->rchid);
        tp->rchid = 0;
    }

rfail:
    /* set fail condition */
    if (tp->rxerrcnt > 0UL) {
        tp->errcnt++;
    }

    return NULL;
}

static void *stress_send_msg(void *arg)
{
    test_params *tp = (test_params *)&s_params;
    struct endpoint *ep;
    uint32_t *ptr;
    uint64_t iterations;
    uint64_t cstart = 0UL;
    uint64_t cend = 0UL;
    uint64_t end;
    uint32_t i;
    int32_t bytes;
    int32_t pid, tid;
    int32_t chid, coid;
    bool newEvent[MAX_EVENTNOTIFIER];
    LwSciError err;

    usleep(3000);
    pid = tp->pid;
    tid = gettid();

    if (tp->evtSvcFlag) { /* LwSciEventService */
        dprintf("[%d:%d]%s:LwSciEventLoopServiceCreate\n",
            pid, tid, __func__);
        err = LwSciEventLoopServiceCreate(1, &tp->tx[0].eventLoopServiceP);
        if (err != LwSciError_Success) {
            printf("[%d:%d]%s:LwSciEventLoopServiceCreate fail (%d)\n",
                pid, tid, __func__, err);
            goto tfail;
        }
    }
    else {  /* native event handling */
        /* create chid/coid */
        chid = ChannelCreate_r(0U);
        if (chid < 0) {
            printf("[%d:%d]%s:ChannelCreate_r(err:%d)\n",
                pid, tid, __func__, chid);
            goto tfail;
        }
        tp->tx[0].chid = chid;
        coid = ConnectAttach_r(0, 0, chid, _NTO_SIDE_CHANNEL, 0);
        if (coid < 0) {
            printf("[%d:%d]%s:ConnectAttach_r(err:%d)\n",
                pid, tid, __func__, coid);
            goto tfail;
        }
        tp->tx[0].coid = coid;
        dprintf("[%d:%d]%s:chid:%d,coid:%d\n", pid, tid, __func__, chid, coid);
    }

    /* open and reset endpoint */
    for (i=0;i<tp->epCnt;i++) {
        sprintf(tp->tx[i].chname, "%s%04d", tp->txprefix, tp->epstart + i);
        err = init_stress_resources(&tp->tx[i], i, tp->ttid, tp->tx[0].coid,
            &tp->tx[0].eventLoopServiceP->EventService);
        if (err != LwSciError_Success) {
            tp->txerrcnt++;
            break;
        }
    }

    if (tp->txerrcnt > 0) {
        tp->txerrcnt++;
        pthread_cancel(tp->rtid);
        goto tfail2;
    }

    iterations = (tp->epCnt * tp->iterations);

    delay(tp->start_delay);
    while ((tp->TotalTxFrmCnt < iterations) && !s_Stop) {

        memset(newEvent, 0, sizeof(newEvent));
        err = wait_multi_events(TX_ROLE, LW_SCI_IPC_EVENT_WRITE,
            tp->tx[0].eventLoopServiceP, newEvent);

        if (cstart == 0ULL) {
            cstart = ClockCycles();
        }

        /* uni-directional test */
        if (err == LwSciError_Success) {
            if (tp->evtSvcFlag) {
                end = tp->evtNotiCnt;
            }
            else {
                end = tp->epCnt;
            }

            for (i=0;i<end;i++) {
                if (newEvent[i] == false)
                    continue;

                ep = &tp->tx[i];
                if (ep->rtxCnt >= tp->iterations)
                    continue;

                ptr = (uint32_t *)ep->buf;
                ptr[0] = ep->rtxCnt;

                err = LwSciIpcWrite(ep->h, ep->buf,
                    ep->info.frame_size, &bytes);
                if (err != LwSciError_Success) {
                    tp->txerrcnt++;
                    break;
                }
                ep->rtxCnt++;
                dprintf("[%d:%d]%s:WR#%d:%d(%s)\n",
                    pid, tid, __func__, ep->rtxCnt, ptr[0], ep->chname);
                tp->TotalTxFrmCnt++;
            }
            if (tp->repeat_delay > 0)
                delay(tp->repeat_delay);
        }
        else {
            printf("[%d:%d]%s:wait_multi_events() error: %d\n",
                pid, tid, __func__, err);
            tp->txerrcnt++;
            break;
        }

        cend = ClockCycles();
        display_stress_progress(TX_ROLE, cend-cstart);
    }

    display_stress_statistics(TX_ROLE, cend-cstart);

tfail2:
    for (i=0;i<tp->epCnt;i++) {
        release_endpoint_resource(&tp->tx[i]);
    }
    if (tp->tcoid != 0) {
        (void)ConnectDetach_r(tp->tcoid);
        tp->tcoid = 0;
    }
    if (tp->tchid != 0) {
        (void)ChannelDestroy_r(tp->tchid);
        tp->tchid = 0;
    }

tfail:
    /* set fail condition */
    if (tp->txerrcnt > 0UL) {
        tp->errcnt++;
    }

    return NULL;
}

static void normal_test(struct endpoint *rx, struct endpoint *tx)
{
    test_params *tp = (test_params *)&s_params;

    if (tp->connrcvtest) {
        rx->iterations = 15;
        tx->iterations = 5;
    }
    rx->role = RX_ROLE;
    pthread_create(&rx->tid, NULL, &proc_msg, rx);
    mprintf("created Rx thread (tid:%u)\n", (uint32_t)rx->tid);
    tx->role = TX_ROLE;
    pthread_create(&tx->tid, NULL, &proc_msg, tx);
    mprintf("created Tx thread (tid:%u)\n", (uint32_t)tx->tid);
    rx->peertid = tx->tid;
    tx->peertid = rx->tid;
    /* wait completion of threads */
    if (tx->tid != 0 && tp->errcnt == 0) {
        pthread_join(tx->tid, NULL);
        mprintf("terminated Tx thread (tid:%u)\n",
            (uint32_t)tx->tid);
    }
    if (tp->connrcvtest && tp->errcnt == 0) {
        /* connection recovery test only */
        pthread_create(&tx->tid, NULL, &proc_msg, tx);
        mprintf("created Tx thread (tid:%u)\n", (uint32_t)tx->tid);
        /* wait completion of threads */
        if (tx->tid != 0) {
            pthread_join(tx->tid, NULL);
            mprintf("terminated Tx thread (tid:%u)\n", (uint32_t)tx->tid);
        }

        pthread_create(&tx->tid, NULL, &proc_msg, tx);
        mprintf("created Tx thread (tid:%u)\n", (uint32_t)tx->tid);
        /* wait completion of threads */
        if (tx->tid != 0) {
            pthread_join(tx->tid, NULL);
            mprintf("terminated Tx thread (tid:%u)\n", (uint32_t)tx->tid);
        }
    }
    if (rx->tid != 0 && tp->errcnt == 0) {
        pthread_join(rx->tid, NULL);
        mprintf("terminated Rx thread (tid:%u)\n",
            (uint32_t)rx->tid);
    }

    if (tp->errcnt != 0) {
        pthread_join(tx->tid, NULL);
        mprintf("terminated Tx thread (tid:%u)\n",
            (uint32_t)tx->tid);
        pthread_join(rx->tid, NULL);
        mprintf("terminated Rx thread (tid:%u)\n",
            (uint32_t)rx->tid);
    }
}

static void stress_test(void)
{
    test_params *tp = (test_params *)&s_params;

    tp->pid = getpid();

    pthread_create(&tp->rtid, NULL, &stress_recv_msg, NULL);
    mprintf("created Rx thread (tid:%u)\n", (uint32_t)tp->rtid);
    pthread_create(&tp->ttid, NULL, &stress_send_msg, NULL);
    mprintf("created Tx thread (tid:%u)\n", (uint32_t)tp->ttid);

    /* wait completion of threads */
    if (tp->ttid != 0 && tp->errcnt == 0) {
        pthread_join(tp->ttid, NULL);
        mprintf("terminated Tx thread (tid:%u)\n",
            (uint32_t)tp->ttid);
    }
    if (tp->rtid != 0 && tp->errcnt == 0) {
        pthread_join(tp->rtid, NULL);
        mprintf("terminated Rx thread (tid:%u)\n",
            (uint32_t)tp->rtid);
    }

    if (tp->errcnt != 0) {
        pthread_join(tp->ttid, NULL);
        mprintf("terminated Tx thread (tid:%u)\n",
            (uint32_t)tp->ttid);
        pthread_join(tp->rtid, NULL);
        mprintf("terminated Rx thread (tid:%u)\n",
            (uint32_t)tp->rtid);
    }
}
#endif /* QNX */

static void print_usage(const char *str)
{
    printf("%s: Options\n", str);
    printf(" -s <endpoint name>: sender endpoint name\n");
    printf(" -r <endpoint name>: receiver endpoint name\n");
    printf(" -u <uid:gid>      : UID and GID setting for test\n");
    printf(" -v                : verbose mode\n");
    printf(" -B                 : "
            "use LwSciIpcBindEventService when EventService is used\n");
    printf(" -E <msec>         : use LwSciEventService with msec timeout\n"
           "                     wait infinitely if msec is U (-E U)\n");
#ifdef __QNX__
    printf(" -t <start delay>  : start delay(ms) in tx thread\n");
    printf(" -w <repeat delay> : repeat delay(ms) in tx thread\n");
    printf(" -a <lo:hi>        : set lo/hi limit on range of ability (hexadecimal)\n");
    printf(" -b                : bi-directional test\n");
    printf(" -c                : connection recovery test\n");
    printf(" -i <priority>     : pulse priority (default: -1, INTR: 21)\n");
    printf(" -R <ns>           : nanospin count between each read iteration\n");
    printf("                     Default is %lu ns\n", DEFAULT_R_BACKOFF_DELAY);
    printf(" -W <ns>           : nanospin count between each write iteration\n");
    printf("                     Default is %lu ns\n", DEFAULT_W_BACKOFF_DELAY);
    printf(" -S <rxPrefix:txPrefix:start#:end#:logPeriodSec: "
            "stress test\n"
            "\t\tex) \"-S itcrx:itctx:0001:0050:60\" means that do stress test\n"
            "\t\twith TX endpoints (itctx0001 - itctx0050)\n"
            "\t\t and RX endpoints (itcrx0001 - itcrx0050).\n"
            "\t\tprint progress message every 60sec\n");
    printf("-A                 : print start timestamp\n");
    printf("-J [id:INFO_STR]   : "
           "generate test log file with test id, info string and result\n");
    printf("-Z                 : print end timestamp\n");
#endif /* __QNX__ */
    printf(" -l <cnt>          : Loop count\n");
    printf("Example) %s -s tx_endpoint -r rx_endpoint\n\n", str);
    printf(" -Q                : quiet mode (no output message)\n");
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

int32_t main(int32_t argc, char **argv)
{
    test_params *tp = (test_params *)&s_params;
    struct endpoint *tx = &s_params.tx[0];
    struct endpoint *rx = &s_params.rx[0];
    int optcnt = 0;
    int32_t opt;
    int ret;
    LwSciError err;

    tp->iterations = DEFAULT_ITERATIONS;
#ifdef __QNX__
    tp->r_backoff_delay = DEFAULT_R_BACKOFF_DELAY;
    tp->w_backoff_delay = DEFAULT_W_BACKOFF_DELAY;
    tp->priority = SIGEV_PULSE_PRIO_INHERIT;    /* default: -1, INTR: 0x15(21) */
    tp->range_lo = 0;
    tp->range_hi = INT_MAX;
    tp->timeout = LW_SCI_EVENT_INFINITE_WAIT; /* default timeout */
#endif /* __QNX__ */

    while ((opt = getopt(argc, argv, "a:bci:l:r:s:t:u:vw:ABE:J:QR:S:W:Z")) != EOF)
    {
        switch (opt) {
#ifdef __QNX__
            case 'a':
                /* Set ability subrange */
                {
                    char* token = strtok(optarg, ":");
                    if (token != NULL) {
                        tp->range_lo = (int32_t)strtoul(token, NULL, 16);
                        token = strtok(NULL, ":");
                        if (token != NULL) {
                            tp->range_hi = (int32_t)strtoul(token, NULL, 16);
                            ret = 0;
                        }
                    }
                    if (ret != 0) {
                        printf("%s: range_lo/range_hi not specified correctly\n", argv[0]);
                        return -EILWAL;
                    }
                    break;
                }
            case 'b':
                /* bi-directional test */
                tp->bidirtest = 1;
                if (tp->connrcvtest) {
                    printf("can not use -b with -c option\n");
                    return -EILWAL;
                }
                break;
            case 'c':
                /* connection recovery test */
                tp->connrcvtest = 1;
                if (tp->bidirtest) {
                    printf("can not use -b with -c option\n");
                    return -EILWAL;
                }
                break;
            case 'B':   /* use LwSciIpcBindEventService */
                tp->bindEvtSvcFlag = true;
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
            case 't':
                tp->start_delay = strtoul(optarg, NULL, 0);
                break;
            case 'w':
                tp->repeat_delay = strtoul(optarg, NULL, 0);
                break;
            case 'A':
                tp->timestampS = 1;
                break;
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
            case 'Z':
                tp->timestampE = 1;
                break;
#endif /* __QNX__ */
            case 'v':
                s_verboseFlag = true;
                break;
            case 's':
                /* Sender IVC endpoint name */
                memcpy(tx->chname, optarg, strlen(optarg));
                tp->epCnt = 1;
                optcnt++;
                break;
            case 'r':
                /* Receiver IVC endpoint name */
                memcpy(rx->chname, optarg, strlen(optarg));
                tp->epCnt = 1;
                optcnt++;
                break;
            case 'u':
                /* Set UID:GID */
                {
                    char* token = strtok(optarg, ":");
                    if (token != NULL) {
                        tp->uid = (uid_t)strtoul(token, NULL, 10);
                        while ((token = strtok(NULL, ":,")) != NULL) {
                            tp->gid[tp->num_gid] = (gid_t)strtoul(token, NULL, 10);
                            tp->num_gid++;
                            ret = 0;
                        }
                    }
                    if (ret != 0) {
                        printf("%s: UID/GID not specified correctly\n", argv[0]);
                        return -EILWAL;
                    }
                    break;
                }
            case 'l':
                /* loop count */
                tp->iterations = strtoul(optarg, NULL, 0);
                break;
#ifdef __QNX__
            case 'S':
                {
                    char* token = strtok(optarg, ":");

                    if (token != NULL) {
                        strncpy(tp->txprefix, token, sizeof(tp->txprefix));
                    }
                    token = strtok(NULL, ":");
                    if (token != NULL) {
                        strncpy(tp->rxprefix, token, sizeof(tp->rxprefix));
                    }
                    token = strtok(NULL, ":");
                    if (token != NULL) {
                        tp->epstart = (int32_t)strtoul(token, NULL, 10);
                    }
                    token = strtok(NULL, ":");
                    if (token != NULL) {
                        tp->epend = (int32_t)strtoul(token, NULL, 10);
                    }
                    token = strtok(NULL, ":");
                    if (token != NULL) {
                        tp->logpersec = (int32_t)strtoul(token, NULL, 10);
                    }
                    if (tp->epstart > tp->epend) {
                        return -EILWAL;
                    }

                    tp->epCnt = tp->epend - tp->epstart + 1;
                    tp->testtype = TT_STRESS;
                    s_quietFlag = true; /* set quiet flag in stress test */
                    optcnt += 2;
                }
                break;
            case 'R':
                /* nanospin count between each receive iteration */
                tp->r_backoff_delay = strtoul(optarg, NULL, 0);
                break;
            case 'W':
                /* nanospin count between each receive iteration */
                tp->w_backoff_delay = strtoul(optarg, NULL, 0);
                break;
            case 'i':
                /* pulse priority */
                tp->priority = strtoul(optarg, NULL, 0);
                break;
#endif /* __QNX__ */
            case 'Q':
                s_quietFlag = true;
                break;
            default:
                print_usage(argv[0]);
                return -1;
        }
    }
    if (optcnt < 2) {
        print_usage(argv[0]);
        return -1;
    }
    tp->prgname = basename(argv[0]);

    mprintf("[%20s] iteration for Native Event : %ld\n", tp->prgname,
            tp->iterations);
    if (tp->evtSvcFlag) {
        mprintf("[%20s] Use LwSciEventService (timeout:%dmsec)\n",
            tp->prgname, tp->timeout);
    }
    else {
        mprintf("[%20s] MsgReceivePulse (timeout:%dmsec)\n",
            tp->prgname, tp->timeout);
    }

    setup_termination_handlers();

#ifdef __QNX__
    err = drop_privileges();
    if (err != LwSciError_Success) {
        goto fail;
    }
    print_uid_gid(argv[0]);
#endif /* QNX */

#ifdef __QNX__
    if (tp->timestampS == 1) {
        printf("TIMESTAMP(S):%s\n",
            get_time_str(get_sec_from_cycles(ClockCycles())));
    }
#endif /* QNX */

    err = LwSciIpcInit();
    if (err != LwSciError_Success) {
        printf("%s:LwSciIpcInit: fail (%d)\n", __func__, err);
        goto fail;
    }
    tp->initFlag = true;

#ifdef LINUX
    /* create send/receive thread */
    pthread_create(&rx->tid, NULL, &recv_msg, &s_params);
    mprintf("created Rx thread (tid:%u)\n", (uint32_t)rx->tid);
    pthread_create(&tx->tid, NULL, &send_msg, &s_params);
    mprintf("created Tx thread (tid:%u)\n", (uint32_t)tx->tid);

    /* wait completion of threads */
    if (rx->tid != 0) {
        pthread_join(rx->tid, NULL);
    }
    if (tx->tid != 0) {
        pthread_join(tx->tid, NULL);
    }
#endif /* LINUX */

#ifdef __QNX__
    switch (tp->testtype) {
        case TT_NORMAL:
            normal_test(rx, tx);
            break;
        case TT_STRESS:
            stress_test();
            break;
    }
#endif /* QNX */

    if (tp->errcnt != 0) {
        goto fail;
    }
    else {
        mprintf("[%20s] : test PASSED\n", tp->prgname);
        writeTestLog(0);
    }

    release_resources();

#ifdef __QNX__
    if (tp->timestampE == 1) {
        printf("TIMESTAMP(E):%s\n",
            get_time_str(get_sec_from_cycles(ClockCycles())));
    }
#endif /* QNX */

    return EXIT_SUCCESS;

fail:
    release_resources();
    mprintf("[%20s] : test FAILED\n", tp->prgname);
    writeTestLog(1);

    return EXIT_FAILURE;
}

