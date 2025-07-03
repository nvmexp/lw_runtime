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
#include <sys/stat.h>
#include <fcntl.h>

#ifdef __QNX__
#include <sys/neutrino.h>
#include <sys/procmgr.h>
#include <sys/syspage.h>
#include <sys/slog2.h>
#define LWIPC_DEBUG 0
#endif /* __QNX__ */

#include <math.h>
#include <inttypes.h>

#ifdef LINUX
#include <stdint.h>
#include <signal.h>
#include <time.h>
#include <pthread.h>
#endif

#include <lwsciipc_internal.h>
#include "lwsciipc_eventlib/test_lwsciipc_read_eventlib.h"

#define MAX_ENDPOINT    100
#define MEGABYTE 1048576
#define PRIO_INTR 0x15
#define TEST_EVENT_CODE 0x1 /* 1 byte */
#define TIMEOUT_CODE 44
#define DEFAULT_TIMEOUT_MS 500
#define DEFAULT_BACKOFF_NS 100U /* 100ns */
#define MAX_LATENCY_SAMPLE 1000000
#define LATENCY_TIMEOUT_MS 5U /* 5ms */

#define MAX_LOCALEVENT 16
#define MAX_EVENTNOTIFIER (MAX_ENDPOINT+MAX_LOCALEVENT)

/*
 * This test tool supports inter-VM and inter-Process usecase
 */

#define MAX_CHANNEL_NAME_SIZE    128
#define DEFAULT_INIT_COUNT       1
#define DEFAULT_COUNT        128
#define MAX_GID_NUM 255

#ifdef LINUX
    #define EOK 0
#endif

/* Test type */
#define TT_NORMAL  0 /* functionality */
#define TT_PERF    1 /* throughput, latency */
#define TT_STRESS  2 /* parallel process instance, CPU load, memleak */
#define TT_STRESS2 3 /* multiple ilwocatoin of APIs */
#define TT_STRESS3 4 /* parallel exelwtion of APIs */

/* API test type */
#define API_LEGACY      0 /* legacy API set without LwSciEventService */
#define API_LWSCIEVENT  1 /* LwSciEventService API set */

#define LWPLAYFAIR_DATA_HEADER "timestamp,latency\ncycles,nsec\n"
#define NOTIFY_OUTPUT_FILENAME "lwsciipc_notify.csv"
#define END2END_OUTPUT_FILENAME "lwsciipc_end2end.csv"
#define MAX_OUTPUT_PATH_LEN 512
#define DEFAULT_OUTPUT_PATH_NAME "/tmp"

struct endpoint
{
    char chname[LWSCIIPC_MAX_ENDPOINT_NAME];    /* endpoint name */
    LwSciIpcEndpoint h;  /* LwSciIpc handle */
    struct LwSciIpcEndpointInfo info; /* endpoint info */
    uint32_t apitype; /* API type: API_LEGACY(0), API_LWSCIEVENT(1) */
#ifdef LINUX
    int32_t fd;   /* fd to get event */
    fd_set rfds;
#endif /* LINUX */
    void *buf;  /* test buffer */
    void *buf2; /* test buffer */
    uint32_t evt_cnt;
    LwSciEventNotifier *eventNotifier;
    uint32_t prevData;  /* for stress test (comparing data) */
    volatile uint32_t rxEcnt; /* event receive count */
    volatile uint32_t rxCnt; /* frame receive count */

    /* for TT_STRESS3 --> */
    pthread_t tid; /* thread id */
    bool initFlag; /* init flag of endpoint thread */
    LwSciEventLoopService *eventLoopServiceP; /* eventloopservice ptr */
    int32_t chid;   /* channel id to get event */
    int32_t coid;   /* connection id to send event in library */
    int32_t retval; /* result of test thread */
    uint32_t rd_count;  /* RD count */
    uint32_t rd_err_cnt;    /* RD error count */
    bool doneFlag; /* done flag for endpoint thread */
    /* <-- for TT_STRESS3 */
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
    /* it's not same with endpoint limitation of LwSciIpc library */
    struct endpoint ep[MAX_ENDPOINT];
    struct localevent loc[MAX_LOCALEVENT];
    LwSciEventNotifier *evtNotiArray[MAX_EVENTNOTIFIER];
    uint32_t epCnt; /* endpoint count */
    uint32_t locEvtCnt; /* local event object count */
    uint32_t evtNotiCnt; /* event notifier count */
    uint64_t iterations;    /* iterations for endpoint data transaction */
    uint64_t loc_iterations; /* iterations for local event */
    uint64_t initIterations;    /* init/deinit iterations for stress test */
    bool evtSvcFlag; /* API variation test flag: LwSciEventService */
    bool bindEvtSvcFlag; /* API variation test flag: LwSciIpcBindEventService */
    bool allEstablished; /* all endpoints have been established */
    uint32_t ignoredata; /* ignore data mismatch flag */
    uint32_t morerwapi; /* test zero copy, peak/poke API variations */
    char *output_path; /* directory path name to store all latency data as a file */
#ifdef __QNX__
    uint32_t prio; /* pulse priority */
    int32_t range_lo; /* lower limit on range of ability */
    int32_t range_hi; /* upper limit on range of ability */
    uint32_t msgdroptest; /* message drop test flag (notification error) */
    int32_t chid;   /* channel id to get event */
    int32_t coid;   /* connection id to send event in library */
#if (LW_IS_SAFETY == 0) && defined(IVC_EVENTLIB)
    uint32_t eventlib; /* eventlib flag */
#endif
    uint32_t timestampS;    /* print start timestamp */
    uint32_t timestampE;    /* print end timestamp */

    bool outputLog; /* print result to output file */
    uint32_t testid;    /* testid to be printed in outputLog */
    char testinfo[255]; /* test info string to be printed in outputLog */
    bool pause; /* no exit */
#endif /* __QNX__ */

    char epprefix[16];  /* endpoint prefix for stress test */
    uint32_t epstart;   /* start endpoint number for stress test */
    uint32_t epend;     /* end endpoint number for stress test */
    uint32_t logpersec; /* logging period in sec */

    volatile uint64_t TotalEvtCnt;   /* total event count */
    volatile uint64_t TotTxLocEvtCnt;   /* total Tx local event count */
    volatile uint64_t TotRxLocEvtCnt;   /* total Rx local event count */
    volatile uint64_t TotalFrmCnt;   /* total frame count */

    uint32_t bidirtest; /* bi-directional test flag */
    uint32_t backoff_delay; /* ns: for perf */
    uint32_t testtype;  /* TT_NORMAL, TT_PERF, TT_STRESS */
    uint64_t *startTimestamp; /* start timestamp */
    uint64_t *latencyBuf; /* store notification latency (write to noti) */
                          /* ideally, this must be latency from sending noti
                             to receiving noti */
    uint64_t *latencyBuf2; /* store end-to-end latency (write to read)*/
    bool latencytest; /* latency test flag */
    uint32_t y_timeout; /* IVC notification timeout for latency test */
    uint64_t *clockBuf; /* read clockcycle */
    int32_t timeout; /* msec timeout for WaitForEvent/MsgReceivePulse */
    bool useThread; /* use thread for test body */
    uid_t uid;
    gid_t gid[MAX_GID_NUM]; /* gid[0]=GID, gid[1..255]=SGIDs */
    uint32_t num_gid;
    char *prgname;
    uint32_t errcnt;
    bool initFlag;  /* init_resources() is exelwted successfully */
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

void print_usage(char *argv[]);
int32_t read_test(struct endpoint *ep);
int32_t read_test_zc(struct endpoint *ep);
int32_t read_test_peek(struct endpoint *ep);
LwSciError wait_for_connection(void);
#ifdef __QNX__
int32_t drop_privileges(void);
void consume_memory(void);
void consume_pulse_queue(void);
#endif // __QNX__
LwSciError wait_event(struct endpoint *ep, int32_t value, uint32_t count);
int32_t read_perf_test(void);
int32_t read_stress_test(void);
void show_latency_statistics(char *title, uint64_t data[], uint32_t count);
static void setup_termination_handlers(void);
static LwSciError init_resources(void);
static void release_resources(void);
void *read_test_main(void *arg);

#ifdef LINUX
#ifdef __x86_64__
static uint64_t getClockCycles(void)
{
    return 0;
}
static uint64_t getCyclesPerSec(void)
{
    return 0;
}
#else
static uint64_t getClockCycles(void)
{
    uint64_t clock_cycles;
    asm volatile("mrs %0, CNTVCT_EL0" : "=r" (clock_cycles) : : "memory");
    return clock_cycles;
}

static uint64_t getCyclesPerSec(void)
{
    uint64_t cycles_per_sec;
    asm volatile("mrs %0, CNTFRQ_EL0" : "=r" (cycles_per_sec) : : "memory");
    return cycles_per_sec;
}
#endif /* __x86_64__ */
#endif /* LINUX */

#ifdef __QNX__
static uint64_t getClockCycles(void)
{
    return ClockCycles();
}

static uint64_t getCyclesPerSec(void)
{
    return(SYSPAGE_ENTRY(qtime)->cycles_per_sec);
}
#endif

void print_usage(char *argv[])
{
    fprintf(stderr, "Usage: %s [OPTION]...\n", argv[0]);
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "\t -h                 : "
            "Print this help screen\n");
#ifdef __QNX__
    fprintf(stderr, "\t -a <lo:hi>         : "
            "set lo/hi limit on range of ability (hexadecimal)\n");
#endif /* __QNX__ (a) */
    fprintf(stderr, "\t -b                 : "
            "bi-directional ping test\n");
    fprintf(stderr, "\t -c <endpoint_name> : "
            "name of LwSciIpc endpoint\n");
    fprintf(stderr, "\t -d backoff delay   : "
            "delay(def:%dns) btw each perf iterations\n", DEFAULT_BACKOFF_NS);
#ifdef __QNX__
    fprintf(stderr, "\t -e                 : "
            "msg drop test by consuming pulse queue\n");
    fprintf(stderr, "\t -i priority        : "
            "pulse event priority (default:-1(INHERIT), 21(INTR)\n");
#endif /* __QNX__ (e,i) */
    fprintf(stderr, "\t -l                 : loop count\n");
    fprintf(stderr, "\t -m                 : ignore data mismatch\n");
    fprintf(stderr, "\t -p                 : performance test\n");
    fprintf(stderr, "\t -s                 : "
            "LwSciEventService test (will be deprecated)\n");
#ifdef __QNX__
    fprintf(stderr, "\t -t start delay     : "
            "start delay(ms) for ping test\n");
    fprintf(stderr, "\t -u <uid:gid,sgid,> : "
            "UID and GID setting for test\n");
#endif /* __QNX__ (t, u) */
    fprintf(stderr, "\t -v                 : verbose mode\n");
    fprintf(stderr, "\t -y <msec>          : "
            "IVC notification latency test with msec timeout\n");
#ifdef __QNX__
    fprintf(stderr, "\t -A                 : "
            "print start timestamp\n");
#endif /* __QNX__ (A) */
    fprintf(stderr, "\t -B                 : "
            "use LwSciIpcBindEventService when EventService is used\n");
    fprintf(stderr, "\t -E <msec>          : "
            "use LwSciEventService with msec timeout\n"
            "\t wait infinitely if msec is U (-E U)\n");
#ifdef __QNX__
    fprintf(stderr, "\t -I [endpointPrefix:start#:end#:initIter:logPeriodSec] : "
            "stress test (multiple API invocation)\n"
            "\t\tex) \"-I ipcrx:0001:0050:5:60\" means that do stress test\n"
            "\t\twith endpoints from ipcrx0001 to ipcrx0050\n"
            "\t\tprint progress message every 60sec\n"
            "\t\t5 init/deinit iterations\n");
    fprintf(stderr, "\t -J [id:INFO_STR]   : "
           "generate test log file with test id, info string and result\n");
#endif /* __QNX__ (I) */
    fprintf(stderr, "\t -M                 : "
            "test more read/write API variations\n");
    fprintf(stderr, "\t -Q                 : "
            "quiet mode (no output message)\n");
    fprintf(stderr, "\t -R                 : "
            "use thread for test body\n");
#ifdef __QNX__
    fprintf(stderr, "\t -S [endpointPrefix:start#:end#:logPeriodSec] : "
            "stress test\n"
            "\t\tex) \"-S ipcrx:0001:0050:60\" means that do stress test\n"
            "\t\twith endpoints from ipcrx0001 to ipcrx0050\n"
            "\t\tprint progress message every 60sec\n");
#if (LW_IS_SAFETY == 0) && defined(IVC_EVENTLIB)
    fprintf(stderr, "\t -T                 : "
            "enable eventlib and dump eventlib logging\n");
#endif
    fprintf(stderr, "\t -W timeout         : "
            "legacy event timeout(msec) for MsgReceivePulse\n");
    fprintf(stderr, "\t -X [endpointPrefix:start#:end#:logPeriodSec] : "
            "stress test (parallel exelwtion of API)\n"
            "\t\tex) \"-X ipcrx:0001:0050:60\" means that do stress test\n"
            "\t\twith endpoints from ipcrx0001 to ipcrx0050\n"
            "\t\tprint progress message every 60sec\n");
    fprintf(stderr, "\t -Z                 : "
            "print end timestamp\n");
    fprintf(stderr, "\t -@                 : "
            "no exit\n");
#endif /* __QNX__ (S,T,W,Z,@) */
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
/* for message drop test */
#define SIZE_1GB 0x40000000U
void consume_memory(void)
{
    test_params *tp = &s_params;
    volatile char *buffer;
    volatile uint32_t cnt;
    uint32_t size;
    uint32_t i;

    /* consume memory by malloc */
    for (i = 0; i <= 30; i++) {
        size = SIZE_1GB >> i;

        cnt = 0;
        do {
            buffer = (char *)malloc(size);
            if (buffer != NULL) {
                cnt++;
            }
        } while (buffer != NULL);

        if (cnt != 0) {
            mprintf("[%20s] %s: malloc(0x%08x) count: %d\n",
                tp->prgname, __func__, size, cnt);
        }
    }
}

/* it takes about 25sec */
void consume_pulse_queue(void)
{
    test_params *tp = &s_params;
    uint64_t start;
    uint64_t end;
    uint32_t sec;
    volatile uint32_t cnt;
    volatile int32_t err;
    volatile uint32_t errcnt = 0;

    start = getClockCycles();

    //consume_memory();

    /* consume memory by MsgSendPulse */
    cnt = 0;
    while(errcnt < 10) {
        err = MsgSendPulse_r(tp->coid, tp->prio, TEST_EVENT_CODE, cnt);
        cnt++;
        if (err != EOK) {
            errcnt++;
        }
    }

    mprintf("[%20s] %s: MsgSendPulse count: %d, err: %d\n",
        tp->prgname, __func__, cnt, err);

    end = getClockCycles();

    sec = (end - start) / getCyclesPerSec();
    mprintf("[%20s] %s: time diff(sec): %d\n", tp->prgname, __func__, sec);
}

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
            PROCMGR_ADN_NONROOT | PROCMGR_AOP_ALLOW | PROCMGR_AOP_SUBRANGE | PROCMGR_AOP_LOCK | abilityId,
            (uint64_t)tp->range_lo, (uint64_t)tp->range_hi,
            PROCMGR_ADN_NONROOT | PROCMGR_AID_EOL);
        if (ret != EOK) {
            mprintf("%s: Unable to reserve procmgr abilities: %d\n",
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
            mprintf("Failed to set GID to %d: %d\n", tp->gid[0], ret);
            return ret;
        }
    }

    /* if uid is not root */
    if (tp->uid != 0) {
        ret = setreuid(tp->uid, tp->uid);
        if (ret != EOK) {
            mprintf("Failed to set UID to %d: %d\n", tp->uid, ret);
            return ret;
        }
    }

    return EOK;
}

/*
 * ep: it's valid in TT_STRESS3. use NULL for other case
 *
 * Returns:
 * LwSciError_Success : success
 * others             : failure
 */
static LwSciError LwMsgReceivePulse(struct endpoint *ep,
    struct _pulse *pulse,
    size_t bytes, struct _msg_info *info, uint32_t *cookie)
{
    test_params *tp = &s_params;
    LwSciError ret;
    int32_t err;

    do {
        if (tp->testtype == TT_STRESS3) {
            /* In using thread per endpoint, need chid/coid per thread */
            err = MsgReceivePulse_r(ep->chid, pulse, bytes, info);
        }
        else {
            err = MsgReceivePulse_r(tp->chid, pulse, bytes, info);
        }

        if (err == EOK) {
            if (pulse->code < 0) {
                /* system pulse */
                continue;
            }
            else if (((uint32_t)pulse->code < TEST_EVENT_CODE) ||
                ((uint32_t)pulse->code >= (TEST_EVENT_CODE + tp->epCnt))) {
                mprintf("%s: invalid pulse: %d\n", __func__, pulse->code);
                ret = -EILWAL;
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
#endif // __QNX__

LwSciError wait_for_connection(void)
{
    test_params *tp = &s_params;
    struct endpoint *ep = &s_params.ep[0];
#ifdef __QNX__
    struct _pulse pulse;
    uint64_t to;
#endif
    uint32_t event = 0;
    int32_t timeout;
    int32_t ret;
    LwSciError err;

    while(!s_Stop) {
        err = LwSciIpcGetEvent(ep->h, &event);
        if (err != LwSciError_Success) {
            mprintf("[%20s] %s: get event: %d\n",
                tp->prgname, __func__, err);
            return err;
        }
        if (event & LW_SCI_IPC_EVENT_CONN_EST_ALL) {
            break;
        }
        if (tp->evtSvcFlag) {
            /* msec to usec */
            timeout = (tp->timeout >= 0)?(tp->timeout*1000):tp->timeout;

            err = tp->eventLoopServiceP->WaitForEvent(ep->eventNotifier,
                timeout);
            if (err != LwSciError_Success) {
                mprintf("[%20s] %s: WaitForEvent err: %d\n",
                    tp->prgname, __func__, err);
                return err;
            }
        }
        else {
#ifdef LINUX
            ret = select(ep->fd + 1, &ep->rfds, NULL, NULL, NULL);
            if ((ret < 0) & (ret != EINTR)) {
                mprintf("[%20s] %s: error in select: %d\n",
                    tp->prgname, __func__, ret);
                exit(-1);
            }
#endif
#ifdef __QNX__
            if (tp->timeout >= 0) {
                to = tp->timeout * 1000000U; /* msec to nsec */
                if ((ret = TimerTimeout_r(CLOCK_MONOTONIC, _NTO_TIMEOUT_RECEIVE, NULL,
                    &to, NULL)) < 0) {
                    err = LwSciIpcErrnoToLwSciErr(ret);
                    mprintf("[%20s] %s: TimerTimeout: %d\n",
                        tp->prgname, __func__, err);
                    return err;
                }
            }
            err = LwMsgReceivePulse(NULL, &pulse, sizeof(pulse), NULL, NULL);
            if (err != LwSciError_Success) {
                mprintf("[%20s] %s: MsgReceivePulse err: %d\n",
                    tp->prgname, __func__, err);
                return err;
            }
#endif /* __QNX__ */
        }
#if LATENCY_DEBUG
        dprintf("[%20s] Received notification\n", tp->prgname);
#endif
    }

    dprintf("[%20s] Connection established\n", tp->prgname);

    return LwSciError_Success;
}

LwSciError wait_event(struct endpoint *ep, int32_t value, uint32_t count)
{
    test_params *tp = &s_params;
#ifdef __QNX__
    struct _pulse pulse;
    uint64_t to;
#endif
    uint32_t event = 0;
    int32_t timeout;
    int32_t ret;
    LwSciError err;

    while(!s_Stop) {
        err = LwSciIpcGetEvent(ep->h, &event);
        if (err != LwSciError_Success) {
            mprintf("[%20s] %s: get event: %d\n",
                tp->prgname, __func__, err);
            return err;
        }
        if (event & value) {
#if LATENCY_DEBUG
            dprintf("[%20s] %s: evtcnt:%d (event:0x%x)\n",
                tp->prgname, __func__, ep->evt_cnt, event);
#endif
            break;
        }
#if LWIPC_DEBUG
        dprintf("[%20s] %s: event: 0x%x\n",
            tp->prgname, __func__, event);
#endif
        if (tp->evtSvcFlag) {
            /* msec to usec */
            timeout = (tp->timeout >= 0)?(tp->timeout*1000):tp->timeout;

            err = tp->eventLoopServiceP->WaitForEvent(ep->eventNotifier,
                timeout);
            if (err != LwSciError_Success) {
                mprintf("[%20s] %s: WaitForEvent err: %d\n",
                    tp->prgname, __func__, err);
                return err;
            }
        }
#ifdef __QNX__
        /* for TT_STRESS3 */
        else if (ep->apitype == API_LWSCIEVENT) {
            /* msec to usec */
            timeout = (tp->timeout >= 0)?(tp->timeout*1000):tp->timeout;

            err = ep->eventLoopServiceP->WaitForEvent(ep->eventNotifier,
                timeout);
            if (err != LwSciError_Success) {
                mprintf("[%20s] %s: WaitForEvent err: %d\n",
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
                    mprintf("[%20s] %s: TimerTimeout: %d\n",
                        tp->prgname, __func__, err);
                    return err;
                }
            }
            err = LwMsgReceivePulse(ep, &pulse, sizeof(pulse), NULL, NULL);
            if (err != LwSciError_Success) {
                mprintf("[%20s] %s: MsgReceivePulse err: %d\n",
                    tp->prgname, __func__, err);
                return err;
            }
#if LWIPC_DEBUG
            dprintf("[%20s] %s: code:%x, val:%x\n",
                tp->prgname, __func__, pulse.code, pulse.value.sival_int);
#endif
        }
        LWSCIIPC_EVENTLIB_LOG(lwsciipc_ivc_pulse_event, (uint32_t)ep->evt_cnt);
#endif /* __QNX__ */
#ifdef LINUX
        else {
            ret = select(ep->fd + 1, &ep->rfds, NULL, NULL, NULL);
            if ((ret < 0) & (ret != EINTR)) {
                mprintf("[%20s] %s: error in select: %d\n",
                    tp->prgname, __func__, ret);
                exit(-1);
            }
        }
        if (FD_ISSET(ep->fd, &ep->rfds)) {
            dprintf("[%20s] %s: waiting data done\n",
                tp->prgname, __func__);
        }
#endif /* LINUX */
        ep->evt_cnt++;
        if (tp->latencytest && count < MAX_LATENCY_SAMPLE &&
            tp->clockBuf[count] == 0UL && LwSciIpcCanRead(ep->h)) {
            tp->clockBuf[count] = getClockCycles();
#if LATENCY_DEBUG
            dprintf("[%20s] %s: [%d] %" PRId64 "\n",
                tp->prgname, __func__, count, tp->clockBuf[count]);
#endif
        }
    }

    return LwSciError_Success;
}

#ifdef __QNX__
static void print_uid_gid(char *str)
{
    dprintf("[%20s] UID: %d, GID: %d, PID: %d\n",
        str, getuid(), getgid(), getpid());
}

#endif // __QNX__

#define BILLION 1000000000U
static int32_t colwert_to_timespec(uint64_t counter, struct timespec *time)
{
    /* Static so that the value is cached */
    static uint64_t cycles_per_sec = 0U;
    int32_t ret;

    if (time == NULL) {
        ret = -1;
    } else {
        if (cycles_per_sec == 0U) {
            cycles_per_sec = getCyclesPerSec();
        }

        time->tv_sec = counter / cycles_per_sec;
        time->tv_nsec = ((counter % cycles_per_sec) * BILLION)
                                / cycles_per_sec;
        /* Handle overflow in the nsec field */
        if (time->tv_nsec >= BILLION) {
            time->tv_nsec -= BILLION;
            time->tv_sec += 1;
        }

        ret = 0;
    }

    return ret;
}

#ifdef __QNX__
static void release_resources_for_PE_test(struct endpoint *ep)
{
    if (ep->buf) {
        free(ep->buf);
        ep->buf = NULL;
    }
    if (ep->buf2) {
        free(ep->buf2);
        ep->buf2 = NULL;
    }

    if (ep->apitype == API_LWSCIEVENT) {
        ep->eventNotifier->Delete(ep->eventNotifier);
    }

    LwSciIpcCloseEndpoint(ep->h);

    if (ep->coid != 0) {
        (void)ConnectDetach_r(ep->coid);
        ep->coid = 0;
    }
    if (ep->chid != 0) {
        (void)ChannelDestroy_r(ep->chid);
        ep->chid = 0;
    }

    if (ep->apitype == API_LWSCIEVENT) {
        ep->eventLoopServiceP->EventService.Delete(&ep->eventLoopServiceP->EventService);
    }
}
#endif /* __QNX__ */

static void release_resources(void)
{
    test_params *tp = &s_params;
    struct endpoint *ep;
    uint32_t idx;

    for (idx = 0; idx < tp->epCnt; idx++) {
        ep = &tp->ep[idx];
        if (ep->buf) {
            free(ep->buf);
            ep->buf = NULL;
        }
        if (ep->buf2) {
            free(ep->buf2);
            ep->buf2 = NULL;
        }

        if (tp->evtSvcFlag) {
            ep->eventNotifier->Delete(ep->eventNotifier);
        }

        dprintf("[%20s] closing LwSciIpc endpoint\n", tp->prgname);
        LwSciIpcCloseEndpoint(ep->h);
    }

    if (tp->startTimestamp) {
        free(tp->startTimestamp);
        tp->startTimestamp = NULL;
    }
    if (tp->latencyBuf) {
        free(tp->latencyBuf);
        tp->latencyBuf = NULL;
    }
    if (tp->latencyBuf2) {
        free(tp->latencyBuf2);
        tp->latencyBuf2 = NULL;
    }
    if (tp->output_path) {
        free(tp->output_path);
        tp->output_path = NULL;
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
#endif /* __QNX__ */

    LwSciIpcDeinit();

    if (tp->evtSvcFlag) {
        tp->eventLoopServiceP->EventService.Delete(&tp->eventLoopServiceP->EventService);
    }

    if (s_Stop) {
        exit(1);
    }
}

static void sig_handler(int sig_num)
{
    test_params *tp = &s_params;

    s_Stop = 1;

    if (tp->initFlag) {
        release_resources();
        tp->initFlag = false;
    }

#ifdef __QNX__
    {
        struct endpoint *ep;
        uint32_t idx;

        /* for TT_STRESS3 */
        for (idx=0;idx < tp->epCnt; idx++) {
            ep = &tp->ep[idx];
            if (ep->initFlag) {
                release_resources_for_PE_test(ep);
                ep->initFlag = false;
            }
        }
    }
#endif /* __QNX__ */
}

static void setup_termination_handlers(void)
{
    signal(SIGINT, sig_handler);
    signal(SIGTERM, sig_handler);
    signal(SIGHUP, sig_handler);
    signal(SIGQUIT, sig_handler);
    signal(SIGABRT, sig_handler);
}

static LwSciError os_setup(void)
{
#ifdef __QNX__
    test_params *tp = &s_params;
    int32_t ret;
    LwSciError err;

    ret = drop_privileges();
    if (ret != EOK) {
        err = LwSciIpcErrnoToLwSciErr(ret);
        return err;
    }
    print_uid_gid(tp->prgname);
#endif  /* __QNX__ */

    return LwSciError_Success;
}

#ifdef __QNX__
/* init resource for parallel exelwtion test */
static LwSciError init_resources_for_PE_test(struct endpoint *ep)
{
    test_params *tp = &s_params;
    int32_t chid;
    int32_t coid;
    LwSciError err;
    int32_t tid;

    tid = gettid();

    if (ep->apitype == API_LWSCIEVENT) {
        dprintf("[%20s:%d] %s: LwSciEventLoopServiceCreate\n",
            tp->prgname, tid, __func__);
        err = LwSciEventLoopServiceCreate(1, &ep->eventLoopServiceP);
        if (err != LwSciError_Success) {
            printf("[%20s:%d] %s: LwSciEventLoopServiceCreate: fail (%d)\n",
                tp->prgname, tid,  __func__, err);
            goto fail;
        }
    }

    if (ep->apitype == API_LEGACY) {
        /* w/o LwSciEventService */
        chid = ChannelCreate_r(0U);
        if (chid < 0) {
            err = LwSciIpcErrnoToLwSciErr(chid);
            printf("[%20s:%d] %s: ChannelCreate_r: fail (%d)\n",
                tp->prgname, tid, __func__, err);
            goto fail;
        }
        ep->chid = chid;
        coid = ConnectAttach_r(0, 0, chid, _NTO_SIDE_CHANNEL, 0);
        if (coid < 0) {
            err = LwSciIpcErrnoToLwSciErr(coid);
            printf("[%20s:%d] %s: ConnectAttach_r: fail (%d)\n",
                tp->prgname, tid,  __func__, err);
            goto fail;
        }
        ep->coid = coid;
        dprintf("[%20s:%d] chid:%d, coid:%d\n", tp->prgname, tid, chid, coid);
    }

    mprintf("[%20s:%d] opening LwSciIpc endpoint: %s\n",
        tp->prgname, tid, ep->chname);
    if (ep->apitype == API_LWSCIEVENT) {
        dprintf("[%20s:%d] %s: LwSciIpcOpenEndpointWithEventService\n",
            tp->prgname, tid, __func__);
        err = LwSciIpcOpenEndpointWithEventService(ep->chname, &ep->h,
            &ep->eventLoopServiceP->EventService);
    }
    else {
        err = LwSciIpcOpenEndpoint(ep->chname, &ep->h);
    }
    if (err != LwSciError_Success) {
        printf("[%20s:%d] %s: LwSciIpcOpenEndpoint(%s): fail (%d)\n",
            tp->prgname, tid,  __func__, ep->chname, err);
        goto fail;
    }
    dprintf("[%20s:%d] endpoint handle: 0x%lx\n", tp->prgname, tid, ep->h);

    /* for LwSciEventService */
    if (ep->apitype == API_LWSCIEVENT) {
        dprintf("[%20s:%d] %s: LwSciIpcGetEventNotifier\n",
            tp->prgname, tid, __func__);
        err = LwSciIpcGetEventNotifier(ep->h, &ep->eventNotifier);
        if (err != LwSciError_Success) {
            printf("[%20s:%d] %s: LwSciIpcGetEventNotifier: fail (%d)\n",
                tp->prgname, tid, __func__, err);
            goto fail;
        }
    }
    else {
        /* pulse code is different per endpoint */
        err = LwSciIpcSetQnxPulseParam(ep->h, ep->coid, tp->prio,
            TEST_EVENT_CODE, (void *)ep->h);
        if (err != LwSciError_Success) {
            printf("[%20s:%d] %s: LwSciIpcSetQnxPulseParam: fail (%d)\n",
                tp->prgname, tid, __func__, err);
            goto fail;
        }
    }

    err = LwSciIpcGetEndpointInfo(ep->h, &ep->info);
    if (err != LwSciError_Success) {
        printf("[%20s:%d] %s: LwSciIpcGetEndpointInfo: fail (%d)\n",
            tp->prgname, tid, __func__, err);
        goto fail;
    }
    mprintf("[%20s:%d] endpoint_info: nframes = %d, frame_size = %d\n",
        tp->prgname, tid, ep->info.nframes, ep->info.frame_size);

    ep->buf = calloc(1, ep->info.frame_size);
    if (ep->buf == NULL) {
        err = LwSciIpcErrnoToLwSciErr(errno);
        printf("[%20s:%d] %s: Failed to allocate buffer of size %u\n",
            tp->prgname, tid, __func__, ep->info.frame_size);
        goto fail;
    }
    ep->buf2 = calloc(1, ep->info.frame_size);
    if (ep->buf2 == NULL) {
        err = LwSciIpcErrnoToLwSciErr(errno);
        printf("[%20s:%d] %s: Failed to allocate buffer of size %u\n",
            tp->prgname, tid, __func__, ep->info.frame_size);
        goto fail;
    }

    LwSciIpcResetEndpoint(ep->h);

    err = LwSciError_Success;
    ep->initFlag = true;

fail:
    return err;
}
#endif /* __QNX__ */

static LwSciError init_resources(void)
{
    test_params *tp = &s_params;
    struct endpoint *ep;
    uint32_t i;
#ifdef __QNX__
    int32_t chid;
    int32_t coid;
#endif /* __QNX__ */
#ifdef LINUX
    int32_t fd;
#endif /* LINUX */
    uint32_t idx;
    LwSciError err;

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

    err = LwSciIpcInit();
    if (err != LwSciError_Success) {
        printf("[%20s] %s: LwSciIpcInit: fail (%d)\n",
            tp->prgname, __func__, err);
        goto fail;
    }

#ifdef __QNX__
    if (!tp->evtSvcFlag) {
        /* w/o LwSciEventService */
        chid = ChannelCreate_r(0U);
        if (chid < 0) {
            err = LwSciIpcErrnoToLwSciErr(chid);
            printf("[%20s] %s: ChannelCreate_r: fail (%d)\n",
                tp->prgname, __func__, err);
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
        dprintf("[%20s] chid:%d, coid:%d\n", tp->prgname, chid, coid);
    }
#endif /* __QNX__ */

    for (idx = 0; idx < tp->epCnt; idx++) {
        ep = &tp->ep[idx];
        if (tp->testtype == TT_STRESS) {
            sprintf(ep->chname, "%s%04d", tp->epprefix, (tp->epstart + idx));
            ep->prevData = 0xDEADBEEF;  /* ilwalidate prevData */
        }

        mprintf("[%20s] opening LwSciIpc endpoint: %s\n", tp->prgname, ep->chname);
        if ((tp->evtSvcFlag == true) && (tp->bindEvtSvcFlag == false)) {
            dprintf("[%20s] %s: LwSciIpcOpenEndpointWithEventService\n",
                tp->prgname, __func__);
            err = LwSciIpcOpenEndpointWithEventService(ep->chname, &ep->h,
                &tp->eventLoopServiceP->EventService);
        }
        else {
            err = LwSciIpcOpenEndpoint(ep->chname, &ep->h);
        }
        if (err != LwSciError_Success) {
            printf("[%20s] %s: LwSciIpcOpenEndpoint(%s): fail (%d)\n",
                tp->prgname, __func__, ep->chname, err);
            goto fail;
        }
        dprintf("[%20s] endpoint handle: 0x%lx\n", tp->prgname, ep->h);
        if ((tp->evtSvcFlag == true) && (tp->bindEvtSvcFlag == true)) {
            dprintf("[%20s] %s: LwSciIpcBindEventService\n",
                tp->prgname, __func__);
            err = LwSciIpcBindEventService(ep->h,
                &tp->eventLoopServiceP->EventService);
            if (err != LwSciError_Success) {
                printf("[%20s] %s: LwSciIpcBindEventService: fail (%d)\n",
                    tp->prgname, __func__, err);
                goto fail;
            }
        }

#ifdef LINUX
        err = LwSciIpcGetLinuxEventFd(ep->h, &fd);
        if (err != LwSciError_Success) {
            printf("%s:LwSciIpcGetLinuxEventFd: fail (%d)\n", __func__, err);
            goto fail;
        }
        mprintf("[%20s] %s:LwSciIpcGetLinuxEventFd: fd(%d)\n",
            tp->prgname, __func__, fd);

        ep->fd = fd;
#endif  /* LINUX */

        /* for LwSciEventService */
        if (tp->evtSvcFlag) {
            dprintf("[%20s] %s: LwSciIpcGetEventNotifier\n",
                tp->prgname, __func__);
            err = LwSciIpcGetEventNotifier(ep->h, &ep->eventNotifier);
            if (err != LwSciError_Success) {
                printf("[%20s] %s: LwSciIpcGetEventNotifier: fail (%d)\n",
                    tp->prgname, __func__, err);
                goto fail;
            }
        }
#ifdef __QNX__
        else {
            /* pulse code is different per endpoint */
            err = LwSciIpcSetQnxPulseParam(ep->h, tp->coid, tp->prio,
                (TEST_EVENT_CODE+idx), (void *)ep->h);
            if (err != LwSciError_Success) {
                printf("[%20s] %s: LwSciIpcSetQnxPulseParam: fail (%d)\n",
                    tp->prgname, __func__, err);
                goto fail;
            }
            /* fill event notifier array for native event */
            tp->evtNotiArray[tp->evtNotiCnt++] = ep->eventNotifier;
        }
#endif /* __QNX__ */

        err = LwSciIpcGetEndpointInfo(ep->h, &ep->info);
        if (err != LwSciError_Success) {
            printf("[%20s] %s: LwSciIpcGetEndpointInfo: fail (%d)\n",
                tp->prgname, __func__, err);
            goto fail;
        }
        mprintf("[%20s] endpoint_info: nframes = %d, frame_size = %d\n",
            tp->prgname, ep->info.nframes, ep->info.frame_size);

        ep->buf = calloc(1, ep->info.frame_size);
        if (ep->buf == NULL) {
            err = LwSciIpcErrnoToLwSciErr(errno);
            printf("[%20s] %s: Failed to allocate buffer of size %u\n",
                tp->prgname, __func__, ep->info.frame_size);
            goto fail;
        }
        ep->buf2 = calloc(1, ep->info.frame_size);
        if (ep->buf2 == NULL) {
            err = LwSciIpcErrnoToLwSciErr(errno);
            printf("[%20s] %s: Failed to allocate buffer of size %u\n",
                tp->prgname, __func__, ep->info.frame_size);
            goto fail;
        }

#ifdef __QNX__
        if (tp->msgdroptest) {
            consume_pulse_queue();
        }
#endif /* __QNX__ */

        LwSciIpcResetEndpoint(ep->h);

#ifdef __QNX__
        if (tp->msgdroptest) {
            consume_pulse_queue();
        }
#endif /* __QNX__ */
    }

    err = LwSciError_Success;
    tp->initFlag = true;

fail:
    return err;
}

int32_t read_test(struct endpoint *ep)
{
    test_params *tp = &s_params;
    uint32_t count = 0;
    uint32_t *ptr;
    int32_t bytes;
    uint32_t rd_count = 0;
    uint32_t rd_err_cnt = 0;
    LwSciError err;
    /* bi-directional test */
    uint32_t *ptr2;
    uint32_t wr_count = 0;
    uint32_t wr_err_cnt = 0;

    /* set default ep */
    if (ep == NULL) {
        ep = &tp->ep[0];
    }

    ptr2 = (uint32_t *)ep->buf2;

    mprintf("[%20s] Ping Test mode (loop: %ld)\n",
        tp->prgname, tp->iterations);

    ptr = (uint32_t *)ep->buf;
#ifdef LINUX
    FD_ZERO(&ep->rfds);
    FD_SET(ep->fd, &ep->rfds);
#endif /* LINUX */

    while ((count < tp->iterations) && !s_Stop) {
        err = wait_event(ep, LW_SCI_IPC_EVENT_READ, count);
        if(err != LwSciError_Success) {
            mprintf("%s: error in waiting RD event: %d\n", __func__, err);
            rd_err_cnt++;
            ep->rd_err_cnt++;
            break;
        }

        err = LwSciIpcRead(ep->h, ep->buf, ep->info.frame_size, &bytes);
        if(err != LwSciError_Success || (uint32_t)bytes != ep->info.frame_size) {
            mprintf("%s: error in reading: %d\n", __func__, err);
            rd_err_cnt++;
            ep->rd_err_cnt++;
            break;
        }
        dprintf("[%20s]%s: RD#%02d: %d %d\n",tp->prgname,
            ep->chname, count, ptr[0], ptr[1]);
        if (tp->ignoredata == 0 && ptr[0] != count) {
            dprintf("%s: mismatch (rx: %d, expected: %d)\n", tp->prgname,
                ptr[0], count);
            rd_err_cnt++;
            ep->rd_err_cnt++;
        }
        rd_count++;
        ep->rd_count++;

        /* bi-directional test */
        if (tp->bidirtest) {
            err = wait_event(ep, LW_SCI_IPC_EVENT_WRITE, count);
            if(err != LwSciError_Success) {
                mprintf("%s: error in waiting WR event: %d\n", __func__, err);
                wr_err_cnt++;
                break;
            }

            ptr2[0] = 0;
            ptr2[1] = count;
            dprintf("[%20s]%s: WR#%02d: %d %d\n", tp->prgname,
                ep->chname, count, ptr2[0], ptr2[1]);
            err = LwSciIpcWrite(ep->h, ep->buf2, ep->info.frame_size, &bytes);
            if(err != LwSciError_Success || (uint32_t)bytes != ep->info.frame_size) {
                mprintf("%s: error in writing: %d\n", __func__, err);
                wr_err_cnt++;
                break;
            }
            wr_count++;
        }

        count++;
    }

    mprintf("[%20s]  read count: %d, rd err count: %d\n",
        tp->prgname, rd_count, rd_err_cnt);
    /* bi-directional test */
    if (tp->bidirtest) {
        mprintf("[%20s] write count: %d, wr err count: %d\n",
            tp->prgname, wr_count, wr_err_cnt);
    }
    dprintf("[%20s] event count: %u\n", tp->prgname, ep->evt_cnt);

    if (rd_err_cnt > 0)
        return EIO;
    else
        return 0;
}

int32_t read_test_zc(struct endpoint *ep)
{
    test_params *tp = &s_params;
    uint32_t count = 0;
    uint32_t rd_count = 0;
    uint32_t rd_err_cnt = 0;
    LwSciError err;
    const volatile uint32_t *rxbuf;
    /* bi-directional test */
    uint32_t wr_count = 0;
    uint32_t wr_err_cnt = 0;
    volatile uint32_t *txbuf;

    /* set default ep */
    if (ep == NULL) {
        ep = &tp->ep[0];
    }

    mprintf("[%20s] Ping Test mode (loop: %ld) - zero copy\n",
        tp->prgname, tp->iterations);

#ifdef LINUX
    FD_ZERO(&ep->rfds);
    FD_SET(ep->fd, &ep->rfds);
#endif /* LINUX */

    while ((count < tp->iterations) && !s_Stop) {
        err = wait_event(ep, LW_SCI_IPC_EVENT_READ, count);
        if(err != LwSciError_Success) {
            mprintf("%s: error in waiting RD event: %d\n", __func__, err);
            rd_err_cnt++;
            ep->rd_err_cnt++;
            break;
        }

        err = LwSciIpcReadGetNextFrame(ep->h, (void *)&rxbuf);
        if(err != LwSciError_Success) {
            mprintf("%s: error in getting read frame: %d\n", __func__, err);
            rd_err_cnt++;
            ep->rd_err_cnt++;
            break;
        }
        dprintf("[%20s]%s: RD#%02d: %d %d\n", tp->prgname,
            ep->chname, count, rxbuf[0], rxbuf[1]);
        if (tp->ignoredata == 0 && rxbuf[0] != count) {
            dprintf("%s: mismatch (rx: %d, expected: %d)\n", tp->prgname,
                rxbuf[0], count);
            rd_err_cnt++;
            ep->rd_err_cnt++;
        }
        err = LwSciIpcReadAdvance(ep->h);
        if (err != LwSciError_Success) {
            mprintf("%s: error in removing next frame: %d\n", __func__, err);
            rd_err_cnt++;
            break;
        }
        rd_count++;
        ep->rd_count++;

        /* bi-directional test */
        if (tp->bidirtest) {
            err = wait_event(ep, LW_SCI_IPC_EVENT_WRITE, count);
            if(err != LwSciError_Success) {
                mprintf("%s: error in waiting WR event: %d\n", __func__, err);
                wr_err_cnt++;
                break;
            }

            err = LwSciIpcWriteGetNextFrame(ep->h, (void *)&txbuf);
            if(err != LwSciError_Success) {
                mprintf("%s: error in getting write frame: %d\n", __func__, err);
                wr_err_cnt++;
                break;
            }
            txbuf[0] = 0;
            txbuf[1] = count;
            dprintf("[%20s] WR#%02d: %d %d\n", tp->prgname, count,
                txbuf[0], txbuf[1]);
            err = LwSciIpcWriteAdvance(ep->h);
            if (err != LwSciError_Success) {
                mprintf("%s: error in writing next frame: %d\n", __func__, err);
                wr_err_cnt++;
                break;
            }
            wr_count++;
        }

        count++;
    }

    mprintf("[%20s]  read count: %d, rd err count: %d\n",
        tp->prgname, rd_count, rd_err_cnt);
    /* bi-directional test */
    if (tp->bidirtest) {
        mprintf("[%20s] write count: %d, wr err count: %d\n",
            tp->prgname, wr_count, wr_err_cnt);
    }
    dprintf("[%20s] event count: %u\n", tp->prgname, ep->evt_cnt);

    if (rd_err_cnt > 0)
        return EIO;
    else
        return 0;
}

int32_t read_test_peek(struct endpoint *ep)
{
    test_params *tp = &s_params;
    uint32_t count = 0;
    uint32_t *ptr;
    int32_t bytes;
    uint32_t rd_count = 0;
    uint32_t rd_err_cnt = 0;
    LwSciError err;
    /* bi-directional test */
    uint32_t *ptr2;
    uint32_t wr_count = 0;
    uint32_t wr_err_cnt = 0;

    /* set default ep */
    if (ep == NULL) {
        ep = &tp->ep[0];
    }

    ptr2 = (uint32_t *)ep->buf2;

    mprintf("[%20s] Ping Test mode (loop: %ld) - peek/poke\n",
        tp->prgname, tp->iterations);

    ptr = (uint32_t *)ep->buf;
#ifdef LINUX
    FD_ZERO(&ep->rfds);
    FD_SET(ep->fd, &ep->rfds);
#endif /* LINUX */

    while ((count < tp->iterations) && !s_Stop) {
        err = wait_event(ep, LW_SCI_IPC_EVENT_READ, count);
        if(err != LwSciError_Success) {
            mprintf("%s: error in waiting RD event: %d\n", __func__, err);
            rd_err_cnt++;
            ep->rd_err_cnt++;
            break;
        }

        err = LwSciIpcReadPeek(ep->h, ep->buf, 0, ep->info.frame_size, &bytes);
        if(err != LwSciError_Success || (uint32_t)bytes != ep->info.frame_size) {
            mprintf("%s: error in peeking frame: %d\n", __func__, err);
            rd_err_cnt++;
            ep->rd_err_cnt++;
            break;
        }
        dprintf("[%20s]%s: RD#%02d: %d %d\n", tp->prgname,
            ep->chname, count, ptr[0], ptr[1]);
        if (tp->ignoredata == 0 && ptr[0] != count) {
            dprintf("%s: mismatch (rx: %d, expected: %d)\n", tp->prgname,
                ptr[0], count);
            rd_err_cnt++;
            ep->rd_err_cnt++;
        }
        err = LwSciIpcReadAdvance(ep->h);
        if (err != LwSciError_Success) {
            mprintf("%s: error in removing next frame: %d\n", __func__, err);
            rd_err_cnt++;
            break;
        }
        rd_count++;
        ep->rd_count++;

        /* bi-directional test */
        if (tp->bidirtest) {
            err = wait_event(ep, LW_SCI_IPC_EVENT_WRITE, count);
            if(err != LwSciError_Success) {
                mprintf("%s: error in waiting WR event: %d\n", __func__, err);
                wr_err_cnt++;
                break;
            }

            ptr2[0] = 0;
            ptr2[1] = count;
            dprintf("[%20s] WR#%02d: %d %d\n", tp->prgname, count,
                ptr2[0], ptr2[1]);

            err = LwSciIpcWritePoke(ep->h, ep->buf2, 0, ep->info.frame_size, &bytes);
            if(err != LwSciError_Success || (uint32_t)bytes != ep->info.frame_size) {
                mprintf("%s: error in poking frame: %d\n", __func__, err);
                wr_err_cnt++;
                break;
            }

            err = LwSciIpcWriteAdvance(ep->h);
            if (err != LwSciError_Success) {
                mprintf("%s: error in writing next frame: %d\n", __func__, err);
                wr_err_cnt++;
                break;
            }
            wr_count++;
        }

        count++;
    }

    mprintf("[%20s]  read count: %d, rd err count: %d\n",
        tp->prgname, rd_count, rd_err_cnt);
    /* bi-directional test */
    if (tp->bidirtest) {
        mprintf("[%20s] write count: %d, wr err count: %d\n",
            tp->prgname, wr_count, wr_err_cnt);
    }
    dprintf("[%20s] event count: %u\n", tp->prgname, ep->evt_cnt);

    if (rd_err_cnt > 0)
        return EIO;
    else
        return 0;
}

static int compare(const void *p1, const void *p2)
{
    return ( *(int64_t *)p1 - *(int64_t *)p2);
}

static void write_latency_statistics(char *filename, uint64_t timestamp[], uint64_t latency[], uint32_t count)
{
    int fd = -1;
    test_params *tp = &s_params;
    struct timespec tm;
    uint64_t latency_ns;
    uint32_t skipcnt = 0;
    char sz[1024];
    uint32_t i;
    /* 95% margin */
    uint32_t latency_timeout_ns = (tp->y_timeout * 1000000U * 95 / 100);

    if (count > MAX_LATENCY_SAMPLE) {
        count = MAX_LATENCY_SAMPLE;
    }

    fd = open(filename,
              O_WRONLY | O_TRUNC | O_CREAT,
              S_IRWXU | S_IRWXG | S_IRWXO);

    if (fd == -1) {
        mprintf("[%20s] Cannot open file, filename=%s\n", "LATENCY STATISTICS", filename);
        goto exit;
    }

    write(fd, LWPLAYFAIR_DATA_HEADER, strlen(LWPLAYFAIR_DATA_HEADER));

    for (i = 0; i < count; i++) {
        colwert_to_timespec(latency[i], &tm);
        latency_ns = tm.tv_sec;
        latency_ns *= BILLION;
        latency_ns += tm.tv_nsec;

        // skip abnormal data
        if (latency[i] == 0 || tm.tv_nsec > latency_timeout_ns) {
            skipcnt++;
            continue;
        }

        snprintf(sz, 1024 - 1, "%" PRIu64 ",%" PRIu64 "\n", timestamp[i], latency_ns);
        write(fd, sz, strlen(sz));
    }

    mprintf("[%20s] All latency data is stored in %s\n", "LATENCY STATISTICS", filename);

exit:
    if (fd == -1) {
        close(fd);
    }
}

void show_latency_statistics(char *title, uint64_t data[], uint32_t count)
{
    test_params *tp = &s_params;
    double sum = 0.0;
    double avg;
    double var = 0.0;
    double stdev = 0.0;
    double min = (double)UINT64_MAX;
    double max = 0.0;
    double pt99 = 0.0;
    uint32_t i;
    uint32_t skipcnt = 0;
    uint32_t minidx = UINT32_MAX;
    uint32_t maxidx = 0;
    uint32_t ptidx99 = 0;   /* index for 99.99 percentile data */
    struct timespec tm;
    /* 95% margin */
    uint32_t latency_timeout_ns = (tp->y_timeout * 1000000U * 95 / 100);

    if (count > MAX_LATENCY_SAMPLE) {
        count = MAX_LATENCY_SAMPLE;
    }

    for (i = 0; i < count; i++) {
        colwert_to_timespec(data[i], &tm);
        // skip abnormal data
        if (data[i] == 0 || tm.tv_nsec > latency_timeout_ns) {
#if LATENCY_DEBUG
//            dprintf("[%20s] skipped [%d]: %lfus\n",
//                tp->prgname, i, tm.tv_nsec/1000.);
#endif
            skipcnt++;
            continue;
        }
        sum += (double)data[i];
        if (min > data[i]) {
            min = data[i];
            minidx = i;
        }
        if (max < data[i]) {
            max = data[i];
            maxidx = i;
        }
#if LATENCY_DEBUG
        dprintf("[%20s][latency%d] %" PRId64 "\n", tp->prgname, i, data[i]);
#endif
    }

    /* get 99.99 percentile latency data */
    qsort(data, count, sizeof(uint64_t), compare);
    ptidx99 = count * 9999 / 10000 - 1;
    pt99 = data[ptidx99];

    dprintf("[%20s] skip count: %d\n", tp->prgname, skipcnt);
    count -= skipcnt;
    avg = sum/count;
    for (i = 0; i < count; i++) {
        var += pow(data[i] - avg, 2);
    }
    var = var/count;
    stdev = sqrt(var);

    mprintf("[%20s] %s\n", "LATENCY STATISTICS", title);
    colwert_to_timespec(min, &tm);
    mprintf("[%20s]    min latency: %lfus (%d)\n", tp->prgname,
        (double)tm.tv_nsec/1000., minidx);
    colwert_to_timespec(avg, &tm);
    mprintf("[%20s]    avg latency: %lfus\n", tp->prgname,
        (double)tm.tv_nsec/1000.);
    colwert_to_timespec(max, &tm);
    mprintf("[%20s]    max latency: %lfus (%d)\n", tp->prgname,
        (double)tm.tv_nsec/1000., maxidx);
    colwert_to_timespec(pt99, &tm);
    if (count >= 10000) {
        mprintf("[%20s] 99.99%% latency: %lfus\n", tp->prgname,
            (double)tm.tv_nsec/1000.);
    }
    else {
        mprintf("[%20s] 99.99%% latency: %d samples are NOT enough\n",
            tp->prgname, count);
    }
    colwert_to_timespec(stdev, &tm);
    mprintf("[%20s]  stdev latency: %lfus\n", tp->prgname,
        (double)tm.tv_nsec/1000.);
}

#if (LW_IS_SAFETY == 0) && defined(IVC_EVENTLIB)
static void lwsciipc_eventlib_notify_callback(uint32_t id)
{
    LWSCIIPC_EVENTLIB_LOG(lwsciipc_ivc_notify_read, id);
}
#endif

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
    return (cycles / ((double)getCyclesPerSec()));
}

static void display_perf_progress(uint64_t count, uint64_t cycles)
{
    test_params *tp = &s_params;
    uint32_t sec = get_sec_from_cycles(cycles);
    static uint32_t oldsec = 0;

    if (sec != oldsec) {
        if ((tp->logpersec != 0) && ((sec - oldsec) >= tp->logpersec)) {
            printf("%20s:%s:RD%5lu/%5lu(%02lu%%):time(%s)\n",
                tp->prgname,
                tp->ep[0].chname,
                count, tp->iterations, count*100/tp->iterations,
                get_time_str(sec));
            oldsec = sec;
        }
    }
}

static void display_perf_statistics(uint32_t rd_count, uint32_t err_count,
    uint64_t cycles)
{
    test_params *tp = &s_params;
    uint32_t sec = get_sec_from_cycles(cycles);

    printf("[%4s]%20s:%s:nf%d:fs%d:%s:LOC%5ld/%5ld:RD%5u/%5ld:ERR%u:"
        "rate%.2lfMB/s:time(%s)\n",
        (tp->errcnt > 0)?"FAIL":"PASS",
        tp->prgname,
        tp->ep[0].chname,
        tp->ep[0].info.nframes,
        tp->ep[0].info.frame_size,
        (tp->evtSvcFlag)?"LWSCIEVT":"NATIVE",
        tp->TotRxLocEvtCnt, tp->TotTxLocEvtCnt,
        rd_count, tp->iterations,
        err_count,
        (double)(rd_count * tp->ep[0].info.frame_size *
            ((double)getCyclesPerSec() / MEGABYTE) / cycles),
        get_time_str(sec));
}

int32_t read_perf_test(void)
{
    test_params *tp = &s_params;
    struct endpoint *ep = &tp->ep[0];
    uint64_t count = 0UL;
    uint64_t *ptr;
    int32_t bytes;
    uint64_t start = 0UL;
    uint64_t end;
    uint64_t readTime;
    double rate = 0;
    uint64_t rd_count = 0UL;
    uint64_t rd_err_cnt = 0UL;
    LwSciError err;

#ifdef LINUX
    FD_ZERO(&ep->rfds);
    FD_SET(ep->fd, &ep->rfds);
#endif /* LINUX */

    mprintf("[%20s] Performance Test mode (loop: %" PRId64 ")\n",
        tp->prgname, tp->iterations);
#if (LW_IS_SAFETY == 0) && defined(IVC_EVENTLIB)
    printf("[%20s] eventlib %s\n",
        tp->prgname, (tp->eventlib != 0) ? "enabled" : "disabled");
#endif

    ptr = (uint64_t *)ep->buf;

    if (tp->latencytest) {
        err = wait_for_connection();
        if (err != LwSciError_Success) {
            rd_err_cnt++;
            goto fail;
        }
    }

#if (LW_IS_SAFETY == 0) && defined(IVC_EVENTLIB)
    if (tp->eventlib != 0) {
        if (lwsciipc_eventlib_init() == 0) {
            LwSciIpcEventlibNotify = lwsciipc_eventlib_notify_callback;
        }
    }
#endif

    while ((count < tp->iterations) && !s_Stop) {
        err = wait_event(ep, LW_SCI_IPC_EVENT_READ, count);
        if(err != LwSciError_Success) {
            mprintf("%s: error in waiting RD event: %d\n", __func__, err);
            rd_err_cnt++;
            break;
        }

        if (start == 0UL) {
            start = getClockCycles();
        }

        LWSCIIPC_EVENTLIB_LOG(lwsciipc_ivc_read_start, (uint32_t)ep->info.frame_size);
        err = LwSciIpcRead(ep->h, ep->buf, ep->info.frame_size, &bytes);
        LWSCIIPC_EVENTLIB_LOG(lwsciipc_ivc_read_done, (uint32_t)bytes);
        if (err != LwSciError_Success) {
            mprintf("%s: error in reading: %d\n", __func__, err);
            rd_err_cnt++;
            break;
        }
        else {
            if (tp->latencytest && count < MAX_LATENCY_SAMPLE &&
            (tp->clockBuf[count] > ptr[1])) {
                readTime = getClockCycles();
                tp->startTimestamp[count] = ptr[1];
                tp->latencyBuf[count] = tp->clockBuf[count] - ptr[1];
                tp->latencyBuf2[count] = readTime - ptr[1];

#if LATENCY_DEBUG
                dprintf("[%20s] %s count:%" PRId64 " timestamp:%" PRId64 "\n",
                    tp->prgname, __func__, ptr[0], ptr[1]);
#endif
            }
            rd_count++;
            if (tp->ignoredata == 0 && ptr[0] != count) {
                rd_err_cnt++;
            }
        }

        count++;
        if (!tp->latencytest) {
#ifdef __QNX__
            nanospin_ns(tp->backoff_delay);
#endif
#ifdef LINUX
            struct timespec sleeptime;
            sleeptime.tv_sec = 0;
            sleeptime.tv_nsec = tp->backoff_delay;
            clock_nanosleep(CLOCK_MONOTONIC, 0, &sleeptime , NULL);
#endif
        }

        /* every 1048576(0x100000) count */
        if ((count & 0xFFFFFUL) == 0x0UL) {
            end = getClockCycles();
            display_perf_progress(count, end-start);
        }
    }

    end = getClockCycles();

    if (tp->testtype == TT_STRESS2) {
        // simple result
        display_perf_statistics(rd_count, rd_err_cnt, end-start);
    }
    else {
        mprintf("[%20s]  read count: %lu, error count: %lu\n",
            tp->prgname, rd_count, rd_err_cnt);

        mprintf("[%20s] Frame size: %u\n", tp->prgname, ep->info.frame_size);
        dprintf("[%20s] Start = %lu, end = %lu\n", tp->prgname, start, end);
        rate = (((double) (rd_count * ep->info.frame_size)) *
            ((double)getCyclesPerSec() / MEGABYTE)) / (end - start);
        mprintf("[%20s] Rate %lf MB/sec (%lf Mfps; %lf us/f)\n", tp->prgname,
            rate, (rate * MEGABYTE)/(double)(ep->info.frame_size * 1000000),
            (double)(ep->info.frame_size * 1000000)/(rate * MEGABYTE));
        mprintf("[%20s] event count: %u\n", tp->prgname, ep->evt_cnt);
    }

    /* check latency data */
    if (tp->latencytest) {
        char filename[MAX_OUTPUT_PATH_LEN * 2];
        /* from write to noti */

        snprintf(filename, (MAX_OUTPUT_PATH_LEN * 2) - 1, "%s/%s",
                 (NULL != tp->output_path) ? tp->output_path : DEFAULT_OUTPUT_PATH_NAME,
                 NOTIFY_OUTPUT_FILENAME);
        write_latency_statistics(filename, tp->startTimestamp, tp->latencyBuf, tp->iterations);
        show_latency_statistics("WRITE TO NOTIFICATION ---------------",
            tp->latencyBuf, tp->iterations);

        /* from write to read */
        snprintf(filename, (MAX_OUTPUT_PATH_LEN * 2) - 1, "%s/%s",
                 (NULL != tp->output_path) ? tp->output_path : DEFAULT_OUTPUT_PATH_NAME,
                 END2END_OUTPUT_FILENAME);
        write_latency_statistics(filename,tp->startTimestamp, tp->latencyBuf2, tp->iterations);
        show_latency_statistics("WRITE TO READ -----------------------",
            tp->latencyBuf2, tp->iterations);
    }

fail:
#if (LW_IS_SAFETY == 0) && defined(IVC_EVENTLIB)
    if (tp->eventlib != 0) {
        dump_lwsciipc_eventlibs(LWSCIIPC_READ_TEST_DUMP_FILENAME);
        lwsciipc_eventlib_close();
    }
#endif
    if (rd_err_cnt > 0)
        return EIO;
    else
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

static LwSciError wait_multi_events(int32_t value, bool *newEvent)
{
    test_params *tp = &s_params;
    struct endpoint *ep;
#ifdef __QNX__
    struct _pulse pulse;
#endif
    uint32_t event = 0;
    int32_t timeout;
    uint32_t i;
    uint32_t conn_cnt;
    bool gotEvent;
    bool bothEventSet = false;
    LwSciError err;

    while(!s_Stop) {
        gotEvent = false;
        conn_cnt = 0;
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
            if (event & LW_SCI_IPC_EVENT_CONN_EST_ALL) {
                conn_cnt++;
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

        /* all endpoint connections are now established */
        if (conn_cnt == tp->epCnt) {
            /* flag to trigger local event */
            tp->allEstablished = true;
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
            err = LwMsgReceivePulse(NULL, &pulse, sizeof(pulse), NULL, NULL);
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

static void display_stress_progress(uint64_t cycles)
{
    test_params *tp = &s_params;
    uint32_t sec = get_sec_from_cycles(cycles);
    static uint32_t oldsec = 0;

    if (sec != oldsec) {
        if ((tp->logpersec != 0) && ((sec - oldsec) >= tp->logpersec)) {
            printf("%20s:%s:%s:RD%5ld/%5ld:time(%s)\n",
                tp->prgname,
                tp->ep[0].chname,
                tp->ep[tp->epCnt-1].chname,
                tp->TotalFrmCnt, (tp->iterations * tp->epCnt),
                get_time_str(sec));
            oldsec = sec;
        }
    }
}


static void display_stress_statistics(uint64_t cycles)
{
    test_params *tp = &s_params;
    uint32_t sec = get_sec_from_cycles(cycles);

    printf("[%4s]%20s:%s:%s:nf%d:fs%d:%s:LOC%5ld/%5ld:RD%5ld/%5ld:ERR%d:"
        "rate%.2lfMB/s:time(%s)\n",
        (tp->errcnt > 0)?"FAIL":"PASS",
        tp->prgname,
        tp->ep[0].chname,
        tp->ep[tp->epCnt-1].chname,
        tp->ep[0].info.nframes,
        tp->ep[0].info.frame_size,
        (tp->evtSvcFlag)?"LWSCIEVT":"NATIVE",
        tp->TotRxLocEvtCnt, tp->TotTxLocEvtCnt,
        tp->TotalFrmCnt, (tp->iterations * tp->epCnt),
        tp->errcnt,
        (double)(tp->TotalFrmCnt * tp->ep[0].info.frame_size *
            ((double)getCyclesPerSec() / MEGABYTE) / cycles),
        get_time_str(sec));
}

#ifdef __QNX__
/* statistics for TT_STRESS3 */
static void display_stress3_progress(uint64_t cycles)
{
    test_params *tp = &s_params;
    struct endpoint *ep;
    uint32_t sec = get_sec_from_cycles(cycles);
    static uint64_t oldsec = 0;
    uint32_t coeff = 1;
    uint32_t i;

    /* add zero copy and peek/poke test count */
    if (tp->morerwapi) {
        coeff = 3;
    }

    /* count data from all threads */
    tp->TotalFrmCnt = 0;
    for (i = 0; i < tp->epCnt; i++) {
        ep = &tp->ep[i];
        tp->TotalFrmCnt += ep->rd_count;
    }

    if (sec != oldsec) {
        if ((tp->logpersec != 0) && ((sec - oldsec) >= tp->logpersec)) {
            printf("%20s:%s:%s:RD%5ld/%5ld:time(%s)\n",
                tp->prgname,
                tp->ep[0].chname,
                tp->ep[tp->epCnt-1].chname,
                tp->TotalFrmCnt, (tp->iterations * coeff * tp->epCnt),
                get_time_str(sec));
            oldsec = sec;
        }
    }
}

/* statistics for TT_STRESS3 */
static void display_stress3_statistics(uint64_t cycles)
{
    test_params *tp = &s_params;
    struct endpoint *ep;
    uint32_t sec = get_sec_from_cycles(cycles);
    uint32_t coeff = 1;
    uint32_t i;

    /* add zero copy and peek/poke test count */
    if (tp->morerwapi) {
        coeff = 3;
    }

    /* count data from all threads */
    tp->TotalFrmCnt = 0;
    for (i = 0; i < tp->epCnt; i++) {
        ep = &tp->ep[i];
        tp->TotalFrmCnt += ep->rd_count;
    }

    printf("[%4s]%20s:%s:%s:nf%d:fs%d:RD%5ld/%5ld:ERR%d:"
        "rate%.2lfMB/s:time(%s)\n",
        (tp->errcnt > 0)?"FAIL":"PASS",
        tp->prgname,
        tp->ep[0].chname,
        tp->ep[tp->epCnt-1].chname,
        tp->ep[0].info.nframes,
        tp->ep[0].info.frame_size,
        tp->TotalFrmCnt, (tp->iterations * coeff * tp->epCnt),
        tp->errcnt,
        (double)(tp->TotalFrmCnt * tp->ep[0].info.frame_size *
            ((double)getCyclesPerSec() / MEGABYTE) / cycles),
        get_time_str(sec));
}
#endif /* __QNX__ */

int32_t read_stress_test(void)
{
    test_params *tp = &s_params;
    struct endpoint *ep;
    uint32_t *ptr;
    int32_t bytes;
    uint64_t cstart = 0UL;
    uint64_t cend = 0UL;
    uint32_t start;
    uint32_t end;
    uint32_t i;
    uint32_t total_iterations;
    bool newEvent[MAX_EVENTNOTIFIER]; /* localEvents + nativeEvents */
    LwSciError err;

    total_iterations = (tp->epCnt * tp->iterations) +
        (tp->loc_iterations * tp->locEvtCnt);

    mprintf("[%20s] Stress Test mode (loop: %ld)\n",
        tp->prgname, tp->iterations);

    while (((tp->TotalFrmCnt + tp->TotRxLocEvtCnt) <
        total_iterations) && !s_Stop) {

        memset(newEvent, 0, sizeof(newEvent));
        err = wait_multi_events(LW_SCI_IPC_EVENT_READ, newEvent);

        if (cstart == 0UL) {
            cstart = getClockCycles();  /* onetime set */
        }

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
                    if ((ep->prevData != 0xDEADBEEF) &&
                        (ptr[0] != (ep->prevData+1))) {
                        tp->errcnt++;   /* data mismatch error */
                        printf("[%20s]%s:RD#%d:%d,%d(%s)\n",
                            tp->prgname, __func__, ep->rxCnt, ptr[0],
                            ep->prevData, ep->chname);
                    }
                    ep->rxCnt++;
                    dprintf("[%20s]%s:RD#%d:%d,%d(%s)\n",
                        tp->prgname, __func__, ep->rxCnt, ptr[0],
                        ep->prevData, ep->chname);
                    ep->prevData = ptr[0];
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

        cend = getClockCycles();
        display_stress_progress(cend-cstart);
    }

    cend = getClockCycles();

    /* callwlate Total Tx Local Event */
    for (i=0;i<tp->locEvtCnt;i++) {
        tp->TotTxLocEvtCnt += tp->loc[i].txEcnt;
    }

    /* Show statistics */
    display_stress_statistics(cend-cstart);

    /* terminate thread */
    if (tp->errcnt > 0) {
        s_Stop = 1;
    }

    if (tp->errcnt > 0)
        return EIO;
    else
        return 0;
}

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

        dprintf("[%20s][T:%d][LO%d]: Signal#%d)\n", tp->prgname,
            (unsigned int)loc->tid, loc->idx, i);

        /* waiting time to prevent signal overlapping.
         * [NOTE] this is not actual usecase, just for event counting
         */
        while(!loc->received && !s_Stop) {
            usleep(1000);   /* 1ms */
        }
    }

    dprintf("[%20s][T:%u][LO%d]: completed\n",
        tp->prgname, (unsigned int)loc->tid, loc->idx);

    return NULL;
}

#ifdef __QNX__
/* stress test for parallel exelwtion of API
 * - thread routine
 */
static void *parallel_exelwtion_thread(void *arg)
{
    test_params *tp = &s_params;
    struct endpoint *ep = (struct endpoint *)arg;
    LwSciError err;
    int retval = 0;

    /* init LwSciIpc resources */
    err = init_resources_for_PE_test(ep);
    if (err != LwSciError_Success) {
        retval = -1;
        goto fail;
    }

    retval = read_test(ep);
    if (tp->morerwapi) {
        retval |= read_test_zc(ep);
        retval |= read_test_peek(ep);
    }

fail:
    if (ep->initFlag) {
        release_resources_for_PE_test(ep);
        ep->initFlag = false;
    }
    ep->retval = retval;

    return NULL;
}

/* stress test for parallel exelwtion of API
 * - main test routine
 */
static void *parallel_exelwtion_main(void *arg)
{
    test_params *tp = &s_params;
    struct endpoint *ep;
    int *result = (int *)arg;
    uint32_t errcnt = 0;
    uint32_t idx = 0;
    struct timespec tm;
    LwSciError err;
    uint32_t done_cnt = 0;
    uint64_t cstart = 0UL;
    uint64_t cend;

    err = LwSciIpcInit();
    if (err != LwSciError_Success) {
        printf("[%20s] %s: LwSciIpcInit: fail (%d)\n",
            tp->prgname, __func__, err);
        goto fail;
    }

    /* create thread per endpoint */
    for (idx=0;idx < tp->epCnt; idx++) {
        ep = &tp->ep[idx];

        /* set API type */
        if (idx < (tp->epCnt / 2)) {
            ep->apitype = API_LEGACY;
        }
        else {
            ep->apitype = API_LWSCIEVENT;
        }

        /* set ep name */
        sprintf(ep->chname, "%s%04d", tp->epprefix, (tp->epstart + idx));

        pthread_create(&ep->tid, NULL, &parallel_exelwtion_thread, (void *)ep);
    }

    cstart = getClockCycles();
    while(!s_Stop && (done_cnt < tp->epCnt)) {
        done_cnt = 0;
        /* wait for completion */
        for (idx=0;idx < tp->epCnt; idx++) {
            ep = &tp->ep[idx];

            if (ep->doneFlag) {
                done_cnt++;
                continue;
            }

            clock_gettime(CLOCK_MONOTONIC, &tm);
            tm.tv_sec += 1;

            err = pthread_timedjoin_monotonic(ep->tid, NULL, &tm);
            if (err == ETIMEDOUT) {
                cend = getClockCycles();
                display_stress3_progress(cend-cstart);
                break;
            }
            else if (err == EOK) {
                ep->doneFlag = true;
                done_cnt++;
            }

            if (ep->retval != 0) {
                errcnt++;
            }
        }
    }

    cend = getClockCycles();
    display_stress3_statistics(cend-cstart);

    LwSciIpcDeinit();

fail:
    if (errcnt > 0) {
        *result = EFAULT;
    }
    else {
        *result = EOK;
    }

    return NULL;
}
#endif /* __QNX__ */

void *read_test_main(void *arg)
{
    test_params *tp = &s_params;
    struct endpoint *ep = &tp->ep[0];
    int *result = (int *)arg;
    int retval = 0;
    uint32_t i;
    uint32_t idx = 0;
    LwSciError err;

    /* create threads for local event */
    if (tp->locEvtCnt > 0) {
        for (i=0;i<tp->locEvtCnt;i++) {
            pthread_create(&tp->loc[i].tid, NULL,
                &signal_localevent_thread, &tp->loc[i]);
            dprintf("[%20s] signal_localevent_thread (tid:%u)\n",
                tp->prgname, (unsigned int)tp->loc[i].tid);
        }
    }

    while (idx < tp->initIterations) {
        if (tp->testtype == TT_STRESS2) {
            /* set ep name */
            sprintf(ep->chname, "%s%04d", tp->epprefix, (tp->epstart + idx));
        }

        /* init LwSciIpc resources */
        err = init_resources();
        if (err != LwSciError_Success) {
            retval = -1;
            goto fail;
        }

        switch(tp->testtype) {
            case TT_NORMAL:
                retval = read_test(NULL);
                if (tp->morerwapi) {
                    retval |= read_test_zc(NULL);
                    retval |= read_test_peek(NULL);
                }
                break;
            case TT_PERF:
                retval = read_perf_test();
                break;
            case TT_STRESS:
                retval = read_stress_test();
                break;
            case TT_STRESS2:
                retval = read_perf_test();
                break;
        }

        /* join threads */
        if (tp->locEvtCnt > 0) {
            for (i=0;i<tp->locEvtCnt;i++) {
                pthread_join(tp->loc[i].tid, NULL);
                dprintf("[%20s] local event thread is terminated (tid:%u)\n",
                    tp->prgname, (unsigned int)tp->loc[i].tid);
            }
        }

fail:
        if (tp->initFlag) {
            release_resources();
            tp->initFlag = false;
        }
        *result = retval;

        if (retval != 0) {
            break;
        }

        idx++;
    } /* initIterations loop */

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
    struct endpoint *ep = &s_params.ep[0];
    int opt;
    static int retval = -1;
    int exitcode;
#ifdef __QNX__
    bool prio_set_f = false;
    bool pulse_timeout = false;

    change_slogbuf_size();

    tp->backoff_delay = DEFAULT_BACKOFF_NS;
    tp->prio = SIGEV_PULSE_PRIO_INHERIT; /* default: -1, INTR: 0x15(21) */
    tp->range_lo = 0;
    tp->range_hi = INT_MAX;
    tp->timeout = LW_SCI_EVENT_INFINITE_WAIT; /* default timeout */
    tp->pause = false;
#endif /* __QNX__ */
    pthread_t tid;
    tp->iterations = DEFAULT_COUNT;
    tp->initIterations = DEFAULT_INIT_COUNT;

    dprintf("[%20s] enter LwSciIpc test\n", argv[0]);

    while ((opt = getopt(argc, argv,
#if (LW_IS_SAFETY == 0) && defined(IVC_EVENTLIB)
                         "a:bc:d:ehi:l:mo:psu:vw:y:ABE:I:J:L:MQRS:TW:X:Z@"
#else
                         "a:bc:d:ehi:l:mo:psu:vw:y:ABE:I:J:L:MQRS:W:X:Z@"
#endif
                         )) != -1)
    {
        switch (opt)
        {
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
                            retval = 0;
                        }
                    }
                    if (retval != 0) {
                        mprintf("%s: range_lo/range_hi not specified correctly\n", argv[0]);
                        goto done;
                    }
                    break;
                }
            case 'b':
                tp->bidirtest = 1;
                break;
#endif /* __QNX__ (a,b) */
            case 'c':
                {
                    unsigned int slen = strlen(optarg);
                    if (slen > LWSCIIPC_MAX_ENDPOINT_NAME - 1) {
                        slen = LWSCIIPC_MAX_ENDPOINT_NAME - 1;
                    }
                    memcpy(ep->chname, optarg, slen);
                    ep->chname[slen] = '\0';
                    tp->epCnt = 1;
                    break;
                }
#ifdef __QNX__
            case 'd':
                tp->backoff_delay = strtoul(optarg, NULL, 0);
                break;
            case 'e':
                tp->msgdroptest = 1;
                break;
#endif /* __QNX__ (d,e) */
            case 'h':
                print_usage(argv);
                retval = 0;
                goto done;
#ifdef __QNX__
            case 'i':
                tp->prio = strtoul(optarg, NULL, 0);
                prio_set_f = true;
                break;
#endif /* __QNX__ (i) */
            case 'l':
                tp->iterations = strtoul(optarg, NULL, 0);
                break;
            case 'm':
                tp->ignoredata = 1;
                break;
            case 'o':
		/* set a directory path to store all latency data as a file.
		   default is "/tmp". */
                {
                    unsigned int slen = strlen(optarg);
                    if (MAX_OUTPUT_PATH_LEN > slen) {
                        tp->output_path = malloc(strlen(optarg) + 1);
                        if (NULL != tp->output_path) {
                            strcpy(tp->output_path, optarg);
                        }
                    }
                }
                break;
            case 'p':
                tp->testtype = TT_PERF;
                break;
            case 's':
                tp->evtSvcFlag = true;
#ifdef __QNX__
                if (prio_set_f) {
                    mprintf("%s: you can't use -i option together\n", argv[0]);
                    goto done;
                }
#endif  /* __QNX__ */
                break;
#ifdef __QNX__
            case 'u':
                /* Set UID:GID */
                {
                    char* token = strtok(optarg, ":");
                    if (token != NULL) {
                        tp->uid = (uid_t)strtoul(token, NULL, 10);
                        while ((token = strtok(NULL, ":,")) != NULL) {
                            tp->gid[tp->num_gid] = (gid_t)strtoul(token, NULL, 10);
                            tp->num_gid++;
                            retval = 0;
                        }
                    }
                    if (retval != 0) {
                        mprintf("%s: UID/GID not specified correctly\n", argv[0]);
                        goto done;
                    }
                    break;
                }
#endif  /* __QNX__ (u) */
            case 'v':
                s_verboseFlag = true;
                break;
            case 'y':
                {
                    uint32_t val;
                    errno = EOK;
                    val = strtoul(optarg, NULL, 0);
                    if (val >= 0 && errno == 0) { // decimal including 0
                        tp->y_timeout = val;
                    }
                    else {
                        tp->y_timeout = LATENCY_TIMEOUT_MS;
                    }

                    tp->startTimestamp = calloc(MAX_LATENCY_SAMPLE,
                                    sizeof(uint64_t));
                    tp->latencyBuf = calloc(MAX_LATENCY_SAMPLE,
                                    sizeof(uint64_t));
                    tp->latencyBuf2 = calloc(MAX_LATENCY_SAMPLE,
                                    sizeof(uint64_t));
                    tp->clockBuf = calloc(MAX_LATENCY_SAMPLE,
                                    sizeof(uint64_t));

                    tp->latencytest = 1;
                    tp->testtype = TT_PERF;
                }
                break;
#ifdef __QNX__
            case 'A':
                tp->timestampS = 1;
                break;
#endif /* __QNX__ (A) */
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
#ifdef  __QNX__
            case 'I':   /* Stress Test: Multiple Invocation of API */
                {
                    char* token = strtok(optarg, ":");

                    if (token != NULL) {
                        strncpy(tp->epprefix, token, sizeof(tp->epprefix));
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
                        tp->initIterations = (int32_t)strtoul(token, NULL, 10);
                    }
                    token = strtok(NULL, ":");
                    if (token != NULL) {
                        tp->logpersec = (int32_t)strtoul(token, NULL, 10);
                    }
                    if (tp->epstart <= tp->epend) {
                        retval = 0;
                    }
                    else {
                        retval = -1;
                    }

                    if (retval != 0) {
                        mprintf("%s: option is invalid\n", argv[0]);
                        goto done;
                    }

                    tp->epCnt = 1;
                    tp->testtype = TT_STRESS2;
                    s_quietFlag = true; /* set quiet flag in stress test */
                }
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
#endif /* __QNX__ (I) */
            case 'L':   /* Local Event */
                tp->locEvtCnt = strtoul(optarg, NULL, 0);
                if (tp->locEvtCnt > MAX_LOCALEVENT) {
                    printf("%s: local event count is out of range\n", argv[0]);
                    goto done;
                }
                break;
            case 'M':
                tp->morerwapi = 1;
                break;
            case 'Q':
                s_quietFlag = true;
                break;
            case 'R':
                tp->useThread = true;
                break;
#ifdef __QNX__
            case 'S': /* Stress Test: parallel process instance, CPU load, memleak */
                {
                    char* token = strtok(optarg, ":");

                    if (token != NULL) {
                        strncpy(tp->epprefix, token, sizeof(tp->epprefix));
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
                    if (tp->epstart <= tp->epend) {
                        retval = 0;
                    }
                    else {
                        retval = -1;
                    }

                    if (retval != 0) {
                        mprintf("%s: option is invalid\n", argv[0]);
                        goto done;
                    }

                    tp->epCnt = tp->epend - tp->epstart + 1;
                    tp->testtype = TT_STRESS;
                    s_quietFlag = true; /* set quiet flag in stress test */
                }
                break;
#if (LW_IS_SAFETY == 0) && defined(IVC_EVENTLIB)
            case 'T':
                tp->eventlib = 1;
                break;
#endif
            case 'W':
                {
                    uint32_t val;
                    errno = EOK;
                    val = strtoul(optarg, NULL, 0);
                    if (val >= 0 && errno == 0) { // decimal including 0
                        tp->timeout = val;
                    }
                    pulse_timeout = true;
                }
                break;
            case 'X':
                {
                    /* Parallel Exelwtion of APIs (stress test) */
                    char* token = strtok(optarg, ":");

                    if (token != NULL) {
                        strncpy(tp->epprefix, token, sizeof(tp->epprefix));
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
                    if (tp->epstart <= tp->epend) {
                        retval = 0;
                    }
                    else {
                        retval = -1;
                    }

                    if (retval != 0) {
                        mprintf("%s: option is invalid\n", argv[0]);
                        goto done;
                    }

                    tp->epCnt = tp->epend - tp->epstart + 1;
                    tp->testtype = TT_STRESS3;
                    s_quietFlag = true; /* set quiet flag in stress test */
                }
                break;
            case 'Z':
                tp->timestampE = 1;
                break;
            case '@':
                tp->pause = true;
                break;
#endif /* __QNX__ (S,T,W,X,Z) */
            case ':':
                fprintf(stderr, "Option `-%c` requires an argument.\n", optopt);
                goto done;
            case '?':
                if (isprint(optopt))
                {
                    fprintf(stderr, "Unknown option `-%c`.\n", optopt);
                }
                else
                {
                    fprintf(stderr, "Unknown option ``\\x%x`.\n", optopt);
                }
                goto done;
            default:
                abort();
        }
    }
    /* handle parameter errors */
#ifdef __QNX__
    if (
        pulse_timeout &&
        tp->evtSvcFlag) {
        mprintf("You can't use -W and -E option at the same time\n");
        goto done;
    }
#endif /* __QNX__ */

    tp->prgname = basename(argv[0]);
    if (tp->epCnt == 0 && tp->locEvtCnt == 0) {
        fprintf(stderr, "need one endpoint or localevent at least\n");
        print_usage(argv);
        goto done;
    }

    if ((tp->testtype < TT_STRESS) || (tp->testtype > TT_STRESS3)) {
        if (!ep->chname[0]) {
            fprintf(stderr, "need to give LwSciIpc endpoint name as input\n");
            print_usage(argv);
            goto done;
        }

        /* Display configuration */
        dprintf("[%20s] endpoint name is *%s*\n", tp->prgname, ep->chname);
    }

    mprintf("[%20s] iteration for Native Event : %ld\n", tp->prgname,
            tp->iterations);
    if (tp->evtSvcFlag) {
        if (tp->locEvtCnt == 0) {
            tp->loc_iterations = 0;
        }
        mprintf("[%20s] Use LwSciEventService (timeout:%dmsec)\n",
            tp->prgname, tp->timeout);
        mprintf("[%20s] Local Event count: %d\n", tp->prgname, tp->locEvtCnt);
        mprintf("[%20s] iteration for Local Event : %ld\n", tp->prgname,
            tp->loc_iterations);
    }
    else {
#ifdef LINUX
        mprintf("[%20s] select (timeout:%dmsec)\n",
            tp->prgname, tp->timeout);
#endif /* LINUX */
#ifdef __QNX__
        mprintf("[%20s] MsgReceivePulse (timeout:%dmsec)\n",
            tp->prgname, tp->timeout);
#endif /* __QNX__ */
    }
    if (tp->latencytest) {
        mprintf("[%20s] IVC notification latency test (timeout:%dmsec)\n",
            tp->prgname, tp->y_timeout);
    }

    /* setup sig handler */
    setup_termination_handlers();

#ifdef __QNX__
    if (tp->timestampS == 1) {
        printf("TIMESTAMP(S):%s\n",
            get_time_str(get_sec_from_cycles(ClockCycles())));
    }
#endif /* __QNX__ */

    os_setup();

    if (tp->useThread) {
        pthread_create(&tid, NULL, &read_test_main, (void *)&retval);
        pthread_join(tid, NULL);
    }
    else
    {
        switch(tp->testtype) {
#ifdef __QNX__
            case TT_STRESS3:
                parallel_exelwtion_main(&retval);
                break;
#endif /* __QNX__ */
            default :
                read_test_main(&retval);
                break;
        }
    }

done:
    if (retval != 0) {
        exitcode = EXIT_FAILURE;
        mprintf("[%20s] : test FAILED\n", tp->prgname);
    }
    else {
        exitcode = EXIT_SUCCESS;
        mprintf("[%20s] : test PASSED\n", tp->prgname);

#ifdef __QNX__
        if (tp->timestampE == 1) {
            printf("TIMESTAMP(E):%s\n",
                get_time_str(get_sec_from_cycles(ClockCycles())));
        }
#endif /* __QNX__ */
    }

    writeTestLog(exitcode);
#ifdef __QNX__
    while (tp->pause) {
        pause();
    }
#endif /* __QNX__ */

    return exitcode;
}

