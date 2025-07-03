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
 * IPC unit test on QNX.
 */

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <unistd.h>
#include <inttypes.h>
#include <string.h>
#include <errno.h>
#include <unistd.h>
#include <stdbool.h>
#include <pthread.h>
#include <libgen.h>
#include <sys/neutrino.h>
#include <sys/syspage.h>
#include <sys/procmgr.h>
#include <sys/mman.h>

#include <lwsciipc_internal.h>

#define DEBUG  1
#define DEBUG2 0

#define DEFAULT_ITERATIONS 3
#define RANDOM     -1
#define COMPARE_ENDS -1

#define WAIT_TIME_MS 2000U /* 2 seconds wait */
#define TEST_HANDLE_SIZE 512

#define TEST_PULSE_PRIO     21
#define TEST_PULSE_CODE1    16
#define TEST_PULSE_CODE2    32
#define MAX_GID_NUM 255

struct endpoint {
    char chname[LWSCIIPC_MAX_ENDPOINT_NAME];    /* endpoint name */
    LwSciIpcEndpoint h;    /* LwSciIpc handle */
    LwSciIpcEndpointInfo info; /* endpoint info */
    LwSciIpcEndpointInfoInternal infoi; /* internal endpoint info */
    LwSciEventNotifier *notifier;
#ifdef __QNX__
    int32_t chid;
    int32_t coid;
    int16_t prio;
    int16_t code;
    void *value;
#endif /* __QNX__ */
    int32_t id;
    uint8_t *buf; /*test buffer */
    LwSciIpcEndpoint tsthandle;    /* test handle */
};

typedef struct {
    struct endpoint e1;
    struct endpoint e2;
    uint32_t iterations;
    uint32_t frame_size; /* test framesize */
    uint32_t test_cnt, err_cnt;
    int32_t range_lo; /* lower limit on range of ability */
    int32_t range_hi; /* upper limit on range of ability */
    uid_t uid;
    gid_t gid[MAX_GID_NUM]; /* gid[0]=GID, gid[1..255]=SGIDs */
    uint32_t num_gid;
    char *prgname;

    bool negative;      /* negative test for accessing resource */
    uint64_t phyaddr;    /* physical memory offset for negative test */
    uint64_t physize;   /* physical memory size for negative test */
    uint32_t irq;   /* IRQ number for negative test */
#ifdef __QNX__
    uint32_t timeout;   /* event waiting timeout (msec) */
    LwSciEventLoopService *eventLoopService;
#endif /* __QNX__ */
} test_params;

static test_params s_params;
static int32_t s_Stop;

/* c.f. lwsciipc_os_error.c */
typedef struct LwSciErrMap {
    LwSciError scierr;
    int32_t oserr;
    const char *oserrstr;
} LwSciErrMap;

static const LwSciErrMap LwSciCoreErrs[] =
{
    {LwSciError_Success,
     0,               "EOK"},
    {LwSciError_NotImplemented,
     ENOSYS,          "ENOSYS"},
    {LwSciError_NotSupported,
     ENOTSUP,         "ENOTSUP"},
    {LwSciError_BadParameter,
     EILWAL,          "EILWAL"},
    {LwSciError_Timeout,
     ETIMEDOUT,       "ETIMEDOUT"},
    {LwSciError_InsufficientMemory,
     ENOMEM,          "ENOMEM"},
    {LwSciError_AccessDenied,
     EACCES,          "EACCES"},
    {LwSciError_TooBig,
     E2BIG,           "E2BIG"},
    {LwSciError_TryItAgain,
     EAGAIN,          "EAGAIN"},
    {LwSciError_BadFileDesc,
     EBADF,           "EBADF"},
    {LwSciError_Busy,
     EBUSY,           "EBUSY"},
    {LwSciError_ConnectionReset,
     ECONNRESET,      "ECONNRESET"},
    {LwSciError_ResourceDeadlock,
     EDEADLK,         "EDEADLK"},
    {LwSciError_FileExists,
     EEXIST,          "EEXIST"},
    {LwSciError_BadAddress,
     EFAULT,          "EFAULT"},
    {LwSciError_FileTooBig,
     EFBIG,           "EFBIG"},
    {LwSciError_InterruptedCall,
     EINTR,           "EINTR"},
    {LwSciError_IO,
     EIO,             "EIO"},
    {LwSciError_IsDirectory,
     EISDIR,          "EISDIR"},
    {LwSciError_TooManySymbolLinks,
     ELOOP,           "ELOOP"},
    {LwSciError_TooManyOpenFiles,
     EMFILE,          "EMFILE"},
    {LwSciError_FileNameTooLong,
     ENAMETOOLONG,    "ENAMETOOLONG"},
    {LwSciError_FileTableOverflow,
     ENFILE,          "ENFILE"},
    {LwSciError_NoSuchDevice,
     ENODEV,          "ENODEV"},
    {LwSciError_NoSuchEntry,
     ENOENT,          "ENOENT"},
    {LwSciError_NoSpace,
     ENOSPC,          "ENOSPC"},
#ifdef ENOTRECOVERABLE
    {LwSciError_MutexNotRecoverable,
     ENOTRECOVERABLE, "ENOTRECOVERABLE"},
#endif
    {LwSciError_NoSuchDevAddr,
     ENXIO,           "ENXIO"},
    {LwSciError_Overflow,
     EOVERFLOW,       "EOVERFLOW"},
#ifdef EOWNERDEAD
    {LwSciError_LockOwnerDead,
     EOWNERDEAD,      "EOWNERDEAD"},
#endif
    {LwSciError_NotPermitted,
     EPERM,           "EPERM"},
    {LwSciError_ReadOnlyFileSys,
     EROFS,           "EROFS"},
    {LwSciError_NoSuchProcess,
     ESRCH,           "ESRCH"},
    {LwSciError_TextFileBusy,
     ETXTBSY,         "ETXTBSY"},
    {LwSciError_IlwalidIoctlNum,
     ENOTTY,          "ENOTTY"},
    {LwSciError_NoData,
     ENODATA,         "ENODATA"},
    {LwSciError_AlreadyInProgress,
     EALREADY,        "EALREADY"},
    {LwSciError_NoDesiredMessage,
     ENOMSG,          "ENOMSG"},
    {LwSciError_MessageSize,
     EMSGSIZE,        "EMSGSIZE"},
#ifdef __QNX__
    {LwSciError_NoRemote,
     ENOREMOTE, "ENOREMOTE"},
    {LwSciError_CorruptedFileSys,
     EBADFSYS,  "EBADFSYS"},
#endif /* __QNX__ */
};
static const size_t LwSciCoreErrCount = sizeof(LwSciCoreErrs)
                                       / sizeof(LwSciCoreErrs[0]);



#define LWIPC_GEN_FMT   "%-40s%02d: %4s (%d)\n"
#define LWIPC_FAIL_FMT  "%-40s%02d: %4s (%d)\n"
#define LWIPC_FAIL_FMTD "%-40s%02d: %4s (%d, bytes:%d)\n"

void unit_test_err_colwersion(void);
void unit_test_init(void);
void unit_test_open_endpoint(struct endpoint *ep, int32_t id, char *chname, LwSciError val);
#ifdef __QNX__
void unit_test_set_event_pulse(struct endpoint *ep, int16_t priority,
    int16_t code, void *value);
#endif /* __QNX__ */
void unit_test_get_ch_info(struct endpoint *ep);
void unit_test_reset(struct endpoint *ep);
void unit_test_send_only(struct endpoint *ep, LwSciError val);
void unit_test_send_with_return(struct endpoint *ep, int32_t val);
void unit_test_send_only_zc(struct endpoint *ep, LwSciError val);
void unit_test_send_with_return_zc(struct endpoint *ep, int32_t val);
void unit_test_send_only_poke(struct endpoint *ep, LwSciError val);
void unit_test_send_with_return_poke(struct endpoint *ep, int32_t val);
void unit_test_receive_only(struct endpoint *ep, LwSciError val);
void unit_test_receive_with_return(struct endpoint *ep, int32_t val);
void unit_test_receive_only_zc(struct endpoint *ep, LwSciError val);
void unit_test_receive_with_return_zc(struct endpoint *ep, int32_t val);
void unit_test_receive_only_peek(struct endpoint *ep, LwSciError val);
void unit_test_receive_with_return_peek(struct endpoint *ep, int32_t val);
void unit_test_poll_receive_only(struct endpoint *ep);
void unit_test_poll_receive_only_zc(struct endpoint *ep);
void unit_test_poll_receive_only_peek(struct endpoint *ep);
void unit_test_poll_notification(test_params *tp);
void unit_test_poll_send(struct endpoint *ep);
void unit_test_poll_send_zc(struct endpoint *ep);
void unit_test_poll_send_poke(struct endpoint *ep);
void unit_test_poll_receive(struct endpoint *ep);
void unit_test_poll_receive_zc(struct endpoint *ep);
void unit_test_poll_receive_peek(struct endpoint *ep);
void unit_test_poll_send_receive(struct endpoint *tx, struct endpoint *rx);
void unit_test_poll_send_receive_zc(struct endpoint *tx, struct endpoint *rx);
void unit_test_poll_send_receive_pokepeek(struct endpoint *tx, struct endpoint *rx);
void unit_test_poll_event(struct endpoint *ep, int32_t value);
void unit_test_wait_event(struct endpoint *ep, int32_t value);
void unit_test_notification(test_params *tp);
void unit_test_wait_n_send(struct endpoint *ep);
void unit_test_wait_n_send_zc(struct endpoint *ep);
void unit_test_wait_n_send_poke(struct endpoint *ep);
void unit_test_wait_n_receive(struct endpoint *ep);
void unit_test_wait_n_receive_zc(struct endpoint *ep);
void unit_test_wait_n_receive_peek(struct endpoint *ep);
void unit_test_send_receive(struct endpoint *tx, struct endpoint *rx);
void unit_test_send_receive_zc(struct endpoint *tx, struct endpoint *rx);
void unit_test_send_receive_pokepeek(struct endpoint *tx, struct endpoint *rx);
void unit_test_compare_data(struct endpoint *tx, struct endpoint *rx, int32_t val);
void unit_test_close_endpoint(struct endpoint *ep);
void unit_test_deinit(test_params *tp);

void unit_test_fill_channel(struct endpoint *ep);
void unit_test_drain_channel(struct endpoint *ep);
void unit_test_write_after_fill(struct endpoint *ep);
void unit_test_read_after_drain(struct endpoint *ep);

void unit_test_call_apis(struct endpoint *ep, LwSciIpcEndpoint h, LwSciError val);
void unit_test_call_apis2(struct endpoint *ep, LwSciIpcEndpoint h, bool val);
void unit_test_error_handle(struct endpoint *ep);
void unit_test_NULL_pointer(struct endpoint *ep, void *ptr, LwSciError val);

void unit_test_init_event_service(void);
void unit_test_open_endpoint_eventservice(struct endpoint *ep, int32_t id,
    char *chname, LwSciError val);
void unit_test_wait_event_notifier(struct endpoint *ep1, struct endpoint *ep2);
void unit_test_delete_event_notifier(struct endpoint *ep);
void unit_test_deinit_event_service(void);

void unit_test_negative_res_access(void);

void setup_buffer(test_params *tp);
void fill_buffer(struct endpoint *ep, int32_t val);
void release_resources(test_params *tp);
void print_buffer(const char *head, uint8_t *buf, uint32_t length);
void print_ids(const char *str);
#ifdef __QNX__
void drop_privileges(void);
int32_t LwMsgReceivePulse(struct endpoint *ep, struct _pulse *pulse,
    size_t bytes, struct _msg_info *info);
#endif /* __QNX__ */

static void print_usage(const char *str)
{
    printf("%s: Options\n", str);
    printf(" -a <lo:hi>              : "
            "set lo/hi limit on range of ability (hexadecimal)\n");
    printf(" -n <phyAddr,phySize,IRQ>: "
            "negative test for resource access "
            "(phyAddr/Size: hexadecimal, IRQ: decimal)\n");
    printf(" -r <endpoint name>      : "
            "receiver endpoint name for loopback test\n");
    printf(" -s <endpoint name>      : "
            "sender endpoint name for loopback test\n");
    printf(" -u <uid:gid>            : "
            "UID and GID setting for test\n");
    printf(" -w <timeout>            : "
            "event waiting timeout in msec\n");
    printf(" -h                      : Help\n");
}

static void sig_handler(int sig_num)
{
    test_params *tp = &s_params;

    s_Stop = 1;

    LwSciIpcCloseEndpoint(tp->e1.h);
    LwSciIpcCloseEndpoint(tp->e2.h);
    LwSciIpcDeinit();

    if (s_Stop) {
        exit(1);
    }
}

static void setup_termination_handlers(void)
{
    signal(SIGINT, sig_handler);
    signal(SIGTERM, sig_handler);
    signal(SIGHUP, sig_handler);
    signal(SIGQUIT, sig_handler);
    signal(SIGABRT, sig_handler);
}

#ifdef __QNX__
/*
 * can be use for dropping root privilege
 * 0000:0000 : root
 * 1000:1000 : lwpu
 * 2000:2000 : lwsciipc
 */
void drop_privileges(void)
{
    test_params *tp = &s_params;
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
        }
    }

    /* set sub group ids */
    if (tp->num_gid > 1) {
        ret = setgroups(tp->num_gid-1, &tp->gid[1]);
        if (ret == -1) {
            printf("%s: Unable to set groups: %d\n",
                __func__, ret);
        }
    }

    /* if gid is not root */
    if (tp->gid[0] != 0) {
        ret = setregid(tp->gid[0], tp->gid[0]);
        if (ret != EOK) {
            printf("Failed to set GID to %d: %d\n", tp->gid[0], ret);
        }
    }

    /* if uid is not root */
    if (tp->uid != 0) {
        ret = setreuid(tp->uid, tp->uid);
        if (ret != EOK) {
            printf("Failed to set UID to %d: %d\n", tp->uid, ret);
        }
    }
}

/*
 * Returns:
 * EOK      : success
 * negative : failure
 */
int32_t LwMsgReceivePulse(struct endpoint *ep, struct _pulse *pulse,
    size_t bytes, struct _msg_info *info)
{
    int32_t ret;

    do {
        ret = MsgReceivePulse_r(ep->chid, pulse, bytes, info);
        if (ret == EOK) {
            if (pulse->code < 0) {
                /* system pulse */
                continue;
            }
            if (pulse->code != ep->code) {
                printf("%-40s%02d: %4s (invalid pulse: %d)\n",
                    "MsgReceivePulse", ep->id, "FAIL", pulse->code);
                ret = -EILWAL;
            }
        }
        else {
            printf("%-40s%02d: %4s (error: %d)\n",
                "MsgReceivePulse", ep->id, "FAIL", ret);
        }
        break;
    } while(true);

    return ret;
}
#endif /* __QNX__ */


/*
 * Error colwersion test
 *
 * test items : LwSciIpcErrnoToLwSciErr
 * test items : LwSciIpcLwSciErrToErrno
 */
void unit_test_err_colwersion(void)
{
    test_params *tp = &s_params;
    LwSciError err;
    int32_t oserr;
    uint32_t i;
    uint32_t errcnt = 0;

    tp->test_cnt++;

    for (i = 0U; (i < LwSciCoreErrCount); i++) {
        err = LwSciIpcErrnoToLwSciErr(LwSciCoreErrs[i].oserr);
        oserr = LwSciIpcLwSciErrToErrno(err);
        if (oserr != LwSciCoreErrs[i].oserr) {
            errcnt++;
        }
    }
    oserr = LwSciIpcLwSciErrToErrno(LwSciError_Unknown);
    if (oserr != -1) {
        errcnt++;
    }
    oserr = LwSciIpcLwSciErrToErrno(INT32_MAX);
    if (oserr != -1) {
        errcnt++;
    }
    err = LwSciIpcErrnoToLwSciErr(INT32_MAX);
    if (err != 1) {
        errcnt++;
    }
    err = LwSciIpcErrnoToLwSciErr(INT32_MIN);
    if (err != 1) {
        errcnt++;
    }

    if (errcnt > 0) {
        printf(LWIPC_GEN_FMT, "LwSciIpcErrnoToLwSciErr", 0,  "FAIL", err);
        printf(LWIPC_GEN_FMT, "LwSciIpcLwSciErrToErrno", 0,  "FAIL", err);
        tp->err_cnt++;
    }
    else {
        printf(LWIPC_GEN_FMT, "LwSciIpcErrnoToLwSciErr", 0,  "PASS", err);
        printf(LWIPC_GEN_FMT, "LwSciIpcLwSciErrToErrno", 0,  "PASS", err);
    }
}

/*
 * Init test
 *
 * test items : LwSciIpcInit
 */
void unit_test_init(void)
{
    test_params *tp = &s_params;
    LwSciError err;

    tp->test_cnt++;

    err = LwSciIpcInit();

    if (err == LwSciError_Success) {
        printf(LWIPC_GEN_FMT, "LwSciIpcInit", 0,  "PASS", err);
    }
    else {
        printf(LWIPC_FAIL_FMT, "LwSciIpcInit", 0, "FAIL", err);
        tp->err_cnt++;
    }
}

/*
 * Open endpoint  test
 *
 * test items : LwSciIpcOpenEndpoint
 */
void unit_test_open_endpoint(struct endpoint *ep, int32_t id, char *chname, LwSciError val)
{
    test_params *tp = &s_params;
    char tst_chname[LWSCIIPC_MAX_ENDPOINT_NAME];
    LwSciError err;

    tp->test_cnt++;

    /* override endpoint name with provided one */
    if (chname != NULL) {
        memcpy(tst_chname, chname, sizeof(tst_chname));
    }
    else {
        memcpy(tst_chname, ep->chname, sizeof(tst_chname));
    }

    err = LwSciIpcOpenEndpoint(tst_chname, &ep->h);
    if (err == val) {
        ep->id = id;    /* set endpoint id */
        printf("%-40s%02d: %4s (ret: %d, chName: %s, Handle: 0x%lx)\n",
            "LwSciIpcOpenEndpoint", ep->id, "PASS", err, tst_chname, ep->h);
    }
    else {
        printf(LWIPC_FAIL_FMT, "LwSciIpcOpenEndpoint", ep->id, "FAIL", err);
        tp->err_cnt++;
    }
}

#ifdef __QNX__
/*
 * Set event pulse test
 *
 * test items : LwSciIpcSetQnxPulseParam
 */
void unit_test_set_event_pulse(struct endpoint *ep, int16_t priority,
    int16_t code, void *value)
{
    test_params *tp = &s_params;
    int32_t ret;

    tp->test_cnt++;

    if (ep->chid == 0) {
        ep->chid = ChannelCreate_r(0U);
        if (ep->chid < 0) {
            printf(LWIPC_FAIL_FMT, "ChannelCreate_r", ep->id, "FAIL", ep->chid);
            tp->err_cnt++;
        }
    }
    if (ep->coid == 0) {
        ep->coid = ConnectAttach_r(0, 0, ep->chid, _NTO_SIDE_CHANNEL, 0);
        if (ep->coid < 0) {
            printf(LWIPC_FAIL_FMT, "ConnectAttach_r", ep->id, "FAIL", ep->coid);
            tp->err_cnt++;
        }
    }

    ret = LwSciIpcSetQnxPulseParam(ep->h, ep->coid, priority, code, value);
    if (ret == LwSciError_Success) {
        ep->prio = priority;
        ep->code = code;
        ep->value = value;
        printf("%-40s%02d: %4s (prio:0x%x, code:0x%x, value:0x%p)\n",
            "LwSciIpcSetQnxPulseParam", ep->id, "PASS", priority, code, value);
    }
    else {
        printf(LWIPC_FAIL_FMT, "LwSciIpcSetQnxPulseParam", ep->id, "FAIL", ret);
        tp->err_cnt++;
    }
}
#endif /* __QNX__ */

/*
 * get endpoint info test
 *
 * test items : LwSciIpcGetEndpointInfo
 */
void unit_test_get_ch_info(struct endpoint *ep)
{
    test_params *tp = &s_params;
    uint32_t errcnt = 0;
    LwSciError err;

    tp->test_cnt++;

    err = LwSciIpcGetEndpointInfo(ep->h, &ep->info);
    if (err == LwSciError_Success) {
        printf("%-40s%02d: %4s (nframes:%d, frame_size:%d)\n",
            "LwSciIpcGetEndpointInfo", ep->id, "PASS",
            ep->info.nframes, ep->info.frame_size);
    }
    else {
        printf(LWIPC_FAIL_FMT, "LwSciIpcGetEndpointInfo", ep->id, "FAIL", err);
        errcnt++;
    }

    err = LwSciIpcGetEndpointInfoInternal(ep->h, &ep->infoi);
    if ((err == LwSciError_Success) || (err == LwSciError_NotSupported)) {
        printf("%-40s%02d: %4s (irq:%d)\n",
            "LwSciIpcGetEndpointInfoInternal", ep->id, "PASS",
            ep->infoi.irq);
    }
    else {
        printf(LWIPC_FAIL_FMT, "LwSciIpcGetEndpointInfoInternal", ep->id, "FAIL", err);
        errcnt++;
    }

    if (errcnt > 0) {
        tp->err_cnt++;
    }
}

void setup_buffer(test_params *tp)
{
    if (tp->e1.info.frame_size) {
        tp->e1.buf = (uint8_t *)calloc(1, tp->e1.info.frame_size);
    }
    else
        printf("%s: EP1 frame_size is not defined\n", __func__);

    if (tp->e2.info.frame_size) {
        tp->e2.buf = (uint8_t *)calloc(1, tp->e2.info.frame_size);
    }
    else
        printf("%s: EP2 frame_size is not defined\n", __func__);
}

/*
 * val
 * -1    : random
 * other : fill (val & 0xFF) to buffer
 */
void fill_buffer(struct endpoint *ep, int32_t val)
{
    uint32_t i;

    /* prepare buffer data */
    if (ep->info.frame_size) {
        /* fill test pattern */
        for(i = 0;i<ep->info.frame_size;i++) {
            if (val == RANDOM)    /* random */
                ep->buf[i] = rand() & 0xFF;
            else
                ep->buf[i] = i & 0xFF;  /* sequential data */
        }
#if DEBUG2
        print_buffer("TXBUF snip", ep->buf, 10);
#endif
    }
}

void print_buffer(const char *head, uint8_t *buf, uint32_t length)
{
    uint32_t i;

    if (buf == NULL) {
        return;
    }

    printf("%s: ", head);
    for (i = 0;i<length;i++) {
        printf("%02x ", buf[i]);
    }
    printf("\n");
}

void release_resources(test_params *tp)
{
    free(tp->e1.buf);
    free(tp->e2.buf);
    tp->e1.buf = NULL;
    tp->e2.buf = NULL;

    if (tp->e1.coid != 0) {
        (void)ConnectDetach_r(tp->e1.coid);
    }
    if (tp->e2.coid != 0) {
        (void)ConnectDetach_r(tp->e2.coid);
    }
    if (tp->e1.chid != 0) {
        (void)ChannelDestroy_r(tp->e1.chid);
    }
    if (tp->e2.chid != 0) {
        (void)ChannelDestroy_r(tp->e2.chid);
    }
}

/*
 * data integrity test
 *
 * val parameter
 *    -1 : compare txbuf w/ rxbuf
 * other : check if rxbuf has val
 *
 * test items (1)
 */
void unit_test_compare_data(struct endpoint *tx, struct endpoint *rx, int32_t val)
{
    test_params *tp = &s_params;
    uint32_t i;
    int32_t err_cnt = 0;

    tp->test_cnt++;

#if DEBUG
    print_buffer("RXBUF snip", rx->buf, 10);
#endif

    if (val == COMPARE_ENDS) {
        if (rx->buf != NULL && tx->buf != NULL) {
            for(i = 0;i<tx->info.frame_size;i++) {
                if (rx->buf[i] != tx->buf[i]) {
#if DEBUG2
                    printf("%-40s%02d: %4s txbuf[%d] 0x%02x != rxbuf[%d] 0x%02x\n",
                        "Data mismatch error", 0, "FAIL",
                        i, tx->buf[i], i, rx->buf[i]);
#endif
                    err_cnt++;
                }
            }
        }
    }
    else {
        for(i = 0;i<rx->info.frame_size;i++) {
            if (rx->buf[i] != val) {
#if DEBUG2
                printf("%-40s%02d: %4s rxbuf[%d] 0x%02x != 0x%02x\n",
                    "Data mismatch error", 0, "FAIL",
                    i, rx->buf[i], val);
#endif
                err_cnt++;
            }
        }
    }
    if (err_cnt) {
        printf("%-40s%02d: %4s - found %d bytes mismatches\n",
            "Comparing data", 0, "FAIL", err_cnt);
        tp->err_cnt++;
    }
    else
        printf(LWIPC_GEN_FMT, "Comparing data", 0, "PASS", 0);
}

/*
 * reset test
 *
 * test items : LwSciIpcResetEndpoint
 * test items : LwSciIpcCanRead
 * test items : LwSciIpcCanWrite
 */
void unit_test_reset(struct endpoint *ep)
{
    test_params *tp = &s_params;
    uint32_t event;
    LwSciError err;
    bool flag;

    tp->test_cnt++;

    LwSciIpcResetEndpoint(ep->h);
    printf(LWIPC_GEN_FMT, "LwSciIpcResetEndpoint", ep->id, "PASS", 0);

    err = LwSciIpcGetEvent(ep->h, &event);
    /* check if it's in reset status */
    if ((err == LwSciError_Success) && (event & LW_SCI_IPC_EVENT_CONN_RESET) ) {
        printf("%-40s%02d: %4s (event: 0x%x)\n",
            "LwSciIpcGetEvent", ep->id, "PASS", event);
    }
    else {
        printf("%-40s%02d: %4s (event: 0x%x, ret: %d)\n",
            "LwSciIpcGetEvent", ep->id, "FAIL", event, err);
        tp->err_cnt++;
    }

    tp->test_cnt++;
    flag = LwSciIpcCanWrite(ep->h);
    if (flag == false) {
        printf("%-40s%02d: %4s (flag: %d)\n",
            "LwSciIpcCanWrite", ep->id, "PASS", flag);
    } else {
        printf("%-40s%02d: %4s (flag: %d)\n",
            "LwSciIpcCanWrite", ep->id, "FAIL", flag);
        tp->err_cnt++;
    }

    tp->test_cnt++;
    flag = LwSciIpcCanRead(ep->h);
    if (flag == false) {
        printf("%-40s%02d: %4s (flag: %d)\n",
            "LwSciIpcCanRead", ep->id, "PASS", flag);
    } else {
        printf("%-40s%02d: %4s (flag: %d)\n",
            "LwSciIpcCanRead", ep->id, "FAIL", flag);
        tp->err_cnt++;
    }

    /* test send/receive and check LwSciError_ConnectionReset error */
    unit_test_send_with_return(ep, LwSciError_ConnectionReset);
    unit_test_receive_with_return(ep, LwSciError_ConnectionReset);
    unit_test_send_with_return_zc(ep, LwSciError_ConnectionReset);
    unit_test_receive_with_return_zc(ep, LwSciError_ConnectionReset);
    unit_test_send_with_return_poke(ep, LwSciError_ConnectionReset);
    unit_test_receive_with_return_peek(ep, LwSciError_ConnectionReset);
}

/*
 * notification test
 *
 * in case of intra-thread, we should call GetEvent() for both ends
 * repeatedly to get establish notification.
 *
 * test items : LwSciIpcGetEvent
 */
void unit_test_poll_notification(test_params *tp)
{
    LwSciError err;
    int32_t errcnt = 0;
    bool tx_est = 0;
    bool rx_est = 0;
    uint32_t txevt = 0;
    uint32_t rxevt = 0;

    tp->test_cnt++;

    while (!s_Stop && (!tx_est || !rx_est)) {
        err = LwSciIpcGetEvent(tp->e1.h, &txevt);
        if (err != LwSciError_Success) {
            printf("%-40s%02d: %4s (EST:%d,event:0x%x)\n",
                "LwSciIpcGetEvent", tp->e1.id, "FAIL", tx_est, txevt);
            errcnt++;
            break;
        }
        if (txevt & LW_SCI_IPC_EVENT_CONN_RESET) {
            tx_est = 0;
        }
        else if (txevt & LW_SCI_IPC_EVENT_CONN_EST_ALL) {
            tx_est = 1;
        }

        err = LwSciIpcGetEvent(tp->e2.h, &rxevt);
        if (err != LwSciError_Success) {
            printf("%-40s%02d: %4s (EST:%d,event:0x%x)\n",
                "LwSciIpcGetEvent", tp->e2.id, "FAIL", rx_est, rxevt);
            errcnt++;
            break;
        }
        if (rxevt & LW_SCI_IPC_EVENT_CONN_RESET) {
            rx_est = 0;
        }
        else if (rxevt & LW_SCI_IPC_EVENT_CONN_EST_ALL) {
            rx_est = 1;
        }
    }

    if (errcnt == 0) {
        printf("%-40s%02d: %4s (EST:%d,event:0x%x)\n",
            "LwSciIpcGetEvent", tp->e1.id, "PASS", tx_est, txevt);
        printf("%-40s%02d: %4s (EST:%d,event:0x%x)\n",
            "LwSciIpcGetEvent", tp->e2.id, "PASS", rx_est, rxevt);
    }
    else
        tp->err_cnt++;
}

/*
 * wait event with MsgReceivePulse
 *
 * test items (1):
 *        LwSciIpcGetEvent (1)
 *        MsgReceivePulse (1)
 *        LwSciIpcCanRead (1)
 *        LwSciIpcCanWrite (1)
 */
void unit_test_wait_event(struct endpoint *ep, int32_t value)
{
    test_params *tp = &s_params;
    struct _pulse pulse;
    LwSciError err;
    int32_t ret;
    int32_t errcnt = 0;
    uint32_t event = 0;
    uint64_t to;
    bool flag;

    tp->test_cnt++;

    while(!s_Stop) {
        err = LwSciIpcGetEvent(ep->h, &event);
        if (err != LwSciError_Success) {
            printf("%-40s%02d: %4s (event: 0x%x)\n",
                "LwSciIpcGetEvent", ep->id, "FAIL", event);
            errcnt++;
            break;
        }
        if (event & value) {
            break;
        }
#if DEBUG
        printf("%s: event: 0x%x\n", __func__, event);
#endif
#ifdef __QNX__
        /* set pulse timeout for test purpose */
        to = tp->timeout * 1000000U; /* msec to nsec */
        if (TimerTimeout(CLOCK_MONOTONIC, _NTO_TIMEOUT_RECEIVE, NULL,
                        &to, NULL) == -1) {
            printf("%-40s%02d: %4s (error: %d)\n",
                "TimerTimeout", ep->id, "FAIL", -errno);
            errcnt++;
            break;
        }
        ret = LwMsgReceivePulse(ep, &pulse, sizeof(pulse), NULL);
        if (ret < 0) {
            errcnt++;
            break;
        }
#if DEBUG
        printf("%s: code:%x, val:%x\n",
            __func__, pulse.code, pulse.value.sival_int);
#endif
#endif /* __QNX__ */
    }

    switch (value) {
        case LW_SCI_IPC_EVENT_WRITE:
            flag = LwSciIpcCanWrite(ep->h);
            if (flag != true) {
                errcnt++;
            }
            break;
        case LW_SCI_IPC_EVENT_READ:
            flag = LwSciIpcCanRead(ep->h);
            if (flag != true) {
                errcnt++;
            }
            break;
    }

    if (errcnt == 0) {
        printf("%-40s%02d: %4s (event: 0x%x)\n",
            "LwSciIpcGetEvent", ep->id, "PASS", event);

        if (value == LW_SCI_IPC_EVENT_READ) {
            printf("%-40s%02d: %4s (flag: %d)\n",
                "LwSciIpcCanRead", ep->id, "PASS", flag);
        }
        if (value == LW_SCI_IPC_EVENT_WRITE) {
            printf("%-40s%02d: %4s (flag: %d)\n",
                "LwSciIpcCanWrite", ep->id, "PASS", flag);
        }
    }
    else
        tp->err_cnt++;
}

/*
 * poll event
 *
 * test items (1):
 *        LwSciIpcGetEvent (1)
 */
void unit_test_poll_event(struct endpoint *ep, int32_t value)
{
    test_params *tp = &s_params;
    uint32_t event;
    uint32_t wait_time;
    LwSciError err;

    tp->test_cnt++;

    wait_time = 0;
    do {
        err = LwSciIpcGetEvent(ep->h, &event);
        if (err != LwSciError_Success)
            break;
        usleep(1000); /* testing purpose for timeout */
        wait_time++;
    } while (!(event & value) && wait_time < tp->timeout);

    if ((wait_time == tp->timeout) || (err != LwSciError_Success)) {
        printf("%-40s%02d: %4s TIMEOUT (event: 0x%x)\n",
            "LwSciIpcGetEvent", ep->id, "FAIL", event);
        tp->err_cnt++;
    }
    else {
        printf("%-40s%02d: %4s (event: 0x%x)\n",
            "LwSciIpcGetEvent", ep->id, "PASS", event);
    }

}

void unit_test_notification(test_params *tp)
{
    struct endpoint *e1 = &tp->e1;
    struct endpoint *e2 = &tp->e2;
    struct _pulse pulse1;
    struct _pulse pulse2;
    bool est1 = 0;
    bool est2 = 0;
    uint32_t evt1 = 0;
    uint32_t evt2 = 0;
    int32_t errcnt = 0;
    int32_t ret;
    LwSciError err;
    uint64_t to;

    while(!s_Stop && (!est1 || !est2)) {
#ifdef __QNX__
        /* set pulse timeout for test purpose */
        to = tp->timeout * 1000000U; /* msec to nsec */
        if (TimerTimeout(CLOCK_MONOTONIC, _NTO_TIMEOUT_RECEIVE, NULL,
                        &to, NULL) == -1) {
            printf("%-40s%02d: %4s (error: %d)\n",
                "TimerTimeout", e1->id, "FAIL", -errno);
            errcnt++;
            break;
        }
        ret = LwMsgReceivePulse(e1, &pulse1, sizeof(pulse1), NULL);
        if (ret < 0) {
            errcnt++;
            break;
        }
#if DEBUG
        printf("%-40s%02d: %4s (code:%x val:%x)\n",
            "MsgReceivePulse", e1->id, "--",
            pulse1.code, pulse1.value.sival_int);
#endif
#endif /* __QNX__ */
        err = LwSciIpcGetEvent(e1->h, &evt1);
        if (err != LwSciError_Success) {
            printf("%-40s%02d: %4s (EST:%d,event:0x%x)\n",
                "LwSciIpcGetEvent", e1->id, "FAIL", est1, evt1);
            errcnt++;
            break;
        }
        if (evt1 & LW_SCI_IPC_EVENT_CONN_RESET) {
            est1 = 0;
        }
        else if (evt1 & LW_SCI_IPC_EVENT_CONN_EST) {
            est1 = 1;
            goto ep2;
        }

#if DEBUG
        printf("%-40s%02d: %4s (EST:%d,event:0x%x)\n",
            "LwSciIpcGetEvent", e1->id, "---", est1, evt1);
#endif

ep2:
#ifdef __QNX__
        /* set pulse timeout for test purpose */
        to = tp->timeout * 1000000U;
        if (TimerTimeout(CLOCK_MONOTONIC, _NTO_TIMEOUT_RECEIVE, NULL,
                        &to, NULL) == -1) {
            printf("%-40s%02d: %4s (error: %d)\n",
                "TimerTimeout", e2->id, "FAIL", -errno);
            errcnt++;
            break;
        }
        ret = LwMsgReceivePulse(e2, &pulse2, sizeof(pulse2), NULL);
        if (ret < 0) {
            errcnt++;
            break;
        }
#if DEBUG
        printf("%-40s%02d: %4s (code:%x val:%x)\n",
            "MsgReceivePulse", e2->id, "--",
            pulse2.code, pulse2.value.sival_int);
#endif
#endif /* __QNX__ */
        err = LwSciIpcGetEvent(e2->h, &evt2);
        if (err != LwSciError_Success) {
            printf("%-40s%02d: %4s (EST:%d,event:0x%x)\n",
                "LwSciIpcGetEvent", e2->id, "FAIL", est2, evt2);
            errcnt++;
            break;
        }
        if (evt2 & LW_SCI_IPC_EVENT_CONN_RESET) {
            est2 = 0;
        }
        else if (evt2 & LW_SCI_IPC_EVENT_CONN_EST) {
            est2 = 1;
            continue;
        }
#if DEBUG
        printf("%-40s%02d: %4s (EST:%d,event:0x%x)\n",
            "LwSciIpcGetEvent", e2->id, "---", est1, evt2);
#endif
    }

    if (errcnt == 0) {
        printf("%-40s%02d: %4s (EST:%d,event:0x%x)\n",
            "LwSciIpcGetEvent", tp->e1.id, "PASS", est1, evt1);
        printf("%-40s%02d: %4s (EST:%d,event:0x%x)\n",
            "LwSciIpcGetEvent", tp->e2.id, "PASS", est2, evt2);
    }
    else
        tp->err_cnt++;
}

/*
 * send only test
 *
 * test items (1):
 *        LwSciIpcWrite (1)
 */
void unit_test_send_only(struct endpoint *ep, LwSciError val)
{
    test_params *tp = &s_params;
    uint32_t frame_size;
    LwSciError err;
    int32_t bytes;

    tp->test_cnt++;

    if (tp->frame_size != 0)
        frame_size = tp->frame_size;
    else
        frame_size = ep->info.frame_size;

    err = LwSciIpcWrite(ep->h, ep->buf, frame_size, &bytes);
    if ((val == LwSciError_Success && err == val && (size_t)bytes == frame_size) ||
        (val != LwSciError_Success && err == val && (size_t)bytes == 0U)) {
        printf("%-40s%02d: %4s (rx: %d bytes)\n",
            "LwSciIpcWrite", ep->id, "PASS", bytes);
    }
    else {
        printf(LWIPC_FAIL_FMTD, "LwSciIpcWrite", ep->id, "FAIL", err, bytes);
        tp->err_cnt++;
    }
}

/*
 * send test with expected return
 * s_params.frame_size should be set before testing
 *
 * test items (2):
 *        LwSciIpcWrite (1)
 */
void unit_test_send_with_return(struct endpoint *ep, int32_t val)
{
    unit_test_send_only(ep, val);
}

/*
 * send only test (zero copy)
 *
 * test items (2):
 *        LwSciIpcWriteGetNextFrame (1)
 *        LwSciIpcWriteAdvance (1)
 */
void unit_test_send_only_zc(struct endpoint *ep, LwSciError val)
{
    test_params *tp = &s_params;
    LwSciError err;
    volatile uint8_t *buf;

    tp->test_cnt++;

    err = LwSciIpcWriteGetNextFrame(ep->h, (void *)&buf);
    if (err != val) {
        printf(LWIPC_FAIL_FMT, "LwSciIpcWriteGetNextFrame", ep->id, "FAIL", err);
        tp->err_cnt++;
    }
    else  {
        printf("%-40s%02d: %4s (ivcbuf: %p, ret: %d)\n",
            "LwSciIpcWriteGetNextFrame", ep->id, "PASS", buf, err);

        if (err == LwSciError_Success) {
            /* copy for test */
            memcpy((void *)buf, ep->buf, ep->info.frame_size);

            tp->test_cnt++;

            err = LwSciIpcWriteAdvance(ep->h);
            if (err == LwSciError_Success) {
                printf(LWIPC_GEN_FMT, "LwSciIpcWriteAdvance", ep->id, "PASS", err);
            }
            else {
                printf(LWIPC_FAIL_FMT, "LwSciIpcWriteAdvance", ep->id, "FAIL", err);
                tp->err_cnt++;
            }
        }
    }
}

/*
 * send only test (poke)
 *
 * test items (2):
 *        LwSciIpcWritePoke (1)
 *        LwSciIpcWriteAdvance (1)
 */
void unit_test_send_only_poke(struct endpoint *ep, LwSciError val)
{
    test_params *tp = &s_params;
    uint32_t frame_size;
    LwSciError err;
    int32_t bytes;

    tp->test_cnt++;

    if (tp->frame_size != 0)
        frame_size = tp->frame_size;
    else
        frame_size = ep->info.frame_size;

    err = LwSciIpcWritePoke(ep->h, ep->buf, 0, frame_size, &bytes);
    if ((val == LwSciError_Success && err == val && (size_t)bytes == frame_size) ||
        (val != LwSciError_Success && err == val && (size_t)bytes == 0U)) {
        printf("%-40s%02d: %4s (tx: %d bytes)\n",
            "LwSciIpcWritePoke", ep->id, "PASS", bytes);

        if (err == LwSciError_Success) {
            tp->test_cnt++;

            err = LwSciIpcWriteAdvance(ep->h);
            if (err == LwSciError_Success) {
                printf(LWIPC_GEN_FMT, "LwSciIpcWriteAdvance", ep->id, "PASS", err);
            }
            else {
                printf(LWIPC_FAIL_FMT, "LwSciIpcWriteAdvance", ep->id, "FAIL", err);
                tp->err_cnt++;
            }
        }
    }
    else {
        printf(LWIPC_FAIL_FMTD, "LwSciIpcWritePoke", ep->id, "FAIL", err, bytes);
        tp->err_cnt++;
    }
}

/*
 * send test with expected return (poke)
 * s_params.frame_size should be set before testing
 *
 * test items (2):
 *        LwSciIpcWritePoke (1)
 */
void unit_test_send_with_return_zc(struct endpoint *ep, int32_t val)
{
    unit_test_send_only_zc(ep, val);
}

/*
 * send test with expected return (poke)
 * s_params.frame_size should be set before testing
 *
 * test items (2):
 *        LwSciIpcWritePoke (1)
 */
void unit_test_send_with_return_poke(struct endpoint *ep, int32_t val)
{
    unit_test_send_only_poke(ep, val);
}

/*
 * wait and send test
 *
 * test items (2):
 *        LwSciIpcGetEvent (1)
 *        LwSciIpcWrite (1)
 */
void unit_test_wait_n_send(struct endpoint *ep)
{
    unit_test_wait_event(ep, LW_SCI_IPC_EVENT_WRITE);
    unit_test_send_only(ep, LwSciError_Success);
}

/*
 * wait and send test (zero copy)
 *
 * test items (3):
 *        LwSciIpcGetEvent (1)
 *        LwSciIpcWriteGetNextFrame (1)
 *        LwSciIpcWriteAdvance (1)
 */
void unit_test_wait_n_send_zc(struct endpoint *ep)
{
    unit_test_wait_event(ep, LW_SCI_IPC_EVENT_WRITE);
    unit_test_send_only_zc(ep, LwSciError_Success);
}

/*
 * wait and send test (poke)
 *
 * test items (3):
 *        LwSciIpcGetEvent (1)
 *        LwSciIpcWritePoke (1)
 *        LwSciIpcWriteAdvance (1)
 */
void unit_test_wait_n_send_poke(struct endpoint *ep)
{
    unit_test_wait_event(ep, LW_SCI_IPC_EVENT_WRITE);
    unit_test_send_only_poke(ep, LwSciError_Success);
}

/*
 * receive only test
 *
 * test items (1):
 *        LwSciIpcRead (1)
 */
void unit_test_receive_only(struct endpoint *ep, LwSciError val)
{
    test_params *tp = &s_params;
    uint32_t frame_size;
    int32_t bytes;
    LwSciError err;

    tp->test_cnt++;

    if (tp->frame_size != 0)
        frame_size = tp->frame_size;
    else
        frame_size = ep->info.frame_size;

    err = LwSciIpcRead(ep->h, ep->buf, frame_size, &bytes);
    if ((val == LwSciError_Success && err == val && (size_t)bytes == frame_size) ||
        (val != LwSciError_Success && err == val && (size_t)bytes == 0U)) {
        printf("%-40s%02d: %4s (rx: %d bytes)\n",
            "LwSciIpcRead", ep->id, "PASS", bytes);
    }
    else {
        printf(LWIPC_FAIL_FMTD, "LwSciIpcRead", ep->id, "FAIL", err, bytes);
        tp->err_cnt++;
    }
}

/*
 * receive test with expected return
 * s_params.frame_size should be set before testing
 *
 * test items (2):
 *        LwSciIpcRead (1)
 */
void unit_test_receive_with_return(struct endpoint *ep, int32_t val)
{
    unit_test_receive_only(ep, val);
}

/*
 * receive only test (zero copy)
 *
 * test items (2):
 *        LwSciIpcReadGetNextFrame (1)
 *        LwSciIpcReadAdvance (1)
 */
void unit_test_receive_only_zc(struct endpoint *ep, LwSciError val)
{
    test_params *tp = &s_params;
    LwSciError err;
    const volatile uint8_t *buf;

    tp->test_cnt++;

    err = LwSciIpcReadGetNextFrame(ep->h, (void *)&buf);
    if (err != val) {
        printf(LWIPC_FAIL_FMT, "LwSciIpcReadGetNextFrame", ep->id, "FAIL", err);
        tp->err_cnt++;
    }
    else {
        printf("%-40s%02d: %4s (ivcbuf: %p, ret: %d)\n",
            "LwSciIpcReadGetNextFrame", ep->id, "PASS", buf, err);

        if (err == LwSciError_Success) {
            /* copy for test */
            memcpy(ep->buf, (void *)buf, ep->info.frame_size);

            tp->test_cnt++;

            err = LwSciIpcReadAdvance(ep->h);
            if (err == LwSciError_Success) {
                printf(LWIPC_GEN_FMT, "LwSciIpcReadAdvance", ep->id, "PASS", err);
            }
            else {
                printf(LWIPC_FAIL_FMT, "LwSciIpcReadAdvance", ep->id, "FAIL", err);
                tp->err_cnt++;
            }
        }
    }
}


/*
 * receive only test (peek)
 *
 * test items (2):
 *        LwSciIpcReadPeek (1)
 *        LwSciIpcReadAdvance (1)
 */
void unit_test_receive_only_peek(struct endpoint *ep, LwSciError val)
{
    test_params *tp = &s_params;
    uint32_t frame_size;
    int32_t bytes;
    LwSciError err;

    tp->test_cnt++;

    if (tp->frame_size != 0)
        frame_size = tp->frame_size;
    else
        frame_size = ep->info.frame_size;

    err = LwSciIpcReadPeek(ep->h, ep->buf, 0, frame_size, &bytes);
    if ((val == LwSciError_Success && err == val && (size_t)bytes == frame_size) ||
        (val != LwSciError_Success && err == val && (size_t)bytes == 0U)) {
        printf("%-40s%02d: %4s (rx: %d bytes)\n",
            "LwSciIpcReadPeek", ep->id, "PASS", bytes);

        if (err == LwSciError_Success) {
            tp->test_cnt++;

            err = LwSciIpcReadAdvance(ep->h);
            if (err == LwSciError_Success) {
                printf(LWIPC_GEN_FMT, "LwSciIpcReadAdvance", ep->id, "PASS", err);
            }
            else {
                printf(LWIPC_FAIL_FMT, "LwSciIpcReadAdvance", ep->id, "FAIL", err);
                tp->err_cnt++;
            }
        }
    }
    else {
        printf(LWIPC_FAIL_FMTD, "LwSciIpcReadPeek", ep->id, "FAIL", err, bytes);
        tp->err_cnt++;
    }
}

/*
 * receive test with exprected return (zero copy)
 * s_params.frame_size should be set before testing
 *
 * test items (2):
 *        LwSciIpcRead (1)
 */
void unit_test_receive_with_return_zc(struct endpoint *ep, int32_t val)
{
    unit_test_receive_only_zc(ep, val);
}

/*
 * receive test with exprected return (peek)
 * s_params.frame_size should be set before testing
 *
 * test items (2):
 *        LwSciIpcReadPeek (1)
 */
void unit_test_receive_with_return_peek(struct endpoint *ep, int32_t val)
{
    unit_test_receive_only_peek(ep, val);
}

/*
 * receive test
 *
 * test items (2):
 *        LwSciIpcGetEvent (1)
 *        LwSciIpcRead (1)
 */
void unit_test_wait_n_receive(struct endpoint *ep)
{
    unit_test_wait_event(ep, LW_SCI_IPC_EVENT_READ);
    unit_test_receive_only(ep, LwSciError_Success);
}

/*
 * wait and receive test (zero copy)
 *
 * test items (3):
 *        LwSciIpcGetEvent (1)
 *        LwSciIpcReadGetNextFrame (1)
 *        LwSciIpcReadAdvance (1)
 */
void unit_test_wait_n_receive_zc(struct endpoint *ep)
{
    unit_test_wait_event(ep, LW_SCI_IPC_EVENT_READ);
    unit_test_receive_only_zc(ep, LwSciError_Success);
}

/*
 * wait and receive test (peek)
 *
 * test items (3):
 *        LwSciIpcGetEvent (1)
 *        LwSciIpcReadPeek (1)
 *        LwSciIpcReadAdvance (1)
 */
void unit_test_wait_n_receive_peek(struct endpoint *ep)
{
    unit_test_wait_event(ep, LW_SCI_IPC_EVENT_READ);
    unit_test_receive_only_peek(ep, LwSciError_Success);
}

/*
 * poll send test
 *
 * test items (2):
 *        LwSciIpcGetEvent (1)
 *        LwSciIpcWrite (1)
 */
void unit_test_poll_send(struct endpoint *ep)
{
    unit_test_poll_event(ep, LW_SCI_IPC_EVENT_WRITE);
    unit_test_send_only(ep, LwSciError_Success);
}


/*
 * poll send test (zero copy)
 *
 * test items (3):
 *        LwSciIpcGetEvent (1)
 *        LwSciIpcWriteGetNextFrame (1)
 *        LwSciIpcWriteAdvance (1)
 */
void unit_test_poll_send_zc(struct endpoint *ep)
{
    unit_test_poll_event(ep, LW_SCI_IPC_EVENT_WRITE);
    unit_test_send_only_zc(ep, LwSciError_Success);
}

/*
 * poll send test (poke)
 *
 * test items (3):
 *        LwSciIpcGetEvent (1)
 *        LwSciIpcWritePoke (1)
 *        LwSciIpcWriteAdvance (1)
 */
void unit_test_poll_send_poke(struct endpoint *ep)
{
    unit_test_poll_event(ep, LW_SCI_IPC_EVENT_WRITE);
    unit_test_send_only_poke(ep, LwSciError_Success);
}

/*
 * poll receive test
 *
 * test items (2):
 *        LwSciIpcGetEvent (1)
 *        LwSciIpcRead (1)
 */
void unit_test_poll_receive(struct endpoint *ep)
{
    unit_test_poll_event(ep, LW_SCI_IPC_EVENT_READ);
    unit_test_receive_only(ep, LwSciError_Success);
}

/*
 * poll receive test (zero copy)
 *
 * test items (3):
 *        LwSciIpcGetEvent (1)
 *        LwSciIpcReadGetNextFrame (1)
 *        LwSciIpcReadAdvance (1)
 */
void unit_test_poll_receive_zc(struct endpoint *ep)
{
    unit_test_poll_event(ep, LW_SCI_IPC_EVENT_READ);
    unit_test_receive_only_zc(ep, LwSciError_Success);
}

/*
 * poll receive test (peek)
 *
 * test items (3):
 *        LwSciIpcGetEvent (1)
 *        LwSciIpcReadPeek (1)
 *        LwSciIpcReadAdvance (1)
 */
void unit_test_poll_receive_peek(struct endpoint *ep)
{
    unit_test_poll_event(ep, LW_SCI_IPC_EVENT_READ);
    unit_test_receive_only_peek(ep, LwSciError_Success);
}

void unit_test_fill_channel(struct endpoint *ep)
{
    uint32_t i;
    int32_t bytes;

    for (i = 0;i<ep->info.nframes;i++) {
        (void)LwSciIpcWrite(ep->h, ep->buf, ep->info.frame_size, &bytes);
    }
}

void unit_test_drain_channel(struct endpoint *ep)
{
    uint32_t i;
    int32_t bytes;

    for (i = 0;i<ep->info.nframes;i++) {
        (void)LwSciIpcRead(ep->h, ep->buf, ep->info.frame_size, &bytes);
    }
}

void unit_test_write_after_fill(struct endpoint *ep)
{
    test_params *tp = &s_params;
    LwSciError err;
    int32_t bytes;

    tp->test_cnt++;

    unit_test_fill_channel(ep);

    err = LwSciIpcWrite(ep->h, ep->buf, ep->info.frame_size, &bytes);
    if (err == LwSciError_InsufficientMemory && bytes == 0U) {
        printf("%-40s%02d: %4s (error: %d)\n",
            "LwSciIpcWrite", ep->id, "PASS", err);
    }
    else {
        tp->err_cnt++;
        printf("%-40s%02d: %4s (error: %d)\n",
            "LwSciIpcWrite", ep->id, "FAIL", err);
    }
}

void unit_test_read_after_drain(struct endpoint *ep)
{
    test_params *tp = &s_params;
    int32_t bytes;
    LwSciError err;

    tp->test_cnt++;

    unit_test_drain_channel(ep);

    err = LwSciIpcRead(ep->h, ep->buf, ep->info.frame_size, &bytes);
    if (err == LwSciError_InsufficientMemory && bytes == 0U) {
        printf("%-40s%02d: %4s (error: %d)\n",
            "LwSciIpcRead", ep->id, "PASS", err);
    }
    else {
        tp->err_cnt++;
        printf("%-40s%02d: %4s (error: %d)\n",
            "LwSciIpcRead", ep->id, "FAIL", err);
    }
}

/*
 * send/receive polling test
 *
 * test items (4):
 *        LwSciIpcGetEvent (2)
 *        LwSciIpcWrite (1)
 *        LwSciIpcRead (1)
 */
void unit_test_poll_send_receive(struct endpoint *tx, struct endpoint *rx)
{
    unit_test_poll_send(tx);
    unit_test_poll_receive(rx);
}

/*
 * send/receive polling test
 *
 * test items (4):
 *        LwSciIpcGetEvent (2)
 *        LwSciIpcWrite (1)
 *        LwSciIpcRead (1)
 */
void unit_test_poll_send_receive_zc(struct endpoint *tx, struct endpoint *rx)
{
    unit_test_poll_send_zc(tx);
    unit_test_poll_receive_zc(rx);
}

/*
 * send/receive polling test (poke/peek)
 *
 * test items (4):
 *        LwSciIpcGetEvent (2)
 *        LwSciIpcWritePoke (1)
 *        LwSciIpcReadPeek (1)
 */
void unit_test_poll_send_receive_pokepeek(struct endpoint *tx, struct endpoint *rx)
{
    unit_test_poll_send_poke(tx);
    unit_test_poll_receive_peek(rx);
}

/*
 * send/receive test with event handle
 *
 * test items (4):
 *        LwSciIpcGetEvent (2)
 *        LwSciIpcWrite (1)
 *        LwSciIpcRead (1)
 */
void unit_test_send_receive(struct endpoint *tx, struct endpoint *rx)
{
    unit_test_wait_n_send(tx);
    unit_test_wait_n_receive(rx);
}

/*
 * send/receive test with event handle
 *
 * test items (4):
 *        LwSciIpcGetEvent (2)
 *        LwSciIpcWrite (1)
 *        LwSciIpcRead (1)
 */
void unit_test_send_receive_zc(struct endpoint *tx, struct endpoint *rx)
{
    unit_test_wait_n_send_zc(tx);
    unit_test_wait_n_receive_zc(rx);
}

/*
 * send/receive test with event handle (poke/peek)
 *
 * test items (4):
 *        LwSciIpcGetEvent (2)
 *        LwSciIpcWritePoke (1)
 *        LwSciIpcReadPeek (1)
 */
void unit_test_send_receive_pokepeek(struct endpoint *tx, struct endpoint *rx)
{
    unit_test_wait_n_send_poke(tx);
    unit_test_wait_n_receive_peek(rx);
}

/*
 * close endpoint test
 *
 * test items : LwSciIpcCloseEndpoint
 */
void unit_test_close_endpoint(struct  endpoint *ep)
{
    test_params *tp = &s_params;

    tp->test_cnt++;
    LwSciIpcCloseEndpoint(ep->h);
    printf(LWIPC_GEN_FMT, "LwSciIpcCloseEndpoint", ep->id, "PASS", 0);
}

/*
 * deinit test
 *
 * test items : LwSciIpcDeinit
 */
void unit_test_deinit(test_params *tp)
{
    tp->test_cnt++;
    LwSciIpcDeinit();
    printf(LWIPC_GEN_FMT, "LwSciIpcDeinit", 0, "PASS", 0);
}

void print_ids(const char *str)
{
    printf("%s: PID: %d, TID: %d, UID: %d, GID: %d\n",
        str, getpid(), gettid(), getuid(), getgid());
}

/*
 * deinit test
 *
 * test items : LwSciIpcDeinit
 */
void unit_test_error_handle(struct endpoint *ep)
{
    uint64_t pid = (uint64_t)getpid();

    /* TODO: call APIs w/ 0 handle (LwSciError_BadParameter) and
     * abnormal handle (tsthandle)
     */
    unit_test_call_apis2(ep, 0, false);
    unit_test_call_apis2(ep, (LwSciIpcEndpoint)(pid << 32 | 10),
        false);
    unit_test_call_apis2(ep, (LwSciIpcEndpoint)ep->tsthandle,
        false);

    unit_test_call_apis(ep, 0, LwSciError_BadParameter);
    unit_test_call_apis(ep, (LwSciIpcEndpoint)(pid << 32 | 10),
        LwSciError_BadParameter);
    unit_test_call_apis(ep, (LwSciIpcEndpoint)ep->tsthandle,
        LwSciError_BadParameter);
}

/*
 * API call test w/ NULL parameter
 *
 */
void unit_test_NULL_pointer(struct endpoint *ep, void *ptr, LwSciError val)
{
    test_params *tp = &s_params;
    struct LwSciIpcEndpointAccessInfo accessInfo;
    LwSciIpcTopoId peerTopoId;
    LwSciIpcEndpointVuid peerVuid;
    LwSciIpcEndpointVuid vuid;
    resmgr_context_t ctp;
    int32_t errcnt = 0;
    int32_t bytes;
    LwSciError err;

    tp->test_cnt++;

    printf("%-40s%02d: %4s\n", "NULL POINTER", ep->id, "");

    /* LwSciIpcOpenEndpoint */
    err = LwSciIpcOpenEndpoint(ptr, 0);
    if (err == val) {
        printf(LWIPC_GEN_FMT, "LwSciIpcOpenEndpoint", ep->id, "PASS", err);
    }
    else {
        errcnt++;
        printf(LWIPC_FAIL_FMT, "LwSciIpcOpenEndpoint", ep->id, "FAIL", err);
    }

    err = LwSciIpcOpenEndpoint(ep->chname, ptr);
    if (err == val) {
        printf(LWIPC_GEN_FMT, "LwSciIpcOpenEndpoint", ep->id, "PASS", err);
    }
    else {
        errcnt++;
        printf(LWIPC_FAIL_FMT, "LwSciIpcOpenEndpoint", ep->id, "FAIL", err);
    }

        /* successful open for next API tests using handle */
        err = LwSciIpcOpenEndpoint(ep->chname, &ep->h);
        if (err == LwSciError_Success) {
            printf("%-40s%02d: %4s (ret: %d, chName: %s, Handle: 0x%lx)\n",
                "LwSciIpcOpenEndpoint", ep->id, "PASS", err, ep->chname, ep->h);
        }
        else {
            printf(LWIPC_FAIL_FMT, "LwSciIpcOpenEndpoint", ep->id, "FAIL", err);
            errcnt++;
        }

    /* LwSciIpcBindEventService */
    err = LwSciIpcBindEventService(ep->h, ptr);
    if (err == val) {
        printf(LWIPC_GEN_FMT, "LwSciIpcBindEventService", ep->id, "PASS", err);
    }
    else {
        errcnt++;
        printf(LWIPC_FAIL_FMT, "LwSciIpcBindEventService", ep->id, "FAIL", err);
    }

    /* LwciIpcGetEndpointInfo */
    err = LwSciIpcGetEndpointInfo(ep->h, ptr);
    if (err == val) {
        printf(LWIPC_GEN_FMT, "LwSciIpcGetEndpointInfo", ep->id, "PASS", err);
    }
    else {
        errcnt++;
        printf(LWIPC_FAIL_FMT, "LwSciIpcGetEndpointInfo", ep->id, "FAIL", err);
    }

    /* LwSciIpcGetEndpointInfoInternal */
    err = LwSciIpcGetEndpointInfoInternal(ep->h, ptr);
    if (err == val) {
        printf(LWIPC_GEN_FMT, "LwSciIpcGetEndpointInfoInternal", ep->id, "PASS", err);
    }
    else {
        errcnt++;
        printf(LWIPC_FAIL_FMT, "LwSciIpcGetEndpointInfoInternal", ep->id, "FAIL", err);
    }

    /* LwSciIpcGetEvent */
    err = LwSciIpcGetEvent(ep->h, ptr);
    if (err == val) {
        printf(LWIPC_GEN_FMT, "LwSciIpcGetEvent", ep->id, "PASS", err);
    }
    else {
        errcnt++;
        printf(LWIPC_FAIL_FMT, "LwSciIpcGetEvent", ep->id, "FAIL", err);
    }

    /* LwSciIpcRead */
    err = LwSciIpcRead(ep->h, ptr, ep->info.frame_size, &bytes);
    if (err == val) {
        printf(LWIPC_GEN_FMT, "LwSciIpcRead", ep->id, "PASS", err);
    }
    else {
        errcnt++;
        printf(LWIPC_FAIL_FMT, "LwSciIpcRead", ep->id, "FAIL", err);
    }
    err = LwSciIpcRead(ep->h, ep->buf, ep->info.frame_size, ptr);
    if (err == val) {
        printf(LWIPC_GEN_FMT, "LwSciIpcRead", ep->id, "PASS", err);
    }
    else {
        errcnt++;
        printf(LWIPC_FAIL_FMT, "LwSciIpcRead", ep->id, "FAIL", err);
    }

    /* LwSciIpcReadGetNextFrame */
    err = LwSciIpcReadGetNextFrame(ep->h, ptr);
    if (err == val) {
        printf(LWIPC_GEN_FMT, "LwSciIpcReadGetNextFrame", ep->id, "PASS", err);
    }
    else {
        errcnt++;
        printf(LWIPC_FAIL_FMT, "LwSciIpcReadGetNextFrame", ep->id, "FAIL", err);
    }

    /* LwSciIpcReadPeek */
    err = LwSciIpcReadPeek(ep->h, ptr, 0, ep->info.frame_size, &bytes);
    if (err == val) {
        printf(LWIPC_FAIL_FMT, "LwSciIpcReadPeek", ep->id, "PASS", err);
    }
    else {
        errcnt++;
        printf(LWIPC_FAIL_FMT, "LwSciIpcReadPeek", ep->id, "FAIL", err);
    }
    err = LwSciIpcReadPeek(ep->h, ep->buf, 0, ep->info.frame_size, ptr);
    if (err == val) {
        printf(LWIPC_FAIL_FMT, "LwSciIpcReadPeek", ep->id, "PASS", err);
    }
    else {
        errcnt++;
        printf(LWIPC_FAIL_FMT, "LwSciIpcReadPeek", ep->id, "FAIL", err);
    }

    /* LwSciIpcWrite */
    err = LwSciIpcWrite(ep->h, ptr, ep->info.frame_size, &bytes);
    if (err == val) {
        printf(LWIPC_GEN_FMT, "LwSciIpcWrite", ep->id, "PASS", err);
    }
    else {
        errcnt++;
        printf(LWIPC_FAIL_FMT, "LwSciIpcWrite", ep->id, "FAIL", err);
    }
    err = LwSciIpcWrite(ep->h, ep->buf, ep->info.frame_size, ptr);
    if (err == val) {
        printf(LWIPC_GEN_FMT, "LwSciIpcWrite", ep->id, "PASS", err);
    }
    else {
        errcnt++;
        printf(LWIPC_FAIL_FMT, "LwSciIpcWrite", ep->id, "FAIL", err);
    }

    /* LwSciIpcWriteGetNextFrame */
    err = LwSciIpcWriteGetNextFrame(ep->h, ptr);
    if (err == val) {
        printf(LWIPC_GEN_FMT, "LwSciIpcWriteGetNextFrame", ep->id, "PASS", err);
    }
    else {
        errcnt++;
        printf(LWIPC_FAIL_FMT, "LwSciIpcWriteGetNextFrame", ep->id, "FAIL", err);
    }

    /* LwSciIpcWritePoke */
    err = LwSciIpcWritePoke(ep->h, ptr, 0, ep->info.frame_size, &bytes);
    if (err == val) {
        printf(LWIPC_FAIL_FMT, "LwSciIpcWritePoke", ep->id, "PASS", err);
    }
    else {
        errcnt++;
        printf(LWIPC_FAIL_FMT, "LwSciIpcWritePoke", ep->id, "FAIL", err);
    }
    err = LwSciIpcWritePoke(ep->h, ep->buf, 0, ep->info.frame_size, ptr);
    if (err == val) {
        printf(LWIPC_FAIL_FMT, "LwSciIpcWritePoke", ep->id, "PASS", err);
    }
    else {
        errcnt++;
        printf(LWIPC_FAIL_FMT, "LwSciIpcWritePoke", ep->id, "FAIL", err);
    }

    /* LwSciIpcEndpointGetVuid */
    err = LwSciIpcEndpointGetVuid(ep->h, ptr);
    if (err == val) {
        printf(LWIPC_FAIL_FMT, "LwSciIpcEndpointGetVuid", ep->id, "PASS", err);
    }
    else {
        errcnt++;
        printf(LWIPC_FAIL_FMT, "LwSciIpcEndpointGetVuid", ep->id, "FAIL", err);
    }

    /* LwSciIpcEndpointGetAuthToken */
    err = LwSciIpcEndpointGetAuthToken(ep->h, ptr);
    if (err == val) {
        printf(LWIPC_FAIL_FMT, "LwSciIpcEndpointGetAuthToken", ep->id, "PASS", err);
    }
    else {
        errcnt++;
        printf(LWIPC_FAIL_FMT, "LwSciIpcEndpointGetAuthToken", ep->id, "FAIL", err);
    }

        /* close endpoint since handle is no more required */
        LwSciIpcCloseEndpoint(ep->h);

    /* LwSciIpcEndpointValidateAuthTokenQnx */
    err = LwSciIpcEndpointValidateAuthTokenQnx(NULL, 0, &vuid);
    if (err == val) {
        printf(LWIPC_FAIL_FMT, "LwSciIpcEndpointValidateAuthTokenQnx", ep->id, "PASS", err);
    }
    else {
        errcnt++;
        printf(LWIPC_FAIL_FMT, "LwSciIpcEndpointValidateAuthTokenQnx", ep->id, "FAIL", err);
    }
    err = LwSciIpcEndpointValidateAuthTokenQnx(&ctp, 0, NULL);
    if (err == val) {
        printf(LWIPC_FAIL_FMT, "LwSciIpcEndpointValidateAuthTokenQnx", ep->id, "PASS", err);
    }
    else {
        errcnt++;
        printf(LWIPC_FAIL_FMT, "LwSciIpcEndpointValidateAuthTokenQnx", ep->id, "FAIL", err);
    }

    /* LwSciIpcEndpointMapVuid */
    err = LwSciIpcEndpointMapVuid(vuid, NULL, &peerVuid);
    if (err == val) {
        printf(LWIPC_FAIL_FMT, "LwSciIpcEndpointMapVuid", ep->id, "PASS", err);
    }
    else {
        errcnt++;
        printf(LWIPC_FAIL_FMT, "LwSciIpcEndpointMapVuid", ep->id, "FAIL", err);
    }
    err = LwSciIpcEndpointMapVuid(vuid, &peerTopoId, NULL);
    if (err == val) {
        printf(LWIPC_FAIL_FMT, "LwSciIpcEndpointMapVuid", ep->id, "PASS", err);
    }
    else {
        errcnt++;
        printf(LWIPC_FAIL_FMT, "LwSciIpcEndpointMapVuid", ep->id, "FAIL", err);
    }

    /* LwSciIpcGetEndpointAccessInfo */
    err = LwSciIpcGetEndpointAccessInfo(NULL, &accessInfo);
    if (err == val) {
        printf(LWIPC_FAIL_FMT, "LwSciIpcEndpointAccessInfo", ep->id, "PASS", err);
    }
    else {
        errcnt++;
        printf(LWIPC_FAIL_FMT, "LwSciIpcEndpointAccessInfo", ep->id, "FAIL", err);
    }
    err = LwSciIpcGetEndpointAccessInfo(ep->chname, NULL);
    if (err == val) {
        printf(LWIPC_FAIL_FMT, "LwSciIpcEndpointAccessInfo", ep->id, "PASS", err);
    }
    else {
        errcnt++;
        printf(LWIPC_FAIL_FMT, "LwSciIpcEndpointAccessInfo", ep->id, "FAIL", err);
    }

    /* LwSciEventLoopServiceCreate with NULL pointer */
    err = LwSciEventLoopServiceCreate(1U, NULL);
    if (err == val) {
        printf(LWIPC_FAIL_FMT, "LwSciEventLoopServiceCreate", ep->id, "PASS", err);
    }
    else {
        errcnt++;
        printf(LWIPC_FAIL_FMT, "LwSciEventLoopServiceCreate", ep->id, "FAIL", err);
    }

    /* successful loopservice creation for next testcases */
    err = LwSciEventLoopServiceCreate(1U, &tp->eventLoopService);
    if (err == LwSciError_Success) {
        printf(LWIPC_FAIL_FMT, "LwSciEventLoopServiceCreate", ep->id, "PASS", err);
    }
    else {
        errcnt++;
        printf(LWIPC_FAIL_FMT, "LwSciEventLoopServiceCreate", ep->id, "FAIL", err);
    }

    /* LwSciIpcOpenEndpointWithEventService */
    err = LwSciIpcOpenEndpointWithEventService(ptr, &ep->h, &tp->eventLoopService->EventService);
    if (err == val) {
        printf(LWIPC_GEN_FMT, "LwSciIpcOpenEndpointWithEventService", ep->id, "PASS", err);
    }
    else {
        errcnt++;
        printf(LWIPC_FAIL_FMT, "LwSciIpcOpenEndpointWithEventService", ep->id, "FAIL", err);
    }

    err = LwSciIpcOpenEndpointWithEventService(ep->chname, ptr, &tp->eventLoopService->EventService);
    if (err == val) {
        printf(LWIPC_GEN_FMT, "LwSciIpcOpenEndpointWithEventService", ep->id, "PASS", err);
    }
    else {
        errcnt++;
        printf(LWIPC_FAIL_FMT, "LwSciIpcOpenEndpointWithEventService", ep->id, "FAIL", err);
    }

    /* LwSciIpcOpenEndpointWithEventService with NULL pointer */
    err = LwSciIpcOpenEndpointWithEventService(ep->chname, &ep->h, ptr);
    if (err == val) {
        printf(LWIPC_GEN_FMT, "LwSciIpcOpenEndpointWithEventService", ep->id, "PASS", err);
    }
    else {
        errcnt++;
        printf(LWIPC_FAIL_FMT, "LwSciIpcOpenEndpointWithEventService", ep->id, "FAIL", err);
    }

    /* successful open endpoint with event service for next testcases */
    err = LwSciIpcOpenEndpointWithEventService(ep->chname, &ep->h,
        &tp->eventLoopService->EventService);
    if (err == LwSciError_Success) {
        printf(LWIPC_GEN_FMT, "LwSciIpcOpenEndpointWithEventService", ep->id, "PASS", err);
    }
    else {
        errcnt++;
        printf(LWIPC_FAIL_FMT, "LwSciIpcOpenEndpointWithEventService", ep->id, "FAIL", err);
    }

    /* LwSciIpcGetEventNotifier */
    err = LwSciIpcGetEventNotifier(ep->h, ptr);
    if (err == val) {
        printf(LWIPC_GEN_FMT, "LwSciIpcGetEventNotifier", ep->id, "PASS", err);
    }
    else {
        errcnt++;
        printf(LWIPC_FAIL_FMT, "LwSciIpcGetEventNotifier", ep->id, "FAIL", err);
    }

    /* WaitForEvent */
    err = tp->eventLoopService->WaitForEvent(ptr, LW_SCI_EVENT_INFINITE_WAIT);
    if (err == val) {
        printf(LWIPC_GEN_FMT, "EventService->WaitForEvent", ep->id, "PASS", err);
    }
    else {
        errcnt++;
        printf(LWIPC_FAIL_FMT, "EventService->WaitForEvent", ep->id, "FAIL", err);
    }

        LwSciIpcCloseEndpoint(ep->h);
        if (tp->eventLoopService != NULL) {
            tp->eventLoopService->EventService.Delete(&tp->eventLoopService->EventService);
        }

    if (errcnt == 0) {
        printf(LWIPC_GEN_FMT, "NULL POINTER TEST", ep->id, "PASS", LwSciError_Success);
    }
    else {
        tp->err_cnt++;
        printf("%-40s%02d: %4s (err cnt: %d)\n",
            "NULL POINTER TEST", ep->id, "FAIL", errcnt);
    }
}

void unit_test_init_event_service(void)
{
    test_params *tp = &s_params;
    LwSciError err;

    tp->test_cnt++;
    /* successful loopservice creation for next testcases */
    err = LwSciEventLoopServiceCreate(1U, &tp->eventLoopService);
    if (err == LwSciError_Success) {
        printf(LWIPC_FAIL_FMT, "LwSciEventLoopServiceCreate", 0, "PASS", err);

        /* NOT supported function */
        err = tp->eventLoopService->CreateEventLoop(tp->eventLoopService, NULL);
        if (err == LwSciError_NotSupported) {
            printf(LWIPC_FAIL_FMT, "LwSciCreateEventLoop", 0, "PASS", err);
        }
        /* NOT supported function */
        err = tp->eventLoopService->EventService.CreateTimerEvent(&tp->eventLoopService->EventService, NULL);
        if (err == LwSciError_NotSupported) {
            printf(LWIPC_FAIL_FMT, "LwSciCreateTimerEvent", 0, "PASS", err);
        }
    }
    else {
        tp->err_cnt++;
        printf(LWIPC_FAIL_FMT, "LwSciEventLoopServiceCreate", 0, "FAIL", err);
    }
}

void unit_test_delete_event_notifier(struct endpoint *ep)
{
    test_params *tp = &s_params;

    tp->test_cnt++;

    ep->notifier->Delete(ep->notifier);
    printf(LWIPC_FAIL_FMT, "EventNotifier->Delete", 0, "PASS", 0);
}

void unit_test_deinit_event_service(void)
{
    test_params *tp = &s_params;

    tp->test_cnt++;

    tp->eventLoopService->EventService.Delete(&tp->eventLoopService->EventService);
    printf(LWIPC_FAIL_FMT, "EventService.Delete", 0, "PASS", 0);
}


void unit_test_wait_event_notifier(struct endpoint *ep1, struct endpoint *ep2)
{
    test_params *tp = &s_params;
    LwSciLocalEvent *local;
    bool newEvent[1];
    LwSciEventNotifier *evtNotiArray[1];
    bool est1 = 0;
    bool est2 = 0;
    uint32_t evt1 = 0;
    uint32_t evt2 = 0;
    int32_t errcnt = 0;
    LwSciError err;

    tp->test_cnt++;

    LwSciIpcResetEndpoint(ep1->h);
    printf(LWIPC_GEN_FMT, "LwSciIpcResetEndpoint", ep1->id, "PASS", 0);
    LwSciIpcResetEndpoint(ep2->h);
    printf(LWIPC_GEN_FMT, "LwSciIpcResetEndpoint", ep2->id, "PASS", 0);

    while (!s_Stop && (!est1 || !est2)) {
        err = LwSciIpcGetEvent(ep1->h, &evt1);
        if (err != LwSciError_Success) {
            printf("%-40s%02d: %4s (EST:%d,event:0x%x)\n",
                "LwSciIpcGetEvent", ep1->id, "FAIL", est1, evt1);
            errcnt++;
            break;
        }
        if (evt1 & LW_SCI_IPC_EVENT_CONN_RESET) {
            est1 = 0;
        }
        else if (evt1 & LW_SCI_IPC_EVENT_CONN_EST_ALL) {
            est1 = 1;
            goto ep2;
        }

        /* WaitForEvent */
        err = tp->eventLoopService->WaitForEvent(ep1->notifier, 1000);
        if (err == LwSciError_Success) {
            printf(LWIPC_GEN_FMT, "EventService->WaitForEvent", ep1->id, "PASS", err);
        }
        else {
            errcnt++;
            printf(LWIPC_FAIL_FMT, "EventService->WaitForEvent", ep1->id, "FAIL", err);
        }


ep2:
        err = LwSciIpcGetEvent(ep2->h, &evt2);
        if (err != LwSciError_Success) {
            printf("%-40s%02d: %4s (EST:%d,event:0x%x)\n",
                "LwSciIpcGetEvent", ep2->id, "FAIL", est2, evt2);
            errcnt++;
            break;
        }
        if (evt2 & LW_SCI_IPC_EVENT_CONN_RESET) {
            est2 = 0;
        }
        else if (evt2 & LW_SCI_IPC_EVENT_CONN_EST_ALL) {
            est2 = 1;
            continue;
        }

        err = tp->eventLoopService->WaitForEvent(ep2->notifier, 1000);
        if (err == LwSciError_Success) {
            printf(LWIPC_GEN_FMT, "EventService->WaitForEvent", ep2->id, "PASS", err);
        }
        else {
            errcnt++;
            printf(LWIPC_FAIL_FMT, "EventService->WaitForEvent", ep2->id, "FAIL", err);
        }
    }

    err = tp->eventLoopService->EventService.CreateLocalEvent(
        &tp->eventLoopService->EventService, &local);
    if (err == LwSciError_Success) {
        printf(LWIPC_GEN_FMT, "EventService->CreateLocalEvent", 0, "PASS", err);
    }
    else {
        errcnt++;
        printf(LWIPC_FAIL_FMT, "EventService->CreateLocalEvent", 0, "FAIL", err);
    }

    err = local->Signal(local);
    if (err == LwSciError_Success) {
        printf(LWIPC_GEN_FMT, "LocalEvent->Signal", 0, "PASS", err);
    }
    else {
        errcnt++;
        printf(LWIPC_FAIL_FMT, "LocalEvent->Signal", 0, "FAIL", err);
    }


    evtNotiArray[0] = local->eventNotifier;
    err = tp->eventLoopService->WaitForMultipleEvents(
                evtNotiArray, 1, LW_SCI_EVENT_INFINITE_WAIT, newEvent);
    if (err == LwSciError_Success) {
        printf(LWIPC_GEN_FMT, "EventService->WaitForMultipleEvents", 0, "PASS", err);
    }
    else {
        errcnt++;
        printf(LWIPC_FAIL_FMT, "EventService->WaitForMultipleEvents", 0, "FAIL", err);
    }

    local->eventNotifier->Delete(local->eventNotifier);
    printf(LWIPC_FAIL_FMT, "EventNotifier->Delete", 0, "PASS", 0);

    local->Delete(local);
    printf(LWIPC_GEN_FMT, "LocalEvent->Delete", 0, "PASS", 0);

    if (errcnt != 0) {
        tp->err_cnt++;
    }
}

static void callback(void *cookie) {}
void unit_test_open_endpoint_eventservice(struct endpoint *ep, int32_t id,
    char *chname, LwSciError val)
{
    test_params *tp = &s_params;
    char tst_chname[LWSCIIPC_MAX_ENDPOINT_NAME];
    int32_t errcnt = 0;
    LwSciError err;

    tp->test_cnt++;

    /* override endpoint name with provided one */
    if (chname != NULL) {
        memcpy(tst_chname, chname, sizeof(tst_chname));
    }
    else {
        memcpy(tst_chname, ep->chname, sizeof(tst_chname));
    }

    err = LwSciIpcOpenEndpointWithEventService(tst_chname, &ep->h,
        &tp->eventLoopService->EventService);
    if (err == val) {
        ep->id = id;    /* set endpoint id */
        printf("%-40s%02d: %4s (ret: %d, chName: %s, Handle: 0x%lx)\n",
            "LwSciIpcOpenEndpointWithEventService", ep->id, "PASS", err, tst_chname, ep->h);
    }
    else {
        printf(LWIPC_FAIL_FMT, "LwSciIpcOpenEndpointWithEventService", ep->id, "FAIL", err);
        errcnt++;
    }

    err = LwSciIpcGetEventNotifier(ep->h, &ep->notifier);
    if (err == LwSciError_Success) {
        printf("%-40s%02d: %4s (notifier: 0x%tx)\n",
            "LwSciIpcGetEventNotifier", ep->id, "PASS", (uintptr_t)ep->notifier);

        /* NOT supported handler test */
        err = ep->notifier->SetHandler(ep->notifier, callback, NULL, 0);
        if (err == LwSciError_NotSupported) {
            printf(LWIPC_FAIL_FMT, "eventNotifier->SetHandler", ep->id, "PASS", err);
        }
    }
    else {
        printf(LWIPC_FAIL_FMT, "LwSciIpcGetEventNotifier", ep->id, "FAIL", err);
        errcnt++;
    }

    if (errcnt != 0) {
        tp->err_cnt++;
    }
}

void unit_test_bind_eventservice(struct endpoint *ep, int32_t id,
    LwSciError val)
{
    test_params *tp = &s_params;
    int32_t errcnt = 0;
    LwSciError err;

    tp->test_cnt++;

    err = LwSciIpcBindEventService(ep->h,
        &tp->eventLoopService->EventService);
    if (err == val) {
        ep->id = id;    /* set endpoint id */
        printf("%-40s%02d: %4s (ret: %d, Handle: 0x%lx)\n",
            "LwSciIpcBindEventService", ep->id, "PASS", err, ep->h);
    }
    else {
        printf(LWIPC_FAIL_FMT, "LwSciIpcBindEventService", ep->id, "FAIL", err);
        errcnt++;
    }

    err = LwSciIpcGetEventNotifier(ep->h, &ep->notifier);
    if (err == LwSciError_Success) {
        printf("%-40s%02d: %4s (notifier: 0x%tx)\n",
            "LwSciIpcGetEventNotifier", ep->id, "PASS", (uintptr_t)ep->notifier);

        /* NOT supported handler test */
        err = ep->notifier->SetHandler(ep->notifier, callback, NULL, 0);
        if (err == LwSciError_NotSupported) {
            printf(LWIPC_FAIL_FMT, "eventNotifier->SetHandler", ep->id, "PASS", err);
        }
    }
    else {
        printf(LWIPC_FAIL_FMT, "LwSciIpcGetEventNotifier", ep->id, "FAIL", err);
        errcnt++;
    }

    if (errcnt != 0) {
        tp->err_cnt++;
    }
}

/*
 * API call test
 *
 * h   : input handle
 * val : expected return val
 */
void unit_test_call_apis(struct endpoint *ep, LwSciIpcEndpoint h, LwSciError val)
{
    test_params *tp = &s_params;
    struct LwSciIpcEndpointInfo info;
    LwSciEventNotifier *notifier;
    uint32_t event;
    const volatile void *rbuf;
    volatile void *wbuf;
    int32_t errcnt = 0;
    int32_t bytes;
    LwSciError err;

    tp->test_cnt++;

    printf("%-40s%02d: %4s (val: 0x%" PRIx64")\n",
            "TEST HANDLE", ep->id, "", h);

    err = LwSciIpcBindEventService(h, NULL);
    if (err == val) {
        printf(LWIPC_GEN_FMT, "LwSciIpcBindEventService", ep->id, "PASS", err);
    }
    else {
        errcnt++;
        printf(LWIPC_FAIL_FMT, "LwSciIpcBindEventService", ep->id, "FAIL", err);
    }

    err = LwSciIpcGetEndpointInfo(h, &info);
    if (err == val) {
        printf(LWIPC_GEN_FMT, "LwSciIpcGetEndpointInfo", ep->id, "PASS", err);
    }
    else {
        errcnt++;
        printf(LWIPC_FAIL_FMT, "LwSciIpcGetEndpointInfo", ep->id, "FAIL", err);
    }
    err = LwSciIpcGetEventNotifier(h, &notifier);
    if (err == val) {
        printf(LWIPC_GEN_FMT, "LwSciIpcGetEventNotifier", ep->id, "PASS", err);
    }
    else {
        errcnt++;
        printf(LWIPC_FAIL_FMT, "LwSciIpcGetEventNotifier", ep->id, "FAIL", err);
    }
    err = LwSciIpcSetQnxPulseParam(h, 1, -1, 1, 0);
    if (err == val) {
        printf(LWIPC_GEN_FMT, "LwSciIpcSetQnxPulseParam", ep->id, "PASS", err);
    }
    else {
        errcnt++;
        printf(LWIPC_FAIL_FMT, "LwSciIpcSetQnxPulseParam", ep->id, "FAIL", err);
    }
    err = LwSciIpcGetEvent(h, &event);
    if (err == val) {
        printf(LWIPC_GEN_FMT, "LwSciIpcGetEvent", ep->id, "PASS", err);
    }
    else {
        errcnt++;
        printf(LWIPC_FAIL_FMT, "LwSciIpcGetEvent", ep->id, "FAIL", err);
    }
    err = LwSciIpcRead(h, ep->buf, ep->info.frame_size, &bytes);
    if (err == val) {
        printf(LWIPC_GEN_FMT, "LwSciIpcRead", ep->id, "PASS", err);
    }
    else {
        errcnt++;
        printf(LWIPC_FAIL_FMT, "LwSciIpcRead", ep->id, "FAIL", err);
    }
    err = LwSciIpcReadGetNextFrame(h, &rbuf);
    if (err == val) {
        printf(LWIPC_GEN_FMT, "LwSciIpcReadGetNextFrame", ep->id, "PASS", err);
    }
    else {
        errcnt++;
        printf(LWIPC_FAIL_FMT, "LwSciIpcReadGetNextFrame", ep->id, "FAIL", err);
    }
    err = LwSciIpcReadAdvance(h);
    if (err == val) {
        printf(LWIPC_GEN_FMT, "LwSciIpcReadAdvance", ep->id, "PASS", err);
    }
    else {
        errcnt++;
        printf(LWIPC_FAIL_FMT, "LwSciIpcReadAdvance", ep->id, "FAIL", err);
    }
    err = LwSciIpcWrite(h, ep->buf, ep->info.frame_size, &bytes);
    if (err == val) {
        printf(LWIPC_GEN_FMT, "LwSciIpcWrite", ep->id, "PASS", err);
    }
    else {
        errcnt++;
        printf(LWIPC_FAIL_FMT, "LwSciIpcWrite", ep->id, "FAIL", err);
    }
    err = LwSciIpcWriteGetNextFrame(h, &wbuf);
    if (err == val) {
        printf(LWIPC_GEN_FMT, "LwSciIpcWriteGetNextFrame", ep->id, "PASS", err);
    }
    else {
        errcnt++;
        printf(LWIPC_FAIL_FMT, "LwSciIpcWriteGetNextFrame", ep->id, "FAIL", err);
    }
    err = LwSciIpcWriteAdvance(h);
    if (err == val) {
        printf(LWIPC_GEN_FMT, "LwSciIpcWriteAdvance", ep->id, "PASS", err);
    }
    else {
        errcnt++;
        printf(LWIPC_FAIL_FMT, "LwSciIpcWriteAdvance", ep->id, "FAIL", err);
    }
    LwSciIpcResetEndpoint(h);
    LwSciIpcCloseEndpoint(h);

    if (errcnt == 0) {
        printf(LWIPC_GEN_FMT, "API CALL TEST", ep->id, "PASS", LwSciError_Success);
    }
    else {
        tp->err_cnt++;
        printf("%-40s%02d: %4s (err cnt: %d)\n",
            "API CALL TEST", ep->id, "FAIL", errcnt);
    }
}

/*
 * API call test2
 *
 * following APIs return bool type
 * LwSciIpcCanRead()
 * LwSciIpcCanWrite()
 *
 * h   : input handle
 * val : expected return val
 */
void unit_test_call_apis2(struct endpoint *ep, LwSciIpcEndpoint h, bool val)
{
    test_params *tp = &s_params;
    int32_t errcnt = 0;
    bool ret;

    tp->test_cnt++;

    printf("%-40s%02d: %4s (val: 0x%" PRIx64")\n",
            "TEST HANDLE", ep->id, "", h);

    ret = LwSciIpcCanRead(h);
    if (ret == val) {
        printf(LWIPC_GEN_FMT, "LwSciIpcCanRead", ep->id, "PASS", ret);
    }
    else {
        errcnt++;
        printf(LWIPC_FAIL_FMT, "LwSciIpcCanRead", ep->id, "FAIL", ret);
    }

    ret = LwSciIpcCanWrite(h);
    if (ret == val) {
        printf(LWIPC_GEN_FMT, "LwSciIpcCanWrite", ep->id, "PASS", ret);
    }
    else {
        errcnt++;
        printf(LWIPC_FAIL_FMT, "LwSciIpcCanWrite", ep->id, "FAIL", ret);
    }

    if (errcnt == 0) {
        printf(LWIPC_GEN_FMT, "API CALL TEST2", ep->id, "PASS", LwSciError_Success);
    }
    else {
        tp->err_cnt++;
        printf("%-40s%02d: %4s (err cnt: %d)\n",
            "API CALL TEST2", ep->id, "FAIL", errcnt);
    }
}


void unit_test_negative_res_access(void)
{
    test_params *tp = &s_params;
    struct endpoint *ep = &tp->e1;
    void *mapaddr;
    struct sigevent event;
    int32_t iid;
    uint64_t mask = (getpagesize() - 1UL);

    tp->test_cnt++;
    tp->physize = (tp->physize + mask) & (~mask);
    mapaddr = mmap(NULL, tp->physize, PROT_READ|PROT_WRITE,
        MAP_SHARED|MAP_PHYS, NOFD, tp->phyaddr);
    if (mapaddr == MAP_FAILED) {
        printf("%-40s%02d: %4s (0x%lx:0x%lx,err:%d)\n",
            "PhyMem Negative", 0, "PASS", tp->phyaddr, tp->physize, errno);
    }
    else {
        (void)munmap(mapaddr, tp->physize);
        tp->err_cnt++;
        printf("%-40s%02d: %4s (0x%lx:0x%lx,err:%d)\n",
            "PhyMem Negative", 0, "FAIL", tp->phyaddr, tp->physize, errno);
    }

    tp->test_cnt++;
    if (ep->chid == 0) {
        ep->chid = ChannelCreate_r(0U);
        if (ep->chid < 0) {
            printf(LWIPC_FAIL_FMT, "ChannelCreate_r", 0, "FAIL", ep->chid);
            tp->err_cnt++;
        }
    }
    if (ep->coid == 0) {
        ep->coid = ConnectAttach_r(0, 0, ep->chid, _NTO_SIDE_CHANNEL, 0);
        if (ep->coid < 0) {
            printf(LWIPC_FAIL_FMT, "ConnectAttach_r", 0, "FAIL", ep->coid);
            tp->err_cnt++;
        }
    }

    SIGEV_PULSE_INIT(&event, ep->coid, TEST_PULSE_PRIO, TEST_PULSE_CODE1, (void *)NULL);
    iid = InterruptAttachEvent_r(tp->irq, &event, _NTO_INTR_FLAGS_TRK_MSK);
    if (iid < 0) {
        printf("%-40s%02d: %4s (irq:%d, err:%d)\n",
            "IRQ Negative", 0, "PASS", tp->irq, iid);
    }
    else {
        (void)InterruptDetach_r(iid);
        tp->err_cnt++;
        printf("%-40s%02d: %4s (irq:%d, err:%d)\n",
            "IRQ Negative", 0, "FAIL", tp->irq, iid);
    }
}

int32_t main(int32_t argc, char **argv)
{
    test_params *tp = &s_params;
    int32_t opt;
    int32_t optcnt = 0;
    int32_t retval = -1;
    uint32_t i;

    tp->iterations = DEFAULT_ITERATIONS;
    tp->range_lo = 0;
    tp->range_hi = INT_MAX;
    tp->negative = false;
    tp->timeout = WAIT_TIME_MS; /* default timeout */

    while ((opt = getopt(argc, argv, "a:n:s:r:u:w:h")) != EOF)
    {
        switch (opt) {
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
                        printf("%s: range_lo/range_hi not specified correctly\n", argv[0]);
                        return -EILWAL;
                    }
                    break;
                }
            case 'n':
                /* negative test for resource access(phy/irq) */
                {
                    char* token = strtok(optarg, ",");

                    if (token != NULL) {
                        tp->phyaddr = strtoul(token, NULL, 16);
                        token = strtok(NULL, ",");
                        if (token != NULL) {
                            tp->physize = strtoul(token, NULL, 16);
                            token = strtok(NULL, ",");
                            if (token != NULL) {
                                tp->irq = strtoul(token, NULL, 10);
                                retval = 0;
                            }
                        }
                    }
                    if (retval != 0) {
                        printf("%s: phyAddr/Size/IRQ not specified correctly\n", argv[0]);
                        return -EILWAL;
                    }
                    tp->negative = true;
                    optcnt += 2;
                    break;
                }
            case 'r':
                /* Rx LwSciIpc endpoint name */
                strncpy(tp->e2.chname, optarg, sizeof(tp->e2.chname));
                printf("Rx Endpoint Name: %s\n", tp->e2.chname);
                optcnt++;
                break;
            case 's':
                /* Tx LwSciIpc endpoint name */
                strncpy(tp->e1.chname, optarg, sizeof(tp->e1.chname));
                printf("Tx Endpoint Name: %s\n", tp->e1.chname);
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
                            retval = 0;
                        }
                    }
                    if (retval != 0) {
                        printf("%s: UID/GID not specified correctly\n", argv[0]);
                        return -EILWAL;
                    }
                    break;
                }
            case 'w':
                /* Set pulse waiting timeout */
                tp->timeout = strtoul(optarg, NULL, 0);
                break;
            case 'h':
            default:
                print_usage(argv[0]);
                return 0;
        }
    }

    tp->prgname = basename(argv[0]);
    if (optcnt < 2) {
        print_usage(argv[0]);
        return -EILWAL;
    }
    setup_termination_handlers();
    srand(time(NULL));

    /* show current id info */
    print_ids(__func__);

    printf("\n[LWIPC UNIT TEST]\n");

#ifdef __QNX__
    drop_privileges();
#endif /* __QNX__ */

    if (!tp->negative) {
        printf("\n[NO-INIT TEST]\n");
        tp->e1.tsthandle = 0x12345678UL;
        tp->e2.tsthandle = 0x12345678UL;
        /* call APIs (non-zero fake handle) w/o init (LwSciError_NotInitialized) */
        unit_test_call_apis(&tp->e1, (LwSciIpcEndpoint)tp->e1.tsthandle,
            LwSciError_NotInitialized);

        printf("\n[ERR COLWERSION TEST]\n");
        unit_test_err_colwersion();
    }

    printf("\n[INIT TEST]\n");
    unit_test_init();

    if (tp->negative) {
        printf("\n[NEGATIVE RES ACCESS TEST]\n");
        unit_test_negative_res_access();
        goto done;
    }

    printf("\n[NO-OPEN TEST]\n");
    /* call APIs (non-zero fake handle) w/o open (LwSciError_BadParameter) */
    /* internal handle is not allocated yet */
    unit_test_call_apis2(&tp->e1, (LwSciIpcEndpoint)tp->e1.tsthandle, false);
    unit_test_call_apis(&tp->e1, (LwSciIpcEndpoint)tp->e1.tsthandle,
        LwSciError_BadParameter);

    printf("\n[OPEN TEST W/ ERROR CH NAME]\n");
    /* open abnormal endpoint name (LwSciError_NoSuchEntry) */
    unit_test_open_endpoint(&tp->e1, 1, "bad endpoint", LwSciError_NoSuchEntry);
    unit_test_open_endpoint(&tp->e2, 2, "bad endpoint", LwSciError_NoSuchEntry);

    printf("\n[CALL API W/ NULL POINTER]\n");
    unit_test_NULL_pointer(&tp->e1, NULL, LwSciError_BadParameter);
    unit_test_NULL_pointer(&tp->e2, NULL, LwSciError_BadParameter);

    printf("\n[OPEN TEST]\n");
    unit_test_open_endpoint(&tp->e1, 1, NULL, LwSciError_Success);
    unit_test_open_endpoint(&tp->e2, 2, NULL, LwSciError_Success);

    printf("\n[OPEN AGAIN TEST]\n");
    /* open again and check error */
    unit_test_open_endpoint(&tp->e1, 1, NULL, LwSciError_Busy);
    unit_test_open_endpoint(&tp->e2, 2, NULL, LwSciError_Busy);

    printf("\n[CALL API W/ ERROR HANDLE]\n");
    unit_test_error_handle(&tp->e1);
    unit_test_error_handle(&tp->e2);

#ifdef __QNX__
    printf("\n[QNX:SET PULSE PARAM]\n");
    unit_test_set_event_pulse(&tp->e1, TEST_PULSE_PRIO,
        TEST_PULSE_CODE1, (void *)NULL);
    unit_test_set_event_pulse(&tp->e2, TEST_PULSE_PRIO,
        TEST_PULSE_CODE2, (void *)NULL);
#endif /* __QNX__ */

    printf("\n[GET INFO TEST]\n");
    unit_test_get_ch_info(&tp->e1);
    unit_test_get_ch_info(&tp->e2);

    setup_buffer(tp);

    printf("\n[RESET CHANNEL]\n");
    unit_test_reset(&tp->e1);
    unit_test_reset(&tp->e2);

    printf("\n[WAIT CONNECTION]\n");
    unit_test_poll_notification(tp);

    printf("\n[R/W TEST W/ ERROR SIZE]\n");
    tp->frame_size = tp->e1.info.frame_size * 2; // error size
    unit_test_send_with_return(&tp->e1, LwSciError_BadParameter);
    tp->frame_size = 0; // back to normal size
    unit_test_send_with_return(&tp->e1, LwSciError_Success);
    tp->frame_size = tp->e1.info.frame_size * 2; // error size
    unit_test_receive_with_return(&tp->e2, LwSciError_BadParameter);
    tp->frame_size = 0; // back to normal size

    printf("\n[NO-MEM TEST]\n");
    fill_buffer(&tp->e1, RANDOM);
    unit_test_write_after_fill(&tp->e1);
    unit_test_read_after_drain(&tp->e2);
    fill_buffer(&tp->e2, RANDOM);
    unit_test_write_after_fill(&tp->e2);
    unit_test_read_after_drain(&tp->e1);

    printf("\n[POLLING TEST]\n");
    for(i = 0;i<tp->iterations;i++) {
        fill_buffer(&tp->e1, RANDOM);
        unit_test_poll_send_receive(&tp->e1, &tp->e2);
        unit_test_compare_data(&tp->e1, &tp->e2, COMPARE_ENDS);

        fill_buffer(&tp->e2, RANDOM);
        unit_test_poll_send_receive(&tp->e2, &tp->e1);
        unit_test_compare_data(&tp->e2, &tp->e1, COMPARE_ENDS);

        fill_buffer(&tp->e1, RANDOM);
        unit_test_poll_send_receive_zc(&tp->e1, &tp->e2);
        unit_test_compare_data(&tp->e1, &tp->e2, COMPARE_ENDS);

        fill_buffer(&tp->e2, RANDOM);
        unit_test_poll_send_receive_zc(&tp->e2, &tp->e1);
        unit_test_compare_data(&tp->e2, &tp->e1, COMPARE_ENDS);

        fill_buffer(&tp->e1, RANDOM);
        unit_test_poll_send_receive_pokepeek(&tp->e1, &tp->e2);
        unit_test_compare_data(&tp->e1, &tp->e2, COMPARE_ENDS);

        fill_buffer(&tp->e2, RANDOM);
        unit_test_poll_send_receive_pokepeek(&tp->e2, &tp->e1);
        unit_test_compare_data(&tp->e2, &tp->e1, COMPARE_ENDS);
    }

    printf("\n[RESET CHANNEL]\n");
    unit_test_reset(&tp->e1);
    unit_test_reset(&tp->e2);

    printf("\n[WAIT CONNECTION]\n");
    unit_test_notification(tp);

    printf("\n[EVENT HANDLE TEST]\n");
    for(i = 0;i<tp->iterations;i++) {
        fill_buffer(&tp->e1, RANDOM);
        unit_test_send_receive(&tp->e1, &tp->e2);
        unit_test_compare_data(&tp->e1, &tp->e2, COMPARE_ENDS);

        fill_buffer(&tp->e2, RANDOM);
        unit_test_send_receive(&tp->e2, &tp->e1);
        unit_test_compare_data(&tp->e2, &tp->e1, COMPARE_ENDS);

        fill_buffer(&tp->e1, RANDOM);
        unit_test_send_receive_zc(&tp->e1, &tp->e2);
        unit_test_compare_data(&tp->e1, &tp->e2, COMPARE_ENDS);

        fill_buffer(&tp->e2, RANDOM);
        unit_test_send_receive_zc(&tp->e2, &tp->e1);
        unit_test_compare_data(&tp->e2, &tp->e1, COMPARE_ENDS);

        fill_buffer(&tp->e1, RANDOM);
        unit_test_send_receive_pokepeek(&tp->e1, &tp->e2);
        unit_test_compare_data(&tp->e1, &tp->e2, COMPARE_ENDS);

        fill_buffer(&tp->e2, RANDOM);
        unit_test_send_receive_pokepeek(&tp->e2, &tp->e1);
        unit_test_compare_data(&tp->e2, &tp->e1, COMPARE_ENDS);
    }

    printf("\n[CLOSE TEST]\n");
    unit_test_close_endpoint(&tp->e1);
    unit_test_close_endpoint(&tp->e2);

    /* call APIs after close. open API should work */
    printf("\n[OPEN TEST (EVENTSERVICE) AFTER CLOSE]\n");

    unit_test_init_event_service();

    unit_test_open_endpoint_eventservice(&tp->e1, 1, NULL, LwSciError_Success);
    unit_test_open_endpoint_eventservice(&tp->e2, 2, NULL, LwSciError_Success);

    unit_test_wait_event_notifier(&tp->e1, &tp->e2);

    unit_test_delete_event_notifier(&tp->e1);
    unit_test_delete_event_notifier(&tp->e2);
    unit_test_close_endpoint(&tp->e1);
    unit_test_close_endpoint(&tp->e2);

    unit_test_deinit_event_service();

    /* call APIs after close. open API should work */
    printf("\n[OPEN TEST (BIND EVENTSERVICE) AFETER CLOSE\n");

    unit_test_init_event_service();

    unit_test_open_endpoint(&tp->e1, 1, NULL, LwSciError_Success);
    unit_test_open_endpoint(&tp->e2, 2, NULL, LwSciError_Success);

    unit_test_bind_eventservice(&tp->e1, 1, LwSciError_Success);
    unit_test_bind_eventservice(&tp->e2, 2, LwSciError_Success);

    unit_test_wait_event_notifier(&tp->e1, &tp->e2);

    unit_test_delete_event_notifier(&tp->e1);
    unit_test_delete_event_notifier(&tp->e2);
    unit_test_close_endpoint(&tp->e1);
    unit_test_close_endpoint(&tp->e2);

    unit_test_deinit_event_service();

    printf("\n[DEINIT TEST]\n");
    unit_test_deinit(tp);

    /* call open after deinit */
    printf("\n[OPEN TEST AFTER DEINIT]\n");
    unit_test_open_endpoint(&tp->e1, 1, NULL, LwSciError_NotInitialized);
    unit_test_open_endpoint(&tp->e2, 2, NULL, LwSciError_NotInitialized);

done:
    printf("\n[TEST RESULT]\n");
    printf(" test count: %d\n", tp->test_cnt);
    printf("error count: %d\n", tp->err_cnt);
    if (tp->err_cnt == 0) {
        printf("[%20s] : test PASSED\n", tp->prgname);
        retval = EXIT_SUCCESS;
    }
    else {
        printf("[%20s] : test FAILED\n", tp->prgname);
        retval = EXIT_FAILURE;
    }

    release_resources(tp);

    return retval;
}

