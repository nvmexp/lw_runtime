/*
 * Copyright (c) 2018-2022, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 */

#ifndef INCLUDED_LWSCIIPC_COMMON_H
#define INCLUDED_LWSCIIPC_COMMON_H

#include <stdbool.h>
#include <pthread.h>
#include <semaphore.h>

#include <sivc.h>
#include <sivc-instance.h>
#include <lwsciipc_internal.h>
#include <lwos_static_analysis.h>
#if defined(LW_IS_TRACER_ENABLED)
#include <tracer.h>
#endif

#include "cheetah-ivc-dev.h"
/* This config file is used for arch/feature migration */
#include "lwsciipc_config.h"

#ifdef LINUX
#define EOK    0
typedef unsigned char      LwBool;
#define LW_TRUE           ((LwBool)(0 == 0))
#define LW_FALSE          ((LwBool)(0 != 0))
#endif /* LINUX */

#define IVC_DEV_NAME        "/dev/ivc"
#define CFG_FILE            "/etc/lwsciipc.cfg"
#define CH_NAME_DT_PROP     "lwsciipc,channel-names"
#define CH_DB_DT_PROP       "lwsciipc,channel-db"

#define BACKEND_ITC_NAME    "INTER_THREAD"
#define BACKEND_IPC_NAME    "INTER_PROCESS"
#define BACKEND_IVC_NAME    "INTER_VM"
#define BACKEND_C2C_NAME    "INTER_CHIP"

#define MAX_BACKEND_NAME      32U
#define MAXBUF                1024U
/* maximum usable endpoint count per process :
 * Restriction 25 of QOS ARR -
 * No process SHALL have more than 100 open channels
 * without disabling kernel preemption.
 * In normal usecase, chid/coid pair (two open channels) is
 * required per endpoint
 */
#define LWSCIIPC_MAX_ENDPOINT 51U   /* 0 index is not used */
#define DEBUG_STR_LEN         256U

#define SIVC_ALIGN_MASK        (SIVC_ALIGN - (size_t)1U)

#define LW_SCI_IPC_EVENT_CONN_MASK  \
    (LW_SCI_IPC_EVENT_CONN_EST | LW_SCI_IPC_EVENT_CONN_RESET)

#ifdef __QNX__
#include "lwsciipc_os_qnx.h"
#endif /* __QNX__ */

#ifndef VERBOSE_LEVEL
/*
 * qnx sys/slog2.h
 * INFO(5), DEBUG1(6), DEBUG2(7)
 */
#define VERBOSE_LEVEL SLOG2_INFO
#endif

/*
 * debug message
 */
#ifdef LWSCIIPC_DEBUG
    #ifdef __QNX__
        #ifdef CONSOLE_DEBUG
            #define LWSCIIPC_DBG_STR(str) \
                printf("!info[L:%d]:%s\n", __LINE__, str)
            #define LWSCIIPC_DBG_STRINT(str, val) \
                printf("!info[L:%d]:%s: %d\n", __LINE__, str, val)
            #define LWSCIIPC_DBG_STR2INT(str, val1, val2) \
                printf("!info[L:%d]:%s: %d, %d\n", __LINE__, str, val1, val2)
            #define LWSCIIPC_DBG_STRUINT(str, val) \
                printf("!info[L:%d]:%s: %u\n", __LINE__, str, val)
            #define LWSCIIPC_DBG_STR2UINT(str, val1, val2) \
                printf("!info[L:%d]:%s: %u, %u\n", __LINE__, str, val1, val2)
            #define LWSCIIPC_DBG_STRULONG(str, val) \
                printf("!info[L:%d]:%s: %lu\n", __LINE__, str, val)
            #define LWSCIIPC_DBG_STR2ULONG(str, val1, val2) \
                printf("!info[L:%d]:%s: %lu, %lu\n", __LINE__, str, val1, val2)
        #else
            #define LWSCIIPC_DBG_STR(str) \
                (void)LwOsDebugPrintStr(LWOS_SLOG_CODE_IPC, VERBOSE_LEVEL, str)
            #define LWSCIIPC_DBG_STRINT(str, val) \
                (void)LwOsDebugPrintStrInt(LWOS_SLOG_CODE_IPC,  \
                VERBOSE_LEVEL, str, val)
            #define LWSCIIPC_DBG_STR2INT(str, val1, val2) \
                (void)LwOsDebugPrintStrWith2Int(LWOS_SLOG_CODE_IPC, \
                VERBOSE_LEVEL, str, val1, val2)
            #define LWSCIIPC_DBG_STRUINT(str, val) \
                (void)LwOsDebugPrintStrUInt(LWOS_SLOG_CODE_IPC, \
                VERBOSE_LEVEL, str, val)
            #define LWSCIIPC_DBG_STR2UINT(str, val1, val2) \
                (void)LwOsDebugPrintStrWith2UInt(LWOS_SLOG_CODE_IPC,    \
                VERBOSE_LEVEL, str, val1, val2)
            #define LWSCIIPC_DBG_STRULONG(str, val) \
                (void)LwOsDebugPrintStrULong(LWOS_SLOG_CODE_IPC,    \
                VERBOSE_LEVEL, str, val)
            #define LWSCIIPC_DBG_STR2ULONG(str, val1, val2) \
                (void)LwOsDebugPrintStrWith2ULong(LWOS_SLOG_CODE_IPC,   \
                VERBOSE_LEVEL, str, val1, val2)
        #endif
    #else /* __QNX__ */
        #define lwsciipc_dbg(fmt, args...) \
            printf("!info[L:%d]:%s: " fmt "\n", __LINE__, __func__, ## args)
        #define LWSCIIPC_DBG_STR(str) \
            printf("!info[L:%d]:%s\n", __LINE__, str)
        #define LWSCIIPC_DBG_STRINT(str, val) \
            printf("!info[L:%d]:%s: %d\n", __LINE__, str, val)
        #define LWSCIIPC_DBG_STR2INT(str, val1, val2) \
            printf("!info[L:%d]:%s: %d, %d\n", __LINE__, str, val1, val2)
        #define LWSCIIPC_DBG_STRUINT(str, val) \
            printf("!info[L:%d]:%s: %u\n", __LINE__, str, val)
        #define LWSCIIPC_DBG_STR2UINT(str, val1, val2) \
            printf("!info[L:%d]:%s: %u, %u\n", __LINE__, str, val1, val2)
        #define LWSCIIPC_DBG_STRULONG(str, val) \
            printf("!info[L:%d]:%s: %lu\n", __LINE__, str, val)
        #define LWSCIIPC_DBG_STR2ULONG(str, val1, val2) \
            printf("!info[L:%d]:%s: %lu, %lu\n", __LINE__, str, val1, val2)
    #endif /* LINUX */
#else
    #define lwsciipc_dbg(fmt, args...)
    #define LWSCIIPC_DBG_STR(str)
    #define LWSCIIPC_DBG_STRINT(str, val)
    #define LWSCIIPC_DBG_STR2INT(str, val1, val2)
    #define LWSCIIPC_DBG_STRUINT(str, val)
    #define LWSCIIPC_DBG_STR2UINT(str, val1, val2)
    #define LWSCIIPC_DBG_STRULONG(str, val)
    #define LWSCIIPC_DBG_STR2ULONG(str, val1, val2)
#endif

#ifdef __QNX__
    #ifdef CONSOLE_DEBUG
        /* ERR */
        #define LWSCIIPC_ERR_STR(str) printf("[L:%d]%s\n", __LINE__, str)
        #define LWSCIIPC_ERR_STRINT(str, val) \
            printf("[L:%d]%s: %d\n", __LINE__, str, val)
        #define LWSCIIPC_ERR_STR2INT(str, val1, val2) \
            printf("[L:%d]%s: %d, %d\n", __LINE__, str, val1, val2)
        #define LWSCIIPC_ERR_STRUINT(str, val) \
            printf("[L:%d]%s: %u\n", __LINE__, str, val)
        #define LWSCIIPC_ERR_STR2UINT(str, val1, val2) \
            printf("[L:%d]%s: %u %u\n", __LINE__, str, val1, val2)
        #define LWSCIIPC_ERR_STRULONG(str, val) \
            printf("[L:%d]%s: %lu\n", __LINE__, str, val)
        #define LWSCIIPC_ERR_STR2ULONG(str, val1, val2) \
            printf("[L:%d]%s: %lu %lu\n", __LINE__, str, val1, val2)
        /* INFO */
        #define LWSCIIPC_INFO_STR(str) printf("[L:%d]%s\n", __LINE__, str)
        #define LWSCIIPC_INFO_STRINT(str, val) \
            printf("[L:%d]%s: %d\n", __LINE__, str, val)
        #define LWSCIIPC_INFO_STR2INT(str, val1, val2) \
            printf("[L:%d]%s: %d, %d\n", __LINE__, str, val1, val2)
        #define LWSCIIPC_INFO_STRUINT(str, val) \
            printf("[L:%d]%s: %u\n", __LINE__, str, val)
        #define LWSCIIPC_INFO_STR2UINT(str, val1, val2) \
            printf("[L:%d]%s: %u %u\n", __LINE__, str, val1, val2)
        #define LWSCIIPC_INFO_STRULONG(str, val) \
            printf("[L:%d]%s: %lu\n", __LINE__, str, val)
        #define LWSCIIPC_INFO_STR2ULONG(str, val1, val2) \
            printf("[L:%d]%s: %lu %lu\n", __LINE__, str, val1, val2)
    #else
        /* ERR */
        #define LWSCIIPC_ERR_STR(str) \
            (void)LwOsDebugPrintStr(LWOS_SLOG_CODE_IPC, SLOG2_ERROR, str)
        #define LWSCIIPC_ERR_STRINT(str, val) \
            (void)LwOsDebugPrintStrInt(LWOS_SLOG_CODE_IPC,  \
            SLOG2_ERROR, str, val)
        #define LWSCIIPC_ERR_STR2INT(str, val1, val2) \
            (void)LwOsDebugPrintStrWith2Int(LWOS_SLOG_CODE_IPC, \
            SLOG2_ERROR, str, val1, val2)
        #define LWSCIIPC_ERR_STRUINT(str, val) \
            (void)LwOsDebugPrintStrUInt(LWOS_SLOG_CODE_IPC, \
            SLOG2_ERROR, str, val)
        #define LWSCIIPC_ERR_STR2UINT(str, val1, val2) \
            (void)LwOsDebugPrintStrWith2UInt(LWOS_SLOG_CODE_IPC,    \
            SLOG2_ERROR, str, val1, val2)
        #define LWSCIIPC_ERR_STRULONG(str, val) \
            (void)LwOsDebugPrintStrULong(LWOS_SLOG_CODE_IPC,    \
            SLOG2_ERROR, str, val)
        #define LWSCIIPC_ERR_STR2ULONG(str, val1, val2) \
            (void)LwOsDebugPrintStrWith2ULong(LWOS_SLOG_CODE_IPC,   \
            SLOG2_ERROR, str, val1, val2)
        /* INFO */
        #define LWSCIIPC_INFO_STR(str) \
            (void)LwOsDebugPrintStr(LWOS_SLOG_CODE_IPC, SLOG2_INFO, str)
        #define LWSCIIPC_INFO_STRINT(str, val) \
            (void)LwOsDebugPrintStrInt(LWOS_SLOG_CODE_IPC,  \
            SLOG2_INFO, str, val)
        #define LWSCIIPC_INFO_STR2INT(str, val1, val2) \
            (void)LwOsDebugPrintStrWith2Int(LWOS_SLOG_CODE_IPC, \
            SLOG2_INFO, str, val1, val2)
        #define LWSCIIPC_INFO_STRUINT(str, val) \
            (void)LwOsDebugPrintStrUInt(LWOS_SLOG_CODE_IPC, \
            SLOG2_INFO, str, val)
        #define LWSCIIPC_INFO_STR2UINT(str, val1, val2) \
            (void)LwOsDebugPrintStrWith2UInt(LWOS_SLOG_CODE_IPC,    \
            SLOG2_INFO, str, val1, val2)
        #define LWSCIIPC_INFO_STRULONG(str, val) \
            (void)LwOsDebugPrintStrULong(LWOS_SLOG_CODE_IPC,    \
            SLOG2_INFO, str, val)
        #define LWSCIIPC_INFO_STR2ULONG(str, val1, val2) \
            (void)LwOsDebugPrintStrWith2ULong(LWOS_SLOG_CODE_IPC,   \
            SLOG2_INFO, str, val1, val2)
    #endif /* CONSOLE_DEBUG */
#else /* __QNX__ */
    /* ERR */
    #define lwsciipc_err(fmt, args...) \
            fprintf(stderr, "!err[L:%d]:%s: " fmt "\n", \
            __LINE__, __func__, ## args)
    #define LWSCIIPC_ERR(str, val)  \
            printf("[L:%d]%s: %d\n", __LINE__, str, val)
    #define LWSCIIPC_ERR_STR(str) printf("%s\n", str)
    #define LWSCIIPC_ERR_STRINT(str, val) \
            printf("%s: %d\n", str, val)
    #define LWSCIIPC_ERR_STR2INT(str, val1, val2) \
            printf("%s: %d, %d\n", str, val1, val2)
    #define LWSCIIPC_ERR_STRUINT(str, val) \
            printf("[L:%d]%s: %u\n", __LINE__, str, val)
    #define LWSCIIPC_ERR_STR2UINT(str, val1, val2) \
            printf("[L:%d]%s: %u %u\n", __LINE__, str, val1, val2)
    #define LWSCIIPC_ERR_STRULONG(str, val) \
            printf("%s: %lu\n", str, val)
    #define LWSCIIPC_ERR_STR2ULONG(str, val1, val2) \
            printf("%s: %lu %lu\n", str, val1, val2)
    /* INFO */
    #define lwsciipc_info(fmt, args...) \
            fprintf(stderr, "!err[L:%d]:%s: " fmt "\n", \
            __LINE__, __func__, ## args)
    #define LWSCIIPC_INFO(str, val)  \
            printf("[L:%d]%s: %d\n", __LINE__, str, val)
    #define LWSCIIPC_INFO_STR(str) printf("%s\n", str)
    #define LWSCIIPC_INFO_STRINT(str, val) \
            printf("%s: %d\n", str, val)
    #define LWSCIIPC_INFO_STR2INT(str, val1, val2) \
            printf("%s: %d, %d\n", str, val1, val2)
    #define LWSCIIPC_INFO_STRUINT(str, val) \
            printf("[L:%d]%s: %u\n", __LINE__, str, val)
    #define LWSCIIPC_INFO_STR2UINT(str, val1, val2) \
            printf("[L:%d]%s: %u %u\n", __LINE__, str, val1, val2)
    #define LWSCIIPC_INFO_STRULONG(str, val) \
            printf("%s: %lu\n", str, val)
    #define LWSCIIPC_INFO_STR2ULONG(str, val1, val2) \
            printf("%s: %lu %lu\n", str, val1, val2)
#endif /* LINUX */

#ifndef container_of
#define container_of(ptr, type, member) \
    (type *)((char *)(ptr) - (char *) &((type *)0)->member)
#endif

enum {
    IDX_BACKEND = 0,    /* backend type */
    IDX_INFO1,    /* VM: ivc qid, PROC: nframes */
    IDX_INFO2,    /* PROC: framesize */
    IDX_ID,    /* endpoint id */
    IDX_MAX
};

enum {
    DT_FMT_TYPE1 = 1, /* inter-thread/process */
    DT_FMT_TYPE2,     /* inter-vm/chip */
};

typedef struct {
    uint64_t index    : 16;
    uint64_t type     : 4;
    uint64_t vmid     : 8;
#ifdef VUID_64BIT
    uint64_t socid    : 28;
    uint64_t reserved : 8;
#else
    uint64_t socid    : 4;
    uint64_t reserved : 32;
#endif
} VuidBitField64;

typedef union {
    LwSciIpcEndpointVuid value;
    VuidBitField64 bit;
} LwSciIpcVUID64;

#define LWSCIIPC_IRQ_ILWALID  0xFFFFFFFFU
#define LWSCIIPC_VMID_ILWALID 0xFFFFFFFFU
#define LWSCIIPC_SHMIDX_ILWALID 0xFFFFFFFFU

struct LwSciIpcConfigEntry {
    char epName[LWSCIIPC_MAX_ENDPOINT_NAME];    /* endpoint name */
//TODO: add LINUX flag
//#if defined(LINUX)
    char devName[LWSCIIPC_MAX_ENDPOINT_NAME];   /* for Linux OS */
//#endif /* LINUX */
    uint32_t backend;    /* backend type */
    uint32_t nFrames;    /* frame count */
    uint32_t frameSize;  /* frame size */
    /* ep id    for inter-Proc/Thread
     * queue id for inter-VM
     * dev id   for inter-Chip */
    uint32_t id;
    /* (QNX) phys mem area address for channel from mem_offset */
    uint64_t chPaddr;
    /* (QNX) phys mem area size for channel from mem_offset */
    uint64_t chPsize;
#ifdef VUID_64BIT
    uint64_t vuid;     /* VM-wide unique id */
    uint64_t peerVuid; /* peer VM-wide unique id */
#else
    uint32_t vuid;     /* VM-wide unique id */
#endif
    /* For Inter-Proc/Thread backend case, we can derive below ids from vuid.
     * peerVuid : vuid^1
     * channelId : (vuid & ~LWSCIIPC_BASE_VUID)>>1
     */
    int32_t gid;    /* group id (signed) */
    uint32_t shmIdx;   /* (QNX) shared memory index */
    uint32_t qsize;    /* queue size for inter-VM (unsigned) */
    uint32_t irq;      /* interrupt number for inter-VM */
    uint32_t peerVmid; /* peer vmid for inter-VM */
    uint32_t rxFirst;  /* flag for rx buffer first */
#if defined(CFGENTRY_CHKSUM)
    uint32_t notiIpaType; /* LWSCIIPC_TRAP_IPA or LWSCIIPC_MSI_IPA */
    uint64_t notiIpa;  /* IO(TRAP/MSI) IPA to used to notify peer in inter-VM */
    uint64_t notiIpaSize; /* size of IO(TRAP/MSI) IPA */
    uint32_t valid; /* valid entry */
    uint32_t checkSum; /* checksum of config entry */
#else
    uint64_t notiIpa;  /* IO(TRAP/MSI) IPA to used to notify peer in inter-VM */
    uint64_t notiIpaSize; /* size of IO(TRAP/MSI) IPA */
    uint32_t notiIpaType; /* LWSCIIPC_TRAP_IPA or LWSCIIPC_MSI_IPA */
    bool valid; /* valid entry */
#endif /* CFGENTRY_CHKSUM */
} __attribute__((aligned(8), packed));

#ifdef __QNX__
static inline struct LwSciIpcConfigEntry *LwSciIpcConfigEntry_addr(
    struct LwSciIpcConfigBlob *blob, uint32_t entry_num)
{
    LWOS_COV_WHITELIST(deviate, LWOS_MISRA(Rule, 11_4), "<QNXBSP>:<lwpu>:<2>:<Bug 200736827>") LWOS_COV_WHITELIST(deviate, LWOS_CERT(INT36_C), "<QNXBSP>:<lwpu>:<1>:<Bug 200736827>")
    return ((struct LwSciIpcConfigEntry *) (((uintptr_t) blob) +
        sizeof(*blob))) + entry_num;
}
#endif /* __QNX__ */

/*
 * This colwerts error code from LwSciIpc and IVC resource manager
 * to LwSciError
 */
static inline LwSciError ResmgrErrnoToLwSciErr(int32_t err)
{
#define MAX_ERROR_NUM 5U
    struct {
        int32_t err;
        LwSciError nerr;
    } err_tbl[MAX_ERROR_NUM + 1] = {
        {EOK,       LwSciError_Success},
        {EPERM,     LwSciError_IlwalidState},
        {EACCES,    LwSciError_IlwalidState},
        {EAGAIN,    LwSciError_TryItAgain},
        {EBUSY,     LwSciError_Busy},
        {-1,        LwSciError_IlwalidState} /* default error */
    };
    uint32_t i;

    for (i = 0U; i < MAX_ERROR_NUM; i++) {
        if (err_tbl[i].err == err) {
            break;
        }
    }
    return err_tbl[i].nerr;
}

#define LW_SCI_IPC_MAGIC 0x5a695063U /* "ZiPc" */

/* 40 Bytes */
struct lwsciipc_internal_handle
{
    uint32_t type;  /* backend type */
    uint32_t magic; /* set LW_SCI_IPC_MAGIC to validate handle */
    struct LwSciIpcConfigEntry *entry;  /* endpoint entry pointer */
    union {
        struct lwsciipc_ivc_handle *ivch; /* inter-VM */
        struct lwsciipc_ipc_handle *ipch; /* inter-thread/process */
#if defined(LINUX) || (LW_IS_SAFETY == 0)
        struct lwsciipc_c2c_handle *c2ch; /* inter-chip */
#endif /* LINUX || (LW_IS_SAFETY == 0) */
        uintptr_t reserved;
    };
    uint64_t index; /* endpoint handle index */
    pthread_mutex_t genMutex;
    pthread_mutex_t wrMutex;
    pthread_mutex_t rdMutex;
} __attribute__((aligned(8)));

/* internal data structure for helper APIs
 * - LwSciIpcOpenEndpointInternal
 * - LwSciIpcWaitEvent
 * need to save OS specific data to minimize effort of user
 */
typedef struct {
    timer_t timer_id;
#ifdef __QNX__
    int32_t chid;    /* private chid for single endpoint per thread */
    int32_t coid;    /* private coid for single endpoint per thread */
#endif /* __QNX__ */
} lwsciipc_event_param_t;

/*
 * OS Abstraction Layer APIs
 */
int32_t lwsciipc_os_mutex_init(void *mutex, void *attr);
int32_t lwsciipc_os_mutex_lock(void *mutex);
int32_t lwsciipc_os_mutex_unlock(void *mutex);
int32_t lwsciipc_os_mutex_destroy(void *mutex);

LwSciError lwsciipc_os_check_pulse_param(int32_t coid, int16_t priority,
    int16_t code);

#ifdef LINUX
int32_t lwsciipc_os_get_endpoint_entry_num(uint32_t *entryNum);
int32_t lwsciipc_os_populate_endpoint_db(struct LwSciIpcConfigEntry **chDB);
#endif /* LINUX */
LwSciError lwsciipc_os_get_config_entry(const char *endpoint,
    struct LwSciIpcConfigEntry **entry);
LwSciError lwsciipc_os_get_config_entry_by_vuid(LwSciIpcEndpointVuid vuid,
    struct LwSciIpcConfigEntry **entry);
int32_t lwsciipc_os_ioctl(int32_t fd, uint32_t request, void *os_args);
void *lwsciipc_os_mmap(void *addr, size_t length, int32_t prot, int32_t flags,
                        int32_t fd, off_t offset, void *os_args);
int32_t lwsciipc_os_munmap(void *addr, size_t length);

LwSciError lwsciipc_os_open_config(void);
LwSciError lwsciipc_os_get_endpoint_access_info(const char *endpoint,
    LwSciIpcEndpointAccessInfo *info);
void lwsciipc_os_close_config(void);

LwSciError lwsciipc_os_get_vmid(uint32_t *vmid);
LwSciError lwsciipc_os_get_socid(uint32_t *socid);

LwSciError lwsciipc_os_get_endpoint_mutex(
    struct LwSciIpcConfigEntry *entry,
    int32_t *mutexfd);
void lwsciipc_os_put_endpoint_mutex(int32_t *fd);
void lwsciipc_os_debug_2strs(const char *str1, const char *str2,
    int32_t ret);
void lwsciipc_os_error_2strs(const char *str1, const char *str2,
    int32_t ret);

#if (LW_IS_SAFETY == 0)
LwSciError lwsciipc_os_poll_event(void *os_args);
LwSciError lwsciipc_os_init_timer(void *os_args);
LwSciError lwsciipc_os_start_timer(timer_t timer_id, uint64_t usecTimeout);
LwSciError lwsciipc_os_stop_timer(timer_t timer_id);
LwSciError lwsciipc_os_deinit_timer(timer_t timer_id);
#endif /* (LW_IS_SAFETY == 0) */

#ifdef LINUX
LwSciError lwsciipc_os_get_vuid(char *ep_name, uint64_t *vuid);
char *lwsciipc_os_ultoa(uint64_t value, char *buffer, size_t size, int radix);
#endif /* LINUX */

#endif /* INCLUDED_LWSCIIPC_COMMON_H */

