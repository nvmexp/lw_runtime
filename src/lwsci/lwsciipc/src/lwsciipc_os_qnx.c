/*
 * Copyright (c) 2019-2022, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <unistd.h>
#include <inttypes.h>
#include <string.h>
#include <errno.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <sys/syspage.h>
#include <sys/neutrino.h>
#include <devctl.h>
#include <search.h>
#include <sched.h>

#include <lwos_static_analysis.h>
#include <lwqnx_common.h>

#include <lwdtcommon.h>
#include <lwscierror.h>

#include <lwsciipc_init.h>
#include <io-lwsciipc.h>

#include "lwsciipc_common.h"
#include "lwsciipc_os_error.h"
#include "lwsciipc_ivc.h"
#include "lwsciipc_ipc.h"
#include "lwsciipc_log.h"

#include "cheetah-ivc-dev.h"

#define MAX_SHM_PATH_LEN 48U
#define MAX_ID_LEN 8U

/* LwSciIpc configuration blob */
struct CfgBlobShmInfo {
    uint32_t initDone; /* LwBoolTrue/LwBoolFalse */
    int32_t shmfd;
    size_t shmSz;
    uint32_t entryCount;
    uint32_t vmid; /* current vmid */
    uint32_t socid; /* current socid */
    void *shmptr;
};
/* LwSciIpc config blob data structure DB */
static struct CfgBlobShmInfo s_cfgInfo;

int32_t lwsciipc_os_mutex_init(void *mutex, void *attr)
{
#ifdef LWSCIIPC_USE_MUTEX
   pthread_mutex_t *m = mutex;
   pthread_mutexattr_t *a = attr;

   return pthread_mutex_init(m, a);
#else
   return 0;
#endif
}

int32_t lwsciipc_os_mutex_lock(void *mutex)
{
#ifdef LWSCIIPC_USE_MUTEX
   pthread_mutex_t *m = mutex;

   return pthread_mutex_lock(m);
#else
   return 0;
#endif
}

int32_t lwsciipc_os_mutex_unlock(void *mutex)
{
#ifdef LWSCIIPC_USE_MUTEX
   pthread_mutex_t *m = mutex;

   return pthread_mutex_unlock(m);
#else
   return 0;
#endif
}

int32_t lwsciipc_os_mutex_destroy(void *mutex)
{
#ifdef LWSCIIPC_USE_MUTEX
   pthread_mutex_t *m = mutex;

   return pthread_mutex_destroy(m);
#else
   return 0;
#endif
}

/*
 * Returns
 * address of the mapped-in object : Success
 * MAP_FAILED : Failure (errno is set)
 */
void *lwsciipc_os_mmap(void *addr, size_t length, int32_t prot, int32_t flags,
            int32_t fd, off_t offset, void *os_args)
{
    struct LwSciIpcConfigEntry *entry = (struct LwSciIpcConfigEntry *)os_args;
    LWOS_COV_WHITELIST(deviate, LWOS_MISRA(Rule, 11_6), "<QNXBSP>:<qnx_asil_header>:<1>:<TID-371>")
    void *mapaddr = MAP_FAILED;
    uint32_t u32flags;
    int32_t err;

    /* unused variables */
    (void)fd;
    (void)offset;

    /* no overflow since flags is 0x1 (MAP_SHARED) in
     * lwsciipc_ivc_init_data().
     */
    u32flags = CastS32toU32WithExit(flags);
    LWOS_COV_WHITELIST(deviate, LWOS_MISRA(Directive, 4_12), "<QNXBSP>:<lwpu>:<1>:<TID-385>") LWOS_COV_WHITELIST(deviate, LWOS_CERT(INT31_C), "<QNXBSP>:<qnx_asil_header>:<1>:<TID-448>")
    mapaddr = mmap(addr, length, prot,
        /* no overflow since flags is 0x10001 */
        CastU32toS32WithExit(u32flags|LW_MAP_PHYS),
        NOFD, (off_t)entry->chPaddr);
    err = errno;
    LWOS_COV_WHITELIST(deviate, LWOS_MISRA(Rule, 11_6), "<QNXBSP>:<qnx_asil_header>:<1>:<TID-371>") LWOS_COV_WHITELIST(deviate, LWOS_CERT(INT36_C), "<QNXBSP>:<lwpu>:<1>:<Bug 200736827>")
    if (mapaddr == MAP_FAILED) {
        LWSCIIPC_ERR_STRINT("error: lwsciipc_os_mmap: mmap: ret", err);
        LWSCIIPC_DBG_STR2ULONG("lwsciipc_os_mmap: addr, size",
            entry->chPaddr, entry->chPsize);
    }

    return mapaddr;
}

/*
 * Returns
 * -1              : Failure (errno is set)
 * any other value : Success
 */
int32_t lwsciipc_os_munmap(void *addr, size_t length)
{
    return munmap(addr, length);
}

/*
 * validate pulse parameters of LwSciIpcSetQnxPulseParam()
 */
LwSciError lwsciipc_os_check_pulse_param(int32_t coid, int16_t priority,
    int16_t code)
{
    struct _server_info info;
    LwSciError ret = LwSciError_Success;
    int32_t sched_policy;
    int32_t err;

    /* validate coid */
    err = ConnectServerInfo_r(getpid(), coid, &info);
    if ((err < 0) || (info.coid != coid)) {
        LWSCIIPC_ERR_STR("error: lwsciipc_os_check_pulse_param: "
            "coid parameter error");
        ret = LwSciError_BadParameter;
    }

    /* validate priority */
    sched_policy = sched_getscheduler(0); /* get policy of lwrr process */
    if ((priority != SIGEV_PULSE_PRIO_INHERIT) &&
        ((priority < sched_get_priority_min(sched_policy)) ||
        (priority > sched_get_priority_max(sched_policy)))) {
        LWSCIIPC_ERR_STR("error: lwsciipc_os_check_pulse_param: "
            "pulse priority is out of range");
        ret = LwSciError_BadParameter;
    }

    /* validate code */
    if ((code < _PULSE_CODE_MINAVAIL) || (code > _PULSE_CODE_MAXAVAIL)) {
        LWSCIIPC_ERR_STR("error: lwsciipc_os_check_pulse_param: "
            "pulse code is out of range");
        ret = LwSciError_BadParameter;
    }

    return ret;
}

/* fill outstr with head string and id
 * get "/dev/shmem/LwSciIpcConfig##" SHM node name
 * get "/dev/ivc##" IVC dev node name
 */
LwSciError lwsciipc_os_get_node_name(const char *head, uint32_t id, size_t len,
    char *outstr)
{
#undef LIB_FUNC
#define LIB_FUNC "lwsciipc_os_get_node_name: "
    char id_str[MAX_ID_LEN] = {0};
    int32_t err;
    size_t retval;
    char *str = NULL;
    /* use default success to remove else routine */
    LwSciError ret = LwSciError_Success;

    /* clear buf */
    LWOS_COV_WHITELIST(deviate, LWOS_MISRA(Rule, 1_4), "<QNXBSP>:<lwpu>:<1>:<TID-1431>")
    err = memset_s(outstr, len, 0, len);
    report_os_errto(err, "memset_s", fail);

    /* add head string */
    retval = strlcat(outstr, head, len);
    report_trunc_errto(retval, len, "strlcat(1)", fail);

    str = utoa(id, id_str, 10); /* decimal to string */
    if (str == NULL) {
        LWSCIIPC_ERR_STR("error: " LIB_FUNC "ltoa");
        ret = LwSciError_IlwalidState;
        goto fail;
    }

    /* head string + id */
    retval = strlcat(outstr, id_str, len);
    report_trunc_errto(retval, len, "strlcat(2)", fail);

fail:
    return ret;
}

/*====================================================================
 * ConfigBlob Access API
 *====================================================================
 */

/**
 * Open LwSciIpc configuration blob shared memory to search endpoint entry
 *
 * @return ::LwSciError, the completion code of the operation:
 * - ::LwSciError_Success         Indicates a successful operation.
 * - ::LwSciError_NotPermitted    Indicates opening configuration blob is failed
 */
LwSciError lwsciipc_os_open_config(void)
{
#undef LIB_FUNC
#define LIB_FUNC "lwsciipc_os_open_config: "
    int32_t shmfd = 0;
    size_t shmSz = 0;
    LWOS_COV_WHITELIST(deviate, LWOS_MISRA(Rule, 11_6), "<QNXBSP>:<qnx_asil_header>:<1>:<TID-371>")
    void *shmptr = MAP_FAILED;
#ifdef CONFIGBLOB_V2
    struct LwSciIpcConfigBlob2 *blob;
#else
    struct LwSciIpcConfigBlob *blob;
#endif
    struct stat buf;
    int32_t err;
    LwSciError ret = LwSciError_NotPermitted;

    LWSCIIPC_DBG_STR(LIB_FUNC "opening config blob");
    shmfd = shm_open(CONFIG_SHM, O_RDONLY, LW_S_IRUSR | LW_S_IRGRP);
    if (shmfd < 0) {
        LWSCIIPC_ERR_STRINT("error: " LIB_FUNC "shm_open err: ", errno);
        goto fail;
    }
    s_cfgInfo.shmfd = shmfd;

    err = fstat(shmfd, &buf);
    if (err == -1) {
        LWSCIIPC_ERR_STRINT("error: " LIB_FUNC "fstat err: ", errno);
        goto fail;
    }
    /* no exit since size can't be negative */
    shmSz = CastS64toU64WithExit(buf.st_size);
    LWSCIIPC_DBG_STRULONG(LIB_FUNC "shmSz", shmSz);
    s_cfgInfo.shmSz = shmSz;

    err = shm_ctl(shmfd,
        /* no overflow since flag is 0x3U */
        CastU32toS32WithExit(LW_SHMCTL_ANON|LW_SHMCTL_PHYS),
        0, shmSz);
    if (err < 0) {
        LWSCIIPC_ERR_STRINT("error: " LIB_FUNC "shm_ctl err: ", errno);
        goto fail;
    }

    LWOS_COV_WHITELIST(deviate, LWOS_MISRA(Directive, 4_12), "<QNXBSP>:<lwpu>:<1>:<TID-385>")
    shmptr = mmap(NULL, shmSz, PROT_READ, MAP_SHARED, shmfd, 0);
    LWOS_COV_WHITELIST(deviate, LWOS_MISRA(Rule, 11_6), "<QNXBSP>:<qnx_asil_header>:<1>:<TID-371>") LWOS_COV_WHITELIST(deviate, LWOS_CERT(INT36_C), "<QNXBSP>:<lwpu>:<1>:<Bug 200736827>")
    if (shmptr == MAP_FAILED) {
        LWSCIIPC_ERR_STRINT("error: " LIB_FUNC "mmap err: ", errno);
        goto fail;
    }
    s_cfgInfo.shmptr = shmptr;
    blob = (struct LwSciIpcConfigBlob *)s_cfgInfo.shmptr;
    s_cfgInfo.vmid = blob->vmid;
#ifdef CONFIGBLOB_V2
    s_cfgInfo.socid = blob->socid;
#else
    s_cfgInfo.socid = 0; /* hardcoding for old structure */
#endif
    LWSCIIPC_DBG_STR2INT(LIB_FUNC "socid, vmid: ",
        s_cfgInfo.socid, s_cfgInfo.vmid);

    s_cfgInfo.initDone = LwBoolTrue;

    ret = LwSciError_Success;

fail:
    if (ret != LwSciError_Success) {
        lwsciipc_os_close_config();
    }

    return ret;
}

LwSciError lwsciipc_os_get_vmid(uint32_t *vmid)
{
    LwSciError ret = LwSciError_NotInitialized;

    if (s_cfgInfo.initDone != LwBoolTrue) {
        goto done;
    }
    else {
        *vmid = s_cfgInfo.vmid;
        ret = LwSciError_Success;
    }

done:
    return ret;
}

LwSciError lwsciipc_os_get_socid(uint32_t *socid)
{
    LwSciError ret = LwSciError_NotInitialized;

    if (s_cfgInfo.initDone != LwBoolTrue) {
        goto done;
    }
    else {
        *socid = s_cfgInfo.socid;
        ret = LwSciError_Success;
    }

done:
    return ret;
}

#ifdef CFGENTRY_CHKSUM
/*
 * data size must be 4Bytes aligned
 */
static uint32_t lwsciipc_os_get_checksum(const void *data, uint32_t len)
{
    const uint32_t *ptr = (const uint32_t *)data;
    uint32_t wlen = len >> 2;
    uint32_t cnt;
    uint64_t sum = 0UL;

    for (cnt = 0; cnt < wlen; cnt++) {
        sum += (uint64_t)ptr[cnt];
    }
    sum = ~sum;

    return (uint32_t)(sum & 0xFFFFFFFFUL);
}

static uint32_t lwsciipc_os_validate_config_entry(
    struct LwSciIpcConfigEntry *entry)
{
    uint32_t chksum;
    uint32_t ret;

    chksum = lwsciipc_os_get_checksum(entry,
        sizeof(struct LwSciIpcConfigEntry)-4U);
    if (chksum == entry->checkSum) {
        ret = LwBoolTrue;
    }
    else {
        ret = LwBoolFalse;
    }

    return ret;
}
#endif /* CFGENTRY_CHKSUM */

/**
 * It finds endpoint with given name by iterating configuration blob and
 * matching endpoint name.

 * @param[in]  endpoint The name of the LwSciIpc endpoint to open.
 * @param[out] entry    Entry structure handle
 *
 * @return ::LwSciError, the completion code of the operation:
 * - ::LwSciError_Success            Indicates a successful operation.
 * - ::LwSciError_NotInitialized     Indicates LwSciIpc is uninitialized.
 * - ::LwSciError_NoSuchEntry        Indicates the @a endpoint was not found.
 */
LwSciError lwsciipc_os_get_config_entry(const char *endpoint,
    struct LwSciIpcConfigEntry **entry)
{
#undef LIB_FUNC
#define LIB_FUNC "lwsciipc_os_get_config_entry: "
    struct LwSciIpcConfigBlob *blob;
    struct LwSciIpcConfigEntry *epEntry;
    char name[LWSCIIPC_MAX_ENDPOINT_NAME] = {0,}; 
    LwSciError ret;
    uint32_t i;

    if (s_cfgInfo.initDone != LwBoolTrue) {
        ret = LwSciError_NotInitialized;
        goto done;
    }

    if (strlcpy(name, endpoint, sizeof(name)) >= sizeof(name)) {
        lwsciipc_os_error_2strs("error: " LIB_FUNC "strlcpy", endpoint, 0);
        ret = LwSciError_BadParameter;
        goto done;
    }

    blob = (struct LwSciIpcConfigBlob *)s_cfgInfo.shmptr;
    for (i=0; i < blob->entryCount; i++) {
        epEntry = LwSciIpcConfigEntry_addr(blob, i);
        if (strncmp(epEntry->epName, name, sizeof(name)) == 0) {
            *entry = epEntry;
#ifdef CFGENTRY_CHKSUM
            if (lwsciipc_os_validate_config_entry(epEntry) == LwBoolTrue) {
                ret = LwSciError_Success;
                goto done;
            }
            else {
                lwsciipc_os_error_2strs("error: " LIB_FUNC "checksum error",
                    endpoint, epEntry->checkSum);
                break;
            }
#else
            ret = LwSciError_Success;
            goto done;
#endif /* CFGENTRY_CHKSUM */
        }
    }

    ret = LwSciError_NoSuchEntry;

done:
    return ret;
}

LwSciError lwsciipc_os_get_config_entry_by_vuid(LwSciIpcEndpointVuid vuid,
    struct LwSciIpcConfigEntry **entry)
{
    struct LwSciIpcConfigBlob *blob;
    struct LwSciIpcConfigEntry *epEntry;
    LwSciError ret;
    uint32_t i;

    if (s_cfgInfo.initDone != LwBoolTrue) {
        ret = LwSciError_NotInitialized;
        goto done;
    }

    blob = (struct LwSciIpcConfigBlob *)s_cfgInfo.shmptr;
    for (i=0; i < blob->entryCount; i++) {
        epEntry = LwSciIpcConfigEntry_addr(blob, i);
        if (vuid == (LwSciIpcEndpointVuid)epEntry->vuid) {
            *entry = epEntry;
#ifdef CFGENTRY_CHKSUM
            if (lwsciipc_os_validate_config_entry(epEntry) == LwBoolTrue) {
                ret = LwSciError_Success;
                goto done;
            }
            else {
                lwsciipc_os_error_2strs("error: " LIB_FUNC "checksum error",
                    epEntry->epName, epEntry->checkSum);
                break;
            }
#else
            ret = LwSciError_Success;
            goto done;
#endif /* CFGENTRY_CHKSUM */
        }
    }

    ret = LwSciError_NoSuchEntry;

done:
    return ret;
}

LwSciError lwsciipc_os_get_endpoint_access_info(const char *endpoint,
    LwSciIpcEndpointAccessInfo *info)
{
    struct LwSciIpcConfigEntry *entry;
    LwSciError ret;
    bool flag;

    if ((info == NULL) || (endpoint == NULL)) {
        ret = LwSciError_BadParameter;
        goto done;
    }

    /* only root user can access this API */
    if (getuid() != 0) {
        ret = LwSciError_NotPermitted;
        goto done;
    }

    ret = lwsciipc_os_get_config_entry(endpoint, &entry);
    if (ret != LwSciError_Success) {
        goto done;
    }

    info->id = entry->id;
    info->gid = entry->gid;
    info->vuid = entry->vuid;

    info->notiIpa = 0UL;
    info->notiIpaSize = 0UL;
    info->notiIpaType = LWSCIIPC_ILWALID_IPA;

    /* GID-SWELREQ-4098302:
     * Set Interrupt range for PROCMGR_AID_INTERRUPT and
     * PROCMGR_AID_INTERRUPTEVENT
     * GID-SWELREQ-4098303:
     * Set physical memory range for PROCMGR_AID_MEM_PHYS
     */
    switch(entry->backend) {
        case LWSCIIPC_BACKEND_IVC :
            info->phyAddr = entry->chPaddr;
            info->phySize = entry->chPsize;
            flag = CastU32toS32(entry->irq, &info->irq);
            if (flag == false) {
                LWSCIIPC_ERR_STRUINT(
                    "error: lwsciipc_os_get_endpoint_access_info: irq",
                    entry->irq);
                ret = LwSciError_IlwalidState;
            }
            info->notiIpa = entry->notiIpa;
            info->notiIpaSize = entry->notiIpaSize;
            info->notiIpaType = entry->notiIpaType;
            break;
        case LWSCIIPC_BACKEND_ITC :
        case LWSCIIPC_BACKEND_IPC :
            info->phyAddr = entry->chPaddr;
            info->phySize = entry->chPsize;
            info->irq = -1;
            break;
#if (LW_IS_SAFETY == 0)
        case LWSCIIPC_BACKEND_C2C :
            info->phyAddr = 0;
            info->phySize = 0;
            info->irq = -1;
            break;
#endif /* (LW_IS_SAFETY == 0) */
        default :
            LWSCIIPC_ERR_STRUINT(
                "error: lwsciipc_os_get_endpoint_access_info: "
                "Unsupported backend type: backend", entry->backend);
            ret = LwSciError_NotSupported;
            break;
    }

    if (ret == LwSciError_Success) {
        /* invalid backend is verified in switch() */
        info->backend = CastU32toS32WithExit(entry->backend);
    }

done:
    return ret;
}

/**
 * Close LwSciIpc configuration blob shared memory.
 */
void lwsciipc_os_close_config(void)
{
    int32_t err;

    LWOS_COV_WHITELIST(deviate, LWOS_MISRA(Rule, 11_6), "<QNXBSP>:<qnx_asil_header>:<1>:<TID-371>") LWOS_COV_WHITELIST(deviate, LWOS_CERT(INT36_C), "<QNXBSP>:<lwpu>:<1>:<Bug 200736827>")
    if ((s_cfgInfo.shmptr != MAP_FAILED) && (s_cfgInfo.shmSz != 0UL)) {
        err = munmap(s_cfgInfo.shmptr, s_cfgInfo.shmSz);
        if (EOK != err) {
            LWSCIIPC_ERR_STRINT("error: lwsciipc_os_close_config: munmap",
                errno);
        }
        s_cfgInfo.shmptr = NULL;
        s_cfgInfo.shmSz = 0UL;
    }

    if (s_cfgInfo.shmfd != 0) {
        err = close(s_cfgInfo.shmfd);
        if (EOK != err) {
            LWSCIIPC_ERR_STRINT("error: lwsciipc_os_close_config: close",
                errno);
        }
        s_cfgInfo.shmfd = 0;
    }
    s_cfgInfo.initDone = LwBoolFalse;
}

/*====================================================================
 * LwSciIpc ResMgr Mutex API
 *====================================================================
 */

LwSciError lwsciipc_os_get_endpoint_mutex(
    struct LwSciIpcConfigEntry *entry,
    int32_t *mutexfd)
{
    LwSciIpcGetMutex msg;
    int32_t err;
    int32_t fd;
    LwSciError ret;

    LWOS_COV_WHITELIST(deviate, LWOS_CERT(FIO32_C), "<QNXBSP>:<lwpu>:<1>:<Bug 200736827>")
    fd = open(LWSCIIPC_MUTEXDEV, O_RDWR);
    if (fd == -1) {
        LWSCIIPC_ERR_STRINT("error: lwsciipc_os_get_endpoint_mutex: "
            "Failed to open /dev/lwsciipc_mutex: ", errno);
        ret = LwSciError_NotPermitted;
        *mutexfd = 0;
        goto fail;
    }
    *mutexfd = fd;

    msg.vuid = entry->vuid;
    msg.gid = entry->gid;

    err = devctl(fd,
            LwColwertFlagUInt32toInt32(DCMD_LWSCIIPC_GET_MUTEX),
            &msg, sizeof(msg), NULL);
    ret = ResmgrErrnoToLwSciErr(err);
    if (ret != LwSciError_Success) {
        LWSCIIPC_ERR_STRINT("error: lwsciipc_os_get_endpoint_mutex: "
            "Failed to get endpoint mutex: ret", (int32_t)ret);
    }

fail:
    return ret;
}

void lwsciipc_os_put_endpoint_mutex(int32_t *fd)
{
    int32_t err;

    if (*fd != 0) {
        err = close(*fd);
        if (EOK != err) {
            LWSCIIPC_ERR_STRINT("error: lwsciipc_os_put_endpoint_mutex: close",
                errno);
        }
        *fd = 0;
    }
}

#if (LW_IS_SAFETY == 0)
/*====================================================================
 * Event polling API
 *====================================================================
 */
LwSciError lwsciipc_os_poll_event(void *os_args)
{
    lwsciipc_event_param_t *param = (lwsciipc_event_param_t *)os_args;
    struct _pulse pulse;
    LwSciError ret;
    int32_t err;

    if ((param == NULL) || (param->chid == 0)) {
        LWSCIIPC_ERR_STR("error: lwsciipc_os_poll_event: Invalid parameter");
        ret = LwSciError_BadParameter;
        goto fail;
    }

    LWSCIIPC_DBG_STRINT("lwsciipc_os_poll_event: chid", param->chid);
    err = MsgReceivePulse_r(param->chid, &pulse, sizeof(pulse), NULL);
    if (err < 0) {
        ret = ErrnoToLwSciErr(err);
        LWSCIIPC_ERR_STRINT("error: lwsciipc_os_poll_event: "
            "MsgReceivePulse_r: ret", err);
        goto fail;
    }

    LWSCIIPC_DBG_STRINT("lwsciipc_os_poll_event: pulse code", pulse.code);
    switch (pulse.code) {
        case PRIV_EVENT_PULSE_CODE :
            ret = LwSciError_Success;
            break;
        case PRIV_TIMEOUT_PULSE_CODE :
            ret = LwSciError_Timeout;
            break;
        default :
            LWSCIIPC_ERR_STRINT("error: lwsciipc_os_poll_event: "
                "Invalid pulse code: code", pulse.code);
            ret = LwSciError_NoDesiredMessage;
            break;
    }

fail:
    return ret;
}

/*====================================================================
 * Timer APIs
 *====================================================================
 */

LwSciError lwsciipc_os_init_timer(void *os_args)
{
    lwsciipc_event_param_t *param = (lwsciipc_event_param_t *)os_args;
    LwSciError ret;
    int32_t id;
    struct sigevent timerEvent;

    if ((param == NULL) || (param->coid == 0)) {
        LWSCIIPC_ERR_STR("error: lwsciipc_os_init_timer: Invalid parameter");
        ret = LwSciError_BadParameter;
        goto fail;
    }

    SIGEV_PULSE_INIT(&timerEvent, param->coid, SIGEV_PULSE_PRIO_INHERIT,
        PRIV_TIMEOUT_PULSE_CODE, NULL);
    id = TimerCreate_r(CLOCK_MONOTONIC, &timerEvent);
    if (id < 0) {
        LWSCIIPC_ERR_STRINT("error: lwsciipc_os_init_timer: "
            "TimerCreate_r: id", id);
        ret = ErrnoToLwSciErr(id);
        param->timer_id = -1;
    }
    else {
        param->timer_id = id;
        ret = LwSciError_Success;
    }
    LWSCIIPC_DBG_STR2INT("lwsciipc_os_init_timer: coid, timer_id",
        param->coid, param->timer_id);

fail:
    return ret;
}

LwSciError lwsciipc_os_start_timer(timer_t timer_id, uint64_t usecTimeout)
{
    LwSciError ret;
    int32_t err;
    struct _itimer itime;

    LWSCIIPC_DBG_STRINT("lwsciipc_os_start_timer: timer_id", timer_id);
    LWSCIIPC_DBG_STRULONG("lwsciipc_os_start_timer: timeout", usecTimeout);

    if (timer_id == -1) {
        ret = LwSciError_BadParameter;
        goto fail;
    }

    itime.nsec = (usecTimeout * 1000U);
    itime.interval_nsec = 0;

    err = TimerSettime_r(timer_id, 0, &itime, NULL);
    ret = ErrnoToLwSciErr(err);
    if (ret != LwSciError_Success) {
        LWSCIIPC_ERR_STRINT("error: lwsciipc_os_start_timer: "
            "TimerSettime_r: ret", err);
    }

fail:
    return ret;
}

LwSciError lwsciipc_os_stop_timer(timer_t timer_id)
{
    LwSciError ret;
    int32_t err;
    struct _itimer itime;

    LWSCIIPC_DBG_STRINT("lwsciipc_os_stop_timer: timer_id", timer_id);

    if (timer_id == -1) {
        ret = LwSciError_BadParameter;
        goto fail;
    }

    itime.nsec = 0;
    itime.interval_nsec = 0;

    err = TimerSettime_r(timer_id, 0, &itime, NULL);
    ret = ErrnoToLwSciErr(err);
    if (ret != LwSciError_Success) {
        LWSCIIPC_ERR_STRINT("error: lwsciipc_os_stop_timer: "
            "TimerSettime_r: ret", err);
    }

fail:
    return ret;
}

LwSciError lwsciipc_os_deinit_timer(timer_t timer_id)
{
    LwSciError ret;
    int32_t err;

    if (timer_id == -1) {
        ret = LwSciError_BadParameter;
        goto fail;
    }

    err = TimerDestroy_r(timer_id);
    ret = ErrnoToLwSciErr(err);
    if (ret != LwSciError_Success) {
        LWSCIIPC_ERR_STRINT("error: lwsciipc_os_deinit_timer: "
            "TimerDestroy_r: ret", err);
    }

fail:
    return ret;
}
#endif /* (LW_IS_SAFETY == 0) */

/*====================================================================
 * Tools API
 *====================================================================
 */

#if (LW_IS_SAFETY == 0)
/*
 * it colwerts from the number of clock cycles (ClockCycles()) to
 * human readable usec unit time.
 * it can be used for profiling any process or function.
 */
uint64_t lwsciipc_qnx_get_us(uint64_t time)
{
    static uint64_t cps = UINT64_C(0);
    uint64_t usec;

    if (cps == UINT64_C(0)) {
        cps = SYSPAGE_ENTRY(qtime)->cycles_per_sec;
    }
    usec = (time * UINT64_C(1000000)) / cps;

    return usec;
}
#endif /* (LW_IS_SAFETY == 0) */

/*
 * debug log message w/ two strings and one value
 */
void lwsciipc_os_debug_2strs(const char *str1, const char *str2,
    int32_t ret)
{
    char errStr[DEBUG_STR_LEN];

#ifndef LWSCIIPC_DEBUG
    (void)ret;
#endif

    /* doesn't handle internal error since this is logging function */
    (void)strlcpy(errStr, str1, sizeof(errStr));
    if (str2 != NULL) {
        /* doesn't handle internal error since this is logging function */
        (void)strlcat(errStr, str2, sizeof(errStr));
    }
    LWSCIIPC_DBG_STRINT(errStr, ret);
}

/*
 * error log message w/ two strings and one value
 */
void lwsciipc_os_error_2strs(const char *str1, const char *str2,
    int32_t ret)
{
    char errStr[DEBUG_STR_LEN];

    /* doesn't handle internal error since this is logging function */
    (void)strlcpy(errStr, str1, sizeof(errStr));
    if (str2 != NULL) {
        /* doesn't handle internal error since this is logging function */
        (void)strlcat(errStr, str2, sizeof(errStr));
    }
    LWSCIIPC_ERR_STRINT(errStr, ret);
}

