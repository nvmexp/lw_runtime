/*
 * Copyright (c) 2018-2021, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 */

/* This file is for QNX OS only */

#include <stdio.h>
#include <stddef.h>
#include <stdbool.h>
#include <string.h>
#include <errno.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <stdlib.h>

#include <sys/neutrino.h>
#include <sys/procmgr.h>
#include <inttypes.h>
#include <devctl.h>

#include <lwos_static_analysis.h>
#include <lwqnx_common.h>
#include <io-lwsciipc.h>

#include "lwsciipc_common.h"
#include "lwsciipc_os_error.h"
#include "lwsciipc_ipc.h"
#include "lwsciipc_log.h"

#define DEBUG_STATE 0
#define MAX_SHM_PATH_LEN 48U

/* internal function definitions */
static LwSciError lwsciipc_ipc_open_pulse_device(
    struct lwsciipc_ipc_handle *ipch);
static void lwsciipc_ipc_close_pulse_device(struct lwsciipc_ipc_handle *ipch);
static LwSciError lwsciipc_ipc_pulse_sel_channel(int32_t fd,
    LwSciIpcEndpointVuid vuid, int32_t gid);
static LwSciError lwsciipc_ipc_pulse_phase1_push(int32_t fd,
    struct sigevent progressEvent, struct sigevent resetFlagEvent,
    int32_t svrChid);
static LwSciError lwsciipc_ipc_pulse_phase1_pull(int32_t fd,
    pid_t *remotePid, int32_t *remoteChid);
static LwSciError lwsciipc_ipc_pulse_phase2_push(int32_t fd,
    int32_t scoid, struct sigevent regEvent);
static LwSciError lwsciipc_ipc_pulse_phase2_pull(int32_t fd,
    int32_t *svrScoid, struct sigevent *svrEvent);

static void lwsciipc_ipc_notify(struct sivc_queue *sivc);
static LwSciError lwsciipc_ipc_release_event_handler(
    struct lwsciipc_ipc_handle *ipch);
static LwSciError lwsciipc_ipc_pulse_rm_reset_state(
    struct lwsciipc_ipc_handle *ipch);
static LwSciError lwsciipc_ipc_pulse_rm_update_state(
    struct lwsciipc_ipc_handle *ipch);
static LwSciError lwsciipc_ipc_register_events(
    struct lwsciipc_ipc_handle *ipch);
static LwSciError lwsciipc_ipc_unregister_events(
    struct lwsciipc_ipc_handle *ipch);
static void lwsciipc_ipc_close_endpoint_internal(
    struct lwsciipc_ipc_handle *ipch);
static LwSciError lwsciipc_ipc_check_channel_data(
    struct LwSciIpcConfigEntry *entry);

/*====================================================================
 * LwSciIpc ResMgr Pulse Connection API
 *====================================================================
 */
static LwSciError lwsciipc_ipc_open_pulse_device(
    struct lwsciipc_ipc_handle *ipch)
{
    int32_t fd;
    LwSciError ret;

    LWOS_COV_WHITELIST(deviate, LWOS_CERT(FIO32_C), "<QNXBSP>:<lwpu>:<1>:<Bug 200736827>")
    fd = open(LWSCIIPC_PULSEDEV, O_RDWR);
    if (fd == -1) {
        LWSCIIPC_ERR_STRINT(
            "error: lwsciipc_ipc_open_pulse_device: "
            "Failed to open /dev/lwsciipc_pulse: ret", errno);
        ret = LwSciError_NotPermitted;
        goto fail;
    }

    ipch->pulsefd = fd;

    ret = LwSciError_Success;

fail:
    return ret;
}

static void lwsciipc_ipc_close_pulse_device(struct lwsciipc_ipc_handle *ipch)
{
#undef LIB_FUNC
#define LIB_FUNC "lwsciipc_ipc_close_pulse_device: "
    int32_t err;

    if (ipch->pulsefd != 0) {
        err = close(ipch->pulsefd);
        if (EOK != err) {
            LWSCIIPC_ERR_STRINT("error: " LIB_FUNC "close", errno);
        }
    }
}

static LwSciError lwsciipc_ipc_pulse_sel_channel(int32_t fd,
    LwSciIpcEndpointVuid vuid, int32_t gid)
{
#undef LIB_FUNC
#define LIB_FUNC "lwsciipc_ipc_pulse_sel_channel: "
    LwSciIpcPulseSelCh msg;
    int32_t err;
    LwSciError ret;
#ifndef VUID_64BIT
    bool flag;
#endif

#ifdef VUID_64BIT
    msg.vuid = vuid;
#else
    flag = CastU64toU32(vuid, &msg.vuid);
    if (flag == false) {
        LWSCIIPC_ERR_STRULONG("error: " LIB_FUNC, vuid);
        ret = LwSciError_IlwalidState;
        goto fail;
    }
#endif /* VUID_64BIT */
    msg.gid = gid;

    /*
     * EOK, EAGAIN, EBADF, EINTR, EILWAL, EIO, EFAULT, ENOTTY, ENXIO
     * EOVERFLOW, EPERM
     */
    err = devctl(fd,
            LwColwertFlagUInt32toInt32(DCMD_LWSCIIPC_SEL_CHANNEL),
            &msg, sizeof(msg), NULL);
    ret = ResmgrErrnoToLwSciErr(err);

#ifndef VUID_64BIT
fail:
#endif
    return ret;
}

static LwSciError lwsciipc_ipc_pulse_phase1_push(int32_t fd,
    struct sigevent progressEvent, struct sigevent resetFlagEvent,
    int32_t svrChid)
{
    LwSciIpcPulsePh1Push msg;
    int32_t err;
    LwSciError ret;

    msg.progressEvent = progressEvent;
    msg.resetFlagEvent = resetFlagEvent;
    msg.svrChid = svrChid;
    /*
     * EOK, EAGAIN, EBADF, EINTR, EILWAL, EIO, EFAULT, ENOTTY, ENXIO
     * EOVERFLOW, EPERM
     */
    err = devctl(fd,
            LwColwertFlagUInt32toInt32(DCMD_LWSCIIPC_PHASE1_PUSH),
            &msg, sizeof(msg), NULL);
    ret = ResmgrErrnoToLwSciErr(err);

    return ret;
}

static LwSciError lwsciipc_ipc_pulse_phase1_pull(int32_t fd,
    pid_t *remotePid, int32_t *remoteChid)
{
    LwSciIpcPulsePh1Pull msg;
    int32_t err;
    LwSciError ret;

    /*
     * EOK, EAGAIN, EBADF, EINTR, EILWAL, EIO, EFAULT, ENOTTY, ENXIO
     * EOVERFLOW, EPERM
     */
    err = devctl(fd,
            LwColwertFlagUInt32toInt32(DCMD_LWSCIIPC_PHASE1_PULL),
            &msg, sizeof(msg), NULL);
    ret = ResmgrErrnoToLwSciErr(err);
    if (ret == LwSciError_Success) {
        *remotePid = msg.remotePid;
        *remoteChid = msg.remoteChid;
    }

    return ret;
}

static LwSciError lwsciipc_ipc_pulse_phase2_push(int32_t fd,
    int32_t scoid, struct sigevent regEvent)
{
    LwSciIpcPulsePh2Push msg;
    int32_t err;
    LwSciError ret;

    msg.scoid = scoid;
    msg.regEvent = regEvent;
    /*
     * EOK, EAGAIN, EBADF, EINTR, EILWAL, EIO, EFAULT, ENOTTY, ENXIO
     * EOVERFLOW, EPERM
     */
    err = devctl(fd,
            LwColwertFlagUInt32toInt32(DCMD_LWSCIIPC_PHASE2_PUSH),
            &msg, sizeof(msg), NULL);
    ret = ResmgrErrnoToLwSciErr(err);

    return ret;
}

static LwSciError lwsciipc_ipc_pulse_phase2_pull(int32_t fd,
    int32_t *svrScoid, struct sigevent *svrEvent)
{
    LwSciIpcPulsePh2Pull msg;
    int32_t err;
    LwSciError ret;

    /*
     * EOK, EAGAIN, EBADF, EINTR, EILWAL, EIO, EFAULT, ENOTTY, ENXIO
     * EOVERFLOW, EPERM
     */
    err = devctl(fd,
            LwColwertFlagUInt32toInt32(DCMD_LWSCIIPC_PHASE2_PULL),
            &msg, sizeof(msg), NULL);
    ret = ResmgrErrnoToLwSciErr(err);
    if (ret == LwSciError_Success) {
        *svrScoid = msg.svrScoid;
        *svrEvent = msg.svrEvent;
    }

    return ret;
}

/*====================================================================
 * LwSciIpc Inter-process backend functions
 *====================================================================
 */

/**
 * @brief Send notification to peer endpoint process (intra-VM)
 *
 * This function is registered as a callback in sivc_init() and
 * it's called in IVC core library context.
 * This notification callback is called whenever IVC connection status or
 * buffer status is changed (RX full to non-full, TX empty to non-empty).
 *
 * @param[in] pointer of IVC instance handle
 */
static void lwsciipc_ipc_notify(struct sivc_queue *sivc)
{
    /* ipch extends sivc */
    const struct lwsciipc_ipc_handle *ipch =
        (const struct lwsciipc_ipc_handle *)(void *)sivc;
    int32_t ret;

    if (sivc == NULL) {
        LWSCIIPC_ERR_STR("error: lwsciipc_ipc_notify: IPC handle is NULL");
        goto fail;
    }

    /* ready to send event to peer */
    if (ipch->rmState == (int32_t)RM_STATE_IDLE) {
        ret = MsgDeliverEvent_r(ipch->serverScoid, &ipch->remoteEvent);
        if (ret != EOK) {
            LWSCIIPC_ERR_STRINT(
                "error: lwsciipc_ipc_notify: MsgDeliverEvent fail: ret",
                ret);
        }
    }
#ifdef LWSCIIPC_DEBUG
    else {
        LWSCIIPC_DBG_STRINT(
            "lwsciipc_ipc_notify: abnormal state",
            ipch->rmState);
    }
#endif /* LWSCIIPC_DEBUG */

fail:
    return;
}

/**
 * @brief Undo a previous ConfigurePulseParams call (Intra-VM)
 *
 * @param[in] ipch  Intra-VM handle
 *
 * @return ::LwSciError, the completion code of the operation:
 * - ::LwSciError_Success         Indicates a successful operation.
 * - ::LwSciError_IlwalidState    Indicates an invalid operation state.
 */
static LwSciError lwsciipc_ipc_release_event_handler(
    struct lwsciipc_ipc_handle *ipch)
{
#undef LIB_FUNC
#define LIB_FUNC "lwsciipc_ipc_release_event_handler: "
    LwSciError ret;
    int32_t err;

    LWSCIIPC_DBG_STR(LIB_FUNC "enter");

    ipch->coid = 0;

    ret = lwsciipc_ipc_unregister_events(ipch);
    if (LwSciError_Success != ret) {
        LWSCIIPC_ERR_STR("error: " LIB_FUNC "unregister events");
        goto fail;
    }

    /* release endpoint oclwpation */
    if (ipch->serverChid != 0) {
        err = ChannelDestroy_r(ipch->serverChid);
        ipch->serverChid = 0;
        report_os_errto(err, "ChannelDestroy_r", fail);
    }

    LWSCIIPC_DBG_STR(LIB_FUNC "exit");

    ret = LwSciError_Success;

fail:
    return ret;
}

LwSciError lwsciipc_ipc_get_endpoint_info(
    const struct lwsciipc_ipc_handle *ipch,
    LwSciIpcEndpointInfo *info)
{
    info->nframes = ipch->entry->nFrames;
    info->frame_size = ipch->entry->frameSize;

    return LwSciError_Success;
}

LwSciError lwsciipc_ipc_get_eventnotifier(struct lwsciipc_ipc_handle *ipch,
    LwSciEventNotifier **eventNotifier)
{
    LwSciEventNotifier *notifier;
    LwSciError ret;

    if (ipch->eventService == NULL) {
        LWSCIIPC_ERR_STR(
            "error: lwsciipc_ipc_get_eventnotifier: "
            "EventService is NOT initialized");
        ret = LwSciError_NotInitialized;
        goto fail;
    }

    notifier = ipch->eventNotifier;

    ret = ipch->eventService->CreateNativeEventNotifier(ipch->eventService,
        &ipch->qnxEvent.nativeEvent, &notifier);
    if (ret != LwSciError_Success) {
        goto fail;
    }

    *eventNotifier = (LwSciEventNotifier *)notifier;

fail:
    return ret;
}

LwSciError lwsciipc_ipc_read(struct lwsciipc_ipc_handle *ipch, void *buf,
    uint32_t size, uint32_t *bytes)
{
    int32_t err;
    LwSciError ret;

    err = sivc_read(&ipch->sivc, buf, size);

    if (err < 0) {
        *bytes = 0U;
        update_sivc_err(err);
    } else {
        *bytes = size;
        ret = LwSciError_Success;
    }

    return ret;
}

const volatile void *lwsciipc_ipc_read_get_next_frame(
    struct lwsciipc_ipc_handle *ipch)
{
    return sivc_get_read_frame(&ipch->sivc);
}

LwSciError lwsciipc_ipc_read_advance(struct lwsciipc_ipc_handle *ipch)
{
    int32_t err;
    LwSciError ret;

    err = sivc_read_advance(&ipch->sivc);

    if (err < 0) {
        update_sivc_err(err);
    } else {
        ret = LwSciError_Success;
    }

    return ret;
}

LwSciError lwsciipc_ipc_write(struct lwsciipc_ipc_handle *ipch,
    const void *buf, uint32_t size, uint32_t *bytes)
{
    int32_t err;
    LwSciError ret;

    err = sivc_write(&ipch->sivc, buf, size);

    if (err < 0) {
        *bytes = 0U;
        update_sivc_err(err);
    } else {
        *bytes = size;
        ret = LwSciError_Success;
    }

    return ret;
}

volatile void *lwsciipc_ipc_write_get_next_frame(
    struct lwsciipc_ipc_handle *ipch)
{
    return sivc_get_write_frame(&ipch->sivc);
}

LwSciError lwsciipc_ipc_write_advance(struct lwsciipc_ipc_handle *ipch)
{
    int32_t err;
    LwSciError ret;

    err = sivc_write_advance(&ipch->sivc);

    if (err < 0) {
        update_sivc_err(err);
    } else {
        ret = LwSciError_Success;
    }

    return ret;
}

/* peek in the next rx buffer at offset off, the count bytes */
LwSciError lwsciipc_ipc_read_peek(struct lwsciipc_ipc_handle *ipch, void *buf,
    uint32_t offset, uint32_t count, uint32_t *bytes)
{
    int32_t err;
    LwSciError ret;

    err = sivc_read_peek(&ipch->sivc, buf, offset, count);

    if (err < 0) {
        *bytes = 0U;
        update_sivc_err(err);
    } else {
        *bytes = count;
        ret = LwSciError_Success;
    }

    return ret;
}

/* poke in the next tx buffer at offset off, the count bytes */
LwSciError lwsciipc_ipc_write_poke(struct lwsciipc_ipc_handle *ipch,
    const void *buf, uint32_t offset, uint32_t count, uint32_t *bytes)
{
    int32_t err;
    LwSciError ret;

    err = sivc_write_poke(&ipch->sivc, buf, offset, count);

    if (err < 0) {
        *bytes = 0U;
        update_sivc_err(err);
    } else {
        *bytes = count;
        ret = LwSciError_Success;
    }

    return ret;
}

static LwSciError lwsciipc_ipc_initialize_data(
    struct lwsciipc_ipc_handle *ipch,
    struct LwSciIpcConfigEntry *entry,
    uintptr_t *tx_base,
    uintptr_t *rx_base
)
{
    void *addr = NULL;
    int32_t val;
    uint64_t u64val;
    LwSciError status = LwSciError_Success;
    bool flag;

    ipch->entry = entry;
    ipch->id = entry->id;
    ipch->peer_id = entry->id^1U;
    val = procmgr_ability_lookup(LWSCIIPC_ABILITY_ID);
    if (val < 0) {
        LWSCIIPC_DBG_STRINT(
            "error: lwsciipc_ipc_initialize_data: ability lookup", val);
        status = LwSciError_NotPermitted;
        goto fail;
    }
    ipch->abilityId = (uint32_t)val;
    ipch->pid = getpid();
    ipch->pulseResetFlag = 0;

    /* Map the share memory area */
    addr = lwsciipc_os_mmap(NULL, entry->chPsize,
            /* no overflow since flag definition is 0x300U */
            CastU32toS32WithExit(LW_PROT_READ|LW_PROT_WRITE),
            MAP_SHARED, NOFD, 0, (void *)entry);
    LWOS_COV_WHITELIST(deviate, LWOS_MISRA(Rule, 11_6), "<QNXBSP>:<qnx_asil_header>:<1>:<TID-371>") LWOS_COV_WHITELIST(deviate, LWOS_CERT(INT36_C), "<QNXBSP>:<lwpu>:<1>:<Bug 200736827>")
    if (addr == MAP_FAILED) {
        LWSCIIPC_ERR_STR(
            "error: lwsciipc_ipc_initialize_data: Failed to map IVC area");
        LWSCIIPC_DBG_STR2ULONG("lwsciipc_ipc_initialize_data: addr, size",
            entry->chPaddr, entry->chPsize);
        status = LwSciError_InsufficientMemory;
        goto fail;
    }
    LWOS_COV_WHITELIST(deviate, LWOS_MISRA(Rule, 11_6), "<QNXBSP>:<qnx_asil_header>:<1>:<TID-371>")
    ipch->shm = (uintptr_t)addr;
    ipch->shm_size = entry->chPsize;

    if (entry->rxFirst == 1U) {
        *rx_base = ipch->shm;
        flag = AddU64((uint64_t)ipch->shm, (uint64_t)entry->qsize, &u64val);
        if (flag == false) {
            LWSCIIPC_ERR_STR(
                "error: lwsciipc_ipc_initialize_data: Failed to get tx_base");
            status = LwSciError_IlwalidState;
            goto fail;
        }
        *tx_base = (uintptr_t)u64val;
    } else {
        *tx_base = ipch->shm;
        flag = AddU64((uint64_t)ipch->shm, (uint64_t)entry->qsize, &u64val);
        if (flag == false) {
            LWSCIIPC_ERR_STR(
                "error: lwsciipc_ipc_initialize_data: Failed to get rx_base");
            status = LwSciError_IlwalidState;
            goto fail;
        }
        *rx_base = (uintptr_t)u64val;
    }

fail:
    return status;
}

static void lwsciipc_ipc_close_endpoint_internal(
    struct lwsciipc_ipc_handle *ipch)
{
    int32_t err = 0;

    if (ipch != NULL) {
        /* release event handle. it doesn't handle error here */
        (void)lwsciipc_ipc_release_event_handler(ipch);

        lwsciipc_ipc_close_pulse_device(ipch);

        if (ipch->shm != 0UL) {
            LWOS_COV_WHITELIST(deviate, LWOS_MISRA(Rule, 11_6), "<QNXBSP>:<qnx_asil_header>:<1>:<TID-371>")
            err = munmap((void *)ipch->shm, ipch->shm_size);
            if (err == -1) {
                LwSciError ret = ErrnoToLwSciErr(err);
                LWSCIIPC_ERR_STRINT(
                    "error: lwsciipc_ipc_close_endpoint_internal: "
                    "Failed to unmap: ret",
                    (int32_t)ret);
                LWSCIIPC_ERR_STR2ULONG(
                    "error: lwsciipc_ipc_close_endpoint_internal: shm, size",
                    (uint64_t)ipch->shm, ipch->shm_size);
            }
            ipch->shm = 0UL;
        }

        lwsciipc_os_put_endpoint_mutex(&ipch->mutexfd);
    }
}

static LwSciError lwsciipc_ipc_check_channel_data(
    struct LwSciIpcConfigEntry *entry)
{
    char shmName[MAX_SHM_PATH_LEN] = {0};
    struct stat st;
    LwSciError ret = LwSciError_Success;

    LWSCIIPC_DBG_STR("lwsciipc_ipc_check_channel_data");

    if (entry->shmIdx == LWSCIIPC_SHMIDX_ILWALID) {
        ret = LwSciError_BadParameter;
        goto fail;
    }

    ret = lwsciipc_os_get_node_name(CHANNEL_SHM, entry->shmIdx,
            sizeof(shmName), shmName);
    if (LwSciError_Success != ret) {
        goto fail;
    }

    if (stat(shmName, &st) != 0) {
        LWSCIIPC_ERR_STR("error: lwsciipc_ipc_check_channel_data");
        LWSCIIPC_ERR_STR(shmName);
        ret = LwSciError_InsufficientMemory;
    }

fail:
    return ret;
}

/**
 * Do internal job for opening intra-VM endpoint by communicating LwSciIpc
 * resource manager.
 * It gets mutex for an endpoint from LwSciIpc resource manager if the endpoint
 * is not oclwpied. After it succeeds in getting the mutex, it initializes
 * ivclib.
 *
 * @param[in]  entry  Configuration structure handle
 * @param[out] ipcp   Internal intra-VM handle
 *
 * @return ::LwSciError, the completion code of the operation:
 * - ::LwSciError_Success            Indicates a successful operation.
 * - ::LwSciError_BadParameter       Indicates any argument was NULL or invalid.
 * - ::LwSciError_Busy               Indicates the @a endpoint is already in use.
 * - ::LwSciError_InsufficientMemory Indicates memory allocation failed for the
 *                                   operation.
 * - ::LwSciError_NotSupported       Indicates API is not supported on provided
 *                                   endpoint backend type.
 * - ::LwSciError_IlwalidState       Indicates an invalid operation state.
 * - ::LwSciError_NotPermitted       Indicates process doesn't have the required
 *                                   privilege.
 */
LwSciError lwsciipc_ipc_open_endpoint(struct lwsciipc_ipc_handle **ipcp,
    struct LwSciIpcConfigEntry *entry)
{
    struct lwsciipc_ipc_handle *ipch;
    uintptr_t tx_base = (uintptr_t)0;
    uintptr_t rx_base = (uintptr_t)0;
    int32_t err;
    LwSciError ret = -1;

    LWOS_COV_WHITELIST(deviate, LWOS_MISRA(Rule, 21_3), "<QNXBSP>:<lwpu>:<1>:<TID-385>")
    ipch = (struct lwsciipc_ipc_handle *)calloc(1,
        sizeof(struct lwsciipc_ipc_handle));
    if (ipch == NULL) {
        LWSCIIPC_ERR_STR("error: lwsciipc_ipc_open_endpoint: Failed to calloc");
        ret = LwSciError_InsufficientMemory;
        goto fail;
    }

    ret = lwsciipc_ipc_check_channel_data(entry);
    if (ret != LwSciError_Success) {
        goto fail;
    }

    ret = lwsciipc_os_get_endpoint_mutex(entry, &ipch->mutexfd);
    if (ret != LwSciError_Success) {
        goto fail;
    }

    ret = lwsciipc_ipc_initialize_data(ipch, entry, &tx_base, &rx_base);
    if (ret != LwSciError_Success) {
        goto fail;
    }

    /* cache_ilwalidate and cache_flush callbacks are not used */
    err = sivc_init(&ipch->sivc, rx_base, tx_base,
        entry->nFrames, entry->frameSize, lwsciipc_ipc_notify, NULL, NULL);
    if (err != 0) {
        ret = ErrnoToLwSciErr(err);
        LWSCIIPC_ERR_STRINT(
            "error: lwsciipc_ipc_open_endpoint: ivc_init failed: ret",
            (int32_t)ret);
        goto fail;
    }

    ret = lwsciipc_ipc_open_pulse_device(ipch);
    if (ret != LwSciError_Success) {
        LWSCIIPC_ERR_STRINT(
            "error: lwsciipc_ipc_open_endpoint: open_pulse_device failed: ret",
            (int32_t)ret);
        goto fail;
    }
#if DEBUG_STATE
    LWSCIIPC_DBG_STR2INT("lwsciipc_ipc_open_endpoint: SelChannel: vuid, pid",
        ipch->entry->vuid, ipch->pid);
#endif
    ret = lwsciipc_ipc_pulse_sel_channel(ipch->pulsefd,
            entry->vuid, entry->gid);
    if (ret != LwSciError_Success) {
        LWSCIIPC_ERR_STRINT(
            "error: lwsciipc_ipc_open_endpoint: pulse_sel_channel failure: ret",
            (int32_t)ret);
        goto fail;
    }
    ipch->rmState = (int32_t)RM_STATE_INIT;

    *ipcp = ipch;

    ipch->is_open = LwBoolTrue;

    LWSCIIPC_DBG_STR2INT("lwsciipc_ipc_open_endpoint: exit: vuid, pid",
        ipch->entry->vuid, ipch->pid);

    ret = LwSciError_Success;    /* success */

fail:
    if (ret != LwSciError_Success) {
        lwsciipc_ipc_close_endpoint_internal(ipch);
        LWOS_COV_WHITELIST(deviate, LWOS_MISRA(Rule, 21_3), "<QNXBSP>:<lwpu>:<1>:<TID-385>")
        free(ipch);
        *ipcp = NULL;
    }

    /* ipch is released in lwsciipc_ipc_close_endpoint(). */
    LWOS_COV_WHITELIST(deviate, LWOS_MISRA(Directive, 4_13), "<QNXBSP>:<lwpu>:<1>:<Bug 200736827>")
    return ret;

}

void lwsciipc_ipc_bind_eventservice(struct lwsciipc_ipc_handle *ipch,
    LwSciEventService *eventService)
{
#undef LIB_FUNC
#define LIB_FUNC "lwsciipc_ipc_bind_eventservice: "
    LWSCIIPC_DBG_STR(LIB_FUNC "enter");

    ipch->eventService = eventService;
    ipch->qnxEvent.handle = (void *)ipch;

    ipch->qnxEvent.nativeEvent.ConfigurePulseParams =
        lwsciipc_ipc_configure_pulse_param;
    ipch->qnxEvent.nativeEvent.UnconfigurePulseParams =
        lwsciipc_ipc_unconfigure_pulse_param;
    ipch->qnxEvent.nativeEvent.UnmaskInterrupt = NULL;

    LWSCIIPC_DBG_STR(LIB_FUNC "exit");
}

LwSciError lwsciipc_ipc_open_endpoint_with_eventservice(
    struct lwsciipc_ipc_handle **ipcp,
    struct LwSciIpcConfigEntry *entry,
    LwSciEventService *eventService)
{
#undef LIB_FUNC
#define LIB_FUNC "lwsciipc_ipc_open_endpoint_with_eventservice: "
    struct lwsciipc_ipc_handle *ipch = NULL;
    LwSciError ret;

    LWSCIIPC_DBG_STR(LIB_FUNC "enter");
    ret =  lwsciipc_ipc_open_endpoint(ipcp, entry);
    if (ret == LwSciError_Success) {
        ipch = *ipcp;
        lwsciipc_ipc_bind_eventservice(ipch, eventService);
    }
    else {
        goto fail;
    }

fail:
    LWSCIIPC_DBG_STR(LIB_FUNC "exit");
    return ret;
}


void lwsciipc_ipc_close_endpoint(struct lwsciipc_ipc_handle *ipch)
{
    LWSCIIPC_DBG_STR("lwsciipc_ipc_close_endpoint: enter");

    lwsciipc_ipc_close_endpoint_internal(ipch);
    LWOS_COV_WHITELIST(deviate, LWOS_MISRA(Rule, 21_3), "<QNXBSP>:<lwpu>:<1>:<TID-385>")
    free(ipch);

    LWSCIIPC_DBG_STR("lwsciipc_ipc_close_endpoint: exit");
}

void lwsciipc_ipc_reset_endpoint(struct lwsciipc_ipc_handle *ipch)
{
    LWSCIIPC_DBG_STR("lwsciipc_ipc_reset_endpoint: enter");

    sivc_reset(&ipch->sivc);

    LWSCIIPC_DBG_STR("lwsciipc_ipc_reset_endpoint: exit");
}

/* This check function always return error since it's called
 * only when following IVC apis return NULL pointer.
 * tegra_ivc_read_get_next_frame()
 * tegra_ivc_write_get_next_frame()
 */
LwSciError lwsciipc_ipc_check_read(struct lwsciipc_ipc_handle *ipch)
{
    LwSciError ret;

    /* At first, RM state should be checked */
    if ((ipch->rmState != (int32_t)RM_STATE_IDLE) ||
    (ipch->prev_conn == LW_SCI_IPC_EVENT_CONN_RESET)) {
        ret = LwSciError_ConnectionReset;
    }
    else {
        ret = LwSciError_InsufficientMemory;
    }

    return ret;
}

LwSciError lwsciipc_ipc_check_write(struct lwsciipc_ipc_handle *ipch)
{
    return lwsciipc_ipc_check_read(ipch);
}

static LwSciError lwsciipc_ipc_unregister_events(
    struct lwsciipc_ipc_handle *ipch)
{
#undef LIB_FUNC
#define LIB_FUNC "lwsciipc_ipc_unregister_events: "
    LwSciError ret = LwSciError_IlwalidState;
    int32_t err = EOK;
    int32_t err2 = EOK;
    int32_t err3 = EOK;

    ipch->pulseResetFlag = 0;

    if (ipch->regProgressEvent.sigev_handle != 0UL) {
        err = MsgUnregisterEvent_r(&ipch->regProgressEvent);
    }
    if (ipch->regSetResetFlagEvent.sigev_handle != 0UL) {
        err2 = MsgUnregisterEvent_r(&ipch->regSetResetFlagEvent);
    }
    if (ipch->regPeerEvent.sigev_handle != 0UL) {
        err3 = MsgUnregisterEvent_r(&ipch->regPeerEvent);
    }

    if ((EOK != err) || (EOK != err2) || (EOK != err3)) {
        LWSCIIPC_DBG_STR(LIB_FUNC "event not found");
        ret = LwSciError_IlwalidState;
    }
    else {
        ipch->regProgressEvent.sigev_handle = 0U;
        ipch->regSetResetFlagEvent.sigev_handle = 0U;
        ipch->regPeerEvent.sigev_handle = 0U;

        ret = LwSciError_Success;
    }

    /* release previous private connection of remote chid for signalling */
    if (ipch->clientCoid != 0) {
        err = ConnectDetach_r(ipch->clientCoid);
        ipch->clientCoid = 0;
        update_os_err(err, "ConnectDetach_r");
    }

    return ret;
}

/**
 * Perform reset state actions
 *
 * @param[in] ipch  LwSciIpc Intra-VM handle
 *
 * @return ::LwSciError, the completion code of the operation:
 * - ::LwSciError_Success         Indicates a successful operation.
 * - ::LwSciError_IlwalidState    Indicates an invalid operation state.
 */
static LwSciError lwsciipc_ipc_pulse_rm_reset_state(
    struct lwsciipc_ipc_handle *ipch)
{
#undef LIB_FUNC
#define LIB_FUNC "lwsciipc_ipc_pulse_rm_reset_state: "
    LwSciError ret;

    LWSCIIPC_DBG_STR2INT(LIB_FUNC "vuid, pid",
        ipch->entry->vuid, ipch->pid);

    ret = lwsciipc_ipc_unregister_events(ipch);
    if (LwSciError_Success != ret) {
        LWSCIIPC_ERR_STR("error: " LIB_FUNC "unregister events");
        goto fail;
    }

#if DEBUG_STATE
    LWSCIIPC_DBG_STR2INT(LIB_FUNC "SelChannel: vuid, pid",
        ipch->entry->vuid, ipch->pid);
#endif
    ret = lwsciipc_ipc_pulse_sel_channel(ipch->pulsefd,
        ipch->entry->vuid, ipch->entry->gid);
    if (ret != LwSciError_Success) {    // EILWAL, EPERM
        LWSCIIPC_ERR_STRINT("error: " LIB_FUNC "pulse_sel_channel failure: ",
            (int32_t)ret);
        ret = LwSciError_IlwalidState;
        goto fail;
    }
    ipch->rmState = (int32_t)RM_STATE_INIT;
    LWSCIIPC_DBG_STR2INT(LIB_FUNC "INIT: vuid, pid",
        ipch->entry->vuid, ipch->pid);

    ret = lwsciipc_ipc_register_events(ipch);
    if (ret != LwSciError_Success) {
        goto fail;
    }

#if DEBUG_STATE
    LWSCIIPC_DBG_STR2INT(LIB_FUNC "Phase1Push: vuid, pid",
        ipch->entry->vuid, ipch->pid);
#endif
    ret = lwsciipc_ipc_pulse_phase1_push(ipch->pulsefd,
        ipch->regProgressEvent, ipch->regSetResetFlagEvent,
        ipch->serverChid);
    if (ret != LwSciError_Success) {    // EILWAL
        LWSCIIPC_ERR_STRINT("error: " LIB_FUNC "pulse_phase1_push failure",
            (int32_t)ret);
        ret = LwSciError_IlwalidState;
        goto fail;
    }
    ipch->rmState = (int32_t)RM_STATE_PHASE1PUSHED;
    LWSCIIPC_DBG_STR2INT(LIB_FUNC "INIT->PH1PUSH: vuid, pid",
        ipch->entry->vuid, ipch->pid);

fail:
    return ret;
}

/**
 * Perform actions for each RM state which is PHASE1PUSHED and PHASE2PUSHED
 *
 * @param[in] ipch  LwSciIpc Intra-VM handle
 *
 * @return ::LwSciError, the completion code of the operation:
 * - ::LwSciError_Success         Indicates a successful operation.
 * - ::LwSciError_IlwalidState    Indicates an invalid operation state.
 */
static LwSciError lwsciipc_ipc_pulse_rm_process_phase1pulled_state(
    struct lwsciipc_ipc_handle *ipch)
{
    LwSciError ret = LwSciError_Success;

#if DEBUG_STATE
    LWSCIIPC_DBG_STR2INT(
        "lwsciipc_ipc_pulse_rm_process_phase1pulled_state: "
        "PH1PULL calls Phase2Push: vuid, pid",
        ipch->entry->vuid, ipch->pid);
#endif
    ret = lwsciipc_ipc_pulse_phase2_push(ipch->pulsefd,
        ipch->remoteScoid, ipch->regPeerEvent);

    if (ret == LwSciError_TryItAgain) {
        /* EAGAIN: still phase1ready state */
        /* client doesn't need to handle EAGAIN */
        ret = LwSciError_Success;
        goto done;
    }
    if (ret != LwSciError_Success) { // EILWAL case
        LWSCIIPC_DBG_STR2INT(
            "lwsciipc_ipc_pulse_rm_process_phase1pulled_state: "
            "pulse_phase2_push error: vuid, pid",
            ipch->entry->vuid, ipch->pid);
        LWSCIIPC_DBG_STRINT(
            "lwsciipc_ipc_pulse_rm_process_phase1pulled_state: "
            "pulse_phase2_push error: ret", ret);
        ret = lwsciipc_ipc_pulse_rm_reset_state(ipch);
        if (ret != LwSciError_Success) {
            LWSCIIPC_ERR_STR(
                "error: lwsciipc_ipc_pulse_rm_process_phase1pulled_state: "
                "unrecoverable error");
            ret = LwSciError_IlwalidState;
        }
        /* reset recovery is done regardless of result,
         * return to caller directly.
         */
        goto done;
    }

    ipch->rmState = (int32_t)RM_STATE_PHASE2PUSHED;
#if DEBUG_STATE
    LWSCIIPC_DBG_STR2INT(
        "lwsciipc_ipc_pulse_rm_process_phase1pulled_state: PH1PULL->PH2PUSH",
        ipch->entry->vuid, ipch->pid);
#endif
    ret = LwSciError_Success;

done:
    return ret;
}

LWOS_COV_WHITELIST(deviate, LWOS_MISRA(Rule, 5_2), "<QNXBSP>:<lwpu>:<1>:<Bug 200603420>")
static LwSciError lwsciipc_ipc_pulse_rm_process_phase1pushed_state(
    struct lwsciipc_ipc_handle *ipch)
{
    int32_t remotePid;
    int32_t remoteChid;
    struct _server_info info;
    int32_t coid;
    int32_t err;
    LwSciError ret = LwSciError_Success;

#if DEBUG_STATE
    LWSCIIPC_DBG_STR2INT(
        "lwsciipc_ipc_pulse_rm_process_phase1pushed_state: "
        "P1PUSH calls Phase1Pull",
        ipch->entry->vuid, ipch->pid);
#endif
    ret = lwsciipc_ipc_pulse_phase1_pull(ipch->pulsefd,
        &remotePid, &remoteChid);
    if (ret == LwSciError_TryItAgain) {
        /* EAGAIN: still phase1pushed state */
        /* client doesn't need to handle EAGAIN */
        ret = LwSciError_Success;
        goto done;
    }
    if (ret != LwSciError_Success) { // EILWAL case
        LWSCIIPC_DBG_STR2INT(
            "lwsciipc_ipc_pulse_rm_process_phase1pushed_state: "
            "pulse_phase1_pull error: vuid, pid",
            ipch->entry->vuid, ipch->pid);
        LWSCIIPC_DBG_STRINT(
            "lwsciipc_ipc_pulse_rm_process_phase1pushed_state: "
            "pulse_phase1_pull error", ret);
        ret = lwsciipc_ipc_pulse_rm_reset_state(ipch);
        if (ret != LwSciError_Success) {
            LWSCIIPC_ERR_STR(
                "error: lwsciipc_ipc_pulse_rm_process_phase1pushed_state: "
                "unrecoverable error");
            ret = LwSciError_IlwalidState;
        }
        /* reset recovery is done regardless of result,
         * return to caller directly.
         */
        goto done;
    }

    /* test invalid usecases to check pids of both endpoints */
    if ((ipch->entry->backend == (uint32_t)LWSCIIPC_BACKEND_ITC) &&
        (remotePid != ipch->pid)) {
        LWSCIIPC_ERR_STR2INT(
            "error: lwsciipc_ipc_pulse_rm_process_phase1pushed_state: "
            "pids are different: local, remote", ipch->pid, remotePid);
        ret = LwSciError_IlwalidState;
        goto done;
    }
    if ((ipch->entry->backend == (uint32_t)LWSCIIPC_BACKEND_IPC) &&
        (remotePid == ipch->pid)) {
        LWSCIIPC_ERR_STRINT(
            "error: lwsciipc_ipc_pulse_rm_process_phase1pushed_state: "
            "pids are NOT different: pid", ipch->pid);
        ret = LwSciError_IlwalidState;
        goto done;
    }

    ipch->remotePid = remotePid;
    ipch->remoteChid = remoteChid;

    /* validate remote pid and chid, then create coid */
    coid = ConnectAttach_r(0, ipch->remotePid,
        ipch->remoteChid, _NTO_SIDE_CHANNEL, _NTO_COF_REG_EVENTS);
    if (coid < 0) {
        LWSCIIPC_ERR_STRINT(
            "error: lwsciipc_ipc_pulse_rm_process_phase1pushed_state: "
            "ConnectAttach_r error: ret", coid);
        ret = LwSciError_IlwalidState;
        goto done;
    }
    ipch->clientCoid = coid;

    err = ConnectServerInfo_r(0, ipch->clientCoid, &info);
    if ((err < 0) || (info.coid != ipch->clientCoid)) {
        LWSCIIPC_ERR_STRINT(
            "error: lwsciipc_ipc_pulse_rm_process_phase1pushed_state: "
            "ConnectServerInfo_r: ret", err);
        ret = LwSciError_IlwalidState;
        goto done;
    }
#if DEBUG_STATE
    LWSCIIPC_DBG_STR2INT("lwsciipc_ipc_pulse_rm_process_phase1pushed_state: "
        "vuid, pid", ipch->entry->vuid, ipch->pid);
    LWSCIIPC_DBG_STR2INT("lwsciipc_ipc_pulse_rm_process_phase1pushed_state: "
        "remotePId, remoteChid", remotePid, remoteChid);
    LWSCIIPC_DBG_STR2INT("lwsciipc_ipc_pulse_rm_process_phase1pushed_state: "
        "info.pid, info.scoid", info.pid, info.scoid);
    LWSCIIPC_DBG_STR2INT("lwsciipc_ipc_pulse_rm_process_phase1pushed_state: "
        "coid, ret", coid, ret);
#endif
    ipch->remoteScoid = info.scoid;

    ipch->regPeerEvent = ipch->userEvent;
    err = MsgRegisterEvent_r(&ipch->regPeerEvent, ipch->clientCoid);
    if (err != EOK) {
        LWSCIIPC_ERR_STRINT(
            "error: lwsciipc_ipc_pulse_rm_process_phase1pushed_state: "
            "MsgRegisterEvent_r(peerEvent): ret", err);
        ret = LwSciError_IlwalidState;
        goto done;
    }
    ipch->rmState = (int32_t)RM_STATE_PHASE1PULLED;
#if DEBUG_STATE
    LWSCIIPC_DBG_STR2INT(
        "lwsciipc_ipc_pulse_rm_process_phase1pushed_state: "
        "PH1PUSH->PH1PULL: vuid, pid", ipch->entry->vuid, ipch->pid);
    LWSCIIPC_DBG_STR2INT(
        "lwsciipc_ipc_pulse_rm_process_phase1pushed_state: "
        "PH1PULL calls Phase2Push: vuid, pid", ipch->entry->vuid, ipch->pid);
#endif

    ret = lwsciipc_ipc_pulse_rm_process_phase1pulled_state(ipch);

done:
    return ret;
}

static LwSciError lwsciipc_ipc_authenticate_remote_endpoint(
    struct lwsciipc_ipc_handle *ipch)
{
#ifdef USE_IOLAUNCHER_FOR_SELWRITY
#undef LIB_FUNC
#define LIB_FUNC "lwsciipc_ipc_authenticate_remote_endpoint: "
    struct _client_info *infop = NULL;
    struct _client_able able = {
        .range_lo = (uint64_t)(ipch->entry->vuid^1UL),
        .range_hi = (uint64_t)(ipch->entry->vuid^1UL)
    };
    int32_t err;
    LwSciError ret;

    able.ability = ipch->abilityId;
    /* EOK, EFAULT, EILWAL */
    err = ConnectClientInfoAble(ipch->serverScoid, &infop, 0, &able, 1);
    /* This is a safety and security mechanism to protect against
     * the possibility that the LwSciIpc pulse RM erroneously or
     * maliciously gave us the pid of a process that doesn't
     * have permission to access the other end of the channel.
     */
    if ((err < 0) || (able.flags == 0U)) {
        ret = lwsciipc_ipc_pulse_rm_reset_state(ipch);
        if (ret != LwSciError_Success) {
            LWSCIIPC_ERR_STR(LIB_FUNC "unrecoverable error");
            ret = LwSciError_IlwalidState;
        }
        /* reset recovery is done regardless of result,
         * return to caller directly.
         */
        goto done;
    }
    ret = LwSciError_Success;

    LWSCIIPC_DBG_STR2INT(LIB_FUNC
        "verified lwsciipc ability of peer: vuid, pid",
        ipch->entry->vuid, ipch->pid);

done:
    if (infop != NULL) {
        err = ClientInfoExtFree(&infop);
        if (err != EOK) {
            LWSCIIPC_ERR_STRINT(LIB_FUNC "ClientInfoExtFree", err);
        }
    }
    return ret;
#else
    return LwSciError_Success;
#endif /* USE_IOLAUNCHER_FOR_SELWRITY */
}

/**
 * Validate data delivered from peer in the PHASE2PULLED
 *
 * @param[in] ipch  LwSciIpc Intra-VM handle
 *
 * @return ::LwSciError, the completion code of the operation:
 * - ::LwSciError_Success         Indicates a successful operation.
 * - ::LwSciError_IlwalidState    Indicates an invalid operation state.
 */
static LwSciError lwsciipc_ipc_pulse_rm_validate_phase2pulled_data(
    struct lwsciipc_ipc_handle *ipch)
{
#undef LIB_FUNC
#define LIB_FUNC "lwsciipc_ipc_pulse_rm_validate_phase2pulled_data: "
    struct _client_info info;
    int32_t err;
    LwSciError ret = LwSciError_IlwalidState;

    /* EOK, EFAULT, EILWAL */
    err = ConnectClientInfo_r(ipch->serverScoid, &info, 0);
    if (err != EOK) {
        LWSCIIPC_ERR_STRINT("error: " LIB_FUNC
            "ConnectClientInfo error: ret", err);
        ret = LwSciError_IlwalidState;
        goto done;
    }
    LWSCIIPC_DBG_STR(LIB_FUNC "ConnectClientInfo scoid was verified");

    /* This is a safety and security mechanism to protect against
     * the possibility that the other end erroneously/maliciously
     * gave us an scoid from a process distinct from
     * the process we connected to.
     */
    /* remotePid is peer client pid which is connected to local server. */
    if (info.pid != ipch->remotePid) {
        LWSCIIPC_DBG_STR2INT(LIB_FUNC "pid match error: vuid, pid",
            ipch->entry->vuid, ipch->pid);
        ret = lwsciipc_ipc_pulse_rm_reset_state(ipch);
        if (ret != LwSciError_Success) {
            LWSCIIPC_ERR_STR( "error: " LIB_FUNC "unrecoverable error");
            ret = LwSciError_IlwalidState;
        }
        /* reset recovery is done regardless of result,
         * return to caller directly.
         */
        goto done;
    }

    /* authenticate remote endpoint process */
    ret = lwsciipc_ipc_authenticate_remote_endpoint(ipch);

done:
    return ret;
}

LWOS_COV_WHITELIST(deviate, LWOS_MISRA(Rule, 5_2), "<QNXBSP>:<lwpu>:<1>:<Bug 200603420>")
static LwSciError lwsciipc_ipc_pulse_rm_process_phase2pushed_state(
    struct lwsciipc_ipc_handle *ipch)
{
    int32_t serverScoid;
    struct sigevent remoteEvent;
    LwSciError ret = LwSciError_Success;

#if DEBUG_STATE
    LWSCIIPC_DBG_STR2INT(
        "lwsciipc_ipc_pulse_rm_process_phase2pushed_state: "
        "PH2PUSH calls Phase2Pull: vuid, pid",
        ipch->entry->vuid, ipch->pid);
#endif
    ret = lwsciipc_ipc_pulse_phase2_pull(ipch->pulsefd,
        &serverScoid, &remoteEvent);
    if (ret == LwSciError_TryItAgain) {
        /* EAGAIN: still phase2pushed state */
        /* client doesn't need to handle EAGAIN */
        ret = LwSciError_Success;
        goto done;
    }
    if (ret != LwSciError_Success) { // EILWAL case
        LWSCIIPC_DBG_STR2INT(
            "lwsciipc_ipc_pulse_rm_process_phase2pushed_state: "
            "pulse_phase2_pull error: vuid, pid",
            ipch->entry->vuid, ipch->pid);
        ret = lwsciipc_ipc_pulse_rm_reset_state(ipch);
        if (ret != LwSciError_Success) {
            LWSCIIPC_ERR_STRINT(
                "error: lwsciipc_ipc_pulse_rm_process_phase2pushed_state: "
                "unrecoverable error: ret", (int32_t)ret);
            ret = LwSciError_IlwalidState;
        }
        /* reset recovery is done regardless of result,
         * return to caller directly.
         */
        goto done;
    }

    ipch->serverScoid = serverScoid;
    ipch->remoteEvent = remoteEvent;
#if DEBUG_STATE
    LWSCIIPC_DBG_STRINT(
        "lwsciipc_ipc_pulse_rm_process_phase2pushed_state: svrScoid",
        serverScoid);
    LWSCIIPC_DBG_STR2INT(
        "lwsciipc_ipc_pulse_rm_process_phase2pushed_state: rmt coid, code",
        remoteEvent.sigev_coid,
        remoteEvent.sigev_code);
    LWSCIIPC_DBG_STR2INT(
        "lwsciipc_ipc_pulse_rm_process_phase2pushed_state: rmt prio, event",
        remoteEvent.sigev_priority,
        remoteEvent.sigev_value.sival_int);
#endif
    ret = lwsciipc_ipc_pulse_rm_validate_phase2pulled_data(ipch);
    if (ret != LwSciError_Success) {
        goto done;
    }

    ipch->rmState = (int32_t)RM_STATE_IDLE;
#if DEBUG_STATE
    LWSCIIPC_DBG_STR2INT(
        "lwsciipc_ipc_pulse_rm_process_phase2pushed_state: "
        "PH2PUSH->IDLE: vuid, pid", ipch->entry->vuid, ipch->pid);
#endif
    ret = LwSciError_Success;

done:
    return ret;
}

/**
 * Perform actions for each RM state which is PHASE1PUSHED and PHASE2PUSHED
 *
 * @param[in] ipch  LwSciIpc Intra-VM handle
 *
 * @return ::LwSciError, the completion code of the operation:
 * - ::LwSciError_Success         Indicates a successful operation.
 * - ::LwSciError_IlwalidState    Indicates an invalid operation state.
 */
static LwSciError lwsciipc_ipc_pulse_rm_update_state(
    struct lwsciipc_ipc_handle *ipch)
{
    LwSciError ret = LwSciError_Success;

    switch(ipch->rmState) {
        /* 1) lwsciipc_ipc_open_endpoint() and
         *    lwsciipc_ipc_set_qnx_pulse_param() are called
         * 2) unrecoverabl state error was oclwrred
         * 3) peer endpoint is restarted in idle status
         * 4) peer endpoint doesn't push phase1 data yet
         */
        case (int32_t)RM_STATE_PHASE1PUSHED:
        {
            ret = lwsciipc_ipc_pulse_rm_process_phase1pushed_state(ipch);
            break;
        }

        /* RM state is still phase1ready. not pulled phase1 data yet. */
        /* FIXME: unreachable state ? */
        case (int32_t)RM_STATE_PHASE1PULLED:
        {
            ret = lwsciipc_ipc_pulse_rm_process_phase1pulled_state(ipch);
            break;
        }

        /* phase2pushed is completed. need to pull phase2 data */
        case (int32_t)RM_STATE_PHASE2PUSHED:
        {
            ret = lwsciipc_ipc_pulse_rm_process_phase2pushed_state(ipch);
            break;
        }

        /* pulse parameter exchanging is completed */
        case (int32_t)RM_STATE_IDLE:
            ret = LwSciError_Success;
            break;

        default:
            LWSCIIPC_ERR_STRINT(
                "error: lwsciipc_ipc_pulse_rm_update_state: "
                "Invalid pulse RM state", ipch->rmState);
            ret = LwSciError_IlwalidState;
            break;
    }

    return ret;
}

/**
 * Look for which event is notified and return all events that has triggered
 * This function returns success by default.
 * It returns error only when mutex call fails.
 *
 * @param[in] ipch      LwSciIpc Intra-VM handle
 * @param[in] inth      LwSciIpc internal object pointer
 * @param[out] connOut  Connection event
 * @param[out] valueOut Read or write event
 *
 * @return ::LwSciError, the completion code of the operation:
 * - ::LwSciError_Success         Indicates a successful operation.
 * - ::LwSciError_IlwalidState    Indicates an invalid operation state.
 */
static LwSciError lwsciipc_ipc_ivc_event_sequence(
    struct lwsciipc_ipc_handle *ipch,
    struct lwsciipc_internal_handle *inth,
    uint32_t *connOut,
    uint32_t *valueOut)
{
    uint32_t value = 0U;
    uint32_t conn = 0U;
    LwSciError ret = LwSciError_Success;
    int32_t err = EAGAIN;
    int32_t merr;

    if (sivc_need_notify(&ipch->sivc) == false) {
        if (sivc_can_write(&ipch->sivc)) {
            value |= LW_SCI_IPC_EVENT_WRITE;
        }
        if (sivc_can_read(&ipch->sivc)) {
            value |= LW_SCI_IPC_EVENT_READ;
        }
    }
    else {
        merr = lwsciipc_os_mutex_lock(&inth->wrMutex); /* wr pos/refcnt */
        report_mutex_errto(merr, "WR mutex_lock", fail);
        merr = lwsciipc_os_mutex_lock(&inth->rdMutex); /* rd pos/refcnt */
        report_mutex_errto(merr, "RD mutex_lock", un_wr_mtx);

        /* get connection status when either endpoint is NOT in
         * established state
         */
        err = sivc_notified(&ipch->sivc);

        merr = lwsciipc_os_mutex_unlock(&inth->rdMutex);
        report_mutex_errto(merr, "RD mutex_unlock", un_wr_mtx);
un_wr_mtx:
        merr = lwsciipc_os_mutex_unlock(&inth->wrMutex);
        update_mutex_err(merr, "WR mutex_unlock");
        if (ret != LwSciError_Success) {
            /* doesn't update valueOut & connOut */
            goto fail;
        }

        if (err == EOK) {
            conn = LW_SCI_IPC_EVENT_CONN_EST;
            /* check buffer status again after establishment */
            if (sivc_can_write(&ipch->sivc)) {
                value |= LW_SCI_IPC_EVENT_WRITE;
            }
            if (sivc_can_read(&ipch->sivc)) {
                value |= LW_SCI_IPC_EVENT_READ;
            }
        }
        else {
            conn = LW_SCI_IPC_EVENT_CONN_RESET;
        }
    }

    *valueOut = value;
    *connOut = conn;
fail:
    return ret;
}

/**
 * Get intra-VM events that has arrived.
 * event return is valid only when return value is LwSciError_Success
 *
 * @param[in]    inth   LwSciIpc internal object pointer
 * @param[inout] ipch   Intra-VM handle
 * @param[out]   events Events that has arrived
 *
 * @return ::LwSciError, the completion code of the operation:
 * - ::LwSciError_Success         Indicates a successful operation.
 * - ::LwSciError_IlwalidState    Indicates an invalid operation state.
 */
LwSciError lwsciipc_ipc_get_event(struct lwsciipc_ipc_handle *ipch,
    uint32_t *event, struct lwsciipc_internal_handle *inth)
{
    uint32_t value;
    uint32_t conn;
    LwSciError ret;

    if (ipch->pulseResetFlag == 1U) {
        LWSCIIPC_DBG_STR2INT(
            "lwsciipc_ipc_get_event: Got ResetFlag from peer: vuid, pid",
            ipch->entry->vuid, ipch->pid);
        ret = lwsciipc_ipc_pulse_rm_reset_state(ipch);
        if (ret != LwSciError_Success) {
            LWSCIIPC_ERR_STR(
                "error: lwsciipc_ipc_get_event: unrecoverable error");
            goto done;
        }
    }

    ret = lwsciipc_ipc_pulse_rm_update_state(ipch);
    if (ret != LwSciError_Success) {
        LWSCIIPC_ERR_STR(
            "error: lwsciipc_ipc_get_event: invalid state");
        goto done;
    }

    if (ipch->rmState != (int32_t)RM_STATE_IDLE) {
        conn = LW_SCI_IPC_EVENT_CONN_RESET;
        if (conn != ipch->prev_conn) {
            LWSCIIPC_DBG_STR2INT("lwsciipc_ipc_get_event0: vuid, pid",
                ipch->entry->vuid, ipch->pid);
            LWSCIIPC_DBG_STR2INT("lwsciipc_ipc_get_event0: prev_conn, conn",
                ipch->prev_conn, conn);
            ipch->prev_conn = conn;
        }
        *event = conn;
        goto done;
    }

    ret = lwsciipc_ipc_ivc_event_sequence(ipch, inth, &conn, &value);
    if (LwSciError_Success != ret) {
        goto done;
    }

    if ((conn & LW_SCI_IPC_EVENT_CONN_MASK) != ipch->prev_conn) {
        *event = (value | conn);
        LWSCIIPC_DBG_STR2INT("lwsciipc_ipc_get_event: vuid, pid",
            ipch->entry->vuid, ipch->pid);
        LWSCIIPC_DBG_STR2INT("lwsciipc_ipc_get_event: prev_conn, conn",
            ipch->prev_conn, conn);
        LWSCIIPC_DBG_STRINT("lwsciipc_ipc_get_event: event", *event);
        ipch->prev_conn = conn;
    }
    else {
        *event = value;
    }

    ret = LwSciError_Success;

done:
    return ret;
}

bool lwsciipc_ipc_can_read(struct lwsciipc_ipc_handle *ipch)
{
    return sivc_can_read(&ipch->sivc);
}

bool lwsciipc_ipc_can_write(struct lwsciipc_ipc_handle *ipch)
{
    return sivc_can_write(&ipch->sivc);
}

/* it returns LwSciError_Success or LwSciError_IlwalidState */
static LwSciError lwsciipc_ipc_register_events(struct lwsciipc_ipc_handle *ipch)
{
    LwSciError ret = LwSciError_Success;
    int32_t err;

    /* We can't call multiple MsgRegisterEvent w/ same sig event and
     * different coid.
     * if we add this code, next time MsgRegisterEvent return EILWAL
     * fd is special coid
     */
    ipch->regProgressEvent = ipch->userEvent;
    /* EAGAIN, EILWAL, ENOMEM */
    err = MsgRegisterEvent_r(&ipch->regProgressEvent, ipch->pulsefd);
    if (err != EOK) {
        LWSCIIPC_ERR_STRINT(
            "error: lwsciipc_ipc_register_events: "
            "MsgRegisterEvent_r(progressEvent): ret", err);
        ret = LwSciError_IlwalidState;
        goto fail;
    }

    ipch->regSetResetFlagEvent = ipch->setResetFlagEvent;
    /* EAGAIN, EILWAL, ENOMEM */
    err = MsgRegisterEvent_r(&ipch->regSetResetFlagEvent, ipch->pulsefd);
    if (err != EOK) {
        LWSCIIPC_ERR_STRINT(
            "error: lwsciipc_ipc_register_events: "
            "MsgRegisterEvent_r(setResetFlagEvent): ret", err);
        ret = LwSciError_IlwalidState;
        goto fail;
    }

fail:
    return ret;
}

/**
 * Perform actions for each RM state which is PHASE1PUSHED and PHASE2PUSHED
 *
 * @param[in] thisNativeEvent   LwSciIpc Intra-VM handle.
 * @param[in] coid              The connection ID created from calling
 *                              ConnectAttach_r().
 * @param[in] pulsePriority     The value for pulse priority.
 * @param[in] pulseCode         The 8-bit positive pulse code specified by
 *                              the user. The values must be between
 *                              _PULSE_CODE_MINAVAIL and _PULSE_CODE_MAXAVAIL.
 * @param[in] pulseValue        A pointer to the user defined pulse value.
 *
 * @return ::LwSciError, the completion code of the operation:
 * - ::LwSciError_Success         Indicates a successful operation.
 * - ::LwSciError_ResourceError   Indicates not enough system resources.
 * - ::LwSciError_IlwalidState    Indicates an invalid operation state.
 */
LwSciError lwsciipc_ipc_set_qnx_pulse_param(
    struct lwsciipc_ipc_handle *ipch, int32_t coid,
    int16_t priority, int16_t code, void *value)
{
#undef LIB_FUNC
#define LIB_FUNC "lwsciipc_ipc_set_qnx_pulse_param: "
    int32_t chid;
    LwSciError ret;
    LwSciError err;

    if (ipch->coid != 0) {
        LWSCIIPC_ERR_STR( "error: " LIB_FUNC
            "connection id is already assigned to endpoint");
        ret = LwSciError_ResourceError;
        goto fail;
    }

    LWSCIIPC_DBG_STR2INT(LIB_FUNC "vuid, resetflag",
        ipch->entry->vuid, ipch->pulseResetFlag);

    /* set pulse parameters */
    LWOS_COV_WHITELIST(deviate, LWOS_MISRA(Rule, 10_1), "<QNXBSP>:<qnx_asil_header>:<2>:<TID-257>") LWOS_COV_WHITELIST(deviate, LWOS_MISRA(Rule, 11_6), "<QNXBSP>:<qnx_asil_header>:<2>:<TID-371>") LWOS_COV_WHITELIST(deviate, LWOS_MISRA(Rule, 12_3), "<QNXBSP>:<qnx_asil_header>:<4>:<Bug 200736827>") LWOS_COV_WHITELIST(deviate, LWOS_MISRA(Rule, 13_4), "<QNXBSP>:<qnx_asil_header>:<5>:<Bug 200736827>")
    SIGEV_PULSE_PTR_INIT(&ipch->userEvent, coid, priority, code, value);
#if DEBUG_STATE
    LWSCIIPC_DBG_STR2INT(LIB_FUNC "coid, code",
        ipch->userEvent.sigev_coid,
        ipch->userEvent.sigev_code);
    LWSCIIPC_DBG_STR2INT(LIB_FUNC "priority, event",
        ipch->userEvent.sigev_priority,
        ipch->userEvent.sigev_value.sival_int);
#endif

    LWOS_COV_WHITELIST(deviate, LWOS_MISRA(Rule, 10_1), "<QNXBSP>:<qnx_asil_header>:<2>:<TID-257>") LWOS_COV_WHITELIST(deviate, LWOS_MISRA(Rule, 12_3), "<QNXBSP>:<qnx_asil_header>:<3>:<Bug 200736827>") LWOS_COV_WHITELIST(deviate, LWOS_MISRA(Rule, 13_4), "<QNXBSP>:<qnx_asil_header>:<4>:<Bug 200736827>")
    SIGEV_MEMORY_INIT(&ipch->setResetFlagEvent, &ipch->pulseResetFlag, 1,
        SIGEV_MEM_ASSIGN);

    /* EAGAIN, EPERM, ENOTSUP */
    chid = ChannelCreate_r(0U);
    if (chid < 0) {
        ret = LwSciError_IlwalidState;
        LWSCIIPC_ERR_STRINT("error: " LIB_FUNC "ChannelCreate_r: chid", chid);
        goto fail;
    }
    ipch->serverChid = chid;

    ret = lwsciipc_ipc_register_events(ipch);
    if (ret != LwSciError_Success) {
        goto fail;
    }

#if DEBUG_STATE
    LWSCIIPC_DBG_STR2INT(LIB_FUNC "calls Phase1Push: vuid, pid",
        ipch->entry->vuid, ipch->pid);
#endif
    ret = lwsciipc_ipc_pulse_phase1_push(ipch->pulsefd,
        ipch->regProgressEvent, ipch->regSetResetFlagEvent, ipch->serverChid);
    if (ret != LwSciError_Success) {
        ret = LwSciError_IlwalidState;
        goto fail;
    }
    ipch->rmState = (int32_t)RM_STATE_PHASE1PUSHED;

    ipch->coid = coid;

    LWSCIIPC_DBG_STR2INT(LIB_FUNC "done: vuid, pid",
        ipch->entry->vuid, ipch->pid);

    ret = LwSciError_Success;

fail:
    if (ret != LwSciError_Success) {
        /* should not override existing error */
        err = lwsciipc_ipc_release_event_handler(ipch);
        log_os_scierr(err, "release event handler");
    }

    return ret;
}

/**
 * @brief Configures the pulse to be generated when this native event
 *        oclwrs (Intra-VM)
 *
 * This function is QNX OS specific and called in LwSciIpcGetEventNotifier()
 * as callback of LwSciEventService infra structure.
 * This callback is registered in LwSciIpcOpenEndpointWithEventService().
 *
 * @param[in] thisNativeEvent LwSciNativeEvent object pointer
 * @param[in] coid          The connection ID created from calling
 *                          @c ConnectAttach_r().
 * @param[in] pulsePriority The value for pulse priority.
 * @param[in] pulseCode     The 8-bit positive pulse code specified by the user.
 *                          The values must be between @c _PULSE_CODE_MINAVAIL
 *                          and @c _PULSE_CODE_MAXAVAIL
 * @param[in] pulseValue    A pointer to the user-defined pulse value.
 *
 * @return ::LwSciError, the completion code of the operation:
 * - ::LwSciError_Success         Indicates a successful operation.
 * - ::LwSciError_ResourceError   Indicates not enough system resources.
 * - ::LwSciError_IlwalidState    Indicates an invalid operation state.
 */
LwSciError lwsciipc_ipc_configure_pulse_param(
    LwSciNativeEvent *thisNativeEvent,
    int32_t coid, int16_t priority, int16_t code, void *value)
{
    struct LwSciQnxNativeEvent *qnxEvent =
        (struct LwSciQnxNativeEvent *)(void *)thisNativeEvent;
    struct lwsciipc_ipc_handle *ipch =
        (struct lwsciipc_ipc_handle *)(void *)qnxEvent->handle;
    LwSciError ret;

    LWSCIIPC_DBG_STR("lwsciipc_ipc_configure_pulse_param: enter");
    ret = lwsciipc_ipc_set_qnx_pulse_param(ipch, coid, priority, code, value);

    LWSCIIPC_DBG_STR("lwsciipc_ipc_configure_pulse_param: exit");

    return ret;
}

/**
 * @brief Undo a previous ConfigurePulseParams call (Intra-VM)
 *
 * This function is QNX OS specific and called in Delete() callback of
 * LwSciEventNotifier object.
 *
 * @param[in] thisNativeEvent LwSciNativeEvent object pointer
 *
 * @pre Invocation of lwsciipc_ipc_configure_pulse_param() is successful.
 */
void lwsciipc_ipc_unconfigure_pulse_param(
    LwSciNativeEvent *thisNativeEvent)
{
    struct LwSciQnxNativeEvent *qnxEvent =
        (struct LwSciQnxNativeEvent *)(void *)thisNativeEvent;
    struct lwsciipc_ipc_handle *ipch =
        (struct lwsciipc_ipc_handle *)(void *)qnxEvent->handle;

    /* current function is void */
    (void)lwsciipc_ipc_release_event_handler(ipch);
}

LwSciError lwsciipc_ipc_endpoint_get_auth_token(
    struct lwsciipc_ipc_handle *ipch, LwSciIpcEndpointAuthToken *authToken)
{
    *authToken = (LwSciIpcEndpointAuthToken)(ipch->entry->vuid);
    return LwSciError_Success;
}

LwSciError lwsciipc_ipc_endpoint_get_vuid(
    struct lwsciipc_ipc_handle *ipch, LwSciIpcEndpointVuid *vuid)
{
    *vuid = (LwSciIpcEndpointVuid)(ipch->entry->vuid);
    return LwSciError_Success;
}

