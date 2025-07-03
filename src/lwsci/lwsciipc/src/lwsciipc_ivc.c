/*
 * Copyright (c) 2018-2021, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 */

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

#ifdef __QNX__
#include <sys/neutrino.h>
#include <inttypes.h>
#include <devctl.h>
#include <assert.h>
#include <lwqnx_common.h>
#endif /* __QNX__ */
#include <lwos_static_analysis.h>

#include "lwsciipc_common.h"
#include "lwsciipc_os_error.h"
#include "lwsciipc_ivc.h"
#include "lwsciipc_log.h"

#define IVC_DEV_ID_LEN 8

#if (LW_IS_SAFETY == 0) && defined(IVC_EVENTLIB)
void (*LwSciIpcEventlibNotify)(uint32_t id);
#endif

static void lwsciipc_ivc_notify(struct sivc_queue *sivc);
#ifdef __QNX__
static LwSciError lwsciipc_ivc_release_event_handle(
    struct lwsciipc_ivc_handle *ivch);
#endif /* __QNX__ */
static void lwsciipc_ivc_close_endpoint_internal(
    struct lwsciipc_ivc_handle *ivch);

/**
 * @brief Send notification to peer endpoint process (inter-VM)
 *
 * This function is registered as a callback in sivc_init() and
 * it's called in IVC core library context.
 * This notification callback is called whenever IVC connection status or
 * buffer status is changed (RX full to non-full, TX empty to non-empty).
 *
 * @param[in] pointer of IVC instance handle
 */
static void lwsciipc_ivc_notify(struct sivc_queue *sivc)
{
    /* ivch extends sivc */
    const struct lwsciipc_ivc_handle *ivch =
        (const struct lwsciipc_ivc_handle *)(void *)sivc;
    int32_t ret = 0;

    if (sivc == NULL) {
        LWSCIIPC_ERR_STR("error: lwsciipc_ivc_notify: IVC handle is NULL");
        goto fail;
    }

    /* raise_irq to other guest id */
#ifdef __QNX__
    const struct lwsciipc_ivc_info *ivci = &(ivch->ivc_info);
#if (LW_IS_SAFETY == 0) && defined(IVC_EVENTLIB)
    if (LwSciIpcEventlibNotify != NULL) {
        LwSciIpcEventlibNotify(ivch->entry->id);
    }
#endif
    /* trigger interrupt to peer in inter VM */
    (*ivch->noti_va) = ivci->noti_irq;
#endif /* __QNX__ */

#ifdef LINUX
    const struct lwsciipc_ivc_info *ivci = &(ivch->ivc_info);
    (*ivch->noti_va) = ivci->noti_irq; /* trigger interrupt to peer in inter VM */
#endif /* LINUX */

    if (ret != EOK) {
        LWSCIIPC_ERR_STRINT(
            "error: lwsciipc_ivc_notify: ioctl failed: ret", ret);
    }
fail:
    return;
}

LwSciError lwsciipc_ivc_get_endpoint_info(
    const struct lwsciipc_ivc_handle *ivch,
    LwSciIpcEndpointInfo *info)
{
#ifdef __QNX__
    info->nframes = ivch->entry->nFrames;
    info->frame_size = ivch->entry->frameSize;
#endif
#ifdef LINUX
    info->nframes = ivch->sivc.nframes;
    info->frame_size = ivch->sivc.frame_size;
#endif
    return LwSciError_Success;
}

#ifdef __QNX__
LwSciError lwsciipc_ivc_get_endpoint_info_internal(
    const struct lwsciipc_ivc_handle *ivch,
    LwSciIpcEndpointInfoInternal *info)
{
    LwSciError ret;

    if (ivch->entry->irq == LWSCIIPC_IRQ_ILWALID) {
        LWSCIIPC_ERR_STR(
            "error: lwsciipc_ivc_get_endpoint_info_internal: "
            "Abnormal interrupt value");
        ret = LwSciError_IlwalidState;
    }
    else {
        info->irq = ivch->entry->irq;
        ret = LwSciError_Success;
    }

    return ret;
}
#endif /* __QNX__ */

LwSciError lwsciipc_ivc_get_eventnotifier(
    struct lwsciipc_ivc_handle *ivch,
    LwSciEventNotifier **eventNotifier)
{
    LwSciEventNotifier *notifier;
    LwSciError ret;

    if (ivch->eventService == NULL) {
        LWSCIIPC_ERR_STR(
            "error: lwsciipc_ivc_get_eventnotifier: "
            "EventService is NOT initialized");
        ret = LwSciError_NotInitialized;
        goto fail;
    }

#ifdef LINUX
    ivch->nativeEvent.fd = ivch->fd;
    ret = ivch->eventService->CreateNativeEventNotifier(ivch->eventService,
         &ivch->nativeEvent, &notifier);
#endif
#ifdef __QNX__
    ret = ivch->eventService->CreateNativeEventNotifier(ivch->eventService,
         &ivch->qnxEvent.nativeEvent, &notifier);
#endif /* __QNX__ */
    if (ret != LwSciError_Success) {
        goto fail;
    }

    *eventNotifier = (LwSciEventNotifier *)notifier;
fail:
    return ret;
}

#ifdef LINUX
LwSciError lwsciipc_ivc_get_eventfd(const struct lwsciipc_ivc_handle *ivch,
    int32_t *fd)
{
    *fd = ivch->fd;
    return LwSciError_Success;
}
#endif

LwSciError lwsciipc_ivc_read(struct lwsciipc_ivc_handle *ivch, void *buf,
    uint32_t size, uint32_t *bytes)
{
    int32_t err;
    LwSciError ret;

    err = sivc_read(&ivch->sivc, buf, size);

    if (err < 0) {
        *bytes = 0U;
        update_sivc_err(err);
    } else {
        *bytes = size;
        ret = LwSciError_Success;
    }

    return ret;
}

const volatile void *lwsciipc_ivc_read_get_next_frame(
    struct lwsciipc_ivc_handle *ivch)
{
    return sivc_get_read_frame(&ivch->sivc);
}

LwSciError lwsciipc_ivc_read_advance(struct lwsciipc_ivc_handle *ivch)
{
    int32_t err;
    LwSciError ret;

    err = sivc_read_advance(&ivch->sivc);

    if (err < 0) {
        update_sivc_err(err);
    } else {
        ret = LwSciError_Success;
    }

    return ret;
}

LwSciError lwsciipc_ivc_write(struct lwsciipc_ivc_handle *ivch,
    const void *buf, uint32_t size, uint32_t *bytes)
{
    int32_t err;
    LwSciError ret;

    err = sivc_write(&ivch->sivc, buf, size);

    if (err < 0) {
        *bytes = 0U;
        update_sivc_err(err);
    } else {
        *bytes = size;
        ret = LwSciError_Success;
    }

    return ret;
}

volatile void *lwsciipc_ivc_write_get_next_frame(
    struct lwsciipc_ivc_handle *ivch)
{
    return sivc_get_write_frame(&ivch->sivc);
}

LwSciError lwsciipc_ivc_write_advance(struct lwsciipc_ivc_handle *ivch)
{
    int32_t err;
    LwSciError ret;

    err = sivc_write_advance(&ivch->sivc);

    if (err < 0) {
        update_sivc_err(err);
    } else {
        ret = LwSciError_Success;
    }

    return ret;
}

/* peek in the next rx buffer at offset off, the count bytes */
LwSciError lwsciipc_ivc_read_peek(struct lwsciipc_ivc_handle *ivch, void *buf,
    uint32_t offset, uint32_t count, uint32_t *bytes)
{
    int32_t err;
    LwSciError ret;

    err = sivc_read_peek(&ivch->sivc, buf, offset, count);

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
LwSciError lwsciipc_ivc_write_poke(struct lwsciipc_ivc_handle *ivch,
    const void *buf, uint32_t offset, uint32_t count, uint32_t *bytes)
{
    int32_t err;
    LwSciError ret;

    err = sivc_write_poke(&ivch->sivc, buf, offset, count);

    if (err < 0) {
        *bytes = 0U;
        update_sivc_err(err);
    } else {
        *bytes = count;
        ret = LwSciError_Success;
    }

    return ret;
}

static LwSciError lwsciipc_ivc_init_data(
    struct lwsciipc_ivc_handle *ivch,
    struct LwSciIpcConfigEntry *entry,
    uintptr_t *tx_base,
    uintptr_t *rx_base
)
{
#ifdef LINUX
    char dev_name[LWSCIIPC_MAX_ENDPOINT_NAME] = {0};
    int32_t err;
#endif /* LINUX */
    struct LwSciIpcConfigEntry noti_info;
    void *addr = NULL;
    struct lwsciipc_ivc_info *ivci;
    uint64_t u64val;
    LwSciError ret = LwSciError_Success;
    bool flag;

#ifdef LINUX
    (void)sprintf(dev_name, "%s%u", IVC_DEV_NAME, entry->id);

    LWSCIIPC_DBG_STRINT("lwsciipc_ivc_init_data: opening /dev/ivc: id",
        entry->id);
    LWOS_COV_WHITELIST(deviate, LWOS_CERT(FIO32_C), "<QNXBSP>:<lwpu>:<1>:<Bug 200736827>")
    err = open(dev_name, O_RDWR);
    if (err == -1) {
        LWSCIIPC_ERR_STRUINT(
            "error: lwsciipc_ivc_init_data: Failed to open /dev/ivc: id",
            entry->id);
        LWSCIIPC_ERR_STRINT("error: lwsciipc_ivc_init_data: "
            "Failed to open IVC device: ret", errno);
        ret = LwSciError_NotPermitted;
        goto fail;
    }
    ivch->fd = err;
#endif /* LINUX */
    ivch->entry = entry;
    ivch->pid = getpid();
#ifdef LINUX
    /* get queue data from IVC device */
    err = lwsciipc_os_ioctl(ivch->fd, LW_SCI_IPC_IVC_IOCTL_GET_INFO, ivch);
    if (err != EOK) {
        ret = ResmgrErrnoToLwSciErr(err);
        LWSCIIPC_ERR_STRINT("error: lwsciipc_ivc_init_data: "
            "ioctl failed", err);
        goto fail;
    }
#endif /* LINUX */

    ivci = &ivch->ivc_info;
#ifdef __QNX__
    ivch->iid = -1;

    /* fill ivci to use common code with Linux */
    ivci->nframes = entry->nFrames;
    ivci->frame_size = entry->frameSize;
    ivci->queue_offset = 0;
    ivci->queue_size = entry->qsize;

    flag = CastU64toU32(entry->chPsize, &ivci->area_size);
    if (flag == false) {
        LWSCIIPC_ERR_STRULONG("error: lwsciipc_ivc_init_data: chPsize",
            entry->chPsize);
        ret = LwSciError_IlwalidState;
        goto fail;
    }

    LWOS_COV_WHITELIST(deviate, LWOS_MISRA(Rule, 10_5), "<QNXBSP>:<lwpu>:<1>:<Bug 200736827>")
    ivci->rx_first = (bool)entry->rxFirst;
    ivci->noti_irq = entry->irq;
#endif /* __QNX__ */

    /* Map the share memory area */
    addr = lwsciipc_os_mmap(NULL, ivci->area_size,
#ifdef __QNX__
            /* no overflow since flag definition is 0x300U */
            CastU32toS32WithExit(LW_PROT_READ|LW_PROT_WRITE),
#endif /* __QNX__ */
#ifdef LINUX
            PROT_READ|PROT_WRITE,
#endif /* LINUX */
            MAP_SHARED, ivch->fd, 0, entry);

    LWOS_COV_WHITELIST(deviate, LWOS_MISRA(Rule, 11_6), "<QNXBSP>:<qnx_asil_header>:<1>:<TID-371>") LWOS_COV_WHITELIST(deviate, LWOS_CERT(INT36_C), "<QNXBSP>:<lwpu>:<1>:<Bug 200736827>")
    if (addr == MAP_FAILED) {
        LWSCIIPC_ERR_STR(
            "error: lwsciipc_ivc_init_data: Failed to map IVC area");
        ret = LwSciError_InsufficientMemory;
        goto fail;
    }
    LWOS_COV_WHITELIST(deviate, LWOS_MISRA(Rule, 11_6), "<QNXBSP>:<qnx_asil_header>:<1>:<TID-371>")
    ivch->shm = (uintptr_t)addr;

    /* set the first channel memory base */
    flag = AddU64((uint64_t)ivch->shm, (uint64_t)ivci->queue_offset, &u64val);
    if (flag == false) {
        LWSCIIPC_ERR_STR2ULONG("error: lwsciipc_ivc_init_data: "
            "shm, queue_offset",
            (uint64_t)ivch->shm, (uint64_t)ivci->queue_offset);
        ret = LwSciError_IlwalidState;
        goto fail;
    }

    if (ivci->rx_first) {
        *rx_base = (uintptr_t)u64val;
    } else {
        *tx_base = (uintptr_t)u64val;
    }

    /* set opposite direction channel memory base */
    flag = AddU64(u64val, (uint64_t)ivci->queue_size, &u64val);
    if (flag == false) {
        LWSCIIPC_ERR_STR2ULONG("error: lwsciipc_ivc_init_data: "
            "base, queue_size",
            u64val, (uint64_t)ivci->queue_size);
        ret = LwSciError_IlwalidState;
        goto fail;
    }

#ifdef LINUX
    noti_info.chPaddr = ivci->noti_ipa;
    noti_info.chPsize = NOTI_VA_SIZE;

    addr = lwsciipc_os_mmap(NULL, NOTI_VA_SIZE,
            PROT_READ|PROT_WRITE,
            MAP_SHARED, ivch->fd, ivci->area_size, &noti_info);
#endif /* LINUX */
#ifdef __QNX__
    noti_info.chPaddr = entry->notiIpa;
    noti_info.chPsize = entry->notiIpaSize;

    addr = lwsciipc_os_mmap(NULL, noti_info.chPsize,
            /* no overflow since flag definition is 0x300U */
            CastU32toS32WithExit(LW_PROT_READ|LW_PROT_WRITE),
            MAP_SHARED, 0, 0, &noti_info);
#endif /* __QNX__ */

    LWOS_COV_WHITELIST(deviate, LWOS_MISRA(Rule, 11_6), "<QNXBSP>:<qnx_asil_header>:<1>:<TID-371>")
    if (addr == MAP_FAILED) {
        LWSCIIPC_ERR_STR(
            "error: lwsciipc_ivc_init_data: Failed to map trap area");
        ret = LwSciError_InsufficientMemory;
        goto fail;
    }
    ivch->noti_va = addr;
#ifdef __QNX__
    ivch->noti_vsz = noti_info.chPsize;
#endif /* __QNX__ */

    if (ivci->rx_first) {
        *tx_base = (uintptr_t)u64val;
    } else {
        *rx_base = (uintptr_t)u64val;
    }

fail:
    return ret;
}

static void lwsciipc_ivc_close_endpoint_internal(
    struct lwsciipc_ivc_handle *ivch)
{
#undef LIB_FUNC
#define LIB_FUNC "lwsciipc_ivc_close_endpoint_internal: "
    const struct lwsciipc_ivc_info *ivci = &ivch->ivc_info;
    int32_t ret = 0;

    if (ivch != NULL) {
#if defined(__QNX__)
        /* release event handle. it doesn't handle error here */
        (void)lwsciipc_ivc_release_event_handle(ivch);
#endif /* __QNX__ */

        if (ivch->shm != 0U) {
            LWOS_COV_WHITELIST(deviate, LWOS_MISRA(Rule, 11_6), "<QNXBSP>:<qnx_asil_header>:<1>:<TID-371>")
            ret = lwsciipc_os_munmap((void *)ivch->shm, ivci->area_size);
            if (ret == -1) {
                LWSCIIPC_ERR_STR2ULONG( "error: " LIB_FUNC
                    "Failed to unmap: shm, size",
                    (uint64_t)ivch->shm, (uint64_t)ivci->area_size);
            }
            ivch->shm = 0U;
        }

        if (ivch->fd != 0) {
            ret = close(ivch->fd);
            if (EOK != ret) {
                LWSCIIPC_ERR_STRINT("error: " LIB_FUNC "close", errno);
            }
            ivch->fd = 0;
        }

        if (ivch->noti_va != NULL) {
#if defined(__QNX__)
            uint64_t size = ivch->noti_vsz;
#endif /* __QNX__ */
#if defined(LINUX)
            uint64_t size = NOTI_VA_SIZE;
#endif /* LINuX */

            LWOS_COV_WHITELIST(deviate, LWOS_MISRA(Rule, 11_6), "<QNXBSP>:<qnx_asil_header>:<1>:<TID-371>")
            ret = lwsciipc_os_munmap((void *)ivch->noti_va, size);
            if (ret == -1) {
                LWSCIIPC_ERR_STRULONG(
                    "error: lwsciipc_ivc_close_endpoint_internal: "
                    "Failed to unmap: noti IPA region",
                    (uint64_t)ivch->noti_va);
            }
        }
        lwsciipc_os_put_endpoint_mutex(&ivch->mutexfd);
    }
}

/**
 * Do internal job for opening inter-VM endpoint by communicating LwSciIpc and
 * IVC resource manager.
 * It gets mutex for an endpoint from LwSciIpc resource manager if the endpoint
 * is not oclwpied and gets authentication for sending inter-VM notification
 * from IVC resource manager. After it succeeds in getting both mutex and
 * authentication, it initializes ivclib.
 *
 * @param[in]  entry  Configuration structure handle
 * @param[out] ivcp   Internal inter-VM handle
 *
 * @return ::LwSciError, the completion code of the operation:
 * - ::LwSciError_Success            Indicates a successful operation.
 * - ::LwSciError_BadParameter       Indicates any argument was NULL or invalid.
 * - ::LwSciError_Busy               Indicates the @a endpoint is already in use.
 * - ::LwSciError_InsufficientMemory Indicates memory allocation failed for the
 *                                   operation.
 * - ::LwSciError_IlwalidState       Indicates an invalid operation state.
 * - ::LwSciError_NotPermitted       Indicates process doesn't have the required
 *                                   privilege.
 */
LwSciError lwsciipc_ivc_open_endpoint(struct lwsciipc_ivc_handle **ivcp,
    struct LwSciIpcConfigEntry *entry)
{
    struct lwsciipc_ivc_handle *ivch = NULL;
    uintptr_t tx_base = (uintptr_t)0;
    uintptr_t rx_base = (uintptr_t)0;
    struct lwsciipc_ivc_info *ivci;
    int32_t err;
    LwSciError ret = LwSciError_Unknown;

    LWOS_COV_WHITELIST(deviate, LWOS_MISRA(Rule, 21_3), "<QNXBSP>:<lwpu>:<1>:<TID-385>")
    ivch = (struct lwsciipc_ivc_handle *)calloc(1,
        sizeof(struct lwsciipc_ivc_handle));
    if (ivch == NULL) {
        LWSCIIPC_ERR_STR("error: lwsciipc_ivc_open_endpoint: Failed to calloc");
        ret = LwSciError_InsufficientMemory;
        goto fail;
    }

    ret = lwsciipc_os_get_endpoint_mutex(entry, &ivch->mutexfd);
    if (ret != LwSciError_Success) {
        goto fail;
    }

    ret = lwsciipc_ivc_init_data(ivch, entry, &tx_base, &rx_base);
    if (ret != LwSciError_Success) {
        goto fail;
    }

    ivci = &ivch->ivc_info;
    /* cache_ilwalidate and cache_flush callbacks are not used */
    err = sivc_init(&ivch->sivc, rx_base, tx_base,
            ivci->nframes, ivci->frame_size, lwsciipc_ivc_notify, NULL, NULL);
    if (err < 0) {
        ret = ErrnoToLwSciErr(err);
        LWSCIIPC_ERR_STRINT(
            "error: lwsciipc_ivc_open_endpoint: ivc_init failed: ret",
            (int32_t)ret);
        goto fail;
    }

    *ivcp = ivch;

    LWSCIIPC_DBG_STR("lwsciipc_ivc_open_endpoint: exit");

    ret = LwSciError_Success; /* success */
    goto done;
fail:
    if (ivch != NULL) {
        lwsciipc_ivc_close_endpoint_internal(ivch);
        LWOS_COV_WHITELIST(deviate, LWOS_MISRA(Rule, 21_3), "<QNXBSP>:<lwpu>:<1>:<TID-385>")
        free(ivch);
    }
    *ivcp = NULL;

done:
    /* Both ivch->fd and ivch->mutexfd are released in
     * lwsciipc_ivc_close_endpoint_internal().
     * ivch is released in lwsciipc_ivc_close_endpoint().
     */
    LWOS_COV_WHITELIST(deviate, LWOS_MISRA(Rule, 4_13), "<QNXBSP>:<qnx_asil_header>:<1>:<Bug 200736827>")
    return ret;
}

void lwsciipc_ivc_bind_eventservice(struct lwsciipc_ivc_handle *ivch,
    LwSciEventService *eventService)
{
#undef LIB_FUNC
#define LIB_FUNC "lwsciipc_ivc_bind_eventservice: "
    LWSCIIPC_DBG_STR(LIB_FUNC "enter");

    ivch->eventService = eventService;
#ifdef __QNX__
    ivch->qnxEvent.handle = ivch;

    ivch->qnxEvent.nativeEvent.ConfigurePulseParams =
        lwsciipc_ivc_configure_pulse_param;
    ivch->qnxEvent.nativeEvent.UnconfigurePulseParams =
        lwsciipc_ivc_unconfigure_pulse_param;
    ivch->qnxEvent.nativeEvent.UnmaskInterrupt =
        lwsciipc_ivc_unmask_interrupt;
#endif /* __QNX__ */

    LWSCIIPC_DBG_STR(LIB_FUNC "exit");
}

LwSciError lwsciipc_ivc_open_endpoint_with_eventservice(
    struct lwsciipc_ivc_handle **ivcp,
    struct LwSciIpcConfigEntry *entry,
    LwSciEventService *eventService)
{
#undef LIB_FUNC
#define LIB_FUNC "lwsciipc_ivc_open_endpoint_with_eventservice: "
    struct lwsciipc_ivc_handle *ivch = NULL;
    LwSciError ret;

    LWSCIIPC_DBG_STR(LIB_FUNC "enter");

    ret =  lwsciipc_ivc_open_endpoint(ivcp, entry);
    if (ret == LwSciError_Success) {
        ivch = *ivcp;
        lwsciipc_ivc_bind_eventservice(ivch, eventService);
    }
    else {
        goto fail;
    }

fail:
    LWSCIIPC_DBG_STR(LIB_FUNC "exit");
    return ret;
}

void lwsciipc_ivc_close_endpoint(struct lwsciipc_ivc_handle *ivch)
{
    LWSCIIPC_DBG_STR("lwsciipc_ivc_close_endpoint: enter");

    lwsciipc_ivc_close_endpoint_internal(ivch);
    LWOS_COV_WHITELIST(deviate, LWOS_MISRA(Rule, 21_3), "<QNXBSP>:<lwpu>:<1>:<TID-385>")
    free(ivch);

    LWSCIIPC_DBG_STR("lwsciipc_ivc_close_endpoint: exit");
}

void lwsciipc_ivc_reset_endpoint(struct lwsciipc_ivc_handle *ivch)
{
    LWSCIIPC_DBG_STR("lwsciipc_ivc_reset_endpoint: enter");

    sivc_reset(&ivch->sivc);

    LWSCIIPC_DBG_STR("lwsciipc_ivc_reset_endpoint: exit");
}

#ifdef __QNX__
/**
 * @brief Undo a previous ConfigurePulseParams call (Inter-VM)
 *
 * @param[in] ivch  Inter-VM handle
 *
 * @return ::LwSciError, the completion code of the operation:
 * - ::LwSciError_Success         Indicates a successful operation.
 * - ::LwSciError_IlwalidState    Indicates an invalid operation state.
 */
static LwSciError lwsciipc_ivc_release_event_handle(
    struct lwsciipc_ivc_handle *ivch)
{
#undef LIB_FUNC
#define LIB_FUNC "lwsciipc_ivc_release_event_handle: "
    LwSciError ret = LwSciError_IlwalidState;
    int32_t err;

    LWSCIIPC_DBG_STR(LIB_FUNC "enter");
    if (ivch->iid != -1) {
        err = InterruptDetach_r(ivch->iid);
        report_os_errto(err, "IntrruptDetach_r", fail);
        ivch->iid = -1;
    }

    LWSCIIPC_DBG_STR(LIB_FUNC "exit");

    ret = LwSciError_Success;

fail:
    return ret;
}
#endif /* __QNX__ */

/* This check function always return error since it's called
 * only when following IVC apis return NULL pointer.
 * tegra_ivc_read_get_next_frame()
 * tegra_ivc_write_get_next_frame()
 */
LwSciError lwsciipc_ivc_check_read(struct lwsciipc_ivc_handle *ivch)
{
    LwSciError ret;

    if (ivch->prev_conn == LW_SCI_IPC_EVENT_CONN_RESET) {
        ret = LwSciError_ConnectionReset;
    }
    else {
        ret = LwSciError_InsufficientMemory;
    }

    return ret;
}

LwSciError lwsciipc_ivc_check_write(struct lwsciipc_ivc_handle *ivch)
{
    return lwsciipc_ivc_check_read(ivch);
}

#ifdef __QNX__
/* interrupt unmask routine for QNX OS */
static LwSciError lwsciipc_ivc_get_event_qnx(struct lwsciipc_ivc_handle *ivch)
{
    int32_t err;
    LwSciError ret = LwSciError_Success;

    if ((ivch->iid != -1) && (ivch->eventService == NULL)) {
        /* interrupt typecast error */
        if (ivch->entry->irq > (uint32_t)INT32_MAX) {
            LWSCIIPC_ERR_STRUINT("error: lwsciipc_ivc_get_event: irq",
                ivch->entry->irq);
            ret = LwSciError_IlwalidState;
            goto fail;
        }
        /* interrupt is attached by set_qnx_pulse_param */
        err = InterruptUnmask((int32_t)ivch->entry->irq, ivch->iid);
        if (err == -1) {
            LWSCIIPC_ERR_STRINT("error: lwsciipc_ivc_get_event: "
                "failed to unmask intr: err", errno);
            ret = LwSciError_IlwalidState;
            goto fail;
        }
    }

fail:
    return ret;
}
#endif /* __QNX__ */

/**
 * Get inter-VM events that have been triggered.
 * In callback routine, you don't need to set LW_SCI_IPC_EVENT_CONN_MASK.
 * it's already read in event interrupt thread.
 * This function returns success by default.
 * It returns error only when mutex call fails.
 *
 * @param[in]    inth   LwSciIpc internal object pointer
 * @param[inout] ivch   Inter-VM handle
 * @param[out]   events Events that has arrived
 *
 * @return ::LwSciError, the completion code of the operation:
 * - ::LwSciError_Success         Indicates a successful operation.
 * - ::LwSciError_IlwalidState    Indicates an invalid operation state.
 */
LwSciError lwsciipc_ivc_get_event(struct lwsciipc_ivc_handle *ivch,
    uint32_t *events, struct lwsciipc_internal_handle *inth)
{
    uint32_t value = 0U;
    uint32_t conn = 0U;
    LwSciError ret = LwSciError_Success;
    int32_t err = EAGAIN;
    int32_t merr;

#ifdef __QNX__
    ret = lwsciipc_ivc_get_event_qnx(ivch);
    if (ret != LwSciError_Success) {
        goto fail;
    }
#endif /* __QNX__ */

    if (sivc_need_notify(&ivch->sivc) == false) {
        if (sivc_can_write(&ivch->sivc)) {
            value |= LW_SCI_IPC_EVENT_WRITE;
        }
        if (sivc_can_read(&ivch->sivc)) {
            value |= LW_SCI_IPC_EVENT_READ;
        }
    }
    else {
        /* get connection status when either endpoint is NOT in
         * established state
         */
        merr = lwsciipc_os_mutex_lock(&inth->wrMutex); /* wr pos/refcnt */
        report_mutex_errto(merr, "WR mutex_lock", fail);
        merr = lwsciipc_os_mutex_lock(&inth->rdMutex); /* rd pos/refcnt */
        report_mutex_errto(merr, "RD mutex_lock", un_wr_mtx);

        err = sivc_notified(&ivch->sivc);

        merr = lwsciipc_os_mutex_unlock(&inth->rdMutex);

        report_mutex_errto(merr, "RD mutex_unlock", un_wr_mtx);
un_wr_mtx:
        merr = lwsciipc_os_mutex_unlock(&inth->wrMutex);
        update_mutex_err(merr, "WR mutex_unlock");
        if (ret != LwSciError_Success) {
            /* doesn't update events */
            goto fail;
        }

        if (err == EOK) {
            conn = LW_SCI_IPC_EVENT_CONN_EST;
            /* check buffer status again after establishment */
            if (sivc_can_write(&ivch->sivc)) {
                value |= LW_SCI_IPC_EVENT_WRITE;
            }
            if (sivc_can_read(&ivch->sivc)) {
                value |= LW_SCI_IPC_EVENT_READ;
            }
        }
        else {
            conn = LW_SCI_IPC_EVENT_CONN_RESET;
        }
    }

    if ((conn & LW_SCI_IPC_EVENT_CONN_MASK) != ivch->prev_conn) {
        *events = (value | conn);
        LWSCIIPC_DBG_STR2INT("lwsciipc_ivc_get_event: vuid, pid",
            ivch->entry->vuid, ivch->pid);
        LWSCIIPC_DBG_STR2INT("lwsciipc_ivc_get_event: prev_conn, conn",
            ivch->prev_conn, conn);
        LWSCIIPC_DBG_STRINT("lwsciipc_ivc_get_event: events",
            *events);
        ivch->prev_conn = conn;
    }
    else {
        *events = value;
    }
fail:
    return ret;
}

bool lwsciipc_ivc_can_read(struct lwsciipc_ivc_handle *ivch)
{
    return sivc_can_read(&ivch->sivc);
}

bool lwsciipc_ivc_can_write(struct lwsciipc_ivc_handle *ivch)
{
    return sivc_can_write(&ivch->sivc);
}

#ifdef __QNX__
/**
 * @brief Configures the pulse to be generated (Inter-VM)
 *
 * @param[in] ivch          Inter-VM handle
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
LwSciError lwsciipc_ivc_set_qnx_pulse_param(
    struct lwsciipc_ivc_handle *ivch, int32_t coid,
    int16_t priority, int16_t code, void *value)
{
    struct sigevent event;
    int32_t iid;
    LwSciError ret;

    LWSCIIPC_DBG_STR("lwsciipc_ivc_set_qnx_pulse_param: enter");

    if (ivch->coid != 0) {
        LWSCIIPC_ERR_STR(
            "error: lwsciipc_ivc_set_qnx_pulse_param: "
            "coid is already assigned to endpoint");
        ret = LwSciError_ResourceError;
        goto fail;
    }

    ivch->coid = coid;

    LWOS_COV_WHITELIST(deviate, LWOS_MISRA(Rule, 10_1), "<QNXBSP>:<qnx_asil_header>:<2>:<TID-257>") LWOS_COV_WHITELIST(deviate, LWOS_MISRA(Rule, 11_6), "<QNXBSP>:<qnx_asil_header>:<2>:<TID-371>") LWOS_COV_WHITELIST(deviate, LWOS_MISRA(Rule, 12_3), "<QNXBSP>:<qnx_asil_header>:<4>:<Bug 200736827>") LWOS_COV_WHITELIST(deviate, LWOS_MISRA(Rule, 13_4), "<QNXBSP>:<qnx_asil_header>:<5>:<Bug 200736827>")
    SIGEV_PULSE_INIT(&event, coid, priority, code, value);

    /* interrupt typecast error */
    if (ivch->entry->irq > (uint32_t)INT32_MAX) {
        LWSCIIPC_ERR_STRUINT("error: lwsciipc_ivc_set_qnx_pulse_param: irq",
            ivch->entry->irq);
        ret = LwSciError_IlwalidState;
        goto fail;
    }

    iid = InterruptAttachEvent_r((int32_t)ivch->entry->irq,
            &event, _NTO_INTR_FLAGS_TRK_MSK);
    if (iid < 0) {
        ivch->iid = -1;
        LWSCIIPC_ERR_STRINT("error: lwsciipc_ivc_set_qnx_pulse_param: "
            "InterruptAttachEvent: ret", iid);
        ret = LwSciError_IlwalidState;
        goto fail;
    }

    ivch->iid = iid;

    LWSCIIPC_DBG_STR2INT("lwsciipc_ivc_set_qnx_pulse_param: irq, iid",
        ivch->entry->irq, iid);

    LWSCIIPC_DBG_STR("lwsciipc_ivc_set_qnx_pulse_param: exit");

    ret = LwSciError_Success;

fail:
    if (ret != LwSciError_Success) {
        /* should not override existing error */
        (void)lwsciipc_ivc_release_event_handle(ivch);
    }

    return ret;

}

/**
 * @brief Configures the pulse to be generated when this native event
 *        oclwrs (Inter-VM)
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
LwSciError lwsciipc_ivc_configure_pulse_param(
    LwSciNativeEvent *thisNativeEvent,
    int32_t coid, int16_t priority, int16_t code, void *value)
{
    struct LwSciQnxNativeEvent *qnxEvent =
        (struct LwSciQnxNativeEvent *)(void *)thisNativeEvent;
    struct lwsciipc_ivc_handle *ivch =
        (struct lwsciipc_ivc_handle *)(void *)qnxEvent->handle;
    LwSciError ret;

    LWSCIIPC_DBG_STR("lwsciipc_ivc_configure_pulse_param: enter");

    ret = lwsciipc_ivc_set_qnx_pulse_param(ivch, coid, priority, code, value);

    LWSCIIPC_DBG_STR("lwsciipc_ivc_configure_pulse_param: exit");

    return ret;
}

/**
 * @brief Undo a previous ConfigurePulseParams call (Inter-VM)
 *
 * This function is QNX OS specific and called in Delete() callback of
 * LwSciEventNotifier object.
 *
 * @param[in] thisNativeEvent LwSciNativeEvent object pointer
 *
 * @pre Invocation of lwsciipc_ipc_configure_pulse_param() is successful.
 */
void lwsciipc_ivc_unconfigure_pulse_param(
    LwSciNativeEvent *thisNativeEvent)
{
    struct LwSciQnxNativeEvent *qnxEvent =
        (struct LwSciQnxNativeEvent *)(void *)thisNativeEvent;
    struct lwsciipc_ivc_handle *ivch =
        (struct lwsciipc_ivc_handle *)(void *)qnxEvent->handle;

    /* current function is void */
    (void)lwsciipc_ivc_release_event_handle(ivch);
}

/**
 * @brief Unmask interrupt (Inter-VM)
 *
 * Unmask interupt which gets triggered when a remote endpoint sends event
 * notification (inter-VM).
 * This function is QNX OS specific and called in WaitForEvent() or
 * WaitForMultpleEvent callback of LwSciEventLoopService object.
 *
 * @param[in] thisNativeEvent LwSciNativeEvent object pointer
 *
 * @return ::LwSciError, the completion code of the operation:
 * - ::LwSciError_Success         Indicates a successful operation.
 * - ::LwSciError_IlwalidState       Indicates an invalid operation state.
 */
LwSciError lwsciipc_ivc_unmask_interrupt(
    LwSciNativeEvent *thisNativeEvent)
{
    struct LwSciQnxNativeEvent *qnxEvent =
        (struct LwSciQnxNativeEvent *)(void *)thisNativeEvent;
    struct lwsciipc_ivc_handle *ivch =
        (struct lwsciipc_ivc_handle *)(void *)qnxEvent->handle;
    LwSciError ret = LwSciError_IlwalidState;
    int32_t err;

    /* interrupt typecast error */
    if (ivch->entry->irq > (uint32_t)INT32_MAX) {
        LWSCIIPC_ERR_STRUINT("error: lwsciipc_ivc_unmask_interrupt: irq",
            ivch->entry->irq);
        ret = LwSciError_IlwalidState;
        goto fail;
    }

    err = InterruptUnmask((int32_t)ivch->entry->irq, ivch->iid);
    if (-1 == err) {
        LWSCIIPC_ERR_STRINT("error: lwsciipc_ivc_unmask_interrupt: "
            "failed to unmask intr: err", errno);
    }
    else {
        ret = LwSciError_Success;
    }
fail:
    return ret;
}
#endif /* __QNX__ */

LwSciError lwsciipc_ivc_endpoint_get_auth_token(
    struct lwsciipc_ivc_handle *ivch, LwSciIpcEndpointAuthToken *authToken)
{
#ifdef __QNX__
    *authToken = (LwSciIpcEndpointAuthToken)(ivch->entry->vuid);
    return LwSciError_Success;
#endif /* __QNX__ */
#ifdef LINUX
    *authToken = LWSCIIPC_ENDPOINT_AUTHTOKEN_ILWALID;
    return LwSciError_NotSupported;
#endif /* LINUX */
}

LwSciError lwsciipc_ivc_endpoint_get_vuid(
    struct lwsciipc_ivc_handle *ivch, LwSciIpcEndpointVuid *vuid)
{
#ifdef __QNX__
    *vuid = (LwSciIpcEndpointVuid)(ivch->entry->vuid);
    return LwSciError_Success;
#endif /* __QNX__ */
#ifdef LINUX
    *vuid = LWSCIIPC_ENDPOINT_VUID_ILWALID;
    return LwSciError_NotSupported;
#endif /* LINUX */
}

