/**
 * Copyright (C) Mellanox Technologies Ltd. 2018-2019.  ALL RIGHTS RESERVED.
 * Copyright (c) 2018, LWPU CORPORATION. All rights reserved.
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "lwda_ipc_iface.h"
#include "lwda_ipc_md.h"
#include "lwda_ipc_ep.h"

#include <uct/lwca/base/lwda_iface.h>
#include <ucs/type/class.h>
#include <ucs/sys/string.h>
#include <sys/eventfd.h>

static ucs_config_field_t uct_lwda_ipc_iface_config_table[] = {

    {"", "", NULL,
     ucs_offsetof(uct_lwda_ipc_iface_config_t, super),
     UCS_CONFIG_TYPE_TABLE(uct_iface_config_table)},

    {"MAX_POLL", "16",
     "Max number of event completions to pick during lwca events polling",
      ucs_offsetof(uct_lwda_ipc_iface_config_t, max_poll), UCS_CONFIG_TYPE_UINT},

    {"MAX_STREAMS", "16",
     "Max number of LWCA streams to make conlwrrent progress on",
      ucs_offsetof(uct_lwda_ipc_iface_config_t, max_streams), UCS_CONFIG_TYPE_UINT},

    {"CACHE", "y",
     "Enable remote endpoint IPC memhandle mapping cache",
     ucs_offsetof(uct_lwda_ipc_iface_config_t, enable_cache),
     UCS_CONFIG_TYPE_BOOL},

    {"MAX_EVENTS", "inf",
     "Max number of lwca events. -1 is infinite",
     ucs_offsetof(uct_lwda_ipc_iface_config_t, max_lwda_ipc_events), UCS_CONFIG_TYPE_UINT},

    {NULL}
};


/* Forward declaration for the delete function */
static void UCS_CLASS_DELETE_FUNC_NAME(uct_lwda_ipc_iface_t)(uct_iface_t*);


static uint64_t uct_lwda_ipc_iface_node_guid(uct_base_iface_t *iface)
{
    return ucs_machine_guid() *
           ucs_string_to_id(iface->md->component->name);
}

ucs_status_t uct_lwda_ipc_iface_get_device_address(uct_iface_t *tl_iface,
                                                   uct_device_addr_t *addr)
{
    uct_base_iface_t *iface = ucs_derived_of(tl_iface, uct_base_iface_t);

    *(uint64_t*)addr = uct_lwda_ipc_iface_node_guid(iface);
    return UCS_OK;
}

static ucs_status_t uct_lwda_ipc_iface_get_address(uct_iface_h tl_iface,
                                                   uct_iface_addr_t *iface_addr)
{
    *(pid_t*)iface_addr = getpid();
    return UCS_OK;
}

static int uct_lwda_ipc_iface_is_reachable(const uct_iface_h tl_iface,
                                           const uct_device_addr_t *dev_addr,
                                           const uct_iface_addr_t *iface_addr)
{
    uct_lwda_ipc_iface_t  *iface = ucs_derived_of(tl_iface, uct_lwda_ipc_iface_t);

    return ((uct_lwda_ipc_iface_node_guid(&iface->super) ==
            *((const uint64_t *)dev_addr)) && ((getpid() != *(pid_t *)iface_addr)));
}

static ucs_status_t uct_lwda_ipc_iface_query(uct_iface_h tl_iface,
                                             uct_iface_attr_t *iface_attr)
{
    uct_lwda_ipc_iface_t *iface = ucs_derived_of(tl_iface, uct_lwda_ipc_iface_t);

    uct_base_iface_query(&iface->super, iface_attr);

    iface_attr->iface_addr_len          = sizeof(pid_t);
    iface_attr->device_addr_len         = sizeof(uint64_t);
    iface_attr->ep_addr_len             = 0;
    iface_attr->max_conn_priv           = 0;
    iface_attr->cap.flags               = UCT_IFACE_FLAG_ERRHANDLE_PEER_FAILURE |
                                          UCT_IFACE_FLAG_CONNECT_TO_IFACE       |
                                          UCT_IFACE_FLAG_PENDING                |
                                          UCT_IFACE_FLAG_GET_ZCOPY              |
                                          UCT_IFACE_FLAG_PUT_ZCOPY              |
                                          UCT_IFACE_FLAG_EVENT_SEND_COMP        |
                                          UCT_IFACE_FLAG_EVENT_RECV;

    iface_attr->cap.put.max_short       = 0;
    iface_attr->cap.put.max_bcopy       = 0;
    iface_attr->cap.put.min_zcopy       = 0;
    iface_attr->cap.put.max_zcopy       = ULONG_MAX;
    iface_attr->cap.put.opt_zcopy_align = 1;
    iface_attr->cap.put.align_mtu       = iface_attr->cap.put.opt_zcopy_align;
    iface_attr->cap.put.max_iov         = 1;

    iface_attr->cap.get.max_bcopy       = 0;
    iface_attr->cap.get.min_zcopy       = 0;
    iface_attr->cap.get.max_zcopy       = ULONG_MAX;
    iface_attr->cap.get.opt_zcopy_align = 1;
    iface_attr->cap.get.align_mtu       = iface_attr->cap.get.opt_zcopy_align;
    iface_attr->cap.get.max_iov         = 1;

    iface_attr->latency.overhead        = 1e-9;
    iface_attr->latency.growth          = 0;
    iface_attr->bandwidth.dedicated     = 0;
    iface_attr->bandwidth.shared        = 24000 * 1024.0 * 1024.0;
    iface_attr->overhead                = 0;
    iface_attr->priority                = 0;

    return UCS_OK;
}

static ucs_status_t
uct_lwda_ipc_iface_flush(uct_iface_h tl_iface, unsigned flags,
                         uct_completion_t *comp)
{
    uct_lwda_ipc_iface_t *iface = ucs_derived_of(tl_iface, uct_lwda_ipc_iface_t);

    if (comp != NULL) {
        return UCS_ERR_UNSUPPORTED;
    }

    if (ucs_queue_is_empty(&iface->outstanding_d2d_event_q)) {
        UCT_TL_IFACE_STAT_FLUSH(ucs_derived_of(tl_iface, uct_base_iface_t));
        return UCS_OK;
    }

    UCT_TL_IFACE_STAT_FLUSH_WAIT(ucs_derived_of(tl_iface, uct_base_iface_t));
    return UCS_INPROGRESS;
}

static ucs_status_t uct_lwda_ipc_iface_event_fd_get(uct_iface_h tl_iface, int *fd_p)
{
    uct_lwda_ipc_iface_t *iface = ucs_derived_of(tl_iface, uct_lwda_ipc_iface_t);

    if (-1 == iface->eventfd) {
        iface->eventfd = eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK);
        if (iface->eventfd == -1) {
            ucs_error("Failed to create event fd: %m");
            return UCS_ERR_IO_ERROR;
        }
    }

    *fd_p = iface->eventfd;
    return UCS_OK;
}

static void uct_lwda_ipc_common_cb(void *lwda_ipc_iface)
{
    uct_lwda_ipc_iface_t *iface = lwda_ipc_iface;
    uint64_t dummy = 1;
    int ret;

    /* No error handling yet */
    do {
        ret = write(iface->eventfd, &dummy, sizeof(dummy));
        if (ret == sizeof(dummy)) {
            return;
        } else if (ret == -1) {
            if (errno == EAGAIN) {
                continue;
            } else if (errno != EINTR) {
                ucs_error("Signaling wakeup failed: %m");
                return;
            }
        } else {
            ucs_assert(ret == 0);
        }
    } while (ret == 0);
}

#if (__LWDACC_VER_MAJOR__ >= 100000)
static void LWDA_CB myHostFn(void *iface)
#else
static void LWDA_CB myHostCallback(LWstream hStream,  LWresult status,
                                   void *iface)
#endif
{
    uct_lwda_ipc_common_cb(iface);
}

static UCS_F_ALWAYS_INLINE unsigned
uct_lwda_ipc_progress_event_q(uct_lwda_ipc_iface_t *iface,
                              ucs_queue_head_t *event_q)
{
    unsigned count = 0;
    uct_lwda_ipc_event_desc_t *lwda_ipc_event;
    ucs_queue_iter_t iter;
    ucs_status_t status;
    unsigned max_events = iface->config.max_poll;

    ucs_queue_for_each_safe(lwda_ipc_event, iter, event_q, queue) {
        status = UCT_LWDADRV_FUNC(lwEventQuery(lwda_ipc_event->event));
        if (UCS_INPROGRESS == status) {
            continue;
        } else if (UCS_OK != status) {
            return status;
        }

        ucs_queue_del_iter(event_q, iter);
        if (lwda_ipc_event->comp != NULL) {
            uct_ilwoke_completion(lwda_ipc_event->comp, UCS_OK);
        }

        status = iface->unmap_memhandle(lwda_ipc_event->cache,
                                        lwda_ipc_event->d_bptr,
                                        lwda_ipc_event->mapped_addr,
                                        iface->config.enable_cache);
        if (status != UCS_OK) {
            ucs_fatal("failed to unmap addr:%p", lwda_ipc_event->mapped_addr);
        }

        ucs_trace_poll("LWDA_IPC Event Done :%p", lwda_ipc_event);
        iface->stream_refcount[lwda_ipc_event->stream_id]--;
        ucs_mpool_put(lwda_ipc_event);
        count++;

        if (count >= max_events) {
            break;
        }
    }

    return count;
}

static unsigned uct_lwda_ipc_iface_progress(uct_iface_h tl_iface)
{
    uct_lwda_ipc_iface_t *iface = ucs_derived_of(tl_iface, uct_lwda_ipc_iface_t);

    return uct_lwda_ipc_progress_event_q(iface, &iface->outstanding_d2d_event_q);
}

static ucs_status_t uct_lwda_ipc_iface_event_fd_arm(uct_iface_h tl_iface,
                                                    unsigned events)
{
    uct_lwda_ipc_iface_t *iface = ucs_derived_of(tl_iface, uct_lwda_ipc_iface_t);
    int ret;
    int i;
    uint64_t dummy;
    ucs_status_t status;

    if (uct_lwda_ipc_progress_event_q(iface, &iface->outstanding_d2d_event_q)) {
        return UCS_ERR_BUSY;
    }

    ucs_assert(iface->eventfd != -1);

    do {
        ret = read(iface->eventfd, &dummy, sizeof(dummy));
        if (ret == sizeof(dummy)) {
            status = UCS_ERR_BUSY;
            return status;
        } else if (ret == -1) {
            if (errno == EAGAIN) {
                break;
            } else if (errno != EINTR) {
                ucs_error("read from internal event fd failed: %m");
                status = UCS_ERR_IO_ERROR;
                return status;
            } else {
                return UCS_ERR_BUSY;
            }
        } else {
            ucs_assert(ret == 0);
        }
    } while (ret != 0);

    if (iface->streams_initialized) {
        for (i = 0; i < iface->config.max_streams; i++) {
            if (iface->stream_refcount[i]) {
                status =
#if (__LWDACC_VER_MAJOR__ >= 100000)
                UCT_LWDADRV_FUNC(lwLaunchHostFunc(iface->stream_d2d[i],
                                                  myHostFn, iface));
#else
                UCT_LWDADRV_FUNC(lwStreamAddCallback(iface->stream_d2d[i],
                                                     myHostCallback, iface, 0));
#endif
                if (UCS_OK != status) {
                    return status;
                }
            }
        }
    }
    return UCS_OK;
}

static uct_iface_ops_t uct_lwda_ipc_iface_ops = {
    .ep_get_zcopy             = uct_lwda_ipc_ep_get_zcopy,
    .ep_put_zcopy             = uct_lwda_ipc_ep_put_zcopy,
    .ep_pending_add           = ucs_empty_function_return_busy,
    .ep_pending_purge         = ucs_empty_function,
    .ep_flush                 = uct_base_ep_flush,
    .ep_fence                 = uct_base_ep_fence,
    .ep_create                = UCS_CLASS_NEW_FUNC_NAME(uct_lwda_ipc_ep_t),
    .ep_destroy               = UCS_CLASS_DELETE_FUNC_NAME(uct_lwda_ipc_ep_t),
    .iface_flush              = uct_lwda_ipc_iface_flush,
    .iface_fence              = uct_base_iface_fence,
    .iface_progress_enable    = uct_base_iface_progress_enable,
    .iface_progress_disable   = uct_base_iface_progress_disable,
    .iface_progress           = uct_lwda_ipc_iface_progress,
    .iface_event_fd_get       = uct_lwda_ipc_iface_event_fd_get,
    .iface_event_arm          = uct_lwda_ipc_iface_event_fd_arm,
    .iface_close              = UCS_CLASS_DELETE_FUNC_NAME(uct_lwda_ipc_iface_t),
    .iface_query              = uct_lwda_ipc_iface_query,
    .iface_get_device_address = uct_lwda_ipc_iface_get_device_address,
    .iface_get_address        = uct_lwda_ipc_iface_get_address,
    .iface_is_reachable       = uct_lwda_ipc_iface_is_reachable,
};

static void uct_lwda_ipc_event_desc_init(ucs_mpool_t *mp, void *obj, void *chunk)
{
    uct_lwda_ipc_event_desc_t *base = (uct_lwda_ipc_event_desc_t *) obj;

    memset(base, 0, sizeof(*base));
    UCT_LWDADRV_FUNC(lwEventCreate(&base->event, LW_EVENT_DISABLE_TIMING));
}

static void uct_lwda_ipc_event_desc_cleanup(ucs_mpool_t *mp, void *obj)
{
    uct_lwda_ipc_event_desc_t *base = (uct_lwda_ipc_event_desc_t *) obj;
    int active;

    UCT_LWDADRV_CTX_ACTIVE(active);

    if (active) {
        UCT_LWDADRV_FUNC(lwEventDestroy(base->event));
    }
}

ucs_status_t uct_lwda_ipc_iface_init_streams(uct_lwda_ipc_iface_t *iface)
{
    ucs_status_t status;
    int i;

    for (i = 0; i < iface->config.max_streams; i++) {
        status = UCT_LWDADRV_FUNC(lwStreamCreate(&iface->stream_d2d[i],
                                                 LW_STREAM_NON_BLOCKING));
        if (UCS_OK != status) {
            return status;
        }

        iface->stream_refcount[i] = 0;
    }

    iface->streams_initialized = 1;

    return UCS_OK;
}

static ucs_mpool_ops_t uct_lwda_ipc_event_desc_mpool_ops = {
    .chunk_alloc   = ucs_mpool_chunk_malloc,
    .chunk_release = ucs_mpool_chunk_free,
    .obj_init      = uct_lwda_ipc_event_desc_init,
    .obj_cleanup   = uct_lwda_ipc_event_desc_cleanup,
};

static UCS_CLASS_INIT_FUNC(uct_lwda_ipc_iface_t, uct_md_h md, uct_worker_h worker,
                           const uct_iface_params_t *params,
                           const uct_iface_config_t *tl_config)
{
    uct_lwda_ipc_iface_config_t *config = NULL;
    ucs_status_t status;

    config = ucs_derived_of(tl_config, uct_lwda_ipc_iface_config_t);
    UCS_CLASS_CALL_SUPER_INIT(uct_base_iface_t, &uct_lwda_ipc_iface_ops, md, worker,
                              params, tl_config UCS_STATS_ARG(params->stats_root)
                              UCS_STATS_ARG("lwda_ipc"));

    if (strncmp(params->mode.device.dev_name,
                UCT_LWDA_DEV_NAME, strlen(UCT_LWDA_DEV_NAME)) != 0) {
        ucs_error("No device was found: %s", params->mode.device.dev_name);
        return UCS_ERR_NO_DEVICE;
    }

    self->config.max_poll            = config->max_poll;
    self->config.max_streams         = config->max_streams;
    self->config.enable_cache        = config->enable_cache;
    self->config.max_lwda_ipc_events = config->max_lwda_ipc_events;

    self->map_memhandle   = uct_lwda_ipc_map_memhandle;
    self->unmap_memhandle = uct_lwda_ipc_unmap_memhandle;

    status = ucs_mpool_init(&self->event_desc,
                            0,
                            sizeof(uct_lwda_ipc_event_desc_t),
                            0,
                            UCS_SYS_CACHE_LINE_SIZE,
                            128,
                            self->config.max_lwda_ipc_events,
                            &uct_lwda_ipc_event_desc_mpool_ops,
                            "LWDA_IPC EVENT objects");
    if (UCS_OK != status) {
        ucs_error("mpool creation failed");
        return UCS_ERR_IO_ERROR;
    }

    self->eventfd = -1;
    self->streams_initialized = 0;
    ucs_queue_head_init(&self->outstanding_d2d_event_q);

    return UCS_OK;
}

static UCS_CLASS_CLEANUP_FUNC(uct_lwda_ipc_iface_t)
{
    ucs_status_t status;
    int i;
    int active;

    UCT_LWDADRV_CTX_ACTIVE(active);

    if (self->streams_initialized && active) {
        for (i = 0; i < self->config.max_streams; i++) {
            status = UCT_LWDADRV_FUNC(lwStreamDestroy(self->stream_d2d[i]));
            if (UCS_OK != status) {
                continue;
            }

            ucs_assert(self->stream_refcount[i] == 0);
        }
        self->streams_initialized = 0;
    }

    uct_base_iface_progress_disable(&self->super.super,
                                    UCT_PROGRESS_SEND | UCT_PROGRESS_RECV);
    ucs_mpool_cleanup(&self->event_desc, 1);
    if (self->eventfd != -1) {
        close(self->eventfd);
    }
}

UCS_CLASS_DEFINE(uct_lwda_ipc_iface_t, uct_base_iface_t);
UCS_CLASS_DEFINE_NEW_FUNC(uct_lwda_ipc_iface_t, uct_iface_t, uct_md_h, uct_worker_h,
                          const uct_iface_params_t*, const uct_iface_config_t*);
static UCS_CLASS_DEFINE_DELETE_FUNC(uct_lwda_ipc_iface_t, uct_iface_t);

UCT_TL_DEFINE(&uct_lwda_ipc_component.super, lwda_ipc, uct_lwda_base_query_devices,
              uct_lwda_ipc_iface_t, "LWDA_IPC_", uct_lwda_ipc_iface_config_table,
              uct_lwda_ipc_iface_config_t);
