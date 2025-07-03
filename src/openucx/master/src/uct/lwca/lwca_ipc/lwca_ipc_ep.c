/**
 * Copyright (C) Mellanox Technologies Ltd. 2018-2019.  ALL RIGHTS RESERVED.
 * Copyright (c) 2018, LWPU CORPORATION. All rights reserved.
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "lwda_ipc_ep.h"
#include "lwda_ipc_iface.h"
#include "lwda_ipc_md.h"

#include <uct/base/uct_log.h>
#include <uct/base/uct_iov.inl>
#include <ucs/debug/memtrack.h>
#include <ucs/sys/math.h>
#include <ucs/type/class.h>
#include <ucs/profile/profile.h>

#define UCT_LWDA_IPC_PUT 0
#define UCT_LWDA_IPC_GET 1

static UCS_CLASS_INIT_FUNC(uct_lwda_ipc_ep_t, const uct_ep_params_t *params)
{
    uct_lwda_ipc_iface_t *iface = ucs_derived_of(params->iface,
                                                 uct_lwda_ipc_iface_t);
    ucs_status_t status;
    char target_name[64];

    UCT_EP_PARAMS_CHECK_DEV_IFACE_ADDRS(params);
    UCS_CLASS_CALL_SUPER_INIT(uct_base_ep_t, &iface->super);
    self->remote_memh_cache = NULL;

    /* create a cache by default; disabling implies remove mapping immediately
     * after use */
    snprintf(target_name, sizeof(target_name), "dest:%d",
            *(pid_t*)params->iface_addr);
    status = uct_lwda_ipc_create_cache(&self->remote_memh_cache, target_name);
    if (status != UCS_OK) {
        ucs_error("could not create create lwca ipc cache: %s",
                  ucs_status_string(status));
        return status;
    }

    return UCS_OK;
}

static UCS_CLASS_CLEANUP_FUNC(uct_lwda_ipc_ep_t)
{
    if (self->remote_memh_cache) {
        uct_lwda_ipc_destroy_cache(self->remote_memh_cache);
    }
}

UCS_CLASS_DEFINE(uct_lwda_ipc_ep_t, uct_base_ep_t)
UCS_CLASS_DEFINE_NEW_FUNC(uct_lwda_ipc_ep_t, uct_ep_t, const uct_ep_params_t *);
UCS_CLASS_DEFINE_DELETE_FUNC(uct_lwda_ipc_ep_t, uct_ep_t);

#define uct_lwda_ipc_trace_data(_addr, _rkey, _fmt, ...)     \
    ucs_trace_data(_fmt " to %"PRIx64"(%+ld)", ## __VA_ARGS__, (_addr), (_rkey))

static UCS_F_ALWAYS_INLINE ucs_status_t
uct_lwda_ipc_post_lwda_async_copy(uct_ep_h tl_ep, uint64_t remote_addr,
                                  const uct_iov_t *iov, uct_rkey_t rkey,
                                  uct_completion_t *comp, int direction)
{
    uct_lwda_ipc_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_lwda_ipc_iface_t);
    uct_lwda_ipc_ep_t *ep       = ucs_derived_of(tl_ep, uct_lwda_ipc_ep_t);
    uct_lwda_ipc_key_t *key     = (uct_lwda_ipc_key_t *) rkey;
    void *mapped_rem_addr;
    void *mapped_addr;
    uct_lwda_ipc_event_desc_t *lwda_ipc_event;
    ucs_queue_head_t *outstanding_queue;
    ucs_status_t status;
    LWdeviceptr dst, src;
    LWstream stream;
    size_t offset;

    if (0 == iov[0].length) {
        ucs_trace_data("Zero length request: skip it");
        return UCS_OK;
    }

    status = iface->map_memhandle((void *)ep->remote_memh_cache, key, &mapped_addr);
    if (status != UCS_OK) {
        return UCS_ERR_IO_ERROR;
    }

    offset          = (uintptr_t)remote_addr - (uintptr_t)key->d_bptr;
    mapped_rem_addr = (void *) ((uintptr_t) mapped_addr + offset);
    ucs_assert(offset <= key->b_len);

    if (!iface->streams_initialized) {
        status = uct_lwda_ipc_iface_init_streams(iface);
        if (UCS_OK != status) {
            return status;
        }
    }

    key->dev_num %= iface->config.max_streams; /* round-robin */

    stream            = iface->stream_d2d[key->dev_num];
    outstanding_queue = &iface->outstanding_d2d_event_q;
    lwda_ipc_event    = ucs_mpool_get(&iface->event_desc);

    if (ucs_unlikely(lwda_ipc_event == NULL)) {
        ucs_error("Failed to allocate lwda_ipc event object");
        return UCS_ERR_NO_MEMORY;
    }

    dst = (LWdeviceptr)
        ((direction == UCT_LWDA_IPC_PUT) ? mapped_rem_addr : iov[0].buffer);
    src = (LWdeviceptr)
        ((direction == UCT_LWDA_IPC_PUT) ? iov[0].buffer : mapped_rem_addr);

    status = UCT_LWDADRV_FUNC(lwMemcpyDtoDAsync(dst, src, iov[0].length, stream));
    if (UCS_OK != status) {
        ucs_mpool_put(lwda_ipc_event);
        return status;
    }

    iface->stream_refcount[key->dev_num]++;
    lwda_ipc_event->stream_id = key->dev_num;

    status = UCT_LWDADRV_FUNC(lwEventRecord(lwda_ipc_event->event, stream));
    if (UCS_OK != status) {
        ucs_mpool_put(lwda_ipc_event);
        return status;
    }

    ucs_queue_push(outstanding_queue, &lwda_ipc_event->queue);
    lwda_ipc_event->comp        = comp;
    lwda_ipc_event->mapped_addr = mapped_addr;
    lwda_ipc_event->cache       = ep->remote_memh_cache;
    lwda_ipc_event->d_bptr      = (uintptr_t)key->d_bptr;
    ucs_trace("lwMemcpyDtoDAsync issued :%p dst:%p, src:%p  len:%ld",
             lwda_ipc_event, (void *) dst, (void *) src, iov[0].length);
    return UCS_INPROGRESS;
}

UCS_PROFILE_FUNC(ucs_status_t, uct_lwda_ipc_ep_get_zcopy,
                 (tl_ep, iov, iovcnt, remote_addr, rkey, comp),
                 uct_ep_h tl_ep, const uct_iov_t *iov, size_t iovcnt,
                 uint64_t remote_addr, uct_rkey_t rkey,
                 uct_completion_t *comp)
{
    ucs_status_t status;

    status = uct_lwda_ipc_post_lwda_async_copy(tl_ep, remote_addr, iov,
                                               rkey, comp, UCT_LWDA_IPC_GET);
    if (UCS_STATUS_IS_ERR(status)) {
        return status;
    }

    UCT_TL_EP_STAT_OP(ucs_derived_of(tl_ep, uct_base_ep_t), GET, ZCOPY,
                      uct_iov_total_length(iov, iovcnt));
    uct_lwda_ipc_trace_data(remote_addr, rkey, "GET_ZCOPY [length %zu]",
                            uct_iov_total_length(iov, iovcnt));
    return status;
}

UCS_PROFILE_FUNC(ucs_status_t, uct_lwda_ipc_ep_put_zcopy,
                 (tl_ep, iov, iovcnt, remote_addr, rkey, comp),
                 uct_ep_h tl_ep, const uct_iov_t *iov, size_t iovcnt,
                 uint64_t remote_addr, uct_rkey_t rkey,
                 uct_completion_t *comp)
{
    ucs_status_t status;

    status = uct_lwda_ipc_post_lwda_async_copy(tl_ep, remote_addr, iov,
                                               rkey, comp, UCT_LWDA_IPC_PUT);
    if (UCS_STATUS_IS_ERR(status)) {
        return status;
    }

    UCT_TL_EP_STAT_OP(ucs_derived_of(tl_ep, uct_base_ep_t), PUT, ZCOPY,
                      uct_iov_total_length(iov, iovcnt));
    uct_lwda_ipc_trace_data(remote_addr, rkey, "PUT_ZCOPY [length %zu]",
                                uct_iov_total_length(iov, iovcnt));
    return status;
}
