/**
 * Copyright (C) Mellanox Technologies Ltd. 2017-2019.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "lwda_copy_ep.h"
#include "lwda_copy_iface.h"

#include <uct/base/uct_log.h>
#include <uct/base/uct_iov.inl>
#include <ucs/profile/profile.h>
#include <ucs/debug/memtrack.h>
#include <ucs/sys/math.h>
#include <ucs/type/class.h>


static UCS_CLASS_INIT_FUNC(uct_lwda_copy_ep_t, const uct_ep_params_t *params)
{
    uct_lwda_copy_iface_t *iface = ucs_derived_of(params->iface,
                                                  uct_lwda_copy_iface_t);

    UCT_EP_PARAMS_CHECK_DEV_IFACE_ADDRS(params);
    UCS_CLASS_CALL_SUPER_INIT(uct_base_ep_t, &iface->super);

    return UCS_OK;
}

static UCS_CLASS_CLEANUP_FUNC(uct_lwda_copy_ep_t)
{
}

UCS_CLASS_DEFINE(uct_lwda_copy_ep_t, uct_base_ep_t)
UCS_CLASS_DEFINE_NEW_FUNC(uct_lwda_copy_ep_t, uct_ep_t, const uct_ep_params_t *);
UCS_CLASS_DEFINE_DELETE_FUNC(uct_lwda_copy_ep_t, uct_ep_t);

#define uct_lwda_copy_trace_data(_remote_addr, _rkey, _fmt, ...) \
     ucs_trace_data(_fmt " to %"PRIx64"(%+ld)", ## __VA_ARGS__, (_remote_addr), \
                    (_rkey))

#define UCT_LWDA_COPY_CHECK_AND_CREATE_STREAM(_strm) \
    if ((_strm) == 0) { \
        ucs_status_t __status; \
        __status = UCT_LWDA_FUNC(lwdaStreamCreateWithFlags(&(_strm), lwdaStreamNonBlocking)); \
        if (UCS_OK != __status) { \
            return UCS_ERR_IO_ERROR; \
        } \
    }

static UCS_F_ALWAYS_INLINE ucs_status_t
uct_lwda_copy_post_lwda_async_copy(uct_ep_h tl_ep, void *dst, void *src, size_t length,
                                   enum lwdaMemcpyKind direction, lwdaStream_t stream,
                                   ucs_queue_head_t *outstanding_queue,
                                   uct_completion_t *comp)
{
    uct_lwda_copy_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_lwda_copy_iface_t);
    uct_lwda_copy_event_desc_t *lwda_event;
    ucs_status_t status;

    if (!length) {
        return UCS_OK;
    }

    lwda_event = ucs_mpool_get(&iface->lwda_event_desc);
    if (ucs_unlikely(lwda_event == NULL)) {
        ucs_error("Failed to allocate lwca event object");
        return UCS_ERR_NO_MEMORY;
    }

    status = UCT_LWDA_FUNC(lwdaMemcpyAsync(dst, src, length, direction, stream));
    if (UCS_OK != status) {
        return UCS_ERR_IO_ERROR;
    }

    status = UCT_LWDA_FUNC(lwdaEventRecord(lwda_event->event, stream));
    if (UCS_OK != status) {
        return UCS_ERR_IO_ERROR;
    }
    ucs_queue_push(outstanding_queue, &lwda_event->queue);
    lwda_event->comp = comp;

    ucs_trace("lwca async issued :%p dst:%p, src:%p  len:%ld",
             lwda_event, dst, src, length);
    return UCS_INPROGRESS;
}

UCS_PROFILE_FUNC(ucs_status_t, uct_lwda_copy_ep_get_zcopy,
                 (tl_ep, iov, iovcnt, remote_addr, rkey, comp),
                 uct_ep_h tl_ep, const uct_iov_t *iov, size_t iovcnt,
                 uint64_t remote_addr, uct_rkey_t rkey,
                 uct_completion_t *comp)
{
    uct_lwda_copy_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_lwda_copy_iface_t);
    ucs_status_t status;

    UCT_LWDA_COPY_CHECK_AND_CREATE_STREAM(iface->stream_d2h);

    status = uct_lwda_copy_post_lwda_async_copy(tl_ep, iov[0].buffer, (void *)remote_addr,
                                                iov[0].length, lwdaMemcpyDeviceToHost,
                                                iface->stream_d2h,
                                                &iface->outstanding_d2h_lwda_event_q, comp);
    if (!UCS_STATUS_IS_ERR(status)) {
        VALGRIND_MAKE_MEM_DEFINED(iov[0].buffer, iov[0].length);
    }

    UCT_TL_EP_STAT_OP(ucs_derived_of(tl_ep, uct_base_ep_t), GET, ZCOPY,
                      uct_iov_total_length(iov, iovcnt));
    uct_lwda_copy_trace_data(remote_addr, rkey, "GET_ZCOPY [length %zu]",
                             uct_iov_total_length(iov, iovcnt));
    return status;
}

UCS_PROFILE_FUNC(ucs_status_t, uct_lwda_copy_ep_put_zcopy,
                 (tl_ep, iov, iovcnt, remote_addr, rkey, comp),
                 uct_ep_h tl_ep, const uct_iov_t *iov, size_t iovcnt,
                 uint64_t remote_addr, uct_rkey_t rkey,
                 uct_completion_t *comp)
{

    uct_lwda_copy_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_lwda_copy_iface_t);
    ucs_status_t status;

    UCT_LWDA_COPY_CHECK_AND_CREATE_STREAM(iface->stream_h2d);

    status = uct_lwda_copy_post_lwda_async_copy(tl_ep, (void *)remote_addr,  iov[0].buffer,
                                                iov[0].length, lwdaMemcpyHostToDevice,
                                                iface->stream_h2d,
                                                &iface->outstanding_h2d_lwda_event_q, comp);

    UCT_TL_EP_STAT_OP(ucs_derived_of(tl_ep, uct_base_ep_t), PUT, ZCOPY,
                      uct_iov_total_length(iov, iovcnt));
    uct_lwda_copy_trace_data(remote_addr, rkey, "GET_ZCOPY [length %zu]",
                             uct_iov_total_length(iov, iovcnt));
    return status;

}

UCS_PROFILE_FUNC(ucs_status_t, uct_lwda_copy_ep_put_short,
                 (tl_ep, buffer, length, remote_addr, rkey),
                 uct_ep_h tl_ep, const void *buffer, unsigned length,
                 uint64_t remote_addr, uct_rkey_t rkey)
{
    uct_lwda_copy_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_lwda_copy_iface_t);
    ucs_status_t status;

    UCT_LWDA_COPY_CHECK_AND_CREATE_STREAM(iface->stream_h2d);

    UCT_LWDA_FUNC(lwdaMemcpyAsync((void *)remote_addr, buffer, length,
                                  lwdaMemcpyHostToDevice, iface->stream_h2d));
    status = UCT_LWDA_FUNC(lwdaStreamSynchronize(iface->stream_h2d));

    UCT_TL_EP_STAT_OP(ucs_derived_of(tl_ep, uct_base_ep_t), PUT, SHORT, length);
    ucs_trace_data("PUT_SHORT size %d from %p to %p",
                   length, buffer, (void *)remote_addr);
    return status;
}

UCS_PROFILE_FUNC(ucs_status_t, uct_lwda_copy_ep_get_short,
                 (tl_ep, buffer, length, remote_addr, rkey),
                 uct_ep_h tl_ep, void *buffer, unsigned length,
                 uint64_t remote_addr, uct_rkey_t rkey)
{
    uct_lwda_copy_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_lwda_copy_iface_t);
    ucs_status_t status;

    UCT_LWDA_COPY_CHECK_AND_CREATE_STREAM(iface->stream_d2h);

    UCT_LWDA_FUNC(lwdaMemcpyAsync(buffer, (void *)remote_addr, length,
                                  lwdaMemcpyDeviceToHost, iface->stream_d2h));
    status = UCT_LWDA_FUNC(lwdaStreamSynchronize(iface->stream_d2h));

    UCT_TL_EP_STAT_OP(ucs_derived_of(tl_ep, uct_base_ep_t), GET, SHORT, length);
    ucs_trace_data("GET_SHORT size %d from %p to %p",
                   length, (void *)remote_addr, buffer);
    return status;
}

