/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2019.  ALL RIGHTS RESERVED.
*
* Copyright (C) UT-Battelle, LLC. 2015. ALL RIGHTS RESERVED.
* See file LICENSE for terms.
*/

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "uct_iface.h"
#include "uct_cm.h"

#include <uct/api/uct.h>
#include <ucs/async/async.h>
#include <ucs/sys/string.h>
#include <ucs/time/time.h>
#include <ucs/debug/debug.h>


#ifdef ENABLE_STATS
static ucs_stats_class_t uct_ep_stats_class = {
    .name = "uct_ep",
    .num_counters = UCT_EP_STAT_LAST,
    .counter_names = {
        [UCT_EP_STAT_AM]          = "am",
        [UCT_EP_STAT_PUT]         = "put",
        [UCT_EP_STAT_GET]         = "get",
        [UCT_EP_STAT_ATOMIC]      = "atomic",
#if IBV_HW_TM
        [UCT_EP_STAT_TAG]         = "tag",
#endif
        [UCT_EP_STAT_BYTES_SHORT] = "bytes_short",
        [UCT_EP_STAT_BYTES_BCOPY] = "bytes_bcopy",
        [UCT_EP_STAT_BYTES_ZCOPY] = "bytes_zcopy",
        [UCT_EP_STAT_NO_RES]      = "no_res",
        [UCT_EP_STAT_FLUSH]       = "flush",
        [UCT_EP_STAT_FLUSH_WAIT]  = "flush_wait",
        [UCT_EP_STAT_PENDING]     = "pending",
        [UCT_EP_STAT_FENCE]       = "fence"
    }
};

static ucs_stats_class_t uct_iface_stats_class = {
    .name = "uct_iface",
    .num_counters = UCT_IFACE_STAT_LAST,
    .counter_names = {
        [UCT_IFACE_STAT_RX_AM]       = "rx_am",
        [UCT_IFACE_STAT_RX_AM_BYTES] = "rx_am_bytes",
        [UCT_IFACE_STAT_TX_NO_DESC]  = "tx_no_desc",
        [UCT_IFACE_STAT_FLUSH]       = "flush",
        [UCT_IFACE_STAT_FLUSH_WAIT]  = "flush_wait",
        [UCT_IFACE_STAT_FENCE]       = "fence"
    }
};
#endif


static ucs_status_t uct_iface_stub_am_handler(void *arg, void *data,
                                              size_t length, unsigned flags)
{
    const size_t dump_len = 64;
    uint8_t id            = (uintptr_t)arg;
    char dump_str[(dump_len * 4) + 1]; /* 1234:5678\n\0 */

    ucs_warn("got active message id %d, but no handler installed", id);
    ucs_warn("payload %zu of %zu bytes:\n%s", ucs_min(length, dump_len), length,
             ucs_str_dump_hex(data, ucs_min(length, dump_len),
                              dump_str, sizeof(dump_str), 16));
    ucs_log_print_backtrace(UCS_LOG_LEVEL_WARN);
    return UCS_OK;
}

static void uct_iface_set_stub_am_handler(uct_base_iface_t *iface, uint8_t id)
{
    iface->am[id].cb    = uct_iface_stub_am_handler;
    iface->am[id].arg   = (void*)(uintptr_t)id;
    iface->am[id].flags = UCT_CB_FLAG_ASYNC;
}

ucs_status_t uct_iface_set_am_handler(uct_iface_h tl_iface, uint8_t id,
                                      uct_am_callback_t cb, void *arg,
                                      uint32_t flags)
{
    uct_base_iface_t *iface = ucs_derived_of(tl_iface, uct_base_iface_t);
    ucs_status_t status;
    uct_iface_attr_t attr;

    if (id >= UCT_AM_ID_MAX) {
        ucs_error("active message id out-of-range (got: %d max: %d)", id,
                  (int)UCT_AM_ID_MAX);
        return UCS_ERR_ILWALID_PARAM;
    }

    if (cb == NULL) {
        uct_iface_set_stub_am_handler(iface, id);
        return UCS_OK;
    }

    status = uct_iface_query(tl_iface, &attr);
    if (status != UCS_OK) {
        return status;
    }

    UCT_CB_FLAGS_CHECK(flags);

    /* If user wants a synchronous callback, it must be supported, or the
     * callback could be called from another thread.
     */
    if (!(flags & UCT_CB_FLAG_ASYNC) && !(attr.cap.flags & UCT_IFACE_FLAG_CB_SYNC)) {
        ucs_error("Synchronous callback requested, but not supported");
        return UCS_ERR_ILWALID_PARAM;
    }

    iface->am[id].cb    = cb;
    iface->am[id].arg   = arg;
    iface->am[id].flags = flags;
    return UCS_OK;
}

ucs_status_t uct_iface_set_am_tracer(uct_iface_h tl_iface, uct_am_tracer_t tracer,
                                     void *arg)
{
    uct_base_iface_t *iface = ucs_derived_of(tl_iface, uct_base_iface_t);

    iface->am_tracer     = tracer;
    iface->am_tracer_arg = arg;
    return UCS_OK;
}

void uct_iface_dump_am(uct_base_iface_t *iface, uct_am_trace_type_t type,
                       uint8_t id, const void *data, size_t length,
                       char *buffer, size_t max)
{
    if (iface->am_tracer != NULL) {
        iface->am_tracer(iface->am_tracer_arg, type, id, data, length, buffer, max);
    }
}

void uct_iface_mpool_empty_warn(uct_base_iface_t *iface, ucs_mpool_t *mp)
{
    static ucs_time_t warn_time = 0;
    ucs_time_t now = ucs_get_time();

    /* Limit the rate of warning to once in 30 seconds. This gives reasonable
     * indication about a deadlock without flooding with warnings messages. */
    if (warn_time == 0) {
        warn_time = now;
    }
    if (now - warn_time > ucs_time_from_sec(30)) {
        ucs_warn("Memory pool %s is empty", ucs_mpool_name(mp));
        warn_time = now;
    }
}

ucs_status_t uct_iface_query(uct_iface_h iface, uct_iface_attr_t *iface_attr)
{
    return iface->ops.iface_query(iface, iface_attr);
}

ucs_status_t uct_iface_get_device_address(uct_iface_h iface, uct_device_addr_t *addr)
{
    return iface->ops.iface_get_device_address(iface, addr);
}

ucs_status_t uct_iface_get_address(uct_iface_h iface, uct_iface_addr_t *addr)
{
    return iface->ops.iface_get_address(iface, addr);
}

int uct_iface_is_reachable(const uct_iface_h iface, const uct_device_addr_t *dev_addr,
                           const uct_iface_addr_t *iface_addr)
{
    return iface->ops.iface_is_reachable(iface, dev_addr, iface_addr);
}

ucs_status_t uct_ep_check(const uct_ep_h ep, unsigned flags,
                          uct_completion_t *comp)
{
    return ep->iface->ops.ep_check(ep, flags, comp);
}

ucs_status_t uct_iface_event_fd_get(uct_iface_h iface, int *fd_p)
{
    return iface->ops.iface_event_fd_get(iface, fd_p);
}

ucs_status_t uct_iface_event_arm(uct_iface_h iface, unsigned events)
{
    return iface->ops.iface_event_arm(iface, events);
}

void uct_iface_close(uct_iface_h iface)
{
    iface->ops.iface_close(iface);
}

void uct_base_iface_progress_enable(uct_iface_h tl_iface, unsigned flags)
{
    uct_base_iface_t *iface = ucs_derived_of(tl_iface, uct_base_iface_t);
    uct_base_iface_progress_enable_cb(iface,
                                      (ucs_callback_t)iface->super.ops.iface_progress,
                                      flags);
}

void uct_base_iface_progress_enable_cb(uct_base_iface_t *iface,
                                       ucs_callback_t cb, unsigned flags)
{
    uct_priv_worker_t *worker = iface->worker;
    unsigned thread_safe;

    UCS_ASYNC_BLOCK(worker->async);

    thread_safe = flags & UCT_PROGRESS_THREAD_SAFE;
    flags      &= ~UCT_PROGRESS_THREAD_SAFE;

    /* Add callback only if previous flags are 0 and new flags != 0 */
    if ((!iface->progress_flags && flags) &&
        (iface->prog.id == UCS_CALLBACKQ_ID_NULL)) {
        if (thread_safe) {
            iface->prog.id = ucs_callbackq_add_safe(&worker->super.progress_q,
                                                    cb, iface,
                                                    UCS_CALLBACKQ_FLAG_FAST);
        } else {
            iface->prog.id = ucs_callbackq_add(&worker->super.progress_q, cb,
                                               iface, UCS_CALLBACKQ_FLAG_FAST);
        }
    }
    iface->progress_flags |= flags;

    UCS_ASYNC_UNBLOCK(worker->async);
}

void uct_base_iface_progress_disable(uct_iface_h tl_iface, unsigned flags)
{
    uct_base_iface_t *iface = ucs_derived_of(tl_iface, uct_base_iface_t);
    uct_priv_worker_t *worker = iface->worker;
    unsigned thread_safe;

    UCS_ASYNC_BLOCK(worker->async);

    thread_safe = flags & UCT_PROGRESS_THREAD_SAFE;
    flags      &= ~UCT_PROGRESS_THREAD_SAFE;

    /* Remove callback only if previous flags != 0, and removing the given
     * flags makes it become 0.
     */
    if ((iface->progress_flags && !(iface->progress_flags & ~flags)) &&
        (iface->prog.id != UCS_CALLBACKQ_ID_NULL)) {
        if (thread_safe) {
            ucs_callbackq_remove_safe(&worker->super.progress_q, iface->prog.id);
        } else {
            ucs_callbackq_remove(&worker->super.progress_q, iface->prog.id);
        }
        iface->prog.id = UCS_CALLBACKQ_ID_NULL;
    }
    iface->progress_flags &= ~flags;

    UCS_ASYNC_UNBLOCK(worker->async);
}

ucs_status_t uct_base_iface_flush(uct_iface_h tl_iface, unsigned flags,
                                  uct_completion_t *comp)
{
    UCT_TL_IFACE_STAT_FLUSH(ucs_derived_of(tl_iface, uct_base_iface_t));
    return UCS_OK;
}

ucs_status_t uct_base_iface_fence(uct_iface_h tl_iface, unsigned flags)
{
    UCT_TL_IFACE_STAT_FENCE(ucs_derived_of(tl_iface, uct_base_iface_t));
    return UCS_OK;
}

ucs_status_t uct_base_ep_flush(uct_ep_h tl_ep, unsigned flags,
                               uct_completion_t *comp)
{
    UCT_TL_EP_STAT_FLUSH(ucs_derived_of(tl_ep, uct_base_ep_t));
    return UCS_OK;
}

ucs_status_t uct_base_ep_fence(uct_ep_h tl_ep, unsigned flags)
{
    UCT_TL_EP_STAT_FENCE(ucs_derived_of(tl_ep, uct_base_ep_t));
    return UCS_OK;
}

static void uct_ep_failed_purge_cb(uct_pending_req_t *self, void *arg)
{
    uct_pending_req_queue_push((ucs_queue_head_t*)arg, self);
}

static void uct_ep_failed_purge(uct_ep_h tl_ep, uct_pending_purge_callback_t cb,
                                void *arg)
{
    uct_failed_iface_t *iface = ucs_derived_of(tl_ep->iface,
                                               uct_failed_iface_t);
    uct_pending_req_t *req;

    ucs_queue_for_each_extract(req, &iface->pend_q, priv, 1) {
        if (cb != NULL) {
            cb(req, arg);
        } else {
            ucs_warn("ep=%p cancelling user pending request %p", tl_ep, req);
        }
    }
}

static void uct_ep_failed_destroy(uct_ep_h tl_ep)
{
    /* Warn user if some pending reqs left*/
    uct_ep_failed_purge (tl_ep, NULL, NULL);

    ucs_free(tl_ep->iface);
    ucs_free(tl_ep);
}

ucs_status_t uct_set_ep_failed(ucs_class_t *cls, uct_ep_h tl_ep,
                               uct_iface_h tl_iface, ucs_status_t status)
{
    uct_failed_iface_t *f_iface;
    uct_iface_ops_t    *ops;
    uct_base_iface_t   *iface = ucs_derived_of(tl_iface, uct_base_iface_t);

    ucs_debug("set ep %p to failed state", tl_ep);

    /* TBD: consider allocating one instance per interface
     * rather than for each endpoint */
    f_iface = ucs_malloc(sizeof(*f_iface), "failed iface");
    if (f_iface == NULL) {
        ucs_error("Could not create failed iface (nomem)");
        return status;
    }

    ucs_queue_head_init(&f_iface->pend_q);
    ops = &f_iface->super.ops;

    /* Move all pending requests to the queue.
     * Failed ep will use that queue for purge. */
    uct_ep_pending_purge(tl_ep, uct_ep_failed_purge_cb, &f_iface->pend_q);

    ops->ep_put_short        = (uct_ep_put_short_func_t)ucs_empty_function_return_ep_timeout;
    ops->ep_put_bcopy        = (uct_ep_put_bcopy_func_t)ucs_empty_function_return_bc_ep_timeout;
    ops->ep_put_zcopy        = (uct_ep_put_zcopy_func_t)ucs_empty_function_return_ep_timeout;
    ops->ep_get_short        = (uct_ep_get_short_func_t)ucs_empty_function_return_ep_timeout;
    ops->ep_get_bcopy        = (uct_ep_get_bcopy_func_t)ucs_empty_function_return_ep_timeout;
    ops->ep_get_zcopy        = (uct_ep_get_zcopy_func_t)ucs_empty_function_return_ep_timeout;
    ops->ep_am_short         = (uct_ep_am_short_func_t)ucs_empty_function_return_ep_timeout;
    ops->ep_am_bcopy         = (uct_ep_am_bcopy_func_t)ucs_empty_function_return_bc_ep_timeout;
    ops->ep_am_zcopy         = (uct_ep_am_zcopy_func_t)ucs_empty_function_return_ep_timeout;
    ops->ep_atomic_cswap64   = (uct_ep_atomic_cswap64_func_t)ucs_empty_function_return_ep_timeout;
    ops->ep_atomic_cswap32   = (uct_ep_atomic_cswap32_func_t)ucs_empty_function_return_ep_timeout;
    ops->ep_atomic64_post    = (uct_ep_atomic64_post_func_t)ucs_empty_function_return_ep_timeout;
    ops->ep_atomic32_post    = (uct_ep_atomic32_post_func_t)ucs_empty_function_return_ep_timeout;
    ops->ep_atomic64_fetch   = (uct_ep_atomic64_fetch_func_t)ucs_empty_function_return_ep_timeout;
    ops->ep_atomic32_fetch   = (uct_ep_atomic32_fetch_func_t)ucs_empty_function_return_ep_timeout;
    ops->ep_tag_eager_short  = (uct_ep_tag_eager_short_func_t)ucs_empty_function_return_ep_timeout;
    ops->ep_tag_eager_bcopy  = (uct_ep_tag_eager_bcopy_func_t)ucs_empty_function_return_ep_timeout;
    ops->ep_tag_eager_zcopy  = (uct_ep_tag_eager_zcopy_func_t)ucs_empty_function_return_ep_timeout;
    ops->ep_tag_rndv_zcopy   = (uct_ep_tag_rndv_zcopy_func_t)ucs_empty_function_return_ep_timeout;
    ops->ep_tag_rndv_cancel  = (uct_ep_tag_rndv_cancel_func_t)ucs_empty_function_return_ep_timeout;
    ops->ep_tag_rndv_request = (uct_ep_tag_rndv_request_func_t)ucs_empty_function_return_ep_timeout;
    ops->ep_pending_add      = (uct_ep_pending_add_func_t)ucs_empty_function_return_ep_timeout;
    ops->ep_pending_purge    = uct_ep_failed_purge;
    ops->ep_flush            = (uct_ep_flush_func_t)ucs_empty_function_return_ep_timeout;
    ops->ep_fence            = (uct_ep_fence_func_t)ucs_empty_function_return_ep_timeout;
    ops->ep_check            = (uct_ep_check_func_t)ucs_empty_function_return_ep_timeout;
    ops->ep_connect_to_ep    = (uct_ep_connect_to_ep_func_t)ucs_empty_function_return_ep_timeout;
    ops->ep_destroy          = uct_ep_failed_destroy;
    ops->ep_get_address      = (uct_ep_get_address_func_t)ucs_empty_function_return_ep_timeout;

    ucs_class_call_cleanup_chain(cls, tl_ep, -1);

    tl_ep->iface = &f_iface->super;

    if (iface->err_handler) {
        return iface->err_handler(iface->err_handler_arg, tl_ep, status);
    } else if (status == UCS_ERR_CANCELED) {
        ucs_debug("error %s was suppressed for ep %p",
                  ucs_status_string(UCS_ERR_CANCELED), tl_ep);
        /* Suppress this since the cancellation is initiated by user. */
        status = UCS_OK;
    } else {
        ucs_debug("error %s was not handled for ep %p",
                  ucs_status_string(status), tl_ep);
    }

    return status;
}

void uct_base_iface_query(uct_base_iface_t *iface, uct_iface_attr_t *iface_attr)
{
    memset(iface_attr, 0, sizeof(*iface_attr));

    iface_attr->max_num_eps   = iface->config.max_num_eps;
    iface_attr->dev_num_paths = 1;
}

ucs_status_t uct_single_device_resource(uct_md_h md, const char *dev_name,
                                        uct_device_type_t dev_type,
                                        uct_tl_device_resource_t **tl_devices_p,
                                        unsigned *num_tl_devices_p)
{
    uct_tl_device_resource_t *device;

    device = ucs_calloc(1, sizeof(*device), "device resource");
    if (NULL == device) {
        ucs_error("failed to allocate device resource");
        return UCS_ERR_NO_MEMORY;
    }

    ucs_snprintf_zero(device->name, sizeof(device->name), "%s", dev_name);
    device->type = dev_type;

    *num_tl_devices_p = 1;
    *tl_devices_p     = device;
    return UCS_OK;
}

UCS_CLASS_INIT_FUNC(uct_iface_t, uct_iface_ops_t *ops)
{
    ucs_assert_always(ops->ep_flush                 != NULL);
    ucs_assert_always(ops->ep_fence                 != NULL);
    ucs_assert_always(ops->ep_destroy               != NULL);
    ucs_assert_always(ops->iface_flush              != NULL);
    ucs_assert_always(ops->iface_fence              != NULL);
    ucs_assert_always(ops->iface_progress_enable    != NULL);
    ucs_assert_always(ops->iface_progress_disable   != NULL);
    ucs_assert_always(ops->iface_progress           != NULL);
    ucs_assert_always(ops->iface_close              != NULL);
    ucs_assert_always(ops->iface_query              != NULL);
    ucs_assert_always(ops->iface_get_device_address != NULL);
    ucs_assert_always(ops->iface_is_reachable       != NULL);

    self->ops = *ops;
    return UCS_OK;
}

UCS_CLASS_CLEANUP_FUNC(uct_iface_t)
{
}

UCS_CLASS_DEFINE(uct_iface_t, void);


UCS_CLASS_INIT_FUNC(uct_base_iface_t, uct_iface_ops_t *ops, uct_md_h md,
                    uct_worker_h worker, const uct_iface_params_t *params,
                    const uct_iface_config_t *config
                    UCS_STATS_ARG(ucs_stats_node_t *stats_parent)
                    UCS_STATS_ARG(const char *iface_name))
{
    uint64_t alloc_methods_bitmap;
    uct_alloc_method_t method;
    unsigned i;
    uint8_t id;

    UCS_CLASS_CALL_SUPER_INIT(uct_iface_t, ops);

    UCT_CB_FLAGS_CHECK((params->field_mask &
                        UCT_IFACE_PARAM_FIELD_ERR_HANDLER_FLAGS) ?
                       params->err_handler_flags : 0);

    self->md                = md;
    self->worker            = ucs_derived_of(worker, uct_priv_worker_t);
    self->am_tracer         = NULL;
    self->am_tracer_arg     = NULL;
    self->err_handler       = (params->field_mask &
                               UCT_IFACE_PARAM_FIELD_ERR_HANDLER) ?
                              params->err_handler : NULL;
    self->err_handler_flags = (params->field_mask &
                               UCT_IFACE_PARAM_FIELD_ERR_HANDLER_FLAGS) ?
                              params->err_handler_flags : 0;
    self->err_handler_arg   = (params->field_mask &
                               UCT_IFACE_PARAM_FIELD_ERR_HANDLER_ARG) ?
                              params->err_handler_arg : NULL;
    self->progress_flags    = 0;
    uct_worker_progress_init(&self->prog);

    for (id = 0; id < UCT_AM_ID_MAX; ++id) {
        uct_iface_set_stub_am_handler(self, id);
    }

    /* Copy allocation methods configuration. In the process, remove duplicates. */
    UCS_STATIC_ASSERT(sizeof(alloc_methods_bitmap) * 8 >= UCT_ALLOC_METHOD_LAST);
    self->config.num_alloc_methods = 0;
    alloc_methods_bitmap           = 0;
    for (i = 0; i < config->alloc_methods.count; ++i) {
        method = config->alloc_methods.methods[i];
        if (alloc_methods_bitmap & UCS_BIT(method)) {
            continue;
        }

        ucs_assert(self->config.num_alloc_methods < UCT_ALLOC_METHOD_LAST);
        self->config.alloc_methods[self->config.num_alloc_methods++] = method;
        alloc_methods_bitmap |= UCS_BIT(method);
    }

    self->config.failure_level = (ucs_log_level_t)config->failure;
    self->config.max_num_eps   = config->max_num_eps;

    return UCS_STATS_NODE_ALLOC(&self->stats, &uct_iface_stats_class,
                                stats_parent, "-%s-%p", iface_name, self);
}

static UCS_CLASS_CLEANUP_FUNC(uct_base_iface_t)
{
    UCS_STATS_NODE_FREE(self->stats);
}

UCS_CLASS_DEFINE(uct_base_iface_t, uct_iface_t);


ucs_status_t uct_iface_accept(uct_iface_h iface,
                              uct_conn_request_h conn_request)
{
    return iface->ops.iface_accept(iface, conn_request);
}


ucs_status_t uct_iface_reject(uct_iface_h iface,
                              uct_conn_request_h conn_request)
{
    return iface->ops.iface_reject(iface, conn_request);
}


ucs_status_t uct_ep_create(const uct_ep_params_t *params, uct_ep_h *ep_p)
{
    if (params->field_mask & UCT_EP_PARAM_FIELD_IFACE) {
        return params->iface->ops.ep_create(params, ep_p);
    } else if (params->field_mask & UCT_EP_PARAM_FIELD_CM) {
        return params->cm->ops->ep_create(params, ep_p);
    }

    return UCS_ERR_ILWALID_PARAM;
}

ucs_status_t uct_ep_disconnect(uct_ep_h ep, unsigned flags)
{
    return ep->iface->ops.ep_disconnect(ep, flags);
}

void uct_ep_destroy(uct_ep_h ep)
{
    ep->iface->ops.ep_destroy(ep);
}

ucs_status_t uct_ep_get_address(uct_ep_h ep, uct_ep_addr_t *addr)
{
    return ep->iface->ops.ep_get_address(ep, addr);
}

ucs_status_t uct_ep_connect_to_ep(uct_ep_h ep, const uct_device_addr_t *dev_addr,
                                  const uct_ep_addr_t *ep_addr)
{
    return ep->iface->ops.ep_connect_to_ep(ep, dev_addr, ep_addr);
}

UCS_CLASS_INIT_FUNC(uct_ep_t, uct_iface_t *iface)
{
    self->iface = iface;
    return UCS_OK;
}

UCS_CLASS_CLEANUP_FUNC(uct_ep_t)
{
}

UCS_CLASS_DEFINE(uct_ep_t, void);


UCS_CLASS_INIT_FUNC(uct_base_ep_t, uct_base_iface_t *iface)
{
    UCS_CLASS_CALL_SUPER_INIT(uct_ep_t, &iface->super);

    return UCS_STATS_NODE_ALLOC(&self->stats, &uct_ep_stats_class, iface->stats,
                                "-%p", self);
}

static UCS_CLASS_CLEANUP_FUNC(uct_base_ep_t)
{
    UCS_STATS_NODE_FREE(self->stats);
}

UCS_CLASS_DEFINE(uct_base_ep_t, uct_ep_t);


UCS_CONFIG_DEFINE_ARRAY(alloc_methods, sizeof(uct_alloc_method_t),
                        UCS_CONFIG_TYPE_ENUM(uct_alloc_method_names));

ucs_config_field_t uct_iface_config_table[] = {
  {"MAX_SHORT", "",
   "The configuration parameter replaced by: "
   "UCX_<IB transport>_TX_MIN_INLINE for IB, UCX_MM_FIFO_SIZE for MM",
   UCS_CONFIG_DEPRECATED_FIELD_OFFSET, UCS_CONFIG_TYPE_DEPRECATED},

  {"MAX_BCOPY", "",
   "The configuration parameter replaced by: "
   "UCX_<transport>_SEG_SIZE where <transport> is one of: IB, MM, SELF, TCP",
   UCS_CONFIG_DEPRECATED_FIELD_OFFSET, UCS_CONFIG_TYPE_DEPRECATED},

  {"ALLOC", "huge,thp,md,mmap,heap",
   "Priority of methods to allocate intermediate buffers for communication",
   ucs_offsetof(uct_iface_config_t, alloc_methods), UCS_CONFIG_TYPE_ARRAY(alloc_methods)},

  {"FAILURE", "error",
   "Level of network failure reporting",
   ucs_offsetof(uct_iface_config_t, failure), UCS_CONFIG_TYPE_ENUM(ucs_log_level_names)},

  {"MAX_NUM_EPS", "inf",
   "Maximum number of endpoints that the transport interface is able to create",
   ucs_offsetof(uct_iface_config_t, max_num_eps), UCS_CONFIG_TYPE_ULUNITS},

  {NULL}
};
