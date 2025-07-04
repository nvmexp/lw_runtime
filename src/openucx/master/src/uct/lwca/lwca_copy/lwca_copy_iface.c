/**
 * Copyright (C) Mellanox Technologies Ltd. 2017-2019.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "lwda_copy_iface.h"
#include "lwda_copy_md.h"
#include "lwda_copy_ep.h"

#include <uct/lwca/base/lwda_iface.h>
#include <ucs/type/class.h>
#include <ucs/sys/string.h>
#include <ucs/arch/cpu.h>


static ucs_config_field_t uct_lwda_copy_iface_config_table[] = {

    {"", "", NULL,
     ucs_offsetof(uct_lwda_copy_iface_config_t, super),
     UCS_CONFIG_TYPE_TABLE(uct_iface_config_table)},

    {"MAX_POLL", "16",
     "Max number of event completions to pick during lwca events polling",
     ucs_offsetof(uct_lwda_copy_iface_config_t, max_poll), UCS_CONFIG_TYPE_UINT},

    {"MAX_EVENTS", "inf",
     "Max number of lwca events. -1 is infinite",
     ucs_offsetof(uct_lwda_copy_iface_config_t, max_lwda_events), UCS_CONFIG_TYPE_UINT},

    {NULL}
};


/* Forward declaration for the delete function */
static void UCS_CLASS_DELETE_FUNC_NAME(uct_lwda_copy_iface_t)(uct_iface_t*);


static ucs_status_t uct_lwda_copy_iface_get_address(uct_iface_h tl_iface,
                                                    uct_iface_addr_t *iface_addr)
{
    uct_lwda_copy_iface_t *iface = ucs_derived_of(tl_iface, uct_lwda_copy_iface_t);

    *(uct_lwda_copy_iface_addr_t*)iface_addr = iface->id;
    return UCS_OK;
}

static int uct_lwda_copy_iface_is_reachable(const uct_iface_h tl_iface,
                                            const uct_device_addr_t *dev_addr,
                                            const uct_iface_addr_t *iface_addr)
{
    uct_lwda_copy_iface_t  *iface = ucs_derived_of(tl_iface, uct_lwda_copy_iface_t);
    uct_lwda_copy_iface_addr_t *addr = (uct_lwda_copy_iface_addr_t*)iface_addr;

    return (addr != NULL) && (iface->id == *addr);
}

static ucs_status_t uct_lwda_copy_iface_query(uct_iface_h tl_iface,
                                              uct_iface_attr_t *iface_attr)
{
    uct_lwda_copy_iface_t *iface = ucs_derived_of(tl_iface, uct_lwda_copy_iface_t);

    uct_base_iface_query(&iface->super, iface_attr);

    iface_attr->iface_addr_len          = sizeof(uct_lwda_copy_iface_addr_t);
    iface_attr->device_addr_len         = 0;
    iface_attr->ep_addr_len             = 0;
    iface_attr->cap.flags               = UCT_IFACE_FLAG_CONNECT_TO_IFACE |
                                          UCT_IFACE_FLAG_GET_SHORT |
                                          UCT_IFACE_FLAG_PUT_SHORT |
                                          UCT_IFACE_FLAG_GET_ZCOPY |
                                          UCT_IFACE_FLAG_PUT_ZCOPY |
                                          UCT_IFACE_FLAG_PENDING;

    iface_attr->cap.put.max_short       = UINT_MAX;
    iface_attr->cap.put.max_bcopy       = 0;
    iface_attr->cap.put.min_zcopy       = 0;
    iface_attr->cap.put.max_zcopy       = SIZE_MAX;
    iface_attr->cap.put.opt_zcopy_align = 1;
    iface_attr->cap.put.align_mtu       = iface_attr->cap.put.opt_zcopy_align;
    iface_attr->cap.put.max_iov         = 1;

    iface_attr->cap.get.max_short       = UINT_MAX;
    iface_attr->cap.get.max_bcopy       = 0;
    iface_attr->cap.get.min_zcopy       = 0;
    iface_attr->cap.get.max_zcopy       = SIZE_MAX;
    iface_attr->cap.get.opt_zcopy_align = 1;
    iface_attr->cap.get.align_mtu       = iface_attr->cap.get.opt_zcopy_align;
    iface_attr->cap.get.max_iov         = 1;

    iface_attr->cap.am.max_short        = 0;
    iface_attr->cap.am.max_bcopy        = 0;
    iface_attr->cap.am.min_zcopy        = 0;
    iface_attr->cap.am.max_zcopy        = 0;
    iface_attr->cap.am.opt_zcopy_align  = 1;
    iface_attr->cap.am.align_mtu        = iface_attr->cap.am.opt_zcopy_align;
    iface_attr->cap.am.max_hdr          = 0;
    iface_attr->cap.am.max_iov          = 1;

    iface_attr->latency.overhead        = 10e-6; /* 10 us */
    iface_attr->latency.growth          = 0;
    iface_attr->bandwidth.dedicated     = 0;
    iface_attr->bandwidth.shared        = 6911 * 1024.0 * 1024.0;
    iface_attr->overhead                = 0;
    iface_attr->priority                = 0;

    return UCS_OK;
}

static ucs_status_t uct_lwda_copy_iface_flush(uct_iface_h tl_iface, unsigned flags,
                                              uct_completion_t *comp)
{
    uct_lwda_copy_iface_t *iface = ucs_derived_of(tl_iface, uct_lwda_copy_iface_t);

    if (comp != NULL) {
        return UCS_ERR_UNSUPPORTED;
    }

    if (ucs_queue_is_empty(&iface->outstanding_d2h_lwda_event_q) &&
        ucs_queue_is_empty(&iface->outstanding_h2d_lwda_event_q)) {
        UCT_TL_IFACE_STAT_FLUSH(ucs_derived_of(tl_iface, uct_base_iface_t));
        return UCS_OK;
    }

    UCT_TL_IFACE_STAT_FLUSH_WAIT(ucs_derived_of(tl_iface, uct_base_iface_t));
    return UCS_INPROGRESS;
}

static UCS_F_ALWAYS_INLINE unsigned
uct_lwda_copy_progress_event_queue(ucs_queue_head_t *event_queue, unsigned max_events)
{
    unsigned count = 0;
    lwdaError_t result = lwdaSuccess;
    uct_lwda_copy_event_desc_t *lwda_event;
    ucs_queue_iter_t iter;

    ucs_queue_for_each_safe(lwda_event, iter, event_queue, queue) {
        result = lwdaEventQuery(lwda_event->event);
        if (lwdaSuccess != result) {
            break;
        }
        ucs_queue_del_iter(event_queue, iter);
        if (lwda_event->comp != NULL) {
            uct_ilwoke_completion(lwda_event->comp, UCS_OK);
        }
        ucs_trace_poll("LWCA Event Done :%p", lwda_event);
        ucs_mpool_put(lwda_event);
        count++;
        if (count >= max_events) {
            break;
        }
    }
    return count;
}

static unsigned uct_lwda_copy_iface_progress(uct_iface_h tl_iface)
{
    uct_lwda_copy_iface_t *iface = ucs_derived_of(tl_iface, uct_lwda_copy_iface_t);
    unsigned max_events = iface->config.max_poll;
    unsigned count;

    count = uct_lwda_copy_progress_event_queue(&iface->outstanding_d2h_lwda_event_q,
                                               max_events);
    count += uct_lwda_copy_progress_event_queue(&iface->outstanding_h2d_lwda_event_q,
                                                (max_events - count));
    return count;
}

static uct_iface_ops_t uct_lwda_copy_iface_ops = {
    .ep_get_short             = uct_lwda_copy_ep_get_short,
    .ep_put_short             = uct_lwda_copy_ep_put_short,
    .ep_get_zcopy             = uct_lwda_copy_ep_get_zcopy,
    .ep_put_zcopy             = uct_lwda_copy_ep_put_zcopy,
    .ep_pending_add           = ucs_empty_function_return_busy,
    .ep_pending_purge         = ucs_empty_function,
    .ep_flush                 = uct_base_ep_flush,
    .ep_fence                 = uct_base_ep_fence,
    .ep_create                = UCS_CLASS_NEW_FUNC_NAME(uct_lwda_copy_ep_t),
    .ep_destroy               = UCS_CLASS_DELETE_FUNC_NAME(uct_lwda_copy_ep_t),
    .iface_flush              = uct_lwda_copy_iface_flush,
    .iface_fence              = uct_base_iface_fence,
    .iface_progress_enable    = uct_base_iface_progress_enable,
    .iface_progress_disable   = uct_base_iface_progress_disable,
    .iface_progress           = uct_lwda_copy_iface_progress,
    .iface_close              = UCS_CLASS_DELETE_FUNC_NAME(uct_lwda_copy_iface_t),
    .iface_query              = uct_lwda_copy_iface_query,
    .iface_get_device_address = (uct_iface_get_device_address_func_t)ucs_empty_function_return_success,
    .iface_get_address        = uct_lwda_copy_iface_get_address,
    .iface_is_reachable       = uct_lwda_copy_iface_is_reachable,
};

static void uct_lwda_copy_event_desc_init(ucs_mpool_t *mp, void *obj, void *chunk)
{
    uct_lwda_copy_event_desc_t *base = (uct_lwda_copy_event_desc_t *) obj;
    ucs_status_t status;

    memset(base, 0 , sizeof(*base));
    status = UCT_LWDA_FUNC(lwdaEventCreateWithFlags(&(base->event),
                           lwdaEventDisableTiming));
    if (UCS_OK != status) {
        ucs_error("lwdaEventCreateWithFlags Failed");
    }
}

static void uct_lwda_copy_event_desc_cleanup(ucs_mpool_t *mp, void *obj)
{
    uct_lwda_copy_event_desc_t *base = (uct_lwda_copy_event_desc_t *) obj;
    int active;

    UCT_LWDADRV_CTX_ACTIVE(active);

    if (active) {
        UCT_LWDA_FUNC(lwdaEventDestroy(base->event));
    }
}

static ucs_mpool_ops_t uct_lwda_copy_event_desc_mpool_ops = {
    .chunk_alloc   = ucs_mpool_chunk_malloc,
    .chunk_release = ucs_mpool_chunk_free,
    .obj_init      = uct_lwda_copy_event_desc_init,
    .obj_cleanup   = uct_lwda_copy_event_desc_cleanup,
};

static UCS_CLASS_INIT_FUNC(uct_lwda_copy_iface_t, uct_md_h md, uct_worker_h worker,
                           const uct_iface_params_t *params,
                           const uct_iface_config_t *tl_config)
{
    uct_lwda_copy_iface_config_t *config = ucs_derived_of(tl_config,
                                                          uct_lwda_copy_iface_config_t);
    ucs_status_t status;

    UCS_CLASS_CALL_SUPER_INIT(uct_base_iface_t, &uct_lwda_copy_iface_ops, md, worker,
                              params, tl_config UCS_STATS_ARG(params->stats_root)
                              UCS_STATS_ARG("lwda_copy"));

    if (strncmp(params->mode.device.dev_name,
                UCT_LWDA_DEV_NAME, strlen(UCT_LWDA_DEV_NAME)) != 0) {
        ucs_error("no device was found: %s", params->mode.device.dev_name);
        return UCS_ERR_NO_DEVICE;
    }

    self->id                     = ucs_generate_uuid((uintptr_t)self);
    self->config.max_poll        = config->max_poll;
    self->config.max_lwda_events = config->max_lwda_events;

    status = ucs_mpool_init(&self->lwda_event_desc,
                            0,
                            sizeof(uct_lwda_copy_event_desc_t),
                            0,
                            UCS_SYS_CACHE_LINE_SIZE,
                            128,
                            self->config.max_lwda_events,
                            &uct_lwda_copy_event_desc_mpool_ops,
                            "LWCA EVENT objects");

    if (UCS_OK != status) {
        ucs_error("mpool creation failed");
        return UCS_ERR_IO_ERROR;
    }

    self->stream_d2h = 0;
    self->stream_h2d = 0;

    ucs_queue_head_init(&self->outstanding_d2h_lwda_event_q);
    ucs_queue_head_init(&self->outstanding_h2d_lwda_event_q);

    return UCS_OK;
}

static UCS_CLASS_CLEANUP_FUNC(uct_lwda_copy_iface_t)
{
    int active;

    UCT_LWDADRV_CTX_ACTIVE(active);

    uct_base_iface_progress_disable(&self->super.super,
                                    UCT_PROGRESS_SEND | UCT_PROGRESS_RECV);
    if (active) {
        if (self->stream_h2d != 0) {
            UCT_LWDA_FUNC(lwdaStreamDestroy(self->stream_h2d));
        }

        if (self->stream_d2h != 0) {
            UCT_LWDA_FUNC(lwdaStreamDestroy(self->stream_d2h));
        }
    }

    ucs_mpool_cleanup(&self->lwda_event_desc, 1);
}

UCS_CLASS_DEFINE(uct_lwda_copy_iface_t, uct_base_iface_t);
UCS_CLASS_DEFINE_NEW_FUNC(uct_lwda_copy_iface_t, uct_iface_t, uct_md_h, uct_worker_h,
                          const uct_iface_params_t*, const uct_iface_config_t*);
static UCS_CLASS_DEFINE_DELETE_FUNC(uct_lwda_copy_iface_t, uct_iface_t);


UCT_TL_DEFINE(&uct_lwda_copy_component, lwda_copy, uct_lwda_base_query_devices,
              uct_lwda_copy_iface_t, "LWDA_COPY_",
              uct_lwda_copy_iface_config_table, uct_lwda_copy_iface_config_t);
