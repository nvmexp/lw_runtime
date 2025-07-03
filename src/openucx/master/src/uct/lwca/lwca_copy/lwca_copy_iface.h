/**
 * Copyright (C) Mellanox Technologies Ltd. 2017.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCT_LWDA_COPY_IFACE_H
#define UCT_LWDA_COPY_IFACE_H

#include <uct/base/uct_iface.h>
#include <uct/lwca/base/lwda_iface.h>


typedef uint64_t uct_lwda_copy_iface_addr_t;


typedef struct uct_lwda_copy_iface {
    uct_base_iface_t            super;
    uct_lwda_copy_iface_addr_t  id;
    ucs_mpool_t                 lwda_event_desc;
    ucs_queue_head_t            outstanding_d2h_lwda_event_q;
    ucs_queue_head_t            outstanding_h2d_lwda_event_q;
    lwdaStream_t                stream_d2h;
    lwdaStream_t                stream_h2d;
    struct {
        unsigned                max_poll;
        unsigned                max_lwda_events;
    } config;
} uct_lwda_copy_iface_t;


typedef struct uct_lwda_copy_iface_config {
    uct_iface_config_t      super;
    unsigned                max_poll;
    unsigned                max_lwda_events;
} uct_lwda_copy_iface_config_t;


typedef struct uct_lwda_copy_event_desc {
    lwdaEvent_t event;
    uct_completion_t *comp;
    ucs_queue_elem_t  queue;
} uct_lwda_copy_event_desc_t;

#endif
