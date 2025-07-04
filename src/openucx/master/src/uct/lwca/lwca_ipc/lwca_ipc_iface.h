/**
 * Copyright (C) Mellanox Technologies Ltd. 2018.  ALL RIGHTS RESERVED.
 * Copyright (c) 2018, LWPU CORPORATION. All rights reserved.
 * See file LICENSE for terms.
 */

#ifndef UCT_LWDA_IPC_IFACE_H
#define UCT_LWDA_IPC_IFACE_H

#include <uct/base/uct_iface.h>
#include <ucs/arch/cpu.h>
#include <lwda_runtime.h>
#include <lwca.h>

#include "lwda_ipc_md.h"
#include "lwda_ipc_ep.h"


#define UCT_LWDA_IPC_MAX_PEERS  16


typedef struct uct_lwda_ipc_iface {
    uct_base_iface_t super;
    ucs_mpool_t      event_desc;              /* lwca event desc */
    ucs_queue_head_t outstanding_d2d_event_q; /* stream for outstanding d2d */
    int              eventfd;              /* get event notifications */
    int              streams_initialized;     /* indicates if stream created */
    LWstream         stream_d2d[UCT_LWDA_IPC_MAX_PEERS];
                                              /* per-peer stream */
    unsigned long    stream_refcount[UCT_LWDA_IPC_MAX_PEERS];
                                              /* per stream outstanding ops */
    struct {
        unsigned     max_poll;                /* query attempts w.o success */
        unsigned     max_streams;             /* # conlwrrent streams for || progress*/
        unsigned     max_lwda_ipc_events;     /* max mpool entries */
        int          enable_cache;            /* enable/disable ipc handle cache */
    } config;
    ucs_status_t     (*map_memhandle)(void *context, uct_lwda_ipc_key_t *key,
                                      void **map_addr);
    ucs_status_t     (*unmap_memhandle)(void *rem_cache, uintptr_t d_bptr,
                                        void *mapped_addr, int cache_enabled);
} uct_lwda_ipc_iface_t;


typedef struct uct_lwda_ipc_iface_config {
    uct_iface_config_t      super;
    unsigned                max_poll;
    unsigned                max_streams;
    int                     enable_cache;
    unsigned                max_lwda_ipc_events;
} uct_lwda_ipc_iface_config_t;


typedef struct uct_lwda_ipc_event_desc {
    LWevent           event;
    void              *mapped_addr;
    unsigned          stream_id;
    uct_completion_t  *comp;
    ucs_queue_elem_t  queue;
    uct_lwda_ipc_ep_t *ep;
    void              *cache;
    uintptr_t         d_bptr;
} uct_lwda_ipc_event_desc_t;


ucs_status_t uct_lwda_ipc_iface_init_streams(uct_lwda_ipc_iface_t *iface);
#endif
