/**
 * Copyright (C) Mellanox Technologies Ltd. 2018.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCT_LWDA_IPC_CACHE_H_
#define UCT_LWDA_IPC_CACHE_H_

#include <ucs/datastruct/pgtable.h>
#include <ucs/datastruct/list.h>
#include "lwda_ipc_md.h"
#include <lwca.h>
#include <lwda_runtime.h>


typedef struct uct_lwda_ipc_cache         uct_lwda_ipc_cache_t;
typedef struct uct_lwda_ipc_cache_region  uct_lwda_ipc_cache_region_t;


typedef struct uct_lwda_ipc_rem_memh uct_lwda_ipc_rem_memh_t;


struct uct_lwda_ipc_cache_region {
    ucs_pgt_region_t        super;        /**< Base class - page table region */
    ucs_list_link_t         list;         /**< List element */
    uct_lwda_ipc_key_t      key;          /**< Remote memory key */
    void                    *mapped_addr; /**< Local mapped address */
    uint64_t                refcount;     /**< Track inflight ops before unmapping*/
};


struct uct_lwda_ipc_cache {
    pthread_rwlock_t      lock;       /**< protests the page table */
    ucs_pgtable_t         pgtable;    /**< Page table to hold the regions */
    char                  *name;      /**< Name */
};


ucs_status_t uct_lwda_ipc_create_cache(uct_lwda_ipc_cache_t **cache,
                                       const char *name);


void uct_lwda_ipc_destroy_cache(uct_lwda_ipc_cache_t *cache);


ucs_status_t uct_lwda_ipc_map_memhandle(void *arg, uct_lwda_ipc_key_t *key,
                                        void **mapped_addr);
ucs_status_t uct_lwda_ipc_unmap_memhandle(void *rem_cache, uintptr_t d_bptr,
                                          void *mapped_addr, int cache_enabled);
#endif
