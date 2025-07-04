/**
 * Copyright (C) Mellanox Technologies Ltd. 2018.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCS_REG_CACHE_INT_H_
#define UCS_REG_CACHE_INT_H_

#include <ucs/type/spinlock.h>

/* Names of rcache stats counters */
enum {
    UCS_RCACHE_GETS,                /* number of get operations */
    UCS_RCACHE_HITS_FAST,           /* number of fast path hits */
    UCS_RCACHE_HITS_SLOW,           /* number of slow path hits */
    UCS_RCACHE_MISSES,              /* number of misses */
    UCS_RCACHE_MERGES,              /* number of region merges */
    UCS_RCACHE_UNMAPS,              /* number of memory unmap events */
    UCS_RCACHE_UNMAP_ILWALIDATES,   /* number of regions ilwalidated because
                                       of unmap events */
    UCS_RCACHE_PUTS,                /* number of put operations */
    UCS_RCACHE_REGS,                /* number of memory registrations */
    UCS_RCACHE_DEREGS,              /* number of memory deregistrations */
    UCS_RCACHE_STAT_LAST
};


struct ucs_rcache {
    ucs_rcache_params_t      params;   /**< rcache parameters (immutable) */
    pthread_rwlock_t         lock;     /**< Protects the page table and all regions
                                            whose refcount is 0 */
    ucs_pgtable_t            pgtable;  /**< page table to hold the regions */

    ucs_relwrsive_spinlock_t ilw_lock; /**< Lock for ilw_q and ilw_mp. This is a
                                          separate lock because we may want to put
                                          regions on ilw_q while the page table
                                          lock is held by the calling context */
    ucs_queue_head_t         ilw_q;    /**< Regions which were ilwalidated during
                                            memory events */
    ucs_mpool_t              ilw_mp;   /**< Memory pool to allocate entries for ilw_q,
                                            since we cannot use regulat malloc().
                                            The backing storage is original mmap()
                                            which does not generate memory events */
    char                     *name;
    UCS_STATS_NODE_DECLARE(stats)
};

#endif
