/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2017.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCP_THREAD_H_
#define UCP_THREAD_H_

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include <ucs/async/async.h>
#include <ucs/async/thread.h>
#include <ucs/type/spinlock.h>


/*
 * Multi-thread mode
 */
typedef enum ucp_mt_type {
    UCP_MT_TYPE_NONE = 0,
    UCP_MT_TYPE_SPINLOCK,
    UCP_MT_TYPE_MUTEX
} ucp_mt_type_t;


/**
 * Multi-thread lock
 */
typedef struct ucp_mt_lock {
    ucp_mt_type_t                 mt_type;
    union {
        /* Lock for multithreading support. Either spinlock or mutex is used at
           at one time. Spinlock is the default option. */
        pthread_mutex_t           mt_mutex;
        ucs_relwrsive_spinlock_t  mt_spinlock;
    } lock;
} ucp_mt_lock_t;


#define UCP_THREAD_IS_REQUIRED(_lock_ptr) \
    ((_lock_ptr)->mt_type)
#define UCP_THREAD_LOCK_INIT(_lock_ptr) \
    do { \
        if ((_lock_ptr)->mt_type == UCP_MT_TYPE_MUTEX) { \
            pthread_mutex_init(&((_lock_ptr)->lock.mt_mutex), NULL); \
        } else { \
            ucs_relwrsive_spinlock_init(&((_lock_ptr)->lock.mt_spinlock), 0); \
        } \
    } while (0)
#define UCP_THREAD_LOCK_FINALIZE(_lock_ptr)                                  \
    do { \
        ucs_status_t _status; \
        \
        if ((_lock_ptr)->mt_type == UCP_MT_TYPE_MUTEX) { \
            pthread_mutex_destroy(&((_lock_ptr)->lock.mt_mutex)); \
        } else { \
            _status = ucs_relwrsive_spinlock_destroy(&((_lock_ptr)->lock.mt_spinlock)); \
            if (_status != UCS_OK) { \
                ucs_warn("ucs_relwrsive_spinlock_destroy() failed (%d)", _status); \
            } \
        } \
    } while (0)
#define UCP_THREAD_CS_ENTER(_lock_ptr)                                  \
    {                                                                   \
        if ((_lock_ptr)->mt_type == UCP_MT_TYPE_MUTEX) {                \
            pthread_mutex_lock(&((_lock_ptr)->lock.mt_mutex));          \
        } else {                                                        \
            ucs_relwrsive_spin_lock(&((_lock_ptr)->lock.mt_spinlock));  \
        }                                                               \
    }
#define UCP_THREAD_CS_EXIT(_lock_ptr)                                   \
    {                                                                   \
        if ((_lock_ptr)->mt_type == UCP_MT_TYPE_MUTEX) {                \
            pthread_mutex_unlock(&((_lock_ptr)->lock.mt_mutex));        \
        } else {                                                        \
            ucs_relwrsive_spin_unlock(&((_lock_ptr)->lock.mt_spinlock));\
        }                                                               \
    }

#endif
