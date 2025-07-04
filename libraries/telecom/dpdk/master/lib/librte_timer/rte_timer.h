/* SPDX-License-Identifier: BSD-3-Clause
 * Copyright(c) 2010-2014 Intel Corporation
 */

#ifndef _RTE_TIMER_H_
#define _RTE_TIMER_H_

/**
 * @file
 RTE Timer
 *
 * This library provides a timer service to RTE Data Plane exelwtion
 * units that allows the exelwtion of callback functions asynchronously.
 *
 * - Timers can be periodic or single (one-shot).
 * - The timers can be loaded from one core and exelwted on another. This has
 *   to be specified in the call to rte_timer_reset().
 * - High precision is possible. NOTE: this depends on the call frequency to
 *   rte_timer_manage() that check the timer expiration for the local core.
 * - If not used in an application, for improved performance, it can be
 *   disabled at compilation time by not calling the rte_timer_manage()
 *   to improve performance.
 *
 * The timer library uses the rte_get_hpet_cycles() function that
 * uses the HPET, when available, to provide a reliable time reference. [HPET
 * routines are provided by EAL, which falls back to using the chip TSC (time-
 * stamp counter) as fallback when HPET is not available]
 *
 * This library provides an interface to add, delete and restart a
 * timer. The API is based on the BSD callout(9) API with a few
 * differences.
 *
 * See the RTE architecture documentation for more information about the
 * design of this library.
 */

#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include <rte_common.h>
#include <rte_config.h>
#include <rte_spinlock.h>

#ifdef __cplusplus
extern "C" {
#endif

#define RTE_TIMER_STOP    0 /**< State: timer is stopped. */
#define RTE_TIMER_PENDING 1 /**< State: timer is scheduled. */
#define RTE_TIMER_RUNNING 2 /**< State: timer function is running. */
#define RTE_TIMER_CONFIG  3 /**< State: timer is being configured. */

#define RTE_TIMER_NO_OWNER -2 /**< Timer has no owner. */

/**
 * Timer type: Periodic or single (one-shot).
 */
enum rte_timer_type {
	SINGLE,
	PERIODICAL
};

/**
 * Timer status: A union of the state (stopped, pending, running,
 * config) and an owner (the id of the lcore that owns the timer).
 */
union rte_timer_status {
	RTE_STD_C11
	struct {
		uint16_t state;  /**< Stop, pending, running, config. */
		int16_t owner;   /**< The lcore that owns the timer. */
	};
	uint32_t u32;            /**< To atomic-set status + owner. */
};

#ifdef RTE_LIBRTE_TIMER_DEBUG
/**
 * A structure that stores the timer statistics (per-lcore).
 */
struct rte_timer_debug_stats {
	uint64_t reset;   /**< Number of success calls to rte_timer_reset(). */
	uint64_t stop;    /**< Number of success calls to rte_timer_stop(). */
	uint64_t manage;  /**< Number of calls to rte_timer_manage(). */
	uint64_t pending; /**< Number of pending/running timers. */
};
#endif

struct rte_timer;

/**
 * Callback function type for timer expiry.
 */
typedef void (*rte_timer_cb_t)(struct rte_timer *, void *);

#define MAX_SKIPLIST_DEPTH 10

/**
 * A structure describing a timer in RTE.
 */
struct rte_timer
{
	uint64_t expire;       /**< Time when timer expire. */
	struct rte_timer *sl_next[MAX_SKIPLIST_DEPTH];
	volatile union rte_timer_status status; /**< Status of timer. */
	uint64_t period;       /**< Period of timer (0 if not periodic). */
	rte_timer_cb_t f;      /**< Callback function. */
	void *arg;             /**< Argument to callback function. */
};


#ifdef __cplusplus
/**
 * A C++ static initializer for a timer structure.
 */
#define RTE_TIMER_INITIALIZER {             \
	0,                                      \
	{NULL},                                 \
	{{RTE_TIMER_STOP, RTE_TIMER_NO_OWNER}}, \
	0,                                      \
	NULL,                                   \
	NULL,                                   \
	}
#else
/**
 * A static initializer for a timer structure.
 */
#define RTE_TIMER_INITIALIZER {                      \
		.status = {{                         \
			.state = RTE_TIMER_STOP,     \
			.owner = RTE_TIMER_NO_OWNER, \
		}},                                  \
	}
#endif

/**
 * Allocate a timer data instance in shared memory to track a set of pending
 * timer lists.
 *
 * @param id_ptr
 *   Pointer to variable into which to write the identifier of the allocated
 *   timer data instance.
 *
 * @return
 *   - 0: Success
 *   - -ENOSPC: maximum number of timer data instances already allocated
 */
int rte_timer_data_alloc(uint32_t *id_ptr);

/**
 * Deallocate a timer data instance.
 *
 * @param id
 *   Identifier of the timer data instance to deallocate.
 *
 * @return
 *   - 0: Success
 *   - -EILWAL: invalid timer data instance identifier
 */
int rte_timer_data_dealloc(uint32_t id);

/**
 * Initialize the timer library.
 *
 * Initializes internal variables (list, locks and so on) for the RTE
 * timer library.
 *
 * @note
 *   This function must be called in every process before using the library.
 *
 * @return
 *   - 0: Success
 *   - -ENOMEM: Unable to allocate memory needed to initialize timer
 *      subsystem
 */
int rte_timer_subsystem_init(void);

/**
 * Free timer subsystem resources.
 */
void rte_timer_subsystem_finalize(void);

/**
 * Initialize a timer handle.
 *
 * The rte_timer_init() function initializes the timer handle *tim*
 * for use. No operations can be performed on a timer before it is
 * initialized.
 *
 * @param tim
 *   The timer to initialize.
 */
void rte_timer_init(struct rte_timer *tim);

/**
 * Reset and start the timer associated with the timer handle.
 *
 * The rte_timer_reset() function resets and starts the timer
 * associated with the timer handle *tim*. When the timer expires after
 * *ticks* HPET cycles, the function specified by *fct* will be called
 * with the argument *arg* on core *tim_lcore*.
 *
 * If the timer associated with the timer handle is already running
 * (in the RUNNING state), the function will fail. The user has to check
 * the return value of the function to see if there is a chance that the
 * timer is in the RUNNING state.
 *
 * If the timer is being configured on another core (the CONFIG state),
 * it will also fail.
 *
 * If the timer is pending or stopped, it will be rescheduled with the
 * new parameters.
 *
 * @param tim
 *   The timer handle.
 * @param ticks
 *   The number of cycles (see rte_get_hpet_hz()) before the callback
 *   function is called.
 * @param type
 *   The type can be either:
 *   - PERIODICAL: The timer is automatically reloaded after exelwtion
 *     (returns to the PENDING state)
 *   - SINGLE: The timer is one-shot, that is, the timer goes to a
 *     STOPPED state after exelwtion.
 * @param tim_lcore
 *   The ID of the lcore where the timer callback function has to be
 *   exelwted. If tim_lcore is LCORE_ID_ANY, the timer library will
 *   launch it on a different core for each call (round-robin).
 * @param fct
 *   The callback function of the timer.
 * @param arg
 *   The user argument of the callback function.
 * @return
 *   - 0: Success; the timer is scheduled.
 *   - (-1): Timer is in the RUNNING or CONFIG state.
 */
int rte_timer_reset(struct rte_timer *tim, uint64_t ticks,
		    enum rte_timer_type type, unsigned tim_lcore,
		    rte_timer_cb_t fct, void *arg);

/**
 * Loop until rte_timer_reset() succeeds.
 *
 * Reset and start the timer associated with the timer handle. Always
 * succeed. See rte_timer_reset() for details.
 *
 * @param tim
 *   The timer handle.
 * @param ticks
 *   The number of cycles (see rte_get_hpet_hz()) before the callback
 *   function is called.
 * @param type
 *   The type can be either:
 *   - PERIODICAL: The timer is automatically reloaded after exelwtion
 *     (returns to the PENDING state)
 *   - SINGLE: The timer is one-shot, that is, the timer goes to a
 *     STOPPED state after exelwtion.
 * @param tim_lcore
 *   The ID of the lcore where the timer callback function has to be
 *   exelwted. If tim_lcore is LCORE_ID_ANY, the timer library will
 *   launch it on a different core for each call (round-robin).
 * @param fct
 *   The callback function of the timer.
 * @param arg
 *   The user argument of the callback function.
 *
 * @note
 *   This API should not be called inside a timer's callback function to
 *   reset another timer; doing so could hang in certain scenarios. Instead,
 *   the rte_timer_reset() API can be called directly and its return code
 *   can be checked for success or failure.
 */
void
rte_timer_reset_sync(struct rte_timer *tim, uint64_t ticks,
		     enum rte_timer_type type, unsigned tim_lcore,
		     rte_timer_cb_t fct, void *arg);

/**
 * Stop a timer.
 *
 * The rte_timer_stop() function stops the timer associated with the
 * timer handle *tim*. It may fail if the timer is lwrrently running or
 * being configured.
 *
 * If the timer is pending or stopped (for instance, already expired),
 * the function will succeed. The timer handle tim must have been
 * initialized using rte_timer_init(), otherwise, undefined behavior
 * will occur.
 *
 * This function can be called safely from a timer callback. If it
 * succeeds, the timer is not referenced anymore by the timer library
 * and the timer structure can be freed (even in the callback
 * function).
 *
 * @param tim
 *   The timer handle.
 * @return
 *   - 0: Success; the timer is stopped.
 *   - (-1): The timer is in the RUNNING or CONFIG state.
 */
int rte_timer_stop(struct rte_timer *tim);

/**
 * Loop until rte_timer_stop() succeeds.
 *
 * After a call to this function, the timer identified by *tim* is
 * stopped. See rte_timer_stop() for details.
 *
 * @param tim
 *   The timer handle.
 *
 * @note
 *   This API should not be called inside a timer's callback function to
 *   stop another timer; doing so could hang in certain scenarios. Instead, the
 *   rte_timer_stop() API can be called directly and its return code can
 *   be checked for success or failure.
 */
void rte_timer_stop_sync(struct rte_timer *tim);

/**
 * Test if a timer is pending.
 *
 * The rte_timer_pending() function tests the PENDING status
 * of the timer handle *tim*. A PENDING timer is one that has been
 * scheduled and whose function has not yet been called.
 *
 * @param tim
 *   The timer handle.
 * @return
 *   - 0: The timer is not pending.
 *   - 1: The timer is pending.
 */
int rte_timer_pending(struct rte_timer *tim);

/**
 * @warning
 * @b EXPERIMENTAL: this API may change without prior notice
 *
 * Time until the next timer on the current lcore
 * This function gives the ticks until the next timer will be active.
 *
 * @return
 *   - -EILWAL: invalid timer data instance identifier
 *   - -ENOENT: no timer pending
 *   - 0: a timer is pending and will run at next rte_timer_manage()
 *   - >0: ticks until the next timer is ready
 */
__rte_experimental
int64_t rte_timer_next_ticks(void);

/**
 * Manage the timer list and execute callback functions.
 *
 * This function must be called periodically from EAL lcores
 * main_loop(). It browses the list of pending timers and runs all
 * timers that are expired.
 *
 * The precision of the timer depends on the call frequency of this
 * function. However, the more often the function is called, the more
 * CPU resources it will use.
 *
 * @return
 *   - 0: Success
 *   - -EILWAL: timer subsystem not yet initialized
 */
int rte_timer_manage(void);

/**
 * Dump statistics about timers.
 *
 * @param f
 *   A pointer to a file for output
 * @return
 *   - 0: Success
 *   - -EILWAL: timer subsystem not yet initialized
 */
int rte_timer_dump_stats(FILE *f);

/**
 * This function is the same as rte_timer_reset(), except that it allows a
 * caller to specify the rte_timer_data instance containing the list to which
 * the timer should be added.
 *
 * @see rte_timer_reset()
 *
 * @param timer_data_id
 *   An identifier indicating which instance of timer data should be used for
 *   this operation.
 * @param tim
 *   The timer handle.
 * @param ticks
 *   The number of cycles (see rte_get_hpet_hz()) before the callback
 *   function is called.
 * @param type
 *   The type can be either:
 *   - PERIODICAL: The timer is automatically reloaded after exelwtion
 *     (returns to the PENDING state)
 *   - SINGLE: The timer is one-shot, that is, the timer goes to a
 *     STOPPED state after exelwtion.
 * @param tim_lcore
 *   The ID of the lcore where the timer callback function has to be
 *   exelwted. If tim_lcore is LCORE_ID_ANY, the timer library will
 *   launch it on a different core for each call (round-robin).
 * @param fct
 *   The callback function of the timer. This parameter can be NULL if (and
 *   only if) rte_timer_alt_manage() will be used to manage this timer.
 * @param arg
 *   The user argument of the callback function.
 * @return
 *   - 0: Success; the timer is scheduled.
 *   - (-1): Timer is in the RUNNING or CONFIG state.
 *   - -EILWAL: invalid timer_data_id
 */
int
rte_timer_alt_reset(uint32_t timer_data_id, struct rte_timer *tim,
		    uint64_t ticks, enum rte_timer_type type,
		    unsigned int tim_lcore, rte_timer_cb_t fct, void *arg);

/**
 * This function is the same as rte_timer_stop(), except that it allows a
 * caller to specify the rte_timer_data instance containing the list from which
 * this timer should be removed.
 *
 * @see rte_timer_stop()
 *
 * @param timer_data_id
 *   An identifier indicating which instance of timer data should be used for
 *   this operation.
 * @param tim
 *   The timer handle.
 * @return
 *   - 0: Success; the timer is stopped.
 *   - (-1): The timer is in the RUNNING or CONFIG state.
 *   - -EILWAL: invalid timer_data_id
 */
int
rte_timer_alt_stop(uint32_t timer_data_id, struct rte_timer *tim);

/**
 * Callback function type for rte_timer_alt_manage().
 */
typedef void (*rte_timer_alt_manage_cb_t)(struct rte_timer *tim);

/**
 * Manage a set of timer lists and execute the specified callback function for
 * all expired timers. This function is similar to rte_timer_manage(), except
 * that it allows a caller to specify the timer_data instance that should
 * be operated on, as well as a set of lcore IDs identifying which timer lists
 * should be processed.  Callback functions of individual timers are ignored.
 *
 * @see rte_timer_manage()
 *
 * @param timer_data_id
 *   An identifier indicating which instance of timer data should be used for
 *   this operation.
 * @param poll_lcores
 *   An array of lcore ids identifying the timer lists that should be processed.
 *   NULL is allowed - if NULL, the timer list corresponding to the lcore
 *   calling this routine is processed (same as rte_timer_manage()).
 * @param n_poll_lcores
 *   The size of the poll_lcores array. If 'poll_lcores' is NULL, this parameter
 *   is ignored.
 * @param f
 *   The callback function which should be called for all expired timers.
 * @return
 *   - 0: success
 *   - -EILWAL: invalid timer_data_id
 */
int
rte_timer_alt_manage(uint32_t timer_data_id, unsigned int *poll_lcores,
		     int n_poll_lcores, rte_timer_alt_manage_cb_t f);

/**
 * Callback function type for rte_timer_stop_all().
 */
typedef void (*rte_timer_stop_all_cb_t)(struct rte_timer *tim, void *arg);

/**
 * Walk the pending timer lists for the specified lcore IDs, and for each timer
 * that is encountered, stop it and call the specified callback function to
 * process it further.
 *
 * @param timer_data_id
 *   An identifier indicating which instance of timer data should be used for
 *   this operation.
 * @param walk_lcores
 *   An array of lcore ids identifying the timer lists that should be processed.
 * @param nb_walk_lcores
 *   The size of the walk_lcores array.
 * @param f
 *   The callback function which should be called for each timers. Can be NULL.
 * @param f_arg
 *   An arbitrary argument that will be passed to f, if it is called.
 * @return
 *   - 0: success
 *   - EILWAL: invalid timer_data_id
 */
int
rte_timer_stop_all(uint32_t timer_data_id, unsigned int *walk_lcores,
		   int nb_walk_lcores, rte_timer_stop_all_cb_t f, void *f_arg);

/**
 * This function is the same as rte_timer_dump_stats(), except that it allows
 * the caller to specify the rte_timer_data instance that should be used.
 *
 * @see rte_timer_dump_stats()
 *
 * @param timer_data_id
 *   An identifier indicating which instance of timer data should be used for
 *   this operation.
 * @param f
 *   A pointer to a file for output
 * @return
 *   - 0: success
 *   - -EILWAL: invalid timer_data_id
 */
int
rte_timer_alt_dump_stats(uint32_t timer_data_id, FILE *f);

#ifdef __cplusplus
}
#endif

#endif /* _RTE_TIMER_H_ */
