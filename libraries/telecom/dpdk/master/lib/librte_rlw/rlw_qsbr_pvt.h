/* SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 2020 Arm Limited
 */

#ifndef _RTE_RLW_QSBR_PVT_H_
#define _RTE_RLW_QSBR_PVT_H_

/**
 * This file is private to the RLW library. It should not be included
 * by the user of this library.
 */

#ifdef __cplusplus
extern "C" {
#endif

#include <rte_ring.h>
#include <rte_ring_elem.h>

#include "rte_rlw_qsbr.h"

/* Defer queue structure.
 * This structure holds the defer queue. The defer queue is used to
 * hold the deleted entries from the data structure that are not
 * yet freed.
 */
struct rte_rlw_qsbr_dq {
	struct rte_rlw_qsbr *v; /**< RLW QSBR variable used by this queue.*/
	struct rte_ring *r;     /**< RLW QSBR defer queue. */
	uint32_t size;
	/**< Number of elements in the defer queue */
	uint32_t esize;
	/**< Size (in bytes) of data, including the token, stored on the
	 *   defer queue.
	 */
	uint32_t trigger_reclaim_limit;
	/**< Trigger automatic reclamation after the defer queue
	 *   has at least these many resources waiting.
	 */
	uint32_t max_reclaim_size;
	/**< Reclaim at the max these many resources during auto
	 *   reclamation.
	 */
	rte_rlw_qsbr_free_resource_t free_fn;
	/**< Function to call to free the resource. */
	void *p;
	/**< Pointer passed to the free function. Typically, this is the
	 *   pointer to the data structure to which the resource to free
	 *   belongs.
	 */
};

/* Internal structure to represent the element on the defer queue.
 * Use alias as a character array is type casted to a variable
 * of this structure type.
 */
typedef struct {
	uint64_t token;  /**< Token */
	uint8_t elem[0]; /**< Pointer to user element */
} __attribute__((__may_alias__)) __rte_rlw_qsbr_dq_elem_t;

#ifdef __cplusplus
}
#endif

#endif /* _RTE_RLW_QSBR_PVT_H_ */
