/* SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 2007-2013 Broadcom Corporation.
 *
 * Eric Davis        <edavis@broadcom.com>
 * David Christensen <davidch@broadcom.com>
 * Gary Zambrano     <zambrano@broadcom.com>
 *
 * Copyright (c) 2013-2015 Brocade Communications Systems, Inc.
 * Copyright (c) 2015-2018 Cavium Inc.
 * All rights reserved.
 * www.cavium.com
 */

#include "bnx2x.h"
#include "ecore_init.h"

/**** Exe Queue interfaces ****/

/**
 * ecore_exe_queue_init - init the Exe Queue object
 *
 * @o:		pointer to the object
 * @exe_len:	length
 * @owner:	pointer to the owner
 * @validate:	validate function pointer
 * @optimize:	optimize function pointer
 * @exec:	execute function pointer
 * @get:	get function pointer
 */
static void
ecore_exe_queue_init(struct bnx2x_softc *sc __rte_unused,
		     struct ecore_exe_queue_obj *o,
		     int exe_len,
		     union ecore_qable_obj *owner,
		     exe_q_validate validate,
		     exe_q_remove remove,
		     exe_q_optimize optimize, exe_q_exelwte exec, exe_q_get get)
{
	ECORE_MEMSET(o, 0, sizeof(*o));

	ECORE_LIST_INIT(&o->exe_queue);
	ECORE_LIST_INIT(&o->pending_comp);

	ECORE_SPIN_LOCK_INIT(&o->lock, sc);

	o->exe_chunk_len = exe_len;
	o->owner = owner;

	/* Owner specific callbacks */
	o->validate = validate;
	o->remove = remove;
	o->optimize = optimize;
	o->execute = exec;
	o->get = get;

	ECORE_MSG(sc, "Setup the exelwtion queue with the chunk length of %d",
		  exe_len);
}

static void ecore_exe_queue_free_elem(struct bnx2x_softc *sc __rte_unused,
				      struct ecore_exeq_elem *elem)
{
	ECORE_MSG(sc, "Deleting an exe_queue element");
	ECORE_FREE(sc, elem, sizeof(*elem));
}

static inline int ecore_exe_queue_length(struct ecore_exe_queue_obj *o)
{
	struct ecore_exeq_elem *elem;
	int cnt = 0;

	ECORE_SPIN_LOCK_BH(&o->lock);

	ECORE_LIST_FOR_EACH_ENTRY(elem, &o->exe_queue, link,
				  struct ecore_exeq_elem) cnt++;

	ECORE_SPIN_UNLOCK_BH(&o->lock);

	return cnt;
}

/**
 * ecore_exe_queue_add - add a new element to the exelwtion queue
 *
 * @sc:		driver handle
 * @o:		queue
 * @cmd:	new command to add
 * @restore:	true - do not optimize the command
 *
 * If the element is optimized or is illegal, frees it.
 */
static int ecore_exe_queue_add(struct bnx2x_softc *sc,
			       struct ecore_exe_queue_obj *o,
			       struct ecore_exeq_elem *elem, int restore)
{
	int rc;

	ECORE_SPIN_LOCK_BH(&o->lock);

	if (!restore) {
		/* Try to cancel this element queue */
		rc = o->optimize(sc, o->owner, elem);
		if (rc)
			goto free_and_exit;

		/* Check if this request is ok */
		rc = o->validate(sc, o->owner, elem);
		if (rc) {
			ECORE_MSG(sc, "Preamble failed: %d", rc);
			goto free_and_exit;
		}
	}

	/* If so, add it to the exelwtion queue */
	ECORE_LIST_PUSH_TAIL(&elem->link, &o->exe_queue);

	ECORE_SPIN_UNLOCK_BH(&o->lock);

	return ECORE_SUCCESS;

free_and_exit:
	ecore_exe_queue_free_elem(sc, elem);

	ECORE_SPIN_UNLOCK_BH(&o->lock);

	return rc;
}

static void __ecore_exe_queue_reset_pending(struct bnx2x_softc *sc, struct ecore_exe_queue_obj
					    *o)
{
	struct ecore_exeq_elem *elem;

	while (!ECORE_LIST_IS_EMPTY(&o->pending_comp)) {
		elem = ECORE_LIST_FIRST_ENTRY(&o->pending_comp,
					      struct ecore_exeq_elem, link);

		ECORE_LIST_REMOVE_ENTRY(&elem->link, &o->pending_comp);
		ecore_exe_queue_free_elem(sc, elem);
	}
}

static inline void ecore_exe_queue_reset_pending(struct bnx2x_softc *sc,
						 struct ecore_exe_queue_obj *o)
{
	ECORE_SPIN_LOCK_BH(&o->lock);

	__ecore_exe_queue_reset_pending(sc, o);

	ECORE_SPIN_UNLOCK_BH(&o->lock);
}

/**
 * ecore_exe_queue_step - execute one exelwtion chunk atomically
 *
 * @sc:			driver handle
 * @o:			queue
 * @ramrod_flags:	flags
 *
 * (Should be called while holding the exe_queue->lock).
 */
static int ecore_exe_queue_step(struct bnx2x_softc *sc,
				struct ecore_exe_queue_obj *o,
				uint32_t *ramrod_flags)
{
	struct ecore_exeq_elem *elem, spacer;
	int lwr_len = 0, rc;

	ECORE_MEMSET(&spacer, 0, sizeof(spacer));

	/* Next step should not be performed until the current is finished,
	 * unless a DRV_CLEAR_ONLY bit is set. In this case we just want to
	 * properly clear object internals without sending any command to the FW
	 * which also implies there won't be any completion to clear the
	 * 'pending' list.
	 */
	if (!ECORE_LIST_IS_EMPTY(&o->pending_comp)) {
		if (ECORE_TEST_BIT(RAMROD_DRV_CLR_ONLY, ramrod_flags)) {
			ECORE_MSG(sc,
				  "RAMROD_DRV_CLR_ONLY requested: resetting a pending_comp list");
			__ecore_exe_queue_reset_pending(sc, o);
		} else {
			return ECORE_PENDING;
		}
	}

	/* Run through the pending commands list and create a next
	 * exelwtion chunk.
	 */
	while (!ECORE_LIST_IS_EMPTY(&o->exe_queue)) {
		elem = ECORE_LIST_FIRST_ENTRY(&o->exe_queue,
					      struct ecore_exeq_elem, link);
		ECORE_DBG_BREAK_IF(!elem->cmd_len);

		if (lwr_len + elem->cmd_len <= o->exe_chunk_len) {
			lwr_len += elem->cmd_len;
			/* Prevent from both lists being empty when moving an
			 * element. This will allow the call of
			 * ecore_exe_queue_empty() without locking.
			 */
			ECORE_LIST_PUSH_TAIL(&spacer.link, &o->pending_comp);
			mb();
			ECORE_LIST_REMOVE_ENTRY(&elem->link, &o->exe_queue);
			ECORE_LIST_PUSH_TAIL(&elem->link, &o->pending_comp);
			ECORE_LIST_REMOVE_ENTRY(&spacer.link, &o->pending_comp);
		} else
			break;
	}

	/* Sanity check */
	if (!lwr_len)
		return ECORE_SUCCESS;

	rc = o->execute(sc, o->owner, &o->pending_comp, ramrod_flags);
	if (rc < 0)
		/* In case of an error return the commands back to the queue
		 *  and reset the pending_comp.
		 */
		ECORE_LIST_SPLICE_INIT(&o->pending_comp, &o->exe_queue);
	else if (!rc)
		/* If zero is returned, means there are no outstanding pending
		 * completions and we may dismiss the pending list.
		 */
		__ecore_exe_queue_reset_pending(sc, o);

	return rc;
}

static inline int ecore_exe_queue_empty(struct ecore_exe_queue_obj *o)
{
	int empty = ECORE_LIST_IS_EMPTY(&o->exe_queue);

	/* Don't reorder!!! */
	mb();

	return empty && ECORE_LIST_IS_EMPTY(&o->pending_comp);
}

static struct ecore_exeq_elem *ecore_exe_queue_alloc_elem(struct
							  bnx2x_softc *sc
							  __rte_unused)
{
	ECORE_MSG(sc, "Allocating a new exe_queue element");
	return ECORE_ZALLOC(sizeof(struct ecore_exeq_elem), GFP_ATOMIC, sc);
}

/************************ raw_obj functions ***********************************/
static bool ecore_raw_check_pending(struct ecore_raw_obj *o)
{
	/*
	 * !! colwerts the value returned by ECORE_TEST_BIT such that it
	 * is guaranteed not to be truncated regardless of int definition.
	 *
	 * Note we cannot simply define the function's return value type
	 * to match the type returned by ECORE_TEST_BIT, as it varies by
	 * platform/implementation.
	 */

	return ! !ECORE_TEST_BIT(o->state, o->pstate);
}

static void ecore_raw_clear_pending(struct ecore_raw_obj *o)
{
	ECORE_SMP_MB_BEFORE_CLEAR_BIT();
	ECORE_CLEAR_BIT(o->state, o->pstate);
	ECORE_SMP_MB_AFTER_CLEAR_BIT();
}

static void ecore_raw_set_pending(struct ecore_raw_obj *o)
{
	ECORE_SMP_MB_BEFORE_CLEAR_BIT();
	ECORE_SET_BIT(o->state, o->pstate);
	ECORE_SMP_MB_AFTER_CLEAR_BIT();
}

/**
 * ecore_state_wait - wait until the given bit(state) is cleared
 *
 * @sc:		device handle
 * @state:	state which is to be cleared
 * @state_p:	state buffer
 *
 */
static int ecore_state_wait(struct bnx2x_softc *sc, int state,
			    uint32_t *pstate)
{
	/* can take a while if any port is running */
	int cnt = 5000;

	if (CHIP_REV_IS_EMUL(sc))
		cnt *= 20;

	ECORE_MSG(sc, "waiting for state to become %d", state);

	ECORE_MIGHT_SLEEP();
	while (cnt--) {
		bnx2x_intr_legacy(sc);
		if (!ECORE_TEST_BIT(state, pstate)) {
#ifdef ECORE_STOP_ON_ERROR
			ECORE_MSG(sc, "exit  (cnt %d)", 5000 - cnt);
#endif
			rte_atomic32_set(&sc->scan_fp, 0);
			return ECORE_SUCCESS;
		}

		ECORE_WAIT(sc, delay_us);

		if (sc->panic) {
			rte_atomic32_set(&sc->scan_fp, 0);
			return ECORE_IO;
		}
	}

	/* timeout! */
	PMD_DRV_LOG(ERR, sc, "timeout waiting for state %d", state);
	rte_atomic32_set(&sc->scan_fp, 0);
#ifdef ECORE_STOP_ON_ERROR
	ecore_panic();
#endif

	return ECORE_TIMEOUT;
}

static int ecore_raw_wait(struct bnx2x_softc *sc, struct ecore_raw_obj *raw)
{
	return ecore_state_wait(sc, raw->state, raw->pstate);
}

/***************** Classification verbs: Set/Del MAC/VLAN/VLAN-MAC ************/
/* credit handling callbacks */
static bool ecore_get_cam_offset_mac(struct ecore_vlan_mac_obj *o, int *offset)
{
	struct ecore_credit_pool_obj *mp = o->macs_pool;

	ECORE_DBG_BREAK_IF(!mp);

	return mp->get_entry(mp, offset);
}

static bool ecore_get_credit_mac(struct ecore_vlan_mac_obj *o)
{
	struct ecore_credit_pool_obj *mp = o->macs_pool;

	ECORE_DBG_BREAK_IF(!mp);

	return mp->get(mp, 1);
}

static bool ecore_put_cam_offset_mac(struct ecore_vlan_mac_obj *o, int offset)
{
	struct ecore_credit_pool_obj *mp = o->macs_pool;

	return mp->put_entry(mp, offset);
}

static bool ecore_put_credit_mac(struct ecore_vlan_mac_obj *o)
{
	struct ecore_credit_pool_obj *mp = o->macs_pool;

	return mp->put(mp, 1);
}

/**
 * __ecore_vlan_mac_h_write_trylock - try getting the writer lock on vlan mac
 * head list.
 *
 * @sc:		device handle
 * @o:		vlan_mac object
 *
 * @details: Non-blocking implementation; should be called under exelwtion
 *           queue lock.
 */
static int __ecore_vlan_mac_h_write_trylock(struct bnx2x_softc *sc __rte_unused,
					    struct ecore_vlan_mac_obj *o)
{
	if (o->head_reader) {
		ECORE_MSG(sc, "vlan_mac_lock writer - There are readers; Busy");
		return ECORE_BUSY;
	}

	ECORE_MSG(sc, "vlan_mac_lock writer - Taken");
	return ECORE_SUCCESS;
}

/**
 * __ecore_vlan_mac_h_exec_pending - execute step instead of a previous step
 * which wasn't able to run due to a taken lock on vlan mac head list.
 *
 * @sc:		device handle
 * @o:		vlan_mac object
 *
 * @details Should be called under exelwtion queue lock; notice it might release
 *          and reclaim it during its run.
 */
static void __ecore_vlan_mac_h_exec_pending(struct bnx2x_softc *sc,
					    struct ecore_vlan_mac_obj *o)
{
	int rc;
	uint32_t ramrod_flags = o->saved_ramrod_flags;

	ECORE_MSG(sc, "vlan_mac_lock execute pending command with ramrod flags %u",
		  ramrod_flags);
	o->head_exe_request = FALSE;
	o->saved_ramrod_flags = 0;
	rc = ecore_exe_queue_step(sc, &o->exe_queue, &ramrod_flags);
	if (rc != ECORE_SUCCESS) {
		PMD_DRV_LOG(ERR, sc,
			    "exelwtion of pending commands failed with rc %d",
			    rc);
#ifdef ECORE_STOP_ON_ERROR
		ecore_panic();
#endif
	}
}

/**
 * __ecore_vlan_mac_h_pend - Pend an exelwtion step which couldn't have been
 * called due to vlan mac head list lock being taken.
 *
 * @sc:			device handle
 * @o:			vlan_mac object
 * @ramrod_flags:	ramrod flags of missed exelwtion
 *
 * @details Should be called under exelwtion queue lock.
 */
static void __ecore_vlan_mac_h_pend(struct bnx2x_softc *sc __rte_unused,
				    struct ecore_vlan_mac_obj *o,
				    uint32_t ramrod_flags)
{
	o->head_exe_request = TRUE;
	o->saved_ramrod_flags = ramrod_flags;
	ECORE_MSG(sc, "Placing pending exelwtion with ramrod flags %u",
		  ramrod_flags);
}

/**
 * __ecore_vlan_mac_h_write_unlock - unlock the vlan mac head list writer lock
 *
 * @sc:			device handle
 * @o:			vlan_mac object
 *
 * @details Should be called under exelwtion queue lock. Notice if a pending
 *          exelwtion exists, it would perform it - possibly releasing and
 *          reclaiming the exelwtion queue lock.
 */
static void __ecore_vlan_mac_h_write_unlock(struct bnx2x_softc *sc,
					    struct ecore_vlan_mac_obj *o)
{
	/* It's possible a new pending exelwtion was added since this writer
	 * exelwted. If so, execute again. [Ad infinitum]
	 */
	while (o->head_exe_request) {
		ECORE_MSG(sc,
			  "vlan_mac_lock - writer release encountered a pending request");
		__ecore_vlan_mac_h_exec_pending(sc, o);
	}
}

/**
 * ecore_vlan_mac_h_write_unlock - unlock the vlan mac head list writer lock
 *
 * @sc:			device handle
 * @o:			vlan_mac object
 *
 * @details Notice if a pending exelwtion exists, it would perform it -
 *          possibly releasing and reclaiming the exelwtion queue lock.
 */
void ecore_vlan_mac_h_write_unlock(struct bnx2x_softc *sc,
				   struct ecore_vlan_mac_obj *o)
{
	ECORE_SPIN_LOCK_BH(&o->exe_queue.lock);
	__ecore_vlan_mac_h_write_unlock(sc, o);
	ECORE_SPIN_UNLOCK_BH(&o->exe_queue.lock);
}

/**
 * __ecore_vlan_mac_h_read_lock - lock the vlan mac head list reader lock
 *
 * @sc:			device handle
 * @o:			vlan_mac object
 *
 * @details Should be called under the exelwtion queue lock. May sleep. May
 *          release and reclaim exelwtion queue lock during its run.
 */
static int __ecore_vlan_mac_h_read_lock(struct bnx2x_softc *sc __rte_unused,
					struct ecore_vlan_mac_obj *o)
{
	/* If we got here, we're holding lock --> no WRITER exists */
	o->head_reader++;
	ECORE_MSG(sc,
		  "vlan_mac_lock - locked reader - number %d", o->head_reader);

	return ECORE_SUCCESS;
}

/**
 * ecore_vlan_mac_h_read_lock - lock the vlan mac head list reader lock
 *
 * @sc:			device handle
 * @o:			vlan_mac object
 *
 * @details May sleep. Claims and releases exelwtion queue lock during its run.
 */
int ecore_vlan_mac_h_read_lock(struct bnx2x_softc *sc,
				      struct ecore_vlan_mac_obj *o)
{
	int rc;

	ECORE_SPIN_LOCK_BH(&o->exe_queue.lock);
	rc = __ecore_vlan_mac_h_read_lock(sc, o);
	ECORE_SPIN_UNLOCK_BH(&o->exe_queue.lock);

	return rc;
}

/**
 * __ecore_vlan_mac_h_read_unlock - unlock the vlan mac head list reader lock
 *
 * @sc:			device handle
 * @o:			vlan_mac object
 *
 * @details Should be called under exelwtion queue lock. Notice if a pending
 *          exelwtion exists, it would be performed if this was the last
 *          reader. possibly releasing and reclaiming the exelwtion queue lock.
 */
static void __ecore_vlan_mac_h_read_unlock(struct bnx2x_softc *sc,
					   struct ecore_vlan_mac_obj *o)
{
	if (!o->head_reader) {
		PMD_DRV_LOG(ERR, sc,
			    "Need to release vlan mac reader lock, but lock isn't taken");
#ifdef ECORE_STOP_ON_ERROR
		ecore_panic();
#endif
	} else {
		o->head_reader--;
		ECORE_MSG(sc, "vlan_mac_lock - decreased readers to %d",
			  o->head_reader);
	}

	/* It's possible a new pending exelwtion was added, and that this reader
	 * was last - if so we need to execute the command.
	 */
	if (!o->head_reader && o->head_exe_request) {
		ECORE_MSG(sc, "vlan_mac_lock - reader release encountered a pending request");

		/* Writer release will do the trick */
		__ecore_vlan_mac_h_write_unlock(sc, o);
	}
}

/**
 * ecore_vlan_mac_h_read_unlock - unlock the vlan mac head list reader lock
 *
 * @sc:			device handle
 * @o:			vlan_mac object
 *
 * @details Notice if a pending exelwtion exists, it would be performed if this
 *          was the last reader. Claims and releases the exelwtion queue lock
 *          during its run.
 */
void ecore_vlan_mac_h_read_unlock(struct bnx2x_softc *sc,
				  struct ecore_vlan_mac_obj *o)
{
	ECORE_SPIN_LOCK_BH(&o->exe_queue.lock);
	__ecore_vlan_mac_h_read_unlock(sc, o);
	ECORE_SPIN_UNLOCK_BH(&o->exe_queue.lock);
}

/**
 * ecore_vlan_mac_h_read_unlock - unlock the vlan mac head list reader lock
 *
 * @sc:			device handle
 * @o:			vlan_mac object
 * @n:			number of elements to get
 * @base:		base address for element placement
 * @stride:		stride between elements (in bytes)
 */
static int ecore_get_n_elements(struct bnx2x_softc *sc,
				struct ecore_vlan_mac_obj *o, int n,
				uint8_t * base, uint8_t stride, uint8_t size)
{
	struct ecore_vlan_mac_registry_elem *pos;
	uint8_t *next = base;
	int counter = 0, read_lock;

	ECORE_MSG(sc, "get_n_elements - taking vlan_mac_lock (reader)");
	read_lock = ecore_vlan_mac_h_read_lock(sc, o);
	if (read_lock != ECORE_SUCCESS)
		PMD_DRV_LOG(ERR, sc,
			    "get_n_elements failed to get vlan mac reader lock; Access without lock");

	/* traverse list */
	ECORE_LIST_FOR_EACH_ENTRY(pos, &o->head, link,
				  struct ecore_vlan_mac_registry_elem) {
		if (counter < n) {
			ECORE_MEMCPY(next, &pos->u, size);
			counter++;
			    ECORE_MSG
			    (sc, "copied element number %d to address %p element was:",
			     counter, next);
			next += stride + size;
		}
	}

	if (read_lock == ECORE_SUCCESS) {
		ECORE_MSG(sc, "get_n_elements - releasing vlan_mac_lock (reader)");
		ecore_vlan_mac_h_read_unlock(sc, o);
	}

	return counter * ETH_ALEN;
}

/* check_add() callbacks */
static int ecore_check_mac_add(struct bnx2x_softc *sc __rte_unused,
			       struct ecore_vlan_mac_obj *o,
			       union ecore_classification_ramrod_data *data)
{
	struct ecore_vlan_mac_registry_elem *pos;

	ECORE_MSG(sc, "Checking MAC %02x:%02x:%02x:%02x:%02x:%02x for ADD command",
		  data->mac.mac[0], data->mac.mac[1], data->mac.mac[2],
		  data->mac.mac[3], data->mac.mac[4], data->mac.mac[5]);

	if (!ECORE_IS_VALID_ETHER_ADDR(data->mac.mac))
		return ECORE_ILWAL;

	/* Check if a requested MAC already exists */
	ECORE_LIST_FOR_EACH_ENTRY(pos, &o->head, link,
				  struct ecore_vlan_mac_registry_elem)
	    if (!ECORE_MEMCMP(data->mac.mac, pos->u.mac.mac, ETH_ALEN) &&
		(data->mac.is_inner_mac == pos->u.mac.is_inner_mac))
		return ECORE_EXISTS;

	return ECORE_SUCCESS;
}

/* check_del() callbacks */
static struct ecore_vlan_mac_registry_elem *ecore_check_mac_del(struct bnx2x_softc
								*sc
								__rte_unused,
								struct
								ecore_vlan_mac_obj
								*o, union
								ecore_classification_ramrod_data
								*data)
{
	struct ecore_vlan_mac_registry_elem *pos;

	ECORE_MSG(sc, "Checking MAC %02x:%02x:%02x:%02x:%02x:%02x for DEL command",
		  data->mac.mac[0], data->mac.mac[1], data->mac.mac[2],
		  data->mac.mac[3], data->mac.mac[4], data->mac.mac[5]);

	ECORE_LIST_FOR_EACH_ENTRY(pos, &o->head, link,
				  struct ecore_vlan_mac_registry_elem)
	if ((!ECORE_MEMCMP(data->mac.mac, pos->u.mac.mac, ETH_ALEN)) &&
	    (data->mac.is_inner_mac == pos->u.mac.is_inner_mac))
		return pos;

	return NULL;
}

/* check_move() callback */
static bool ecore_check_move(struct bnx2x_softc *sc,
			    struct ecore_vlan_mac_obj *src_o,
			    struct ecore_vlan_mac_obj *dst_o,
			    union ecore_classification_ramrod_data *data)
{
	struct ecore_vlan_mac_registry_elem *pos;
	int rc;

	/* Check if we can delete the requested configuration from the first
	 * object.
	 */
	pos = src_o->check_del(sc, src_o, data);

	/*  check if configuration can be added */
	rc = dst_o->check_add(sc, dst_o, data);

	/* If this classification can not be added (is already set)
	 * or can't be deleted - return an error.
	 */
	if (rc || !pos)
		return FALSE;

	return TRUE;
}

static bool ecore_check_move_always_err(__rte_unused struct bnx2x_softc *sc,
				       __rte_unused struct ecore_vlan_mac_obj
				       *src_o, __rte_unused struct ecore_vlan_mac_obj
				       *dst_o, __rte_unused union
				       ecore_classification_ramrod_data *data)
{
	return FALSE;
}

static uint8_t ecore_vlan_mac_get_rx_tx_flag(struct ecore_vlan_mac_obj
					     *o)
{
	struct ecore_raw_obj *raw = &o->raw;
	uint8_t rx_tx_flag = 0;

	if ((raw->obj_type == ECORE_OBJ_TYPE_TX) ||
	    (raw->obj_type == ECORE_OBJ_TYPE_RX_TX))
		rx_tx_flag |= ETH_CLASSIFY_CMD_HEADER_TX_CMD;

	if ((raw->obj_type == ECORE_OBJ_TYPE_RX) ||
	    (raw->obj_type == ECORE_OBJ_TYPE_RX_TX))
		rx_tx_flag |= ETH_CLASSIFY_CMD_HEADER_RX_CMD;

	return rx_tx_flag;
}

void ecore_set_mac_in_nig(struct bnx2x_softc *sc,
				 bool add, unsigned char *dev_addr, int index)
{
	uint32_t wb_data[2];
	uint32_t reg_offset = ECORE_PORT_ID(sc) ? NIG_REG_LLH1_FUNC_MEM :
	    NIG_REG_LLH0_FUNC_MEM;

	if (!ECORE_IS_MF_SI_MODE(sc) && !IS_MF_AFEX(sc))
		return;

	if (index > ECORE_LLH_CAM_MAX_PF_LINE)
		return;

	ECORE_MSG(sc, "Going to %s LLH configuration at entry %d",
		  (add ? "ADD" : "DELETE"), index);

	if (add) {
		/* LLH_FUNC_MEM is a uint64_t WB register */
		reg_offset += 8 * index;

		wb_data[0] = ((dev_addr[2] << 24) | (dev_addr[3] << 16) |
			      (dev_addr[4] << 8) | dev_addr[5]);
		wb_data[1] = ((dev_addr[0] << 8) | dev_addr[1]);

		ECORE_REG_WR_DMAE_LEN(sc, reg_offset, wb_data, 2);
	}

	REG_WR(sc, (ECORE_PORT_ID(sc) ? NIG_REG_LLH1_FUNC_MEM_ENABLE :
		    NIG_REG_LLH0_FUNC_MEM_ENABLE) + 4 * index, add);
}

/**
 * ecore_vlan_mac_set_cmd_hdr_e2 - set a header in a single classify ramrod
 *
 * @sc:		device handle
 * @o:		queue for which we want to configure this rule
 * @add:	if TRUE the command is an ADD command, DEL otherwise
 * @opcode:	CLASSIFY_RULE_OPCODE_XXX
 * @hdr:	pointer to a header to setup
 *
 */
static void ecore_vlan_mac_set_cmd_hdr_e2(struct ecore_vlan_mac_obj *o,
					  bool add, int opcode,
					  struct eth_classify_cmd_header
					  *hdr)
{
	struct ecore_raw_obj *raw = &o->raw;

	hdr->client_id = raw->cl_id;
	hdr->func_id = raw->func_id;

	/* Rx or/and Tx (internal switching) configuration ? */
	hdr->cmd_general_data |= ecore_vlan_mac_get_rx_tx_flag(o);

	if (add)
		hdr->cmd_general_data |= ETH_CLASSIFY_CMD_HEADER_IS_ADD;

	hdr->cmd_general_data |=
	    (opcode << ETH_CLASSIFY_CMD_HEADER_OPCODE_SHIFT);
}

/**
 * ecore_vlan_mac_set_rdata_hdr_e2 - set the classify ramrod data header
 *
 * @cid:	connection id
 * @type:	ECORE_FILTER_XXX_PENDING
 * @hdr:	pointer to header to setup
 * @rule_cnt:
 *
 * lwrrently we always configure one rule and echo field to contain a CID and an
 * opcode type.
 */
static void ecore_vlan_mac_set_rdata_hdr_e2(uint32_t cid, int type, struct eth_classify_header
					    *hdr, int rule_cnt)
{
	hdr->echo = ECORE_CPU_TO_LE32((cid & ECORE_SWCID_MASK) |
				      (type << ECORE_SWCID_SHIFT));
	hdr->rule_cnt = (uint8_t) rule_cnt;
}

/* hw_config() callbacks */
static void ecore_set_one_mac_e2(struct bnx2x_softc *sc,
				 struct ecore_vlan_mac_obj *o,
				 struct ecore_exeq_elem *elem, int rule_idx,
				 __rte_unused int cam_offset)
{
	struct ecore_raw_obj *raw = &o->raw;
	struct eth_classify_rules_ramrod_data *data =
	    (struct eth_classify_rules_ramrod_data *)(raw->rdata);
	int rule_cnt = rule_idx + 1, cmd = elem->cmd_data.vlan_mac.cmd;
	union eth_classify_rule_cmd *rule_entry = &data->rules[rule_idx];
	bool add = (cmd == ECORE_VLAN_MAC_ADD) ? TRUE : FALSE;
	uint32_t *vlan_mac_flags = &elem->cmd_data.vlan_mac.vlan_mac_flags;
	uint8_t *mac = elem->cmd_data.vlan_mac.u.mac.mac;

	/* Set LLH CAM entry: lwrrently only iSCSI and ETH macs are
	 * relevant. In addition, current implementation is tuned for a
	 * single ETH MAC.
	 *
	 * When multiple unicast ETH MACs PF configuration in switch
	 * independent mode is required (NetQ, multiple netdev MACs,
	 * etc.), consider better utilisation of 8 per function MAC
	 * entries in the LLH register. There is also
	 * NIG_REG_P[01]_LLH_FUNC_MEM2 registers that complete the
	 * total number of CAM entries to 16.
	 *
	 * Lwrrently we won't configure NIG for MACs other than a primary ETH
	 * MAC and iSCSI L2 MAC.
	 *
	 * If this MAC is moving from one Queue to another, no need to change
	 * NIG configuration.
	 */
	if (cmd != ECORE_VLAN_MAC_MOVE) {
		if (ECORE_TEST_BIT(ECORE_ISCSI_ETH_MAC, vlan_mac_flags))
			ecore_set_mac_in_nig(sc, add, mac,
					     ECORE_LLH_CAM_ISCSI_ETH_LINE);
		else if (ECORE_TEST_BIT(ECORE_ETH_MAC, vlan_mac_flags))
			ecore_set_mac_in_nig(sc, add, mac,
					     ECORE_LLH_CAM_ETH_LINE);
	}

	/* Reset the ramrod data buffer for the first rule */
	if (rule_idx == 0)
		ECORE_MEMSET(data, 0, sizeof(*data));

	/* Setup a command header */
	ecore_vlan_mac_set_cmd_hdr_e2(o, add, CLASSIFY_RULE_OPCODE_MAC,
				      &rule_entry->mac.header);

	ECORE_MSG(sc, "About to %s MAC %02x:%02x:%02x:%02x:%02x:%02x for Queue %d",
		  (add ? "add" : "delete"), mac[0], mac[1], mac[2], mac[3],
		  mac[4], mac[5], raw->cl_id);

	/* Set a MAC itself */
	ecore_set_fw_mac_addr(&rule_entry->mac.mac_msb,
			      &rule_entry->mac.mac_mid,
			      &rule_entry->mac.mac_lsb, mac);
	rule_entry->mac.inner_mac = elem->cmd_data.vlan_mac.u.mac.is_inner_mac;

	/* MOVE: Add a rule that will add this MAC to the target Queue */
	if (cmd == ECORE_VLAN_MAC_MOVE) {
		rule_entry++;
		rule_cnt++;

		/* Setup ramrod data */
		ecore_vlan_mac_set_cmd_hdr_e2(elem->cmd_data.
					      vlan_mac.target_obj, TRUE,
					      CLASSIFY_RULE_OPCODE_MAC,
					      &rule_entry->mac.header);

		/* Set a MAC itself */
		ecore_set_fw_mac_addr(&rule_entry->mac.mac_msb,
				      &rule_entry->mac.mac_mid,
				      &rule_entry->mac.mac_lsb, mac);
		rule_entry->mac.inner_mac =
		    elem->cmd_data.vlan_mac.u.mac.is_inner_mac;
	}

	/* Set the ramrod data header */
	ecore_vlan_mac_set_rdata_hdr_e2(raw->cid, raw->state, &data->header,
					rule_cnt);
}

/**
 * ecore_vlan_mac_set_rdata_hdr_e1x - set a header in a single classify ramrod
 *
 * @sc:		device handle
 * @o:		queue
 * @type:
 * @cam_offset:	offset in cam memory
 * @hdr:	pointer to a header to setup
 *
 * E1H
 */
static void ecore_vlan_mac_set_rdata_hdr_e1x(struct ecore_vlan_mac_obj
					     *o, int type, int cam_offset, struct mac_configuration_hdr
					     *hdr)
{
	struct ecore_raw_obj *r = &o->raw;

	hdr->length = 1;
	hdr->offset = (uint8_t) cam_offset;
	hdr->client_id = ECORE_CPU_TO_LE16(0xff);
	hdr->echo = ECORE_CPU_TO_LE32((r->cid & ECORE_SWCID_MASK) |
				      (type << ECORE_SWCID_SHIFT));
}

static void ecore_vlan_mac_set_cfg_entry_e1x(struct ecore_vlan_mac_obj
					     *o, int add, int opcode,
					     uint8_t * mac,
					     uint16_t vlan_id, struct
					     mac_configuration_entry
					     *cfg_entry)
{
	struct ecore_raw_obj *r = &o->raw;
	uint32_t cl_bit_vec = (1 << r->cl_id);

	cfg_entry->clients_bit_vector = ECORE_CPU_TO_LE32(cl_bit_vec);
	cfg_entry->pf_id = r->func_id;
	cfg_entry->vlan_id = ECORE_CPU_TO_LE16(vlan_id);

	if (add) {
		ECORE_SET_FLAG(cfg_entry->flags,
			       MAC_CONFIGURATION_ENTRY_ACTION_TYPE,
			       T_ETH_MAC_COMMAND_SET);
		ECORE_SET_FLAG(cfg_entry->flags,
			       MAC_CONFIGURATION_ENTRY_VLAN_FILTERING_MODE,
			       opcode);

		/* Set a MAC in a ramrod data */
		ecore_set_fw_mac_addr(&cfg_entry->msb_mac_addr,
				      &cfg_entry->middle_mac_addr,
				      &cfg_entry->lsb_mac_addr, mac);
	} else
		ECORE_SET_FLAG(cfg_entry->flags,
			       MAC_CONFIGURATION_ENTRY_ACTION_TYPE,
			       T_ETH_MAC_COMMAND_ILWALIDATE);
}

static void ecore_vlan_mac_set_rdata_e1x(struct bnx2x_softc *sc
					 __rte_unused,
					 struct ecore_vlan_mac_obj *o,
					 int type, int cam_offset,
					 int add, uint8_t * mac,
					 uint16_t vlan_id, int opcode,
					 struct mac_configuration_cmd
					 *config)
{
	struct mac_configuration_entry *cfg_entry = &config->config_table[0];

	ecore_vlan_mac_set_rdata_hdr_e1x(o, type, cam_offset, &config->hdr);
	ecore_vlan_mac_set_cfg_entry_e1x(o, add, opcode, mac, vlan_id,
					 cfg_entry);

	ECORE_MSG(sc, "%s MAC %02x:%02x:%02x:%02x:%02x:%02x CLID %d CAM offset %d",
		  (add ? "setting" : "clearing"),
		  mac[0], mac[1], mac[2], mac[3], mac[4], mac[5],
		  o->raw.cl_id, cam_offset);
}

/**
 * ecore_set_one_mac_e1x - fill a single MAC rule ramrod data
 *
 * @sc:		device handle
 * @o:		ecore_vlan_mac_obj
 * @elem:	ecore_exeq_elem
 * @rule_idx:	rule_idx
 * @cam_offset: cam_offset
 */
static void ecore_set_one_mac_e1x(struct bnx2x_softc *sc,
				  struct ecore_vlan_mac_obj *o,
				  struct ecore_exeq_elem *elem,
				  __rte_unused int rule_idx, int cam_offset)
{
	struct ecore_raw_obj *raw = &o->raw;
	struct mac_configuration_cmd *config =
	    (struct mac_configuration_cmd *)(raw->rdata);
	/* 57711 do not support MOVE command,
	 * so it's either ADD or DEL
	 */
	int add = (elem->cmd_data.vlan_mac.cmd == ECORE_VLAN_MAC_ADD) ?
	    TRUE : FALSE;

	/* Reset the ramrod data buffer */
	ECORE_MEMSET(config, 0, sizeof(*config));

	ecore_vlan_mac_set_rdata_e1x(sc, o, raw->state,
				     cam_offset, add,
				     elem->cmd_data.vlan_mac.u.mac.mac, 0,
				     ETH_VLAN_FILTER_ANY_VLAN, config);
}

/**
 * ecore_vlan_mac_restore - reconfigure next MAC/VLAN/VLAN-MAC element
 *
 * @sc:		device handle
 * @p:		command parameters
 * @ppos:	pointer to the cookie
 *
 * reconfigure next MAC/VLAN/VLAN-MAC element from the
 * previously configured elements list.
 *
 * from command parameters only RAMROD_COMP_WAIT bit in ramrod_flags is	taken
 * into an account
 *
 * pointer to the cookie  - that should be given back in the next call to make
 * function handle the next element. If *ppos is set to NULL it will restart the
 * iterator. If returned *ppos == NULL this means that the last element has been
 * handled.
 *
 */
static int ecore_vlan_mac_restore(struct bnx2x_softc *sc,
				  struct ecore_vlan_mac_ramrod_params *p,
				  struct ecore_vlan_mac_registry_elem **ppos)
{
	struct ecore_vlan_mac_registry_elem *pos;
	struct ecore_vlan_mac_obj *o = p->vlan_mac_obj;

	/* If list is empty - there is nothing to do here */
	if (ECORE_LIST_IS_EMPTY(&o->head)) {
		*ppos = NULL;
		return 0;
	}

	/* make a step... */
	if (*ppos == NULL)
		*ppos = ECORE_LIST_FIRST_ENTRY(&o->head, struct
					       ecore_vlan_mac_registry_elem,
					       link);
	else
		*ppos = ECORE_LIST_NEXT(*ppos, link,
					struct ecore_vlan_mac_registry_elem);

	pos = *ppos;

	/* If it's the last step - return NULL */
	if (ECORE_LIST_IS_LAST(&pos->link, &o->head))
		*ppos = NULL;

	/* Prepare a 'user_req' */
	ECORE_MEMCPY(&p->user_req.u, &pos->u, sizeof(pos->u));

	/* Set the command */
	p->user_req.cmd = ECORE_VLAN_MAC_ADD;

	/* Set vlan_mac_flags */
	p->user_req.vlan_mac_flags = pos->vlan_mac_flags;

	/* Set a restore bit */
	ECORE_SET_BIT_NA(RAMROD_RESTORE, &p->ramrod_flags);

	return ecore_config_vlan_mac(sc, p);
}

/* ecore_exeq_get_mac/ecore_exeq_get_vlan/ecore_exeq_get_vlan_mac return a
 * pointer to an element with a specific criteria and NULL if such an element
 * hasn't been found.
 */
static struct ecore_exeq_elem *ecore_exeq_get_mac(struct ecore_exe_queue_obj *o,
						  struct ecore_exeq_elem *elem)
{
	struct ecore_exeq_elem *pos;
	struct ecore_mac_ramrod_data *data = &elem->cmd_data.vlan_mac.u.mac;

	/* Check pending for exelwtion commands */
	ECORE_LIST_FOR_EACH_ENTRY(pos, &o->exe_queue, link,
				  struct ecore_exeq_elem)
	if (!ECORE_MEMCMP(&pos->cmd_data.vlan_mac.u.mac, data,
			  sizeof(*data)) &&
	    (pos->cmd_data.vlan_mac.cmd == elem->cmd_data.vlan_mac.cmd))
		return pos;

	return NULL;
}

/**
 * ecore_validate_vlan_mac_add - check if an ADD command can be exelwted
 *
 * @sc:		device handle
 * @qo:		ecore_qable_obj
 * @elem:	ecore_exeq_elem
 *
 * Checks that the requested configuration can be added. If yes and if
 * requested, consume CAM credit.
 *
 * The 'validate' is run after the 'optimize'.
 *
 */
static int ecore_validate_vlan_mac_add(struct bnx2x_softc *sc,
				       union ecore_qable_obj *qo,
				       struct ecore_exeq_elem *elem)
{
	struct ecore_vlan_mac_obj *o = &qo->vlan_mac;
	struct ecore_exe_queue_obj *exeq = &o->exe_queue;
	int rc;

	/* Check the registry */
	rc = o->check_add(sc, o, &elem->cmd_data.vlan_mac.u);
	if (rc) {
		ECORE_MSG(sc,
			  "ADD command is not allowed considering current registry state.");
		return rc;
	}

	/* Check if there is a pending ADD command for this
	 * MAC/VLAN/VLAN-MAC. Return an error if there is.
	 */
	if (exeq->get(exeq, elem)) {
		ECORE_MSG(sc, "There is a pending ADD command already");
		return ECORE_EXISTS;
	}

	/* Consume the credit if not requested not to */
	if (!(ECORE_TEST_BIT(ECORE_DONT_CONSUME_CAM_CREDIT,
			     &elem->cmd_data.vlan_mac.vlan_mac_flags) ||
	      o->get_credit(o)))
		return ECORE_ILWAL;

	return ECORE_SUCCESS;
}

/**
 * ecore_validate_vlan_mac_del - check if the DEL command can be exelwted
 *
 * @sc:		device handle
 * @qo:		quable object to check
 * @elem:	element that needs to be deleted
 *
 * Checks that the requested configuration can be deleted. If yes and if
 * requested, returns a CAM credit.
 *
 * The 'validate' is run after the 'optimize'.
 */
static int ecore_validate_vlan_mac_del(struct bnx2x_softc *sc,
				       union ecore_qable_obj *qo,
				       struct ecore_exeq_elem *elem)
{
	struct ecore_vlan_mac_obj *o = &qo->vlan_mac;
	struct ecore_vlan_mac_registry_elem *pos;
	struct ecore_exe_queue_obj *exeq = &o->exe_queue;
	struct ecore_exeq_elem query_elem;

	/* If this classification can not be deleted (doesn't exist)
	 * - return a ECORE_EXIST.
	 */
	pos = o->check_del(sc, o, &elem->cmd_data.vlan_mac.u);
	if (!pos) {
		ECORE_MSG(sc,
			  "DEL command is not allowed considering current registry state");
		return ECORE_EXISTS;
	}

	/* Check if there are pending DEL or MOVE commands for this
	 * MAC/VLAN/VLAN-MAC. Return an error if so.
	 */
	ECORE_MEMCPY(&query_elem, elem, sizeof(query_elem));

	/* Check for MOVE commands */
	query_elem.cmd_data.vlan_mac.cmd = ECORE_VLAN_MAC_MOVE;
	if (exeq->get(exeq, &query_elem)) {
		PMD_DRV_LOG(ERR, sc, "There is a pending MOVE command already");
		return ECORE_ILWAL;
	}

	/* Check for DEL commands */
	if (exeq->get(exeq, elem)) {
		ECORE_MSG(sc, "There is a pending DEL command already");
		return ECORE_EXISTS;
	}

	/* Return the credit to the credit pool if not requested not to */
	if (!(ECORE_TEST_BIT(ECORE_DONT_CONSUME_CAM_CREDIT,
			     &elem->cmd_data.vlan_mac.vlan_mac_flags) ||
	      o->put_credit(o))) {
		PMD_DRV_LOG(ERR, sc, "Failed to return a credit");
		return ECORE_ILWAL;
	}

	return ECORE_SUCCESS;
}

/**
 * ecore_validate_vlan_mac_move - check if the MOVE command can be exelwted
 *
 * @sc:		device handle
 * @qo:		quable object to check (source)
 * @elem:	element that needs to be moved
 *
 * Checks that the requested configuration can be moved. If yes and if
 * requested, returns a CAM credit.
 *
 * The 'validate' is run after the 'optimize'.
 */
static int ecore_validate_vlan_mac_move(struct bnx2x_softc *sc,
					union ecore_qable_obj *qo,
					struct ecore_exeq_elem *elem)
{
	struct ecore_vlan_mac_obj *src_o = &qo->vlan_mac;
	struct ecore_vlan_mac_obj *dest_o = elem->cmd_data.vlan_mac.target_obj;
	struct ecore_exeq_elem query_elem;
	struct ecore_exe_queue_obj *src_exeq = &src_o->exe_queue;
	struct ecore_exe_queue_obj *dest_exeq = &dest_o->exe_queue;

	/* Check if we can perform this operation based on the current registry
	 * state.
	 */
	if (!src_o->check_move(sc, src_o, dest_o, &elem->cmd_data.vlan_mac.u)) {
		ECORE_MSG(sc,
			  "MOVE command is not allowed considering current registry state");
		return ECORE_ILWAL;
	}

	/* Check if there is an already pending DEL or MOVE command for the
	 * source object or ADD command for a destination object. Return an
	 * error if so.
	 */
	ECORE_MEMCPY(&query_elem, elem, sizeof(query_elem));

	/* Check DEL on source */
	query_elem.cmd_data.vlan_mac.cmd = ECORE_VLAN_MAC_DEL;
	if (src_exeq->get(src_exeq, &query_elem)) {
		PMD_DRV_LOG(ERR, sc,
			    "There is a pending DEL command on the source queue already");
		return ECORE_ILWAL;
	}

	/* Check MOVE on source */
	if (src_exeq->get(src_exeq, elem)) {
		ECORE_MSG(sc, "There is a pending MOVE command already");
		return ECORE_EXISTS;
	}

	/* Check ADD on destination */
	query_elem.cmd_data.vlan_mac.cmd = ECORE_VLAN_MAC_ADD;
	if (dest_exeq->get(dest_exeq, &query_elem)) {
		PMD_DRV_LOG(ERR, sc,
			    "There is a pending ADD command on the destination queue already");
		return ECORE_ILWAL;
	}

	/* Consume the credit if not requested not to */
	if (!(ECORE_TEST_BIT(ECORE_DONT_CONSUME_CAM_CREDIT_DEST,
			     &elem->cmd_data.vlan_mac.vlan_mac_flags) ||
	      dest_o->get_credit(dest_o)))
		return ECORE_ILWAL;

	if (!(ECORE_TEST_BIT(ECORE_DONT_CONSUME_CAM_CREDIT,
			     &elem->cmd_data.vlan_mac.vlan_mac_flags) ||
	      src_o->put_credit(src_o))) {
		/* return the credit taken from dest... */
		dest_o->put_credit(dest_o);
		return ECORE_ILWAL;
	}

	return ECORE_SUCCESS;
}

static int ecore_validate_vlan_mac(struct bnx2x_softc *sc,
				   union ecore_qable_obj *qo,
				   struct ecore_exeq_elem *elem)
{
	switch (elem->cmd_data.vlan_mac.cmd) {
	case ECORE_VLAN_MAC_ADD:
		return ecore_validate_vlan_mac_add(sc, qo, elem);
	case ECORE_VLAN_MAC_DEL:
		return ecore_validate_vlan_mac_del(sc, qo, elem);
	case ECORE_VLAN_MAC_MOVE:
		return ecore_validate_vlan_mac_move(sc, qo, elem);
	default:
		return ECORE_ILWAL;
	}
}

static int ecore_remove_vlan_mac(__rte_unused struct bnx2x_softc *sc,
				 union ecore_qable_obj *qo,
				 struct ecore_exeq_elem *elem)
{
	int rc = 0;

	/* If consumption wasn't required, nothing to do */
	if (ECORE_TEST_BIT(ECORE_DONT_CONSUME_CAM_CREDIT,
			   &elem->cmd_data.vlan_mac.vlan_mac_flags))
		return ECORE_SUCCESS;

	switch (elem->cmd_data.vlan_mac.cmd) {
	case ECORE_VLAN_MAC_ADD:
	case ECORE_VLAN_MAC_MOVE:
		rc = qo->vlan_mac.put_credit(&qo->vlan_mac);
		break;
	case ECORE_VLAN_MAC_DEL:
		rc = qo->vlan_mac.get_credit(&qo->vlan_mac);
		break;
	default:
		return ECORE_ILWAL;
	}

	if (rc != TRUE)
		return ECORE_ILWAL;

	return ECORE_SUCCESS;
}

/**
 * ecore_wait_vlan_mac - passively wait for 5 seconds until all work completes.
 *
 * @sc:		device handle
 * @o:		ecore_vlan_mac_obj
 *
 */
static int ecore_wait_vlan_mac(struct bnx2x_softc *sc,
			       struct ecore_vlan_mac_obj *o)
{
	int cnt = 5000, rc;
	struct ecore_exe_queue_obj *exeq = &o->exe_queue;
	struct ecore_raw_obj *raw = &o->raw;

	while (cnt--) {
		/* Wait for the current command to complete */
		rc = raw->wait_comp(sc, raw);
		if (rc)
			return rc;

		/* Wait until there are no pending commands */
		if (!ecore_exe_queue_empty(exeq))
			ECORE_WAIT(sc, 1000);
		else
			return ECORE_SUCCESS;
	}

	return ECORE_TIMEOUT;
}

static int __ecore_vlan_mac_exelwte_step(struct bnx2x_softc *sc,
					 struct ecore_vlan_mac_obj *o,
					 uint32_t *ramrod_flags)
{
	int rc = ECORE_SUCCESS;

	ECORE_SPIN_LOCK_BH(&o->exe_queue.lock);

	ECORE_MSG(sc, "vlan_mac_exelwte_step - trying to take writer lock");
	rc = __ecore_vlan_mac_h_write_trylock(sc, o);

	if (rc != ECORE_SUCCESS) {
		__ecore_vlan_mac_h_pend(sc, o, *ramrod_flags);

		/** Calling function should not diffrentiate between this case
		 *  and the case in which there is already a pending ramrod
		 */
		rc = ECORE_PENDING;
	} else {
		rc = ecore_exe_queue_step(sc, &o->exe_queue, ramrod_flags);
	}
	ECORE_SPIN_UNLOCK_BH(&o->exe_queue.lock);

	return rc;
}

/**
 * ecore_complete_vlan_mac - complete one VLAN-MAC ramrod
 *
 * @sc:		device handle
 * @o:		ecore_vlan_mac_obj
 * @cqe:
 * @cont:	if TRUE schedule next exelwtion chunk
 *
 */
static int ecore_complete_vlan_mac(struct bnx2x_softc *sc,
				   struct ecore_vlan_mac_obj *o,
				   union event_ring_elem *cqe,
				   uint32_t *ramrod_flags)
{
	struct ecore_raw_obj *r = &o->raw;
	int rc;

	/* Reset pending list */
	ecore_exe_queue_reset_pending(sc, &o->exe_queue);

	/* Clear pending */
	r->clear_pending(r);

	/* If ramrod failed this is most likely a SW bug */
	if (cqe->message.error)
		return ECORE_ILWAL;

	/* Run the next bulk of pending commands if requested */
	if (ECORE_TEST_BIT(RAMROD_CONT, ramrod_flags)) {
		rc = __ecore_vlan_mac_exelwte_step(sc, o, ramrod_flags);
		if (rc < 0)
			return rc;
	}

	/* If there is more work to do return PENDING */
	if (!ecore_exe_queue_empty(&o->exe_queue))
		return ECORE_PENDING;

	return ECORE_SUCCESS;
}

/**
 * ecore_optimize_vlan_mac - optimize ADD and DEL commands.
 *
 * @sc:		device handle
 * @o:		ecore_qable_obj
 * @elem:	ecore_exeq_elem
 */
static int ecore_optimize_vlan_mac(struct bnx2x_softc *sc,
				   union ecore_qable_obj *qo,
				   struct ecore_exeq_elem *elem)
{
	struct ecore_exeq_elem query, *pos;
	struct ecore_vlan_mac_obj *o = &qo->vlan_mac;
	struct ecore_exe_queue_obj *exeq = &o->exe_queue;

	ECORE_MEMCPY(&query, elem, sizeof(query));

	switch (elem->cmd_data.vlan_mac.cmd) {
	case ECORE_VLAN_MAC_ADD:
		query.cmd_data.vlan_mac.cmd = ECORE_VLAN_MAC_DEL;
		break;
	case ECORE_VLAN_MAC_DEL:
		query.cmd_data.vlan_mac.cmd = ECORE_VLAN_MAC_ADD;
		break;
	default:
		/* Don't handle anything other than ADD or DEL */
		return 0;
	}

	/* If we found the appropriate element - delete it */
	pos = exeq->get(exeq, &query);
	if (pos) {

		/* Return the credit of the optimized command */
		if (!ECORE_TEST_BIT(ECORE_DONT_CONSUME_CAM_CREDIT,
				    &pos->cmd_data.vlan_mac.vlan_mac_flags)) {
			if ((query.cmd_data.vlan_mac.cmd ==
			     ECORE_VLAN_MAC_ADD) && !o->put_credit(o)) {
				PMD_DRV_LOG(ERR, sc,
					    "Failed to return the credit for the optimized ADD command");
				return ECORE_ILWAL;
			} else if (!o->get_credit(o)) {	/* VLAN_MAC_DEL */
				PMD_DRV_LOG(ERR, sc,
					    "Failed to recover the credit from the optimized DEL command");
				return ECORE_ILWAL;
			}
		}

		ECORE_MSG(sc, "Optimizing %s command",
			  (elem->cmd_data.vlan_mac.cmd == ECORE_VLAN_MAC_ADD) ?
			  "ADD" : "DEL");

		ECORE_LIST_REMOVE_ENTRY(&pos->link, &exeq->exe_queue);
		ecore_exe_queue_free_elem(sc, pos);
		return 1;
	}

	return 0;
}

/**
 * ecore_vlan_mac_get_registry_elem - prepare a registry element
 *
 * @sc:	  device handle
 * @o:
 * @elem:
 * @restore:
 * @re:
 *
 * prepare a registry element according to the current command request.
 */
static int ecore_vlan_mac_get_registry_elem(struct bnx2x_softc *sc,
					    struct ecore_vlan_mac_obj *o,
					    struct ecore_exeq_elem *elem,
					    int restore, struct
					    ecore_vlan_mac_registry_elem
					    **re)
{
	enum ecore_vlan_mac_cmd cmd = elem->cmd_data.vlan_mac.cmd;
	struct ecore_vlan_mac_registry_elem *reg_elem;

	/* Allocate a new registry element if needed. */
	if (!restore &&
	    ((cmd == ECORE_VLAN_MAC_ADD) || (cmd == ECORE_VLAN_MAC_MOVE))) {
		reg_elem = ECORE_ZALLOC(sizeof(*reg_elem), GFP_ATOMIC, sc);
		if (!reg_elem)
			return ECORE_NOMEM;

		/* Get a new CAM offset */
		if (!o->get_cam_offset(o, &reg_elem->cam_offset)) {
			/* This shall never happen, because we have checked the
			 * CAM availability in the 'validate'.
			 */
			ECORE_DBG_BREAK_IF(1);
			ECORE_FREE(sc, reg_elem, sizeof(*reg_elem));
			return ECORE_ILWAL;
		}

		ECORE_MSG(sc, "Got cam offset %d", reg_elem->cam_offset);

		/* Set a VLAN-MAC data */
		ECORE_MEMCPY(&reg_elem->u, &elem->cmd_data.vlan_mac.u,
			     sizeof(reg_elem->u));

		/* Copy the flags (needed for DEL and RESTORE flows) */
		reg_elem->vlan_mac_flags =
		    elem->cmd_data.vlan_mac.vlan_mac_flags;
	} else			/* DEL, RESTORE */
		reg_elem = o->check_del(sc, o, &elem->cmd_data.vlan_mac.u);

	*re = reg_elem;
	return ECORE_SUCCESS;
}

/**
 * ecore_exelwte_vlan_mac - execute vlan mac command
 *
 * @sc:			device handle
 * @qo:
 * @exe_chunk:
 * @ramrod_flags:
 *
 * go and send a ramrod!
 */
static int ecore_exelwte_vlan_mac(struct bnx2x_softc *sc,
				  union ecore_qable_obj *qo,
				  ecore_list_t * exe_chunk,
				  uint32_t *ramrod_flags)
{
	struct ecore_exeq_elem *elem;
	struct ecore_vlan_mac_obj *o = &qo->vlan_mac, *cam_obj;
	struct ecore_raw_obj *r = &o->raw;
	int rc, idx = 0;
	int restore = ECORE_TEST_BIT(RAMROD_RESTORE, ramrod_flags);
	int drv_only = ECORE_TEST_BIT(RAMROD_DRV_CLR_ONLY, ramrod_flags);
	struct ecore_vlan_mac_registry_elem *reg_elem;
	enum ecore_vlan_mac_cmd cmd;

	/* If DRIVER_ONLY exelwtion is requested, cleanup a registry
	 * and exit. Otherwise send a ramrod to FW.
	 */
	if (!drv_only) {

		/* Set pending */
		r->set_pending(r);

		/* Fill the ramrod data */
		ECORE_LIST_FOR_EACH_ENTRY(elem, exe_chunk, link,
					  struct ecore_exeq_elem) {
			cmd = elem->cmd_data.vlan_mac.cmd;
			/* We will add to the target object in MOVE command, so
			 * change the object for a CAM search.
			 */
			if (cmd == ECORE_VLAN_MAC_MOVE)
				cam_obj = elem->cmd_data.vlan_mac.target_obj;
			else
				cam_obj = o;

			rc = ecore_vlan_mac_get_registry_elem(sc, cam_obj,
							      elem, restore,
							      &reg_elem);
			if (rc)
				goto error_exit;

			ECORE_DBG_BREAK_IF(!reg_elem);

			/* Push a new entry into the registry */
			if (!restore &&
			    ((cmd == ECORE_VLAN_MAC_ADD) ||
			     (cmd == ECORE_VLAN_MAC_MOVE)))
				ECORE_LIST_PUSH_HEAD(&reg_elem->link,
						     &cam_obj->head);

			/* Configure a single command in a ramrod data buffer */
			o->set_one_rule(sc, o, elem, idx, reg_elem->cam_offset);

			/* MOVE command consumes 2 entries in the ramrod data */
			if (cmd == ECORE_VLAN_MAC_MOVE)
				idx += 2;
			else
				idx++;
		}

		/*
		 *  No need for an explicit memory barrier here as long we would
		 *  need to ensure the ordering of writing to the SPQ element
		 *  and updating of the SPQ producer which ilwolves a memory
		 *  read and we will have to put a full memory barrier there
		 *  (inside ecore_sp_post()).
		 */

		rc = ecore_sp_post(sc, o->ramrod_cmd, r->cid,
				   r->rdata_mapping, ETH_CONNECTION_TYPE);
		if (rc)
			goto error_exit;
	}

	/* Now, when we are done with the ramrod - clean up the registry */
	ECORE_LIST_FOR_EACH_ENTRY(elem, exe_chunk, link, struct ecore_exeq_elem) {
		cmd = elem->cmd_data.vlan_mac.cmd;
		if ((cmd == ECORE_VLAN_MAC_DEL) || (cmd == ECORE_VLAN_MAC_MOVE)) {
			reg_elem = o->check_del(sc, o,
						&elem->cmd_data.vlan_mac.u);

			ECORE_DBG_BREAK_IF(!reg_elem);

			o->put_cam_offset(o, reg_elem->cam_offset);
			ECORE_LIST_REMOVE_ENTRY(&reg_elem->link, &o->head);
			ECORE_FREE(sc, reg_elem, sizeof(*reg_elem));
		}
	}

	if (!drv_only)
		return ECORE_PENDING;
	else
		return ECORE_SUCCESS;

error_exit:
	r->clear_pending(r);

	/* Cleanup a registry in case of a failure */
	ECORE_LIST_FOR_EACH_ENTRY(elem, exe_chunk, link, struct ecore_exeq_elem) {
		cmd = elem->cmd_data.vlan_mac.cmd;

		if (cmd == ECORE_VLAN_MAC_MOVE)
			cam_obj = elem->cmd_data.vlan_mac.target_obj;
		else
			cam_obj = o;

		/* Delete all newly added above entries */
		if (!restore &&
		    ((cmd == ECORE_VLAN_MAC_ADD) ||
		     (cmd == ECORE_VLAN_MAC_MOVE))) {
			reg_elem = o->check_del(sc, cam_obj,
						&elem->cmd_data.vlan_mac.u);
			if (reg_elem) {
				ECORE_LIST_REMOVE_ENTRY(&reg_elem->link,
							&cam_obj->head);
				ECORE_FREE(sc, reg_elem, sizeof(*reg_elem));
			}
		}
	}

	return rc;
}

static int ecore_vlan_mac_push_new_cmd(struct bnx2x_softc *sc, struct
				       ecore_vlan_mac_ramrod_params *p)
{
	struct ecore_exeq_elem *elem;
	struct ecore_vlan_mac_obj *o = p->vlan_mac_obj;
	int restore = ECORE_TEST_BIT(RAMROD_RESTORE, &p->ramrod_flags);

	/* Allocate the exelwtion queue element */
	elem = ecore_exe_queue_alloc_elem(sc);
	if (!elem)
		return ECORE_NOMEM;

	/* Set the command 'length' */
	switch (p->user_req.cmd) {
	case ECORE_VLAN_MAC_MOVE:
		elem->cmd_len = 2;
		break;
	default:
		elem->cmd_len = 1;
	}

	/* Fill the object specific info */
	ECORE_MEMCPY(&elem->cmd_data.vlan_mac, &p->user_req,
		     sizeof(p->user_req));

	/* Try to add a new command to the pending list */
	return ecore_exe_queue_add(sc, &o->exe_queue, elem, restore);
}

/**
 * ecore_config_vlan_mac - configure VLAN/MAC/VLAN_MAC filtering rules.
 *
 * @sc:	  device handle
 * @p:
 *
 */
int ecore_config_vlan_mac(struct bnx2x_softc *sc,
			  struct ecore_vlan_mac_ramrod_params *p)
{
	int rc = ECORE_SUCCESS;
	struct ecore_vlan_mac_obj *o = p->vlan_mac_obj;
	uint32_t *ramrod_flags = &p->ramrod_flags;
	int cont = ECORE_TEST_BIT(RAMROD_CONT, ramrod_flags);
	struct ecore_raw_obj *raw = &o->raw;

	/*
	 * Add new elements to the exelwtion list for commands that require it.
	 */
	if (!cont) {
		rc = ecore_vlan_mac_push_new_cmd(sc, p);
		if (rc)
			return rc;
	}

	/* If nothing will be exelwted further in this iteration we want to
	 * return PENDING if there are pending commands
	 */
	if (!ecore_exe_queue_empty(&o->exe_queue))
		rc = ECORE_PENDING;

	if (ECORE_TEST_BIT(RAMROD_DRV_CLR_ONLY, ramrod_flags)) {
		ECORE_MSG(sc,
			  "RAMROD_DRV_CLR_ONLY requested: clearing a pending bit.");
		raw->clear_pending(raw);
	}

	/* Execute commands if required */
	if (cont || ECORE_TEST_BIT(RAMROD_EXEC, ramrod_flags) ||
	    ECORE_TEST_BIT(RAMROD_COMP_WAIT, ramrod_flags)) {
		rc = __ecore_vlan_mac_exelwte_step(sc, p->vlan_mac_obj,
						   &p->ramrod_flags);
		if (rc < 0)
			return rc;
	}

	/* RAMROD_COMP_WAIT is a superset of RAMROD_EXEC. If it was set
	 * then user want to wait until the last command is done.
	 */
	if (ECORE_TEST_BIT(RAMROD_COMP_WAIT, &p->ramrod_flags)) {
		/* Wait maximum for the current exe_queue length iterations plus
		 * one (for the current pending command).
		 */
		int max_iterations = ecore_exe_queue_length(&o->exe_queue) + 1;

		while (!ecore_exe_queue_empty(&o->exe_queue) &&
		       max_iterations--) {

			/* Wait for the current command to complete */
			rc = raw->wait_comp(sc, raw);
			if (rc)
				return rc;

			/* Make a next step */
			rc = __ecore_vlan_mac_exelwte_step(sc,
							   p->vlan_mac_obj,
							   &p->ramrod_flags);
			if (rc < 0)
				return rc;
		}

		return ECORE_SUCCESS;
	}

	return rc;
}

/**
 * ecore_vlan_mac_del_all - delete elements with given vlan_mac_flags spec
 *
 * @sc:			device handle
 * @o:
 * @vlan_mac_flags:
 * @ramrod_flags:	exelwtion flags to be used for this deletion
 *
 * if the last operation has completed successfully and there are no
 * more elements left, positive value if the last operation has completed
 * successfully and there are more previously configured elements, negative
 * value is current operation has failed.
 */
static int ecore_vlan_mac_del_all(struct bnx2x_softc *sc,
				  struct ecore_vlan_mac_obj *o,
				  uint32_t *vlan_mac_flags,
				  uint32_t *ramrod_flags)
{
	struct ecore_vlan_mac_registry_elem *pos = NULL;
	int rc = 0, read_lock;
	struct ecore_vlan_mac_ramrod_params p;
	struct ecore_exe_queue_obj *exeq = &o->exe_queue;
	struct ecore_exeq_elem *exeq_pos, *exeq_pos_n;

	/* Clear pending commands first */

	ECORE_SPIN_LOCK_BH(&exeq->lock);

	ECORE_LIST_FOR_EACH_ENTRY_SAFE(exeq_pos, exeq_pos_n,
				       &exeq->exe_queue, link,
				       struct ecore_exeq_elem) {
		if (exeq_pos->cmd_data.vlan_mac.vlan_mac_flags ==
		    *vlan_mac_flags) {
			rc = exeq->remove(sc, exeq->owner, exeq_pos);
			if (rc) {
				PMD_DRV_LOG(ERR, sc, "Failed to remove command");
				ECORE_SPIN_UNLOCK_BH(&exeq->lock);
				return rc;
			}
			ECORE_LIST_REMOVE_ENTRY(&exeq_pos->link,
						&exeq->exe_queue);
			ecore_exe_queue_free_elem(sc, exeq_pos);
		}
	}

	ECORE_SPIN_UNLOCK_BH(&exeq->lock);

	/* Prepare a command request */
	ECORE_MEMSET(&p, 0, sizeof(p));
	p.vlan_mac_obj = o;
	p.ramrod_flags = *ramrod_flags;
	p.user_req.cmd = ECORE_VLAN_MAC_DEL;

	/* Add all but the last VLAN-MAC to the exelwtion queue without actually
	 * exelwtion anything.
	 */
	ECORE_CLEAR_BIT_NA(RAMROD_COMP_WAIT, &p.ramrod_flags);
	ECORE_CLEAR_BIT_NA(RAMROD_EXEC, &p.ramrod_flags);
	ECORE_CLEAR_BIT_NA(RAMROD_CONT, &p.ramrod_flags);

	ECORE_MSG(sc, "vlan_mac_del_all -- taking vlan_mac_lock (reader)");
	read_lock = ecore_vlan_mac_h_read_lock(sc, o);
	if (read_lock != ECORE_SUCCESS)
		return read_lock;

	ECORE_LIST_FOR_EACH_ENTRY(pos, &o->head, link,
				  struct ecore_vlan_mac_registry_elem) {
		if (pos->vlan_mac_flags == *vlan_mac_flags) {
			p.user_req.vlan_mac_flags = pos->vlan_mac_flags;
			ECORE_MEMCPY(&p.user_req.u, &pos->u, sizeof(pos->u));
			rc = ecore_config_vlan_mac(sc, &p);
			if (rc < 0) {
				PMD_DRV_LOG(ERR, sc,
					    "Failed to add a new DEL command");
				ecore_vlan_mac_h_read_unlock(sc, o);
				return rc;
			}
		}
	}

	ECORE_MSG(sc, "vlan_mac_del_all -- releasing vlan_mac_lock (reader)");
	ecore_vlan_mac_h_read_unlock(sc, o);

	p.ramrod_flags = *ramrod_flags;
	ECORE_SET_BIT_NA(RAMROD_CONT, &p.ramrod_flags);

	return ecore_config_vlan_mac(sc, &p);
}

static void ecore_init_raw_obj(struct ecore_raw_obj *raw, uint8_t cl_id,
			       uint32_t cid, uint8_t func_id,
			       void *rdata,
			       ecore_dma_addr_t rdata_mapping, int state,
			       uint32_t *pstate, ecore_obj_type type)
{
	raw->func_id = func_id;
	raw->cid = cid;
	raw->cl_id = cl_id;
	raw->rdata = rdata;
	raw->rdata_mapping = rdata_mapping;
	raw->state = state;
	raw->pstate = pstate;
	raw->obj_type = type;
	raw->check_pending = ecore_raw_check_pending;
	raw->clear_pending = ecore_raw_clear_pending;
	raw->set_pending = ecore_raw_set_pending;
	raw->wait_comp = ecore_raw_wait;
}

static void ecore_init_vlan_mac_common(struct ecore_vlan_mac_obj *o,
				       uint8_t cl_id, uint32_t cid,
				       uint8_t func_id, void *rdata,
				       ecore_dma_addr_t rdata_mapping,
				       int state, uint32_t *pstate,
				       ecore_obj_type type,
				       struct ecore_credit_pool_obj
				       *macs_pool, struct ecore_credit_pool_obj
				       *vlans_pool)
{
	ECORE_LIST_INIT(&o->head);
	o->head_reader = 0;
	o->head_exe_request = FALSE;
	o->saved_ramrod_flags = 0;

	o->macs_pool = macs_pool;
	o->vlans_pool = vlans_pool;

	o->delete_all = ecore_vlan_mac_del_all;
	o->restore = ecore_vlan_mac_restore;
	o->complete = ecore_complete_vlan_mac;
	o->wait = ecore_wait_vlan_mac;

	ecore_init_raw_obj(&o->raw, cl_id, cid, func_id, rdata, rdata_mapping,
			   state, pstate, type);
}

void ecore_init_mac_obj(struct bnx2x_softc *sc,
			struct ecore_vlan_mac_obj *mac_obj,
			uint8_t cl_id, uint32_t cid, uint8_t func_id,
			void *rdata, ecore_dma_addr_t rdata_mapping, int state,
			uint32_t *pstate, ecore_obj_type type,
			struct ecore_credit_pool_obj *macs_pool)
{
	union ecore_qable_obj *qable_obj = (union ecore_qable_obj *)mac_obj;

	ecore_init_vlan_mac_common(mac_obj, cl_id, cid, func_id, rdata,
				   rdata_mapping, state, pstate, type,
				   macs_pool, NULL);

	/* CAM credit pool handling */
	mac_obj->get_credit = ecore_get_credit_mac;
	mac_obj->put_credit = ecore_put_credit_mac;
	mac_obj->get_cam_offset = ecore_get_cam_offset_mac;
	mac_obj->put_cam_offset = ecore_put_cam_offset_mac;

	if (CHIP_IS_E1x(sc)) {
		mac_obj->set_one_rule = ecore_set_one_mac_e1x;
		mac_obj->check_del = ecore_check_mac_del;
		mac_obj->check_add = ecore_check_mac_add;
		mac_obj->check_move = ecore_check_move_always_err;
		mac_obj->ramrod_cmd = RAMROD_CMD_ID_ETH_SET_MAC;

		/* Exe Queue */
		ecore_exe_queue_init(sc,
				     &mac_obj->exe_queue, 1, qable_obj,
				     ecore_validate_vlan_mac,
				     ecore_remove_vlan_mac,
				     ecore_optimize_vlan_mac,
				     ecore_exelwte_vlan_mac,
				     ecore_exeq_get_mac);
	} else {
		mac_obj->set_one_rule = ecore_set_one_mac_e2;
		mac_obj->check_del = ecore_check_mac_del;
		mac_obj->check_add = ecore_check_mac_add;
		mac_obj->check_move = ecore_check_move;
		mac_obj->ramrod_cmd = RAMROD_CMD_ID_ETH_CLASSIFICATION_RULES;
		mac_obj->get_n_elements = ecore_get_n_elements;

		/* Exe Queue */
		ecore_exe_queue_init(sc,
				     &mac_obj->exe_queue, CLASSIFY_RULES_COUNT,
				     qable_obj, ecore_validate_vlan_mac,
				     ecore_remove_vlan_mac,
				     ecore_optimize_vlan_mac,
				     ecore_exelwte_vlan_mac,
				     ecore_exeq_get_mac);
	}
}

/* RX_MODE verbs: DROP_ALL/ACCEPT_ALL/ACCEPT_ALL_MULTI/ACCEPT_ALL_VLAN/NORMAL */
static void __storm_memset_mac_filters(struct bnx2x_softc *sc, struct
				       tstorm_eth_mac_filter_config
				       *mac_filters, uint16_t pf_id)
{
	size_t size = sizeof(struct tstorm_eth_mac_filter_config);

	uint32_t addr = BAR_TSTRORM_INTMEM +
	    TSTORM_MAC_FILTER_CONFIG_OFFSET(pf_id);

	ecore_storm_memset_struct(sc, addr, size, (uint32_t *) mac_filters);
}

static int ecore_set_rx_mode_e1x(struct bnx2x_softc *sc,
				 struct ecore_rx_mode_ramrod_params *p)
{
	/* update the sc MAC filter structure */
	uint32_t mask = (1 << p->cl_id);

	struct tstorm_eth_mac_filter_config *mac_filters =
	    (struct tstorm_eth_mac_filter_config *)p->rdata;

	/* initial setting is drop-all */
	uint8_t drop_all_ucast = 1, drop_all_mcast = 1;
	uint8_t accp_all_ucast = 0, accp_all_bcast = 0, accp_all_mcast = 0;
	uint8_t unmatched_unicast = 0;

	/* In e1x there we only take into account rx accept flag since tx switching
	 * isn't enabled. */
	if (ECORE_TEST_BIT(ECORE_ACCEPT_UNICAST, &p->rx_accept_flags))
		/* accept matched ucast */
		drop_all_ucast = 0;

	if (ECORE_TEST_BIT(ECORE_ACCEPT_MULTICAST, &p->rx_accept_flags))
		/* accept matched mcast */
		drop_all_mcast = 0;

	if (ECORE_TEST_BIT(ECORE_ACCEPT_ALL_UNICAST, &p->rx_accept_flags)) {
		/* accept all mcast */
		drop_all_ucast = 0;
		accp_all_ucast = 1;
	}
	if (ECORE_TEST_BIT(ECORE_ACCEPT_ALL_MULTICAST, &p->rx_accept_flags)) {
		/* accept all mcast */
		drop_all_mcast = 0;
		accp_all_mcast = 1;
	}
	if (ECORE_TEST_BIT(ECORE_ACCEPT_BROADCAST, &p->rx_accept_flags))
		/* accept (all) bcast */
		accp_all_bcast = 1;
	if (ECORE_TEST_BIT(ECORE_ACCEPT_UNMATCHED, &p->rx_accept_flags))
		/* accept unmatched unicasts */
		unmatched_unicast = 1;

	mac_filters->ucast_drop_all = drop_all_ucast ?
	    mac_filters->ucast_drop_all | mask :
	    mac_filters->ucast_drop_all & ~mask;

	mac_filters->mcast_drop_all = drop_all_mcast ?
	    mac_filters->mcast_drop_all | mask :
	    mac_filters->mcast_drop_all & ~mask;

	mac_filters->ucast_accept_all = accp_all_ucast ?
	    mac_filters->ucast_accept_all | mask :
	    mac_filters->ucast_accept_all & ~mask;

	mac_filters->mcast_accept_all = accp_all_mcast ?
	    mac_filters->mcast_accept_all | mask :
	    mac_filters->mcast_accept_all & ~mask;

	mac_filters->bcast_accept_all = accp_all_bcast ?
	    mac_filters->bcast_accept_all | mask :
	    mac_filters->bcast_accept_all & ~mask;

	mac_filters->unmatched_unicast = unmatched_unicast ?
	    mac_filters->unmatched_unicast | mask :
	    mac_filters->unmatched_unicast & ~mask;

	ECORE_MSG(sc, "drop_ucast 0x%xdrop_mcast 0x%x accp_ucast 0x%x"
		  "accp_mcast 0x%xaccp_bcast 0x%x",
		  mac_filters->ucast_drop_all, mac_filters->mcast_drop_all,
		  mac_filters->ucast_accept_all, mac_filters->mcast_accept_all,
		  mac_filters->bcast_accept_all);

	/* write the MAC filter structure */
	__storm_memset_mac_filters(sc, mac_filters, p->func_id);

	/* The operation is completed */
	ECORE_CLEAR_BIT(p->state, p->pstate);
	ECORE_SMP_MB_AFTER_CLEAR_BIT();

	return ECORE_SUCCESS;
}

/* Setup ramrod data */
static void ecore_rx_mode_set_rdata_hdr_e2(uint32_t cid, struct eth_classify_header
					   *hdr, uint8_t rule_cnt)
{
	hdr->echo = ECORE_CPU_TO_LE32(cid);
	hdr->rule_cnt = rule_cnt;
}

static void ecore_rx_mode_set_cmd_state_e2(uint32_t *accept_flags,
			struct eth_filter_rules_cmd *cmd, int clear_accept_all)
{
	uint16_t state;

	/* start with 'drop-all' */
	state = ETH_FILTER_RULES_CMD_UCAST_DROP_ALL |
	    ETH_FILTER_RULES_CMD_MCAST_DROP_ALL;

	if (ECORE_TEST_BIT(ECORE_ACCEPT_UNICAST, accept_flags))
		state &= ~ETH_FILTER_RULES_CMD_UCAST_DROP_ALL;

	if (ECORE_TEST_BIT(ECORE_ACCEPT_MULTICAST, accept_flags))
		state &= ~ETH_FILTER_RULES_CMD_MCAST_DROP_ALL;

	if (ECORE_TEST_BIT(ECORE_ACCEPT_ALL_UNICAST, accept_flags)) {
		state &= ~ETH_FILTER_RULES_CMD_UCAST_DROP_ALL;
		state |= ETH_FILTER_RULES_CMD_UCAST_ACCEPT_ALL;
	}

	if (ECORE_TEST_BIT(ECORE_ACCEPT_ALL_MULTICAST, accept_flags)) {
		state |= ETH_FILTER_RULES_CMD_MCAST_ACCEPT_ALL;
		state &= ~ETH_FILTER_RULES_CMD_MCAST_DROP_ALL;
	}
	if (ECORE_TEST_BIT(ECORE_ACCEPT_BROADCAST, accept_flags))
		state |= ETH_FILTER_RULES_CMD_BCAST_ACCEPT_ALL;

	if (ECORE_TEST_BIT(ECORE_ACCEPT_UNMATCHED, accept_flags)) {
		state &= ~ETH_FILTER_RULES_CMD_UCAST_DROP_ALL;
		state |= ETH_FILTER_RULES_CMD_UCAST_ACCEPT_UNMATCHED;
	}
	if (ECORE_TEST_BIT(ECORE_ACCEPT_ANY_VLAN, accept_flags))
		state |= ETH_FILTER_RULES_CMD_ACCEPT_ANY_VLAN;

	/* Clear ACCEPT_ALL_XXX flags for FCoE L2 Queue */
	if (clear_accept_all) {
		state &= ~ETH_FILTER_RULES_CMD_MCAST_ACCEPT_ALL;
		state &= ~ETH_FILTER_RULES_CMD_BCAST_ACCEPT_ALL;
		state &= ~ETH_FILTER_RULES_CMD_UCAST_ACCEPT_ALL;
		state &= ~ETH_FILTER_RULES_CMD_UCAST_ACCEPT_UNMATCHED;
	}

	cmd->state = ECORE_CPU_TO_LE16(state);
}

static int ecore_set_rx_mode_e2(struct bnx2x_softc *sc,
				struct ecore_rx_mode_ramrod_params *p)
{
	struct eth_filter_rules_ramrod_data *data = p->rdata;
	int rc;
	uint8_t rule_idx = 0;

	/* Reset the ramrod data buffer */
	ECORE_MEMSET(data, 0, sizeof(*data));

	/* Setup ramrod data */

	/* Tx (internal switching) */
	if (ECORE_TEST_BIT(RAMROD_TX, &p->ramrod_flags)) {
		data->rules[rule_idx].client_id = p->cl_id;
		data->rules[rule_idx].func_id = p->func_id;

		data->rules[rule_idx].cmd_general_data =
		    ETH_FILTER_RULES_CMD_TX_CMD;

		ecore_rx_mode_set_cmd_state_e2(&p->tx_accept_flags,
					       &(data->rules[rule_idx++]),
					       FALSE);
	}

	/* Rx */
	if (ECORE_TEST_BIT(RAMROD_RX, &p->ramrod_flags)) {
		data->rules[rule_idx].client_id = p->cl_id;
		data->rules[rule_idx].func_id = p->func_id;

		data->rules[rule_idx].cmd_general_data =
		    ETH_FILTER_RULES_CMD_RX_CMD;

		ecore_rx_mode_set_cmd_state_e2(&p->rx_accept_flags,
					       &(data->rules[rule_idx++]),
					       FALSE);
	}

	/* If FCoE Queue configuration has been requested configure the Rx and
	 * internal switching modes for this queue in separate rules.
	 *
	 * FCoE queue shell never be set to ACCEPT_ALL packets of any sort:
	 * MCAST_ALL, UCAST_ALL, BCAST_ALL and UNMATCHED.
	 */
	if (ECORE_TEST_BIT(ECORE_RX_MODE_FCOE_ETH, &p->rx_mode_flags)) {
		/*  Tx (internal switching) */
		if (ECORE_TEST_BIT(RAMROD_TX, &p->ramrod_flags)) {
			data->rules[rule_idx].client_id = ECORE_FCOE_CID(sc);
			data->rules[rule_idx].func_id = p->func_id;

			data->rules[rule_idx].cmd_general_data =
			    ETH_FILTER_RULES_CMD_TX_CMD;

			ecore_rx_mode_set_cmd_state_e2(&p->tx_accept_flags,
						       &(data->rules
							 [rule_idx++]), TRUE);
		}

		/* Rx */
		if (ECORE_TEST_BIT(RAMROD_RX, &p->ramrod_flags)) {
			data->rules[rule_idx].client_id = ECORE_FCOE_CID(sc);
			data->rules[rule_idx].func_id = p->func_id;

			data->rules[rule_idx].cmd_general_data =
			    ETH_FILTER_RULES_CMD_RX_CMD;

			ecore_rx_mode_set_cmd_state_e2(&p->rx_accept_flags,
						       &(data->rules
							 [rule_idx++]), TRUE);
		}
	}

	/* Set the ramrod header (most importantly - number of rules to
	 * configure).
	 */
	ecore_rx_mode_set_rdata_hdr_e2(p->cid, &data->header, rule_idx);

	    ECORE_MSG
	    (sc, "About to configure %d rules, rx_accept_flags 0x%x, tx_accept_flags 0x%x",
	     data->header.rule_cnt, p->rx_accept_flags, p->tx_accept_flags);

	/* No need for an explicit memory barrier here as long we would
	 * need to ensure the ordering of writing to the SPQ element
	 * and updating of the SPQ producer which ilwolves a memory
	 * read and we will have to put a full memory barrier there
	 * (inside ecore_sp_post()).
	 */

	/* Send a ramrod */
	rc = ecore_sp_post(sc,
			   RAMROD_CMD_ID_ETH_FILTER_RULES,
			   p->cid, p->rdata_mapping, ETH_CONNECTION_TYPE);
	if (rc)
		return rc;

	/* Ramrod completion is pending */
	return ECORE_PENDING;
}

static int ecore_wait_rx_mode_comp_e2(struct bnx2x_softc *sc,
				      struct ecore_rx_mode_ramrod_params *p)
{
	return ecore_state_wait(sc, p->state, p->pstate);
}

static int ecore_empty_rx_mode_wait(__rte_unused struct bnx2x_softc *sc,
				    __rte_unused struct
				    ecore_rx_mode_ramrod_params *p)
{
	/* Do nothing */
	return ECORE_SUCCESS;
}

int ecore_config_rx_mode(struct bnx2x_softc *sc,
			 struct ecore_rx_mode_ramrod_params *p)
{
	int rc;

	/* Configure the new classification in the chip */
	if (p->rx_mode_obj->config_rx_mode) {
		rc = p->rx_mode_obj->config_rx_mode(sc, p);
		if (rc < 0)
			return rc;

		/* Wait for a ramrod completion if was requested */
		if (ECORE_TEST_BIT(RAMROD_COMP_WAIT, &p->ramrod_flags)) {
			rc = p->rx_mode_obj->wait_comp(sc, p);
			if (rc)
				return rc;
		}
	} else {
		ECORE_MSG(sc, "ERROR: config_rx_mode is NULL");
		return -1;
	}

	return rc;
}

void ecore_init_rx_mode_obj(struct bnx2x_softc *sc, struct ecore_rx_mode_obj *o)
{
	if (CHIP_IS_E1x(sc)) {
		o->wait_comp = ecore_empty_rx_mode_wait;
		o->config_rx_mode = ecore_set_rx_mode_e1x;
	} else {
		o->wait_comp = ecore_wait_rx_mode_comp_e2;
		o->config_rx_mode = ecore_set_rx_mode_e2;
	}
}

/********************* Multicast verbs: SET, CLEAR ****************************/
static uint8_t ecore_mcast_bin_from_mac(uint8_t * mac)
{
	return (ECORE_CRC32_LE(0, mac, ETH_ALEN) >> 24) & 0xff;
}

struct ecore_mcast_mac_elem {
	ecore_list_entry_t link;
	uint8_t mac[ETH_ALEN];
	uint8_t pad[2];		/* For a natural alignment of the following buffer */
};

struct ecore_pending_mcast_cmd {
	ecore_list_entry_t link;
	int type;		/* ECORE_MCAST_CMD_X */
	union {
		ecore_list_t macs_head;
		uint32_t macs_num;	/* Needed for DEL command */
		int next_bin;	/* Needed for RESTORE flow with aprox match */
	} data;

	int done;		/* set to TRUE, when the command has been handled,
				 * practically used in 57712 handling only, where one pending
				 * command may be handled in a few operations. As long as for
				 * other chips every operation handling is completed in a
				 * single ramrod, there is no need to utilize this field.
				 */
};

static int ecore_mcast_wait(struct bnx2x_softc *sc, struct ecore_mcast_obj *o)
{
	if (ecore_state_wait(sc, o->sched_state, o->raw.pstate) ||
	    o->raw.wait_comp(sc, &o->raw))
		return ECORE_TIMEOUT;

	return ECORE_SUCCESS;
}

static int ecore_mcast_enqueue_cmd(struct bnx2x_softc *sc __rte_unused,
				   struct ecore_mcast_obj *o,
				   struct ecore_mcast_ramrod_params *p,
				   enum ecore_mcast_cmd cmd)
{
	int total_sz;
	struct ecore_pending_mcast_cmd *new_cmd;
	struct ecore_mcast_mac_elem *lwr_mac = NULL;
	struct ecore_mcast_list_elem *pos;
	int macs_list_len = ((cmd == ECORE_MCAST_CMD_ADD) ?
			     p->mcast_list_len : 0);

	/* If the command is empty ("handle pending commands only"), break */
	if (!p->mcast_list_len)
		return ECORE_SUCCESS;

	total_sz = sizeof(*new_cmd) +
	    macs_list_len * sizeof(struct ecore_mcast_mac_elem);

	/* Add mcast is called under spin_lock, thus calling with GFP_ATOMIC */
	new_cmd = ECORE_ZALLOC(total_sz, GFP_ATOMIC, sc);

	if (!new_cmd)
		return ECORE_NOMEM;

	ECORE_MSG(sc, "About to enqueue a new %d command. macs_list_len=%d",
		  cmd, macs_list_len);

	ECORE_LIST_INIT(&new_cmd->data.macs_head);

	new_cmd->type = cmd;
	new_cmd->done = FALSE;

	switch (cmd) {
	case ECORE_MCAST_CMD_ADD:
		lwr_mac = (struct ecore_mcast_mac_elem *)
		    ((uint8_t *) new_cmd + sizeof(*new_cmd));

		/* Push the MACs of the current command into the pending command
		 * MACs list: FIFO
		 */
		ECORE_LIST_FOR_EACH_ENTRY(pos, &p->mcast_list, link,
					  struct ecore_mcast_list_elem) {
			ECORE_MEMCPY(lwr_mac->mac, pos->mac, ETH_ALEN);
			ECORE_LIST_PUSH_TAIL(&lwr_mac->link,
					     &new_cmd->data.macs_head);
			lwr_mac++;
		}

		break;

	case ECORE_MCAST_CMD_DEL:
		new_cmd->data.macs_num = p->mcast_list_len;
		break;

	case ECORE_MCAST_CMD_RESTORE:
		new_cmd->data.next_bin = 0;
		break;

	default:
		ECORE_FREE(sc, new_cmd, total_sz);
		PMD_DRV_LOG(ERR, sc, "Unknown command: %d", cmd);
		return ECORE_ILWAL;
	}

	/* Push the new pending command to the tail of the pending list: FIFO */
	ECORE_LIST_PUSH_TAIL(&new_cmd->link, &o->pending_cmds_head);

	o->set_sched(o);

	return ECORE_PENDING;
}

/**
 * ecore_mcast_get_next_bin - get the next set bin (index)
 *
 * @o:
 * @last:	index to start looking from (including)
 *
 * returns the next found (set) bin or a negative value if none is found.
 */
static int ecore_mcast_get_next_bin(struct ecore_mcast_obj *o, int last)
{
	int i, j, inner_start = last % BIT_VEC64_ELEM_SZ;

	for (i = last / BIT_VEC64_ELEM_SZ; i < ECORE_MCAST_VEC_SZ; i++) {
		if (o->registry.aprox_match.vec[i])
			for (j = inner_start; j < BIT_VEC64_ELEM_SZ; j++) {
				int lwr_bit = j + BIT_VEC64_ELEM_SZ * i;
				if (BIT_VEC64_TEST_BIT
				    (o->registry.aprox_match.vec, lwr_bit)) {
					return lwr_bit;
				}
			}
		inner_start = 0;
	}

	/* None found */
	return -1;
}

/**
 * ecore_mcast_clear_first_bin - find the first set bin and clear it
 *
 * @o:
 *
 * returns the index of the found bin or -1 if none is found
 */
static int ecore_mcast_clear_first_bin(struct ecore_mcast_obj *o)
{
	int lwr_bit = ecore_mcast_get_next_bin(o, 0);

	if (lwr_bit >= 0)
		BIT_VEC64_CLEAR_BIT(o->registry.aprox_match.vec, lwr_bit);

	return lwr_bit;
}

static uint8_t ecore_mcast_get_rx_tx_flag(struct ecore_mcast_obj *o)
{
	struct ecore_raw_obj *raw = &o->raw;
	uint8_t rx_tx_flag = 0;

	if ((raw->obj_type == ECORE_OBJ_TYPE_TX) ||
	    (raw->obj_type == ECORE_OBJ_TYPE_RX_TX))
		rx_tx_flag |= ETH_MULTICAST_RULES_CMD_TX_CMD;

	if ((raw->obj_type == ECORE_OBJ_TYPE_RX) ||
	    (raw->obj_type == ECORE_OBJ_TYPE_RX_TX))
		rx_tx_flag |= ETH_MULTICAST_RULES_CMD_RX_CMD;

	return rx_tx_flag;
}

static void ecore_mcast_set_one_rule_e2(struct bnx2x_softc *sc __rte_unused,
					struct ecore_mcast_obj *o, int idx,
					union ecore_mcast_config_data *cfg_data,
					enum ecore_mcast_cmd cmd)
{
	struct ecore_raw_obj *r = &o->raw;
	struct eth_multicast_rules_ramrod_data *data =
	    (struct eth_multicast_rules_ramrod_data *)(r->rdata);
	uint8_t func_id = r->func_id;
	uint8_t rx_tx_add_flag = ecore_mcast_get_rx_tx_flag(o);
	int bin;

	if ((cmd == ECORE_MCAST_CMD_ADD) || (cmd == ECORE_MCAST_CMD_RESTORE))
		rx_tx_add_flag |= ETH_MULTICAST_RULES_CMD_IS_ADD;

	data->rules[idx].cmd_general_data |= rx_tx_add_flag;

	/* Get a bin and update a bins' vector */
	switch (cmd) {
	case ECORE_MCAST_CMD_ADD:
		bin = ecore_mcast_bin_from_mac(cfg_data->mac);
		BIT_VEC64_SET_BIT(o->registry.aprox_match.vec, bin);
		break;

	case ECORE_MCAST_CMD_DEL:
		/* If there were no more bins to clear
		 * (ecore_mcast_clear_first_bin() returns -1) then we would
		 * clear any (0xff) bin.
		 * See ecore_mcast_validate_e2() for explanation when it may
		 * happen.
		 */
		bin = ecore_mcast_clear_first_bin(o);
		break;

	case ECORE_MCAST_CMD_RESTORE:
		bin = cfg_data->bin;
		break;

	default:
		PMD_DRV_LOG(ERR, sc, "Unknown command: %d", cmd);
		return;
	}

	ECORE_MSG(sc, "%s bin %d",
		  ((rx_tx_add_flag & ETH_MULTICAST_RULES_CMD_IS_ADD) ?
		   "Setting" : "Clearing"), bin);

	data->rules[idx].bin_id = (uint8_t) bin;
	data->rules[idx].func_id = func_id;
	data->rules[idx].engine_id = o->engine_id;
}

/**
 * ecore_mcast_handle_restore_cmd_e2 - restore configuration from the registry
 *
 * @sc:		device handle
 * @o:
 * @start_bin:	index in the registry to start from (including)
 * @rdata_idx:	index in the ramrod data to start from
 *
 * returns last handled bin index or -1 if all bins have been handled
 */
static int ecore_mcast_handle_restore_cmd_e2(struct bnx2x_softc *sc,
					     struct ecore_mcast_obj *o,
					     int start_bin, int *rdata_idx)
{
	int lwr_bin, cnt = *rdata_idx;
	union ecore_mcast_config_data cfg_data = { NULL };

	/* go through the registry and configure the bins from it */
	for (lwr_bin = ecore_mcast_get_next_bin(o, start_bin); lwr_bin >= 0;
	     lwr_bin = ecore_mcast_get_next_bin(o, lwr_bin + 1)) {

		cfg_data.bin = (uint8_t) lwr_bin;
		o->set_one_rule(sc, o, cnt, &cfg_data, ECORE_MCAST_CMD_RESTORE);

		cnt++;

		ECORE_MSG(sc, "About to configure a bin %d", lwr_bin);

		/* Break if we reached the maximum number
		 * of rules.
		 */
		if (cnt >= o->max_cmd_len)
			break;
	}

	*rdata_idx = cnt;

	return lwr_bin;
}

static void ecore_mcast_hdl_pending_add_e2(struct bnx2x_softc *sc,
					   struct ecore_mcast_obj *o,
					   struct ecore_pending_mcast_cmd
					   *cmd_pos, int *line_idx)
{
	struct ecore_mcast_mac_elem *pmac_pos, *pmac_pos_n;
	int cnt = *line_idx;
	union ecore_mcast_config_data cfg_data = { NULL };

	ECORE_LIST_FOR_EACH_ENTRY_SAFE(pmac_pos, pmac_pos_n,
				       &cmd_pos->data.macs_head, link,
				       struct ecore_mcast_mac_elem) {

		cfg_data.mac = &pmac_pos->mac[0];
		o->set_one_rule(sc, o, cnt, &cfg_data, cmd_pos->type);

		cnt++;

		    ECORE_MSG
		    (sc, "About to configure %02x:%02x:%02x:%02x:%02x:%02x mcast MAC",
		     pmac_pos->mac[0], pmac_pos->mac[1], pmac_pos->mac[2],
		     pmac_pos->mac[3], pmac_pos->mac[4], pmac_pos->mac[5]);

		ECORE_LIST_REMOVE_ENTRY(&pmac_pos->link,
					&cmd_pos->data.macs_head);

		/* Break if we reached the maximum number
		 * of rules.
		 */
		if (cnt >= o->max_cmd_len)
			break;
	}

	*line_idx = cnt;

	/* if no more MACs to configure - we are done */
	if (ECORE_LIST_IS_EMPTY(&cmd_pos->data.macs_head))
		cmd_pos->done = TRUE;
}

static void ecore_mcast_hdl_pending_del_e2(struct bnx2x_softc *sc,
					   struct ecore_mcast_obj *o,
					   struct ecore_pending_mcast_cmd
					   *cmd_pos, int *line_idx)
{
	int cnt = *line_idx;

	while (cmd_pos->data.macs_num) {
		o->set_one_rule(sc, o, cnt, NULL, cmd_pos->type);

		cnt++;

		cmd_pos->data.macs_num--;

		ECORE_MSG(sc, "Deleting MAC. %d left,cnt is %d",
			  cmd_pos->data.macs_num, cnt);

		/* Break if we reached the maximum
		 * number of rules.
		 */
		if (cnt >= o->max_cmd_len)
			break;
	}

	*line_idx = cnt;

	/* If we cleared all bins - we are done */
	if (!cmd_pos->data.macs_num)
		cmd_pos->done = TRUE;
}

static void ecore_mcast_hdl_pending_restore_e2(struct bnx2x_softc *sc,
					       struct ecore_mcast_obj *o, struct
					       ecore_pending_mcast_cmd
					       *cmd_pos, int *line_idx)
{
	cmd_pos->data.next_bin = o->hdl_restore(sc, o, cmd_pos->data.next_bin,
						line_idx);

	if (cmd_pos->data.next_bin < 0)
		/* If o->set_restore returned -1 we are done */
		cmd_pos->done = TRUE;
	else
		/* Start from the next bin next time */
		cmd_pos->data.next_bin++;
}

static int ecore_mcast_handle_pending_cmds_e2(struct bnx2x_softc *sc, struct
					      ecore_mcast_ramrod_params
					      *p)
{
	struct ecore_pending_mcast_cmd *cmd_pos, *cmd_pos_n;
	int cnt = 0;
	struct ecore_mcast_obj *o = p->mcast_obj;

	ECORE_LIST_FOR_EACH_ENTRY_SAFE(cmd_pos, cmd_pos_n,
				       &o->pending_cmds_head, link,
				       struct ecore_pending_mcast_cmd) {
		switch (cmd_pos->type) {
		case ECORE_MCAST_CMD_ADD:
			ecore_mcast_hdl_pending_add_e2(sc, o, cmd_pos, &cnt);
			break;

		case ECORE_MCAST_CMD_DEL:
			ecore_mcast_hdl_pending_del_e2(sc, o, cmd_pos, &cnt);
			break;

		case ECORE_MCAST_CMD_RESTORE:
			ecore_mcast_hdl_pending_restore_e2(sc, o, cmd_pos,
							   &cnt);
			break;

		default:
			PMD_DRV_LOG(ERR, sc,
				    "Unknown command: %d", cmd_pos->type);
			return ECORE_ILWAL;
		}

		/* If the command has been completed - remove it from the list
		 * and free the memory
		 */
		if (cmd_pos->done) {
			ECORE_LIST_REMOVE_ENTRY(&cmd_pos->link,
						&o->pending_cmds_head);
			ECORE_FREE(sc, cmd_pos, cmd_pos->alloc_len);
		}

		/* Break if we reached the maximum number of rules */
		if (cnt >= o->max_cmd_len)
			break;
	}

	return cnt;
}

static void ecore_mcast_hdl_add(struct bnx2x_softc *sc,
				struct ecore_mcast_obj *o,
				struct ecore_mcast_ramrod_params *p,
				int *line_idx)
{
	struct ecore_mcast_list_elem *mlist_pos;
	union ecore_mcast_config_data cfg_data = { NULL };
	int cnt = *line_idx;

	ECORE_LIST_FOR_EACH_ENTRY(mlist_pos, &p->mcast_list, link,
				  struct ecore_mcast_list_elem) {
		cfg_data.mac = mlist_pos->mac;
		o->set_one_rule(sc, o, cnt, &cfg_data, ECORE_MCAST_CMD_ADD);

		cnt++;

		    ECORE_MSG
		    (sc, "About to configure %02x:%02x:%02x:%02x:%02x:%02x mcast MAC",
		     mlist_pos->mac[0], mlist_pos->mac[1], mlist_pos->mac[2],
		     mlist_pos->mac[3], mlist_pos->mac[4], mlist_pos->mac[5]);
	}

	*line_idx = cnt;
}

static void ecore_mcast_hdl_del(struct bnx2x_softc *sc,
				struct ecore_mcast_obj *o,
				struct ecore_mcast_ramrod_params *p,
				int *line_idx)
{
	int cnt = *line_idx, i;

	for (i = 0; i < p->mcast_list_len; i++) {
		o->set_one_rule(sc, o, cnt, NULL, ECORE_MCAST_CMD_DEL);

		cnt++;

		ECORE_MSG(sc,
			  "Deleting MAC. %d left", p->mcast_list_len - i - 1);
	}

	*line_idx = cnt;
}

/**
 * ecore_mcast_handle_lwrrent_cmd -
 *
 * @sc:		device handle
 * @p:
 * @cmd:
 * @start_cnt:	first line in the ramrod data that may be used
 *
 * This function is called if there is enough place for the current command in
 * the ramrod data.
 * Returns number of lines filled in the ramrod data in total.
 */
static int ecore_mcast_handle_lwrrent_cmd(struct bnx2x_softc *sc, struct
					  ecore_mcast_ramrod_params *p,
					  enum ecore_mcast_cmd cmd,
					  int start_cnt)
{
	struct ecore_mcast_obj *o = p->mcast_obj;
	int cnt = start_cnt;

	ECORE_MSG(sc, "p->mcast_list_len=%d", p->mcast_list_len);

	switch (cmd) {
	case ECORE_MCAST_CMD_ADD:
		ecore_mcast_hdl_add(sc, o, p, &cnt);
		break;

	case ECORE_MCAST_CMD_DEL:
		ecore_mcast_hdl_del(sc, o, p, &cnt);
		break;

	case ECORE_MCAST_CMD_RESTORE:
		o->hdl_restore(sc, o, 0, &cnt);
		break;

	default:
		PMD_DRV_LOG(ERR, sc, "Unknown command: %d", cmd);
		return ECORE_ILWAL;
	}

	/* The current command has been handled */
	p->mcast_list_len = 0;

	return cnt;
}

static int ecore_mcast_validate_e2(__rte_unused struct bnx2x_softc *sc,
				   struct ecore_mcast_ramrod_params *p,
				   enum ecore_mcast_cmd cmd)
{
	struct ecore_mcast_obj *o = p->mcast_obj;
	int reg_sz = o->get_registry_size(o);

	switch (cmd) {
		/* DEL command deletes all lwrrently configured MACs */
	case ECORE_MCAST_CMD_DEL:
		o->set_registry_size(o, 0);
		/* fall-through */

		/* RESTORE command will restore the entire multicast configuration */
	case ECORE_MCAST_CMD_RESTORE:
		/* Here we set the approximate amount of work to do, which in
		 * fact may be only less as some MACs in postponed ADD
		 * command(s) scheduled before this command may fall into
		 * the same bin and the actual number of bins set in the
		 * registry would be less than we estimated here. See
		 * ecore_mcast_set_one_rule_e2() for further details.
		 */
		p->mcast_list_len = reg_sz;
		break;

	case ECORE_MCAST_CMD_ADD:
	case ECORE_MCAST_CMD_CONT:
		/* Here we assume that all new MACs will fall into new bins.
		 * However we will correct the real registry size after we
		 * handle all pending commands.
		 */
		o->set_registry_size(o, reg_sz + p->mcast_list_len);
		break;

	default:
		PMD_DRV_LOG(ERR, sc, "Unknown command: %d", cmd);
		return ECORE_ILWAL;
	}

	/* Increase the total number of MACs pending to be configured */
	o->total_pending_num += p->mcast_list_len;

	return ECORE_SUCCESS;
}

static void ecore_mcast_revert_e2(__rte_unused struct bnx2x_softc *sc,
				  struct ecore_mcast_ramrod_params *p,
				  int old_num_bins,
				  enum ecore_mcast_cmd cmd)
{
	struct ecore_mcast_obj *o = p->mcast_obj;

	o->set_registry_size(o, old_num_bins);
	o->total_pending_num -= p->mcast_list_len;

	if (cmd == ECORE_MCAST_CMD_SET)
		o->total_pending_num -= o->max_cmd_len;
}

/**
 * ecore_mcast_set_rdata_hdr_e2 - sets a header values
 *
 * @sc:		device handle
 * @p:
 * @len:	number of rules to handle
 */
static void ecore_mcast_set_rdata_hdr_e2(__rte_unused struct bnx2x_softc
					 *sc, struct ecore_mcast_ramrod_params
					 *p, uint8_t len)
{
	struct ecore_raw_obj *r = &p->mcast_obj->raw;
	struct eth_multicast_rules_ramrod_data *data =
	    (struct eth_multicast_rules_ramrod_data *)(r->rdata);

	data->header.echo = ECORE_CPU_TO_LE32((r->cid & ECORE_SWCID_MASK) |
					      (ECORE_FILTER_MCAST_PENDING <<
					       ECORE_SWCID_SHIFT));
	data->header.rule_cnt = len;
}

/**
 * ecore_mcast_refresh_registry_e2 - recallwlate the actual number of set bins
 *
 * @sc:		device handle
 * @o:
 *
 * Recallwlate the actual number of set bins in the registry using Brian
 * Kernighan's algorithm: it's exelwtion complexity is as a number of set bins.
 */
static int ecore_mcast_refresh_registry_e2(struct ecore_mcast_obj *o)
{
	int i, cnt = 0;
	uint64_t elem;

	for (i = 0; i < ECORE_MCAST_VEC_SZ; i++) {
		elem = o->registry.aprox_match.vec[i];
		for (; elem; cnt++)
			elem &= elem - 1;
	}

	o->set_registry_size(o, cnt);

	return ECORE_SUCCESS;
}

static int ecore_mcast_setup_e2(struct bnx2x_softc *sc,
				struct ecore_mcast_ramrod_params *p,
				enum ecore_mcast_cmd cmd)
{
	struct ecore_raw_obj *raw = &p->mcast_obj->raw;
	struct ecore_mcast_obj *o = p->mcast_obj;
	struct eth_multicast_rules_ramrod_data *data =
	    (struct eth_multicast_rules_ramrod_data *)(raw->rdata);
	int cnt = 0, rc;

	/* Reset the ramrod data buffer */
	ECORE_MEMSET(data, 0, sizeof(*data));

	cnt = ecore_mcast_handle_pending_cmds_e2(sc, p);

	/* If there are no more pending commands - clear SCHEDULED state */
	if (ECORE_LIST_IS_EMPTY(&o->pending_cmds_head))
		o->clear_sched(o);

	/* The below may be TRUE if there was enough room in ramrod
	 * data for all pending commands and for the current
	 * command. Otherwise the current command would have been added
	 * to the pending commands and p->mcast_list_len would have been
	 * zeroed.
	 */
	if (p->mcast_list_len > 0)
		cnt = ecore_mcast_handle_lwrrent_cmd(sc, p, cmd, cnt);

	/* We've pulled out some MACs - update the total number of
	 * outstanding.
	 */
	o->total_pending_num -= cnt;

	/* send a ramrod */
	ECORE_DBG_BREAK_IF(o->total_pending_num < 0);
	ECORE_DBG_BREAK_IF(cnt > o->max_cmd_len);

	ecore_mcast_set_rdata_hdr_e2(sc, p, (uint8_t) cnt);

	/* Update a registry size if there are no more pending operations.
	 *
	 * We don't want to change the value of the registry size if there are
	 * pending operations because we want it to always be equal to the
	 * exact or the approximate number (see ecore_mcast_validate_e2()) of
	 * set bins after the last requested operation in order to properly
	 * evaluate the size of the next DEL/RESTORE operation.
	 *
	 * Note that we update the registry itself during command(s) handling
	 * - see ecore_mcast_set_one_rule_e2(). That's because for 57712 we
	 * aggregate multiple commands (ADD/DEL/RESTORE) into one ramrod but
	 * with a limited amount of update commands (per MAC/bin) and we don't
	 * know in this scope what the actual state of bins configuration is
	 * going to be after this ramrod.
	 */
	if (!o->total_pending_num)
		ecore_mcast_refresh_registry_e2(o);

	/* If CLEAR_ONLY was requested - don't send a ramrod and clear
	 * RAMROD_PENDING status immediately.
	 */
	if (ECORE_TEST_BIT(RAMROD_DRV_CLR_ONLY, &p->ramrod_flags)) {
		raw->clear_pending(raw);
		return ECORE_SUCCESS;
	} else {
		/* No need for an explicit memory barrier here as long we would
		 * need to ensure the ordering of writing to the SPQ element
		 * and updating of the SPQ producer which ilwolves a memory
		 * read and we will have to put a full memory barrier there
		 * (inside ecore_sp_post()).
		 */

		/* Send a ramrod */
		rc = ecore_sp_post(sc,
				   RAMROD_CMD_ID_ETH_MULTICAST_RULES,
				   raw->cid,
				   raw->rdata_mapping, ETH_CONNECTION_TYPE);
		if (rc)
			return rc;

		/* Ramrod completion is pending */
		return ECORE_PENDING;
	}
}

static int ecore_mcast_validate_e1h(__rte_unused struct bnx2x_softc *sc,
				    struct ecore_mcast_ramrod_params *p,
				    enum ecore_mcast_cmd cmd)
{
	/* Mark, that there is a work to do */
	if ((cmd == ECORE_MCAST_CMD_DEL) || (cmd == ECORE_MCAST_CMD_RESTORE))
		p->mcast_list_len = 1;

	return ECORE_SUCCESS;
}

static void ecore_mcast_revert_e1h(__rte_unused struct bnx2x_softc *sc,
				   __rte_unused struct ecore_mcast_ramrod_params
				   *p, __rte_unused int old_num_bins,
				   __rte_unused enum ecore_mcast_cmd cmd)
{
	/* Do nothing */
}

#define ECORE_57711_SET_MC_FILTER(filter, bit) \
do { \
	(filter)[(bit) >> 5] |= (1 << ((bit) & 0x1f)); \
} while (0)

static void ecore_mcast_hdl_add_e1h(struct bnx2x_softc *sc __rte_unused,
				    struct ecore_mcast_obj *o,
				    struct ecore_mcast_ramrod_params *p,
				    uint32_t * mc_filter)
{
	struct ecore_mcast_list_elem *mlist_pos;
	int bit;

	ECORE_LIST_FOR_EACH_ENTRY(mlist_pos, &p->mcast_list, link,
				  struct ecore_mcast_list_elem) {
		bit = ecore_mcast_bin_from_mac(mlist_pos->mac);
		ECORE_57711_SET_MC_FILTER(mc_filter, bit);

		    ECORE_MSG
		    (sc, "About to configure %02x:%02x:%02x:%02x:%02x:%02x mcast MAC, bin %d",
		     mlist_pos->mac[0], mlist_pos->mac[1], mlist_pos->mac[2],
		     mlist_pos->mac[3], mlist_pos->mac[4], mlist_pos->mac[5],
		     bit);

		/* bookkeeping... */
		BIT_VEC64_SET_BIT(o->registry.aprox_match.vec, bit);
	}
}

static void ecore_mcast_hdl_restore_e1h(struct bnx2x_softc *sc
					__rte_unused,
					struct ecore_mcast_obj *o,
					uint32_t * mc_filter)
{
	int bit;

	for (bit = ecore_mcast_get_next_bin(o, 0);
	     bit >= 0; bit = ecore_mcast_get_next_bin(o, bit + 1)) {
		ECORE_57711_SET_MC_FILTER(mc_filter, bit);
		ECORE_MSG(sc, "About to set bin %d", bit);
	}
}

/* On 57711 we write the multicast MACs' approximate match
 * table by directly into the TSTORM's internal RAM. So we don't
 * really need to handle any tricks to make it work.
 */
static int ecore_mcast_setup_e1h(struct bnx2x_softc *sc,
				 struct ecore_mcast_ramrod_params *p,
				 enum ecore_mcast_cmd cmd)
{
	int i;
	struct ecore_mcast_obj *o = p->mcast_obj;
	struct ecore_raw_obj *r = &o->raw;

	/* If CLEAR_ONLY has been requested - clear the registry
	 * and clear a pending bit.
	 */
	if (!ECORE_TEST_BIT(RAMROD_DRV_CLR_ONLY, &p->ramrod_flags)) {
		uint32_t mc_filter[ECORE_MC_HASH_SIZE] = { 0 };

		/* Set the multicast filter bits before writing it into
		 * the internal memory.
		 */
		switch (cmd) {
		case ECORE_MCAST_CMD_ADD:
			ecore_mcast_hdl_add_e1h(sc, o, p, mc_filter);
			break;

		case ECORE_MCAST_CMD_DEL:
			ECORE_MSG(sc, "Ilwalidating multicast MACs configuration");

			/* clear the registry */
			ECORE_MEMSET(o->registry.aprox_match.vec, 0,
				     sizeof(o->registry.aprox_match.vec));
			break;

		case ECORE_MCAST_CMD_RESTORE:
			ecore_mcast_hdl_restore_e1h(sc, o, mc_filter);
			break;

		default:
			PMD_DRV_LOG(ERR, sc, "Unknown command: %d", cmd);
			return ECORE_ILWAL;
		}

		/* Set the mcast filter in the internal memory */
		for (i = 0; i < ECORE_MC_HASH_SIZE; i++)
			REG_WR(sc, ECORE_MC_HASH_OFFSET(sc, i), mc_filter[i]);
	} else
		/* clear the registry */
		ECORE_MEMSET(o->registry.aprox_match.vec, 0,
			     sizeof(o->registry.aprox_match.vec));

	/* We are done */
	r->clear_pending(r);

	return ECORE_SUCCESS;
}

static int ecore_mcast_get_registry_size_aprox(struct ecore_mcast_obj *o)
{
	return o->registry.aprox_match.num_bins_set;
}

static void ecore_mcast_set_registry_size_aprox(struct ecore_mcast_obj *o,
						int n)
{
	o->registry.aprox_match.num_bins_set = n;
}

int ecore_config_mcast(struct bnx2x_softc *sc,
		       struct ecore_mcast_ramrod_params *p,
		       enum ecore_mcast_cmd cmd)
{
	struct ecore_mcast_obj *o = p->mcast_obj;
	struct ecore_raw_obj *r = &o->raw;
	int rc = 0, old_reg_size;

	/* This is needed to recover number of lwrrently configured mcast macs
	 * in case of failure.
	 */
	old_reg_size = o->get_registry_size(o);

	/* Do some callwlations and checks */
	rc = o->validate(sc, p, cmd);
	if (rc)
		return rc;

	/* Return if there is no work to do */
	if ((!p->mcast_list_len) && (!o->check_sched(o)))
		return ECORE_SUCCESS;

	    ECORE_MSG
	    (sc, "o->total_pending_num=%d p->mcast_list_len=%d o->max_cmd_len=%d",
	     o->total_pending_num, p->mcast_list_len, o->max_cmd_len);

	/* Enqueue the current command to the pending list if we can't complete
	 * it in the current iteration
	 */
	if (r->check_pending(r) ||
	    ((o->max_cmd_len > 0) && (o->total_pending_num > o->max_cmd_len))) {
		rc = o->enqueue_cmd(sc, p->mcast_obj, p, cmd);
		if (rc < 0)
			goto error_exit1;

		/* As long as the current command is in a command list we
		 * don't need to handle it separately.
		 */
		p->mcast_list_len = 0;
	}

	if (!r->check_pending(r)) {

		/* Set 'pending' state */
		r->set_pending(r);

		/* Configure the new classification in the chip */
		rc = o->config_mcast(sc, p, cmd);
		if (rc < 0)
			goto error_exit2;

		/* Wait for a ramrod completion if was requested */
		if (ECORE_TEST_BIT(RAMROD_COMP_WAIT, &p->ramrod_flags))
			rc = o->wait_comp(sc, o);
	}

	return rc;

error_exit2:
	r->clear_pending(r);

error_exit1:
	o->revert(sc, p, old_reg_size, cmd);

	return rc;
}

static void ecore_mcast_clear_sched(struct ecore_mcast_obj *o)
{
	ECORE_SMP_MB_BEFORE_CLEAR_BIT();
	ECORE_CLEAR_BIT(o->sched_state, o->raw.pstate);
	ECORE_SMP_MB_AFTER_CLEAR_BIT();
}

static void ecore_mcast_set_sched(struct ecore_mcast_obj *o)
{
	ECORE_SMP_MB_BEFORE_CLEAR_BIT();
	ECORE_SET_BIT(o->sched_state, o->raw.pstate);
	ECORE_SMP_MB_AFTER_CLEAR_BIT();
}

static bool ecore_mcast_check_sched(struct ecore_mcast_obj *o)
{
	return ! !ECORE_TEST_BIT(o->sched_state, o->raw.pstate);
}

static bool ecore_mcast_check_pending(struct ecore_mcast_obj *o)
{
	return o->raw.check_pending(&o->raw) || o->check_sched(o);
}

void ecore_init_mcast_obj(struct bnx2x_softc *sc,
			  struct ecore_mcast_obj *mcast_obj,
			  uint8_t mcast_cl_id, uint32_t mcast_cid,
			  uint8_t func_id, uint8_t engine_id, void *rdata,
			  ecore_dma_addr_t rdata_mapping, int state,
			  uint32_t *pstate, ecore_obj_type type)
{
	ECORE_MEMSET(mcast_obj, 0, sizeof(*mcast_obj));

	ecore_init_raw_obj(&mcast_obj->raw, mcast_cl_id, mcast_cid, func_id,
			   rdata, rdata_mapping, state, pstate, type);

	mcast_obj->engine_id = engine_id;

	ECORE_LIST_INIT(&mcast_obj->pending_cmds_head);

	mcast_obj->sched_state = ECORE_FILTER_MCAST_SCHED;
	mcast_obj->check_sched = ecore_mcast_check_sched;
	mcast_obj->set_sched = ecore_mcast_set_sched;
	mcast_obj->clear_sched = ecore_mcast_clear_sched;

	if (CHIP_IS_E1H(sc)) {
		mcast_obj->config_mcast = ecore_mcast_setup_e1h;
		mcast_obj->enqueue_cmd = NULL;
		mcast_obj->hdl_restore = NULL;
		mcast_obj->check_pending = ecore_mcast_check_pending;

		/* 57711 doesn't send a ramrod, so it has unlimited credit
		 * for one command.
		 */
		mcast_obj->max_cmd_len = -1;
		mcast_obj->wait_comp = ecore_mcast_wait;
		mcast_obj->set_one_rule = NULL;
		mcast_obj->validate = ecore_mcast_validate_e1h;
		mcast_obj->revert = ecore_mcast_revert_e1h;
		mcast_obj->get_registry_size =
		    ecore_mcast_get_registry_size_aprox;
		mcast_obj->set_registry_size =
		    ecore_mcast_set_registry_size_aprox;
	} else {
		mcast_obj->config_mcast = ecore_mcast_setup_e2;
		mcast_obj->enqueue_cmd = ecore_mcast_enqueue_cmd;
		mcast_obj->hdl_restore = ecore_mcast_handle_restore_cmd_e2;
		mcast_obj->check_pending = ecore_mcast_check_pending;
		mcast_obj->max_cmd_len = 16;
		mcast_obj->wait_comp = ecore_mcast_wait;
		mcast_obj->set_one_rule = ecore_mcast_set_one_rule_e2;
		mcast_obj->validate = ecore_mcast_validate_e2;
		mcast_obj->revert = ecore_mcast_revert_e2;
		mcast_obj->get_registry_size =
		    ecore_mcast_get_registry_size_aprox;
		mcast_obj->set_registry_size =
		    ecore_mcast_set_registry_size_aprox;
	}
}

/*************************** Credit handling **********************************/

/**
 * atomic_add_ifless - add if the result is less than a given value.
 *
 * @v:	pointer of type ecore_atomic_t
 * @a:	the amount to add to v...
 * @u:	...if (v + a) is less than u.
 *
 * returns TRUE if (v + a) was less than u, and FALSE otherwise.
 *
 */
static bool __atomic_add_ifless(ecore_atomic_t *v, int a, int u)
{
	int c, old;

	c = ECORE_ATOMIC_READ(v);
	for (;;) {
		if (ECORE_UNLIKELY(c + a >= u))
			return FALSE;

		old = ECORE_ATOMIC_CMPXCHG((v), c, c + a);
		if (ECORE_LIKELY(old == c))
			break;
		c = old;
	}

	return TRUE;
}

/**
 * atomic_dec_ifmoe - dec if the result is more or equal than a given value.
 *
 * @v:	pointer of type ecore_atomic_t
 * @a:	the amount to dec from v...
 * @u:	...if (v - a) is more or equal than u.
 *
 * returns TRUE if (v - a) was more or equal than u, and FALSE
 * otherwise.
 */
static bool __atomic_dec_ifmoe(ecore_atomic_t *v, int a, int u)
{
	int c, old;

	c = ECORE_ATOMIC_READ(v);
	for (;;) {
		if (ECORE_UNLIKELY(c - a < u))
			return FALSE;

		old = ECORE_ATOMIC_CMPXCHG((v), c, c - a);
		if (ECORE_LIKELY(old == c))
			break;
		c = old;
	}

	return TRUE;
}

static bool ecore_credit_pool_get(struct ecore_credit_pool_obj *o, int cnt)
{
	bool rc;

	ECORE_SMP_MB();
	rc = __atomic_dec_ifmoe(&o->credit, cnt, 0);
	ECORE_SMP_MB();

	return rc;
}

static bool ecore_credit_pool_put(struct ecore_credit_pool_obj *o, int cnt)
{
	bool rc;

	ECORE_SMP_MB();

	/* Don't let to refill if credit + cnt > pool_sz */
	rc = __atomic_add_ifless(&o->credit, cnt, o->pool_sz + 1);

	ECORE_SMP_MB();

	return rc;
}

static int ecore_credit_pool_check(struct ecore_credit_pool_obj *o)
{
	int lwr_credit;

	ECORE_SMP_MB();
	lwr_credit = ECORE_ATOMIC_READ(&o->credit);

	return lwr_credit;
}

static bool ecore_credit_pool_always_TRUE(__rte_unused struct
					 ecore_credit_pool_obj *o,
					 __rte_unused int cnt)
{
	return TRUE;
}

static bool ecore_credit_pool_get_entry(struct ecore_credit_pool_obj *o,
				       int *offset)
{
	int idx, vec, i;

	*offset = -1;

	/* Find "internal cam-offset" then add to base for this object... */
	for (vec = 0; vec < ECORE_POOL_VEC_SIZE; vec++) {

		/* Skip the current vector if there are no free entries in it */
		if (!o->pool_mirror[vec])
			continue;

		/* If we've got here we are going to find a free entry */
		for (idx = vec * BIT_VEC64_ELEM_SZ, i = 0;
		     i < BIT_VEC64_ELEM_SZ; idx++, i++)

			if (BIT_VEC64_TEST_BIT(o->pool_mirror, idx)) {
				/* Got one!! */
				BIT_VEC64_CLEAR_BIT(o->pool_mirror, idx);
				*offset = o->base_pool_offset + idx;
				return TRUE;
			}
	}

	return FALSE;
}

static bool ecore_credit_pool_put_entry(struct ecore_credit_pool_obj *o,
				       int offset)
{
	if (offset < o->base_pool_offset)
		return FALSE;

	offset -= o->base_pool_offset;

	if (offset >= o->pool_sz)
		return FALSE;

	/* Return the entry to the pool */
	BIT_VEC64_SET_BIT(o->pool_mirror, offset);

	return TRUE;
}

static bool ecore_credit_pool_put_entry_always_TRUE(__rte_unused struct
						   ecore_credit_pool_obj *o,
						   __rte_unused int offset)
{
	return TRUE;
}

static bool ecore_credit_pool_get_entry_always_TRUE(__rte_unused struct
						   ecore_credit_pool_obj *o,
						   __rte_unused int *offset)
{
	*offset = -1;
	return TRUE;
}

/**
 * ecore_init_credit_pool - initialize credit pool internals.
 *
 * @p:
 * @base:	Base entry in the CAM to use.
 * @credit:	pool size.
 *
 * If base is negative no CAM entries handling will be performed.
 * If credit is negative pool operations will always succeed (unlimited pool).
 *
 */
void ecore_init_credit_pool(struct ecore_credit_pool_obj *p,
				   int base, int credit)
{
	/* Zero the object first */
	ECORE_MEMSET(p, 0, sizeof(*p));

	/* Set the table to all 1s */
	ECORE_MEMSET(&p->pool_mirror, 0xff, sizeof(p->pool_mirror));

	/* Init a pool as full */
	ECORE_ATOMIC_SET(&p->credit, credit);

	/* The total poll size */
	p->pool_sz = credit;

	p->base_pool_offset = base;

	/* Commit the change */
	ECORE_SMP_MB();

	p->check = ecore_credit_pool_check;

	/* if pool credit is negative - disable the checks */
	if (credit >= 0) {
		p->put = ecore_credit_pool_put;
		p->get = ecore_credit_pool_get;
		p->put_entry = ecore_credit_pool_put_entry;
		p->get_entry = ecore_credit_pool_get_entry;
	} else {
		p->put = ecore_credit_pool_always_TRUE;
		p->get = ecore_credit_pool_always_TRUE;
		p->put_entry = ecore_credit_pool_put_entry_always_TRUE;
		p->get_entry = ecore_credit_pool_get_entry_always_TRUE;
	}

	/* If base is negative - disable entries handling */
	if (base < 0) {
		p->put_entry = ecore_credit_pool_put_entry_always_TRUE;
		p->get_entry = ecore_credit_pool_get_entry_always_TRUE;
	}
}

void ecore_init_mac_credit_pool(struct bnx2x_softc *sc,
				struct ecore_credit_pool_obj *p,
				uint8_t func_id, uint8_t func_num)
{

#define ECORE_CAM_SIZE_EMUL 5

	int cam_sz;

	if (CHIP_IS_E1H(sc)) {
		/* CAM credit is equally divided between all active functions
		 * on the PORT!.
		 */
		if (func_num > 0) {
			if (!CHIP_REV_IS_SLOW(sc))
				cam_sz = (MAX_MAC_CREDIT_E1H / (2 * func_num));
			else
				cam_sz = ECORE_CAM_SIZE_EMUL;
			ecore_init_credit_pool(p, func_id * cam_sz, cam_sz);
		} else {
			/* this should never happen! Block MAC operations. */
			ecore_init_credit_pool(p, 0, 0);
		}

	} else {

		/*
		 * CAM credit is equaly divided between all active functions
		 * on the PATH.
		 */
		if (func_num > 0) {
			if (!CHIP_REV_IS_SLOW(sc))
				cam_sz = (MAX_MAC_CREDIT_E2 / func_num);
			else
				cam_sz = ECORE_CAM_SIZE_EMUL;

			/* No need for CAM entries handling for 57712 and
			 * newer.
			 */
			ecore_init_credit_pool(p, -1, cam_sz);
		} else {
			/* this should never happen! Block MAC operations. */
			ecore_init_credit_pool(p, 0, 0);
		}
	}
}

void ecore_init_vlan_credit_pool(struct bnx2x_softc *sc,
				 struct ecore_credit_pool_obj *p,
				 uint8_t func_id, uint8_t func_num)
{
	if (CHIP_IS_E1x(sc)) {
		/* There is no VLAN credit in HW on 57711 only
		 * MAC / MAC-VLAN can be set
		 */
		ecore_init_credit_pool(p, 0, -1);
	} else {
		/* CAM credit is equally divided between all active functions
		 * on the PATH.
		 */
		if (func_num > 0) {
			int credit = MAX_VLAN_CREDIT_E2 / func_num;
			ecore_init_credit_pool(p, func_id * credit, credit);
		} else
			/* this should never happen! Block VLAN operations. */
			ecore_init_credit_pool(p, 0, 0);
	}
}

/****************** RSS Configuration ******************/

/**
 * ecore_setup_rss - configure RSS
 *
 * @sc:		device handle
 * @p:		rss configuration
 *
 * sends on UPDATE ramrod for that matter.
 */
static int ecore_setup_rss(struct bnx2x_softc *sc,
			   struct ecore_config_rss_params *p)
{
	struct ecore_rss_config_obj *o = p->rss_obj;
	struct ecore_raw_obj *r = &o->raw;
	struct eth_rss_update_ramrod_data *data =
	    (struct eth_rss_update_ramrod_data *)(r->rdata);
	uint8_t rss_mode = 0;
	int rc;

	ECORE_MEMSET(data, 0, sizeof(*data));

	ECORE_MSG(sc, "Configuring RSS");

	/* Set an echo field */
	data->echo = ECORE_CPU_TO_LE32((r->cid & ECORE_SWCID_MASK) |
				       (r->state << ECORE_SWCID_SHIFT));

	/* RSS mode */
	if (ECORE_TEST_BIT(ECORE_RSS_MODE_DISABLED, &p->rss_flags))
		rss_mode = ETH_RSS_MODE_DISABLED;
	else if (ECORE_TEST_BIT(ECORE_RSS_MODE_REGULAR, &p->rss_flags))
		rss_mode = ETH_RSS_MODE_REGULAR;

	data->rss_mode = rss_mode;

	ECORE_MSG(sc, "rss_mode=%d", rss_mode);

	/* RSS capabilities */
	if (ECORE_TEST_BIT(ECORE_RSS_IPV4, &p->rss_flags))
		data->capabilities |=
		    ETH_RSS_UPDATE_RAMROD_DATA_IPV4_CAPABILITY;

	if (ECORE_TEST_BIT(ECORE_RSS_IPV4_TCP, &p->rss_flags))
		data->capabilities |=
		    ETH_RSS_UPDATE_RAMROD_DATA_IPV4_TCP_CAPABILITY;

	if (ECORE_TEST_BIT(ECORE_RSS_IPV4_UDP, &p->rss_flags))
		data->capabilities |=
		    ETH_RSS_UPDATE_RAMROD_DATA_IPV4_UDP_CAPABILITY;

	if (ECORE_TEST_BIT(ECORE_RSS_IPV6, &p->rss_flags))
		data->capabilities |=
		    ETH_RSS_UPDATE_RAMROD_DATA_IPV6_CAPABILITY;

	if (ECORE_TEST_BIT(ECORE_RSS_IPV6_TCP, &p->rss_flags))
		data->capabilities |=
		    ETH_RSS_UPDATE_RAMROD_DATA_IPV6_TCP_CAPABILITY;

	if (ECORE_TEST_BIT(ECORE_RSS_IPV6_UDP, &p->rss_flags))
		data->capabilities |=
		    ETH_RSS_UPDATE_RAMROD_DATA_IPV6_UDP_CAPABILITY;

	/* Hashing mask */
	data->rss_result_mask = p->rss_result_mask;

	/* RSS engine ID */
	data->rss_engine_id = o->engine_id;

	ECORE_MSG(sc, "rss_engine_id=%d", data->rss_engine_id);

	/* Indirection table */
	ECORE_MEMCPY(data->indirection_table, p->ind_table,
		     T_ETH_INDIRECTION_TABLE_SIZE);

	/* Remember the last configuration */
	ECORE_MEMCPY(o->ind_table, p->ind_table, T_ETH_INDIRECTION_TABLE_SIZE);

	/* RSS keys */
	if (ECORE_TEST_BIT(ECORE_RSS_SET_SRCH, &p->rss_flags)) {
		ECORE_MEMCPY(&data->rss_key[0], &p->rss_key[0],
			     sizeof(data->rss_key));
		data->capabilities |= ETH_RSS_UPDATE_RAMROD_DATA_UPDATE_RSS_KEY;
	}

	/* No need for an explicit memory barrier here as long we would
	 * need to ensure the ordering of writing to the SPQ element
	 * and updating of the SPQ producer which ilwolves a memory
	 * read and we will have to put a full memory barrier there
	 * (inside ecore_sp_post()).
	 */

	/* Send a ramrod */
	rc = ecore_sp_post(sc,
			   RAMROD_CMD_ID_ETH_RSS_UPDATE,
			   r->cid, r->rdata_mapping, ETH_CONNECTION_TYPE);

	if (rc < 0)
		return rc;

	return ECORE_PENDING;
}

int ecore_config_rss(struct bnx2x_softc *sc, struct ecore_config_rss_params *p)
{
	int rc;
	struct ecore_rss_config_obj *o = p->rss_obj;
	struct ecore_raw_obj *r = &o->raw;

	/* Do nothing if only driver cleanup was requested */
	if (ECORE_TEST_BIT(RAMROD_DRV_CLR_ONLY, &p->ramrod_flags))
		return ECORE_SUCCESS;

	r->set_pending(r);

	rc = o->config_rss(sc, p);
	if (rc < 0) {
		r->clear_pending(r);
		return rc;
	}

	if (ECORE_TEST_BIT(RAMROD_COMP_WAIT, &p->ramrod_flags))
		rc = r->wait_comp(sc, r);

	return rc;
}

void ecore_init_rss_config_obj(struct bnx2x_softc *sc __rte_unused,
			       struct ecore_rss_config_obj *rss_obj,
			       uint8_t cl_id, uint32_t cid, uint8_t func_id,
			       uint8_t engine_id,
			       void *rdata, ecore_dma_addr_t rdata_mapping,
			       int state, uint32_t *pstate,
			       ecore_obj_type type)
{
	ecore_init_raw_obj(&rss_obj->raw, cl_id, cid, func_id, rdata,
			   rdata_mapping, state, pstate, type);

	rss_obj->engine_id = engine_id;
	rss_obj->config_rss = ecore_setup_rss;
}

/********************** Queue state object ***********************************/

/**
 * ecore_queue_state_change - perform Queue state change transition
 *
 * @sc:		device handle
 * @params:	parameters to perform the transition
 *
 * returns 0 in case of successfully completed transition, negative error
 * code in case of failure, positive (EBUSY) value if there is a completion
 * to that is still pending (possible only if RAMROD_COMP_WAIT is
 * not set in params->ramrod_flags for asynchronous commands).
 *
 */
int ecore_queue_state_change(struct bnx2x_softc *sc,
			     struct ecore_queue_state_params *params)
{
	struct ecore_queue_sp_obj *o = params->q_obj;
	int rc, pending_bit;
	uint32_t *pending = &o->pending;

	/* Check that the requested transition is legal */
	rc = o->check_transition(sc, o, params);
	if (rc) {
		PMD_DRV_LOG(ERR, sc, "check transition returned an error. rc %d",
			    rc);
		return ECORE_ILWAL;
	}

	/* Set "pending" bit */
	ECORE_MSG(sc, "pending bit was=%x", o->pending);
	pending_bit = o->set_pending(o, params);
	ECORE_MSG(sc, "pending bit now=%x", o->pending);

	/* Don't send a command if only driver cleanup was requested */
	if (ECORE_TEST_BIT(RAMROD_DRV_CLR_ONLY, &params->ramrod_flags))
		o->complete_cmd(sc, o, pending_bit);
	else {
		/* Send a ramrod */
		rc = o->send_cmd(sc, params);
		if (rc) {
			o->next_state = ECORE_Q_STATE_MAX;
			ECORE_CLEAR_BIT(pending_bit, pending);
			ECORE_SMP_MB_AFTER_CLEAR_BIT();
			return rc;
		}

		if (ECORE_TEST_BIT(RAMROD_COMP_WAIT, &params->ramrod_flags)) {
			rc = o->wait_comp(sc, o, pending_bit);
			if (rc)
				return rc;

			return ECORE_SUCCESS;
		}
	}

	return ECORE_RET_PENDING(pending_bit, pending);
}

static int ecore_queue_set_pending(struct ecore_queue_sp_obj *obj,
				   struct ecore_queue_state_params *params)
{
	enum ecore_queue_cmd cmd = params->cmd, bit;

	/* ACTIVATE and DEACTIVATE commands are implemented on top of
	 * UPDATE command.
	 */
	if ((cmd == ECORE_Q_CMD_ACTIVATE) || (cmd == ECORE_Q_CMD_DEACTIVATE))
		bit = ECORE_Q_CMD_UPDATE;
	else
		bit = cmd;

	ECORE_SET_BIT(bit, &obj->pending);
	return bit;
}

static int ecore_queue_wait_comp(struct bnx2x_softc *sc,
				 struct ecore_queue_sp_obj *o,
				 enum ecore_queue_cmd cmd)
{
	return ecore_state_wait(sc, cmd, &o->pending);
}

/**
 * ecore_queue_comp_cmd - complete the state change command.
 *
 * @sc:		device handle
 * @o:
 * @cmd:
 *
 * Checks that the arrived completion is expected.
 */
static int ecore_queue_comp_cmd(struct bnx2x_softc *sc __rte_unused,
				struct ecore_queue_sp_obj *o,
				enum ecore_queue_cmd cmd)
{
	uint32_t lwr_pending = o->pending;

	if (!ECORE_TEST_AND_CLEAR_BIT(cmd, &lwr_pending)) {
		PMD_DRV_LOG(ERR, sc,
			    "Bad MC reply %d for queue %d in state %d pending 0x%x, next_state %d",
			    cmd, o->cids[ECORE_PRIMARY_CID_INDEX], o->state,
			    lwr_pending, o->next_state);
		return ECORE_ILWAL;
	}

	if (o->next_tx_only >= o->max_cos)
		/* >= because tx only must always be smaller than cos since the
		 * primary connection supports COS 0
		 */
		PMD_DRV_LOG(ERR, sc,
			    "illegal value for next tx_only: %d. max cos was %d",
			    o->next_tx_only, o->max_cos);

	ECORE_MSG(sc, "Completing command %d for queue %d, setting state to %d",
		  cmd, o->cids[ECORE_PRIMARY_CID_INDEX], o->next_state);

	if (o->next_tx_only)	/* print num tx-only if any exist */
		ECORE_MSG(sc, "primary cid %d: num tx-only cons %d",
			  o->cids[ECORE_PRIMARY_CID_INDEX], o->next_tx_only);

	o->state = o->next_state;
	o->num_tx_only = o->next_tx_only;
	o->next_state = ECORE_Q_STATE_MAX;

	/* It's important that o->state and o->next_state are
	 * updated before o->pending.
	 */
	wmb();

	ECORE_CLEAR_BIT(cmd, &o->pending);
	ECORE_SMP_MB_AFTER_CLEAR_BIT();

	return ECORE_SUCCESS;
}

static void ecore_q_fill_setup_data_e2(struct ecore_queue_state_params
				       *cmd_params,
				       struct client_init_ramrod_data *data)
{
	struct ecore_queue_setup_params *params = &cmd_params->params.setup;

	/* Rx data */

	/* IPv6 TPA supported for E2 and above only */
	data->rx.tpa_en |= ECORE_TEST_BIT(ECORE_Q_FLG_TPA_IPV6,
					  &params->flags) *
	    CLIENT_INIT_RX_DATA_TPA_EN_IPV6;
}

static void ecore_q_fill_init_general_data(struct bnx2x_softc *sc __rte_unused,
					   struct ecore_queue_sp_obj *o,
					   struct ecore_general_setup_params
					   *params, struct client_init_general_data
					   *gen_data, uint32_t *flags)
{
	gen_data->client_id = o->cl_id;

	if (ECORE_TEST_BIT(ECORE_Q_FLG_STATS, flags)) {
		gen_data->statistics_counter_id = params->stat_id;
		gen_data->statistics_en_flg = 1;
		gen_data->statistics_zero_flg =
		    ECORE_TEST_BIT(ECORE_Q_FLG_ZERO_STATS, flags);
	} else
		gen_data->statistics_counter_id =
		    DISABLE_STATISTIC_COUNTER_ID_VALUE;

	gen_data->is_fcoe_flg = ECORE_TEST_BIT(ECORE_Q_FLG_FCOE, flags);
	gen_data->activate_flg = ECORE_TEST_BIT(ECORE_Q_FLG_ACTIVE, flags);
	gen_data->sp_client_id = params->spcl_id;
	gen_data->mtu = ECORE_CPU_TO_LE16(params->mtu);
	gen_data->func_id = o->func_id;

	gen_data->cos = params->cos;

	gen_data->traffic_type =
	    ECORE_TEST_BIT(ECORE_Q_FLG_FCOE, flags) ?
	    LLFC_TRAFFIC_TYPE_FCOE : LLFC_TRAFFIC_TYPE_NW;

	ECORE_MSG(sc, "flags: active %d, cos %d, stats en %d",
		  gen_data->activate_flg, gen_data->cos,
		  gen_data->statistics_en_flg);
}

static void ecore_q_fill_init_tx_data(struct ecore_txq_setup_params *params,
				      struct client_init_tx_data *tx_data,
				      uint32_t *flags)
{
	tx_data->enforce_selwrity_flg =
	    ECORE_TEST_BIT(ECORE_Q_FLG_TX_SEC, flags);
	tx_data->default_vlan = ECORE_CPU_TO_LE16(params->default_vlan);
	tx_data->default_vlan_flg = ECORE_TEST_BIT(ECORE_Q_FLG_DEF_VLAN, flags);
	tx_data->tx_switching_flg =
	    ECORE_TEST_BIT(ECORE_Q_FLG_TX_SWITCH, flags);
	tx_data->anti_spoofing_flg =
	    ECORE_TEST_BIT(ECORE_Q_FLG_ANTI_SPOOF, flags);
	tx_data->force_default_pri_flg =
	    ECORE_TEST_BIT(ECORE_Q_FLG_FORCE_DEFAULT_PRI, flags);
	tx_data->refuse_outband_vlan_flg =
	    ECORE_TEST_BIT(ECORE_Q_FLG_REFUSE_OUTBAND_VLAN, flags);
	tx_data->tunnel_non_lso_pcsum_location =
	    ECORE_TEST_BIT(ECORE_Q_FLG_PCSUM_ON_PKT, flags) ? CSUM_ON_PKT :
	    CSUM_ON_BD;

	tx_data->tx_status_block_id = params->fw_sb_id;
	tx_data->tx_sb_index_number = params->sb_cq_index;
	tx_data->tss_leading_client_id = params->tss_leading_cl_id;

	tx_data->tx_bd_page_base.lo =
	    ECORE_CPU_TO_LE32(U64_LO(params->dscr_map));
	tx_data->tx_bd_page_base.hi =
	    ECORE_CPU_TO_LE32(U64_HI(params->dscr_map));

	/* Don't configure any Tx switching mode during queue SETUP */
	tx_data->state = 0;
}

static void ecore_q_fill_init_pause_data(struct rxq_pause_params *params,
					 struct client_init_rx_data *rx_data)
{
	/* flow control data */
	rx_data->cqe_pause_thr_low = ECORE_CPU_TO_LE16(params->rcq_th_lo);
	rx_data->cqe_pause_thr_high = ECORE_CPU_TO_LE16(params->rcq_th_hi);
	rx_data->bd_pause_thr_low = ECORE_CPU_TO_LE16(params->bd_th_lo);
	rx_data->bd_pause_thr_high = ECORE_CPU_TO_LE16(params->bd_th_hi);
	rx_data->sge_pause_thr_low = ECORE_CPU_TO_LE16(params->sge_th_lo);
	rx_data->sge_pause_thr_high = ECORE_CPU_TO_LE16(params->sge_th_hi);
	rx_data->rx_cos_mask = ECORE_CPU_TO_LE16(params->pri_map);
}

static void ecore_q_fill_init_rx_data(struct ecore_rxq_setup_params *params,
				      struct client_init_rx_data *rx_data,
				      uint32_t *flags)
{
	rx_data->tpa_en = ECORE_TEST_BIT(ECORE_Q_FLG_TPA, flags) *
	    CLIENT_INIT_RX_DATA_TPA_EN_IPV4;
	rx_data->tpa_en |= ECORE_TEST_BIT(ECORE_Q_FLG_TPA_GRO, flags) *
	    CLIENT_INIT_RX_DATA_TPA_MODE;
	rx_data->vmqueue_mode_en_flg = 0;

	rx_data->extra_data_over_sgl_en_flg =
	    ECORE_TEST_BIT(ECORE_Q_FLG_OOO, flags);
	rx_data->cache_line_alignment_log_size = params->cache_line_log;
	rx_data->enable_dynamic_hc = ECORE_TEST_BIT(ECORE_Q_FLG_DHC, flags);
	rx_data->client_qzone_id = params->cl_qzone_id;
	rx_data->max_agg_size = ECORE_CPU_TO_LE16(params->tpa_agg_sz);

	/* Always start in DROP_ALL mode */
	rx_data->state = ECORE_CPU_TO_LE16(CLIENT_INIT_RX_DATA_UCAST_DROP_ALL |
					   CLIENT_INIT_RX_DATA_MCAST_DROP_ALL);

	/* We don't set drop flags */
	rx_data->drop_ip_cs_err_flg = 0;
	rx_data->drop_tcp_cs_err_flg = 0;
	rx_data->drop_ttl0_flg = 0;
	rx_data->drop_udp_cs_err_flg = 0;
	rx_data->inner_vlan_removal_enable_flg =
	    ECORE_TEST_BIT(ECORE_Q_FLG_VLAN, flags);
	rx_data->outer_vlan_removal_enable_flg =
	    ECORE_TEST_BIT(ECORE_Q_FLG_OV, flags);
	rx_data->status_block_id = params->fw_sb_id;
	rx_data->rx_sb_index_number = params->sb_cq_index;
	rx_data->max_tpa_queues = params->max_tpa_queues;
	rx_data->max_bytes_on_bd = ECORE_CPU_TO_LE16(params->buf_sz);
	rx_data->bd_page_base.lo = ECORE_CPU_TO_LE32(U64_LO(params->dscr_map));
	rx_data->bd_page_base.hi = ECORE_CPU_TO_LE32(U64_HI(params->dscr_map));
	rx_data->cqe_page_base.lo = ECORE_CPU_TO_LE32(U64_LO(params->rcq_map));
	rx_data->cqe_page_base.hi = ECORE_CPU_TO_LE32(U64_HI(params->rcq_map));
	rx_data->is_leading_rss = ECORE_TEST_BIT(ECORE_Q_FLG_LEADING_RSS,
						 flags);

	if (ECORE_TEST_BIT(ECORE_Q_FLG_MCAST, flags)) {
		rx_data->approx_mcast_engine_id = params->mcast_engine_id;
		rx_data->is_approx_mcast = 1;
	}

	rx_data->rss_engine_id = params->rss_engine_id;

	/* silent vlan removal */
	rx_data->silent_vlan_removal_flg =
	    ECORE_TEST_BIT(ECORE_Q_FLG_SILENT_VLAN_REM, flags);
	rx_data->silent_vlan_value =
	    ECORE_CPU_TO_LE16(params->silent_removal_value);
	rx_data->silent_vlan_mask =
	    ECORE_CPU_TO_LE16(params->silent_removal_mask);
}

/* initialize the general, tx and rx parts of a queue object */
static void ecore_q_fill_setup_data_cmn(struct bnx2x_softc *sc, struct ecore_queue_state_params
					*cmd_params,
					struct client_init_ramrod_data *data)
{
	ecore_q_fill_init_general_data(sc, cmd_params->q_obj,
				       &cmd_params->params.setup.gen_params,
				       &data->general,
				       &cmd_params->params.setup.flags);

	ecore_q_fill_init_tx_data(&cmd_params->params.setup.txq_params,
				  &data->tx, &cmd_params->params.setup.flags);

	ecore_q_fill_init_rx_data(&cmd_params->params.setup.rxq_params,
				  &data->rx, &cmd_params->params.setup.flags);

	ecore_q_fill_init_pause_data(&cmd_params->params.setup.pause_params,
				     &data->rx);
}

/* initialize the general and tx parts of a tx-only queue object */
static void ecore_q_fill_setup_tx_only(struct bnx2x_softc *sc, struct ecore_queue_state_params
				       *cmd_params,
				       struct tx_queue_init_ramrod_data *data)
{
	ecore_q_fill_init_general_data(sc, cmd_params->q_obj,
				       &cmd_params->params.tx_only.gen_params,
				       &data->general,
				       &cmd_params->params.tx_only.flags);

	ecore_q_fill_init_tx_data(&cmd_params->params.tx_only.txq_params,
				  &data->tx, &cmd_params->params.tx_only.flags);

	ECORE_MSG(sc, "cid %d, tx bd page lo %x hi %x",
		  cmd_params->q_obj->cids[0],
		  data->tx.tx_bd_page_base.lo, data->tx.tx_bd_page_base.hi);
}

/**
 * ecore_q_init - init HW/FW queue
 *
 * @sc:		device handle
 * @params:
 *
 * HW/FW initial Queue configuration:
 *      - HC: Rx and Tx
 *      - CDU context validation
 *
 */
static int ecore_q_init(struct bnx2x_softc *sc,
			struct ecore_queue_state_params *params)
{
	struct ecore_queue_sp_obj *o = params->q_obj;
	struct ecore_queue_init_params *init = &params->params.init;
	uint16_t hc_usec;
	uint8_t cos;

	/* Tx HC configuration */
	if (ECORE_TEST_BIT(ECORE_Q_TYPE_HAS_TX, &o->type) &&
	    ECORE_TEST_BIT(ECORE_Q_FLG_HC, &init->tx.flags)) {
		hc_usec = init->tx.hc_rate ? 1000000 / init->tx.hc_rate : 0;

		ECORE_UPDATE_COALESCE_SB_INDEX(sc, init->tx.fw_sb_id,
					       init->tx.sb_cq_index,
					       !ECORE_TEST_BIT
					       (ECORE_Q_FLG_HC_EN,
						&init->tx.flags), hc_usec);
	}

	/* Rx HC configuration */
	if (ECORE_TEST_BIT(ECORE_Q_TYPE_HAS_RX, &o->type) &&
	    ECORE_TEST_BIT(ECORE_Q_FLG_HC, &init->rx.flags)) {
		hc_usec = init->rx.hc_rate ? 1000000 / init->rx.hc_rate : 0;

		ECORE_UPDATE_COALESCE_SB_INDEX(sc, init->rx.fw_sb_id,
					       init->rx.sb_cq_index,
					       !ECORE_TEST_BIT
					       (ECORE_Q_FLG_HC_EN,
						&init->rx.flags), hc_usec);
	}

	/* Set CDU context validation values */
	for (cos = 0; cos < o->max_cos; cos++) {
		ECORE_MSG(sc, "setting context validation. cid %d, cos %d",
			  o->cids[cos], cos);
		ECORE_MSG(sc, "context pointer %p", init->cxts[cos]);
		ECORE_SET_CTX_VALIDATION(sc, init->cxts[cos], o->cids[cos]);
	}

	/* As no ramrod is sent, complete the command immediately  */
	o->complete_cmd(sc, o, ECORE_Q_CMD_INIT);

	ECORE_MMIOWB();
	ECORE_SMP_MB();

	return ECORE_SUCCESS;
}

static int ecore_q_send_setup_e1x(struct bnx2x_softc *sc, struct ecore_queue_state_params
				  *params)
{
	struct ecore_queue_sp_obj *o = params->q_obj;
	struct client_init_ramrod_data *rdata =
	    (struct client_init_ramrod_data *)o->rdata;
	ecore_dma_addr_t data_mapping = o->rdata_mapping;
	int ramrod = RAMROD_CMD_ID_ETH_CLIENT_SETUP;

	/* Clear the ramrod data */
	ECORE_MEMSET(rdata, 0, sizeof(*rdata));

	/* Fill the ramrod data */
	ecore_q_fill_setup_data_cmn(sc, params, rdata);

	/* No need for an explicit memory barrier here as long we would
	 * need to ensure the ordering of writing to the SPQ element
	 * and updating of the SPQ producer which ilwolves a memory
	 * read and we will have to put a full memory barrier there
	 * (inside ecore_sp_post()).
	 */

	return ecore_sp_post(sc,
			     ramrod,
			     o->cids[ECORE_PRIMARY_CID_INDEX],
			     data_mapping, ETH_CONNECTION_TYPE);
}

static int ecore_q_send_setup_e2(struct bnx2x_softc *sc,
				 struct ecore_queue_state_params *params)
{
	struct ecore_queue_sp_obj *o = params->q_obj;
	struct client_init_ramrod_data *rdata =
	    (struct client_init_ramrod_data *)o->rdata;
	ecore_dma_addr_t data_mapping = o->rdata_mapping;
	int ramrod = RAMROD_CMD_ID_ETH_CLIENT_SETUP;

	/* Clear the ramrod data */
	ECORE_MEMSET(rdata, 0, sizeof(*rdata));

	/* Fill the ramrod data */
	ecore_q_fill_setup_data_cmn(sc, params, rdata);
	ecore_q_fill_setup_data_e2(params, rdata);

	/* No need for an explicit memory barrier here as long we would
	 * need to ensure the ordering of writing to the SPQ element
	 * and updating of the SPQ producer which ilwolves a memory
	 * read and we will have to put a full memory barrier there
	 * (inside ecore_sp_post()).
	 */

	return ecore_sp_post(sc,
			     ramrod,
			     o->cids[ECORE_PRIMARY_CID_INDEX],
			     data_mapping, ETH_CONNECTION_TYPE);
}

static int ecore_q_send_setup_tx_only(struct bnx2x_softc *sc, struct ecore_queue_state_params
				      *params)
{
	struct ecore_queue_sp_obj *o = params->q_obj;
	struct tx_queue_init_ramrod_data *rdata =
	    (struct tx_queue_init_ramrod_data *)o->rdata;
	ecore_dma_addr_t data_mapping = o->rdata_mapping;
	int ramrod = RAMROD_CMD_ID_ETH_TX_QUEUE_SETUP;
	struct ecore_queue_setup_tx_only_params *tx_only_params =
	    &params->params.tx_only;
	uint8_t cid_index = tx_only_params->cid_index;

	if (ECORE_TEST_BIT(ECORE_Q_TYPE_FWD, &o->type))
		ramrod = RAMROD_CMD_ID_ETH_FORWARD_SETUP;
	ECORE_MSG(sc, "sending forward tx-only ramrod");

	if (cid_index >= o->max_cos) {
		PMD_DRV_LOG(ERR, sc, "queue[%d]: cid_index (%d) is out of range",
			    o->cl_id, cid_index);
		return ECORE_ILWAL;
	}

	ECORE_MSG(sc, "parameters received: cos: %d sp-id: %d",
		  tx_only_params->gen_params.cos,
		  tx_only_params->gen_params.spcl_id);

	/* Clear the ramrod data */
	ECORE_MEMSET(rdata, 0, sizeof(*rdata));

	/* Fill the ramrod data */
	ecore_q_fill_setup_tx_only(sc, params, rdata);

	    ECORE_MSG
	    (sc, "sending tx-only ramrod: cid %d, client-id %d, sp-client id %d, cos %d",
	     o->cids[cid_index], rdata->general.client_id,
	     rdata->general.sp_client_id, rdata->general.cos);

	/* No need for an explicit memory barrier here as long we would
	 * need to ensure the ordering of writing to the SPQ element
	 * and updating of the SPQ producer which ilwolves a memory
	 * read and we will have to put a full memory barrier there
	 * (inside ecore_sp_post()).
	 */

	return ecore_sp_post(sc, ramrod, o->cids[cid_index],
			     data_mapping, ETH_CONNECTION_TYPE);
}

static void ecore_q_fill_update_data(struct ecore_queue_sp_obj *obj,
				     struct ecore_queue_update_params *params,
				     struct client_update_ramrod_data *data)
{
	/* Client ID of the client to update */
	data->client_id = obj->cl_id;

	/* Function ID of the client to update */
	data->func_id = obj->func_id;

	/* Default VLAN value */
	data->default_vlan = ECORE_CPU_TO_LE16(params->def_vlan);

	/* Inner VLAN stripping */
	data->inner_vlan_removal_enable_flg =
	    ECORE_TEST_BIT(ECORE_Q_UPDATE_IN_VLAN_REM, &params->update_flags);
	data->inner_vlan_removal_change_flg =
	    ECORE_TEST_BIT(ECORE_Q_UPDATE_IN_VLAN_REM_CHNG,
			   &params->update_flags);

	/* Outer VLAN stripping */
	data->outer_vlan_removal_enable_flg =
	    ECORE_TEST_BIT(ECORE_Q_UPDATE_OUT_VLAN_REM, &params->update_flags);
	data->outer_vlan_removal_change_flg =
	    ECORE_TEST_BIT(ECORE_Q_UPDATE_OUT_VLAN_REM_CHNG,
			   &params->update_flags);

	/* Drop packets that have source MAC that doesn't belong to this
	 * Queue.
	 */
	data->anti_spoofing_enable_flg =
	    ECORE_TEST_BIT(ECORE_Q_UPDATE_ANTI_SPOOF, &params->update_flags);
	data->anti_spoofing_change_flg =
	    ECORE_TEST_BIT(ECORE_Q_UPDATE_ANTI_SPOOF_CHNG,
			   &params->update_flags);

	/* Activate/Deactivate */
	data->activate_flg =
	    ECORE_TEST_BIT(ECORE_Q_UPDATE_ACTIVATE, &params->update_flags);
	data->activate_change_flg =
	    ECORE_TEST_BIT(ECORE_Q_UPDATE_ACTIVATE_CHNG, &params->update_flags);

	/* Enable default VLAN */
	data->default_vlan_enable_flg =
	    ECORE_TEST_BIT(ECORE_Q_UPDATE_DEF_VLAN_EN, &params->update_flags);
	data->default_vlan_change_flg =
	    ECORE_TEST_BIT(ECORE_Q_UPDATE_DEF_VLAN_EN_CHNG,
			   &params->update_flags);

	/* silent vlan removal */
	data->silent_vlan_change_flg =
	    ECORE_TEST_BIT(ECORE_Q_UPDATE_SILENT_VLAN_REM_CHNG,
			   &params->update_flags);
	data->silent_vlan_removal_flg =
	    ECORE_TEST_BIT(ECORE_Q_UPDATE_SILENT_VLAN_REM,
			   &params->update_flags);
	data->silent_vlan_value =
	    ECORE_CPU_TO_LE16(params->silent_removal_value);
	data->silent_vlan_mask = ECORE_CPU_TO_LE16(params->silent_removal_mask);

	/* tx switching */
	data->tx_switching_flg =
	    ECORE_TEST_BIT(ECORE_Q_UPDATE_TX_SWITCHING, &params->update_flags);
	data->tx_switching_change_flg =
	    ECORE_TEST_BIT(ECORE_Q_UPDATE_TX_SWITCHING_CHNG,
			   &params->update_flags);
}

static int ecore_q_send_update(struct bnx2x_softc *sc,
			       struct ecore_queue_state_params *params)
{
	struct ecore_queue_sp_obj *o = params->q_obj;
	struct client_update_ramrod_data *rdata =
	    (struct client_update_ramrod_data *)o->rdata;
	ecore_dma_addr_t data_mapping = o->rdata_mapping;
	struct ecore_queue_update_params *update_params =
	    &params->params.update;
	uint8_t cid_index = update_params->cid_index;

	if (cid_index >= o->max_cos) {
		PMD_DRV_LOG(ERR, sc, "queue[%d]: cid_index (%d) is out of range",
			    o->cl_id, cid_index);
		return ECORE_ILWAL;
	}

	/* Clear the ramrod data */
	ECORE_MEMSET(rdata, 0, sizeof(*rdata));

	/* Fill the ramrod data */
	ecore_q_fill_update_data(o, update_params, rdata);

	/* No need for an explicit memory barrier here as long we would
	 * need to ensure the ordering of writing to the SPQ element
	 * and updating of the SPQ producer which ilwolves a memory
	 * read and we will have to put a full memory barrier there
	 * (inside ecore_sp_post()).
	 */

	return ecore_sp_post(sc, RAMROD_CMD_ID_ETH_CLIENT_UPDATE,
			     o->cids[cid_index], data_mapping,
			     ETH_CONNECTION_TYPE);
}

/**
 * ecore_q_send_deactivate - send DEACTIVATE command
 *
 * @sc:		device handle
 * @params:
 *
 * implemented using the UPDATE command.
 */
static int ecore_q_send_deactivate(struct bnx2x_softc *sc, struct ecore_queue_state_params
				   *params)
{
	struct ecore_queue_update_params *update = &params->params.update;

	ECORE_MEMSET(update, 0, sizeof(*update));

	ECORE_SET_BIT_NA(ECORE_Q_UPDATE_ACTIVATE_CHNG, &update->update_flags);

	return ecore_q_send_update(sc, params);
}

/**
 * ecore_q_send_activate - send ACTIVATE command
 *
 * @sc:		device handle
 * @params:
 *
 * implemented using the UPDATE command.
 */
static int ecore_q_send_activate(struct bnx2x_softc *sc,
				 struct ecore_queue_state_params *params)
{
	struct ecore_queue_update_params *update = &params->params.update;

	ECORE_MEMSET(update, 0, sizeof(*update));

	ECORE_SET_BIT_NA(ECORE_Q_UPDATE_ACTIVATE, &update->update_flags);
	ECORE_SET_BIT_NA(ECORE_Q_UPDATE_ACTIVATE_CHNG, &update->update_flags);

	return ecore_q_send_update(sc, params);
}

static int ecore_q_send_update_tpa(__rte_unused struct bnx2x_softc *sc,
				   __rte_unused struct
				   ecore_queue_state_params *params)
{
	/* Not implemented yet. */
	return -1;
}

static int ecore_q_send_halt(struct bnx2x_softc *sc,
			     struct ecore_queue_state_params *params)
{
	struct ecore_queue_sp_obj *o = params->q_obj;

	/* build eth_halt_ramrod_data.client_id in a big-endian friendly way */
	ecore_dma_addr_t data_mapping = 0;
	data_mapping = (ecore_dma_addr_t) o->cl_id;

	return ecore_sp_post(sc,
			     RAMROD_CMD_ID_ETH_HALT,
			     o->cids[ECORE_PRIMARY_CID_INDEX],
			     data_mapping, ETH_CONNECTION_TYPE);
}

static int ecore_q_send_cfc_del(struct bnx2x_softc *sc,
				struct ecore_queue_state_params *params)
{
	struct ecore_queue_sp_obj *o = params->q_obj;
	uint8_t cid_idx = params->params.cfc_del.cid_index;

	if (cid_idx >= o->max_cos) {
		PMD_DRV_LOG(ERR, sc, "queue[%d]: cid_index (%d) is out of range",
			    o->cl_id, cid_idx);
		return ECORE_ILWAL;
	}

	return ecore_sp_post(sc, RAMROD_CMD_ID_COMMON_CFC_DEL,
			     o->cids[cid_idx], 0, NONE_CONNECTION_TYPE);
}

static int ecore_q_send_terminate(struct bnx2x_softc *sc, struct ecore_queue_state_params
				  *params)
{
	struct ecore_queue_sp_obj *o = params->q_obj;
	uint8_t cid_index = params->params.terminate.cid_index;

	if (cid_index >= o->max_cos) {
		PMD_DRV_LOG(ERR, sc, "queue[%d]: cid_index (%d) is out of range",
			    o->cl_id, cid_index);
		return ECORE_ILWAL;
	}

	return ecore_sp_post(sc, RAMROD_CMD_ID_ETH_TERMINATE,
			     o->cids[cid_index], 0, ETH_CONNECTION_TYPE);
}

static int ecore_q_send_empty(struct bnx2x_softc *sc,
			      struct ecore_queue_state_params *params)
{
	struct ecore_queue_sp_obj *o = params->q_obj;

	return ecore_sp_post(sc, RAMROD_CMD_ID_ETH_EMPTY,
			     o->cids[ECORE_PRIMARY_CID_INDEX], 0,
			     ETH_CONNECTION_TYPE);
}

static int ecore_queue_send_cmd_cmn(struct bnx2x_softc *sc, struct ecore_queue_state_params
				    *params)
{
	switch (params->cmd) {
	case ECORE_Q_CMD_INIT:
		return ecore_q_init(sc, params);
	case ECORE_Q_CMD_SETUP_TX_ONLY:
		return ecore_q_send_setup_tx_only(sc, params);
	case ECORE_Q_CMD_DEACTIVATE:
		return ecore_q_send_deactivate(sc, params);
	case ECORE_Q_CMD_ACTIVATE:
		return ecore_q_send_activate(sc, params);
	case ECORE_Q_CMD_UPDATE:
		return ecore_q_send_update(sc, params);
	case ECORE_Q_CMD_UPDATE_TPA:
		return ecore_q_send_update_tpa(sc, params);
	case ECORE_Q_CMD_HALT:
		return ecore_q_send_halt(sc, params);
	case ECORE_Q_CMD_CFC_DEL:
		return ecore_q_send_cfc_del(sc, params);
	case ECORE_Q_CMD_TERMINATE:
		return ecore_q_send_terminate(sc, params);
	case ECORE_Q_CMD_EMPTY:
		return ecore_q_send_empty(sc, params);
	default:
		PMD_DRV_LOG(ERR, sc, "Unknown command: %d", params->cmd);
		return ECORE_ILWAL;
	}
}

static int ecore_queue_send_cmd_e1x(struct bnx2x_softc *sc,
				    struct ecore_queue_state_params *params)
{
	switch (params->cmd) {
	case ECORE_Q_CMD_SETUP:
		return ecore_q_send_setup_e1x(sc, params);
	case ECORE_Q_CMD_INIT:
	case ECORE_Q_CMD_SETUP_TX_ONLY:
	case ECORE_Q_CMD_DEACTIVATE:
	case ECORE_Q_CMD_ACTIVATE:
	case ECORE_Q_CMD_UPDATE:
	case ECORE_Q_CMD_UPDATE_TPA:
	case ECORE_Q_CMD_HALT:
	case ECORE_Q_CMD_CFC_DEL:
	case ECORE_Q_CMD_TERMINATE:
	case ECORE_Q_CMD_EMPTY:
		return ecore_queue_send_cmd_cmn(sc, params);
	default:
		PMD_DRV_LOG(ERR, sc, "Unknown command: %d", params->cmd);
		return ECORE_ILWAL;
	}
}

static int ecore_queue_send_cmd_e2(struct bnx2x_softc *sc,
				   struct ecore_queue_state_params *params)
{
	switch (params->cmd) {
	case ECORE_Q_CMD_SETUP:
		return ecore_q_send_setup_e2(sc, params);
	case ECORE_Q_CMD_INIT:
	case ECORE_Q_CMD_SETUP_TX_ONLY:
	case ECORE_Q_CMD_DEACTIVATE:
	case ECORE_Q_CMD_ACTIVATE:
	case ECORE_Q_CMD_UPDATE:
	case ECORE_Q_CMD_UPDATE_TPA:
	case ECORE_Q_CMD_HALT:
	case ECORE_Q_CMD_CFC_DEL:
	case ECORE_Q_CMD_TERMINATE:
	case ECORE_Q_CMD_EMPTY:
		return ecore_queue_send_cmd_cmn(sc, params);
	default:
		PMD_DRV_LOG(ERR, sc, "Unknown command: %d", params->cmd);
		return ECORE_ILWAL;
	}
}

/**
 * ecore_queue_chk_transition - check state machine of a regular Queue
 *
 * @sc:		device handle
 * @o:
 * @params:
 *
 * (not Forwarding)
 * It both checks if the requested command is legal in a current
 * state and, if it's legal, sets a `next_state' in the object
 * that will be used in the completion flow to set the `state'
 * of the object.
 *
 * returns 0 if a requested command is a legal transition,
 *         ECORE_ILWAL otherwise.
 */
static int ecore_queue_chk_transition(struct bnx2x_softc *sc __rte_unused,
				      struct ecore_queue_sp_obj *o,
				      struct ecore_queue_state_params *params)
{
	enum ecore_q_state state = o->state, next_state = ECORE_Q_STATE_MAX;
	enum ecore_queue_cmd cmd = params->cmd;
	struct ecore_queue_update_params *update_params =
	    &params->params.update;
	uint8_t next_tx_only = o->num_tx_only;

	/* Forget all pending for completion commands if a driver only state
	 * transition has been requested.
	 */
	if (ECORE_TEST_BIT(RAMROD_DRV_CLR_ONLY, &params->ramrod_flags)) {
		o->pending = 0;
		o->next_state = ECORE_Q_STATE_MAX;
	}

	/* Don't allow a next state transition if we are in the middle of
	 * the previous one.
	 */
	if (o->pending) {
		PMD_DRV_LOG(ERR, sc, "Blocking transition since pending was %x",
			    o->pending);
		return ECORE_BUSY;
	}

	switch (state) {
	case ECORE_Q_STATE_RESET:
		if (cmd == ECORE_Q_CMD_INIT)
			next_state = ECORE_Q_STATE_INITIALIZED;

		break;
	case ECORE_Q_STATE_INITIALIZED:
		if (cmd == ECORE_Q_CMD_SETUP) {
			if (ECORE_TEST_BIT(ECORE_Q_FLG_ACTIVE,
					   &params->params.setup.flags))
				next_state = ECORE_Q_STATE_ACTIVE;
			else
				next_state = ECORE_Q_STATE_INACTIVE;
		}

		break;
	case ECORE_Q_STATE_ACTIVE:
		if (cmd == ECORE_Q_CMD_DEACTIVATE)
			next_state = ECORE_Q_STATE_INACTIVE;

		else if ((cmd == ECORE_Q_CMD_EMPTY) ||
			 (cmd == ECORE_Q_CMD_UPDATE_TPA))
			next_state = ECORE_Q_STATE_ACTIVE;

		else if (cmd == ECORE_Q_CMD_SETUP_TX_ONLY) {
			next_state = ECORE_Q_STATE_MULTI_COS;
			next_tx_only = 1;
		}

		else if (cmd == ECORE_Q_CMD_HALT)
			next_state = ECORE_Q_STATE_STOPPED;

		else if (cmd == ECORE_Q_CMD_UPDATE) {
			/* If "active" state change is requested, update the
			 *  state accordingly.
			 */
			if (ECORE_TEST_BIT(ECORE_Q_UPDATE_ACTIVATE_CHNG,
					   &update_params->update_flags) &&
			    !ECORE_TEST_BIT(ECORE_Q_UPDATE_ACTIVATE,
					    &update_params->update_flags))
				next_state = ECORE_Q_STATE_INACTIVE;
			else
				next_state = ECORE_Q_STATE_ACTIVE;
		}

		break;
	case ECORE_Q_STATE_MULTI_COS:
		if (cmd == ECORE_Q_CMD_TERMINATE)
			next_state = ECORE_Q_STATE_MCOS_TERMINATED;

		else if (cmd == ECORE_Q_CMD_SETUP_TX_ONLY) {
			next_state = ECORE_Q_STATE_MULTI_COS;
			next_tx_only = o->num_tx_only + 1;
		}

		else if ((cmd == ECORE_Q_CMD_EMPTY) ||
			 (cmd == ECORE_Q_CMD_UPDATE_TPA))
			next_state = ECORE_Q_STATE_MULTI_COS;

		else if (cmd == ECORE_Q_CMD_UPDATE) {
			/* If "active" state change is requested, update the
			 *  state accordingly.
			 */
			if (ECORE_TEST_BIT(ECORE_Q_UPDATE_ACTIVATE_CHNG,
					   &update_params->update_flags) &&
			    !ECORE_TEST_BIT(ECORE_Q_UPDATE_ACTIVATE,
					    &update_params->update_flags))
				next_state = ECORE_Q_STATE_INACTIVE;
			else
				next_state = ECORE_Q_STATE_MULTI_COS;
		}

		break;
	case ECORE_Q_STATE_MCOS_TERMINATED:
		if (cmd == ECORE_Q_CMD_CFC_DEL) {
			next_tx_only = o->num_tx_only - 1;
			if (next_tx_only == 0)
				next_state = ECORE_Q_STATE_ACTIVE;
			else
				next_state = ECORE_Q_STATE_MULTI_COS;
		}

		break;
	case ECORE_Q_STATE_INACTIVE:
		if (cmd == ECORE_Q_CMD_ACTIVATE)
			next_state = ECORE_Q_STATE_ACTIVE;

		else if ((cmd == ECORE_Q_CMD_EMPTY) ||
			 (cmd == ECORE_Q_CMD_UPDATE_TPA))
			next_state = ECORE_Q_STATE_INACTIVE;

		else if (cmd == ECORE_Q_CMD_HALT)
			next_state = ECORE_Q_STATE_STOPPED;

		else if (cmd == ECORE_Q_CMD_UPDATE) {
			/* If "active" state change is requested, update the
			 * state accordingly.
			 */
			if (ECORE_TEST_BIT(ECORE_Q_UPDATE_ACTIVATE_CHNG,
					   &update_params->update_flags) &&
			    ECORE_TEST_BIT(ECORE_Q_UPDATE_ACTIVATE,
					   &update_params->update_flags)) {
				if (o->num_tx_only == 0)
					next_state = ECORE_Q_STATE_ACTIVE;
				else	/* tx only queues exist for this queue */
					next_state = ECORE_Q_STATE_MULTI_COS;
			} else
				next_state = ECORE_Q_STATE_INACTIVE;
		}

		break;
	case ECORE_Q_STATE_STOPPED:
		if (cmd == ECORE_Q_CMD_TERMINATE)
			next_state = ECORE_Q_STATE_TERMINATED;

		break;
	case ECORE_Q_STATE_TERMINATED:
		if (cmd == ECORE_Q_CMD_CFC_DEL)
			next_state = ECORE_Q_STATE_RESET;

		break;
	default:
		PMD_DRV_LOG(ERR, sc, "Illegal state: %d", state);
	}

	/* Transition is assured */
	if (next_state != ECORE_Q_STATE_MAX) {
		ECORE_MSG(sc, "Good state transition: %d(%d)->%d",
			  state, cmd, next_state);
		o->next_state = next_state;
		o->next_tx_only = next_tx_only;
		return ECORE_SUCCESS;
	}

	ECORE_MSG(sc, "Bad state transition request: %d %d", state, cmd);

	return ECORE_ILWAL;
}

/**
 * ecore_queue_chk_fwd_transition - check state machine of a Forwarding Queue.
 *
 * @sc:		device handle
 * @o:
 * @params:
 *
 * It both checks if the requested command is legal in a current
 * state and, if it's legal, sets a `next_state' in the object
 * that will be used in the completion flow to set the `state'
 * of the object.
 *
 * returns 0 if a requested command is a legal transition,
 *         ECORE_ILWAL otherwise.
 */
static int ecore_queue_chk_fwd_transition(struct bnx2x_softc *sc __rte_unused,
					  struct ecore_queue_sp_obj *o,
					  struct ecore_queue_state_params
					  *params)
{
	enum ecore_q_state state = o->state, next_state = ECORE_Q_STATE_MAX;
	enum ecore_queue_cmd cmd = params->cmd;

	switch (state) {
	case ECORE_Q_STATE_RESET:
		if (cmd == ECORE_Q_CMD_INIT)
			next_state = ECORE_Q_STATE_INITIALIZED;

		break;
	case ECORE_Q_STATE_INITIALIZED:
		if (cmd == ECORE_Q_CMD_SETUP_TX_ONLY) {
			if (ECORE_TEST_BIT(ECORE_Q_FLG_ACTIVE,
					   &params->params.tx_only.flags))
				next_state = ECORE_Q_STATE_ACTIVE;
			else
				next_state = ECORE_Q_STATE_INACTIVE;
		}

		break;
	case ECORE_Q_STATE_ACTIVE:
	case ECORE_Q_STATE_INACTIVE:
		if (cmd == ECORE_Q_CMD_CFC_DEL)
			next_state = ECORE_Q_STATE_RESET;

		break;
	default:
		PMD_DRV_LOG(ERR, sc, "Illegal state: %d", state);
	}

	/* Transition is assured */
	if (next_state != ECORE_Q_STATE_MAX) {
		ECORE_MSG(sc, "Good state transition: %d(%d)->%d",
			  state, cmd, next_state);
		o->next_state = next_state;
		return ECORE_SUCCESS;
	}

	ECORE_MSG(sc, "Bad state transition request: %d %d", state, cmd);
	return ECORE_ILWAL;
}

void ecore_init_queue_obj(struct bnx2x_softc *sc,
			  struct ecore_queue_sp_obj *obj,
			  uint8_t cl_id, uint32_t * cids, uint8_t cid_cnt,
			  uint8_t func_id, void *rdata,
			  ecore_dma_addr_t rdata_mapping, uint32_t type)
{
	ECORE_MEMSET(obj, 0, sizeof(*obj));

	/* We support only ECORE_MULTI_TX_COS Tx CoS at the moment */
	ECORE_BUG_ON(ECORE_MULTI_TX_COS < cid_cnt);

	rte_memcpy(obj->cids, cids, sizeof(obj->cids[0]) * cid_cnt);
	obj->max_cos = cid_cnt;
	obj->cl_id = cl_id;
	obj->func_id = func_id;
	obj->rdata = rdata;
	obj->rdata_mapping = rdata_mapping;
	obj->type = type;
	obj->next_state = ECORE_Q_STATE_MAX;

	if (CHIP_IS_E1x(sc))
		obj->send_cmd = ecore_queue_send_cmd_e1x;
	else
		obj->send_cmd = ecore_queue_send_cmd_e2;

	if (ECORE_TEST_BIT(ECORE_Q_TYPE_FWD, &type))
		obj->check_transition = ecore_queue_chk_fwd_transition;
	else
		obj->check_transition = ecore_queue_chk_transition;

	obj->complete_cmd = ecore_queue_comp_cmd;
	obj->wait_comp = ecore_queue_wait_comp;
	obj->set_pending = ecore_queue_set_pending;
}

/********************** Function state object *********************************/
enum ecore_func_state ecore_func_get_state(__rte_unused struct bnx2x_softc *sc,
					   struct ecore_func_sp_obj *o)
{
	/* in the middle of transaction - return INVALID state */
	if (o->pending)
		return ECORE_F_STATE_MAX;

	/* unsure the order of reading of o->pending and o->state
	 * o->pending should be read first
	 */
	rmb();

	return o->state;
}

static int ecore_func_wait_comp(struct bnx2x_softc *sc,
				struct ecore_func_sp_obj *o,
				enum ecore_func_cmd cmd)
{
	return ecore_state_wait(sc, cmd, &o->pending);
}

/**
 * ecore_func_state_change_comp - complete the state machine transition
 *
 * @sc:		device handle
 * @o:
 * @cmd:
 *
 * Called on state change transition. Completes the state
 * machine transition only - no HW interaction.
 */
static int
ecore_func_state_change_comp(struct bnx2x_softc *sc __rte_unused,
			     struct ecore_func_sp_obj *o,
			     enum ecore_func_cmd cmd)
{
	uint32_t lwr_pending = o->pending;

	if (!ECORE_TEST_AND_CLEAR_BIT(cmd, &lwr_pending)) {
		PMD_DRV_LOG(ERR, sc,
			    "Bad MC reply %d for func %d in state %d pending 0x%x, next_state %d",
			    cmd, ECORE_FUNC_ID(sc), o->state, lwr_pending,
			    o->next_state);
		return ECORE_ILWAL;
	}

	ECORE_MSG(sc, "Completing command %d for func %d, setting state to %d",
		  cmd, ECORE_FUNC_ID(sc), o->next_state);

	o->state = o->next_state;
	o->next_state = ECORE_F_STATE_MAX;

	/* It's important that o->state and o->next_state are
	 * updated before o->pending.
	 */
	wmb();

	ECORE_CLEAR_BIT(cmd, &o->pending);
	ECORE_SMP_MB_AFTER_CLEAR_BIT();

	return ECORE_SUCCESS;
}

/**
 * ecore_func_comp_cmd - complete the state change command
 *
 * @sc:		device handle
 * @o:
 * @cmd:
 *
 * Checks that the arrived completion is expected.
 */
static int ecore_func_comp_cmd(struct bnx2x_softc *sc,
			       struct ecore_func_sp_obj *o,
			       enum ecore_func_cmd cmd)
{
	/* Complete the state machine part first, check if it's a
	 * legal completion.
	 */
	int rc = ecore_func_state_change_comp(sc, o, cmd);
	return rc;
}

/**
 * ecore_func_chk_transition - perform function state machine transition
 *
 * @sc:		device handle
 * @o:
 * @params:
 *
 * It both checks if the requested command is legal in a current
 * state and, if it's legal, sets a `next_state' in the object
 * that will be used in the completion flow to set the `state'
 * of the object.
 *
 * returns 0 if a requested command is a legal transition,
 *         ECORE_ILWAL otherwise.
 */
static int ecore_func_chk_transition(struct bnx2x_softc *sc __rte_unused,
				     struct ecore_func_sp_obj *o,
				     struct ecore_func_state_params *params)
{
	enum ecore_func_state state = o->state, next_state = ECORE_F_STATE_MAX;
	enum ecore_func_cmd cmd = params->cmd;

	/* Forget all pending for completion commands if a driver only state
	 * transition has been requested.
	 */
	if (ECORE_TEST_BIT(RAMROD_DRV_CLR_ONLY, &params->ramrod_flags)) {
		o->pending = 0;
		o->next_state = ECORE_F_STATE_MAX;
	}

	/* Don't allow a next state transition if we are in the middle of
	 * the previous one.
	 */
	if (o->pending)
		return ECORE_BUSY;

	switch (state) {
	case ECORE_F_STATE_RESET:
		if (cmd == ECORE_F_CMD_HW_INIT)
			next_state = ECORE_F_STATE_INITIALIZED;

		break;
	case ECORE_F_STATE_INITIALIZED:
		if (cmd == ECORE_F_CMD_START)
			next_state = ECORE_F_STATE_STARTED;

		else if (cmd == ECORE_F_CMD_HW_RESET)
			next_state = ECORE_F_STATE_RESET;

		break;
	case ECORE_F_STATE_STARTED:
		if (cmd == ECORE_F_CMD_STOP)
			next_state = ECORE_F_STATE_INITIALIZED;
		/* afex ramrods can be sent only in started mode, and only
		 * if not pending for function_stop ramrod completion
		 * for these events - next state remained STARTED.
		 */
		else if ((cmd == ECORE_F_CMD_AFEX_UPDATE) &&
			 (!ECORE_TEST_BIT(ECORE_F_CMD_STOP, &o->pending)))
			next_state = ECORE_F_STATE_STARTED;

		else if ((cmd == ECORE_F_CMD_AFEX_VIFLISTS) &&
			 (!ECORE_TEST_BIT(ECORE_F_CMD_STOP, &o->pending)))
			next_state = ECORE_F_STATE_STARTED;

		/* Switch_update ramrod can be sent in either started or
		 * tx_stopped state, and it doesn't change the state.
		 */
		else if ((cmd == ECORE_F_CMD_SWITCH_UPDATE) &&
			 (!ECORE_TEST_BIT(ECORE_F_CMD_STOP, &o->pending)))
			next_state = ECORE_F_STATE_STARTED;

		else if (cmd == ECORE_F_CMD_TX_STOP)
			next_state = ECORE_F_STATE_TX_STOPPED;

		break;
	case ECORE_F_STATE_TX_STOPPED:
		if ((cmd == ECORE_F_CMD_SWITCH_UPDATE) &&
		    (!ECORE_TEST_BIT(ECORE_F_CMD_STOP, &o->pending)))
			next_state = ECORE_F_STATE_TX_STOPPED;

		else if (cmd == ECORE_F_CMD_TX_START)
			next_state = ECORE_F_STATE_STARTED;

		break;
	default:
		PMD_DRV_LOG(ERR, sc, "Unknown state: %d", state);
	}

	/* Transition is assured */
	if (next_state != ECORE_F_STATE_MAX) {
		ECORE_MSG(sc, "Good function state transition: %d(%d)->%d",
			  state, cmd, next_state);
		o->next_state = next_state;
		return ECORE_SUCCESS;
	}

	ECORE_MSG(sc,
		  "Bad function state transition request: %d %d", state, cmd);

	return ECORE_ILWAL;
}

/**
 * ecore_func_init_func - performs HW init at function stage
 *
 * @sc:		device handle
 * @drv:
 *
 * Init HW when the current phase is
 * FW_MSG_CODE_DRV_LOAD_FUNCTION: initialize only FUNCTION-only
 * HW blocks.
 */
static int ecore_func_init_func(struct bnx2x_softc *sc,
				const struct ecore_func_sp_drv_ops *drv)
{
	return drv->init_hw_func(sc);
}

/**
 * ecore_func_init_port - performs HW init at port stage
 *
 * @sc:		device handle
 * @drv:
 *
 * Init HW when the current phase is
 * FW_MSG_CODE_DRV_LOAD_PORT: initialize PORT-only and
 * FUNCTION-only HW blocks.
 *
 */
static int ecore_func_init_port(struct bnx2x_softc *sc,
				const struct ecore_func_sp_drv_ops *drv)
{
	int rc = drv->init_hw_port(sc);
	if (rc)
		return rc;

	return ecore_func_init_func(sc, drv);
}

/**
 * ecore_func_init_cmn_chip - performs HW init at chip-common stage
 *
 * @sc:		device handle
 * @drv:
 *
 * Init HW when the current phase is
 * FW_MSG_CODE_DRV_LOAD_COMMON_CHIP: initialize COMMON_CHIP,
 * PORT-only and FUNCTION-only HW blocks.
 */
static int ecore_func_init_cmn_chip(struct bnx2x_softc *sc, const struct ecore_func_sp_drv_ops
				    *drv)
{
	int rc = drv->init_hw_cmn_chip(sc);
	if (rc)
		return rc;

	return ecore_func_init_port(sc, drv);
}

/**
 * ecore_func_init_cmn - performs HW init at common stage
 *
 * @sc:		device handle
 * @drv:
 *
 * Init HW when the current phase is
 * FW_MSG_CODE_DRV_LOAD_COMMON_CHIP: initialize COMMON,
 * PORT-only and FUNCTION-only HW blocks.
 */
static int ecore_func_init_cmn(struct bnx2x_softc *sc,
			       const struct ecore_func_sp_drv_ops *drv)
{
	int rc = drv->init_hw_cmn(sc);
	if (rc)
		return rc;

	return ecore_func_init_port(sc, drv);
}

static int ecore_func_hw_init(struct bnx2x_softc *sc,
			      struct ecore_func_state_params *params)
{
	uint32_t load_code = params->params.hw_init.load_phase;
	struct ecore_func_sp_obj *o = params->f_obj;
	const struct ecore_func_sp_drv_ops *drv = o->drv;
	int rc = 0;

	ECORE_MSG(sc, "function %d  load_code %x",
		  ECORE_ABS_FUNC_ID(sc), load_code);

	/* Prepare FW */
	rc = drv->init_fw(sc);
	if (rc) {
		PMD_DRV_LOG(ERR, sc, "Error loading firmware");
		goto init_err;
	}

	/* Handle the beginning of COMMON_XXX pases separately... */
	switch (load_code) {
	case FW_MSG_CODE_DRV_LOAD_COMMON_CHIP:
		rc = ecore_func_init_cmn_chip(sc, drv);
		if (rc)
			goto init_err;

		break;
	case FW_MSG_CODE_DRV_LOAD_COMMON:
		rc = ecore_func_init_cmn(sc, drv);
		if (rc)
			goto init_err;

		break;
	case FW_MSG_CODE_DRV_LOAD_PORT:
		rc = ecore_func_init_port(sc, drv);
		if (rc)
			goto init_err;

		break;
	case FW_MSG_CODE_DRV_LOAD_FUNCTION:
		rc = ecore_func_init_func(sc, drv);
		if (rc)
			goto init_err;

		break;
	default:
		PMD_DRV_LOG(ERR, sc, "Unknown load_code (0x%x) from MCP",
			    load_code);
		rc = ECORE_ILWAL;
	}

init_err:
	/* In case of success, complete the command immediately: no ramrods
	 * have been sent.
	 */
	if (!rc)
		o->complete_cmd(sc, o, ECORE_F_CMD_HW_INIT);

	return rc;
}

/**
 * ecore_func_reset_func - reset HW at function stage
 *
 * @sc:		device handle
 * @drv:
 *
 * Reset HW at FW_MSG_CODE_DRV_UNLOAD_FUNCTION stage: reset only
 * FUNCTION-only HW blocks.
 */
static void ecore_func_reset_func(struct bnx2x_softc *sc, const struct ecore_func_sp_drv_ops
				  *drv)
{
	drv->reset_hw_func(sc);
}

/**
 * ecore_func_reset_port - reser HW at port stage
 *
 * @sc:		device handle
 * @drv:
 *
 * Reset HW at FW_MSG_CODE_DRV_UNLOAD_PORT stage: reset
 * FUNCTION-only and PORT-only HW blocks.
 *
 *                 !!!IMPORTANT!!!
 *
 * It's important to call reset_port before reset_func() as the last thing
 * reset_func does is pf_disable() thus disabling PGLUE_B, which
 * makes impossible any DMAE transactions.
 */
static void ecore_func_reset_port(struct bnx2x_softc *sc, const struct ecore_func_sp_drv_ops
				  *drv)
{
	drv->reset_hw_port(sc);
	ecore_func_reset_func(sc, drv);
}

/**
 * ecore_func_reset_cmn - reser HW at common stage
 *
 * @sc:		device handle
 * @drv:
 *
 * Reset HW at FW_MSG_CODE_DRV_UNLOAD_COMMON and
 * FW_MSG_CODE_DRV_UNLOAD_COMMON_CHIP stages: reset COMMON,
 * COMMON_CHIP, FUNCTION-only and PORT-only HW blocks.
 */
static void ecore_func_reset_cmn(struct bnx2x_softc *sc,
				 const struct ecore_func_sp_drv_ops *drv)
{
	ecore_func_reset_port(sc, drv);
	drv->reset_hw_cmn(sc);
}

static int ecore_func_hw_reset(struct bnx2x_softc *sc,
			       struct ecore_func_state_params *params)
{
	uint32_t reset_phase = params->params.hw_reset.reset_phase;
	struct ecore_func_sp_obj *o = params->f_obj;
	const struct ecore_func_sp_drv_ops *drv = o->drv;

	ECORE_MSG(sc, "function %d  reset_phase %x", ECORE_ABS_FUNC_ID(sc),
		  reset_phase);

	switch (reset_phase) {
	case FW_MSG_CODE_DRV_UNLOAD_COMMON:
		ecore_func_reset_cmn(sc, drv);
		break;
	case FW_MSG_CODE_DRV_UNLOAD_PORT:
		ecore_func_reset_port(sc, drv);
		break;
	case FW_MSG_CODE_DRV_UNLOAD_FUNCTION:
		ecore_func_reset_func(sc, drv);
		break;
	default:
		PMD_DRV_LOG(ERR, sc, "Unknown reset_phase (0x%x) from MCP",
			    reset_phase);
		break;
	}

	/* Complete the command immediately: no ramrods have been sent. */
	o->complete_cmd(sc, o, ECORE_F_CMD_HW_RESET);

	return ECORE_SUCCESS;
}

static int ecore_func_send_start(struct bnx2x_softc *sc,
				 struct ecore_func_state_params *params)
{
	struct ecore_func_sp_obj *o = params->f_obj;
	struct function_start_data *rdata =
	    (struct function_start_data *)o->rdata;
	ecore_dma_addr_t data_mapping = o->rdata_mapping;
	struct ecore_func_start_params *start_params = &params->params.start;

	ECORE_MEMSET(rdata, 0, sizeof(*rdata));

	/* Fill the ramrod data with provided parameters */
	rdata->function_mode = (uint8_t) start_params->mf_mode;
	rdata->sd_vlan_tag = ECORE_CPU_TO_LE16(start_params->sd_vlan_tag);
	rdata->path_id = ECORE_PATH_ID(sc);
	rdata->network_cos_mode = start_params->network_cos_mode;

	/*
	 *  No need for an explicit memory barrier here as long we would
	 *  need to ensure the ordering of writing to the SPQ element
	 *  and updating of the SPQ producer which ilwolves a memory
	 *  read and we will have to put a full memory barrier there
	 *  (inside ecore_sp_post()).
	 */

	return ecore_sp_post(sc, RAMROD_CMD_ID_COMMON_FUNCTION_START, 0,
			     data_mapping, NONE_CONNECTION_TYPE);
}

static int ecore_func_send_switch_update(struct bnx2x_softc *sc, struct ecore_func_state_params
					 *params)
{
	struct ecore_func_sp_obj *o = params->f_obj;
	struct function_update_data *rdata =
	    (struct function_update_data *)o->rdata;
	ecore_dma_addr_t data_mapping = o->rdata_mapping;
	struct ecore_func_switch_update_params *switch_update_params =
	    &params->params.switch_update;

	ECORE_MEMSET(rdata, 0, sizeof(*rdata));

	/* Fill the ramrod data with provided parameters */
	if (ECORE_TEST_BIT(ECORE_F_UPDATE_TX_SWITCH_SUSPEND_CHNG,
			   &switch_update_params->changes)) {
		rdata->tx_switch_suspend_change_flg = 1;
		rdata->tx_switch_suspend =
			ECORE_TEST_BIT(ECORE_F_UPDATE_TX_SWITCH_SUSPEND,
				       &switch_update_params->changes);
	}

	rdata->echo = SWITCH_UPDATE;

	return ecore_sp_post(sc, RAMROD_CMD_ID_COMMON_FUNCTION_UPDATE, 0,
			     data_mapping, NONE_CONNECTION_TYPE);
}

static int ecore_func_send_afex_update(struct bnx2x_softc *sc, struct ecore_func_state_params
				       *params)
{
	struct ecore_func_sp_obj *o = params->f_obj;
	struct function_update_data *rdata =
	    (struct function_update_data *)o->afex_rdata;
	ecore_dma_addr_t data_mapping = o->afex_rdata_mapping;
	struct ecore_func_afex_update_params *afex_update_params =
	    &params->params.afex_update;

	ECORE_MEMSET(rdata, 0, sizeof(*rdata));

	/* Fill the ramrod data with provided parameters */
	rdata->vif_id_change_flg = 1;
	rdata->vif_id = ECORE_CPU_TO_LE16(afex_update_params->vif_id);
	rdata->afex_default_vlan_change_flg = 1;
	rdata->afex_default_vlan =
	    ECORE_CPU_TO_LE16(afex_update_params->afex_default_vlan);
	rdata->allowed_priorities_change_flg = 1;
	rdata->allowed_priorities = afex_update_params->allowed_priorities;
	rdata->echo = AFEX_UPDATE;

	/*  No need for an explicit memory barrier here as long we would
	 *  need to ensure the ordering of writing to the SPQ element
	 *  and updating of the SPQ producer which ilwolves a memory
	 *  read and we will have to put a full memory barrier there
	 *  (inside ecore_sp_post()).
	 */
	ECORE_MSG(sc, "afex: sending func_update vif_id 0x%x dvlan 0x%x prio 0x%x",
		  rdata->vif_id,
		  rdata->afex_default_vlan, rdata->allowed_priorities);

	return ecore_sp_post(sc, RAMROD_CMD_ID_COMMON_FUNCTION_UPDATE, 0,
			     data_mapping, NONE_CONNECTION_TYPE);
}

static
inline int ecore_func_send_afex_viflists(struct bnx2x_softc *sc,
					 struct ecore_func_state_params *params)
{
	struct ecore_func_sp_obj *o = params->f_obj;
	struct afex_vif_list_ramrod_data *rdata =
	    (struct afex_vif_list_ramrod_data *)o->afex_rdata;
	struct ecore_func_afex_viflists_params *afex_vif_params =
	    &params->params.afex_viflists;
	uint64_t *p_rdata = (uint64_t *) rdata;

	ECORE_MEMSET(rdata, 0, sizeof(*rdata));

	/* Fill the ramrod data with provided parameters */
	rdata->vif_list_index =
	    ECORE_CPU_TO_LE16(afex_vif_params->vif_list_index);
	rdata->func_bit_map = afex_vif_params->func_bit_map;
	rdata->afex_vif_list_command = afex_vif_params->afex_vif_list_command;
	rdata->func_to_clear = afex_vif_params->func_to_clear;

	/* send in echo type of sub command */
	rdata->echo = afex_vif_params->afex_vif_list_command;

	/*  No need for an explicit memory barrier here as long we would
	 *  need to ensure the ordering of writing to the SPQ element
	 *  and updating of the SPQ producer which ilwolves a memory
	 *  read and we will have to put a full memory barrier there
	 *  (inside ecore_sp_post()).
	 */

	    ECORE_MSG
	    (sc, "afex: ramrod lists, cmd 0x%x index 0x%x func_bit_map 0x%x func_to_clr 0x%x",
	     rdata->afex_vif_list_command, rdata->vif_list_index,
	     rdata->func_bit_map, rdata->func_to_clear);

	/* this ramrod sends data directly and not through DMA mapping */
	return ecore_sp_post(sc, RAMROD_CMD_ID_COMMON_AFEX_VIF_LISTS, 0,
			     *p_rdata, NONE_CONNECTION_TYPE);
}

static int ecore_func_send_stop(struct bnx2x_softc *sc, __rte_unused struct
				ecore_func_state_params *params)
{
	return ecore_sp_post(sc, RAMROD_CMD_ID_COMMON_FUNCTION_STOP, 0, 0,
			     NONE_CONNECTION_TYPE);
}

static int ecore_func_send_tx_stop(struct bnx2x_softc *sc, __rte_unused struct
				   ecore_func_state_params *params)
{
	return ecore_sp_post(sc, RAMROD_CMD_ID_COMMON_STOP_TRAFFIC, 0, 0,
			     NONE_CONNECTION_TYPE);
}

static int ecore_func_send_tx_start(struct bnx2x_softc *sc, struct ecore_func_state_params
				    *params)
{
	struct ecore_func_sp_obj *o = params->f_obj;
	struct flow_control_configuration *rdata =
	    (struct flow_control_configuration *)o->rdata;
	ecore_dma_addr_t data_mapping = o->rdata_mapping;
	struct ecore_func_tx_start_params *tx_start_params =
	    &params->params.tx_start;
	uint32_t i;

	ECORE_MEMSET(rdata, 0, sizeof(*rdata));

	rdata->dcb_enabled = tx_start_params->dcb_enabled;
	rdata->dcb_version = tx_start_params->dcb_version;
	rdata->dont_add_pri_0_en = tx_start_params->dont_add_pri_0_en;

	for (i = 0; i < ARRAY_SIZE(rdata->traffic_type_to_priority_cos); i++)
		rdata->traffic_type_to_priority_cos[i] =
		    tx_start_params->traffic_type_to_priority_cos[i];

	return ecore_sp_post(sc, RAMROD_CMD_ID_COMMON_START_TRAFFIC, 0,
			     data_mapping, NONE_CONNECTION_TYPE);
}

static int ecore_func_send_cmd(struct bnx2x_softc *sc,
			       struct ecore_func_state_params *params)
{
	switch (params->cmd) {
	case ECORE_F_CMD_HW_INIT:
		return ecore_func_hw_init(sc, params);
	case ECORE_F_CMD_START:
		return ecore_func_send_start(sc, params);
	case ECORE_F_CMD_STOP:
		return ecore_func_send_stop(sc, params);
	case ECORE_F_CMD_HW_RESET:
		return ecore_func_hw_reset(sc, params);
	case ECORE_F_CMD_AFEX_UPDATE:
		return ecore_func_send_afex_update(sc, params);
	case ECORE_F_CMD_AFEX_VIFLISTS:
		return ecore_func_send_afex_viflists(sc, params);
	case ECORE_F_CMD_TX_STOP:
		return ecore_func_send_tx_stop(sc, params);
	case ECORE_F_CMD_TX_START:
		return ecore_func_send_tx_start(sc, params);
	case ECORE_F_CMD_SWITCH_UPDATE:
		return ecore_func_send_switch_update(sc, params);
	default:
		PMD_DRV_LOG(ERR, sc, "Unknown command: %d", params->cmd);
		return ECORE_ILWAL;
	}
}

void ecore_init_func_obj(__rte_unused struct bnx2x_softc *sc,
			 struct ecore_func_sp_obj *obj,
			 void *rdata, ecore_dma_addr_t rdata_mapping,
			 void *afex_rdata, ecore_dma_addr_t afex_rdata_mapping,
			 struct ecore_func_sp_drv_ops *drv_iface)
{
	ECORE_MEMSET(obj, 0, sizeof(*obj));

	ECORE_MUTEX_INIT(&obj->one_pending_mutex);

	obj->rdata = rdata;
	obj->rdata_mapping = rdata_mapping;
	obj->afex_rdata = afex_rdata;
	obj->afex_rdata_mapping = afex_rdata_mapping;
	obj->send_cmd = ecore_func_send_cmd;
	obj->check_transition = ecore_func_chk_transition;
	obj->complete_cmd = ecore_func_comp_cmd;
	obj->wait_comp = ecore_func_wait_comp;
	obj->drv = drv_iface;
}

/**
 * ecore_func_state_change - perform Function state change transition
 *
 * @sc:		device handle
 * @params:	parameters to perform the transaction
 *
 * returns 0 in case of successfully completed transition,
 *         negative error code in case of failure, positive
 *         (EBUSY) value if there is a completion to that is
 *         still pending (possible only if RAMROD_COMP_WAIT is
 *         not set in params->ramrod_flags for asynchronous
 *         commands).
 */
int ecore_func_state_change(struct bnx2x_softc *sc,
			    struct ecore_func_state_params *params)
{
	struct ecore_func_sp_obj *o = params->f_obj;
	int rc, cnt = 300;
	enum ecore_func_cmd cmd = params->cmd;
	uint32_t *pending = &o->pending;

	ECORE_MUTEX_LOCK(&o->one_pending_mutex);

	/* Check that the requested transition is legal */
	rc = o->check_transition(sc, o, params);
	if ((rc == ECORE_BUSY) &&
	    (ECORE_TEST_BIT(RAMROD_RETRY, &params->ramrod_flags))) {
		while ((rc == ECORE_BUSY) && (--cnt > 0)) {
			ECORE_MUTEX_UNLOCK(&o->one_pending_mutex);
			ECORE_MSLEEP(10);
			ECORE_MUTEX_LOCK(&o->one_pending_mutex);
			rc = o->check_transition(sc, o, params);
		}
		if (rc == ECORE_BUSY) {
			ECORE_MUTEX_UNLOCK(&o->one_pending_mutex);
			PMD_DRV_LOG(ERR, sc,
				    "timeout waiting for previous ramrod completion");
			return rc;
		}
	} else if (rc) {
		ECORE_MUTEX_UNLOCK(&o->one_pending_mutex);
		return rc;
	}

	/* Set "pending" bit */
	ECORE_SET_BIT(cmd, pending);

	/* Don't send a command if only driver cleanup was requested */
	if (ECORE_TEST_BIT(RAMROD_DRV_CLR_ONLY, &params->ramrod_flags)) {
		ecore_func_state_change_comp(sc, o, cmd);
		ECORE_MUTEX_UNLOCK(&o->one_pending_mutex);
	} else {
		/* Send a ramrod */
		rc = o->send_cmd(sc, params);

		ECORE_MUTEX_UNLOCK(&o->one_pending_mutex);

		if (rc) {
			o->next_state = ECORE_F_STATE_MAX;
			ECORE_CLEAR_BIT(cmd, pending);
			ECORE_SMP_MB_AFTER_CLEAR_BIT();
			return rc;
		}

		if (ECORE_TEST_BIT(RAMROD_COMP_WAIT, &params->ramrod_flags)) {
			rc = o->wait_comp(sc, o, cmd);
			if (rc)
				return rc;

			return ECORE_SUCCESS;
		}
	}

	return ECORE_RET_PENDING(cmd, pending);
}

/******************************************************************************
 * Description:
 *	   Callwlates crc 8 on a word value: polynomial 0-1-2-8
 *	   Code was translated from Verilog.
 * Return:
 *****************************************************************************/
uint8_t ecore_calc_crc8(uint32_t data, uint8_t crc)
{
	uint8_t D[32];
	uint8_t NewCRC[8];
	uint8_t C[8];
	uint8_t crc_res;
	uint8_t i;

	/* split the data into 31 bits */
	for (i = 0; i < 32; i++) {
		D[i] = (uint8_t) (data & 1);
		data = data >> 1;
	}

	/* split the crc into 8 bits */
	for (i = 0; i < 8; i++) {
		C[i] = crc & 1;
		crc = crc >> 1;
	}

	NewCRC[0] = D[31] ^ D[30] ^ D[28] ^ D[23] ^ D[21] ^ D[19] ^ D[18] ^
	    D[16] ^ D[14] ^ D[12] ^ D[8] ^ D[7] ^ D[6] ^ D[0] ^ C[4] ^
	    C[6] ^ C[7];
	NewCRC[1] = D[30] ^ D[29] ^ D[28] ^ D[24] ^ D[23] ^ D[22] ^ D[21] ^
	    D[20] ^ D[18] ^ D[17] ^ D[16] ^ D[15] ^ D[14] ^ D[13] ^
	    D[12] ^ D[9] ^ D[6] ^ D[1] ^ D[0] ^ C[0] ^ C[4] ^ C[5] ^ C[6];
	NewCRC[2] = D[29] ^ D[28] ^ D[25] ^ D[24] ^ D[22] ^ D[17] ^ D[15] ^
	    D[13] ^ D[12] ^ D[10] ^ D[8] ^ D[6] ^ D[2] ^ D[1] ^ D[0] ^
	    C[0] ^ C[1] ^ C[4] ^ C[5];
	NewCRC[3] = D[30] ^ D[29] ^ D[26] ^ D[25] ^ D[23] ^ D[18] ^ D[16] ^
	    D[14] ^ D[13] ^ D[11] ^ D[9] ^ D[7] ^ D[3] ^ D[2] ^ D[1] ^
	    C[1] ^ C[2] ^ C[5] ^ C[6];
	NewCRC[4] = D[31] ^ D[30] ^ D[27] ^ D[26] ^ D[24] ^ D[19] ^ D[17] ^
	    D[15] ^ D[14] ^ D[12] ^ D[10] ^ D[8] ^ D[4] ^ D[3] ^ D[2] ^
	    C[0] ^ C[2] ^ C[3] ^ C[6] ^ C[7];
	NewCRC[5] = D[31] ^ D[28] ^ D[27] ^ D[25] ^ D[20] ^ D[18] ^ D[16] ^
	    D[15] ^ D[13] ^ D[11] ^ D[9] ^ D[5] ^ D[4] ^ D[3] ^ C[1] ^
	    C[3] ^ C[4] ^ C[7];
	NewCRC[6] = D[29] ^ D[28] ^ D[26] ^ D[21] ^ D[19] ^ D[17] ^ D[16] ^
	    D[14] ^ D[12] ^ D[10] ^ D[6] ^ D[5] ^ D[4] ^ C[2] ^ C[4] ^ C[5];
	NewCRC[7] = D[30] ^ D[29] ^ D[27] ^ D[22] ^ D[20] ^ D[18] ^ D[17] ^
	    D[15] ^ D[13] ^ D[11] ^ D[7] ^ D[6] ^ D[5] ^ C[3] ^ C[5] ^ C[6];

	crc_res = 0;
	for (i = 0; i < 8; i++) {
		crc_res |= (NewCRC[i] << i);
	}

	return crc_res;
}

uint32_t
ecore_calc_crc32(uint32_t crc, uint8_t const *p, uint32_t len, uint32_t magic)
{
	int i;
	while (len--) {
		crc ^= *p++;
		for (i = 0; i < 8; i++)
			crc = (crc >> 1) ^ ((crc & 1) ? magic : 0);
	}
	return crc;
}
