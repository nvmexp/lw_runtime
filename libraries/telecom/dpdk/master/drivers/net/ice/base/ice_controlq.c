/* SPDX-License-Identifier: BSD-3-Clause
 * Copyright(c) 2001-2020 Intel Corporation
 */

#include "ice_common.h"

#define ICE_CQ_INIT_REGS(qinfo, prefix)				\
do {								\
	(qinfo)->sq.head = prefix##_ATQH;			\
	(qinfo)->sq.tail = prefix##_ATQT;			\
	(qinfo)->sq.len = prefix##_ATQLEN;			\
	(qinfo)->sq.bah = prefix##_ATQBAH;			\
	(qinfo)->sq.bal = prefix##_ATQBAL;			\
	(qinfo)->sq.len_mask = prefix##_ATQLEN_ATQLEN_M;	\
	(qinfo)->sq.len_ena_mask = prefix##_ATQLEN_ATQENABLE_M;	\
	(qinfo)->sq.len_crit_mask = prefix##_ATQLEN_ATQCRIT_M;	\
	(qinfo)->sq.head_mask = prefix##_ATQH_ATQH_M;		\
	(qinfo)->rq.head = prefix##_ARQH;			\
	(qinfo)->rq.tail = prefix##_ARQT;			\
	(qinfo)->rq.len = prefix##_ARQLEN;			\
	(qinfo)->rq.bah = prefix##_ARQBAH;			\
	(qinfo)->rq.bal = prefix##_ARQBAL;			\
	(qinfo)->rq.len_mask = prefix##_ARQLEN_ARQLEN_M;	\
	(qinfo)->rq.len_ena_mask = prefix##_ARQLEN_ARQENABLE_M;	\
	(qinfo)->rq.len_crit_mask = prefix##_ARQLEN_ARQCRIT_M;	\
	(qinfo)->rq.head_mask = prefix##_ARQH_ARQH_M;		\
} while (0)

/**
 * ice_adminq_init_regs - Initialize AdminQ registers
 * @hw: pointer to the hardware structure
 *
 * This assumes the alloc_sq and alloc_rq functions have already been called
 */
static void ice_adminq_init_regs(struct ice_hw *hw)
{
	struct ice_ctl_q_info *cq = &hw->adminq;

	ice_debug(hw, ICE_DBG_TRACE, "%s\n", __func__);

	ICE_CQ_INIT_REGS(cq, PF_FW);
}

/**
 * ice_mailbox_init_regs - Initialize Mailbox registers
 * @hw: pointer to the hardware structure
 *
 * This assumes the alloc_sq and alloc_rq functions have already been called
 */
static void ice_mailbox_init_regs(struct ice_hw *hw)
{
	struct ice_ctl_q_info *cq = &hw->mailboxq;

	ICE_CQ_INIT_REGS(cq, PF_MBX);
}

/**
 * ice_check_sq_alive
 * @hw: pointer to the HW struct
 * @cq: pointer to the specific Control queue
 *
 * Returns true if Queue is enabled else false.
 */
bool ice_check_sq_alive(struct ice_hw *hw, struct ice_ctl_q_info *cq)
{
	/* check both queue-length and queue-enable fields */
	if (cq->sq.len && cq->sq.len_mask && cq->sq.len_ena_mask)
		return (rd32(hw, cq->sq.len) & (cq->sq.len_mask |
						cq->sq.len_ena_mask)) ==
			(cq->num_sq_entries | cq->sq.len_ena_mask);

	return false;
}

/**
 * ice_alloc_ctrlq_sq_ring - Allocate Control Transmit Queue (ATQ) rings
 * @hw: pointer to the hardware structure
 * @cq: pointer to the specific Control queue
 */
static enum ice_status
ice_alloc_ctrlq_sq_ring(struct ice_hw *hw, struct ice_ctl_q_info *cq)
{
	size_t size = cq->num_sq_entries * sizeof(struct ice_aq_desc);

	cq->sq.desc_buf.va = ice_alloc_dma_mem(hw, &cq->sq.desc_buf, size);
	if (!cq->sq.desc_buf.va)
		return ICE_ERR_NO_MEMORY;

	cq->sq.cmd_buf = ice_calloc(hw, cq->num_sq_entries,
				    sizeof(struct ice_sq_cd));
	if (!cq->sq.cmd_buf) {
		ice_free_dma_mem(hw, &cq->sq.desc_buf);
		return ICE_ERR_NO_MEMORY;
	}

	return ICE_SUCCESS;
}

/**
 * ice_alloc_ctrlq_rq_ring - Allocate Control Receive Queue (ARQ) rings
 * @hw: pointer to the hardware structure
 * @cq: pointer to the specific Control queue
 */
static enum ice_status
ice_alloc_ctrlq_rq_ring(struct ice_hw *hw, struct ice_ctl_q_info *cq)
{
	size_t size = cq->num_rq_entries * sizeof(struct ice_aq_desc);

	cq->rq.desc_buf.va = ice_alloc_dma_mem(hw, &cq->rq.desc_buf, size);
	if (!cq->rq.desc_buf.va)
		return ICE_ERR_NO_MEMORY;
	return ICE_SUCCESS;
}

/**
 * ice_free_cq_ring - Free control queue ring
 * @hw: pointer to the hardware structure
 * @ring: pointer to the specific control queue ring
 *
 * This assumes the posted buffers have already been cleaned
 * and de-allocated
 */
static void ice_free_cq_ring(struct ice_hw *hw, struct ice_ctl_q_ring *ring)
{
	ice_free_dma_mem(hw, &ring->desc_buf);
}

/**
 * ice_alloc_rq_bufs - Allocate pre-posted buffers for the ARQ
 * @hw: pointer to the hardware structure
 * @cq: pointer to the specific Control queue
 */
static enum ice_status
ice_alloc_rq_bufs(struct ice_hw *hw, struct ice_ctl_q_info *cq)
{
	int i;

	/* We'll be allocating the buffer info memory first, then we can
	 * allocate the mapped buffers for the event processing
	 */
	cq->rq.dma_head = ice_calloc(hw, cq->num_rq_entries,
				     sizeof(cq->rq.desc_buf));
	if (!cq->rq.dma_head)
		return ICE_ERR_NO_MEMORY;
	cq->rq.r.rq_bi = (struct ice_dma_mem *)cq->rq.dma_head;

	/* allocate the mapped buffers */
	for (i = 0; i < cq->num_rq_entries; i++) {
		struct ice_aq_desc *desc;
		struct ice_dma_mem *bi;

		bi = &cq->rq.r.rq_bi[i];
		bi->va = ice_alloc_dma_mem(hw, bi, cq->rq_buf_size);
		if (!bi->va)
			goto unwind_alloc_rq_bufs;

		/* now configure the descriptors for use */
		desc = ICE_CTL_Q_DESC(cq->rq, i);

		desc->flags = CPU_TO_LE16(ICE_AQ_FLAG_BUF);
		if (cq->rq_buf_size > ICE_AQ_LG_BUF)
			desc->flags |= CPU_TO_LE16(ICE_AQ_FLAG_LB);
		desc->opcode = 0;
		/* This is in accordance with Admin queue design, there is no
		 * register for buffer size configuration
		 */
		desc->datalen = CPU_TO_LE16(bi->size);
		desc->retval = 0;
		desc->cookie_high = 0;
		desc->cookie_low = 0;
		desc->params.generic.addr_high =
			CPU_TO_LE32(ICE_HI_DWORD(bi->pa));
		desc->params.generic.addr_low =
			CPU_TO_LE32(ICE_LO_DWORD(bi->pa));
		desc->params.generic.param0 = 0;
		desc->params.generic.param1 = 0;
	}
	return ICE_SUCCESS;

unwind_alloc_rq_bufs:
	/* don't try to free the one that failed... */
	i--;
	for (; i >= 0; i--)
		ice_free_dma_mem(hw, &cq->rq.r.rq_bi[i]);
	cq->rq.r.rq_bi = NULL;
	ice_free(hw, cq->rq.dma_head);
	cq->rq.dma_head = NULL;

	return ICE_ERR_NO_MEMORY;
}

/**
 * ice_alloc_sq_bufs - Allocate empty buffer structs for the ATQ
 * @hw: pointer to the hardware structure
 * @cq: pointer to the specific Control queue
 */
static enum ice_status
ice_alloc_sq_bufs(struct ice_hw *hw, struct ice_ctl_q_info *cq)
{
	int i;

	/* No mapped memory needed yet, just the buffer info structures */
	cq->sq.dma_head = ice_calloc(hw, cq->num_sq_entries,
				     sizeof(cq->sq.desc_buf));
	if (!cq->sq.dma_head)
		return ICE_ERR_NO_MEMORY;
	cq->sq.r.sq_bi = (struct ice_dma_mem *)cq->sq.dma_head;

	/* allocate the mapped buffers */
	for (i = 0; i < cq->num_sq_entries; i++) {
		struct ice_dma_mem *bi;

		bi = &cq->sq.r.sq_bi[i];
		bi->va = ice_alloc_dma_mem(hw, bi, cq->sq_buf_size);
		if (!bi->va)
			goto unwind_alloc_sq_bufs;
	}
	return ICE_SUCCESS;

unwind_alloc_sq_bufs:
	/* don't try to free the one that failed... */
	i--;
	for (; i >= 0; i--)
		ice_free_dma_mem(hw, &cq->sq.r.sq_bi[i]);
	cq->sq.r.sq_bi = NULL;
	ice_free(hw, cq->sq.dma_head);
	cq->sq.dma_head = NULL;

	return ICE_ERR_NO_MEMORY;
}

static enum ice_status
ice_cfg_cq_regs(struct ice_hw *hw, struct ice_ctl_q_ring *ring, u16 num_entries)
{
	/* Clear Head and Tail */
	wr32(hw, ring->head, 0);
	wr32(hw, ring->tail, 0);

	/* set starting point */
	wr32(hw, ring->len, (num_entries | ring->len_ena_mask));
	wr32(hw, ring->bal, ICE_LO_DWORD(ring->desc_buf.pa));
	wr32(hw, ring->bah, ICE_HI_DWORD(ring->desc_buf.pa));

	/* Check one register to verify that config was applied */
	if (rd32(hw, ring->bal) != ICE_LO_DWORD(ring->desc_buf.pa))
		return ICE_ERR_AQ_ERROR;

	return ICE_SUCCESS;
}

/**
 * ice_cfg_sq_regs - configure Control ATQ registers
 * @hw: pointer to the hardware structure
 * @cq: pointer to the specific Control queue
 *
 * Configure base address and length registers for the transmit queue
 */
static enum ice_status
ice_cfg_sq_regs(struct ice_hw *hw, struct ice_ctl_q_info *cq)
{
	return ice_cfg_cq_regs(hw, &cq->sq, cq->num_sq_entries);
}

/**
 * ice_cfg_rq_regs - configure Control ARQ register
 * @hw: pointer to the hardware structure
 * @cq: pointer to the specific Control queue
 *
 * Configure base address and length registers for the receive (event queue)
 */
static enum ice_status
ice_cfg_rq_regs(struct ice_hw *hw, struct ice_ctl_q_info *cq)
{
	enum ice_status status;

	status = ice_cfg_cq_regs(hw, &cq->rq, cq->num_rq_entries);
	if (status)
		return status;

	/* Update tail in the HW to post pre-allocated buffers */
	wr32(hw, cq->rq.tail, (u32)(cq->num_rq_entries - 1));

	return ICE_SUCCESS;
}

#define ICE_FREE_CQ_BUFS(hw, qi, ring)					\
do {									\
	/* free descriptors */						\
	if ((qi)->ring.r.ring##_bi) {					\
		int i;							\
									\
		for (i = 0; i < (qi)->num_##ring##_entries; i++)	\
			if ((qi)->ring.r.ring##_bi[i].pa)		\
				ice_free_dma_mem((hw),			\
					&(qi)->ring.r.ring##_bi[i]);	\
	}								\
	/* free the buffer info list */					\
	if ((qi)->ring.cmd_buf)						\
		ice_free(hw, (qi)->ring.cmd_buf);			\
	/* free DMA head */						\
	ice_free(hw, (qi)->ring.dma_head);				\
} while (0)

/**
 * ice_init_sq - main initialization routine for Control ATQ
 * @hw: pointer to the hardware structure
 * @cq: pointer to the specific Control queue
 *
 * This is the main initialization routine for the Control Send Queue
 * Prior to calling this function, the driver *MUST* set the following fields
 * in the cq->structure:
 *     - cq->num_sq_entries
 *     - cq->sq_buf_size
 *
 * Do *NOT* hold the lock when calling this as the memory allocation routines
 * called are not going to be atomic context safe
 */
static enum ice_status ice_init_sq(struct ice_hw *hw, struct ice_ctl_q_info *cq)
{
	enum ice_status ret_code;

	ice_debug(hw, ICE_DBG_TRACE, "%s\n", __func__);

	if (cq->sq.count > 0) {
		/* queue already initialized */
		ret_code = ICE_ERR_NOT_READY;
		goto init_ctrlq_exit;
	}

	/* verify input for valid configuration */
	if (!cq->num_sq_entries || !cq->sq_buf_size) {
		ret_code = ICE_ERR_CFG;
		goto init_ctrlq_exit;
	}

	cq->sq.next_to_use = 0;
	cq->sq.next_to_clean = 0;

	/* allocate the ring memory */
	ret_code = ice_alloc_ctrlq_sq_ring(hw, cq);
	if (ret_code)
		goto init_ctrlq_exit;

	/* allocate buffers in the rings */
	ret_code = ice_alloc_sq_bufs(hw, cq);
	if (ret_code)
		goto init_ctrlq_free_rings;

	/* initialize base registers */
	ret_code = ice_cfg_sq_regs(hw, cq);
	if (ret_code)
		goto init_ctrlq_free_rings;

	/* success! */
	cq->sq.count = cq->num_sq_entries;
	goto init_ctrlq_exit;

init_ctrlq_free_rings:
	ICE_FREE_CQ_BUFS(hw, cq, sq);
	ice_free_cq_ring(hw, &cq->sq);

init_ctrlq_exit:
	return ret_code;
}

/**
 * ice_init_rq - initialize ARQ
 * @hw: pointer to the hardware structure
 * @cq: pointer to the specific Control queue
 *
 * The main initialization routine for the Admin Receive (Event) Queue.
 * Prior to calling this function, the driver *MUST* set the following fields
 * in the cq->structure:
 *     - cq->num_rq_entries
 *     - cq->rq_buf_size
 *
 * Do *NOT* hold the lock when calling this as the memory allocation routines
 * called are not going to be atomic context safe
 */
static enum ice_status ice_init_rq(struct ice_hw *hw, struct ice_ctl_q_info *cq)
{
	enum ice_status ret_code;

	ice_debug(hw, ICE_DBG_TRACE, "%s\n", __func__);

	if (cq->rq.count > 0) {
		/* queue already initialized */
		ret_code = ICE_ERR_NOT_READY;
		goto init_ctrlq_exit;
	}

	/* verify input for valid configuration */
	if (!cq->num_rq_entries || !cq->rq_buf_size) {
		ret_code = ICE_ERR_CFG;
		goto init_ctrlq_exit;
	}

	cq->rq.next_to_use = 0;
	cq->rq.next_to_clean = 0;

	/* allocate the ring memory */
	ret_code = ice_alloc_ctrlq_rq_ring(hw, cq);
	if (ret_code)
		goto init_ctrlq_exit;

	/* allocate buffers in the rings */
	ret_code = ice_alloc_rq_bufs(hw, cq);
	if (ret_code)
		goto init_ctrlq_free_rings;

	/* initialize base registers */
	ret_code = ice_cfg_rq_regs(hw, cq);
	if (ret_code)
		goto init_ctrlq_free_rings;

	/* success! */
	cq->rq.count = cq->num_rq_entries;
	goto init_ctrlq_exit;

init_ctrlq_free_rings:
	ICE_FREE_CQ_BUFS(hw, cq, rq);
	ice_free_cq_ring(hw, &cq->rq);

init_ctrlq_exit:
	return ret_code;
}

/**
 * ice_shutdown_sq - shutdown the Control ATQ
 * @hw: pointer to the hardware structure
 * @cq: pointer to the specific Control queue
 *
 * The main shutdown routine for the Control Transmit Queue
 */
static enum ice_status
ice_shutdown_sq(struct ice_hw *hw, struct ice_ctl_q_info *cq)
{
	enum ice_status ret_code = ICE_SUCCESS;

	ice_debug(hw, ICE_DBG_TRACE, "%s\n", __func__);

	ice_acquire_lock(&cq->sq_lock);

	if (!cq->sq.count) {
		ret_code = ICE_ERR_NOT_READY;
		goto shutdown_sq_out;
	}

	/* Stop firmware AdminQ processing */
	wr32(hw, cq->sq.head, 0);
	wr32(hw, cq->sq.tail, 0);
	wr32(hw, cq->sq.len, 0);
	wr32(hw, cq->sq.bal, 0);
	wr32(hw, cq->sq.bah, 0);

	cq->sq.count = 0;	/* to indicate uninitialized queue */

	/* free ring buffers and the ring itself */
	ICE_FREE_CQ_BUFS(hw, cq, sq);
	ice_free_cq_ring(hw, &cq->sq);

shutdown_sq_out:
	ice_release_lock(&cq->sq_lock);
	return ret_code;
}

/**
 * ice_aq_ver_check - Check the reported AQ API version.
 * @hw: pointer to the hardware structure
 *
 * Checks if the driver should load on a given AQ API version.
 *
 * Return: 'true' iff the driver should attempt to load. 'false' otherwise.
 */
static bool ice_aq_ver_check(struct ice_hw *hw)
{
	if (hw->api_maj_ver > EXP_FW_API_VER_MAJOR) {
		/* Major API version is newer than expected, don't load */
		ice_warn(hw, "The driver for the device stopped because the LWM image is newer than expected. You must install the most recent version of the network driver.\n");
		return false;
	} else if (hw->api_maj_ver == EXP_FW_API_VER_MAJOR) {
		if (hw->api_min_ver > (EXP_FW_API_VER_MINOR + 2))
			ice_info(hw, "The driver for the device detected a newer version of the LWM image than expected. Please install the most recent version of the network driver.\n");
		else if ((hw->api_min_ver + 2) < EXP_FW_API_VER_MINOR)
			ice_info(hw, "The driver for the device detected an older version of the LWM image than expected. Please update the LWM image.\n");
	} else {
		/* Major API version is older than expected, log a warning */
		ice_info(hw, "The driver for the device detected an older version of the LWM image than expected. Please update the LWM image.\n");
	}
	return true;
}

/**
 * ice_shutdown_rq - shutdown Control ARQ
 * @hw: pointer to the hardware structure
 * @cq: pointer to the specific Control queue
 *
 * The main shutdown routine for the Control Receive Queue
 */
static enum ice_status
ice_shutdown_rq(struct ice_hw *hw, struct ice_ctl_q_info *cq)
{
	enum ice_status ret_code = ICE_SUCCESS;

	ice_debug(hw, ICE_DBG_TRACE, "%s\n", __func__);

	ice_acquire_lock(&cq->rq_lock);

	if (!cq->rq.count) {
		ret_code = ICE_ERR_NOT_READY;
		goto shutdown_rq_out;
	}

	/* Stop Control Queue processing */
	wr32(hw, cq->rq.head, 0);
	wr32(hw, cq->rq.tail, 0);
	wr32(hw, cq->rq.len, 0);
	wr32(hw, cq->rq.bal, 0);
	wr32(hw, cq->rq.bah, 0);

	/* set rq.count to 0 to indicate uninitialized queue */
	cq->rq.count = 0;

	/* free ring buffers and the ring itself */
	ICE_FREE_CQ_BUFS(hw, cq, rq);
	ice_free_cq_ring(hw, &cq->rq);

shutdown_rq_out:
	ice_release_lock(&cq->rq_lock);
	return ret_code;
}

/**
 * ice_init_check_adminq - Check version for Admin Queue to know if its alive
 * @hw: pointer to the hardware structure
 */
static enum ice_status ice_init_check_adminq(struct ice_hw *hw)
{
	struct ice_ctl_q_info *cq = &hw->adminq;
	enum ice_status status;

	ice_debug(hw, ICE_DBG_TRACE, "%s\n", __func__);

	status = ice_aq_get_fw_ver(hw, NULL);
	if (status)
		goto init_ctrlq_free_rq;

	if (!ice_aq_ver_check(hw)) {
		status = ICE_ERR_FW_API_VER;
		goto init_ctrlq_free_rq;
	}

	return ICE_SUCCESS;

init_ctrlq_free_rq:
	ice_shutdown_rq(hw, cq);
	ice_shutdown_sq(hw, cq);
	return status;
}

/**
 * ice_init_ctrlq - main initialization routine for any control Queue
 * @hw: pointer to the hardware structure
 * @q_type: specific Control queue type
 *
 * Prior to calling this function, the driver *MUST* set the following fields
 * in the cq->structure:
 *     - cq->num_sq_entries
 *     - cq->num_rq_entries
 *     - cq->rq_buf_size
 *     - cq->sq_buf_size
 *
 * NOTE: this function does not initialize the controlq locks
 */
static enum ice_status ice_init_ctrlq(struct ice_hw *hw, enum ice_ctl_q q_type)
{
	struct ice_ctl_q_info *cq;
	enum ice_status ret_code;

	ice_debug(hw, ICE_DBG_TRACE, "%s\n", __func__);

	switch (q_type) {
	case ICE_CTL_Q_ADMIN:
		ice_adminq_init_regs(hw);
		cq = &hw->adminq;
		break;
	case ICE_CTL_Q_MAILBOX:
		ice_mailbox_init_regs(hw);
		cq = &hw->mailboxq;
		break;
	default:
		return ICE_ERR_PARAM;
	}
	cq->qtype = q_type;

	/* verify input for valid configuration */
	if (!cq->num_rq_entries || !cq->num_sq_entries ||
	    !cq->rq_buf_size || !cq->sq_buf_size) {
		return ICE_ERR_CFG;
	}

	/* setup SQ command write back timeout */
	cq->sq_cmd_timeout = ICE_CTL_Q_SQ_CMD_TIMEOUT;

	/* allocate the ATQ */
	ret_code = ice_init_sq(hw, cq);
	if (ret_code)
		return ret_code;

	/* allocate the ARQ */
	ret_code = ice_init_rq(hw, cq);
	if (ret_code)
		goto init_ctrlq_free_sq;

	/* success! */
	return ICE_SUCCESS;

init_ctrlq_free_sq:
	ice_shutdown_sq(hw, cq);
	return ret_code;
}

/**
 * ice_shutdown_ctrlq - shutdown routine for any control queue
 * @hw: pointer to the hardware structure
 * @q_type: specific Control queue type
 *
 * NOTE: this function does not destroy the control queue locks.
 */
static void ice_shutdown_ctrlq(struct ice_hw *hw, enum ice_ctl_q q_type)
{
	struct ice_ctl_q_info *cq;

	ice_debug(hw, ICE_DBG_TRACE, "%s\n", __func__);

	switch (q_type) {
	case ICE_CTL_Q_ADMIN:
		cq = &hw->adminq;
		if (ice_check_sq_alive(hw, cq))
			ice_aq_q_shutdown(hw, true);
		break;
	case ICE_CTL_Q_MAILBOX:
		cq = &hw->mailboxq;
		break;
	default:
		return;
	}

	ice_shutdown_sq(hw, cq);
	ice_shutdown_rq(hw, cq);
}

/**
 * ice_shutdown_all_ctrlq - shutdown routine for all control queues
 * @hw: pointer to the hardware structure
 *
 * NOTE: this function does not destroy the control queue locks. The driver
 * may call this at runtime to shutdown and later restart control queues, such
 * as in response to a reset event.
 */
void ice_shutdown_all_ctrlq(struct ice_hw *hw)
{
	ice_debug(hw, ICE_DBG_TRACE, "%s\n", __func__);
	/* Shutdown FW admin queue */
	ice_shutdown_ctrlq(hw, ICE_CTL_Q_ADMIN);
	/* Shutdown PF-VF Mailbox */
	ice_shutdown_ctrlq(hw, ICE_CTL_Q_MAILBOX);
}

/**
 * ice_init_all_ctrlq - main initialization routine for all control queues
 * @hw: pointer to the hardware structure
 *
 * Prior to calling this function, the driver MUST* set the following fields
 * in the cq->structure for all control queues:
 *     - cq->num_sq_entries
 *     - cq->num_rq_entries
 *     - cq->rq_buf_size
 *     - cq->sq_buf_size
 *
 * NOTE: this function does not initialize the controlq locks.
 */
enum ice_status ice_init_all_ctrlq(struct ice_hw *hw)
{
	enum ice_status status;
	u32 retry = 0;

	ice_debug(hw, ICE_DBG_TRACE, "%s\n", __func__);

	/* Init FW admin queue */
	do {
		status = ice_init_ctrlq(hw, ICE_CTL_Q_ADMIN);
		if (status)
			return status;

		status = ice_init_check_adminq(hw);
		if (status != ICE_ERR_AQ_FW_CRITICAL)
			break;

		ice_debug(hw, ICE_DBG_AQ_MSG, "Retry Admin Queue init due to FW critical error\n");
		ice_shutdown_ctrlq(hw, ICE_CTL_Q_ADMIN);
		ice_msec_delay(ICE_CTL_Q_ADMIN_INIT_MSEC, true);
	} while (retry++ < ICE_CTL_Q_ADMIN_INIT_TIMEOUT);

	if (status)
		return status;
	/* Init Mailbox queue */
	return ice_init_ctrlq(hw, ICE_CTL_Q_MAILBOX);
}

/**
 * ice_init_ctrlq_locks - Initialize locks for a control queue
 * @cq: pointer to the control queue
 *
 * Initializes the send and receive queue locks for a given control queue.
 */
static void ice_init_ctrlq_locks(struct ice_ctl_q_info *cq)
{
	ice_init_lock(&cq->sq_lock);
	ice_init_lock(&cq->rq_lock);
}

/**
 * ice_create_all_ctrlq - main initialization routine for all control queues
 * @hw: pointer to the hardware structure
 *
 * Prior to calling this function, the driver *MUST* set the following fields
 * in the cq->structure for all control queues:
 *     - cq->num_sq_entries
 *     - cq->num_rq_entries
 *     - cq->rq_buf_size
 *     - cq->sq_buf_size
 *
 * This function creates all the control queue locks and then calls
 * ice_init_all_ctrlq. It should be called once during driver load. If the
 * driver needs to re-initialize control queues at run time it should call
 * ice_init_all_ctrlq instead.
 */
enum ice_status ice_create_all_ctrlq(struct ice_hw *hw)
{
	ice_init_ctrlq_locks(&hw->adminq);
	ice_init_ctrlq_locks(&hw->mailboxq);

	return ice_init_all_ctrlq(hw);
}

/**
 * ice_destroy_ctrlq_locks - Destroy locks for a control queue
 * @cq: pointer to the control queue
 *
 * Destroys the send and receive queue locks for a given control queue.
 */
static void ice_destroy_ctrlq_locks(struct ice_ctl_q_info *cq)
{
	ice_destroy_lock(&cq->sq_lock);
	ice_destroy_lock(&cq->rq_lock);
}

/**
 * ice_destroy_all_ctrlq - exit routine for all control queues
 * @hw: pointer to the hardware structure
 *
 * This function shuts down all the control queues and then destroys the
 * control queue locks. It should be called once during driver unload. The
 * driver should call ice_shutdown_all_ctrlq if it needs to shut down and
 * reinitialize control queues, such as in response to a reset event.
 */
void ice_destroy_all_ctrlq(struct ice_hw *hw)
{
	/* shut down all the control queues first */
	ice_shutdown_all_ctrlq(hw);

	ice_destroy_ctrlq_locks(&hw->adminq);
	ice_destroy_ctrlq_locks(&hw->mailboxq);
}

/**
 * ice_clean_sq - cleans Admin send queue (ATQ)
 * @hw: pointer to the hardware structure
 * @cq: pointer to the specific Control queue
 *
 * returns the number of free desc
 */
static u16 ice_clean_sq(struct ice_hw *hw, struct ice_ctl_q_info *cq)
{
	struct ice_ctl_q_ring *sq = &cq->sq;
	u16 ntc = sq->next_to_clean;
	struct ice_sq_cd *details;
	struct ice_aq_desc *desc;

	desc = ICE_CTL_Q_DESC(*sq, ntc);
	details = ICE_CTL_Q_DETAILS(*sq, ntc);

	while (rd32(hw, cq->sq.head) != ntc) {
		ice_debug(hw, ICE_DBG_AQ_MSG, "ntc %d head %d.\n", ntc, rd32(hw, cq->sq.head));
		ice_memset(desc, 0, sizeof(*desc), ICE_DMA_MEM);
		ice_memset(details, 0, sizeof(*details), ICE_NONDMA_MEM);
		ntc++;
		if (ntc == sq->count)
			ntc = 0;
		desc = ICE_CTL_Q_DESC(*sq, ntc);
		details = ICE_CTL_Q_DETAILS(*sq, ntc);
	}

	sq->next_to_clean = ntc;

	return ICE_CTL_Q_DESC_UNUSED(sq);
}

/**
 * ice_debug_cq
 * @hw: pointer to the hardware structure
 * @desc: pointer to control queue descriptor
 * @buf: pointer to command buffer
 * @buf_len: max length of buf
 *
 * Dumps debug log about control command with descriptor contents.
 */
static void ice_debug_cq(struct ice_hw *hw, void *desc, void *buf, u16 buf_len)
{
	struct ice_aq_desc *cq_desc = (struct ice_aq_desc *)desc;
	u16 datalen, flags;

	if (!((ICE_DBG_AQ_DESC | ICE_DBG_AQ_DESC_BUF) & hw->debug_mask))
		return;

	if (!desc)
		return;

	datalen = LE16_TO_CPU(cq_desc->datalen);
	flags = LE16_TO_CPU(cq_desc->flags);

	ice_debug(hw, ICE_DBG_AQ_DESC, "CQ CMD: opcode 0x%04X, flags 0x%04X, datalen 0x%04X, retval 0x%04X\n",
		  LE16_TO_CPU(cq_desc->opcode), flags, datalen,
		  LE16_TO_CPU(cq_desc->retval));
	ice_debug(hw, ICE_DBG_AQ_DESC, "\tcookie (h,l) 0x%08X 0x%08X\n",
		  LE32_TO_CPU(cq_desc->cookie_high),
		  LE32_TO_CPU(cq_desc->cookie_low));
	ice_debug(hw, ICE_DBG_AQ_DESC, "\tparam (0,1)  0x%08X 0x%08X\n",
		  LE32_TO_CPU(cq_desc->params.generic.param0),
		  LE32_TO_CPU(cq_desc->params.generic.param1));
	ice_debug(hw, ICE_DBG_AQ_DESC, "\taddr (h,l)   0x%08X 0x%08X\n",
		  LE32_TO_CPU(cq_desc->params.generic.addr_high),
		  LE32_TO_CPU(cq_desc->params.generic.addr_low));
	/* Dump buffer iff 1) one exists and 2) is either a response indicated
	 * by the DD and/or CMP flag set or a command with the RD flag set.
	 */
	if (buf && cq_desc->datalen != 0 &&
	    (flags & (ICE_AQ_FLAG_DD | ICE_AQ_FLAG_CMP) ||
	     flags & ICE_AQ_FLAG_RD)) {
		ice_debug(hw, ICE_DBG_AQ_DESC_BUF, "Buffer:\n");
		ice_debug_array(hw, ICE_DBG_AQ_DESC_BUF, 16, 1, (u8 *)buf,
				MIN_T(u16, buf_len, datalen));
	}
}

/**
 * ice_sq_done - check if FW has processed the Admin Send Queue (ATQ)
 * @hw: pointer to the HW struct
 * @cq: pointer to the specific Control queue
 *
 * Returns true if the firmware has processed all descriptors on the
 * admin send queue. Returns false if there are still requests pending.
 */
static bool ice_sq_done(struct ice_hw *hw, struct ice_ctl_q_info *cq)
{
	/* AQ designers suggest use of head for better
	 * timing reliability than DD bit
	 */
	return rd32(hw, cq->sq.head) == cq->sq.next_to_use;
}

/**
 * ice_sq_send_cmd_nolock - send command to Control Queue (ATQ)
 * @hw: pointer to the HW struct
 * @cq: pointer to the specific Control queue
 * @desc: prefilled descriptor describing the command (non DMA mem)
 * @buf: buffer to use for indirect commands (or NULL for direct commands)
 * @buf_size: size of buffer for indirect commands (or 0 for direct commands)
 * @cd: pointer to command details structure
 *
 * This is the main send command routine for the ATQ. It runs the queue,
 * cleans the queue, etc.
 */
static enum ice_status
ice_sq_send_cmd_nolock(struct ice_hw *hw, struct ice_ctl_q_info *cq,
		       struct ice_aq_desc *desc, void *buf, u16 buf_size,
		       struct ice_sq_cd *cd)
{
	struct ice_dma_mem *dma_buf = NULL;
	struct ice_aq_desc *desc_on_ring;
	bool cmd_completed = false;
	enum ice_status status = ICE_SUCCESS;
	struct ice_sq_cd *details;
	u32 total_delay = 0;
	u16 retval = 0;
	u32 val = 0;

	/* if reset is in progress return a soft error */
	if (hw->reset_ongoing)
		return ICE_ERR_RESET_ONGOING;

	cq->sq_last_status = ICE_AQ_RC_OK;

	if (!cq->sq.count) {
		ice_debug(hw, ICE_DBG_AQ_MSG, "Control Send queue not initialized.\n");
		status = ICE_ERR_AQ_EMPTY;
		goto sq_send_command_error;
	}

	if ((buf && !buf_size) || (!buf && buf_size)) {
		status = ICE_ERR_PARAM;
		goto sq_send_command_error;
	}

	if (buf) {
		if (buf_size > cq->sq_buf_size) {
			ice_debug(hw, ICE_DBG_AQ_MSG, "Invalid buffer size for Control Send queue: %d.\n",
				  buf_size);
			status = ICE_ERR_ILWAL_SIZE;
			goto sq_send_command_error;
		}

		desc->flags |= CPU_TO_LE16(ICE_AQ_FLAG_BUF);
		if (buf_size > ICE_AQ_LG_BUF)
			desc->flags |= CPU_TO_LE16(ICE_AQ_FLAG_LB);
	}

	val = rd32(hw, cq->sq.head);
	if (val >= cq->num_sq_entries) {
		ice_debug(hw, ICE_DBG_AQ_MSG, "head overrun at %d in the Control Send Queue ring\n",
			  val);
		status = ICE_ERR_AQ_EMPTY;
		goto sq_send_command_error;
	}

	details = ICE_CTL_Q_DETAILS(cq->sq, cq->sq.next_to_use);
	if (cd)
		*details = *cd;
	else
		ice_memset(details, 0, sizeof(*details), ICE_NONDMA_MEM);

	/* Call clean and check queue available function to reclaim the
	 * descriptors that were processed by FW/MBX; the function returns the
	 * number of desc available. The clean function called here could be
	 * called in a separate thread in case of asynchronous completions.
	 */
	if (ice_clean_sq(hw, cq) == 0) {
		ice_debug(hw, ICE_DBG_AQ_MSG, "Error: Control Send Queue is full.\n");
		status = ICE_ERR_AQ_FULL;
		goto sq_send_command_error;
	}

	/* initialize the temp desc pointer with the right desc */
	desc_on_ring = ICE_CTL_Q_DESC(cq->sq, cq->sq.next_to_use);

	/* if the desc is available copy the temp desc to the right place */
	ice_memcpy(desc_on_ring, desc, sizeof(*desc_on_ring),
		   ICE_NONDMA_TO_DMA);

	/* if buf is not NULL assume indirect command */
	if (buf) {
		dma_buf = &cq->sq.r.sq_bi[cq->sq.next_to_use];
		/* copy the user buf into the respective DMA buf */
		ice_memcpy(dma_buf->va, buf, buf_size, ICE_NONDMA_TO_DMA);
		desc_on_ring->datalen = CPU_TO_LE16(buf_size);

		/* Update the address values in the desc with the pa value
		 * for respective buffer
		 */
		desc_on_ring->params.generic.addr_high =
			CPU_TO_LE32(ICE_HI_DWORD(dma_buf->pa));
		desc_on_ring->params.generic.addr_low =
			CPU_TO_LE32(ICE_LO_DWORD(dma_buf->pa));
	}

	/* Debug desc and buffer */
	ice_debug(hw, ICE_DBG_AQ_DESC, "ATQ: Control Send queue desc and buffer:\n");

	ice_debug_cq(hw, (void *)desc_on_ring, buf, buf_size);

	(cq->sq.next_to_use)++;
	if (cq->sq.next_to_use == cq->sq.count)
		cq->sq.next_to_use = 0;
	wr32(hw, cq->sq.tail, cq->sq.next_to_use);

	do {
		if (ice_sq_done(hw, cq))
			break;

		ice_usec_delay(ICE_CTL_Q_SQ_CMD_USEC, false);
		total_delay++;
	} while (total_delay < cq->sq_cmd_timeout);

	/* if ready, copy the desc back to temp */
	if (ice_sq_done(hw, cq)) {
		ice_memcpy(desc, desc_on_ring, sizeof(*desc),
			   ICE_DMA_TO_NONDMA);
		if (buf) {
			/* get returned length to copy */
			u16 copy_size = LE16_TO_CPU(desc->datalen);

			if (copy_size > buf_size) {
				ice_debug(hw, ICE_DBG_AQ_MSG, "Return len %d > than buf len %d\n",
					  copy_size, buf_size);
				status = ICE_ERR_AQ_ERROR;
			} else {
				ice_memcpy(buf, dma_buf->va, copy_size,
					   ICE_DMA_TO_NONDMA);
			}
		}
		retval = LE16_TO_CPU(desc->retval);
		if (retval) {
			ice_debug(hw, ICE_DBG_AQ_MSG, "Control Send Queue command 0x%04X completed with error 0x%X\n",
				  LE16_TO_CPU(desc->opcode),
				  retval);

			/* strip off FW internal code */
			retval &= 0xff;
		}
		cmd_completed = true;
		if (!status && retval != ICE_AQ_RC_OK)
			status = ICE_ERR_AQ_ERROR;
		cq->sq_last_status = (enum ice_aq_err)retval;
	}

	ice_debug(hw, ICE_DBG_AQ_MSG, "ATQ: desc and buffer writeback:\n");

	ice_debug_cq(hw, (void *)desc, buf, buf_size);

	/* save writeback AQ if requested */
	if (details->wb_desc)
		ice_memcpy(details->wb_desc, desc_on_ring,
			   sizeof(*details->wb_desc), ICE_DMA_TO_NONDMA);

	/* update the error if time out oclwrred */
	if (!cmd_completed) {
		if (rd32(hw, cq->rq.len) & cq->rq.len_crit_mask ||
		    rd32(hw, cq->sq.len) & cq->sq.len_crit_mask) {
			ice_debug(hw, ICE_DBG_AQ_MSG, "Critical FW error.\n");
			status = ICE_ERR_AQ_FW_CRITICAL;
		} else {
			ice_debug(hw, ICE_DBG_AQ_MSG, "Control Send Queue Writeback timeout.\n");
			status = ICE_ERR_AQ_TIMEOUT;
		}
	}

sq_send_command_error:
	return status;
}

/**
 * ice_sq_send_cmd - send command to Control Queue (ATQ)
 * @hw: pointer to the HW struct
 * @cq: pointer to the specific Control queue
 * @desc: prefilled descriptor describing the command (non DMA mem)
 * @buf: buffer to use for indirect commands (or NULL for direct commands)
 * @buf_size: size of buffer for indirect commands (or 0 for direct commands)
 * @cd: pointer to command details structure
 *
 * This is the main send command routine for the ATQ. It runs the queue,
 * cleans the queue, etc.
 */
enum ice_status
ice_sq_send_cmd(struct ice_hw *hw, struct ice_ctl_q_info *cq,
		struct ice_aq_desc *desc, void *buf, u16 buf_size,
		struct ice_sq_cd *cd)
{
	enum ice_status status = ICE_SUCCESS;

	/* if reset is in progress return a soft error */
	if (hw->reset_ongoing)
		return ICE_ERR_RESET_ONGOING;

	ice_acquire_lock(&cq->sq_lock);
	status = ice_sq_send_cmd_nolock(hw, cq, desc, buf, buf_size, cd);
	ice_release_lock(&cq->sq_lock);

	return status;
}

/**
 * ice_fill_dflt_direct_cmd_desc - AQ descriptor helper function
 * @desc: pointer to the temp descriptor (non DMA mem)
 * @opcode: the opcode can be used to decide which flags to turn off or on
 *
 * Fill the desc with default values
 */
void ice_fill_dflt_direct_cmd_desc(struct ice_aq_desc *desc, u16 opcode)
{
	/* zero out the desc */
	ice_memset(desc, 0, sizeof(*desc), ICE_NONDMA_MEM);
	desc->opcode = CPU_TO_LE16(opcode);
	desc->flags = CPU_TO_LE16(ICE_AQ_FLAG_SI);
}

/**
 * ice_clean_rq_elem
 * @hw: pointer to the HW struct
 * @cq: pointer to the specific Control queue
 * @e: event info from the receive descriptor, includes any buffers
 * @pending: number of events that could be left to process
 *
 * This function cleans one Admin Receive Queue element and returns
 * the contents through e. It can also return how many events are
 * left to process through 'pending'.
 */
enum ice_status
ice_clean_rq_elem(struct ice_hw *hw, struct ice_ctl_q_info *cq,
		  struct ice_rq_event_info *e, u16 *pending)
{
	u16 ntc = cq->rq.next_to_clean;
	enum ice_status ret_code = ICE_SUCCESS;
	struct ice_aq_desc *desc;
	struct ice_dma_mem *bi;
	u16 desc_idx;
	u16 datalen;
	u16 flags;
	u16 ntu;

	/* pre-clean the event info */
	ice_memset(&e->desc, 0, sizeof(e->desc), ICE_NONDMA_MEM);

	/* take the lock before we start messing with the ring */
	ice_acquire_lock(&cq->rq_lock);

	if (!cq->rq.count) {
		ice_debug(hw, ICE_DBG_AQ_MSG, "Control Receive queue not initialized.\n");
		ret_code = ICE_ERR_AQ_EMPTY;
		goto clean_rq_elem_err;
	}

	/* set next_to_use to head */
	ntu = (u16)(rd32(hw, cq->rq.head) & cq->rq.head_mask);

	if (ntu == ntc) {
		/* nothing to do - shouldn't need to update ring's values */
		ret_code = ICE_ERR_AQ_NO_WORK;
		goto clean_rq_elem_out;
	}

	/* now clean the next descriptor */
	desc = ICE_CTL_Q_DESC(cq->rq, ntc);
	desc_idx = ntc;

	cq->rq_last_status = (enum ice_aq_err)LE16_TO_CPU(desc->retval);
	flags = LE16_TO_CPU(desc->flags);
	if (flags & ICE_AQ_FLAG_ERR) {
		ret_code = ICE_ERR_AQ_ERROR;
		ice_debug(hw, ICE_DBG_AQ_MSG, "Control Receive Queue Event 0x%04X received with error 0x%X\n",
			  LE16_TO_CPU(desc->opcode),
			  cq->rq_last_status);
	}
	ice_memcpy(&e->desc, desc, sizeof(e->desc), ICE_DMA_TO_NONDMA);
	datalen = LE16_TO_CPU(desc->datalen);
	e->msg_len = MIN_T(u16, datalen, e->buf_len);
	if (e->msg_buf && e->msg_len)
		ice_memcpy(e->msg_buf, cq->rq.r.rq_bi[desc_idx].va,
			   e->msg_len, ICE_DMA_TO_NONDMA);

	ice_debug(hw, ICE_DBG_AQ_DESC, "ARQ: desc and buffer:\n");

	ice_debug_cq(hw, (void *)desc, e->msg_buf, cq->rq_buf_size);

	/* Restore the original datalen and buffer address in the desc,
	 * FW updates datalen to indicate the event message size
	 */
	bi = &cq->rq.r.rq_bi[ntc];
	ice_memset(desc, 0, sizeof(*desc), ICE_DMA_MEM);

	desc->flags = CPU_TO_LE16(ICE_AQ_FLAG_BUF);
	if (cq->rq_buf_size > ICE_AQ_LG_BUF)
		desc->flags |= CPU_TO_LE16(ICE_AQ_FLAG_LB);
	desc->datalen = CPU_TO_LE16(bi->size);
	desc->params.generic.addr_high = CPU_TO_LE32(ICE_HI_DWORD(bi->pa));
	desc->params.generic.addr_low = CPU_TO_LE32(ICE_LO_DWORD(bi->pa));

	/* set tail = the last cleaned desc index. */
	wr32(hw, cq->rq.tail, ntc);
	/* ntc is updated to tail + 1 */
	ntc++;
	if (ntc == cq->num_rq_entries)
		ntc = 0;
	cq->rq.next_to_clean = ntc;
	cq->rq.next_to_use = ntu;

clean_rq_elem_out:
	/* Set pending if needed, unlock and return */
	if (pending) {
		/* re-read HW head to callwlate actual pending messages */
		ntu = (u16)(rd32(hw, cq->rq.head) & cq->rq.head_mask);
		*pending = (u16)((ntc > ntu ? cq->rq.count : 0) + (ntu - ntc));
	}
clean_rq_elem_err:
	ice_release_lock(&cq->rq_lock);

	return ret_code;
}
