/* SPDX-License-Identifier: BSD-3-Clause
 * Copyright(c) 2014-2018 Broadcom
 * All rights reserved.
 */

#ifndef _BNXT_CPR_H_
#define _BNXT_CPR_H_
#include <stdbool.h>

#include <rte_io.h>

struct bnxt_db_info;

#define CMP_VALID(cmp, raw_cons, ring)					\
	(!!(rte_le_to_cpu_32(((struct cmpl_base *)(cmp))->info3_v) &	\
	    CMPL_BASE_V) == !((raw_cons) & ((ring)->ring_size)))

#define CMPL_VALID(cmp, v)						\
	(!!(rte_le_to_cpu_32(((struct cmpl_base *)(cmp))->info3_v) &	\
	    CMPL_BASE_V) == !(v))

#define NQ_CMP_VALID(nqcmp, raw_cons, ring)		\
	(!!((nqcmp)->v & rte_cpu_to_le_32(NQ_CN_V)) ==	\
	 !((raw_cons) & ((ring)->ring_size)))

#define CMP_TYPE(cmp)						\
	(((struct cmpl_base *)cmp)->type & CMPL_BASE_TYPE_MASK)

#define ADV_RAW_CMP(idx, n)	((idx) + (n))
#define NEXT_RAW_CMP(idx)	ADV_RAW_CMP(idx, 1)
#define RING_CMP(ring, idx)	((idx) & (ring)->ring_mask)
#define RING_CMPL(ring_mask, idx)	((idx) & (ring_mask))
#define NEXT_CMP(idx)		RING_CMP(ADV_RAW_CMP(idx, 1))
#define FLIP_VALID(cons, mask, val)	((cons) >= (mask) ? !(val) : (val))

#define DB_CP_REARM_FLAGS	(DB_KEY_CP | DB_IDX_VALID)
#define DB_CP_FLAGS		(DB_KEY_CP | DB_IDX_VALID | DB_IRQ_DIS)

#define NEXT_CMPL(cpr, idx, v, inc)	do { \
	(idx) += (inc); \
	if (unlikely((idx) >= (cpr)->cp_ring_struct->ring_size)) { \
		(v) = !(v); \
		(idx) = 0; \
	} \
} while (0)
#define B_CP_DB_REARM(cpr, raw_cons)					\
	rte_write32((DB_CP_REARM_FLAGS |				\
		    RING_CMP(((cpr)->cp_ring_struct), raw_cons)),	\
		    ((cpr)->cp_db.doorbell))

#define B_CP_DB_ARM(cpr)	rte_write32((DB_KEY_CP),		\
					    ((cpr)->cp_db.doorbell))

#define B_CP_DB_DISARM(cpr)	(*(uint32_t *)((cpr)->cp_db.doorbell) = \
				 DB_KEY_CP | DB_IRQ_DIS)

#define B_CP_DB_IDX_ARM(cpr, cons)					\
		(*(uint32_t *)((cpr)->cp_db.doorbell) = (DB_CP_REARM_FLAGS | \
				(cons)))

#define B_CP_DB_IDX_DISARM(cpr, cons)	do {				\
		rte_smp_wmb();						\
		(*(uint32_t *)((cpr)->cp_db.doorbell) = (DB_CP_FLAGS |	\
				(cons));				\
} while (0)
#define B_CP_DIS_DB(cpr, raw_cons)					\
	rte_write32_relaxed((DB_CP_FLAGS |				\
			    RING_CMP(((cpr)->cp_ring_struct), raw_cons)), \
			    ((cpr)->cp_db.doorbell))

#define B_CP_DB(cpr, raw_cons, ring_mask)				\
	rte_write32((DB_CP_FLAGS |					\
		    RING_CMPL((ring_mask), raw_cons)),	\
		    ((cpr)->cp_db.doorbell))

struct bnxt_db_info {
	void                    *doorbell;
	union {
		uint64_t        db_key64;
		uint32_t        db_key32;
	};
	bool                    db_64;
};

struct bnxt_ring;
struct bnxt_cp_ring_info {
	uint32_t		cp_raw_cons;

	struct cmpl_base	*cp_desc_ring;
	struct bnxt_db_info     cp_db;
	rte_iova_t		cp_desc_mapping;

	struct ctx_hw_stats	*hw_stats;
	rte_iova_t		hw_stats_map;
	uint32_t		hw_stats_ctx_id;

	struct bnxt_ring	*cp_ring_struct;
	uint16_t		cp_cons;
	bool			valid;
};

#define RX_CMP_L2_ERRORS						\
	(RX_PKT_CMPL_ERRORS_BUFFER_ERROR_MASK | RX_PKT_CMPL_ERRORS_CRC_ERROR)

struct bnxt;
void bnxt_handle_async_event(struct bnxt *bp, struct cmpl_base *cmp);
void bnxt_handle_fwd_req(struct bnxt *bp, struct cmpl_base *cmp);
int bnxt_event_hwrm_resp_handler(struct bnxt *bp, struct cmpl_base *cmp);
void bnxt_dev_reset_and_resume(void *arg);
void bnxt_wait_for_device_shutdown(struct bnxt *bp);

#define EVENT_DATA1_REASON_CODE_FW_EXCEPTION_FATAL     \
	HWRM_ASYNC_EVENT_CMPL_RESET_NOTIFY_EVENT_DATA1_REASON_CODE_FW_EXCEPTION_FATAL
#define EVENT_DATA1_REASON_CODE_MASK                   \
	HWRM_ASYNC_EVENT_CMPL_RESET_NOTIFY_EVENT_DATA1_REASON_CODE_MASK

#define EVENT_DATA1_FLAGS_MASK                         \
	HWRM_ASYNC_EVENT_CMPL_ERROR_RECOVERY_EVENT_DATA1_FLAGS_MASK

#define EVENT_DATA1_FLAGS_MASTER_FUNC                  \
	HWRM_ASYNC_EVENT_CMPL_ERROR_RECOVERY_EVENT_DATA1_FLAGS_MASTER_FUNC

#define EVENT_DATA1_FLAGS_RECOVERY_ENABLED             \
	HWRM_ASYNC_EVENT_CMPL_ERROR_RECOVERY_EVENT_DATA1_FLAGS_RECOVERY_ENABLED

bool bnxt_is_recovery_enabled(struct bnxt *bp);
bool bnxt_is_master_func(struct bnxt *bp);

void bnxt_stop_rxtx(struct bnxt *bp);
#endif
