/* SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 2016 - 2018 Cavium Inc.
 * All rights reserved.
 * www.cavium.com
 */

#include "bcm_osal.h"
#include "reg_addr.h"
#include "common_hsi.h"
#include "ecore_hsi_common.h"
#include "ecore_hsi_eth.h"
#include "ecore_rt_defs.h"
#include "ecore_status.h"
#include "ecore.h"
#include "ecore_init_ops.h"
#include "ecore_init_fw_funcs.h"
#include "ecore_cxt.h"
#include "ecore_hw.h"
#include "ecore_dev_api.h"
#include "ecore_sriov.h"
#include "ecore_mcp.h"

/* Doorbell-Queue constants */
#define DQ_RANGE_SHIFT	4
#define DQ_RANGE_ALIGN	(1 << DQ_RANGE_SHIFT)

/* Searcher constants */
#define SRC_MIN_NUM_ELEMS 256

/* GFS constants */
#define RGFS_MIN_NUM_ELEMS	256
#define TGFS_MIN_NUM_ELEMS	256

/* Timers constants */
#define TM_SHIFT	7
#define TM_ALIGN	(1 << TM_SHIFT)
#define TM_ELEM_SIZE	4

/* ILT constants */
#define ILT_DEFAULT_HW_P_SIZE	4

#define ILT_PAGE_IN_BYTES(hw_p_size)	(1U << ((hw_p_size) + 12))
#define ILT_CFG_REG(cli, reg)		PSWRQ2_REG_##cli##_##reg##_RT_OFFSET

/* ILT entry structure */
#define ILT_ENTRY_PHY_ADDR_MASK		0x000FFFFFFFFFFFULL
#define ILT_ENTRY_PHY_ADDR_SHIFT	0
#define ILT_ENTRY_VALID_MASK		0x1ULL
#define ILT_ENTRY_VALID_SHIFT		52
#define ILT_ENTRY_IN_REGS		2
#define ILT_REG_SIZE_IN_BYTES		4

/* connection context union */
union conn_context {
	struct core_conn_context core_ctx;
	struct eth_conn_context eth_ctx;
};

/* TYPE-0 task context - iSCSI, FCOE */
union type0_task_context {
};

/* TYPE-1 task context - ROCE */
union type1_task_context {
	struct regpair reserved; /* @DPDK */
};

struct src_ent {
	u8 opaque[56];
	u64 next;
};

#define CDUT_SEG_ALIGNMET 3	/* in 4k chunks */
#define CDUT_SEG_ALIGNMET_IN_BYTES (1 << (CDUT_SEG_ALIGNMET + 12))

#define CONN_CXT_SIZE(p_hwfn) \
	ALIGNED_TYPE_SIZE(union conn_context, p_hwfn)

#define SRQ_CXT_SIZE (sizeof(struct regpair) * 8) /* @DPDK */

#define TYPE0_TASK_CXT_SIZE(p_hwfn) \
	ALIGNED_TYPE_SIZE(union type0_task_context, p_hwfn)

/* Alignment is inherent to the type1_task_context structure */
#define TYPE1_TASK_CXT_SIZE(p_hwfn) sizeof(union type1_task_context)

static OSAL_INLINE bool tm_cid_proto(enum protocol_type type)
{
	return type == PROTOCOLID_TOE;
}

static bool tm_tid_proto(enum protocol_type type)
{
	return type == PROTOCOLID_FCOE;
}

/* counts the iids for the CDU/CDUC ILT client configuration */
struct ecore_cdu_iids {
	u32 pf_cids;
	u32 per_vf_cids;
};

static void ecore_cxt_cdu_iids(struct ecore_cxt_mngr *p_mngr,
			       struct ecore_cdu_iids *iids)
{
	u32 type;

	for (type = 0; type < MAX_CONN_TYPES; type++) {
		iids->pf_cids += p_mngr->conn_cfg[type].cid_count;
		iids->per_vf_cids += p_mngr->conn_cfg[type].cids_per_vf;
	}
}

/* counts the iids for the Searcher block configuration */
struct ecore_src_iids {
	u32 pf_cids;
	u32 per_vf_cids;
};

static void ecore_cxt_src_iids(struct ecore_cxt_mngr *p_mngr,
			       struct ecore_src_iids *iids)
{
	u32 i;

	for (i = 0; i < MAX_CONN_TYPES; i++) {
		iids->pf_cids += p_mngr->conn_cfg[i].cid_count;
		iids->per_vf_cids += p_mngr->conn_cfg[i].cids_per_vf;
	}

	/* Add L2 filtering filters in addition */
	iids->pf_cids += p_mngr->arfs_count;
}

/* counts the iids for the Timers block configuration */
struct ecore_tm_iids {
	u32 pf_cids;
	u32 pf_tids[NUM_TASK_PF_SEGMENTS];	/* per segment */
	u32 pf_tids_total;
	u32 per_vf_cids;
	u32 per_vf_tids;
};

static void ecore_cxt_tm_iids(struct ecore_hwfn *p_hwfn,
			      struct ecore_cxt_mngr *p_mngr,
			      struct ecore_tm_iids *iids)
{
	struct ecore_conn_type_cfg *p_cfg;
	bool tm_vf_required = false;
	bool tm_required = false;
	u32 i, j;

	for (i = 0; i < MAX_CONN_TYPES; i++) {
		p_cfg = &p_mngr->conn_cfg[i];

		if (tm_cid_proto(i) || tm_required) {
			if (p_cfg->cid_count)
				tm_required = true;

			iids->pf_cids += p_cfg->cid_count;
		}

		if (tm_cid_proto(i) || tm_vf_required) {
			if (p_cfg->cids_per_vf)
				tm_vf_required = true;

		}

		if (tm_tid_proto(i)) {
			struct ecore_tid_seg *segs = p_cfg->tid_seg;

			/* for each segment there is at most one
			 * protocol for which count is not 0.
			 */
			for (j = 0; j < NUM_TASK_PF_SEGMENTS; j++)
				iids->pf_tids[j] += segs[j].count;

			/* The last array elelment is for the VFs. As for PF
			 * segments there can be only one protocol for
			 * which this value is not 0.
			 */
			iids->per_vf_tids += segs[NUM_TASK_PF_SEGMENTS].count;
		}
	}

	iids->pf_cids = ROUNDUP(iids->pf_cids, TM_ALIGN);
	iids->per_vf_cids = ROUNDUP(iids->per_vf_cids, TM_ALIGN);
	iids->per_vf_tids = ROUNDUP(iids->per_vf_tids, TM_ALIGN);

	for (iids->pf_tids_total = 0, j = 0; j < NUM_TASK_PF_SEGMENTS; j++) {
		iids->pf_tids[j] = ROUNDUP(iids->pf_tids[j], TM_ALIGN);
		iids->pf_tids_total += iids->pf_tids[j];
	}
}

static void ecore_cxt_qm_iids(struct ecore_hwfn *p_hwfn,
			      struct ecore_qm_iids *iids)
{
	struct ecore_cxt_mngr *p_mngr = p_hwfn->p_cxt_mngr;
	struct ecore_tid_seg *segs;
	u32 vf_cids = 0, type, j;
	u32 vf_tids = 0;

	for (type = 0; type < MAX_CONN_TYPES; type++) {
		iids->cids += p_mngr->conn_cfg[type].cid_count;
		vf_cids += p_mngr->conn_cfg[type].cids_per_vf;

		segs = p_mngr->conn_cfg[type].tid_seg;
		/* for each segment there is at most one
		 * protocol for which count is not 0.
		 */
		for (j = 0; j < NUM_TASK_PF_SEGMENTS; j++)
			iids->tids += segs[j].count;

		/* The last array elelment is for the VFs. As for PF
		 * segments there can be only one protocol for
		 * which this value is not 0.
		 */
		vf_tids += segs[NUM_TASK_PF_SEGMENTS].count;
	}

	iids->vf_cids += vf_cids * p_mngr->vf_count;
	iids->tids += vf_tids * p_mngr->vf_count;

	DP_VERBOSE(p_hwfn, ECORE_MSG_ILT,
		   "iids: CIDS %08x vf_cids %08x tids %08x vf_tids %08x\n",
		   iids->cids, iids->vf_cids, iids->tids, vf_tids);
}

static struct ecore_tid_seg *ecore_cxt_tid_seg_info(struct ecore_hwfn *p_hwfn,
						    u32 seg)
{
	struct ecore_cxt_mngr *p_cfg = p_hwfn->p_cxt_mngr;
	u32 i;

	/* Find the protocol with tid count > 0 for this segment.
	 * Note: there can only be one and this is already validated.
	 */
	for (i = 0; i < MAX_CONN_TYPES; i++) {
		if (p_cfg->conn_cfg[i].tid_seg[seg].count)
			return &p_cfg->conn_cfg[i].tid_seg[seg];
	}
	return OSAL_NULL;
}

static void ecore_cxt_set_srq_count(struct ecore_hwfn *p_hwfn, u32 num_srqs)
{
	struct ecore_cxt_mngr *p_mgr = p_hwfn->p_cxt_mngr;

	p_mgr->srq_count = num_srqs;
}

u32 ecore_cxt_get_srq_count(struct ecore_hwfn *p_hwfn)
{
	struct ecore_cxt_mngr *p_mgr = p_hwfn->p_cxt_mngr;

	return p_mgr->srq_count;
}

/* set the iids (cid/tid) count per protocol */
static void ecore_cxt_set_proto_cid_count(struct ecore_hwfn *p_hwfn,
				   enum protocol_type type,
				   u32 cid_count, u32 vf_cid_cnt)
{
	struct ecore_cxt_mngr *p_mgr = p_hwfn->p_cxt_mngr;
	struct ecore_conn_type_cfg *p_conn = &p_mgr->conn_cfg[type];

	p_conn->cid_count = ROUNDUP(cid_count, DQ_RANGE_ALIGN);
	p_conn->cids_per_vf = ROUNDUP(vf_cid_cnt, DQ_RANGE_ALIGN);
}

u32 ecore_cxt_get_proto_cid_count(struct ecore_hwfn *p_hwfn,
				  enum protocol_type type, u32 *vf_cid)
{
	if (vf_cid)
		*vf_cid = p_hwfn->p_cxt_mngr->conn_cfg[type].cids_per_vf;

	return p_hwfn->p_cxt_mngr->conn_cfg[type].cid_count;
}

u32 ecore_cxt_get_proto_cid_start(struct ecore_hwfn *p_hwfn,
				  enum protocol_type type)
{
	return p_hwfn->p_cxt_mngr->acquired[type].start_cid;
}

u32 ecore_cxt_get_proto_tid_count(struct ecore_hwfn *p_hwfn,
					 enum protocol_type type)
{
	u32 cnt = 0;
	int i;

	for (i = 0; i < TASK_SEGMENTS; i++)
		cnt += p_hwfn->p_cxt_mngr->conn_cfg[type].tid_seg[i].count;

	return cnt;
}

static OSAL_INLINE void
ecore_cxt_set_proto_tid_count(struct ecore_hwfn *p_hwfn,
			      enum protocol_type proto,
			      u8 seg, u8 seg_type, u32 count, bool has_fl)
{
	struct ecore_cxt_mngr *p_mngr = p_hwfn->p_cxt_mngr;
	struct ecore_tid_seg *p_seg = &p_mngr->conn_cfg[proto].tid_seg[seg];

	p_seg->count = count;
	p_seg->has_fl_mem = has_fl;
	p_seg->type = seg_type;
}

/* the *p_line parameter must be either 0 for the first invocation or the
 * value returned in the previous invocation.
 */
static void ecore_ilt_cli_blk_fill(struct ecore_ilt_client_cfg *p_cli,
				   struct ecore_ilt_cli_blk *p_blk,
				   u32 start_line,
				   u32 total_size, u32 elem_size)
{
	u32 ilt_size = ILT_PAGE_IN_BYTES(p_cli->p_size.val);

	/* verify that it's called once for each block */
	if (p_blk->total_size)
		return;

	p_blk->total_size = total_size;
	p_blk->real_size_in_page = 0;
	if (elem_size)
		p_blk->real_size_in_page = (ilt_size / elem_size) * elem_size;
	p_blk->start_line = start_line;
}

static void ecore_ilt_cli_adv_line(struct ecore_hwfn *p_hwfn,
				   struct ecore_ilt_client_cfg *p_cli,
				   struct ecore_ilt_cli_blk *p_blk,
				   u32 *p_line, enum ilt_clients client_id)
{
	if (!p_blk->total_size)
		return;

	if (!p_cli->active)
		p_cli->first.val = *p_line;

	p_cli->active = true;
	*p_line += DIV_ROUND_UP(p_blk->total_size, p_blk->real_size_in_page);
	p_cli->last.val = *p_line - 1;

	DP_VERBOSE(p_hwfn, ECORE_MSG_ILT,
		   "ILT[Client %d] - Lines: [%08x - %08x]. Block - Size %08x"
		   " [Real %08x] Start line %d\n",
		   client_id, p_cli->first.val, p_cli->last.val,
		   p_blk->total_size, p_blk->real_size_in_page,
		   p_blk->start_line);
}

static void ecore_ilt_get_dynamic_line_range(struct ecore_hwfn *p_hwfn,
					     enum ilt_clients ilt_client,
					     u32 *dynamic_line_offset,
					     u32 *dynamic_line_cnt)
{
	struct ecore_ilt_client_cfg *p_cli;
	struct ecore_conn_type_cfg *p_cfg;
	u32 cxts_per_p;

	/* TBD MK: ILT code should be simplified once PROTO enum is changed */

	*dynamic_line_offset = 0;
	*dynamic_line_cnt = 0;

	if (ilt_client == ILT_CLI_CDUC) {
		p_cli = &p_hwfn->p_cxt_mngr->clients[ILT_CLI_CDUC];
		p_cfg = &p_hwfn->p_cxt_mngr->conn_cfg[PROTOCOLID_ROCE];

		cxts_per_p = ILT_PAGE_IN_BYTES(p_cli->p_size.val) /
		    (u32)CONN_CXT_SIZE(p_hwfn);

		*dynamic_line_cnt = p_cfg->cid_count / cxts_per_p;
	}
}

static struct ecore_ilt_client_cfg *
ecore_cxt_set_cli(struct ecore_ilt_client_cfg *p_cli)
{
	p_cli->active = false;
	p_cli->first.val = 0;
	p_cli->last.val = 0;
	return p_cli;
}

static struct ecore_ilt_cli_blk *
ecore_cxt_set_blk(struct ecore_ilt_cli_blk *p_blk)
{
	p_blk->total_size = 0;
	return p_blk;
	}

static u32
ecore_cxt_src_elements(struct ecore_cxt_mngr *p_mngr)
{
	struct ecore_src_iids src_iids;
	u32 elem_num = 0;

	OSAL_MEM_ZERO(&src_iids, sizeof(src_iids));
	ecore_cxt_src_iids(p_mngr, &src_iids);

	/* Both the PF and VFs searcher connections are stored in the per PF
	 * database. Thus sum the PF searcher cids and all the VFs searcher
	 * cids.
	 */
	elem_num = src_iids.pf_cids +
		   src_iids.per_vf_cids * p_mngr->vf_count;
	if (elem_num == 0)
		return elem_num;

	elem_num = OSAL_MAX_T(u32, elem_num, SRC_MIN_NUM_ELEMS);
	elem_num = OSAL_ROUNDUP_POW_OF_TWO(elem_num);

	return elem_num;
}

enum _ecore_status_t ecore_cxt_cfg_ilt_compute(struct ecore_hwfn *p_hwfn)
{
	u32 lwrr_line, total, i, task_size, line, total_size, elem_size;
	struct ecore_cxt_mngr *p_mngr = p_hwfn->p_cxt_mngr;
	struct ecore_ilt_client_cfg *p_cli;
	struct ecore_ilt_cli_blk *p_blk;
	struct ecore_cdu_iids cdu_iids;
	struct ecore_qm_iids qm_iids;
	struct ecore_tm_iids tm_iids;
	struct ecore_tid_seg *p_seg;

	OSAL_MEM_ZERO(&qm_iids, sizeof(qm_iids));
	OSAL_MEM_ZERO(&cdu_iids, sizeof(cdu_iids));
	OSAL_MEM_ZERO(&tm_iids, sizeof(tm_iids));

	p_mngr->pf_start_line = RESC_START(p_hwfn, ECORE_ILT);

	DP_VERBOSE(p_hwfn, ECORE_MSG_ILT,
		   "hwfn [%d] - Set context mngr starting line to be 0x%08x\n",
		   p_hwfn->my_id, p_hwfn->p_cxt_mngr->pf_start_line);

	/* CDUC */
	p_cli = ecore_cxt_set_cli(&p_mngr->clients[ILT_CLI_CDUC]);

	lwrr_line = p_mngr->pf_start_line;

	/* CDUC PF */
	p_cli->pf_total_lines = 0;

	/* get the counters for the CDUC,CDUC and QM clients  */
	ecore_cxt_cdu_iids(p_mngr, &cdu_iids);

	p_blk = ecore_cxt_set_blk(&p_cli->pf_blks[CDUC_BLK]);

	total = cdu_iids.pf_cids * CONN_CXT_SIZE(p_hwfn);

	ecore_ilt_cli_blk_fill(p_cli, p_blk, lwrr_line,
			       total, CONN_CXT_SIZE(p_hwfn));

	ecore_ilt_cli_adv_line(p_hwfn, p_cli, p_blk, &lwrr_line, ILT_CLI_CDUC);
	p_cli->pf_total_lines = lwrr_line - p_blk->start_line;

	ecore_ilt_get_dynamic_line_range(p_hwfn, ILT_CLI_CDUC,
					 &p_blk->dynamic_line_offset,
					 &p_blk->dynamic_line_cnt);

	/* CDUC VF */
	p_blk = ecore_cxt_set_blk(&p_cli->vf_blks[CDUC_BLK]);
	total = cdu_iids.per_vf_cids * CONN_CXT_SIZE(p_hwfn);

	ecore_ilt_cli_blk_fill(p_cli, p_blk, lwrr_line,
			       total, CONN_CXT_SIZE(p_hwfn));

	ecore_ilt_cli_adv_line(p_hwfn, p_cli, p_blk, &lwrr_line, ILT_CLI_CDUC);
	p_cli->vf_total_lines = lwrr_line - p_blk->start_line;

	for (i = 1; i < p_mngr->vf_count; i++)
		ecore_ilt_cli_adv_line(p_hwfn, p_cli, p_blk, &lwrr_line,
				       ILT_CLI_CDUC);

	/* CDUT PF */
	p_cli = ecore_cxt_set_cli(&p_mngr->clients[ILT_CLI_CDUT]);
	p_cli->first.val = lwrr_line;

	/* first the 'working' task memory */
	for (i = 0; i < NUM_TASK_PF_SEGMENTS; i++) {
		p_seg = ecore_cxt_tid_seg_info(p_hwfn, i);
		if (!p_seg || p_seg->count == 0)
			continue;

		p_blk = ecore_cxt_set_blk(&p_cli->pf_blks[CDUT_SEG_BLK(i)]);
		total = p_seg->count * p_mngr->task_type_size[p_seg->type];
		ecore_ilt_cli_blk_fill(p_cli, p_blk, lwrr_line, total,
				       p_mngr->task_type_size[p_seg->type]);

		ecore_ilt_cli_adv_line(p_hwfn, p_cli, p_blk, &lwrr_line,
				       ILT_CLI_CDUT);
	}

	/* next the 'init' task memory (forced load memory) */
	for (i = 0; i < NUM_TASK_PF_SEGMENTS; i++) {
		p_seg = ecore_cxt_tid_seg_info(p_hwfn, i);
		if (!p_seg || p_seg->count == 0)
			continue;

		p_blk =
		     ecore_cxt_set_blk(&p_cli->pf_blks[CDUT_FL_SEG_BLK(i, PF)]);

		if (!p_seg->has_fl_mem) {
			/* The segment is active (total size pf 'working'
			 * memory is > 0) but has no FL (forced-load, Init)
			 * memory. Thus:
			 *
			 * 1.   The total-size in the corrsponding FL block of
			 *      the ILT client is set to 0 - No ILT line are
			 *      provisioned and no ILT memory allocated.
			 *
			 * 2.   The start-line of said block is set to the
			 *      start line of the matching working memory
			 *      block in the ILT client. This is later used to
			 *      configure the CDU segment offset registers and
			 *      results in an FL command for TIDs of this
			 *      segment behaves as regular load commands
			 *      (loading TIDs from the working memory).
			 */
			line = p_cli->pf_blks[CDUT_SEG_BLK(i)].start_line;

			ecore_ilt_cli_blk_fill(p_cli, p_blk, line, 0, 0);
			continue;
		}
		total = p_seg->count * p_mngr->task_type_size[p_seg->type];

		ecore_ilt_cli_blk_fill(p_cli, p_blk,
				       lwrr_line, total,
				       p_mngr->task_type_size[p_seg->type]);

		ecore_ilt_cli_adv_line(p_hwfn, p_cli, p_blk, &lwrr_line,
				       ILT_CLI_CDUT);
	}
	p_cli->pf_total_lines = lwrr_line - p_cli->first.val;

	/* CDUT VF */
	p_seg = ecore_cxt_tid_seg_info(p_hwfn, TASK_SEGMENT_VF);
	if (p_seg && p_seg->count) {
		/* Stricly speaking we need to iterate over all VF
		 * task segment types, but a VF has only 1 segment
		 */

		/* 'working' memory */
		total = p_seg->count * p_mngr->task_type_size[p_seg->type];

		p_blk = ecore_cxt_set_blk(&p_cli->vf_blks[CDUT_SEG_BLK(0)]);
		ecore_ilt_cli_blk_fill(p_cli, p_blk,
				       lwrr_line, total,
				       p_mngr->task_type_size[p_seg->type]);

		ecore_ilt_cli_adv_line(p_hwfn, p_cli, p_blk, &lwrr_line,
				       ILT_CLI_CDUT);

		/* 'init' memory */
		p_blk =
		     ecore_cxt_set_blk(&p_cli->vf_blks[CDUT_FL_SEG_BLK(0, VF)]);
		if (!p_seg->has_fl_mem) {
			/* see comment above */
			line = p_cli->vf_blks[CDUT_SEG_BLK(0)].start_line;
			ecore_ilt_cli_blk_fill(p_cli, p_blk, line, 0, 0);
		} else {
			task_size = p_mngr->task_type_size[p_seg->type];
			ecore_ilt_cli_blk_fill(p_cli, p_blk,
					       lwrr_line, total, task_size);
			ecore_ilt_cli_adv_line(p_hwfn, p_cli, p_blk, &lwrr_line,
					       ILT_CLI_CDUT);
		}
		p_cli->vf_total_lines = lwrr_line - (p_cli->first.val +
						     p_cli->pf_total_lines);

		/* Now for the rest of the VFs */
		for (i = 1; i < p_mngr->vf_count; i++) {
			/* don't set p_blk i.e. don't clear total_size */
			p_blk = &p_cli->vf_blks[CDUT_SEG_BLK(0)];
			ecore_ilt_cli_adv_line(p_hwfn, p_cli, p_blk, &lwrr_line,
					       ILT_CLI_CDUT);

			/* don't set p_blk i.e. don't clear total_size */
			p_blk = &p_cli->vf_blks[CDUT_FL_SEG_BLK(0, VF)];
			ecore_ilt_cli_adv_line(p_hwfn, p_cli, p_blk, &lwrr_line,
					       ILT_CLI_CDUT);
		}
	}

	/* QM */
	p_cli = ecore_cxt_set_cli(&p_mngr->clients[ILT_CLI_QM]);
	p_blk = ecore_cxt_set_blk(&p_cli->pf_blks[0]);

	/* At this stage, after the first QM configuration, the PF PQs amount
	 * is the highest possible. Save this value at qm_info->ilt_pf_pqs to
	 * detect overflows in the future.
	 * Even though VF PQs amount can be larger than VF count, use vf_count
	 * because each VF requires only the full amount of CIDs.
	 */
	ecore_cxt_qm_iids(p_hwfn, &qm_iids);
	total = ecore_qm_pf_mem_size(p_hwfn, qm_iids.cids,
				     qm_iids.vf_cids, qm_iids.tids,
				     p_hwfn->qm_info.num_pqs + OFLD_GRP_SIZE,
				     p_hwfn->qm_info.num_vf_pqs);

	DP_VERBOSE(p_hwfn, ECORE_MSG_ILT,
		   "QM ILT Info, (cids=%d, vf_cids=%d, tids=%d, num_pqs=%d,"
		   " num_vf_pqs=%d, memory_size=%d)\n",
		   qm_iids.cids, qm_iids.vf_cids, qm_iids.tids,
		   p_hwfn->qm_info.num_pqs, p_hwfn->qm_info.num_vf_pqs, total);

	ecore_ilt_cli_blk_fill(p_cli, p_blk, lwrr_line, total * 0x1000,
			       QM_PQ_ELEMENT_SIZE);

	ecore_ilt_cli_adv_line(p_hwfn, p_cli, p_blk, &lwrr_line, ILT_CLI_QM);
	p_cli->pf_total_lines = lwrr_line - p_blk->start_line;

	/* TM PF */
	p_cli = ecore_cxt_set_cli(&p_mngr->clients[ILT_CLI_TM]);
	ecore_cxt_tm_iids(p_hwfn, p_mngr, &tm_iids);
	total = tm_iids.pf_cids + tm_iids.pf_tids_total;
	if (total) {
		p_blk = ecore_cxt_set_blk(&p_cli->pf_blks[0]);
		ecore_ilt_cli_blk_fill(p_cli, p_blk, lwrr_line,
				       total * TM_ELEM_SIZE,
				       TM_ELEM_SIZE);

		ecore_ilt_cli_adv_line(p_hwfn, p_cli, p_blk, &lwrr_line,
				       ILT_CLI_TM);
		p_cli->pf_total_lines = lwrr_line - p_blk->start_line;
	}

	/* TM VF */
	total = tm_iids.per_vf_cids + tm_iids.per_vf_tids;
	if (total) {
		p_blk = ecore_cxt_set_blk(&p_cli->vf_blks[0]);
		ecore_ilt_cli_blk_fill(p_cli, p_blk, lwrr_line,
				       total * TM_ELEM_SIZE, TM_ELEM_SIZE);

		ecore_ilt_cli_adv_line(p_hwfn, p_cli, p_blk, &lwrr_line,
				       ILT_CLI_TM);

		p_cli->vf_total_lines = lwrr_line - p_blk->start_line;
		for (i = 1; i < p_mngr->vf_count; i++) {
			ecore_ilt_cli_adv_line(p_hwfn, p_cli, p_blk, &lwrr_line,
					       ILT_CLI_TM);
		}
	}

	/* SRC */
	p_cli = ecore_cxt_set_cli(&p_mngr->clients[ILT_CLI_SRC]);
	total = ecore_cxt_src_elements(p_mngr);

	if (total) {
		total_size = total * sizeof(struct src_ent);
		elem_size = sizeof(struct src_ent);

		p_blk = ecore_cxt_set_blk(&p_cli->pf_blks[0]);
		ecore_ilt_cli_blk_fill(p_cli, p_blk, lwrr_line,
				       total_size, elem_size);
		ecore_ilt_cli_adv_line(p_hwfn, p_cli, p_blk, &lwrr_line,
				       ILT_CLI_SRC);
		p_cli->pf_total_lines = lwrr_line - p_blk->start_line;
	}

	/* TSDM (SRQ CONTEXT) */
	total = ecore_cxt_get_srq_count(p_hwfn);

	if (total) {
		p_cli = ecore_cxt_set_cli(&p_mngr->clients[ILT_CLI_TSDM]);
		p_blk = ecore_cxt_set_blk(&p_cli->pf_blks[SRQ_BLK]);
		ecore_ilt_cli_blk_fill(p_cli, p_blk, lwrr_line,
				       total * SRQ_CXT_SIZE, SRQ_CXT_SIZE);

		ecore_ilt_cli_adv_line(p_hwfn, p_cli, p_blk, &lwrr_line,
				       ILT_CLI_TSDM);
		p_cli->pf_total_lines = lwrr_line - p_blk->start_line;
	}

	if (lwrr_line - p_hwfn->p_cxt_mngr->pf_start_line >
	    RESC_NUM(p_hwfn, ECORE_ILT)) {
		DP_ERR(p_hwfn, "too many ilt lines...#lines=%d\n",
		       lwrr_line - p_hwfn->p_cxt_mngr->pf_start_line);
		return ECORE_ILWAL;
	}

	return ECORE_SUCCESS;
}

static void ecore_cxt_src_t2_free(struct ecore_hwfn *p_hwfn)
{
	struct ecore_src_t2 *p_t2 = &p_hwfn->p_cxt_mngr->src_t2;
	u32 i;

	if (!p_t2 || !p_t2->dma_mem)
		return;

	for (i = 0; i < p_t2->num_pages; i++)
		if (p_t2->dma_mem[i].virt_addr)
			OSAL_DMA_FREE_COHERENT(p_hwfn->p_dev,
					       p_t2->dma_mem[i].virt_addr,
					       p_t2->dma_mem[i].phys_addr,
					       p_t2->dma_mem[i].size);

	OSAL_FREE(p_hwfn->p_dev, p_t2->dma_mem);
	p_t2->dma_mem = OSAL_NULL;
}

static enum _ecore_status_t
ecore_cxt_t2_alloc_pages(struct ecore_hwfn *p_hwfn,
			 struct ecore_src_t2 *p_t2,
			 u32 total_size, u32 page_size)
{
	void **p_virt;
	u32 size, i;

	if (!p_t2 || !p_t2->dma_mem)
		return ECORE_ILWAL;

	for (i = 0; i < p_t2->num_pages; i++) {
		size = OSAL_MIN_T(u32, total_size, page_size);
		p_virt = &p_t2->dma_mem[i].virt_addr;

		*p_virt = OSAL_DMA_ALLOC_COHERENT(p_hwfn->p_dev,
						  &p_t2->dma_mem[i].phys_addr,
						  size);
		if (!p_t2->dma_mem[i].virt_addr)
			return ECORE_NOMEM;

		OSAL_MEM_ZERO(*p_virt, size);
		p_t2->dma_mem[i].size = size;
		total_size -= size;
	}

	return ECORE_SUCCESS;
}

static enum _ecore_status_t ecore_cxt_src_t2_alloc(struct ecore_hwfn *p_hwfn)
{
	struct ecore_cxt_mngr *p_mngr = p_hwfn->p_cxt_mngr;
	u32 conn_num, total_size, ent_per_page, psz, i;
	struct phys_mem_desc *p_t2_last_page;
	struct ecore_ilt_client_cfg *p_src;
	struct ecore_src_iids src_iids;
	struct ecore_src_t2 *p_t2;
	enum _ecore_status_t rc;

	OSAL_MEM_ZERO(&src_iids, sizeof(src_iids));

	/* if the SRC ILT client is inactive - there are no connection
	 * requiring the searcer, leave.
	 */
	p_src = &p_hwfn->p_cxt_mngr->clients[ILT_CLI_SRC];
	if (!p_src->active)
		return ECORE_SUCCESS;

	ecore_cxt_src_iids(p_mngr, &src_iids);
	conn_num = src_iids.pf_cids + src_iids.per_vf_cids * p_mngr->vf_count;
	total_size = conn_num * sizeof(struct src_ent);

	/* use the same page size as the SRC ILT client */
	psz = ILT_PAGE_IN_BYTES(p_src->p_size.val);
	p_t2 = &p_mngr->src_t2;
	p_t2->num_pages = DIV_ROUND_UP(total_size, psz);

	/* allocate t2 */
	p_t2->dma_mem = OSAL_ZALLOC(p_hwfn->p_dev, GFP_KERNEL,
				    p_t2->num_pages *
				    sizeof(struct phys_mem_desc));
	if (!p_t2->dma_mem) {
		DP_NOTICE(p_hwfn, false, "Failed to allocate t2 table\n");
		rc = ECORE_NOMEM;
		goto t2_fail;
	}

	rc = ecore_cxt_t2_alloc_pages(p_hwfn, p_t2, total_size, psz);
	if (rc)
		goto t2_fail;

	/* Set the t2 pointers */

	/* entries per page - must be a power of two */
	ent_per_page = psz / sizeof(struct src_ent);

	p_t2->first_free = (u64)p_t2->dma_mem[0].phys_addr;

	p_t2_last_page = &p_t2->dma_mem[(conn_num - 1) / ent_per_page];
	p_t2->last_free = (u64)p_t2_last_page->phys_addr +
			  ((conn_num - 1) & (ent_per_page - 1)) *
			  sizeof(struct src_ent);

	for (i = 0; i < p_t2->num_pages; i++) {
		u32 ent_num = OSAL_MIN_T(u32, ent_per_page, conn_num);
		struct src_ent *entries = p_t2->dma_mem[i].virt_addr;
		u64 p_ent_phys = (u64)p_t2->dma_mem[i].phys_addr, val;
		u32 j;

		for (j = 0; j < ent_num - 1; j++) {
			val = p_ent_phys + (j + 1) * sizeof(struct src_ent);
			entries[j].next = OSAL_CPU_TO_BE64(val);
		}

		if (i < p_t2->num_pages - 1)
			val = (u64)p_t2->dma_mem[i + 1].phys_addr;
		else
			val = 0;
		entries[j].next = OSAL_CPU_TO_BE64(val);

		conn_num -= ent_num;
	}

	return ECORE_SUCCESS;

t2_fail:
	ecore_cxt_src_t2_free(p_hwfn);
	return rc;
}

#define for_each_ilt_valid_client(pos, clients)		\
	for (pos = 0; pos < MAX_ILT_CLIENTS; pos++)		\
		if (!clients[pos].active) {		\
			continue;			\
		} else					\


/* Total number of ILT lines used by this PF */
static u32 ecore_cxt_ilt_shadow_size(struct ecore_ilt_client_cfg *ilt_clients)
{
	u32 size = 0;
	u32 i;

	for_each_ilt_valid_client(i, ilt_clients)
		size += (ilt_clients[i].last.val -
			 ilt_clients[i].first.val + 1);

	return size;
}

static void ecore_ilt_shadow_free(struct ecore_hwfn *p_hwfn)
{
	struct ecore_ilt_client_cfg *p_cli = p_hwfn->p_cxt_mngr->clients;
	struct ecore_cxt_mngr *p_mngr = p_hwfn->p_cxt_mngr;
	u32 ilt_size, i;

	if (p_mngr->ilt_shadow == OSAL_NULL)
		return;

	ilt_size = ecore_cxt_ilt_shadow_size(p_cli);

	for (i = 0; p_mngr->ilt_shadow && i < ilt_size; i++) {
		struct phys_mem_desc *p_dma = &p_mngr->ilt_shadow[i];

		if (p_dma->virt_addr)
			OSAL_DMA_FREE_COHERENT(p_hwfn->p_dev,
					       p_dma->p_virt,
					       p_dma->phys_addr, p_dma->size);
		p_dma->virt_addr = OSAL_NULL;
	}
	OSAL_FREE(p_hwfn->p_dev, p_mngr->ilt_shadow);
	p_mngr->ilt_shadow = OSAL_NULL;
}

static enum _ecore_status_t
ecore_ilt_blk_alloc(struct ecore_hwfn *p_hwfn,
		    struct ecore_ilt_cli_blk *p_blk,
		    enum ilt_clients ilt_client, u32 start_line_offset)
{
	struct phys_mem_desc *ilt_shadow = p_hwfn->p_cxt_mngr->ilt_shadow;
	u32 lines, line, sz_left, lines_to_skip, first_skipped_line;

	/* Special handling for RoCE that supports dynamic allocation */
	if (ilt_client == ILT_CLI_CDUT || ilt_client == ILT_CLI_TSDM)
		return ECORE_SUCCESS;

	if (!p_blk->total_size)
		return ECORE_SUCCESS;

	sz_left = p_blk->total_size;
	lines_to_skip = p_blk->dynamic_line_cnt;
	lines = DIV_ROUND_UP(sz_left, p_blk->real_size_in_page) - lines_to_skip;
	line = p_blk->start_line + start_line_offset -
	       p_hwfn->p_cxt_mngr->pf_start_line;
	first_skipped_line = line + p_blk->dynamic_line_offset;

	while (lines) {
		dma_addr_t p_phys;
		void *p_virt;
		u32 size;

		if (lines_to_skip && (line == first_skipped_line)) {
			line += lines_to_skip;
			continue;
		}

		size = OSAL_MIN_T(u32, sz_left, p_blk->real_size_in_page);

/* @DPDK */
#define ILT_BLOCK_ALIGN_SIZE 0x1000
		p_virt = OSAL_DMA_ALLOC_COHERENT_ALIGNED(p_hwfn->p_dev,
							 &p_phys, size,
							 ILT_BLOCK_ALIGN_SIZE);
		if (!p_virt)
			return ECORE_NOMEM;
		OSAL_MEM_ZERO(p_virt, size);

		ilt_shadow[line].phys_addr = p_phys;
		ilt_shadow[line].virt_addr = p_virt;
		ilt_shadow[line].size = size;

		DP_VERBOSE(p_hwfn, ECORE_MSG_ILT,
			   "ILT shadow: Line [%d] Physical 0x%lx"
			   " Virtual %p Size %d\n",
			   line, (unsigned long)p_phys, p_virt, size);

		sz_left -= size;
		line++;
		lines--;
	}

	return ECORE_SUCCESS;
}

static enum _ecore_status_t ecore_ilt_shadow_alloc(struct ecore_hwfn *p_hwfn)
{
	struct ecore_cxt_mngr *p_mngr = p_hwfn->p_cxt_mngr;
	struct ecore_ilt_client_cfg *clients = p_mngr->clients;
	struct ecore_ilt_cli_blk *p_blk;
	u32 size, i, j, k;
	enum _ecore_status_t rc;

	size = ecore_cxt_ilt_shadow_size(clients);
	p_mngr->ilt_shadow = OSAL_ZALLOC(p_hwfn->p_dev, GFP_KERNEL,
					 size * sizeof(struct phys_mem_desc));

	if (!p_mngr->ilt_shadow) {
		DP_NOTICE(p_hwfn, false, "Failed to allocate ilt shadow table\n");
		rc = ECORE_NOMEM;
		goto ilt_shadow_fail;
	}

	DP_VERBOSE(p_hwfn, ECORE_MSG_ILT,
		   "Allocated 0x%x bytes for ilt shadow\n",
		   (u32)(size * sizeof(struct phys_mem_desc)));

	for_each_ilt_valid_client(i, clients) {
		for (j = 0; j < ILT_CLI_PF_BLOCKS; j++) {
			p_blk = &clients[i].pf_blks[j];
			rc = ecore_ilt_blk_alloc(p_hwfn, p_blk, i, 0);
			if (rc != ECORE_SUCCESS)
				goto ilt_shadow_fail;
		}
		for (k = 0; k < p_mngr->vf_count; k++) {
			for (j = 0; j < ILT_CLI_VF_BLOCKS; j++) {
				u32 lines = clients[i].vf_total_lines * k;

				p_blk = &clients[i].vf_blks[j];
				rc = ecore_ilt_blk_alloc(p_hwfn, p_blk,
							 i, lines);
				if (rc != ECORE_SUCCESS)
					goto ilt_shadow_fail;
			}
		}
	}

	return ECORE_SUCCESS;

ilt_shadow_fail:
	ecore_ilt_shadow_free(p_hwfn);
	return rc;
}

static void ecore_cid_map_free(struct ecore_hwfn *p_hwfn)
{
	u32 type, vf, max_num_vfs = NUM_OF_VFS(p_hwfn->p_dev);
	struct ecore_cxt_mngr *p_mngr = p_hwfn->p_cxt_mngr;

	for (type = 0; type < MAX_CONN_TYPES; type++) {
		OSAL_FREE(p_hwfn->p_dev, p_mngr->acquired[type].cid_map);
		p_mngr->acquired[type].cid_map = OSAL_NULL;
		p_mngr->acquired[type].max_count = 0;
		p_mngr->acquired[type].start_cid = 0;

		for (vf = 0; vf < max_num_vfs; vf++) {
			OSAL_FREE(p_hwfn->p_dev,
				  p_mngr->acquired_vf[type][vf].cid_map);
			p_mngr->acquired_vf[type][vf].cid_map = OSAL_NULL;
			p_mngr->acquired_vf[type][vf].max_count = 0;
			p_mngr->acquired_vf[type][vf].start_cid = 0;
		}
	}
}

static enum _ecore_status_t
__ecore_cid_map_alloc_single(struct ecore_hwfn *p_hwfn, u32 type,
			   u32 cid_start, u32 cid_count,
			   struct ecore_cid_acquired_map *p_map)
{
	u32 size;

	if (!cid_count)
		return ECORE_SUCCESS;

	size = MAP_WORD_SIZE * DIV_ROUND_UP(cid_count, BITS_PER_MAP_WORD);
	p_map->cid_map = OSAL_ZALLOC(p_hwfn->p_dev, GFP_KERNEL, size);
	if (p_map->cid_map == OSAL_NULL)
		return ECORE_NOMEM;

	p_map->max_count = cid_count;
	p_map->start_cid = cid_start;

	DP_VERBOSE(p_hwfn, ECORE_MSG_CXT,
		   "Type %08x start: %08x count %08x\n",
		   type, p_map->start_cid, p_map->max_count);

	return ECORE_SUCCESS;
}

static enum _ecore_status_t
ecore_cid_map_alloc_single(struct ecore_hwfn *p_hwfn, u32 type, u32 start_cid,
			   u32 vf_start_cid)
{
	struct ecore_cxt_mngr *p_mngr = p_hwfn->p_cxt_mngr;
	u32 vf, max_num_vfs = NUM_OF_VFS(p_hwfn->p_dev);
	struct ecore_cid_acquired_map *p_map;
	struct ecore_conn_type_cfg *p_cfg;
	enum _ecore_status_t rc;

	p_cfg = &p_mngr->conn_cfg[type];

		/* Handle PF maps */
		p_map = &p_mngr->acquired[type];
	rc = __ecore_cid_map_alloc_single(p_hwfn, type, start_cid,
					  p_cfg->cid_count, p_map);
	if (rc != ECORE_SUCCESS)
		return rc;

	/* Handle VF maps */
	for (vf = 0; vf < max_num_vfs; vf++) {
		p_map = &p_mngr->acquired_vf[type][vf];
		rc = __ecore_cid_map_alloc_single(p_hwfn, type, vf_start_cid,
						  p_cfg->cids_per_vf, p_map);
		if (rc != ECORE_SUCCESS)
			return rc;
	}

	return ECORE_SUCCESS;
}

static enum _ecore_status_t ecore_cid_map_alloc(struct ecore_hwfn *p_hwfn)
{
	struct ecore_cxt_mngr *p_mngr = p_hwfn->p_cxt_mngr;
	u32 start_cid = 0, vf_start_cid = 0;
	u32 type;
	enum _ecore_status_t rc;

	for (type = 0; type < MAX_CONN_TYPES; type++) {
		rc = ecore_cid_map_alloc_single(p_hwfn, type, start_cid,
						vf_start_cid);
		if (rc != ECORE_SUCCESS)
			goto cid_map_fail;

		start_cid += p_mngr->conn_cfg[type].cid_count;
		vf_start_cid += p_mngr->conn_cfg[type].cids_per_vf;
	}

	return ECORE_SUCCESS;

cid_map_fail:
	ecore_cid_map_free(p_hwfn);
	return rc;
}

enum _ecore_status_t ecore_cxt_mngr_alloc(struct ecore_hwfn *p_hwfn)
{
	struct ecore_cid_acquired_map *acquired_vf;
	struct ecore_ilt_client_cfg *clients;
	struct ecore_cxt_mngr *p_mngr;
	u32 i, max_num_vfs;

	p_mngr = OSAL_ZALLOC(p_hwfn->p_dev, GFP_KERNEL, sizeof(*p_mngr));
	if (!p_mngr) {
		DP_NOTICE(p_hwfn, false, "Failed to allocate `struct ecore_cxt_mngr'\n");
		return ECORE_NOMEM;
	}

	/* Initialize ILT client registers */
	clients = p_mngr->clients;
	clients[ILT_CLI_CDUC].first.reg = ILT_CFG_REG(CDUC, FIRST_ILT);
	clients[ILT_CLI_CDUC].last.reg  = ILT_CFG_REG(CDUC, LAST_ILT);
	clients[ILT_CLI_CDUC].p_size.reg = ILT_CFG_REG(CDUC, P_SIZE);

	clients[ILT_CLI_QM].first.reg   = ILT_CFG_REG(QM, FIRST_ILT);
	clients[ILT_CLI_QM].last.reg    = ILT_CFG_REG(QM, LAST_ILT);
	clients[ILT_CLI_QM].p_size.reg  = ILT_CFG_REG(QM, P_SIZE);

	clients[ILT_CLI_TM].first.reg   = ILT_CFG_REG(TM, FIRST_ILT);
	clients[ILT_CLI_TM].last.reg    = ILT_CFG_REG(TM, LAST_ILT);
	clients[ILT_CLI_TM].p_size.reg  = ILT_CFG_REG(TM, P_SIZE);

	clients[ILT_CLI_SRC].first.reg  = ILT_CFG_REG(SRC, FIRST_ILT);
	clients[ILT_CLI_SRC].last.reg   = ILT_CFG_REG(SRC, LAST_ILT);
	clients[ILT_CLI_SRC].p_size.reg = ILT_CFG_REG(SRC, P_SIZE);

	clients[ILT_CLI_CDUT].first.reg = ILT_CFG_REG(CDUT, FIRST_ILT);
	clients[ILT_CLI_CDUT].last.reg  = ILT_CFG_REG(CDUT, LAST_ILT);
	clients[ILT_CLI_CDUT].p_size.reg = ILT_CFG_REG(CDUT, P_SIZE);

	clients[ILT_CLI_TSDM].first.reg = ILT_CFG_REG(TSDM, FIRST_ILT);
	clients[ILT_CLI_TSDM].last.reg  = ILT_CFG_REG(TSDM, LAST_ILT);
	clients[ILT_CLI_TSDM].p_size.reg = ILT_CFG_REG(TSDM, P_SIZE);

	/* default ILT page size for all clients is 64K */
	for (i = 0; i < MAX_ILT_CLIENTS; i++)
		p_mngr->clients[i].p_size.val = ILT_DEFAULT_HW_P_SIZE;

	/* due to removal of ISCSI/FCoE files union type0_task_context
	 * task_type_size will be 0. So hardcoded for now.
	 */
	p_mngr->task_type_size[0] = 512; /* @DPDK */
	p_mngr->task_type_size[1] = 128; /* @DPDK */

	if (p_hwfn->p_dev->p_iov_info)
		p_mngr->vf_count = p_hwfn->p_dev->p_iov_info->total_vfs;

	/* Initialize the dynamic ILT allocation mutex */
#ifdef CONFIG_ECORE_LOCK_ALLOC
	if (OSAL_MUTEX_ALLOC(p_hwfn, &p_mngr->mutex)) {
		DP_NOTICE(p_hwfn, false, "Failed to alloc p_mngr->mutex\n");
		return ECORE_NOMEM;
	}
#endif
	OSAL_MUTEX_INIT(&p_mngr->mutex);

	/* Set the cxt mangr pointer prior to further allocations */
	p_hwfn->p_cxt_mngr = p_mngr;

	max_num_vfs = NUM_OF_VFS(p_hwfn->p_dev);
	for (i = 0; i < MAX_CONN_TYPES; i++) {
		acquired_vf = OSAL_CALLOC(p_hwfn->p_dev, GFP_KERNEL,
					  max_num_vfs, sizeof(*acquired_vf));
		if (!acquired_vf) {
			DP_NOTICE(p_hwfn, false,
				  "Failed to allocate an array of `struct ecore_cid_acquired_map'\n");
			return ECORE_NOMEM;
		}

		p_mngr->acquired_vf[i] = acquired_vf;
	}

	return ECORE_SUCCESS;
}

enum _ecore_status_t ecore_cxt_tables_alloc(struct ecore_hwfn *p_hwfn)
{
	enum _ecore_status_t rc;

	/* Allocate the ILT shadow table */
	rc = ecore_ilt_shadow_alloc(p_hwfn);
	if (rc) {
		DP_NOTICE(p_hwfn, false, "Failed to allocate ilt memory\n");
		goto tables_alloc_fail;
	}

	/* Allocate the T2  table */
	rc = ecore_cxt_src_t2_alloc(p_hwfn);
	if (rc) {
		DP_NOTICE(p_hwfn, false, "Failed to allocate T2 memory\n");
		goto tables_alloc_fail;
	}

	/* Allocate and initialize the acquired cids bitmaps */
	rc = ecore_cid_map_alloc(p_hwfn);
	if (rc) {
		DP_NOTICE(p_hwfn, false, "Failed to allocate cid maps\n");
		goto tables_alloc_fail;
	}

	return ECORE_SUCCESS;

tables_alloc_fail:
	ecore_cxt_mngr_free(p_hwfn);
	return rc;
}

void ecore_cxt_mngr_free(struct ecore_hwfn *p_hwfn)
{
	u32 i;

	if (!p_hwfn->p_cxt_mngr)
		return;

	ecore_cid_map_free(p_hwfn);
	ecore_cxt_src_t2_free(p_hwfn);
	ecore_ilt_shadow_free(p_hwfn);
#ifdef CONFIG_ECORE_LOCK_ALLOC
	OSAL_MUTEX_DEALLOC(&p_hwfn->p_cxt_mngr->mutex);
#endif
	for (i = 0; i < MAX_CONN_TYPES; i++)
		OSAL_FREE(p_hwfn->p_dev, p_hwfn->p_cxt_mngr->acquired_vf[i]);
	OSAL_FREE(p_hwfn->p_dev, p_hwfn->p_cxt_mngr);

	p_hwfn->p_cxt_mngr = OSAL_NULL;
}

void ecore_cxt_mngr_setup(struct ecore_hwfn *p_hwfn)
{
	struct ecore_cxt_mngr *p_mngr = p_hwfn->p_cxt_mngr;
	u32 len, max_num_vfs = NUM_OF_VFS(p_hwfn->p_dev);
	struct ecore_cid_acquired_map *p_map;
	struct ecore_conn_type_cfg *p_cfg;
	int type;

	/* Reset acquired cids */
	for (type = 0; type < MAX_CONN_TYPES; type++) {
		u32 vf;

		p_cfg = &p_mngr->conn_cfg[type];
		if (p_cfg->cid_count) {
			p_map = &p_mngr->acquired[type];
			len = DIV_ROUND_UP(p_map->max_count,
					   BITS_PER_MAP_WORD) *
			      MAP_WORD_SIZE;
			OSAL_MEM_ZERO(p_map->cid_map, len);
		}

		if (!p_cfg->cids_per_vf)
			continue;

		for (vf = 0; vf < max_num_vfs; vf++) {
			p_map = &p_mngr->acquired_vf[type][vf];
			len = DIV_ROUND_UP(p_map->max_count,
					   BITS_PER_MAP_WORD) *
			      MAP_WORD_SIZE;
			OSAL_MEM_ZERO(p_map->cid_map, len);
		}
	}
}

/* HW initialization helper (per Block, per phase) */

/* CDU Common */
#define CDUC_CXT_SIZE_SHIFT						\
	CDU_REG_CID_ADDR_PARAMS_CONTEXT_SIZE_SHIFT

#define CDUC_CXT_SIZE_MASK						\
	(CDU_REG_CID_ADDR_PARAMS_CONTEXT_SIZE >> CDUC_CXT_SIZE_SHIFT)

#define CDUC_BLOCK_WASTE_SHIFT						\
	CDU_REG_CID_ADDR_PARAMS_BLOCK_WASTE_SHIFT

#define CDUC_BLOCK_WASTE_MASK						\
	(CDU_REG_CID_ADDR_PARAMS_BLOCK_WASTE >> CDUC_BLOCK_WASTE_SHIFT)

#define CDUC_NCIB_SHIFT							\
	CDU_REG_CID_ADDR_PARAMS_NCIB_SHIFT

#define CDUC_NCIB_MASK							\
	(CDU_REG_CID_ADDR_PARAMS_NCIB >> CDUC_NCIB_SHIFT)

#define CDUT_TYPE0_CXT_SIZE_SHIFT					\
	CDU_REG_SEGMENT0_PARAMS_T0_TID_SIZE_SHIFT

#define CDUT_TYPE0_CXT_SIZE_MASK					\
	(CDU_REG_SEGMENT0_PARAMS_T0_TID_SIZE >>				\
	CDUT_TYPE0_CXT_SIZE_SHIFT)

#define CDUT_TYPE0_BLOCK_WASTE_SHIFT					\
	CDU_REG_SEGMENT0_PARAMS_T0_TID_BLOCK_WASTE_SHIFT

#define CDUT_TYPE0_BLOCK_WASTE_MASK					\
	(CDU_REG_SEGMENT0_PARAMS_T0_TID_BLOCK_WASTE >>			\
	CDUT_TYPE0_BLOCK_WASTE_SHIFT)

#define CDUT_TYPE0_NCIB_SHIFT						\
	CDU_REG_SEGMENT0_PARAMS_T0_NUM_TIDS_IN_BLOCK_SHIFT

#define CDUT_TYPE0_NCIB_MASK						\
	(CDU_REG_SEGMENT0_PARAMS_T0_NUM_TIDS_IN_BLOCK >>		\
	CDUT_TYPE0_NCIB_SHIFT)

#define CDUT_TYPE1_CXT_SIZE_SHIFT					\
	CDU_REG_SEGMENT1_PARAMS_T1_TID_SIZE_SHIFT

#define CDUT_TYPE1_CXT_SIZE_MASK					\
	(CDU_REG_SEGMENT1_PARAMS_T1_TID_SIZE >>				\
	CDUT_TYPE1_CXT_SIZE_SHIFT)

#define CDUT_TYPE1_BLOCK_WASTE_SHIFT					\
	CDU_REG_SEGMENT1_PARAMS_T1_TID_BLOCK_WASTE_SHIFT

#define CDUT_TYPE1_BLOCK_WASTE_MASK					\
	(CDU_REG_SEGMENT1_PARAMS_T1_TID_BLOCK_WASTE >>			\
	CDUT_TYPE1_BLOCK_WASTE_SHIFT)

#define CDUT_TYPE1_NCIB_SHIFT						\
	CDU_REG_SEGMENT1_PARAMS_T1_NUM_TIDS_IN_BLOCK_SHIFT

#define CDUT_TYPE1_NCIB_MASK						\
	(CDU_REG_SEGMENT1_PARAMS_T1_NUM_TIDS_IN_BLOCK >>		\
	CDUT_TYPE1_NCIB_SHIFT)

static void ecore_cdu_init_common(struct ecore_hwfn *p_hwfn)
{
	u32 page_sz, elems_per_page, block_waste, cxt_size, cdu_params = 0;

	/* CDUC - connection configuration */
	page_sz = p_hwfn->p_cxt_mngr->clients[ILT_CLI_CDUC].p_size.val;
	cxt_size = CONN_CXT_SIZE(p_hwfn);
	elems_per_page = ILT_PAGE_IN_BYTES(page_sz) / cxt_size;
	block_waste = ILT_PAGE_IN_BYTES(page_sz) - elems_per_page * cxt_size;

	SET_FIELD(cdu_params, CDUC_CXT_SIZE, cxt_size);
	SET_FIELD(cdu_params, CDUC_BLOCK_WASTE, block_waste);
	SET_FIELD(cdu_params, CDUC_NCIB, elems_per_page);
	STORE_RT_REG(p_hwfn, CDU_REG_CID_ADDR_PARAMS_RT_OFFSET, cdu_params);

	/* CDUT - type-0 tasks configuration */
	page_sz = p_hwfn->p_cxt_mngr->clients[ILT_CLI_CDUT].p_size.val;
	cxt_size = p_hwfn->p_cxt_mngr->task_type_size[0];
	elems_per_page = ILT_PAGE_IN_BYTES(page_sz) / cxt_size;
	block_waste = ILT_PAGE_IN_BYTES(page_sz) - elems_per_page * cxt_size;

	/* cxt size and block-waste are multipes of 8 */
	cdu_params = 0;
	SET_FIELD(cdu_params, CDUT_TYPE0_CXT_SIZE, (cxt_size >> 3));
	SET_FIELD(cdu_params, CDUT_TYPE0_BLOCK_WASTE, (block_waste >> 3));
	SET_FIELD(cdu_params, CDUT_TYPE0_NCIB, elems_per_page);
	STORE_RT_REG(p_hwfn, CDU_REG_SEGMENT0_PARAMS_RT_OFFSET, cdu_params);

	/* CDUT - type-1 tasks configuration */
	cxt_size = p_hwfn->p_cxt_mngr->task_type_size[1];
	elems_per_page = ILT_PAGE_IN_BYTES(page_sz) / cxt_size;
	block_waste = ILT_PAGE_IN_BYTES(page_sz) - elems_per_page * cxt_size;

	/* cxt size and block-waste are multipes of 8 */
	cdu_params = 0;
	SET_FIELD(cdu_params, CDUT_TYPE1_CXT_SIZE, (cxt_size >> 3));
	SET_FIELD(cdu_params, CDUT_TYPE1_BLOCK_WASTE, (block_waste >> 3));
	SET_FIELD(cdu_params, CDUT_TYPE1_NCIB, elems_per_page);
	STORE_RT_REG(p_hwfn, CDU_REG_SEGMENT1_PARAMS_RT_OFFSET, cdu_params);
}

/* CDU PF */
#define CDU_SEG_REG_TYPE_SHIFT		CDU_SEG_TYPE_OFFSET_REG_TYPE_SHIFT
#define CDU_SEG_REG_TYPE_MASK		0x1
#define CDU_SEG_REG_OFFSET_SHIFT	0
#define CDU_SEG_REG_OFFSET_MASK		CDU_SEG_TYPE_OFFSET_REG_OFFSET_MASK

static void ecore_cdu_init_pf(struct ecore_hwfn *p_hwfn)
{
	struct ecore_ilt_client_cfg *p_cli;
	struct ecore_tid_seg *p_seg;
	u32 cdu_seg_params, offset;
	int i;

	static const u32 rt_type_offset_arr[] = {
		CDU_REG_PF_SEG0_TYPE_OFFSET_RT_OFFSET,
		CDU_REG_PF_SEG1_TYPE_OFFSET_RT_OFFSET,
		CDU_REG_PF_SEG2_TYPE_OFFSET_RT_OFFSET,
		CDU_REG_PF_SEG3_TYPE_OFFSET_RT_OFFSET
	};

	static const u32 rt_type_offset_fl_arr[] = {
		CDU_REG_PF_FL_SEG0_TYPE_OFFSET_RT_OFFSET,
		CDU_REG_PF_FL_SEG1_TYPE_OFFSET_RT_OFFSET,
		CDU_REG_PF_FL_SEG2_TYPE_OFFSET_RT_OFFSET,
		CDU_REG_PF_FL_SEG3_TYPE_OFFSET_RT_OFFSET
	};

	p_cli = &p_hwfn->p_cxt_mngr->clients[ILT_CLI_CDUT];

	/* There are initializations only for CDUT during pf Phase */
	for (i = 0; i < NUM_TASK_PF_SEGMENTS; i++) {
		/* Segment 0 */
		p_seg = ecore_cxt_tid_seg_info(p_hwfn, i);
		if (!p_seg)
			continue;

		/* Note: start_line is already adjusted for the CDU
		 * segment register granularity, so we just need to
		 * divide. Adjustment is implicit as we assume ILT
		 * Page size is larger than 32K!
		 */
		offset = (ILT_PAGE_IN_BYTES(p_cli->p_size.val) *
			  (p_cli->pf_blks[CDUT_SEG_BLK(i)].start_line -
			   p_cli->first.val)) / CDUT_SEG_ALIGNMET_IN_BYTES;

		cdu_seg_params = 0;
		SET_FIELD(cdu_seg_params, CDU_SEG_REG_TYPE, p_seg->type);
		SET_FIELD(cdu_seg_params, CDU_SEG_REG_OFFSET, offset);
		STORE_RT_REG(p_hwfn, rt_type_offset_arr[i], cdu_seg_params);

		offset = (ILT_PAGE_IN_BYTES(p_cli->p_size.val) *
			  (p_cli->pf_blks[CDUT_FL_SEG_BLK(i, PF)].start_line -
			   p_cli->first.val)) / CDUT_SEG_ALIGNMET_IN_BYTES;

		cdu_seg_params = 0;
		SET_FIELD(cdu_seg_params, CDU_SEG_REG_TYPE, p_seg->type);
		SET_FIELD(cdu_seg_params, CDU_SEG_REG_OFFSET, offset);
		STORE_RT_REG(p_hwfn, rt_type_offset_fl_arr[i], cdu_seg_params);
	}
}

void ecore_qm_init_pf(struct ecore_hwfn *p_hwfn, struct ecore_ptt *p_ptt,
		      bool is_pf_loading)
{
	struct ecore_qm_info *qm_info = &p_hwfn->qm_info;
	struct ecore_qm_iids iids;

	OSAL_MEM_ZERO(&iids, sizeof(iids));
	ecore_cxt_qm_iids(p_hwfn, &iids);
	ecore_qm_pf_rt_init(p_hwfn, p_ptt, p_hwfn->rel_pf_id,
			    qm_info->max_phys_tcs_per_port,
			    is_pf_loading,
			    iids.cids, iids.vf_cids, iids.tids,
			    qm_info->start_pq,
			    qm_info->num_pqs - qm_info->num_vf_pqs,
			    qm_info->num_vf_pqs,
			    qm_info->start_vport,
			    qm_info->num_vports, qm_info->pf_wfq,
			    qm_info->pf_rl,
			    p_hwfn->qm_info.qm_pq_params,
			    p_hwfn->qm_info.qm_vport_params);
}

/* CM PF */
static void ecore_cm_init_pf(struct ecore_hwfn *p_hwfn)
{
	STORE_RT_REG(p_hwfn, XCM_REG_CON_PHY_Q3_RT_OFFSET,
		     ecore_get_cm_pq_idx(p_hwfn, PQ_FLAGS_LB));
}

/* DQ PF */
static void ecore_dq_init_pf(struct ecore_hwfn *p_hwfn)
{
	struct ecore_cxt_mngr *p_mngr = p_hwfn->p_cxt_mngr;
	u32 dq_pf_max_cid = 0, dq_vf_max_cid = 0;

	dq_pf_max_cid += (p_mngr->conn_cfg[0].cid_count >> DQ_RANGE_SHIFT);
	STORE_RT_REG(p_hwfn, DORQ_REG_PF_MAX_ICID_0_RT_OFFSET, dq_pf_max_cid);

	dq_vf_max_cid += (p_mngr->conn_cfg[0].cids_per_vf >> DQ_RANGE_SHIFT);
	STORE_RT_REG(p_hwfn, DORQ_REG_VF_MAX_ICID_0_RT_OFFSET, dq_vf_max_cid);

	dq_pf_max_cid += (p_mngr->conn_cfg[1].cid_count >> DQ_RANGE_SHIFT);
	STORE_RT_REG(p_hwfn, DORQ_REG_PF_MAX_ICID_1_RT_OFFSET, dq_pf_max_cid);

	dq_vf_max_cid += (p_mngr->conn_cfg[1].cids_per_vf >> DQ_RANGE_SHIFT);
	STORE_RT_REG(p_hwfn, DORQ_REG_VF_MAX_ICID_1_RT_OFFSET, dq_vf_max_cid);

	dq_pf_max_cid += (p_mngr->conn_cfg[2].cid_count >> DQ_RANGE_SHIFT);
	STORE_RT_REG(p_hwfn, DORQ_REG_PF_MAX_ICID_2_RT_OFFSET, dq_pf_max_cid);

	dq_vf_max_cid += (p_mngr->conn_cfg[2].cids_per_vf >> DQ_RANGE_SHIFT);
	STORE_RT_REG(p_hwfn, DORQ_REG_VF_MAX_ICID_2_RT_OFFSET, dq_vf_max_cid);

	dq_pf_max_cid += (p_mngr->conn_cfg[3].cid_count >> DQ_RANGE_SHIFT);
	STORE_RT_REG(p_hwfn, DORQ_REG_PF_MAX_ICID_3_RT_OFFSET, dq_pf_max_cid);

	dq_vf_max_cid += (p_mngr->conn_cfg[3].cids_per_vf >> DQ_RANGE_SHIFT);
	STORE_RT_REG(p_hwfn, DORQ_REG_VF_MAX_ICID_3_RT_OFFSET, dq_vf_max_cid);

	dq_pf_max_cid += (p_mngr->conn_cfg[4].cid_count >> DQ_RANGE_SHIFT);
	STORE_RT_REG(p_hwfn, DORQ_REG_PF_MAX_ICID_4_RT_OFFSET, dq_pf_max_cid);

	dq_vf_max_cid += (p_mngr->conn_cfg[4].cids_per_vf >> DQ_RANGE_SHIFT);
	STORE_RT_REG(p_hwfn, DORQ_REG_VF_MAX_ICID_4_RT_OFFSET, dq_vf_max_cid);

	dq_pf_max_cid += (p_mngr->conn_cfg[5].cid_count >> DQ_RANGE_SHIFT);
	STORE_RT_REG(p_hwfn, DORQ_REG_PF_MAX_ICID_5_RT_OFFSET, dq_pf_max_cid);

	dq_vf_max_cid += (p_mngr->conn_cfg[5].cids_per_vf >> DQ_RANGE_SHIFT);
	STORE_RT_REG(p_hwfn, DORQ_REG_VF_MAX_ICID_5_RT_OFFSET, dq_vf_max_cid);

	/* Connection types 6 & 7 are not in use, yet they must be configured
	 * as the highest possible connection. Not configuring them means the
	 * defaults will be  used, and with a large number of cids a bug may
	 * occur, if the defaults will be smaller than dq_pf_max_cid /
	 * dq_vf_max_cid.
	 */
	STORE_RT_REG(p_hwfn, DORQ_REG_PF_MAX_ICID_6_RT_OFFSET, dq_pf_max_cid);
	STORE_RT_REG(p_hwfn, DORQ_REG_VF_MAX_ICID_6_RT_OFFSET, dq_vf_max_cid);

	STORE_RT_REG(p_hwfn, DORQ_REG_PF_MAX_ICID_7_RT_OFFSET, dq_pf_max_cid);
	STORE_RT_REG(p_hwfn, DORQ_REG_VF_MAX_ICID_7_RT_OFFSET, dq_vf_max_cid);
}

static void ecore_ilt_bounds_init(struct ecore_hwfn *p_hwfn)
{
	struct ecore_ilt_client_cfg *ilt_clients;
	int i;

	ilt_clients = p_hwfn->p_cxt_mngr->clients;
	for_each_ilt_valid_client(i, ilt_clients) {
		STORE_RT_REG(p_hwfn,
			     ilt_clients[i].first.reg,
			     ilt_clients[i].first.val);
		STORE_RT_REG(p_hwfn,
			     ilt_clients[i].last.reg, ilt_clients[i].last.val);
		STORE_RT_REG(p_hwfn,
			     ilt_clients[i].p_size.reg,
			     ilt_clients[i].p_size.val);
	}
}

static void ecore_ilt_vf_bounds_init(struct ecore_hwfn *p_hwfn)
{
	struct ecore_ilt_client_cfg *p_cli;
	u32 blk_factor;

	/* For simplicty  we set the 'block' to be an ILT page */
	if (p_hwfn->p_dev->p_iov_info) {
		struct ecore_hw_sriov_info *p_iov = p_hwfn->p_dev->p_iov_info;

		STORE_RT_REG(p_hwfn,
			     PSWRQ2_REG_VF_BASE_RT_OFFSET,
			     p_iov->first_vf_in_pf);
		STORE_RT_REG(p_hwfn,
			     PSWRQ2_REG_VF_LAST_ILT_RT_OFFSET,
			     p_iov->first_vf_in_pf + p_iov->total_vfs);
	}

	p_cli = &p_hwfn->p_cxt_mngr->clients[ILT_CLI_CDUC];
	blk_factor = OSAL_LOG2(ILT_PAGE_IN_BYTES(p_cli->p_size.val) >> 10);
	if (p_cli->active) {
		STORE_RT_REG(p_hwfn,
			     PSWRQ2_REG_CDUC_BLOCKS_FACTOR_RT_OFFSET,
			     blk_factor);
		STORE_RT_REG(p_hwfn,
			     PSWRQ2_REG_CDUC_NUMBER_OF_PF_BLOCKS_RT_OFFSET,
			     p_cli->pf_total_lines);
		STORE_RT_REG(p_hwfn,
			     PSWRQ2_REG_CDUC_VF_BLOCKS_RT_OFFSET,
			     p_cli->vf_total_lines);
	}

	p_cli = &p_hwfn->p_cxt_mngr->clients[ILT_CLI_CDUT];
	blk_factor = OSAL_LOG2(ILT_PAGE_IN_BYTES(p_cli->p_size.val) >> 10);
	if (p_cli->active) {
		STORE_RT_REG(p_hwfn,
			     PSWRQ2_REG_CDUT_BLOCKS_FACTOR_RT_OFFSET,
			     blk_factor);
		STORE_RT_REG(p_hwfn,
			     PSWRQ2_REG_CDUT_NUMBER_OF_PF_BLOCKS_RT_OFFSET,
			     p_cli->pf_total_lines);
		STORE_RT_REG(p_hwfn,
			     PSWRQ2_REG_CDUT_VF_BLOCKS_RT_OFFSET,
			     p_cli->vf_total_lines);
	}

	p_cli = &p_hwfn->p_cxt_mngr->clients[ILT_CLI_TM];
	blk_factor = OSAL_LOG2(ILT_PAGE_IN_BYTES(p_cli->p_size.val) >> 10);
	if (p_cli->active) {
		STORE_RT_REG(p_hwfn,
			     PSWRQ2_REG_TM_BLOCKS_FACTOR_RT_OFFSET, blk_factor);
		STORE_RT_REG(p_hwfn,
			     PSWRQ2_REG_TM_NUMBER_OF_PF_BLOCKS_RT_OFFSET,
			     p_cli->pf_total_lines);
		STORE_RT_REG(p_hwfn,
			     PSWRQ2_REG_TM_VF_BLOCKS_RT_OFFSET,
			     p_cli->vf_total_lines);
	}
}

/* ILT (PSWRQ2) PF */
static void ecore_ilt_init_pf(struct ecore_hwfn *p_hwfn)
{
	struct ecore_ilt_client_cfg *clients;
	struct ecore_cxt_mngr *p_mngr;
	struct phys_mem_desc *p_shdw;
	u32 line, rt_offst, i;

	ecore_ilt_bounds_init(p_hwfn);
	ecore_ilt_vf_bounds_init(p_hwfn);

	p_mngr = p_hwfn->p_cxt_mngr;
	p_shdw = p_mngr->ilt_shadow;
	clients = p_hwfn->p_cxt_mngr->clients;

	for_each_ilt_valid_client(i, clients) {
		/* Client's 1st val and RT array are absolute, ILT shadows'
		 * lines are relative.
		 */
		line = clients[i].first.val - p_mngr->pf_start_line;
		rt_offst = PSWRQ2_REG_ILT_MEMORY_RT_OFFSET +
		    clients[i].first.val * ILT_ENTRY_IN_REGS;

		for (; line <= clients[i].last.val - p_mngr->pf_start_line;
		     line++, rt_offst += ILT_ENTRY_IN_REGS) {
			u64 ilt_hw_entry = 0;

			/** p_virt could be OSAL_NULL incase of dynamic
			 *  allocation
			 */
			if (p_shdw[line].virt_addr != OSAL_NULL) {
				SET_FIELD(ilt_hw_entry, ILT_ENTRY_VALID, 1ULL);
				SET_FIELD(ilt_hw_entry, ILT_ENTRY_PHY_ADDR,
					  (p_shdw[line].phys_addr >> 12));

				DP_VERBOSE(p_hwfn, ECORE_MSG_ILT,
					"Setting RT[0x%08x] from"
					" ILT[0x%08x] [Client is %d] to"
					" Physical addr: 0x%lx\n",
					rt_offst, line, i,
					(unsigned long)(p_shdw[line].
							phys_addr >> 12));
			}

			STORE_RT_REG_AGG(p_hwfn, rt_offst, ilt_hw_entry);
		}
	}
}

/* SRC (Searcher) PF */
static void ecore_src_init_pf(struct ecore_hwfn *p_hwfn)
{
	struct ecore_cxt_mngr *p_mngr = p_hwfn->p_cxt_mngr;
	u32 rounded_conn_num, conn_num, conn_max;
	struct ecore_src_iids src_iids;

	OSAL_MEM_ZERO(&src_iids, sizeof(src_iids));
	ecore_cxt_src_iids(p_mngr, &src_iids);
	conn_num = src_iids.pf_cids + src_iids.per_vf_cids * p_mngr->vf_count;
	if (!conn_num)
		return;

	conn_max = OSAL_MAX_T(u32, conn_num, SRC_MIN_NUM_ELEMS);
	rounded_conn_num = OSAL_ROUNDUP_POW_OF_TWO(conn_max);

	STORE_RT_REG(p_hwfn, SRC_REG_COUNTFREE_RT_OFFSET, conn_num);
	STORE_RT_REG(p_hwfn, SRC_REG_NUMBER_HASH_BITS_RT_OFFSET,
		     OSAL_LOG2(rounded_conn_num));

	STORE_RT_REG_AGG(p_hwfn, SRC_REG_FIRSTFREE_RT_OFFSET,
			 p_hwfn->p_cxt_mngr->src_t2.first_free);
	STORE_RT_REG_AGG(p_hwfn, SRC_REG_LASTFREE_RT_OFFSET,
			 p_hwfn->p_cxt_mngr->src_t2.last_free);
	DP_VERBOSE(p_hwfn, ECORE_MSG_ILT,
		   "Configured SEARCHER for 0x%08x connections\n",
		   conn_num);
}

/* Timers PF */
#define TM_CFG_NUM_IDS_SHIFT		0
#define TM_CFG_NUM_IDS_MASK		0xFFFFULL
#define TM_CFG_PRE_SCAN_OFFSET_SHIFT	16
#define TM_CFG_PRE_SCAN_OFFSET_MASK	0x1FFULL
#define TM_CFG_PARENT_PF_SHIFT		25
#define TM_CFG_PARENT_PF_MASK		0x7ULL

#define TM_CFG_CID_PRE_SCAN_ROWS_SHIFT	30
#define TM_CFG_CID_PRE_SCAN_ROWS_MASK	0x1FFULL

#define TM_CFG_TID_OFFSET_SHIFT		30
#define TM_CFG_TID_OFFSET_MASK		0x7FFFFULL
#define TM_CFG_TID_PRE_SCAN_ROWS_SHIFT	49
#define TM_CFG_TID_PRE_SCAN_ROWS_MASK	0x1FFULL

static void ecore_tm_init_pf(struct ecore_hwfn *p_hwfn)
{
	struct ecore_cxt_mngr *p_mngr = p_hwfn->p_cxt_mngr;
	u32 active_seg_mask = 0, tm_offset, rt_reg;
	struct ecore_tm_iids tm_iids;
	u64 cfg_word;
	u8 i;

	OSAL_MEM_ZERO(&tm_iids, sizeof(tm_iids));
	ecore_cxt_tm_iids(p_hwfn, p_mngr, &tm_iids);

	/* @@@TBD No pre-scan for now */

		cfg_word = 0;
		SET_FIELD(cfg_word, TM_CFG_NUM_IDS, tm_iids.per_vf_cids);
		SET_FIELD(cfg_word, TM_CFG_PARENT_PF, p_hwfn->rel_pf_id);
	SET_FIELD(cfg_word, TM_CFG_PRE_SCAN_OFFSET, 0);
		SET_FIELD(cfg_word, TM_CFG_CID_PRE_SCAN_ROWS, 0); /* scan all */

	/* Note: We assume conselwtive VFs for a PF */
	for (i = 0; i < p_mngr->vf_count; i++) {
		rt_reg = TM_REG_CONFIG_CONN_MEM_RT_OFFSET +
		    (sizeof(cfg_word) / sizeof(u32)) *
		    (p_hwfn->p_dev->p_iov_info->first_vf_in_pf + i);
		STORE_RT_REG_AGG(p_hwfn, rt_reg, cfg_word);
	}

	cfg_word = 0;
	SET_FIELD(cfg_word, TM_CFG_NUM_IDS, tm_iids.pf_cids);
	SET_FIELD(cfg_word, TM_CFG_PRE_SCAN_OFFSET, 0);
	SET_FIELD(cfg_word, TM_CFG_PARENT_PF, 0);	/* n/a for PF */
	SET_FIELD(cfg_word, TM_CFG_CID_PRE_SCAN_ROWS, 0); /* scan all   */

	rt_reg = TM_REG_CONFIG_CONN_MEM_RT_OFFSET +
	    (sizeof(cfg_word) / sizeof(u32)) *
	    (NUM_OF_VFS(p_hwfn->p_dev) + p_hwfn->rel_pf_id);
	STORE_RT_REG_AGG(p_hwfn, rt_reg, cfg_word);

	/* enable scan */
	STORE_RT_REG(p_hwfn, TM_REG_PF_ENABLE_CONN_RT_OFFSET,
		     tm_iids.pf_cids ? 0x1 : 0x0);

	/* @@@TBD how to enable the scan for the VFs */

	tm_offset = tm_iids.per_vf_cids;

	/* Note: We assume conselwtive VFs for a PF */
	for (i = 0; i < p_mngr->vf_count; i++) {
		cfg_word = 0;
		SET_FIELD(cfg_word, TM_CFG_NUM_IDS, tm_iids.per_vf_tids);
		SET_FIELD(cfg_word, TM_CFG_PRE_SCAN_OFFSET, 0);
		SET_FIELD(cfg_word, TM_CFG_PARENT_PF, p_hwfn->rel_pf_id);
		SET_FIELD(cfg_word, TM_CFG_TID_OFFSET, tm_offset);
		SET_FIELD(cfg_word, TM_CFG_TID_PRE_SCAN_ROWS, (u64)0);

		rt_reg = TM_REG_CONFIG_TASK_MEM_RT_OFFSET +
		    (sizeof(cfg_word) / sizeof(u32)) *
		    (p_hwfn->p_dev->p_iov_info->first_vf_in_pf + i);

		STORE_RT_REG_AGG(p_hwfn, rt_reg, cfg_word);
	}

	tm_offset = tm_iids.pf_cids;
	for (i = 0; i < NUM_TASK_PF_SEGMENTS; i++) {
		cfg_word = 0;
		SET_FIELD(cfg_word, TM_CFG_NUM_IDS, tm_iids.pf_tids[i]);
		SET_FIELD(cfg_word, TM_CFG_PRE_SCAN_OFFSET, 0);
		SET_FIELD(cfg_word, TM_CFG_PARENT_PF, 0);
		SET_FIELD(cfg_word, TM_CFG_TID_OFFSET, tm_offset);
		SET_FIELD(cfg_word, TM_CFG_TID_PRE_SCAN_ROWS, (u64)0);

		rt_reg = TM_REG_CONFIG_TASK_MEM_RT_OFFSET +
		    (sizeof(cfg_word) / sizeof(u32)) *
		    (NUM_OF_VFS(p_hwfn->p_dev) +
		     p_hwfn->rel_pf_id * NUM_TASK_PF_SEGMENTS + i);

		STORE_RT_REG_AGG(p_hwfn, rt_reg, cfg_word);
		active_seg_mask |= (tm_iids.pf_tids[i] ? (1 << i) : 0);

		tm_offset += tm_iids.pf_tids[i];
	}

	STORE_RT_REG(p_hwfn, TM_REG_PF_ENABLE_TASK_RT_OFFSET, active_seg_mask);

	/* @@@TBD how to enable the scan for the VFs */
}

static void ecore_prs_init_pf(struct ecore_hwfn *p_hwfn)
{
	struct ecore_cxt_mngr *p_mngr = p_hwfn->p_cxt_mngr;
	struct ecore_conn_type_cfg *p_fcoe;
	struct ecore_tid_seg *p_tid;

	p_fcoe = &p_mngr->conn_cfg[PROTOCOLID_FCOE];

	/* If FCoE is active set the MAX OX_ID (tid) in the Parser */
	if (!p_fcoe->cid_count)
		return;

	p_tid = &p_fcoe->tid_seg[ECORE_CXT_FCOE_TID_SEG];
	STORE_RT_REG_AGG(p_hwfn,
			PRS_REG_TASK_ID_MAX_INITIATOR_PF_RT_OFFSET,
			p_tid->count);
}

void ecore_cxt_hw_init_common(struct ecore_hwfn *p_hwfn)
{
	/* CDU configuration */
	ecore_cdu_init_common(p_hwfn);
}

void ecore_cxt_hw_init_pf(struct ecore_hwfn *p_hwfn, struct ecore_ptt *p_ptt)
{
	ecore_qm_init_pf(p_hwfn, p_ptt, true);
	ecore_cm_init_pf(p_hwfn);
	ecore_dq_init_pf(p_hwfn);
	ecore_cdu_init_pf(p_hwfn);
	ecore_ilt_init_pf(p_hwfn);
	ecore_src_init_pf(p_hwfn);
	ecore_tm_init_pf(p_hwfn);
	ecore_prs_init_pf(p_hwfn);
}

enum _ecore_status_t _ecore_cxt_acquire_cid(struct ecore_hwfn *p_hwfn,
					    enum protocol_type type,
					    u32 *p_cid, u8 vfid)
{
	u32 rel_cid, max_num_vfs = NUM_OF_VFS(p_hwfn->p_dev);
	struct ecore_cxt_mngr *p_mngr = p_hwfn->p_cxt_mngr;
	struct ecore_cid_acquired_map *p_map;

	if (type >= MAX_CONN_TYPES) {
		DP_NOTICE(p_hwfn, true, "Invalid protocol type %d", type);
		return ECORE_ILWAL;
	}

	if (vfid >= max_num_vfs && vfid != ECORE_CXT_PF_CID) {
		DP_NOTICE(p_hwfn, true, "VF [%02x] is out of range\n", vfid);
		return ECORE_ILWAL;
	}

	/* Determine the right map to take this CID from */
	if (vfid == ECORE_CXT_PF_CID)
		p_map = &p_mngr->acquired[type];
	else
		p_map = &p_mngr->acquired_vf[type][vfid];

	if (p_map->cid_map == OSAL_NULL) {
		DP_NOTICE(p_hwfn, true, "Invalid protocol type %d", type);
		return ECORE_ILWAL;
	}

	rel_cid = OSAL_FIND_FIRST_ZERO_BIT(p_map->cid_map,
					   p_map->max_count);

	if (rel_cid >= p_map->max_count) {
		DP_NOTICE(p_hwfn, false, "no CID available for protocol %d\n",
			  type);
		return ECORE_NORESOURCES;
	}

	OSAL_SET_BIT(rel_cid, p_map->cid_map);

	*p_cid = rel_cid + p_map->start_cid;

	DP_VERBOSE(p_hwfn, ECORE_MSG_CXT,
		   "Acquired cid 0x%08x [rel. %08x] vfid %02x type %d\n",
		   *p_cid, rel_cid, vfid, type);

	return ECORE_SUCCESS;
}

enum _ecore_status_t ecore_cxt_acquire_cid(struct ecore_hwfn *p_hwfn,
					   enum protocol_type type,
					   u32 *p_cid)
{
	return _ecore_cxt_acquire_cid(p_hwfn, type, p_cid, ECORE_CXT_PF_CID);
}

static bool ecore_cxt_test_cid_acquired(struct ecore_hwfn *p_hwfn,
					u32 cid, u8 vfid,
					enum protocol_type *p_type,
					struct ecore_cid_acquired_map **pp_map)
{
	struct ecore_cxt_mngr *p_mngr = p_hwfn->p_cxt_mngr;
	u32 rel_cid;

	/* Iterate over protocols and find matching cid range */
	for (*p_type = 0; *p_type < MAX_CONN_TYPES; (*p_type)++) {
		if (vfid == ECORE_CXT_PF_CID)
			*pp_map = &p_mngr->acquired[*p_type];
		else
			*pp_map = &p_mngr->acquired_vf[*p_type][vfid];

		if (!((*pp_map)->cid_map))
			continue;
		if (cid >= (*pp_map)->start_cid &&
		    cid < (*pp_map)->start_cid + (*pp_map)->max_count) {
			break;
		}
	}
	if (*p_type == MAX_CONN_TYPES) {
		DP_NOTICE(p_hwfn, true, "Invalid CID %d vfid %02x", cid, vfid);
		goto fail;
	}

	rel_cid = cid - (*pp_map)->start_cid;
	if (!OSAL_GET_BIT(rel_cid, (*pp_map)->cid_map)) {
		DP_NOTICE(p_hwfn, true,
			  "CID %d [vifd %02x] not acquired", cid, vfid);
		goto fail;
	}

	return true;
fail:
	*p_type = MAX_CONN_TYPES;
	*pp_map = OSAL_NULL;
	return false;
}

void _ecore_cxt_release_cid(struct ecore_hwfn *p_hwfn, u32 cid, u8 vfid)
{
	u32 rel_cid, max_num_vfs = NUM_OF_VFS(p_hwfn->p_dev);
	struct ecore_cid_acquired_map *p_map = OSAL_NULL;
	enum protocol_type type;
	bool b_acquired;

	if (vfid != ECORE_CXT_PF_CID && vfid > max_num_vfs) {
		DP_NOTICE(p_hwfn, true,
			  "Trying to return incorrect CID belonging to VF %02x\n",
			  vfid);
		return;
	}

	/* Test acquired and find matching per-protocol map */
	b_acquired = ecore_cxt_test_cid_acquired(p_hwfn, cid, vfid,
						 &type, &p_map);

	if (!b_acquired)
		return;

	rel_cid = cid - p_map->start_cid;
	OSAL_CLEAR_BIT(rel_cid, p_map->cid_map);

	DP_VERBOSE(p_hwfn, ECORE_MSG_CXT,
		   "Released CID 0x%08x [rel. %08x] vfid %02x type %d\n",
		   cid, rel_cid, vfid, type);
}

void ecore_cxt_release_cid(struct ecore_hwfn *p_hwfn, u32 cid)
{
	_ecore_cxt_release_cid(p_hwfn, cid, ECORE_CXT_PF_CID);
}

enum _ecore_status_t ecore_cxt_get_cid_info(struct ecore_hwfn *p_hwfn,
					    struct ecore_cxt_info *p_info)
{
	struct ecore_cxt_mngr *p_mngr = p_hwfn->p_cxt_mngr;
	struct ecore_cid_acquired_map *p_map = OSAL_NULL;
	u32 conn_cxt_size, hw_p_size, cxts_per_p, line;
	enum protocol_type type;
	bool b_acquired;

	/* Test acquired and find matching per-protocol map */
	b_acquired = ecore_cxt_test_cid_acquired(p_hwfn, p_info->iid,
						 ECORE_CXT_PF_CID,
						 &type, &p_map);

	if (!b_acquired)
		return ECORE_ILWAL;

	/* set the protocl type */
	p_info->type = type;

	/* compute context virtual pointer */
	hw_p_size = p_hwfn->p_cxt_mngr->clients[ILT_CLI_CDUC].p_size.val;

	conn_cxt_size = CONN_CXT_SIZE(p_hwfn);
	cxts_per_p = ILT_PAGE_IN_BYTES(hw_p_size) / conn_cxt_size;
	line = p_info->iid / cxts_per_p;

	/* Make sure context is allocated (dynamic allocation) */
	if (!p_mngr->ilt_shadow[line].virt_addr)
		return ECORE_ILWAL;

	p_info->p_cxt = (u8 *)p_mngr->ilt_shadow[line].virt_addr +
	    p_info->iid % cxts_per_p * conn_cxt_size;

	DP_VERBOSE(p_hwfn, (ECORE_MSG_ILT | ECORE_MSG_CXT),
		"Accessing ILT shadow[%d]: CXT pointer is at %p (for iid %d)\n",
		(p_info->iid / cxts_per_p), p_info->p_cxt, p_info->iid);

	return ECORE_SUCCESS;
}

enum _ecore_status_t ecore_cxt_set_pf_params(struct ecore_hwfn *p_hwfn)
{
	/* Set the number of required CORE connections */
	u32 core_cids = 1;	/* SPQ */

	ecore_cxt_set_proto_cid_count(p_hwfn, PROTOCOLID_CORE, core_cids, 0);

	switch (p_hwfn->hw_info.personality) {
	case ECORE_PCI_ETH:
		{
		u32 count = 0;

		struct ecore_eth_pf_params *p_params =
			    &p_hwfn->pf_params.eth_pf_params;

		if (!p_params->num_vf_cons)
			p_params->num_vf_cons = ETH_PF_PARAMS_VF_CONS_DEFAULT;
		ecore_cxt_set_proto_cid_count(p_hwfn, PROTOCOLID_ETH,
					      p_params->num_cons,
					      p_params->num_vf_cons);

		count = p_params->num_arfs_filters;

		if (!OSAL_GET_BIT(ECORE_MF_DISABLE_ARFS,
				   &p_hwfn->p_dev->mf_bits))
			p_hwfn->p_cxt_mngr->arfs_count = count;

		break;
		}
	default:
		return ECORE_ILWAL;
	}

	return ECORE_SUCCESS;
}

/* This function is very RoCE oriented, if another protocol in the future
 * will want this feature we'll need to modify the function to be more generic
 */
enum _ecore_status_t
ecore_cxt_dynamic_ilt_alloc(struct ecore_hwfn *p_hwfn,
			    enum ecore_cxt_elem_type elem_type,
			    u32 iid)
{
	u32 reg_offset, shadow_line, elem_size, hw_p_size, elems_per_p, line;
	struct ecore_ilt_client_cfg *p_cli;
	struct ecore_ilt_cli_blk *p_blk;
	struct ecore_ptt *p_ptt;
	dma_addr_t p_phys;
	u64 ilt_hw_entry;
	void *p_virt;
	enum _ecore_status_t rc = ECORE_SUCCESS;

	switch (elem_type) {
	case ECORE_ELEM_CXT:
		p_cli = &p_hwfn->p_cxt_mngr->clients[ILT_CLI_CDUC];
		elem_size = CONN_CXT_SIZE(p_hwfn);
		p_blk = &p_cli->pf_blks[CDUC_BLK];
		break;
	case ECORE_ELEM_SRQ:
		p_cli = &p_hwfn->p_cxt_mngr->clients[ILT_CLI_TSDM];
		elem_size = SRQ_CXT_SIZE;
		p_blk = &p_cli->pf_blks[SRQ_BLK];
		break;
	case ECORE_ELEM_TASK:
		p_cli = &p_hwfn->p_cxt_mngr->clients[ILT_CLI_CDUT];
		elem_size = TYPE1_TASK_CXT_SIZE(p_hwfn);
		p_blk = &p_cli->pf_blks[CDUT_SEG_BLK(ECORE_CXT_ROCE_TID_SEG)];
		break;
	default:
		DP_NOTICE(p_hwfn, false,
			  "ECORE_ILWALID elem type = %d", elem_type);
		return ECORE_ILWAL;
	}

	/* Callwlate line in ilt */
	hw_p_size = p_cli->p_size.val;
	elems_per_p = ILT_PAGE_IN_BYTES(hw_p_size) / elem_size;
	line = p_blk->start_line + (iid / elems_per_p);
	shadow_line = line - p_hwfn->p_cxt_mngr->pf_start_line;

	/* If line is already allocated, do nothing, otherwise allocate it and
	 * write it to the PSWRQ2 registers.
	 * This section can be run in parallel from different contexts and thus
	 * a mutex protection is needed.
	 */

	OSAL_MUTEX_ACQUIRE(&p_hwfn->p_cxt_mngr->mutex);

	if (p_hwfn->p_cxt_mngr->ilt_shadow[shadow_line].virt_addr)
		goto out0;

	p_ptt = ecore_ptt_acquire(p_hwfn);
	if (!p_ptt) {
		DP_NOTICE(p_hwfn, false,
			  "ECORE_TIME_OUT on ptt acquire - dynamic allocation");
		rc = ECORE_TIMEOUT;
		goto out0;
	}

	p_virt = OSAL_DMA_ALLOC_COHERENT(p_hwfn->p_dev,
					 &p_phys,
					 p_blk->real_size_in_page);
	if (!p_virt) {
		rc = ECORE_NOMEM;
		goto out1;
	}
	OSAL_MEM_ZERO(p_virt, p_blk->real_size_in_page);

	p_hwfn->p_cxt_mngr->ilt_shadow[shadow_line].virt_addr = p_virt;
	p_hwfn->p_cxt_mngr->ilt_shadow[shadow_line].phys_addr = p_phys;
	p_hwfn->p_cxt_mngr->ilt_shadow[shadow_line].size =
		p_blk->real_size_in_page;

	/* compute absolute offset */
	reg_offset = PSWRQ2_REG_ILT_MEMORY +
		     (line * ILT_REG_SIZE_IN_BYTES * ILT_ENTRY_IN_REGS);

	ilt_hw_entry = 0;
	SET_FIELD(ilt_hw_entry, ILT_ENTRY_VALID, 1ULL);
	SET_FIELD(ilt_hw_entry,
		  ILT_ENTRY_PHY_ADDR,
		 (p_hwfn->p_cxt_mngr->ilt_shadow[shadow_line].phys_addr >> 12));

/* Write via DMAE since the PSWRQ2_REG_ILT_MEMORY line is a wide-bus */

	ecore_dmae_host2grc(p_hwfn, p_ptt, (u64)(osal_uintptr_t)&ilt_hw_entry,
			    reg_offset, sizeof(ilt_hw_entry) / sizeof(u32),
			    OSAL_NULL /* default parameters */);

out1:
	ecore_ptt_release(p_hwfn, p_ptt);
out0:
	OSAL_MUTEX_RELEASE(&p_hwfn->p_cxt_mngr->mutex);

	return rc;
}

/* This function is very RoCE oriented, if another protocol in the future
 * will want this feature we'll need to modify the function to be more generic
 */
static enum _ecore_status_t
ecore_cxt_free_ilt_range(struct ecore_hwfn *p_hwfn,
			 enum ecore_cxt_elem_type elem_type,
			 u32 start_iid, u32 count)
{
	u32 start_line, end_line, shadow_start_line, shadow_end_line;
	u32 reg_offset, elem_size, hw_p_size, elems_per_p;
	struct ecore_ilt_client_cfg *p_cli;
	struct ecore_ilt_cli_blk *p_blk;
	u32 end_iid = start_iid + count;
	struct ecore_ptt *p_ptt;
	u64 ilt_hw_entry = 0;
	u32 i;

	switch (elem_type) {
	case ECORE_ELEM_CXT:
		p_cli = &p_hwfn->p_cxt_mngr->clients[ILT_CLI_CDUC];
		elem_size = CONN_CXT_SIZE(p_hwfn);
		p_blk = &p_cli->pf_blks[CDUC_BLK];
		break;
	case ECORE_ELEM_SRQ:
		p_cli = &p_hwfn->p_cxt_mngr->clients[ILT_CLI_TSDM];
		elem_size = SRQ_CXT_SIZE;
		p_blk = &p_cli->pf_blks[SRQ_BLK];
		break;
	case ECORE_ELEM_TASK:
		p_cli = &p_hwfn->p_cxt_mngr->clients[ILT_CLI_CDUT];
		elem_size = TYPE1_TASK_CXT_SIZE(p_hwfn);
		p_blk = &p_cli->pf_blks[CDUT_SEG_BLK(ECORE_CXT_ROCE_TID_SEG)];
		break;
	default:
		DP_NOTICE(p_hwfn, false,
			  "ECORE_ILWALID elem type = %d", elem_type);
		return ECORE_ILWAL;
	}

	/* Callwlate line in ilt */
	hw_p_size = p_cli->p_size.val;
	elems_per_p = ILT_PAGE_IN_BYTES(hw_p_size) / elem_size;
	start_line = p_blk->start_line + (start_iid / elems_per_p);
	end_line = p_blk->start_line + (end_iid / elems_per_p);
	if (((end_iid + 1) / elems_per_p) != (end_iid / elems_per_p))
		end_line--;

	shadow_start_line = start_line - p_hwfn->p_cxt_mngr->pf_start_line;
	shadow_end_line = end_line - p_hwfn->p_cxt_mngr->pf_start_line;

	p_ptt = ecore_ptt_acquire(p_hwfn);
	if (!p_ptt) {
		DP_NOTICE(p_hwfn, false,
			  "ECORE_TIME_OUT on ptt acquire - dynamic allocation");
		return ECORE_TIMEOUT;
	}

	for (i = shadow_start_line; i < shadow_end_line; i++) {
		if (!p_hwfn->p_cxt_mngr->ilt_shadow[i].virt_addr)
			continue;

		OSAL_DMA_FREE_COHERENT(p_hwfn->p_dev,
				    p_hwfn->p_cxt_mngr->ilt_shadow[i].virt_addr,
				    p_hwfn->p_cxt_mngr->ilt_shadow[i].phys_addr,
				    p_hwfn->p_cxt_mngr->ilt_shadow[i].size);

		p_hwfn->p_cxt_mngr->ilt_shadow[i].virt_addr = OSAL_NULL;
		p_hwfn->p_cxt_mngr->ilt_shadow[i].phys_addr = 0;
		p_hwfn->p_cxt_mngr->ilt_shadow[i].size = 0;

		/* compute absolute offset */
		reg_offset = PSWRQ2_REG_ILT_MEMORY +
		    ((start_line++) * ILT_REG_SIZE_IN_BYTES *
		     ILT_ENTRY_IN_REGS);

		/* Write via DMAE since the PSWRQ2_REG_ILT_MEMORY line is a
		 * wide-bus.
		 */
		ecore_dmae_host2grc(p_hwfn, p_ptt,
				    (u64)(osal_uintptr_t)&ilt_hw_entry,
				    reg_offset,
				    sizeof(ilt_hw_entry) / sizeof(u32),
				    OSAL_NULL /* default parameters */);
	}

	ecore_ptt_release(p_hwfn, p_ptt);

	return ECORE_SUCCESS;
}

static u16 ecore_blk_callwlate_pages(struct ecore_ilt_cli_blk *p_blk)
{
	if (p_blk->real_size_in_page == 0)
		return 0;

	return DIV_ROUND_UP(p_blk->total_size, p_blk->real_size_in_page);
}

u16 ecore_get_cdut_num_pf_init_pages(struct ecore_hwfn *p_hwfn)
{
	struct ecore_ilt_client_cfg *p_cli;
	struct ecore_ilt_cli_blk *p_blk;
	u16 i, pages = 0;

	p_cli = &p_hwfn->p_cxt_mngr->clients[ILT_CLI_CDUT];
	for (i = 0; i < NUM_TASK_PF_SEGMENTS; i++) {
		p_blk = &p_cli->pf_blks[CDUT_FL_SEG_BLK(i, PF)];
		pages += ecore_blk_callwlate_pages(p_blk);
	}

	return pages;
}

u16 ecore_get_cdut_num_vf_init_pages(struct ecore_hwfn *p_hwfn)
{
	struct ecore_ilt_client_cfg *p_cli;
	struct ecore_ilt_cli_blk *p_blk;
	u16 i, pages = 0;

	p_cli = &p_hwfn->p_cxt_mngr->clients[ILT_CLI_CDUT];
	for (i = 0; i < NUM_TASK_VF_SEGMENTS; i++) {
		p_blk = &p_cli->vf_blks[CDUT_FL_SEG_BLK(i, VF)];
		pages += ecore_blk_callwlate_pages(p_blk);
	}

	return pages;
}

u16 ecore_get_cdut_num_pf_work_pages(struct ecore_hwfn *p_hwfn)
{
	struct ecore_ilt_client_cfg *p_cli;
	struct ecore_ilt_cli_blk *p_blk;
	u16 i, pages = 0;

	p_cli = &p_hwfn->p_cxt_mngr->clients[ILT_CLI_CDUT];
	for (i = 0; i < NUM_TASK_PF_SEGMENTS; i++) {
		p_blk = &p_cli->pf_blks[CDUT_SEG_BLK(i)];
		pages += ecore_blk_callwlate_pages(p_blk);
	}

	return pages;
}

u16 ecore_get_cdut_num_vf_work_pages(struct ecore_hwfn *p_hwfn)
{
	struct ecore_ilt_client_cfg *p_cli;
	struct ecore_ilt_cli_blk *p_blk;
	u16 pages = 0, i;

	p_cli = &p_hwfn->p_cxt_mngr->clients[ILT_CLI_CDUT];
	for (i = 0; i < NUM_TASK_VF_SEGMENTS; i++) {
		p_blk = &p_cli->vf_blks[CDUT_SEG_BLK(i)];
		pages += ecore_blk_callwlate_pages(p_blk);
	}

	return pages;
}
