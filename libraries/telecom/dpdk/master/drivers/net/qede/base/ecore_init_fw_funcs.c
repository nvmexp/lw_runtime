/* SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 2016 - 2018 Cavium Inc.
 * All rights reserved.
 * www.cavium.com
 */

#include "bcm_osal.h"
#include "ecore_hw.h"
#include "ecore_init_ops.h"
#include "reg_addr.h"
#include "ecore_rt_defs.h"
#include "ecore_hsi_init_func.h"
#include "ecore_hsi_init_tool.h"
#include "ecore_iro.h"
#include "ecore_init_fw_funcs.h"
static u16 con_region_offsets[3][NUM_OF_CONNECTION_TYPES] = {
	{ 400,  336,  352,  368,  304,  384,  416,  352}, /* region 3 offsets */
	{ 528,  496,  416,  512,  448,  512,  544,  480}, /* region 4 offsets */
	{ 608,  544,  496,  576,  576,  592,  624,  560}  /* region 5 offsets */
};
static u16 task_region_offsets[1][NUM_OF_CONNECTION_TYPES] = {
	{ 240,  240,  112,    0,    0,    0,    0,   96}  /* region 1 offsets */
};

/* General constants */
#define QM_PQ_MEM_4KB(pq_size) \
	(pq_size ? DIV_ROUND_UP((pq_size + 1) * QM_PQ_ELEMENT_SIZE, 0x1000) : 0)
#define QM_PQ_SIZE_256B(pq_size) \
	(pq_size ? DIV_ROUND_UP(pq_size, 0x100) - 1 : 0)
#define QM_ILWALID_PQ_ID		0xffff

/* Max link speed (in Mbps) */
#define QM_MAX_LINK_SPEED		100000

/* Feature enable */
#define QM_BYPASS_EN			1
#define QM_BYTE_CRD_EN			1

/* Other PQ constants */
#define QM_OTHER_PQS_PER_PF		4

/* VOQ constants */
#define MAX_NUM_VOQS			(MAX_NUM_PORTS_K2 * NUM_TCS_4PORT_K2)
#define VOQS_BIT_MASK			((1 << MAX_NUM_VOQS) - 1)

/* WFQ constants: */

/* Upper bound in MB, 10 * burst size of 1ms in 50Gbps */
#define QM_WFQ_UPPER_BOUND		62500000

/* Bit  of VOQ in WFQ VP PQ map */
#define QM_WFQ_VP_PQ_VOQ_SHIFT		0

/* Bit  of PF in WFQ VP PQ map */
#define QM_WFQ_VP_PQ_PF_SHIFT		5

/* 0x9000 = 4*9*1024 */
#define QM_WFQ_INC_VAL(weight)		((weight) * 0x9000)

/* Max WFQ increment value is 0.7 * upper bound */
#define QM_WFQ_MAX_INC_VAL		((QM_WFQ_UPPER_BOUND * 7) / 10)

/* RL constants: */

/* Period in us */
#define QM_RL_PERIOD			5

/* Period in 25MHz cycles */
#define QM_RL_PERIOD_CLK_25M		(25 * QM_RL_PERIOD)

/* RL increment value - rate is specified in mbps. the factor of 1.01 was
 * added after seeing only 99% factor reached in a 25Gbps port with DPDK RFC
 * 2544 test. In this scenario the PF RL was reducing the line rate to 99%
 * although the credit increment value was the correct one and FW callwlated
 * correct packet sizes. The reason for the inaclwracy of the RL is unknown at
 * this point.
 */
#define QM_RL_INC_VAL(rate) \
	OSAL_MAX_T(u32, (u32)(((rate ? rate : 100000) * QM_RL_PERIOD * 101) / \
	(8 * 100)), 1)

/* PF RL Upper bound is set to 10 * burst size of 1ms in 50Gbps */
#define QM_PF_RL_UPPER_BOUND		62500000

/* Max PF RL increment value is 0.7 * upper bound */
#define QM_PF_RL_MAX_INC_VAL		((QM_PF_RL_UPPER_BOUND * 7) / 10)

/* Vport RL Upper bound, link speed is in Mpbs */
#define QM_VP_RL_UPPER_BOUND(speed) \
	((u32)OSAL_MAX_T(u32, QM_RL_INC_VAL(speed), 9700 + 1000))

/* Max Vport RL increment value is the Vport RL upper bound */
#define QM_VP_RL_MAX_INC_VAL(speed)	QM_VP_RL_UPPER_BOUND(speed)

/* Vport RL credit threshold in case of QM bypass */
#define QM_VP_RL_BYPASS_THRESH_SPEED	(QM_VP_RL_UPPER_BOUND(10000) - 1)

/* AFullOprtnstcCrdMask constants */
#define QM_OPPOR_LINE_VOQ_DEF		1
#define QM_OPPOR_FW_STOP_DEF		0
#define QM_OPPOR_PQ_EMPTY_DEF		1

/* Command Queue constants: */

/* Pure LB CmdQ lines (+spare) */
#define PBF_CMDQ_PURE_LB_LINES		150

#define PBF_CMDQ_LINES_RT_OFFSET(ext_voq) \
	(PBF_REG_YCMD_QS_NUM_LINES_VOQ0_RT_OFFSET + \
	 ext_voq * \
	 (PBF_REG_YCMD_QS_NUM_LINES_VOQ1_RT_OFFSET - \
	  PBF_REG_YCMD_QS_NUM_LINES_VOQ0_RT_OFFSET))

#define PBF_BTB_GUARANTEED_RT_OFFSET(ext_voq) \
	(PBF_REG_BTB_GUARANTEED_VOQ0_RT_OFFSET + \
	 ext_voq * \
	 (PBF_REG_BTB_GUARANTEED_VOQ1_RT_OFFSET - \
	  PBF_REG_BTB_GUARANTEED_VOQ0_RT_OFFSET))

#define QM_VOQ_LINE_CRD(pbf_cmd_lines) \
((((pbf_cmd_lines) - 4) * 2) | QM_LINE_CRD_REG_SIGN_BIT)

/* BTB: blocks constants (block size = 256B) */

/* 256B blocks in 9700B packet */
#define BTB_JUMBO_PKT_BLOCKS		38

/* Headroom per-port */
#define BTB_HEADROOM_BLOCKS		BTB_JUMBO_PKT_BLOCKS
#define BTB_PURE_LB_FACTOR		10

/* Factored (hence really 0.7) */
#define BTB_PURE_LB_RATIO		7

/* QM stop command constants */
#define QM_STOP_PQ_MASK_WIDTH		32
#define QM_STOP_CMD_ADDR		2
#define QM_STOP_CMD_STRUCT_SIZE		2
#define QM_STOP_CMD_PAUSE_MASK_OFFSET	0
#define QM_STOP_CMD_PAUSE_MASK_SHIFT	0
#define QM_STOP_CMD_PAUSE_MASK_MASK	0xffffffff /* @DPDK */
#define QM_STOP_CMD_GROUP_ID_OFFSET	1
#define QM_STOP_CMD_GROUP_ID_SHIFT	16
#define QM_STOP_CMD_GROUP_ID_MASK	15
#define QM_STOP_CMD_PQ_TYPE_OFFSET	1
#define QM_STOP_CMD_PQ_TYPE_SHIFT	24
#define QM_STOP_CMD_PQ_TYPE_MASK	1
#define QM_STOP_CMD_MAX_POLL_COUNT	100
#define QM_STOP_CMD_POLL_PERIOD_US	500

/* QM command macros */
#define QM_CMD_STRUCT_SIZE(cmd) cmd##_STRUCT_SIZE
#define QM_CMD_SET_FIELD(var, cmd, field, value) \
	SET_FIELD(var[cmd##_##field##_OFFSET], cmd##_##field, value)

#define QM_INIT_TX_PQ_MAP(p_hwfn, map, pq_id, vp_pq_id, \
			   rl_valid, rl_id, voq, wrr) \
	do { \
		OSAL_MEMSET(&(map), 0, sizeof(map)); \
		SET_FIELD(map.reg, QM_RF_PQ_MAP_PQ_VALID, 1); \
		SET_FIELD(map.reg, QM_RF_PQ_MAP_RL_VALID, rl_valid ? 1 : 0); \
		SET_FIELD(map.reg, QM_RF_PQ_MAP_RL_ID, rl_id); \
		SET_FIELD(map.reg, QM_RF_PQ_MAP_VP_PQ_ID, vp_pq_id); \
		SET_FIELD(map.reg, QM_RF_PQ_MAP_VOQ, voq); \
		SET_FIELD(map.reg, QM_RF_PQ_MAP_WRR_WEIGHT_GROUP, wrr); \
		STORE_RT_REG(p_hwfn, QM_REG_TXPQMAP_RT_OFFSET + (pq_id), \
			     *((u32 *)&(map))); \
	} while (0)

#define WRITE_PQ_INFO_TO_RAM		1

#define PQ_INFO_ELEMENT(vp_pq_id, pf, tc, port, rl_valid, rl_id) \
	(((vp_pq_id) << 0) | ((pf) << 12) | ((tc) << 16) | ((port) << 20) | \
	 ((rl_valid ? 1 : 0) << 22) | (((rl_id) & 255) << 24) | \
	 (((rl_id) >> 8) << 9))

#define PQ_INFO_RAM_GRC_ADDRESS(pq_id) (XSEM_REG_FAST_MEMORY + \
	SEM_FAST_REG_INT_RAM + XSTORM_PQ_INFO_OFFSET(pq_id))

/******************** INTERNAL IMPLEMENTATION *********************/

/* Prepare PF RL enable/disable runtime init values */
static void ecore_enable_pf_rl(struct ecore_hwfn *p_hwfn, bool pf_rl_en)
{
	STORE_RT_REG(p_hwfn, QM_REG_RLPFENABLE_RT_OFFSET, pf_rl_en ? 1 : 0);
	if (pf_rl_en) {
		/* Enable RLs for all VOQs */
		STORE_RT_REG(p_hwfn, QM_REG_RLPFVOQENABLE_RT_OFFSET,
			     VOQS_BIT_MASK);

		/* Write RL period */
		STORE_RT_REG(p_hwfn, QM_REG_RLPFPERIOD_RT_OFFSET,
			     QM_RL_PERIOD_CLK_25M);
		STORE_RT_REG(p_hwfn, QM_REG_RLPFPERIODTIMER_RT_OFFSET,
			     QM_RL_PERIOD_CLK_25M);

		/* Set credit threshold for QM bypass flow */
		if (QM_BYPASS_EN)
			STORE_RT_REG(p_hwfn, QM_REG_AFULLQMBYPTHRPFRL_RT_OFFSET,
				     QM_PF_RL_UPPER_BOUND);
	}
}

/* Prepare PF WFQ enable/disable runtime init values */
static void ecore_enable_pf_wfq(struct ecore_hwfn *p_hwfn, bool pf_wfq_en)
{
	STORE_RT_REG(p_hwfn, QM_REG_WFQPFENABLE_RT_OFFSET, pf_wfq_en ? 1 : 0);

	/* Set credit threshold for QM bypass flow */
	if (pf_wfq_en && QM_BYPASS_EN)
		STORE_RT_REG(p_hwfn, QM_REG_AFULLQMBYPTHRPFWFQ_RT_OFFSET,
			     QM_WFQ_UPPER_BOUND);
}

/* Prepare global RL enable/disable runtime init values */
static void ecore_enable_global_rl(struct ecore_hwfn *p_hwfn,
				   bool global_rl_en)
{
	STORE_RT_REG(p_hwfn, QM_REG_RLGLBLENABLE_RT_OFFSET,
		     global_rl_en ? 1 : 0);
	if (global_rl_en) {
		/* Write RL period (use timer 0 only) */
		STORE_RT_REG(p_hwfn, QM_REG_RLGLBLPERIOD_0_RT_OFFSET,
			     QM_RL_PERIOD_CLK_25M);
		STORE_RT_REG(p_hwfn, QM_REG_RLGLBLPERIODTIMER_0_RT_OFFSET,
			     QM_RL_PERIOD_CLK_25M);

		/* Set credit threshold for QM bypass flow */
		if (QM_BYPASS_EN)
			STORE_RT_REG(p_hwfn,
				     QM_REG_AFULLQMBYPTHRGLBLRL_RT_OFFSET,
				     QM_VP_RL_BYPASS_THRESH_SPEED);
	}
}

/* Prepare VPORT WFQ enable/disable runtime init values */
static void ecore_enable_vport_wfq(struct ecore_hwfn *p_hwfn, bool vport_wfq_en)
{
	STORE_RT_REG(p_hwfn, QM_REG_WFQVPENABLE_RT_OFFSET,
		     vport_wfq_en ? 1 : 0);

	/* Set credit threshold for QM bypass flow */
	if (vport_wfq_en && QM_BYPASS_EN)
		STORE_RT_REG(p_hwfn, QM_REG_AFULLQMBYPTHRVPWFQ_RT_OFFSET,
			     QM_WFQ_UPPER_BOUND);
}

/* Prepare runtime init values to allocate PBF command queue lines for
 * the specified VOQ
 */
static void ecore_cmdq_lines_voq_rt_init(struct ecore_hwfn *p_hwfn,
					 u8 voq,
					 u16 cmdq_lines)
{
	u32 qm_line_crd = QM_VOQ_LINE_CRD(cmdq_lines);

	OVERWRITE_RT_REG(p_hwfn, PBF_CMDQ_LINES_RT_OFFSET(voq),
			 (u32)cmdq_lines);
	STORE_RT_REG(p_hwfn, QM_REG_VOQCRDLINE_RT_OFFSET + voq, qm_line_crd);
	STORE_RT_REG(p_hwfn, QM_REG_VOQINITCRDLINE_RT_OFFSET + voq,
		     qm_line_crd);
}

/* Prepare runtime init values to allocate PBF command queue lines. */
static void ecore_cmdq_lines_rt_init(struct ecore_hwfn *p_hwfn,
				     u8 max_ports_per_engine,
				     u8 max_phys_tcs_per_port,
				     struct init_qm_port_params
				     port_params[MAX_NUM_PORTS])
{
	u8 tc, voq, port_id, num_tcs_in_port;

	/* Clear PBF lines of all VOQs */
	for (voq = 0; voq < MAX_NUM_VOQS; voq++)
		STORE_RT_REG(p_hwfn, PBF_CMDQ_LINES_RT_OFFSET(voq), 0);

	for (port_id = 0; port_id < max_ports_per_engine; port_id++) {
		u16 phys_lines, phys_lines_per_tc;

		if (!port_params[port_id].active)
			continue;

		/* Find number of command queue lines to divide between the
		 * active physical TCs.
		 */
		phys_lines = port_params[port_id].num_pbf_cmd_lines;
		phys_lines -= PBF_CMDQ_PURE_LB_LINES;

		/* Find #lines per active physical TC */
		num_tcs_in_port = 0;
		for (tc = 0; tc < max_phys_tcs_per_port; tc++)
			if (((port_params[port_id].active_phys_tcs >> tc) &
			      0x1) == 1)
				num_tcs_in_port++;
		phys_lines_per_tc = phys_lines / num_tcs_in_port;

		/* Init registers per active TC */
		for (tc = 0; tc < max_phys_tcs_per_port; tc++) {
			voq = VOQ(port_id, tc, max_phys_tcs_per_port);
			if (((port_params[port_id].active_phys_tcs >>
			      tc) & 0x1) == 1)
				ecore_cmdq_lines_voq_rt_init(p_hwfn, voq,
							     phys_lines_per_tc);
		}

		/* Init registers for pure LB TC */
		voq = VOQ(port_id, PURE_LB_TC, max_phys_tcs_per_port);
		ecore_cmdq_lines_voq_rt_init(p_hwfn, voq,
					     PBF_CMDQ_PURE_LB_LINES);
	}
}

/*
 * Prepare runtime init values to allocate guaranteed BTB blocks for the
 * specified port. The guaranteed BTB space is divided between the TCs as
 * follows (shared space Is lwrrently not used):
 * 1. Parameters:
 *     B BTB blocks for this port
 *     C Number of physical TCs for this port
 * 2. Callwlation:
 *     a. 38 blocks (9700B jumbo frame) are allocated for global per port
 *        headroom
 *     b. B = B 38 (remainder after global headroom allocation)
 *     c. MAX(38,B/(C+0.7)) blocks are allocated for the pure LB VOQ.
 *     d. B = B MAX(38, B/(C+0.7)) (remainder after pure LB allocation).
 *     e. B/C blocks are allocated for each physical TC.
 * Assumptions:
 * - MTU is up to 9700 bytes (38 blocks)
 * - All TCs are considered symmetrical (same rate and packet size)
 * - No optimization for lossy TC (all are considered lossless). Shared space is
 *   not enabled and allocated for each TC.
 */
static void ecore_btb_blocks_rt_init(struct ecore_hwfn *p_hwfn,
				     u8 max_ports_per_engine,
				     u8 max_phys_tcs_per_port,
				     struct init_qm_port_params
				     port_params[MAX_NUM_PORTS])
{
	u32 usable_blocks, pure_lb_blocks, phys_blocks;
	u8 tc, voq, port_id, num_tcs_in_port;

	for (port_id = 0; port_id < max_ports_per_engine; port_id++) {
		if (!port_params[port_id].active)
			continue;

		/* Subtract headroom blocks */
		usable_blocks = port_params[port_id].num_btb_blocks -
				BTB_HEADROOM_BLOCKS;

		/* Find blocks per physical TC. use factor to avoid floating
		 * arithmethic.
		 */
		num_tcs_in_port = 0;
		for (tc = 0; tc < NUM_OF_PHYS_TCS; tc++)
			if (((port_params[port_id].active_phys_tcs >> tc) &
			      0x1) == 1)
				num_tcs_in_port++;

		pure_lb_blocks = (usable_blocks * BTB_PURE_LB_FACTOR) /
				  (num_tcs_in_port * BTB_PURE_LB_FACTOR +
				   BTB_PURE_LB_RATIO);
		pure_lb_blocks = OSAL_MAX_T(u32, BTB_JUMBO_PKT_BLOCKS,
					    pure_lb_blocks /
					    BTB_PURE_LB_FACTOR);
		phys_blocks = (usable_blocks - pure_lb_blocks) /
			      num_tcs_in_port;

		/* Init physical TCs */
		for (tc = 0; tc < NUM_OF_PHYS_TCS; tc++) {
			if (((port_params[port_id].active_phys_tcs >> tc) &
			     0x1) == 1) {
				voq = VOQ(port_id, tc, max_phys_tcs_per_port);
				STORE_RT_REG(p_hwfn,
					PBF_BTB_GUARANTEED_RT_OFFSET(voq),
					phys_blocks);
			}
		}

		/* Init pure LB TC */
		voq = VOQ(port_id, PURE_LB_TC, max_phys_tcs_per_port);
		STORE_RT_REG(p_hwfn, PBF_BTB_GUARANTEED_RT_OFFSET(voq),
			     pure_lb_blocks);
	}
}

/* Prepare runtime init values for the specified RL.
 * If global_rl_params is OSAL_NULL, max link speed (100Gbps) is used instead.
 * Return -1 on error.
 */
static int ecore_global_rl_rt_init(struct ecore_hwfn *p_hwfn,
				   struct init_qm_global_rl_params
				     global_rl_params[COMMON_MAX_QM_GLOBAL_RLS])
{
	u32 upper_bound = QM_VP_RL_UPPER_BOUND(QM_MAX_LINK_SPEED) |
			  (u32)QM_RL_CRD_REG_SIGN_BIT;
	u32 inc_val;
	u16 rl_id;

	/* Go over all global RLs */
	for (rl_id = 0; rl_id < MAX_QM_GLOBAL_RLS; rl_id++) {
		u32 rate_limit = global_rl_params ?
				 global_rl_params[rl_id].rate_limit : 0;

		inc_val = QM_RL_INC_VAL(rate_limit ?
					rate_limit : QM_MAX_LINK_SPEED);
		if (inc_val > QM_VP_RL_MAX_INC_VAL(QM_MAX_LINK_SPEED)) {
			DP_NOTICE(p_hwfn, true, "Invalid rate limit configuration.\n");
			return -1;
		}

		STORE_RT_REG(p_hwfn, QM_REG_RLGLBLCRD_RT_OFFSET + rl_id,
			     (u32)QM_RL_CRD_REG_SIGN_BIT);
		STORE_RT_REG(p_hwfn, QM_REG_RLGLBLUPPERBOUND_RT_OFFSET + rl_id,
			     upper_bound);
		STORE_RT_REG(p_hwfn, QM_REG_RLGLBLINCVAL_RT_OFFSET + rl_id,
			     inc_val);
	}

	return 0;
}

/* Prepare Tx PQ mapping runtime init values for the specified PF */
static int ecore_tx_pq_map_rt_init(struct ecore_hwfn *p_hwfn,
				    struct ecore_ptt *p_ptt,
				    u8 pf_id,
				    u8 max_phys_tcs_per_port,
						bool is_pf_loading,
				    u32 num_pf_cids,
				    u32 num_vf_cids,
				    u16 start_pq,
				    u16 num_pf_pqs,
				    u16 num_vf_pqs,
				   u16 start_vport,
				    u32 base_mem_addr_4kb,
				    struct init_qm_pq_params *pq_params,
				    struct init_qm_vport_params *vport_params)
{
	/* A bit per Tx PQ indicating if the PQ is associated with a VF */
	u32 tx_pq_vf_mask[MAX_QM_TX_QUEUES / QM_PF_QUEUE_GROUP_SIZE] = { 0 };
	u32 num_tx_pq_vf_masks = MAX_QM_TX_QUEUES / QM_PF_QUEUE_GROUP_SIZE;
	u16 num_pqs, first_pq_group, last_pq_group, i, j, pq_id, pq_group;
	u32 pq_mem_4kb, vport_pq_mem_4kb, mem_addr_4kb;
	#if (WRITE_PQ_INFO_TO_RAM != 0)
		u32 pq_info = 0;
	#endif

	num_pqs = num_pf_pqs + num_vf_pqs;

	first_pq_group = start_pq / QM_PF_QUEUE_GROUP_SIZE;
	last_pq_group = (start_pq + num_pqs - 1) / QM_PF_QUEUE_GROUP_SIZE;

	pq_mem_4kb = QM_PQ_MEM_4KB(num_pf_cids);
	vport_pq_mem_4kb = QM_PQ_MEM_4KB(num_vf_cids);
	mem_addr_4kb = base_mem_addr_4kb;

	/* Set mapping from PQ group to PF */
	for (pq_group = first_pq_group; pq_group <= last_pq_group; pq_group++)
		STORE_RT_REG(p_hwfn, QM_REG_PQTX2PF_0_RT_OFFSET + pq_group,
			     (u32)(pf_id));

	/* Set PQ sizes */
	STORE_RT_REG(p_hwfn, QM_REG_MAXPQSIZE_0_RT_OFFSET,
		     QM_PQ_SIZE_256B(num_pf_cids));
	STORE_RT_REG(p_hwfn, QM_REG_MAXPQSIZE_1_RT_OFFSET,
		     QM_PQ_SIZE_256B(num_vf_cids));

	/* Go over all Tx PQs */
	for (i = 0, pq_id = start_pq; i < num_pqs; i++, pq_id++) {
		u16 first_tx_pq_id, vport_id_in_pf;
		struct qm_rf_pq_map tx_pq_map;
		bool is_vf_pq;
		u8 voq;

		voq = VOQ(pq_params[i].port_id, pq_params[i].tc_id,
			  max_phys_tcs_per_port);
		is_vf_pq = (i >= num_pf_pqs);

		/* Update first Tx PQ of VPORT/TC */
		vport_id_in_pf = pq_params[i].vport_id - start_vport;
		first_tx_pq_id =
		vport_params[vport_id_in_pf].first_tx_pq_id[pq_params[i].tc_id];
		if (first_tx_pq_id == QM_ILWALID_PQ_ID) {
			u32 map_val = (voq << QM_WFQ_VP_PQ_VOQ_SHIFT) |
				      (pf_id << QM_WFQ_VP_PQ_PF_SHIFT);

			/* Create new VP PQ */
			vport_params[vport_id_in_pf].
			    first_tx_pq_id[pq_params[i].tc_id] = pq_id;
			first_tx_pq_id = pq_id;

			/* Map VP PQ to VOQ and PF */
			STORE_RT_REG(p_hwfn, QM_REG_WFQVPMAP_RT_OFFSET +
				     first_tx_pq_id, map_val);
		}

		/* Prepare PQ map entry */
		QM_INIT_TX_PQ_MAP(p_hwfn, tx_pq_map, pq_id, first_tx_pq_id,
				  pq_params[i].rl_valid, pq_params[i].rl_id,
				  voq, pq_params[i].wrr_group);

		/* Set PQ base address */
		STORE_RT_REG(p_hwfn, QM_REG_BASEADDRTXPQ_RT_OFFSET + pq_id,
			     mem_addr_4kb);

		/* Clear PQ pointer table entry (64 bit) */
		if (is_pf_loading)
			for (j = 0; j < 2; j++)
				STORE_RT_REG(p_hwfn, QM_REG_PTRTBLTX_RT_OFFSET +
					     (pq_id * 2) + j, 0);

		/* Write PQ info to RAM */
#if (WRITE_PQ_INFO_TO_RAM != 0)
		pq_info = PQ_INFO_ELEMENT(first_tx_pq_id, pf_id,
					  pq_params[i].tc_id,
					  pq_params[i].port_id,
					  pq_params[i].rl_valid,
					  pq_params[i].rl_id);
		ecore_wr(p_hwfn, p_ptt, PQ_INFO_RAM_GRC_ADDRESS(pq_id),
			 pq_info);
#endif

		/* If VF PQ, add indication to PQ VF mask */
		if (is_vf_pq) {
			tx_pq_vf_mask[pq_id / QM_PF_QUEUE_GROUP_SIZE] |=
				(1 << (pq_id % QM_PF_QUEUE_GROUP_SIZE));
			mem_addr_4kb += vport_pq_mem_4kb;
		} else {
			mem_addr_4kb += pq_mem_4kb;
		}
	}

	/* Store Tx PQ VF mask to size select register */
	for (i = 0; i < num_tx_pq_vf_masks; i++)
		if (tx_pq_vf_mask[i])
			STORE_RT_REG(p_hwfn, QM_REG_MAXPQSIZETXSEL_0_RT_OFFSET +
				     i, tx_pq_vf_mask[i]);

	return 0;
}

/* Prepare Other PQ mapping runtime init values for the specified PF */
static void ecore_other_pq_map_rt_init(struct ecore_hwfn *p_hwfn,
				       u8 pf_id,
				       bool is_pf_loading,
				       u32 num_pf_cids,
				       u32 num_tids,
				       u32 base_mem_addr_4kb)
{
	u32 pq_size, pq_mem_4kb, mem_addr_4kb;
	u16 i, j, pq_id, pq_group;

	/* A single other PQ group is used in each PF, where PQ group i is used
	 * in PF i.
	 */
	pq_group = pf_id;
	pq_size = num_pf_cids + num_tids;
	pq_mem_4kb = QM_PQ_MEM_4KB(pq_size);
	mem_addr_4kb = base_mem_addr_4kb;

	/* Map PQ group to PF */
	STORE_RT_REG(p_hwfn, QM_REG_PQOTHER2PF_0_RT_OFFSET + pq_group,
		     (u32)(pf_id));

	/* Set PQ sizes */
	STORE_RT_REG(p_hwfn, QM_REG_MAXPQSIZE_2_RT_OFFSET,
		     QM_PQ_SIZE_256B(pq_size));

	for (i = 0, pq_id = pf_id * QM_PF_QUEUE_GROUP_SIZE;
	     i < QM_OTHER_PQS_PER_PF; i++, pq_id++) {
		/* Set PQ base address */
		STORE_RT_REG(p_hwfn, QM_REG_BASEADDROTHERPQ_RT_OFFSET + pq_id,
			     mem_addr_4kb);

		/* Clear PQ pointer table entry */
		if (is_pf_loading)
			for (j = 0; j < 2; j++)
				STORE_RT_REG(p_hwfn,
					     QM_REG_PTRTBLOTHER_RT_OFFSET +
					     (pq_id * 2) + j, 0);

		mem_addr_4kb += pq_mem_4kb;
	}
}

/* Prepare PF WFQ runtime init values for the specified PF.
 * Return -1 on error.
 */
static int ecore_pf_wfq_rt_init(struct ecore_hwfn *p_hwfn,
				u8 pf_id,
				u16 pf_wfq,
				u8 max_phys_tcs_per_port,
				u16 num_tx_pqs,
				struct init_qm_pq_params *pq_params)
{
	u32 inc_val, crd_reg_offset;
	u8 voq;
	u16 i;

	inc_val = QM_WFQ_INC_VAL(pf_wfq);
	if (!inc_val || inc_val > QM_WFQ_MAX_INC_VAL) {
		DP_NOTICE(p_hwfn, true,
			  "Invalid PF WFQ weight configuration\n");
		return -1;
	}

	for (i = 0; i < num_tx_pqs; i++) {
		voq = VOQ(pq_params[i].port_id, pq_params[i].tc_id,
			  max_phys_tcs_per_port);
		crd_reg_offset = (pf_id < MAX_NUM_PFS_BB ?
				  QM_REG_WFQPFCRD_RT_OFFSET :
				  QM_REG_WFQPFCRD_MSB_RT_OFFSET) +
				 voq * MAX_NUM_PFS_BB +
				 (pf_id % MAX_NUM_PFS_BB);
		OVERWRITE_RT_REG(p_hwfn, crd_reg_offset,
				 (u32)QM_WFQ_CRD_REG_SIGN_BIT);
	}

	STORE_RT_REG(p_hwfn, QM_REG_WFQPFUPPERBOUND_RT_OFFSET +
		     pf_id, QM_WFQ_UPPER_BOUND | (u32)QM_WFQ_CRD_REG_SIGN_BIT);
	STORE_RT_REG(p_hwfn, QM_REG_WFQPFWEIGHT_RT_OFFSET + pf_id, inc_val);

	return 0;
}

/* Prepare PF RL runtime init values for the specified PF.
 * Return -1 on error.
 */
static int ecore_pf_rl_rt_init(struct ecore_hwfn *p_hwfn, u8 pf_id, u32 pf_rl)
{
	u32 inc_val;

	inc_val = QM_RL_INC_VAL(pf_rl);
	if (inc_val > QM_PF_RL_MAX_INC_VAL) {
		DP_NOTICE(p_hwfn, true,
			  "Invalid PF rate limit configuration\n");
		return -1;
	}

	STORE_RT_REG(p_hwfn, QM_REG_RLPFCRD_RT_OFFSET + pf_id,
		     (u32)QM_RL_CRD_REG_SIGN_BIT);
	STORE_RT_REG(p_hwfn, QM_REG_RLPFUPPERBOUND_RT_OFFSET + pf_id,
		     QM_PF_RL_UPPER_BOUND | (u32)QM_RL_CRD_REG_SIGN_BIT);
	STORE_RT_REG(p_hwfn, QM_REG_RLPFINCVAL_RT_OFFSET + pf_id, inc_val);

	return 0;
}

/* Prepare VPORT WFQ runtime init values for the specified VPORTs.
 * Return -1 on error.
 */
static int ecore_vp_wfq_rt_init(struct ecore_hwfn *p_hwfn,
				u16 num_vports,
				struct init_qm_vport_params *vport_params)
{
	u16 vp_pq_id, vport_id;
	u32 inc_val;
	u8 tc;

	/* Go over all PF VPORTs */
	for (vport_id = 0; vport_id < num_vports; vport_id++) {
		if (!vport_params[vport_id].wfq)
			continue;

		inc_val = QM_WFQ_INC_VAL(vport_params[vport_id].wfq);
		if (inc_val > QM_WFQ_MAX_INC_VAL) {
			DP_NOTICE(p_hwfn, true,
				  "Invalid VPORT WFQ weight configuration\n");
			return -1;
		}

		/* Each VPORT can have several VPORT PQ IDs for various TCs */
		for (tc = 0; tc < NUM_OF_TCS; tc++) {
			vp_pq_id = vport_params[vport_id].first_tx_pq_id[tc];
			if (vp_pq_id == QM_ILWALID_PQ_ID)
				continue;

			STORE_RT_REG(p_hwfn, QM_REG_WFQVPCRD_RT_OFFSET +
				     vp_pq_id, (u32)QM_WFQ_CRD_REG_SIGN_BIT);
			STORE_RT_REG(p_hwfn, QM_REG_WFQVPWEIGHT_RT_OFFSET +
				     vp_pq_id, inc_val);
		}
	}

	return 0;
}

static bool ecore_poll_on_qm_cmd_ready(struct ecore_hwfn *p_hwfn,
				       struct ecore_ptt *p_ptt)
{
	u32 reg_val, i;

	for (i = 0, reg_val = 0; i < QM_STOP_CMD_MAX_POLL_COUNT && !reg_val;
	     i++) {
		OSAL_UDELAY(QM_STOP_CMD_POLL_PERIOD_US);
		reg_val = ecore_rd(p_hwfn, p_ptt, QM_REG_SDMCMDREADY);
	}

	/* Check if timeout while waiting for SDM command ready */
	if (i == QM_STOP_CMD_MAX_POLL_COUNT) {
		DP_VERBOSE(p_hwfn, ECORE_MSG_DEBUG,
			   "Timeout waiting for QM SDM cmd ready signal\n");
		return false;
	}

	return true;
}

static bool ecore_send_qm_cmd(struct ecore_hwfn *p_hwfn,
			      struct ecore_ptt *p_ptt,
							  u32 cmd_addr,
							  u32 cmd_data_lsb,
							  u32 cmd_data_msb)
{
	if (!ecore_poll_on_qm_cmd_ready(p_hwfn, p_ptt))
		return false;

	ecore_wr(p_hwfn, p_ptt, QM_REG_SDMCMDADDR, cmd_addr);
	ecore_wr(p_hwfn, p_ptt, QM_REG_SDMCMDDATALSB, cmd_data_lsb);
	ecore_wr(p_hwfn, p_ptt, QM_REG_SDMCMDDATAMSB, cmd_data_msb);
	ecore_wr(p_hwfn, p_ptt, QM_REG_SDMCMDGO, 1);
	ecore_wr(p_hwfn, p_ptt, QM_REG_SDMCMDGO, 0);

	return ecore_poll_on_qm_cmd_ready(p_hwfn, p_ptt);
}

/******************** INTERFACE IMPLEMENTATION *********************/

u32 ecore_qm_pf_mem_size(struct ecore_hwfn *p_hwfn,
			 u32 num_pf_cids,
						 u32 num_vf_cids,
						 u32 num_tids,
						 u16 num_pf_pqs,
						 u16 num_vf_pqs)
{
	return QM_PQ_MEM_4KB(num_pf_cids) * num_pf_pqs +
	    QM_PQ_MEM_4KB(num_vf_cids) * num_vf_pqs +
	    QM_PQ_MEM_4KB(num_pf_cids + num_tids) * QM_OTHER_PQS_PER_PF;
}

int ecore_qm_common_rt_init(struct ecore_hwfn *p_hwfn,
			    u8 max_ports_per_engine,
			    u8 max_phys_tcs_per_port,
			    bool pf_rl_en,
			    bool pf_wfq_en,
			    bool global_rl_en,
			    bool vport_wfq_en,
			    struct init_qm_port_params
				   port_params[MAX_NUM_PORTS],
			    struct init_qm_global_rl_params
				   global_rl_params[COMMON_MAX_QM_GLOBAL_RLS])
{
	u32 mask = 0;

	/* Init AFullOprtnstcCrdMask */
	SET_FIELD(mask, QM_RF_OPPORTUNISTIC_MASK_LINEVOQ,
		  QM_OPPOR_LINE_VOQ_DEF);
	SET_FIELD(mask, QM_RF_OPPORTUNISTIC_MASK_BYTEVOQ, QM_BYTE_CRD_EN);
	SET_FIELD(mask, QM_RF_OPPORTUNISTIC_MASK_PFWFQ, pf_wfq_en);
	SET_FIELD(mask, QM_RF_OPPORTUNISTIC_MASK_VPWFQ, vport_wfq_en);
	SET_FIELD(mask, QM_RF_OPPORTUNISTIC_MASK_PFRL, pf_rl_en);
	SET_FIELD(mask, QM_RF_OPPORTUNISTIC_MASK_VPQCNRL, global_rl_en);
	SET_FIELD(mask, QM_RF_OPPORTUNISTIC_MASK_FWPAUSE, QM_OPPOR_FW_STOP_DEF);
	SET_FIELD(mask, QM_RF_OPPORTUNISTIC_MASK_QUEUEEMPTY,
		  QM_OPPOR_PQ_EMPTY_DEF);
	STORE_RT_REG(p_hwfn, QM_REG_AFULLOPRTNSTCCRDMASK_RT_OFFSET, mask);

	/* Enable/disable PF RL */
	ecore_enable_pf_rl(p_hwfn, pf_rl_en);

	/* Enable/disable PF WFQ */
	ecore_enable_pf_wfq(p_hwfn, pf_wfq_en);

	/* Enable/disable global RL */
	ecore_enable_global_rl(p_hwfn, global_rl_en);

	/* Enable/disable VPORT WFQ */
	ecore_enable_vport_wfq(p_hwfn, vport_wfq_en);

	/* Init PBF CMDQ line credit */
	ecore_cmdq_lines_rt_init(p_hwfn, max_ports_per_engine,
				 max_phys_tcs_per_port, port_params);

	/* Init BTB blocks in PBF */
	ecore_btb_blocks_rt_init(p_hwfn, max_ports_per_engine,
				 max_phys_tcs_per_port, port_params);

	ecore_global_rl_rt_init(p_hwfn, global_rl_params);

	return 0;
}

int ecore_qm_pf_rt_init(struct ecore_hwfn *p_hwfn,
			struct ecore_ptt *p_ptt,
			u8 pf_id,
			u8 max_phys_tcs_per_port,
			bool is_pf_loading,
			u32 num_pf_cids,
			u32 num_vf_cids,
			u32 num_tids,
			u16 start_pq,
			u16 num_pf_pqs,
			u16 num_vf_pqs,
			u16 start_vport,
			u16 num_vports,
			u16 pf_wfq,
			u32 pf_rl,
			struct init_qm_pq_params *pq_params,
			struct init_qm_vport_params *vport_params)
{
	u32 other_mem_size_4kb;
	u16 vport_id;
	u8 tc;

	other_mem_size_4kb = QM_PQ_MEM_4KB(num_pf_cids + num_tids) *
			     QM_OTHER_PQS_PER_PF;

	/* Clear first Tx PQ ID array for each VPORT */
	for (vport_id = 0; vport_id < num_vports; vport_id++)
		for (tc = 0; tc < NUM_OF_TCS; tc++)
			vport_params[vport_id].first_tx_pq_id[tc] =
				QM_ILWALID_PQ_ID;

	/* Map Other PQs (if any) */
#if QM_OTHER_PQS_PER_PF > 0
	ecore_other_pq_map_rt_init(p_hwfn, pf_id, is_pf_loading, num_pf_cids,
				   num_tids, 0);
#endif

	/* Map Tx PQs */
	if (ecore_tx_pq_map_rt_init(p_hwfn, p_ptt, pf_id, max_phys_tcs_per_port,
				    is_pf_loading, num_pf_cids, num_vf_cids,
				    start_pq, num_pf_pqs, num_vf_pqs,
				    start_vport, other_mem_size_4kb, pq_params,
				    vport_params))
		return -1;

	/* Init PF WFQ */
	if (pf_wfq)
		if (ecore_pf_wfq_rt_init(p_hwfn, pf_id, pf_wfq,
					 max_phys_tcs_per_port,
					 num_pf_pqs + num_vf_pqs, pq_params))
			return -1;

	/* Init PF RL */
	if (ecore_pf_rl_rt_init(p_hwfn, pf_id, pf_rl))
		return -1;

	/* Init VPORT WFQ */
	if (ecore_vp_wfq_rt_init(p_hwfn, num_vports, vport_params))
		return -1;

	return 0;
}

int ecore_init_pf_wfq(struct ecore_hwfn *p_hwfn,
		      struct ecore_ptt *p_ptt, u8 pf_id, u16 pf_wfq)
{
	u32 inc_val;

	inc_val = QM_WFQ_INC_VAL(pf_wfq);
	if (!inc_val || inc_val > QM_WFQ_MAX_INC_VAL) {
		DP_NOTICE(p_hwfn, true,
			  "Invalid PF WFQ weight configuration\n");
		return -1;
	}

	ecore_wr(p_hwfn, p_ptt, QM_REG_WFQPFWEIGHT + pf_id * 4, inc_val);

	return 0;
}

int ecore_init_pf_rl(struct ecore_hwfn *p_hwfn,
		     struct ecore_ptt *p_ptt, u8 pf_id, u32 pf_rl)
{
	u32 inc_val;

	inc_val = QM_RL_INC_VAL(pf_rl);
	if (inc_val > QM_PF_RL_MAX_INC_VAL) {
		DP_NOTICE(p_hwfn, true,
			  "Invalid PF rate limit configuration\n");
		return -1;
	}

	ecore_wr(p_hwfn, p_ptt, QM_REG_RLPFCRD + pf_id * 4,
		 (u32)QM_RL_CRD_REG_SIGN_BIT);
	ecore_wr(p_hwfn, p_ptt, QM_REG_RLPFINCVAL + pf_id * 4, inc_val);

	return 0;
}

int ecore_init_vport_wfq(struct ecore_hwfn *p_hwfn,
			 struct ecore_ptt *p_ptt,
			 u16 first_tx_pq_id[NUM_OF_TCS],
			 u16 wfq)
{
	u16 vp_pq_id;
	u32 inc_val;
	u8 tc;

	inc_val = QM_WFQ_INC_VAL(wfq);
	if (!inc_val || inc_val > QM_WFQ_MAX_INC_VAL) {
		DP_NOTICE(p_hwfn, true,
			  "Invalid VPORT WFQ weight configuration\n");
		return -1;
	}

	/* A VPORT can have several VPORT PQ IDs for various TCs */
	for (tc = 0; tc < NUM_OF_TCS; tc++) {
		vp_pq_id = first_tx_pq_id[tc];
		if (vp_pq_id != QM_ILWALID_PQ_ID) {
			ecore_wr(p_hwfn, p_ptt,
				 QM_REG_WFQVPWEIGHT + vp_pq_id * 4, inc_val);
		}
	}

	return 0;
		}

int ecore_init_global_rl(struct ecore_hwfn *p_hwfn,
			 struct ecore_ptt *p_ptt,
			 u16 rl_id,
			 u32 rate_limit)
{
	u32 inc_val;

	inc_val = QM_RL_INC_VAL(rate_limit);
	if (inc_val > QM_VP_RL_MAX_INC_VAL(rate_limit)) {
		DP_NOTICE(p_hwfn, true, "Invalid rate limit configuration.\n");
		return -1;
	}

	ecore_wr(p_hwfn, p_ptt, QM_REG_RLGLBLCRD + rl_id * 4,
		 (u32)QM_RL_CRD_REG_SIGN_BIT);
	ecore_wr(p_hwfn, p_ptt, QM_REG_RLGLBLINCVAL + rl_id * 4, inc_val);

	return 0;
}

int ecore_init_vport_rl(struct ecore_hwfn *p_hwfn,
			struct ecore_ptt *p_ptt, u8 vport_id,
						u32 vport_rl,
						u32 link_speed)
{
	u32 inc_val, max_qm_global_rls = MAX_QM_GLOBAL_RLS;

	if (vport_id >= max_qm_global_rls) {
		DP_NOTICE(p_hwfn, true,
			  "Invalid VPORT ID for rate limiter configuration\n");
		return -1;
	}

	inc_val = QM_RL_INC_VAL(vport_rl ? vport_rl : link_speed);
	if (inc_val > QM_VP_RL_MAX_INC_VAL(link_speed)) {
		DP_NOTICE(p_hwfn, true,
			  "Invalid VPORT rate-limit configuration\n");
		return -1;
	}

	ecore_wr(p_hwfn, p_ptt, QM_REG_RLGLBLCRD + vport_id * 4,
		 (u32)QM_RL_CRD_REG_SIGN_BIT);
	ecore_wr(p_hwfn, p_ptt, QM_REG_RLGLBLINCVAL + vport_id * 4, inc_val);

	return 0;
}

bool ecore_send_qm_stop_cmd(struct ecore_hwfn *p_hwfn,
			    struct ecore_ptt *p_ptt,
			    bool is_release_cmd,
			    bool is_tx_pq, u16 start_pq, u16 num_pqs)
{
	u32 cmd_arr[QM_CMD_STRUCT_SIZE(QM_STOP_CMD)] = { 0 };
	u32 pq_mask = 0, last_pq, pq_id;

	last_pq = start_pq + num_pqs - 1;

	/* Set command's PQ type */
	QM_CMD_SET_FIELD(cmd_arr, QM_STOP_CMD, PQ_TYPE, is_tx_pq ? 0 : 1);

	/* Go over requested PQs */
	for (pq_id = start_pq; pq_id <= last_pq; pq_id++) {
		/* Set PQ bit in mask (stop command only) */
		if (!is_release_cmd)
			pq_mask |= (1 << (pq_id % QM_STOP_PQ_MASK_WIDTH));

		/* If last PQ or end of PQ mask, write command */
		if ((pq_id == last_pq) ||
		    (pq_id % QM_STOP_PQ_MASK_WIDTH ==
		    (QM_STOP_PQ_MASK_WIDTH - 1))) {
			QM_CMD_SET_FIELD(cmd_arr, QM_STOP_CMD, PAUSE_MASK,
					 pq_mask);
			QM_CMD_SET_FIELD(cmd_arr, QM_STOP_CMD, GROUP_ID,
					 pq_id / QM_STOP_PQ_MASK_WIDTH);
			if (!ecore_send_qm_cmd
			    (p_hwfn, p_ptt, QM_STOP_CMD_ADDR, cmd_arr[0],
			     cmd_arr[1]))
				return false;
			pq_mask = 0;
		}
	}

	return true;
}

#ifndef UNUSED_HSI_FUNC

/* NIG: ETS configuration constants */
#define NIG_TX_ETS_CLIENT_OFFSET	4
#define NIG_LB_ETS_CLIENT_OFFSET	1
#define NIG_ETS_MIN_WFQ_BYTES		1600

/* NIG: ETS constants */
#define NIG_ETS_UP_BOUND(weight, mtu) \
	(2 * ((weight) > (mtu) ? (weight) : (mtu)))

/* NIG: RL constants */

/* Byte base type value */
#define NIG_RL_BASE_TYPE		1

/* Period in us */
#define NIG_RL_PERIOD			1

/* Period in 25MHz cycles */
#define NIG_RL_PERIOD_CLK_25M		(25 * NIG_RL_PERIOD)

/* Rate in mbps */
#define NIG_RL_INC_VAL(rate)		(((rate) * NIG_RL_PERIOD) / 8)

#define NIG_RL_MAX_VAL(inc_val, mtu) \
	(2 * ((inc_val) > (mtu) ? (inc_val) : (mtu)))

/* NIG: packet prioritry configuration constants */
#define NIG_PRIORITY_MAP_TC_BITS	4


void ecore_init_nig_ets(struct ecore_hwfn *p_hwfn,
			struct ecore_ptt *p_ptt,
			struct init_ets_req *req, bool is_lb)
{
	u32 min_weight, tc_weight_base_addr, tc_weight_addr_diff;
	u32 tc_bound_base_addr, tc_bound_addr_diff;
	u8 sp_tc_map = 0, wfq_tc_map = 0;
	u8 tc, num_tc, tc_client_offset;

	num_tc = is_lb ? NUM_OF_TCS : NUM_OF_PHYS_TCS;
	tc_client_offset = is_lb ? NIG_LB_ETS_CLIENT_OFFSET :
				   NIG_TX_ETS_CLIENT_OFFSET;
	min_weight = 0xffffffff;
	tc_weight_base_addr = is_lb ? NIG_REG_LB_ARB_CREDIT_WEIGHT_0 :
				      NIG_REG_TX_ARB_CREDIT_WEIGHT_0;
	tc_weight_addr_diff = is_lb ? NIG_REG_LB_ARB_CREDIT_WEIGHT_1 -
				      NIG_REG_LB_ARB_CREDIT_WEIGHT_0 :
				      NIG_REG_TX_ARB_CREDIT_WEIGHT_1 -
				      NIG_REG_TX_ARB_CREDIT_WEIGHT_0;
	tc_bound_base_addr = is_lb ? NIG_REG_LB_ARB_CREDIT_UPPER_BOUND_0 :
				     NIG_REG_TX_ARB_CREDIT_UPPER_BOUND_0;
	tc_bound_addr_diff = is_lb ? NIG_REG_LB_ARB_CREDIT_UPPER_BOUND_1 -
				     NIG_REG_LB_ARB_CREDIT_UPPER_BOUND_0 :
				     NIG_REG_TX_ARB_CREDIT_UPPER_BOUND_1 -
				     NIG_REG_TX_ARB_CREDIT_UPPER_BOUND_0;

	for (tc = 0; tc < num_tc; tc++) {
		struct init_ets_tc_req *tc_req = &req->tc_req[tc];

		/* Update SP map */
		if (tc_req->use_sp)
			sp_tc_map |= (1 << tc);

		if (!tc_req->use_wfq)
			continue;

		/* Update WFQ map */
		wfq_tc_map |= (1 << tc);

		/* Find minimal weight */
		if (tc_req->weight < min_weight)
			min_weight = tc_req->weight;
	}

	/* Write SP map */
	ecore_wr(p_hwfn, p_ptt,
		 is_lb ? NIG_REG_LB_ARB_CLIENT_IS_STRICT :
		 NIG_REG_TX_ARB_CLIENT_IS_STRICT,
		 (sp_tc_map << tc_client_offset));

	/* Write WFQ map */
	ecore_wr(p_hwfn, p_ptt,
		 is_lb ? NIG_REG_LB_ARB_CLIENT_IS_SUBJECT2WFQ :
		 NIG_REG_TX_ARB_CLIENT_IS_SUBJECT2WFQ,
		 (wfq_tc_map << tc_client_offset));
	/* write WFQ weights */
	for (tc = 0; tc < num_tc; tc++, tc_client_offset++) {
		struct init_ets_tc_req *tc_req = &req->tc_req[tc];
		u32 byte_weight;

		if (!tc_req->use_wfq)
			continue;

		/* Translate weight to bytes */
		byte_weight = (NIG_ETS_MIN_WFQ_BYTES * tc_req->weight) /
			      min_weight;

		/* Write WFQ weight */
		ecore_wr(p_hwfn, p_ptt, tc_weight_base_addr +
			 tc_weight_addr_diff * tc_client_offset, byte_weight);

		/* Write WFQ upper bound */
		ecore_wr(p_hwfn, p_ptt, tc_bound_base_addr +
			 tc_bound_addr_diff * tc_client_offset,
			 NIG_ETS_UP_BOUND(byte_weight, req->mtu));
	}
}

void ecore_init_nig_lb_rl(struct ecore_hwfn *p_hwfn,
			  struct ecore_ptt *p_ptt,
			  struct init_nig_lb_rl_req *req)
{
	u32 ctrl, inc_val, reg_offset;
	u8 tc;

	/* Disable global MAC+LB RL */
	ctrl =
	    NIG_RL_BASE_TYPE <<
	    NIG_REG_TX_LB_GLBRATELIMIT_CTRL_TX_LB_GLBRATELIMIT_BASE_TYPE_SHIFT;
	ecore_wr(p_hwfn, p_ptt, NIG_REG_TX_LB_GLBRATELIMIT_CTRL, ctrl);

	/* Configure and enable global MAC+LB RL */
	if (req->lb_mac_rate) {
		/* Configure  */
		ecore_wr(p_hwfn, p_ptt, NIG_REG_TX_LB_GLBRATELIMIT_INC_PERIOD,
			 NIG_RL_PERIOD_CLK_25M);
		inc_val = NIG_RL_INC_VAL(req->lb_mac_rate);
		ecore_wr(p_hwfn, p_ptt, NIG_REG_TX_LB_GLBRATELIMIT_INC_VALUE,
			 inc_val);
		ecore_wr(p_hwfn, p_ptt, NIG_REG_TX_LB_GLBRATELIMIT_MAX_VALUE,
			 NIG_RL_MAX_VAL(inc_val, req->mtu));

		/* Enable */
		ctrl |=
		    1 <<
		    NIG_REG_TX_LB_GLBRATELIMIT_CTRL_TX_LB_GLBRATELIMIT_EN_SHIFT;
		ecore_wr(p_hwfn, p_ptt, NIG_REG_TX_LB_GLBRATELIMIT_CTRL, ctrl);
	}

	/* Disable global LB-only RL */
	ctrl =
	    NIG_RL_BASE_TYPE <<
	    NIG_REG_LB_BRBRATELIMIT_CTRL_LB_BRBRATELIMIT_BASE_TYPE_SHIFT;
	ecore_wr(p_hwfn, p_ptt, NIG_REG_LB_BRBRATELIMIT_CTRL, ctrl);

	/* Configure and enable global LB-only RL */
	if (req->lb_rate) {
		/* Configure  */
		ecore_wr(p_hwfn, p_ptt, NIG_REG_LB_BRBRATELIMIT_INC_PERIOD,
			 NIG_RL_PERIOD_CLK_25M);
		inc_val = NIG_RL_INC_VAL(req->lb_rate);
		ecore_wr(p_hwfn, p_ptt, NIG_REG_LB_BRBRATELIMIT_INC_VALUE,
			 inc_val);
		ecore_wr(p_hwfn, p_ptt, NIG_REG_LB_BRBRATELIMIT_MAX_VALUE,
			 NIG_RL_MAX_VAL(inc_val, req->mtu));

		/* Enable */
		ctrl |=
		    1 << NIG_REG_LB_BRBRATELIMIT_CTRL_LB_BRBRATELIMIT_EN_SHIFT;
		ecore_wr(p_hwfn, p_ptt, NIG_REG_LB_BRBRATELIMIT_CTRL, ctrl);
	}

	/* Per-TC RLs */
	for (tc = 0, reg_offset = 0; tc < NUM_OF_PHYS_TCS;
	     tc++, reg_offset += 4) {
		/* Disable TC RL */
		ctrl =
		    NIG_RL_BASE_TYPE <<
		NIG_REG_LB_TCRATELIMIT_CTRL_0_LB_TCRATELIMIT_BASE_TYPE_0_SHIFT;
		ecore_wr(p_hwfn, p_ptt,
			 NIG_REG_LB_TCRATELIMIT_CTRL_0 + reg_offset, ctrl);

		/* Configure and enable TC RL */
		if (!req->tc_rate[tc])
			continue;

		/* Configure */
		ecore_wr(p_hwfn, p_ptt, NIG_REG_LB_TCRATELIMIT_INC_PERIOD_0 +
			 reg_offset, NIG_RL_PERIOD_CLK_25M);
		inc_val = NIG_RL_INC_VAL(req->tc_rate[tc]);
		ecore_wr(p_hwfn, p_ptt, NIG_REG_LB_TCRATELIMIT_INC_VALUE_0 +
			 reg_offset, inc_val);
		ecore_wr(p_hwfn, p_ptt, NIG_REG_LB_TCRATELIMIT_MAX_VALUE_0 +
			 reg_offset, NIG_RL_MAX_VAL(inc_val, req->mtu));

		/* Enable */
		ctrl |= 1 <<
			NIG_REG_LB_TCRATELIMIT_CTRL_0_LB_TCRATELIMIT_EN_0_SHIFT;
		ecore_wr(p_hwfn, p_ptt, NIG_REG_LB_TCRATELIMIT_CTRL_0 +
			 reg_offset, ctrl);
	}
}

void ecore_init_nig_pri_tc_map(struct ecore_hwfn *p_hwfn,
			       struct ecore_ptt *p_ptt,
			       struct init_nig_pri_tc_map_req *req)
{
	u8 tc_pri_mask[NUM_OF_PHYS_TCS] = { 0 };
	u32 pri_tc_mask = 0;
	u8 pri, tc;

	for (pri = 0; pri < NUM_OF_VLAN_PRIORITIES; pri++) {
		if (!req->pri[pri].valid)
			continue;

		pri_tc_mask |= (req->pri[pri].tc_id <<
				(pri * NIG_PRIORITY_MAP_TC_BITS));
		tc_pri_mask[req->pri[pri].tc_id] |= (1 << pri);
	}

	/* Write priority -> TC mask */
	ecore_wr(p_hwfn, p_ptt, NIG_REG_PKT_PRIORITY_TO_TC, pri_tc_mask);

	/* Write TC -> priority mask */
	for (tc = 0; tc < NUM_OF_PHYS_TCS; tc++) {
		ecore_wr(p_hwfn, p_ptt, NIG_REG_PRIORITY_FOR_TC_0 + tc * 4,
			 tc_pri_mask[tc]);
		ecore_wr(p_hwfn, p_ptt, NIG_REG_RX_TC0_PRIORITY_MASK + tc * 4,
			 tc_pri_mask[tc]);
	}
}

#endif /* UNUSED_HSI_FUNC */

#ifndef UNUSED_HSI_FUNC

/* PRS: ETS configuration constants */
#define PRS_ETS_MIN_WFQ_BYTES		1600
#define PRS_ETS_UP_BOUND(weight, mtu) \
	(2 * ((weight) > (mtu) ? (weight) : (mtu)))


void ecore_init_prs_ets(struct ecore_hwfn *p_hwfn,
			struct ecore_ptt *p_ptt, struct init_ets_req *req)
{
	u32 tc_weight_addr_diff, tc_bound_addr_diff, min_weight = 0xffffffff;
	u8 tc, sp_tc_map = 0, wfq_tc_map = 0;

	tc_weight_addr_diff = PRS_REG_ETS_ARB_CREDIT_WEIGHT_1 -
			      PRS_REG_ETS_ARB_CREDIT_WEIGHT_0;
	tc_bound_addr_diff = PRS_REG_ETS_ARB_CREDIT_UPPER_BOUND_1 -
			     PRS_REG_ETS_ARB_CREDIT_UPPER_BOUND_0;

	for (tc = 0; tc < NUM_OF_TCS; tc++) {
		struct init_ets_tc_req *tc_req = &req->tc_req[tc];

		/* Update SP map */
		if (tc_req->use_sp)
			sp_tc_map |= (1 << tc);

		if (!tc_req->use_wfq)
			continue;

		/* Update WFQ map */
		wfq_tc_map |= (1 << tc);

		/* Find minimal weight */
		if (tc_req->weight < min_weight)
			min_weight = tc_req->weight;
	}

	/* write SP map */
	ecore_wr(p_hwfn, p_ptt, PRS_REG_ETS_ARB_CLIENT_IS_STRICT, sp_tc_map);

	/* write WFQ map */
	ecore_wr(p_hwfn, p_ptt, PRS_REG_ETS_ARB_CLIENT_IS_SUBJECT2WFQ,
		 wfq_tc_map);

	/* write WFQ weights */
	for (tc = 0; tc < NUM_OF_TCS; tc++) {
		struct init_ets_tc_req *tc_req = &req->tc_req[tc];
		u32 byte_weight;

		if (!tc_req->use_wfq)
			continue;

		/* Translate weight to bytes */
		byte_weight = (PRS_ETS_MIN_WFQ_BYTES * tc_req->weight) /
			      min_weight;

		/* Write WFQ weight */
		ecore_wr(p_hwfn, p_ptt, PRS_REG_ETS_ARB_CREDIT_WEIGHT_0 + tc *
			 tc_weight_addr_diff, byte_weight);

		/* Write WFQ upper bound */
		ecore_wr(p_hwfn, p_ptt, PRS_REG_ETS_ARB_CREDIT_UPPER_BOUND_0 +
			 tc * tc_bound_addr_diff, PRS_ETS_UP_BOUND(byte_weight,
								   req->mtu));
	}
}

#endif /* UNUSED_HSI_FUNC */
#ifndef UNUSED_HSI_FUNC

/* BRB: RAM configuration constants */
#define BRB_TOTAL_RAM_BLOCKS_BB	4800
#define BRB_TOTAL_RAM_BLOCKS_K2	5632
#define BRB_BLOCK_SIZE		128
#define BRB_MIN_BLOCKS_PER_TC	9
#define BRB_HYST_BYTES		10240
#define BRB_HYST_BLOCKS		(BRB_HYST_BYTES / BRB_BLOCK_SIZE)

/* Temporary big RAM allocation - should be updated */
void ecore_init_brb_ram(struct ecore_hwfn *p_hwfn,
			struct ecore_ptt *p_ptt, struct init_brb_ram_req *req)
{
	u32 tc_headroom_blocks, min_pkt_size_blocks, total_blocks;
	u32 active_port_blocks, reg_offset = 0;
	u8 port, active_ports = 0;

	tc_headroom_blocks = (u32)DIV_ROUND_UP(req->headroom_per_tc,
					       BRB_BLOCK_SIZE);
	min_pkt_size_blocks = (u32)DIV_ROUND_UP(req->min_pkt_size,
						BRB_BLOCK_SIZE);
	total_blocks = ECORE_IS_K2(p_hwfn->p_dev) ? BRB_TOTAL_RAM_BLOCKS_K2 :
						    BRB_TOTAL_RAM_BLOCKS_BB;

	/* Find number of active ports */
	for (port = 0; port < MAX_NUM_PORTS; port++)
		if (req->num_active_tcs[port])
			active_ports++;

	active_port_blocks = (u32)(total_blocks / active_ports);

	for (port = 0; port < req->max_ports_per_engine; port++) {
		u32 port_blocks, port_shared_blocks, port_guaranteed_blocks;
		u32 full_xoff_th, full_xon_th, pause_xoff_th, pause_xon_th;
		u32 tc_guaranteed_blocks;
		u8 tc;

		/* Callwlate per-port sizes */
		tc_guaranteed_blocks = (u32)DIV_ROUND_UP(req->guranteed_per_tc,
							 BRB_BLOCK_SIZE);
		port_blocks = req->num_active_tcs[port] ? active_port_blocks :
							  0;
		port_guaranteed_blocks = req->num_active_tcs[port] *
					 tc_guaranteed_blocks;
		port_shared_blocks = port_blocks - port_guaranteed_blocks;
		full_xoff_th = req->num_active_tcs[port] *
			       BRB_MIN_BLOCKS_PER_TC;
		full_xon_th = full_xoff_th + min_pkt_size_blocks;
		pause_xoff_th = tc_headroom_blocks;
		pause_xon_th = pause_xoff_th + min_pkt_size_blocks;

		/* Init total size per port */
		ecore_wr(p_hwfn, p_ptt, BRB_REG_TOTAL_MAC_SIZE + port * 4,
			 port_blocks);

		/* Init shared size per port */
		ecore_wr(p_hwfn, p_ptt, BRB_REG_SHARED_HR_AREA + port * 4,
			 port_shared_blocks);

		for (tc = 0; tc < NUM_OF_TCS; tc++, reg_offset += 4) {
			/* Clear init values for non-active TCs */
			if (tc == req->num_active_tcs[port]) {
				tc_guaranteed_blocks = 0;
				full_xoff_th = 0;
				full_xon_th = 0;
				pause_xoff_th = 0;
				pause_xon_th = 0;
			}

			/* Init guaranteed size per TC */
			ecore_wr(p_hwfn, p_ptt,
				 BRB_REG_TC_GUARANTIED_0 + reg_offset,
				 tc_guaranteed_blocks);
			ecore_wr(p_hwfn, p_ptt,
				 BRB_REG_MAIN_TC_GUARANTIED_HYST_0 + reg_offset,
				 BRB_HYST_BLOCKS);

			/* Init pause/full thresholds per physical TC - for
			 * loopback traffic.
			 */
			ecore_wr(p_hwfn, p_ptt,
				 BRB_REG_LB_TC_FULL_XOFF_THRESHOLD_0 +
				 reg_offset, full_xoff_th);
			ecore_wr(p_hwfn, p_ptt,
				 BRB_REG_LB_TC_FULL_XON_THRESHOLD_0 +
				 reg_offset, full_xon_th);
			ecore_wr(p_hwfn, p_ptt,
				 BRB_REG_LB_TC_PAUSE_XOFF_THRESHOLD_0 +
				 reg_offset, pause_xoff_th);
			ecore_wr(p_hwfn, p_ptt,
				 BRB_REG_LB_TC_PAUSE_XON_THRESHOLD_0 +
				 reg_offset, pause_xon_th);

			/* Init pause/full thresholds per physical TC - for
			 * main traffic.
			 */
			ecore_wr(p_hwfn, p_ptt,
				 BRB_REG_MAIN_TC_FULL_XOFF_THRESHOLD_0 +
				 reg_offset, full_xoff_th);
			ecore_wr(p_hwfn, p_ptt,
				 BRB_REG_MAIN_TC_FULL_XON_THRESHOLD_0 +
				 reg_offset, full_xon_th);
			ecore_wr(p_hwfn, p_ptt,
				 BRB_REG_MAIN_TC_PAUSE_XOFF_THRESHOLD_0 +
				 reg_offset, pause_xoff_th);
			ecore_wr(p_hwfn, p_ptt,
				 BRB_REG_MAIN_TC_PAUSE_XON_THRESHOLD_0 +
				 reg_offset, pause_xon_th);
		}
	}
}

#endif /* UNUSED_HSI_FUNC */
#ifndef UNUSED_HSI_FUNC

#define ARR_REG_WR(dev, ptt, addr, arr, arr_size)		\
	do {							\
		u32 i;						\
		for (i = 0; i < (arr_size); i++)		\
			ecore_wr(dev, ptt, ((addr) + (4 * i)),	\
				 ((u32 *)&(arr))[i]);		\
	} while (0)

#ifndef DWORDS_TO_BYTES
#define DWORDS_TO_BYTES(dwords)		((dwords) * REG_SIZE)
#endif


/**
 * @brief ecore_dmae_to_grc - is an internal function - writes from host to
 * wide-bus registers (split registers are not supported yet)
 *
 * @param p_hwfn -       HW device data
 * @param p_ptt -       ptt window used for writing the registers.
 * @param pData - pointer to source data.
 * @param addr - Destination register address.
 * @param len_in_dwords - data length in DWARDS (u32)
 */
static int ecore_dmae_to_grc(struct ecore_hwfn *p_hwfn,
			     struct ecore_ptt *p_ptt,
			     u32 *pData,
			     u32 addr,
			     u32 len_in_dwords)
{
	struct dmae_params params;
	bool read_using_dmae = false;

	if (!pData)
		return -1;

	/* Set DMAE params */
	OSAL_MEMSET(&params, 0, sizeof(params));

	SET_FIELD(params.flags, DMAE_PARAMS_COMPLETION_DST, 1);

	/* Execute DMAE command */
	read_using_dmae = !ecore_dmae_host2grc(p_hwfn, p_ptt,
					       (u64)(osal_uintptr_t)(pData),
					       addr, len_in_dwords, &params);
	if (!read_using_dmae)
		DP_VERBOSE(p_hwfn, ECORE_MSG_DEBUG,
			   "Failed writing to chip using DMAE, using GRC instead\n");

	/* If not read using DMAE, read using GRC */
	if (!read_using_dmae)
		/* write to registers using GRC */
		ARR_REG_WR(p_hwfn, p_ptt, addr, pData, len_in_dwords);

	return len_in_dwords;
}

/* In MF, should be called once per port to set EtherType of OuterTag */
void ecore_set_port_mf_ovlan_eth_type(struct ecore_hwfn *p_hwfn, u32 ethType)
{
	/* Update DORQ register */
	STORE_RT_REG(p_hwfn, DORQ_REG_TAG1_ETHERTYPE_RT_OFFSET, ethType);
}

#endif /* UNUSED_HSI_FUNC */

#define SET_TUNNEL_TYPE_ENABLE_BIT(var, offset, enable) \
(var = ((var) & ~(1 << (offset))) | ((enable) ? (1 << (offset)) : 0))
#define PRS_ETH_TUNN_OUTPUT_FORMAT        -188897008
#define PRS_ETH_OUTPUT_FORMAT             -46832

void ecore_set_vxlan_dest_port(struct ecore_hwfn *p_hwfn,
			       struct ecore_ptt *p_ptt, u16 dest_port)
{
	/* Update PRS register */
	ecore_wr(p_hwfn, p_ptt, PRS_REG_VXLAN_PORT, dest_port);

	/* Update NIG register */
	ecore_wr(p_hwfn, p_ptt, NIG_REG_VXLAN_CTRL, dest_port);

	/* Update PBF register */
	ecore_wr(p_hwfn, p_ptt, PBF_REG_VXLAN_PORT, dest_port);
}

void ecore_set_vxlan_enable(struct ecore_hwfn *p_hwfn,
			    struct ecore_ptt *p_ptt, bool vxlan_enable)
{
	u32 reg_val;

	/* Update PRS register */
	reg_val = ecore_rd(p_hwfn, p_ptt, PRS_REG_ENCAPSULATION_TYPE_EN);
	SET_TUNNEL_TYPE_ENABLE_BIT(reg_val,
			   PRS_REG_ENCAPSULATION_TYPE_EN_VXLAN_ENABLE_SHIFT,
			   vxlan_enable);
	ecore_wr(p_hwfn, p_ptt, PRS_REG_ENCAPSULATION_TYPE_EN, reg_val);
	if (reg_val) { /* TODO: handle E5 init */
		reg_val = ecore_rd(p_hwfn, p_ptt,
				   PRS_REG_OUTPUT_FORMAT_4_0_BB_K2);

		/* Update output  only if tunnel blocks not included. */
		if (reg_val == (u32)PRS_ETH_OUTPUT_FORMAT)
			ecore_wr(p_hwfn, p_ptt, PRS_REG_OUTPUT_FORMAT_4_0_BB_K2,
				 (u32)PRS_ETH_TUNN_OUTPUT_FORMAT);
	}

	/* Update NIG register */
	reg_val = ecore_rd(p_hwfn, p_ptt, NIG_REG_ENC_TYPE_ENABLE);
	SET_TUNNEL_TYPE_ENABLE_BIT(reg_val,
				   NIG_REG_ENC_TYPE_ENABLE_VXLAN_ENABLE_SHIFT,
				   vxlan_enable);
	ecore_wr(p_hwfn, p_ptt, NIG_REG_ENC_TYPE_ENABLE, reg_val);

	/* Update DORQ register */
	ecore_wr(p_hwfn, p_ptt, DORQ_REG_L2_EDPM_TUNNEL_VXLAN_EN,
		 vxlan_enable ? 1 : 0);
}

void ecore_set_gre_enable(struct ecore_hwfn *p_hwfn,
			  struct ecore_ptt *p_ptt,
			  bool eth_gre_enable, bool ip_gre_enable)
{
	u32 reg_val;

	/* Update PRS register */
	reg_val = ecore_rd(p_hwfn, p_ptt, PRS_REG_ENCAPSULATION_TYPE_EN);
	SET_TUNNEL_TYPE_ENABLE_BIT(reg_val,
		   PRS_REG_ENCAPSULATION_TYPE_EN_ETH_OVER_GRE_ENABLE_SHIFT,
		   eth_gre_enable);
	SET_TUNNEL_TYPE_ENABLE_BIT(reg_val,
		   PRS_REG_ENCAPSULATION_TYPE_EN_IP_OVER_GRE_ENABLE_SHIFT,
		   ip_gre_enable);
	ecore_wr(p_hwfn, p_ptt, PRS_REG_ENCAPSULATION_TYPE_EN, reg_val);
	if (reg_val) { /* TODO: handle E5 init */
		reg_val = ecore_rd(p_hwfn, p_ptt,
				   PRS_REG_OUTPUT_FORMAT_4_0_BB_K2);

		/* Update output  only if tunnel blocks not included. */
		if (reg_val == (u32)PRS_ETH_OUTPUT_FORMAT)
			ecore_wr(p_hwfn, p_ptt, PRS_REG_OUTPUT_FORMAT_4_0_BB_K2,
				 (u32)PRS_ETH_TUNN_OUTPUT_FORMAT);
	}

	/* Update NIG register */
	reg_val = ecore_rd(p_hwfn, p_ptt, NIG_REG_ENC_TYPE_ENABLE);
	SET_TUNNEL_TYPE_ENABLE_BIT(reg_val,
		   NIG_REG_ENC_TYPE_ENABLE_ETH_OVER_GRE_ENABLE_SHIFT,
		   eth_gre_enable);
	SET_TUNNEL_TYPE_ENABLE_BIT(reg_val,
		   NIG_REG_ENC_TYPE_ENABLE_IP_OVER_GRE_ENABLE_SHIFT,
		   ip_gre_enable);
	ecore_wr(p_hwfn, p_ptt, NIG_REG_ENC_TYPE_ENABLE, reg_val);

	/* Update DORQ registers */
	ecore_wr(p_hwfn, p_ptt, DORQ_REG_L2_EDPM_TUNNEL_GRE_ETH_EN,
		 eth_gre_enable ? 1 : 0);
	ecore_wr(p_hwfn, p_ptt, DORQ_REG_L2_EDPM_TUNNEL_GRE_IP_EN,
		 ip_gre_enable ? 1 : 0);
}

void ecore_set_geneve_dest_port(struct ecore_hwfn *p_hwfn,
				struct ecore_ptt *p_ptt, u16 dest_port)
{
	/* Update PRS register */
	ecore_wr(p_hwfn, p_ptt, PRS_REG_NGE_PORT, dest_port);

	/* Update NIG register */
	ecore_wr(p_hwfn, p_ptt, NIG_REG_NGE_PORT, dest_port);

	/* Update PBF register */
	ecore_wr(p_hwfn, p_ptt, PBF_REG_NGE_PORT, dest_port);
}

void ecore_set_geneve_enable(struct ecore_hwfn *p_hwfn,
			     struct ecore_ptt *p_ptt,
			     bool eth_geneve_enable, bool ip_geneve_enable)
{
	u32 reg_val;

	/* Update PRS register */
	reg_val = ecore_rd(p_hwfn, p_ptt, PRS_REG_ENCAPSULATION_TYPE_EN);
	SET_TUNNEL_TYPE_ENABLE_BIT(reg_val,
		   PRS_REG_ENCAPSULATION_TYPE_EN_ETH_OVER_GENEVE_ENABLE_SHIFT,
		   eth_geneve_enable);
	SET_TUNNEL_TYPE_ENABLE_BIT(reg_val,
		   PRS_REG_ENCAPSULATION_TYPE_EN_IP_OVER_GENEVE_ENABLE_SHIFT,
		   ip_geneve_enable);
	ecore_wr(p_hwfn, p_ptt, PRS_REG_ENCAPSULATION_TYPE_EN, reg_val);
	if (reg_val) { /* TODO: handle E5 init */
		reg_val = ecore_rd(p_hwfn, p_ptt,
				   PRS_REG_OUTPUT_FORMAT_4_0_BB_K2);

		/* Update output  only if tunnel blocks not included. */
		if (reg_val == (u32)PRS_ETH_OUTPUT_FORMAT)
			ecore_wr(p_hwfn, p_ptt, PRS_REG_OUTPUT_FORMAT_4_0_BB_K2,
				 (u32)PRS_ETH_TUNN_OUTPUT_FORMAT);
	}

	/* Update NIG register */
	ecore_wr(p_hwfn, p_ptt, NIG_REG_NGE_ETH_ENABLE,
		 eth_geneve_enable ? 1 : 0);
	ecore_wr(p_hwfn, p_ptt, NIG_REG_NGE_IP_ENABLE,
		 ip_geneve_enable ? 1 : 0);

	/* EDPM with geneve tunnel not supported in BB */
	if (ECORE_IS_BB_B0(p_hwfn->p_dev))
		return;

	/* Update DORQ registers */
	ecore_wr(p_hwfn, p_ptt, DORQ_REG_L2_EDPM_TUNNEL_NGE_ETH_EN_K2,
		 eth_geneve_enable ? 1 : 0);
	ecore_wr(p_hwfn, p_ptt, DORQ_REG_L2_EDPM_TUNNEL_NGE_IP_EN_K2,
		 ip_geneve_enable ? 1 : 0);
}

#define PRS_ETH_VXLAN_NO_L2_ENABLE_OFFSET      3
#define PRS_ETH_VXLAN_NO_L2_OUTPUT_FORMAT   -925189872

void ecore_set_vxlan_no_l2_enable(struct ecore_hwfn *p_hwfn,
				  struct ecore_ptt *p_ptt,
				  bool enable)
{
	u32 reg_val, cfg_mask;

	/* read PRS config register */
	reg_val = ecore_rd(p_hwfn, p_ptt, PRS_REG_MSG_INFO);

	/* set VXLAN_NO_L2_ENABLE mask */
	cfg_mask = (1 << PRS_ETH_VXLAN_NO_L2_ENABLE_OFFSET);

	if (enable) {
		/* set VXLAN_NO_L2_ENABLE flag */
		reg_val |= cfg_mask;

		/* update PRS FIC Format register */
		ecore_wr(p_hwfn, p_ptt, PRS_REG_OUTPUT_FORMAT_4_0_BB_K2,
		 (u32)PRS_ETH_VXLAN_NO_L2_OUTPUT_FORMAT);
		/* clear VXLAN_NO_L2_ENABLE flag */
		reg_val &= ~cfg_mask;
	}

	/* write PRS config register */
	ecore_wr(p_hwfn, p_ptt, PRS_REG_MSG_INFO, reg_val);
}

#ifndef UNUSED_HSI_FUNC

#define T_ETH_PACKET_ACTION_GFT_EVENTID  23
#define PARSER_ETH_CONN_GFT_ACTION_CM_HDR  272
#define T_ETH_PACKET_MATCH_RFS_EVENTID 25
#define PARSER_ETH_CONN_CM_HDR 0
#define CAM_LINE_SIZE sizeof(u32)
#define RAM_LINE_SIZE sizeof(u64)
#define REG_SIZE sizeof(u32)

void ecore_gft_disable(struct ecore_hwfn *p_hwfn,
		       struct ecore_ptt *p_ptt,
		       u16 pf_id)
{
	struct regpair ram_line;
	OSAL_MEMSET(&ram_line, 0, sizeof(ram_line));

	/* disable gft search for PF */
	ecore_wr(p_hwfn, p_ptt, PRS_REG_SEARCH_GFT, 0);

	/* Clean ram & cam for next gft session*/

	/* Zero camline */
	ecore_wr(p_hwfn, p_ptt, PRS_REG_GFT_CAM + CAM_LINE_SIZE * pf_id, 0);

	/* Zero ramline */
	ecore_dmae_to_grc(p_hwfn, p_ptt, (u32 *)&ram_line,
			  PRS_REG_GFT_PROFILE_MASK_RAM + RAM_LINE_SIZE * pf_id,
			  sizeof(ram_line) / REG_SIZE);

}


void ecore_set_gft_event_id_cm_hdr(struct ecore_hwfn *p_hwfn,
				   struct ecore_ptt *p_ptt)
{
	u32 rfs_cm_hdr_event_id;

	/* Set RFS event ID to be awakened i Tstorm By Prs */
	rfs_cm_hdr_event_id = ecore_rd(p_hwfn, p_ptt, PRS_REG_CM_HDR_GFT);
	rfs_cm_hdr_event_id |= T_ETH_PACKET_ACTION_GFT_EVENTID <<
	    PRS_REG_CM_HDR_GFT_EVENT_ID_SHIFT;
	rfs_cm_hdr_event_id |= PARSER_ETH_CONN_GFT_ACTION_CM_HDR <<
	    PRS_REG_CM_HDR_GFT_CM_HDR_SHIFT;
	ecore_wr(p_hwfn, p_ptt, PRS_REG_CM_HDR_GFT, rfs_cm_hdr_event_id);
}

void ecore_gft_config(struct ecore_hwfn *p_hwfn,
			       struct ecore_ptt *p_ptt,
			       u16 pf_id,
			       bool tcp,
			       bool udp,
			       bool ipv4,
			       bool ipv6,
			       enum gft_profile_type profile_type)
{
	u32 reg_val, cam_line, search_non_ip_as_gft;
	struct regpair ram_line = { 0 };

	if (!ipv6 && !ipv4)
		DP_NOTICE(p_hwfn, true, "gft_config: must accept at least on of - ipv4 or ipv6'\n");
	if (!tcp && !udp)
		DP_NOTICE(p_hwfn, true, "gft_config: must accept at least on of - udp or tcp\n");
	if (profile_type >= MAX_GFT_PROFILE_TYPE)
		DP_NOTICE(p_hwfn, true, "gft_config: unsupported gft_profile_type\n");

	/* Set RFS event ID to be awakened i Tstorm By Prs */
	reg_val = T_ETH_PACKET_MATCH_RFS_EVENTID <<
		  PRS_REG_CM_HDR_GFT_EVENT_ID_SHIFT;
	reg_val |= PARSER_ETH_CONN_CM_HDR << PRS_REG_CM_HDR_GFT_CM_HDR_SHIFT;
	ecore_wr(p_hwfn, p_ptt, PRS_REG_CM_HDR_GFT, reg_val);

	/* Do not load context only cid in PRS on match. */
	ecore_wr(p_hwfn, p_ptt, PRS_REG_LOAD_L2_FILTER, 0);

	/* Do not use tenant ID exist bit for gft search*/
	ecore_wr(p_hwfn, p_ptt, PRS_REG_SEARCH_TENANT_ID, 0);

	/* Set Cam */
	cam_line = 0;
	SET_FIELD(cam_line, GFT_CAM_LINE_MAPPED_VALID, 1);

	/* Filters are per PF!! */
	SET_FIELD(cam_line, GFT_CAM_LINE_MAPPED_PF_ID_MASK,
		  GFT_CAM_LINE_MAPPED_PF_ID_MASK_MASK);
	SET_FIELD(cam_line, GFT_CAM_LINE_MAPPED_PF_ID, pf_id);

	if (!(tcp && udp)) {
		SET_FIELD(cam_line,
			  GFT_CAM_LINE_MAPPED_UPPER_PROTOCOL_TYPE_MASK,
			  GFT_CAM_LINE_MAPPED_UPPER_PROTOCOL_TYPE_MASK_MASK);
		if (tcp)
			SET_FIELD(cam_line,
				  GFT_CAM_LINE_MAPPED_UPPER_PROTOCOL_TYPE,
				  GFT_PROFILE_TCP_PROTOCOL);
		else
			SET_FIELD(cam_line,
				  GFT_CAM_LINE_MAPPED_UPPER_PROTOCOL_TYPE,
				  GFT_PROFILE_UDP_PROTOCOL);
	}

	if (!(ipv4 && ipv6)) {
		SET_FIELD(cam_line, GFT_CAM_LINE_MAPPED_IP_VERSION_MASK, 1);
		if (ipv4)
			SET_FIELD(cam_line, GFT_CAM_LINE_MAPPED_IP_VERSION,
				  GFT_PROFILE_IPV4);
		else
			SET_FIELD(cam_line, GFT_CAM_LINE_MAPPED_IP_VERSION,
				  GFT_PROFILE_IPV6);
	}

	/* Write characteristics to cam */
	ecore_wr(p_hwfn, p_ptt, PRS_REG_GFT_CAM + CAM_LINE_SIZE * pf_id,
		 cam_line);
	cam_line = ecore_rd(p_hwfn, p_ptt,
			    PRS_REG_GFT_CAM + CAM_LINE_SIZE * pf_id);

	/* Write line to RAM - compare to filter 4 tuple */

	/* Search no IP as GFT */
	search_non_ip_as_gft = 0;

	/* Tunnel type */
	SET_FIELD(ram_line.lo, GFT_RAM_LINE_TUNNEL_DST_PORT, 1);
	SET_FIELD(ram_line.lo, GFT_RAM_LINE_TUNNEL_OVER_IP_PROTOCOL, 1);

	if (profile_type == GFT_PROFILE_TYPE_4_TUPLE) {
		SET_FIELD(ram_line.hi, GFT_RAM_LINE_DST_IP, 1);
		SET_FIELD(ram_line.hi, GFT_RAM_LINE_SRC_IP, 1);
		SET_FIELD(ram_line.hi, GFT_RAM_LINE_OVER_IP_PROTOCOL, 1);
		SET_FIELD(ram_line.lo, GFT_RAM_LINE_ETHERTYPE, 1);
		SET_FIELD(ram_line.lo, GFT_RAM_LINE_SRC_PORT, 1);
		SET_FIELD(ram_line.lo, GFT_RAM_LINE_DST_PORT, 1);
	} else if (profile_type == GFT_PROFILE_TYPE_L4_DST_PORT) {
		SET_FIELD(ram_line.hi, GFT_RAM_LINE_OVER_IP_PROTOCOL, 1);
		SET_FIELD(ram_line.lo, GFT_RAM_LINE_ETHERTYPE, 1);
		SET_FIELD(ram_line.lo, GFT_RAM_LINE_DST_PORT, 1);
	} else if (profile_type == GFT_PROFILE_TYPE_IP_DST_ADDR) {
		SET_FIELD(ram_line.hi, GFT_RAM_LINE_DST_IP, 1);
		SET_FIELD(ram_line.lo, GFT_RAM_LINE_ETHERTYPE, 1);
	} else if (profile_type == GFT_PROFILE_TYPE_IP_SRC_ADDR) {
		SET_FIELD(ram_line.hi, GFT_RAM_LINE_SRC_IP, 1);
		SET_FIELD(ram_line.lo, GFT_RAM_LINE_ETHERTYPE, 1);
	} else if (profile_type == GFT_PROFILE_TYPE_TUNNEL_TYPE) {
		SET_FIELD(ram_line.lo, GFT_RAM_LINE_TUNNEL_ETHERTYPE, 1);

		/* Allow tunneled traffic without inner IP */
		search_non_ip_as_gft = 1;
	}

	ecore_wr(p_hwfn, p_ptt, PRS_REG_SEARCH_NON_IP_AS_GFT,
		 search_non_ip_as_gft);
	ecore_dmae_to_grc(p_hwfn, p_ptt, (u32 *)&ram_line,
			  PRS_REG_GFT_PROFILE_MASK_RAM + RAM_LINE_SIZE * pf_id,
			  sizeof(ram_line) / REG_SIZE);

	/* Set default profile so that no filter match will happen */
	ram_line.lo = 0xffffffff;
	ram_line.hi = 0x3ff;
	ecore_dmae_to_grc(p_hwfn, p_ptt, (u32 *)&ram_line,
			  PRS_REG_GFT_PROFILE_MASK_RAM + RAM_LINE_SIZE *
			  PRS_GFT_CAM_LINES_NO_MATCH,
			  sizeof(ram_line) / REG_SIZE);

	/* Enable gft search */
	ecore_wr(p_hwfn, p_ptt, PRS_REG_SEARCH_GFT, 1);
}


#endif /* UNUSED_HSI_FUNC */

/* Configure VF zone size mode */
void ecore_config_vf_zone_size_mode(struct ecore_hwfn *p_hwfn,
				    struct ecore_ptt *p_ptt, u16 mode,
				    bool runtime_init)
{
	u32 msdm_vf_size_log = MSTORM_VF_ZONE_DEFAULT_SIZE_LOG;
	u32 msdm_vf_offset_mask;

	if (mode == VF_ZONE_SIZE_MODE_DOUBLE)
		msdm_vf_size_log += 1;
	else if (mode == VF_ZONE_SIZE_MODE_QUAD)
		msdm_vf_size_log += 2;

	msdm_vf_offset_mask = (1 << msdm_vf_size_log) - 1;

	if (runtime_init) {
		STORE_RT_REG(p_hwfn,
			     PGLUE_REG_B_MSDM_VF_SHIFT_B_RT_OFFSET,
			     msdm_vf_size_log);
		STORE_RT_REG(p_hwfn,
			     PGLUE_REG_B_MSDM_OFFSET_MASK_B_RT_OFFSET,
			     msdm_vf_offset_mask);
	} else {
		ecore_wr(p_hwfn, p_ptt,
			 PGLUE_B_REG_MSDM_VF_SHIFT_B, msdm_vf_size_log);
		ecore_wr(p_hwfn, p_ptt,
			 PGLUE_B_REG_MSDM_OFFSET_MASK_B, msdm_vf_offset_mask);
	}
}

/* Get mstorm statistics for offset by VF zone size mode */
u32 ecore_get_mstorm_queue_stat_offset(struct ecore_hwfn *p_hwfn,
				       u16 stat_cnt_id,
				       u16 vf_zone_size_mode)
{
	u32 offset = MSTORM_QUEUE_STAT_OFFSET(stat_cnt_id);

	if ((vf_zone_size_mode != VF_ZONE_SIZE_MODE_DEFAULT) &&
	    (stat_cnt_id > MAX_NUM_PFS)) {
		if (vf_zone_size_mode == VF_ZONE_SIZE_MODE_DOUBLE)
			offset += (1 << MSTORM_VF_ZONE_DEFAULT_SIZE_LOG) *
			    (stat_cnt_id - MAX_NUM_PFS);
		else if (vf_zone_size_mode == VF_ZONE_SIZE_MODE_QUAD)
			offset += 3 * (1 << MSTORM_VF_ZONE_DEFAULT_SIZE_LOG) *
			    (stat_cnt_id - MAX_NUM_PFS);
	}

	return offset;
}

/* Get mstorm VF producer offset by VF zone size mode */
u32 ecore_get_mstorm_eth_vf_prods_offset(struct ecore_hwfn *p_hwfn,
					 u8 vf_id,
					 u8 vf_queue_id,
					 u16 vf_zone_size_mode)
{
	u32 offset = MSTORM_ETH_VF_PRODS_OFFSET(vf_id, vf_queue_id);

	if (vf_zone_size_mode != VF_ZONE_SIZE_MODE_DEFAULT) {
		if (vf_zone_size_mode == VF_ZONE_SIZE_MODE_DOUBLE)
			offset += (1 << MSTORM_VF_ZONE_DEFAULT_SIZE_LOG) *
				   vf_id;
		else if (vf_zone_size_mode == VF_ZONE_SIZE_MODE_QUAD)
			offset += 3 * (1 << MSTORM_VF_ZONE_DEFAULT_SIZE_LOG) *
				  vf_id;
	}

	return offset;
}

#ifndef LINUX_REMOVE
#define CRC8_INIT_VALUE 0xFF
#endif
static u8 cdu_crc8_table[CRC8_TABLE_SIZE];

/* Callwlate and return CDU validation byte per connection type / region /
 * cid
 */
static u8 ecore_calc_cdu_validation_byte(struct ecore_hwfn *p_hwfn,
					 u8 conn_type, u8 region, u32 cid)
{
	static u8 crc8_table_valid;	/*automatically initialized to 0*/
	u8 crc, validation_byte = 0;
	u32 validation_string = 0;
	u32 data_to_crc;

	if (crc8_table_valid == 0) {
		OSAL_CRC8_POPULATE(cdu_crc8_table, 0x07);
		crc8_table_valid = 1;
	}

	/*
	 * The CRC is callwlated on the String-to-compress:
	 * [31:8]  = {CID[31:20],CID[11:0]}
	 * [7:4]   = Region
	 * [3:0]   = Type
	 */
#if ((CDU_CONTEXT_VALIDATION_DEFAULT_CFG >> \
	CDU_CONTEXT_VALIDATION_CFG_USE_CID) & 1)
	validation_string |= (cid & 0xFFF00000) | ((cid & 0xFFF) << 8);
#endif

#if ((CDU_CONTEXT_VALIDATION_DEFAULT_CFG >> \
	CDU_CONTEXT_VALIDATION_CFG_USE_REGION) & 1)
	validation_string |= ((region & 0xF) << 4);
#endif

#if ((CDU_CONTEXT_VALIDATION_DEFAULT_CFG >> \
	CDU_CONTEXT_VALIDATION_CFG_USE_TYPE) & 1)
	validation_string |= (conn_type & 0xF);
#endif
	/* Colwert to big-endian and callwlate CRC8*/
	data_to_crc = OSAL_BE32_TO_CPU(validation_string);

	crc = OSAL_CRC8(cdu_crc8_table, (u8 *)&data_to_crc, sizeof(data_to_crc),
			CRC8_INIT_VALUE);

	/* The validation byte [7:0] is composed:
	 * for type A validation
	 * [7]		= active configuration bit
	 * [6:0]	= crc[6:0]
	 *
	 * for type B validation
	 * [7]		= active configuration bit
	 * [6:3]	= connection_type[3:0]
	 * [2:0]	= crc[2:0]
	 */
	validation_byte |= ((CDU_CONTEXT_VALIDATION_DEFAULT_CFG >>
			     CDU_CONTEXT_VALIDATION_CFG_USE_ACTIVE) & 1) << 7;

#if ((CDU_CONTEXT_VALIDATION_DEFAULT_CFG >> \
	CDU_CONTEXT_VALIDATION_CFG_VALIDATION_TYPE_SHIFT) & 1)
	validation_byte |= ((conn_type & 0xF) << 3) | (crc & 0x7);
#else
	validation_byte |= crc & 0x7F;
#endif
	return validation_byte;
}

/* Callwalte and set validation bytes for session context */
void ecore_calc_session_ctx_validation(struct ecore_hwfn *p_hwfn,
				       void *p_ctx_mem, u16 ctx_size,
				       u8 ctx_type, u32 cid)
{
	u8 *x_val_ptr, *t_val_ptr, *u_val_ptr, *p_ctx;

	p_ctx = (u8 *)p_ctx_mem;

	x_val_ptr = &p_ctx[con_region_offsets[0][ctx_type]];
	t_val_ptr = &p_ctx[con_region_offsets[1][ctx_type]];
	u_val_ptr = &p_ctx[con_region_offsets[2][ctx_type]];

	OSAL_MEMSET(p_ctx, 0, ctx_size);

	*x_val_ptr = ecore_calc_cdu_validation_byte(p_hwfn, ctx_type, 3, cid);
	*t_val_ptr = ecore_calc_cdu_validation_byte(p_hwfn, ctx_type, 4, cid);
	*u_val_ptr = ecore_calc_cdu_validation_byte(p_hwfn, ctx_type, 5, cid);
}

/* Callwalte and set validation bytes for task context */
void ecore_calc_task_ctx_validation(struct ecore_hwfn *p_hwfn, void *p_ctx_mem,
				    u16 ctx_size, u8 ctx_type, u32 tid)
{
	u8 *p_ctx, *region1_val_ptr;

	p_ctx = (u8 *)p_ctx_mem;
	region1_val_ptr = &p_ctx[task_region_offsets[0][ctx_type]];

	OSAL_MEMSET(p_ctx, 0, ctx_size);

	*region1_val_ptr = ecore_calc_cdu_validation_byte(p_hwfn, ctx_type, 1,
							  tid);
}

/* Memset session context to 0 while preserving validation bytes */
void ecore_memset_session_ctx(struct ecore_hwfn *p_hwfn, void *p_ctx_mem,
			      u32 ctx_size, u8 ctx_type)
{
	u8 *x_val_ptr, *t_val_ptr, *u_val_ptr, *p_ctx;
	u8 x_val, t_val, u_val;

	p_ctx = (u8 *)p_ctx_mem;

	x_val_ptr = &p_ctx[con_region_offsets[0][ctx_type]];
	t_val_ptr = &p_ctx[con_region_offsets[1][ctx_type]];
	u_val_ptr = &p_ctx[con_region_offsets[2][ctx_type]];

	x_val = *x_val_ptr;
	t_val = *t_val_ptr;
	u_val = *u_val_ptr;

	OSAL_MEMSET(p_ctx, 0, ctx_size);

	*x_val_ptr = x_val;
	*t_val_ptr = t_val;
	*u_val_ptr = u_val;
}

/* Memset task context to 0 while preserving validation bytes */
void ecore_memset_task_ctx(struct ecore_hwfn *p_hwfn, void *p_ctx_mem,
			   u32 ctx_size, u8 ctx_type)
{
	u8 *p_ctx, *region1_val_ptr;
	u8 region1_val;

	p_ctx = (u8 *)p_ctx_mem;
	region1_val_ptr = &p_ctx[task_region_offsets[0][ctx_type]];

	region1_val = *region1_val_ptr;

	OSAL_MEMSET(p_ctx, 0, ctx_size);

	*region1_val_ptr = region1_val;
}

/* Enable and configure context validation */
void ecore_enable_context_validation(struct ecore_hwfn *p_hwfn,
				     struct ecore_ptt *p_ptt)
{
	u32 ctx_validation;

	/* Enable validation for connection region 3: CCFC_CTX_VALID0[31:24] */
	ctx_validation = CDU_CONTEXT_VALIDATION_DEFAULT_CFG << 24;
	ecore_wr(p_hwfn, p_ptt, CDU_REG_CCFC_CTX_VALID0, ctx_validation);

	/* Enable validation for connection region 5: CCFC_CTX_VALID1[15:8] */
	ctx_validation = CDU_CONTEXT_VALIDATION_DEFAULT_CFG << 8;
	ecore_wr(p_hwfn, p_ptt, CDU_REG_CCFC_CTX_VALID1, ctx_validation);

	/* Enable validation for connection region 1: TCFC_CTX_VALID0[15:8] */
	ctx_validation = CDU_CONTEXT_VALIDATION_DEFAULT_CFG << 8;
	ecore_wr(p_hwfn, p_ptt, CDU_REG_TCFC_CTX_VALID0, ctx_validation);
}

#define PHYS_ADDR_DWORDS        DIV_ROUND_UP(sizeof(dma_addr_t), 4)
#define OVERLAY_HDR_SIZE_DWORDS (sizeof(struct fw_overlay_buf_hdr) / 4)

static u32 ecore_get_overlay_addr_ram_addr(struct ecore_hwfn *p_hwfn,
					   u8 storm_id)
{
	switch (storm_id) {
	case 0: return TSEM_REG_FAST_MEMORY + SEM_FAST_REG_INT_RAM +
			TSTORM_OVERLAY_BUF_ADDR_OFFSET;
	case 1: return MSEM_REG_FAST_MEMORY + SEM_FAST_REG_INT_RAM +
			MSTORM_OVERLAY_BUF_ADDR_OFFSET;
	case 2: return USEM_REG_FAST_MEMORY + SEM_FAST_REG_INT_RAM +
			USTORM_OVERLAY_BUF_ADDR_OFFSET;
	case 3: return XSEM_REG_FAST_MEMORY + SEM_FAST_REG_INT_RAM +
			XSTORM_OVERLAY_BUF_ADDR_OFFSET;
	case 4: return YSEM_REG_FAST_MEMORY + SEM_FAST_REG_INT_RAM +
			YSTORM_OVERLAY_BUF_ADDR_OFFSET;
	case 5: return PSEM_REG_FAST_MEMORY + SEM_FAST_REG_INT_RAM +
			PSTORM_OVERLAY_BUF_ADDR_OFFSET;

	default: return 0;
	}
}

struct phys_mem_desc *ecore_fw_overlay_mem_alloc(struct ecore_hwfn *p_hwfn,
					 const u32 *const fw_overlay_in_buf,
					 u32 buf_size_in_bytes)
{
	u32 buf_size = buf_size_in_bytes / sizeof(u32), buf_offset = 0;
	struct phys_mem_desc *allocated_mem;

	if (!buf_size)
		return OSAL_NULL;

	allocated_mem = (struct phys_mem_desc *)OSAL_ZALLOC(p_hwfn->p_dev,
							    GFP_KERNEL,
							    NUM_STORMS *
						  sizeof(struct phys_mem_desc));
	if (!allocated_mem)
		return OSAL_NULL;

	OSAL_MEMSET(allocated_mem, 0, NUM_STORMS *
		    sizeof(struct phys_mem_desc));

	/* For each Storm, set physical address in RAM */
	while (buf_offset < buf_size) {
		struct phys_mem_desc *storm_mem_desc;
		struct fw_overlay_buf_hdr *hdr;
		u32 storm_buf_size;
		u8 storm_id;

		hdr =
		    (struct fw_overlay_buf_hdr *)&fw_overlay_in_buf[buf_offset];
		storm_buf_size = GET_FIELD(hdr->data,
					   FW_OVERLAY_BUF_HDR_BUF_SIZE);
		storm_id = GET_FIELD(hdr->data, FW_OVERLAY_BUF_HDR_STORM_ID);
		storm_mem_desc = allocated_mem + storm_id;
		storm_mem_desc->size = storm_buf_size * sizeof(u32);

		/* Allocate physical memory for Storm's overlays buffer */
		storm_mem_desc->virt_addr =
			OSAL_DMA_ALLOC_COHERENT(p_hwfn->p_dev,
						&storm_mem_desc->phys_addr,
						storm_mem_desc->size);
		if (!storm_mem_desc->virt_addr)
			break;

		/* Skip overlays buffer header */
		buf_offset += OVERLAY_HDR_SIZE_DWORDS;

		/* Copy Storm's overlays buffer to allocated memory */
		OSAL_MEMCPY(storm_mem_desc->virt_addr,
			    &fw_overlay_in_buf[buf_offset],
			    storm_mem_desc->size);

		/* Advance to next Storm */
		buf_offset += storm_buf_size;
	}

	/* If memory allocation has failed, free all allocated memory */
	if (buf_offset < buf_size) {
		ecore_fw_overlay_mem_free(p_hwfn, allocated_mem);
		return OSAL_NULL;
	}

	return allocated_mem;
}

void ecore_fw_overlay_init_ram(struct ecore_hwfn *p_hwfn,
			       struct ecore_ptt *p_ptt,
			       struct phys_mem_desc *fw_overlay_mem)
{
	u8 storm_id;

	for (storm_id = 0; storm_id < NUM_STORMS; storm_id++) {
		struct phys_mem_desc *storm_mem_desc =
			      (struct phys_mem_desc *)fw_overlay_mem + storm_id;
		u32 ram_addr, i;

		/* Skip Storms with no FW overlays */
		if (!storm_mem_desc->virt_addr)
			continue;

		/* Callwlate overlay RAM GRC address of current PF */
		ram_addr = ecore_get_overlay_addr_ram_addr(p_hwfn, storm_id) +
			   sizeof(dma_addr_t) * p_hwfn->rel_pf_id;

		/* Write Storm's overlay physical address to RAM */
		for (i = 0; i < PHYS_ADDR_DWORDS; i++, ram_addr += sizeof(u32))
			ecore_wr(p_hwfn, p_ptt, ram_addr,
				 ((u32 *)&storm_mem_desc->phys_addr)[i]);
	}
}

void ecore_fw_overlay_mem_free(struct ecore_hwfn *p_hwfn,
			       struct phys_mem_desc *fw_overlay_mem)
{
	u8 storm_id;

	if (!fw_overlay_mem)
		return;

	for (storm_id = 0; storm_id < NUM_STORMS; storm_id++) {
		struct phys_mem_desc *storm_mem_desc =
			      (struct phys_mem_desc *)fw_overlay_mem + storm_id;

		/* Free Storm's physical memory */
		if (storm_mem_desc->virt_addr)
			OSAL_DMA_FREE_COHERENT(p_hwfn->p_dev,
					       storm_mem_desc->virt_addr,
					       storm_mem_desc->phys_addr,
					       storm_mem_desc->size);
	}

	/* Free allocated virtual memory */
	OSAL_FREE(p_hwfn->p_dev, fw_overlay_mem);
}
