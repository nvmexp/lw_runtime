/* SPDX-License-Identifier: BSD-3-Clause
 * Copyright(c) 2014-2018 Broadcom
 * All rights reserved.
 */

#include <inttypes.h>
#include <stdbool.h>

#include <rte_dev.h>
#include <rte_ethdev_driver.h>
#include <rte_ethdev_pci.h>
#include <rte_malloc.h>
#include <rte_cycles.h>
#include <rte_alarm.h>
#include <rte_kvargs.h>
#include <rte_vect.h>

#include "bnxt.h"
#include "bnxt_filter.h"
#include "bnxt_hwrm.h"
#include "bnxt_irq.h"
#include "bnxt_reps.h"
#include "bnxt_ring.h"
#include "bnxt_rxq.h"
#include "bnxt_rxr.h"
#include "bnxt_stats.h"
#include "bnxt_txq.h"
#include "bnxt_txr.h"
#include "bnxt_vnic.h"
#include "hsi_struct_def_dpdk.h"
#include "bnxt_lwm_defs.h"
#include "bnxt_tf_common.h"
#include "ulp_flow_db.h"
#include "rte_pmd_bnxt.h"

#define DRV_MODULE_NAME		"bnxt"
static const char bnxt_version[] =
	"Broadcom NetXtreme driver " DRV_MODULE_NAME;

/*
 * The set of PCI devices this driver supports
 */
static const struct rte_pci_id bnxt_pci_id_map[] = {
	{ RTE_PCI_DEVICE(PCI_VENDOR_ID_BROADCOM,
			 BROADCOM_DEV_ID_STRATUS_NIC_VF1) },
	{ RTE_PCI_DEVICE(PCI_VENDOR_ID_BROADCOM,
			 BROADCOM_DEV_ID_STRATUS_NIC_VF2) },
	{ RTE_PCI_DEVICE(PCI_VENDOR_ID_BROADCOM, BROADCOM_DEV_ID_STRATUS_NIC) },
	{ RTE_PCI_DEVICE(PCI_VENDOR_ID_BROADCOM, BROADCOM_DEV_ID_57414_VF) },
	{ RTE_PCI_DEVICE(PCI_VENDOR_ID_BROADCOM, BROADCOM_DEV_ID_57301) },
	{ RTE_PCI_DEVICE(PCI_VENDOR_ID_BROADCOM, BROADCOM_DEV_ID_57302) },
	{ RTE_PCI_DEVICE(PCI_VENDOR_ID_BROADCOM, BROADCOM_DEV_ID_57304_PF) },
	{ RTE_PCI_DEVICE(PCI_VENDOR_ID_BROADCOM, BROADCOM_DEV_ID_57304_VF) },
	{ RTE_PCI_DEVICE(PCI_VENDOR_ID_BROADCOM, BROADCOM_DEV_ID_NS2) },
	{ RTE_PCI_DEVICE(PCI_VENDOR_ID_BROADCOM, BROADCOM_DEV_ID_57402) },
	{ RTE_PCI_DEVICE(PCI_VENDOR_ID_BROADCOM, BROADCOM_DEV_ID_57404) },
	{ RTE_PCI_DEVICE(PCI_VENDOR_ID_BROADCOM, BROADCOM_DEV_ID_57406_PF) },
	{ RTE_PCI_DEVICE(PCI_VENDOR_ID_BROADCOM, BROADCOM_DEV_ID_57406_VF) },
	{ RTE_PCI_DEVICE(PCI_VENDOR_ID_BROADCOM, BROADCOM_DEV_ID_57402_MF) },
	{ RTE_PCI_DEVICE(PCI_VENDOR_ID_BROADCOM, BROADCOM_DEV_ID_57407_RJ45) },
	{ RTE_PCI_DEVICE(PCI_VENDOR_ID_BROADCOM, BROADCOM_DEV_ID_57404_MF) },
	{ RTE_PCI_DEVICE(PCI_VENDOR_ID_BROADCOM, BROADCOM_DEV_ID_57406_MF) },
	{ RTE_PCI_DEVICE(PCI_VENDOR_ID_BROADCOM, BROADCOM_DEV_ID_57407_SFP) },
	{ RTE_PCI_DEVICE(PCI_VENDOR_ID_BROADCOM, BROADCOM_DEV_ID_57407_MF) },
	{ RTE_PCI_DEVICE(PCI_VENDOR_ID_BROADCOM, BROADCOM_DEV_ID_5741X_VF) },
	{ RTE_PCI_DEVICE(PCI_VENDOR_ID_BROADCOM, BROADCOM_DEV_ID_5731X_VF) },
	{ RTE_PCI_DEVICE(PCI_VENDOR_ID_BROADCOM, BROADCOM_DEV_ID_57314) },
	{ RTE_PCI_DEVICE(PCI_VENDOR_ID_BROADCOM, BROADCOM_DEV_ID_57417_MF) },
	{ RTE_PCI_DEVICE(PCI_VENDOR_ID_BROADCOM, BROADCOM_DEV_ID_57311) },
	{ RTE_PCI_DEVICE(PCI_VENDOR_ID_BROADCOM, BROADCOM_DEV_ID_57312) },
	{ RTE_PCI_DEVICE(PCI_VENDOR_ID_BROADCOM, BROADCOM_DEV_ID_57412) },
	{ RTE_PCI_DEVICE(PCI_VENDOR_ID_BROADCOM, BROADCOM_DEV_ID_57414) },
	{ RTE_PCI_DEVICE(PCI_VENDOR_ID_BROADCOM, BROADCOM_DEV_ID_57416_RJ45) },
	{ RTE_PCI_DEVICE(PCI_VENDOR_ID_BROADCOM, BROADCOM_DEV_ID_57417_RJ45) },
	{ RTE_PCI_DEVICE(PCI_VENDOR_ID_BROADCOM, BROADCOM_DEV_ID_57412_MF) },
	{ RTE_PCI_DEVICE(PCI_VENDOR_ID_BROADCOM, BROADCOM_DEV_ID_57317_RJ45) },
	{ RTE_PCI_DEVICE(PCI_VENDOR_ID_BROADCOM, BROADCOM_DEV_ID_57417_SFP) },
	{ RTE_PCI_DEVICE(PCI_VENDOR_ID_BROADCOM, BROADCOM_DEV_ID_57416_SFP) },
	{ RTE_PCI_DEVICE(PCI_VENDOR_ID_BROADCOM, BROADCOM_DEV_ID_57317_SFP) },
	{ RTE_PCI_DEVICE(PCI_VENDOR_ID_BROADCOM, BROADCOM_DEV_ID_57414_MF) },
	{ RTE_PCI_DEVICE(PCI_VENDOR_ID_BROADCOM, BROADCOM_DEV_ID_57416_MF) },
	{ RTE_PCI_DEVICE(PCI_VENDOR_ID_BROADCOM, BROADCOM_DEV_ID_58802) },
	{ RTE_PCI_DEVICE(PCI_VENDOR_ID_BROADCOM, BROADCOM_DEV_ID_58804) },
	{ RTE_PCI_DEVICE(PCI_VENDOR_ID_BROADCOM, BROADCOM_DEV_ID_58808) },
	{ RTE_PCI_DEVICE(PCI_VENDOR_ID_BROADCOM, BROADCOM_DEV_ID_58802_VF) },
	{ RTE_PCI_DEVICE(PCI_VENDOR_ID_BROADCOM, BROADCOM_DEV_ID_57508) },
	{ RTE_PCI_DEVICE(PCI_VENDOR_ID_BROADCOM, BROADCOM_DEV_ID_57504) },
	{ RTE_PCI_DEVICE(PCI_VENDOR_ID_BROADCOM, BROADCOM_DEV_ID_57502) },
	{ RTE_PCI_DEVICE(PCI_VENDOR_ID_BROADCOM, BROADCOM_DEV_ID_57500_VF1) },
	{ RTE_PCI_DEVICE(PCI_VENDOR_ID_BROADCOM, BROADCOM_DEV_ID_57500_VF2) },
	{ RTE_PCI_DEVICE(PCI_VENDOR_ID_BROADCOM, BROADCOM_DEV_ID_57508_MF1) },
	{ RTE_PCI_DEVICE(PCI_VENDOR_ID_BROADCOM, BROADCOM_DEV_ID_57504_MF1) },
	{ RTE_PCI_DEVICE(PCI_VENDOR_ID_BROADCOM, BROADCOM_DEV_ID_57502_MF1) },
	{ RTE_PCI_DEVICE(PCI_VENDOR_ID_BROADCOM, BROADCOM_DEV_ID_57508_MF2) },
	{ RTE_PCI_DEVICE(PCI_VENDOR_ID_BROADCOM, BROADCOM_DEV_ID_57504_MF2) },
	{ RTE_PCI_DEVICE(PCI_VENDOR_ID_BROADCOM, BROADCOM_DEV_ID_57502_MF2) },
	{ .vendor_id = 0, /* sentinel */ },
};

#define BNXT_DEVARG_TRUFLOW	"host-based-truflow"
#define BNXT_DEVARG_FLOW_XSTAT	"flow-xstat"
#define BNXT_DEVARG_MAX_NUM_KFLOWS  "max-num-kflows"
#define BNXT_DEVARG_REPRESENTOR	"representor"
#define BNXT_DEVARG_REP_BASED_PF  "rep-based-pf"
#define BNXT_DEVARG_REP_IS_PF  "rep-is-pf"
#define BNXT_DEVARG_REP_Q_R2F  "rep-q-r2f"
#define BNXT_DEVARG_REP_Q_F2R  "rep-q-f2r"
#define BNXT_DEVARG_REP_FC_R2F  "rep-fc-r2f"
#define BNXT_DEVARG_REP_FC_F2R  "rep-fc-f2r"

static const char *const bnxt_dev_args[] = {
	BNXT_DEVARG_REPRESENTOR,
	BNXT_DEVARG_TRUFLOW,
	BNXT_DEVARG_FLOW_XSTAT,
	BNXT_DEVARG_MAX_NUM_KFLOWS,
	BNXT_DEVARG_REP_BASED_PF,
	BNXT_DEVARG_REP_IS_PF,
	BNXT_DEVARG_REP_Q_R2F,
	BNXT_DEVARG_REP_Q_F2R,
	BNXT_DEVARG_REP_FC_R2F,
	BNXT_DEVARG_REP_FC_F2R,
	NULL
};

/*
 * truflow == false to disable the feature
 * truflow == true to enable the feature
 */
#define	BNXT_DEVARG_TRUFLOW_ILWALID(truflow)	((truflow) > 1)

/*
 * flow_xstat == false to disable the feature
 * flow_xstat == true to enable the feature
 */
#define	BNXT_DEVARG_FLOW_XSTAT_ILWALID(flow_xstat)	((flow_xstat) > 1)

/*
 * rep_is_pf == false to indicate VF representor
 * rep_is_pf == true to indicate PF representor
 */
#define	BNXT_DEVARG_REP_IS_PF_ILWALID(rep_is_pf)	((rep_is_pf) > 1)

/*
 * rep_based_pf == Physical index of the PF
 */
#define	BNXT_DEVARG_REP_BASED_PF_ILWALID(rep_based_pf)	((rep_based_pf) > 15)
/*
 * rep_q_r2f == Logical COS Queue index for the rep to endpoint direction
 */
#define	BNXT_DEVARG_REP_Q_R2F_ILWALID(rep_q_r2f)	((rep_q_r2f) > 3)

/*
 * rep_q_f2r == Logical COS Queue index for the endpoint to rep direction
 */
#define	BNXT_DEVARG_REP_Q_F2R_ILWALID(rep_q_f2r)	((rep_q_f2r) > 3)

/*
 * rep_fc_r2f == Flow control for the representor to endpoint direction
 */
#define BNXT_DEVARG_REP_FC_R2F_ILWALID(rep_fc_r2f)	((rep_fc_r2f) > 1)

/*
 * rep_fc_f2r == Flow control for the endpoint to representor direction
 */
#define BNXT_DEVARG_REP_FC_F2R_ILWALID(rep_fc_f2r)	((rep_fc_f2r) > 1)

int bnxt_cfa_code_dynfield_offset = -1;

/*
 * max_num_kflows must be >= 32
 * and must be a power-of-2 supported value
 * return: 1 -> invalid
 *         0 -> valid
 */
static int bnxt_devarg_max_num_kflow_ilwalid(uint16_t max_num_kflows)
{
	if (max_num_kflows < 32 || !rte_is_power_of_2(max_num_kflows))
		return 1;
	return 0;
}

static int bnxt_vlan_offload_set_op(struct rte_eth_dev *dev, int mask);
static int bnxt_dev_uninit(struct rte_eth_dev *eth_dev);
static int bnxt_init_resources(struct bnxt *bp, bool reconfig_dev);
static int bnxt_uninit_resources(struct bnxt *bp, bool reconfig_dev);
static void bnxt_cancel_fw_health_check(struct bnxt *bp);
static int bnxt_restore_vlan_filters(struct bnxt *bp);
static void bnxt_dev_recover(void *arg);
static void bnxt_free_error_recovery_info(struct bnxt *bp);
static void bnxt_free_rep_info(struct bnxt *bp);

int is_bnxt_in_error(struct bnxt *bp)
{
	if (bp->flags & BNXT_FLAG_FATAL_ERROR)
		return -EIO;
	if (bp->flags & BNXT_FLAG_FW_RESET)
		return -EBUSY;

	return 0;
}

/***********************/

/*
 * High level utility functions
 */

static uint16_t bnxt_rss_ctxts(const struct bnxt *bp)
{
	if (!BNXT_CHIP_THOR(bp))
		return 1;

	return RTE_ALIGN_MUL_CEIL(bp->rx_nr_rings,
				  BNXT_RSS_ENTRIES_PER_CTX_THOR) /
				    BNXT_RSS_ENTRIES_PER_CTX_THOR;
}

uint16_t bnxt_rss_hash_tbl_size(const struct bnxt *bp)
{
	if (!BNXT_CHIP_THOR(bp))
		return HW_HASH_INDEX_SIZE;

	return bnxt_rss_ctxts(bp) * BNXT_RSS_ENTRIES_PER_CTX_THOR;
}

static void bnxt_free_parent_info(struct bnxt *bp)
{
	rte_free(bp->parent);
}

static void bnxt_free_pf_info(struct bnxt *bp)
{
	rte_free(bp->pf);
}

static void bnxt_free_link_info(struct bnxt *bp)
{
	rte_free(bp->link_info);
}

static void bnxt_free_leds_info(struct bnxt *bp)
{
	if (BNXT_VF(bp))
		return;

	rte_free(bp->leds);
	bp->leds = NULL;
}

static void bnxt_free_flow_stats_info(struct bnxt *bp)
{
	rte_free(bp->flow_stat);
	bp->flow_stat = NULL;
}

static void bnxt_free_cos_queues(struct bnxt *bp)
{
	rte_free(bp->rx_cos_queue);
	rte_free(bp->tx_cos_queue);
}

static void bnxt_free_mem(struct bnxt *bp, bool reconfig)
{
	bnxt_free_filter_mem(bp);
	bnxt_free_vnic_attributes(bp);
	bnxt_free_vnic_mem(bp);

	/* tx/rx rings are configured as part of *_queue_setup callbacks.
	 * If the number of rings change across fw update,
	 * we don't have much choice except to warn the user.
	 */
	if (!reconfig) {
		bnxt_free_stats(bp);
		bnxt_free_tx_rings(bp);
		bnxt_free_rx_rings(bp);
	}
	bnxt_free_async_cp_ring(bp);
	bnxt_free_rxtx_nq_ring(bp);

	rte_free(bp->grp_info);
	bp->grp_info = NULL;
}

static int bnxt_alloc_parent_info(struct bnxt *bp)
{
	bp->parent = rte_zmalloc("bnxt_parent_info",
				 sizeof(struct bnxt_parent_info), 0);
	if (bp->parent == NULL)
		return -ENOMEM;

	return 0;
}

static int bnxt_alloc_pf_info(struct bnxt *bp)
{
	bp->pf = rte_zmalloc("bnxt_pf_info", sizeof(struct bnxt_pf_info), 0);
	if (bp->pf == NULL)
		return -ENOMEM;

	return 0;
}

static int bnxt_alloc_link_info(struct bnxt *bp)
{
	bp->link_info =
		rte_zmalloc("bnxt_link_info", sizeof(struct bnxt_link_info), 0);
	if (bp->link_info == NULL)
		return -ENOMEM;

	return 0;
}

static int bnxt_alloc_leds_info(struct bnxt *bp)
{
	if (BNXT_VF(bp))
		return 0;

	bp->leds = rte_zmalloc("bnxt_leds",
			       BNXT_MAX_LED * sizeof(struct bnxt_led_info),
			       0);
	if (bp->leds == NULL)
		return -ENOMEM;

	return 0;
}

static int bnxt_alloc_cos_queues(struct bnxt *bp)
{
	bp->rx_cos_queue =
		rte_zmalloc("bnxt_rx_cosq",
			    BNXT_COS_QUEUE_COUNT *
			    sizeof(struct bnxt_cos_queue_info),
			    0);
	if (bp->rx_cos_queue == NULL)
		return -ENOMEM;

	bp->tx_cos_queue =
		rte_zmalloc("bnxt_tx_cosq",
			    BNXT_COS_QUEUE_COUNT *
			    sizeof(struct bnxt_cos_queue_info),
			    0);
	if (bp->tx_cos_queue == NULL)
		return -ENOMEM;

	return 0;
}

static int bnxt_alloc_flow_stats_info(struct bnxt *bp)
{
	bp->flow_stat = rte_zmalloc("bnxt_flow_xstat",
				    sizeof(struct bnxt_flow_stat_info), 0);
	if (bp->flow_stat == NULL)
		return -ENOMEM;

	return 0;
}

static int bnxt_alloc_mem(struct bnxt *bp, bool reconfig)
{
	int rc;

	rc = bnxt_alloc_ring_grps(bp);
	if (rc)
		goto alloc_mem_err;

	rc = bnxt_alloc_async_ring_struct(bp);
	if (rc)
		goto alloc_mem_err;

	rc = bnxt_alloc_vnic_mem(bp);
	if (rc)
		goto alloc_mem_err;

	rc = bnxt_alloc_vnic_attributes(bp);
	if (rc)
		goto alloc_mem_err;

	rc = bnxt_alloc_filter_mem(bp);
	if (rc)
		goto alloc_mem_err;

	rc = bnxt_alloc_async_cp_ring(bp);
	if (rc)
		goto alloc_mem_err;

	rc = bnxt_alloc_rxtx_nq_ring(bp);
	if (rc)
		goto alloc_mem_err;

	if (BNXT_FLOW_XSTATS_EN(bp)) {
		rc = bnxt_alloc_flow_stats_info(bp);
		if (rc)
			goto alloc_mem_err;
	}

	return 0;

alloc_mem_err:
	bnxt_free_mem(bp, reconfig);
	return rc;
}

static int bnxt_setup_one_vnic(struct bnxt *bp, uint16_t vnic_id)
{
	struct rte_eth_conf *dev_conf = &bp->eth_dev->data->dev_conf;
	struct bnxt_vnic_info *vnic = &bp->vnic_info[vnic_id];
	uint64_t rx_offloads = dev_conf->rxmode.offloads;
	struct bnxt_rx_queue *rxq;
	unsigned int j;
	int rc;

	rc = bnxt_vnic_grp_alloc(bp, vnic);
	if (rc)
		goto err_out;

	PMD_DRV_LOG(DEBUG, "vnic[%d] = %p vnic->fw_grp_ids = %p\n",
		    vnic_id, vnic, vnic->fw_grp_ids);

	rc = bnxt_hwrm_vnic_alloc(bp, vnic);
	if (rc)
		goto err_out;

	/* Alloc RSS context only if RSS mode is enabled */
	if (dev_conf->rxmode.mq_mode & ETH_MQ_RX_RSS) {
		int j, nr_ctxs = bnxt_rss_ctxts(bp);

		rc = 0;
		for (j = 0; j < nr_ctxs; j++) {
			rc = bnxt_hwrm_vnic_ctx_alloc(bp, vnic, j);
			if (rc)
				break;
		}
		if (rc) {
			PMD_DRV_LOG(ERR,
				    "HWRM vnic %d ctx %d alloc failure rc: %x\n",
				    vnic_id, j, rc);
			goto err_out;
		}
		vnic->num_lb_ctxts = nr_ctxs;
	}

	/*
	 * Firmware sets pf pair in default vnic cfg. If the VLAN strip
	 * setting is not available at this time, it will not be
	 * configured correctly in the CFA.
	 */
	if (rx_offloads & DEV_RX_OFFLOAD_VLAN_STRIP)
		vnic->vlan_strip = true;
	else
		vnic->vlan_strip = false;

	rc = bnxt_hwrm_vnic_cfg(bp, vnic);
	if (rc)
		goto err_out;

	rc = bnxt_set_hwrm_vnic_filters(bp, vnic);
	if (rc)
		goto err_out;

	for (j = 0; j < bp->rx_num_qs_per_vnic; j++) {
		rxq = bp->eth_dev->data->rx_queues[j];

		PMD_DRV_LOG(DEBUG,
			    "rxq[%d]->vnic=%p vnic->fw_grp_ids=%p\n",
			    j, rxq->vnic, rxq->vnic->fw_grp_ids);

		if (BNXT_HAS_RING_GRPS(bp) && rxq->rx_deferred_start)
			rxq->vnic->fw_grp_ids[j] = ILWALID_HW_RING_ID;
		else
			vnic->rx_queue_cnt++;
	}

	PMD_DRV_LOG(DEBUG, "vnic->rx_queue_cnt = %d\n", vnic->rx_queue_cnt);

	rc = bnxt_vnic_rss_configure(bp, vnic);
	if (rc)
		goto err_out;

	bnxt_hwrm_vnic_plcmode_cfg(bp, vnic);

	if (rx_offloads & DEV_RX_OFFLOAD_TCP_LRO)
		bnxt_hwrm_vnic_tpa_cfg(bp, vnic, 1);
	else
		bnxt_hwrm_vnic_tpa_cfg(bp, vnic, 0);

	return 0;
err_out:
	PMD_DRV_LOG(ERR, "HWRM vnic %d cfg failure rc: %x\n",
		    vnic_id, rc);
	return rc;
}

static int bnxt_register_fc_ctx_mem(struct bnxt *bp)
{
	int rc = 0;

	rc = bnxt_hwrm_ctx_rgtr(bp, bp->flow_stat->rx_fc_in_tbl.dma,
				&bp->flow_stat->rx_fc_in_tbl.ctx_id);
	if (rc)
		return rc;

	PMD_DRV_LOG(DEBUG,
		    "rx_fc_in_tbl.va = %p rx_fc_in_tbl.dma = %p"
		    " rx_fc_in_tbl.ctx_id = %d\n",
		    bp->flow_stat->rx_fc_in_tbl.va,
		    (void *)((uintptr_t)bp->flow_stat->rx_fc_in_tbl.dma),
		    bp->flow_stat->rx_fc_in_tbl.ctx_id);

	rc = bnxt_hwrm_ctx_rgtr(bp, bp->flow_stat->rx_fc_out_tbl.dma,
				&bp->flow_stat->rx_fc_out_tbl.ctx_id);
	if (rc)
		return rc;

	PMD_DRV_LOG(DEBUG,
		    "rx_fc_out_tbl.va = %p rx_fc_out_tbl.dma = %p"
		    " rx_fc_out_tbl.ctx_id = %d\n",
		    bp->flow_stat->rx_fc_out_tbl.va,
		    (void *)((uintptr_t)bp->flow_stat->rx_fc_out_tbl.dma),
		    bp->flow_stat->rx_fc_out_tbl.ctx_id);

	rc = bnxt_hwrm_ctx_rgtr(bp, bp->flow_stat->tx_fc_in_tbl.dma,
				&bp->flow_stat->tx_fc_in_tbl.ctx_id);
	if (rc)
		return rc;

	PMD_DRV_LOG(DEBUG,
		    "tx_fc_in_tbl.va = %p tx_fc_in_tbl.dma = %p"
		    " tx_fc_in_tbl.ctx_id = %d\n",
		    bp->flow_stat->tx_fc_in_tbl.va,
		    (void *)((uintptr_t)bp->flow_stat->tx_fc_in_tbl.dma),
		    bp->flow_stat->tx_fc_in_tbl.ctx_id);

	rc = bnxt_hwrm_ctx_rgtr(bp, bp->flow_stat->tx_fc_out_tbl.dma,
				&bp->flow_stat->tx_fc_out_tbl.ctx_id);
	if (rc)
		return rc;

	PMD_DRV_LOG(DEBUG,
		    "tx_fc_out_tbl.va = %p tx_fc_out_tbl.dma = %p"
		    " tx_fc_out_tbl.ctx_id = %d\n",
		    bp->flow_stat->tx_fc_out_tbl.va,
		    (void *)((uintptr_t)bp->flow_stat->tx_fc_out_tbl.dma),
		    bp->flow_stat->tx_fc_out_tbl.ctx_id);

	memset(bp->flow_stat->rx_fc_out_tbl.va,
	       0,
	       bp->flow_stat->rx_fc_out_tbl.size);
	rc = bnxt_hwrm_cfa_counter_cfg(bp, BNXT_DIR_RX,
				       CFA_COUNTER_CFG_IN_COUNTER_TYPE_FC,
				       bp->flow_stat->rx_fc_out_tbl.ctx_id,
				       bp->flow_stat->max_fc,
				       true);
	if (rc)
		return rc;

	memset(bp->flow_stat->tx_fc_out_tbl.va,
	       0,
	       bp->flow_stat->tx_fc_out_tbl.size);
	rc = bnxt_hwrm_cfa_counter_cfg(bp, BNXT_DIR_TX,
				       CFA_COUNTER_CFG_IN_COUNTER_TYPE_FC,
				       bp->flow_stat->tx_fc_out_tbl.ctx_id,
				       bp->flow_stat->max_fc,
				       true);

	return rc;
}

static int bnxt_alloc_ctx_mem_buf(char *type, size_t size,
				  struct bnxt_ctx_mem_buf_info *ctx)
{
	if (!ctx)
		return -EILWAL;

	ctx->va = rte_zmalloc(type, size, 0);
	if (ctx->va == NULL)
		return -ENOMEM;
	rte_mem_lock_page(ctx->va);
	ctx->size = size;
	ctx->dma = rte_mem_virt2iova(ctx->va);
	if (ctx->dma == RTE_BAD_IOVA)
		return -ENOMEM;

	return 0;
}

static int bnxt_init_fc_ctx_mem(struct bnxt *bp)
{
	struct rte_pci_device *pdev = bp->pdev;
	char type[RTE_MEMZONE_NAMESIZE];
	uint16_t max_fc;
	int rc = 0;

	max_fc = bp->flow_stat->max_fc;

	sprintf(type, "bnxt_rx_fc_in_" PCI_PRI_FMT, pdev->addr.domain,
		pdev->addr.bus, pdev->addr.devid, pdev->addr.function);
	/* 4 bytes for each counter-id */
	rc = bnxt_alloc_ctx_mem_buf(type,
				    max_fc * 4,
				    &bp->flow_stat->rx_fc_in_tbl);
	if (rc)
		return rc;

	sprintf(type, "bnxt_rx_fc_out_" PCI_PRI_FMT, pdev->addr.domain,
		pdev->addr.bus, pdev->addr.devid, pdev->addr.function);
	/* 16 bytes for each counter - 8 bytes pkt_count, 8 bytes byte_count */
	rc = bnxt_alloc_ctx_mem_buf(type,
				    max_fc * 16,
				    &bp->flow_stat->rx_fc_out_tbl);
	if (rc)
		return rc;

	sprintf(type, "bnxt_tx_fc_in_" PCI_PRI_FMT, pdev->addr.domain,
		pdev->addr.bus, pdev->addr.devid, pdev->addr.function);
	/* 4 bytes for each counter-id */
	rc = bnxt_alloc_ctx_mem_buf(type,
				    max_fc * 4,
				    &bp->flow_stat->tx_fc_in_tbl);
	if (rc)
		return rc;

	sprintf(type, "bnxt_tx_fc_out_" PCI_PRI_FMT, pdev->addr.domain,
		pdev->addr.bus, pdev->addr.devid, pdev->addr.function);
	/* 16 bytes for each counter - 8 bytes pkt_count, 8 bytes byte_count */
	rc = bnxt_alloc_ctx_mem_buf(type,
				    max_fc * 16,
				    &bp->flow_stat->tx_fc_out_tbl);
	if (rc)
		return rc;

	rc = bnxt_register_fc_ctx_mem(bp);

	return rc;
}

static int bnxt_init_ctx_mem(struct bnxt *bp)
{
	int rc = 0;

	if (!(bp->fw_cap & BNXT_FW_CAP_ADV_FLOW_COUNTERS) ||
	    !(BNXT_PF(bp) || BNXT_VF_IS_TRUSTED(bp)) ||
	    !BNXT_FLOW_XSTATS_EN(bp))
		return 0;

	rc = bnxt_hwrm_cfa_counter_qcaps(bp, &bp->flow_stat->max_fc);
	if (rc)
		return rc;

	rc = bnxt_init_fc_ctx_mem(bp);

	return rc;
}

static int bnxt_update_phy_setting(struct bnxt *bp)
{
	struct rte_eth_link new;
	int rc;

	rc = bnxt_get_hwrm_link_config(bp, &new);
	if (rc) {
		PMD_DRV_LOG(ERR, "Failed to get link settings\n");
		return rc;
	}

	/*
	 * On BCM957508-N2100 adapters, FW will not allow any user other
	 * than BMC to shutdown the port. bnxt_get_hwrm_link_config() call
	 * always returns link up. Force phy update always in that case.
	 */
	if (!new.link_status || IS_BNXT_DEV_957508_N2100(bp)) {
		rc = bnxt_set_hwrm_link_config(bp, true);
		if (rc) {
			PMD_DRV_LOG(ERR, "Failed to update PHY settings\n");
			return rc;
		}
	}

	return rc;
}

static int bnxt_init_chip(struct bnxt *bp)
{
	struct rte_pci_device *pci_dev = RTE_ETH_DEV_TO_PCI(bp->eth_dev);
	struct rte_intr_handle *intr_handle = &pci_dev->intr_handle;
	uint32_t intr_vector = 0;
	uint32_t queue_id, base = BNXT_MISC_VEC_ID;
	uint32_t vec = BNXT_MISC_VEC_ID;
	unsigned int i, j;
	int rc;

	if (bp->eth_dev->data->mtu > RTE_ETHER_MTU) {
		bp->eth_dev->data->dev_conf.rxmode.offloads |=
			DEV_RX_OFFLOAD_JUMBO_FRAME;
		bp->flags |= BNXT_FLAG_JUMBO;
	} else {
		bp->eth_dev->data->dev_conf.rxmode.offloads &=
			~DEV_RX_OFFLOAD_JUMBO_FRAME;
		bp->flags &= ~BNXT_FLAG_JUMBO;
	}

	/* THOR does not support ring groups.
	 * But we will use the array to save RSS context IDs.
	 */
	if (BNXT_CHIP_THOR(bp))
		bp->max_ring_grps = BNXT_MAX_RSS_CTXTS_THOR;

	rc = bnxt_alloc_all_hwrm_stat_ctxs(bp);
	if (rc) {
		PMD_DRV_LOG(ERR, "HWRM stat ctx alloc failure rc: %x\n", rc);
		goto err_out;
	}

	rc = bnxt_alloc_hwrm_rings(bp);
	if (rc) {
		PMD_DRV_LOG(ERR, "HWRM ring alloc failure rc: %x\n", rc);
		goto err_out;
	}

	rc = bnxt_alloc_all_hwrm_ring_grps(bp);
	if (rc) {
		PMD_DRV_LOG(ERR, "HWRM ring grp alloc failure: %x\n", rc);
		goto err_out;
	}

	if (!(bp->vnic_cap_flags & BNXT_VNIC_CAP_COS_CLASSIFY))
		goto skip_cosq_cfg;

	for (j = 0, i = 0; i < BNXT_COS_QUEUE_COUNT; i++) {
		if (bp->rx_cos_queue[i].id != 0xff) {
			struct bnxt_vnic_info *vnic = &bp->vnic_info[j++];

			if (!vnic) {
				PMD_DRV_LOG(ERR,
					    "Num pools more than FW profile\n");
				rc = -EILWAL;
				goto err_out;
			}
			vnic->cos_queue_id = bp->rx_cos_queue[i].id;
			bp->rx_cosq_cnt++;
		}
	}

skip_cosq_cfg:
	rc = bnxt_mq_rx_configure(bp);
	if (rc) {
		PMD_DRV_LOG(ERR, "MQ mode configure failure rc: %x\n", rc);
		goto err_out;
	}

	/* VNIC configuration */
	for (i = 0; i < bp->nr_vnics; i++) {
		rc = bnxt_setup_one_vnic(bp, i);
		if (rc)
			goto err_out;
	}

	rc = bnxt_hwrm_cfa_l2_set_rx_mask(bp, &bp->vnic_info[0], 0, NULL);
	if (rc) {
		PMD_DRV_LOG(ERR,
			"HWRM cfa l2 rx mask failure rc: %x\n", rc);
		goto err_out;
	}

	/* check and configure queue intr-vector mapping */
	if ((rte_intr_cap_multiple(intr_handle) ||
	     !RTE_ETH_DEV_SRIOV(bp->eth_dev).active) &&
	    bp->eth_dev->data->dev_conf.intr_conf.rxq != 0) {
		intr_vector = bp->eth_dev->data->nb_rx_queues;
		PMD_DRV_LOG(DEBUG, "intr_vector = %d\n", intr_vector);
		if (intr_vector > bp->rx_cp_nr_rings) {
			PMD_DRV_LOG(ERR, "At most %d intr queues supported",
					bp->rx_cp_nr_rings);
			return -ENOTSUP;
		}
		rc = rte_intr_efd_enable(intr_handle, intr_vector);
		if (rc)
			return rc;
	}

	if (rte_intr_dp_is_en(intr_handle) && !intr_handle->intr_vec) {
		intr_handle->intr_vec =
			rte_zmalloc("intr_vec",
				    bp->eth_dev->data->nb_rx_queues *
				    sizeof(int), 0);
		if (intr_handle->intr_vec == NULL) {
			PMD_DRV_LOG(ERR, "Failed to allocate %d rx_queues"
				" intr_vec", bp->eth_dev->data->nb_rx_queues);
			rc = -ENOMEM;
			goto err_disable;
		}
		PMD_DRV_LOG(DEBUG, "intr_handle->intr_vec = %p "
			"intr_handle->nb_efd = %d intr_handle->max_intr = %d\n",
			 intr_handle->intr_vec, intr_handle->nb_efd,
			intr_handle->max_intr);
		for (queue_id = 0; queue_id < bp->eth_dev->data->nb_rx_queues;
		     queue_id++) {
			intr_handle->intr_vec[queue_id] =
							vec + BNXT_RX_VEC_START;
			if (vec < base + intr_handle->nb_efd - 1)
				vec++;
		}
	}

	/* enable uio/vfio intr/eventfd mapping */
	rc = rte_intr_enable(intr_handle);
#ifndef RTE_EXEC_ELW_FREEBSD
	/* In FreeBSD OS, nic_uio driver does not support interrupts */
	if (rc)
		goto err_free;
#endif

	rc = bnxt_update_phy_setting(bp);
	if (rc)
		goto err_free;

	bp->mark_table = rte_zmalloc("bnxt_mark_table", BNXT_MARK_TABLE_SZ, 0);
	if (!bp->mark_table)
		PMD_DRV_LOG(ERR, "Allocation of mark table failed\n");

	return 0;

err_free:
	rte_free(intr_handle->intr_vec);
err_disable:
	rte_intr_efd_disable(intr_handle);
err_out:
	/* Some of the error status returned by FW may not be from errno.h */
	if (rc > 0)
		rc = -EIO;

	return rc;
}

static int bnxt_shutdown_nic(struct bnxt *bp)
{
	bnxt_free_all_hwrm_resources(bp);
	bnxt_free_all_filters(bp);
	bnxt_free_all_vnics(bp);
	return 0;
}

/*
 * Device configuration and status function
 */

uint32_t bnxt_get_speed_capabilities(struct bnxt *bp)
{
	uint32_t link_speed = bp->link_info->support_speeds;
	uint32_t speed_capa = 0;

	/* If PAM4 is configured, use PAM4 supported speed */
	if (link_speed == 0 && bp->link_info->support_pam4_speeds > 0)
		link_speed = bp->link_info->support_pam4_speeds;

	if (link_speed & HWRM_PORT_PHY_QCFG_OUTPUT_LINK_SPEED_100MB)
		speed_capa |= ETH_LINK_SPEED_100M;
	if (link_speed & HWRM_PORT_PHY_QCFG_OUTPUT_SUPPORT_SPEEDS_100MBHD)
		speed_capa |= ETH_LINK_SPEED_100M_HD;
	if (link_speed & HWRM_PORT_PHY_QCFG_OUTPUT_SUPPORT_SPEEDS_1GB)
		speed_capa |= ETH_LINK_SPEED_1G;
	if (link_speed & HWRM_PORT_PHY_QCFG_OUTPUT_SUPPORT_SPEEDS_2_5GB)
		speed_capa |= ETH_LINK_SPEED_2_5G;
	if (link_speed & HWRM_PORT_PHY_QCFG_OUTPUT_SUPPORT_SPEEDS_10GB)
		speed_capa |= ETH_LINK_SPEED_10G;
	if (link_speed & HWRM_PORT_PHY_QCFG_OUTPUT_SUPPORT_SPEEDS_20GB)
		speed_capa |= ETH_LINK_SPEED_20G;
	if (link_speed & HWRM_PORT_PHY_QCFG_OUTPUT_SUPPORT_SPEEDS_25GB)
		speed_capa |= ETH_LINK_SPEED_25G;
	if (link_speed & HWRM_PORT_PHY_QCFG_OUTPUT_SUPPORT_SPEEDS_40GB)
		speed_capa |= ETH_LINK_SPEED_40G;
	if (link_speed & HWRM_PORT_PHY_QCFG_OUTPUT_SUPPORT_SPEEDS_50GB)
		speed_capa |= ETH_LINK_SPEED_50G;
	if (link_speed & HWRM_PORT_PHY_QCFG_OUTPUT_SUPPORT_SPEEDS_100GB)
		speed_capa |= ETH_LINK_SPEED_100G;
	if (link_speed & HWRM_PORT_PHY_QCFG_OUTPUT_SUPPORT_PAM4_SPEEDS_50G)
		speed_capa |= ETH_LINK_SPEED_50G;
	if (link_speed & HWRM_PORT_PHY_QCFG_OUTPUT_SUPPORT_PAM4_SPEEDS_100G)
		speed_capa |= ETH_LINK_SPEED_100G;
	if (link_speed & HWRM_PORT_PHY_QCFG_OUTPUT_SUPPORT_PAM4_SPEEDS_200G)
		speed_capa |= ETH_LINK_SPEED_200G;

	if (bp->link_info->auto_mode ==
	    HWRM_PORT_PHY_QCFG_OUTPUT_AUTO_MODE_NONE)
		speed_capa |= ETH_LINK_SPEED_FIXED;
	else
		speed_capa |= ETH_LINK_SPEED_AUTONEG;

	return speed_capa;
}

static int bnxt_dev_info_get_op(struct rte_eth_dev *eth_dev,
				struct rte_eth_dev_info *dev_info)
{
	struct rte_pci_device *pdev = RTE_DEV_TO_PCI(eth_dev->device);
	struct bnxt *bp = eth_dev->data->dev_private;
	uint16_t max_vnics, i, j, vpool, vrxq;
	unsigned int max_rx_rings;
	int rc;

	rc = is_bnxt_in_error(bp);
	if (rc)
		return rc;

	/* MAC Specifics */
	dev_info->max_mac_addrs = bp->max_l2_ctx;
	dev_info->max_hash_mac_addrs = 0;

	/* PF/VF specifics */
	if (BNXT_PF(bp))
		dev_info->max_vfs = pdev->max_vfs;

	max_rx_rings = BNXT_MAX_RINGS(bp);
	/* For the sake of symmetry, max_rx_queues = max_tx_queues */
	dev_info->max_rx_queues = max_rx_rings;
	dev_info->max_tx_queues = max_rx_rings;
	dev_info->reta_size = bnxt_rss_hash_tbl_size(bp);
	dev_info->hash_key_size = 40;
	max_vnics = bp->max_vnics;

	/* MTU specifics */
	dev_info->min_mtu = RTE_ETHER_MIN_MTU;
	dev_info->max_mtu = BNXT_MAX_MTU;

	/* Fast path specifics */
	dev_info->min_rx_bufsize = 1;
	dev_info->max_rx_pktlen = BNXT_MAX_PKT_LEN;

	dev_info->rx_offload_capa = BNXT_DEV_RX_OFFLOAD_SUPPORT;
	if (bp->flags & BNXT_FLAG_PTP_SUPPORTED)
		dev_info->rx_offload_capa |= DEV_RX_OFFLOAD_TIMESTAMP;
	dev_info->tx_queue_offload_capa = DEV_TX_OFFLOAD_MBUF_FAST_FREE;
	dev_info->tx_offload_capa = BNXT_DEV_TX_OFFLOAD_SUPPORT |
				    dev_info->tx_queue_offload_capa;
	dev_info->flow_type_rss_offloads = BNXT_ETH_RSS_SUPPORT;

	dev_info->speed_capa = bnxt_get_speed_capabilities(bp);

	/* *INDENT-OFF* */
	dev_info->default_rxconf = (struct rte_eth_rxconf) {
		.rx_thresh = {
			.pthresh = 8,
			.hthresh = 8,
			.wthresh = 0,
		},
		.rx_free_thresh = 32,
		.rx_drop_en = BNXT_DEFAULT_RX_DROP_EN,
	};

	dev_info->default_txconf = (struct rte_eth_txconf) {
		.tx_thresh = {
			.pthresh = 32,
			.hthresh = 0,
			.wthresh = 0,
		},
		.tx_free_thresh = 32,
		.tx_rs_thresh = 32,
	};
	eth_dev->data->dev_conf.intr_conf.lsc = 1;

	eth_dev->data->dev_conf.intr_conf.rxq = 1;
	dev_info->rx_desc_lim.nb_min = BNXT_MIN_RING_DESC;
	dev_info->rx_desc_lim.nb_max = BNXT_MAX_RX_RING_DESC;
	dev_info->tx_desc_lim.nb_min = BNXT_MIN_RING_DESC;
	dev_info->tx_desc_lim.nb_max = BNXT_MAX_TX_RING_DESC;

	if (BNXT_PF(bp) || BNXT_VF_IS_TRUSTED(bp)) {
		dev_info->switch_info.name = eth_dev->device->name;
		dev_info->switch_info.domain_id = bp->switch_domain_id;
		dev_info->switch_info.port_id =
				BNXT_PF(bp) ? BNXT_SWITCH_PORT_ID_PF :
				    BNXT_SWITCH_PORT_ID_TRUSTED_VF;
	}

	/* *INDENT-ON* */

	/*
	 * TODO: default_rxconf, default_txconf, rx_desc_lim, and tx_desc_lim
	 *       need further ilwestigation.
	 */

	/* VMDq resources */
	vpool = 64; /* ETH_64_POOLS */
	vrxq = 128; /* ETH_VMDQ_DCB_NUM_QUEUES */
	for (i = 0; i < 4; vpool >>= 1, i++) {
		if (max_vnics > vpool) {
			for (j = 0; j < 5; vrxq >>= 1, j++) {
				if (dev_info->max_rx_queues > vrxq) {
					if (vpool > vrxq)
						vpool = vrxq;
					goto found;
				}
			}
			/* Not enough resources to support VMDq */
			break;
		}
	}
	/* Not enough resources to support VMDq */
	vpool = 0;
	vrxq = 0;
found:
	dev_info->max_vmdq_pools = vpool;
	dev_info->vmdq_queue_num = vrxq;

	dev_info->vmdq_pool_base = 0;
	dev_info->vmdq_queue_base = 0;

	return 0;
}

/* Configure the device based on the configuration provided */
static int bnxt_dev_configure_op(struct rte_eth_dev *eth_dev)
{
	struct bnxt *bp = eth_dev->data->dev_private;
	uint64_t rx_offloads = eth_dev->data->dev_conf.rxmode.offloads;
	int rc;

	bp->rx_queues = (void *)eth_dev->data->rx_queues;
	bp->tx_queues = (void *)eth_dev->data->tx_queues;
	bp->tx_nr_rings = eth_dev->data->nb_tx_queues;
	bp->rx_nr_rings = eth_dev->data->nb_rx_queues;

	rc = is_bnxt_in_error(bp);
	if (rc)
		return rc;

	if (BNXT_VF(bp) && (bp->flags & BNXT_FLAG_NEW_RM)) {
		rc = bnxt_hwrm_check_vf_rings(bp);
		if (rc) {
			PMD_DRV_LOG(ERR, "HWRM insufficient resources\n");
			return -ENOSPC;
		}

		/* If a resource has already been allocated - in this case
		 * it is the async completion ring, free it. Reallocate it after
		 * resource reservation. This will ensure the resource counts
		 * are callwlated correctly.
		 */

		pthread_mutex_lock(&bp->def_cp_lock);

		if (!BNXT_HAS_NQ(bp) && bp->async_cp_ring) {
			bnxt_disable_int(bp);
			bnxt_free_cp_ring(bp, bp->async_cp_ring);
		}

		rc = bnxt_hwrm_func_reserve_vf_resc(bp, false);
		if (rc) {
			PMD_DRV_LOG(ERR, "HWRM resource alloc fail:%x\n", rc);
			pthread_mutex_unlock(&bp->def_cp_lock);
			return -ENOSPC;
		}

		if (!BNXT_HAS_NQ(bp) && bp->async_cp_ring) {
			rc = bnxt_alloc_async_cp_ring(bp);
			if (rc) {
				pthread_mutex_unlock(&bp->def_cp_lock);
				return rc;
			}
			bnxt_enable_int(bp);
		}

		pthread_mutex_unlock(&bp->def_cp_lock);
	} else {
		/* legacy driver needs to get updated values */
		rc = bnxt_hwrm_func_qcaps(bp);
		if (rc) {
			PMD_DRV_LOG(ERR, "hwrm func qcaps fail:%d\n", rc);
			return rc;
		}
	}

	/* Inherit new configurations */
	if (eth_dev->data->nb_rx_queues > bp->max_rx_rings ||
	    eth_dev->data->nb_tx_queues > bp->max_tx_rings ||
	    eth_dev->data->nb_rx_queues + eth_dev->data->nb_tx_queues
		+ BNXT_NUM_ASYNC_CPR(bp) > bp->max_cp_rings ||
	    eth_dev->data->nb_rx_queues + eth_dev->data->nb_tx_queues >
	    bp->max_stat_ctx)
		goto resource_error;

	if (BNXT_HAS_RING_GRPS(bp) &&
	    (uint32_t)(eth_dev->data->nb_rx_queues) > bp->max_ring_grps)
		goto resource_error;

	if (!(eth_dev->data->dev_conf.rxmode.mq_mode & ETH_MQ_RX_RSS) &&
	    bp->max_vnics < eth_dev->data->nb_rx_queues)
		goto resource_error;

	bp->rx_cp_nr_rings = bp->rx_nr_rings;
	bp->tx_cp_nr_rings = bp->tx_nr_rings;

	if (eth_dev->data->dev_conf.rxmode.mq_mode & ETH_MQ_RX_RSS_FLAG)
		rx_offloads |= DEV_RX_OFFLOAD_RSS_HASH;
	eth_dev->data->dev_conf.rxmode.offloads = rx_offloads;

	if (rx_offloads & DEV_RX_OFFLOAD_JUMBO_FRAME) {
		eth_dev->data->mtu =
			eth_dev->data->dev_conf.rxmode.max_rx_pkt_len -
			RTE_ETHER_HDR_LEN - RTE_ETHER_CRC_LEN - VLAN_TAG_SIZE *
			BNXT_NUM_VLANS;
		bnxt_mtu_set_op(eth_dev, eth_dev->data->mtu);
	}
	return 0;

resource_error:
	PMD_DRV_LOG(ERR,
		    "Insufficient resources to support requested config\n");
	PMD_DRV_LOG(ERR,
		    "Num Queues Requested: Tx %d, Rx %d\n",
		    eth_dev->data->nb_tx_queues,
		    eth_dev->data->nb_rx_queues);
	PMD_DRV_LOG(ERR,
		    "MAX: TxQ %d, RxQ %d, CQ %d Stat %d, Grp %d, Vnic %d\n",
		    bp->max_tx_rings, bp->max_rx_rings, bp->max_cp_rings,
		    bp->max_stat_ctx, bp->max_ring_grps, bp->max_vnics);
	return -ENOSPC;
}

void bnxt_print_link_info(struct rte_eth_dev *eth_dev)
{
	struct rte_eth_link *link = &eth_dev->data->dev_link;

	if (link->link_status)
		PMD_DRV_LOG(INFO, "Port %d Link Up - speed %u Mbps - %s\n",
			eth_dev->data->port_id,
			(uint32_t)link->link_speed,
			(link->link_duplex == ETH_LINK_FULL_DUPLEX) ?
			("full-duplex") : ("half-duplex\n"));
	else
		PMD_DRV_LOG(INFO, "Port %d Link Down\n",
			eth_dev->data->port_id);
}

/*
 * Determine whether the current configuration requires support for scattered
 * receive; return 1 if scattered receive is required and 0 if not.
 */
static int bnxt_scattered_rx(struct rte_eth_dev *eth_dev)
{
	uint16_t buf_size;
	int i;

	if (eth_dev->data->dev_conf.rxmode.offloads & DEV_RX_OFFLOAD_SCATTER)
		return 1;

	for (i = 0; i < eth_dev->data->nb_rx_queues; i++) {
		struct bnxt_rx_queue *rxq = eth_dev->data->rx_queues[i];

		buf_size = (uint16_t)(rte_pktmbuf_data_room_size(rxq->mb_pool) -
				      RTE_PKTMBUF_HEADROOM);
		if (eth_dev->data->dev_conf.rxmode.max_rx_pkt_len > buf_size)
			return 1;
	}
	return 0;
}

static eth_rx_burst_t
bnxt_receive_function(struct rte_eth_dev *eth_dev)
{
	struct bnxt *bp = eth_dev->data->dev_private;

#if defined(RTE_ARCH_X86) || defined(RTE_ARCH_ARM64)
#ifndef RTE_LIBRTE_IEEE1588
	/*
	 * Vector mode receive can be enabled only if scatter rx is not
	 * in use and rx offloads are limited to VLAN stripping and
	 * CRC stripping.
	 */
	if (!eth_dev->data->scattered_rx &&
	    !(eth_dev->data->dev_conf.rxmode.offloads &
	      ~(DEV_RX_OFFLOAD_VLAN_STRIP |
		DEV_RX_OFFLOAD_KEEP_CRC |
		DEV_RX_OFFLOAD_JUMBO_FRAME |
		DEV_RX_OFFLOAD_IPV4_CKSUM |
		DEV_RX_OFFLOAD_UDP_CKSUM |
		DEV_RX_OFFLOAD_TCP_CKSUM |
		DEV_RX_OFFLOAD_OUTER_IPV4_CKSUM |
		DEV_RX_OFFLOAD_RSS_HASH |
		DEV_RX_OFFLOAD_VLAN_FILTER)) &&
	    !BNXT_TRUFLOW_EN(bp) && BNXT_NUM_ASYNC_CPR(bp) &&
	    rte_vect_get_max_simd_bitwidth() >= RTE_VECT_SIMD_128) {
		PMD_DRV_LOG(INFO, "Using vector mode receive for port %d\n",
			    eth_dev->data->port_id);
		bp->flags |= BNXT_FLAG_RX_VECTOR_PKT_MODE;
		return bnxt_recv_pkts_vec;
	}
	PMD_DRV_LOG(INFO, "Vector mode receive disabled for port %d\n",
		    eth_dev->data->port_id);
	PMD_DRV_LOG(INFO,
		    "Port %d scatter: %d rx offload: %" PRIX64 "\n",
		    eth_dev->data->port_id,
		    eth_dev->data->scattered_rx,
		    eth_dev->data->dev_conf.rxmode.offloads);
#endif
#endif
	bp->flags &= ~BNXT_FLAG_RX_VECTOR_PKT_MODE;
	return bnxt_recv_pkts;
}

static eth_tx_burst_t
bnxt_transmit_function(__rte_unused struct rte_eth_dev *eth_dev)
{
#if defined(RTE_ARCH_X86) || defined(RTE_ARCH_ARM64)
#ifndef RTE_LIBRTE_IEEE1588
	uint64_t offloads = eth_dev->data->dev_conf.txmode.offloads;
	struct bnxt *bp = eth_dev->data->dev_private;

	/*
	 * Vector mode transmit can be enabled only if not using scatter rx
	 * or tx offloads.
	 */
	if (!eth_dev->data->scattered_rx &&
	    !(offloads & ~DEV_TX_OFFLOAD_MBUF_FAST_FREE) &&
	    !BNXT_TRUFLOW_EN(bp) &&
	    rte_vect_get_max_simd_bitwidth() >= RTE_VECT_SIMD_128) {
		PMD_DRV_LOG(INFO, "Using vector mode transmit for port %d\n",
			    eth_dev->data->port_id);
		return bnxt_xmit_pkts_vec;
	}
	PMD_DRV_LOG(INFO, "Vector mode transmit disabled for port %d\n",
		    eth_dev->data->port_id);
	PMD_DRV_LOG(INFO,
		    "Port %d scatter: %d tx offload: %" PRIX64 "\n",
		    eth_dev->data->port_id,
		    eth_dev->data->scattered_rx,
		    offloads);
#endif
#endif
	return bnxt_xmit_pkts;
}

static int bnxt_handle_if_change_status(struct bnxt *bp)
{
	int rc;

	/* Since fw has undergone a reset and lost all contexts,
	 * set fatal flag to not issue hwrm during cleanup
	 */
	bp->flags |= BNXT_FLAG_FATAL_ERROR;
	bnxt_uninit_resources(bp, true);

	/* clear fatal flag so that re-init happens */
	bp->flags &= ~BNXT_FLAG_FATAL_ERROR;
	rc = bnxt_init_resources(bp, true);

	bp->flags &= ~BNXT_FLAG_IF_CHANGE_HOT_FW_RESET_DONE;

	return rc;
}

static int bnxt_dev_start_op(struct rte_eth_dev *eth_dev)
{
	struct bnxt *bp = eth_dev->data->dev_private;
	uint64_t rx_offloads = eth_dev->data->dev_conf.rxmode.offloads;
	int vlan_mask = 0;
	int rc, retry_cnt = BNXT_IF_CHANGE_RETRY_COUNT;

	if (!eth_dev->data->nb_tx_queues || !eth_dev->data->nb_rx_queues) {
		PMD_DRV_LOG(ERR, "Queues are not configured yet!\n");
		return -EILWAL;
	}

	if (bp->rx_cp_nr_rings > RTE_ETHDEV_QUEUE_STAT_CNTRS) {
		PMD_DRV_LOG(ERR,
			"RxQ cnt %d > RTE_ETHDEV_QUEUE_STAT_CNTRS %d\n",
			bp->rx_cp_nr_rings, RTE_ETHDEV_QUEUE_STAT_CNTRS);
	}

	do {
		rc = bnxt_hwrm_if_change(bp, true);
		if (rc == 0 || rc != -EAGAIN)
			break;

		rte_delay_ms(BNXT_IF_CHANGE_RETRY_INTERVAL);
	} while (retry_cnt--);

	if (rc)
		return rc;

	if (bp->flags & BNXT_FLAG_IF_CHANGE_HOT_FW_RESET_DONE) {
		rc = bnxt_handle_if_change_status(bp);
		if (rc)
			return rc;
	}

	bnxt_enable_int(bp);

	rc = bnxt_init_chip(bp);
	if (rc)
		goto error;

	eth_dev->data->scattered_rx = bnxt_scattered_rx(eth_dev);
	eth_dev->data->dev_started = 1;

	bnxt_link_update_op(eth_dev, 1);

	if (rx_offloads & DEV_RX_OFFLOAD_VLAN_FILTER)
		vlan_mask |= ETH_VLAN_FILTER_MASK;
	if (rx_offloads & DEV_RX_OFFLOAD_VLAN_STRIP)
		vlan_mask |= ETH_VLAN_STRIP_MASK;
	rc = bnxt_vlan_offload_set_op(eth_dev, vlan_mask);
	if (rc)
		goto error;

	/* Initialize bnxt ULP port details */
	rc = bnxt_ulp_port_init(bp);
	if (rc)
		goto error;

	eth_dev->rx_pkt_burst = bnxt_receive_function(eth_dev);
	eth_dev->tx_pkt_burst = bnxt_transmit_function(eth_dev);

	bnxt_schedule_fw_health_check(bp);

	return 0;

error:
	bnxt_shutdown_nic(bp);
	bnxt_free_tx_mbufs(bp);
	bnxt_free_rx_mbufs(bp);
	bnxt_hwrm_if_change(bp, false);
	eth_dev->data->dev_started = 0;
	return rc;
}

static int bnxt_dev_set_link_up_op(struct rte_eth_dev *eth_dev)
{
	struct bnxt *bp = eth_dev->data->dev_private;
	int rc = 0;

	if (!bp->link_info->link_up)
		rc = bnxt_set_hwrm_link_config(bp, true);
	if (!rc)
		eth_dev->data->dev_link.link_status = 1;

	bnxt_print_link_info(eth_dev);
	return rc;
}

static int bnxt_dev_set_link_down_op(struct rte_eth_dev *eth_dev)
{
	struct bnxt *bp = eth_dev->data->dev_private;

	eth_dev->data->dev_link.link_status = 0;
	bnxt_set_hwrm_link_config(bp, false);
	bp->link_info->link_up = 0;

	return 0;
}

static void bnxt_free_switch_domain(struct bnxt *bp)
{
	int rc = 0;

	if (bp->switch_domain_id) {
		rc = rte_eth_switch_domain_free(bp->switch_domain_id);
		if (rc)
			PMD_DRV_LOG(ERR, "free switch domain:%d fail: %d\n",
				    bp->switch_domain_id, rc);
	}
}

/* Unload the driver, release resources */
static int bnxt_dev_stop_op(struct rte_eth_dev *eth_dev)
{
	struct bnxt *bp = eth_dev->data->dev_private;
	struct rte_pci_device *pci_dev = RTE_ETH_DEV_TO_PCI(eth_dev);
	struct rte_intr_handle *intr_handle = &pci_dev->intr_handle;
	struct rte_eth_link link;
	int ret;

	eth_dev->data->dev_started = 0;
	eth_dev->data->scattered_rx = 0;

	/* Prevent crashes when queues are still in use */
	eth_dev->rx_pkt_burst = &bnxt_dummy_recv_pkts;
	eth_dev->tx_pkt_burst = &bnxt_dummy_xmit_pkts;

	bnxt_disable_int(bp);

	/* disable uio/vfio intr/eventfd mapping */
	rte_intr_disable(intr_handle);

	/* Stop the child representors for this device */
	ret = bnxt_rep_stop_all(bp);
	if (ret != 0)
		return ret;

	/* delete the bnxt ULP port details */
	bnxt_ulp_port_deinit(bp);

	bnxt_cancel_fw_health_check(bp);

	/* Do not bring link down during reset recovery */
	if (!is_bnxt_in_error(bp)) {
		bnxt_dev_set_link_down_op(eth_dev);
		/* Wait for link to be reset */
		if (BNXT_SINGLE_PF(bp))
			rte_delay_ms(500);
		/* clear the recorded link status */
		memset(&link, 0, sizeof(link));
		rte_eth_linkstatus_set(eth_dev, &link);
	}

	/* Clean queue intr-vector mapping */
	rte_intr_efd_disable(intr_handle);
	if (intr_handle->intr_vec != NULL) {
		rte_free(intr_handle->intr_vec);
		intr_handle->intr_vec = NULL;
	}

	bnxt_hwrm_port_clr_stats(bp);
	bnxt_free_tx_mbufs(bp);
	bnxt_free_rx_mbufs(bp);
	/* Process any remaining notifications in default completion queue */
	bnxt_int_handler(eth_dev);
	bnxt_shutdown_nic(bp);
	bnxt_hwrm_if_change(bp, false);

	rte_free(bp->mark_table);
	bp->mark_table = NULL;

	bp->flags &= ~BNXT_FLAG_RX_VECTOR_PKT_MODE;
	bp->rx_cosq_cnt = 0;
	/* All filters are deleted on a port stop. */
	if (BNXT_FLOW_XSTATS_EN(bp))
		bp->flow_stat->flow_count = 0;

	return 0;
}

static int bnxt_dev_close_op(struct rte_eth_dev *eth_dev)
{
	struct bnxt *bp = eth_dev->data->dev_private;
	int ret = 0;

	if (rte_eal_process_type() != RTE_PROC_PRIMARY)
		return 0;

	/* cancel the recovery handler before remove dev */
	rte_eal_alarm_cancel(bnxt_dev_reset_and_resume, (void *)bp);
	rte_eal_alarm_cancel(bnxt_dev_recover, (void *)bp);
	bnxt_cancel_fc_thread(bp);

	if (eth_dev->data->dev_started)
		ret = bnxt_dev_stop_op(eth_dev);

	bnxt_free_switch_domain(bp);

	bnxt_uninit_resources(bp, false);

	bnxt_free_leds_info(bp);
	bnxt_free_cos_queues(bp);
	bnxt_free_link_info(bp);
	bnxt_free_pf_info(bp);
	bnxt_free_parent_info(bp);

	rte_memzone_free((const struct rte_memzone *)bp->tx_mem_zone);
	bp->tx_mem_zone = NULL;
	rte_memzone_free((const struct rte_memzone *)bp->rx_mem_zone);
	bp->rx_mem_zone = NULL;

	bnxt_hwrm_free_vf_info(bp);

	rte_free(bp->grp_info);
	bp->grp_info = NULL;

	return ret;
}

static void bnxt_mac_addr_remove_op(struct rte_eth_dev *eth_dev,
				    uint32_t index)
{
	struct bnxt *bp = eth_dev->data->dev_private;
	uint64_t pool_mask = eth_dev->data->mac_pool_sel[index];
	struct bnxt_vnic_info *vnic;
	struct bnxt_filter_info *filter, *temp_filter;
	uint32_t i;

	if (is_bnxt_in_error(bp))
		return;

	/*
	 * Loop through all VNICs from the specified filter flow pools to
	 * remove the corresponding MAC addr filter
	 */
	for (i = 0; i < bp->nr_vnics; i++) {
		if (!(pool_mask & (1ULL << i)))
			continue;

		vnic = &bp->vnic_info[i];
		filter = STAILQ_FIRST(&vnic->filter);
		while (filter) {
			temp_filter = STAILQ_NEXT(filter, next);
			if (filter->mac_index == index) {
				STAILQ_REMOVE(&vnic->filter, filter,
						bnxt_filter_info, next);
				bnxt_hwrm_clear_l2_filter(bp, filter);
				bnxt_free_filter(bp, filter);
			}
			filter = temp_filter;
		}
	}
}

static int bnxt_add_mac_filter(struct bnxt *bp, struct bnxt_vnic_info *vnic,
			       struct rte_ether_addr *mac_addr, uint32_t index,
			       uint32_t pool)
{
	struct bnxt_filter_info *filter;
	int rc = 0;

	/* Attach requested MAC address to the new l2_filter */
	STAILQ_FOREACH(filter, &vnic->filter, next) {
		if (filter->mac_index == index) {
			PMD_DRV_LOG(DEBUG,
				    "MAC addr already existed for pool %d\n",
				    pool);
			return 0;
		}
	}

	filter = bnxt_alloc_filter(bp);
	if (!filter) {
		PMD_DRV_LOG(ERR, "L2 filter alloc failed\n");
		return -ENODEV;
	}

	/* bnxt_alloc_filter copies default MAC to filter->l2_addr. So,
	 * if the MAC that's been programmed now is a different one, then,
	 * copy that addr to filter->l2_addr
	 */
	if (mac_addr)
		memcpy(filter->l2_addr, mac_addr, RTE_ETHER_ADDR_LEN);
	filter->flags |= HWRM_CFA_L2_FILTER_ALLOC_INPUT_FLAGS_OUTERMOST;

	rc = bnxt_hwrm_set_l2_filter(bp, vnic->fw_vnic_id, filter);
	if (!rc) {
		filter->mac_index = index;
		if (filter->mac_index == 0)
			STAILQ_INSERT_HEAD(&vnic->filter, filter, next);
		else
			STAILQ_INSERT_TAIL(&vnic->filter, filter, next);
	} else {
		bnxt_free_filter(bp, filter);
	}

	return rc;
}

static int bnxt_mac_addr_add_op(struct rte_eth_dev *eth_dev,
				struct rte_ether_addr *mac_addr,
				uint32_t index, uint32_t pool)
{
	struct bnxt *bp = eth_dev->data->dev_private;
	struct bnxt_vnic_info *vnic = &bp->vnic_info[pool];
	int rc = 0;

	rc = is_bnxt_in_error(bp);
	if (rc)
		return rc;

	if (BNXT_VF(bp) && !BNXT_VF_IS_TRUSTED(bp)) {
		PMD_DRV_LOG(ERR, "Cannot add MAC address to a VF interface\n");
		return -ENOTSUP;
	}

	if (!vnic) {
		PMD_DRV_LOG(ERR, "VNIC not found for pool %d!\n", pool);
		return -EILWAL;
	}

	/* Filter settings will get applied when port is started */
	if (!eth_dev->data->dev_started)
		return 0;

	rc = bnxt_add_mac_filter(bp, vnic, mac_addr, index, pool);

	return rc;
}

int bnxt_link_update_op(struct rte_eth_dev *eth_dev, int wait_to_complete)
{
	int rc = 0;
	struct bnxt *bp = eth_dev->data->dev_private;
	struct rte_eth_link new;
	int cnt = wait_to_complete ? BNXT_MAX_LINK_WAIT_CNT :
			BNXT_MIN_LINK_WAIT_CNT;

	rc = is_bnxt_in_error(bp);
	if (rc)
		return rc;

	memset(&new, 0, sizeof(new));
	do {
		/* Retrieve link info from hardware */
		rc = bnxt_get_hwrm_link_config(bp, &new);
		if (rc) {
			new.link_speed = ETH_LINK_SPEED_100M;
			new.link_duplex = ETH_LINK_FULL_DUPLEX;
			PMD_DRV_LOG(ERR,
				"Failed to retrieve link rc = 0x%x!\n", rc);
			goto out;
		}

		if (!wait_to_complete || new.link_status)
			break;

		rte_delay_ms(BNXT_LINK_WAIT_INTERVAL);
	} while (cnt--);

	/* Only single function PF can bring phy down.
	 * When port is stopped, report link down for VF/MH/NPAR functions.
	 */
	if (!BNXT_SINGLE_PF(bp) && !eth_dev->data->dev_started)
		memset(&new, 0, sizeof(new));

out:
	/* Timed out or success */
	if (new.link_status != eth_dev->data->dev_link.link_status ||
	new.link_speed != eth_dev->data->dev_link.link_speed) {
		rte_eth_linkstatus_set(eth_dev, &new);

		rte_eth_dev_callback_process(eth_dev,
					     RTE_ETH_EVENT_INTR_LSC,
					     NULL);

		bnxt_print_link_info(eth_dev);
	}

	return rc;
}

static int bnxt_promislwous_enable_op(struct rte_eth_dev *eth_dev)
{
	struct bnxt *bp = eth_dev->data->dev_private;
	struct bnxt_vnic_info *vnic;
	uint32_t old_flags;
	int rc;

	rc = is_bnxt_in_error(bp);
	if (rc)
		return rc;

	/* Filter settings will get applied when port is started */
	if (!eth_dev->data->dev_started)
		return 0;

	if (bp->vnic_info == NULL)
		return 0;

	vnic = BNXT_GET_DEFAULT_VNIC(bp);

	old_flags = vnic->flags;
	vnic->flags |= BNXT_VNIC_INFO_PROMISC;
	rc = bnxt_hwrm_cfa_l2_set_rx_mask(bp, vnic, 0, NULL);
	if (rc != 0)
		vnic->flags = old_flags;

	return rc;
}

static int bnxt_promislwous_disable_op(struct rte_eth_dev *eth_dev)
{
	struct bnxt *bp = eth_dev->data->dev_private;
	struct bnxt_vnic_info *vnic;
	uint32_t old_flags;
	int rc;

	rc = is_bnxt_in_error(bp);
	if (rc)
		return rc;

	/* Filter settings will get applied when port is started */
	if (!eth_dev->data->dev_started)
		return 0;

	if (bp->vnic_info == NULL)
		return 0;

	vnic = BNXT_GET_DEFAULT_VNIC(bp);

	old_flags = vnic->flags;
	vnic->flags &= ~BNXT_VNIC_INFO_PROMISC;
	rc = bnxt_hwrm_cfa_l2_set_rx_mask(bp, vnic, 0, NULL);
	if (rc != 0)
		vnic->flags = old_flags;

	return rc;
}

static int bnxt_allmulticast_enable_op(struct rte_eth_dev *eth_dev)
{
	struct bnxt *bp = eth_dev->data->dev_private;
	struct bnxt_vnic_info *vnic;
	uint32_t old_flags;
	int rc;

	rc = is_bnxt_in_error(bp);
	if (rc)
		return rc;

	/* Filter settings will get applied when port is started */
	if (!eth_dev->data->dev_started)
		return 0;

	if (bp->vnic_info == NULL)
		return 0;

	vnic = BNXT_GET_DEFAULT_VNIC(bp);

	old_flags = vnic->flags;
	vnic->flags |= BNXT_VNIC_INFO_ALLMULTI;
	rc = bnxt_hwrm_cfa_l2_set_rx_mask(bp, vnic, 0, NULL);
	if (rc != 0)
		vnic->flags = old_flags;

	return rc;
}

static int bnxt_allmulticast_disable_op(struct rte_eth_dev *eth_dev)
{
	struct bnxt *bp = eth_dev->data->dev_private;
	struct bnxt_vnic_info *vnic;
	uint32_t old_flags;
	int rc;

	rc = is_bnxt_in_error(bp);
	if (rc)
		return rc;

	/* Filter settings will get applied when port is started */
	if (!eth_dev->data->dev_started)
		return 0;

	if (bp->vnic_info == NULL)
		return 0;

	vnic = BNXT_GET_DEFAULT_VNIC(bp);

	old_flags = vnic->flags;
	vnic->flags &= ~BNXT_VNIC_INFO_ALLMULTI;
	rc = bnxt_hwrm_cfa_l2_set_rx_mask(bp, vnic, 0, NULL);
	if (rc != 0)
		vnic->flags = old_flags;

	return rc;
}

/* Return bnxt_rx_queue pointer corresponding to a given rxq. */
static struct bnxt_rx_queue *bnxt_qid_to_rxq(struct bnxt *bp, uint16_t qid)
{
	if (qid >= bp->rx_nr_rings)
		return NULL;

	return bp->eth_dev->data->rx_queues[qid];
}

/* Return rxq corresponding to a given rss table ring/group ID. */
static uint16_t bnxt_rss_to_qid(struct bnxt *bp, uint16_t fwr)
{
	struct bnxt_rx_queue *rxq;
	unsigned int i;

	if (!BNXT_HAS_RING_GRPS(bp)) {
		for (i = 0; i < bp->rx_nr_rings; i++) {
			rxq = bp->eth_dev->data->rx_queues[i];
			if (rxq->rx_ring->rx_ring_struct->fw_ring_id == fwr)
				return rxq->index;
		}
	} else {
		for (i = 0; i < bp->rx_nr_rings; i++) {
			if (bp->grp_info[i].fw_grp_id == fwr)
				return i;
		}
	}

	return ILWALID_HW_RING_ID;
}

static int bnxt_reta_update_op(struct rte_eth_dev *eth_dev,
			    struct rte_eth_rss_reta_entry64 *reta_conf,
			    uint16_t reta_size)
{
	struct bnxt *bp = eth_dev->data->dev_private;
	struct rte_eth_conf *dev_conf = &bp->eth_dev->data->dev_conf;
	struct bnxt_vnic_info *vnic = BNXT_GET_DEFAULT_VNIC(bp);
	uint16_t tbl_size = bnxt_rss_hash_tbl_size(bp);
	uint16_t idx, sft;
	int i, rc;

	rc = is_bnxt_in_error(bp);
	if (rc)
		return rc;

	if (!vnic->rss_table)
		return -EILWAL;

	if (!(dev_conf->rxmode.mq_mode & ETH_MQ_RX_RSS_FLAG))
		return -EILWAL;

	if (reta_size != tbl_size) {
		PMD_DRV_LOG(ERR, "The configured hash table lookup size "
			"(%d) must equal the size supported by the hardware "
			"(%d)\n", reta_size, tbl_size);
		return -EILWAL;
	}

	for (i = 0; i < reta_size; i++) {
		struct bnxt_rx_queue *rxq;

		idx = i / RTE_RETA_GROUP_SIZE;
		sft = i % RTE_RETA_GROUP_SIZE;

		if (!(reta_conf[idx].mask & (1ULL << sft)))
			continue;

		rxq = bnxt_qid_to_rxq(bp, reta_conf[idx].reta[sft]);
		if (!rxq) {
			PMD_DRV_LOG(ERR, "Invalid ring in reta_conf.\n");
			return -EILWAL;
		}

		if (BNXT_CHIP_THOR(bp)) {
			vnic->rss_table[i * 2] =
				rxq->rx_ring->rx_ring_struct->fw_ring_id;
			vnic->rss_table[i * 2 + 1] =
				rxq->cp_ring->cp_ring_struct->fw_ring_id;
		} else {
			vnic->rss_table[i] =
			    vnic->fw_grp_ids[reta_conf[idx].reta[sft]];
		}
	}

	bnxt_hwrm_vnic_rss_cfg(bp, vnic);
	return 0;
}

static int bnxt_reta_query_op(struct rte_eth_dev *eth_dev,
			      struct rte_eth_rss_reta_entry64 *reta_conf,
			      uint16_t reta_size)
{
	struct bnxt *bp = eth_dev->data->dev_private;
	struct bnxt_vnic_info *vnic = BNXT_GET_DEFAULT_VNIC(bp);
	uint16_t tbl_size = bnxt_rss_hash_tbl_size(bp);
	uint16_t idx, sft, i;
	int rc;

	rc = is_bnxt_in_error(bp);
	if (rc)
		return rc;

	/* Retrieve from the default VNIC */
	if (!vnic)
		return -EILWAL;
	if (!vnic->rss_table)
		return -EILWAL;

	if (reta_size != tbl_size) {
		PMD_DRV_LOG(ERR, "The configured hash table lookup size "
			"(%d) must equal the size supported by the hardware "
			"(%d)\n", reta_size, tbl_size);
		return -EILWAL;
	}

	for (idx = 0, i = 0; i < reta_size; i++) {
		idx = i / RTE_RETA_GROUP_SIZE;
		sft = i % RTE_RETA_GROUP_SIZE;

		if (reta_conf[idx].mask & (1ULL << sft)) {
			uint16_t qid;

			if (BNXT_CHIP_THOR(bp))
				qid = bnxt_rss_to_qid(bp,
						      vnic->rss_table[i * 2]);
			else
				qid = bnxt_rss_to_qid(bp, vnic->rss_table[i]);

			if (qid == ILWALID_HW_RING_ID) {
				PMD_DRV_LOG(ERR, "Ilw. entry in rss table.\n");
				return -EILWAL;
			}
			reta_conf[idx].reta[sft] = qid;
		}
	}

	return 0;
}

static int bnxt_rss_hash_update_op(struct rte_eth_dev *eth_dev,
				   struct rte_eth_rss_conf *rss_conf)
{
	struct bnxt *bp = eth_dev->data->dev_private;
	struct rte_eth_conf *dev_conf = &bp->eth_dev->data->dev_conf;
	struct bnxt_vnic_info *vnic;
	int rc;

	rc = is_bnxt_in_error(bp);
	if (rc)
		return rc;

	/*
	 * If RSS enablement were different than dev_configure,
	 * then return -EILWAL
	 */
	if (dev_conf->rxmode.mq_mode & ETH_MQ_RX_RSS_FLAG) {
		if (!rss_conf->rss_hf)
			PMD_DRV_LOG(ERR, "Hash type NONE\n");
	} else {
		if (rss_conf->rss_hf & BNXT_ETH_RSS_SUPPORT)
			return -EILWAL;
	}

	bp->flags |= BNXT_FLAG_UPDATE_HASH;
	memcpy(&eth_dev->data->dev_conf.rx_adv_conf.rss_conf,
	       rss_conf,
	       sizeof(*rss_conf));

	/* Update the default RSS VNIC(s) */
	vnic = BNXT_GET_DEFAULT_VNIC(bp);
	vnic->hash_type = bnxt_rte_to_hwrm_hash_types(rss_conf->rss_hf);
	vnic->hash_mode =
		bnxt_rte_to_hwrm_hash_level(bp, rss_conf->rss_hf,
					    ETH_RSS_LEVEL(rss_conf->rss_hf));

	/*
	 * If hashkey is not specified, use the previously configured
	 * hashkey
	 */
	if (!rss_conf->rss_key)
		goto rss_config;

	if (rss_conf->rss_key_len != HW_HASH_KEY_SIZE) {
		PMD_DRV_LOG(ERR,
			    "Invalid hashkey length, should be 16 bytes\n");
		return -EILWAL;
	}
	memcpy(vnic->rss_hash_key, rss_conf->rss_key, rss_conf->rss_key_len);

rss_config:
	bnxt_hwrm_vnic_rss_cfg(bp, vnic);
	return 0;
}

static int bnxt_rss_hash_conf_get_op(struct rte_eth_dev *eth_dev,
				     struct rte_eth_rss_conf *rss_conf)
{
	struct bnxt *bp = eth_dev->data->dev_private;
	struct bnxt_vnic_info *vnic = BNXT_GET_DEFAULT_VNIC(bp);
	int len, rc;
	uint32_t hash_types;

	rc = is_bnxt_in_error(bp);
	if (rc)
		return rc;

	/* RSS configuration is the same for all VNICs */
	if (vnic && vnic->rss_hash_key) {
		if (rss_conf->rss_key) {
			len = rss_conf->rss_key_len <= HW_HASH_KEY_SIZE ?
			      rss_conf->rss_key_len : HW_HASH_KEY_SIZE;
			memcpy(rss_conf->rss_key, vnic->rss_hash_key, len);
		}

		hash_types = vnic->hash_type;
		rss_conf->rss_hf = 0;
		if (hash_types & HWRM_VNIC_RSS_CFG_INPUT_HASH_TYPE_IPV4) {
			rss_conf->rss_hf |= ETH_RSS_IPV4;
			hash_types &= ~HWRM_VNIC_RSS_CFG_INPUT_HASH_TYPE_IPV4;
		}
		if (hash_types & HWRM_VNIC_RSS_CFG_INPUT_HASH_TYPE_TCP_IPV4) {
			rss_conf->rss_hf |= ETH_RSS_NONFRAG_IPV4_TCP;
			hash_types &=
				~HWRM_VNIC_RSS_CFG_INPUT_HASH_TYPE_TCP_IPV4;
		}
		if (hash_types & HWRM_VNIC_RSS_CFG_INPUT_HASH_TYPE_UDP_IPV4) {
			rss_conf->rss_hf |= ETH_RSS_NONFRAG_IPV4_UDP;
			hash_types &=
				~HWRM_VNIC_RSS_CFG_INPUT_HASH_TYPE_UDP_IPV4;
		}
		if (hash_types & HWRM_VNIC_RSS_CFG_INPUT_HASH_TYPE_IPV6) {
			rss_conf->rss_hf |= ETH_RSS_IPV6;
			hash_types &= ~HWRM_VNIC_RSS_CFG_INPUT_HASH_TYPE_IPV6;
		}
		if (hash_types & HWRM_VNIC_RSS_CFG_INPUT_HASH_TYPE_TCP_IPV6) {
			rss_conf->rss_hf |= ETH_RSS_NONFRAG_IPV6_TCP;
			hash_types &=
				~HWRM_VNIC_RSS_CFG_INPUT_HASH_TYPE_TCP_IPV6;
		}
		if (hash_types & HWRM_VNIC_RSS_CFG_INPUT_HASH_TYPE_UDP_IPV6) {
			rss_conf->rss_hf |= ETH_RSS_NONFRAG_IPV6_UDP;
			hash_types &=
				~HWRM_VNIC_RSS_CFG_INPUT_HASH_TYPE_UDP_IPV6;
		}

		rss_conf->rss_hf |=
			bnxt_hwrm_to_rte_rss_level(bp, vnic->hash_mode);

		if (hash_types) {
			PMD_DRV_LOG(ERR,
				"Unknown RSS config from firmware (%08x), RSS disabled",
				vnic->hash_type);
			return -ENOTSUP;
		}
	} else {
		rss_conf->rss_hf = 0;
	}
	return 0;
}

static int bnxt_flow_ctrl_get_op(struct rte_eth_dev *dev,
			       struct rte_eth_fc_conf *fc_conf)
{
	struct bnxt *bp = dev->data->dev_private;
	struct rte_eth_link link_info;
	int rc;

	rc = is_bnxt_in_error(bp);
	if (rc)
		return rc;

	rc = bnxt_get_hwrm_link_config(bp, &link_info);
	if (rc)
		return rc;

	memset(fc_conf, 0, sizeof(*fc_conf));
	if (bp->link_info->auto_pause)
		fc_conf->autoneg = 1;
	switch (bp->link_info->pause) {
	case 0:
		fc_conf->mode = RTE_FC_NONE;
		break;
	case HWRM_PORT_PHY_QCFG_OUTPUT_PAUSE_TX:
		fc_conf->mode = RTE_FC_TX_PAUSE;
		break;
	case HWRM_PORT_PHY_QCFG_OUTPUT_PAUSE_RX:
		fc_conf->mode = RTE_FC_RX_PAUSE;
		break;
	case (HWRM_PORT_PHY_QCFG_OUTPUT_PAUSE_TX |
			HWRM_PORT_PHY_QCFG_OUTPUT_PAUSE_RX):
		fc_conf->mode = RTE_FC_FULL;
		break;
	}
	return 0;
}

static int bnxt_flow_ctrl_set_op(struct rte_eth_dev *dev,
			       struct rte_eth_fc_conf *fc_conf)
{
	struct bnxt *bp = dev->data->dev_private;
	int rc;

	rc = is_bnxt_in_error(bp);
	if (rc)
		return rc;

	if (!BNXT_SINGLE_PF(bp) || BNXT_VF(bp)) {
		PMD_DRV_LOG(ERR, "Flow Control Settings cannot be modified\n");
		return -ENOTSUP;
	}

	switch (fc_conf->mode) {
	case RTE_FC_NONE:
		bp->link_info->auto_pause = 0;
		bp->link_info->force_pause = 0;
		break;
	case RTE_FC_RX_PAUSE:
		if (fc_conf->autoneg) {
			bp->link_info->auto_pause =
					HWRM_PORT_PHY_CFG_INPUT_AUTO_PAUSE_RX;
			bp->link_info->force_pause = 0;
		} else {
			bp->link_info->auto_pause = 0;
			bp->link_info->force_pause =
					HWRM_PORT_PHY_CFG_INPUT_FORCE_PAUSE_RX;
		}
		break;
	case RTE_FC_TX_PAUSE:
		if (fc_conf->autoneg) {
			bp->link_info->auto_pause =
					HWRM_PORT_PHY_CFG_INPUT_AUTO_PAUSE_TX;
			bp->link_info->force_pause = 0;
		} else {
			bp->link_info->auto_pause = 0;
			bp->link_info->force_pause =
					HWRM_PORT_PHY_CFG_INPUT_FORCE_PAUSE_TX;
		}
		break;
	case RTE_FC_FULL:
		if (fc_conf->autoneg) {
			bp->link_info->auto_pause =
					HWRM_PORT_PHY_CFG_INPUT_AUTO_PAUSE_TX |
					HWRM_PORT_PHY_CFG_INPUT_AUTO_PAUSE_RX;
			bp->link_info->force_pause = 0;
		} else {
			bp->link_info->auto_pause = 0;
			bp->link_info->force_pause =
					HWRM_PORT_PHY_CFG_INPUT_FORCE_PAUSE_TX |
					HWRM_PORT_PHY_CFG_INPUT_FORCE_PAUSE_RX;
		}
		break;
	}
	return bnxt_set_hwrm_link_config(bp, true);
}

/* Add UDP tunneling port */
static int
bnxt_udp_tunnel_port_add_op(struct rte_eth_dev *eth_dev,
			 struct rte_eth_udp_tunnel *udp_tunnel)
{
	struct bnxt *bp = eth_dev->data->dev_private;
	uint16_t tunnel_type = 0;
	int rc = 0;

	rc = is_bnxt_in_error(bp);
	if (rc)
		return rc;

	switch (udp_tunnel->prot_type) {
	case RTE_TUNNEL_TYPE_VXLAN:
		if (bp->vxlan_port_cnt) {
			PMD_DRV_LOG(ERR, "Tunnel Port %d already programmed\n",
				udp_tunnel->udp_port);
			if (bp->vxlan_port != udp_tunnel->udp_port) {
				PMD_DRV_LOG(ERR, "Only one port allowed\n");
				return -ENOSPC;
			}
			bp->vxlan_port_cnt++;
			return 0;
		}
		tunnel_type =
			HWRM_TUNNEL_DST_PORT_ALLOC_INPUT_TUNNEL_TYPE_VXLAN;
		bp->vxlan_port_cnt++;
		break;
	case RTE_TUNNEL_TYPE_GENEVE:
		if (bp->geneve_port_cnt) {
			PMD_DRV_LOG(ERR, "Tunnel Port %d already programmed\n",
				udp_tunnel->udp_port);
			if (bp->geneve_port != udp_tunnel->udp_port) {
				PMD_DRV_LOG(ERR, "Only one port allowed\n");
				return -ENOSPC;
			}
			bp->geneve_port_cnt++;
			return 0;
		}
		tunnel_type =
			HWRM_TUNNEL_DST_PORT_ALLOC_INPUT_TUNNEL_TYPE_GENEVE;
		bp->geneve_port_cnt++;
		break;
	default:
		PMD_DRV_LOG(ERR, "Tunnel type is not supported\n");
		return -ENOTSUP;
	}
	rc = bnxt_hwrm_tunnel_dst_port_alloc(bp, udp_tunnel->udp_port,
					     tunnel_type);
	return rc;
}

static int
bnxt_udp_tunnel_port_del_op(struct rte_eth_dev *eth_dev,
			 struct rte_eth_udp_tunnel *udp_tunnel)
{
	struct bnxt *bp = eth_dev->data->dev_private;
	uint16_t tunnel_type = 0;
	uint16_t port = 0;
	int rc = 0;

	rc = is_bnxt_in_error(bp);
	if (rc)
		return rc;

	switch (udp_tunnel->prot_type) {
	case RTE_TUNNEL_TYPE_VXLAN:
		if (!bp->vxlan_port_cnt) {
			PMD_DRV_LOG(ERR, "No Tunnel port configured yet\n");
			return -EILWAL;
		}
		if (bp->vxlan_port != udp_tunnel->udp_port) {
			PMD_DRV_LOG(ERR, "Req Port: %d. Configured port: %d\n",
				udp_tunnel->udp_port, bp->vxlan_port);
			return -EILWAL;
		}
		if (--bp->vxlan_port_cnt)
			return 0;

		tunnel_type =
			HWRM_TUNNEL_DST_PORT_FREE_INPUT_TUNNEL_TYPE_VXLAN;
		port = bp->vxlan_fw_dst_port_id;
		break;
	case RTE_TUNNEL_TYPE_GENEVE:
		if (!bp->geneve_port_cnt) {
			PMD_DRV_LOG(ERR, "No Tunnel port configured yet\n");
			return -EILWAL;
		}
		if (bp->geneve_port != udp_tunnel->udp_port) {
			PMD_DRV_LOG(ERR, "Req Port: %d. Configured port: %d\n",
				udp_tunnel->udp_port, bp->geneve_port);
			return -EILWAL;
		}
		if (--bp->geneve_port_cnt)
			return 0;

		tunnel_type =
			HWRM_TUNNEL_DST_PORT_FREE_INPUT_TUNNEL_TYPE_GENEVE;
		port = bp->geneve_fw_dst_port_id;
		break;
	default:
		PMD_DRV_LOG(ERR, "Tunnel type is not supported\n");
		return -ENOTSUP;
	}

	rc = bnxt_hwrm_tunnel_dst_port_free(bp, port, tunnel_type);
	return rc;
}

static int bnxt_del_vlan_filter(struct bnxt *bp, uint16_t vlan_id)
{
	struct bnxt_filter_info *filter;
	struct bnxt_vnic_info *vnic;
	int rc = 0;
	uint32_t chk = HWRM_CFA_L2_FILTER_ALLOC_INPUT_ENABLES_L2_IVLAN;

	vnic = BNXT_GET_DEFAULT_VNIC(bp);
	filter = STAILQ_FIRST(&vnic->filter);
	while (filter) {
		/* Search for this matching MAC+VLAN filter */
		if (bnxt_vlan_filter_exists(bp, filter, chk, vlan_id)) {
			/* Delete the filter */
			rc = bnxt_hwrm_clear_l2_filter(bp, filter);
			if (rc)
				return rc;
			STAILQ_REMOVE(&vnic->filter, filter,
				      bnxt_filter_info, next);
			bnxt_free_filter(bp, filter);
			PMD_DRV_LOG(INFO,
				    "Deleted vlan filter for %d\n",
				    vlan_id);
			return 0;
		}
		filter = STAILQ_NEXT(filter, next);
	}
	return -ENOENT;
}

static int bnxt_add_vlan_filter(struct bnxt *bp, uint16_t vlan_id)
{
	struct bnxt_filter_info *filter;
	struct bnxt_vnic_info *vnic;
	int rc = 0;
	uint32_t en = HWRM_CFA_L2_FILTER_ALLOC_INPUT_ENABLES_L2_IVLAN |
		HWRM_CFA_L2_FILTER_ALLOC_INPUT_ENABLES_L2_IVLAN_MASK;
	uint32_t chk = HWRM_CFA_L2_FILTER_ALLOC_INPUT_ENABLES_L2_IVLAN;

	/* Implementation notes on the use of VNIC in this command:
	 *
	 * By default, these filters belong to default vnic for the function.
	 * Once these filters are set up, only destination VNIC can be modified.
	 * If the destination VNIC is not specified in this command,
	 * then the HWRM shall only create an l2 context id.
	 */

	vnic = BNXT_GET_DEFAULT_VNIC(bp);
	filter = STAILQ_FIRST(&vnic->filter);
	/* Check if the VLAN has already been added */
	while (filter) {
		if (bnxt_vlan_filter_exists(bp, filter, chk, vlan_id))
			return -EEXIST;

		filter = STAILQ_NEXT(filter, next);
	}

	/* No match found. Alloc a fresh filter and issue the L2_FILTER_ALLOC
	 * command to create MAC+VLAN filter with the right flags, enables set.
	 */
	filter = bnxt_alloc_filter(bp);
	if (!filter) {
		PMD_DRV_LOG(ERR,
			    "MAC/VLAN filter alloc failed\n");
		return -ENOMEM;
	}
	/* MAC + VLAN ID filter */
	/* If l2_ivlan == 0 and l2_ivlan_mask != 0, only
	 * untagged packets are received
	 *
	 * If l2_ivlan != 0 and l2_ivlan_mask != 0, untagged
	 * packets and only the programmed vlan's packets are received
	 */
	filter->l2_ivlan = vlan_id;
	filter->l2_ivlan_mask = 0x0FFF;
	filter->enables |= en;
	filter->flags |= HWRM_CFA_L2_FILTER_ALLOC_INPUT_FLAGS_OUTERMOST;

	rc = bnxt_hwrm_set_l2_filter(bp, vnic->fw_vnic_id, filter);
	if (rc) {
		/* Free the newly allocated filter as we were
		 * not able to create the filter in hardware.
		 */
		bnxt_free_filter(bp, filter);
		return rc;
	}

	filter->mac_index = 0;
	/* Add this new filter to the list */
	if (vlan_id == 0)
		STAILQ_INSERT_HEAD(&vnic->filter, filter, next);
	else
		STAILQ_INSERT_TAIL(&vnic->filter, filter, next);

	PMD_DRV_LOG(INFO,
		    "Added Vlan filter for %d\n", vlan_id);
	return rc;
}

static int bnxt_vlan_filter_set_op(struct rte_eth_dev *eth_dev,
		uint16_t vlan_id, int on)
{
	struct bnxt *bp = eth_dev->data->dev_private;
	int rc;

	rc = is_bnxt_in_error(bp);
	if (rc)
		return rc;

	if (!eth_dev->data->dev_started) {
		PMD_DRV_LOG(ERR, "port must be started before setting vlan\n");
		return -EILWAL;
	}

	/* These operations apply to ALL existing MAC/VLAN filters */
	if (on)
		return bnxt_add_vlan_filter(bp, vlan_id);
	else
		return bnxt_del_vlan_filter(bp, vlan_id);
}

static int bnxt_del_dflt_mac_filter(struct bnxt *bp,
				    struct bnxt_vnic_info *vnic)
{
	struct bnxt_filter_info *filter;
	int rc;

	filter = STAILQ_FIRST(&vnic->filter);
	while (filter) {
		if (filter->mac_index == 0 &&
		    !memcmp(filter->l2_addr, bp->mac_addr,
			    RTE_ETHER_ADDR_LEN)) {
			rc = bnxt_hwrm_clear_l2_filter(bp, filter);
			if (!rc) {
				STAILQ_REMOVE(&vnic->filter, filter,
					      bnxt_filter_info, next);
				bnxt_free_filter(bp, filter);
			}
			return rc;
		}
		filter = STAILQ_NEXT(filter, next);
	}
	return 0;
}

static int
bnxt_config_vlan_hw_filter(struct bnxt *bp, uint64_t rx_offloads)
{
	struct bnxt_vnic_info *vnic;
	unsigned int i;
	int rc;

	vnic = BNXT_GET_DEFAULT_VNIC(bp);
	if (!(rx_offloads & DEV_RX_OFFLOAD_VLAN_FILTER)) {
		/* Remove any VLAN filters programmed */
		for (i = 0; i < RTE_ETHER_MAX_VLAN_ID; i++)
			bnxt_del_vlan_filter(bp, i);

		rc = bnxt_add_mac_filter(bp, vnic, NULL, 0, 0);
		if (rc)
			return rc;
	} else {
		/* Default filter will allow packets that match the
		 * dest mac. So, it has to be deleted, otherwise, we
		 * will endup receiving vlan packets for which the
		 * filter is not programmed, when hw-vlan-filter
		 * configuration is ON
		 */
		bnxt_del_dflt_mac_filter(bp, vnic);
		/* This filter will allow only untagged packets */
		bnxt_add_vlan_filter(bp, 0);
	}
	PMD_DRV_LOG(DEBUG, "VLAN Filtering: %d\n",
		    !!(rx_offloads & DEV_RX_OFFLOAD_VLAN_FILTER));

	return 0;
}

static int bnxt_free_one_vnic(struct bnxt *bp, uint16_t vnic_id)
{
	struct bnxt_vnic_info *vnic = &bp->vnic_info[vnic_id];
	unsigned int i;
	int rc;

	/* Destroy vnic filters and vnic */
	if (bp->eth_dev->data->dev_conf.rxmode.offloads &
	    DEV_RX_OFFLOAD_VLAN_FILTER) {
		for (i = 0; i < RTE_ETHER_MAX_VLAN_ID; i++)
			bnxt_del_vlan_filter(bp, i);
	}
	bnxt_del_dflt_mac_filter(bp, vnic);

	rc = bnxt_hwrm_vnic_free(bp, vnic);
	if (rc)
		return rc;

	rte_free(vnic->fw_grp_ids);
	vnic->fw_grp_ids = NULL;

	vnic->rx_queue_cnt = 0;

	return 0;
}

static int
bnxt_config_vlan_hw_stripping(struct bnxt *bp, uint64_t rx_offloads)
{
	struct bnxt_vnic_info *vnic = BNXT_GET_DEFAULT_VNIC(bp);
	int rc;

	/* Destroy, recreate and reconfigure the default vnic */
	rc = bnxt_free_one_vnic(bp, 0);
	if (rc)
		return rc;

	/* default vnic 0 */
	rc = bnxt_setup_one_vnic(bp, 0);
	if (rc)
		return rc;

	if (bp->eth_dev->data->dev_conf.rxmode.offloads &
	    DEV_RX_OFFLOAD_VLAN_FILTER) {
		rc = bnxt_add_vlan_filter(bp, 0);
		if (rc)
			return rc;
		rc = bnxt_restore_vlan_filters(bp);
		if (rc)
			return rc;
	} else {
		rc = bnxt_add_mac_filter(bp, vnic, NULL, 0, 0);
		if (rc)
			return rc;
	}

	rc = bnxt_hwrm_cfa_l2_set_rx_mask(bp, vnic, 0, NULL);
	if (rc)
		return rc;

	PMD_DRV_LOG(DEBUG, "VLAN Strip Offload: %d\n",
		    !!(rx_offloads & DEV_RX_OFFLOAD_VLAN_STRIP));

	return rc;
}

static int
bnxt_vlan_offload_set_op(struct rte_eth_dev *dev, int mask)
{
	uint64_t rx_offloads = dev->data->dev_conf.rxmode.offloads;
	struct bnxt *bp = dev->data->dev_private;
	int rc;

	rc = is_bnxt_in_error(bp);
	if (rc)
		return rc;

	/* Filter settings will get applied when port is started */
	if (!dev->data->dev_started)
		return 0;

	if (mask & ETH_VLAN_FILTER_MASK) {
		/* Enable or disable VLAN filtering */
		rc = bnxt_config_vlan_hw_filter(bp, rx_offloads);
		if (rc)
			return rc;
	}

	if (mask & ETH_VLAN_STRIP_MASK) {
		/* Enable or disable VLAN stripping */
		rc = bnxt_config_vlan_hw_stripping(bp, rx_offloads);
		if (rc)
			return rc;
	}

	if (mask & ETH_VLAN_EXTEND_MASK) {
		if (rx_offloads & DEV_RX_OFFLOAD_VLAN_EXTEND)
			PMD_DRV_LOG(DEBUG, "Extend VLAN supported\n");
		else
			PMD_DRV_LOG(INFO, "Extend VLAN unsupported\n");
	}

	return 0;
}

static int
bnxt_vlan_tpid_set_op(struct rte_eth_dev *dev, enum rte_vlan_type vlan_type,
		      uint16_t tpid)
{
	struct bnxt *bp = dev->data->dev_private;
	int qinq = dev->data->dev_conf.rxmode.offloads &
		   DEV_RX_OFFLOAD_VLAN_EXTEND;

	if (vlan_type != ETH_VLAN_TYPE_INNER &&
	    vlan_type != ETH_VLAN_TYPE_OUTER) {
		PMD_DRV_LOG(ERR,
			    "Unsupported vlan type.");
		return -EILWAL;
	}
	if (!qinq) {
		PMD_DRV_LOG(ERR,
			    "QinQ not enabled. Needs to be ON as we can "
			    "accelerate only outer vlan\n");
		return -EILWAL;
	}

	if (vlan_type == ETH_VLAN_TYPE_OUTER) {
		switch (tpid) {
		case RTE_ETHER_TYPE_QINQ:
			bp->outer_tpid_bd =
				TX_BD_LONG_CFA_META_VLAN_TPID_TPID88A8;
				break;
		case RTE_ETHER_TYPE_VLAN:
			bp->outer_tpid_bd =
				TX_BD_LONG_CFA_META_VLAN_TPID_TPID8100;
				break;
		case RTE_ETHER_TYPE_QINQ1:
			bp->outer_tpid_bd =
				TX_BD_LONG_CFA_META_VLAN_TPID_TPID9100;
				break;
		case RTE_ETHER_TYPE_QINQ2:
			bp->outer_tpid_bd =
				TX_BD_LONG_CFA_META_VLAN_TPID_TPID9200;
				break;
		case RTE_ETHER_TYPE_QINQ3:
			bp->outer_tpid_bd =
				 TX_BD_LONG_CFA_META_VLAN_TPID_TPID9300;
				break;
		default:
			PMD_DRV_LOG(ERR, "Invalid TPID: %x\n", tpid);
			return -EILWAL;
		}
		bp->outer_tpid_bd |= tpid;
		PMD_DRV_LOG(INFO, "outer_tpid_bd = %x\n", bp->outer_tpid_bd);
	} else if (vlan_type == ETH_VLAN_TYPE_INNER) {
		PMD_DRV_LOG(ERR,
			    "Can accelerate only outer vlan in QinQ\n");
		return -EILWAL;
	}

	return 0;
}

static int
bnxt_set_default_mac_addr_op(struct rte_eth_dev *dev,
			     struct rte_ether_addr *addr)
{
	struct bnxt *bp = dev->data->dev_private;
	/* Default Filter is tied to VNIC 0 */
	struct bnxt_vnic_info *vnic = BNXT_GET_DEFAULT_VNIC(bp);
	int rc;

	rc = is_bnxt_in_error(bp);
	if (rc)
		return rc;

	if (BNXT_VF(bp) && !BNXT_VF_IS_TRUSTED(bp))
		return -EPERM;

	if (rte_is_zero_ether_addr(addr))
		return -EILWAL;

	/* Filter settings will get applied when port is started */
	if (!dev->data->dev_started)
		return 0;

	/* Check if the requested MAC is already added */
	if (memcmp(addr, bp->mac_addr, RTE_ETHER_ADDR_LEN) == 0)
		return 0;

	/* Destroy filter and re-create it */
	bnxt_del_dflt_mac_filter(bp, vnic);

	memcpy(bp->mac_addr, addr, RTE_ETHER_ADDR_LEN);
	if (dev->data->dev_conf.rxmode.offloads & DEV_RX_OFFLOAD_VLAN_FILTER) {
		/* This filter will allow only untagged packets */
		rc = bnxt_add_vlan_filter(bp, 0);
	} else {
		rc = bnxt_add_mac_filter(bp, vnic, addr, 0, 0);
	}

	PMD_DRV_LOG(DEBUG, "Set MAC addr\n");
	return rc;
}

static int
bnxt_dev_set_mc_addr_list_op(struct rte_eth_dev *eth_dev,
			  struct rte_ether_addr *mc_addr_set,
			  uint32_t nb_mc_addr)
{
	struct bnxt *bp = eth_dev->data->dev_private;
	char *mc_addr_list = (char *)mc_addr_set;
	struct bnxt_vnic_info *vnic;
	uint32_t off = 0, i = 0;
	int rc;

	rc = is_bnxt_in_error(bp);
	if (rc)
		return rc;

	vnic = BNXT_GET_DEFAULT_VNIC(bp);

	if (nb_mc_addr > BNXT_MAX_MC_ADDRS) {
		vnic->flags |= BNXT_VNIC_INFO_ALLMULTI;
		goto allmulti;
	}

	/* TODO Check for Duplicate mcast addresses */
	vnic->flags &= ~BNXT_VNIC_INFO_ALLMULTI;
	for (i = 0; i < nb_mc_addr; i++) {
		memcpy(vnic->mc_list + off, &mc_addr_list[i],
			RTE_ETHER_ADDR_LEN);
		off += RTE_ETHER_ADDR_LEN;
	}

	vnic->mc_addr_cnt = i;
	if (vnic->mc_addr_cnt)
		vnic->flags |= BNXT_VNIC_INFO_MCAST;
	else
		vnic->flags &= ~BNXT_VNIC_INFO_MCAST;

allmulti:
	return bnxt_hwrm_cfa_l2_set_rx_mask(bp, vnic, 0, NULL);
}

static int
bnxt_fw_version_get(struct rte_eth_dev *dev, char *fw_version, size_t fw_size)
{
	struct bnxt *bp = dev->data->dev_private;
	uint8_t fw_major = (bp->fw_ver >> 24) & 0xff;
	uint8_t fw_minor = (bp->fw_ver >> 16) & 0xff;
	uint8_t fw_updt = (bp->fw_ver >> 8) & 0xff;
	uint8_t fw_rsvd = bp->fw_ver & 0xff;
	int ret;

	ret = snprintf(fw_version, fw_size, "%d.%d.%d.%d",
			fw_major, fw_minor, fw_updt, fw_rsvd);

	ret += 1; /* add the size of '\0' */
	if (fw_size < (uint32_t)ret)
		return ret;
	else
		return 0;
}

static void
bnxt_rxq_info_get_op(struct rte_eth_dev *dev, uint16_t queue_id,
	struct rte_eth_rxq_info *qinfo)
{
	struct bnxt *bp = dev->data->dev_private;
	struct bnxt_rx_queue *rxq;

	if (is_bnxt_in_error(bp))
		return;

	rxq = dev->data->rx_queues[queue_id];

	qinfo->mp = rxq->mb_pool;
	qinfo->scattered_rx = dev->data->scattered_rx;
	qinfo->nb_desc = rxq->nb_rx_desc;

	qinfo->conf.rx_free_thresh = rxq->rx_free_thresh;
	qinfo->conf.rx_drop_en = rxq->drop_en;
	qinfo->conf.rx_deferred_start = rxq->rx_deferred_start;
	qinfo->conf.offloads = dev->data->dev_conf.rxmode.offloads;
}

static void
bnxt_txq_info_get_op(struct rte_eth_dev *dev, uint16_t queue_id,
	struct rte_eth_txq_info *qinfo)
{
	struct bnxt *bp = dev->data->dev_private;
	struct bnxt_tx_queue *txq;

	if (is_bnxt_in_error(bp))
		return;

	txq = dev->data->tx_queues[queue_id];

	qinfo->nb_desc = txq->nb_tx_desc;

	qinfo->conf.tx_thresh.pthresh = txq->pthresh;
	qinfo->conf.tx_thresh.hthresh = txq->hthresh;
	qinfo->conf.tx_thresh.wthresh = txq->wthresh;

	qinfo->conf.tx_free_thresh = txq->tx_free_thresh;
	qinfo->conf.tx_rs_thresh = 0;
	qinfo->conf.tx_deferred_start = txq->tx_deferred_start;
	qinfo->conf.offloads = txq->offloads;
}

static const struct {
	eth_rx_burst_t pkt_burst;
	const char *info;
} bnxt_rx_burst_info[] = {
	{bnxt_recv_pkts,	"Scalar"},
#if defined(RTE_ARCH_X86)
	{bnxt_recv_pkts_vec,	"Vector SSE"},
#elif defined(RTE_ARCH_ARM64)
	{bnxt_recv_pkts_vec,	"Vector Neon"},
#endif
};

static int
bnxt_rx_burst_mode_get(struct rte_eth_dev *dev, __rte_unused uint16_t queue_id,
		       struct rte_eth_burst_mode *mode)
{
	eth_rx_burst_t pkt_burst = dev->rx_pkt_burst;
	size_t i;

	for (i = 0; i < RTE_DIM(bnxt_rx_burst_info); i++) {
		if (pkt_burst == bnxt_rx_burst_info[i].pkt_burst) {
			snprintf(mode->info, sizeof(mode->info), "%s",
				 bnxt_rx_burst_info[i].info);
			return 0;
		}
	}

	return -EILWAL;
}

static const struct {
	eth_tx_burst_t pkt_burst;
	const char *info;
} bnxt_tx_burst_info[] = {
	{bnxt_xmit_pkts,	"Scalar"},
#if defined(RTE_ARCH_X86)
	{bnxt_xmit_pkts_vec,	"Vector SSE"},
#elif defined(RTE_ARCH_ARM64)
	{bnxt_xmit_pkts_vec,	"Vector Neon"},
#endif
};

static int
bnxt_tx_burst_mode_get(struct rte_eth_dev *dev, __rte_unused uint16_t queue_id,
		       struct rte_eth_burst_mode *mode)
{
	eth_tx_burst_t pkt_burst = dev->tx_pkt_burst;
	size_t i;

	for (i = 0; i < RTE_DIM(bnxt_tx_burst_info); i++) {
		if (pkt_burst == bnxt_tx_burst_info[i].pkt_burst) {
			snprintf(mode->info, sizeof(mode->info), "%s",
				 bnxt_tx_burst_info[i].info);
			return 0;
		}
	}

	return -EILWAL;
}

int bnxt_mtu_set_op(struct rte_eth_dev *eth_dev, uint16_t new_mtu)
{
	struct bnxt *bp = eth_dev->data->dev_private;
	uint32_t new_pkt_size;
	uint32_t rc = 0;
	uint32_t i;

	rc = is_bnxt_in_error(bp);
	if (rc)
		return rc;

	/* Exit if receive queues are not configured yet */
	if (!eth_dev->data->nb_rx_queues)
		return rc;

	new_pkt_size = new_mtu + RTE_ETHER_HDR_LEN + RTE_ETHER_CRC_LEN +
		       VLAN_TAG_SIZE * BNXT_NUM_VLANS;

	/*
	 * Disallow any MTU change that would require scattered receive support
	 * if it is not already enabled.
	 */
	if (eth_dev->data->dev_started &&
	    !eth_dev->data->scattered_rx &&
	    (new_pkt_size >
	     eth_dev->data->min_rx_buf_size - RTE_PKTMBUF_HEADROOM)) {
		PMD_DRV_LOG(ERR,
			    "MTU change would require scattered rx support. ");
		PMD_DRV_LOG(ERR, "Stop port before changing MTU.\n");
		return -EILWAL;
	}

	if (new_mtu > RTE_ETHER_MTU) {
		bp->flags |= BNXT_FLAG_JUMBO;
		bp->eth_dev->data->dev_conf.rxmode.offloads |=
			DEV_RX_OFFLOAD_JUMBO_FRAME;
	} else {
		bp->eth_dev->data->dev_conf.rxmode.offloads &=
			~DEV_RX_OFFLOAD_JUMBO_FRAME;
		bp->flags &= ~BNXT_FLAG_JUMBO;
	}

	/* Is there a change in mtu setting? */
	if (eth_dev->data->dev_conf.rxmode.max_rx_pkt_len == new_pkt_size)
		return rc;

	for (i = 0; i < bp->nr_vnics; i++) {
		struct bnxt_vnic_info *vnic = &bp->vnic_info[i];
		uint16_t size = 0;

		vnic->mru = BNXT_VNIC_MRU(new_mtu);
		rc = bnxt_hwrm_vnic_cfg(bp, vnic);
		if (rc)
			break;

		size = rte_pktmbuf_data_room_size(bp->rx_queues[0]->mb_pool);
		size -= RTE_PKTMBUF_HEADROOM;

		if (size < new_mtu) {
			rc = bnxt_hwrm_vnic_plcmode_cfg(bp, vnic);
			if (rc)
				return rc;
		}
	}

	if (!rc)
		eth_dev->data->dev_conf.rxmode.max_rx_pkt_len = new_pkt_size;

	PMD_DRV_LOG(INFO, "New MTU is %d\n", new_mtu);

	return rc;
}

static int
bnxt_vlan_pvid_set_op(struct rte_eth_dev *dev, uint16_t pvid, int on)
{
	struct bnxt *bp = dev->data->dev_private;
	uint16_t vlan = bp->vlan;
	int rc;

	rc = is_bnxt_in_error(bp);
	if (rc)
		return rc;

	if (!BNXT_SINGLE_PF(bp) || BNXT_VF(bp)) {
		PMD_DRV_LOG(ERR,
			"PVID cannot be modified for this function\n");
		return -ENOTSUP;
	}
	bp->vlan = on ? pvid : 0;

	rc = bnxt_hwrm_set_default_vlan(bp, 0, 0);
	if (rc)
		bp->vlan = vlan;
	return rc;
}

static int
bnxt_dev_led_on_op(struct rte_eth_dev *dev)
{
	struct bnxt *bp = dev->data->dev_private;
	int rc;

	rc = is_bnxt_in_error(bp);
	if (rc)
		return rc;

	return bnxt_hwrm_port_led_cfg(bp, true);
}

static int
bnxt_dev_led_off_op(struct rte_eth_dev *dev)
{
	struct bnxt *bp = dev->data->dev_private;
	int rc;

	rc = is_bnxt_in_error(bp);
	if (rc)
		return rc;

	return bnxt_hwrm_port_led_cfg(bp, false);
}

static uint32_t
bnxt_rx_queue_count_op(struct rte_eth_dev *dev, uint16_t rx_queue_id)
{
	struct bnxt *bp = (struct bnxt *)dev->data->dev_private;
	uint32_t desc = 0, raw_cons = 0, cons;
	struct bnxt_cp_ring_info *cpr;
	struct bnxt_rx_queue *rxq;
	struct rx_pkt_cmpl *rxcmp;
	int rc;

	rc = is_bnxt_in_error(bp);
	if (rc)
		return rc;

	rxq = dev->data->rx_queues[rx_queue_id];
	cpr = rxq->cp_ring;
	raw_cons = cpr->cp_raw_cons;

	while (1) {
		cons = RING_CMP(cpr->cp_ring_struct, raw_cons);
		rte_prefetch0(&cpr->cp_desc_ring[cons]);
		rxcmp = (struct rx_pkt_cmpl *)&cpr->cp_desc_ring[cons];

		if (!CMP_VALID(rxcmp, raw_cons, cpr->cp_ring_struct)) {
			break;
		} else {
			raw_cons++;
			desc++;
		}
	}

	return desc;
}

static int
bnxt_rx_descriptor_status_op(void *rx_queue, uint16_t offset)
{
	struct bnxt_rx_queue *rxq = (struct bnxt_rx_queue *)rx_queue;
	struct bnxt_rx_ring_info *rxr;
	struct bnxt_cp_ring_info *cpr;
	struct rte_mbuf *rx_buf;
	struct rx_pkt_cmpl *rxcmp;
	uint32_t cons, cp_cons;
	int rc;

	if (!rxq)
		return -EILWAL;

	rc = is_bnxt_in_error(rxq->bp);
	if (rc)
		return rc;

	cpr = rxq->cp_ring;
	rxr = rxq->rx_ring;

	if (offset >= rxq->nb_rx_desc)
		return -EILWAL;

	cons = RING_CMP(cpr->cp_ring_struct, offset);
	cp_cons = cpr->cp_raw_cons;
	rxcmp = (struct rx_pkt_cmpl *)&cpr->cp_desc_ring[cons];

	if (cons > cp_cons) {
		if (CMPL_VALID(rxcmp, cpr->valid))
			return RTE_ETH_RX_DESC_DONE;
	} else {
		if (CMPL_VALID(rxcmp, !cpr->valid))
			return RTE_ETH_RX_DESC_DONE;
	}
	rx_buf = rxr->rx_buf_ring[cons];
	if (rx_buf == NULL || rx_buf == &rxq->fake_mbuf)
		return RTE_ETH_RX_DESC_UNAVAIL;


	return RTE_ETH_RX_DESC_AVAIL;
}

static int
bnxt_tx_descriptor_status_op(void *tx_queue, uint16_t offset)
{
	struct bnxt_tx_queue *txq = (struct bnxt_tx_queue *)tx_queue;
	struct bnxt_tx_ring_info *txr;
	struct bnxt_cp_ring_info *cpr;
	struct bnxt_sw_tx_bd *tx_buf;
	struct tx_pkt_cmpl *txcmp;
	uint32_t cons, cp_cons;
	int rc;

	if (!txq)
		return -EILWAL;

	rc = is_bnxt_in_error(txq->bp);
	if (rc)
		return rc;

	cpr = txq->cp_ring;
	txr = txq->tx_ring;

	if (offset >= txq->nb_tx_desc)
		return -EILWAL;

	cons = RING_CMP(cpr->cp_ring_struct, offset);
	txcmp = (struct tx_pkt_cmpl *)&cpr->cp_desc_ring[cons];
	cp_cons = cpr->cp_raw_cons;

	if (cons > cp_cons) {
		if (CMPL_VALID(txcmp, cpr->valid))
			return RTE_ETH_TX_DESC_UNAVAIL;
	} else {
		if (CMPL_VALID(txcmp, !cpr->valid))
			return RTE_ETH_TX_DESC_UNAVAIL;
	}
	tx_buf = &txr->tx_buf_ring[cons];
	if (tx_buf->mbuf == NULL)
		return RTE_ETH_TX_DESC_DONE;

	return RTE_ETH_TX_DESC_FULL;
}

int
bnxt_filter_ctrl_op(struct rte_eth_dev *dev,
		    enum rte_filter_type filter_type,
		    enum rte_filter_op filter_op, void *arg)
{
	struct bnxt *bp = dev->data->dev_private;
	int ret = 0;

	if (!bp)
		return -EIO;

	if (BNXT_ETH_DEV_IS_REPRESENTOR(dev)) {
		struct bnxt_representor *vfr = dev->data->dev_private;
		bp = vfr->parent_dev->data->dev_private;
		/* parent is deleted while children are still valid */
		if (!bp) {
			PMD_DRV_LOG(DEBUG, "BNXT Port:%d VFR Error %d:%d\n",
				    dev->data->port_id,
				    filter_type,
				    filter_op);
			return -EIO;
		}
	}

	ret = is_bnxt_in_error(bp);
	if (ret)
		return ret;

	switch (filter_type) {
	case RTE_ETH_FILTER_GENERIC:
		if (filter_op != RTE_ETH_FILTER_GET)
			return -EILWAL;

		/* PMD supports thread-safe flow operations.  rte_flow API
		 * functions can avoid mutex for multi-thread safety.
		 */
		dev->data->dev_flags |= RTE_ETH_DEV_FLOW_OPS_THREAD_SAFE;

		if (BNXT_TRUFLOW_EN(bp))
			*(const void **)arg = &bnxt_ulp_rte_flow_ops;
		else
			*(const void **)arg = &bnxt_flow_ops;
		break;
	default:
		PMD_DRV_LOG(ERR,
			"Filter type (%d) not supported", filter_type);
		ret = -EILWAL;
		break;
	}
	return ret;
}

static const uint32_t *
bnxt_dev_supported_ptypes_get_op(struct rte_eth_dev *dev)
{
	static const uint32_t ptypes[] = {
		RTE_PTYPE_L2_ETHER_VLAN,
		RTE_PTYPE_L3_IPV4_EXT_UNKNOWN,
		RTE_PTYPE_L3_IPV6_EXT_UNKNOWN,
		RTE_PTYPE_L4_ICMP,
		RTE_PTYPE_L4_TCP,
		RTE_PTYPE_L4_UDP,
		RTE_PTYPE_INNER_L3_IPV4_EXT_UNKNOWN,
		RTE_PTYPE_INNER_L3_IPV6_EXT_UNKNOWN,
		RTE_PTYPE_INNER_L4_ICMP,
		RTE_PTYPE_INNER_L4_TCP,
		RTE_PTYPE_INNER_L4_UDP,
		RTE_PTYPE_UNKNOWN
	};

	if (!dev->rx_pkt_burst)
		return NULL;

	return ptypes;
}

static int bnxt_map_regs(struct bnxt *bp, uint32_t *reg_arr, int count,
			 int reg_win)
{
	uint32_t reg_base = *reg_arr & 0xfffff000;
	uint32_t win_off;
	int i;

	for (i = 0; i < count; i++) {
		if ((reg_arr[i] & 0xfffff000) != reg_base)
			return -ERANGE;
	}
	win_off = BNXT_GRCPF_REG_WINDOW_BASE_OUT + (reg_win - 1) * 4;
	rte_write32(reg_base, (uint8_t *)bp->bar0 + win_off);
	return 0;
}

static int bnxt_map_ptp_regs(struct bnxt *bp)
{
	struct bnxt_ptp_cfg *ptp = bp->ptp_cfg;
	uint32_t *reg_arr;
	int rc, i;

	reg_arr = ptp->rx_regs;
	rc = bnxt_map_regs(bp, reg_arr, BNXT_PTP_RX_REGS, 5);
	if (rc)
		return rc;

	reg_arr = ptp->tx_regs;
	rc = bnxt_map_regs(bp, reg_arr, BNXT_PTP_TX_REGS, 6);
	if (rc)
		return rc;

	for (i = 0; i < BNXT_PTP_RX_REGS; i++)
		ptp->rx_mapped_regs[i] = 0x5000 + (ptp->rx_regs[i] & 0xfff);

	for (i = 0; i < BNXT_PTP_TX_REGS; i++)
		ptp->tx_mapped_regs[i] = 0x6000 + (ptp->tx_regs[i] & 0xfff);

	return 0;
}

static void bnxt_unmap_ptp_regs(struct bnxt *bp)
{
	rte_write32(0, (uint8_t *)bp->bar0 +
			 BNXT_GRCPF_REG_WINDOW_BASE_OUT + 16);
	rte_write32(0, (uint8_t *)bp->bar0 +
			 BNXT_GRCPF_REG_WINDOW_BASE_OUT + 20);
}

static uint64_t bnxt_cc_read(struct bnxt *bp)
{
	uint64_t ns;

	ns = rte_le_to_cpu_32(rte_read32((uint8_t *)bp->bar0 +
			      BNXT_GRCPF_REG_SYNC_TIME));
	ns |= (uint64_t)(rte_le_to_cpu_32(rte_read32((uint8_t *)bp->bar0 +
					  BNXT_GRCPF_REG_SYNC_TIME + 4))) << 32;
	return ns;
}

static int bnxt_get_tx_ts(struct bnxt *bp, uint64_t *ts)
{
	struct bnxt_ptp_cfg *ptp = bp->ptp_cfg;
	uint32_t fifo;

	fifo = rte_le_to_cpu_32(rte_read32((uint8_t *)bp->bar0 +
				ptp->tx_mapped_regs[BNXT_PTP_TX_FIFO]));
	if (fifo & BNXT_PTP_TX_FIFO_EMPTY)
		return -EAGAIN;

	fifo = rte_le_to_cpu_32(rte_read32((uint8_t *)bp->bar0 +
				ptp->tx_mapped_regs[BNXT_PTP_TX_FIFO]));
	*ts = rte_le_to_cpu_32(rte_read32((uint8_t *)bp->bar0 +
				ptp->tx_mapped_regs[BNXT_PTP_TX_TS_L]));
	*ts |= (uint64_t)rte_le_to_cpu_32(rte_read32((uint8_t *)bp->bar0 +
				ptp->tx_mapped_regs[BNXT_PTP_TX_TS_H])) << 32;

	return 0;
}

static int bnxt_get_rx_ts(struct bnxt *bp, uint64_t *ts)
{
	struct bnxt_ptp_cfg *ptp = bp->ptp_cfg;
	struct bnxt_pf_info *pf = bp->pf;
	uint16_t port_id;
	uint32_t fifo;

	if (!ptp)
		return -ENODEV;

	fifo = rte_le_to_cpu_32(rte_read32((uint8_t *)bp->bar0 +
				ptp->rx_mapped_regs[BNXT_PTP_RX_FIFO]));
	if (!(fifo & BNXT_PTP_RX_FIFO_PENDING))
		return -EAGAIN;

	port_id = pf->port_id;
	rte_write32(1 << port_id, (uint8_t *)bp->bar0 +
	       ptp->rx_mapped_regs[BNXT_PTP_RX_FIFO_ADV]);

	fifo = rte_le_to_cpu_32(rte_read32((uint8_t *)bp->bar0 +
				   ptp->rx_mapped_regs[BNXT_PTP_RX_FIFO]));
	if (fifo & BNXT_PTP_RX_FIFO_PENDING) {
/*		bnxt_clr_rx_ts(bp);	  TBD  */
		return -EBUSY;
	}

	*ts = rte_le_to_cpu_32(rte_read32((uint8_t *)bp->bar0 +
				ptp->rx_mapped_regs[BNXT_PTP_RX_TS_L]));
	*ts |= (uint64_t)rte_le_to_cpu_32(rte_read32((uint8_t *)bp->bar0 +
				ptp->rx_mapped_regs[BNXT_PTP_RX_TS_H])) << 32;

	return 0;
}

static int
bnxt_timesync_write_time(struct rte_eth_dev *dev, const struct timespec *ts)
{
	uint64_t ns;
	struct bnxt *bp = dev->data->dev_private;
	struct bnxt_ptp_cfg *ptp = bp->ptp_cfg;

	if (!ptp)
		return 0;

	ns = rte_timespec_to_ns(ts);
	/* Set the timecounters to a new value. */
	ptp->tc.nsec = ns;

	return 0;
}

static int
bnxt_timesync_read_time(struct rte_eth_dev *dev, struct timespec *ts)
{
	struct bnxt *bp = dev->data->dev_private;
	struct bnxt_ptp_cfg *ptp = bp->ptp_cfg;
	uint64_t ns, systime_cycles = 0;
	int rc = 0;

	if (!ptp)
		return 0;

	if (BNXT_CHIP_THOR(bp))
		rc = bnxt_hwrm_port_ts_query(bp, BNXT_PTP_FLAGS_LWRRENT_TIME,
					     &systime_cycles);
	else
		systime_cycles = bnxt_cc_read(bp);

	ns = rte_timecounter_update(&ptp->tc, systime_cycles);
	*ts = rte_ns_to_timespec(ns);

	return rc;
}
static int
bnxt_timesync_enable(struct rte_eth_dev *dev)
{
	struct bnxt *bp = dev->data->dev_private;
	struct bnxt_ptp_cfg *ptp = bp->ptp_cfg;
	uint32_t shift = 0;
	int rc;

	if (!ptp)
		return 0;

	ptp->rx_filter = 1;
	ptp->tx_tstamp_en = 1;
	ptp->rxctl = BNXT_PTP_MSG_EVENTS;

	rc = bnxt_hwrm_ptp_cfg(bp);
	if (rc)
		return rc;

	memset(&ptp->tc, 0, sizeof(struct rte_timecounter));
	memset(&ptp->rx_tstamp_tc, 0, sizeof(struct rte_timecounter));
	memset(&ptp->tx_tstamp_tc, 0, sizeof(struct rte_timecounter));

	ptp->tc.cc_mask = BNXT_CYCLECOUNTER_MASK;
	ptp->tc.cc_shift = shift;
	ptp->tc.nsec_mask = (1ULL << shift) - 1;

	ptp->rx_tstamp_tc.cc_mask = BNXT_CYCLECOUNTER_MASK;
	ptp->rx_tstamp_tc.cc_shift = shift;
	ptp->rx_tstamp_tc.nsec_mask = (1ULL << shift) - 1;

	ptp->tx_tstamp_tc.cc_mask = BNXT_CYCLECOUNTER_MASK;
	ptp->tx_tstamp_tc.cc_shift = shift;
	ptp->tx_tstamp_tc.nsec_mask = (1ULL << shift) - 1;

	if (!BNXT_CHIP_THOR(bp))
		bnxt_map_ptp_regs(bp);

	return 0;
}

static int
bnxt_timesync_disable(struct rte_eth_dev *dev)
{
	struct bnxt *bp = dev->data->dev_private;
	struct bnxt_ptp_cfg *ptp = bp->ptp_cfg;

	if (!ptp)
		return 0;

	ptp->rx_filter = 0;
	ptp->tx_tstamp_en = 0;
	ptp->rxctl = 0;

	bnxt_hwrm_ptp_cfg(bp);

	if (!BNXT_CHIP_THOR(bp))
		bnxt_unmap_ptp_regs(bp);

	return 0;
}

static int
bnxt_timesync_read_rx_timestamp(struct rte_eth_dev *dev,
				 struct timespec *timestamp,
				 uint32_t flags __rte_unused)
{
	struct bnxt *bp = dev->data->dev_private;
	struct bnxt_ptp_cfg *ptp = bp->ptp_cfg;
	uint64_t rx_tstamp_cycles = 0;
	uint64_t ns;

	if (!ptp)
		return 0;

	if (BNXT_CHIP_THOR(bp))
		rx_tstamp_cycles = ptp->rx_timestamp;
	else
		bnxt_get_rx_ts(bp, &rx_tstamp_cycles);

	ns = rte_timecounter_update(&ptp->rx_tstamp_tc, rx_tstamp_cycles);
	*timestamp = rte_ns_to_timespec(ns);
	return  0;
}

static int
bnxt_timesync_read_tx_timestamp(struct rte_eth_dev *dev,
				 struct timespec *timestamp)
{
	struct bnxt *bp = dev->data->dev_private;
	struct bnxt_ptp_cfg *ptp = bp->ptp_cfg;
	uint64_t tx_tstamp_cycles = 0;
	uint64_t ns;
	int rc = 0;

	if (!ptp)
		return 0;

	if (BNXT_CHIP_THOR(bp))
		rc = bnxt_hwrm_port_ts_query(bp, BNXT_PTP_FLAGS_PATH_TX,
					     &tx_tstamp_cycles);
	else
		rc = bnxt_get_tx_ts(bp, &tx_tstamp_cycles);

	ns = rte_timecounter_update(&ptp->tx_tstamp_tc, tx_tstamp_cycles);
	*timestamp = rte_ns_to_timespec(ns);

	return rc;
}

static int
bnxt_timesync_adjust_time(struct rte_eth_dev *dev, int64_t delta)
{
	struct bnxt *bp = dev->data->dev_private;
	struct bnxt_ptp_cfg *ptp = bp->ptp_cfg;

	if (!ptp)
		return 0;

	ptp->tc.nsec += delta;

	return 0;
}

static int
bnxt_get_eeprom_length_op(struct rte_eth_dev *dev)
{
	struct bnxt *bp = dev->data->dev_private;
	int rc;
	uint32_t dir_entries;
	uint32_t entry_length;

	rc = is_bnxt_in_error(bp);
	if (rc)
		return rc;

	PMD_DRV_LOG(INFO, PCI_PRI_FMT "\n",
		    bp->pdev->addr.domain, bp->pdev->addr.bus,
		    bp->pdev->addr.devid, bp->pdev->addr.function);

	rc = bnxt_hwrm_lwm_get_dir_info(bp, &dir_entries, &entry_length);
	if (rc != 0)
		return rc;

	return dir_entries * entry_length;
}

static int
bnxt_get_eeprom_op(struct rte_eth_dev *dev,
		struct rte_dev_eeprom_info *in_eeprom)
{
	struct bnxt *bp = dev->data->dev_private;
	uint32_t index;
	uint32_t offset;
	int rc;

	rc = is_bnxt_in_error(bp);
	if (rc)
		return rc;

	PMD_DRV_LOG(INFO, PCI_PRI_FMT " in_eeprom->offset = %d len = %d\n",
		    bp->pdev->addr.domain, bp->pdev->addr.bus,
		    bp->pdev->addr.devid, bp->pdev->addr.function,
		    in_eeprom->offset, in_eeprom->length);

	if (in_eeprom->offset == 0) /* special offset value to get directory */
		return bnxt_get_lwram_directory(bp, in_eeprom->length,
						in_eeprom->data);

	index = in_eeprom->offset >> 24;
	offset = in_eeprom->offset & 0xffffff;

	if (index != 0)
		return bnxt_hwrm_get_lwram_item(bp, index - 1, offset,
					   in_eeprom->length, in_eeprom->data);

	return 0;
}

static bool bnxt_dir_type_is_ape_bin_format(uint16_t dir_type)
{
	switch (dir_type) {
	case BNX_DIR_TYPE_CHIMP_PATCH:
	case BNX_DIR_TYPE_BOOTCODE:
	case BNX_DIR_TYPE_BOOTCODE_2:
	case BNX_DIR_TYPE_APE_FW:
	case BNX_DIR_TYPE_APE_PATCH:
	case BNX_DIR_TYPE_KONG_FW:
	case BNX_DIR_TYPE_KONG_PATCH:
	case BNX_DIR_TYPE_BONO_FW:
	case BNX_DIR_TYPE_BONO_PATCH:
		/* FALLTHROUGH */
		return true;
	}

	return false;
}

static bool bnxt_dir_type_is_other_exec_format(uint16_t dir_type)
{
	switch (dir_type) {
	case BNX_DIR_TYPE_AVS:
	case BNX_DIR_TYPE_EXP_ROM_MBA:
	case BNX_DIR_TYPE_PCIE:
	case BNX_DIR_TYPE_TSCF_UCODE:
	case BNX_DIR_TYPE_EXT_PHY:
	case BNX_DIR_TYPE_CCM:
	case BNX_DIR_TYPE_ISCSI_BOOT:
	case BNX_DIR_TYPE_ISCSI_BOOT_IPV6:
	case BNX_DIR_TYPE_ISCSI_BOOT_IPV4N6:
		/* FALLTHROUGH */
		return true;
	}

	return false;
}

static bool bnxt_dir_type_is_exelwtable(uint16_t dir_type)
{
	return bnxt_dir_type_is_ape_bin_format(dir_type) ||
		bnxt_dir_type_is_other_exec_format(dir_type);
}

static int
bnxt_set_eeprom_op(struct rte_eth_dev *dev,
		struct rte_dev_eeprom_info *in_eeprom)
{
	struct bnxt *bp = dev->data->dev_private;
	uint8_t index, dir_op;
	uint16_t type, ext, ordinal, attr;
	int rc;

	rc = is_bnxt_in_error(bp);
	if (rc)
		return rc;

	PMD_DRV_LOG(INFO, PCI_PRI_FMT " in_eeprom->offset = %d len = %d\n",
		    bp->pdev->addr.domain, bp->pdev->addr.bus,
		    bp->pdev->addr.devid, bp->pdev->addr.function,
		    in_eeprom->offset, in_eeprom->length);

	if (!BNXT_PF(bp)) {
		PMD_DRV_LOG(ERR, "LWM write not supported from a VF\n");
		return -EILWAL;
	}

	type = in_eeprom->magic >> 16;

	if (type == 0xffff) { /* special value for directory operations */
		index = in_eeprom->magic & 0xff;
		dir_op = in_eeprom->magic >> 8;
		if (index == 0)
			return -EILWAL;
		switch (dir_op) {
		case 0x0e: /* erase */
			if (in_eeprom->offset != ~in_eeprom->magic)
				return -EILWAL;
			return bnxt_hwrm_erase_lwram_directory(bp, index - 1);
		default:
			return -EILWAL;
		}
	}

	/* Create or re-write an LWM item: */
	if (bnxt_dir_type_is_exelwtable(type) == true)
		return -EOPNOTSUPP;
	ext = in_eeprom->magic & 0xffff;
	ordinal = in_eeprom->offset >> 16;
	attr = in_eeprom->offset & 0xffff;

	return bnxt_hwrm_flash_lwram(bp, type, ordinal, ext, attr,
				     in_eeprom->data, in_eeprom->length);
}

/*
 * Initialization
 */

static const struct eth_dev_ops bnxt_dev_ops = {
	.dev_infos_get = bnxt_dev_info_get_op,
	.dev_close = bnxt_dev_close_op,
	.dev_configure = bnxt_dev_configure_op,
	.dev_start = bnxt_dev_start_op,
	.dev_stop = bnxt_dev_stop_op,
	.dev_set_link_up = bnxt_dev_set_link_up_op,
	.dev_set_link_down = bnxt_dev_set_link_down_op,
	.stats_get = bnxt_stats_get_op,
	.stats_reset = bnxt_stats_reset_op,
	.rx_queue_setup = bnxt_rx_queue_setup_op,
	.rx_queue_release = bnxt_rx_queue_release_op,
	.tx_queue_setup = bnxt_tx_queue_setup_op,
	.tx_queue_release = bnxt_tx_queue_release_op,
	.rx_queue_intr_enable = bnxt_rx_queue_intr_enable_op,
	.rx_queue_intr_disable = bnxt_rx_queue_intr_disable_op,
	.reta_update = bnxt_reta_update_op,
	.reta_query = bnxt_reta_query_op,
	.rss_hash_update = bnxt_rss_hash_update_op,
	.rss_hash_conf_get = bnxt_rss_hash_conf_get_op,
	.link_update = bnxt_link_update_op,
	.promislwous_enable = bnxt_promislwous_enable_op,
	.promislwous_disable = bnxt_promislwous_disable_op,
	.allmulticast_enable = bnxt_allmulticast_enable_op,
	.allmulticast_disable = bnxt_allmulticast_disable_op,
	.mac_addr_add = bnxt_mac_addr_add_op,
	.mac_addr_remove = bnxt_mac_addr_remove_op,
	.flow_ctrl_get = bnxt_flow_ctrl_get_op,
	.flow_ctrl_set = bnxt_flow_ctrl_set_op,
	.udp_tunnel_port_add  = bnxt_udp_tunnel_port_add_op,
	.udp_tunnel_port_del  = bnxt_udp_tunnel_port_del_op,
	.vlan_filter_set = bnxt_vlan_filter_set_op,
	.vlan_offload_set = bnxt_vlan_offload_set_op,
	.vlan_tpid_set = bnxt_vlan_tpid_set_op,
	.vlan_pvid_set = bnxt_vlan_pvid_set_op,
	.mtu_set = bnxt_mtu_set_op,
	.mac_addr_set = bnxt_set_default_mac_addr_op,
	.xstats_get = bnxt_dev_xstats_get_op,
	.xstats_get_names = bnxt_dev_xstats_get_names_op,
	.xstats_reset = bnxt_dev_xstats_reset_op,
	.fw_version_get = bnxt_fw_version_get,
	.set_mc_addr_list = bnxt_dev_set_mc_addr_list_op,
	.rxq_info_get = bnxt_rxq_info_get_op,
	.txq_info_get = bnxt_txq_info_get_op,
	.rx_burst_mode_get = bnxt_rx_burst_mode_get,
	.tx_burst_mode_get = bnxt_tx_burst_mode_get,
	.dev_led_on = bnxt_dev_led_on_op,
	.dev_led_off = bnxt_dev_led_off_op,
	.rx_queue_start = bnxt_rx_queue_start,
	.rx_queue_stop = bnxt_rx_queue_stop,
	.tx_queue_start = bnxt_tx_queue_start,
	.tx_queue_stop = bnxt_tx_queue_stop,
	.filter_ctrl = bnxt_filter_ctrl_op,
	.dev_supported_ptypes_get = bnxt_dev_supported_ptypes_get_op,
	.get_eeprom_length    = bnxt_get_eeprom_length_op,
	.get_eeprom           = bnxt_get_eeprom_op,
	.set_eeprom           = bnxt_set_eeprom_op,
	.timesync_enable      = bnxt_timesync_enable,
	.timesync_disable     = bnxt_timesync_disable,
	.timesync_read_time   = bnxt_timesync_read_time,
	.timesync_write_time   = bnxt_timesync_write_time,
	.timesync_adjust_time = bnxt_timesync_adjust_time,
	.timesync_read_rx_timestamp = bnxt_timesync_read_rx_timestamp,
	.timesync_read_tx_timestamp = bnxt_timesync_read_tx_timestamp,
};

static uint32_t bnxt_map_reset_regs(struct bnxt *bp, uint32_t reg)
{
	uint32_t offset;

	/* Only pre-map the reset GRC registers using window 3 */
	rte_write32(reg & 0xfffff000, (uint8_t *)bp->bar0 +
		    BNXT_GRCPF_REG_WINDOW_BASE_OUT + 8);

	offset = BNXT_GRCP_WINDOW_3_BASE + (reg & 0xffc);

	return offset;
}

int bnxt_map_fw_health_status_regs(struct bnxt *bp)
{
	struct bnxt_error_recovery_info *info = bp->recovery_info;
	uint32_t reg_base = 0xffffffff;
	int i;

	/* Only pre-map the monitoring GRC registers using window 2 */
	for (i = 0; i < BNXT_FW_STATUS_REG_CNT; i++) {
		uint32_t reg = info->status_regs[i];

		if (BNXT_FW_STATUS_REG_TYPE(reg) != BNXT_FW_STATUS_REG_TYPE_GRC)
			continue;

		if (reg_base == 0xffffffff)
			reg_base = reg & 0xfffff000;
		if ((reg & 0xfffff000) != reg_base)
			return -ERANGE;

		/* Use mask 0xffc as the Lower 2 bits indicates
		 * address space location
		 */
		info->mapped_status_regs[i] = BNXT_GRCP_WINDOW_2_BASE +
						(reg & 0xffc);
	}

	if (reg_base == 0xffffffff)
		return 0;

	rte_write32(reg_base, (uint8_t *)bp->bar0 +
		    BNXT_GRCPF_REG_WINDOW_BASE_OUT + 4);

	return 0;
}

static void bnxt_write_fw_reset_reg(struct bnxt *bp, uint32_t index)
{
	struct bnxt_error_recovery_info *info = bp->recovery_info;
	uint32_t delay = info->delay_after_reset[index];
	uint32_t val = info->reset_reg_val[index];
	uint32_t reg = info->reset_reg[index];
	uint32_t type, offset;

	type = BNXT_FW_STATUS_REG_TYPE(reg);
	offset = BNXT_FW_STATUS_REG_OFF(reg);

	switch (type) {
	case BNXT_FW_STATUS_REG_TYPE_CFG:
		rte_pci_write_config(bp->pdev, &val, sizeof(val), offset);
		break;
	case BNXT_FW_STATUS_REG_TYPE_GRC:
		offset = bnxt_map_reset_regs(bp, offset);
		rte_write32(val, (uint8_t *)bp->bar0 + offset);
		break;
	case BNXT_FW_STATUS_REG_TYPE_BAR0:
		rte_write32(val, (uint8_t *)bp->bar0 + offset);
		break;
	}
	/* wait on a specific interval of time until core reset is complete */
	if (delay)
		rte_delay_ms(delay);
}

static void bnxt_dev_cleanup(struct bnxt *bp)
{
	bp->eth_dev->data->dev_link.link_status = 0;
	bp->link_info->link_up = 0;
	if (bp->eth_dev->data->dev_started)
		bnxt_dev_stop_op(bp->eth_dev);

	bnxt_uninit_resources(bp, true);
}

static int bnxt_restore_vlan_filters(struct bnxt *bp)
{
	struct rte_eth_dev *dev = bp->eth_dev;
	struct rte_vlan_filter_conf *vfc;
	int vidx, vbit, rc;
	uint16_t vlan_id;

	for (vlan_id = 1; vlan_id <= RTE_ETHER_MAX_VLAN_ID; vlan_id++) {
		vfc = &dev->data->vlan_filter_conf;
		vidx = vlan_id / 64;
		vbit = vlan_id % 64;

		/* Each bit corresponds to a VLAN id */
		if (vfc->ids[vidx] & (UINT64_C(1) << vbit)) {
			rc = bnxt_add_vlan_filter(bp, vlan_id);
			if (rc)
				return rc;
		}
	}

	return 0;
}

static int bnxt_restore_mac_filters(struct bnxt *bp)
{
	struct rte_eth_dev *dev = bp->eth_dev;
	struct rte_eth_dev_info dev_info;
	struct rte_ether_addr *addr;
	uint64_t pool_mask;
	uint32_t pool = 0;
	uint16_t i;
	int rc;

	if (BNXT_VF(bp) && !BNXT_VF_IS_TRUSTED(bp))
		return 0;

	rc = bnxt_dev_info_get_op(dev, &dev_info);
	if (rc)
		return rc;

	/* replay MAC address configuration */
	for (i = 1; i < dev_info.max_mac_addrs; i++) {
		addr = &dev->data->mac_addrs[i];

		/* skip zero address */
		if (rte_is_zero_ether_addr(addr))
			continue;

		pool = 0;
		pool_mask = dev->data->mac_pool_sel[i];

		do {
			if (pool_mask & 1ULL) {
				rc = bnxt_mac_addr_add_op(dev, addr, i, pool);
				if (rc)
					return rc;
			}
			pool_mask >>= 1;
			pool++;
		} while (pool_mask);
	}

	return 0;
}

static int bnxt_restore_filters(struct bnxt *bp)
{
	struct rte_eth_dev *dev = bp->eth_dev;
	int ret = 0;

	if (dev->data->all_multicast) {
		ret = bnxt_allmulticast_enable_op(dev);
		if (ret)
			return ret;
	}
	if (dev->data->promislwous) {
		ret = bnxt_promislwous_enable_op(dev);
		if (ret)
			return ret;
	}

	ret = bnxt_restore_mac_filters(bp);
	if (ret)
		return ret;

	ret = bnxt_restore_vlan_filters(bp);
	/* TODO restore other filters as well */
	return ret;
}

static void bnxt_dev_recover(void *arg)
{
	struct bnxt *bp = arg;
	int timeout = bp->fw_reset_max_msecs;
	int rc = 0;

	/* Clear Error flag so that device re-init should happen */
	bp->flags &= ~BNXT_FLAG_FATAL_ERROR;

	do {
		rc = bnxt_hwrm_ver_get(bp, SHORT_HWRM_CMD_TIMEOUT);
		if (rc == 0)
			break;
		rte_delay_ms(BNXT_FW_READY_WAIT_INTERVAL);
		timeout -= BNXT_FW_READY_WAIT_INTERVAL;
	} while (rc && timeout);

	if (rc) {
		PMD_DRV_LOG(ERR, "FW is not Ready after reset\n");
		goto err;
	}

	rc = bnxt_init_resources(bp, true);
	if (rc) {
		PMD_DRV_LOG(ERR,
			    "Failed to initialize resources after reset\n");
		goto err;
	}
	/* clear reset flag as the device is initialized now */
	bp->flags &= ~BNXT_FLAG_FW_RESET;

	rc = bnxt_dev_start_op(bp->eth_dev);
	if (rc) {
		PMD_DRV_LOG(ERR, "Failed to start port after reset\n");
		goto err_start;
	}

	rc = bnxt_restore_filters(bp);
	if (rc)
		goto err_start;

	PMD_DRV_LOG(INFO, "Recovered from FW reset\n");
	return;
err_start:
	bnxt_dev_stop_op(bp->eth_dev);
err:
	bp->flags |= BNXT_FLAG_FATAL_ERROR;
	bnxt_uninit_resources(bp, false);
	PMD_DRV_LOG(ERR, "Failed to recover from FW reset\n");
}

void bnxt_dev_reset_and_resume(void *arg)
{
	struct bnxt *bp = arg;
	int rc;

	bnxt_dev_cleanup(bp);

	bnxt_wait_for_device_shutdown(bp);

	rc = rte_eal_alarm_set(US_PER_MS * bp->fw_reset_min_msecs,
			       bnxt_dev_recover, (void *)bp);
	if (rc)
		PMD_DRV_LOG(ERR, "Error setting recovery alarm");
}

uint32_t bnxt_read_fw_status_reg(struct bnxt *bp, uint32_t index)
{
	struct bnxt_error_recovery_info *info = bp->recovery_info;
	uint32_t reg = info->status_regs[index];
	uint32_t type, offset, val = 0;

	type = BNXT_FW_STATUS_REG_TYPE(reg);
	offset = BNXT_FW_STATUS_REG_OFF(reg);

	switch (type) {
	case BNXT_FW_STATUS_REG_TYPE_CFG:
		rte_pci_read_config(bp->pdev, &val, sizeof(val), offset);
		break;
	case BNXT_FW_STATUS_REG_TYPE_GRC:
		offset = info->mapped_status_regs[index];
		/* FALLTHROUGH */
	case BNXT_FW_STATUS_REG_TYPE_BAR0:
		val = rte_le_to_cpu_32(rte_read32((uint8_t *)bp->bar0 +
				       offset));
		break;
	}

	return val;
}

static int bnxt_fw_reset_all(struct bnxt *bp)
{
	struct bnxt_error_recovery_info *info = bp->recovery_info;
	uint32_t i;
	int rc = 0;

	if (info->flags & BNXT_FLAG_ERROR_RECOVERY_HOST) {
		/* Reset through master function driver */
		for (i = 0; i < info->reg_array_cnt; i++)
			bnxt_write_fw_reset_reg(bp, i);
		/* Wait for time specified by FW after triggering reset */
		rte_delay_ms(info->master_func_wait_period_after_reset);
	} else if (info->flags & BNXT_FLAG_ERROR_RECOVERY_CO_CPU) {
		/* Reset with the help of Kong processor */
		rc = bnxt_hwrm_fw_reset(bp);
		if (rc)
			PMD_DRV_LOG(ERR, "Failed to reset FW\n");
	}

	return rc;
}

static void bnxt_fw_reset_cb(void *arg)
{
	struct bnxt *bp = arg;
	struct bnxt_error_recovery_info *info = bp->recovery_info;
	int rc = 0;

	/* Only Master function can do FW reset */
	if (bnxt_is_master_func(bp) &&
	    bnxt_is_recovery_enabled(bp)) {
		rc = bnxt_fw_reset_all(bp);
		if (rc) {
			PMD_DRV_LOG(ERR, "Adapter recovery failed\n");
			return;
		}
	}

	/* if recovery method is ERROR_RECOVERY_CO_CPU, KONG will send
	 * EXCEPTION_FATAL_ASYNC event to all the functions
	 * (including MASTER FUNC). After receiving this Async, all the active
	 * drivers should treat this case as FW initiated recovery
	 */
	if (info->flags & BNXT_FLAG_ERROR_RECOVERY_HOST) {
		bp->fw_reset_min_msecs = BNXT_MIN_FW_READY_TIMEOUT;
		bp->fw_reset_max_msecs = BNXT_MAX_FW_RESET_TIMEOUT;

		/* To recover from error */
		rte_eal_alarm_set(US_PER_MS, bnxt_dev_reset_and_resume,
				  (void *)bp);
	}
}

/* Driver should poll FW heartbeat, reset_counter with the frequency
 * advertised by FW in HWRM_ERROR_RECOVERY_QCFG.
 * When the driver detects heartbeat stop or change in reset_counter,
 * it has to trigger a reset to recover from the error condition.
 * A “master PF” is the function who will have the privilege to
 * initiate the chimp reset. The master PF will be elected by the
 * firmware and will be notified through async message.
 */
static void bnxt_check_fw_health(void *arg)
{
	struct bnxt *bp = arg;
	struct bnxt_error_recovery_info *info = bp->recovery_info;
	uint32_t val = 0, wait_msec;

	if (!info || !bnxt_is_recovery_enabled(bp) ||
	    is_bnxt_in_error(bp))
		return;

	val = bnxt_read_fw_status_reg(bp, BNXT_FW_HEARTBEAT_CNT_REG);
	if (val == info->last_heart_beat)
		goto reset;

	info->last_heart_beat = val;

	val = bnxt_read_fw_status_reg(bp, BNXT_FW_RECOVERY_CNT_REG);
	if (val != info->last_reset_counter)
		goto reset;

	info->last_reset_counter = val;

	rte_eal_alarm_set(US_PER_MS * info->driver_polling_freq,
			  bnxt_check_fw_health, (void *)bp);

	return;
reset:
	/* Stop DMA to/from device */
	bp->flags |= BNXT_FLAG_FATAL_ERROR;
	bp->flags |= BNXT_FLAG_FW_RESET;

	PMD_DRV_LOG(ERR, "Detected FW dead condition\n");

	if (bnxt_is_master_func(bp))
		wait_msec = info->master_func_wait_period;
	else
		wait_msec = info->normal_func_wait_period;

	rte_eal_alarm_set(US_PER_MS * wait_msec,
			  bnxt_fw_reset_cb, (void *)bp);
}

void bnxt_schedule_fw_health_check(struct bnxt *bp)
{
	uint32_t polling_freq;

	pthread_mutex_lock(&bp->health_check_lock);

	if (!bnxt_is_recovery_enabled(bp))
		goto done;

	if (bp->flags & BNXT_FLAG_FW_HEALTH_CHECK_SCHEDULED)
		goto done;

	polling_freq = bp->recovery_info->driver_polling_freq;

	rte_eal_alarm_set(US_PER_MS * polling_freq,
			  bnxt_check_fw_health, (void *)bp);
	bp->flags |= BNXT_FLAG_FW_HEALTH_CHECK_SCHEDULED;

done:
	pthread_mutex_unlock(&bp->health_check_lock);
}

static void bnxt_cancel_fw_health_check(struct bnxt *bp)
{
	if (!bnxt_is_recovery_enabled(bp))
		return;

	rte_eal_alarm_cancel(bnxt_check_fw_health, (void *)bp);
	bp->flags &= ~BNXT_FLAG_FW_HEALTH_CHECK_SCHEDULED;
}

static bool bnxt_vf_pciid(uint16_t device_id)
{
	switch (device_id) {
	case BROADCOM_DEV_ID_57304_VF:
	case BROADCOM_DEV_ID_57406_VF:
	case BROADCOM_DEV_ID_5731X_VF:
	case BROADCOM_DEV_ID_5741X_VF:
	case BROADCOM_DEV_ID_57414_VF:
	case BROADCOM_DEV_ID_STRATUS_NIC_VF1:
	case BROADCOM_DEV_ID_STRATUS_NIC_VF2:
	case BROADCOM_DEV_ID_58802_VF:
	case BROADCOM_DEV_ID_57500_VF1:
	case BROADCOM_DEV_ID_57500_VF2:
		/* FALLTHROUGH */
		return true;
	default:
		return false;
	}
}

static bool bnxt_thor_device(uint16_t device_id)
{
	switch (device_id) {
	case BROADCOM_DEV_ID_57508:
	case BROADCOM_DEV_ID_57504:
	case BROADCOM_DEV_ID_57502:
	case BROADCOM_DEV_ID_57508_MF1:
	case BROADCOM_DEV_ID_57504_MF1:
	case BROADCOM_DEV_ID_57502_MF1:
	case BROADCOM_DEV_ID_57508_MF2:
	case BROADCOM_DEV_ID_57504_MF2:
	case BROADCOM_DEV_ID_57502_MF2:
	case BROADCOM_DEV_ID_57500_VF1:
	case BROADCOM_DEV_ID_57500_VF2:
		/* FALLTHROUGH */
		return true;
	default:
		return false;
	}
}

bool bnxt_stratus_device(struct bnxt *bp)
{
	uint16_t device_id = bp->pdev->id.device_id;

	switch (device_id) {
	case BROADCOM_DEV_ID_STRATUS_NIC:
	case BROADCOM_DEV_ID_STRATUS_NIC_VF1:
	case BROADCOM_DEV_ID_STRATUS_NIC_VF2:
		/* FALLTHROUGH */
		return true;
	default:
		return false;
	}
}

static int bnxt_init_board(struct rte_eth_dev *eth_dev)
{
	struct rte_pci_device *pci_dev = RTE_ETH_DEV_TO_PCI(eth_dev);
	struct bnxt *bp = eth_dev->data->dev_private;

	/* enable device (incl. PCI PM wakeup), and bus-mastering */
	bp->bar0 = (void *)pci_dev->mem_resource[0].addr;
	bp->doorbell_base = (void *)pci_dev->mem_resource[2].addr;
	if (!bp->bar0 || !bp->doorbell_base) {
		PMD_DRV_LOG(ERR, "Unable to access Hardware\n");
		return -ENODEV;
	}

	bp->eth_dev = eth_dev;
	bp->pdev = pci_dev;

	return 0;
}

static int bnxt_alloc_ctx_mem_blk(struct bnxt *bp,
				  struct bnxt_ctx_pg_info *ctx_pg,
				  uint32_t mem_size,
				  const char *suffix,
				  uint16_t idx)
{
	struct bnxt_ring_mem_info *rmem = &ctx_pg->ring_mem;
	const struct rte_memzone *mz = NULL;
	char mz_name[RTE_MEMZONE_NAMESIZE];
	rte_iova_t mz_phys_addr;
	uint64_t valid_bits = 0;
	uint32_t sz;
	int i;

	if (!mem_size)
		return 0;

	rmem->nr_pages = RTE_ALIGN_MUL_CEIL(mem_size, BNXT_PAGE_SIZE) /
			 BNXT_PAGE_SIZE;
	rmem->page_size = BNXT_PAGE_SIZE;
	rmem->pg_arr = ctx_pg->ctx_pg_arr;
	rmem->dma_arr = ctx_pg->ctx_dma_arr;
	rmem->flags = BNXT_RMEM_VALID_PTE_FLAG;

	valid_bits = PTU_PTE_VALID;

	if (rmem->nr_pages > 1) {
		snprintf(mz_name, RTE_MEMZONE_NAMESIZE,
			 "bnxt_ctx_pg_tbl%s_%x_%d",
			 suffix, idx, bp->eth_dev->data->port_id);
		mz_name[RTE_MEMZONE_NAMESIZE - 1] = 0;
		mz = rte_memzone_lookup(mz_name);
		if (!mz) {
			mz = rte_memzone_reserve_aligned(mz_name,
						rmem->nr_pages * 8,
						SOCKET_ID_ANY,
						RTE_MEMZONE_2MB |
						RTE_MEMZONE_SIZE_HINT_ONLY |
						RTE_MEMZONE_IOVA_CONTIG,
						BNXT_PAGE_SIZE);
			if (mz == NULL)
				return -ENOMEM;
		}

		memset(mz->addr, 0, mz->len);
		mz_phys_addr = mz->iova;

		rmem->pg_tbl = mz->addr;
		rmem->pg_tbl_map = mz_phys_addr;
		rmem->pg_tbl_mz = mz;
	}

	snprintf(mz_name, RTE_MEMZONE_NAMESIZE, "bnxt_ctx_%s_%x_%d",
		 suffix, idx, bp->eth_dev->data->port_id);
	mz = rte_memzone_lookup(mz_name);
	if (!mz) {
		mz = rte_memzone_reserve_aligned(mz_name,
						 mem_size,
						 SOCKET_ID_ANY,
						 RTE_MEMZONE_1GB |
						 RTE_MEMZONE_SIZE_HINT_ONLY |
						 RTE_MEMZONE_IOVA_CONTIG,
						 BNXT_PAGE_SIZE);
		if (mz == NULL)
			return -ENOMEM;
	}

	memset(mz->addr, 0, mz->len);
	mz_phys_addr = mz->iova;

	for (sz = 0, i = 0; sz < mem_size; sz += BNXT_PAGE_SIZE, i++) {
		rmem->pg_arr[i] = ((char *)mz->addr) + sz;
		rmem->dma_arr[i] = mz_phys_addr + sz;

		if (rmem->nr_pages > 1) {
			if (i == rmem->nr_pages - 2 &&
			    (rmem->flags & BNXT_RMEM_RING_PTE_FLAG))
				valid_bits |= PTU_PTE_NEXT_TO_LAST;
			else if (i == rmem->nr_pages - 1 &&
				 (rmem->flags & BNXT_RMEM_RING_PTE_FLAG))
				valid_bits |= PTU_PTE_LAST;

			rmem->pg_tbl[i] = rte_cpu_to_le_64(rmem->dma_arr[i] |
							   valid_bits);
		}
	}

	rmem->mz = mz;
	if (rmem->vmem_size)
		rmem->vmem = (void **)mz->addr;
	rmem->dma_arr[0] = mz_phys_addr;
	return 0;
}

static void bnxt_free_ctx_mem(struct bnxt *bp)
{
	int i;

	if (!bp->ctx || !(bp->ctx->flags & BNXT_CTX_FLAG_INITED))
		return;

	bp->ctx->flags &= ~BNXT_CTX_FLAG_INITED;
	rte_memzone_free(bp->ctx->qp_mem.ring_mem.mz);
	rte_memzone_free(bp->ctx->srq_mem.ring_mem.mz);
	rte_memzone_free(bp->ctx->cq_mem.ring_mem.mz);
	rte_memzone_free(bp->ctx->vnic_mem.ring_mem.mz);
	rte_memzone_free(bp->ctx->stat_mem.ring_mem.mz);
	rte_memzone_free(bp->ctx->qp_mem.ring_mem.pg_tbl_mz);
	rte_memzone_free(bp->ctx->srq_mem.ring_mem.pg_tbl_mz);
	rte_memzone_free(bp->ctx->cq_mem.ring_mem.pg_tbl_mz);
	rte_memzone_free(bp->ctx->vnic_mem.ring_mem.pg_tbl_mz);
	rte_memzone_free(bp->ctx->stat_mem.ring_mem.pg_tbl_mz);

	for (i = 0; i < bp->ctx->tqm_fp_rings_count + 1; i++) {
		if (bp->ctx->tqm_mem[i])
			rte_memzone_free(bp->ctx->tqm_mem[i]->ring_mem.mz);
	}

	rte_free(bp->ctx);
	bp->ctx = NULL;
}

#define bnxt_roundup(x, y)   ((((x) + ((y) - 1)) / (y)) * (y))

#define min_t(type, x, y) ({                    \
	type __min1 = (x);                      \
	type __min2 = (y);                      \
	__min1 < __min2 ? __min1 : __min2; })

#define max_t(type, x, y) ({                    \
	type __max1 = (x);                      \
	type __max2 = (y);                      \
	__max1 > __max2 ? __max1 : __max2; })

#define clamp_t(type, _x, min, max)     min_t(type, max_t(type, _x, min), max)

int bnxt_alloc_ctx_mem(struct bnxt *bp)
{
	struct bnxt_ctx_pg_info *ctx_pg;
	struct bnxt_ctx_mem_info *ctx;
	uint32_t mem_size, ena, entries;
	uint32_t entries_sp, min;
	int i, rc;

	rc = bnxt_hwrm_func_backing_store_qcaps(bp);
	if (rc) {
		PMD_DRV_LOG(ERR, "Query context mem capability failed\n");
		return rc;
	}
	ctx = bp->ctx;
	if (!ctx || (ctx->flags & BNXT_CTX_FLAG_INITED))
		return 0;

	ctx_pg = &ctx->qp_mem;
	ctx_pg->entries = ctx->qp_min_qp1_entries + ctx->qp_max_l2_entries;
	mem_size = ctx->qp_entry_size * ctx_pg->entries;
	rc = bnxt_alloc_ctx_mem_blk(bp, ctx_pg, mem_size, "qp_mem", 0);
	if (rc)
		return rc;

	ctx_pg = &ctx->srq_mem;
	ctx_pg->entries = ctx->srq_max_l2_entries;
	mem_size = ctx->srq_entry_size * ctx_pg->entries;
	rc = bnxt_alloc_ctx_mem_blk(bp, ctx_pg, mem_size, "srq_mem", 0);
	if (rc)
		return rc;

	ctx_pg = &ctx->cq_mem;
	ctx_pg->entries = ctx->cq_max_l2_entries;
	mem_size = ctx->cq_entry_size * ctx_pg->entries;
	rc = bnxt_alloc_ctx_mem_blk(bp, ctx_pg, mem_size, "cq_mem", 0);
	if (rc)
		return rc;

	ctx_pg = &ctx->vnic_mem;
	ctx_pg->entries = ctx->vnic_max_vnic_entries +
		ctx->vnic_max_ring_table_entries;
	mem_size = ctx->vnic_entry_size * ctx_pg->entries;
	rc = bnxt_alloc_ctx_mem_blk(bp, ctx_pg, mem_size, "vnic_mem", 0);
	if (rc)
		return rc;

	ctx_pg = &ctx->stat_mem;
	ctx_pg->entries = ctx->stat_max_entries;
	mem_size = ctx->stat_entry_size * ctx_pg->entries;
	rc = bnxt_alloc_ctx_mem_blk(bp, ctx_pg, mem_size, "stat_mem", 0);
	if (rc)
		return rc;

	min = ctx->tqm_min_entries_per_ring;

	entries_sp = ctx->qp_max_l2_entries +
		     ctx->vnic_max_vnic_entries +
		     2 * ctx->qp_min_qp1_entries + min;
	entries_sp = bnxt_roundup(entries_sp, ctx->tqm_entries_multiple);

	entries = ctx->qp_max_l2_entries + ctx->qp_min_qp1_entries;
	entries = bnxt_roundup(entries, ctx->tqm_entries_multiple);
	entries = clamp_t(uint32_t, entries, min,
			  ctx->tqm_max_entries_per_ring);
	for (i = 0, ena = 0; i < ctx->tqm_fp_rings_count + 1; i++) {
		ctx_pg = ctx->tqm_mem[i];
		ctx_pg->entries = i ? entries : entries_sp;
		mem_size = ctx->tqm_entry_size * ctx_pg->entries;
		rc = bnxt_alloc_ctx_mem_blk(bp, ctx_pg, mem_size, "tqm_mem", i);
		if (rc)
			return rc;
		ena |= HWRM_FUNC_BACKING_STORE_CFG_INPUT_ENABLES_TQM_SP << i;
	}

	ena |= FUNC_BACKING_STORE_CFG_INPUT_DFLT_ENABLES;
	rc = bnxt_hwrm_func_backing_store_cfg(bp, ena);
	if (rc)
		PMD_DRV_LOG(ERR,
			    "Failed to configure context mem: rc = %d\n", rc);
	else
		ctx->flags |= BNXT_CTX_FLAG_INITED;

	return rc;
}

static int bnxt_alloc_stats_mem(struct bnxt *bp)
{
	struct rte_pci_device *pci_dev = bp->pdev;
	char mz_name[RTE_MEMZONE_NAMESIZE];
	const struct rte_memzone *mz = NULL;
	uint32_t total_alloc_len;
	rte_iova_t mz_phys_addr;

	if (pci_dev->id.device_id == BROADCOM_DEV_ID_NS2)
		return 0;

	snprintf(mz_name, RTE_MEMZONE_NAMESIZE,
		 "bnxt_" PCI_PRI_FMT "-%s", pci_dev->addr.domain,
		 pci_dev->addr.bus, pci_dev->addr.devid,
		 pci_dev->addr.function, "rx_port_stats");
	mz_name[RTE_MEMZONE_NAMESIZE - 1] = 0;
	mz = rte_memzone_lookup(mz_name);
	total_alloc_len =
		RTE_CACHE_LINE_ROUNDUP(sizeof(struct rx_port_stats) +
				       sizeof(struct rx_port_stats_ext) + 512);
	if (!mz) {
		mz = rte_memzone_reserve(mz_name, total_alloc_len,
					 SOCKET_ID_ANY,
					 RTE_MEMZONE_2MB |
					 RTE_MEMZONE_SIZE_HINT_ONLY |
					 RTE_MEMZONE_IOVA_CONTIG);
		if (mz == NULL)
			return -ENOMEM;
	}
	memset(mz->addr, 0, mz->len);
	mz_phys_addr = mz->iova;

	bp->rx_mem_zone = (const void *)mz;
	bp->hw_rx_port_stats = mz->addr;
	bp->hw_rx_port_stats_map = mz_phys_addr;

	snprintf(mz_name, RTE_MEMZONE_NAMESIZE,
		 "bnxt_" PCI_PRI_FMT "-%s", pci_dev->addr.domain,
		 pci_dev->addr.bus, pci_dev->addr.devid,
		 pci_dev->addr.function, "tx_port_stats");
	mz_name[RTE_MEMZONE_NAMESIZE - 1] = 0;
	mz = rte_memzone_lookup(mz_name);
	total_alloc_len =
		RTE_CACHE_LINE_ROUNDUP(sizeof(struct tx_port_stats) +
				       sizeof(struct tx_port_stats_ext) + 512);
	if (!mz) {
		mz = rte_memzone_reserve(mz_name,
					 total_alloc_len,
					 SOCKET_ID_ANY,
					 RTE_MEMZONE_2MB |
					 RTE_MEMZONE_SIZE_HINT_ONLY |
					 RTE_MEMZONE_IOVA_CONTIG);
		if (mz == NULL)
			return -ENOMEM;
	}
	memset(mz->addr, 0, mz->len);
	mz_phys_addr = mz->iova;

	bp->tx_mem_zone = (const void *)mz;
	bp->hw_tx_port_stats = mz->addr;
	bp->hw_tx_port_stats_map = mz_phys_addr;
	bp->flags |= BNXT_FLAG_PORT_STATS;

	/* Display extended statistics if FW supports it */
	if (bp->hwrm_spec_code < HWRM_SPEC_CODE_1_8_4 ||
	    bp->hwrm_spec_code == HWRM_SPEC_CODE_1_9_0 ||
	    !(bp->flags & BNXT_FLAG_EXT_STATS_SUPPORTED))
		return 0;

	bp->hw_rx_port_stats_ext = (void *)
		((uint8_t *)bp->hw_rx_port_stats +
		 sizeof(struct rx_port_stats));
	bp->hw_rx_port_stats_ext_map = bp->hw_rx_port_stats_map +
		sizeof(struct rx_port_stats);
	bp->flags |= BNXT_FLAG_EXT_RX_PORT_STATS;

	if (bp->hwrm_spec_code < HWRM_SPEC_CODE_1_9_2 ||
	    bp->flags & BNXT_FLAG_EXT_STATS_SUPPORTED) {
		bp->hw_tx_port_stats_ext = (void *)
			((uint8_t *)bp->hw_tx_port_stats +
			 sizeof(struct tx_port_stats));
		bp->hw_tx_port_stats_ext_map =
			bp->hw_tx_port_stats_map +
			sizeof(struct tx_port_stats);
		bp->flags |= BNXT_FLAG_EXT_TX_PORT_STATS;
	}

	return 0;
}

static int bnxt_setup_mac_addr(struct rte_eth_dev *eth_dev)
{
	struct bnxt *bp = eth_dev->data->dev_private;
	int rc = 0;

	eth_dev->data->mac_addrs = rte_zmalloc("bnxt_mac_addr_tbl",
					       RTE_ETHER_ADDR_LEN *
					       bp->max_l2_ctx,
					       0);
	if (eth_dev->data->mac_addrs == NULL) {
		PMD_DRV_LOG(ERR, "Failed to alloc MAC addr tbl\n");
		return -ENOMEM;
	}

	if (!BNXT_HAS_DFLT_MAC_SET(bp)) {
		if (BNXT_PF(bp))
			return -EILWAL;

		/* Generate a random MAC address, if none was assigned by PF */
		PMD_DRV_LOG(INFO, "VF MAC address not assigned by Host PF\n");
		bnxt_eth_hw_addr_random(bp->mac_addr);
		PMD_DRV_LOG(INFO,
			    "Assign random MAC:%02X:%02X:%02X:%02X:%02X:%02X\n",
			    bp->mac_addr[0], bp->mac_addr[1], bp->mac_addr[2],
			    bp->mac_addr[3], bp->mac_addr[4], bp->mac_addr[5]);

		rc = bnxt_hwrm_set_mac(bp);
		if (rc)
			return rc;
	}

	/* Copy the permanent MAC from the FUNC_QCAPS response */
	memcpy(&eth_dev->data->mac_addrs[0], bp->mac_addr, RTE_ETHER_ADDR_LEN);

	return rc;
}

static int bnxt_restore_dflt_mac(struct bnxt *bp)
{
	int rc = 0;

	/* MAC is already configured in FW */
	if (BNXT_HAS_DFLT_MAC_SET(bp))
		return 0;

	/* Restore the old MAC configured */
	rc = bnxt_hwrm_set_mac(bp);
	if (rc)
		PMD_DRV_LOG(ERR, "Failed to restore MAC address\n");

	return rc;
}

static void bnxt_config_vf_req_fwd(struct bnxt *bp)
{
	if (!BNXT_PF(bp))
		return;

	memset(bp->pf->vf_req_fwd, 0, sizeof(bp->pf->vf_req_fwd));

	if (!(bp->fw_cap & BNXT_FW_CAP_LINK_ADMIN))
		BNXT_HWRM_CMD_TO_FORWARD(HWRM_PORT_PHY_QCFG);
	BNXT_HWRM_CMD_TO_FORWARD(HWRM_FUNC_CFG);
	BNXT_HWRM_CMD_TO_FORWARD(HWRM_FUNC_VF_CFG);
	BNXT_HWRM_CMD_TO_FORWARD(HWRM_CFA_L2_FILTER_ALLOC);
	BNXT_HWRM_CMD_TO_FORWARD(HWRM_OEM_CMD);
}

uint16_t
bnxt_get_svif(uint16_t port_id, bool func_svif,
	      enum bnxt_ulp_intf_type type)
{
	struct rte_eth_dev *eth_dev;
	struct bnxt *bp;

	eth_dev = &rte_eth_devices[port_id];
	if (BNXT_ETH_DEV_IS_REPRESENTOR(eth_dev)) {
		struct bnxt_representor *vfr = eth_dev->data->dev_private;
		if (!vfr)
			return 0;

		if (type == BNXT_ULP_INTF_TYPE_VF_REP)
			return vfr->svif;

		eth_dev = vfr->parent_dev;
	}

	bp = eth_dev->data->dev_private;

	return func_svif ? bp->func_svif : bp->port_svif;
}

uint16_t
bnxt_get_vnic_id(uint16_t port, enum bnxt_ulp_intf_type type)
{
	struct rte_eth_dev *eth_dev;
	struct bnxt_vnic_info *vnic;
	struct bnxt *bp;

	eth_dev = &rte_eth_devices[port];
	if (BNXT_ETH_DEV_IS_REPRESENTOR(eth_dev)) {
		struct bnxt_representor *vfr = eth_dev->data->dev_private;
		if (!vfr)
			return 0;

		if (type == BNXT_ULP_INTF_TYPE_VF_REP)
			return vfr->dflt_vnic_id;

		eth_dev = vfr->parent_dev;
	}

	bp = eth_dev->data->dev_private;

	vnic = BNXT_GET_DEFAULT_VNIC(bp);

	return vnic->fw_vnic_id;
}

uint16_t
bnxt_get_fw_func_id(uint16_t port, enum bnxt_ulp_intf_type type)
{
	struct rte_eth_dev *eth_dev;
	struct bnxt *bp;

	eth_dev = &rte_eth_devices[port];
	if (BNXT_ETH_DEV_IS_REPRESENTOR(eth_dev)) {
		struct bnxt_representor *vfr = eth_dev->data->dev_private;
		if (!vfr)
			return 0;

		if (type == BNXT_ULP_INTF_TYPE_VF_REP)
			return vfr->fw_fid;

		eth_dev = vfr->parent_dev;
	}

	bp = eth_dev->data->dev_private;

	return bp->fw_fid;
}

enum bnxt_ulp_intf_type
bnxt_get_interface_type(uint16_t port)
{
	struct rte_eth_dev *eth_dev;
	struct bnxt *bp;

	eth_dev = &rte_eth_devices[port];
	if (BNXT_ETH_DEV_IS_REPRESENTOR(eth_dev))
		return BNXT_ULP_INTF_TYPE_VF_REP;

	bp = eth_dev->data->dev_private;
	if (BNXT_PF(bp))
		return BNXT_ULP_INTF_TYPE_PF;
	else if (BNXT_VF_IS_TRUSTED(bp))
		return BNXT_ULP_INTF_TYPE_TRUSTED_VF;
	else if (BNXT_VF(bp))
		return BNXT_ULP_INTF_TYPE_VF;

	return BNXT_ULP_INTF_TYPE_ILWALID;
}

uint16_t
bnxt_get_phy_port_id(uint16_t port_id)
{
	struct bnxt_representor *vfr;
	struct rte_eth_dev *eth_dev;
	struct bnxt *bp;

	eth_dev = &rte_eth_devices[port_id];
	if (BNXT_ETH_DEV_IS_REPRESENTOR(eth_dev)) {
		vfr = eth_dev->data->dev_private;
		if (!vfr)
			return 0;

		eth_dev = vfr->parent_dev;
	}

	bp = eth_dev->data->dev_private;

	return BNXT_PF(bp) ? bp->pf->port_id : bp->parent->port_id;
}

uint16_t
bnxt_get_parif(uint16_t port_id, enum bnxt_ulp_intf_type type)
{
	struct rte_eth_dev *eth_dev;
	struct bnxt *bp;

	eth_dev = &rte_eth_devices[port_id];
	if (BNXT_ETH_DEV_IS_REPRESENTOR(eth_dev)) {
		struct bnxt_representor *vfr = eth_dev->data->dev_private;
		if (!vfr)
			return 0;

		if (type == BNXT_ULP_INTF_TYPE_VF_REP)
			return vfr->fw_fid - 1;

		eth_dev = vfr->parent_dev;
	}

	bp = eth_dev->data->dev_private;

	return BNXT_PF(bp) ? bp->fw_fid - 1 : bp->parent->fid - 1;
}

uint16_t
bnxt_get_vport(uint16_t port_id)
{
	return (1 << bnxt_get_phy_port_id(port_id));
}

static void bnxt_alloc_error_recovery_info(struct bnxt *bp)
{
	struct bnxt_error_recovery_info *info = bp->recovery_info;

	if (info) {
		if (!(bp->fw_cap & BNXT_FW_CAP_HCOMM_FW_STATUS))
			memset(info, 0, sizeof(*info));
		return;
	}

	if (!(bp->fw_cap & BNXT_FW_CAP_ERROR_RECOVERY))
		return;

	info = rte_zmalloc("bnxt_hwrm_error_recovery_qcfg",
			   sizeof(*info), 0);
	if (!info)
		bp->fw_cap &= ~BNXT_FW_CAP_ERROR_RECOVERY;

	bp->recovery_info = info;
}

static void bnxt_check_fw_status(struct bnxt *bp)
{
	uint32_t fw_status;

	if (!(bp->recovery_info &&
	      (bp->fw_cap & BNXT_FW_CAP_HCOMM_FW_STATUS)))
		return;

	fw_status = bnxt_read_fw_status_reg(bp, BNXT_FW_STATUS_REG);
	if (fw_status != BNXT_FW_STATUS_HEALTHY)
		PMD_DRV_LOG(ERR, "Firmware not responding, status: %#x\n",
			    fw_status);
}

static int bnxt_map_hcomm_fw_status_reg(struct bnxt *bp)
{
	struct bnxt_error_recovery_info *info = bp->recovery_info;
	uint32_t status_loc;
	uint32_t sig_ver;

	rte_write32(HCOMM_STATUS_STRUCT_LOC, (uint8_t *)bp->bar0 +
		    BNXT_GRCPF_REG_WINDOW_BASE_OUT + 4);
	sig_ver = rte_le_to_cpu_32(rte_read32((uint8_t *)bp->bar0 +
				   BNXT_GRCP_WINDOW_2_BASE +
				   offsetof(struct hcomm_status,
					    sig_ver)));
	/* If the signature is absent, then FW does not support this feature */
	if ((sig_ver & HCOMM_STATUS_SIGNATURE_MASK) !=
	    HCOMM_STATUS_SIGNATURE_VAL)
		return 0;

	if (!info) {
		info = rte_zmalloc("bnxt_hwrm_error_recovery_qcfg",
				   sizeof(*info), 0);
		if (!info)
			return -ENOMEM;
		bp->recovery_info = info;
	} else {
		memset(info, 0, sizeof(*info));
	}

	status_loc = rte_le_to_cpu_32(rte_read32((uint8_t *)bp->bar0 +
				      BNXT_GRCP_WINDOW_2_BASE +
				      offsetof(struct hcomm_status,
					       fw_status_loc)));

	/* Only pre-map the FW health status GRC register */
	if (BNXT_FW_STATUS_REG_TYPE(status_loc) != BNXT_FW_STATUS_REG_TYPE_GRC)
		return 0;

	info->status_regs[BNXT_FW_STATUS_REG] = status_loc;
	info->mapped_status_regs[BNXT_FW_STATUS_REG] =
		BNXT_GRCP_WINDOW_2_BASE + (status_loc & BNXT_GRCP_OFFSET_MASK);

	rte_write32((status_loc & BNXT_GRCP_BASE_MASK), (uint8_t *)bp->bar0 +
		    BNXT_GRCPF_REG_WINDOW_BASE_OUT + 4);

	bp->fw_cap |= BNXT_FW_CAP_HCOMM_FW_STATUS;

	return 0;
}

static int bnxt_init_fw(struct bnxt *bp)
{
	uint16_t mtu;
	int rc = 0;

	bp->fw_cap = 0;

	rc = bnxt_map_hcomm_fw_status_reg(bp);
	if (rc)
		return rc;

	rc = bnxt_hwrm_ver_get(bp, DFLT_HWRM_CMD_TIMEOUT);
	if (rc) {
		bnxt_check_fw_status(bp);
		return rc;
	}

	rc = bnxt_hwrm_func_reset(bp);
	if (rc)
		return -EIO;

	rc = bnxt_hwrm_vnic_qcaps(bp);
	if (rc)
		return rc;

	rc = bnxt_hwrm_queue_qportcfg(bp);
	if (rc)
		return rc;

	/* Get the MAX capabilities for this function.
	 * This function also allocates context memory for TQM rings and
	 * informs the firmware about this allocated backing store memory.
	 */
	rc = bnxt_hwrm_func_qcaps(bp);
	if (rc)
		return rc;

	rc = bnxt_hwrm_func_qcfg(bp, &mtu);
	if (rc)
		return rc;

	bnxt_hwrm_port_mac_qcfg(bp);

	bnxt_hwrm_parent_pf_qcfg(bp);

	bnxt_hwrm_port_phy_qcaps(bp);

	bnxt_alloc_error_recovery_info(bp);
	/* Get the adapter error recovery support info */
	rc = bnxt_hwrm_error_recovery_qcfg(bp);
	if (rc)
		bp->fw_cap &= ~BNXT_FW_CAP_ERROR_RECOVERY;

	bnxt_hwrm_port_led_qcaps(bp);

	return 0;
}

static int
bnxt_init_locks(struct bnxt *bp)
{
	int err;

	err = pthread_mutex_init(&bp->flow_lock, NULL);
	if (err) {
		PMD_DRV_LOG(ERR, "Unable to initialize flow_lock\n");
		return err;
	}

	err = pthread_mutex_init(&bp->def_cp_lock, NULL);
	if (err)
		PMD_DRV_LOG(ERR, "Unable to initialize def_cp_lock\n");

	err = pthread_mutex_init(&bp->health_check_lock, NULL);
	if (err)
		PMD_DRV_LOG(ERR, "Unable to initialize health_check_lock\n");
	return err;
}

static int bnxt_init_resources(struct bnxt *bp, bool reconfig_dev)
{
	int rc = 0;

	rc = bnxt_init_fw(bp);
	if (rc)
		return rc;

	if (!reconfig_dev) {
		rc = bnxt_setup_mac_addr(bp->eth_dev);
		if (rc)
			return rc;
	} else {
		rc = bnxt_restore_dflt_mac(bp);
		if (rc)
			return rc;
	}

	bnxt_config_vf_req_fwd(bp);

	rc = bnxt_hwrm_func_driver_register(bp);
	if (rc) {
		PMD_DRV_LOG(ERR, "Failed to register driver");
		return -EBUSY;
	}

	if (BNXT_PF(bp)) {
		if (bp->pdev->max_vfs) {
			rc = bnxt_hwrm_allocate_vfs(bp, bp->pdev->max_vfs);
			if (rc) {
				PMD_DRV_LOG(ERR, "Failed to allocate VFs\n");
				return rc;
			}
		} else {
			rc = bnxt_hwrm_allocate_pf_only(bp);
			if (rc) {
				PMD_DRV_LOG(ERR,
					    "Failed to allocate PF resources");
				return rc;
			}
		}
	}

	rc = bnxt_alloc_mem(bp, reconfig_dev);
	if (rc)
		return rc;

	rc = bnxt_setup_int(bp);
	if (rc)
		return rc;

	rc = bnxt_request_int(bp);
	if (rc)
		return rc;

	rc = bnxt_init_ctx_mem(bp);
	if (rc) {
		PMD_DRV_LOG(ERR, "Failed to init adv_flow_counters\n");
		return rc;
	}

	rc = bnxt_init_locks(bp);
	if (rc)
		return rc;

	return 0;
}

static int
bnxt_parse_devarg_truflow(__rte_unused const char *key,
			  const char *value, void *opaque_arg)
{
	struct bnxt *bp = opaque_arg;
	unsigned long truflow;
	char *end = NULL;

	if (!value || !opaque_arg) {
		PMD_DRV_LOG(ERR,
			    "Invalid parameter passed to truflow devargs.\n");
		return -EILWAL;
	}

	truflow = strtoul(value, &end, 10);
	if (end == NULL || *end != '\0' ||
	    (truflow == ULONG_MAX && errno == ERANGE)) {
		PMD_DRV_LOG(ERR,
			    "Invalid parameter passed to truflow devargs.\n");
		return -EILWAL;
	}

	if (BNXT_DEVARG_TRUFLOW_ILWALID(truflow)) {
		PMD_DRV_LOG(ERR,
			    "Invalid value passed to truflow devargs.\n");
		return -EILWAL;
	}

	if (truflow) {
		bp->flags |= BNXT_FLAG_TRUFLOW_EN;
		PMD_DRV_LOG(INFO, "Host-based truflow feature enabled.\n");
	} else {
		bp->flags &= ~BNXT_FLAG_TRUFLOW_EN;
		PMD_DRV_LOG(INFO, "Host-based truflow feature disabled.\n");
	}

	return 0;
}

static int
bnxt_parse_devarg_flow_xstat(__rte_unused const char *key,
			     const char *value, void *opaque_arg)
{
	struct bnxt *bp = opaque_arg;
	unsigned long flow_xstat;
	char *end = NULL;

	if (!value || !opaque_arg) {
		PMD_DRV_LOG(ERR,
			    "Invalid parameter passed to flow_xstat devarg.\n");
		return -EILWAL;
	}

	flow_xstat = strtoul(value, &end, 10);
	if (end == NULL || *end != '\0' ||
	    (flow_xstat == ULONG_MAX && errno == ERANGE)) {
		PMD_DRV_LOG(ERR,
			    "Invalid parameter passed to flow_xstat devarg.\n");
		return -EILWAL;
	}

	if (BNXT_DEVARG_FLOW_XSTAT_ILWALID(flow_xstat)) {
		PMD_DRV_LOG(ERR,
			    "Invalid value passed to flow_xstat devarg.\n");
		return -EILWAL;
	}

	bp->flags |= BNXT_FLAG_FLOW_XSTATS_EN;
	if (BNXT_FLOW_XSTATS_EN(bp))
		PMD_DRV_LOG(INFO, "flow_xstat feature enabled.\n");

	return 0;
}

static int
bnxt_parse_devarg_max_num_kflows(__rte_unused const char *key,
					const char *value, void *opaque_arg)
{
	struct bnxt *bp = opaque_arg;
	unsigned long max_num_kflows;
	char *end = NULL;

	if (!value || !opaque_arg) {
		PMD_DRV_LOG(ERR,
			"Invalid parameter passed to max_num_kflows devarg.\n");
		return -EILWAL;
	}

	max_num_kflows = strtoul(value, &end, 10);
	if (end == NULL || *end != '\0' ||
		(max_num_kflows == ULONG_MAX && errno == ERANGE)) {
		PMD_DRV_LOG(ERR,
			"Invalid parameter passed to max_num_kflows devarg.\n");
		return -EILWAL;
	}

	if (bnxt_devarg_max_num_kflow_ilwalid(max_num_kflows)) {
		PMD_DRV_LOG(ERR,
			"Invalid value passed to max_num_kflows devarg.\n");
		return -EILWAL;
	}

	bp->max_num_kflows = max_num_kflows;
	if (bp->max_num_kflows)
		PMD_DRV_LOG(INFO, "max_num_kflows set as %ldK.\n",
				max_num_kflows);

	return 0;
}

static int
bnxt_parse_devarg_rep_is_pf(__rte_unused const char *key,
			    const char *value, void *opaque_arg)
{
	struct bnxt_representor *vfr_bp = opaque_arg;
	unsigned long rep_is_pf;
	char *end = NULL;

	if (!value || !opaque_arg) {
		PMD_DRV_LOG(ERR,
			    "Invalid parameter passed to rep_is_pf devargs.\n");
		return -EILWAL;
	}

	rep_is_pf = strtoul(value, &end, 10);
	if (end == NULL || *end != '\0' ||
	    (rep_is_pf == ULONG_MAX && errno == ERANGE)) {
		PMD_DRV_LOG(ERR,
			    "Invalid parameter passed to rep_is_pf devargs.\n");
		return -EILWAL;
	}

	if (BNXT_DEVARG_REP_IS_PF_ILWALID(rep_is_pf)) {
		PMD_DRV_LOG(ERR,
			    "Invalid value passed to rep_is_pf devargs.\n");
		return -EILWAL;
	}

	vfr_bp->flags |= rep_is_pf;
	if (BNXT_REP_PF(vfr_bp))
		PMD_DRV_LOG(INFO, "PF representor\n");
	else
		PMD_DRV_LOG(INFO, "VF representor\n");

	return 0;
}

static int
bnxt_parse_devarg_rep_based_pf(__rte_unused const char *key,
			       const char *value, void *opaque_arg)
{
	struct bnxt_representor *vfr_bp = opaque_arg;
	unsigned long rep_based_pf;
	char *end = NULL;

	if (!value || !opaque_arg) {
		PMD_DRV_LOG(ERR,
			    "Invalid parameter passed to rep_based_pf "
			    "devargs.\n");
		return -EILWAL;
	}

	rep_based_pf = strtoul(value, &end, 10);
	if (end == NULL || *end != '\0' ||
	    (rep_based_pf == ULONG_MAX && errno == ERANGE)) {
		PMD_DRV_LOG(ERR,
			    "Invalid parameter passed to rep_based_pf "
			    "devargs.\n");
		return -EILWAL;
	}

	if (BNXT_DEVARG_REP_BASED_PF_ILWALID(rep_based_pf)) {
		PMD_DRV_LOG(ERR,
			    "Invalid value passed to rep_based_pf devargs.\n");
		return -EILWAL;
	}

	vfr_bp->rep_based_pf = rep_based_pf;
	vfr_bp->flags |= BNXT_REP_BASED_PF_VALID;

	PMD_DRV_LOG(INFO, "rep-based-pf = %d\n", vfr_bp->rep_based_pf);

	return 0;
}

static int
bnxt_parse_devarg_rep_q_r2f(__rte_unused const char *key,
			    const char *value, void *opaque_arg)
{
	struct bnxt_representor *vfr_bp = opaque_arg;
	unsigned long rep_q_r2f;
	char *end = NULL;

	if (!value || !opaque_arg) {
		PMD_DRV_LOG(ERR,
			    "Invalid parameter passed to rep_q_r2f "
			    "devargs.\n");
		return -EILWAL;
	}

	rep_q_r2f = strtoul(value, &end, 10);
	if (end == NULL || *end != '\0' ||
	    (rep_q_r2f == ULONG_MAX && errno == ERANGE)) {
		PMD_DRV_LOG(ERR,
			    "Invalid parameter passed to rep_q_r2f "
			    "devargs.\n");
		return -EILWAL;
	}

	if (BNXT_DEVARG_REP_Q_R2F_ILWALID(rep_q_r2f)) {
		PMD_DRV_LOG(ERR,
			    "Invalid value passed to rep_q_r2f devargs.\n");
		return -EILWAL;
	}

	vfr_bp->rep_q_r2f = rep_q_r2f;
	vfr_bp->flags |= BNXT_REP_Q_R2F_VALID;
	PMD_DRV_LOG(INFO, "rep-q-r2f = %d\n", vfr_bp->rep_q_r2f);

	return 0;
}

static int
bnxt_parse_devarg_rep_q_f2r(__rte_unused const char *key,
			    const char *value, void *opaque_arg)
{
	struct bnxt_representor *vfr_bp = opaque_arg;
	unsigned long rep_q_f2r;
	char *end = NULL;

	if (!value || !opaque_arg) {
		PMD_DRV_LOG(ERR,
			    "Invalid parameter passed to rep_q_f2r "
			    "devargs.\n");
		return -EILWAL;
	}

	rep_q_f2r = strtoul(value, &end, 10);
	if (end == NULL || *end != '\0' ||
	    (rep_q_f2r == ULONG_MAX && errno == ERANGE)) {
		PMD_DRV_LOG(ERR,
			    "Invalid parameter passed to rep_q_f2r "
			    "devargs.\n");
		return -EILWAL;
	}

	if (BNXT_DEVARG_REP_Q_F2R_ILWALID(rep_q_f2r)) {
		PMD_DRV_LOG(ERR,
			    "Invalid value passed to rep_q_f2r devargs.\n");
		return -EILWAL;
	}

	vfr_bp->rep_q_f2r = rep_q_f2r;
	vfr_bp->flags |= BNXT_REP_Q_F2R_VALID;
	PMD_DRV_LOG(INFO, "rep-q-f2r = %d\n", vfr_bp->rep_q_f2r);

	return 0;
}

static int
bnxt_parse_devarg_rep_fc_r2f(__rte_unused const char *key,
			     const char *value, void *opaque_arg)
{
	struct bnxt_representor *vfr_bp = opaque_arg;
	unsigned long rep_fc_r2f;
	char *end = NULL;

	if (!value || !opaque_arg) {
		PMD_DRV_LOG(ERR,
			    "Invalid parameter passed to rep_fc_r2f "
			    "devargs.\n");
		return -EILWAL;
	}

	rep_fc_r2f = strtoul(value, &end, 10);
	if (end == NULL || *end != '\0' ||
	    (rep_fc_r2f == ULONG_MAX && errno == ERANGE)) {
		PMD_DRV_LOG(ERR,
			    "Invalid parameter passed to rep_fc_r2f "
			    "devargs.\n");
		return -EILWAL;
	}

	if (BNXT_DEVARG_REP_FC_R2F_ILWALID(rep_fc_r2f)) {
		PMD_DRV_LOG(ERR,
			    "Invalid value passed to rep_fc_r2f devargs.\n");
		return -EILWAL;
	}

	vfr_bp->flags |= BNXT_REP_FC_R2F_VALID;
	vfr_bp->rep_fc_r2f = rep_fc_r2f;
	PMD_DRV_LOG(INFO, "rep-fc-r2f = %lu\n", rep_fc_r2f);

	return 0;
}

static int
bnxt_parse_devarg_rep_fc_f2r(__rte_unused const char *key,
			     const char *value, void *opaque_arg)
{
	struct bnxt_representor *vfr_bp = opaque_arg;
	unsigned long rep_fc_f2r;
	char *end = NULL;

	if (!value || !opaque_arg) {
		PMD_DRV_LOG(ERR,
			    "Invalid parameter passed to rep_fc_f2r "
			    "devargs.\n");
		return -EILWAL;
	}

	rep_fc_f2r = strtoul(value, &end, 10);
	if (end == NULL || *end != '\0' ||
	    (rep_fc_f2r == ULONG_MAX && errno == ERANGE)) {
		PMD_DRV_LOG(ERR,
			    "Invalid parameter passed to rep_fc_f2r "
			    "devargs.\n");
		return -EILWAL;
	}

	if (BNXT_DEVARG_REP_FC_F2R_ILWALID(rep_fc_f2r)) {
		PMD_DRV_LOG(ERR,
			    "Invalid value passed to rep_fc_f2r devargs.\n");
		return -EILWAL;
	}

	vfr_bp->flags |= BNXT_REP_FC_F2R_VALID;
	vfr_bp->rep_fc_f2r = rep_fc_f2r;
	PMD_DRV_LOG(INFO, "rep-fc-f2r = %lu\n", rep_fc_f2r);

	return 0;
}

static void
bnxt_parse_dev_args(struct bnxt *bp, struct rte_devargs *devargs)
{
	struct rte_kvargs *kvlist;

	if (devargs == NULL)
		return;

	kvlist = rte_kvargs_parse(devargs->args, bnxt_dev_args);
	if (kvlist == NULL)
		return;

	/*
	 * Handler for "truflow" devarg.
	 * Ilwoked as for ex: "-a 0000:00:0d.0,host-based-truflow=1"
	 */
	rte_kvargs_process(kvlist, BNXT_DEVARG_TRUFLOW,
			   bnxt_parse_devarg_truflow, bp);

	/*
	 * Handler for "flow_xstat" devarg.
	 * Ilwoked as for ex: "-a 0000:00:0d.0,flow_xstat=1"
	 */
	rte_kvargs_process(kvlist, BNXT_DEVARG_FLOW_XSTAT,
			   bnxt_parse_devarg_flow_xstat, bp);

	/*
	 * Handler for "max_num_kflows" devarg.
	 * Ilwoked as for ex: "-a 000:00:0d.0,max_num_kflows=32"
	 */
	rte_kvargs_process(kvlist, BNXT_DEVARG_MAX_NUM_KFLOWS,
			   bnxt_parse_devarg_max_num_kflows, bp);

	rte_kvargs_free(kvlist);
}

static int bnxt_alloc_switch_domain(struct bnxt *bp)
{
	int rc = 0;

	if (BNXT_PF(bp) || BNXT_VF_IS_TRUSTED(bp)) {
		rc = rte_eth_switch_domain_alloc(&bp->switch_domain_id);
		if (rc)
			PMD_DRV_LOG(ERR,
				    "Failed to alloc switch domain: %d\n", rc);
		else
			PMD_DRV_LOG(INFO,
				    "Switch domain allocated %d\n",
				    bp->switch_domain_id);
	}

	return rc;
}

static int
bnxt_dev_init(struct rte_eth_dev *eth_dev, void *params __rte_unused)
{
	struct rte_pci_device *pci_dev = RTE_ETH_DEV_TO_PCI(eth_dev);
	static int version_printed;
	struct bnxt *bp;
	int rc;

	if (version_printed++ == 0)
		PMD_DRV_LOG(INFO, "%s\n", bnxt_version);

	eth_dev->dev_ops = &bnxt_dev_ops;
	eth_dev->rx_queue_count = bnxt_rx_queue_count_op;
	eth_dev->rx_descriptor_status = bnxt_rx_descriptor_status_op;
	eth_dev->tx_descriptor_status = bnxt_tx_descriptor_status_op;
	eth_dev->rx_pkt_burst = &bnxt_recv_pkts;
	eth_dev->tx_pkt_burst = &bnxt_xmit_pkts;

	/*
	 * For secondary processes, we don't initialise any further
	 * as primary has already done this work.
	 */
	if (rte_eal_process_type() != RTE_PROC_PRIMARY)
		return 0;

	rte_eth_copy_pci_info(eth_dev, pci_dev);
	eth_dev->data->dev_flags |= RTE_ETH_DEV_AUTOFILL_QUEUE_XSTATS;

	bp = eth_dev->data->dev_private;

	/* Parse dev arguments passed on when starting the DPDK application. */
	bnxt_parse_dev_args(bp, pci_dev->device.devargs);

	bp->flags &= ~BNXT_FLAG_RX_VECTOR_PKT_MODE;

	if (bnxt_vf_pciid(pci_dev->id.device_id))
		bp->flags |= BNXT_FLAG_VF;

	if (bnxt_thor_device(pci_dev->id.device_id))
		bp->flags |= BNXT_FLAG_THOR_CHIP;

	if (pci_dev->id.device_id == BROADCOM_DEV_ID_58802 ||
	    pci_dev->id.device_id == BROADCOM_DEV_ID_58804 ||
	    pci_dev->id.device_id == BROADCOM_DEV_ID_58808 ||
	    pci_dev->id.device_id == BROADCOM_DEV_ID_58802_VF)
		bp->flags |= BNXT_FLAG_STINGRAY;

	if (BNXT_TRUFLOW_EN(bp)) {
		/* extra mbuf field is required to store CFA code from mark */
		static const struct rte_mbuf_dynfield bnxt_cfa_code_dynfield_desc = {
			.name = RTE_PMD_BNXT_CFA_CODE_DYNFIELD_NAME,
			.size = sizeof(bnxt_cfa_code_dynfield_t),
			.align = __alignof__(bnxt_cfa_code_dynfield_t),
		};
		bnxt_cfa_code_dynfield_offset =
			rte_mbuf_dynfield_register(&bnxt_cfa_code_dynfield_desc);
		if (bnxt_cfa_code_dynfield_offset < 0) {
			PMD_DRV_LOG(ERR,
			    "Failed to register mbuf field for TruFlow mark\n");
			return -rte_errno;
		}
	}

	rc = bnxt_init_board(eth_dev);
	if (rc) {
		PMD_DRV_LOG(ERR,
			    "Failed to initialize board rc: %x\n", rc);
		return rc;
	}

	rc = bnxt_alloc_pf_info(bp);
	if (rc)
		goto error_free;

	rc = bnxt_alloc_link_info(bp);
	if (rc)
		goto error_free;

	rc = bnxt_alloc_parent_info(bp);
	if (rc)
		goto error_free;

	rc = bnxt_alloc_hwrm_resources(bp);
	if (rc) {
		PMD_DRV_LOG(ERR,
			    "Failed to allocate hwrm resource rc: %x\n", rc);
		goto error_free;
	}
	rc = bnxt_alloc_leds_info(bp);
	if (rc)
		goto error_free;

	rc = bnxt_alloc_cos_queues(bp);
	if (rc)
		goto error_free;

	rc = bnxt_init_resources(bp, false);
	if (rc)
		goto error_free;

	rc = bnxt_alloc_stats_mem(bp);
	if (rc)
		goto error_free;

	bnxt_alloc_switch_domain(bp);

	PMD_DRV_LOG(INFO,
		    DRV_MODULE_NAME "found at mem %" PRIX64 ", node addr %pM\n",
		    pci_dev->mem_resource[0].phys_addr,
		    pci_dev->mem_resource[0].addr);

	return 0;

error_free:
	bnxt_dev_uninit(eth_dev);
	return rc;
}


static void bnxt_free_ctx_mem_buf(struct bnxt_ctx_mem_buf_info *ctx)
{
	if (!ctx)
		return;

	if (ctx->va)
		rte_free(ctx->va);

	ctx->va = NULL;
	ctx->dma = RTE_BAD_IOVA;
	ctx->ctx_id = BNXT_CTX_VAL_ILWAL;
}

static void bnxt_unregister_fc_ctx_mem(struct bnxt *bp)
{
	bnxt_hwrm_cfa_counter_cfg(bp, BNXT_DIR_RX,
				  CFA_COUNTER_CFG_IN_COUNTER_TYPE_FC,
				  bp->flow_stat->rx_fc_out_tbl.ctx_id,
				  bp->flow_stat->max_fc,
				  false);

	bnxt_hwrm_cfa_counter_cfg(bp, BNXT_DIR_TX,
				  CFA_COUNTER_CFG_IN_COUNTER_TYPE_FC,
				  bp->flow_stat->tx_fc_out_tbl.ctx_id,
				  bp->flow_stat->max_fc,
				  false);

	if (bp->flow_stat->rx_fc_in_tbl.ctx_id != BNXT_CTX_VAL_ILWAL)
		bnxt_hwrm_ctx_unrgtr(bp, bp->flow_stat->rx_fc_in_tbl.ctx_id);
	bp->flow_stat->rx_fc_in_tbl.ctx_id = BNXT_CTX_VAL_ILWAL;

	if (bp->flow_stat->rx_fc_out_tbl.ctx_id != BNXT_CTX_VAL_ILWAL)
		bnxt_hwrm_ctx_unrgtr(bp, bp->flow_stat->rx_fc_out_tbl.ctx_id);
	bp->flow_stat->rx_fc_out_tbl.ctx_id = BNXT_CTX_VAL_ILWAL;

	if (bp->flow_stat->tx_fc_in_tbl.ctx_id != BNXT_CTX_VAL_ILWAL)
		bnxt_hwrm_ctx_unrgtr(bp, bp->flow_stat->tx_fc_in_tbl.ctx_id);
	bp->flow_stat->tx_fc_in_tbl.ctx_id = BNXT_CTX_VAL_ILWAL;

	if (bp->flow_stat->tx_fc_out_tbl.ctx_id != BNXT_CTX_VAL_ILWAL)
		bnxt_hwrm_ctx_unrgtr(bp, bp->flow_stat->tx_fc_out_tbl.ctx_id);
	bp->flow_stat->tx_fc_out_tbl.ctx_id = BNXT_CTX_VAL_ILWAL;
}

static void bnxt_uninit_fc_ctx_mem(struct bnxt *bp)
{
	bnxt_unregister_fc_ctx_mem(bp);

	bnxt_free_ctx_mem_buf(&bp->flow_stat->rx_fc_in_tbl);
	bnxt_free_ctx_mem_buf(&bp->flow_stat->rx_fc_out_tbl);
	bnxt_free_ctx_mem_buf(&bp->flow_stat->tx_fc_in_tbl);
	bnxt_free_ctx_mem_buf(&bp->flow_stat->tx_fc_out_tbl);
}

static void bnxt_uninit_ctx_mem(struct bnxt *bp)
{
	if (BNXT_FLOW_XSTATS_EN(bp))
		bnxt_uninit_fc_ctx_mem(bp);
}

static void
bnxt_free_error_recovery_info(struct bnxt *bp)
{
	rte_free(bp->recovery_info);
	bp->recovery_info = NULL;
	bp->fw_cap &= ~BNXT_FW_CAP_ERROR_RECOVERY;
}

static void
bnxt_uninit_locks(struct bnxt *bp)
{
	pthread_mutex_destroy(&bp->flow_lock);
	pthread_mutex_destroy(&bp->def_cp_lock);
	pthread_mutex_destroy(&bp->health_check_lock);
	if (bp->rep_info) {
		pthread_mutex_destroy(&bp->rep_info->vfr_lock);
		pthread_mutex_destroy(&bp->rep_info->vfr_start_lock);
	}
}

static int
bnxt_uninit_resources(struct bnxt *bp, bool reconfig_dev)
{
	int rc;

	bnxt_free_int(bp);
	bnxt_free_mem(bp, reconfig_dev);

	bnxt_hwrm_func_buf_unrgtr(bp);
	rte_free(bp->pf->vf_req_buf);

	rc = bnxt_hwrm_func_driver_unregister(bp, 0);
	bp->flags &= ~BNXT_FLAG_REGISTERED;
	bnxt_free_ctx_mem(bp);
	if (!reconfig_dev) {
		bnxt_free_hwrm_resources(bp);
		bnxt_free_error_recovery_info(bp);
	}

	bnxt_uninit_ctx_mem(bp);

	bnxt_uninit_locks(bp);
	bnxt_free_flow_stats_info(bp);
	bnxt_free_rep_info(bp);
	rte_free(bp->ptp_cfg);
	bp->ptp_cfg = NULL;
	return rc;
}

static int
bnxt_dev_uninit(struct rte_eth_dev *eth_dev)
{
	if (rte_eal_process_type() != RTE_PROC_PRIMARY)
		return -EPERM;

	PMD_DRV_LOG(DEBUG, "Calling Device uninit\n");

	if (eth_dev->state != RTE_ETH_DEV_UNUSED)
		bnxt_dev_close_op(eth_dev);

	return 0;
}

static int bnxt_pci_remove_dev_with_reps(struct rte_eth_dev *eth_dev)
{
	struct bnxt *bp = eth_dev->data->dev_private;
	struct rte_eth_dev *vf_rep_eth_dev;
	int ret = 0, i;

	if (!bp)
		return -EILWAL;

	for (i = 0; i < bp->num_reps; i++) {
		vf_rep_eth_dev = bp->rep_info[i].vfr_eth_dev;
		if (!vf_rep_eth_dev)
			continue;
		PMD_DRV_LOG(DEBUG, "BNXT Port:%d VFR pci remove\n",
			    vf_rep_eth_dev->data->port_id);
		rte_eth_dev_destroy(vf_rep_eth_dev, bnxt_representor_uninit);
	}
	PMD_DRV_LOG(DEBUG, "BNXT Port:%d pci remove\n",
		    eth_dev->data->port_id);
	ret = rte_eth_dev_destroy(eth_dev, bnxt_dev_uninit);

	return ret;
}

static void bnxt_free_rep_info(struct bnxt *bp)
{
	rte_free(bp->rep_info);
	bp->rep_info = NULL;
	rte_free(bp->cfa_code_map);
	bp->cfa_code_map = NULL;
}

static int bnxt_init_rep_info(struct bnxt *bp)
{
	int i = 0, rc;

	if (bp->rep_info)
		return 0;

	bp->rep_info = rte_zmalloc("bnxt_rep_info",
				   sizeof(bp->rep_info[0]) * BNXT_MAX_VF_REPS,
				   0);
	if (!bp->rep_info) {
		PMD_DRV_LOG(ERR, "Failed to alloc memory for rep info\n");
		return -ENOMEM;
	}
	bp->cfa_code_map = rte_zmalloc("bnxt_cfa_code_map",
				       sizeof(*bp->cfa_code_map) *
				       BNXT_MAX_CFA_CODE, 0);
	if (!bp->cfa_code_map) {
		PMD_DRV_LOG(ERR, "Failed to alloc memory for cfa_code_map\n");
		bnxt_free_rep_info(bp);
		return -ENOMEM;
	}

	for (i = 0; i < BNXT_MAX_CFA_CODE; i++)
		bp->cfa_code_map[i] = BNXT_VF_IDX_ILWALID;

	rc = pthread_mutex_init(&bp->rep_info->vfr_lock, NULL);
	if (rc) {
		PMD_DRV_LOG(ERR, "Unable to initialize vfr_lock\n");
		bnxt_free_rep_info(bp);
		return rc;
	}

	rc = pthread_mutex_init(&bp->rep_info->vfr_start_lock, NULL);
	if (rc) {
		PMD_DRV_LOG(ERR, "Unable to initialize vfr_start_lock\n");
		bnxt_free_rep_info(bp);
		return rc;
	}

	return rc;
}

static int bnxt_rep_port_probe(struct rte_pci_device *pci_dev,
			       struct rte_eth_devargs *eth_da,
			       struct rte_eth_dev *backing_eth_dev,
			       const char *dev_args)
{
	struct rte_eth_dev *vf_rep_eth_dev;
	char name[RTE_ETH_NAME_MAX_LEN];
	struct bnxt *backing_bp;
	uint16_t num_rep;
	int i, ret = 0;
	struct rte_kvargs *kvlist = NULL;

	num_rep = eth_da->nb_representor_ports;
	if (num_rep > BNXT_MAX_VF_REPS) {
		PMD_DRV_LOG(ERR, "nb_representor_ports = %d > %d MAX VF REPS\n",
			    num_rep, BNXT_MAX_VF_REPS);
		return -EILWAL;
	}

	if (num_rep >= RTE_MAX_ETHPORTS) {
		PMD_DRV_LOG(ERR,
			    "nb_representor_ports = %d > %d MAX ETHPORTS\n",
			    num_rep, RTE_MAX_ETHPORTS);
		return -EILWAL;
	}

	backing_bp = backing_eth_dev->data->dev_private;

	if (!(BNXT_PF(backing_bp) || BNXT_VF_IS_TRUSTED(backing_bp))) {
		PMD_DRV_LOG(ERR,
			    "Not a PF or trusted VF. No Representor support\n");
		/* Returning an error is not an option.
		 * Applications are not handling this correctly
		 */
		return 0;
	}

	if (bnxt_init_rep_info(backing_bp))
		return 0;

	for (i = 0; i < num_rep; i++) {
		struct bnxt_representor representor = {
			.vf_id = eth_da->representor_ports[i],
			.switch_domain_id = backing_bp->switch_domain_id,
			.parent_dev = backing_eth_dev
		};

		if (representor.vf_id >= BNXT_MAX_VF_REPS) {
			PMD_DRV_LOG(ERR, "VF-Rep id %d >= %d MAX VF ID\n",
				    representor.vf_id, BNXT_MAX_VF_REPS);
			continue;
		}

		/* representor port net_bdf_port */
		snprintf(name, sizeof(name), "net_%s_representor_%d",
			 pci_dev->device.name, eth_da->representor_ports[i]);

		kvlist = rte_kvargs_parse(dev_args, bnxt_dev_args);
		if (kvlist) {
			/*
			 * Handler for "rep_is_pf" devarg.
			 * Ilwoked as for ex: "-a 000:00:0d.0,
			 * rep-based-pf=<pf index> rep-is-pf=<VF=0 or PF=1>"
			 */
			ret = rte_kvargs_process(kvlist, BNXT_DEVARG_REP_IS_PF,
						 bnxt_parse_devarg_rep_is_pf,
						 (void *)&representor);
			if (ret) {
				ret = -EILWAL;
				goto err;
			}
			/*
			 * Handler for "rep_based_pf" devarg.
			 * Ilwoked as for ex: "-a 000:00:0d.0,
			 * rep-based-pf=<pf index> rep-is-pf=<VF=0 or PF=1>"
			 */
			ret = rte_kvargs_process(kvlist,
						 BNXT_DEVARG_REP_BASED_PF,
						 bnxt_parse_devarg_rep_based_pf,
						 (void *)&representor);
			if (ret) {
				ret = -EILWAL;
				goto err;
			}
			/*
			 * Handler for "rep_based_pf" devarg.
			 * Ilwoked as for ex: "-a 000:00:0d.0,
			 * rep-based-pf=<pf index> rep-is-pf=<VF=0 or PF=1>"
			 */
			ret = rte_kvargs_process(kvlist, BNXT_DEVARG_REP_Q_R2F,
						 bnxt_parse_devarg_rep_q_r2f,
						 (void *)&representor);
			if (ret) {
				ret = -EILWAL;
				goto err;
			}
			/*
			 * Handler for "rep_based_pf" devarg.
			 * Ilwoked as for ex: "-a 000:00:0d.0,
			 * rep-based-pf=<pf index> rep-is-pf=<VF=0 or PF=1>"
			 */
			ret = rte_kvargs_process(kvlist, BNXT_DEVARG_REP_Q_F2R,
						 bnxt_parse_devarg_rep_q_f2r,
						 (void *)&representor);
			if (ret) {
				ret = -EILWAL;
				goto err;
			}
			/*
			 * Handler for "rep_based_pf" devarg.
			 * Ilwoked as for ex: "-a 000:00:0d.0,
			 * rep-based-pf=<pf index> rep-is-pf=<VF=0 or PF=1>"
			 */
			ret = rte_kvargs_process(kvlist, BNXT_DEVARG_REP_FC_R2F,
						 bnxt_parse_devarg_rep_fc_r2f,
						 (void *)&representor);
			if (ret) {
				ret = -EILWAL;
				goto err;
			}
			/*
			 * Handler for "rep_based_pf" devarg.
			 * Ilwoked as for ex: "-a 000:00:0d.0,
			 * rep-based-pf=<pf index> rep-is-pf=<VF=0 or PF=1>"
			 */
			ret = rte_kvargs_process(kvlist, BNXT_DEVARG_REP_FC_F2R,
						 bnxt_parse_devarg_rep_fc_f2r,
						 (void *)&representor);
			if (ret) {
				ret = -EILWAL;
				goto err;
			}
		}

		ret = rte_eth_dev_create(&pci_dev->device, name,
					 sizeof(struct bnxt_representor),
					 NULL, NULL,
					 bnxt_representor_init,
					 &representor);
		if (ret) {
			PMD_DRV_LOG(ERR, "failed to create bnxt vf "
				    "representor %s.", name);
			goto err;
		}

		vf_rep_eth_dev = rte_eth_dev_allocated(name);
		if (!vf_rep_eth_dev) {
			PMD_DRV_LOG(ERR, "Failed to find the eth_dev"
				    " for VF-Rep: %s.", name);
			ret = -ENODEV;
			goto err;
		}

		PMD_DRV_LOG(DEBUG, "BNXT Port:%d VFR pci probe\n",
			    backing_eth_dev->data->port_id);
		backing_bp->rep_info[representor.vf_id].vfr_eth_dev =
							 vf_rep_eth_dev;
		backing_bp->num_reps++;

	}

	rte_kvargs_free(kvlist);
	return 0;

err:
	/* If num_rep > 1, then rollback already created
	 * ports, since we'll be failing the probe anyway
	 */
	if (num_rep > 1)
		bnxt_pci_remove_dev_with_reps(backing_eth_dev);
	rte_errno = -ret;
	rte_kvargs_free(kvlist);

	return ret;
}

static int bnxt_pci_probe(struct rte_pci_driver *pci_drv __rte_unused,
			  struct rte_pci_device *pci_dev)
{
	struct rte_eth_devargs eth_da = { .nb_representor_ports = 0 };
	struct rte_eth_dev *backing_eth_dev;
	uint16_t num_rep;
	int ret = 0;

	if (pci_dev->device.devargs) {
		ret = rte_eth_devargs_parse(pci_dev->device.devargs->args,
					    &eth_da);
		if (ret)
			return ret;
	}

	num_rep = eth_da.nb_representor_ports;
	PMD_DRV_LOG(DEBUG, "nb_representor_ports = %d\n",
		    num_rep);

	/* We could come here after first level of probe is already ilwoked
	 * as part of an application bringup(OVS-DPDK vswitchd), so first check
	 * for already allocated eth_dev for the backing device (PF/Trusted VF)
	 */
	backing_eth_dev = rte_eth_dev_allocated(pci_dev->device.name);
	if (backing_eth_dev == NULL) {
		ret = rte_eth_dev_create(&pci_dev->device, pci_dev->device.name,
					 sizeof(struct bnxt),
					 eth_dev_pci_specific_init, pci_dev,
					 bnxt_dev_init, NULL);

		if (ret || !num_rep)
			return ret;

		backing_eth_dev = rte_eth_dev_allocated(pci_dev->device.name);
	}
	PMD_DRV_LOG(DEBUG, "BNXT Port:%d pci probe\n",
		    backing_eth_dev->data->port_id);

	if (!num_rep)
		return ret;

	/* probe representor ports now */
	ret = bnxt_rep_port_probe(pci_dev, &eth_da, backing_eth_dev,
				  pci_dev->device.devargs->args);

	return ret;
}

static int bnxt_pci_remove(struct rte_pci_device *pci_dev)
{
	struct rte_eth_dev *eth_dev;

	eth_dev = rte_eth_dev_allocated(pci_dev->device.name);
	if (!eth_dev)
		return 0; /* Ilwoked typically only by OVS-DPDK, by the
			   * time it comes here the eth_dev is already
			   * deleted by rte_eth_dev_close(), so returning
			   * +ve value will at least help in proper cleanup
			   */

	PMD_DRV_LOG(DEBUG, "BNXT Port:%d pci remove\n", eth_dev->data->port_id);
	if (rte_eal_process_type() == RTE_PROC_PRIMARY) {
		if (eth_dev->data->dev_flags & RTE_ETH_DEV_REPRESENTOR)
			return rte_eth_dev_destroy(eth_dev,
						   bnxt_representor_uninit);
		else
			return rte_eth_dev_destroy(eth_dev,
						   bnxt_dev_uninit);
	} else {
		return rte_eth_dev_pci_generic_remove(pci_dev, NULL);
	}
}

static struct rte_pci_driver bnxt_rte_pmd = {
	.id_table = bnxt_pci_id_map,
	.drv_flags = RTE_PCI_DRV_NEED_MAPPING | RTE_PCI_DRV_INTR_LSC |
			RTE_PCI_DRV_PROBE_AGAIN, /* Needed in case of VF-REPs
						  * and OVS-DPDK
						  */
	.probe = bnxt_pci_probe,
	.remove = bnxt_pci_remove,
};

static bool
is_device_supported(struct rte_eth_dev *dev, struct rte_pci_driver *drv)
{
	if (strcmp(dev->device->driver->name, drv->driver.name))
		return false;

	return true;
}

bool is_bnxt_supported(struct rte_eth_dev *dev)
{
	return is_device_supported(dev, &bnxt_rte_pmd);
}

RTE_LOG_REGISTER(bnxt_logtype_driver, pmd.net.bnxt.driver, NOTICE);
RTE_PMD_REGISTER_PCI(net_bnxt, bnxt_rte_pmd);
RTE_PMD_REGISTER_PCI_TABLE(net_bnxt, bnxt_pci_id_map);
RTE_PMD_REGISTER_KMOD_DEP(net_bnxt, "* igb_uio | uio_pci_generic | vfio-pci");
