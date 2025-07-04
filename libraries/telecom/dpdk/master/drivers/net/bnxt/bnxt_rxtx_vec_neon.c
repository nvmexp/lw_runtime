/* SPDX-License-Identifier: BSD-3-Clause */
/* Copyright(c) 2019-2020 Broadcom All rights reserved. */

#include <inttypes.h>
#include <stdbool.h>

#include <rte_bitmap.h>
#include <rte_byteorder.h>
#include <rte_malloc.h>
#include <rte_memory.h>
#include <rte_vect.h>

#include "bnxt.h"
#include "bnxt_cpr.h"
#include "bnxt_ring.h"

#include "bnxt_txq.h"
#include "bnxt_txr.h"
#include "bnxt_rxtx_vec_common.h"

/*
 * RX Ring handling
 */

#define GET_OL_FLAGS(rss_flags, ol_idx, errors, pi, ol_flags)		       \
{									       \
	uint32_t tmp, of;						       \
									       \
	of = vgetq_lane_u32((rss_flags), (pi)) |			       \
		   bnxt_ol_flags_table[vgetq_lane_u32((ol_idx), (pi))];	       \
									       \
	tmp = vgetq_lane_u32((errors), (pi));				       \
	if (tmp)							       \
		of |= bnxt_ol_flags_err_table[tmp];			       \
	(ol_flags) = of;						       \
}

#define GET_DESC_FIELDS(rxcmp, rxcmp1, shuf_msk, ptype_idx, pkt_idx, ret)      \
{									       \
	uint32_t ptype;							       \
	uint16_t vlan_tci;						       \
	uint32x4_t r;							       \
									       \
	/* Set mbuf pkt_len, data_len, and rss_hash fields. */		       \
	r = vreinterpretq_u32_u8(vqtbl1q_u8(vreinterpretq_u8_u32(rxcmp),       \
					      (shuf_msk)));		       \
									       \
	/* Set packet type. */						       \
	ptype = bnxt_ptype_table[vgetq_lane_u32((ptype_idx), (pkt_idx))];      \
	r = vsetq_lane_u32(ptype, r, 0);				       \
									       \
	/* Set vlan_tci. */						       \
	vlan_tci = vgetq_lane_u32((rxcmp1), 1);				       \
	r = vreinterpretq_u32_u16(vsetq_lane_u16(vlan_tci,		       \
				vreinterpretq_u16_u32(r), 5));		       \
	(ret) = r;							       \
}

static void
descs_to_mbufs(uint32x4_t mm_rxcmp[4], uint32x4_t mm_rxcmp1[4],
	       uint64x2_t mb_init, struct rte_mbuf **mbuf)
{
	const uint8x16_t shuf_msk = {
		0xFF, 0xFF, 0xFF, 0xFF,    /* pkt_type (zeroes) */
		2, 3, 0xFF, 0xFF,          /* pkt_len */
		2, 3,                      /* data_len */
		0xFF, 0xFF,                /* vlan_tci (zeroes) */
		12, 13, 14, 15             /* rss hash */
	};
	const uint32x4_t flags_type_mask =
		vdupq_n_u32(RX_PKT_CMPL_FLAGS_ITYPE_MASK);
	const uint32x4_t flags2_mask1 =
		vdupq_n_u32(RX_PKT_CMPL_FLAGS2_META_FORMAT_VLAN |
			    RX_PKT_CMPL_FLAGS2_T_IP_CS_CALC);
	const uint32x4_t flags2_mask2 =
		vdupq_n_u32(RX_PKT_CMPL_FLAGS2_IP_TYPE);
	const uint32x4_t rss_mask =
		vdupq_n_u32(RX_PKT_CMPL_FLAGS_RSS_VALID);
	const uint32x4_t flags2_index_mask = vdupq_n_u32(0x1F);
	const uint32x4_t flags2_error_mask = vdupq_n_u32(0x0F);
	uint32x4_t flags_type, flags2, index, errors, rss_flags;
	uint32x4_t tmp, ptype_idx;
	uint64x2_t t0, t1;
	uint32_t ol_flags;

	/* Compute packet type table indexes for four packets */
	t0 = vreinterpretq_u64_u32(vzip1q_u32(mm_rxcmp[0], mm_rxcmp[1]));
	t1 = vreinterpretq_u64_u32(vzip1q_u32(mm_rxcmp[2], mm_rxcmp[3]));

	flags_type = vreinterpretq_u32_u64(vcombine_u64(vget_low_u64(t0),
							vget_low_u64(t1)));
	ptype_idx =
		vshrq_n_u32(vandq_u32(flags_type, flags_type_mask), 9);

	t0 = vreinterpretq_u64_u32(vzip1q_u32(mm_rxcmp1[0], mm_rxcmp1[1]));
	t1 = vreinterpretq_u64_u32(vzip1q_u32(mm_rxcmp1[2], mm_rxcmp1[3]));

	flags2 = vreinterpretq_u32_u64(vcombine_u64(vget_low_u64(t0),
						    vget_low_u64(t1)));

	ptype_idx = vorrq_u32(ptype_idx,
			vshrq_n_u32(vandq_u32(flags2, flags2_mask1), 2));
	ptype_idx = vorrq_u32(ptype_idx,
			vshrq_n_u32(vandq_u32(flags2, flags2_mask2), 7));

	/* Extract RSS valid flags for four packets. */
	rss_flags = vshrq_n_u32(vandq_u32(flags_type, rss_mask), 9);

	flags2 = vandq_u32(flags2, flags2_index_mask);

	/* Extract errors_v2 fields for four packets. */
	t0 = vreinterpretq_u64_u32(vzip2q_u32(mm_rxcmp1[0], mm_rxcmp1[1]));
	t1 = vreinterpretq_u64_u32(vzip2q_u32(mm_rxcmp1[2], mm_rxcmp1[3]));

	errors = vreinterpretq_u32_u64(vcombine_u64(vget_low_u64(t0),
						    vget_low_u64(t1)));

	/* Compute ol_flags and checksum error indexes for four packets. */
	errors = vandq_u32(vshrq_n_u32(errors, 4), flags2_error_mask);
	errors = vandq_u32(errors, flags2);

	index = vbicq_u32(flags2, errors);

	/* Update mbuf rearm_data for four packets. */
	GET_OL_FLAGS(rss_flags, index, errors, 0, ol_flags);
	vst1q_u32((uint32_t *)&mbuf[0]->rearm_data,
		  vsetq_lane_u32(ol_flags, vreinterpretq_u32_u64(mb_init), 2));
	GET_OL_FLAGS(rss_flags, index, errors, 1, ol_flags);
	vst1q_u32((uint32_t *)&mbuf[1]->rearm_data,
		  vsetq_lane_u32(ol_flags, vreinterpretq_u32_u64(mb_init), 2));
	GET_OL_FLAGS(rss_flags, index, errors, 2, ol_flags);
	vst1q_u32((uint32_t *)&mbuf[2]->rearm_data,
		  vsetq_lane_u32(ol_flags, vreinterpretq_u32_u64(mb_init), 2));
	GET_OL_FLAGS(rss_flags, index, errors, 3, ol_flags);
	vst1q_u32((uint32_t *)&mbuf[3]->rearm_data,
		  vsetq_lane_u32(ol_flags, vreinterpretq_u32_u64(mb_init), 2));

	/* Update mbuf rx_descriptor_fields1 for four packets. */
	GET_DESC_FIELDS(mm_rxcmp[0], mm_rxcmp1[0], shuf_msk, ptype_idx, 0, tmp);
	vst1q_u32((uint32_t *)&mbuf[0]->rx_descriptor_fields1, tmp);
	GET_DESC_FIELDS(mm_rxcmp[1], mm_rxcmp1[1], shuf_msk, ptype_idx, 1, tmp);
	vst1q_u32((uint32_t *)&mbuf[1]->rx_descriptor_fields1, tmp);
	GET_DESC_FIELDS(mm_rxcmp[2], mm_rxcmp1[2], shuf_msk, ptype_idx, 2, tmp);
	vst1q_u32((uint32_t *)&mbuf[2]->rx_descriptor_fields1, tmp);
	GET_DESC_FIELDS(mm_rxcmp[3], mm_rxcmp1[3], shuf_msk, ptype_idx, 3, tmp);
	vst1q_u32((uint32_t *)&mbuf[3]->rx_descriptor_fields1, tmp);
}

uint16_t
bnxt_recv_pkts_vec(void *rx_queue, struct rte_mbuf **rx_pkts,
		   uint16_t nb_pkts)
{
	struct bnxt_rx_queue *rxq = rx_queue;
	struct bnxt_cp_ring_info *cpr = rxq->cp_ring;
	struct bnxt_rx_ring_info *rxr = rxq->rx_ring;
	uint16_t cp_ring_size = cpr->cp_ring_struct->ring_size;
	uint16_t rx_ring_size = rxr->rx_ring_struct->ring_size;
	struct cmpl_base *cp_desc_ring = cpr->cp_desc_ring;
	uint64_t valid, desc_valid_mask = ~0UL;
	const uint32x4_t info3_v_mask = vdupq_n_u32(CMPL_BASE_V);
	uint32_t raw_cons = cpr->cp_raw_cons;
	uint32_t cons, mbcons;
	int nb_rx_pkts = 0;
	const uint64x2_t mb_init = {rxq->mbuf_initializer, 0};
	const uint32x4_t valid_target =
		vdupq_n_u32(!!(raw_cons & cp_ring_size));
	int i;

	/* If Rx Q was stopped return */
	if (unlikely(!rxq->rx_started))
		return 0;

	if (rxq->rxrearm_nb >= rxq->rx_free_thresh)
		bnxt_rxq_rearm(rxq, rxr);

	/* Return no more than RTE_BNXT_MAX_RX_BURST per call. */
	nb_pkts = RTE_MIN(nb_pkts, RTE_BNXT_MAX_RX_BURST);

	cons = raw_cons & (cp_ring_size - 1);
	mbcons = (raw_cons / 2) & (rx_ring_size - 1);

	/* Prefetch first four descriptor pairs. */
	rte_prefetch0(&cp_desc_ring[cons]);
	rte_prefetch0(&cp_desc_ring[cons + 4]);

	/* Ensure that we do not go past the ends of the rings. */
	nb_pkts = RTE_MIN(nb_pkts, RTE_MIN(rx_ring_size - mbcons,
					   (cp_ring_size - cons) / 2));
	/*
	 * If we are at the end of the ring, ensure that descriptors after the
	 * last valid entry are not treated as valid. Otherwise, force the
	 * maximum number of packets to receive to be a multiple of the per-
	 * loop count.
	 */
	if (nb_pkts < RTE_BNXT_DESCS_PER_LOOP)
		desc_valid_mask >>= 16 * (RTE_BNXT_DESCS_PER_LOOP - nb_pkts);
	else
		nb_pkts = RTE_ALIGN_FLOOR(nb_pkts, RTE_BNXT_DESCS_PER_LOOP);

	/* Handle RX burst request */
	for (i = 0; i < nb_pkts; i += RTE_BNXT_DESCS_PER_LOOP,
				  cons += RTE_BNXT_DESCS_PER_LOOP * 2,
				  mbcons += RTE_BNXT_DESCS_PER_LOOP) {
		uint32x4_t rxcmp1[RTE_BNXT_DESCS_PER_LOOP];
		uint32x4_t rxcmp[RTE_BNXT_DESCS_PER_LOOP];
		uint32x4_t info3_v;
		uint64x2_t t0, t1;
		uint32_t num_valid;

		/* Copy four mbuf pointers to output array. */
		t0 = vld1q_u64((void *)&rxr->rx_buf_ring[mbcons]);
#ifdef RTE_ARCH_ARM64
		t1 = vld1q_u64((void *)&rxr->rx_buf_ring[mbcons + 2]);
#endif
		vst1q_u64((void *)&rx_pkts[i], t0);
#ifdef RTE_ARCH_ARM64
		vst1q_u64((void *)&rx_pkts[i + 2], t1);
#endif

		/* Prefetch four descriptor pairs for next iteration. */
		if (i + RTE_BNXT_DESCS_PER_LOOP < nb_pkts) {
			rte_prefetch0(&cp_desc_ring[cons + 8]);
			rte_prefetch0(&cp_desc_ring[cons + 12]);
		}

		/*
		 * Load the four current descriptors into SSE registers in
		 * reverse order to ensure consistent state.
		 */
		rxcmp1[3] = vld1q_u32((void *)&cpr->cp_desc_ring[cons + 7]);
		rte_io_rmb();
		rxcmp[3] = vld1q_u32((void *)&cpr->cp_desc_ring[cons + 6]);

		rxcmp1[2] = vld1q_u32((void *)&cpr->cp_desc_ring[cons + 5]);
		rte_io_rmb();
		rxcmp[2] = vld1q_u32((void *)&cpr->cp_desc_ring[cons + 4]);

		t1 = vreinterpretq_u64_u32(vzip2q_u32(rxcmp1[2], rxcmp1[3]));

		rxcmp1[1] = vld1q_u32((void *)&cpr->cp_desc_ring[cons + 3]);
		rte_io_rmb();
		rxcmp[1] = vld1q_u32((void *)&cpr->cp_desc_ring[cons + 2]);

		rxcmp1[0] = vld1q_u32((void *)&cpr->cp_desc_ring[cons + 1]);
		rte_io_rmb();
		rxcmp[0] = vld1q_u32((void *)&cpr->cp_desc_ring[cons + 0]);

		t0 = vreinterpretq_u64_u32(vzip2q_u32(rxcmp1[0], rxcmp1[1]));

		/* Isolate descriptor status flags. */
		info3_v = vreinterpretq_u32_u64(vcombine_u64(vget_low_u64(t0),
							     vget_low_u64(t1)));
		info3_v = vandq_u32(info3_v, info3_v_mask);
		info3_v = veorq_u32(info3_v, valid_target);

		/*
		 * Pack the 128-bit array of valid descriptor flags into 64
		 * bits and count the number of set bits in order to determine
		 * the number of valid descriptors.
		 */
		valid = vget_lane_u64(vreinterpret_u64_u16(vqmovn_u32(info3_v)),
				      0);
		/*
		 * At this point, 'valid' is a 64-bit value containing four
		 * 16-bit fields, each of which is either 0x0001 or 0x0000.
		 * Compute number of valid descriptors from the index of
		 * the highest non-zero field.
		 */
		num_valid = (sizeof(uint64_t) / sizeof(uint16_t)) -
				(__builtin_clzl(valid & desc_valid_mask) / 16);

		switch (num_valid) {
		case 4:
			rxr->rx_buf_ring[mbcons + 3] = NULL;
			/* FALLTHROUGH */
		case 3:
			rxr->rx_buf_ring[mbcons + 2] = NULL;
			/* FALLTHROUGH */
		case 2:
			rxr->rx_buf_ring[mbcons + 1] = NULL;
			/* FALLTHROUGH */
		case 1:
			rxr->rx_buf_ring[mbcons + 0] = NULL;
			break;
		case 0:
			goto out;
		}

		descs_to_mbufs(rxcmp, rxcmp1, mb_init, &rx_pkts[nb_rx_pkts]);
		nb_rx_pkts += num_valid;

		if (num_valid < RTE_BNXT_DESCS_PER_LOOP)
			break;
	}

out:
	if (nb_rx_pkts) {
		rxr->rx_prod =
			RING_ADV(rxr->rx_ring_struct, rxr->rx_prod, nb_rx_pkts);

		rxq->rxrearm_nb += nb_rx_pkts;
		cpr->cp_raw_cons += 2 * nb_rx_pkts;
		cpr->valid =
			!!(cpr->cp_raw_cons & cpr->cp_ring_struct->ring_size);
		bnxt_db_cq(cpr);
	}

	return nb_rx_pkts;
}

static void
bnxt_handle_tx_cp_vec(struct bnxt_tx_queue *txq)
{
	struct bnxt_cp_ring_info *cpr = txq->cp_ring;
	uint32_t raw_cons = cpr->cp_raw_cons;
	uint32_t cons;
	uint32_t nb_tx_pkts = 0;
	struct tx_cmpl *txcmp;
	struct cmpl_base *cp_desc_ring = cpr->cp_desc_ring;
	struct bnxt_ring *cp_ring_struct = cpr->cp_ring_struct;
	uint32_t ring_mask = cp_ring_struct->ring_mask;

	do {
		cons = RING_CMPL(ring_mask, raw_cons);
		txcmp = (struct tx_cmpl *)&cp_desc_ring[cons];

		if (!CMP_VALID(txcmp, raw_cons, cp_ring_struct))
			break;

		if (likely(CMP_TYPE(txcmp) == TX_CMPL_TYPE_TX_L2))
			nb_tx_pkts += txcmp->opaque;
		else
			RTE_LOG_DP(ERR, PMD,
				   "Unhandled CMP type %02x\n",
				   CMP_TYPE(txcmp));
		raw_cons = NEXT_RAW_CMP(raw_cons);
	} while (nb_tx_pkts < ring_mask);

	cpr->valid = !!(raw_cons & cp_ring_struct->ring_size);
	if (nb_tx_pkts) {
		if (txq->offloads & DEV_TX_OFFLOAD_MBUF_FAST_FREE)
			bnxt_tx_cmp_vec_fast(txq, nb_tx_pkts);
		else
			bnxt_tx_cmp_vec(txq, nb_tx_pkts);
		cpr->cp_raw_cons = raw_cons;
		bnxt_db_cq(cpr);
	}
}

static uint16_t
bnxt_xmit_fixed_burst_vec(void *tx_queue, struct rte_mbuf **tx_pkts,
			  uint16_t nb_pkts)
{
	struct bnxt_tx_queue *txq = tx_queue;
	struct bnxt_tx_ring_info *txr = txq->tx_ring;
	uint16_t prod = txr->tx_prod;
	struct rte_mbuf *tx_mbuf;
	struct tx_bd_long *txbd = NULL;
	struct bnxt_sw_tx_bd *tx_buf;
	uint16_t to_send;

	nb_pkts = RTE_MIN(nb_pkts, bnxt_tx_avail(txq));

	if (unlikely(nb_pkts == 0))
		return 0;

	/* Handle TX burst request */
	to_send = nb_pkts;
	while (to_send) {
		tx_mbuf = *tx_pkts++;
		rte_prefetch0(tx_mbuf);

		tx_buf = &txr->tx_buf_ring[prod];
		tx_buf->mbuf = tx_mbuf;
		tx_buf->nr_bds = 1;

		txbd = &txr->tx_desc_ring[prod];
		txbd->address = tx_mbuf->buf_iova + tx_mbuf->data_off;
		txbd->len = tx_mbuf->data_len;
		txbd->flags_type = bnxt_xmit_flags_len(tx_mbuf->data_len,
						       TX_BD_FLAGS_NOCMPL);
		prod = RING_NEXT(txr->tx_ring_struct, prod);
		to_send--;
	}

	/* Request a completion for last packet in burst */
	if (txbd) {
		txbd->opaque = nb_pkts;
		txbd->flags_type &= ~TX_BD_LONG_FLAGS_NO_CMPL;
	}

	rte_compiler_barrier();
	bnxt_db_write(&txr->tx_db, prod);

	txr->tx_prod = prod;

	return nb_pkts;
}

uint16_t
bnxt_xmit_pkts_vec(void *tx_queue, struct rte_mbuf **tx_pkts,
		   uint16_t nb_pkts)
{
	int nb_sent = 0;
	struct bnxt_tx_queue *txq = tx_queue;

	/* Tx queue was stopped; wait for it to be restarted */
	if (unlikely(!txq->tx_started)) {
		PMD_DRV_LOG(DEBUG, "Tx q stopped;return\n");
		return 0;
	}

	/* Handle TX completions */
	if (bnxt_tx_bds_in_hw(txq) >= txq->tx_free_thresh)
		bnxt_handle_tx_cp_vec(txq);

	while (nb_pkts) {
		uint16_t ret, num;

		num = RTE_MIN(nb_pkts, RTE_BNXT_MAX_TX_BURST);
		ret = bnxt_xmit_fixed_burst_vec(tx_queue,
						&tx_pkts[nb_sent],
						num);
		nb_sent += ret;
		nb_pkts -= ret;
		if (ret < num)
			break;
	}

	return nb_sent;
}

int __rte_cold
bnxt_rxq_vec_setup(struct bnxt_rx_queue *rxq)
{
	return bnxt_rxq_vec_setup_common(rxq);
}
