/* SPDX-License-Identifier: BSD-3-Clause
 * Copyright(c) 2014-2018 Broadcom
 * All rights reserved.
 */

#ifndef _BNXT_TXR_H_
#define _BNXT_TXR_H_

#include <rte_io.h>

#define BNXT_MAX_TSO_SEGS	32
#define BNXT_MIN_PKT_SIZE	52

struct bnxt_tx_ring_info {
	uint16_t		tx_prod;
	uint16_t		tx_cons;
	struct bnxt_db_info     tx_db;

	struct tx_bd_long	*tx_desc_ring;
	struct bnxt_sw_tx_bd	*tx_buf_ring;

	rte_iova_t		tx_desc_mapping;

	struct bnxt_ring	*tx_ring_struct;
};

struct bnxt_sw_tx_bd {
	struct rte_mbuf		*mbuf; /* mbuf associated with TX descriptor */
	unsigned short		nr_bds;
};

static inline uint32_t bnxt_tx_bds_in_hw(struct bnxt_tx_queue *txq)
{
	return ((txq->tx_ring->tx_prod - txq->tx_ring->tx_cons) &
		txq->tx_ring->tx_ring_struct->ring_mask);
}

static inline uint32_t bnxt_tx_avail(struct bnxt_tx_queue *txq)
{
	/* Tell compiler to fetch tx indices from memory. */
	rte_compiler_barrier();

	return ((txq->tx_ring->tx_ring_struct->ring_size -
		 bnxt_tx_bds_in_hw(txq)) - 1);
}

void bnxt_free_tx_rings(struct bnxt *bp);
int bnxt_init_one_tx_ring(struct bnxt_tx_queue *txq);
int bnxt_init_tx_ring_struct(struct bnxt_tx_queue *txq, unsigned int socket_id);
uint16_t bnxt_xmit_pkts(void *tx_queue, struct rte_mbuf **tx_pkts,
			       uint16_t nb_pkts);
#if defined(RTE_ARCH_X86) || defined(RTE_ARCH_ARM64)
uint16_t bnxt_xmit_pkts_vec(void *tx_queue, struct rte_mbuf **tx_pkts,
			    uint16_t nb_pkts);
#endif

int bnxt_tx_queue_start(struct rte_eth_dev *dev, uint16_t tx_queue_id);
int bnxt_tx_queue_stop(struct rte_eth_dev *dev, uint16_t tx_queue_id);

#define PKT_TX_OIP_IIP_TCP_UDP_CKSUM	(PKT_TX_TCP_CKSUM | PKT_TX_UDP_CKSUM | \
					PKT_TX_IP_CKSUM | PKT_TX_OUTER_IP_CKSUM)
#define PKT_TX_OIP_IIP_UDP_CKSUM	(PKT_TX_UDP_CKSUM | \
					PKT_TX_IP_CKSUM | PKT_TX_OUTER_IP_CKSUM)
#define PKT_TX_OIP_IIP_TCP_CKSUM	(PKT_TX_TCP_CKSUM | \
					PKT_TX_IP_CKSUM | PKT_TX_OUTER_IP_CKSUM)
#define PKT_TX_IIP_TCP_UDP_CKSUM	(PKT_TX_TCP_CKSUM | PKT_TX_UDP_CKSUM | \
					PKT_TX_IP_CKSUM)
#define PKT_TX_IIP_TCP_CKSUM		(PKT_TX_TCP_CKSUM | PKT_TX_IP_CKSUM)
#define PKT_TX_IIP_UDP_CKSUM		(PKT_TX_UDP_CKSUM | PKT_TX_IP_CKSUM)
#define PKT_TX_OIP_TCP_UDP_CKSUM	(PKT_TX_TCP_CKSUM | PKT_TX_UDP_CKSUM | \
					PKT_TX_OUTER_IP_CKSUM)
#define PKT_TX_OIP_UDP_CKSUM		(PKT_TX_UDP_CKSUM | \
					PKT_TX_OUTER_IP_CKSUM)
#define PKT_TX_OIP_TCP_CKSUM		(PKT_TX_TCP_CKSUM | \
					PKT_TX_OUTER_IP_CKSUM)
#define PKT_TX_OIP_IIP_CKSUM		(PKT_TX_IP_CKSUM |	\
					 PKT_TX_OUTER_IP_CKSUM)
#define PKT_TX_TCP_UDP_CKSUM		(PKT_TX_TCP_CKSUM | PKT_TX_UDP_CKSUM)


#define TX_BD_FLG_TIP_IP_TCP_UDP_CHKSUM	(TX_BD_LONG_LFLAGS_TCP_UDP_CHKSUM | \
					TX_BD_LONG_LFLAGS_T_IP_CHKSUM | \
					TX_BD_LONG_LFLAGS_IP_CHKSUM)
#define TX_BD_FLG_IP_TCP_UDP_CHKSUM	(TX_BD_LONG_LFLAGS_TCP_UDP_CHKSUM | \
					TX_BD_LONG_LFLAGS_IP_CHKSUM)
#define TX_BD_FLG_TIP_IP_CHKSUM		(TX_BD_LONG_LFLAGS_T_IP_CHKSUM | \
					TX_BD_LONG_LFLAGS_IP_CHKSUM)
#define TX_BD_FLG_TIP_TCP_UDP_CHKSUM	(TX_BD_LONG_LFLAGS_TCP_UDP_CHKSUM | \
					TX_BD_LONG_LFLAGS_T_IP_CHKSUM)

#endif
