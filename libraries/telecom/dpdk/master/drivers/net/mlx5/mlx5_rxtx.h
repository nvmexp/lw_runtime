/* SPDX-License-Identifier: BSD-3-Clause
 * Copyright 2015 6WIND S.A.
 * Copyright 2015 Mellanox Technologies, Ltd
 */

#ifndef RTE_PMD_MLX5_RXTX_H_
#define RTE_PMD_MLX5_RXTX_H_

#include <stddef.h>
#include <stdint.h>
#include <sys/queue.h>

#include <rte_mbuf.h>
#include <rte_mempool.h>
#include <rte_common.h>
#include <rte_hexdump.h>
#include <rte_spinlock.h>
#include <rte_io.h>
#include <rte_bus_pci.h>
#include <rte_malloc.h>
#include <rte_cycles.h>

#include <mlx5_glue.h>
#include <mlx5_prm.h>
#include <mlx5_common.h>
#include <mlx5_common_mr.h>

#include "mlx5_defs.h"
#include "mlx5_utils.h"
#include "mlx5.h"
#include "mlx5_autoconf.h"
#include "mlx5_mr.h"

/* Support tunnel matching. */
#define MLX5_FLOW_TUNNEL 10

/* Mbuf dynamic flag offset for inline. */
extern uint64_t rte_net_mlx5_dynf_inline_mask;

struct mlx5_rxq_stats {
#ifdef MLX5_PMD_SOFT_COUNTERS
	uint64_t ipackets; /**< Total of successfully received packets. */
	uint64_t ibytes; /**< Total of successfully received bytes. */
#endif
	uint64_t idropped; /**< Total of packets dropped when RX ring full. */
	uint64_t rx_nombuf; /**< Total of RX mbuf allocation failures. */
};

struct mlx5_txq_stats {
#ifdef MLX5_PMD_SOFT_COUNTERS
	uint64_t opackets; /**< Total of successfully sent packets. */
	uint64_t obytes; /**< Total of successfully sent bytes. */
#endif
	uint64_t oerrors; /**< Total number of failed transmitted packets. */
};

struct mlx5_priv;

/* Compressed CQE context. */
struct rxq_zip {
	uint16_t ai; /* Array index. */
	uint16_t ca; /* Current array index. */
	uint16_t na; /* Next array index. */
	uint16_t cq_ci; /* The next CQE. */
	uint32_t cqe_cnt; /* Number of CQEs. */
};

/* Multi-Packet RQ buffer header. */
struct mlx5_mprq_buf {
	struct rte_mempool *mp;
	uint16_t refcnt; /* Atomically accessed refcnt. */
	uint8_t pad[RTE_PKTMBUF_HEADROOM]; /* Headroom for the first packet. */
	struct rte_mbuf_ext_shared_info shinfos[];
	/*
	 * Shared information per stride.
	 * More memory will be allocated for the first stride head-room and for
	 * the strides data.
	 */
} __rte_cache_aligned;

/* Get pointer to the first stride. */
#define mlx5_mprq_buf_addr(ptr, strd_n) (RTE_PTR_ADD((ptr), \
				sizeof(struct mlx5_mprq_buf) + \
				(strd_n) * \
				sizeof(struct rte_mbuf_ext_shared_info) + \
				RTE_PKTMBUF_HEADROOM))

#define MLX5_MIN_SINGLE_STRIDE_LOG_NUM_BYTES 6
#define MLX5_MIN_SINGLE_WQE_LOG_NUM_STRIDES 9

enum mlx5_rxq_err_state {
	MLX5_RXQ_ERR_STATE_NO_ERROR = 0,
	MLX5_RXQ_ERR_STATE_NEED_RESET,
	MLX5_RXQ_ERR_STATE_NEED_READY,
};

enum mlx5_rqx_code {
	MLX5_RXQ_CODE_EXIT = 0,
	MLX5_RXQ_CODE_NOMBUF,
	MLX5_RXQ_CODE_DROPPED,
};

struct mlx5_eth_rxseg {
	struct rte_mempool *mp; /**< Memory pool to allocate segment from. */
	uint16_t length; /**< Segment data length, configures split point. */
	uint16_t offset; /**< Data offset from beginning of mbuf data buffer. */
	uint32_t reserved; /**< Reserved field. */
};

/* RX queue descriptor. */
struct mlx5_rxq_data {
	unsigned int csum:1; /* Enable checksum offloading. */
	unsigned int hw_timestamp:1; /* Enable HW timestamp. */
	unsigned int rt_timestamp:1; /* Realtime timestamp format. */
	unsigned int vlan_strip:1; /* Enable VLAN stripping. */
	unsigned int crc_present:1; /* CRC must be subtracted. */
	unsigned int sges_n:3; /* Log 2 of SGEs (max buffers per packet). */
	unsigned int cqe_n:4; /* Log 2 of CQ elements. */
	unsigned int elts_n:4; /* Log 2 of Mbufs. */
	unsigned int rss_hash:1; /* RSS hash result is enabled. */
	unsigned int mark:1; /* Marked flow available on the queue. */
	unsigned int strd_num_n:5; /* Log 2 of the number of stride. */
	unsigned int strd_sz_n:4; /* Log 2 of stride size. */
	unsigned int strd_shift_en:1; /* Enable 2bytes shift on a stride. */
	unsigned int err_state:2; /* enum mlx5_rxq_err_state. */
	unsigned int strd_scatter_en:1; /* Scattered packets from a stride. */
	unsigned int lro:1; /* Enable LRO. */
	unsigned int dynf_meta:1; /* Dynamic metadata is configured. */
	unsigned int mcqe_format:3; /* Dynamic metadata is configured. */
	volatile uint32_t *rq_db;
	volatile uint32_t *cq_db;
	uint16_t port_id;
	uint32_t elts_ci;
	uint32_t rq_ci;
	uint16_t consumed_strd; /* Number of consumed strides in WQE. */
	uint32_t rq_pi;
	uint32_t cq_ci;
	uint16_t rq_repl_thresh; /* Threshold for buffer replenishment. */
	uint32_t byte_mask;
	union {
		struct rxq_zip zip; /* Compressed context. */
		uint16_t decompressed;
		/* Number of ready mbufs decompressed from the CQ. */
	};
	struct mlx5_mr_ctrl mr_ctrl; /* MR control descriptor. */
	uint16_t mprq_max_memcpy_len; /* Maximum size of packet to memcpy. */
	volatile void *wqes;
	volatile struct mlx5_cqe(*cqes)[];
	struct rte_mbuf *(*elts)[];
	struct mlx5_mprq_buf *(*mprq_bufs)[];
	struct rte_mempool *mp;
	struct rte_mempool *mprq_mp; /* Mempool for Multi-Packet RQ. */
	struct mlx5_mprq_buf *mprq_repl; /* Stashed mbuf for replenish. */
	struct mlx5_dev_ctx_shared *sh; /* Shared context. */
	uint16_t idx; /* Queue index. */
	struct mlx5_rxq_stats stats;
	rte_xmm_t mbuf_initializer; /* Default rearm/flags for vectorized Rx. */
	struct rte_mbuf fake_mbuf; /* elts padding for vectorized Rx. */
	void *cq_uar; /* Verbs CQ user access region. */
	uint32_t cqn; /* CQ number. */
	uint8_t cq_arm_sn; /* CQ arm seq number. */
#ifndef RTE_ARCH_64
	rte_spinlock_t *uar_lock_cq;
	/* CQ (UAR) access lock required for 32bit implementations */
#endif
	uint32_t tunnel; /* Tunnel information. */
	int timestamp_offset; /* Dynamic mbuf field for timestamp. */
	uint64_t timestamp_rx_flag; /* Dynamic mbuf flag for timestamp. */
	uint64_t flow_meta_mask;
	int32_t flow_meta_offset;
	uint32_t rxseg_n; /* Number of split segment descriptions. */
	struct mlx5_eth_rxseg rxseg[MLX5_MAX_RXQ_NSEG];
	/* Buffer split segment descriptions - sizes, offsets, pools. */
} __rte_cache_aligned;

enum mlx5_rxq_type {
	MLX5_RXQ_TYPE_STANDARD, /* Standard Rx queue. */
	MLX5_RXQ_TYPE_HAIRPIN, /* Hairpin Rx queue. */
	MLX5_RXQ_TYPE_UNDEFINED,
};

/* RX queue control descriptor. */
struct mlx5_rxq_ctrl {
	struct mlx5_rxq_data rxq; /* Data path structure. */
	LIST_ENTRY(mlx5_rxq_ctrl) next; /* Pointer to the next element. */
	uint32_t refcnt; /* Reference counter. */
	struct mlx5_rxq_obj *obj; /* Verbs/DevX elements. */
	struct mlx5_priv *priv; /* Back pointer to private data. */
	enum mlx5_rxq_type type; /* Rxq type. */
	unsigned int socket; /* CPU socket ID for allocations. */
	unsigned int irq:1; /* Whether IRQ is enabled. */
	uint32_t flow_mark_n; /* Number of Mark/Flag flows using this Queue. */
	uint32_t flow_tunnels_n[MLX5_FLOW_TUNNEL]; /* Tunnels counters. */
	uint32_t wqn; /* WQ number. */
	uint16_t dump_file_n; /* Number of dump files. */
	struct mlx5_devx_dbr_page *rq_dbrec_page;
	uint64_t rq_dbr_offset;
	/* Storing RQ door-bell information, needed when freeing door-bell. */
	struct mlx5_devx_dbr_page *cq_dbrec_page;
	uint64_t cq_dbr_offset;
	/* Storing CQ door-bell information, needed when freeing door-bell. */
	void *wq_umem; /* WQ buffer registration info. */
	void *cq_umem; /* CQ buffer registration info. */
	struct rte_eth_hairpin_conf hairpin_conf; /* Hairpin configuration. */
	uint32_t hairpin_status; /* Hairpin binding status. */
};

/* TX queue send local data. */
__extension__
struct mlx5_txq_local {
	struct mlx5_wqe *wqe_last; /* last sent WQE pointer. */
	struct rte_mbuf *mbuf; /* first mbuf to process. */
	uint16_t pkts_copy; /* packets copied to elts. */
	uint16_t pkts_sent; /* packets sent. */
	uint16_t pkts_loop; /* packets sent on loop entry. */
	uint16_t elts_free; /* available elts remain. */
	uint16_t wqe_free; /* available wqe remain. */
	uint16_t mbuf_off; /* data offset in current mbuf. */
	uint16_t mbuf_nseg; /* number of remaining mbuf. */
};

/* TX queue descriptor. */
__extension__
struct mlx5_txq_data {
	uint16_t elts_head; /* Current counter in (*elts)[]. */
	uint16_t elts_tail; /* Counter of first element awaiting completion. */
	uint16_t elts_comp; /* elts index since last completion request. */
	uint16_t elts_s; /* Number of mbuf elements. */
	uint16_t elts_m; /* Mask for mbuf elements indices. */
	/* Fields related to elts mbuf storage. */
	uint16_t wqe_ci; /* Consumer index for work queue. */
	uint16_t wqe_pi; /* Producer index for work queue. */
	uint16_t wqe_s; /* Number of WQ elements. */
	uint16_t wqe_m; /* Mask Number for WQ elements. */
	uint16_t wqe_comp; /* WQE index since last completion request. */
	uint16_t wqe_thres; /* WQE threshold to request completion in CQ. */
	/* WQ related fields. */
	uint16_t cq_ci; /* Consumer index for completion queue. */
	uint16_t cq_pi; /* Production index for completion queue. */
	uint16_t cqe_s; /* Number of CQ elements. */
	uint16_t cqe_m; /* Mask for CQ indices. */
	/* CQ related fields. */
	uint16_t elts_n:4; /* elts[] length (in log2). */
	uint16_t cqe_n:4; /* Number of CQ elements (in log2). */
	uint16_t wqe_n:4; /* Number of WQ elements (in log2). */
	uint16_t tso_en:1; /* When set hardware TSO is enabled. */
	uint16_t tunnel_en:1;
	/* When set TX offload for tunneled packets are supported. */
	uint16_t swp_en:1; /* Whether SW parser is enabled. */
	uint16_t vlan_en:1; /* VLAN insertion in WQE is supported. */
	uint16_t db_nc:1; /* Doorbell mapped to non-cached region. */
	uint16_t db_heu:1; /* Doorbell heuristic write barrier. */
	uint16_t inlen_send; /* Ordinary send data inline size. */
	uint16_t inlen_empw; /* eMPW max packet size to inline. */
	uint16_t inlen_mode; /* Minimal data length to inline. */
	uint32_t qp_num_8s; /* QP number shifted by 8. */
	uint64_t offloads; /* Offloads for Tx Queue. */
	struct mlx5_mr_ctrl mr_ctrl; /* MR control descriptor. */
	struct mlx5_wqe *wqes; /* Work queue. */
	struct mlx5_wqe *wqes_end; /* Work queue array limit. */
#ifdef RTE_LIBRTE_MLX5_DEBUG
	uint32_t *fcqs; /* Free completion queue (debug extended). */
#else
	uint16_t *fcqs; /* Free completion queue. */
#endif
	volatile struct mlx5_cqe *cqes; /* Completion queue. */
	volatile uint32_t *qp_db; /* Work queue doorbell. */
	volatile uint32_t *cq_db; /* Completion queue doorbell. */
	uint16_t port_id; /* Port ID of device. */
	uint16_t idx; /* Queue index. */
	uint64_t ts_mask; /* Timestamp flag dynamic mask. */
	int32_t ts_offset; /* Timestamp field dynamic offset. */
	struct mlx5_dev_ctx_shared *sh; /* Shared context. */
	struct mlx5_txq_stats stats; /* TX queue counters. */
#ifndef RTE_ARCH_64
	rte_spinlock_t *uar_lock;
	/* UAR access lock required for 32bit implementations */
#endif
	struct rte_mbuf *elts[0];
	/* Storage for queued packets, must be the last field. */
} __rte_cache_aligned;

enum mlx5_txq_type {
	MLX5_TXQ_TYPE_STANDARD, /* Standard Tx queue. */
	MLX5_TXQ_TYPE_HAIRPIN, /* Hairpin Rx queue. */
};

/* TX queue control descriptor. */
struct mlx5_txq_ctrl {
	LIST_ENTRY(mlx5_txq_ctrl) next; /* Pointer to the next element. */
	uint32_t refcnt; /* Reference counter. */
	unsigned int socket; /* CPU socket ID for allocations. */
	enum mlx5_txq_type type; /* The txq ctrl type. */
	unsigned int max_inline_data; /* Max inline data. */
	unsigned int max_tso_header; /* Max TSO header size. */
	struct mlx5_txq_obj *obj; /* Verbs/DevX queue object. */
	struct mlx5_priv *priv; /* Back pointer to private data. */
	off_t uar_mmap_offset; /* UAR mmap offset for non-primary process. */
	void *bf_reg; /* BlueFlame register from Verbs. */
	uint16_t dump_file_n; /* Number of dump files. */
	struct rte_eth_hairpin_conf hairpin_conf; /* Hairpin configuration. */
	uint32_t hairpin_status; /* Hairpin binding status. */
	struct mlx5_txq_data txq; /* Data path structure. */
	/* Must be the last field in the structure, contains elts[]. */
};

#define MLX5_TX_BFREG(txq) \
		(MLX5_PROC_PRIV((txq)->port_id)->uar_table[(txq)->idx])

/* mlx5_rxq.c */

extern uint8_t rss_hash_default_key[];

int mlx5_check_mprq_support(struct rte_eth_dev *dev);
int mlx5_rxq_mprq_enabled(struct mlx5_rxq_data *rxq);
int mlx5_mprq_enabled(struct rte_eth_dev *dev);
unsigned int mlx5_rxq_cqe_num(struct mlx5_rxq_data *rxq_data);
int mlx5_mprq_free_mp(struct rte_eth_dev *dev);
int mlx5_mprq_alloc_mp(struct rte_eth_dev *dev);
int mlx5_rx_queue_start(struct rte_eth_dev *dev, uint16_t queue_id);
int mlx5_rx_queue_stop(struct rte_eth_dev *dev, uint16_t queue_id);
int mlx5_rx_queue_start_primary(struct rte_eth_dev *dev, uint16_t queue_id);
int mlx5_rx_queue_stop_primary(struct rte_eth_dev *dev, uint16_t queue_id);
int mlx5_rx_queue_setup(struct rte_eth_dev *dev, uint16_t idx, uint16_t desc,
			unsigned int socket, const struct rte_eth_rxconf *conf,
			struct rte_mempool *mp);
int mlx5_rx_hairpin_queue_setup
	(struct rte_eth_dev *dev, uint16_t idx, uint16_t desc,
	 const struct rte_eth_hairpin_conf *hairpin_conf);
void mlx5_rx_queue_release(void *dpdk_rxq);
int mlx5_rx_intr_vec_enable(struct rte_eth_dev *dev);
void mlx5_rx_intr_vec_disable(struct rte_eth_dev *dev);
int mlx5_rx_intr_enable(struct rte_eth_dev *dev, uint16_t rx_queue_id);
int mlx5_rx_intr_disable(struct rte_eth_dev *dev, uint16_t rx_queue_id);
int mlx5_rxq_obj_verify(struct rte_eth_dev *dev);
struct mlx5_rxq_ctrl *mlx5_rxq_new(struct rte_eth_dev *dev, uint16_t idx,
				   uint16_t desc, unsigned int socket,
				   const struct rte_eth_rxconf *conf,
				   const struct rte_eth_rxseg_split *rx_seg,
				   uint16_t n_seg);
struct mlx5_rxq_ctrl *mlx5_rxq_hairpin_new
	(struct rte_eth_dev *dev, uint16_t idx, uint16_t desc,
	 const struct rte_eth_hairpin_conf *hairpin_conf);
struct mlx5_rxq_ctrl *mlx5_rxq_get(struct rte_eth_dev *dev, uint16_t idx);
int mlx5_rxq_release(struct rte_eth_dev *dev, uint16_t idx);
int mlx5_rxq_verify(struct rte_eth_dev *dev);
int rxq_alloc_elts(struct mlx5_rxq_ctrl *rxq_ctrl);
int mlx5_ind_table_obj_verify(struct rte_eth_dev *dev);
struct mlx5_ind_table_obj *mlx5_ind_table_obj_get(struct rte_eth_dev *dev,
						  const uint16_t *queues,
						  uint32_t queues_n);
int mlx5_ind_table_obj_release(struct rte_eth_dev *dev,
			       struct mlx5_ind_table_obj *ind_tbl,
			       bool standalone);
int mlx5_ind_table_obj_setup(struct rte_eth_dev *dev,
			     struct mlx5_ind_table_obj *ind_tbl);
int mlx5_ind_table_obj_modify(struct rte_eth_dev *dev,
			      struct mlx5_ind_table_obj *ind_tbl,
			      uint16_t *queues, const uint32_t queues_n,
			      bool standalone);
struct mlx5_cache_entry *mlx5_hrxq_create_cb(struct mlx5_cache_list *list,
		struct mlx5_cache_entry *entry __rte_unused, void *cb_ctx);
int mlx5_hrxq_match_cb(struct mlx5_cache_list *list,
		       struct mlx5_cache_entry *entry,
		       void *cb_ctx);
void mlx5_hrxq_remove_cb(struct mlx5_cache_list *list,
			 struct mlx5_cache_entry *entry);
uint32_t mlx5_hrxq_get(struct rte_eth_dev *dev,
		       struct mlx5_flow_rss_desc *rss_desc);
int mlx5_hrxq_release(struct rte_eth_dev *dev, uint32_t hxrq_idx);
uint32_t mlx5_hrxq_verify(struct rte_eth_dev *dev);


enum mlx5_rxq_type mlx5_rxq_get_type(struct rte_eth_dev *dev, uint16_t idx);
const struct rte_eth_hairpin_conf *mlx5_rxq_get_hairpin_conf
	(struct rte_eth_dev *dev, uint16_t idx);
struct mlx5_hrxq *mlx5_drop_action_create(struct rte_eth_dev *dev);
void mlx5_drop_action_destroy(struct rte_eth_dev *dev);
uint64_t mlx5_get_rx_port_offloads(void);
uint64_t mlx5_get_rx_queue_offloads(struct rte_eth_dev *dev);
void mlx5_rxq_timestamp_set(struct rte_eth_dev *dev);
int mlx5_hrxq_modify(struct rte_eth_dev *dev, uint32_t hxrq_idx,
		     const uint8_t *rss_key, uint32_t rss_key_len,
		     uint64_t hash_fields,
		     const uint16_t *queues, uint32_t queues_n);

/* mlx5_txq.c */

int mlx5_tx_queue_start(struct rte_eth_dev *dev, uint16_t queue_id);
int mlx5_tx_queue_stop(struct rte_eth_dev *dev, uint16_t queue_id);
int mlx5_tx_queue_start_primary(struct rte_eth_dev *dev, uint16_t queue_id);
int mlx5_tx_queue_stop_primary(struct rte_eth_dev *dev, uint16_t queue_id);
int mlx5_tx_queue_setup(struct rte_eth_dev *dev, uint16_t idx, uint16_t desc,
			unsigned int socket, const struct rte_eth_txconf *conf);
int mlx5_tx_hairpin_queue_setup
	(struct rte_eth_dev *dev, uint16_t idx, uint16_t desc,
	 const struct rte_eth_hairpin_conf *hairpin_conf);
void mlx5_tx_queue_release(void *dpdk_txq);
void txq_uar_init(struct mlx5_txq_ctrl *txq_ctrl);
int mlx5_tx_uar_init_secondary(struct rte_eth_dev *dev, int fd);
void mlx5_tx_uar_uninit_secondary(struct rte_eth_dev *dev);
int mlx5_txq_obj_verify(struct rte_eth_dev *dev);
struct mlx5_txq_ctrl *mlx5_txq_new(struct rte_eth_dev *dev, uint16_t idx,
				   uint16_t desc, unsigned int socket,
				   const struct rte_eth_txconf *conf);
struct mlx5_txq_ctrl *mlx5_txq_hairpin_new
	(struct rte_eth_dev *dev, uint16_t idx, uint16_t desc,
	 const struct rte_eth_hairpin_conf *hairpin_conf);
struct mlx5_txq_ctrl *mlx5_txq_get(struct rte_eth_dev *dev, uint16_t idx);
int mlx5_txq_release(struct rte_eth_dev *dev, uint16_t idx);
int mlx5_txq_releasable(struct rte_eth_dev *dev, uint16_t idx);
int mlx5_txq_verify(struct rte_eth_dev *dev);
void txq_alloc_elts(struct mlx5_txq_ctrl *txq_ctrl);
void txq_free_elts(struct mlx5_txq_ctrl *txq_ctrl);
uint64_t mlx5_get_tx_port_offloads(struct rte_eth_dev *dev);
void mlx5_txq_dynf_timestamp_set(struct rte_eth_dev *dev);

/* mlx5_rxtx.c */

extern uint32_t mlx5_ptype_table[];
extern uint8_t mlx5_cksum_table[];
extern uint8_t mlx5_swp_types_table[];

void mlx5_set_ptype_table(void);
void mlx5_set_cksum_table(void);
void mlx5_set_swp_types_table(void);
uint16_t mlx5_rx_burst(void *dpdk_rxq, struct rte_mbuf **pkts, uint16_t pkts_n);
void mlx5_rxq_initialize(struct mlx5_rxq_data *rxq);
__rte_noinline int mlx5_rx_err_handle(struct mlx5_rxq_data *rxq, uint8_t vec);
void mlx5_mprq_buf_free_cb(void *addr, void *opaque);
void mlx5_mprq_buf_free(struct mlx5_mprq_buf *buf);
uint16_t mlx5_rx_burst_mprq(void *dpdk_rxq, struct rte_mbuf **pkts,
			    uint16_t pkts_n);
uint16_t removed_tx_burst(void *dpdk_txq, struct rte_mbuf **pkts,
			  uint16_t pkts_n);
uint16_t removed_rx_burst(void *dpdk_rxq, struct rte_mbuf **pkts,
			  uint16_t pkts_n);
int mlx5_rx_descriptor_status(void *rx_queue, uint16_t offset);
int mlx5_tx_descriptor_status(void *tx_queue, uint16_t offset);
uint32_t mlx5_rx_queue_count(struct rte_eth_dev *dev, uint16_t rx_queue_id);
void mlx5_dump_debug_information(const char *path, const char *title,
				 const void *buf, unsigned int len);
int mlx5_queue_state_modify_primary(struct rte_eth_dev *dev,
			const struct mlx5_mp_arg_queue_state_modify *sm);
void mlx5_rxq_info_get(struct rte_eth_dev *dev, uint16_t queue_id,
		       struct rte_eth_rxq_info *qinfo);
void mlx5_txq_info_get(struct rte_eth_dev *dev, uint16_t queue_id,
		       struct rte_eth_txq_info *qinfo);
int mlx5_rx_burst_mode_get(struct rte_eth_dev *dev, uint16_t rx_queue_id,
			   struct rte_eth_burst_mode *mode);
int mlx5_tx_burst_mode_get(struct rte_eth_dev *dev, uint16_t tx_queue_id,
			   struct rte_eth_burst_mode *mode);

/* Vectorized version of mlx5_rxtx.c */
int mlx5_rxq_check_vec_support(struct mlx5_rxq_data *rxq_data);
int mlx5_check_vec_rx_support(struct rte_eth_dev *dev);
uint16_t mlx5_rx_burst_vec(void *dpdk_txq, struct rte_mbuf **pkts,
			   uint16_t pkts_n);
uint16_t mlx5_rx_burst_mprq_vec(void *dpdk_txq, struct rte_mbuf **pkts,
				uint16_t pkts_n);

/* mlx5_mr.c */

void mlx5_mr_flush_local_cache(struct mlx5_mr_ctrl *mr_ctrl);
uint32_t mlx5_rx_addr2mr_bh(struct mlx5_rxq_data *rxq, uintptr_t addr);
uint32_t mlx5_tx_mb2mr_bh(struct mlx5_txq_data *txq, struct rte_mbuf *mb);
uint32_t mlx5_tx_update_ext_mp(struct mlx5_txq_data *txq, uintptr_t addr,
			       struct rte_mempool *mp);
int mlx5_dma_map(struct rte_pci_device *pdev, void *addr, uint64_t iova,
		 size_t len);
int mlx5_dma_unmap(struct rte_pci_device *pdev, void *addr, uint64_t iova,
		   size_t len);

/**
 * Provide safe 64bit store operation to mlx5 UAR region for both 32bit and
 * 64bit architectures.
 *
 * @param val
 *   value to write in CPU endian format.
 * @param addr
 *   Address to write to.
 * @param lock
 *   Address of the lock to use for that UAR access.
 */
static __rte_always_inline void
__mlx5_uar_write64_relaxed(uint64_t val, void *addr,
			   rte_spinlock_t *lock __rte_unused)
{
#ifdef RTE_ARCH_64
	*(uint64_t *)addr = val;
#else /* !RTE_ARCH_64 */
	rte_spinlock_lock(lock);
	*(uint32_t *)addr = val;
	rte_io_wmb();
	*((uint32_t *)addr + 1) = val >> 32;
	rte_spinlock_unlock(lock);
#endif
}

/**
 * Provide safe 64bit store operation to mlx5 UAR region for both 32bit and
 * 64bit architectures while guaranteeing the order of exelwtion with the
 * code being exelwted.
 *
 * @param val
 *   value to write in CPU endian format.
 * @param addr
 *   Address to write to.
 * @param lock
 *   Address of the lock to use for that UAR access.
 */
static __rte_always_inline void
__mlx5_uar_write64(uint64_t val, void *addr, rte_spinlock_t *lock)
{
	rte_io_wmb();
	__mlx5_uar_write64_relaxed(val, addr, lock);
}

/* Assist macros, used instead of directly calling the functions they wrap. */
#ifdef RTE_ARCH_64
#define mlx5_uar_write64_relaxed(val, dst, lock) \
		__mlx5_uar_write64_relaxed(val, dst, NULL)
#define mlx5_uar_write64(val, dst, lock) __mlx5_uar_write64(val, dst, NULL)
#else
#define mlx5_uar_write64_relaxed(val, dst, lock) \
		__mlx5_uar_write64_relaxed(val, dst, lock)
#define mlx5_uar_write64(val, dst, lock) __mlx5_uar_write64(val, dst, lock)
#endif

/**
 * Get Memory Pool (MP) from mbuf. If mbuf is indirect, the pool from which the
 * cloned mbuf is allocated is returned instead.
 *
 * @param buf
 *   Pointer to mbuf.
 *
 * @return
 *   Memory pool where data is located for given mbuf.
 */
static inline struct rte_mempool *
mlx5_mb2mp(struct rte_mbuf *buf)
{
	if (unlikely(RTE_MBUF_CLONED(buf)))
		return rte_mbuf_from_indirect(buf)->pool;
	return buf->pool;
}

/**
 * Query LKey from a packet buffer for Rx. No need to flush local caches for Rx
 * as mempool is pre-configured and static.
 *
 * @param rxq
 *   Pointer to Rx queue structure.
 * @param addr
 *   Address to search.
 *
 * @return
 *   Searched LKey on success, UINT32_MAX on no match.
 */
static __rte_always_inline uint32_t
mlx5_rx_addr2mr(struct mlx5_rxq_data *rxq, uintptr_t addr)
{
	struct mlx5_mr_ctrl *mr_ctrl = &rxq->mr_ctrl;
	uint32_t lkey;

	/* Linear search on MR cache array. */
	lkey = mlx5_mr_lookup_lkey(mr_ctrl->cache, &mr_ctrl->mru,
				   MLX5_MR_CACHE_N, addr);
	if (likely(lkey != UINT32_MAX))
		return lkey;
	/* Take slower bottom-half (Binary Search) on miss. */
	return mlx5_rx_addr2mr_bh(rxq, addr);
}

#define mlx5_rx_mb2mr(rxq, mb) mlx5_rx_addr2mr(rxq, (uintptr_t)((mb)->buf_addr))

/**
 * Query LKey from a packet buffer for Tx. If not found, add the mempool.
 *
 * @param txq
 *   Pointer to Tx queue structure.
 * @param addr
 *   Address to search.
 *
 * @return
 *   Searched LKey on success, UINT32_MAX on no match.
 */
static __rte_always_inline uint32_t
mlx5_tx_mb2mr(struct mlx5_txq_data *txq, struct rte_mbuf *mb)
{
	struct mlx5_mr_ctrl *mr_ctrl = &txq->mr_ctrl;
	uintptr_t addr = (uintptr_t)mb->buf_addr;
	uint32_t lkey;

	/* Check generation bit to see if there's any change on existing MRs. */
	if (unlikely(*mr_ctrl->dev_gen_ptr != mr_ctrl->lwr_gen))
		mlx5_mr_flush_local_cache(mr_ctrl);
	/* Linear search on MR cache array. */
	lkey = mlx5_mr_lookup_lkey(mr_ctrl->cache, &mr_ctrl->mru,
				   MLX5_MR_CACHE_N, addr);
	if (likely(lkey != UINT32_MAX))
		return lkey;
	/* Take slower bottom-half on miss. */
	return mlx5_tx_mb2mr_bh(txq, mb);
}

/**
 * Ring TX queue doorbell and flush the update if requested.
 *
 * @param txq
 *   Pointer to TX queue structure.
 * @param wqe
 *   Pointer to the last WQE posted in the NIC.
 * @param cond
 *   Request for write memory barrier after BlueFlame update.
 */
static __rte_always_inline void
mlx5_tx_dbrec_cond_wmb(struct mlx5_txq_data *txq, volatile struct mlx5_wqe *wqe,
		       int cond)
{
	uint64_t *dst = MLX5_TX_BFREG(txq);
	volatile uint64_t *src = ((volatile uint64_t *)wqe);

	rte_io_wmb();
	*txq->qp_db = rte_cpu_to_be_32(txq->wqe_ci);
	/* Ensure ordering between DB record and BF copy. */
	rte_wmb();
	mlx5_uar_write64_relaxed(*src, dst, txq->uar_lock);
	if (cond)
		rte_wmb();
}

/**
 * Ring TX queue doorbell and flush the update by write memory barrier.
 *
 * @param txq
 *   Pointer to TX queue structure.
 * @param wqe
 *   Pointer to the last WQE posted in the NIC.
 */
static __rte_always_inline void
mlx5_tx_dbrec(struct mlx5_txq_data *txq, volatile struct mlx5_wqe *wqe)
{
	mlx5_tx_dbrec_cond_wmb(txq, wqe, 1);
}

/**
 * Colwert timestamp from HW format to linear counter
 * from Packet Pacing Clock Queue CQE timestamp format.
 *
 * @param sh
 *   Pointer to the device shared context. Might be needed
 *   to colwert according current device configuration.
 * @param ts
 *   Timestamp from CQE to colwert.
 * @return
 *   UTC in nanoseconds
 */
static __rte_always_inline uint64_t
mlx5_txpp_colwert_rx_ts(struct mlx5_dev_ctx_shared *sh, uint64_t ts)
{
	RTE_SET_USED(sh);
	return (ts & UINT32_MAX) + (ts >> 32) * NS_PER_S;
}

/**
 * Colwert timestamp from mbuf format to linear counter
 * of Clock Queue completions (24 bits)
 *
 * @param sh
 *   Pointer to the device shared context to fetch Tx
 *   packet pacing timestamp and parameters.
 * @param ts
 *   Timestamp from mbuf to colwert.
 * @return
 *   positive or zero value - completion ID to wait
 *   negative value - colwersion error
 */
static __rte_always_inline int32_t
mlx5_txpp_colwert_tx_ts(struct mlx5_dev_ctx_shared *sh, uint64_t mts)
{
	uint64_t ts, ci;
	uint32_t tick;

	do {
		/*
		 * Read atomically two uint64_t fields and compare lsb bits.
		 * It there is no match - the timestamp was updated in
		 * the service thread, data should be re-read.
		 */
		rte_compiler_barrier();
		ci = __atomic_load_n(&sh->txpp.ts.ci_ts, __ATOMIC_RELAXED);
		ts = __atomic_load_n(&sh->txpp.ts.ts, __ATOMIC_RELAXED);
		rte_compiler_barrier();
		if (!((ts ^ ci) << (64 - MLX5_CQ_INDEX_WIDTH)))
			break;
	} while (true);
	/* Perform the skew correction, positive value to send earlier. */
	mts -= sh->txpp.skew;
	mts -= ts;
	if (unlikely(mts >= UINT64_MAX / 2)) {
		/* We have negative integer, mts is in the past. */
		__atomic_fetch_add(&sh->txpp.err_ts_past,
				   1, __ATOMIC_RELAXED);
		return -1;
	}
	tick = sh->txpp.tick;
	MLX5_ASSERT(tick);
	/* Colwert delta to completions, round up. */
	mts = (mts + tick - 1) / tick;
	if (unlikely(mts >= (1 << MLX5_CQ_INDEX_WIDTH) / 2 - 1)) {
		/* We have mts is too distant future. */
		__atomic_fetch_add(&sh->txpp.err_ts_future,
				   1, __ATOMIC_RELAXED);
		return -1;
	}
	mts <<= 64 - MLX5_CQ_INDEX_WIDTH;
	ci += mts;
	ci >>= 64 - MLX5_CQ_INDEX_WIDTH;
	return ci;
}

/**
 * Set timestamp in mbuf dynamic field.
 *
 * @param mbuf
 *   Structure to write into.
 * @param offset
 *   Dynamic field offset in mbuf structure.
 * @param timestamp
 *   Value to write.
 */
static __rte_always_inline void
mlx5_timestamp_set(struct rte_mbuf *mbuf, int offset,
		rte_mbuf_timestamp_t timestamp)
{
	*RTE_MBUF_DYNFIELD(mbuf, offset, rte_mbuf_timestamp_t *) = timestamp;
}

/**
 * Replace MPRQ buffer.
 *
 * @param rxq
 *   Pointer to Rx queue structure.
 * @param rq_idx
 *   RQ index to replace.
 */
static __rte_always_inline void
mprq_buf_replace(struct mlx5_rxq_data *rxq, uint16_t rq_idx)
{
	const uint32_t strd_n = 1 << rxq->strd_num_n;
	struct mlx5_mprq_buf *rep = rxq->mprq_repl;
	volatile struct mlx5_wqe_data_seg *wqe =
		&((volatile struct mlx5_wqe_mprq *)rxq->wqes)[rq_idx].dseg;
	struct mlx5_mprq_buf *buf = (*rxq->mprq_bufs)[rq_idx];
	void *addr;

	if (__atomic_load_n(&buf->refcnt, __ATOMIC_RELAXED) > 1) {
		MLX5_ASSERT(rep != NULL);
		/* Replace MPRQ buf. */
		(*rxq->mprq_bufs)[rq_idx] = rep;
		/* Replace WQE. */
		addr = mlx5_mprq_buf_addr(rep, strd_n);
		wqe->addr = rte_cpu_to_be_64((uintptr_t)addr);
		/* If there's only one MR, no need to replace LKey in WQE. */
		if (unlikely(mlx5_mr_btree_len(&rxq->mr_ctrl.cache_bh) > 1))
			wqe->lkey = mlx5_rx_addr2mr(rxq, (uintptr_t)addr);
		/* Stash a mbuf for next replacement. */
		if (likely(!rte_mempool_get(rxq->mprq_mp, (void **)&rep)))
			rxq->mprq_repl = rep;
		else
			rxq->mprq_repl = NULL;
		/* Release the old buffer. */
		mlx5_mprq_buf_free(buf);
	} else if (unlikely(rxq->mprq_repl == NULL)) {
		struct mlx5_mprq_buf *rep;

		/*
		 * Lwrrently, the MPRQ mempool is out of buffer
		 * and doing memcpy regardless of the size of Rx
		 * packet. Retry allocation to get back to
		 * normal.
		 */
		if (!rte_mempool_get(rxq->mprq_mp, (void **)&rep))
			rxq->mprq_repl = rep;
	}
}

/**
 * Attach or copy MPRQ buffer content to a packet.
 *
 * @param rxq
 *   Pointer to Rx queue structure.
 * @param pkt
 *   Pointer to a packet to fill.
 * @param len
 *   Packet length.
 * @param buf
 *   Pointer to a MPRQ buffer to take the data from.
 * @param strd_idx
 *   Stride index to start from.
 * @param strd_cnt
 *   Number of strides to consume.
 */
static __rte_always_inline enum mlx5_rqx_code
mprq_buf_to_pkt(struct mlx5_rxq_data *rxq, struct rte_mbuf *pkt, uint32_t len,
		struct mlx5_mprq_buf *buf, uint16_t strd_idx, uint16_t strd_cnt)
{
	const uint32_t strd_n = 1 << rxq->strd_num_n;
	const uint16_t strd_sz = 1 << rxq->strd_sz_n;
	const uint16_t strd_shift =
		MLX5_MPRQ_STRIDE_SHIFT_BYTE * rxq->strd_shift_en;
	const int32_t hdrm_overlap =
		len + RTE_PKTMBUF_HEADROOM - strd_cnt * strd_sz;
	const uint32_t offset = strd_idx * strd_sz + strd_shift;
	void *addr = RTE_PTR_ADD(mlx5_mprq_buf_addr(buf, strd_n), offset);

	/*
	 * Memcpy packets to the target mbuf if:
	 * - The size of packet is smaller than mprq_max_memcpy_len.
	 * - Out of buffer in the Mempool for Multi-Packet RQ.
	 * - The packet's stride overlaps a headroom and scatter is off.
	 */
	if (len <= rxq->mprq_max_memcpy_len ||
	    rxq->mprq_repl == NULL ||
	    (hdrm_overlap > 0 && !rxq->strd_scatter_en)) {
		if (likely(len <=
			   (uint32_t)(pkt->buf_len - RTE_PKTMBUF_HEADROOM))) {
			rte_memcpy(rte_pktmbuf_mtod(pkt, void *),
				   addr, len);
			DATA_LEN(pkt) = len;
		} else if (rxq->strd_scatter_en) {
			struct rte_mbuf *prev = pkt;
			uint32_t seg_len = RTE_MIN(len, (uint32_t)
				(pkt->buf_len - RTE_PKTMBUF_HEADROOM));
			uint32_t rem_len = len - seg_len;

			rte_memcpy(rte_pktmbuf_mtod(pkt, void *),
				   addr, seg_len);
			DATA_LEN(pkt) = seg_len;
			while (rem_len) {
				struct rte_mbuf *next =
					rte_pktmbuf_alloc(rxq->mp);

				if (unlikely(next == NULL))
					return MLX5_RXQ_CODE_NOMBUF;
				NEXT(prev) = next;
				SET_DATA_OFF(next, 0);
				addr = RTE_PTR_ADD(addr, seg_len);
				seg_len = RTE_MIN(rem_len, (uint32_t)
					(next->buf_len - RTE_PKTMBUF_HEADROOM));
				rte_memcpy
					(rte_pktmbuf_mtod(next, void *),
					 addr, seg_len);
				DATA_LEN(next) = seg_len;
				rem_len -= seg_len;
				prev = next;
				++NB_SEGS(pkt);
			}
		} else {
			return MLX5_RXQ_CODE_DROPPED;
		}
	} else {
		rte_iova_t buf_iova;
		struct rte_mbuf_ext_shared_info *shinfo;
		uint16_t buf_len = strd_cnt * strd_sz;
		void *buf_addr;

		/* Increment the refcnt of the whole chunk. */
		__atomic_add_fetch(&buf->refcnt, 1, __ATOMIC_RELAXED);
		MLX5_ASSERT(__atomic_load_n(&buf->refcnt,
			    __ATOMIC_RELAXED) <= strd_n + 1);
		buf_addr = RTE_PTR_SUB(addr, RTE_PKTMBUF_HEADROOM);
		/*
		 * MLX5 device doesn't use iova but it is necessary in a
		 * case where the Rx packet is transmitted via a
		 * different PMD.
		 */
		buf_iova = rte_mempool_virt2iova(buf) +
			   RTE_PTR_DIFF(buf_addr, buf);
		shinfo = &buf->shinfos[strd_idx];
		rte_mbuf_ext_refcnt_set(shinfo, 1);
		/*
		 * EXT_ATTACHED_MBUF will be set to pkt->ol_flags when
		 * attaching the stride to mbuf and more offload flags
		 * will be added below by calling rxq_cq_to_mbuf().
		 * Other fields will be overwritten.
		 */
		rte_pktmbuf_attach_extbuf(pkt, buf_addr, buf_iova,
					  buf_len, shinfo);
		/* Set mbuf head-room. */
		SET_DATA_OFF(pkt, RTE_PKTMBUF_HEADROOM);
		MLX5_ASSERT(pkt->ol_flags == EXT_ATTACHED_MBUF);
		MLX5_ASSERT(rte_pktmbuf_tailroom(pkt) >=
			len - (hdrm_overlap > 0 ? hdrm_overlap : 0));
		DATA_LEN(pkt) = len;
		/*
		 * Copy the last fragment of a packet (up to headroom
		 * size bytes) in case there is a stride overlap with
		 * a next packet's headroom. Allocate a separate mbuf
		 * to store this fragment and link it. Scatter is on.
		 */
		if (hdrm_overlap > 0) {
			MLX5_ASSERT(rxq->strd_scatter_en);
			struct rte_mbuf *seg =
				rte_pktmbuf_alloc(rxq->mp);

			if (unlikely(seg == NULL))
				return MLX5_RXQ_CODE_NOMBUF;
			SET_DATA_OFF(seg, 0);
			rte_memcpy(rte_pktmbuf_mtod(seg, void *),
				RTE_PTR_ADD(addr, len - hdrm_overlap),
				hdrm_overlap);
			DATA_LEN(seg) = hdrm_overlap;
			DATA_LEN(pkt) = len - hdrm_overlap;
			NEXT(pkt) = seg;
			NB_SEGS(pkt) = 2;
		}
	}
	return MLX5_RXQ_CODE_EXIT;
}

#endif /* RTE_PMD_MLX5_RXTX_H_ */
