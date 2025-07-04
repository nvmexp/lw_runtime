/* SPDX-License-Identifier: BSD-3-Clause
 * Copyright(c) 2010-2017 Intel Corporation
 */

#ifndef _TESTPMD_H_
#define _TESTPMD_H_

#include <stdbool.h>

#include <rte_pci.h>
#include <rte_bus_pci.h>
#include <rte_gro.h>
#include <rte_gso.h>
#include <cmdline.h>
#include <sys/queue.h>

#define RTE_PORT_ALL            (~(portid_t)0x0)

#define RTE_TEST_RX_DESC_MAX    2048
#define RTE_TEST_TX_DESC_MAX    2048

#define RTE_PORT_STOPPED        (uint16_t)0
#define RTE_PORT_STARTED        (uint16_t)1
#define RTE_PORT_CLOSED         (uint16_t)2
#define RTE_PORT_HANDLING       (uint16_t)3

/*
 * It is used to allocate the memory for hash key.
 * The hash key size is NIC dependent.
 */
#define RSS_HASH_KEY_LENGTH 64

/*
 * Default size of the mbuf data buffer to receive standard 1518-byte
 * Ethernet frames in a mono-segment memory buffer.
 */
#define DEFAULT_MBUF_DATA_SIZE RTE_MBUF_DEFAULT_BUF_SIZE
/**< Default size of mbuf data buffer. */

/*
 * The maximum number of segments per packet is used when creating
 * scattered transmit packets composed of a list of mbufs.
 */
#define RTE_MAX_SEGS_PER_PKT 255 /**< nb_segs is a 8-bit unsigned char. */

/*
 * The maximum number of segments per packet is used to configure
 * buffer split feature, also specifies the maximum amount of
 * optional Rx pools to allocate mbufs to split.
 */
#define MAX_SEGS_BUFFER_SPLIT 8 /**< nb_segs is a 8-bit unsigned char. */

/* The prefix of the mbuf pool names created by the application. */
#define MBUF_POOL_NAME_PFX "mb_pool"

#define MAX_PKT_BURST 512
#define DEF_PKT_BURST 32

#define DEF_MBUF_CACHE 250

#define RTE_CACHE_LINE_SIZE_ROUNDUP(size) \
	(RTE_CACHE_LINE_SIZE * ((size + RTE_CACHE_LINE_SIZE - 1) / RTE_CACHE_LINE_SIZE))

#define NUMA_NO_CONFIG 0xFF
#define UMA_NO_CONFIG  0xFF

typedef uint8_t  lcoreid_t;
typedef uint16_t portid_t;
typedef uint16_t queueid_t;
typedef uint16_t streamid_t;

enum {
	PORT_TOPOLOGY_PAIRED,
	PORT_TOPOLOGY_CHAINED,
	PORT_TOPOLOGY_LOOP,
};

enum {
	MP_ALLOC_NATIVE, /**< allocate and populate mempool natively */
	MP_ALLOC_ANON,
	/**< allocate mempool natively, but populate using anonymous memory */
	MP_ALLOC_XMEM,
	/**< allocate and populate mempool using anonymous memory */
	MP_ALLOC_XMEM_HUGE,
	/**< allocate and populate mempool using anonymous hugepage memory */
	MP_ALLOC_XBUF
	/**< allocate mempool natively, use rte_pktmbuf_pool_create_extbuf */
};

/**
 * The data structure associated with RX and TX packet burst statistics
 * that are recorded for each forwarding stream.
 */
struct pkt_burst_stats {
	unsigned int pkt_burst_spread[MAX_PKT_BURST];
};

/** Information for a given RSS type. */
struct rss_type_info {
	const char *str; /**< Type name. */
	uint64_t rss_type; /**< Type value. */
};

/**
 * RSS type information table.
 *
 * An entry with a NULL type name terminates the list.
 */
extern const struct rss_type_info rss_type_table[];

/**
 * Dynf name array.
 *
 * Array that holds the name for each dynf.
 */
extern char dynf_names[64][RTE_MBUF_DYN_NAMESIZE];

/**
 * The data structure associated with a forwarding stream between a receive
 * port/queue and a transmit port/queue.
 */
struct fwd_stream {
	/* "read-only" data */
	portid_t   rx_port;   /**< port to poll for received packets */
	queueid_t  rx_queue;  /**< RX queue to poll on "rx_port" */
	portid_t   tx_port;   /**< forwarding port of received packets */
	queueid_t  tx_queue;  /**< TX queue to send forwarded packets */
	streamid_t peer_addr; /**< index of peer ethernet address of packets */

	unsigned int retry_enabled;

	/* "read-write" results */
	uint64_t rx_packets;  /**< received packets */
	uint64_t tx_packets;  /**< received packets transmitted */
	uint64_t fwd_dropped; /**< received packets not forwarded */
	uint64_t rx_bad_ip_csum ; /**< received packets has bad ip checksum */
	uint64_t rx_bad_l4_csum ; /**< received packets has bad l4 checksum */
	uint64_t rx_bad_outer_l4_csum;
	/**< received packets has bad outer l4 checksum */
	unsigned int gro_times;	/**< GRO operation times */
	uint64_t     core_cycles; /**< used for RX and TX processing */
	struct pkt_burst_stats rx_burst_stats;
	struct pkt_burst_stats tx_burst_stats;
};

/**
 * Age action context types, must be included inside the age action
 * context structure.
 */
enum age_action_context_type {
	ACTION_AGE_CONTEXT_TYPE_FLOW,
	ACTION_AGE_CONTEXT_TYPE_SHARED_ACTION,
};

/** Descriptor for a single flow. */
struct port_flow {
	struct port_flow *next; /**< Next flow in list. */
	struct port_flow *tmp; /**< Temporary linking. */
	uint32_t id; /**< Flow rule ID. */
	struct rte_flow *flow; /**< Opaque flow object returned by PMD. */
	struct rte_flow_colw_rule rule; /**< Saved flow rule description. */
	enum age_action_context_type age_type; /**< Age action context type. */
	uint8_t data[]; /**< Storage for flow rule description */
};

/* Descriptor for shared action */
struct port_shared_action {
	struct port_shared_action *next; /**< Next flow in list. */
	uint32_t id; /**< Shared action ID. */
	enum rte_flow_action_type type; /**< Action type. */
	struct rte_flow_shared_action *action;	/**< Shared action handle. */
	enum age_action_context_type age_type; /**< Age action context type. */
};

struct port_flow_tunnel {
	LIST_ENTRY(port_flow_tunnel) chain;
	struct rte_flow_action *pmd_actions;
	struct rte_flow_item   *pmd_items;
	uint32_t id;
	uint32_t num_pmd_actions;
	uint32_t num_pmd_items;
	struct rte_flow_tunnel tunnel;
	struct rte_flow_action *actions;
	struct rte_flow_item *items;
};

struct tunnel_ops {
	uint32_t id;
	char type[16];
	uint32_t enabled:1;
	uint32_t actions:1;
	uint32_t items:1;
};

/**
 * The data structure associated with each port.
 */
struct rte_port {
	struct rte_eth_dev_info dev_info;   /**< PCI info + driver name */
	struct rte_eth_conf     dev_conf;   /**< Port configuration. */
	struct rte_ether_addr       eth_addr;   /**< Port ethernet address */
	struct rte_eth_stats    stats;      /**< Last port statistics */
	unsigned int            socket_id;  /**< For NUMA support */
	uint16_t		parse_tunnel:1; /**< Parse internal headers */
	uint16_t                tso_segsz;  /**< Segmentation offload MSS for non-tunneled packets. */
	uint16_t                tunnel_tso_segsz; /**< Segmentation offload MSS for tunneled pkts. */
	uint16_t                tx_vlan_id;/**< The tag ID */
	uint16_t                tx_vlan_id_outer;/**< The outer tag ID */
	uint8_t                 tx_queue_stats_mapping_enabled;
	uint8_t                 rx_queue_stats_mapping_enabled;
	volatile uint16_t        port_status;    /**< port started or not */
	uint8_t                 need_setup;     /**< port just attached */
	uint8_t                 need_reconfig;  /**< need reconfiguring port or not */
	uint8_t                 need_reconfig_queues; /**< need reconfiguring queues or not */
	uint8_t                 rss_flag;   /**< enable rss or not */
	uint8_t                 dcb_flag;   /**< enable dcb */
	uint16_t                nb_rx_desc[RTE_MAX_QUEUES_PER_PORT+1]; /**< per queue rx desc number */
	uint16_t                nb_tx_desc[RTE_MAX_QUEUES_PER_PORT+1]; /**< per queue tx desc number */
	struct rte_eth_rxconf   rx_conf[RTE_MAX_QUEUES_PER_PORT+1]; /**< per queue rx configuration */
	struct rte_eth_txconf   tx_conf[RTE_MAX_QUEUES_PER_PORT+1]; /**< per queue tx configuration */
	struct rte_ether_addr   *mc_addr_pool; /**< pool of multicast addrs */
	uint32_t                mc_addr_nb; /**< nb. of addr. in mc_addr_pool */
	uint8_t                 slave_flag; /**< bonding slave port */
	struct port_flow        *flow_list; /**< Associated flows. */
	struct port_shared_action *actions_list;
	/**< Associated shared actions. */
	LIST_HEAD(, port_flow_tunnel) flow_tunnel_list;
	const struct rte_eth_rxtx_callback *rx_dump_cb[RTE_MAX_QUEUES_PER_PORT+1];
	const struct rte_eth_rxtx_callback *tx_dump_cb[RTE_MAX_QUEUES_PER_PORT+1];
	/**< metadata value to insert in Tx packets. */
	uint32_t		tx_metadata;
	const struct rte_eth_rxtx_callback *tx_set_md_cb[RTE_MAX_QUEUES_PER_PORT+1];
	/**< dynamic flags. */
	uint64_t		mbuf_dynf;
	const struct rte_eth_rxtx_callback *tx_set_dynf_cb[RTE_MAX_QUEUES_PER_PORT+1];
};

/**
 * The data structure associated with each forwarding logical core.
 * The logical cores are internally numbered by a core index from 0 to
 * the maximum number of logical cores - 1.
 * The system CPU identifier of all logical cores are setup in a global
 * CPU id. configuration table.
 */
struct fwd_lcore {
	struct rte_gso_ctx gso_ctx;     /**< GSO context */
	struct rte_mempool *mbp; /**< The mbuf pool to use by this core */
	void *gro_ctx;		/**< GRO context */
	streamid_t stream_idx;   /**< index of 1st stream in "fwd_streams" */
	streamid_t stream_nb;    /**< number of streams in "fwd_streams" */
	lcoreid_t  cpuid_idx;    /**< index of logical core in CPU id table */
	queueid_t  tx_queue;     /**< TX queue to send forwarded packets */
	volatile char stopped;   /**< stop forwarding when set */
};

/*
 * Forwarding mode operations:
 *   - IO forwarding mode (default mode)
 *     Forwards packets unchanged.
 *
 *   - MAC forwarding mode
 *     Set the source and the destination Ethernet addresses of packets
 *     before forwarding them.
 *
 *   - IEEE1588 forwarding mode
 *     Check that received IEEE1588 Precise Time Protocol (PTP) packets are
 *     filtered and timestamped by the hardware.
 *     Forwards packets unchanged on the same port.
 *     Check that sent IEEE1588 PTP packets are timestamped by the hardware.
 */
typedef void (*port_fwd_begin_t)(portid_t pi);
typedef void (*port_fwd_end_t)(portid_t pi);
typedef void (*packet_fwd_t)(struct fwd_stream *fs);

struct fwd_engine {
	const char       *fwd_mode_name; /**< Forwarding mode name. */
	port_fwd_begin_t port_fwd_begin; /**< NULL if nothing special to do. */
	port_fwd_end_t   port_fwd_end;   /**< NULL if nothing special to do. */
	packet_fwd_t     packet_fwd;     /**< Mandatory. */
};

#define BURST_TX_WAIT_US 1
#define BURST_TX_RETRIES 64

extern uint32_t burst_tx_delay_time;
extern uint32_t burst_tx_retry_num;

extern struct fwd_engine io_fwd_engine;
extern struct fwd_engine mac_fwd_engine;
extern struct fwd_engine mac_swap_engine;
extern struct fwd_engine flow_gen_engine;
extern struct fwd_engine rx_only_engine;
extern struct fwd_engine tx_only_engine;
extern struct fwd_engine csum_fwd_engine;
extern struct fwd_engine icmp_echo_engine;
extern struct fwd_engine noisy_vnf_engine;
extern struct fwd_engine five_tuple_swap_fwd_engine;
#ifdef RTE_LIBRTE_IEEE1588
extern struct fwd_engine ieee1588_fwd_engine;
#endif

extern struct fwd_engine * fwd_engines[]; /**< NULL terminated array. */
extern cmdline_parse_inst_t cmd_set_raw;
extern cmdline_parse_inst_t cmd_show_set_raw;
extern cmdline_parse_inst_t cmd_show_set_raw_all;

extern uint16_t mempool_flags;

/**
 * Forwarding Configuration
 *
 */
struct fwd_config {
	struct fwd_engine *fwd_eng; /**< Packet forwarding mode. */
	streamid_t nb_fwd_streams;  /**< Nb. of forward streams to process. */
	lcoreid_t  nb_fwd_lcores;   /**< Nb. of logical cores to launch. */
	portid_t   nb_fwd_ports;    /**< Nb. of ports ilwolved. */
};

/**
 * DCB mode enable
 */
enum dcb_mode_enable
{
	DCB_VT_ENABLED,
	DCB_ENABLED
};

#define MAX_TX_QUEUE_STATS_MAPPINGS 1024 /* MAX_PORT of 32 @ 32 tx_queues/port */
#define MAX_RX_QUEUE_STATS_MAPPINGS 4096 /* MAX_PORT of 32 @ 128 rx_queues/port */

struct queue_stats_mappings {
	portid_t port_id;
	uint16_t queue_id;
	uint8_t stats_counter_id;
} __rte_cache_aligned;

extern struct queue_stats_mappings tx_queue_stats_mappings_array[];
extern struct queue_stats_mappings rx_queue_stats_mappings_array[];

/* Assign both tx and rx queue stats mappings to the same default values */
extern struct queue_stats_mappings *tx_queue_stats_mappings;
extern struct queue_stats_mappings *rx_queue_stats_mappings;

extern uint16_t nb_tx_queue_stats_mappings;
extern uint16_t nb_rx_queue_stats_mappings;

extern uint8_t xstats_hide_zero; /**< Hide zero values for xstats display */

/* globals used for configuration */
extern uint8_t record_core_cycles; /**< Enables measurement of CPU cycles */
extern uint8_t record_burst_stats; /**< Enables display of RX and TX bursts */
extern uint16_t verbose_level; /**< Drives messages being displayed, if any. */
extern int testpmd_logtype; /**< Log type for testpmd logs */
extern uint8_t  interactive;
extern uint8_t  auto_start;
extern uint8_t  tx_first;
extern char cmdline_filename[PATH_MAX]; /**< offline commands file */
extern uint8_t  numa_support; /**< set by "--numa" parameter */
extern uint16_t port_topology; /**< set by "--port-topology" parameter */
extern uint8_t no_flush_rx; /**<set by "--no-flush-rx" parameter */
extern uint8_t flow_isolate_all; /**< set by "--flow-isolate-all */
extern uint8_t  mp_alloc_type;
/**< set by "--mp-anon" or "--mp-alloc" parameter */
extern uint8_t no_link_check; /**<set by "--disable-link-check" parameter */
extern uint8_t no_device_start; /**<set by "--disable-device-start" parameter */
extern volatile int test_done; /* stop packet forwarding when set to 1. */
extern uint8_t lsc_interrupt; /**< disabled by "--no-lsc-interrupt" parameter */
extern uint8_t rmv_interrupt; /**< disabled by "--no-rmv-interrupt" parameter */
extern uint32_t event_print_mask;
/**< set by "--print-event xxxx" and "--mask-event xxxx parameters */
extern bool setup_on_probe_event; /**< disabled by port setup-on iterator */
extern uint8_t hot_plug; /**< enable by "--hot-plug" parameter */
extern int do_mlockall; /**< set by "--mlockall" or "--no-mlockall" parameter */
extern uint8_t clear_ptypes; /**< disabled by set ptype cmd */

#ifdef RTE_LIBRTE_IXGBE_BYPASS
extern uint32_t bypass_timeout; /**< Store the NIC bypass watchdog timeout */
#endif

/*
 * Store specified sockets on which memory pool to be used by ports
 * is allocated.
 */
extern uint8_t port_numa[RTE_MAX_ETHPORTS];

/*
 * Store specified sockets on which RX ring to be used by ports
 * is allocated.
 */
extern uint8_t rxring_numa[RTE_MAX_ETHPORTS];

/*
 * Store specified sockets on which TX ring to be used by ports
 * is allocated.
 */
extern uint8_t txring_numa[RTE_MAX_ETHPORTS];

extern uint8_t socket_num;

/*
 * Configuration of logical cores:
 * nb_fwd_lcores <= nb_cfg_lcores <= nb_lcores
 */
extern lcoreid_t nb_lcores; /**< Number of logical cores probed at init time. */
extern lcoreid_t nb_cfg_lcores; /**< Number of configured logical cores. */
extern lcoreid_t nb_fwd_lcores; /**< Number of forwarding logical cores. */
extern unsigned int fwd_lcores_cpuids[RTE_MAX_LCORE];
extern unsigned int num_sockets;
extern unsigned int socket_ids[RTE_MAX_NUMA_NODES];

/*
 * Configuration of Ethernet ports:
 * nb_fwd_ports <= nb_cfg_ports <= nb_ports
 */
extern portid_t nb_ports; /**< Number of ethernet ports probed at init time. */
extern portid_t nb_cfg_ports; /**< Number of configured ports. */
extern portid_t nb_fwd_ports; /**< Number of forwarding ports. */
extern portid_t fwd_ports_ids[RTE_MAX_ETHPORTS];
extern struct rte_port *ports;

extern struct rte_eth_rxmode rx_mode;
extern struct rte_eth_txmode tx_mode;

extern uint64_t rss_hf;

extern queueid_t nb_hairpinq;
extern queueid_t nb_rxq;
extern queueid_t nb_txq;

extern uint16_t nb_rxd;
extern uint16_t nb_txd;

extern int16_t rx_free_thresh;
extern int8_t rx_drop_en;
extern int16_t tx_free_thresh;
extern int16_t tx_rs_thresh;

extern uint16_t noisy_tx_sw_bufsz;
extern uint16_t noisy_tx_sw_buf_flush_time;
extern uint64_t noisy_lkup_mem_sz;
extern uint64_t noisy_lkup_num_writes;
extern uint64_t noisy_lkup_num_reads;
extern uint64_t noisy_lkup_num_reads_writes;

extern uint8_t dcb_config;
extern uint8_t dcb_test;

extern uint32_t mbuf_data_size_n;
extern uint16_t mbuf_data_size[MAX_SEGS_BUFFER_SPLIT];
/**< Mbuf data space size. */
enum mbuf_mem_type {
	MBUF_MEM_CPU,
	MBUF_MEM_GPU
};
extern enum mbuf_mem_type mbuf_mem_types[MAX_SEGS_BUFFER_SPLIT];
extern uint32_t param_total_num_mbufs;

extern uint16_t stats_period;

extern uint16_t hairpin_mode;

#ifdef RTE_LIB_LATENCYSTATS
extern uint8_t latencystats_enabled;
extern lcoreid_t latencystats_lcore_id;
#endif

#ifdef RTE_LIB_BITRATESTATS
extern lcoreid_t bitrate_lcore_id;
extern uint8_t bitrate_enabled;
#endif

extern struct rte_fdir_conf fdir_conf;

/*
 * Configuration of packet segments used to scatter received packets
 * if some of split features is configured.
 */
extern uint16_t rx_pkt_seg_lengths[MAX_SEGS_BUFFER_SPLIT];
extern uint8_t  rx_pkt_nb_segs; /**< Number of segments to split */
extern uint16_t rx_pkt_seg_offsets[MAX_SEGS_BUFFER_SPLIT];
extern uint8_t  rx_pkt_nb_offs; /**< Number of specified offsets */

/*
 * Configuration of packet segments used by the "txonly" processing engine.
 */
#define TXONLY_DEF_PACKET_LEN 64
extern uint16_t tx_pkt_length; /**< Length of TXONLY packet */
extern uint16_t tx_pkt_seg_lengths[RTE_MAX_SEGS_PER_PKT]; /**< Seg. lengths */
extern uint8_t  tx_pkt_nb_segs; /**< Number of segments in TX packets */
extern uint32_t tx_pkt_times_intra;
extern uint32_t tx_pkt_times_inter;

enum tx_pkt_split {
	TX_PKT_SPLIT_OFF,
	TX_PKT_SPLIT_ON,
	TX_PKT_SPLIT_RND,
};

extern enum tx_pkt_split tx_pkt_split;

extern uint8_t txonly_multi_flow;

extern uint16_t nb_pkt_per_burst;
extern uint16_t mb_mempool_cache;
extern int8_t rx_pthresh;
extern int8_t rx_hthresh;
extern int8_t rx_wthresh;
extern int8_t tx_pthresh;
extern int8_t tx_hthresh;
extern int8_t tx_wthresh;

extern uint16_t tx_udp_src_port;
extern uint16_t tx_udp_dst_port;

extern uint32_t tx_ip_src_addr;
extern uint32_t tx_ip_dst_addr;

extern struct fwd_config lwr_fwd_config;
extern struct fwd_engine *lwr_fwd_eng;
extern uint32_t retry_enabled;
extern struct fwd_lcore  **fwd_lcores;
extern struct fwd_stream **fwd_streams;

extern uint16_t vxlan_gpe_udp_port; /**< UDP port of tunnel VXLAN-GPE. */
extern uint16_t geneve_udp_port; /**< UDP port of tunnel GENEVE. */

extern portid_t nb_peer_eth_addrs; /**< Number of peer ethernet addresses. */
extern struct rte_ether_addr peer_eth_addrs[RTE_MAX_ETHPORTS];

extern uint32_t burst_tx_delay_time; /**< Burst tx delay time(us) for mac-retry. */
extern uint32_t burst_tx_retry_num;  /**< Burst tx retry number for mac-retry. */

#define GRO_DEFAULT_ITEM_NUM_PER_FLOW 32
#define GRO_DEFAULT_FLOW_NUM (RTE_GRO_MAX_BURST_ITEM_NUM / \
		GRO_DEFAULT_ITEM_NUM_PER_FLOW)

#define GRO_DEFAULT_FLUSH_CYCLES 1
#define GRO_MAX_FLUSH_CYCLES 4

struct gro_status {
	struct rte_gro_param param;
	uint8_t enable;
};
extern struct gro_status gro_ports[RTE_MAX_ETHPORTS];
extern uint8_t gro_flush_cycles;

#define GSO_MAX_PKT_BURST 2048
struct gso_status {
	uint8_t enable;
};
extern struct gso_status gso_ports[RTE_MAX_ETHPORTS];
extern uint16_t gso_max_segment_size;

/* VXLAN encap/decap parameters. */
struct vxlan_encap_conf {
	uint32_t select_ipv4:1;
	uint32_t select_vlan:1;
	uint32_t select_tos_ttl:1;
	uint8_t vni[3];
	rte_be16_t udp_src;
	rte_be16_t udp_dst;
	rte_be32_t ipv4_src;
	rte_be32_t ipv4_dst;
	uint8_t ipv6_src[16];
	uint8_t ipv6_dst[16];
	rte_be16_t vlan_tci;
	uint8_t ip_tos;
	uint8_t ip_ttl;
	uint8_t eth_src[RTE_ETHER_ADDR_LEN];
	uint8_t eth_dst[RTE_ETHER_ADDR_LEN];
};

extern struct vxlan_encap_conf vxlan_encap_conf;

/* LWGRE encap/decap parameters. */
struct lwgre_encap_conf {
	uint32_t select_ipv4:1;
	uint32_t select_vlan:1;
	uint8_t tni[3];
	rte_be32_t ipv4_src;
	rte_be32_t ipv4_dst;
	uint8_t ipv6_src[16];
	uint8_t ipv6_dst[16];
	rte_be16_t vlan_tci;
	uint8_t eth_src[RTE_ETHER_ADDR_LEN];
	uint8_t eth_dst[RTE_ETHER_ADDR_LEN];
};

extern struct lwgre_encap_conf lwgre_encap_conf;

/* L2 encap parameters. */
struct l2_encap_conf {
	uint32_t select_ipv4:1;
	uint32_t select_vlan:1;
	rte_be16_t vlan_tci;
	uint8_t eth_src[RTE_ETHER_ADDR_LEN];
	uint8_t eth_dst[RTE_ETHER_ADDR_LEN];
};
extern struct l2_encap_conf l2_encap_conf;

/* L2 decap parameters. */
struct l2_decap_conf {
	uint32_t select_vlan:1;
};
extern struct l2_decap_conf l2_decap_conf;

/* MPLSoGRE encap parameters. */
struct mplsogre_encap_conf {
	uint32_t select_ipv4:1;
	uint32_t select_vlan:1;
	uint8_t label[3];
	rte_be32_t ipv4_src;
	rte_be32_t ipv4_dst;
	uint8_t ipv6_src[16];
	uint8_t ipv6_dst[16];
	rte_be16_t vlan_tci;
	uint8_t eth_src[RTE_ETHER_ADDR_LEN];
	uint8_t eth_dst[RTE_ETHER_ADDR_LEN];
};
extern struct mplsogre_encap_conf mplsogre_encap_conf;

/* MPLSoGRE decap parameters. */
struct mplsogre_decap_conf {
	uint32_t select_ipv4:1;
	uint32_t select_vlan:1;
};
extern struct mplsogre_decap_conf mplsogre_decap_conf;

/* MPLSoUDP encap parameters. */
struct mplsoudp_encap_conf {
	uint32_t select_ipv4:1;
	uint32_t select_vlan:1;
	uint8_t label[3];
	rte_be16_t udp_src;
	rte_be16_t udp_dst;
	rte_be32_t ipv4_src;
	rte_be32_t ipv4_dst;
	uint8_t ipv6_src[16];
	uint8_t ipv6_dst[16];
	rte_be16_t vlan_tci;
	uint8_t eth_src[RTE_ETHER_ADDR_LEN];
	uint8_t eth_dst[RTE_ETHER_ADDR_LEN];
};
extern struct mplsoudp_encap_conf mplsoudp_encap_conf;

/* MPLSoUDP decap parameters. */
struct mplsoudp_decap_conf {
	uint32_t select_ipv4:1;
	uint32_t select_vlan:1;
};
extern struct mplsoudp_decap_conf mplsoudp_decap_conf;

extern enum rte_eth_rx_mq_mode rx_mq_mode;

static inline unsigned int
lcore_num(void)
{
	unsigned int i;

	for (i = 0; i < RTE_MAX_LCORE; ++i)
		if (fwd_lcores_cpuids[i] == rte_lcore_id())
			return i;

	rte_panic("lcore_id of current thread not found in fwd_lcores_cpuids\n");
}

void
parse_fwd_portlist(const char *port);

static inline struct fwd_lcore *
lwrrent_fwd_lcore(void)
{
	return fwd_lcores[lcore_num()];
}

/* Mbuf Pools */
static inline void
mbuf_poolname_build(unsigned int sock_id, char *mp_name,
		    int name_size, uint16_t idx, enum mbuf_mem_type mem_type)
{
	const char *suffix = mem_type == MBUF_MEM_GPU ? "_gpu" : "";

	if (!idx)
		snprintf(mp_name, name_size,
			 MBUF_POOL_NAME_PFX "_%u%s", sock_id, suffix);
	else
		snprintf(mp_name, name_size,
			 MBUF_POOL_NAME_PFX "_%hu_%hu%s", (uint16_t)sock_id, idx, suffix);
}

static inline struct rte_mempool *
mbuf_pool_find(unsigned int sock_id, uint16_t idx, enum mbuf_mem_type mem_type)
{
	char pool_name[RTE_MEMPOOL_NAMESIZE];

	mbuf_poolname_build(sock_id, pool_name, sizeof(pool_name), idx, mem_type);
	return rte_mempool_lookup((const char *)pool_name);
}

/**
 * Read/Write operations on a PCI register of a port.
 */
static inline uint32_t
port_pci_reg_read(struct rte_port *port, uint32_t reg_off)
{
	const struct rte_pci_device *pci_dev;
	const struct rte_bus *bus;
	void *reg_addr;
	uint32_t reg_v;

	if (!port->dev_info.device) {
		printf("Invalid device\n");
		return 0;
	}

	bus = rte_bus_find_by_device(port->dev_info.device);
	if (bus && !strcmp(bus->name, "pci")) {
		pci_dev = RTE_DEV_TO_PCI(port->dev_info.device);
	} else {
		printf("Not a PCI device\n");
		return 0;
	}

	reg_addr = ((char *)pci_dev->mem_resource[0].addr + reg_off);
	reg_v = *((volatile uint32_t *)reg_addr);
	return rte_le_to_cpu_32(reg_v);
}

#define port_id_pci_reg_read(pt_id, reg_off) \
	port_pci_reg_read(&ports[(pt_id)], (reg_off))

static inline void
port_pci_reg_write(struct rte_port *port, uint32_t reg_off, uint32_t reg_v)
{
	const struct rte_pci_device *pci_dev;
	const struct rte_bus *bus;
	void *reg_addr;

	if (!port->dev_info.device) {
		printf("Invalid device\n");
		return;
	}

	bus = rte_bus_find_by_device(port->dev_info.device);
	if (bus && !strcmp(bus->name, "pci")) {
		pci_dev = RTE_DEV_TO_PCI(port->dev_info.device);
	} else {
		printf("Not a PCI device\n");
		return;
	}

	reg_addr = ((char *)pci_dev->mem_resource[0].addr + reg_off);
	*((volatile uint32_t *)reg_addr) = rte_cpu_to_le_32(reg_v);
}

#define port_id_pci_reg_write(pt_id, reg_off, reg_value) \
	port_pci_reg_write(&ports[(pt_id)], (reg_off), (reg_value))

static inline void
get_start_cycles(uint64_t *start_tsc)
{
	if (record_core_cycles)
		*start_tsc = rte_rdtsc();
}

static inline void
get_end_cycles(struct fwd_stream *fs, uint64_t start_tsc)
{
	if (record_core_cycles)
		fs->core_cycles += rte_rdtsc() - start_tsc;
}

static inline void
inc_rx_burst_stats(struct fwd_stream *fs, uint16_t nb_rx)
{
	if (record_burst_stats)
		fs->rx_burst_stats.pkt_burst_spread[nb_rx]++;
}

static inline void
inc_tx_burst_stats(struct fwd_stream *fs, uint16_t nb_tx)
{
	if (record_burst_stats)
		fs->tx_burst_stats.pkt_burst_spread[nb_tx]++;
}

/* Prototypes */
unsigned int parse_item_list(char* str, const char* item_name,
			unsigned int max_items,
			unsigned int *parsed_items, int check_unique_values);
unsigned int parse_buffer_list(char* str, const char* item_name,
			unsigned int max_items,
			unsigned int *parsed_items, int check_unique_values);
void launch_args_parse(int argc, char** argv);
void cmdline_read_from_file(const char *filename);
void prompt(void);
void prompt_exit(void);
void nic_stats_display(portid_t port_id);
void nic_stats_clear(portid_t port_id);
void nic_xstats_display(portid_t port_id);
void nic_xstats_clear(portid_t port_id);
void nic_stats_mapping_display(portid_t port_id);
void device_infos_display(const char *identifier);
void port_infos_display(portid_t port_id);
void port_summary_display(portid_t port_id);
void port_eeprom_display(portid_t port_id);
void port_module_eeprom_display(portid_t port_id);
void port_summary_header_display(void);
void port_offload_cap_display(portid_t port_id);
void rx_queue_infos_display(portid_t port_idi, uint16_t queue_id);
void tx_queue_infos_display(portid_t port_idi, uint16_t queue_id);
void fwd_lcores_config_display(void);
void pkt_fwd_config_display(struct fwd_config *cfg);
void rxtx_config_display(void);
void fwd_config_setup(void);
void set_def_fwd_config(void);
void reconfig(portid_t new_port_id, unsigned socket_id);
int init_fwd_streams(void);
void update_fwd_ports(portid_t new_pid);

void set_fwd_eth_peer(portid_t port_id, char *peer_addr);

void port_mtu_set(portid_t port_id, uint16_t mtu);
void port_reg_bit_display(portid_t port_id, uint32_t reg_off, uint8_t bit_pos);
void port_reg_bit_set(portid_t port_id, uint32_t reg_off, uint8_t bit_pos,
		      uint8_t bit_v);
void port_reg_bit_field_display(portid_t port_id, uint32_t reg_off,
				uint8_t bit1_pos, uint8_t bit2_pos);
void port_reg_bit_field_set(portid_t port_id, uint32_t reg_off,
			    uint8_t bit1_pos, uint8_t bit2_pos, uint32_t value);
void port_reg_display(portid_t port_id, uint32_t reg_off);
void port_reg_set(portid_t port_id, uint32_t reg_off, uint32_t value);
int port_shared_action_create(portid_t port_id, uint32_t id,
			      const struct rte_flow_shared_action_conf *conf,
			      const struct rte_flow_action *action);
int port_shared_action_destroy(portid_t port_id,
			       uint32_t n, const uint32_t *action);
struct rte_flow_shared_action *port_shared_action_get_by_id(portid_t port_id,
							    uint32_t id);
int port_shared_action_update(portid_t port_id, uint32_t id,
			      const struct rte_flow_action *action);
int port_flow_validate(portid_t port_id,
		       const struct rte_flow_attr *attr,
		       const struct rte_flow_item *pattern,
		       const struct rte_flow_action *actions,
		       const struct tunnel_ops *tunnel_ops);
int port_flow_create(portid_t port_id,
		     const struct rte_flow_attr *attr,
		     const struct rte_flow_item *pattern,
		     const struct rte_flow_action *actions,
		     const struct tunnel_ops *tunnel_ops);
int port_shared_action_query(portid_t port_id, uint32_t id);
void update_age_action_context(const struct rte_flow_action *actions,
		     struct port_flow *pf);
int port_flow_destroy(portid_t port_id, uint32_t n, const uint32_t *rule);
int port_flow_flush(portid_t port_id);
int port_flow_dump(portid_t port_id, const char *file_name);
int port_flow_query(portid_t port_id, uint32_t rule,
		    const struct rte_flow_action *action);
void port_flow_list(portid_t port_id, uint32_t n, const uint32_t *group);
void port_flow_aged(portid_t port_id, uint8_t destroy);
const char *port_flow_tunnel_type(struct rte_flow_tunnel *tunnel);
struct port_flow_tunnel *
port_flow_locate_tunnel(uint16_t port_id, struct rte_flow_tunnel *tun);
void port_flow_tunnel_list(portid_t port_id);
void port_flow_tunnel_destroy(portid_t port_id, uint32_t tunnel_id);
void port_flow_tunnel_create(portid_t port_id, const struct tunnel_ops *ops);
int port_flow_isolate(portid_t port_id, int set);

void rx_ring_desc_display(portid_t port_id, queueid_t rxq_id, uint16_t rxd_id);
void tx_ring_desc_display(portid_t port_id, queueid_t txq_id, uint16_t txd_id);

int set_fwd_lcores_list(unsigned int *lcorelist, unsigned int nb_lc);
int set_fwd_lcores_mask(uint64_t lcoremask);
void set_fwd_lcores_number(uint16_t nb_lc);

void set_fwd_ports_list(unsigned int *portlist, unsigned int nb_pt);
void set_fwd_ports_mask(uint64_t portmask);
void set_fwd_ports_number(uint16_t nb_pt);
int port_is_forwarding(portid_t port_id);

void rx_vlan_strip_set(portid_t port_id, int on);
void rx_vlan_strip_set_on_queue(portid_t port_id, uint16_t queue_id, int on);

void rx_vlan_filter_set(portid_t port_id, int on);
void rx_vlan_all_filter_set(portid_t port_id, int on);
void rx_vlan_qinq_strip_set(portid_t port_id, int on);
int rx_vft_set(portid_t port_id, uint16_t vlan_id, int on);
void vlan_extend_set(portid_t port_id, int on);
void vlan_tpid_set(portid_t port_id, enum rte_vlan_type vlan_type,
		   uint16_t tp_id);
void tx_vlan_set(portid_t port_id, uint16_t vlan_id);
void tx_qinq_set(portid_t port_id, uint16_t vlan_id, uint16_t vlan_id_outer);
void tx_vlan_reset(portid_t port_id);
void tx_vlan_pvid_set(portid_t port_id, uint16_t vlan_id, int on);

void set_qmap(portid_t port_id, uint8_t is_rx, uint16_t queue_id, uint8_t map_value);

void set_xstats_hide_zero(uint8_t on_off);

void set_record_core_cycles(uint8_t on_off);
void set_record_burst_stats(uint8_t on_off);
void set_verbose_level(uint16_t vb_level);
void set_rx_pkt_segments(unsigned int *seg_lengths, unsigned int nb_segs);
void show_rx_pkt_segments(void);
void set_rx_pkt_offsets(unsigned int *seg_offsets, unsigned int nb_offs);
void show_rx_pkt_offsets(void);
void set_tx_pkt_segments(unsigned int *seg_lengths, unsigned int nb_segs);
void show_tx_pkt_segments(void);
void set_tx_pkt_times(unsigned int *tx_times);
void show_tx_pkt_times(void);
void set_tx_pkt_split(const char *name);
int parse_fec_mode(const char *name, enum rte_eth_fec_mode *mode);
void show_fec_capability(uint32_t num, struct rte_eth_fec_capa *speed_fec_capa);
void set_nb_pkt_per_burst(uint16_t pkt_burst);
char *list_pkt_forwarding_modes(void);
char *list_pkt_forwarding_retry_modes(void);
void set_pkt_forwarding_mode(const char *fwd_mode);
void start_packet_forwarding(int with_tx_first);
void fwd_stats_display(void);
void fwd_stats_reset(void);
void stop_packet_forwarding(void);
void dev_set_link_up(portid_t pid);
void dev_set_link_down(portid_t pid);
void init_port_config(void);
void set_port_slave_flag(portid_t slave_pid);
void clear_port_slave_flag(portid_t slave_pid);
uint8_t port_is_bonding_slave(portid_t slave_pid);

int init_port_dcb_config(portid_t pid, enum dcb_mode_enable dcb_mode,
		     enum rte_eth_nb_tcs num_tcs,
		     uint8_t pfc_en);
int start_port(portid_t pid);
void stop_port(portid_t pid);
void close_port(portid_t pid);
void reset_port(portid_t pid);
void attach_port(char *identifier);
void detach_devargs(char *identifier);
void detach_port_device(portid_t port_id);
int all_ports_stopped(void);
int port_is_stopped(portid_t port_id);
int port_is_started(portid_t port_id);
void pmd_test_exit(void);
#if defined(RTE_NET_I40E) || defined(RTE_NET_IXGBE)
void fdir_get_infos(portid_t port_id);
#endif
void fdir_set_flex_mask(portid_t port_id,
			   struct rte_eth_fdir_flex_mask *cfg);
void fdir_set_flex_payload(portid_t port_id,
			   struct rte_eth_flex_payload_cfg *cfg);
void port_rss_reta_info(portid_t port_id,
			struct rte_eth_rss_reta_entry64 *reta_conf,
			uint16_t nb_entries);

void set_vf_traffic(portid_t port_id, uint8_t is_rx, uint16_t vf, uint8_t on);

int
rx_queue_setup(uint16_t port_id, uint16_t rx_queue_id,
	       uint16_t nb_rx_desc, unsigned int socket_id,
	       struct rte_eth_rxconf *rx_conf, struct rte_mempool *mp);

int set_queue_rate_limit(portid_t port_id, uint16_t queue_idx, uint16_t rate);
int set_vf_rate_limit(portid_t port_id, uint16_t vf, uint16_t rate,
				uint64_t q_msk);

void port_rss_hash_conf_show(portid_t port_id, int show_rss_key);
void port_rss_hash_key_update(portid_t port_id, char rss_type[],
			      uint8_t *hash_key, uint hash_key_len);
int rx_queue_id_is_ilwalid(queueid_t rxq_id);
int tx_queue_id_is_ilwalid(queueid_t txq_id);
void setup_gro(const char *onoff, portid_t port_id);
void setup_gro_flush_cycles(uint8_t cycles);
void show_gro(portid_t port_id);
void setup_gso(const char *mode, portid_t port_id);
int eth_dev_info_get_print_err(uint16_t port_id,
			struct rte_eth_dev_info *dev_info);
void eth_set_promisc_mode(uint16_t port_id, int enable);
void eth_set_allmulticast_mode(uint16_t port, int enable);
int eth_link_get_nowait_print_err(uint16_t port_id, struct rte_eth_link *link);
int eth_macaddr_get_print_err(uint16_t port_id,
			struct rte_ether_addr *mac_addr);

/* Functions to display the set of MAC addresses added to a port*/
void show_macs(portid_t port_id);
void show_mcast_macs(portid_t port_id);

/* Functions to manage the set of filtered Multicast MAC addresses */
void mcast_addr_add(portid_t port_id, struct rte_ether_addr *mc_addr);
void mcast_addr_remove(portid_t port_id, struct rte_ether_addr *mc_addr);
void port_dcb_info_display(portid_t port_id);

uint8_t *open_file(const char *file_path, uint32_t *size);
int save_file(const char *file_path, uint8_t *buf, uint32_t size);
int close_file(uint8_t *buf);

void port_queue_region_info_display(portid_t port_id, void *buf);

enum print_warning {
	ENABLED_WARN = 0,
	DISABLED_WARN
};
int port_id_is_ilwalid(portid_t port_id, enum print_warning warning);
void print_valid_ports(void);
int new_socket_id(unsigned int socket_id);

queueid_t get_allowed_max_nb_rxq(portid_t *pid);
int check_nb_rxq(queueid_t rxq);
queueid_t get_allowed_max_nb_txq(portid_t *pid);
int check_nb_txq(queueid_t txq);
int check_nb_rxd(queueid_t rxd);
int check_nb_txd(queueid_t txd);
queueid_t get_allowed_max_nb_hairpinq(portid_t *pid);
int check_nb_hairpinq(queueid_t hairpinq);

uint16_t dump_rx_pkts(uint16_t port_id, uint16_t queue, struct rte_mbuf *pkts[],
		      uint16_t nb_pkts, __rte_unused uint16_t max_pkts,
		      __rte_unused void *user_param);

uint16_t dump_tx_pkts(uint16_t port_id, uint16_t queue, struct rte_mbuf *pkts[],
		      uint16_t nb_pkts, __rte_unused void *user_param);

void add_rx_dump_callbacks(portid_t portid);
void remove_rx_dump_callbacks(portid_t portid);
void add_tx_dump_callbacks(portid_t portid);
void remove_tx_dump_callbacks(portid_t portid);
void configure_rxtx_dump_callbacks(uint16_t verbose);

uint16_t tx_pkt_set_md(uint16_t port_id, __rte_unused uint16_t queue,
		       struct rte_mbuf *pkts[], uint16_t nb_pkts,
		       __rte_unused void *user_param);
void add_tx_md_callback(portid_t portid);
void remove_tx_md_callback(portid_t portid);

uint16_t tx_pkt_set_dynf(uint16_t port_id, __rte_unused uint16_t queue,
			 struct rte_mbuf *pkts[], uint16_t nb_pkts,
			 __rte_unused void *user_param);
void add_tx_dynf_callback(portid_t portid);
void remove_tx_dynf_callback(portid_t portid);

/*
 * Work-around of a compilation error with ICC on ilwocations of the
 * rte_be_to_cpu_16() function.
 */
#ifdef __GCC__
#define RTE_BE_TO_CPU_16(be_16_v)  rte_be_to_cpu_16((be_16_v))
#define RTE_CPU_TO_BE_16(cpu_16_v) rte_cpu_to_be_16((cpu_16_v))
#else
#if RTE_BYTE_ORDER == RTE_BIG_ENDIAN
#define RTE_BE_TO_CPU_16(be_16_v)  (be_16_v)
#define RTE_CPU_TO_BE_16(cpu_16_v) (cpu_16_v)
#else
#define RTE_BE_TO_CPU_16(be_16_v) \
	(uint16_t) ((((be_16_v) & 0xFF) << 8) | ((be_16_v) >> 8))
#define RTE_CPU_TO_BE_16(cpu_16_v) \
	(uint16_t) ((((cpu_16_v) & 0xFF) << 8) | ((cpu_16_v) >> 8))
#endif
#endif /* __GCC__ */

#define TESTPMD_LOG(level, fmt, args...) \
	rte_log(RTE_LOG_ ## level, testpmd_logtype, "testpmd: " fmt, ## args)

#endif /* _TESTPMD_H_ */
