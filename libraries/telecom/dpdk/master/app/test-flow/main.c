/* SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 2019 - 2020, LWPU CORPORATION.  All rights reserved.
 */

#include <rte_eal.h>
#include <rte_debug.h>
#include <rte_ethdev.h>
#include <rte_flow.h>
#include <rte_cycles.h>
#include <getopt.h>
#include <signal.h>
#include <lwca.h>
#include <lwda_runtime.h>
#include <lwda_runtime_api.h>

#define PORT_ID 0
#define MEMPOOL_SZ 8192
#define MEMPOOL_CACHE 256
#define GPU_ID 0
#define RXDTXD 1024
#define PKT_SZ 512
#define TX_MBUFS 512
#define RX_MBUFS (TX_MBUFS/2)
#define TX_MBUFS_ITER 8
#define PRINT_MBUFS 4
// how far the message is offset from the beginning of the packet has to be
// larger than sizeof(struct rte_ether_hdr) and smaller than PKT_SZ
#define MSG_OFF_0 128
#define MSG_OFF_1 256
#define MAX_PKT_LEN 1024
// RX Queue 0 is a regular, non-scattered queue
#define DROOM_SZ_SYS_Q0 (MAX_PKT_LEN + RTE_PKTMBUF_HEADROOM)
// RX Queue 1 is a scattered queue. The first mempool is a sysmem mempool with
// MSG_OFF_1+HEADROOM bytes of dataroom. The second mempool is a vidmem mempool
// with dataroom necessary to reach MAX_PKT_LEN. Note that only the first
// mempool carries extra dataroom for HEADROOM.
// This way, using RX scattering across multiple mempools, message 1 will be
// received in first mempool and second message will be received in second
// mempool (sysmem and vidmem, respectively).
#define DROOM_SZ_SYS_Q1 (MSG_OFF_1 + RTE_PKTMBUF_HEADROOM)
#define DROOM_SZ_VID (MAX_PKT_LEN - MSG_OFF_1)
#define NB_SEGS 2
#define DISP_INTERVAL 2000

#ifdef LW_TEGRA
    #define LW_GPU_PAGE_SHIFT 12
#else
    #define LW_GPU_PAGE_SHIFT 16
#endif
#define LW_GPU_PAGE_SIZE (1UL << LW_GPU_PAGE_SHIFT)

#define LWDA_CHECK(expr) \
    do { \
        lwdaError_t e = expr; \
        if (e != lwdaSuccess) \
            rte_panic("LWCA ERROR. %s returned %d:%s\n", #expr, e, lwdaGetErrorString(e)); \
    } while (0)

static int o_recv = 0;
static int o_send = 0;
static int o_forever = 0;

// we don't enable scatter on the entire port, just on queue 1
static struct rte_eth_conf port_conf = {
	.rxmode = {
		.split_hdr_size = 0,
		.max_rx_pkt_len = MAX_PKT_LEN,
	},
	.txmode = {
		.mq_mode = ETH_MQ_TX_NONE,
		.offloads = 0,
	},
};

static struct rte_ether_addr src_mac0 = {
	.addr_bytes = {0x00, 0x01, 0x02, 0x03, 0x04, 0x05}
};
static uint32_t src_ip0 = (192U << 24) | (11 << 16) | (2 << 8) | 0;
static uint16_t vlan_tci0 = 0x11;

static char payload0_0 = (char)0x11;
static char payload0_1 = (char)0x22;

static struct rte_ether_addr src_mac1 = {
	.addr_bytes = {0x00, 0x02, 0x02, 0x03, 0x04, 0x05}
};
static uint32_t src_ip1 = (192U << 24) | (11 << 16) | (2 << 8) | 1;
static uint16_t vlan_tci1 = 0x13;

static char payload1_0 = (char)0xaa;
static char payload1_1 = (char)0xbb;

static struct rte_ether_addr dst_mac = {
	.addr_bytes = {0xe0, 0xe1, 0xe2, 0xe3, 0xe4, 0xe5}
};
static struct rte_ipv4_hdr ip_template = {
	.version_ihl = 0x45,
	.type_of_service = 0,
	.total_length = PKT_SZ - sizeof(struct rte_ether_hdr),
	.packet_id = 0,
	.fragment_offset = 0,
	.time_to_live = 64,
	.next_proto_id = 17,
	.hdr_checksum = 0, /* need to recallwlate it every time */
	.dst_addr = (192U << 24) | (11 << 16) | (2 << 8) | 5
};

static struct rte_ether_addr eth_mask_full = {
	.addr_bytes = {0xff, 0xff, 0xff, 0xff, 0xff, 0xff}
};
static uint32_t ip_addr_mask_full = (uint32_t) -1;

static struct rte_mempool *mempool_q0_sys, *mempool_q1_sys, *mempool_q1_vid;
static uint64_t tot_nb_rx_pkts0 = 0, tot_nb_rx_pkts1 = 0;

void parse_opts(int argc, char **argv);
int receiver(void);
int sender(void);
void signal_handler(int s);
void setup_dpdk(void);
struct rte_flow *setup_rules(uint16_t vlan_tci, struct rte_ether_addr *src_eth, uint32_t src_ip, uint16_t qidx);
/* Return a pointer at a specific offset, taking care of chained mbufs.
 * If exceeds bounds, return NULL */
char *chained_pktmbuf_mtod_offset(struct rte_mbuf *mbuf, uint16_t off);
/* Print a character pointer to by the address followed by a (GPU) or (NON-GPU)
 * identifier to tell whether the address points GPU memory or not. Performs a
 * lwdaMemcpy as necessary.
 * Returns 0 if ptr pointed to CPU memory, 1 if pointed to GPU memory */
int print_char_and_mem(char *ptr);
void print_messages(struct rte_mbuf **mbufs, uint16_t nb_mbufs, const char *hdr,
                    const char *epi, int is_msg0_gpu_mem, int is_msg1_gpu_mem);
void recalc_ip_chksum(struct rte_ipv4_hdr *hdr);

int main(int argc, char **argv)
{
	int eal_args;
	if ((eal_args = rte_eal_init(argc, argv)) < 0) {
		rte_panic("Cannot init EAL\n");
	}
	argc -= eal_args;
	argv += eal_args;
	if (rte_eth_dev_count_avail() == 0) {
		rte_panic("No Ethernet ports - bye\n");
	}
	parse_opts(argc, argv);
	signal(SIGINT, signal_handler);
	signal(SIGTERM, signal_handler);
	signal(SIGUSR1, signal_handler);

	if (o_recv) {
		return receiver();
	}
	if (o_send) {
		return sender();
	}
	return EXIT_FAILURE;
}

void parse_opts(int argc, char **argv)
{
	int c;
	while (1) {
		static struct option long_options[] = {
			{"recv", no_argument, &o_recv, 1},
			{"send", no_argument, &o_send, 1},
			{"forever", no_argument, &o_forever, 1},
			{0, 0, 0, 0}
		};
		int option_index = 0;
		c = getopt_long(argc, argv, "", long_options, &option_index);

		if (c == -1) {
			break;
		}

		switch (c) {
		case 0:
			/* long option */
			break;
		case '?':
			break;
		default:
			rte_panic("Error in processing options.\n");
		}
	}

	if (o_send && o_recv) {
		rte_panic("Cannot set both --recv and --send modes\n");
	}
	if (!o_send && !o_recv) {
		rte_panic("One of --send or --recv modes has to be set\n");
	}

	if (o_forever) {
		printf("Will run forever; ^C to quit\n");
	} else if (o_send) {
		printf("Will transmit %d packets in total\n", TX_MBUFS);
	} else if (o_recv) {
		printf("Will receive %d packets in total, %d on each queue\n", TX_MBUFS, RX_MBUFS);
	}

}

int receiver(void)
{
	struct rte_flow_error flowerr;
	if (rte_flow_isolate(PORT_ID, 1, &flowerr)) {
		rte_panic("Flow isolation failed: %s\n", flowerr.message);
	}
	setup_dpdk();

	struct rte_flow *f0 = setup_rules(vlan_tci0, &src_mac0, src_ip0, 0);
	struct rte_flow *f1 = setup_rules(vlan_tci1, &src_mac1, src_ip1, 1);

	struct rte_mbuf *rx_pkts0[RX_MBUFS], *rx_pkts1[RX_MBUFS];
	uint16_t nb_rx_pkts0, nb_rx_pkts1;
	unsigned int iter = 0;
	unsigned int i = 0;
	unsigned int count_rx_pkts0=0, count_rx_pkts1=0;

	do {
		tot_nb_rx_pkts0 += nb_rx_pkts0 =
		                       rte_eth_rx_burst(PORT_ID, 0, &(rx_pkts0[count_rx_pkts0]),
		                                        (RX_MBUFS-count_rx_pkts0));
		count_rx_pkts0 += nb_rx_pkts0;
		tot_nb_rx_pkts1 += nb_rx_pkts1 =
		                       rte_eth_rx_burst(PORT_ID, 1, &(rx_pkts1[count_rx_pkts1]),
		                                        (RX_MBUFS-count_rx_pkts1));
		count_rx_pkts1 += nb_rx_pkts1;

		if (count_rx_pkts0 == RX_MBUFS) {
			print_messages(rx_pkts0, PRINT_MBUFS, "Queue 0 msgs:\n",
			               "All messages received in correct memory locations\n", 0, 0);
			for (i = 0; i < count_rx_pkts0; ++i) {
				rte_pktmbuf_free(rx_pkts0[i]);
			}
			count_rx_pkts0 = 0;
		}

		if (count_rx_pkts1 == RX_MBUFS) {
			print_messages(rx_pkts1, PRINT_MBUFS, "Queue 1 msgs:\n",
			               "All messages received in correct memory locations\n", 0, 1);
			for (i = 0; i < count_rx_pkts1; ++i) {
				rte_pktmbuf_free(rx_pkts1[i]);
			}
			count_rx_pkts1 = 0;
		}

		rte_delay_us_block(500);
		if (iter % DISP_INTERVAL == 0)
			printf("RX iteration %u complete. Total pkts RX'd: %"
			       PRIu64 " / %" PRIu64 "\n", iter, tot_nb_rx_pkts0,
			       tot_nb_rx_pkts1);
		++iter;
		/*
		   if (tot_nb_rx_pkts1 > 0)
		   break;
		 */
	} while (o_forever || tot_nb_rx_pkts0 + tot_nb_rx_pkts1 < TX_MBUFS);
	rte_flow_destroy(PORT_ID, f0, NULL);
	rte_flow_destroy(PORT_ID, f1, NULL);
	return EXIT_SUCCESS;
}

int sender(void)
{
	setup_dpdk();

	unsigned int pkts_txd;
	struct rte_mbuf *tx_pkts[TX_MBUFS_ITER];
	unsigned int i = 0;
	unsigned int iter = 0;
	uint64_t tot_nb_tx_pkts = 0;
	do {
		if (rte_pktmbuf_alloc_bulk(mempool_q0_sys, tx_pkts, TX_MBUFS_ITER))
			rte_panic
			("Could not get any more mbufs from the mempool\n");
		for (i = 0; i < TX_MBUFS_ITER; ++i) {
			struct rte_mbuf *m = tx_pkts[i];
			rte_pktmbuf_reset(m);
			m->pkt_len = PKT_SZ;
			m->data_len = PKT_SZ;
			m->nb_segs = 1;
			m->l2_len = sizeof(struct rte_ether_hdr);
			struct rte_ether_hdr *pkt_eth =
			    rte_pktmbuf_mtod(m, struct rte_ether_hdr *);

			struct rte_vlan_hdr *pkt_vlan =
			    rte_pktmbuf_mtod_offset(m, struct rte_vlan_hdr *,
			                            sizeof(struct rte_ether_hdr));
			struct rte_ipv4_hdr *pkt_ip =
			    rte_pktmbuf_mtod_offset(m, struct rte_ipv4_hdr *,
			                            sizeof(struct rte_ether_hdr) +
			                            sizeof(struct rte_vlan_hdr)
			                           );
			char * m0 = rte_pktmbuf_mtod_offset(m, char *, MSG_OFF_0);
			char * m1 = rte_pktmbuf_mtod_offset(m, char *, MSG_OFF_1);


			rte_memcpy(pkt_ip, &ip_template, sizeof(struct rte_ipv4_hdr));
			pkt_eth->ether_type = rte_cpu_to_be_16(RTE_ETHER_TYPE_VLAN);
			pkt_vlan->eth_proto = rte_cpu_to_be_16(RTE_ETHER_TYPE_IPV4);
			rte_ether_addr_copy(&dst_mac, &pkt_eth->d_addr);
			if (i % 2) {
				rte_ether_addr_copy(&src_mac0, &pkt_eth->s_addr);
				pkt_ip->src_addr = src_ip0;
				pkt_vlan->vlan_tci = rte_cpu_to_be_16(vlan_tci0);
				*m0 = payload0_0;
				*m1 = payload0_1;
			} else {
				rte_ether_addr_copy(&src_mac1, &pkt_eth->s_addr);
				pkt_ip->src_addr = src_ip1;
				pkt_vlan->vlan_tci = rte_cpu_to_be_16(vlan_tci1);
				*m0 = payload1_0;
				*m1 = payload1_1;
			}
			recalc_ip_chksum(pkt_ip);
		}
		pkts_txd = 0;
		while (rte_eth_tx_burst
		        (PORT_ID, 0, tx_pkts + pkts_txd,
		         TX_MBUFS_ITER - pkts_txd) != TX_MBUFS_ITER - pkts_txd) ;
		tot_nb_tx_pkts += TX_MBUFS_ITER;
		rte_delay_ms(1);
		if (iter % DISP_INTERVAL == 0)
			printf("TX iteration %u complete. "
			       "src_mac0 payload: (0x%x,0x%x). "
			       "src_mac1 payload: (0x%x,0x%x)\n",
			       iter,
			       payload0_0 & 0xff,
			       payload0_1 & 0xff,
			       payload1_0 & 0xff, payload1_1 & 0xff);
		++iter;
	} while (o_forever || tot_nb_tx_pkts < TX_MBUFS);

	return EXIT_SUCCESS;
}

void signal_handler(int signum)
{
	struct rte_eth_stats stats;
	int num_queues=2, index_queue=0;

	if (signum == SIGINT || signum == SIGTERM || signum == SIGUSR1) {
		printf("Signal %d received, preparing to exit...\n", signum);

		printf("\n============================\n");
		printf("Application packets stats\n");
		printf("============================\n");
		printf("Total packets Queue 0: %ld\nTotal packets Queue 1: %ld\n", tot_nb_rx_pkts0, tot_nb_rx_pkts1);
		rte_eth_stats_get(PORT_ID, &stats);

		printf("\n============================\n");
		printf("DPDK packets stats\n");
		printf("============================\n");
		printf("RX queues %d:\n", num_queues);
		for(index_queue=0; index_queue < num_queues; index_queue++) {
			printf("\tQueue %d -> packets = %ld bytes = %ld dropped = %ld\n", index_queue, stats.q_ipackets[index_queue], stats.q_ibytes[index_queue], stats.q_errors[index_queue]);
		}
		printf("\t-------------------------------------------\n");
		printf("\tTot received packets: %ld Tot received bytes: %ld\n", stats.ipackets, stats.ibytes);

		printf("TX queues %d:\n", num_queues);
		for(index_queue=0; index_queue < num_queues; index_queue++) {
			printf("\tQueue %d -> packets = %ld bytes = %ld\n", index_queue, stats.q_opackets[index_queue], stats.q_obytes[index_queue]);
		}
		printf("\t-------------------------------------------\n");
		printf("\tTot sent packets: %ld, Tot sent bytes: %ld\n", stats.opackets, stats.obytes);

		printf("ERRORS:\n");
		printf("\tRX packets dropped by the HW (RX queues are full) = %" PRIu64 "\n", stats.imissed);
		printf("\tTotal number of erroneous RX packets = %" PRIu64 "\n", stats.ierrors);
		printf("\tTotal number of RX mbuf allocation failures = %" PRIu64 "\n", stats.rx_nombuf);
		printf("\tTotal number of failed TX packets = %" PRIu64 "\n", stats.oerrors);
		printf("\n");

		rte_exit(EXIT_SUCCESS, "BYE\n");
	}
}

void setup_dpdk(void)
{
	if (rte_eth_dev_configure(PORT_ID, 2, 2, &port_conf)) {
		rte_panic("rte_eth_dev_configure failed\n");
	}

	struct rte_eth_dev_info dev_info;
	rte_eth_dev_info_get(PORT_ID, &dev_info);
	struct rte_eth_rxconf rxconf_q1;
	struct rte_eth_rxseg_split *rx_seg;
	memcpy(&rxconf_q1, &dev_info.default_rxconf, sizeof(rxconf_q1));
	
	rxconf_q1.offloads = DEV_RX_OFFLOAD_SCATTER | RTE_ETH_RX_OFFLOAD_BUFFER_SPLIT;
	rxconf_q1.rx_nseg = NB_SEGS;
	union rte_eth_rxseg rx_useg[NB_SEGS] = {};
	rxconf_q1.rx_seg = rx_useg;

	LWDA_CHECK(lwdaSetDevice(GPU_ID));
	struct rte_pktmbuf_extmem gpu_mem;
	gpu_mem.buf_iova = RTE_BAD_IOVA;
	size_t gpu_buf_len = MEMPOOL_SZ * DROOM_SZ_VID;
	gpu_mem.buf_len = RTE_ALIGN(gpu_buf_len, LW_GPU_PAGE_SIZE);
	gpu_mem.elt_size = DROOM_SZ_VID;

	LWDA_CHECK(lwdaMalloc(&gpu_mem.buf_ptr, gpu_mem.buf_len));
	if (!gpu_mem.buf_ptr) {
		rte_panic("Could not allocate GPU device memory\n");
	}

	int ret = rte_extmem_register(gpu_mem.buf_ptr, gpu_mem.buf_len, NULL, gpu_mem.buf_iova, LW_GPU_PAGE_SIZE);
    if (ret) {
        rte_panic("Unable to register addr 0x%p, ret %d\n", gpu_mem.buf_ptr, ret);
    }

	if (rte_dev_dma_map(rte_eth_devices[PORT_ID].device, gpu_mem.buf_ptr, gpu_mem.buf_iova, gpu_mem.buf_len)) {
        rte_panic("Unable to DMA map addr 0x%p for device %s\n", gpu_mem.buf_ptr, rte_eth_devices[PORT_ID].data->name);
    }

	mempool_q1_vid = rte_pktmbuf_pool_create_extbuf("vidmem-mempool", MEMPOOL_SZ,
					 MEMPOOL_CACHE, 0, DROOM_SZ_VID, rte_socket_id(), &gpu_mem, 1);
	if (!mempool_q1_vid) {
		rte_panic("Could not create vidmem mempool\n");
	}

	rx_seg = &rx_useg[1].split;
	rx_seg->mp = mempool_q1_vid;
	rx_seg->length = 0;
	rx_seg->offset = 0;

	mempool_q0_sys = rte_pktmbuf_pool_create("sysmem-mempool_0", MEMPOOL_SZ,
	                 MEMPOOL_CACHE, 0,
	                 DROOM_SZ_SYS_Q0,
	                 rte_socket_id());
	if (!mempool_q0_sys) {
		rte_panic("Could not create sysmem mempool for queue 0\n");
	}

	mempool_q1_sys = rte_pktmbuf_pool_create("sysmem-mempool_1", MEMPOOL_SZ,
	                 MEMPOOL_CACHE, 0,
	                 DROOM_SZ_SYS_Q1,
	                 rte_socket_id());
	if (!mempool_q1_sys) {
		rte_panic("Could not create sysmem mempool for queue 1\n");
	}

	rx_seg = &rx_useg[0].split;
	rx_seg->mp = mempool_q1_sys;
	rx_seg->length = 0;
	rx_seg->offset = 0;

	if (rte_eth_rx_queue_setup
	        (PORT_ID, 0, RXDTXD, rte_socket_id(), NULL, mempool_q0_sys)) {
		rte_panic("Could not setup RX queue 0\n");
	}

	if (rte_eth_rx_queue_setup
			(PORT_ID, 1, RXDTXD, rte_socket_id(), &rxconf_q1, NULL)) {
		rte_panic("Could not setup RX queue 1\n");
	}

	if (rte_eth_tx_queue_setup(PORT_ID, 0, RXDTXD, rte_socket_id(), NULL)) {
		rte_panic("Could not setup TX queue 0\n");
	}

	if (rte_eth_tx_queue_setup(PORT_ID, 1, RXDTXD, rte_socket_id(), NULL)) {
		rte_panic("Could not setup TX queue 1\n");
	}

	if (rte_eth_dev_start(PORT_ID)) {
		rte_panic("Could not start the device");
	}
}

struct rte_flow *setup_rules(uint16_t vlan_tci, struct rte_ether_addr *src_eth, uint32_t src_ip, uint16_t qidx)
{
	struct rte_flow_attr attr;
	struct rte_flow_item patterns[4];
	struct rte_flow_action actions[2];
	struct rte_flow_error err;
	struct rte_flow_action_queue queue = {.index = qidx };
	struct rte_flow_item_eth eth_spec, eth_mask;
	struct rte_flow_item_vlan vlan_spec, vlan_mask;
	struct rte_flow_item_ipv4 ip_spec, ip_mask;

	memset(&attr, 0, sizeof(attr));
	memset(patterns, 0, sizeof(patterns));
	memset(actions, 0, sizeof(actions));
	memset(&eth_spec, 0, sizeof(eth_spec));
	memset(&eth_mask, 0, sizeof(eth_mask));
	memset(&vlan_spec, 0, sizeof(vlan_spec));
	memset(&vlan_mask, 0, sizeof(vlan_mask));
	memset(&ip_spec, 0, sizeof(ip_spec));
	memset(&ip_mask, 0, sizeof(ip_mask));

	attr.ingress = 1;

	rte_ether_addr_copy(&eth_mask_full, &eth_mask.src);
	rte_ether_addr_copy(src_eth, &eth_spec.src);
	vlan_spec.tci = rte_cpu_to_be_16(vlan_tci);
	vlan_mask.tci = rte_cpu_to_be_16(0x0fff); /* lower 12 bits only */
	ip_mask.hdr.src_addr = ip_addr_mask_full;
	ip_spec.hdr.src_addr = src_ip;

	actions[0].type = RTE_FLOW_ACTION_TYPE_QUEUE;
	actions[0].conf = &queue;
	actions[1].type = RTE_FLOW_ACTION_TYPE_END;

	patterns[0].type = RTE_FLOW_ITEM_TYPE_ETH;
	patterns[0].spec = &eth_spec;
	patterns[0].mask = &eth_mask;
	patterns[1].type = RTE_FLOW_ITEM_TYPE_VLAN;
	patterns[1].spec = &vlan_spec;
	patterns[1].mask = &vlan_mask;
	patterns[2].type = RTE_FLOW_ITEM_TYPE_IPV4;
	patterns[2].spec = &ip_spec;
	patterns[2].mask = &ip_mask;
	patterns[3].type = RTE_FLOW_ITEM_TYPE_END;

	if (rte_flow_validate(PORT_ID, &attr, patterns, actions, &err)) {
		rte_panic("Invalid flow rule: %s\n", err.message);
	}
	return rte_flow_create(PORT_ID, &attr, patterns, actions, &err);
}

char *chained_pktmbuf_mtod_offset(struct rte_mbuf *mbuf, uint16_t off)
{
	uint16_t dlen;
	while (mbuf != NULL) {
		dlen = rte_pktmbuf_data_len(mbuf);
		if (dlen > off) {
			return rte_pktmbuf_mtod_offset(mbuf, char *, off);
		}
		mbuf = mbuf->next;
		off -= dlen;
	}
	return NULL;
}

int print_char_and_mem(char *ptr)
{
	if (ptr == NULL) {
		printf("%s", "[NULL]");
		return -1;
	}

	char c;
	int is_gpu_mem = 0;
	struct lwdaPointerAttributes attr;
	lwdaError_t err;
	err = lwdaPointerGetAttributes(&attr, ptr);
	if (err == lwdaSuccess && attr.type == lwdaMemoryTypeDevice) {
		is_gpu_mem = 1;
	}
	if (is_gpu_mem) {
		LWDA_CHECK(lwdaMemcpy(&c, ptr, sizeof(c), lwdaMemcpyDefault));
	} else {
		c = *ptr;
	}
	printf(" 0x%x (%s)", c & 0xff, (is_gpu_mem) ? "GPU" : "NON-GPU");
	return is_gpu_mem;
}

void print_messages(struct rte_mbuf **mbufs, uint16_t nb_mbufs, const char *hdr,
                    const char *epi, int is_msg0_gpu_mem, int is_msg1_gpu_mem)
{
	uint16_t i;
	int ret;
	if (hdr) {
		printf("%s", hdr);
	}
	for (i = 0; i < nb_mbufs; ++i) {
		printf("%s", "    ==> ");
		ret = print_char_and_mem(chained_pktmbuf_mtod_offset
		                         (mbufs[i], MSG_OFF_0));
		if (ret != is_msg0_gpu_mem) {
			rte_panic("Wrong memory type for message 0: %d (expected %d)\n", ret, is_msg0_gpu_mem);
		}
		printf(" | ");
		ret = print_char_and_mem(chained_pktmbuf_mtod_offset
		                         (mbufs[i], MSG_OFF_1));
		if (ret != is_msg1_gpu_mem) {
			rte_panic("Wrong memory type for message 1: %d (expected %d)\n", ret, is_msg1_gpu_mem);
		}
		printf("%s", "\n");
	}
	if (epi) {
		printf("%s", epi);
	}
}

/* code lifted from test-pmd/txonly.c */
void recalc_ip_chksum(struct rte_ipv4_hdr *ip_hdr)
{
	uint32_t ip_cksum;
	/*
	 * Compute IP header checksum.
	 */
	uint16_t *ptr16 = (unaligned_uint16_t*) ip_hdr;
	ip_cksum = 0;
	ip_cksum += ptr16[0];
	ip_cksum += ptr16[1];
	ip_cksum += ptr16[2];
	ip_cksum += ptr16[3];
	ip_cksum += ptr16[4];
	ip_cksum += ptr16[6];
	ip_cksum += ptr16[7];
	ip_cksum += ptr16[8];
	ip_cksum += ptr16[9];

	/*
	 * Reduce 32 bit checksum to 16 bits and complement it.
	 */
	ip_cksum = ((ip_cksum & 0xFFFF0000) >> 16) +
	           (ip_cksum & 0x0000FFFF);
	if (ip_cksum > 65535) {
		ip_cksum -= 65535;
	}
	ip_cksum = (~ip_cksum) & 0x0000FFFF;
	if (ip_cksum == 0) {
		ip_cksum = 0xFFFF;
	}
	ip_hdr->hdr_checksum = (uint16_t) ip_cksum;
}
