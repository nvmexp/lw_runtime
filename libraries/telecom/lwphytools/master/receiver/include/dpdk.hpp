/*
 * Copyright 1993-2020 LWPU Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to LWPU intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to LWPU and is being provided under the terms and
 * conditions of a form of LWPU software license agreement by and
 * between LWPU and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of LWPU is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, LWPU MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * LWPU DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL LWPU BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */

#ifndef DPDK_HPP__
#define DPDK_HPP__

 /* ===== DPDK ===== */
#include <rte_ethdev.h>
#include <rte_mempool.h>
#include <rte_mbuf.h>
#include <rte_malloc.h>
#include <rte_ring.h>
#include <rte_ether.h>
#include <rte_ip.h>
#include <rte_tcp.h>
#include <atomic>
#include <fcntl.h>
//#include <rte_udp.h>

/* ===== DPDK-LW ===== */
#ifdef LWDA_ENABLED
    #include <rte_lw.h>
    #define DPDK_MAX_MEMPOOLS MAX_LW_MEMPOOLS
#else
    #define DPDK_MAX_MEMPOOLS 16
#endif

////////////////////////////////////////////////////////////
/// Defines
////////////////////////////////////////////////////////////
#define DPDK_RX_DESC_DEFAULT    1024
#define DPDK_TX_DESC_DEFAULT    2048
#define DPDK_MAX_MBUFS_BURST    1024
#define DPDK_MAX_QUEUES         16
#define DPDK_MAX_MBUFS_PAYLOAD  9216 //4096
#define DPDK_MAX_CORES          16
#define DPDK_DEF_MBUF_MP        65536
#define DPDK_CMSG_MBUF_MP       8192
#define DPDK_GET_MBUF_ADDR(x, o) rte_pktmbuf_mtod_offset(x, uint8_t *, o)

#define DPDK_HDR_ADDR_SIZE sizeof(struct rte_ether_hdr) // + sizeof(struct ipv4_hdr) // + sizeof(struct udp_hdr)

#define DPDK_UDP_SRC_PORT       1024
#define DPDK_UDP_DST_PORT       1024

#define DPDK_IP_SRC_ADDR_1 ((192U << 24) | (168 << 16) | (0 << 8) | 1)
#define DPDK_IP_DST_ADDR_1 ((192U << 24) | (168 << 16) | (0 << 8) | 2)

#define DPDK_IP_SRC_ADDR_2 ((192U << 24) | (168 << 16) | (0 << 8) | 250)
#define DPDK_IP_DST_ADDR_2 ((192U << 24) | (168 << 16) | (0 << 8) | 251)

#define DPDK_IP_DEFTTL          64   /* from RFC 1340. */
#define DPDK_IP_VERSION         0x40
#define DPDK_IP_HDRLEN          0x05 /* default IP header length == five 32-bits words. */
#define DPDK_IP_VHL_DEF         (DPDK_IP_VERSION | DPDK_IP_HDRLEN)

#define DPDK_MIN_PKTS_X_BURST   4

#define PTP_SRC "/dev/ptp5" // the sending NIC, i.e. b5:00.0
#define NUM_LEAP_SECONDS 0 //37
#define CLKQ_TICK_NS (1 * NS_X_US)
#define CLKQ_MBUF_CACHE_SIZE 250
#define CLKQ_BURST_SIZE 1024
#define MBUF_CLOCK_SIZE 60
////////////////////////////////////////////////////////////
/// Memory structures
////////////////////////////////////////////////////////////
struct dpdk_device_ctx {
    int                     enabled;
    //Port
    int                     port_id; //Multi-port not supported yet (RTE_MAX_ETHPORTS)
    struct rte_eth_conf     port_conf;
    struct rte_ether_addr   eth_addr;
    
    //Queue
    int tot_rxq;
    int tot_txq;
    uint16_t tot_rxd;
    uint16_t tot_txd;

    struct rte_eth_dev_info nic_info;
};

struct dpdk_pipeline_ctx {
    //Port
    int                     socket_id;
    int                     port_id; //Multi-port not supported yet (RTE_MAX_ETHPORTS)
    struct rte_ether_addr   peer_eth_addr;
    uint16_t                vlan;
    enum pt_flow_ident_method flow_ident_method;
    
    //Mempool
    int memp_num;
    int memp_cache;
    int memp_mbuf_num;
    lw_mempool_type memp_type;

    //Mbuf
    int mbuf_payload_size_rx;
    int mbuf_payload_size_tx;
    int mbuf_x_burst;
    int mbuf_x_slot;
    int max_burst_x_slot;
    
    //Queue
    int start_rxq;
    int rxq;
    int start_txq;
    int txq;
    int c_txq;
    int dl_txq;
    int ts_txq;
    uint16_t rxd;
    uint16_t txd;
    int hds;

    std::array<struct rte_mempool *, DPDK_MAX_MEMPOOLS> rx_dpdk_mempool;
    std::array<struct lw_mempool_info *, DPDK_MAX_MEMPOOLS> rx_lw_mempool;
    std::array<struct rte_mempool *, DPDK_MAX_MEMPOOLS> tx_dpdk_mempool;
    std::array<struct lw_mempool_info *, DPDK_MAX_MEMPOOLS> tx_lw_mempool;
    int rx_memp;

    struct rte_mempool * c_mempool;
};

typedef std::unique_ptr<struct rte_mbuf *, decltype(&rte_free)> rte_unique_mbufs;

////////////////////////////////////////////////////////////
/// DPDK functions
////////////////////////////////////////////////////////////
int dpdk_get_ports(void);
int dpdk_setup(struct phytools_ctx * ptctx);
struct rte_mempool * dpdk_create_mempool(const char * mp_name, int mbufs_num, int cache_size, int mbufs_payload, int socket_id);
struct lw_mempool_info * dpdk_create_lwmempool(const char * mp_name, 
                                            int mbufs_num, int cache_size, int mbufs_payload, 
                                            int port_id, int socket_id,
                                            lw_mempool_type memp_type);
struct rte_mempool * dpdk_get_mempool_from_lwmempool(struct lw_mempool_info * lw_mp);
struct rte_mbuf ** dpdk_allocate_mbufs(int mbufs_num);

int dpdk_mem_dma_map(uint16_t port_id, void *addr, size_t size);
int dpdk_tx_burst_pkts(rte_unique_mbufs &mbufs, int port_id, int txq, int tot_pkts, int flush_tx_write, std::atomic<bool> &force_quit);
int dpdk_rx_burst_pkts(rte_unique_mbufs &mbufs, int port_id, int rxq, int tot_pkts);

int dpdk_pull_pkts(rte_unique_mbufs &mbufs, struct rte_mempool *mp, int tot_pkts);
int dpdk_setup_rx_queues(struct dpdk_pipeline_ctx& dpdkctx);
int dpdk_uplink_network_setup(struct dpdk_pipeline_ctx& dpdkctx, int index_p, int memp_num);
int dpdk_setup_tx_queues(struct dpdk_pipeline_ctx& dpdkctx, int time_ss);
int dpdk_downlink_network_setup(struct phytools_ctx& ptctx, struct dpdk_pipeline_ctx& dpdkctx, int index_p, int memp_num, int war);
int dpdk_cplane_network_setup(struct dpdk_pipeline_ctx& dpdkctx, int index_p);

int dpdk_create_mempool_queues(struct phytools_ctx * ptctx);
int dpdk_start_nic(struct phytools_ctx * ptctx);
int dpdk_finalize(struct phytools_ctx * ptctx);
void dpdk_print_stats(struct phytools_ctx * ptctx);
int dpdk_print_ctx(struct dpdk_ctx * dctx);

void dpdk_copy_buf_to_pkt_segs(void* buf, unsigned len, struct rte_mbuf *pkt, unsigned offset);
void dpdk_copy_buf_to_pkt(void* buf, unsigned len, struct rte_mbuf *pkt, unsigned offset);
void dpdk_setup_pkt_udp_ip_headers(struct rte_ipv4_hdr *ip_hdr, struct udp_hdr *udp_hdr, uint16_t pkt_data_len);

void * dpdk_alloc_aligned_memory(size_t input_size, size_t *out_size, size_t page_size);
int dpdk_register_ext_mem(void * addr, size_t mem_size, size_t page_size, struct rte_eth_dev *dev);
int dpdk_unregister_ext_mem(void * addr, size_t mem_size, struct rte_eth_dev *dev);

int dpdk_setup_rules(struct phytools_ctx * ptctx);
int dpdk_flow_isolate(struct phytools_ctx * ptctx);

#endif
