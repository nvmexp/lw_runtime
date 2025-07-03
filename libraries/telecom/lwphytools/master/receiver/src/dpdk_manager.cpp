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

#include "lwphytools.hpp"
#include <rte_eal.h>
#include <rte_common.h>
#include <rte_launch.h>
#include <rte_lcore.h>
#include <rte_per_lcore.h>
#include <rte_mbuf.h>
#include <rte_ether.h>
#include <rte_ethdev.h>

////////////////////////////////////////////////
//// Utils
////////////////////////////////////////////////

int dpdk_get_ports(void) {
    return rte_eth_dev_count_avail();
}

int dpdk_setup(struct phytools_ctx * ptctx)
{
    int ret = PT_OK;
    struct rte_eth_rss_conf rss_conf;
    struct rte_eth_dev_info dev_info;
    
    //rss_conf.rss_key_len=0;
    //rss_conf.rss_hf = dev_info.flow_type_rss_offloads;
    //dev_info.flow_type_rss_offloads=0;

    if(!ptctx)
        return PT_EILWAL;

    for(int i=0; i < RTE_MAX_ETHPORTS; i++)
    {
        if(ptctx->dpdk_dev[i].enabled)
        {
            rte_eth_dev_info_get(ptctx->dpdk_dev[i].port_id, &(ptctx->dpdk_dev[i].nic_info));

            //pt_info("Device driver name in use: %s\n", ptctx->dpdk_dev[i].nic_info.driver_name);

            if(
                strstr(ptctx->dpdk_dev[i].nic_info.driver_name, "i40e") != NULL || 
                strstr(ptctx->dpdk_dev[i].nic_info.driver_name, "ixgbe") != NULL
            )
            {
                ptctx->dpdk_dev[i].port_conf.rx_adv_conf.rss_conf.rss_hf = ETH_RSS_PROTO_MASK;
            }
            ptctx->dpdk_dev[i].port_conf.txmode.offloads = DEV_TX_OFFLOAD_MULTI_SEGS;
            if (ptctx->dpdk_dev[i].port_conf.rxmode.max_rx_pkt_len > 1514);
                ptctx->dpdk_dev[i].port_conf.rxmode.offloads |= DEV_RX_OFFLOAD_JUMBO_FRAME;

            if (ptctx->plctx[0].dpdkctx.hds > 0)
                ptctx->dpdk_dev[i].port_conf.rxmode.offloads |= DEV_RX_OFFLOAD_SCATTER;

            if (ptctx->plctx[0].dpdkctx.flow_ident_method == PT_FLOW_IDENT_METHOD_eCPRI)
                ptctx->dpdk_dev[i].port_conf.rxmode.offloads |= ~DEV_RX_OFFLOAD_VLAN_STRIP & DEV_RX_OFFLOAD_VLAN_FILTER;

            pt_info("Initializing device %s port %u with %d RX queues (offloads=%lx) and %d TX queues (offloads=%lx)\n",
                        ptctx->dpdk_dev[i].nic_info.device->name, ptctx->dpdk_dev[i].port_id,
                        ptctx->dpdk_dev[i].tot_rxq, ptctx->dpdk_dev[i].port_conf.rxmode.offloads,
                        ptctx->dpdk_dev[i].tot_txq, ptctx->dpdk_dev[i].port_conf.txmode.offloads);
            
            ret = rte_eth_dev_configure(ptctx->dpdk_dev[i].port_id, 
                                        ptctx->dpdk_dev[i].tot_rxq,
                                        ptctx->dpdk_dev[i].tot_txq, 
                                        &(ptctx->dpdk_dev[i].port_conf)
                                    );
            if (ret < 0)
            {
                pt_err("Cannot configure device: err=%d, port=%u\n", ret, ptctx->dpdk_dev[i].port_id);
                goto out_err;
            }

            //Backup
            #if 0
                int vlan_offload;
                int diag;

                vlan_offload = rte_eth_dev_get_vlan_offload(ptctx->dpdk_dev[i].port_id);
                vlan_offload |= ETH_VLAN_FILTER_OFFLOAD;

                pt_info("filter vlan_offload=%x\n", vlan_offload);
                diag = rte_eth_dev_set_vlan_offload(ptctx->dpdk_dev[i].port_id, vlan_offload);
                if (diag < 0) printf("rx_vlan_strip_set(port_pi=%d) failed diag=%d\n", ptctx->dpdk_dev[0].port_id, diag);

                pt_info("rte_eth_dev_vlan_filter(%d, 2, 1)\n", ptctx->dpdk_dev[i].port_id);
                diag = rte_eth_dev_vlan_filter(ptctx->dpdk_dev[i].port_id, 2, 1);
                if (diag < 0)
                {
                        printf("rte_eth_dev_vlan_filter(port_pi=%d, vlan_id=2) failed diag=%d\n", ptctx->dpdk_dev[0].port_id, diag);
                        return -1;
                }

                vlan_offload = rte_eth_dev_get_vlan_offload(ptctx->dpdk_dev[i].port_id);
                vlan_offload &= ~ETH_VLAN_STRIP_OFFLOAD;

                pt_info("strip vlan_offload=%x\n", vlan_offload);
                diag = rte_eth_dev_set_vlan_offload(ptctx->dpdk_dev[i].port_id, vlan_offload);
                if (diag < 0) printf("rx_vlan_strip_set(port_pi=%d) failed diag=%d\n", ptctx->dpdk_dev[0].port_id, diag);
            #endif

            ret = rte_eth_dev_adjust_nb_rx_tx_desc(ptctx->dpdk_dev[i].port_id, &(ptctx->dpdk_dev[i].tot_rxd), &(ptctx->dpdk_dev[i].tot_txd));
            if (ret < 0)
            {
                pt_err("Cannot adjust number of descriptors: err=%d, port=%u\n", ret, ptctx->dpdk_dev[i].port_id);
                goto out_err;
            }

            rte_eth_dev_info_get(ptctx->dpdk_dev[i].port_id, &(ptctx->dpdk_dev[i].nic_info));
            rte_eth_macaddr_get(ptctx->dpdk_dev[i].port_id, &(ptctx->dpdk_dev[i].eth_addr));
            
            pt_info("Port %u has %d RX descriptors, %d TX descriptors, %d RX queues, %d TX queues, MAC address: %02X:%02X:%02X:%02X:%02X:%02X\n",
                    ptctx->dpdk_dev[i].port_id,
                    ptctx->dpdk_dev[i].tot_rxd, ptctx->dpdk_dev[i].tot_txd,
                    ptctx->dpdk_dev[i].tot_rxq, ptctx->dpdk_dev[i].tot_txq,
                    ptctx->dpdk_dev[i].eth_addr.addr_bytes[0], ptctx->dpdk_dev[i].eth_addr.addr_bytes[1],
                    ptctx->dpdk_dev[i].eth_addr.addr_bytes[2], ptctx->dpdk_dev[i].eth_addr.addr_bytes[3],
                    ptctx->dpdk_dev[i].eth_addr.addr_bytes[4], ptctx->dpdk_dev[i].eth_addr.addr_bytes[5]
            );
        }
    }

    return PT_OK;

    out_err:
        dpdk_finalize(ptctx);
        return PT_ERR;
}

struct rte_mempool * dpdk_create_mempool(const char * mp_name, int mbufs_num, int cache_size, int mbufs_payload, int socket_id)
{
    if(
        !mp_name || mbufs_num <=0 || cache_size <=0 ||
        mbufs_payload <= RTE_PKTMBUF_HEADROOM || mbufs_payload >= DPDK_MAX_MBUFS_PAYLOAD ||
        socket_id < 0
    )
    {
        pt_err("Invalid input parameters");
        return NULL;
    }

    return rte_pktmbuf_pool_create(mp_name, mbufs_num, cache_size, 0, mbufs_payload, socket_id);
}

int dpdk_mem_dma_map(uint16_t port_id, void *addr, size_t size)
{
    struct rte_device *dev = rte_eth_devices[port_id].device;
    int ret;

    // if(!rte_is_power_of_2(size))
    // {
    //     pt_err("Size %zd is not power of 2\n", size);
    //     // goto out_err;
    // }

    // int ret = rte_extmem_register(addr, size, NULL, 0, sysconf(_SC_PAGESIZE));
    // if (ret)
    // {
    //     pt_err("unable to register addr 0x%p, ret %d\n", addr, ret);
    //     goto out_err;
    // }

    // pt_info("Registering 0x%p : %zu with device %s for DMA\n", addr, size, dev->name);
    ret = rte_dev_dma_map(dev, addr, RTE_BAD_IOVA, size);
    if (ret != 0)
    {
        pt_err("rte_dev_dma_map failed. %d: %s", ret, rte_strerror(rte_errno));
        goto out_err;
    }

    return PT_OK;

out_err:
    return PT_ERR;

}

static void dpdk_txonly_mp_dma_map(uint16_t port_id, struct rte_mempool *mp)
{
    // traverse memsegs
    auto cb = [](struct rte_mempool *mp, void *opaque,
              struct rte_mempool_memhdr *memhdr, unsigned mem_idx) -> void {
        uint16_t port_id = *((uint16_t *)opaque);
        // pt_info("Registering segment %u of mempool %s\n", mem_idx, mp->name);
        dpdk_mem_dma_map(port_id, memhdr->addr, memhdr->len);
    };
    rte_mempool_mem_iter(mp, cb, (void *)&port_id);
}

struct rte_mbuf ** dpdk_allocate_mbufs(int mbufs_num)
{
    struct rte_mbuf ** mbufs = NULL;

    if(mbufs_num <= 0)
        return NULL;

    mbufs = (struct rte_mbuf **) rte_zmalloc(NULL, sizeof(struct rte_mbuf*) * mbufs_num, sysconf(_SC_PAGESIZE));
    if(mbufs == NULL)
        pt_err("can't allocate %d mbufs\n", mbufs_num);

    return mbufs;
}

int dpdk_tx_burst_pkts(rte_unique_mbufs &mbufs, int port_id, int txq, int tot_pkts, int flush_tx_write, std::atomic<bool> &force_quit)
{
    int nb_tx = 0;
    struct rte_mbuf ** mbufs_local = NULL;

    if(tot_pkts <= 0)
        return PT_EILWAL;

    mbufs_local = mbufs.get();

    while (nb_tx < tot_pkts && check_force_quit(force_quit) == false) {
        nb_tx += rte_eth_tx_burst(port_id, txq, mbufs_local + nb_tx, tot_pkts-nb_tx);
    }
 
    if(flush_tx_write)
        rte_wmb();

    return nb_tx;
}

int dpdk_rx_burst_pkts(rte_unique_mbufs &mbufs, int port_id, int rxq, int tot_pkts)
{
    int nb_rx = 0;

    if(tot_pkts <= 0)
        return PT_EILWAL;

    nb_rx = rte_eth_rx_burst(port_id, rxq, (mbufs.get())+nb_rx, tot_pkts - nb_rx);

    return nb_rx;
}

int dpdk_pull_pkts(rte_unique_mbufs &mbufs, struct rte_mempool *mp, int tot_pkts)
{
    if (rte_pktmbuf_alloc_bulk(mp, mbufs.get(), tot_pkts) != 0)
    {
        pt_err("Pulling %d mbufs from the pool 0x%lx failed.\n", tot_pkts, (uintptr_t)mp);
        return PT_ERR;
    }

    return PT_OK;
}
////////////////////////////////////////////////
//// U-plane RX
////////////////////////////////////////////////

int dpdk_setup_rx_queues(struct dpdk_pipeline_ctx& dpdkctx)
{
    int ret=PT_OK, index_q=0, num_q=0, index_m=0, lcore_socket_id=0, q_x_m=0;
    struct rte_eth_rxconf rxconf_queue;
    memset(&rxconf_queue, 0, sizeof(rxconf_queue));
    rxconf_queue.offloads = DEV_RX_OFFLOAD_JUMBO_FRAME;
    if (dpdkctx.flow_ident_method == PT_FLOW_IDENT_METHOD_eCPRI)
        rxconf_queue.offloads = ~DEV_RX_OFFLOAD_VLAN_STRIP & DEV_RX_OFFLOAD_VLAN_FILTER;

    //How many queues x mempool
    q_x_m = (dpdkctx.rx_memp/dpdkctx.rxq)+1;

    for(index_q=dpdkctx.start_rxq; index_q < (dpdkctx.start_rxq+dpdkctx.rxq); index_q++)
    {
        lcore_socket_id = (uint8_t)rte_lcore_to_socket_id(index_q);
        if(dpdkctx.socket_id != lcore_socket_id)
        {
            pt_warn("Config socket id is %d while dpdk core %d has socket id %d\n", 
                        dpdkctx.socket_id, index_q, lcore_socket_id);
        }

        pt_info("RX queue %d dpdkctx.port_id=%d, index_q=%d, index_m=%d, dpdkctx.rxd=%d, lcore_socket_id=%d, mempool=%p\n",
                index_q, dpdkctx.port_id, index_q, index_m, dpdkctx.rxd, lcore_socket_id, dpdkctx.rx_dpdk_mempool[index_m]);
        
        ret = rte_eth_rx_queue_setup(dpdkctx.port_id, index_q, dpdkctx.rxd, lcore_socket_id, &rxconf_queue, dpdkctx.rx_dpdk_mempool[index_m]);
        if (ret < 0)
        {
            pt_err("rte_eth_rx_queue_setup (%d): err=%d, port=%u\n", index_q, ret, dpdkctx.port_id);
            goto out_err;
        }

        if((num_q > 0) && (num_q % q_x_m) == 0 && index_m < (dpdkctx.rx_memp-1))
            index_m++;
    }
    
    return PT_OK;

out_err:
    return PT_ERR;
}


int dpdk_setup_rx_queues_hds(struct dpdk_pipeline_ctx& dpdkctx)
{
    int ret=PT_OK, index_q=0, num_q=0, index_m=0, lcore_socket_id=0, q_x_m=0;
    struct rte_eth_rxconf rxconf_queue;
    memset(&rxconf_queue, 0, sizeof(rxconf_queue));
    rxconf_queue.offloads = DEV_RX_OFFLOAD_SCATTER | DEV_RX_OFFLOAD_JUMBO_FRAME;
    if (dpdkctx.flow_ident_method == PT_FLOW_IDENT_METHOD_eCPRI)
        rxconf_queue.offloads |= ~DEV_RX_OFFLOAD_VLAN_STRIP & DEV_RX_OFFLOAD_VLAN_FILTER;

    //How many queues x mempool
    q_x_m = (dpdkctx.rx_memp/dpdkctx.rxq)+1;

    for(index_q=dpdkctx.start_rxq; index_q < (dpdkctx.start_rxq+dpdkctx.rxq); index_q++)
    {
        lcore_socket_id = (uint8_t)rte_lcore_to_socket_id(index_q);
        if(dpdkctx.socket_id != lcore_socket_id)
            pt_warn("Config socket id is %d while dpdk core %d has socket id %d\n", dpdkctx.socket_id, index_q, lcore_socket_id);

        pt_info("RX queue %d dpdkctx.port_id=%d, index_q=%d, index_m=%d, dpdkctx.rxd=%d, lcore_socket_id=%d, mempool=%p\n",
                    index_q, dpdkctx.port_id, index_q, index_m, dpdkctx.rxd, lcore_socket_id, dpdkctx.rx_dpdk_mempool[index_m]);
        
        struct rte_mempool *mps[] = { dpdkctx.rx_dpdk_mempool[index_m], dpdk_get_mempool_from_lwmempool(dpdkctx.rx_lw_mempool[index_m])};
        ret = rte_eth_rx_queue_setup_ex(dpdkctx.port_id, index_q, dpdkctx.rxd, lcore_socket_id, &rxconf_queue, mps, sizeof(mps) / sizeof(mps[0]));
        if (ret < 0)
        {
            pt_err("rte_eth_rx_queue_setup (%d): err=%d, port=%u\n", index_q, ret, dpdkctx.port_id);
            goto out_err;
        }

        if((num_q > 0) && (num_q % q_x_m) == 0 && index_m < (dpdkctx.rx_memp-1))
            index_m++;
    }
    
    return PT_OK;

out_err:
    return PT_ERR;
}

int dpdk_uplink_network_setup(struct dpdk_pipeline_ctx& dpdkctx, int index_p, int memp_num) {
    int index = 0;
    char mp_name[1024];

    dpdkctx.rx_memp = memp_num;
    for(index = 0; index < dpdkctx.rx_memp; index++)
    {
        snprintf(mp_name, sizeof(mp_name), "lwpool-rx-p%d", index);
        dpdkctx.rx_lw_mempool[index] = dpdk_create_lwmempool(mp_name, 
                                            dpdkctx.memp_mbuf_num,
                                            dpdkctx.memp_cache,
                                            (dpdkctx.mbuf_payload_size_rx + RTE_PKTMBUF_HEADROOM),
                                            dpdkctx.port_id,
                                            dpdkctx.socket_id,
                                            dpdkctx.memp_type);
        if(dpdkctx.rx_lw_mempool[index] == NULL)
        {
            pt_warn("rx_lw_mempool[%d] is NULL\n", index);
            goto out_err;
        }

        //Header/Data split
        if(dpdkctx.hds > 0)
        {
            snprintf(mp_name, sizeof(mp_name), "pool-rx-p%d", index);
            dpdkctx.rx_dpdk_mempool[index] = dpdk_create_mempool(mp_name,
                                dpdkctx.memp_mbuf_num,
                                dpdkctx.memp_cache,
                                ORAN_IQ_HDR_SZ + RTE_PKTMBUF_HEADROOM,
                                dpdkctx.socket_id);
            if(dpdkctx.rx_dpdk_mempool[index] == NULL)
            {
                pt_warn("rx_dpdk_mempool[%d] is NULL\n", index);
                goto out_err;
            }

            dpdk_txonly_mp_dma_map(dpdkctx.port_id, dpdkctx.rx_dpdk_mempool[index]);
        }
        else
        {
            dpdkctx.rx_dpdk_mempool[index] = dpdk_get_mempool_from_lwmempool(dpdkctx.rx_lw_mempool[index]);
            if(dpdkctx.rx_dpdk_mempool[index] == NULL)
            {
                pt_warn("rx_dpdk_mempool[%d] is NULL\n", index);
                goto out_err;
            }
        }
    }

    //Header/Data split
    if(dpdkctx.hds > 0)
    {
        if(dpdk_setup_rx_queues_hds(dpdkctx) != PT_OK) {
            pt_warn("dpdk_setup_rx_queues error\n");
            goto out_err;
        }
    }
    else
    {
        if(dpdk_setup_rx_queues(dpdkctx) != PT_OK) {
            pt_warn("dpdk_setup_rx_queues error\n");
            goto out_err;
        }
    }

    return PT_OK;
out_err:
    return PT_ERR;
}

int dpdk_downlink_network_setup(struct phytools_ctx& ptctx, struct dpdk_pipeline_ctx& dpdkctx, int index_p, int memp_num, int war) {
    int index = 0, ret = 0;
    char mp_name[1024];
    struct rte_eth_txconf txconf;
    struct rte_eth_dev_info dev_info;

    //WAR DPDK 19.11, downlink only
    if(war)
    {
        dpdkctx.rx_memp = 1;
        snprintf(mp_name, sizeof(mp_name), "pool-rx-p0");
        dpdkctx.rx_dpdk_mempool[0] = dpdk_create_mempool(mp_name,
                            1024,
                            dpdkctx.memp_cache,
                            512 + RTE_PKTMBUF_HEADROOM,
                            dpdkctx.socket_id);
        if(dpdkctx.rx_dpdk_mempool[0] == NULL)
        {
            pt_warn("rx_dpdk_mempool is NULL, rte_errno: %d\n", rte_errno);
            goto out_err;
        }

        if(dpdk_setup_rx_queues(dpdkctx) != PT_OK) {
            pt_warn("dpdk_setup_rx_queues error\n");
            goto out_err;
        }
    }

    snprintf(mp_name, sizeof(mp_name), "hdrpool-tx-p%d", index_p);
    dpdkctx.tx_dpdk_mempool[0] = dpdk_create_mempool(mp_name,
                                dpdkctx.memp_mbuf_num,
                                dpdkctx.memp_cache,
                                RTE_PKTMBUF_HEADROOM + ORAN_IQ_HDR_SZ,
                                dpdkctx.socket_id);
    if(dpdkctx.tx_dpdk_mempool[0] == NULL)
    {
        pt_warn("tx_dpdk_mempool 0 is NULL\n");
        goto out_err;
    }
    
    dpdk_txonly_mp_dma_map(dpdkctx.port_id, dpdkctx.tx_dpdk_mempool[0]);

    snprintf(mp_name, sizeof(mp_name), "datapool-tx-p%d", index_p);
    dpdkctx.tx_dpdk_mempool[1] = dpdk_create_mempool(mp_name,
                                dpdkctx.memp_mbuf_num,
                                dpdkctx.memp_cache,
                                RTE_PKTMBUF_HEADROOM + dpdkctx.mbuf_payload_size_tx,
                                dpdkctx.socket_id);
    if(dpdkctx.tx_dpdk_mempool[1] == NULL)
    {
        pt_warn("tx_dpdk_mempool 1 is NULL\n");
        goto out_err;
    }
    // dpdk_txonly_mp_dma_map(dpdkctx.port_id, dpdkctx.tx_dpdk_mempool[1]);

    rte_eth_dev_info_get(dpdkctx.port_id, &dev_info);
    memset(&txconf, 0, sizeof(txconf));
    if (dev_info.tx_offload_capa & DEV_TX_OFFLOAD_MBUF_FAST_FREE)
		ptctx.dpdk_dev[dpdkctx.port_id].port_conf.txmode.offloads |= DEV_TX_OFFLOAD_MBUF_FAST_FREE;

    txconf = dev_info.default_txconf;
    txconf.offloads = ptctx.dpdk_dev[dpdkctx.port_id].port_conf.txmode.offloads;

    ret = rte_eth_tx_queue_setup(dpdkctx.port_id, dpdkctx.dl_txq, dpdkctx.txd, rte_socket_id(), &txconf);
    if (ret < 0)
    {
        pt_err("rte_eth_tx_queue_setup (%d): err=%d, port=%u\n", dpdkctx.dl_txq, ret, dpdkctx.port_id);
        goto out_err;
    }

    return PT_OK;
out_err:
    return PT_ERR;
}

////////////////////////////////////////////////
//// C-plane TX
////////////////////////////////////////////////
int dpdk_cplane_network_setup(struct dpdk_pipeline_ctx& dpdkctx, int index_p) {
    int ret=0;
    char mp_name[1024];

    snprintf(mp_name, sizeof(mp_name), "cpool-tx-p%d", index_p);
    dpdkctx.c_mempool = dpdk_create_mempool(mp_name,
                                DPDK_CMSG_MBUF_MP,
                                dpdkctx.memp_cache,
                                (512 + RTE_PKTMBUF_HEADROOM), //ORAN_CMSG_ULDL_UNCOMPRESSED_SECTION_OVERHEAD
                                dpdkctx.socket_id);
    if(dpdkctx.c_mempool == NULL)
    {
        pt_warn("c_mempool is NULL\n");
        goto out_err;
    }

    dpdk_txonly_mp_dma_map(dpdkctx.port_id, dpdkctx.c_mempool);

    //RX queue for c-plane is useless now
    pt_info("C-plane TX queue %d\n", dpdkctx.c_txq);
    ret = rte_eth_tx_queue_setup(dpdkctx.port_id, dpdkctx.c_txq, dpdkctx.txd, dpdkctx.socket_id, NULL);
    if (ret < 0)
    {
        pt_err("rte_eth_tx_queue_setup (%d): err=%d, port=%u\n", dpdkctx.c_txq, ret, dpdkctx.port_id);
        goto out_err;
    }

    return PT_OK;
out_err:
    return PT_ERR;
}

////////////////////////////////////////////////
//// Flow identification
////////////////////////////////////////////////
int dpdk_flow_isolate(struct phytools_ctx * ptctx)
{
    int ret = 0;
    struct rte_flow_error flowerr;

    if(!ptctx)
        return PT_EILWAL;

    for(int i=0; i < RTE_MAX_ETHPORTS; i++)
    {
        if(ptctx->dpdk_dev[i].enabled)
        {
            if(rte_flow_isolate(ptctx->dpdk_dev[i].port_id, 1, &flowerr))
                rte_panic("Flow isolation failed: %s\n", flowerr.message);
        }
    }

    return PT_OK;

out_err:
    return PT_ERR;
}

int dpdk_setup_rules_ecpri(struct phytools_ctx *ptctx)
{
    int ret = 0, num_pattern = 0;
    struct rte_flow_attr attr;
    struct rte_flow_item patterns[4];
    struct rte_flow_action actions[2];
    struct rte_flow_error err;
    struct rte_flow_action_queue queue = {.index = 0 };
    struct rte_flow_item_eth eth_spec, eth_mask;
    struct rte_flow_item_vlan vlan_spec, vlan_mask;
    struct rte_flow_item_ipv4 ip_spec, ip_mask;
    struct dpdk_pipeline_ctx * dpdkctx;
    DECLARE_FOREACH_PIPELINE

    if(!ptctx)
        return PT_EILWAL;

    OPEN_FOREACH_PIPELINE
        dpdkctx = &(plctx->dpdkctx);

        //One RX for C-msg and one RX for U-msg
        for(int i=0; i<dpdkctx->rxq; i++)
        {
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

            eth_spec.src.addr_bytes[0] = dpdkctx->peer_eth_addr.addr_bytes[0];
            eth_spec.src.addr_bytes[1] = dpdkctx->peer_eth_addr.addr_bytes[1];
            eth_spec.src.addr_bytes[2] = dpdkctx->peer_eth_addr.addr_bytes[2];
            eth_spec.src.addr_bytes[3] = dpdkctx->peer_eth_addr.addr_bytes[3];
            eth_spec.src.addr_bytes[4] = dpdkctx->peer_eth_addr.addr_bytes[4];
            eth_spec.src.addr_bytes[5] = dpdkctx->peer_eth_addr.addr_bytes[5];

            eth_mask.src.addr_bytes[0] = 0xFF;
            eth_mask.src.addr_bytes[1] = 0xFF;
            eth_mask.src.addr_bytes[2] = 0xFF;
            eth_mask.src.addr_bytes[3] = 0xFF;
            eth_mask.src.addr_bytes[4] = 0xFF;
            eth_mask.src.addr_bytes[5] = 0xFF;

            eth_spec.dst.addr_bytes[0] = ptctx->dpdk_dev[dpdkctx->port_id].eth_addr.addr_bytes[0];
            eth_spec.dst.addr_bytes[1] = ptctx->dpdk_dev[dpdkctx->port_id].eth_addr.addr_bytes[1];
            eth_spec.dst.addr_bytes[2] = ptctx->dpdk_dev[dpdkctx->port_id].eth_addr.addr_bytes[2];
            eth_spec.dst.addr_bytes[3] = ptctx->dpdk_dev[dpdkctx->port_id].eth_addr.addr_bytes[3];
            eth_spec.dst.addr_bytes[4] = ptctx->dpdk_dev[dpdkctx->port_id].eth_addr.addr_bytes[4];
            eth_spec.dst.addr_bytes[5] = ptctx->dpdk_dev[dpdkctx->port_id].eth_addr.addr_bytes[5];

            eth_mask.dst.addr_bytes[0] = 0xFF;
            eth_mask.dst.addr_bytes[1] = 0xFF;
            eth_mask.dst.addr_bytes[2] = 0xFF;
            eth_mask.dst.addr_bytes[3] = 0xFF;
            eth_mask.dst.addr_bytes[4] = 0xFF;
            eth_mask.dst.addr_bytes[5] = 0xFF;
        
            eth_spec.type =rte_cpu_to_be_16(RTE_ETHER_TYPE_VLAN);
            eth_mask.type = 0xFFFF;

            vlan_spec.tci = rte_cpu_to_be_16(dpdkctx->vlan);
            
            // 0x007a1234: Message Type is 0x7a and Ecpri PCID is 0x1234
            // DISABLED: First queue is for C-plane
            // if(i == (plctx->dpdkctx.rxq-1)) ip_spec.hdr.src_addr = (uint32_t)(((uint32_t)ECPRI_MSG_TYPE_RTC) << 16 | ((uint32_t)plctx->flow_id));

            ip_spec.hdr.src_addr = (uint32_t)(((uint32_t)ECPRI_MSG_TYPE_IQ) << 16 | ((uint32_t)plctx->flow_list[i]));
            ip_spec.hdr.src_addr = rte_cpu_to_be_32(ip_spec.hdr.src_addr);
            ip_mask.hdr.src_addr = 0xFFFFFFFF;

            ip_spec.hdr.dst_addr = 0x0;
            ip_mask.hdr.dst_addr = 0x0;

            queue.index = dpdkctx->start_rxq + i;

            pt_info("Pipeline %d RX Queue: %d VLAN ID %d eCPRI flow value %08x Eth Src=%02X:%02X:%02X:%02X:%02X:%02X Eth Dst=%02X:%02X:%02X:%02X:%02X:%02X\n", 
                plctx->index, queue.index, dpdkctx->vlan, ip_spec.hdr.src_addr,
                eth_spec.src.addr_bytes[0], eth_spec.src.addr_bytes[1], eth_spec.src.addr_bytes[2],
                eth_spec.src.addr_bytes[3], eth_spec.src.addr_bytes[4], eth_spec.src.addr_bytes[5],
                eth_spec.dst.addr_bytes[0], eth_spec.dst.addr_bytes[1], eth_spec.dst.addr_bytes[2],
                eth_spec.dst.addr_bytes[3], eth_spec.dst.addr_bytes[4], eth_spec.dst.addr_bytes[5]
            );

            actions[0].type = RTE_FLOW_ACTION_TYPE_QUEUE;
            actions[0].conf = &queue;
            actions[1].type = RTE_FLOW_ACTION_TYPE_END;

            num_pattern = 0;

            // Ethernet rule
            patterns[num_pattern].type = RTE_FLOW_ITEM_TYPE_ETH;
            patterns[num_pattern].spec = &eth_spec;
            // patterns[num_pattern].mask = &eth_mask;
            num_pattern++;

            // VLAN rule
            patterns[num_pattern].type = RTE_FLOW_ITEM_TYPE_VLAN;
            patterns[num_pattern].spec = &vlan_spec;
            // patterns[num_pattern].mask = &vlan_mask;
            num_pattern++;

            // IP/eCPRI rule
            patterns[num_pattern].type = RTE_FLOW_ITEM_TYPE_IPV4;
            patterns[num_pattern].spec = &ip_spec;
            patterns[num_pattern].mask = &ip_mask;
            num_pattern++;

            patterns[num_pattern].type = RTE_FLOW_ITEM_TYPE_END;

            if (rte_flow_validate(dpdkctx->port_id, &attr, patterns, actions, &err))
                rte_panic("Invalid flow rule: %s\n", err.message);
            
            if(!rte_flow_create(dpdkctx->port_id, &attr, patterns, actions, &err))
                rte_panic("Invalid flow create rule: %s\n", err.message);
        }

    CLOSE_FOREACH_PIPELINE

    return PT_OK;
}

/* test the following:
 * - dst MAC matching our MAC addr
 * - src MAC matchin peerethaddr from JSON
 * - VLAN TCI matching vlan from JSON + flow_list entries */
int dpdk_setup_rules_vlan(struct phytools_ctx * ptctx)
{
    int ret = 0, num_pattern = 0;
    struct rte_flow_attr attr;
    struct rte_flow_item patterns[4];
    struct rte_flow_action actions[2];
    struct rte_flow_error err;
    struct rte_flow_action_queue queue = {.index = 0 };
    struct rte_flow_item_eth eth_spec, eth_mask;
    struct rte_flow_item_vlan vlan_spec, vlan_mask;
    struct dpdk_pipeline_ctx * dpdkctx;
    DECLARE_FOREACH_PIPELINE

    if(!ptctx)
        return PT_EILWAL;

    OPEN_FOREACH_PIPELINE
        dpdkctx = &(plctx->dpdkctx);

        //One RX for C-msg and one RX for U-msg
        for(int i=0; i<dpdkctx->rxq; i++)
        {
            memset(&attr, 0, sizeof(attr));
            memset(patterns, 0, sizeof(patterns));
            memset(actions, 0, sizeof(actions));
            memset(&eth_spec, 0, sizeof(eth_spec));
            memset(&eth_mask, 0, sizeof(eth_mask));
            memset(&vlan_spec, 0, sizeof(vlan_spec));
            memset(&vlan_mask, 0, sizeof(vlan_mask));
            attr.ingress = 1;

	    rte_ether_addr_copy(&dpdkctx->peer_eth_addr, &eth_spec.src);
	    memset(&eth_mask.src, 0xFF, sizeof(struct rte_ether_addr));

	    rte_ether_addr_copy(&ptctx->dpdk_dev[dpdkctx->port_id].eth_addr, &eth_spec.dst);
	    memset(&eth_mask.dst, 0xFF, sizeof(struct rte_ether_addr));

            eth_spec.type = rte_cpu_to_be_16(RTE_ETHER_TYPE_VLAN);
            eth_mask.type = 0xFFFF;

            vlan_spec.tci = rte_cpu_to_be_16(dpdkctx->vlan + (uint16_t)plctx->flow_list[i]);
	    vlan_mask.tci = 0xFFFF;

            queue.index = dpdkctx->start_rxq + i;

            pt_info("Pipeline %d RX Queue: %d VLAN ID %d Eth Src=%02X:%02X:%02X:%02X:%02X:%02X Eth Dst=%02X:%02X:%02X:%02X:%02X:%02X\n", 
                plctx->index, queue.index, rte_be_to_cpu_16(vlan_spec.tci),
                eth_spec.src.addr_bytes[0], eth_spec.src.addr_bytes[1], eth_spec.src.addr_bytes[2],
                eth_spec.src.addr_bytes[3], eth_spec.src.addr_bytes[4], eth_spec.src.addr_bytes[5],
                eth_spec.dst.addr_bytes[0], eth_spec.dst.addr_bytes[1], eth_spec.dst.addr_bytes[2],
                eth_spec.dst.addr_bytes[3], eth_spec.dst.addr_bytes[4], eth_spec.dst.addr_bytes[5]
            );

            actions[0].type = RTE_FLOW_ACTION_TYPE_QUEUE;
            actions[0].conf = &queue;
            actions[1].type = RTE_FLOW_ACTION_TYPE_END;

            num_pattern = 0;

            // Ethernet rule
            patterns[num_pattern].type = RTE_FLOW_ITEM_TYPE_ETH;
            patterns[num_pattern].spec = &eth_spec;
            patterns[num_pattern].mask = &eth_mask;
            num_pattern++;

            // VLAN rule
            patterns[num_pattern].type = RTE_FLOW_ITEM_TYPE_VLAN;
            patterns[num_pattern].spec = &vlan_spec;
            patterns[num_pattern].mask = &vlan_mask;
            num_pattern++;

            patterns[num_pattern].type = RTE_FLOW_ITEM_TYPE_END;

            if (rte_flow_validate(dpdkctx->port_id, &attr, patterns, actions, &err))
                rte_panic("Invalid flow rule: %s\n", err.message);

            if(!rte_flow_create(dpdkctx->port_id, &attr, patterns, actions, &err))
                rte_panic("Invalid flow create rule: %s\n", err.message);
        }

    CLOSE_FOREACH_PIPELINE

    return PT_OK;
}

int dpdk_setup_rules(struct phytools_ctx * ptctx)
{
    struct dpdk_pipeline_ctx &dpdkctx = ptctx->plctx[0].dpdkctx;
    if (dpdkctx.flow_ident_method == PT_FLOW_IDENT_METHOD_eCPRI)
	    return dpdk_setup_rules_ecpri(ptctx);
    return dpdk_setup_rules_vlan(ptctx);
}

////////////////////////////////////////////////
//// Setup NIC
////////////////////////////////////////////////

int dpdk_start_nic(struct phytools_ctx * ptctx)
{
    int ret = 0;
    if(!ptctx)
        return PT_EILWAL;

    for(int i=0; i < RTE_MAX_ETHPORTS; i++)
    {
        if(ptctx->dpdk_dev[i].enabled)
        {
            ret = rte_eth_dev_start(ptctx->dpdk_dev[i].port_id);
            if (ret != 0)
            {
                pt_err("rte_eth_dev_start: err=%d, port=%u\n", ret, ptctx->dpdk_dev[i].port_id);
                goto out_err;
            }
            // Cannot enable promislwous mode in flow isolation mode
            // rte_eth_promislwous_enable(ptctx->dpdk_dev[i].port_id);
        }
    }

    return PT_OK;

out_err:
    return PT_ERR;
}

//More stuff here....
int dpdk_finalize(struct phytools_ctx * ptctx)
{
    if(!ptctx)
        return PT_EILWAL;

    for(int i=0; i < RTE_MAX_ETHPORTS; i++)
    {
        if(ptctx->dpdk_dev[i].enabled)
        {
            pt_info("Closing port %d...\n", ptctx->dpdk_dev[i].port_id);
            rte_eth_dev_stop(ptctx->dpdk_dev[i].port_id);
            rte_eth_dev_close(ptctx->dpdk_dev[i].port_id);
        }
    }

    return PT_OK;
}

void dpdk_print_stats(struct phytools_ctx * ptctx)
{
    struct rte_eth_stats stats;
    
	if (ptctx->no_stats || !ptctx)
        return;

    for(int i=0; i < RTE_MAX_ETHPORTS; i++)
    {
        if(ptctx->dpdk_dev[i].enabled)
        {
            rte_eth_stats_get(ptctx->dpdk_dev[i].port_id, &stats);
            printf("\n================================================\n");
            printf("DPDK Stats Port %d\n", ptctx->dpdk_dev[i].port_id);
            printf("================================================\n");

            printf("RX queues %d:\n", ptctx->dpdk_dev[i].tot_rxq);
            for(int index_queue=0; index_queue < ptctx->dpdk_dev[i].tot_rxq; index_queue++)
                printf("\tQueue %d -> packets = %ld bytes = %ld dropped = %ld\n", index_queue,
                    stats.q_ipackets[index_queue], stats.q_ibytes[index_queue], stats.q_errors[index_queue]);
            printf("\t-------------------------------------------\n");
            printf("\tTot received packets: %ld Tot received bytes: %ld\n", stats.ipackets, stats.ibytes);

            printf("TX queues %d:\n", ptctx->dpdk_dev[i].tot_txq);
            for(int index_queue=0; index_queue < ptctx->dpdk_dev[i].tot_txq; index_queue++)
                printf("\tQueue %d -> packets = %ld bytes = %ld\n", index_queue, stats.q_opackets[index_queue], stats.q_obytes[index_queue]);
            printf("\t-------------------------------------------\n");
            printf("\tTot sent packets: %ld, Tot sent bytes: %ld\n", stats.opackets, stats.obytes);

            printf("Errors:\n");
            printf("\tRX packets dropped by the HW (RX queues are full) = %" PRIu64 "\n", stats.imissed);
            printf("\tTotal number of erroneous RX packets = %" PRIu64 "\n", stats.ierrors);
            printf("\tTotal number of RX mbuf allocation failures = %" PRIu64 "\n", stats.rx_nombuf);
            printf("\tTotal number of failed TX packets = %" PRIu64 "\n", stats.oerrors);
            printf("\n");
        }
    }
}

/*
 * Credits dpdk/testpmd
 */
void dpdk_copy_buf_to_pkt_segs(void* buf, unsigned len, struct rte_mbuf *pkt, unsigned offset)
{
    struct rte_mbuf *seg;
    void *seg_buf;
    unsigned copy_len;

    seg = pkt;
    while (offset >= seg->data_len) {
        offset -= seg->data_len;
        seg = seg->next;
    }
    copy_len = seg->data_len - offset;
    seg_buf = rte_pktmbuf_mtod_offset(seg, char *, offset);
    while (len > copy_len) {
        rte_memcpy(seg_buf, buf, (size_t) copy_len);
        len -= copy_len;
        buf = ((char*) buf + copy_len);
        seg = seg->next;
        seg_buf = rte_pktmbuf_mtod(seg, char *);
        copy_len = seg->data_len;
    }
    rte_memcpy(seg_buf, buf, (size_t) len);
}

void dpdk_copy_buf_to_pkt(void* buf, unsigned len, struct rte_mbuf *pkt, unsigned offset)
{
    if (offset + len <= pkt->data_len) {
        rte_memcpy(rte_pktmbuf_mtod_offset(pkt, char *, offset), buf, (size_t) len);
        return;
    }
    dpdk_copy_buf_to_pkt_segs(buf, len, pkt, offset);
}

/** Create IPv4 address */
#define IPv4(a, b, c, d) ((uint32_t)(((a) & 0xff) << 24) | (((b) & 0xff) << 16) | (((c) & 0xff) << 8)  | ((d) & 0xff))

void dpdk_setup_pkt_udp_ip_headers(struct rte_ipv4_hdr *ip_hdr, struct udp_hdr *udp_hdr, uint16_t pkt_data_len)
{
    uint16_t *ptr16;
    uint32_t ip_cksum;
    uint16_t pkt_len = pkt_data_len;

#if 0
    /*
     * Initialize UDP header.
     */
    pkt_len = (uint16_t) (pkt_data_len + sizeof(struct udp_hdr));
    udp_hdr->src_port = rte_cpu_to_be_16(DPDK_UDP_SRC_PORT);
    udp_hdr->dst_port = rte_cpu_to_be_16(DPDK_UDP_DST_PORT);
    udp_hdr->dgram_len      = rte_cpu_to_be_16(pkt_len);
    udp_hdr->dgram_cksum    = 0; /* No UDP checksum. */
#endif

    /*
     * Initialize IP header.
     */
    pkt_len = (uint16_t) (pkt_len + sizeof(struct rte_ipv4_hdr));
    ip_hdr->version_ihl   = DPDK_IP_VHL_DEF;
    ip_hdr->type_of_service   = 0;
    ip_hdr->fragment_offset = 0;
    ip_hdr->time_to_live   = DPDK_IP_DEFTTL;
    ip_hdr->next_proto_id = IPPROTO_UDP;
    ip_hdr->packet_id = 0;
    ip_hdr->total_length   = rte_cpu_to_be_16(pkt_len);
    ip_hdr->src_addr = rte_cpu_to_be_32(IPv4(10, 254, 0, 0));
    //Simulate RSS flow
    ip_hdr->dst_addr = rte_cpu_to_be_32(IPv4(10, 254, 0, 0));
    
    /*
     * Compute IP header checksum.
     */
    ptr16 = (unaligned_uint16_t*) ip_hdr;
    ip_cksum = 0;
    ip_cksum += ptr16[0]; ip_cksum += ptr16[1];
    ip_cksum += ptr16[2]; ip_cksum += ptr16[3];
    ip_cksum += ptr16[4];
    ip_cksum += ptr16[6]; ip_cksum += ptr16[7];
    ip_cksum += ptr16[8]; ip_cksum += ptr16[9];

    /*
     * Reduce 32 bit checksum to 16 bits and complement it.
     */
    ip_cksum = ((ip_cksum & 0xFFFF0000) >> 16) +
        (ip_cksum & 0x0000FFFF);
    if (ip_cksum > 65535)
        ip_cksum -= 65535;
    ip_cksum = (~ip_cksum) & 0x0000FFFF;
    if (ip_cksum == 0)
        ip_cksum = 0xFFFF;
    ip_hdr->hdr_checksum = (uint16_t) ip_cksum;
}

////////////////////////////////////////////////////////////
/// Memory management
////////////////////////////////////////////////////////////
void * dpdk_alloc_aligned_memory(size_t input_size, size_t *out_size, size_t page_size)
{
    size_t local_mem_size = input_size;
    void * ptr;

    if(input_size <= 0)
    {
        pt_err("Input input_size is invalid\n");
        goto fail;
    }

    if(!rte_is_power_of_2(page_size))
    {
        pt_err("Input page size (%zd) is not a power of 2\n", page_size);
        goto fail;
    }

    if(RTE_ALIGN(input_size, page_size) != input_size)
    {
        pt_err("Warning: input memory size (%zd) is not a multiple of page size (%zd). "
                        "Increasing total memory size to %zd\n",
                        input_size, page_size, RTE_ALIGN(input_size, page_size)
        );
        local_mem_size = RTE_ALIGN(input_size, page_size);
    }

    if(out_size != NULL)
        *out_size = local_mem_size;


    ptr = rte_zmalloc("rte_lw_mem_chunk", local_mem_size, page_size);
    if(ptr == NULL)
    {
        pt_err("rte_zmalloc error\n");
        goto fail;
    }

    return ptr;
fail:
    return NULL;
}

int dpdk_register_ext_mem(void * addr, size_t mem_size, size_t page_size, struct rte_eth_dev *dev)
{
    int ret=0;

    if(!addr || !dev)
    {
        pt_err("lw_register_ext_mem input error, addr=%p, dev=%p\n", addr, dev);
        goto fail;
    }

    ret = rte_extmem_register(addr, mem_size, NULL, 0, page_size);
    if (ret)
    {
        pt_err("unable to register addr 0x%p\n", addr);
        goto fail;
    }

    ret = rte_dev_dma_map(dev->device, addr, 0, mem_size);
    if (ret)
    {
        pt_err("unable to DMA map addr 0x%p for device %s\n", addr, dev->data->name);
        goto fail;
    }

    return 0;
fail:
    return 1;
}

int dpdk_unregister_ext_mem(void * addr, size_t mem_size, struct rte_eth_dev *dev)
{
    int ret=0;

    if(!addr || !dev)
    {
        pt_err("lw_unregister_ext_mem input error, addr=%p, dev=%p\n", addr, dev);
        goto fail;
    }

    ret = rte_dev_dma_unmap(dev->device, addr, 0, mem_size);
    if (ret)
    {
        pt_err("unable to DMA unmap addr 0x%p for device %s\n", addr, dev->data->name);
        goto fail;
    }

    ret = rte_extmem_unregister(addr, mem_size);
    if (ret)
    {
        pt_err("unable to unregister addr 0x%p\n", addr);
        goto fail;
    }

    return 0;

fail:
    return 1;
}
////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////
/// DPDK GPU
////////////////////////////////////////////////////////////
#ifdef LWDA_ENABLED
    struct lw_mempool_info * dpdk_create_lwmempool(const char * mp_name, 
                                                int mbufs_num, int cache_size, int mbufs_payload, 
                                                int port_id, int socket_id,
                                                lw_mempool_type memp_type)
    {
        if(
            !mp_name || mbufs_num <=0 || cache_size <=0 ||
            mbufs_payload <= RTE_PKTMBUF_HEADROOM || mbufs_payload >= DPDK_MAX_MBUFS_PAYLOAD || 
            port_id < 0 || socket_id < 0
        )
        {
            pt_err("Invalid input parameters");
            return NULL;
        }

        return lw_mempool_create(mp_name, mbufs_num, cache_size, 0, mbufs_payload, port_id, socket_id, memp_type);
    }

    struct rte_mempool * dpdk_get_mempool_from_lwmempool(struct lw_mempool_info * lw_mp)
    {
        if(!lw_mp)
        {
            pt_err("Invalid input parameters");
            return NULL;
        }

        return lw_get_rtemempool(lw_mp);
    }

#else
    struct lw_mempool_info * dpdk_create_lwmempool(const char * mp_name, 
                                                int mbufs_num, int cache_size, int mbufs_payload, 
                                                int port_id, int socket_id,
                                                lw_mempool_type memp_type) { 
        pt_warn("Can't use this function\n");
        return NULL; 
    }

    struct rte_mempool * dpdk_get_mempool_from_lwmempool(struct lw_mempool_info * lw_mp) {
        pt_warn("Can't use this function\n"); 
        return NULL;
    }
#endif
////////////////////////////////////////////////////////////
