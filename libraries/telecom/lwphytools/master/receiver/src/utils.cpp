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
#include "hdf5hpp.hpp"
#include <iostream>
#include <cstring>
#include <jsoncpp/json/json.h>
#include <jsoncpp/json/writer.h>
#include <fstream>

////////////////////////////////////////////////////////////
/// General init/finalize
////////////////////////////////////////////////////////////
int pt_finalize(struct phytools_ctx * ptctx) {
    int ret = PT_OK, index=0;
    DECLARE_FOREACH_PIPELINE

    if(!ptctx)
        return PT_EILWAL;

    lwphy_finalize(ptctx);
    lw_finalize();

    for(int i=0; i < PT_RING_ELEMS-1; i++)
    {
        ptctx->l2_info_ul[i].~l2_control_slot_info();
        ptctx->l2_info_dl[i].~l2_control_slot_info();
    }
    rte_free(ptctx->l2_info_ul);
    rte_free(ptctx->l2_info_dl);
    rte_ring_free(ptctx->ring_start_ul);
    rte_ring_free(ptctx->ring_start_dl);
    rte_ring_free(ptctx->ring_free_ul);
    rte_ring_free(ptctx->ring_free_dl);

    OPEN_FOREACH_PIPELINE
        for(index=0; index < plctx->tot_pusch_tv; index++)
            rte_free(plctx->pusch_tv_list[index].idata_h);
        rte_free(plctx->pusch_tv_list);
    CLOSE_FOREACH_PIPELINE

    //Close DPDK elw
    ret = dpdk_finalize(ptctx);
    if(ret != PT_OK)
    {
        fprintf(stderr, "dpdk_finalize returned error %d\n", ret);
        return ret;
    }

    rte_free(ptctx->plctx);

    return PT_OK;
}
////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////
/// Utils functions
////////////////////////////////////////////////////////////
char * pt_strtok(char *string, const char *delimiter)
{
    static char *last = NULL;
    char *endp;

    if (!string)
        string = last;
    if (!string || *string == 0)
        return last = NULL;

    endp = strpbrk(string, delimiter);
    if (endp) {
        *endp = '\0';
        last = endp + 1;
    } else
        last = NULL;
    return string;
}

////////////////////////////////////////////////////////////
/// Time functions
////////////////////////////////////////////////////////////
double timerdiff_us(uint64_t t_end, uint64_t t_start) {
    uint64_t timer_hz = rte_get_timer_hz(); //__thread
    return 1.e6*(double)(t_end - t_start)/timer_hz;
}

double timerdiff_ns(uint64_t t_end, uint64_t t_start) {
    // return (1.e9*((double)(t_end - t_start)))/rte_get_tsc_hz();
    return ((double)(t_end - t_start));// /rte_get_tsc_hz();
}

double get_us_from_ns(double colwert_us) {
    return (double)(colwert_us/1000);
}

void wait_ns(uint64_t ns)
{
    uint64_t end_t = get_ns() + ns, start_t = 0;
    while ((start_t = get_ns()) < end_t) {
        for (int spin_cnt = 0; spin_cnt < 1000; ++spin_cnt) {
            __asm__ __volatile__ ("");
        }
    }
}

void wait_s(int s)
{
    for(int index = 0; index < s; index++)
    {
        uint64_t end_t = get_ns() + 1000000000, start_t = 0;
        while ((start_t = get_ns()) < end_t) {
            for (int spin_cnt = 0; spin_cnt < 1000; ++spin_cnt) {
                __asm__ __volatile__ ("");
            }
        }
    }
}

int flush_gmem(struct pipeline_ctx * plctx) {
    uint8_t flush_value=0;

    PUSH_RANGE("FLUSH", LWLWTX_COL_12);
        flush_value = PT_ACCESS_ONCE(*((uint32_t*)plctx->flush_h));
        if(flush_value != PK_FLUSH_VALUE) {
            pt_err("flush_value %d != PK_FLUSH_VALUE %d\n", flush_value, PK_FLUSH_VALUE);
            goto err;
        }
        rte_mb();
    POP_RANGE;

    return PT_OK;
err:
    return PT_ERR;
}

int set_mbatch_status(struct pipeline_ctx * plctx, int start_mbatch, int num_mbatch, int value)
{
    int index=0;

    for(index=0; index < num_mbatch; index++)
    {
        if(plctx->measure_time >= PT_TIMER_PIPELINE)
            plctx->mbatch_meta[start_mbatch].t_mbatch_ready_start = get_ns();

        PT_ACCESS_ONCE(((uint32_t*)plctx->mbufs_batch_ready_flags_h)[start_mbatch]) = value;
        rte_wmb();
        if(flush_gmem(plctx) == PT_ERR)
            goto err;

        if(plctx->measure_time >= PT_TIMER_PIPELINE)
            plctx->mbatch_meta[start_mbatch].t_mbatch_ready_end = get_ns();

        start_mbatch = (start_mbatch+1)%PT_MBUFS_BATCH_TOT;
    }
    return PT_OK;
err:
    return PT_ERR;
}
////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////
//// Setup
////////////////////////////////////////////////////////////
int pt_init(struct phytools_ctx *ptctx, int argc, char **argv)
{
    int ret=PT_OK, eal_args = 0;
    char *prgname = argv[0];

    if(!ptctx)
        return PT_EILWAL;

    //Init EAL
    eal_args = rte_eal_init(argc, argv);
    if (eal_args < 0)
    {
        pt_err("Invalid EAL arguments\n");
        return PT_EILWAL;
    }

    argc -= eal_args;
    argv += eal_args; 

    //PT ctx initialized
    ptctx->init = 1;

    ret = pt_parse_optargs(ptctx, argc, argv, prgname);
    if (ret != PT_OK)
    {
        if(ret != PT_STOP)
            pt_err("Invalid command line options (ret=%d)\n", ret);
        return ret;
    }

    //https://doc.dpdk.org/guides-18.08/nics/mlx5.html
    //http://patches.dpdk.org/patch/30662/#61023
    //If set, doorbell register is IO-mapped (not cached) instead of being write-combining register (buffered)
    (getelw("MLX5_SHUT_UP_BF") == NULL) ? ptctx->flush_tx_write=1 : ptctx->flush_tx_write = !(atoi(getelw("MLX5_SHUT_UP_BF")));

    ptctx->l2_info_ul = (struct l2_control_slot_info *) rte_zmalloc(NULL, sizeof(struct l2_control_slot_info)*PT_RING_ELEMS, 0);
    if(ptctx->l2_info_ul == NULL)
        return PT_ERR;
    
    ptctx->l2_info_dl = (struct l2_control_slot_info *) rte_zmalloc(NULL, sizeof(struct l2_control_slot_info)*PT_RING_ELEMS, 0);
    if(ptctx->l2_info_dl == NULL)
        return PT_ERR;

    std::atomic<uint64_t> val(0);
    for(int i=0; i < SLOT_WRAP; i++)
        ptctx->slot_3gpp_ref_ts.push_back(val);

    return PT_OK;
}

int pt_setup_gpu(struct phytools_ctx * ptctx)
{
    lwdaError_t lwda_ret = lwdaSuccess;
    struct lwdaDeviceProp deviceProp;
    int totDevs=0, index=0;
    DECLARE_FOREACH_PIPELINE

    if(!ptctx)
        return PT_EILWAL;

    // *************** GPU CHECK ***************
    lwda_ret = lwdaGetDeviceCount(&totDevs);
    if(lwda_ret != lwdaSuccess)
    {
        pt_err("lwdaGetDeviceCount error %d\n", lwda_ret);
        return PT_ERR;            
    }

    OPEN_FOREACH_PIPELINE

        if(totDevs < plctx->lwdactx.gpu_id)
        {
            pt_err("Erroneous GPU device ID (%d). Tot GPUs: %d\n", plctx->lwdactx.gpu_id, totDevs);
            return PT_ERR;
        }

        lwda_ret = lwdaGetDeviceProperties(&deviceProp, plctx->lwdactx.gpu_id);
        if(lwda_ret != lwdaSuccess)
        {
            pt_err("lwdaGetDeviceProperties error %d\n", lwda_ret);
            return PT_ERR;
        }
        
        memcpy(plctx->lwdactx.gpu_name, deviceProp.name, PT_MIN(strlen(deviceProp.name), PT_GPU_NAME_LEN));
        plctx->lwdactx.pciBusID = deviceProp.pciBusID;
        plctx->lwdactx.pciDeviceID = deviceProp.pciDeviceID;
        plctx->lwdactx.pciDomainID = deviceProp.pciDomainID;
        plctx->lwdactx.clockRate = deviceProp.clockRate; //kHz
        plctx->lwdactx.Hz = int64_t(plctx->lwdactx.clockRate) * 1000;

    CLOSE_FOREACH_PIPELINE

    return PT_OK;
}

int pt_setup_rings(struct phytools_ctx * ptctx)
{
    int ring_flags=0, i=0;
    char ring_name[1024];
    DECLARE_FOREACH_PIPELINE

    if(!ptctx)
        return PT_EILWAL;

    OPEN_FOREACH_PIPELINE

    snprintf(ring_name, sizeof(ring_name), "startul");
    ptctx->ring_start_ul = rte_ring_create(ring_name, PT_RING_ELEMS, plctx->dpdkctx.socket_id, RING_F_EXACT_SZ | RING_F_SP_ENQ | RING_F_SC_DEQ);
    if(ptctx->ring_start_ul == NULL)
    {
        pt_err("rte_ring_create ring_start_ul error\n");
        return PT_ERR;
    }

    snprintf(ring_name, sizeof(ring_name), "startdl");
    ptctx->ring_start_dl = rte_ring_create(ring_name, PT_RING_ELEMS, plctx->dpdkctx.socket_id, RING_F_EXACT_SZ | RING_F_SP_ENQ | RING_F_SC_DEQ);
    if(ptctx->ring_start_dl == NULL)
    {
        pt_err("rte_ring_create ring_start_dl error\n");
        return PT_ERR;
    }

    snprintf(ring_name, sizeof(ring_name), "freeul");
    ptctx->ring_free_ul = rte_ring_create(ring_name, PT_RING_ELEMS, plctx->dpdkctx.socket_id, RING_F_EXACT_SZ | RING_F_SP_ENQ | RING_F_SC_DEQ);
    if(ptctx->ring_free_ul == NULL)
    {
        pt_err("rte_ring_create ring_free_ul error\n");
        return PT_ERR;
    }

    snprintf(ring_name, sizeof(ring_name), "freedl");
    ptctx->ring_free_dl = rte_ring_create(ring_name, PT_RING_ELEMS, plctx->dpdkctx.socket_id, RING_F_EXACT_SZ | RING_F_SP_ENQ | RING_F_SC_DEQ);
    if(ptctx->ring_free_dl == NULL)
    {
        pt_err("rte_ring_create ring_free_dl error\n");
        return PT_ERR;
    }

    if(ptctx->l2_info_ul)
    {
        for(i=0; i < PT_RING_ELEMS; i++)
        {
            new (&ptctx->l2_info_ul[i])l2_control_slot_info();
            if(rte_ring_enqueue(ptctx->ring_free_ul, (void*)&(ptctx->l2_info_ul[i])) != 0)
            {
                pt_err("rte_ring_enqueue failed at %d", i);
                return PT_ERR;       
            }
        }
    }
    else
    {
        pt_err("l2_info_ul is empy\n");
        return PT_ERR;
    }

    if(ptctx->l2_info_dl)
    {
        for(i=0; i < PT_RING_ELEMS; i++)
        {
            new (&ptctx->l2_info_dl[i])l2_control_slot_info();
            if(rte_ring_enqueue(ptctx->ring_free_dl, (void*)&(ptctx->l2_info_dl[i])) != 0)
            {
                pt_err("rte_ring_enqueue failed at %d", i);
                return PT_ERR;       
            }
        }

        // if(rte_ring_enqueue_bulk(ptctx->ring_start_free, (void**)ptctx->list_pipeline, PT_RING_ELEMS, NULL) == 0)
        // {
        //     pt_err("rte_ring_enqueue_bulk failed");
        //     return PT_ERR;
        // }
    }
    else
    {
        pt_err("l2_info_dl is empy\n");
        return PT_ERR;
    }
    CLOSE_FOREACH_PIPELINE

    return PT_OK;
}

//////////////////////////////////
///// Generic
//////////////////////////////////
int pt_print_ctx(struct phytools_ctx * ptctx) {
    
    DECLARE_FOREACH_PIPELINE

    if(!ptctx)
        return PT_EILWAL;

    OPEN_FOREACH_PIPELINE
        printf("\n================================================\n");
        printf("PIPELINE %d: %s, Port %d\n", index_pipeline, plctx->name.c_str(), plctx->dpdkctx.port_id);
        printf("================================================\n");
        printf("\tNetwork card %s, MAC addr: %02X:%02X:%02X:%02X:%02X:%02X, Driver %s, NUMA node %d, Socket %d\n",
            ptctx->dpdk_dev[plctx->dpdkctx.port_id].nic_info.device->name,
            ptctx->dpdk_dev[plctx->dpdkctx.port_id].eth_addr.addr_bytes[0], ptctx->dpdk_dev[plctx->dpdkctx.port_id].eth_addr.addr_bytes[1],
            ptctx->dpdk_dev[plctx->dpdkctx.port_id].eth_addr.addr_bytes[2], ptctx->dpdk_dev[plctx->dpdkctx.port_id].eth_addr.addr_bytes[3],
            ptctx->dpdk_dev[plctx->dpdkctx.port_id].eth_addr.addr_bytes[4], ptctx->dpdk_dev[plctx->dpdkctx.port_id].eth_addr.addr_bytes[5],
            ptctx->dpdk_dev[plctx->dpdkctx.port_id].nic_info.driver_name,
            ptctx->dpdk_dev[plctx->dpdkctx.port_id].nic_info.device->numa_node,
            plctx->dpdkctx.socket_id //This is the process' socket, not the NIC's socket
        );

        printf("\tFlow info: MAC addr=%02X:%02X:%02X:%02X:%02X:%02X VLAN=%d Port=%d Pcid list[%d]=",
            plctx->dpdkctx.peer_eth_addr.addr_bytes[0], plctx->dpdkctx.peer_eth_addr.addr_bytes[1],
            plctx->dpdkctx.peer_eth_addr.addr_bytes[2], plctx->dpdkctx.peer_eth_addr.addr_bytes[3],
            plctx->dpdkctx.peer_eth_addr.addr_bytes[4], plctx->dpdkctx.peer_eth_addr.addr_bytes[5],
            plctx->dpdkctx.vlan, plctx->dpdkctx.port_id, plctx->flow_tot
        );

        for(int index=0; index < plctx->flow_tot; index++)
            printf("%d ", plctx->flow_list[index]);
        printf("\n");

        printf("\tUplink: %s TTI: %d us Slots: %d With HDS: %s\n", 
                plctx->uplink > 0 ? "Yes" : "No",
                plctx->tti,
                plctx->uplink_slots,
                plctx->dpdkctx.hds == 1 ? "Yes" : "No"
            );

        printf("\tDownlink: %s TTI: %d us, Slots: %d\n", 
                plctx->downlink == 1 ? "Yes" : "No",
                plctx->tti,
                plctx->downlink_slots
            );

        printf("\tDPDK info: Mempool memory %s, %d mbufs, %dB RX payload, %dB TX payload, %d mbufs x burst, %dB cache, %d RX queues, %d TX queues\n",
            pt_mepmem_to_string(plctx->dpdkctx.memp_type), plctx->dpdkctx.memp_mbuf_num, 
            plctx->dpdkctx.mbuf_payload_size_rx, plctx->dpdkctx.mbuf_payload_size_tx, 
            plctx->dpdkctx.mbuf_x_burst, plctx->dpdkctx.memp_cache, plctx->dpdkctx.rxq, plctx->dpdkctx.txq
        );

        printf("\tGPU & LWCA\n\t\tGPU #%d %s, Bus ID %02x:%02x:%x, clock rate = %ld, Hz = %ld\n"
            "\t\tOrder kernel grid size %dB x %dT\n"
            , 
            plctx->lwdactx.gpu_id, plctx->lwdactx.gpu_name,
            plctx->lwdactx.pciBusID, plctx->lwdactx.pciDeviceID, plctx->lwdactx.pciDomainID,
            plctx->lwdactx.clockRate, plctx->lwdactx.Hz,
            plctx->lwdactx.order_kernel_blocks, plctx->lwdactx.order_kernel_threads
        );

        printf("\tPUSCH test vectors=%d\n", plctx->tot_pusch_tv);
        for(auto i=0; i < plctx->tot_pusch_tv; i++)
            printf("\t\t%d) %s\n", i, plctx->pusch_tv_list[i].file_name.c_str());
        printf("\n");

        printf("\tPDSCH test vectors=%d\n", plctx->tot_pdsch_tv);
        for(auto i=0; i < plctx->tot_pdsch_tv; i++)
            printf("\t\t%d) %s\n", i, plctx->pdsch_tv_list[i].file_name.c_str());
        printf("\n");

        printf("\tconfig test vectors=%d\n", plctx->tot_config_tv);
        for(auto i=0; i < plctx->tot_config_tv; i++)
            printf("\t\t%d) %s\n", i, plctx->config_tv_list[i].file_name.c_str());
        printf("\n");

        printf("\tOther settings\n"
            "\t\tTimers enabled = %s (level %d)\n"
            "\t\tValidation = %x\n"
            "\t\tDump PUSCH input = %s\n"
            "\t\tDump PUSCH output = %s\n"
            "\t\tDump DL output = %s\n"
            "\t\tFirst AP only = %s\n"
            "\t\t3GPP max slot = %d\n"
            "\t\tTX C/U-plane delayed TTI = %d\n"
            ,
            pt_timers_to_string(plctx->measure_time), plctx->measure_time,
            plctx->validation,
            plctx->dump_pusch_input == 1 ? "Yes" : "No",
            plctx->dump_pusch_output == 1 ? "Yes" : "No",
            plctx->dump_dl_output == 1 ? "Yes" : "No",
            plctx->first_ap_only == 1 ? "Yes" : "No",
            plctx->slot_num_max_3gpp,
            plctx->sync_tx_tti
        );

        printf("\tPUSCH slot numbers: [ ");
        for(auto i=0; i < PT_MAX_SLOTS_X_CHANNEL; i++)
        {
            if(plctx->pusch_slot_list[i] == -1)
                break;
            printf("%d ", plctx->pusch_slot_list[i]);
        }    
        printf(" ]\n");

        printf("\tPBCH slot numbers: [ ");
        for(auto i=0; i < PT_MAX_SLOTS_X_CHANNEL; i++)
        {
            if(plctx->pbch_slot_list[i] == -1)
                break;
            printf("%d ", plctx->pbch_slot_list[i]);
        }    
        printf(" ]\n");

        printf("\tPDSCH slot numbers: [ ");
        for(auto i=0; i < PT_MAX_SLOTS_X_CHANNEL; i++)
        {
            if(plctx->pdsch_slot_list[i] == -1)
                break;
            printf("%d ", plctx->pdsch_slot_list[i]);
        }    
        printf(" ]\n");

        printf("\tPDCCH UL slot numbers: [ ");
        for(auto i=0; i < PT_MAX_SLOTS_X_CHANNEL; i++)
        {
            if(plctx->pdcch_ul_slot_list[i] == -1)
                break;
            printf("%d ", plctx->pdcch_ul_slot_list[i]);
        }    
        printf(" ]\n");

        printf("\tPDCCH DL slot numbers: [ ");
        for(auto i=0; i < PT_MAX_SLOTS_X_CHANNEL; i++)
        {
            if(plctx->pdcch_dl_slot_list[i] == -1)
                break;
            printf("%d ", plctx->pdcch_dl_slot_list[i]);
        }    
        printf(" ]\n");

        printf("\n");
    CLOSE_FOREACH_PIPELINE

    printf("\n");

    return PT_OK;
}

int pt_prepare_cplane_messages(rte_unique_mbufs &mbufs_c, int mbufs_num, int other_ap_prbs,
                uint8_t ecpriSeqid_c, uint8_t frameId, uint8_t subFrameId, uint8_t slotId,
                int startPrbc, int numPrbc, int numSym, int startSym,
                enum oran_pkt_dir direction,
                struct rte_ether_addr src_eth_addr,
                struct pipeline_ctx * plctx)
{
    uint8_t * payload;
    uint16_t vlan = plctx->dpdkctx.vlan;
    if (plctx->dpdkctx.flow_ident_method == PT_FLOW_IDENT_METHOD_VLAN && direction == DIRECTION_DOWNLINK)
    {
        vlan = plctx->dpdkctx.vlan + 1;
    }

    if(!plctx)
        return PT_EILWAL;

    for (int i = 0; i < plctx->flow_tot * PT_PKT_X_CMSG; ++i)
    {
        if(other_ap_prbs > 0 && i > 0)
            numPrbc = other_ap_prbs;

        struct rte_mbuf *m = (mbufs_c.get())[i];
        if (m == NULL)
            return PT_ERR;

        payload = (uint8_t *)DPDK_GET_MBUF_ADDR(m, 0);
    
        if(oran_create_cmsg_uldl((uint8_t**)&payload, 
                    src_eth_addr, 
                    plctx->dpdkctx.peer_eth_addr, vlan, 
                    //(int)(ORAN_CMSG_ULDL_UNCOMPRESSED_SECTION_OVERHEAD - ORAN_CMSG_HDR_OFFSET),
                    20, //Section size
                    plctx->flow_list[(i /*/PT_PKT_X_CMSG*/)], ecpriSeqid_c,
                    direction, 
                    frameId, subFrameId, slotId, startSym,
                    CSEC_ULDL, ORAN_DEF_SECTION_ID, 
                    startPrbc, numPrbc, numSym,
                    ORAN_REMASK_ALL, ORAN_EF_NO, ORAN_BEAMFORMING_NO))
        {
            pt_err("oran_create_cmsg_uldl error\n");
            set_force_quit(force_quit);
            return PT_ERR;
        }            

        m->nb_segs = 1;
        m->next = NULL;
        m->l2_len = sizeof(struct rte_ether_hdr);
        m->port = plctx->dpdkctx.port_id;
        m->ol_flags = 0;
        m->data_len = ORAN_CMSG_ULDL_UNCOMPRESSED_SECTION_OVERHEAD;
        m->pkt_len = m->data_len;
        rte_pktmbuf_reset_headroom(m);
        rte_mbuf_sanity_check(m, 1);
        // rte_pktmbuf_dump(stdout, m, ORAN_CMSG_ULDL_UNCOMPRESSED_SECTION_OVERHEAD);
    }
    
    return PT_OK;
}

using namespace std;

int pt_dump_pusch_output(std::string json_file, uint8_t * buffer, size_t buffer_size)
{
    ofstream out_fid;
    char * hex_output;
    int x=0;

    if(!buffer)
        return PT_EILWAL;

    hex_output = (char *) rte_zmalloc(NULL, buffer_size*4*sizeof(uint8_t), 0);
    if(hex_output == NULL)
        return PT_ERR;

    try
    {
        out_fid.open(json_file);
        for(int i=0; i < (int)buffer_size; i++)
            x += sprintf(hex_output + x, "%02x", buffer[i]);   

        out_fid << hex_output;
        out_fid.close();
    }
    catch (std::exception& e) {
		pt_err("%s\n", e.what());
        return PT_ERR;
	}
    catch (...) {
		pt_err("Uncaught exception");
        return PT_ERR;
	}

    rte_free(hex_output);
    return PT_OK;
}

int pt_dump_slot_buffer(std::string json_file, uint8_t * buffer, size_t buffer_size, size_t prb_sz, struct pipeline_ctx * plctx) {
    Json::Value ant_obj;
    Json::Value sym_obj;
    Json::Value prb_obj;
    string prb_string;
    Json::Reader reader;
    Json::Value root;
    Json::Value resultValue;
    char * prb_hex_string;
    unsigned char * tmp_char;
    ofstream json_fid;

    if(!buffer)
        return PT_EILWAL;
    
    prb_hex_string = (char *) rte_zmalloc(NULL, (prb_sz*4)+1, 0);
    if(prb_hex_string == NULL)
        return PT_ERR;

    try
    {
        json_fid.open(json_file);
        Slot slot_output = buffer_to_slot(buffer, buffer_size, plctx->flow_tot, prb_sz);
        for(int i=0; i < plctx->flow_tot; i++)
        {
            ant_obj["antenna"][i]["id"] = plctx->flow_list[i];
            for(int j=0; j < SLOT_NUM_SYMS; j++)
            {
                for(int k=0; k < slot_output.prbs_per_symbol; k++)
                {
                    prb_obj[k]["id"] = k;
                    tmp_char = (unsigned char*)(slot_output.ptrs[i][j][k]);
                    int x=0;
                    for(int z=0; z < (int)prb_sz; z++)
                    {
                        x += sprintf(prb_hex_string + x, "%02x", tmp_char[z]);
                        // if(j == 0 && k == 0) printf("%02x ", tmp_char[z]);
                    }

                    prb_string.assign((const char*)prb_hex_string, prb_sz*2);
                    prb_obj[k]["value"] = prb_string;
                    // out_obj["antenna"]["symbol"]["prb"] = k;
                    // out_obj["antenna"]["symbol"]["prb"][k] = slot_output.ptrs[i][j][k]; //antenna, symbol, prb
                }

                sym_obj["id"] = j;
                sym_obj["prb"] = prb_obj;
                ant_obj["antenna"][i]["symbol"][j] = sym_obj;
            }
        }

        printf("Writing dump data...\n");
        // cout<<"creating nested Json::Value Example pretty print: "
        //     <<endl<<ant_obj.toStyledString()
        //     <<endl;

        Json::StyledWriter styledWriter;
        json_fid << styledWriter.write(ant_obj);

        json_fid.close();
    }
    catch (std::exception& e) {
		pt_err("%s\n", e.what());
        return PT_ERR;
	}
    catch (...) {
		pt_err("Uncaught exception");
        return PT_ERR;
	}

    rte_free(prb_hex_string);

    return PT_OK;
}

int pt_set_max_thread_priority()
{
	pthread_t t = pthread_self();
	struct sched_param schedprm;
    schedprm.sched_priority = sched_get_priority_max(SCHED_FIFO);
	int ret = pthread_setschedparam(t, SCHED_FIFO, &schedprm);
	if (ret != 0)
		do_throw("Could not set max thread priority");
	int schedpol;
	ret = pthread_getschedparam(t, &schedpol, &schedprm);
	if (ret != 0)
		do_throw("Could not get thread scheduling info");

	if (schedpol != SCHED_FIFO)
		do_throw("Failed to apply SCHED_FIFO policy");
	return schedprm.sched_priority;
}

int pt_get_thread_priority()
{
    pthread_t t = pthread_self();
	struct sched_param schedprm;
    int ret=0, schedpol;
	ret = pthread_getschedparam(t, &schedpol, &schedprm);
	if (ret != 0)
		do_throw("Could not get thread scheduling info");
	return schedprm.sched_priority;
}

void pt_increase_slot(uint8_t& frame, uint8_t& subframe, uint8_t& slot)
{
    /* frame Id wraps an 8-bit unsigned int naturally */
    /* subframe Id wraps at 10 - i.e. 10 subframes in a frame */
    /* slot Id wraps based on the numerology;
        * FIXME here we only consider numerology 1, in which there are two
        * slots per subframe */
    if (slot++ == 1) {
        slot = 0;
        if (subframe++ == 9) {
            subframe = 0;
            frame++;
        }
    }
}

uint32_t pt_checksum_adler32(uint8_t * i_buf, size_t i_elems)
{
    uint32_t a = 1, b = 0;

    // Process each byte of the data in order
    for(int index = 0; index < (int)i_elems; ++index)
    {
        a = (a + i_buf[index]) % MOD_CHECKSUM_ADLER32;
        // b = (b + a) % MOD_CHECKSUM_ADLER32;
        b = (b + ((i_elems-index)*i_buf[index])) % MOD_CHECKSUM_ADLER32;
    }
    b += i_elems;
    b = (b%MOD_CHECKSUM_ADLER32);
    
    return (b << 16) | a;
}
