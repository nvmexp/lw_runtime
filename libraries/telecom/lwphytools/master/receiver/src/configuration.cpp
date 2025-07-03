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
#include "json_parser.hpp"

Json::Value jtree;

using namespace std;

const char short_options[] =
    "j:"  /* JSON config file path */
    "h"   /* help */
;

struct option long_options[] =
{
    {LGOPT_help,    no_argument,        NULL,   SHOPT_help},
    {LGOPT_json,    required_argument,  NULL,   SHOPT_json},
    {0, 0, 0, 0}
};

inline void jsonp_assign_dpdk_memory(Json::Value jtree, enum lw_mempool_type& value) {
    value = (jtree[JVALUE_PIPELINE_DPDK_MEMORY].asUInt() == 1 ? LW_MEMP_DEVMEM : LW_MEMP_HOST_PINNED);
}
    
void pt_usage(const char *prgname)
{
    printf("\n\n%s [EAL options] -- [lwPHYTools options] -- [lwPHY options]\n"
        "\n[lwPHYTools Receiver options]:\n"
        "\t-%c|--%-20s JSON config file\n"
        "\t-%c|--%-20s HELP: Print this help\n"
        "\n[lwPHY options]:\n"
        "\t--T\tNumber of transport blocks\n"
        "\t--L\tNumber of layers\n"
        "\t--C\tNumber of code blocks per transport block\n"
        "\t--I\tSize of input layer\n"
        "\t--S\tTransport block size\n"
        "\t--F\tNumber of filler bits per code block\n"
        "\t--D\tDisable descrambling\n"
        "\n"
        ,
        prgname,
        SHOPT_json, LGOPT_json,
        SHOPT_help, LGOPT_help
    );
}

static int pt_create_pipelines(struct phytools_ctx * ptctx) {
    DECLARE_FOREACH_PIPELINE

    if(!ptctx || ptctx->num_pipelines <= 0)
        return PT_EILWAL;

    ptctx->plctx = (struct pipeline_ctx *) rte_zmalloc(NULL, sizeof(struct pipeline_ctx) * ptctx->num_pipelines, 0);
    if(ptctx->plctx == NULL)
    {
        pt_err("ptctx->plctx == NULL\n");
        return PT_ERR;
    }

    //////////////////////////////////////////////////////////////////////////////
    //// Init common parameters
    //////////////////////////////////////////////////////////////////////////////
    for(int i=0; i < RTE_MAX_ETHPORTS; i++)
    {
        ptctx->dpdk_dev[i].tot_rxq = 0;
        ptctx->dpdk_dev[i].tot_txq = 0;
        ptctx->dpdk_dev[i].port_id = i;
        ptctx->dpdk_dev[i].tot_rxd = 0;
        ptctx->dpdk_dev[i].tot_txd = 0;
        ptctx->dpdk_dev[i].enabled = 0;

        ptctx->dpdk_dev[i].port_conf.rxmode.split_hdr_size      = 0;
        ptctx->dpdk_dev[i].port_conf.rxmode.max_rx_pkt_len      = DPDK_MAX_MBUFS_PAYLOAD;
        ptctx->dpdk_dev[i].port_conf.rxmode.offloads            = DEV_RX_OFFLOAD_JUMBO_FRAME;
        ptctx->dpdk_dev[i].port_conf.rxmode.mq_mode             = ETH_MQ_RX_NONE;
        ptctx->dpdk_dev[i].port_conf.txmode.mq_mode             = ETH_MQ_TX_NONE;
        ptctx->dpdk_dev[i].port_conf.txmode.offloads            = 0;
    }

    //Setup default values for general PT context
    ptctx->slot_chunk_tot                                       = 14;
    ptctx->slot_chunk_size                                      = 4096;
    ptctx->tot_pkts_x_batch                                     = PT_ORDER_PKTS_BUFFERING;
    ptctx->no_stats                                             = 0;
    ptctx->controller                                           = CONTROLLER_DPDK;

    //https://doc.dpdk.org/guides-18.08/nics/mlx5.html
    //http://patches.dpdk.org/patch/30662/#61023
    //If set, doorbell register is IO-mapped (not cached) instead of being write-combining register (buffered)
    (getelw("MLX5_SHUT_UP_BF") == NULL) ? ptctx->flush_tx_write=1 : ptctx->flush_tx_write = !(atoi(getelw("MLX5_SHUT_UP_BF")));


    OPEN_FOREACH_PIPELINE

        plctx->index                                            = index_pipeline;
        plctx->name.assign("pipeline"+index_pipeline);
        
        //////////////////////////////////////////////////////////////////////////////
        //// DPDK
        //////////////////////////////////////////////////////////////////////////////
        plctx->dpdkctx.socket_id                                = rte_socket_id();
        plctx->dpdkctx.rxq                                      = 1;
        plctx->dpdkctx.txq                                      = 1;
        plctx->dpdkctx.rxd                                      = DPDK_RX_DESC_DEFAULT;
        plctx->dpdkctx.txd                                      = DPDK_TX_DESC_DEFAULT;
        plctx->dpdkctx.c_txq                                    = 0;
        plctx->dpdkctx.dl_txq                                   = 1;
        plctx->dpdkctx.ts_txq                                   = 2;
        plctx->dpdkctx.memp_type                                = LW_MEMP_HOST_PINNED;
        plctx->dpdkctx.memp_num                                 = 1;
        plctx->dpdkctx.memp_cache                               = RTE_MEMPOOL_CACHE_MAX_SIZE;
        plctx->dpdkctx.memp_mbuf_num                            = DPDK_DEF_MBUF_MP;
        plctx->dpdkctx.mbuf_payload_size_rx                     = PT_MBUF_PAYLOAD_SIZE;
        plctx->dpdkctx.mbuf_payload_size_tx                     = PT_MBUF_PAYLOAD_SIZE;
        plctx->dpdkctx.mbuf_x_burst                             = 64;
        plctx->dpdkctx.vlan                                     = 1;
        plctx->dpdkctx.port_id                                  = 0; //Different device!
        plctx->dpdkctx.peer_eth_addr.addr_bytes[0]              = 0x01;
        plctx->dpdkctx.peer_eth_addr.addr_bytes[1]              = 0x02;
        plctx->dpdkctx.peer_eth_addr.addr_bytes[2]              = 0x03;
        plctx->dpdkctx.peer_eth_addr.addr_bytes[3]              = 0x04;
        plctx->dpdkctx.peer_eth_addr.addr_bytes[4]              = 0x05;
        plctx->dpdkctx.peer_eth_addr.addr_bytes[5]              = (0x06 + index_pipeline);
        plctx->dpdkctx.hds                                      = 0;

        plctx->ul_num_processed_slots                           = 0;
        plctx->dl_num_processed_slots                           = 0;
        plctx->tot_pusch_tv                                     = 0;
        plctx->tot_pdsch_tv                                     = 0;
        plctx->tot_config_tv                                    = 0;
        plctx->measure_time                                     = PT_TIMER_NO;

        ///////////////////////////////////////
        //// Receiver
        ///////////////////////////////////////
        plctx->validation                                       = 0;
        plctx->lwphy_descrambling                               = 1;
        plctx->lwdactx.gpu_id                                   = 0;
        plctx->lwdactx.order_kernel_blocks                      = PK_LWDA_BLOCKS;
        plctx->lwdactx.order_kernel_threads                     = PK_LWDA_THREADS;
        plctx->stream_ul                                        = 0;
        plctx->stream_dl                                        = 0;
        plctx->rx_batching_us                                   = PT_RX_DEFAULT_TIMER; //ideally 2 symbols
        plctx->flow_tot                                         = 0;
        plctx->uplink                                           = 0;
        plctx->downlink                                         = 0;
        plctx->tti                                              = 500;
        plctx->downlink_slots                                   = -1;
        plctx->uplink_slots                                     = -1;
        plctx->dump_pusch_input                                 = 0;
        plctx->dump_pusch_output                                = 0;
        plctx->dump_dl_output                                   = 0;
        plctx->first_ap_only                                    = 0;
        plctx->wait_downlink_slot                               = 8;
        plctx->slot_num_max_3gpp                                = PT_MAX_SLOTS_X_CHANNEL;
        plctx->sync_tx_tti                                      = 4;
        plctx->tv_slot_3gpp                                     = -1;
        plctx->ul_cplane_delay                                  = 300;

        for(int index=0; index < PT_MAX_SLOTS_X_CHANNEL; index++)
        {
            plctx->pbch_slot_list[index] = -1;
            plctx->pusch_slot_list[index] = -1;
            plctx->pdsch_slot_list[index] = -1;
            plctx->pdcch_ul_slot_list[index] = -1;
            plctx->pdcch_dl_slot_list[index] = -1;
        }

        plctx->pusch_tv_list = (struct tv_info *) rte_zmalloc(NULL, sizeof(struct tv_info) * PT_MAX_INPUT_FILES, 0);
        if(plctx->pusch_tv_list == NULL)
        {
            pt_err("Pipeline %d pusch_tv_list is NULL\n", index_pipeline);
            return PT_ERR;
        }

        plctx->pdsch_tv_list = (struct tv_info *) rte_zmalloc(NULL, sizeof(struct tv_info) * PT_MAX_INPUT_FILES, 0);
        if(plctx->pdsch_tv_list == NULL)
        {
            pt_err("Pipeline %d pdsch_tv_list is NULL\n", index_pipeline);
            return PT_ERR;
        }

        plctx->config_tv_list = (struct tv_info *) rte_zmalloc(NULL, sizeof(struct tv_info) * PT_MAX_INPUT_FILES, 0);
        if(plctx->config_tv_list == NULL)
        {
            pt_err("Pipeline %d config_tv_list is NULL\n", index_pipeline);
            return PT_ERR;
        }

        plctx->config_tv_list = (struct tv_info *) rte_zmalloc(NULL, sizeof(struct tv_info) * PT_MAX_INPUT_FILES, 0);
        if(plctx->config_tv_list == NULL)
        {
            pt_err("Pipeline %d config_tv_list is NULL\n", index_pipeline);
            return PT_ERR;
        }
    CLOSE_FOREACH_PIPELINE

    return PT_OK;
}

static int pt_validate_optargs(struct phytools_ctx * ptctx)
{
    DECLARE_FOREACH_PIPELINE

    if(!ptctx)
        return PT_EILWAL;

    OPEN_FOREACH_PIPELINE

        // DPDK validation
        if(plctx->dpdkctx.mbuf_x_burst <= DPDK_MIN_PKTS_X_BURST)
        {
            pt_err("Pipeline %d: Minimum value allowed for mbufs x burst is %d\n", plctx->index, DPDK_MIN_PKTS_X_BURST);
            return PT_ERR;
        }

        if(plctx->dpdkctx.memp_mbuf_num <= 0)
        {
            pt_err("Pipeline %d: Invalid value of mbufs x mempool\n", plctx->index);
            return PT_ERR;
        }

        if(plctx->dpdkctx.mbuf_payload_size_rx <= (sizeof(struct rte_ether_hdr) + sizeof(struct rte_vlan_hdr) + 1))
        {
            pt_err("Pipeline %d: Invalid mbuf payload size. Minimum is %zd\n", 
                    plctx->index, (sizeof(struct rte_ether_hdr) + sizeof(struct rte_vlan_hdr) + 1)); //DPDK_HDR_ADDR_SIZE
            return PT_ERR;
        }

        if(plctx->dpdkctx.memp_type != LW_MEMP_DEVMEM && plctx->dpdkctx.memp_type != LW_MEMP_HOST_PINNED)
        {
            pt_err("Pipeline %d: Invalid mempool external memory type\n", plctx->index);
            return PT_ERR;
        }

        if(plctx->uplink && plctx->tot_pusch_tv  <= 0)
        {
            pt_err("Pipeline %d: You need to specify at least 1 UL test vector file\n", plctx->index);
            return PT_ERR;
        }

        if(plctx->lwdactx.order_kernel_blocks == 0 || plctx->lwdactx.order_kernel_threads == 0)
        {
            pt_err("Erroneous LWCA parameters: order_kernel_blocks=%d, order_kernel_threads=%d\n",
                        plctx->lwdactx.order_kernel_blocks, plctx->lwdactx.order_kernel_threads);
            return PT_ERR;
        }
        
        //Temporary patch: Order kernel can support 1 LWCA Block only for the moment
        if(plctx->lwdactx.order_kernel_blocks != PK_LWDA_BLOCKS)
        {
            pt_warn("Order kernel can support only %d LWCA blocks while you specified %d. Fixing this number\n", PK_LWDA_BLOCKS, plctx->lwdactx.order_kernel_blocks);
            plctx->lwdactx.order_kernel_blocks = PK_LWDA_BLOCKS;
        }

        if(plctx->uplink == 0 && plctx->downlink == 0)
        {
            pt_err("Uplink or Downlink must be specified\n");
            return PT_ERR;
        }

        if(plctx->uplink == 0 && plctx->downlink == 1)
        {
            pt_info("Enabled DL only. Getting rid of RX queues\n");
            //WAR DPDK 19.11, downlink only: Create at least 1 RX queue
            for(int i=0; i < RTE_MAX_ETHPORTS; i++)
            {
                if(ptctx->dpdk_dev[i].enabled == 1)
                {
                    ptctx->dpdk_dev[i].tot_rxq = 1;
                    ptctx->dpdk_dev[i].port_conf.rxmode.max_rx_pkt_len = 256;
                }
            }
            plctx->dpdkctx.rxq = 1;
        }

#ifndef LWPHYCONTROLLER
        if(ptctx->controller == CONTROLLER_LWPHY)
        {
            pt_err("lwPHY Controller not built with lwPHYTools\n");
            return PT_ERR;
        }
#endif

#ifdef LWPHYCONTROLLER
        if(ptctx->controller == CONTROLLER_LWPHY && !(plctx->validation & PT_VALIDATION_CRC))
        {
            plctx->validation |= PT_VALIDATION_CRC;
            pt_warn("lwPHYController requires CRC validation. Enabling it: %x\n", plctx->validation);
        }
#endif
        if((plctx->dump_pusch_input || plctx->dump_dl_output) && plctx->dpdkctx.memp_type == LW_MEMP_DEVMEM)
        {
            pt_err("Can't dump GPU memory. Forcing host pinned memory\n");
            plctx->dpdkctx.memp_type = LW_MEMP_HOST_PINNED;
        }

    CLOSE_FOREACH_PIPELINE
    
    if((ptctx->totCores > rte_lcore_count()) || ptctx->totCores > DPDK_MAX_CORES)
    {
        pt_err("Minimum cores required (%d), cores launched (%d) max cores allowed %d\n",
                                            ptctx->totCores, rte_lcore_count(), DPDK_MAX_CORES);
        return PT_ERR;
    }

    return PT_OK;
}

static int pt_parse_tv_uplink(struct pipeline_ctx * plctx, Json::Value tv_list) {
    int index=0;
    vector<string> tv_files;

    if(!plctx || tv_list.empty())
        return PT_EILWAL;

    jsonp_parse_tv_uplink(tv_list, plctx->tot_pusch_tv,  tv_files);
    if(plctx->tot_pusch_tv  <= 0 || plctx->tot_pusch_tv  >= PT_MAX_INPUT_FILES)
        return PT_ERR;

    for(auto index=0; index < plctx->tot_pusch_tv;  index++)
    {
        plctx->pusch_tv_list[index].file_name = tv_files[index];
        plctx->pusch_tv_list[index].index=index;
        plctx->pusch_tv_list[index].size=0;
        plctx->pusch_tv_list[index].idata_h=NULL;
        plctx->pusch_tv_list[index].odata_h=NULL;
    }

    return PT_OK;
}

static int pt_parse_tv_downlink(struct pipeline_ctx * plctx, Json::Value tv_list) {
    int index=0;
    vector<string> tv_files;

    if(!plctx || tv_list.empty())
        return PT_EILWAL;

    jsonp_parse_tv_downlink(tv_list, plctx->tot_pdsch_tv, tv_files);
    if(plctx->tot_pdsch_tv <= 0 || plctx->tot_pdsch_tv >= PT_MAX_INPUT_FILES)
        return PT_ERR;

    for(auto index=0; index < plctx->tot_pdsch_tv; index++)
    {
        plctx->pdsch_tv_list[index].file_name = tv_files[index];
        plctx->pdsch_tv_list[index].index=index;
        plctx->pdsch_tv_list[index].size=0;
        plctx->pdsch_tv_list[index].idata_h=NULL;
        plctx->pdsch_tv_list[index].odata_h=NULL;
    }

    return PT_OK;
}

static int pt_parse_tv_config(struct pipeline_ctx * plctx, Json::Value tv_list) {
    int index=0;
    vector<string> tv_files;

    if(!plctx || tv_list.empty())
        return PT_EILWAL;

    jsonp_parse_tv_config(tv_list, plctx->tot_config_tv, tv_files);
    if(plctx->tot_config_tv <= 0 || plctx->tot_config_tv >= PT_MAX_INPUT_FILES)
        return PT_ERR;

    for(auto index=0; index < plctx->tot_config_tv; index++)
    {
        plctx->config_tv_list[index].file_name = tv_files[index];
        plctx->config_tv_list[index].index=index;
        plctx->config_tv_list[index].size=0;
        plctx->config_tv_list[index].idata_h=NULL;
        plctx->config_tv_list[index].odata_h=NULL;
    }

    return PT_OK;
}

static int pt_parse_config_json_file(struct phytools_ctx * ptctx, const char * q_arg)
{
    int index_tv=0;
    string tmp;
    
    DECLARE_FOREACH_PIPELINE

    if(!ptctx || !q_arg)
        return PT_EILWAL;

    if(access( q_arg, R_OK ) != 0 || strlen(q_arg) > PT_DEFAULT_CHAR_BUFFER)
        return PT_ERR;

    string input_file_name = q_arg;

    jtree = jsonp_return_tree(input_file_name);
    if(jtree == Json::nullValue)
    {
        pt_err("jsonp_return_tree returned NULL object\n");
        return PT_EILWAL;
    }

    jsonp_assign_pipelines_number(jtree, ptctx->num_pipelines);

    const Json::Value& pipelines = jtree[JVALUE_PIPELINES]; // array of pipelines
    
    if(pt_create_pipelines(ptctx) != PT_OK)
    {
        pt_err("pt_create_pipelines error\n");
        return PT_ERR;
    }

    //Generic options
    OPEN_FOREACH_PIPELINE
        jsonp_assign_validation(jtree, plctx->validation);
        jsonp_assign_pipeline_name(pipelines[index_pipeline], plctx->name);
        jsonp_assign_peer_eth(pipelines[index_pipeline], plctx->dpdkctx.peer_eth_addr);
        jsonp_assign_vlan(pipelines[index_pipeline], plctx->dpdkctx.vlan);
        jsonp_assign_port(pipelines[index_pipeline], plctx->dpdkctx.port_id);

        for(int i=0; i<PT_MAX_FLOWS_X_PIPELINE; i++)
            plctx->flow_list[i] = 0xFFFFFFFF;
        jsonp_assign_flow_list(pipelines[index_pipeline], plctx->flow_list);
        plctx->dpdkctx.rxq = 0;
        for(auto tmp : plctx->flow_list)
        {
            if(tmp == 0xFFFFFFFF)
                break;
            plctx->flow_tot++;
        }
        plctx->dpdkctx.rxq = plctx->flow_tot;
        jsonp_flow_ident_method(pipelines[index_pipeline], plctx->dpdkctx.flow_ident_method);

        if(plctx->dpdkctx.rxq == 0 || plctx->dpdkctx.rxq > PT_MAX_FLOWS_X_PIPELINE)
        {
            pt_err("Erroneous RX queues number %d\n", plctx->dpdkctx.rxq);
            return PT_ERR;
        }

        //DPDK
        jsonp_assign_dpdk_burst(pipelines[index_pipeline], plctx->dpdkctx.mbuf_x_burst);
        jsonp_assign_dpdk_mbufs(pipelines[index_pipeline], plctx->dpdkctx.memp_mbuf_num);
        jsonp_assign_dpdk_payload_rx(pipelines[index_pipeline], plctx->dpdkctx.mbuf_payload_size_rx);
        jsonp_assign_dpdk_payload_tx(pipelines[index_pipeline], plctx->dpdkctx.mbuf_payload_size_tx);
        jsonp_assign_dpdk_cache(pipelines[index_pipeline], plctx->dpdkctx.memp_cache);
        jsonp_assign_dpdk_memory(pipelines[index_pipeline], plctx->dpdkctx.memp_type);
        // jsonp_assign_dpdk_rxd(pipelines[index_pipeline], plctx->dpdkctx.rxd);

        jsonp_assign_hds(pipelines[index_pipeline], plctx->dpdkctx.hds);

        jsonp_assign_descrambling(pipelines[index_pipeline], plctx->lwphy_descrambling);
        jsonp_assign_timers(pipelines[index_pipeline], plctx->measure_time);
        jsonp_assign_batching(pipelines[index_pipeline], plctx->rx_batching_us);
        jsonp_assign_uplink(pipelines[index_pipeline], plctx->uplink);
        jsonp_assign_downlink(pipelines[index_pipeline], plctx->downlink);
        jsonp_assign_tti(pipelines[index_pipeline], plctx->tti);
        jsonp_assign_slots_dl(pipelines[index_pipeline], plctx->downlink_slots);
        jsonp_assign_slots_ul(pipelines[index_pipeline], plctx->uplink_slots);
        jsonp_assign_controller(pipelines[index_pipeline], ptctx->controller);
        jsonp_assign_controller_file(pipelines[index_pipeline], ptctx->controller_file);
        jsonp_assign_dump_pusch_input(pipelines[index_pipeline], plctx->dump_pusch_input);
        jsonp_assign_dump_pusch_output(pipelines[index_pipeline], plctx->dump_pusch_output);
        jsonp_assign_dump_dl_output(pipelines[index_pipeline], plctx->dump_dl_output);
        jsonp_first_ap_only(pipelines[index_pipeline], plctx->first_ap_only);
        jsonp_assign_wait_downlink_slot(pipelines[index_pipeline], plctx->wait_downlink_slot);
        jsonp_assign_sync_tx_tti(pipelines[index_pipeline], plctx->sync_tx_tti);
        jsonp_assign_tv_slot_3gpp(pipelines[index_pipeline], plctx->tv_slot_3gpp);
        jsonp_assign_ul_cplane_delay(pipelines[index_pipeline], plctx->ul_cplane_delay);

        //Test vector files
        if(plctx->uplink && pt_parse_tv_uplink(plctx, pipelines[index_pipeline]) != PT_OK)
        {
            pt_err("Error parsing UL test vector files\n");
            return PT_ERR;
        }

        //Test vector files
        if(plctx->downlink && pt_parse_tv_downlink(plctx, pipelines[index_pipeline]) != PT_OK)
        {
            pt_err("Error parsing DL test vector files\n");
            return PT_ERR;
        }

        //Test vector files
    	// Config TV only used for DL at the moment
        if(plctx->downlink && pt_parse_tv_config(plctx, pipelines[index_pipeline]) != PT_OK)
        {
            pt_err("Error parsing config test vector files\n");
            return PT_ERR;
        }

        jsonp_assign_3gpp_slot_max(pipelines[index_pipeline], plctx->slot_num_max_3gpp);
        jsonp_assign_pbch_slots(pipelines[index_pipeline], plctx->pbch_slot_list);
        jsonp_assign_pusch_slots(pipelines[index_pipeline], plctx->pusch_slot_list);
        jsonp_assign_pdsch_slots(pipelines[index_pipeline], plctx->pdsch_slot_list);
        jsonp_assign_pdcch_ul_slots(pipelines[index_pipeline], plctx->pdcch_ul_slot_list);
        jsonp_assign_pdcch_dl_slots(pipelines[index_pipeline], plctx->pdcch_dl_slot_list);

        //LWCA
        jsonp_assign_gpuid(pipelines[index_pipeline], plctx->lwdactx.gpu_id);
        jsonp_assign_gpu_oblocks(pipelines[index_pipeline], plctx->lwdactx.order_kernel_blocks);
        jsonp_assign_gpu_othreads(pipelines[index_pipeline], plctx->lwdactx.order_kernel_threads);

        //Previous pipelines may have a different number of RX/TX queues
        plctx->dpdkctx.start_rxq = ptctx->dpdk_dev[plctx->dpdkctx.port_id].tot_rxq;
        ptctx->dpdk_dev[plctx->dpdkctx.port_id].tot_rxq += plctx->dpdkctx.rxq;
        
        //First queue: C-plane
        if(plctx->downlink)
        {
            plctx->dpdkctx.dl_txq = ptctx->dpdk_dev[plctx->dpdkctx.port_id].tot_txq;
            ptctx->dpdk_dev[plctx->dpdkctx.port_id].tot_txq++;
        }
        plctx->dpdkctx.c_txq = ptctx->dpdk_dev[plctx->dpdkctx.port_id].tot_txq;
        ptctx->dpdk_dev[plctx->dpdkctx.port_id].tot_txq++;

        ptctx->dpdk_dev[plctx->dpdkctx.port_id].tot_rxd += plctx->dpdkctx.rxd;
        ptctx->dpdk_dev[plctx->dpdkctx.port_id].tot_txd += ptctx->dpdk_dev[plctx->dpdkctx.port_id].tot_txq*DPDK_TX_DESC_DEFAULT;
        ptctx->dpdk_dev[plctx->dpdkctx.port_id].port_conf.rxmode.max_rx_pkt_len = plctx->dpdkctx.mbuf_payload_size_rx;
        ptctx->dpdk_dev[plctx->dpdkctx.port_id].enabled = 1;
        
        plctx->totCores = 2; //master + c-plane
        if(plctx->uplink) plctx->totCores += 4; //rx, prepare, timer, endpoint

        ptctx->totCores += plctx->totCores;
    
    CLOSE_FOREACH_PIPELINE
    
    if(pt_validate_optargs(ptctx) != PT_OK)
        return PT_ERR;

    return PT_OK;
}

int pt_parse_optargs(struct phytools_ctx *ptctx, int argc, char **argv, char *prgrname) 
{
    int opt, ret=0, tmp, index_core, option_index=0, totCores=0, index_pipeline=0;
    char *prgname = argv[0];

    if(!ptctx || !argv)
        return PT_EILWAL;
    
    if(ptctx->init == 0)
    {
        pt_err("You need to initialize the PT context before!\n");
        return PT_EILWAL;
    }

    while ((opt = getopt_long(argc, argv, short_options, long_options, &option_index)) != EOF) {
        switch (opt) {
            case SHOPT_help:
                pt_usage(prgname);
                return PT_STOP;

            case SHOPT_json:
                if(pt_parse_config_json_file(ptctx, optarg) != PT_OK) {
                    pt_err("invalid input file\n");
                    return PT_ERR;
                }
                break;

            default:
                pt_err("Invalid option %c\n", opt);
                pt_usage(prgname);
                return PT_EILWAL;
        }
    }

    return PT_OK;
}

////////////////////////////////////////////////////////////

