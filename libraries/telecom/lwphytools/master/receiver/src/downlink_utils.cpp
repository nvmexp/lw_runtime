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

int pt_setup_dl_slot_table(struct phytools_ctx * ptctx)
{
    int index=0, ret=PT_OK, size_o=0, size_i=0;
    void *ptr_o_h, *ptr_i_h;
    char tmp_hdr[ORAN_IQ_HDR_SZ];
    struct pt_slot_table * slot_table_ptr;
    std::string tv_filename = "";
    uint16_t vlan;
    DECLARE_FOREACH_PIPELINE

    if(!ptctx)
        return PT_EILWAL;

    OPEN_FOREACH_PIPELINE

        LW_LWDA_CHECK(lwdaSetDevice(plctx->lwdactx.gpu_id));
        lwdaFree(0);

        plctx->pdsch_slot_table_entry = (struct pt_slot_table *) rte_zmalloc(NULL, sizeof(struct pt_slot_table)*PT_MAX_SLOT_ID, 0);
        if(plctx->pdsch_slot_table_entry == NULL)
            return PT_ERR;
        plctx->pdcch_dl_table_entry = (struct pt_slot_table *) rte_zmalloc(NULL, sizeof(struct pt_slot_table)*PT_MAX_SLOT_ID, 0);
        if(plctx->pdcch_dl_table_entry == NULL)
            return PT_ERR;
        plctx->pdcch_ul_table_entry = (struct pt_slot_table *) rte_zmalloc(NULL, sizeof(struct pt_slot_table)*PT_MAX_SLOT_ID, 0);
        if(plctx->pdcch_ul_table_entry == NULL)
            return PT_ERR;
        plctx->pbch_table_entry = (struct pt_slot_table *) rte_zmalloc(NULL, sizeof(struct pt_slot_table)*PT_MAX_SLOT_ID, 0);
        if(plctx->pbch_table_entry == NULL)
            return PT_ERR;

        pt_info("Preparing %d PDSCH RX pipelines on GPU %d pipeline %d\n", PT_MAX_SLOT_ID, plctx->lwdactx.gpu_id, index_pipeline);
        for(index = 0; index < PT_MAX_SLOT_ID; index++)
        {
            if(ptctx->controller == CONTROLLER_DPDK) {
                tv_filename = plctx->pdsch_tv_list[index%plctx->tot_pdsch_tv].file_name;
            } else {
                tv_filename = "";
            }

            ret = lwphy_pdsch_prepare(plctx->lwdactx.gpu_id,
                                            &(plctx->pdsch_slot_table_entry[index]),
                                            tv_filename,
                                            plctx->stream_dl,
                                            plctx->dump_dl_output == 1 ? 0 : 1 //HOST / GPU memory
                                            );
            if(ret != PT_OK)
                return ret;
            ret = lwphy_pdcch_dl_301_prepare(plctx->lwdactx.gpu_id,
                                            &(plctx->pdcch_dl_table_entry[index]),
                                            plctx->config_tv_list[index%plctx->tot_config_tv].file_name,
                                            plctx->stream_dl,
                                            plctx->pdsch_slot_table_entry[index].pdsch_dl_phy->get_obuf_size(),
                                            plctx->pdsch_slot_table_entry[index].pdsch_dl_phy->get_obuf_addr<uint8_t*>()
                                            ); //Assuming no-overlap
            if(ret != PT_OK)
                return ret;
            ret = lwphy_pdcch_dl_301a_prepare(plctx->lwdactx.gpu_id,
                                            &(plctx->pdcch_dl_table_entry[index]),
                                            plctx->config_tv_list[index%plctx->tot_config_tv].file_name,
                                            plctx->stream_dl,
                                            plctx->pdsch_slot_table_entry[index].pdsch_dl_phy->get_obuf_size(),
                                            plctx->pdsch_slot_table_entry[index].pdsch_dl_phy->get_obuf_addr<uint8_t*>()
                                            ); //Assuming no-overlap
            if(ret != PT_OK)
                return ret;
            ret = lwphy_pdcch_dl_301b_prepare(plctx->lwdactx.gpu_id,
                                            &(plctx->pdcch_dl_table_entry[index]),
                                            plctx->config_tv_list[index%plctx->tot_config_tv].file_name,
                                            plctx->stream_dl,
                                            plctx->pdsch_slot_table_entry[index].pdsch_dl_phy->get_obuf_size(),
                                            plctx->pdsch_slot_table_entry[index].pdsch_dl_phy->get_obuf_addr<uint8_t*>()
                                            ); //Assuming no-overlap
            if(ret != PT_OK)
                return ret;

            ret = lwphy_pdcch_ul_prepare(plctx->lwdactx.gpu_id,
                                            &(plctx->pdcch_ul_table_entry[index]),
                                            plctx->config_tv_list[index%plctx->tot_config_tv].file_name,
                                            plctx->stream_dl,
                                            plctx->pdsch_slot_table_entry[index].pdsch_dl_phy->get_obuf_size(),
                                            plctx->pdsch_slot_table_entry[index].pdsch_dl_phy->get_obuf_addr<uint8_t*>()
                                            ); //Assuming no-overlap
            if(ret != PT_OK)
                return ret;

            ret = lwphy_pbch_prepare(plctx->lwdactx.gpu_id,
                                            &(plctx->pbch_table_entry[index]),
                                            plctx->config_tv_list[index%plctx->tot_config_tv].file_name,
                                            plctx->stream_dl,
                                            plctx->pdsch_slot_table_entry[index].pdsch_dl_phy->get_obuf_size(),
                                            plctx->pdsch_slot_table_entry[index].pdsch_dl_phy->get_obuf_addr<uint8_t*>()
                                            ); //Assuming no-overlap
            if(ret != PT_OK)
                return ret;

            // pt_info("Registering Slot flow count %d, 0x%p: %zd with device %s for DMA\n",
            //         plctx->flow_tot,
            //         (uint8_t*)(plctx->pdsch_slot_table_entry[index].pdsch_dl_phy->get_obuf_addr<uint8_t*>()),
            //         (plctx->pdsch_slot_table_entry[index].pdsch_dl_phy->get_obuf_size()),
            //         rte_eth_devices[plctx->dpdkctx.port_id].device->name
            //     );

            ret = lw_register_ext_mem((void*)(plctx->pdsch_slot_table_entry[index].pdsch_dl_phy->get_obuf_addr<uint8_t*>()),
                (plctx->pdsch_slot_table_entry[index].pdsch_dl_phy->get_obuf_size()),
                // LW_GPU_PAGE_SIZE,
                1,
                &(rte_eth_devices[plctx->dpdkctx.port_id]));

            if (ret) {
                pt_err("lw_register_ext_mem error, rte_errno: %d\n", rte_errno);
                return ret;
            }

            slot_table_ptr = &(plctx->pdsch_slot_table_entry[index]);

            slot_table_ptr->index         = index;
            slot_table_ptr->num_packets   = 0;
            slot_table_ptr->flow_tot = plctx->flow_tot;
            slot_table_ptr->slot_dims = buffer_to_slot(
                                                    slot_table_ptr->pdsch_dl_phy->get_obuf_addr<uint8_t*>(),
                                                    slot_table_ptr->pdsch_dl_phy->get_otensor_size(),
                                                    slot_table_ptr->flow_tot,
                                                    PRB_SIZE_16F
                                                );
            slot_table_ptr->slot_dims.pkts_per_slot = 0;

            for (uint8_t sym_idx = 0; sym_idx < SLOT_NUM_SYMS; ++sym_idx) {
                slot_table_ptr->slot_dims.pkts_per_sym[sym_idx] = 0;

                for (uint8_t flow_idx = 0; flow_idx <  plctx->flow_tot; ++flow_idx) {
                    slot_table_ptr->slot_dims.prbs_per_pkt[flow_idx][sym_idx] = (plctx->dpdkctx.mbuf_payload_size_tx / PRB_SIZE_16F);
                    slot_table_ptr->slot_dims.pkts_per_flow[flow_idx][sym_idx] = (slot_table_ptr->slot_dims.prbs_per_flow[flow_idx][sym_idx] + slot_table_ptr->slot_dims.prbs_per_pkt[flow_idx][sym_idx] - 1) / slot_table_ptr->slot_dims.prbs_per_pkt[flow_idx][sym_idx];
                    slot_table_ptr->slot_dims.pkts_per_sym[sym_idx] += slot_table_ptr->slot_dims.pkts_per_flow[flow_idx][sym_idx];
                }

                slot_table_ptr->slot_dims.pkts_per_slot += slot_table_ptr->slot_dims.pkts_per_sym[sym_idx];
            }
        }

        pt_info("Slot parameters: Tot data %zd pkts_per_flow %d pkts_per_sym %d pkts_per_slot %d prbs_per_sym %d prbs_per_pkt %d\n",
                    ptctx->plctx[0].pdsch_slot_table_entry[0].slot_dims.data_sz,
                    ptctx->plctx[0].pdsch_slot_table_entry[0].slot_dims.pkts_per_flow[0][0],
                    ptctx->plctx[0].pdsch_slot_table_entry[0].slot_dims.pkts_per_sym[0],
                    ptctx->plctx[0].pdsch_slot_table_entry[0].slot_dims.pkts_per_slot,
                    ptctx->plctx[0].pdsch_slot_table_entry[0].slot_dims.prbs_per_sym[0],
                    ptctx->plctx[0].pdsch_slot_table_entry[0].slot_dims.prbs_per_pkt[0][0]
                );

        LW_LWDA_CHECK(lwdaDeviceSynchronize());

        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Prepare U-plane message headers
        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        memset(tmp_hdr, 0, ORAN_IQ_HDR_SZ);

        vlan = plctx->dpdkctx.vlan;
        if (plctx->dpdkctx.flow_ident_method == PT_FLOW_IDENT_METHOD_VLAN)
        {
            vlan = plctx->dpdkctx.vlan + 1;
        }

        if(oran_fill_eth_vlan_hdr((struct oran_eth_hdr *)(&(tmp_hdr[0])),
                                    ptctx->dpdk_dev[plctx->dpdkctx.port_id].eth_addr,
                                    plctx->dpdkctx.peer_eth_addr, vlan)
            )
            do_throw("oran_fill_eth_vlan_hdr error");

        if(oran_fill_ecpri_hdr((struct oran_ecpri_hdr *)(&(tmp_hdr[ORAN_ETH_HDR_SIZE])), 0, 0, 0, ECPRI_MSG_TYPE_IQ))
            do_throw("oran_fill_ecpri_hdr error");

        if(oran_fill_umsg_iq_hdr((struct oran_umsg_iq_hdr *)(&(tmp_hdr[ORAN_IQ_HDR_OFFSET])), DIRECTION_DOWNLINK, 0, 0, 0, 0))
            do_throw("oran_fill_umsg_iq_hdr error");

        plctx->oran_uplane_hdr.assign((const char *)tmp_hdr, ORAN_IQ_HDR_SZ);

        plctx->dl_tx_list = (struct dl_tx_info *) rte_zmalloc(NULL, sizeof(struct dl_tx_info)*PT_MAX_SLOT_ID, 0);
        if(plctx->dl_tx_list == NULL)
            return PT_ERR;

    CLOSE_FOREACH_PIPELINE

    return ret;
}

int pt_setup_dl_rings(struct phytools_ctx * ptctx)
{
    int ring_flags=0, i=0;
    char ring_name[1024];
    DECLARE_FOREACH_PIPELINE

    if(!ptctx)
        return PT_EILWAL;

    OPEN_FOREACH_PIPELINE

    snprintf(ring_name, sizeof(ring_name), "txdl-%04d", index_pipeline);
    plctx->ring_tx_dl = rte_ring_create(ring_name, PT_RING_ELEMS, plctx->dpdkctx.socket_id, 0);
    if (plctx->ring_tx_dl == NULL)
    {
        pt_err("rte_zmalloc %d error\n", index_pipeline);
        return PT_ERR;
    }

    CLOSE_FOREACH_PIPELINE

    return PT_OK;
}

int pt_setup_dl_gpu_work(struct phytools_ctx * ptctx)
{
    size_t useless_alloc_size;
    uintptr_t useless_phy;
    int index=0;
    DECLARE_FOREACH_PIPELINE

    OPEN_FOREACH_PIPELINE

        LW_LWDA_CHECK(lwdaSetDevice(plctx->lwdactx.gpu_id));
        lwdaFree(0);

        LW_LWDA_CHECK(lwdaStreamCreateWithFlags(&(plctx->stream_dl), lwdaStreamNonBlocking));

        LW_LWDA_CHECK(lwdaMallocHost((void**)&(plctx->pdsch_phy_done_h), PT_MAX_SLOT_ID * sizeof(uint32_t)));
        memset(plctx->pdsch_phy_done_h, PT_SLOT_FREE, PT_MAX_SLOT_ID * sizeof(uint32_t));

        LW_LWDA_CHECK(lwdaMallocHost((void**)&(plctx->pdcch_ul_phy_done_h), PT_MAX_SLOT_ID * sizeof(uint32_t)));
        memset(plctx->pdcch_ul_phy_done_h, PT_SLOT_FREE, PT_MAX_SLOT_ID * sizeof(uint32_t));

        LW_LWDA_CHECK(lwdaMallocHost((void**)&(plctx->pdcch_dl_301_phy_done_h), PT_MAX_SLOT_ID * sizeof(uint32_t)));
        memset(plctx->pdcch_dl_301_phy_done_h, PT_SLOT_FREE, PT_MAX_SLOT_ID * sizeof(uint32_t));

        LW_LWDA_CHECK(lwdaMallocHost((void**)&(plctx->pdcch_dl_301a_phy_done_h), PT_MAX_SLOT_ID * sizeof(uint32_t)));
        memset(plctx->pdcch_dl_301a_phy_done_h, PT_SLOT_FREE, PT_MAX_SLOT_ID * sizeof(uint32_t));

        LW_LWDA_CHECK(lwdaMallocHost((void**)&(plctx->pdcch_dl_301b_phy_done_h), PT_MAX_SLOT_ID * sizeof(uint32_t)));
        memset(plctx->pdcch_dl_301b_phy_done_h, PT_SLOT_FREE, PT_MAX_SLOT_ID * sizeof(uint32_t));

        LW_LWDA_CHECK(lwdaMallocHost((void**)&(plctx->pbch_phy_done_h), PT_MAX_SLOT_ID * sizeof(uint32_t)));
        memset(plctx->pbch_phy_done_h, PT_SLOT_FREE, PT_MAX_SLOT_ID * sizeof(uint32_t));

        plctx->enabled = 1;

    CLOSE_FOREACH_PIPELINE

    return PT_OK;
}

int pt_dl_finalize(struct phytools_ctx * ptctx)
{
    int ret = PT_OK, index=0;
    lwdaError_t result=lwdaSuccess;
    DECLARE_FOREACH_PIPELINE

    if(!ptctx)
        return PT_EILWAL;

    OPEN_FOREACH_PIPELINE

        rte_free(plctx->pdsch_slot_table_entry);
        rte_free(plctx->pdcch_dl_table_entry);
        rte_free(plctx->pdcch_ul_table_entry);
        rte_free(plctx->pbch_table_entry);
        rte_free(plctx->dl_tx_list);
        LW_LWDA_CHECK(lwdaStreamDestroy(plctx->stream_dl));
        lwdaFreeHost(plctx->pdsch_phy_done_h);
        lwdaFreeHost(plctx->pdcch_ul_phy_done_h);
        lwdaFreeHost(plctx->pdcch_dl_301_phy_done_h);
        lwdaFreeHost(plctx->pdcch_dl_301a_phy_done_h);
        lwdaFreeHost(plctx->pdcch_dl_301b_phy_done_h);
        lwdaFreeHost(plctx->pbch_phy_done_h);
        rte_ring_free(plctx->ring_tx_dl);
    CLOSE_FOREACH_PIPELINE

    return PT_OK;
}
