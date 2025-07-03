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

//5 payloads just for lwvp profiling
#ifdef PROFILE_LWTX_RANGES
    static int uplink_endpoint_core_count = 0;
#endif

static int start_pusch_pipeline(int index_slot, struct phytools_ctx * ptctx, struct pipeline_ctx * plctx, lwdaStream_t stream, uint8_t prev_frameId, uint8_t prev_subFrameId, uint8_t prev_slotId)
{
    int index=0;
    uint64_t pip_start, order_start;
    struct pt_slot_table * slot_table_ptr;

    if(!plctx)
        return PT_EILWAL;

    slot_table_ptr = &(plctx->pusch_slot_table_entry[index_slot]);

    if(plctx->measure_time >= PT_TIMER_PIPELINE)
        slot_table_ptr->t_start_enqueue = get_ns();

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Order kernels
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // order_start = get_ns();
    if(plctx->dpdkctx.hds)
        pt_launch_pk_copy(
                plctx->mbatch, ((uint32_t*)plctx->mbufs_batch_ready_flags_d), 
                ptctx->tot_pkts_x_batch, 
                (plctx->measure_time >= PT_TIMER_PIPELINE ? 1 : 0),
                &(plctx->mbufs_slot_start_flags_h[index_slot]),
                &(plctx->mbufs_slot_order_flags_h[index_slot]),
                plctx->lwdactx.order_kernel_blocks, plctx->lwdactx.order_kernel_threads, stream
            );
    else
        pt_launch_pk_order(
                plctx->cache_count_prbs, 
                (plctx->first_ap_only == 1 ? ((273*14*1)+(1*14*3)) : slot_table_ptr->slot_dims.prbs_per_slot), 
                slot_table_ptr->slot_dims.prbs_per_symbol,
                plctx->mbatch, ((uint32_t*)plctx->mbufs_batch_ready_flags_d), 
                // ((uint16_t*)plctx->slot_status_d), 
                ((uintptr_t*)plctx->gbuf_table_cache_ptr),
                ptctx->tot_pkts_x_batch, index_slot,
                (plctx->measure_time >= PT_TIMER_PIPELINE ? 1 : 0),
                &(plctx->mbufs_slot_start_flags_h[index_slot]),
                &(plctx->mbufs_slot_order_flags_h[index_slot]),
                &(plctx->map_slot_to_last_mbatch[index_slot]),
                prev_frameId, prev_subFrameId, prev_slotId,
                plctx->lwdactx.order_kernel_blocks, plctx->lwdactx.order_kernel_threads, stream
            );

    // pt_launch_kernel_write(&(plctx->mbufs_slot_order_flags_h[index_slot]), PT_SLOT_ORDERED, stream);

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // lwPHY pipeline & validations
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    lwphy_pusch_run(slot_table_ptr->pusch_ul_phy);
    if(plctx->validation & PT_VALIDATION_CHECKSUM)
    {
        pt_launch_checksum(slot_table_ptr->pusch_ul_phy->get_ibuf_addr<uint8_t*>(),
                                slot_table_ptr->pusch_ul_phy->get_ibuf_size()/plctx->flow_tot, //only first antenna
                                plctx->ul_checksum_runtime, stream);
    }

    if(plctx->validation & PT_VALIDATION_CRC)
    {
        pt_launch_check_crc(slot_table_ptr->pusch_ul_phy->get_tbcrc_addr(),
                                slot_table_ptr->pusch_ul_phy->get_tbcrc_size(),
                                slot_table_ptr->pusch_ul_phy->crc_errors, stream);
    }
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    /*
     * Here we prefer to use the kernel because StreamMemOp aren't always allowed/enabled on the system.
     * Also, with the proper fence, we ensure the visibility of the write on the host
     */
    pt_launch_kernel_write(&(plctx->mbufs_slot_done_flags_h[index_slot]), PT_SLOT_DONE, stream);
    // LW_LW_CHECK(lwStreamWriteValue32(plctx->stream_pipeline, (LWdeviceptr) &(plctx->mbufs_slot_done_flags_h[index_slot]), PT_SLOT_DONE, 0));

    if(plctx->measure_time >= PT_TIMER_PIPELINE)
        slot_table_ptr->t_stop_enqueue = get_ns();

    return PT_OK;
err:
    return PT_ERR;
}

int uplink_timer_core(void *param) {
    SET_THREAD_NAME(__FUNCTION__);
    int iteration=0;
    struct pt_slot_table * slot_table_entry;
    struct mbufs_batch * mbatch;
    struct core_info * cinfo = (struct core_info *) param;
    struct phytools_ctx * ptctx = cinfo->ptctx;
    struct pipeline_ctx * plctx = cinfo->plctx;
    int pipeline_index          = plctx->index;
    char jname[512];

    timerdiff_us(0,0); // initialize timer_hz inside

    pt_info("Pipeline %d Timer core on lcore %u Sched. priority %d\n", pipeline_index, rte_lcore_index(rte_lcore_id()), pt_get_thread_priority());

    while(check_force_quit(force_quit) == false)
    {
        slot_table_entry = NULL;

        //Order Kernel for the next TTI has been launched
        while(check_force_quit(force_quit) == false && rte_ring_dequeue(plctx->ring_timer_slot, (void **)&slot_table_entry) != 0);
        CHECK_FORCE_QUIT_STRING(force_quit, "uplink_timer_core stopped during rte_ring_dequeue\n")
        // pt_info("Polling on slot %d\n", slot_table_entry->index);

        iteration=0;

        while(check_force_quit(force_quit) == false)
        {
            mbatch = NULL;
            // New batch is ready
            if(rte_ring_dequeue(plctx->ring_timer_batch, (void **)&mbatch) == 0 && mbatch != NULL)
            {
                if(plctx->dpdkctx.hds && plctx->mbatch[mbatch->index_mbatch].last_mbatch == 1)
                {
                    set_mbatch_status(plctx, mbatch->index_mbatch, 1, PT_MBATCH_LAST);
                    plctx->mbatch[mbatch->index_mbatch].last_mbatch = 0;
                }
                else
                    set_mbatch_status(plctx, mbatch->index_mbatch, 1, PT_MBATCH_READY);
            }

            if(iteration == 0)
            {
                if(PT_ACCESS_ONCE(plctx->mbufs_slot_start_flags_h[slot_table_entry->index]) >= PT_SLOT_START)
                {
                    slot_table_entry->t_start_order = get_ns();
                    iteration=1;
                }
            }

            if(PT_ACCESS_ONCE(plctx->mbufs_slot_order_flags_h[slot_table_entry->index]) >= PT_SLOT_ORDERED)
            {
                slot_table_entry->t_start_ch = get_ns();
                // pt_info("ORDER KERNEL SLOT %d ENDED after %6.2f us. ORDER TS: %lu START CH TS: %lu. Waiting for lwPHY\n", slot_table_entry->index, get_us_from_ns(get_ns() -  slot_table_entry->t_start_order), slot_table_entry->t_start_order, slot_table_entry->t_start_ch);                
                
                if(plctx->dump_pusch_input)
                {
                    // LW_LWDA_CHECK(lwdaStreamSynchronize(stream));
                    slot_table_entry->pusch_ul_phy->CopyInputToCPU();
                    uint8_t * tmp = slot_table_entry->pusch_ul_phy->get_ibuf_addr_h<uint8_t*>();

                    snprintf(jname, sizeof(jname), "pusch_input_%d_frame%d_subframe%d_slot%d.json", 
                                                    slot_table_entry->index, slot_table_entry->l2_slot_info->frameId, slot_table_entry->l2_slot_info->subFrameId, slot_table_entry->l2_slot_info->slotId);
                    std::string json_file(jname);
                    if(pt_dump_slot_buffer(json_file, 
                                            slot_table_entry->pusch_ul_phy->get_ibuf_addr_h<uint8_t*>(), 
                                            slot_table_entry->pusch_ul_phy->get_ibuf_size(), 
                                            PRB_SIZE_16F, 
                                            plctx) != PT_OK)
                    {
                        pt_err("pt_dump_slot_buffer error\n");
                        goto err;
                    }
                }

                //Move the slot to the endpoint core
                while(check_force_quit(force_quit) == false && rte_ring_enqueue(plctx->ring_tx_ul, (void*)&(plctx->pusch_slot_table_entry[slot_table_entry->index])) != 0);
                CHECK_FORCE_QUIT_STRING(force_quit, "uplink_timer_core stopped during rte_ring_dequeue\n")
                break;
            }
        }
    }

err:
    pt_warn("EXIT\n");
    return PT_OK;
}

int uplink_c_core(void *param) {
    SET_THREAD_NAME(__FUNCTION__);
    int nb_tx=0, index_slot=0, index_stream=0, slot_num=0, startPrbc=0, startSym=0, cnt_slots=0, numSym=ORAN_ALL_SYMBOLS, numPrbc=ORAN_RB_ALL;
    //Timers
    uint64_t waitu_start_t, waitu_end_t, slot_tick_ts, last_ul_t=0;
    //Overallocation    
    rte_unique_mbufs mbufs_c{nullptr, &rte_free};
    //Mbuf payload addresses
    uint8_t * payload;
    uint8_t frameId=0, subFrameId=0, slotId=0, slotId_3GPP=0, ecpriSeqid_c=0;
    uint8_t prev_frameId=0, prev_subFrameId=0, prev_slotId=0;
    struct l2_control_slot_info * l2_slot_info = NULL;
    char oran_hdr_template[ORAN_IQ_HDR_SZ];

    struct core_info * cinfo = (struct core_info *) param;
    int txq                     = cinfo->txq;
    int port_id                 = cinfo->port_id;
    struct phytools_ctx * ptctx = cinfo->ptctx;
    struct pipeline_ctx * plctx = cinfo->plctx;
    int pipeline_index          = plctx->index;
    
    timerdiff_us(0,0); // initialize timer_hz inside

    LW_LWDA_CHECK(lwdaSetDevice(plctx->lwdactx.gpu_id));

    mbufs_c.reset(dpdk_allocate_mbufs(plctx->flow_tot * PT_PKT_X_CMSG));
    if(mbufs_c.get() == NULL)
    {
        pt_err("mbufs_c allocation error\n");
        return PT_ERR;
    }

    pt_info("Pipeline %d TX core %d on lcore %u port id %d socket %d Sched. priority %d\n", 
            pipeline_index, txq, rte_lcore_index(rte_lcore_id()), port_id, rte_socket_id(), pt_get_thread_priority());

    while(check_force_quit(force_quit) == false)
    {
        startPrbc=0; startSym=0; numSym=ORAN_ALL_SYMBOLS; numPrbc = ORAN_RB_ALL;

        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Wait for next slot available in memory
        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        while(check_force_quit(force_quit) == false && rte_ring_dequeue(ptctx->ring_start_ul, (void **)&l2_slot_info) != 0);
        CHECK_FORCE_QUIT_STRING(force_quit, "stopped during rte_ring_dequeue\n")
        
        if(rte_ring_count(ptctx->ring_free_ul) <= PT_RING_ELEMS-3)
        {
            pt_info("UL pipeline %d | 3GPP slot %d | Delta from slot tick %6.2fus | still processing another slot. "
                    "Dropping new fh cmd 3GPP slot %d frame %u subframe %u slot %u tick %" PRIu64 ". Elems in the free ring %d\n",
                cnt_slots, slotId_3GPP,
                get_us_from_ns(get_ns() - slot_tick_ts),
                l2_slot_info->frameId, l2_slot_info->frameId, l2_slot_info->subFrameId, l2_slot_info->slotId, l2_slot_info->tick,
                rte_ring_count(ptctx->ring_free_ul)
            );
            rte_ring_enqueue(ptctx->ring_free_ul, (void*)l2_slot_info);
            continue;
        }
        if(last_ul_t > 0)
        {
            pt_info("UL pipeline %d | 3GPP slot %d | Delta from slot tick %6.2f us (TS=%lu) | Delta from last UL %6.2f\n",
                cnt_slots, l2_slot_info->slotId_3GPP, get_us_from_ns(get_ns() - l2_slot_info->tick), l2_slot_info->tick, get_us_from_ns(get_ns() - last_ul_t)
            );
        }
        last_ul_t = get_ns();

        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        //// Double check no previous UL pipeline is still running before starting GPU work and C-plane
        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        uint64_t wait_order_t = l2_slot_info->tick + ((plctx->tti * NS_X_US) * (plctx->sync_tx_tti - 1));
        while (get_ns() < wait_order_t)
        {
            for (int spin_cnt = 0; spin_cnt < 1000; ++spin_cnt)
                __asm__ __volatile__ ("");
        }
        // If there is a UL pipeline still running after 3 slots, that means it is late, drop the current pipeline
        if(rte_ring_count(ptctx->ring_free_ul) < PT_RING_ELEMS-1) {
            pt_info("UL pipeline %d | 3GPP slot %d | Delta from slot tick %6.2fus | still processing another slot after %d TTI. "
                    "Dropping the fh cmd 3GPP slot %d frame %u subframe %u slot %u tick %" PRIu64 ". Elems in the free ring %d\n",
                cnt_slots, slotId_3GPP,
                get_us_from_ns(get_ns() - slot_tick_ts),
                l2_slot_info->slotId_3GPP, l2_slot_info->frameId, l2_slot_info->frameId, l2_slot_info->subFrameId, l2_slot_info->slotId, l2_slot_info->tick,
                rte_ring_count(ptctx->ring_free_ul)
            );

            rte_ring_enqueue(ptctx->ring_free_ul, (void*)l2_slot_info);
            continue;
        }

        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        frameId = (uint8_t)l2_slot_info->frameId;
        subFrameId = (uint8_t)l2_slot_info->subFrameId;
        slotId = (uint8_t)l2_slot_info->slotId;
        slotId_3GPP = (uint8_t)l2_slot_info->slotId_3GPP;
        index_slot = ((frameId * ORAN_MAX_SUBFRAME_ID * ORAN_MAX_SLOT_ID) + (subFrameId * ORAN_MAX_SLOT_ID) + slotId)%PT_MAX_SLOT_ID;
        plctx->pusch_slot_table_entry[index_slot].l2_slot_info = l2_slot_info;
        startPrbc = l2_slot_info->all_startPrb[PHY_PUSCH];
        numPrbc = l2_slot_info->all_numPrb[PHY_PUSCH];
        startSym = l2_slot_info->all_startSym[PHY_PUSCH];
        numSym = l2_slot_info->all_numSym[PHY_PUSCH];
        slot_tick_ts = l2_slot_info->tick;

        if(numPrbc == ORAN_MAX_PRB_X_SLOT)
            numPrbc = ORAN_RB_ALL;
        else if(numPrbc > ORAN_MAX_PRB_X_SECTION)
        {
            pt_err("Asking for %d PRBs. Multi-section is not supported yet\n", numPrbc);
            set_force_quit(force_quit);
            goto err;
        }

        while(check_force_quit(force_quit) == false && (PT_ACCESS_ONCE(plctx->mbufs_slot_done_flags_h[index_slot]) != PT_SLOT_FREE));
        CHECK_FORCE_QUIT_STRING(force_quit, "stopped waiting for free slot\n")
        
        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Prepare C-plane message
        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        if(dpdk_pull_pkts(mbufs_c, plctx->dpdkctx.c_mempool, plctx->flow_tot * PT_PKT_X_CMSG) != PT_OK)
        {
            pt_err("Asking for %d packets from C-plane mempool error.\n", plctx->flow_tot * PT_PKT_X_CMSG);
            set_force_quit(force_quit);
            goto err;
        }

        if(pt_prepare_cplane_messages(mbufs_c, plctx->flow_tot * PT_PKT_X_CMSG, 
                                        (plctx->first_ap_only == 1 ? 1 : 0),
                                        ecpriSeqid_c, frameId, subFrameId, slotId, 
                                        startPrbc, numPrbc, numSym, startSym,
                                        DIRECTION_UPLINK,
                                        ptctx->dpdk_dev[plctx->dpdkctx.port_id].eth_addr,
                                        plctx
                            ) != PT_OK)
        {
            pt_err("pt_prepare_cplane_messages error\n");
            set_force_quit(force_quit);
            return PT_ERR;
        }

        slot_num = get_slot_number_from_packet(oran_cmsg_get_frame_id(((uint8_t *)rte_pktmbuf_mtod_offset((mbufs_c.get())[0], uint8_t*, 0))),
                                    oran_cmsg_get_subframe_id(((uint8_t *)rte_pktmbuf_mtod_offset((mbufs_c.get())[0], uint8_t*, 0))),
                                    oran_cmsg_get_slot_id(((uint8_t *)rte_pktmbuf_mtod_offset((mbufs_c.get())[0], uint8_t*, 0))));

        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Trigger pipeline on slot number
        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        uint64_t pusch_cpu_t = get_ns();
        if(ptctx->controller == CONTROLLER_LWPHY)
        {
            if(lwphy_configure_pusch_pipeline(plctx->pusch_slot_table_entry[index_slot].pusch_ul_phy, l2_slot_info->all_cell_params[PHY_PUSCH], l2_slot_info->all_block_params[PHY_PUSCH]) != PT_OK)
            {
                pt_err("start_pusch_pipeline error on slot %d\n", index_slot);
                return PT_ERR;
            }
        }
        else
        {
            if(plctx->tv_slot_3gpp >= 0) {
                if(lwphy_pusch_slot(plctx->pusch_slot_table_entry[index_slot].pusch_ul_phy, plctx->tv_slot_3gpp) != PT_OK)
                {
                    pt_err("start_pusch_pipeline error on slot %d\n", index_slot);
                    return PT_ERR;
                }
            }
            else 
            {
                if(lwphy_pusch_slot(plctx->pusch_slot_table_entry[index_slot].pusch_ul_phy, slotId_3GPP) != PT_OK)
                {
                    pt_err("start_pusch_pipeline error on slot %d\n", index_slot);
                    return PT_ERR;
                }
            }
        }

        // uint64_t start_cpu_t = get_ns();
        if(start_pusch_pipeline(index_slot, ptctx, plctx, 
                                plctx->pusch_slot_table_entry[index_slot].pusch_ul_phy->getStream(),
                                prev_frameId, prev_subFrameId, prev_slotId
                                ) != PT_OK)
        {
            pt_err("start_pusch_pipeline error on slot %d\n", index_slot);
            return PT_ERR;
        }

        pt_info("UL pipeline %d | 3GPP slot %d | Delta from slot tick %6.2fus | PUSCH CPU time %6.2fus\n",
                cnt_slots, l2_slot_info->slotId_3GPP,
                get_us_from_ns(get_ns() - slot_tick_ts),
                get_us_from_ns(get_ns() - pusch_cpu_t)
            );
        
        // while(check_force_quit(force_quit) == false && rte_ring_enqueue(plctx->ring_tx_ul, (void*)&(plctx->pusch_slot_table_entry[index_slot])) != 0);
        while(check_force_quit(force_quit) == false && rte_ring_enqueue(plctx->ring_timer_slot, (void*)&(plctx->pusch_slot_table_entry[index_slot])) != 0);
        CHECK_FORCE_QUIT_STRING(force_quit, "ring_timer_slot stopped during rte_ring_enqueue ring_timer_slot\n")

        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Send C-plane message
        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        uint64_t end2_t = slot_tick_ts + ((plctx->tti * NS_X_US) * (plctx->sync_tx_tti) + plctx->ul_cplane_delay * NS_X_US);
        while (get_ns() < end2_t)
        {
            for (int spin_cnt = 0; spin_cnt < 1000; ++spin_cnt)
                __asm__ __volatile__ ("");
        }

        pt_info("UL pipeline %d | 3GPP slot %d | Delta from slot tick %6.2fus | C-plane | ORAN HDR: frameId=%d, subFrameId=%d, slotId=%d startPrb=%d, numPrb=%d startSym=%d numSym=%d\n",
                cnt_slots, l2_slot_info->slotId_3GPP,
                get_us_from_ns(get_ns() - slot_tick_ts),
                frameId, subFrameId, slotId, startPrbc, numPrbc, startSym, numSym
            );

        PUSH_RANGE("TX_C_UL", LWLWTX_COL_3);
        int num_tx = dpdk_tx_burst_pkts(mbufs_c, port_id, plctx->dpdkctx.c_txq, (plctx->flow_tot * PT_PKT_X_CMSG), ptctx->flush_tx_write, force_quit);
        POP_RANGE;
        
        prev_frameId = frameId;
        prev_subFrameId = subFrameId;
        prev_slotId = slotId;
        ecpriSeqid_c = ((ecpriSeqid_c+1)%256);
        // index_slot = (index_slot+1)%PT_MAX_SLOT_ID;
        cnt_slots++;
    }

err:
    for(index_slot=0; index_slot < PT_MAX_SLOT_ID; index_slot++)
    {
        PT_ACCESS_ONCE(plctx->mbufs_slot_start_flags_h[index_slot]) = PT_SLOT_EXIT;
        PT_ACCESS_ONCE(plctx->mbufs_slot_order_flags_h[index_slot]) = PT_SLOT_EXIT;
        PT_ACCESS_ONCE(plctx->mbufs_slot_done_flags_h[index_slot]) = PT_SLOT_EXIT;
    }
    rte_wmb();
    flush_gmem(plctx);
    LW_LWDA_CHECK(lwdaDeviceSynchronize());

    pt_warn("EXIT\n");

    return PT_OK;
}

int uplink_endpoint_core(void *param) {
    SET_THREAD_NAME(__FUNCTION__);
    int i=0, nb_tx=0, last_slot=(PT_MAX_SLOT_ID-1), index=0, ret = PT_OK;
    int local_index_mbatch=0;
    struct pt_slot_table * slot_table_entry, * last_slot_table_entry;
    //Overallocation
    struct rte_mbuf * m;
    //Mbuf payload addresses
    uint64_t last_mbatch_end_time;
    double tot_rx_prep=0;
    lwdaStream_t copy_stream;

    struct core_info * cinfo = (struct core_info *) param;
    int txq                     = cinfo->txq;
    int port_id                 = cinfo->port_id;
    struct phytools_ctx * ptctx = cinfo->ptctx;
    struct pipeline_ctx * plctx = cinfo->plctx;
    int pipeline_index          = plctx->index;
    char jname[512];

    timerdiff_us(0,0); // initialize timer_hz inside

    LW_LWDA_CHECK(lwdaSetDevice(plctx->lwdactx.gpu_id));
    LW_LWDA_CHECK(lwdaStreamCreateWithFlags(&(copy_stream), lwdaStreamNonBlocking));

    pt_info("Pipeline %d endpoint lcore %u port id %d socket %d Sched. priority %d\n", 
        pipeline_index, rte_lcore_index(rte_lcore_id()), port_id, rte_socket_id(), pt_get_thread_priority());

    last_slot_table_entry = NULL;
    while(check_force_quit(force_quit) == false)
    {
        slot_table_entry = NULL;
        tot_rx_prep = -1;

        //////////////////////////////////////////////////////////////////////////////
        /// Pick up the next slot
        //////////////////////////////////////////////////////////////////////////////
        while(check_force_quit(force_quit) == false && rte_ring_dequeue(plctx->ring_tx_ul, (void **)&slot_table_entry) != 0);
        CHECK_FORCE_QUIT_STRING(force_quit, "stopped during rte_ring_dequeue\n")
        // pt_info("Polling on order slot %d\n", slot_table_entry->index);

        //////////////////////////////////////////////////////////////////////////////
        /// Wait for the end of the lwPHY pipeline
        //////////////////////////////////////////////////////////////////////////////
        // A bit ugly: CPU cores may not be aligned on the time. Anyway this isn't a relevant timer
        while(check_force_quit(force_quit) == false && (PT_ACCESS_ONCE(plctx->mbufs_slot_done_flags_h[slot_table_entry->index]) < PT_SLOT_DONE));
        CHECK_FORCE_QUIT_STRING(force_quit, "uplink_endpoint_core stopped during wait done\n")
        // pt_info("Slot %d completed\n", slot_table_entry->index);
        if(plctx->measure_time >= PT_TIMER_PIPELINE)
            slot_table_entry->t_slot_end = get_ns();
        //lwdaEventSynchronize(slot_table_entry->eventEndCh);

        last_slot = slot_table_entry->index;

        if(plctx->dump_pusch_output)
        {
            slot_table_entry->pusch_ul_phy->get_tensor()->copyOutputToCPU(slot_table_entry->pusch_ul_phy->getStream(), nullptr, nullptr, slot_table_entry->tb_output);
            lwdaStreamSynchronize(slot_table_entry->pusch_ul_phy->getStream());
            snprintf(jname, sizeof(jname), "pusch_output_%d_frame%d_subframe%d_slot%d.json", 
                                slot_table_entry->index, slot_table_entry->l2_slot_info->frameId, slot_table_entry->l2_slot_info->subFrameId, slot_table_entry->l2_slot_info->slotId);
            std::string json_file(jname);
            if(pt_dump_pusch_output(json_file, 
                                    slot_table_entry->tb_output, 
                                    slot_table_entry->pusch_ul_phy->get_obuf_size()) != PT_OK)
            {
                pt_err("pt_dump_pusch_output error\n");
                goto err;
            }

            pt_info("Output file %s created\n", jname);
        }

    #ifdef LWPHYCONTROLLER
        if(ptctx->controller == CONTROLLER_LWPHY)
        {
            if(plctx->measure_time >= PT_TIMER_PIPELINE)
                    slot_table_entry->t_copyOutputToCPU_start = get_ns();

            uint32_t cbCrcSize = slot_table_entry->pusch_ul_phy->get_tensor()->getCfgPrms().bePrms.CSum;
            uint32_t tbCrcSize = slot_table_entry->pusch_ul_phy->get_tensor()->getCfgPrms().bePrms.nTb;
            
        	LW_LWDA_CHECK(lwdaStreamSynchronize(copy_stream));    
            slot_table_entry->pusch_ul_phy->get_tensor()->copyOutputToCPU(copy_stream);
            LW_LWDA_CHECK(lwdaStreamSynchronize(copy_stream));

            if(plctx->measure_time >= PT_TIMER_PIPELINE)
                slot_table_entry->t_copyOutputToCPU_end = get_ns();

            slot_table_entry->l2_slot_info->fh_cmd.channel_params.block_params[fh::PUSCH].crc_buffer_output_size = cbCrcSize;
            slot_table_entry->l2_slot_info->fh_cmd.channel_params.block_params[fh::PUSCH].tb_crc_buffer_ouput_size = tbCrcSize;
            slot_table_entry->l2_slot_info->fh_cmd.channel_params.block_params[fh::PUSCH].output_buf_size = slot_table_entry->pusch_ul_phy->get_obuf_size();

            const uint8_t* srcTbBuf = slot_table_entry->pusch_ul_phy->get_obuf_addr<uint8_t*>();
            uint8_t* destTbBuf = (uint8_t*)(slot_table_entry->l2_slot_info->fh_cmd.channel_params.block_params[fh::PUSCH].output_data_buf);
            const uint32_t* srcCbCrcBuf = slot_table_entry->pusch_ul_phy->get_tensor()->getCRCs();
            uint32_t* destCbCrcBuf = slot_table_entry->l2_slot_info->fh_cmd.channel_params.block_params[fh::PUSCH].cb_crc_buffer;
            const uint32_t* srcTbCrcBuf = slot_table_entry->pusch_ul_phy->get_tensor()->getTbCRCs();
            uint32_t* destTbCrcBuf = slot_table_entry->l2_slot_info->fh_cmd.channel_params.block_params[fh::PUSCH].tb_crc_buffer;
            uint32_t ilwalid_CRC = *(slot_table_entry->pusch_ul_phy->crc_errors);
            /*
            for (uint32_t i = 0; i < tbCrcSize; i++) {
                if (srcCbCrcBuf[i] == 0) {
                    valid_CRC++;
                }
            }
            */

            if(plctx->measure_time >= PT_TIMER_PIPELINE)
                slot_table_entry->t_alloc_fn_start = get_ns();
            if (ilwalid_CRC == 0) {
                if(slot_table_entry->l2_slot_info->fh_cmd.alloc_fn == nullptr) {
                    pt_err("Allocate FN function is NULL for fh_cmd\n");
                }

                slot_table_entry->l2_slot_info->fh_cmd.alloc_fn((void**)&destTbBuf);
                if( destTbBuf == nullptr) {
                    pt_err("Unable to allocate CPU buffer for pusch pipeline\n");
                }
            } else {
                pt_err("No Valid CRC\n");
            }

            if(plctx->measure_time >= PT_TIMER_PIPELINE)
                slot_table_entry->t_alloc_fn_end = slot_table_entry->t_callback_start = get_ns();

            pt_info("Src TB buf=%p, Dest TB buf=%p TB size=%zd SrcCBCrcBuf=%p, DestCBCrcBuf=%p CB CRC size=%d SrcTbCrcBuf=%p, DestTbCrcBuf=%p TB CRC size=%d\n",
                    srcTbBuf, destTbBuf, slot_table_entry->pusch_ul_phy->get_obuf_size(),
                    srcCbCrcBuf, destCbCrcBuf, cbCrcSize,
                    srcTbCrcBuf, destTbCrcBuf, tbCrcSize);
            if (srcTbBuf != nullptr && destTbBuf != nullptr) {
                std::copy(srcTbBuf, srcTbBuf + slot_table_entry->pusch_ul_phy->get_obuf_size(), destTbBuf);
            } else {
                if (srcTbBuf == nullptr) {
                    pt_info("srcTbBuf is nullptr\n");
                } else {
                    pt_info("destTbBuf is nullptr\n");
                }
            }

            if(srcCbCrcBuf != nullptr && destCbCrcBuf != nullptr) {
                std::copy(srcCbCrcBuf, srcCbCrcBuf + cbCrcSize , destCbCrcBuf);
            } else {
                if (srcCbCrcBuf == nullptr) {
                    pt_info("srcCbCrcBuf is nullptr\n");
                } else {
                    pt_info("destCbCrcBuf is nullptr\n");
                }
            }

            if(srcTbCrcBuf != nullptr && destTbCrcBuf != nullptr) {
                std::copy(srcTbCrcBuf, srcTbCrcBuf + tbCrcSize, destTbCrcBuf);
            } else {
                if (srcTbCrcBuf == nullptr) {
                    pt_info("srcTbCrcBuf is nullptr\n");
                } else {
                    pt_info("destTbCrcBuf is nullptr\n");
                }
            }
            slot_table_entry->l2_slot_info->fh_cmd.post_callback(slot_table_entry->l2_slot_info->fh_cmd.slot_info.slot,
                                                    slot_table_entry->l2_slot_info->fh_cmd.channel_params,
                                                    slot_table_entry->pusch_ul_phy->get_obuf_size()
                                                );
            if(plctx->measure_time >= PT_TIMER_PIPELINE)
                slot_table_entry->t_callback_end = get_ns();

            pt_info("UL pipeline %d | 3GPP slot %d | Delta from slot tick %6.2fus | Copy to CPU %6.2f us | Alloc fn %6.2f us | Callback to controller done %6.2f us\n",
                    plctx->ul_num_processed_slots, slot_table_entry->l2_slot_info->slotId_3GPP,
                    get_us_from_ns(get_ns() - slot_table_entry->l2_slot_info->tick),
                    get_us_from_ns(slot_table_entry->t_copyOutputToCPU_end - slot_table_entry->t_copyOutputToCPU_start),
                    get_us_from_ns(slot_table_entry->t_callback_start - slot_table_entry->t_alloc_fn_start),
                    get_us_from_ns(slot_table_entry->t_callback_end - slot_table_entry->t_callback_start)
            );
        }
    #endif
        if(plctx->measure_time >= PT_TIMER_PIPELINE)
            slot_table_entry->t_pipeline_end = get_ns();
        while(check_force_quit(force_quit) == false && rte_ring_enqueue(ptctx->ring_free_ul, (void*)(slot_table_entry->l2_slot_info)) != 0);
        CHECK_FORCE_QUIT_STRING(force_quit, "stopped during rte_ring_enqueue ring_free_ul\n")

        //////////////////////////////////////////////////////////////////////////////
        /// PUSCH output validation
        //////////////////////////////////////////////////////////////////////////////
        if(plctx->validation & PT_VALIDATION_INPUT)
        {
            ret = lwphy_pusch_validate_input(slot_table_entry->pusch_ul_phy);
            if(ret != PT_OK)
                pt_info("Input Validation error, slot %d [global idx %d]\n", slot_table_entry->index, plctx->ul_num_processed_slots);
            else
                pt_info("Input Validation OK, slot %d [global idx %d]\n", slot_table_entry->index, plctx->ul_num_processed_slots);
        }

        #if 0
            //This is from CPU
            if(plctx->validation & PT_VALIDATION_CRC)
            {
                ret = lwphy_pusch_validate_crc(slot_table_entry->pusch_ul_phy);
                if(ret != PT_OK)
                    pt_info("CRC validation error %d, slot %d [global idx %d]\n", ret, slot_table_entry->index, plctx->ul_num_processed_slots);
                else
                    pt_info("CRC validation OK, slot %d [global idx %d]\n", slot_table_entry->index, plctx->ul_num_processed_slots);
            }
        #else
            if(plctx->validation & PT_VALIDATION_CRC)
            {
                pt_info("Slot 3GPP %d CRC errors: %d\n", slot_table_entry->l2_slot_info->slotId_3GPP, *(slot_table_entry->pusch_ul_phy->crc_errors));
                *(slot_table_entry->pusch_ul_phy->crc_errors) = 0;
            }
        #endif

        if(plctx->validation & PT_VALIDATION_CHECKSUM)
            pt_info("Slot 3GPP %d checksum %x\n", slot_table_entry->l2_slot_info->slotId_3GPP, plctx->ul_checksum_runtime[0]);

        if(plctx->measure_time)
            pt_info("//////////////////// GBuffer index %d (tot %d) Pipeline %d ////////////////////\n", 
                slot_table_entry->index,
                plctx->ul_num_processed_slots, pipeline_index);
        // (uint8_t)(PT_ACCESS_ONCE(((uint16_t*)plctx->slot_status_h)[slot_table_entry->index])),

        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Reset batch of mbufs
        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        for( ; local_index_mbatch != (plctx->map_slot_to_last_mbatch[slot_table_entry->index]); local_index_mbatch = (local_index_mbatch+1)%PT_MBUFS_BATCH_TOT)
        {
            for(i=0; i < plctx->mbatch[local_index_mbatch].mbufs_num; i++)
                rte_pktmbuf_free((struct rte_mbuf *)(plctx->mbatch[local_index_mbatch].mbufs[i]));

            //The first batch should not be considered!
            if(tot_rx_prep == -1)
                tot_rx_prep = 0;
            else
            {
                tot_rx_prep += get_us_from_ns(timerdiff_ns(plctx->mbatch_meta[local_index_mbatch].t_mbatch_rxprepare_end, plctx->mbatch_meta[local_index_mbatch].t_mbatch_wait));
                tot_rx_prep += get_us_from_ns(timerdiff_ns(plctx->mbatch_meta[local_index_mbatch].t_mbatch_ready_end, plctx->mbatch_meta[local_index_mbatch].t_mbatch_ready_start));
            }

            if(plctx->measure_time >= PT_TIMER_BATCH)
            {
                pt_info("BATCH TIMERS ==> [Slot %d - Pipeline %d] "
                        "RX and prepare %d mbufs in batch %d = %4.2f us | "
                        "prepare only = %4.2f us | "
                        "set mbatch ready = %4.2f us\n",
                        slot_table_entry->index, pipeline_index,
                        plctx->mbatch[local_index_mbatch].mbufs_num, plctx->mbatch[local_index_mbatch].index_mbatch,
                        //Tot
                        get_us_from_ns(timerdiff_ns(plctx->mbatch_meta[local_index_mbatch].t_mbatch_rxprepare_end, plctx->mbatch_meta[local_index_mbatch].t_mbatch_wait)),
                        //Effective
                        get_us_from_ns(timerdiff_ns(plctx->mbatch_meta[local_index_mbatch].t_mbatch_rxprepare_end, plctx->mbatch_meta[local_index_mbatch].t_mbatch_prepare)),
                        //Flush
                        get_us_from_ns(timerdiff_ns(plctx->mbatch_meta[local_index_mbatch].t_mbatch_ready_end, plctx->mbatch_meta[local_index_mbatch].t_mbatch_ready_start))
                        //get_us_from_ns(timerdiff_ns(plctx->mbatch_meta[local_index_mbatch].t_worker_done, plctx->mbatch_meta[local_index_mbatch].t_mbatch_ready_start))
                );

                #ifdef ORDER_KERNEL_TIMERS
                    pt_info("Kernel batch timers. Wait = %4.2fus, Prepare = %4.2fus, Copy = %4.2fus\n",
                        (( ((float)(plctx->mbatch[local_index_mbatch].timers[TIMER_START_PREPARE] - plctx->mbatch[local_index_mbatch].timers[TIMER_START_WAIT])     /   (float)plctx->lwdactx.Hz)) *1000)*1000,
                        (( ((float)(plctx->mbatch[local_index_mbatch].timers[TIMER_START_COPY]    - plctx->mbatch[local_index_mbatch].timers[TIMER_START_PREPARE])  /   (float)plctx->lwdactx.Hz)) *1000)*1000,
                        (( ((float)(plctx->mbatch[local_index_mbatch].timers[TIMER_START_DONE]    - plctx->mbatch[local_index_mbatch].timers[TIMER_START_COPY])     /   (float)plctx->lwdactx.Hz)) *1000)*1000
                        //(( ((float)(plctx->mbatch[local_index_mbatch].timers[TIMER_END_DONE]      - plctx->mbatch[local_index_mbatch].timers[TIMER_START_DONE])     /   (float)plctx->lwdactx.Hz)) *1000)*1000
                    );
                #endif
            }

            plctx->mbatch[local_index_mbatch].mbufs_num = 0;
            last_mbatch_end_time = plctx->mbatch_meta[local_index_mbatch].t_mbatch_ready_end;
            
            // Free the mbatch
            PT_ACCESS_ONCE(((uint32_t*)plctx->mbufs_batch_ready_flags_h)[plctx->mbatch[local_index_mbatch].index_mbatch]) = PT_MBATCH_FREE;
            rte_wmb();
            flush_gmem(plctx);
        }

        pt_info("UL slot table %d last mbatch was %d\n", slot_table_entry->index, local_index_mbatch);

        PT_ACCESS_ONCE(plctx->map_slot_to_last_mbatch[slot_table_entry->index]) = 0;
        rte_mb();

        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Slot timers
        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        if(plctx->measure_time >= PT_TIMER_PIPELINE)
        {
            if(ptctx->controller == CONTROLLER_LWPHY)
            {

                pt_info("\n[Slot %d - Pipeline %d]\n"
                        "\tOverlap prepare packets and order = %4.2f us\n"
                        "\tOrder Kernel time = %4.2f us\n"
                        // "\tTime for all batch - 1 = %4.2f us\n"
                        "\tLast mbatch latency = %4.2f us\n" //" = ready (%4.2f) - flush (%4.2f) "
                        "\tlwPHY CPU launch time = %4.2f us\n"
                        "\tlwPHY GPU pipeline time = %4.2f us\n" //"READY time %4.2f us\n"
                        "\tAlloc function time = %4.2f us\n"
                        "\tCopyOutputToCPU launch time = %4.2f us\n"
                        "\tCallback time = %4.2f us\n"
                        "\tTotal pipeline time = %4.2f us\n"
                        "\n",
                        slot_table_entry->index, pipeline_index,
                        tot_rx_prep,
                        //Order Kernel time
                        get_us_from_ns(timerdiff_ns(slot_table_entry->t_start_ch, slot_table_entry->t_start_order)),
                        //Last batch time
                        get_us_from_ns(timerdiff_ns(slot_table_entry->t_start_ch, last_mbatch_end_time)), //last_mbatch_end_time) - get_us_from_ns(timerdiff_ns(last_mbatch_ready_flush, last_mbatch_ready)),
                        //lwPHY pipeline time
                        get_us_from_ns(timerdiff_ns(slot_table_entry->t_stop_enqueue, slot_table_entry->t_start_enqueue)),
                        get_us_from_ns(timerdiff_ns(slot_table_entry->t_slot_end, slot_table_entry->t_start_ch)),
                        //TB buffer alloc time
                        get_us_from_ns(timerdiff_ns(slot_table_entry->t_alloc_fn_end, slot_table_entry->t_alloc_fn_start)),
                        //copyOutputToCPU time
                        get_us_from_ns(timerdiff_ns(slot_table_entry->t_copyOutputToCPU_end, slot_table_entry->t_copyOutputToCPU_start)),
                        //Callback time
                        get_us_from_ns(timerdiff_ns(slot_table_entry->t_callback_end, slot_table_entry->t_callback_start)),
                        get_us_from_ns(timerdiff_ns(slot_table_entry->t_pipeline_end, slot_table_entry->t_start_order))
                );
            }
            else
            {
                pt_info("\n[Slot %d - Pipeline %d]\n"
                        "\tOverlap prepare packets and order = %4.2f us\n"
                        "\tOrder Kernel time = %4.2f us\n"
                        // "\tTime for all batch - 1 = %4.2f us\n"
                        "\tLast mbatch latency = %4.2f us\n" //" = ready (%4.2f) - flush (%4.2f) "
                        "\tlwPHY CPU launch time = %4.2f us\n"
                        "\tlwPHY GPU pipeline time = %4.2f us\n" //"READY time %4.2f us\n"
                        "\tTotal pipeline time = %4.2f us\n"
                        "\n",
                        slot_table_entry->index, pipeline_index,
                        tot_rx_prep,
                        //Order Kernel time
                        get_us_from_ns(timerdiff_ns(slot_table_entry->t_start_ch, slot_table_entry->t_start_order)),
                        //Last batch time
                        get_us_from_ns(timerdiff_ns(slot_table_entry->t_start_ch, last_mbatch_end_time)), //last_mbatch_end_time) - get_us_from_ns(timerdiff_ns(last_mbatch_ready_flush, last_mbatch_ready)),
                        //lwPHY pipeline time
                        get_us_from_ns(timerdiff_ns(slot_table_entry->t_stop_enqueue, slot_table_entry->t_start_enqueue)),
                        get_us_from_ns(timerdiff_ns(slot_table_entry->t_slot_end, slot_table_entry->t_start_ch)),
                        get_us_from_ns(timerdiff_ns(slot_table_entry->t_pipeline_end, slot_table_entry->t_start_order))
                );
            }
        }
       //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        PT_ACCESS_ONCE(plctx->mbufs_slot_start_flags_h[slot_table_entry->index]) = PT_SLOT_FREE;
        PT_ACCESS_ONCE(plctx->mbufs_slot_order_flags_h[slot_table_entry->index]) = PT_SLOT_FREE;
        PT_ACCESS_ONCE(plctx->mbufs_slot_done_flags_h[slot_table_entry->index]) = PT_SLOT_FREE;

        // PT_ACCESS_ONCE(((uint16_t*)plctx->slot_status_h)[slot_table_entry->index]) = PT_SLOT_STATUS_FREE;

        if(plctx->measure_time)
            pt_info("//////////////////////////////////////////////////////////////////////////////////////\n\n");
        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        #ifdef PROFILE_LWTX_RANGES
            uplink_endpoint_core_count++;
            if(uplink_endpoint_core_count == 10)
            {
                set_force_quit(force_quit);
                break;
            }
        #endif

        plctx->ul_num_processed_slots++;
        last_slot_table_entry = slot_table_entry;

        if(ptctx->controller != CONTROLLER_LWPHY && plctx->uplink_slots > 0 && plctx->ul_num_processed_slots >= plctx->uplink_slots)
        {
            pt_info("Max UL slots %d reached. Exit..\n", plctx->uplink_slots);
            break;
        }

    }

err:
    pt_warn("EXIT\n");
    lwdaStreamDestroy(copy_stream);
    return PT_OK;
}

int uplink_prepare_core(void *param) {
    SET_THREAD_NAME(__FUNCTION__);
    int index_mbufs=0, index=0, index_mbatch=0, start_mbatch=0, index_ready=0, local_index_mbatch=0, mbufs_deq=0, local_index_mbufs=0;
    int index_stream=0, queue_index=0, slot_num=0, slot_offset=0, slot_index=0;
    int last_mbatch=0;
    unsigned avail=0;
    uint64_t start_batching=0;
    rte_unique_mbufs mbufs{nullptr, &rte_free};
    float pk_ms;

    struct core_info * cinfo = (struct core_info *) param;
    struct phytools_ctx * ptctx = cinfo->ptctx;
    struct pipeline_ctx * plctx = cinfo->plctx;
    int pipeline_index          = plctx->index;

    timerdiff_us(0,0); // initialize timer_hz inside

    LW_LWDA_CHECK(lwdaSetDevice(plctx->lwdactx.gpu_id));

    mbufs.reset(dpdk_allocate_mbufs(ptctx->tot_pkts_x_batch));
    if(mbufs.get() == NULL)
    {
        pt_err("mbufs allocation error\n");
        return PT_ERR;
    }

    pt_info("Pipeline %d Order core on lcore %u socket %d Sched. priority %d\n", pipeline_index, rte_lcore_id(), rte_socket_id(), pt_get_thread_priority());

    while(check_force_quit(force_quit) == false)
    {
        while(check_force_quit(force_quit) == false && (PT_ACCESS_ONCE(((uint32_t*)plctx->mbufs_batch_ready_flags_h)[index_mbatch]) != PT_MBATCH_FREE));
        CHECK_FORCE_QUIT_STRING(force_quit, "stopped during ready != 0\n")

        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        /// Prepare packets for new batch
        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////        
        if(plctx->measure_time >= PT_TIMER_PIPELINE)
            plctx->mbatch_meta[index_mbatch].t_mbatch_wait = get_ns();

        mbufs_deq=0;
        start_batching=0;
        index_mbufs=0;
        last_mbatch=0;

        memset(plctx->mbatch[index_mbatch].mbufs_size, 0, ptctx->tot_pkts_x_batch * sizeof(uint32_t));
        while(check_force_quit(force_quit) == false && index_mbufs < ptctx->tot_pkts_x_batch)
        {
            mbufs_deq = rte_ring_dequeue_burst(plctx->ring_rxmbufs[queue_index], (void **)&((mbufs.get())[0]), ptctx->tot_pkts_x_batch-index_mbufs, &avail);
            CHECK_FORCE_QUIT_STRING(force_quit, "stopped during rte_ring_dequeue mbufs\n")
            if(mbufs_deq > 0)
            {
                if(index_mbufs == 0)
                {
                    start_batching = get_ns();
                    if(plctx->measure_time >= PT_TIMER_PIPELINE)
                        plctx->mbatch_meta[index_mbatch].t_mbatch_prepare = start_batching;
                }

                for(
                    local_index_mbufs=0;
                    local_index_mbufs < mbufs_deq && index_mbufs < ptctx->tot_pkts_x_batch;
                    local_index_mbufs++, index_mbufs++
                )
                {
                    plctx->mbatch[index_mbatch].mbufs[index_mbufs] = (mbufs.get())[local_index_mbufs];

                    if(plctx->dpdkctx.hds)
                    {
                        assert((mbufs.get())[local_index_mbufs]->nb_segs == 2);
                        // slot_num    = oran_get_slot_from_hdr((uint8_t*)(DPDK_GET_MBUF_ADDR((mbufs.get())[local_index_mbufs], 0)));
                        // slot_index = slot_num % PT_MAX_SLOT_ID;
                        slot_offset = oran_get_offset_from_hdr((uint8_t*)(DPDK_GET_MBUF_ADDR((mbufs.get())[local_index_mbufs], 0)),
                                                    queue_index, SLOT_NUM_SYMS, 
                                                    plctx->pusch_slot_table_entry[slot_index].slot_dims.prbs_per_symbol, PRB_SIZE_16F
                                                );
                        plctx->mbatch[index_mbatch].mbufs_size[index_mbufs]        = (uint32_t)((mbufs.get())[local_index_mbufs]->next->data_len);
                        plctx->mbatch[index_mbatch].mbufs_payload_src[index_mbufs] = (uintptr_t)(DPDK_GET_MBUF_ADDR((mbufs.get())[local_index_mbufs]->next, 0));
                        plctx->mbatch[index_mbatch].mbufs_payload_dst[index_mbufs] = (uintptr_t) (((uint8_t*)plctx->pusch_slot_table_entry[slot_index].pusch_ul_phy->get_ibuf_addr<uintptr_t>()) + slot_offset);
                        plctx->cache_count_prbs[slot_index] += oran_umsg_get_num_prb((uint8_t*)(DPDK_GET_MBUF_ADDR((mbufs.get())[local_index_mbufs], 0)));

                        if(plctx->cache_count_prbs[slot_index] >= plctx->pusch_slot_table_entry[slot_index].slot_dims.prbs_per_slot)
                        {
                            last_mbatch=1;
                            PT_ACCESS_ONCE(plctx->map_slot_to_last_mbatch[slot_index]) = (index_mbatch+1)%PT_MBUFS_BATCH_TOT;
                            plctx->cache_count_prbs[slot_index] = 0;
                            slot_index = (slot_index+1)%PT_MAX_SLOT_ID;
                        }

                        pt_info("Batch %d Last %d Computed Slot %d Used slot %d (frame %d subframe %d slot %d) offset %x PRBs %d prbs_per_slot %zd tot slot size %zd size %d slot gbuf %p src %lx dst %lx nb_segs=%d\n", 
                                index_mbatch, last_mbatch, slot_num, slot_index,
                                oran_umsg_get_frame_id((uint8_t*)(DPDK_GET_MBUF_ADDR((mbufs.get())[local_index_mbufs], 0))),
                                oran_umsg_get_subframe_id((uint8_t*)(DPDK_GET_MBUF_ADDR((mbufs.get())[local_index_mbufs], 0))),
                                oran_umsg_get_slot_id((uint8_t*)(DPDK_GET_MBUF_ADDR((mbufs.get())[local_index_mbufs], 0))),
                                slot_offset, plctx->cache_count_prbs[slot_index], 
                                plctx->pusch_slot_table_entry[slot_index].slot_dims.prbs_per_slot,
                                plctx->pusch_slot_table_entry[slot_index].slot_dims.data_sz,
                                plctx->mbatch[index_mbatch].mbufs_size[index_mbufs],
                                ((uint8_t*)plctx->pusch_slot_table_entry[slot_index].pusch_ul_phy->get_ibuf_addr<uintptr_t>()),
                                plctx->mbatch[index_mbatch].mbufs_payload_src[index_mbufs],
                                plctx->mbatch[index_mbatch].mbufs_payload_dst[index_mbufs],
                                (mbufs.get())[local_index_mbufs]->nb_segs
                            );
                    }
                    else
                    {
                        plctx->mbatch[index_mbatch].mbufs_size[index_mbufs]         = (uint32_t) ((mbufs.get())[local_index_mbufs]->data_len);
                        plctx->mbatch[index_mbatch].mbufs_payload_src[index_mbufs]  = (uintptr_t) DPDK_GET_MBUF_ADDR((mbufs.get())[local_index_mbufs], 0);
                        plctx->mbatch[index_mbatch].mbufs_flow[index_mbufs]         = (uint16_t) queue_index;
                    }
                    
                    plctx->mbatch[index_mbatch].mbufs_num++;
                }
            }

            if(index_mbufs > 0)
            {
                if(( get_us_from_ns(timerdiff_ns(get_ns(), start_batching)) >= (double)(plctx->rx_batching_us)))
                {
                    start_batching=0;
                    break;
                }
            }

            queue_index++;
            if(queue_index == plctx->flow_tot)
                queue_index=0;
        }

        if(plctx->measure_time >= PT_TIMER_PIPELINE)
            plctx->mbatch_meta[index_mbatch].t_mbatch_rxprepare_end = get_ns();
        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        rte_mb();
        flush_gmem(plctx);

        // pt_info("Pipeline [%d] mbatch[%d] ready with %d pkts\n", pipeline_index, index_mbatch, plctx->mbatch[index_mbatch].mbufs_num);
        if(plctx->dpdkctx.hds && last_mbatch == 1)
            plctx->mbatch[index_mbatch].last_mbatch = 1;

        while(check_force_quit(force_quit) == false && rte_ring_enqueue(plctx->ring_timer_batch, (void*)&(plctx->mbatch[index_mbatch])) != 0);
        CHECK_FORCE_QUIT_STRING(force_quit, "ring_timer_batch stopped during rte_ring_enqueue\n")

        index_mbatch = (index_mbatch+1)%PT_MBUFS_BATCH_TOT;
        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    }

err:

    //Make all PK/MK kernels exit!
    for(index_mbatch=0; index_mbatch < PT_MBUFS_BATCH_TOT; index_mbatch++)
        PT_ACCESS_ONCE(((uint32_t*)plctx->mbufs_batch_ready_flags_h)[index_mbatch]) = PT_MBATCH_EXIT;

    rte_wmb();
    flush_gmem(plctx);
    LW_LWDA_CHECK(lwdaDeviceSynchronize());
    pt_warn("EXIT\n");

    return PT_OK;
}

int uplink_rx_core(void *param) {
    SET_THREAD_NAME(__FUNCTION__);
    int nb_rx=0, rx_retries=0, index_mbufs=0, totrx=0, rxq=0;
    unsigned free=0;
    rte_unique_mbufs mbufs{nullptr, &rte_free};
    struct core_info * cinfo = (struct core_info *) param;
    // int rxq                     = cinfo->rxq;
    int port_id                 = cinfo->port_id;
    struct phytools_ctx * ptctx = cinfo->ptctx;
    struct pipeline_ctx * plctx = cinfo->plctx;
    int pipeline_index          = plctx->index;

    pt_info("Pipeline %d RX queue %d on lcore %u port id %d socket %d Sched. priority %d\n", 
        pipeline_index, rxq, rte_lcore_index(rte_lcore_id()), port_id, rte_socket_id(), pt_get_thread_priority());

    mbufs.reset(dpdk_allocate_mbufs(ptctx->tot_pkts_x_batch));
    if(mbufs.get() == NULL)
    {
        pt_err("mbufs allocation error\n");
        return PT_ERR;
    }

    while(check_force_quit(force_quit) == false) {
        nb_rx=0;
        rx_retries=0;

        PUSH_RANGE("RX", LWLWTX_COL_1);

        while(check_force_quit(force_quit) == false && nb_rx == 0) //&& nb_rx < (plctx->dpdkctx.mbuf_x_burst - PT_DRIVER_MIN_RX_PKTS) && rx_retries < PT_RX_RETRY
        {   
            // Can't use plctx->flow_list
            nb_rx += dpdk_rx_burst_pkts(mbufs, port_id, rxq, plctx->dpdkctx.mbuf_x_burst);
            if(nb_rx > 0)
            {
                while(check_force_quit(force_quit) == false && (rte_ring_enqueue_bulk(plctx->ring_rxmbufs[rxq], (void**)(mbufs.get()), nb_rx, &free) == 0));
                CHECK_FORCE_QUIT_STRING(force_quit, "stopped during rte_ring_enqueue_bulk\n")
            }

            rxq++;
            if(rxq == plctx->flow_tot)
                rxq=0;
        }
        POP_RANGE;
        CHECK_FORCE_QUIT_STRING(force_quit, "stopped during rte_eth_rx_burst\n")
    }

err:
    pt_warn("EXIT\n");
    return PT_OK;
}
