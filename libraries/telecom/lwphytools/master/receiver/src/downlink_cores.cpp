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
#include <algorithm>

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///// DL pipelines
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

static int enable_pdsch=0, enable_pddch_ul=0, enable_pdcch_dl_301=0, enable_pdcch_dl_301a=0, enable_pdcch_dl_301b=0, enable_pbch=0;

static int cleanup_dl_output(int index_slot, struct pipeline_ctx * plctx) {
    struct pt_slot_table * slot_table_ptr;
    if(!plctx)
        return PT_EILWAL;

    slot_table_ptr = &(plctx->pdsch_slot_table_entry[index_slot]);
    slot_table_ptr->pdsch_dl_phy->CleanupOutput();

    return PT_OK;
}

//Need to add wait function!
static int start_pdsch_pipeline(int index_slot, struct phytools_ctx * ptctx, struct pipeline_ctx * plctx) {
    int index=0;
    uint64_t pip_start, order_start;
    struct pt_slot_table * slot_table_ptr;

    if(!plctx)
        return PT_EILWAL;

    slot_table_ptr = &(plctx->pdsch_slot_table_entry[index_slot]);

    // if(plctx->measure_time >= PT_TIMER_PIPELINE)
    //    slot_table_ptr->t_start_enqueue = get_ns();

    lwphy_pdsch_run(slot_table_ptr->pdsch_dl_phy);

    //if(plctx->measure_time >= PT_TIMER_PIPELINE)
    //    slot_table_ptr->t_stop_enqueue = get_ns();

    pt_launch_kernel_write(&(plctx->pdsch_phy_done_h[index_slot]), PT_SLOT_DONE, slot_table_ptr->pdsch_dl_phy->getStream());

    enable_pdsch=1;

    return PT_OK;
err:
    return PT_ERR;
}

static int wait_pdsch_pipeline(int index_slot, struct pipeline_ctx * plctx) {
    struct pt_slot_table * slot_table_ptr;

    if(!plctx)
        return PT_EILWAL;

    if(enable_pdsch == 0)
        return PT_OK;

    slot_table_ptr = &(plctx->pdsch_slot_table_entry[index_slot]);

    while(check_force_quit(force_quit) == false && (PT_ACCESS_ONCE(plctx->pdsch_phy_done_h[index_slot]) < PT_SLOT_DONE));
    CHECK_FORCE_QUIT_STRING(force_quit, "start_pdsch_pipeline stopped during wait done\n")

    // if(plctx->measure_time >= PT_TIMER_PIPELINE) {
    //     slot_table_ptr->t_slot_end = get_ns();
    //     pt_info("Slot %d ==> PDSCH CPU time: %4.2f us, GPU time: %4.2f us\n",
    //             index_slot,
    //             get_us_from_ns(timerdiff_ns(slot_table_ptr->t_stop_enqueue, slot_table_ptr->t_start_enqueue)),
    //             get_us_from_ns(timerdiff_ns(slot_table_ptr->t_slot_end, slot_table_ptr->t_stop_enqueue))
    //     );
    // }

    enable_pdsch=0;

    return PT_OK;
err:
    return PT_ERR;
}

static int start_pdcch_ul_pipeline(int index_slot, uint16_t slotId_3GPP, struct phytools_ctx * ptctx, struct pipeline_ctx * plctx) {
    int index=0;
    uint64_t pip_start, order_start;
    struct pt_slot_table * slot_table_ptr;

    if(!plctx)
        return PT_EILWAL;

    slot_table_ptr = &(plctx->pdcch_ul_table_entry[index_slot]);

    // if(plctx->measure_time >= PT_TIMER_PIPELINE)
    //    slot_table_ptr->t_start_enqueue = get_ns();

    slot_table_ptr->pdcch_ul_phy->Run(0, slotId_3GPP);
    lwdaStream_t stream = slot_table_ptr->pdcch_ul_phy->getStream();

    //if(plctx->measure_time >= PT_TIMER_PIPELINE)
    //    slot_table_ptr->t_stop_enqueue = get_ns();

    pt_launch_kernel_write(&(plctx->pdcch_ul_phy_done_h[index_slot]), PT_SLOT_DONE, stream);

    enable_pddch_ul=1;

    return PT_OK;
err:
    return PT_ERR;
}

static int wait_pdcch_ul_pipeline(int index_slot, struct pipeline_ctx * plctx) {
    struct pt_slot_table * slot_table_ptr;

    if(!plctx)
        return PT_EILWAL;

    if(enable_pddch_ul == 0)
        return PT_OK;

    slot_table_ptr = &(plctx->pdcch_ul_table_entry[index_slot]);

    while(check_force_quit(force_quit) == false && (PT_ACCESS_ONCE(plctx->pdcch_ul_phy_done_h[index_slot]) < PT_SLOT_DONE));
    CHECK_FORCE_QUIT_STRING(force_quit, "start_pdcch_ul_pipeline stopped during wait done\n")

    // if(plctx->measure_time >= PT_TIMER_PIPELINE) {
    //     slot_table_ptr->t_slot_end = get_ns();
    //     pt_info("Slot %d ==> PDCCH UL CPU time: %4.2f us, GPU time: %4.2f us\n",
    //             index_slot,
    //             get_us_from_ns(timerdiff_ns(slot_table_ptr->t_stop_enqueue, slot_table_ptr->t_start_enqueue)),
    //             get_us_from_ns(timerdiff_ns(slot_table_ptr->t_slot_end, slot_table_ptr->t_stop_enqueue))
    //     );
    // }

    enable_pddch_ul=0;

    return PT_OK;
err:
    return PT_ERR;
}

static int start_pdcch_dl_301_pipeline(int index_slot, uint16_t slotId_3GPP, struct phytools_ctx * ptctx, struct pipeline_ctx * plctx) {
    int index=0;
    uint64_t pip_start, order_start;
    struct pt_slot_table * slot_table_ptr;

    if(!plctx)
        return PT_EILWAL;

    slot_table_ptr = &(plctx->pdcch_dl_table_entry[index_slot]);

    // if(plctx->measure_time >= PT_TIMER_PIPELINE)
    //    slot_table_ptr->t_start_enqueue = get_ns();

    slot_table_ptr->pdcch_dl_301_phy->Run(0, slotId_3GPP);
    lwdaStream_t stream = slot_table_ptr->pdcch_dl_301_phy->getStream();

    //if(plctx->measure_time >= PT_TIMER_PIPELINE)
    //    slot_table_ptr->t_stop_enqueue = get_ns();

    pt_launch_kernel_write(&(plctx->pdcch_dl_301_phy_done_h[index_slot]), PT_SLOT_DONE, stream);

    enable_pdcch_dl_301=1;

    return PT_OK;
err:
    return PT_ERR;
}

static int wait_pdcch_dl_301_pipeline(int index_slot, struct pipeline_ctx * plctx) {
    struct pt_slot_table * slot_table_ptr;

    if(!plctx)
        return PT_EILWAL;

    if(enable_pdcch_dl_301 == 0)
        return PT_OK;

    slot_table_ptr = &(plctx->pdcch_dl_table_entry[index_slot]);

    while(check_force_quit(force_quit) == false && (PT_ACCESS_ONCE(plctx->pdcch_dl_301_phy_done_h[index_slot]) < PT_SLOT_DONE));
    CHECK_FORCE_QUIT_STRING(force_quit, "wait_pdcch_dl_301_pipeline stopped during wait done\n")

    // if(plctx->measure_time >= PT_TIMER_PIPELINE) {
    //     slot_table_ptr->t_slot_end = get_ns();
    //     pt_info("Slot %d ==> PDCCH DL 301 CPU time: %4.2f us, GPU time: %4.2f us\n",
    //             index_slot,
    //             get_us_from_ns(timerdiff_ns(slot_table_ptr->t_stop_enqueue, slot_table_ptr->t_start_enqueue)),
    //             get_us_from_ns(timerdiff_ns(slot_table_ptr->t_slot_end, slot_table_ptr->t_stop_enqueue))
    //     );
    // }

    enable_pdcch_dl_301=0;

    return PT_OK;
err:
    return PT_ERR;
}

static int start_pdcch_dl_301a_pipeline(int index_slot, uint16_t slotId_3GPP, struct phytools_ctx * ptctx, struct pipeline_ctx * plctx) {
    int index=0;
    uint64_t pip_start, order_start;
    struct pt_slot_table * slot_table_ptr;

    if(!plctx)
        return PT_EILWAL;

    slot_table_ptr = &(plctx->pdcch_dl_table_entry[index_slot]);

    // if(plctx->measure_time >= PT_TIMER_PIPELINE)
    //    slot_table_ptr->t_start_enqueue = get_ns();

    slot_table_ptr->pdcch_dl_301a_phy->Run(0, slotId_3GPP);
    lwdaStream_t stream = slot_table_ptr->pdcch_dl_301a_phy->getStream();

    //if(plctx->measure_time >= PT_TIMER_PIPELINE)
    //    slot_table_ptr->t_stop_enqueue = get_ns();

    pt_launch_kernel_write(&(plctx->pdcch_dl_301a_phy_done_h[index_slot]), PT_SLOT_DONE, stream);

    enable_pdcch_dl_301a=1;

    return PT_OK;
err:
    return PT_ERR;
}

static int wait_pdcch_dl_301a_pipeline(int index_slot, struct pipeline_ctx * plctx) {
    int index=0;
    uint64_t pip_start, order_start;
    struct pt_slot_table * slot_table_ptr;

    if(!plctx)
        return PT_EILWAL;

    if(enable_pdcch_dl_301a == 0)
        return PT_OK;

    slot_table_ptr = &(plctx->pdcch_dl_table_entry[index_slot]);

    while(check_force_quit(force_quit) == false && (PT_ACCESS_ONCE(plctx->pdcch_dl_301a_phy_done_h[index_slot]) < PT_SLOT_DONE));
    CHECK_FORCE_QUIT_STRING(force_quit, "start_pdcch_dl_301a_pipeline stopped during wait done\n")

    // if(plctx->measure_time >= PT_TIMER_PIPELINE) {
    //     slot_table_ptr->t_slot_end = get_ns();
    //     pt_info("Slot %d ==> PDCCH DL 301a CPU time: %4.2f us, GPU time: %4.2f us\n",
    //             index_slot,
    //             get_us_from_ns(timerdiff_ns(slot_table_ptr->t_stop_enqueue, slot_table_ptr->t_start_enqueue)),
    //             get_us_from_ns(timerdiff_ns(slot_table_ptr->t_slot_end, slot_table_ptr->t_stop_enqueue))
    //     );
    // }

    enable_pdcch_dl_301a=0;

    return PT_OK;
err:
    return PT_ERR;
}

static int start_pdcch_dl_301b_pipeline(int index_slot, uint16_t slotId_3GPP, struct phytools_ctx * ptctx,
				   struct pipeline_ctx * plctx) {
    int index=0;
    uint64_t pip_start, order_start;
    struct pt_slot_table * slot_table_ptr;

    if(!plctx)
        return PT_EILWAL;

    slot_table_ptr = &(plctx->pdcch_dl_table_entry[index_slot]);

    // if(plctx->measure_time >= PT_TIMER_PIPELINE)
    //     slot_table_ptr->t_start_enqueue = get_ns();

    slot_table_ptr->pdcch_dl_301b_phy->Run(0, slotId_3GPP);
    lwdaStream_t stream = slot_table_ptr->pdcch_dl_301b_phy->getStream();

    // if(plctx->measure_time >= PT_TIMER_PIPELINE)
    //     slot_table_ptr->t_stop_enqueue = get_ns();

    pt_launch_kernel_write(&(plctx->pdcch_dl_301b_phy_done_h[index_slot]), PT_SLOT_DONE, stream);

    enable_pdcch_dl_301b=1;

    return PT_OK;
err:
    return PT_ERR;
}

static int wait_pdcch_dl_301b_pipeline(int index_slot, struct pipeline_ctx * plctx) {
    int index=0;
    uint64_t pip_start, order_start;
    struct pt_slot_table * slot_table_ptr;

    if(!plctx)
        return PT_EILWAL;

    if(enable_pdcch_dl_301b == 0)
        return PT_OK;

    slot_table_ptr = &(plctx->pdcch_dl_table_entry[index_slot]);

    while(check_force_quit(force_quit) == false && (PT_ACCESS_ONCE(plctx->pdcch_dl_301b_phy_done_h[index_slot]) < PT_SLOT_DONE));
    CHECK_FORCE_QUIT_STRING(force_quit, "start_pdcch_dl_301b_pipeline stopped during wait done\n")

    // if(plctx->measure_time >= PT_TIMER_PIPELINE) {
    //     slot_table_ptr->t_slot_end = get_ns();
    //     pt_info("Slot %d ==> PDCCH DL B CPU time: %4.2f us, GPU time: %4.2f us\n",
    //             index_slot,
    //             get_us_from_ns(timerdiff_ns(slot_table_ptr->t_stop_enqueue, slot_table_ptr->t_start_enqueue)),
    //             get_us_from_ns(timerdiff_ns(slot_table_ptr->t_slot_end, slot_table_ptr->t_stop_enqueue))
    //     );
    // }

    enable_pdcch_dl_301b=0;

    return PT_OK;
err:
    return PT_ERR;
}

static int start_pbch_pipeline(int index_slot, uint16_t slotId_3GPP, struct phytools_ctx * ptctx, struct pipeline_ctx * plctx) {
    int index=0;
    uint64_t pip_start, order_start;
    struct pt_slot_table * slot_table_ptr;

    if(!plctx)
        return PT_EILWAL;

    slot_table_ptr = &(plctx->pbch_table_entry[index_slot]);

    // if(plctx->measure_time >= PT_TIMER_PIPELINE)
    //     slot_table_ptr->t_start_enqueue = get_ns();

    slot_table_ptr->pbch_phy->Run(0, slotId_3GPP);
    lwdaStream_t stream = slot_table_ptr->pbch_phy->getStream();

    // if(plctx->measure_time >= PT_TIMER_PIPELINE)
    //     slot_table_ptr->t_stop_enqueue = get_ns();

    pt_launch_kernel_write(&(plctx->pbch_phy_done_h[index_slot]), PT_SLOT_DONE, stream);

    enable_pbch=1;

    return PT_OK;
err:
    return PT_ERR;
}

static int wait_pbch_pipeline(int index_slot, struct pipeline_ctx * plctx) {
    struct pt_slot_table * slot_table_ptr;

    if(!plctx)
        return PT_EILWAL;

    if(enable_pbch == 0)
        return PT_OK;

    slot_table_ptr = &(plctx->pbch_table_entry[index_slot]);

    while(check_force_quit(force_quit) == false && (PT_ACCESS_ONCE(plctx->pbch_phy_done_h[index_slot]) < PT_SLOT_DONE));
    CHECK_FORCE_QUIT_STRING(force_quit, "start_pbch_pipeline stopped during wait done\n")

    // if(plctx->measure_time >= PT_TIMER_PIPELINE) {
    //     slot_table_ptr->t_slot_end = get_ns();
    //     pt_info("Slot %d ==> PBCH CPU time: %4.2f us, GPU time: %4.2f us\n",
    //             index_slot,
    //             get_us_from_ns(timerdiff_ns(slot_table_ptr->t_stop_enqueue, slot_table_ptr->t_start_enqueue)),
    //             get_us_from_ns(timerdiff_ns(slot_table_ptr->t_slot_end, slot_table_ptr->t_stop_enqueue))
    //     );
    // }

    enable_pbch=0;

    return PT_OK;
err:
    return PT_ERR;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///// DL core
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int downlink_processing_core(void *param)
{
    SET_THREAD_NAME(__FUNCTION__);
    int index_slot=0, index_tx_list=0, ret=0, i=0, j=0, created_pkts=0, startPrbc=0, tot_pkts_x_symb=0, tot_pkts_x_slot=0, startSym=0, numSym=ORAN_ALL_SYMBOLS;
    uint8_t ecpriSeqid_c=0, frameId=0, subFrameId=0, slotId=0, sym_idx=0, slotId_3GPP=0;
    uint8_t pdsch_on=0, pdcch_ul_on=0, pdcch_dl_on=0, pbch_on=0;
    uint16_t nb_tx = 0;
    uint32_t numPrbc=ORAN_RB_ALL;
    uint64_t slot_tick_ts, last_dl_t=0;
    std::vector<uint16_t> tv_flow_count;
    uint16_t* ecpriSeqid_u;
    Slot * txPdschSlot;
    struct l2_control_slot_info * l2_slot_info;
    char jname[512];

    struct core_info * cinfo = (struct core_info *) param;
    int port_id                 = cinfo->port_id;
    struct phytools_ctx * ptctx = cinfo->ptctx;
    struct pipeline_ctx * plctx = cinfo->plctx;
    int pipeline_index          = plctx->index;

    timerdiff_us(0,0); // initialize timer_hz inside

    pt_info("Pipeline %d TX queue clock %d TX queue U %d TX queue C %d on lcore %u port id %d socket %d Sched. priority %d\n",
            pipeline_index, plctx->dpdkctx.ts_txq, plctx->dpdkctx.dl_txq, plctx->dpdkctx.c_txq,
            rte_lcore_index(rte_lcore_id()), port_id, rte_socket_id(), pt_get_thread_priority());

    while(check_force_quit(force_quit) == false)
    {
        txPdschSlot = &(plctx->pdsch_slot_table_entry[index_slot].slot_dims);
        l2_slot_info = NULL;

        while(check_force_quit(force_quit) == false && rte_ring_dequeue(ptctx->ring_start_dl, (void **)&l2_slot_info) != 0);
        CHECK_FORCE_QUIT_STRING(force_quit, "stopped during rte_ring_dequeue\n")

        if(rte_ring_count(ptctx->ring_free_dl) <= PT_RING_ELEMS-5)
        {
            pt_info("DL pipeline %d | 3GPP slot %d | Delta from slot tick %6.2fus | Still processing. Dropping new DL FH cmd 3GPP[%u] frame[%u] subframe[%u] slot[%u]\n",
                plctx->dl_num_processed_slots, slotId_3GPP, get_us_from_ns(get_ns() - slot_tick_ts),
                l2_slot_info->slotId_3GPP, l2_slot_info->frameId, l2_slot_info->subFrameId, l2_slot_info->slotId
            );

            #ifdef LWPHYCONTROLLER
                if(ptctx->controller == CONTROLLER_LWPHY)
                {
                    if (l2_slot_info->fh_cmd.post_callback != nullptr) {
                        // pt_info("Calling callback immediately  after receiving fh command\n");
                        l2_slot_info->fh_cmd.post_callback(l2_slot_info->fh_cmd.slot_info.slot, l2_slot_info->fh_cmd.channel_params, plctx->pdsch_slot_table_entry[index_slot].pdsch_dl_phy->get_obuf_size());
                    }
                }
            #endif

            rte_ring_enqueue(ptctx->ring_free_dl, (void*)l2_slot_info);
            continue;
        }

        slotId_3GPP = l2_slot_info->slotId_3GPP;
        slot_tick_ts = l2_slot_info->tick;

        if(last_dl_t > 0)
        {
            pt_info("DL pipeline %d | 3GPP slot %d | Delta from last DL %6.2f | Delta from slot tick %6.2fus | DL free elems %d / %d\n",
                plctx->dl_num_processed_slots, slotId_3GPP,
                get_us_from_ns(get_ns() - last_dl_t),
                get_us_from_ns(get_ns() - slot_tick_ts),
                rte_ring_count(ptctx->ring_free_dl), PT_RING_ELEMS
            );
        }
        last_dl_t = get_ns();

        CHECK_FORCE_QUIT_STRING(force_quit, "stopped during rte_ring_dequeue\n")

        #if 0
            if(cleanup_dl_output(index_slot, plctx) != PT_OK)
            {
                pt_err("cleanup_dl_output error on slot %d\n", index_slot);
                return PT_ERR;
            }
        #endif

        pdsch_on=0; pdcch_ul_on=0; pdcch_dl_on=0; pbch_on=0;
        for (auto &map_entry : l2_slot_info->all_cell_params) {
            auto &ch = map_entry.first;
            switch (ch)
            {
                case PHY_PDSCH:
                    if(ptctx->controller == CONTROLLER_LWPHY)
                    {
                        if(lwphy_configure_pdsch_pipeline(plctx->pdsch_slot_table_entry[index_slot].pdsch_dl_phy,
                                            l2_slot_info->all_cell_params[PHY_PDSCH],
                                            l2_slot_info->all_block_params[PHY_PDSCH],
                                            (uintptr_t)l2_slot_info->all_input_addr[PHY_PDSCH],
                                            l2_slot_info->all_input_size[PHY_PDSCH]
                                        )
                        != PT_OK)
                        {
                            pt_err("lwphy_configure_pdsch_pipeline error on slot %d\n", index_slot);
                            return PT_ERR;
                        }
                    } else {
                        //pt_info("Does not configure PDSCH :(\n");
                        if(lwphy_configure_pdsch_pipeline(plctx->pdsch_slot_table_entry[index_slot].pdsch_dl_phy,
                                            {} /* tb_pars */,
                                            l2_slot_info->all_block_params[PHY_PDSCH] /*gnb_pars - not used*/,
                                            (uintptr_t)nullptr, 0)
                        != PT_OK)
                        {
                            pt_err("lwphy_configure_pdsch_pipeline error on slot %d\n", index_slot);
                            return PT_ERR;
                        }
                    }
                    if(start_pdsch_pipeline(index_slot, ptctx, plctx) != PT_OK)
                    {
                        pt_err("start_pdsch_pipeline error on slot %d\n", index_slot);
                        return PT_ERR;
                    }

                    pdsch_on=1;
                    break;
                case PHY_PBCH:
                    if(start_pbch_pipeline(index_slot, l2_slot_info->slotId_3GPP, ptctx, plctx) != PT_OK)
                    {
                        pt_err("start_pbch_pipeline error on slot %d\n", index_slot);
                        return PT_ERR;
                    }
                    pbch_on=1;
                    break;
                case PHY_PDCCH:
                    /* first: DCI_0_0 (if present) */
                    if (std::find(std::begin(l2_slot_info->phy_dci_format_list), std::end(l2_slot_info->phy_dci_format_list), PHY_DCI_0_0) != std::end(l2_slot_info->phy_dci_format_list)) {
                        pt_info("DL pipeline %d | 3GPP slot %d | Delta from slot tick %6.2fus | PDCCH UL\n", plctx->dl_num_processed_slots, l2_slot_info->slotId_3GPP, get_us_from_ns(get_ns() - slot_tick_ts));
                        if(start_pdcch_ul_pipeline(index_slot, l2_slot_info->slotId_3GPP, ptctx, plctx) != PT_OK) {
                            pt_err("start_pdcch_ul_pipeline error on slot %d\n", index_slot);
                            return PT_ERR;
                        }
                        pdcch_ul_on=1;
                    }
                    /* second: DCI_1_1 (if present) */
                    if (std::find(std::begin(l2_slot_info->phy_dci_format_list), std::end(l2_slot_info->phy_dci_format_list), PHY_DCI_1_1) != std::end(l2_slot_info->phy_dci_format_list)) {
                        if(ptctx->controller == CONTROLLER_LWPHY)
                        {
                            /* sanity check: we must have PDSCH in the slot if we're doing PDCCH DL */
                            if (!l2_slot_info->all_cell_params.count(PHY_PDSCH)) {
                                pt_err("Asked to run PDCCH w/ DCI 1_1 but PDSCH is not present in this slot\n");
                                return PT_ERR;
                            }
                        }

                        if (l2_slot_info->all_cell_params.count(PHY_PBCH)) {
                        /* if we also have PBCH in the slot, this is case 301b */
                            if (start_pdcch_dl_301b_pipeline(index_slot, l2_slot_info->slotId_3GPP, ptctx, plctx) != PT_OK) {
                                pt_err("start_pdcch_dl_301b_pipeline error on slot %d\n", index_slot);
                                return PT_ERR;
                            }
                        } else if (std::find(std::begin(l2_slot_info->phy_dci_format_list), std::end(l2_slot_info->phy_dci_format_list), PHY_DCI_0_0) != std::end(l2_slot_info->phy_dci_format_list)) {
                        /* else if we also have PDCCH 0_0, this is case 301a */
                            if (start_pdcch_dl_301a_pipeline(index_slot, l2_slot_info->slotId_3GPP, ptctx, plctx) != PT_OK) {
                                pt_err("start_pdcch_dl_301a_pipeline error on slot %d\n", index_slot);
                                return PT_ERR;
                            }
                        } else {
                        /* else this is case 301 */
                            if (start_pdcch_dl_301_pipeline(index_slot, l2_slot_info->slotId_3GPP, ptctx, plctx) != PT_OK) {
                                pt_err("start_pdcch_dl_301_pipeline error on slot %d\n", index_slot);
                                return PT_ERR;
                            }
                        }
                        pdcch_dl_on=1;
                    }
                    break;
                default:
                    pt_err("Invalid channel type in downlink: %d\n", ch);
                    break;
            }
        }

        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        //// Wait GPU work completion
        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        #if 1
            LW_LWDA_CHECK(lwdaStreamSynchronize(plctx->stream_dl));
            enable_pdsch=0; enable_pddch_ul=0; enable_pdcch_dl_301=0; enable_pdcch_dl_301a=0; enable_pdcch_dl_301b=0; enable_pbch=0;
        #else
            wait_pdsch_pipeline(index_slot, plctx);
            wait_pbch_pipeline(index_slot, plctx);
            wait_pdcch_ul_pipeline(index_slot, plctx);
            wait_pdcch_dl_301_pipeline(index_slot, plctx);
            wait_pdcch_dl_301a_pipeline(index_slot, plctx);
            wait_pdcch_dl_301b_pipeline(index_slot, plctx);
        #endif

        #ifdef LWPHYCONTROLLER
            if(ptctx->controller == CONTROLLER_LWPHY)
            {
                if (l2_slot_info->fh_cmd.post_callback != nullptr) {
                    pt_info("DL pipeline %d | 3GPP slot %d | controller callback delta from slot tick %6.2fus\n",
                        plctx->dl_num_processed_slots, l2_slot_info->slotId_3GPP,
                        get_us_from_ns(get_ns() - slot_tick_ts)
                    );

                    l2_slot_info->fh_cmd.post_callback(l2_slot_info->fh_cmd.slot_info.slot, l2_slot_info->fh_cmd.channel_params,
                                                plctx->pdsch_slot_table_entry[index_slot].pdsch_dl_phy->get_obuf_size()
                                            );
                }
            }
        #endif

        if(plctx->dump_dl_output)
        {
            snprintf(jname, sizeof(jname), "dl_output_%d_frame%d_subframe%d_slot%d.json",
                                index_slot, l2_slot_info->frameId, l2_slot_info->subFrameId, l2_slot_info->slotId);
            std::string json_file(jname);

            if(pt_dump_slot_buffer(json_file,
                                    plctx->pdsch_slot_table_entry[index_slot].pdsch_dl_phy->get_obuf_addr<uint8_t*>(),
                                    plctx->pdsch_slot_table_entry[index_slot].pdsch_dl_phy->get_otensor_size(),
                                    PRB_SIZE_16F,
                                    plctx) != PT_OK)
            {
                pt_err("pt_dump_slot_buffer error\n");
                goto err;
            }
        }

        pt_info("DL pipeline %d | 3GPP slot %d | channel processing time %6.2fus | PBCH %d, PDSCH %d, PDCCH UL %d, PDCCH DL %d\n",
                    plctx->dl_num_processed_slots, l2_slot_info->slotId_3GPP,
                    get_us_from_ns(get_ns() - last_dl_t),
                    pbch_on, pdsch_on, pdcch_ul_on, pdcch_dl_on
                );

        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        //// Prepare info TX core
        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        plctx->dl_tx_list[index_tx_list].l2_slot_info = l2_slot_info;
        plctx->dl_tx_list[index_tx_list].tx_buf = (void*) plctx->pdsch_slot_table_entry[index_slot].pdsch_dl_phy->get_obuf_addr<uint8_t*>();
        plctx->dl_tx_list[index_tx_list].tx_size = (size_t)plctx->pdsch_slot_table_entry[index_slot].pdsch_dl_phy->get_otensor_size();
        plctx->dl_tx_list[index_tx_list].tx_c_time = slot_tick_ts + ((plctx->tti * 1000) * (plctx->sync_tx_tti-1));
        plctx->dl_tx_list[index_tx_list].tx_u_time = slot_tick_ts + ((plctx->tti * 1000) * (plctx->sync_tx_tti));
        plctx->dl_tx_list[index_tx_list].slot_index = index_slot;

        pt_info("DL pipeline %d | 3GPP slot %d | Slot tick: %" PRIu64 ", C-plane TS: %" PRIu64 " (diff is %6.2fus), U-plane TS: %" PRIu64 "(diff is %6.2fus)\n",
            plctx->dl_num_processed_slots, l2_slot_info->slotId_3GPP, slot_tick_ts,
            plctx->dl_tx_list[index_tx_list].tx_c_time, get_us_from_ns(plctx->dl_tx_list[index_tx_list].tx_c_time - slot_tick_ts),
            plctx->dl_tx_list[index_tx_list].tx_u_time, get_us_from_ns(plctx->dl_tx_list[index_tx_list].tx_u_time - slot_tick_ts)
        );

        while(check_force_quit(force_quit) == false && rte_ring_enqueue(plctx->ring_tx_dl, (void*)&(plctx->dl_tx_list[index_tx_list])) != 0);
        CHECK_FORCE_QUIT_STRING(force_quit, "stopped during rte_ring_enqueue ring_free_dl\n")

        index_tx_list = (index_tx_list+1)%PT_MAX_SLOT_ID;
        index_slot = (index_slot+1)%PT_MAX_SLOT_ID;

    }

err:
    rte_wmb();
    LW_LWDA_CHECK(lwdaDeviceSynchronize());
    pt_warn("EXIT\n");
    return PT_OK;
}

int downlink_tx_core(void *param)
{
    SET_THREAD_NAME(__FUNCTION__);
    int index_slot=0, ret=0, i=0, j=0, created_pkts=0;
    int startPrbc=0, startSym=0, numSym=ORAN_ALL_SYMBOLS, tot_pkts_x_symb=0, tot_pkts_x_slot=0;
    uint8_t ecpriSeqid_c=0, frameId=0, subFrameId=0, slotId=0, sym_idx=0, slotId_3GPP=0;
    uint16_t nb_tx = 0;
    uint32_t numPrbc=ORAN_RB_ALL;

    std::vector<uint16_t> tv_flow_count;
    uint16_t* ecpriSeqid_u;
    uint64_t endc_t, endu_t, slot_tx_t, slot_tick_ts, last_dl_t=0;

    Slot * txPdschSlot;
    struct dl_tx_info * dl_tx_entry;

    rte_unique_mbufs mbufs_c{nullptr, &rte_free};
    rte_unique_mbufs mbufs_hdr{nullptr, &rte_free};
    rte_unique_mbufs mbufs_ext{nullptr, &rte_free};

    struct core_info * cinfo = (struct core_info *) param;
    int port_id                 = cinfo->port_id;
    struct phytools_ctx * ptctx = cinfo->ptctx;
    struct pipeline_ctx * plctx = cinfo->plctx;
    int pipeline_index          = plctx->index;

    timerdiff_us(0,0); // initialize timer_hz inside

    size_t shinfo_arr_sz = sizeof(struct rte_mbuf_ext_shared_info) * 1024;
    struct rte_mbuf_ext_shared_info *shinfo = (struct rte_mbuf_ext_shared_info *)rte_zmalloc(NULL, shinfo_arr_sz, 64);
    if (shinfo == NULL)
        do_throw(sb() << "Could not allocate an array for shinfo.");

    mbufs_c.reset(dpdk_allocate_mbufs(plctx->flow_tot * PT_PKT_X_CMSG));
    if(mbufs_c.get() == NULL)
    {
        pt_err("rte_zmalloc mbufs_c error\n");
        return PT_ERR;
    }

    //USE MAX PRBS possible
    mbufs_hdr.reset(dpdk_allocate_mbufs(1024)); //max is 504 x symbol now
    if(mbufs_hdr.get() == NULL)
    {
        pt_err("rte_zmalloc mbufs_hdr error\n");
        return PT_ERR;
    }

    mbufs_ext.reset(dpdk_allocate_mbufs(1024)); //max is 504 x symbol now
    if(mbufs_ext.get() == NULL)
    {
        pt_err("rte_zmalloc mbufs_ext error\n");
        return PT_ERR;
    }

    ecpriSeqid_u = (uint16_t*) calloc(plctx->flow_tot, sizeof(uint16_t));
    if(ecpriSeqid_u == NULL)
    {
        pt_err("ecpriSeqid_u is NULL\n");
        return PT_ERR;
    }

    //mbuf_ext callback
    auto ext_cb = [](void *addr, void *opaque) {
            struct rte_mbuf_ext_shared_info *shinfo = (struct rte_mbuf_ext_shared_info *)opaque;
            // int16_t refcnt = rte_atomic16_sub_return(&shinfo->refcnt_atomic, 1);
            // if (refcnt == 0) {
            //     rte_free(shinfo->fcb_opaque);
            // }
        };

    pt_info("Pipeline %d TX queue clock %d TX queue U %d TX queue C %d on lcore %u port id %d socket %d Sched. Priority %d\n",
                pipeline_index, plctx->dpdkctx.ts_txq, plctx->dpdkctx.dl_txq, plctx->dpdkctx.c_txq,
                rte_lcore_index(rte_lcore_id()), port_id, rte_socket_id(), pt_get_thread_priority()
            );

    while(check_force_quit(force_quit) == false)
    {
        dl_tx_entry=NULL;

        //Setup default values
        startPrbc=0; startSym=0; numSym=ORAN_ALL_SYMBOLS; numPrbc = ORAN_RB_ALL;

        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // New DL slot entry
        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        while(check_force_quit(force_quit) == false && rte_ring_dequeue(plctx->ring_tx_dl, (void **)&dl_tx_entry) != 0);
        CHECK_FORCE_QUIT_STRING(force_quit, "stopped during rte_ring_dequeue\n")

        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Get info
        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        slot_tick_ts = dl_tx_entry->l2_slot_info->tick;
        index_slot = dl_tx_entry->slot_index;
        txPdschSlot = &(plctx->pdsch_slot_table_entry[index_slot].slot_dims);
        frameId = (uint8_t)dl_tx_entry->l2_slot_info->frameId;
        subFrameId = (uint8_t)dl_tx_entry->l2_slot_info->subFrameId;
        slotId = (uint8_t)dl_tx_entry->l2_slot_info->slotId;
        slotId_3GPP = (uint8_t)dl_tx_entry->l2_slot_info->slotId_3GPP;

        if(last_dl_t > 0)
        {
            pt_info("DL pipeline %d | 3GPP slot %d | Delta from last DL %6.2f | Delta from slot tick %6.2fus\n",
                    index_slot, slotId_3GPP,
                    get_us_from_ns(get_ns() - last_dl_t),
                    get_us_from_ns(get_ns() - slot_tick_ts)
            );
        }
        last_dl_t = get_ns();

        ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        ///// Packets setup
        ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        tot_pkts_x_slot = txPdschSlot->pkts_per_slot;
        if(plctx->first_ap_only) tot_pkts_x_slot = (txPdschSlot->pkts_per_flow[0][0] + (plctx->flow_tot-1)) * SLOT_NUM_SYMS;

        shinfo[tot_pkts_x_slot].fcb_opaque = (void *)shinfo;
        rte_atomic16_set(&shinfo[tot_pkts_x_slot].refcnt_atomic, tot_pkts_x_slot);

        if(dpdk_pull_pkts(mbufs_hdr, plctx->dpdkctx.tx_dpdk_mempool[0], tot_pkts_x_slot) != PT_OK)
        {
            set_force_quit(force_quit);
            return PT_ERR;
        }

        if(dpdk_pull_pkts(mbufs_ext, plctx->dpdkctx.tx_dpdk_mempool[1], tot_pkts_x_slot) != PT_OK)
        {
            set_force_quit(force_quit);
            return PT_ERR;
        }

        for (auto &map_entry : dl_tx_entry->l2_slot_info->all_cell_params) {
            auto &ch = map_entry.first;
            switch (ch)
            {
                case PHY_PDSCH:
                    startPrbc = (uint8_t)dl_tx_entry->l2_slot_info->all_startPrb[PHY_PDSCH];
                    numPrbc = (uint16_t)dl_tx_entry->l2_slot_info->all_numPrb[PHY_PDSCH];
                    startSym = (uint8_t)dl_tx_entry->l2_slot_info->all_startSym[PHY_PDSCH];
                    numSym = (uint8_t)dl_tx_entry->l2_slot_info->all_numSym[PHY_PDSCH];
                    // pt_info("PDSCH startPrbc=%d, numPrbc=%d, startSym=%d, numSym=%d\n", startPrbc, numPrbc, startSym, numSym);
                    break;
                case PHY_PBCH:
                    startPrbc = (uint8_t)dl_tx_entry->l2_slot_info->all_startPrb[PHY_PBCH];
                    numPrbc = (uint16_t)dl_tx_entry->l2_slot_info->all_numPrb[PHY_PBCH];
                    startSym = (uint8_t)dl_tx_entry->l2_slot_info->all_startSym[PHY_PBCH];
                    numSym = (uint8_t)dl_tx_entry->l2_slot_info->all_numSym[PHY_PBCH];
                    // pt_info("PBCH startPrbc=%d, numPrbc=%d, startSym=%d, numSym=%d\n", startPrbc, numPrbc, startSym, numSym);
                    break;
                case PHY_PDCCH:
                    startPrbc = (uint8_t)dl_tx_entry->l2_slot_info->all_startPrb[PHY_PDCCH];
                    numPrbc = (uint16_t)dl_tx_entry->l2_slot_info->all_numPrb[PHY_PDCCH];
                    startSym = (uint8_t)dl_tx_entry->l2_slot_info->all_startSym[PHY_PDCCH];
                    numSym = (uint8_t)dl_tx_entry->l2_slot_info->all_numSym[PHY_PDCCH];
                    // pt_info("PDCCH startPrbc=%d, numPrbc=%d, startSym=%d, numSym=%d\n", startPrbc, numPrbc, startSym, numSym);
                    break;
                default:
                    pt_err("Invalid channel type in downlink: %d\n", ch);
                    break;
            }
        }

        if(numPrbc == ORAN_MAX_PRB_X_SLOT)
            numPrbc = ORAN_RB_ALL;
        else if(numPrbc > ORAN_MAX_PRB_X_SECTION)
        {
            pt_err("Asking for %d PRBs. Multi-section is not supported yet\n", numPrbc);
            set_force_quit(force_quit);
            goto err;
        }

        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Pull mbufs for C/U-plane
        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        if(dpdk_pull_pkts(mbufs_c, plctx->dpdkctx.c_mempool, plctx->flow_tot * PT_PKT_X_CMSG) != PT_OK)
        {
            set_force_quit(force_quit);
            return PT_ERR;
        }

        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Prepare C-plane message
        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        if(pt_prepare_cplane_messages(mbufs_c, plctx->flow_tot * PT_PKT_X_CMSG,
                                        (plctx->first_ap_only == 1 ? txPdschSlot->prbs_per_pkt[0][0] : 0),
                                        ecpriSeqid_c, frameId, subFrameId, slotId,
                                        startPrbc, numPrbc, numSym, startSym,
                                        DIRECTION_DOWNLINK,
                                        ptctx->dpdk_dev[plctx->dpdkctx.port_id].eth_addr,
                                        plctx
                            ) != PT_OK)
        {
            pt_err("pt_prepare_cplane_messages error\n");
            set_force_quit(force_quit);
            return PT_ERR;
        }
        ecpriSeqid_c = ((ecpriSeqid_c+1)%256);

        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Send C-plane message
        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        //Wait right time
        endc_t = dl_tx_entry->tx_c_time;
        while (get_ns() < endc_t)
        {
            for (int spin_cnt = 0; spin_cnt < 1000; ++spin_cnt)
                __asm__ __volatile__ ("");
        }

        pt_info("DL pipeline %d | 3GPP slot %d | Delta from slot tick %6.2fus | C-plane | ORAN HDR: frameId=%d, subFrameId=%d, slotId=%d startPrb=%d, numPrb=%d startSym=%d numSym=%d\n",
                index_slot, slotId_3GPP,
                get_us_from_ns(get_ns() - slot_tick_ts),
                frameId, subFrameId, slotId, startPrbc, numPrbc, startSym, numSym
            );

        PUSH_RANGE("TX_ACK", LWLWTX_COL_3);
        dpdk_tx_burst_pkts(mbufs_c, port_id, plctx->dpdkctx.c_txq, (plctx->flow_tot * PT_PKT_X_CMSG), ptctx->flush_tx_write, force_quit);
        POP_RANGE;

        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        /// Prepare U-plane
        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        uint64_t slot_start_t = get_ns();
        try {
            created_pkts=0;
            for (sym_idx = 0; sym_idx < SLOT_NUM_SYMS && check_force_quit(force_quit) == false; ++sym_idx)
            {
                i=0;
                // uint64_t t_start_prep = get_ns();
                for (uint8_t flow_idx = 0; flow_idx < plctx->flow_tot; ++flow_idx)
                {
                    uint32_t remaining_prbs = txPdschSlot->prbs_per_sym[sym_idx] / plctx->flow_tot;
                    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                    /// TX New symbol
                    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                    uint16_t prb_idx = 0;
                    uint16_t pkts_per_flow = txPdschSlot->pkts_per_flow[flow_idx][sym_idx];
                    uint16_t prbs_per_pkt = txPdschSlot->prbs_per_pkt[flow_idx][sym_idx];
                    for (;
                            i < (flow_idx + 1) * pkts_per_flow  &&
                            remaining_prbs > 0                                                  &&
                            created_pkts < tot_pkts_x_slot                                      &&
                            check_force_quit(force_quit) == false;
                        ++i, created_pkts++)
                    {
                        // struct rte_mbuf *hdr = mbufs_tx_hdr[created_pkts];
                        struct rte_mbuf *hdr = (mbufs_hdr.get())[created_pkts];
                        // struct rte_mbuf *ext = mbufs_tx_ext[created_pkts];
                        struct rte_mbuf *ext = (mbufs_ext.get())[created_pkts];

                        /* attach external buffer */
                        uint16_t num_prbs = RTE_MIN(remaining_prbs, prbs_per_pkt);
                        uint16_t buf_len =  num_prbs * PRB_SIZE_16F;

                        void *ext_ptr = txPdschSlot->ptrs[flow_idx][sym_idx][prb_idx];
                        //Always assign the last one to decrement counter
                        shinfo[created_pkts].fcb_opaque = (void *)&shinfo[tot_pkts_x_slot];
                        shinfo[created_pkts].free_cb = ext_cb;
                        rte_atomic16_set(&shinfo[created_pkts].refcnt_atomic, 1);
                        rte_pktmbuf_attach_extbuf(ext, ext_ptr, RTE_BAD_IOVA, buf_len, &shinfo[created_pkts]);

                        /* setup the headers */
                        rte_memcpy(rte_pktmbuf_mtod(hdr, void *), plctx->oran_uplane_hdr.c_str(), ORAN_IQ_HDR_SZ);
                        struct oran_ecpri_hdr *ecpri =
                            rte_pktmbuf_mtod_offset(hdr, struct oran_ecpri_hdr *,
                                                            (sizeof(struct rte_ether_hdr)
                                                            + sizeof(struct rte_vlan_hdr)));
                        ecpri->ecpriSeqid       = ecpriSeqid_u[flow_idx];
                        ecpriSeqid_u[flow_idx]  = (ecpriSeqid_u[flow_idx]+1)%256;
                        //Mellanox parser in Network order
                        ecpri->ecpriPcid        = rte_cpu_to_be_16(plctx->flow_list[flow_idx]);

                        struct oran_umsg_iq_hdr *iq_df =
                            rte_pktmbuf_mtod_offset(hdr, struct oran_umsg_iq_hdr *,
                                                            (sizeof(struct rte_ether_hdr)
                                                            + sizeof(struct rte_vlan_hdr)
                                                            + sizeof(struct oran_ecpri_hdr)));
                        iq_df->frameId          = frameId;
                        iq_df->subframeId       = subFrameId;
                        iq_df->slotId           = slotId;
                        iq_df->symbolId         = (uint8_t)sym_idx; //not working properly

                        struct oran_u_section_uncompressed *u_sec =
                            rte_pktmbuf_mtod_offset(hdr, struct oran_u_section_uncompressed *,
                                                            (sizeof(struct rte_ether_hdr)
                                                            + sizeof(struct rte_vlan_hdr)
                                                            + sizeof(struct oran_ecpri_hdr)
                                                            + sizeof(struct oran_umsg_iq_hdr)));

                        u_sec->sectionId        = ORAN_DEF_SECTION_ID;
                        u_sec->rb               = 0;
                        u_sec->symInc           = 0;
                        u_sec->startPrbu        = prb_idx;
                        u_sec->numPrbu          = num_prbs;

                        hdr->data_len           = ORAN_IQ_HDR_SZ;
                        hdr->pkt_len            = hdr->data_len;
                        hdr->nb_segs           = 1;

                        ext->ol_flags           = EXT_ATTACHED_MBUF;
                        ext->data_len           = buf_len;
                        ext->pkt_len            = ext->data_len;

                        if(rte_pktmbuf_chain(hdr, ext))
                            do_throw(sb() << "rte_pktmbuf_chain error");

                        ecpri->ecpriPayload     = rte_cpu_to_be_16(hdr->pkt_len-ORAN_IQ_HDR_OFFSET+4); //Temp fix: Need to rework ORAN structs

                        /* update counters */
                        remaining_prbs -= num_prbs;
                        prb_idx += num_prbs;

                        rte_mbuf_sanity_check(hdr, 1);
                        // rte_pktmbuf_dump(stdout, hdr, 128);

                        if(plctx->first_ap_only && flow_idx > 0 && prb_idx > 0)
                        {
                            i = (flow_idx + 1) * txPdschSlot->pkts_per_flow[flow_idx][sym_idx];
                            break;
                        }
                    }
                }
                // pt_info("Prepare Symbol %d pkts %d took %ld ns\n", sym_idx, created_pkts, get_ns() - t_start_prep);
            }
        } catch (std::runtime_error &e) {
            do_throw(sb() << e.what() << " Slot counter: " << dl_tx_entry->slot_index << ".");
        }

        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        /// Send U-plane symbol by symbol
        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        try {
            struct rte_mbuf ** mbufs_tx = mbufs_hdr.get();
            //Wait right time
            endu_t = dl_tx_entry->tx_u_time;
            while (get_ns() < endu_t)
            {
                for (int spin_cnt = 0; spin_cnt < 1000; ++spin_cnt)
                    __asm__ __volatile__ ("");
            }

            pt_info("DL pipeline %d | 3GPP slot %d | Delta from slot tick %6.2fus | Sending U-plane frame %d, subFrame %d, slot %d, num PRBs %d start Symbol %d num Symbols %d\n",
                index_slot, slotId_3GPP, get_us_from_ns(get_ns() - slot_tick_ts),
                frameId, subFrameId, slotId, numPrbc, startSym, numSym
            );

            slot_tx_t = get_ns();
            for (sym_idx = 0;
                    sym_idx < SLOT_NUM_SYMS                     &&
                    tot_pkts_x_symb * sym_idx < created_pkts    &&
                    check_force_quit(force_quit) == false;
                ++sym_idx)
            {
                uint64_t tx_start_t = get_ns();
                tot_pkts_x_symb = txPdschSlot->pkts_per_sym[sym_idx];

                if(plctx->first_ap_only)
                    tot_pkts_x_symb = txPdschSlot->pkts_per_flow[0][sym_idx] + (plctx->flow_tot-1);

                nb_tx = 0;
                while (nb_tx < tot_pkts_x_symb && check_force_quit(force_quit) == false) {
                    nb_tx += rte_eth_tx_burst(port_id, plctx->dpdkctx.dl_txq, mbufs_tx + (tot_pkts_x_symb * sym_idx) + nb_tx, tot_pkts_x_symb-nb_tx);
                }
                CHECK_FORCE_QUIT_STRING(force_quit, "stopped during rte_eth_tx_burst\n")
                if(ptctx->flush_tx_write) rte_wmb();

                wait_ns(CPU_SYM_US);
            }
        } catch (std::runtime_error &e) {
            do_throw(sb() << e.what() << " Slot counter: " << plctx->downlink_slots << ".");
        }

        pt_info("Closing DL pipeline %d | 3GPP slot %d | Delta from slot tick %6.2fus | U-plane TX time %6.2fus | Tot slot time %6.2fus\n",
                index_slot, dl_tx_entry->l2_slot_info->slotId_3GPP, get_us_from_ns(get_ns() - slot_tick_ts),
                get_us_from_ns(get_ns() - slot_tx_t), get_us_from_ns(get_ns() - last_dl_t)

            );

        while(check_force_quit(force_quit) == false && rte_ring_enqueue(ptctx->ring_free_dl, (void*)dl_tx_entry->l2_slot_info) != 0);
        CHECK_FORCE_QUIT_STRING(force_quit, "stopped during rte_ring_enqueue ring_free_dl\n")

        plctx->dl_num_processed_slots++;

        if(ptctx->controller != CONTROLLER_LWPHY && plctx->downlink_slots > 0 && plctx->dl_num_processed_slots >= plctx->downlink_slots)
        {
            pt_info("Max DL slots %d reached. Exit..\n", plctx->downlink_slots);
            break;
        }
    }

err:
    rte_wmb();
    LW_LWDA_CHECK(lwdaDeviceSynchronize());
    pt_warn("EXIT\n");
    return PT_OK;
}
