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

int check_right_3gpp_slot(int lwrrent_slot, int advance_slots, int max_3gpp_slots) {
    return (lwrrent_slot+advance_slots)%max_3gpp_slots;
}

int get_next_3gpp_slot(int lwrrent_slot, int max_3gpp_slots) {
    return (lwrrent_slot+1)%max_3gpp_slots;
}

void setup_dl_channel(struct l2_control_slot_info* l2_slot_info, enum phy_channel_type phy_ch, gnb_pars& gnb_params) {
    l2_slot_info->all_startPrb[phy_ch] = 0;
    l2_slot_info->all_numPrb[phy_ch] = ORAN_RB_ALL;
    l2_slot_info->all_startSym[phy_ch] = 0;
    l2_slot_info->all_numSym[phy_ch] = ORAN_ALL_SYMBOLS;
    l2_slot_info->all_cell_params[phy_ch] = gnb_params;
    return;
}

int controller_core(void *param) {
    SET_THREAD_NAME(__FUNCTION__);
    struct phytools_ctx * ptctx = (struct phytools_ctx *)param;
    struct pipeline_ctx * plctx = &ptctx->plctx[0];
    int cnt_slots=0, cnt_ul=0, cnt_dl=0, cnt_pbch=0, slot_3gpp=plctx->slot_num_max_3gpp-plctx->sync_tx_tti; //to consider slot 0
    //ORAN time info are 1 slot after the 3GPP slot numeration
    uint8_t frameId=0, subFrameId=0, slotId=0;
    int pbch_index=0, pusch_index=0, pdsch_index=0, pdcch_ul_index=0, pdcch_dl_index=0;
    int set_pbch=0, set_pdsch=0, set_pdcch_ul=0, set_pdcch_dl=0;
    struct l2_control_slot_info * l2_slot_info = NULL;
    uint64_t last_ul_t=0, last_dl_t=0;
    gnb_pars fake_gnb_params;
    int pipeline_index          = plctx->index;

    pt_info("Pipeline %d Controller core on lcore %u Sched. priority %d\n", pipeline_index, rte_lcore_index(rte_lcore_id()), pt_get_thread_priority());

    if(!plctx->uplink) cnt_ul = plctx->uplink_slots;
    if(!plctx->downlink) cnt_dl = plctx->downlink_slots;

    while(check_force_quit(force_quit) == false)
    {
        if( ((plctx->uplink && cnt_ul >= plctx->uplink_slots && plctx->uplink_slots > 0) || !plctx->uplink) ||
            ((plctx->downlink && cnt_dl >= plctx->downlink_slots && plctx->downlink_slots > 0) || !plctx->downlink))
        {
            break;
        }
        wait_ns(plctx->tti * NS_X_US);
        std::atomic<uint64_t>atom_ts{(uint64_t)get_ns()};

        set_pbch=0, set_pdsch=0, set_pdcch_ul=0, set_pdcch_dl=0;

        if(plctx->downlink && (cnt_dl < plctx->downlink_slots || plctx->downlink_slots <= 0))
        {
            while(check_force_quit(force_quit) == false && rte_ring_dequeue(ptctx->ring_free_dl, (void **)&l2_slot_info) != 0);
            CHECK_FORCE_QUIT_STRING(force_quit, "controller_core stopped during rte_ring_dequeue\n")
            if(l2_slot_info == NULL)
            {
                pt_err("l2_slot_info is an invalid pointer\n");
                exit(EXIT_FAILURE);
            }
            l2_slot_info->reset();

            //at least 1 pbch every frame (should not be a problem)
            if(check_right_3gpp_slot(slot_3gpp, plctx->sync_tx_tti, plctx->slot_num_max_3gpp) == plctx->pbch_slot_list[pbch_index])
            {
                set_pbch=1;
                setup_dl_channel(l2_slot_info, PHY_PBCH, fake_gnb_params);
                pbch_index++;
                if(plctx->pbch_slot_list[pbch_index] == -1)
                    pbch_index=0;
                cnt_pbch++;
            }

            if(check_right_3gpp_slot(slot_3gpp, plctx->sync_tx_tti, plctx->slot_num_max_3gpp) == plctx->pdsch_slot_list[pdsch_index])
            {
                set_pdsch=1;
                setup_dl_channel(l2_slot_info, PHY_PDSCH, fake_gnb_params);
                pdsch_index++;
                if(plctx->pdsch_slot_list[pdsch_index] == -1)
                    pdsch_index=0;
            }
            if(check_right_3gpp_slot(slot_3gpp, plctx->sync_tx_tti, plctx->slot_num_max_3gpp) == plctx->pdcch_ul_slot_list[pdcch_ul_index])
            {
                set_pdcch_ul=1;
                setup_dl_channel(l2_slot_info, PHY_PDCCH, fake_gnb_params);
                auto phy_dci_format = PHY_DCI_0_0;
                l2_slot_info->phy_dci_format_list.push_back(phy_dci_format);
                pdcch_ul_index++;
                if(plctx->pdcch_ul_slot_list[pdcch_ul_index] == -1)
                    pdcch_ul_index=0;
            }
            if(check_right_3gpp_slot(slot_3gpp, plctx->sync_tx_tti, plctx->slot_num_max_3gpp) == plctx->pdcch_dl_slot_list[pdcch_dl_index])
            {
                set_pdcch_dl=1;
                setup_dl_channel(l2_slot_info, PHY_PDCCH, fake_gnb_params);
                auto phy_dci_format = PHY_DCI_1_1;
                l2_slot_info->phy_dci_format_list.push_back(phy_dci_format);
                pdcch_dl_index++;
                if(plctx->pdcch_dl_slot_list[pdcch_dl_index] == -1)
                    pdcch_dl_index=0;
            }

            //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            //// START DL PIPELINE
            //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            if(set_pbch==1 || set_pdsch==1 || set_pdcch_ul==1 || set_pdcch_dl==1)
            {
                //C-plane for the next slot
                l2_slot_info->tick = atom_ts.load();
                l2_slot_info->frameId = frameId;
                l2_slot_info->subFrameId = subFrameId;
                l2_slot_info->slotId = slotId;
                l2_slot_info->slotId_3GPP = (slot_3gpp + plctx->sync_tx_tti)%plctx->slot_num_max_3gpp;
                ptctx->slot_3gpp_ref_ts[(int)l2_slot_info->slotId_3GPP] = atom_ts;

                pt_info("Start DL pipeline %d (pbch %d pdsch %d pdcch ul %d pdcch dl %d) frame %d subframe %d slot %d\n",
                        cnt_slots,
                        set_pbch, set_pdsch, set_pdcch_ul, set_pdcch_dl,
                        frameId, subFrameId, slotId);
                if(last_dl_t > 0)
                        pt_info("From last DL #%d: %6.2f us\n", cnt_dl-1, get_us_from_ns(get_ns() - last_dl_t));

                l2_slot_info->recv_tstamp = get_ns();
                while(check_force_quit(force_quit) == false && rte_ring_enqueue(ptctx->ring_start_dl, (void*)l2_slot_info) != 0);
                CHECK_FORCE_QUIT_STRING(force_quit, "ring_start_dl stopped during rte_ring_enqueue ring_start_dl\n")
                last_dl_t = get_ns();

                cnt_dl++;
            }
            else
            {
                while(check_force_quit(force_quit) == false && rte_ring_enqueue(ptctx->ring_free_dl, (void*)l2_slot_info) != 0);
                CHECK_FORCE_QUIT_STRING(force_quit, "stopped during rte_ring_enqueue ring_free_dl\n")
            }
        }

        if(plctx->uplink && (cnt_ul < plctx->uplink_slots || plctx->uplink_slots <= 0))
        {
            if(plctx->downlink && cnt_pbch < plctx->wait_downlink_slot)
                goto end_loop;

            //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            //// START UL PIPELINE
            //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            // pt_info("next 3gpp slot %d, pusch_slot_list[%d]=%d\n", check_right_3gpp_slot(slot_3gpp, plctx->sync_tx_tti, plctx->slot_num_max_3gpp), pusch_index, plctx->pusch_slot_list[pusch_index]);
            if(check_right_3gpp_slot(slot_3gpp, plctx->sync_tx_tti, plctx->slot_num_max_3gpp) == plctx->pusch_slot_list[pusch_index])
            {
                while(check_force_quit(force_quit) == false && rte_ring_dequeue(ptctx->ring_free_ul, (void **)&l2_slot_info) != 0);
                CHECK_FORCE_QUIT_STRING(force_quit, "controller_core stopped during rte_ring_dequeue\n")

                l2_slot_info->tick = atom_ts.load();
                l2_slot_info->frameId = frameId;
                l2_slot_info->subFrameId = subFrameId;
                l2_slot_info->slotId = slotId;

                l2_slot_info->slotId_3GPP = (slot_3gpp + plctx->sync_tx_tti)%plctx->slot_num_max_3gpp;
                ptctx->slot_3gpp_ref_ts[(int)l2_slot_info->slotId_3GPP] = atom_ts;
                l2_slot_info->startPrb = 0;
                l2_slot_info->numPrb = ORAN_RB_ALL;
                l2_slot_info->startSym = 0;
                l2_slot_info->numSym = ORAN_ALL_SYMBOLS;

                pt_info("Start UL pipeline %d frame %d subframe %d slot %d\n", cnt_ul, frameId, subFrameId, slotId);
                if(last_ul_t > 0)
                    pt_info("From last UL #%d: %6.2f us\n", cnt_ul-1, get_us_from_ns(get_ns() - last_ul_t));

                l2_slot_info->recv_tstamp = get_ns();

                while(check_force_quit(force_quit) == false && rte_ring_enqueue(ptctx->ring_start_ul, (void*)l2_slot_info) != 0);
                CHECK_FORCE_QUIT_STRING(force_quit, "ring_start_ul stopped during rte_ring_enqueue ring_start_ul\n")
                last_ul_t = get_ns();
                cnt_ul++;
                pusch_index++;
                if(plctx->pusch_slot_list[pusch_index] == -1)
                    pusch_index=0;
            }
        }

    end_loop:
        pt_increase_slot(frameId, subFrameId, slotId);
        slot_3gpp = get_next_3gpp_slot(slot_3gpp, plctx->slot_num_max_3gpp);
        cnt_slots++;
    }

    if(plctx->downlink) while(check_force_quit(force_quit) == false && PT_ACCESS_ONCE(plctx->dl_num_processed_slots) < plctx->downlink_slots)    { wait_ns(( (10 * plctx->tti) * 1000)); }
    if(plctx->uplink)   while(check_force_quit(force_quit) == false && PT_ACCESS_ONCE(plctx->ul_num_processed_slots) < plctx->uplink_slots)      { wait_ns(( (10 * plctx->tti) * 1000)); }

    fflush(stdout);
    fflush(stderr);
    set_force_quit(force_quit);
    return PT_OK;

err:
    pt_err("EXIT\n");
    set_force_quit(force_quit);
    return PT_OK;
}
