/*
 * Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 */

#include "altran_fapi_msg.hpp"
#include "altran_fapi_msg_helpers.hpp"
#include "lw_phy_mac_transport.hpp"
#include "altran_defaults.h"

#include <cstring>
#include "lw_altran_phy.hpp"
#include "lw_phy_utils.hpp"

#include <pthread.h>

int ul_number=0;
int dl_number=0;

static uint64_t counter = 0;
static uint64_t ul_drop_fh_cmds = 0;
static uint64_t dl_drop_fh_cmds = 0;
static uint64_t dropped_pdcch_fh= 0;
namespace lw_altran_stack
{
    uint64_t slot_indication_ts=0;

    lw_altran_phy::lw_altran_phy(lw::PHY_module& module, yaml::node config, struct phytools_ctx * _ptctx) :
        altran_fapi::phy(module, config)
    {
        LOGF(INFO, "lw_altran_phy::lw_altran_phy()\n");

        if(config.has_key("timer_thread_config"))
        {
            yaml::node thread_config        = config["timer_thread_config"];
            timer_thread_cfg.cpu_affinity   = thread_config["cpu_affinity"].as<int32_t>();
            timer_thread_cfg.sched_priority = thread_config["sched_priority"].as<int32_t>();
            has_thread_cfg                  = true;
        }

        ptctx = _ptctx;
    }

    bool lw_altran_phy::isEqual(uint8_t* a, uint8_t* b, int len)
    {
        for(int i = 0; i < len; i++)
        {
            if(a[i] != b[i]) return false;
        }
        return true;
    }

    bool lw_altran_phy::on_msg(lw_ipc_msg_t& mesg)
    {
        LOGF(INFO, "lw_altran_phy::on_msg()\n");
        altran_fapi_hdr_t& hdr = mesg.msg_len == 0 && mesg.data_buf != NULL ? *(static_cast<altran_fapi_hdr_t*>(mesg.data_buf)) : *(static_cast<altran_fapi_hdr_t*>(mesg.msg_buf));
        //TbPrms
        //BBUPrms
        LOGF(INFO,"ON_MSG COUNTER=%lu, lwrrent_tick=%lu", counter++, lwrrent_slot_info().tick_);
        bool status = true;
        if(hdr.msgLen < 1)
        {
            LOGF(INFO, "Invalid message length for message id %d\n", hdr.msgId);
            return status;
        }

        LOGF(INFO, "lw_altran_phy message id:  %d, length: %d\n", hdr.msgId, hdr.msgLen);

        switch(hdr.msgId)
        {
        case MAC_PHY_CELL_CONFIG_REQ:
        {
            LOGF(INFO, "lw_altran_phy::MAC_PHY_CELL_CONFIG_REQ \n");
            altran_fapi_config_req_t& msg_body = reinterpret_cast<altran_fapi_config_req_t&>(hdr.msgBody);

            /*if(isEqual(default_config_req_payload, (uint8_t*)(mesg.msg_buf), sizeof(default_config_req_payload)))
            {
                LOGF(INFO, "default_config_req_payload verified\n");
            }
            else
            {
                LOGF(INFO, "default_config_req_payload failed to verify\n");
            }*/
            on_mac_phy_cell_config_req(msg_body);
            send_cell_config_resp();
        }
        break;
        case MAC_PHY_CELL_START_REQ:
        {
            LOGF(INFO, "lw_altran_phy::MAC_PHY_CELL_START_REQ \n");
            altran_fapi_start_req_t& msg_body = reinterpret_cast<altran_fapi_start_req_t&>(hdr.msgBody);
            /*if(isEqual(default_start_req_payload, (uint8_t*)(mesg.msg_buf), sizeof(default_start_req_payload)))
            {
                LOGF(INFO, "default_start_req_payload verified\n");
            }
            else
            {
                LOGF(INFO, "default_start_req_payload failed to verify\n");
            }*/
            on_mac_phy_cell_start_req(msg_body);
            send_cell_start_resp();
            start_slot_indication_thread();
        }
        break;
        case MAC_PHY_CELL_STOP_REQ:
        {
            LOGF(INFO, "lw_altran_phy::MAC_PHY_CELL_STOP_REQ\n");
            altran_fapi_stop_req_t& msg_body = reinterpret_cast<altran_fapi_stop_req_t&>(hdr.msgBody);
            on_mac_phy_cell_stop_req(msg_body);
        }
        break;
        case PHY_DL_CONFIG_REQUEST:
        {
            LOGF(INFO, "lw_altran_phy::PHY_DL_CONFIG_REQUEST\n");
            altran_fapi_dl_config_req_t& msg_body = reinterpret_cast<altran_fapi_dl_config_req_t&>(hdr.msgBody);
            on_phy_dl_config_request(msg_body);
        }
        break;
        case PHY_UL_CONFIG_REQUEST:
        {
            LOGF(INFO, "lw_altran_phy::PHY_UL_CONFIG_REQUEST \n");
            altran_fapi_ul_config_req_t& msg_body = reinterpret_cast<altran_fapi_ul_config_req_t&>(hdr.msgBody);

            /*if(isEqual(default_ul_config_req, (uint8_t*)(mesg.msg_buf), sizeof(default_ul_config_req)))
            {
                LOGF(INFO, "default_ul_config_req verified\n");
            }
            else
            {
                LOGF(INFO, "default_ul_config_req failed to verify\n");
            }*/
            on_phy_ul_config_request(msg_body);
            send_fh_command();
        }
        break;
        case PHY_UL_DCI_REQUEST:
        {
            LOGF(INFO, "lw_altran_phy::PHY_UL_DCI_REQUEST\n");
            altran_fapi_ul_dci_req_t& msg_body = reinterpret_cast<altran_fapi_ul_dci_req_t&>(hdr.msgBody);
            on_phy_ul_dci_request(msg_body);
	        if (!is_valid_dci_rx_slot()) {
                    if (send_fh_command() == -1) {
                        dropped_pdcch_fh++;
                        LOGF(INFO, "Dropping PDCCH command %lu", dropped_pdcch_fh);
                    }
                } else {
                    LOGF(INFO, "NO FH for PHY_UL_DCI_REQUEST");
	        }
        }
        break;
        case PHY_DL_TX_REQUEST:
        {
            LOGF(INFO, "lw_altran_phy::PHY_DL_TX_REQUEST\n");
            altran_fapi_phy_data_tx_req_t& msg_body = reinterpret_cast<altran_fapi_phy_data_tx_req_t&>(hdr.msgBody);
            if (on_phy_dl_tx_request(msg_body, mesg)) {
                fh::fh_command& fh_cmd = *(altran_fapi::phy::get_lwrrent_fh_command());
                if (containsChannel(fh_cmd, fh::channel_type::PDSCH)) {
                    status = false;
                    LOGF(INFO, "PDSCH is present");
                } else {
                    status = true;
                    LOGF(INFO, "PDSCH is not present");
                }
                if(send_fh_command() == -1) {
	            status = true;
	        }  
	    }
        }
        break;
        default:
            on_unknown_message_id(hdr);
            break;
        }
        return status;
    }

    void lw_altran_phy::start_slot_indication_thread()
    {
        if(!started)
        {
    #ifdef SLOT_INDICATION_POLLING
            std::thread t(&lw_altran_phy::slot_indication_thread_poll_method, this);
    #else
            std::thread t(&lw_altran_phy::slot_indication_thread_timer_fd_method, this);
    #endif
            timer_thread.swap(t);

            if(has_thread_cfg)
            {
                sched_param sch;
                int         policy;
                int         status = 0;
                //-  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
                // Set thread priority
                status = pthread_getschedparam(timer_thread.native_handle(), &policy, &sch);
                if(status != 0)
                {
                    LOG(WARNING) << "timer_thread pthread_getschedparam failed with status : " << std::strerror(status) << '\n';
                }
                sch.sched_priority = timer_thread_cfg.sched_priority;

                status = pthread_setschedparam(timer_thread.native_handle(), SCHED_FIFO, &sch);
                if(status != 0)
                {
                    LOG(WARNING) << "timer_thread setschedparam failed with status : " << std::strerror(status) << '\n';
                }

                //-  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
                // Set thread CPU affinity
                cpu_set_t cpuset;
                CPU_ZERO(&cpuset);
                CPU_SET(timer_thread_cfg.cpu_affinity, &cpuset);
                status = pthread_setaffinity_np(timer_thread.native_handle(), sizeof(cpu_set_t), &cpuset);
                if(status)
                {
                    LOG(WARNING) << "timer_thread setaffinity_np  failed with status : " << std::strerror(status) << '\n';
                }
            }
            lw::slot_indication& slot_info = lwrrent_slot_info();
            // LWPU - Advance slot
            // slot_info.slot_ += ADVANCE_SLOTS;
            // slot_info.tick_ = ADVANCE_SLOTS;

            //ASSUMING ONLY 1 PIPELINE
            LOGF(INFO,"Advance slot indications by %d slots", ptctx->plctx[0].sync_tx_tti);
            slot_info.slot_ += ptctx->plctx[0].sync_tx_tti;
            slot_info.tick_ = ptctx->plctx[0].sync_tx_tti;
            started = true;
        }
    }

    inline uint64_t lw_altran_phy::sys_clock_time_handler()
    {
        using namespace std::chrono;
        return (uint64_t)duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();
    }

    void lw_altran_phy::slot_indication_thread_poll_method()
    {
    #ifdef dbg
        if(has_thread_cfg)
        {
            sched_param sch;
            int         policy;
            int         status = 0;
            status             = pthread_getschedparam(pthread_self(), &policy, &sch);
            if(status == 0)
            {
                LOG(DEBUG) << "slot_indication_thread_poll_method sched_priority " << sch.sched_priority << '\n';
                LOG(DEBUG) << "slot_indication_thread_poll_method on CPU " << sched_getcpu() << "\n";
            }
            else
            {
                LOG(DEBUG) << "pthread_getschedparam failed with status: " << std::strerror(status) << "\n";
            }
        }
    #endif

        bool     first = true;
        uint64_t last_actual, last_expected, first_time;
        uint64_t runtime;
        uint64_t count          = 0;
        uint64_t sum_abs_offset = 0;
        int32_t  min_offset     = 0;
        int32_t  max_offset     = 0;
        int32_t  max_abs_offset = 0;

        lw::phy_config* config = phy_config();
        window_nsec            = lw::mu_to_ns(config->ssb_config_.sub_c_common);
	    //window_nsec            = PT_SLOT_INDICATION_NS;


        while(1)
        {
            if(check_force_quit(force_quit)) {
                LOGF(INFO, "Force quit in timer thread");
                break;
            }
            uint64_t lwrr = sys_clock_time_handler();
            if(first)
            {
                LOG(DEBUG) << "start time " << lwrr << '\n';
                first_time    = lwrr;
                last_expected = lwrr;
                last_actual   = lwrr;
                first         = false;
                continue;
            }

            if(lwrr < last_actual)
            {
                LOG(WARNING) << "error lwrr " << lwrr << " last " << last_actual << '\n';
            }

            uint64_t diff = lwrr - last_expected;
            if(diff < window_nsec - pre_window)
            {
                continue;
            }

            slot_indication_ts = lwrr;
            rte_mb();
            slot_indication_handler();

            uint64_t new_expected = last_expected + window_nsec;
            int32_t  offset       = (int32_t)((int64_t)lwrr - (int64_t)new_expected);
            int32_t  abs_offset   = ABS(offset);
            // stats
            if(offset < min_offset)
                min_offset = offset;
            if(offset > max_offset)
                max_offset = offset;
            if(abs_offset > max_abs_offset)
                max_abs_offset = abs_offset;
            count += 1;
            sum_abs_offset += (uint64_t)abs_offset;

    #ifdef dbg
            LOG(DEBUG) << count << " " << offset << " " << abs_offset << " " << sum_abs_offset << " " << sum_abs_offset / count << '\n';
    #endif
            runtime = lwrr - first_time;

            if(abs_offset > allowed_offset_nsec)
            {
                LOG(WARNING) << "jitter error, offset " << offset << '\n';
            }

    #ifdef dbg
            LOG(DEBUG) << "lwrr: " << lwrr << ", diff: " << diff << ", offset: " << offset << '\n';
    #endif
            last_expected = new_expected;
            last_actual   = lwrr;
        }

        // print stats
        LOG(DEBUG) << "total run time: " << runtime / (1000000000) << " sec" << '\n';
        LOG(DEBUG) << "event count:    " << count << '\n';
        LOG(DEBUG) << "sum:            " << sum_abs_offset << '\n';
        LOG(DEBUG) << "min offset:     " << min_offset << '\n';
        LOG(DEBUG) << "max offset:     " << max_offset << '\n';
        LOG(DEBUG) << "max abs offset: " << max_abs_offset << '\n';
        LOG(DEBUG) << "avg abs offset: " << sum_abs_offset / count << '\n';
    }

    void lw_altran_phy::slot_indication_thread_timer_fd_method()
    {
    #ifdef dbg
        if(has_thread_cfg)
        {
            sched_param sch;
            int         policy;

            int status = 0;
            status     = pthread_getschedparam(pthread_self(), &policy, &sch);
            if(status == 0)
            {
                LOG(DEBUG) << "slot_indication_thread_timer_fd_method sched_priority " << sch.sched_priority << '\n';
                LOG(DEBUG) << "slot_indication_thread_timer_fd_method on CPU " << sched_getcpu() << "\n";
            }
            else
            {
                LOG(DEBUG) << "pthread_getschedparam failed with status: " << std::strerror(status) << "\n";
            }
        }
    #endif

        lw::phy_config*                                  config = phy_config();
        unique_ptr<member_event_callback<lw_altran_phy>> mcb_p(new member_event_callback<lw_altran_phy>(this, &lw_altran_phy::slot_indication_handler));
        unique_ptr<timer_fd>                             fd_p(new timer_fd(lw::mu_to_ns(config->ssb_config_.sub_c_common), true));
        //unique_ptr<timer_fd>                             fd_p(new timer_fd(PT_SLOT_INDICATION_NS, true));
        //unique_ptr<timer_fd> fd_p(new timer_fd(500000, true));

        epoll_ctx.add_fd(fd_p->get_fd(), mcb_p.get());

        timer_fd_p  = std::move(fd_p);
        timer_mcb_p = std::move(mcb_p);
        try
        {
            epoll_ctx.start_event_loop();
        }
        catch(std::exception& e)
        {
            fprintf(stderr,
                    "lw_altran_phy::thread_func() exception: %s\n",
                    e.what());
        }
        catch(...)
        {
            fprintf(stderr,
                    "lw_altran_phy::thread_func() unknown exception\n");
        }
    }

    void lw_altran_phy::send_rx_ulsch_indication()
    {
        LOGF(INFO, "lw_altran_phy sending rx ulsch indication...\n");
        lw::tx_msg_desc           tx_msg_desc(phy_module().transport());
        altran_fapi_config_req_t& request = altran_fapi::message::add_fapi_hdr<altran_fapi_config_req_t>(tx_msg_desc.msg_desc);
        phy_module().transport().tx_send(tx_msg_desc.msg_desc);
        phy_module().transport().tx_post();

        //-  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
        // clear timer fd signal
        timer_fd_p->clear();
    }

    void lw_altran_phy::send_crc_indication()
    {
        LOGF(INFO, "lw_altran_phy sending crc indication...\n");
        lw::tx_msg_desc           tx_msg_desc(phy_module().transport());
        altran_fapi_config_req_t& request = altran_fapi::message::add_fapi_hdr<altran_fapi_config_req_t>(tx_msg_desc.msg_desc);

        //        altran_fapi_start_req_t(request);
        //-  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
        // Send the message over the transport
        phy_module().transport().tx_send(tx_msg_desc.msg_desc);
        phy_module().transport().tx_post();

        //-  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
        // clear timer fd signal
        timer_fd_p->clear();
    }

    void lw_altran_phy::set_default_slot_indication(altran_fapi_slot_ind_t& slot_ind)
    {
        slot_ind.sfn_slot_carrier = 0x00000005;
        slot_ind.global_tick      = 0x00000000;
    }

    void lw_altran_phy::send_slot_indication()
    {
        lw::tx_msg_desc         tx_msg_desc(phy_module().transport());
        altran_fapi_slot_ind_t& slot_ind = altran_fapi::message::add_fapi_hdr<altran_fapi_slot_ind_t>(tx_msg_desc.msg_desc);
        //set_default_slot_indication(slot_ind);
        lw::slot_indication& slot_info = lwrrent_slot_info();
        slot_ind.sfn_slot_carrier      = lw::get_sfn_slot_carrier(get_carrier_id(), slot_info.slot_, slot_info.sfn_);
        slot_ind.global_tick           = slot_info.tick_;

        LOGF(INFO, "lw_altran_phy sending slot indication...tick:%u, SFN:%u, Slot:%u\n", slot_ind.global_tick, slot_info.sfn_, slot_info.slot_);

        if(slot_info.slot_ == 19)
        {
            slot_info.sfn_ = (slot_info.sfn_ + 1) % 1024;
        }
        slot_info.slot_ = (slot_info.slot_ + 1) % SLOT_WRAP;
        slot_info.tick_++;

        std::atomic<uint64_t>atom_ts{(uint64_t)slot_indication_ts};
        ptctx->slot_3gpp_ref_ts[(int)slot_info.slot_] = atom_ts;
        LOGF(INFO, "Current TS: %" PRIu64 " 3GPP slot %d TS: %" PRIu64 "\n", get_ns(), (int)slot_info.slot_, ptctx->slot_3gpp_ref_ts[(int)slot_info.slot_].load());

        tx_msg_desc.msg_desc.msg_len = sizeof(altran_fapi_slot_ind_t);
        //-  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
        // Send the message over the transport
        phy_module().transport().tx_send(tx_msg_desc.msg_desc);
        phy_module().transport().notify(IPC_NOTIFY_VALUE);

    #ifndef SLOT_INDICATION_POLLING
        //-  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
        // clear timer fd signal
        timer_fd_p->clear();
    #endif
    }

    void lw_altran_phy::slot_indication_handler()
    {
    #ifdef dbg
        auto   lwr_tp = chrono::high_resolution_clock::now();
        double delay  = chrono::duration_cast<chrono::microseconds>(lwr_tp - prev_tp).count();
        LOG(DEBUG) << "slot indication interval is : " << delay << " micro secs" << '\n';
        prev_tp = lwr_tp;
    #endif

        send_slot_indication();
    }

    int lw_altran_phy::send_fh_command() {
        
        struct l2_control_slot_info * l2_slot_info = NULL;
        std::map<fh::channel_type, fh::prb_sym_loc>::iterator it;

        fh::fh_command * fh_cmd = altran_fapi::phy::get_lwrrent_fh_command();
        if(fh_cmd == NULL)
            return -1;
        
        l2_slot_info = NULL;

        if(fh_cmd->slot_info.type == fh::SLOT_UPLINK)
        {
	        if (rte_ring_empty(ptctx->ring_free_ul) == 1) {
                ul_drop_fh_cmds++;
                LOGF(INFO, "Dropping the UL FH command at SFN = %d, Slot %d Drop Count: UL = %lu, DL = %lu",
                    fh_cmd->slot_info.slot.sfn_, fh_cmd->slot_info.slot.slot_, ul_drop_fh_cmds, dl_drop_fh_cmds);

		        return -1;
            }

            while(rte_ring_dequeue(ptctx->ring_free_ul, (void **)&l2_slot_info) != 0);
            // LOGF(INFO, "ring_free_ul Queue size=%d", rte_ring_count(ptctx->ring_free_ul));
            if(check_force_quit(force_quit)) {
                LOGF(INFO, "Force quit during dequeue ring_free_ul");
                exit(EXIT_SUCCESS);
            }

            if(l2_slot_info == NULL)
            {
                LOGF(INFO, "l2_slot_info is NULL\n");
                exit(EXIT_FAILURE);
            }

            l2_slot_info->reset();
            l2_slot_info->recv_tstamp = get_ns();
            l2_slot_info->frameId = fh_cmd->slot_info.oran_slot_info.oframe_id_;
            l2_slot_info->subFrameId = fh_cmd->slot_info.oran_slot_info.osfid_;
            l2_slot_info->slotId = fh_cmd->slot_info.oran_slot_info.oslotid_;
            l2_slot_info->slotId_3GPP = fh_cmd->slot_info.slot.slot_;
            l2_slot_info->tick = ptctx->slot_3gpp_ref_ts[(int)l2_slot_info->slotId_3GPP].load();
            l2_slot_info->fh_cmd = *fh_cmd;

            it = fh_cmd->slot_info.slot_ch_info.find(fh::PUSCH);
            if (it != fh_cmd->slot_info.slot_ch_info.end())
            {
                enum phy_channel_type phy_ch = PHY_PUSCH;
                struct fh::prb_sym_loc &prb_info = fh_cmd->slot_info.slot_ch_info[fh::PUSCH];
                l2_slot_info->all_startPrb[phy_ch] = prb_info.startPrb;
                l2_slot_info->all_numPrb[phy_ch] = prb_info.numPrb;
                l2_slot_info->all_startSym[phy_ch] = prb_info.startSym;
                l2_slot_info->all_numSym[phy_ch] = prb_info.numSym;
                struct fh::ch_block_params block_params = fh_cmd->channel_params.block_params[fh::PUSCH];
                l2_slot_info->all_block_params[phy_ch] = block_params.pars;
                l2_slot_info->all_cell_params[phy_ch] = fh_cmd->channel_params.cell_params;
                l2_slot_info->phy_ch_type = PHY_PUSCH;

                LOGF(INFO, "FH cmd UL PUSCH pipeline %d at %" PRIu64 " with startPrb=%d, numPrb=%d startSym=%d numSym=%d frameId=%d, subframeid=%d, slotid=%d, 3GPP slot: %d, tick: %" PRIu64 "\n", 
                    ul_number, get_ns(),
                    l2_slot_info->all_startPrb[PHY_PUSCH],  l2_slot_info->all_numPrb[PHY_PUSCH],  
                    l2_slot_info->all_startSym[PHY_PUSCH], l2_slot_info->all_numSym[PHY_PUSCH], 
                    l2_slot_info->frameId, l2_slot_info->subFrameId, l2_slot_info->slotId, l2_slot_info->slotId_3GPP, l2_slot_info->tick
                );
            }

            it = fh_cmd->slot_info.slot_ch_info.find(fh::PUCCH);
            if (it != fh_cmd->slot_info.slot_ch_info.end())
            {
                enum phy_channel_type phy_ch = PHY_PUCCH;
                struct fh::prb_sym_loc &prb_info = fh_cmd->slot_info.slot_ch_info[fh::PUCCH];
                l2_slot_info->all_startPrb[phy_ch] = prb_info.startPrb;
                l2_slot_info->all_numPrb[phy_ch] = prb_info.numPrb;
                l2_slot_info->all_startSym[phy_ch] = prb_info.startSym;
                l2_slot_info->all_numSym[phy_ch] = prb_info.numSym;
                struct fh::ch_block_params block_params = fh_cmd->channel_params.block_params[fh::PUCCH];
                l2_slot_info->all_block_params[phy_ch] = block_params.pars;
                l2_slot_info->all_cell_params[phy_ch] = fh_cmd->channel_params.cell_params;
                l2_slot_info->phy_ch_type = PHY_PUCCH;

                LOGF(INFO, "FH cmd UL PUCCH pipeline %d at %" PRIu64 " with startPrb=%d, numPrb=%d startSym=%d numSym=%d frameId=%d, subframeid=%d, slotid=%d, 3GPP slot: %d, tick: %" PRIu64 "\n", 
                    ul_number, get_ns(),
                    l2_slot_info->all_startPrb[PHY_PUCCH],  l2_slot_info->all_numPrb[PHY_PUCCH],  
                    l2_slot_info->all_startSym[PHY_PUCCH], l2_slot_info->all_numSym[PHY_PUCCH], 
                    l2_slot_info->frameId, l2_slot_info->subFrameId, l2_slot_info->slotId, l2_slot_info->slotId_3GPP, l2_slot_info->tick
                );
            }

            while(rte_ring_enqueue(ptctx->ring_start_ul, (void*)l2_slot_info) != 0);
            ul_number++;
            if(check_force_quit(force_quit)) {
                LOGF(INFO, "Force quit during enqueue ring_start_ul");
                exit(EXIT_SUCCESS);
            }
        }

        if(fh_cmd->slot_info.type == fh::SLOT_DOWNLINK)
        {
	        if (rte_ring_empty(ptctx->ring_free_dl) == 1) {
                dl_drop_fh_cmds++;
                LOGF(INFO, "Dropping the DL FH command at SFN = %d, Slot %d Drop Count: UL = %lu, DL = %lu",
                    fh_cmd->slot_info.slot.sfn_, fh_cmd->slot_info.slot.slot_, ul_drop_fh_cmds, dl_drop_fh_cmds);
                return -1;
            }

            while(rte_ring_dequeue(ptctx->ring_free_dl, (void **)&l2_slot_info) != 0);
            if(check_force_quit(force_quit)) {
                LOGF(INFO, "Force quit during dequeue ring_free_dl");
                exit(EXIT_SUCCESS);
            }
            if(l2_slot_info == NULL)
            {
                LOGF(INFO, "l2_slot_info is NULL\n");
                exit(EXIT_FAILURE);
            }
            l2_slot_info->reset();
            l2_slot_info->recv_tstamp = get_ns();
            l2_slot_info->frameId = fh_cmd->slot_info.oran_slot_info.oframe_id_;
            l2_slot_info->subFrameId = fh_cmd->slot_info.oran_slot_info.osfid_;
            l2_slot_info->slotId = fh_cmd->slot_info.oran_slot_info.oslotid_;
            l2_slot_info->slotId_3GPP = fh_cmd->slot_info.slot.slot_;
            l2_slot_info->tick = ptctx->slot_3gpp_ref_ts[(int)l2_slot_info->slotId_3GPP].load();
            l2_slot_info->fh_cmd = *fh_cmd;

            bool PDCCH_UL = false;
            bool PDCCH_DL = false;
            
            if (fh_cmd->slot_info.slot_ch_info.count(fh::PDCCH)) {
                auto&  dcis = fh_cmd->channel_params.block_params[fh::PDCCH].dci_format_list;
                if (std::find(std::begin(dcis), std::end(dcis), fh::DCI_0_0) != std::end(dcis)) {
                    PDCCH_UL = true;
                }
                if (std::find(std::begin(dcis), std::end(dcis), fh::DCI_1_1) != std::end(dcis)) {
                    PDCCH_DL = true;
                }
            }

            it = fh_cmd->slot_info.slot_ch_info.find(fh::PDSCH);
            if (it != fh_cmd->slot_info.slot_ch_info.end())
            {
                enum phy_channel_type phy_ch = PHY_PDSCH;
                struct fh::prb_sym_loc &prb_info = fh_cmd->slot_info.slot_ch_info[fh::PDSCH];
                l2_slot_info->all_startPrb[phy_ch] = prb_info.startPrb;
                l2_slot_info->all_numPrb[phy_ch] = prb_info.numPrb;
                l2_slot_info->all_startSym[phy_ch] = prb_info.startSym;
                l2_slot_info->all_numSym[phy_ch] = prb_info.numSym;

                struct fh::ch_block_params &block_params = fh_cmd->channel_params.block_params[fh::PDSCH];
                if (block_params.input_data_buf == NULL || block_params.input_data_buf_size == 0) {
                    LOGF(FATAL, "FH cmd: skipping slot. PDSCH input: (%p:%zd). O-RAN: frameId %u subFrameId %u slotId %u. 3GPP slotId %u\n",
                        block_params.input_data_buf,
                        block_params.input_data_buf_size,
                        fh_cmd->slot_info.oran_slot_info.oframe_id_,
                        fh_cmd->slot_info.oran_slot_info.osfid_,
                        fh_cmd->slot_info.oran_slot_info.oslotid_,
                        fh_cmd->slot_info.slot.slot_);
                        fh_cmd->post_callback(fh_cmd->slot_info.slot, fh_cmd->channel_params, 0);
                    rte_ring_enqueue(ptctx->ring_free_dl, (void *)l2_slot_info);
                    return 0;
                }
                l2_slot_info->all_cell_params[phy_ch] = fh_cmd->channel_params.cell_params;
                l2_slot_info->all_block_params[phy_ch] = block_params.pars;
                l2_slot_info->all_input_addr[phy_ch] = block_params.input_data_buf;
                l2_slot_info->all_input_size[phy_ch] = block_params.input_data_buf_size;

                // LOGF(INFO, "Start DL PDSCH pipeline %d with startPrb=%d, numPrb=%d startSym=%d numSym=%d frameId=%d, subframeid=%d, slotid=%d, 3GPP slot: %d\n", 
                //     dl_number, 
                //     l2_slot_info->all_startPrb[PHY_PDSCH],  l2_slot_info->all_numPrb[PHY_PDSCH],  
                //     l2_slot_info->all_startSym[PHY_PDSCH], l2_slot_info->all_numSym[PHY_PDSCH], 
                //     l2_slot_info->frameId, l2_slot_info->subFrameId, l2_slot_info->slotId, l2_slot_info->slotId_3GPP
                // );
            }

            // push all channels to be exelwted in this slot we should simply std::move the slot_info and other stuff into l2_slot_info, not "translate" the structures
            it = fh_cmd->slot_info.slot_ch_info.find(fh::PDCCH);
            if (it != fh_cmd->slot_info.slot_ch_info.end())
            {
                //printf("Run() slot %u has PDCCH\n", l2_slot_info->slotId_3GPP);
                enum phy_channel_type phy_ch = PHY_PDCCH;
                struct fh::prb_sym_loc &prb_info = fh_cmd->slot_info.slot_ch_info[fh::PDCCH];
                l2_slot_info->all_startPrb[phy_ch] = prb_info.startPrb;
                l2_slot_info->all_numPrb[phy_ch] = prb_info.numPrb;
                l2_slot_info->all_startSym[phy_ch] = prb_info.startSym;
                l2_slot_info->all_numSym[phy_ch] = prb_info.numSym;

                struct fh::ch_block_params &block_params = fh_cmd->channel_params.block_params[fh::PDCCH];
                l2_slot_info->all_block_params[phy_ch] = block_params.pars;
                l2_slot_info->all_cell_params[phy_ch] = fh_cmd->channel_params.cell_params;
                for (auto dci_format : block_params.dci_format_list) {
                	auto phy_dci_format = (dci_format == fh::DCI_0_0) ? PHY_DCI_0_0 : PHY_DCI_1_1;
                    // printf("Run() slot %u has PDCCH format %s\n", l2_slot_info->slotId_3GPP, (dci_format == fh::DCI_0_0) ?
                	//        "DCI_0_0" : "DCI_1_1");
                	l2_slot_info->phy_dci_format_list.push_back(phy_dci_format);
                }

                // LOGF(INFO, "Start DL PDCCH pipeline %d with startPrb=%d, numPrb=%d startSym=%d numSym=%d frameId=%d, subframeid=%d, slotid=%d, 3GPP slot: %d\n", 
                //     dl_number,
                //     l2_slot_info->all_startPrb[PHY_PDCCH],  l2_slot_info->all_numPrb[PHY_PDCCH],  
                //     l2_slot_info->all_startSym[PHY_PDCCH], l2_slot_info->all_numSym[PHY_PDCCH], 
                //     l2_slot_info->frameId, l2_slot_info->subFrameId, l2_slot_info->slotId, l2_slot_info->slotId_3GPP
                // );
            }

            it = fh_cmd->slot_info.slot_ch_info.find(fh::PBCH);
            if (it != fh_cmd->slot_info.slot_ch_info.end())
            {
                enum phy_channel_type phy_ch = PHY_PBCH;
                struct fh::prb_sym_loc &prb_info = fh_cmd->slot_info.slot_ch_info[fh::PBCH];
                l2_slot_info->all_startPrb[phy_ch] = prb_info.startPrb;
                l2_slot_info->all_numPrb[phy_ch] = prb_info.numPrb;
                l2_slot_info->all_startSym[phy_ch] = prb_info.startSym;
                l2_slot_info->all_numSym[phy_ch] = prb_info.numSym;

                struct fh::ch_block_params &block_params = fh_cmd->channel_params.block_params[fh::PBCH];
                l2_slot_info->all_block_params[phy_ch] = block_params.pars;
                l2_slot_info->all_cell_params[phy_ch] = fh_cmd->channel_params.cell_params;
                
                LOGF(INFO, "Start DL PBCH pipeline %d with startPrb=%d, numPrb=%d startSym=%d numSym=%d frameId=%d, subframeid=%d, slotid=%d, 3GPP slot: %d\n", 
                    dl_number, 
                    l2_slot_info->all_startPrb[PHY_PBCH],  l2_slot_info->all_numPrb[PHY_PBCH],  
                    l2_slot_info->all_startSym[PHY_PBCH], l2_slot_info->all_numSym[PHY_PBCH], 
                    l2_slot_info->frameId, l2_slot_info->subFrameId, l2_slot_info->slotId, l2_slot_info->slotId_3GPP
                );
            }
	        
            LOGF(INFO, "FH cmd DL slot %d at %" PRIu64 ". O-RAN: frameId %u subFrameId %u slotId %u. 3GPP slotId %u tick %" PRIu64 ". Channels present:%s%s%s%s\n",
                dl_number, get_ns(),
                fh_cmd->slot_info.oran_slot_info.oframe_id_,
                fh_cmd->slot_info.oran_slot_info.osfid_,
                fh_cmd->slot_info.oran_slot_info.oslotid_,
                fh_cmd->slot_info.slot.slot_,
                l2_slot_info->tick,
                fh_cmd->slot_info.slot_ch_info.count(fh::PDSCH) ? " PDSCH" : "",
                PDCCH_UL ? " PDCCH_UL" : "",
                PDCCH_DL ? " PDCCH_DL" : "",
                fh_cmd->slot_info.slot_ch_info.count(fh::PBCH) ? " PBCH" : ""
            );

            while(rte_ring_enqueue(ptctx->ring_start_dl, (void*)l2_slot_info) != 0);
            dl_number++;
            if(check_force_quit(force_quit)) {
                LOGF(INFO, "Force quit during enqueue ring_start_dl");
                exit(EXIT_SUCCESS);
            }
        }

        reset_lwrrent_fh_command();
    }
} // namespace lw_altran_stack
