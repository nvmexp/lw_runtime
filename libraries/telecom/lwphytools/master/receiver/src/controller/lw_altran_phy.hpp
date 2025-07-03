/*
 * Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 */

#ifndef LW_ALTRAN_PHY_HPP_INCLUDED_
#define LW_ALTRAN_PHY_HPP_INCLUDED_

#include "altran_fapi_includes.hpp"
#include "altran_fapi_phy.hpp"

#include "lw_phy_instance.hpp"
#include "lw_phy_fapi_msg_common.hpp"
#include "altran_fapi_msg_helpers.hpp"
#include "lw_g3logger_enabler.hpp"

#include "lw_phy_epoll_context.hpp"
#include <chrono>

#include <thread>

#include "lwphytools.hpp"

using namespace std;
using namespace lw;

// Turn on DEBUG for Timer thread
//#define dbg

#define SLOT_INDICATION_POLLING
#define ABS(N) (((N) < 0) ? (-(N)) : (N))
#define allowed_offset_nsec 100000
#define test_time_sec (20)
#define pre_window 200
#define ADVANCE_SLOTS 4

namespace lw_altran_stack
{
    struct thread_config
    {
        size_t cpu_affinity;
        int    sched_priority;
    };

    class lw_altran_phy : public altran_fapi::phy {
        public:
            //------------------------------------------------------------------
            // lw_altran_phy()
            lw_altran_phy(lw::PHY_module& module, yaml::node config, struct phytools_ctx * _ptctx);
            //------------------------------------------------------------------
            // on_msg()
            virtual bool on_msg(lw_ipc_msg_t& msg) override;

        private:
            void send_rx_ulsch_indication();
            void send_crc_indication();
            void send_slot_indication();

            void set_default_slot_indication(altran_fapi_slot_ind_t& slot_ind);

            void start_slot_indication_thread();

            void     slot_indication_thread_poll_method();
            uint64_t sys_clock_time_handler();

            void slot_indication_thread_timer_fd_method();
            void timer_thread_func();
            void slot_indication_handler();
            int send_fh_command();

            bool isEqual(uint8_t* a, uint8_t* b, int len);
            //------------------------------------------------------------------
            // void on_mac_phy_cell_start_req(altran_fapi_start_req_t& request);

            void on_mac_phy_cell_config_req_test(altran_fapi_config_req_t& request);

            std::thread timer_thread; // timer thread
        #ifdef dbg
            //Measure slot indication interval
            chrono::high_resolution_clock::time_point prev_tp;
        #endif
            thread_config timer_thread_cfg;
            uint32_t      window_nsec;
            volatile bool has_thread_cfg = false;
            volatile bool started        = false;

            phy_epoll_context                                epoll_ctx;
            unique_ptr<timer_fd>                             timer_fd_p;
            unique_ptr<member_event_callback<lw_altran_phy>> timer_mcb_p;

            struct phytools_ctx * ptctx;
    };
} // namespace lw_altran_stack

#endif //LW_ALTRAN_PHY_HPP_INCLUDED_
