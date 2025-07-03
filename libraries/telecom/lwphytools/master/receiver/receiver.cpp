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
#include <rte_log.h>

#include "lwphytools.hpp"

static struct phytools_ctx ptctx;
// The main "everybody quit" switch. Can be triggered by any core.
// std::atomic_bool force_quit{0};
// std::atomic<std::bool> force_quit.store(false);
std::atomic<bool> force_quit;

static uint64_t timer_hz = 0;
static struct core_info crx_info[DPDK_MAX_CORES];
static struct core_info ctx_info[DPDK_MAX_CORES];
static struct core_info cord_info[DPDK_MAX_CORES];
static uint64_t t_start_next_slot=0;

static void signal_handler(int signum)
{
    if (signum == SIGINT || signum == SIGTERM || signum == SIGUSR1) {
        if (check_force_quit(force_quit) == true) {
            rte_exit(EXIT_FAILURE, "Signal %d received; quitting the hard way\n", signum);
        }
        pt_warn("Signal %d received, preparing to exit...\n", signum);
        set_force_quit(force_quit);
    }
}

int main(int argc, char *argv[])
{
    int ret=PT_OK, slave_core=0, icore=0, i=0, j=0, offset=0;
    uint8_t * tmp_buffer;
    DECLARE_FOREACH_PIPELINE

    unset_force_quit(force_quit);

    ////////////////////////////////////////////////
    //// Initial setup
    ////////////////////////////////////////////////
    ret = pt_init(&(ptctx), argc, argv);
    if(ret != PT_OK)
    {
        if (ret != PT_STOP)
            pt_err("pt_init returned error %d\n", ret);
        exit(EXIT_FAILURE);
    }

    pt_info("====> Initial setup\n");
    ret = pt_setup_gpu(&(ptctx));
    if(ret != PT_OK)
    {
        pt_err("pt_setup_gpu returned error %d\n", ret);
        exit(EXIT_FAILURE);
    }

    ret = dpdk_setup(&(ptctx));
    if(ret != PT_OK)
    {
        pt_err("dpdk_setup returned error %d\n", ret);
        exit(EXIT_FAILURE);
    }

    // timerdiff_us(0,0); // initialize timer_hz inside

    pt_print_ctx(&(ptctx));
 
    ////////////////////////////////////////////////////////////
    //// Setup pipeline flow rules
    ////////////////////////////////////////////////////////////
    ret = dpdk_flow_isolate(&(ptctx));
    if(ret != PT_OK)
    {
        pt_err("dpdk_flow_isolate returned error %d\n", ret);
        exit(EXIT_FAILURE);
    }

    ////////////////////////////////////////////////
    //// DPDK mempools and queues
    ////////////////////////////////////////////////
    for(index_pipeline=0; index_pipeline < ptctx.num_pipelines; index_pipeline++)
    {
        plctx = &(ptctx.plctx[index_pipeline]);
        LW_LWDA_CHECK(lwdaSetDevice(plctx->lwdactx.gpu_id));
        lwdaFree(0);

        if(plctx->uplink)
        {
            if(dpdk_uplink_network_setup(plctx->dpdkctx, index_pipeline, plctx->dpdkctx.hds > 0 ? plctx->dpdkctx.rxq : 1) == PT_ERR) {
                pt_err("dpdk_uplink_setup returned error %d\n", ret);
                dpdk_finalize(&ptctx);
            }
        }

        if(plctx->downlink)
        {
            int war = 0;
            if(plctx->uplink == 0) war = 1;
            if(dpdk_downlink_network_setup(ptctx, plctx->dpdkctx, index_pipeline, 1, war) == PT_ERR)
            {
                pt_err("dpdk_downlink_setup returned error %d\n", ret);
                dpdk_finalize(&ptctx);
            }
        }

        if(dpdk_cplane_network_setup(plctx->dpdkctx, index_pipeline) == PT_ERR)
        {
            pt_err("dpdk_cplane_setup returned error %d\n", ret);
            dpdk_finalize(&ptctx);
        }
    }

    {
        /* programatically disable c-states in case we're running on unoptimized platform */
        uint32_t lat = 0;
        int fd = open("/dev/cpu_dma_latency", O_RDWR);
        if (fd == -1)
            rte_exit(EXIT_FAILURE, "Failed to open cpu_dma_latency: error %s\n", strerror(errno));
        ssize_t ret = write(fd, &lat, sizeof(lat));
        if (ret != sizeof(lat))
            rte_exit(EXIT_FAILURE, "Write to cpu_dma_latency failed: error %s\n", strerror(errno));
    }

    //Start NIC
    ret = dpdk_start_nic(&(ptctx));
    if(ret != PT_OK)
    {
        pt_err("dpdk_start_nic returned error %d\n", ret);
        exit(EXIT_FAILURE);
    }

    ////////////////////////////////////////////////////////////
    //// Signal handler
    ////////////////////////////////////////////////////////////
    PT_ACCESS_ONCE(force_quit)=0;
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    signal(SIGUSR1, signal_handler);

    #ifdef PROFILE_LWTX_RANGES
        lwdaProfilerStart();
    #endif

    ret = pt_setup_rings(&(ptctx));
    if(ret != PT_OK)
    {
        pt_err("pt_setup_rings returned error %d\n", ret);
        exit(EXIT_FAILURE);
    }

    ////////////////////////////////////////////////
    //// Uplink setup
    ////////////////////////////////////////////////
    if(plctx->uplink)
    {
        ret = pt_setup_ul_gpu_work(&(ptctx));
        if(ret != PT_OK)
        {
            pt_err("pt_setup_kernels returned error %d\n", ret);
            exit(EXIT_FAILURE);
        }
        
        ret = pt_setup_ul_slot_table(&(ptctx));
        if(ret != PT_OK)
        {
            pt_err("pt_setup_ul_slot_table returned error %d\n", ret);
            exit(EXIT_FAILURE);
        }

        ret = pt_setup_ul_rings(&(ptctx));
        if(ret != PT_OK)
        {
            pt_err("pt_setup_ul_rings returned error %d\n", ret);
            exit(EXIT_FAILURE);
        }
        
        ////////////////////////////////////////////////////////////
        //// Setup pipeline flow rules
        ////////////////////////////////////////////////////////////
        ret = dpdk_setup_rules(&(ptctx));
        if(ret != PT_OK)
        {
            pt_err("dpdk_setup_rules returned error %d\n", ret);
            exit(EXIT_FAILURE);
        }
    }

    if(plctx->downlink)
    {
        ret = pt_setup_dl_gpu_work(&(ptctx));
        if(ret != PT_OK)
        {
            pt_err("pt_setup_kernels returned error %d\n", ret);
            exit(EXIT_FAILURE);
        }

        ret = pt_setup_dl_slot_table(&(ptctx));
        if(ret != PT_OK)
        {
            pt_err("pt_setup_dl_slot_table returned error %d\n", ret);
            exit(EXIT_FAILURE);
        }

        ret = pt_setup_dl_rings(&(ptctx));
        if(ret != PT_OK)
        {
            pt_err("pt_setup_ul_rings returned error %d\n", ret);
            exit(EXIT_FAILURE);
        }
    }

    ////////////////////////////////////////////////////////////
    //// Start slave cores
    ////////////////////////////////////////////////////////////
    pt_info("====> Launching the slave cores\n");
    icore=0;

    

    for(index_pipeline=0; index_pipeline < ptctx.num_pipelines; index_pipeline++)
    {
        plctx = &(ptctx.plctx[index_pipeline]);
        plctx->ul_num_processed_slots=0;
        plctx->dl_num_processed_slots=0;

        //C-plane core
        ctx_info[index_pipeline].txq = plctx->dpdkctx.start_txq;
        ctx_info[index_pipeline].port_id = plctx->dpdkctx.port_id;
        ctx_info[index_pipeline].ptctx = &(ptctx);
        ctx_info[index_pipeline].plctx = &(ptctx.plctx[index_pipeline]);
    
        if(plctx->uplink)
        {
            icore = rte_get_next_lcore(icore,1,0);
            rte_eal_remote_launch(uplink_c_core, &(ctx_info[index_pipeline]), icore);

            //RX cores
            crx_info[index_pipeline].rxq = plctx->dpdkctx.start_rxq+i;
            crx_info[index_pipeline].port_id = plctx->dpdkctx.port_id;
            crx_info[index_pipeline].ptctx = &(ptctx);
            crx_info[index_pipeline].plctx = &(ptctx.plctx[index_pipeline]);
            icore = rte_get_next_lcore(icore,1,0);
            rte_eal_remote_launch(uplink_rx_core, &(crx_info[index_pipeline]), icore); 

            cord_info[index_pipeline].ptctx = &(ptctx);
            cord_info[index_pipeline].plctx = &(ptctx.plctx[index_pipeline]);

            //Prepare core -- with 25Gbps, can we merge this with RX cores?
            icore = rte_get_next_lcore(icore,1,0);
            rte_eal_remote_launch(uplink_prepare_core, &(cord_info[index_pipeline]), icore);

            //Timer core
            icore = rte_get_next_lcore(icore,1,0);
            rte_eal_remote_launch(uplink_timer_core, &(cord_info[index_pipeline]), icore);

            //Endpoint core
            icore = rte_get_next_lcore(icore,1,0);
            rte_eal_remote_launch(uplink_endpoint_core, &(cord_info[index_pipeline]), icore);
        }

        if(plctx->downlink)
        {
            // Calc core
            icore = rte_get_next_lcore(icore,1,0);
            rte_eal_remote_launch(downlink_processing_core, &(ctx_info[index_pipeline]), icore);
            
            // TX core
            icore = rte_get_next_lcore(icore,1,0);
            rte_eal_remote_launch(downlink_tx_core, &(ctx_info[index_pipeline]), icore);
        }
    }

    ////////////////////////////////////////////////////////////
    //// Controller core
    ////////////////////////////////////////////////////////////
    if(ptctx.controller == CONTROLLER_DPDK)
    {
        icore = rte_get_next_lcore(icore,1,0);
        rte_eal_remote_launch(controller_core, &(ptctx), icore);
    }
#ifdef LWPHYCONTROLLER
    else if(ptctx.controller == CONTROLLER_LWPHY)
        lwphy_controller_init(ptctx.controller_file, &(ptctx));
#endif

#ifdef NO_PRINTS
    fprintf(stderr, "lwPHYTools initialization complete.\n");
#endif
    ////////////////////////////////////////////////////////////
    //// Wait for the slave cores
    ////////////////////////////////////////////////////////////
    icore=0;
    RTE_LCORE_FOREACH_SLAVE(icore) {
        if (rte_eal_wait_lcore(icore) < 0) {
            pt_err("bad exit for coreid: %d\n", icore);
            break;
        }
    }

    #ifdef PROFILE_LWTX_RANGES
        lwdaProfilerStop();
    #endif

end:
    ////////////////////////////////////////////////////////////
    //// Finalize and exit
    ////////////////////////////////////////////////////////////
    dpdk_print_stats(&(ptctx));
    if(plctx->uplink)
    {
        ret = pt_ul_finalize(&(ptctx));
        if(ret != PT_OK)
        {
            pt_err("pt_ul_finalize returned error %d\n", ret);
            return ret;
        }
    }

    if(plctx->downlink)
    {
        ret = pt_dl_finalize(&(ptctx));
        if(ret != PT_OK)
        {
            pt_err("pt_dl_finalize returned error %d\n", ret);
            return ret;
        }
    }

    ret = pt_finalize(&(ptctx));
    if(ret != PT_OK)
    {
        pt_err("pt_finalize returned error %d\n", ret);
        return ret;
    }

// #ifdef LWPHYCONTROLLER
//     ret = lwphy_controller_finalize();
//     if(ret != PT_OK)
//     {
//         pt_err("lwphy_controller_finalize returned error %d\n", ret);
//         return ret;
//     }
// #endif
    // pt_info("Done. Bye!\n");
    rte_exit(EXIT_SUCCESS, "Exitting. Bye!\n");
    return 0;
}
