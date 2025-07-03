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

////////////////////////////////////////////////////////////
/// Defines & Enums
////////////////////////////////////////////////////////////
#ifndef PHYTOOLS_HPP__
#define PHYTOOLS_HPP__

#include "general.hpp"
#include "lwca.hpp"
#include "oran_structs.hpp"

#include <signal.h>
#include <rte_cycles.h>
#include <rte_malloc.h>
#include <rte_errno.h>
#include <atomic>
#include <map>

#include "dpdk.hpp"
#include "tv_parser.hpp"
#include "pipeline_phy.hpp"

#ifdef LWPHYCONTROLLER
    #include "altran_fh_commands.hpp"
#endif
////////////////////////////////////////////////////////////
/// Defines combined
////////////////////////////////////////////////////////////
#define PT_GET_PKT_HDR(x) DPDK_GET_MBUF_ADDR(x, (PT_PKT_HDR_SIZE/2)) //0) 
#define PT_GET_PKT_PAYLOAD(x) DPDK_GET_MBUF_ADDR(x, PT_PKT_HDR_SIZE)
#define PT_GET_BUF_OFFSET(b, t, o) ((t)((char *)b + (o)))
#define PIPELINE_UL 0
#define PIPELINE_DL 1
#define SLOT_WRAP 20

template <typename T>
struct atomicwrapper
{
    std::atomic<T> _a;

    atomicwrapper()
        :_a()
    {}

    atomicwrapper(const std::atomic<T> &a)
        :_a(a.load())
    {}

    atomicwrapper(const atomicwrapper &other)
        :_a(other._a.load())
    {}

    atomicwrapper &operator=(const atomicwrapper &other)
    {
        _a.store(other._a.load()); //, std::memory_order_relaxed); <-- no synchronization
    }

    atomicwrapper &operator=(std::atomic<T> &other)
    {
        _a.store(other.load());
    }

    T load()
    {
        return _a.load();
    }
};

enum {
    CONTROLLER_DPDK     = 0,
    CONTROLLER_LWPHY    = 1,
};

////////////////////////////////////////////////////////////
/// Memory structures
////////////////////////////////////////////////////////////

struct pt_slot_table {
    uint8_t     * ptr_i_h;

    PuschPHY * pusch_ul_phy;
    PdschPHY * pdsch_dl_phy;
    PdcchPHY * pdcch_ul_phy;
    PdcchPHY * pdcch_dl_301_phy;
    PdcchPHY * pdcch_dl_301a_phy;
    PdcchPHY * pdcch_dl_301b_phy;
    PbchPHY  * pbch_phy;

    int         size;
    uint32_t    num_packets;
    int         index;
    int         start_mbatch;
    int         end_mbatch;

    Slot slot_dims;
    
    uint16_t flow_tot;

    struct l2_control_slot_info * l2_slot_info;
    uint8_t * tb_output;

    //Slot timers
    uint64_t t_launch_ch_start;
    uint64_t t_launch_ch_end;
    uint64_t t_enque_ch_start;
    uint64_t t_enque_ch_end;
    uint64_t t_start_order;
    uint64_t t_start_ch;
    uint64_t t_end_ch;

    uint64_t t_slot_start;
    uint64_t t_slot_prepare_start;
    uint64_t t_slot_prepare_stop;
    uint64_t t_slot_stop;
    uint64_t t_slot_end;

    uint64_t t_start_enqueue;
    uint64_t t_stop_enqueue;

    uint64_t t_alloc_fn_start = 0;
    uint64_t t_alloc_fn_end = 0;

    uint64_t t_copyOutputToCPU_start = 0;
    uint64_t t_copyOutputToCPU_end = 0;

    uint64_t t_callback_start = 0;
    uint64_t t_callback_end = 0;

    uint64_t t_pipeline_end = 0;
};

struct mbufs_batch_meta {
    uint64_t t_mbatch_wait;
    uint64_t t_mbatch_rxprepare_start;
    uint64_t t_mbatch_prepare;
    uint64_t t_mbatch_rxprepare_end;

    //Flusher in the middle
    uint64_t t_mbatch_ready_start;
    uint64_t t_mbatch_ready_end;
    uint64_t t_worker_done;
};

struct tv_info {
    int index;
    std::string file_name;
    uint8_t * idata_h;
    uint8_t * odata_h;
    size_t size;
};

struct core_info {
    int txq;
    int rxq;
    int port_id;
    int socket_id;
    struct phytools_ctx * ptctx;
    struct pipeline_ctx * plctx;
};

struct pt_packet_header {
    uint16_t slot_number;
    uint16_t packet_number;
} __attribute__((packed));

struct dl_tx_info {
    struct l2_control_slot_info * l2_slot_info;
    void * tx_buf;
    size_t tx_size;
    uint64_t tx_c_time;
    uint64_t tx_u_time;
    int slot_index;
};

///////////////////////////////////////
//// Pipeline specific parameters
///////////////////////////////////////
struct pipeline_ctx {
    int index;
    struct tv_info * pusch_tv_list;
    struct tv_info * pdsch_tv_list;
    struct tv_info * config_tv_list;
    int tot_pusch_tv;
    int tot_pdsch_tv;
    int tot_config_tv;
    std::string name;
    int enabled;
    int ul_num_processed_slots;
    int dl_num_processed_slots;
    int validation;
    enum pt_timer_level measure_time;
    int lwphy_descrambling;
    int totCores;
    std::string oran_uplane_hdr;

    uint16_t flow_tot; //ecpriPcid/ecpriRtcid
    std::array<uint32_t, PT_MAX_FLOWS_X_PIPELINE> flow_list;
    uint16_t * flow_cache;

    int prbs_per_pkt;
    int rx_batching_us;
    int uplink;
    int downlink;
    int tti;
    int downlink_slots;
    int uplink_slots;
    int first_ap_only;
    int wait_downlink_slot;
    int dump_pusch_input;
    int dump_pusch_output;
    int dump_dl_output;
    int sync_tx_tti;
    int tv_slot_3gpp;
    int ul_cplane_delay;

    uint32_t * ul_checksum_runtime;
    uint32_t * ul_checksum_original;

    int slot_num_max_3gpp;
    std::array<int, PT_MAX_SLOTS_X_CHANNEL> pbch_slot_list;
    std::array<int, PT_MAX_SLOTS_X_CHANNEL> pusch_slot_list;
    std::array<int, PT_MAX_SLOTS_X_CHANNEL> pdsch_slot_list;
    std::array<int, PT_MAX_SLOTS_X_CHANNEL> pdcch_ul_slot_list;
    std::array<int, PT_MAX_SLOTS_X_CHANNEL> pdcch_dl_slot_list;

    ///////////////////////////////////////
    //// GDRCopy
    ///////////////////////////////////////
    //GDRCopy for pinning flags
    gdr_t g;
    gdr_mh_t mh;
    lwdaStream_t stream_ul;
    lwdaStream_t stream_dl;
 
    //Flush -- CPU read into device memory
    uintptr_t flush_d; //must be device memory
    uintptr_t flush_h; //must be host memory
    uintptr_t flush_free; //must be used to free memory
    size_t flush_size;
    
    ///////////////////////////////////////
    //// CPU - Order kernel communication
    ///////////////////////////////////////
    struct mbufs_batch *mbatch;
    struct mbufs_batch_meta *mbatch_meta;
    //Batch status
    uintptr_t mbufs_batch_ready_flags_h;
    uintptr_t mbufs_batch_ready_flags_d;
    uintptr_t mbufs_batch_ready_flags_free;
    size_t mbufs_batch_ready_flags_size;
    //uint32_t * mbufs_mbatch_status_flag_h;

    //Slot status
    uint32_t * mbufs_slot_start_flags_h;
    uint32_t * mbufs_slot_order_flags_h;
    uint32_t * mbufs_slot_done_flags_h;
    uint32_t * mbufs_slot_sync_flags_d;
    uint32_t * pdsch_phy_done_h;
    uint32_t * pdcch_ul_phy_done_h;
    uint32_t * pdcch_dl_301_phy_done_h;
    uint32_t * pdcch_dl_301a_phy_done_h;
    uint32_t * pdcch_dl_301b_phy_done_h;
    uint32_t * pbch_phy_done_h;
    //Slot mapping to buffer
    uint32_t * cache_count_prbs;
    uintptr_t * gbuf_table_cache_ptr;
    //Mbatch mapping to slot
    uint16_t * map_slot_to_last_mbatch;
    //DL TX info
    struct dl_tx_info * dl_tx_list;

    //uint16_t Assuming PT_MAX_SLOT_ID < 256 ==> | mask | value |
    uintptr_t slot_status_d;
    uintptr_t slot_status_h;
    uintptr_t slot_status_free;
    size_t slot_status_size;
    ///////////////////////////////////////
    //// Other contexts
    ///////////////////////////////////////
    struct dpdk_pipeline_ctx dpdkctx;
    struct lwda_ctx lwdactx;    

    ///////////////////////////////////////
    //// Memory structures
    ///////////////////////////////////////
    struct rte_ring ** ring_rxmbufs;
    struct rte_ring * ring_tx_ul;
    struct rte_ring * ring_timer_slot;
    struct rte_ring * ring_timer_batch;
    struct rte_ring * ring_tx_dl;
    struct pt_slot_table * pusch_slot_table_entry;
    struct pt_slot_table * pdsch_slot_table_entry;
    struct pt_slot_table * pdcch_dl_table_entry;
    struct pt_slot_table * pdcch_ul_table_entry;
    struct pt_slot_table * pbch_table_entry;
};

///////////////////////////////////////
//// General parameters
///////////////////////////////////////
struct phytools_ctx {
    int init;
    int totCores;

    int flush_tx_write;
    int no_stats;
    int packets_tot;
    int tot_slot_packets;
    int tot_pkts_x_batch;
    int tot_bursts_x_slot;
    int tot_batch_x_slot;
    int tot_pkts_x_slot;
    int pkts_x_chunk;
    int bursts_x_chunk;
    int num_pipelines;
    int slot_chunk_tot;
    int slot_chunk_size;
    int slot_tot_size;
    int controller;
    std::string controller_file;

    struct dpdk_device_ctx dpdk_dev[RTE_MAX_ETHPORTS];
    
    struct pipeline_ctx * plctx;
    struct rte_ring * ring_start_ul;
    struct rte_ring * ring_start_dl;
    struct rte_ring * ring_free_ul;
    struct rte_ring * ring_free_dl;
    struct l2_control_slot_info * l2_info_ul;
    struct l2_control_slot_info * l2_info_dl;
    std::vector<atomicwrapper<uint64_t>> slot_3gpp_ref_ts;
};

struct l2_control_slot_info {
    //ORAN
    uint32_t startPrb;
    uint32_t numPrb;
    uint32_t startSym;
    uint32_t numSym;
    uint8_t frameId;
    uint8_t subFrameId;
    uint8_t slotId;

    enum phy_channel_type phy_ch_type;
    std::vector<enum phy_dci_format> phy_dci_format_list;

    gnb_pars cell_params;
    std::vector<tb_pars> block_params;
    uint16_t slotId_3GPP;
    uint64_t tick;

#ifdef LWPHYCONTROLLER
    struct fh::fh_command fh_cmd;
#endif
    uint64_t recv_tstamp;
    // per-channel information
    std::map<enum phy_channel_type, uint32_t> all_startPrb;
    std::map<enum phy_channel_type, uint32_t> all_numPrb;
    std::map<enum phy_channel_type, uint32_t> all_startSym;
    std::map<enum phy_channel_type, uint32_t> all_numSym;
    std::map<enum phy_channel_type, uint8_t> all_frameId;
    std::map<enum phy_channel_type, uint8_t> all_subFrameId;
    std::map<enum phy_channel_type, uint8_t> all_slotId;
    std::map<enum phy_channel_type, gnb_pars> all_cell_params;
    std::map<enum phy_channel_type, std::vector<tb_pars>> all_block_params;
    std::map<enum phy_channel_type, void*> all_input_addr;
    std::map<enum phy_channel_type, size_t> all_input_size;
    void reset() {
        frameId = (uint8_t)-1;
        subFrameId = (uint8_t)-1;
        slotId = (uint8_t)-1;
        recv_tstamp = (uint64_t)-1;
        phy_ch_type = (enum phy_channel_type)0;
        phy_dci_format_list.clear();
        slotId_3GPP = 0;
        tick=0;
#ifdef LWPHYCONTROLLER
        fh_cmd.slot_info.slot_ch_info.clear();
        fh_cmd.channel_params.block_params.clear();
        fh_cmd.post_callback = nullptr;
        fh_cmd.alloc_fn = nullptr;
#endif //ifdef LWPHYCONTROLLER
        all_startPrb.clear();
        all_numPrb.clear();
        all_startSym.clear();
        all_numSym.clear();
        all_frameId.clear();
        all_subFrameId.clear();
        all_slotId.clear();
        all_cell_params.clear();
        all_block_params.clear();
        all_input_addr.clear();
        all_input_size.clear();
    }
};

// extern std::atomic_bool force_quit;
extern std::atomic<bool> force_quit;

////////////////////////////////////////////////////////////
/// PT functions
////////////////////////////////////////////////////////////
inline const char * pt_timers_to_string(enum pt_timer_level x) {
    if(x == PT_TIMER_NO)
        return "No Timers";
    if(x == PT_TIMER_PIPELINE)
        return "Yes -- pipeline macro components";
    if(x == PT_TIMER_BATCH)
        return "Yes -- batch timers";
    if(x == PT_TIMER_ALL)
        return "Yes -- all";

    return "Unknown";
}

inline const char * pt_mepmem_to_string(enum lw_mempool_type x) {
    if(x == LW_MEMP_HOST_PINNED)
        return "Host pinned memory";
    if(x == LW_MEMP_DEVMEM)
        return "GPU memory";
    return "Unknown";
}

inline void do_throw(std::string const& what)
{
    throw std::runtime_error(what);
}

inline uint64_t get_ns() {
    struct timespec t;
    int ret;
    // ret = clock_gettime(CLOCK_MONOTONIC, &t);
    ret = clock_gettime(CLOCK_REALTIME, &t);
    if (ret != 0) {
        do_throw("clock_gettime failed");
    }
    return (uint64_t) t.tv_nsec + (uint64_t)t.tv_sec * 1000 * 1000 * 1000;
}

inline bool check_force_quit(std::atomic_bool &force_quit)
{
    return force_quit.load();
}

inline void set_force_quit(std::atomic_bool &force_quit)
{
    force_quit.store(true);
}

inline void unset_force_quit(std::atomic_bool &force_quit)
{
    force_quit.store(false);
}


#define CHECK_FORCE_QUIT_STRING(force_quit, x) if(check_force_quit(force_quit)) { pt_warn(x); goto err; }

////////////////////////////////////////////////////////////
/// Functions
////////////////////////////////////////////////////////////
int pt_get_number_ctx(void);
int pt_init(struct phytools_ctx * ptctx, int argc, char **argv);
int pt_finalize(struct phytools_ctx * ptctx);
int pt_print_ctx(struct phytools_ctx * ptctx);
char * pt_strtok(char *string, const char *delimiter);
int pt_parse_inputs(struct phytools_ctx * ptctx);
void * pt_alloc_aligned_memory(size_t input_size, size_t *out_size, size_t page_size);
int pt_parse_optargs(struct phytools_ctx *ptctx, int argc, char **argv, char *prgrname);
int flush_gmem(struct pipeline_ctx * plctx);
int set_mbatch_status(struct pipeline_ctx * plctx, int start_mbatch, int num_mbatch, int value);
double timerdiff_us(uint64_t t_end, uint64_t t_start);
double timerdiff_ns(uint64_t t_end, uint64_t t_start = 0);
double get_us_from_ns(double colwert_us);
void wait_ns(uint64_t ns);
void wait_s(int s);

int pt_setup_gpu(struct phytools_ctx * ptctx);
int pt_setup_ul_slot_table(struct phytools_ctx * ptctx);
int pt_setup_dl_slot_table(struct phytools_ctx * ptctx);
int pt_setup_ul_gpu_work(struct phytools_ctx * ptctx);
int pt_setup_dl_gpu_work(struct phytools_ctx * ptctx);
int pt_setup_rings(struct phytools_ctx * ptctx);
int pt_setup_ul_rings(struct phytools_ctx * ptctx);
int pt_setup_dl_rings(struct phytools_ctx * ptctx);
int pt_ul_finalize(struct phytools_ctx * ptctx);
int pt_dl_finalize(struct phytools_ctx * ptctx);
int pt_init(struct phytools_ctx * ptctx, int argc, char **argv);
int pt_print_ctx(struct phytools_ctx * ptctx);
int pt_prepare_cplane_messages(rte_unique_mbufs &mbufs_c, int mbufs_num, int other_ap_prbs,
                uint8_t ecpriSeqid_c, uint8_t frameId, uint8_t subFrameId, uint8_t slotId,
                int startPrbc, int numPrbc, int numSym, int startSym,
                enum oran_pkt_dir direction,
                struct rte_ether_addr src_eth_addr,
                struct pipeline_ctx * plctx);
int pt_dump_slot_buffer(std::string json_file, uint8_t * buffer, size_t buffer_size, size_t prb_sz, struct pipeline_ctx * plctx);
int pt_set_max_thread_priority();
int pt_get_thread_priority();
void pt_increase_slot(uint8_t& frame, uint8_t& subframe, uint8_t& slot);
uint32_t pt_checksum_adler32(uint8_t * i_buf, size_t i_elems);
int pt_dump_pusch_output(std::string json_file, uint8_t * buffer, size_t buffer_size);

//lwPHY
int lwphy_pusch_prepare(int gpu_id, int descramblingOn, struct pt_slot_table * slot_table_entry, std::string tv_file, lwdaStream_t stream);
int lwphy_configure_pusch_pipeline(PuschPHY * pusch_ul_phy, gnb_pars cell_params, std::vector<tb_pars> block_params);
int lwphy_pusch_slot(PuschPHY * pusch_ul_phy, int slot_num);
int lwphy_pusch_run(PuschPHY * pusch_ul_phy);
int lwphy_pusch_validate_crc(PuschPHY * pusch_ul_phy);
int lwphy_pusch_validate_input(PuschPHY * pusch_ul_phy); 
int lwphy_pdsch_prepare(int gpu_id, struct pt_slot_table * slot_table_entry, std::string tv_file, lwdaStream_t stream, int gpu_memory);
int lwphy_configure_pdsch_pipeline(PdschPHY * pdsch_dl_phy, 
                                    gnb_pars cell_params, 
                                    std::vector<tb_pars> block_params,
                                    uintptr_t input_buffer, size_t input_size);
int lwphy_pdsch_run(PdschPHY * pdsch_dl_phy);
int lwphy_pdsch_validate_crc(PdschPHY * pdsch_dl_phy);
int lwphy_pdcch_dl_301_prepare(int gpu_id, struct pt_slot_table * slot_table_entry,
			   std::string tv_file, lwdaStream_t stream, 
               uint32_t txOutSize, uint8_t * txOutBuf);
int lwphy_pdcch_dl_301a_prepare(int gpu_id, struct pt_slot_table * slot_table_entry,
			   std::string tv_file, lwdaStream_t stream, 
               uint32_t txOutSize, uint8_t * txOutBuf);
int lwphy_pdcch_dl_301b_prepare(int gpu_id, struct pt_slot_table * slot_table_entry,
			   std::string tv_file, lwdaStream_t stream, 
               uint32_t txOutSize, uint8_t * txOutBuf);
int lwphy_pdcch_dl_run(PdcchPHY *phy);
int lwphy_pdcch_ul_prepare(int gpu_id, struct pt_slot_table * slot_table_entry,
			   std::string tv_file, lwdaStream_t stream,
               uint32_t txOutSize, uint8_t * txOutBuf);
int lwphy_pdcch_ul_run(PdcchPHY *phy);
int lwphy_pbch_prepare(int gpu_id, struct pt_slot_table * slot_table_entry,
			   std::string tv_file, lwdaStream_t stream,
               uint32_t txOutSize, uint8_t * txOutBuf);
int lwphy_pbch_run(PbchPHY *phy);
int lwphy_finalize(struct phytools_ctx *ptctx);

//Standalone
int controller_core(void *param);

//Downlink
int downlink_processing_core(void *param);
int downlink_tx_core(void *param);

//Uplink
int uplink_timer_core(void *param);
int uplink_c_core(void *param);
int uplink_endpoint_core(void *param);
int uplink_prepare_core(void *param);
int uplink_rx_core(void *param);

//Controller
int lwphy_controller_init(std::string controller_file, struct phytools_ctx * _ptctx);
int lwphy_controller_finalize();

//LWCA kernels
extern "C"
void pt_prepare_slot_table_cache(uintptr_t * slot_table_cache_ptr, int table_size);

extern "C"
void pt_launch_pk_order(uint32_t * cache_count_prbs,  int tot_prbs, int prbs_per_symbol,
                    struct mbufs_batch * mbatch, uint32_t * mbufs_batch_ready_flags_d, 
                    uintptr_t * gbuf_table_cache_ptr, int max_mbufs_x_batch, int gbuf_index_pipeline,
                    int timer_mode, uint32_t * start_flag, uint32_t * order_flag, 
                    uint16_t *map_slot_to_last_mbatch,
                    uint8_t prev_frameId, uint8_t prev_subFrameId, uint8_t prev_slotId,
                    int lwda_blocks, int lwda_threads, lwdaStream_t stream);

extern "C"
void pt_launch_pk_copy(struct mbufs_batch * mbatch, uint32_t * mbufs_batch_ready_flags_d, 
                        int tot_pkts_x_batch,
                        int timer_mode, 
                        uint32_t * start_flag, uint32_t * order_flag,
                        int lwda_blocks, int lwda_threads, lwdaStream_t stream);

extern "C"
void pt_launch_kernel_write(uint32_t * addr, uint32_t value, lwdaStream_t stream);

extern "C"
void pt_launch_stream_write_kernel(uint32_t * addr, int value, int slot, lwdaStream_t stream);

extern "C"
void pt_launch_kernel_print(int index, uintptr_t addr1, uintptr_t addr2, size_t size, lwdaStream_t stream);

extern "C"
void pt_launch_checksum(uint8_t * i_buf, size_t i_elems, uint32_t * out, lwdaStream_t stream);

extern "C"
void pt_launch_check_crc(const uint32_t * i_buf, size_t i_elems, uint32_t * out, lwdaStream_t stream);

#endif
