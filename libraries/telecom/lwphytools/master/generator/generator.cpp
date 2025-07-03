/*
 * Copyright 2019-2020 LWPU Corporation.  All rights reserved.
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

/* A note about timing.
 * This version of the generator relies on CPU-side spinning to ensure that the
 * sends are perfomed at specified times (i.e. the sends are not posted before a
 * certain time has been reached). This is a stopgap solution until we can use
 * NIC offloads to perform the wait on the chip. As such, this solution is prone
 * to timing issues and should not be considered a foolproof solution.
 *
 * Beware of trying to get the time as precise and spot-on as possible; heavy
 * use of RDTSC(P) with CPUID will cause processor stalls, especially if called
 * in a loop; this can lead to odd behaviour. The TSC-based implementations have
 * been left for reference in generator.hpp but use them with care!
 */

#include <rte_eal.h>
#include <rte_errno.h>
#include <rte_malloc.h>
#include <rte_ethdev.h>
#include <stdexcept>
#include <memory>
#include <getopt.h>
#include "generator.hpp"
#include "hdf5hpp.hpp"

/* short option values corresponding to long options */
/*
enum {
	SHOPT_input = 1000, // non-printable
	SHOPT_max_pkt_size,
	SHOPT_slot_tti,
	SHOPT_slot_interval,
};
*/

#define GEN_USAGE "USAGE: \n" \
			  "    lwPHYTools_receiver [EAL arguments] -- [arguments] \n" \
			  "        Arguments documentation:\n" \
			  "            --json CONFIG_FILE      : mandatory, a JSON file with configuration parameters\n" \
		// "            --input INPUT_FILE         : mandatory, a text file with one TV per line\n" \
	// "            --max_pkt_size PKT_SIZE    : excludes 4-byte FCS. Default: " STR(DEFAULT_MAX_PKT_SIZE) "\n" \
	// "            --slot_tti US         : slot tti in microseconds. Default: 500\n" \
	// "            --slot_interval US         : interval between slots in microseconds. Default: 500\n" \
		  ""

/* global variables: options */
//std::unique_ptr
std::vector<std::string> opt_tv_file; //{nullptr};
std::vector<uint16_t> tv_flow_count;
size_t opt_max_pkt_size{DEFAULT_MAX_PKT_SIZE};
int opt_slot_tti_us{500};
int opt_ack_mode = 0;
int opt_tot_slots = 1000;
Json::Value jtree;
enum pt_timer_level opt_measure_time;
int opt_send_slot_by_slot=0; //0: symbol by symbol, 1: slot by slot

/* global variables */
std::vector<Slot> slots;
typedef std::unique_ptr<struct rte_mempool, decltype(&rte_mempool_free)> rte_unique_mp;
rte_unique_mp mp_hdr{nullptr, &rte_mempool_free};
rte_unique_mp mp_ext{nullptr, &rte_mempool_free};
rte_unique_mp mp_rxq{nullptr, &rte_mempool_free};
char oran_hdr_template[ORAN_IQ_HDR_SZ];
std::map<uint16_t, uint8_t> ecpriSeqid_map; // k: ecpriPcid, v: ecpriSeqid
inline uint8_t next_ecpriSeqid(uint16_t ecpriPcid) {
    auto it = ecpriSeqid_map.find(ecpriPcid);
    if (it == ecpriSeqid_map.end())
        ecpriSeqid_map[ecpriPcid] = 0;
    return ecpriSeqid_map[ecpriPcid]++;
}
int c_interval_us = 100;
struct dpdk_info dpdk;
struct rte_mbuf_ext_shared_info *shinfo;


static void init_config()
{
	dpdk.socket_id = 0;
	dpdk.port_id = PORT_ID;
	dpdk.vlan = 0;
	dpdk.memp_cache = 0;
	dpdk.memp_mbuf_num = 0;
	dpdk.mbuf_payload_size_rx = DEFAULT_MAX_PKT_SIZE;
	dpdk.mbuf_payload_size_tx = DEFAULT_MAX_PKT_SIZE;
	dpdk.mbuf_x_burst = 0;
}

static void parse_config_json(std::string json_file)
{
	int tot_input_files=0;

	jtree = jsonp_return_tree(json_file);
	if(jtree == Json::nullValue)
		do_throw("Error with file"+json_file);

	//Generator is mono-pipeline right now
	const Json::Value& jtree_info = jtree[JVALUE_PIPELINES][0];

	jsonp_assign_peer_eth(jtree_info, dpdk.peer_eth_addr);
	jsonp_assign_vlan(jtree_info, dpdk.vlan);
	jsonp_assign_port(jtree_info, dpdk.port_id);
	jsonp_assign_timers(jtree_info, opt_measure_time);

	//DPDK
	jsonp_assign_dpdk_burst(jtree_info, dpdk.mbuf_x_burst);
	jsonp_assign_dpdk_mbufs(jtree_info, dpdk.memp_mbuf_num);
	jsonp_assign_dpdk_payload_rx(jtree_info, dpdk.mbuf_payload_size_rx);
	jsonp_assign_dpdk_payload_tx(jtree_info, dpdk.mbuf_payload_size_tx);
	jsonp_assign_dpdk_cache(jtree_info, dpdk.memp_cache);
	opt_max_pkt_size = (size_t)dpdk.mbuf_payload_size_tx;
	enum pt_flow_ident_method fim;
	jsonp_flow_ident_method(jtree_info, fim);
	if (fim != PT_FLOW_IDENT_METHOD_VLAN)
		do_throw("Only VLAN flow identification method is supported.");
	//dpdk_port_conf.rxmode.max_rx_pkt_len = dpdk_mbuf_payload_total_size;

	jsonp_assign_ack(jtree_info, opt_ack_mode);
	jsonp_assign_tti(jtree_info, opt_slot_tti_us);
	jsonp_assign_slots(jtree_info, opt_tot_slots);
	jsonp_assign_send_slot(jtree_info, opt_send_slot_by_slot);
	jsonp_parse_tv_uplink(jtree_info, tot_input_files, opt_tv_file);
	if(tot_input_files <= 0)
		do_throw("No TV files in the JSON");
	jsonp_assign_cinterval(jtree_info, c_interval_us);
}

static inline int check_flow_exists(std::vector<uint16_t> flow_list, int flowId) {
	for(uint i = 0; i < flow_list.size(); ++i) {
		if(flowId == flow_list[i]) {
			return 0;
		}
	}
	return -1;
}

void get_options(int argc, char ** argv)
{
	std::string json_file;

	const char short_options[] =
		"j:"  /* JSON config file path */
		"h"   /* help */
	;

	static struct option opts[] =
	{
		{"json", required_argument, 0, SHOPT_json},
		{"help", no_argument,    NULL, SHOPT_help},
		{0, 0, 0, 0}
	};
	int option_index = 0;
	int c;
	while (-1 != (c = getopt_long(argc, argv, "", opts, &option_index))) {
		switch (c) {
		case SHOPT_json:
			json_file.assign(optarg);
			parse_config_json(json_file);
			break;
		case SHOPT_help:
			do_throw(GEN_USAGE);
			break;
		default:
			do_throw("Bad argument.\n" GEN_USAGE);
			break;
		}
	}
}

static double timerdiff_ns(uint64_t t_end, uint64_t t_start = 0) {
	return 1.e9*(double)((double)t_end - (double)t_start)/rte_get_tsc_hz();
}

static double timerdiff_us(uint64_t t_end, uint64_t t_start = 0) {
	return timerdiff_ns(t_end, t_start)/1.e3;
}

/* frame, subframe, slot Ids */
struct fssId {
	uint8_t frameId;
	uint8_t subframeId;
	uint8_t slotId;
};

struct tx_symbol_timers {
	/* measured by tx_symbol */
	uint64_t aux_alloc;
	uint64_t mbuf_alloc;
	uint64_t mbuf_setup;
	uint64_t spin;
	uint64_t tx;
	/* measured from the "outside" by tx_slot */
	uint64_t overall;
};

static int tx_symbol(Slot const& s, struct fssId const& fss, uint32_t sym_idx,
		     uint16_t flow_idx, uint16_t flow, uint64_t tstart, struct
		     tx_symbol_timers& tmrs)
{
	uint64_t rte_malloc_start_t, rte_malloc_end_t;
	uint64_t mp_alloc_start_t, mp_alloc_end_t;
	uint64_t mbuf_setup_start_t, mbuf_setup_end_t;
	uint64_t tx_start_t, tx_end_t;

	if(opt_measure_time > PT_TIMER_BATCH) {
		rte_malloc_start_t = get_ns();
	}

	struct rte_mbuf **mbufs_hdr = (struct rte_mbuf **)(s.mbufs_hdr_uniq[flow_idx][sym_idx]);
	struct rte_mbuf **mbufs_ext = (struct rte_mbuf **)(s.mbufs_ext_uniq[flow_idx][sym_idx]);

	/* ext buf shinfo
	 * we'll use the last entry in the array to keep the refcount of this
	 * entire allocation so that the last extbuf free callback can free
	 * this.
	 * It's a pity we can't cleanly pass a NULL callback to indicate we're
	 * working with read-only memory and we have to do this dance instead.
	 */
	shinfo[s.pkts_per_flow[flow_idx][sym_idx]].fcb_opaque = (void *)shinfo;
	rte_atomic16_set(&shinfo[s.pkts_per_flow[flow_idx][sym_idx]].refcnt_atomic, s.pkts_per_flow[flow_idx][sym_idx]);

	if(opt_measure_time > PT_TIMER_BATCH) {
		rte_malloc_end_t = mp_alloc_start_t = get_ns();
	}

	if (rte_pktmbuf_alloc_bulk(mp_hdr.get(), mbufs_hdr, s.pkts_per_flow[flow_idx][sym_idx]))
	do_throw(sb() << "Could not allocate " << s.pkts_per_flow[flow_idx][sym_idx] << " hdr mbufs.");

	if (rte_pktmbuf_alloc_bulk(mp_ext.get(), mbufs_ext, s.pkts_per_flow[flow_idx][sym_idx]))
	do_throw(sb() << "Could not allocate " << s.pkts_per_flow[flow_idx][sym_idx] << " ext mbufs.");

	if(opt_measure_time > PT_TIMER_BATCH) {
		mp_alloc_end_t = mbuf_setup_start_t = get_ns();
	}

	auto ext_cb = [](void *addr, void *opaque) {
		struct rte_mbuf_ext_shared_info *shinfo = (struct rte_mbuf_ext_shared_info *)opaque;
		/*
		int16_t refcnt = rte_atomic16_sub_return(&shinfo->refcnt_atomic, 1);
		if (refcnt == 0) {
			rte_free(shinfo->fcb_opaque);
		}
		*/
	};

	uint32_t remaining_prbs = s.prbs_per_flow[flow_idx][sym_idx];

	uint16_t prb_idx = 0;
	for (uint32_t i = 0; i < s.pkts_per_flow[flow_idx][sym_idx]; ++i) {
		struct rte_mbuf *hdr = mbufs_hdr[i];
		struct rte_mbuf *ext = mbufs_ext[i];

		hdr->nb_segs = 2;
		hdr->next = ext;

		/* attach external buffer */
		uint16_t num_prbs = RTE_MIN(remaining_prbs, s.prbs_per_pkt[flow_idx][sym_idx]);
		uint16_t buf_len =  num_prbs * PRB_SIZE_16F;
		void *ext_ptr = get_prb(s, flow_idx, sym_idx, prb_idx);
		shinfo[i].fcb_opaque = (void *)&shinfo[s.pkts_per_flow[flow_idx][sym_idx]];
		shinfo[i].free_cb = ext_cb;
		rte_atomic16_set(&shinfo[i].refcnt_atomic, 1);
		rte_pktmbuf_attach_extbuf(ext, ext_ptr, RTE_BAD_IOVA,
					  buf_len, &shinfo[i]);

		/* setup the headers */
		rte_memcpy(rte_pktmbuf_mtod(hdr, void *), oran_hdr_template,
			   ORAN_IQ_HDR_SZ);

		struct oran_eth_hdr *oran_eth = rte_pktmbuf_mtod(hdr, struct oran_eth_hdr *);
		oran_eth->vlan_hdr.vlan_tci = rte_cpu_to_be_16(
			rte_be_to_cpu_16(oran_eth->vlan_hdr.vlan_tci) + flow);

		struct oran_ecpri_hdr *ecpri =
			rte_pktmbuf_mtod_offset(hdr, struct oran_ecpri_hdr *,
						(sizeof(struct rte_ether_hdr)
						+ sizeof(struct rte_vlan_hdr)));
        ecpri->ecpriSeqid       = next_ecpriSeqid(flow);
		ecpri->ecpriPcid 		= rte_cpu_to_be_16(flow);

		struct oran_umsg_iq_hdr *iq_df =
			rte_pktmbuf_mtod_offset(hdr, struct oran_umsg_iq_hdr *,
						(sizeof(struct rte_ether_hdr)
						+ sizeof(struct rte_vlan_hdr)
						+ sizeof(struct oran_ecpri_hdr)));
		iq_df->frameId 			= fss.frameId;
		iq_df->subframeId 		= fss.subframeId;
		iq_df->slotId 			= fss.slotId;
		iq_df->symbolId 		= (uint8_t)sym_idx; //not working properly

		struct oran_u_section_uncompressed *u_sec =
			rte_pktmbuf_mtod_offset(hdr, struct oran_u_section_uncompressed *,
						(sizeof(struct rte_ether_hdr)
						+ sizeof(struct rte_vlan_hdr)
						+ sizeof(struct oran_ecpri_hdr)
						+ sizeof(struct oran_umsg_iq_hdr)));

		u_sec->sectionId 		= ORAN_DEF_SECTION_ID;
		u_sec->rb 			= 0;
		u_sec->symInc 			= 0;
		u_sec->startPrbu 		= prb_idx;

		u_sec->numPrbu 			= num_prbs;
		/* fix lengths */
		hdr->data_len 			= ORAN_IQ_HDR_SZ;
		ext->data_len 			= buf_len;
        hdr->pkt_len            = hdr->data_len + ext->data_len;
		ecpri->ecpriPayload 		= rte_cpu_to_be_16(hdr->pkt_len-ORAN_IQ_HDR_OFFSET+4);

		/* update counters */
		remaining_prbs -= num_prbs;
		prb_idx += num_prbs;
	}
	// pt_info("Umsg, s.prbs_per_pkt[%d]=%d prb_idx=%d s.num_pkts[sym_idx]=%d\n", sym_idx, s.prbs_per_pkt[sym_idx], prb_idx, s.num_pkts[sym_idx]);
	// oran_dump_umsg_hdrs((struct oran_umsg_hdrs *) ((uint8_t*)rte_pktmbuf_mtod_offset(mbufs_hdr[s.num_pkts[sym_idx]-1], uint8_t*, 0)));

	if(opt_measure_time > PT_TIMER_BATCH) {
		mbuf_setup_end_t = tx_start_t = get_ns();
	}

	uint16_t nb_tx = s.pkts_per_flow[flow_idx][sym_idx];

	while (nb_tx && !check_force_quit()) {
		nb_tx -= rte_eth_tx_burst(dpdk.port_id, 0,
					  &mbufs_hdr[s.pkts_per_flow[flow_idx][sym_idx] - nb_tx], nb_tx);
		rte_wmb();
	}

	if (check_force_quit())
	{
		pt_info("Quit forced during TX function: Symbol %d Slot %d nb_tx %d\n", sym_idx, fss.slotId, nb_tx);
		return -1;
	}
		// do_throw(sb() << "force quit while TXing");
	if(opt_measure_time > PT_TIMER_BATCH) {
		tx_end_t = get_ns();
		tmrs.aux_alloc += rte_malloc_end_t - rte_malloc_start_t;
		tmrs.mbuf_alloc += mp_alloc_end_t - mp_alloc_start_t;
		tmrs.mbuf_setup += mbuf_setup_end_t - mbuf_setup_start_t;
		tmrs.tx += tx_end_t - tx_start_t;
	}
	return 0;

}

static int tx_slot(Slot const& s, struct fssId const& fss, uint8_t startSymbol, std::vector<uint16_t> flow_list, uint64_t tstart, uint64_t symbol_tti_ns)
{
	uint64_t symbols_start_t, symbols_end_t;
	uint64_t spin_start_t, spin_end_t;
	uint64_t tx_slot_start_t, tx_slot_end_t;

	tx_slot_start_t = symbols_start_t = get_ns();


	struct tx_symbol_timers* tmrs = (struct tx_symbol_timers* )rte_zmalloc(NULL, sizeof(struct tx_symbol_timers) * SLOT_NUM_SYMS, 0);
	for (uint8_t sym_idx = startSymbol; sym_idx < SLOT_NUM_SYMS; ++sym_idx) {
		for (uint8_t flow_idx = 0; flow_idx < flow_list.size(); ++flow_idx) {
			if (tx_symbol(s, fss, sym_idx, flow_idx, flow_list[flow_idx], tstart, tmrs[sym_idx])) {
				return -1;
			}
		}
		
		// TX symbol by symbol timer for next symbol to tx after symbol_tti_ns
		
		spin_start_t = spin_end_t = get_ns();
		if(opt_send_slot_by_slot == 0) {
			while ((symbols_end_t = spin_end_t = get_ns()) < symbols_start_t + symbol_tti_ns) {
				for (int spin_cnt = 0; spin_cnt < 500; ++spin_cnt) {
					__asm__ __volatile__ ("");
				}
			}
		}
		symbols_end_t = get_ns();
		tmrs[sym_idx].spin = spin_end_t-spin_start_t;
		tmrs[sym_idx].overall = symbols_end_t-symbols_start_t;
		symbols_start_t = get_ns();

	}


	if(opt_measure_time > PT_TIMER_NO) {
		tx_slot_end_t = get_ns();
	}

	if(opt_measure_time > PT_TIMER_BATCH) {
		for (uint8_t sym_idx = startSymbol; sym_idx < SLOT_NUM_SYMS; ++sym_idx) {
			pt_info("Symbol %d ==> TX time: %4.2f, Spin: %4.2f, Mbuf alloc: %4.2f, Mbuf setup: %4.2f\n",
					sym_idx,
					((double)tmrs[sym_idx].overall)/1000,
					((double)tmrs[sym_idx].spin)/1000,
					((double)tmrs[sym_idx].mbuf_alloc)/1000, 
					((double)tmrs[sym_idx].mbuf_setup)/1000);
		}
	}

	if(opt_measure_time > PT_TIMER_NO) {
		pt_info("Tot tx slot time: %4.2f\n\n", ((double)(tx_slot_end_t-tx_slot_start_t))/1000);
	}
	
	rte_free((uint8_t*)tmrs);

	while ((get_ns()) < tx_slot_start_t + (opt_slot_tti_us * 1000)) {
		for (int spin_cnt = 0; spin_cnt < 500; ++spin_cnt) {
			__asm__ __volatile__ ("");
		}
	}

	if(opt_measure_time > PT_TIMER_NO) {
		pt_info("End slot time: %4.2f\n\n", ((double)(get_ns()-tx_slot_start_t))/1000);
	}

	return 0;
}

static int tx_core_throw(__rte_unused void *arg) {
	int nb_mbufs_rx=0, nb_mbufs_to_process = 0;
	uint8_t startSymbol=0, numPrbc=0;
	uint16_t startPrbc=0, flowId=0;
	uint64_t cmsg_start_t, loop_end_t, loop_t, slot_start_t;
	struct rte_mbuf *mbufs_rx[512], *mbufs_to_process[512];

	std::vector<uint16_t> flow_list;
	flow_list.reserve(PT_MAX_FLOWS_X_PIPELINE);


	pt_dbg("Starting TX core (app thread %u) - socket %u. "
		   "Thread is affinitized to CPUs %s. "
		   "Thread priority is %d\n",
		   rte_lcore_id(), rte_socket_id(), affinity_cpu_list().c_str(),
		   set_max_thread_priority());
	uint64_t symbol_tti_ns = opt_slot_tti_us * 1000 / SLOT_NUM_SYMS;
	// give ourselves 10 ms of slack
	uint64_t beginning_of_time_ns = get_ns() + 1000 * 1000 * 10;
	pt_dbg("symbol_tti_ns: %" PRIu64 " tot slots: %d\n", symbol_tti_ns, opt_tot_slots);

	size_t shinfo_arr_sz = sizeof(struct rte_mbuf_ext_shared_info) * 1024;
	shinfo = (struct rte_mbuf_ext_shared_info *)rte_zmalloc(NULL, shinfo_arr_sz, 64);
	if (shinfo == NULL)
		do_throw(sb() << "Could not allocate an array for shinfo.");

	uint64_t counter = 0;
	struct fssId fss{0,0,0};

	while (!check_force_quit()) {
        uint16_t slot_dset_idx = counter % slots.size();
#if 0
		if(counter) {
			set_force_quit();
			break;
		}
#endif
		////////////////////////////////////////////////
		//// Receive C-msg
		////////////////////////////////////////////////
		flow_list.clear();
		flow_list.reserve(PT_MAX_FLOWS_X_PIPELINE);
		while (!check_force_quit() && nb_mbufs_rx < tv_flow_count[0] * PT_PKT_X_CMSG /* - PT_DRIVER_MIN_RX_PKTS */) {
			nb_mbufs_rx += rte_eth_rx_burst(dpdk.port_id, 0, //RX queue 
									  (struct rte_mbuf **) (&(mbufs_rx[nb_mbufs_rx])),
									  RTE_MAX(tv_flow_count[0] * PT_PKT_X_CMSG - nb_mbufs_rx, PT_DRIVER_MIN_RX_PKTS)
			);
		}
		if(check_force_quit())
			break;

        // we might receive C-plane messages for the next slot above, leave
        // those for processing later, sort of "buffering" them
        nb_mbufs_to_process = tv_flow_count[0] * PT_PKT_X_CMSG;
        for (size_t i = 0; i < nb_mbufs_to_process; ++i) {
            mbufs_to_process[i] = mbufs_rx[i];
        }
        for (size_t i = nb_mbufs_to_process; i < nb_mbufs_rx; ++i) {
            mbufs_rx[i-nb_mbufs_to_process] = mbufs_rx[i];
        }
        nb_mbufs_rx -= nb_mbufs_to_process;
		// pt_info("RX CMSG %d\n", nb_mbufs_to_process);
		// oran_dump_cmsg_hdrs((struct oran_cmsg_uldl_hdrs *)(uint8_t *)rte_pktmbuf_mtod_offset(mbufs_rx[0], uint8_t*, 0));
			// nb_mbufs_to_process/2 index should ensures we're not using old packets not received from the previous TTI..
		
		//Go through each c-msg to read all of the unique flowIDs -> need improvement, may read from the wrong TTI
		for(int index_rx = 0; index_rx < nb_mbufs_to_process; ++index_rx)
		{
			flowId = oran_cmsg_get_flowid((uint8_t *)rte_pktmbuf_mtod_offset(mbufs_to_process[index_rx], uint8_t*, 0));
			if(check_flow_exists(flow_list, flowId)) {
				flow_list.emplace_back(flowId);
			}
		}

        if (flow_list.size() != tv_flow_count[slot_dset_idx]) {
            do_throw(sb() << "Receiver requested " << flow_list.size() << " flows, which is different "
                    "from what TV #" << slot_dset_idx << " contains, i.e. " << tv_flow_count[slot_dset_idx]);
        }

		if(check_force_quit()) {
			pt_info("Signal received while reading flow ids.\n");
		 }

		fss.frameId = oran_cmsg_get_frame_id((uint8_t *)rte_pktmbuf_mtod_offset(mbufs_to_process[nb_mbufs_to_process/2], uint8_t*, 0));
		fss.subframeId = oran_cmsg_get_subframe_id((uint8_t *)rte_pktmbuf_mtod_offset(mbufs_to_process[nb_mbufs_to_process/2], uint8_t*, 0));
		fss.slotId = oran_cmsg_get_slot_id((uint8_t *)rte_pktmbuf_mtod_offset(mbufs_to_process[nb_mbufs_to_process/2], uint8_t*, 0));
		startSymbol = oran_cmsg_get_startsymbol_id((uint8_t *)rte_pktmbuf_mtod_offset(mbufs_to_process[nb_mbufs_to_process/2], uint8_t*, 0));
		startPrbc = oran_cmsg_get_startprbc((uint8_t *)rte_pktmbuf_mtod_offset(mbufs_to_process[nb_mbufs_to_process/2], uint8_t*, 0));
		numPrbc = oran_cmsg_get_numprbc((uint8_t *)rte_pktmbuf_mtod_offset(mbufs_to_process[nb_mbufs_to_process/2], uint8_t*, 0));

		if(startPrbc != 0)
			do_throw(sb() << "startPrbc (" << (int) startPrbc
				 << ") is not 0, not supported yet!");
		if(numPrbc != 0)
			do_throw(sb() << "numPrbc (" << (int) numPrbc
				 << ") is not 0 (ALL), not supported yet!");

		for(int index_rx=0; index_rx < nb_mbufs_to_process; index_rx++)
			rte_pktmbuf_free((struct rte_mbuf *)(mbufs_to_process[index_rx]));

		////////////////////////////////////////////////
		//// Wait for c_interval_us after the CMSG
		////////////////////////////////////////////////
		if(opt_measure_time > PT_TIMER_NO)
			cmsg_start_t = get_ns();

		loop_end_t = get_ns() + (c_interval_us * 5000);
		while ((loop_t = get_ns()) < loop_end_t) {
			for (int spin_cnt = 0; spin_cnt < 1000; ++spin_cnt) {
				__asm__ __volatile__ ("");
			}
		}

		if(opt_measure_time > PT_TIMER_NO)
			pt_info("Wait after C-msg: %4.2f\n", ((double)(get_ns()-cmsg_start_t))/1000);

		////////////////////////////////////////////////
		//// Send UMSG
		////////////////////////////////////////////////

		Slot const& s = slots[slot_dset_idx];
		/*
		 --------------------- ------------
		|         TTI         | cmsg timer |
		 --------------------- ------------
		*/
		uint64_t t = get_ns() + symbol_tti_ns;
		printf("==========> Sending frame %d subframe %d slot %d (slot count %" PRIu64 ") startSymbol %d Send slot by slot %d\n", 
			fss.frameId, fss.subframeId, fss.slotId, counter, startSymbol, opt_send_slot_by_slot);
        fflush(stdout);
		try {

			tx_slot(s, fss, startSymbol, flow_list, t, symbol_tti_ns);

		} catch (std::runtime_error &e) {
			do_throw(sb() << e.what() << " Slot counter: "
				 << counter << ".");
		}

		++counter;

		if(opt_tot_slots > 0)
		{
			if(counter >= opt_tot_slots)
			{
				set_force_quit();
				break;
			}

		}
	}

	if(opt_tot_slots > 0)
		pt_dbg("Exiting after %" PRIu64 " slots\n", counter);
	else
		pt_dbg("Exiting\n");

	//Assuming only 1 TX queue for the moment
	struct rte_eth_stats stats;
	rte_eth_stats_get(dpdk.port_id, &stats);
	printf("\nDPDK Stats:\n");
		// printf("\tQueue 0 -> packets = %ld bytes = %ld\n", stats.q_opackets[0], stats.q_obytes[0]);
		// printf("\t-------------------------------------------\n");
		printf("\tTot TX packets: %ld, Tot TX bytes: %ld\n", stats.opackets, stats.obytes);
		// printf("\tQueue 0 -> packets = %ld bytes = %ld dropped = %ld\n", stats.q_ipackets[0], stats.q_ibytes[0], stats.q_errors[0]);
		// printf("\t-------------------------------------------\n");
		printf("\tTot RX packets: %ld Tot RX bytes: %ld\n", stats.ipackets, stats.ibytes);

		printf("DPDK errors:\n");
		printf("\tRX packets dropped by the HW (RX queues are full) = %" PRIu64 "\n", stats.imissed);
		printf("\tTotal number of erroneous RX packets = %" PRIu64 "\n", stats.ierrors);
		printf("\tTotal number of RX mbuf allocation failures = %" PRIu64 "\n", stats.rx_nombuf);
		printf("\tTotal number of failed TX packets = %" PRIu64 "\n", stats.oerrors);
		printf("\n");

	return EXIT_SUCCESS;
}

static int tx_core(void *arg) {
	try {
		return tx_core_throw(arg);
	} catch (std::exception& e) {
		rte_exit(EXIT_FAILURE, "Exception: %s\n", e.what());
	} catch (...) {
		rte_exit(EXIT_FAILURE, "Unknown exception");
	}
}


int main_throw(int argc, char ** argv)
{
	std::vector<dataset_map> datasets;


	signal_setup();

	// Initialize DPDK
	int eal_args = rte_eal_init(argc, argv);
	if (eal_args < 0)
		do_throw(std::string("EAL init failed: ") + rte_strerror(rte_errno));

	// Get our options
	init_config();
	get_options(argc - eal_args, argv + eal_args);
	print_max_pkt_size_info(opt_max_pkt_size);
	pt_dbg("A slot tti of %u us has been chosen, and thus %u ns is used as a rough approximation of symbol tti (incl. CP)\n",
		opt_slot_tti_us, (1000 * opt_slot_tti_us) / SLOT_NUM_SYMS);

	// Load the requested test vectors

	tv_flow_count = std::vector<uint16_t>(opt_tv_file.size());

	for (int i = 0; i < opt_tv_file.size(); ++i) {
		dataset_map d = load_test_vectors(opt_tv_file[i], {"DataRx"}, tv_flow_count[i]);
		slots.push_back(dataset_to_slot(std::move(d["DataRx"]), tv_flow_count[i]));
		slot_dma_map(dpdk.port_id, slots.back());
	}

	// Create a mempool for headers (eth, vlan, eCPRI, O-RAN).
	mp_hdr.reset(rte_pktmbuf_pool_create("mp_hdr", MP_HDR_CNT, MP_HDR_CACHE_SZ, 0,
						 MP_HDR_SZ, rte_socket_id()));
	// Create a mempool for mbufs with attached extbufs (PRBs)
	mp_ext.reset(rte_pktmbuf_pool_create("mp_ext", MP_EXT_CNT, MP_EXT_CACHE_SZ, 0,
						 MP_EXT_SZ, rte_socket_id()));
	// Create a mempool for RX queue
	mp_rxq.reset(rte_pktmbuf_pool_create("mp_rxq", MP_RXQ_CNT, MP_RXQ_CACHE_SZ, 0,
						 MP_RXQ_SZ, rte_socket_id()));

	// DMA-register the hdr mempool which is not getting used for RX
	//    so that it can be used for TX
	mempool_dma_map(dpdk.port_id, mp_hdr.get());

	rte_eth_macaddr_get(dpdk.port_id, &(dpdk.src_eth_addr));

	// Prepare a template for all the headers
	fill_hdr_template(oran_hdr_template, dpdk.vlan, dpdk.peer_eth_addr, dpdk.src_eth_addr);

	start_eth(dpdk.port_id, dpdk.vlan, 1, 1, 8192, 8192, mp_rxq.get());

	pt_dbg("Master core (app thread %u, socket %u) is affinitized to CPUs %s\n",
		   rte_lcore_id(), rte_socket_id(), affinity_cpu_list().c_str());

	
/* 	
	_________________________________________________________________________________________________________________
	|						Antenna 1						|						Antenna 2						|
	| 		  Sym0			|  ...	|		  Sym13			| 		  Sym0			|  ...	|		  Sym13			|
	|prb0	|  ...	|prb271	|		|prb0	|  ...	|prb271	|prb0	|  ...	|prb271	|		|prb0	|  ...	|prb271	|
	|_______________________________________________________|_______________________________________________________|
	
*/

	//Here we assume same-size packets (same number of PRBs) for every symbol
	for (auto &slot: slots) {
		for (uint8_t sym_idx = 0; sym_idx < SLOT_NUM_SYMS; ++sym_idx) {
			slot.pkts_per_sym[sym_idx] = 0;

			for (uint8_t flow_idx = 0; flow_idx < tv_flow_count[0]; ++flow_idx) {
				slot.prbs_per_pkt[flow_idx][sym_idx] = ((opt_max_pkt_size - ORAN_IQ_HDR_SZ) / PRB_SIZE_16F);
				slot.pkts_per_flow[flow_idx][sym_idx] = (slot.prbs_per_flow[flow_idx][sym_idx] + slot.prbs_per_pkt[flow_idx][sym_idx] - 1) / slot.prbs_per_pkt[flow_idx][sym_idx];
				slot.pkts_per_sym[sym_idx] += slot.pkts_per_flow[flow_idx][sym_idx];
				
				slot.mbufs_arr_sz[flow_idx][sym_idx] = sizeof(struct rte_mbuf *) * slot.pkts_per_flow[flow_idx][sym_idx];

				slot.mbufs_hdr_uniq[flow_idx][sym_idx] = (uintptr_t)(rte_zmalloc(NULL, slot.mbufs_arr_sz[flow_idx][sym_idx], 0));
				if (slot.mbufs_hdr_uniq[flow_idx][sym_idx] == 0)
					do_throw(sb() << "Could not allocate an array for " << slot.pkts_per_flow[flow_idx][sym_idx] << " hdr mbufs.");

				slot.mbufs_ext_uniq[flow_idx][sym_idx] = (uintptr_t)(rte_zmalloc(NULL, slot.mbufs_arr_sz[flow_idx][sym_idx], 0));
				if (slot.mbufs_ext_uniq[flow_idx][sym_idx] == 0)
					do_throw(sb() << "Could not allocate an array for " << slot.pkts_per_flow[flow_idx][sym_idx] << " ext mbufs.");


			}
			pt_info("Symbol %d: %d packets, %d PRBs\n", sym_idx, slot.pkts_per_sym[sym_idx], slot.prbs_per_sym[sym_idx]);
		}
	}

	// just one pipeline for now; when extending to more pipelines,
	// will need to make many options per-pipeline and pass a pipeline
	// configuration to the generator


	unsigned int lcore = rte_get_next_lcore(0, 1, 0);
	rte_eal_remote_launch(tx_core, NULL, lcore);

	RTE_LCORE_FOREACH_SLAVE(lcore) {
		if (rte_eal_wait_lcore(lcore) < 0) {
			pt_err("bad exit for lcore: %d\n", lcore);
			break;
		}
	}

	for (auto &slot : slots) {
		for (uint8_t flow_idx; flow_idx < tv_flow_count[0]; ++ flow_idx) {
			for (uint8_t sym_idx = 0; sym_idx < SLOT_NUM_SYMS; ++sym_idx) {
				rte_free((uint8_t*)slot.mbufs_hdr_uniq[flow_idx][sym_idx]);
				rte_free((uint8_t*)slot.mbufs_ext_uniq[flow_idx][sym_idx]);
			}
		}
		
	}
	rte_free((void*)shinfo);
	return EXIT_SUCCESS;
}

int main(int argc, char **argv)
{
	try {
		return main_throw(argc, argv);
	} catch (std::exception& e) {
		pt_err("%s\n", e.what());
	} catch (...) {
		pt_err("Uncaught exception");
	}
	return EXIT_FAILURE;
}
