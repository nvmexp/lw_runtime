/*
 * Copyright 2019-2020 LWPU Corporation. All rights reserved.
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

#include "generator.hpp"
#include <atomic>
#include <rte_errno.h>
#include <rte_ethdev.h>
#include <signal.h>
#include <sched.h>
#include <fstream>
#include "hdf5hpp.hpp"


// The main "everybody quit" switch. Can be triggered by any core.
static std::atomic_bool force_quit{0};

bool check_force_quit()
{
	return force_quit.load();
}

void set_force_quit()
{
	force_quit.store(true);
}

void signal_handler(int signum)
{
    if (signum == SIGINT || signum == SIGTERM || signum == SIGUSR1) {
        if (check_force_quit()) {
            rte_exit(EXIT_FAILURE, "Signal %d received; quitting the hard way\n", signum);
        }
        pt_warn("Signal %d received, preparing to exit...\n", signum);
	set_force_quit();
    }
}

void signal_setup()
{
	signal(SIGINT, signal_handler);
	signal(SIGTERM, signal_handler);
	signal(SIGUSR1, signal_handler);
}

static void _dma_map(uint16_t port_id, void *addr, size_t size)
{
	struct rte_device *dev = rte_eth_devices[port_id].device;
	pt_dbg("Registering 0x%p : %zu with device %s for DMA\n",
	       addr, size, dev->name);
	int r = rte_dev_dma_map(dev, addr, RTE_BAD_IOVA, size);
	if (r != 0)
		do_throw(std::string("EAL init failed: ") +
					 rte_strerror(rte_errno));
}

void slot_dma_map(uint16_t port_id, Slot const& s)
{
	_dma_map(port_id, s.raw_data.data.get(), s.raw_data.size);
}

void mempool_dma_map(uint16_t port_id, struct rte_mempool *mp)
{
	// traverse memsegs
	auto cb = [](struct rte_mempool *mp, void *opaque,
		      struct rte_mempool_memhdr *memhdr, unsigned mem_idx) -> void {
		uint16_t port_id = *((uint16_t *)opaque);
		pt_dbg("Registering segment %u of mempool %s\n", mem_idx,
		       mp->name);
		_dma_map(port_id, memhdr->addr, memhdr->len);
	};
	rte_mempool_mem_iter(mp, cb, (void *)&port_id);
}

#define offset_as(b, t, o) ((t)((char *)b + (o)))

int fill_hdr_template(char *buf, uint16_t vlan_tci, struct rte_ether_addr dst_eth_addr, struct rte_ether_addr src_eth_addr)
{
	memset(buf, 0, ORAN_IQ_HDR_SZ);
	// struct ether_hdr *eth = offset_as(buf, struct ether_hdr *, 0);
	// struct vlan_hdr *vlan = offset_as(eth, struct vlan_hdr *, sizeof(*eth));
	// struct oran_ecpri_hdr *ecpri = offset_as(vlan, struct oran_ecpri_hdr *, sizeof(*vlan));
	// struct oran_umsg_iq_hdr *iq_df = offset_as(ecpri, struct oran_umsg_iq_hdr *, sizeof(*ecpri));
	
	if(oran_fill_eth_vlan_hdr(
		(struct oran_eth_hdr *)(&(buf[0])),
		src_eth_addr, dst_eth_addr, vlan_tci))
		do_throw("oran_fill_eth_vlan_hdr error");

	if(oran_fill_ecpri_hdr(
		//(struct oran_ecpri_hdr *) (&(offset_as(vlan, struct oran_ecpri_hdr *, sizeof(*vlan))[0])),
		(struct oran_ecpri_hdr *)(&(buf[ORAN_ETH_HDR_SIZE])),
		0, //payloadSize
		0, //initial flow id
		0, //ecpriSeqid
		ECPRI_MSG_TYPE_IQ)
	)
		do_throw("oran_fill_ecpri_hdr error");

	if(oran_fill_umsg_iq_hdr((struct oran_umsg_iq_hdr *)(&(buf[ORAN_IQ_HDR_OFFSET])), 
		DIRECTION_UPLINK,
		0, 0, 0, 0))
		do_throw("oran_fill_umsg_iq_hdr error");

	// TODO check completeness if those are all the fields we can set ahead
	// of all time; there will be other fields that we can set per-slot and
	// per-symbol

	return PT_OK;
}

void print_max_pkt_size_info(size_t max_pkt_sz)
{
	size_t pkt_overhead = ORAN_IQ_HDR_SZ;
	size_t prbs_per_pkt = (max_pkt_sz - pkt_overhead) / PRB_SIZE_16F;
	pt_dbg("Max packet size: %zu bytes. Headers overhead: %zu bytes\n",
	       max_pkt_sz, pkt_overhead);
	pt_dbg("At most %zu PRBs per packet. Effective max packet size: %zu bytes\n",
		prbs_per_pkt, pkt_overhead + prbs_per_pkt * PRB_SIZE_16F);
}

static void print_nic_info(uint16_t port_id, std::string const& pre,
			   std::string const& post)
{
	struct rte_eth_dev_info dev_info;
	rte_eth_dev_info_get(port_id, &dev_info);
	pt_dbg("%sNIC %s managed by PMD \"%s\"%s\n", pre.c_str(),
	       dev_info.device->name, dev_info.driver_name, post.c_str());
}

// Only setting a requirement on VLAN tci and ethtype within the VLAN
// NOT checking the ethernet addresses per se
struct rte_flow *setup_rules(uint16_t port_id, uint16_t vlan_tci)
{
	struct rte_flow_attr attr;
	struct rte_flow_item patterns[3];
	struct rte_flow_action actions[2];
	struct rte_flow_error err;
	struct rte_flow_action_queue queue = {.index = 0 };
	struct rte_flow_item_eth eth_spec, eth_mask;
	struct rte_flow_item_vlan vlan_spec, vlan_mask;

	memset(&attr, 0, sizeof(attr));
	memset(patterns, 0, sizeof(patterns));
	memset(actions, 0, sizeof(actions));
	memset(&eth_spec, 0, sizeof(eth_spec));
	memset(&eth_mask, 0, sizeof(eth_mask));
	memset(&vlan_spec, 0, sizeof(vlan_spec));
	memset(&vlan_mask, 0, sizeof(vlan_mask));

	attr.ingress = 1;

	eth_spec.type = rte_cpu_to_be_16(RTE_ETHER_TYPE_VLAN);
	eth_mask.type = 0xffff;

	vlan_spec.tci = rte_cpu_to_be_16(vlan_tci);
	vlan_mask.tci = rte_cpu_to_be_16(0x0fff); /* lower 12 bits only */

	vlan_spec.inner_type = rte_cpu_to_be_16(ETHER_TYPE_ECPRI);
	vlan_mask.inner_type = 0xffff;

	actions[0].type = RTE_FLOW_ACTION_TYPE_QUEUE;
	actions[0].conf = &queue;
	actions[1].type = RTE_FLOW_ACTION_TYPE_END;

	patterns[0].type = RTE_FLOW_ITEM_TYPE_ETH;
	patterns[0].spec = &eth_spec;
	patterns[0].mask = &eth_mask;
	patterns[1].type = RTE_FLOW_ITEM_TYPE_VLAN;
	patterns[1].spec = &vlan_spec;
	patterns[1].mask = &vlan_mask;
	patterns[2].type = RTE_FLOW_ITEM_TYPE_END;

	if (rte_flow_validate(port_id, &attr, patterns, actions, &err))
		rte_panic("Invalid flow rule: %s\n", err.message);
	return rte_flow_create(port_id, &attr, patterns, actions, &err);
}

void start_eth(uint16_t port_id, uint16_t vlan, uint16_t rxq, uint16_t txq, uint16_t rxd,
		uint16_t txd, struct rte_mempool *rxmp)
{
	print_nic_info(port_id, "Starting ", sb() << ", using mempool "
		       << rxmp->name << " for RX; " << rxq << "RX queues, "
		       << txq << "TX queues.");
	struct rte_flow_error flowerr;
	if (rte_flow_isolate(port_id, 1, &flowerr))
		rte_panic("Flow isolation failed: %s\n", flowerr.message);

	int ret;
	const unsigned sid = rte_socket_id();

	struct rte_eth_conf eth_conf;
	memset(&eth_conf, 0, sizeof(eth_conf));
	eth_conf.txmode.offloads = DEV_TX_OFFLOAD_MULTI_SEGS;
        if (0 != (ret = rte_eth_dev_configure(port_id, rxq, txq, &eth_conf)))
		do_throw(sb() << "rte_eth_dev_configure returned " << ret);

        if (0 != (ret = rte_eth_dev_adjust_nb_rx_tx_desc(port_id, &rxd, &txd)))
		do_throw(sb() << "rte_eth_dev_adjust_nb_rx_tx_desc returned " << ret);

	struct rte_eth_txconf tx_conf;
	memset(&tx_conf, 0, sizeof(tx_conf));
	tx_conf.offloads = DEV_TX_OFFLOAD_MULTI_SEGS;
	for (uint16_t i = 0; i < txq; ++i) {
		if (0 != (ret = rte_eth_tx_queue_setup(port_id, i, txd, sid, &tx_conf)))
			do_throw(sb() << "rte_eth_tx_queue_setup returned " << ret);
	}

	struct rte_eth_rxconf rx_conf;
	memset(&rx_conf, 0, sizeof(rx_conf));
	for (uint16_t i = 0; i < rxq; ++i) {
		if (0 != (ret = rte_eth_rx_queue_setup(port_id, i, rxd, sid, &rx_conf, rxmp)))
			do_throw(sb() << "rte_eth_rx_queue_setup returned " << ret);
	}

	if (0 != (ret = rte_eth_dev_start(port_id)))
		do_throw(sb() << "rte_eth_dev_start returned " << ret);

	struct rte_flow *f = setup_rules(port_id, vlan);
	if (f == NULL)
		do_throw(sb() << "Could not set up receive flow control rule");
}

std::string affinity_cpu_list()
{
	sb out;
	cpu_set_t s;
	int ret = pthread_getaffinity_np(pthread_self(), sizeof(s), &s);
	if (ret != 0)
		do_throw("pthread_getaffinity_np() failed");
	int count = CPU_COUNT(&s);
	int printed = 0;
	printf("CPU count is %d\n", count);
	for (int i = 0; printed < count; ++i) {
		if (CPU_ISSET(i, &s)) {
			if (printed)
				out << "," << i;
			else
				out << i;
			++printed;
		}
	}
	return out;
}

int set_max_thread_priority()
{
	pthread_t t = pthread_self();
	struct sched_param schedprm;
	schedprm.sched_priority = sched_get_priority_max(SCHED_FIFO);
	int ret = pthread_setschedparam(t, SCHED_FIFO, &schedprm);
	if (ret != 0)
		do_throw("Could not set max thread priority");
	int schedpol;
	ret = pthread_getschedparam(t, &schedpol, &schedprm);
	if (ret != 0)
		do_throw("Could not get thread scheduling info");

	if (schedpol != SCHED_FIFO)
		do_throw("Failed to apply SCHED_FIFO policy");
	return schedprm.sched_priority;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Load test vector in sysmem
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
std::vector<std::string> parse_input_file(std::string const& input_file)
{
	std::ifstream ifs(input_file);
	pt_dbg("Parsing input test vector file %s\n", input_file.c_str());
	std::vector<std::string> ret;
	std::string l;
	while (std::getline(ifs, l)) {
		if (l.empty())
			continue;
		ret.push_back(l);
	}
	pt_dbg("After parsing, there are %zu input test vectors\n", ret.size());
	return ret;
}
