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

#ifndef GENERATOR_HPP__
#define GENERATOR_HPP__

#include "general.hpp"
#include "oran_structs.hpp"
#include "json_parser.hpp"
#include "tv_parser.hpp"
#include <memory>

#define PORT_ID 0

#define DEFAULT_MAX_PKT_SIZE 4092

#define MP_CNT ((1U << 15) - 1)
#define MP_HDR_CNT MP_CNT
#define MP_EXT_CNT MP_CNT
#define MP_RXQ_CNT MP_CNT

#define MP_CACHE_SZ 512
#define MP_HDR_CACHE_SZ MP_CACHE_SZ
#define MP_EXT_CACHE_SZ MP_CACHE_SZ
#define MP_RXQ_CACHE_SZ MP_CACHE_SZ

#define MP_HDR_SZ (RTE_PKTMBUF_HEADROOM + ORAN_IQ_HDR_SZ)
#define MP_EXT_SZ (RTE_PKTMBUF_HEADROOM + 128)
#define MP_RXQ_SZ (RTE_PKTMBUF_HEADROOM + 1522)

struct dpdk_info {
	int                     socket_id;
	int                     port_id;
	//struct rte_eth_conf     port_conf;
	struct rte_ether_addr       src_eth_addr;
	struct rte_ether_addr       peer_eth_addr;
	uint16_t                vlan;

	//Mempool
	int                     memp_cache;
	int                     memp_mbuf_num;

	//Mbuf
	int mbuf_payload_size_rx;
	int mbuf_payload_size_tx;
	int mbuf_x_burst;

	// struct rte_eth_dev_info nic_info;
};

typedef std::unordered_map<std::string, Dataset> dataset_map;

/* prb_id is _within_ the symbol */
inline void *get_prb(Slot const& s, uint16_t antenna_id, uint16_t symbol_id, size_t prb_id)
{
	return s.ptrs[antenna_id][symbol_id][prb_id];
}

/* Parses an input TV file into a std vector of std string */
std::vector<std::string> parse_input_file(std::string const& input_file);

// register slot's memory for DMA by dpdk so that it can be attached as external
// buffer
void slot_dma_map(uint16_t port_id, Slot const& s);

// register a mempool's underlying storage for DMA
// Useful for when you want to use a mempool for transmitting but don't want
// to provide it to RX queues AND you want to pay the registration cost upfront
void mempool_dma_map(uint16_t port_id, struct rte_mempool *mp);

int fill_hdr_template(char *buf, uint16_t vlan_tci, struct rte_ether_addr d_addr, struct rte_ether_addr s_addr);

// just some debug info
void print_max_pkt_size_info(size_t max_pkt_sz);

// Configure and start up a DPDK port. Anticipate more overloads of this
// function
void start_eth(uint16_t port_id, uint16_t vlan, uint16_t rxq, uint16_t txq, uint16_t rxd,
		uint16_t txd, struct rte_mempool *rxmp);

// Set up signal handlers
void signal_setup();

// check the value of the "force quit" flag
bool check_force_quit();

// set the the "force quit" flag
void set_force_quit();

// A wrapper around throw std::runtime_error
// void do_throw(std::string const& what);

// return a comma-separated string listing all the CPUs the calling thread is
// affinitized to
std::string affinity_cpu_list();

// set the thread priority to the maximum
// returns the resulting priority
int set_max_thread_priority();

inline uint64_t get_ns() {
	struct timespec t;
	int ret;
	ret = clock_gettime(CLOCK_MONOTONIC, &t);
	if (ret != 0) {
		do_throw("clock_gettime failed");
	}
	return (uint64_t) t.tv_nsec + (uint64_t)t.tv_sec * 1000 * 1000 * 1000;
}

/*
 * For reference; do not use in tight loops.
 *
 * How to Benchmark Code Exelwtion Times on Intel IA-32 and IA-64 Instruction
 * Set Architectures:
 * https://www.intel.com/content/dam/www/public/us/en/dolwments/white-papers/ia-32-ia-64-benchmark-code-exelwtion-paper.pdf
 */
inline uint64_t tsc_before() {
	uint32_t lo, hi;
	asm volatile(
			"cpuid\n\t"
			"rdtsc\n\t"
		:"=a"(lo), "=d"(hi)
		:
		: "rbx", "rcx");
	return (uint64_t)lo | (((uint64_t)hi)<<32);
}

/*
 * For reference; do not use in tight loops.
 *
 * How to Benchmark Code Exelwtion Times on Intel IA-32 and IA-64 Instruction
 * Set Architectures:
 * https://www.intel.com/content/dam/www/public/us/en/dolwments/white-papers/ia-32-ia-64-benchmark-code-exelwtion-paper.pdf
 */
inline uint64_t tsc_after() {
	uint32_t lo, hi;
	asm volatile(
			"rdtscp\n\t"
			"mov %%eax, %0\n\t"
			"mov %%edx, %1\n\t"
			"cpuid\n\t"
		: "=g" (lo), "=g" (hi)
		:
		: "rax", "rbx", "rcx", "rdx");
	return (uint64_t)lo | (((uint64_t)hi)<<32);
}

#endif //ifndef GENERATOR_HPP__
