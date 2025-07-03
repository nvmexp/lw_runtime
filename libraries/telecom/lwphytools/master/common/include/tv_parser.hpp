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

#ifndef TVPARSER_HPP__
#define TVPARSER_HPP__

#include <atomic>
#include <rte_errno.h>
#include <rte_ethdev.h>
#include <signal.h>
#include <sched.h>
#include <fstream>
#include "hdf5hpp.hpp"
#include "general.hpp"
#include "oran_structs.hpp"
#include <rte_malloc.h>

/* uses rte_malloc/rte_free for managing memory */
typedef std::unique_ptr<void, decltype(&rte_free)> rte_unique;
struct Dataset {
    Dataset() : size(0), data(nullptr, &rte_free) {};
    size_t size;
    rte_unique data;
};

struct Slot {
    Slot(size_t size) {
        ptrs = std::vector<std::vector<std::vector<void*>>>(size, std::vector<std::vector<void *>>(SLOT_NUM_SYMS, std::vector<void *>(MAX_NUM_PRBS_PER_SYMBOL/size)));
        // ptrs = std::vector<std::array<prb_pointers, SLOT_NUM_SYMS>>(size);
        prbs_per_flow = std::vector<std::array<uint32_t, SLOT_NUM_SYMS>>(size);
        prbs_per_pkt = std::vector<std::array<uint32_t, SLOT_NUM_SYMS>>(size);
        pkts_per_flow = std::vector<std::array<uint32_t, SLOT_NUM_SYMS>>(size);
        mbufs_arr_sz = std::vector<std::array<size_t, SLOT_NUM_SYMS>>(size);
        mbufs_hdr_uniq = std::vector<std::array<uintptr_t, SLOT_NUM_SYMS>>(size);
        mbufs_ext_uniq = std::vector<std::array<uintptr_t, SLOT_NUM_SYMS>>(size);
    }

    Dataset raw_data;
    /* Pointers for quickly accessing pointers to PRBs */
    std::vector<std::vector<std::vector<void *>>> ptrs;
    /* Number of PRBs for each symbol */
    std::vector<std::array<uint32_t, SLOT_NUM_SYMS>> prbs_per_flow;
    std::vector<std::array<uint32_t, SLOT_NUM_SYMS>> prbs_per_pkt;
    std::vector<std::array<uint32_t, SLOT_NUM_SYMS>> pkts_per_flow;
    std::array<uint32_t, SLOT_NUM_SYMS> pkts_per_sym;
    std::array<uint32_t, SLOT_NUM_SYMS> prbs_per_sym;
    std::vector<std::array<size_t, SLOT_NUM_SYMS>> mbufs_arr_sz;

    //Vector doesn't require static inizialization via constructor of the unique_ptr destructor
    /* header mbufs */
    // std::vector<rte_unique> mbufs_hdr_uniq;
    std::vector<std::array<uintptr_t, SLOT_NUM_SYMS>> mbufs_hdr_uniq;
    /* PRBs mbufs */
    // std::vector<rte_unique> mbufs_ext_uniq;
    std::vector<std::array<uintptr_t, SLOT_NUM_SYMS>> mbufs_ext_uniq;
    // std::vector<std::array<uintptr_t, SLOT_NUM_SYMS>> mbufs_prbs_uniq;
    size_t data_sz;
    size_t antenna_sz;
    size_t symbol_sz;
    size_t prb_sz;
    size_t prbs_per_symbol;
    size_t prbs_per_slot;
    int pkts_per_slot;
};

typedef std::unordered_map<std::string, Dataset> dataset_map;

void do_throw(std::string const& what);
// Colwert a DataRx dataset to a slot, pre-filling all the pointers
Slot dataset_to_slot(Dataset d, size_t num_ante);
/* Loads test vectors' requested datasets into an unordered map */
dataset_map load_test_vectors(std::string const& file, std::vector<std::string> const& datasets, uint16_t& num_ante);

Slot buffer_to_slot(uint8_t * buffer, size_t buffer_size, size_t num_ante, size_t prb_sz);

#endif //ifndef TVPARSER_HPP__
