/*
 * Copyright 1993-2020 LWPU Corporation. All rights reserved.
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

#include "tv_parser.hpp"

void do_throw(std::string const& what)
{
    throw std::runtime_error(what);
}

Slot dataset_to_slot(Dataset d, size_t num_ante)
{
    Slot ret(num_ante);
    size_t data_sz = d.size;
    if (data_sz % SLOT_NUM_SYMS)
        do_throw(sb() << "Slot size " << data_sz
             << " doesn't divide into the number of symbols "
             << SLOT_NUM_SYMS);

    size_t antenna_sz = d.size / num_ante;
    size_t symbol_sz = antenna_sz / SLOT_NUM_SYMS;
    if (symbol_sz % PRB_SIZE_16F)
        do_throw(sb() << "Symbol size " << symbol_sz
             << " doesn't divide into the size of a PRB "
             << PRB_SIZE_16F);

    size_t prbs_per_symbol = symbol_sz / PRB_SIZE_16F;
    if (prbs_per_symbol > MAX_NUM_PRBS_PER_SYMBOL)
        do_throw(sb() << "Resulting number of PRBs per symbol "
             << prbs_per_symbol << " is higher than MAX_NUM_PRBS_PER_SYMBOL. "
             << "Please recompile with a higher value for "
             << "MAX_NUM_PRBS_PER_SYMBOL.");


    char *base_ptr = (char *)d.data.get();
    for (size_t symbol_idx = 0; symbol_idx < SLOT_NUM_SYMS; ++symbol_idx) {
        ret.prbs_per_sym[symbol_idx] = 0;
        for (size_t antenna_idx = 0; antenna_idx < num_ante; ++antenna_idx) {

            ret.prbs_per_flow[antenna_idx][symbol_idx] = prbs_per_symbol;
            ret.prbs_per_sym[symbol_idx] += prbs_per_symbol;

            for(size_t prb_idx = 0; prb_idx < prbs_per_symbol; ++prb_idx) {
                ret.ptrs[antenna_idx][symbol_idx][prb_idx] = 
                (void*)(base_ptr 
                            + antenna_idx * antenna_sz
                            + symbol_idx * symbol_sz
                            + prb_idx * PRB_SIZE_16F
                        );
            }
        }
    }

    /*
    for (size_t symbol_idx = 0; symbol_idx < SLOT_NUM_SYMS; ++symbol_idx) {
        ret.num_prbs[symbol_idx] = prbs_per_symbol;
        for (size_t prb_idx = 0; prb_idx < prbs_per_symbol; ++prb_idx) {
            ret.ptrs[symbol_idx][prb_idx] = (void *)
                   (base_ptr + symbol_idx*symbol_sz + prb_idx*PRB_SIZE_16F);
        }
    }
    */

    ret.raw_data = std::move(d);
    return ret;
}

Slot buffer_to_slot(uint8_t * buffer, size_t buffer_size, size_t num_antennas, size_t prb_sz)
{
    Slot ret(num_antennas);

    // The sizes below assume your buffer pointer is __half2, i.e, QAM elements. If it's bytes you'd need to multiply by 4.
    // size_t symbol_sz = PRB_NUM_RE * 273 * sizeof(uint32_t);
    // size_t antenna_sz = data_sz / 4;
    // size_t prb_sz = PRB_NUM_RE * sizeof(uint32_t);

    ret.data_sz = buffer_size;
    if (ret.data_sz % SLOT_NUM_SYMS)
        do_throw(sb() << "Slot size " << ret.data_sz
             << " doesn't divide into the number of symbols "
             << SLOT_NUM_SYMS);

    ret.antenna_sz = ret.data_sz / num_antennas;
    ret.symbol_sz = ret.antenna_sz / SLOT_NUM_SYMS;
    ret.prb_sz = prb_sz;
    if (ret.symbol_sz % ret.prb_sz)
        do_throw(sb() << "Symbol size " << ret.symbol_sz
             << " doesn't divide into the size of a PRB "
             << ret.prb_sz);

    ret.prbs_per_symbol = ret.symbol_sz / ret.prb_sz;
    if (ret.prbs_per_symbol > MAX_NUM_PRBS_PER_SYMBOL)
        do_throw(sb() << "Resulting number of PRBs per symbol "
             << ret.prbs_per_symbol << " is higher than MAX_NUM_PRBS_PER_SYMBOL. "
             << "Please recompile with a higher value for "
             << "MAX_NUM_PRBS_PER_SYMBOL.");

    ret.prbs_per_slot = ret.prbs_per_symbol * SLOT_NUM_SYMS * num_antennas;

    char *base_ptr = (char *)buffer;
    for (size_t symbol_idx = 0; symbol_idx < SLOT_NUM_SYMS; ++symbol_idx) {
        ret.prbs_per_sym[symbol_idx] = 0;
        for (size_t antenna_idx = 0; antenna_idx < num_antennas; ++antenna_idx) {

            ret.prbs_per_flow[antenna_idx][symbol_idx] = ret.prbs_per_symbol;
            ret.prbs_per_sym[symbol_idx] += ret.prbs_per_symbol;

            for(size_t prb_idx = 0; prb_idx < ret.prbs_per_symbol; ++prb_idx) {
                ret.ptrs[antenna_idx][symbol_idx][prb_idx] = (void*)
                        (base_ptr 
                            + antenna_idx * ret.antenna_sz
                            + symbol_idx * ret.symbol_sz
                            + prb_idx * ret.prb_sz);
            }
        }
    }
#if 0
    printf("ret.data_sz %zd\n", ret.data_sz);
    printf("ret.antenna_sz %zd\n", ret.antenna_sz);
    printf("ret.symbol_sz %zd\n", ret.symbol_sz);
    printf("ret.prb_sz %zd\n", ret.prb_sz);
    printf("ret.prbs_per_symbol %zd\n", ret.prbs_per_symbol);
    printf("ret.prbs_per_slot %zd\n", ret.prbs_per_slot);
#endif
    return ret;
}

dataset_map load_test_vectors(std::string const& file, std::vector<std::string> const& datasets, uint16_t& num_ante)
{
    dataset_map ret;
    hdf5hpp::hdf5_file fInput = hdf5hpp::hdf5_file::open(file.c_str());
    for (auto d : datasets) {
        Dataset insert;
        hdf5hpp::hdf5_dataset dset = fInput.open_dataset(d.c_str());

        //Get dimensions of the dataset to determine how many antennas/flows
        num_ante = dset.get_dataspace().get_dimensions()[0];


        insert.size = dset.get_buffer_size_bytes();
        pt_info("Opened %zu-byte dataset %s from %s\n", insert.size, d.c_str(), file.c_str());
        insert.data.reset(rte_zmalloc(NULL, insert.size, sysconf(_SC_PAGESIZE)));
        if (insert.data.get() == nullptr)
            do_throw(sb() << "rte_malloc testvector data failed. rte_errno: "
                 << rte_errno << " rte_strerror(rte_errno): "
                 << rte_strerror(rte_errno));
        dset.read(insert.data.get());
        ret[d] = std::move(insert);
    }
    return ret;
}
