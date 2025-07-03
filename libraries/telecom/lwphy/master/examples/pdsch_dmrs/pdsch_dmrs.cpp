/*
 * Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 */

#include <cstdlib>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include "lwphy.h"
#include "lwphy.hpp"
#include "lwphy_hdf5.hpp"
#include "hdf5hpp.hpp"
#include "util.hpp"

using namespace std;
using namespace lwphy;

template<typename Tscalar>
bool compare_approx(const Tscalar &a, const Tscalar &b) {
    const Tscalar tolerance = 0.0001f; //FIXME update tolerance as needed.
    Tscalar diff = fabs(a - b);
    Tscalar m = std::max(fabs(a), fabs(b));
    Tscalar ratio = (diff >= tolerance) ? (Tscalar)(diff / m) : diff;

    return (ratio <= tolerance);
}

template<typename Tcomplex, typename Tscalar>
bool complex_approx_equal(Tcomplex & a, Tcomplex & b) {
    return (compare_approx<Tscalar>(a.x, b.x) && compare_approx<Tscalar>(a.y, b.y));
}

void usage() {

    std::cout << "pdsch_dmrs [options]" << std::endl;
    std::cout << "  Options:" << std::endl;
    std::cout << "     -h                              (Display usage information)" << std::endl;
    std::cout << "     input_filename                  (Input HDF5 filename)" << std::endl;


    std::cout << std::endl;
    std::cout << "  Examples:" << std::endl;
    std::cout << "      ./pdsch_dmrs ~/input_file.h5" << std::endl;
    std::cout << "      ./pdsch_dmrs -h" << std::endl;
}

//Update from file - on the host
void update_dmrs_params(std::vector<PdschDmrsParams> & h_dmrs_params,
                        hdf5hpp::hdf5_file & input_file) {

    //Support multiple-TBs code
    hdf5hpp::hdf5_dataset tb_pars_dataset = input_file.open_dataset("tb_pars");
    hdf5hpp::hdf5_dataspace tb_dataspace = tb_pars_dataset.get_dataspace();
    int num_TBs = tb_dataspace.get_dimensions()[0];
    h_dmrs_params.resize(num_TBs);

    // Read gnb params
    lwphy::lwphyHDF5_struct gnb_config = lwphy::get_HDF5_struct(input_file, "gnb_pars");
    int Nf = gnb_config.get_value_as<uint32_t>("Nf");
    int Nt = gnb_config.get_value_as<uint32_t>("Nt");
    int slot_number = gnb_config.get_value_as<uint32_t>("slotNumber");
    int cell_id = gnb_config.get_value_as<uint32_t>("cellId");

    // Update per-TB DmrsParams. Note that the gnb params fields are replicated across TBs.
    for (int TB_id = 0; TB_id < num_TBs; TB_id++) {
        h_dmrs_params[TB_id].Nf = Nf;
        h_dmrs_params[TB_id].Nt = Nt;
        h_dmrs_params[TB_id].slot_number = slot_number;
        h_dmrs_params[TB_id].cell_id = cell_id;

        // Update TB specific params.
        lwphy::lwphyHDF5_struct tb_config = lwphy::get_HDF5_struct_index(tb_pars_dataset, TB_id);

        h_dmrs_params[TB_id].beta_dmrs = sqrt(tb_config.get_value_as<uint32_t>("dmrsEnergy") * 1.0f);

        h_dmrs_params[TB_id].num_dmrs_symbols = tb_config.get_value_as<uint32_t>("dmrsMaxLength");
        if ((h_dmrs_params[TB_id].num_dmrs_symbols < 1) ||
            (h_dmrs_params[TB_id].num_dmrs_symbols > 2)) {
            throw std::runtime_error("Invalid number of DMRS symbols. Only one or two are supported.");
        }
        h_dmrs_params[TB_id].num_data_symbols = tb_config.get_value_as<uint32_t>("numSym") - h_dmrs_params[TB_id].num_dmrs_symbols;

        h_dmrs_params[TB_id].symbol_number = tb_config.get_value_as<uint32_t>("startSym");
        h_dmrs_params[TB_id].num_layers = tb_config.get_value_as<uint32_t>("numLayers");
        h_dmrs_params[TB_id].start_Rb = tb_config.get_value_as<uint32_t>("startPrb");
        h_dmrs_params[TB_id].num_Rbs = tb_config.get_value_as<uint32_t>("numPRb");
        if (h_dmrs_params[TB_id].num_Rbs == 0) {
            throw std::runtime_error("Zero PRBs allocated for DMRS!");
        }

        // Up to 8 layers are encoded in an uint32_t, 4 bits at a time.
        uint32_t port_index = tb_config.get_value_as<uint32_t>("nPortIndex");
        for (int i = 0; i < h_dmrs_params[TB_id].num_layers; i++) {
            h_dmrs_params[TB_id].port_ids[i] = 1000 + ((port_index >> (28 - 4 * i)) & 0x0FU);
        }

        h_dmrs_params[TB_id].n_scid = tb_config.get_value_as<uint32_t>("nSCID");
        h_dmrs_params[TB_id].dmrs_scid = tb_config.get_value_as<uint32_t>("dmrsScramId");
    }
}


int main(int argc, char* argv[]) {

    using tensor_pinned_C_32F = typed_tensor<LWPHY_C_32F, pinned_alloc>;
    using tensor_pinned_C_64F = typed_tensor<LWPHY_C_64F, pinned_alloc>;

    const int ELEMENT_SIZE = sizeof(uint32_t) * 8; // 32 bits
    lwdaStream_t strm = 0;

    if ((argc != 2) || ((argc == 2) && (argv[1][0] == '-') && (argv[1][1] == 'h'))) {
        usage();
        exit(1);
    }

    // Open HDF5 input file
    std::unique_ptr<hdf5hpp::hdf5_file> input_file(new hdf5hpp::hdf5_file(hdf5hpp::hdf5_file::open(argv[1])));

    // Parse PdschDmrsParams
    std::vector<PdschDmrsParams> h_dmrs_params;
    update_dmrs_params(h_dmrs_params, *input_file);

    int num_TBs = h_dmrs_params.size();
    unique_device_ptr<PdschDmrsParams> d_params = make_unique_device<PdschDmrsParams>(num_TBs);
    LWDA_CHECK(lwdaMemcpy(d_params.get(), &h_dmrs_params[0], num_TBs * sizeof(PdschDmrsParams), lwdaMemcpyHostToDevice));

    // Allocate device buffers for final and intermediate results
    int Nf = h_dmrs_params[0].Nf;
    int Nt = h_dmrs_params[0].Nt;
    tensor_device dmrs_scram_seq(lwphy::tensor_info(LWPHY_C_16F,
                                {(Nf/2)*2, num_TBs}), /* multiplied by 2 assumes 2 symbols, overprovisioned*/
                                LWPHY_TENSOR_ALIGN_TIGHT);

    tensor_device re_mapped_dmrs(lwphy::tensor_info(LWPHY_C_16F,
                                {(int) (LWPHY_N_TONES_PER_PRB * 273), OFDM_SYMBOLS_PER_SLOT, 16}), /*16 is MAX_DL_UEs */
                                LWPHY_TENSOR_ALIGN_TIGHT);

    int output_bytes = re_mapped_dmrs.desc().get_size_in_bytes();
    LWDA_CHECK(lwdaMemset(re_mapped_dmrs.addr(), 0, output_bytes));

    LWDA_CHECK(lwdaDeviceSynchronize());

    lwphyPdschDmrs(d_params.get(),
                   dmrs_scram_seq.desc().handle(),
                   dmrs_scram_seq.addr(),
                   re_mapped_dmrs.desc().handle(),
                   re_mapped_dmrs.addr(),
                   strm);


    LWDA_CHECK(lwdaDeviceSynchronize());

    //Reference comparison after resource element mapping of DMRS. Will only check the DMRS QAMs.
    tensor_pinned_C_64F re_mapped_ref_data = typed_tensor_from_dataset<LWPHY_C_64F, pinned_alloc>((*input_file).open_dataset("Xtf"));

    typed_tensor<LWPHY_C_16F, pinned_alloc> h_re_mapped_dmrs_tensor(re_mapped_dmrs.layout());
    h_re_mapped_dmrs_tensor = re_mapped_dmrs;
    int num_hdf5_PRBs = re_mapped_ref_data.layout().dimensions()[0] / LWPHY_N_TONES_PER_PRB;

    uint32_t re_mapped_gpu_mismatch = 0;
    uint32_t checked_elements = 0;

    for (int TB_id = 0; TB_id < num_TBs; TB_id++) {
        int start_Rb = h_dmrs_params[TB_id].start_Rb;
        int num_Rbs = h_dmrs_params[TB_id].num_Rbs;
        for (int tmp_layer = 0; tmp_layer < h_dmrs_params[TB_id].num_layers; tmp_layer++) {
            int layer_id = (h_dmrs_params[TB_id].port_ids[tmp_layer] - 1000) + 8 * h_dmrs_params[TB_id].n_scid;

            for (int freq_idx = 0; freq_idx < LWPHY_N_TONES_PER_PRB * 273; freq_idx++) {
                for (int dmrs_symbol = 0; dmrs_symbol < h_dmrs_params[TB_id].num_dmrs_symbols; dmrs_symbol++)  {
                    int symbol_id = h_dmrs_params[TB_id].symbol_number + dmrs_symbol;
                    __half2 gpu_symbol = h_re_mapped_dmrs_tensor({freq_idx, symbol_id, layer_id});
                    __half2 ref_symbol;
                    if (num_hdf5_PRBs == 273) {
                        ref_symbol.x = (half) re_mapped_ref_data({freq_idx, symbol_id, layer_id}).x;
                        ref_symbol.y = (half) re_mapped_ref_data({freq_idx, symbol_id, layer_id}).y;
                    } else if (num_hdf5_PRBs == num_Rbs) { // HDF5 only holds the allocated PRBs
                        if ((freq_idx < (start_Rb * LWPHY_N_TONES_PER_PRB)) || (freq_idx >= ((start_Rb + num_Rbs) * LWPHY_N_TONES_PER_PRB))) {
                            ref_symbol.x = (half) 0.0f;
                            ref_symbol.y = (half) 0.0f;
                        } else {
                            ref_symbol.x = (half) re_mapped_ref_data({freq_idx - (start_Rb * LWPHY_N_TONES_PER_PRB), symbol_id, layer_id}).x;
                            ref_symbol.y = (half) re_mapped_ref_data({freq_idx - (start_Rb * LWPHY_N_TONES_PER_PRB), symbol_id, layer_id}).y;
                        }
                    } else {
                        printf("Number of PRBs %d\n", num_hdf5_PRBs);
                        throw std::runtime_error("The HDF5 file contains an unexpected number of PRBs");
                    }
                    checked_elements += 1;

                    if (!complex_approx_equal<__half2, __half>(gpu_symbol, ref_symbol)) {
                        printf("Error! Mismatch for (freq. bin %d, symbol %d, layer_id %d) - expected=%f + i %f vs. gpu=%f + i %f\n",
                               freq_idx, symbol_id, layer_id,
                               (float) ref_symbol.x, (float) ref_symbol.y,
                               (float) gpu_symbol.x, (float) gpu_symbol.y);
                        re_mapped_gpu_mismatch += 1;
                     }
                }
            }
        }
    }

   std::cout << "Found " << re_mapped_gpu_mismatch << " mismatched RE mapped DMRS symbols out of " << checked_elements << std::endl;

   return 0;
}
