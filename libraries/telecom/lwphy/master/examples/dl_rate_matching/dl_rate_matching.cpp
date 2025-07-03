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
#include "utils.lwh"

using namespace std;
using namespace lwphy;

template <typename T>
T div_round_up(T val, T divide_by) {
    return ((val + (divide_by - 1)) / divide_by);
}

void read_gnb_pars_from_file(gnb_pars & gnb_params, hdf5hpp::hdf5_file & input_file) {

    lwphy::lwphyHDF5_struct gnbConfig = lwphy::get_HDF5_struct(input_file, "gnb_pars");
    gnb_params.fc                        = gnbConfig.get_value_as<uint32_t>("fc");
    gnb_params.mu                        = gnbConfig.get_value_as<uint32_t>("mu");
    gnb_params.nRx                       = gnbConfig.get_value_as<uint32_t>("nRx");
    gnb_params.nPrb                      = gnbConfig.get_value_as<uint32_t>("nPrb");
    gnb_params.cellId                    = gnbConfig.get_value_as<uint32_t>("cellId");
    gnb_params.slotNumber                = gnbConfig.get_value_as<uint32_t>("slotNumber");
    gnb_params.Nf                        = gnbConfig.get_value_as<uint32_t>("Nf");
    gnb_params.Nt                        = gnbConfig.get_value_as<uint32_t>("Nt");
    gnb_params.df                        = gnbConfig.get_value_as<uint32_t>("df");
    gnb_params.dt                        = gnbConfig.get_value_as<uint32_t>("dt");
    gnb_params.numBsAnt                  = gnbConfig.get_value_as<uint32_t>("numBsAnt");
    gnb_params.numBbuLayers              = gnbConfig.get_value_as<uint32_t>("numBbuLayers");
    gnb_params.numTb                     = gnbConfig.get_value_as<uint32_t>("numTb");
    gnb_params.ldpcnIterations           = gnbConfig.get_value_as<uint32_t>("ldpcnIterations");
    gnb_params.ldpcEarlyTermination      = gnbConfig.get_value_as<uint32_t>("ldpcEarlyTermination");
    gnb_params.ldpcAlgoIndex             = gnbConfig.get_value_as<uint32_t>("ldpcAlgoIndex");
    gnb_params.ldpcFlags                 = gnbConfig.get_value_as<uint32_t>("ldpcFlags");
    gnb_params.ldplwseHalf               = gnbConfig.get_value_as<uint32_t>("ldplwseHalf");
}


void read_tb_pars_from_file(std::vector<tb_pars> & tb_params, hdf5hpp::hdf5_dataset & input_dataset) {

    int num_TBs = tb_params.size();
    for (int TB_id = 0; TB_id < num_TBs; TB_id++) {

        lwphy::lwphyHDF5_struct tb_config        = lwphy::get_HDF5_struct_index(input_dataset, TB_id);
        tb_params[TB_id].numLayers        = tb_config.get_value_as<uint32_t>("numLayers");
        tb_params[TB_id].layerMap         = tb_config.get_value_as<uint32_t>("layerMap");
        tb_params[TB_id].startPrb         = tb_config.get_value_as<uint32_t>("startPrb");
        tb_params[TB_id].numPrb           = tb_config.get_value_as<uint32_t>("numPRb");
        tb_params[TB_id].startSym         = tb_config.get_value_as<uint32_t>("startSym");
        tb_params[TB_id].numSym           = tb_config.get_value_as<uint32_t>("numSym");
        tb_params[TB_id].dmrsMaxLength    = tb_config.get_value_as<uint32_t>("dmrsMaxLength");
        tb_params[TB_id].dataScramId      = tb_config.get_value_as<uint32_t>("dataScramId");
        tb_params[TB_id].mcsTableIndex    = tb_config.get_value_as<uint32_t>("mcsTableIndex");
        tb_params[TB_id].mcsIndex         = tb_config.get_value_as<uint32_t>("mcsIndex");
        tb_params[TB_id].rv               = tb_config.get_value_as<uint32_t>("rv");
        tb_params[TB_id].dmrsType         = tb_config.get_value_as<uint32_t>("dmrsType");
        tb_params[TB_id].dmrsAddlPosition = tb_config.get_value_as<uint32_t>("dmrsAddlPosition");
        tb_params[TB_id].dmrsMaxLength    = tb_config.get_value_as<uint32_t>("dmrsMaxLength");
        tb_params[TB_id].dmrsScramId      = tb_config.get_value_as<uint32_t>("dmrsScramId");
        tb_params[TB_id].dmrsEnergy       = tb_config.get_value_as<uint32_t>("dmrsEnergy");
        tb_params[TB_id].nRnti            = tb_config.get_value_as<uint32_t>("nRnti");
        tb_params[TB_id].dmrsCfg          = tb_config.get_value_as<uint32_t>("dmrsCfg");
        tb_params[TB_id].nPortIndex       = tb_config.get_value_as<uint32_t>("nPortIndex");
        tb_params[TB_id].nSCID            = tb_config.get_value_as<uint32_t>("nSCID");
    }
}

void usage() {

    std::cout << "dl_rate_matching [options]" << std::endl;
    std::cout << "  Options:" << std::endl;
    std::cout << "     -h                              (Display usage information)" << std::endl;
    std::cout << "     input_filename  num_iterations  (Input HDF5 filename, Number of iterations)" << std::endl;


    std::cout << std::endl;
    std::cout << "  Examples:" << std::endl;
    std::cout << "      ./dl_rate_matching ~/input_file.h5 20" << std::endl;
    std::cout << "      ./dl_rate_matching -h" << std::endl;
}

void print_TB_config_params(std::vector<PerTbParams> & kernel_params, int TB_id,
                            bool layer_mapping, bool scrambling) {

    int num_TBs = kernel_params.size();

    // Current codebase expects these config. params to be the same across TBs.
    if (TB_id == 0) {
        std::cout << "Config. Parameters shared across all " <<  num_TBs << " TB(s):" << std::endl;

        std::cout << "* layer_mapping is " << layer_mapping << std::endl;
        std::cout << "* scrambling is " << scrambling << std::endl;
    }

    // Config. parameters that vary across TBs.
    std::cout << std::endl;
    std::cout << "Config. Parameters specific to TB " << TB_id << ": " << std::endl;

    std::cout << "* rv = " << kernel_params[TB_id].rv << std::endl;
    std::cout << "* Qm = " << kernel_params[TB_id].Qm << std::endl;
    std::cout << "* bg = " << kernel_params[TB_id].bg << std::endl;
    std::cout << "* Nl = " << kernel_params[TB_id].Nl << std::endl;
    std::cout << "* num_CBs = " << kernel_params[TB_id].num_CBs << std::endl;
    std::cout << "* Zc = " << kernel_params[TB_id].Zc << std::endl;

    std::cout << "* N = " << kernel_params[TB_id].N << std::endl;
    //std::cout << "Ncb = " << kernel_params[TB_id].Ncb << std::endl;
    std::cout << "* G = " << kernel_params[TB_id].G << std::endl;
    std::cout << "* K = " << kernel_params[TB_id].K << std::endl;
    std::cout << "* F = " << kernel_params[TB_id].F << std::endl;

    std::cout << "* cinit = " << kernel_params[TB_id].cinit << std::endl;
    int TB_layers = kernel_params[TB_id].Nl;
    std::cout << "* layer_map[" << TB_layers << "] = {";
    for (int layer_cnt = 0; layer_cnt < TB_layers; layer_cnt++) {
        std::cout << kernel_params[TB_id].layer_map_array[layer_cnt];
        if (layer_cnt != TB_layers - 1) {
            std::cout << ", ";
        } else {
            std::cout << "}" << std::endl;
        }
    } 
}


int main(int argc, char* argv[]) {

    using tensor_pinned_R_64F = typed_tensor<LWPHY_R_64F, pinned_alloc>;

    const int ELEMENT_SIZE = sizeof(uint32_t) * 8; // 32 bits
    lwdaStream_t strm = 0; // update as needed

    if ((argc != 3) || ((argc == 2) && (argv[1][0] == '-') && (argv[1][1] == 'h'))) {
        usage();
        exit(1);
    }

    int num_iterations = stoi(argv[2]);
    if (num_iterations <= 0) {
        std::cerr << "Invalid number of iterations: " << num_iterations << ". Should be > 0." << std::endl;
        exit(1);
    }

    // Read from HDF5 input file.
    std::string hdf5_filename = argv[1];

    hdf5hpp::hdf5_file input_file = hdf5hpp::hdf5_file::open(hdf5_filename.c_str());
    hdf5hpp::hdf5_dataset tb_pars_dataset = input_file.open_dataset("tb_pars");
    hdf5hpp::hdf5_dataspace tb_dataspace = tb_pars_dataset.get_dataspace();
    int num_TBs = tb_dataspace.get_dimensions()[0];

    gnb_pars gnb_params;
    read_gnb_pars_from_file(gnb_params, input_file);

    std::vector<tb_pars> tb_params(num_TBs);
    read_tb_pars_from_file(tb_params, tb_pars_dataset);

    std::vector<PerTbParams> kernel_params(num_TBs);
    lwphyStatus_t params_status = lwphySetTBParamsFromStructs(&kernel_params[0], (const tb_pars *) &tb_params[0], (const gnb_pars *) &gnb_params);
    if (params_status != LWPHY_STATUS_SUCCESS) {
        throw std::runtime_error("Error when setting TB config parameters!");
    }

    int N_max = kernel_params[0].N;
    uint32_t FIXME_CMAX = kernel_params[0].num_CBs; // Could alternatively overprovision w/ MAX_N_CBS_PER_TB_SUPPORTED
    for (int i = 1; i < num_TBs; i++) {
        N_max = std::max(N_max, (int) kernel_params[i].N);
        FIXME_CMAX = std::max(FIXME_CMAX, kernel_params[i].num_CBs);
    }
    bool scrambling = true; // Set to true or false as needed.
    bool layer_mapping = true; // Set to true or false as needed. If layer_mapping is true, then scrambling also has to be true.

    int num_layers = gnb_params.numBbuLayers;
    uint32_t Emax = 0;
    uint32_t Cmax = 0;

    // Allocate workspace and load parameters.
    size_t allocated_workspace_size = lwphyDlRateMatchingWorkspaceSize(num_TBs, FIXME_CMAX);
    unique_device_ptr<uint32_t> config_workspace = make_unique_device<uint32_t>(div_round_up<uint32_t>(allocated_workspace_size, sizeof(uint32_t)));
    unique_pinned_ptr<uint32_t> h_workspace = make_unique_pinned<uint32_t>((2 + FIXME_CMAX) * num_TBs);

    dl_rateMatchingElw * dl_rate_matching_elw[1];
    lwphyStatus_t load_params_status = lwphyDlRateMatchingLoadParams(dl_rate_matching_elw, num_TBs, kernel_params.data(),
                               &Emax, &Cmax, num_layers, scrambling, layer_mapping, config_workspace.get(),
                               allocated_workspace_size, h_workspace.get(), strm);

    if (load_params_status != LWPHY_STATUS_SUCCESS) {
        throw std::runtime_error("Invalid argument(s) in dl_rate_matchingLoadParams");
    }

    size_t output_elements = (layer_mapping) ? div_round_up<uint32_t>(num_layers * Cmax * Emax, ELEMENT_SIZE) : div_round_up<uint32_t>(num_TBs * Cmax * Emax, ELEMENT_SIZE);

    std::vector<uint32_t> h_rate_matching_output(output_elements);
    unique_device_ptr<uint32_t> d_rate_matching_output = make_unique_device<uint32_t>(output_elements);

    // Read and prep input. Lwrrently each HDF5 element in tb*_codedbcbs is a double.
    // Rate matching expects bits packed into uint32_t elements
    tensor_device d_in_tensor = (tensor_info(LWPHY_BIT, {N_max, (int) Cmax, num_TBs}));

    std::vector<tensor_device> single_TB_d_in_tensor(num_TBs);
    int single_TB_in_tensor_bytes = d_in_tensor.desc().get_size_in_bytes() / num_TBs;

    for (int TB_id = 0; TB_id < num_TBs; TB_id++) {
        int num_CBs = kernel_params[TB_id].num_CBs;
        uint32_t in_TB_offset = TB_id * single_TB_in_tensor_bytes;
        single_TB_d_in_tensor[TB_id] = tensor_device((void*)((uint8_t*)d_in_tensor.addr() + in_TB_offset),
                                            (tensor_info(LWPHY_BIT, {(int) kernel_params[TB_id].N, num_CBs})));

        std::string input_dataset_name = "tb" + std::to_string(TB_id) + "_codedcbs";
        tensor_pinned_R_64F input_data = typed_tensor_from_dataset<LWPHY_R_64F, pinned_alloc>(input_file.open_dataset(input_dataset_name.c_str()));
        int N = kernel_params[TB_id].N;
        typed_tensor<LWPHY_BIT, pinned_alloc> single_TB_h_in_tensor(single_TB_d_in_tensor[TB_id].layout());

        for (int CB = 0; CB < num_CBs; CB++) {
            for (int element_start = 0; element_start < N; element_start += ELEMENT_SIZE)  {
                uint32_t bits = 0;
                for (int offset = 0; (offset < ELEMENT_SIZE) && (element_start + offset < N); offset++) {
                    uint32_t bit = (input_data({element_start + offset, CB}) == 1) ? 1 : 0;
		    bits |= (bit << offset);
                 }
                 single_TB_h_in_tensor({element_start / ELEMENT_SIZE, CB, TB_id}) = bits;
            }
        }
        LWDA_CHECK(lwdaMemcpy(single_TB_d_in_tensor[TB_id].addr(), single_TB_h_in_tensor.addr(),
                              single_TB_h_in_tensor.desc().get_size_in_bytes(), lwdaMemcpyHostToDevice));

        print_TB_config_params(kernel_params, TB_id, layer_mapping, scrambling);
    }

    LWDA_CHECK(lwdaDeviceSynchronize());


    // Launch rate matching kernel
    lwdaEvent_t start, stop;
    lwdaEventCreate(&start);
    lwdaEventCreate(&stop);

    float time1 = 0.0;
    lwdaEventRecord(start);

    for (int iter = 0; iter < num_iterations; iter++) {

        lwphyDlRateMatching(dl_rate_matching_elw[0],  (const uint32_t*)d_in_tensor.addr(),
                                       d_rate_matching_output.get(), strm);
    }

    lwdaError_t lwda_error = lwdaGetLastError();
    if (lwda_error != lwdaSuccess) {
        std::cout << "LWCA Error " << lwdaGetErrorString(lwda_error) << std::endl;
    }

    lwdaEventRecord(stop);
    lwdaEventSynchronize(stop);
    lwdaEventElapsedTime(&time1, start, stop);

    lwdaEventDestroy(start);
    lwdaEventDestroy(stop);

    time1 /= num_iterations;
    printf("\nDL Rate Matching Kernel: %.2f us (avg. over %d iterations)\n", time1 * 1000, num_iterations);

    LWDA_CHECK(lwdaMemcpy(h_rate_matching_output.data(), d_rate_matching_output.get(), output_elements * sizeof(uint32_t), lwdaMemcpyDeviceToHost));

    // Compare rate-matching's output w/ reference output
    int ref_bit = 0;
    unsigned long long error_cnt = 0;
    std::vector<uint32_t> Er(Cmax *  num_TBs);
    // Note could potentially expose that as part of loadParams but then that function would need to allocate host memory.
    lwphyStatus_t status = lwphyCopyErValuesToHost(dl_rate_matching_elw, Er.data(), Cmax, num_TBs, strm);
    LWDA_CHECK(lwdaDeviceSynchronize());
    if (status != LWPHY_STATUS_SUCCESS) {
        throw std::runtime_error("Invalid argument in lwphyCopyErValuesToHost.");
    }

    std::string dataset = (!layer_mapping) ? (scrambling ? "_scramcbs" : "_ratematcbs") : "_layer_mapped";

    uint32_t total_ref_bits = 0;
    for (int TB_id = 0; TB_id < num_TBs; TB_id++) {
        int TB_num_layers = (!layer_mapping) ? 1 : kernel_params[TB_id].Nl;
        int ref_bit = 0;

        std::string ref_dataset_name = "tb" + std::to_string(TB_id) + dataset;
        tensor_pinned_R_64F ref_data = typed_tensor_from_dataset<LWPHY_R_64F, pinned_alloc>(input_file.open_dataset(ref_dataset_name.c_str()));


        for (int layer_cnt = 0; layer_cnt < TB_num_layers; layer_cnt++) {
            int layer_or_TB_id = layer_mapping ? kernel_params[TB_id].layer_map_array[layer_cnt] : TB_id;
            for (int CB = 0; CB < kernel_params[TB_id].num_CBs; CB++) {
                for (int Er_bit = 0; Er_bit < Er[TB_id * Cmax + CB]/TB_num_layers; Er_bit++) {
                    uint32_t ref_value = (ref_data({ref_bit, 0}) == 0.0) ? 0 : 1;

                    int out_index = layer_or_TB_id * Cmax *Emax + CB * Emax + Er_bit;
                    int out_word = out_index / ELEMENT_SIZE;
                    int out_bits = out_index % ELEMENT_SIZE;
                    uint32_t computed_value = (h_rate_matching_output[out_word] >> out_bits) & 0x1;
                    if (ref_value != computed_value) {
                        error_cnt += 1;
                        /*std::cerr << std::endl << "GPU vs. reference output mismatch!" << std::endl;
                        std::cerr << "TB " << TB_id;
                        if (layer_mapping) std::cerr  << ", Layer " << layer_or_TB_id;
                        std::cerr  << ", CB " << CB << ", Er bit " << Er_bit;
                        std::cerr << ": computed value " << computed_value << " vs. reference " << ref_value << std::endl;*/
                    }
                    ref_bit += 1;
                    total_ref_bits += 1;
                }
            }
        }
    }

    std::cout << std::endl << "Rate Matching Error Count: " << error_cnt << " bits out of " << total_ref_bits;
    std::cout << "; GPU output compared w/ reference dataset <tb*" << dataset << "> from <" << hdf5_filename << ">" << std::endl;

    lwphyDlRateMatchingCleanUp(dl_rate_matching_elw);

    return 0;
}
