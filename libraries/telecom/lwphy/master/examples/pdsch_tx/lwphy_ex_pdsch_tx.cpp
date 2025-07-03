/*
 * Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 */

#include "pdsch_tx.hpp"

/**
 *  @brief Print usage information for the DL pipeline example.
 */
void usage() {

    std::cout << "lwphy_ex_pdsch_tx [options]" << std::endl;
    std::cout << "  Options:" << std::endl;
    std::cout << "     -h                Display usage information" << std::endl;
    std::cout << "     input_filename  num_iterations  AAS_mode (Input HDF5 filename, Number of iterations, Enable AAS mode)" << std::endl;

    std::cout << std::endl;
    std::cout << "  Examples:" << std::endl;
    std::cout << "      ./pdsch_tx -h" << std::endl;
    std::cout << "      ./pdsch_tx ~/dl_pipeline.h5 20 0" << std::endl;
    std::cout << "      ./pdsch_tx ~/dl_pipeline.h5 20 1" << std::endl;
}


/**
 *  @brief Time the DL Pipeline. Print average time over num_iterations iterations.
 *         NB: The following are not timed: Reading the HDF5 file, allocations, memory transfers.
 *         Also, the timing code does not include any reference checks as these will lwrrently
 *         fail (some buffers need to be reset across iterations).
 *  @param[in] hdf5_filename: the HDF5 input file that drives the pipeline
 *  @param[in] num_iterations: number of iterations to run the pipeline for
 *  @param[in] aas_mode: set pipeline in AAS mode (no layer mapping, modulation, DMRS).
 *  @param[in] identical_LDPC_configs: assume LDPC configs are identical. A runtime check resets
 *                                     LDPC configs to non-identical if they are not.
 */
void time_pipeline(std::string hdf5_filename, int num_iterations, bool aas_mode,
                   bool identical_LDPC_configs) {

    tensor_device data_tx_tensor = tensor_device(tensor_info(LWPHY_C_16F,
                            {(int) (LWPHY_N_TONES_PER_PRB * 273), (int)OFDM_SYMBOLS_PER_SLOT, MAX_DL_LAYERS_PER_TB}), LWPHY_TENSOR_ALIGN_TIGHT);;

    std::cout << std::endl;
    std::string pipeline_mode = (aas_mode) ? "AAS" : "non AAS";
    std::cout << "Timing the DL pipeline in " << pipeline_mode << " mode." << std::endl;
    std::cout << "- NB: Allocations not included. Ref. checks will fail!" << std::endl << std::endl;

    stream strm(lwdaStreamNonBlocking);
    lwdaStream_t strm_handle = strm.handle();
    PdschTx pdsch_example(strm_handle, hdf5_filename, aas_mode, identical_LDPC_configs);
    gnb_pars gnb_params;
    pdsch_example.expandParameters({}, gnb_params, nullptr, 0, strm_handle); // tb_params and gnb_params are not used.
    bool ref_check = false;

    event_timer lwphy_timer;
    lwphy_timer.record_begin(strm);

    for (int iter = 0; iter < num_iterations; iter++) {
        //pdsch_example.expandParameters({}, gnb_params, nullptr, 0, strm_handle);
        pdsch_example.Run(data_tx_tensor, strm_handle, ref_check);
    }

    lwphy_timer.record_end(strm_handle);
    lwphy_timer.synchronize();
    float time1 = lwphy_timer.elapsed_time_ms();
    time1 /= num_iterations;

    printf("DL pipeline: %.2f us (avg. over %d iterations)\n", time1 * 1000, num_iterations);

}



int main(int argc, char* argv[]) {

    if ((argc != 4) || ((argc == 2) && (argv[1][0] == '-') && (argv[1][1] == 'h'))) {
        usage();
        exit(1);
    }

    int num_iterations = std::stoi(argv[2]);
    if (num_iterations <= 0) {
        std::cerr << "Invalid number of iterations: " << num_iterations << ". Should be > 0." << std::endl;
        exit(1);
    }

    int cfg_aas_mode = std::stoi(argv[3]);
    if (cfg_aas_mode < 0) {
        std::cout << "Negative AAS mode value treated as 0." << std::endl;
    } else if (cfg_aas_mode > 1) {
        std::cout << "AAS mode value > 1 treated as 1." << std::endl;
    }

    LWDA_CHECK(lwdaSetDevice(0));

    // Downlink pipeline includes: (a) CRC, (b) LDPC  encoder, (c) Rate-Matching,
    // (d) Modulation Mapper and (e) DMRS components.
    stream strm(lwdaStreamNonBlocking);
    lwdaStream_t strm_handle = strm.handle();

    // Large buffer added as lwPHY tools needs a buffer with a power of 2 size.
    int large_buffer_bytes = 4194304;
    unique_device_ptr<lwFloatComplex> large_buffer = make_unique_device<lwFloatComplex>(large_buffer_bytes/sizeof(lwFloatComplex));
    tensor_device data_tx_tensor = tensor_device(large_buffer.get(), tensor_info(LWPHY_C_16F,
                            {(int) (LWPHY_N_TONES_PER_PRB * 273), (int)OFDM_SYMBOLS_PER_SLOT, MAX_DL_LAYERS_PER_TB}), LWPHY_TENSOR_ALIGN_TIGHT);;


    int data_tx_tensor_bytes = data_tx_tensor.desc().get_size_in_bytes();
    if (data_tx_tensor_bytes > large_buffer_bytes) {
        std::cerr << "Buffer (" << large_buffer_bytes << " bytes) is smaller than data_tx_tensor (" << data_tx_tensor_bytes << ")" << std::endl;
        exit(1);
    }
    // Reset output buffer
    LWDA_CHECK(lwdaMemsetAsync(data_tx_tensor.addr(), 0, data_tx_tensor_bytes, strm_handle));

    bool ref_check = true;
    bool aas_mode = (cfg_aas_mode >= 1);
    bool identical_LDPC_configs = true; // A runtime check resets LDPC configs to non-identical if they are not.
    PdschTx pdsch_example(strm_handle, argv[1], aas_mode, identical_LDPC_configs);
    std::string pipeline_mode = (aas_mode) ? "AAS" : "non AAS";
    std::cout << "Running DL pipeline once w/ reference checks enabled in " << pipeline_mode << " mode." << std::endl;
    gnb_pars gnb_params;
    pdsch_example.expandParameters({}, gnb_params, nullptr, 0, strm_handle); // tb_params and gnb_params are not used.
    pdsch_example.Run(data_tx_tensor, strm_handle, ref_check);

    // Copy pipeline's output to the CPU. In aas_mode this won't be populated.
    if (!aas_mode) {
        typed_tensor<LWPHY_C_16F, pinned_alloc> h_pdsch_out_tensor(data_tx_tensor.layout());
        h_pdsch_out_tensor.colwert(data_tx_tensor, strm_handle);
    }
    strm.synchronize();

#if 0
    // Example to show the layout of the output tensor
    int layer_id = 1;
    int symbol_id = 3;
    int freq_bin = 0;
    printf("Symbol at (freq. bin %d, symbol %d, layer %d) =  (%f + i %f)\n", freq_bin, symbol_id, layer_id,
          (float) h_pdsch_out_tensor({freq_bin, symbol_id, layer_id}).x,
          (float) h_pdsch_out_tensor({freq_bin, symbol_id, layer_id}).y);
#endif

    // Time pipeline. Does not time allocations etc.
    time_pipeline(argv[1], num_iterations, aas_mode, identical_LDPC_configs);

    return 0;
}

