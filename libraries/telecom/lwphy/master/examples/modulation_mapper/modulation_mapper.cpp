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
    const Tscalar tolerance = 0.0001f; //update tolerance as needed.
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
    std::cout << "modulation_mapper [options]" << std::endl;
    std::cout << "  Options:" << std::endl;
    std::cout << "     -h                              (Display usage information)" << std::endl;
    std::cout << "     input_filename  num_iterations  (Input HDF5 filename, Number of iterations)" << std::endl;


    std::cout << std::endl;
    std::cout << "  Examples:" << std::endl;
    std::cout << "      ./modulation_mapper ~/input_file.h5 20" << std::endl;

}

int main(int argc, char* argv[]) {

    using tensor_pinned_R_64F = typed_tensor<LWPHY_R_64F, pinned_alloc>;
    using tensor_pinned_C_64F = typed_tensor<LWPHY_C_64F, pinned_alloc>;

    const int ELEMENT_SIZE = sizeof(uint32_t) * 8; // 32 bits
    lwdaStream_t strm = 0;

    if ((argc != 3) || ((argc == 2) && (argv[1][0] == '-') && (argv[1][1] == 'h'))) {
        usage();
        exit(1);
    }

    int num_iterations = stoi(argv[2]);
    if (num_iterations <= 0) {
        std::cerr << "Invalid number of iterations: " << num_iterations << ". Should be > 0." << std::endl;
        exit(1);
    }

    // Read input HDF5 file to read rate-matching output.
    hdf5hpp::hdf5_file input_file = hdf5hpp::hdf5_file::open(argv[1]);

    // This example only processes the first TB (Transport Block) of the input HDF file.
    int num_TBs = 1;
    tensor_pinned_R_64F input_data = typed_tensor_from_dataset<LWPHY_R_64F, pinned_alloc>(input_file.open_dataset("tb0_layer_mapped"));
    tensor_pinned_C_64F output_data = typed_tensor_from_dataset<LWPHY_C_64F, pinned_alloc>(input_file.open_dataset("tb0_qams"));

    const int rate_matched_bits = input_data.layout().dimensions()[0];
    const int qam_elements = output_data.layout().dimensions()[0];
    int modulation_order  = rate_matched_bits / qam_elements;

    tensor_device d_in_tensor(tensor_info(LWPHY_BIT, {rate_matched_bits}));
    buffer<uint32_t, pinned_alloc> h_in_tensor(d_in_tensor.desc().get_size_in_bytes());

    tensor_device modulation_output(tensor_info(LWPHY_C_16F, {qam_elements}),
                                    LWPHY_TENSOR_ALIGN_TIGHT);

    for (int element_start = 0; element_start < rate_matched_bits; element_start += ELEMENT_SIZE)  {
            uint32_t bits = 0;
            for (int offset = 0; offset < ELEMENT_SIZE; offset++) {
                uint32_t bit = (input_data({element_start + offset}) == 1) ? 1 : 0;
                // 1st element of HDF5 file's tb_codedcbs datatset will map to the
		// least significant bit of a tensor element
		bits |= (bit << offset);
           }
           uint32_t* word = h_in_tensor.addr() + (element_start / ELEMENT_SIZE);
           *word          = bits;
    }

    std::vector<PerTbParams> h_workspace(num_TBs);
    for (int i = 0; i < num_TBs; i++) {
        h_workspace[i].G = rate_matched_bits;
        h_workspace[i].Qm = modulation_order;
        //remaining fields unitialized. Unused in modulation kernel.
    }
    unique_device_ptr<PerTbParams> d_workspace = make_unique_device<PerTbParams>(num_TBs);

    LWDA_CHECK(lwdaMemcpy(d_in_tensor.addr(), h_in_tensor.addr(), d_in_tensor.desc().get_size_in_bytes(), lwdaMemcpyHostToDevice));
    LWDA_CHECK(lwdaMemcpy(d_workspace.get(), h_workspace.data(), num_TBs * sizeof(PerTbParams), lwdaMemcpyHostToDevice));

    std::cout << "Will run modulation_mapper for Transport Block (TB) 0 (HDF5 file: " << argv[1] << ")" << std::endl;
    std::cout << std::endl << "# rate matched bits (input) = " << rate_matched_bits << std::endl;
    std::cout << "modulation order = " << modulation_order << std::endl;
    std::cout << "# symbols (output) = " << qam_elements << std::endl << std::endl;

    lwphyStatus_t status;
    event_timer lwphy_timer;
    lwphy_timer.record_begin();

    for (int iter = 0; iter < num_iterations; iter++) {
        // Note: Calling lwphyModulation with nullptr as a first argument assumes output is a contiguous buffer rather than a 3D {3276, 14, 4} tensor
        // as in the PdschTx example.
        status = lwphyModulation(nullptr, d_in_tensor.desc().handle(), d_in_tensor.addr(),
                                 qam_elements, num_TBs, d_workspace.get(), modulation_output.desc().handle(),
                                 modulation_output.addr(), strm);
    }

    lwdaError_t lwda_error = lwdaGetLastError();
    if (lwda_error != lwdaSuccess) {
        std::cerr << "LWCA Error " << lwdaGetErrorString(lwda_error) << std::endl;
    }

    lwphy_timer.record_end();
    lwphy_timer.synchronize();
    float time1 = lwphy_timer.elapsed_time_ms();

    if (status != LWPHY_STATUS_SUCCESS) {
        throw std::runtime_error("Invalid argument(s) for lwphyModulation");
    }

    time1 /= num_iterations;

    printf("Modulation Mapper Kernel: %.2f us (avg. over %d iterations)\n", time1 * 1000, num_iterations);


    std::vector<__half2> h_modulation_output(qam_elements);
    LWDA_CHECK(lwdaMemcpy(h_modulation_output.data(), modulation_output.addr(), modulation_output.desc().get_size_in_bytes(), lwdaMemcpyDeviceToHost));

    //Reference comparison
    uint32_t gpu_mismatch = 0;
    for (int symbol_id = 0; symbol_id < qam_elements; symbol_id += 1) {
        __half2 ref_symbol;
        ref_symbol.x = (half) output_data({symbol_id}).x;
        ref_symbol.y = (half) output_data({symbol_id}).y;

        if (!complex_approx_equal<__half2, __half>(h_modulation_output[symbol_id], ref_symbol)) {
            printf("Error! Mismatch for QAM symbol %d - expected=%f + i %f vs. gpu=%f + i %f\n", symbol_id,
                   (float) ref_symbol.x, (float) ref_symbol.y,
                   (float) h_modulation_output[symbol_id].x, (float) h_modulation_output[symbol_id].y);
            gpu_mismatch += 1;
        }
    }
    std::cout << "Found " << gpu_mismatch << " mismatched QAM symbols out of " << qam_elements << std::endl;

    return 0;
}
