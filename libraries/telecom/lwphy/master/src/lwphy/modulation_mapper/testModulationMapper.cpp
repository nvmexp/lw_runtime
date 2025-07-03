/*
 * Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 */

#include <gtest/gtest.h>
#include <iostream>
#include <vector>
#include "lwphy.h"
#include "lwphy_internal.h"

//using namespace std;
using namespace lwphy_i;

struct TestParams {
    int num_symbols;
    int modulation_order;
    unsigned int seed;
};

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

__half2 symbol_modulation(uint32_t shifted_input_element, uint32_t modulation_order) {

    uint32_t symbol_bits = shifted_input_element & ((1 << modulation_order) - 0x1U);
    __half2 cpu_computed_symbol;

    std::vector<uint32_t> bit_vals(modulation_order);
    for (int i = 0; i < modulation_order; i++) {
        bit_vals[i] = ((symbol_bits >> i) & 0x1);
    }

    if (modulation_order == LWPHY_QAM_4) {
        // QPSK  modulation
	cpu_computed_symbol.x =  (int)(1 - 2*bit_vals[0]) / sqrt(2);
	cpu_computed_symbol.y =  (int)(1 - 2*bit_vals[1]) / sqrt(2);
    } else if (modulation_order == LWPHY_QAM_16) {
        // QAM16 modulation
	cpu_computed_symbol.x =  (int)((1 - 2*bit_vals[0]) * (1 + 2*bit_vals[2])) / sqrt(10);
	cpu_computed_symbol.y =  (int)((1 - 2*bit_vals[1]) * (1 + 2*bit_vals[3])) / sqrt(10);
    } else if (modulation_order == LWPHY_QAM_64) {
        // QAM64 modulation
        cpu_computed_symbol.x =  (int)((1 - 2*bit_vals[0]) * (4 - (1 - 2*bit_vals[2]) * (1 + 2*bit_vals[4]))) / sqrt(42);
        cpu_computed_symbol.y =  (int)((1 - 2*bit_vals[1]) * (4 - (1 - 2*bit_vals[3]) * (1 + 2*bit_vals[5]))) / sqrt(42);
    } else if (modulation_order == LWPHY_QAM_256) {
        // QAM256 modulation
        cpu_computed_symbol.x =  (int)((1 - 2*bit_vals[0]) * (8 - (1 - 2*bit_vals[2]) * (4 - (1 - 2*bit_vals[4]) * (1 + 2*bit_vals[6])))) / sqrt(170);
        cpu_computed_symbol.y =  (int)((1 - 2*bit_vals[1]) * (8 - (1 - 2*bit_vals[3]) * (4 - (1 - 2*bit_vals[5]) * (1 + 2*bit_vals[7])))) / sqrt(170);
    } else {
        std::cout << "Unsupported moudlation order " << modulation_order << std::endl;
    }

    return cpu_computed_symbol;
}


int reference_comparison(std::vector<uint32_t> & h_modulation_input, std::vector<__half2> & h_modulation_output, int modulation_order, int num_symbols) {

    const int uint32_t_bits = 32;
    int mismatch_cnt = 0;

    for (int symbol_id = 0; symbol_id < num_symbols; symbol_id++) {
        int input_element = (symbol_id * modulation_order) / uint32_t_bits;
        int symbol_start_bit = (symbol_id * modulation_order) % uint32_t_bits;
        uint32_t shifted_input_element = (h_modulation_input[input_element] >> symbol_start_bit);

        if (modulation_order == LWPHY_QAM_64) {
            if (symbol_start_bit == 28) {
                shifted_input_element &= 0x0FU;
                shifted_input_element |= ((h_modulation_input[input_element + 1] & 0x03U) << 4);
            } else if (symbol_start_bit == 30) {
                shifted_input_element &= 0x03U;
                shifted_input_element |= ((h_modulation_input[input_element + 1] & 0x0FU) << 2);
            }
        }

        __half2 cpu_computed_symbol = symbol_modulation(shifted_input_element, modulation_order);

        if (!complex_approx_equal<__half2, __half>(cpu_computed_symbol, h_modulation_output[symbol_id])) {
            if (mismatch_cnt == 0) {
                //std::cout << "Mismatch for symbol " << symbol_id << " = " << std::hex << shifted_input_element;
                std::cout << "First Mismatch for symbol " << symbol_id << " = " << std::hex << shifted_input_element;
                std::cout << ": CPU val. " << (float) cpu_computed_symbol.x << " + i " << (float) cpu_computed_symbol.y;
                std::cout << " vs. GPU val. " << (float) h_modulation_output[symbol_id].x  << " + i " << (float) h_modulation_output[symbol_id].y << std::dec << std::endl;
            }

            mismatch_cnt += 1;
        }
    }

    std::cout << "Found " << mismatch_cnt << " mismatches out of " << num_symbols << " symbols." << std::endl;
    return mismatch_cnt;
}

void test_modulation(TestParams & test_params, int & gpu_mismatch, int num_iterations) {

    lwdaStream_t strm = 0;

    // Randomly populate
    srand(test_params.seed);

    int num_symbols = test_params.num_symbols;
    int modulation_order = test_params.modulation_order;

    if ((modulation_order != LWPHY_QAM_4) && (modulation_order != LWPHY_QAM_16) &&
       (modulation_order != LWPHY_QAM_64) && (modulation_order != LWPHY_QAM_256)) {
        std::cerr << "Invalid Modulation order " << modulation_order << " is not supported" << std::endl;
        return;
    }

    int num_TBs = 1;
    int num_bits = num_symbols * modulation_order;
    int input_elements = div_round_up<uint32_t>(num_bits, 8*sizeof(uint32_t));
    unique_device_ptr<uint32_t> modulation_input = make_unique_device<uint32_t>(input_elements);
    std::vector<PerTbParams> h_workspace(num_TBs);
    for (int i = 0; i < num_TBs; i++) {
        h_workspace[i].G = num_bits;
        h_workspace[i].Qm = modulation_order;
        //TODO remaining fields unitialized. Unused in modulation kernel.
    }
    unique_device_ptr<PerTbParams> d_workspace = make_unique_device<PerTbParams>(num_TBs);
    lwphyTensorDescriptor_t input_desc, output_desc;
    lwphyCreateTensorDescriptor(&input_desc);
    lwphyCreateTensorDescriptor(&output_desc);
    int input_dims[1] = {input_elements};
    int output_dims[1] = {num_symbols};
    lwphySetTensorDescriptor(input_desc, LWPHY_R_32U, 1, input_dims, nullptr, 0);
    lwphySetTensorDescriptor(output_desc, LWPHY_C_16F, 1, output_dims, nullptr, 0);

    unique_device_ptr<__half2> modulation_output = make_unique_device<__half2>(num_symbols);

    //Randomly populate modulation_input.
    std::vector<uint32_t> h_modulation_input(input_elements);
    for (int i = 0; i < input_elements; i++) {
        h_modulation_input[i] = rand();
    }
    LWDA_CHECK(lwdaMemcpy(modulation_input.get(), h_modulation_input.data(), input_elements * sizeof(uint32_t), lwdaMemcpyHostToDevice));
    LWDA_CHECK(lwdaMemcpy(d_workspace.get(), h_workspace.data(), num_TBs * sizeof(PerTbParams), lwdaMemcpyHostToDevice));

    //A dummy call is needed or otherwise the event measurement for the first call is incorrect.
    lwphyModulation(nullptr, input_desc, nullptr, num_symbols, num_TBs, d_workspace.get(), output_desc, nullptr, strm);
    LWDA_CHECK(lwdaDeviceSynchronize());

    float time1 = 0.0;

    lwphyStatus_t status;
    lwdaEvent_t start, stop;
    LWDA_CHECK(lwdaEventCreate(&start, lwdaEventBlockingSync));
    LWDA_CHECK(lwdaEventCreate(&stop, lwdaEventBlockingSync));

    LWDA_CHECK(lwdaEventRecord(start));

    for (int iter = 0; iter < num_iterations; iter++) {
        status = lwphyModulation(nullptr, input_desc, modulation_input.get(), num_symbols, num_TBs,
                                 d_workspace.get(), output_desc, modulation_output.get(), strm);
    }

    lwdaError_t lwda_error = lwdaGetLastError();
    if (lwda_error != lwdaSuccess) {
        std::cerr << "LWCA Error " << lwdaGetErrorString(lwda_error) << std::endl;
    }

    lwdaEventRecord(stop);
    LWDA_CHECK(lwdaEventSynchronize(stop));
    LWDA_CHECK(lwdaEventElapsedTime(&time1, start, stop));

    LWDA_CHECK(lwdaEventDestroy(start));
    LWDA_CHECK(lwdaEventDestroy(stop));

    if (status != LWPHY_STATUS_SUCCESS) {
        throw std::runtime_error("Invalid argument(s) for lwphyModulation");
    }

    time1 /= num_iterations;


    printf("Modulation Mapper Kernel: %.2f us (avg. over %d iterations)\n", time1 * 1000, num_iterations);

    std::vector<__half2> h_modulation_output(num_symbols);
    LWDA_CHECK(lwdaMemcpy(h_modulation_output.data(), modulation_output.get(), num_symbols * sizeof(__half2), lwdaMemcpyDeviceToHost));
    gpu_mismatch = reference_comparison(h_modulation_input, h_modulation_output, modulation_order, num_symbols);

}

class ModulationMapperTest: public ::testing::TestWithParam<TestParams> {
public:
    void basicTest() {
        params = ::testing::TestWithParam<TestParams>::GetParam();
        test_modulation(params, gpu_mismatch, num_iterations);
    }

    void SetUp() override {basicTest(); }

    void TearDown() override {
        gpu_mismatch = -1;
    }

protected:
    TestParams params;
    int gpu_mismatch = -1;
    int num_iterations = 20;
};

TEST_P(ModulationMapperTest, CONFIGS) {
    //EXPECT_EQ(0, gpu_mismatch);
    ASSERT_TRUE(gpu_mismatch == 0);
}

const std::vector<TestParams> CONFIGS = {
    /* number of symbols, modulation order, random seed to initialize input bits */

    {100, LWPHY_QAM_4, 2019},
    {235872, LWPHY_QAM_4, 1024},
    {10000001, LWPHY_QAM_4, 1024},

    {100, LWPHY_QAM_16, 2019},
    {235872, LWPHY_QAM_16, 1024},
    {10000001, LWPHY_QAM_16, 1024},

    {100, LWPHY_QAM_64, 2019},
    {235872, LWPHY_QAM_64, 1024},
    {10000001, LWPHY_QAM_64, 1024},

    {100, LWPHY_QAM_256, 2019},
    {36036, LWPHY_QAM_256, 2019},
    {235872, LWPHY_QAM_256, 1024},
    {10000001, LWPHY_QAM_256, 1024}

};

INSTANTIATE_TEST_CASE_P(ModulationMapperTests, ModulationMapperTest,
                        ::testing::ValuesIn(CONFIGS));

//TODO add test cases + code to test non identical TB configs too

int main(int argc, char** argv) {

#if 1
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
#else // To debug an individual config
    int gpu_mismatch = -1;
    TestParams params = {36036, LWPHY_QAM_256, 2019};
    test_modulation(params, gpu_mismatch, 1);

    return 0;
#endif
}
