/*
 * Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 */

#include "lwphy.h"
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>
#include <getopt.h>
#include "lwphy.hpp"
#include "lwphy_hdf5.hpp"
#include "hdf5hpp.hpp"

using namespace lwphy;

uint32_t N_ITER = 200;

void usage(char *prog)
{
    printf("%s [options] [test-vector-file]\n", prog);
    printf("  Options:\n");
    printf("    -g base_graph          Base graph (default: 1)\n");
    printf("    -p                     enable punlwture (default: no)\n");
}

int main(int argc, char* argv[])
{
    std::string inputFilename = "mat_gen_data.hdf5";
    extern char *optarg;
    extern int optind;
    int option, bg = 1;
    bool puncture = false;

    while((option = getopt(argc, argv, "pg:")) != -1) {
        switch (option) {
        case 'p':
                puncture = true;
                break;
        case 'g':
                bg = atoi(optarg);
                if (bg < 1 || bg > 2) {
                    printf("Invalid basegraph number: %d\n", bg);
                    usage(argv[0]);
                    return (EILWAL);
                }
                break;
        default :
                printf("Invalid option: %s", argv[optind]);
                usage(argv[0]);
                return (EILWAL);
        }
    }

    if (optind < argc) {
        inputFilename.assign(argv[optind]);
    }

    hdf5hpp::hdf5_file fInput = hdf5hpp::hdf5_file::open(inputFilename.c_str());

    using tensor_pinned_R_8U = typed_tensor<LWPHY_R_8U, pinned_alloc>;

    tensor_pinned_R_8U tSourceData = typed_tensor_from_dataset<LWPHY_R_8U, pinned_alloc>(fInput.open_dataset("sourceData"));
    tensor_device d_EncodedData;
    try {
        lwphy::disable_hdf5_error_print();
        d_EncodedData = tensor_from_dataset(fInput.open_dataset("encodedData"));
    } catch(std::exception& e) {
        // test vectors may be using 'inputCodeWord'
        d_EncodedData = tensor_from_dataset(fInput.open_dataset("inputCodeWord"));
    }
    tensor_pinned_R_8U tEncodedData(d_EncodedData.layout());
    lwphyStatus_t s  = lwphyColwertTensor(tEncodedData.desc().handle(),
                                            tEncodedData.addr(),
                                            d_EncodedData.desc().handle(),
                                            d_EncodedData.addr(), 0);

    const int  K        = tSourceData.dimensions()[0];
    const int  C        = tSourceData.dimensions()[1];
    int F = 0;
    int BG = bg;
    int ncwnodes, Kb, Z;

    if (BG == 1) {
        Kb = 22;
        ncwnodes = 66;
        Z = K / Kb;
    } else {
        Z = K / 10;
        if (Z > 64)
            Kb = 10;
        else if(Z > 56)
            Kb = 9;
        else if(Z >= 20)
            Kb = 8;
        else
            Kb = 6;
        F = (10 - Kb) * Z;
        ncwnodes = 50;
    }

    const int  N        = Z * (ncwnodes + (puncture ? 0 : 2));
    printf("K: %d C: %d BG: %d Z: %d N: %d\n", K, C, BG, Z, N);

    if(N != tEncodedData.dimensions()[0])
    {
        printf("ERROR: the wrongly structured reference output: %d vs. %d\n",
               N,
               tEncodedData.dimensions()[0]);
        exit(1);
    }

    int           K_padding = (K % 32 != 0) * (32 - (K % 32));
    tensor_device d_in_tensor(tensor_info(LWPHY_BIT, {K + K_padding, C}));

    buffer<uint32_t, pinned_alloc> h_in_tensor(d_in_tensor.desc().get_size_in_bytes());

    for(int c = 0; c < C; c++)
    {
        for(int k = 0; k < K; k += 32)
        {
            uint32_t bits = 0;
            for(int o = 0; o < 32; o++)
            {
                if(k + o < K)
                {
                    uint32_t bit = tSourceData({k + o, c}) & 0x1;
                    bits |= (bit << o);
                }
            }
            uint32_t* word = h_in_tensor.addr() + (k / 32) + ((K + K_padding) / 32) * c;
            *word          = bits;
        }
    }
    lwdaMemcpy(d_in_tensor.addr(), h_in_tensor.addr(), d_in_tensor.desc().get_size_in_bytes(), lwdaMemcpyHostToDevice);

    int           N_padding = (N % 32 != 0) * (32 - (N % 32));
    int           rv = 0; //redundancy version
    int           max_parity_nodes = 0; // treated as unknown when 0
    tensor_device d_out_tensor(tensor_info(LWPHY_BIT, {N + N_padding, C}));

    printf("lwphyErrorCorrectionLDPCEncode\n");
    lwphyErrorCorrectionLDPCEncode(d_in_tensor.desc().handle(),
                                   d_in_tensor.addr(),
                                   d_out_tensor.desc().handle(),
                                   d_out_tensor.addr(),
                                   BG,
                                   Kb,
                                   Z,
                                   puncture,
                                   max_parity_nodes,
                                   rv);
    lwdaDeviceSynchronize();

    typed_tensor<LWPHY_BIT, pinned_alloc> h_out_tensor(d_out_tensor.layout());
    h_out_tensor = d_out_tensor;

    int k_error_word_count = 0;
    int n_error_word_count = 0;
    for(int c = 0; c < C; c++)
    {
        for(int n = 0; n < N; n += 32)
        {
            uint32_t bits = 0;
            for(int o = 0; o < 32; o++)
            {
                if(n + o < N)
                {
                    uint32_t bit = tEncodedData({n + o, c}) & 0x1;
                    bits |= (bit << o);
                }
            }
            uint32_t val = h_out_tensor({n / 32, c});
            if(val != bits)
            {
                printf("(%d) %x vs. %x in (%d, %d)\n", bits == val, bits, val, c, n);
                if(n < K)
                    k_error_word_count++;
                else
                    n_error_word_count++;
            }
        }
    }
    printf("k_error_word_count: %d n_error_word_count: %d\n", k_error_word_count, n_error_word_count);

    if(k_error_word_count || n_error_word_count)
    {
        exit(1);
    }

    lwdaEvent_t start, stop;
    lwdaEventCreate(&start);
    lwdaEventCreate(&stop);

    float time = 0.0f;
    lwdaEventRecord(start);

    for(int i = 0; i < N_ITER; i++)
    {
        lwphyErrorCorrectionLDPCEncode(d_in_tensor.desc().handle(),
                                       d_in_tensor.addr(),
                                       d_out_tensor.desc().handle(),
                                       d_out_tensor.addr(),
                                       BG,
                                       Kb,
                                       Z,
                                       puncture,
                                       max_parity_nodes,
                                       rv);
    }

    lwdaEventRecord(stop);
    lwdaEventSynchronize(stop);
    lwdaEventElapsedTime(&time, start, stop);

    time /= N_ITER;
    printf("LDPC Encoder: %.3f us\n", time * 1000);

    lwdaError_t err = lwdaGetLastError();
    if(err != lwdaSuccess)
    {
        printf("ERROR: %s\n", lwdaGetErrorString(err));
        exit(1);
    }

    return 0;
}
