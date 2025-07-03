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
#include "lwphy.hpp"
#include "lwphy_hdf5.hpp"
#include "hdf5hpp.hpp"
#include "util.hpp"

#include <cstdlib>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

using namespace std;
using namespace lwphy;

template <typename TScalar>
bool compare_approx(const TScalar& lhs, const TScalar& rhs, const TScalar threshold)
{
    TScalar diff  = fabs(lhs - rhs);
    TScalar m     = std::max(fabs(lhs), fabs(rhs));
    TScalar ratio = (diff > threshold) ? (diff / m) : diff;

    return ratio < threshold;
}

int main(int argc, char* argv[])
{
    if(argc != 2)
    {
        std::cerr << "no input file specified" << std::endl;
        exit(1);
    }

    hdf5hpp::hdf5_file input_file = hdf5hpp::hdf5_file::open(argv[1]);

    typed_tensor<LWPHY_C_32F, pinned_alloc> h_tf_signal32   = typed_tensor_from_dataset<LWPHY_C_32F, pinned_alloc>(input_file.open_dataset("tf_signal"));
    typed_tensor<LWPHY_C_32F, pinned_alloc> h_qam_payload32 = typed_tensor_from_dataset<LWPHY_C_32F, pinned_alloc>(input_file.open_dataset("qam_payload"));
    typed_tensor<LWPHY_C_32F, pinned_alloc> ref_output      = typed_tensor_from_dataset<LWPHY_C_32F, pinned_alloc>(input_file.open_dataset("ref_output"));

    auto ph5 = get_HDF5_struct(input_file, "params");

    PdcchParams params;

    // read params from the file except qam_payload
    params.n_f         = ph5.get_value_as<int>("n_f");
    params.n_t         = ph5.get_value_as<int>("n_t");
    params.slot_number = ph5.get_value_as<int>("slot_number");
    params.start_rb    = ph5.get_value_as<int>("startRb");
    params.n_rb        = ph5.get_value_as<int>("n_rb");
    params.start_sym   = ph5.get_value_as<int>("start_sym");
    params.n_sym       = 1;
    params.dmrs_id     = ph5.get_value_as<int>("dmrs_id");
    params.beta_qam    = ph5.get_value_as<float>("beta_qam");
    params.beta_dmrs   = ph5.get_value_as<float>("beta_dmrs");

    buffer<__half2, pinned_alloc> h_tf_signal16(params.n_f * params.n_t * sizeof(__half2));
    for(int t = 0; t < params.n_t; t++)
    {
        for(int f = 0; f < params.n_f; f++)
        {
            float2   src = h_tf_signal32({f, t});
            __half2* dst = h_tf_signal16.addr() + t + params.n_t * f;
            dst->x       = src.x;
            dst->y       = src.y;
        }
    }
    tensor_device d_tf_signal(tensor_info(LWPHY_C_16F, {params.n_f, params.n_t}));
    lwdaMemcpy(d_tf_signal.addr(), h_tf_signal16.addr(), d_tf_signal.desc().get_size_in_bytes(), lwdaMemcpyHostToDevice);

    int n_qam = params.n_rb * 12;

    buffer<__half2, pinned_alloc> h_qam_payload16(n_qam * sizeof(__half2));
    for(int i = 0; i < n_qam; i++)
    {
        float2   src = h_qam_payload32({i});
        __half2* dst = h_qam_payload16.addr() + i;
        dst->x       = src.x;
        dst->y       = src.y;
    }
    tensor_device d_qam_payload(tensor_info(LWPHY_C_16F, {n_qam}));
    lwdaMemcpy(d_qam_payload.addr(), h_qam_payload16.addr(), d_qam_payload.desc().get_size_in_bytes(), lwdaMemcpyHostToDevice);

    // set params.qam_payload
    params.qam_payload = (__half2*)d_qam_payload.addr();

    printf("PDCCH parameters....\n");
    printf(" n_f: %d\n", params.n_f);
    printf(" n_t: %d\n", params.n_t);
    printf(" slot_number: %d\n", params.slot_number);
    printf(" start_rb: %d\n", params.start_rb);
    printf(" n_rb: %d\n", params.n_rb);
    printf(" start_sym: %d\n", params.start_sym);
    printf(" n_sym: %d\n", params.n_sym);
    printf(" dmrs_id: %d\n", params.dmrs_id);
    printf(" beta_qam: %.3f\n", params.beta_qam);
    printf(" beta_dmrs: %.3f\n", params.beta_dmrs);
    printf(" qam_payload addr: %p\n", (void*)params.qam_payload);

    printf("Run lwphyPdcchTfSignal...\n");
    lwphyPdcchTfSignal(d_tf_signal.desc().handle(), d_tf_signal.addr(), params, 0);

    lwdaDeviceSynchronize();

    typed_tensor<LWPHY_C_16F, pinned_alloc> h_output(d_tf_signal.layout());
    h_output = d_tf_signal;

    printf("Check Correctness....\n");
    int n_qam_mismatch  = 0;
    int n_dmrs_mismatch = 0;
    int n_other         = 0;
    for(int t = 0; t < params.n_t; t++)
    {
        for(int f = 0; f < params.n_f; f++)
        {
            float2  ref     = ref_output({f, t});
            __half2 out     = h_output({f, t});
            bool    valid_x = compare_approx<float>(ref.x, (float)out.x, 1e-4);
            bool    valid_y = compare_approx<float>(ref.y, (float)out.y, 1e-4);
            if(!valid_x || !valid_y)
            {
                if(f >= (params.start_rb * 12) && f < ((params.start_rb + params.n_rb) * 12))
                {
                    if(f % 4 == 1)
                    {
                        printf("dmrs: (%d, %d): %f vs %f and %f vs %f\n", f, t, ref.x, (float)out.x, ref.y, (float)out.y);
                        n_dmrs_mismatch++;
                    }
                    else
                    {
                        printf("qam: (%d, %d): %f vs %f and %f vs %f\n", f, t, ref.x, (float)out.x, ref.y, (float)out.y);
                        n_qam_mismatch++;
                    }
                }
                else
                {
                    n_other++;
                }
            }
        }
    }

    printf("# qam mismatch: %d # dmrs mismatch: %d # other: %d\n", n_qam_mismatch, n_dmrs_mismatch, n_other);

    return 0;
}
