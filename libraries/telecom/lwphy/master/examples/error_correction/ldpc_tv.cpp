/*
 * Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>
#include "lwphy.hpp"
#include "util.hpp"
#include "hdf5hpp.hpp"
#include "lwphy_hdf5.hpp"
#include "ldpc_tv.hpp"
#include "ldpc_tv_util.hpp"

using namespace lwphy;

// Construct an LDPC test vector from a test vector file
ldpc_tv::ldpc_tv(const char* inputFile, int bg, bool useHalf) :
    inputFilename(inputFile),
    BG(bg),
    useHalf(useHalf)
{
    lwphyDataType_t                                LLR_type = useHalf ? LWPHY_R_16F : LWPHY_R_32F;
    typedef typed_tensor<LWPHY_R_8U, pinned_alloc> tensor_pinned_R_8U;
    hdf5hpp::hdf5_file                             fInput         = hdf5hpp::hdf5_file::open(inputFilename);
    tensor_pinned_R_8U                             tsourceData    = typed_tensor_from_dataset<LWPHY_R_8U, pinned_alloc>(fInput.open_dataset("sourceData"));
    tensor_pinned_R_8U                             tinputCodeWord = typed_tensor_from_dataset<LWPHY_R_8U, pinned_alloc>(fInput.open_dataset("inputCodeWord"));
    if(useHalf)
    {
        d_LLR_half_tensor = tensor_from_dataset(fInput.open_dataset("inputLLR"), LLR_type, LWPHY_TENSOR_ALIGN_COALESCE);
        C                 = d_LLR_half_tensor.dimensions()[1];
        N                 = d_LLR_half_tensor.dimensions()[0];
    }
    else
    {
        d_LLR_tensor = tensor_from_dataset(fInput.open_dataset("inputLLR"), LLR_type, LWPHY_TENSOR_ALIGN_COALESCE);
        C            = d_LLR_tensor.dimensions()[1];
        N            = d_LLR_tensor.dimensions()[0];
    }

    K = tsourceData.dimensions()[0];

    if(BG == 1)
    {
        Kb = 22;
    }
    else
    {
        // 3GPP TS 38.212, Sec. 5.2.2
        if(K > 640)
        {
            Kb = 10;
        }
        else if(K > 560)
        {
            Kb = 9;
        }
        else if(K > 192)
        {
            Kb = 8;
        }
        else
        {
            Kb = 6;
        }
    }

    Z = K / ((BG == 1) ? 22 : 10);

    // allocate all tensors that are needed
    d_source_tensor  = tensor_device(tensor_info(LWPHY_BIT, tsourceData.layout()));
    d_encoded_tensor = tensor_device(tensor_info(LWPHY_BIT, tinputCodeWord.layout()));
    colwert_to_bit(tsourceData.addr(), tsourceData.desc().get_size_in_bytes(), d_source_tensor.addr());
    colwert_to_bit(tinputCodeWord.addr(), tinputCodeWord.desc().get_size_in_bytes(), d_encoded_tensor.addr());
    validcfg = true;
}

// Construct an LDPC test vector for random data generation
ldpc_tv::ldpc_tv(int bg, int z, int c, bool useHalf) :
    BG(bg),
    Z(z),
    C(c),
    useHalf(useHalf)
{
    validcfg = true;
    switch(BG)
    {
    case 1:
        Kb = 22;
        K  = Z * Kb;
        F  = 0;      // No filter bits for BG1
        N  = 68 * Z; //TOOD: any better way to callwlate this?
        break;
    case 2:
        K = 10 * Z;
        if(Z > 64)
            Kb = 10;
        else if(Z > 56)
            Kb = 9;
        else if(Z >= 20)
            Kb = 8;
        else
            Kb = 6;
        F = (10 - Kb) * Z; // Filter bits for Z <=64
        N = 52 * Z;
        break;
    default:
        validcfg = false;
    }

    if(!validcfg)
        return;

    // Round to 32
    int K_rnd = K + ((K % 32 != 0) ? (32 - (K % 32)) : 0);
    int N_rnd = N + ((N % 32 != 0) ? (32 - (N % 32)) : 0);

    // allocate all tensors that are needed
    d_source_tensor  = tensor_device(tensor_info(LWPHY_BIT, {K_rnd, 1}));
    d_encoded_tensor = tensor_device(tensor_info(LWPHY_BIT, {N_rnd, 1}));
    d_LLR_tensor     = tensor_device(tensor_info(LWPHY_R_32F, {N_rnd, C}));
    if(useHalf)
        d_LLR_half_tensor = tensor_device(tensor_info(LWPHY_R_16F, {N_rnd, C}));
}

// Generate random test vector data
lwphyStatus_t ldpc_tv::generate_tv(float SNR, bool puncture, hdf5hpp::hdf5_file* dbgfile)
{
    int   redv     = 0;    //redundancy version
    float codeRate = 0.0f; //code rate

    this->SNR      = SNR;
    this->puncture = puncture;

    if(inputFilename)
        return (LWPHY_STATUS_UNSUPPORTED_TYPE);

    if(!validcfg)
        return (LWPHY_STATUS_ILWALID_ARGUMENT);

    gpu_gen_rand_bit(d_source_tensor.addr(), get_num_elements(d_source_tensor) - F);
    LWPHY_CHECK(lwphyErrorCorrectionLDPCEncode(d_source_tensor.desc().handle(),
                                               d_source_tensor.addr(),
                                               d_encoded_tensor.desc().handle(),
                                               d_encoded_tensor.addr(),
                                               BG,
                                               Kb,
                                               Z,
                                               false,
                                               codeRate,
                                               redv));
    lwdaDeviceSynchronize();

    // Colwert the tensor to LWPHY_R_32F format for adding noise
    tensor_device d_colw_tensor = tensor_device(tensor_info(LWPHY_R_32F, d_encoded_tensor.layout()));
    LWPHY_CHECK(lwphyColwertTensor(d_colw_tensor.desc().handle(), d_colw_tensor.addr(), d_encoded_tensor.desc().handle(), d_encoded_tensor.addr(), 0));

    // Colwert the data to symbols
    LWDA_CHECK(gpu_colwert_symbol(static_cast<float*>(d_colw_tensor.addr()), get_num_elements(d_colw_tensor)));

    // replicate it
    auto dl = d_colw_tensor.layout().dimensions();
    LWDA_CHECK(gpu_repmat(static_cast<float*>(d_colw_tensor.addr()),
                          static_cast<float*>(d_LLR_tensor.addr()),
                          dl[0],
                          dl[1],
                          C));
    if(dbgfile) lwphy::write_HDF5_dataset(*dbgfile, d_LLR_tensor, "replicatedData");

    LWDA_CHECK(gpu_add_noise(static_cast<float*>(d_LLR_tensor.addr()), get_num_elements(d_LLR_tensor), SNR));
    if(dbgfile) lwphy::write_HDF5_dataset(*dbgfile, d_LLR_tensor, "LLRData");

    if(F != 0)
    {
        float inf = std::numeric_limits<float>::infinity();
        auto  dl  = d_LLR_tensor.layout().dimensions();

        LWDA_CHECK(gpu_init_elem(static_cast<float*>(d_LLR_tensor.addr()), dl[0], dl[1], K - F, F, inf));
        if(dbgfile) lwphy::write_HDF5_dataset(*dbgfile, d_LLR_tensor, "LLRDataFilled");
    }

    if(puncture)
    {
        auto dl = d_LLR_tensor.layout().dimensions();
        LWDA_CHECK(gpu_init_elem(static_cast<float*>(d_LLR_tensor.addr()), dl[0], dl[1], 0, 2 * Z, 0.0f));
        if(dbgfile) lwphy::write_HDF5_dataset(*dbgfile, d_LLR_tensor, "LLRdataPunctured");
    }
    if(useHalf)
    {
        LWPHY_CHECK(lwphyColwertTensor(d_LLR_half_tensor.desc().handle(), d_LLR_half_tensor.addr(), d_LLR_tensor.desc().handle(), d_LLR_tensor.addr(), 0));
        lwdaDeviceSynchronize();
    }
    if(lwdaGetLastError() != lwdaSuccess)
        return (LWPHY_STATUS_INTERNAL_ERROR);
    return LWPHY_STATUS_SUCCESS;
}

// Verify the decoded data with that of the source data
lwphyStatus_t ldpc_tv::verify_decode(tensor_device& tDecode, uint32_t& bitErr, uint32_t& blockErr, int numCBLimit)
{
    int                                   codeBlocks = (numCBLimit > 0) ? numCBLimit : tDecode.dimensions()[1];
    int                                   words      = (Kb * Z) / 32;
    typed_tensor<LWPHY_BIT, pinned_alloc> source(d_source_tensor.layout());
    typed_tensor<LWPHY_BIT, pinned_alloc> decoded(tDecode.layout());

    source       = d_source_tensor;
    decoded      = tDecode;
    uint32_t* s  = (uint32_t*)source.addr();
    uint32_t* d  = (uint32_t*)decoded.addr();
    int       d0 = decoded.dimensions()[0] / 32;

    bitErr   = 0;
    blockErr = 0;
    bool berr;
    for(int i = 0; i < codeBlocks; i++)
    {
        berr = false;
        for(int j = 0; j < words; j++)
        {
            if(s[j] != d[i * d0 + j])
            {
                uint32_t diff = s[j] ^ d[i * d0 + j];
                while(diff)
                {
                    bitErr++;
                    diff = diff & (diff - 1);
                }
                berr = true;
            }
        }
        if(berr)
            blockErr++;
    }
    if(bitErr)
    {
        return (LWPHY_STATUS_INTERNAL_ERROR);
    }
    return (LWPHY_STATUS_SUCCESS);
}

lwphyStatus_t ldpc_tv::verify_decode(tensor_device& tDecode, int numCBLimit)
{
    uint32_t bitErr, blockErr;

    return (verify_decode(tDecode, bitErr, blockErr, numCBLimit));
}

// Save the test vector to a file
void ldpc_tv::save_tv(const char* filename)
{
    char fname[100];
    if(!filename)
        sprintf(fname, "ldpc_BG%d_K%d_SNR%.2f_%d_%s.h5", BG, K, SNR, C, puncture ? "p" : "");
    else
        sprintf(fname, "%s", filename);

    hdf5hpp::hdf5_file ofile = hdf5hpp::hdf5_file::create(fname);
    lwphy::write_HDF5_dataset(ofile, d_source_tensor, "sourceData");
    lwphy::write_HDF5_dataset(ofile, d_encoded_tensor, "inputCodeWord");
    lwphy::write_HDF5_dataset(ofile, d_LLR_tensor, "inputLLR");
}
