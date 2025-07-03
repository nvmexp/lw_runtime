/*
 * Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 */

#if !defined(LDPC_TV_HPP_INCLUDED_)
#define LDPC_TV_HPP_INCLUDED_

#include "lwphy.hpp"
#include "hdf5hpp.hpp"

/*
 * LDPC Test Vector generator class.
 */
class ldpc_tv {
public:
    // Create an instance with an existing test vector file
    ldpc_tv(const char* inputFilename, int BG = 1, bool useHalf = false);

    // Create an instance with essential arguments for random data generation.
    ldpc_tv(int BG, int Z, int C, bool useHalf = false);

    // Generate the random test vector data
    lwphyStatus_t generate_tv(float               SNR,
                              bool                puncture,
                              hdf5hpp::hdf5_file* dbgfile = NULL);

    // Verify the given decoded data matches with the source data
    lwphyStatus_t verify_decode(lwphy::tensor_device& tDecode, int numCBLimit = -1);
    lwphyStatus_t verify_decode(lwphy::tensor_device& tDecode,
                                uint32_t&             bitErr,
                                uint32_t&             blockErr,
                                int                   numCBLimit = -1);

    // Save the test vector to a file
    void save_tv(const char* filename = NULL);

    lwphy::tensor_device& get_source_tensor()
    {
        return (d_source_tensor);
    }

    lwphy::tensor_device& get_encoded_tensor()
    {
        return (d_encoded_tensor);
    }

    lwphy::tensor_device& get_LLR_tensor()
    {
        if(useHalf)
            return (d_LLR_half_tensor);
        return (d_LLR_tensor);
    }

    void get_params(int& BG, int& Z, int& K, int& Kb, int& codeBlocks, int& N)
    {
        BG         = this->BG;
        K          = this->K;
        Kb         = this->Kb;
        codeBlocks = this->C;
        N          = this->N;
        Z          = this->Z;
    }

private:
    // Assumes 2D tensor
    int get_num_elements(lwphy::tensor_device& t)
    {
        auto dl = t.layout().dimensions();
        return (dl[0] * dl[1]);
    }

    int         Z;                    // lifting size
    int         C;                    // number of code blocks
    int         BG;                   // Base graph number
    int         K;                    // coded bits
    int         Kb;                   // message bits
    int         N;                    //
    int         F;                    // filler bits
    float       SNR;                  // Signal to noise ratio
    bool        puncture;             // puncture the LLR data?
    bool        useHalf;              // Use half precision LLR data
    const char* inputFilename = NULL; // test vector file name
    bool        validcfg;             // configuration is valid?

    lwphy::tensor_device d_source_tensor;
    lwphy::tensor_device d_encoded_tensor;
    lwphy::tensor_device d_LLR_tensor;
    lwphy::tensor_device d_LLR_half_tensor;
};

#endif // !defined(LDPC_TV_HPP_INCLUDED_
