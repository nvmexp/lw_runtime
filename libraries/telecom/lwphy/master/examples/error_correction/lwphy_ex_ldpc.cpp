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
#include "lwphy.hpp"
#include "lwphy_hdf5.hpp"
#include "hdf5hpp.hpp"
#include "ldpc_tv.hpp"

using namespace lwphy;

////////////////////////////////////////////////////////////////////////
// usage()
void usage()
{
    printf("lwphy_ex_ldpc [options]\n");
    printf("  Options:\n");
    printf("    -a algo_index          Use specific implementation (1 or 2) (default: 0 - let library decide)\n");
    printf("    -d                     Disable early termination\n");
    printf("    -f                     Use half precision instead of single precision (Volta and later only)\n");
    printf("    -g base_graph          Base graph (default: 1)\n");
    printf("    -h                     Display usage information\n");
    printf("    -i input_filename      Input HDF5 file name, which must contain the following datasets:\n");
    printf("                               sourceData:    uint8 data set with source information bits\n");
    printf("                               inputLLR:      Log-likelihood ratios for coded, modulated (BPSK) symbols\n");
    printf("                               inputCodeWord: uint8 data set with encoded bits (optional)\n");
    printf("                                              (Initial bits are the same as sourceData. No puncturing assumed.)\n");
    printf("    -k                     Skip 'warmup' run before timing loop\n");
    printf("    -m normalization       Normalization factor for min-sum (default: 0.8125)\n");
    printf("    -n num_iter            Maximum number of LDPC iterations (default: 1)\n");
    printf("    -p mb                  Number of parity nodes mb (must be between 4 and 46 for BG1) (default: 8)\n");
    printf("    -r num_runs            Number of times to perform batch decoding (default: 1)\n");
    printf("    -s                     Skip comparison of decoder output to input data\n");
    printf("    -w numCBLimit          Decode numCBLimit code blocks (instead of the number contained in the input file).\n");
    printf("                           numCBLimit must be less than the numer of codeblocks in the input file.\n");
    printf("                           For random data, numCBLimit number of codeblocks generated.\n");
    printf("    -x flags               Internal development flags (default: 0)\n");
    printf("    -P                     Puncture the generated test vector data(default: false)\n");
    printf("    -S SNR                 SNR for random data generation\n");
    printf("    -Z size                lifting size for random data generation\n");
}

////////////////////////////////////////////////////////////////////////
// main()
int main(int argc, char* argv[])
{
    int returlwalue = 0;
    try
    {
        //------------------------------------------------------------------
        // Parse command line arguments
        int          iArg = 1;
        std::string  inputFilename;
        int          numIterations       = 1;
        bool         useHalf             = false;
        int          parityNodes         = 8;
        int          algoIndex           = 0;
        int          flags               = 0;
        bool         compareDecodeOutput = true;
        bool         earlyTermination    = true;
        unsigned int numRuns             = 1;
        int          numCBLimit          = -1;
        int          doWarmup            = true;
        float        minSumNorm          = 0.8125f;
        int          BG                  = 1;
        bool         puncture            = false;
        int          Zi                  = 384;
        float        SNR                 = 10.0f;

        while(iArg < argc)
        {
            if('-' == argv[iArg][0])
            {
                switch(argv[iArg][1])
                {
                case 'a':
                    if((++iArg >= argc) ||
                       (1 != sscanf(argv[iArg], "%i", &algoIndex)) ||
                       (algoIndex < 0))
                    {
                        fprintf(stderr, "ERROR: Invalid algorithm index: %s\n", argv[iArg]);
                        exit(1);
                    }
                    ++iArg;
                    break;
                case 'd':
                    earlyTermination = false;
                    ++iArg;
                    break;
                case 'f':
                    useHalf = true;
                    ++iArg;
                    break;
                case 'g':
                    if((++iArg >= argc) ||
                       (1 != sscanf(argv[iArg], "%i", &BG)) ||
                       (BG < 1)                             ||
                       (BG > 2))
                    {
                        fprintf(stderr, "ERROR: Invalid base graph\n");
                        exit(1);
                    }
                    ++iArg;
                    break;
                case 'h':
                    usage();
                    exit(0);
                    break;
                case 'i':
                    if(++iArg >= argc)
                    {
                        fprintf(stderr, "ERROR: No input file name specified\n");
                        exit(1);
                    }
                    inputFilename.assign(argv[iArg++]);
                    break;
                case 'k':
                    doWarmup = false;
                    ++iArg;
                    break;
                case 'm':
                    if((++iArg >= argc) ||
                       (1 != sscanf(argv[iArg], "%f", &minSumNorm)))
                    {
                        fprintf(stderr, "ERROR: Invalid normalization\n");
                        exit(1);
                    }
                    ++iArg;
                    break;
                case 'n':
                    if((++iArg >= argc) ||
                       (1 != sscanf(argv[iArg], "%i", &numIterations)) ||
                       (numIterations < 0))
                    {
                        fprintf(stderr, "ERROR: Invalid number of iterations\n");
                        exit(1);
                    }
                    ++iArg;
                    break;
                case 'p':
                    if((++iArg >= argc) ||
                       (1 != sscanf(argv[iArg], "%i", &parityNodes)) ||
                       (parityNodes <= 3) ||
                       (parityNodes > 46))
                    {
                        fprintf(stderr, "ERROR: Invalid number of parity nodes: %s\n", argv[iArg]);
                        exit(1);
                    }
                    ++iArg;
                    break;
                case 'r':
                    if((++iArg >= argc) ||
                       (1 != sscanf(argv[iArg], "%u", &numRuns)) ||
                       (numRuns < 1))
                    {
                        fprintf(stderr, "ERROR: Invalid number of runs: %s\n", argv[iArg]);
                        exit(1);
                    }
                    ++iArg;
                    break;
                case 's':
                    compareDecodeOutput = false;
                    ++iArg;
                    break;
                case 'w':
                    if((++iArg >= argc) ||
                       (1 != sscanf(argv[iArg], "%i", &numCBLimit)) ||
                       (numCBLimit < 1))
                    {
                        fprintf(stderr, "ERROR: Invalid number of codewords: %s\n", argv[iArg]);
                        exit(1);
                    }
                    ++iArg;
                    break;
                case 'x':
                    if((++iArg >= argc) ||
                       (1 != sscanf(argv[iArg], "%i", &flags)))
                    {
                        fprintf(stderr, "ERROR: Invalid flags: %s\n", argv[iArg]);
                        exit(1);
                    }
                    ++iArg;
                    break;
                case 'P':
                    puncture = true;
                    ++iArg;
                    break;
                case 'Z':
                    if((++iArg >= argc) ||
                       (1 != sscanf(argv[iArg], "%u", &Zi)) ||
                       (Zi < 1))
                    {
                        fprintf(stderr, "ERROR: Invalid Z : %s\n", argv[iArg]);
                        exit(1);
                    }
                    ++iArg;
                    break;
                case 'S':
                    if((++iArg >= argc) ||
                       (1 != sscanf(argv[iArg], "%f", &SNR)))
                    {
                        fprintf(stderr, "ERROR: Invalid SNR value\n");
                        exit(1);
                    }
                    ++iArg;
                    break;
                default:
                    fprintf(stderr, "ERROR: Unknown option: %s\n", argv[iArg]);
                    usage();
                    exit(1);
                    break;
                }
            }
            else
            {
                fprintf(stderr, "ERROR: Invalid command line argument: %s\n", argv[iArg]);
                exit(1);
            }
        }

        std::unique_ptr<ldpc_tv> tv;
        if(!inputFilename.empty())
        {
            tv.reset(new ldpc_tv(inputFilename.c_str(), BG, useHalf));
        } else {
            if (Zi <= 0 || SNR <= 0.0f) {
                fprintf(stderr, "ERROR: Invalid arguments for random data generation\n");
                usage();
                exit(1);
            }
            if (numCBLimit <= 0)
                numCBLimit = 80; // default number of codeblocks to generate
            tv.reset(new ldpc_tv(BG, Zi, numCBLimit, useHalf));
            LWPHY_CHECK(tv->generate_tv(SNR, puncture));
        }
        lwphy::device gpuDevice;
        printf("%s\n", gpuDevice.desc().c_str());
        lwphyDataType_t LLR_type = useHalf ? LWPHY_R_16F : LWPHY_R_32F;

        tensor_device &tLLR = tv->get_LLR_tensor();
        tensor_device &tSourceData = tv->get_source_tensor();

        //------------------------------------------------------------------
        // Check the requested number of codewords if the user provided an
        // override.
        if((numCBLimit > 0) &&
           (numCBLimit > tLLR.dimensions()[1]))
        {
            fprintf(stderr,
                    "Invalid number of codewords requested (%i, file has %i)\n",
                    numCBLimit,
                    tLLR.dimensions()[1]);
            exit(1);
        }

        //------------------------------------------------------------------
        // get LDPC parameters from the test vector
        int N, K, numCodeblocks, Z, Kb;

        tv->get_params(BG, Z, K, Kb, numCodeblocks, N);
        numCodeblocks = (numCBLimit > 0) ? numCBLimit : numCodeblocks;
        const int mb = parityNodes;
        const int unpuncturedTransmitNodes = (1 == BG) ? (22 + mb) : (10 + mb);

        printf("*********************************************************************\n");
        printf("LDPC Configuration:\n");
        printf("*********************************************************************\n");
        printf("HDF5 input file     = %s\n", inputFilename.c_str());
        printf("Iterations          = %i\n", numIterations);
        printf("BG (base graph)     = %i\n", BG);
        printf("Kb (info nodes)     = %i\n", Kb);
        printf("Z  (lifting size)   = %i\n", Z);
        printf("mb (parity nodes)   = %i\n", mb);
        printf("M  (mb * Z)         = %i\n", mb * Z);
        printf("N  (LLR block size) = %i\n", N);
        printf("K  (CB size)        = %i\n", K);
        printf("F  (filler bits)    = %i\n", K - (Kb * Z));
        printf("Number of CBs       = %i\n", numCodeblocks);
        printf("Rate (punctured)    = (%i / (%i - 2) = %0.3f\n",
               Kb,
               unpuncturedTransmitNodes,
               static_cast<float>(Kb) / (unpuncturedTransmitNodes - 2));
        printf("Early termination   = %s\n\n", earlyTermination ? "true" : "false");

        printf("*********************************************************************\n");
        printf("Input Tensors:\n");
        printf("*********************************************************************\n");
        printf("inputLLR:      %s\n", tLLR.desc().get_info().to_string(false).c_str());
        //printf("inputCodeWord: %s\n", tCodeWord.desc().get_info().to_string(false).c_str());
        printf("sourceData:    %s\n\n", tSourceData.desc().get_info().to_string(false).c_str());

#if 0
        tensor_pinned_R_8U tCodeWord = typed_tensor_from_dataset<LWPHY_R_8U, pinned_alloc>(fInput.open_dataset("inputCodeWord"));
        for(int i = 0; i < tCodeWord.dimensions()[0]; ++i)
        {
            printf("%i: %i\n", i, tCodeWord({i}));
        }
#endif
        //------------------------------------------------------------------
        // Validate dimensions of inputs
        if(tLLR.dimensions()[0] < ((Kb + mb) * Z))
        {
            fprintf(stderr, "ERROR: Input LLR data dimensions not large enough for chosen configuration.\n");
            usage();
            exit(1);
        }

        //------------------------------------------------------------------
        // Check for a flag that indicates "diagnostic" mode (temporary,
        // for development purposes). In "diagnostic" mode, we will provide
        // a tensor to hold APP output for each codeword at the end of each
        // iteration. (This will only have an effect when the library is
        // compiled with the ENABLE_LDPC_DIAGNOSTIC variable defined.)
        tensor_device tDiag;
        if(2 == flags)
        {
            tDiag = tensor_device(tensor_info(LLR_type,
                                              {K, numCodeblocks, numIterations}),
                                  LWPHY_TENSOR_ALIGN_COALESCE);
        }
        lwphyLDPCDiagnostic_t reserved = {tDiag.desc().handle(), tDiag.addr()};
        //------------------------------------------------------------------
        // Allocate an output buffer based on the input dimensions.
        const int MAX_INFO_SIZE_BITS = 8448;
        tensor_device tDecode(tensor_info(LWPHY_BIT,
                                          {MAX_INFO_SIZE_BITS,
                                           numCodeblocks}),
                              LWPHY_TENSOR_ALIGN_COALESCE);

        printf("*********************************************************************\n");
        printf("Allocated Tensor Layout:\n");
        printf("*********************************************************************\n");
        printf("LLR:    addr: %p, %s, size: %.1f kB\n",
               tLLR.addr(),
               tLLR.desc().get_info().to_string().c_str(),
               tLLR.desc().get_size_in_bytes() / 1024.0);
        printf("Decode: addr: %p, %s, size: %.1f kB\n\n",
               tDecode.addr(),
               tDecode.desc().get_info().to_string().c_str(),
               tDecode.desc().get_size_in_bytes() / 1024.0);
        //------------------------------------------------------------------
        // Create a separate descriptor for LLR input, to handle overriding
        // the number of codewords on the command line. This descriptor must
        // match the layout of the tLLR tensor read from the file (since we
        // will refer to the same memory address with this descriptor), but
        // may have a reduced number of codewords.
        tensor_layout inputLLRLayout(tLLR.desc().get_info().layout());
        inputLLRLayout.dimensions()[1] = numCodeblocks;
        tensor_desc inputLLRDesc(tensor_info(tLLR.type(),
                                             inputLLRLayout));
        //------------------------------------------------------------------
#if 0
        for(int i = 0; i < tSourceData.dimensions()[0]; ++i)
        {
            printf("%i: %i\n", i, tSourceData({i}));
        }
#endif

        typedef buffer<lwphyLDPCResults_t, device_alloc> device_results_buf_t;
        typedef buffer<lwphyLDPCResults_t, pinned_alloc> pinned_results_buf_t;
        
        //--------------------------------------------------------------
        // Create a lwPHY context
        lwphy::context ctx;

        //--------------------------------------------------------------
        // Create an LDPC decoder instance
        lwphy::LDPC_decoder dec(ctx);

        //--------------------------------------------------------------
        // Allocate a workspace buffer, to be used by the LDPC implementation
        size_t szBuf = dec.get_workspace_size(BG,            // BG
                                              Kb,            // Kb
                                              mb,            // mb
                                              Z,             // Z
                                              numCodeblocks, // numCodeblocks
                                              LLR_type,      // type
                                              algoIndex);    // algorithm
        printf("Workspace size: %lu bytes\n", szBuf);
        buffer<char, device_alloc> workspaceBuffer(szBuf);

        device_results_buf_t LDPC_results(tLLR.dimensions()[1]);
        //--------------------------------------------------------------
        // Warmup run
        if(doWarmup)
        {
            dec.decode(tDecode.desc().handle(),             // output descriptor
                       tDecode.addr(),                      // output address
                       inputLLRDesc.handle(),               // LLR descriptor
                       tLLR.addr(),                         // LLR address
                       BG,                                  // base graph
                       Kb,                                  // info nodes
                       Z,                                   // lifting value
                       mb,                                  // parity nodes
                       numIterations,                       // max iterations
                       minSumNorm,                          // normalization
                       earlyTermination ? 1 : 0,            // early termination
                       LDPC_results.addr(),                 // results output
                       algoIndex,                           // algorithm index
                       workspaceBuffer.addr(),              // workspace
                       flags,                               // internal flags
                       0,                                   // stream
                       (flags == 2) ? &reserved : nullptr); // optional diagnostic buffers
            lwdaDeviceSynchronize();
        }

        lwphy::event_timer tmr;

        tmr.record_begin();
        for(unsigned int uRun = 0; uRun < numRuns; ++uRun)
        {
            //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            // Decode
            dec.decode(tDecode.desc().handle(),             // output descriptor
                       tDecode.addr(),                      // output address
                       inputLLRDesc.handle(),               // LLR descriptor
                       tLLR.addr(),                         // LLR address
                       BG,                                  // base graph
                       Kb,                                  // info nodes
                       Z,                                   // lifting value
                       mb,                                  // parity nodes
                       numIterations,                       // max iterations
                       minSumNorm,                          // normalization
                       earlyTermination ? 1 : 0,            // early termination
                       LDPC_results.addr(),                 // results output
                       algoIndex,                           // algorithm index
                       workspaceBuffer.addr(),              // workspace
                       flags,                               // internal flags
                       0,                                   // stream
                       (flags == 2) ? &reserved : nullptr); // optional diagnostic buffers
        }
        tmr.record_end();
        tmr.synchronize();
        float avg_time_sec = tmr.elapsed_time_ms() / ((1000.0f) * numRuns);
        printf("Average (%u runs) elapsed time in usec = %.1f, throughput = %.2f Gbps\n",
               numRuns,
               tmr.elapsed_time_ms() * 1000 / numRuns,
               (Kb * Z * numCodeblocks) / avg_time_sec / 1.0e9);
#if 0
        {
            const int NWORDS = (K + 31) / 32;
            std::vector<uint32_t> debugInfoHost(NWORDS);
            lwdaMemcpy(debugInfoHost.data(), tDecode.addr(), sizeof(uint32_t) * NWORDS, lwdaMemcpyDeviceToHost);
            //for(size_t i = 0; i < NWORDS; ++i)
            //{
            //    printf("%lu: 0x%X\n", i, debugInfoHost[i]);
            //}
            //------------------------------------------------------------------
            // Compare library output to data from input file
            size_t error_count = 0;
            for(int i = 0; i < (Kb * Z); ++i)
            {
                const int      WORD_INDEX   = i / 32;
                const int      BIT_INDEX    = i % 32;
                const uint32_t DECODE_WORD  = debugInfoHost[WORD_INDEX];
                const uint32_t DECODE_VALUE = (DECODE_WORD >> BIT_INDEX) & 1;
                const int      FILE_VALUE   = tSourceData({i});
                if(DECODE_VALUE != FILE_VALUE)
                {
                    printf("i = %i, FILE VALUE = %i, DECODE VALUE = %i (word index = %i, value = 0x%X)\n",
                           i, FILE_VALUE, DECODE_VALUE, WORD_INDEX, DECODE_WORD);
                    ++error_count;
                }
            }
            printf("error_count = %lu, error_rate = (%lu / %i) = %g\n",
                   error_count,
                   error_count,
                   (Kb * Z * numCodeblocks),
                   static_cast<float>(error_count) / (Kb * Z * numCodeblocks));
        }
#endif
#if 0
        // Note: kernels are lwrrently not populating output
        //------------------------------------------------------------------
        // Copy results to host and create a histogram of iterations
        std::vector<int> iterationHistogram(numIterations + 1);
        pinned_results_buf_t LDPC_results_host(LDPC_results);
        size_t iterationSum = 0;
        for(size_t i = 0; i < LDPC_results_host.size(); ++i)
        {
            const lwphyLDPCResults_t& res = LDPC_results_host[i];
            ++iterationHistogram[res.numIterations];
            iterationSum += res.numIterations;
            //printf("CODEWORD %lu: numIterations = %i, checkErrorCount = %i\n",
            //       i,
            //       LDPC_results_host[i].numIterations,
            //       LDPC_results_host[i].checkErrorCount);
        }
        printf("Number of iterations: (mean = %.1f)\n", iterationSum / static_cast<float>(numCodeblocks));
        for(size_t i = 0; i < iterationHistogram.size(); ++i)
        {
            if(iterationHistogram[i] > 0)
            {
                printf("%2lu: %5i\n", i, iterationHistogram[i]);
            }
        }
#endif
        if(compareDecodeOutput)
        {
            uint32_t bit_error_count = 0;
            uint32_t block_error_count = 0;

            tv->verify_decode(tDecode, bit_error_count, block_error_count, numCBLimit);
            printf("bit error count = %u, bit error rate (BER) = (%u / %i) = %.5e, block error rate (BLER) = (%u / %i) = %.5e\n",
                   bit_error_count,
                   bit_error_count,
                   (Kb * Z * numCodeblocks), // Don't count filler bits
                   static_cast<float>(bit_error_count) / (Kb * Z * numCodeblocks),
                   block_error_count,
                   numCodeblocks,
                   static_cast<float>(block_error_count) / (numCodeblocks));
        }
        // Optionaly write an output file with diagnostic per-iteration data
        if(2 == flags)
        {
            printf("Writing LDPC diagnostic output: LDPC_diag.h5\n");
            hdf5hpp::hdf5_file fDiag = hdf5hpp::hdf5_file::create("LDPC_diag.h5");
            write_HDF5_dataset(fDiag, tDiag, "LLR");
        }
    }

    catch(std::exception& e)
    {
        fprintf(stderr, "EXCEPTION: %s\n", e.what());
        returlwalue = 1;
    }
    catch(...)
    {
        fprintf(stderr, "UNKNOWN EXCEPTION\n");
        returlwalue = 2;
    }
    return returlwalue;
}
