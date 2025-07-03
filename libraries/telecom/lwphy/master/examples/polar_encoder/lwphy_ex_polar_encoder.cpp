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
#include "lwda_profiler_api.h"
#include "lwphy.h"
#include "lwphy.hpp"
#include "util.hpp"
#include "lwphy_hdf5.hpp"
#include "hdf5hpp.hpp"
#include <chrono>
using Clock     = std::chrono::steady_clock;
using TimePoint = std::chrono::time_point<Clock>;
template <typename T, typename unit>
using duration = std::chrono::duration<T, unit>;
template <typename T>
using ms = std::chrono::milliseconds;
template <typename T>
using us = std::chrono::microseconds;

using namespace lwphy;

////////////////////////////////////////////////////////////////////////
// usage()
void usage()
{
    printf("lwphy_ex_polar_encoder [options]\n");
    printf("  Options:\n");
    printf("    -h                  Display usage information\n");
    printf("    -i  input_filename  Input HDF5 filename, which must contain the following datasets:\n");
    printf("    -o  output_filename Write pipeline tensors to an HDF5 output file.\n");
    printf("                        (Not recommended for use during timing runs.)\n");
    printf("    -r  # of iterations Number of iterations to run\n");
    // printf("    --I                 Number of info bits\n");
    // printf("    --T                 Number of transmit bits\n");
    printf("    --V                 Verbose logging\n");
}

int main(int argc, char* argv[])
{
    int returlwalue = 0;
    try
    {
        //------------------------------------------------------------------
        // Parse command line arguments
        int         iArg = 1;
        std::string inputFilename;
        std::string outputFilename;
        uint32_t    nInfoBits = 0;
        uint32_t    nTxBits   = 0;
        bool        enLwprof  = false;
        bool        verbose   = false;
        uint32_t    nIter     = 1000;

        lwdaStream_t lwStream;
        lwdaStreamCreateWithFlags(&lwStream, lwdaStreamNonBlocking);

        while(iArg < argc)
        {
            if('-' == argv[iArg][0])
            {
                switch(argv[iArg][1])
                {
                case 'i':
                    if(++iArg >= argc)
                    {
                        fprintf(stderr, "ERROR: No filename provided.\n");
                    }
                    inputFilename.assign(argv[iArg++]);
                    break;
                case 'h':
                    usage();
                    exit(0);
                    break;
                case 'o':
                    if(++iArg >= argc)
                    {
                        fprintf(stderr, "ERROR: No output file name given.\n");
                    }
                    outputFilename.assign(argv[iArg++]);
                    break;
                case 'r':
                    if((++iArg >= argc) || (1 != sscanf(argv[iArg], "%i", &nIter)) || ((nIter <= 0)))
                    {
                        fprintf(stderr, "ERROR: Invalid number of run iterations\n");
                        exit(1);
                    }
                    ++iArg;
                    break;
                case '-':
                    switch(argv[iArg][2])
                    {
                    case 'I':
                        if((++iArg >= argc) || (1 != sscanf(argv[iArg], "%i", &nInfoBits)) || ((nInfoBits < 1) || (nInfoBits > LWPHY_POLAR_ENC_MAX_INFO_BITS)))
                        {
                            fprintf(stderr, "ERROR: # of information bits invalid %d\n", nInfoBits);
                            exit(1);
                        }
                        ++iArg;
                        break;
                    case 'T':
                        if((++iArg >= argc) || (1 != sscanf(argv[iArg], "%i", &nTxBits)) || ((nTxBits < 1) || (nTxBits > LWPHY_POLAR_ENC_MAX_TX_BITS)))
                        {
                            fprintf(stderr, "ERROR: # of transmit bits invalid %d\n", nTxBits);
                            exit(1);
                        }
                        ++iArg;
                        break;
                    case 'V':
                        verbose = true;
                        ++iArg;
                        break;
                    case 'P':
                        enLwprof = true;
                        nIter    = 1;
                        ++iArg;
                        break;
                    default:
                        fprintf(stderr, "ERROR: Unknown option: %s\n", argv[iArg]);
                        usage();
                        exit(1);
                        break;
                    }
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
        if(inputFilename.empty())
        {
            usage();
            exit(1);
        }

        lwdaEvent_t eStart, eStop;
        LWDA_CHECK(lwdaEventCreateWithFlags(&eStart, lwdaEventBlockingSync));
        LWDA_CHECK(lwdaEventCreateWithFlags(&eStop, lwdaEventBlockingSync));

        //------------------------------------------------------------------
        // Open the input file
        hdf5hpp::hdf5_file fInput = hdf5hpp::hdf5_file::open(inputFilename.c_str());
        using tensor_pinned_R_8U  = typed_tensor<LWPHY_R_8U, pinned_alloc>;

        lwphy::tensor_device tGpuInfoBits  = lwphy::tensor_from_dataset(fInput.open_dataset("InfoBits"), LWPHY_R_8U, LWPHY_TENSOR_ALIGN_TIGHT, lwStream);
        tensor_pinned_R_8U   tCpuCodedBits = typed_tensor_from_dataset<LWPHY_R_8U, pinned_alloc>(fInput.open_dataset("CodedBits"), LWPHY_TENSOR_ALIGN_TIGHT, lwStream);
        tensor_pinned_R_8U   tCpuTxBits    = typed_tensor_from_dataset<LWPHY_R_8U, pinned_alloc>(fInput.open_dataset("TxBits"), LWPHY_TENSOR_ALIGN_TIGHT, lwStream);

        uint32_t nExpectedCodedBits = 0;

        lwphy::disable_hdf5_error_print(); // Disable HDF5 stderr printing
        try
        {
            lwphy::lwphyHDF5_struct encPrms = lwphy::get_HDF5_struct(fInput, "encPrms");
            nInfoBits                       = encPrms.get_value_as<uint32_t>("nInfoBits");
            nExpectedCodedBits              = encPrms.get_value_as<uint32_t>("nCodedBits");
            nTxBits                         = encPrms.get_value_as<uint32_t>("nTxBits");
        }
        catch(const std::exception& exc)
        {
            printf("%s\n", exc.what());
            throw exc;
            // Continue using command line arguments if the input file does not
            // have an encPrms struct.

            nExpectedCodedBits = ((nInfoBits + 31) / 32) * 32;
        }
        lwphy::enable_hdf5_error_print(); // Re-enable HDF5 stderr printing

        // Allocate output tensors
        // For coded bits provide the worst case storage
        lwphy::tensor_device tGpuCodedBits(lwphy::tensor_info(LWPHY_R_8U,
                                                              {static_cast<int>((LWPHY_POLAR_ENC_MAX_CODED_BITS + 7) / 8)}),
                                           LWPHY_TENSOR_ALIGN_TIGHT);
        lwphy::tensor_device tGpuTxBits(lwphy::tensor_info(LWPHY_R_8U,
                                                           {static_cast<int>((((LWPHY_POLAR_ENC_MAX_TX_BITS + 31) / 32) * 32) / 8)}), // roundup to nearest 32b boundary (multiple of words)
                                        LWPHY_TENSOR_ALIGN_TIGHT);

        lwdaStreamSynchronize(lwStream);
        lwdaDeviceSynchronize(); // Needed because typed_tensor does not support non-default streams

        //------------------------------------------------------------------
        // Run the test
        if(enLwprof) lwdaProfilerStart();

        TimePoint startTime = Clock::now();
        LWDA_CHECK(lwdaEventRecord(eStart, lwStream));
        uint32_t nCodedBits = 0;

        for(uint32_t i = 0; i < nIter; ++i)
        {
            lwphyStatus_t polarEncStat = lwphyPolarEncRateMatch(nInfoBits,
                                                                nTxBits,
                                                                static_cast<uint8_t const*>(tGpuInfoBits.addr()),
                                                                &nCodedBits,
                                                                static_cast<uint8_t*>(tGpuCodedBits.addr()),
                                                                static_cast<uint8_t*>(tGpuTxBits.addr()),
                                                                lwStream);
            if(LWPHY_STATUS_SUCCESS != polarEncStat) throw lwphy::lwphy_exception(polarEncStat);
        }

        LWDA_CHECK(lwdaEventRecord(eStop, lwStream));
        LWDA_CHECK(lwdaEventSynchronize(eStop));

        lwdaStreamSynchronize(lwStream);

        TimePoint stopTime = Clock::now();

        if(enLwprof) lwdaProfilerStop();

        //------------------------------------------------------------------
        // Display exelwtion times
        float elapsedMs = 0.0f;
        lwdaEventElapsedTime(&elapsedMs, eStart, eStop);

        printf("Exelwtion time: Polar encoding + Rate matching \n");
        printf("---------------------------------------------------------------\n");
        printf("Average (over %d runs) elapsed time in usec (LWCA event) = %.0f\n",
               nIter,
               elapsedMs * 1000 / nIter);

        duration<float, std::milli> diff = stopTime - startTime;
        printf("Average (over %d runs) elapsed time in usec (wall clock) w/ 1s delay kernel = %.0f\n",
               nIter,
               diff.count() * 1000 / nIter);

        //------------------------------------------------------------------
        // Verify results
        // Coded bits are always a multiple of 32
        tensor_pinned_R_8U tCpuCpyCodedBits(tGpuCodedBits.layout(), LWPHY_TENSOR_ALIGN_TIGHT);
        // typed_tensor<LWPHY_BIT, pinned_alloc> tCpuCpyCodedBits(tGpuCodedBits.layout(), LWPHY_TENSOR_ALIGN_TIGHT);
        tCpuCpyCodedBits = tGpuCodedBits;

        tensor_pinned_R_8U tCpuCpyTxBits(tGpuTxBits.layout(), LWPHY_TENSOR_ALIGN_TIGHT);
        // typed_tensor<LWPHY_BIT, pinned_alloc> tCpuCpyInfoBits(tGpuInfoBits.layout(), LWPHY_TENSOR_ALIGN_TIGHT);
        tCpuCpyTxBits = tGpuTxBits;

        // Wait for copy to complete
        lwdaStreamSynchronize(lwStream);
        lwdaDeviceSynchronize(); // Needed becase typed_tensor does not support non-default streams

        // Compare expected vs observed
        printf("nInfoBits: %d nExpectedCodedBits: %d nComputedCodedBits: %d nTxBits: %d\n", nInfoBits, nExpectedCodedBits, nCodedBits, nTxBits);
        // Coded bits
        printf("---------------------------------------------------------------\n");
        printf("Comparing coded bits\n");
        uint32_t nCodedByteErrs = 0;
        uint32_t nCodedBytes    = nExpectedCodedBits / 8; // nExpectedCodedBits is a multiple of 32
        for(int n = 0; n < nCodedBytes; ++n)
        {
            uint32_t expectedCodedByte = tCpuCodedBits({n});
            uint32_t observedCodedByte = tCpuCpyCodedBits({n});
            if(expectedCodedByte != observedCodedByte)
            {
                printf("Error: Byte[%03d] Expected 0x%02x Observed 0x%02x\n", n, tCpuCodedBits({n}), tCpuCpyCodedBits({n}));
                nCodedByteErrs++;
            }
        }

        if(0 == nCodedByteErrs)
        {
            printf("No errors detected in coded bits\n");
        }

        // Transmit bits
        printf("---------------------------------------------------------------\n");
        printf("Comparing transmit bits\n");
        uint32_t nTxByteErrs = 0;
        uint32_t nTxBytes    = (nTxBits + 7) / 8; // nTxBits is not a multiple of 8, needs rounding
        for(int n = 0; n < nTxBytes; ++n)
        {
            uint32_t expectedTxByte = tCpuTxBits({n});
            uint32_t observedTxByte = tCpuCpyTxBits({n});
            if(expectedTxByte != observedTxByte)
            {
                printf("Error: Byte[%03d] Expected 0x%02x Observed 0x%02x\n", n, tCpuTxBits({n}), tCpuCpyTxBits({n}));
                nTxByteErrs++;
            }
        }

        if(0 == nTxByteErrs)
        {
            printf("No errors detected in transmit bits\n");
        }

        if(verbose)
        {
            // Coded bits
            printf("---------------------------------------------------------------\n");
            printf("Dumping coded bits (formatted as %d bytes)\n", nCodedBytes);
            for(int n = 0; n < nCodedBytes; ++n)
            {
                uint32_t expectedCodedByte = tCpuCodedBits({n});
                uint32_t observedCodedByte = tCpuCpyCodedBits({n});
                printf("Byte[%03d] Expected 0x%02x Observed 0x%02x\n", n, tCpuCodedBits({n}), tCpuCpyCodedBits({n}));
            }

            // Transmit bits
            printf("---------------------------------------------------------------\n");
            printf("Dumping transmit bits (formatted as %d bytes)\n", nTxBytes);
            for(int n = 0; n < nTxBytes; ++n)
            {
                uint32_t expectedTxByte = tCpuTxBits({n});
                uint32_t observedTxByte = tCpuCpyTxBits({n});
                printf("Byte[%03d] Expected 0x%02x Observed 0x%02x\n", n, tCpuTxBits({n}), tCpuCpyTxBits({n}));
            }
        }

        //------------------------------------------------------------------
        // Cleanup
        LWDA_CHECK(lwdaEventDestroy(eStart));
        LWDA_CHECK(lwdaEventDestroy(eStop));

        lwdaStreamSynchronize(lwStream);

        lwdaDeviceSynchronize();
        lwdaStreamDestroy(lwStream);
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

