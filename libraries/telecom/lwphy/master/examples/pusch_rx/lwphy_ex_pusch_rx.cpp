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
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <functional>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <unistd.h>  /* For SYS_xxx definitions */
#include <syscall.h> /* For SYS_xxx definitions */
#include <sched.h>

#include "lwda_profiler_api.h"
#include "lwphy.h"
#include "lwphy.hpp"
#include "lwphy_hdf5.hpp"
#include "util.hpp"

#include "pusch_rx.hpp"
#include "hdf5hpp.hpp"

#include <chrono>

#define OUTPUT_TB_FNAME ("outputBits")

using Clock     = std::chrono::high_resolution_clock;
using TimePoint = std::chrono::time_point<Clock>;
template <typename T, typename unit>
using duration = std::chrono::duration<T, unit>;
template <typename T>
using ms = std::chrono::milliseconds;
template <typename T>
using us = std::chrono::microseconds;

////////////////////////////////////////////////////////////////////////
// usage()
void usage()
{
    printf("lwphy_ex_pusch_rx_multi_pipe [options]\n");
    printf("  Options:\n");
    printf("    -h                     Display usage information\n");
    printf("    -d                     Disable descrambling\n");
    printf("    -i  input_filename     Input HDF5 filename\n");
    printf("    -c  CPU Id             CPU Id used to run the first pipeline, cpuIdPipeline[i] = cpuIdFirstPipeline + i\n");
    printf("    -g  GPU Id             GPU Id used to run all the pipelines\n");
    printf("    -o  outfile            Write pipeline tensors to an HDF5 output file.\n");
    printf("                           (Not recommended for use during timing runs.)\n");
    printf("    -r  # of iterations    Number of run iterations to run\n");
    printf("    -v                     Enable lwperf profiling with 1 iteration\n");
    printf("    --H                    Use half precision (FP16) for back-end\n");
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
        int         iArg = 1;
        std::string inputFilename;
        std::string outputFilename;
        bool        b16   = false;
        int32_t     nGPUs = 0;
        LWDA_CHECK(lwdaGetDeviceCount(&nGPUs));
        int32_t               nMaxConlwrentThrds = std::thread::hardware_conlwrrency();
        int32_t               cpuIdFirstInst     = 0;
        int32_t               gpuId              = 0;
        uint32_t              nIterations        = 1000;
        bool                  enable_lwprof      = false;
        int                   descramblingOn     = 1;
        PuschRx::ConfigParams cfgPrms;
        cfgPrms.bePrms.ldpcPrms.useHalf = false;
        uint32_t nInst                  = 1;

        uint32_t totalBits; // encoded input size for throughput callwlation
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
                case 'p':
                    b16 = true;
                    ++iArg;
                    break;
                case 'r':
                    if((++iArg >= argc) || (1 != sscanf(argv[iArg], "%i", &nIterations)) || ((nIterations <= 0)))
                    {
                        fprintf(stderr, "ERROR: Invalid number of run iterations\n");
                        exit(1);
                    }
                    ++iArg;
                    break;
                case 'v':
                    enable_lwprof = true;
                    nIterations   = 1;
                    ++iArg;
                    break;
                case 'g':
                    if((++iArg >= argc) ||
                       (1 != sscanf(argv[iArg], "%i", &gpuId)) ||
                       ((gpuId < 0) || (gpuId >= nGPUs)))
                    {
                        fprintf(stderr, "ERROR: Invalid GPU Id (should be within [0,%d])\n", nGPUs - 1);
                        exit(1);
                    }
                    ++iArg;
                    break;
                case 'c':
                    if((++iArg >= argc) ||
                       (1 != sscanf(argv[iArg], "%i", &cpuIdFirstInst)) ||
                       ((cpuIdFirstInst < 0) || (cpuIdFirstInst > nMaxConlwrentThrds - nInst)))
                    {
                        fprintf(stderr, "ERROR: Invalid CPU Id (should be within [0,%d], nMaxConlwrrentThrds %d)\n", nMaxConlwrentThrds - nInst, nMaxConlwrentThrds);
                        exit(1);
                    }
                    ++iArg;
                    break;
                case 'd':
                    descramblingOn = 0;
                    ++iArg;
                    break;
                case 'o':
                    if(++iArg >= argc)
                    {
                        fprintf(stderr, "ERROR: No output file name given.\n");
                    }
                    outputFilename.assign(argv[iArg++]);
                    break;
                case '-':
                    switch(argv[iArg][2])
                    {
                    case 'H':
                        ++iArg;
                        cfgPrms.bePrms.ldpcPrms.useHalf = true;
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
            else // if('-' == argv[iArg][0])
            {
                fprintf(stderr, "ERROR: Invalid command line argument: %s\n", argv[iArg]);
                exit(1);
            }
        } // while (iArg < argc)
        if(inputFilename.empty())
        {
            usage();
            exit(1);
        }
        std::unique_ptr<hdf5hpp::hdf5_file> debugFile;
        if(!outputFilename.empty())
        {
            debugFile.reset(new hdf5hpp::hdf5_file(hdf5hpp::hdf5_file::create(outputFilename.c_str())));
        }

        lwdaSetDevice(gpuId);

        printf("Run config: GPU Id %d\n", gpuId);

        //------------------------------------------------------------------
        // Open the input file and required datasets
        hdf5hpp::hdf5_file fInput(hdf5hpp::hdf5_file::open(inputFilename.c_str()));
        //-------------------------------------------------------------
        // Check for configuration information in the input file. Newer
        // input files will have configuration values in the file, so
        // that they don't need to be specified on the command line.

        std::vector<tb_pars> lwrrentTbsPrmsArray;
        gnb_pars             BBUPrms;
        uint32_t             slotNumber = 0;

        try
        {
            lwphy::lwphyHDF5_struct gnbConfig = lwphy::get_HDF5_struct(fInput, "gnb_pars");
            slotNumber                        = gnbConfig.get_value_as<uint32_t>("slotNumber");
            BBUPrms.fc                        = gnbConfig.get_value_as<uint32_t>("fc");
            BBUPrms.mu                        = gnbConfig.get_value_as<uint32_t>("mu");
            BBUPrms.nRx                       = gnbConfig.get_value_as<uint32_t>("nRx");
            BBUPrms.nPrb                      = gnbConfig.get_value_as<uint32_t>("nPrb");
            BBUPrms.cellId                    = gnbConfig.get_value_as<uint32_t>("cellId");
            BBUPrms.slotNumber                = gnbConfig.get_value_as<uint32_t>("slotNumber");
            BBUPrms.Nf                        = gnbConfig.get_value_as<uint32_t>("Nf");
            BBUPrms.Nt                        = gnbConfig.get_value_as<uint32_t>("Nt");
            BBUPrms.df                        = gnbConfig.get_value_as<uint32_t>("df");
            BBUPrms.dt                        = gnbConfig.get_value_as<uint32_t>("dt");
            BBUPrms.numBsAnt                  = gnbConfig.get_value_as<uint32_t>("numBsAnt");
            BBUPrms.numBbuLayers              = gnbConfig.get_value_as<uint32_t>("numBbuLayers");
            BBUPrms.numTb                     = gnbConfig.get_value_as<uint32_t>("numTb");
            BBUPrms.ldpcnIterations           = gnbConfig.get_value_as<uint32_t>("ldpcnIterations");
            BBUPrms.ldpcEarlyTermination      = gnbConfig.get_value_as<uint32_t>("ldpcEarlyTermination");
            BBUPrms.ldpcAlgoIndex             = gnbConfig.get_value_as<uint32_t>("ldpcAlgoIndex");
            BBUPrms.ldpcFlags                 = gnbConfig.get_value_as<uint32_t>("ldpcFlags");
            BBUPrms.ldplwseHalf               = gnbConfig.get_value_as<uint32_t>("ldplwseHalf");

            lwrrentTbsPrmsArray.resize(BBUPrms.numTb);

            // parse array of tb_pars structs

            hdf5hpp::hdf5_dataset tbpDset = fInput.open_dataset("tb_pars");

            for(int i = 0; i < BBUPrms.numTb; i++)
            {
                lwphy::lwphyHDF5_struct tbConfig        = lwphy::get_HDF5_struct_index(tbpDset, i);
                lwrrentTbsPrmsArray[i].numLayers        = tbConfig.get_value_as<uint32_t>("numLayers");
                lwrrentTbsPrmsArray[i].layerMap         = tbConfig.get_value_as<uint32_t>("layerMap");
                lwrrentTbsPrmsArray[i].startPrb         = tbConfig.get_value_as<uint32_t>("startPrb");
                lwrrentTbsPrmsArray[i].numPrb           = tbConfig.get_value_as<uint32_t>("numPRb");
                lwrrentTbsPrmsArray[i].startSym         = tbConfig.get_value_as<uint32_t>("startSym");
                lwrrentTbsPrmsArray[i].numSym           = tbConfig.get_value_as<uint32_t>("numSym");
                lwrrentTbsPrmsArray[i].dmrsMaxLength    = tbConfig.get_value_as<uint32_t>("dmrsMaxLength");
                lwrrentTbsPrmsArray[i].dataScramId      = tbConfig.get_value_as<uint32_t>("dataScramId");
                lwrrentTbsPrmsArray[i].mcsTableIndex    = tbConfig.get_value_as<uint32_t>("mcsTableIndex");
                lwrrentTbsPrmsArray[i].mcsIndex         = tbConfig.get_value_as<uint32_t>("mcsIndex");
                lwrrentTbsPrmsArray[i].rv               = tbConfig.get_value_as<uint32_t>("rv");
                lwrrentTbsPrmsArray[i].dmrsType         = tbConfig.get_value_as<uint32_t>("dmrsType");
                lwrrentTbsPrmsArray[i].dmrsAddlPosition = tbConfig.get_value_as<uint32_t>("dmrsAddlPosition");
                lwrrentTbsPrmsArray[i].dmrsMaxLength    = tbConfig.get_value_as<uint32_t>("dmrsMaxLength");
                lwrrentTbsPrmsArray[i].dmrsScramId      = tbConfig.get_value_as<uint32_t>("dmrsScramId");
                lwrrentTbsPrmsArray[i].dmrsEnergy       = tbConfig.get_value_as<uint32_t>("dmrsEnergy");
                lwrrentTbsPrmsArray[i].nRnti            = tbConfig.get_value_as<uint32_t>("nRnti");
                lwrrentTbsPrmsArray[i].dmrsCfg          = tbConfig.get_value_as<uint32_t>("dmrsCfg");
            }
            // END NEW PARAM STRUCT PARSING
        }
        catch(const std::exception& exc)
        {
            printf("%s\n", exc.what());
            throw exc;
            // Continue using command line arguments if the input file does not
            // have a config struct.
        }

        lwphy::enable_hdf5_error_print(); // Re-enable HDF5 stderr printing
        //-------------------------------------------------------------
        PuschRxDataset d(fInput, slotNumber);

        const int                   iBufSize = 40;
        std::vector<PuschRxDataset> dv;

        for(int i = 0; i < iBufSize; i++)
        {
            dv.push_back(PuschRxDataset(fInput, slotNumber));
        }

        PuschRx      puschRx;
        lwdaStream_t lwStrm;
        lwdaStreamCreate(&lwStrm);

        totalBits = puschRx.expandParameters(d.tWFreq, lwrrentTbsPrmsArray, BBUPrms, lwStrm);

        PuschRx::ConfigParams const& prms = puschRx.getCfgPrms();
        puschRx.printInfo();

        //puschRx.loadWFreq(d);

        lwdaDeviceSynchronize();

        lwdaEvent_t eStart, eStop;
        LWDA_CHECK(lwdaEventCreateWithFlags(&eStart, lwdaEventBlockingSync));
        LWDA_CHECK(lwdaEventCreateWithFlags(&eStop, lwdaEventBlockingSync));

        if(enable_lwprof)
        {
            lwdaProfilerStart();
        }

        LWDA_CHECK(lwdaEventRecord(eStart, lwStrm));
        TimePoint startTime = Clock::now();

        {
            for(uint32_t i = 0; i < nIterations; ++i)
            {
                puschRx.Run(lwStrm,
                            dv[i % iBufSize].slotNumber,
                            dv[i % iBufSize].tDataRx,
                            dv[i % iBufSize].tShiftSeq,
                            dv[i % iBufSize].tUnShiftSeq,
                            dv[i % iBufSize].tDataSymLoc,
                            dv[i % iBufSize].tQamInfo,
                            dv[i % iBufSize].tNoisePwr,
                            descramblingOn,
                            (0 == i) ? debugFile.get() : nullptr);
            }
        }

        LWDA_CHECK(lwdaEventRecord(eStop, lwStrm));
        LWDA_CHECK(lwdaStreamSynchronize(lwStrm));
        LWDA_CHECK(lwdaEventSynchronize(eStop));

        TimePoint stopTime = Clock::now();

        if(enable_lwprof)
        {
            lwdaProfilerStop();
        }

        float elapsedMs = 0.0f;
        lwdaEventElapsedTime(&elapsedMs, eStart, eStop);

        duration<float, std::milli> diff = stopTime - startTime;

        LWDA_CHECK(lwdaEventDestroy(eStart));
        LWDA_CHECK(lwdaEventDestroy(eStop));

        puschRx.copyOutputToCPU(lwStrm);
        lwdaStreamDestroy(lwStrm);
        const uint32_t* crcs            = puschRx.getCRCs();
        const uint32_t* tbCRCs          = puschRx.getTbCRCs();
        const uint8_t*  transportBlocks = puschRx.getTransportBlocks();
        uint32_t        nCRCErrors      = 0;

        // write output bits to file
        std::ofstream of(OUTPUT_TB_FNAME, std::ofstream::binary);
      of.write(reinterpret_cast<const char*>(transportBlocks), totalBits / 8 + ((totalBits % 8) != 0));
        of.close();

        for(int i = 0; i < prms.bePrms.CSum; i++)
        {
            if(crcs[i] != 0)
            {
                nCRCErrors++;
                printf("ERROR: CRC of code block [%d] failed!\n", i);
            }
        }
        for(int i = 0; i < prms.bePrms.nTb; i++)
        {
            if(tbCRCs[i] != 0)
            {
                printf("ERROR: CRC of transport block [%d] failed!\n", i);
            }
        }
        printf("Metric - Throughput            : %4.4f Gbps (encoded input bits %d) \n",
               (static_cast<float>(totalBits) / ((elapsedMs * 1e-3) / nIterations)) / 1e9,
               totalBits);

        printf("Metric - Block Error Rate      : %4.4f (Error CBs %d, Total CBs %d)\n",
               static_cast<float>(nCRCErrors) / static_cast<float>(puschRx.getCfgPrms().bePrms.CSum),
               nCRCErrors,
               puschRx.getCfgPrms().bePrms.CSum);

        printf("Metric - Average exelwtion time: %4.4f usec (over %d runs, using LWCA event)\n",
               elapsedMs * 1000 / nIterations,
               nIterations);
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
