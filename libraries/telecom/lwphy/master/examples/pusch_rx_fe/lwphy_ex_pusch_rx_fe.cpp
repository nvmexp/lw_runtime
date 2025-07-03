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
#include "lwda_profiler_api.h"
#include "lwphy.h"
#include "lwphy.hpp"
#include "lwphy_hdf5.hpp"
#include "hdf5hpp.hpp"
#include "util.hpp"
#include <chrono>
using Clock     = std::chrono::steady_clock;
using TimePoint = std::chrono::time_point<Clock>;
template <typename T, typename unit>
using duration = std::chrono::duration<T, unit>;
template <typename T>
using ms = std::chrono::milliseconds;
template <typename T>
using us = std::chrono::microseconds;

#define SYMB_EQ
#define EQ_COEF_APPLY_VER (2)

////////////////////////////////////////////////////////////////////////
// usage()
void usage()
{
    printf("lwphy_ex_pusch_rx_fe [options]\n");
    printf("  Options:\n");
    printf("    -h                  Display usage information\n");
    printf("    -i  input_filename  Input HDF5 filename, which must contain the following datasets:\n");
    printf("                           Data_rx      : received data (frequency-time) to be equalized\n");
    printf("                           WFreq        : interpolation filter coefficients used in channel estimation\n");
    printf("                           ShiftSeq     : sequence to be applied to DMRS tones containing descrambling\n");
    printf("                                          code and delay shift for channel centering\n");
    printf("                           UnShiftSeq   : sequence to remove the delay shift from estimated channel\n");
    printf("                           Data_sym_loc : locations of data symbols within the subframe\n");
    printf("                           RxxIlw       : Symbol energy covariance\n");
    printf("                           Noise_pwr    : noise power at frequency-time bins where channel (H) is estimated\n");
    printf("                           Data_eq      : equalized output data (in frequency-time)\n");
    printf("    -o  outfile            Write pipeline tensors to an HDF5 output file.\n");
    printf("                           (Not recommended for use during timing runs.)\n");
    printf("    --M                 DMRS grid bitmask\n");
    printf("    --I                 Number of iterations to run\n");
    printf("    --P                 Enable lwperf profiling with 1 iteration\n");
    printf("    --H                 0(default): No FP16\n");
    printf("                        1         : FP16 format used for received data samples only\n");
    printf("                        2         : FP16 format used for all front end params\n");
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
        bool        b16                = false;
        uint32_t    nLayers            = 8;                       // Number of layers
        uint32_t    activeDMRSGridBmsk = (1UL << 0) | (1UL << 1); // (1UL << 1);// (1UL << 0) | (1UL << 1);
        uint32_t    nIter              = 1000;
        bool        enLwprof           = false;
        uint32_t    fp16Mode           = 0xBAD;

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
                case 'p':
                    b16 = true;
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
                    case 'M':
                        if(++iArg >= argc)
                        {
                            fprintf(stderr, "ERROR: DMRS grid bitmask not specified\n");
                            exit(1);
                        }
                        activeDMRSGridBmsk = std::stoi(argv[iArg++]);
                        break;
                    case 'I':
                        if((++iArg >= argc) || (1 != sscanf(argv[iArg], "%i", &nIter)) || ((nIter <= 0)))
                        {
                            fprintf(stderr, "ERROR: Invalid number of run iterations %d\n", nIter);
                            exit(1);
                        }
                        ++iArg;
                        break;
                    case 'P':
                        enLwprof = true;
                        nIter    = 1;
                        ++iArg;
                        break;
                    case 'H':
                        if((++iArg >= argc) || (1 != sscanf(argv[iArg], "%i", &fp16Mode)) || (3 <= fp16Mode))
                        {
                            fprintf(stderr, "ERROR: Invalid FP16 mode 0x%x\n", fp16Mode);
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
        //------------------------------------------------------------------
        // Open the input file
        hdf5hpp::hdf5_file fInput = hdf5hpp::hdf5_file::open(inputFilename.c_str());

        //------------------------------------------------------------------
        // Copy file data to tensors in device memory

        // Check FP16 mode of operation
        bool isDataRxFp16   = true; 
        bool isChannelFp16  = false;
        switch(fp16Mode)
        {
            case 0 : isDataRxFp16 = false; isChannelFp16 = false; break;
            case 1 : isDataRxFp16 = true ; isChannelFp16 = false; break;
            case 2 : isDataRxFp16 = true ; isChannelFp16 = true ; break;
            default: isDataRxFp16 = false; isChannelFp16 = false; break;
        }
        lwphyDataType_t typeDataRx = isDataRxFp16 ? LWPHY_R_16F : LWPHY_R_32F;
        lwphyDataType_t cplxTypeDataRx = isDataRxFp16 ? LWPHY_C_16F : LWPHY_C_32F;

        lwphyDataType_t typeFeChannel = isChannelFp16 ? LWPHY_R_16F : LWPHY_R_32F;
        lwphyDataType_t cplxTypeFeChannel = isChannelFp16 ? LWPHY_C_16F : LWPHY_C_32F;

        lwphy::tensor_device tDataRx        = lwphy::tensor_from_dataset(fInput.open_dataset("DataRx")       , cplxTypeDataRx   , LWPHY_TENSOR_ALIGN_TIGHT, lwStream);
        lwphy::tensor_device tWFreq         = lwphy::tensor_from_dataset(fInput.open_dataset("WFreq")        , typeFeChannel    , LWPHY_TENSOR_ALIGN_TIGHT, lwStream);
        lwphy::tensor_device tShiftSeq      = lwphy::tensor_from_dataset(fInput.open_dataset("ShiftSeq"), cplxTypeFeChannel, LWPHY_TENSOR_ALIGN_TIGHT, lwStream);
        lwphy::tensor_device tUnShiftSeq    = lwphy::tensor_from_dataset(fInput.open_dataset("UnShiftSeq")   , cplxTypeFeChannel, LWPHY_TENSOR_ALIGN_TIGHT, lwStream);
        lwphy::tensor_device tData_sym_loc  = lwphy::tensor_from_dataset(fInput.open_dataset("Data_sym_loc") ,                    LWPHY_TENSOR_ALIGN_TIGHT, lwStream);
        lwphy::tensor_device tRxxIlw        = lwphy::tensor_from_dataset(fInput.open_dataset("RxxIlw")       , typeFeChannel    , LWPHY_TENSOR_ALIGN_TIGHT, lwStream);
#ifdef SYMB_EQ
        lwphy::tensor_device tNoisePwr      = lwphy::tensor_from_dataset(fInput.open_dataset("Noise_pwr")    , cplxTypeFeChannel, LWPHY_TENSOR_ALIGN_TIGHT, lwStream);
        lwphy::tensor_device tQamInfo       = lwphy::tensor_from_dataset(fInput.open_dataset("QamInfo")      ,                    LWPHY_TENSOR_ALIGN_TIGHT, lwStream);
#else
        lwphy::tensor_device tNoisePwr      = lwphy::tensor_from_dataset(fInput.open_dataset("Noise_pwr")    , typeFeChannel    , LWPHY_TENSOR_ALIGN_TIGHT, lwStream);
#endif //SYMB_EQ

        // Ensure colwersion completes
        // lwdaDeviceSynchronize();
        lwdaStreamSynchronize(lwStream);

        printf("Input tensors:\n");
        printf("---------------------------------------------------------------\n");
        printf("DataRx         : %s\n", tDataRx.desc().get_info().to_string(false).c_str());
        printf("WFreq          : %s\n", tWFreq.desc().get_info().to_string(false).c_str());
        printf("ShiftSeq       : %s\n", tShiftSeq.desc().get_info().to_string(false).c_str());
        printf("UnShiftSeq     : %s\n", tUnShiftSeq.desc().get_info().to_string(false).c_str());
        printf("Data_sym_loc   : %s\n", tData_sym_loc.desc().get_info().to_string(false).c_str());
        printf("RxxIlw         : %s\n", tRxxIlw.desc().get_info().to_string(false).c_str());
        printf("NoisePwr       : %s\n\n", tNoisePwr.desc().get_info().to_string(false).c_str());

#ifdef SYMB_EQ
        printf("QamInfo        : %s\n", tQamInfo.desc().get_info().to_string(false).c_str());
#endif //SYMB_EQ

        //------------------------------------------------------------------
        // Parameters assumed and derived from the input data
        constexpr uint32_t N_TONES_PER_PRB              = 12;
        constexpr uint32_t N_DMRS_GRIDS_PER_PRB         = 2;
        constexpr uint32_t N_DMRS_GRID_TONES_PER_PRB    = N_TONES_PER_PRB / N_DMRS_GRIDS_PER_PRB;
        constexpr uint32_t N_INTERP_DMRS_TONES_PER_GRID = N_TONES_PER_PRB;

        uint32_t cellId           = 0;
        uint32_t slotNumber       = 0;
        uint32_t nBSAnts          = 0; 
        uint32_t Nf               = 0; 
        uint32_t Nprb             = 0; 
        uint32_t nDMRSSyms        = tShiftSeq.dimensions()[1];
        uint32_t nDMRSGridsPerPRB = N_DMRS_GRIDS_PER_PRB;
        uint32_t nTotalDMRSPRB    = tShiftSeq.dimensions()[0] / N_DMRS_GRID_TONES_PER_PRB;
        uint32_t nTotalDataPRB    = nTotalDMRSPRB;
        uint32_t Nh               = 1;
        uint32_t Nd               = tData_sym_loc.dimensions()[0];
        uint32_t nTotalDataTones  = nTotalDMRSPRB * N_TONES_PER_PRB;

        try
        {
           lwphy::lwphyHDF5_struct gnbConfig = lwphy::get_HDF5_struct(fInput, "gnb_pars");
           cellId                            = gnbConfig.get_value_as<uint32_t>("cellId");
           slotNumber                        = gnbConfig.get_value_as<uint32_t>("slotNumber");
           nBSAnts                           = gnbConfig.get_value_as<uint32_t>("numBsAnt");
           nLayers                           = gnbConfig.get_value_as<uint32_t>("numBbuLayers");
           Nf                                = gnbConfig.get_value_as<uint32_t>("Nf");
           Nprb                              = gnbConfig.get_value_as<uint32_t>("nPrb");
        }
        catch(const std::exception& exc)
        {
           printf("%s\n", exc.what());
           throw exc;
           // Continue using command line arguments if the input file does not
           // have a config struct.
        }
        lwphy::enable_hdf5_error_print(); // Re-enable HDF5 stderr printing

        printf("Assumed and Derived parameters:\n");
        printf("---------------------------------------------------------------\n");
        printf("cellId             : %i\n", cellId);
        printf("slotNumber         : %i\n", slotNumber);
        printf("nBSAnts            : %i\n", nBSAnts);
        printf("nLayers            : %i\n", nLayers);
        printf("Nf                 : %i\n", Nf); // # of estimates of H in frequency
        printf("Nprb               : %i\n", Nprb);
        printf("nDMRSSyms          : %i\n", nDMRSSyms);
        printf("nDMRSGridsPerPRB   : %i\n", nDMRSGridsPerPRB);
        printf("activeDMRSGridBmsk : 0x%x\n", activeDMRSGridBmsk);
        printf("nTotalDMRSPRB      : %i\n", nTotalDMRSPRB);
        printf("nTotalDataPRB      : %i\n", nTotalDataPRB);
        printf("Nh                 : %i\n", Nh); // # of estimates of H in time
        printf("Nd                 : %i\n", Nd); // # of data symbols

        //------------------------------------------------------------------
        // Allocate output tensors in device memory
        lwphy::tensor_device tHEst(lwphy::tensor_info(cplxTypeFeChannel,
                                                      {static_cast<int>(nBSAnts),
                                                       static_cast<int>(nLayers),
                                                       static_cast<int>(nTotalDataTones),
                                                       static_cast<int>(Nh)}),
                                   LWPHY_TENSOR_ALIGN_TIGHT);
#if (EQ_COEF_APPLY_VER == 1)
        lwphy::tensor_device tCoef(lwphy::tensor_info(cplxTypeFeChannel,
                                                      {static_cast<int>(nLayers),
                                                       static_cast<int>(nBSAnts),
                                                       static_cast<int>(Nf),
                                                       static_cast<int>(Nh)}),
                                      LWPHY_TENSOR_ALIGN_TIGHT);
        lwphy::tensor_device tReeDiag(lwphy::tensor_info(typeDataRx,
                                                          {static_cast<int>(nLayers),
                                                           static_cast<int>(Nf),
                                                           static_cast<int>(Nh)}),
                                       LWPHY_TENSOR_ALIGN_TIGHT);
#elif (EQ_COEF_APPLY_VER == 2)
        lwphy::tensor_device tCoef(lwphy::tensor_info(cplxTypeFeChannel,
                                                     {static_cast<int>(nBSAnts),
                                                      static_cast<int>(LWPHY_N_TONES_PER_PRB),
                                                      static_cast<int>(nLayers),
                                                      static_cast<int>(Nprb)}),
                                    LWPHY_TENSOR_ALIGN_TIGHT);
        lwphy::tensor_device tReeDiag(lwphy::tensor_info(typeDataRx,
                                                        {static_cast<int>(LWPHY_N_TONES_PER_PRB),
                                                         static_cast<int>(nLayers),
                                                         static_cast<int>(Nprb)}),
                                      LWPHY_TENSOR_ALIGN_TIGHT);
#endif // EQ_COEF_APPLY_VER 
        lwphy::tensor_device tDataEq(lwphy::tensor_info(cplxTypeDataRx,
                                                         {static_cast<int>(nLayers),
                                                          static_cast<int>(Nf),
                                                          static_cast<int>(Nd)}),
                                      LWPHY_TENSOR_ALIGN_TIGHT);
       lwphy::tensor_device tLLR(lwphy::tensor_info(typeDataRx,
                                                     {static_cast<int>(LWPHY_QAM_256),
                                                      static_cast<int>(nLayers),
                                                      static_cast<int>(Nf),
                                                      static_cast<int>(Nd)}),
                                  LWPHY_TENSOR_ALIGN_TIGHT);
        lwphy::tensor_device tDbg(lwphy::tensor_info(cplxTypeFeChannel,
                                                     {static_cast<int>(nLayers),
                                                      static_cast<int>(nLayers),
                                                      static_cast<int>(Nf),
                                                      static_cast<int>(Nd)}),
                                   LWPHY_TENSOR_ALIGN_TIGHT);


        printf("Tensor layout:\n");
        printf("---------------------------------------------------------------\n");
        printf("tDataSymLoc : addr: %p, %s, size: %.1f kB\n",
               tData_sym_loc.addr(),
               tData_sym_loc.desc().get_info().to_string().c_str(),
               tData_sym_loc.desc().get_size_in_bytes() / 1024.0);
#ifdef SYMB_EQ
        printf("tQamInfo    : addr: %p, %s, size: %.1f kB\n",
               tQamInfo.addr(),
               tQamInfo.desc().get_info().to_string().c_str(),
               tQamInfo.desc().get_size_in_bytes() / 1024.0);
#endif // SYMB_EQ
        printf("tDataRx       : addr: %p, %s, size: %.1f kB\n",
               tDataRx.addr(),
               tDataRx.desc().get_info().to_string().c_str(),
               tDataRx.desc().get_size_in_bytes() / 1024.0);
        printf("tWFreq        : addr: %p, %s, size: %.1f kB\n",
               tWFreq.addr(),
               tWFreq.desc().get_info().to_string().c_str(),
               tWFreq.desc().get_size_in_bytes() / 1024.0);
        printf("tShiftSeq: addr: %p, %s, size: %.1f kB\n",
               tShiftSeq.addr(),
               tShiftSeq.desc().get_info().to_string().c_str(),
               tShiftSeq.desc().get_size_in_bytes() / 1024.0);
        printf("tUnShiftSeq   : addr: %p, %s, size: %.1f kB\n",
               tUnShiftSeq.addr(),
               tUnShiftSeq.desc().get_info().to_string().c_str(),
               tUnShiftSeq.desc().get_size_in_bytes() / 1024.0);
        printf("tDataSymLoc   : addr: %p, %s, size: %.1f kB\n",
               tData_sym_loc.addr(),
               tData_sym_loc.desc().get_info().to_string().c_str(),
               tData_sym_loc.desc().get_size_in_bytes() / 1024.0);
        printf("tRxxIlw       : addr: %p, %s, size: %.1f kB\n",
               tRxxIlw.addr(),
               tRxxIlw.desc().get_info().to_string().c_str(),
               tRxxIlw.desc().get_size_in_bytes() / 1024.0);
        printf("tNoisePwr    : addr: %p, %s, size: %.1f kB\n",
               tNoisePwr.addr(),
               tNoisePwr.desc().get_info().to_string().c_str(),
               tNoisePwr.desc().get_size_in_bytes() / 1024.0);
        printf("tHEst         : addr: %p, %s, size: %.1f kB\n",
               tHEst.addr(),
               tHEst.desc().get_info().to_string().c_str(),
               tHEst.desc().get_size_in_bytes() / 1024.0);
        printf("tCoef         : addr: %p, %s, size: %.1f kB\n",
               tCoef.addr(),
               tCoef.desc().get_info().to_string().c_str(),
               tCoef.desc().get_size_in_bytes() / 1024.0);
        printf("tDataEq      : addr: %p, %s, size: %.1f kB\n",
               tDataEq.addr(),
               tDataEq.desc().get_info().to_string().c_str(),
               tDataEq.desc().get_size_in_bytes() / 1024.0);
        printf("tReeDiag     : addr: %p, %s, size: %.1f kB\n",
               tReeDiag.addr(),
               tReeDiag.desc().get_info().to_string().c_str(),
               tReeDiag.desc().get_size_in_bytes() / 1024.0);
        printf("tLLR         : addr: %p, %s, size: %.1f kB\n\n",
               tLLR.addr(),
               tLLR.desc().get_info().to_string().c_str(),
               tLLR.desc().get_size_in_bytes() / 1024.0);
        printf("tDbg        : addr: %p, %s, size: %.1f kB\n",
               tDbg.addr(),
               tDbg.desc().get_info().to_string().c_str(),
               tDbg.desc().get_size_in_bytes() / 1024.0);


        lwdaEvent_t eStart, eStop;

        LWDA_CHECK(lwdaEventCreateWithFlags(&eStart, lwdaEventBlockingSync));
        LWDA_CHECK(lwdaEventCreateWithFlags(&eStop, lwdaEventBlockingSync));

        if(enLwprof) lwdaProfilerStart();

        // Launch a delay kernel to keep GPU busy
        //    gpu_ms_delay(1000);
        TimePoint startTime = Clock::now();

        LWDA_CHECK(lwdaEventRecord(eStart, lwStream));

        for(uint32_t i = 0; i < nIter; ++i)
        {
            lwphyStatus_t chEstStat = lwphyChannelEst(cellId,
                                                      slotNumber,
                                                      nBSAnts,
                                                      nLayers,
                                                      nDMRSSyms,
                                                      nDMRSGridsPerPRB,
                                                      activeDMRSGridBmsk,
                                                      nTotalDMRSPRB,
                                                      nTotalDataPRB,
                                                      Nh,
                                                      tDataRx.desc().handle(),
                                                      tDataRx.addr(),
                                                      tWFreq.desc().handle(),
                                                      tWFreq.addr(),
                                                      tShiftSeq.desc().handle(),
                                                      tShiftSeq.addr(),
                                                      tUnShiftSeq.desc().handle(),
                                                      tUnShiftSeq.addr(),
                                                      tHEst.desc().handle(),
                                                      tHEst.addr(),
                                                      tDbg.desc().handle(),
                                                      tDbg.addr(),
                                                      lwStream);

            if(LWPHY_STATUS_SUCCESS != chEstStat) throw lwphy::lwphy_exception(chEstStat);

#ifdef SYMB_EQ
            lwphyStatus_t chEqCoefComputeStat = lwphyChannelEqCoefCompute(nBSAnts,
                                                                         nLayers,
                                                                         Nh,
                                                                         Nprb,
                                                                         tHEst.desc().handle(),
                                                                         tHEst.addr(),
                                                                         tNoisePwr.desc().handle(),
                                                                         tNoisePwr.addr(),
                                                                         tCoef.desc().handle(),
                                                                         tCoef.addr(),
                                                                         tReeDiag.desc().handle(),
                                                                         tReeDiag.addr(),
                                                                         tDbg.desc().handle(),
                                                                         tDbg.addr(),
                                                                         lwStream);

           if(LWPHY_STATUS_SUCCESS != chEqCoefComputeStat) throw lwphy::lwphy_exception(chEqCoefComputeStat);

           lwphyStatus_t chEqSoftDemapStat = lwphyChannelEqSoftDemap(nBSAnts,
                                                                     nLayers,
                                                                     Nh,
                                                                     Nd,
                                                                     Nprb,
                                                                     tData_sym_loc.desc().handle(),
                                                                     tData_sym_loc.addr(),
                                                                     tQamInfo.desc().handle(),
                                                                     tQamInfo.addr(),
                                                                     tCoef.desc().handle(),
                                                                     tCoef.addr(),
                                                                     tReeDiag.desc().handle(),
                                                                     tReeDiag.addr(),
                                                                     tDataRx.desc().handle(),
                                                                     tDataRx.addr(),
                                                                     tDataEq.desc().handle(),
                                                                     tDataEq.addr(),
                                                                     tLLR.desc().handle(),
                                                                     tLLR.addr(),
                                                                     tDbg.desc().handle(),
                                                                     tDbg.addr(),
                                                                     lwStream);

          if(LWPHY_STATUS_SUCCESS != chEqSoftDemapStat) throw lwphy::lwphy_exception(chEqSoftDemapStat);
#else
            lwphyStatus_t chEqStat = lwphyChannelEq(nBSAnts,
                                                    nLayers,
                                                    Nh,
                                                    Nf,
                                                    Nd,
                                                    qam,
                                                    tData_sym_loc.desc().handle(),
                                                    tData_sym_loc.addr(),
                                                    tDataRx.desc().handle(),
                                                    tDataRx.addr(),
                                                    tHEst.desc().handle(),
                                                    tHEst.addr(),
                                                    tNoisePwr.desc().handle(),
                                                    tNoisePwr.addr(),
                                                    tDataEq.desc().handle(),
                                                    tDataEq.addr(),
                                                    tReeDiag.desc().handle(),
                                                    tReeDiag.addr(),
                                                    tLLR.desc().handle(),
                                                    tLLR.addr(),
                                                    lwStream);

            if(LWPHY_STATUS_SUCCESS != chEqStat)  throw lwphy::lwphy_exception(chEqStat);
#endif // SYMB_EQ
       }

        LWDA_CHECK(lwdaEventRecord(eStop, lwStream));
        LWDA_CHECK(lwdaEventSynchronize(eStop));

        lwdaStreamSynchronize(lwStream);
        // lwdaDeviceSynchronize();

        TimePoint stopTime = Clock::now();

        if(enLwprof) lwdaProfilerStop();

        float elapsedMs = 0.0f;
        lwdaEventElapsedTime(&elapsedMs, eStart, eStop);

        printf("Exelwtion time (PUSCH RX pipeline front end)\n");
        printf("---------------------------------------------------------------\n");
        printf("Average (over %d runs) elapsed time in usec (LWCA event) = %.0f\n",
               nIter,
               elapsedMs * 1000 / nIter);

        duration<float, std::milli> diff = stopTime - startTime;
        printf("Average (over %d runs) elapsed time in usec (wall clock) w/ 1s delay kernel = %.0f\n",
               nIter,
               diff.count() * 1000 / nIter);

        LWDA_CHECK(lwdaEventDestroy(eStart));
        LWDA_CHECK(lwdaEventDestroy(eStop));
    
        // Colwert to FP32 format for MATLAB readability
        lwphy::tensor_device tOutHEst(lwphy::tensor_info(LWPHY_C_32F, tHEst.layout()));
        lwphy::tensor_device tOutCoef(lwphy::tensor_info(LWPHY_C_32F, tCoef.layout()));
        lwphy::tensor_device tOutDataEq(lwphy::tensor_info(LWPHY_C_32F, tDataEq.layout()));
        lwphy::tensor_device tOutReeDiag(lwphy::tensor_info(LWPHY_R_32F, tReeDiag.layout()));
        lwphy::tensor_device tOutLLR(lwphy::tensor_info(LWPHY_R_32F, tLLR.layout()));
        lwphy::tensor_device tOutDbg(lwphy::tensor_info(LWPHY_C_32F, tDbg.layout()));

        lwphyStatus_t tensorColwertStat = lwphyColwertTensor(tOutHEst.desc().handle(), // dst tensor
                                                             tOutHEst.addr(),          // dst address
                                                             tHEst.desc().handle(),    // src tensor
                                                             tHEst.addr(),             // src address
                                                             lwStream);                // LWCA stream
        if(LWPHY_STATUS_SUCCESS != tensorColwertStat) throw lwphy::lwphy_exception(tensorColwertStat);
 
        tensorColwertStat = lwphyColwertTensor(tOutCoef.desc().handle(), // dst tensor
                                               tOutCoef.addr(),          // dst address
                                               tCoef.desc().handle(),    // src tensor
                                               tCoef.addr(),             // src address
                                               lwStream);                // LWCA stream
        if(LWPHY_STATUS_SUCCESS != tensorColwertStat) throw lwphy::lwphy_exception(tensorColwertStat);
        
        tensorColwertStat = lwphyColwertTensor(tOutDataEq.desc().handle(), // dst tensor
                                               tOutDataEq.addr(),          // dst address
                                               tDataEq.desc().handle(),    // src tensor
                                               tDataEq.addr(),             // src address
                                               lwStream);                  // LWCA stream
        if(LWPHY_STATUS_SUCCESS != tensorColwertStat) throw lwphy::lwphy_exception(tensorColwertStat);

        tensorColwertStat = lwphyColwertTensor(tOutReeDiag.desc().handle(), // dst tensor
                                               tOutReeDiag.addr(),          // dst address
                                               tReeDiag.desc().handle(),    // src tensor
                                               tReeDiag.addr(),             // src address
                                               lwStream);                   // LWCA stream
        if(LWPHY_STATUS_SUCCESS != tensorColwertStat) throw lwphy::lwphy_exception(tensorColwertStat);

        tensorColwertStat = lwphyColwertTensor(tOutLLR.desc().handle(), // dst tensor
                                               tOutLLR.addr(),          // dst address
                                               tLLR.desc().handle(),    // src tensor
                                               tLLR.addr(),             // src address
                                               lwStream);               // LWCA stream
        if(LWPHY_STATUS_SUCCESS != tensorColwertStat) throw lwphy::lwphy_exception(tensorColwertStat);

        tensorColwertStat = lwphyColwertTensor(tOutDbg.desc().handle(), // dst tensor
                                               tOutDbg.addr(),          // dst address
                                               tDbg.desc().handle(),    // src tensor
                                               tDbg.addr(),             // src address
                                               lwStream);               // LWCA stream
        if(LWPHY_STATUS_SUCCESS != tensorColwertStat) throw lwphy::lwphy_exception(tensorColwertStat);

        // Wait for copy to complete
        // lwdaDeviceSynchronize();
        lwdaStreamSynchronize(lwStream);

        // Write outputs
        std::unique_ptr<hdf5hpp::hdf5_file> dbgProbeUqPtr;
        if(!outputFilename.empty())
        {
            dbgProbeUqPtr.reset(new hdf5hpp::hdf5_file(hdf5hpp::hdf5_file::create(outputFilename.c_str())));
           
            // Write channel estimator outputs
            lwphy::write_HDF5_dataset(*dbgProbeUqPtr, tOutHEst   , "HEst");
    
            // Write channel equalizer outputs
            lwphy::write_HDF5_dataset(*dbgProbeUqPtr, tOutCoef   , "Coef");
            lwphy::write_HDF5_dataset(*dbgProbeUqPtr, tOutDataEq , "DataEq");
#ifdef SYMB_EQ
            lwphy::write_HDF5_dataset(*dbgProbeUqPtr, tOutReeDiag, "ReeDiagIlw");
#else
            lwphy::write_HDF5_dataset(*dbgProbeUqPtr, tOutReeDiag, "ReeDiag");
#endif // SYMB_EQ
            lwphy::write_HDF5_dataset(*dbgProbeUqPtr, tOutLLR    , "LLR");
            lwphy::write_HDF5_dataset(*dbgProbeUqPtr, tOutDbg    , "Dbg");
        }
        // Wait for writes to complete
        // lwdaDeviceSynchronize();
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
