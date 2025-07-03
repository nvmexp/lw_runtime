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
#include <numeric>
#include "hdf5hpp.hpp"
#include "lwphy_hdf5.hpp"
#include "lwphy.hpp"

#define EQ_COEF_APPLY_VER (2)

////////////////////////////////////////////////////////////////////////
// usage()
void usage()
{
    printf("lwphy_ex_channel_eq [options]\n");
    printf("  Options:\n");
    printf("    -i  input_filename     Input HDF5 filename, which must contain the following datasets:\n");
    printf("                           Data_sym_loc : locations of data symbols within the subframe\n");
    printf("                           RxxIlw   : Symbol energy covariance\n");
    printf("                           Noise_pwr: noise power at frequency-time bins where channel (H) is estimated\n");
    printf("                           H        : Channel coupling matrix in frequency-time \n");
    printf("                           Data_rx  : received data (frequency-time) to be equalized\n");
    printf("                           Data_eq  : equalized output data (in frequency-time)\n");
    printf("    -h                     Display usage information\n");
    printf("    -o  outfile            Write pipeline tensors to an HDF5 output file.\n");
    printf("                           (Not recommended for use during timing runs.)\n");
    printf("    --H                    0(default): No FP16\n");
    printf("                           1         : FP16 format used for received data samples only\n");
    printf("                           2         : FP16 format used for all front end params\n");
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
        uint32_t    fp16Mode = 0xBAD;
 
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
                case '-':
                    switch(argv[iArg][2])
                    {
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
        // Open the input file and required datasets
        hdf5hpp::hdf5_file fInput = hdf5hpp::hdf5_file::open(inputFilename.c_str());

        //------------------------------------------------------------------
        // Allocate tensors in device memory

        // Check FP16 mode of operation
        bool isDataFp16     = false; 
        bool isChannelFp16  = false;
        switch(fp16Mode)
        {
            case 0 : isDataFp16 = false; isChannelFp16 = false; break;
            case 1 : isDataFp16 = true ; isChannelFp16 = false; break;
            case 2 : isDataFp16 = true ; isChannelFp16 = true ; break;
            default: isDataFp16 = false; isChannelFp16 = false; break;
        }
        lwphyDataType_t feDataType = isDataFp16 ? LWPHY_R_16F : LWPHY_R_32F;
        lwphyDataType_t feCplxDataType = isDataFp16 ? LWPHY_C_16F : LWPHY_C_32F;

        lwphyDataType_t feChannelType = isChannelFp16 ? LWPHY_R_16F : LWPHY_R_32F;
        lwphyDataType_t feCplxChannelType = isChannelFp16 ? LWPHY_C_16F : LWPHY_C_32F;
        
        // clang-format off
        lwphy::tensor_device tData_sym_loc  = lwphy::tensor_from_dataset(fInput.open_dataset("Data_sym_loc") ,                    LWPHY_TENSOR_ALIGN_TIGHT, lwStream);
        lwphy::tensor_device tQamInfo       = lwphy::tensor_from_dataset(fInput.open_dataset("QamInfo")      ,                    LWPHY_TENSOR_ALIGN_TIGHT, lwStream);
        lwphy::tensor_device tDataRx        = lwphy::tensor_from_dataset(fInput.open_dataset("DataRx")       , feCplxDataType   , LWPHY_TENSOR_ALIGN_TIGHT, lwStream);
        lwphy::tensor_device tRxxIlw        = lwphy::tensor_from_dataset(fInput.open_dataset("RxxIlw")       , feChannelType    , LWPHY_TENSOR_ALIGN_TIGHT, lwStream);
#if 0        
        lwphy::tensor_device tNoisePwr      = lwphy::tensor_from_dataset(fInput.open_dataset("Noise_pwr")    , feChannelType, LWPHY_TENSOR_ALIGN_TIGHT, lwStream);
#else        
        lwphy::tensor_device tNoisePwr      = lwphy::tensor_from_dataset(fInput.open_dataset("Noise_pwr")    , feCplxChannelType, LWPHY_TENSOR_ALIGN_TIGHT, lwStream);
#endif        
        lwphy::tensor_device tHEst          = lwphy::tensor_from_dataset(fInput.open_dataset("H")            , feCplxChannelType, LWPHY_TENSOR_ALIGN_TIGHT, lwStream);
        // clang-format on

        // Ensure colwersion completes
        // lwdaDeviceSynchronize();
        lwdaStreamSynchronize(lwStream);

        printf("Input tensors:\n");
        printf("---------------------------------------------------------------\n");
        printf("Data_sym_loc   : %s\n", tData_sym_loc.desc().get_info().to_string(false).c_str());
        printf("QamInfo        : %s\n", tQamInfo.desc().get_info().to_string(false).c_str());
        printf("RxxIlw         : %s\n", tRxxIlw.desc().get_info().to_string(false).c_str());
        printf("Noise_pwr      : %s\n", tNoisePwr.desc().get_info().to_string(false).c_str());
        printf("H              : %s\n", tHEst.desc().get_info().to_string(false).c_str());
        printf("Data_rx        : %s\n", tDataRx.desc().get_info().to_string(false).c_str());

        //------------------------------------------------------------------
        // Determine parameters derived from the input data
        uint32_t nBSAnts = tHEst.layout().dimensions()[0];
        uint32_t nLayers = tHEst.layout().dimensions()[1];
        uint32_t Nf      = (tHEst.layout().dimensions()[2] == 0) ? 1 : tHEst.layout().dimensions()[2];
        uint32_t Nprb    = Nf/LWPHY_N_TONES_PER_PRB;
        uint32_t Nh      = (tHEst.layout().dimensions()[3] == 0) ? 1 : tHEst.layout().dimensions()[3];
        uint32_t Nd      = tData_sym_loc.layout().dimensions()[0];

        printf("Derived parameters:\n");
        printf("---------------------------------------------------------------\n");
        printf("nBSAnts            : %i\n", nBSAnts);
        printf("nLayers            : %i\n", nLayers);
        printf("Nf                 : %i\n", Nf); // # of estimates of H in frequency
        printf("Nprb               : %i\n", Nprb);
        printf("Nh                 : %i\n", Nh); // # of estimates of H in time
        printf("Nd                 : %i\n", Nd); // # of data symbols

        //------------------------------------------------------------------
        // Allocate tensors in device memory
        // clang-format off

#if (EQ_COEF_APPLY_VER == 1)
        lwphy::tensor_device tCoef(lwphy::tensor_info(feCplxChannelType,
                                                     {static_cast<int>(nLayers),
                                                      static_cast<int>(nBSAnts),
                                                      static_cast<int>(Nf),
                                                      static_cast<int>(Nh)}),
                                    LWPHY_TENSOR_ALIGN_TIGHT);
        lwphy::tensor_device tReeDiag(lwphy::tensor_info(feDataType,
                                                        {static_cast<int>(nLayers),
                                                         static_cast<int>(Nf),
                                                         static_cast<int>(Nh)}),
                                      LWPHY_TENSOR_ALIGN_TIGHT);
#elif (EQ_COEF_APPLY_VER == 2)
        lwphy::tensor_device tCoef(lwphy::tensor_info(feCplxChannelType,
                                                     {static_cast<int>(nBSAnts),
                                                      static_cast<int>(LWPHY_N_TONES_PER_PRB),
                                                      static_cast<int>(nLayers),
                                                      static_cast<int>(Nprb)}),
                                    LWPHY_TENSOR_ALIGN_TIGHT);
        lwphy::tensor_device tReeDiag(lwphy::tensor_info(feDataType,
                                                        {static_cast<int>(LWPHY_N_TONES_PER_PRB),
                                                         static_cast<int>(nLayers),
                                                         static_cast<int>(Nprb)}),
                                      LWPHY_TENSOR_ALIGN_TIGHT);
#endif
        lwphy::tensor_device tDataEq(lwphy::tensor_info(feCplxDataType,
                                                       {static_cast<int>(nLayers),
                                                        static_cast<int>(Nf),
                                                        static_cast<int>(Nd)}),
                                     LWPHY_TENSOR_ALIGN_TIGHT);

       
        lwphy::tensor_device tLLR(lwphy::tensor_info(LWPHY_R_32F, // feDataType, keeping LLR format to FP32 until backend supports it
                                                    {static_cast<int>(LWPHY_QAM_256),
                                                     static_cast<int>(nLayers),
                                                     static_cast<int>(Nf),
                                                     static_cast<int>(Nd)}),
                                  LWPHY_TENSOR_ALIGN_TIGHT);

         lwphy::tensor_device tDbg(lwphy::tensor_info(feCplxChannelType,
                                                     {static_cast<int>(nLayers), // static_cast<int>(nLayers), // static_cast<int>(nBSAnts),
                                                      static_cast<int>(nLayers), // static_cast<int>(nBSAnts), // static_cast<int>(nLayers),
                                                      static_cast<int>(Nf),      // static_cast<int>(Nf),
                                                      static_cast<int>(Nh)}),
                                   LWPHY_TENSOR_ALIGN_TIGHT);

       // clang-format on

        printf("Tensor layout:\n");
        printf("---------------------------------------------------------------\n");
        printf("tDataSymLoc : addr: %p, %s, size: %.1f kB\n",
               tData_sym_loc.addr(),
               tData_sym_loc.desc().get_info().to_string().c_str(),
               tData_sym_loc.desc().get_size_in_bytes() / 1024.0);
        printf("tQamInfo    : addr: %p, %s, size: %.1f kB\n",
               tQamInfo.addr(),
               tQamInfo.desc().get_info().to_string().c_str(),
               tQamInfo.desc().get_size_in_bytes() / 1024.0);
        printf("tRxxIlw     : addr: %p, %s, size: %.1f kB\n",
               tRxxIlw.addr(),
               tRxxIlw.desc().get_info().to_string().c_str(),
               tRxxIlw.desc().get_size_in_bytes() / 1024.0);
        printf("tNoisePwr  : addr: %p, %s, size: %.1f kB\n",
               tNoisePwr.addr(),
               tNoisePwr.desc().get_info().to_string().c_str(),
               tNoisePwr.desc().get_size_in_bytes() / 1024.0);
        printf("tHEst      : addr: %p, %s, size: %.1f kB\n",
               tHEst.addr(),
               tHEst.desc().get_info().to_string().c_str(),
               tHEst.desc().get_size_in_bytes() / 1024.0);
        printf("tCoef      : addr: %p, %s, size: %.1f kB\n",
               tCoef.addr(),
               tCoef.desc().get_info().to_string().c_str(),
               tCoef.desc().get_size_in_bytes() / 1024.0);
        printf("tDataRx    : addr: %p, %s, size: %.1f kB\n",
               tDataRx.addr(),
               tDataRx.desc().get_info().to_string().c_str(),
               tDataRx.desc().get_size_in_bytes() / 1024.0);
        printf("tDataEq    : addr: %p, %s, size: %.1f kB\n",
               tDataEq.addr(),
               tDataEq.desc().get_info().to_string().c_str(),
               tDataEq.desc().get_size_in_bytes() / 1024.0);
        printf("tReeDiag   : addr: %p, %s, size: %.1f kB\n",
               tReeDiag.addr(),
               tReeDiag.desc().get_info().to_string().c_str(),
               tReeDiag.desc().get_size_in_bytes() / 1024.0);
        printf("tLLR        : addr: %p, %s, size: %.1f kB\n",
               tLLR.addr(),
               tLLR.desc().get_info().to_string().c_str(),
               tLLR.desc().get_size_in_bytes() / 1024.0);
        printf("tDbg        : addr: %p, %s, size: %.1f kB\n",
               tDbg.addr(),
               tDbg.desc().get_info().to_string().c_str(),
               tDbg.desc().get_size_in_bytes() / 1024.0);


#if 0        
        lwphyStatus_t s = lwphyChannelEq(nBSAnts,
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

        if(LWPHY_STATUS_SUCCESS != s)
        {
            throw lwphy::lwphy_exception(s);
        }
#else
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

#endif
        // Wait for kernel to complte exelwtion
        lwdaStreamSynchronize(lwStream);
        
        // Colwert to FP32 format for MATLAB readability
        lwphy::tensor_device tOutCoef(lwphy::tensor_info(LWPHY_C_32F, tCoef.layout()));
        lwphy::tensor_device tOutDataEq(lwphy::tensor_info(LWPHY_C_32F, tDataEq.layout()));
        lwphy::tensor_device tOutReeDiag(lwphy::tensor_info(LWPHY_R_32F, tReeDiag.layout()));
        lwphy::tensor_device tOutLLR(lwphy::tensor_info(LWPHY_R_32F, tLLR.layout()));
        lwphy::tensor_device tOutDbg(lwphy::tensor_info(LWPHY_C_32F, tDbg.layout()));

        lwphyStatus_t tensorColwertStat = lwphyColwertTensor(tOutDataEq.desc().handle(), // dst tensor
                                                             tOutDataEq.addr(),          // dst address
                                                             tDataEq.desc().handle(),    // src tensor
                                                             tDataEq.addr(),             // src address
                                                             lwStream);                  // LWCA stream
        if(LWPHY_STATUS_SUCCESS != tensorColwertStat) throw lwphy::lwphy_exception(tensorColwertStat);

        tensorColwertStat = lwphyColwertTensor(tOutCoef.desc().handle(), // dst tensor
                                               tOutCoef.addr(),          // dst address
                                               tCoef.desc().handle(),    // src tensor
                                               tCoef.addr(),             // src address
                                               lwStream);                // LWCA stream
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
#if 1           
           // Write channel equalizer outputs
            lwphy::write_HDF5_dataset(*dbgProbeUqPtr, tOutCoef   , "Coef");
            lwphy::write_HDF5_dataset(*dbgProbeUqPtr, tOutDataEq , "DataEq");
            lwphy::write_HDF5_dataset(*dbgProbeUqPtr, tOutReeDiag, "ReeDiagIlw");
            lwphy::write_HDF5_dataset(*dbgProbeUqPtr, tOutLLR    , "LLR");
            lwphy::write_HDF5_dataset(*dbgProbeUqPtr, tOutDbg    , "Dbg");
#else
           // Write channel equalizer outputs
#if 0
            lwphy::write_HDF5_dataset(*dbgProbeUqPtr, tCoef   , "Coef");
            lwphy::write_HDF5_dataset(*dbgProbeUqPtr, tDataEq , "DataEq");
            lwphy::write_HDF5_dataset(*dbgProbeUqPtr, tReeDiag, "ReeDiag");
            lwphy::write_HDF5_dataset(*dbgProbeUqPtr, tLLR    , "LLR");
            lwphy::write_HDF5_dataset(*dbgProbeUqPtr, tDbg    , "Dbg");
#endif 
#endif
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
