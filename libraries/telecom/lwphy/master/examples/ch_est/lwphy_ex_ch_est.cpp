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
#include "hdf5hpp.hpp"
#include "lwphy_hdf5.hpp"
#include "lwphy.hpp"

#include <cstring>
#include <iostream>
#include <unistd.h> // for getcwd()
#include <dirent.h> // opendir, readdir
#include <errno.h>
#include <sys/stat.h> // for mkdir

int writeProbe(char const* pFName, void const* pBuffer, size_t nBytes)
{
    FILE*  pFileHandle;
    size_t nWritten;

    /* Write debug files into out directory, if out does not exist already then create it */
    char const* pDirName = "out";
    DIR*        pDir;
    if((pDir = opendir(pDirName)) == NULL)
    {
        if(ENOENT == errno)
        {
            if(mkdir(pDirName, 0777) != 0)
            {
                printf("writeDebugFile: failed to create directory, error: %s\n", strerror(errno));
                return -2;
            }
        }
        else
        {
            printf("writeDebugFile: failed to open existing directory, error: %s\n", strerror(errno));
            return -3;
        }
    }

    /* Append the directory name, file name, instance index and UE index */
    std::string fName = std::string(pDirName) + "/" + std::string(pFName) + "_l.out";

    if(NULL == (pFileHandle = fopen(fName.c_str(), "ab")))
    {
        closedir(pDir);
        printf("Problem opening file, error: %s\n", strerror(errno));
        return -4;
    }

    nWritten = fwrite(pBuffer, 1, (size_t)nBytes, pFileHandle);
    printf("Wrote %d bytes, asked to write %d bytes, error: %s\n", (int)nWritten, (int)nBytes, strerror(errno));
    fclose(pFileHandle);
    closedir(pDir);

    return 0;
}

////////////////////////////////////////////////////////////////////////
// usage()
void usage()
{
    printf("lwphy_ex_channel_eq [options]\n");
    printf("  Options:\n");
    printf("    -i  input_filename     Input HDF5 filename, which must contain the following datasets:\n");
    printf("                           Data_rx      : received data (frequency-time) to be equalized\n");
    printf("                           WFreq        : interpolation filter coefficients used in channel estimation\n");
    printf("                           ShiftSeq     : sequence to be applied to DMRS tones containing descrambling code and delay shift for channel centering\n");
    printf("                           UnShiftSeq   : sequence to remove the delay shift from estimated channel\n");
    printf("    -h                     Display usage information\n");
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
        bool        b16 = false;
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

        // clang-format off
        //------------------------------------------------------------------
        // Copy file input data to tensors in device memory
        lwphy::tensor_device tDataRx        = lwphy::tensor_from_dataset(fInput.open_dataset("DataRx"),        LWPHY_TENSOR_ALIGN_TIGHT);
        lwphy::tensor_device tWFreq         = lwphy::tensor_from_dataset(fInput.open_dataset("WFreq"),         LWPHY_TENSOR_ALIGN_TIGHT);
        lwphy::tensor_device tShiftSeq      = lwphy::tensor_from_dataset(fInput.open_dataset("ShiftSeq"), LWPHY_TENSOR_ALIGN_TIGHT);
        lwphy::tensor_device tUnShiftSeq    = lwphy::tensor_from_dataset(fInput.open_dataset("UnShiftSeq"),    LWPHY_TENSOR_ALIGN_TIGHT);
        // clang-format on


        printf("Input tensors:\n");
        printf("---------------------------------------------------------------\n");
        printf("DataRx         : %s\n", tDataRx.desc().get_info().to_string(false).c_str());
        printf("WFreq          : %s\n", tWFreq.desc().get_info().to_string(false).c_str());
        printf("ShiftSeq       : %s\n", tShiftSeq.desc().get_info().to_string(false).c_str());
        printf("UnShiftSeq     : %s\n", tUnShiftSeq.desc().get_info().to_string(false).c_str());

        //------------------------------------------------------------------
        // Parameters assumed and derived from the input data
        //
        constexpr uint32_t N_TONES_PER_PRB              = 12;
        constexpr uint32_t N_DMRS_GRIDS_PER_PRB         = 2;
        constexpr uint32_t N_DMRS_GRID_TONES_PER_PRB    = N_TONES_PER_PRB / N_DMRS_GRIDS_PER_PRB;
        constexpr uint32_t N_INTERP_DMRS_TONES_PER_GRID = N_TONES_PER_PRB;

        uint32_t cellId             = 0;
        uint32_t slotNumber         = 0;
        uint32_t nBSAnts            = tDataRx.dimensions()[2];
        uint32_t nLayers            = 8; // N_LAYERS = N_DMRS_SYMS_FOCC*N_DMRS_SYMS_TOCC*N_DMRS_GRIDS
        uint32_t activeDMRSGridBmsk = (1UL << 0) | (1UL << 1);
        uint32_t Nf                 = tDataRx.dimensions()[0];
        uint32_t nDMRSSyms          = tShiftSeq.dimensions()[1];
        uint32_t nDMRSGridsPerPRB   = N_DMRS_GRIDS_PER_PRB;
        uint32_t nTotalDMRSPRB      = tShiftSeq.dimensions()[0] / N_DMRS_GRID_TONES_PER_PRB;
        uint32_t nTotalDataPRB      = nTotalDMRSPRB;
        uint32_t Nh                 = 1;
        uint32_t Nd                 = tDataRx.dimensions()[1];
        uint32_t nTotalDataTones    = nTotalDMRSPRB * N_TONES_PER_PRB;
        // uint32_t NfTst           = UnShiftSeq_info.layout.dimensions()[0];

        printf("Assumed and Derived parameters:\n");
        printf("---------------------------------------------------------------\n");
        printf("cellId             : %i\n", cellId);
        printf("slotNumber         : %i\n", slotNumber);
        printf("nBSAnts            : %i\n", nBSAnts);
        printf("nLayers            : %i\n", nLayers);
        printf("Nf                 : %i\n", Nf); // # of estimates of H in frequency
        printf("nDMRSSyms          : %i\n", nDMRSSyms);
        printf("nDMRSGridsPerPRB   : %i\n", nDMRSGridsPerPRB);
        printf("activeDMRSGridBmsk : 0x%x\n", activeDMRSGridBmsk);
        printf("nTotalDMRSPRB      : %i\n", nTotalDMRSPRB);
        printf("nTotalDataPRB      : %i\n", nTotalDataPRB);
        printf("Nh                 : %i\n", Nh);   // # of estimates of H in time
        printf("Nd                 : %i\n\n", Nd); // # of data symbols

        //------------------------------------------------------------------
        // Allocate an output tensor in device memory
        lwphy::tensor_device tHEst(lwphy::tensor_info(LWPHY_C_32F,
                                                      {static_cast<int>(nBSAnts),
                                                       static_cast<int>(nLayers),
                                                       static_cast<int>(nTotalDataTones),
                                                       static_cast<int>(Nh)}),
                                   LWPHY_TENSOR_ALIGN_TIGHT);

        lwphy::tensor_device tDbg(lwphy::tensor_info(LWPHY_C_32F,
                                                     {static_cast<int>(Nf/2),
                                                      static_cast<int>(nDMRSSyms),
                                                      static_cast<int>(1),
                                                      static_cast<int>(1)}),
                                   LWPHY_TENSOR_ALIGN_TIGHT);


        printf("Tensor layout:\n");
        printf("---------------------------------------------------------------\n");
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
        printf("tHEst         : addr: %p, %s, size: %.1f kB\n",
               tHEst.addr(),
               tHEst.desc().get_info().to_string().c_str(),
               tHEst.desc().get_size_in_bytes() / 1024.0);
        printf("tDbg        : addr: %p, %s, size: %.1f kB\n",
               tDbg.addr(),
               tDbg.desc().get_info().to_string().c_str(),
               tDbg.desc().get_size_in_bytes() / 1024.0);


        lwphyStatus_t s = lwphyChannelEst(cellId,
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
                                          0);

        if(LWPHY_STATUS_SUCCESS != s)
        {
            throw lwphy::lwphy_exception(s);
        }

        lwdaDeviceSynchronize();

        void* tHEstHost;
        lwdaHostAlloc(&tHEstHost, tHEst.desc().get_size_in_bytes(), lwdaHostAllocMapped | lwdaHostAllocWriteCombined);
        lwdaMemcpy(tHEstHost, tHEst.addr(), tHEst.desc().get_size_in_bytes(), lwdaMemcpyDeviceToHost);

        int errCode = writeProbe("dbgHEst", tHEstHost, tHEst.desc().get_size_in_bytes());
        printf("nBytesWritten %lu errCode %d\n", tHEst.desc().get_size_in_bytes(), errCode);
        lwdaFreeHost(tHEstHost);

        // Write outputs
        hdf5hpp::hdf5_file fOutput1 = hdf5hpp::hdf5_file::create("HEst.h5");
        lwphy::write_HDF5_dataset(fOutput1, tHEst, "HEst");

#if 0
        lwphy::tensor_info tinfo = tW_time.desc().get_info();
        std::vector<float> W_time_host(tW_time.desc().get_size_in_bytes() / sizeof(float));
        if(lwdaSuccess != lwdaMemcpy(W_time_host.data(), tW_time.addr(), tW_time.desc().get_size_in_bytes(), lwdaMemcpyDeviceToHost))
        {
            fprintf(stderr, "Error performing memcpy\n");
        }

        for(int j = 0; j < 4; ++j)
        {
            for(int i = 0; i < 14; ++i)
            {
                size_t offset = tinfo.layout.get_offset(i, j, 11);
                printf("offset: i = %i, j = %i, %lu, value: %f\n", i, j, offset, W_time_host[offset]);
            }
        }
#endif
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
