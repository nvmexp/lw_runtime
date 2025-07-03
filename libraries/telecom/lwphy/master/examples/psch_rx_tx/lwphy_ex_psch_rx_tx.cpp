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
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <unistd.h>  /* For SYS_xxx definitions */
#include <syscall.h> /* For SYS_xxx definitions */

#include "lwda_profiler_api.h"
#include "util.hpp"
#include "pusch_rx_test.hpp"

#include <chrono>
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
    printf("lwphy_ex_sch_rx_tx [options]\n");
    printf("  Options:\n");
    printf("    -h                     Display usage information\n");
    printf("    -d                     Enable debug\n");
    printf("    -i  input_filenames    Input HDF5 filename\n");
    printf("    -n  # of pipelines     Number of pipelines to run <1,2>\n");
    printf("    -c  CPU Id             CPU Id used to run the first pipeline, cpuIdPipeline[i] = cpuIdFirstPipeline + i\n");
    printf("    -g  GPU Id             GPU Id used to run all the pipelines\n");
    printf("    -o  outfile            Write pipeline tensors to an HDF5 output file.\n");
    printf("                           (Not recommended for use during timing runs.)\n");
    printf("    -r  # of iterations    Number of run iterations to run\n");
    printf("    -v                     Enable lwperf profiling with 1 iteration\n");
    printf("    -w  delay_ms           Set the initial GPU delay in milliseconds (default: 2000)\n");
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
        constexpr uint32_t N_MAX_INST = 8;
        int                iArg       = 1;
        std::string        inputFilename;
        std::string        outputFilename;
        bool               b16   = false;
        uint32_t           nInst = 1;
        int32_t            nGPUs = 0;
        LWDA_CHECK(lwdaGetDeviceCount(&nGPUs));
        int32_t  nMaxConlwrentThrds = std::thread::hardware_conlwrrency();
        int32_t  rxCpuIdFirstInst   = 0;
        int32_t  txCpuIdFirstInst   = 0;
        int32_t  gpuId              = 0;
        uint32_t nIterations        = 1000;
        bool     enable_lwprof      = false;
        bool     debug              = false;
        int      descramblingOn     = 1;
        uint32_t delay_ms           = 2000;
        uint32_t fp16Mode           = 1;

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
                case 'n':
                    if((++iArg >= argc) ||
                       (1 != sscanf(argv[iArg], "%i", &nInst)) ||
                       ((nInst <= 0) || (nInst > N_MAX_INST)))
                    {
                        fprintf(stderr,
                                "ERROR: Invalid number of instances (should be within [1,%d])\n",
                                N_MAX_INST);
                        exit(1);
                    }
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
                case 'w':
                    if((++iArg >= argc) ||
                       (1 != sscanf(argv[iArg], "%u", &delay_ms)))
                    {
                        fprintf(stderr, "ERROR: Invalid delay\n");
                        exit(1);
                    }
                    ++iArg;
                    break;
                case 'c':
                    switch(argv[iArg][2])
                    {
                      case 'r':
                        if((++iArg >= argc) ||
                           (1 != sscanf(argv[iArg], "%i", &rxCpuIdFirstInst)) ||
                           ((rxCpuIdFirstInst < 0) || (rxCpuIdFirstInst > nMaxConlwrentThrds - nInst)))
                        {
                            fprintf(stderr, "ERROR: Invalid Rx CPU Id (should be within [0,%d], nMaxConlwrrentThrds %d)\n", nMaxConlwrentThrds - nInst, nMaxConlwrentThrds);
                            exit(1);
                        }
                        ++iArg;
                        break;
                      case 't':
                        if((++iArg >= argc) ||
                           (1 != sscanf(argv[iArg], "%i", &txCpuIdFirstInst)) ||
                           ((txCpuIdFirstInst < 0) || (txCpuIdFirstInst > nMaxConlwrentThrds - nInst)))
                        {
                            fprintf(stderr, "ERROR: Invalid Tx CPU Id (should be within [0,%d], nMaxConlwrrentThrds %d)\n", nMaxConlwrentThrds - nInst, nMaxConlwrentThrds);
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
 
                case 'd':
                    debug = true;
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
                        if((++iArg >= argc) || (1 != sscanf(argv[iArg], "%i", &fp16Mode)) || (1 < fp16Mode))
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

        // Set thread affinity to specified CPU
        int cpuId = rxCpuIdFirstInst;
        pid_t  pid = (pid_t)syscall(SYS_gettid);
        cpu_set_t cpuSet;
        CPU_ZERO(&cpuSet);
        CPU_SET(cpuId, &cpuSet);
    
        int affinitySetRet = sched_setaffinity(pid, sizeof(cpuSet), &cpuSet);
        if(0 == affinitySetRet)
        {
            printf("Main thread: pid %d set affinity to CPU %d\n", pid, cpuId);
        }
        else
        {
            printf("Main thread: failed to set affinity pid %d to CPU %d, return code %d err %s\n",
                   pid,
                   cpuId,
                   affinitySetRet,
                   strerror(errno));
        }
    
        int minLwStrmPrio, maxLwStrmPrio;
        LWDA_CHECK(lwdaDeviceGetStreamPriorityRange(&minLwStrmPrio, &maxLwStrmPrio));
        printf("Stream priority range [%d, %d]\n", minLwStrmPrio, maxLwStrmPrio);

        lwdaStream_t delayLwStrm;
        LWDA_CHECK(lwdaStreamCreateWithPriority(&delayLwStrm, lwdaStreamNonBlocking, maxLwStrmPrio));

        bool                       startSyncPt = false;
        std::mutex                 cvStartSyncPtMutex;
        std::condition_variable    cvStartSyncPt;
        std::atomic<std::uint32_t> atmSyncPtWaitCnt(0);

        int rxLwStrmPrio = minLwStrmPrio;
        std::vector<int> rxLwStrmPrios(nInst, rxLwStrmPrio);
        PuschRxTest puschRxTest("PuschRx", nInst, delayLwStrm, rxLwStrmPrios, startSyncPt, cvStartSyncPtMutex, cvStartSyncPt, atmSyncPtWaitCnt);
 
        int txLwStrmPrio = maxLwStrmPrio;
        std::vector<int> txLwStrmPrios(nInst, txLwStrmPrio);
        // using a PuschRxTest object as a stub for "pdschTxTest" object
        PuschRxTest pdschTxTest("PdschTx", nInst, delayLwStrm, txLwStrmPrios, startSyncPt, cvStartSyncPtMutex, cvStartSyncPt, atmSyncPtWaitCnt);

        lwdaSetDevice(gpuId);

        // @todo:
        int rxGpuId = gpuId;
        int txGpuId = gpuId;

        // GPU and CPU Ids
        std::vector<int> rxGpuIds(nInst, rxGpuId);
        std::vector<int> rxCpuIds(nInst);
        uint32_t         instCpuId = 0;
        for(auto& cpuId : rxCpuIds) { cpuId = instCpuId + rxCpuIdFirstInst; }

        std::vector<int> txGpuIds(nInst, txGpuId);
        std::vector<int> txCpuIds(nInst);
        for(auto& cpuId : txCpuIds) { cpuId = instCpuId + txCpuIdFirstInst; }

        // Scheduling policy and priorities
        std::vector<int> schdPolicies(nInst, SCHED_RR); // SCHED_FIFO // SCHED_RR
        std::vector<int> rxThrdPrios;
        std::vector<int> txThrdPrios;
        for(auto const& policy : schdPolicies)
        {
            int txThrdPrio = sched_get_priority_max(policy);
            txThrdPrios.emplace_back(txThrdPrio);

            int rxThrdPrio = txThrdPrio - 1; 
            rxThrdPrios.emplace_back(rxThrdPrio);
        }

        uint32_t rxTxDelayUs = 100;
        
        // Disabled unitl HDF5 supports thread-safe access
#if 0
        auto pdschTxTestExelwte = [&pdschTxTest,
                                   &nIterations,
                                   &enable_lwprof,
                                   &descramblingOn,
                                   &rxTxDelayUs,
                                   &txCpuIds,
                                   &schdPolicies,
                                   &txThrdPrios,
                                   &txGpuIds,
                                   &inputFilename,
                                   &outputFilename,
                                   &fp16Mode,
                                   &debug]()
        {
            pdschTxTest.Setup(nIterations, enable_lwprof, descramblingOn, 0, rxTxDelayUs, txCpuIds, schdPolicies, txThrdPrios,
                              txGpuIds, inputFilename, outputFilename, fp16Mode);
            pdschTxTest.WaitForCompletion();
            pdschTxTest.DisplayMetrics(debug, false, false);
        }; // auto pdschTxTestExelwte 

        auto puschRxTestExelwte = [&puschRxTest,
                                   &nIterations,
                                   &enable_lwprof,
                                   &descramblingOn,
                                   &rxTxDelayUs,
                                   &rxCpuIds,
                                   &schdPolicies,
                                   &rxThrdPrios,
                                   &rxGpuIds,
                                   &inputFilename,
                                   &outputFilename,
                                   &fp16Mode,
                                   &debug]()
        {
             puschRxTest.Setup(nIterations, enable_lwprof, descramblingOn, 0, 0, rxCpuIds, schdPolicies, rxThrdPrios,
                               rxGpuIds, inputFilename, outputFilename, fp16Mode);
             puschRxTest.WaitForCompletion();
             puschRxTest.DisplayMetrics(debug, false, false);
        }; // auto puschRxTestExelwte 
        
        // Start the worker threads in both pipelines and wait for sync point 
        std::thread rxThrd = std::thread(puschRxTestExelwte);
        std::thread txThrd = std::thread(pdschTxTestExelwte);

        // Allow all worker threads to arrive at start sync point
        uint32_t syncPtWaitCnt = 0;
        uint32_t nSyncPtWaits  = nInst*2; // 2 pipelines of nInst each
        while((syncPtWaitCnt = atmSyncPtWaitCnt.load(std::memory_order_seq_cst)) < nSyncPtWaits)
        {
           std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        printf("Run config: GPU Id %d, # of pipelines %d\n", gpuId, nInst);

        // Launch a delay kernel to keep GPU busy until the CPU bursts out kernel launches (for all pipelines)
        gpu_ms_delay(delay_ms, gpuId, delayLwStrm);

        // Release start sync point  
        {
            std::lock_guard<std::mutex> cvStartSyncPtMutexGuard(cvStartSyncPtMutex);
            startSyncPt = true;
        }
        cvStartSyncPt.notify_all();

        rxThrd.join();
        txThrd.join();
#else
        pdschTxTest.Setup(nIterations, enable_lwprof, descramblingOn, 0, rxTxDelayUs, txCpuIds, schdPolicies, txThrdPrios,
                          txGpuIds, inputFilename, outputFilename, fp16Mode);

        puschRxTest.Setup(nIterations, enable_lwprof, descramblingOn, 0, 0, rxCpuIds, schdPolicies, rxThrdPrios,
                           rxGpuIds, inputFilename, outputFilename, fp16Mode);

        // Allow all worker threads to arrive at start sync point
        uint32_t syncPtWaitCnt = 0;
        uint32_t nSyncPtWaits  = nInst*2; // 2 pipelines of nInst each
        while((syncPtWaitCnt = atmSyncPtWaitCnt.load(std::memory_order_seq_cst)) < nSyncPtWaits)
        {
           std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }


        printf("Run config: GPU Id %d, # of pipelines %d\n", gpuId, nInst);

        // Launch a delay kernel to keep GPU busy until the CPU bursts out kernel launches (for all pipelines)
        gpu_ms_delay(delay_ms, gpuId, delayLwStrm);

        // Release start sync point  
        {
            std::lock_guard<std::mutex> cvStartSyncPtMutexGuard(cvStartSyncPtMutex);
            startSyncPt = true;
        }
        cvStartSyncPt.notify_all();


        pdschTxTest.WaitForCompletion();
        puschRxTest.WaitForCompletion();
        pdschTxTest.DisplayMetrics(debug, false, false);
        puschRxTest.DisplayMetrics(debug, false, false);
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
