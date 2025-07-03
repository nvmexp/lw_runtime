/*
 * Copyright 1993-2013 LWPU Corporation.  All rights reserved.
 *
 * Please refer to the LWPU end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#include <cstdio>
#include <vector>
#include <omp.h>
#include <unistd.h>
#include "lwca.h"
#include "lwda_runtime.h"
#include "BusGrindMain.h"
#include <sstream>
#include <errno.h>
#include "common.h"
#include "PluginCommon.h"
#include "timelib.h"

#ifdef __x86_64__
__asm__(".symver memcpy,memcpy@GLIBC_2.2.5");
#endif

using namespace std;

/*****************************************************************************/
#ifndef MAX
#define MAX(a,b) ((a)>(b) ? (a) : (b))
#endif

/*****************************************************************************/
/* For now, use a heap struct */
BusGrindGlobals g_bgGlobals;

/*****************************************************************************/

/* 
 * Macro for checking lwca errors following a lwca launch or api call 
 * (Avoid using this macro directly, use the lwdaCheckError* macros where possible)
 * **IMPORTANT**: gpuIndex is the index of the gpu in the bgGlobals->gpu vector 
 * 
 * Note: Lwrrently this macro sets the result of the plugin to failed for all GPUs. This is to maintain existing 
 * behavior (a lwca failure always resulted in the test being stopped and all GPUs being marked as failing the test).
 */
#define BG_lwdaCheckError(callName, args, mask, gpuIndex, isGpuSpecific)                    \
    do                                                                                      \
    {                                                                                       \
        lwdaError_t e = callName args;                                                      \
        if (e != lwdaSuccess)                                                               \
        {                                                                                   \
            if (mask)                                                                       \
            {                                                                               \
                lwvsCommon.errorMask |= mask;                                               \
            }                                                                               \
            if (isGpuSpecific)                                                              \
            {                                                                               \
                unsigned int gpuId = bgGlobals->gpu[gpuIndex]->dcgmDeviceIndex;             \
                LOG_LWDA_ERROR_FOR_PLUGIN(bgGlobals->busGrind, #callName, e, gpuId);        \
            }                                                                               \
            else                                                                            \
            {                                                                               \
                LOG_LWDA_ERROR_FOR_PLUGIN(bgGlobals->busGrind, #callName, e, 0, 0, false);  \
            }                                                                               \
            bgGlobals->busGrind->SetResult(LWVS_RESULT_FAIL);                               \
            return -1;                                                                      \
        }                                                                                   \
    } while (0)

// Macros for checking lwca errors following a lwca launch or api call
#define lwdaCheckError(callName, args, mask, gpuIndex) BG_lwdaCheckError(callName, args, mask, gpuIndex, true)
#define lwdaCheckErrorGeneral(callName, args, mask) BG_lwdaCheckError(callName, args, mask, 0, false)

//Macro for checking lwca errors following a lwca launch or api call from an OMP pragma
//we need separate code for this since you are not allowed to exit from OMP
#define lwdaCheckErrorOmp(callName, args, mask, gpuIndex)                                   \
    do                                                                                      \
    {                                                                                       \
        lwdaError_t e = callName args;                                                      \
        if (e != lwdaSuccess)                                                               \
        {                                                                                   \
            if (mask)                                                                       \
            {                                                                               \
                lwvsCommon.errorMask |= mask;                                               \
            }                                                                               \
            unsigned int gpuId = bgGlobals->gpu[gpuIndex]->dcgmDeviceIndex;                 \
            LOG_LWDA_ERROR_FOR_PLUGIN(bgGlobals->busGrind, #callName, e, gpuId);            \
            bgGlobals->busGrind->SetResultForGpu(gpuId, LWVS_RESULT_FAIL);                  \
        }                                                                                   \
    } while (0)

/*****************************************************************************/
//enables P2P for all GPUs
int enableP2P(BusGrindGlobals *bgGlobals)
{
    int lwdaIndexI, lwdaIndexJ;

    for (size_t i = 0; i < bgGlobals->gpu.size(); i++)
    {
        lwdaIndexI = bgGlobals->gpu[i]->lwdaDeviceIdx;
        lwdaSetDevice(lwdaIndexI);

        for (size_t j = 0; j < bgGlobals->gpu.size(); j++)
        {
            int access;
            lwdaIndexJ = bgGlobals->gpu[j]->lwdaDeviceIdx;
            lwdaDeviceCanAccessPeer(&access, lwdaIndexI, lwdaIndexJ);

            if (access)
            {
                lwdaCheckError(lwdaDeviceEnablePeerAccess, (lwdaIndexJ, 0), BG_ERR_PEER_ACCESS_DENIED, i);
            }
        }
    }
    return 0;
}

/*****************************************************************************/
// disables P2P for all GPUs
void disableP2P(BusGrindGlobals *bgGlobals)
{
    int lwdaIndexI, lwdaIndexJ;
    lwdaError_t lwdaReturn;

    for (size_t i=0; i<bgGlobals->gpu.size(); i++)
    {
        lwdaIndexI = bgGlobals->gpu[i]->lwdaDeviceIdx;
        lwdaSetDevice(lwdaIndexI);

        for (size_t j=0; j<bgGlobals->gpu.size(); j++)
        {
            lwdaIndexJ = bgGlobals->gpu[j]->lwdaDeviceIdx;

            lwdaDeviceDisablePeerAccess(lwdaIndexJ);
            //Check for errors and clear any error that may have oclwrred if P2P is not supported
            lwdaReturn = lwdaGetLastError();

            // Note: If the error returned is lwdaErrorPeerAccessNotEnabled,
            // then do not print a message in the console. We are trying to disable peer addressing
            // when it has not yet been enabled.
            //
            // Keep the console clean, and log an error message to the debug file.

            if (lwdaErrorPeerAccessNotEnabled == lwdaReturn)
            {
                PRINT_INFO("%d %d %s", "lwdaDeviceDisablePeerAccess for device (%d) returned error (%d): %s \n",
                        bgGlobals->gpu[j]->lwmlDeviceIndex, (int)lwdaReturn, lwdaGetErrorString(lwdaReturn));
            }
            else if (lwdaSuccess != lwdaReturn)
            {
                std::stringstream ss;
                ss << "lwdaDeviceDisablePeerAccess returned error " << lwdaGetErrorString(lwdaReturn) 
                   << " for device " << bgGlobals->gpu[j]->lwmlDeviceIndex << std::endl;
                bgGlobals->busGrind->AddInfoVerboseForGpu(bgGlobals->gpu[j]->dcgmDeviceIndex, ss.str());
            }
        }
    }
}

/*****************************************************************************/
//outputs latency information to addInfo for verbose reporting
void addLatencyInfo(BusGrindGlobals *bgGlobals, unsigned int gpu, std::string key, double latency)
{
    std::stringstream ss;
    ss.setf(std::ios::fixed, std::ios::floatfield);
    ss.precision(3);

    ss << "GPU " << gpu << " ";
    if (key == "bidir")
        ss << "bidirectional latency:" << "\t\t";
    else if (key == "d2h")
        ss << "GPU to Host latency:" << "\t\t";
    else if (key == "h2d")
        ss << "Host to GPU latency:" << "\t\t";
    ss << latency << " us";

    bgGlobals->busGrind->AddInfoVerboseForGpu(gpu, ss.str());
}


/*****************************************************************************/
//outputs bandwidth information to addInfo for verbose reporting
void addBandwidthInfo(BusGrindGlobals *bgGlobals, unsigned int gpu, std::string key, double bandwidth)
{
    std::stringstream ss;
    ss.setf(std::ios::fixed, std::ios::floatfield);
    ss.precision(2);

    ss << "GPU " << gpu << " ";
    if (key == "bidir")
        ss << "bidirectional bandwidth:" << "\t";
    else if (key == "d2h")
        ss << "GPU to Host bandwidth:" << "\t\t";
    else if (key == "h2d")
        ss << "Host to GPU bandwidth:" << "\t\t";
    ss << bandwidth << " GB/s";

    bgGlobals->busGrind->AddInfoVerboseForGpu(gpu, ss.str());
}

/*****************************************************************************/
int bg_check_pci_link(BusGrindGlobals *bgGlobals, std::string subTest)
{
    int minPcieLinkGen = (int)bgGlobals->testParameters->GetSubTestDouble(subTest, BG_STR_MIN_PCI_GEN);
    int minPcieLinkWidth = (int)bgGlobals->testParameters->GetSubTestDouble(subTest, BG_STR_MIN_PCI_WIDTH);
    int Nfailed = 0;
    lwmlReturn_t lwmlReturn;
    lwmlPstates_t pstateBefore, pstateAfter;
    char errorBuf[256] = {0};
    unsigned int lwrrLinkGen = 0, lwrrLinkWidth = 0;

    for (size_t gpuIdx = 0; gpuIdx < bgGlobals->gpu.size(); gpuIdx++)
    {
        PluginDevice *gpu = bgGlobals->gpu[gpuIdx];

        /* First verify we are at P0 so that it's even valid to check PCI version */
        lwmlReturn = lwmlDeviceGetPerformanceState(gpu->lwmlDevice, &pstateBefore);
        if (lwmlReturn != LWML_SUCCESS)
        {
            PRINT_ERROR("%d %u", "lwmlDeviceGetPerformanceState returned unexpected %d for GPU %u (BEFORE)",
                        (int)lwmlReturn, gpu->lwmlDeviceIndex);
            continue;
        }

        if (pstateBefore != LWML_PSTATE_0)
        {
            snprintf(errorBuf, sizeof(errorBuf), "Skipping PCI-E link check for GPU %u in pstate %d\n",
                     gpu->lwmlDeviceIndex, (int)pstateBefore);
            PRINT_WARNING("%s", "%s", errorBuf);
            bgGlobals->busGrind->AddInfoVerboseForGpu(gpu->dcgmDeviceIndex, errorBuf);
            continue;
        }

        /* Read the link generation */
        lwmlReturn = lwmlDeviceGetLwrrPcieLinkGeneration(gpu->lwmlDevice, &lwrrLinkGen);
        if (lwmlReturn != LWML_SUCCESS)
        {
            PRINT_WARNING("%d %u", "lwmlDeviceGetLwrrPcieLinkGeneration returned %d for GPU %u\n",
                          (int)lwmlReturn, lwrrLinkGen);
            continue;
        }

        /* Read the link width now */
        lwmlReturn = lwmlDeviceGetLwrrPcieLinkWidth(gpu->lwmlDevice, &lwrrLinkWidth);
        if (lwmlReturn != LWML_SUCCESS)
        {
            PRINT_WARNING("%d %u", "lwmlDeviceGetLwrrPcieLinkWidth returned %d for GPU %u\n",
                          (int)lwmlReturn, lwrrLinkGen);
            continue;
        }

        /* Verify we are still in P0 after or the link width and generation aren't valid */
        lwmlReturn = lwmlDeviceGetPerformanceState(gpu->lwmlDevice, &pstateAfter);
        if (lwmlReturn != LWML_SUCCESS)
        {
            PRINT_ERROR("%d %u", "lwmlDeviceGetPerformanceState returned unexpected %d for GPU %u (AFTER)",
                        (int)lwmlReturn, gpu->lwmlDeviceIndex);
            continue;
        }

        if (pstateAfter != LWML_PSTATE_0)
        {
            snprintf(errorBuf, sizeof(errorBuf), "Skipping PCI-E link check for GPU %u in pstate %d (was previously %d)\n",
                     gpu->lwmlDeviceIndex, (int)pstateBefore, (int)pstateAfter);
            PRINT_WARNING("%s", "%s", errorBuf);
            bgGlobals->busGrind->AddInfoVerboseForGpu(gpu->dcgmDeviceIndex, errorBuf);
            continue;
        }

        char buf[512];
        snprintf(buf, sizeof(buf), "%s.%s", subTest.c_str(), BG_STR_MIN_PCI_GEN);

        bgGlobals->busGrind->RecordObservedMetric(gpu->lwmlDeviceIndex, buf, lwrrLinkGen);

        /* Now check the link generation we read */
        if (lwrrLinkGen < minPcieLinkGen)
        {
            DcgmError d;
            DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_PCIE_GENERATION, d, gpu->lwmlDeviceIndex, lwrrLinkGen,
                                      minPcieLinkGen, BG_STR_MIN_PCI_GEN);
            PRINT_ERROR("%s", "%s", d.GetMessage().c_str());
            bgGlobals->busGrind->AddErrorForGpu(gpu->dcgmDeviceIndex, d);
            bgGlobals->busGrind->SetResultForGpu(gpu->dcgmDeviceIndex, LWVS_RESULT_FAIL);
            Nfailed++;
        }

        /* And check the link width we read */
        if (lwrrLinkWidth < minPcieLinkWidth)
        {
            DcgmError d;
            DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_PCIE_WIDTH, d, gpu->lwmlDeviceIndex, lwrrLinkWidth,
                                      minPcieLinkWidth, BG_STR_MIN_PCI_WIDTH);
            PRINT_ERROR("%s", "%s", d.GetMessage().c_str());
            bgGlobals->busGrind->AddErrorForGpu(gpu->dcgmDeviceIndex, d);
            bgGlobals->busGrind->SetResultForGpu(gpu->dcgmDeviceIndex, LWVS_RESULT_FAIL);
            Nfailed++;
        }
    }

    if (Nfailed > 0)
    {
        return 1;
    }
    else
    {
        return 0;
    }
}

/*****************************************************************************/
//this test measures the bus bandwidth between the host and each GPU one at a time
//inputs:
//        int numGPUs:  The number of GPUs to test
//        bool pinned:   Indicates if the host memory should be pinned or not
int outputHostDeviceBandwidthMatrix(BusGrindGlobals *bgGlobals, bool pinned)
{
    vector<int *> d_buffers(bgGlobals->gpu.size());
    int *h_buffer = 0;
    vector<lwdaEvent_t> start(bgGlobals->gpu.size());
    vector<lwdaEvent_t> stop(bgGlobals->gpu.size());
    vector<lwdaStream_t> stream1(bgGlobals->gpu.size());
    vector<lwdaStream_t> stream2(bgGlobals->gpu.size());
    float time_ms;
    double time_s;
    double gb;
    std::string key;
    std::string groupName = "";

    /* Initialize buffers to make valgrind happy */
    for (size_t i = 0; i < bgGlobals->gpu.size(); i++)
    {
        d_buffers[i] = 0;
        stream1[i] = 0;
        stream2[i] = 0;
    }

    if (pinned)
    {
        groupName = BG_SUBTEST_H2D_D2H_SINGLE_PINNED;
    }
    else
    {
        groupName = BG_SUBTEST_H2D_D2H_SINGLE_UNPINNED;
    }

    int numElems = (int)bgGlobals->testParameters->GetSubTestDouble(groupName,
                                                                    BG_STR_INTS_PER_COPY);
    int repeat= (int)bgGlobals->testParameters->GetSubTestDouble(groupName,
                                                                 BG_STR_ITERATIONS);

    if (pinned)
    {
        lwdaCheckErrorGeneral(lwdaMallocHost, (&h_buffer,numElems*sizeof(int)), BG_ERR_LWDA_ALLOC_FAIL);
    }
    else
    {
        h_buffer=(int*)malloc(numElems*sizeof(int));
    }

    for (size_t d=0; d<bgGlobals->gpu.size(); d++)
    {
        lwdaSetDevice(bgGlobals->gpu[d]->lwdaDeviceIdx);
        lwdaCheckError(lwdaMalloc, (&d_buffers[d],numElems*sizeof(int)), BG_ERR_LWDA_ALLOC_FAIL, d);
        lwdaCheckError(lwdaEventCreate, (&start[d]), BG_ERR_LWDA_EVENT_FAIL, d);
        lwdaCheckError(lwdaEventCreate, (&stop[d]), BG_ERR_LWDA_EVENT_FAIL, d);
        lwdaCheckError(lwdaStreamCreate, (&stream1[d]), BG_ERR_LWDA_STREAM_FAIL, d);
        lwdaCheckError(lwdaStreamCreate, (&stream2[d]), BG_ERR_LWDA_STREAM_FAIL, d);
    }

    vector<double> bandwidthMatrix(6*bgGlobals->gpu.size());

    for (size_t i = 0; i < bgGlobals->gpu.size(); i++)
    {
        lwdaSetDevice(bgGlobals->gpu[i]->lwdaDeviceIdx);

        //D2H bandwidth test
        lwdaCheckError(lwdaDeviceSynchronize, (), BG_ERR_LWDA_SYNC_FAIL, i);
        lwdaEventRecord(start[i]);

        for (int r = 0; r < repeat; r++)
        {
            lwdaMemcpyAsync(h_buffer,d_buffers[i],sizeof(int)*numElems,lwdaMemcpyDeviceToHost);
        }

        lwdaEventRecord(stop[i]);
        lwdaCheckError(lwdaDeviceSynchronize, (), BG_ERR_LWDA_SYNC_FAIL, i);

        lwdaEventElapsedTime(&time_ms,start[i],stop[i]);
        time_s=time_ms/1e3;

        gb=numElems*sizeof(int)*repeat/(double)1e9;
        bandwidthMatrix[0*bgGlobals->gpu.size()+i]=gb/time_s;

        //H2D bandwidth test
        lwdaCheckError(lwdaDeviceSynchronize, (), BG_ERR_LWDA_SYNC_FAIL, i);
        lwdaEventRecord(start[i]);

        for (int r=0; r<repeat; r++)
        {
            lwdaMemcpyAsync(d_buffers[i],h_buffer,sizeof(int)*numElems,lwdaMemcpyHostToDevice);
        }
        lwdaEventRecord(stop[i]);
        lwdaCheckError(lwdaDeviceSynchronize, (), BG_ERR_LWDA_SYNC_FAIL, i);

        lwdaEventElapsedTime(&time_ms,start[i],stop[i]);
        time_s=time_ms/1e3;

        gb=numElems*sizeof(int)*repeat/(double)1e9;
        bandwidthMatrix[1*bgGlobals->gpu.size()+i]=gb/time_s;

        //Bidirectional
        lwdaCheckError(lwdaDeviceSynchronize, (), BG_ERR_LWDA_SYNC_FAIL, i);
        lwdaEventRecord(start[i]);

        for (int r=0; r<repeat; r++)
        {
            lwdaMemcpyAsync(d_buffers[i],h_buffer,sizeof(int)*numElems,lwdaMemcpyHostToDevice,stream1[i]);
            lwdaMemcpyAsync(h_buffer,d_buffers[i],sizeof(int)*numElems,lwdaMemcpyDeviceToHost,stream2[i]);
        }

        lwdaEventRecord(stop[i]);
        lwdaCheckError(lwdaDeviceSynchronize, (), BG_ERR_LWDA_SYNC_FAIL, i);

        lwdaEventElapsedTime(&time_ms,start[i],stop[i]);
        time_s=time_ms/1e3;

        gb=2*numElems*sizeof(int)*repeat/(double)1e9;
        bandwidthMatrix[2*bgGlobals->gpu.size()+i]=gb/time_s;
    }

    char labels[][20]={"d2h","h2d","bidir"};
    std::stringstream ss;
    int failedTests = 0;
    double bandwidth;
    double minimumBandwidth = bgGlobals->testParameters->GetSubTestDouble(groupName, BG_STR_MIN_BANDWIDTH);
    char   statNameBuf[512];

    for (int i = 0; i < 3; i++)
    {
        for (size_t j = 0; j < bgGlobals->gpu.size(); j++)
        {
            ss.str("");
            ss << bgGlobals->gpu[j]->lwmlDeviceIndex;
            ss << "_";
            ss << labels[i];
            key = ss.str();

            bandwidth = bandwidthMatrix[(i*bgGlobals->gpu.size())+j];

            bgGlobals->m_dcgmRecorder->SetGroupedStat(groupName, key, bandwidth);
            if (pinned)
                addBandwidthInfo(bgGlobals, bgGlobals->gpu[j]->lwmlDeviceIndex, labels[i], bandwidth);

            snprintf(statNameBuf, sizeof(statNameBuf), "%s.%s-%s", groupName.c_str(), BG_STR_MIN_BANDWIDTH, labels[i]);
            bgGlobals->busGrind->RecordObservedMetric(bgGlobals->gpu[j]->dcgmDeviceIndex, statNameBuf, bandwidth);

            if (bandwidth < minimumBandwidth)
            {
                DcgmError d;
                DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_LOW_BANDWIDTH, d, bgGlobals->gpu[j]->lwmlDeviceIndex,
                                          labels[i], bandwidth, minimumBandwidth);
                bgGlobals->busGrind->AddErrorForGpu(bgGlobals->gpu[j]->dcgmDeviceIndex, d);
                lwvsCommon.errorMask |= BG_ERR_BW_TOO_LOW;
                PRINT_ERROR("%s", "%s", d.GetMessage().c_str());
                bgGlobals->busGrind->SetResultForGpu(bgGlobals->gpu[j]->dcgmDeviceIndex, LWVS_RESULT_FAIL);
                failedTests++;
            }
        }
    }

    /* Check our PCI link status after we've done some work on the link above */
    if (bg_check_pci_link(bgGlobals, groupName))
    {
        failedTests++;
    }

    if (pinned)
    {
        lwdaFreeHost(h_buffer);
    }
    else
    {
        free(h_buffer);
    }

    for (size_t d = 0; d < bgGlobals->gpu.size(); d++)
    {
        lwdaSetDevice(bgGlobals->gpu[d]->lwdaDeviceIdx);
        lwdaCheckError(lwdaFree, (d_buffers[d]), BG_ERR_LWDA_ALLOC_FAIL, d);
        lwdaCheckError(lwdaEventDestroy, (start[d]), BG_ERR_LWDA_EVENT_FAIL, d);
        lwdaCheckError(lwdaEventDestroy, (stop[d]), BG_ERR_LWDA_EVENT_FAIL, d);
        lwdaCheckError(lwdaStreamDestroy, (stream1[d]), BG_ERR_LWDA_STREAM_FAIL, d);
        lwdaCheckError(lwdaStreamDestroy, (stream2[d]), BG_ERR_LWDA_STREAM_FAIL, d);
    }

    if (failedTests > 0)
    {
        return 1;
    }
    else
    {
        return 0;
    }
}

/*****************************************************************************/
//this test measures the bus bandwidth between the host and each GPU conlwrrently
//inputs:
//        int numGPUs:  The number of GPUs to test
//        bool pinned:   Indicates if the host memory should be pinned or not
int outputConlwrrentHostDeviceBandwidthMatrix(BusGrindGlobals *bgGlobals, bool pinned)
{
    vector<int *> buffers(bgGlobals->gpu.size()), d_buffers(bgGlobals->gpu.size());
    vector<lwdaEvent_t> start(bgGlobals->gpu.size());
    vector<lwdaEvent_t> stop(bgGlobals->gpu.size());
    vector<lwdaStream_t> stream1(bgGlobals->gpu.size());
    vector<lwdaStream_t> stream2(bgGlobals->gpu.size());
    vector<double> bandwidthMatrix(3*bgGlobals->gpu.size());

    /* Initialize buffers to make valgrind happy */
    for(size_t i = 0; i < bgGlobals->gpu.size(); i++)
    {
        buffers[i] = 0;
        d_buffers[i] = 0;
        stream1[i] = 0;
        stream2[i] = 0;
    }

    omp_set_num_threads(bgGlobals->gpu.size());

    std::string key;
    std::string groupName;

    if (pinned)
    {
        groupName = BG_SUBTEST_H2D_D2H_CONLWRRENT_PINNED;
    }
    else
    {
        groupName = BG_SUBTEST_H2D_D2H_CONLWRRENT_UNPINNED;
    }

    int numElems = (int)bgGlobals->testParameters->GetSubTestDouble(groupName,
                                                                    BG_STR_INTS_PER_COPY);
    int repeat= (int)bgGlobals->testParameters->GetSubTestDouble(groupName,
                                                                 BG_STR_ITERATIONS);

  //one thread per GPU
#pragma omp parallel
    {
    int d=omp_get_thread_num();
    lwdaSetDevice(bgGlobals->gpu[d]->lwdaDeviceIdx);

    if(pinned)
        lwdaMallocHost(&buffers[d],numElems*sizeof(int));
    else
        buffers[d]=(int*)malloc(numElems*sizeof(int));
    lwdaCheckErrorOmp(lwdaMalloc, (&d_buffers[d],numElems*sizeof(int)), BG_ERR_LWDA_ALLOC_FAIL, d);
    lwdaCheckErrorOmp(lwdaEventCreate, (&start[d]), BG_ERR_LWDA_EVENT_FAIL, d);
    lwdaCheckErrorOmp(lwdaEventCreate, (&stop[d]), BG_ERR_LWDA_EVENT_FAIL, d);
    lwdaCheckErrorOmp(lwdaStreamCreate, (&stream1[d]), BG_ERR_LWDA_STREAM_FAIL, d);
    lwdaCheckErrorOmp(lwdaStreamCreate, (&stream2[d]), BG_ERR_LWDA_STREAM_FAIL, d);

    lwdaDeviceSynchronize();

#pragma omp barrier
    lwdaEventRecord(start[d]);
    //initiate H2D copies
    for (int r=0; r<repeat; r++)
    {
        lwdaMemcpyAsync(d_buffers[d],buffers[d],sizeof(int)*numElems,lwdaMemcpyHostToDevice,stream1[d]);
    }
    lwdaEventRecord(stop[d]);
    lwdaCheckErrorOmp(lwdaDeviceSynchronize, (), BG_ERR_LWDA_SYNC_FAIL, d);

    float time_ms;
    lwdaEventElapsedTime(&time_ms,start[d],stop[d]);
    double time_s=time_ms/1e3;
    double gb=numElems*sizeof(int)*repeat/(double)1e9;

    bandwidthMatrix[0*bgGlobals->gpu.size()+d]=gb/time_s;

    lwdaDeviceSynchronize();
#pragma omp barrier
    lwdaEventRecord(start[d]);
    for (int r=0; r<repeat; r++)
    {
      lwdaMemcpyAsync(buffers[d],d_buffers[d],sizeof(int)*numElems,lwdaMemcpyDeviceToHost,stream1[d]);
    }
    lwdaEventRecord(stop[d]);
    lwdaCheckErrorOmp(lwdaDeviceSynchronize, (), BG_ERR_LWDA_SYNC_FAIL, d);

    lwdaEventElapsedTime(&time_ms,start[d],stop[d]);
    time_s=time_ms/1e3;
    gb=numElems*sizeof(int)*repeat/(double)1e9;

    bandwidthMatrix[1*bgGlobals->gpu.size()+d]=gb/time_s;

    lwdaDeviceSynchronize();
#pragma omp barrier
    lwdaEventRecord(start[d]);
    //Bidirectional
    for (int r=0; r<repeat; r++)
    {
      lwdaMemcpyAsync(d_buffers[d],buffers[d],sizeof(int)*numElems,lwdaMemcpyHostToDevice,stream1[d]);
      lwdaMemcpyAsync(buffers[d],d_buffers[d],sizeof(int)*numElems,lwdaMemcpyDeviceToHost,stream2[d]);
    }

    lwdaEventRecord(stop[d]);
    lwdaCheckErrorOmp(lwdaDeviceSynchronize, (), BG_ERR_LWDA_SYNC_FAIL, d);
#pragma omp barrier

    lwdaEventElapsedTime(&time_ms,start[d],stop[d]);
    time_s=time_ms/1e3;
    gb=2.0*numElems*sizeof(int)*repeat/(double)1e9;

    bandwidthMatrix[2*bgGlobals->gpu.size()+d]=gb/time_s;
    }  //end omp parallel


    char labels[][20]={"h2d","d2h", "bidir"};
    std::stringstream ss;
    int failedTests = 0;
    double bandwidth, minimumBandwidth = bgGlobals->testParameters->GetSubTestDouble(groupName, BG_STR_MIN_BANDWIDTH);

    for (int i=0; i<3; i++)
    {
        double sum=0.0;
        for (size_t j=0; j<bgGlobals->gpu.size(); j++)
        {
            sum+=bandwidthMatrix[i*bgGlobals->gpu.size()+j];
            ss.str("");
            ss << bgGlobals->gpu[j]->lwmlDeviceIndex;
            ss << "_";
            ss << labels[i];
            key = ss.str();

            bandwidth = bandwidthMatrix[i*bgGlobals->gpu.size()+j];
            bgGlobals->m_dcgmRecorder->SetGroupedStat(groupName, key, bandwidth);

            if (bandwidth < minimumBandwidth)
            {
                DcgmError d;
                DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_LOW_BANDWIDTH, d, bgGlobals->gpu[j]->lwmlDeviceIndex,
                                                  labels[i], bandwidth, minimumBandwidth);
                bgGlobals->busGrind->AddErrorForGpu(bgGlobals->gpu[j]->dcgmDeviceIndex, d);
                lwvsCommon.errorMask |= BG_ERR_BW_TOO_LOW;
                PRINT_ERROR("%s", "%s", d.GetMessage().c_str());
                bgGlobals->busGrind->SetResultForGpu(bgGlobals->gpu[j]->dcgmDeviceIndex, LWVS_RESULT_FAIL);
                failedTests++;
            }
        }

        key = "sum_";
        key += labels[i];
        bgGlobals->m_dcgmRecorder->SetGroupedStat(groupName, key, sum);
    }

    for (int d = 0; d < bgGlobals->gpu.size(); d++)
    {
        lwdaSetDevice(bgGlobals->gpu[d]->lwdaDeviceIdx);
        if (pinned)
        {
            lwdaFreeHost(buffers[d]);
        }
        else
        {
            free(buffers[d]);
        }
        lwdaCheckError(lwdaFree, (d_buffers[d]), BG_ERR_LWDA_ALLOC_FAIL, d);
        lwdaCheckError(lwdaEventDestroy, (start[d]), BG_ERR_LWDA_EVENT_FAIL, d);
        lwdaCheckError(lwdaEventDestroy, (stop[d]), BG_ERR_LWDA_EVENT_FAIL, d);
        lwdaCheckError(lwdaStreamDestroy, (stream1[d]), BG_ERR_LWDA_STREAM_FAIL, d);
        lwdaCheckError(lwdaStreamDestroy, (stream2[d]), BG_ERR_LWDA_STREAM_FAIL, d);
    }

    if (failedTests > 0)
    {
        return 1;
    }
    else
    {
        return 0;
    }
}

/*****************************************************************************/
//this test measures the bus latency between the host and each GPU one at a time
//inputs:
//        int numGPUs:  The number of GPUs to test
//        bool pinned:   Indicates if the host memory should be pinned or not
int outputHostDeviceLatencyMatrix(BusGrindGlobals *bgGlobals, bool pinned)
{
    int *h_buffer = 0;
    vector<int *> d_buffers(bgGlobals->gpu.size());
    vector<lwdaEvent_t> start(bgGlobals->gpu.size());
    vector<lwdaEvent_t> stop(bgGlobals->gpu.size());
    vector<lwdaStream_t> stream1(bgGlobals->gpu.size());
    vector<lwdaStream_t> stream2(bgGlobals->gpu.size());
    float time_ms;
    std::string key;
    std::string groupName;

    /* Initialize buffers to make valgrind happy */
    for (size_t i = 0; i < bgGlobals->gpu.size(); i++)
    {
        d_buffers[i] = 0;
        stream1[i] = 0;
        stream2[i] = 0;
    }

    if (pinned)
    {
        groupName = BG_SUBTEST_H2D_D2H_LATENCY_PINNED;
    }
    else
    {
        groupName = BG_SUBTEST_H2D_D2H_LATENCY_UNPINNED;
    }

    int repeat= (int)bgGlobals->testParameters->GetSubTestDouble(groupName, BG_STR_ITERATIONS);

    for (size_t d=0; d<bgGlobals->gpu.size(); d++)
    {
        lwdaSetDevice(bgGlobals->gpu[d]->lwdaDeviceIdx);
        lwdaCheckError(lwdaMalloc, (&d_buffers[d],1), BG_ERR_LWDA_ALLOC_FAIL, d);
        lwdaCheckError(lwdaEventCreate, (&start[d]), BG_ERR_LWDA_EVENT_FAIL, d);
        lwdaCheckError(lwdaEventCreate, (&stop[d]), BG_ERR_LWDA_EVENT_FAIL, d);
        lwdaCheckError(lwdaStreamCreate, (&stream1[d]), BG_ERR_LWDA_STREAM_FAIL, d);
        lwdaCheckError(lwdaStreamCreate, (&stream2[d]), BG_ERR_LWDA_STREAM_FAIL, d);
    }

    if (pinned)
    {
        lwdaCheckErrorGeneral(lwdaMallocHost, (&h_buffer, sizeof(int)), BG_ERR_LWDA_ALLOC_FAIL);
    }
    else
    {
        h_buffer=(int*)malloc(sizeof(int));
    }
    vector<double> latencyMatrix(3*bgGlobals->gpu.size());

    for (size_t i=0; i<bgGlobals->gpu.size(); i++)
    {
        lwdaSetDevice(bgGlobals->gpu[i]->lwdaDeviceIdx);

        //D2H tests
        lwdaCheckError(lwdaDeviceSynchronize, (), BG_ERR_LWDA_SYNC_FAIL, i);
        lwdaEventRecord(start[i]);

        for (int r=0; r<repeat; r++)
        {
            lwdaMemcpyAsync(h_buffer,d_buffers[i],1,lwdaMemcpyDeviceToHost);
        }

        lwdaEventRecord(stop[i]);
        lwdaCheckError(lwdaDeviceSynchronize, (), BG_ERR_LWDA_SYNC_FAIL, i);

        lwdaEventElapsedTime(&time_ms,start[i],stop[i]);

        latencyMatrix[0*bgGlobals->gpu.size()+i]=time_ms*1e3/repeat;

        //H2D tests
        lwdaCheckError(lwdaDeviceSynchronize, (), BG_ERR_LWDA_SYNC_FAIL, i);
        lwdaEventRecord(start[i]);

        for (int r=0; r<repeat; r++)
        {
            lwdaMemcpyAsync(d_buffers[i],h_buffer,1,lwdaMemcpyHostToDevice);
        }

        lwdaEventRecord(stop[i]);
        lwdaCheckError(lwdaDeviceSynchronize, (), BG_ERR_LWDA_SYNC_FAIL, i);

        lwdaEventElapsedTime(&time_ms,start[i],stop[i]);

        latencyMatrix[1*bgGlobals->gpu.size()+i]=time_ms*1e3/repeat;

        //Bidirectional tests
        lwdaCheckError(lwdaDeviceSynchronize, (), BG_ERR_LWDA_SYNC_FAIL, i);
        lwdaEventRecord(start[i]);

        for (int r=0; r<repeat; r++)
        {
            lwdaMemcpyAsync(d_buffers[i],h_buffer,1,lwdaMemcpyHostToDevice,stream1[i]);
            lwdaMemcpyAsync(h_buffer,d_buffers[i],1,lwdaMemcpyDeviceToHost,stream2[i]);
        }

        lwdaEventRecord(stop[i]);
        lwdaCheckError(lwdaDeviceSynchronize, (), BG_ERR_LWDA_SYNC_FAIL, i);

        lwdaEventElapsedTime(&time_ms,start[i],stop[i]);

        latencyMatrix[2*bgGlobals->gpu.size()+i]=time_ms*1e3/repeat;
    }

    char labels[][20]={"d2h","h2d","bidir"};
    std::stringstream ss;
    double maxLatency = bgGlobals->testParameters->GetSubTestDouble(groupName, BG_STR_MAX_LATENCY);
    double latency;
    std::string errorString;
    int Nfailures = 0;
    char statNameBuf[512];

    for (int i=0; i<3; i++)
    {
        for (size_t j=0; j<bgGlobals->gpu.size(); j++)
        {
            ss.str("");
            ss << bgGlobals->gpu[j]->lwmlDeviceIndex;
            ss << "_";
            ss << labels[i];
            key = ss.str();

            latency = latencyMatrix[i*bgGlobals->gpu.size()+j];

            bgGlobals->m_dcgmRecorder->SetGroupedStat(groupName, key, latency);
            if (pinned)
                addLatencyInfo(bgGlobals, bgGlobals->gpu[j]->lwmlDeviceIndex, labels[i], latency);
            
            snprintf(statNameBuf, sizeof(statNameBuf), "%s.%s-%s", groupName.c_str(), BG_STR_MIN_BANDWIDTH, labels[i]);
            bgGlobals->busGrind->RecordObservedMetric(bgGlobals->gpu[j]->dcgmDeviceIndex, statNameBuf, latency);

            if (latency > maxLatency)
            {
                DcgmError d;
                DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_HIGH_LATENCY, d, labels[i],
                                                  bgGlobals->gpu[j]->lwmlDeviceIndex, latency, maxLatency);
                lwvsCommon.errorMask |= BG_ERR_LATENCY_TOO_HIGH;
                PRINT_ERROR("%s", "%s", d.GetMessage().c_str());
                bgGlobals->busGrind->AddErrorForGpu(bgGlobals->gpu[j]->dcgmDeviceIndex, d);
                bgGlobals->busGrind->SetResultForGpu(bgGlobals->gpu[j]->dcgmDeviceIndex, LWVS_RESULT_FAIL);
                Nfailures++;
            }
        }
    }


    if (pinned)
    {
        lwdaFreeHost(h_buffer);
    }
    else
    {
        free(h_buffer);
    }

    for (size_t d = 0; d < bgGlobals->gpu.size(); d++)
    {
        lwdaSetDevice(bgGlobals->gpu[d]->lwdaDeviceIdx);
        lwdaCheckError(lwdaFree, (d_buffers[d]), BG_ERR_LWDA_ALLOC_FAIL, d);
        lwdaCheckError(lwdaEventDestroy, (start[d]), BG_ERR_LWDA_EVENT_FAIL, d);
        lwdaCheckError(lwdaEventDestroy, (stop[d]), BG_ERR_LWDA_EVENT_FAIL, d);
        lwdaCheckError(lwdaStreamDestroy, (stream1[d]), BG_ERR_LWDA_STREAM_FAIL, d);
        lwdaCheckError(lwdaStreamDestroy, (stream2[d]), BG_ERR_LWDA_STREAM_FAIL, d);
    }

    if (Nfailures > 0)
    {
        return 1;
    }
    else
    {
        return 0;
    }
}

/*****************************************************************************/
//This test measures the bus bandwidth between pairs of GPUs one at a time
//inputs:
//        int numGPUs:  The number of GPUs to test
//        bool p2p:   Indicates if GPUDirect P2P should be enabled or not
int outputP2PBandwidthMatrix(BusGrindGlobals *bgGlobals, bool p2p)
{
    vector<int *> buffers(bgGlobals->gpu.size());
    vector<lwdaEvent_t> start(bgGlobals->gpu.size());
    vector<lwdaEvent_t> stop(bgGlobals->gpu.size());
    vector<lwdaStream_t> stream0(bgGlobals->gpu.size());
    vector<lwdaStream_t> stream1(bgGlobals->gpu.size());
    std::string key;
    std::string groupName;

    /* Initialize buffers to make valgrind happy */
    for (size_t i = 0; i < bgGlobals->gpu.size(); i++)
    {
        buffers[i] = 0;
        stream0[i] = 0;
        stream1[i] = 0;
    }

    if (p2p)
    {
        groupName = BG_SUBTEST_P2P_BW_P2P_ENABLED;
    }
    else
    {
        groupName = BG_SUBTEST_P2P_BW_P2P_DISABLED;
    }

    int numElems = (int)bgGlobals->testParameters->GetSubTestDouble(groupName, BG_STR_INTS_PER_COPY);
    int repeat= (int)bgGlobals->testParameters->GetSubTestDouble(groupName, BG_STR_ITERATIONS);

    for (size_t d=0; d<bgGlobals->gpu.size(); d++)
    {
        lwdaSetDevice(bgGlobals->gpu[d]->lwdaDeviceIdx);
        lwdaCheckError(lwdaMalloc, (&buffers[d],numElems*sizeof(int)), BG_ERR_LWDA_ALLOC_FAIL, d);
        lwdaCheckError(lwdaEventCreate, (&start[d]), BG_ERR_LWDA_EVENT_FAIL, d);
        lwdaCheckError(lwdaEventCreate, (&stop[d]), BG_ERR_LWDA_EVENT_FAIL, d);
        lwdaCheckError(lwdaStreamCreate, (&stream0[d]), BG_ERR_LWDA_STREAM_FAIL, d);
        lwdaCheckError(lwdaStreamCreate, (&stream1[d]), BG_ERR_LWDA_STREAM_FAIL, d);
    }

    vector<double> bandwidthMatrix(bgGlobals->gpu.size()*bgGlobals->gpu.size());

    if (p2p)
    {
        enableP2P(bgGlobals);
    }

    //for each device
    for (size_t i = 0; i < bgGlobals->gpu.size(); i++)
    {
        lwdaSetDevice(bgGlobals->gpu[i]->lwdaDeviceIdx);

        //for each device
        for (size_t j = 0; j < bgGlobals->gpu.size(); j++)
        {
            //measure bandwidth between device i and device j
            lwdaCheckError(lwdaDeviceSynchronize, (), BG_ERR_LWDA_SYNC_FAIL, i);
            lwdaEventRecord(start[i]);

            for (int r=0; r<repeat; r++)
            {
                lwdaMemcpyPeerAsync(buffers[i],bgGlobals->gpu[i]->lwdaDeviceIdx,buffers[j],
                                    bgGlobals->gpu[j]->lwdaDeviceIdx,sizeof(int)*numElems);
            }

            lwdaEventRecord(stop[i]);
            lwdaCheckError(lwdaDeviceSynchronize, (), BG_ERR_LWDA_SYNC_FAIL, i);

            float time_ms;
            lwdaEventElapsedTime(&time_ms,start[i],stop[i]);
            double time_s=time_ms/1e3;

            double gb=numElems*sizeof(int)*repeat/(double)1e9;
            bandwidthMatrix[i*bgGlobals->gpu.size()+j]=gb/time_s;
        }
    }

    std::stringstream ss;

    for (size_t i=0; i<bgGlobals->gpu.size(); i++)
    {
        for (size_t j=0; j<bgGlobals->gpu.size(); j++)
        {
            ss.str("");
            ss << bgGlobals->gpu[i]->lwmlDeviceIndex;
            ss << "_";
            ss << bgGlobals->gpu[j]->lwmlDeviceIndex;
            ss << "_onedir";
            key = ss.str();

            bgGlobals->m_dcgmRecorder->SetGroupedStat(groupName, key, bandwidthMatrix[i*bgGlobals->gpu.size()+j]);
        }
    }

    for (size_t i = 0; i < bgGlobals->gpu.size(); i++)
    {
        lwdaSetDevice(bgGlobals->gpu[i]->lwdaDeviceIdx);

        for (size_t j = 0; j < bgGlobals->gpu.size(); j++)
        {
            lwdaCheckError(lwdaDeviceSynchronize, (), BG_ERR_LWDA_SYNC_FAIL, i);
            lwdaEventRecord(start[i]);

            for (int r=0; r<repeat; r++)
            {
                lwdaMemcpyPeerAsync(buffers[i],bgGlobals->gpu[i]->lwdaDeviceIdx,buffers[j],
                        bgGlobals->gpu[j]->lwdaDeviceIdx,sizeof(int)*numElems,stream0[i]);
                lwdaMemcpyPeerAsync(buffers[j],bgGlobals->gpu[j]->lwdaDeviceIdx,buffers[i],
                        bgGlobals->gpu[i]->lwdaDeviceIdx,sizeof(int)*numElems,stream1[i]);
            }

            lwdaEventRecord(stop[i]);
            lwdaCheckError(lwdaDeviceSynchronize, (), BG_ERR_LWDA_SYNC_FAIL, i);

            float time_ms;
            lwdaEventElapsedTime(&time_ms,start[i],stop[i]);
            double time_s=time_ms/1e3;

            double gb=2.0*numElems*sizeof(int)*repeat/(double)1e9;
            bandwidthMatrix[i*bgGlobals->gpu.size()+j]=gb/time_s;
        }
    }

    for (size_t i=0; i<bgGlobals->gpu.size(); i++)
    {
        for (size_t j=0; j<bgGlobals->gpu.size(); j++)
        {
            ss.str("");
            ss << bgGlobals->gpu[i]->lwmlDeviceIndex;
            ss << "_";
            ss << bgGlobals->gpu[j]->lwmlDeviceIndex;
            ss << "_bidir";
            key = ss.str();

            bgGlobals->m_dcgmRecorder->SetGroupedStat(groupName, key, bandwidthMatrix[i*bgGlobals->gpu.size()+j]);
        }
    }

    if (p2p)
    {
        disableP2P(bgGlobals);
    }

    for (size_t d = 0; d < bgGlobals->gpu.size(); d++)
    {
        lwdaSetDevice(bgGlobals->gpu[d]->lwdaDeviceIdx);
        lwdaCheckError(lwdaFree, (buffers[d]), BG_ERR_LWDA_ALLOC_FAIL, d);
        lwdaCheckError(lwdaEventDestroy, (start[d]), BG_ERR_LWDA_EVENT_FAIL, d);
        lwdaCheckError(lwdaEventDestroy, (stop[d]), BG_ERR_LWDA_EVENT_FAIL, d);
        lwdaCheckError(lwdaStreamDestroy, (stream0[d]), BG_ERR_LWDA_STREAM_FAIL, d);
        lwdaCheckError(lwdaStreamDestroy, (stream1[d]), BG_ERR_LWDA_STREAM_FAIL, d);
    }

    return 0;
}

/*****************************************************************************/
//This test measures the bus bandwidth between neighboring GPUs conlwrrently.
//Neighbors are defined by device_id/2 being equal, i.e. (0,1), (2,3), etc.
//inputs:
//        int numGPUs:  The number of GPUs to test
//        bool p2p:   Indicates if GPUDirect P2P should be enabled or not
int outputConlwrrentPairsP2PBandwidthMatrix(BusGrindGlobals *bgGlobals, bool p2p)
{
    //only run this test if p2p tests are enabled
    int numGPUs = bgGlobals->gpu.size()/2*2; //round to the neared even number of GPUs
    if (p2p)
    {
        enableP2P(bgGlobals);
    }
    vector<int *> buffers(numGPUs);
    vector<lwdaEvent_t> start(numGPUs);
    vector<lwdaEvent_t> stop(numGPUs);
    vector<lwdaStream_t> stream(numGPUs);
    vector<double> bandwidthMatrix(3*numGPUs/2);
    std::string key;
    std::string groupName;

    if (numGPUs <= 0)
    {
        if (! bgGlobals->m_printedConlwrrentGpuErrorMessage)
        {
            DcgmError d;
            DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_CONLWRRENT_GPUS, d);
            bgGlobals->busGrind->AddError(d);
            bgGlobals->m_printedConlwrrentGpuErrorMessage = true;
        }
        return 0;
    }

    /* Initialize buffers to make valgrind happy */
    for (int i = 0; i < numGPUs; i++)
    {
        buffers[i] = 0;
        stream[i] = 0;
    }

    if (p2p)
    {
        groupName = BG_SUBTEST_P2P_BW_CONLWRRENT_P2P_ENABLED;
    }
    else
    {
        groupName = BG_SUBTEST_P2P_BW_CONLWRRENT_P2P_DISABLED;
    }

    int numElems = (int)bgGlobals->testParameters->GetSubTestDouble(groupName, BG_STR_INTS_PER_COPY);
    int repeat= (int)bgGlobals->testParameters->GetSubTestDouble(groupName, BG_STR_ITERATIONS);

    omp_set_num_threads(numGPUs);
#pragma omp parallel
    {
    int d = omp_get_thread_num();
    lwdaSetDevice(bgGlobals->gpu[d]->lwdaDeviceIdx);
    lwdaCheckErrorOmp(lwdaMalloc, (&buffers[d], numElems*sizeof(int)), BG_ERR_LWDA_ALLOC_FAIL, d);
    lwdaCheckErrorOmp(lwdaEventCreate, (&start[d]), BG_ERR_LWDA_EVENT_FAIL, d);
    lwdaCheckErrorOmp(lwdaEventCreate, (&stop[d]), BG_ERR_LWDA_EVENT_FAIL, d);
    lwdaCheckErrorOmp(lwdaStreamCreate, (&stream[d]), BG_ERR_LWDA_STREAM_FAIL, d);

    lwdaDeviceSynchronize();

#pragma omp barrier
    lwdaEventRecord(start[d],stream[d]);
    //right to left tests
    if(d%2==0)
    {
        for (int r=0; r<repeat; r++)
        {
            lwdaMemcpyPeerAsync(buffers[d+1],bgGlobals->gpu[d+1]->lwdaDeviceIdx,buffers[d],
                                bgGlobals->gpu[d]->lwdaDeviceIdx,sizeof(int)*numElems,stream[d]);
        }
        lwdaEventRecord(stop[d],stream[d]);
        lwdaDeviceSynchronize();

        float time_ms;
        lwdaEventElapsedTime(&time_ms,start[d],stop[d]);
        double time_s=time_ms/1e3;
        double gb=numElems*sizeof(int)*repeat/(double)1e9;

        bandwidthMatrix[0*numGPUs/2+d/2]=gb/time_s;
    }

    lwdaDeviceSynchronize();
#pragma omp barrier
    lwdaEventRecord(start[d],stream[d]);
    //left to right tests
    if(d%2==1)
    {
        for (int r=0; r<repeat; r++)
        {
            lwdaMemcpyPeerAsync(buffers[d-1],bgGlobals->gpu[d-1]->lwdaDeviceIdx, buffers[d],
                                bgGlobals->gpu[d]->lwdaDeviceIdx,sizeof(int)*numElems,stream[d]);
        }
        lwdaEventRecord(stop[d],stream[d]);
        lwdaDeviceSynchronize();

        float time_ms;
        lwdaEventElapsedTime(&time_ms,start[d],stop[d]);
        double time_s=time_ms/1e3;
        double gb=numElems*sizeof(int)*repeat/(double)1e9;

        bandwidthMatrix[1*numGPUs/2+d/2]=gb/time_s;
    }

    lwdaDeviceSynchronize();
#pragma omp barrier
    lwdaEventRecord(start[d],stream[d]);
    //Bidirectional tests
    if(d%2==0)
    {
        for (int r=0; r<repeat; r++)
        {
            lwdaMemcpyPeerAsync(buffers[d+1],bgGlobals->gpu[d+1]->lwdaDeviceIdx,buffers[d],
                                bgGlobals->gpu[d]->lwdaDeviceIdx,sizeof(int)*numElems,stream[d]);
        }
    }
    else
    {
        for (int r=0; r<repeat; r++)
        {
            lwdaMemcpyPeerAsync(buffers[d-1],bgGlobals->gpu[d-1]->lwdaDeviceIdx,buffers[d],
                                bgGlobals->gpu[d]->lwdaDeviceIdx,sizeof(int)*numElems,stream[d]);
        }
    }

    lwdaEventRecord(stop[d],stream[d]);
    lwdaDeviceSynchronize();
#pragma omp barrier

    if(d%2==0)
    {
        float time_ms1, time_ms2;
        lwdaEventElapsedTime(&time_ms1,start[d],stop[d]);
        lwdaEventElapsedTime(&time_ms2,start[d+1],stop[d+1]);
        double time_s=max(time_ms1,time_ms2)/1e3;
        double gb=2.0*numElems*sizeof(int)*repeat/(double)1e9;

        bandwidthMatrix[2*numGPUs/2+d/2]=gb/time_s;
    }

    } //omp parallel

    char labels[][20]={"r2l","l2r", "bidir"};
    std::stringstream ss;

    for (int i=0; i<3; i++)
    {
        double sum=0.0;
        for (int j=0; j<numGPUs/2; j++)
        {
            ss.str("");
            ss << labels[i];
            ss << "_";
            ss << bgGlobals->gpu[j]->lwmlDeviceIndex;
            ss << "_";
            ss << bgGlobals->gpu[j+1]->lwmlDeviceIndex;

            key = ss.str();

            sum+=bandwidthMatrix[i*numGPUs/2+j];

            bgGlobals->m_dcgmRecorder->SetGroupedStat(groupName, key, bandwidthMatrix[i*numGPUs/2+j]);
        }

        ss.str("");
        ss << labels[i];
        ss << "_sum";
        key = ss.str();
        bgGlobals->m_dcgmRecorder->SetGroupedStat(groupName, key, sum);
    }
    if (p2p)
    {
        disableP2P(bgGlobals);
    }

    for (int d = 0; d < numGPUs; d++)
    {
        lwdaSetDevice(bgGlobals->gpu[d]->lwdaDeviceIdx);
        lwdaCheckError(lwdaFree, (buffers[d]), BG_ERR_LWDA_ALLOC_FAIL, d);
        lwdaCheckError(lwdaEventDestroy, (start[d]), BG_ERR_LWDA_EVENT_FAIL, d);
        lwdaCheckError(lwdaEventDestroy, (stop[d]), BG_ERR_LWDA_EVENT_FAIL, d);
        lwdaCheckError(lwdaStreamDestroy, (stream[d]), BG_ERR_LWDA_STREAM_FAIL, d);
    }

    return 0;
}

/*****************************************************************************/
//This test measures the bus bandwidth for a 1D exchange algorithm with all GPUs transferring conlwrrently.
//L2R: indicates that everyone sends up one device_id
//R2L: indicates that everyone sends down one device_id
//inputs:
//        int numGPUs:  The number of GPUs to test
//        bool p2p:   Indicates if GPUDirect P2P should be enabled or not
int outputConlwrrent1DExchangeBandwidthMatrix(BusGrindGlobals *bgGlobals, bool p2p)
{
    int numGPUs = bgGlobals->gpu.size()/2*2; //round to the neared even number of GPUs
    vector<int *> buffers(numGPUs);
    vector<lwdaEvent_t> start(numGPUs);
    vector<lwdaEvent_t> stop(numGPUs);
    vector<lwdaStream_t> stream1(numGPUs), stream2(numGPUs);
    vector<double> bandwidthMatrix(3*numGPUs);
    std::string key;
    std::string groupName;

    if (numGPUs <= 0) 
    {
        if (! bgGlobals->m_printedConlwrrentGpuErrorMessage)
        {
            DcgmError d;
            DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_CONLWRRENT_GPUS, d);
            bgGlobals->busGrind->AddError(d);
            bgGlobals->m_printedConlwrrentGpuErrorMessage = true;
        }
        return 0;
    }

    /* Initialize buffers to make valgrind happy */
    for (int i = 0; i < numGPUs; i++)
    {
        buffers[i] = 0;
        stream1[i] = 0;
    }


    if (p2p)
    {
        groupName = BG_SUBTEST_1D_EXCH_BW_P2P_ENABLED;
    }
    else
    {
        groupName = BG_SUBTEST_1D_EXCH_BW_P2P_DISABLED;
    }

    int numElems = (int)bgGlobals->testParameters->GetSubTestDouble(groupName, BG_STR_INTS_PER_COPY);
    int repeat= (int)bgGlobals->testParameters->GetSubTestDouble(groupName, BG_STR_ITERATIONS);

    if (p2p)
    {
        enableP2P(bgGlobals);
    }
    omp_set_num_threads(numGPUs);
#pragma omp parallel
    {
    int d = omp_get_thread_num();
    lwdaSetDevice(bgGlobals->gpu[d]->lwdaDeviceIdx);
    lwdaCheckErrorOmp(lwdaMalloc, (&buffers[d], numElems*sizeof(int)), BG_ERR_LWDA_ALLOC_FAIL, d);
    lwdaCheckErrorOmp(lwdaEventCreate, (&start[d]), BG_ERR_LWDA_EVENT_FAIL, d);
    lwdaCheckErrorOmp(lwdaEventCreate, (&stop[d]), BG_ERR_LWDA_EVENT_FAIL, d);
    lwdaCheckErrorOmp(lwdaStreamCreate, (&stream1[d]), BG_ERR_LWDA_STREAM_FAIL, d);
    lwdaCheckErrorOmp(lwdaStreamCreate, (&stream2[d]), BG_ERR_LWDA_STREAM_FAIL, d);

    lwdaDeviceSynchronize();


#pragma omp barrier
    lwdaEventRecord(start[d],stream1[d]);
    //L2R tests
    for (int r=0; r<repeat; r++)
    {
        if(d+1<numGPUs)
        {
            lwdaMemcpyPeerAsync(buffers[d+1],bgGlobals->gpu[d+1]->lwdaDeviceIdx, buffers[d],
                                bgGlobals->gpu[d]->lwdaDeviceIdx, sizeof(int)*numElems,stream1[d]);
        }
    }
    lwdaEventRecord(stop[d],stream1[d]);
    lwdaDeviceSynchronize();

    float time_ms;
    lwdaEventElapsedTime(&time_ms,start[d],stop[d]);
    double time_s=time_ms/1e3;
    double gb=numElems*sizeof(int)*repeat/(double)1e9;

    if(d==numGPUs-1)
        gb=0;
    bandwidthMatrix[0*numGPUs+d]=gb/time_s;
    lwdaDeviceSynchronize();
#pragma omp barrier
    lwdaEventRecord(start[d],stream1[d]);
    //R2L tests
    for (int r=0; r<repeat; r++)
    {
        if(d>0)
            lwdaMemcpyPeerAsync(buffers[d-1],bgGlobals->gpu[d-1]->lwdaDeviceIdx,buffers[d],
                                bgGlobals->gpu[d]->lwdaDeviceIdx,sizeof(int)*numElems,stream1[d]);
    }
    lwdaEventRecord(stop[d],stream1[d]);
    lwdaDeviceSynchronize();

    lwdaEventElapsedTime(&time_ms,start[d],stop[d]);
    time_s=time_ms/1e3;
    gb=numElems*sizeof(int)*repeat/(double)1e9;

    if(d==0)
        gb=0;

    bandwidthMatrix[1*numGPUs+d]=gb/time_s;

    lwdaDeviceSynchronize();
    } //omp parallel

    char labels[][20]={"r2l","l2r"};
    std::stringstream ss;

    for (int i=0; i<2; i++)
    {
        double sum=0.0;
        for (int j=0; j<numGPUs; j++)
        {
            sum+=bandwidthMatrix[i*numGPUs+j];
            ss.str("");
            ss << labels[i];
            ss << "_";
            ss << bgGlobals->gpu[j]->lwmlDeviceIndex;
            key = ss.str();
            bgGlobals->m_dcgmRecorder->SetGroupedStat(groupName, key, bandwidthMatrix[i*numGPUs+j]);
        }

        ss.str("");
        ss << labels[i];
        ss << "_sum";
        key = ss.str();
        bgGlobals->m_dcgmRecorder->SetGroupedStat(groupName, key, sum);
    }

    for (int d = 0; d < numGPUs; d++)
    {
        lwdaSetDevice(bgGlobals->gpu[d]->lwdaDeviceIdx);
        lwdaCheckError(lwdaFree, (buffers[d]), BG_ERR_LWDA_ALLOC_FAIL, d);
        lwdaCheckError(lwdaEventDestroy, (start[d]), BG_ERR_LWDA_EVENT_FAIL, d);
        lwdaCheckError(lwdaEventDestroy, (stop[d]), BG_ERR_LWDA_EVENT_FAIL, d);
        lwdaCheckError(lwdaStreamDestroy, (stream1[d]), BG_ERR_LWDA_STREAM_FAIL, d);
        lwdaCheckError(lwdaStreamDestroy, (stream2[d]), BG_ERR_LWDA_STREAM_FAIL, d);
    }

    if (p2p)
    {
        disableP2P(bgGlobals);
    }

    return 0;
}

/*****************************************************************************/
//This test measures the bus latency between pairs of GPUs one at a time
//inputs:
//        int numGPUs:  The number of GPUs to test
//        bool p2p:   Indicates if GPUDirect P2P should be enabled or not
int outputP2PLatencyMatrix(BusGrindGlobals *bgGlobals, bool p2p)
{
    vector<int *> buffers(bgGlobals->gpu.size());
    vector<lwdaEvent_t> start(bgGlobals->gpu.size());
    vector<lwdaEvent_t> stop(bgGlobals->gpu.size());
    std::string key;
    std::string groupName;

    /* Initialize buffers to make valgrind happy */
    for (size_t i = 0; i < bgGlobals->gpu.size(); i++)
    {
        buffers[i] = 0;
    }

    if (p2p)
    {
        groupName = BG_SUBTEST_P2P_LATENCY_P2P_ENABLED;
    }
    else
    {
        groupName = BG_SUBTEST_P2P_LATENCY_P2P_DISABLED;
    }

    int repeat= (int)bgGlobals->testParameters->GetSubTestDouble(groupName, BG_STR_ITERATIONS);

    if (p2p)
    {
        enableP2P(bgGlobals);
    }

    for (size_t d = 0; d < bgGlobals->gpu.size(); d++)
    {
        lwdaSetDevice(bgGlobals->gpu[d]->lwdaDeviceIdx);
        lwdaCheckError(lwdaMalloc, (&buffers[d],1), BG_ERR_LWDA_ALLOC_FAIL, d);
        lwdaCheckError(lwdaEventCreate, (&start[d]), BG_ERR_LWDA_EVENT_FAIL, d);
        lwdaCheckError(lwdaEventCreate, (&stop[d]), BG_ERR_LWDA_EVENT_FAIL, d);
    }

    vector<double> latencyMatrix(bgGlobals->gpu.size()*bgGlobals->gpu.size());

    for (size_t i = 0; i < bgGlobals->gpu.size(); i++)
    {
        lwdaSetDevice(bgGlobals->gpu[i]->lwdaDeviceIdx);

        for (size_t j = 0; j < bgGlobals->gpu.size(); j++)
        {
            lwdaCheckError(lwdaDeviceSynchronize, (), BG_ERR_LWDA_SYNC_FAIL, i);
            lwdaEventRecord(start[i]);

            for (int r = 0; r < repeat; r++)
            {
                lwdaMemcpyPeerAsync(buffers[i],bgGlobals->gpu[i]->lwdaDeviceIdx,buffers[j],
                                    bgGlobals->gpu[j]->lwdaDeviceIdx,1);
            }

            lwdaEventRecord(stop[i]);
            lwdaCheckError(lwdaDeviceSynchronize, (), BG_ERR_LWDA_SYNC_FAIL, i);

            float time_ms;
            lwdaEventElapsedTime(&time_ms,start[i],stop[i]);

            latencyMatrix[i*bgGlobals->gpu.size()+j]=time_ms*1e3/repeat;
        }
    }

    std::stringstream ss;

    for (size_t i=0; i<bgGlobals->gpu.size(); i++)
    {
        for (size_t j=0; j<bgGlobals->gpu.size(); j++)
        {
            ss.str("");
            ss << bgGlobals->gpu[i]->lwmlDeviceIndex;
            ss << "_";
            ss << bgGlobals->gpu[j]->lwmlDeviceIndex;
            key = ss.str();
            bgGlobals->m_dcgmRecorder->SetGroupedStat(groupName, key, latencyMatrix[i*bgGlobals->gpu.size()+j]);

        }
    }
    if (p2p)
    {
        disableP2P(bgGlobals);
    }

    for (size_t d = 0; d < bgGlobals->gpu.size(); d++)
    {
        lwdaSetDevice(bgGlobals->gpu[d]->lwdaDeviceIdx);
        lwdaCheckError(lwdaFree, (buffers[d]), BG_ERR_LWDA_ALLOC_FAIL, d);
        lwdaCheckError(lwdaEventDestroy, (start[d]), BG_ERR_LWDA_EVENT_FAIL, d);
        lwdaCheckError(lwdaEventDestroy, (stop[d]), BG_ERR_LWDA_EVENT_FAIL, d);
    }

    return 0;
}

/*****************************************************************************/
int bg_cache_and_check_parameters(BusGrindGlobals *bgGlobals)
{
    /* Set defaults before we parse parameters */
    bgGlobals->test_pinned = bgGlobals->testParameters->GetBoolFromString(BG_STR_TEST_PINNED);
    bgGlobals->test_unpinned = bgGlobals->testParameters->GetBoolFromString(BG_STR_TEST_UNPINNED);
    bgGlobals->test_p2p_on = bgGlobals->testParameters->GetBoolFromString(BG_STR_TEST_P2P_ON);
    bgGlobals->test_p2p_off = bgGlobals->testParameters->GetBoolFromString(BG_STR_TEST_P2P_OFF);
    return 0;
}

/*****************************************************************************/
void bg_cleanup(BusGrindGlobals *bgGlobals)
{
    PluginDevice *bgGpu;

    bgGlobals->m_printedConlwrrentGpuErrorMessage = false;

    if (bgGlobals->m_dcgmRecorder)
    {
        delete(bgGlobals->m_dcgmRecorder);
        bgGlobals->m_dcgmRecorder = 0;
    }

    for (size_t bgGpuIdx = 0; bgGpuIdx < bgGlobals->gpu.size(); bgGpuIdx++)
    {
        bgGpu = bgGlobals->gpu[bgGpuIdx];
        delete bgGpu;
    }

    bgGlobals->gpu.clear();

    /* Unload our lwca context for each gpu in the current process. We enumerate all GPUs because
       lwca opens a context on all GPUs, even if we don't use them */
    int deviceIdx, lwdaDeviceCount;
    lwdaError_t lwSt;
    lwSt = lwdaGetDeviceCount(&lwdaDeviceCount);
    if (lwSt == lwdaSuccess)
    {
        for (deviceIdx = 0; deviceIdx < lwdaDeviceCount; deviceIdx++)
        {
            lwdaSetDevice(deviceIdx);
            lwdaDeviceReset();
        }
    }

    if (bgGlobals->lwmlInitialized)
    {
        lwmlShutdown();
    }
}

/*****************************************************************************/
int bg_lwml_init(BusGrindGlobals *bgGlobals, const std::vector<unsigned int> &gpuList)
{
    int i, st, gpuListIndex;
    lwmlReturn_t lwmlSt;
    char buildPciStr[32] = {0};
    unsigned int maxPowerUInt;
    char buf[256] = {0};
    lwmlPciInfo_t pciInfo;
    lwdaError_t lwSt;
    lwvsReturn_t lwvsReturn;

    lwmlSt = lwmlInit();
    if(lwmlSt != LWML_SUCCESS)
    {
        fprintf(stderr, "lwmlInit returned %d (%s)\n", lwmlSt,
                lwmlErrorString(lwmlSt));
        DcgmError d;
        DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_LWML_API, d, "lwmlInit", lwmlErrorString(lwmlSt));
        bgGlobals->busGrind->AddError(d);
        PRINT_ERROR("%s", "%s", d.GetMessage().c_str());
        return -1;
    }


    for(gpuListIndex = 0; gpuListIndex < gpuList.size(); gpuListIndex++)
    {
        PluginDevice *pd = 0;
        try
        {
            pd = new PluginDevice(gpuList[gpuListIndex], bgGlobals->busGrind);
        }
        catch (DcgmError &d)
        {
            bgGlobals->busGrind->AddErrorForGpu(gpuList[gpuListIndex], d);
            delete pd;
            return(-1);
        }

        if (pd->warning.size() > 0)
        {
            bgGlobals->busGrind->AddInfoVerboseForGpu(gpuList[gpuListIndex], pd->warning);
        }

        /* At this point, we consider this GPU part of our set */
        bgGlobals->gpu.push_back(pd);

        /* Try to set the affinity of the device */
        st = pd->lwvsDevice->SetCpuAffinity();
        /* Failure considered nonfatal */
    }

    return 0;
}

/*****************************************************************************/
void bg_record_cliques(BusGrindGlobals *bgGlobals)
{
    //compute cliques
    //a clique is a group of GPUs that can P2P
    vector<vector<int> > cliques;

    //vector indicating if a GPU has already been processed
    vector<bool> added(bgGlobals->gpu.size(),false);

    for (size_t i = 0; i < bgGlobals->gpu.size(); i++)
    {
        if (added[i] == true)
            continue;         //already processed

        //create new clique with i
        vector<int> clique;
        added[i] = true;
        clique.push_back(i);

        for (size_t j = i + 1; j < bgGlobals->gpu.size(); j++)
        {
            int access;
            lwdaDeviceCanAccessPeer(&access, bgGlobals->gpu[i]->lwdaDeviceIdx, bgGlobals->gpu[j]->lwdaDeviceIdx);

            //if GPU i can acces j then add to current clique
            if (access)
            {
                clique.push_back(j);
                //mark that GPU has been added to a clique
                added[j] = true;
            }
        }

        cliques.push_back(clique);
    }

    std::string p2pGroup("p2p_cliques");
    char buf[64] = {0};
    std::string key(""), temp("");

    /* Write p2p cliques to the stats as "1" => "1 2 3" */
    for (int c = 0; c < (int)cliques.size(); c++)
    {
        snprintf(buf, sizeof(buf), "%d", bgGlobals->gpu[c]->lwmlDeviceIndex);
        key = buf;

        temp = "";
        for (int j = 0; j < (int)cliques[c].size() - 1; j++)
        {
            snprintf(buf, sizeof(buf), "%d ", bgGlobals->gpu[cliques[c][j]]->lwmlDeviceIndex);
            temp += buf;
        }

        snprintf(buf, sizeof(buf), "%d", bgGlobals->gpu[cliques[c][cliques[c].size()-1]]->lwmlDeviceIndex);
        temp += buf;

        bgGlobals->m_dcgmRecorder->SetSingleGroupStat(key, p2pGroup, temp);
    }
}

/*****************************************************************************/
int bg_should_stop(BusGrindGlobals *bgGlobals)
{
    if (!main_should_stop)
    {
        return 0;
    }

    DcgmError d;
    DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_ABORTED, d);
    bgGlobals->busGrind->AddError(d);
    bgGlobals->busGrind->SetResult(LWVS_RESULT_SKIP);
    return 1;
}

/*****************************************************************************/
dcgmReturn_t bg_check_per_second_error_conditions(BusGrindGlobals *bgGlobals, unsigned int gpuId,
                                                  std::vector<DcgmError> &errorList, timelib64_t startTime)
{
    std::vector<unsigned short> fieldIds;
    std::vector<dcgmFieldValue_v1> failureThresholds;
    
    fieldIds.push_back(DCGM_FI_DEV_LWLINK_CRC_FLIT_ERROR_COUNT_TOTAL);
    fieldIds.push_back(DCGM_FI_DEV_LWLINK_CRC_DATA_ERROR_COUNT_TOTAL);
    
    double crcErrorThreshold = bgGlobals->testParameters->GetDouble(BG_STR_CRC_ERROR_THRESHOLD);
    dcgmFieldValue_v1 fv = {0};

    fv.fieldType = DCGM_FT_INT64;
    fv.value.i64 = static_cast<uint64_t>(crcErrorThreshold);
    failureThresholds.push_back(fv);
    failureThresholds.push_back(fv); // insert once for each field id

    return bgGlobals->m_dcgmRecorder->CheckPerSecondErrorConditions(fieldIds, failureThresholds, gpuId, errorList,
                                                                    startTime);
}

/*****************************************************************************/
bool bg_check_error_conditions(BusGrindGlobals *bgGlobals, unsigned int gpuId, std::vector<DcgmError> &errorList,
                               timelib64_t startTime, timelib64_t endTime)
{
    bool passed = true;
    std::vector<unsigned short> fieldIds;
    std::vector<dcgmTimeseriesInfo_t> failureThresholds;

    fieldIds.push_back(DCGM_FI_DEV_PCIE_REPLAY_COUNTER);
    fieldIds.push_back(DCGM_FI_DEV_LWLINK_REPLAY_ERROR_COUNT_L0);
    fieldIds.push_back(DCGM_FI_DEV_LWLINK_REPLAY_ERROR_COUNT_L1);
    fieldIds.push_back(DCGM_FI_DEV_LWLINK_REPLAY_ERROR_COUNT_L2);
    fieldIds.push_back(DCGM_FI_DEV_LWLINK_REPLAY_ERROR_COUNT_L3);
    fieldIds.push_back(DCGM_FI_DEV_LWLINK_REPLAY_ERROR_COUNT_L4);
    fieldIds.push_back(DCGM_FI_DEV_LWLINK_REPLAY_ERROR_COUNT_L5);
    fieldIds.push_back(DCGM_FI_DEV_LWLINK_RECOVERY_ERROR_COUNT_L0);
    fieldIds.push_back(DCGM_FI_DEV_LWLINK_RECOVERY_ERROR_COUNT_L1);
    fieldIds.push_back(DCGM_FI_DEV_LWLINK_RECOVERY_ERROR_COUNT_L2);
    fieldIds.push_back(DCGM_FI_DEV_LWLINK_RECOVERY_ERROR_COUNT_L3);
    fieldIds.push_back(DCGM_FI_DEV_LWLINK_RECOVERY_ERROR_COUNT_L4);
    fieldIds.push_back(DCGM_FI_DEV_LWLINK_RECOVERY_ERROR_COUNT_L5);
    fieldIds.push_back(DCGM_FI_DEV_LWSWITCH_FATAL_ERRORS);
    
    if (bgGlobals->testParameters->GetBoolFromString(BG_STR_LWSWITCH_NON_FATAL_CHECK))
    {
        fieldIds.push_back(DCGM_FI_DEV_LWSWITCH_NON_FATAL_ERRORS);
    }
    dcgmTimeseriesInfo_t dt;
    memset(&dt, 0, sizeof(dt));

    // Record the maximum allowed replays
    dt.isInt = true;
    dt.val.i64 = static_cast<uint64_t>(bgGlobals->testParameters->GetDouble(BG_STR_MAX_PCIE_REPLAYS));
    failureThresholds.push_back(dt);
    
    // Every field after the first one counts as a failure if even one happens
    dt.val.i64 = 0;
    for (int i = 1; i < fieldIds.size(); i++)
    {
        failureThresholds.push_back(dt);
    }

    int ret = bgGlobals->m_dcgmRecorder->CheckErrorFields(fieldIds, &failureThresholds, gpuId, errorList,
                                                          startTime, endTime);
    if (ret == DR_COMM_ERROR)
    {
        bgGlobals->m_dcgmCommErrorOclwrred = true;
        return false;
    }
    else if (ret == DR_VIOLATION)
    {
        passed = false;
    }

    dcgmReturn_t st = bg_check_per_second_error_conditions(bgGlobals, gpuId, errorList, startTime);
    if (st != DCGM_ST_OK)
    {
        passed = false;
    }

    return passed;
}

/*****************************************************************************/
bool bg_check_global_pass_fail(BusGrindGlobals *bgGlobals, timelib64_t startTime, timelib64_t endTime)
{
    bool passed;
    bool allPassed = true;
    PluginDevice *bgGpu;
    std::vector<DcgmError> errorList;
    BusGrind *plugin = bgGlobals->busGrind;

    /* Get latest values for watched fields before checking pass fail
     * If there are errors getting the latest values, error information is added to errorList.
     */
    bgGlobals->m_dcgmRecorder->GetLatestValuesForWatchedFields(0, errorList);

    for (size_t i = 0; i < bgGlobals->gpu.size(); i++)
    {
        bgGpu = bgGlobals->gpu[i];
        passed = bg_check_error_conditions(bgGlobals, bgGpu->dcgmDeviceIndex, errorList, startTime, endTime);
        /* Some tests set the GPU result to fail when error conditions occur. Only consider the test passed 
         * if the existing result is not set to FAIL 
         */
        if (passed && plugin->GetGpuResults().find(bgGpu->dcgmDeviceIndex)->second != LWVS_RESULT_FAIL)
        {
            plugin->SetResultForGpu(bgGpu->dcgmDeviceIndex, LWVS_RESULT_PASS);
        }
        else
        {
            allPassed = false;
            plugin->SetResultForGpu(bgGpu->dcgmDeviceIndex, LWVS_RESULT_FAIL);
            // Log warnings for this gpu
            for (size_t j = 0; j < errorList.size(); j++)
            {
                plugin->AddErrorForGpu(bgGpu->dcgmDeviceIndex, errorList[j]);
            }
        }
    }

    return allPassed;
}

/*****************************************************************************/
int main_entry_wrapped(BusGrindGlobals *bgGlobals, const std::vector<unsigned int> &gpuList)
{
    int st, i;
    bool outputStats = true;
    timelib64_t startTime = timelib_usecSince1970();
    timelib64_t endTime;

    st = bg_cache_and_check_parameters(bgGlobals);
    if (st)
    {
        bg_cleanup(bgGlobals);
        return 1;
    }

    bgGlobals->m_dcgmRecorder = new DcgmRecorder();
    bgGlobals->m_dcgmRecorder->Init(lwvsCommon.dcgmHostname);
    /* Is binary logging enabled for this stat collection? */
    std::string logFileName = bgGlobals->testParameters->GetString(PS_LOGFILE);
    int logFileType = (int)bgGlobals->testParameters->GetDouble(PS_LOGFILE_TYPE);

    st = bg_lwml_init(bgGlobals, gpuList);
    if (st)
    {
        bg_cleanup(bgGlobals);
        return 1;
    }

    std::vector<unsigned short> fieldIds;
    fieldIds.push_back(DCGM_FI_DEV_PCIE_REPLAY_COUNTER);
    fieldIds.push_back(DCGM_FI_DEV_LWLINK_REPLAY_ERROR_COUNT_TOTAL);
    fieldIds.push_back(DCGM_FI_DEV_LWLINK_REPLAY_ERROR_COUNT_L0);
    fieldIds.push_back(DCGM_FI_DEV_LWLINK_REPLAY_ERROR_COUNT_L1);
    fieldIds.push_back(DCGM_FI_DEV_LWLINK_REPLAY_ERROR_COUNT_L2);
    fieldIds.push_back(DCGM_FI_DEV_LWLINK_REPLAY_ERROR_COUNT_L3);
    fieldIds.push_back(DCGM_FI_DEV_LWLINK_REPLAY_ERROR_COUNT_L4);
    fieldIds.push_back(DCGM_FI_DEV_LWLINK_REPLAY_ERROR_COUNT_L5);
    fieldIds.push_back(DCGM_FI_DEV_LWLINK_RECOVERY_ERROR_COUNT_TOTAL);
    fieldIds.push_back(DCGM_FI_DEV_LWLINK_RECOVERY_ERROR_COUNT_L0);
    fieldIds.push_back(DCGM_FI_DEV_LWLINK_RECOVERY_ERROR_COUNT_L1);
    fieldIds.push_back(DCGM_FI_DEV_LWLINK_RECOVERY_ERROR_COUNT_L2);
    fieldIds.push_back(DCGM_FI_DEV_LWLINK_RECOVERY_ERROR_COUNT_L3);
    fieldIds.push_back(DCGM_FI_DEV_LWLINK_RECOVERY_ERROR_COUNT_L4);
    fieldIds.push_back(DCGM_FI_DEV_LWLINK_RECOVERY_ERROR_COUNT_L5);
    fieldIds.push_back(DCGM_FI_DEV_LWLINK_CRC_FLIT_ERROR_COUNT_TOTAL); // Previously unchecked
    fieldIds.push_back(DCGM_FI_DEV_LWLINK_CRC_DATA_ERROR_COUNT_TOTAL); // Previously unchecked
    fieldIds.push_back(DCGM_FI_DEV_LWLINK_BANDWIDTH_TOTAL);
    fieldIds.push_back(DCGM_FI_DEV_LWSWITCH_NON_FATAL_ERRORS);
    fieldIds.push_back(DCGM_FI_DEV_LWSWITCH_FATAL_ERRORS);
    char fieldGroupName[128];
    char groupName[128];
    snprintf(fieldGroupName, sizeof(fieldGroupName), "pcie_field_group");
    snprintf(groupName, sizeof(groupName), "pcie_group");

    // 300.0 assumes this test will take less than 5 minutes
    bgGlobals->m_dcgmRecorder->AddWatches(fieldIds, gpuList, false, fieldGroupName, groupName, 300.0);

    bg_record_cliques(bgGlobals);

    /* For the following tests, a return of 0 is success. > 0 is
     * a failure of a test condition, and a < 0 is a fatal error
     * that ends BusGrind. Test condition failures are not fatal
     */

    /******************Host/Device Tests**********************/
    if ((lwvsCommon.training || bgGlobals->test_pinned) && !bg_should_stop(bgGlobals))
    {
        st = outputHostDeviceBandwidthMatrix(bgGlobals,true);
        if (st < 0)
        {
            goto NO_MORE_TESTS;
        }
    }
    if ((lwvsCommon.training || bgGlobals->test_unpinned) && !bg_should_stop(bgGlobals))
    {
        st = outputHostDeviceBandwidthMatrix(bgGlobals,false);
        if (st < 0)
        {
            goto NO_MORE_TESTS;
        }
    }

    if (bgGlobals->test_pinned && !bg_should_stop(bgGlobals))
    {
        st = outputConlwrrentHostDeviceBandwidthMatrix(bgGlobals,true);
        if (st < 0)
        {
            goto NO_MORE_TESTS;
        }
    }
    if (bgGlobals->test_unpinned && !bg_should_stop(bgGlobals))
    {
        st = outputConlwrrentHostDeviceBandwidthMatrix(bgGlobals,false);
        if (st < 0)
        {
            goto NO_MORE_TESTS;
        }
    }

    /*************************P2P Tests************************/
    if (bgGlobals->test_p2p_on && !bg_should_stop(bgGlobals))
    {
        st = outputP2PBandwidthMatrix(bgGlobals,true);
        if (st < 0)
        {
            goto NO_MORE_TESTS;
        }
    }
    if (bgGlobals->test_p2p_off && !bg_should_stop(bgGlobals))
    {
        st = outputP2PBandwidthMatrix(bgGlobals,false);
        if (st < 0)
        {
            goto NO_MORE_TESTS;
        }
    }

    if (bgGlobals->test_p2p_on && !bg_should_stop(bgGlobals))
    {
        st = outputConlwrrentPairsP2PBandwidthMatrix(bgGlobals,true);
        if (st < 0)
        {
            goto NO_MORE_TESTS;
        }
    }
    if (bgGlobals->test_p2p_off && !bg_should_stop(bgGlobals))
    {
        st = outputConlwrrentPairsP2PBandwidthMatrix(bgGlobals,false);
        if (st < 0)
        {
            goto NO_MORE_TESTS;
        }
    }

    if (bgGlobals->test_p2p_on && !bg_should_stop(bgGlobals))
    {
        st = outputConlwrrent1DExchangeBandwidthMatrix(bgGlobals,true);
        if (st < 0)
        {
            goto NO_MORE_TESTS;
        }
    }

    if (bgGlobals->test_p2p_off && !bg_should_stop(bgGlobals))
    {
        st = outputConlwrrent1DExchangeBandwidthMatrix(bgGlobals,false);
        if (st < 0)
        {
            goto NO_MORE_TESTS;
        }
    }

  /******************Latency Tests****************************/
    if ((lwvsCommon.training || bgGlobals->test_pinned) && !bg_should_stop(bgGlobals))
    {
        st = outputHostDeviceLatencyMatrix(bgGlobals,true);
        if (st < 0)
        {
            goto NO_MORE_TESTS;
        }
    }

    if ((lwvsCommon.training || bgGlobals->test_unpinned) && !bg_should_stop(bgGlobals))
    {
        st = outputHostDeviceLatencyMatrix(bgGlobals,false);
        if (st < 0)
        {
            goto NO_MORE_TESTS;
        }
    }

    if (bgGlobals->test_p2p_on && !bg_should_stop(bgGlobals))
    {
        st = outputP2PLatencyMatrix(bgGlobals,true);
        if (st < 0)
        {
            goto NO_MORE_TESTS;
        }
    }

    if (bgGlobals->test_p2p_off && !bg_should_stop(bgGlobals))
    {
        st = outputP2PLatencyMatrix(bgGlobals,false);
        if (st < 0)
        {
            goto NO_MORE_TESTS;
        }
    }

    /* This should come after all of the tests have run */
    NO_MORE_TESTS:

    endTime = timelib_usecSince1970();

    /* Check for global failures monitored by DCGM and set pass/fail status for each GPU.
     */
    bool testPassed = bg_check_global_pass_fail(bgGlobals, startTime, endTime);
    if (lwvsCommon.statsOnlyOnFail && !testPassed)
    {
        outputStats = false;
    }

    /* Should we write out a log file of stats? */
    if (logFileName.size() > 0 && outputStats)
    {
        st = bgGlobals->m_dcgmRecorder->WriteToFile(logFileName, logFileType, startTime);
        if (st)
        {
            PRINT_ERROR("%s", "Unable to write to log file %s", logFileName.c_str());
            std::string error = "There was an error writing test statistics to file '";
            error += logFileName + "'.";
            bgGlobals->busGrind->AddInfo(error);
        }
    }

    bg_cleanup(bgGlobals);
    return 0;
}

/*****************************************************************************/
int main_entry(const std::vector<unsigned int> &gpuList,
                       BusGrind *busGrind, TestParameters *testParameters)
{
    int st = 1; /* Default to failed in case we catch an exception */
    BusGrindGlobals *bgGlobals = &g_bgGlobals;

    memset(bgGlobals, 0, sizeof(*bgGlobals));
    bgGlobals->busGrind = busGrind;
    bgGlobals->testParameters = testParameters;


    try
    {
        st = main_entry_wrapped(bgGlobals, gpuList);
    }
    catch (const std::runtime_error &e)
    {
        DcgmError d;
        const char *err_str = e.what();
        DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_INTERNAL, d, err_str);
        PRINT_ERROR("%s", "Caught runtime_error %s", err_str);
        bgGlobals->busGrind->AddError(d);
        bgGlobals->busGrind->SetResult(LWVS_RESULT_FAIL);
        /* Clean up in case main wasn't able to */
        bg_cleanup(bgGlobals);
        // Let the TestFramework report the exception information.
        throw;
    }

    return st;
}


/*****************************************************************************/
