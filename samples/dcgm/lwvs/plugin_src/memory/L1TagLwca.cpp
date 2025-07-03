/*
 * LWIDIA_COPYRIGHT_BEGIN
 *
 * Copyright 2019 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * LWIDIA_COPYRIGHT_END
 */
#include <assert.h>
#include <string.h>
#include <lwos.h>
#include <lwca.h>
#include <lwml.h>
#include "L1TagLwda.h"
#include "l1tag_ptx_string.h"
#include "timelib.h"
#include <PluginCommon.h>
#include "LwvsDeviceList.h"
#include "lwml_internal.h"

void L1TagLwda::Cleanup(void)
{
    LWresult lwRes;

    if (m_lwMod)
    {
        return;
    }
    
    if (m_hostErrorLog)
    {
        delete m_hostErrorLog;
    }

    if (m_l1Data)
    {
        lwRes = lwMemFree(m_l1Data);
        if (LWDA_SUCCESS != lwRes)
        {
            LOG_LWDA_ERROR_FOR_PLUGIN(m_plugin, "lwMemFree", lwRes, m_gpuIndex);
        }
    }

    if (m_devMiscompareCount)
    {
        lwRes = lwMemFree(m_devMiscompareCount);
        if (LWDA_SUCCESS != lwRes)
        {
            LOG_LWDA_ERROR_FOR_PLUGIN(m_plugin, "lwMemFree", lwRes, m_gpuIndex);
        }
    }

    if (m_devErrorLog)
    {
        lwRes = lwMemFree(m_devErrorLog);
        if (LWDA_SUCCESS != lwRes)
        {
            LOG_LWDA_ERROR_FOR_PLUGIN(m_plugin, "lwMemFree", lwRes, m_gpuIndex);
        }
    }

    lwRes = lwModuleUnload(m_lwMod);
    if (LWDA_SUCCESS != lwRes)
    {
        LOG_LWDA_ERROR_FOR_PLUGIN(m_plugin, "lwModuleUnload", lwRes, m_gpuIndex);
    }
}

int L1TagLwda::AllocDeviceMem(int size, LWdeviceptr *ptr)
{
    if (LWDA_SUCCESS != lwMemAlloc(ptr, size))
    {
        DcgmError d;
        DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_MEMORY_ALLOC, d, size, m_gpuIndex);
        m_plugin->AddErrorForGpu(m_gpuIndex, d);
        lwvsCommon.errorMask |= MEM_ERR_LWDA_ALLOC_FAIL;
        PRINT_ERROR("%s", "%s", d.GetMessage().c_str());
        return 1;
    }

    return 0;
}

lwvsPluginResult_t L1TagLwda::GetMaxL1CacheSizePerSM(uint32_t &l1PerSMBytes)
{
    // Determine L1 cache size
    //
    // This test, supports volta and later if cache size is known.  It also has a 256KB cache per SM limit.
    // There is no API lwrrently to get L1 cache size, so for now, this test will support volta only,
    // which has 128KB cache per SM on all models.

    lwmlReturn_t lwmlSt;
    etblLWMLCommonInternal *etbl;
    lwmlSt = lwmlInternalGetExportTable((const void**)&etbl,
                                        &ETID_LWMLCommonInternal);
    if (lwmlSt != LWML_SUCCESS)
    {
        DcgmError d;
        DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_LWML_API, d, "lwmlInternalAPI_1", lwmlErrorString(lwmlSt));
        PRINT_ERROR("%d", "Got lwml error %d from lwmlInternalGetExportTable", (int)lwmlSt);
        m_plugin->AddError(d);
        return LWVS_RESULT_FAIL;
    }

    lwmlChipArchitecture_t arch;
    lwmlSt = LWML_CALL_ETBL(etbl, DeviceGetChipArchitecture, (m_lwmlDevice, &arch));
    if (lwmlSt != LWML_SUCCESS)
    {
        DcgmError d;
        DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_LWML_API, d, "lwmlInternalAPI_2", lwmlErrorString(lwmlSt));
        PRINT_ERROR("%d %d", "Got lwml error %d from lwmlDeviceGetChipArchitecture of lwmlIndex %d",
                    (int)lwmlSt, m_gpuIndex);
        m_plugin->AddError(d);
        return LWVS_RESULT_FAIL;
    }
    PRINT_INFO("%d","Got chip arch = %d\n", arch);

    l1PerSMBytes = 0;
    switch(arch)
    {
        case LWML_CHIP_ARCH_VOLTA:
            l1PerSMBytes = 128<<10;
            break;
        default:
            DcgmError d;
            DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_L1TAG_UNSUPPORTED, d);
            m_plugin->AddInfoVerboseForGpu(m_gpuIndex, d.GetMessage());
            return LWVS_RESULT_SKIP;
    }
    return LWVS_RESULT_PASS;
}

lwvsPluginResult_t L1TagLwda::LogLwdaFail(const char *msg, const char *lwdaFuncS, LWresult lwRes)
{
    DcgmError d;
    std::string error = AppendLwdaDriverError(msg, lwRes);
    DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_LWDA_API, d, lwdaFuncS);
    d.AddDetail(error);
    m_plugin->AddErrorForGpu(m_gpuIndex, d);
    return LWVS_RESULT_FAIL;
}

lwvsPluginResult_t L1TagLwda::RunTest(void)
{
    LWresult lwRes;
    int      attr;

    lwRes = lwModuleLoadData(&m_lwMod, l1tag_ptx_string);
    if (LWDA_SUCCESS != lwRes)
    {
        return LogLwdaFail("Unable to load LWCA module from PTX string", "lwModuleLoadData", lwRes);
    }

    // Seed rng
    srand(time(NULL));

    // Determine L1 cache size
    // This test only supports L1 cache sizes up to 256KB
    uint32_t l1PerSMBytes;
    lwvsPluginResult_t ret = GetMaxL1CacheSizePerSM(l1PerSMBytes);
    if (ret != LWVS_RESULT_PASS)
    {
        return ret;
    }

    if (l1PerSMBytes > (256<<10))
    {
        DcgmError d;
        DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_L1TAG_UNSUPPORTED, d);
        m_plugin->AddInfoVerboseForGpu(m_gpuIndex, d.GetMessage());
        return LWVS_RESULT_SKIP;
    }

    uint32_t numBlocks = 0;

    lwRes = lwDeviceGetAttribute(&attr, LW_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, m_lwDevice);
    if (LWDA_SUCCESS != lwRes)
    {
        return LogLwdaFail("Unable to get multiprocessor count", "lwDeviceGetAttribute", lwRes);
    }
    numBlocks = (uint32_t)attr;

    // Get Compute capability
    int lwMajor;
    int lwMinor;
    lwRes = lwDeviceGetAttribute(&lwMajor, LW_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, m_lwDevice);
    if (LWDA_SUCCESS != lwRes)
    {
        return LogLwdaFail("Unable to get compute capability major", "lwDeviceGetAttribute", lwRes);
    }

    lwRes = lwDeviceGetAttribute(&lwMinor, LW_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, m_lwDevice);
    if (LWDA_SUCCESS != lwRes)
    {
        return LogLwdaFail("Unable to get compute capability minor", "lwDeviceGetAttribute", lwRes);
    }

    if (lwMajor < 7)
    {
        DcgmError d;
        DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_L1TAG_UNSUPPORTED, d);
        m_plugin->AddInfoVerboseForGpu(m_gpuIndex, d.GetMessage());
        return LWVS_RESULT_SKIP;
    }

    // Set number of threads.  
    uint32_t numThreads = 0;
    uint32_t maxThreads;

    lwRes = lwDeviceGetAttribute(&attr, LW_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, m_lwDevice);
    if (LWDA_SUCCESS != lwRes)
    {
        return LogLwdaFail("Unable to get max threads per block", "lwDeviceGetAttribute", lwRes);
    }
    maxThreads = (uint32_t) attr;

    numThreads = l1PerSMBytes / L1_LINE_SIZE_BYTES;
    assert(l1PerSMBytes % L1_LINE_SIZE_BYTES == 0);
    assert (numThreads <= maxThreads);

    // Allocate memory for L1
    int          l1Size = numBlocks * l1PerSMBytes;
    uint64_t     hostMiscompareCount;

    if (AllocDeviceMem(l1Size, &m_l1Data))
    {
        return LWVS_RESULT_FAIL;
    }

    // Allocate miscompare count & error log
    if (AllocDeviceMem(sizeof(uint64_t), &m_devMiscompareCount))
    {
        return LWVS_RESULT_FAIL;
    }

    if (AllocDeviceMem(sizeof(L1TagError)*m_errorLogLen, &m_devErrorLog))
    {
        return LWVS_RESULT_FAIL;
    }
    m_hostErrorLog = new L1TagError[m_errorLogLen];

    // Format kernel parameters
    m_kernelParams.data           = m_l1Data;
    m_kernelParams.sizeBytes      = l1Size;
    m_kernelParams.errorCountPtr  = m_devMiscompareCount;
    m_kernelParams.errorLogPtr    = m_devErrorLog;
    m_kernelParams.errorLogLen    = m_errorLogLen;
    m_kernelParams.iterations     = m_innerIterations;

    PRINT_INFO("%u","L1tag #processor = %u\n",numBlocks);
    PRINT_INFO("%d %d","Compute cap=%d.%d\n",lwMajor,lwMinor);
    PRINT_INFO("%u %u","Threads = %u, %u max\n",numThreads,maxThreads);
    PRINT_INFO("%d","L1 Size = %d\n",l1Size);

    double durationMs = 0.0;

    // Get Init function
    LWfunction initL1DataFunc;
    lwRes = lwModuleGetFunction(&initL1DataFunc, m_lwMod, InitL1Data_func_name);
    if (LWDA_SUCCESS != lwRes)
    {
        return LogLwdaFail("Unable to load module function InitL1Data", "lwModuleGetFunction", lwRes);
    }

    // Get tag test (run) function.
    LWfunction testRunDataFunc;
    lwRes = lwModuleGetFunction(&testRunDataFunc, m_lwMod, L1TagTest_func_name);
    if (LWDA_SUCCESS != lwRes)
    {
        return LogLwdaFail("Unable to load module function L1TagTest", "lwModuleGetFunction", lwRes);
    }

    // Create events for timing kernel
    LWevent startEvent;
    lwRes = lwEventCreate(&startEvent, LW_EVENT_DEFAULT);
    if (LWDA_SUCCESS != lwRes)
    {
        return LogLwdaFail("Unable create LWCA event", "lwEventCreate", lwRes);
    }

    LWevent stopEvent;
    lwRes = lwEventCreate(&stopEvent, LW_EVENT_DEFAULT);
    if (LWDA_SUCCESS != lwRes)
    {
        return LogLwdaFail("Unable create LWCA event", "lwEventCreate", lwRes);
    }

    // Create stream to synchronize accesses
    LWstream stream;
    lwRes = lwStreamCreate(&stream, LW_STREAM_NON_BLOCKING);
    if (LWDA_SUCCESS != lwRes)
    {
        return LogLwdaFail("Unable create LWCA stream", "lwStreamCreate", lwRes);
    }

    // Run for runtimeMs if it is nonzero.
    // Otherwise run for m_testLoops loops.
    uint64_t totalNumErrors = 0;
    uint64_t kernLaunchCount = 0;
    for (uint64_t loop = 0;
         m_runtimeMs ? durationMs < static_cast<double>(m_runtimeMs) : loop < m_testLoops;
         loop++)
    {
        // Clear error counter
        uint64_t zeroVal = 0;
        lwRes = lwMemcpyHtoDAsync(m_devMiscompareCount, &zeroVal, sizeof(uint64_t),stream);
        if (LWDA_SUCCESS != lwRes)
        {
            return LogLwdaFail("Failed to clear m_devMiscompareCount", "lwMemsetD32Async", lwRes);
        }

        // Use a different RNG seed each loop
        m_kernelParams.randSeed = (uint32_t)rand();

        // Run the init data buffer kernel
        void *paramPtrs[] = { &m_kernelParams };
        lwRes = lwLaunchKernel(initL1DataFunc,
                               numBlocks,       // gridDimX
                               1,               // gridDimY
                               1,               // gridDimZ
                               numThreads,      // blockDimX
                               1,               // blockDimY
                               1,               // blockDimZ
                               0,               // sharedMemSize
                               stream,
                               paramPtrs,
                               NULL);
        if (LWDA_SUCCESS != lwRes)
        {
            return LogLwdaFail("Failed to launch InitL1Data kernel", "lwLaunchKernel", lwRes);
        }

        // The run the test kernel, recording elaspsed time with events
        lwRes = lwEventRecord(startEvent, stream);
        if (LWDA_SUCCESS != lwRes)
        {
            return LogLwdaFail("Failed to record start event", "lwEventRecord", lwRes);
        }

        lwRes = lwLaunchKernel(testRunDataFunc,
                               numBlocks,       // gridDimX
                               1,               // gridDimY
                               1,               // gridDimZ
                               numThreads,      // blockDimX
                               1,               // blockDimY
                               1,               // blockDimZ
                               0,               // sharedMemSize
                               stream,
                               paramPtrs,
                               NULL);
        if (LWDA_SUCCESS != lwRes)
        {
            return LogLwdaFail("Failed to launch L1TagTest kernel", "lwLaunchKernel", lwRes);
        }
        kernLaunchCount++;

        lwRes = lwEventRecord(stopEvent, stream);
        if (LWDA_SUCCESS != lwRes)
        {
            return LogLwdaFail("Failed to record stop event", "lwEventRecord", lwRes);
        }

        // Get error count
        lwRes = lwMemcpyDtoHAsync(&hostMiscompareCount , m_devMiscompareCount, sizeof(uint64_t), stream);
        if (LWDA_SUCCESS != lwRes)
        {
            return LogLwdaFail("Failed to schedule miscompareCount copy", "lwMemcpyDtoHAsync", lwRes);
        }

        // Synchronize and get time for kernel completion
        lwRes = lwStreamSynchronize(stream);
        if (LWDA_SUCCESS != lwRes)
        {
            return LogLwdaFail("Failed to synchronize", "lwStreamSynchronize", lwRes);
        }

        float elapsedMs;
        lwRes = lwEventElapsedTime(&elapsedMs, startEvent, stopEvent);
        if (LWDA_SUCCESS != lwRes)
        {
            return LogLwdaFail("Failed elapsed time callwlation", "lwEventElapsedTime", lwRes);
        }
        durationMs += elapsedMs;

        // Update test progress
        if (m_runtimeMs)
        {
            m_plugin->progressOut->updatePluginProgress(
                 durationMs / static_cast<double>(m_runtimeMs) * 100, false);
        }
        else
        {
            m_plugin->progressOut->updatePluginProgress(
                 static_cast<double>(loop) / static_cast<double>(m_testLoops) * 100, false);
        }

        // Handle errors
        totalNumErrors += hostMiscompareCount;
        if (hostMiscompareCount > 0)
        {
            PRINT_ERROR("%lu %lu",
                        "LwdaL1Tag found %lu miscompare(s) on loop %lu\n",
                        hostMiscompareCount, loop);

            lwRes = lwMemcpyDtoH(m_hostErrorLog, m_devErrorLog, sizeof(L1TagError)*m_errorLogLen);
            if (LWDA_SUCCESS != lwRes)
            {
                return LogLwdaFail("Failed to copy error log to host", "lwMemcpyDtoH", lwRes);
            }

            DcgmError d;
            DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_L1TAG_MISCOMPARE, d);
            m_plugin->AddErrorForGpu(m_gpuIndex, d);

            if (hostMiscompareCount > m_errorLogLen)
            {
                PRINT_WARNING("%lu %u",
                            "%lu miscompares, but error log only contains %u entries. "
                            "Some failing SMID/TPCs may not be reported.\n",
                            hostMiscompareCount,
                            m_errorLogLen);
            }

            for (uint32_t i = 0; i < hostMiscompareCount && i < m_errorLogLen; i++)
            {
                L1TagError &error = m_hostErrorLog[i];
                if (m_dumpMiscompares)
                {
                    PRINT_ERROR("%u %s %04X %04X %lu %d %d %d %d",
                           "Iteration  : %u\n"
                           "TestStage  : %s\n"
                           "DecodedOff : 0x%04X\n"
                           "ExpectedOff: 0x%04X\n"
                           "Iteration  : %lu\n"
                           "InnerLoop  : %d\n"
                           "Smid       : %d\n"
                           "Warpid     : %d\n"
                           "Laneid     : %d\n"
                           "\n",
                           i,
                           (error.testStage == PreLoad) ? "PreLoad" : "RandomLoad",
                           error.decodedOff,
                           error.expectedOff,
                           error.iteration,
                           error.innerLoop,
                           error.smid,
                           error.warpid,
                           error.laneid
                   );
                }
            }
            return LWVS_RESULT_FAIL;
        }
    }

    m_plugin->progressOut->updatePluginProgress ( 100, true );
    PRINT_INFO("%f","Complete  durationMs = %f\n",durationMs);

    // Kernel runtime and error prints useful for debugging
    // Guard against divide-by-zero errors (that shouldn't occur)
    const double durationSec = durationMs / 1000.0;
    if (totalNumErrors && durationMs)
    {
        PRINT_INFO("%lu %f",
                   "L1tag TotalErrors: %lu\n"
                   "L1tag Errors/s   : %f\n",
                   totalNumErrors,
                   static_cast<double>(totalNumErrors) / durationSec);
    }

    if (kernLaunchCount && durationMs)
    {
        PRINT_INFO("%f %f",
                   "L1tag Total Kernel Runtime: %fms\n"
                   "L1tag Avg Kernel Runtime: %fms\n",
                   durationMs, durationMs / kernLaunchCount);
    }
    return LWVS_RESULT_PASS;
}

lwvsPluginResult_t L1TagLwda::TestMain(unsigned int lwmlGpuIndex)
{
    lwvsPluginResult_t result;

    m_gpuIndex = lwmlGpuIndex;

    m_runtimeMs = 1000*(uint32_t)m_testParameters->GetDouble(MEMORY_L1TAG_STR_TEST_DURATION);
    m_testLoops = (uint64_t)m_testParameters->GetDouble(MEMORY_L1TAG_STR_TEST_LOOPS);
    m_innerIterations = (uint64_t)m_testParameters->GetDouble(MEMORY_L1TAG_STR_INNER_ITERATIONS);
    m_errorLogLen = (uint32_t)m_testParameters->GetDouble(MEMORY_L1TAG_STR_ERROR_LOG_LEN);
    m_dumpMiscompares = m_testParameters->GetBoolFromString(MEMORY_L1TAG_STR_DUMP_MISCOMPARES);

    result = RunTest();

    Cleanup();

    return result;
}
