#define __STDC_LIMIT_MACROS
#include <stdint.h>

#include "GpuBurnPlugin.h"
#include "gpuburn_ptx_string.h"
#include "LwvsThread.h"

/*
 * This code is adapted from Gpu Burn, written by Ville Tomonen. He wrote it under
 * the following license:
 */

/*
 * Copyright (c) 2016, Ville Timonen
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROLWREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * The views and conclusions contained in the software and documentation are those
 * of the authors and should not be interpreted as representing official policies,
 * either expressed or implied, of the FreeBSD Project.
 */


#define SIZE 2048ul // Matrices are SIZE*SIZE..  2048^2 should be efficiently implemented in LWBLAS
#define USEMEM 0.9 // Try to allocate 90% of memory

// Used to report op/s, measured through Visual Profiler, LWBLAS from LWCA 7.5
// (Seems that they indeed take the naive dim^3 approach)
#define OPS_PER_MUL 17188257792ul

/*****************************************************************************/
/* 
 * GpuBurnPlugin Implementation 
 */
/*****************************************************************************/
GpuBurnPlugin::GpuBurnPlugin(): m_lwmlInitialized(false), m_dcgmRecorderInitialized(false), 
                                m_dcgmCommErrorOclwrred(false)
{
    TestParameters *tp;

    m_infoStruct.name = GPUBURN_PLUGIN_NAME;
    m_infoStruct.shortDescription = "This plugin will stress the framebuffer of a list of GPUs.";
    m_infoStruct.testGroups = "Hardware";
    m_infoStruct.selfParallel = true; // ?
    m_infoStruct.logFileTag = GPUBURN_PLUGIN_LF_NAME;

    // Populate default test parameters
    tp = new TestParameters();
    tp->AddString(PS_RUN_IF_GOM_ENABLED, "False");
    // 3 minutes should be enough to catch any heat issues on the GPUs.
    tp->AddDouble(GPUBURN_STR_TEST_DURATION, 180.0, 1.0, 86400.0);
    tp->AddString(GPUBURN_STR_USE_DOUBLES, "False");
    tp->AddDouble(GPUBURN_STR_TEMPERATURE_MAX, 100.0, 30.0, 120.0);
    tp->AddDouble(GPUBURN_STR_SBE_ERROR_THRESHOLD, DCGM_FP64_BLANK, 0.0, DCGM_FP64_BLANK);
    tp->AddString(GPUBURN_STR_IS_ALLOWED, "False");
    m_infoStruct.defaultTestParameters = tp;
}

/*****************************************************************************/
GpuBurnPlugin::~GpuBurnPlugin()
{
    Cleanup();
}

/*****************************************************************************/
void GpuBurnPlugin::Cleanup()
{
    for (size_t deviceIdx = 0; deviceIdx < m_device.size(); deviceIdx++)
    {
        GpuBurnDevice *gbd = m_device[deviceIdx];
        lwdaSetDevice(gbd->lwdaDeviceIdx);
        delete gbd;
    }

    m_device.clear();

    // We don't own testParameters, so we don't delete them
    if (m_dcgmRecorderInitialized)
    {
        m_dcgmRecorder.Shutdown();
    }
    m_dcgmRecorderInitialized = false;

    if (m_lwmlInitialized)
    {
        lwmlShutdown();
    }
    m_lwmlInitialized = false;
}

/*****************************************************************************/
bool GpuBurnPlugin::LwmlInit(const std::vector<unsigned int> &gpuList)
{
    lwmlReturn_t lwmlSt;
    char buf[256] = { 0 };
    lwdaError_t   lwdaSt;

    PRINT_DEBUG("", "Begin LwmlInit");

    // Attach to every device by index and reset it in case a previous plugin
    // didn't clean up after itself.
    int lwdaDeviceCount;

    lwdaSt = lwdaGetDeviceCount(&lwdaDeviceCount);
    if (lwdaSt == lwdaSuccess)
    {
        for (int deviceIdx = 0; deviceIdx < lwdaDeviceCount; deviceIdx++)
        {
            lwdaSetDevice(deviceIdx);
            lwdaDeviceReset();
            PRINT_DEBUG("%d", "Reset device %d", deviceIdx);
        }
    }
    else
    {
        lwvsCommon.errorMask |= GPUBURN_ERR_GENERIC;
        LOG_LWDA_ERROR("lwdaGetDeviceCount", lwdaSt, 0, 0, false);
        return false;
    }

    lwmlSt = lwmlInit();
    if (lwmlSt != LWML_SUCCESS)
    {
        lwvsCommon.errorMask |= GPUBURN_ERR_GENERIC;
        DcgmError d;
        DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_LWML_API, d, "lwmlInit", lwmlErrorString(lwmlSt));
        AddError(d);
        PRINT_ERROR("%s", "%s", d.GetMessage().c_str());
        return false;
    }
    m_lwmlInitialized = true;

    for (size_t gpuListIndex = 0; gpuListIndex < gpuList.size(); gpuListIndex++)
    {
        GpuBurnDevice *gbDevice = NULL;
        try
        {
            gbDevice = new GpuBurnDevice(gpuList[gpuListIndex], this);
        }
        catch (DcgmError &d)
        {
            if (gbDevice != NULL)
            {
                AddErrorForGpu(gpuList[gpuListIndex], d);
                delete gbDevice;
            }
            else
            {
                AddErrorForGpu(gpuList[gpuListIndex], d);
                PRINT_ERROR("%s", "%s", d.GetMessage().c_str());
            }
            return false;
        }
        // At this point, we consider this GPU part of our set
        m_device.push_back(gbDevice);
    }

    PRINT_DEBUG("", "End LwmlInit");
    return true;
}

/*****************************************************************************/
void GpuBurnPlugin::Go(TestParameters *testParameters, const std::vector<unsigned int> &gpuList)
{
    bool result;
    InitializeForGpuList(gpuList);

    if (!testParameters->GetBoolFromString(GPUBURN_STR_IS_ALLOWED))
    {
        DcgmError d;
        DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_TEST_DISABLED, d, GPUBURN_PLUGIN_NAME);
        AddError(d);
        SetResult(LWVS_RESULT_SKIP);
        return;
    }

    /* Cache test parameters */
    m_testParameters = testParameters; // DO NOT DELETE
    m_testDuration = m_testParameters->GetDouble(GPUBURN_STR_TEST_DURATION); /* test length, in seconds */
    m_sbeFailureThreshold = m_testParameters->GetDouble(GPUBURN_STR_SBE_ERROR_THRESHOLD);

    std::string useDoubles = m_testParameters->GetString(GPUBURN_STR_USE_DOUBLES);
    if (useDoubles.size() > 0)
    {
        if (useDoubles[0] == 't' || useDoubles[0] == 'T')
        {
            m_useDoubles = true;
        }
    }

    if (m_useDoubles)
    {
        result = RunTest<double>(gpuList);
    }
    else
    {
        result = RunTest<float>(gpuList);
    }
    if (main_should_stop)
    {
        DcgmError d;
        DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_ABORTED, d);
        AddError(d);
        SetResult(LWVS_RESULT_SKIP);
    }
    else if (!result)
    {
        // There was an error running the test - set result for all gpus to failed
        SetResult(LWVS_RESULT_FAIL);
    }
}

/*************************************************************************/
/*
 * Check if our temperature stayed within the proscribed limits.
 */
bool GpuBurnPlugin::CheckGpuTemperature(GpuBurnDevice *device, std::vector<DcgmError> &errorList,
                                        timelib64_t startTime, timelib64_t earliestStopTime, bool testFinished)
{
    std::string infoMsg;
    long long maxTemp = static_cast<long long>(m_testParameters->GetDouble(GPUBURN_STR_TEMPERATURE_MAX));
    long long highTempObserved = 0;
    int st = m_dcgmRecorder.CheckGpuTemperature(device->dcgmDeviceIndex, errorList, maxTemp, infoMsg, startTime,
                                                earliestStopTime, highTempObserved);

    if (testFinished && highTempObserved != 0)
    {
        RecordObservedMetric(device->dcgmDeviceIndex, GPUBURN_STR_TEMPERATURE_MAX, highTempObserved);
    }

    if (testFinished)
    {
        AddInfoVerboseForGpu(device->dcgmDeviceIndex, infoMsg);
    }

    if (st == DR_VIOLATION)
    {
        lwvsCommon.errorMask |= GPUBURN_ERR_GENERIC;
    }
    // Check for a communication error
    else if (st == DR_COMM_ERROR)
    {
        m_dcgmCommErrorOclwrred = true;
    }

    return errorList.empty();
}

/*****************************************************************************/
/*
 * Check for single or double bit errors as well as XID errors
 */
bool GpuBurnPlugin::CheckLwmlEvents(GpuBurnDevice *device, std::vector<DcgmError> &errorList,
                                    timelib64_t startTime, timelib64_t endTime)
{
    std::vector<dcgmTimeseriesInfo_t> failureThresholds;
    std::vector<dcgmTimeseriesInfo_t> *thresholdsPtr = 0;
    std::vector<unsigned short> fieldIds;
    dcgmTimeseriesInfo_t tsInfo;

    if (DCGM_FP64_IS_BLANK(m_sbeFailureThreshold) == 0)
    {
        // Only add this field if there is a failure threshold
        fieldIds.push_back(DCGM_FI_DEV_ECC_SBE_VOL_TOTAL);

        memset(&tsInfo, 0, sizeof(tsInfo));
        tsInfo.isInt = true;
        tsInfo.val.i64 = static_cast<long long>(m_sbeFailureThreshold);
        failureThresholds.push_back(tsInfo);

        // Add thresholds of 0 for the other 2 fields This is not necessary if we are not checking
        // for SBE errors because DcgmRecorder::CheckErrorFields assumes a default threshold of 0 when
        // thresholdsPtr is NULL
        tsInfo.val.i64 = 0;
        for (int i = 0; i < 2; i++)
        {
            failureThresholds.push_back(tsInfo);
        }

        thresholdsPtr = &failureThresholds;
    }

    fieldIds.push_back(DCGM_FI_DEV_ECC_DBE_VOL_TOTAL);
    fieldIds.push_back(DCGM_FI_DEV_XID_ERRORS);

    int st = m_dcgmRecorder.CheckErrorFields(fieldIds, thresholdsPtr, device->dcgmDeviceIndex, errorList,
                                             startTime, endTime);
    // Check for a communication error
    if (st == DR_COMM_ERROR)
    {
        m_dcgmCommErrorOclwrred = true;
    }

    // Return true if there are no errors, false otherwise
    return errorList.empty();
}

/*************************************************************************/
/*
 * Check for pass or failure for a single device
 */
bool GpuBurnPlugin::CheckPassFailSingleGpu(GpuBurnDevice *device, std::vector<DcgmError> &errorList,
                                           timelib64_t startTime, timelib64_t earliestStopTime, bool testFinished)
{
    DcgmLockGuard lock(&m_mutex);
    bool passed = true;
    bool result = false;
    int st;

    // Check for errors in the plugin
    result = CheckGpuTemperature(device, errorList, startTime, earliestStopTime, testFinished);
    passed &= (result && !m_dcgmCommErrorOclwrred); // passed if temp check passed and no dcgmCommError oclwrred

    result = CheckLwmlEvents(device, errorList, startTime, earliestStopTime);
    passed &= (result && !m_dcgmCommErrorOclwrred); // passed if lwml check passed and no dcgmCommError oclwrred

    if (m_dcgmRecorder.CheckThermalViolations(device->dcgmDeviceIndex, errorList,
                                              startTime, earliestStopTime) != DR_SUCCESS)
    {
        passed = false;
    }

    st = m_dcgmRecorder.CheckForThrottling(device->dcgmDeviceIndex, startTime, errorList);
    if (st == DR_COMM_ERROR)
    {
        m_dcgmCommErrorOclwrred = true;
        passed = false;
    }
    else if (st == DR_VIOLATION)
    {
        passed = false;
    }

    return passed;
}

/*************************************************************************/
/*
 * Set pass/fail status for each gpu. Returns true plugin passed for all gpus, false otherwise
 */
bool GpuBurnPlugin::CheckPassFail(timelib64_t startTime, timelib64_t earliestStopTime,
                                  const std::vector<int> &errorCount)
{
    bool passed, allPassed = true;
    std::vector<DcgmError> errorList;
    std::vector<DcgmError> errorListAllGpus;

    /* Get latest values for watched fields before checking pass fail
     * If there are errors getting the latest values, error information is added to errorListAllGpus.
     */
    m_dcgmRecorder.GetLatestValuesForWatchedFields(0, errorListAllGpus);

    for (size_t i = 0; i < m_device.size(); i++)
    {
        errorList.clear();
        // Check for memory errors found in the plugin
        passed = CheckPassFailSingleGpu(m_device[i], errorList, startTime, earliestStopTime);
        if (passed && errorCount[i] == 0)
        {
            SetResultForGpu(m_device[i]->dcgmDeviceIndex, LWVS_RESULT_PASS);
        }
        else
        {
            allPassed = false;
            if (errorCount[i])
            {
                DcgmError d;
                DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_FAULTY_MEMORY, d, errorCount[i], m_device[i]->lwmlDeviceIndex);
                errorList.push_back(d);
            }

            SetResultForGpu(m_device[i]->dcgmDeviceIndex, LWVS_RESULT_FAIL);
            for (size_t j = 0; j < errorList.size(); j++)
            {
                AddErrorForGpu(m_device[i]->dcgmDeviceIndex, errorList[j]);
            }
        }
    }

    for (size_t i = 0; i < errorListAllGpus.size(); i++)
    {
        AddError(errorListAllGpus[i]);
    }

    return allPassed;
}

/*************************************************************************/
template <class T> class GpuBurnWorker : public LwvsThread
{
public:

    /*************************************************************************/
    /*
     * Constructor
     */
    GpuBurnWorker(GpuBurnDevice *gpuBurnDevice, GpuBurnPlugin &plugin, bool useDoubles, double testDuration,
                  DcgmRecorder &dcgmRecorder);

    /*************************************************************************/
    /*
     * Destructor
     */
    virtual ~GpuBurnWorker();

    /*************************************************************************/
    /*
     * Get the time this thread stopped
     */
    timelib64_t getStopTime() const
    {
        return m_stopTime;
    }

    /*************************************************************************/
    /*
     * Get the total number of errors detected by the test
     */
    long long getTotalErrors() const
    {
        return m_totalErrors;
    }

    /*************************************************************************/
    /*
     * Get the total matrix multiplications performed
     */
    long long getTotalOperations() const
    {
        return m_totalOperations;
    }

    /*************************************************************************/
    /*
     * Set our lwca context
     */
    int bind();

    /*************************************************************************/
    /*
     * Get available memory
     */
    size_t availMemory(int &st);

    /*************************************************************************/
    /*
     * Allocate the buffers
     */
    void allocBuffers()
    {
        // Initting A and B with random data
        m_A = (T*) malloc(sizeof(T)*SIZE*SIZE);
        m_B = (T*) malloc(sizeof(T)*SIZE*SIZE);
        srand(10);
        for (size_t i = 0; i < SIZE*SIZE; ++i)
        {
            m_A[i] = (T)((double)(rand()%1000000)/100000.0);
            m_B[i] = (T)((double)(rand()%1000000)/100000.0);
        }
    }

    /*************************************************************************/
    /*
     * Initialize the buffers
     */
    int initBuffers();

    /*************************************************************************/
    /*
     * Load the compare LWCA functions compiled separately from our ptx string
     */
    int initCompareKernel();

    /*************************************************************************/
    /*
     * Check for incorrect memory
     */
    int compare();

    /*************************************************************************/
    /*
     * Perform some matrix math
     */
    int compute();

    /*************************************************************************/
    /*
     * Worker thread main
     */
    void run();

private:
    GpuBurnDevice  *m_device;
    GpuBurnPlugin  &m_plugin;
    bool            m_useDoubles;
    double          m_testDuration;
    lwblasHandle_t  m_lwblas;
    long long int   m_error;
    size_t          m_iters;
    size_t          m_resultSize;

    static const int g_blockSize = 16;

    LWmodule        m_module;
    LWfunction      m_function;

    LWdeviceptr     m_Cdata;
    LWdeviceptr     m_Adata;
    LWdeviceptr     m_Bdata;
    LWdeviceptr     m_faultyElemData;
    T              *m_A;
    T              *m_B;
    timelib64_t     m_stopTime;
    long long       m_totalOperations;
    long long       m_totalErrors;
    DcgmRecorder   &m_dcgmRecorder;
};


/****************************************************************************/
/* 
 * GpuBurnPlugin::RunTest implementation.
 */
/****************************************************************************/
template<class T> bool GpuBurnPlugin::RunTest(const std::vector<unsigned int> &gpuList)
{
    std::vector<int> errorCount;
    std::vector<long long> operationsPerformed;
    GpuBurnWorker<T> *workerThreads[GPUBURN_MAX_DEVICES] = { 0 };
    int               st;
    int               activeThreadCount = 0;
    bool              testPassed;
    timelib64_t       earliestStopTime;
    timelib64_t       startTime = timelib_usecSince1970();
    unsigned int      timeCount = 0;
 
    std::string logFileName = m_testParameters->GetString(PS_LOGFILE);
    int logFileType = static_cast<int>(m_testParameters->GetDouble(PS_LOGFILE_TYPE));
    std::string dcgmError;
    std::vector<unsigned short> fieldIds;
 
    if (!LwmlInit(gpuList))
    {
        Cleanup();
        return false;
    }

    std::string errStr = m_dcgmRecorder.Init(lwvsCommon.dcgmHostname);
    if (!errStr.empty())
    {
        DcgmError d;
        DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_HOSTENGINE_CONN, d, errStr.c_str());
        PRINT_ERROR("%s", "%s", d.GetMessage().c_str());
        AddError(d);
        Cleanup();
        return false;
    }
    m_dcgmRecorderInitialized = true;
 
    fieldIds.push_back(DCGM_FI_DEV_POWER_USAGE);
    fieldIds.push_back(DCGM_FI_DEV_SM_CLOCK);
    fieldIds.push_back(DCGM_FI_DEV_MEM_CLOCK);
    fieldIds.push_back(DCGM_FI_DEV_GPU_TEMP);
    fieldIds.push_back(DCGM_FI_DEV_POWER_VIOLATION);
    fieldIds.push_back(DCGM_FI_DEV_THERMAL_VIOLATION);
    fieldIds.push_back(DCGM_FI_DEV_ECC_SBE_VOL_TOTAL);
    fieldIds.push_back(DCGM_FI_DEV_ECC_DBE_VOL_TOTAL);
    fieldIds.push_back(DCGM_FI_DEV_XID_ERRORS);
    fieldIds.push_back(DCGM_FI_DEV_CLOCK_THROTTLE_REASONS);
    fieldIds.push_back(DCGM_FI_DEV_GPU_UTIL);
    errStr = m_dcgmRecorder.AddWatches(fieldIds, gpuList, false, "diagnostic_field_group", "diagnostic_group",
                                        m_testDuration);
    if (!errStr.empty())
    {
        DcgmError d;
        DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_CANNOT_SET_WATCHES, d, errStr.c_str());
        AddError(d);
        Cleanup();
        return false;
    }
 
    /* Catch any runtime errors */
    try
    {
        /* Create and initialize worker threads */
        for (size_t i = 0; i < m_device.size(); i++)
        {
            PRINT_DEBUG("%d", "Creating worker thread for lwmlIndex %d", (int)gpuList[i]);
            workerThreads[i] = new GpuBurnWorker<T>(m_device[i], *this, m_useDoubles, m_testDuration,
                                                    m_dcgmRecorder);
            // initialize the worker
            st = workerThreads[i]->initBuffers();
            if (st)
            {
                // Couldn't initialize the worker - stop all launched workers and exit
                for (size_t j = 0; j <= i; j++)
                {
                    // Ask each worker to stop and wait up to 3 seconds for the thread to stop
                    st = workerThreads[i]->StopAndWait(3000);
                    if (st)
                    {
                        // Thread did not stop
                        workerThreads[i]->Kill();
                    }
                    delete(workerThreads[i]);
                    workerThreads[i] = NULL;
                }
                Cleanup();
                std::stringstream ss;
                ss << "Unable to initialize test for GPU "  << m_device[i]->dcgmDeviceIndex << ". Aborting.";
                std::string error = ss.str();
                PRINT_ERROR("%s", "%s", error.c_str());
                return false;
            }
            // Start the worker thread
            workerThreads[i]->Start();
            activeThreadCount++;
        }
        /* Wait for all workers to finish */
        while (activeThreadCount > 0)
        {
            activeThreadCount = 0;
 
            for (size_t i = 0; i < m_device.size(); i++)
            {
                st = workerThreads[i]->Wait(1000);
                if (st)
                {
                    activeThreadCount++;
                }
            }
 
            if (timeCount % 5 == 0)
            {
                progressOut->updatePluginProgress((unsigned int)(timeCount/m_testDuration * 100), false);
            }
            timeCount++;
        }
    }
    catch (const std::runtime_error &e)
    {
        PRINT_ERROR("%s", "Caught exception %s", e.what());
        DcgmError d;
        DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_INTERNAL, d, e.what());
        AddError(d);
        SetResult(LWVS_RESULT_FAIL);
        for (size_t i = 0; i < m_device.size(); i++)
        {
            // If a worker was not initialized, we skip over it (e.g. we caught a bad_alloc exception)
            if (workerThreads[i] == NULL)
            {
                continue;
            }
            // Ask each worker to stop and wait up to 3 seconds for the thread to stop
            st = workerThreads[i]->StopAndWait(3000);
            if (st)
            {
                // Thread did not stop
                workerThreads[i]->Kill();
            }
            delete(workerThreads[i]);
            workerThreads[i] = NULL;
        }
        Cleanup();
        // Let the TestFramework report the exception information.
        throw;
    }
 
    // Get the earliest stop time, read information from each thread, and then delete the threads
    earliestStopTime = INT64_MAX;
    for (size_t i = 0; i < m_device.size(); i++)
    { 
        errorCount.push_back(workerThreads[i]->getTotalErrors());
        operationsPerformed.push_back(workerThreads[i]->getTotalOperations());
 
        earliestStopTime = MIN(earliestStopTime, workerThreads[i]->getStopTime());
 
        delete(workerThreads[i]);
        workerThreads[i] = NULL;
    }
 
    progressOut->updatePluginProgress(timeCount, true);
    // Don't check pass / fail if early stop was requested
    if (main_should_stop)
    {
        Cleanup();
        return false; // Caller will check for main_should_stop and set the test skipped
    }
 
    for (size_t i = 0; i < operationsPerformed.size(); i++)
    {
        // Callwlate the approximate gigaflops and record it as info for this test
        double gigaflops = operationsPerformed[i] * OPS_PER_MUL / (1024 * 1024 * 1024) / m_testDuration;
        char buf[1024];
        snprintf(buf, sizeof(buf), "GPU %u callwlated at approximately %.2f gigaflops during this test",
                m_device[i]->lwmlDeviceIndex, gigaflops);
        AddInfoVerboseForGpu(m_device[i]->dcgmDeviceIndex, buf);
    }
 
    /* Set pass/failed status. 
     * Do NOT return false after this point as the test has run without issues. (Test failures do not count as issues).
     */
    testPassed = CheckPassFail(startTime, earliestStopTime, errorCount);
 
    if (!testPassed || !lwvsCommon.statsOnlyOnFail)
    {
        if (!logFileName.empty())
        {
            st = m_dcgmRecorder.WriteToFile(logFileName, logFileType, startTime);
            if (st)
            {
                PRINT_ERROR("%s", "Unable to write to log file %s", logFileName.c_str());
                std::string error = "There was an error writing test statistics to file '";
                error += logFileName + "'.";
                AddInfo(error);
                // not returning or cleaning up here since it is done at the end of the method
            }
        }
    }
    
    Cleanup();
    return true;
}


/****************************************************************************/
/*
 * GpuBurnWorker implementation.
 */
/****************************************************************************/
template <class T>
GpuBurnWorker<T>::GpuBurnWorker(GpuBurnDevice *device, GpuBurnPlugin &plugin, bool useDoubles, double testDuration,
                                DcgmRecorder &dr):

        m_device(device), m_plugin(plugin), m_useDoubles(useDoubles), m_testDuration(testDuration), 
        m_lwblas(0), m_error(0), m_Cdata(0), m_Adata(0), m_Bdata(0), m_faultyElemData(0), m_A(0), m_B(0), 
        m_stopTime(0), m_totalOperations(0), m_totalErrors(0), m_dcgmRecorder(dr)
{
}

/****************************************************************************/
/*
 * Macro for checking LWCA/LWBLAS errors and returning -1 in case of errors.
 * For use by GpuBurnWorker only.
 */
/****************************************************************************/
#define CHECK_LWDA_ERROR(callName, lwSt)                                                \
    if (lwSt != LWDA_SUCCESS)                                                           \
    {                                                                                   \
        LOG_LWDA_ERROR_FOR_PLUGIN(&m_plugin, callName, lwSt, m_device->dcgmDeviceIndex);\
        return -1;                                                                      \
    }                                                                                   \
    else                                                                                \
        (void)0

#define CHECK_LWBLAS_ERROR(callName, lwbSt)                                                 \
    if (lwbSt != LWBLAS_STATUS_SUCCESS)                                                     \
    {                                                                                       \
        LOG_LWBLAS_ERROR_FOR_PLUGIN(&m_plugin, callName, lwbSt, m_device->dcgmDeviceIndex); \
        return -1;                                                                          \
    }                                                                                       \
    else                                                                                    \
        (void)0


/****************************************************************************/
template <class T>
GpuBurnWorker<T>::~GpuBurnWorker()
{
    bind();
    LWresult lwSt;
    if (m_Adata)
    {
        lwSt = lwMemFree(m_Adata);
        if (lwSt != LWDA_SUCCESS)
        {
            LOG_LWDA_ERROR_FOR_PLUGIN(&m_plugin, "lwMemFree", lwSt, m_device->dcgmDeviceIndex);
        }
    }
    if (m_Bdata)
    {
        lwSt = lwMemFree(m_Bdata);
        if (lwSt != LWDA_SUCCESS)
        {
            LOG_LWDA_ERROR_FOR_PLUGIN(&m_plugin, "lwMemFree", lwSt, m_device->dcgmDeviceIndex);
        }
    }
    if (m_Cdata)
    {
        lwSt = lwMemFree(m_Cdata);
        if (lwSt != LWDA_SUCCESS)
        {
            LOG_LWDA_ERROR_FOR_PLUGIN(&m_plugin, "lwMemFree", lwSt, m_device->dcgmDeviceIndex);
        }
    }

    if (m_A)
    {
        free(m_A);
        m_A = NULL;
    }

    if (m_B)
    {
        free(m_B);
        m_B = NULL;
    }

    if (m_lwblas)
    {
        lwblasDestroy(m_lwblas);
    }
}

/****************************************************************************/
template <class T>
int GpuBurnWorker<T>::bind()
{
    /* Make sure we are pointing at the right device */
    lwdaSetDevice(m_device->lwDevice);

    /* Grab the context from the runtime */
    CHECK_LWDA_ERROR("lwCtxGetLwrrent", lwCtxGetLwrrent(&m_device->lwContext));

    if (!m_device->lwContext)
    {
        DcgmError d;
        DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_LWDA_UNBOUND, d, m_device->lwDevice);
        PRINT_ERROR("%s", "%s", d.GetMessage().c_str());
        m_plugin.AddErrorForGpu(m_device->dcgmDeviceIndex, d);
        return -1;
    }
    else
    {
        CHECK_LWDA_ERROR("lwCtxSetLwrrent", lwCtxSetLwrrent(m_device->lwContext));
    }
    return 0;

}

/****************************************************************************/
template <class T>
size_t GpuBurnWorker<T>::availMemory(int &st)
{
    int ret;
    ret = bind();
    if (ret)
    {
        st = -1;
        return 0;
    }
    size_t freeMem;
    size_t totalMem;
    LWresult lwSt = lwMemGetInfo(&freeMem, &totalMem);
    if (lwSt != LWDA_SUCCESS)
    {
        LOG_LWDA_ERROR_FOR_PLUGIN(&m_plugin, "lwMemGetInfo", lwSt, m_device->dcgmDeviceIndex);
        st = -1;
        return 0;
    }
    return freeMem;
}

/****************************************************************************/
template <class T>
int GpuBurnWorker<T>::initBuffers()
{
    allocBuffers();

    int st = bind();
    if (st)
    {
        return st;
    }

    size_t useBytes = (size_t)((double)availMemory(st)*USEMEM);
    if (st)
    {
        return st;
    }
    size_t resultSize = sizeof(T)*SIZE*SIZE;
    m_iters = (useBytes - 2*resultSize)/resultSize; // We remove A and B sizes
    CHECK_LWDA_ERROR("lwMemAlloc", lwMemAlloc(&m_Cdata, m_iters*resultSize));
    CHECK_LWDA_ERROR("lwMemAlloc", lwMemAlloc(&m_Adata, resultSize));
    CHECK_LWDA_ERROR("lwMemAlloc", lwMemAlloc(&m_Bdata, resultSize));
    
    CHECK_LWDA_ERROR("lwMemAlloc", lwMemAlloc(&m_faultyElemData, sizeof(int)));

    // Populating matrices A and B
    CHECK_LWDA_ERROR("lwMemcpyHtoD", lwMemcpyHtoD(m_Adata, m_A, resultSize));
    CHECK_LWDA_ERROR("lwMemcpyHtoD", lwMemcpyHtoD(m_Bdata, m_B, resultSize));

    return initCompareKernel();
}

/****************************************************************************/
template <class T>
int GpuBurnWorker<T>::initCompareKernel()
{
    CHECK_LWDA_ERROR("lwModuleLoadData", lwModuleLoadData(&m_module, (const char *)gpuburn_ptx_string));
    CHECK_LWDA_ERROR("lwModuleGetFunction", lwModuleGetFunction(&m_function, m_module,
                m_useDoubles ? "compareDP64" : "compareFP64"));

    CHECK_LWDA_ERROR("lwFuncSetCacheConfig", lwFuncSetCacheConfig(m_function, LW_FUNC_CACHE_PREFER_L1));
    CHECK_LWDA_ERROR("lwParamSetSize", lwParamSetSize(m_function, __alignof(T*) + __alignof(int*) + __alignof(size_t)));
    CHECK_LWDA_ERROR("lwParamSetv", lwParamSetv(m_function, 0, &m_Cdata, sizeof(T*)));
    CHECK_LWDA_ERROR("lwParamSetv", lwParamSetv(m_function, __alignof(T*), &m_faultyElemData, sizeof(T*)));
    CHECK_LWDA_ERROR("lwParamSetv", lwParamSetv(m_function, __alignof(T*) + __alignof(int*), &m_iters, sizeof(size_t)));

    CHECK_LWDA_ERROR("lwFuncSetBlockShape", lwFuncSetBlockShape(m_function, g_blockSize, g_blockSize, 1));
    return 0;
}

/****************************************************************************/
template <class T>
int GpuBurnWorker<T>::compare()
{
    int faultyElems;
    CHECK_LWDA_ERROR("lwMemsetD32", lwMemsetD32(m_faultyElemData, 0, 1));
    CHECK_LWDA_ERROR("lwLaunchGrid", lwLaunchGrid(m_function, SIZE/g_blockSize, SIZE/g_blockSize));
    CHECK_LWDA_ERROR("lwMemcpyDtoH", lwMemcpyDtoH(&faultyElems, m_faultyElemData, sizeof(int)));
    if (faultyElems)
    {
        m_error += (long long int)faultyElems;
    }

    #if 0 /* DON'T CHECK IN ENABLED. Generate an API error */
    checkError(LWDA_ERROR_LAUNCH_TIMEOUT, "Injected error.");
    #endif
    return 0;
}

/****************************************************************************/
template <class T>
int GpuBurnWorker<T>::compute()
{
    int st = bind();
    if (st)
    {
        return -1;
    }
    static const float alpha = 1.0f;
    static const float beta = 0.0f;
    static const double alphaD = 1.0;
    static const double betaD = 0.0;

    for (size_t i = 0; i < m_iters; i++)
    {
        if (m_useDoubles)
        {
            CHECK_LWBLAS_ERROR("lwblasDgemm", 
                                lwblasDgemm(m_lwblas, LWBLAS_OP_N, LWBLAS_OP_N, SIZE, SIZE, SIZE, &alphaD,
                                            (const double*)m_Adata, SIZE, (const double*)m_Bdata, SIZE, &betaD,
                                            (double*)m_Cdata + i*SIZE*SIZE, SIZE));
        }
        else
        {
            CHECK_LWBLAS_ERROR("lwblasSgemm", 
                                lwblasSgemm(m_lwblas, LWBLAS_OP_N, LWBLAS_OP_N, SIZE, SIZE, SIZE, &alpha, 
                                            (const float*)m_Adata, SIZE, (const float*)m_Bdata, SIZE, &beta, 
                                            (float*)m_Cdata + i*SIZE*SIZE, SIZE));
        }
    }
    return 0;
}

/****************************************************************************/
template <class T>
void GpuBurnWorker<T>::run()
{
    double startTime;
    timelib64_t lastFailureCheckTime = 0;  /* last time we checked for failures */
    timelib64_t now;
    double iterEnd;
    std::string gflopsKey(PERF_STAT_NAME);
    unsigned long long failCheckIntervalUsec = lwvsCommon.failCheckInterval * 1000000;

    int st = bind();
    if (st)
    {
        m_stopTime = timelib_usecSince1970();
        return;
    }

    lwblasStatus_t lwbSt = lwblasCreate(&m_lwblas);
    if (lwbSt != LWBLAS_STATUS_SUCCESS)
    {
        LOG_LWBLAS_ERROR_FOR_PLUGIN(&m_plugin, "lwblasCreate", lwbSt, 0, 0, false);
        m_stopTime = timelib_usecSince1970();
        return;
    }

    if (m_device->lwvsDevice->SetCpuAffinity())
    {
        char buf[1024];
        snprintf(buf, sizeof(buf), "Top performance cannot be guaranteed for GPU %u because "
                    "we could not set cpu affinity", m_device->dcgmDeviceIndex);
        m_plugin.AddInfoVerboseForGpu(m_device->dcgmDeviceIndex, buf);
    }
    startTime = timelib_dsecSince1970();
    lastFailureCheckTime = timelib_usecSince1970();
    std::vector<DcgmError> errorList;
    
    do
    {
        double iterStart = timelib_dsecSince1970();
        // Clear previous error counts
        m_error = 0;

        // Perform the callwlations and check the results
        st = compute();
        if (st)
        {
            break;
        }
        st = compare();
        if (st)
        {
            break;
        }

        // Save the error and work totals
        m_totalErrors += m_error;
        m_totalOperations += m_iters;
        iterEnd = timelib_dsecSince1970();

        double gflops = m_iters * OPS_PER_MUL / (1024 * 1024 * 1024) / (iterEnd - iterStart);
        m_dcgmRecorder.SetGpuStat(m_device->lwmlDeviceIndex, gflopsKey, gflops);

        /* Time to check for failure? */
        now = timelib_usecSince1970();
        if (lwvsCommon.failEarly && now - lastFailureCheckTime > failCheckIntervalUsec)
        {
            bool result = m_plugin.CheckPassFailSingleGpu(m_device, errorList, lastFailureCheckTime, now, false);
            if (!result)
            {
                // Stop the test because a failure oclwrred
                PRINT_DEBUG("%d", "Test failure detected for GPU %d. Stopping test early.",
                            m_device->dcgmDeviceIndex);
                break;
            }
            lastFailureCheckTime = now;
        }
    } while (iterEnd - startTime < m_testDuration && !ShouldStop());
    m_stopTime = timelib_usecSince1970();
}


extern "C"
{
    Plugin *maker()
    {
        return new GpuBurnPlugin;
    }
    class proxy
    {
    public:
        proxy()
            {
                factory[GPUBURN_PLUGIN_WL_NAME] = maker;
            }
    };
    proxy p;
}
