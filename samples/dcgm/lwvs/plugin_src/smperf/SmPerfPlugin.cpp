#define __STDC_LIMIT_MACROS
#include <stdint.h>

#include "SmPerfPlugin.h"
#include "LwvsThread.h"
#include "lwml.h"
#include "logging.h"
#include "lwda_runtime_api.h"
#include "PluginStrings.h"

/*****************************************************************************/
SmPerfPlugin::SmPerfPlugin(): m_lwmlInitialized(false), m_dcgmRecorderInitialized(false), m_dcgmCommErrorOclwrred(false)
{
    TestParameters *tp;

    m_infoStruct.name = SMPERF_PLUGIN_NAME;
    m_infoStruct.shortDescription = "This plugin will keep the SMs on the list of GPUs at a constant stress level.";
    m_infoStruct.testGroups = "Perf";
    m_infoStruct.selfParallel = true;
    m_infoStruct.logFileTag = SMPERF_PLUGIN_LF_NAME;

    /* Populate default test parameters */
    tp = new TestParameters();
    tp->AddString(PS_RUN_IF_GOM_ENABLED, "False");
    tp->AddString(SMPERF_STR_USE_DGEMM, "True");
    tp->AddString(SMPERF_STR_IS_ALLOWED, "False");
    tp->AddDouble(SMPERF_STR_TEST_DURATION, 90.0, 1.0, 86400.0);
    tp->AddDouble(SMPERF_STR_TARGET_PERF, 100.0, 1.0, 100000.0);
    tp->AddDouble(SMPERF_STR_TARGET_PERF_MIN_RATIO, 0.95, 0.5, 1.0);
    tp->AddDouble(SMPERF_STR_TEMPERATURE_MAX, 100.0, 30.0, 120.0);
    tp->AddDouble(SMPERF_STR_MAX_MEMORY_CLOCK, 0.0, 0.0, 100000.0);
    tp->AddDouble(SMPERF_STR_MAX_GRAPHICS_CLOCK, 0.0, 0.0, 100000.0);
    tp->AddDouble(SMPERF_STR_SBE_ERROR_THRESHOLD, DCGM_FP64_BLANK, 0.0, DCGM_FP64_BLANK);
    m_infoStruct.defaultTestParameters = tp;
}

/*****************************************************************************/
SmPerfPlugin::~SmPerfPlugin()
{
    /* Just call our cleanup function */
    Cleanup();
}

/*****************************************************************************/
void SmPerfPlugin::Cleanup(void)
{
    /* This code should be callable multiple times since exit paths and the
     * destructor will call this */
    SmPerfDevice *smDevice = 0;

    for (size_t deviceIdx = 0; deviceIdx < m_device.size(); deviceIdx++)
    {
        smDevice = m_device[deviceIdx];
        lwdaSetDevice(smDevice->lwdaDeviceIdx);
        delete smDevice;
    }

    m_device.clear();

    /* Do not delete m_testParameters. We don't own it */

    if (m_dcgmRecorderInitialized)
    {
        m_dcgmRecorder.Shutdown();
    }
    m_dcgmRecorderInitialized = false;

    /* Unload our lwca context for each gpu in the current process. We enumerate all GPUs because
       lwca opens a context on all GPUs, even if we don't use them */
    int lwdaDeviceCount;
    lwdaError_t lwSt;
    lwSt = lwdaGetDeviceCount(&lwdaDeviceCount);
    if (lwSt == lwdaSuccess)
    {
        for (int deviceIdx = 0; deviceIdx < lwdaDeviceCount; deviceIdx++)
        {
            lwdaSetDevice(deviceIdx);
            lwdaDeviceReset();
        }
    }

    if (m_lwmlInitialized)
    {
        lwmlShutdown();
    }
    m_lwmlInitialized = false;
}

/*****************************************************************************/
bool SmPerfPlugin::LwmlInit(const std::vector<unsigned int> &gpuList)
{
    int i, st, gpuListIndex;
    lwmlReturn_t lwmlSt;
    lwmlEnableState_t autoBoostEnabled, defaultAutoBoostEnabled;
    SmPerfDevice *smDevice = 0;
    char buf[256] = {0};
    lwmlPciInfo_t pciInfo;
    lwdaError_t lwSt;
    lwvsReturn_t lwvsReturn;

    m_device.reserve(gpuList.size());

    /* Attach to every device by index and reset it in case a previous plugin
       didn't clean up after itself */
    int lwdaDeviceCount, deviceIdx;
    lwSt = lwdaGetDeviceCount(&lwdaDeviceCount);
    if (lwSt == lwdaSuccess)
    {
        for (deviceIdx = 0; deviceIdx < lwdaDeviceCount; deviceIdx++)
        {
            lwdaSetDevice(deviceIdx);
            lwdaDeviceReset();
        }
    }

    lwmlSt = lwmlInit();
    if (lwmlSt != LWML_SUCCESS)
    {
        lwvsCommon.errorMask |= SMPERF_ERR_GENERIC;
        DcgmError d;
        DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_LWML_API, d, "lwmlInit", lwmlErrorString(lwmlSt));
        AddError(d);
        PRINT_ERROR("%s", "%s", d.GetMessage().c_str());
        return false;
    }
    m_lwmlInitialized = true;

    m_device.reserve(gpuList.size());
    for (gpuListIndex = 0; gpuListIndex < gpuList.size(); gpuListIndex++)
    {
        try
        {
            smDevice = new SmPerfDevice(gpuList[gpuListIndex], this);
        }
        catch (DcgmError &d)
        {
            AddErrorForGpu(gpuList[gpuListIndex], d);
            delete smDevice;
            return false;
        }

        /* At this point, we consider this GPU part of our set */
        m_device.push_back(smDevice);
    }
    return true;
}

/*****************************************************************************/
int SmPerfPlugin::LwdaInit(void)
{
    int i, j, count, valueSize;
    size_t arrayByteSize, arrayNelem;
    lwdaError_t lwSt;
    lwblasStatus_t lwbSt;
    unsigned int hostAllocFlags = 0;

    lwSt = lwdaGetDeviceCount(&count);
    if (lwSt != lwdaSuccess)
    {
        LOG_LWDA_ERROR("lwdaGetDeviceCount", lwSt, 0, 0, false);
        return -1;
    }

    if (m_useDgemm)
    {
        valueSize = sizeof(double);
    }
    else
    {
        valueSize = sizeof(float);
    }

    arrayByteSize = valueSize * SMPERF_TEST_DIMENSION * SMPERF_TEST_DIMENSION;
    arrayNelem = SMPERF_TEST_DIMENSION * SMPERF_TEST_DIMENSION;

    /* Do per-device initialization */
    for (size_t deviceIdx = 0; deviceIdx < m_device.size(); deviceIdx++)
    {
        SmPerfDevice *device = m_device[deviceIdx];

        if (device->lwdaDeviceIdx < 0 || device->lwdaDeviceIdx >= count)
        {
            PRINT_ERROR("%d %d", "Invalid lwca device index %d >= count of %d or < 0",
                        device->lwdaDeviceIdx, count);
            return -1;
        }

        /* Make all subsequent lwca calls link to this device */
        lwdaSetDevice(device->lwdaDeviceIdx);

        lwSt = lwdaGetDeviceProperties(&device->lwdaDevProp, device->lwdaDeviceIdx);
        if (lwSt != lwdaSuccess)
        {
            LOG_LWDA_ERROR("lwdaGetDeviceProperties", lwSt, device->dcgmDeviceIndex);
            return -1;
        }

        /* Fill the arrays with random values */
        srand(time(NULL));

        lwSt = lwdaHostAlloc(&device->hostA, arrayByteSize, hostAllocFlags);
        if (lwSt != lwdaSuccess)
        {
            lwvsCommon.errorMask |= SMPERF_ERR_GENERIC;
            LOG_LWDA_ERROR("lwdaHostAlloc", lwSt, device->dcgmDeviceIndex, arrayByteSize);
            return -1;
        }
        lwSt = lwdaHostAlloc(&device->hostB, arrayByteSize, hostAllocFlags);
        if (lwSt != lwdaSuccess)
        {
            lwvsCommon.errorMask |= SMPERF_ERR_GENERIC;
            LOG_LWDA_ERROR("lwdaHostAlloc", lwSt, device->dcgmDeviceIndex, arrayByteSize);
            return -1;
        }
        lwSt = lwdaHostAlloc(&device->hostC, arrayByteSize, hostAllocFlags);
        if (lwSt != lwdaSuccess)
        {
            lwvsCommon.errorMask |= SMPERF_ERR_GENERIC;
            LOG_LWDA_ERROR("lwdaHostAlloc", lwSt, device->dcgmDeviceIndex, arrayByteSize);
            return -1;
        }

        if (m_useDgemm)
        {
            double *doubleHostA = (double *)device->hostA;
            double *doubleHostB = (double *)device->hostB;
            double *doubleHostC = (double *)device->hostC;

            for (j=0; j<arrayNelem; j++)
            {
                doubleHostA[j] = (double)rand() / 100.0;
                doubleHostB[j] = (double)rand() / 100.0;
                doubleHostC[j] = 0.0;
            }
        }
        else
        {
            /* sgemm */
            float *floatHostA = (float *)device->hostA;
            float *floatHostB = (float *)device->hostB;
            float *floatHostC = (float *)device->hostC;

            for (j=0; j<arrayNelem; j++)
            {
                floatHostA[j] = (float)rand() / 100.0;
                floatHostB[j] = (float)rand() / 100.0;
                floatHostC[j] = 0.0;
            }
        }

        /* Initialize lwblas */
        lwbSt = lwblasCreate(&device->lwblasHandle);
        if (lwbSt != LWBLAS_STATUS_SUCCESS)
        {
            lwvsCommon.errorMask |= SMPERF_ERR_GENERIC;
            LOG_LWBLAS_ERROR("lwblasCreate", lwbSt, device->dcgmDeviceIndex);
            return -1;
        }
        PRINT_DEBUG("%d %p", "lwblasCreate lwdaDeviceIdx %d, handle %p",
                    device->lwdaDeviceIdx, device->lwblasHandle);
        device->allocatedLwblasHandle = 1;

        /* Allocate device memory */
        lwSt = lwdaMalloc((void **)&device->deviceA, arrayByteSize);
        if (lwSt != lwdaSuccess)
        {
            lwvsCommon.errorMask |= SMPERF_ERR_GENERIC;
            LOG_LWDA_ERROR("lwdaMalloc", lwSt, device->dcgmDeviceIndex, arrayByteSize);
            return -1;
        }
        lwSt = lwdaMalloc((void **)&device->deviceB, arrayByteSize);
        if (lwSt != lwdaSuccess)
        {
            lwvsCommon.errorMask |= SMPERF_ERR_GENERIC;
            LOG_LWDA_ERROR("lwdaMalloc", lwSt, device->dcgmDeviceIndex, arrayByteSize);
            return -1;
        }
        lwSt = lwdaMalloc((void **)&device->deviceC, arrayByteSize);
        if (lwSt != lwdaSuccess)
        {
            lwvsCommon.errorMask |= SMPERF_ERR_GENERIC;
            LOG_LWDA_ERROR("lwdaMalloc", lwSt, device->dcgmDeviceIndex, arrayByteSize);
            return -1;
        }
    }

    return 0;
}

void SmPerfPlugin::Go(TestParameters *testParameters, const std::vector<unsigned int> &gpuList)
{
    bool result;
    InitializeForGpuList(gpuList);

    if (!testParameters->GetBoolFromString(SMPERF_STR_IS_ALLOWED))
    {
        DcgmError d;
        DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_TEST_DISABLED, d, SMPERF_PLUGIN_NAME);
        AddError(d);
        SetResult(LWVS_RESULT_SKIP);
        return;
    }

    /* Cache test parameters */
    m_testParameters = testParameters; // DO NOT DELETE
    m_testDuration = testParameters->GetDouble(SMPERF_STR_TEST_DURATION);
    m_targetPerf = testParameters->GetDouble(SMPERF_STR_TARGET_PERF);
    m_useDgemm = testParameters->GetBoolFromString(SMPERF_STR_USE_DGEMM);
    m_sbeFailureThreshold = testParameters->GetDouble(SMPERF_STR_SBE_ERROR_THRESHOLD);

    result = RunTest(gpuList);
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

/*****************************************************************************/
bool SmPerfPlugin::CheckGpuPerf(SmPerfDevice *smDevice, std::vector<DcgmError> &errorList,
                                timelib64_t startTime, timelib64_t endTime)
{
    std::vector<dcgmTimeseriesInfo_t> data;
    int errorSt;

    data = m_dcgmRecorder.GetLwstomGpuStat(smDevice->lwmlDeviceIndex, PERF_STAT_NAME);
    if (data.size() == 0)
    {
        DcgmError d;
        DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_CANNOT_GET_STAT, d, PERF_STAT_NAME, smDevice->dcgmDeviceIndex);
        errorList.push_back(d);
        return false;
    }

    double maxVal = 0.0;
    double avg = 0.0;

    for (size_t i = 0; i < data.size(); i++)
    {
        avg += data[i].val.fp64;
        if (data[i].val.fp64 > maxVal)
        {
            maxVal = data[i].val.fp64;
        }
    }
    avg = avg / data.size();

    RecordObservedMetric(smDevice->lwmlDeviceIndex, SMPERF_STR_TARGET_PERF, maxVal);

    double minRatio = m_testParameters->GetDouble(SMPERF_STR_TARGET_PERF_MIN_RATIO);
    if (maxVal < minRatio * m_targetPerf)
    {
        lwvsCommon.errorMask |= SMPERF_ERR_GENERIC;
        DcgmError d;
        DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_STRESS_LEVEL, d, maxVal, m_targetPerf, smDevice->dcgmDeviceIndex);
        std::string utilNote = m_dcgmRecorder.GetGpuUtilizationNote(smDevice->dcgmDeviceIndex, startTime, endTime);
        if (utilNote.empty() == false)
        {
            d.AddDetail(utilNote);
        }

        errorList.push_back(d);
        return false;
    }

    std::stringstream ss;
    ss.setf(std::ios::fixed, std::ios::floatfield);
    ss.precision(0);
    ss << "GPU " << smDevice->lwmlDeviceIndex << " relative stress level:\t" << avg;
    AddInfoVerboseForGpu(smDevice->dcgmDeviceIndex, ss.str());
    return true;
}

/*****************************************************************************/
bool SmPerfPlugin::CheckGpuTemperature(SmPerfDevice *smDevice, std::vector<DcgmError> &errorList,
                                       timelib64_t startTime, timelib64_t earliestStopTime, bool testFinished)
{
    std::string infoMsg;
    long long maxTemp = static_cast<long long>(m_testParameters->GetDouble(SMPERF_STR_TEMPERATURE_MAX));
    long long highTempObserved = 0;
    int st = m_dcgmRecorder.CheckGpuTemperature(smDevice->dcgmDeviceIndex, errorList, maxTemp, infoMsg, startTime,
                                                 earliestStopTime, highTempObserved);

    if (testFinished && highTempObserved != 0)
    {
        RecordObservedMetric(smDevice->lwmlDeviceIndex, SMPERF_STR_TEMPERATURE_MAX, highTempObserved);
    }

    if (st == DR_VIOLATION)
    {
        lwvsCommon.errorMask |= SMPERF_ERR_GENERIC;
    }
    // Check for a communication error
    else if (st == DR_COMM_ERROR)
    {
        m_dcgmCommErrorOclwrred = true;
    }

    if (testFinished)
    {
        AddInfoVerboseForGpu(smDevice->dcgmDeviceIndex, infoMsg);
    }

    return errorList.empty();
}

/*****************************************************************************/
bool SmPerfPlugin::CheckLwmlEvents(SmPerfDevice *smDevice, std::vector<DcgmError> &errorList,
                                   timelib64_t startTime, timelib64_t earliestStopTime)
{
    std::vector<unsigned short> fieldIds;
    std::vector<dcgmTimeseriesInfo_t> *thresholdsPtr = 0;
    std::vector<dcgmTimeseriesInfo_t> failureThresholds;

    // Only use SBE as a field id if there is a failure threshold, otherwise ignore this field
    if (DCGM_FP64_IS_BLANK(m_sbeFailureThreshold) == 0)
    {
        fieldIds.push_back(DCGM_FI_DEV_ECC_SBE_VOL_TOTAL);

        dcgmTimeseriesInfo_t dti;
        memset(&dti, 0, sizeof(dti));
        dti.val.i64 = static_cast<long long>(m_sbeFailureThreshold);
        dti.isInt = true;
        failureThresholds.push_back(dti);

        // Make the failure value 0 for the other two fields. This is not necessary if we are not checking
        // for SBE errors because DcgmRecorder::CheckErrorFields assumes a default threshold of 0 when
        // thresholdsPtr is NULL
        dti.val.i64 = 0;
        for (int i = 0; i < 2; i++)
        {
            failureThresholds.push_back(dti);
        }
        thresholdsPtr = &failureThresholds;
    }

    fieldIds.push_back(DCGM_FI_DEV_ECC_DBE_VOL_TOTAL);
    fieldIds.push_back(DCGM_FI_DEV_XID_ERRORS);

    int st = m_dcgmRecorder.CheckErrorFields(fieldIds, thresholdsPtr, smDevice->lwmlDeviceIndex,
                                             errorList, startTime, earliestStopTime);

    // Check for a communication error
    if (st == DR_COMM_ERROR)
    {
        m_dcgmCommErrorOclwrred = true;
    }

    // Return true if there are no errors, false otherwise
    return errorList.empty();
}

/*****************************************************************************/
bool SmPerfPlugin::CheckPassFailSingleGpu(SmPerfDevice *device, std::vector<DcgmError> &errorList,
                                         timelib64_t startTime, timelib64_t earliestStopTime, bool testFinished)
{
    DcgmLockGuard lock(&m_mutex); // prevent conlwrrent failure checks from workers
    bool result;
    int st;

    if (testFinished)
    {
        // This check is only performed once the test is finished
        result = CheckGpuPerf(device, errorList, startTime, earliestStopTime);
        if (!result || m_dcgmCommErrorOclwrred)
        {
            return false;
        }
    }
    /* Check GPU temperature against specified max temp */
    result = CheckGpuTemperature(device, errorList, startTime, earliestStopTime, testFinished);
    if (!result || m_dcgmCommErrorOclwrred)
    {
        return false;
    }

    st = m_dcgmRecorder.CheckThermalViolations(device->dcgmDeviceIndex, errorList, startTime, earliestStopTime);
    if (st == DR_VIOLATION)
    {
        return false;
    }
    st = m_dcgmRecorder.CheckForThrottling(device->dcgmDeviceIndex, startTime, errorList);
    if (st == DR_COMM_ERROR)
    {
        m_dcgmCommErrorOclwrred = true;
        return false;
    }
    else if (st == DR_VIOLATION)
    {
        return false;
    }

    result = CheckLwmlEvents(device, errorList, startTime, earliestStopTime);
    if (!result || m_dcgmCommErrorOclwrred)
    {
        return false;
    }

    return true;
}

/*****************************************************************************/
bool SmPerfPlugin::CheckPassFail(timelib64_t startTime, timelib64_t earliestStopTime)
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
        passed = CheckPassFailSingleGpu(m_device[i], errorList, startTime, earliestStopTime);
        CheckAndSetResult(this, m_gpuList, i, passed, errorList, allPassed, m_dcgmCommErrorOclwrred);
        if (m_dcgmCommErrorOclwrred)
        {
            /* No point in checking other GPUs until communication is restored */
            break;
        }
    }

    for (size_t i = 0; i < errorListAllGpus.size(); i++)
    {
        AddError(errorListAllGpus[i]);
    }

    return allPassed;
}

/*****************************************************************************/
class SmPerfWorker : public LwvsThread
{
private:
    SmPerfDevice    *m_device;          /* Which device this worker thread is running on */
    SmPerfPlugin    &m_plugin;          /* SmPerfPlugin for logging and failure checks */
    TestParameters  *m_testParameters;  /* Read-only test parameters */
    int             m_useDgemm;         /* Wheter to use dgemm (1) or sgemm (0) for operations */
    double          m_targetPerf;       /* Target stress in gflops */
    double          m_testDuration;     /* Target test duration in seconds */
    timelib64_t     m_stopTime;         /* Timestamp when run() finished */
    DcgmRecorder    &m_dcgmRecorder;

public:
    /*************************************************************************/
    SmPerfWorker(SmPerfDevice *device, SmPerfPlugin &plugin, TestParameters *tp, DcgmRecorder &dr);

    /*************************************************************************/
    virtual ~SmPerfWorker() /* Virtual to satisfy ancient compiler */
    {
    }

    /*************************************************************************/
    timelib64_t GetStopTime()
    {
        return m_stopTime;
    }

    /*************************************************************************/
    /*
     * Worker thread main.
     *
     */
    void run(void);

private:
    /*************************************************************************/
    /*
     * Do a single matrix multiplication operation.
     *
     * Returns 0 if OK
     *        <0 on error
     *
     */
    int DoOneMatrixMultiplication(float *floatAlpha, double *doubleAlpha, float *floatBeta, double *doubleBeta);
};

/****************************************************************************/
/*
 * SmPerfPlugin RunTest
 */
/*****************************************************************************/
bool SmPerfPlugin::RunTest(const std::vector<unsigned int> &gpuList)
{
    int st = 0, Nrunning = 0;
    bool testPassed = false;
    SmPerfWorker *workerThreads[SMPERF_MAX_DEVICES] = {0};
    bool outputStats = true;
    unsigned int timeCount = 0;
    timelib64_t earliestStopTime;
    timelib64_t startTime = timelib_usecSince1970();

    if (!LwmlInit(gpuList))
    {
        Cleanup();
        return false;
    }

    st = LwdaInit();
    if (st)
    {
        // The specific error has already been added to this plugin
        Cleanup();
        return false;
    }

    // Initialize our interface to DCGM
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

    /* Is binary logging enabled for this stat collection? */
    std::string logFileName = m_testParameters->GetString(PS_LOGFILE);
    int logFileType = (int)m_testParameters->GetDouble(PS_LOGFILE_TYPE);

    std::vector<unsigned short> fieldIds;
    fieldIds.push_back(DCGM_FI_DEV_POWER_USAGE);
    fieldIds.push_back(DCGM_FI_DEV_SM_CLOCK);
    fieldIds.push_back(DCGM_FI_DEV_MEM_CLOCK);
    fieldIds.push_back(DCGM_FI_DEV_ECC_SBE_VOL_TOTAL);
    fieldIds.push_back(DCGM_FI_DEV_ECC_DBE_VOL_TOTAL);
    fieldIds.push_back(DCGM_FI_DEV_XID_ERRORS);
    fieldIds.push_back(DCGM_FI_DEV_POWER_VIOLATION);
    fieldIds.push_back(DCGM_FI_DEV_THERMAL_VIOLATION);
    fieldIds.push_back(DCGM_FI_DEV_GPU_TEMP);
    fieldIds.push_back(DCGM_FI_DEV_CLOCK_THROTTLE_REASONS);
    fieldIds.push_back(DCGM_FI_DEV_GPU_UTIL);
    m_dcgmRecorder.AddWatches(fieldIds, gpuList, false, "sm_stress_field_group", "sm_stress_group",
                              m_testDuration);

    try /* Catch runtime errors */
    {
        /* Create and start all workers */
        for (size_t i = 0; i < m_device.size(); i++)
        {
            workerThreads[i] = new SmPerfWorker(m_device[i], *this, m_testParameters, m_dcgmRecorder);
            workerThreads[i]->Start();
            Nrunning++;
        }
        /* Wait for all workers to finish */
        while (Nrunning > 0)
        {
            Nrunning = 0;
            /* Just go in a round-robin loop around our workers until
             * they have all exited. These calls will return immediately
             * once they have all exited. Otherwise, they serve to keep
             * the main thread from sitting busy */
            for (size_t i = 0; i < m_device.size(); i++)
            {
                st = workerThreads[i]->Wait(1000);
                if (st)
                {
                    Nrunning++;
                }
            }
            if (timeCount % 5 == 0)
            {
                progressOut->updatePluginProgress((unsigned int)(timeCount/m_testDuration * 100), false);
            }
            timeCount++;
        }
    }
    catch (const std::exception &e)
    {
        DcgmError d;
        DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_INTERNAL, d, e.what());
        PRINT_ERROR("%s", "Caught exception %s", e.what());
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

    // Get the earliestStopTime and delete the threads
    earliestStopTime = INT64_MAX;
    for (size_t i=0; i<m_device.size(); i++)
    {
        earliestStopTime = MIN(earliestStopTime, workerThreads[i]->GetStopTime());
        delete(workerThreads[i]);
        workerThreads[i] = NULL;
    }

    progressOut->updatePluginProgress(timeCount, true);
    /* Don't check pass/fail if early stop was requested */
    if (main_should_stop)
    {
        Cleanup();
        return false; /* Caller will check for main_should_stop and set the test result appropriately */
    }

    /* Set pass/failed status. 
     * Do NOT return false after this point as the test has run without issues. (Test failures do not count as issues).
     */
    testPassed = CheckPassFail(startTime, earliestStopTime);

    if (testPassed && lwvsCommon.statsOnlyOnFail)
    {
        outputStats = false;
    }

    // Write telemetry to a file unless we don't need it
    if (logFileName.size() > 0 && outputStats)
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

    Cleanup();
    return true;
}


/****************************************************************************/
/*
 * SmPerfWorker implementation.
 */
/****************************************************************************/
SmPerfWorker::SmPerfWorker(SmPerfDevice *device, SmPerfPlugin &plugin, TestParameters *tp, DcgmRecorder &dr):
        m_device(device), m_plugin(plugin), m_testParameters(tp), m_dcgmRecorder(dr)
{
    m_useDgemm = m_testParameters->GetBoolFromString(SMPERF_STR_USE_DGEMM);
    m_targetPerf = m_testParameters->GetDouble(SMPERF_STR_TARGET_PERF);
    m_testDuration = m_testParameters->GetDouble(SMPERF_STR_TEST_DURATION);
}

/*****************************************************************************/
int SmPerfWorker::DoOneMatrixMultiplication(float *floatAlpha, double *doubleAlpha, float *floatBeta,
                                            double *doubleBeta)
{
    lwblasStatus_t lwblasSt;

    if (m_useDgemm)
    {
        lwblasSt = lwblasDgemm(m_device->lwblasHandle, LWBLAS_OP_N, LWBLAS_OP_N, SMPERF_TEST_DIMENSION,
                                SMPERF_TEST_DIMENSION, SMPERF_TEST_DIMENSION, doubleAlpha,
                                (double *)m_device->deviceA, SMPERF_TEST_DIMENSION,
                                (double *)m_device->deviceB, SMPERF_TEST_DIMENSION, doubleBeta,
                                (double *)m_device->deviceC, SMPERF_TEST_DIMENSION);
        if (lwblasSt != LWBLAS_STATUS_SUCCESS)
        {
            LOG_LWBLAS_ERROR_FOR_PLUGIN(&m_plugin, "lwblasDgemm", lwblasSt, m_device->dcgmDeviceIndex);
            DcgmLockGuard lock(&m_sync_mutex);
            lwvsCommon.errorMask |= SMPERF_ERR_GENERIC;
            return -1;
        }
    }
    else
    {
        lwblasSt = lwblasSgemm(m_device->lwblasHandle, LWBLAS_OP_N, LWBLAS_OP_N, SMPERF_TEST_DIMENSION,
                                SMPERF_TEST_DIMENSION, SMPERF_TEST_DIMENSION, floatAlpha,
                                (float *)m_device->deviceA, SMPERF_TEST_DIMENSION,
                                (float *)m_device->deviceB, SMPERF_TEST_DIMENSION, floatBeta,
                                (float *)m_device->deviceC, SMPERF_TEST_DIMENSION);
        if (lwblasSt != LWBLAS_STATUS_SUCCESS)
        {
            LOG_LWBLAS_ERROR_FOR_PLUGIN(&m_plugin, "lwblasSgemm", lwblasSt, m_device->dcgmDeviceIndex);
            DcgmLockGuard lock(&m_sync_mutex);
            lwvsCommon.errorMask |= SMPERF_ERR_GENERIC;
            return -1;
        }
    }

    return 0;
}

/*****************************************************************************/
void SmPerfWorker::run(void)
{
    double doubleAlpha, doubleBeta;
    float floatAlpha, floatBeta;
    double startTime;
    double lastPrintTime = 0.0;  /* last time we printed out the current perf */
    double lastFailureCheckTime = 0.0;  /* last time we checked for failures */
    double now, elapsed;
    long long Nops = 0, NopsBefore;
    lwdaError_t lwSt;
    int valueSize, arrayByteSize;
    int st;


    int opsPerResync = 100; /* Maximum ops to do before checking to see if the plugin should exit
                                early. Making this larger has less overhead for resyncing the clock
                                but makes the plugin less responsive to CTRL-C or per-second statistics */

    if (m_device->lwvsDevice->SetCpuAffinity())
    {
        char buf[1024];
        snprintf(buf, sizeof(buf), "Top performance cannot be guaranteed for GPU %u because "
                    "we could not set cpu affinity.", m_device->dcgmDeviceIndex);
        m_plugin.AddInfoVerboseForGpu(m_device->dcgmDeviceIndex, buf);
    }

    if (m_useDgemm)
    {
        valueSize = sizeof(double);
    }
    else
    {
        valueSize = sizeof(float);
    }
    arrayByteSize = valueSize * SMPERF_TEST_DIMENSION * SMPERF_TEST_DIMENSION;

    /* Copy the host arrays to the device arrays */
    lwSt = lwdaMemcpy(m_device->deviceA, m_device->hostA, arrayByteSize, lwdaMemcpyHostToDevice);
    if (lwSt != lwdaSuccess)
    {
        LOG_LWDA_ERROR_FOR_PLUGIN(&m_plugin, "lwdaMemcpy", lwSt, m_device->dcgmDeviceIndex, arrayByteSize);
        DcgmLockGuard lock(&m_sync_mutex);
        lwvsCommon.errorMask |= SMPERF_ERR_GENERIC;
        m_stopTime = timelib_usecSince1970();
        return;
    }
    lwSt = lwdaMemcpyAsync(m_device->deviceB, m_device->hostB, arrayByteSize, lwdaMemcpyHostToDevice);
    if (lwSt != lwdaSuccess)
    {
        LOG_LWDA_ERROR_FOR_PLUGIN(&m_plugin, "lwdaMemcpyAsync", lwSt, m_device->dcgmDeviceIndex, arrayByteSize);
        DcgmLockGuard lock(&m_sync_mutex);
        lwvsCommon.errorMask |= SMPERF_ERR_GENERIC;
        m_stopTime = timelib_usecSince1970();
        return;
    }

    double flopsPerOp = 2.0 * (double)SMPERF_TEST_DIMENSION *
                        (double)SMPERF_TEST_DIMENSION * (double)SMPERF_TEST_DIMENSION;
    double opsPerSec = m_targetPerf / (flopsPerOp / 1000000000.0);
    long long maxOpsSoFar;

    /* Set initial test values */
    doubleAlpha = 1.01 + ((double)(rand() % 100)/10.0);
    doubleBeta = 1.01 + ((double)(rand() % 100)/10.0);
    floatAlpha = (float)doubleAlpha;
    floatBeta = (float)doubleBeta;

    std::string gflopsKey;
    gflopsKey = std::string(PERF_STAT_NAME);

    /* Record some of our static callwlated parameters in case we need them for debugging */
    m_dcgmRecorder.SetGpuStat(m_device->lwmlDeviceIndex, "flops_per_op", flopsPerOp);
    m_dcgmRecorder.SetGpuStat(m_device->lwmlDeviceIndex, "try_ops_per_sec", opsPerSec);

    /* Lock to our assigned GPU */
    lwdaSetDevice(m_device->lwdaDeviceIdx);

    std::stringstream ss;
    ss << "Running for " << m_testDuration << " seconds";
    m_plugin.AddInfo(ss.str());
    startTime = timelib_dsecSince1970();
    lastPrintTime = startTime;
    lastFailureCheckTime = startTime;
    std::vector<DcgmError> errorList;

    now = timelib_dsecSince1970();

    while (now - startTime < m_testDuration && !ShouldStop())
    {
        now = timelib_dsecSince1970();
        elapsed = now - startTime;
        maxOpsSoFar = (long long)(elapsed * opsPerSec);
        NopsBefore = Nops;

        // If we're training, don't check maxOpsSoFar or we can't train past the target
        for (int i = 0; i < opsPerResync && (lwvsCommon.training || Nops < maxOpsSoFar); i++)
        {
            st = DoOneMatrixMultiplication(&floatAlpha, &doubleAlpha, &floatBeta, &doubleBeta);
            if (st)
            {
                // There was an error - stop test
                m_stopTime = timelib_usecSince1970();
                return;
            }
            Nops++;
        }

        /* If we didn't queue any work, sleep a bit so we don't busy wait */
        if (NopsBefore == Nops)
        {
            usleep(1000);
            now = timelib_dsecSince1970(); /* Resync now since we slept */
        }

        /* Time to print? */
        if (now - lastPrintTime > 1.0)
        {
            elapsed = now - startTime;
            double gflops = (flopsPerOp * (double)Nops)/(1000000000.0 * elapsed);

            m_dcgmRecorder.SetGpuStat(m_device->lwmlDeviceIndex, gflopsKey, gflops);
            m_dcgmRecorder.SetGpuStat(m_device->lwmlDeviceIndex, "nops_so_far", Nops);

            ss.str("");
            ss << "LwmlIdx " << m_device->lwmlDeviceIndex <<  ", ops " << Nops << ", gflops " << gflops;
            m_plugin.AddInfo(ss.str());
            lastPrintTime = now;
        }
        /* Time to check for failure? */
        if (lwvsCommon.failEarly && now - lastFailureCheckTime > lwvsCommon.failCheckInterval)
        {
            bool result = m_plugin.CheckPassFailSingleGpu(m_device, errorList, lastFailureCheckTime * 1000000,
                                                            now * 1000000, false);
            if (!result)
            {
                // Stop the test because a failure oclwrred
                PRINT_DEBUG("%d", "Test failure detected for GPU %d. Stopping test early.",
                            m_device->dcgmDeviceIndex);
                break;
            }
            lastFailureCheckTime = now;
        }
    }

    m_stopTime = timelib_usecSince1970();
    PRINT_DEBUG("%d %lld", "SmPerfWorker deviceIndex %d finished at %lld", m_device->lwmlDeviceIndex,
                (long long)m_stopTime);
}

extern "C" {
    Plugin *maker() {
        return new SmPerfPlugin;
    }
    class proxy {
    public:
        proxy()
        {
            factory["sm perf"] = maker;
        }
    };
    proxy p;
}
