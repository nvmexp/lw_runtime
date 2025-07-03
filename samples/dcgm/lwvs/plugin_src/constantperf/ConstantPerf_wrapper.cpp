#define __STDC_LIMIT_MACROS
#include <stdint.h>

#include "ConstantPerf_wrapper.h"
#include <stdio.h>
#include <iostream>
#include <stdexcept>
#include <sstream>

#include "lwda_runtime_api.h"
#include "PluginStrings.h"
#include "LwvsThread.h"

ConstantPerf::ConstantPerf(): m_testParameters(NULL), m_lwmlInitialized(false), m_dcgmCommErrorOclwrred(false),
                              m_dcgmRecorderInitialized(false)
{
    TestParameters *tp;

    m_infoStruct.name = CPERF_PLUGIN_NAME;
    m_infoStruct.shortDescription = "This plugin will keep the list of GPUs at a constant stress level.";
    m_infoStruct.testGroups = "Perf";
    m_infoStruct.selfParallel = true;
    m_infoStruct.logFileTag = CPERF_PLUGIN_LF_NAME;

    /* Populate default test parameters */
    tp = new TestParameters();
    tp->AddString(PS_RUN_IF_GOM_ENABLED, "False");
    tp->AddString(CPERF_STR_USE_DGEMM, "True");
    tp->AddString(CPERF_STR_IS_ALLOWED, "False");
    tp->AddDouble(CPERF_STR_TEST_DURATION, 120.0, 1.0, 86400.0);
    tp->AddDouble(CPERF_STR_TARGET_PERF, 100.0, 1.0, 100000.0);
    tp->AddDouble(CPERF_STR_TARGET_PERF_MIN_RATIO, 0.95, 0.5, 1.0);
    tp->AddDouble(CPERF_STR_LWDA_STREAMS_PER_GPU, CPERF_MAX_STREAMS_PER_DEVICE, 1.0, CPERF_MAX_STREAMS_PER_DEVICE);
    tp->AddDouble(CPERF_STR_LWDA_OPS_PER_STREAM, 100.0, 1.0, CPERF_MAX_CONLWRRENT_OPS_PER_STREAM);
    tp->AddDouble(CPERF_STR_TEMPERATURE_MAX, 100.0, 30.0, 120.0);
    tp->AddDouble(CPERF_STR_MAX_PCIE_REPLAYS, 160.0, 1.0, 1000000.0);
    tp->AddDouble(CPERF_STR_MAX_MEMORY_CLOCK, 0.0, 0.0, 100000.0);
    tp->AddDouble(CPERF_STR_MAX_GRAPHICS_CLOCK, 0.0, 0.0, 100000.0);
    tp->AddDouble(CPERF_STR_SBE_ERROR_THRESHOLD, DCGM_FP64_BLANK, 0.0, DCGM_FP64_BLANK);
    m_infoStruct.defaultTestParameters = tp;
}

ConstantPerf::~ConstantPerf()
{
    Cleanup();
}

/*****************************************************************************/
void ConstantPerf::Cleanup()
{
    size_t i;
    CPerfDevice *device = NULL;

    for (size_t deviceIdx = 0; deviceIdx < m_device.size(); deviceIdx++)
    {
        device = m_device[deviceIdx];
        lwdaSetDevice(device->lwdaDeviceIdx);

        /* Wait for all streams to finish */
        for (i = 0; i < device->Nstreams; i++)
        {
            lwdaStreamSynchronize(device->streams[i].lwdaStream);
        }
        delete device;
    }

    m_device.clear();

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
bool ConstantPerf::LwmlInit(const std::vector<unsigned int> &gpuList)
{
    lwmlReturn_t lwmlSt;
    lwdaError_t lwSt;
    CPerfDevice *cpDevice = 0;

    /* Attach to every device by index and reset it in case a previous plugin
       didn't clean up after itself */
    int lwdaDeviceCount;
    lwSt = lwdaGetDeviceCount(&lwdaDeviceCount);
    if (lwSt == lwdaSuccess)
    {
        for (int deviceIdx = 0; deviceIdx < lwdaDeviceCount; deviceIdx++)
        {
            lwdaSetDevice(deviceIdx);
            lwdaDeviceReset();
        }
    }

    lwmlSt = lwmlInit();
    if (lwmlSt != LWML_SUCCESS)
    {
        DcgmError d;
        DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_LWML_API, d, "lwmlInit", lwmlErrorString(lwmlSt));
        AddError(d);
        lwvsCommon.errorMask |= CPERF_ERR_LWML_FAIL;
        PRINT_ERROR("%s", "%s", d.GetMessage().c_str());
        return false;
    }
    m_lwmlInitialized = true;

    for (int gpuListIndex = 0; gpuListIndex < gpuList.size(); gpuListIndex++)
    {
        try
        {
            cpDevice = new CPerfDevice(gpuList[gpuListIndex], this);
        }
        catch (DcgmError &d)
        {
            AddErrorForGpu(gpuList[gpuListIndex], d);
            delete cpDevice;
            return false;
        }
        /* At this point, we consider this GPU part of our set */
        m_device.push_back(cpDevice);
    }
    return true;
}

/*****************************************************************************/
int ConstantPerf::LwdaInit()
{
    lwdaError_t lwSt;
    int i, j, count, valueSize;
    size_t arrayByteSize, arrayNelem;
    lwblasStatus_t lwbSt;
    CPerfDevice *device = 0;
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

    arrayByteSize = valueSize * CPERF_TEST_DIMENSION * CPERF_TEST_DIMENSION;
    arrayNelem = CPERF_TEST_DIMENSION * CPERF_TEST_DIMENSION;

    int streamsPerGpu = (int)m_testParameters->GetDouble(CPERF_STR_LWDA_STREAMS_PER_GPU);
    if (streamsPerGpu < 1)
    {
        streamsPerGpu = 1;
    }
    else if (streamsPerGpu > CPERF_MAX_STREAMS_PER_DEVICE)
    {
        streamsPerGpu = CPERF_MAX_STREAMS_PER_DEVICE;
    }

    /* Do per-device initialization */
    for (size_t deviceIdx = 0; deviceIdx < m_device.size(); deviceIdx++)
    {
        device = m_device[deviceIdx];

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

        /* Initialize lwca streams */
        for (i = 0; i < CPERF_MAX_STREAMS_PER_DEVICE; i++)
        {
            cperf_stream_p cpStream = &device->streams[i];
            lwSt = lwdaStreamCreate(&cpStream->lwdaStream);
            if (lwSt != lwdaSuccess)
            {
                lwvsCommon.errorMask |= CPERF_ERR_LWDA_STREAM_FAIL;
                std::stringstream ss;
                ss << "for GPU " << device->dcgmDeviceIndex << "(Lwca device index " << device->lwdaDeviceIdx
                   << "): " << lwdaGetErrorString(lwSt);
                DcgmError d;
                DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_LWDA_API, d, "lwdaStreamCreate");
                d.AddDetail(ss.str());
                AddError(d);
                return -1;
            }

            lwSt = lwdaEventCreate(&cpStream->afterWorkBlock);
            if (lwSt != lwdaSuccess)
            {
                LOG_LWDA_ERROR("lwdaEventCreate", lwSt, device->dcgmDeviceIndex);
                return -1;
            }

            for (j = 0; j < m_atATime; j++)
            {
                lwSt = lwdaEventCreate(&cpStream->beforeCopyH2D[j]);
                if (lwSt != lwdaSuccess)
                {
                    LOG_LWDA_ERROR("lwdaEventCreate", lwSt, device->dcgmDeviceIndex);
                    return -1;
                }
                lwSt = lwdaEventCreate(&cpStream->beforeGemm[j]);
                if (lwSt != lwdaSuccess)
                {
                    LOG_LWDA_ERROR("lwdaEventCreate", lwSt, device->dcgmDeviceIndex);
                    return -1;
                }
                lwSt = lwdaEventCreate(&cpStream->beforeCopyD2H[j]);
                if (lwSt != lwdaSuccess)
                {
                    LOG_LWDA_ERROR("lwdaEventCreate", lwSt, device->dcgmDeviceIndex);
                    return -1;
                }
                lwSt = lwdaEventCreate(&cpStream->afterCopyD2H[j]);
                if (lwSt != lwdaSuccess)
                {
                    LOG_LWDA_ERROR("lwdaEventCreate", lwSt, device->dcgmDeviceIndex);
                    return -1;
                }
            }
            cpStream->NeventsInitalized = m_atATime;

            /* Fill the arrays with random values */
            srand(time(NULL));

            lwSt = lwdaHostAlloc(&cpStream->hostA, arrayByteSize, hostAllocFlags);
            if (lwSt != lwdaSuccess)
            {
                lwvsCommon.errorMask |= CPERF_ERR_LWDA_ALLOC_FAIL;
                LOG_LWDA_ERROR("lwdaHostAlloc", lwSt, device->dcgmDeviceIndex, arrayByteSize);
                return -1;
            }

            lwSt = lwdaHostAlloc(&cpStream->hostB, arrayByteSize, hostAllocFlags);
            if (lwSt != lwdaSuccess)
            {
                lwvsCommon.errorMask |= CPERF_ERR_LWDA_ALLOC_FAIL;
                LOG_LWDA_ERROR("lwdaHostAlloc", lwSt, device->dcgmDeviceIndex, arrayByteSize);
                return -1;
            }

            lwSt = lwdaHostAlloc(&cpStream->hostC, arrayByteSize, hostAllocFlags);
            if (lwSt != lwdaSuccess)
            {
                lwvsCommon.errorMask |= CPERF_ERR_LWDA_ALLOC_FAIL;
                LOG_LWDA_ERROR("lwdaHostAlloc", lwSt, device->dcgmDeviceIndex, arrayByteSize);
                return -1;
            }

            if (m_useDgemm)
            {
                double *doubleHostA = (double *)cpStream->hostA;
                double *doubleHostB = (double *)cpStream->hostB;
                double *doubleHostC = (double *)cpStream->hostC;

                for (j = 0; j < arrayNelem; j++)
                {
                    doubleHostA[j] = (double)rand() / 100.0;
                    doubleHostB[j] = (double)rand() / 100.0;
                    doubleHostC[j] = 0.0;
                }
            }
            else
            {
                /* sgemm */
                float *floatHostA = (float *)cpStream->hostA;
                float *floatHostB = (float *)cpStream->hostB;
                float *floatHostC = (float *)cpStream->hostC;

                for (j = 0; j < arrayNelem; j++)
                {
                    floatHostA[j] = (float)rand() / 100.0;
                    floatHostB[j] = (float)rand() / 100.0;
                    floatHostC[j] = 0.0;
                }
            }

            device->Nstreams++;
        }

        /* Initialize lwblas */
        lwbSt = lwblasCreate(&device->lwblasHandle);
        if (lwbSt != LWBLAS_STATUS_SUCCESS)
        {
            lwvsCommon.errorMask |= CPERF_ERR_LWBLAS_FAIL;
            LOG_LWBLAS_ERROR("lwblasCreate", lwbSt, device->dcgmDeviceIndex);
            return -1;
        }
        device->allocatedLwblasHandle = 1;

        for (i = 0; i < device->Nstreams; i++)
        {
            cperf_stream_p cpStream = &device->streams[i];

            lwSt = lwdaMalloc((void **)&cpStream->deviceA, arrayByteSize);
            if (lwSt != lwdaSuccess)
            {
                lwvsCommon.errorMask |= CPERF_ERR_LWDA_ALLOC_FAIL;
                LOG_LWDA_ERROR("lwdaMalloc", lwSt, device->dcgmDeviceIndex, arrayByteSize);
                return -1;
            }

            lwSt = lwdaMalloc((void **)&cpStream->deviceB, arrayByteSize);
            if (lwSt != lwdaSuccess)
            {
                lwvsCommon.errorMask |= CPERF_ERR_LWDA_ALLOC_FAIL;
                LOG_LWDA_ERROR("lwdaMalloc", lwSt, device->dcgmDeviceIndex, arrayByteSize);
                return -1;
            }

            lwSt = lwdaMalloc((void **)&cpStream->deviceC, arrayByteSize);
            if (lwSt != lwdaSuccess)
            {
                lwvsCommon.errorMask |= CPERF_ERR_LWDA_ALLOC_FAIL;
                LOG_LWDA_ERROR("lwdaMalloc", lwSt, device->dcgmDeviceIndex, arrayByteSize);
                return -1;
            }
        }
    }

    return 0;
}

/*****************************************************************************/
void ConstantPerf::Go(TestParameters *testParameters, const std::vector<unsigned int> &gpuList)
{
    bool result;
    InitializeForGpuList(gpuList);

    if (!testParameters->GetBoolFromString(CPERF_STR_IS_ALLOWED))
    {
        DcgmError d;
        DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_TEST_DISABLED, d, CPERF_PLUGIN_NAME);
        AddError(d);
        SetResult(LWVS_RESULT_SKIP);
        return;
    }

    /* Cache test parameters */
    m_testParameters = testParameters;
    m_useDgemm = testParameters->GetBoolFromString(CPERF_STR_USE_DGEMM);
    m_testDuration = testParameters->GetDouble(CPERF_STR_TEST_DURATION);
    m_targetPerf = testParameters->GetDouble(CPERF_STR_TARGET_PERF);
    m_atATime = testParameters->GetDouble(CPERF_STR_LWDA_OPS_PER_STREAM);
    m_sbeFailureThreshold = testParameters->GetDouble(CPERF_STR_SBE_ERROR_THRESHOLD);

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
bool ConstantPerf::CheckGpuPerf(CPerfDevice *device, std::vector<DcgmError> &errorList,
                                timelib64_t startTime, timelib64_t endTime)
{
    std::vector<dcgmTimeseriesInfo_t> data;
    std::stringstream buf;

    data = m_dcgmRecorder.GetLwstomGpuStat(device->lwmlDeviceIndex, PERF_STAT_NAME);
    if (data.size() == 0)
    {
        DcgmError d;
        DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_CANNOT_GET_STAT, d, PERF_STAT_NAME, device->dcgmDeviceIndex);
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

    /* If performance was low, it might because we're D2H/H2D transfer bound.
       Discount our expected perf by how much time we actually spent doing
       dgemm vs doing copies */
    double discountMultiplier = 1.0;
    double totalActiveUsec = device->usecInCopies + device->usecInGemm;
    if (totalActiveUsec > 0.0)
    {
        discountMultiplier = device->usecInGemm / totalActiveUsec;
        PRINT_DEBUG("%u %f %f", "lwmlGpuIndex %u, discountMultiplier %f, totalActiveUsec %f",
                    device->dcgmDeviceIndex, discountMultiplier, totalActiveUsec);
    }

    double minRatio = m_testParameters->GetDouble(CPERF_STR_TARGET_PERF_MIN_RATIO);

    RecordObservedMetric(device->dcgmDeviceIndex, CPERF_STR_TARGET_PERF, maxVal);

    if (maxVal < discountMultiplier * minRatio * m_targetPerf)
    {
        lwvsCommon.errorMask |= CPERF_ERR_GPU_PERF_TOO_LOW;
        DcgmError d;
        DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_STRESS_LEVEL, d, maxVal, m_targetPerf, device->dcgmDeviceIndex);
        std::string utilNote = m_dcgmRecorder.GetGpuUtilizationNote(device->dcgmDeviceIndex, startTime, endTime);
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
    ss << "GPU " << device->lwmlDeviceIndex << " relative stress level\t" << avg;
    AddInfoVerboseForGpu(device->dcgmDeviceIndex, ss.str());
    return true;
}

/*****************************************************************************/
bool ConstantPerf::CheckGpuTemperature(CPerfDevice *device, std::vector<DcgmError> &errorList,
                                       timelib64_t startTime, timelib64_t earliestStopTime, bool testFinished)
{
    std::string infoMsg;
    long long maxTemp = static_cast<long long>(m_testParameters->GetDouble(CPERF_STR_TEMPERATURE_MAX));
    long long highTempObserved = 0;
    int st = m_dcgmRecorder.CheckGpuTemperature(device->dcgmDeviceIndex, errorList, maxTemp, infoMsg,
                                                 startTime, earliestStopTime, highTempObserved);

    if (testFinished && highTempObserved != 0)
    {
        RecordObservedMetric(device->dcgmDeviceIndex, CPERF_STR_TEMPERATURE_MAX, highTempObserved);
    }

    if (st == DR_VIOLATION)
    {
        lwvsCommon.errorMask |= CPERF_ERR_GPU_TEMP_TOO_HIGH;
    }
    // Check for a communication error
    else if (st == DR_COMM_ERROR)
    {
        m_dcgmCommErrorOclwrred = true;
    }

    if (testFinished)
    {
        AddInfoVerboseForGpu(device->dcgmDeviceIndex, infoMsg);
    }

    return errorList.empty();
}

/*****************************************************************************/
bool ConstantPerf::CheckLwmlEvents(CPerfDevice *device, std::vector<DcgmError> &errorList, timelib64_t startTime,
                                   timelib64_t earliestStopTime)
{
    std::vector<unsigned short> fieldIds;
    dcgmTimeseriesInfo_t dti;
    std::vector<dcgmTimeseriesInfo_t> failureThresholds;

    memset(&dti, 0, sizeof(dti));
    dti.isInt = true;

    if (DCGM_FP64_IS_BLANK(m_sbeFailureThreshold) == 0)
    {
        // Only evaluate this field if it has a failure threshold set
        fieldIds.push_back(DCGM_FI_DEV_ECC_SBE_VOL_TOTAL);
        // Set the stated error threshold
        dti.val.i64 = static_cast<long long>(m_sbeFailureThreshold);
        failureThresholds.push_back(dti);
    }

    fieldIds.push_back(DCGM_FI_DEV_ECC_DBE_VOL_TOTAL);
    fieldIds.push_back(DCGM_FI_DEV_XID_ERRORS);

    // DBEs and XIDs have a 0 failure threshold
    dti.val.i64 = 0;
    for (int i = 0; i < 2; i++)
    {
        failureThresholds.push_back(dti);
    }

    // PCIE replays may have a non-zero failure threshold
    fieldIds.push_back(DCGM_FI_DEV_PCIE_REPLAY_COUNTER);
    dti.val.i64 = static_cast<long long>(m_testParameters->GetDouble(CPERF_STR_MAX_PCIE_REPLAYS));
    failureThresholds.push_back(dti);

    int ret = m_dcgmRecorder.CheckErrorFields(fieldIds, &failureThresholds, device->lwmlDeviceIndex,
                                               errorList, startTime, earliestStopTime);
    // Check for a communication error
    if (ret == DR_COMM_ERROR)
    {
        m_dcgmCommErrorOclwrred = true;
    }

    // Return true if there are no errors, false otherwise
    return errorList.empty();
}

/*****************************************************************************/
bool ConstantPerf::CheckPassFailSingleGpu(CPerfDevice *device, std::vector<DcgmError> &errorList,
                                          timelib64_t startTime, timelib64_t earliestStopTime, bool testFinished)
{
    DcgmLockGuard lock(&m_mutex); // prevent conlwrrent failure checks from workers
    bool result;
    int st;

    if (testFinished)
    {
        // This check is only run once the test is finished
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
bool ConstantPerf::CheckPassFail(timelib64_t startTime, timelib64_t earliestStopTime)
{
    bool passed, allPassed = true;
    std::vector<DcgmError> errorList;
    std::vector<DcgmError> errorListAllGpus;
    
    /* Get latest values for watched fields before checking pass fail.
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

    // Check for generic errors and whether we passed on all gpus
    for (size_t i = 0; i < errorListAllGpus.size(); i++)
    {
        AddError(errorListAllGpus[i]);
    }

    return allPassed;
}

/*****************************************************************************/
class ConstantPerfWorker: public LwvsThread
{
private:
    CPerfDevice     *m_device;          /* Which device this worker thread is running on */
    ConstantPerf    &m_plugin;          /* ConstantPerf plugin for logging and failure checks */
    TestParameters  *m_testParameters;  /* Read-only test parameters */
    int             m_useDgemm;         /* Wheter to use dgemm (1) or sgemm (0) for operations */
    double          m_targetPerf;       /* Target stress in gflops */
    double          m_testDuration;     /* Target test duration in seconds */
    timelib64_t     m_stopTime;         /* Timestamp when run() finished */
    int             m_atATime;          /* Number of ops to queue to the stream at a time */
    DcgmRecorder    &m_dcgmRecorder;

public:
    /*************************************************************************/
    ConstantPerfWorker(CPerfDevice *device, ConstantPerf &plugin, TestParameters *tp, DcgmRecorder &dr);

    /*************************************************************************/
    virtual ~ConstantPerfWorker() /* Virtual to satisfy ancient compiler */
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
    /*****************************************************************************/
    /*
     * Read the timing from all of the lwca events of a given stream and journal
     * them to the stream object
     *
     * Returns: 0 on success
     *         <0 on error
     */
    int RecordTiming(cperf_stream_p cpStream);

    /*****************************************************************************/
    /*
     * Queue one unit of work to one stream
     *
     * Note that errors returned from async calls can be from any other async call
     *
     * Returns: 0 on success
     *         !0 on error
     *
     */
    int QueueOne(int streamIdx, int opIdx, float *floatAlpha, double *doubleAlpha, float *floatBeta, 
                 double *doubleBeta);

};

/****************************************************************************/
/*
 * ConstantPerf RunTest implementation
 * 
 * Method returns whether the test ran sucessfully - this is *not* the same as whether the test passed
 */
/*****************************************************************************/
bool ConstantPerf::RunTest(const std::vector<unsigned int> &gpuList)
{
    int st, Nrunning = 0;
    ConstantPerfWorker *workerThreads[CPERF_MAX_DEVICES] = {0};
    bool outputStats = true;
    unsigned int timeCount = 0;
    timelib64_t earliestStopTime;
    timelib64_t startTime = timelib_usecSince1970();
    bool testPassed;


    if (gpuList.size() >= CPERF_MAX_DEVICES)
    {
        PRINT_ERROR("%d %d", "Bad gpuList size: %d (max allowed: %d)",
                (int)gpuList.size(), CPERF_MAX_DEVICES-1);
        return false;
    }

    if (!LwmlInit(gpuList))
    {
        Cleanup();
        return false;
    }

    st = LwdaInit();
    if (st)
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

    /* Is binary logging enabled for this stat collection? */
    std::string logFileName = m_testParameters->GetString(PS_LOGFILE);
    int logFileType = (int)m_testParameters->GetDouble(PS_LOGFILE_TYPE);

    std::vector<unsigned short> fieldIds;
    fieldIds.push_back(DCGM_FI_DEV_POWER_USAGE);
    fieldIds.push_back(DCGM_FI_DEV_GPU_TEMP);
    fieldIds.push_back(DCGM_FI_DEV_SM_CLOCK);
    fieldIds.push_back(DCGM_FI_DEV_MEM_CLOCK);
    fieldIds.push_back(DCGM_FI_DEV_ECC_SBE_VOL_TOTAL);
    fieldIds.push_back(DCGM_FI_DEV_ECC_DBE_VOL_TOTAL);
    fieldIds.push_back(DCGM_FI_DEV_XID_ERRORS);
    fieldIds.push_back(DCGM_FI_DEV_POWER_VIOLATION);
    fieldIds.push_back(DCGM_FI_DEV_THERMAL_VIOLATION);
    fieldIds.push_back(DCGM_FI_DEV_PCIE_REPLAY_COUNTER);
    fieldIds.push_back(DCGM_FI_DEV_CLOCK_THROTTLE_REASONS);
    fieldIds.push_back(DCGM_FI_DEV_GPU_UTIL);
    m_dcgmRecorder.AddWatches(fieldIds, gpuList, false, "targeted_stress_field_group", "targeted_stress_group",
                              m_testDuration);

    try /* Catch runtime errors */
    {
        /* Create and start all workers */
        for (size_t i = 0; i < m_device.size(); i++)
        {
            workerThreads[i] = new ConstantPerfWorker(m_device[i], *this, m_testParameters, m_dcgmRecorder);
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

    // Get the earliestStopTime and delete the threads
    earliestStopTime = INT64_MAX;
    for (size_t i = 0; i < m_device.size(); i++)
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

    /* Should we write out a log file of stats? */
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
 * ConstantPerffWorker implementation.
 */
/****************************************************************************/
ConstantPerfWorker::ConstantPerfWorker(CPerfDevice *device, ConstantPerf &plugin, TestParameters *tp, DcgmRecorder &dr):
        m_device(device), m_plugin(plugin), m_testParameters(tp), m_dcgmRecorder(dr)
{
    m_useDgemm = tp->GetBoolFromString(CPERF_STR_USE_DGEMM);
    m_targetPerf = tp->GetDouble(CPERF_STR_TARGET_PERF);
    m_testDuration = tp->GetDouble(CPERF_STR_TEST_DURATION);
    m_atATime = tp->GetDouble(CPERF_STR_LWDA_OPS_PER_STREAM);
}

/****************************************************************************/
int ConstantPerfWorker::RecordTiming(cperf_stream_p cpStream)
{
    int i;
    lwdaError_t lwSt = lwdaSuccess;
    float fp32Val = 0.0;

    for (i = 0; i < m_atATime; i++)
    {
        lwSt = lwdaEventElapsedTime(&fp32Val, cpStream->beforeCopyH2D[i], cpStream->beforeGemm[i]);
        if (lwSt != lwdaSuccess)
        {
            break;
        }
        cpStream->usecInCopies += 1000.0 * ((double)fp32Val);

        lwSt = lwdaEventElapsedTime(&fp32Val, cpStream->beforeGemm[i], cpStream->beforeCopyD2H[i]);
        if (lwSt != lwdaSuccess)
        {
            break;
        }
        cpStream->usecInGemm += 1000.0 * ((double)fp32Val);

        lwSt = lwdaEventElapsedTime(&fp32Val, cpStream->beforeCopyD2H[i], cpStream->afterCopyD2H[i]);
        if (lwSt != lwdaSuccess)
        {
            break;
        }
        cpStream->usecInCopies += 1000.0 * ((double)fp32Val);
    }

    if (lwSt != lwdaSuccess)
    {
        LOG_LWDA_ERROR_FOR_PLUGIN(&m_plugin, "lwdaEventElapsedTime", lwSt, m_device->dcgmDeviceIndex);
        std::stringstream ss;
        ss << "Results for GPU " << m_device->dcgmDeviceIndex << " will be inaclwrate because there was an "
            << "error getting elapsed time.";
        m_plugin.AddInfoVerboseForGpu(m_device->dcgmDeviceIndex, ss.str());
        return -1;
    }

    return 0;
}


/****************************************************************************/
int ConstantPerfWorker::QueueOne(int streamIdx, int opIdx, float *floatAlpha, double *doubleAlpha, float *floatBeta, 
                                 double *doubleBeta)
{
    int valueSize, arrayByteSize;
    lwdaError_t lwSt;
    lwblasStatus_t lwbSt;
    cperf_stream_p cpStream = &m_device->streams[streamIdx];

    if (m_useDgemm)
    {
        valueSize = sizeof(double);
    }
    else
    {
        valueSize = sizeof(float);
    }

    arrayByteSize = valueSize * CPERF_TEST_DIMENSION * CPERF_TEST_DIMENSION;

    lwSt = lwdaEventRecord(cpStream->beforeCopyH2D[opIdx], cpStream->lwdaStream);
    if (lwSt != lwdaSuccess)
    {
        LOG_LWDA_ERROR_FOR_PLUGIN(&m_plugin, "lwdaEventRecord", lwSt, m_device->dcgmDeviceIndex);
        return -1;
    }

    /* Copy the host arrays to the device arrays */
    lwSt = lwdaMemcpyAsync(cpStream->deviceA, cpStream->hostA, arrayByteSize,
                        lwdaMemcpyHostToDevice, cpStream->lwdaStream);
    if (lwSt != lwdaSuccess)
    {
        lwvsCommon.errorMask |= CPERF_ERR_LWDA_MEMCPY_FAIL;
        LOG_LWDA_ERROR_FOR_PLUGIN(&m_plugin, "lwdaMemcpyAsync", lwSt, m_device->dcgmDeviceIndex, arrayByteSize);
        return -1;
    }
    lwSt = lwdaMemcpyAsync(cpStream->deviceB, cpStream->hostB, arrayByteSize,
                        lwdaMemcpyHostToDevice, cpStream->lwdaStream);
    if (lwSt != lwdaSuccess)
    {
        lwvsCommon.errorMask |= CPERF_ERR_LWDA_MEMCPY_FAIL;
        LOG_LWDA_ERROR_FOR_PLUGIN(&m_plugin, "lwdaMemcpyAsync", lwSt, m_device->dcgmDeviceIndex, arrayByteSize);
        return -1;
    }

    lwSt = lwdaEventRecord(cpStream->beforeGemm[opIdx], cpStream->lwdaStream);
    if (lwSt != lwdaSuccess)
    {
        LOG_LWDA_ERROR_FOR_PLUGIN(&m_plugin, "lwdaEventRecord", lwSt, m_device->dcgmDeviceIndex);
        return -1;
    }

    lwbSt = lwblasSetStream(m_device->lwblasHandle, cpStream->lwdaStream);
    if (lwbSt != LWBLAS_STATUS_SUCCESS)
    {
        LOG_LWBLAS_ERROR_FOR_PLUGIN(&m_plugin, "lwblasSetStream", lwbSt, m_device->dcgmDeviceIndex);
        return -1;
    }

    if (m_useDgemm)
    {
        lwbSt = lwblasDgemm(m_device->lwblasHandle, LWBLAS_OP_N, LWBLAS_OP_N, CPERF_TEST_DIMENSION,
                            CPERF_TEST_DIMENSION, CPERF_TEST_DIMENSION, doubleAlpha,
                            (double *)cpStream->deviceA, CPERF_TEST_DIMENSION,
                            (double *)cpStream->deviceB, CPERF_TEST_DIMENSION, doubleBeta,
                            (double *)cpStream->deviceC, CPERF_TEST_DIMENSION);
        if (lwbSt != LWBLAS_STATUS_SUCCESS)
        {
            LOG_LWBLAS_ERROR_FOR_PLUGIN(&m_plugin, "lwblasDgemm", lwbSt, m_device->dcgmDeviceIndex);
            return -1;
        }
    }
    else
    {
        lwbSt = lwblasSgemm(m_device->lwblasHandle, LWBLAS_OP_N, LWBLAS_OP_N, CPERF_TEST_DIMENSION,
                            CPERF_TEST_DIMENSION, CPERF_TEST_DIMENSION, floatAlpha,
                            (float *)cpStream->deviceA, CPERF_TEST_DIMENSION,
                            (float *)cpStream->deviceB, CPERF_TEST_DIMENSION, floatBeta,
                            (float *)cpStream->deviceC, CPERF_TEST_DIMENSION);
        if (lwbSt != LWBLAS_STATUS_SUCCESS)
        {
            LOG_LWBLAS_ERROR_FOR_PLUGIN(&m_plugin, "lwblasSgemm", lwbSt, m_device->dcgmDeviceIndex);
            return -1;
        }
    }

    lwSt = lwdaEventRecord(cpStream->beforeCopyD2H[opIdx], cpStream->lwdaStream);
    if (lwSt != lwdaSuccess)
    {
        LOG_LWDA_ERROR_FOR_PLUGIN(&m_plugin, "lwdaEventRecord", lwSt, m_device->dcgmDeviceIndex);
        return -1;
    }

    /* Copy the destination matrix back */
    lwSt = lwdaMemcpyAsync(cpStream->hostC, cpStream->deviceC, arrayByteSize,
                        lwdaMemcpyDeviceToHost, cpStream->lwdaStream);
    if (lwSt != lwdaSuccess)
    {
        lwvsCommon.errorMask |= CPERF_ERR_LWDA_MEMCPY_FAIL;
        LOG_LWDA_ERROR_FOR_PLUGIN(&m_plugin, "lwdaMemcpyAsync", lwSt, m_device->dcgmDeviceIndex, arrayByteSize);
        return -1;
    }

    lwSt = lwdaEventRecord(cpStream->afterCopyD2H[opIdx], cpStream->lwdaStream);
    if (lwSt != lwdaSuccess)
    {
        LOG_LWDA_ERROR_FOR_PLUGIN(&m_plugin, "lwdaEventRecord", lwSt, m_device->dcgmDeviceIndex);
        return -1;
    }

    return 0;
}

/****************************************************************************/
void ConstantPerfWorker::run(void)
{
    int j, st = 0;
    double doubleAlpha, doubleBeta;
    float floatAlpha, floatBeta;
    double startTime;
    double lastPrintTime;  /* last time we printed out the current perf */
    double lastFailureCheckTime;  /* last time we checked for failures */
    double now, elapsed;
    int useNstreams;
    int NstreamsRequeued = 0;
    long long Nops = 0;
    lwdaError_t lwSt = lwdaSuccess;
    int valueSize;
    std::vector<DcgmError> errorList;

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

    double copyBytesPerOp = 3.0 * (double)valueSize * (double)CPERF_TEST_DIMENSION *
                        (double)CPERF_TEST_DIMENSION;
    double flopsPerOp = 2.0 * (double)CPERF_TEST_DIMENSION *
                        (double)CPERF_TEST_DIMENSION * (double)CPERF_TEST_DIMENSION;
    double opsPerSec = m_targetPerf / (flopsPerOp / 1000000000.0);
    long long maxOpsSoFar;

    /* Set initial test values */
    useNstreams = CPERF_MAX_STREAMS_PER_DEVICE;
    doubleAlpha = 1.01 + ((double)(rand() % 100)/10.0);
    doubleBeta = 1.01 + ((double)(rand() % 100)/10.0);
    floatAlpha = (float)doubleAlpha;
    floatBeta = (float)doubleBeta;

    std::string gflopsKey;
    gflopsKey = std::string(PERF_STAT_NAME);

    /* Record some of our static callwlated parameters in case we need them for debugging */
    m_dcgmRecorder.SetGpuStat(m_device->lwmlDeviceIndex, std::string("flops_per_op"),
                            flopsPerOp);
    m_dcgmRecorder.SetGpuStat(m_device->lwmlDeviceIndex, std::string("bytes_copied_per_op"),
                            copyBytesPerOp);
    m_dcgmRecorder.SetGpuStat(m_device->lwmlDeviceIndex, std::string("num_lwda_streams"),
                            (long long)useNstreams);
    m_dcgmRecorder.SetGpuStat(m_device->lwmlDeviceIndex, std::string("try_ops_per_sec"),
                            opsPerSec);

    /* Lock to our assigned GPU */
    lwdaSetDevice(m_device->lwdaDeviceIdx);

    std::stringstream ss;
    ss << "Running for " << m_testDuration << " seconds";
    m_plugin.AddInfo(ss.str());

    startTime = timelib_dsecSince1970();
    lastPrintTime = startTime;
    lastFailureCheckTime = startTime;
    now = timelib_dsecSince1970();

    while (now - startTime < m_testDuration && !ShouldStop())
    {
        NstreamsRequeued = 0;
        now = timelib_dsecSince1970();
        elapsed = now - startTime;
        maxOpsSoFar = (long long)(elapsed * opsPerSec);

        for (int i = 0; i < useNstreams && (lwvsCommon.training || Nops < maxOpsSoFar) && !ShouldStop(); i++)
        {
            cperf_stream_p cpStream = &m_device->streams[i];

            /* Query each stream to see if it's idle (lwdaSuccess return) */
            if (cpStream->blocksQueued < 1 || lwdaSuccess == lwdaEventQuery(cpStream->afterWorkBlock))
            {
                /* Have we queued any blocks before? If so, compute timing for those runs */
                if (cpStream->blocksQueued)
                {
                    st = RecordTiming(cpStream);
                    if (st)
                    {
                        break;
                    }
                    PRINT_DEBUG("%d %d %f %f", "deviceIdx %d, streamIdx %d, usecInCopies %f, usecInGemm %f",
                                m_device->dcgmDeviceIndex, i, cpStream->usecInCopies, cpStream->usecInGemm);
                }

                for (j = 0; j < m_atATime; j++)
                {
                    st = QueueOne(i, j, &floatAlpha, &doubleAlpha, &floatBeta, &doubleBeta);
                    if (st)
                    {
                        break;
                    }
                    Nops++;
                }
                // Check to see if QueueOne had an error
                if (st)
                {
                    break;
                }
                NstreamsRequeued++;
                cpStream->blocksQueued++;

                /* Record an event at the end. This will be the event we check to see if
                our block of work has completed */
                lwSt = lwdaEventRecord(cpStream->afterWorkBlock,
                                    m_device->streams[i].lwdaStream);
                if (lwSt != lwdaSuccess)
                {
                    LOG_LWDA_ERROR_FOR_PLUGIN(&m_plugin, "lwdaEventRecord", lwSt, m_device->dcgmDeviceIndex);
                    /* An error here causes problems for the rest of the test due to time callwlations. */
                    break;
                }

            }
        }

        if (st || lwSt != lwdaSuccess)
        {
            // We had an error - stop the test
            break;
        }

        /* If we didn't queue any work, sleep a bit so we don't busy wait */
        if (!NstreamsRequeued)
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
            m_dcgmRecorder.SetGpuStat(m_device->lwmlDeviceIndex, "nops_so_far", (long long)Nops);
            ss.str("");
            ss << "DeviceIdx " << m_device->dcgmDeviceIndex <<  ", ops " << Nops << ", gflops " << gflops;
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

    m_device->usecInCopies = 0.0;
    m_device->usecInGemm = 0.0;
    /* Aggregate per-stream metrics to device metrics */
    for (int i = 0; i < useNstreams; i++)
    {
        cperf_stream_p cpStream = &m_device->streams[i];

        for (j = 0; j < m_atATime; j++)
        {
            m_device->usecInCopies += cpStream->usecInCopies;
            m_device->usecInGemm += cpStream->usecInGemm;
        }
    }

    m_stopTime = timelib_usecSince1970();
    PRINT_DEBUG("%d %lld", "ConstantPerfWorker deviceIndex %d finished at %lld", m_device->dcgmDeviceIndex,
                (long long)m_stopTime);
}

extern "C" {
    Plugin *maker() {
        return new ConstantPerf;
    }
    class proxy {
    public:
        proxy()
        {
            factory["Constant Perf"] = maker;
        }
    };
    proxy p;
}
