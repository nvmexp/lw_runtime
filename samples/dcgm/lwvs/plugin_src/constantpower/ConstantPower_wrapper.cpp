#define __STDC_LIMIT_MACROS
#include <stdint.h>

#include "ConstantPower_wrapper.h"
#include <stdexcept>

#include "LwvsThread.h"
#include "PluginStrings.h"

/*************************************************************************/
ConstantPower::ConstantPower(): m_testParameters(NULL), m_lwmlInitialized(false), m_dcgmCommErrorOclwrred(false),
                                m_dcgmRecorderInitialized(false)
{
    TestParameters *tp;

    m_infoStruct.name = CP_PLUGIN_NAME;
    m_infoStruct.shortDescription = "This plugin will keep the list of GPUs at a constant power level.";
    m_infoStruct.testGroups = "Power";
    m_infoStruct.selfParallel = true;
    m_infoStruct.logFileTag = CP_PLUGIN_LF_NAME;

    /* Populate default test parameters */
    tp = new TestParameters();
    tp->AddString(PS_RUN_IF_GOM_ENABLED, "False");
    tp->AddString(CP_STR_USE_DGEMM, "True");
    tp->AddString(CP_STR_FAIL_ON_CLOCK_DROP, "True");
    tp->AddDouble(CP_STR_TEST_DURATION, 120.0, 1.0, 86400.0);
    tp->AddDouble(CP_STR_TARGET_POWER, 100.0, 1.0, 500.0);
    tp->AddDouble(CP_STR_LWDA_STREAMS_PER_GPU, 4.0, 1.0, (double)CP_MAX_STREAMS_PER_DEVICE);
    tp->AddDouble(CP_STR_READJUST_INTERVAL, 2.0, 1.0, 10.0);
    tp->AddDouble(CP_STR_PRINT_INTERVAL, 1.0, 1.0, 300.0);
    tp->AddDouble(CP_STR_TARGET_POWER_MIN_RATIO, 0.75, 0.5, 1.0);
    tp->AddDouble(CP_STR_TARGET_POWER_MAX_RATIO, 1.2, 1.0, 2.0);
    tp->AddDouble(CP_STR_MOV_AVG_PERIODS, 15.0, 1.0, 86400.0); //Max is same as max for test duration
    tp->AddDouble(CP_STR_TARGET_MOVAVG_MIN_RATIO, 0.95, 0.5, 1.0);
    tp->AddDouble(CP_STR_TARGET_MOVAVG_MAX_RATIO, 1.05, 1.0, 2.0);
    tp->AddDouble(CP_STR_TEMPERATURE_MAX, 100.0, 30.0, 120.0);
    tp->AddDouble(CP_STR_MAX_MEMORY_CLOCK, 0.0, 0.0, 100000.0);
    tp->AddDouble(CP_STR_MAX_GRAPHICS_CLOCK, 0.0, 0.0, 100000.0);
    tp->AddDouble(CP_STR_OPS_PER_REQUEUE, 1.0, 1.0, 32.0);
    tp->AddDouble(CP_STR_STARTING_MATRIX_DIM, 1.0, 1.0, 1024.0);
    tp->AddDouble(CP_STR_SBE_ERROR_THRESHOLD, DCGM_FP64_BLANK, 0.0, DCGM_FP64_BLANK);
    tp->AddString(CP_STR_IS_ALLOWED, "False");
    m_infoStruct.defaultTestParameters = tp;
}

/*************************************************************************/
ConstantPower::~ConstantPower()
{
    Cleanup();
}

void ConstantPower::Cleanup()
{
    int i;
    CPDevice *device = NULL;

    if (m_hostA)
    {
        free(m_hostA);
    }
    m_hostA = NULL;
    
    if (m_hostB)
    {
        free(m_hostB);
    }
    m_hostB = NULL;
    
    if (m_hostC)
    {
        free(m_hostC);
    }
    m_hostC = NULL;

    for (size_t deviceIdx = 0; deviceIdx < m_device.size(); deviceIdx++)
    {
        device = m_device[deviceIdx];

        lwdaSetDevice(device->lwdaDeviceIdx);

        /* Wait for all streams to finish */
        for (i = 0; i < device->NlwdaStreams; i++)
        {
            lwdaStreamSynchronize(device->lwdaStream[i]);
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
    m_lwmlInitialized = NULL;
}

/*************************************************************************/
bool ConstantPower::LwmlInit(const std::vector<unsigned int> &gpuList)
{
    lwmlReturn_t lwmlSt;
    CPDevice *device = 0;
    lwdaError_t lwSt;

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
        lwvsCommon.errorMask |= CP_ERR_LWML_FAIL;
        DcgmError d;
        DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_LWML_API, d, "lwmlInit", lwmlErrorString(lwmlSt));
        AddError(d);
        PRINT_ERROR("%s", "%s", d.GetMessage().c_str());
        return false;
    }
    m_lwmlInitialized = true;

    for (int gpuListIndex = 0; gpuListIndex < gpuList.size(); gpuListIndex++)
    {
        try
        {
            device = new CPDevice(gpuList[gpuListIndex], this);

            /* Get the power management limits for the device */
            dcgmDeviceAttributes_t attrs;
            dcgmReturn_t ret = m_dcgmRecorder.GetDeviceAttributes(gpuList[gpuListIndex], attrs);
            if (ret == DCGM_ST_OK)
            {
                device->maxPowerTarget = attrs.powerLimits.enforcedPowerLimit;
            }
            else
            {
                DcgmError d;
                DCGM_ERROR_FORMAT_MESSAGE_DCGM(DCGM_FR_DCGM_API, d, ret, "dcgmGetDeviceAttributes");
                AddErrorForGpu(gpuList[gpuListIndex], d);
                PRINT_ERROR("%s", "Can't get the enforced power limit: %s", d.GetMessage().c_str());
                return false;
            }
        }
        catch (const DcgmError &d)
        {
            AddErrorForGpu(gpuList[gpuListIndex], d);
            delete device;
            return false;
        }
        catch (const std::runtime_error &re)
        {
            DcgmError d;
            DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_INTERNAL, d, re.what());
            AddErrorForGpu(gpuList[gpuListIndex], d);

            delete device;
            return false;
        }

        /* At this point, we consider this GPU part of our set */
        m_device.push_back(device);
    }

    return true;
}

/*************************************************************************/
int ConstantPower::LwdaInit()
{
    int count, valueSize;
    size_t arrayByteSize, arrayNelem;
    lwdaError_t lwSt;
    lwblasStatus_t lwbSt;
    CPDevice *device = 0;

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

    arrayByteSize = valueSize * CP_MAX_DIMENSION * CP_MAX_DIMENSION;
    arrayNelem = CP_MAX_DIMENSION * CP_MAX_DIMENSION;
    
    m_hostA = malloc(arrayByteSize);
    m_hostB = malloc(arrayByteSize);
    m_hostC = malloc(arrayByteSize);
    if (!m_hostA || ! m_hostB || !m_hostC)
    {
        lwvsCommon.errorMask |= CP_ERR_LWDA_ALLOC_FAIL;
        PRINT_ERROR("%d", "Error allocating %d bytes x 3 on the host (malloc)", (int)arrayByteSize);
        DcgmError d;
        DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_MEMORY_ALLOC_HOST, d, arrayByteSize);
        AddError(d);
        return -1;
    }

    /* Fill the arrays with random values */
    srand(time(NULL));

    if (m_useDgemm)
    {
        double *doubleHostA = (double *)m_hostA;
        double *doubleHostB = (double *)m_hostB;
        double *doubleHostC = (double *)m_hostC;

        for (int i = 0; i < arrayNelem; i++)
        {
            doubleHostA[i] = (double)rand() / 100.0;
            doubleHostB[i] = (double)rand() / 100.0;
            doubleHostC[i] = (double)rand() / 100.0;
        }
    }
    else
    {
        /* sgemm */
        float *floatHostA = (float *)m_hostA;
        float *floatHostB = (float *)m_hostB;
        float *floatHostC = (float *)m_hostC;

        for (int i = 0; i < arrayNelem; i++)
        {
            floatHostA[i] = (float)rand() / 100.0;
            floatHostB[i] = (float)rand() / 100.0;
            floatHostC[i] = (float)rand() / 100.0;
        }
    }

    /* Do per-device initialization */
    for (size_t deviceIdx = 0; deviceIdx < m_device.size(); deviceIdx++)
    {
        device = m_device[deviceIdx];
        device->minMatrixDim = 1;

        /* Make all subsequent lwca calls link to this device */
        lwdaSetDevice(device->lwdaDeviceIdx);

        lwSt = lwdaGetDeviceProperties(&device->lwdaDevProp, device->lwdaDeviceIdx);
        if (lwSt != lwdaSuccess)
        {
            LOG_LWDA_ERROR("lwdaGetDeviceProperties", lwSt, device->dcgmDeviceIndex);
            return -1;
        }

        /* Initialize lwca streams */
        for (int i = 0; i < CP_MAX_STREAMS_PER_DEVICE; i++)
        {
            lwSt = lwdaStreamCreate(&device->lwdaStream[i]);
            if (lwSt != lwdaSuccess)
            {
                DcgmError d;
                DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_LWDA_API, d, "lwdaStreamCreate");
                std::stringstream ss;
                ss << "'" << lwdaGetErrorString(lwSt) << "' for GPU " << device->dcgmDeviceIndex;
                d.AddDetail(ss.str());
                AddErrorForGpu(device->dcgmDeviceIndex, d);
                lwvsCommon.errorMask |= CP_ERR_LWDA_STREAM_FAIL;
                return -1;
            }
            device->NlwdaStreams++;
        }

        /* Initialize lwblas */
        lwbSt = lwblasCreate(&device->lwblasHandle);
        if (lwbSt != LWBLAS_STATUS_SUCCESS)
        {
            lwvsCommon.errorMask |= CP_ERR_LWBLAS_FAIL;
            LOG_LWBLAS_ERROR("lwblasCreate", lwbSt, device->dcgmDeviceIndex);
            return -1;
        }
        device->allocatedLwblasHandle = 1;

        lwSt = lwdaMalloc((void **)&device->deviceA, arrayByteSize);
        if (lwSt != lwdaSuccess)
        {
            lwvsCommon.errorMask |= CP_ERR_LWDA_ALLOC_FAIL;
            LOG_LWDA_ERROR("lwdaMalloc", lwSt, device->dcgmDeviceIndex, arrayByteSize);
            return -1;
        }
        lwSt = lwdaMalloc((void **)&device->deviceB, arrayByteSize);
        if (lwSt != lwdaSuccess)
        {
            lwvsCommon.errorMask |= CP_ERR_LWDA_ALLOC_FAIL;
            LOG_LWDA_ERROR("lwdaMalloc", lwSt, device->dcgmDeviceIndex, arrayByteSize);
            return -1;
        }

        device->NdeviceC = 0;
        for (int i = 0; i < CP_MAX_OUTPUT_MATRICES; i++)
        {
            lwSt = lwdaMalloc((void **)&device->deviceC[i], arrayByteSize);
            if (lwSt != lwdaSuccess)
            {
                lwvsCommon.errorMask |= CP_ERR_LWDA_ALLOC_FAIL;
                LOG_LWDA_ERROR("lwdaMalloc", lwSt, device->dcgmDeviceIndex, arrayByteSize);
                return -1;
            }
            device->NdeviceC++;
        }

        /* Copy the host arrays to the device arrays */
        lwSt = lwdaMemcpy(device->deviceA, m_hostA, arrayByteSize, lwdaMemcpyHostToDevice);
        if (lwSt != lwdaSuccess)
        {
            lwvsCommon.errorMask |= CP_ERR_LWDA_MEMCPY_FAIL;
            LOG_LWDA_ERROR("lwdaMemcpy", lwSt, device->dcgmDeviceIndex, arrayByteSize);
            return -1;
        }

        lwSt = lwdaMemcpy(device->deviceB, m_hostB, arrayByteSize, lwdaMemcpyHostToDevice);
        if (lwSt != lwdaSuccess)
        {
            lwvsCommon.errorMask |= CP_ERR_LWDA_MEMCPY_FAIL;
            LOG_LWDA_ERROR("lwdaMemcpy", lwSt, device->dcgmDeviceIndex, arrayByteSize);
            return -1;
        }

        lwSt = lwdaMemcpy(device->deviceC[0], m_hostC, arrayByteSize, lwdaMemcpyHostToDevice);
        if (lwSt != lwdaSuccess)
        {
            lwvsCommon.errorMask |= CP_ERR_LWDA_MEMCPY_FAIL;
            LOG_LWDA_ERROR("lwdaMemcpy", lwSt, device->dcgmDeviceIndex, arrayByteSize);
            return -1;
        }
        /* Copy the rest of the C arrays from the first C array */
        for (int i = 0; i < device->NdeviceC; i++)
        {
            lwSt = lwdaMemcpy(device->deviceC[i], device->deviceC[0], arrayByteSize, lwdaMemcpyDeviceToDevice);
            if (lwSt != lwdaSuccess)
            {
                lwvsCommon.errorMask |= CP_ERR_LWDA_MEMCPY_FAIL;
                LOG_LWDA_ERROR("lwdaMemcpy", lwSt, device->dcgmDeviceIndex, arrayByteSize);
                return -1;
            }
        }
    }

    return 0;
}

/*************************************************************************/
void ConstantPower::Go(TestParameters *testParameters, const std::vector<unsigned int> &gpuList)
{
    bool result;
    InitializeForGpuList(gpuList);

    if (!testParameters->GetBoolFromString(CP_STR_IS_ALLOWED))
    {
        DcgmError d;
        DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_TEST_DISABLED, d, CP_PLUGIN_NAME);
        AddError(d);
        SetResult(LWVS_RESULT_SKIP);
        return;
    }

    /* Cache test parameters */
    m_testParameters = testParameters; // DO NOT DELETE
    m_useDgemm = testParameters->GetBoolFromString(CP_STR_USE_DGEMM);
    m_testDuration = testParameters->GetDouble(CP_STR_TEST_DURATION);
    m_targetPower = testParameters->GetDouble(CP_STR_TARGET_POWER);
    m_sbeFailureThreshold = testParameters->GetDouble(CP_STR_SBE_ERROR_THRESHOLD);

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



/*************************************************************************/
bool ConstantPower::CheckGpuPowerUsage(CPDevice *device, std::vector<DcgmError> &errorList, 
                                       timelib64_t startTime, timelib64_t earliestStopTime)
{
    double                      maxVal;
    double                      avg;
    dcgmFieldSummaryRequest_t   fsr;
    
    memset(&fsr, 0, sizeof(fsr));
    fsr.fieldId = DCGM_FI_DEV_POWER_USAGE;
    fsr.entityGroupId = DCGM_FE_GPU;
    fsr.entityId = device->dcgmDeviceIndex;
    fsr.summaryTypeMask = DCGM_SUMMARY_MAX | DCGM_SUMMARY_AVG;
    fsr.startTime = startTime;
    fsr.endTime = earliestStopTime;

    dcgmReturn_t ret = m_dcgmRecorder.GetFieldSummary(fsr);

    if (ret != DCGM_ST_OK)
    {
        DcgmError d;
        DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_CANNOT_GET_STAT, d, "power usage", device->dcgmDeviceIndex);
        errorList.push_back(d);
        return false;
    }

    maxVal = fsr.response.values[0].fp64;
    double minRatio = m_testParameters->GetDouble(CP_STR_TARGET_POWER_MIN_RATIO);
    double maxRatio = m_testParameters->GetDouble(CP_STR_TARGET_POWER_MAX_RATIO);
    double minRatioTarget = minRatio * m_targetPower;
    double maxRatioTarget = maxRatio * m_targetPower;
        
    RecordObservedMetric(device->dcgmDeviceIndex, CP_STR_TARGET_POWER, maxVal);

    if (maxVal < minRatioTarget)
    {
        if (minRatioTarget >= device->maxPowerTarget)
        {
            // Just warn if the enforced power limit is lower than the minRatioTarget
            std::stringstream buf;
            buf.setf(std::ios::fixed, std::ios::floatfield);
            buf.precision(0);
            buf << "Max power of " << maxVal << " did not reach desired power minimum "
                << CP_STR_TARGET_POWER_MIN_RATIO << " of " << minRatioTarget 
                << " for GPU " << device->dcgmDeviceIndex 
                << " because the enforced power limit has been set to " << device->maxPowerTarget;
            AddInfoVerboseForGpu(device->dcgmDeviceIndex, buf.str());
        }
        else
        {
            lwvsCommon.errorMask |= CP_ERR_GPU_POWER_TOO_LOW;
            DcgmError d;
            DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_TARGET_POWER, d, maxVal, CP_STR_TARGET_POWER_MIN_RATIO,
                                      minRatioTarget, device->dcgmDeviceIndex);
            
            std::string utilNote = m_dcgmRecorder.GetGpuUtilizationNote(device->dcgmDeviceIndex,
                                                                        startTime, earliestStopTime);
            if (utilNote.empty() == false)
            {
                d.AddDetail(utilNote);
            }

            errorList.push_back(d);
            return false;
        }
    }

    // Add a message about the average power usage
    std::stringstream ss;
    avg = fsr.response.values[1].fp64;
    ss.setf(std::ios::fixed, std::ios::floatfield);
    ss.precision(0);
    ss << "GPU " << device->dcgmDeviceIndex << " power average:\t" << avg << " W";
    AddInfoVerboseForGpu(device->dcgmDeviceIndex, ss.str());

    return true;
}

/*************************************************************************/
bool ConstantPower::CheckGpuTemperature(CPDevice *device, std::vector<DcgmError> &errorList,
                                        timelib64_t startTime, timelib64_t earliestStopTime, bool testFinished)
{
    std::string infoMsg;
    long long   maxTemp = static_cast<long long>(m_testParameters->GetDouble(CP_STR_TEMPERATURE_MAX));
    long long   highTempObserved = 0;
    int st = m_dcgmRecorder.CheckGpuTemperature(device->dcgmDeviceIndex, errorList, maxTemp, infoMsg,
                                                      startTime, earliestStopTime, highTempObserved);

    if (testFinished && highTempObserved != 0)
    {
        RecordObservedMetric(device->dcgmDeviceIndex, CP_STR_TEMPERATURE_MAX, highTempObserved);
    }

    if (st == DR_VIOLATION)
    {
        lwvsCommon.errorMask |= CP_ERR_GPU_TEMP_TOO_HIGH;
    }
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

/*************************************************************************/
bool ConstantPower::CheckLwmlEvents(CPDevice *device, std::vector<DcgmError> &errorList, timelib64_t startTime, 
                                    timelib64_t earliestStopTime)
{
    std::vector<unsigned short> fieldIds;
    std::vector<dcgmTimeseriesInfo_t> *thresholdsPtr = 0;
    std::vector<dcgmTimeseriesInfo_t> failureThresholds;

    if (DCGM_FP64_IS_BLANK(m_sbeFailureThreshold) == 0)
    {
        // Only evaluate this field if a failure threshold is set
        fieldIds.push_back(DCGM_FI_DEV_ECC_SBE_VOL_TOTAL);

        dcgmTimeseriesInfo_t dti;
        memset(&dti, 0, sizeof(dti));
        dti.isInt = true;
        dti.val.i64 = static_cast<long long>(m_sbeFailureThreshold);
        failureThresholds.push_back(dti);

        // Set a failure threshold of 0 for the other two fields. This is not necessary if we are not checking
        // for SBE errors because DcgmRecorder::CheckErrorFields assumes a default threshold of 0 when
        // thresholdsPtr is NULL
        dti.val.i64 = 0;
        for (int i = 0; i < 2; i ++)
        {
            failureThresholds.push_back(dti);
        }
        thresholdsPtr = &failureThresholds;
    }

    fieldIds.push_back(DCGM_FI_DEV_ECC_DBE_VOL_TOTAL);
    fieldIds.push_back(DCGM_FI_DEV_XID_ERRORS);

    int st = m_dcgmRecorder.CheckErrorFields(fieldIds, thresholdsPtr, device->dcgmDeviceIndex, errorList,
                                              startTime, earliestStopTime);

    // Check for a communication error
    if (st == DR_COMM_ERROR)
    {
        m_dcgmCommErrorOclwrred = true;
    }

    // Return true if there are no errors, false otherwise
    return errorList.empty();
}

/*************************************************************************/
bool ConstantPower::CheckPassFailSingleGpu(CPDevice *device, std::vector<DcgmError> &errorList, 
                                           timelib64_t startTime, timelib64_t earliestStopTime, bool testFinished)
{
    DcgmLockGuard lock(&m_mutex); // prevent conlwrrent failure checks from workers   
    bool result;
    int st;

    if (testFinished)
    {
        /* This check is only run once the test is finished */
        result = CheckGpuPowerUsage(device, errorList, startTime, earliestStopTime);
        if (!result || m_dcgmCommErrorOclwrred)
        {
            return false;
        }
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

    /* Check GPU temperature against specified max temp */
    result = CheckGpuTemperature(device, errorList, startTime, earliestStopTime, testFinished);
    if (!result || m_dcgmCommErrorOclwrred)
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

/*************************************************************************/
bool ConstantPower::CheckPassFail(timelib64_t startTime, timelib64_t earliestStopTime)
{
    bool passed, allPassed = true;
    std::vector<DcgmError> errorList;
    std::vector<DcgmError> errorListAllGpus;
    char buf[256] = {0};

    if (m_testDuration < 30.0)
    {
        snprintf(buf, sizeof(buf), "Test duration of %.1f will not produce useful results as "
                 "this test takes at least 30 seconds to get to target power.", m_testDuration);
        AddInfo(buf);
    }

    /* Get latest values for watched fields before checking pass fail
     * If there are errors getting the latest values, error information is added to errorListAllGpus.
     */
    m_dcgmRecorder.GetLatestValuesForWatchedFields(0, errorListAllGpus);

    for (size_t i = 0; i < m_device.size(); i++)
    {
        if (m_device[i]->m_lowPowerLimit)
        {
            continue;
        }

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

bool ConstantPower::EnforcedPowerLimitTooLow()
{
    double minRatio = m_testParameters->GetDouble(CP_STR_TARGET_POWER_MIN_RATIO);
    double minRatioTarget = minRatio * m_targetPower;
    bool   allTooLow = true;

    for (size_t i = 0; i < m_device.size(); i++)
    {
        if (minRatioTarget >= m_device[i]->maxPowerTarget)
        {
            // Enforced power limit is too low. Skip the test.
            DcgmError d;
            DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_ENFORCED_POWER_LIMIT, d, m_device[i]->dcgmDeviceIndex,
                                      m_device[i]->maxPowerTarget);
            AddErrorForGpu(m_device[i]->dcgmDeviceIndex, d);
            SetResultForGpu(m_device[i]->dcgmDeviceIndex, LWVS_RESULT_SKIP);
            m_device[i]->m_lowPowerLimit = true;
        }
        else
        {
            allTooLow = false;
        }
    }

    return allTooLow;
}

/****************************************************************************/
class ConstantPowerWorker : public LwvsThread
{
private:
    CPDevice        *m_device;          /* Which device this worker thread is running on */
    ConstantPower   &m_plugin;          /* ConstantPower plugin for logging and failure checks */
    TestParameters  *m_testParameters;  /* Read-only test parameters */
    DcgmRecorder    &m_dcgmRecorder;
    int             m_useDgemm;         /* Wheter to use dgemm (1) or sgemm (0) for operations */
    double          m_targetPower;      /* Target stress in gflops */
    double          m_testDuration;     /* Target test duration in seconds */
    timelib64_t     m_stopTime;         /* Timestamp when run() finished */
    double          m_reAdjustInterval; /* How often to change the matrix size in seconds */
    double          m_printInterval;    /* How often to print out status to stdout */
    int             m_opsPerRequeue;    /* How many lwblas operations to queue to each stream each time we queue work
                                           to it */
    int             m_startingMatrixDim;/* Matrix size to start at when ramping up to target power. Since we ramp 
                                           up our matrix size slowly, setting this higher will decrease the ramp up 
                                           time needed */

public:
    ConstantPowerWorker(CPDevice *device, ConstantPower &plugin, TestParameters *tp, DcgmRecorder &dr);

    virtual ~ConstantPowerWorker() /* Virtual to satisfy ancient compiler */
    {}

    timelib64_t GetStopTime()
    {
        return m_stopTime;
    }

    /*****************************************************************************/
    /*
    * Worker thread main - streams version
    *
    */
    void run(void);

private:
    /*****************************************************************************/
    /*
     * Return the current power in watts of the device.
     * 
     * Returns < 0.0 on error
     */
    double ReadPower();

    /*****************************************************************************/
    /*
     * Callwlate the percent difference between a and b
     */
    static double PercentDiff(double a, double b);

    /*****************************************************************************/
    /*
     * Return the new matrix dimension to use for ramping up to the target power.
     */
    int RecalcMatrixDim(int lwrrentMatrixDim, double power);

};

/****************************************************************************/
/*
 * ConstantPower RunTest
 */
/****************************************************************************/
bool ConstantPower::RunTest(const std::vector<unsigned int> &gpuList)
{
    int st, Nrunning = 0;
    ConstantPowerWorker *workerThreads[CP_MAX_DEVICES] = {0};
    bool outputStats = true;
    unsigned int timeCount = 0;
    timelib64_t earliestStopTime;
    timelib64_t startTime = timelib_usecSince1970();
    bool testPassed;

    if (gpuList.size() < 1 || gpuList.size() > CP_MAX_DEVICES)
    {
        PRINT_ERROR("%d %d", "Bad gpuList size: %d (max allowed: %d)",
                    (int)gpuList.size(), CP_MAX_DEVICES);
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

    if (!LwmlInit(gpuList))
    {
        Cleanup();
        return false;
    }

    if (EnforcedPowerLimitTooLow())
    {
        Cleanup();
        // Returning false will produce a failure result, we are skipping
        return true;
    }

    st = LwdaInit();
    if (st)
    {
        // Errors added from LwdaInit, no need to add here
        Cleanup();
        return false;
    }

    std::string logFileName = m_testParameters->GetString(PS_LOGFILE);
    int logFileType = (int)m_testParameters->GetDouble(PS_LOGFILE_TYPE);

    /* Create the stats collection */
    std::vector<unsigned short> fieldIds;
    fieldIds.push_back(DCGM_FI_DEV_POWER_USAGE);
    fieldIds.push_back(DCGM_FI_DEV_MEM_CLOCK);
    fieldIds.push_back(DCGM_FI_DEV_SM_CLOCK);
    fieldIds.push_back(DCGM_FI_DEV_POWER_VIOLATION);
    fieldIds.push_back(DCGM_FI_DEV_THERMAL_VIOLATION);
    fieldIds.push_back(DCGM_FI_DEV_GPU_TEMP);
    fieldIds.push_back(DCGM_FI_DEV_XID_ERRORS);
    fieldIds.push_back(DCGM_FI_DEV_ECC_SBE_VOL_TOTAL);
    fieldIds.push_back(DCGM_FI_DEV_ECC_DBE_VOL_TOTAL);
    fieldIds.push_back(DCGM_FI_DEV_CLOCK_THROTTLE_REASONS);
    fieldIds.push_back(DCGM_FI_DEV_GPU_UTIL);
    m_dcgmRecorder.AddWatches(fieldIds, gpuList, false, "targeted_power_field_group", "targeted_power_group",
                              m_testDuration);

    
    try /* Catch runtime errors */
    {

        /* Create and start all workers */
        for (size_t i = 0; i < m_device.size(); i++)
        {
            if (m_device[i]->m_lowPowerLimit == false)
            {
                workerThreads[i] = new ConstantPowerWorker(m_device[i], *this, m_testParameters, m_dcgmRecorder);
                workerThreads[i]->Start();
                Nrunning++;
            }
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
                // If a worker was not initialized, we skip over it (e.g. we caught a bad_alloc exception)
                if (workerThreads[i] == NULL)
                {
                    continue;
                }

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
        PRINT_ERROR("%s", "Caught runtime_error %s", e.what());
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

    /* Clean up the worker threads */
    earliestStopTime = INT64_MAX;
    for (size_t i = 0; i < m_device.size(); i++)
    {
        // If a worker was not initialized, we skip over it (e.g. we caught a bad_alloc exception)
        if (workerThreads[i] == NULL)
        {
            continue;
        }

        earliestStopTime = MIN(earliestStopTime, workerThreads[i]->GetStopTime());
        delete(workerThreads[i]);
        workerThreads[i] = NULL;
    }
    
    PRINT_DEBUG("%lld", "Workers stopped. Earliest stop time: %lld", (long long)earliestStopTime);

    progressOut->updatePluginProgress(0, true);
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
 * ConstantPowerWorker implementation.
 */
/****************************************************************************/
ConstantPowerWorker::ConstantPowerWorker(CPDevice *device, ConstantPower &plugin, TestParameters *tp, DcgmRecorder &dr):
        m_device(device), m_plugin(plugin), m_testParameters(tp), m_dcgmRecorder(dr), m_stopTime(0)
{
    m_useDgemm = tp->GetBoolFromString(CP_STR_USE_DGEMM);
    m_targetPower = tp->GetDouble(CP_STR_TARGET_POWER);
    m_testDuration = tp->GetDouble(CP_STR_TEST_DURATION);
    m_reAdjustInterval = tp->GetDouble(CP_STR_READJUST_INTERVAL);
    m_printInterval = tp->GetDouble(CP_STR_PRINT_INTERVAL);
    m_opsPerRequeue = (int)tp->GetDouble(CP_STR_OPS_PER_REQUEUE);
    m_startingMatrixDim = (int)tp->GetDouble(CP_STR_STARTING_MATRIX_DIM);
}

/****************************************************************************/
double ConstantPowerWorker::ReadPower()
{
    dcgmReturn_t st;
    dcgmFieldValue_v2 powerUsage;
    unsigned int uintPower;

    st = m_dcgmRecorder.GetLwrrentFieldValue(m_device->dcgmDeviceIndex, DCGM_FI_DEV_POWER_USAGE, powerUsage, 
                                             0);
    if (st)
    {
        lwvsCommon.errorMask |= CP_ERR_BIT_ERROR;
        // We do not add a warning or stop the test because we want to allow some tolerance for when we cannot 
        // read the power. Instead we log the error and return -1 as the power value
        PRINT_ERROR("%d %s", "Could not retrieve power reading for GPU %d. DcgmRecorder returned: %s",
                    m_device->dcgmDeviceIndex, errorString(st));
        return -1.0;
    }

    return powerUsage.value.dbl; // power usage in watts
}

/****************************************************************************/
double ConstantPowerWorker::PercentDiff(double a, double b)
{
    double retVal = a - b;
    retVal /= (a + b);
    retVal *= 200.0;
    return retVal;
}

/****************************************************************************/
int ConstantPowerWorker::RecalcMatrixDim(int lwrrentMatrixDim, double power)
{
    int matrixDim;
    double pctDiff, workPctDiff;

    /* if we're targeting close to max power, just go for it  */
    if (m_targetPower >= (0.90 * m_device->maxPowerTarget))
    {
        return CP_MAX_DIMENSION;
    }

    pctDiff = PercentDiff(power, m_targetPower);

    matrixDim = lwrrentMatrixDim;

    /* If we are below our target power, set a floor so that we never go below this matrix size */
    if (pctDiff < 0.0)
    {
        m_device->minMatrixDim = MAX(lwrrentMatrixDim, m_device->minMatrixDim);
        PRINT_DEBUG("%d %d", "device %u, minMatrixDim: %d\n", m_device->lwmlDeviceIndex, lwrrentMatrixDim);
    }

    /* Ramp up */
    if (!m_device->onlySmallAdjustments && pctDiff <= -50.0)
    {
        matrixDim += 20; /* Ramp up */
    }
    else if (!m_device->onlySmallAdjustments && (pctDiff <= -5.0 || pctDiff >= 5.0))
    {
        /* Try to guess jump in load based on pct change desired and pct change in matrix ops */
        if (pctDiff < 0.0)
        {
            for (workPctDiff = 0.0; workPctDiff < (-pctDiff) && matrixDim < CP_MAX_DIMENSION; matrixDim++)
            {
                workPctDiff = PercentDiff(matrixDim * matrixDim, lwrrentMatrixDim * lwrrentMatrixDim);
                //printf("loop pctdiff %.2f. workPctDiff %.2f\n", pctDiff, workPctDiff);
            }
        }
         else
        {
            for (workPctDiff = 0.0; workPctDiff > (-pctDiff) && matrixDim > m_device->minMatrixDim; matrixDim--)
            {
                workPctDiff = PercentDiff(matrixDim * matrixDim, lwrrentMatrixDim * lwrrentMatrixDim);
                //printf("loop2 pctdiff %.2f. workPctDiff %.2f\n", pctDiff, workPctDiff);
            }
        }
    }
    else if (pctDiff < 0.0)
    {
        matrixDim++; /* Very small adjustment */
        //m_device->onlySmallAdjustments = 1; /* Continue to make large adjustments if need be */
    }
    else
    {
        matrixDim--; /* Very small adjustment */
        //m_device->onlySmallAdjustments = 1; /* Continue to make large adjustments if need be */
    }

    //printf("pctdiff %.2f\n", pctDiff);

    if (matrixDim < 1)
    {
        matrixDim = 1;
    }
    if (matrixDim > CP_MAX_DIMENSION)
    {
        matrixDim = CP_MAX_DIMENSION;
    }

    return matrixDim;
}

/****************************************************************************/
void ConstantPowerWorker::run()
{
    int j;
    double alpha, beta;
    float floatAlpha, floatBeta;
    double startTime;
    double lastAdjustTime = 0.0; /* Last time we changed matrixDim */
    double lastPrintTime = 0.0;  /* last time we printed out the current power */
    double lastFailureCheckTime = 0.0;  /* last time we checked for failures */
    double now;
    double power;
    int useNstreams;
    int NstreamsRequeued = 0;
    int matrixDim = 1; /* Dimension of the matrix. Start small */
    lwblasStatus_t lwbSt;

    if (m_device->lwvsDevice->SetCpuAffinity())
    {
        char buf[1024];
        snprintf(buf, sizeof(buf), "Top performance cannot be guaranteed for GPU %u because we could "
                "not set cpu affinity.", m_device->dcgmDeviceIndex);
        m_plugin.AddInfoVerboseForGpu(m_device->dcgmDeviceIndex, buf);
    }

    /* Set initial test values */
    useNstreams = (int)m_testParameters->GetDouble(CP_STR_LWDA_STREAMS_PER_GPU);
    matrixDim = m_startingMatrixDim;
    alpha = 1.01 + ((double)(rand() % 100)/10.0);
    beta = 1.01 + ((double)(rand() % 100)/10.0);
    floatAlpha = (float)alpha;
    floatBeta = (float)beta;

    /* Lock to our assigned GPU */
    lwdaSetDevice(m_device->lwdaDeviceIdx);

   // printf("Running for %.1f seconds\n", m_testDuration);
    startTime = timelib_dsecSince1970();
    lastPrintTime = startTime;
    lastFailureCheckTime = startTime;
    std::vector<DcgmError> errorList;

    while (timelib_dsecSince1970() - startTime < m_testDuration && !ShouldStop())
    {
        NstreamsRequeued = 0;

        for (int i = 0; i < useNstreams; i++)
        {
            /* Query each stream to see if it's idle (lwdaSuccess return) */
            if (lwdaSuccess == lwdaStreamQuery(m_device->lwdaStream[i]))
            {
                for (j = 0; j < m_opsPerRequeue; j++)
                {
                    int Cindex = ((i * useNstreams) + j) % m_device->NdeviceC;

                    lwbSt = lwblasSetStream(m_device->lwblasHandle, m_device->lwdaStream[i]);
                    if (lwbSt != LWBLAS_STATUS_SUCCESS)
                    {
                        LOG_LWBLAS_ERROR_FOR_PLUGIN(&m_plugin, "lwblasSetStream", lwbSt, m_device->dcgmDeviceIndex);
                        m_stopTime = timelib_usecSince1970();
                        return;
                    }
                    /* Make sure all streams have work. These are async calls, so they will
                       return immediately */
                    if (m_useDgemm)
                    {
                        lwbSt = lwblasDgemm(m_device->lwblasHandle, LWBLAS_OP_N, LWBLAS_OP_N, matrixDim, matrixDim, 
                                            matrixDim, &alpha, (double *)m_device->deviceA, matrixDim, 
                                            (double *)m_device->deviceB, matrixDim, &beta,
                                            (double *)m_device->deviceC[Cindex], matrixDim);
                        if (lwbSt != LWBLAS_STATUS_SUCCESS)
                        {
                            LOG_LWBLAS_ERROR_FOR_PLUGIN(&m_plugin, "lwblasDgemm", lwbSt, m_device->dcgmDeviceIndex);
                            m_stopTime = timelib_usecSince1970();
                            return;
                        }
                    }
                    else
                    {
                        lwbSt = lwblasSgemm(m_device->lwblasHandle, LWBLAS_OP_N, LWBLAS_OP_N, matrixDim, matrixDim, 
                                            matrixDim, &floatAlpha, (float *)m_device->deviceA, matrixDim, 
                                            (float *)m_device->deviceB, matrixDim, &floatBeta, 
                                            (float *)m_device->deviceC[Cindex], matrixDim);
                        if (lwbSt != LWBLAS_STATUS_SUCCESS)
                        {
                            LOG_LWBLAS_ERROR_FOR_PLUGIN(&m_plugin, "lwblasSgemm", lwbSt, m_device->dcgmDeviceIndex);
                            m_stopTime = timelib_usecSince1970();
                            return;
                        }

                    }
                }
                NstreamsRequeued++;
            }
        }

        /* If we didn't queue any work, sleep a bit so we don't busy wait */
        if (!NstreamsRequeued)
        {
            usleep(1000);
        }

        now = timelib_dsecSince1970();

        /* Time to adjust? */
        if (now - lastAdjustTime > m_reAdjustInterval)
        {
            power = ReadPower();
            matrixDim = RecalcMatrixDim(matrixDim, power);
            lastAdjustTime = now;
        }

        /* Time to print? */
        if (now - lastPrintTime > m_printInterval)
        {
            power = ReadPower();
            PRINT_DEBUG("%d %f %d %d", "DeviceIdx %d, Power %.2f W. dim: %d. minDim: %d\n",
                        m_device->dcgmDeviceIndex, power, matrixDim, m_device->minMatrixDim);
            lastPrintTime = now;
        }
        /* Time to check for failure? */
        if (lwvsCommon.failEarly && now - lastFailureCheckTime > lwvsCommon.failCheckInterval)
        {
            bool result;
            result = m_plugin.CheckPassFailSingleGpu(m_device, errorList, lastFailureCheckTime * 1000000, 
                                                     now * 1000000, false);
            if (!result)
            {
                // Stop the test because a failure oclwrred
                PRINT_DEBUG("%d", "Test failure detected for GPU %d. Stopping test early.", m_device->dcgmDeviceIndex);
                break;
            }
            lastFailureCheckTime = now;
        }
    }
    m_stopTime = timelib_usecSince1970();
    PRINT_DEBUG("%d %lld", "ConstantPowerWorker deviceIndex %d finished at %lld", m_device->dcgmDeviceIndex,
                (long long)m_stopTime);
}

extern "C" {
    Plugin *maker() {
        return new ConstantPower;
    }
    class proxy {
    public:
        proxy()
        {
            factory["Constant Power"] = maker;
        }
    };    
    proxy p;
}   
