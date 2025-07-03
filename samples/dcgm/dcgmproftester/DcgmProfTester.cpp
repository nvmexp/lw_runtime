#include "DcgmProfTester.h"
#include "LwcmSettings.h"
#include "lwblas_v2.h"
#include "lwca-loader.h"
#include "lwca.h"
#include "dcgm_fields.h"
#include "dcgm_fields_internal.h"
#include "logging.h"
#include "timelib.h"
#include "vector_types.h"

#include <tclap/CmdLine.h>
#include <tclap/Arg.h>
#include <tclap/ValueArg.h>
#include <tclap/ValuesConstraint.h>
#include <tclap/SwitchArg.h>

#include <pthread.h>
#include <vector>

#define DCGMPROFTESTER_VERSION "1.1.0"


/*****************************************************************************/
/* ctor/dtor */
DcgmProfTester::DcgmProfTester()
{
    m_lwdaDevice = 0;
    m_lwdaContext = NULL;
    memset(&m_dcgmDeviceAttr, 0, sizeof(m_dcgmDeviceAttr));
    m_duration = 30.0;
    m_targetMaxValue = false;
    m_gpuId = 0;
    m_dcgmIsInitialized = false;
    m_dcgmHandle = NULL;
    m_groupId = NULL;
    m_fieldGroupId = NULL;
    m_testFieldId = DCGM_FI_PROF_SM_ACTIVE;
    m_sinceTimestamp = 0;
    m_startDcgm = true;
    m_dvsOutput = false;
}

/*****************************************************************************/
DcgmProfTester::~DcgmProfTester()
{
    if(m_lwdaContext != NULL)
    {
        lwCtxDestroy(m_lwdaContext);
        m_lwdaContext = NULL;
    }

    if(m_dcgmIsInitialized)
    {
        dcgmShutdown();
    }
}

/*****************************************************************************/
dcgmReturn_t DcgmProfTester::ParseCommandLine(int argc, char *argv[])
{
    try
    {
        class FieldIdConstraint_t : public TCLAP::Constraint<unsigned int>
        {
        public:
            virtual std::string description() const
            {
                std::stringstream ss;
                ss << "Valid value for the FieldId is an integer between " << DCGM_FI_PROF_FIRST_ID << " and "
                   << DCGM_FI_PROF_LAST_ID;
                return ss.str();
            }

            virtual std::string shortID() const
            {
                return "FieldId";
            }

            virtual bool check(unsigned int const& value) const
            {
                return DCGM_FI_PROF_FIRST_ID <= value && value <= DCGM_FI_PROF_LAST_ID;
            }
        };

        class DurationConstraint_t : public TCLAP::Constraint<double>
        {
        public:
            virtual std::string description() const
            {
                return "Duration sould be longer that 1 second";
            }

            virtual std::string shortID() const
            {
                return "Duration in seconds";
            }

            virtual bool check(double const& value) const
            {
                return value >= 1.0;
            }
        };

        class GpuIdConstraint_t : public TCLAP::Constraint<unsigned int>
        {
        public:
            virtual std::string description() const
            {
                return "gpuId should be between 0 and 15";
            }

            virtual std::string shortID() const
            {
                return "gpuId to run on";
            }

            virtual bool check(unsigned int const& value) const
            {
                return value < DCGM_MAX_NUM_DEVICES;
            }
        };

        FieldIdConstraint_t fieldIdConstraint;
        DurationConstraint_t durationConstraint;
        GpuIdConstraint_t gpuIdConstraint;

        TCLAP::CmdLine cmd("dcgmproftester", ' ', DCGMPROFTESTER_VERSION);

        TCLAP::ValueArg<double> durationArg(
            "d", "duration", "Duration of the test in seconds", false, m_duration, &durationConstraint, cmd);

        TCLAP::ValueArg<unsigned int> fieldIdArg(
            "t", "fieldId", "Profiling FieldId", false, m_testFieldId, &fieldIdConstraint, cmd);

        TCLAP::SwitchArg maxArg(
            "m", "target-max-value", "Target maximum value", cmd, false);

        TCLAP::ValueArg<unsigned int> gpuIdArg(
            "i", "gpuid", "GPU IDs to run on. Use dcgmi discovery -l to list valid gpuIds.", 
            false, m_gpuId, &gpuIdConstraint, cmd);
        
        TCLAP::SwitchArg noDcgmArg(
            "", "no-dcgm-validation", "If set, we will NOT self-validate DCGM metrics. "
            "This is useful for using this tool to generate a workload rather "
            "than self-validating the DCP metrics.", 
            cmd, false);
        
        TCLAP::SwitchArg dvsOutput(
            "", "dvs", "If set, we will append DVS tags to our stdout so this tool can be used in DVS", 
            cmd, false);

        cmd.parse(argc, argv);
        m_duration    = durationArg.getValue();
        m_testFieldId = fieldIdArg.getValue();
        m_targetMaxValue = maxArg.getValue();
        m_gpuId = gpuIdArg.getValue();
        m_startDcgm = !noDcgmArg.getValue();
        m_dvsOutput = dvsOutput.getValue();

    }
    catch (TCLAP::ArgException const& ex)
    {
        std::cerr << "Error: " << ex.argId() << " " << ex.error() << std::endl;
        return DCGM_ST_BADPARAM;
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmProfTester::LoadLwdaModule(void)
{
    LWresult lwSt;
    int i;

    lwSt = lwCtxSetLwrrent(m_lwdaContext);
    if(lwSt)
    {
        fprintf(stderr, "lwCtxSetLwrrent failed. lwSt: %d\n", lwSt);
        return DCGM_ST_GENERIC_ERROR;
    }

    /* Load our lwca module so we can find all of the functions */
    lwSt = lwModuleLoad(&m_lwdaModule, "DcgmProfTesterKernels.ptx");
    if(lwSt)
    {
        fprintf(stderr, "Unable to load lwca module DcgmProfTesterKernels.ptx. lwSt: %d\n", lwSt);
        return DCGM_ST_GENERIC_ERROR;
    }

    /* Load functions from our lwca module */
    lwSt = lwModuleGetFunction(&m_lwFuncWaitNs, m_lwdaModule, "waitNs");
    if(lwSt)
    {
        fprintf(stderr, "Unable to load lwca function %s. lwSt: %d\n", "waitNs\n", lwSt);
        return DCGM_ST_GENERIC_ERROR;
    }

    /* Load functions from our lwca module */
    lwSt = lwModuleGetFunction(&m_lwFuncWaitCycles, m_lwdaModule, "waitCycles");
    if(lwSt)
    {
        fprintf(stderr, "Unable to load lwca function %s. lwSt: %d\n", "waitCycles\n", lwSt);
        return DCGM_ST_GENERIC_ERROR;
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmProfTester::LwdaInit(void)
{
    int i, j, st, valueSize;
    LWresult lwSt;
    unsigned int hostAllocFlags = 0;

    /* Load the lwca library */
    if (LWDA_LIBRARY_LOAD_SUCCESS != loadDefaultLwdaLibrary())
    {
        fprintf(stderr, "Unable to load the LWCA library, check for correct LD_LIBRARY_PATH\n");
        return DCGM_ST_GENERIC_ERROR;
    }

    lwSt = lwInit(0);
    if(lwSt)
    {
        fprintf(stderr, "lwInit returned %d\n", lwSt);
        return DCGM_ST_GENERIC_ERROR;
    }

    /* Find the corresponding lwca device to our DCGM device */
    lwSt = lwDeviceGetByPCIBusId(&m_lwdaDevice, m_dcgmDeviceAttr.identifiers.pciBusId);
    if(lwSt)
    {
        fprintf(stderr, "lwDeviceGetByPCIBusId returned %d for busId %s\n", lwSt, m_dcgmDeviceAttr.identifiers.pciBusId);
        return DCGM_ST_GENERIC_ERROR;
    }


    /* Do per-device initialization */
    lwSt = lwCtxCreate(&m_lwdaContext, 0, m_lwdaDevice);
    if(lwSt)
    {
        fprintf(stderr, "lwCtxCreate returned %d", lwSt);
        return DCGM_ST_GENERIC_ERROR;
    }

    lwSt = lwDeviceGetAttribute(&m_maxThreadsPerMultiProcessor, LW_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR, m_lwdaDevice);
    if(lwSt)
    {
        fprintf(stderr, "lwDeviceGetAttribute returned %d\n", lwSt);
        return DCGM_ST_GENERIC_ERROR;
    }
    printf("LW_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR: %d\n", m_maxThreadsPerMultiProcessor);

    lwSt = lwDeviceGetAttribute(&m_multiProcessorCount, LW_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, m_lwdaDevice);
    if(lwSt)
    {
        fprintf(stderr, "lwDeviceGetAttribute returned %d\n", lwSt);
        return DCGM_ST_GENERIC_ERROR;
    }
    printf("LW_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT: %d\n", m_multiProcessorCount);

    lwSt = lwDeviceGetAttribute(&m_sharedMemPerMultiprocessor, LW_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR, m_lwdaDevice);
    if(lwSt)
    {
        fprintf(stderr, "lwDeviceGetAttribute returned %d\n", lwSt);
        return DCGM_ST_GENERIC_ERROR;
    }
    printf("LW_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR: %d\n", m_sharedMemPerMultiprocessor);

    lwSt = lwDeviceGetAttribute(&m_computeCapabilityMajor, LW_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, m_lwdaDevice);
    if(lwSt)
    {
        fprintf(stderr, "lwDeviceGetAttribute returned %d\n", lwSt);
        return DCGM_ST_GENERIC_ERROR;
    }
    printf("LW_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: %d\n", m_computeCapabilityMajor);

    lwSt = lwDeviceGetAttribute(&m_computeCapabilityMinor, LW_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, m_lwdaDevice);
    if(lwSt)
    {
        fprintf(stderr, "lwDeviceGetAttribute returned %d\n", lwSt);
        return DCGM_ST_GENERIC_ERROR;
    }
    printf("LW_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: %d\n", m_computeCapabilityMinor);

    m_computeCapability = (double)m_computeCapabilityMajor + ((double)m_computeCapabilityMinor / 10.0);

    lwSt = lwDeviceGetAttribute(&m_memoryBusWidth, LW_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH, m_lwdaDevice);
    if(lwSt)
    {
        fprintf(stderr, "lwDeviceGetAttribute returned %d\n", lwSt);
        return DCGM_ST_GENERIC_ERROR;
    }
    printf("LW_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH: %d\n", m_memoryBusWidth);
    

    lwSt = lwDeviceGetAttribute(&m_maxMemoryClockMhz, LW_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, m_lwdaDevice);
    if(lwSt)
    {
        fprintf(stderr, "lwDeviceGetAttribute returned %d\n", lwSt);
        return DCGM_ST_GENERIC_ERROR;
    }
    printf("LW_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE: %d\n", m_maxMemoryClockMhz);
    /* Colwert to MHz */
    m_maxMemoryClockMhz /= 1000;

    /* memory bandwidth in bytes = memClockMhz * 1000000 bytes per MiB * 2 copies per cycle. 
       bitWidth / 8 bits per byte */
    m_maximumMemBandwidth = (double)m_maxMemoryClockMhz * 1000000.0 * 2.0 * (double)m_memoryBusWidth / 8.0;
    printf("Max Memory bandwidth: %.0f bytes (%.2f GiB)\n", 
           m_maximumMemBandwidth, m_maximumMemBandwidth / (1.0e9));

    /* The modules must be loaded after we've created our contexts */
    st = LoadLwdaModule();
    if(st)
    {
        fprintf(stderr, "LoadLwdaModule failed with %d\n", st);
        return DCGM_ST_GENERIC_ERROR;
    }

    printf("LwdaInit completed successfully.\n\n");

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmProfTester::CreateDcgmGroups(void)
{
    dcgmReturn_t dcgmReturn = DCGM_ST_OK;

    if(!m_startDcgm)
    {
        printf("Skipping CreateDcgmGroups() since DCGM validation is disabled\n");
        return DCGM_ST_OK;
    }

    char groupName[16] = {0};
    snprintf(groupName, sizeof(groupName), "dpt_%d", getpid());

    dcgmReturn = dcgmGroupCreate(m_dcgmHandle, DCGM_GROUP_EMPTY, groupName, &m_groupId);
    if(dcgmReturn != DCGM_ST_OK)
    {
        fprintf(stderr, "dcgmGroupCreate() returned %d\n", dcgmReturn);
        return dcgmReturn;
    }

    dcgmReturn = dcgmGroupAddDevice(m_dcgmHandle, m_groupId, m_gpuId);
    if(dcgmReturn != DCGM_ST_OK)
    {
        fprintf(stderr, "dcgmGroupAddDevice() returned %d\n", dcgmReturn);
        return dcgmReturn;
    }

    /* Note: using groupName again on purpose since field groups and GPU groups are keyed separately */
    unsigned short fieldId = m_testFieldId;
    dcgmReturn = dcgmFieldGroupCreate(m_dcgmHandle, 1, &fieldId, groupName, &m_fieldGroupId);
    if(dcgmReturn != DCGM_ST_OK)
    {
        fprintf(stderr, "dcgmFieldGroupCreate() returned %d\n", dcgmReturn);
        return dcgmReturn;
    }

    return dcgmReturn;
}

/*****************************************************************************/
dcgmReturn_t DcgmProfTester::WatchFields(long long updateIntervalUsec)
{
    dcgmReturn_t dcgmReturn = DCGM_ST_OK;
    
    int maxKeepSamples = 0; /* Use maxKeepAge instead */
    double maxKeepAge = m_duration; /* Keep data for the entire test run */

    if(!m_startDcgm)
    {
        printf("Skipping WatchFields() since DCGM validation is disabled\n");
        return DCGM_ST_OK;
    }

    dcgmReturn = dcgmWatchFields(m_dcgmHandle, m_groupId, m_fieldGroupId, 
                                 updateIntervalUsec, maxKeepAge, maxKeepSamples);
    if(dcgmReturn == DCGM_ST_REQUIRES_ROOT)
    {
        fprintf(stderr, "Profiling requires running as root.\n");
    }
    else if(dcgmReturn == DCGM_ST_PROFILING_NOT_SUPPORTED)
    {
        fprintf(stderr, "Profiling is not supported for gpuId %u\n", m_gpuId);
    }
    else if(dcgmReturn == DCGM_ST_INSUFFICIENT_DRIVER_VERSION)
    {
        fprintf(stderr, "Either your driver is older than 418.75 (TRD3) or you "
                "are not running dcgmproftester as root.\n");
    }
    else if(dcgmReturn == DCGM_ST_IN_USE)
    {
        fprintf(stderr, "Another process is already using the profiling infrastucture. "
                "If lw-hostengine is running on your box, please kill it before running "
                "dcgmproftester or use the --no-dcgm-validation option to only generate a workload\n");
    }
    else if(dcgmReturn == DCGM_ST_NOT_SUPPORTED)
    {
        fprintf(stderr, "Field ID %u is is not supported for your GPU.\n", m_testFieldId);
    }
    else if(dcgmReturn != DCGM_ST_OK)
    {
        fprintf(stderr, "dcgmWatchFields() returned %d\n", dcgmReturn);
    }

    return dcgmReturn;
}

/*****************************************************************************/
dcgmReturn_t DcgmProfTester::UnwatchFields(void)
{
    dcgmReturn_t dcgmReturn;

    if(!m_startDcgm)
    {
        printf("Skipping UnwatchFields() since DCGM validation is disabled\n");
        return DCGM_ST_OK;
    }
    
    int maxKeepSamples = 0; /* Use maxKeepAge instead */
    double maxKeepAge = m_duration; /* Keep data for the entire test run */

    dcgmReturn = dcgmUnwatchFields(m_dcgmHandle, m_groupId, m_fieldGroupId);
    if(dcgmReturn != DCGM_ST_OK)
    {
        fprintf(stderr, "dcgmUnwatchFields() returned %d\n", dcgmReturn);
        return dcgmReturn;
    }

    return dcgmReturn;
}

/*****************************************************************************/
dcgmReturn_t DcgmProfTester::DcgmInit(void)
{
    dcgmReturn_t dcgmReturn = dcgmInit();
    if(dcgmReturn != DCGM_ST_OK)
    {
        fprintf(stderr, "dcgmInit() returned %d\n", dcgmReturn);
        return dcgmReturn;
    }

    dcgmReturn = dcgmStartEmbedded(DCGM_OPERATION_MODE_AUTO, &m_dcgmHandle);
    if(dcgmReturn != DCGM_ST_OK)
    {
        fprintf(stderr, "dcgmStartEmbedded() returned %d\n", dcgmReturn);
        return dcgmReturn;
    }

    int count = 0;
    unsigned int gpuIdList[DCGM_MAX_NUM_DEVICES];
    dcgmReturn = dcgmGetAllDevices(m_dcgmHandle, gpuIdList, &count);
    if(dcgmReturn != DCGM_ST_OK)
    {
        fprintf(stderr, "dcgmStartEmbedded() returned %d\n", dcgmReturn);
        return dcgmReturn;
    }

    if(count < 1)
    {
        fprintf(stderr, "DCGM found 0 GPUs. There's nothing to test on.\n");
        return DCGM_ST_GENERIC_ERROR;
    }

    bool found = false;
    for(int i = 0; i < count; i++)
    {
        if(gpuIdList[i] == m_gpuId)
        {
            found = true;
            break;
        }
    }
    if(!found)
    {
        fprintf(stderr, "Unable to find gpuId %u in %d gpuIds returned from DCGM. "
                "Pass a different value to -i\n", m_gpuId, count);
        return DCGM_ST_BADPARAM;
    }

    m_dcgmDeviceAttr.version = dcgmDeviceAttributes_version;
    dcgmReturn = dcgmGetDeviceAttributes(m_dcgmHandle, m_gpuId, &m_dcgmDeviceAttr);
    if(dcgmReturn != DCGM_ST_OK)
    {
        fprintf(stderr, "dcgmGetDeviceAttributes() returned %d\n", dcgmReturn);
        return dcgmReturn;
    }

    dcgmReturn = CreateDcgmGroups();
    if(dcgmReturn != DCGM_ST_OK)
        return dcgmReturn;

    m_dcgmIsInitialized = true;
    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmProfTester::Init(int argc, char *argv[])
{
    dcgmReturn_t dcgmReturn = DCGM_ST_OK;

    dcgmReturn = ParseCommandLine(argc, argv);
    if(dcgmReturn != DCGM_ST_OK)
        return dcgmReturn;

    /* Start DCGM before we initialize lwca since we're going to find the lwca
       device based on the PCI busId of our DCGM gpuId */
    dcgmReturn = DcgmInit();
    if(dcgmReturn != DCGM_ST_OK)
        return dcgmReturn;

    dcgmReturn = LwdaInit();
    if(dcgmReturn != DCGM_ST_OK)
        return dcgmReturn;

    return dcgmReturn;
}

/*****************************************************************************/
dcgmReturn_t DcgmProfTester::BeginSubtest(std::string testTitle, 
                                          std::string testTag, 
                                          bool isLinearTest)
{
    if(m_subtestInProgress)
        EndSubtest();
    
    if(m_dvsOutput)
        printf("&&&& RUNNING %s\n", testTag.c_str());

    m_subtestDcgmValues.clear();
    m_subtestGelwalues.clear();
    m_subtestTitle = testTitle;
    m_subtestTag = testTag;
    m_subtestIsLinear = isLinearTest;

    m_subtestInProgress = true;
    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmProfTester::EndSubtest(void)
{
    if(!m_subtestInProgress)
        return DCGM_ST_OK; /* Nothing to do */
    
    /* Todo: check the two arrays to make sure they are similar
    */

    m_subtestInProgress = false;

    char filename[128] = {0};
    snprintf(filename, sizeof(filename), "%s.results", m_subtestTag.c_str());
    FILE *testResultsFp = fopen(filename, "wt");
    if(!testResultsFp)
    {
        int cachedErrno = errno;
        fprintf(stderr, "Unable to open %s (errno %d). Can't write test results file.\n", 
                filename, cachedErrno);
        return DCGM_ST_GENERIC_ERROR;
    }

    fprintf(testResultsFp, "# TestTitle: \"%s\"\n", m_subtestTitle.c_str());

    fprintf(testResultsFp, "# Columns: \"generated\", \"dcgm\"\n");

    for(size_t i = 0; i < m_subtestDcgmValues.size(); i++)
    {
        fprintf(testResultsFp, "%.3f, %.3f\n", m_subtestGelwalues[i], m_subtestDcgmValues[i]);
    }

    fprintf(testResultsFp, "# TestResult: PASSED\n"); /* Todo: check and actually change this to WARNING or FAILED */
    fprintf(testResultsFp, "# TestResultReason: \n"); /* Todo: Populate with optional text */

    if(m_dvsOutput)
        printf("&&&& PASSED %s\n", m_subtestTag.c_str());

    fclose(testResultsFp);
    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmProfTester::AppendSubtestRecord(double generatedValue, double dcgmValue)
{
    m_subtestGelwalues.push_back(generatedValue);
    m_subtestDcgmValues.push_back(dcgmValue);
    return DCGM_ST_OK;
}

/*****************************************************************************/
static int dptGetLatestDcgmValueCB(unsigned int gpuId, dcgmFieldValue_v1 *values,
                                   int numValues, void *userData)
{
    std::vector<dcgmFieldValue_v1> *valuesVector = (std::vector<dcgmFieldValue_v1> *)userData;
    
    /* NOTE: This assumes we only have one entity in our group and one fieldId */

    valuesVector->insert(valuesVector->end(), values, values+numValues);
    return 0;
}

/*****************************************************************************/
dcgmReturn_t DcgmProfTester::GetLatestDcgmValue(dcgmFieldValue_v1 *value)
{
    dcgmReturn_t dcgmReturn;
    long long nextSinceTimestamp = 0;

    memset(value, 0, sizeof(*value));

    if(!m_startDcgm)
        return DCGM_ST_OK;

    dcgmReturn = dcgmGetValuesSince(m_dcgmHandle, m_groupId, m_fieldGroupId, m_sinceTimestamp, 
                                    &nextSinceTimestamp, dptGetLatestDcgmValueCB, &m_dcgmValues);

    if(dcgmReturn == DCGM_ST_OK)
    {
        m_sinceTimestamp = nextSinceTimestamp;

        if(m_dcgmValues.size() > 0)
        {
            *value = m_dcgmValues[m_dcgmValues.size()-1];
        }
        return DCGM_ST_OK;
    }
    
    fprintf(stderr, "dcgmGetValuesSince returned %d\n", dcgmReturn);
    return dcgmReturn;
}

/*****************************************************************************/
int DcgmProfTester::RunSleepKernel(unsigned int numSms, unsigned int threadsPerSm, unsigned int runForUsec)
{
    LWresult lwSt;
    dim3 blockDim; /* Defaults to 1,1,1 */
    dim3 gridDim; /* Defaults to 1,1,1 */
    void *kernelParams[2];
    unsigned int sharedMemBytes = 0;

    if(numSms < 1 || (int)numSms > m_multiProcessorCount)
    {
        fprintf(stderr, "numSms %d must be 1 <= X <= %d\n", 
                numSms, m_multiProcessorCount);
        return -1;
    }
    
    gridDim.x = numSms;

    /* blockDim.x has a limit of 1024. m_maxThreadsPerMultiProcessor is 2048 on current hardware. 
       So if we're > 1024, just divide by 2 and double the number of blocks we launch */
    if(threadsPerSm > 1024)
    {
        blockDim.x = threadsPerSm / 2;
        gridDim.x *= 2;
    }
    else
        blockDim.x = threadsPerSm;


    uint64_t *d_a = NULL;
    kernelParams[0] = &d_a;

    uint32_t waitInNs = runForUsec * 1000;
    kernelParams[1] = &waitInNs;
    
    lwSt = lwLaunchKernel(m_lwFuncWaitNs, gridDim.x, gridDim.y, gridDim.z, blockDim.x, 
                          blockDim.y, blockDim.z, sharedMemBytes, 
                          NULL, kernelParams, NULL);
    if(lwSt)
    {
        fprintf(stderr, "lwLaunchKernel returned %d\n", lwSt);
        return -1;
    }

    return 0;
}

/*****************************************************************************/
int DcgmProfTester::RunSubtestSmOclwpancyTargetMax(void)
{
    dcgmFieldValue_v1 value;
    double duration = m_duration;

    BeginSubtest("SM Oclwpancy - Max", "sm_oclwpancy_max", false);

    printf("Testing Maximum SmOclwpancy/SmActivity/GrActivity for %.1f seconds.\n", duration);
    printf("-------------------------------------------------------------\n");

    /* Generate SM activity */
    double startTime = timelib_dsecSince1970();
    double testEnd = startTime + duration; 
    double now = timelib_dsecSince1970();

    while(now < testEnd)
    {
        now = timelib_dsecSince1970();

        unsigned int numSms = m_multiProcessorCount;
        unsigned int threadsPerSm = m_maxThreadsPerMultiProcessor;

        int st = RunSleepKernel(numSms, threadsPerSm, 1000000);
        if(st)
            return st;

        /* Wait for this kernel to finish */
        lwCtxSynchronize();

        GetLatestDcgmValue(&value);

        /* Skip the first record since DCGM lags behind by one reporting period */
        if(now - startTime > 1.0)
        {
            printf("SmOclwpancy generated 1.0, dcgm %.3f at %.1f seconds. threadsPerSm %u\n", 
                value.value.dbl, now - startTime, threadsPerSm);

            AppendSubtestRecord(1.0, value.value.dbl);
        }
    }

    EndSubtest();

    return 0;
}

/*****************************************************************************/
int DcgmProfTester::RunSubtestSmOclwpancy(void)
{
    /* Generate SM oclwpancy */
    
    double duration = m_duration / 3.0;
    double prevOclwpancy; /* Actual oclwpancy is one loop iteration ahead of reported.
                             So we report prevOclwpancy to match dcgm data up with generated values */
    dcgmFieldValue_v1 value;

    printf("Testing SmOclwpancy scaling by num threads for %.1f seconds.\n", duration);
    printf("-------------------------------------------------------------\n");

    BeginSubtest("SM Oclwpancy - Num Threads", "sm_oclwpancy_num_threads", true);

    double now = timelib_dsecSince1970();
    double startTime = timelib_dsecSince1970();
    double testEnd = startTime + duration; 
    prevOclwpancy = 0.0;
    while(now < testEnd)
    {
        now = timelib_dsecSince1970();

        double howFarIn = (now - startTime) / duration;
        unsigned int numSms = m_multiProcessorCount;
        unsigned int threadsPerSm = (unsigned int)(howFarIn * m_maxThreadsPerMultiProcessor);
        if(threadsPerSm < 1)
            threadsPerSm = 1;
        if((int)threadsPerSm > m_maxThreadsPerMultiProcessor)
            threadsPerSm = m_maxThreadsPerMultiProcessor;
        
        double expectedSmOclwpancy = (double)threadsPerSm / (double)m_maxThreadsPerMultiProcessor;

        int st = RunSleepKernel(numSms, threadsPerSm, 1000000);
        if(st)
            return st;

        /* Wait for this kernel to finish */
        lwCtxSynchronize();

        long long timestamp = 0;
        
        GetLatestDcgmValue(&value);

        printf("SmOclwpancy generated %.3f, dcgm %.3f at %.1f seconds. threadsPerSm %u / %u.\n", 
               prevOclwpancy, value.value.dbl, now - startTime, 
               threadsPerSm, m_maxThreadsPerMultiProcessor);
        AppendSubtestRecord(prevOclwpancy, value.value.dbl);
        
        prevOclwpancy = expectedSmOclwpancy;
    }

    EndSubtest();

    printf("Sleeping to let previous test fall off.\n");
    usleep(2000000);

    printf("Testing SmOclwpancy scaling by SM count for %.1f seconds.\n", duration);
    printf("----------------------------------------------------------\n");

    BeginSubtest("SM Oclwpancy - SM Count", "sm_oclwpancy_sm_count", true);

    now = timelib_dsecSince1970();
    startTime = now;
    testEnd = startTime + duration; 
    prevOclwpancy = 0.0;

    while(now < testEnd)
    {
        now = timelib_dsecSince1970();

        double howFarIn = (now - startTime) / duration;
        unsigned int numSms = (unsigned int)(howFarIn * m_multiProcessorCount);
        if(numSms < 1)
            numSms = 1;
        if((int)numSms > m_multiProcessorCount)
            numSms = m_multiProcessorCount;
        
        double expectedSmOclwpancy = (double)numSms / (double)m_multiProcessorCount;

        int st = RunSleepKernel(numSms, m_maxThreadsPerMultiProcessor, 1000000);
        if(st)
            return st;

        /* Wait for this kernel to finish */
        lwCtxSynchronize();

        GetLatestDcgmValue(&value);

        printf("SmOclwpancy generated %.3f, dcgm %.3f at %.1f seconds. numSms %u / %u.\n", 
               prevOclwpancy, value.value.dbl, now - startTime, numSms, m_multiProcessorCount);
        
        AppendSubtestRecord(prevOclwpancy, value.value.dbl);
        prevOclwpancy = expectedSmOclwpancy;
    }

    EndSubtest();

    printf("Sleeping to let previous test fall off.\n");
    usleep(2000000);

    printf("\n");

    printf("Testing SmOclwpancy scaling by CPU sleeps for %.1f seconds.\n", duration);
    printf("------------------------------------------------------------\n");

    BeginSubtest("SM Oclwpancy - CPU Sleeps", "sm_oclwpancy_cpu_sleeps", true);

    now = timelib_dsecSince1970();
    startTime = now;
    testEnd = startTime + duration; 
    prevOclwpancy = 0.0;

    while(now < testEnd)
    {
        now = timelib_dsecSince1970();

        double howFarIn = (now - startTime) / duration;
        unsigned int runKernelUsec = (unsigned int)(howFarIn * 1000000.0);
        double expectedSmOclwpancy = howFarIn;

        int st = RunSleepKernel(m_multiProcessorCount, m_maxThreadsPerMultiProcessor, runKernelUsec);
        if(st)
            return st;
        
        /* Kernel launch was asynch and nearly instant. 
           Sleep for a second and then wait for the kernel to finish */
        usleep(1000000); 

        /* Wait for this kernel to finish */
        lwCtxSynchronize();

        GetLatestDcgmValue(&value);

        printf("SmOclwpancy generated %.3f, dcgm %.3f at %.1f seconds.\n", 
               prevOclwpancy, value.value.dbl, now - startTime);

        AppendSubtestRecord(prevOclwpancy, value.value.dbl);
        prevOclwpancy = expectedSmOclwpancy;
    }

    EndSubtest();

    return 0;
}

/*****************************************************************************/
int DcgmProfTester::RunSubtestSmActivity(void)
{
    double duration = m_duration / 2.0;
    dcgmFieldValue_v1 value;

    printf("Testing SmActivity scaling by SM count for %.1f seconds.\n", duration);
    printf("---------------------------------------------------------\n");
    BeginSubtest("SM Activity - SM Count", "sm_activity_sm_count", true);

    /* Generate SM activity */
    double startTime = timelib_dsecSince1970();
    double testEnd = startTime + duration; 
    double now = timelib_dsecSince1970();
    double prevSmActivity = 0.0;

    while(now < testEnd)
    {
        now = timelib_dsecSince1970();

        double howFarIn = (now - startTime) / duration;
        unsigned int numSms = (unsigned int)(howFarIn * m_multiProcessorCount);
        if(numSms < 1)
            numSms = 1;
        if((int)numSms > m_multiProcessorCount)
            numSms = m_multiProcessorCount;
        
        double expectedSmActivity = (double)numSms / (double)m_multiProcessorCount;

        int st = RunSleepKernel(numSms, 1, 1000000);
        if(st)
            return st;

        /* Wait for this kernel to finish */
        lwCtxSynchronize();

        GetLatestDcgmValue(&value);

        printf("SmActivity generated %.3f, dcgm %.3f at %.1f seconds. numSms %u / %u.\n", 
               prevSmActivity, value.value.dbl, now - startTime, numSms, m_multiProcessorCount);
        AppendSubtestRecord(prevSmActivity, value.value.dbl);

        prevSmActivity = expectedSmActivity;
    }

    EndSubtest();

    printf("Sleeping to let previous test fall off.\n");
    usleep(2000000);

    printf("\n");

    printf("Testing SmActivity scaling by CPU sleeps for %.1f seconds.\n", duration);
    printf("-----------------------------------------------------------\n");

    BeginSubtest("SM Activity - CPU Sleeps", "sm_activity_cpu_sleeps", true);

    now = timelib_dsecSince1970();
    startTime = now;
    testEnd = startTime + duration; 

    while(now < testEnd)
    {
        now = timelib_dsecSince1970();

        double howFarIn = (now - startTime) / duration;
        unsigned int runKernelUsec = (unsigned int)(howFarIn * 1000000.0);

        int st = RunSleepKernel(m_multiProcessorCount, 1, runKernelUsec);
        if(st)
            return st;
        
        /* Kernel launch was asynch and nearly instant. 
           Sleep for a second and then wait for the kernel to finish */
        usleep(1000000);

        /* Wait for this kernel to finish */
        lwCtxSynchronize();

        GetLatestDcgmValue(&value);

        printf("SmActivity: generated %.3f, dcgm %.3f at %.1f seconds.\n", 
               prevSmActivity, value.value.dbl, now - startTime);
        AppendSubtestRecord(prevSmActivity, value.value.dbl);

        prevSmActivity = howFarIn;
    }

    EndSubtest();

    return 0;
}

/*****************************************************************************/
int DcgmProfTester::RunSubtestGrActivity(void)
{
    dcgmFieldValue_v1 value;

    BeginSubtest("Graphics Activity", "gr_activity", true);

    /* Generate graphics and SM activity */
    double startTime = timelib_dsecSince1970();
    double duration = m_duration;
    double testEnd = startTime + duration; 
    double now = timelib_dsecSince1970();
    double prevGrAct = 0.0;

    while(now < testEnd)
    {
        now = timelib_dsecSince1970();

        double howFarIn = (now - startTime) / duration;
        unsigned int runKernelUsec = (unsigned int)(howFarIn * 1000000.0);

        int st = RunSleepKernel(1, 1, runKernelUsec);
        if(st)
            return st;
        
        /* Kernel launch was asynch and nearly instant. 
           Sleep for a second and then wait for the kernel to finish */
        usleep(1000000); 

        /* Wait for this kernel to finish */
        lwCtxSynchronize();

        GetLatestDcgmValue(&value);

        printf("GrActivity: generated %.3f, dcgm %.3f at %.1f seconds.\n", 
               prevGrAct, value.value.dbl, now - startTime);
        
        AppendSubtestRecord(prevGrAct, value.value.dbl);

        prevGrAct = howFarIn;
    }

    EndSubtest();

    return 0;
}

/*****************************************************************************/
int DcgmProfTester::RunSubtestPcieBandwidth(void)
{
    LWresult lwSt;
    int retSt = 0;
    dcgmFieldValue_v1 value;

    /* Allocate 100 MB of FB and pinned memory */
    const size_t bufferSize = 100 * 1024 * 1024;
    void *hostMem = NULL;
    LWdeviceptr deviceMem = (LWdeviceptr)NULL;

    printf("Allocating host mem\n");
    lwSt = lwMemAllocHost(&hostMem, bufferSize);
    if(lwSt)
    {
        fprintf(stderr, "lwMemAllocHost returned %d\n", lwSt);
        return -1;
    }
    printf("Clearing host mem\n");
    memset(hostMem, 0, bufferSize);

    printf("Allocating device mem\n");
    lwSt = lwMemAlloc(&deviceMem, bufferSize);
    if(lwSt)
    {
        lwMemFreeHost(hostMem);
        fprintf(stderr, "lwMemAlloc returned %d\n", lwSt);
        return -1;
    }
    printf("Clearing device mem\n");
    lwMemsetD32(deviceMem, 0, bufferSize);

    const char *fieldHeading;
    const char *subtestTag;
    if(m_testFieldId == DCGM_FI_PROF_PCIE_RX_BYTES)
    {
        fieldHeading = "PcieRxBytes";
        subtestTag = "pcie_rx_bytes";
    }
    else
    {
        fieldHeading = "PcieTxBytes";
        subtestTag = "pcie_tx_bytes";
    }
    
    BeginSubtest(fieldHeading, subtestTag, false);

    /* Set timers after we've allocated memory since that takes a while */
    double startTime = timelib_dsecSince1970();
    double duration = m_duration;
    double testEnd = startTime + duration; 
    double now = timelib_dsecSince1970();
    unsigned int i;
    double prevPerSecond = 0.0;
    double oneMiB = 1000000;

    while(now < testEnd)
    {
        now = timelib_dsecSince1970();

        size_t bytesTransferred = 0;
        double startLoop = now;
        unsigned int copiesPerIteration = 10000; /* Set high. We are going to drop out every second anyway */

        for(i = 0; i < copiesPerIteration && now - startLoop < 1.0; i++)
        {
            if(m_testFieldId == DCGM_FI_PROF_PCIE_RX_BYTES)
            {
                lwSt = lwMemcpyHtoD(deviceMem, hostMem, bufferSize);
            }
            else /* DCGM_FI_PROF_PCIE_TX_BYTES */
            {
                lwSt = lwMemcpyDtoH(hostMem, deviceMem, bufferSize);
            }

            bytesTransferred += bufferSize;

            if(lwSt)
            {
                fprintf(stderr, "lwMemcpy returned %d\n", lwSt);
                retSt = -1;
                goto CLEANUP;
            }

            now = timelib_dsecSince1970();
        }

        GetLatestDcgmValue(&value);

        /* Colwert int64->double */
        double dcgmValue = (double)value.value.i64;
        dcgmValue /= oneMiB;

        double afterLoopDsec = timelib_dsecSince1970();
        double perSecond = (double)bytesTransferred / (afterLoopDsec - startLoop) / oneMiB;
        perSecond *= 1.123; /* We've seen a 12.3% overhead in testing, verified secondarily by looking at
                              lwpu-smi dmon -s */

        printf("%s: generated %.1f, dcgm %.1f MiB/sec\n", fieldHeading, prevPerSecond, dcgmValue);
        AppendSubtestRecord(prevPerSecond, dcgmValue);

        prevPerSecond = perSecond;
    }

CLEANUP:
    lwMemFreeHost(hostMem);
    lwMemFree(deviceMem);

    return retSt;
}

/*****************************************************************************/
dcgmReturn_t DcgmProfTester::HelperGetBestLwLinkPeer(unsigned int *peerGpuId, LWdevice *peerLwdaOrdinal)
{
    int i;
    LWresult lwSt;

    dcgmDeviceTopology_v1 deviceTopo;
    memset(&deviceTopo, 0, sizeof(deviceTopo));
    deviceTopo.version = dcgmDeviceTopology_version1;
    dcgmReturn_t dcgmReturn = dcgmGetDeviceTopology(m_dcgmHandle, m_gpuId, &deviceTopo);
    if(dcgmReturn != DCGM_ST_OK)
    {
        fprintf(stderr, "dcgmGetDeviceTopology failed with %d for gpuId %u\n", dcgmReturn, m_gpuId);
        return dcgmReturn;
    }

    int lwLinkMasks = DCGM_TOPOLOGY_LWLINK1 | DCGM_TOPOLOGY_LWLINK2 | DCGM_TOPOLOGY_LWLINK3 | 
                      DCGM_TOPOLOGY_LWLINK4 | DCGM_TOPOLOGY_LWLINK5 | DCGM_TOPOLOGY_LWLINK6;

    /* Find the GPU we have the most LwLink connections to */
    unsigned int bestGpuId = 0;
    unsigned int maxLinkMask = 0;
    for(i = 0; i < (int)deviceTopo.numGpus; i++)
    {
        if(!deviceTopo.gpuPaths[i].localLwLinkIds & lwLinkMasks)
            continue;

        /* More links = higher mask value */
        if(deviceTopo.gpuPaths[i].localLwLinkIds < maxLinkMask)
            continue;
        
        bestGpuId = deviceTopo.gpuPaths[i].gpuId;
        maxLinkMask = deviceTopo.gpuPaths[i].localLwLinkIds;
    }

    if(!maxLinkMask)
    {
        printf("gpuId %u has no LwLink peers. Skipping test.\n", m_gpuId);
        return DCGM_ST_NOT_SUPPORTED;
    }

    *peerGpuId = bestGpuId;

    dcgmDeviceAttributes_v1 peerDeviceAttr;
    memset(&peerDeviceAttr, 0, sizeof(peerDeviceAttr));
    peerDeviceAttr.version = dcgmDeviceAttributes_version1;
    dcgmReturn = dcgmGetDeviceAttributes(m_dcgmHandle, bestGpuId, &peerDeviceAttr);
    if(dcgmReturn != DCGM_ST_OK)
    {
        fprintf(stderr, "dcgmGetDeviceAttributes failed with %d for gpuId %u\n", dcgmReturn, bestGpuId);
        return dcgmReturn;
    }

    /* Find the corresponding lwca device to our peer DCGM device */
    lwSt = lwDeviceGetByPCIBusId(peerLwdaOrdinal, peerDeviceAttr.identifiers.pciBusId);
    if(lwSt)
    {
        fprintf(stderr, "lwDeviceGetByPCIBusId returned %d for busId %s\n", lwSt, peerDeviceAttr.identifiers.pciBusId);
        return DCGM_ST_GENERIC_ERROR;
    }

    printf("The best peer of gpuId %u is gpuId %u, LWdevice %u, linkMask x%X\n", 
           m_gpuId, bestGpuId, *peerLwdaOrdinal, maxLinkMask);

    return DCGM_ST_OK;
}

/*****************************************************************************/
int DcgmProfTester::RunSubtestLwLinkBandwidth(void)
{
    unsigned int peerGpuId = 0;
    LWdevice peerLwDevice = 0;
    LWresult lwSt;
    int retSt = 0;
    dcgmFieldValue_v1 value;
    double prevPerSecond, oneMiB, duration, startTime, testEnd, now;

    /* Get our best peer to do LwLink copies to */
    dcgmReturn_t dcgmReturn = HelperGetBestLwLinkPeer(&peerGpuId, &peerLwDevice);
    if(dcgmReturn != DCGM_ST_OK)
        return dcgmReturn;

    /* Create a context on the other GPU */
    LWcontext deviceCtx1;
    lwSt = lwCtxCreate(&deviceCtx1, 0, peerLwDevice);
    if(lwSt)
    {
        fprintf(stderr, "lwCtxCreate returned %d", lwSt);
        return DCGM_ST_GENERIC_ERROR;
    }

    /* Allocate 100 MB of FB memory on both devices */
    const size_t bufferSize = 100 * 1024 * 1024;
    LWdeviceptr deviceMem0 = (LWdeviceptr)NULL;
    LWdeviceptr deviceMem1 = (LWdeviceptr)NULL;

    lwCtxSetLwrrent(m_lwdaContext);

    printf("Allocating device 0 mem\n");
    lwSt = lwMemAlloc(&deviceMem0, bufferSize);
    if(lwSt)
    {
        fprintf(stderr, "lwMemAlloc returned %d\n", lwSt);
        retSt = -1;
        goto CLEANUP;
    }
    printf("Clearing device 0 mem\n");
    lwMemsetD32(deviceMem0, 0, bufferSize);

    lwCtxSetLwrrent(deviceCtx1);

    printf("Allocating device 1 mem\n");
    lwSt = lwMemAlloc(&deviceMem1, bufferSize);
    if(lwSt)
    {
        fprintf(stderr, "lwMemAlloc returned %d\n", lwSt);
        retSt = -1;
        goto CLEANUP;
    }
    printf("Clearing device 1 mem\n");
    lwMemsetD32(deviceMem1, 0, bufferSize);

    lwCtxSetLwrrent(m_lwdaContext);

    lwSt = lwCtxEnablePeerAccess(deviceCtx1, 0);
    if(lwSt)
    {
        fprintf(stderr, "lwCtxEnablePeerAccess returned %d\n", lwSt);
        retSt = -1;
        goto CLEANUP;
    }

    const char *fieldHeading;
    const char *subtestTag;
    if(m_testFieldId == DCGM_FI_PROF_LWLINK_RX_BYTES)
    {
        fieldHeading = "LwLinkRxBytes";
        subtestTag = "lwlink_rx_bytes";
    }
    else
    {
        fieldHeading = "LwLinkTxBytes";
        subtestTag = "lwlink_tx_bytes";
    }
    
    BeginSubtest(fieldHeading, subtestTag, false);

    /* Set timers after we've allocated memory since that takes a while */
    startTime = timelib_dsecSince1970();
    duration = m_duration;
    testEnd = startTime + duration; 
    now = timelib_dsecSince1970();
    unsigned int i;
    prevPerSecond = 0.0;
    oneMiB = 1000000;

    while(now < testEnd)
    {
        now = timelib_dsecSince1970();

        size_t bytesTransferred = 0;
        double startLoop = now;
        unsigned int copiesPerIteration = 10000; /* Set high. We are going to drop out every second anyway */

        for(i = 0; i < copiesPerIteration && now - startLoop < 1.0; i++)
        {
            if(m_testFieldId == DCGM_FI_PROF_LWLINK_RX_BYTES)
            {
                lwSt = lwMemcpyDtoD(deviceMem0, deviceMem1, bufferSize);
            }
            else /* DCGM_FI_PROF_LWLINK_TX_BYTES */
            {
                lwSt = lwMemcpyDtoD(deviceMem1, deviceMem0, bufferSize);
            }

            bytesTransferred += bufferSize;

            if(lwSt)
            {
                fprintf(stderr, "lwMemcpy returned %d\n", lwSt);
                retSt = -1;
                goto CLEANUP;
            }

            now = timelib_dsecSince1970();
        }

        GetLatestDcgmValue(&value);

        /* Colwert int64->double */
        double dcgmValue = (double)value.value.i64;
        dcgmValue /= oneMiB;

        double afterLoopDsec = timelib_dsecSince1970();
        double perSecond = (double)bytesTransferred / (afterLoopDsec - startLoop) / oneMiB;

        printf("%s: generated %.1f, dcgm %.1f MiB/sec\n", fieldHeading, prevPerSecond, dcgmValue);
        AppendSubtestRecord(prevPerSecond, dcgmValue);

        prevPerSecond = perSecond;
    }

CLEANUP:
    
    if(deviceMem0)
        lwMemFree(deviceMem0);
    if(deviceMem1)
        lwMemFree(deviceMem1);

    return retSt;
    
    return DCGM_ST_OK;
}

/*****************************************************************************/
int DcgmProfTester::RunSubtestDramUtil(void)
{
    LWresult lwSt;
    int retSt = 0;
    dcgmFieldValue_v1 value;

    /* Allocate 100 MB of FB x 2 */
    const size_t bufferSize = 100 * 1024 * 1024;
    LWdeviceptr deviceMem = (LWdeviceptr)NULL;
    LWdeviceptr deviceMem2 = (LWdeviceptr)NULL;

    printf("Allocating device mem x 2\n");
    lwSt = lwMemAlloc(&deviceMem, bufferSize);
    if(lwSt)
    {
        fprintf(stderr, "lwMemAlloc returned %d\n", lwSt);
        return -1;
    }
    lwSt = lwMemAlloc(&deviceMem2, bufferSize);
    if(lwSt)
    {
        lwMemFree(deviceMem);
        fprintf(stderr, "lwMemAlloc returned %d\n", lwSt);
        return -1;
    }


    printf("Clearing device mem x 2\n");
    lwMemsetD32(deviceMem, 0, bufferSize);
    lwMemsetD32(deviceMem2, 0, bufferSize);

    BeginSubtest("DRAM Activity", "dram_activity", false);

    /* Set timers after we've allocated memory since that takes a while */
    double startTime = timelib_dsecSince1970();
    double duration = m_duration;
    double testEnd = startTime + duration; 
    double now = timelib_dsecSince1970();
    unsigned int i;
    double prevDramAct = 0.0;
    double prevPerSecond = 0.0;

    while(now < testEnd)
    {
        now = timelib_dsecSince1970();

        size_t bytesTransferred = 0;
        double startLoop = now;
        unsigned int copiesPerIteration = 10000; /* Set high. We are going to drop out every second anyway */

        for(i = 0; i < copiesPerIteration && now - startLoop < 1.0; i++)
        {
            lwSt = lwMemcpy(deviceMem, deviceMem2, bufferSize);
            if(lwSt)
            {
                fprintf(stderr, "lwMemcpy returned %d\n", lwSt);
                retSt = -1;
                goto CLEANUP;
            }

            bytesTransferred += bufferSize * 2; /* We have to include both source and dest bw */

            now = timelib_dsecSince1970();
        }

        double afterLoopDsec = timelib_dsecSince1970();
        double perSecond = (double)bytesTransferred / (afterLoopDsec - startLoop);
        double utilRate = perSecond / m_maximumMemBandwidth;
        perSecond = perSecond / (1000.0 * 1000.0 * 1000.0);

        GetLatestDcgmValue(&value);

        printf("DramUtil generated %.3f, dcgm %.3f (%.1f GiB/sec)\n", prevDramAct, value.value.dbl, prevPerSecond);
        AppendSubtestRecord(prevDramAct, value.value.dbl);

        prevDramAct = utilRate;
        prevPerSecond = perSecond;
    }

CLEANUP:
    lwMemFree(deviceMem);
    lwMemFree(deviceMem2);

    EndSubtest();

    return retSt;
}

/*****************************************************************************/
int DcgmProfTester::RunSubtestGemmUtil(void)
{
    int retSt = 0;
    const size_t arrayDim = 4096;
    LWdeviceptr deviceA = (LWdeviceptr)NULL;
    LWdeviceptr deviceB = (LWdeviceptr)NULL;
    LWdeviceptr deviceC = (LWdeviceptr)NULL;
    void *hostA = NULL;
    void *hostB = NULL;
    LWresult lwSt, lwSt2, lwSt3;
    lwblasStatus_t lwbSt;
    lwblasHandle_t lwblasHandle = NULL;
    const char *testHeader = NULL; 
    const char *testTag = NULL;
    size_t valueSize = 0;
    unsigned int i;
    double startTime, duration, testEnd, now;
    double alpha = 1.01 + ((double)(rand() % 100)/10.0);
    double beta = 1.01 + ((double)(rand() % 100)/10.0);
    float floatAlpha = (float)alpha;
    float floatBeta = (float)beta;
    double flopsPerOp = 2.0 * (double)arrayDim * (double)arrayDim * (double)arrayDim;
    dcgmFieldValue_v1 value;
    
    /* Used https://en.wikipedia.org/wiki/Half-precision_floating-point_format to make these constants, as
       the lwca functions are device-side only */
    __half_raw oneAsHalf;
    oneAsHalf.x = 0x3C00; /* 1.0 */
    __half fp16Alpha = oneAsHalf;
    __half fp16Beta = oneAsHalf;

    switch(m_testFieldId)
    {
        case DCGM_FI_PROF_PIPE_FP32_ACTIVE:
            valueSize = sizeof(float);
            testHeader = "Fp32EngineActive";
            testTag = "fp32_active";
            break;
        case DCGM_FI_PROF_PIPE_FP64_ACTIVE:
            valueSize = sizeof(double);
            testHeader = "Fp64EngineActive";
            testTag = "fp64_active";
            break;
        case DCGM_FI_PROF_PIPE_FP16_ACTIVE:
            valueSize = sizeof(unsigned short);
            testHeader = "Fp16EngineActive";
            testTag = "fp16_active";
            break;
        case DCGM_FI_PROF_PIPE_TENSOR_ACTIVE:
            valueSize = sizeof(unsigned short);
            testHeader = "TensorEngineActive";
            testTag = "tensor_active";
            break;
        default:
            fprintf(stderr, "fieldId %u is unhandled.\n", m_testFieldId);
            return -1;
    }

    BeginSubtest(testHeader, testTag, false);

    lwbSt = lwblasCreate(&lwblasHandle);
    if(lwbSt != LWBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "lwblasCreate returned %d", lwbSt);
        return -1;
    }

    size_t arrayCount = arrayDim * arrayDim;
    size_t arrayByteSize = valueSize * arrayCount;
    
    /* Once we've allocated memory, goto CLEANUP instead of returning  */

    lwSt = lwMemAlloc(&deviceA, arrayByteSize);
    lwSt2 = lwMemAlloc(&deviceB, arrayByteSize);
    lwSt3 = lwMemAlloc(&deviceC, arrayByteSize);
    if(lwSt || lwSt2 || lwSt3)
    {
        fprintf(stderr, "lwMemAlloc returned %d %d %d for %d bytes\n", lwSt, lwSt2, lwSt3, (int)arrayByteSize);
        retSt = -1;
        goto CLEANUP;
    }

    hostA = malloc(arrayByteSize);
    hostB = malloc(arrayByteSize);
    if(!hostA || !hostB)
    {
        fprintf(stderr, "Unable to allocate %d bytes x2\n", (int)arrayByteSize);
        retSt = -1;
        goto CLEANUP;
    }

    switch(m_testFieldId)
    {
        case DCGM_FI_PROF_PIPE_FP32_ACTIVE:
        {
            float *floatHostA = (float *)hostA;
            float *floatHostB = (float *)hostB;
            for(i=0; i<arrayCount; i++)
            {
                floatHostA[i] = (float)rand() / 100.0;
                floatHostB[i] = (float)rand() / 100.0;
            }
            break;
        }
        
        case DCGM_FI_PROF_PIPE_FP64_ACTIVE:
        {
            double *doubleHostA = (double *)hostA;
            double *doubleHostB = (double *)hostB;

            for(i=0; i<arrayCount; i++)
            {
                doubleHostA[i] = (double)rand() / 100.0;
                doubleHostB[i] = (double)rand() / 100.0;
            }
            break;
        }

        case DCGM_FI_PROF_PIPE_FP16_ACTIVE:
        case DCGM_FI_PROF_PIPE_TENSOR_ACTIVE:
        {
            __half *halfHostA = (__half *)hostA;
            __half *halfHostB = (__half *)hostB;
            __half_raw rawA, rawB;

            for(i=0; i<arrayCount; i++)
            {
                rawA.x = rand() % 65536;
                rawB.x = rand() % 65536;

                halfHostA[i] = rawA;
                halfHostB[i] = rawB;
            }
            break;
        }
        
        default:
            fprintf(stderr, "fieldId %u is unhandled.\n", m_testFieldId);
            retSt = -1;
            goto CLEANUP;
    }

    /* Just zero the output array */
    lwMemsetD32(deviceC, 0, arrayByteSize);

    /* Copy A and B to the device */
    lwSt = lwMemcpyHtoD(deviceA, hostA, arrayByteSize);
    lwSt2 = lwMemcpyHtoD(deviceB, hostB, arrayByteSize);
    if(lwSt || lwSt2)
    {
        fprintf(stderr, "lwMemcpyHtoD failed %d %d\n", lwSt, lwSt2);
        retSt = -1;
        goto CLEANUP;
    }

    /* Should we enable tensor cores? */
    if(m_testFieldId == DCGM_FI_PROF_PIPE_TENSOR_ACTIVE)
        lwbSt = lwblasSetMathMode(lwblasHandle, LWBLAS_TENSOR_OP_MATH);
    else
        lwbSt = lwblasSetMathMode(lwblasHandle, LWBLAS_DEFAULT_MATH);
    if(lwbSt != LWBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "lwbalsSetMathMode returned %d\n", lwbSt);
        retSt = -1;
        goto CLEANUP;
    }

    /* deviceA and deviceB now have our matricies. Run our test */

    startTime = timelib_dsecSince1970();
    duration = m_duration;
    testEnd = startTime + duration; 
    now = timelib_dsecSince1970();

    while(now < testEnd)
    {
        now = timelib_dsecSince1970();

        double startLoop = now;
        unsigned int opsPerIteration = 1000; /* Set high. We are going to drop out every second anyway */

        for(i = 0; i < opsPerIteration && now - startLoop < 1.0; i++)
        {
            switch(m_testFieldId)
            {
                case DCGM_FI_PROF_PIPE_FP32_ACTIVE:
                    lwbSt = lwblasSgemm(lwblasHandle, LWBLAS_OP_N, LWBLAS_OP_N, arrayDim, arrayDim, arrayDim,
                                        &floatAlpha, (float *)deviceA, arrayDim, (float *)deviceB, arrayDim,
                                        &floatBeta, (float *)deviceC, arrayDim);
                    break;
                
                case DCGM_FI_PROF_PIPE_FP64_ACTIVE:
                    lwbSt = lwblasDgemm(lwblasHandle, LWBLAS_OP_N, LWBLAS_OP_N, arrayDim, arrayDim, arrayDim,
                                        &alpha, (double *)deviceA, arrayDim, (double *)deviceB, arrayDim, &beta,
                                        (double *)deviceC, arrayDim);
                    break;
                
                case DCGM_FI_PROF_PIPE_FP16_ACTIVE:
                case DCGM_FI_PROF_PIPE_TENSOR_ACTIVE:
                    lwbSt = lwblasHgemm(lwblasHandle, LWBLAS_OP_N, LWBLAS_OP_N, arrayDim, arrayDim, arrayDim,
                                        &fp16Alpha, (__half *)deviceA, arrayDim, (__half *)deviceB, arrayDim, &fp16Beta,
                                        (__half *)deviceC, arrayDim);
                    break;

                default:
                    fprintf(stderr, "Shouldn't get here.\n");
                    retSt = -1;
                    goto CLEANUP;
            }

            if(lwbSt != LWBLAS_STATUS_SUCCESS)
            {
                fprintf(stderr, "lwblas gemm returned %d\n", lwbSt);
                retSt = -1;
                goto CLEANUP;
            }

            /* Wait for this kernel to finish */
            lwCtxSynchronize();

            now = timelib_dsecSince1970();
        }

        GetLatestDcgmValue(&value);

        double flopsInLoop = flopsPerOp * (double)i;
        double afterLoopDsec = timelib_dsecSince1970();
        double utilRate = 1.0; //perSecond / m_maximumMemBandwidth;
        double gflops = (flopsInLoop / 1000000000.0) / (afterLoopDsec - startLoop);
        printf("%s: generated ???, dcgm %.3f (%.1f gflops)\n", testHeader, value.value.dbl, gflops);

        AppendSubtestRecord(0.0, value.value.dbl);
    }


CLEANUP:
    EndSubtest();

    if(deviceA)
        lwMemFree(deviceA);
    if(deviceB)
        lwMemFree(deviceB);
    if(deviceC)
        lwMemFree(deviceC);
    if(hostA)
        free(hostA);
    if(hostB)
        free(hostB);
    
    lwblasDestroy(lwblasHandle);

    return retSt;
}

/*****************************************************************************/
int DcgmProfTester::RunTests(void)
{
    int retSt = 0;

    if(WatchFields(1000000))
        return -1;

    switch(m_testFieldId)
    {
        case DCGM_FI_PROF_GR_ENGINE_ACTIVE:
            if(m_targetMaxValue)
                retSt = RunSubtestSmOclwpancyTargetMax();
            else
                retSt = RunSubtestGrActivity();
            break;
        
        case DCGM_FI_PROF_SM_ACTIVE:
            if(m_targetMaxValue)
                retSt = RunSubtestSmOclwpancyTargetMax();
            else
                retSt = RunSubtestSmActivity();
            break;
        
        case DCGM_FI_PROF_SM_OCLWPANCY:
            if(m_targetMaxValue)
                retSt = RunSubtestSmOclwpancyTargetMax();
            else
                retSt = RunSubtestSmOclwpancy();
            break;
        
        case DCGM_FI_PROF_PCIE_RX_BYTES:
        case DCGM_FI_PROF_PCIE_TX_BYTES:
            retSt = RunSubtestPcieBandwidth();
            break;

        case DCGM_FI_PROF_DRAM_ACTIVE:
            retSt = RunSubtestDramUtil();
            break;
        
        case DCGM_FI_PROF_PIPE_FP32_ACTIVE:
        case DCGM_FI_PROF_PIPE_FP64_ACTIVE:
        case DCGM_FI_PROF_PIPE_FP16_ACTIVE:
        case DCGM_FI_PROF_PIPE_TENSOR_ACTIVE:
            retSt = RunSubtestGemmUtil();
            break;

        case DCGM_FI_PROF_LWLINK_RX_BYTES:
        case DCGM_FI_PROF_LWLINK_TX_BYTES:
            retSt = RunSubtestLwLinkBandwidth();
            break;

        default:
            fprintf(stderr, "A test for fieldId %u has not been implemented yet.", m_testFieldId);
            retSt = -1;
            break;
    }

    /* If we forgot to end our subtest, end it */
    EndSubtest();
    
    UnwatchFields();

    return retSt;
}

/*****************************************************************************/
int main(int argc, char **argv)
{
    int st;
    DcgmProfTester dpt;
    dcgmReturn_t dcgmReturn;

    //lwosSetElw(DCGM_ELW_DBG_LVL, "WARNING");
    //lwosSetElw(DCGM_ELW_DBG_FILE, "dcgmproftester.log");

    loggingInit((char *)DCGM_ELW_DBG_LVL, (char *)DCGM_ELW_DBG_APPEND,
                (char *)DCGM_ELW_DBG_FILE);

    dcgmReturn = dpt.Init(argc, argv);
    if(dcgmReturn)
    {
        fprintf(stderr, "Error %d from Init(). Exiting.\n", dcgmReturn);
        return -((int)dcgmReturn);
    }

    st = dpt.RunTests();
    if(st)
    {
        fprintf(stderr, "Error %d from RunTests(). Exiting.\n", st);
        return -((int)DCGM_ST_GENERIC_ERROR);
    }

    return 0;
}

/*****************************************************************************/
/* Stubbing in a pthread function that isn't in our ancient GCC 4.1.1 */
extern "C"
int pthread_mutexattr_setprotocol(pthread_mutexattr_t * /*attr*/, int /*protocol*/)
{
    return 0;
}

/*****************************************************************************/

